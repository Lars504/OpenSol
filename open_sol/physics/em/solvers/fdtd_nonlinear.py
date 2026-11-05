"""C5 nonlinear media of the 2D FDTD engines (time domain only): instantaneous chi(2) (optionally poled), Kerr,
two-photon absorption (TPA) and the free-carrier density N it generates (free-carrier absorption FCA and dispersion
FCD), as one fused implicit E correction per E component box after the linear update (kernel notes at
_correction_kernel)."""
import math

import numpy as np
from ....core.backend import cp
from ....core.kernels import twin

from ....core.constants import C0, DTYPE, EPS0, HBAR
from ..fields import collocate_to_centres
from ..yee_grid import face_average
from .fdtd_2d_base import _bounding_box

# Per-cell kernel coefficients (nonlinear_cell_coefficients); the carrier ones exist only when a region has carriers.
_BASE_COEFFICIENTS = ('kerr', 'tpa', 'chi2')
_CARRIER_COEFFICIENTS = ('fca', 'fcd', 'dt_tau', 'pair')
_NEWTON_ITERATIONS = 3   # of the scalar cubic; the lagged chi2 / Kerr / TPA start is O(chi E dE) off the root
# hz faces of weight w (the face_average of the region cells) hold w x each coefficient; face dt_tau / w and pair / w^2
# restore the medium's rates (decay dt / tau, generation (pair / w) (w tpa) E^4), so N is the bulk density while fca /
# fcd / tpa / kerr / chi2 stay averaged (w x the bulk response, as fdtd_gain's weight)
_PER_MATERIAL_POWERS = {'dt_tau': 1, 'pair': 2}


def nonlinear_cell_coefficients(dt, n2=0.0, tpa_beta=0.0, carrier_lifetime=0.0, fca_cross_section=0.0,
                                fcd_coefficient=0.0, pump_wavelength=1.55e-6, d_eff=0.0):
    """Kernel coefficients of one medium from its SI parameters (n0^2 = the local permittivity array, read live; d_eff
    [m/V] is not scaled by n0)."""
    values = {'n2': n2, 'tpa_beta': tpa_beta, 'carrier_lifetime': carrier_lifetime,
              'fca_cross_section': fca_cross_section, 'fcd_coefficient': fcd_coefficient,
              'pump_wavelength': pump_wavelength, 'd_eff': d_eff}
    for name, value in values.items():
        if not math.isfinite(float(value)):
            raise ValueError(f'nonlinear medium {name} must be finite, got {value}')
    if tpa_beta < 0 or carrier_lifetime < 0 or fca_cross_section < 0 or fcd_coefficient < 0:
        raise ValueError(f'nonlinear medium tpa_beta, carrier_lifetime, fca_cross_section and fcd_coefficient must be '
                         f'>= 0, got {tpa_beta}, {carrier_lifetime}, {fca_cross_section}, {fcd_coefficient}')
    if not pump_wavelength > 0:
        raise ValueError(f'nonlinear medium pump_wavelength must be > 0, got {pump_wavelength}')
    carriers = fca_cross_section > 0 or fcd_coefficient > 0
    if carriers and not carrier_lifetime > 0:
        raise ValueError('nonlinear medium: free-carrier absorption / dispersion need carrier_lifetime > 0')
    return {
        'kerr': 4.0 / 3.0 * EPS0 * C0 * n2,                     # chi3 = kerr * n0^2 [m^2/V^2]
        'tpa': 2.0 / 3.0 * tpa_beta * (C0 * EPS0) ** 2,          # TPA conductivity per E^2 = tpa * n0^2
        'chi2': 2.0 * d_eff,                                     # P = eps0 chi2 E^2 [m/V] (times the poling sign)
        'fca': fca_cross_section * EPS0 * C0,                    # FCA conductivity per carrier = fca * n0
        'fcd': -2.0 * fcd_coefficient,                           # delta eps per carrier = fcd * n0 (dn = -k N)
        'dt_tau': dt / carrier_lifetime if carriers else 0.0,
        # 1 / (2 hbar omega): pairs per absorbed J; 0 without carrier effects (no undecaying density next to a region
        # with carriers)
        'pair': pump_wavelength / (4.0 * math.pi * HBAR * C0) if carriers else 0.0,
        'carriers': carriers,
    }


# Implicit correction of one E component after the linear update (Yee, CPML / BC post, pole corrections), from
#   eps (x - E^n) / dt + sigma_e (x + E^n) / 2 + J_pole + (P(x) - P(E^n)) / dt + g (x + E^n) / 2 = curl H,
# P(E) = eps0 (chi2 E + chi3 E^2 + d eps) E (instantaneous chi2 + Kerr + carrier index), g = TPA + FCA conductivity:
# subtracting the linear update x_lin (cb = dt / (eps (1 + k)), the Yee / pole factor) gives, with s = cb eps0 / dt,
#   x (1 + s d_new + cb g_c / 2) + s chi2 x^2 + s chi3 x^3 + cb c_tpa m^3
#     = x_lin + s (chi2 E_n + chi3 E_n^2 + d_old) E_n - cb g_c E_n / 2,
# m = (x + E_n) / 2, g = c_tpa m^2 + g_c, g_c = fca n0 N^{n+1/2}: a monotonic cubic (chi3, c_tpa >= 0, or |chi3| E^2
# << eps; |chi2 E| << eps) solved by Newton from the lagged guess (g, chi2 and Kerr from E_n). Both P(x) and P(E^n) are
# pointwise in the integer-time E samples, so a cw E^n = Re(E1 e^{j w n dt}) drives P(2w) = eps0 (chi2 / 2) E1^2 (= eps0
# d E1^2, the SHG phasor of the NLFD tables) exactly on the lattice. g must be time-centred: c_tpa E_n^2 lags by
# dt / 2 and turns omega dt / 3 of the TPA current reactive (dn = -(omega dt / 3) beta I / (2 k0): -6 % of the Kerr
# phase at beta / (2 k0 n2) = 2, 71 steps per period). A pure cosine of amplitude A gives the cycle means 3/4 chi3 A^2
# (SPM: dn = n2 I) and 3/8 c_tpa A^4 = beta I^2, I = n0 c eps0 A^2 / 2.
# N^{n+1/2} = (N^{n-1/2} + dt pair c_tpa E_n^4) / (1 + dt / tau) (implicit Euler, float64: dt / tau ~ 1e-8 vanishes in
# float32); generation = the instantaneous TPA power / (2 hbar omega). PEC (cb = 0) and cells with zero coefficients
# keep x = x_lin exactly. With `loss`: += loss_scale g m^2 (ez q_accum [J/m^3]: dt; hz face pole power totals
# [W/m^3]: 1), the time-centred TPA + FCA work.
def _correction_kernel(carriers, loss):
    """ElementwiseKernel of the correction with / without the carrier ODE and / without the loss output."""
    in_params = ['float32 cb', 'float32 eps', 'float32 eps_scale', 'float32 eps0_inv_dt', 'float32 e_prev',
                 'float32 kerr', 'float32 tpa', 'float32 chi2']
    out_params = ['float32 e']
    body = ['float eps_rel = eps * eps_scale;',
            'float s = cb * eps0_inv_dt;',
            'float en = e_prev;',
            'float en2 = en * en;',
            'float chi3 = kerr * eps_rel;',
            'float ctpa = tpa * eps_rel;',
            'float g_carrier = 0.0f;',
            'float d_old = 0.0f;',
            'float d_new = 0.0f;']
    if carriers:
        in_params += ['float32 fca', 'float32 fcd', 'float32 dt_tau', 'float32 pair', 'float64 dt']
        out_params.append('float64 density')
        body += ['float n0 = sqrtf(eps_rel);',
                 'double n_old = density;',
                 'double generation = (double)pair * (double)(ctpa * en2) * (double)en2;',
                 'double n_new = (n_old + dt * generation) / (1.0 + (double)dt_tau);',
                 'd_old = (float)((double)(fcd * n0) * n_old);',
                 'd_new = (float)((double)(fcd * n0) * n_new);',
                 'g_carrier = (float)((double)(fca * n0) * n_new);',
                 'density = n_new;']
    if loss:
        in_params.append('float32 loss_scale')
        out_params.append('float32 loss')
    body += ['float half_cb = 0.5f * cb;',
             'float a = 1.0f + s * d_new + half_cb * g_carrier;',
             'float sq = s * chi2;',
             'float sk = s * chi3;',
             'float tk = cb * ctpa;',
             'float rhs = e + s * (chi2 * en + chi3 * en2 + d_old) * en - half_cb * g_carrier * en;',
             'float half_cb_g_lag = half_cb * ctpa * en2;',
             'float x = (rhs - half_cb_g_lag * en) / (a + half_cb_g_lag + sq * en + sk * en2);',
             f'for (int it = 0; it < {_NEWTON_ITERATIONS}; ++it) {{',
             '    float x2 = x * x;',
             '    float m = 0.5f * (x + en);',
             '    x -= (a * x + sq * x2 + sk * x2 * x + tk * m * m * m - rhs)',
             '         / (a + 2.0f * sq * x + 3.0f * sk * x2 + 1.5f * tk * m * m);',
             '}']
    if loss:
        body += ['float mean_e = 0.5f * (x + en);',
                 'float mean_e2 = mean_e * mean_e;',
                 'loss += loss_scale * (g_carrier + ctpa * mean_e2) * mean_e2;']
    body.append('e = x;')
    name = 'nonlinear_e_correction' + ('_carriers' if carriers else '') + ('_loss' if loss else '')
    return twin(f'em2d.{name}', cp.ElementwiseKernel(', '.join(in_params), ', '.join(out_params), '\n'.join(body),
                                                     name), _correction_host(carriers, loss))


def _correction_host(carriers, loss):
    """numpy twin of _correction_kernel(carriers, loss): the same arguments (inputs, then outputs written in place)
    and arithmetic (float32, the carrier ODE in float64); a change to the cuda body changes it."""
    def correct(*arguments):
        cb, eps, eps_scale, eps0_inv_dt, e_prev, kerr, tpa, chi2 = arguments[:8]
        position = 8
        if carriers:
            fca, fcd, dt_tau, pair, dt = arguments[8:13]
            position = 13
        if loss:
            loss_scale = arguments[position]
            position += 1
        e = arguments[position]
        density = arguments[position + 1] if carriers else None
        loss_out = arguments[-1] if loss else None
        eps_rel = eps * eps_scale
        s = cb * eps0_inv_dt
        en = e_prev
        en2 = en * en
        chi3 = kerr * eps_rel
        ctpa = tpa * eps_rel
        g_carrier = d_old = d_new = np.float32(0.0)
        if carriers:
            n0 = np.sqrt(eps_rel)
            n_old = density.copy()
            generation = pair.astype(np.float64) * (ctpa * en2).astype(np.float64) * en2.astype(np.float64)
            n_new = (n_old + dt * generation) / (1.0 + dt_tau.astype(np.float64))
            d_old = ((fcd * n0).astype(np.float64) * n_old).astype(np.float32)
            d_new = ((fcd * n0).astype(np.float64) * n_new).astype(np.float32)
            g_carrier = ((fca * n0).astype(np.float64) * n_new).astype(np.float32)
            density[...] = n_new
        half_cb = 0.5 * cb
        a = 1.0 + s * d_new + half_cb * g_carrier
        sq = s * chi2
        sk = s * chi3
        tk = cb * ctpa
        rhs = e + s * (chi2 * en + chi3 * en2 + d_old) * en - half_cb * g_carrier * en
        half_cb_g_lag = half_cb * ctpa * en2
        x = (rhs - half_cb_g_lag * en) / (a + half_cb_g_lag + sq * en + sk * en2)
        for _ in range(_NEWTON_ITERATIONS):
            x2 = x * x
            m = 0.5 * (x + en)
            x = x - (a * x + sq * x2 + sk * x2 * x + tk * m * m * m - rhs) / (a + 2.0 * sq * x + 3.0 * sk * x2
                                                                               + 1.5 * tk * m * m)
        if loss:
            mean_e = 0.5 * (x + en)
            mean_e2 = mean_e * mean_e
            loss_out += loss_scale * (g_carrier + ctpa * mean_e2) * mean_e2
        e[...] = x
    return correct


_CORRECTION_KERNELS = {(carriers, loss): _correction_kernel(carriers, loss)
                       for carriers in (False, True) for loss in (False, True)}


def add_nonlinear_region(engine, mask, poling=None, **params):
    """Add a chi2 / Kerr / TPA / free-carrier region (nonlinear_cell_coefficients params) on the run-backend bool (nx, ny) cell
    mask to engine.nonlinear (created at the first call; later regions win on overlaps); poling = full-grid real values
    (< 0 inverts chi2 in that cell) or None. NotImplementedError with Bloch k != 0 (E^2 of the real set is not |E|^2)."""
    if engine.complex_fields:
        raise NotImplementedError('nonlinear media (chi2, Kerr, TPA, free carriers) are not available with a Bloch '
                                  'phase k != 0: the response of the complex field mixes the +k and -k parts')
    engine._check_cell_mask(mask, 'Nonlinear medium mask')
    coefficients = nonlinear_cell_coefficients(float(engine.dt), **params)
    if engine.nonlinear is None:
        engine.nonlinear = NonlinearMedia(engine)
    engine.nonlinear.add(mask, coefficients, poling)


def _region_clusters(boxes, shape, axis, periodic):
    """Index lists of the cell boxes that overlap or touch (transitively; along a periodic face_average axis also
    across the wrap): regions sharing a cell or face share one component box, in their order (later ones win)."""
    def touch(first, second):
        for dimension in range(2):
            a, b = first[dimension], second[dimension]
            adjacent = a.start <= b.stop and b.start <= a.stop
            if not adjacent and periodic and dimension == axis:
                count = shape[dimension]
                adjacent = (a.start == 0 and b.stop == count) or (b.start == 0 and a.stop == count)
            if not adjacent:
                return False
        return True

    clusters = []
    for index, box in enumerate(boxes):
        joined = [cluster for cluster in clusters if any(touch(box, boxes[other]) for other in cluster)]
        merged = sorted([other for cluster in joined for other in cluster] + [index])
        clusters = [cluster for cluster in clusters if cluster not in joined] + [merged]
    return clusters


class NonlinearMedia:
    """Nonlinear regions of one FDTD engine: per E component (ez: the cells; hz: E_x / E_y faces with the face_average
    of the cell coefficients, carrier rates per medium: _PER_MATERIAL_POWERS) and cluster of touching regions the coefficient arrays, the E^n buffer and the float64
    carrier density N on the bounding box of its weights; cached box views of the field, cb and permittivity (all
    rewritten in place only)."""

    def __init__(self, engine):
        self._engine = engine
        self._regions = []        # (cell box, cropped cell mask, coefficients, cropped float32 chi2 sign or None)
        self.components = []
        self.has_carriers = False
        self._eps0_inv_dt = DTYPE(EPS0 / float(engine.dt))
        self._dt_f64 = np.float64(engine.dt)

    def add(self, mask, coefficients, poling=None):
        """Register a region (nothing for an empty mask; poling: full-grid values, < 0 -> chi2 sign -1) and rebuild the
        component arrays (build time: host syncs)."""
        box = _bounding_box(mask)
        if box is None:
            return
        xp = self._engine.xp
        sign = None if poling is None else xp.where(xp.asarray(poling)[box] < 0, -1.0, 1.0).astype(DTYPE)
        self._regions.append((box, mask[box].copy(), coefficients, sign))
        self._rebuild()

    def _rebuild(self):
        """Coefficient arrays per E component and cluster of touching regions (_region_clusters: distant regions get
        boxes of their own; full-grid temporaries; carrier densities reset)."""
        engine = self._engine
        xp = engine.xp
        shape = (engine.nx, engine.ny)
        self.has_carriers = any(coefficients['carriers'] for _, _, coefficients, _ in self._regions)
        names = _BASE_COEFFICIENTS + (_CARRIER_COEFFICIENTS if self.has_carriers else ())
        self.components = []
        for spec in engine._nonlinear_components():
            axis = spec['axis']
            for cluster in _region_clusters([region[0] for region in self._regions], shape, axis, spec['periodic']):
                union = xp.zeros(shape, dtype=DTYPE)
                cell_values = {name: xp.zeros(shape, dtype=DTYPE) for name in names}
                for box, cropped_mask, coefficients, chi2_sign in (self._regions[index] for index in cluster):
                    xp.copyto(union[box], DTYPE(1.0), where=cropped_mask)
                    for name in names:
                        value = DTYPE(coefficients[name])
                        if name == 'chi2' and chi2_sign is not None:
                            value = chi2_sign * value   # poled domains (face_average mixes signs at hz domain walls)
                        xp.copyto(cell_values[name][box], value, where=cropped_mask)
                weight = union if axis is None else face_average(union, axis, spec['periodic'], spec['widths'], xp=xp)
                box = _bounding_box(weight)
                if box is None:
                    continue
                arrays = {}
                face_weight = None if axis is None else xp.where(weight[box] > 0, weight[box], DTYPE(1.0))
                for name in names:
                    values = (cell_values[name] if axis is None
                              else face_average(cell_values[name], axis, spec['periodic'], spec['widths'], xp=xp))
                    arrays[name] = values[box].copy()   # copy: a view would keep the full-grid temporary alive
                    if face_weight is not None and name in _PER_MATERIAL_POWERS:
                        arrays[name] /= face_weight ** _PER_MATERIAL_POWERS[name]
                del union, cell_values, weight, face_weight
                box_shape = arrays['kerr'].shape
                component = {'name': spec['name'], 'axis': axis, 'periodic': spec['periodic'], 'box': box,
                             'arrays': arrays, 'e_prev': xp.zeros(box_shape, dtype=DTYPE),
                             'field': spec['field'][box], 'cb': spec['cb'][box], 'eps': spec['eps'][box],
                             'eps_scale': DTYPE(spec['eps_scale'])}
                if self.has_carriers:
                    component['density'] = xp.zeros(box_shape, dtype=np.float64)
                self.components.append(component)

    def store_previous(self):
        """Keep E^n of every component box (before the Yee update overwrites it)."""
        xp = self._engine.xp
        for component in self.components:
            xp.copyto(component['e_prev'], component['field'])

    def correct(self, accumulate):
        """Implicit nonlinear correction of every component after the linear corrections (+ the TPA / FCA work into the
        engine's loss target while accumulating)."""
        engine = self._engine
        for component in self.components:
            arrays = component['arrays']
            inputs = [component['cb'], component['eps'], component['eps_scale'], self._eps0_inv_dt,
                      component['e_prev'], arrays['kerr'], arrays['tpa'], arrays['chi2']]
            outputs = [component['field']]
            if self.has_carriers:
                inputs += [arrays['fca'], arrays['fcd'], arrays['dt_tau'], arrays['pair'], self._dt_f64]
                outputs.append(component['density'])
            loss_target = engine._nonlinear_loss_target(component['name'], component['box']) if accumulate else None
            if loss_target is not None:
                target, scale = loss_target
                inputs.append(scale)
                outputs.append(target)
            _CORRECTION_KERNELS[(self.has_carriers, loss_target is not None)](*inputs, *outputs)

    def carrier_density(self):
        """Full-grid float32 free-carrier density [1/m^3] at the cell centres (hz: the E_x and E_y face densities each
        averaged onto the centres, then summed); zeros without carriers."""
        engine = self._engine
        shape = (engine.nx, engine.ny)
        xp = engine.xp
        total = xp.zeros(shape, dtype=DTYPE)
        for component in self.components:
            if 'density' not in component:
                continue
            values = xp.zeros(shape, dtype=DTYPE)
            values[component['box']] = component['density'].astype(DTYPE)
            total += values if component['axis'] is None else collocate_to_centres(values, component['axis'],
                                                                                   component['periodic'])
        return total
