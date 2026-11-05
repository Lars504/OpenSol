"""Coupled-mode SHG overlap of em_modes cross-section modes (host numpy, nonlinear_optics_plan stage 5 (e)): kappa =
sum A conj(e_2w) . (d : e_w e_w) over the chi2 cells, the normalised efficiency and the phase mismatch per pump solve."""
# Modes carry 1 W (1/2 Re integral (e x h*).z dA, exp(j (omega t - beta z))), SHG P(2w)_i = eps0 sum_jk d_ijk E_j E_k
# (d = chi2 / 2, model axes). Coupled-mode theory (excitation by J = j 2w P): dB/dz = -j (2 w eps0 / 4) kappa A^2
# exp(j delta_beta z), delta_beta = Re beta(2w) - 2 Re beta(w), so P_2w = eta P_w^2 L^2 sinc^2(delta_beta L / 2) with
# eta = (w eps0 / 2)^2 |kappa|^2 [1/(W m^2)]; a uniform plane-wave mode of area A gives 8 pi^2 d^2 / (eps0 c n_w^2 n_2w
# lambda^2 A). First-order quasi phase matching (50 % duty) multiplies eta by (2 / pi)^2. Fields are collocated to the
# cell centres by plain face means (cross_section_modes.centre_fields: first order in the normal E at interfaces).
# Bend (symmetry 2.19, s = r / R at the cell centres): z = the arc R phi at the reference x, dV = J dA dz, physical E =
# the mapped components / (h_x, h_y, h_z). 'exact' (x = r): J = s, h = (1, 1, s) (E_z = s E_phi); 'conformal' (x = the
# conformal coordinate, fields read as physical): J = s^2, h = 1, exact for the E_y family like the map itself. d stays
# in the local (radial x, vertical y, azimuthal z) frame ('local'): constant around the ring only for a d invariant
# about y. Ring crystal 'fixed' (an x-cut ring): d is given at the reference azimuth phi = 0 and the crystal stays put,
# so the local frame at phi has rows Q(phi) = (e_r, y, e_phi) = ((cos, 0, sin), (0, 1, 0), (-sin, 0, cos)) in the
# phi = 0 axes (propagation +z = +phi, centre on the -x side) and d_loc(phi) = Q Q Q : d. kappa(phi) = d_loc(phi) : T
# with the phi-free overlap tensor T_ijk = sum J A conj(h_i) p_j p_k is a trig polynomial of degree 3, kappa(phi) =
# sum_{m=-3..3} kappa_m exp(j m phi) (exact from RING_SAMPLES equidistant phi), and with z = R phi the order m is phase
# matched when delta_beta R + m = 0: the ring order m* = round(-delta_beta R) gives P_2w = eta_m* P_w^2 L^2 sinc^2(
# (delta_beta + m* / R) L / 2) over an arc L (no ring harmonic beyond |m| = 3: eta 0).
import math

import numpy as np

from ...core.constants import C0, EPS0
from ...core.wavelengths import wavelength_label
from .cross_section_modes import centre_fields

VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))   # Voigt l = 1..6 -> field index pair (j, k)
VOIGT_NAMES = tuple(f'd{i}{l}' for i in (1, 2, 3) for l in range(1, 7))   # d_il, i = 1 x, 2 y, 3 z (model axes)
QPM_FIRST_ORDER = (2.0 / math.pi) ** 2
PERCENT_PER_W_CM2 = 1e-2   # 1 / (W m^2) in % / (W cm^2)
SHG_OVERLAP_SERIES = ('n_eff_pump', 'n_eff_harmonic', 'delta_beta', 'coherence_length', 'poling_period', 'overlap',
                      'efficiency', 'efficiency_qpm')
SHG_OVERLAP_UNITS = ('1', '1', '1/m', 'm', 'm', 'V^2/W^1.5', '%/(W cm^2)', '%/(W cm^2)')
RING_CRYSTALS = ('local', 'fixed')
RING_ORDERS = tuple(range(-3, 4))
RING_SAMPLES = 8   # >= 7 equidistant azimuths resolve the orders -3..3 of a rank-3 tensor exactly
RING_SERIES = ('azimuthal_mismatch', 'ring_order', 'ring_mismatch', 'ring_overlap', 'ring_efficiency',
               *(f"ring_overlap_{'minus' if order < 0 else 'plus' if order > 0 else ''}{abs(order)}"
                 for order in RING_ORDERS))
RING_UNITS = ('1', '1', '1/m', 'V^2/W^1.5', '%/(W cm^2)', *('V^2/W^1.5',) * len(RING_ORDERS))
_E_COMPONENTS = ('ex', 'ey', 'ez')


def voigt_tensor(values):
    """(3, 3, 3) float64 d_ijk [m/V] in model axes from {d_il name: value} (missing names = 0), d_ijk = d_ikj."""
    tensor = np.zeros((3, 3, 3))
    for name in VOIGT_NAMES:
        value = float(values.get(name, 0.0))
        i, (j, k) = int(name[1]) - 1, VOIGT_PAIRS[int(name[2]) - 1]
        tensor[i, j, k] = tensor[i, k, j] = value
    return tensor


def second_order_polarization(d_tensor, fields):
    """P / eps0 = sum_jk d_ijk E_j E_k [V/m] (the SHG phasor) of complex fields (3, ...)."""
    return np.einsum('ijk,j...,k...->i...', d_tensor, fields, fields)


def bend_overlap_factors(bend, n_x):
    """(volume J (nx,), field (3, nx) = 1 / h) of the overlap at the cell centres for a cross_section_modes bend tuple
    (radius, reference_x, x_edges, model) or None (all 1): module header."""
    volume, field = np.ones(n_x), np.ones((3, n_x))
    if bend is None or float(bend[0]) == 0.0:
        return volume, field
    radius, reference_x, x_edges, model = bend
    x_edges = np.asarray(x_edges, dtype=np.float64)
    scale = 1.0 + (0.5 * (x_edges[:-1] + x_edges[1:]) - float(reference_x)) / float(radius)
    if model == 'conformal':
        return scale ** 2, field
    if model != 'exact':
        raise ValueError(f"bend model must be 'exact' or 'conformal', got {model!r}")
    field[2] = 1.0 / scale
    return scale, field


def overlap_tensor(pump_centres, harmonic_centres, area, bend=None):
    """Complex (3, 3, 3) T_ijk = sum J area conj(e_2w,i) e_w,j e_w,k [V^2/W^1.5 per m/V] of centre-field dicts (ex, ey,
    ez), physical components of a bend (bend_overlap_factors); area (nx, ny) [m^2] = the chi2 cells (0 elsewhere)."""
    volume, field = bend_overlap_factors(bend, np.shape(area)[0])
    pump = np.stack([pump_centres[name] for name in _E_COMPONENTS]) * field[:, :, None]
    harmonic = np.stack([harmonic_centres[name] for name in _E_COMPONENTS]) * field[:, :, None]
    weight = area * volume[:, None]
    return np.einsum('xy,ixy,jxy,kxy->ijk', weight, np.conj(harmonic), pump, pump, optimize=True)


def shg_overlap(pump_centres, harmonic_centres, d_tensor, area, bend=None):
    """kappa = sum J area conj(e_2w) . (d : e_w e_w) = d : T [V^2/W^1.5] (overlap_tensor arguments)."""
    return complex(np.sum(np.asarray(d_tensor) * overlap_tensor(pump_centres, harmonic_centres, area, bend)))


def ring_frame(phi):
    """(3, 3) rows = the local (e_r, y, e_phi) axes at azimuth phi [rad] in the phi = 0 model axes (module header)."""
    c, s = math.cos(phi), math.sin(phi)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotated_tensor(d_tensor, rotation):
    """d'_ijk = Q_ia Q_jb Q_kc d_abc of a (3, 3, 3) tensor for rotation rows Q (3, 3)."""
    return np.einsum('ia,jb,kc,abc->ijk', rotation, rotation, rotation, d_tensor)


def ring_harmonics(d_tensor, overlap):
    """{m: kappa_m} for m in RING_ORDERS of kappa(phi) = (Q(phi) Q Q : d) : T = sum_m kappa_m exp(j m phi): a crystal
    fixed around the ring (d at phi = 0), overlap T = overlap_tensor (module header)."""
    phis = 2.0 * math.pi * np.arange(RING_SAMPLES) / RING_SAMPLES
    samples = np.array([np.sum(rotated_tensor(d_tensor, ring_frame(phi)) * overlap) for phi in phis])
    return {order: complex(np.mean(samples * np.exp(-1j * order * phis))) for order in RING_ORDERS}


def ring_row(delta_beta, radius, harmonics, omega):
    """RING_SERIES values: delta_beta R, the nearest order m* = round(-delta_beta R) (NaN beyond |3|: no harmonic),
    delta_beta + m* / R, |kappa_m*| and its efficiency [%/(W cm^2)] (0 beyond |3|), then |kappa_m| per RING_ORDERS."""
    mismatch = delta_beta * radius
    order = int(round(-mismatch)) if math.isfinite(mismatch) else None
    if order is None or abs(order) > RING_ORDERS[-1]:
        head = [mismatch, math.nan, math.nan, 0.0, 0.0]
    else:
        kappa = harmonics[order]
        head = [mismatch, float(order), delta_beta + order / radius, abs(kappa),
                shg_efficiency(kappa, omega) * PERCENT_PER_W_CM2]
    return head + [abs(harmonics[order_m]) for order_m in RING_ORDERS]


def shg_efficiency(kappa, omega):
    """Phase-matched normalised SHG efficiency (omega eps0 / 2)^2 |kappa|^2 [1/(W m^2)] at pump omega [rad/s]."""
    return (0.5 * omega * EPS0) ** 2 * abs(kappa) ** 2


def shg_row(pump_mode, harmonic_mode, kappa):
    """SHG_OVERLAP_SERIES values of a mode pair (pump at omega, harmonic at 2 omega) with overlap kappa; lengths NaN at
    delta_beta = 0."""
    delta_beta = complex(harmonic_mode.beta).real - 2.0 * complex(pump_mode.beta).real
    efficiency = shg_efficiency(kappa, pump_mode.omega) * PERCENT_PER_W_CM2
    coherence = math.pi / abs(delta_beta) if delta_beta != 0.0 else math.nan
    return [complex(pump_mode.n_eff).real, complex(harmonic_mode.n_eff).real, delta_beta, coherence, 2.0 * coherence,
            abs(kappa), efficiency, QPM_FIRST_ORDER * efficiency]


class ShgOverlapMonitor:
    """SHG overlap of pump mode `pump_mode` with harmonic mode `harmonic_mode` of a 2 omega solve the monitor runs after
    every em_modes solve (host); cell_mask = host bool (nx, ny) chi2 cells. One row per solved omega; ring_crystal
    'fixed' (a bend only) appends RING_SERIES."""

    def __init__(self, d_tensor, cell_mask, pump_mode=1, harmonic_mode=1, n_harmonic_modes=4,
                 harmonic_n_eff_guess=None, ring_crystal='local'):
        self.d_tensor = np.asarray(d_tensor, dtype=np.float64)
        self.cell_mask = np.asarray(cell_mask, dtype=bool)
        self.pump_mode = int(pump_mode)
        self.harmonic_mode = int(harmonic_mode)
        self.n_harmonic_modes = int(n_harmonic_modes)
        self.harmonic_n_eff_guess = harmonic_n_eff_guess
        if not 1 <= self.harmonic_mode <= self.n_harmonic_modes or self.pump_mode < 1:
            raise ValueError(f'shg_overlap: pump_mode >= 1 and 1 <= harmonic_mode <= n_harmonic_modes needed, got '
                             f'{self.pump_mode}, {self.harmonic_mode}, {self.n_harmonic_modes}')
        if ring_crystal not in RING_CRYSTALS:
            raise ValueError(f'shg_overlap: ring_crystal must be one of {RING_CRYSTALS}, got {ring_crystal!r}')
        self.ring_crystal = ring_crystal
        self.series = SHG_OVERLAP_SERIES + (RING_SERIES if ring_crystal == 'fixed' else ())
        self.units = SHG_OVERLAP_UNITS + (RING_UNITS if ring_crystal == 'fixed' else ())
        self._area = None
        self._records = {}     # pump omega [rad/s] -> self.series values
        self.n_harmonic_solves = 0

    def record(self, physics):
        """Solve the 2 omega modes of physics' current materials, then the row of the selected pair (NaN when a mode
        is missing, with a runtime warning); ValueError for a fixed ring crystal without a bend."""
        omega = float(physics.solved_omega)
        label = f'{wavelength_label(2.0 * math.pi * C0 / omega)} shg_overlap'
        nan_row = [math.nan] * len(self.series)
        bend = physics.bend
        if self.ring_crystal == 'fixed' and (bend is None or float(bend[0]) == 0.0):
            raise ValueError("shg_overlap ring_crystal 'fixed' needs a bend (bend_radius != 0): the crystal turns "
                             'relative to the guide only around a ring')
        if self.pump_mode > len(physics.modes):
            physics.runtime_warnings.append(f'{label}: pump mode {self.pump_mode} not found ({len(physics.modes)} '
                                            f'modes): NaN row')
            self._records[omega] = nan_row
            return
        if physics.modes[self.pump_mode - 1].mirror_factor != 1.0:   # before the 2 omega solve
            raise ValueError('shg_overlap needs the whole cross-section: a mirror face keeps only one parity of the '
                             'overlap integrand; use pec / pmc / cpml / periodic faces')
        solve = physics.solve_detached(2.0 * omega, self.n_harmonic_modes, self.harmonic_n_eff_guess)
        self.n_harmonic_solves += 1
        physics.runtime_warnings.extend(f'{label} (second harmonic): {warning}' for warning in solve.info['warnings']
                                        if warning not in physics.finalize_warnings)
        if self.harmonic_mode > len(solve.modes):
            physics.runtime_warnings.append(f'{label}: harmonic mode {self.harmonic_mode} not found '
                                            f'({len(solve.modes)} modes at 2 omega): NaN row (set harmonic_n_eff_guess '
                                            f'or raise n_harmonic_modes)')
            self._records[omega] = nan_row
            return
        pump, harmonic = physics.modes[self.pump_mode - 1], solve.modes[self.harmonic_mode - 1]
        if self._area is None:   # cell areas inside the walls (wall ghosts 0) on the chi2 cells, once
            self._area = np.where(self.cell_mask, physics.point_materials().centre_area, 0.0)
        bc = physics.boundary_config
        periodic = (bool(bc.periodic_x), bool(bc.periodic_y))
        overlap = overlap_tensor(centre_fields(pump, *periodic), centre_fields(harmonic, *periodic), self._area, bend)
        row = shg_row(pump, harmonic, complex(np.sum(self.d_tensor * overlap)))
        if self.ring_crystal == 'fixed':
            row += ring_row(row[2], float(bend[0]), ring_harmonics(self.d_tensor, overlap), pump.omega)
        self._records[omega] = row

    def recorded(self):
        """(omega float64 in ascending wavelength order, float64 (n, len(self.series)) rows); ValueError before the
        first solve."""
        if not self._records:
            raise ValueError('not solved yet: the SHG overlap has no solve (run a mode_analysis or frequency_domain '
                             'study)')
        omegas = sorted(self._records, reverse=True)
        return (np.array(omegas, dtype=np.float64),
                np.array([self._records[omega] for omega in omegas], dtype=np.float64))


def shg_overlap_result(monitor) -> dict:
    """'xy' of a ShgOverlapMonitor vs ascending pump wavelength [m]: SHG_OVERLAP_SERIES (+ RING_SERIES for a fixed
    ring crystal); extras frequency, wavelength, pump_mode, harmonic_mode."""
    omegas, values = monitor.recorded()
    frequency = omegas / (2.0 * math.pi)
    wavelength = C0 / frequency
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Pump wavelength', 'x_unit': 'm',
            'y_label': 'SHG efficiency', 'series': {name: values[:, column]
                                                    for column, name in enumerate(monitor.series)},
            'series_units': dict(zip(monitor.series, monitor.units)), 'frequency': frequency,
            'wavelength': wavelength.copy(), 'pump_mode': monitor.pump_mode, 'harmonic_mode': monitor.harmonic_mode}
