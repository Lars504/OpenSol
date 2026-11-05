"""C5 four-level gain media of the 2D FDTD engines (time domain only): a Lorentz polarisation driven by the population
difference of a four-level two-electron system (Chang and Taflove, Opt. Express 12, 3827 (2004)), allocated per region
and E component set on the bounding box of its weights (kernel notes at _polarization_update)."""
import math

import numpy as np
from ....core.backend import cp
from ....core.kernels import twin

from ....core.constants import C0, DTYPE, EPS0, HBAR
from ..fields import collocate_to_centres
from ..yee_grid import face_average
from .fdtd_2d_base import _bounding_box
from .fdtd_ez_2d_kernels import lorentz_e_correction

# d2P/dt2 + gamma dP/dt + omega_a^2 P = kappa (N1 - N2) E, kappa = 6 pi eps0 c^3 / (omega_a^2 tau_21), stepped in the
# velocity form of the Lorentz pole (fdtd_ez_2d_kernels: dp = P^{n+1} - P^n from (dp, p, E^n)); drive_per_n = kappa
# dt^2 / (1 + gamma dt / 2) is ~1e-39 (below the float32 range), so drive_per_n (N1 - N2) is formed in float64. P is
# the face / cell weight w times the polarisation of the active material (hz interface faces: w = 1/2). E^n is kept for
# the population step. N1 > N2 absorbs like a passive Lorentz pole, N2 > N1 amplifies. core.kernels twins: the numpy
# twins follow every change of the cupy sources.
def _polarization_update_host(drive_per_n, damping, restoring, weight, e, n1, n2, p, dp, e_prev):
    drive = (drive_per_n * (n1 - n2)).astype(np.float32) * weight
    dp += damping * dp - restoring * p + drive * e
    p += dp
    e_prev[...] = e


_polarization_update = twin('em2d.gain_polarization_update', cp.ElementwiseKernel(
    'float64 drive_per_n, float32 damping, float32 restoring, float32 weight, float32 e, float64 n1, float64 n2',
    'float32 p, float32 dp, float32 e_prev',
    '''
    float drive = (float)(drive_per_n * (n1 - n2)) * weight;
    dp += damping * dp - restoring * p + drive * e;
    p += dp;
    e_prev = e;
    ''',
    'gain_polarization_update'), _polarization_update_host)

# Rate equations after the E correction, flows from the old populations (explicit, conserving N0 + N1 + N2 + N3 exactly;
# stable for dt << every tau and dt W_p << 1): pump W_p N0 -> N3, N3 / tau_32 -> N2, N2 / tau_21 -> N1, N1 / tau_10 -> N0,
# and the stimulated transitions N1 -> N2 per active volume (E^n + E^{n+1}) / 2 (P^{n+1} - P^n) / (hbar omega_a w), the
# time-centred work on the polarisation (> 0 absorbs). Cells / faces with w = 0 hold no population.
def _population_update_host(e, e_prev, dp, weight, inv_hbar_omega, dt, pump, inv_tau32, inv_tau21, inv_tau10,
                            n0, n1, n2, n3):
    active = weight > 0.0
    safe_weight = np.where(active, weight, np.float32(1.0)).astype(np.float64)
    stimulated = np.where(active, 0.5 * (e.astype(np.float64) + e_prev.astype(np.float64)) * dp.astype(np.float64)
                          * inv_hbar_omega / safe_weight, 0.0)
    pumped = dt * pump * n0
    decay32 = dt * inv_tau32 * n3
    decay21 = dt * inv_tau21 * n2
    decay10 = dt * inv_tau10 * n1
    n3 += pumped - decay32
    n2 += decay32 - decay21 + stimulated
    n1 += decay21 - decay10 - stimulated
    n0 += decay10 - pumped


_population_update = twin('em2d.gain_population_update', cp.ElementwiseKernel(
    'float32 e, float32 e_prev, float32 dp, float32 weight, float64 inv_hbar_omega, float64 dt, float64 pump, '
    'float64 inv_tau32, float64 inv_tau21, float64 inv_tau10',
    'float64 n0, float64 n1, float64 n2, float64 n3',
    '''
    double stimulated = 0.0;
    if (weight > 0.0f) {
        stimulated = 0.5 * ((double)e + (double)e_prev) * (double)dp * inv_hbar_omega / (double)weight;
    }
    double pumped = dt * pump * n0;
    double decay32 = dt * inv_tau32 * n3;
    double decay21 = dt * inv_tau21 * n2;
    double decay10 = dt * inv_tau10 * n1;
    n3 += pumped - decay32;
    n2 += decay32 - decay21 + stimulated;
    n1 += decay21 - decay10 - stimulated;
    n0 += decay10 - pumped;
    ''',
    'gain_population_update'), _population_update_host)


def gain_coupling(wavelength, tau_21):
    """kappa = 6 pi eps0 c^3 / (omega_a^2 tau_21) of the polarisation drive [C^2 / kg] (radiative lifetime tau_21)."""
    omega = 2.0 * math.pi * C0 / wavelength
    return 6.0 * math.pi * EPS0 * C0 ** 3 / (omega ** 2 * tau_21)


def add_gain_region(engine, mask, wavelength, linewidth, density, pump_rate=0.0, tau_32=1e-13, tau_21=1e-9,
                    tau_10=1e-13, initial_upper_fraction=0.0):
    """Append a four-level gain region (SI parameters) on the run-backend bool (nx, ny) cell mask to engine.gain_media;
    regions add (two overlapping regions are two species). NotImplementedError with Bloch k != 0."""
    if engine.complex_fields:
        raise NotImplementedError('gain media are not available with a Bloch phase k != 0: the stimulated transitions '
                                  'follow E.dP/dt of the real field set only')
    engine._check_cell_mask(mask, 'Gain medium mask')
    values = {'wavelength': wavelength, 'linewidth': linewidth, 'density': density, 'pump_rate': pump_rate,
              'tau_32': tau_32, 'tau_21': tau_21, 'tau_10': tau_10}
    if not all(math.isfinite(float(value)) and value >= 0 for value in values.values()):
        raise ValueError(f'gain medium parameters must be finite and >= 0, got {values}')
    if not (wavelength > 0 and density > 0 and tau_32 > 0 and tau_21 > 0 and tau_10 > 0):
        raise ValueError(f'gain medium wavelength, density and the three lifetimes must be > 0, got {values}')
    if not 0.0 <= initial_upper_fraction <= 1.0:
        raise ValueError(f'gain medium initial_upper_fraction must lie in [0, 1], got {initial_upper_fraction}')
    region = GainRegion(engine, mask, wavelength, linewidth, density, pump_rate, tau_32, tau_21, tau_10,
                        initial_upper_fraction)
    if region.components:
        engine.gain_media.append(region)


class GainRegion:
    """One gain region of an FDTD engine: per E component (ez: cells; hz: E_x / E_y faces, face_average of the cell
    mask as weight) on the bounding box of its weight: weight, P, dP, E^n (float32) and the four float64 populations."""

    def __init__(self, engine, mask, wavelength, linewidth, density, pump_rate, tau_32, tau_21, tau_10,
                 initial_upper_fraction):
        dt = float(engine.dt)
        omega = 2.0 * math.pi * C0 / wavelength
        denominator = 1.0 + linewidth * dt / 2.0
        self._damping = DTYPE(-linewidth * dt / denominator)
        self._restoring = DTYPE(omega ** 2 * dt ** 2 / denominator)
        self._drive_per_n = np.float64(gain_coupling(wavelength, tau_21) * dt ** 2 / denominator)
        self._inv_hbar_omega = np.float64(1.0 / (HBAR * omega))
        self._dt = np.float64(dt)
        self._inv_dt = DTYPE(1.0 / dt)
        self._rates = tuple(np.float64(rate) for rate in (pump_rate, 1.0 / tau_32, 1.0 / tau_21, 1.0 / tau_10))
        self.components = []
        self._xp = xp = engine.xp
        cell_weight = mask.astype(DTYPE)
        for spec in engine._nonlinear_components():
            axis = spec['axis']
            weight = (cell_weight if axis is None
                      else face_average(cell_weight, axis, spec['periodic'], spec['widths'], xp=xp))
            box = _bounding_box(weight)
            if box is None:
                continue
            weight_box = weight[box].copy()   # copy: a view would keep the full-grid temporary alive
            active = (weight_box > 0).astype(np.float64) * float(density)
            self.components.append({
                'name': spec['name'], 'axis': axis, 'periodic': spec['periodic'], 'box': box, 'weight': weight_box,
                'field': spec['field'][box], 'cb': spec['cb'][box],
                'p': xp.zeros(weight_box.shape, dtype=DTYPE), 'dp': xp.zeros(weight_box.shape, dtype=DTYPE),
                'e_prev': xp.zeros(weight_box.shape, dtype=DTYPE),
                'populations': [active * (1.0 - initial_upper_fraction), xp.zeros_like(active),
                                active * initial_upper_fraction, xp.zeros_like(active)]})

    def update_polarization(self):
        """P^{n+1} from E^n and N1 - N2 at step n (pole stage, before the Yee update); keeps E^n."""
        for component in self.components:
            populations = component['populations']
            _polarization_update(self._drive_per_n, self._damping, self._restoring, component['weight'],
                                 component['field'], populations[1], populations[2],
                                 component['p'], component['dp'], component['e_prev'])

    def correct_field(self):
        """E^{n+1} -= (cb / dt) (P^{n+1} - P^n), as a Lorentz pole."""
        for component in self.components:
            lorentz_e_correction(component['dp'], component['cb'], self._inv_dt, component['field'])

    def update_populations(self):
        """Rate equations with the time-centred stimulated transitions (after every E correction)."""
        pump, inv_tau32, inv_tau21, inv_tau10 = self._rates
        for component in self.components:
            _population_update(component['field'], component['e_prev'], component['dp'], component['weight'],
                               self._inv_hbar_omega, self._dt, pump, inv_tau32, inv_tau21, inv_tau10,
                               *component['populations'])

    def inversion(self, shape):
        """Full-grid float32 N2 - N1 [1/m^3] at the cell centres (hz: the mean of the E_x and E_y face values)."""
        total = self._xp.zeros(shape, dtype=DTYPE)
        n_sets = len(self.components)
        for component in self.components:
            values = self._xp.zeros(shape, dtype=DTYPE)
            populations = component['populations']
            values[component['box']] = (populations[2] - populations[1]).astype(DTYPE)
            if component['axis'] is not None:
                values = collocate_to_centres(values, component['axis'], component['periodic'])
            total += values * DTYPE(1.0 / n_sets if component['axis'] is not None else 1.0)
        return total
