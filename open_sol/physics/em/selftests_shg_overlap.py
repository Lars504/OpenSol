"""Self-tests of the em_modes shg_overlap feature (nonlinear_overlap): plane-wave modes of a fully periodic anisotropic
cross-section as a model run with the Voigt contraction, and the fixed ring crystal on CPU; exact references."""
import math

import numpy as np

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import relative_error
from ...model.spec import DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec

_TYPE = 'em_modes'      # = definition_modes.EM_MODES.name (definition_modes imports this module)
_PHYSICS = 'modes'
_WAVELENGTH = 1.55e-6   # m, pump
_CELL = 50e-9           # m
_CELLS = 4              # per axis: periodic on every face, so the k_t = 0 modes are uniform (TEM) plane waves
_AREA = (_CELLS * _CELL) ** 2
_D11, _D21 = 2.5e-11, 4e-12   # m/V: P_x = eps0 d11 E_x^2, P_y = eps0 d21 E_x^2
_TOLERANCE = 1e-6       # relative: the TEM modes are exact discrete solutions, the budget is ARPACK / round-off

# Model case: background eps_xx 4.5, eps_r (= eps_yy = eps_zz) 4.0 plus one lossless Lorentz pole on every component.
# E_x sees eps_xx, E_y eps_yy, so mode 1 is the E_x plane wave (index sqrt(eps_xx)), mode 2 the E_y one.
_MODEL_EPS_XX, _MODEL_EPS_R = 4.5, 4.0
_POLE_DELTA_EPS, _POLE_WAVELENGTH = 1.0, 0.4e-6


def plane_wave_efficiency(d_value, pump_index, harmonic_index, wavelength, area):
    """Normalised SHG efficiency [1/(W m^2)] of uniform 1 W plane-wave modes: 8 pi^2 d^2 / (eps0 c n_w^2 n_2w lambda^2 A)."""
    return 8.0 * math.pi ** 2 * d_value ** 2 / (EPS0 * C0 * pump_index ** 2 * harmonic_index * wavelength ** 2 * area)


def _voigt_contraction_error(nonlinear_overlap):
    """Max relative error of second_order_polarization against the written-out Voigt sums for random d_il and fields."""
    generator = np.random.default_rng(7)
    values = dict(zip(nonlinear_overlap.VOIGT_NAMES, generator.normal(size=18)))
    a, b, c = generator.normal(size=3) + 1j * generator.normal(size=3)
    pairs = (a * a, b * b, c * c, 2 * b * c, 2 * a * c, 2 * a * b)
    expected = np.array([sum(values[f'd{i}{l}'] * pairs[l - 1] for l in range(1, 7)) for i in (1, 2, 3)])
    measured = nonlinear_overlap.second_order_polarization(nonlinear_overlap.voigt_tensor(values),
                                                           np.array([a, b, c]))
    return float(np.max(np.abs(measured - expected)) / np.max(np.abs(expected)))


_RING_RADIUS = 20e-6        # m
_RING_TOLERANCE = 1e-12     # relative: trig-polynomial identities, round-off only
_RING_PROBE_ANGLES = (0.37, 2.1, -1.3)   # rad, off the 8 sample azimuths


def _check_ring(_results):
    """Fixed ring crystal (nonlinear_overlap.ring_harmonics / ring_row) on random fields: the orders rebuild kappa(phi)
    at off-sample azimuths, d11 alone gives cos^3 (orders +-1 3/8, +-3 1/8), a d invariant about y only order 0 (= the
    local overlap), and ring_row picks m* = round(-delta_beta R) for both signs of R."""
    from . import nonlinear_overlap as nlo
    generator = np.random.default_rng(11)
    shape = (5, 3)

    def random_fields():
        return {name: generator.normal(size=shape) + 1j * generator.normal(size=shape) for name in ('ex', 'ey', 'ez')}

    overlap = nlo.overlap_tensor(random_fields(), random_fields(), generator.uniform(0.5, 1.5, size=shape))
    scale = float(np.max(np.abs(overlap)))
    errors = {}
    d_random = nlo.voigt_tensor(dict(zip(nlo.VOIGT_NAMES, generator.normal(size=18))))
    harmonics = nlo.ring_harmonics(d_random, overlap)
    for phi in _RING_PROBE_ANGLES:
        direct = complex(np.sum(nlo.rotated_tensor(d_random, nlo.ring_frame(phi)) * overlap))
        rebuilt = sum(kappa * np.exp(1j * order * phi) for order, kappa in harmonics.items())
        errors[f'rebuild phi = {phi}'] = abs(rebuilt - direct) / abs(direct)
    single = np.zeros((3, 3, 3), dtype=np.complex128)
    single[0, 0, 0] = overlap[0, 0, 0]
    cubic = nlo.ring_harmonics(nlo.voigt_tensor({'d11': 1.0}), single)
    expected_cubic = {-3: 0.125, -1: 0.375, 1: 0.375, 3: 0.125}
    errors['cos^3 orders'] = max(abs(cubic[order] - expected_cubic.get(order, 0.0) * single[0, 0, 0])
                                 for order in nlo.RING_ORDERS) / abs(single[0, 0, 0])
    a, b, c, e = generator.normal(size=4)
    invariant = nlo.voigt_tensor({'d22': a, 'd21': b, 'd23': b, 'd16': c, 'd34': c, 'd14': e, 'd36': -e})
    invariant_harmonics = nlo.ring_harmonics(invariant, overlap)
    local = complex(np.sum(invariant * overlap))
    errors['invariant order 0 = local'] = abs(invariant_harmonics[0] - local) / abs(local)
    errors['invariant orders != 0'] = max(abs(invariant_harmonics[order]) for order in nlo.RING_ORDERS
                                          if order) / scale
    omega = 2.0 * math.pi * C0 / _WAVELENGTH
    for radius, expected_order in ((_RING_RADIUS, -1), (-_RING_RADIUS, 1)):
        delta_beta = 1.02 / _RING_RADIUS
        row = dict(zip(nlo.RING_SERIES, nlo.ring_row(delta_beta, radius, harmonics, omega)))
        expected_mismatch = delta_beta + expected_order / radius
        efficiency = nlo.shg_efficiency(harmonics[expected_order], omega) * nlo.PERCENT_PER_W_CM2
        errors[f'ring_row R = {radius:g}'] = max(
            abs(row['ring_order'] - expected_order), abs(row['ring_mismatch'] / expected_mismatch - 1.0),
            abs(row['azimuthal_mismatch'] / (delta_beta * radius) - 1.0),
            abs(row['ring_efficiency'] / efficiency - 1.0),
            abs(row['ring_overlap'] / abs(harmonics[expected_order]) - 1.0))
    far = nlo.ring_row(10.0 / _RING_RADIUS, _RING_RADIUS, harmonics, omega)
    errors['beyond order 3: NaN order, 0 efficiency'] = (0.0 if math.isnan(far[1]) and far[4] == 0.0 else math.inf)
    worst = max(errors.values())
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _RING_TOLERANCE,
            'details': {'errors': errors}}


def _pole_eps(background, omega):
    """Relative eps of a model component at omega: background + the Lorentz pole (float32 pole parameters)."""
    omega_0 = float(np.float32(2.0 * math.pi * C0 / _POLE_WAVELENGTH))
    return background + float(np.float32(_POLE_DELTA_EPS)) * omega_0 ** 2 / (omega_0 ** 2 - omega ** 2)


def _model_spec():
    """em_modes on a fully periodic 4 x 4 cell uniaxial cross-section, mode_analysis, shg_overlap d11 / d21 and a d11
    one onto the orthogonal E_y harmonic."""
    size = _CELLS * _CELL
    settings = {f'bc_{face}': 'periodic' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    settings.update(wavelength=_WAVELENGTH, n_modes=2)
    common = {'pump_mode': 1, 'n_harmonic_modes': 2}
    return ModelSpec(
        title='em_modes shg_overlap plane waves',
        domain=DomainSpec(size_x=size, size_y=size),
        materials=[MaterialSpec('crystal', properties={'eps_r': _MODEL_EPS_R, 'eps_xx': _MODEL_EPS_XX},
                                selection=['all'],
                                dispersion=[{'model': 'lorentz', 'delta_eps': _POLE_DELTA_EPS,
                                             'omega_0': 2.0 * math.pi * C0 / _POLE_WAVELENGTH, 'gamma': 0.0}])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[
            FeatureSpec('shg_same', 'shg_overlap', ['all'], {**common, 'harmonic_mode': 1, 'd11': _D11}),
            FeatureSpec('shg_cross', 'shg_overlap', ['all'], {**common, 'harmonic_mode': 2, 'd21': _D21}),
            FeatureSpec('shg_orthogonal', 'shg_overlap', ['all'], {**common, 'harmonic_mode': 2, 'd11': _D11})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='mode_analysis'))


def _check_model(results_by_name):
    """Efficiency, n_eff and delta_beta of shg_same / shg_cross vs the plane-wave closed form at the pole-dispersed
    indices, the shg_orthogonal zero and the Voigt contraction."""
    from . import nonlinear_overlap
    results = results_by_name['modes']
    omega = 2.0 * math.pi * C0 / _WAVELENGTH
    n_pump = math.sqrt(_pole_eps(_MODEL_EPS_XX, omega))
    harmonic_index = {'shg_same': math.sqrt(_pole_eps(_MODEL_EPS_XX, 2.0 * omega)),
                      'shg_cross': math.sqrt(_pole_eps(_MODEL_EPS_R, 2.0 * omega))}
    d_values = {'shg_same': _D11, 'shg_cross': _D21}
    errors, details = {}, {}
    for feature, n_harmonic in harmonic_index.items():
        series = results.result(_PHYSICS, feature)['series']
        expected = (plane_wave_efficiency(d_values[feature], n_pump, n_harmonic, _WAVELENGTH, _AREA) * 1e-2)
        delta_beta = 2.0 * omega / C0 * (n_harmonic - n_pump)
        measured = {name: float(np.asarray(series[name])[0])
                    for name in ('n_eff_pump', 'n_eff_harmonic', 'delta_beta', 'efficiency')}
        errors[f'{feature}_efficiency'] = relative_error(measured['efficiency'], expected)
        errors[f'{feature}_n_eff_pump'] = relative_error(measured['n_eff_pump'], n_pump)
        errors[f'{feature}_n_eff_harmonic'] = relative_error(measured['n_eff_harmonic'], n_harmonic)
        errors[f'{feature}_delta_beta'] = relative_error(measured['delta_beta'], delta_beta)
        details[feature] = {'measured': measured, 'expected_efficiency_percent_per_w_cm2': expected,
                            'expected_delta_beta': delta_beta}
    orthogonal = float(np.asarray(results.result(_PHYSICS, 'shg_orthogonal')['series']['efficiency'])[0])
    same_expected = details['shg_same']['expected_efficiency_percent_per_w_cm2']
    errors['shg_orthogonal_efficiency'] = abs(orthogonal) / same_expected
    details['shg_orthogonal_efficiency'] = orthogonal
    errors['voigt_contraction'] = _voigt_contraction_error(nonlinear_overlap)
    worst = max(errors.values())
    if math.isnan(worst):
        worst = math.inf
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, **details}}


SHG_OVERLAP_SELFTEST_CASES = (
    SelfTestCase(
        name='em_shg_overlap_ring',
        label='EM modes SHG overlap: crystal fixed around a ring (CPU)',
        description="CPU check of shg_overlap ring_crystal 'fixed' (nonlinear_overlap.ring_harmonics / ring_row, no "
                    'solve) on random complex fields: the azimuthal orders m = -3..3 from 8 sample azimuths must rebuild '
                    'kappa(phi) = (Q(phi) Q Q : d) : T at three off-sample azimuths for a random d; d11 alone must give '
                    'kappa(phi) = cos^3(phi) kappa(0) (orders +-1 3/8, +-3 1/8, the rest 0); a d invariant about y '
                    '(d22, d21 = d23, d16 = d34, d14 = -d36) only order 0, equal to the local overlap; ring_row must '
                    'pick m* = round(-delta_beta R) for R = +-20 um (delta_beta R = +-1.02) with its mismatch, overlap '
                    'and efficiency, and report NaN order / 0 efficiency beyond |m| = 3. Tolerance 1e-12 relative '
                    '(round-off).',
        specs=lambda: {}, check=_check_ring),
    SelfTestCase(
        name='em_modes_shg_overlap',
        label='EM modes SHG overlap: model run',
        description='Fully periodic 4 x 4 cell (50 nm) em_modes model of a uniaxial medium (eps_xx 4.5, eps_r 4.0, one '
                    'lossless Lorentz pole at 400 nm on every component, mode_analysis at 1550 nm): the k_t = 0 modes '
                    'are uniform 1 W plane waves, exact on the lattice (mode 1 E_x, mode 2 E_y). shg_overlap features: '
                    'd11 to harmonic mode 1 and d21 to harmonic mode 2 must give the efficiency (%/(W cm^2)) 8 pi^2 '
                    'd^2 / (eps0 c n_w^2 n_2w lambda^2 A), n_eff of both modes and delta_beta = 2 k0 (n_2w - n_w) at '
                    'the pole-dispersed indices (float32 pole parameters); d11 to the E_y harmonic mode 2 must vanish '
                    '(relative to the d11 efficiency); plus the Voigt contraction (factor 2 on the mixed pairs) of '
                    'second_order_polarization on random d_il and fields. Tolerance 1e-6 relative (eigen-solve and '
                    'round-off only).',
        specs=lambda: {'modes': _model_spec()}, check=_check_model),
)
