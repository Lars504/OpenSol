"""Thermorefractive noise validation cases (thermal_noise study): the exact Lorentzian of a cosine eigenmode of the
insulated finite-volume operator, the fluctuation-dissipation sum rule across a material interface (expression
weightings), the mode-pair weighting of the TE01 mode of a pec box (full and mirrored half), the
waveguide_thermal_noise template and its line-block GPU solve against SuperLU."""
import copy
import math

import numpy as np

from ...core.constants import C0, EPS0, K_B
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, SolverSpec, StudySpec)

HEAT_TYPE_NAME = 'heat'            # = definition.PHYSICS_TYPE_NAME (definition imports this module)
MODE_TYPE_NAME = 'em_modes'        # = physics.em.definition_modes type name
NOISE_FEATURE_TYPE = 'thermorefractive_noise'
NOISE_STUDY_TYPE = 'thermal_noise'
PHYSICS_NAME = 'heat'
MODE_PHYSICS_NAME = 'modes'
NOISE_FEATURE_NAME = 'noise'
RUN_NAME = 'run'

REFERENCE_TEMPERATURE = 300.0      # K, heat T0 (the noise scales with T0^2)
HEAT_SETTINGS = {'T0': REFERENCE_TEMPERATURE}
INTERACTION_LENGTH = 1e-4          # m, uniform axial profile length L
WATER_LIKE = {'k': 0.6, 'rho': 1000.0, 'cp': 4000.0}               # W/(m*K), kg/m^3, J/(kg*K)
NIOBATE_LIKE = {'k': 4.6, 'rho': 4640.0, 'cp': 628.0}


def _stored(value):
    """value as the float32 material arrays hold it, back in float64."""
    return float(np.float32(value))


def _capacity(properties):
    """rho * cp [J/(m^3*K)] of the float32-stored material values, multiplied in float64 like the noise operator."""
    return _stored(properties['rho']) * _stored(properties['cp'])


def _log_band(start, stop, count):
    """The log-spaced inclusive frequencies [Hz] the thermal_noise study solves."""
    return np.logspace(math.log10(start), math.log10(stop), count)


def _noise_feature(weight_expression, axial_wavenumber):
    """thermorefractive_noise on all cells: expression weighting, uniform axial profile over INTERACTION_LENGTH."""
    return FeatureSpec(NOISE_FEATURE_NAME, NOISE_FEATURE_TYPE, ['all'], {
        'weighting': 'expression', 'weight': weight_expression, 'axial_wavenumber': axial_wavenumber,
        'axial_profile': 'uniform', 'interaction_length': INTERACTION_LENGTH})


def _noise_study(band):
    start, stop, count = band
    return StudySpec(type=NOISE_STUDY_TYPE, settings={'frequency_start': start, 'frequency_stop': stop,
                                                      'n_frequencies': count})


def _noise_spectrum(results):
    """(noise result dict, float64 frequencies [Hz], float64 one-sided psd [1/Hz]) of one run."""
    result = results.result(PHYSICS_NAME, NOISE_FEATURE_NAME)
    return (result, np.asarray(result['x'], dtype=np.float64),
            np.asarray(result['series']['psd'], dtype=np.float64))


# heat_noise_cosine_lorentzian: a uniform water-like rectangle, every face insulated, uniform cells, weight
# w = cos(pi x / Lx) (x from the x_lo edge). The cell-centred cosine is an exact eigenvector of the insulated FV
# operator with eigenvalue k (2/dx)^2 sin^2(pi / (2 N)) per unit area, so with the axial term k q^2 the noise system is
# diagonal in it: S(f) = 4 k_B T0^2 L (Lx Ly / 2) K_m / ((omega rho cp)^2 + K_m^2), K_m = k ((2/dx)^2 sin^2(pi/(2N)) +
# q^2), and the variance k_B T0^2 L (Lx Ly / 2) / (rho cp) (sum over cells of cos^2 = N / 2 exactly).
COSINE_SIZE = (2e-6, 1e-6)         # m, Lx x Ly
COSINE_CELL = 5e-8                 # m, uniform mesh
COSINE_N_X = round(COSINE_SIZE[0] / COSINE_CELL)   # 40 cells along x
COSINE_WAVENUMBER = 1e7            # 1/m, axial q
COSINE_BAND = (1e4, 1e10, 25)      # Hz, start, stop, count (corner ~2.4e6 Hz in the middle)
COSINE_TOLERANCE = 1e-5            # on the max relative error; float32 weights / areas / k faces ~1e-7, SuperLU ~1e-12


def _cosine_mode_conductance():
    """K_m [W/(m^3*K)] of the cosine mode: k ((2/dx)^2 sin^2(pi / (2 N)) + q^2) with the float32-stored k."""
    in_plane = (2.0 / COSINE_CELL) ** 2 * math.sin(math.pi / (2.0 * COSINE_N_X)) ** 2
    return _stored(WATER_LIKE['k']) * (in_plane + COSINE_WAVENUMBER ** 2)


def _cosine_mode_weight_integral():
    """sum of w^2 A over the cells = Lx Ly / 2 [m^2]."""
    return 0.5 * COSINE_SIZE[0] * COSINE_SIZE[1]


COSINE_CORNER = _cosine_mode_conductance() / (2.0 * math.pi * _capacity(WATER_LIKE))   # Hz


def _cosine_expected_psd(frequencies):
    """Exact one-sided PSD [1/Hz] of the cosine mode at frequencies [Hz]."""
    conductance = _cosine_mode_conductance()
    omega_capacity = 2.0 * math.pi * frequencies * _capacity(WATER_LIKE)
    return (4.0 * K_B * REFERENCE_TEMPERATURE ** 2 * INTERACTION_LENGTH * _cosine_mode_weight_integral()
            * conductance / (omega_capacity ** 2 + conductance ** 2))


def _cosine_lorentzian_runs():
    size_x, size_y = COSINE_SIZE
    return {RUN_NAME: ModelSpec(
        title='Thermal noise of a cosine eigenmode',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        materials=[MaterialSpec('water_like', properties=dict(WATER_LIKE), selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, settings=dict(HEAT_SETTINGS), features=[
            _noise_feature(f'cos(pi * x / {size_x!r})', COSINE_WAVENUMBER)])],
        mesh=MeshSpec(mode='uniform', dx=COSINE_CELL),
        study=_noise_study(COSINE_BAND))}


def _check_cosine_lorentzian(results_by_name):
    """Max relative error of the PSD against the exact Lorentzian, and of the 'variance' key; inf when the solved
    frequencies are not the requested band."""
    result, frequencies, psd = _noise_spectrum(results_by_name[RUN_NAME])
    requested = _log_band(*COSINE_BAND)
    if frequencies.shape != requested.shape or psd.shape != requested.shape:
        return {'measured': math.nan, 'expected': float(_cosine_expected_psd(requested)[0]), 'error': math.inf,
                'tolerance': COSINE_TOLERANCE,
                'details': {'n_frequencies': int(frequencies.size), 'expected_frequencies': requested.size}}
    expected_psd = _cosine_expected_psd(frequencies)
    expected_variance = (K_B * REFERENCE_TEMPERATURE ** 2 * INTERACTION_LENGTH * _cosine_mode_weight_integral()
                         / _capacity(WATER_LIKE))
    relative_errors = np.abs(psd - expected_psd) / expected_psd
    errors = {'psd': float(np.max(relative_errors)),   # NaN propagates: a non-finite psd fails the case
              'variance': abs(float(result['variance']) - expected_variance) / expected_variance,
              'frequencies': float(np.max(np.abs(frequencies - requested) / requested))}
    return {'measured': float(psd[0]), 'expected': float(expected_psd[0]),
            'error': max(errors.values()), 'tolerance': COSINE_TOLERANCE,
            'details': {'errors': errors, 'corner_frequency_Hz': COSINE_CORNER,
                        'mode_conductance_W_per_m3K': _cosine_mode_conductance(),
                        'variance': float(result['variance']), 'expected_variance': expected_variance,
                        'band_fraction': float(result['band_fraction']),
                        'relative_error_per_frequency': float_list(relative_errors),
                        'frequencies_Hz': float_list(frequencies)}}


# heat_noise_sum_rule: a water-like background with a niobate-like block, temperature held on y_lo, a Gaussian
# weighting centred on the block's x_lo edge. Fluctuation-dissipation: integral_0^inf S(f) df = k_B T0^2 L
# sum |w|^2 A / (rho cp) for any symmetric K >= 0, independent of the conductivities and boundaries.
SUM_RULE_SIZE = (4e-6, 2e-6)       # m
SUM_RULE_CELL = 5e-8               # m, uniform mesh (80 x 40 cells; the block edges are cell edges)
CRYSTAL_BOX = (1.5e-6, 0.5e-6, 2.5e-6, 1.5e-6)   # m, x0, y0, x1, y1 of the niobate-like block
SUM_RULE_MATERIALS = ('water_like', 'niobate_like')   # background, block: one noise contribution each
GAUSSIAN_CENTRE = (1.5e-6, 1.0e-6)   # m, on the block's x_lo edge: the weighting straddles the interface
GAUSSIAN_RADIUS = 3e-7             # m, s of w = exp(-r^2 / s^2)
SUM_RULE_WAVENUMBER = 5e6          # 1/m, axial q
SUM_RULE_BAND = (1e3, 1e11, 33)    # Hz, 4 points per decade
# trapezoid in ln f (sech-like bumps, step 0.58: ~exp(-pi^2 / h) ~ 4e-8) and the tail models (~3e-6 low, ~6e-5 high)
SUM_RULE_TOLERANCE = 1e-3


def _diffusivity(properties):
    return _stored(properties['k']) / _capacity(properties)


def _relaxation_band():
    """(slowest, fastest) relaxation frequency estimates [Hz]: min D q^2 / (2 pi) is a strict lower bound of the
    generalized eigenvalues; max D (8 / dx^2 + q^2) / (2 pi) the in-plane checkerboard scale."""
    diffusivities = (_diffusivity(WATER_LIKE), _diffusivity(NIOBATE_LIKE))
    slowest = min(diffusivities) * SUM_RULE_WAVENUMBER ** 2 / (2.0 * math.pi)
    fastest = max(diffusivities) * (8.0 / SUM_RULE_CELL ** 2 + SUM_RULE_WAVENUMBER ** 2) / (2.0 * math.pi)
    return slowest, fastest


def _sum_rule_weight_expression():
    centre_x, centre_y = GAUSSIAN_CENTRE
    return f'exp(-((x - {centre_x!r})**2 + (y - {centre_y!r})**2) / {GAUSSIAN_RADIUS!r}**2)'


def _sum_rule_expected_variance():
    """k_B T0^2 L sum w^2 A / (rho cp) [1] over the uniform cells (block cells by half-open centre test)."""
    count_x = round(SUM_RULE_SIZE[0] / SUM_RULE_CELL)
    count_y = round(SUM_RULE_SIZE[1] / SUM_RULE_CELL)
    x_centers = (np.arange(count_x) + 0.5) * (SUM_RULE_SIZE[0] / count_x)
    y_centers = (np.arange(count_y) + 0.5) * (SUM_RULE_SIZE[1] / count_y)
    x_grid, y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
    weight = np.exp(-((x_grid - GAUSSIAN_CENTRE[0]) ** 2 + (y_grid - GAUSSIAN_CENTRE[1]) ** 2) / GAUSSIAN_RADIUS ** 2)
    x0, y0, x1, y1 = CRYSTAL_BOX
    in_crystal = (x_grid >= x0) & (x_grid < x1) & (y_grid >= y0) & (y_grid < y1)
    capacity = np.where(in_crystal, _capacity(NIOBATE_LIKE), _capacity(WATER_LIKE))
    cell_area = (SUM_RULE_SIZE[0] / count_x) * (SUM_RULE_SIZE[1] / count_y)
    return K_B * REFERENCE_TEMPERATURE ** 2 * INTERACTION_LENGTH * float(np.sum(weight ** 2 * cell_area / capacity))


def _sum_rule_runs():
    size_x, size_y = SUM_RULE_SIZE
    x0, y0, x1, y1 = CRYSTAL_BOX
    return {RUN_NAME: ModelSpec(
        title='Thermal noise sum rule across an interface',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('crystal', 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})],
        materials=[MaterialSpec(SUM_RULE_MATERIALS[0], properties=dict(WATER_LIKE), selection=['all']),
                   MaterialSpec(SUM_RULE_MATERIALS[1], properties=dict(NIOBATE_LIKE), selection=['crystal'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, settings=dict(HEAT_SETTINGS), features=[
            FeatureSpec('heat_sink', 'temperature', ['y_lo'], {'temperature': REFERENCE_TEMPERATURE}),
            _noise_feature(_sum_rule_weight_expression(), SUM_RULE_WAVENUMBER)])],
        mesh=MeshSpec(mode='uniform', dx=SUM_RULE_CELL),
        study=_noise_study(SUM_RULE_BAND))}


def _band_and_tails(psd, frequencies):
    """(trapezoid integral of psd f in ln f, flat low tail psd[0] f[0], 1/f^2 high tail psd[-1] f[-1])."""
    integrand = psd * frequencies
    band_integral = float(np.sum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(np.log(frequencies))))
    return band_integral, float(psd[0] * frequencies[0]), float(psd[-1] * frequencies[-1])


def _check_sum_rule(results_by_name):
    """Trapezoid integral of psd f in ln f plus the flat low tail psd[0] f[0] and the 1/f^2 high tail psd[-1] f[-1]
    against the result's 'variance', and that 'variance' against the cell sum; each material's psd_<name> against its
    contribution variance and psd_cross against 0 (relative to variance); max relative error."""
    result, frequencies, psd = _noise_spectrum(results_by_name[RUN_NAME])
    if frequencies.size < 2 or psd.shape != frequencies.shape:
        return {'measured': math.nan, 'expected': float(result['variance']), 'error': math.inf,
                'tolerance': SUM_RULE_TOLERANCE, 'details': {'n_frequencies': int(frequencies.size)}}
    band_integral, low_tail, high_tail = _band_and_tails(psd, frequencies)
    integrated = band_integral + low_tail + high_tail
    variance = float(result['variance'])
    expected_variance = _sum_rule_expected_variance()
    errors = {'sum_rule': abs(integrated - variance) / variance,
              'variance': abs(variance - expected_variance) / expected_variance}
    contribution_variances = result.get('contribution_variances') or {}
    if sorted(contribution_variances) != sorted(SUM_RULE_MATERIALS) or 'psd_cross' not in result['series']:
        errors['contributions'] = math.inf   # both materials lie under the selection: one self term each
    else:
        for name, contribution_variance in contribution_variances.items():
            part_integral = sum(_band_and_tails(np.asarray(result['series'][f'psd_{name}'], dtype=np.float64),
                                                frequencies))
            errors[f'sum_rule_{name}'] = abs(part_integral - contribution_variance) / contribution_variance
        errors['variance_split'] = abs(sum(contribution_variances.values()) - variance) / variance
        errors['cross_integral'] = abs(sum(_band_and_tails(
            np.asarray(result['series']['psd_cross'], dtype=np.float64), frequencies))) / variance
    slowest, fastest = _relaxation_band()
    return {'measured': integrated, 'expected': variance, 'error': max(errors.values()),
            'tolerance': SUM_RULE_TOLERANCE,
            'details': {'errors': errors, 'band_integral': band_integral, 'low_tail': low_tail,
                        'high_tail': high_tail, 'expected_variance': expected_variance,
                        'band_fraction': float(result['band_fraction']),
                        'relaxation_estimate_Hz': {'slowest': slowest, 'fastest': fastest},
                        'band_Hz': [float(frequencies[0]), float(frequencies[-1])]}}


_SLOWEST, _FASTEST = _relaxation_band()


# heat_noise_pec_box_mode: an em_modes pec box a x b (b > a) filled with an n = 1.5 silica-like solid; its discrete TE01
# mode is E_x = E0 sin(pi y / b) at the cell centres (y from the wall plane on the inner face of the outermost cells,
# uniform in x), beta^2 = (n k0)^2 - (2/dy)^2 sin^2(pi / (2 N_y)) and 1 W gives E0^2 = 4 omega mu0 / (beta a b)
# (H_y = beta E_x / (omega mu0) holds on the Yee points). Forward self-scattering (phase noise, q0 = 0): w = W0 sin^2 =
# W0 / 2 - W0 / 2 cos(2 pi y / b), W0 = omega eps0 n dndT E0^2 / 2. The ghost ring outside the wall planes has no heat
# material, so heat is the insulated box: the constant part is its null mode at q = 0 (static_variance s L W0^2 a b /
# (4 C)), the cell-centred cosine an exact eigenvector of the insulated FV operator (rate lambda = (k / C) (2/dy)^2
# sin^2(pi / N_y)): S = 4 s L (W0^2 a b / 8) / C lambda / (omega^2 + lambda^2), variance s L (3/8) W0^2 a b / C, s =
# k_B T0^2, C = rho cp. The half run keeps y in [b/2, b] with a symmetric (E even) mirror on y_lo: the plane is the
# window edge (the em ghost row is padded outside the window), so heat's insulated y_lo face is the even mirror and
# the result doubles.
BOX_CELL = 5e-8                    # m, uniform mesh
BOX_CELLS = (20, 40)               # em box a x b in cells, inside the one-cell ghost ring
BOX_WIDTH = BOX_CELLS[0] * BOX_CELL    # m, a
BOX_HEIGHT = BOX_CELLS[1] * BOX_CELL   # m, b
BOX_PERMITTIVITY = 2.25            # n = 1.5, exact in float32
BOX_DNDT = 1e-5                    # 1/K
BOX_WAVELENGTH = 1.55e-6           # m
SILICA_LIKE = {'k': 1.4, 'rho': 2200.0, 'cp': 740.0}
BOX_BAND = (1e2, 1e11, 28)         # Hz, 3 per decade (relaxation ~1.4 MHz)
BOX_TOLERANCE = 1e-4               # exact discrete closed form: ARPACK / SuperLU round-off, trapezoid in ln f ~3e-6
BOX_MIRROR_FACTORS = {'full': 1.0, 'half': 2.0}
BOX_CPU_SUFFIX = '_cpu'            # run name suffix of the cpu-backend copies (the plain names run on 'auto' = gpu)


def _pec_box_spec(run_name):
    """em_modes pec box ('half': y_lo symmetric mirror through the box centre) + heat with the forward phase-noise
    weighting of the fundamental quasi-TE mode on the box material."""
    half = run_name == 'half'
    rows = BOX_CELLS[1] // 2 if half else BOX_CELLS[1]
    settings = {f'bc_{face}': 'pec' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    if half:
        settings['bc_y_lo'] = 'symmetric'
    settings.update(wavelength=BOX_WAVELENGTH, n_modes=3)
    material = dict(SILICA_LIKE, eps_r=BOX_PERMITTIVITY, eps_xx=BOX_PERMITTIVITY, eps_yy=BOX_PERMITTIVITY,
                    dndT=BOX_DNDT)
    lo_ghost_rows = 0 if half else 1   # half: the mirror plane is the window edge y = 0
    core = {'x0': BOX_CELL, 'y0': lo_ghost_rows * BOX_CELL,
            'x1': (BOX_CELLS[0] + 1) * BOX_CELL, 'y1': (rows + lo_ghost_rows) * BOX_CELL}
    return ModelSpec(
        title=f'Thermal noise of the TE01 mode of a pec box ({run_name})',
        domain=DomainSpec(size_x=(BOX_CELLS[0] + 2) * BOX_CELL, size_y=(rows + lo_ghost_rows + 1) * BOX_CELL),
        geometry=[GeometrySpec('core', 'rectangle', core)],
        materials=[MaterialSpec('box_material', properties=material, selection=['core'])],
        physics=[PhysicsSpec(MODE_PHYSICS_NAME, MODE_TYPE_NAME, settings=settings),
                 PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, settings=dict(HEAT_SETTINGS), features=[
                     FeatureSpec(NOISE_FEATURE_NAME, NOISE_FEATURE_TYPE, ['core'], {
                         'weighting': 'mode_pair', 'mode_physics': MODE_PHYSICS_NAME,
                         'incident_selection': 'collected', 'incident_direction': 'forward',
                         'collected_selection': 'te', 'collected_direction': 'forward',
                         'axial_profile': 'uniform', 'interaction_length': INTERACTION_LENGTH})])],
        mesh=MeshSpec(mode='uniform', dx=BOX_CELL),
        study=_noise_study(BOX_BAND))


def _pec_box_runs():
    """'full' / 'half' on the default backend and their cpu-backend copies 'full_cpu' / 'half_cpu'."""
    runs = {}
    for run_name in BOX_MIRROR_FACTORS:
        runs[run_name] = _pec_box_spec(run_name)
        runs[run_name + BOX_CPU_SUFFIX] = _pec_box_spec(run_name)
        runs[run_name + BOX_CPU_SUFFIX].solver_defaults.backend = 'cpu'
    return runs


def _pec_box_closed_form():
    """Discrete closed form: n_eff of TE01, relaxation rate [1/s], variance, static variance and PSD scale (L
    included) of the forward phase noise."""
    omega = 2.0 * math.pi * C0 / BOX_WAVELENGTH
    k0 = omega / C0
    index = math.sqrt(_stored(BOX_PERMITTIVITY))
    transverse = (2.0 / BOX_CELL) ** 2 * math.sin(math.pi / (2.0 * BOX_CELLS[1])) ** 2
    beta = math.sqrt((index * k0) ** 2 - transverse)
    field_squared = 4.0 * omega / (EPS0 * C0 ** 2) / (beta * BOX_WIDTH * BOX_HEIGHT)   # E0^2, mu0 = 1 / (eps0 c^2)
    amplitude = omega * EPS0 * index * _stored(BOX_DNDT) * field_squared / 2.0          # W0 [1/(K m^3)]
    capacity = _capacity(SILICA_LIKE)
    rate = (_stored(SILICA_LIKE['k']) / capacity * (2.0 / BOX_CELL) ** 2
            * math.sin(math.pi / BOX_CELLS[1]) ** 2)
    scale = K_B * REFERENCE_TEMPERATURE ** 2 * INTERACTION_LENGTH * amplitude ** 2 * BOX_WIDTH * BOX_HEIGHT / capacity
    return {'n_eff': beta / k0, 'rate': rate, 'variance': 0.375 * scale, 'static_variance': 0.25 * scale,
            'psd_scale': 0.5 * scale}


_BOX_FORM = _pec_box_closed_form()


def _check_pec_box(results_by_name):
    """Per run that ran (the default-backend runs are skipped without a GPU): max relative PSD error against the
    Lorentzian, relative variance / static variance / n_eff errors, absolute band_fraction error (in-band share of the
    Lorentzian) and mirror factor; with both full runs max |psd_cpu - psd_gpu| / max psd; error = the largest."""
    form = _BOX_FORM
    requested = _log_band(*BOX_BAND)
    errors, measured = {}, math.nan
    run_names = [name + suffix for name in BOX_MIRROR_FACTORS for suffix in ('', BOX_CPU_SUFFIX)]
    for run_name in [name for name in run_names if name in results_by_name]:
        mirror_factor = BOX_MIRROR_FACTORS[run_name.removesuffix(BOX_CPU_SUFFIX)]
        result, frequencies, psd = _noise_spectrum(results_by_name[run_name])
        if frequencies.shape != requested.shape or psd.shape != requested.shape:
            errors[f'{run_name}_frequencies'] = math.inf
            continue
        omega = 2.0 * math.pi * frequencies
        expected_psd = form['psd_scale'] * form['rate'] / (omega ** 2 + form['rate'] ** 2)
        expected_band = 2.0 / math.pi * (math.atan(omega[-1] / form['rate']) - math.atan(omega[0] / form['rate']))
        if run_name.startswith('full') and not math.isfinite(measured):
            measured = float(psd[0])
        for key in ('variance', 'static_variance'):
            errors[f'{run_name}_{key}'] = abs(float(result.get(key, math.nan)) - form[key]) / form[key]
        errors.update({
            f'{run_name}_psd': float(np.max(np.abs(psd - expected_psd) / expected_psd)),
            f'{run_name}_band_fraction': abs(float(result.get('band_fraction', math.nan)) - expected_band),
            f'{run_name}_n_eff': abs(float(result.get('collected_n_eff', math.nan)) - form['n_eff']) / form['n_eff'],
            f'{run_name}_mirror_factor': abs(float(result.get('mirror_factor', math.nan)) - mirror_factor),
            f'{run_name}_frequencies': float(np.max(np.abs(frequencies - requested) / requested))})
    full_cpu = 'full' + BOX_CPU_SUFFIX
    compared = 'full' in results_by_name and full_cpu in results_by_name
    if compared:
        _, _, gpu_psd = _noise_spectrum(results_by_name['full'])
        _, _, cpu_psd = _noise_spectrum(results_by_name[full_cpu])
        errors['cpu_vs_gpu_psd'] = (float(np.max(np.abs(cpu_psd - gpu_psd))) / float(np.max(np.abs(gpu_psd)))
                                    if cpu_psd.shape == gpu_psd.shape else math.inf)
    error = float(np.max(np.asarray(list(errors.values()), dtype=np.float64)))   # NaN propagates
    return {'measured': measured,
            'expected': float(form['psd_scale'] * form['rate'] / ((2.0 * math.pi * BOX_BAND[0]) ** 2
                                                                  + form['rate'] ** 2)),
            'error': error, 'tolerance': BOX_TOLERANCE,
            'details': {'errors': errors, 'expected_n_eff': form['n_eff'],
                        'relaxation_frequency_Hz': form['rate'] / (2.0 * math.pi),
                        'expected_variance': form['variance'], 'expected_static_variance': form['static_variance'],
                        'gpu_runs': 'compared' if compared else 'skipped (no GPU)'}}


# waveguide_thermal_noise template: a 480 x 220 nm silicon strip in silica at 1550 nm, half domain (x_lo antisymmetric
# mirror through the strip centre: quasi-TE family), em CPML on the other faces, substrate sink on y_lo; the
# thermorefractive backscatter of the quasi-TE mode into itself over 1 mm, split into silica / silicon terms.
WAVEGUIDE_CELL = 2e-8              # m, uniform (the x_lo mirror plane is the window edge x = 0)
WAVEGUIDE_STRIP = (4.8e-7, 2.2e-7)   # m, strip width, height
WAVEGUIDE_BOX = 2e-6               # m, silica below the strip (window; continues into the em PML)
WAVEGUIDE_TOP = 1.78e-6            # m, silica above the strip
WAVEGUIDE_LATERAL = 2e-6           # m, window from the mirror plane to x_hi
WAVEGUIDE_LENGTH = 1e-3            # m, uniform interaction length
WAVEGUIDE_WAVELENGTH = 1.55e-6     # m
WAVEGUIDE_BAND = (1e3, 1e10, 29)   # Hz, 4 per decade
SINK_TEMPERATURE = 293.15          # K, y_lo and T0
WAVEGUIDE_TEMPLATE_SUMMARY = (
    'Thermal noise of a silicon strip waveguide: the light that temperature fluctuations scatter back into the '
    'guided mode, as a spectrum over frequency, split into the silica and silicon parts.')
WAVEGUIDE_TEMPLATE_DESCRIPTION = (
    f'thermal_noise study of a {WAVEGUIDE_STRIP[0] * 1e9:g} x {WAVEGUIDE_STRIP[1] * 1e9:g} nm silicon strip in silica '
    f'(library materials) at {WAVEGUIDE_WAVELENGTH * 1e9:g} nm: em_modes on the half cross-section (x_lo '
    'antisymmetric mirror through the strip centre, quasi-TE family; em CPML on the other faces) and heat with the '
    f'substrate sink T = {SINK_TEMPERATURE:g} K on y_lo ({WAVEGUIDE_BOX * 1e6:g} um below the strip), other faces '
    'insulated. One thermorefractive_noise feature on all cells (mode_pair: the quasi-TE mode backscattered into '
    f'itself, q0 = 2 beta, uniform over {WAVEGUIDE_LENGTH * 1e3:g} mm): psd in 1/Hz (backscattered power per incident '
    'power per Hz) with the silica and silicon self terms and their cross term. '
    f'{WAVEGUIDE_BAND[2]} log-spaced frequencies {WAVEGUIDE_BAND[0]:g}..{WAVEGUIDE_BAND[1]:g} Hz, uniform '
    f'{WAVEGUIDE_CELL * 1e9:g} nm mesh; derived values n_eff_1 and the PSD at 1 MHz.')


def waveguide_thermal_noise_spec():
    """'waveguide_thermal_noise' template: half silicon strip in silica, em_modes + heat thermal_noise backscatter."""
    strip_width, strip_height = WAVEGUIDE_STRIP
    size_y = WAVEGUIDE_BOX + strip_height + WAVEGUIDE_TOP
    return ModelSpec(
        title='Silicon strip waveguide: thermorefractive backscatter noise',
        description=WAVEGUIDE_TEMPLATE_DESCRIPTION,
        domain=DomainSpec(size_x=WAVEGUIDE_LATERAL, size_y=size_y),
        geometry=[GeometrySpec('strip', 'rectangle', {'x0': 0.0, 'y0': WAVEGUIDE_BOX, 'x1': strip_width / 2.0,
                                                      'y1': WAVEGUIDE_BOX + strip_height})],
        materials=[MaterialSpec('silica', library='silica', selection=['all']),
                   MaterialSpec('silicon', library='silicon', selection=['strip'])],
        physics=[PhysicsSpec(MODE_PHYSICS_NAME, MODE_TYPE_NAME, settings={
                     'bc_x_lo': 'antisymmetric', 'bc_x_hi': 'cpml', 'bc_y_lo': 'cpml', 'bc_y_hi': 'cpml',
                     'wavelength': WAVEGUIDE_WAVELENGTH, 'n_modes': 2}),
                 PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, settings={'T0': SINK_TEMPERATURE}, features=[
                     FeatureSpec('substrate', 'temperature', ['y_lo'], {'temperature': SINK_TEMPERATURE}),
                     FeatureSpec('backscatter', NOISE_FEATURE_TYPE, ['all'], {
                         'weighting': 'mode_pair', 'mode_physics': MODE_PHYSICS_NAME,
                         'incident_selection': 'collected', 'incident_direction': 'forward',
                         'collected_selection': 'te', 'collected_direction': 'backward',
                         'axial_profile': 'uniform', 'interaction_length': WAVEGUIDE_LENGTH})])],
        mesh=MeshSpec(mode='uniform', dx=WAVEGUIDE_CELL),
        study=_noise_study(WAVEGUIDE_BAND),
        derived_values=[DerivedValueSpec('n_eff_1', f'{MODE_PHYSICS_NAME}_n_eff_1'),   # scalar output (E5)
                        DerivedValueSpec('S_1MHz', 'psd', operation='at', source=f'{PHYSICS_NAME}.backscatter',
                                         at=1e6)])


# heat_noise_line_block: the waveguide_thermal_noise template (fewer frequencies) with a second, identical heat physics
# LINE_BLOCK_PHYSICS_NAME in the same run: 'heat' keeps the reference host SuperLU path (Automatic shifted solver:
# sparse_direct), LINE_BLOCK_PHYSICS_NAME takes a line_block solver node (mixed precision) as its shifted solver; both
# weight the one em_modes solve. Every linear psd series must agree and no shift may be rejected (re-solved by SuperLU).
LINE_BLOCK_PHYSICS_NAME = 'heat_line_block'
LINE_BLOCK_SOLVER_NAME = 'noise_line_block'
LINE_BLOCK_BAND = (1e3, 1e10, 8)   # Hz, 1 per decade: the check is pointwise per frequency
WAVEGUIDE_FEATURE_NAME = 'backscatter'
LINE_BLOCK_TOLERANCE = SUM_RULE_TOLERANCE      # solver_plan Part II stage 1; refinement reaches ~1e-9 residuals


def _line_block_runs():
    spec = waveguide_thermal_noise_spec()
    spec.study = _noise_study(LINE_BLOCK_BAND)
    line_block_heat = copy.deepcopy(spec.physics_by_name(PHYSICS_NAME))
    line_block_heat.name = LINE_BLOCK_PHYSICS_NAME
    line_block_heat.solvers['shifted'] = LINE_BLOCK_SOLVER_NAME
    spec.physics.append(line_block_heat)
    spec.solvers.append(SolverSpec(LINE_BLOCK_SOLVER_NAME, 'line_block', {'precision': 'mixed'}))
    return {RUN_NAME: spec}


def _check_line_block(results_by_name):
    """Per linear series the max |line_block - sparse_direct| relative to max |psd| of the reference (the cross term
    crosses 0), plus the rejected shift count of the line-block physics (missing record: inf); error = the largest."""
    results = results_by_name[RUN_NAME]
    reference = results.result(PHYSICS_NAME, WAVEGUIDE_FEATURE_NAME)
    batched = results.result(LINE_BLOCK_PHYSICS_NAME, WAVEGUIDE_FEATURE_NAME)
    reference_psd = np.asarray(reference['series']['psd'], dtype=np.float64)
    psd_scale = float(np.max(np.abs(reference_psd)))
    errors = {}
    for name, values in reference['series'].items():
        if name.endswith('_db'):
            continue
        values = np.asarray(values, dtype=np.float64)
        other = np.asarray(batched['series'].get(name, []), dtype=np.float64)
        errors[name] = (float(np.max(np.abs(other - values))) / psd_scale if other.shape == values.shape
                        else math.inf)   # NaN propagates
    solvers = results.summary().get('solvers') or {}
    reference_type = (((solvers.get(PHYSICS_NAME) or {}).get('roles') or {}).get('shifted') or {}).get('type')
    shifted = ((solvers.get(LINE_BLOCK_PHYSICS_NAME) or {}).get('roles') or {}).get('shifted') or {}
    work = shifted.get('counters') or {}
    errors['rejected_shifts'] = (float(work.get('rejected_shifts', math.inf))
                                 if shifted.get('type') == 'line_block' and reference_type == 'sparse_direct'
                                 else math.inf)
    error = float(np.max(np.asarray(list(errors.values()), dtype=np.float64)))
    return {'measured': float(np.asarray(batched['series']['psd'], dtype=np.float64)[0]),
            'expected': float(reference_psd[0]), 'error': error, 'tolerance': LINE_BLOCK_TOLERANCE,
            'details': {'errors': errors, 'line_block_work': dict(work), 'reference_shifted_type': reference_type}}


THERMAL_NOISE_SELFTEST_CASES = (
    SelfTestCase(
        name='heat_noise_cosine_lorentzian', label='Heat: thermal noise of a cosine eigenmode',
        description=(f'thermal_noise study of an insulated {COSINE_SIZE[0] * 1e6:g} x {COSINE_SIZE[1] * 1e6:g} um '
                     f'rectangle (k = {WATER_LIKE["k"]:g} W/(m*K), rho = {WATER_LIKE["rho"]:g} kg/m^3, cp = '
                     f'{WATER_LIKE["cp"]:g} J/(kg*K), uniform {COSINE_CELL * 1e9:g} nm cells, N = {COSINE_N_X} along '
                     f'x, T0 = {REFERENCE_TEMPERATURE:g} K) with a thermorefractive_noise feature of weight w = '
                     f'cos(pi x / Lx), axial q = {COSINE_WAVENUMBER:g} 1/m, uniform axial length L = '
                     f'{INTERACTION_LENGTH:g} m, {COSINE_BAND[2]} log-spaced frequencies {COSINE_BAND[0]:g}..'
                     f'{COSINE_BAND[1]:g} Hz. The cell-centred cosine is an exact eigenvector of the insulated '
                     'finite-volume operator (eigenvalue k (2/dx)^2 sin^2(pi / (2 N)) per unit area), so exactly S(f) '
                     '= 4 k_B T0^2 L (Lx Ly / 2) K_m / ((2 pi f rho cp)^2 + K_m^2), K_m = k ((2/dx)^2 sin^2(pi / (2 '
                     f'N)) + q^2) (corner {COSINE_CORNER:.3g} Hz), and the variance k_B T0^2 L (Lx Ly / 2) / (rho cp). '
                     'error = max(max relative PSD error, relative variance error, relative frequency error); the '
                     'expected values use the float32-stored k, rho, cp. Tolerance '
                     f'{COSINE_TOLERANCE:g}: float32 weights, cell areas and face conductances give ~1e-7, the host '
                     'sparse solve ~1e-12. A wrong capacity, k_zz q^2 term, T0^2 or 4 k_B factor, cell area or '
                     'one-sided convention fails by far more.'),
        specs=_cosine_lorentzian_runs, check=_check_cosine_lorentzian),
    SelfTestCase(
        name='heat_noise_sum_rule', label='Heat: thermal noise sum rule across an interface',
        description=(f'thermal_noise study of a {SUM_RULE_SIZE[0] * 1e6:g} x {SUM_RULE_SIZE[1] * 1e6:g} um water-like '
                     f'background (k {WATER_LIKE["k"]:g}, rho {WATER_LIKE["rho"]:g}, cp {WATER_LIKE["cp"]:g}) with a '
                     f'niobate-like block (k {NIOBATE_LIKE["k"]:g}, rho {NIOBATE_LIKE["rho"]:g}, cp '
                     f'{NIOBATE_LIKE["cp"]:g}) at x {CRYSTAL_BOX[0] * 1e6:g}..{CRYSTAL_BOX[2] * 1e6:g} um, y '
                     f'{CRYSTAL_BOX[1] * 1e6:g}..{CRYSTAL_BOX[3] * 1e6:g} um, temperature held on y_lo, uniform '
                     f'{SUM_RULE_CELL * 1e9:g} nm cells, a Gaussian weight w = exp(-r^2 / s^2), s = '
                     f'{GAUSSIAN_RADIUS * 1e6:g} um, centred on the block x_lo edge, q = {SUM_RULE_WAVENUMBER:g} 1/m, '
                     f'L = {INTERACTION_LENGTH:g} m, {SUM_RULE_BAND[2]} log-spaced frequencies {SUM_RULE_BAND[0]:g}..'
                     f'{SUM_RULE_BAND[1]:g} Hz (relaxation rates between ~{_SLOWEST:.2g} Hz, a strict lower bound '
                     f'min D q^2 / (2 pi), and ~{_FASTEST:.2g} Hz, max D (8 / dx^2 + q^2) / (2 pi)). '
                     'Fluctuation-dissipation sum rule: integral_0^inf S(f) df = k_B T0^2 L sum w^2 A / (rho cp) for '
                     'any symmetric conduction operator, interfaces and Dirichlet faces included. measured = '
                     'trapezoid integral of S f in ln f + S(f_first) f_first (flat low tail) + S(f_last) f_last (1/f^2 '
                     "high tail); expected = the result's variance, which is itself compared with the cell sum "
                     'computed here. The same identity holds for the self term of each material (psd_<material> '
                     'against its contribution variance; the two variances sum to the total) and the integral of the '
                     'interference term psd_cross is 0 (equal-time temperature noise is uncorrelated between cells), '
                     'checked relative to the total variance; error = the largest relative error. Tolerance '
                     f'{SUM_RULE_TOLERANCE:g}: S f is a positive sum of sech-shaped bumps in ln f, so the trapezoid '
                     'error is exponentially small at 4 points per decade (~4e-8), and the tail models miss by ~3e-6 '
                     '(low, the band ~2.8 decades below the slowest estimate) and ~6e-5 (high, ~2 decades above the '
                     'fastest). A mis-scaled capacity, a non-symmetric interface conductance or a wrong frequency / '
                     f'omega factor breaks the identity. Not yet run at {SUM_RULE_BAND[2]} frequencies.'),
        specs=_sum_rule_runs, check=_check_sum_rule),
    SelfTestCase(
        name='heat_noise_pec_box_mode', label='Heat: thermal noise of the TE01 mode of a pec box',
        description=(f'thermal_noise study of an em_modes pec box a x b = {BOX_WIDTH * 1e6:g} x {BOX_HEIGHT * 1e6:g} um '
                     f'({BOX_CELLS[0]} x {BOX_CELLS[1]} uniform {BOX_CELL * 1e9:g} nm cells inside the one-cell ghost '
                     f'ring of the wall planes) filled with n = {math.sqrt(BOX_PERMITTIVITY):g}, dndT = {BOX_DNDT:g} '
                     f'1/K, k = {SILICA_LIKE["k"]:g} W/(m*K), rho = {SILICA_LIKE["rho"]:g} kg/m^3, cp = '
                     f'{SILICA_LIKE["cp"]:g} J/(kg*K) (no heat material on the ghost ring, so heat is the insulated '
                     f'box), lambda = {BOX_WAVELENGTH * 1e9:g} nm, T0 = {REFERENCE_TEMPERATURE:g} K; a '
                     'thermorefractive_noise mode_pair feature on the box: the fundamental quasi-TE mode (TE01, E_x = '
                     'E0 sin(pi y / b)) forward-scattered into itself (phase noise, q0 = 0), uniform over L = '
                     f'{INTERACTION_LENGTH:g} m, {BOX_BAND[2]} log-spaced frequencies {BOX_BAND[0]:g}..{BOX_BAND[1]:g} '
                     'Hz. Exact discrete closed form: beta^2 = (n k0)^2 - (2/dy)^2 sin^2(pi / (2 N_y)), 1 W E0^2 = 4 '
                     'omega mu0 / (beta a b), w = W0 sin^2(pi y / b) with W0 = omega eps0 n dndT E0^2 / 2 = a constant '
                     '(the null mode of the insulated box at q = 0: static_variance s L W0^2 a b / (4 rho cp)) plus a '
                     'cell-centred cosine (exact FV eigenvector, rate lambda = k / (rho cp) (2/dy)^2 sin^2(pi / N_y)): '
                     'S = 4 s L (W0^2 a b / 8) / (rho cp) lambda / (omega^2 + lambda^2), variance s L (3/8) W0^2 a b / '
                     '(rho cp), s = k_B T0^2, and band_fraction = the in-band share of that Lorentzian. A second run '
                     'solves the half y >= b/2 with a symmetric (E even) mirror on y_lo, the plane on the window edge '
                     '(its em ghost row is padded outside the window, so the insulated heat face is the even mirror), '
                     'and mirror_factor 2: it must give the same numbers. Both runs go on the default (gpu) backend '
                     f'and as copies *{BOX_CPU_SUFFIX} on solver_defaults.backend cpu (numpy MaterialGrid and mode '
                     'fields; the noise operator, SuperLU and ARPACK are host code on both backends). error = the '
                     'largest of, per run, the max relative PSD error, the relative variance, static_variance and '
                     'n_eff errors, the absolute band_fraction and mirror_factor errors, and max |psd_cpu - psd_gpu| / '
                     f'max psd of the full box; tolerance {BOX_TOLERANCE:g} (round-off of the eigen / sparse solves and '
                     'float32 storage, exact via the stored values; trapezoid in ln f ~3e-6; the backends agree to '
                     'round-off). Without a usable GPU the gpu runs are skipped and the cpu runs carry the closed '
                     'form. A wrong 1 W normalisation, mode pair factor (2 n dndT, omega eps0 / 4, partner '
                     'orientation), mirror factor, a mirror plane one cell off the heat face (~10 % in lambda) or a '
                     'missing static share (band_fraction 1/3) fails by far more. Not yet run at '
                     f'{BOX_BAND[2]} frequencies with the cpu copies.'),
        specs=_pec_box_runs, check=_check_pec_box),
    SelfTestCase(
        name='heat_noise_line_block', label='Heat: thermal noise by the line-block GPU solver',
        description=('The waveguide_thermal_noise template (half silicon strip in silica, quasi-TE backscatter, '
                     'contributions silica / silicon) at '
                     f'{LINE_BLOCK_BAND[2]} frequencies {LINE_BLOCK_BAND[0]:g}..{LINE_BLOCK_BAND[1]:g} Hz (the '
                     'template has more; the check is pointwise per frequency) with a second, identical heat physics '
                     f'{LINE_BLOCK_PHYSICS_NAME!r} in the same run, both weighting the one em_modes solve: heat keeps '
                     "the shifted solver Automatic = 'sparse_direct' (one host SuperLU factorisation per frequency, "
                     f"the reference), {LINE_BLOCK_PHYSICS_NAME!r} takes a 'line_block' solver node (mixed "
                     'precision: all frequencies of the channel factored in GPU batches as a block-tridiagonal '
                     'system, equilibrated, refined against the float64 operator). error = the largest of, per '
                     'linear series '
                     '(psd, psd_silica, psd_silicon, psd_cross), max |line_block - sparse_direct| / max |psd| of the '
                     'reference, and the number of shifts the line-block solver rejected (re-solved by SuperLU; must '
                     f'be 0 on this anchored, well-conditioned template). Tolerance {LINE_BLOCK_TOLERANCE:g} (that of '
                     'heat_noise_sum_rule); refined solves agree to ~1e-8. A wrong block ordering, equilibration, '
                     'shift term or rhs layout fails by far more; a fallback on every shift (misplaced acceptance '
                     'tolerance) fails through rejected_shifts. Not yet run as one two-heat-physics run at '
                     f'{LINE_BLOCK_BAND[2]} frequencies.'),
        specs=_line_block_runs, check=_check_line_block, heavy=True),
)
