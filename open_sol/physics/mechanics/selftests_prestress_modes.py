"""Self-tests of the solid prestress, damped-mode, frequency-probe and solved-physics wiring (new_modules_plan K2a, K2,
K3, K4, K0 free-body response): model runs on triangles and tetrahedra against closed forms (roller boxes under a
uniform initial stress, a released initial stress, the diluted Q of a lossy prestressed box, a driven lossy bar's complex
probe and reaction, a free bar's response) and the StudySpec.solved_physics selection; definition / definition3d
register them."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cpu_defaults, float32_value, last_value, scaled_worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests_fem import _BOX, _BOX_DENSITY, _BOX_H, _BOX_ORTHOTROPIC, _N_MODES, _box_frequencies
from .selftests_fem3d import (_MODES_CELL, _MODES_DENSITY, _MODES_ELASTIC, _MODES_LOWEST, _MODES_SIZE,
                              _MODES_TOLERANCE, _isotropic_constants, _point, _probes, _solid_spec)

_SOLID = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME
_TRIANGLE_MESH = 'triangles'
_TABLE = 'modes'
_ROLLERS_2D = ['x_lo', 'x_hi', 'y_lo', 'y_hi']


def _point_2d(name, x, y):
    return GeometrySpec(name, 'point', {'x': x, 'y': y})


def _triangle_model(title, size, h, physics, study, material, geometry=()):
    """2D model on a uniform triangle mesh (cpu backend) with one material on 'all'."""
    return ModelSpec(title=title, domain=DomainSpec(size_x=size[0], size_y=size[1]), geometry=list(geometry),
                     materials=[MaterialSpec('block', properties=material, selection=['all'])],
                     physics=list(physics),
                     meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=h)],
                     solver_defaults=cpu_defaults(), study=study)


# ---- solid_fem_prestressed_roller_box (K2, K3): the selftests_fem roller box (10 x 4 mm, P2, h 0.25 mm, orthotropic
# c11 / c12 / c22 / c66) under a uniform initial stress sigma0_xx = S. Rollers on all four faces hold every normal
# displacement, so the static state is u = 0 (the initial-stress force of a uniform sigma0 is -(its boundary
# tractions): normal ones on the held x faces only) and the total stress is sigma0. K_sigma = S int dN/dx dN/dx per
# component; the roller modes u_x = A sin(kx x) cos(ky y), u_y = B cos(kx x) sin(ky y) stay eigenfunctions of -S d2/dx2
# (S kx^2 for sin and cos alike, natural conditions met), so the 2 x 2 symbol per (p, q) gains S kx^2 I: rho omega^2 =
# mu_i + S kx^2. Lossy run (mechanical_loss_factor eta, K_sigma lossless): rho omega^2 = mu_i (1 + j eta) + S kx^2, so
# Q = Re omega / (2 Im omega) is diluted by the prestress. 'release': sigma0_xx = S with rollers on x_lo / y_lo only,
# plane strain: the total stress relaxes to 0, C (eps_xx, eps_yy) = (-S, 0) (exact linear field in P2).
_PRESTRESS_2D = 6e9            # Pa (float32-exact)
_LOSS_FACTOR = 1e-2
_FREQUENCY_TOLERANCE_2D = 1e-4   # relative frequency (P2 eigenvalues converge as h^4; the unstressed box passes at it)
_Q_TOLERANCE = 1e-3              # relative Q (first estimate)
_RELEASE_POINT_2D = (0.6 * _BOX[0], 0.55 * _BOX[1])
_EXACT_TOLERANCE = 1e-5          # float32 probe samples of exact fields


def _prestressed_box_modes(prestress, loss_factor=0.0):
    """[(frequency [Hz], Q)] of the lowest _N_MODES continuum modes of the prestressed (lossy) roller box, sorted by
    frequency (Q infinite without loss)."""
    c11, c12, c22, c66 = (_BOX_ORTHOTROPIC[name] for name in ('c11', 'c12', 'c22', 'c66'))
    modes = []
    for p in range(13):
        for q in range(7):
            kx, ky = p * math.pi / _BOX[0], q * math.pi / _BOX[1]
            present = [axis for axis, order in ((0, p), (1, q)) if order >= 1]
            if not present:
                continue
            symbol = np.array([[c11 * kx * kx + c66 * ky * ky, (c12 + c66) * kx * ky],
                               [(c12 + c66) * kx * ky, c22 * ky * ky + c66 * kx * kx]])
            for value in np.linalg.eigvalsh(symbol[np.ix_(present, present)]):
                omega = complex(np.sqrt(complex((value * (1.0 + 1j * loss_factor) + prestress * kx * kx)
                                                / _BOX_DENSITY)))
                q_factor = omega.real / (2.0 * omega.imag) if omega.imag else math.inf
                modes.append((omega.real / (2.0 * math.pi), q_factor))
    return sorted(modes)[:_N_MODES]


def _box_material(**extra):
    return {**_BOX_ORTHOTROPIC, 'rho': _BOX_DENSITY, 'initial_stress_xx': _PRESTRESS_2D, **extra}


def _prestressed_box_specs():
    modal = StudySpec(type='eigenfrequency', settings={'n_resonances': _N_MODES})
    features = [FeatureSpec('rollers', 'roller', list(_ROLLERS_2D)), FeatureSpec(_TABLE, 'resonance_table')]
    physics = [PhysicsSpec(_SOLID, _SOLID_TYPE, settings={'stress_stiffening': True}, features=features)]
    release_features = [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
                        *(FeatureSpec(f'{name}_p', 'solid_probe', ['p'], {'component': name})
                          for name in ('u_x', 'u_y', 'stress_xx'))]
    return {
        'prestressed': _triangle_model('Self-test: prestressed roller box on triangles', _BOX, _BOX_H, physics, modal,
                                       _box_material()),
        'lossy': _triangle_model('Self-test: lossy prestressed roller box on triangles', _BOX, _BOX_H, physics, modal,
                                 _box_material(mechanical_loss_factor=_LOSS_FACTOR)),
        'release': _triangle_model('Self-test: released initial stress on triangles', _BOX, _BOX_H,
                                   [PhysicsSpec(_SOLID, _SOLID_TYPE, features=release_features)],
                                   StudySpec(type='stationary'), _box_material(),
                                   geometry=[_point_2d('p', *_RELEASE_POINT_2D)])}


def _check_prestressed_box(results_by_name):
    """Prestressed / lossy tables against the shifted continuum (frequency, Q), the released probes against C eps =
    (-S, 0)."""
    errors, tolerances, details = {}, {}, {}
    for run, loss_factor in (('prestressed', 0.0), ('lossy', float32_value(_LOSS_FACTOR))):
        table = results_by_name[run].result(_SOLID, _TABLE)
        columns = table['columns']
        rows = sorted(table['rows'], key=lambda row: float(row[columns.index('frequency')]))
        expected = _prestressed_box_modes(_PRESTRESS_2D, loss_factor)
        measured = [float(row[columns.index('frequency')]) for row in rows]
        if len(measured) < len(expected):
            errors[f'{run} found'], tolerances[f'{run} found'] = math.inf, 1.0
            continue
        errors[f'{run} frequency'] = max(abs(value / reference - 1.0)
                                         for value, (reference, _) in zip(measured, expected))
        tolerances[f'{run} frequency'] = _FREQUENCY_TOLERANCE_2D
        details[run] = {'frequency_measured': measured, 'frequency_expected': [value for value, _ in expected]}
        if loss_factor:
            q_measured = [float(row[columns.index('q_factor')]) for row in rows]
            errors[f'{run} Q'] = max(abs(value / reference - 1.0)
                                     for value, (_, reference) in zip(q_measured, expected))
            tolerances[f'{run} Q'] = _Q_TOLERANCE
            details[run].update(q_measured=q_measured, q_expected=[value for _, value in expected],
                                q_unstressed=(1.0 + math.sqrt(1.0 + loss_factor ** 2)) / (2.0 * loss_factor))
    c11, c12, c22 = (float32_value(_BOX_ORTHOTROPIC[name]) for name in ('c11', 'c12', 'c22'))
    determinant = c11 * c22 - c12 * c12
    strain_xx, strain_yy = -_PRESTRESS_2D * c22 / determinant, _PRESTRESS_2D * c12 / determinant
    x, y = _RELEASE_POINT_2D
    release = results_by_name['release']
    scale = abs(strain_xx) * _BOX[0]
    errors['release u_x'] = abs(last_value(release, _SOLID, 'u_x_p') - strain_xx * x) / scale
    errors['release u_y'] = abs(last_value(release, _SOLID, 'u_y_p') - strain_yy * y) / scale
    errors['release stress_xx'] = abs(last_value(release, _SOLID, 'stress_xx_p')) / _PRESTRESS_2D
    tolerances.update({'release u_x': _EXACT_TOLERANCE, 'release u_y': _EXACT_TOLERANCE,
                       'release stress_xx': _EXACT_TOLERANCE})
    details['release'] = {'strain_xx': strain_xx, 'strain_yy': strain_yy}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem_frequency_probe (K4, K0 free-body response): a 10 x 1 mm bar on P2 triangles (h 0.25 mm), rollers on
# y_lo / y_hi (u_y = 0: one-dimensional plane-strain waves of modulus M = lambda + 2 mu), traction T on x_hi, loss eta:
# M* = M (1 + j eta), k = omega sqrt(rho / M*). 'fixed' (roller on x_lo): u = T sin(k x) / (M* k cos(k L)), reaction of
# the x_lo roller F_x = -T H / cos(k L) per m of depth. 'free' (no x hold: a free body, regular at f > 0): u = -T cos(k
# x) / (M* k sin(k L)). Frequencies below, near and above the first fixed-free resonance c / (4 L) ~ 194 kHz.
_BAR = (10e-3, 1e-3)
_BAR_H = 0.25e-3
_BAR_MATERIAL = (100e9, 0.25, 2000.0)   # E [Pa], nu, rho [kg/m^3]
_BAR_TRACTION = 1e6                     # Pa on x_hi
_BAR_FREQUENCIES = [50e3, 150e3, 250e3]
_BAR_POINT = (0.75 * _BAR[0], 0.5 * _BAR[1])
_PROBE_TOLERANCE = 1e-5                 # relative to |u| (P2 at k h <= 0.05; first estimate)


def _bar_wave():
    """(complex modulus M*, wavenumber function k(f)) of the stored (float32) bar constants."""
    youngs_modulus, poisson_ratio, density = (float32_value(value) for value in _BAR_MATERIAL)
    lame_lambda = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    modulus = (lame_lambda + 2.0 * mu) * (1.0 + 1j * float32_value(_LOSS_FACTOR))
    return modulus, lambda frequency: 2.0 * math.pi * frequency * np.sqrt(density / modulus)


def _bar_specs():
    youngs_modulus, poisson_ratio, density = _BAR_MATERIAL
    material = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'rho': density,
                'mechanical_loss_factor': _LOSS_FACTOR}
    study = StudySpec(type='frequency_response', settings={'frequencies': list(_BAR_FREQUENCIES)})
    common = [FeatureSpec('sides', 'roller', ['y_lo', 'y_hi']),
              FeatureSpec('drive', 'boundary_load', ['x_hi'], {'t_x': _BAR_TRACTION, 't_y': 0.0}),
              FeatureSpec('tip', 'frequency_probe', ['p'])]
    specs = {}
    for run, features in (('fixed', [FeatureSpec('left', 'roller', ['x_lo'])] + common), ('free', common)):
        specs[run] = _triangle_model(f'Self-test: driven lossy bar on triangles ({run})', _BAR, _BAR_H,
                                     [PhysicsSpec(_SOLID, _SOLID_TYPE, features=features)], study, material,
                                     geometry=[_point_2d('p', *_BAR_POINT)])
    return specs


def _check_bar(results_by_name):
    """Probe series u_x_re / u_x_im per frequency (both runs) and the fixed run's x_lo complex reaction."""
    modulus, wavenumber = _bar_wave()
    length, height = _BAR
    x = _BAR_POINT[0]
    errors, tolerances, details = {}, {}, {}
    for run in ('fixed', 'free'):
        result = results_by_name[run].result(_SOLID, 'tip')
        frequencies = np.asarray(result['x'], dtype=np.float64)
        measured = (np.asarray(result['series']['u_x_re'], dtype=np.float64)
                    + 1j * np.asarray(result['series']['u_x_im'], dtype=np.float64))
        expected = []
        for frequency in frequencies:
            k = wavenumber(frequency)
            expected.append(_BAR_TRACTION * np.sin(k * x) / (modulus * k * np.cos(k * length)) if run == 'fixed'
                            else -_BAR_TRACTION * np.cos(k * x) / (modulus * k * np.sin(k * length)))
        expected = np.asarray(expected)
        errors[f'{run} probe'] = (float(np.max(np.abs(measured - expected) / np.abs(expected)))
                                  if len(measured) == len(_BAR_FREQUENCIES) else math.inf)
        tolerances[f'{run} probe'] = _PROBE_TOLERANCE
        details[run] = {'frequencies': frequencies.tolist(), 'measured': [str(value) for value in measured],
                        'expected': [str(value) for value in expected]}
    table = results_by_name['fixed'].result(_SOLID, 'left')
    columns = table['columns']
    reactions = np.array([float(row[columns.index('F_x_re')]) + 1j * float(row[columns.index('F_x_im')])
                          for row in table['rows']])
    expected = np.array([-_BAR_TRACTION * height / np.cos(wavenumber(float(row[0])) * length) for row in table['rows']])
    errors['fixed reaction'] = (float(np.max(np.abs(reactions - expected) / np.abs(expected)))
                                if len(reactions) == len(_BAR_FREQUENCIES) else math.inf)
    tolerances['fixed reaction'] = _PROBE_TOLERANCE
    details['reaction'] = {'measured': [str(value) for value in reactions], 'expected': [str(value) for value in expected]}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_solved_physics_selection (K2a): two solid_mechanics physics on the roller-box triangle mesh in one
# eigenfrequency study with StudySpec.solved_physics ['modal']: 'modal' (rollers on all faces) gives the box modes of
# solid_fem_roller_box_modes, 'static' (rollers x_lo / y_lo, traction T on x_hi) runs as a stationary background physics
# (no eigen solve): its x_lo reaction is -T H exactly (uniform uniaxial stress, exact in P2). Validation: a
# solved_physics naming no physics and one in a stationary study are errors.
_SELECTION_TRACTION = 2e6


def _selection_spec():
    material = {**_BOX_ORTHOTROPIC, 'rho': _BOX_DENSITY}
    modal = PhysicsSpec('modal', _SOLID_TYPE, features=[FeatureSpec('rollers', 'roller', list(_ROLLERS_2D)),
                                                        FeatureSpec(_TABLE, 'resonance_table')])
    static = PhysicsSpec('static', _SOLID_TYPE, features=[
        FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
        FeatureSpec('pull', 'boundary_load', ['x_hi'], {'t_x': _SELECTION_TRACTION, 't_y': 0.0})])
    study = StudySpec(type='eigenfrequency', settings={'n_resonances': _N_MODES}, solved_physics=['modal'])
    return _triangle_model('Self-test: StudySpec.solved_physics (one eigen-solved solid, one background solid)', _BOX,
                           _BOX_H, [modal, static], study, material)


def _check_selection(results_by_name):
    """Modal table vs the box continuum, one snapshot per mode, the background solid's static reaction, and the
    validation of two bad solved_physics lists."""
    from ...model.validate import validate
    results = results_by_name['selection']
    errors, tolerances, details = {}, {}, {}
    expected = _box_frequencies(_BOX_ORTHOTROPIC)
    measured = sorted(float(row[1]) for row in results.result('modal', _TABLE)['rows'])
    errors['modal frequency'] = (max(abs(value / reference - 1.0) for value, reference in zip(measured, expected))
                                 if len(measured) >= len(expected) else math.inf)
    tolerances['modal frequency'] = _FREQUENCY_TOLERANCE_2D
    labels = results.summary().get('snapshot_labels') or []
    errors['snapshots'], tolerances['snapshots'] = float(abs(len(labels) - _N_MODES)), 0.5
    force_x = float(results.result('static', 'left')['F_x'])
    reference = -_SELECTION_TRACTION * _BOX[1]
    errors['static reaction'], tolerances['static reaction'] = abs(force_x / reference - 1.0), 1e-6
    refused = {}
    for name, change in (('unknown', {'solved_physics': ['ghost']}),
                         ('stationary', {'type': 'stationary', 'settings': {}, 'solved_physics': ['modal']})):
        spec = _selection_spec()
        for key, value in change.items():
            setattr(spec.study, key, value)
        refused[name] = any(issue.level == 'error' and 'solved_physics' in issue.message for issue in validate(spec))
    errors['validation'] = 0.0 if all(refused.values()) else math.inf
    tolerances['validation'] = 1.0
    details.update(frequency_measured=measured, frequency_expected=expected, snapshot_labels=labels,
                   static_reaction=force_x, reaction_expected=reference, refused=refused)
    return scaled_worst_result(errors, tolerances, **details)


PRESTRESS_CASES = (
    SelfTestCase(
        name='solid_fem_prestressed_roller_box',
        label='Solid mechanics on triangles: prestressed (and lossy) roller-box modes, released initial stress',
        description=('The 10 x 4 mm orthotropic roller box of solid_fem_roller_box_modes (P2, h 0.25 mm) with '
                     f'initial_stress_xx = {_PRESTRESS_2D:g} Pa and stress_stiffening: the static state is u = 0, the '
                     'total stress sigma0, and each continuum mode (p, q) shifts to rho omega^2 = mu_i + S kx^2; the '
                     f'{_N_MODES} lowest table frequencies (sorted) within {_FREQUENCY_TOLERANCE_2D:g}. A second run '
                     f'adds mechanical_loss_factor {_LOSS_FACTOR:g} (K_sigma lossless): q_factor against Re omega / (2 '
                     f'Im omega) of rho omega^2 = mu_i (1 + j eta) + S kx^2 (dissipation dilution) within '
                     f'{_Q_TOLERANCE:g}. A stationary run releases the initial stress (rollers x_lo / y_lo): probes '
                     'u_x, u_y = C^-1 (-S, 0) . (x, y) and stress_xx = 0 (exact in P2) within '
                     f'{_EXACT_TOLERANCE:g}. measured = error in tolerance units, tolerance 1 (first estimates, not '
                     'yet run).'),
        specs=_prestressed_box_specs, check=_check_prestressed_box),
    SelfTestCase(
        name='solid_fem_frequency_probe',
        label='Solid mechanics on triangles: complex frequency probe and reaction of a driven lossy bar',
        description=('A 10 x 1 mm plane-strain bar (P2, h 0.25 mm, E 100 GPa, nu 0.25, rho 2000, '
                     f'mechanical_loss_factor {_LOSS_FACTOR:g}) with rollers on y_lo / y_hi and a traction on x_hi at '
                     f'{", ".join(f"{value / 1e3:g}" for value in _BAR_FREQUENCIES)} kHz: the frequency_probe series '
                     'u_x_re + j u_x_im at x = 0.75 L against the 1D lossy wave (fixed x_lo: T sin(k x) / (M* k cos(k '
                     'L)); free x_lo, a free body: -T cos(k x) / (M* k sin(k L))) and the fixed run\'s complex x_lo '
                     f'reaction table against -T H / cos(k L), within {_PROBE_TOLERANCE:g} relative. measured = error in '
                     'tolerance units, tolerance 1 (first estimate, not yet run).'),
        specs=_bar_specs, check=_check_bar),
    SelfTestCase(
        name='solid_solved_physics_selection',
        label='Studies: StudySpec.solved_physics runs an eigen-capable solid as a background physics',
        description=('Two solid_mechanics physics on one triangle mesh, eigenfrequency study with solved_physics '
                     "['modal']: the modal solid's table matches the roller-box continuum (solid_fem_roller_box_modes, "
                     f'{_FREQUENCY_TOLERANCE_2D:g}) with {_N_MODES} snapshots, the other solid is solved statically as '
                     'a background physics (its x_lo reaction = -T H, 1e-6); validation errors for a solved_physics '
                     'naming no physics and for one in a stationary study. measured = error in tolerance units, '
                     'tolerance 1 (first estimate, not yet run).'),
        specs=lambda: {'selection': _selection_spec()}, check=_check_selection),
)


# ---- solid_fem3d_prestressed_modes (K2, tetrahedra): the selftests_fem3d roller box (1 x 0.625 x 0.375 um, P2, cell
# 62.5 nm, isotropic) with initial_stress_xx = S and stress_stiffening: u = 0 static, every continuum mode's 3 x 3 symbol
# gains S kx^2 I. 'release': a 1 x 0.5 x 0.5 um block (P1, exact linear field) with rollers x_lo / y_lo / z_lo and
# sigma0_xx = S0 relaxes to zero total stress: eps_xx = -S0 / E, eps_yy = eps_zz = nu S0 / E.
_PRESTRESS_3D = 5e9
_RELEASE_STRESS_3D = 1e9
_RELEASE_SIZE = (1.0e-6, 0.5e-6, 0.5e-6)
_RELEASE_ELASTIC = (100e9, 0.3)
_RELEASE_POINT = (0.6e-6, 0.3e-6, 0.2e-6)


def _prestressed_modes_3d(prestress, orders=4):
    """Sorted lowest _MODES_LOWEST continuum frequencies [Hz] of the prestressed isotropic roller box."""
    c = _isotropic_constants()
    shear = np.array([[0.0, c['c66'], c['c55']], [c['c66'], 0.0, c['c44']], [c['c55'], c['c44'], 0.0]])
    normal = np.array([c['c11'], c['c22'], c['c33']])
    coupling = np.array([[0.0, c['c12'] + c['c66'], c['c13'] + c['c55']],
                         [c['c12'] + c['c66'], 0.0, c['c23'] + c['c44']],
                         [c['c13'] + c['c55'], c['c23'] + c['c44'], 0.0]])
    frequencies = []
    for triple in np.ndindex(orders, orders, orders):
        k = np.array([order * math.pi / side for order, side in zip(triple, _MODES_SIZE)])
        present = [axis for axis in range(3) if triple[axis] > 0]
        if not present:
            continue
        symbol = coupling * np.outer(k, k) + np.diag(normal * k ** 2 + shear @ k ** 2) + prestress * k[0] ** 2 * np.eye(3)
        values = np.linalg.eigvalsh(symbol[np.ix_(present, present)]) / _MODES_DENSITY
        frequencies.extend(math.sqrt(value) / (2.0 * math.pi) for value in values)
    return sorted(frequencies)[:_MODES_LOWEST]


def _prestressed_3d_specs():
    youngs_modulus, poisson_ratio = _MODES_ELASTIC
    modes = _solid_spec(
        'Self-test: prestressed roller-box modes on tetrahedra (quadratic)', _MODES_SIZE,
        [MaterialSpec('block', properties={'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio,
                                           'rho': _MODES_DENSITY, 'initial_stress_xx': _PRESTRESS_3D},
                      selection=['all'])],
        [FeatureSpec('rollers', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi']),
         FeatureSpec(_TABLE, 'resonance_table')], cell=_MODES_CELL,
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _MODES_LOWEST}),
        settings={'target_frequency': 0.0, 'stress_stiffening': True})
    release_youngs, release_nu = _RELEASE_ELASTIC
    release = _solid_spec(
        'Self-test: released initial stress on tetrahedra (linear)', _RELEASE_SIZE,
        [MaterialSpec('block', properties={'youngs_modulus': release_youngs, 'poisson_ratio': release_nu,
                                           'initial_stress_xx': _RELEASE_STRESS_3D}, selection=['all'])],
        [FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
         FeatureSpec('back', 'roller', ['z_lo']), *_probes('p', ('u_x', 'u_y', 'u_z', 'stress_xx', 'von_mises'))],
        [_point('p', *_RELEASE_POINT)], discretization='linear')
    return {'prestressed': modes, 'release': release}


def _check_prestressed_3d(results_by_name):
    """Prestressed table vs the shifted continuum; released probes vs the free uniaxial contraction."""
    errors, tolerances, details = {}, {}, {}
    rows = results_by_name['prestressed'].result(_SOLID, _TABLE)['rows']
    measured = sorted(float(row[1]) for row in rows)
    expected = _prestressed_modes_3d(_PRESTRESS_3D)
    errors['prestressed frequency'] = (max(abs(value / reference - 1.0)
                                           for value, reference in zip(measured, expected))
                                       if len(measured) >= len(expected) else math.inf)
    tolerances['prestressed frequency'] = _MODES_TOLERANCE
    unstressed = _prestressed_modes_3d(0.0)
    details['prestressed'] = {'measured': measured, 'expected': expected, 'unstressed': unstressed}
    youngs_modulus, poisson_ratio = (float32_value(value) for value in _RELEASE_ELASTIC)
    strain_xx = -_RELEASE_STRESS_3D / youngs_modulus
    strain_lateral = poisson_ratio * _RELEASE_STRESS_3D / youngs_modulus
    x, y, z = _RELEASE_POINT
    release = results_by_name['release']
    scale = abs(strain_xx) * _RELEASE_SIZE[0]
    for name, value in (('u_x', strain_xx * x), ('u_y', strain_lateral * y), ('u_z', strain_lateral * z)):
        errors[f'release {name}'] = abs(last_value(release, _SOLID, f'{name}_p') - value) / scale
        tolerances[f'release {name}'] = _EXACT_TOLERANCE
    for name in ('stress_xx', 'von_mises'):
        errors[f'release {name}'] = abs(last_value(release, _SOLID, f'{name}_p')) / _RELEASE_STRESS_3D
        tolerances[f'release {name}'] = _EXACT_TOLERANCE
    return scaled_worst_result(errors, tolerances, **details)


PRESTRESS_3D_CASES = (
    SelfTestCase(
        name='solid_fem3d_prestressed_modes',
        label='Solid mechanics 3D on tetrahedra: prestressed roller-box modes and a released initial stress',
        description=('The 1 x 0.625 x 0.375 um isotropic roller box of solid_fem3d_roller_box_modes (P2, 62.5 nm) '
                     f'with initial_stress_xx = {_PRESTRESS_3D:g} Pa and stress_stiffening (static state u = 0): the '
                     f'{_MODES_LOWEST} lowest table frequencies against the continuum with S kx^2 added to every '
                     f'symbol, within {_MODES_TOLERANCE:g}; a stationary 1 x 0.5 x 0.5 um block (P1) with rollers '
                     f'x_lo / y_lo / z_lo and initial_stress_xx = {_RELEASE_STRESS_3D:g} Pa relaxes to zero total '
                     'stress (probes u = (-S0 x, nu S0 y, nu S0 z) / E, stress_xx and von_mises 0) within '
                     f'{_EXACT_TOLERANCE:g}. measured = error in tolerance units, tolerance 1 (first estimates, not yet '
                     'run).'),
        specs=_prestressed_3d_specs, check=_check_prestressed_3d),
)
