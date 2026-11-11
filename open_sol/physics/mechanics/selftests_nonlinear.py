"""Self-tests of the geometrically nonlinear solid_mechanics on triangles (new_modules_plan K6, 2D): exact uniform
total-Lagrangian states (model runs: plane-stress stretch with its nominal reaction, released initial stress,
Prescribed displacement (domain), an axisymmetric released ring; validation refusals), the geometric Duffing coefficient
of a prestressed in-plane string against the Hirsch E4 closed forms (CPU, structured mesh) and the released nanostring
modes linearised about its nonlinear state (Hirsch E1 B). definition registers them."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, last_value, scaled_worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)
from .selftests_prestress_string import (_FILM_STRESS, _FREQUENCY_TOLERANCE, _MATERIAL, _MODES_B, _STRESS_TOLERANCE,
                                         _TABLE, _beam_frequencies, _string_spec, _table_frequencies)

UM = 1e-6
_SOLID = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME


# ---- solid_fem_nonlinear_uniform (model runs, P1 triangles): a 1 x 0.5 um block, geometric nonlinearity, load_steps
# 2, roller y_lo. Every state is a homogeneous deformation x -> diag(l) x (exact in P1). Plane stress: 'stretch' (roller
# x_lo, u_x = d on x_hi, 10 %): E_xx = (l_x^2 - 1) / 2, E_yy = -nu E_xx, S_xx = E E_xx, nominal reaction l_x S_xx H per
# m on x_hi; W = 1/2 E E_xx^2. 'release' (roller x_lo, initial_stress_xx S0): S = 0, E_xx = -S0 / E, E_yy = nu S0 / E,
# W = S0^2 / (2 E). 'domain_hold': Prescribed displacement (domain) u_x = 0 on the whole block with the same S0: u = 0,
# S_xx = S0. Axisymmetric 'ring_release' (the axis holds u_r): initial_stress_xx = _zz (rr, hoop) = S0 released to S = 0:
# E_rr = E_tt = -(1 - nu) S0 / E, E_zz = 2 nu S0 / E (hoop Green-Lagrange u_r / r + (u_r / r)^2 / 2), W = (1 - nu) S0^2
# / E. Validation refuses the setting with azimuthal_order 1, an elastic PML and piezoelectric.
_UNIFORM_SIZE = (1.0 * UM, 0.5 * UM)
_UNIFORM_CELL = 0.25 * UM
_UNIFORM_ELASTIC = (100e9, 0.3)
_UNIFORM_STRETCH = 0.1 * UM
_UNIFORM_STRESS = 1e10             # Pa (float32-exact): release strain 0.1
_UNIFORM_POINT = (0.6 * UM, 0.3 * UM)
_UNIFORM_COMPONENTS = ('u_x', 'u_y', 'strain_xx', 'strain_yy', 'stress_xx', 'stress_yy', 'strain_energy_density',
                       'kinetic_energy_density')
_RING_COMPONENTS = ('elastic_strain_zz', 'stress_zz')   # the hoop components (axisymmetric)
_UNIFORM_TOLERANCE = 1e-5          # float32 probe samples of exact fields (first estimate)
_REACTION_TOLERANCE = 1e-6
_RUNS = ('stretch', 'release', 'domain_hold', 'ring_release')


def _uniform_spec(run, hold_u_x=0.0, **extra_settings):
    youngs_modulus, poisson_ratio = _UNIFORM_ELASTIC
    material = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio}
    settings = {'geometric_nonlinearity': True, 'load_steps': 2, 'discretization': 'linear'}
    features = [FeatureSpec('bottom', 'roller', ['y_lo'])]
    components = _UNIFORM_COMPONENTS
    if run == 'ring_release':
        settings['geometry'] = 'axisymmetric'
        material.update(initial_stress_xx=_UNIFORM_STRESS, initial_stress_zz=_UNIFORM_STRESS)
        components = components + _RING_COMPONENTS
    else:
        settings['out_of_plane'] = 'plane_stress'
        if run == 'stretch':
            features += [FeatureSpec('left', 'roller', ['x_lo']),
                         FeatureSpec('pull', 'displacement', ['x_hi'], {'u_x': _UNIFORM_STRETCH, 'u_y': None})]
        elif run == 'release':
            features.append(FeatureSpec('left', 'roller', ['x_lo']))
            material['initial_stress_xx'] = _UNIFORM_STRESS
        else:
            features.append(FeatureSpec('hold', 'prescribed_displacement_domain', ['all'],
                                        {'u_x': hold_u_x, 'u_y': None}))
            material['initial_stress_xx'] = _UNIFORM_STRESS
    settings.update(extra_settings)
    probes = [FeatureSpec(f'{name}_p', 'solid_probe', ['p'], {'component': name}) for name in components]
    return ModelSpec(
        title=f'Self-test: geometrically nonlinear uniform block on triangles ({run})',
        domain=DomainSpec(size_x=_UNIFORM_SIZE[0], size_y=_UNIFORM_SIZE[1]),
        geometry=[GeometrySpec('p', 'point', {'x': _UNIFORM_POINT[0], 'y': _UNIFORM_POINT[1]})],
        materials=[MaterialSpec('block', properties=material, selection=['all'])],
        physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, settings=settings, features=features + probes)],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_UNIFORM_CELL)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'), study=StudySpec(type='stationary'))


def _uniform_specs():
    return {run: _uniform_spec(run) for run in _RUNS}


def _uniform_expected(normal_strains, stress_xx, energy, hoop=None):
    """Probe values of the homogeneous state of Green-Lagrange normal strains (E_xx, E_yy) (+ the hoop E_tt)."""
    stretches = [math.sqrt(1.0 + 2.0 * strain) for strain in normal_strains]
    expected = {f'u_{axis}': (stretches[index] - 1.0) * _UNIFORM_POINT[index] for index, axis in enumerate('xy')}
    expected.update(strain_xx=normal_strains[0], strain_yy=normal_strains[1], stress_xx=stress_xx, stress_yy=0.0,
                    strain_energy_density=energy, kinetic_energy_density=0.0)
    if hoop is not None:
        expected.update(elastic_strain_zz=hoop, stress_zz=0.0)
    return expected, stretches


def _refusal_errors(errors, tolerances):
    """Validation: an empty Prescribed displacement (domain) and geometric_nonlinearity with azimuthal_order 1, an
    elastic PML and piezoelectric are errors."""
    from ...model.validate import validate
    checks = {'empty domain hold refused': (_uniform_spec('domain_hold', hold_u_x=None), 'holds nothing'),
              'azimuthal order refused': (_uniform_spec('ring_release', azimuthal_order=1),
                                          'geometric_nonlinearity is not available with azimuthal_order'),
              'elastic PML refused': (_uniform_spec('release', pml_x_hi=True, target_frequency=1e9),
                                      'geometric_nonlinearity is not available with an elastic PML'),
              'piezoelectric refused': (_uniform_spec('release', piezoelectric=True),
                                        'geometric_nonlinearity is not available with piezoelectric')}
    for name, (spec, text) in checks.items():
        refused = any(issue.level == 'error' and text in issue.message for issue in validate(spec))
        errors[name], tolerances[name] = (0.0 if refused else math.inf), 1.0


def _check_uniform(results_by_name):
    """Probes of the four homogeneous states, the stretch's nominal x_hi reaction and the validation refusals."""
    youngs_modulus, poisson_ratio = (float32_value(value) for value in _UNIFORM_ELASTIC)
    stretch_strain = ((1.0 + _UNIFORM_STRETCH / _UNIFORM_SIZE[0]) ** 2 - 1.0) / 2.0
    release_strain = _UNIFORM_STRESS / youngs_modulus
    radial_strain = -(1.0 - poisson_ratio) * release_strain
    cases = {
        'stretch': _uniform_expected((stretch_strain, -poisson_ratio * stretch_strain),
                                     youngs_modulus * stretch_strain, 0.5 * youngs_modulus * stretch_strain ** 2),
        'release': _uniform_expected((-release_strain, poisson_ratio * release_strain), 0.0,
                                     0.5 * _UNIFORM_STRESS * release_strain),
        'domain_hold': _uniform_expected((0.0, 0.0), _UNIFORM_STRESS, 0.0),
        'ring_release': _uniform_expected((radial_strain, 2.0 * poisson_ratio * release_strain), 0.0,
                                          (1.0 - poisson_ratio) * _UNIFORM_STRESS * release_strain, hoop=radial_strain)}
    scales = {'u': max(_UNIFORM_STRETCH, release_strain * _UNIFORM_SIZE[0]), 'strain': 0.1, 'stress': _UNIFORM_STRESS,
              'energy': 0.5 * _UNIFORM_STRESS * release_strain}
    errors, tolerances, details = {}, {}, {}
    for run, (expected, stretches) in cases.items():
        results = results_by_name[run]
        measured = {name: last_value(results, _SOLID, f'{name}_p') for name in expected}
        for name, value in expected.items():
            kind = 'u' if name.startswith('u_') else 'strain' if 'strain' in name and 'energy' not in name else (
                'energy' if 'energy' in name else 'stress')
            errors[f'{run} {name}'] = abs(measured[name] - value) / scales[kind]
            tolerances[f'{run} {name}'] = _UNIFORM_TOLERANCE
        details[run] = {'measured': measured, 'expected': expected, 'stretches': stretches}
    stretch_x = 1.0 + _UNIFORM_STRETCH / _UNIFORM_SIZE[0]
    expected_force = stretch_x * youngs_modulus * stretch_strain * _UNIFORM_SIZE[1]   # nominal force per m of depth
    table = results_by_name['stretch'].result(_SOLID, 'pull')
    forces = (float(table['F_x']), float(table['F_y']))
    errors['stretch reaction'] = max(abs(forces[0] - expected_force), abs(forces[1])) / expected_force
    tolerances['stretch reaction'] = _REACTION_TOLERANCE
    details['reaction'] = {'measured': forces, 'expected_F_x': expected_force,
                           'linear_K_u_F_x': youngs_modulus * (stretch_x - 1.0) * _UNIFORM_SIZE[1]}
    _refusal_errors(errors, tolerances)
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem_duffing_string (CPU, no model): the Hirsch E4 string in the drawing plane, L = 100 um, h = 100 nm, per m
# of depth (plane stress: uniaxial E), Si3N4 (E 250 GPa, nu 0.23), sigma0_xx = 1 GPa, P2 on 64 x 2 structured right
# triangles, f_y = f0 sin(pi x / L) per element (centroid). Supports on the neutral axis (y = h / 2): u_y = 0 at the
# neutral-axis node of both end faces, dead end tractions +-sigma0 that balance the initial stress (u = 0 at f0 = 0) and
# u_x = 0 at those two nodes only ('a', axial dofs free: uniform tension, alpha = E h pi^4 / (8 L^3) = 3.044e10 N/m^3 per
# 100 nm of depth, Kirchhoff-Carrier) or at every neutral-axis node ('b', no-slip: 1.5 x that). Pinning only the neutral
# axis keeps the ends simply supported (sin(pi x / L) the exact linear shape, bending only adds to k); a hold of u_x on
# every node across would lock the cross-section rotation, and one of u_y its term eta (cos theta - 1): a forced
# E_yy = theta^2 / 2 in the end elements, alpha + 2.4 % at 64 cells, growing with the element length. F = int f_y psi dA
# against the modal amplitude a = int u_y psi dA / int psi^2 dA (psi = sin(pi x / L)): k = F / a of a load at alpha a^2
# / k = 1e-6, then F / a - k = alpha a^2 + c a^4 over alpha a^2 / k = 0.01 .. 0.1 (a free k trades against alpha).
_STRING = (100.0 * UM, 0.1 * UM)
_STRING_CELLS = 64
_STRING_LAYERS = 2                 # cells across: a node line on the neutral axis
_SILICON_NITRIDE = (250e9, 0.23)
_STRING_STRESS = 1e9
_RATIOS = (0.01, 0.025, 0.05, 0.075, 0.1)
_LINEAR_RATIO = 1e-6               # alpha a^2 / k of the load that measures k
_DUFFING_TOLERANCE = 2e-2          # relative alpha (Hirsch plan E4)
_NONLINEAR_SETTINGS = {'geometric_nonlinearity': True, 'nonlinear_tolerance': 1e-10, 'nonlinear_max_iterations': 30}
_SHAPE_DEGREE = 6                  # rule of int N psi dA (the largest triangle rule)


def _string_alpha(no_slip):
    """Kirchhoff-Carrier E h pi^4 / (8 L^3) per m of depth (x 1.5 with the axial dofs clamped) [N/m^3 x m]."""
    length, height = _STRING
    return (1.5 if no_slip else 1.0) * _SILICON_NITRIDE[0] * height * math.pi ** 4 / (8.0 * length ** 3)


def pinned_duffing_fit(linear_point, points):
    """(k, alpha) from (amplitude, force) samples: k = F / a of the linear-load point, alpha of the least-squares fit
    F / a - k = alpha a^2 + c a^4 over the other points."""
    stiffness = linear_point[1] / linear_point[0]
    a = np.array([point[0] for point in points])
    forces = np.array([point[1] for point in points])
    design = np.stack([a ** 2, a ** 4], axis=1)
    coefficients = np.linalg.lstsq(design, forces / a - stiffness, rcond=None)[0]
    return float(stiffness), float(coefficients[0])


def _string_physics(**settings):
    """SolidMechanicsFEMPhysics (P2 plane stress, cpu, geometric nonlinearity, + settings) on the structured strip, Si3N4
    with initial_stress_xx (float32, as a build writes it), end faces loaded by the balancing tractions (no hold yet)."""
    from ...fem.domain import TriangleDomain
    from ...fem.selftests_harmonic import _rectangle_mesh
    from .solver_fem import SolidMechanicsFEMPhysics
    length, height = _STRING
    mesh = _rectangle_mesh(np.linspace(0.0, length, _STRING_CELLS + 1), np.linspace(0.0, height, _STRING_LAYERS + 1))
    domain = TriangleDomain(mesh, backend='cpu')
    physics = SolidMechanicsFEMPhysics(domain, out_of_plane='plane_stress', discretization='quadratic',
                                       **dict(_NONLINEAR_SETTINGS, **settings))
    youngs_modulus, poisson_ratio = _SILICON_NITRIDE
    material = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'initial_stress_xx': _STRING_STRESS}
    domain.materials.add_properties({name: 0.0 for name in material if name not in domain.materials.property_names()})
    for name, value in material.items():
        domain.materials.assign(name, None, value)
    for face, sign in (('x_lo', -1.0), ('x_hi', 1.0)):
        physics.add_face_traction(face, sign * _STRING_STRESS, 0.0)
    return physics


def _held_string_physics(no_slip, **settings):
    """_string_physics(**settings) with the neutral-axis holds of one variant (no load)."""
    length, height = _STRING
    physics = _string_physics(**settings)
    tolerance = 1e-6 * height
    full_dofs = physics.system.full_dofs
    points = full_dofs.dof_points()
    neutral = np.abs(points[:, 1] - 0.5 * height) < tolerance
    ends = neutral & ((points[:, 0] < tolerance) | (points[:, 0] > length - tolerance))
    axial = neutral if no_slip else ends
    held = full_dofs.global_ids[np.flatnonzero(axial)]
    if held.size < (2 * _STRING_CELLS + 1 if no_slip else 2) or np.count_nonzero(ends) != 2:
        raise RuntimeError(f'string self-test: {held.size} neutral-axis dofs found (P2 on {_STRING_LAYERS} layers puts '
                           'a node line on y = h / 2)')
    physics.system.hold(full_dofs.global_ids[np.flatnonzero(ends)], u_y=0.0)
    physics.system.hold(held, u_x=0.0)
    return physics


def _string_stiffness():
    """Linear string stiffness sigma0 h pi^2 / (2 L) of the first mode per m of depth [N/m^2]."""
    length, height = _STRING
    return _STRING_STRESS * height * math.pi ** 2 / (2.0 * length)


def _string_run(no_slip):
    """(k, alpha, points) of one string variant (points[0]: the linear-load sample)."""
    from ...fem.assembly import assemble_load
    from ...fem.dofs import lagrange_dofs
    length, height = _STRING
    physics = _held_string_physics(no_slip)
    stiffness = _string_stiffness()
    alpha_expected = _string_alpha(no_slip)
    mode_norm = 0.5 * length * height

    def psi(quadrature_points, elements):
        return np.sin(math.pi * quadrature_points[..., 0] / length)

    mesh = physics.domain.mesh
    element_weights = assemble_load(lagrange_dofs(mesh, 0), psi, degree=_SHAPE_DEGREE)   # int_e psi dA
    shape = np.sin(math.pi * mesh.centroids()[:, 0] / length)
    unit_force = float(element_weights @ shape)   # F of f0 = 1
    samples = []
    for ratio in (_LINEAR_RATIO,) + _RATIOS:
        target = math.sqrt(ratio * stiffness / alpha_expected)
        force = stiffness * target * (1.0 + ratio)
        physics.inject_source('body_force_y', (force / unit_force) * shape)
        physics.step(0.0, math.inf)
        system = physics.system   # the solid map of the last assembly carries the unknown
        amplitude = float(assemble_load(system.dofs, psi, degree=_SHAPE_DEGREE) @ system.unknown[:, 1]) / mode_norm
        samples.append((amplitude, force, physics.last_converged))
    fitted_stiffness, alpha = pinned_duffing_fit(samples[0], samples[1:])
    return fitted_stiffness, alpha, samples


def _check_string(_):
    errors, tolerances, details = {}, {}, {}
    for run, no_slip in (('axial free (a)', False), ('no-slip (b)', True)):
        stiffness, alpha, samples = _string_run(no_slip)
        expected = _string_alpha(no_slip)
        errors[f'{run} alpha'], tolerances[f'{run} alpha'] = abs(alpha / expected - 1.0), _DUFFING_TOLERANCE
        converged = all(sample[2] for sample in samples)
        errors[f'{run} converged'], tolerances[f'{run} converged'] = (0.0 if converged else math.inf), 1.0
        details[run] = {'alpha': alpha, 'alpha_expected': expected, 'k': stiffness,
                        'k_string': _STRING_STRESS * _STRING[1] * math.pi ** 2 / (2.0 * _STRING[0]),
                        'amplitudes': [sample[0] for sample in samples], 'forces': [sample[1] for sample in samples]}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem_nonlinear_warm_start (CPU, plan C2; 3D twin solid_fem3d_nonlinear_warm_start): the axial-free string
# at alpha a^2 / k = 0.05 seeds (warm_start_fields -> seed_warm_start, the sweep path) a new physics on the same mesh
# solving 0.1: equal to a cold 0.1 solve within the Newton tolerance with fewer Newton iterations; a -5 u seed and a
# non-finite seed (which must fall back to the cold ramp) still match. 0.05 -> 0.1 is strongly nonlinear (a ~ +40 %).
# Full Newton: the iteration count measures the seed only there (when_slow, the default, trades factorisations for
# more, linearly converging iterations: a closer seed can then take more iterations, 10 against 9 cold)
WARM_TANGENT_UPDATE = 'every_iteration'
WARM_RATIOS = (0.05, 0.1)           # alpha a^2 / k of the seeding solve and of the seeded one
WARM_TOLERANCE = 1e-9               # relative to max |u| (Newton tolerance 1e-10, quadratic convergence)


def warm_string_load(ratio, stiffness, alpha, unit_force_density):
    """Body force amplitude f0 aiming the string mode at alpha a^2 / k = ratio (F = k a (1 + ratio), F = f0 x
    unit_force_density)."""
    amplitude = math.sqrt(ratio * stiffness / alpha)
    return stiffness * amplitude * (1.0 + ratio) / unit_force_density


def warm_start_runs(make_physics, apply_load, displacement):
    """{run: (physics, its displacement)}: 'seed source' (WARM_RATIOS[0]), then at WARM_RATIOS[1] 'cold', 'seeded',
    'bad seed (-5 u)' and 'non-finite seed' (new physics seeded with transformed warm_start_fields of the source)."""
    source = make_physics()
    apply_load(source, WARM_RATIOS[0])
    fields = source.warm_start_fields()
    runs = {'seed source': (source, displacement(source))}
    transforms = {'cold': None, 'seeded': lambda u: u, 'bad seed (-5 u)': lambda u: -5.0 * u,
                  'non-finite seed': lambda u: np.full_like(u, np.nan)}
    for run, transform in transforms.items():
        physics = make_physics()
        if transform is not None:
            physics.seed_warm_start(dict(fields, u=transform(np.asarray(fields['u'], dtype=np.float64))))
        apply_load(physics, WARM_RATIOS[1])
        runs[run] = (physics, displacement(physics))
    return runs


def warm_start_errors(runs):
    """(errors, tolerances, details) of warm_start_runs: displacements against the cold run, fewer Newton iterations
    seeded, the start texts (sweep seed, cold-ramp fallback), every run converged."""
    errors, tolerances = {}, {}
    cold = runs['cold'][1]
    scale = float(np.max(np.abs(cold)))
    counters = {run: physics.work_counters() for run, (physics, _) in runs.items()}
    iterations = {run: int(counts.get('nonlinear_iterations', 0)) for run, counts in counters.items()}
    factorizations = {run: int(counts.get('tangent_factorizations', 0)) for run, counts in counters.items()}
    texts = {run: physics._nonlinear_text() for run, (physics, _) in runs.items()}
    for run in ('seeded', 'bad seed (-5 u)', 'non-finite seed'):
        errors[f'{run} u'] = float(np.max(np.abs(runs[run][1] - cold))) / scale
        tolerances[f'{run} u'] = WARM_TOLERANCE
    checks = {'seeded fewer iterations': iterations['seeded'] < iterations['cold'],
              'seeded from the sweep seed': 'sweep seed' in texts['seeded'],
              'non-finite seed fell back': 'cold ramp' in texts['non-finite seed'],
              'every run converged': all(physics.last_converged for physics, _ in runs.values())}
    for name, passed in checks.items():
        errors[name], tolerances[name] = (0.0 if passed else math.inf), 1.0
    return errors, tolerances, {'iterations': iterations, 'tangent_factorizations': factorizations, 'texts': texts,
                                'max_u': scale}


def _check_warm_start(_):
    length, height = _STRING
    stiffness, alpha = _string_stiffness(), _string_alpha(False)
    unit_force_density = 0.5 * length * height   # F of f0 = 1 (load shape ~ the mode shape), per m of depth

    def apply_load(physics, ratio):
        shape = np.sin(math.pi * physics.domain.mesh.centroids()[:, 0] / length)
        physics.inject_source('body_force_y', warm_string_load(ratio, stiffness, alpha, unit_force_density) * shape)
        physics.step(0.0, math.inf)

    runs = warm_start_runs(lambda: _held_string_physics(False, tangent_update=WARM_TANGENT_UPDATE), apply_load,
                           lambda physics: np.array(physics.system.unknown, dtype=np.float64))
    errors, tolerances, details = warm_start_errors(runs)
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem_tangent_update (CPU, plan G4; 3D twin solid_fem3d_tangent_update): the axial-free string with
# load_steps 2 solved at alpha a^2 / k = 0.05 then 0.1 (the second solve starts from the first: a factor kept across
# load steps and solves) with tangent_update 'every_iteration' and 'when_slow'. Modified Newton keeps the factor while
# the residual falls below 0.25 x the previous one, so it converges linearly at < 1/4: its state equals full Newton
# within the Newton tolerance (1e-10) with fewer tangent factorisations; full Newton factorises every iteration.
TANGENT_RATIOS = (0.05, 0.1)        # alpha a^2 / k of the two solves
TANGENT_LOAD_STEPS = 2


def tangent_update_runs(make_physics, apply_load, displacement):
    """{tangent_update: (physics, its displacement)} of make_physics(tangent_update=..., load_steps=TANGENT_LOAD_STEPS)
    after the TANGENT_RATIOS solves in sequence."""
    runs = {}
    for tangent_update in ('every_iteration', 'when_slow'):
        physics = make_physics(tangent_update=tangent_update, load_steps=TANGENT_LOAD_STEPS)
        for ratio in TANGENT_RATIOS:
            apply_load(physics, ratio)
        runs[tangent_update] = (physics, displacement(physics))
    return runs


def tangent_update_errors(runs):
    """(errors, tolerances, details) of tangent_update_runs: when_slow against every_iteration, fewer factorisations,
    some reuse, full Newton one factorisation per iteration, both converged."""
    full, modified = runs['every_iteration'], runs['when_slow']
    scale = float(np.max(np.abs(full[1])))
    counters = {run: physics.work_counters() for run, (physics, _) in runs.items()}
    errors = {'when_slow u': float(np.max(np.abs(modified[1] - full[1]))) / scale}
    tolerances = {'when_slow u': WARM_TOLERANCE}
    full_counts, modified_counts = counters['every_iteration'], counters['when_slow']
    checks = {'when_slow fewer factorisations': (modified_counts.get('tangent_factorizations', 0)
                                                 < full_counts.get('tangent_factorizations', 0)),
              'when_slow reused a factor': modified_counts.get('tangent_reuses', 0) > 0,
              'every_iteration factorises each iteration': (
                  full_counts.get('tangent_factorizations') == full_counts.get('nonlinear_iterations')
                  and full_counts.get('tangent_reuses') == 0),
              'every run converged': all(physics.last_converged for physics, _ in runs.values())}
    for name, passed in checks.items():
        errors[name], tolerances[name] = (0.0 if passed else math.inf), 1.0
    return errors, tolerances, {'counters': counters, 'max_u': scale}


def _check_tangent_update(_):
    length, height = _STRING
    stiffness, alpha = _string_stiffness(), _string_alpha(False)
    unit_force_density = 0.5 * length * height

    def apply_load(physics, ratio):
        shape = np.sin(math.pi * physics.domain.mesh.centroids()[:, 0] / length)
        physics.inject_source('body_force_y', warm_string_load(ratio, stiffness, alpha, unit_force_density) * shape)
        physics.step(0.0, math.inf)

    runs = tangent_update_runs(lambda **settings: _held_string_physics(False, **settings), apply_load,
                               lambda physics: np.array(physics.system.unknown, dtype=np.float64))
    errors, tolerances, details = tangent_update_errors(runs)
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem_nonlinear_released_string (Hirsch E1 B on the nonlinear state): the released nanostring of
# selftests_prestress_string with geometric_nonlinearity on and stress_stiffening off. Total-Lagrangian: away from the
# clamps H_xx = 0 and S_yy = 0 give S_xx = sigma0 (1 - nu) exactly as the linear theory, and the tension of the
# reference section is S_xx h (F_xx = 1), so the modes of the tangent K_T match the same clamped tensioned-beam roots
# (the thinning H_yy ~ -3e-3 changes only the bending share, ~1 % of f, by ~1e-4); tolerances of the linear case.
def _released_specs():
    specs = {}
    for run, study, features, geometry in (
            ('modes', StudySpec(type='eigenfrequency', settings={'n_resonances': _MODES_B}),
             [FeatureSpec(_TABLE, 'resonance_table')], ()),
            ('static', StudySpec(type='stationary'),
             [FeatureSpec(f'{name}_mid', 'solid_probe', ['mid'], {'component': name})
              for name in ('stress_xx', 'stress_yy')],
             [GeometrySpec('mid', 'point', {'x': 50e-6, 'y': 50e-9})])):
        spec = _string_spec(f'Self-test: nonlinear released nanostring ({run}, hirsch E1 B)', True, study, features,
                            geometry)
        spec.physics[0].settings.update(geometric_nonlinearity=True, stress_stiffening=False)
        specs[run] = spec
    return specs


def _check_released(results_by_name):
    _, poisson_ratio, _ = (float32_value(value) for value in _MATERIAL)
    released_stress = _FILM_STRESS * (1.0 - poisson_ratio)
    expected = _beam_frequencies(released_stress, _MODES_B)
    measured = _table_frequencies(results_by_name['modes'])
    errors = {'frequency': (max(abs(value / reference - 1.0) for value, reference in zip(measured, expected))
                            if len(measured) >= _MODES_B else math.inf)}
    static = results_by_name['static']
    stress_xx, stress_yy = last_value(static, _SOLID, 'stress_xx_mid'), last_value(static, _SOLID, 'stress_yy_mid')
    errors['mid-span stress_xx'] = abs(stress_xx / released_stress - 1.0)
    errors['mid-span stress_yy'] = abs(stress_yy) / released_stress
    tolerances = {'frequency': _FREQUENCY_TOLERANCE, 'mid-span stress_xx': _STRESS_TOLERANCE,
                  'mid-span stress_yy': _STRESS_TOLERANCE}
    details = {'measured': measured, 'expected': expected, 'stress_xx': stress_xx, 'stress_yy': stress_yy,
               'released_stress': released_stress}
    return scaled_worst_result(errors, tolerances, **details)


NONLINEAR_CASES = (
    SelfTestCase(
        name='solid_fem_nonlinear_uniform',
        label='Solid mechanics on triangles: geometrically nonlinear homogeneous states (stretch, release, domain hold, '
              'axisymmetric ring)',
        description=('A 1 x 0.5 um block (P1, 0.25 um elements, E 100 GPa, nu 0.3) with geometric_nonlinearity and '
                     'load_steps 2: in plane stress a 10 % stretch by a Displacement on x_hi (Green-Lagrange E_xx = '
                     '(l^2 - 1) / 2, E_yy = -nu E_xx, S_xx = E E_xx, the nominal reaction l S_xx H of the internal '
                     f'force), a released initial_stress_xx = {_UNIFORM_STRESS:g} Pa (S = 0) and a Prescribed '
                     'displacement (domain) u_x = 0 on the whole block holding it (u = 0, S_xx = S0); axisymmetric a '
                     'released rr + hoop initial stress (E_rr = E_tt = -(1 - nu) S0 / E, E_zz = 2 nu S0 / E: the '
                     'Green-Lagrange hoop terms); probes u, strain, stress, hoop strain / stress and the energy '
                     f'densities within {_UNIFORM_TOLERANCE:g} of their scales (exact in P1), the reaction within '
                     f'{_REACTION_TOLERANCE:g}; validation refuses an empty domain hold and the setting with '
                     'azimuthal_order 1, an elastic PML or piezoelectric. measured = error in tolerance units, '
                     'tolerance 1 (first estimates, not yet run).'),
        specs=_uniform_specs, check=_check_uniform),
    SelfTestCase(
        name='solid_fem_duffing_string',
        label='Solid mechanics on triangles: geometric Duffing coefficient of a prestressed in-plane string (Hirsch E4)',
        description=('The Hirsch E4 string in the drawing plane (100 um x 100 nm Si3N4, plane stress per m of depth, '
                     'sigma0 1 GPa, P2 on 64 x 2 structured triangles, CPU, no model) under f_y = f0 sin(pi x / L), '
                     'supports at the neutral-axis end nodes only, geometrically nonlinear static solves at a linear '
                     'load (k) and five loads: the fit F / a - k = alpha a^2 + c a^4 of the generalized force against '
                     'the modal amplitude gives alpha = E h pi^4 / (8 L^3) with axial dofs free (Kirchhoff-Carrier) and '
                     '1.5 x that with the neutral axis held axially (no-slip), within '
                     f'{_DUFFING_TOLERANCE:g}. measured = error in tolerance units, tolerance 1.'),
        specs=dict, check=_check_string),
    SelfTestCase(
        name='solid_fem_nonlinear_warm_start',
        label='Solid mechanics on triangles: sweep warm start of the nonlinear Newton solve',
        description=('The axial-free in-plane string of solid_fem_duffing_string (CPU, no model, full Newton): a solve at alpha a^2 '
                     f'/ k = {WARM_RATIOS[0]:g} hands warm_start_fields to seed_warm_start of a new physics on the same '
                     f'mesh solving {WARM_RATIOS[1]:g} (the sweep warm start path); its displacement equals a cold '
                     f'solve within {WARM_TOLERANCE:g} of max |u| with fewer Newton iterations (work_counters), the '
                     'start text names the sweep seed, a -5 u seed still matches and a non-finite seed falls back to '
                     'the cold ramp and matches. measured = error in tolerance units, tolerance 1 (first estimate, not '
                     'yet run).'),
        specs=dict, check=_check_warm_start),
    SelfTestCase(
        name='solid_fem_tangent_update',
        label='Solid mechanics on triangles: modified Newton (tangent_update when_slow) of the nonlinear solve',
        description=('The axial-free in-plane string of solid_fem_duffing_string (CPU, no model, load_steps '
                     f'{TANGENT_LOAD_STEPS}) solved at alpha a^2 / k = {TANGENT_RATIOS[0]:g} then '
                     f'{TANGENT_RATIOS[1]:g} with tangent_update every_iteration and when_slow: the when_slow '
                     f'displacement equals full Newton within {WARM_TOLERANCE:g} of max |u| with fewer '
                     'tangent_factorizations and some tangent_reuses (work_counters); full Newton factorises once per '
                     'iteration; both converge. measured = error in tolerance units, tolerance 1 (first estimate, not '
                     'yet run).'),
        specs=dict, check=_check_tangent_update),
    SelfTestCase(
        name='solid_fem_nonlinear_released_string',
        label='Solid mechanics on triangles: released nanostring modes about its nonlinear state (Hirsch E1 B)',
        description=('The released Si3N4 nanostring of solid_fem_prestressed_string (equibiaxial 1 GPa film, plane '
                     'stress, 50 nm P2 triangles, Fixed ends) with geometric_nonlinearity on and stress_stiffening off: '
                     'the eigenfrequency pencil is the tangent K_T of the nonlinear static state; f1 .. '
                     f'f{_MODES_B} against the clamped tensioned-beam roots at sigma0 (1 - nu) within '
                     f'{_FREQUENCY_TOLERANCE:g}, the second Piola-Kirchhoff mid-span stress_xx = sigma0 (1 - nu) and '
                     f'stress_yy = 0 of a stationary run within {_STRESS_TOLERANCE:g}. measured = error in tolerance '
                     'units, tolerance 1 (first estimates, not yet run).'),
        specs=_released_specs, check=_check_released),
)
