"""Self-tests of the geometrically nonlinear solid_mechanics_3d on tetrahedra (new_modules_plan K6) and its energy
densities (K5): exact uniform total-Lagrangian states (model runs: stretch with its nominal reaction, released initial
stress, Prescribed displacement (domain)), the geometric Duffing coefficient of a prestressed string and of a square
membrane against the Hirsch plan's closed forms (CPU, structured meshes) and the modal energy identities (CPU).
definition3d registers them."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, last_value, scaled_worst_result, skip_without_cudss
from ...model.spec import FeatureSpec, MaterialSpec
from .selftests_fem3d import _point, _probes, _solid_spec
from .selftests_nonlinear import (TANGENT_LOAD_STEPS, TANGENT_RATIOS, WARM_RATIOS, WARM_TANGENT_UPDATE, WARM_TOLERANCE,
                                  pinned_duffing_fit, string_displacement, tangent_update_errors, tangent_update_runs,
                                  warm_start_errors, warm_start_runs, warm_string_load)

UM = 1e-6
_SOLID = 'solid'


# ---- solid_fem3d_nonlinear_uniform (model runs, P1): a 1 x 0.5 x 0.5 um block, rollers x_lo / y_lo / z_lo, geometric
# nonlinearity. Every state is a homogeneous deformation x -> diag(l) x (exact in P1). 'stretch': u_x = d on x_hi (10 %),
# uniaxial Saint Venant-Kirchhoff: E_xx = (l_x^2 - 1) / 2, E_yy = E_zz = -nu E_xx, S_xx = E E_xx, nominal reaction l_x
# S_xx A on x_hi; W = 1/2 E E_xx^2, kinetic 0. 'release': initial_stress_xx S0 relaxes to S = C E + S0 = 0: E_xx = -S0 /
# E, E_yy = E_zz = nu S0 / E, W = S0^2 / (2 E). 'domain_hold': Prescribed displacement (domain) u_x = 0 on the whole
# block, rollers y_lo / z_lo, the same S0: u = 0 and S_xx = S0 (without the hold it would relax).
_UNIFORM_SIZE = (1.0 * UM, 0.5 * UM, 0.5 * UM)
_UNIFORM_CELL = 0.25 * UM
_UNIFORM_ELASTIC = (100e9, 0.3)
_UNIFORM_STRETCH = 0.1 * UM
_UNIFORM_STRESS = 1e10             # Pa (float32-exact): release strain 0.1
_UNIFORM_POINT = (0.6 * UM, 0.3 * UM, 0.2 * UM)   # inside an element
_UNIFORM_COMPONENTS = ('u_x', 'u_y', 'u_z', 'strain_xx', 'strain_yy', 'stress_xx', 'stress_yy', 'strain_energy_density',
                       'kinetic_energy_density')
_UNIFORM_TOLERANCE = 1e-5          # float32 probe samples of exact fields (first estimate)
_REACTION_TOLERANCE = 1e-6


def _uniform_spec(run, hold_u_x=0.0):
    youngs_modulus, poisson_ratio = _UNIFORM_ELASTIC
    material = {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio}
    features = [FeatureSpec('bottom', 'roller', ['y_lo']), FeatureSpec('back', 'roller', ['z_lo'])]
    if run == 'stretch':
        features += [FeatureSpec('left', 'roller', ['x_lo']),
                     FeatureSpec('pull', 'displacement', ['x_hi'], {'u_x': _UNIFORM_STRETCH, 'u_y': None, 'u_z': None})]
    elif run == 'release':
        features.append(FeatureSpec('left', 'roller', ['x_lo']))
        material['initial_stress_xx'] = _UNIFORM_STRESS
    else:
        features.append(FeatureSpec('hold', 'prescribed_displacement_domain', ['all'],
                                    {'u_x': hold_u_x, 'u_y': None, 'u_z': None}))
        material['initial_stress_xx'] = _UNIFORM_STRESS
    return _solid_spec(f'Self-test: geometrically nonlinear uniform block on tetrahedra ({run})', _UNIFORM_SIZE,
                       [MaterialSpec('block', properties=material, selection=['all'])],
                       features + _probes('p', _UNIFORM_COMPONENTS), [_point('p', *_UNIFORM_POINT)],
                       discretization='linear', cell=_UNIFORM_CELL,
                       settings={'geometric_nonlinearity': True, 'load_steps': 2})


def _uniform_specs():
    return {run: _uniform_spec(run) for run in ('stretch', 'release', 'domain_hold')}


def _uniform_expected(normal_strains, stress_xx, energy):
    """Probe values of the homogeneous state of Green-Lagrange normal strains (E_xx, E_yy, E_zz)."""
    stretches = [math.sqrt(1.0 + 2.0 * strain) for strain in normal_strains]
    expected = {f'u_{axis}': (stretches[index] - 1.0) * _UNIFORM_POINT[index] for index, axis in enumerate('xyz')}
    expected.update(strain_xx=normal_strains[0], strain_yy=normal_strains[1], stress_xx=stress_xx, stress_yy=0.0,
                    strain_energy_density=energy, kinetic_energy_density=0.0)
    return expected, stretches


def _check_uniform(results_by_name):
    """Probes of the three homogeneous states, the stretch's nominal x_hi reaction and the validation of an empty
    Prescribed displacement (domain)."""
    from ...model.validate import validate
    youngs_modulus, poisson_ratio = (float32_value(value) for value in _UNIFORM_ELASTIC)
    stretch_strain = ((1.0 + _UNIFORM_STRETCH / _UNIFORM_SIZE[0]) ** 2 - 1.0) / 2.0
    release_strain = _UNIFORM_STRESS / youngs_modulus
    cases = {
        'stretch': _uniform_expected((stretch_strain, -poisson_ratio * stretch_strain, -poisson_ratio * stretch_strain),
                                     youngs_modulus * stretch_strain, 0.5 * youngs_modulus * stretch_strain ** 2),
        'release': _uniform_expected((-release_strain, poisson_ratio * release_strain, poisson_ratio * release_strain),
                                     0.0, 0.5 * _UNIFORM_STRESS * release_strain),
        'domain_hold': _uniform_expected((0.0, 0.0, 0.0), _UNIFORM_STRESS, 0.0)}
    scales = {'u': max(_UNIFORM_STRETCH, release_strain * _UNIFORM_SIZE[0]), 'strain': 0.1, 'stress': _UNIFORM_STRESS,
              'energy': 0.5 * _UNIFORM_STRESS * release_strain}
    errors, tolerances, details = {}, {}, {}
    for run, (expected, stretches) in cases.items():
        results = results_by_name[run]
        measured = {name: last_value(results, _SOLID, f'{name}_p') for name in _UNIFORM_COMPONENTS}
        for name, value in expected.items():
            kind = 'u' if name.startswith('u_') else name.split('_')[0]
            scale = scales['energy' if 'energy' in name else kind]
            errors[f'{run} {name}'] = abs(measured[name] - value) / scale
            tolerances[f'{run} {name}'] = _UNIFORM_TOLERANCE
        details[run] = {'measured': measured, 'expected': expected, 'stretches': stretches}
    stretch_x = 1.0 + _UNIFORM_STRETCH / _UNIFORM_SIZE[0]
    area = _UNIFORM_SIZE[1] * _UNIFORM_SIZE[2]
    expected_force = stretch_x * youngs_modulus * stretch_strain * area   # nominal (first Piola-Kirchhoff) force
    table = results_by_name['stretch'].result(_SOLID, 'pull')
    forces = (float(table['F_x']), float(table['F_y']), float(table['F_z']))
    errors['stretch reaction'] = max(abs(forces[0] - expected_force), abs(forces[1]), abs(forces[2])) / expected_force
    tolerances['stretch reaction'] = _REACTION_TOLERANCE
    details['reaction'] = {'measured': forces, 'expected_F_x': expected_force,
                           'linear_K_u_F_x': youngs_modulus * (stretch_x - 1.0) * area}
    empty = _uniform_spec('domain_hold', hold_u_x=None)
    refused = any(issue.level == 'error' and 'holds nothing' in issue.message for issue in validate(empty))
    errors['empty domain hold refused'], tolerances['empty domain hold refused'] = (0.0 if refused else math.inf), 1.0
    return scaled_worst_result(errors, tolerances, **details)


# ---- direct (no model) prestressed structures on structured Kuhn meshes (anisotropic cells, P2) ----
def _direct_physics(edges, material, backend='cpu', **settings):
    """SolidMechanicsFEMPhysics3D (P2, geometric nonlinearity; cpu unless backend) on structured_box_mesh(*edges) with
    one material {array name: value} on every element (float32, as a build writes it)."""
    from ...fem.domain3d import TetrahedronDomain
    from ...fem.mesh3d import structured_box_mesh
    from .solver_fem3d import SolidMechanicsFEMPhysics3D
    domain = TetrahedronDomain(structured_box_mesh(*edges), backend=backend)
    physics = SolidMechanicsFEMPhysics3D(domain, order=2, **settings)
    domain.materials.add_properties({name: 0.0 for name in material if name not in domain.materials.property_names()})
    for name, value in material.items():
        domain.materials.assign(name, None, value)
    return physics


def _line_dofs(physics, keep):
    """Scalar dof ids whose points satisfy keep(points (n, 3)) -> bool (n,)."""
    return np.flatnonzero(keep(physics.system.full.dof_points()))


def _modal_amplitude(physics, mode_shape, mode_norm):
    """(load weights per element (n_elements,) = int_e psi dV, a = int u_z psi dV / int psi^2 dV of the solved state)
    for the dimensionless mode shape psi(points (m, q, 3)) -> (m, q)."""
    from ...fem.assembly3d import assemble_lagrange_load_3d
    from ...fem.dofs3d import lagrange_dofs_3d
    full = physics.system.full

    def psi(points, elements):
        return mode_shape(points)

    weights = assemble_lagrange_load_3d(full, psi, degree=8)   # int N_i psi dV
    element_weights = assemble_lagrange_load_3d(lagrange_dofs_3d(full.mesh, 0), psi, degree=8)
    return element_weights, lambda: float(weights @ physics.system.unknown[:, 2]) / mode_norm


def _duffing_fit(physics, mode_shape, mode_norm, centroid_shape, stiffness, alpha, ratios):
    """Generalized force F = int f_z psi dV against the modal amplitude a over loads f_z = f0 psi(centroid) aimed at
    alpha a^2 / k = _LINEAR_RATIO (k = F / a) and ratios: least squares F / a - k = alpha a^2 + c a^4 -> (k, alpha,
    points; points[0] the linear-load sample)."""
    element_weights, amplitude = _modal_amplitude(physics, mode_shape, mode_norm)
    shape = centroid_shape(physics.domain.mesh)
    unit_force = float(element_weights @ shape)   # F of f0 = 1
    points = []
    for ratio in (_LINEAR_RATIO,) + tuple(ratios):
        target = math.sqrt(ratio * stiffness / alpha)
        force = stiffness * target * (1.0 + ratio)
        physics.inject_source('body_force_z', (force / unit_force) * shape)
        physics.step(0.0, math.inf)
        points.append((amplitude(), force, physics.last_converged))
    fitted_stiffness, fitted_alpha = pinned_duffing_fit(points[0], points[1:])
    return fitted_stiffness, fitted_alpha, points


def _centroids(mesh):
    return mesh.points[mesh.tets].mean(axis=1)


# ---- solid_fem3d_duffing_string (CPU): the Hirsch E4 string, L = 100 um, 100 x 100 nm, Si3N4 (E 250 GPa, nu 0.23),
# sigma0_xx = 1 GPa, P2 on 32 x 1 x 1 Kuhn cells, f_z = f0 sin(pi x / L). Supports on the neutral axis (y = z = h / 2):
# on both end faces u_z = 0 at the nodes of z = h / 2 and u_y = 0 at those of y = w / 2 (no twist), dead end tractions
# +-sigma0 that balance the initial stress (u = 0 at f0 = 0) and u_x = 0 at the neutral-axis nodes of the end faces only
# ('a', axial dofs free: the tension is uniform, alpha = E A pi^4 / (8 L^3) = 3.044e10 N/m^3, Kirchhoff-Carrier) or at
# every neutral-axis node ('b', no-slip: alpha = 1.5 x that = 4.566e10). Pinning the neutral axis (not whole faces) keeps
# the ends simply supported, so sin(pi x / L) stays the exact shape of the linear problem and bending only adds to k.
# Holding u_x on every node of the 3D string instead would lock the cross-section rotation and let the thickness change
# relieve the stretch (no 1.5 x): see Prescribed displacement (domain); holding u_z on the whole end face locks the
# rotation's eta (cos theta - 1) term (alpha + 4.6 % at 32 cells, growing with the element length). F = int f_z psi dV
# and the modal amplitude a = int u_z psi dV / int psi^2 dV (psi = sin(pi x / L)) satisfy F = k a + alpha a^3 + O(a^5)
# with the Galerkin alpha (the peak deflection would carry the third-harmonic shift: 8 / 9 of alpha for 'b'); k = F / a
# of a load at alpha a^2 / k = 1e-6, then a fit F / a - k = alpha a^2 + c a^4 over alpha a^2 / k = 0.01 .. 0.1 (a
# fitted k trades against alpha).
_STRING = (100.0 * UM, 0.1 * UM, 0.1 * UM)
_STRING_CELLS = 32
_SILICON_NITRIDE = (250e9, 0.23)
_STRING_STRESS = 1e9
_RATIOS = (0.01, 0.025, 0.05, 0.075, 0.1)
_LINEAR_RATIO = 1e-6            # alpha a^2 / k of the load that measures k
_DUFFING_TOLERANCE = 2e-2       # relative alpha (Hirsch plan E4)
_NONLINEAR_SETTINGS = {'geometric_nonlinearity': True, 'nonlinear_tolerance': 1e-10, 'nonlinear_max_iterations': 30}


def _string_alpha(no_slip):
    """Kirchhoff-Carrier E A pi^4 / (8 L^3) (x 1.5 with the axial dofs clamped) [N/m^3]."""
    length, width, height = _STRING
    return (1.5 if no_slip else 1.0) * _SILICON_NITRIDE[0] * width * height * math.pi ** 4 / (8.0 * length ** 3)


def _string_physics(no_slip, **settings):
    """The string physics of one variant (+ solver settings): material, balancing end tractions and the neutral-axis
    holds (no load)."""
    length, width, height = _STRING
    edges = (np.linspace(0.0, length, _STRING_CELLS + 1), np.array([0.0, width]), np.array([0.0, height]))
    youngs_modulus, poisson_ratio = _SILICON_NITRIDE
    physics = _direct_physics(edges, {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio,
                                      'initial_stress_xx': _STRING_STRESS}, **dict(_NONLINEAR_SETTINGS, **settings))
    for face, sign in (('x_lo', -1.0), ('x_hi', 1.0)):
        physics.add_face_traction(face, sign * _STRING_STRESS, 0.0, 0.0)
    tolerance = 1e-6 * height

    def end(points):
        return (points[:, 0] < tolerance) | (points[:, 0] > length - tolerance)

    physics.system.hold(_line_dofs(physics, lambda points: end(points) & (np.abs(points[:, 2] - 0.5 * height)
                                                                         < tolerance)), u_z=0.0)
    physics.system.hold(_line_dofs(physics, lambda points: end(points) & (np.abs(points[:, 1] - 0.5 * width)
                                                                         < tolerance)), u_y=0.0)

    def neutral(points):
        on_axis = (np.abs(points[:, 1] - 0.5 * width) < tolerance) & (np.abs(points[:, 2] - 0.5 * height) < tolerance)
        if no_slip:
            return on_axis
        return on_axis & end(points)

    held = _line_dofs(physics, neutral)
    if held.size < (2 * _STRING_CELLS + 1 if no_slip else 2):
        raise RuntimeError(f'string self-test: {held.size} neutral-axis dofs found (the Kuhn split puts one at every '
                           'face and cube diagonal midpoint)')
    physics.system.hold(held, u_x=0.0)
    return physics


def _string_stiffness():
    """Linear string stiffness k = sigma0 A pi^2 / (2 L) of the first mode [N/m]."""
    length, width, height = _STRING
    return _STRING_STRESS * width * height * math.pi ** 2 / (2.0 * length)


def _string_run(no_slip):
    """(k, alpha, points) of one string variant."""
    length, width, height = _STRING
    physics = _string_physics(no_slip)
    stiffness = _string_stiffness()

    def mode_shape(points):
        return np.sin(math.pi * points[..., 0] / length)

    return _duffing_fit(physics, mode_shape, 0.5 * width * height * length,
                        lambda mesh: np.sin(math.pi * _centroids(mesh)[:, 0] / length), stiffness,
                        _string_alpha(no_slip), _RATIOS)


def _check_string(_):
    errors, tolerances, details = {}, {}, {}
    for run, no_slip in (('axial free (a)', False), ('no-slip (b)', True)):
        stiffness, alpha, points = _string_run(no_slip)
        expected = _string_alpha(no_slip)
        errors[f'{run} alpha'], tolerances[f'{run} alpha'] = abs(alpha / expected - 1.0), _DUFFING_TOLERANCE
        converged = all(point[2] for point in points)
        errors[f'{run} converged'], tolerances[f'{run} converged'] = (0.0 if converged else math.inf), 1.0
        details[run] = {'alpha': alpha, 'alpha_expected': expected, 'k': stiffness,
                        'k_string': _STRING_STRESS * _STRING[1] * _STRING[2] * math.pi ** 2 / (2.0 * _STRING[0]),
                        'amplitudes': [point[0] for point in points], 'forces': [point[1] for point in points]}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_nonlinear_warm_start (CPU, plan C2): the axial-free string at alpha a^2 / k = 0.05 seeds (through
# warm_start_fields / seed_warm_start, as a sweep does) a new physics on the same mesh solving 0.1; its displacement
# equals a cold 0.1 solve within the Newton tolerance with fewer Newton iterations. A -5 u seed and a non-finite seed
# (the latter must fall back to the cold ramp) still match. 0.05 -> 0.1 is a strongly nonlinear step (a ~ +40 %).
# Helpers shared with the 2D twin: selftests_nonlinear (warm_start_runs / warm_start_errors).
def _apply_string_load(physics, ratio):
    """Solve the axial-free string under the sin(pi x / L) body force aiming at alpha a^2 / k = ratio."""
    length, width, height = _STRING
    unit_force_density = 0.5 * width * height * length   # F of f0 = 1 (load shape ~ the mode shape)
    shape = np.sin(math.pi * _centroids(physics.domain.mesh)[:, 0] / length)
    load = warm_string_load(ratio, _string_stiffness(), _string_alpha(False), unit_force_density)
    physics.inject_source('body_force_z', load * shape)
    physics.step(0.0, math.inf)


def _check_warm_start(_):
    runs = warm_start_runs(lambda: _string_physics(False, tangent_update=WARM_TANGENT_UPDATE), _apply_string_load,
                           string_displacement)
    errors, tolerances, details = warm_start_errors(runs)
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_tangent_update (CPU, plan G4): the axial-free string with load_steps 2 at alpha a^2 / k = 0.05 then
# 0.1, tangent_update every_iteration vs when_slow (modified Newton): equal states within the Newton tolerance, fewer
# tangent factorisations. Helpers shared with the 2D twin: selftests_nonlinear (tangent_update_runs / _errors).
def _check_tangent_update(_):
    runs = tangent_update_runs(lambda **settings: _string_physics(False, **settings), _apply_string_load,
                               string_displacement)
    errors, tolerances, details = tangent_update_errors(runs)
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_newton_gpu (plan G5): the axial-free string at alpha a^2 / k = _GPU_RATIO on the gpu backend with a
# cuDSS 'linear' solver (device TangentAssembler3D, device-value refactorisations) against the cpu SuperLU solve, with
# tangent_update every_iteration and when_slow: equal states within the Newton tolerance, a device assembler, and (full
# Newton) refactorisations on one cuDSS plan.
_GPU_RATIO = 0.05


def _check_newton_gpu(_):
    from ...solvers.sparse_direct import SparseDirectSolver
    skipped = skip_without_cudss()
    if skipped:
        return skipped
    length, width, height = _STRING
    stiffness, alpha = _string_stiffness(), _string_alpha(False)
    load = warm_string_load(_GPU_RATIO, stiffness, alpha, 0.5 * width * height * length)

    def solve(**settings):
        physics = _string_physics(False, **settings)
        physics.inject_source('body_force_z', load * np.sin(math.pi * _centroids(physics.domain.mesh)[:, 0] / length))
        physics.step(0.0, math.inf)
        return physics, string_displacement(physics)

    _, reference = solve()
    scale = float(np.max(np.abs(reference)))
    errors, tolerances, details = {}, {}, {}
    for tangent_update in ('every_iteration', 'when_slow'):
        physics, displacement = solve(backend='gpu', linear_solver=SparseDirectSolver('cudss', backend='gpu'),
                                      tangent_update=tangent_update)
        counters = physics.system.linear_solver.work_counters()
        assembler = physics._tangent_assembler
        errors[f'{tangent_update} displacement'] = float(np.max(np.abs(displacement - reference))) / scale
        tolerances[f'{tangent_update} displacement'] = WARM_TOLERANCE
        on_device = assembler is not None and assembler.device
        errors[f'{tangent_update} device assembler'] = 0.0 if on_device else math.inf
        errors[f'{tangent_update} converged'] = 0.0 if physics.last_converged else math.inf
        tolerances[f'{tangent_update} device assembler'] = tolerances[f'{tangent_update} converged'] = 1.0
        if tangent_update == 'every_iteration':   # one plan, every later Newton factor refreshed on the device
            refreshed = counters['cudss_plans'] == 1 and counters['factorizations'] > 1
            errors['every_iteration device refresh on one plan'] = 0.0 if refreshed else math.inf
            tolerances['every_iteration device refresh on one plan'] = 1.0
        details[tangent_update] = {'linear': counters, 'newton': physics.work_counters(),
                                   'device_bytes': assembler.device_bytes if on_device else None}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_duffing_membrane (CPU, heavy): the Hirsch E3 square, W = 80 um, h = 80 nm, Si3N4, equibiaxial sigma0
# = 0.67 GPa, P2 on 12 x 12 x 1 Kuhn cells, f_z = f0 sin(pi x / W) sin(pi y / W). Side faces: u_z = 0 at their mid-line
# nodes (z = h / 2; a whole-face hold locks the edge rotation as in the string) and dead tractions sigma0 n. 'b' (no-slip): u_x = u_y = 0 at every mid-plane node (z = h / 2): alpha_fixed = (10 / 64) pi^4 E h / ((1 -
# nu^2) W^2) = 5.022e13 N/m^3 (graded, 2 %); 'a' (physical): only the side faces' mid-line nodes pinned in-plane, alpha_a
# reported against the tutorial's alpha_54 = (9 / 64) pi^4 E h / W^2 (must lie below alpha_fixed). Amplitude and fit as
# the string.
_MEMBRANE = (80.0 * UM, 80.0 * UM, 0.08 * UM)
_MEMBRANE_CELLS = 12
_MEMBRANE_STRESS = 0.67e9


def _membrane_alpha():
    width, _, height = _MEMBRANE
    youngs_modulus, poisson_ratio = _SILICON_NITRIDE
    fixed = 10.0 / 64.0 * math.pi ** 4 * youngs_modulus * height / ((1.0 - poisson_ratio ** 2) * width ** 2)
    return fixed, 9.0 / 64.0 * math.pi ** 4 * youngs_modulus * height / width ** 2


def _membrane_run(no_slip):
    width, _, height = _MEMBRANE
    edges = (np.linspace(0.0, width, _MEMBRANE_CELLS + 1), np.linspace(0.0, width, _MEMBRANE_CELLS + 1),
             np.array([0.0, height]))
    youngs_modulus, poisson_ratio = _SILICON_NITRIDE
    physics = _direct_physics(edges, {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio,
                                      'initial_stress_xx': _MEMBRANE_STRESS, 'initial_stress_yy': _MEMBRANE_STRESS},
                              **_NONLINEAR_SETTINGS)
    for face, sign in (('x_lo', -1.0), ('x_hi', 1.0), ('y_lo', -1.0), ('y_hi', 1.0)):
        traction = sign * _MEMBRANE_STRESS
        physics.add_face_traction(face, traction if face[0] == 'x' else 0.0, traction if face[0] == 'y' else 0.0, 0.0)
    tolerance = 1e-6 * height

    def middle(points):
        return np.abs(points[:, 2] - 0.5 * height) < tolerance

    def edge(points):
        return ((points[:, 0] < tolerance) | (points[:, 0] > width - tolerance) | (points[:, 1] < tolerance)
                | (points[:, 1] > width - tolerance))

    physics.system.hold(_line_dofs(physics, lambda points: middle(points) & edge(points)), u_z=0.0)
    in_plane = middle if no_slip else (lambda points: middle(points) & edge(points))
    physics.system.hold(_line_dofs(physics, in_plane), u_x=0.0, u_y=0.0)
    stiffness = _MEMBRANE_STRESS * height * math.pi ** 2 / 2.0
    fixed, _ = _membrane_alpha()

    def mode_shape(points):
        return np.sin(math.pi * points[..., 0] / width) * np.sin(math.pi * points[..., 1] / width)

    def centroid_shape(mesh):
        centres = _centroids(mesh)
        return np.sin(math.pi * centres[:, 0] / width) * np.sin(math.pi * centres[:, 1] / width)

    return _duffing_fit(physics, mode_shape, 0.25 * width * width * height, centroid_shape, stiffness, fixed, _RATIOS)


def _check_membrane(_):
    fixed, tutorial = _membrane_alpha()
    errors, tolerances, details = {}, {}, {}
    _, alpha_b, points_b = _membrane_run(True)
    errors['no-slip (b) alpha'], tolerances['no-slip (b) alpha'] = abs(alpha_b / fixed - 1.0), _DUFFING_TOLERANCE
    _, alpha_a, points_a = _membrane_run(False)
    converged = all(point[2] for point in points_a + points_b)
    errors['converged'], tolerances['converged'] = (0.0 if converged else math.inf), 1.0
    details.update(alpha_b=alpha_b, alpha_fixed=fixed, alpha_a=alpha_a, alpha_54=tutorial,
                   alpha_a_over_alpha_54=alpha_a / tutorial, alpha_a_below_fixed=alpha_a < fixed,
                   amplitudes_b=[point[0] for point in points_b], amplitudes_a=[point[0] for point in points_a])
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem3d_energy_density (CPU): a 1 x 0.75 x 0.5 um roller box (P1, 4 x 3 x 2 cells, E 70 GPa, nu 0.25, rho
# 2200), mode 1 shown: P1 strains are constant per element, so the centroid strain_energy_density times the element
# volumes sums exactly to u^T K u / 4 (= omega^2 u^T M u / 4 by the eigen equation), and the nodal kinetic density
# (the mean over the elements at a dof of rho omega^2 |u_dof|^2 / 4, one material) is exact at every dof.
_ENERGY_SIZE = (1.0 * UM, 0.75 * UM, 0.5 * UM)
_ENERGY_CELLS = (4, 3, 2)
_ENERGY_MATERIAL = (70e9, 0.25, 2200.0)
_ENERGY_TOLERANCE = 1e-6   # float32 per-element state arrays (first estimate)


def _check_energy(_):
    from ...fem.assembly3d import assemble_lagrange_load_3d
    from ...fem.dofs3d import lagrange_dofs_3d
    from ...fem.domain3d import TetrahedronDomain
    from ...fem.mesh3d import structured_box_mesh
    from ...core.constants import FACES_3D
    from .solver_fem3d import SolidMechanicsFEMPhysics3D
    edges = [np.linspace(0.0, side, cells + 1) for side, cells in zip(_ENERGY_SIZE, _ENERGY_CELLS)]
    domain = TetrahedronDomain(structured_box_mesh(*edges), backend='cpu')
    physics = SolidMechanicsFEMPhysics3D(domain, order=1)
    youngs_modulus, poisson_ratio, density = _ENERGY_MATERIAL
    for name, value in (('youngs_modulus', youngs_modulus), ('poisson_ratio', poisson_ratio), ('rho', density)):
        domain.materials.assign(name, None, value)
    for face in FACES_3D:
        physics.hold_face(face, **{f'u_{face[0]}': 0.0})
    physics.finalize()
    solve = physics.solve_eigenfrequencies(2)
    physics.select_resonance(1)
    resonance = solve.resonances[0]
    mode = np.asarray(resonance.displacement, dtype=np.float64)
    omega = float(resonance.omega)
    stiffness_energy = 0.25 * float(mode @ (physics.system.operator @ mode))
    kinetic_energy = 0.25 * omega ** 2 * float(mode @ (physics.modal_mass @ mode))
    volumes = assemble_lagrange_load_3d(lagrange_dofs_3d(domain.mesh, 0), 1.0)
    state = physics.state()
    strain_total = float(np.asarray(state['strain_energy_density'], dtype=np.float64) @ volumes)
    _, kinetic_nodal = physics.lagrange_field('kinetic_energy_density')
    displacement = mode.reshape(-1, 3)
    active = physics.system.active
    expected_nodal = 0.25 * float32_value(density) * omega ** 2 * np.sum(displacement ** 2, axis=1)
    nodal_error = float(np.max(np.abs(kinetic_nodal - expected_nodal)[active])) / float(np.max(expected_nodal))
    errors = {'strain energy integral': abs(strain_total / stiffness_energy - 1.0),
              'virial (eigen pencil)': abs(kinetic_energy / stiffness_energy - 1.0),
              'nodal kinetic density': nodal_error}
    tolerances = {name: _ENERGY_TOLERANCE for name in errors}
    details = {'strain_energy_integral': strain_total, 'quarter_u_K_u': stiffness_energy,
               'quarter_omega2_u_M_u': kinetic_energy, 'frequency': resonance.frequency}
    return scaled_worst_result(errors, tolerances, **details)


NONLINEAR_3D_CASES = (
    SelfTestCase(
        name='solid_fem3d_nonlinear_uniform',
        label='Solid mechanics 3D on tetrahedra: geometrically nonlinear homogeneous states (stretch, release, domain '
              'hold)',
        description=('A 1 x 0.5 x 0.5 um block (P1, 0.25 um cells, E 100 GPa, nu 0.3) with geometric_nonlinearity and '
                     'load_steps 2, rollers x_lo / y_lo / z_lo: a 10 % stretch by a Displacement on x_hi (Green-Lagrange '
                     'E_xx = (l^2 - 1) / 2, E_yy = -nu E_xx, S_xx = E E_xx, the nominal reaction l S_xx A of the internal '
                     f'force), a released initial_stress_xx = {_UNIFORM_STRESS:g} Pa (S = 0, E = -C^-1 S0) and a '
                     'Prescribed displacement (domain) u_x = 0 on the whole block holding that stress (u = 0, S_xx = '
                     'S0); probes u, strain, stress, strain / kinetic energy density within '
                     f'{_UNIFORM_TOLERANCE:g} of their scales (exact in P1), the reaction within '
                     f'{_REACTION_TOLERANCE:g}, and an empty domain hold refused by validation. measured = error in '
                     'tolerance units, tolerance 1 (first estimates, not yet run).'),
        specs=_uniform_specs, check=_check_uniform),
    SelfTestCase(
        name='solid_fem3d_duffing_string',
        label='Solid mechanics 3D on tetrahedra: geometric Duffing coefficient of a prestressed string (Hirsch E4)',
        description=('The Hirsch E4 string (100 um x 100 x 100 nm Si3N4, sigma0 1 GPa, P2 on 32 x 1 x 1 Kuhn cells, '
                     'CPU, no model) under f_z = f0 sin(pi x / L), supports on the mid-lines of the end faces only, '
                     'geometrically nonlinear static solves at a linear load (k) and five loads: the fit F / a - k = '
                     'alpha a^2 + c a^4 of the generalized force against the modal amplitude gives alpha = E A pi^4 / '
                     '(8 L^3) = 3.044e10 N/m^3 with axial dofs free (Kirchhoff-Carrier) and 1.5 x that with the neutral '
                     f'axis held axially (no-slip), within {_DUFFING_TOLERANCE:g}. measured = error in tolerance units, '
                     'tolerance 1.'),
        specs=dict, check=_check_string),
    SelfTestCase(
        name='solid_fem3d_nonlinear_warm_start',
        label='Solid mechanics 3D on tetrahedra: sweep warm start of the nonlinear Newton solve',
        description=('The axial-free Hirsch E4 string of solid_fem3d_duffing_string (CPU, no model, full Newton): a solve at alpha '
                     f'a^2 / k = {WARM_RATIOS[0]:g} hands warm_start_fields to seed_warm_start of a new physics on the '
                     f'same mesh solving {WARM_RATIOS[1]:g} (the sweep warm start path); its displacement equals a '
                     f'cold solve within {WARM_TOLERANCE:g} of max |u| with fewer Newton iterations (work_counters), '
                     'the start text names the sweep seed, a -5 u seed still matches and a non-finite seed falls back '
                     'to the cold ramp and matches. measured = error in tolerance units, tolerance 1 (first estimate, '
                     'not yet run).'),
        specs=dict, check=_check_warm_start),
    SelfTestCase(
        name='solid_fem3d_tangent_update',
        label='Solid mechanics 3D on tetrahedra: modified Newton (tangent_update when_slow) of the nonlinear solve',
        description=('The axial-free Hirsch E4 string of solid_fem3d_duffing_string (CPU, no model, load_steps '
                     f'{TANGENT_LOAD_STEPS}) solved at alpha a^2 / k = {TANGENT_RATIOS[0]:g} then '
                     f'{TANGENT_RATIOS[1]:g} with tangent_update every_iteration and when_slow: the when_slow '
                     f'displacement equals full Newton within {WARM_TOLERANCE:g} of max |u| with fewer '
                     'tangent_factorizations and some tangent_reuses (work_counters); full Newton factorises once per '
                     'iteration; both converge. measured = error in tolerance units, tolerance 1 (first estimate, not '
                     'yet run).'),
        specs=dict, check=_check_tangent_update),
    SelfTestCase(
        name='solid_fem3d_newton_gpu',
        label='Solid mechanics 3D on tetrahedra: Newton tangents assembled and refactorised on the GPU',
        description=('The axial-free Hirsch E4 string of solid_fem3d_duffing_string (no model) at alpha a^2 / k = '
                     f'{_GPU_RATIO:g} on the gpu backend with a cuDSS linear solver (device TangentAssembler3D, '
                     'CudssFactorization.refactorize_device_values), tangent_update every_iteration and when_slow: the '
                     f'displacement equals the cpu SuperLU solve within {WARM_TOLERANCE:g} of max |u|, the assembler '
                     'runs on the device, both converge, and full Newton refactorises on a single cuDSS plan. Passes '
                     'noted without nvmath-python. measured = error in tolerance units, tolerance 1 (first estimate, '
                     'not yet run).'),
        specs=dict, check=_check_newton_gpu, backends=('gpu',)),
    SelfTestCase(
        name='solid_fem3d_duffing_membrane',
        label='Solid mechanics 3D on tetrahedra: geometric Duffing coefficient of a square membrane (Hirsch E4)',
        description=('The Hirsch E3 square (80 um, h 80 nm Si3N4, sigma0 0.67 GPa equibiaxial, P2 on 12 x 12 x 1 Kuhn '
                     'cells, CPU, no model) under f_z = f0 sin sin, side faces held in u_z on their mid-lines only: '
                     'with the mid-plane held in-plane (no-slip) the '
                     'fitted alpha = alpha_fixed = (10 / 64) pi^4 E h / ((1 - nu^2) W^2) = 5.022e13 N/m^3 within '
                     f'{_DUFFING_TOLERANCE:g}; the physical variant (only the edges pinned) is reported with alpha_a / '
                     'alpha_54. measured = error in tolerance units, tolerance 1 (first estimate, not yet run).'),
        specs=dict, check=_check_membrane, heavy=True),
    SelfTestCase(
        name='solid_fem3d_energy_density',
        label='Solid mechanics 3D on tetrahedra: time-averaged strain / kinetic energy densities of a mode',
        description=('A 1 x 0.75 x 0.5 um roller box (P1, 4 x 3 x 2 cells, CPU, no model), mode 1 shown: the element '
                     'sum of strain_energy_density x volume = u^T K u / 4 (P1 strains are constant per element) = '
                     'omega^2 u^T M u / 4, and the nodal kinetic_energy_density = rho omega^2 |u|^2 / 4 at every dof, '
                     f'within {_ENERGY_TOLERANCE:g}. measured = error in tolerance units, tolerance 1 (first estimate, '
                     'not yet run).'),
        specs=dict, check=_check_energy),
)
