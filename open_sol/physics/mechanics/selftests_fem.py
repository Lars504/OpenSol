"""Self-tests of solid_mechanics on triangle meshes (mesh plan stage T5): CPU elasticity patch / operator checks
(fem.elasticity, system_fem, check({})), and model runs against closed forms: the Kirsch hole, a heated two-material
cylinder (planar with the temperature from heat on a tensor mesh, and axisymmetric), the vibration modes of an
orthotropic roller box, generalized plane strain (uniform and bending), the elastic frequency response of a bar, and
the cuDSS eigen factors against SuperLU (GPU). No physics imports at module level."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import SubChecks, checks_result, cpu_defaults, float32_value, scaled_worst_result
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, PhysicsSpec, RefinementSpec, StudySpec)
from .selftests import _elastic, _lame, _rectangle
from .selftests_formulations import (_BEND_AXIAL_FORCE, _BEND_FIRST_MOMENT, _BEND_INTERFACE, _BEND_LENGTH, _BEND_WIDTH,
                                     _GPS_AXIAL_FORCE, _GPS_INTERFACE_X, _GPS_LAYERS, _GPS_RISE, _GPS_SIZE,
                                     _gps_closed_forms)

_PHYSICS = 'solid'
_SOLID_TYPE = 'solid_mechanics'                 # = selftests.SOLID_TYPE_NAME
_HEAT_TYPE = 'heat'
_THERMAL_STRAIN = 'heat_solid_thermal_strain'   # = selftests.THERMAL_STRAIN_COUPLING
_TRIANGLE_MESH = 'triangles'


def _triangle_mesh_spec(dx, curvature_factor=0.3, refinements=(), h_min=None):
    """Triangle MeshSpec; every size (curvature, refinements) is floored at h_min (None: dx / 100)."""
    return MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='nonuniform', dx=dx, curvature_factor=curvature_factor,
                    refinements=list(refinements), h_min=h_min)


# ---- solid_fem_patch (CPU) ----
# 2 x 2 um window with a disc (curved interior edges), uniform h 0.3 um. Planar: a monoclinic C, linear displacement
# (P1, P2 isoparametric) and a quadratic one with its constant body force (P2 on straight elements) with every
# boundary dof held: the discrete solution and the strains are exact. Axisymmetric (x = r from 0): u_r = a r, u_z = b z
# + c with an orthotropic hoop C and the body force f_r = -(sigma_rr - sigma_tt) / r that keeps it in equilibrium.
# ElasticFEMSystem: free plane-strain thermal expansion (rollers on x_lo / y_lo) is exact, its reaction 0; a support
# set without u_y raises. K annihilates the rigid modes; the mass totals rho x area (ring volume).
_UM = 1e-6
_PATCH_SIZE = 2.0 * _UM
_PATCH_H = 0.3 * _UM
_DISPLACEMENT = 1e-9          # m, scale of the patch displacements
_CURVATURE = 0.7              # quadratic term of u_x / _DISPLACEMENT in (x / 1 um)^2 / 2
_MONOCLINIC = {'c11': 2.0e11, 'c12': 0.6e11, 'c22': 1.5e11, 'c66': 0.5e11, 'c16': 0.2e11, 'c26': -0.1e11}
_ORTHOTROPIC_HOOP = {'c11': 2.0e11, 'c12': 0.6e11, 'c13': 0.7e11, 'c22': 1.8e11, 'c23': 0.5e11, 'c33': 1.6e11,
                     'c66': 0.5e11}
_EXACT_RTOL = 1e-9            # direct solves of exactly representable fields
_RIGID_RTOL = 1e-12
_MASS_RTOL = 1e-12
_PATCH_DENSITY = 2500.0


def _patch_mesh():
    from ...model.triangle_meshing import triangulate

    spec = ModelSpec(title='Self-test: elasticity patch', domain=DomainSpec(size_x=_PATCH_SIZE, size_y=_PATCH_SIZE),
                     geometry=[GeometrySpec('disc', 'circle', {'cx': 1.0 * _UM, 'cy': 1.0 * _UM, 'r': 0.5 * _UM})],
                     meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_PATCH_H)])
    return triangulate(spec, {})


def _solve_held(stiffness, load, dofs, exact):
    """Full solution of stiffness u = load with every boundary dof (both components) held at exact."""
    from scipy.sparse.linalg import spsolve

    from ...fem.assembly import eliminate_dirichlet

    boundary = dofs.edge_dofs(dofs.boundary_edges())
    fixed = (2 * boundary[:, None] + np.arange(2)).ravel()
    system = eliminate_dirichlet(stiffness, load, fixed, exact.reshape(-1)[fixed])
    return system.expand(spsolve(system.matrix.tocsc(), system.rhs)).reshape(-1, 2)


def _planar_exact(points, quadratic):
    """(n, 2) u = 1e-9 m (3 X - 2 Y [+ 0.35 X^2], X + 4 Y), X = x / 1 um."""
    x, y = points[:, 0] / _UM, points[:, 1] / _UM
    u_x = 3.0 * x - 2.0 * y + (0.5 * _CURVATURE * x * x if quadratic else 0.0)
    return _DISPLACEMENT * np.column_stack([u_x, x + 4.0 * y])


def _planar_patch(mesh, order, geometry_order, quadratic):
    """(nodal error, centroid strain error, recovered strain error) relative, of the planar monoclinic patch."""
    from ...fem.assembly import nodal_average
    from ...fem.dofs import lagrange_dofs
    from ...fem.elasticity import (CENTROID, assemble_body_load, assemble_elastic_stiffness, element_strains,
                                   local_dof_points)

    dofs = lagrange_dofs(mesh, order, geometry_order=geometry_order)
    coefficients = {name: np.full(mesh.n_elements, value) for name, value in _MONOCLINIC.items()}
    stiffness = assemble_elastic_stiffness(dofs, coefficients)
    exact = _planar_exact(dofs.dof_points(), quadratic)
    load = np.zeros(2 * dofs.n_dofs)
    if quadratic:   # -div(C eps): d sigma_xx / dx = c11 u'' and d sigma_xy / dx = c16 u''
        second = _DISPLACEMENT * _CURVATURE / _UM ** 2
        load = assemble_body_load(dofs, -_MONOCLINIC['c11'] * second, -_MONOCLINIC['c16'] * second)
    solution = _solve_held(stiffness, load, dofs, exact)
    nodal_error = float(np.abs(solution - exact).max() / np.abs(exact).max())
    if quadratic:
        return nodal_error, 0.0, 0.0
    strain_scale = _DISPLACEMENT / _UM
    expected = np.array([3.0, 4.0, -1.0]) * strain_scale   # (xx, yy, g = -2 + 1)
    _, centre = element_strains(dofs, solution, CENTROID)
    _, local = element_strains(dofs, solution, local_dof_points(order))
    recovered = nodal_average(dofs, local)
    return (nodal_error, float(np.abs(centre - expected).max() / strain_scale),
            float(np.abs(recovered - expected).max() / strain_scale))


def _axisymmetric_patch(mesh, order, geometry_order):
    """Relative nodal error of u_r = a r, u_z = b z + c (orthotropic hoop C, body force f_r = -(s_rr - s_tt) / r)."""
    from ...fem.dofs import lagrange_dofs
    from ...fem.elasticity import assemble_body_load, assemble_elastic_stiffness

    dofs = lagrange_dofs(mesh, order, geometry_order=geometry_order)
    coefficients = {name: np.full(mesh.n_elements, value) for name, value in _ORTHOTROPIC_HOOP.items()}
    stiffness = assemble_elastic_stiffness(dofs, coefficients, axisymmetric=True)
    radial, axial = 2.0 * _DISPLACEMENT / _UM, -_DISPLACEMENT / _UM
    points = dofs.dof_points()
    exact = np.column_stack([radial * points[:, 0], axial * points[:, 1] + 0.5 * _DISPLACEMENT])
    c = _ORTHOTROPIC_HOOP
    stress_rr = c['c11'] * radial + c['c12'] * axial + c['c13'] * radial
    stress_tt = c['c13'] * radial + c['c23'] * axial + c['c33'] * radial

    def radial_force(quadrature_points, elements):
        return -(stress_rr - stress_tt) / quadrature_points[..., 0]

    load = assemble_body_load(dofs, radial_force, 0.0, axisymmetric=True)
    solution = _solve_held(stiffness, load, dofs, exact)
    return float(np.abs(solution - exact).max() / np.abs(exact).max())


def _isotropic_coefficients(n_elements, youngs_modulus, poisson_ratio):
    mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    values = {'c11': lame_lambda + 2.0 * mu, 'c22': lame_lambda + 2.0 * mu, 'c33': lame_lambda + 2.0 * mu,
              'c12': lame_lambda, 'c13': lame_lambda, 'c23': lame_lambda, 'c66': mu}
    return {name: np.full(n_elements, value) for name, value in values.items()}


def _free_expansion(mesh, add):
    """ElasticFEMSystem: plane-strain thermal expansion with rollers on x_lo / y_lo (alpha_zz != alpha) is the uniform
    strain eps = C_inplane^-1 beta dT; the roller reaction is 0; holding u_x only raises."""
    from ...fem.elasticity import assemble_stress_load
    from .system_fem import ElasticFEMSystem

    coefficients = _isotropic_coefficients(mesh.n_elements, 1.0e11, 0.3)
    alpha, alpha_zz, temperature_change = 2e-6, 5e-6, 50.0
    c = {name: float(values[0]) for name, values in coefficients.items()}
    beta = np.array([c['c11'] * alpha + c['c12'] * alpha + c['c13'] * alpha_zz,
                     c['c12'] * alpha + c['c22'] * alpha + c['c23'] * alpha_zz])
    strain = np.linalg.solve(np.array([[c['c11'], c['c12']], [c['c12'], c['c22']]]), beta * temperature_change)
    system = ElasticFEMSystem(mesh, 2, False, lambda: 'patch system')
    left = system.hold(system.global_ids_of_edges(system.face_edges('x_lo')), u_x=0.0)
    system.hold(system.global_ids_of_edges(system.face_edges('y_lo')), u_y=0.0)
    system.assemble(coefficients, np.ones(mesh.n_elements, dtype=bool))
    stress = np.zeros((mesh.n_elements, 3))
    stress[:, :2] = beta * temperature_change
    system.solve(assemble_stress_load(system.dofs, stress))
    points = system.dofs.dof_points()
    exact = points * strain
    error = float(np.abs(system.unknown - exact).max() / np.abs(exact).max())
    add('system_free_expansion', error <= _EXACT_RTOL, error)
    force_x, force_y = system.reaction(left)
    reaction = max(abs(force_x), abs(force_y)) / (float(beta.max()) * temperature_change * _PATCH_SIZE)
    add('system_free_expansion_reaction', reaction <= _EXACT_RTOL, reaction)
    unsupported = ElasticFEMSystem(mesh, 1, False, lambda: 'patch system')
    unsupported.hold(unsupported.global_ids_of_edges(unsupported.face_edges('x_lo')), u_x=0.0)
    try:
        unsupported.assemble(coefficients, np.ones(mesh.n_elements, dtype=bool))
        add('system_rigid_mode_check', False, 'no error for a solid without a held u_y')
    except ValueError as exc:
        add('system_rigid_mode_check', 'slide along y' in str(exc), str(exc))


def _check_patch(results_by_name):
    from ...fem.dofs import lagrange_dofs
    from ...fem.elasticity import RING, assemble_elastic_mass, assemble_elastic_stiffness

    checks = SubChecks()
    add = checks.add
    mesh = _patch_mesh()
    add('mesh_has_curved_edges', bool((mesh.edge_curve >= 0).any()), int((mesh.edge_curve >= 0).sum()))
    for name, order, geometry_order, quadratic in (('p1_linear', 1, 1, False), ('p2_linear_curved', 2, 2, False),
                                                   ('p2_quadratic_body_force', 2, 1, True)):
        nodal, centre, recovered = _planar_patch(mesh, order, geometry_order, quadratic)
        add(f'{name}_displacement', nodal <= _EXACT_RTOL, nodal)
        if not quadratic:
            add(f'{name}_centroid_strain', centre <= _EXACT_RTOL, centre)
            add(f'{name}_recovered_strain', recovered <= _EXACT_RTOL, recovered)
    for name, order, geometry_order in (('axisymmetric_p1', 1, 1), ('axisymmetric_p2_curved', 2, 2)):
        error = _axisymmetric_patch(mesh, order, geometry_order)
        add(name, error <= _EXACT_RTOL, error)
    _free_expansion(mesh, add)

    dofs = lagrange_dofs(mesh, 2)
    stiffness = assemble_elastic_stiffness(dofs, {name: np.full(mesh.n_elements, value)
                                                  for name, value in _MONOCLINIC.items()})
    points = dofs.dof_points()
    scale = float(abs(stiffness).sum(axis=1).max())
    for name, mode in (('translation_x', np.column_stack([np.ones(len(points)), np.zeros(len(points))])),
                       ('translation_y', np.column_stack([np.zeros(len(points)), np.ones(len(points))])),
                       ('rotation', np.column_stack([-points[:, 1], points[:, 0]]) / _UM)):
        residual = float(np.abs(stiffness @ mode.reshape(-1)).max() / (scale * np.abs(mode).max()))
        add(f'rigid_{name}', residual <= _RIGID_RTOL, residual)
    asymmetry = float(abs(stiffness - stiffness.T).max() / abs(stiffness).max())
    add('stiffness_symmetric', asymmetry <= _RIGID_RTOL, asymmetry)
    ones_x = np.zeros(2 * dofs.n_dofs)
    ones_x[0::2] = 1.0
    for name, axisymmetric, expected in (
            ('mass_total_planar', False, _PATCH_DENSITY * _PATCH_SIZE ** 2),
            ('mass_total_axisymmetric', True, _PATCH_DENSITY * RING * _PATCH_SIZE * _PATCH_SIZE ** 2 / 2.0)):
        mass = assemble_elastic_mass(dofs, _PATCH_DENSITY, axisymmetric=axisymmetric)
        error = abs(float(ones_x @ (mass @ ones_x)) / expected - 1.0)
        add(name, error <= _MASS_RTOL, error)
    return checks_result(checks)


# ---- solid_fem_kirsch: quarter plate (side L = 100 a) with a hole of radius a at the corner, plane stress, rollers on
# x_lo / y_lo, uniform tension sigma on x_hi. Kirsch: hoop stress 3 sigma at (0, a) (stress_xx there), -sigma at (a, 0)
# (stress_yy); the finite plate shifts them by O((a / L)^2) ~ 1e-4. Hole arcs at h = 0.02 a (curvature factor), growth
# 1.3 to h_max = L / 10. The support reactions are exact: F_x of the x_lo roller = -sigma L, F_y of the y_lo one = 0.
_HOLE_RADIUS = 1e-3
_PLATE_SIDE = 100.0 * _HOLE_RADIUS
_TENSION = 1e6
_KIRSCH_CURVATURE = 0.02
_KIRSCH_STRESS_TOLERANCE = 1e-3     # relative to sigma (first estimate: P2 node-averaged stress, h / a = 0.02)
_PROBE_TOLERANCE = 1e-6             # probe (float32 sample) = the derived nodal value
_REACTION_TOLERANCE = 1e-9


def _kirsch_specs():
    a, side = _HOLE_RADIUS, _PLATE_SIDE
    return {'plate': ModelSpec(
        title='Self-test: Kirsch plate with a hole on a triangle mesh (solid mechanics)',
        domain=DomainSpec(size_x=side, size_y=side),
        geometry=[GeometrySpec('hole', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': a}),
                  GeometrySpec('top', 'point', {'x': 0.0, 'y': a}), GeometrySpec('side', 'point', {'x': a, 'y': 0.0})],
        materials=[MaterialSpec('steel', properties={'youngs_modulus': 200e9, 'poisson_ratio': 0.3}, selection=['all']),
                   MaterialSpec('hole_void', properties={'youngs_modulus': 0.0}, selection=['hole'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings={'out_of_plane': 'plane_stress'},
                             features=[FeatureSpec('roller_left', 'roller', ['x_lo']),
                                       FeatureSpec('roller_bottom', 'roller', ['y_lo']),
                                       FeatureSpec('tension', 'boundary_load', ['x_hi'], {'t_x': _TENSION, 't_y': 0.0}),
                                       FeatureSpec('probe_top', 'solid_probe', ['top'], {'component': 'stress_xx'})])],
        meshes=[_triangle_mesh_spec(side / 10.0, _KIRSCH_CURVATURE,
                                    [RefinementSpec('top', _KIRSCH_CURVATURE * a),
                                     RefinementSpec('side', _KIRSCH_CURVATURE * a)],
                                    h_min=0.5 * _KIRSCH_CURVATURE * a)],   # the default dx / 100 = 0.1 a floors h
        solver_defaults=cpu_defaults(),   # host SuperLU either way: the gpu backend only adds device copies
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('hoop_top', 'stress_xx', [], 'at', point='top'),
                        DerivedValueSpec('hoop_side', 'stress_yy', [], 'at', point='side')])}


def _check_kirsch(results_by_name):
    results = results_by_name['plate']
    derived = results.summary()['derived']
    hoop_top, hoop_side = float(derived['hoop_top']), float(derived['hoop_side'])
    probe = results.result(_PHYSICS, 'probe_top')['values']
    probe_value = float(probe[-1]) if len(probe) else math.nan
    left, bottom = results.result(_PHYSICS, 'roller_left'), results.result(_PHYSICS, 'roller_bottom')
    total = _TENSION * _PLATE_SIDE
    errors = {'hoop_top': abs(hoop_top - 3.0 * _TENSION) / _TENSION,
              'hoop_side': abs(hoop_side + _TENSION) / _TENSION,
              'probe_top': abs(probe_value - hoop_top) / abs(hoop_top),
              'reaction_x': abs(float(left['F_x']) + total) / total,
              'reaction_y': abs(float(bottom['F_y'])) / total}
    tolerances = {'hoop_top': _KIRSCH_STRESS_TOLERANCE, 'hoop_side': _KIRSCH_STRESS_TOLERANCE,
                  'probe_top': _PROBE_TOLERANCE, 'reaction_x': _REACTION_TOLERANCE, 'reaction_y': _REACTION_TOLERANCE}
    details = {'hoop_top': hoop_top, 'hoop_side': hoop_side, 'probe_top': probe_value, 'F_x_left': left['F_x'],
               'F_y_bottom': bottom['F_y'], 'expected': {'hoop_top': 3.0 * _TENSION, 'hoop_side': -_TENSION,
                                                        'F_x_left': -total}}
    return scaled_worst_result(errors, tolerances, **details)


# ---- heated two-material cylinder, plane strain: core r < a (1), shell a < r < b (2), uniform dT, free outer surface.
# Interface pressure p = ((1 + nu1) alpha1 - (1 + nu2) alpha2) dT / ((1 + nu1)(1 - 2 nu1) / E1 + (1 + nu2) / E2 ((1 -
# nu2)(b^2 + a^2) / (b^2 - a^2) + nu2)); core sigma_rr = sigma_tt = -p; shell hoop at b = 2 p a^2 / (b^2 - a^2);
# u(a) = a ((1 + nu1) alpha1 dT - (1 + nu1)(1 - 2 nu1) p / E1), u(b) = b (1 + nu2) ((1 - nu2) sigma_tt(b) / E2 + alpha2
# dT). Planar ('two_meshes'): a quarter disc (rollers on x_lo / y_lo) in a 1.25 b window (void outside), dT from a heat
# physics on its own tensor mesh; axisymmetric: the (r, z) section of a long cylinder with rollers on both ends (eps_zz =
# 0: the same plane-strain solution), dT from a temperature_change feature.
_CORE_RADIUS = 0.4e-3
_SHELL_RADIUS = 1.0e-3
_CORE = {'youngs_modulus': 100e9, 'poisson_ratio': 0.3, 'alpha': 2e-5}
_SHELL = {'youngs_modulus': 200e9, 'poisson_ratio': 0.25, 'alpha': 1e-5}
_HEATING = 100.0                     # K
_CYLINDER_H = _SHELL_RADIUS / 16.0
_CYLINDER_CURVATURE = 0.05
_DISPLACEMENT_TOLERANCE = 1e-4       # relative (P2 isoparametric: the shell's 1 / r field is not polynomial)
_CYLINDER_STRESS_TOLERANCE = 2e-3    # relative to p (node-averaged stress, first estimate)
_REFERENCE_TEMPERATURE = 293.15      # heat T0 default
_CONDUCTIVITY = 10.0


def _cylinder_closed_form():
    """{u_inner, u_outer [m], hoop_outer [Pa] (sigma_tt at b), core_stress [Pa] (= -p)}."""
    a, b, d_t = _CORE_RADIUS, _SHELL_RADIUS, _HEATING
    e1, nu1, alpha1 = _CORE['youngs_modulus'], _CORE['poisson_ratio'], _CORE['alpha']
    e2, nu2, alpha2 = _SHELL['youngs_modulus'], _SHELL['poisson_ratio'], _SHELL['alpha']
    ratio = (b * b + a * a) / (b * b - a * a)
    compliance = (1.0 + nu1) * (1.0 - 2.0 * nu1) / e1 + (1.0 + nu2) / e2 * ((1.0 - nu2) * ratio + nu2)
    pressure = ((1.0 + nu1) * alpha1 - (1.0 + nu2) * alpha2) * d_t / compliance
    hoop_outer = 2.0 * pressure * a * a / (b * b - a * a)
    return {'u_inner': a * ((1.0 + nu1) * alpha1 * d_t - (1.0 + nu1) * (1.0 - 2.0 * nu1) * pressure / e1),
            'u_outer': b * (1.0 + nu2) * ((1.0 - nu2) * hoop_outer / e2 + alpha2 * d_t),
            'hoop_outer': hoop_outer, 'core_stress': -pressure}


def _cylinder_derived(hoop_field):
    return [DerivedValueSpec('u_inner', 'u_x', [], 'at', point='inner'),
            DerivedValueSpec('u_outer', 'u_x', [], 'at', point='outer'),
            DerivedValueSpec('hoop_outer', hoop_field, [], 'at', point='outer'),
            DerivedValueSpec('core_stress', 'stress_xx', [], 'at', point='centre')]


def _planar_cylinder_spec():
    """Quarter disc in a 1.25 b window, dT from a heat physics on a tensor mesh (all faces held at T0 + dT) through
    heat_solid_thermal_strain."""
    a, b = _CORE_RADIUS, _SHELL_RADIUS
    window = 1.25 * b
    conduction = {'k': _CONDUCTIVITY}
    materials = [MaterialSpec('air', properties={'k': _CONDUCTIVITY}, selection=['all']),
                 MaterialSpec('shell', properties={**_SHELL, **conduction}, selection=['shell']),
                 MaterialSpec('core', properties={**_CORE, **conduction}, selection=['core'])]
    physics = [PhysicsSpec('heat', _HEAT_TYPE, features=[FeatureSpec(
                   'hot_faces', 'temperature', ['x_lo', 'x_hi', 'y_lo', 'y_hi'],
                   {'temperature': _REFERENCE_TEMPERATURE + _HEATING})]),
               PhysicsSpec(_PHYSICS, _SOLID_TYPE, mesh=_TRIANGLE_MESH, features=[
                   FeatureSpec('roller_left', 'roller', ['x_lo']), FeatureSpec('roller_bottom', 'roller', ['y_lo'])])]
    meshes = [MeshSpec(name='grid', mode='uniform', dx=window / 25.0),
              _triangle_mesh_spec(_CYLINDER_H, _CYLINDER_CURVATURE)]
    couplings = [CouplingSpec('expansion', _THERMAL_STRAIN, 'heat', _PHYSICS)]
    return ModelSpec(
        title='Self-test: heated two-material cylinder on triangles (solid mechanics, heat on a tensor mesh)',
        domain=DomainSpec(size_x=window, size_y=window),
        geometry=[GeometrySpec('shell', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': b}),
                  GeometrySpec('core', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': a}),
                  GeometrySpec('inner', 'point', {'x': a, 'y': 0.0}),
                  GeometrySpec('outer', 'point', {'x': b, 'y': 0.0}),
                  GeometrySpec('centre', 'point', {'x': 0.3 * a, 'y': 0.3 * a})],
        materials=materials, physics=physics, couplings=couplings, meshes=meshes, solver_defaults=cpu_defaults(),
        study=StudySpec(type='stationary'), derived_values=_cylinder_derived('stress_yy'))


def _axisymmetric_cylinder_spec():
    """(r, z) section of a long two-material cylinder, height b / 2, rollers on both ends."""
    a, b = _CORE_RADIUS, _SHELL_RADIUS
    height = 0.5 * b
    return ModelSpec(
        title='Self-test: heated two-material cylinder, axisymmetric, on triangles (solid mechanics)',
        domain=DomainSpec(size_x=b, size_y=height),
        geometry=[GeometrySpec('core', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': a, 'y1': height}),
                  GeometrySpec('inner', 'point', {'x': a, 'y': 0.5 * height}),
                  GeometrySpec('outer', 'point', {'x': b, 'y': 0.5 * height}),
                  GeometrySpec('centre', 'point', {'x': a / 3.0, 'y': 0.5 * height})],
        materials=[MaterialSpec('shell', properties=dict(_SHELL), selection=['all']),
                   MaterialSpec('core', properties=dict(_CORE), selection=['core'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings={'geometry': 'axisymmetric'},
                             features=[FeatureSpec('ends', 'roller', ['y_lo', 'y_hi']),
                                       FeatureSpec('heating', 'temperature_change', ['all'], {'delta_T': _HEATING})])],
        meshes=[_triangle_mesh_spec(_CYLINDER_H)], solver_defaults=cpu_defaults(), study=StudySpec(type='stationary'),
        derived_values=_cylinder_derived('stress_zz'))


def _cylinder_specs():
    return {'two_meshes': _planar_cylinder_spec(), 'axisymmetric': _axisymmetric_cylinder_spec()}


def _cylinder_errors(results, run, errors, tolerances, details):
    """Add the four closed-form comparisons of one run (displacements relative to themselves, stresses to p)."""
    expected = _cylinder_closed_form()
    derived = results.summary()['derived']
    scale = abs(expected['core_stress'])
    measured = {name: float(derived[name]) for name in expected}
    for name, value in measured.items():
        reference = expected[name]
        if name.startswith('u_'):
            errors[f'{run} {name}'] = abs(value / reference - 1.0)
            tolerances[f'{run} {name}'] = _DISPLACEMENT_TOLERANCE
        else:
            errors[f'{run} {name}'] = abs(value - reference) / scale
            tolerances[f'{run} {name}'] = _CYLINDER_STRESS_TOLERANCE
    details[run] = {'measured': measured, 'expected': expected}


def _check_cylinder(results_by_name):
    errors, tolerances, details = {}, {}, {}
    for run, results in results_by_name.items():
        _cylinder_errors(results, run, errors, tolerances, details)
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem_roller_box_modes: 10 x 4 mm block, rollers on all faces, triangles h = 0.25 mm (P2). Continuum modes
# u_x = A sin(kx x) cos(ky y), u_y = B cos(kx x) sin(ky y), kx = p pi / Lx, ky = q pi / Ly: per (p, q) the 2 x 2
# problem [[c11 kx^2 + c66 ky^2, (c12 + c66) kx ky], [., c22 ky^2 + c66 kx^2]] = omega^2 rho (u_x for p >= 1, u_y for
# q >= 1). One orthotropic run (c11, c12, c22, c66 only: the general operator; the E / nu path on triangles is pinned by
# kirsch and the heated cylinder). The 6 lowest frequencies, sorted, within 1e-4 (P2 eigenvalues converge as h^4).
_BOX = (10e-3, 4e-3)
_BOX_H = 0.25e-3
_BOX_DENSITY = 2200.0
_BOX_ORTHOTROPIC = {'c11': 1.2e11, 'c12': 0.3e11, 'c22': 0.8e11, 'c66': 0.35e11}
_N_MODES = 6
_MODE_TOLERANCE = 1e-4
_TABLE = 'modes'


def _box_frequencies(properties):
    """Sorted lowest continuum frequencies [Hz] of the roller box of c_ij properties."""
    c11, c12, c22, c66 = (properties[name] for name in ('c11', 'c12', 'c22', 'c66'))
    frequencies = []
    for p in range(13):
        for q in range(7):
            kx, ky = p * math.pi / _BOX[0], q * math.pi / _BOX[1]
            present = [axis for axis, order in ((0, p), (1, q)) if order >= 1]
            if not present:
                continue
            symbol = np.array([[c11 * kx * kx + c66 * ky * ky, (c12 + c66) * kx * ky],
                               [(c12 + c66) * kx * ky, c22 * ky * ky + c66 * kx * kx]])
            values = np.linalg.eigvalsh(symbol[np.ix_(present, present)]) / _BOX_DENSITY
            frequencies.extend(math.sqrt(value) / (2.0 * math.pi) for value in values)
    return sorted(frequencies)[:_N_MODES]


def _box_spec(properties, label):
    return ModelSpec(
        title=f'Self-test: vibration modes of a roller-supported block on triangles ({label})',
        domain=DomainSpec(size_x=_BOX[0], size_y=_BOX[1]),
        materials=[MaterialSpec('block', properties={**properties, 'rho': _BOX_DENSITY}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE,
                             features=[FeatureSpec('rollers', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
                                       FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_BOX_H)],
        solver_defaults=cpu_defaults(),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _N_MODES}))


def _box_specs():
    return {'orthotropic': _box_spec(_BOX_ORTHOTROPIC, 'orthotropic c_ij')}


def _check_box(results_by_name):
    results = results_by_name['orthotropic']
    expected = _box_frequencies(_BOX_ORTHOTROPIC)
    rows = results.result(_PHYSICS, _TABLE)['rows']
    measured = sorted(float(row[1]) for row in rows)
    if len(measured) < len(expected):
        raise ValueError(f'orthotropic: expected {len(expected)} modes in the table, got {len(measured)}')
    labels = results.summary().get('snapshot_labels') or []
    errors = {'orthotropic frequency': max(abs(value / reference - 1.0) for value, reference in zip(measured, expected)),
              'orthotropic snapshots': abs(len(labels) - len(expected))}
    tolerances = {'orthotropic frequency': _MODE_TOLERANCE, 'orthotropic snapshots': 0.5}
    details = {'orthotropic': {'frequency_expected': expected, 'frequency_measured': measured,
                               'snapshot_labels': labels}}
    return scaled_worst_result(errors, tolerances, **details)


# ---- solid_fem_generalized_plane_strain: the selftests_formulations layers (40 x 20 um, layer a x < 16 um) on P2
# triangles h = 1 um, rollers on x_lo / y_lo / y_hi, x_hi free, 100 K. 'uniform' (bending none, axial_force F): u_x is
# piecewise linear, exact in P1 / P2, so the closed form of solid_generalized_plane_strain holds to round-off. 'bending'
# (bending xy, F, first moment int x sigma_zz dA = Q, M_x = F times the mid-height: e2 = 0): eps_xx = (beta_x dT -
# lambda eps_zz) / L is linear in x per layer, u_x quadratic: exact in P2, and the continuum conditions int sigma_zz dA
# = F, int x sigma_zz dA = Q fix (e0, e1) of eps_zz = e0 + e1 x. Points away from the interface (nodal averaging smears
# stresses across it), probes at the loaded end; the transverse roller reaction F_y(top) = int sigma_yy dx.
_GPS_H = 1e-6
_GPS_POINTS = {'a_left': 4e-6, 'a_right': 12e-6, 'b_left': 24e-6, 'b_right': 36e-6, 'end': 40e-6}
_GPS_FEM_TOLERANCE = 1e-6     # exact fields: SuperLU round-off and nodal averaging of exact values (first estimate)
_GPS_FIELDS = ('stress_zz', 'stress_yy', 'elastic_strain_zz', 'u_x')


def _gps_fem_spec(run):
    """Triangle ModelSpec of run 'uniform' or 'bending'."""
    length, height = _GPS_SIZE
    materials = [MaterialSpec(name, properties=_elastic(youngs, nu, alpha_xx=alpha_xx, alpha_yy=alpha_yy,
                                                        alpha_zz=alpha_zz), selection=[region])
                 for name, region, ((youngs, nu), (alpha_xx, alpha_yy, alpha_zz))
                 in zip(('layer_a', 'layer_b'), ('left_part', 'right_part'), _GPS_LAYERS)]
    settings = {'out_of_plane': 'generalized_plane_strain'}
    if run == 'uniform':
        settings.update(axial_force=_GPS_AXIAL_FORCE, bending='none')
    else:
        settings.update(axial_force=_BEND_AXIAL_FORCE, bending='xy', bending_moment_x=0.5 * height * _BEND_AXIAL_FORCE,
                        bending_moment_y=-_BEND_FIRST_MOMENT)
    points = [GeometrySpec(name, 'point', {'x': x, 'y': 0.5 * height}) for name, x in _GPS_POINTS.items()]
    derived = [DerivedValueSpec(f'{field}_{name}', field, [], 'at', point=name)
               for name in _GPS_POINTS for field in _GPS_FIELDS]
    return ModelSpec(
        title=f'Self-test: generalized plane strain on triangles, {run} (solid mechanics)',
        domain=DomainSpec(size_x=length, size_y=height),
        geometry=[_rectangle('left_part', 0.0, 0.0, _GPS_INTERFACE_X, height),
                  _rectangle('right_part', _GPS_INTERFACE_X, 0.0, length, height)] + points,
        materials=materials,
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings=settings, features=[
            FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
            FeatureSpec('top', 'roller', ['y_hi']),
            FeatureSpec('heating', 'temperature_change', ['all'], {'delta_T': _GPS_RISE})])],
        meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_GPS_H)],
        solver_defaults=cpu_defaults(),
        study=StudySpec(type='stationary'), derived_values=derived)


def _gps_fem_specs():
    return {'uniform': _gps_fem_spec('uniform'), 'bending': _gps_fem_spec('bending')}


def _layer_constants():
    """Per layer (lambda, L = lambda + 2 mu, beta_x, beta_y, beta_z, alpha_zz) of the GPS layers."""
    constants = []
    for (youngs, nu), (alpha_xx, alpha_yy, alpha_zz) in _GPS_LAYERS:
        lame_lambda, mu = _lame(youngs, nu)
        longitudinal = lame_lambda + 2.0 * mu
        constants.append((lame_lambda, longitudinal, longitudinal * alpha_xx + lame_lambda * (alpha_yy + alpha_zz),
                          longitudinal * alpha_yy + lame_lambda * (alpha_xx + alpha_zz),
                          longitudinal * alpha_zz + lame_lambda * (alpha_xx + alpha_yy), alpha_zz))
    return constants


def _bending_continuum():
    """(e0, e1) of eps_zz = e0 + e1 x and the closed-form fields {field: f(x)} of the 'bending' run, plus F_y(top)."""
    layers = ((0.0, _BEND_INTERFACE), (_BEND_INTERFACE, _BEND_LENGTH))
    constants = _layer_constants()
    matrix, right_hand = np.zeros((2, 2)), np.array([_BEND_AXIAL_FORCE, _BEND_FIRST_MOMENT]) / _BEND_WIDTH
    for (low, high), (lame_lambda, longitudinal, beta_x, _, beta_z, _) in zip(layers, constants):
        moments = [(high ** (p + 1) - low ** (p + 1)) / (p + 1) for p in range(3)]   # int x^p dx over the layer
        reduced = longitudinal - lame_lambda ** 2 / longitudinal
        offset = (beta_z - lame_lambda * beta_x / longitudinal) * _GPS_RISE
        matrix += reduced * np.array([[moments[0], moments[1]], [moments[1], moments[2]]])
        right_hand += offset * np.array(moments[:2])
    e0, e1 = np.linalg.solve(matrix, right_hand)

    def strain_xx(index, x):
        lame_lambda, longitudinal, beta_x, _, _, _ = constants[index]
        return (beta_x * _GPS_RISE - lame_lambda * (e0 + e1 * x)) / longitudinal

    def stress_yy(index, x):
        lame_lambda, _, _, beta_y, _, _ = constants[index]
        return lame_lambda * (strain_xx(index, x) + e0 + e1 * x) - beta_y * _GPS_RISE

    def fields(x):
        index = 0 if x < _BEND_INTERFACE else 1
        lame_lambda, longitudinal, _, _, beta_z, alpha_zz = constants[index]
        strain_zz = e0 + e1 * x
        u_x = sum(0.5 * (strain_xx(layer, low) + strain_xx(layer, min(high, x))) * (min(high, x) - low)
                  for layer, (low, high) in enumerate(layers) if x > low)   # eps_xx linear per layer: trapezoids
        return {'stress_zz': lame_lambda * strain_xx(index, x) + longitudinal * strain_zz - beta_z * _GPS_RISE,
                'stress_yy': stress_yy(index, x), 'elastic_strain_zz': strain_zz - alpha_zz * _GPS_RISE, 'u_x': u_x}
    top_force = sum(0.5 * (stress_yy(layer, low) + stress_yy(layer, high)) * (high - low)
                    for layer, (low, high) in enumerate(layers))   # int sigma_yy dx, linear per layer
    return (e0, e1), fields, top_force


def _uniform_fields():
    """Closed-form fields {field: f(x)} of the 'uniform' run (solid_generalized_plane_strain) and F_y(top)."""
    axial_strain, layers, widths = _gps_closed_forms()
    (strain_a, stress_yy_a, stress_zz_a, alpha_a), (strain_b, stress_yy_b, stress_zz_b, alpha_b) = layers

    def fields(x):
        left = x < _GPS_INTERFACE_X
        u_x = strain_a * x if left else strain_a * _GPS_INTERFACE_X + strain_b * (x - _GPS_INTERFACE_X)
        return {'stress_zz': stress_zz_a if left else stress_zz_b, 'stress_yy': stress_yy_a if left else stress_yy_b,
                'elastic_strain_zz': axial_strain - (alpha_a if left else alpha_b) * _GPS_RISE, 'u_x': u_x}
    return axial_strain, fields, stress_yy_a * widths[0] + stress_yy_b * widths[1]


def _check_gps_fem(results_by_name):
    """Both runs: the derived point values and F_y of the top roller against the closed forms (stresses relative to
    the largest closed-form stress, strains to |eps_zz|, u_x to its value at the end)."""
    errors, details = {}, {}
    for run in ('uniform', 'bending'):
        results = results_by_name[run]
        derived = results.summary()['derived']
        if run == 'uniform':
            terms, fields, top_force = _uniform_fields()
        else:
            terms, fields, top_force = _bending_continuum()
        expected = {name: fields(x) for name, x in _GPS_POINTS.items()}
        stress_scale = max(abs(values[field]) for values in expected.values() for field in ('stress_zz', 'stress_yy'))
        scales = {'stress_zz': stress_scale, 'stress_yy': stress_scale,
                  'elastic_strain_zz': max(abs(values['elastic_strain_zz']) for values in expected.values()),
                  'u_x': abs(expected['end']['u_x'])}
        for name, values in expected.items():
            for field in _GPS_FIELDS:
                errors[f'{run} {field} {name}'] = abs(float(derived[f'{field}_{name}']) - values[field]) / scales[field]
        top_y = float(results.result(_PHYSICS, 'top')['F_y'])
        errors[f'{run} F_top'] = abs(top_y - top_force) / (stress_scale * _GPS_SIZE[0])
        details[run] = {'expected': expected, 'eps_zz_terms': terms, 'F_top': top_y, 'F_top_expected': top_force}
    return scaled_worst_result(errors, {item: _GPS_FEM_TOLERANCE for item in errors}, **details)


# ---- solid_fem_frequency_response: a 10 x 1 mm bar on P2 triangles h = 0.125 mm, rollers on y_lo / y_hi and x_lo,
# traction t on x_hi, mechanical_loss_factor eta, plane strain. 1D continuum: u = t sin(k x) / (c k cos(k L)), c = c11
# (1 + j eta), k = omega sqrt(rho / c) (exp(+j omega t)); runs 'below' (0.5 f_1) and 'above' (1.5 f_1), f_1 = sqrt(c11
# / rho) / (4 L). Re u_x and u_abs = |u| at x = L / 2 and L against it (P2 error ~ (k h)^3: first-estimate tolerance).
_BAR = (10e-3, 1e-3)
_BAR_H = 0.125e-3
_BAR_YOUNGS = float32_value(200e9)
_BAR_POISSON = float32_value(0.3)
_BAR_DENSITY = float32_value(7800.0)
_BAR_LOSS = float32_value(2e-3)
_BAR_TRACTION = 1e6
_BAR_FACTORS = {'below': 0.5, 'above': 1.5}
_BAR_POINTS = {'middle': 0.5, 'end': 1.0}   # x / L
_BAR_TOLERANCE = 1e-4         # relative to |u(L)| (first estimate: P2 on h = L / 80)


def _bar_modulus():
    lame_lambda, mu = _lame(_BAR_YOUNGS, _BAR_POISSON)
    return lame_lambda + 2.0 * mu


def _bar_first_frequency():
    return math.sqrt(_bar_modulus() / _BAR_DENSITY) / (4.0 * _BAR[0])


def _bar_displacement(frequency, x):
    """Complex continuum u_x(x) of the driven lossy bar."""
    modulus = _bar_modulus() * (1.0 + 1j * _BAR_LOSS)
    wavenumber = 2.0 * math.pi * frequency * np.sqrt(_BAR_DENSITY / modulus)
    return _BAR_TRACTION * np.sin(wavenumber * x) / (modulus * wavenumber * np.cos(wavenumber * _BAR[0]))


def _bar_spec(run):
    length, width = _BAR
    points = [GeometrySpec(name, 'point', {'x': fraction * length, 'y': 0.5 * width})
              for name, fraction in _BAR_POINTS.items()]
    derived = [DerivedValueSpec(f'{field}_{name}', field, [], 'at', point=name)
               for name in _BAR_POINTS for field in ('u_x', 'u_abs')]
    return ModelSpec(
        title=f'Self-test: driven lossy bar on triangles, {run} (solid mechanics frequency response)',
        domain=DomainSpec(size_x=length, size_y=width), geometry=points,
        materials=[MaterialSpec('bar', properties={**_elastic(_BAR_YOUNGS, _BAR_POISSON), 'rho': _BAR_DENSITY,
                                                   'mechanical_loss_factor': _BAR_LOSS}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, features=[
            FeatureSpec('rails', 'roller', ['y_lo', 'y_hi']), FeatureSpec('anchor', 'roller', ['x_lo']),
            FeatureSpec('push', 'boundary_load', ['x_hi'], {'t_x': _BAR_TRACTION, 't_y': 0.0})])],
        meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_BAR_H)],
        solver_defaults=cpu_defaults(),
        study=StudySpec(type='frequency_response',
                        settings={'frequencies': [_BAR_FACTORS[run] * _bar_first_frequency()]}),
        derived_values=derived)


def _bar_specs():
    return {run: _bar_spec(run) for run in _BAR_FACTORS}


def _check_bar(results_by_name):
    errors, details = {}, {}
    for run, factor in _BAR_FACTORS.items():
        frequency = factor * _bar_first_frequency()
        derived = results_by_name[run].summary()['derived']
        scale = abs(_bar_displacement(frequency, _BAR[0]))
        details[run] = {'frequency_Hz': frequency}
        for name, fraction in _BAR_POINTS.items():
            exact = _bar_displacement(frequency, fraction * _BAR[0])
            measured = {field: float(derived[f'{field}_{name}']) for field in ('u_x', 'u_abs')}
            errors[f'{run} u_x {name}'] = abs(measured['u_x'] - exact.real) / scale
            errors[f'{run} u_abs {name}'] = abs(measured['u_abs'] - abs(exact)) / scale
            details[run][name] = {'measured': measured, 'expected_re': float(exact.real),
                                  'expected_abs': float(abs(exact))}
    return scaled_worst_result(errors, {item: _BAR_TOLERANCE for item in errors}, **details)


# ---- solid_fem_modes_cudss: the roller-box pencil (P2 triangles, h = 0.1 mm) by eigsh_shifted with factorization
# superlu, cudss (real LDL^T, double) and cudss_mixed (float32 LDL^T refined to 1e-12), at 0 and at a target between the
# 4th and 5th continuum modes (K - sigma M indefinite, default |diag|^-1/2 scaling); GPU, passes noted without nvmath.
_CUDSS_BOX_H = 0.1e-3
_CUDSS_EIGEN_TOLERANCE = 1e-9   # relative eigenvalue difference (both refined to full accuracy; first estimate)


def _check_modes_cudss(_):
    """Largest relative eigenvalue difference of cudss / cudss_mixed to superlu over both shifts; details record the
    factor timers, device bytes, single_residual and refinement steps."""
    from ...model.triangle_meshing import triangulate
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from ...solvers.cudss_factor import cudss_available
    from .system_fem import ElasticFEMSystem
    if not cudss_available():
        return {'measured': 0.0, 'expected': 0.0, 'error': 0.0, 'tolerance': _CUDSS_EIGEN_TOLERANCE,
                'details': {'skipped': 'nvmath-python (cuDSS) is not installed'}}
    spec = ModelSpec(title='Self-test: cuDSS roller box mesh', domain=DomainSpec(size_x=_BOX[0], size_y=_BOX[1]),
                     meshes=[MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=_CUDSS_BOX_H)])
    mesh = triangulate(spec, {})
    system = ElasticFEMSystem(mesh, 2, False, lambda: 'cuDSS self-test')
    for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'):
        system.hold(system.global_ids_of_edges(system.face_edges(face)), **{f'u_{face[0]}': 0.0})
    system.assemble({name: np.full(mesh.n_elements, value) for name, value in _BOX_ORTHOTROPIC.items()},
                    np.ones(mesh.n_elements, dtype=bool))
    free = np.flatnonzero(system.free)
    stiffness = system.operator[free][:, free].tocsc()
    mass = system.mass(np.full(mesh.n_elements, _BOX_DENSITY))[free][:, free].tocsc()
    start = np.random.default_rng(6).standard_normal(free.size)
    continuum = _box_frequencies(_BOX_ORTHOTROPIC)
    shifts = {'lowest': 0.0, 'target': (math.pi * (continuum[3] + continuum[4])) ** 2}   # omega_t^2, f_t = mid 4 / 5
    errors, details = {}, {'free_dofs': int(free.size), 'continuum_Hz': continuum}
    for shift_name, shift in shifts.items():
        eigenvalues = {}
        for factorization in ('superlu', 'cudss', 'cudss_mixed'):
            solver = ArpackShiftInvertSolver(factorization=factorization)
            values, _, applications, converged, factor = solver.eigsh_shifted(stiffness, mass, shift, _N_MODES,
                                                                              start_vector=start)
            eigenvalues[factorization] = np.sort(values)
            details[f'{shift_name} {factorization}'] = {
                'converged': bool(converged), 'applications': int(applications),
                'factor': getattr(factor, 'description', 'SuperLU'), 'host_bytes': int(factor.host_bytes or 0),
                **{name: getattr(factor, name) for name in ('device_bytes', 'plan_seconds', 'factorize_seconds',
                                                            'factor_seconds', 'single_residual', 'probe_residual',
                                                            'refinement_steps', 'symmetric', 'solve_seconds',
                                                            'n_solves') if hasattr(factor, name)}}
            if hasattr(factor, 'release'):
                factor.release()
        reference = eigenvalues['superlu']
        for factorization in ('cudss', 'cudss_mixed'):
            errors[f'{shift_name} {factorization}'] = float(np.max(np.abs(eigenvalues[factorization] / reference - 1.0)))
        details[f'{shift_name} frequencies_Hz'] = (np.sqrt(reference) / (2.0 * math.pi)).tolist()
    return scaled_worst_result(errors, {item: _CUDSS_EIGEN_TOLERANCE for item in errors}, **details)


FEM_CASES = (
    SelfTestCase(
        name='solid_fem_patch',
        label='Solid mechanics on triangles: patch tests and operator checks',
        description='CPU (no run) on a 2 x 2 um triangle mesh with a disc (curved interior edges): fem.elasticity with '
                    'every boundary dof held at the exact field reproduces a linear displacement with a monoclinic C '
                    '(P1, P2 isoparametric; centroid and node-averaged strains exact), a quadratic one with its '
                    'constant body force (P2 straight) and, axisymmetric (x = r from 0), u_r = a r, u_z = b z + c with '
                    'an orthotropic hoop C and the body force -(sigma_rr - sigma_tt) / r (P1, P2 curved), all to '
                    f'{_EXACT_RTOL:g}; physics.mechanics.system_fem: plane-strain free thermal expansion (alpha_zz != '
                    'alpha) on rollers exact with zero reaction, a support set without u_y refused (rigid-mode check); '
                    f'K symmetric and annihilating both translations and the rotation ({_RIGID_RTOL:g}); the consistent '
                    f'mass totals rho x area and rho x ring volume ({_MASS_RTOL:g}). error = number of failed '
                    'sub-checks (tolerance 0), values in details.',
        specs=lambda: {},
        check=_check_patch),
    SelfTestCase(
        name='solid_fem_kirsch',
        label='Solid mechanics on triangles: Kirsch plate with a hole',
        description=f'Quarter plate of side {_PLATE_SIDE * 1e3:g} mm with a hole of radius {_HOLE_RADIUS * 1e3:g} mm '
                    'at its corner (a void material), plane stress, rollers on x_lo / y_lo, tension '
                    f'{_TENSION:g} Pa on x_hi; triangles with h = {_KIRSCH_CURVATURE:g} a along the hole (curvature '
                    'factor and point sizes), growth 1.3 to L / 10, P2 isoparametric. Kirsch: the hoop stress is 3 '
                    'sigma at the top of the hole (stress_xx at (0, a)) and -sigma at its side (stress_yy at (a, 0)); the '
                    f'finite plate shifts it by O((a / L)^2) ~ 1e-4; tolerance {_KIRSCH_STRESS_TOLERANCE:g} of sigma '
                    '(first estimate). A solid_probe of stress_xx at the top matches the derived value '
                    f'({_PROBE_TOLERANCE:g}), and the roller reactions are exact by equilibrium (F_x of x_lo = -sigma L, '
                    f'F_y of y_lo = 0; {_REACTION_TOLERANCE:g}). cpu backend (the solve is a host SuperLU on either; '
                    'the last gpu run took 58 s, 29 s of it in the study; heavy flag to re-time). measured = error = the '
                    'worst deviation in tolerance units. Not yet run on the cpu backend.',
        specs=_kirsch_specs,
        check=_check_kirsch,
        heavy=True),
    SelfTestCase(
        name='solid_fem_heated_cylinder',
        label='Solid mechanics on triangles: heated two-material cylinder',
        description='Core r < a = 0.4 mm (E 100 GPa, nu 0.3, alpha 2e-5) in a shell a < r < b = 1 mm (E 200 GPa, nu '
                    f'0.25, alpha 1e-5), dT = {_HEATING:g} K, free outer surface, plane strain; run two_meshes (planar '
                    'quarter disc in a 1.25 b window, void outside, rollers on x_lo / y_lo, curved P2 edges; dT from a '
                    'heat physics on its own uniform tensor mesh, all four faces held at T0 + dT, so T is uniform, '
                    f'through {_THERMAL_STRAIN} across the meshes: value transfer, interpolated onto the elements, '
                    'exact for a constant) and axisymmetric (the (r, z) section with rollers on both ends: eps_zz = 0, '
                    'the same solution; dT by temperature_change); cpu backend, triangles h = b / 16. Closed form '
                    f'(Lame): interface pressure p, u(a), u(b) (tolerance {_DISPLACEMENT_TOLERANCE:g} relative), the '
                    'hoop stress at b = 2 p a^2 / (b^2 - a^2) and the core stress sigma_rr = -p (at (0.3 a, 0.3 a) / (a '
                    f'/ 3, h / 2); tolerance {_CYLINDER_STRESS_TOLERANCE:g} of p: node-averaged stresses, first '
                    'estimate). measured = error = the worst deviation in tolerance units. It holds the former '
                    'solid_fem_thermal_strain_two_mesh (its run two_meshes, same mesh, closed form and tolerances; '
                    'the former planar temperature_change run solved the same quarter disc). Not yet run as this '
                    'merged pair.',
        specs=_cylinder_specs,
        check=_check_cylinder),
    SelfTestCase(
        name='solid_fem_roller_box_modes',
        label='Solid mechanics on triangles: vibration modes of a roller box',
        description=f'Eigenfrequency study of a {_BOX[0] * 1e3:g} x {_BOX[1] * 1e3:g} mm block with rollers on all '
                    f'faces on triangles of h = {_BOX_H * 1e3:g} mm (P2), rho {_BOX_DENSITY:g}; run orthotropic (c11, '
                    'c12, c22, c66 only: the stiffness object without youngs_modulus; the isotropic E / nu path on '
                    'triangles is pinned by solid_fem_kirsch and solid_fem_heated_cylinder); cpu backend. Continuum '
                    'modes sin / cos per order pair (p, q), the 2 x 2 symbol [[c11 kx^2 + c66 ky^2, (c12 + c66) kx '
                    f'ky], [., c22 ky^2 + c66 kx^2]] = omega^2 rho: the {_N_MODES} lowest table frequencies (sorted) '
                    f'within {_MODE_TOLERANCE:g} relative (P2 eigenvalue error ~ h^4) and one snapshot per mode. '
                    'measured = error = the worst deviation in tolerance units.',
        specs=_box_specs,
        check=_check_box),
    SelfTestCase(
        name='solid_fem_generalized_plane_strain',
        label='Solid mechanics on triangles: generalized plane strain, uniform and bending',
        description='The heated layers of solid_generalized_plane_strain (40 x 20 um, layer a below x = 16 um, rollers '
                    f'on x_lo / y_lo / y_hi, x_hi free, {_GPS_RISE:g} K) on P2 triangles of h = {_GPS_H * 1e6:g} um, '
                    f'cpu backend. Run uniform (bending none, axial_force {_GPS_AXIAL_FORCE:g} N): u_x piecewise '
                    'linear, the closed form of the tensor case exactly; run bending (bending xy, axial force '
                    f'{_BEND_AXIAL_FORCE:g} N, int x sigma_zz dA = {_BEND_FIRST_MOMENT:g} N m via bending_moment_y, '
                    'bending_moment_x = F H / 2 so e2 = 0): eps_xx linear and u_x quadratic per layer, exact in P2, '
                    'against the continuum (e0, e1) of the two conditions. Derived stress_zz, stress_yy, '
                    'elastic_strain_zz and u_x at x = 4, 12, 24, 36, 40 um (mid-height; away from the interface, where '
                    'nodal averaging smears stresses) and F_y of the top roller = int sigma_yy dx, within '
                    f'{_GPS_FEM_TOLERANCE:g} of their scales (first estimate). Pins the unit-term loads, the condition '
                    'quadrature, the moment signs about the drawing origin and the eps_zz outputs. measured = error = '
                    'the worst deviation in tolerance units. Not yet run.',
        specs=_gps_fem_specs,
        check=_check_gps_fem),
    SelfTestCase(
        name='solid_fem_frequency_response',
        label='Solid mechanics on triangles: driven lossy bar (frequency response)',
        description=f'frequency_response runs of a {_BAR[0] * 1e3:g} x {_BAR[1] * 1e3:g} mm bar on P2 triangles of h '
                    f'= {_BAR_H * 1e3:g} mm (rollers on y_lo / y_hi and x_lo, traction {_BAR_TRACTION:g} Pa on x_hi, '
                    f'mechanical_loss_factor {_BAR_LOSS:g}, plane strain, float32-exact constants), cpu backend, at '
                    f"{', '.join(f'{factor:g}' for factor in _BAR_FACTORS.values())} x f_1 = sqrt(c11 / rho) / (4 L) "
                    '(runs below / above): derived u_x (real part) and u_abs at x = L / 2 and L against the continuum '
                    'u = t sin(k x) / (c k cos(k L)), c = c11 (1 + j eta), k = omega sqrt(rho / c), within '
                    f'{_BAR_TOLERANCE:g} of |u(L)| (P2 error ~ (k h)^3; first estimate). measured = error = the worst '
                    'deviation in tolerance units. Not yet run.',
        specs=_bar_specs,
        check=_check_bar),
    SelfTestCase(
        name='solid_fem_modes_cudss',
        label='Solid mechanics on triangles: cuDSS eigen factor vs SuperLU',
        description=f'GPU, no model run: the {_BOX[0] * 1e3:g} x {_BOX[1] * 1e3:g} mm orthotropic roller-box pencil '
                    f'(P2 triangles of h = {_CUDSS_BOX_H * 1e3:g} mm, physics.mechanics.system_fem) by eigsh_shifted '
                    f'for the {_N_MODES} eigenvalues nearest 0 and nearest a target between the 4th and 5th continuum '
                    'modes (K - sigma M indefinite) with factorization superlu, cudss (real LDL^T, double) and '
                    f'cudss_mixed (float32 LDL^T refined to 1e-12 per solve): within {_CUDSS_EIGEN_TOLERANCE:g} of '
                    'SuperLU (first estimate). details record plan / factorise seconds, device bytes, single_residual '
                    'and refinement steps (the triangle answer to the 1e-3 gate of the mixed factor). Passes noted '
                    'without nvmath-python. Not yet run.',
        specs=dict, check=_check_modes_cudss, backends=('gpu',)),
)
