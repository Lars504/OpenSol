"""Self-tests of the elastic PML ring of the 2D triangle solid and its energy-density outputs (new_modules_plan K5): the
stretched operator identities (CPU, no model), the radiation Q of a leaky Fabry-Perot bar in plane strain against its
closed form, and the static / modal energy densities; definition registers them."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cpu_defaults, scaled_worst_result, worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests_fem import _BOX, _BOX_DENSITY, _BOX_H, _BOX_ORTHOTROPIC

UM = 1e-6
_SOLID = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME
_TABLE = 'modes'


# ---- solid_fem_pml_operator: the stretched matrices of a strip with an x_hi ring (CPU, no model) ----
# 1 x 0.25 um window of 0.125 um structured right triangles (P2) plus a 0.5 um ring beyond x_hi (fem selftests_harmonic
# _rectangle_mesh), isotropic (E 100 GPa, nu 0.3, rho 2000): the stretched K, M and K_sigma complex symmetric (transpose,
# not conjugate), the window-only block real, a unit stretch equal to the plain K, s = 1 at the window side and 1 - j
# beta at the outer side (beta = 3 ln(1 / R) c / (2 omega w), R 1e-6, c = sqrt(c11 / rho), f 1 GHz), the ring-only dofs
# disjoint from the window; the same symmetry for the axisymmetric (r~) and the azimuthal order 2 forms.
_STEP = 0.125 * UM
_WINDOW = (1.0 * UM, 0.25 * UM)
_RING = 0.5 * UM
_ELASTIC = (100e9, 0.3, 2000.0)
_OPERATOR_TOLERANCE = 1e-12


def _check_operator(_):
    """Symmetry, real window block, unit stretch, profile ends, ring-only dofs (planar, axisymmetric, order 2)."""
    from ...fem.elasticity import StressField, assemble_elastic_stiffness
    from ...fem.pml import DEFAULT_PML_REFLECTION
    from ...fem.selftests_harmonic import _rectangle_mesh
    from .pml2d import elastic_pml_ring
    from .selftests_fem import _isotropic_coefficients
    from .system_fem import ElasticFEMSystem
    length, height = _WINDOW
    x_edges = np.arange(0.0, length + _RING + 0.5 * _STEP, _STEP) + 0.5 * UM   # r > 0 for the axisymmetric forms
    y_edges = np.arange(0.0, height + 0.5 * _STEP, _STEP)
    window = (x_edges[0], 0.0, x_edges[0] + length, height)
    mesh = _rectangle_mesh(x_edges, y_edges, window=window)
    youngs, poisson, density = _ELASTIC
    coefficients = _isotropic_coefficients(mesh.n_elements, youngs, poisson)
    omega = 2.0 * math.pi * 1e9
    wave_speed = math.sqrt(float(coefficients['c11'][0]) / density)
    errors, details = {}, {}
    solid = np.ones(mesh.n_elements, dtype=bool)
    for name, axisymmetric, order in (('planar', False, 0), ('axisymmetric', True, 0), ('order2', True, 2)):
        pml = elastic_pml_ring(mesh, ('x_hi',), omega, wave_speed, axisymmetric=axisymmetric)
        system = ElasticFEMSystem(mesh, 2, axisymmetric, lambda: 'pml operator', azimuthal_order=order)
        system.pml = pml
        for face in ('x_lo',):
            system.hold(system.global_ids_of_edges(system.face_edges(face)), 0.0, 0.0, 0.0)
        system.hold(system.global_ids_of_edges(np.asarray(mesh.edge_tags['x_hi'], dtype=np.int64)), 0.0, 0.0, 0.0)
        system.assemble(coefficients, solid)
        stretched = system.stiffness(coefficients, stretched=True)
        mass = system.mass(np.full(mesh.n_elements, density))
        prestress = np.zeros((mesh.n_elements, 4 if axisymmetric else 3))
        prestress[:, 0], prestress[:, 1] = 1e8, -5e7
        if axisymmetric:
            prestress[:, 2] = 3e7
        geometric = system.geometric_stiffness(StressField(prescribed=prestress))
        for label, matrix in (('stiffness', stretched), ('mass', mass), ('geometric', geometric)):
            scale = float(abs(matrix).max())
            errors[f'{name} {label} symmetric'] = float(abs(matrix - matrix.T).max()) / scale
            errors[f'{name} {label} complex'] = float(not np.iscomplexobj(matrix.data)
                                                     or abs(matrix.imag).max() < 1e-6 * scale)
        ring = system.ring_only_dofs()
        components = system.components
        in_window = np.zeros(system.dofs.n_dofs, dtype=bool)
        window_rows = mesh.pml_faces[system.dofs.elements] == 0
        in_window[system.dofs.element_dofs[window_rows].ravel()] = True
        in_ring = np.zeros(system.dofs.n_dofs, dtype=bool)
        in_ring[system.dofs.element_dofs[~window_rows].ravel()] = True
        inner = np.flatnonzero(np.repeat(in_window & ~in_ring, components))
        errors[f'{name} window real'] = (float(abs(stretched[inner][:, inner].imag).max())
                                         / float(abs(stretched).max()))
        errors[f'{name} ring-only dofs'] = float(not ring.any() or bool((ring & np.repeat(in_window, components)).any()))
        details[f'{name} beta'] = pml.strength['x_hi']
    plain_system = ElasticFEMSystem(mesh, 2, False, lambda: 'pml operator')
    plain_system.hold(plain_system.global_ids_of_edges(plain_system.face_edges('x_lo')), 0.0, 0.0)
    plain_system.assemble(coefficients, solid)
    unit = assemble_elastic_stiffness(plain_system.dofs, coefficients,
                                      stretch=lambda points, elements: np.ones(points.shape, dtype=np.complex128))
    errors['unit stretch'] = float(abs(unit - plain_system.operator).max()) / float(abs(plain_system.operator).max())
    pml = elastic_pml_ring(mesh, ('x_hi',), omega, wave_speed)
    edge = window[2]
    ends = pml.stretches(np.array([[edge, 0.1 * UM], [edge + _RING, 0.1 * UM]]))
    beta = 3.0 * math.log(1.0 / DEFAULT_PML_REFLECTION) * wave_speed / (2.0 * omega * _RING)
    errors['profile'] = max(abs(ends[0, 0] - 1.0), abs(ends[1, 0] - (1.0 - 1j * beta)) / beta,
                            float(np.max(np.abs(ends[:, 1] - 1.0))))
    return worst_result(errors, _OPERATOR_TOLERANCE, **details, beta_expected=beta)


# ---- solid_fem_pml_bar_radiation: the leaky Fabry-Perot bar of selftests_pml3d in plane strain ----
# 2 x 0.1 um strip of 50 nm P2 triangles, nu = 0 (plane longitudinal waves need no lateral holds, so the free y sides of
# the ring match the window's rollers): cavity A (E 100 GPa) on x < 1 um clamped at x_lo, then B (E 400 GPa: Z_A / Z_B =
# 1 / 2) to 2 um and into a 16-layer elastic PML ring on x_hi (pml_cells 16, MeshSpec pml_layers 16, pml_reflection
# 1e-4). tan(k_A L) = j r: omega_n = (c_A / L) (n pi + j atanh(r)), f_n = n c_A / (2 L), Q_n = n pi / (2 atanh r).
# Tolerances as 3D (first estimates): f 2e-3, Q 3e-2.
_BAR_STEP = 0.05 * UM
_BAR_CAVITY = 1.0 * UM
_BAR_LENGTH = 2.0 * UM
_BAR_SIDE = 0.1 * UM
_BAR_DENSITY = 2000.0
_BAR_E_CAVITY = 100e9
_BAR_E_OUTSIDE = 400e9
_BAR_PML_CELLS = 16
_BAR_REFLECTION = 1e-4
_BAR_MODES = 2
_FREQUENCY_TOLERANCE = 2e-3
_Q_TOLERANCE = 3e-2


def _bar_expected():
    """[(f_n [Hz], Q_n)] of the first _BAR_MODES leaky modes (module comment)."""
    speed = math.sqrt(_BAR_E_CAVITY / _BAR_DENSITY)
    ratio = math.sqrt(_BAR_E_CAVITY / _BAR_E_OUTSIDE)
    return [(n * speed / (2.0 * _BAR_CAVITY), n * math.pi / (2.0 * math.atanh(ratio))) for n in range(1, _BAR_MODES + 1)]


def _rectangle(name, x0, x1, y1):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': 0.0, 'x1': x1, 'y1': y1})


def _elastic(youngs_modulus, poisson_ratio, density):
    return {'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio, 'rho': density}


def _bar_spec():
    settings = {'discretization': 'quadratic', 'target_frequency': 1.5 * _bar_expected()[0][0], 'pml_x_hi': True,
                'pml_cells': _BAR_PML_CELLS, 'pml_reflection': _BAR_REFLECTION}
    features = [FeatureSpec('clamp', 'fixed', ['x_lo']), FeatureSpec('sides', 'roller', ['y_lo', 'y_hi']),
                FeatureSpec(_TABLE, 'resonance_table')]
    return ModelSpec(
        title='Self-test: leaky Fabry-Perot bar into a 2D elastic PML',
        domain=DomainSpec(size_x=_BAR_LENGTH, size_y=_BAR_SIDE),
        geometry=[_rectangle('cavity', 0.0, _BAR_CAVITY, _BAR_SIDE),
                  _rectangle('outside', _BAR_CAVITY, _BAR_LENGTH, _BAR_SIDE)],
        materials=[MaterialSpec('soft', properties=_elastic(_BAR_E_CAVITY, 0.0, _BAR_DENSITY), selection=['cavity']),
                   MaterialSpec('stiff', properties=_elastic(_BAR_E_OUTSIDE, 0.0, _BAR_DENSITY),
                                selection=['outside'])],
        physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, settings=settings, features=features)],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_BAR_STEP,
                         pml_layers=_BAR_PML_CELLS)],
        solver_defaults=cpu_defaults(), study=StudySpec(type='eigenfrequency', settings={'n_resonances': _BAR_MODES}))


def _check_bar(results):
    """Re f and Q of the table rows (sorted by f) against the leaky Fabry-Perot closed form."""
    table = results.result(_SOLID, _TABLE)
    columns = table['columns']
    rows = sorted(table['rows'], key=lambda row: row[columns.index('frequency')])
    measured = [(float(row[columns.index('frequency')]), float(row[columns.index('q_factor')])) for row in rows]
    expected = _bar_expected()
    errors, tolerances = {}, {}
    if len(measured) != len(expected):
        errors['count'], tolerances['count'] = math.inf, 1.0
    for n, ((frequency, quality), (frequency_expected, quality_expected)) in enumerate(zip(measured, expected),
                                                                                     start=1):
        errors[f'f{n}'], tolerances[f'f{n}'] = abs(frequency / frequency_expected - 1.0), _FREQUENCY_TOLERANCE
        errors[f'Q{n}'], tolerances[f'Q{n}'] = abs(quality / quality_expected - 1.0), _Q_TOLERANCE
    return scaled_worst_result(errors, tolerances, measured_f_Q=measured, expected_f_Q=expected)


# ---- solid_fem_energy_density: the static and modal energy densities ----
# 'static': a 10 x 1 mm plane-stress bar (P2, h 0.25 mm, E 100 GPa, nu 0.25), rollers x_lo / y_lo, traction T on x_hi:
# uniaxial sigma_xx = T, so strain_energy_density = T^2 / (2 E) everywhere (exact in P2) and kinetic 0 (probes, 1e-5).
# 'modes': the orthotropic roller box of solid_fem_roller_box_modes, mode 1 live after the study: its time-averaged
# strain and kinetic energies are equal (u^T K u = omega^2 u^T M u); compared as the means of the live display fields
# over the window raster (nodal recovery and raster sampling, first estimate 1e-2).
_STATIC_SIZE = (10e-3, 1e-3)
_STATIC_H = 0.25e-3
_STATIC_MATERIAL = (100e9, 0.25)
_TRACTION = 2e7
_STATIC_TOLERANCE = 1e-5
_VIRIAL_TOLERANCE = 1e-2


def _energy_specs():
    youngs_modulus, poisson_ratio = _STATIC_MATERIAL
    probes = [FeatureSpec(f'{name}_p', 'solid_probe', ['p'], {'component': name})
              for name in ('strain_energy_density', 'kinetic_energy_density')]
    static = ModelSpec(
        title='Self-test: strain energy density of a uniaxial bar on triangles',
        domain=DomainSpec(size_x=_STATIC_SIZE[0], size_y=_STATIC_SIZE[1]),
        geometry=[GeometrySpec('p', 'point', {'x': 0.6 * _STATIC_SIZE[0], 'y': 0.4 * _STATIC_SIZE[1]})],
        materials=[MaterialSpec('bar', properties={'youngs_modulus': youngs_modulus, 'poisson_ratio': poisson_ratio},
                                selection=['all'])],
        physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, settings={'out_of_plane': 'plane_stress'}, features=[
            FeatureSpec('left', 'roller', ['x_lo']), FeatureSpec('bottom', 'roller', ['y_lo']),
            FeatureSpec('pull', 'boundary_load', ['x_hi'], {'t_x': _TRACTION, 't_y': 0.0}), *probes])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_STATIC_H)],
        solver_defaults=cpu_defaults(), study=StudySpec(type='stationary'))
    modes = ModelSpec(
        title='Self-test: modal energy densities of the roller box on triangles',
        domain=DomainSpec(size_x=_BOX[0], size_y=_BOX[1]),
        materials=[MaterialSpec('box', properties={**_BOX_ORTHOTROPIC, 'rho': _BOX_DENSITY}, selection=['all'])],
        physics=[PhysicsSpec(_SOLID, _SOLID_TYPE, features=[
            FeatureSpec('rollers', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']), FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_BOX_H)],
        solver_defaults=cpu_defaults(), study=StudySpec(type='eigenfrequency', settings={'n_resonances': 2}))
    return {'static': static, 'modes': modes}


def _check_energy(results_by_name):
    """Static probes against T^2 / (2 E) and 0; the live mode's mean strain and kinetic energy densities equal."""
    from ...model.selftest_helpers import float32_value
    static = results_by_name['static']
    youngs_modulus = float32_value(_STATIC_MATERIAL[0])
    expected = _TRACTION ** 2 / (2.0 * youngs_modulus)
    errors, tolerances, details = {}, {}, {}
    for name, reference in (('strain_energy_density', expected), ('kinetic_energy_density', 0.0)):
        values = np.asarray(static.result(_SOLID, f'{name}_p')['values'], dtype=np.float64)
        value = float(values[-1]) if values.size else math.nan
        errors[f'static {name}'] = abs(value - reference) / expected
        tolerances[f'static {name}'] = _STATIC_TOLERANCE
        details[f'static {name}'] = value
    modes = results_by_name['modes']
    strain = float(np.nanmean(np.asarray(modes.field('strain_energy_density').values, dtype=np.float64)))
    kinetic = float(np.nanmean(np.asarray(modes.field('kinetic_energy_density').values, dtype=np.float64)))
    errors['mode virial'] = abs(strain / kinetic - 1.0) if kinetic > 0.0 else math.inf
    tolerances['mode virial'] = _VIRIAL_TOLERANCE
    details.update(static_expected=expected, mode_strain_mean=strain, mode_kinetic_mean=kinetic)
    return scaled_worst_result(errors, tolerances, **details)


PML_2D_CASES = (
    SelfTestCase(
        name='solid_fem_pml_operator', label='Solid mechanics on triangles: elastic PML ring operator identities',
        description=('fem.elasticity stretch, fem.pml and system_fem directly (no model run, P2): a 1 x 0.25 um strip of '
                     '0.125 um right triangles with a 0.5 um ring beyond x_hi: the stretched K, M and K_sigma complex '
                     'symmetric (planar, axisymmetric with r~, azimuthal order 2), the window-only block real, a unit '
                     'stretch equal to the plain K, s = 1 at the window side and 1 - j beta at the outer side (beta = '
                     '3 ln(1 / R) c / (2 omega w)), the ring-only dofs off the window, within '
                     f'{_OPERATOR_TOLERANCE:g} (first estimate, not yet run).'),
        specs=dict, check=_check_operator),
    SelfTestCase(
        name='solid_fem_pml_bar_radiation', label='Solid mechanics on triangles: PML radiation Q of a leaky bar',
        description=('eigenfrequency study of solid_mechanics (P2 triangles of 50 nm, plane strain, nu 0) on a 2 x 0.1 '
                     'um strip: a 1 um cavity (E 100 GPa) clamped at x_lo, then E 400 GPa (impedance ratio 1 / 2) into '
                     'a 16-layer elastic PML ring on x_hi: the two lowest leaky modes against omega_n = (c_A / L) (n pi '
                     f'+ j atanh(1 / 2)): f_n within {_FREQUENCY_TOLERANCE:g}, Q_n = n pi / (2 atanh(1 / 2)) within '
                     f'{_Q_TOLERANCE:g} (the 3D case solid_fem3d_pml_bar_radiation in 2D; first estimates, not yet '
                     'run). measured = error in tolerance units, tolerance 1.'),
        specs=lambda: {'bar': _bar_spec()}, check=lambda results_by_name: _check_bar(results_by_name['bar'])),
    SelfTestCase(
        name='solid_fem_energy_density', label='Solid mechanics on triangles: strain and kinetic energy densities',
        description=('A plane-stress bar under a uniaxial traction T (P2): probes strain_energy_density = T^2 / (2 E) '
                     f'and kinetic_energy_density = 0 within {_STATIC_TOLERANCE:g}; the orthotropic roller box mode 1 '
                     '(live after the eigenfrequency study): the window means of the time-averaged strain and kinetic '
                     f'energy densities equal within {_VIRIAL_TOLERANCE:g}. measured = error in tolerance units, '
                     'tolerance 1 (first estimates, not yet run).'),
        specs=_energy_specs, check=_check_energy),
)
