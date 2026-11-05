"""Elastic PML shell of solid_mechanics_3d on tetrahedra (piezo_3d_plan stage 6, D13): the stretched operator identities
(CPU, no model) and the radiation Q of a leaky Fabry-Perot bar against its closed form. definition3d registers them."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import scaled_worst_result
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)
from .selftests import _elastic

UM = 1e-6
_SOLID = 'solid'
_TABLE = 'modes'


# ---- solid_fem3d_pml_operator: the stretched matrices of a bar with an x_hi shell (CPU, no model) ----
# 1 x 0.25 x 0.25 um window of 0.125 um Kuhn cubes (P2) plus a 0.5 um shell beyond x_hi, isotropic (E 100 GPa, nu 0.3,
# rho 2000): K and M complex symmetric (transpose, not conjugate) to round-off, the window block real, a unit stretch
# reproduces the plain shell matrices, s = 1 on the window side and 1 - j beta at the outer side with beta = 3 ln(1 / R)
# c / (2 omega w) (R 1e-6, c = sqrt(c11 / rho), f 1 GHz).
_OPERATOR_STEP = 0.125 * UM
_OPERATOR_LENGTH = 1.0 * UM
_OPERATOR_SIDE = 0.25 * UM
_OPERATOR_SHELL = 0.5 * UM
_OPERATOR_MATERIAL = (100e9, 0.3, 2000.0)
_OPERATOR_TOLERANCE = 1e-12


def _check_operator(_):
    """Complex symmetry, the real window block, the unit stretch and the profile end values."""
    from ...fem.elasticity3d import assemble_elastic_stiffness_3d, isotropic_matrix_3d
    from ...fem.mesh3d import structured_box_mesh
    from .pml3d import DEFAULT_PML_REFLECTION, elastic_pml_shell_3d
    from .selftests import _lame
    from .system_fem3d import ElasticFEMSystem3D
    x_edges = np.arange(0.0, _OPERATOR_LENGTH + _OPERATOR_SHELL + 0.5 * _OPERATOR_STEP, _OPERATOR_STEP)
    side = np.arange(0.0, _OPERATOR_SIDE + 0.5 * _OPERATOR_STEP, _OPERATOR_STEP)
    mesh = structured_box_mesh(x_edges, side, side, window=(0.0, 0.0, 0.0, _OPERATOR_LENGTH, _OPERATOR_SIDE,
                                                            _OPERATOR_SIDE))
    youngs, poisson, density = _OPERATOR_MATERIAL
    lame_lambda, mu = _lame(youngs, poisson)
    material = isotropic_matrix_3d(np.full(mesh.n_elements, lame_lambda), np.full(mesh.n_elements, mu))
    omega = 2.0 * math.pi * 1e9
    wave_speed = math.sqrt((lame_lambda + 2.0 * mu) / density)
    shell = elastic_pml_shell_3d(mesh, ('x_hi',), omega, wave_speed)
    system = ElasticFEMSystem3D(mesh, 2, lambda: 'pml operator')
    system.pml = shell
    for face in ('x_lo', 'x_hi'):   # the clamped end and the shell's outer side
        system.hold(system.full.face_dofs(np.asarray(mesh.face_tags[face], dtype=np.int64)), 0.0, 0.0, 0.0)
    solid = np.ones(mesh.n_elements, dtype=bool)
    system.assemble(material, solid)
    stiffness, mass = system.operator, system.mass(np.full(mesh.n_elements, density))
    errors, details = {}, {'n_elements': mesh.n_elements, 'beta': shell.strength['x_hi']}
    for name, matrix in (('stiffness', stiffness), ('mass', mass)):
        scale = float(abs(matrix).max())
        errors[f'{name}_symmetric'] = float(abs(matrix - matrix.T).max()) / scale
        errors[f'{name}_complex'] = float(not np.iscomplexobj(matrix.data) or abs(matrix.imag).max() < 1e-6 * scale)
    window = mesh.window_elements()
    window_dofs = np.zeros(system.n_dofs, dtype=bool)
    window_dofs[system.full.element_dofs[window].ravel()] = True
    shell_dofs = np.zeros(system.n_dofs, dtype=bool)
    shell_dofs[system.full.element_dofs[~window].ravel()] = True
    inner = np.flatnonzero(np.repeat(window_dofs & ~shell_dofs, 3))
    errors['window_real'] = float(abs(stiffness[inner][:, inner].imag).max()) / float(abs(stiffness).max())
    plain = assemble_elastic_stiffness_3d(system.full, material, elements=~window)
    unit = assemble_elastic_stiffness_3d(system.full, material, elements=~window,
                                         stretch=lambda points, elements: np.ones(points.shape, dtype=np.complex128))
    errors['unit_stretch'] = float(abs(unit - plain).max()) / float(abs(plain).max())
    ends = shell.stretches(np.array([[_OPERATOR_LENGTH, 0.1 * UM, 0.1 * UM],
                                     [_OPERATOR_LENGTH + _OPERATOR_SHELL, 0.1 * UM, 0.1 * UM]]))
    beta = 3.0 * math.log(1.0 / DEFAULT_PML_REFLECTION) * wave_speed / (2.0 * omega * _OPERATOR_SHELL)
    errors['profile'] = max(abs(ends[0, 0] - 1.0), abs(ends[1, 0] - (1.0 - 1j * beta)) / beta,
                            float(np.max(np.abs(ends[:, 1:] - 1.0))))
    worst = float(max(errors.values()))
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _OPERATOR_TOLERANCE,
            'details': {**details, 'errors': errors}}


# ---- solid_fem3d_pml_bar_radiation: a leaky Fabry-Perot bar against its closed form ----
# Bar along x, 0.1 x 0.1 um cross-section of 50 nm Kuhn cubes (P2), nu = 0 (plane longitudinal waves need no lateral
# holds, so the free sides of the shell match the window's rollers): cavity A (E 100 GPa) on x < 1 um clamped at x_lo,
# then material B (E 400 GPa, so the impedance ratio r = Z_A / Z_B = 1 / 2) to 2 um and on into a 16-layer PML on x_hi
# (pml_reflection 1e-4). An outgoing wave in B gives tan(k_A L) = j r: omega_n = (c_A / L) (n pi + j atanh(r)), f_n = n
# c_A / (2 L), Q_n = n pi / (2 atanh r) (n = 1: 2.86, n = 2: 5.73). Rollers on the y / z window faces. Tolerances first
# estimates: f 2e-3, Q 3e-2.
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


def _bar_spec():
    target = 1.5 * _bar_expected()[0][0]
    box = {'y0': 0.0, 'z0': 0.0, 'y1': _BAR_SIDE, 'z1': _BAR_SIDE}
    settings = {'discretization': 'quadratic', 'target_frequency': target, 'pml_x_hi': True,
                'pml_cells': _BAR_PML_CELLS, 'pml_reflection': _BAR_REFLECTION}
    features = [FeatureSpec('clamp', 'fixed', ['x_lo']),
                FeatureSpec('sides', 'roller', ['y_lo', 'y_hi', 'z_lo', 'z_hi']),
                FeatureSpec(_TABLE, 'resonance_table')]
    return ModelSpec(
        title='leaky Fabry-Perot bar',
        domain=DomainSpec(size_x=_BAR_LENGTH, size_y=_BAR_SIDE, size_z=_BAR_SIDE),
        geometry=[GeometrySpec('cavity', 'box', {'x0': 0.0, 'x1': _BAR_CAVITY, **box}),
                  GeometrySpec('outside', 'box', {'x0': _BAR_CAVITY, 'x1': _BAR_LENGTH, **box})],
        materials=[MaterialSpec('soft', properties=_elastic(_BAR_E_CAVITY, 0.0, rho=_BAR_DENSITY), selection=['cavity']),
                   MaterialSpec('stiff', properties=_elastic(_BAR_E_OUTSIDE, 0.0, rho=_BAR_DENSITY),
                                selection=['outside'])],
        physics=[PhysicsSpec(_SOLID, 'solid_mechanics_3d', settings=settings, features=features)],
        meshes=[MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=_BAR_STEP, mesher='builtin')],
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _BAR_MODES}))


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
    pairs = zip(measured, expected)
    for n, ((frequency, quality), (frequency_expected, quality_expected)) in enumerate(pairs, start=1):
        errors[f'f{n}'], tolerances[f'f{n}'] = abs(frequency / frequency_expected - 1.0), _FREQUENCY_TOLERANCE
        errors[f'Q{n}'], tolerances[f'Q{n}'] = abs(quality / quality_expected - 1.0), _Q_TOLERANCE
    return scaled_worst_result(errors, tolerances, measured_f_Q=measured, expected_f_Q=expected)


PML_3D_CASES = (
    SelfTestCase(
        name='solid_fem3d_pml_operator', label='Solid mechanics 3D on tetrahedra: elastic PML operator identities',
        description=('fem.elasticity3d stretch and pml3d directly (no model run, P2): a 1 x 0.25 x 0.25 um bar of 0.125 '
                     'um Kuhn cubes with a 0.5 um shell beyond x_hi: the stretched K and M complex symmetric, the '
                     'window block real, a unit stretch equal to the plain shell matrices, s = 1 at the window side '
                     f'and 1 - j beta at the outer side (beta = 3 ln(1 / R) c / (2 omega w)) within '
                     f'{_OPERATOR_TOLERANCE:g}.'),
        specs=dict, check=_check_operator),
    SelfTestCase(
        name='solid_fem3d_pml_bar_radiation', label='Solid mechanics 3D on tetrahedra: PML radiation Q of a leaky bar',
        description=('eigenfrequency study of solid_mechanics_3d (quadratic builtin tetrahedra, 50 nm Kuhn cubes) on a '
                     '0.1 x 0.1 um bar, nu 0: a 1 um cavity (E 100 GPa) clamped at x_lo, then E 400 GPa (impedance '
                     'ratio 1 / 2) to 2 um and into a 16-layer elastic PML on x_hi: the two lowest leaky modes against '
                     'omega_n = (c_A / L) (n pi + j atanh(1 / 2)): f_n = n c_A / (2 L) within '
                     f'{_FREQUENCY_TOLERANCE:g}, Q_n = n pi / (2 atanh(1 / 2)) within {_Q_TOLERANCE:g} (first '
                     'estimates, not yet run). Pins the stretched pencil, the clamped shell, eigs_shifted with a '
                     'complex mass and the PML-mode filter. measured = error = the worst deviation in tolerance units, '
                     'tolerance 1.'),
        specs=lambda: {'bar': _bar_spec()}, check=lambda results_by_name: _check_bar(results_by_name['bar'])),
)
