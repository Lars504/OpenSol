"""Free-free vibration modes of solid_mechanics_3d on tetrahedra (new_modules_plan K0): one case, the eigenfrequency
study of an unsupported nu = 0 bar plus its counted rigid motions and dropped rigid-body modes (CPU, no model), both
against the exact longitudinal mode f = c / (2 L). definition3d registers it."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, scaled_worst_result
from ...model.spec import FeatureSpec, MaterialSpec, StudySpec
from .selftests import _elastic, _lame
from .selftests_fem3d import _solid_spec

UM = 1e-6
_SOLID = 'solid'
_TABLE = 'modes'
# A free bar L x a x a (1 x 0.25 x 0.25 um, 0.125 um Kuhn cubes, P2), E 100 GPa, nu = 0, rho 2000. With nu = 0 (lambda
# = 0) u = (cos(pi x / L), 0, 0) is an exact continuum mode of the free body: sigma_xx = E u_x' is the only stress, 0 on
# x_lo / x_hi, and the lateral faces carry no traction; f_L = sqrt(E / rho) / (2 L). The P2 space holds every piecewise
# quadratic of x on the x cell planes, so the discrete mode lies between f_L and the 1D P2 value, (k h)^4 / 1440 above
# in frequency (~2e-5 at k h = pi / 8; estimate). The target 0.25 f_L sits below the first elastic mode (bending, ~0.5
# f_L): the six rigid-body modes are the nearest eigenvalues, so they are found and must be dropped (with rigid_motions
# extra eigenvalues asked, the elastic modes still fill n_resonances). Lanczos from one start vector need not return
# every copy of the six-fold lambda = 0 (5 seen), so 1 .. 6 dropped and no kept mode below _RIGID_FLOOR f_L. The
# longitudinal mode lies 0.05 % from a bending mode, which the Kuhn split mixes in (overlap 0.99 alone): the overlap is
# that of cos(pi x / L) with the span of the kept modes within _CLUSTER_WIDTH of the best-overlapping one.
_BAR_SIZE = (1.0 * UM, 0.25 * UM, 0.25 * UM)
_BAR_CELL = 0.125 * UM
_BAR_MATERIAL = (100e9, 0.0, 2000.0)   # E [Pa], nu, rho [kg/m^3]
_BAR_MODES = 6
_TARGET_FRACTION = 0.25                # target_frequency / f_L
_FREQUENCY_TOLERANCE = 2e-4            # relative frequency of the longitudinal mode (first estimate)
_OVERLAP_TOLERANCE = 1e-3              # 1 - mass-weighted overlap of the identified mode with cos(pi x / L)
_RIGID_FLOOR = 1e-2                    # every kept mode above this x f_L (no rigid-body mode left in the table)
_CLUSTER_WIDTH = 1e-2                  # relative frequency band of the modes whose span the overlap measures


def _longitudinal_frequency(stored=False):
    """f_L = sqrt(E / rho) / (2 L) [Hz] (stored: of the float32 values the MaterialGrid keeps)."""
    youngs_modulus, _, density = _BAR_MATERIAL
    if stored:
        youngs_modulus, density = float32_value(youngs_modulus), float32_value(density)
    return math.sqrt(youngs_modulus / density) / (2.0 * _BAR_SIZE[0])


def _bar_system(holds=(), allowed=True):
    """Assembled ElasticFEMSystem3D of the free bar (P2) with optional (face, component) holds at 0."""
    from ...fem.elasticity3d import isotropic_matrix_3d
    from ...fem.mesh3d import structured_box_mesh
    from .system_fem3d import ElasticFEMSystem3D
    edges = [np.linspace(0.0, side, int(round(side / _BAR_CELL)) + 1) for side in _BAR_SIZE]
    mesh = structured_box_mesh(*edges)
    system = ElasticFEMSystem3D(mesh, 2, lambda: 'free-free self-test')
    system.rigid_motions_allowed = allowed
    for face, component in holds:
        system.hold(system.full.face_dofs(system.face_faces(face)), **{component: 0.0})
    lame_lambda, mu = _lame(*_BAR_MATERIAL[:2])
    count = mesh.n_elements
    system.assemble(isotropic_matrix_3d(np.full(count, lame_lambda), np.full(count, mu)), np.ones(count, dtype=bool))
    return system


def _rigid_dropped(warnings) -> int:
    """Count of dropped rigid-body modes in a ModalSolve's warnings (0 when none)."""
    from .modal import RIGID_TEXT_HEAD
    for line in warnings:
        if RIGID_TEXT_HEAD in line:
            return int(line.split()[0])
    return 0


def _operator_items():
    """(errors, tolerances, details): rigid motions counted (free bar 6, a roller on x_lo 3), refused without the
    allowance, and modal.solve_modes at 0.25 f_L: rigid-body modes dropped, the longitudinal mode by its overlap."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from .modal import solve_modes
    errors, tolerances, details = {}, {}, {}
    counts = {'free': _bar_system().n_rigid_motions,
              'roller_x_lo': _bar_system([('x_lo', 'u_x')]).n_rigid_motions}
    try:
        _bar_system(allowed=False)
        refused = False
    except ValueError as exc:
        refused = 'free body' in str(exc)
        details['refused_message'] = str(exc)
    passed = counts == {'free': 6, 'roller_x_lo': 3} and refused
    errors['rigid_counts'], tolerances['rigid_counts'] = (0.0 if passed else math.inf), 1.0
    details.update({'rigid_counts': counts, 'refused_without_allowance': refused})
    system = _bar_system()
    density = np.full(system.mesh.n_elements, _BAR_MATERIAL[2])
    mass = system.mass(density)
    reference = _longitudinal_frequency()
    solve = solve_modes(system.operator, mass, system.free, 2.0 * math.pi * _TARGET_FRACTION * reference, _BAR_MODES,
                        3, ArpackShiftInvertSolver(), rigid_motions=system.n_rigid_motions)
    dropped = _rigid_dropped(solve.info['warnings'])
    kept_elastic = all(resonance.frequency > _RIGID_FLOOR * reference for resonance in solve.resonances)
    errors['rigid_dropped'], tolerances['rigid_dropped'] = (0.0 if 1 <= dropped <= 6 and kept_elastic
                                                            else math.inf), 1.0
    exact = np.zeros((system.n_dofs, 3))
    exact[:, 0] = np.cos(math.pi * system.full.dof_points()[:, 0] / _BAR_SIZE[0])
    exact = exact.reshape(-1)
    mass_exact = mass @ exact
    overlaps = []
    for resonance in solve.resonances:
        mode = np.asarray(resonance.displacement, dtype=np.float64)
        overlaps.append(abs(float(mode @ mass_exact))
                        / math.sqrt(float(exact @ mass_exact) * float(mode @ (mass @ mode))))
    best = int(np.argmax(overlaps)) if overlaps else -1
    measured = solve.resonances[best].frequency if overlaps else math.nan
    cluster = [overlap for resonance, overlap in zip(solve.resonances, overlaps)
               if abs(resonance.frequency / measured - 1.0) <= _CLUSTER_WIDTH]
    errors['overlap'] = 1.0 - math.sqrt(sum(overlap ** 2 for overlap in cluster)) if overlaps else math.inf
    errors['longitudinal'] = abs(measured / reference - 1.0)
    errors['found'] = 0.0 if len(solve.resonances) == _BAR_MODES else math.inf
    tolerances.update({'overlap': _OVERLAP_TOLERANCE, 'longitudinal': _FREQUENCY_TOLERANCE, 'found': 1.0})
    details.update({'rigid_dropped': dropped, 'frequencies_Hz': [r.frequency for r in solve.resonances],
                    'overlaps': overlaps, 'longitudinal_Hz': measured, 'continuum_Hz': reference,
                    'warnings': solve.info['warnings']})
    return errors, tolerances, details


def _free_bar_spec():
    youngs_modulus, poisson_ratio, density = _BAR_MATERIAL
    return _solid_spec('Self-test: free-free bar on tetrahedra (nu = 0, no supports)', _BAR_SIZE,
                       [MaterialSpec('bar', properties=_elastic(youngs_modulus, poisson_ratio, rho=density),
                                     selection=['all'])],
                       [FeatureSpec(_TABLE, 'resonance_table')], cell=_BAR_CELL,
                       study=StudySpec(type='eigenfrequency', settings={'n_resonances': _BAR_MODES}),
                       settings={'target_frequency': _TARGET_FRACTION * _longitudinal_frequency()})


def _check_free_bar(results_by_name):
    """Study table of the unsupported bar (n_resonances rows, none rigid, one within tolerance of f_L) plus the direct
    operator items of _operator_items (prefixed 'operator ')."""
    rows = results_by_name['free'].result(_SOLID, _TABLE)['rows']
    frequencies = [float(row[1]) for row in rows]
    reference = _longitudinal_frequency(stored=True)
    nearest = min(frequencies, key=lambda value: abs(value - reference)) if frequencies else math.nan
    errors = {'longitudinal': abs(nearest / reference - 1.0),
              'found': 0.0 if len(frequencies) == _BAR_MODES else math.inf,
              'no_rigid_rows': 0.0 if frequencies and min(frequencies) > _RIGID_FLOOR * reference else math.inf}
    tolerances = {'longitudinal': _FREQUENCY_TOLERANCE, 'found': 1.0, 'no_rigid_rows': 1.0}
    operator_errors, operator_tolerances, operator_details = _operator_items()
    errors.update({f'operator {item}': error for item, error in operator_errors.items()})
    tolerances.update({f'operator {item}': tolerance for item, tolerance in operator_tolerances.items()})
    return scaled_worst_result(errors, tolerances, frequencies_Hz=frequencies, longitudinal_Hz=nearest,
                               continuum_Hz=reference, operator=operator_details)


FREE_MODES_3D_CASES = (
    SelfTestCase(
        name='solid_fem3d_free_free_bar',
        label='Solid mechanics 3D on tetrahedra: free-free bar, eigenfrequency study and operator',
        description=('A free 1 x 0.25 x 0.25 um bar of 0.125 um Kuhn cubes (P2, E 100 GPa, nu 0, rho 2000). Study: '
                     'eigenfrequency study of solid_mechanics_3d without any support (validation must accept it at '
                     f'target_frequency > 0, the build count the rigid motions), target 0.25 f_L, {_BAR_MODES} modes: '
                     f'the resonance table has {_BAR_MODES} rows, all above {_RIGID_FLOOR:g} f_L (rigid-body modes '
                     f'dropped), and the row nearest f_L lies within {_FREQUENCY_TOLERANCE:g} of it (cpu backend, host '
                     'SuperLU: the cuDSS row scale may cancel at a target > 0). Operator items (ElasticFEMSystem3D and '
                     'modal.solve_modes directly, the former solid_fem3d_free_modes_operator): with '
                     'rigid_motions_allowed the rigid check counts 6 free motions (3 with a roller on x_lo) and without '
                     'it refuses the bar naming the free-body remedy; solve_modes at 0.25 f_L with rigid_motions = 6 '
                     f'drops 1 .. 6 rigid-body modes (Lanczos may miss copies of lambda = 0), keeps {_BAR_MODES} '
                     f'elastic ones, cos(pi x / L) lies in the span of the kept modes within {_CLUSTER_WIDTH:g} of the '
                     f'best-overlapping one (mass-weighted overlap within {_OVERLAP_TOLERANCE:g} of 1) and that mode '
                     f'within {_FREQUENCY_TOLERANCE:g} of the exact f_L = sqrt(E / rho) / (2 L) (exact for nu = 0). '
                     'measured = error = the worst deviation in tolerance units, tolerance 1 (first estimates; the merged '
                     'case not yet run).'),
        specs=lambda: {'free': _free_bar_spec()}, check=_check_free_bar),
)
