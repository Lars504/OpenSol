"""Model-run self-tests of the eigenfrequency study on solid_mechanics and solid_mechanics_3d: roller-supported blocks,
whose Q1 / trilinear vibration modes have a discrete closed form (2 x 2 / 3 x 3 symbol per sin / cos order set); no
physics imports."""
import math

import numpy as np

from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import float32_value, scaled_worst_result, solver_role, with_exact_cg
from ...model.spec import DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, SolverSpec, StudySpec
from .selftests import _lame

_PHYSICS = 'solid'
_SOLID_TYPE = 'solid_mechanics'   # = selftests.SOLID_TYPE_NAME
_STUDY = 'eigenfrequency'         # = model.study_eigenfrequency.STUDY_NAME
_TABLE = 'modes'
_FREQUENCY_COLUMN = 1             # frequency [Hz] in the resonance_table rows

# ---- roller box: 10 x 4 mm, uniform 0.5 mm cells (20 x 8), orthotropic c11 != c22 (c_ij only, no youngs_modulus), rho
# 2330, rollers on all four faces, plus a piezo_e11 entry that piezoelectric off must ignore. Mirror symmetry of the
# rollers makes u_x = A sin(p pi i / N_x) cos(q pi j / N_y), u_y = B cos(p pi i / N_x) sin(q pi j / N_y) (nodal) exact
# discrete modes of the uniform Q1 stiffness / consistent mass, with the 1D linear-element symbols K(t) = (2 / h)(1 -
# cos t), M(t) = (h / 3)(2 + cos t), S(t) = sin t: per order pair the 2 x 2 problem [[c11 Kx My + c66 Mx Ky, (c12 +
# c66) Sx Sy], [., c22 Mx Ky + c66 Kx My]] = omega^2 rho Mx My; u_x exists for 0 < p < N_x, u_y for 0 < q < N_y (2 N_x
# N_y - 2 modes = the free dofs). Lowest: (1, 0) longitudinal along x, then shear (1, 1), (2, 1), ... Run 'lowest':
# target 0, 6 modes; run 'shifted': target 1.03 x the 4th lowest, 3 modes ranked by |omega - omega_t| (pins the shift of
# K - sigma M).
_SIZE = (10e-3, 4e-3)             # m
_CELL = 0.5e-3                    # m
_CELLS = (20, 8)
_ORTHOTROPIC = {name: float32_value(value)   # float32-exact (MaterialGrid): 165.7e9 alone shifts mode 1 by 2e-8
                for name, value in (('c11', 165.7e9), ('c12', 63.9e9), ('c22', 120.0e9), ('c66', 79.6e9))}
_ORTHOTROPIC_DENSITY = 2330.0     # kg/m^3
_IGNORED_PIEZO_E11 = 1.0          # C/m^2: piezoelectric off, so it must not change the modes
_YOUNGS_MODULUS = 70e9            # Pa, 3D box
_POISSON_RATIO = 0.25
_DENSITY = 2200.0                 # kg/m^3, 3D box
_N_LOWEST = 6
_N_SHIFTED = 3
_SHIFT_INDEX = 3                  # the shifted target sits 3 % above the 4th lowest mode
_SHIFT_FACTOR = 1.03
_FREQUENCY_TOLERANCE = 1e-8       # relative (float64 K and M, eigsh to round-off)
_FIELD_TOLERANCE = 1e-5           # mode 1 cell-centre u (peak node |u| = 1 m, float32 outputs)


def _symbols(theta, cell=_CELL):
    """(K, M, S) of the 1D linear element on `cell` at phase theta."""
    return 2.0 / cell * (1.0 - math.cos(theta)), cell / 3.0 * (2.0 + math.cos(theta)), math.sin(theta)


def _table_frequencies(results, count, run):
    """The first `count` resonance_table frequencies [Hz] of a run (ValueError when fewer)."""
    rows = results.result(_PHYSICS, _TABLE)['rows']
    if len(rows) < count:
        raise ValueError(f'{run}: expected {count} modes in the table, got {len(rows)}')
    return [float(row[_FREQUENCY_COLUMN]) for row in rows[:count]]


def _add_rank_items(run, results, references, n_snapshots, tolerance, errors, tolerances):
    """Add '<run> frequency' (worst relative deviation of the first table frequencies from `references` [Hz], rank
    order) and '<run> snapshots' (stored snapshots against n_snapshots, tolerance 0.5); (measured [Hz], labels)."""
    measured = _table_frequencies(results, len(references), run)
    errors[f'{run} frequency'] = max(abs(value / reference - 1.0) for value, reference in zip(measured, references))
    tolerances[f'{run} frequency'] = tolerance
    labels = results.summary().get('snapshot_labels') or []
    errors[f'{run} snapshots'] = abs(len(labels) - n_snapshots)
    tolerances[f'{run} snapshots'] = 0.5
    return measured, labels


def _discrete_modes(cells=_CELLS, stiffness=None):
    """Sorted [(frequency [Hz], (p, q, branch))] of every discrete mode of the roller box of `cells` (_CELL cells) and
    c_ij `stiffness` (None: the orthotropic box)."""
    stiffness = _ORTHOTROPIC if stiffness is None else stiffness
    c11, c12, c22, c66 = (stiffness[name] for name in ('c11', 'c12', 'c22', 'c66'))
    n_x, n_y = cells
    modes = []
    for p in range(n_x + 1):
        for q in range(n_y + 1):
            stiff_x, mass_x, slope_x = _symbols(p * math.pi / n_x)
            stiff_y, mass_y, slope_y = _symbols(q * math.pi / n_y)
            row_x = c11 * stiff_x * mass_y + c66 * mass_x * stiff_y
            row_y = c22 * mass_x * stiff_y + c66 * stiff_x * mass_y
            coupling = (c12 + c66) * slope_x * slope_y
            mass = _ORTHOTROPIC_DENSITY * mass_x * mass_y
            has_x, has_y = 0 < p < n_x, 0 < q < n_y
            if has_x and has_y:
                mean, half_gap = 0.5 * (row_x + row_y), math.sqrt(0.25 * (row_x - row_y) ** 2 + coupling ** 2)
                values = (('lower', mean - half_gap), ('upper', mean + half_gap))
            elif has_x:
                values = (('x', row_x),)
            elif has_y:
                values = (('y', row_y),)
            else:
                continue
            modes.extend((math.sqrt(value / mass) / (2.0 * math.pi), (p, q, branch)) for branch, value in values)
    modes.sort(key=lambda mode: mode[0])
    return modes


def _shifted_target():
    """target_frequency [Hz] of the 'shifted' run."""
    return _SHIFT_FACTOR * _discrete_modes()[_SHIFT_INDEX][0]


def _box_spec(target_frequency, n_resonances, size=_SIZE, stiffness=None, study_settings=None):
    """Roller box with a resonance table, eigenfrequency study at target_frequency [Hz] (size [m], c_ij stiffness
    None = the orthotropic box; study_settings added to n_resonances, e.g. a search)."""
    stiffness = _ORTHOTROPIC if stiffness is None else stiffness
    return ModelSpec(
        title='Self-test: vibration modes of an orthotropic roller-supported block (solid mechanics)',
        domain=DomainSpec(size_x=size[0], size_y=size[1]),
        materials=[MaterialSpec('block', properties={**stiffness, 'rho': _ORTHOTROPIC_DENSITY,
                                                     'piezo_e11': _IGNORED_PIEZO_E11}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings={'target_frequency': target_frequency},
                             features=[FeatureSpec('rollers', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
                                       FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type=_STUDY, settings={'n_resonances': n_resonances, **(study_settings or {})}))


def _box_specs():
    return {'lowest': _box_spec(0.0, _N_LOWEST), 'shifted': _box_spec(_shifted_target(), _N_SHIFTED)}


def _mode_one_error(results):
    """max |u - closed form| of mode 1 (the live field, the (1, 0) longitudinal mode): cell centres u_x = sin(pi (i +
    1/2) / N_x) cos(pi / (2 N_x)) (mean of the corner values, peak node 1), u_y = 0."""
    u_x = np.asarray(results.field(f'{_PHYSICS}.u_x').values, dtype=np.float64)
    u_y = np.asarray(results.field(f'{_PHYSICS}.u_y').values, dtype=np.float64)
    n_x, _ = _CELLS
    if u_x.shape != _CELLS:
        raise ValueError(f'u_x window shape {u_x.shape} differs from the {_CELLS} block cells')
    expected = (np.sin(math.pi * (np.arange(n_x) + 0.5) / n_x) * math.cos(math.pi / (2.0 * n_x)))[:, None]
    return max(float(np.max(np.abs(u_x - expected))), float(np.max(np.abs(u_y))))


def _check_box(results_by_name):
    """Frequencies of both runs (in rank order) against the discrete closed form, snapshot counts and the mode-1 shape;
    measured = error = the worst deviation in tolerance units."""
    modes = _discrete_modes()
    if modes[0][1] != (1, 0, 'x'):
        raise ValueError(f'expected the (1, 0) longitudinal mode lowest, got {modes[0][1]}')
    target = _shifted_target()
    expected_runs = {'lowest': modes[:_N_LOWEST],
                     'shifted': sorted(modes, key=lambda mode: abs(mode[0] - target))[:_N_SHIFTED]}
    errors, tolerances, details = {}, {}, {}
    for run, expected in expected_runs.items():
        measured, labels = _add_rank_items(run, results_by_name[run], [frequency for frequency, _ in expected],
                                           len(expected), _FREQUENCY_TOLERANCE, errors, tolerances)
        details[run] = {'orders': [list(orders) for _, orders in expected],
                        'frequency_expected': [frequency for frequency, _ in expected],
                        'frequency_measured': measured, 'snapshot_labels': list(labels)}
    errors['lowest mode 1 field'] = _mode_one_error(results_by_name['lowest'])
    tolerances['lowest mode 1 field'] = _FIELD_TOLERANCE
    return scaled_worst_result(errors, tolerances, **details, target_frequency_shifted=target)


# ---- 3D roller box: 4.5 x 2.5 x 2 mm, uniform 0.5 mm cells (9 x 5 x 4: pairwise coprime cell counts, so no two axes
# share a phase p pi / N and no cross-axis degenerate pair appears), E 70 GPa, nu 0.25, rho 2200, rollers on all six
# faces. u_a = A_a sin(p_a pi i_a / N_a) prod_{d != a} cos(p_d pi i_d / N_d) are exact discrete modes of the uniform
# trilinear stiffness / consistent mass; per order triple the (up to) 3 x 3 symbol B_aa = c11 K_a M_b M_c + c66 (M_a K_b
# M_c + M_a M_b K_c), B_ab = (c12 + c66) S_a S_b M_c = omega^2 rho M_x M_y M_z, component a present for 0 < p_a < N_a.
# Lowest: (1, 0, 0) longitudinal along x (x the longest axis; the check raises otherwise); by a continuum estimate the 6
# lowest include the x-z and y-z coupled (1, 0, 1) and (0, 1, 1). One run (3D searches the lowest modes), exact CG
# (1e-12) for the inner solves (the Lanczos tolerance is 1e3 x it: keep 1e-12 against the 1e-8 frequency tolerance).
_SIZE_3D = (4.5e-3, 2.5e-3, 2e-3)     # m
_CELLS_3D = (9, 5, 4)
_SOLID_3D_TYPE = 'solid_mechanics_3d'   # = selftests_3d.SOLID_3D_TYPE_NAME
_N_LOWEST_3D = 6


def _discrete_modes_3d():
    """Sorted [(frequency [Hz], (p, q, r, branch))] of every discrete mode of the 3D roller box."""
    lame_lambda, mu = _lame(_YOUNGS_MODULUS, _POISSON_RATIO)
    c11, c12, c66 = lame_lambda + 2.0 * mu, lame_lambda, mu
    modes = []
    for orders in np.ndindex(*(n + 1 for n in _CELLS_3D)):
        stiff, mass, slope = zip(*(_symbols(order * math.pi / n) for order, n in zip(orders, _CELLS_3D)))
        present = [axis for axis in range(3) if 0 < orders[axis] < _CELLS_3D[axis]]
        if not present:
            continue
        symbol = np.zeros((3, 3))
        for a in range(3):
            b, c = (axis for axis in range(3) if axis != a)
            symbol[a, a] = c11 * stiff[a] * mass[b] * mass[c] + c66 * mass[a] * (stiff[b] * mass[c]
                                                                                 + mass[b] * stiff[c])
            for b in range(3):
                if b != a:
                    symbol[a, b] = (c12 + c66) * slope[a] * slope[b] * mass[3 - a - b]
        values = np.linalg.eigvalsh(symbol[np.ix_(present, present)]) / (_DENSITY * mass[0] * mass[1] * mass[2])
        modes.extend((math.sqrt(value) / (2.0 * math.pi), (*orders, branch)) for branch, value in enumerate(values))
    modes.sort(key=lambda mode: mode[0])
    return modes


def _box_3d_specs():
    """{'lowest': 3D roller box with a resonance table, eigenfrequency study of _N_LOWEST_3D modes, exact CG}."""
    return {'lowest': with_exact_cg(ModelSpec(
        title='Self-test: vibration modes of a roller-supported 3D block (solid mechanics 3D)',
        domain=DomainSpec(size_x=_SIZE_3D[0], size_y=_SIZE_3D[1], size_z=_SIZE_3D[2]),
        materials=[MaterialSpec('block', properties={'youngs_modulus': _YOUNGS_MODULUS, 'poisson_ratio': _POISSON_RATIO,
                                                     'rho': _DENSITY}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_3D_TYPE,
                             features=[FeatureSpec('rollers', 'roller', ['x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo',
                                                                         'z_hi']),
                                       FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type=_STUDY, settings={'n_resonances': _N_LOWEST_3D})))}


def _mode_one_error_3d(results):
    """max |u - closed form| of mode 1 ((1, 0, 0) longitudinal) on the mid-height z cut: u_x = sin(pi (i + 1/2) / N_x)
    cos(pi / (2 N_x)) / max_i sin(pi i / N_x) at every y (odd N_x: no node at the sine peak), u_y = u_z = 0."""
    n_x = _CELLS_3D[0]
    u_x = np.asarray(results.field(f'{_PHYSICS}.u_x').values, dtype=np.float64)
    if u_x.shape != _CELLS_3D[:2]:
        raise ValueError(f'u_x cut shape {u_x.shape} differs from the {_CELLS_3D[:2]} block cells')
    peak_node_sine = float(np.max(np.sin(math.pi * np.arange(n_x + 1) / n_x)))
    expected = (np.sin(math.pi * (np.arange(n_x) + 0.5) / n_x) * math.cos(math.pi / (2.0 * n_x))
                / peak_node_sine)[:, None]
    others = (np.asarray(results.field(f'{_PHYSICS}.{name}').values, dtype=np.float64) for name in ('u_y', 'u_z'))
    return max(float(np.max(np.abs(u_x - expected))), *(float(np.max(np.abs(values))) for values in others))


def _box_3d_items(results):
    """(errors, tolerances, details) of the 3D box run: the lowest frequencies against the discrete closed form, the
    snapshot count and the mode-1 shape."""
    modes = _discrete_modes_3d()
    if modes[0][1][:3] != (1, 0, 0):
        raise ValueError(f'expected the (1, 0, 0) longitudinal mode lowest, got {modes[0][1]}')
    expected = modes[:_N_LOWEST_3D]
    rows = results.result(_PHYSICS, _TABLE)['rows']
    if len(rows) < len(expected):
        raise ValueError(f'expected {len(expected)} modes in the table, got {len(rows)}')
    measured = [float(row[_FREQUENCY_COLUMN]) for row in rows[:len(expected)]]
    labels = results.summary().get('snapshot_labels') or []
    errors = {'frequency': max(abs(value / reference - 1.0) for value, (reference, _) in zip(measured, expected)),
              'snapshots': abs(len(labels) - len(expected)), 'mode 1 field': _mode_one_error_3d(results)}
    tolerances = {'frequency': _FREQUENCY_TOLERANCE, 'snapshots': 0.5, 'mode 1 field': _FIELD_TOLERANCE}
    details = {'orders': [[int(value) for value in orders] for _, orders in expected],
               'frequency_expected': [frequency for frequency, _ in expected], 'frequency_measured': measured,
               'snapshot_labels': list(labels)}
    return errors, tolerances, details


def _check_box_3d(results_by_name):
    """Lowest frequencies against the discrete closed form, the snapshot count and the mode-1 shape; measured = error =
    the worst deviation in tolerance units."""
    errors, tolerances, details = _box_3d_items(results_by_name['lowest'])
    return scaled_worst_result(errors, tolerances, **details)


# ---- the same 3D box by the factor-free block eigensolver: an opt-in 'block_eigen' eigen node at its defaults (mixed:
# float32 LOBPCG phase finished in float64 to a 1e-8 relative residual, eigenvalue error ~1e-15; preconditioner
# multigrid), so the closed form and tolerances of solid_3d_eigenfrequency_roller_box hold unchanged; the exact CG node
# stays the 'linear' solver (no inner solve runs, its multigrid is built privately). The summary 'solvers' record of the
# eigen role must name block_eigen with at least one solve and none unconverged.
_BLOCK_EIGEN_NODE = 'block_eigen_modes'   # never the variant pattern <variant>_<physics>_<role>
_BLOCK_EIGEN_TYPE = 'block_eigen'         # = solvers.definition BLOCK_EIGEN.name


def _box_3d_block_eigen_specs():
    """{'lowest': the 3D roller box of _box_3d_specs with the block_eigen node as its 'eigen' solver}."""
    spec = _box_3d_specs()['lowest']
    spec.solvers.append(SolverSpec(_BLOCK_EIGEN_NODE, _BLOCK_EIGEN_TYPE, {}))
    spec.physics[0].solvers['eigen'] = _BLOCK_EIGEN_NODE
    return {'lowest': spec}


def _check_box_3d_block_eigen(results_by_name):
    """_check_box_3d of the block_eigen run plus its eigen role record (type block_eigen, eigen_solves >= 1,
    eigen_unconverged 0); measured = error = the worst deviation in tolerance units."""
    results = results_by_name['lowest']
    errors, tolerances, details = _box_3d_items(results)
    record = solver_role(results, _PHYSICS, 'eigen')
    counters = record.get('counters') or {}
    errors['eigen solver type'] = 0.0 if record.get('type') == _BLOCK_EIGEN_TYPE else 1.0
    errors['eigen solves'] = 0.0 if counters.get('eigen_solves', 0) >= 1 else 1.0
    errors['eigen unconverged'] = float(counters.get('eigen_unconverged', math.inf))
    tolerances.update({'eigen solver type': 0.5, 'eigen solves': 0.5, 'eigen unconverged': 0.5})
    return scaled_worst_result(errors, tolerances, **details, eigen_type=record.get('type'),
                               eigen_backend=record.get('backend'), eigen_precision=record.get('precision'),
                               eigen_settings=record.get('settings'), eigen_counters=counters)


EIGENFREQUENCY_3D_CASES = (
    SelfTestCase(
        name='solid_3d_eigenfrequency_roller_box',
        label='Solid mechanics 3D: vibration modes of a roller-supported block',
        description=f'eigenfrequency study of solid_mechanics_3d on a {_SIZE_3D[0] * 1e3:g} x {_SIZE_3D[1] * 1e3:g} x '
                    f'{_SIZE_3D[2] * 1e3:g} mm block of uniform {_CELL * 1e3:g} mm cells ({_CELLS_3D[0]} x '
                    f'{_CELLS_3D[1]} x {_CELLS_3D[2]}; E {_YOUNGS_MODULUS:g} Pa, nu {_POISSON_RATIO:g}, rho '
                    f'{_DENSITY:g} kg/m^3) with rollers on all six faces, a resonance table and an exact CG (double, '
                    f'1e-12) for the inner solves of the matrix-free shift-invert; {_N_LOWEST_3D} lowest modes. The '
                    'roller mirrors make u_a = sin(p_a pi i_a / N_a) prod cos(p_d pi i_d / N_d) exact discrete modes: '
                    'per order triple a 3 x 3 problem of the 1D linear-element symbols K(t) = (2 / h)(1 - cos t), M(t) '
                    '= (h / 3)(2 + cos t), S(t) = sin t (B_aa = c11 K_a M_b M_c + c66 (M_a K_b M_c + M_a M_b K_c), B_ab '
                    '= (c12 + c66) S_a S_b M_c, mass rho M_x M_y M_z). The table frequencies must match the lowest of '
                    f'it in order (relative {_FREQUENCY_TOLERANCE:g}), the run stores one snapshot per mode, and the '
                    'live mode 1 ((1, 0, 0), longitudinal along x) has cell-centre u_x = sin(pi (i + 1/2) / N_x) cos(pi '
                    '/ (2 N_x)) / max_i sin(pi i / N_x), u_y = u_z = 0 on the mid-height cut within '
                    f'{_FIELD_TOLERANCE:g} (peak node |u| = 1 m; odd N_x has no node at the sine peak). measured = '
                    'error = the worst deviation in tolerance units, tolerance 1 (E is not float32-exact: its storage '
                    'shifts the frequencies by ~7e-9). This pins the mass kernel, the three-axis stiffness coupling, '
                    'the homogeneous supports, the inexact-inverse Lanczos and the peak normalisation. Variant '
                    "'mixed': the same with pcg precision mixed for the inner solves (float32 CG passes refined to the "
                    'float64 1e-12 residual), same tolerance.',
        specs=_box_3d_specs,
        check=_check_box_3d,
        variants=(SelfTestVariant('mixed', tolerance=1.0, precision='mixed', roles=('linear',)),)),
    SelfTestCase(
        name='solid_3d_eigenfrequency_block_eigen',
        label='Solid mechanics 3D: vibration modes of a roller-supported block by the block eigensolver',
        description='solid_3d_eigenfrequency_roller_box (the same block, rollers, resonance table, exact CG linear '
                    f"node and {_N_LOWEST_3D} lowest modes) with an opt-in eigen solver node '{_BLOCK_EIGEN_NODE}' of "
                    'type block_eigen at its defaults (precision mixed: a float32 LOBPCG phase finished in float64 to '
                    'the 1e-8 relative residual; preconditioner multigrid; guard vectors auto) instead of the '
                    'shift-invert Lanczos: ElasticSystem3D.lowest_modes on its block path (brick product, consistent '
                    'mass and preconditioner as block callbacks on the device, no inner CG solve). The same discrete '
                    f'closed form and tolerances (frequencies {_FREQUENCY_TOLERANCE:g} relative in order, one snapshot '
                    f'per mode, mode-1 shape {_FIELD_TOLERANCE:g}), and the summary solvers record of the eigen role '
                    'names type block_eigen with eigen_solves >= 1 and eigen_unconverged 0. measured = error = the '
                    "worst deviation in tolerance units, tolerance 1. Variant 'double': the block_eigen node at "
                    'precision double (float64 throughout), same tolerance. Its CPU run: '
                    'solid_mechanics_3d_block_eigen_cpu. Written, not run.',
        specs=_box_3d_block_eigen_specs,
        check=_check_box_3d_block_eigen,
        variants=(SelfTestVariant('double', tolerance=1.0, precision='double', roles=('eigen',)),)),
)

EIGENFREQUENCY_CASES = (
    SelfTestCase(
        name='solid_eigenfrequency_roller_box',
        label='Solid mechanics: vibration modes of an orthotropic roller-supported block',
        description=f'eigenfrequency study of solid_mechanics on a {_SIZE[0] * 1e3:g} x {_SIZE[1] * 1e3:g} mm block of '
                    f'uniform {_CELL * 1e3:g} mm cells ({_CELLS[0]} x {_CELLS[1]}; orthotropic float32-exact c11 '
                    f"{_ORTHOTROPIC['c11']:g}, c12 {_ORTHOTROPIC['c12']:g}, c22 {_ORTHOTROPIC['c22']:g}, c66 "
                    f"{_ORTHOTROPIC['c66']:g} Pa only (no youngs_modulus), rho {_ORTHOTROPIC_DENSITY:g} kg/m^3, and a "
                    f'piezo_e11 {_IGNORED_PIEZO_E11:g} C/m^2 entry that piezoelectric off must ignore) with rollers on '
                    "all four faces and a resonance table. Runs 'lowest' (target_frequency 0, "
                    f"{_N_LOWEST} modes) and 'shifted' (target "
                    f'{_SHIFT_FACTOR:g} x the {_SHIFT_INDEX + 1}th lowest mode, {_N_SHIFTED} modes). The roller mirrors '
                    'make u_x = sin(p pi i / N_x) cos(q pi j / N_y), u_y = cos sin exact discrete modes: per order pair '
                    'a 2 x 2 problem of the 1D linear-element symbols K(t) = (2 / h)(1 - cos t), M(t) = (h / 3)(2 + '
                    'cos t), S(t) = sin t ([[c11 Kx My + c66 Mx Ky, (c12 + c66) Sx Sy], [., c22 Mx Ky + c66 Kx My]] = '
                    'omega^2 rho Mx My). The table frequencies must match it in rank order (relative '
                    f'{_FREQUENCY_TOLERANCE:g}; the shifted run ranked by |omega - omega_target|), each run stores one '
                    'snapshot per mode, and the live mode 1 ((1, 0), longitudinal along x) has cell-centre u_x = sin(pi '
                    f'(i + 1/2) / N_x) cos(pi / (2 N_x)), u_y = 0 within {_FIELD_TOLERANCE:g} (peak node |u| = 1 m). '
                    'measured = error = the worst deviation in tolerance units, tolerance 1: this pins the consistent '
                    'mass, the c_ij-only solid cells, c22 != c11 and the coupling terms (c12 + c66) in the operator, '
                    'the homogeneous supports, the shift-invert map, the ranking, the peak normalisation and that a '
                    'piezo entry leaves the purely elastic modes alone (it holds the former '
                    'solid_orthotropic_roller_box_modes and the piezoelectric-off check of the former unstiffened run '
                    'of solid_piezo_bar_modes). Not yet run with the orthotropic material, nor with the host float64 '
                    'K / M of the frequency response as the modal pencil.',
        specs=_box_specs,
        check=_check_box),
)
