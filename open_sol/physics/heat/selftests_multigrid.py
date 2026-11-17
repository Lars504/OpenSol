"""Model-run self-test of the pcg 'multigrid' preconditioner (solvers.multigrid, solver_plan S7.0 / S7.1): a heated dot
on an outer-graded mesh solved by multigrid-PCG against a cpu SuperLU field, and far fewer CG iterations than
Jacobi-PCG; specs fresh. Also the gpu device aggregation against the host numpy one (no model run)."""
import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import EXACT_CG_TOLERANCE, solver_role
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           RefinementSpec, SolverDefaultsSpec, SolverSpec, StudySpec)

HEAT_TYPE_NAME = 'heat'              # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'heat'
TEMPERATURE_FIELD = f'{PHYSICS_NAME}.T'
DOMAIN_SIZE = 1e-3                   # m, square window, every outer face held at WALL_TEMPERATURE
DOT_HALF_WIDTH = 1e-6                # m, centred heated square
BASE_CELL, DOT_CELL = 1e-6, 0.2e-6   # m
OUTER_CELL, OUTER_RATIO = 100e-6, 1.3   # m / 1: cells grow from the refined core to 100 um (aspect ratios up to ~500)
MAX_RATIO = 1.15
K_BACKGROUND, K_DOT = 1.4, 150.0     # W/(m*K): oxide around a silicon dot (~100x contrast)
POWER_DENSITY = 1e13                 # W/m^3 in the dot: ~25 K rise
WALL_TEMPERATURE = 300.0             # K
JACOBI_MAX_ITERATIONS = 1_000_000    # the Jacobi count run must converge on the graded mesh
JACOBI_COUNT_TOLERANCE = 1e-8        # Jacobi run: iteration count only (~2/3 of the iterations to 1e-12)
FIELD_TOLERANCE = 1e-4               # max |T_mg - T_direct| / peak rise: CG stops 1e-12 x condition number, float32 T
MIN_ITERATION_RATIO = 5.0            # Jacobi iterations (to 1e-8) / multigrid iterations (to 1e-12) at least this
DIRECT_RUN = 'direct'                # cpu backend, Automatic linear solver (host SuperLU): the exact field reference
PCG_RUNS = {'jacobi': {'tolerance': JACOBI_COUNT_TOLERANCE},
            'multigrid': {'tolerance': EXACT_CG_TOLERANCE, 'preconditioner': 'multigrid'}}


def _heated_dot_spec(run_name):
    """The heated dot: DIRECT_RUN on the cpu backend with the Automatic solver, else a double pcg node 'cg' with the
    run's tolerance and preconditioner."""
    centre = 0.5 * DOMAIN_SIZE
    dot = GeometrySpec('dot', 'rectangle', {'x0': centre - DOT_HALF_WIDTH, 'y0': centre - DOT_HALF_WIDTH,
                                            'x1': centre + DOT_HALF_WIDTH, 'y1': centre + DOT_HALF_WIDTH})
    spec = ModelSpec(
        title=f'Heated dot on an outer-graded mesh ({run_name})',
        domain=DomainSpec(size_x=DOMAIN_SIZE, size_y=DOMAIN_SIZE),
        geometry=[dot],
        materials=[MaterialSpec('oxide', properties={'k': K_BACKGROUND}, selection=['all']),
                   MaterialSpec('silicon', properties={'k': K_DOT}, selection=['dot'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=[
            FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi', 'y_lo', 'y_hi'], {'temperature': WALL_TEMPERATURE}),
            FeatureSpec('source', 'heat_source', ['dot'], {'Q': POWER_DENSITY})])],
        mesh=MeshSpec(mode='nonuniform', dx=BASE_CELL, max_ratio=MAX_RATIO, outer_dx=OUTER_CELL,
                      outer_ratio=OUTER_RATIO, refinements=[RefinementSpec('dot', DOT_CELL)]),
        solver_defaults=SolverDefaultsSpec(backend='cpu' if run_name == DIRECT_RUN else 'gpu'),
        study=StudySpec(type='stationary'))
    if run_name != DIRECT_RUN:
        spec.solvers = [SolverSpec('cg', 'pcg', {'precision': 'double', 'max_iterations': JACOBI_MAX_ITERATIONS,
                                                 **PCG_RUNS[run_name]})]
        spec.physics[0].solvers = {'linear': 'cg'}
    return spec


def _heated_dot_runs():
    return {run_name: _heated_dot_spec(run_name) for run_name in (DIRECT_RUN, *PCG_RUNS)}


def _check_heated_dot(results_by_name):
    """max |T_multigrid - T_direct| / peak rise; raises unless the reference is the cpu SuperLU, the Jacobi run
    converged, the multigrid run recorded >= 2 levels and used MIN_ITERATION_RATIO x fewer CG iterations."""
    direct_role = solver_role(results_by_name[DIRECT_RUN], PHYSICS_NAME)
    if direct_role.get('backend') != 'cpu' or direct_role.get('type') != 'sparse_direct':
        raise ValueError(f"expected a cpu sparse_direct reference, summary 'solvers' records {direct_role!r}")
    counters = {run_name: solver_role(results_by_name[run_name], PHYSICS_NAME).get('counters') or {}
                for run_name in PCG_RUNS}
    if counters['jacobi'].get('unconverged_solves', 1) != 0:
        raise ValueError(f"the Jacobi count run did not converge: counters {counters['jacobi']!r}")
    if counters['multigrid'].get('multigrid_levels', 0) < 2:
        raise ValueError(f"expected a multigrid hierarchy of >= 2 levels, counters {counters['multigrid']!r}")
    iterations = {run_name: int(counters[run_name].get('cg_iterations', 0)) for run_name in PCG_RUNS}
    iteration_ratio = iterations['jacobi'] / max(iterations['multigrid'], 1)
    if iteration_ratio < MIN_ITERATION_RATIO:
        raise ValueError(f'multigrid saved too few CG iterations: {iterations!r} (ratio {iteration_ratio:.1f} < '
                         f'{MIN_ITERATION_RATIO:g})')
    fields = {run_name: np.asarray(results_by_name[run_name].field(TEMPERATURE_FIELD).values, dtype=np.float64)
              for run_name in (DIRECT_RUN, 'multigrid')}
    rise = float(np.max(fields[DIRECT_RUN])) - WALL_TEMPERATURE
    deviation = float(np.max(np.abs(fields['multigrid'] - fields[DIRECT_RUN])))   # NaN propagates
    measured = deviation / rise
    return {'measured': measured, 'expected': 0.0, 'error': measured, 'tolerance': FIELD_TOLERANCE,
            'details': {'peak_rise_K': rise, 'max_abs_deviation_K': deviation, 'cg_iterations': iterations,
                        'iteration_ratio': iteration_ratio, 'multigrid_counters': counters['multigrid'],
                        'n_cells': list(fields[DIRECT_RUN].shape)}}


def _face_laplacian(shape, conductances, extra_diagonal):
    """Host CSR of sum over the grid faces g (u_a - u_b)^2 (conductances per axis: shape one shorter along it) plus a
    diagonal."""
    import scipy.sparse as sp
    n = int(np.prod(shape))
    index = np.arange(n).reshape(shape)
    rows, columns, values = [np.arange(n)], [np.arange(n)], [np.ravel(extra_diagonal)]
    for axis, conductance in enumerate(conductances):
        lower = index[tuple(slice(0, -1) if a == axis else slice(None) for a in range(len(shape)))].ravel()
        upper = index[tuple(slice(1, None) if a == axis else slice(None) for a in range(len(shape)))].ravel()
        g = np.ravel(conductance)
        rows += [lower, upper, lower, upper]
        columns += [upper, lower, lower, upper]
        values += [-g, -g, g, g]
    return sp.coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                         shape=(n, n)).tocsr()


def _aggregation_operators():
    """{name: host CSR}: a 2D 5-point operator with 1e-2..1 conductances, held identity rows and Dirichlet-like
    edges; a 3D 7-point one with a 100x block and C/dt-dominated rows; a float32 operator of mixed-sign couplings
    with one-way (unsymmetric pattern) entries."""
    import scipy.sparse as sp
    rng = np.random.default_rng(11)
    shape = (160, 120)
    conductances = [10.0 ** rng.uniform(-2.0, 0.0, size=(shape[0] - 1, shape[1])),
                    10.0 ** rng.uniform(-2.0, 0.0, size=(shape[0], shape[1] - 1))]
    edge = np.zeros(shape)
    edge[0, :] = edge[-1, :] = 2.0
    graded = _face_laplacian(shape, conductances, edge)
    free = (rng.random(graded.shape[0]) >= 0.03).astype(np.float64)   # 3 % held cells: identity rows
    graded = (sp.diags(free) @ graded @ sp.diags(free) + sp.diags(1.0 - free)).tocsr()
    shape_3d = (24, 20, 18)
    conductances_3d = [np.ones(tuple(s - (a == axis) for a, s in enumerate(shape_3d))) for axis in range(3)]
    for conductance in conductances_3d:
        conductance[4:12, 4:12, 4:12] = 100.0
    capacity = np.full(shape_3d, 1e-3)
    capacity[:, :, :9] = 0.5   # C/dt above the couplings: the row-sum floor forces no link there
    capacity_3d = _face_laplacian(shape_3d, conductances_3d, capacity)
    n_random = 3000
    couplings = sp.random(n_random, n_random, density=2e-3, rng=5, format='csr')
    couplings.data = rng.uniform(-1.0, 0.4, size=couplings.nnz)   # mostly negative, some positive couplings
    one_way = sp.random(n_random, n_random, density=2e-4, rng=6, format='csr')
    one_way.data = -rng.uniform(0.0, 1.0, size=one_way.nnz)
    off_diagonal = (couplings + couplings.T + one_way).tolil()
    off_diagonal.setdiag(0.0)
    off_diagonal = off_diagonal.tocsr()
    random_single = (off_diagonal + sp.diags(np.asarray(abs(off_diagonal).sum(axis=1)).ravel() + 0.1)).tocsr()
    return {'graded_2d_held': graded, 'capacity_3d': capacity_3d,
            'random_float32': random_single.astype(np.float32)}


def _check_device_aggregation(_results_by_name):
    """Mismatched aggregates (entries + count) of device_aggregate_strong_links against the host
    aggregate_strong_links: on synthetic operators and on every level of a gpu MultigridPreconditioner setup."""
    from ...core.backend import gpu_available, sparse_module, to_host
    from ...model.selftest_helpers import failed_count_result, skip_result
    from ...solvers import multigrid
    if not gpu_available():
        return skip_result('no usable GPU')
    sparse = sparse_module('gpu')

    def mismatch(indptr, indices, data, n_unknowns, device_result):
        host_aggregate, host_count = multigrid.aggregate_strong_links(to_host(indptr), to_host(indices),
                                                                      to_host(data), n_unknowns)
        device_aggregate, device_count = device_result
        return float(np.count_nonzero(host_aggregate != device_aggregate) + abs(host_count - device_count))

    checks, counts = {}, {}
    for name, host in _aggregation_operators().items():
        device = sparse.csr_matrix(host)
        result = multigrid.device_aggregate_strong_links(device.indptr, device.indices, device.data, host.shape[0])
        checks[name] = (mismatch(device.indptr, device.indices, device.data, host.shape[0], result), 0.0)
        counts[name] = result[1]
    preconditioner = multigrid.MultigridPreconditioner('gpu')
    preconditioner.setup(sparse.csr_matrix(_aggregation_operators()['graded_2d_held']))
    for depth, aggregates in enumerate(preconditioner._aggregates):
        level_operator = preconditioner._levels[depth].operator
        checks[f'hierarchy_level_{depth}'] = (mismatch(level_operator.indptr, level_operator.indices,
                                                       level_operator.data, level_operator.shape[0], aggregates), 0.0)
    return failed_count_result(checks, {'aggregate_counts': counts, 'levels': preconditioner.n_levels})


MULTIGRID_SELFTEST_CASES = (
    SelfTestCase(
        name='multigrid_device_aggregation',
        label='Multigrid: device aggregation equals the host aggregation',
        description=('The gpu twin of the smoothed-aggregation coarsening (device_aggregate_strong_links: strength '
                     'test, symmetric union, distance-2 Luby roots, joins) against the host numpy '
                     'aggregate_strong_links: a 2D 5-point operator (conductances 1e-2..1, 3 % held identity rows), a '
                     '3D 7-point one (100x block, C/dt-dominated rows), a float32 operator with positive couplings '
                     'and one-way pattern entries, and every level of a gpu MultigridPreconditioner setup of the 2D '
                     'one (float32 Galerkin operators). Passes when every aggregate index and count agrees (0 '
                     'mismatches); skipped without a GPU. Not yet run.'),
        specs=dict,
        check=_check_device_aggregation,
        backends=('gpu',)),
    SelfTestCase(
        name='heat_multigrid_outer_graded',
        label='Heat: multigrid-preconditioned CG on an outer-graded mesh',
        description=(f'A {DOMAIN_SIZE * 1e3:g} mm square (k = {K_BACKGROUND:g} W/(m*K), every face at '
                     f'{WALL_TEMPERATURE:g} K) with a {2 * DOT_HALF_WIDTH * 1e6:g} um silicon dot (k = {K_DOT:g}) '
                     f'heated at {POWER_DENSITY:g} W/m^3, meshed at {DOT_CELL * 1e6:g} um in the dot and graded '
                     f'(ratio {OUTER_RATIO:g}) to {OUTER_CELL * 1e6:g} um outer cells, solved three times: on the cpu '
                     'backend by the Automatic linear solver (host SuperLU, the exact field reference) and on the gpu '
                     'by a double pcg node with the multigrid preconditioner (smoothed-aggregation V-cycles, '
                     f'tolerance {EXACT_CG_TOLERANCE:g}) and with Jacobi (tolerance {JACOBI_COUNT_TOLERANCE:g}, up '
                     f'to {JACOBI_MAX_ITERATIONS:,} iterations, for its iteration count only). Passes when the '
                     'multigrid field agrees with the SuperLU one to the tolerance (relative to the peak rise; CG '
                     'stop times the condition number of the graded operator and float32 T), the Jacobi run '
                     f'converged, the hierarchy has >= 2 levels and multigrid needs >= {MIN_ITERATION_RATIO:g}x '
                     'fewer CG iterations than Jacobi (to the looser Jacobi tolerance: a stricter test than at equal '
                     'tolerances). Not yet run with the SuperLU reference and the 1e-8 Jacobi count run.'),
        specs=_heated_dot_runs,
        check=_check_heated_dot,
        backends=('gpu',),
        max_wall_time=600.0),
)
