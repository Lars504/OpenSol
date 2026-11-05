"""Model-run self-test of the pcg 'multigrid' preconditioner (solvers.multigrid, solver_plan S7.0 / S7.1): a heated dot
on an outer-graded mesh solved by multigrid-PCG against a cpu SuperLU field, and far fewer CG iterations than
Jacobi-PCG; specs fresh."""
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


MULTIGRID_SELFTEST_CASES = (
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
        heavy=True,
        max_wall_time=600.0),
)
