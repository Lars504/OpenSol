"""Self-test heat_time_dependent_values: feature values and a material property of time t in transient heat
(insulated box with a ramped heat_source; heat capacity cp(t); a slab whose x_lo temperature and x_hi heat flux ramp in
time) against the exact implicit-Euler recursions with every t-value taken at the step start."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import EXACT_CG_TOLERANCE, with_exact_cg
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)

HEAT_TYPE_NAME = 'heat'            # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'heat'
REFERENCE_T = 293.15               # K, the heat T0 default: initial and reference temperature
CELL = 1e-5                        # m, uniform mesh of run 'slab'
CONDUCTIVITY = 2.0                 # W/(m K)
DENSITY = 1000.0                   # kg/m^3
CAPACITY = 1000.0                  # J/(kg K): rho cp = 1e6 J/(m^3 K)
TOLERANCE = 1e-5                   # float32 T (~3e-5 K near 300 K) over rises of ~15-60 K; float64 CG to 1e-12
# run 'source': insulated BOX x BOX, Q = Q_RAMP t; uniform T: theta_n = Q_RAMP dt^2 n (n - 1) / (2 rho cp)
BOX = 1e-4                         # m
BOX_CELL = 0.5 * BOX               # m: 2 x 2 cells (T is uniform, so any size is exact)
SOURCE_STEP = 1e-3                 # s
SOURCE_STEPS = 10
Q_RAMP = 4e11                      # W/(m^3 s): 18 K after 10 steps
# run 'capacity': the same box, constant Q_CONSTANT, cp = CAPACITY (1 + t / CP_TIME): theta_n = sum_k<n Q dt / (rho
# cp(k dt))
Q_CONSTANT = 1e10                  # W/m^3: 10 K in the first step
CP_TIME = 5e-3                     # s (5 steps: the run spans 2 CP_TIME)
# run 'slab': SLAB_CELLS along x (2 cells across, insulated y faces), x_lo temperature T0 + T_RATE t, x_hi inward flux
# FLUX_RATE t: the exact 1D finite-volume implicit Euler with both face values at t_n
SLAB_CELLS = 20
SLAB_STEP = 1e-4                   # s
SLAB_STEPS = 20
T_RATE = 2.5e4                     # K/s: 50 K at the end
FLUX_RATE = 2.5e9                  # W/(m^2 s): 5e6 W/m^2 at the end
SLAB_PROBE_CELL = 5
BOX_PROBE = (0.5 * BOX_CELL, 0.5 * BOX_CELL)   # m, centre of cell (0, 0)


def _transient(time_step):
    return {'time_mode': 'transient', 'time_step': time_step}


def _box_spec(title, properties, source, steps):
    """Insulated 2 x 2 cell box, rho / cp / k from properties, heat_source Q = source, a cell-centre probe."""
    return with_exact_cg(ModelSpec(
        title=title, domain=DomainSpec(size_x=BOX, size_y=BOX),
        geometry=[GeometrySpec('probe_point', 'point', {'x': BOX_PROBE[0], 'y': BOX_PROBE[1]})],
        materials=[MaterialSpec('conductor', properties=dict(properties), selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, settings=_transient(SOURCE_STEP), features=[
            FeatureSpec('source', 'heat_source', ['all'], {'Q': source}),
            FeatureSpec('centre', 'temperature_probe', ['probe_point'])])],
        mesh=MeshSpec(mode='uniform', dx=BOX_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': steps * SOURCE_STEP})))


def _time_value_runs():
    """'source': Q of t; 'capacity': cp of t (a DynamicMaterialUpdate); 'slab': temperature and heat_flux of t."""
    material = {'k': CONDUCTIVITY, 'rho': DENSITY, 'cp': CAPACITY}
    runs = {'source': _box_spec('Self-test: heat source ramped in time', material, f'{Q_RAMP!r} * t', SOURCE_STEPS),
            'capacity': _box_spec('Self-test: heat capacity changing in time',
                                  {**material, 'cp': f'{CAPACITY!r} * (1 + t / {CP_TIME!r})'}, Q_CONSTANT,
                                  SOURCE_STEPS)}
    length = SLAB_CELLS * CELL
    runs['slab'] = with_exact_cg(ModelSpec(
        title='Self-test: slab face temperature and heat flux ramped in time',
        domain=DomainSpec(size_x=length, size_y=2.0 * CELL),
        geometry=[GeometrySpec('probe_point', 'point', {'x': (SLAB_PROBE_CELL + 0.5) * CELL, 'y': 0.5 * CELL})],
        materials=[MaterialSpec('conductor', properties=material, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, settings=_transient(SLAB_STEP), features=[
            FeatureSpec('ramp_wall', 'temperature', ['x_lo'], {'temperature': f'{REFERENCE_T!r} + {T_RATE!r} * t'}),
            FeatureSpec('ramp_flux', 'heat_flux', ['x_hi'], {'flux': f'{FLUX_RATE!r} * t'}),
            FeatureSpec('probe', 'temperature_probe', ['probe_point'])])],
        mesh=MeshSpec(mode='uniform', dx=CELL),
        study=StudySpec(type='time_dependent', settings={'duration': SLAB_STEPS * SLAB_STEP})))
    return runs


def _source_rises():
    """theta_n [K], n = 1..SOURCE_STEPS, of the ramped source (Q at the step start k dt)."""
    steps = np.arange(1, SOURCE_STEPS + 1)
    return Q_RAMP * SOURCE_STEP ** 2 * steps * (steps - 1) / (2.0 * DENSITY * CAPACITY)


def _capacity_rises():
    """theta_n [K] of the constant source with cp at the step start."""
    starts = SOURCE_STEP * np.arange(SOURCE_STEPS)
    increments = Q_CONSTANT * SOURCE_STEP / (DENSITY * CAPACITY * (1.0 + starts / CP_TIME))
    return np.cumsum(increments)


def _slab_rises():
    """(SLAB_STEPS, SLAB_CELLS) theta of the 1D finite-volume implicit Euler: interior conductance k / h, x_lo
    half cell 2 k / h at T_RATE t_n, x_hi inward flux FLUX_RATE t_n, capacity rho cp h (per unit face area)."""
    conductance = CONDUCTIVITY / CELL
    operator = np.zeros((SLAB_CELLS, SLAB_CELLS))
    for cell in range(SLAB_CELLS - 1):
        operator[cell, cell] += conductance
        operator[cell + 1, cell + 1] += conductance
        operator[cell, cell + 1] -= conductance
        operator[cell + 1, cell] -= conductance
    operator[0, 0] += 2.0 * conductance
    capacity = DENSITY * CAPACITY * CELL / SLAB_STEP
    system = operator + capacity * np.eye(SLAB_CELLS)
    theta = np.zeros(SLAB_CELLS)
    rises = []
    for step in range(SLAB_STEPS):
        start = step * SLAB_STEP
        rhs = capacity * theta
        rhs[0] += 2.0 * conductance * T_RATE * start
        rhs[-1] += FLUX_RATE * start
        theta = np.linalg.solve(system, rhs)
        rises.append(theta)
    return np.array(rises)


def _probe_error(results, feature, expected_rises, time_step):
    """(max |T - T0 - theta_n| / max theta, max |t_n - n dt| / dt) of a probe series; inf for a wrong sample count."""
    probe = results.result(PHYSICS_NAME, feature)
    values = np.asarray(probe['values'], dtype=np.float64)
    times = np.asarray(probe['t'], dtype=np.float64)
    if values.shape != expected_rises.shape:
        return math.inf, math.inf
    scale = float(np.max(np.abs(expected_rises)))
    expected_times = time_step * np.arange(1, expected_rises.size + 1)
    return (float(np.max(np.abs(values - REFERENCE_T - expected_rises))) / scale,
            float(np.max(np.abs(times - expected_times))) / time_step)


def _check_time_values(results_by_name):
    """Probe series of the three runs and the final slab profile against the recursions."""
    errors = {}
    errors['source'], errors['source_times'] = _probe_error(results_by_name['source'], 'centre', _source_rises(),
                                                            SOURCE_STEP)
    errors['capacity'], errors['capacity_times'] = _probe_error(results_by_name['capacity'], 'centre',
                                                                _capacity_rises(), SOURCE_STEP)
    slab = _slab_rises()
    errors['slab'], errors['slab_times'] = _probe_error(results_by_name['slab'], 'probe', slab[:, SLAB_PROBE_CELL],
                                                        SLAB_STEP)
    field = np.asarray(results_by_name['slab'].field(f'{PHYSICS_NAME}.T').values, dtype=np.float64)
    final = slab[-1]
    errors['slab_final_field'] = (float(np.max(np.abs(field - REFERENCE_T - final[:, None])))
                                  / float(np.max(np.abs(final))))
    lagged_first = Q_RAMP * SOURCE_STEP ** 2 / (DENSITY * CAPACITY)   # what an end-of-step Q would add in step 1
    return {'measured': float(slab[-1, SLAB_PROBE_CELL]), 'expected': float(slab[-1, SLAB_PROBE_CELL]),
            'error': float(np.max(np.asarray(list(errors.values())))), 'tolerance': TOLERANCE,
            'details': {'errors': errors, 'source_rise_end_K': float(_source_rises()[-1]),
                        'capacity_rise_end_K': float(_capacity_rises()[-1]), 'slab_face_rise_end_K': float(final[0]),
                        'end_of_step_source_first_rise_K': lagged_first}}


TIME_SELFTEST_CASES = (
    SelfTestCase(
        name='heat_time_dependent_values', label='Heat: source, heat capacity and boundary values of time t',
        description=('Transient heat (implicit Euler, k = 2 W/(m K), rho cp = 1e6 J/(m^3 K)) with values '
                     'of the time t, each re-evaluated at the start of every step (t = t_n for the step to t_n + '
                     f'dt). Run source: insulated {BOX * 1e6:g} um box of 2 x 2 cells (T stays uniform), heat_source '
                     f'Q = {Q_RAMP:g} t: uniform theta_n '
                     f'= Q_r dt^2 n (n - 1) / (2 rho cp) ({SOURCE_STEPS} steps of {SOURCE_STEP:g} s). Run capacity: the '
                     f'same box, Q = {Q_CONSTANT:g} W/m^3, material cp = {CAPACITY:g} (1 + t / {CP_TIME:g} s) (a '
                     'time-dependent DynamicMaterialUpdate): theta_n = sum_(k < n) Q dt / (rho cp(k dt)). Run slab: '
                     f'{SLAB_CELLS} cells of {CELL * 1e6:g} um along x, temperature T0 + {T_RATE:g} t on x_lo (the '
                     'in-place face rhs update), '
                     f'heat_flux {FLUX_RATE:g} t on x_hi: the exact 1D finite-volume implicit Euler with both face '
                     f'values at t_n ({SLAB_STEPS} steps of {SLAB_STEP:g} s), probe series and final profile. error = '
                     'max over the runs of |T - T0 - theta| / max theta and of the sample times; tolerance '
                     f'{TOLERANCE:g} (float32 T, CG to {EXACT_CG_TOLERANCE:g}). A value taken at the step end instead '
                     'misses by about 1 / n (5e-2 and more), a constant value by O(1). Not yet run at '
                     f'{SOURCE_STEPS} / {SLAB_STEPS} steps on the 2 x 2 cell box.'),
        specs=_time_value_runs, check=_check_time_values),
)
