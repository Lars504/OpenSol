"""Heat on triangle meshes (mesh plan stage T4, FEMHeatPhysics): a heated cylinder in a held coaxial shell (quadratic
and linear elements, curved outlines) against the closed form, the decay rate of a transient slab against the
implicit-Euler rate of the continuum mode, and the cpu vs gpu backends; model runs read through feature results
(probes) only. Registered by definition.register. heat_fem_host_direct: the disc heater on the MUMPS / PARDISO back ends
against SuperLU (skipped without python-mumps / pydiso)."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import (available_host_direct, skip_without_host_direct, solver_role, with_direct_node,
                                       with_exact_cg)
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)

HEAT_TYPE_NAME = 'heat'            # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'heat'
REFERENCE_T = 293.15               # K
UM = 1e-6

# ---- heat_fem_disc_heater: core r < a (k_core, Q) inside a shell a < r < b (k_shell) held at T0 on r = b ----
CENTRE = (1.6 * UM, 1.6 * UM)
CORE_RADIUS = 0.5 * UM             # a
SHELL_RADIUS = 1.5 * UM            # b
PROBE_RADIUS = 1.0 * UM            # second probe, in the shell
K_CORE = 1.4                       # W/(m K)
K_SHELL = 0.2
POWER_DENSITY = 1e14               # W/m^3: 25 W/m, rises ~73 K at the centre
# uniform element size per discretization: P2 at a / 5 (est. <= ~1.5e-5, 8-16x the 8.9e-7 recorded at a / 10), P1 at
# a / 7 (est. ~2.2e-3, h^2 from the 1.1e-3 recorded at a / 10)
DISC_ELEMENTS = {'quadratic': CORE_RADIUS / 5.0, 'linear': CORE_RADIUS / 7.0}
DISC_TOLERANCES = {'quadratic': 1e-4, 'linear': 5e-3}   # relative probe rise error

# ---- heat_fem_transient_slab: L x W, T0 on x_lo / x_hi, initial T0 + 100 K, insulated y faces ----
SLAB_LENGTH = 1e-3                 # m
SLAB_WIDTH = 0.25e-3
SLAB_ELEMENT = SLAB_LENGTH / 20.0
SLAB_PROPERTIES = {'k': 1.0, 'rho': 1000.0, 'cp': 1000.0}   # D = 1e-6 m^2/s
SLAB_STEP = 0.01                   # s: mu dt ~ 0.099
SLAB_STEPS = 30
SLAB_WINDOW = (20, 30)             # steps between which the decay rate is measured
INITIAL_RISE = 100.0               # K
# relative decay-rate error: P2 eigenvalue ~1e-6, float32 probes ~4e-6, third mode at step 20 ~7e-6
SLAB_TOLERANCE = 1e-4

_BACKEND_TOLERANCE = 1e-5          # relative probe-rise difference cpu vs gpu (float32 T, exact / 1e-12 solves)


def _probe_rise(results, feature):
    """Last sample of a temperature probe minus T0 [K]."""
    values = np.asarray(results.result(PHYSICS_NAME, feature)['values'], dtype=np.float64)
    return float(values[-1] - REFERENCE_T) if values.size else math.nan


# ---- disc heater ----
def disc_heater_spec(discretization='quadratic', backend='cpu') -> ModelSpec:
    """Heated core in a held coaxial shell on a uniform triangle mesh (DISC_ELEMENTS); probes at the centre and at
    PROBE_RADIUS."""
    cx, cy = CENTRE
    return ModelSpec(
        title=f'Self-test: heated cylinder in a held shell on triangles ({discretization})',
        domain=DomainSpec(size_x=2.0 * cx, size_y=2.0 * cy),
        geometry=[GeometrySpec('shell', 'circle', {'cx': cx, 'cy': cy, 'r': SHELL_RADIUS}),
                  GeometrySpec('core', 'circle', {'cx': cx, 'cy': cy, 'r': CORE_RADIUS}),
                  GeometrySpec('centre_point', 'point', {'x': cx, 'y': cy}),
                  GeometrySpec('shell_point', 'point', {'x': cx + PROBE_RADIUS, 'y': cy})],
        materials=[MaterialSpec('shell_material', properties={'k': K_SHELL}, selection=['shell']),
                   MaterialSpec('core_material', properties={'k': K_CORE}, selection=['core'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME,
                             settings={'T0': REFERENCE_T, 'discretization': discretization}, features=[
                                 FeatureSpec('sink', 'temperature', ['shell'], {'temperature': REFERENCE_T}),
                                 FeatureSpec('source', 'heat_source', ['core'], {'Q': POWER_DENSITY}),
                                 FeatureSpec('centre', 'temperature_probe', ['centre_point']),
                                 FeatureSpec('shell_probe', 'temperature_probe', ['shell_point'])])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=DISC_ELEMENTS[discretization])],
        solver_defaults=SolverDefaultsSpec(backend=backend),
        study=StudySpec(type='stationary'))


def _disc_expected():
    """Closed-form rises [K]: centre Q a^2 / (4 k_c) + Q a^2 / (2 k_s) ln(b / a); shell probe Q a^2 / (2 k_s) ln(b / r)."""
    line_power = POWER_DENSITY * CORE_RADIUS ** 2   # Q a^2 [W/m] (= P / pi)
    return {'centre': line_power / (4.0 * K_CORE) + line_power / (2.0 * K_SHELL) * math.log(SHELL_RADIUS / CORE_RADIUS),
            'shell_probe': line_power / (2.0 * K_SHELL) * math.log(SHELL_RADIUS / PROBE_RADIUS)}


def _disc_errors(results):
    expected = _disc_expected()
    return {name: abs(_probe_rise(results, name) / value - 1.0) for name, value in expected.items()}


def _disc_runs():
    return {order: disc_heater_spec(order) for order in DISC_TOLERANCES}


def _check_disc(results_by_name):
    """error = largest probe-rise relative error over its discretization's tolerance; tolerance 1."""
    errors = {order: _disc_errors(results_by_name[order]) for order in DISC_TOLERANCES}
    scaled = [value / DISC_TOLERANCES[order] for order, by_probe in errors.items() for value in by_probe.values()]
    error = max(scaled) if all(np.isfinite(scaled)) else math.nan
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'relative_errors': errors, 'tolerances': DISC_TOLERANCES, 'expected_rises_K': _disc_expected(),
                        'measured_rises_K': {order: {name: _probe_rise(results_by_name[order], name)
                                                     for name in ('centre', 'shell_probe')}
                                             for order in DISC_TOLERANCES}}}


# ---- transient slab ----
def _slab_runs():
    """'slab' on the Automatic host SuperLU, 'slab_pcg' on an exact pcg node (steps stop against their initial
    residual)."""
    return {'slab': _slab_spec(), 'slab_pcg': with_exact_cg(_slab_spec())}   # with_exact_cg modifies in place


def _slab_spec():
    """The transient slab on uniform P2 triangles, cpu backend."""
    length, width = SLAB_LENGTH, SLAB_WIDTH
    return ModelSpec(
        title='Self-test: transient slab on triangles',
        domain=DomainSpec(size_x=length, size_y=width),
        geometry=[GeometrySpec('probe_point', 'point', {'x': 0.5 * length, 'y': 0.5 * width})],
        materials=[MaterialSpec('solid', properties=dict(SLAB_PROPERTIES), selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME,
                             settings={'T0': REFERENCE_T, 'time_mode': 'transient', 'time_step': SLAB_STEP},
                             features=[
                                 FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi'], {'temperature': REFERENCE_T}),
                                 FeatureSpec('hot_start', 'initial_temperature', ['all'],
                                             {'temperature': REFERENCE_T + INITIAL_RISE}),
                                 FeatureSpec('centre', 'temperature_probe', ['probe_point'])])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=SLAB_ELEMENT)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='time_dependent', settings={'duration': SLAB_STEPS * SLAB_STEP}))


def _check_slab(results_by_name):
    """Decay rate of the centre probe between the SLAB_WINDOW steps against ln(1 + mu dt) / dt, mu = D pi^2 / L^2
    (the implicit-Euler rate of the slowest continuum mode) in both runs; error = the larger relative deviation."""
    checks = {name: _slab_rate(results_by_name[name]) for name in ('slab', 'slab_pcg')}
    errors = [check['error'] for check in checks.values()]
    error = max(errors) if all(np.isfinite(errors)) else math.nan
    return {'measured': checks['slab']['measured'], 'expected': checks['slab']['expected'], 'error': error,
            'tolerance': SLAB_TOLERANCE, 'details': checks}


def _decay_rate(results, time_step, window, properties, length):
    """(measured, expected, sampled rises, continuum rate mu, probe sample count) of the 'centre' probe of a transient
    slab (also selftests_fem3d): its decay rate between the window steps against ln(1 + mu dt) / dt, mu = D pi^2 / L^2
    (the implicit-Euler rate of the slowest continuum mode); measured NaN unless both rises are positive."""
    probe = results.result(PHYSICS_NAME, 'centre')
    times = np.asarray(probe['t'], dtype=np.float64)
    rises = np.asarray(probe['values'], dtype=np.float64) - REFERENCE_T
    samples = []
    for step in window:
        index = int(np.argmin(np.abs(times - step * time_step))) if times.size else -1
        found = index >= 0 and abs(times[index] - step * time_step) <= 1e-6 * time_step
        samples.append(rises[index] if found else math.nan)
    diffusivity = properties['k'] / (properties['rho'] * properties['cp'])
    mode_rate = diffusivity * math.pi ** 2 / length ** 2
    expected = math.log1p(mode_rate * time_step) / time_step
    measured = (math.log(samples[0] / samples[1]) / ((window[1] - window[0]) * time_step)
                if all(value > 0.0 for value in samples) else math.nan)
    return measured, expected, samples, mode_rate, int(times.size)


def _slab_rate(results):
    """Measured / expected decay rate and relative error of one slab run."""
    measured, expected, samples, mode_rate, n_samples = _decay_rate(results, SLAB_STEP, SLAB_WINDOW, SLAB_PROPERTIES,
                                                                    SLAB_LENGTH)
    return {'measured': measured, 'expected': expected, 'error': abs(measured / expected - 1.0), 'rises_K': samples,
            'steps': SLAB_WINDOW, 'continuum_rate_per_s': mode_rate, 'n_samples': n_samples}


# ---- cpu vs gpu ----
def _backend_runs():
    """The quadratic disc heater on the cpu (Automatic SuperLU), on the gpu (Automatic: cuDSS with nvmath-python, else
    SuperLU through cupyx) and on the gpu with an exact pcg node (device CSR, multigrid)."""
    return {'cpu': disc_heater_spec('quadratic', 'cpu'), 'gpu': disc_heater_spec('quadratic', 'gpu'),
            'gpu_pcg': with_exact_cg(disc_heater_spec('quadratic', 'gpu'))}


def _check_backends(results_by_name):
    """Largest relative probe-rise difference of the gpu runs against the cpu run; without the gpu runs the cpu run
    against the closed form (scaled onto this tolerance)."""
    cpu = results_by_name['cpu']
    gpu_runs = [name for name in ('gpu', 'gpu_pcg') if name in results_by_name]
    if not gpu_runs:
        relative = max(_disc_errors(cpu).values())
        error = relative / DISC_TOLERANCES['quadratic'] * _BACKEND_TOLERANCE
        return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _BACKEND_TOLERANCE,
                'details': {'gpu_runs': 'skipped (no usable GPU)', 'cpu_relative_error': relative}}
    differences = {}
    for name in gpu_runs:
        for probe in ('centre', 'shell_probe'):
            reference = _probe_rise(cpu, probe)
            differences[f'{name}.{probe}'] = abs(_probe_rise(results_by_name[name], probe) - reference) / reference
    error = max(differences.values()) if all(np.isfinite(list(differences.values()))) else math.nan
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _BACKEND_TOLERANCE,
            'details': {'relative_differences': differences}}


# ---- MUMPS / PARDISO 'linear' back ends (DiffusionDirectSolver: a real Cholesky / LDL^T) ----
# the installed ones of these back ends run (literals: the selftest rule); the reference is a scipy_splu node
_HOST_DIRECT_BACK_ENDS = ('mumps', 'pardiso', 'mumps_mixed')
_HOST_DIRECT_REFERENCE = 'scipy_splu'
_HOST_DIRECT_TOLERANCE = 1e-5      # relative probe-rise difference (float32 T; both solves exact to ~1e-12)


def _host_direct_runs():
    """The quadratic disc heater on the cpu with a scipy_splu node and one node per installed back end of
    _HOST_DIRECT_BACK_ENDS; {} without python-mumps / pydiso."""
    back_ends = available_host_direct(_HOST_DIRECT_BACK_ENDS)
    if not back_ends:
        return {}
    return {back_end: with_direct_node(disc_heater_spec('quadratic', 'cpu'), back_end, physics_names=(PHYSICS_NAME,))
            for back_end in (_HOST_DIRECT_REFERENCE, *back_ends)}


def _check_host_direct(results_by_name):
    """Per host direct back end: relative probe-rise differences from the SuperLU run; its 'linear' record must show
    that back end with a symmetric (Cholesky / LDL^T) factor. error = the largest difference (inf on a wrong record)."""
    if not results_by_name:
        return skip_without_host_direct(_HOST_DIRECT_BACK_ENDS)
    reference = results_by_name[_HOST_DIRECT_REFERENCE]
    differences, records = {}, {}
    for back_end, results in results_by_name.items():
        if back_end == _HOST_DIRECT_REFERENCE:
            continue
        for probe in ('centre', 'shell_probe'):
            expected = _probe_rise(reference, probe)
            differences[f'{back_end}.{probe}'] = abs(_probe_rise(results, probe) - expected) / expected
        counters = solver_role(results, PHYSICS_NAME).get('counters') or {}
        records[back_end] = dict(counters)
        if counters.get('back_end') != back_end or not int(counters.get('host_direct_symmetric_factors') or 0):
            differences[f'{back_end}.record'] = math.inf
    error = max(differences.values()) if all(not math.isnan(value) for value in differences.values()) else math.nan
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _HOST_DIRECT_TOLERANCE,
            'details': {'relative_differences': differences, 'counters': records}}


FEM_SELFTEST_CASES = (
    SelfTestCase(
        name='heat_fem_host_direct', label='Heat on triangles: MUMPS / PARDISO linear solver vs SuperLU',
        description=('The quadratic heat_fem_disc_heater model (h = a / 5, cpu backend) with a sparse_direct node '
                     'pinned to scipy_splu (host SuperLU, the reference) and one node per installed back end of mumps, '
                     'pardiso, mumps_mixed as the linear solver: DiffusionDirectSolver factorises the SPD operator by '
                     'the library (PARDISO Cholesky, MUMPS LDL^T; mixed: float32 factor + float64 GMRES refinement). '
                     'Relative probe-rise differences from SuperLU at the centre and the shell probe; the linear '
                     'record must show that back end with a symmetric factor. Tolerance '
                     f'{_HOST_DIRECT_TOLERANCE:g} (float32 probes); passes as skipped without python-mumps / pydiso. '
                     'Not run yet.'),
        specs=_host_direct_runs, check=_check_host_direct),
    SelfTestCase(
        name='heat_fem_disc_heater', label='Heat on triangles: heated cylinder in a held shell',
        description=('Stationary heat on a uniform triangle mesh (h = a / 5 quadratic, a / 7 linear): a core r < a = '
                     f'{CORE_RADIUS / UM:g} um (k = {K_CORE:g} W/(m K), Q = {POWER_DENSITY:g} W/m^3) inside a shell a < '
                     f'r < b = {SHELL_RADIUS / UM:g} um (k = {K_SHELL:g}) whose outline is held at T0 (temperature on '
                     'the shell object: Dirichlet on its curved outline edges); the elements outside the shell have no '
                     'k and drop out. Expected: theta(0) = Q a^2 / (4 k_c) + Q a^2 / (2 k_s) ln(b / a) and theta(r) = '
                     f'Q a^2 / (2 k_s) ln(b / r) at r = {PROBE_RADIUS / UM:g} um (probe nodes). Runs quadratic '
                     '(isoparametric P2 on the circles) and linear (P1, straight chords). error = largest relative '
                     'probe error over its tolerance (quadratic 1e-4, linear 5e-3); tolerance 1. At h = a / 10 the '
                     'errors were 8.9e-7 (P2, the float32 floor) and 1.1e-3 (P1); estimated here <= ~1.5e-5 (P2) and '
                     '~2.2e-3 (P1). Not yet run at a / 5 and a / 7.'),
        specs=_disc_runs, check=_check_disc),
    SelfTestCase(
        name='heat_fem_transient_slab', label='Heat on triangles: transient slab decay rate',
        description=(f'Transient heat (implicit Euler, dt = {SLAB_STEP:g} s, P2) on a {SLAB_LENGTH * 1e3:g} x '
                     f'{SLAB_WIDTH * 1e3:g} mm slab (h = L / 20, k = 1, rho cp = 1e6: D = 1e-6 m^2/s), T0 held on x_lo '
                     f'/ x_hi, initial T0 + {INITIAL_RISE:g} K, insulated y faces: after the higher modes died out the '
                     f'centre rise decays by 1 / (1 + mu_h dt) per step. The rate ln(theta_{SLAB_WINDOW[0]} / '
                     f'theta_{SLAB_WINDOW[1]}) / ({SLAB_WINDOW[1] - SLAB_WINDOW[0]} dt) is compared with ln(1 + mu dt) '
                     '/ dt of the continuum mode mu = D pi^2 / L^2 (pins stiffness, consistent capacity mass and the '
                     f'step). Runs slab (Automatic host SuperLU) and slab_pcg (an exact pcg node: each step stops '
                     'against its initial residual, PHY-1); error = the larger deviation. Tolerance '
                     f'{SLAB_TOLERANCE:g} (third mode at step {SLAB_WINDOW[0]} ~7e-6, float32 probes ~4e-6). Not yet '
                     f'run with {SLAB_STEPS} steps.'),
        specs=_slab_runs, check=_check_slab),
    SelfTestCase(
        name='heat_fem_cpu_gpu', label='Heat on triangles: CPU backend vs GPU',
        description=('The quadratic heat_fem_disc_heater model (h = a / 5) on solver_defaults.backend cpu '
                     '(Automatic host SuperLU) against gpu with the Automatic sparse_direct (a double cuDSS Cholesky '
                     'with nvmath-python, else SuperLU via cupyx) and with an exact pcg node '
                     '(device CSR, double, tolerance 1e-12): relative probe-rise differences. Tolerance '
                     f'{_BACKEND_TOLERANCE:g} (float32 probes). Without a usable GPU the cpu run is checked against '
                     'the closed form instead.'),
        specs=_backend_runs, check=_check_backends, backends=('cpu', 'gpu')),
)
