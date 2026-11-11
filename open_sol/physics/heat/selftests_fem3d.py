"""Heat on tetrahedron meshes (FEMHeatPhysics3D, builtin Kuhn 'tetrahedra' meshes, no gmsh): a heated slab between held
faces (quadratic and linear) and with a convective face (quadratic) against the closed forms, a temperature held on an
object's interface faces, the decay rate of a transient slab, and the cpu vs gpu backends; model runs read through
feature results (probes) only. Registered by definition.register."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)
from .selftests_fem import _decay_rate, _probe_rise   # same PHYSICS_NAME and REFERENCE_T

HEAT_TYPE_NAME = 'heat'            # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'heat'
REFERENCE_T = 293.15               # K
UM = 1e-6

# ---- heat_fem3d_slab: L x W x W, Q uniform, x_lo held at T0, x_hi held (or convection to T0) ----
SLAB_SIZE = (1.0 * UM, 0.5 * UM, 0.5 * UM)
SLAB_CELL = 0.25 * UM              # builtin Kuhn cubes: probes on nodes
CONDUCTIVITY = 1.4                 # W/(m K)
POWER_DENSITY = 1e15               # W/m^3: Q L^2 / (8 k) ~ 89 K
TRANSFER_COEFFICIENT = 2e6         # W/(m^2 K): h L / k ~ 1.4
PROBE_X = {'centre': 0.5 * UM, 'quarter': 0.25 * UM}
# quadratic: exact (P2 holds the parabola); linear: nodally exact on the uniform Kuhn split (its P1 stiffness is the
# 7-point stencil, the consistent load Q h^3 per node); float32 probes ~1e-7: first estimates
SLAB_TOLERANCE = 1e-5              # relative probe-rise error

# ---- heat_fem3d_interface: object 'right' (x > L / 2) held at T0 on its interface faces, x_lo at T0 + dT ----
INTERFACE_RISE = 40.0              # K
INTERFACE_TOLERANCE = 1e-5         # relative to dT (linear field: exact for both orders, float32 probes)

# ---- heat_fem3d_transient_slab: L x W x W, T0 held on x_lo / x_hi, initial T0 + 100 K ----
TRANSIENT_SIZE = (1e-3, 0.25e-3, 0.25e-3)
TRANSIENT_CELL = 1e-3 / 8.0
TRANSIENT_PROPERTIES = {'k': 1.0, 'rho': 1000.0, 'cp': 1000.0}   # D = 1e-6 m^2/s
TRANSIENT_STEP = 0.01              # s: mu dt ~ 0.099
TRANSIENT_STEPS = 30
TRANSIENT_WINDOW = (20, 30)
INITIAL_RISE = 100.0               # K
# relative decay-rate error: P2 eigenvalue at h = L / 8 ~ (pi h / L)^4 / 1440 ~ 2e-5 (1D estimate), third mode at
# step 20 ~7e-6, float32 probes ~4e-6: first estimate
TRANSIENT_TOLERANCE = 3e-4

_BACKEND_TOLERANCE = 1e-5          # relative probe-rise difference cpu vs gpu (float32 T, exact solves)


def _probe_points(size):
    """GeometrySpecs of the slab probes at PROBE_X on the centre line (y, z = W / 2)."""
    return [GeometrySpec(f'{name}_point', 'point', {'x': x, 'y': 0.5 * size[1], 'z': 0.5 * size[2]})
            for name, x in PROBE_X.items()]


def _probe_features():
    return [FeatureSpec(name, 'temperature_probe', [f'{name}_point']) for name in PROBE_X]


def _tetrahedra_mesh(cell):
    return [MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=cell, mesher='builtin')]


# ---- heated slab ----
def slab_spec(discretization='quadratic', far_face='temperature', backend='cpu') -> ModelSpec:
    """Uniformly heated slab on a builtin tetrahedra mesh: x_lo at T0, x_hi held at T0 or cooled toward T0."""
    far = (FeatureSpec('far', 'temperature', ['x_hi'], {'temperature': REFERENCE_T}) if far_face == 'temperature'
           else FeatureSpec('far', 'convection', ['x_hi'], {'h': TRANSFER_COEFFICIENT, 'T_ext': REFERENCE_T}))
    return ModelSpec(
        title=f'Self-test: heated slab on tetrahedra ({discretization}, {far_face})',
        domain=DomainSpec(size_x=SLAB_SIZE[0], size_y=SLAB_SIZE[1], size_z=SLAB_SIZE[2]),
        geometry=_probe_points(SLAB_SIZE),
        materials=[MaterialSpec('solid', properties={'k': CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME,
                             settings={'T0': REFERENCE_T, 'discretization': discretization}, features=[
                                 FeatureSpec('sink', 'temperature', ['x_lo'], {'temperature': REFERENCE_T}), far,
                                 FeatureSpec('source', 'heat_source', ['all'], {'Q': POWER_DENSITY}),
                                 *_probe_features()])],
        meshes=_tetrahedra_mesh(SLAB_CELL),
        solver_defaults=SolverDefaultsSpec(backend=backend),
        study=StudySpec(type='stationary'))


def _slab_expected(far_face):
    """Rises [K] at the probes: Q x (L - x) / (2 k) between held faces; with h (T - T0) at x_hi, T0 + a x - Q x^2 /
    (2 k), a = Q L (1 + h L / (2 k)) / (k + h L)."""
    length, q, k, h = SLAB_SIZE[0], POWER_DENSITY, CONDUCTIVITY, TRANSFER_COEFFICIENT
    if far_face == 'temperature':
        return {name: q * x * (length - x) / (2.0 * k) for name, x in PROBE_X.items()}
    slope = q * length * (1.0 + h * length / (2.0 * k)) / (k + h * length)
    return {name: slope * x - q * x * x / (2.0 * k) for name, x in PROBE_X.items()}


_SLAB_RUNS = {'quadratic': ('quadratic', 'temperature'), 'linear': ('linear', 'temperature'),
              'quadratic_convection': ('quadratic', 'convection')}


def _check_slab(results_by_name):
    """error = largest relative probe-rise error over the runs."""
    errors, measured = {}, {}
    for run, (_, far_face) in _SLAB_RUNS.items():
        expected = _slab_expected(far_face)
        measured[run] = {name: _probe_rise(results_by_name[run], name) for name in expected}
        errors[run] = {name: abs(measured[run][name] / value - 1.0) for name, value in expected.items()}
    values = [value for by_probe in errors.values() for value in by_probe.values()]
    error = max(values) if all(np.isfinite(values)) else math.nan
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': SLAB_TOLERANCE,
            'details': {'relative_errors': errors, 'measured_rises_K': measured,
                        'expected_rises_K': {run: _slab_expected(far) for run, (_, far) in _SLAB_RUNS.items()}}}


# ---- held object interface ----
def _interface_spec(discretization):
    length, width, height = SLAB_SIZE
    return ModelSpec(
        title=f'Self-test: temperature on object interface faces ({discretization})',
        domain=DomainSpec(size_x=length, size_y=width, size_z=height),
        geometry=[GeometrySpec('right', 'box', {'x0': 0.5 * length, 'y0': 0.0, 'z0': 0.0,
                                                'x1': length, 'y1': width, 'z1': height}),
                  *_probe_points(SLAB_SIZE)],
        materials=[MaterialSpec('solid', properties={'k': CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME,
                             settings={'T0': REFERENCE_T, 'discretization': discretization}, features=[
                                 FeatureSpec('hot', 'temperature', ['x_lo'],
                                             {'temperature': REFERENCE_T + INTERFACE_RISE}),
                                 FeatureSpec('held_object', 'temperature', ['right'], {'temperature': REFERENCE_T}),
                                 *_probe_features()])],
        meshes=_tetrahedra_mesh(SLAB_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'))


def _check_interface(results_by_name):
    """Linear T from T0 + dT at x = 0 to T0 at the interface x = L / 2: rises dT / 2 (quarter) and 0 (centre: on the
    held interface); error = largest deviation / dT."""
    expected = {'quarter': 0.5 * INTERFACE_RISE, 'centre': 0.0}
    deviations = {run: {name: abs(_probe_rise(results, name) - value) / INTERFACE_RISE
                        for name, value in expected.items()} for run, results in results_by_name.items()}
    values = [value for by_probe in deviations.values() for value in by_probe.values()]
    error = max(values) if all(np.isfinite(values)) else math.nan
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': INTERFACE_TOLERANCE,
            'details': {'relative_deviations': deviations, 'expected_rises_K': expected}}


# ---- transient slab ----
def _transient_runs():
    length, width, height = TRANSIENT_SIZE
    return {'slab': ModelSpec(
        title='Self-test: transient slab on tetrahedra',
        domain=DomainSpec(size_x=length, size_y=width, size_z=height),
        geometry=[GeometrySpec('probe_point', 'point', {'x': 0.5 * length, 'y': 0.5 * width, 'z': 0.5 * height})],
        materials=[MaterialSpec('solid', properties=dict(TRANSIENT_PROPERTIES), selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME,
                             settings={'T0': REFERENCE_T, 'time_mode': 'transient', 'time_step': TRANSIENT_STEP},
                             features=[
                                 FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi'], {'temperature': REFERENCE_T}),
                                 FeatureSpec('hot_start', 'initial_temperature', ['all'],
                                             {'temperature': REFERENCE_T + INITIAL_RISE}),
                                 FeatureSpec('centre', 'temperature_probe', ['probe_point'])])],
        meshes=_tetrahedra_mesh(TRANSIENT_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='time_dependent', settings={'duration': TRANSIENT_STEPS * TRANSIENT_STEP}))}


def _check_transient(results_by_name):
    """Decay rate of the centre probe between the TRANSIENT_WINDOW steps against ln(1 + mu dt) / dt, mu = D pi^2 / L^2;
    error = its relative deviation."""
    measured, expected, samples, mode_rate, n_samples = _decay_rate(
        results_by_name['slab'], TRANSIENT_STEP, TRANSIENT_WINDOW, TRANSIENT_PROPERTIES, TRANSIENT_SIZE[0])
    error = abs(measured / expected - 1.0)
    return {'measured': measured, 'expected': expected, 'error': error, 'tolerance': TRANSIENT_TOLERANCE,
            'details': {'rises_K': samples, 'steps': TRANSIENT_WINDOW, 'continuum_rate_per_s': mode_rate,
                        'n_samples': n_samples}}


# ---- cpu vs gpu ----
def _backend_runs():
    """The quadratic held slab on the cpu (Automatic host SuperLU) and on the gpu (Automatic: cuDSS with nvmath-python,
    else SuperLU via cupyx)."""
    return {'cpu': slab_spec('quadratic', 'temperature', 'cpu'), 'gpu': slab_spec('quadratic', 'temperature', 'gpu')}


def _check_backends(results_by_name):
    """Largest relative probe-rise difference gpu vs cpu; without the gpu run the cpu run against the closed form."""
    cpu = results_by_name['cpu']
    expected = _slab_expected('temperature')
    if 'gpu' not in results_by_name:
        error = max(abs(_probe_rise(cpu, name) / value - 1.0) for name, value in expected.items())
        return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _BACKEND_TOLERANCE,
                'details': {'gpu_run': 'skipped (no usable GPU)'}}
    differences = {name: abs(_probe_rise(results_by_name['gpu'], name) / _probe_rise(cpu, name) - 1.0)
                   for name in expected}
    error = max(differences.values()) if all(np.isfinite(list(differences.values()))) else math.nan
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _BACKEND_TOLERANCE,
            'details': {'relative_differences': differences}}


FEM_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='heat_fem3d_slab', label='Heat on tetrahedra: heated slab',
        description=(f'Stationary heat on a builtin tetrahedra mesh (Kuhn cubes of {SLAB_CELL / UM:g} um, no gmsh) of a '
                     f'{SLAB_SIZE[0] / UM:g} x {SLAB_SIZE[1] / UM:g} x {SLAB_SIZE[2] / UM:g} um slab (k = '
                     f'{CONDUCTIVITY:g}, Q = {POWER_DENSITY:g} W/m^3, insulated y / z faces), x_lo held at T0 and x_hi '
                     f'held at T0 (quadratic and linear) or cooled by h = {TRANSFER_COEFFICIENT:g} toward T0 '
                     '(quadratic): probe rises at x = L / 4 and L / 2 (mesh nodes) against Q x (L - x) / (2 k) / the '
                     'Robin parabola. Quadratic elements hold the parabola exactly; linear ones are nodally exact on '
                     'the uniform Kuhn split (7-point stiffness). error = largest relative probe error, tolerance '
                     f'{SLAB_TOLERANCE:g} (float32 probes; first estimate, not yet run).'),
        specs=lambda: {run: slab_spec(*settings) for run, settings in _SLAB_RUNS.items()}, check=_check_slab),
    SelfTestCase(
        name='heat_fem3d_interface', label='Heat on tetrahedra: temperature on object interface faces',
        description=('The slab geometry without source: the box object right (x > L / 2) held at T0 by a temperature '
                     'feature on the object (Dirichlet on its interface faces, mesh.object_faces), x_lo at T0 + '
                     f'{INTERFACE_RISE:g} K: the left half carries a linear profile (probe at L / 4: dT / 2; at L / '
                     '2, on the interface: 0), quadratic and linear elements. error = largest deviation / dT, '
                     f'tolerance {INTERFACE_TOLERANCE:g} (first estimate, not yet run).'),
        specs=lambda: {order: _interface_spec(order) for order in ('quadratic', 'linear')}, check=_check_interface),
    SelfTestCase(
        name='heat_fem3d_transient_slab', label='Heat on tetrahedra: transient slab decay rate',
        description=(f'Transient heat (implicit Euler, dt = {TRANSIENT_STEP:g} s, P2) on a builtin tetrahedra mesh '
                     f'(h = L / 8) of a {TRANSIENT_SIZE[0] * 1e3:g} x {TRANSIENT_SIZE[1] * 1e3:g} x '
                     f'{TRANSIENT_SIZE[2] * 1e3:g} mm slab (k = 1, rho cp = 1e6), T0 held on x_lo / x_hi, initial T0 '
                     f'+ {INITIAL_RISE:g} K: the centre decay rate between steps {TRANSIENT_WINDOW[0]} and '
                     f'{TRANSIENT_WINDOW[1]} against ln(1 + mu dt) / dt of the continuum mode mu = D pi^2 / L^2. '
                     f'Tolerance {TRANSIENT_TOLERANCE:g} (first estimate, not yet run).'),
        specs=_transient_runs, check=_check_transient),
    SelfTestCase(
        name='heat_fem3d_cpu_gpu', label='Heat on tetrahedra: CPU backend vs GPU',
        description=('The quadratic held heat_fem3d_slab on solver_defaults.backend cpu (Automatic host SuperLU) and '
                     'gpu (Automatic sparse_direct: a double cuDSS Cholesky with nvmath-python, else SuperLU; the gpu '
                     'run assembles on the host and moves the operator through cupyx): relative '
                     f'probe-rise differences, tolerance {_BACKEND_TOLERANCE:g}. Without a usable GPU the cpu run is '
                     'checked against the closed form instead.'),
        specs=_backend_runs, check=_check_backends, backends=('cpu', 'gpu')),
)
