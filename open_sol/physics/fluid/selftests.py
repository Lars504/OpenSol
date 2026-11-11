"""Analytic self-tests of laminar_flow (force- and pressure-driven Poiseuille channels, plug flow carrying heat through
fluid_heat_convection) and the 'heated_microchannel' template; specs built fresh."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import solver_role, with_exact_cg
from ...model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, SolverSpec, StudySpec)

FLUID_TYPE_NAME = 'laminar_flow'      # = definition.FLUID_TYPE_NAME (selftests never import the definition)
CONVECTION_TYPE_NAME = 'fluid_heat_convection'
_FLUID = 'fluid'                      # physics instance names = the types' default instances
_HEAT = 'heat'
_RUN = 'run'

# Shared channel: uniform 1 um cells, a liquid of viscosity 1e-3 Pa s and density 1000 kg/m^3 (nu = 1e-6 m^2/s, so
# relaxation_time 1 gives dt_lbm = 0.5 dx^2 / (3 nu) = 1.67e-7 s), steady tolerance 1e-6 (the float32 lattice stalls
# near 1e-7..1e-6, so a smaller one only stops at that floor). Cell centres y_j = (j + 1/2)
# dx; the halfway bounce-back walls sit on the y faces, where TRT with the magic parameter 3/16 makes the
# Poiseuille parabola exact at the cell centres (only float32 storage and the steady tolerance remain).
_CELL = 1e-6                          # m
_VISCOSITY = 1e-3                     # Pa s
_DENSITY = 1000.0                     # kg/m^3
_STEADY_SETTINGS = {'tolerance': 1e-6}
_CHANNEL_HEIGHT = 20e-6               # m (20 cells): pressure and start-up channels
_PEAK_VELOCITY = 5e-3                 # m/s, centreline speed of all Poiseuille cases (lattice Mach ~1.4e-3)
# force channel: 8 x 10 cells (the steady march length grows as the height^2; 8 columns still test x-invariance and
# the periodic wrap)
_FORCE_LENGTH = 8e-6                  # m, periodic channel
_FORCE_HEIGHT = 10e-6                 # m
_PRESSURE_LENGTH = 40e-6              # m, pressure-driven channel
# force density / pressure gradient giving the peak velocity: u_max = G H^2 / (8 mu)
_DRIVE_GRADIENT = 8.0 * _VISCOSITY * _PEAK_VELOCITY / _CHANNEL_HEIGHT ** 2   # 1e5 N/m^3 = Pa/m
_FORCE_GRADIENT = 8.0 * _VISCOSITY * _PEAK_VELOCITY / _FORCE_HEIGHT ** 2     # 4e5 N/m^3
_PRESSURE_DROP = _DRIVE_GRADIENT * _PRESSURE_LENGTH                           # 4 Pa
_FORCE_TOLERANCE = 1e-3               # exact scheme: ~1e-6 expected, the margin covers a forcing-scheme slip error
_PRESSURE_TOLERANCE = 1e-2            # anti-bounce-back pressure faces (exact for Poiseuille with the TRT shear term at
                                      # the link midpoints up to O(u^2) and float32); ~0.15 without that term, ~0.013
                                      # with it taken at the boundary cell
# Start-up with half_storage populations: 10 x 20 cells, 240 lattice steps (dt_lbm = 1/6 us) ~ one decay time of the
# slowest mode (tau_v = H^2 / (pi^2 nu) ~ 40.5 us); single precision gives ~2e-3 (lattice diffusion of the modes,
# the end time within a step), fp16 storage adds its rounding noise (~5e-4 of each population per step)
_STARTUP_LENGTH = 10e-6               # m, periodic channel
_STARTUP_DURATION = 240 * 0.5 * _CELL ** 2 / (3.0 * _VISCOSITY / _DENSITY)   # s, 240 lattice steps at tau = 1
_STARTUP_TOLERANCE = 2e-2

# Plug flow with heat: free-slip y faces, uniform inlet U on x_lo, pressure 0 on x_hi (uniform u = U); heat T_in on
# x_lo, T_out on x_hi, insulated y faces; Peclet number Pe = rho cp U L / k = 1.
_PLUG_LENGTH = 100e-6                 # m (100 cells)
_PLUG_HEIGHT = 10e-6                  # m (10 cells)
_PLUG_CONDUCTIVITY = 0.6              # W/(m K)
_PLUG_HEAT_CAPACITY = 4000.0          # J/(kg K)
_PECLET = 1.0
_PLUG_VELOCITY = _PECLET * _PLUG_CONDUCTIVITY / (_DENSITY * _PLUG_HEAT_CAPACITY * _PLUG_LENGTH)   # 1.5e-3 m/s
_T_IN, _T_OUT = 293.15, 303.15        # K
_PLUG_TOLERANCE = 2e-3                # central advection O(dx^2) + the one-sided inflow face cell
_PLUG_LOOP_TOLERANCE = 1e-5           # Picard loop: error ~ change g / (1 - g) ~ 5e-6 at gain g ~ Pe / pi; 1e-7 sat on
                                      # the float32 field floor (one ulp ~1e-7 of the rise)
# The flow starts at the answer (initial_velocity u_x = U): uniform U at pressure 0 is a fixed point of correct face
# rules, so the steady march stops at the float32 floor after ~3 checks instead of ~280k steps from rest (decay length
# 200 cells: one pressure face on x); wrong face rules move it at the first check and the march grades their state.


def _liquid(selection, **extra):
    return MaterialSpec('liquid', properties={'viscosity': _VISCOSITY, 'rho': _DENSITY, **extra}, selection=selection)


def _channel_model(title, length, height, fluid_features, materials, physics=(), couplings=(), study=None):
    """Stationary laminar_flow channel model on the uniform _CELL mesh (no PML)."""
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=length, size_y=height),
        materials=list(materials),
        physics=[PhysicsSpec(_FLUID, FLUID_TYPE_NAME, settings=dict(_STEADY_SETTINGS), features=list(fluid_features)),
                 *physics],
        couplings=list(couplings),
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=study or StudySpec(type='stationary'))


def _force_specs():
    """'run': periodic x, no-slip y faces, volume force G along x."""
    return {_RUN: _channel_model(
        'Self-test: force-driven Poiseuille channel (laminar flow)', _FORCE_LENGTH, _FORCE_HEIGHT,
        [FeatureSpec('periodic_x', 'periodic', ['x_lo', 'x_hi']),
         FeatureSpec('walls', 'wall', ['y_lo', 'y_hi']),
         FeatureSpec('drive', 'body_force', ['all'], {'f_x': _FORCE_GRADIENT, 'f_y': 0.0})],
        [_liquid(['all'])])}


def _startup_specs(precision):
    """'run': the force channel (_STARTUP_LENGTH long) started from rest, transient; half_storage on a stepper node,
    single on the Automatic stepper."""
    spec = _channel_model(
        f'Self-test: Poiseuille start-up, {precision} populations (laminar flow)', _STARTUP_LENGTH, _CHANNEL_HEIGHT,
        [FeatureSpec('periodic_x', 'periodic', ['x_lo', 'x_hi']),
         FeatureSpec('walls', 'wall', ['y_lo', 'y_hi']),
         FeatureSpec('drive', 'body_force', ['all'], {'f_x': _DRIVE_GRADIENT, 'f_y': 0.0})],
        [_liquid(['all'])], study=StudySpec(type='time_dependent', settings={'duration': _STARTUP_DURATION}))
    spec.physics[0].settings = {'time_mode': 'transient'}
    if precision != 'single':
        spec.solvers = [SolverSpec('stepper_populations', 'explicit_kernels', {'precision': precision})]
        spec.physics[0].solvers = {'stepper': 'stepper_populations'}
    return {_RUN: spec}


def _startup_profile(y_centres, time):
    """Force-driven start-up from rest: the parabola minus the odd sine modes decaying as exp(-n^2 t / tau_v)."""
    steady = _poiseuille(y_centres, _DRIVE_GRADIENT, _CHANNEL_HEIGHT)
    tau_v = _CHANNEL_HEIGHT ** 2 / (math.pi ** 2 * _VISCOSITY / _DENSITY)
    transient = sum(np.sin(n * math.pi * y_centres / _CHANNEL_HEIGHT) * math.exp(-n * n * time / tau_v) / n ** 3
                    for n in range(1, 40, 2))
    return steady - 4.0 * _DRIVE_GRADIENT * _CHANNEL_HEIGHT ** 2 / (_VISCOSITY * math.pi ** 3) * transient


def _check_startup(results_by_name, expected_precision):
    """u_x after _STARTUP_DURATION against the start-up series in every column, u_y = 0, relative to the peak; raises
    unless summary 'solvers' records the stepper at expected_precision."""
    results = results_by_name[_RUN]
    precision = solver_role(results, _FLUID, 'stepper').get('precision')
    if precision != expected_precision:
        raise ValueError(f"expected the stepper to run {expected_precision}, summary 'solvers' records {precision!r}")
    errors, velocity_x, _, expected = _channel_errors(results, _STARTUP_LENGTH, _CHANNEL_HEIGHT,
                                                      lambda y: _startup_profile(y, _STARTUP_DURATION))
    return {'measured': float(np.max(velocity_x)), 'expected': float(np.max(expected)),
            'error': max(errors.values()), 'tolerance': _STARTUP_TOLERANCE,
            'details': {'errors': errors, 'duration_s': _STARTUP_DURATION, 'n_cells': list(velocity_x.shape)}}


def _pressure_specs():
    """'run': pressure dp on x_lo, 0 on x_hi, no-slip y faces."""
    return {_RUN: _channel_model(
        'Self-test: pressure-driven Poiseuille channel (laminar flow)', _PRESSURE_LENGTH, _CHANNEL_HEIGHT,
        [FeatureSpec('inlet', 'pressure', ['x_lo'], {'pressure': _PRESSURE_DROP}),
         FeatureSpec('outlet', 'pressure', ['x_hi'], {'pressure': 0.0}),
         FeatureSpec('walls', 'wall', ['y_lo', 'y_hi'])],
        [_liquid(['all'])])}


def _plug_specs():
    """'run': plug flow (slip walls) carrying heat between two held temperatures, fluid_heat_convection."""
    return {_RUN: with_exact_cg(_channel_model(
        'Self-test: plug flow advection-diffusion (laminar flow + heat)', _PLUG_LENGTH, _PLUG_HEIGHT,
        [FeatureSpec('inlet', 'inlet_velocity', ['x_lo'], {'velocity': _PLUG_VELOCITY}),
         FeatureSpec('outlet', 'pressure', ['x_hi'], {'pressure': 0.0}),
         FeatureSpec('symmetry', 'slip_wall', ['y_lo', 'y_hi']),
         FeatureSpec('start', 'initial_velocity', ['all'], {'u_x': _PLUG_VELOCITY, 'u_y': 0.0, 'pressure': 0.0})],
        [_liquid(['all'], k=_PLUG_CONDUCTIVITY, cp=_PLUG_HEAT_CAPACITY)],
        physics=[PhysicsSpec(_HEAT, 'heat', features=[
            FeatureSpec('inflow', 'temperature', ['x_lo'], {'temperature': _T_IN}),
            FeatureSpec('outflow', 'temperature', ['x_hi'], {'temperature': _T_OUT})])],
        couplings=[CouplingSpec('convection', CONVECTION_TYPE_NAME, _FLUID, _HEAT)],
        study=StudySpec(type='stationary', settings={'tolerance': _PLUG_LOOP_TOLERANCE, 'max_iterations': 200})))}


# ---- checks (CPU, on Results-like objects) ----
def _window_field(results, field_name, shape):
    """float64 (nx, ny) window values of a field and its float64 x / y cell centres; ValueError for another shape."""
    field_data = results.field(field_name)
    values = np.asarray(field_data.values, dtype=np.float64)
    if values.shape != shape:
        raise ValueError(f'{field_name}: expected the window shape {shape}, got {values.shape}')
    return values, np.asarray(field_data.x_centers, dtype=np.float64), np.asarray(field_data.y_centers, dtype=np.float64)


def _cells(length):
    return round(length / _CELL)


def _poiseuille(y_centres, gradient, height):
    """Exact Poiseuille velocity [m/s] at y between no-slip walls y = 0 and y = height, driving gradient G [N/m^3]."""
    return gradient / (2.0 * _VISCOSITY) * y_centres * (height - y_centres)


def _channel_errors(results, length, height, profile):
    """(errors dict, u_x window, x centres, expected u_x at the y centres) of a channel: u_x against profile(y
    centres) in every column, u_y against 0, both relative to the peak velocity."""
    shape = (_cells(length), _cells(height))
    velocity_x, x_centres, y_centres = _window_field(results, f'{_FLUID}.u_x', shape)
    velocity_y, _, _ = _window_field(results, f'{_FLUID}.u_y', shape)
    expected = profile(y_centres)
    errors = {'u_x': float(np.max(np.abs(velocity_x - expected[None, :]))) / _PEAK_VELOCITY,
              'u_y': float(np.max(np.abs(velocity_y))) / _PEAK_VELOCITY}
    return errors, velocity_x, x_centres, expected


def _check_force(results_by_name):
    """Force-driven channel: u_x = G y (H - y) / (2 mu) in every column, u_y = 0."""
    errors, velocity_x, _, expected = _channel_errors(results_by_name[_RUN], _FORCE_LENGTH, _FORCE_HEIGHT,
                                                      lambda y: _poiseuille(y, _FORCE_GRADIENT, _FORCE_HEIGHT))
    return {'measured': float(np.max(velocity_x)), 'expected': float(np.max(expected)), 'error': max(errors.values()),
            'tolerance': _FORCE_TOLERANCE,
            'details': {'errors': errors, 'force_density_N_per_m3': _FORCE_GRADIENT,
                        'mean_velocity_m_per_s': float(np.mean(velocity_x)),
                        'mean_velocity_closed_m_per_s': _FORCE_GRADIENT * _FORCE_HEIGHT ** 2 / (12.0 * _VISCOSITY),
                        'n_cells': list(velocity_x.shape)}}


def _check_pressure(results_by_name):
    """Pressure-driven channel: the parabola of G = dp / L in every column, u_y = 0, p = dp (1 - x / L)."""
    results = results_by_name[_RUN]
    errors, velocity_x, x_centres, _ = _channel_errors(results, _PRESSURE_LENGTH, _CHANNEL_HEIGHT,
                                                       lambda y: _poiseuille(y, _DRIVE_GRADIENT, _CHANNEL_HEIGHT))
    pressure, _, _ = _window_field(results, f'{_FLUID}.pressure', velocity_x.shape)
    expected_pressure = _PRESSURE_DROP * (1.0 - x_centres / _PRESSURE_LENGTH)
    errors['pressure'] = float(np.max(np.abs(pressure - expected_pressure[:, None]))) / _PRESSURE_DROP
    flow_rate = float(np.mean(np.sum(velocity_x, axis=1) * _CELL))
    flow_rate_closed = _DRIVE_GRADIENT * _CHANNEL_HEIGHT ** 3 / (12.0 * _VISCOSITY)
    return {'measured': flow_rate, 'expected': flow_rate_closed, 'error': max(errors.values()),
            'tolerance': _PRESSURE_TOLERANCE,
            'details': {'errors': errors, 'flow_rate_m2_per_s': flow_rate,
                        'flow_rate_closed_m2_per_s': flow_rate_closed, 'pressure_drop_Pa': _PRESSURE_DROP,
                        'n_cells': list(velocity_x.shape)}}


def _plug_profile(x_centres):
    """Exact steady advection-diffusion T(x) between T_in (x = 0) and T_out (x = L) at Peclet number _PECLET."""
    shape = np.expm1(_PECLET * x_centres / _PLUG_LENGTH) / math.expm1(_PECLET)
    return _T_IN + (_T_OUT - _T_IN) * shape


def _check_plug(results_by_name):
    """Plug flow: u_x = U, u_y = 0 and T(x) = T_in + dT (exp(Pe x / L) - 1) / (exp(Pe) - 1)."""
    results = results_by_name[_RUN]
    shape = (_cells(_PLUG_LENGTH), _cells(_PLUG_HEIGHT))
    temperature, x_centres, _ = _window_field(results, f'{_HEAT}.T', shape)
    velocity_x, _, _ = _window_field(results, f'{_FLUID}.u_x', shape)
    velocity_y, _, _ = _window_field(results, f'{_FLUID}.u_y', shape)
    expected = _plug_profile(x_centres)
    rise = _T_OUT - _T_IN
    errors = {'T': float(np.max(np.abs(temperature - expected[:, None]))) / rise,
              'u_x': float(np.max(np.abs(velocity_x - _PLUG_VELOCITY))) / _PLUG_VELOCITY,
              'u_y': float(np.max(np.abs(velocity_y))) / _PLUG_VELOCITY}
    middle = shape[0] // 2
    return {'measured': float(np.mean(temperature[middle])), 'expected': float(expected[middle]),
            'error': max(errors.values()), 'tolerance': _PLUG_TOLERANCE,
            'details': {'errors': errors, 'peclet': _PECLET, 'velocity_m_per_s': _PLUG_VELOCITY,
                        'T_without_flow_K': _T_IN + rise * float(x_centres[middle]) / _PLUG_LENGTH,
                        'n_cells': list(temperature.shape)}}


# ---- template ----
# heated_microchannel: a 300 x 100 um silica chip (uniform 2 um cells, 150 x 50) with a 20 um water channel along x
# (y = 40..60 um, 10 cells), parabolic inflow of mean U_in = 2 mm/s on x_lo, pressure 0 on x_hi; a 60 x 6 um heater
# 10 um above the channel (Q_heater 2e13 W/m^3 = 7.2 W/m), the substrate face y_lo held at 293.15 K and the inflow
# face x_lo too. Water nu = 1e-6: dt_lbm = 6.7e-7 s, lattice Mach ~2e-3; rho cp U H / k ~ 0.3 across the channel, so the
# convection loop converges in ~10-20 stationary iterations. Validation: no errors.
HEATED_MICROCHANNEL_LABEL = 'Heated microchannel'
HEATED_MICROCHANNEL_DESCRIPTION = (
    'Laminar flow + heat, stationary: water flows (parabolic inflow, mean U_in = 2 mm/s) through a 20 um channel in a '
    'silica chip past a 60 x 6 um heater (Q_heater, ~7 W/m) 10 um above it; fluid_heat_convection carries the heat '
    'downstream; derived T_max, T_outlet (mean water temperature at the outlet) and U_max.')
_CHIP = (300e-6, 100e-6)
_CHANNEL_Y = (40e-6, 60e-6)
_HEATER = (120e-6, 70e-6, 180e-6, 76e-6)


def heated_microchannel_spec():
    """Template 'heated_microchannel' (laminar_flow + heat + fluid_heat_convection, stationary)."""
    size_x, size_y = _CHIP
    channel_y0, channel_y1 = _CHANNEL_Y

    def rectangle(name, x0, y0, x1, y1):
        return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})

    return ModelSpec(
        title=HEATED_MICROCHANNEL_LABEL,
        parameters=[ParameterSpec('U_in', 2e-3, 'mean inflow velocity [m/s]'),
                    ParameterSpec('Q_heater', 2e13, 'heater power density [W/m^3]')],
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[rectangle('channel', 0.0, channel_y0, size_x, channel_y1),
                  rectangle('heater', *_HEATER),
                  rectangle('outlet_band', size_x - 10e-6, channel_y0, size_x, channel_y1)],
        materials=[MaterialSpec('chip', library='silica', selection=['all']),
                   MaterialSpec('water', library='water', selection=['channel'])],
        physics=[PhysicsSpec(_FLUID, FLUID_TYPE_NAME, features=[
                     FeatureSpec('inlet', 'inlet_velocity', ['x_lo'], {'velocity': 'U_in', 'profile': 'parabolic'}),
                     FeatureSpec('outlet', 'pressure', ['x_hi'], {'pressure': 0.0})]),
                 PhysicsSpec(_HEAT, 'heat', features=[
                     FeatureSpec('substrate', 'temperature', ['y_lo'], {'temperature': 293.15}),
                     FeatureSpec('inflow', 'temperature', ['x_lo'], {'temperature': 293.15}),
                     FeatureSpec('heater', 'heat_source', ['heater'], {'Q': 'Q_heater'})])],
        couplings=[CouplingSpec('convection', CONVECTION_TYPE_NAME, _FLUID, _HEAT)],
        mesh=MeshSpec(mode='uniform', dx=2e-6),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T_max', 'T', [], 'max'),
                        DerivedValueSpec('T_outlet', 'T', ['outlet_band'], 'mean'),
                        DerivedValueSpec('U_max', 'u_abs', [], 'max')])


# ---- cases ----
_CHANNEL_TEXT = (f'uniform {_CELL * 1e6:g} um cells (no PML), a liquid of viscosity {_VISCOSITY:g} Pa s and density '
                 f'{_DENSITY:g} kg/m^3 everywhere, steady tolerance {_STEADY_SETTINGS["tolerance"]:g}, stationary '
                 'study')


def startup_case(precision):
    """Transient Poiseuille start-up case at a stepper precision: 'half_storage' (registered) or 'single' (on the
    Automatic stepper; wrapped by selftests_backend)."""
    if precision == 'half_storage':
        name, label = ('fluid_poiseuille_startup_half_storage',
                       'Laminar flow: Poiseuille start-up with half-precision populations')
        stepper = ('on an explicit_kernels stepper node at precision half_storage (fp16 populations x 1024, float32 '
                   'arithmetic)')
        margin = '(single precision ~2e-3; a wrong storage scale or conversion is off by orders of magnitude)'
    elif precision == 'single':
        name, label = 'fluid_poiseuille_startup', 'Laminar flow: Poiseuille start-up'
        stepper = 'on the Automatic stepper (single precision)'
        margin = '(~2e-3 expected: lattice diffusion of the modes, the end time within a step)'
    else:
        raise ValueError(f"startup_case precision must be 'single' or 'half_storage', got {precision!r}")
    return SelfTestCase(
        name=name, label=label,
        description=(f'Transient laminar flow in a {_STARTUP_LENGTH * 1e6:g} x {_CHANNEL_HEIGHT * 1e6:g} um periodic '
                     f'channel ({_CELL * 1e6:g} um cells, viscosity {_VISCOSITY:g} Pa s, density {_DENSITY:g} '
                     f'kg/m^3, body force G = {_DRIVE_GRADIENT:g} N/m^3 along x from rest), time_dependent study '
                     f'of {_STARTUP_DURATION * 1e6:.4g} us (240 lattice steps, about one decay time of the slowest '
                     f'mode) {stepper}. Exact: u_x = G y (H - y) / (2 mu) - (4 G H^2 / (mu pi^3)) sum_odd n '
                     'sin(n pi y / H) exp(-n^2 t / tau_v) / n^3, u_y = 0. error = max(max |u_x - u_exact|, max |u_y|) '
                     f'/ peak over all cells; tolerance {_STARTUP_TOLERANCE:g} {margin}. The check raises unless '
                     f"summary 'solvers' records the stepper precision {precision}."),
        specs=lambda: _startup_specs(precision),
        check=lambda results_by_name: _check_startup(results_by_name, precision))


# FLUID_CASES: every analytic case (selftests_backend wraps them in its cpu-vs-gpu parity cases); SELFTEST_CASES
# registers only those no example checks: the force channel is examples/fluid/couette_poiseuille + falling_film
# (periodic faces, walls, body force, steady), the pressure channel poiseuille_channel_pressure, the plug flow
# slug_flow_graetz (slip faces, velocity inlet, initial_velocity start, fluid_heat_convection).
_COVERED_BY_EXAMPLES = frozenset({'fluid_poiseuille_force', 'fluid_poiseuille_pressure',
                                  'fluid_heat_convection_plug'})
FLUID_CASES = (
    SelfTestCase(
        name='fluid_poiseuille_force', label='Laminar flow: force-driven Poiseuille channel',
        description=(f'Laminar flow in a {_FORCE_LENGTH * 1e6:g} x {_FORCE_HEIGHT * 1e6:g} um channel of '
                     f'{_CHANNEL_TEXT}: periodic x faces, no-slip walls on y_lo / y_hi, body_force G = '
                     f'{_FORCE_GRADIENT:g} N/m^3 along x. Exact: u_x = G y (H - y) / (2 mu) (peak '
                     f'{_PEAK_VELOCITY:g} m/s), u_y = 0; the TRT collision with the magic parameter 3/16 puts the '
                     'halfway bounce-back walls exactly on the faces, so the lattice solution equals the parabola at '
                     'the cell centres at any height (expected error ~1e-6 from float32 and the steady tolerance; a '
                     'wall half a cell off gives ~10 %). error = max(max |u_x - u_exact|, max |u_y|) / peak over all '
                     f'cells; measured = max u_x; tolerance {_FORCE_TOLERANCE:g}. This pins the viscosity (relaxation '
                     'time), the Guo body force, periodic faces, the wall location and the lattice -> SI velocity '
                     'scale. Not yet confirmed by a run.'),
        specs=_force_specs, check=_check_force),
    startup_case('half_storage'),
    SelfTestCase(
        name='fluid_poiseuille_pressure', label='Laminar flow: pressure-driven Poiseuille channel',
        description=(f'Laminar flow in a {_PRESSURE_LENGTH * 1e6:g} x {_CHANNEL_HEIGHT * 1e6:g} um channel of '
                     f'{_CHANNEL_TEXT}: pressure {_PRESSURE_DROP:g} Pa on x_lo and 0 on x_hi (anti-bounce-back), '
                     'no-slip walls on y_lo / y_hi. Exact: u_x = (dp / L) y (H - y) / (2 mu) in every column, u_y = '
                     '0, p = dp (1 - x / L) at the cell centres, flow rate per m of depth Q = dp H^3 / (12 mu L). '
                     'error = max(max |u_x - u_exact| / peak, max |u_y| / peak, max |p - p_exact| / dp); measured = '
                     f'Q from the mean column sum of u_x dy; tolerance {_PRESSURE_TOLERANCE:g} (pressure faces and '
                     'the weak compressibility of the lattice, density deviation ~3e-4). This pins the pressure faces, '
                     'the lattice -> SI pressure scale and the He-Luo incompressible equilibrium. Not yet confirmed by '
                     'a run.'),
        specs=_pressure_specs, check=_check_pressure),
    SelfTestCase(
        name='fluid_heat_convection_plug', label='Laminar flow + heat: plug-flow advection-diffusion',
        description=(f'Plug flow in a {_PLUG_LENGTH * 1e6:g} x {_PLUG_HEIGHT * 1e6:g} um channel of uniform '
                     f'{_CELL * 1e6:g} um cells: free-slip y faces, inlet_velocity U = {_PLUG_VELOCITY:g} m/s on '
                     'x_lo, pressure 0 on x_hi, so u_x = U everywhere; the flow starts there by initial_velocity '
                     '(a fixed point of correct face rules: the steady march stops after a few checks; wrong rules '
                     'move it and the march grades their steady state); heat (float64 CG 1e-12) with T_in = '
                     f'{_T_IN:g} K on x_lo and T_out = {_T_OUT:g} K on x_hi, insulated y faces; k = '
                     f'{_PLUG_CONDUCTIVITY:g} W/(m K), rho cp = {_DENSITY * _PLUG_HEAT_CAPACITY:g} J/(m^3 K); '
                     'fluid_heat_convection feeds -rho cp u . grad T into heat, stationary study (tolerance '
                     f'{_PLUG_LOOP_TOLERANCE:g}, '
                     f'the Picard loop converges with gain ~Pe / pi). Pe = rho cp U L / k = {_PECLET:g}; exact: T = '
                     'T_in + (T_out - T_in) (exp(Pe x / L) - 1) / (exp(Pe) - 1). error = max(max |T - T_exact| / '
                     '(T_out - T_in), max |u_x - U| / U, max |u_y| / U); measured = the mean T of the middle column; '
                     f'tolerance {_PLUG_TOLERANCE:g} (second-order central advection, the one-sided inflow face cell). '
                     'This pins the advective source, its sign and the Dirichlet inflow face, the slip and velocity '
                     'faces and the E13 contribution to volumetric_power (not the build-up from rest of a velocity '
                     'inlet). Not yet confirmed by a run.'),
        specs=_plug_specs, check=_check_plug),
)
SELFTEST_CASES = tuple(case for case in FLUID_CASES if case.name not in _COVERED_BY_EXAMPLES)
