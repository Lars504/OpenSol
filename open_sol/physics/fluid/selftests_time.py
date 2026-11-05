"""Transient laminar_flow self-tests of the initial_velocity feature (free Taylor-Green decay in a periodic box) and of
feature values of time t (an oscillating body force in a periodic box; Stokes' second problem: a wall oscillating
along itself under a slip face)."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_centre
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)

FLUID_TYPE_NAME = 'laminar_flow'      # = definition.FLUID_TYPE_NAME (definition imports this module)
_FLUID = 'fluid'
_CELL = 1e-6                          # m, uniform mesh
_VISCOSITY = 1e-3                     # Pa s
_DENSITY = 1000.0                     # kg/m^3
_KINEMATIC = _VISCOSITY / _DENSITY    # 1e-6 m^2/s
_LATTICE_STEP = 0.5 * _CELL ** 2 / (3.0 * _KINEMATIC)   # s, dt_lbm at relaxation_time 1 (1/6 us)

# fluid_taylor_green_decay: periodic TG_CELLS^2 box, u = U (sin kx cos ky, -cos kx sin ky), p = -rho U^2 (cos 2kx +
# cos 2ky) / 4, decaying as exp(-2 nu k^2 t) (an exact Navier-Stokes solution). The probe u_x gives the rate and the
# start amplitude (log-linear fit after TG_SKIP_STEPS: the start from pure equilibrium populations relaxes within a few
# steps, shifting the fitted amplitude by ~ the decay of one step, 3e-3).
TG_CELLS = 64
TG_VELOCITY = 1e-3                    # m/s (lattice Mach ~3e-4, Re = U L / nu = 0.06)
TG_STEPS = 620                        # ~2 decay times (2 nu k^2 dt_lbm = 3.2e-3 per step)
TG_SKIP_STEPS = 40
TG_PROBE_CELL = (15, 0)               # u_x there ~ 0.998 U
TG_TOLERANCE = 1e-2                   # estimate: lattice O((k dx)^2) ~ 1e-2 x a small coefficient, the start transient

# fluid_time_dependent_drives: run 'force' = a periodic FORCE_CELLS^2 box under f_x = F sin(omega t): uniform u =
# F (1 - cos omega t) / (rho omega) (no spatial error; the published velocity lags the record time by about dt / 2);
# run 'wall' = Stokes' second problem: x periodic, wall y_lo moving along x at U cos(omega t), slip y_hi at H; after
# WALL_PERIODS the periodic state u = Re(U e^(i omega t) cosh(kappa (H - y)) / cosh(kappa H)), kappa = (1 + i) / delta,
# delta = sqrt(2 nu / omega), compared over the last period at the probe.
FORCE_CELLS = 4
FORCE_PERIOD_STEPS = 500              # error = the half-step lag omega dt / 2 = pi / 500 ~ 6.3e-3 (no spatial error)
FORCE_OMEGA = 2.0 * math.pi / (FORCE_PERIOD_STEPS * _LATTICE_STEP)
FORCE_PEAK_VELOCITY = 1e-3            # m/s = 2 F / (rho omega)
FORCE_DENSITY = 0.5 * FORCE_PEAK_VELOCITY * _DENSITY * FORCE_OMEGA   # N/m^3
WALL_DELTA_CELLS = 6                  # Stokes layer depth delta in cells (period pi delta^2 / nu ~ 679 lattice steps)
WALL_OMEGA = 2.0 * _KINEMATIC / (WALL_DELTA_CELLS * _CELL) ** 2   # rad/s
WALL_HEIGHT_CELLS = 12                # H = 2 delta
WALL_VELOCITY = 1e-3                  # m/s
WALL_PERIODS = 5                      # slowest transient (quarter wave, 4 H^2 / (pi^2 nu) ~ 0.52 periods) is down to
                                      # ~4e-4 when the compared last period starts
WALL_PROBE_CELL = 3                   # y cell of the probe (x cell 1; y / delta ~ 0.58)
DRIVE_TOLERANCE = 2e-2                # estimate: the half-step lag (omega dt / 2 ~ 6e-3 force, ~5e-3 x the probe
                                      # amplitude wall), lattice O((dx / delta)^2) at 6 cells per delta (~3e-3)


def _liquid():
    return MaterialSpec('liquid', properties={'viscosity': _VISCOSITY, 'rho': _DENSITY}, selection=['all'])


def _transient_model(title, size, features, probe_xy, duration):
    """Transient laminar_flow on the uniform _CELL mesh with a u_x flow_probe at probe_xy."""
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=size[0], size_y=size[1]),
        geometry=[GeometrySpec('probe_point', 'point', {'x': probe_xy[0], 'y': probe_xy[1]})],
        materials=[_liquid()],
        physics=[PhysicsSpec(_FLUID, FLUID_TYPE_NAME, settings={'time_mode': 'transient'}, features=[
            *features, FeatureSpec('probe', 'flow_probe', ['probe_point'], {'component': 'u_x'})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': duration}))


def _taylor_green_specs():
    """'run': fully periodic box started from the Taylor-Green vortex by initial_velocity."""
    size = TG_CELLS * _CELL
    k = f'(2 * pi / {size!r})'
    velocity, pressure_scale = repr(TG_VELOCITY), repr(0.25 * _DENSITY * TG_VELOCITY ** 2)
    start = {'u_x': f'{velocity} * sin({k} * x) * cos({k} * y)', 'u_y': f'-{velocity} * cos({k} * x) * sin({k} * y)',
             'pressure': f'-{pressure_scale} * (cos(2 * {k} * x) + cos(2 * {k} * y))'}
    probe = tuple(cell_centre(index, _CELL) for index in TG_PROBE_CELL)
    return {'run': _transient_model(
        'Self-test: Taylor-Green vortex decay (laminar flow)', (size, size),
        [FeatureSpec('periodic_x', 'periodic', ['x_lo', 'x_hi']), FeatureSpec('periodic_y', 'periodic', ['y_lo', 'y_hi']),
         FeatureSpec('vortex', 'initial_velocity', ['all'], start)],
        probe, TG_STEPS * _LATTICE_STEP)}


def _probe_series(results):
    """(t, u_x) float64 of the flow probe."""
    probe = results.result(_FLUID, 'probe')
    return np.asarray(probe['t'], dtype=np.float64), np.asarray(probe['values'], dtype=np.float64)


def _check_taylor_green(results_by_name):
    """Decay rate of the probe u_x (log-linear fit) against 2 nu k^2 and the fit's t = 0 amplitude against the start
    value at the probe."""
    times, values = _probe_series(results_by_name['run'])
    k = 2.0 * math.pi / (TG_CELLS * _CELL)
    rate = 2.0 * _KINEMATIC * k * k
    x_probe, y_probe = (cell_centre(index, _CELL) for index in TG_PROBE_CELL)
    shape = TG_VELOCITY * math.sin(k * x_probe) * math.cos(k * y_probe)
    fitted = times[TG_SKIP_STEPS:]
    if fitted.size < 2 or not np.all(values[TG_SKIP_STEPS:] > 0.0):
        return {'measured': math.nan, 'expected': rate, 'error': math.inf, 'tolerance': TG_TOLERANCE,
                'details': {'n_samples': int(times.size), 'last_u_x': float(values[-1]) if values.size else None}}
    slope, intercept = np.polyfit(fitted, np.log(values[TG_SKIP_STEPS:]), 1)
    amplitude = math.exp(intercept)
    errors = {'rate': abs(-slope - rate) / rate, 'amplitude': abs(amplitude - shape) / abs(shape)}
    return {'measured': float(-slope), 'expected': rate, 'error': max(errors.values()), 'tolerance': TG_TOLERANCE,
            'details': {'errors': errors, 'fitted_amplitude_m_per_s': amplitude, 'expected_amplitude_m_per_s': shape,
                        'first_sample_m_per_s': float(values[0]), 'k_dx': k * _CELL, 'n_samples': int(times.size)}}


def _drive_specs():
    """'force': periodic box under f_x = F sin(omega t); 'wall': y_lo wall at U cos(omega t), slip y_hi."""
    box = FORCE_CELLS * _CELL
    force = _transient_model(
        'Self-test: oscillating body force in a periodic box (laminar flow)', (box, box),
        [FeatureSpec('periodic_x', 'periodic', ['x_lo', 'x_hi']), FeatureSpec('periodic_y', 'periodic', ['y_lo', 'y_hi']),
         FeatureSpec('drive', 'body_force', ['all'],
                     {'f_x': f'{FORCE_DENSITY!r} * sin({FORCE_OMEGA!r} * t)', 'f_y': 0.0})],
        (cell_centre(1, _CELL), cell_centre(1, _CELL)), FORCE_PERIOD_STEPS * _LATTICE_STEP)
    wall = _transient_model(
        "Self-test: Stokes' second problem, oscillating wall (laminar flow)",
        (FORCE_CELLS * _CELL, WALL_HEIGHT_CELLS * _CELL),
        [FeatureSpec('periodic_x', 'periodic', ['x_lo', 'x_hi']),
         FeatureSpec('shaker', 'wall', ['y_lo'],
                     {'tangential_velocity': f'{WALL_VELOCITY!r} * cos({WALL_OMEGA!r} * t)'}),
         FeatureSpec('free', 'slip_wall', ['y_hi'])],
        (cell_centre(1, _CELL), cell_centre(WALL_PROBE_CELL, _CELL)), WALL_PERIODS * 2.0 * math.pi / WALL_OMEGA)
    return {'force': force, 'wall': wall}


def _stokes_layer(y, times):
    """Periodic state of the oscillating wall under a slip face at H (Stokes' second problem, finite depth)."""
    delta = math.sqrt(2.0 * _KINEMATIC / WALL_OMEGA)
    kappa = (1.0 + 1.0j) / delta
    height = WALL_HEIGHT_CELLS * _CELL
    profile = np.cosh(kappa * (height - y)) / np.cosh(kappa * height)
    return np.real(WALL_VELOCITY * np.exp(1j * WALL_OMEGA * times) * profile)


def _check_drives(results_by_name):
    """Run force: u_x over the period against F (1 - cos omega t) / (rho omega); run wall: the last period against the
    Stokes layer; both relative to their peak velocity."""
    times, values = _probe_series(results_by_name['force'])
    expected = FORCE_DENSITY * (1.0 - np.cos(FORCE_OMEGA * times)) / (_DENSITY * FORCE_OMEGA)
    errors = {'force': float(np.max(np.abs(values - expected))) / FORCE_PEAK_VELOCITY if times.size else math.inf}
    wall_times, wall_values = _probe_series(results_by_name['wall'])
    period = 2.0 * math.pi / WALL_OMEGA
    last = wall_times >= wall_times[-1] - period if wall_times.size else wall_times.astype(bool)
    wall_expected = _stokes_layer(cell_centre(WALL_PROBE_CELL, _CELL), wall_times[last])
    errors['wall'] = (float(np.max(np.abs(wall_values[last] - wall_expected))) / WALL_VELOCITY if np.any(last)
                      else math.inf)
    return {'measured': float(np.max(values)) if values.size else math.nan, 'expected': FORCE_PEAK_VELOCITY,
            'error': max(errors.values()), 'tolerance': DRIVE_TOLERANCE,
            'details': {'errors': errors, 'force_N_per_m3': FORCE_DENSITY, 'force_omega_rad_per_s': FORCE_OMEGA,
                        'wall_omega_rad_per_s': WALL_OMEGA, 'wall_probe_amplitude_m_per_s':
                            float(np.max(np.abs(_stokes_layer(cell_centre(WALL_PROBE_CELL, _CELL),
                                                              np.linspace(0.0, period, 64))))),
                        'n_samples': {'force': int(times.size), 'wall': int(wall_times.size)}}}


TIME_SELFTEST_CASES = (
    SelfTestCase(
        name='fluid_taylor_green_decay', label='Laminar flow: Taylor-Green vortex decay from initial_velocity',
        description=(f'Transient laminar_flow in a fully periodic {TG_CELLS} x {TG_CELLS} box of {_CELL * 1e6:g} um '
                     f'cells (nu = {_KINEMATIC:g} m^2/s, relaxation_time 1) started by initial_velocity from the '
                     f'Taylor-Green vortex u = U (sin kx cos ky, -cos kx sin ky), U = {TG_VELOCITY:g} m/s, with its '
                     'pressure -rho U^2 (cos 2kx + cos 2ky) / 4: an exact Navier-Stokes solution decaying as exp(-2 nu '
                     f'k^2 t). A u_x flow_probe over {TG_STEPS} lattice steps: measured = the log-linear decay rate '
                     f'after {TG_SKIP_STEPS} steps vs 2 nu k^2, plus the first sample vs the start amplitude; '
                     f'tolerance {TG_TOLERANCE:g} (estimate: lattice dispersion at k dx = '
                     f'{2.0 * math.pi / TG_CELLS:.2g}; a missing start gives 0, a start on the wrong positions another '
                     'rate).'),
        specs=_taylor_green_specs, check=_check_taylor_green),
    SelfTestCase(
        name='fluid_time_dependent_drives', label='Laminar flow: body force and wall velocity of time t',
        description=('Transient laminar_flow with feature values of t, re-set at the start of every lattice step. Run '
                     f'force: fully periodic {FORCE_CELLS} x {FORCE_CELLS} box, body_force f_x = F sin(omega t) '
                     f'(period {FORCE_PERIOD_STEPS} steps): uniform u_x = F (1 - cos omega t) / (rho omega), peak '
                     f'{FORCE_PEAK_VELOCITY:g} m/s. Run wall: Stokes second problem, x periodic, wall y_lo moving at U '
                     f'cos(omega t) (U = {WALL_VELOCITY:g} m/s, delta = {WALL_DELTA_CELLS} cells), slip y_hi at H = '
                     f'{WALL_HEIGHT_CELLS} cells; after {WALL_PERIODS} periods the probe at y cell {WALL_PROBE_CELL} vs '
                     'Re(U e^(i omega t) cosh(kappa (H - y)) / cosh(kappa H)) over the last period. error = max |u - '
                     f'u_exact| / peak over both runs; tolerance {DRIVE_TOLERANCE:g} (estimate: the half-step lag of '
                     f'the published velocity, omega dt / 2 = pi / {FORCE_PERIOD_STEPS} in run force, and the lattice '
                     f'error at {WALL_DELTA_CELLS} cells per delta; constant values would miss by O(1)). Not yet '
                     'confirmed by a run.'),
        specs=_drive_specs, check=_check_drives),
)
