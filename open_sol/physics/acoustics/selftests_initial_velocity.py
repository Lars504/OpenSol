"""Self-test elastic_initial_velocity_mode: the initial_velocity feature releases the lowest standing P mode of a
rigid-walled fluid strip, compared with the exact leap-frog recursion of that discrete mode."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)

ELASTIC_TYPE_NAME = 'elastic_waves'   # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'acoustics'
# Fluid strip MODE_CELLS long between rigid (slip) x walls, periodic in y (4 cells): v_x = V sin(pi x / L) on the x faces
# and s_xx = S cos(pi x / L) at the centres are an exact discrete mode of the staggered scheme; with g = (2 / h)
# sin(pi h / (2 L)) the leap-frog runs A <- A - (dt / rho) g B (velocity), B <- B + dt c11 g A (stress) from A = V,
# B = 0 (the initial velocity is the state the first step starts from).
MODE_FLUID = {'rho': 1000.0, 'sound_speed': 1500.0}
MODE_CELL = 1e-4                      # m
MODE_CELLS = 40                       # L = 4 mm: f = c / (2 L) = 187.5 kHz, ~126 steps per period at courant 0.9
MODE_VELOCITY = 1e-3                  # m/s
MODE_DURATION = 1.1e-5                # s: about two periods
MODE_PROBE_CELL = 9                   # vx on the +x face of this cell, x = 10 h (sin = 0.71)
MODE_TOLERANCE = 1e-4                 # float32 fields over ~260 steps (a centre-sampled start mixes modes: ~4e-2)


def _mode_specs():
    """'run': rigid x walls, periodic y strip, initial_velocity v_x = V sin(pi x / L), a vx probe."""
    length, width = MODE_CELLS * MODE_CELL, 4 * MODE_CELL
    settings = {'bc_x_lo': 'rigid', 'bc_x_hi': 'rigid', 'bc_y_lo': 'periodic', 'bc_y_hi': 'periodic'}
    probe = {'x': (MODE_PROBE_CELL + 0.5) * MODE_CELL, 'y': 1.5 * MODE_CELL}
    return {'run': ModelSpec(
        title='Self-test: standing P mode released by an initial velocity (elastic waves)',
        domain=DomainSpec(size_x=length, size_y=width),
        geometry=[GeometrySpec('probe_point', 'point', probe)],
        materials=[MaterialSpec('fluid', properties=dict(MODE_FLUID), selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, ELASTIC_TYPE_NAME, settings=settings, features=[
            FeatureSpec('release', 'initial_velocity', ['all'],
                        {'v_x': f'{MODE_VELOCITY!r} * sin(pi * x / {length!r})', 'v_y': 0.0}),
            FeatureSpec('probe', 'field_probe', ['probe_point'], {'component': 'vx'})])],
        mesh=MeshSpec(mode='uniform', dx=MODE_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': MODE_DURATION}))}


def _mode_velocities(n_steps, time_step):
    """Probe-face velocity after each of n_steps leap-frog steps of the discrete mode (float64)."""
    length = MODE_CELLS * MODE_CELL
    rho = MODE_FLUID['rho']
    stiffness = rho * MODE_FLUID['sound_speed'] ** 2
    gradient = 2.0 / MODE_CELL * math.sin(math.pi * MODE_CELL / (2.0 * length))
    shape = math.sin(math.pi * (MODE_PROBE_CELL + 1) * MODE_CELL / length)
    velocity, stress, samples = MODE_VELOCITY, 0.0, []
    for _ in range(n_steps):
        velocity -= time_step / rho * gradient * stress
        samples.append(velocity * shape)
        stress += time_step * stiffness * gradient * velocity
    return np.array(samples)


def _check_mode(results_by_name):
    """Probe vx (times t_n - dt / 2) against the recursion with dt from the probe times, relative to V."""
    probe = results_by_name['run'].result(PHYSICS_NAME, 'probe')
    times = np.asarray(probe['t'], dtype=np.float64)
    values = np.asarray(probe['values'], dtype=np.float64)
    if times.size < 3:
        return {'measured': math.nan, 'expected': MODE_VELOCITY, 'error': math.inf, 'tolerance': MODE_TOLERANCE,
                'details': {'n_samples': int(times.size)}}
    time_step = float(times[1] - times[0])
    expected = _mode_velocities(times.size, time_step)
    errors = {'velocity': float(np.max(np.abs(values - expected))) / MODE_VELOCITY,
              'first_time': abs(times[0] - 0.5 * time_step) / time_step}
    return {'measured': float(values[-1]), 'expected': float(expected[-1]), 'error': max(errors.values()),
            'tolerance': MODE_TOLERANCE,
            'details': {'errors': errors, 'dt_s': time_step, 'n_steps': int(times.size),
                        'period_steps': 2.0 * MODE_CELLS * MODE_CELL / MODE_FLUID['sound_speed'] / time_step}}


INITIAL_VELOCITY_CASES = (
    SelfTestCase(
        name='elastic_initial_velocity_mode', label='Elastic waves: standing mode released by initial_velocity',
        description=(f'A fluid strip (rho {MODE_FLUID["rho"]:g}, c {MODE_FLUID["sound_speed"]:g} m/s) of {MODE_CELLS} '
                     f'cells of {MODE_CELL * 1e3:g} mm between rigid x walls, periodic in y, released at t = 0 by '
                     f'initial_velocity v_x = {MODE_VELOCITY:g} sin(pi x / L) (evaluated at the +x face centres). That '
                     'is the lowest discrete P mode of the staggered grid, so the vx probe follows the scalar '
                     'leap-frog recursion A <- A - (dt / rho) g B, B <- B + dt c11 g A, g = (2 / h) sin(pi h / 2 L), '
                     'from A = V, B = 0 exactly (dt read from the probe times, the first at dt / 2). error = max |vx - '
                     f'A sin(pi x_f / L)| / V over ~2 periods; tolerance {MODE_TOLERANCE:g} (float32 fields; centre '
                     'values would excite other modes at ~4e-2, a missing start reads 1).'),
        specs=_mode_specs, check=_check_mode),
)
