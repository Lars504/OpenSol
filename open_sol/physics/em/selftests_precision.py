"""Self-test of the double-precision 2D FDTD stepper (explicit_kernels precision 'double', solver_plan S5 / stage D): the
phase drift of a lossless pec cavity mode over a 5 ps ring-down against the exact discrete Yee frequency."""
import math

import numpy as np

from ...core.constants import DTYPE, EPS0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_centre, solver_role
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverSpec, StudySpec)

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_ez'
_PROBE_FEATURE = 'probe'
_STEPPER_NODE = 'fp64_fields'

# em_fdtd_double_cavity_phase: lossless eps_r 2.25 filling a 1.0 x 0.6 um box of pec walls on a 40 nm mesh (E_z pinned
# to 0 in the outermost cells), TM11 (vacuum wavelength ~1.453 um, ~52 steps per period) excited by a narrow pulse at
# the box centre. The step count sets the run time (the grid is tiny); theta below is exact at any resolution.
_CELL = 40e-9                      # m
# The kernels' cell width: the float32 grid.dx (yee_grid.grid_cell_widths), 6.1e-9 below _CELL, which shifts theta by
# ~6e-9 of itself (~3e-5 rad over the window separation): the reference must use it.
_KERNEL_CELL = float(DTYPE(_CELL))
_CELLS = (25, 15)                  # interior (1..23) x (1..13), odd counts: one centre cell
_EPS_R = 2.25
_SOURCE_CELL = (12, 7)             # the centre cell, on every even mode's nodal line (TM21, TM12, ...): not excited
_PROBE_CELL = (12, 7)
_PULSE = {'waveform': 'gaussian', 'wavelength': 1.453e-6, 'bandwidth': 0.1, 'current': 1e-3}   # ends after ~0.14 ps
_DURATION = 5e-12                  # s: ~5.4e4 steps, ~1030 periods
_FIRST_WINDOW_STEP = 2000          # first fit window starts after the pulse (~1500 steps)
_WINDOW_STEPS = 10000              # samples per phase fit (~190 periods)
_TOLERANCE = 1e-5                  # rad of TM11 phase drift between the windows (double)


def _cavity_spec():
    """The pec cavity with a point-source pulse and an E_z field probe on an explicit_kernels stepper node at precision
    double."""
    size_x, size_y = (cells * _CELL for cells in _CELLS)
    source_x, source_y = (cell_centre(index, _CELL) for index in _SOURCE_CELL)
    probe_x, probe_y = (cell_centre(index, _CELL) for index in _PROBE_CELL)
    spec = ModelSpec(
        title='Self-test: lossless pec cavity phase drift (double precision stepper)',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('source_point', 'point', {'x': source_x, 'y': source_y}),
                  GeometrySpec('probe_point', 'point', {'x': probe_x, 'y': probe_y})],
        materials=[MaterialSpec('dielectric', properties={'eps_r': _EPS_R}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE,
                             settings={'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec'},
                             features=[FeatureSpec('source', 'point_source', ['source_point'], dict(_PULSE)),
                                       FeatureSpec(_PROBE_FEATURE, 'field_probe', ['probe_point'],
                                                   {'component': 'ez'})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _DURATION}))
    spec.solvers = [SolverSpec(_STEPPER_NODE, 'explicit_kernels', {'precision': 'double'})]
    spec.physics[0].solvers = {'stepper': _STEPPER_NODE}
    return spec


def _cavity_specs():
    """'double': the one run (explicit_kernels precision double)."""
    return {'double': _cavity_spec()}


def _tm11_step_phase(dt):
    """TM11 phase advance per step [rad] of the lossless Yee recursion: sin(theta / 2) = dt kappa / (2 sqrt(eps
    mu0)), kappa^2 = (2 / dx)^2 (sin^2(pi / (2 (nx - 1))) + sin^2(pi / (2 (ny - 1)))), dx the kernels' float32 width."""
    nx, ny = _CELLS
    kappa = (2.0 / _KERNEL_CELL) * math.sqrt(math.sin(math.pi / (2 * (nx - 1))) ** 2
                                      + math.sin(math.pi / (2 * (ny - 1))) ** 2)
    return 2.0 * math.asin(dt * kappa / (2.0 * math.sqrt(_EPS_R * EPS0 * MU0)))


def _window_phase(steps, values, theta):
    """Phase phi of the least-squares fit values ~ a cos(theta n + phi) over the window's step indices n."""
    design = np.stack((np.cos(theta * steps), np.sin(theta * steps)), axis=1)
    (cos_part, sin_part), *_ = np.linalg.lstsq(design, values, rcond=None)
    return math.atan2(-sin_part, cos_part)


def _phase_drift(results):
    """(drift [rad] of the TM11 phase between the first and the last fit window, details) of one run's probe trace."""
    trace = results.result(_PHYSICS_NAME, _PROBE_FEATURE)
    times = np.asarray(trace['t'], dtype=np.float64)
    values = np.asarray(trace['values'], dtype=np.float64)
    if times.size < _FIRST_WINDOW_STEP + 2 * _WINDOW_STEPS:
        raise ValueError(f'the probe recorded {times.size} samples, fewer than the two fit windows need')
    dt = float(times[-1] - times[0]) / (times.size - 1)
    steps = np.rint(times / dt)   # integer step count of each sample (the host clock accumulates rounding)
    theta = _tm11_step_phase(dt)
    first = slice(_FIRST_WINDOW_STEP, _FIRST_WINDOW_STEP + _WINDOW_STEPS)
    last = slice(times.size - _WINDOW_STEPS, times.size)
    early = _window_phase(steps[first], values[first], theta)
    late = _window_phase(steps[last], values[last], theta)
    drift = float(np.angle(np.exp(1j * (late - early))))
    window_separation = float(np.mean(steps[last]) - np.mean(steps[first]))
    return drift, {'drift_rad': drift, 'relative_frequency_error': drift / (theta * window_separation),
                   'dt_s': dt, 'steps_per_period': 2.0 * math.pi / theta, 'window_separation_steps': window_separation,
                   'peak_field': float(np.max(np.abs(values[last])))}


def _check_cavity_phase(results_by_name):
    """TM11 phase drift of the double run (error, rad) against the tolerance. Raises unless summary 'solvers' records
    the double stepper precision."""
    precision = solver_role(results_by_name['double'], _PHYSICS_NAME, 'stepper').get('precision')
    if precision != 'double':
        raise ValueError(f"expected the stepper to run double, summary 'solvers' records {precision!r}")
    drift, details = _phase_drift(results_by_name['double'])
    error = abs(drift)
    return {'measured': drift, 'expected': 0.0, 'error': error if math.isfinite(error) else math.inf,
            'tolerance': _TOLERANCE, 'details': details}


PRECISION_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdtd_double_cavity_phase',
        label='EM FDTD: double-precision stepper, cavity mode phase drift',
        description='em_fdtd_ez box of pec walls (25 x 15 cells of 40 nm, E_z pinned to 0 in the outermost cells) '
                    'filled with lossless eps_r 2.25; a narrow gaussian point-source pulse (1.453 um, bandwidth 0.1) '
                    'in the centre cell rings TM11 for 5 ps (~5.4e4 steps, ~1030 periods; the centre is on the nodal '
                    'lines of TM21 / TM12, so they stay unexcited) on an explicit_kernels stepper node at precision '
                    'double. The E_z probe trace in the centre cell is fitted by a cos(theta n + phi) on two '
                    '10000-step windows (after the pulse and at the end) at '
                    'the exact discrete TM11 step phase of the lossless Yee recursion, sin(theta / 2) = dt kappa / (2 '
                    'sqrt(eps mu0)), kappa^2 = (2 / dx)^2 (sin^2(pi / 48) + sin^2(pi / 28)), dx the float32 grid.dx '
                    'the kernels use (6.1e-9 below 40 nm, ~3e-5 rad of drift against the nominal width); the drift = '
                    'phi_late - '
                    'phi_early measures the frequency error of the run. error = |drift|, tolerance 1e-5 rad (expected '
                    '~1e-7: float32 probe samples and weak higher-mode leakage); a float32 stepper drifts by its '
                    'coefficient rounding (~1e-8 to 1e-7 of the frequency, >= ~5e-5 rad over the ~4.2e4 steps between '
                    'the windows), so a silent single-precision fallback fails. Raises unless summary '
                    "'solvers' records the stepper precision double. 5 ps run not yet confirmed by a run.",
        specs=_cavity_specs,
        check=_check_cavity_phase),
)
