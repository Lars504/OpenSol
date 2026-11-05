"""The 'thermal_noise' StudyType: background physics to a stationary fixed point, the mode solve of every em_modes
physics, then the thermorefractive noise spectra of every heat physics (thermorefractive_noise features) on a
log- or linearly spaced frequency list."""
import math
import time

from .schema import Param, StudyType
from .solver_hooks import ModeSolver, ThermalNoiseSolver
from .study import (_MODE_STAGE, Cancelled, LogEvent, _background_fixed_point_params, _background_stage,
                    _count_progress, _frequency_list, _frequency_range_issues, _physics_text,
                    _refuse_stepping_background, _scipy_issues, _solve_mode_snapshots)

STUDY_NAME = 'thermal_noise'
_NOISE_STAGE = 'Noise spectra'


def _run_thermal_noise(session, values, emit, control, service):
    """Background stationary solve, background -> mode / noise couplings once, every mode solve with one snapshot per
    mode number, then the noise spectra of every heat physics; one final snapshot."""
    runner = session.runner
    frequencies = _frequency_list(values, ascending=True)   # the band integral needs them ascending
    noise = [name for name, solver in runner.physics.items() if isinstance(solver, ThermalNoiseSolver)]
    modes = [name for name, solver in runner.physics.items() if isinstance(solver, ModeSolver)]   # solved first
    background = [name for name in runner.physics if name not in noise and name not in modes]
    if not noise:
        raise ValueError(f"no physics has a thermal noise solve; physics: {', '.join(runner.physics)}; add a heat "
                         'physics with a thermorefractive_noise feature')
    _refuse_stepping_background(runner.physics, background, 'a thermal noise study')
    stage = _background_stage(
        session, background, noise + modes, values, emit, control, service,
        ('thermal noise', 'a thermal noise study assumes a uniform temperature T0 and feeds no mode or noise solution '
                          'back'),
        f'Thermal noise study: {len(frequencies)} frequencies, {min(frequencies):.4g}-{max(frequencies):.4g} Hz; '
        f"noise physics {', '.join(noise)}; mode physics {', '.join(modes) or 'none'}; background physics "
        f"{', '.join(background) or 'none'}",
        [_MODE_STAGE] * bool(modes) + [_NOISE_STAGE])
    status, n_solved = stage.status, 0
    if status == 'completed' and modes:   # one snapshot per mode number ('mode k: n_eff = ...'), then the noise
        status, _n_modes = _solve_mode_snapshots(session, modes, stage.recorder, stage.live,
                                                 stage.stages.stage_emit(_MODE_STAGE), control, service,
                                                 final_last=False)
    if status == 'completed':
        try:
            n_solved = _solve_noise(session, noise, frequencies, stage.stages.stage_emit(_NOISE_STAGE), control,
                                    service)
        except Cancelled:
            status = 'cancelled'
    stage.recorder.record(final=True, label='thermal noise' if status == 'completed' else None)
    stage.live.update(final=True)
    return {'status': status, 'ticks': n_solved, 'tick_unit': 'factorisation', 't_end': runner.t,
            'fixed_point': stage.fixed_point,
            'noise_frequencies': frequencies if status == 'completed' else []}


def _solve_noise(session, noise, frequencies, emit, control, service):
    """solve_thermal_noise of every noise physics with per-factorisation progress (each monitor an equal share) and
    cancel checks; logs each feature's summary -> number of factorisations."""
    runner = session.runner
    n_monitors = max(sum(len(runner.physics[name].noise_monitors) for name in noise), 1)
    wall_start = time.perf_counter()
    done = [0]
    monitor = {'index': -1, 'last_count': math.inf, 'physics': ''}   # progress restarts its count per monitor

    def progress(count, total):
        if count <= monitor['last_count']:
            monitor['index'] += 1
        monitor['last_count'] = count
        done[0] += 1
        emit(_count_progress(runner, min((monitor['index'] + count / total) / n_monitors, 1.0), done[0], wall_start,
                             f"{monitor['physics']}: noise monitor {monitor['index'] + 1} of {n_monitors}, "
                             f'factorisation {count} of {total}', 'factorisation'))
        control.checkpoint(service)

    for name in noise:
        solver = runner.physics[name]
        monitor['physics'] = _physics_text(session, name)
        monitor['last_count'] = math.inf   # the next physics starts a new monitor
        solver.solve_thermal_noise(runner.physics, frequencies, progress)
        for monitor_index, noise_monitor in enumerate(solver.noise_monitors):   # not `monitor`: progress reads it
            emit(LogEvent('info', f'{name}: thermal noise monitor {monitor_index + 1}: {noise_monitor.summary()}'))
            band_fraction = noise_monitor.band_fraction()
            if band_fraction < 0.5:
                emit(LogEvent('warning', f'{name}: thermal noise monitor {monitor_index + 1}: the solved band holds '
                                         f'only {100.0 * band_fraction:.3g} % of the relaxing variance; widen the '
                                         'frequency range to see the whole spectrum'))
    return done[0]


def _validate_thermal_noise(vctx, values):
    """Errors without scipy (host SuperLU) and for a frequency range that does not increase (unless a list is given)."""
    return _scipy_issues(STUDY_NAME, 'host sparse factorisation') + _frequency_range_issues(values)


_THERMAL_NOISE_PARAMS = (
    Param('frequency_start', 'float', 1e3, label='Start frequency', unit='Hz', min=0.0, min_exclusive=True,
          help='Lowest noise frequency in Hz of the range (used when frequencies is empty). Below the '
               'slowest thermal relaxation rate the spectrum is flat. At q_z = 0 the mean temperature of a region '
               'without temperature anchor (insulated) never relaxes: that share is the static_variance of the '
               'result, outside every band. Typical 1e2-1e4.',
          tip='Lowest noise frequency, in hertz.'),
    Param('frequency_stop', 'float', 1e9, label='Stop frequency', unit='Hz', min=0.0, min_exclusive=True,
          help='Highest noise frequency in Hz; must exceed frequency_start. Above the fastest relaxation rate D q^2 / '
               '(2 pi) of the weighting the spectrum falls as 1/f^2 (for water and q = 1.5e7 1/m about 5 MHz). '
               'Typical 1e9-1e11.',
          tip="Highest noise frequency, in hertz. Must be larger than 'Start frequency'."),
    Param('n_frequencies', 'int', 31, label='Number of frequencies', min=1,
          help='Frequencies of the range, spaced as log_spacing selects (both ends included; 1 = the geometric mean, '
               'or the arithmetic mean with log_spacing off). Each costs one sparse factorisation per axial '
               'sample of every thermorefractive_noise feature (host SuperLU, or batched on the GPU with a '
               'line_block solver as the heat shifted solver).',
          tip='Number of frequencies between the start and stop frequency.\n\nEach costs one sparse solve per '
              'noise feature.'),
    Param('log_spacing', 'bool', True, label='Logarithmic spacing',
          help='true: the range frequencies are spaced evenly in log f (geomspace; suits spectra spanning decades); '
               'false: evenly in f (linspace). Ignored when frequencies lists any frequency.',
          tip='On: frequencies spaced evenly on a logarithmic scale, suited to a range spanning several decades.'
              '\n\nOff: frequencies spaced evenly (equal steps in hertz).'),
    Param('frequencies', 'float_list', None, label='Frequency list', unit='Hz', min=0.0, min_exclusive=True,
          optional=True,
          help='Explicit noise frequencies in Hz (each > 0), solved and reported in ascending order; a non-empty '
               'list overrides frequency_start, frequency_stop and n_frequencies.',
          tip="Noise frequencies in hertz; when filled, it replaces the range above."),
    *_background_fixed_point_params('the mode and noise solves', 'the mode and noise solves'),
)
_DESCRIPTION = (
    'Thermorefractive noise of a z-invariant cross-section (fluctuation-dissipation theorem; Levin-type direct '
    'approach). Every heat physics (always planar 2D, never CG-solved here and needing no temperature anchor) '
    'computes the one-sided PSD of X = integral w(x, y) g(z) exp(-j q0 z) dT dV of each of its thermorefractive_noise '
    'features: S_X(f) = 4 k_B T0^2 integral dq/2pi |g~(q - q0)|^2 Re sum conj(w) u dA, where (j omega rho cp + k_zz q^2 '
    '- div k grad) u = w on the heat window (k, boundary conditions, held cells and volumetric losses as assembled for '
    'a steady solve: temperature / fixed-temperature cells are ideal reservoirs and do not fluctuate, insulated faces '
    'are mirrors), one host SuperLU factorisation per (frequency, axial sample), or GPU line-block batches of them '
    '(the heat shifted solver: sparse_direct by default, or line_block). The temperature is uniform T0 (the '
    'heat T0 setting; a heated background is not supported). Every em_modes physics is solved first (its modes give '
    'the mode-pair weightings); every other physics must support the stationary study and runs as a background '
    'physics first. particle_noise features solve the same way for dilute particles diffusing in a liquid (4 n0 '
    'instead of 4 k_B T0^2, diffusivity instead of k / rho cp). Couplings reading heat or em_modes are a validation '
    'error. Results: one xy result per feature '
    '(psd versus frequency) with the exact total variance (sum rule k_B T0^2 integral |w|^2 / (rho cp) dV), its '
    'never-relaxing static part and the share of the rest inside the solved band; one snapshot per mode number after the mode solve (labelled "mode <k>: n_eff '
    '= <value>", as in mode_analysis: step through the modes with the snapshot slider) and one final snapshot '
    'labelled "thermal noise" (Displayed mode of each em_modes physics); ticks = factorisations; '
    'summary noise_frequencies.')


def register(registry):
    """Register the 'thermal_noise' StudyType (background study 'stationary')."""
    registry.register_study(StudyType(
        STUDY_NAME, 'Thermal noise',
        summary='Computes the spectrum of temperature-fluctuation (thermorefractive) noise seen by light in a '
                'waveguide cross-section, for example light scattered into a guided mode. The waveguide modes are '
                'solved first.',
        description=_DESCRIPTION, params=_THERMAL_NOISE_PARAMS, background_study='stationary',
        run=_run_thermal_noise, validate=_validate_thermal_noise,
        feedback_alternative='the thermal noise study feeds no solution back into other physics',
        solver_protocols=(ThermalNoiseSolver, ModeSolver)))
