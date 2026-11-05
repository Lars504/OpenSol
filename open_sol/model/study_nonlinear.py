"""The 'nonlinear_frequency_domain' StudyType (NLFD-chi): background physics to a stationary fixed point, then the
cascaded chi(n) solve of every solver with solve_nonlinear (em_fdfd_nonlinear), one labelled snapshot per channel."""
import math
import time
from dataclasses import asdict

from ..core.constants import C0
from ..core.wavelengths import wavelength_label
from .schema import StudyType
from .solver_hooks import NonlinearSolver
from .study import (_FEEDBACK_ROUNDS_PARAM, Cancelled, LogEvent, _SolverWarnings, _background_fixed_point_params,
                    _background_stage, _count_progress, _feedback_couplings, _feedback_rounds, _log_solve_summaries,
                    _solver_roles)

STUDY_NAME = 'nonlinear_frequency_domain'
_CHANNEL_STAGE = 'Channel solves'
_FEEDBACK_STAGE = 'Feedback rounds'
_TICK_UNIT = 'channel solve'


def _run_nonlinear_frequency_domain(session, values, emit, control, service):
    """Background physics to a stationary fixed point, background -> nonlinear couplings once, solve_nonlinear of
    every nonlinear physics, the feedback rounds when a coupling feeds a nonlinear physics back into the background,
    then one snapshot per channel in solve order; the first input channel stays shown."""
    runner = session.runner
    solved, background = _solver_roles(runner.physics, NonlinearSolver, 'nonlinear frequency-domain',
                                       'add an em_fdfd_nonlinear physics')
    feedback = _feedback_couplings(runner.couplings, solved, background)
    stage = _background_stage(
        session, background, solved, values, emit, control, service,
        ('nonlinear', 'a nonlinear frequency-domain study feeds channel fields back only into background physics'),
        f"Nonlinear frequency-domain study: nonlinear physics {', '.join(solved)}; background physics "
        f"{', '.join(background) or 'none'}",
        [_CHANNEL_STAGE] + [_FEEDBACK_STAGE] * bool(feedback), feedback=feedback)
    status, records, n_snapshots, feedback_record = stage.status, {}, 0, None
    if status == 'completed':
        status, records = _solve_channels(session, solved, stage.stages.stage_emit(_CHANNEL_STAGE), control, service)
    if status == 'completed' and feedback:
        def solve_again(emit_round):
            round_status, round_records = _solve_channels(session, solved, emit_round, control, service,
                                                          log_details=False)
            records.update(round_records)
            return round_status

        status, feedback_record = _feedback_rounds(session, background, solved, feedback, values, solve_again,
                                                   stage.stages.stage_emit(_FEEDBACK_STAGE), control, service,
                                                   stage.live)
    if status == 'completed':
        n_snapshots = _channel_snapshots(session, solved, stage.recorder, stage.live)
    stage.live.update(final=True)
    return {'status': status, 'ticks': n_snapshots, 'tick_unit': 'channel', 't_end': runner.t,
            'fixed_point': stage.fixed_point,
            'n_channels': n_snapshots, 'nonlinear_solve': records, 'feedback': feedback_record}


def _channel_text(channel):
    return f'channel {channel.name}: {wavelength_label(2.0 * math.pi * C0 / channel.omega)}'


def _sweep_fraction(step, steps, residual, tolerance):
    """Progress fraction of a pump-depleted solve: continuation steps done + log10 progress of the residual."""
    if residual <= tolerance:
        within = 1.0
    elif not (math.isfinite(residual) and residual < 1.0):
        within = 0.0
    else:   # here tolerance < residual < 1
        within = min(1.0, math.log(1.0 / residual) / math.log(1.0 / tolerance))
    return (step - 1 + within) / steps


def _solve_channels(session, solved, emit, control, service, log_details=True):
    """solve_nonlinear of every nonlinear physics with Progress (and the cancel check) before each channel solve and a
    log line per pump-depleted sweep; logs non-convergence and (log_details: not in feedback rounds) the mixing terms
    and solve summaries -> (status, {physics: CascadeRecord dict})."""
    runner = session.runner
    solvers = {name: runner.physics[name] for name in solved}
    solver_warnings = _SolverWarnings(solvers, emit)
    n_total = sum(len(solver.channels) for solver in solvers.values())
    wall_start = time.perf_counter()
    records, done = {}, 0
    for position, (name, solver) in enumerate(solvers.items()):
        for line in (solver.mixing_lines() if log_details else ()):
            emit(LogEvent('info', f'{name}: {line}'))
        pump_depleted = solver.coupling_mode == 'pump_depleted'
        settings = solver.iteration_settings
        fraction = [0.0]   # this solver's share, updated per sweep

        def before_channel(channel, name=name, position=position, pump_depleted=pump_depleted, fraction=fraction):
            nonlocal done
            control.checkpoint(service)   # Cancelled between channel solves (a solve cannot be cancelled)
            overall = (position + fraction[0]) / len(solvers) if pump_depleted else done / n_total
            count = f'solve {done + 1}' if pump_depleted else f'{done + 1}/{n_total}'
            emit(_count_progress(runner, overall, done, wall_start,
                                 f'Solving {name} {_channel_text(channel)} ({count}); stopping waits for this solve',
                                 _TICK_UNIT))
            done += 1

        newton = settings.iteration_method == 'newton'
        iteration_word = 'Newton step' if newton else 'sweep'

        def after_sweep(step, sweep, residual, name=name, settings=settings, fraction=fraction,
                        iteration_word=iteration_word):
            fraction[0] = _sweep_fraction(step, settings.continuation_steps, residual, settings.tolerance)
            step_text = (f'continuation step {step}/{settings.continuation_steps}, '
                         if settings.continuation_steps > 1 else '')
            emit(LogEvent('info', f'{name}: {step_text}{iteration_word} {sweep}: largest relative channel change '
                                  f'{residual:.3e} (tolerance {settings.tolerance:.1e})'))

        try:
            cascade = solver.solve_nonlinear(before_channel, after_sweep)
        except Cancelled:
            return 'cancelled', records
        records[name] = asdict(cascade)
        if pump_depleted:
            work = (f'{cascade.sweeps} Newton step(s), {cascade.gmres_iterations} GMRES product(s), '
                    f'{cascade.channel_solves} channel solves' if newton else f'{cascade.sweeps} sweep(s)')
            remedy = ('continuation_steps > 1 or more iterations' if newton
                      else "relaxation < 1, continuation_steps > 1, iteration_method 'newton' or more iterations")
            if cascade.converged:
                emit(LogEvent('info', f'{name}: pump-depleted solve converged after {work} (residual '
                                      f'{cascade.residual:.2e})'))
            else:
                emit(LogEvent('warning', f'{name}: pump-depleted solve did not converge in {work} (residual '
                                         f'{cascade.residual:.2e} > tolerance {settings.tolerance:.1e}); the last '
                                         f'iterate is stored (snapshot labels say so): set {remedy}'))
        solver_warnings.check()
        if log_details:
            _log_solve_summaries({name: solver}, emit, '')
    emit(_count_progress(runner, 1.0, done, wall_start, f'{done} channel solve(s)', _TICK_UNIT))
    return 'completed', records


def _channel_snapshots(session, solved, recorder, live):
    """One snapshot per channel of each nonlinear physics in solve order, labelled 'channel <name>: lambda = <nm> nm'
    (several physics: '<physics> ' first; the others NaN), the last final; then each shows its first input channel."""
    runner = session.runner
    solvers = {name: runner.physics[name] for name in solved}
    shown = {name: solver.selected_channel for name, solver in solvers.items()}
    entries = [(name, solver.channels[solver.channel_index(channel_name)]) for name, solver in solvers.items()
               for channel_name in solver.cascade.order]
    for position, (name, channel) in enumerate(entries):
        solvers[name].select_channel(channel.name)
        text = _channel_text(channel) + ('' if solvers[name].cascade.converged else ' (not converged)')
        recorder.record(final=position == len(entries) - 1, label=text if len(solvers) == 1 else f'{name} {text}',
                        blank_physics={other for other in solvers if other != name})
        live.update()
    for name, solver in solvers.items():
        solver.select_channel(shown[name])
    return len(entries)


_PARAMS = (*_background_fixed_point_params('the nonlinear solve', 'the nonlinear optics solve'), _FEEDBACK_ROUNDS_PARAM)
_DESCRIPTION = (
    'Cascaded frequency-domain nonlinear optics of every physics listing nonlinear_frequency_domain among its '
    'supported studies (em_fdfd_nonlinear): its harmonic channels are solved together, not one wavelength after the '
    'other as in frequency_domain. Undepleted mode: each channel is one FDFD solve in rank order (inputs, then the '
    'channels generated from them, each driven by the nonlinear polarisation of the lower ranks); pump_depleted: '
    'sweeps (picard) or Newton steps (newton) over all channels until self-consistent (a log line per sweep / step '
    'with its residual; not converged within max_iterations: a warning, the last iterate is stored and the snapshot '
    'labels end in "(not converged)"); Progress '
    'before each channel solve, cancel between them. Every other physics must support the stationary study and runs as a '
    'background physics first (heat is always steady here), iterated to a stationary fixed point, then every coupling '
    'from background to nonlinear physics (heat_em_thermo_optic, field_transfer, state-dependent material '
    'properties) is applied once. Solution feedback: a coupling from a nonlinear physics into a background physics '
    '(em_heat_joule into heat: linear, two-photon and free-carrier absorption of every channel) makes the run '
    'alternate rounds of background solve (that coupling applied before each of its sweeps) and nonlinear solve '
    '(pump_depleted warm-started from the last fields) until every primary field changes by at most tolerance between '
    'rounds or feedback_rounds is reached (warning), with a log line per round and the summary key feedback '
    '{couplings, rounds, converged, relative_change, tolerance, history} (None without feedback); couplings into a '
    'nonlinear physics from a nonlinear physics and material properties reading one are validation errors. The run '
    'log lists the mixing terms and each channel solve summary. Stores one snapshot per channel in '
    'solve order of the stored fields (default the primary e_abs), labelled "channel <name>: lambda = <nm> nm" '
    '(several nonlinear physics: their name first, the others NaN); afterwards each shows its first input channel '
    '(Results.field). ticks = n_channels = the channel snapshots; summary nonlinear_solve = {physics: {mode, order '
    '(channel names in solve order), channel_solves, sweeps, converged, residual, tolerance, residual_history, '
    "continuation_steps, method ('picard' | 'newton', '' undepleted), gmres_iterations, backtracks}}. Spectra (a QPM "
    'tuning curve, SHG versus pump '
    'wavelength) and power curves are parameter sweeps over the poling period, the input wavelength or the source '
    'amplitude, with derived values of channel monitors or the power_exchange_<channel> scalars.')


def register(registry):
    """Register the 'nonlinear_frequency_domain' StudyType (background study 'stationary')."""
    registry.register_study(StudyType(
        STUDY_NAME, 'Nonlinear frequency domain',
        summary='Solves nonlinear optics at several coupled wavelengths, such as a pump and its second harmonic. Any '
                'heat or current physics is first solved to a steady state.',
        description=_DESCRIPTION, params=_PARAMS, background_study='stationary',
        run=_run_nonlinear_frequency_domain, solution_feedback=True,
        feedback_alternative='feed a nonlinear physics back through a coupling into a background physics instead (e.g. '
                             'em_heat_joule into heat)', solver_protocols=(NonlinearSolver,)))
