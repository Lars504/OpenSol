"""Built-in study types and the study driver: build the spec (once per sweep value), run its StudyType and
report through plain CPU-only events (they cross to the GUI thread); sessions are closed explicitly."""
import math
import threading
import time
import traceback
from dataclasses import dataclass, replace

import numpy as np

from ..core import gpu_memory
from ..core.backend import array_module_of, cp, to_host
from ..core.constants import C0
from ..core.damped_iteration import AndersonHistory
from ..core.kernels import axis_first_reduce
from ..core.mirror import mirror_faces, parity_tag, sector_specs
from ..core.renderer import primary_field_info
from ..core.time_estimate import format_seconds, merge_time_checks, time_check, time_check_warnings
from ..core.wavelengths import resolved_wavelengths, wavelength_label
from ..solvers import cudss_factor
from ..solvers.cudss_factor import cudss_orderings_kept
from ..solvers.roles import resolved_precision
from .analytic import derived_references, is_segment_3d, line_endpoints
from .builder import build as build_model, external_memory_record, run_memory_record
from .meshing import element_meshes_kept
from .definition_helpers import wavelength_range_issues
from .expressions import evaluate_parameters, evaluate_scalar
from .schema import RESULT_KINDS, Issue, Param, StudyType, get_registry, resolve_values
from .session import Snapshot, earlier_derived_names, live_snapshot_index
from .solver_hooks import (HarmonicFrequencyPlanner, HarmonicSolver, ModeSolver, PrestressSolver, PulsedHarmonicSolver,
                           WarmStartSolver, hook_names)

__all__ = [
    'Progress', 'LiveFrame', 'LiveResults', 'LogEvent', 'SweepStep', 'SweepValueDone', 'Built', 'BuildFailed',
    'StudyStarted', 'StudyFinished', 'StudyFailed', 'Cancelled', 'RunControl', 'run_study', 'build_session',
    'release_session', 'solve_stationary', 'register',
]

_LIVE_PIXELS = 250_000                 # default LiveFrame resolution
_SNAPSHOT_PIXELS = 250_000             # per stored snapshot field
_SNAPSHOT_BYTE_LIMIT = 1 << 30         # intermediate snapshots stop above 1 GB (the final state is always stored)
_LIVE_RESULT_POINTS = 4000             # samples per LiveResults timeseries
_LIVE_RESULT_PERIOD = 1.0              # s of wall time between LiveResults
_CHUNK_SECONDS = 0.1                   # target wall time of one runner.advance chunk
_PAUSE_POLL_SECONDS = 0.02
_CONVERGENCE_FLOOR = 1e-30
_GROWTH_STOP_ITERATIONS = 3            # successive growing relative changes (from iteration 4) that stop the loop
_BACKGROUND_STAGE = 'Background solve'  # Progress stage names (GUI / CLI progress text)
_PULSE_STAGE = 'FDTD pulse'
_WAVELENGTH_STAGE = 'Wavelength solves'
_MODE_STAGE = 'Mode solve'
# fixed-point monitor reductions; the NaN test keeps a NaN (a plain max would drop it and hide a diverged loop)
_NAN_KEEPING_MAX = '(a > b || a != a) ? a : b'
_max_abs_difference = cp.ReductionKernel(
    'T current, T previous', 'T change', 'current > previous ? current - previous : previous - current',
    _NAN_KEEPING_MAX, 'change = a', '0', 'open_sol_max_abs_difference')
_max_abs = cp.ReductionKernel('T values', 'T peak', 'values < 0 ? -values : values', _NAN_KEEPING_MAX, 'peak = a', '0',
                              'open_sol_max_abs')


def _peak(partials):
    """NaN-keeping max |.| of axis-first partial maxima (axis-first again while they are large: 3D fields)."""
    return axis_first_reduce(_max_abs, partials, combine=_peak)


# ---- events (plain dataclasses holding CPU data only) ----
@dataclass
class Progress:
    """Study progress: fraction of the whole run, simulated t / t_end [s], ticks done and their wall-clock rate;
    tick_unit names a tick ('step', 'iteration', 'wavelength', 'mode solve', 'eigen solve', 'factorisation', 'channel
    solve'). stages: the run's stage names in order, stages[stage_index] the current one with its own fraction and
    wall time (None: fraction / wall_elapsed)."""
    fraction: float
    t: float
    t_end: float
    tick: int
    wall_elapsed: float
    ticks_per_second: float
    message: str = ''
    tick_unit: str = 'step'
    stages: tuple = ()
    stage_index: int = 0
    stage_fraction: float | None = None
    stage_elapsed: float | None = None


def tick_count_text(summary) -> str:
    """'120 steps', '1 wavelength': a study summary's ticks counted in its tick_unit (GUI and run log share it)."""
    ticks = summary.get('ticks')
    unit = summary.get('tick_unit') or 'step'
    return f"{'?' if ticks is None else ticks} {unit}{'' if ticks == 1 else 's'}"


@dataclass
class LiveFrame:
    """Latest live field (session.FieldData, numpy values)."""
    field: object


@dataclass
class LiveResults:
    """(physics, feature) -> timeseries result dict, min / max decimated to <= 4000 samples ('decimation_stride')."""
    results: dict


@dataclass
class LogEvent:
    level: str      # 'info' | 'warning' | 'error'
    message: str


@dataclass
class SweepStep:
    """A sweep value is about to be built; index is 0-based."""
    index: int
    count: int
    parameter: str
    value: float


@dataclass
class SweepValueDone:
    """A sweep value ended (computed or failed; index 0-based); sweep = a copy of the sweep table so far (the layout
    of StudyFinished.summary['sweep'], CPU data), so views fill in while the sweep runs."""
    index: int
    count: int
    parameter: str
    value: float
    sweep: dict


@dataclass
class Built:
    """A build succeeded; summary is the session.BuildSummary; session_serial: BackendWorker's number of the new
    session (0 outside the worker), which WorkerBridge passes with each call."""
    summary: object
    session_serial: int = 0


@dataclass
class BuildFailed:
    """A build failed; issues are model.schema.Issue (empty for non-model errors); out_of_memory: GPU OOM;
    build_memory: the builder's ledger so far + estimate (Session.build_memory layout), None before any GPU stage."""
    message: str
    traceback: str
    issues: list
    out_of_memory: bool = False
    build_memory: dict | None = None


@dataclass
class StudyStarted:
    pass


@dataclass
class StudyFinished:
    """summary: status, wall_time, ticks + tick_unit, t_end, snapshot_times, snapshot_labels, live_snapshot (index of
    the snapshot holding the live state, None = the last), stored_fields, stored_cuts, derived,
    derived_reference / derived_error (derived values with a reference), derived_series (the source-free derived
    values at every time_dependent / frequency_domain snapshot as an 'xy' dict of lists, None below 2 points),
    fixed_point (solve_stationary's record or None), study_seconds, solvers (_solver_records), solvers_at_study_start
    (the same at the study's start: counter-timed work done at build), run_memory (builder.run_memory_record or None), external_memory
    (builder.external_memory_record: cuDSS outside the pool, or None), parameters
    (of the session these values come from), sweep, sweep_index (the value they belong to; None without a sweep), session_live
    (False when no session is left to query: a sweep whose last value failed); frequency_domain adds wavelengths,
    mode_analysis n_modes (mode snapshots stored)."""
    summary: dict


@dataclass
class StudyFailed:
    """A study failed; out_of_memory: a GPU out-of-memory closed a session of this study; session_live: a session of
    this run is still open to query (False: the worker holds none); build_memory: on OOM the ledger + estimate of the
    build that ran out (or of the session whose run did), else None."""
    message: str
    traceback: str
    out_of_memory: bool = False
    session_live: bool = False
    build_memory: dict | None = None


class Cancelled(Exception):
    """Raised by RunControl.checkpoint once the study was cancelled."""


class RunControl:
    """Thread-safe pause / cancel flags and the live-field request shared by the GUI side and the running study;
    live_results=False (headless) skips building LiveResults."""

    def __init__(self, live_results=True):
        self._lock = threading.Lock()
        self._paused = False
        self._cancelled = False
        self._live_field = None
        self.live_results = bool(live_results)

    def pause(self):
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    def cancel(self):
        with self._lock:
            self._cancelled = True

    def reset(self):
        """Clear pause and cancel; the live-field request is kept."""
        with self._lock:
            self._paused = False
            self._cancelled = False

    def set_live_field(self, physics, component, max_pixels=_LIVE_PIXELS):
        """Field streamed as LiveFrames (physics None = unique owner of the component)."""
        if int(max_pixels) < 1:
            raise ValueError(f'max_pixels must be >= 1, got {max_pixels}')
        with self._lock:
            self._live_field = (physics, component, int(max_pixels))

    @property
    def live_field(self):
        """(physics, component, max_pixels) requested for LiveFrames, or None (primary field of the first physics)."""
        with self._lock:
            return self._live_field

    def checkpoint(self, service=None):
        """service() on every call; raise Cancelled when cancelled; while paused keep calling service() every 20 ms."""
        while True:
            if service is not None:
                service()
            with self._lock:
                cancelled, paused = self._cancelled, self._paused
            if cancelled:
                raise Cancelled('study cancelled')
            if not paused:
                return
            time.sleep(_PAUSE_POLL_SECONDS)


# ---- study driver ----
def run_study(spec, emit, control, build=None, service=None, on_session=None):
    """Build `spec` (once per sweep value), run its StudyType and emit events; errors become StudyFailed (a GPU OOM
    also closes the session). Returns the last built Session still alive or None; never closes one it did not build."""
    wall_start = time.perf_counter()
    latest_session = None
    build_seconds_total = 0.0   # every build of the study (sweep values, refinement levels, parity sectors)

    def track_session(session):
        nonlocal latest_session, build_seconds_total
        latest_session = session
        if session is not None:
            build_seconds_total += session.build_seconds
        if on_session is not None:
            on_session(session)

    failure = None   # (message, traceback text, out_of_memory): the exception itself is not kept
    try:
        registry = get_registry()
        with cudss_orderings_kept():   # a cuDSS re-plan of a pattern planned in this study skips the reordering
            if spec.study.estimation is not None:   # parameter estimation; validation refuses sweeps / adaptation
                from .study_estimation import run_estimation   # it imports this module's helpers
                summary = run_estimation(spec, registry, build, emit, control, service, track_session)
            elif spec.study.adaptation is not None:   # mesh refinement levels (S7.7); validation refuses sweeps / sectors
                from .study_adaptation import run_adaptation   # it imports this module's helpers
                summary = run_adaptation(spec, registry, build, emit, control, service, track_session)
            elif _parity_sectors_requested(spec, registry):
                from .study_sectors import run_parity_sectors   # it imports this module's helpers
                with element_meshes_kept():   # sectors (and their sweep values) with the same mesh mesh once
                    summary = run_parity_sectors(spec, registry, build, emit, control, service, track_session)
            elif spec.study.sweep is None:
                summary = _run_single(spec, registry, build, emit, control, service, track_session)
            else:
                with element_meshes_kept():   # geometry-free sweep values reuse the element mesh
                    summary = _run_sweep(spec, registry, build, emit, control, service, track_session)
    except Exception as exc:
        failure = _record_failure(exc)
    if failure is not None:
        message, failure_traceback, out_of_memory = failure
        build_memory = None
        if out_of_memory:   # OOM is a normal failure: close the session that ran out and return its memory
            build_memory = getattr(latest_session, 'build_memory', None)
            if build_memory is not None:
                emit(LogEvent('info', f'session that ran out: {gpu_memory.ledger_text(build_memory)}'))
            try:
                release_session(track_session, latest_session, emit)   # resets latest_session to None
            except Exception as exc:   # logged, not raised: StudyFailed stays the run's final event
                emit(LogEvent('error', f'freeing GPU memory after the failed run: {_failure_message(exc)}'))
        emit(StudyFailed(message, failure_traceback, out_of_memory=out_of_memory,
                         session_live=latest_session is not None, build_memory=build_memory))
        return latest_session
    if summary is not None:
        summary['wall_time'] = time.perf_counter() - wall_start
        summary['build_seconds_total'] = build_seconds_total
        summary['session_live'] = latest_session is not None   # False: a sweep's last value failed, nothing to query
        emit(StudyFinished(summary))
    return latest_session


def build_session(spec, emit, on_session=None, build=None, registry=None, live_parameters=()):
    """Emit Progress('Building'), build, on_session(session), emit Built; returns (session, None), or
    (None, BuildFailed) after releasing the partial build and then emitting BuildFailed. live_parameters: a reused
    sweep build's (builder.build keyword, passed only when set)."""
    emit(Progress(0.0, 0.0, 0.0, 0, 0.0, 0.0, message='Building', stages=('Building',)))
    # cached blocks of an earlier live session's run (api Results kept open) would count in the build peak and run memory
    gpu_memory.release_all(collect_garbage=False)
    baseline_bytes = gpu_memory.pool_used_bytes()
    # free chunk remainders release_all could not return (split around those Results' arrays): not this run's memory
    baseline_free_bytes = gpu_memory.pool_reserved_bytes() - baseline_bytes
    cudss_factor.reset_external_high_water()   # the outside-pool counters of the live cuDSS factors
    external_baseline_bytes = cudss_factor.external_live_bytes()   # factors of earlier open Results
    session = None
    failure = None
    try:
        if build is None:
            build = build_model
        if registry is None:
            registry = get_registry()
        options = {'live_parameters': frozenset(live_parameters)} if live_parameters else {}
        session = build(spec, registry=registry, log=_log_to(emit), **options)
        session.pool_baseline_bytes = baseline_bytes
        session.pool_baseline_free_bytes = baseline_free_bytes
        session.external_baseline_bytes = external_baseline_bytes
        summary = session.summary()
        gpu_memory.release_all(collect_garbage=False)   # cached blocks of the build temporaries back to the device
        if on_session is not None:
            on_session(session)
    except Exception as exc:   # record text only: the traceback keeps the failed build's frames (and arrays) alive
        out_of_memory = gpu_memory.is_out_of_memory(exc)
        issues = list(getattr(exc, 'issues', None) or ())
        if out_of_memory and not issues:
            message = gpu_memory.out_of_memory_message(exc)
            issues = [Issue('error', ('model',), message)]
        else:
            message = _failure_message(exc)
        build_memory = getattr(exc, 'build_memory', None) or getattr(session, 'build_memory', None)
        failure = BuildFailed(message, traceback.format_exc(), issues, out_of_memory=out_of_memory,
                              build_memory=build_memory)
    if failure is not None:
        if failure.build_memory is not None:   # where the failed build spent its time (and memory, after an OOM)
            if failure.out_of_memory:
                emit(LogEvent('info', gpu_memory.ledger_text(failure.build_memory)))
            emit(LogEvent('info', gpu_memory.ledger_time_text(failure.build_memory)))
        try:   # a session built before the tail failed is closed (and taken back from on_session) here
            release_session(on_session, session, emit=emit, baseline_bytes=baseline_bytes)
        except Exception as exc:   # logged, not raised: BuildFailed stays the build's only final event
            emit(LogEvent('error', f'freeing GPU memory after the failed build: {_failure_message(exc)}'))
        session = None
        emit(failure)   # after any leak warning
        return None, failure
    emit(Built(summary))
    return session, None


def release_session(on_session=None, session=None, emit=None, baseline_bytes=None):
    """on_session(None), session.close(), then free cupy's pools (GPU thread); with emit, pool use above the baseline
    (baseline_bytes, else session.pool_baseline_bytes) is logged as a leak. Returns the pool's used bytes after."""
    if baseline_bytes is None and session is not None:
        baseline_bytes = session.pool_baseline_bytes
    if on_session is not None:
        on_session(None)
    if session is not None:
        session.close()
    gpu_memory.release_all()
    used_bytes = gpu_memory.pool_used_bytes()
    if emit is not None and baseline_bytes is not None:
        residual_bytes = used_bytes - baseline_bytes
        if residual_bytes > gpu_memory.LEAK_TOLERANCE_BYTES:
            emit(LogEvent('warning', f'GPU memory leak: {gpu_memory.format_bytes(residual_bytes)} still allocated '
                                     f'after closing the previous session (pool used '
                                     f'{gpu_memory.format_bytes(used_bytes)}, baseline before its build '
                                     f'{gpu_memory.format_bytes(baseline_bytes)})'))
    return used_bytes


def _run_single(spec, registry, build, emit, control, service, on_session):
    """Build and run one spec; summary dict, or None after a build failure (StudyFailed emitted)."""
    release_session(on_session)
    session, failure = build_session(spec, emit, on_session, build=build, registry=registry)
    if failure is not None:
        emit(StudyFailed(f'build failed: {failure.message}', failure.traceback,
                         out_of_memory=failure.out_of_memory,
                         build_memory=failure.build_memory if failure.out_of_memory else None))
        return None
    emit(StudyStarted())
    summary = _run_built(session, spec, registry, emit, control, service)
    summary['sweep'], summary['sweep_index'] = None, None
    return summary


def _parity_sectors_requested(spec, registry):
    """True when the study setting parity_sectors is on (over the base parameters); False when it does not resolve
    (the build then reports the problem)."""
    try:
        study_type = registry.study_type(spec.study.type)
        values = resolve_values(study_type.params, spec.study.settings, evaluate_parameters(spec.parameters))
    except Exception:   # reported by the build's validation
        return False
    return bool(values.get('parity_sectors'))


def _run_sweep(spec, registry, build, emit, control, service, on_session):
    """Run one spec copy per sweep value, closing the previous value's session first -> last run's summary + 'sweep'
    table (NaN / {} for failed values, listed in 'failed'; no live session if the last failed), or None (StudyFailed)
    if none ran."""
    parameter = spec.study.sweep.parameter
    setup_failure = None
    try:
        parameter, sweep_values = sweep_values_of(spec)
        study_values = resolve_values(registry.study_type(spec.study.type).params, spec.study.settings,
                                      evaluate_parameters(spec.parameters))
        warm_start = bool(study_values.get('warm_start'))   # absent (time_dependent) = cold
        reuse_build = spec.study.sweep.reuse_build is True   # build once, re-set the following loads per value
        live_parameters = frozenset()
        if reuse_build:
            from .sweep_reuse import live_parameter_names, non_rerun_physics, reuse_issues   # reused sweeps only
            refusals = reuse_issues(spec, registry)
            if refusals:   # validation of the full spec names every one at its node (_fail_sweep_setup)
                raise ValueError(f'reuse_build is refused: {refusals[0][1]}'
                                 + (f' (and {len(refusals) - 1} more)' if len(refusals) > 1 else ''))
            live_parameters = live_parameter_names(spec, parameter)
    except Exception as exc:   # text only; reported below through the build path
        setup_failure = _record_failure(exc)
    if setup_failure is not None:
        _fail_sweep_setup(spec, registry, build, emit, on_session, setup_failure)
        return None
    derived_names = [derived.name for derived in spec.derived_values]
    reference_names = [derived.name for derived in spec.derived_values if derived.reference is not None]
    table = {'parameter': parameter, 'values': [], 'derived': {name: [] for name in derived_names},
             'reference': {name: [] for name in reference_names},   # also for failed values (parameters only)
             'error': {name: [] for name in reference_names},
             'results': [], 'failed': [],   # failed: [{index, value, message}] of the values with NaN / {} entries
             'converged': [],   # per value: fixed_point converged, None (failed value or no fixed point)
             'derived_series': [],   # per value: the run's derived_series xy, or None
             'builds': 0}   # model builds of the sweep (1 with reuse_build)
    kept_fields = _KeptSweepFields(emit) if spec.study.sweep.keep_fields else None
    if kept_fields is not None:
        table['snapshots'] = []   # per value: its host Snapshots ([] for a failed value)
    last_summary = None
    last_index = None          # sweep value index of last_summary
    value_runs = []            # [(label, summary)] of every value that ran (summed_runs)
    unavailable_results = set()   # result keys already reported as not available (warned once per sweep)
    first_run_failure = None   # (message, traceback text, out_of_memory) of the first value whose run raised
    session_closed_by_oom = False   # some value's session was closed after a GPU OOM (on_session got None)
    started = False
    summary_is_live = False    # last_summary describes the session that is still alive
    status = 'completed'
    previous_session = None    # the previous value's session (on_session holds it too), closed before the next build
    warm_state = None          # host state of the last computed value (warm_start only), seeds the next build
    reused_session = None      # reuse_build: the kept session that takes the next value (None: build it)
    for index, value in enumerate(sweep_values):
        if index > 0:
            try:
                control.checkpoint(service)
            except Cancelled:
                status = 'cancelled'
                break
        emit(SweepStep(index, len(sweep_values), parameter, value))
        value_spec = sweep_value_spec(spec, parameter, value)
        built_now = reused_session is None
        if built_now:
            if index == 0:
                release_session(on_session)
            else:
                release_session(on_session, previous_session, emit)
                previous_session = None
            session, failure = build_session(value_spec, emit, on_session, build=build, registry=registry,
                                             live_parameters=live_parameters)
            table['builds'] += 1
            if failure is None and reuse_build:
                cannot_rerun = non_rerun_physics(session)
                if cannot_rerun:   # their solvers cannot rerun a kept build: one build per value, as without reuse
                    reuse_build, live_parameters = False, frozenset()
                    emit(LogEvent('info', f"sweep reuse_build: physics {', '.join(cannot_rerun)} cannot rerun a kept "
                                          'build (no reset_records); every sweep value is built'))
                else:
                    reused_session = session
                    emit(LogEvent('info', f'sweep reuse_build: this build takes every value of {parameter} (the '
                                          'following feature values are re-set per value)'))
        else:   # a run failure drops reused_session, so this is a computed value's session
            session, failure = reused_session, None
        summary_is_live = False
        derived, results, converged, derived_series, kept_snapshots = {}, {}, None, None, []
        if failure is not None:
            emit(LogEvent('error', f'sweep {parameter} = {value:g}: build failed; its results are NaN'))
            table['failed'].append({'index': index, 'value': value, 'message': f'build failed: {failure.message}'})
        else:
            if not started:
                emit(StudyStarted())
                started = True
            run_failure = None   # (message, traceback text, out_of_memory): the exception itself is not kept
            try:
                if not built_now:   # the kept build takes this value: loads re-set, run records dropped
                    session.set_sweep_value(parameter, value)
                    session.reset_run_records()
                if warm_state is not None:
                    _seed_warm_start(session, warm_state, emit, value)
                value_summary = _run_built(session, value_spec, registry, emit, control, service)
                cancelled = value_summary['status'] == 'cancelled'
                value_results = {} if cancelled else _feature_results(session, emit, unavailable_results)
                if warm_start and not cancelled:
                    warm_state = _warm_start_state(session, value, warm_state)
            except Exception as exc:   # NaN like a failed build; the remaining values still run
                run_failure = _record_failure(exc)
            if run_failure is not None:
                message, _, out_of_memory = run_failure
                if first_run_failure is None:
                    first_run_failure = run_failure
                emit(LogEvent('error', f'sweep {parameter} = {value:g}: {message}; its results are NaN'))
                table['failed'].append({'index': index, 'value': value, 'message': message})
                reused_session = None   # a failed run's state is not rerun: the next value builds anew
                if out_of_memory:   # close it now so the next value starts from a freed GPU
                    release_session(on_session, session, emit)
                    session = None
                    session_closed_by_oom = True
            else:
                last_summary, summary_is_live, last_index = value_summary, True, index
                value_runs.append((f'{parameter} = {value:g}', value_summary))
                if cancelled:   # partial run: not a sweep point
                    status = 'cancelled'
                    break
                derived, results = value_summary['derived'], value_results
                derived_series = value_summary.get('derived_series')
                if value_summary['fixed_point'] is not None:
                    converged = value_summary['fixed_point']['converged']
                if kept_fields is not None:
                    kept_snapshots = kept_fields.take(session.snapshots)
        previous_session, session = session, None
        table['values'].append(value)
        table['converged'].append(converged)
        for name in derived_names:
            table['derived'][name].append(float(derived.get(name, math.nan)))
        references, errors, _ = derived_references(spec.derived_values,
                                                   evaluate_parameters(value_spec.parameters, errors={}), derived)
        for name in reference_names:   # failing references were logged by the value's run
            table['reference'][name].append(float(references[name]))
            table['error'][name].append(float(errors[name]))
        table['results'].append(results)
        table['derived_series'].append(derived_series)
        if kept_fields is not None:
            table['snapshots'].append(kept_snapshots)
        emit(SweepValueDone(index, len(sweep_values), parameter, value, sweep_table_copy(table)))
    if last_summary is None:   # previous_session: a non-OOM failed run's session, still open (on_session holds it)
        if status == 'cancelled':
            emit(StudyFailed('the sweep was cancelled before any value was computed', '',
                             out_of_memory=session_closed_by_oom, session_live=previous_session is not None))
        elif first_run_failure is not None:
            message, failure_traceback, _ = first_run_failure   # an OOM there also set session_closed_by_oom
            emit(StudyFailed(f"no value of the sweep over '{parameter}' could be computed; first error: {message}",
                             failure_traceback, out_of_memory=session_closed_by_oom,
                             session_live=previous_session is not None))
        else:
            emit(StudyFailed(f"no value of the sweep over '{parameter}' could be built", ''))
        return None
    if not summary_is_live:   # the last processed value failed: last_summary's snapshots went with its session
        last_summary['snapshot_times'] = []
        last_summary['snapshot_labels'] = []
        last_summary['live_snapshot'] = None
        if previous_session is not None:   # its half-run session (non-OOM run failure) must not pose as the results
            release_session(on_session, previous_session, emit)   # resets run_study's latest_session
    last_summary.update(summed_runs(value_runs))
    last_summary['status'] = status
    last_summary['sweep'], last_summary['sweep_index'] = table, last_index
    return last_summary


def summed_runs(labelled_summaries):
    """{ticks, study_seconds, time_check} summed over the run summaries [(label, summary)] of one study (sweep values,
    refinement levels, parity sectors); time_check merged with item labels '<label>: <item>'."""
    pairs = list(labelled_summaries)
    return {'ticks': sum(int(summary.get('ticks') or 0) for _, summary in pairs),
            'study_seconds': sum(float(summary.get('study_seconds') or 0.0) for _, summary in pairs),
            'time_check': merge_time_checks((label, summary.get('time_check')) for label, summary in pairs)}


def sweep_value_spec(spec, parameter, value):
    """Copy of spec at one sweep value: the parameter set to it, no sweep, no analytic curves targeting 'sweep' (they
    belong to the whole sweep: api / GUI draw them from spec, else each value warns about them)."""
    value_spec = spec.copy()
    value_spec.parameter(parameter).expression = repr(float(value))
    value_spec.study.sweep = None
    value_spec.analytic_curves = [curve for curve in value_spec.analytic_curves if curve.target != 'sweep']
    return value_spec


def sweep_values_of(spec):
    """(parameter, [float sweep values]) of spec.study.sweep over the base parameters; KeyError / ValueError when the
    parameter is unknown or the list empty."""
    parameter = spec.study.sweep.parameter
    spec.parameter(parameter)   # KeyError naming the known parameters
    base_parameters = evaluate_parameters(spec.parameters)
    values = [evaluate_scalar(value, base_parameters) for value in spec.study.sweep.point_values()]
    if not values:
        raise ValueError(f"the sweep over '{parameter}' has no values")
    return parameter, values


def sweep_table_copy(table):
    """Copy of a sweep table whose lists (and the lists of its dict columns) later values no longer extend; the result
    dicts, xy dicts and Snapshots inside are shared (never modified afterwards)."""
    return {key: ({name: list(column) for name, column in value.items()} if isinstance(value, dict)
                  else list(value) if isinstance(value, list) else value)
            for key, value in table.items()}


def _field_bytes(field_data):
    """Host bytes a stored FieldData holds of its own: values plus triangle node values (the display triangulation's
    nodes / triangles are shared by every FieldData of a mesh view)."""
    triangulation = field_data.triangulation
    return field_data.values.nbytes + (0 if triangulation is None else np.asarray(triangulation[2]).nbytes)


def _snapshot_bytes(snapshots):
    return sum(_field_bytes(field_data) for snapshot in snapshots
               for fields in [snapshot.fields, *snapshot.cuts] for field_data in fields.values())


class _KeptSweepFields:
    """SweepSpec.keep_fields: the host Snapshots of every sweep value within _SNAPSHOT_BYTE_LIMIT in total; past it
    only each value's final snapshot is kept (one warning)."""

    def __init__(self, emit):
        self._emit = emit
        self._stored_bytes = 0
        self._finals_only = False

    def take(self, snapshots):
        """List of the snapshots kept of one value (read before its session closes)."""
        snapshots = list(snapshots)
        if not self._finals_only and self._stored_bytes + _snapshot_bytes(snapshots) > _SNAPSHOT_BYTE_LIMIT:
            self._finals_only = True
            self._emit(LogEvent('warning', f'kept sweep snapshots exceed {_SNAPSHOT_BYTE_LIMIT / 2**30:.0f} GB: from '
                                           'this value on only the final snapshot of each value is kept'))
        if self._finals_only:
            snapshots = snapshots[-1:]
        self._stored_bytes += _snapshot_bytes(snapshots)
        return snapshots


def _fail_sweep_setup(spec, registry, build, emit, on_session, setup_failure, what='sweep'):
    """Report a sweep (or the `what` loop, e.g. 'parity sector') that cannot be set up through build_session(spec): its
    validation raises the ModelError with the node paths (BuildFailed; the build callback sees it) before any GPU
    work. Emits StudyFailed."""
    release_session(on_session)
    session, failure = build_session(spec, emit, on_session, build=build, registry=registry)
    if failure is not None:
        emit(StudyFailed(f'build failed: {failure.message}', failure.traceback, out_of_memory=failure.out_of_memory))
        return
    release_session(on_session, session, emit)   # a build callback that skips validation: drop the base build
    message, failure_traceback, out_of_memory = setup_failure
    emit(StudyFailed(f'{what} set-up failed: {message}', failure_traceback, out_of_memory=out_of_memory))


def _run_built(session, spec, registry, emit, control, service):
    """Run the spec's StudyType on a built session; per-run summary (status, ticks, t_end, snapshots, derived)."""
    study_type = registry.study_type(spec.study.type)
    if study_type.run is None:
        raise ValueError(f"study type '{study_type.name}' has no run function")
    values = resolve_values(study_type.params, spec.study.settings, session.parameters)
    solvers_at_start = _solver_records(session)   # counter-timed work done at build (a factorisation in finalize)
    study_start = time.perf_counter()
    try:
        additions = study_type.run(session, values, emit, control, service) or {}
    except Cancelled:   # a run that does not handle cancellation itself
        additions = {'status': 'cancelled'}
    session.domain.backend.synchronize()   # kernels still queued belong to the study time (cpu: nothing)
    study_seconds = time.perf_counter() - study_start
    # the pool keeps its freed blocks during a run: reserved - baseline = kept + run high-water mark (VER-5); the cpu
    # backend has no pool and traces the build only (MemoryLedger), so it records no run memory; free blocks stranded
    # before the build are left out (a lower bound when the run reused them), at least the used-bytes growth
    unmeasured = session.pool_baseline_bytes is None or session.domain.backend.name == 'cpu'
    run_memory, run_memory_message = (None, '') if unmeasured else run_memory_record(
        session.build_memory,
        max(gpu_memory.pool_reserved_bytes() - session.pool_baseline_bytes - session.pool_baseline_free_bytes,
            gpu_memory.pool_used_bytes() - session.pool_baseline_bytes))
    if run_memory_message:
        emit(LogEvent('warning', run_memory_message))
    external_memory, external_memory_message = None, ''
    if not unmeasured and session.external_baseline_bytes is not None:
        external_memory, external_memory_message = external_memory_record(
            session.build_memory,
            max(cudss_factor.external_high_water_bytes() - session.external_baseline_bytes, 0))
    if external_memory_message:
        emit(LogEvent('warning', external_memory_message))
    derived = session.derived_values()
    references, errors, messages = derived_references(spec.derived_values, session.parameters, derived)
    for message in messages:
        emit(LogEvent('warning', message))
    stored_keys = _stored_field_keys(session)
    summary = {'status': 'completed', 'ticks': 0, 'tick_unit': 'step', 't_end': float(session.time),
               'snapshot_times': [float(snapshot.time) for snapshot in session.snapshots],
               'snapshot_labels': [snapshot.label for snapshot in session.snapshots],
               'live_snapshot': live_snapshot_index(session.snapshots),
               'stored_fields': stored_keys, 'stored_cuts': _stored_cut_records(session),
               'meshes': session.mesh_records(), 'field_meshes': field_meshes(session, stored_keys),
               'derived': derived, 'derived_reference': references, 'derived_error': errors, 'derived_series': None,
               'fixed_point': None,
               'study_seconds': study_seconds, 'solvers': _solver_records(session),
               'solvers_at_study_start': solvers_at_start, 'run_memory': run_memory,
               'external_memory': external_memory,
               'parameters': {name: float(value) for name, value in session.parameters.items()}}
    summary.update(additions)
    summary['time_check'] = _time_check(session, summary, emit)
    return summary


def _time_check(session, summary, emit):
    """summary time_check of the build's run-time estimate against this run (None without one); its warnings and a
    one-line comparison go to the log."""
    estimate = getattr(session, 'time_estimate', None)
    if estimate is None or summary.get('status') != 'completed':
        return None
    record = time_check(estimate, summary)
    ratio = record['total_ratio']
    emit(LogEvent('info', f"run time {format_seconds(record['study_seconds'])}, estimated "
                          f"{format_seconds(record['predicted_seconds'])}"
                          + (f' ({ratio:.2g}x)' if ratio is not None else '')))
    for message in time_check_warnings(record):
        emit(LogEvent('warning', message))
    return record


def field_meshes(session, keys):
    """{'physics.component': mesh name} of the keys whose physics the (open) session holds (S7.2)."""
    return {key: session.mesh_of(key.partition('.')[0]) for key in keys if key.partition('.')[0] in session.physics}


def _stored_cut_records(session):
    """[[axis, cell-centre position [m]]] of the stored cuts the snapshots hold (Snapshot.cuts order), [] without."""
    snapshot = next((snapshot for snapshot in reversed(session.snapshots) if snapshot.cuts), None)
    if snapshot is None:
        return []
    return [[str(plane[0]), float(plane[1])]
            for plane in (next(iter(fields.values())).plane for fields in snapshot.cuts if fields)]


def _solver_records(session):
    """{physics: {'roles': {role: {type, solver (node name, None = Automatic), backend, precision, settings (resolved
    values), counters}}, 'counters': the physics' own work_counters() (loops and solves outside a role)}}."""
    records = {}
    built = session.context.solver_records
    for name, physics in session.runner.physics.items():
        roles = {role: {'type': entry.resolved.solver_type.name, 'solver': entry.resolved.node,
                        'backend': entry.backend,
                        'precision': resolved_precision(entry.resolved.solver_type, entry.resolved.values),
                        'settings': dict(entry.resolved.values),
                        'counters': entry.runtime.work_counters() if hasattr(entry.runtime, 'work_counters') else {}}
                 for role, entry in built.get(name, {}).items()}
        counters = physics.work_counters() if hasattr(physics, 'work_counters') else {}
        if roles or counters:
            records[name] = {'roles': roles, 'counters': counters}
    return records


def _feature_results(session, emit, reported):
    """'physics.feature' -> result dict of every result feature (dft_fields: metadata only, no freq_index); a result
    that cannot be produced (an FDFD probe in an eigenfrequency sweep) is left out with a warning (once per key in
    `reported`); a GPU OOM still raises."""
    results = {}
    for physics, feature in _result_features(session, RESULT_KINDS):
        key = f'{physics}.{feature}'
        try:
            results[key] = session.feature_result(physics, feature)
        except Exception as exc:   # text only: one result must not fail the sweep value
            if gpu_memory.is_out_of_memory(exc):
                raise
            if key not in reported:
                reported.add(key)
                emit(LogEvent('warning', f'sweep: result {key} is not available ({_failure_message(exc)}); it is left '
                                         'out of the sweep table (further values not reported)'))
    return results


def _warm_start_state(session, value, previous=None):
    """{'mesh_key', 'fields' {physics: host fields} of every solver with warm_start_fields (heat, potential, magnetic,
    solids), 'value' (this sweep value), 'earlier' ((value, fields) of the previous state on the same meshes, or None:
    the secant extrapolation)} to seed the next sweep value; host copies, so closing the session frees its GPU memory."""
    mesh_key = _mesh_key(session)
    fields = {name: solver.warm_start_fields() for name, solver in session.runner.physics.items()
              if isinstance(solver, WarmStartSolver)}
    earlier = None
    if previous is not None and _same_mesh_key(previous['mesh_key'], mesh_key):
        earlier = (previous['value'], previous['fields'])
    return {'mesh_key': mesh_key, 'fields': fields, 'value': float(value), 'earlier': earlier}


def _extrapolated_fields(latest, warm_state, name, value):
    """Fields of physics `name` linearly extrapolated in the sweep parameter to `value` from the last two states (every
    float array of equal shape in both; others as latest); the factor (value - last) / (last - earlier) is clamped to
    [-1, 1], never past one previous step. latest unchanged without an earlier state."""
    if warm_state['earlier'] is None:
        return latest
    earlier_value, earlier_fields = warm_state['earlier']
    before = earlier_fields.get(name) or {}
    step = warm_state['value'] - earlier_value
    if not before or step == 0.0 or not math.isfinite(step):
        return latest
    factor = min(1.0, max(-1.0, (float(value) - warm_state['value']) / step))
    result = {}
    for key, array in latest.items():
        previous = before.get(key)
        extrapolate = (isinstance(array, np.ndarray) and isinstance(previous, np.ndarray)
                       and array.shape == previous.shape and np.issubdtype(array.dtype, np.inexact))
        result[key] = array + factor * (array - previous) if extrapolate else array
    return result


def _mesh_key(session):
    """{mesh name: (cell widths per axis, window slices) or an element mesh's digest} of every built mesh."""
    return {name: (domain.mesh.digest if domain.grid_kind != 'tensor'
                   else ([np.array(widths) for widths in domain.cell_widths()], domain.physical_window()))
            for name, domain in session.domains.items()}


def _same_mesh_key(previous, current):
    """True when two _mesh_key records describe the same meshes."""
    if previous.keys() != current.keys():
        return False
    for name, entry in current.items():
        before = previous[name]
        if isinstance(entry, str) or isinstance(before, str):
            if entry != before:
                return False
            continue
        (widths_before, window_before), (widths, window) = before, entry
        if not (len(widths_before) == len(widths) and window_before == window
                and all(np.array_equal(old, new) for old, new in zip(widths_before, widths))):
            return False
    return True


def _seed_warm_start(session, warm_state, emit, value):
    """Seed the solvers of a new build (sweep value `value`) from the previous value's _warm_start_state when the meshes
    are unchanged (a solver with warm_start_extrapolate True, the nonlinear solids: extrapolated over the last two
    values), then apply_now (runner order) every coupling reading only seeded slots, so the fixed point restarts from
    the last one."""
    if not _same_mesh_key(warm_state['mesh_key'], _mesh_key(session)):
        emit(LogEvent('info', 'warm start skipped: a mesh differs from the previous sweep value'))
        return
    physics = session.runner.physics
    seeded = set()
    for name, solver_fields in warm_state['fields'].items():
        solver = physics.get(name)
        if isinstance(solver, WarmStartSolver):
            if getattr(solver, 'warm_start_extrapolate', False):
                solver_fields = _extrapolated_fields(solver_fields, warm_state, name, value)
            seeded.update((name, slot) for slot in solver.seed_warm_start(solver_fields))
    for coupling in session.runner.couplings:
        if coupling.source_fields and coupling.source_fields <= seeded:
            coupling.apply_now(physics)


# ---- built-in study types ----
def _run_time_dependent(session, values, emit, control, service):
    """Advance the runner by `duration` in chunks of ~0.1 s wall time with live output and snapshots; flush the
    couplings at the end, also when cancelled."""
    runner = session.runner
    dt_sync = runner.dt_sync
    if not math.isfinite(dt_sync):
        raise ValueError('nothing steps in time (every physics is quasi-static); use a stationary study')
    duration = values['duration']
    output_interval = values['output_interval']
    t_start = runner.t
    t_final = t_start + duration
    half_tick = 0.5 * dt_sync
    tick_start = runner.tick
    recorder = _SnapshotRecorder(session, emit, series_axis='time')
    live = _LiveFeed(session, control, emit, values['live_update_hz'])
    solver_warnings = _SolverWarnings(runner.physics, emit)
    device = session.domain.backend   # synchronize(): the device on the gpu, nothing on the cpu
    emit(LogEvent('info', f'Time-dependent study: {duration:.4g} s in steps of {dt_sync:.4g} s '
                          f'(~{round(duration / dt_sync)} ticks)'))
    next_output = t_start if output_interval > 0.0 else math.inf
    chunk_ticks = 1
    status = 'completed'
    progress_emit = _StageProgress(emit, ['Time stepping']).stage_emit('Time stepping')
    stepping_text = f"Stepping {', '.join(_physics_text(session, name) for name in runner.physics)}"
    wall_start = time.perf_counter()
    try:
        while runner.t < t_final - half_tick:
            control.checkpoint(service)   # before the due snapshot: a cancel here leaves only the final one
            if runner.t >= next_output - half_tick:
                recorder.record()
                next_output = (math.inf if recorder.full
                               else _next_output_time(runner.t, t_start, output_interval, dt_sync))
            chunk_start = time.perf_counter()
            ticks_done = runner.advance(chunk_ticks, min(t_final, next_output))
            device.synchronize()
            chunk_ticks = _next_chunk_ticks(chunk_ticks, ticks_done, time.perf_counter() - chunk_start)
            time.sleep(0)   # hand the GIL to the GUI thread (any positive sleep lasts >= ~1.4 ms on Windows)
            progress_emit(_time_progress(runner, t_start, t_final, tick_start, wall_start, message=stepping_text))
            live.update()
            solver_warnings.check()
    except Cancelled:
        status = 'cancelled'
    runner.flush()
    solver_warnings.check()
    recorder.record(final=True)
    live.update(final=True)
    progress_emit(_time_progress(runner, t_start, t_final, tick_start, wall_start, message=status.capitalize()))
    ticks = runner.tick - tick_start
    if status == 'completed' and ticks == 0:
        emit(LogEvent('warning', f'duration {duration:.4g} s is shorter than half a time step ({dt_sync:.4g} s); '
                                 f'nothing was advanced'))
    return {'status': status, 'ticks': ticks, 't_end': runner.t, 'derived_series': recorder.derived_series()}


def solve_stationary(session, physics_names, values, emit, control, service, live=None,
                     label='Stationary study', inputs=(), log_solves=True):
    """Fixed point at t = 0 of the named physics and the couplings among them (runner prepared): Gauss-Seidel sweeps
    of the loop physics (loop-back transfers relaxed, or Anderson-mixed between sweeps with anderson_depth > 0) until
    their primary fields change by <= tolerance, max_iterations
    or a change that stops decreasing (a non-finite change raises), then one sweep of the sinks (fed by the loop) -> {status, iterations, relative_change,
    fixed_point}; fixed_point = {converged, iterations, relative_change, tolerance}, None without a loop. inputs:
    physics held fixed (not stepped) whose couplings into the named physics apply before every sweep (feedback);
    log_solves False skips the solve_summary lines (repeated solves of a coupled wavelength scan)."""
    runner = session.runner
    subset = frozenset(physics_names)
    unknown = sorted(subset.difference(runner.physics))
    if unknown:
        raise KeyError(f"no physics {', '.join(unknown)}; built: {', '.join(runner.physics)}")
    physics = {name: solver for name, solver in runner.physics.items() if name in subset}
    inputs = frozenset(inputs)
    couplings = [coupling for coupling in runner.couplings
                 if _coupling_physics(coupling) <= subset | inputs and _coupling_physics(coupling) & subset
                 and not _coupling_targets(coupling) & inputs]
    loop, sinks = _stationary_roles(physics, couplings)
    loop_members = frozenset(loop)
    loop_schedule = _coupling_schedule(loop, [coupling for coupling in couplings
                                              if not _writes_only_sinks(coupling, loop_members)])
    sink_schedule = _coupling_schedule(sinks, [coupling for coupling in couplings
                                               if _writes_only_sinks(coupling, loop_members)])
    tolerance = values['tolerance']
    relaxation = values['relaxation']
    max_iterations = values['max_iterations'] if loop else 1
    # sweep 1 solves with loop-back slots from before this solve (a warm-start seed, the previous wavelength): its
    # change can be ~0 at a stale state (slug_flow_graetz sweep: every second value kept the previous T)
    loop_back = _has_loop_back(loop_schedule, loop)
    min_iterations = 2 if loop_back else 1
    mixer = _LoopBackMixer(values['anderson_depth'], relaxation) if loop_back and values['anderson_depth'] else None
    if mixer is not None:
        loop_schedule = _wrap_loop_back(loop_schedule, loop, lambda coupling: _MixedTransfer(coupling, mixer))
    elif relaxation != 1.0:
        loop_schedule = _wrap_loop_back(loop_schedule, loop, lambda coupling: _RelaxedTransfer(coupling, relaxation))
    # the change of sweep k >= 3 follows a step w f (compared / w); an Anderson step estimates the remaining error,
    # so only w < 1 (its first, relaxed step) still scales it
    change_scale = relaxation if mixer is None else min(relaxation, 1.0)
    if live is None:
        live = _LiveFeed(session, control, emit, values['live_update_hz'])
    solver_warnings = _SolverWarnings(physics, emit)
    convergence = _FixedPointMonitor(session, loop_members) if loop else None
    iteration = 0
    relative_change = math.inf
    first_change = math.nan   # relative change of iteration 1: start of the convergence scale
    loop_gain = math.nan   # ratio of the last two relative changes
    growing = 0            # successive iterations whose relative change did not decrease
    status = 'completed'
    # the loop fills [0, loop_share] of the fraction, the single sink pass the rest; the fraction never decreases
    loop_share = len(loop) / (len(loop) + len(sinks)) if loop else 0.0
    fraction = 0.0
    wall_start = time.perf_counter()

    def report(message):
        emit(_count_progress(runner, fraction, iteration, wall_start, message, 'iteration'))

    def announce_loop_solve(name):   # before each solve: an FDFD factorisation can take minutes
        report(f'Iteration {iteration + 1} (at most {max_iterations}): solving {_physics_text(session, name)}')

    def announce_sink_solve(name):
        nonlocal fraction
        fraction = max(fraction, loop_share + (1.0 - loop_share) * sinks.index(name) / len(sinks))
        report(f'Solving {_physics_text(session, name)}')

    try:
        while loop and iteration < max_iterations:
            control.checkpoint(service)
            if mixer is not None:
                mixer.mix(runner.physics)   # the mixed iterate this sweep starts from (re-injected from sweep 3)
            _gauss_seidel_sweep(runner, loop, loop_schedule, announce_loop_solve)
            iteration += 1
            if iteration == 1 and log_solves:
                _log_solve_summaries({name: physics[name] for name in loop}, emit, '')
            # sweeps 1 and 2 see unrelaxed slots (the first loop-back transfer has no previous value)
            previous_change = relative_change
            relative_change = convergence.relative_change() / (change_scale if iteration >= 3 else 1.0)
            loop_gain = relative_change / max(previous_change, _CONVERGENCE_FLOOR)
            # from iteration 4 on both changes carry the same 1/w scaling (and the cold first change is past)
            growing = growing + 1 if iteration >= 4 and relative_change >= previous_change else 0
            if iteration == 1:
                first_change = relative_change
            fraction = max(fraction, loop_share * _fixed_point_fraction(iteration, max_iterations, first_change,
                                                                        relative_change, tolerance))
            report(f'Iteration {iteration}: relative change {relative_change:.2e} (tolerance {tolerance:.1e})')
            live.update()
            solver_warnings.check()
            if not math.isfinite(relative_change):
                raise RuntimeError(f'{label}: the coupling loop fields became non-finite (NaN / inf) in iteration '
                                   f'{iteration}; check the source power and the material data, or damp the loop '
                                   'with relaxation < 1')
            if (relative_change <= tolerance and iteration >= min_iterations) or growing >= _GROWTH_STOP_ITERATIONS:
                break
        if sinks:
            control.checkpoint(service)
            _gauss_seidel_sweep(runner, sinks, sink_schedule, announce_sink_solve)
            iteration = max(iteration, 1)   # without a loop the sink sweep is the single solve
            if log_solves:
                _log_solve_summaries({name: physics[name] for name in sinks}, emit, '')
            fraction = 1.0
            report('Solved')
            live.update()
            solver_warnings.check()
    except Cancelled:
        status = 'cancelled'
    fixed_point = None
    # the loop's own stop rule: a loop-back sweep 1 sees stale slots, so its (small) change alone is no convergence
    converged = relative_change <= tolerance and iteration >= min_iterations
    if convergence is not None:   # a cancelled loop counts as not converged unless it had already converged
        fixed_point = {'converged': converged, 'iterations': iteration,
                       'relative_change': relative_change, 'tolerance': tolerance}
    if convergence is not None and status == 'completed':
        if converged:
            emit(LogEvent('info', f'{label} converged after {iteration} iteration(s) '
                                  f'(relative change {relative_change:.2e})'))
        elif growing >= _GROWTH_STOP_ITERATIONS:
            emit(LogEvent('warning', f'{label} stopped after {iteration} iterations: the relative change did not '
                                     f'decrease for {_GROWTH_STOP_ITERATIONS} iterations in a row (now '
                                     f'{relative_change:.2e} > tolerance {tolerance:.2e}, |loop gain| ~ '
                                     f'{loop_gain:.2f} >= 1): lower the power, or set relaxation < 1 for an '
                                     'oscillating loop; a tolerance below the float32 noise floor stalls too'))
        elif relative_change <= tolerance:   # max_iterations below min_iterations
            emit(LogEvent('warning', f'{label} stopped after {iteration} iteration(s): a loop-back coupling needs at '
                                     f'least {min_iterations} before its relative change means convergence; raise '
                                     'max_iterations'))
        else:
            emit(LogEvent('warning', f'{label} did not converge in {max_iterations} iterations '
                                     f'(relative change {relative_change:.2e} > tolerance {tolerance:.2e})'))
    return {'status': status, 'iterations': iteration, 'relative_change': relative_change, 'fixed_point': fixed_point}


def _run_stationary(session, values, emit, control, service):
    """Steady solve of every physics at t = 0 (solve_stationary over all of them), then the final snapshot."""
    runner = session.runner
    runner.prepare()
    recorder = _SnapshotRecorder(session, emit)
    live = _LiveFeed(session, control, emit, values['live_update_hz'])
    stage_emit = _StageProgress(emit, ['Stationary solve']).stage_emit('Stationary solve')
    outcome = solve_stationary(session, list(runner.physics), values, stage_emit, control, service, live=live)
    recorder.record(final=True)
    live.update(final=True)
    return {'status': outcome['status'], 'ticks': outcome['iterations'], 'tick_unit': 'iteration',
            't_end': runner.t, 'fixed_point': outcome['fixed_point']}


def _run_frequency_domain(session, values, emit, control, service):
    """Background physics to a stationary fixed point, background -> harmonic couplings once, the pulse of the
    time-stepped harmonic physics (FDTD, _run_harmonic_pulse), then per wavelength solve_harmonic of every harmonic
    physics (and select_harmonic_frequency of every pulsed one) with one labelled snapshot each (the last stays live)."""
    runner = session.runner
    wavelengths = resolved_wavelengths(values)
    selected = _selected_physics(session)
    pulsed = [name for name, solver in runner.physics.items() if isinstance(solver, PulsedHarmonicSolver)
              and (not selected or name in selected)]
    harmonic, background = _solver_roles(runner.physics, HarmonicSolver, 'frequency-domain',
                                         'use a stationary study', stepped=pulsed, selected=selected)
    readers = [coupling.name for coupling in runner.couplings if _coupling_sources(coupling) & set(harmonic + pulsed)]
    if readers:
        return _run_coupled_wavelengths(session, harmonic, background, pulsed, readers, wavelengths, values, emit,
                                        control, service)
    stage = _background_stage(
        session, background, harmonic + pulsed, values, emit, control, service,
        ('harmonic', 'such readers take the coupled wavelength scan (never reached)'),
        _frequency_domain_text(wavelengths, harmonic + [f'{name} (FDTD pulse)' for name in pulsed], background),
        [_PULSE_STAGE] * bool(pulsed) + [_WAVELENGTH_STAGE])
    stage.recorder.series_axis = 'wavelength'
    status, n_solved, pulse = stage.status, 0, None
    if status == 'completed' and pulsed:
        status, pulse = _run_harmonic_pulse(session, pulsed, wavelengths, values, stage.live,
                                            stage.stages.stage_emit(_PULSE_STAGE), control, service)
    if status == 'completed':
        n_solved, status = _solve_wavelengths(session, harmonic, wavelengths, stage.recorder, stage.live,
                                              stage.stages.stage_emit(_WAVELENGTH_STAGE), control, service,
                                              pulsed=pulsed)
    stage.live.update(final=True)
    # Only the solved ones: one snapshot each, so a cancel / timeout never lists a wavelength without its snapshot.
    return {'status': status, 'ticks': n_solved, 'tick_unit': 'wavelength', 't_end': runner.t,
            'wavelengths': wavelengths[:n_solved], 'fixed_point': stage.fixed_point, 'harmonic_pulse': pulse,
            'derived_series': stage.recorder.derived_series()}


def _run_coupled_wavelengths(session, harmonic, background, pulsed, readers, wavelengths, values, emit, control,
                             service):
    """frequency_domain with couplings reading a harmonic physics: per wavelength in list order (nothing reset, so
    each continues from the previous one) set omega of every harmonic physics and solve_stationary over every
    physics, then one labelled snapshot; summary fixed_point over all wavelengths + wavelength_fixed_points."""
    runner = session.runner
    if pulsed:
        raise ValueError(f"couplings {', '.join(readers)} read a harmonic physics, so every wavelength is a coupled "
                         f"stationary solve, which the time-stepped {', '.join(pulsed)} cannot join")
    for name in harmonic:
        omega_property = getattr(type(runner.physics[name]), 'omega', None)
        if not isinstance(omega_property, property) or omega_property.fset is None:
            raise ValueError(f"physics '{name}' cannot be solved inside a coupled wavelength scan (no settable omega)")
    runner.prepare()
    recorder = _SnapshotRecorder(session, emit, series_axis='wavelength')
    live = _LiveFeed(session, control, emit, values['live_update_hz'])
    emit(LogEvent('info', _frequency_domain_text(wavelengths, harmonic, background)
                  + f"; coupled through {', '.join(readers)}: each wavelength is a stationary fixed point of every "
                    'physics, continued from the previous wavelength in list order'))
    stage_emit = _StageProgress(emit, [_WAVELENGTH_STAGE]).stage_emit(_WAVELENGTH_STAGE)
    names, n_wavelengths, status, records = list(runner.physics), len(wavelengths), 'completed', []
    wall_start = time.perf_counter()
    for index, wavelength in enumerate(wavelengths):
        try:
            control.checkpoint(service)
        except Cancelled:
            status = 'cancelled'
            break
        label = wavelength_label(wavelength)
        for name in harmonic:
            runner.physics[name].omega = 2.0 * math.pi * C0 / wavelength
        point_emit = _fraction_share(stage_emit, index / n_wavelengths, 1.0 / n_wavelengths,
                                     prefix=f'{label} ({index + 1}/{n_wavelengths})')
        outcome = solve_stationary(session, names, values, point_emit, control, service, live=live,
                                   label=f'{label}: coupled solve', log_solves=index == 0)
        if outcome['status'] != 'completed':   # a cancel inside: this wavelength has no snapshot
            status = outcome['status']
            break
        if outcome['fixed_point'] is not None:
            records.append({'wavelength': float(wavelength), **outcome['fixed_point']})
        recorder.record(final=index == n_wavelengths - 1, label=label, wavelength=wavelength)
        live.update()
        if index == 0:
            wall = time.perf_counter() - wall_start
            emit(LogEvent('info', f'first wavelength {_duration_text(wall)}, all {n_wavelengths} about '
                                  f'{_duration_text(wall * n_wavelengths)}'))
    live.update(final=True)
    n_solved = index + 1 if status == 'completed' else index
    fixed_point = None if not records else {
        'converged': all(record['converged'] for record in records),
        'iterations': max(record['iterations'] for record in records),
        'relative_change': max(record['relative_change'] for record in records), 'tolerance': values['tolerance']}
    return {'status': status, 'ticks': n_solved, 'tick_unit': 'wavelength', 't_end': runner.t,
            'wavelengths': wavelengths[:n_solved], 'fixed_point': fixed_point, 'wavelength_fixed_points': records, 'harmonic_pulse': None,
            'derived_series': recorder.derived_series()}


def _selected_physics(session) -> tuple:
    """StudySpec.solved_physics of the session's spec (() = the protocol decides which physics the study solves)."""
    selection = session.spec.study.solved_physics
    return tuple(selection) if isinstance(selection, list) else ()


def _solver_roles(physics, protocol, study_text, missing_remedy, stepped=(), selected=()):
    """(solved, background) physics names in runner order: solved ones satisfy the solver_hooks `protocol`
    (HarmonicSolver, ModeSolver) and, with `selected` (StudySpec.solved_physics), are listed there; `stepped` ones
    (harmonic FDTD) are neither; ValueError when no physics is solved or stepped, for a selected physics without the
    protocol, or for a background solver that steps in time (finite dt_limit)."""
    hooks = ', '.join(hook_names(protocol))
    if selected:
        unknown = [name for name in selected if name not in physics]
        refused = [name for name in selected if name in physics and name not in stepped
                   and not isinstance(physics[name], protocol)]
        if unknown or refused:
            problem = (f"names {', '.join(unknown)}, which the model does not have" if unknown else
                       f"lists {', '.join(refused)}, whose solver has no {hooks} solve of a {study_text} study")
            raise ValueError(f'StudySpec.solved_physics {problem}')
    solved = [name for name, solver in physics.items() if isinstance(solver, protocol)
              and (not selected or name in selected)]
    background = [name for name in physics if name not in solved and name not in stepped]
    if not solved and not stepped:
        raise ValueError(f"no physics has the {hooks} solve of a {study_text} study; physics: {', '.join(physics)}; "
                         f'{missing_remedy}')
    _refuse_stepping_background(physics, background, f'a {study_text} study', f' and has no {hooks}')
    return solved, background


def _refuse_stepping_background(physics, names, study_phrase, missing=''):
    """ValueError for the first of `names` whose solver steps in time (finite dt_limit): `study_phrase` (e.g. 'a
    thermal noise study') solves its background physics as stationary ones; `missing` ends the physics clause."""
    for name in names:
        dt_limit = physics[name].dt_limit()
        if math.isfinite(dt_limit):
            raise ValueError(f"physics '{name}' steps in time (dt_limit {dt_limit:.4g} s){missing}: {study_phrase} "
                             'solves it as a stationary background physics (use a time-dependent study, or a steady '
                             'mode of that physics)')


def _refuse_solution_readers(couplings, solved, kind, reason, allowed=()):
    """ValueError naming the first coupling that reads one of the `solved` physics (named `kind` in the message),
    `allowed` couplings (the feedback ones) excepted."""
    for coupling in couplings:
        read = sorted(_coupling_sources(coupling).intersection(solved))
        if read and not any(coupling is other for other in allowed):
            raise ValueError(f"coupling '{coupling.name}' reads the {kind} physics {', '.join(read)}: {reason}")


def _feedback_couplings(couplings, solved, background):
    """Runner couplings reading a `solved` physics and writing background physics only (StudyType.solution_feedback);
    a DynamicMaterialUpdate (writes every physics) never is one."""
    solved, background = frozenset(solved), frozenset(background)
    return [coupling for coupling in couplings if _coupling_sources(coupling) & solved
            and _coupling_targets(coupling) and _coupling_targets(coupling) <= background]


def _apply_background_couplings(runner, background):
    """apply_now, in runner order, every coupling reading only background physics that is not part of their fixed
    point (it writes a harmonic physics: thermo-optic, field transfer)."""
    background = frozenset(background)
    for coupling in runner.couplings:
        if _coupling_sources(coupling) <= background and not _coupling_physics(coupling) <= background:
            coupling.apply_now(runner.physics)


@dataclass
class _BackgroundStage:
    """State _background_stage hands its study run: background status / fixed_point, snapshot recorder, live feed and
    the run's _StageProgress."""
    status: str
    fixed_point: dict | None
    recorder: object
    live: object
    stages: object


def _background_stage(session, background, solved, values, emit, control, service, refused, intro, stage_names,
                      feedback=()):
    """Shared start of the studies with a stationary background (frequency_domain, mode_analysis, eigenfrequency,
    thermal_noise): refuse couplings reading the solved physics (refused = (kind, reason); `feedback` couplings
    excepted, which then apply in the background solve with the solved physics as inputs), runner.prepare, recorder,
    live feed, the intro log line, stages ([_BACKGROUND_STAGE] with a background + stage_names), the background fixed
    point, then (completed) the background couplings once."""
    runner = session.runner
    _refuse_solution_readers(runner.couplings, solved, *refused, allowed=feedback)
    runner.prepare()
    recorder = _SnapshotRecorder(session, emit)
    live = _LiveFeed(session, control, emit, values['live_update_hz'])
    emit(LogEvent('info', intro))
    stages = _StageProgress(emit, [_BACKGROUND_STAGE] * bool(background) + list(stage_names))
    status, fixed_point = 'completed', None
    if background:
        outcome = solve_stationary(session, background, values, stages.stage_emit(_BACKGROUND_STAGE), control,
                                   service, live=live, label='Background stationary solve',
                                   inputs=solved if feedback else ())
        status, fixed_point = outcome['status'], outcome['fixed_point']
    if status == 'completed':
        _apply_background_couplings(runner, background)
    return _BackgroundStage(status, fixed_point, recorder, live, stages)


def _solve_static_states(runner, solved, with_loads, emit):
    """solve_static_state(with_loads) of every solved PrestressSolver that is prestressed (stress stiffening: K_sigma
    of that state joins its pencils), after the background stage; one info line each."""
    for name in solved:
        solver = runner.physics[name]
        if isinstance(solver, PrestressSolver) and solver.prestressed:
            emit(LogEvent('info', f'{name}: {solver.solve_static_state(with_loads)}'))


def _clamped_fraction(event) -> float:
    """A Progress event's fraction clamped to [0, 1] (0 when not finite)."""
    return min(max(event.fraction, 0.0), 1.0) if math.isfinite(event.fraction) else 0.0


def _fraction_share(emit, start, share, prefix=''):
    """emit re-scaling each Progress fraction f to start + share f (one part of a stage); a prefix leads its message
    ('<prefix>: <message>')."""
    def scaled(event):
        if isinstance(event, Progress):
            fraction = _clamped_fraction(event)
            message = f'{prefix}: {event.message}' if prefix else event.message
            event = replace(event, fraction=start + share * fraction, message=message)
        emit(event)
    return scaled


def _feedback_rounds(session, background, solved, feedback, values, solve, emit, control, service, live):
    """Rounds of background solve (solved physics as inputs) + background couplings + solve(emit) -> status until the
    primary fields change by <= tolerance (StudyType.solution_feedback; model sub file) -> (status, record)."""
    runner = session.runner
    background_set, solved_set = frozenset(background), frozenset(solved)
    # a DynamicMaterialUpdate writes every physics, so one reading background physics feeds the solved ones too
    loop_back = any(_coupling_sources(coupling) & background_set
                    and (not _coupling_targets(coupling) or _coupling_targets(coupling) & solved_set)
                    for coupling in runner.couplings)
    tolerance, max_rounds = values['tolerance'], values['feedback_rounds']
    names = [coupling.name for coupling in feedback]
    emit(LogEvent('info', f"Feedback of {', '.join(solved)} into {', '.join(background)} through {', '.join(names)}: "
                          + (f'up to {max_rounds} rounds of background and study solves (tolerance {tolerance:.1e})'
                             if loop_back else 'one more background solve (nothing feeds back into the study physics)')))
    monitor = _FixedPointMonitor(session, background_set | solved_set) if loop_back else None
    status, rounds, relative_change, history, growing, converged = 'completed', 0, math.nan, [], 0, False
    try:
        while rounds < max_rounds:
            control.checkpoint(service)
            start, share = rounds / max_rounds, 1.0 / max_rounds
            rounds += 1
            outcome = solve_stationary(session, background, values, _fraction_share(emit, start, 0.5 * share),
                                       control, service, live=live, label=f'Feedback round {rounds}: background solve',
                                       inputs=solved)
            status = outcome['status']
            if status != 'completed' or not loop_back:
                converged = status == 'completed'
                break
            _apply_background_couplings(runner, background)
            status = solve(_fraction_share(emit, start + 0.5 * share, 0.5 * share))
            if status != 'completed':
                break
            previous_change, relative_change = relative_change, monitor.relative_change()
            history.append(relative_change)
            emit(LogEvent('info', f'feedback round {rounds}: largest relative change of the primary fields '
                                  f'{relative_change:.3e} (tolerance {tolerance:.1e})'))
            if not math.isfinite(relative_change):
                raise RuntimeError(f'feedback round {rounds}: the fields became non-finite (NaN / inf); lower the '
                                   'source power or the coupling strength')
            growing = growing + 1 if rounds >= 4 and relative_change >= previous_change else 0
            if relative_change <= tolerance:
                converged = True
                break
            if growing >= _GROWTH_STOP_ITERATIONS:
                break
    except Cancelled:
        status = 'cancelled'
    if status == 'completed' and loop_back:
        if converged:
            emit(LogEvent('info', f'feedback converged after {rounds} round(s) (relative change {relative_change:.2e})'))
        else:
            reason = (f'the change did not decrease for {_GROWTH_STOP_ITERATIONS} rounds in a row'
                      if growing >= _GROWTH_STOP_ITERATIONS else f'feedback_rounds = {max_rounds} reached')
            emit(LogEvent('warning', f'feedback did not converge after {rounds} round(s): {reason} (relative change '
                                     f'{relative_change:.2e} > tolerance {tolerance:.1e}); the last round is stored'))
    return status, {'couplings': names, 'rounds': rounds, 'converged': converged,
                    'relative_change': None if monitor is None else relative_change, 'tolerance': tolerance,
                    'history': history}


def _solve_wavelengths(session, harmonic, wavelengths, recorder, live, emit, control, service, pulsed=()):
    """_solve_wavelength_list between plan_harmonic_frequencies(omegas) and (): physics with that hook (FDFD
    line-block wavelength batches, solver_plan Part II stage 3) learn the solve order first."""
    planners = [session.runner.physics[name] for name in harmonic
                if isinstance(session.runner.physics[name], HarmonicFrequencyPlanner)]
    for solver in planners:
        solver.plan_harmonic_frequencies([2.0 * math.pi * C0 / wavelength for wavelength in wavelengths])
    try:
        return _solve_wavelength_list(session, harmonic, wavelengths, recorder, live, emit, control, service, pulsed)
    finally:
        for solver in planners:
            solver.plan_harmonic_frequencies(())


def _solve_wavelength_list(session, harmonic, wavelengths, recorder, live, emit, control, service, pulsed):
    """solve_harmonic of every harmonic physics (select_harmonic_frequency of every pulsed one) per wavelength in list
    order with one labelled snapshot each (the last always stored) -> (wavelengths solved, status)."""
    runner = session.runner
    solvers = {name: runner.physics[name] for name in harmonic}
    pulsed_solvers = [runner.physics[name] for name in pulsed]
    solver_warnings = _SolverWarnings(solvers, emit)
    n_wavelengths = len(wavelengths)
    wall_start = time.perf_counter()
    for index, wavelength in enumerate(wavelengths):
        try:
            control.checkpoint(service)
        except Cancelled:
            return index, 'cancelled'
        label = wavelength_label(wavelength)
        # Before the solve too: one factorisation can take minutes, and cancel acts only between wavelengths.
        emit(_count_progress(runner, index / n_wavelengths, index, wall_start,
                             f"Solving {', '.join(harmonic + list(pulsed))} at {label} ({index + 1}/{n_wavelengths}); "
                             'stopping waits for this solve', 'wavelength'))
        for solver in pulsed_solvers:
            solver.select_harmonic_frequency(index)
        for solver in solvers.values():
            solver.solve_harmonic(2.0 * math.pi * C0 / wavelength)
        recorder.record(final=index == n_wavelengths - 1, label=label, wavelength=wavelength)
        live.update()
        solver_warnings.check()
        wall = time.perf_counter() - wall_start
        if index == 0:
            _log_solve_summaries(solvers, emit, f'; first wavelength {_duration_text(wall)}, all {n_wavelengths} '
                                                f'about {_duration_text(wall * n_wavelengths)}')
        emit(_count_progress(runner, (index + 1) / n_wavelengths, index + 1, wall_start,
                             f'{label} ({index + 1}/{n_wavelengths})', 'wavelength'))
    return n_wavelengths, 'completed'


def _run_harmonic_pulse(session, pulsed, wavelengths, values, live, emit, control, service):
    """Pulsed stage of a frequency_domain study: attach_harmonic at the study wavelengths to every time-stepped physics,
    step only them (no coupling reads them; background physics stay frozen) until their window field energy fell below
    auto_shutoff x its peak after the sources ended, or max_duration (warning); then normalise -> (status, pulse record
    {steps, simulated_time, stop_reason 'decayed' | 'max_duration' | 'cancelled', final_energy_ratio, source_end})."""
    runner = session.runner
    solvers = {name: runner.physics[name] for name in pulsed}
    frequencies = [C0 / wavelength for wavelength in wavelengths]
    accumulators = {name: solver.attach_harmonic(frequencies) for name, solver in solvers.items()}
    dt = runner.dt_sync
    schedule = [(solver, runner.strides[name], runner.strides[name] * dt) for name, solver in solvers.items()]
    source_end = max(accumulator.source_end for accumulator in accumulators.values())
    auto_shutoff, max_duration = values['auto_shutoff'], values['max_duration']
    t_start, tick_start = runner.t, runner.tick
    solver_warnings = _SolverWarnings(solvers, emit)
    emit(LogEvent('info', f"Harmonic FDTD pulse of {', '.join(pulsed)}: steps of {dt:.4g} s until the field energy "
                          f'falls below {auto_shutoff:.3g} of its peak after the sources end ({source_end:.4g} s), at '
                          f'most {max_duration:.4g} s'))
    peak_energy, energy_ratio, chunk_ticks, stop = 0.0, 0.0, 1, None
    wall_start = time.perf_counter()
    try:
        while stop is None:
            control.checkpoint(service)
            chunk_start = time.perf_counter()
            n_ticks = max(1, min(chunk_ticks, math.ceil((t_start + max_duration - runner.t) / dt)))
            for _ in range(n_ticks):
                tick, t = runner.tick, runner.t
                for solver, stride, span in schedule:
                    if tick % stride == 0:
                        solver.step(t, span)
                runner.t, runner.tick = t + dt, tick + 1
            # one host sync per chunk: the energy reduction
            energy = sum(float(accumulator.field_energy(solvers[name])) for name, accumulator in accumulators.items())
            if not math.isfinite(energy):
                raise RuntimeError(f'harmonic FDTD pulse: the field energy became non-finite at t = {runner.t:.4g} s '
                                   '(unstable time step or material data)')
            chunk_ticks = _next_chunk_ticks(chunk_ticks, n_ticks, time.perf_counter() - chunk_start)
            peak_energy = max(peak_energy, energy)
            energy_ratio = energy / peak_energy if peak_energy > 0.0 else 0.0
            elapsed = runner.t - t_start
            if elapsed >= source_end and energy_ratio <= auto_shutoff:
                stop = 'decayed'
            elif elapsed >= max_duration * (1.0 - 1e-9):
                stop = 'max_duration'
            time.sleep(0)   # hand the GIL to the GUI thread (any positive sleep lasts >= ~1.4 ms on Windows)
            wall = time.perf_counter() - wall_start
            ticks = runner.tick - tick_start
            emit(Progress(min(elapsed / max_duration, 1.0), runner.t, t_start + max_duration, runner.tick, wall,
                          ticks / wall if wall > 0.0 else 0.0,
                          f"Pulse of {', '.join(pulsed)}: t = {elapsed * 1e15:.1f} fs, field energy "
                          f'{energy_ratio:.1e} of its peak'))
            live.update()
            solver_warnings.check()
    except Cancelled:
        stop = 'cancelled'
    pulse = {'steps': runner.tick - tick_start, 'simulated_time': runner.t - t_start, 'stop_reason': stop,
             'final_energy_ratio': energy_ratio, 'source_end': source_end}
    if stop == 'cancelled':
        return 'cancelled', pulse
    text = (f'Harmonic FDTD pulse: {runner.tick - tick_start} steps, {runner.t - t_start:.4g} s, field energy '
            f'{energy_ratio:.2e} of its peak')
    if stop == 'max_duration':
        emit(LogEvent('warning', f'{text}; max_duration {max_duration:.4g} s ended the pulse before the energy fell '
                                 f'below auto_shutoff {auto_shutoff:.3g}, so the harmonic fields miss the rest of the '
                                 'ring-down (raise max_duration)'))
    else:
        emit(LogEvent('info', text))
    for name, accumulator in accumulators.items():
        for message in accumulator.normalize(solvers[name]):
            emit(LogEvent('warning', message))
    return 'completed', pulse


def _frequency_domain_text(wavelengths, harmonic, background):
    """Start-of-study log line: wavelength count and range, harmonic and background physics."""
    nanometres = [wavelength * 1e9 for wavelength in wavelengths]
    span = (f'{nanometres[0]:.4g} nm' if len(nanometres) == 1
            else f'{min(nanometres):.4g}-{max(nanometres):.4g} nm')
    return (f"Frequency-domain study: {len(wavelengths)} wavelength(s), {span}; harmonic physics "
            f"{', '.join(harmonic)}; background physics {', '.join(background) or 'none'}")


def _run_mode_analysis(session, values, emit, control, service):
    """Background physics to a stationary fixed point, background -> mode couplings once, solve_modes() of every mode
    physics, then one labelled snapshot per mode number; each mode physics keeps its mode_index mode live."""
    runner = session.runner
    mode_physics, background = _solver_roles(runner.physics, ModeSolver, 'mode analysis',
                                             'add an em_modes physics', selected=_selected_physics(session))
    stage = _background_stage(
        session, background, mode_physics, values, emit, control, service,
        ('mode', 'a mode analysis feeds no mode solution back into other physics'),
        f"Mode analysis: mode physics {', '.join(mode_physics)}; background physics {', '.join(background) or 'none'}",
        [_MODE_STAGE])
    status, n_modes = stage.status, 0
    if status == 'completed':
        status, n_modes = _solve_mode_snapshots(session, mode_physics, stage.recorder, stage.live,
                                                stage.stages.stage_emit(_MODE_STAGE), control, service)
    stage.live.update(final=True)
    return {'status': status, 'ticks': n_modes, 'tick_unit': 'mode', 't_end': runner.t,
            'fixed_point': stage.fixed_point, 'n_modes': n_modes}


def _solve_mode_snapshots(session, mode_physics, recorder, live, emit, control, service, final_last=True):
    """_numbered_snapshots of the modes: solve_modes() of every mode physics, one snapshot per mode number labelled
    'mode k: n_eff = ...', then mode_index selected again -> (status, largest mode count)."""
    def solve(name, solver):
        count = solver.solve_modes()
        _log_solve_summaries({name: solver}, emit, '')
        return [solver.n_eff_text(number) for number in range(1, count + 1)]

    def reselect(solver, count):
        solver.select_mode(solver.mode_index)
        return solver.mode_index

    return _numbered_snapshots(session, mode_physics, solve, lambda solver, number: solver.select_mode(number),
                               reselect, 'mode', 'mode solve', recorder, live, emit, control, service,
                               final_last=final_last)


def _numbered_snapshots(session, names, solve, select, reselect, noun, tick_unit, recorder, live, emit, control,
                        service, final_last=True):
    """solve(name, solver) of every named physics -> its numbered solutions' label texts (cancel checks between
    solves), then per number k one snapshot labelled '<noun> k: <text>' (several physics: '<name> <text>' joined;
    mirror faces add their parity, '<noun> k [x_lo symmetric]: ...') after select(solver, k) of each physics with a
    k-th solution, the fields of the others NaN (they hold another number); the last final unless final_last is
    False; then reselect(solver, count) -> the number left selected (None: none), whose snapshot is flagged live when
    every physics left the same one (final_last False: a later snapshot holds the live state) -> (status, largest
    count)."""
    runner = session.runner
    solvers = {name: runner.physics[name] for name in names}
    solver_warnings = _SolverWarnings(solvers, emit)
    wall_start = time.perf_counter()
    parity = parity_tag(session.spec, names)
    parity_text = f' [{parity}]' if parity else ''
    texts = {}
    for index, (name, solver) in enumerate(solvers.items()):
        try:
            control.checkpoint(service)
        except Cancelled:
            return 'cancelled', 0
        emit(_count_progress(runner, index / len(solvers), index, wall_start,
                             f'Solving the {noun}s of {_physics_text(session, name)}; stopping waits for this solve',
                             tick_unit))
        texts[name] = solve(name, solver)
        solver_warnings.check()
    count = max((len(entries) for entries in texts.values()), default=0)
    if count == 0:
        emit(LogEvent('warning', f'no {noun} was found; no {noun} snapshot was stored'))
    stored = {}   # number -> its Snapshot (the recorder skips them once full)
    for number in range(1, count + 1):
        parts, blank = [], set()
        for name, solver in solvers.items():
            if number <= len(texts[name]):
                select(solver, number)
                text = texts[name][number - 1]
                parts.append(text if len(solvers) == 1 else f'{name} {text}')
            else:
                blank.add(name)
        n_stored = len(session.snapshots)
        recorder.record(final=final_last and number == count,
                        label=f"{noun} {number}{parity_text}: {', '.join(parts)}", blank_physics=blank)
        if len(session.snapshots) > n_stored:
            stored[number] = session.snapshots[-1]
        live.update()
    live_numbers = {reselect(solver, len(texts[name])) for name, solver in solvers.items()}
    if final_last and len(live_numbers) == 1 and (live_number := live_numbers.pop()) in stored:
        stored[live_number].live = True
    emit(_count_progress(runner, 1.0, len(solvers), wall_start, f'{count} {noun}(s) found', tick_unit))
    return 'completed', count


def _validate_frequency_domain(vctx, values):
    """Error for a wavelength range that does not increase (equal ends only with n_wavelengths = 1) unless an explicit
    wavelength list replaces it."""
    return [] if values['wavelengths'] else wavelength_range_issues(values, ('study',))


_LIVE_UPDATE_PARAM = Param(
    'live_update_hz', 'float', 5.0, label='Live update rate', unit='Hz', min=0.0,
    help='Maximum rate of the live field frames the GUI shows while computing; 0 disables them. Frames never '
         'change the results (the headless api.run sets 0).',
    tip='How often per second the field view refreshes while the study runs; 0 turns it off.\n\n'
        'Does not change the results.')
_RELAXATION_PARAM = Param(
    'relaxation', 'float', 1.0, label='Coupling relaxation', unit='1', min=0.0, min_exclusive=True,
    tip='Fraction of each coupling update that is applied: 1 applies it fully.\n\n'
        'Below 1 calms a loop that oscillates, above 1 speeds up a slow one. Keep 1 unless convergence is poor.',
    help='Relaxation w of the loop-back couplings (those writing a physics solved earlier in the iteration, e.g. '
         'heat_em_thermo_optic in an FDFD + heat loop): each injects previous + w (new - previous); 1 = plain '
         'Picard. For a loop gain g (about the ratio of successive relative changes, negative when the fields '
         'alternate) w ~ 1/(1 - g): w > 1 speeds a slow monotone loop (plain Picard needs ~ln(tolerance)/ln(g) '
         'iterations), w < 1 damps an oscillating one (e.g. a thermo-optic shift past a resonance). Nothing cures a '
         'physical runaway (monotone g > 1). The relative change compared with the tolerance is divided by w. With '
         'anderson_depth > 0 it is the Anderson mixing parameter instead (the change is then divided by w only '
         'when w < 1).')
_WARM_START_PARAM = Param(
    'warm_start', 'bool', False, label='Sweep warm start',
    tip='In a parameter sweep, start each value from the previous solution to save time.\n\n'
        'A geometrically nonlinear solid starts its Newton solve from the displacement predicted from the previous '
        'two values, which cuts its iterations.\n\n'
        'Off makes every value independent and exactly reproducible.',
    help='Parameter sweeps only: start each sweep value from the previous value\'s solution when the mesh is '
         'unchanged (heat T and Q, potential V and its source, magnetic potentials; the couplings reading only these '
         'are applied first), which cuts coupling iterations and CG iterations. drift_diffusion ramps its contact / '
         'gate voltages and generation from the previous value\'s converged state instead of from equilibrium (an I-V '
         'sweep then takes about one continuation step per value). A geometrically nonlinear solid '
         '(solid_mechanics on triangles, solid_mechanics_3d on tetrahedra) starts its Newton solve from the '
         'displacement of the previous value, linearly extrapolated in the swept parameter from the last two values '
         '(at most one previous step ahead), and ramps its load_steps from that state\'s loads and held values; a '
         'start that does not converge falls back to the cold ramp from u = 0. Results then differ from a cold start '
         'within the tolerance; off, every value starts cold (T0, V = 0, u = 0) and is reproducible on its own.')
_PARITY_SECTORS_PARAM = Param(
    'parity_sectors', 'bool', False, label='Parity sectors',
    tip='Splits a mirror-symmetric model into its even and odd parts, solves each on the reduced domain and combines '
        'them.\n\n'
        'Use it for a source off the symmetry plane, or to sort modes and resonances into even and odd ones. Needs '
        'a symmetric or antisymmetric boundary; each part is one more run.',
    help="Symmetry 2.6: run the study once per symmetric / antisymmetric combination of the model's mirror faces "
         '(2^k sector runs for k faces with a bc_<face> symmetric / antisymmetric or a membrane symmetry / '
         'antisymmetry face feature, swapped per sector; each solves the drawn sources plus their images on the drawn '
         'half (quarter, ...) at that parity) and merge them. stationary / frequency_domain / time_dependent / '
         'frequency_response: the response of the drawn sources alone (no images) on the unfolded domain: per stored '
         "snapshot one combined snapshot (its label, 'combined' for a stationary one) holding the mean of the "
         'unfolded sector fields of every linear signed stored field (Re / Im phasors, raw components such as the '
         "FDTD E_z; |E|, intensities, losses and phases are left out), then the unfolded sector snapshots labelled "
         "'<label> [<sector>]'; derived values are combined by their form: integrals / means quadratic in the "
         'fields (power, flux, absorbed power, intensity, w_re^2 + w_im^2 over regions and surfaces symmetric about '
         'every mirror plane) as quadratic totals, mean / 2^k; integrals, means and point values linear in signed '
         'fields (Re / Im phasors, raw components) as the sector mean; feature-sourced values from the combined '
         'feature results below; values of earlier derived values and parameters only, such as ratios, are '
         're-evaluated from the combined ones; anything else (max / min, mode volumes, phases, scalar outputs) is NaN '
         'with a warning; far_field results combined per angle (the mean of the '
         'complex amplitudes) and flux_spectrum results whose line / plane / box reaches every mirror plane '
         '(quadratic totals), membrane field_probe phasors (the sector mean) and port / mode_projection guide modes '
         '(renumbered to the unfolded guide: a free mirror wall gives modes 2k - 1, a held one 2k; amplitudes the '
         'signed sector mean, port powers and fractions recomputed from them) are stored with the run: '
         'Results.sector_result(physics, feature, combined=True). A membrane port incident mode on a guide ending on '
         'a mirror face is driven at twice its amplitude in the sector of its parity only. The combination '
         'assumes a linear model: couplings out of the wave physics (Joule heat, thermo-optic shifts) see each '
         "sector's own field, so their combined values are not the drawn source's (validation warns). "
         'mode_analysis / eigenfrequency: the modes or resonances of every sector, sector by sector, labelled with '
         "their parity ('mode k [x_lo symmetric]: ...'); derived values NaN. Every run: per-sector derived values and "
         'result digests in summary parity_sectors (labels, derived, results, combined_results, snapshot_sectors), '
         'the feature results '
         'per sector by Results.sector_result; the live state (Results.field, feature results) is the last sector, '
         "the one with the model's own mirror signs. With a parameter sweep every sweep value runs its sectors (cold "
         'starts): sweep derived = the combined values per value, sweep results = the last sector\'s, sweep '
         'sector_derived the per-sector values, parity_sectors the last value\'s. Needs a mirror face.')


def _parity_sector_issues(vctx, values):
    """parity_sectors: an error without a mirror face, a warning with enabled couplings (the combination assumes a
    linear model), else an info naming the sector runs (per sweep value with a sweep)."""
    if not values.get('parity_sectors'):
        return []
    faces = sorted(mirror_faces(vctx.spec))
    if not faces:
        return [Issue('error', ('study',), 'parity_sectors needs a mirror face: set a bc_<face> of the wave physics to '
                                           'symmetric or antisymmetric, or give a membrane a symmetry / antisymmetry '
                                           'face (the drawn half is then solved at every parity), or switch '
                                           'parity_sectors off')]
    issues = []
    couplings = [coupling.name for coupling in vctx.spec.couplings if coupling.enabled]
    if couplings:
        issues.append(Issue('warning', ('study',), f"parity_sectors combines linear responses, but the enabled "
                                                   f"couplings {', '.join(couplings)} see each sector's own field: "
                                                   'combined values of physics they drive (e.g. Joule heat, a '
                                                   "thermo-optic shift) are not the drawn source's"))
    per_value = ' per sweep value' if vctx.spec.study.sweep is not None else ''
    issues.append(Issue('info', ('study',), f"parity_sectors: {2 ** len(faces)} sector runs{per_value} over the "
                                            f"mirror faces {', '.join(faces)} ({'; '.join(sector_specs(vctx.spec))})"))
    return issues


def _with_parity_sectors(validate=None):
    """StudyType.validate of a study offering parity_sectors: `validate` (if any) plus _parity_sector_issues."""
    def checked(vctx, values):
        return (list(validate(vctx, values)) if validate is not None else []) + _parity_sector_issues(vctx, values)
    return checked


_FEEDBACK_ROUNDS_PARAM = Param(
    'feedback_rounds', 'int', 20, label='Feedback rounds', min=1,
    help='Studies with solution feedback only, used when an enabled coupling reads a physics the study solves itself '
         'and writes a background physics (e.g. em_heat_joule from em_fdfd_nonlinear into heat: two-photon '
         'absorption heating): rounds of background solve (the feedback couplings applied before each of its sweeps) '
         'and study solve at most, until every primary field of these physics changes between rounds by at most the '
         'tolerance (the same relative measure as the background fixed point); a warning when not reached or when '
         'the change stops decreasing for 3 rounds (from round 4), the last round is kept. Without a coupling back '
         'into the study physics one background solve follows the study solve. No relaxation acts between rounds. '
         'Typically 5-20.',
    tip='Largest number of rounds in which a coupled background physics (for example heat warmed by the light) and '
        'the main solve are repeated until they agree.\n\nUsed only when the main solution feeds a background physics.')


def _log_eigen_solve(name, eigen_solve, emit):
    """Info line of an eigen solve's summary() and a warning per info['warnings'] entry of physics `name`."""
    emit(LogEvent('info', f'{name}: {eigen_solve.summary()}'))
    for warning in eigen_solve.info['warnings']:
        emit(LogEvent('warning', f"physics '{name}': {warning}"))


def _reselect_first_resonance(solver, count):
    """_numbered_snapshots reselect of resonance studies: resonance 1 live again (None without any)."""
    if count:
        solver.select_resonance(1)
        return 1
    return None


def _scipy_issues(study_name, what_for) -> list:
    """[error at ('study',)] when scipy, which the study needs for what_for (e.g. 'host sparse factorisation'), is not
    installed; else []."""
    import importlib.util
    if importlib.util.find_spec('scipy') is not None:
        return []
    return [Issue('error', ('study',), f'the {study_name} study needs scipy ({what_for}), which is not installed: pip '
                                       'install scipy')]


def _frequency_list(values, ascending=False) -> list:
    """Study frequencies [Hz] in solve order: the explicit list (sorted with ascending), else n_frequencies from
    frequency_start to frequency_stop, linear or (log_spacing) geometric; 1 = the arithmetic / geometric mean."""
    if values['frequencies']:
        frequencies = [float(frequency) for frequency in values['frequencies']]
        return sorted(frequencies) if ascending else frequencies
    start, stop, count = values['frequency_start'], values['frequency_stop'], values['n_frequencies']
    if count == 1:
        return [math.sqrt(start * stop) if values['log_spacing'] else 0.5 * (start + stop)]
    spaced = np.geomspace(start, stop, count) if values['log_spacing'] else np.linspace(start, stop, count)
    return [float(frequency) for frequency in spaced]


def _frequency_range_issues(values) -> list:
    """[error] for a frequency range that does not increase (more than one frequency, no explicit list)."""
    if values['frequencies'] or values['frequency_stop'] > values['frequency_start'] or values['n_frequencies'] <= 1:
        return []
    return [Issue('error', ('study',), f"frequency_stop {values['frequency_stop']:.6g} Hz must exceed frequency_start "
                                       f"{values['frequency_start']:.6g} Hz")]


def _fixed_point_params(scope, max_iterations_tip, tolerance_tip):
    """max_iterations, tolerance, relaxation, anderson_depth, warm_start, live_update_hz of a fixed-point study; scope
    prefixes the max_iterations, tolerance and anderson_depth helps."""
    return (
        Param('max_iterations', 'int', 50, label='Maximum coupling iterations', min=1, tip=max_iterations_tip,
              help=f'{scope}Gauss-Seidel iterations over the physics in a coupling loop (each solved in model order, '
                   'each coupling applied once its sources are solved); physics only fed by the loop (sinks: they feed no '
                   'coupling back) are solved once afterwards, and a model without a loop is a single pass. Typically 10-50; a warning is logged when the tolerance is not reached. The loop '
                   'stops early with a warning when the change does not decrease for 3 iterations in a row (from '
                   'iteration 4) and fails the run when the fields turn non-finite.'),
        Param('tolerance', 'float', 1e-4, label='Relative tolerance', unit='1', min=0.0, min_exclusive=True,
              tip=tolerance_tip,
              help=f'{scope}Converged when every primary field of the loop physics changes between iterations by at '
                   'most this fraction of its range max|f| (the heat temperature T is compared as its rise T - T0). '
                   'The remaining error is about change * g / (1 - g) for a loop gain g (ratio of successive '
                   'changes): at g = 0.9 a change of 1e-4 means an error of ~1e-3. Anderson steps (anderson_depth) '
                   'approach the remaining error itself, so there it is about the change. Typically 1e-4.'),
        _RELAXATION_PARAM,
        Param('anderson_depth', 'int', 0, label='Coupling acceleration depth', min=0,
              tip='Number of earlier rounds combined to speed up a slowly converging loop; 0 switches it off.\n\n'
                  'Use it when the coupled physics creep toward their steady state over many rounds. It can also '
                  'settle on an unstable or unphysical state (a runaway, the unstable branch of a bistable loop) '
                  'that the plain rounds would expose, so keep 0 there.',
              help=f'{scope}Anderson acceleration of the loop-back couplings (Walker-Ni, depth m = this value, mixing '
                   'parameter = relaxation): each loop-back injects its new value during the sweep (later physics of '
                   'the sweep see it), then before the next sweep every loop-back value is replaced by the '
                   'least-squares combination of the last m iterates and their changes (each value scaled by its '
                   'largest magnitude; host float64 copies, ~3 m + 3 per value); 0 = plain relaxation. A monotone '
                   'loop of gain g needs ~ln(tolerance)/ln(g) Picard iterations (~120 at g = 0.9 and 1e-6), far '
                   'fewer accelerated. Like any secant method it can converge to a fixed point Picard cannot reach: '
                   'an unstable branch (bistable thermo-optic loops) or, past a runaway (monotone g > 1), an '
                   'unphysical one, so it is opt-in. Typically 3-5.'),
        _WARM_START_PARAM,
        _LIVE_UPDATE_PARAM,
    )


_TIME_DEPENDENT_PARAMS = (
    Param('duration', 'float', 100e-15, label='Duration', unit='s', min=0.0, min_exclusive=True,
          tip='Simulated time span.\n\n'
              'Light pulses need about 100 to 1000 femtoseconds (1e-13 to 1e-12 s); transient heat needs milliseconds '
              'to seconds.',
          help='Simulated time span in s; the run stops at the tick nearest the duration (less than half a time '
               'step advances nothing). EM runs typically take 100-1000 fs: the pulse (2*t0 = '
               '9/(pi*bandwidth*f0); ricker 8/f0) plus the transit time and the ring-down (validation warns when '
               'the duration ends before the longest pulse, plus the domain diagonal * n_max / C0 with a flux, DFT, '
               'mode or far-field monitor); transient heat takes ms to s.'),
    Param('output_interval', 'float', 0.0, label='Output interval', unit='s', min=0.0,
          tip='Simulated time between stored field snapshots; 0 stores only the final state.',
          help="Time in s between snapshots of the stored fields (study stored_fields, default each physics' "
               'primary field), from t = 0; 0 stores only the final state. Intermediate snapshots stop once they '
               'exceed 1 GB; the final state is always stored. Snapshots hold the native window cells (graded '
               'included) up to 250,000 cells, else nearest-cell values on square pixels of max(finest cell, '
               'sqrt(window area / 250,000)) (native_shape set): fine features can then be skipped.'),
    _LIVE_UPDATE_PARAM,
    _PARITY_SECTORS_PARAM,
)
def _background_fixed_point_params(before, gui_before, note=''):
    """_fixed_point_params of a study's background fixed point over the background physics only, run before `before`
    (help) / `gui_before` (tips), e.g. 'the mode solve' / 'the modes are computed'; note extends the unused-without
    remark."""
    return _fixed_point_params(
        f'Background fixed point, before {before} over the background physics only (unused without background '
        f'physics{note}): ',
        max_iterations_tip=f'Largest number of rounds for the steady heat or current solve before {gui_before}, when '
                           'those physics are coupled.\n\n10 to 50 is typical.',
        tolerance_tip=f'Convergence of the steady heat or current solve before {gui_before}: it stops when no field '
                      'changes between rounds by more than this fraction.\n\n1e-4 is typical.')


_STATIONARY_PARAMS = _fixed_point_params(
    '',
    max_iterations_tip='Largest number of rounds of solving the coupled physics in turn.\n\n'
                       "10 to 50 is typical. A warning is logged when 'Relative tolerance' is not reached.",
    tolerance_tip='The coupled physics count as converged when no field changes between rounds by more than this '
                  'fraction of its largest value.\n\n1e-4 is typical.') + (_PARITY_SECTORS_PARAM,)
_FREQUENCY_DOMAIN_PARAMS = (
    Param('wavelength_start', 'float', 1.5e-6, label='Start wavelength', unit='m', min=0.0, min_exclusive=True,
          tip='First vacuum wavelength of the range, in metres (1.5e-6 for 1500 nm).\n\n'
              "The study wavelengths replace the 'Wavelength' set in the wave physics.",
          help='First vacuum wavelength in m of the range solved when wavelengths is null or empty: n_wavelengths '
               'wavelengths spaced linearly in wavelength from wavelength_start to wavelength_stop, solved in that '
               'order. The study wavelengths override the wavelength setting of every FDFD physics. Typically 0.4e-6 '
               'to 2e-6.'),
    Param('wavelength_stop', 'float', 1.6e-6, label='Stop wavelength', unit='m', min=0.0, min_exclusive=True,
          tip="Last vacuum wavelength of the range, in metres (1.6e-6 for 1600 nm). Must be larger than 'Start "
              "wavelength'.",
          help='Last vacuum wavelength in m of the range; must be greater than wavelength_start (equal is allowed '
               'with n_wavelengths = 1). Ignored when wavelengths lists any wavelength.'),
    Param('n_wavelengths', 'int', 11, label='Number of wavelengths', min=1,
          tip='Wavelengths evenly spaced from start to stop; 1 solves only the middle one.\n\n'
              'Each wavelength costs one full solve.',
          help='Wavelengths of the range, spaced linearly in wavelength (not in frequency); 1 solves only the centre '
               '(wavelength_start + wavelength_stop) / 2. Each wavelength costs one assembly, factorisation and '
               'solve per FDFD physics (harmonic FDTD: one DFT update per sample and 8 / 24 B per window cell of GPU memory). '
               'Ignored when wavelengths lists any wavelength.'),
    Param('wavelengths', 'float_list', None, label='Wavelength list', unit='m', min=0.0, min_exclusive=True,
          optional=True,
          tip='Vacuum wavelengths in metres, solved in this order.\n\n'
              "When filled, it replaces 'Start wavelength', 'Stop wavelength' and 'Number of wavelengths'.",
          help='Explicit vacuum wavelengths in m (each > 0), solved in list order; a non-empty list overrides '
               'wavelength_start, wavelength_stop and n_wavelengths. null or [] = use the range.'),
    Param('auto_shutoff', 'float', 1e-4, label='Pulse shutoff level', unit='1', min=0.0, min_exclusive=True,
          max=1.0,
          tip='Time-domain light only: the pulse run stops once the field energy has fallen to this fraction '
              'of its peak after the source pulse ended.\n\n'
              'Smaller is more accurate and takes longer; 1e-4 is typical, 1e-6 for precise comparisons.',
          help='em_fdtd_* physics only (harmonic FDTD): the pulse ends once t is past the source pulse (2 t0) and the '
               'EM energy 0.5 sum(eps0 eps E^2 + mu0 mu H^2) dA over the physical window (checked about 10 times per '
               'wall second) is at most auto_shutoff times its running peak, or at max_duration. The truncated '
               'ring-down errs the harmonic fields by roughly sqrt(auto_shutoff) times the ring-down share of the '
               'response (more for slowly decaying resonances); 1e-4 typical, 1e-6 for percent-level comparisons.'),
    Param('max_duration', 'float', 2e-12, label='Maximum pulse duration', unit='s', min=0.0, min_exclusive=True,
          tip='Time-domain light only: the longest simulated time of the pulse run; a warning is logged when it, '
              "not 'Pulse shutoff level', ends the run.",
          help='em_fdtd_* physics only (harmonic FDTD): upper bound in s of the simulated pulse time; when it ends the '
               'run before the energy fell below auto_shutoff a warning is logged (high-Q resonances: raise it). '
               'Typical 1e-12 to 1e-11 s.'),
    *_background_fixed_point_params(
        'the wavelength loop', 'the wavelengths',
        note='; a coupled wavelength scan: the fixed point of every physics at each wavelength'),
    _PARITY_SECTORS_PARAM,
)
_MODE_ANALYSIS_PARAMS = (*_background_fixed_point_params('the mode solve', 'the modes are computed'),
                         _PARITY_SECTORS_PARAM)


def _time_dependent_counts(values):
    """Run-time estimate counts (StudyType.time_counts): the simulated duration."""
    return {'duration': float(values['duration'])}


def _fixed_point_counts(values):
    """... of a fixed-point study: its iteration limit."""
    return {'fixed_point_limit': int(values['max_iterations'])}


def _frequency_domain_counts(values):
    """... of frequency_domain: wavelengths, the pulse limit of harmonic FDTD, the background iteration limit."""
    wavelengths = values.get('wavelengths') or []
    return {'harmonic_points': len(wavelengths) if wavelengths else int(values['n_wavelengths']),
            'max_duration': float(values['max_duration']), 'fixed_point_limit': int(values['max_iterations'])}


def _selftest_cases():
    """The model-layer and fem SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from ..fem.selftests import FEM_SELFTEST_CASES   # mesh plan T1+: the fem package's cases
    from .selftests_3d_core import CORE_3D_SELFTEST_CASES
    from .selftests_adaptation import ADAPTATION_SELFTEST_CASES   # mesh plan A1 (S7.7 refinement levels)
    from .selftests_analytic import ANALYTIC_SELFTEST_CASES
    from .selftests_conformal import CONFORMAL_SELFTEST_CASES
    from .selftests_conformal_cuts import CONFORMAL_CUT_SELFTEST_CASES
    from .selftests_conformal_polar import CONFORMAL_POLAR_SELFTEST_CASES
    from .selftests_estimation import ESTIMATION_SELFTEST_CASES   # StudySpec.estimation (parameter estimation)
    from .selftests_geometry_3d import GEOMETRY_3D_SELFTEST_CASES
    from .selftests_geometry_array import GEOMETRY_ARRAY_SELFTEST_CASES
    from .selftests_interface import INTERFACE_SELFTEST_CASES
    from .selftests_line import LINE_SELFTEST_CASES
    from .selftests_mesh import MESH_SELFTEST_CASES
    from .selftests_per_mesh import PER_MESH_SELFTEST_CASES   # mesh plan P3 (Session / results per mesh)
    from .selftests_subpixel import SUBPIXEL_SELFTEST_CASES
    from .selftests_sweep import SWEEP_SELFTEST_CASES
    from .selftests_symmetry_check import SYMMETRY_CHECK_SELFTEST_CASES
    from .selftests_time import TIME_SELFTEST_CASES
    return (*CORE_3D_SELFTEST_CASES, *GEOMETRY_3D_SELFTEST_CASES, *GEOMETRY_ARRAY_SELFTEST_CASES,
            *ANALYTIC_SELFTEST_CASES, *MESH_SELFTEST_CASES,
            *SUBPIXEL_SELFTEST_CASES, *SYMMETRY_CHECK_SELFTEST_CASES, *CONFORMAL_SELFTEST_CASES,
            *CONFORMAL_CUT_SELFTEST_CASES, *CONFORMAL_POLAR_SELFTEST_CASES, *SWEEP_SELFTEST_CASES,
            *INTERFACE_SELFTEST_CASES, *LINE_SELFTEST_CASES, *TIME_SELFTEST_CASES, *FEM_SELFTEST_CASES,
            *ADAPTATION_SELFTEST_CASES, *PER_MESH_SELFTEST_CASES, *ESTIMATION_SELFTEST_CASES)


def register(registry):
    """Register the built-in StudyTypes 'time_dependent', 'stationary', 'frequency_domain', 'mode_analysis',
    'eigenfrequency' (study_eigenfrequency), 'optimization' (study_optimization), 'thermal_noise'
    (study_thermal_noise), 'nonlinear_frequency_domain' (study_nonlinear), 'optomechanics' (study_optomechanics) and
    'frequency_response' (study_frequency_response), then the model-layer selftest loader."""
    registry.register_study(StudyType(
        'time_dependent', 'Time dependent',
        summary='Steps every physics forward in time and stores field snapshots along the way. Needs at least one '
                'physics that changes in time, such as time-domain light or transient heat.',
        description='Advances every physics in ticks of dt_sync, the smallest dt_limit of the physics (EM: its '
                    'Courant step); a physics with every = n steps on every n-th tick over n * dt_sync. Couplings '
                    'are applied after every tick at their own every and flushed at the end of the run, also when '
                    'it is cancelled. Needs at least one physics that steps in time; snapshots at the output '
                    'interval plus the final state.',
        params=_TIME_DEPENDENT_PARAMS, requires_time_stepping=True, run=_run_time_dependent,
        validate=_with_parity_sectors(), time_counts=_time_dependent_counts))
    registry.register_study(StudyType(
        'stationary', 'Stationary',
        summary='Solves the steady state of every physics, repeating coupled physics until they agree. Time-domain '
                'wave physics cannot use it.',
        description='Steady solve of every physics at t = 0; with couplings, solving and applying every coupling '
                    'is repeated until the primary fields reach a fixed point (tolerance) or max_iterations. '
                    'Stores the final state only. Heat is always solved as a steady state (time_mode transient is '
                    'ignored). EM FDFD (em_fdfd_ez, em_fdfd_hz, em_fdfd_bor, em_fdfd_conical), em_eme and em_eme_3d '
                    'are solved at their '
                    'wavelength setting; with heat (em_heat_joule, heat_em_thermo_optic) every iteration re-solves '
                    'them with the permittivity of the latest temperature and hands their absorbed power to heat. EM FDTD (em_fdtd_ez, em_fdtd_hz, '
                    'em_fdtd_3d, em_fdtd_bor) does not support it: use time_dependent; neither does em_modes: use '
                    'mode_analysis.',
        params=_STATIONARY_PARAMS, run=_run_stationary, validate=_with_parity_sectors(), supports_adaptation=True,
        time_counts=_fixed_point_counts))
    registry.register_study(StudyType(
        'frequency_domain', 'Frequency domain',
        summary='Solves the frequency-domain wave physics at a list of wavelengths. Any heat or current physics is '
                'first solved to a steady state, which every wavelength then shares; when the absorbed light heats '
                'the model back, each wavelength is solved together with the heat instead, continuing from the '
                'previous wavelength.',
        description='Time-harmonic solves at a list of vacuum wavelengths (exp(+j omega t), omega = 2 pi c / '
                    'wavelength). Harmonic physics (types listing frequency_domain among their supported studies: '
                    'em_fdfd_ez, em_fdfd_hz, em_fdfd_bor, em_fdfd_conical (at a fixed k_z), em_modes, em_eme, '
                    'em_eme_3d) are solved once per wavelength '
                    'in list order; the study '
                    'wavelengths override the wavelength setting of an FDFD or em_modes physics (em_modes: n_modes '
                    'modes per wavelength, the mode of its mode_number setting in the snapshot, mode_dispersion '
                    'results versus '
                    'wavelength). Time-domain types (em_fdtd_ez, em_fdtd_hz, em_fdtd_3d, em_fdtd_bor: harmonic FDTD) '
                    'instead run one pulse '
                    'after the background solve, accumulating a running DFT of the whole physical window at every '
                    'study wavelength until the field energy fell below auto_shutoff of its peak (or max_duration), '
                    'divided by the spectrum of their shared source waveform (every source of one such physics: the '
                    'same gaussian or ricker waveform, wavelength and bandwidth): the phasor response to a unit '
                    'source at phase 0, as the FDFD gives it (output fields: <component>_re / _im / _abs / _phase, '
                    'e_abs_harmonic, e_intensity_harmonic and absorbed_power_harmonic (time average); ez / hz: the '
                    'out-of-plane component, stored by default ez_abs / hz_abs; em_fdtd_3d: ex, ey, ez at the cell '
                    'centres, stored by default e_abs_harmonic; em_fdtd_bor: its own set, see its description); '
                    'their monitors keep their own unnormalised pulse spectra. '
                    'Memory: 8 (ez) / 24 (hz, 3D) bytes per window cell and wavelength. '
                    'Every other physics must support the '
                    'stationary study and runs as a background physics, solved first (heat is always steady here): '
                    'the background physics and the couplings among them iterate to a stationary fixed point '
                    '(max_iterations, tolerance), then every coupling from background to harmonic physics '
                    '(heat_em_thermo_optic, field_transfer, material properties depending on background fields such '
                    'as T) is applied once, so every wavelength sees the same background state. Coupled wavelength '
                    'scan: when an enabled coupling or material property reads a harmonic physics (em_heat_joule or '
                    'field_transfer from an FDFD physics, a property depending on its fields; every physics must then '
                    'support the stationary study, so no em_modes or time-domain type), each wavelength is instead a '
                    'stationary fixed point of every physics (max_iterations, tolerance, relaxation as the stationary '
                    'study), warm-started from the previous wavelength in list order: an ascending list scans blue '
                    'to red (the thermal triangle of a red-shifting ring), a descending one red to blue (hysteresis); '
                    'one log line per wavelength with its iteration count, a warning when one does not converge (the '
                    'scan continues), summary fixed_point over all wavelengths (converged = all, iterations and '
                    'relative_change the largest) and wavelength_fixed_points [{wavelength, converged, iterations, '
                    'relative_change, tolerance}]; a cancel inside a wavelength leaves its monitors holding the last '
                    'iterate. Needs at least one harmonic physics. Stores one snapshot per wavelength '
                    'of the stored fields (default each primary field, plus <field>_im next to an FDFD <field>_re, so '
                    '|field|^2 per wavelength can be rebuilt; native window cells up to 250,000 cells, else '
                    'nearest-cell resampled), labelled "lambda = <nm> nm"; the last '
                    'wavelength stays the live state (Results.field), and feature results (probes, flux spectra) '
                    'hold every solved wavelength (sorted by wavelength; snapshot labels keep the solve order).',
        params=_FREQUENCY_DOMAIN_PARAMS, background_study='stationary', run=_run_frequency_domain,
        validate=_with_parity_sectors(_validate_frequency_domain), coupled_points=True, supports_adaptation=True,
        feedback_alternative='use a stationary study for one wavelength, a parameter sweep over the wavelength for '
                             'several', time_counts=_frequency_domain_counts,
        solver_protocols=(HarmonicSolver, PulsedHarmonicSolver)))
    registry.register_study(StudyType(
        'mode_analysis', 'Mode analysis',
        summary='Computes the guided modes of a waveguide cross-section at one wavelength. Any heat or current physics '
                'is first solved to a steady state, so the modes see, for example, a heated waveguide.',
        description='Cross-section eigenmode solve: every physics listing mode_analysis among its supported studies '
                    '(em_modes) computes its n_modes modes at its wavelength setting (the xy plane is the waveguide '
                    'cross-section, propagation along z). Every other physics must support the stationary study and '
                    'runs as a background physics, solved first (heat is always steady here): the background physics '
                    'and the couplings among them iterate to a stationary fixed point (max_iterations, tolerance), '
                    'then every coupling from background to mode physics (heat_em_thermo_optic, field_transfer, '
                    'material properties depending on background fields such as T) is applied once before the mode '
                    'solve. Couplings or material properties that read a mode physics are a validation error (mode '
                    'solutions are not fed back). Stores one snapshot per mode number k of the stored fields (default '
                    'each primary field: em_modes e_intensity), labelled "mode k: n_eff = <value>" (a complex value '
                    'when lossy; with several em_modes physics their names and values joined; mirror faces add the '
                    'parity, "mode k [x_lo symmetric]: ..."), in the em_modes rank '
                    'order (descending Re(beta^2)); afterwards each em_modes physics shows the mode of its mode_number setting (Results.field). ticks = the '
                    'number of mode snapshots. For n_eff versus a parameter use a parameter sweep with a derived value '
                    'of the mode_table result; for n_eff versus wavelength a frequency_domain study.',
        params=_MODE_ANALYSIS_PARAMS, background_study='stationary', run=_run_mode_analysis,
        validate=_with_parity_sectors(), supports_adaptation=True, time_counts=_fixed_point_counts,
        solver_protocols=(ModeSolver,)))
    from .study_eigenfrequency import register as register_eigenfrequency   # it imports this module's helpers
    register_eigenfrequency(registry)
    from .study_optimization import register as register_optimization   # likewise
    register_optimization(registry)
    from .study_thermal_noise import register as register_thermal_noise   # likewise
    register_thermal_noise(registry)
    from .study_nonlinear import register as register_nonlinear   # likewise
    register_nonlinear(registry)
    from .study_optomechanics import register as register_optomechanics   # likewise
    register_optomechanics(registry)
    from .study_frequency_response import register as register_frequency_response   # likewise
    register_frequency_response(registry)
    registry.register_selftests(_selftest_cases)


# ---- helpers ----
class _SnapshotRecorder:
    """Appends Snapshots of the stored fields to session.snapshots (native window cells up to _SNAPSHOT_PIXELS cells,
    graded or not, else nearest-cell resampled; 3D: the mid-height z cut plus each StudySpec.stored_cuts plane in
    Snapshot.cuts, and the profile of every 3D-segment line curve in Snapshot.segments); intermediate ones stop above
    1 GB. With a series_axis ('time' | 'wavelength') every record also
    evaluates the source-free derived values on the live state (derived_series)."""

    def __init__(self, session, emit, series_axis=None):
        self._session = session
        self._emit = emit
        self.series_axis = series_axis
        self._series_x = []
        self._series = {}   # source-free derived values whose referenced derived values are in the series too
        for derived in session.spec.derived_values:
            if derived.source is None and earlier_derived_names(session.spec, derived) <= set(self._series):
                self._series[derived.name] = []
        self._series_failed = set()   # derived value names whose evaluation failed (warned once)
        self._cuts = [(axis, position) for axis, position in stored_cut_planes(session.spec, session.parameters)]
        available = {(physics, info.name) for physics, info in session.available_fields()}
        self._fields = []   # (key, physics, component, native per cut: [default cut, *stored cuts])
        for key in _stored_field_keys(session):
            physics, _, component = key.partition('.')
            if (physics, component) not in available:
                raise ValueError(f"stored field '{key}' is not an output field of this model; available: "
                                 f"{sorted(f'{name}.{field_name}' for name, field_name in available)}")
            # native window cells of the physics' own mesh up to _SNAPSHOT_PIXELS per cut (3D: the mid-height z cut;
            # a triangle mesh: its elements, always a raster)
            window_shape = session.window_shape(physics)
            native = [math.prod(window_shape[:2]) <= _SNAPSHOT_PIXELS]
            for axis, _ in self._cuts:
                in_plane = [count for name, count in zip('xyz', window_shape) if name != axis]
                native.append(in_plane[0] * in_plane[1] <= _SNAPSHOT_PIXELS)
            self._fields.append((key, physics, component, native))
        self._segments = _segment_curves(session, available, emit)
        self._last_reads = {}   # (key, cut index) | ('segment', name) -> the last FieldData / LineProfile read
        self._stored_bytes = _snapshot_bytes(session.snapshots)
        self.full = False

    def record(self, final=False, label=None, blank_physics=(), wavelength=None):
        """Store the current state, labelled e.g. 'lambda = 1550 nm' (skipped once full unless final); the fields of
        blank_physics are NaN (they hold a solution other than the label's, e.g. fewer modes). The derived series
        takes a point at session.time or `wavelength` [m] (series_axis) even when full."""
        session = self._session
        if self.series_axis is not None:
            self._record_series(session.time if self.series_axis == 'time' else wavelength)
        if self.full and not final:
            return
        planes = [None] + list(self._cuts)
        stored = [{} for _ in planes]   # per cut: 0 the default cut, then the stored cuts
        computed_segments, pending = {}, list(self._segments)
        for key, physics, component, native in self._fields:
            with session.shared_plane_reads():   # 3D without device_plane: every cut of this field from one volume
                for cut_index, plane in enumerate(planes):
                    stored[cut_index][key] = self._field_data(key, physics, component, native, plane, cut_index,
                                                              blank_physics)
                for segment in [segment for segment in pending if segment[1:3] == (physics, component)]:
                    computed_segments[segment[0]] = self._segment(segment, blank_physics)
                    pending.remove(segment)
        for segment in pending:
            computed_segments[segment[0]] = self._segment(segment, blank_physics)
        fields, cuts = stored[0], stored[1:]
        segments = {segment[0]: computed_segments[segment[0]] for segment in self._segments}
        session.snapshots.append(Snapshot(time=float(session.time), fields=fields, label=label, cuts=cuts,
                                          segments=segments))
        self._stored_bytes += sum(_field_bytes(field_data) for stored in [fields, *cuts]
                                  for field_data in stored.values())
        if not (final or self.full) and self._stored_bytes > _SNAPSHOT_BYTE_LIMIT:
            self.full = True
            self._emit(LogEvent('warning', f'stored snapshots exceed {_SNAPSHOT_BYTE_LIMIT / 2**30:.0f} GB; '
                                           f'only the final state will be stored'))

    def derived_series(self):
        """'xy' dict (plain lists) of the source-free derived values at every recorded point (x = time [s] or
        wavelength [m] in record order), None below 2 points or without such derived values."""
        if len(self._series_x) < 2 or not self._series:
            return None
        time_axis = self.series_axis == 'time'
        return {'kind': 'xy', 'x': list(self._series_x), 'x_label': 'Time' if time_axis else 'Wavelength',
                'x_unit': 's' if time_axis else 'm', 'y_label': 'Derived value',
                'series': {name: list(values) for name, values in self._series.items()},
                'series_units': {name: '' for name in self._series}}

    def _record_series(self, x):
        """One derived-series point at x: each source-free derived value of the live state (NaN + one warning per
        name when it fails; a GPU OOM raises)."""
        if x is None or not self._series:
            return
        session = self._session
        self._series_x.append(float(x))
        point_values = {}   # this point's values, for derived values referencing earlier ones
        with session.shared_field_reads():   # each window field read once per point
            for derived in session.spec.derived_values:
                if derived.name not in self._series:
                    continue
                try:
                    value = float(session.derived_value(derived, earlier=point_values))
                except Exception as exc:   # text only: user input, as Session.derived_values
                    if gpu_memory.is_out_of_memory(exc):
                        raise
                    value = math.nan
                    if derived.name not in self._series_failed:
                        self._series_failed.add(derived.name)
                        self._emit(LogEvent('warning', f"derived value '{derived.name}' along the study: "
                                                       f'{_failure_message(exc)} (NaN; further points not reported)'))
                point_values[derived.name] = value
                self._series[derived.name].append(value)

    def _field_data(self, key, physics, component, native, plane, cut_index, blank_physics):
        """FieldData of a stored field in one cut (None = the default cut; cut_index 0, the stored cuts 1, 2, ...); a
        blank physics' field is NaN, copied from its last stored FieldData without a read when there is one."""
        last = self._last_reads.get((key, cut_index))
        if physics in blank_physics and last is not None:
            return replace(last, values=np.full_like(last.values, np.nan))
        session = self._session
        field_data = (session.field(component, physics=physics, plane=plane) if native[cut_index]
                      else session.display_field(component, physics=physics, max_pixels=_SNAPSHOT_PIXELS, plane=plane))
        if physics in blank_physics:
            field_data = replace(field_data, values=np.full_like(field_data.values, np.nan))
        self._last_reads[(key, cut_index)] = field_data
        return field_data

    def _segment(self, segment, blank_physics):
        """LineProfile of one 3D-segment curve (_segment_curves entry); NaN for a blank physics (as _field_data)."""
        name, physics, component, start, end, sampling = segment
        last = self._last_reads.get(('segment', name))
        if physics in blank_physics and last is not None:
            return replace(last, values=np.full_like(last.values, np.nan))
        profile = self._session.segment_profile(component, start, end, physics=physics, sampling=sampling)
        if physics in blank_physics:
            profile = replace(profile, values=np.full_like(profile.values, np.nan))
        self._last_reads[('segment', name)] = profile
        return profile


def _segment_curves(session, available, emit):
    """[(curve name, physics, component, start, end, sampling)] of the 3D-segment line curves a snapshot records
    (Snapshot.segments); a curve whose field this study does not offer or whose line does not evaluate is skipped
    with a warning."""
    if session.domain.dimensions != 3:
        return []
    segments = []
    for curve in session.spec.analytic_curves:
        if curve.target != 'line' or not is_segment_3d(curve.line):
            continue
        physics, _, component = curve.field.partition('.')
        try:
            start, end = line_endpoints(curve.line, session.parameters)
        except ValueError as exc:
            emit(LogEvent('warning', f"line curve '{curve.name}': not recorded in the snapshots ({exc})"))
            continue
        if (physics, component) not in available:
            emit(LogEvent('warning', f"line curve '{curve.name}': field '{curve.field}' is not available in this "
                                     'study, so the snapshots hold no profile of it'))
            continue
        segments.append((curve.name, physics, component, start, end, curve.sampling))
    return segments


def stored_cut_planes(spec, parameters):
    """[(axis, position [m])] of StudySpec.stored_cuts (the Session.field planes snapshots add); ValueError on a bad
    entry (validate reports it first)."""
    planes = []
    for cut in spec.study.stored_cuts:
        if not (isinstance(cut, (list, tuple)) and len(cut) == 2 and cut[0] in ('x', 'y', 'z')):
            raise ValueError(f"stored cut must be ['x' | 'y' | 'z', position in m], got {cut!r}")
        planes.append((cut[0], evaluate_scalar(cut[1], parameters)))
    return planes


class _LiveFeed:
    """Rate-limited LiveFrames (control.live_field, default the first physics' primary field) and LiveResults
    (none when control.live_results is False)."""

    def __init__(self, session, control, emit, frame_rate_hz):
        self._session = session
        self._control = control
        self._emit = emit
        self._frame_period = 1.0 / frame_rate_hz if frame_rate_hz > 0.0 else None   # None = no frames
        self._last_frame = -math.inf
        self._last_results = -math.inf
        self._default_field = next(iter(_primary_fields(session)), None)   # first physics' primary field
        self._timeseries = _result_features(session, ('timeseries',)) if control.live_results else []
        self._rejected_request = None

    def update(self, final=False):
        """Emit what is due (everything when final)."""
        now = time.perf_counter()
        if self._frame_period is not None and (final or now - self._last_frame >= self._frame_period):
            self._last_frame = now
            frame = self._frame()
            if frame is not None:
                self._emit(LiveFrame(frame))
        if self._timeseries and (final or now - self._last_results >= _LIVE_RESULT_PERIOD):
            self._last_results = now
            self._emit(LiveResults({key: _decimated(self._session.feature_result(*key))
                                    for key in self._timeseries}))

    def _frame(self):
        """FieldData of the requested live field, else the default one; a failing field is reported once (a
        renderer error must not end the study; a GPU OOM still raises); None without a field."""
        request = self._control.live_field
        if request is not None:
            physics, component, max_pixels = request
            try:
                return self._session.display_field(component, physics=physics, max_pixels=max_pixels)
            except Exception as exc:   # text only
                if gpu_memory.is_out_of_memory(exc):
                    raise
                if request != self._rejected_request:   # report each bad request once
                    self._rejected_request = request
                    self._emit(LogEvent('warning', f'live field {physics}.{component} is not available '
                                                   f'({_failure_message(exc)}); showing the default field'))
        if self._default_field is None:
            return None
        physics, component = self._default_field
        try:
            return self._session.display_field(component, physics=physics, max_pixels=_LIVE_PIXELS)
        except Exception as exc:   # text only: no live frames for the rest of the run
            if gpu_memory.is_out_of_memory(exc):
                raise
            self._default_field = None
            self._emit(LogEvent('warning', f'live field {physics}.{component} failed ({_failure_message(exc)}); no '
                                           'live view for the rest of this run'))
            return None


def _log_solve_summaries(physics, emit, suffix):
    """Info line '<name>: <solve_summary()><suffix>' per solver with a solve_summary (FDFD: back end, unknowns, times,
    factor size), so the log names the linear solver and the cost of one solve."""
    for name, solver in physics.items():
        summary = solver.solve_summary() if hasattr(solver, 'solve_summary') else None
        if summary:
            emit(LogEvent('info', f'{name}: {summary}{suffix}'))


def _duration_text(seconds):
    """'12.3 s', '3.52 min' or '4.27 h' of a wall time [s]."""
    if seconds < 120.0:
        return f'{seconds:.3g} s'
    if seconds < 7200.0:
        return f'{seconds / 60.0:.3g} min'
    return f'{seconds / 3600.0:.3g} h'


class _StageProgress:
    """The run's stages (names in order, equal shares of the run fraction): stage_emit(name) re-emits each Progress of
    that stage with its fraction / wall time as stage_fraction / stage_elapsed and the run's as fraction / wall_elapsed;
    other events pass through."""

    def __init__(self, emit, names):
        self._emit = emit
        self.names = tuple(names)
        self._wall_start = time.perf_counter()

    def stage_emit(self, name):
        index = self.names.index(name)

        def emit(event):
            if isinstance(event, Progress):
                stage_fraction = _clamped_fraction(event)
                event = replace(event, fraction=(index + stage_fraction) / len(self.names),
                                wall_elapsed=time.perf_counter() - self._wall_start, stages=self.names,
                                stage_index=index, stage_fraction=stage_fraction, stage_elapsed=event.wall_elapsed)
            self._emit(event)
        return emit


def _physics_text(session, name):
    """'name (type label)' of a built physics for progress messages."""
    for physics_spec in session.spec.physics:
        if physics_spec.name == name:
            return f'{name} ({session.registry.physics_type(physics_spec.type).label})'
    return name


class _SolverWarnings:
    """Logs the first unconverged solve of each physics in a run (solvers with n_unconverged / last_warning) and the
    entries a solver appends to its runtime_warnings list during the run."""

    def __init__(self, physics, emit):
        self._emit = emit
        self._baseline = {name: (solver, solver.n_unconverged) for name, solver in physics.items()
                          if hasattr(solver, 'n_unconverged')}
        self._runtime_logged = {name: (solver, len(solver.runtime_warnings)) for name, solver in physics.items()
                                if hasattr(solver, 'runtime_warnings')}

    def check(self):
        for name, (solver, count_before) in list(self._baseline.items()):
            if solver.n_unconverged > count_before:
                del self._baseline[name]
                self._emit(LogEvent('warning', f"{solver.last_warning}; further unconverged solves of '{name}' "
                                               f"in this run are not reported"))
        for name, (solver, n_logged) in list(self._runtime_logged.items()):
            for message in solver.runtime_warnings[n_logged:]:
                self._emit(LogEvent('warning', f"physics '{name}': {message}"))
            self._runtime_logged[name] = (solver, len(solver.runtime_warnings))


class _FixedPointMonitor:
    """Relative change of the primary fields of the named physics between stationary iterations; heat T is monitored
    as its rise T_rise (float32 T near 300 K quantises at ~3e-5 K, a floor above the tolerance for small rises)."""

    def __init__(self, session, physics_names):
        self._fields = []
        for name, component in _primary_fields(session):
            if name not in physics_names:
                continue
            renderer = session.renderer(name)
            if component == 'T' and 'T_rise' in renderer.field_infos:
                component = 'T_rise'
            self._fields.append((renderer, component, renderer.device_field(component).copy()))

    def relative_change(self):
        """max over fields of max|f_new - f_old| / max(max|f_new|, 1e-30) (fused reductions: no window-sized
        temporaries; NaN propagates); keeps f_new as f_old."""
        ratios = []
        for renderer, component, previous in self._fields:
            current = renderer.device_field(component)
            xp = array_module_of(current)
            if xp is np:   # cpu backend: numpy reductions (np.max keeps a NaN like the fused kernels)
                scale = np.maximum(np.max(np.abs(current)), _CONVERGENCE_FLOOR)
                ratios.append(np.max(np.abs(current - previous)) / scale)
            else:
                scale = cp.maximum(axis_first_reduce(_max_abs, current, combine=_peak), _CONVERGENCE_FLOOR)
                ratios.append(axis_first_reduce(_max_abs_difference, current, previous, combine=_peak) / scale)
            xp.copyto(previous, current)
        return float(array_module_of(ratios[0]).stack(ratios).max())


def _primary_fields(session):
    """[(physics, component)] of each physics' primary output field (its first one when none is marked primary)."""
    primaries = []
    for physics in session.physics:
        primary = primary_field_info(session.renderer(physics).field_infos.values())
        if primary is not None:
            primaries.append((physics, primary.name))
    return primaries


def _coupling_sources(coupling):
    """frozenset of the physics names a runner coupling reads (its source_fields)."""
    return frozenset(physics for physics, _ in coupling.source_fields)


def _coupling_targets(coupling):
    """frozenset of the physics names a runner coupling writes (a CouplingGroup: its terms' targets); empty for a
    DynamicMaterialUpdate, which writes the shared MaterialGrid."""
    targets = {term.target for term in getattr(coupling, 'terms', ())}
    target = getattr(coupling, 'target', None)
    if target is not None:
        targets.add(target)
    return frozenset(targets)


def _coupling_physics(coupling):
    """Physics names a runner coupling reads or writes: source_fields plus its targets."""
    return _coupling_sources(coupling) | _coupling_targets(coupling)


def _writes_only_sinks(coupling, loop_members):
    """True for a coupling with named targets outside the fixed-point loop (a DynamicMaterialUpdate writes every
    physics, so it never is)."""
    targets = _coupling_targets(coupling)
    return bool(targets) and not targets & loop_members


def _stationary_roles(physics_names, couplings):
    """(loop, sinks) for solve_stationary: a sink feeds no coupling writing a remaining physics (a DynamicMaterialUpdate
    writes all of them) and is solved once after the fixed point; loop in runner order, sinks upstream first."""
    loop = list(physics_names)
    removed = []   # downstream first
    while True:
        members = frozenset(loop)
        sink = next((name for name in loop if not _feeds_members(name, couplings, members)), None)
        if sink is None:
            return loop, removed[::-1]
        loop.remove(sink)
        removed.append(sink)


def _feeds_members(name, couplings, members):
    """True when a coupling reading physics `name` writes one of members (a DynamicMaterialUpdate: any of them)."""
    return any(name in _coupling_sources(coupling) and (_coupling_targets(coupling) or members) & members
               for coupling in couplings)


def _coupling_schedule(order, couplings):
    """Gauss-Seidel apply lists: [0] = couplings reading no physics of `order` (applied before the first step), [i] =
    couplings whose last source in `order` is order[i - 1] (applied right after its step); runner order kept."""
    position = {name: index for index, name in enumerate(order, start=1)}
    schedule = [[] for _ in range(len(order) + 1)]
    for coupling in couplings:
        last_source = max((position[name] for name in _coupling_sources(coupling) if name in position), default=0)
        schedule[last_source].append(coupling)
    return schedule


def _gauss_seidel_sweep(runner, order, schedule, before_step=None):
    """step(0.0, inf) of each physics in order (before_step(name) first), each coupling apply_now'd as soon as all its
    sources in order stepped."""
    for coupling in schedule[0]:
        coupling.apply_now(runner.physics)   # every solver: a material update notifies all of them
    for index, name in enumerate(order, start=1):
        if before_step is not None:
            before_step(name)
        runner.physics[name].step(0.0, math.inf)
        for coupling in schedule[index]:
            coupling.apply_now(runner.physics)


def _is_loop_back(coupling, index, position):
    """True for a coupling applied after step `index` that writes a physics already stepped in the sweep (a
    DynamicMaterialUpdate writes all); position = {physics: 1-based sweep index}."""
    target_positions = [position[name] for name in _coupling_targets(coupling) if name in position]
    return not target_positions or min(target_positions) <= index


def _has_loop_back(schedule, order):
    """True when a loop schedule holds a loop-back coupling (its first sweep solves with slot values from before it)."""
    position = {name: index for index, name in enumerate(order, start=1)}
    return any(_is_loop_back(coupling, index, position)
               for index, couplings in enumerate(schedule[1:], start=1) for coupling in couplings)


def _wrap_loop_back(schedule, order, wrap):
    """Copy of a loop schedule with each loop-back coupling replaced by wrap(coupling) (_RelaxedTransfer,
    _MixedTransfer): the iterate the next sweep starts from."""
    position = {name: index for index, name in enumerate(order, start=1)}
    wrapped = [list(schedule[0])]   # reads no loop physics: constant input, nothing to relax
    for index, couplings in enumerate(schedule[1:], start=1):
        wrapped.append([wrap(coupling) if _is_loop_back(coupling, index, position) else coupling
                        for coupling in couplings])
    return wrapped


class _RelaxedTransfer:
    """Stationary loop-back coupling injecting previous + relaxation * (new - previous) per term (CouplingTerm,
    CouplingGroup terms, DynamicMaterialUpdate); the first transfer goes in unrelaxed."""

    def __init__(self, coupling, relaxation):
        self._parts = getattr(coupling, 'terms', (coupling,))
        self._relaxation = relaxation
        self._previous = {}   # part -> last injected value (own copy: transfer values may be live state arrays)

    def apply_now(self, physics_by_name):
        for part in self._parts:
            value = part.transfer_value(physics_by_name)
            if value is None:
                continue
            previous = self._previous.get(part)
            if previous is not None:
                value = previous + self._relaxation * (value - previous)
            self._previous[part] = value.copy() if hasattr(value, 'copy') else value
            part.write(physics_by_name, value)


class _MixedTransfer:
    """Stationary loop-back coupling under Anderson mixing: injects each part's new value at once (physics later in
    the sweep see it, as plain Gauss-Seidel) and records it for the _LoopBackMixer's re-injection before the next
    sweep."""

    def __init__(self, coupling, mixer):
        self._parts = getattr(coupling, 'terms', (coupling,))
        self._mixer = mixer

    def apply_now(self, physics_by_name):
        for part in self._parts:
            value = part.transfer_value(physics_by_name)
            if value is None:
                continue
            self._mixer.record(part, value)
            part.write(physics_by_name, value)


class _LoopBackMixer:
    """Anderson acceleration of a stationary loop (core.damped_iteration.AndersonHistory, depth m, mixing parameter =
    relaxation) on host float64 copies of every loop-back part, each scaled by its largest magnitude: mix() before
    sweep 2 takes the injected values as the first iterate, before sweep 3 re-injects a relaxed step, from sweep 4
    the mixed one. Complex values are mixed as real pairs (real coefficients); a change of the recorded parts restarts
    the history."""

    def __init__(self, depth, relaxation):
        self._depth = int(depth)
        self._relaxation = float(relaxation)
        self._history = None
        self._layout = None   # ((part, size), ...) of the stacked vector the history holds
        self._current = {}    # part -> host float64 iterate the running sweep started from
        self._pending = {}    # part -> (host float64 new value, (shape, dtype, array module) of the transfer value)

    def record(self, part, value):
        """Keep a part's new value of this sweep (host float64 copy) for the next mix."""
        array = np.array(to_host(value), copy=True)
        flat = (np.ascontiguousarray(array, dtype=np.complex128).ravel().view(np.float64)
                if np.iscomplexobj(array) else array.astype(np.float64).ravel())
        module = array_module_of(value) if hasattr(value, 'shape') else None
        self._pending[part] = (flat, (array.shape, array.dtype, module))

    def mix(self, physics_by_name):
        """Re-inject every part recorded in the last sweep with the mixed iterate the next sweep starts from."""
        pending, self._pending = self._pending, {}
        if not pending:
            return
        layout = tuple((part, flat.size) for part, (flat, _) in pending.items())
        if layout != self._layout:   # first sweep or other parts: the injected values start a new history
            self._history = AndersonHistory(self._depth)
            self._layout = layout
            self._current = {part: flat for part, (flat, _) in pending.items()}
            return
        state = np.concatenate([self._current[part] for part, _ in layout])
        new = np.concatenate([pending[part][0] for part, _ in layout])
        scale = np.concatenate([np.full(size, max(float(np.max(np.abs(pending[part][0]), initial=0.0)),
                                                  float(np.max(np.abs(self._current[part]), initial=0.0)),
                                                  _CONVERGENCE_FLOOR))
                                for part, size in layout])
        mixed = state + self._history.step(state, new - state, scale, self._relaxation)
        offset = 0
        for part, size in layout:
            flat = mixed[offset:offset + size]
            offset += size
            self._current[part] = flat
            part.write(physics_by_name, _restored_value(flat, *pending[part][1]))


def _restored_value(flat, shape, dtype, module):
    """A transfer value of the recorded shape, dtype and array module (None: a Python scalar) from host float64."""
    values = flat[0::2] + 1j * flat[1::2] if np.issubdtype(dtype, np.complexfloating) else flat
    if module is None:
        return values.astype(dtype).reshape(shape).item()
    return module.asarray(values.astype(dtype).reshape(shape))


def _stored_field_keys(session):
    """'physics.component' keys stored in snapshots: StudySpec.stored_fields, or each physics' primary field plus
    its '<base>_im' partner of a '<base>_re' primary (harmonic physics), so |field|^2 can be rebuilt; a frequency_domain
    study takes a renderer's harmonic_primary instead (harmonic FDTD: ez_abs / hz_abs); then the StudyType's
    default_stored_fields (e.g. optimization: design_density)."""
    requested = list(session.spec.study.stored_fields)
    if requested:
        return requested
    frequency_domain = session.spec.study.type == 'frequency_domain'
    keys = []
    for physics, component in _primary_fields(session):
        harmonic_primary = getattr(session.renderer(physics), 'harmonic_primary', '') if frequency_domain else ''
        component = harmonic_primary or component
        keys.append(f'{physics}.{component}')
        imaginary_component = component[:-len('_re')] + '_im' if component.endswith('_re') else None
        if imaginary_component in session.renderer(physics).field_infos:
            keys.append(f'{physics}.{imaginary_component}')
    default_stored_fields = session.registry.study_type(session.spec.study.type).default_stored_fields
    if default_stored_fields is not None:
        keys += [key for key in default_stored_fields(session.spec) if key not in keys]
    return keys


def _result_features(session, result_kinds):
    """[(physics, feature)] of built features whose FeatureType.result_kind is in result_kinds (spec order)."""
    registry = session.registry
    found = []
    for physics_spec in session.spec.physics:
        physics_type = registry.physics_type(physics_spec.type)
        for feature_spec in physics_spec.features:
            key = (physics_spec.name, feature_spec.name)
            if key in session.features and physics_type.feature_type(feature_spec.type).result_kind in result_kinds:
                found.append(key)
    return found


def _decimated(result):
    """Timeseries result with at most 4000 samples: the min and max sample of each bin in time order (keeps the
    envelope, no stride aliasing), 'decimation_stride' = samples per bin; too sparse for an FFT."""
    times = np.asarray(result['t'])
    values = np.asarray(result['values'])
    n_samples = times.size
    if n_samples <= _LIVE_RESULT_POINTS:
        return result
    bin_size = -(-n_samples // (_LIVE_RESULT_POINTS // 2))
    n_bins = -(-n_samples // bin_size)
    padded = np.full(n_bins * bin_size, values[-1], dtype=values.dtype)   # last bin padded with its last sample
    padded[:n_samples] = values
    bins = padded.reshape(n_bins, bin_size)
    bin_starts = np.arange(n_bins) * bin_size
    min_index = np.minimum(bin_starts + np.argmin(bins, axis=1), n_samples - 1)
    max_index = np.minimum(bin_starts + np.argmax(bins, axis=1), n_samples - 1)
    picked = np.sort(np.stack([min_index, max_index], axis=1), axis=1).ravel()
    return {**result, 't': times[picked], 'values': values[picked], 'decimation_stride': int(bin_size)}


def _next_chunk_ticks(chunk_ticks, ticks_done, seconds):
    """Ticks for the next chunk aiming at ~0.1 s wall time, growing at most 4x per chunk."""
    if ticks_done == 0 or seconds <= 0.0:
        return chunk_ticks
    return max(1, min(4 * chunk_ticks, int(ticks_done * _CHUNK_SECONDS / seconds)))


def _next_output_time(t_now, t_start, output_interval, dt_sync):
    """First output time t_start + k * interval beyond t_now + dt/2, at least one tick ahead (float safety)."""
    k = math.floor((t_now + 0.5 * dt_sync - t_start) / output_interval) + 1
    return max(t_start + k * output_interval, t_now + dt_sync)


def _time_progress(runner, t_start, t_final, tick_start, wall_start, message=''):
    """Progress of a time-dependent run (rate = mean ticks per wall second since the start)."""
    wall = time.perf_counter() - wall_start
    span = t_final - t_start
    fraction = min(max((runner.t - t_start) / span, 0.0), 1.0) if span > 0.0 else 1.0
    ticks = runner.tick - tick_start
    return Progress(fraction, runner.t, t_final, runner.tick, wall, ticks / wall if wall > 0.0 else 0.0, message)


def _count_progress(runner, fraction, ticks, wall_start, message, tick_unit):
    """Progress of a stage counting solves (iterations, wavelengths, mode / eigen solves, factorisations): ticks done
    at their mean rate since wall_start, at the runner's current t."""
    wall = time.perf_counter() - wall_start
    return Progress(fraction, runner.t, runner.t, ticks, wall, ticks / wall if wall > 0.0 else 0.0, message,
                    tick_unit=tick_unit)


def _fixed_point_fraction(iteration, max_iterations, first_change, relative_change, tolerance):
    """Fixed-point progress: 1 once converged, else the larger of iterations / max_iterations and the share of the
    decades from the first relative change down to the tolerance already covered (at most 0.99)."""
    if relative_change <= tolerance:
        return 1.0
    fraction = iteration / max_iterations
    if math.isfinite(first_change) and first_change > tolerance and 0.0 < relative_change < first_change:
        decades = math.log(first_change / relative_change) / math.log(first_change / tolerance)
        fraction = max(fraction, min(decades, 0.99))
    return min(fraction, 1.0)


def _log_to(emit):
    """log(level, message) callback (BuildContext.log) that emits LogEvents."""
    def log(level, message):
        emit(LogEvent(level, message))
    return log


def _failure_message(exc):
    """str(exc) for model errors (they carry issues), else 'TypeName: message'."""
    if getattr(exc, 'issues', None) is not None:
        return str(exc)
    return f'{type(exc).__name__}: {exc}'


def _record_failure(exc):
    """(message, traceback text, out_of_memory) of the exception being handled; a GPU OOM message names the budget
    and the remedies. Call inside `except` and keep only this text (the traceback holds the run's GPU arrays)."""
    out_of_memory = gpu_memory.is_out_of_memory(exc)
    message = gpu_memory.out_of_memory_message(exc) if out_of_memory else _failure_message(exc)
    return message, traceback.format_exc(), out_of_memory
