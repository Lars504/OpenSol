"""Headless interface for agents and scripts (no Qt): describe, template / load / save, check, run, selftest, Results.
run and live Results use the GPU on the calling thread: never mix them with a GUI BackendWorker in one process."""
import copy
import itertools
import math
import os
import shutil
import sys
import time
import traceback
import warnings
import weakref
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import __version__
from .core import gpu_memory, host_memory
from .core.backend import default_backend_name
from .core.mirror import mirror_faces, unfold_field
from .core.renderer import primary_field_info
from .core.time_estimate import format_seconds
from .core.time_profile import profile_path, profile_text, run_profile
from .core.wavelengths import resolved_wavelengths, wavelength_label
from .io.model_io import export_volume, load_model, save_model
from .io.results_io import (LOG_FILE, SUMMARY_FILE, FieldRecord, ResultsData, field_digest, json_safe, load_results,
                            read_json, result_digest, save_results, write_json)
from .model.analytic import (CURVE_RESULT_KINDS, FFT_SERIES, RATIO_SERIES, built_parameters, curve_error,
                             derived_curve_plots, feature_curve, feature_curve_samples, field_plane_height,
                             is_segment_3d, line_cell_curve_values, line_curve, line_endpoints,
                             line_plane, parameters_with, snapshot_index, split_key, sweep_curve_values,
                             sweep_overlay_xy, sweep_richardson, sweep_simulated, sweep_table_xy, sweep_value_curve,
                             table_xy)
from .model.builder import ESTIMATE_FLOOR_BYTES, build as build_model, estimate_gap, estimate_verdict
from .model.describe import REFERENCE_FILE, REGENERATE_COMMAND, TEMPLATES_FILE, describe, reference_files
from .model.expressions import error_message, evaluate_parameters
from .model.import_contracts import check as check_import_contracts, rule_text
from .model.interface_sampling import interface_line_profile, line_curve_field_keys, projected_info
from .model.meshing import (MESH_ESTIMATE_ERRORS, built_mesh_estimates, estimate_memory, estimate_mesh,
                            estimate_meshes, fitted_memory_estimate, line_block_hint, mesh_transfers,
                            mesh_values, physics_settings)
from .model.scene_3d import build_scene
from .model.schema import (ISSUE_LEVELS, Issue, get_registry, node_label, resolve_values, resolved_solvers,
                           solver_backend, solver_grid)
from .solvers.roles import resolved_precision
from .model.selftest_cache import CACHE_FILE, DependencyTrace, SelftestCache, environment_key, file_hash
from .model.selftest_examples import (ANALYTIC_REPORT_FILE, analytic_report, example_record, failed_example_record,
                                      has_analytic_comparison)
from .model.session import VOLUME_MAX_BYTES
from .model.spec import ModelSpec, SolverSpec
from .model.structure_index import CONVENTIONS_PATH, STRUCTURE_INDEX_PATH, structure_index_text
from .model.study import (BuildFailed, Built, LiveFrame, LiveResults, LogEvent, Progress, RunControl, StudyFailed,
                          StudyFinished, StudyStarted, SweepStep, SweepValueDone, field_meshes, release_session,
                          run_study, tick_count_text)
from .model.study_sectors import ordered_sector_specs
from .model.time_estimate import estimate_overheads, estimate_time, estimated_runs, stored_profile
from .model.templates import make_template, template_entries
from .model.validate import internal_error_text, refused_on_estimated_counts, validate
from .viz.common import line_profile
from .viz.result_plots import (far_field_wavelengths, save_field_figure, save_line_profile_figure,
                               save_result_figure)
from .viz.scene_figure import DEFAULT_VIEW, scene_figure as new_scene_figure

__all__ = ['API_REFERENCE_PATH', 'API_TEMPLATES_PATH', 'BUILTIN_SELFTESTS', 'describe', 'template', 'load', 'save', 'check', 'run',
           'save_session_run', 'scene_figure', 'selftest', 'selftest_examples', 'profile_machine', 'Results', 'RunError',
           'ProgressPrinter',
           # run() progress events (open_sol.model.study dataclasses)
           'LogEvent', 'SweepStep', 'SweepValueDone', 'Built', 'BuildFailed', 'StudyStarted', 'Progress',
           'StudyFinished', 'StudyFailed']

API_REFERENCE_PATH = Path(__file__).resolve().parent / REFERENCE_FILE
API_TEMPLATES_PATH = API_REFERENCE_PATH.parent / TEMPLATES_FILE   # template JSON, written beside API_REFERENCE_PATH
BUILTIN_SELFTESTS = ('templates_validate', 'examples_validate', 'api_reference_current', 'import_contracts',
                     'structure_index_current')   # CPU, run first (a named run: only those named)
# report subset -> report file in out_dir; a run of subset 'all' writes the 'light' and the 'heavy' file, then the
# examples run once (ANALYTIC_REPORT_FILE).
SELFTEST_REPORT_FILES = {'light': 'selftest_report_light.json', 'heavy': 'selftest_report_heavy.json',
                         'named': 'selftest_report_named.json'}
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / 'examples'   # repository checkout only (examples/README.md)
_REGENERATE_REFERENCE = f'regenerate it with: {REGENERATE_COMMAND}'
_CHECK_KEYS = ('measured', 'expected', 'error', 'tolerance', 'details')     # SelfTestCase.check() result
_MAX_LINE_CHARS = 200                  # api_reference_current: quoted line length
_BUILD_SEQUENCE = itertools.count(1)   # build order of live Results (leak check on close)
_LIVE_RESULTS = weakref.WeakSet()      # Results whose GPU session is still open
_LIVE_RESULTS_WARNING_PATTERN = r'\d+ earlier Results still hold a GPU session'   # start of _live_results_warning
_FLUX_FIGURE_SERIES = ('', 'power')    # flux_spectrum curve series drawn on its figure's power panel (as the GUI)
_DERIVED_SERIES_QUANTITY = 'derived_series:'   # Results.sweep_figure quantity prefix of a derived-series overlay
# study summary keys copied into summary.json when the study set them (optimization, thermal_noise, mode_analysis,
# eigenfrequency / optomechanics, nonlinear_frequency_domain, frequency_response; convergence: StudySpec.adaptation,
# also written as convergence.json; estimation: StudySpec.estimation)
_STUDY_SUMMARY_KEYS = ('optimization', 'noise_frequencies', 'n_modes', 'n_resonances', 'n_channels', 'nonlinear_solve',
                       'feedback', 'response_frequencies', 'frequency_sweep', 'convergence', 'estimation')


class RunError(Exception):
    """run() without out_dir: the study failed after building (message, traceback text)."""

    def __init__(self, message, traceback_text):
        super().__init__(message)
        self.message = message
        self.traceback = traceback_text


# ---- model files ----
def template(key) -> ModelSpec:
    """Fresh ModelSpec of a template (built-in or registered); KeyError listing the keys."""
    return make_template(key)


def load(path) -> ModelSpec:
    """ModelSpec of a JSON model file; ValueError naming the file for invalid content, OSError when unreadable."""
    return load_model(path)


def save(spec, path) -> None:
    """Write a spec (ModelSpec, dict or model file path) as a JSON model file."""
    save_model(_as_spec(spec), path)


# ---- 3D scene ----
def scene_figure(spec, path, parameters=None, highlight=(), z_stretch=1.0, view=None) -> str:
    """CPU-only perspective picture (.png / .pdf / .svg by suffix) of a 3D model's objects in their material colours,
    as the GUI 3D preview; parameters {name: expression} override the model, highlight = entity names drawn in the
    highlight colour, view (elevation, azimuth) [deg]. ValueError for a 2D model; returns str(path)."""
    model_spec = _as_spec(spec)
    _apply_parameters(model_spec, parameters)
    return _save_scene_figure(model_spec, evaluate_parameters(model_spec.parameters, errors={}), path,
                              highlight=highlight, z_stretch=z_stretch, view=view)


def _save_scene_figure(spec, parameter_values, path, field_planes=(), highlight=(), z_stretch=1.0, view=None):
    """Save the scene_figure of spec's Scene3D (field_planes: FieldData cuts); objects that failed to resolve are a
    RuntimeWarning. Creates the parent directory; returns str(path)."""
    scene = build_scene(spec, parameter_values, get_registry(), highlight=highlight)
    if scene.problems:
        warnings.warn(f"3D scene: not drawn: {'; '.join(scene.problems)}", RuntimeWarning, stacklevel=3)
    figure = new_scene_figure(scene, field_planes=field_planes, z_stretch=float(z_stretch),
                              view=DEFAULT_VIEW if view is None else tuple(float(angle) for angle in view))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    return str(path)


# ---- machine profile (run-time estimates) ----
def profile_machine(backend=None, quick=False, progress=print) -> dict:
    """Measure and store the machine profile of the run-time estimate for the backend ('gpu' / 'cpu'; None = the
    default backend) in time_profiles/ next to the package (OPEN_SOL_PROFILE_DIR overrides): ~30 s, quick ~5 s
    (coarser curves). progress(text) per benchmark (None: silent). Returns the profile record {key, date, quick,
    text, path}."""
    backend = backend or default_backend_name()
    profile = run_profile(backend, quick=quick, progress=progress)
    return dict(profile.record(), text=profile_text(profile), path=profile_path(backend))


# ---- check ----
def check(spec, out_dir=None, parameters=None, memory_budget=None, host_budget=None, backend=None,
          max_wall_time=None) -> dict:
    """CPU-only report of a spec (issues, mesh, memory and run-time estimates, time step and step count); out_dir ->
    <out_dir>/check.json. parameters {name: expression} overrides parameter expressions (an unknown name is an issue
    at ('parameters',)); memory_budget [bytes] (e.g. a run's summary gpu_budget) adds the GPU memory check,
    host_budget [bytes] (e.g. core.host_memory.available_bytes(); the CLI passes it) the host RAM check (an estimate
    over a budget on size-based element counts: the triangle / tetrahedron meshes are built first and the report is
    on their counts, as a build judges); backend ('auto' | 'gpu' | 'cpu') overrides solver_defaults.backend;
    max_wall_time [s]: a warning when even the shortest estimated run time exceeds it."""
    registry = get_registry()
    meshes, mesh_errors, study, memory, solvers, transfers = {}, {}, None, None, None, []
    time_record = None
    by_mesh = {}   # mesh name -> MemoryEstimate (check.json meshes gpu_bytes / host_bytes)
    override_issues = []
    try:
        model_spec = _as_spec(spec)
    except ValueError as exc:   # invalid content (JSON, keys, version) is a model error, not an exception
        model_spec = None
        issues = [Issue('error', ('model',), error_message(exc))]
    if model_spec is not None:
        _apply_backend(model_spec, backend)
        try:
            _apply_parameters(model_spec, parameters)
        except KeyError as exc:   # reported, then the spec is checked without the overrides
            override_issues = [Issue('error', ('parameters',), error_message(exc))]
        parameter_values = evaluate_parameters(model_spec.parameters, errors={})
        for mesh_spec in model_spec.used_meshes():
            try:
                meshes[mesh_spec.name] = estimate_mesh(model_spec, parameter_values, registry, mesh_spec.name)
            except MESH_ESTIMATE_ERRORS as exc:
                mesh_errors[mesh_spec.name] = error_message(exc)
            except Exception as exc:   # a crashing definition callback (domain_requirements), reported as validate does
                mesh_errors[mesh_spec.name] = internal_error_text('the mesh estimate', exc)
        if meshes and not mesh_errors:
            try:
                memory = estimate_memory(model_spec, parameter_values, registry, meshes, by_mesh=by_mesh)
            except Exception:   # model errors and crashing estimators: validate reports them
                memory, by_mesh = None, {}
            transfers = mesh_transfers(model_spec, registry, meshes)
        if not mesh_errors and refused_on_estimated_counts(model_spec, meshes, memory, memory_budget, host_budget):
            meshes, memory, by_mesh, override_issues = _judged_estimates(
                model_spec, registry, parameter_values, meshes, memory, by_mesh, override_issues)
        plain_roles = None   # D1 4b: the build's fit guard on these estimates (None: validate fits itself)
        if memory is not None and not mesh_errors:
            try:
                memory, plain_roles = fitted_memory_estimate(model_spec, parameter_values, registry, meshes,
                                                             memory_budget, estimate=memory, by_mesh=by_mesh,
                                                             host_budget=host_budget)
            except Exception:   # model errors and crashing estimators: validate reports them
                plain_roles = None
        issues = override_issues + validate(model_spec, registry, mesh_estimate=meshes, mesh_error=mesh_errors,
                                            memory_budget=memory_budget, memory_estimate=memory,
                                            host_budget=host_budget, memory_parts=by_mesh or None,
                                            plain_roles=plain_roles)
        study = _study_estimate(model_spec, registry, parameter_values, list(meshes.values()))
        solvers = _solver_report(model_spec, registry, parameter_values, plain_roles or frozenset())
        if meshes and not mesh_errors and memory is not None:
            time_record, time_issue = _time_report(model_spec, parameter_values, registry, meshes, max_wall_time)
            issues.append(time_issue)
    counts = {level: sum(1 for issue in issues if issue.level == level) for level in ISSUE_LEVELS}
    used = [] if model_spec is None else model_spec.used_meshes()
    default_name = used[0].name if used else None   # 'mesh' / 'mesh_error': the first used mesh
    report = {
        'open_sol_version': __version__,
        'status': 'errors' if counts['error'] else ('warnings' if counts['warning'] else 'ok'),
        'model': {'title': None if model_spec is None else model_spec.title, 'file': _model_file(spec)},
        'n_errors': counts['error'], 'n_warnings': counts['warning'], 'n_infos': counts['info'],
        'issues': [_issue_record(issue) for issue in issues],
        'mesh': _mesh_record(meshes.get(default_name)),
        'mesh_error': mesh_errors.get(default_name),
        'meshes': [] if model_spec is None else _mesh_records(model_spec, parameter_values, meshes, mesh_errors,
                                                              by_mesh),
        'transfers': transfers,
        'study': study,
        'memory_estimate': None if memory is None else {**memory.as_dict(), 'budget': memory_budget,
                                                        'host_budget': host_budget},
        'time_estimate': time_record,
        'solvers': solvers,
    }
    if out_dir is not None:
        write_json(Path(out_dir) / 'check.json', report)
    return report


def _judged_estimates(spec, registry, parameters, meshes, memory, by_mesh, issues):
    """(meshes, memory, by_mesh, issues) for the final memory judgement of check: the element meshes built and counted
    (meshing.built_mesh_estimates) unless the model has another error; a mesher failure keeps the estimates and adds
    its error Issue."""
    if any(issue.level == 'error' for issue in validate(spec, registry, mesh_estimate=meshes, memory_estimate=memory)):
        return meshes, memory, by_mesh, issues   # no mesher run for a model that fails anyway
    try:
        built = built_mesh_estimates(spec, parameters, registry, meshes)
        built_parts = {}
        return built, estimate_memory(spec, parameters, registry, built, by_mesh=built_parts), built_parts, issues
    except Exception as exc:
        return meshes, memory, by_mesh, issues + [Issue('error', ('meshes',), 'the element meshes could not be built '
                                                        f'for the memory check: {error_message(exc)}')]


def _time_report(spec, parameters, registry, meshes, max_wall_time=None):
    """(check.json time_estimate record or None, its Issue): info with the estimate text, or why there is none; a
    warning when its shortest run time exceeds max_wall_time."""
    runs = None
    try:
        if stored_profile(solver_backend(spec))[0] is not None:   # each run's meshes once, for both estimates
            runs = estimated_runs(spec, parameters, registry, meshes)
        estimate, message = estimate_time(spec, parameters, registry, meshes, runs=runs)
    except Exception as exc:   # model errors are validate's; a crashing recipe must not cost the report
        return None, Issue('info', ('study',), f'run time not estimated: {error_message(exc)}')
    if estimate is None:
        return None, Issue('info', ('study',), message)
    record = dict(estimate.as_dict(), build=None)
    build_text = ''
    try:   # the build of every run (meshing included); a failure costs only this part
        build_part = estimate_overheads(spec, parameters, registry, meshes, runs=runs).get('build')
    except Exception:   # model errors are validate's
        build_part = None
    if build_part is not None:
        record['build'] = build_part.as_dict()
        build_text = f'; build estimated {format_seconds(build_part.seconds_guess)}'
    if max_wall_time is not None and estimate.seconds_low > float(max_wall_time):
        return record, Issue('warning', ('study',),
                             f'{estimate.text()}: even the shortest estimate exceeds max_wall_time '
                             f'{format_seconds(float(max_wall_time))}')
    return record, Issue('info', ('study',), estimate.text(count=3) + build_text)


def _solver_report(spec, registry, parameters, plain_roles=frozenset()):
    """check.json 'solvers' block: per physics the resolved role solvers {role: {type, solver (node or None), backend,
    precision, settings}} of the study, None for a physics whose solvers do not resolve (validate reports it)."""
    backend = solver_backend(spec)
    report = {}
    for physics_spec in spec.physics:
        try:
            resolved = resolved_solvers(spec, registry, physics_spec, spec.study.type, parameters, plain_roles)
        except Exception:   # unknown type / node, bad settings or a crashing callback: validate reports each
            report[physics_spec.name] = None
            continue
        report[physics_spec.name] = {
            role: {'type': entry.solver_type.name, 'solver': entry.node, 'backend': backend,
                   'precision': resolved_precision(entry.solver_type, entry.values), 'settings': dict(entry.values)}
            for role, entry in resolved.items()}
    return report


def _study_estimate(spec, registry, parameters, meshes):
    """check.json 'study' block: smallest time step (Courant dt of the estimated meshes, time_step of time-stepping
    physics), the number of steps (duration / dt per sweep value) and of frequency_domain wavelengths; None for what
    cannot be resolved."""
    try:
        study_type = registry.study_type(spec.study.type)
    except Exception:   # unknown study type: validate reports it
        study_type = None
    time_steps = [float(mesh.dt) for mesh in meshes if mesh.dt is not None]
    if study_type is None or study_type.requires_time_stepping:   # other studies force steady physics (no time_step)
        for physics_spec in spec.physics:
            try:
                physics_type = registry.physics_type(physics_spec.type)
                settings = physics_settings(physics_type, physics_spec, parameters)
                steps_in_time = physics_type.steps_in_time is not None and physics_type.steps_in_time(settings)
            except Exception:   # unknown type, bad settings or a crashing callback: validate reports each of them
                continue
            time_step = settings.get('time_step')
            if steps_in_time and isinstance(time_step, (int, float)) and not isinstance(time_step, bool):
                time_steps.append(float(time_step))
    dt_estimate = min(time_steps) if time_steps else None
    try:
        study_values = {} if study_type is None else resolve_values(study_type.params, spec.study.settings, parameters)
    except Exception:   # bad settings: validate reports them
        study_values = {}
    duration = study_values.get('duration')
    n_steps = (round(duration / dt_estimate) if dt_estimate and isinstance(duration, (int, float))
               else None)
    n_wavelengths = (len(resolved_wavelengths(study_values))
                     if spec.study.type == 'frequency_domain' and study_values else None)
    sweep = spec.study.sweep
    n_sweep_values = 1 if sweep is None else (len(sweep.point_values()) or None)
    total_steps = None if n_steps is None or n_sweep_values is None else n_steps * n_sweep_values
    return {'type': spec.study.type, 'dt_estimate': dt_estimate, 'n_steps': n_steps,
            'n_sweep_values': n_sweep_values, 'total_steps': total_steps, 'n_wavelengths': n_wavelengths}


# ---- run ----
def run(spec, out_dir=None, max_wall_time=None, progress=None, figures=False, save_arrays=True,
        parameters=None, backend=None) -> 'Results':
    """Build + run a spec on this thread, blocking (never next to a GUI BackendWorker in one process) -> Results;
    progress: callable(event) or True (ProgressPrinter to stdout); parameters {name: expression} and backend ('auto' |
    'gpu' | 'cpu': solver_defaults.backend) override the model (kept in model.json). With out_dir it always writes
    summary.json + log.txt (a failed report write: status 'failed'); without, a failed build re-raises, a failed run raises RunError (no GPU budget: RuntimeError)."""
    if progress is True:
        progress = ProgressPrinter()
    max_wall_time = None if max_wall_time is None else float(max_wall_time)
    if max_wall_time is not None and not max_wall_time > 0.0:
        raise ValueError(f'max_wall_time must be > 0 s (or None), got {max_wall_time!r}')
    out_dir = None if out_dir is None else Path(out_dir)
    wall_start = time.perf_counter()
    collector = _Collector(progress, wall_start)
    live_warning = _live_results_warning()
    if live_warning is not None:
        collector.add_line('warning', live_warning)
        warnings.warn(live_warning, RuntimeWarning, stacklevel=2)
    spec_issues, setup_failure = [], None   # setup_failure: message of a run that cannot start
    try:
        input_spec = _as_spec(spec)
    except ValueError as exc:   # invalid model content: a model error in the report (OSError / TypeError propagate)
        if out_dir is None:
            raise
        input_spec = None
        spec_issues = [Issue('error', ('model',), error_message(exc))]
        setup_failure = f'invalid model: {spec_issues[0].message}'
        collector.add_line('error', setup_failure)
    if input_spec is not None:
        _apply_backend(input_spec, backend)   # a bad value is a validation error of the build
        try:
            _apply_parameters(input_spec, parameters)
        except KeyError as exc:   # unknown parameter name: a model error at ('parameters',)
            if out_dir is None:
                raise
            input_spec = None
            spec_issues = [Issue('error', ('parameters',), error_message(exc))]
            setup_failure = f'invalid parameter override: {spec_issues[0].message}'
            collector.add_line('error', setup_failure)
    budget_error = _log_device(collector, solver_backend(input_spec) if input_spec is not None
                               else default_backend_name())
    if budget_error is not None and setup_failure is None:   # never run without a pool limit
        if out_dir is None:
            raise RuntimeError(budget_error)
        spec_issues, setup_failure = [Issue('error', ('model',), budget_error)], budget_error
    written_files = {}   # summary 'files' key -> name of each file save_results wrote in this run
    recorder, tracker = _BuildRecorder(), _SessionTracker()
    results, interrupted = None, None
    try:
        session = None
        if setup_failure is None:
            session, timed_out = _execute_study(input_spec, collector, recorder, tracker, max_wall_time, wall_start)
            status, message, traceback_text = _final_status(collector, timed_out, max_wall_time)
        else:
            status, message, traceback_text = 'failed', setup_failure, ''
        wall_time = time.perf_counter() - wall_start
        if out_dir is None and status == 'failed':
            _raise_failure(session, collector, recorder, message, traceback_text)
        results = Results(status, input_spec, out_dir, session, collector.built_summary, collector.lines,
                          log_start=wall_start)
        final = collector.final_event
        issues = spec_issues or ([] if collector.build_failure is None else list(collector.build_failure.issues))
        outcome = _RunOutcome(status, message, traceback_text,
                              final.summary if isinstance(final, StudyFinished) else None, issues,
                              _model_file(spec), wall_time)
        _post_run(results, outcome, collector, figures, save_arrays, written_files)
    except KeyboardInterrupt as exc:
        gpu_memory.drop_exception_frames(exc)
        interrupted = exc
    if interrupted is not None:
        _after_interrupt(collector, tracker, results, out_dir, written_files)
        raise interrupted
    return results


def _live_results_warning():
    """Warning text when earlier api Results still hold GPU sessions (they stay allocated during this build), else
    None."""
    live_results = [results for results in list(_LIVE_RESULTS) if results.is_live]
    if not live_results:
        return None
    known_bytes = [results.build_summary.session_gpu_bytes for results in live_results
                   if results.build_summary is not None and results.build_summary.session_gpu_bytes is not None]
    memory_text = f', {gpu_memory.format_bytes(sum(known_bytes))} on the GPU' if known_bytes else ''
    return (f'{len(live_results)} earlier Results still hold a GPU session{memory_text}, which stays allocated '
            f'while this run builds (r = run(...) in a loop frees the old r only after the new run); close() them '
            f'first or use "with run(...) as r:"')


@dataclass
class _RunOutcome:
    """What run_study left behind, before the post-run stage turns it into the report."""
    status: str                 # 'completed' | 'partial' | 'timeout' | 'cancelled' | 'failed'
    message: str
    traceback: str
    study_summary: dict | None  # StudyFinished.summary
    issues: list                # Issue of the last failed build (or of the invalid spec)
    model_file: str | None
    wall_time: float


class _Collector:
    """emit target of run(): log.txt lines, warning / error texts, the last Built summary and BuildFailed and the final
    StudyFinished / StudyFailed; every event except LiveFrame / LiveResults goes to progress (dropped if it raises)."""

    def __init__(self, progress, start):
        self.progress = progress
        self.start = start                  # perf_counter of the run start (log elapsed times)
        self.lines = []
        self.warnings = []
        self.errors = []
        self.built_summary = None
        self.build_failure = None
        self.final_event = None

    def add_line(self, level, message):
        """One '[elapsed] [level] message' log line; warnings and errors are also kept as text."""
        self.lines.append(_log_line(self.start, level, message))
        if level == 'warning':
            self.warnings.append(message)
        elif level == 'error':
            self.errors.append(message)

    def add_detail(self, text):
        """Multi-line detail (issues, traceback) indented under the previous line."""
        self.lines.extend(f'    {line}' for line in text.rstrip('\n').split('\n'))

    def emit(self, event):
        if isinstance(event, LogEvent):
            self.add_line(event.level, event.message)
        elif isinstance(event, Built):
            self.built_summary = event.summary
            self.add_line('info', _built_text(event.summary))
        elif isinstance(event, BuildFailed):
            self.build_failure = event
            self.add_line('error', f'build failed: {event.message}')
            for issue in event.issues:
                self.add_detail(f'[{issue.level}] {node_label(issue.path)}: {issue.message}')
            if event.traceback:
                self.add_detail(event.traceback)
        elif isinstance(event, SweepStep):
            self.add_line('info', _sweep_step_text(event))
        elif isinstance(event, StudyStarted):
            self.add_line('info', 'study started')
        elif isinstance(event, StudyFinished):
            self.final_event = event
            summary = event.summary
            t_end_text = (f"t_end {float(summary.get('t_end', math.nan)):.6g} s, "
                          if summary.get('tick_unit', 'step') == 'step' else '')   # time-stepped studies only
            self.add_line('info', f"study finished ({summary.get('status')}): {tick_count_text(summary)}, "
                                  f"{t_end_text}wall {float(summary.get('wall_time', math.nan)):.2f} s")
        elif isinstance(event, StudyFailed):
            self.final_event = event
            self.add_line('error', f'study failed: {event.message}')
            if event.traceback and (self.build_failure is None
                                    or event.traceback != self.build_failure.traceback):
                self.add_detail(event.traceback)
        if self.progress is not None and not isinstance(event, (LiveFrame, LiveResults)):
            try:
                self.progress(event)
            except Exception as exc:   # e.g. a closed stdout pipe: must not end the run; text only (no exc kept)
                self.progress = None
                self.add_line('warning', f'progress callback failed: {_error_text(exc)}; progress disabled')


class ProgressPrinter:
    """run() progress callback (also the CLI's): build, log and sweep events at once, Progress lines at most every
    period s of wall time, to stream (None: sys.stdout at each print)."""

    def __init__(self, stream=None, period=2.0):
        self.stream = stream
        self.period = float(period)
        self._last_progress = -math.inf

    def __call__(self, event):
        if isinstance(event, Progress):
            now = time.perf_counter()
            if now - self._last_progress < self.period:
                return
            self._last_progress = now
            text = _progress_text(event)
        elif isinstance(event, LogEvent):
            text = f'[{event.level}] {event.message}'
        elif isinstance(event, Built):
            summary = event.summary
            text = (f'built {_cells_text(summary.nx, summary.ny, summary.nz, summary.n_cells)} in '
                    f'{summary.build_seconds:.1f} s')
        elif isinstance(event, BuildFailed):
            text = f'[error] build failed: {event.message}'
        elif isinstance(event, SweepStep):
            text = _sweep_step_text(event)
        else:
            return
        print(text, file=sys.stdout if self.stream is None else self.stream, flush=True)


def _progress_text(event):
    """'  42% t = ... / ... s, 1234 steps/s' (time-dependent) or '  42% <message>' (build, stationary); a run of
    several stages adds '[stage 2/3 <name> 45%]' after the run percentage."""
    percent = f'{100.0 * event.fraction:3.0f}%'
    if len(event.stages) > 1 and 0 <= event.stage_index < len(event.stages):
        stage_percent = 100.0 * (event.fraction if event.stage_fraction is None else event.stage_fraction)
        percent += (f' [stage {event.stage_index + 1}/{len(event.stages)} {event.stages[event.stage_index]} '
                    f'{stage_percent:.0f}%]')
    if event.t_end > 0.0:
        text = (f'{percent} t = {event.t:.4g} / {event.t_end:.4g} s, '
                f'{event.ticks_per_second:.0f} {event.tick_unit}s/s')
        return f'  {text}' + (f'  {event.message}' if event.message else '')
    return f'  {percent} {event.message}'


def _sweep_step_text(event):
    """'sweep 1/3: P = 2' of a SweepStep (the GUI's wording)."""
    return f'sweep {event.index + 1}/{event.count}: {event.parameter} = {event.value:g}'


class _BuildRecorder:
    """build callback of run_study: model.builder.build that keeps the last failure's exception with its frames
    cleared (no GPU arrays) for run() to re-raise, then re-raises it."""

    def __init__(self):
        self.exception = None

    def __call__(self, spec, registry=None, log=None, **options):
        try:   # options: build keywords such as live_parameters (a reused sweep build)
            return build_model(spec, registry=registry, log=log, **options)
        except Exception as exc:
            gpu_memory.drop_exception_frames(exc)
            self.exception = exc
            raise


class _SessionTracker:
    """on_session callback of run_study: remembers the newest session (never closes it)."""

    def __init__(self):
        self.session = None

    def __call__(self, session):
        self.session = session


def _execute_study(input_spec, collector, recorder, tracker, max_wall_time, wall_start):
    """run_study on a copy with live output off -> (session or None, timed_out); its service cancels the study once
    max_wall_time has passed (checked at every study checkpoint, so not during a build)."""
    run_spec = input_spec.copy()
    _disable_live_output(run_spec)
    control = RunControl(live_results=False)   # emit drops LiveResults headless: never build them
    timed_out = False

    def stop_after_max_wall_time():
        nonlocal timed_out
        if not timed_out and time.perf_counter() - wall_start > max_wall_time:
            timed_out = True
            control.cancel()

    session = run_study(run_spec, collector.emit, control, build=recorder,
                        service=None if max_wall_time is None else stop_after_max_wall_time, on_session=tracker)
    if timed_out:
        collector.add_line('warning', _timeout_text(max_wall_time))
    return session, timed_out


def _disable_live_output(run_spec):
    """live_update_hz = 0 when the study type has that param (nobody watches live frames headless)."""
    try:
        study_type = get_registry().study_type(run_spec.study.type)
    except KeyError:   # run_study reports the unknown study type
        return
    if (isinstance(run_spec.study.settings, dict)
            and any(param.name == 'live_update_hz' for param in study_type.params)):
        run_spec.study.settings['live_update_hz'] = 0.0


def _final_status(collector, timed_out, max_wall_time):
    """(status, message, traceback text) from the final study event and the time-out flag; a finished sweep with
    failed values is 'partial'; an unconverged fixed point is noted in the message only."""
    final = collector.final_event
    if isinstance(final, StudyFinished):
        if final.summary.get('status') != 'cancelled':
            return (*_finished_status(final.summary), '')
        if timed_out:
            return 'timeout', _timeout_text(max_wall_time), ''
        return 'cancelled', 'the study was cancelled', ''
    if isinstance(final, StudyFailed):
        if timed_out:
            return 'timeout', f'{_timeout_text(max_wall_time)} ({final.message})', final.traceback
        return 'failed', final.message, final.traceback
    return 'failed', 'run_study ended without a final event', ''


def _finished_status(study_summary):
    """(status, message) of a study that finished uncancelled: 'partial' with failed sweep values, else 'completed';
    an unconverged fixed point / refinement is noted in the message only."""
    sweep = study_summary.get('sweep')
    convergence_note = _unconverged_text(study_summary)
    if sweep is not None and sweep['failed']:
        return 'partial', '; '.join(filter(None, (_sweep_failure_text(sweep), convergence_note)))
    return 'completed', convergence_note


def _timeout_text(max_wall_time):
    return f'stopped after max_wall_time = {max_wall_time:g} s; partial results'


def _sweep_failure_text(sweep):
    """'k/n sweep values failed' message of a partial sweep, naming the failed values and the first error."""
    failed_values = sweep['failed']
    value_text = ', '.join(f"{float(entry['value']):g}" for entry in failed_values)
    return (f"{len(failed_values)}/{len(sweep['values'])} sweep values failed ({sweep['parameter']} = {value_text}; "
            f"NaN / empty results); first error: {failed_values[0]['message']}")


def _unconverged_text(study_summary):
    """'fixed point did not converge ...' note of a finished study (sweep: the unconverged values) and the message of
    unconverged mesh refinement levels (S7.7 'convergence'), else ''."""
    sweep = study_summary.get('sweep')
    if sweep is not None:
        unconverged = [value for value, converged in zip(sweep['values'], sweep['converged']) if converged is False]
        if not unconverged:
            return ''
        return (f"fixed point did not converge for {len(unconverged)}/{len(sweep['values'])} sweep values "
                f"({sweep['parameter']} = {', '.join(f'{float(value):g}' for value in unconverged)}; sweep.converged)")
    notes = []
    fixed_point = study_summary.get('fixed_point')
    if fixed_point is not None and not fixed_point['converged']:
        notes.append(f"fixed point did not converge in {fixed_point['iterations']} iterations (relative change "
                     f"{fixed_point['relative_change']:.2e} > tolerance {fixed_point['tolerance']:.2e}); results not "
                     'converged')
    convergence = study_summary.get('convergence')
    if isinstance(convergence, dict) and not convergence.get('converged'):
        notes.append(f"mesh refinement: {convergence.get('message')}")
    return '; '.join(notes)


def _raise_failure(session, collector, recorder, message, traceback_text):
    """No out_dir and a failed run: close the session nobody will receive, then re-raise the failed build's own
    exception (frames cleared) when no build succeeded, else raise RunError."""
    release_session(None, session, emit=collector.emit)
    if collector.built_summary is None and recorder.exception is not None:
        build_exception = recorder.exception
        gpu_memory.drop_exception_frames(build_exception)   # build_session's frame has finished since it was kept
        raise build_exception
    raise RunError(message, traceback_text)


def _after_interrupt(collector, tracker, results, out_dir, written_files):
    """KeyboardInterrupt in run(): close the newest session, then (out_dir) write the minimal report, status
    'cancelled'."""
    session, tracker.session = tracker.session, None
    if results is not None:
        results.session = None
        _LIVE_RESULTS.discard(results)
    release_session(None, session, emit=collector.emit)
    collector.add_line('warning', 'interrupted by the user')
    if out_dir is not None:
        figure_files = [] if results is None else results._data.figure_files
        _write_fallback_report(out_dir, 'cancelled', 'interrupted by the user', collector.errors, '',
                               written_files, figure_files, collector.lines)


def _log_device(collector, backend='gpu'):
    """Recompute the GPU budget (other programs take and free memory between runs) and log the device line; returns
    the budget failure text (logged as an error: the run must not start), else None. backend 'cpu': the host RAM
    line only (no GPU budget, no driver query)."""
    if backend == 'cpu':
        collector.add_line('info', f'CPU backend: {gpu_memory.format_bytes(host_memory.available_bytes())} host '
                                   'memory available')
        return None
    try:
        gpu_memory.set_budget()
    except Exception as exc:
        budget_error = f'GPU memory budget not set: {_error_text(exc)}'
        collector.add_line('error', budget_error)
        return budget_error
    try:
        info = gpu_memory.budget_device_info()   # set_budget's memory numbers: one driver memory query per run
    except Exception as exc:   # informational line; the build reports a real GPU problem itself
        collector.add_line('warning', f'GPU query failed: {_error_text(exc)}')
        return None
    collector.add_line('info', f'GPU {info.name}: {gpu_memory.format_bytes(info.total_bytes)} total, '
                               f'{gpu_memory.format_bytes(info.free_bytes)} free; memory budget '
                               f'{gpu_memory.format_bytes(gpu_memory.budget_bytes())}')
    return None


# ---- post-run stage: summary, host cache, figures, save ----
def _post_run(results, outcome, collector, figures, save_arrays, written_files):
    """Summary + host cache, figures and save. With out_dir a failure here becomes a minimal summary.json + log.txt;
    without, it propagates after the session was closed."""
    out_dir = results.out_dir
    stage_exception, failure_text, failure_traceback = None, None, ''
    try:
        # save() and, without out_dir, the caller's close() would stack them anyway: fetch every frequency once
        stack_dft = save_arrays
        results._data.summary = _run_summary(results, outcome, collector, stack_dft)
        if figures:
            _render_figures(results, collector)
        if out_dir is not None:
            results._save(out_dir, save_arrays, written_files)
    except Exception as exc:
        if out_dir is None:
            gpu_memory.drop_exception_frames(exc)
            stage_exception = exc
        else:   # text only: the session stays usable for the returned Results
            failure_text = f'post-run failure: {_error_text(exc)}'
            failure_traceback = traceback.format_exc()
    if stage_exception is not None:
        results._close(copy_to_host=False)   # nobody receives this Results
        raise stage_exception
    if failure_text is not None:   # incomplete report: status failed (exit 1), the study's own status in message
        collector.add_line('error', failure_text)
        study_text = f'the study ended {outcome.status}' + (f' ({outcome.message})' if outcome.message else '')
        results.status = 'failed'
        results._data.summary = _write_fallback_report(out_dir, 'failed', f'{failure_text}; {study_text}',
                                                       collector.errors, failure_traceback, written_files,
                                                       results._data.figure_files, collector.lines)


def save_session_run(directory, spec, session, build_summary, study_summary, log_entries, model_file=None,
                     save_arrays=True) -> str:
    """Write the run directory run(out_dir=) would (summary.json, log.txt, model.json, results.npz, meshes.npz) of a
    live session its caller built and ran (the GUI worker, on its thread); the session stays open. log_entries =
    [(clock text, level, message)]; returns the directory."""
    if study_summary is None:
        raise ValueError('no finished study to save: compute first (a build alone has no results)')
    collector = _Collector(None, time.perf_counter())
    for clock, level, message in log_entries:
        collector.lines.append(f'{clock} [{level}] {message}')
        if level in ('warning', 'error'):
            (collector.warnings if level == 'warning' else collector.errors).append(message)
    if study_summary.get('status') == 'cancelled':
        status, message = 'cancelled', 'the study was cancelled'
    else:
        status, message = _finished_status(study_summary)
    build_seconds = math.nan if build_summary is None else float(build_summary.build_seconds)
    wall_time = build_seconds + float(study_summary.get('wall_time', math.nan))
    results = Results(status, spec, None, session, build_summary, collector.lines)
    try:
        outcome = _RunOutcome(status, message, '', study_summary, [], model_file, wall_time)
        results._data.summary = _run_summary(results, outcome, collector, save_arrays)
        results._save(Path(directory), save_arrays, {})
    finally:
        results.session = None   # the caller's session: never closed here
        _LIVE_RESULTS.discard(results)
    return str(directory)


def _run_summary(results, outcome, collector, stack_dft):
    """summary.json content of a run; fills the host cache (fields, results, sweep results, snapshots) on the way
    (stack_dft: the DFT 'intensity_all' too)."""
    session, data, build_summary, spec = results.session, results._data, results.build_summary, results.spec
    study_summary = outcome.study_summary
    study = study_summary or {}   # its .get reads None / defaults without a study summary
    if study_summary is not None:
        stored_keys = list(study_summary.get('stored_fields', []))
    elif session is not None:   # failed run with a live session: each physics' primary field
        stored_keys = _primary_field_keys(session)
    else:
        stored_keys = []
    field_digests = {} if session is None else _field_digests(session, stored_keys, data.fields)
    result_digests = ({} if session is None or build_summary is None
                      else _result_digests(session, build_summary.feature_results, data.results, stack_dft))
    sweep = study.get('sweep')
    if sweep is not None:
        data.sweep_results = list(sweep['results'])
        data.sweep_snapshots = [list(snapshots) for snapshots in sweep.get('snapshots') or []]
    sectors = study.get('parity_sectors')
    if sectors is not None:
        data.sector_results = list(sectors['results'])
        data.sector_combined_results = dict(sectors.get('combined_results') or {})
    data.snapshots = [] if session is None else list(session.snapshots)
    data.meshes = {} if session is None else session.mesh_arrays()   # S7.2: meshes.npz
    mesh_records = study.get('meshes')
    stored_meshes = study.get('field_meshes')
    if session is not None:
        mesh_records = session.mesh_records() if mesh_records is None else mesh_records
        stored_meshes = field_meshes(session, stored_keys) if stored_meshes is None else stored_meshes
    no_curves = spec is None or study_summary is None
    curve_errors, sweep_curve_errors = ({}, {}) if no_curves else _curve_errors(results, sweep, study_summary)
    if study_summary is not None:
        t_end = study_summary.get('t_end')
    else:
        t_end = None if session is None else session.time
    summary = {
        'open_sol_version': __version__,
        'status': outcome.status,
        'message': outcome.message,
        'model': {'title': None if spec is None else spec.title, 'file': outcome.model_file,
                  'study_type': None if spec is None else spec.study.type},
        'wall_time': outcome.wall_time,
        'build': _build_record(build_summary),
        'build_seconds_total': study.get('build_seconds_total'),
        'study_seconds': study.get('study_seconds'),
        'solvers': dict(study.get('solvers', {})),
        'run_memory': study.get('run_memory'),
        'external_memory': study.get('external_memory'),
        'time_check': study.get('time_check'),
        'parameters': None if study_summary is None else _recorded_parameters(study_summary),
        'sweep_index': study.get('sweep_index'),
        'ticks': study.get('ticks'),
        'tick_unit': study.get('tick_unit'),
        't_end': t_end,
        'warnings': list(collector.warnings),
        'errors': list(collector.errors),
        'issues': [_issue_record(issue) for issue in outcome.issues],
        'traceback': outcome.traceback,
        'memory': _memory_record(build_summary),
        'failed_build_memory': None if collector.build_failure is None else collector.build_failure.build_memory,
        'derived': dict(study.get('derived', {})),
        'derived_reference': dict(study.get('derived_reference', {})),
        'derived_error': dict(study.get('derived_error', {})),
        'curve_error': curve_errors,
        'derived_curves': {} if no_curves else _derived_curve_records(results),
        'fixed_point': study.get('fixed_point'),
        'derived_series': study.get('derived_series'),
        'sweep': None if sweep is None else _sweep_record(sweep, sweep_curve_errors),
        'parity_sectors': None if sectors is None else _sector_record(sectors),
        'stored_fields': stored_keys,
        'meshes': mesh_records or [],
        'field_meshes': dict(stored_meshes or {}),
        'stored_cuts': list(study.get('stored_cuts') or []),
        'fields': field_digests,
        'results': result_digests,
        'snapshot_times': [float(snapshot.time) for snapshot in data.snapshots],
        'snapshot_labels': [snapshot.label for snapshot in data.snapshots],
    }
    if study.get('wavelengths') is not None:   # frequency_domain
        summary['wavelengths'] = [float(wavelength) for wavelength in study_summary['wavelengths']]
        summary['harmonic_pulse'] = study_summary.get('harmonic_pulse')   # harmonic FDTD pulse statistics or None
        if study_summary.get('wavelength_fixed_points') is not None:   # coupled wavelength scan
            summary['wavelength_fixed_points'] = list(study_summary['wavelength_fixed_points'])
    for key in _STUDY_SUMMARY_KEYS:   # study-specific records (describe('results'))
        if study.get(key) is not None:
            summary[key] = json_safe(study[key])
    summary['files'] = None   # set by results_io.save_results
    return summary


def _field_digests(session, keys, records):
    """'physics.component' -> field_digest (or {'error'}) of each key; the host FieldRecords go into records."""
    digests = {}
    for key in keys:
        try:
            physics, _, component = key.partition('.')
            records[key] = _host_field_record(session, physics, component)
            digests[key] = json_safe(field_digest(records[key]))
        except Exception as exc:   # one failing field must not cost the rest of the report
            digests[key] = {'error': _error_text(exc)}
    return digests


def _result_digests(session, feature_results, results, stack_dft):
    """'physics.feature' -> result_digest (or {'kind', 'error'}); the host result dicts go into results. dft_fields:
    stack_dft stacks 'intensity_all' now and takes the maxima from it, else fetches them one frequency at a time."""
    digests = {}
    for physics, feature, kind in feature_results:
        key = f'{physics}.{feature}'
        try:
            result = results[key] = session.feature_result(physics, feature)
            max_intensity = None
            if kind == 'dft_fields' and stack_dft:   # each frequency is fetched once (result_digest reads the stack)
                result['intensity_all'] = _dft_intensity_stack(session, physics, feature, result)
            elif kind == 'dft_fields':
                max_intensity = _dft_max_intensity(session, physics, feature, len(result['frequency']))
            digests[key] = json_safe(result_digest(result, max_intensity=max_intensity))
        except Exception as exc:   # one failing result must not cost the rest of the report
            digests[key] = {'kind': kind, 'error': _error_text(exc)}
    return digests


def _recorded_parameters(study_summary):
    """{name: float} of the study summary's session parameters, or None when it recorded none."""
    recorded = study_summary.get('parameters')
    return None if recorded is None else {name: float(value) for name, value in recorded.items()}


def _curve_errors(results, sweep, study_summary):
    """(curve_error {name: {max, rms, n_points}}, sweep curve_error {name: [record per sweep value]}) of the spec's
    analytic curves: feature targets on the final results (and per sweep value), line targets on the field profile,
    sweep targets over the swept values; a curve that cannot be compared: NaN, n_points 0 and a note."""
    spec, data = results.spec, results._data
    parameters = built_parameters(spec, study_summary)
    final, per_value = {}, {}
    for curve in spec.analytic_curves:
        try:
            if curve.target == 'sweep':
                if sweep is None:
                    raise ValueError('the study has no sweep')
                final[curve.name] = curve_error(sweep_simulated(curve.series, sweep),
                                                sweep_curve_values(curve.expression, spec, sweep['values']),
                                                per_point=True)
            elif curve.target == 'line':
                profile = results._line_profile(curve, parameters)
                final[curve.name] = curve_error(profile.samples, line_cell_curve_values(
                    curve.expression, parameters, profile.start, profile.end, profile.centres, z=profile.height))
            else:
                if curve.target not in data.results:
                    raise KeyError(f'no result {curve.target!r} in this run')
                final[curve.name] = curve_error(*feature_curve_samples(curve, data.results[curve.target], parameters))
        except Exception as exc:   # text only: one curve must not cost the summary
            final[curve.name] = _uncompared_curve(exc)
        if sweep is not None and curve.target not in ('sweep', 'line'):
            per_value[curve.name] = [_value_curve_error(curve, value_results.get(curve.target),
                                                        parameters_with(spec, sweep['parameter'], value))
                                     for value, value_results in zip(sweep['values'], sweep['results'])]
    return final, per_value


def _derived_curve_records(results):
    """summary.json 'derived_curves': {curve name: {plot, label, unit, x_label, x_unit, x, y (+ y_reference,
    curve_error with a reference)} or {plot, error}} over the final results (a sweep: its last value)."""
    records = {}
    for plot, xy, errors in results.derived_curves():
        for name, error in errors.items():
            records[name] = {'plot': plot, 'error': error}
        for name, series in ({} if xy is None else xy['curve_series']).items():
            record = {'plot': plot, 'label': series, 'unit': xy['series_units'][series], 'x_label': xy['x_label'],
                      'x_unit': xy['x_unit'], 'x': np.asarray(xy['x']).tolist(),
                      'y': np.asarray(xy['series'][series]).tolist()}
            if name in xy['curve_error']:   # analytic.derived_curve_plots: curves with a reference
                record['curve_error'] = xy['curve_error'][name]
                reference_series = xy['reference_series'].get(name)
                if reference_series is not None:
                    record['y_reference'] = np.asarray(xy['series'][reference_series]).tolist()
            records[name] = json_safe(record)
    return records


def _value_curve_error(curve, result, parameters):
    """curve_error record of a feature-target curve on one sweep value's result (None: the value failed)."""
    try:
        if result is None:
            raise KeyError(f'no result {curve.target!r} for this sweep value')
        return curve_error(*feature_curve_samples(curve, result, parameters))
    except Exception as exc:   # text only
        return _uncompared_curve(exc)


def _uncompared_curve(exc):
    return {'max': math.nan, 'rms': math.nan, 'n_points': 0, 'note': _error_text(exc)}


def _result_digest_list(results_per_run):
    """[{key: result_digest or {kind, error}}] of a list of {'physics.feature': result dict} (sweep values, sectors)."""
    digests = []
    for run_results in results_per_run:
        run_digests = {}
        for key, result in run_results.items():
            try:
                run_digests[key] = json_safe(result_digest(result))
            except Exception as exc:   # one failing digest must not cost the table
                run_digests[key] = {'kind': result.get('kind'), 'error': _error_text(exc)}
        digests.append(run_digests)
    return digests


def _sweep_record(sweep, curve_errors):
    """summary.json 'sweep' block: parameter, values, derived (+ reference / error) columns, per-value curve errors and
    result digests, failed values, per-value fixed-point convergence, derived series and the model build count (+
    sector_derived of a parity-sector sweep, snapshot_times / snapshot_labels per value with keep_fields)."""
    kept = sweep.get('snapshots')
    kept_record = {} if kept is None else {
        'snapshot_times': [[float(snapshot.time) for snapshot in snapshots] for snapshots in kept],
        'snapshot_labels': [[snapshot.label for snapshot in snapshots] for snapshots in kept]}
    return {'parameter': sweep['parameter'], 'values': [float(value) for value in sweep['values']],
            'derived': {name: list(values) for name, values in sweep['derived'].items()},
            'reference': {name: list(values) for name, values in sweep.get('reference', {}).items()},
            'error': {name: list(values) for name, values in sweep.get('error', {}).items()},
            'curve_error': curve_errors, 'results': _result_digest_list(sweep['results']),
            'failed': [{'index': entry['index'], 'value': float(entry['value']), 'message': entry['message']}
                       for entry in sweep['failed']],
            'converged': list(sweep['converged']),
            'derived_series': list(sweep.get('derived_series') or []), **kept_record,
            **({'builds': int(sweep['builds'])} if 'builds' in sweep else {}),
            **({'sector_derived': [{name: list(values) for name, values in per_value.items()}
                                   for per_value in sweep['sector_derived']]} if 'sector_derived' in sweep else {})}


def _sector_record(sectors):
    """summary.json 'parity_sectors' block (model.study_sectors): mirror faces, sector labels / statuses, whether the
    derived values were combined, per-sector derived values, result digests (per sector and of the combined results)
    and the sector of every snapshot."""
    return {'faces': list(sectors['faces']), 'labels': list(sectors['labels']), 'status': list(sectors['status']),
            'combined': bool(sectors['combined']),
            'derived': {name: list(values) for name, values in sectors['derived'].items()},
            'results': _result_digest_list(sectors['results']),
            'combined_results': _result_digest_list([sectors.get('combined_results') or {}])[0],
            'snapshot_sectors': list(sectors['snapshot_sectors'])}


def _render_figures(results, collector):
    """figures/<key>.png of every saved field and feature result, figures/derived_<plot>.png per derived curve plot,
    derived_series.png, the sweep overlays (Results._sweep_figure_jobs) and a 3D run's scene_3d.png (needs out_dir);
    failures are logged, not raised."""
    data = results._data
    if results.out_dir is None:
        collector.add_line('warning', 'figures=True needs out_dir; no figures were written')
    else:
        for key, render in results._figure_jobs():
            relative = f'figures/{key}.png'
            try:
                render(results.out_dir / relative)
            except Exception as exc:   # a figure must not cost the report
                collector.add_line('warning', f'{relative} not written: {_error_text(exc)}')
            else:
                data.figure_files.append(relative)
    data.summary['warnings'] = list(collector.warnings)


def _scene_figure_jobs(results):
    """[('scene_3d', render(key, path))] of a 3D run: Results.scene_figure of the first saved field (objects only
    without one); [] in 2D."""
    if results.spec is None or results.spec.domain.size_z is None:
        return []
    field = next(iter(results._data.fields), None)
    return [('scene_3d', lambda key, path: results.scene_figure(path, field=field))]


def _write_fallback_report(out_dir, status, message, errors, traceback_text, written_files, figure_files,
                           log_lines):
    """Minimal summary.json (plain values only) + log.txt after a post-run failure or an interrupt, returned; 'files'
    lists only what save_results wrote in this run. Never raises: an unwritable out_dir is noted in errors, files None."""
    files = {'summary': SUMMARY_FILE, 'log': LOG_FILE, 'arrays': written_files.get('arrays'),
             'meshes': written_files.get('meshes'), 'model': written_files.get('model'), 'figures': list(figure_files)}
    summary = {'open_sol_version': __version__, 'status': status, 'message': message, 'errors': list(errors),
               'traceback': traceback_text, 'files': files}
    try:   # log.txt first: summary.json names it
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / LOG_FILE).write_text(''.join(f'{line}\n' for line in log_lines), encoding='utf-8', newline='\n')
        write_json(out_dir / SUMMARY_FILE, summary)
    except Exception as exc:   # run() must still return its Results (or re-raise the interrupt), session intact
        summary['errors'].append(f'report not written to {out_dir}: {_error_text(exc)}')
        summary['files'] = None
    return summary


# ---- Results ----
def _stored_cut_index(snapshot, plane):
    """Index into snapshot.cuts of plane: an int (that stored cut) or (axis, position [m]) = the stored cut on that
    axis with the nearest cell-centre position; ValueError when the snapshot holds none there."""
    planes = [next(iter(fields.values())).plane if fields else None for fields in snapshot.cuts]
    stored_text = ', '.join(f'{plane_entry[0]} = {plane_entry[1]:.6g} m' for plane_entry in planes if plane_entry)
    if isinstance(plane, (int, np.integer)) and not isinstance(plane, bool):
        if not -len(planes) <= plane < len(planes):
            raise ValueError(f'stored cut {plane} out of range for {len(planes)} stored cuts ({stored_text or "none"})')
        return int(plane) % len(planes)
    if not (isinstance(plane, (tuple, list)) and len(plane) == 2 and plane[0] in ('x', 'y', 'z')):
        raise ValueError(f"plane must be (axis 'x' | 'y' | 'z', position in m) or a stored cut index, got {plane!r}")
    on_axis = [index for index, plane_entry in enumerate(planes) if plane_entry and plane_entry[0] == plane[0]]
    if not on_axis:
        raise ValueError(f"this snapshot stores no cut normal to {plane[0]!r} (study stored_cuts: "
                         f"{stored_text or 'none'}; the default cut is plane=None)")
    return min(on_axis, key=lambda index: abs(planes[index][1] - float(plane[1])))


@dataclass
class _CurveProfile:
    """Field profile along a line curve: containing-cell samples at distance [m] from start, the sampled cell centres
    (x, y; a 3D segment also z), FieldInfo, height = the z of a 2D line's 3D cut (None otherwise), figure title."""
    start: tuple
    end: tuple
    distance: np.ndarray
    samples: np.ndarray
    centres: tuple
    info: object
    height: float | None
    title: str


class Results:
    """Outcome of run() (live: holds the GPU session until close()) or Results.load (CPU only): status, summary(),
    fields, feature results, figures and save. Live calls use the GPU on the calling thread."""

    def __init__(self, status, spec, out_dir, session, build_summary, log, data=None, log_start=None):
        self.status = status
        self.spec = spec                    # the user's coerced spec (model.json), None when it was invalid
        self.out_dir = out_dir
        self.session = session
        self.build_summary = build_summary
        self.log = log                      # log.txt lines; close-time warnings are appended
        self._data = data if data is not None else ResultsData(
            summary={}, fields={}, results={}, sweep_results=[], snapshots=[], log_lines=[], spec=spec,
            figure_files=[])                # host cache: what save() writes and a closed Results serves
        self._log_start = time.perf_counter() if log_start is None else log_start
        self.build_sequence = None
        if session is not None:
            self.build_sequence = next(_BUILD_SEQUENCE)
            _LIVE_RESULTS.add(self)

    @staticmethod
    def load(path) -> 'Results':
        """CPU-only Results of a saved run directory (FileNotFoundError without summary.json)."""
        directory = Path(path)
        data = load_results(directory)
        return Results(data.summary.get('status', 'unknown'), data.spec, directory, None, None,
                       list(data.log_lines), data=data)

    @property
    def is_live(self) -> bool:
        """True while the GPU session is open."""
        return self.session is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        return False

    def summary(self) -> dict:
        """Deep copy of the summary.json content (JSON-able after results_io.json_safe)."""
        return copy.deepcopy(self._data.summary)

    def field_names(self) -> list:
        """'physics.component' keys: every output field while live, else the saved ones."""
        if self.is_live:
            return [f'{physics}.{info.name}' for physics, info in self.session.available_fields()]
        return list(self._data.fields)

    def result_names(self) -> list:
        """'physics.feature' keys: every built result feature while live, else the saved ones."""
        if self.is_live and self.build_summary is not None:
            return [f'{physics}.{feature}' for physics, feature, _kind in self.build_summary.feature_results]
        return list(self._data.results)

    def field(self, name, physics=None, plane=None, unfold=False, native=False):
        """FieldData on the native window cells (values float32 (nx, ny), x_centers / y_centers float64); name is
        'physics.component' or a component unique among field_names() (or narrowed by physics). 3D models: plane
        (axis 'x' | 'y' | 'z', position [m]) picks the cut (live Results only; saved fields are the mid-height z
        cut). unfold: the field mirrored
        over the model's x / y mirror faces (core.mirror.unfold_field; ValueError for a field without a parity; a
        parity-sector run: the last sector's signs, whose state this is). Polar / spherical domains: a containing-cell
        image over the sector's box, or with native (live only) the (r, angle) window cells (axes ('r', 'phi' |
        'theta'), y_centers in rad, mapped_centres the drawing-plane x / y)."""
        matches = self._field_matches(name, physics)
        if len(matches) != 1:
            raise self._field_lookup_error(name, physics, matches)
        key = matches[0]
        field_physics, _, component = key.partition('.')
        if self.is_live:
            field_data = self.session.field(component, physics=field_physics, plane=plane, native=native)
        elif plane is not None:
            raise ValueError('plane= needs live Results (saved fields are the mid-height z cut; the study '
                             "stored_cuts planes are in the snapshots: snapshot_field(-1, name, plane=...))")
        elif native and self.spec is not None and self.spec.domain.coordinates != 'cartesian' \
                and self.spec.dimensions != 1:   # 1D fields are their cells either way
            raise ValueError('native=True needs live Results: saved polar / spherical fields are images over the '
                             "sector's box")
        else:
            field_data = self._data.fields[key].field_data()
        if not unfold:
            return field_data
        if self.spec is None:
            raise ValueError('unfold=True needs the model spec of the run (missing from this run directory)')
        return unfold_field(field_data, mirror_faces(self._state_spec(), field_physics))

    def _state_spec(self):
        """The spec whose mirror signs the live / saved state has: a parity-sector run's last sector, else self.spec."""
        sectors = self._data.summary.get('parity_sectors')
        if not sectors or not sectors.get('labels'):
            return self.spec
        return dict(ordered_sector_specs(self.spec))[sectors['labels'][-1]]

    def result(self, physics, feature=None, **options) -> dict:
        """Feature result dict of (physics, feature) or one 'physics.feature' key. Live: Session.feature_result(
        **options); otherwise the saved dict, where only freq_index (dft_fields) is accepted (adds freq_index +
        intensity like the live call)."""
        physics, feature = _result_key_parts(physics, feature)
        if self.is_live:
            return self.session.feature_result(physics, feature, **options)
        key = f'{physics}.{feature}'
        stored = self._data.results.get(key)
        if stored is None:
            raise KeyError(f"no saved result {key!r}; saved: {', '.join(self._data.results) or 'none'}")
        freq_index = options.pop('freq_index', None)
        if options:
            raise ValueError(f'result options {sorted(options)} need a live session; a closed or loaded Results '
                             f'only takes freq_index (dft_fields)')
        result = {name: copy.deepcopy(value) for name, value in stored.items() if name != 'intensity_all'}
        if freq_index is None:
            return result
        if stored.get('kind') != 'dft_fields':
            raise ValueError(f"freq_index applies to dft_fields results; {key!r} is a {stored.get('kind')} result")
        intensity_all = stored.get('intensity_all')
        if intensity_all is None:
            raise ValueError(f'{key!r} has no saved DFT intensities (saved without arrays)')
        if isinstance(freq_index, bool) or not isinstance(freq_index, (int, np.integer)):
            raise TypeError(f'freq_index must be an int, got {freq_index!r}')
        if not 0 <= int(freq_index) < intensity_all.shape[0]:
            raise IndexError(f'freq_index {freq_index} out of range for {intensity_all.shape[0]} DFT frequencies')
        result['freq_index'] = int(freq_index)
        result['intensity'] = np.array(intensity_all[int(freq_index)], dtype=np.float64)
        return result

    def snapshots(self) -> list:
        """[(time [s], label, ['physics.component' keys])] of the stored snapshots, live, closed or loaded alike
        (frequency_domain: one per wavelength, label 'lambda = <nm> nm'; time snapshots: label None); 3D stored cut
        planes: summary()['stored_cuts']."""
        return [(snapshot.time, snapshot.label, list(snapshot.fields)) for snapshot in self._stored_snapshots()]

    def snapshot_field(self, index, name, physics=None, plane=None):
        """FieldData of snapshot index (as in snapshots(); negative counts from the end), name as in field(): native
        window cells with x_centers / y_centers up to 250,000 cells, else nearest-cell resampled (x_centers None).
        3D: plane None = the mid-height z cut, (axis, position [m]) = the study stored_cuts plane on that axis nearest
        position (FieldData.plane says where), an int = that stored cut."""
        stored = self._stored_snapshots()
        if not -len(stored) <= index < len(stored):
            raise IndexError(f'snapshot index {index} out of range for {len(stored)} snapshots')
        return self._snapshot_entry_field(stored[index], name, physics, plane)

    def _snapshot_entry_field(self, snapshot, name, physics, plane):
        """FieldData of name in one Snapshot (plane: as snapshot_field)."""
        snapshot_fields = snapshot.fields if plane is None else snapshot.cuts[_stored_cut_index(snapshot, plane)]
        matches = self._field_matches(name, physics, list(snapshot_fields))
        if len(matches) != 1:
            raise self._field_lookup_error(name, physics, matches, list(snapshot_fields))
        return snapshot_fields[matches[0]]

    def volume(self, name, physics=None, max_bytes=VOLUME_MAX_BYTES):
        """session.VolumeData (values float32 (wx, wy, wz), x / y / z cell centres) of a 3D field on the window cells,
        name as in field(); live Results only, ValueError above max_bytes of host values (default 1 GiB)."""
        if not self.is_live:
            raise ValueError('volume() needs live Results (run directories hold cut planes only)')
        matches = self._field_matches(name, physics)
        if len(matches) != 1:
            raise self._field_lookup_error(name, physics, matches)
        field_physics, _, component = matches[0].partition('.')
        return self.session.volume(component, physics=field_physics, max_bytes=max_bytes)

    def export_volume(self, name, path, physics=None, max_bytes=VOLUME_MAX_BYTES) -> Path:
        """Write volume(name, physics, max_bytes) as .npz (io.export_volume: values, x / y / z centres, metadata);
        returns the path."""
        path = Path(path)
        export_volume(self.volume(name, physics=physics, max_bytes=max_bytes), path)
        return path

    def sweep_result(self, physics, feature=None) -> list:
        """Result dict of (physics, feature) or 'physics.feature' per sweep value (summary()['sweep']['values']
        order; None for a failed value), live, closed or loaded alike."""
        physics, feature = _result_key_parts(physics, feature)
        key = f'{physics}.{feature}'
        sweep_results = self._data.sweep_results
        if not any(key in value_results for value_results in sweep_results):
            saved = sorted({name for value_results in sweep_results for name in value_results})
            raise KeyError(f"no sweep result {key!r}; sweep results: {', '.join(saved) or 'none (no sweep)'}")
        return [copy.deepcopy(value_results[key]) if key in value_results else None
                for value_results in sweep_results]

    def sweep_xy(self, physics, feature=None, column='frequency', sort=True) -> dict:
        """'xy' result of a per-sweep-value 'table' result vs the swept parameter (analytic.sweep_table_xy): series
        '<column>_<k>' = the k-th smallest (sort) or k-th row value, e.g. the bands of a resonance_monitor k sweep."""
        sweep = self._sweep_summary()
        return sweep_table_xy(sweep['parameter'], sweep['values'], self.sweep_result(physics, feature), column, sort)

    def sweep_richardson(self, name) -> dict:
        """Richardson fit (analytic.richardson_fit: extrapolated, order, coefficient, error estimates) of derived value
        name vs the swept parameter (a step size or cell count > 0)."""
        sweep = self._sweep_summary()
        return sweep_richardson(sweep['parameter'], sweep['values'], sweep['derived'] or {}, name)

    def sweep(self) -> dict:
        """Copy of the summary's sweep table: parameter, values, derived (+ reference / error), converged, failed and
        derived_series per value; KeyError without a sweep."""
        sweep = self._sweep_summary()
        return copy.deepcopy({key: sweep.get(key) for key in ('parameter', 'values', 'derived', 'reference', 'error',
                                                              'converged', 'failed', 'derived_series')})

    def convergence(self) -> dict:
        """Copy of the mesh refinement report of a StudySpec.adaptation run (convergence.json: target, levels, Richardson
        estimate, converged, stop_reason, adapted meshes); KeyError without one."""
        report = self._data.summary.get('convergence')
        if not report:
            raise KeyError('the run has no mesh refinement levels (StudySpec.adaptation)')
        return copy.deepcopy(report)

    def sweep_field(self, index, name, snapshot=-1, physics=None, plane=None):
        """FieldData of a kept snapshot (SweepSpec.keep_fields) of sweep value index (summary()['sweep']['values']
        order); snapshot, name, physics and plane as in snapshot_field; live, closed or loaded alike."""
        kept = self._data.sweep_snapshots
        if not kept:
            raise KeyError('no kept sweep snapshots: run the sweep with keep_fields true')
        if not -len(kept) <= index < len(kept):
            raise IndexError(f'sweep value index {index} out of range for {len(kept)} sweep values')
        snapshots = kept[index]
        if not -len(snapshots) <= snapshot < len(snapshots):
            raise IndexError(f'snapshot {snapshot} out of range for the {len(snapshots)} kept snapshots of sweep value '
                             f'{index}')
        return self._snapshot_entry_field(snapshots[snapshot], name, physics, plane)

    def sweep_figure(self, quantity, path, series=None, title='') -> str:
        """PNG of a sweep: quantity 'physics.feature' = that result per swept value, one curve each
        (analytic.sweep_overlay_xy; series picks an xy / flux_spectrum series), 'derived' = every derived value vs the
        swept parameter, 'derived_series:<name>' = that derived value along the study axis per swept value."""
        sweep = self._sweep_summary()
        parameter, values = sweep['parameter'], sweep['values']
        if quantity == 'derived':
            order = np.argsort(np.asarray(values, dtype=np.float64))
            xy = {'kind': 'xy', 'x': np.asarray(values, dtype=np.float64)[order], 'x_label': parameter, 'x_unit': '',
                  'y_label': 'Derived value',
                  'series': {name: np.asarray(column, dtype=np.float64)[order]
                             for name, column in sweep['derived'].items()},
                  'series_units': dict.fromkeys(sweep['derived'], '')} if sweep['derived'] else None
            heading = f'Derived values vs {parameter}'
        elif quantity.startswith(_DERIVED_SERIES_QUANTITY):
            name = quantity[len(_DERIVED_SERIES_QUANTITY):]
            xy = sweep_overlay_xy(parameter, values, sweep.get('derived_series') or [], name)
            heading = f'{name} along the study, per {parameter}'
        else:
            xy = sweep_overlay_xy(parameter, values, self.sweep_result(quantity), series or '')
            heading = f"{quantity}{f' {series}' if series else ''} per {parameter}"
        if xy is None:
            raise ValueError(f"no sweep value has a curve of {quantity!r}{f' series {series!r}' if series else ''}")
        return save_result_figure(xy, path, title=title or heading)

    def _sweep_summary(self):
        sweep = self._data.summary.get('sweep')
        if not sweep:
            raise KeyError('the run has no parameter sweep')
        return sweep

    def _figure_jobs(self, series=None):
        """[(figure stem, render(path))] of every figure of a run (api.run figures=True, CLI show --figures): saved
        fields and feature results, derived_series, the sweep overlays, derived_<plot> per derived curve plot,
        <key>_vs_<x> per table curve, curve_<name> per line curve, a 3D run's scene_3d; series (CLI --series): only
        that series of the xy results and sweep overlays that have it."""
        data = self._data
        curves = [] if self.spec is None else self.spec.analytic_curves
        line_curves = [curve for curve in curves if curve.target == 'line'
                       and (self.session is not None or curve.snapshot is not None)]
        table_plots = list(dict.fromkeys(   # (result key, x column) of the curves on a table result, in curve order
            (curve.target, curve.x_column) for curve in curves
            if (data.results.get(curve.target) or {}).get('kind') == 'table'))
        derived_plots = [(plot, xy) for plot, xy, _errors in self.derived_curves() if xy is not None]
        derived_series = data.summary.get('derived_series')
        xy_with_series = {key for key, result in data.results.items()
                          if series and result.get('kind') == 'xy' and series in result.get('series', {})}
        sweep_with_series = {key for value_results in data.sweep_results for key, result in value_results.items()
                             if series and sweep_value_curve(result, series) is not None}
        return ([(key, lambda path, key=key: self._field_figure(key, path)) for key in data.fields]
                + [(key, lambda path, key=key: self._result_figure(
                    key, path, series=series if key in xy_with_series else None)) for key in data.results]
                + ([('derived_series', lambda path, xy=derived_series: save_result_figure(
                    xy, path, title='Derived values along the study'))] if derived_series else [])
                + [(stem, lambda path, quantity=quantity: self.sweep_figure(
                    quantity, path, series=series if quantity in sweep_with_series else None))
                   for stem, quantity in self._sweep_figure_jobs()]
                + [(f'derived_{plot}', lambda path, plot=plot, xy=xy: save_result_figure(xy, path, title=plot))
                   for plot, xy in derived_plots]
                + [(f"{key}_vs_{x_column or 'row'}",
                    lambda path, key=key, x_column=x_column: self._table_curve_figure(key, x_column, path))
                   for key, x_column in table_plots]
                + [(f'curve_{curve.name}', lambda path, curve=curve: self._line_curve_figure(curve, path))
                   for curve in line_curves]
                + [(stem, lambda path, stem=stem, render=render: render(stem, path))
                   for stem, render in _scene_figure_jobs(self)])

    def _sweep_figure_jobs(self):
        """[(figure stem, sweep_figure quantity)] of a sweep run: sweep_derived, sweep_<key> per result with a curve,
        sweep_series_<name> per derived-series name; [] without a sweep."""
        sweep = self._data.summary.get('sweep')
        if not sweep:
            return []
        jobs = [('sweep_derived', 'derived')] if sweep.get('derived') else []
        keys = dict.fromkeys(key for value_results in self._data.sweep_results for key, result in value_results.items()
                             if sweep_value_curve(result) is not None)
        names = dict.fromkeys(name for xy in sweep.get('derived_series') or [] if xy for name in xy['series'])
        return (jobs + [(f'sweep_{key}', key) for key in keys]
                + [(f'sweep_series_{name}', f'{_DERIVED_SERIES_QUANTITY}{name}') for name in names])

    def sector_result(self, physics, feature=None, combined=False):
        """Result dict of (physics, feature) or 'physics.feature' per parity sector of a parity_sectors run
        (summary()['parity_sectors']['labels'] order; None where a sector has none), live, closed or loaded alike;
        combined=True: the one stored result of the drawn source alone (core.mirror.combine_sector_result: far fields
        per angle, fluxes symmetric about every plane, membrane probes, ports and mode projections), KeyError where no
        rule combined it."""
        physics, feature = _result_key_parts(physics, feature)
        key = f'{physics}.{feature}'
        if combined:
            combined_results = self._data.sector_combined_results
            if key not in combined_results:
                raise KeyError(f"no combined sector result {key!r}; combined: "
                               f"{', '.join(sorted(combined_results)) or 'none'} (far_field, flux_spectrum on "
                               'a line / plane / box reaching every mirror plane, membrane field_probe / port / '
                               'mode_projection, of a run where every sector ran)')
            return copy.deepcopy(combined_results[key])
        sector_results = self._data.sector_results
        if not any(key in results for results in sector_results):
            saved = sorted({name for results in sector_results for name in results})
            raise KeyError(f"no sector result {key!r}; sector results: "
                           f"{', '.join(saved) or 'none (no parity sectors)'}")
        return [copy.deepcopy(results[key]) if key in results else None for results in sector_results]

    def derived_curves(self) -> list:
        """[(plot, 'xy' result or None, {curve name: error text})] of the spec's derived curves
        (analytic.derived_curve_plots) over the final feature results and the last build's parameters."""
        if self.spec is None or not self.spec.derived_curves:
            return []
        return derived_curve_plots(self.spec.derived_curves, self._cached_result, self._built_parameters())

    def figure(self, what, path, title='', freq_index=None, series=None) -> str:
        """PNG (viz.result_plots) of a field ('physics.component' or a unique component; fields win) or a feature
        result 'physics.feature' (freq_index: DFT frequency, default the brightest; series: only that series of an xy
        result); returns the path. Sweep overlays: sweep_figure."""
        matches = self._field_matches(what, None)
        if len(matches) > 1:
            raise self._field_lookup_error(what, None, matches)
        if matches:
            if freq_index is not None or series is not None:
                raise ValueError(f'freq_index and series apply to feature results, not to the field {matches[0]!r}')
            return self._field_figure(matches[0], path, title)
        if what in self.result_names():
            return self._result_figure(what, path, title, freq_index, series)
        raise KeyError(f"no field or feature result {what!r}; fields: {', '.join(self.field_names()) or 'none'}; "
                       f"results: {', '.join(self.result_names()) or 'none'}")

    def scene_figure(self, path, field=None, physics=None, planes=None, z_stretch=1.0, view=None) -> str:
        """3D models: api.scene_figure of the run's model (built parameters) with field (name as in field(); None =
        objects only) on cut planes in one colour scale: live, planes [(axis, position [m])] (default the middle window
        cell on x, y and z); closed / loaded, the saved mid-height z cut plus the stored cuts of a last snapshot at
        the same time."""
        if self.spec is None:
            raise ValueError('scene_figure needs the model spec of the run (missing from this run directory)')
        if self.spec.domain.size_z is None:
            raise ValueError('scene_figure needs a 3D model (domain size_z set)')
        field_planes = [] if field is None else self._scene_planes(field, physics, planes)
        return _save_scene_figure(self.spec, self._built_parameters(), path, field_planes=field_planes,
                                  z_stretch=z_stretch, view=view)

    def _scene_planes(self, name, physics, planes):
        """FieldData cuts of scene_figure (see there); ValueError when none is available."""
        if self.is_live:
            if planes is None:
                planes = [(axis, centres[len(centres) // 2]) for axis in ('x', 'y', 'z')
                          for centres in (self.session.cut_centers(axis),) if centres]
            return [self.field(name, physics, plane=tuple(plane)) for plane in planes]
        if planes is not None:
            raise ValueError('planes= needs live Results (saved fields are the mid-height z cut; the study '
                             'stored_cuts planes come with the snapshots)')
        field_planes = [self.field(name, physics)]
        stored = self._stored_snapshots()
        same_state = bool(stored) and stored[-1].time == field_planes[0].time   # never mix times in one picture
        for cut in stored[-1].cuts if same_state else ():
            matches = self._field_matches(name, physics, list(cut))
            if len(matches) == 1:
                field_planes.append(cut[matches[0]])
        field_planes = [field_data for field_data in field_planes if field_data.plane is not None]
        if not field_planes:
            raise ValueError(f'no saved cut plane of field {name!r} (an older run directory without plane records)')
        return field_planes

    def save(self, path, save_arrays=True) -> Path:
        """Write summary.json, log.txt, model.json and (save_arrays) results.npz into the directory path, copying the
        figures/*.png of out_dir (missing ones are unlisted); a live Results first stacks every DFT frequency into
        'intensity_all'. Returns the directory."""
        return self._save(Path(path), save_arrays, {})

    def _save(self, directory, save_arrays, written_files):
        """save() body; written_files (dict) receives 'arrays' / 'model' -> file name as save_results writes them."""
        if save_arrays and self.is_live:
            for key in self._unstacked_dft_keys():
                self._stack_dft(key)
        self._data.log_lines = list(self.log)
        data = self._data
        if self.out_dir is not None and directory.resolve() != Path(self.out_dir).resolve():
            # a summary copy: save_results sets its 'files', which must keep describing out_dir
            data = replace(data, figure_files=self._copy_figures(directory), summary=dict(data.summary))
        save_results(directory, data, save_arrays=save_arrays, written_files=written_files)
        return directory

    def close(self, copy_to_host=True):
        """Copy the stored fields and feature results (DFT: every frequency) to host memory unless copy_to_host is
        False, then close the GPU session (leak check); afterwards CPU-only. Idempotent."""
        self._close(copy_to_host)

    # ---- internal ----
    def _close(self, copy_to_host):
        if self.session is None:
            return
        problems = self._copy_to_host() if copy_to_host else []
        lines = [_log_line(self._log_start, 'warning', f'before closing: {text}') for text in problems]
        lines += self._release()
        if lines:
            self.log.extend(lines)
            self._append_to_log_file(lines)

    def _copy_to_host(self):
        """Cache every stored field and built feature result still missing, then stack the DFT intensities; returns
        the failures as text (the session is closed regardless)."""
        session, data = self.session, self._data
        problems = []
        for key in data.summary.get('stored_fields') or []:
            if key in data.fields:
                continue
            physics, _, component = key.partition('.')
            try:
                data.fields[key] = _host_field_record(session, physics, component)
            except Exception as exc:   # recorded as text only (rule: no exception kept across the release)
                problems.append(f'field {key}: {_error_text(exc)}')
        for key in self.result_names():
            if key in data.results:
                continue
            physics, _, feature = key.partition('.')
            try:
                data.results[key] = session.feature_result(physics, feature)
            except Exception as exc:
                problems.append(f'result {key}: {_error_text(exc)}')
        for key in self._unstacked_dft_keys():
            try:
                self._stack_dft(key)
            except Exception as exc:
                problems.append(f'DFT intensities of {key}: {_error_text(exc)}')
        return problems

    def _release(self):
        """study.release_session on the session -> its log lines (leak warnings). The leak check runs only when no
        live api Results was built later (sessions built earlier sit inside this one's baseline)."""
        session, self.session = self.session, None
        _LIVE_RESULTS.discard(self)
        built_later = any(other.build_sequence > self.build_sequence for other in list(_LIVE_RESULTS))
        lines = []

        def keep_log_line(event):
            if isinstance(event, LogEvent):
                lines.append(_log_line(self._log_start, event.level, event.message))

        release_session(None, session, emit=None if built_later else keep_log_line)
        return lines

    def _copy_figures(self, directory):
        """Copy the listed figures from out_dir into directory; returns the relative paths that existed and were
        copied."""
        copied = []
        for relative in self._data.figure_files:
            source = Path(self.out_dir) / relative
            if not source.is_file():
                continue
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(relative)
        return copied

    def _append_to_log_file(self, lines):
        """Append close-time lines to <out_dir>/log.txt (a failure is noted in self.log only)."""
        if self.out_dir is None:
            return
        log_path = self.out_dir / 'log.txt'
        try:
            with open(log_path, 'a', encoding='utf-8', newline='\n') as handle:   # LF, as results_io.save_results
                handle.write(''.join(f'{line}\n' for line in lines))
        except OSError as exc:
            self.log.append(_log_line(self._log_start, 'warning', f'cannot append to {log_path}: {exc}'))

    def _field_matches(self, name, physics, names=None):
        """Entries of names (default field_names()) matching 'physics.component' or a bare component (and physics
        when given)."""
        names = self.field_names() if names is None else names
        if '.' in name:
            matches = [name] if name in names else []
        else:
            matches = [key for key in names if key.partition('.')[2] == name]
        return [key for key in matches if physics is None or key.partition('.')[0] == physics]

    def _field_lookup_error(self, name, physics, matches, names=None):
        names = self.field_names() if names is None else names
        owner = '' if physics is None else f' of physics {physics!r}'
        problem = f'is ambiguous ({", ".join(matches)})' if matches else 'is not available'
        return KeyError(f"field {name!r}{owner} {problem}; fields: {', '.join(names) or 'none'}")

    def _stored_snapshots(self):
        """Host Snapshots: the cache filled after the run (also closed / loaded), else the live session's list."""
        if not self._data.snapshots and self.is_live:
            return self.session.snapshots
        return self._data.snapshots

    def _field_figure(self, key, path, title=''):
        """Field PNG from the host cache (else fetched live); default title the field label and time label, as the
        GUI."""
        record = self._data.fields.get(key)
        if record is None:
            physics, _, component = key.partition('.')
            record = _host_field_record(self.session, physics, component)
        several_meshes = self.spec is not None and len(self.spec.used_meshes()) > 1   # S7.2: name the field's mesh
        return save_field_figure(record.field_data(), path, title=title, show_mesh=several_meshes)

    def _built_parameters(self):
        """Parameters of the session the results come from (summary 'parameters'; an older summary: from the spec)."""
        return built_parameters(self.spec, self._data.summary)

    def _cached_result(self, key):
        """Host-cached result dict of 'physics.feature', else result(key) (fetched live; KeyError when not saved)."""
        result = self._data.results.get(key)
        return result if result is not None else self.result(key)

    def _result_figure(self, key, path, title='', freq_index=None, series=None):
        """Feature-result PNG from the host cache (else fetched live); a live DFT result without stacked intensities
        is fetched at one frequency (the given one, else the brightest); series keeps one series of an xy result."""
        physics, _, feature = key.partition('.')
        result = self._data.results.get(key)
        if result is None:
            result = self.session.feature_result(physics, feature)
        kind = result.get('kind')
        if freq_index is not None and kind != 'dft_fields':
            raise ValueError(f'freq_index applies to dft_fields results; {key!r} is a {kind} result')
        if series is not None:
            if kind != 'xy' or series not in result['series']:
                available = ', '.join(result['series']) if kind == 'xy' else f'none (a {kind} result)'
                raise ValueError(f'{key!r} has no xy series {series!r}; series: {available}')
            result = {**result, 'series': {series: result['series'][series]},
                      'series_units': {series: (result.get('series_units') or {}).get(series, '')}}
        if self.is_live and kind == 'dft_fields' and 'intensity_all' not in result:
            if freq_index is None:
                freq_index = self._brightest_frequency(key, physics, feature, len(result['frequency']))
            result = self.session.feature_result(physics, feature, freq_index=freq_index)
            freq_index = None   # the result now holds the 2D 'intensity' of that frequency
        return save_result_figure(result, path, title=title or key, freq_index=freq_index,
                                  curves=self._analytic_overlays(key, result))

    def _analytic_overlays(self, key, result):
        """[(name, abscissa, values, series)] of the analytic curves targeting result key (far field: per curve one at
        each drawn wavelength, labelled with it; ratio-plot and FFT curves have no figure, table curves their own, flux
        spectra only series '' / 'power': the figure's one panel), over the last build's parameters."""
        kind = result.get('kind')
        curves = [] if self.spec is None else [curve for curve in self.spec.analytic_curves
                                               if curve.target == key and curve.series not in (RATIO_SERIES, FFT_SERIES)
                                               and (kind != 'flux_spectrum' or curve.series in _FLUX_FIGURE_SERIES)]
        if not curves or kind not in CURVE_RESULT_KINDS or kind == 'table':
            return []
        parameters = self._built_parameters()
        if kind == 'far_field':   # save_result_figure colours curve k like far_field_wavelengths(result)[k % n]
            drawn = [float(wavelength) for wavelength in far_field_wavelengths(result)]
            return [(curve.name if len(drawn) == 1 else f'{curve.name}, {wavelength_label(wavelength)}',
                     *feature_curve(curve, result, parameters, wavelength=wavelength), curve.series)
                    for curve in curves for wavelength in drawn]
        return [(curve.name, *feature_curve(curve, result, parameters), curve.series) for curve in curves]

    def _table_curve_figure(self, key, x_column, path):
        """PNG of a table result's numeric columns vs x_column ('' = row) with its curves of that x_column dashed."""
        result = self._data.results[key]
        parameters = self._built_parameters()
        curves = [(curve.name, *feature_curve(curve, result, parameters), curve.series)
                  for curve in self.spec.analytic_curves if curve.target == key and curve.x_column == x_column]
        return save_result_figure(table_xy(result, x_column), path, title=f"{key} vs {x_column or 'row'}",
                                  curves=curves)

    def _line_curve_figure(self, curve, path):
        """PNG of the field profile along an analytic curve's line (final state, its plane or its snapshot) with the
        curve dashed."""
        parameters = self._built_parameters()
        profile = self._line_profile(curve, parameters)
        curve_distance, curve_values = line_curve(curve.expression, parameters, profile.start, profile.end,
                                                  z=profile.height)
        return save_line_profile_figure(profile.distance, profile.samples, profile.info.label, profile.info.unit,
                                        path, title=profile.title, curves=[(curve.name, curve_distance, curve_values)])

    def _line_profile(self, curve, parameters):
        """_CurveProfile of the field along a line curve: final state (its plane; a 3D segment: live Results only) or
        its snapshot; the figure and curve_error share it."""
        start, end = line_endpoints(curve.line, parameters)
        title = f'{curve.field} along {curve.name}'
        if curve.snapshot is not None:
            stored = self._stored_snapshots()
            index = snapshot_index(curve.snapshot, [snapshot.label for snapshot in stored])
            title += f" ({stored[index].label or f't = {stored[index].time:.6g} s'})"
        if is_segment_3d(curve.line):
            if curve.snapshot is not None:   # the profile each snapshot stored (also in a saved run)
                profile = stored[index].segments.get(curve.name)
                if profile is None:
                    raise ValueError(f"snapshot {index} holds no profile of line curve '{curve.name}' (snapshots "
                                     'store the 3D segment line curves of the model they ran)')
            elif not self.is_live:
                raise ValueError('a 3D-segment line curve of the final state needs live Results (the run directory '
                                 'holds cut planes and the snapshot profiles only: give the curve a snapshot)')
            else:
                physics, component = split_key(curve.field, 'physics.component')
                profile = self.session.segment_profile(component, start, end, physics=physics,
                                                       sampling=curve.sampling)
            return _CurveProfile(start, end, profile.distance, profile.values, profile.centres, profile.info, None,
                                 title)
        keys = line_curve_field_keys(curve)   # projected: the y component too
        if curve.snapshot is None:
            # polar / spherical interface curves read the (r, angle) cells themselves when live, else the image
            native = bool(curve.interface) and self.is_live and self.spec.domain.coordinates != 'cartesian'
            plane = line_plane(curve, parameters)
            fields = [self.field(key, plane=plane, native=native) for key in keys]
        else:
            fields = [self.snapshot_field(index, key) for key in keys]
        field_data, info = fields[0], fields[0].info
        if curve.interface:   # one-sided samples at the line points themselves (igor 1.8)
            distance, samples, (centre_x, centre_y) = interface_line_profile(
                self.spec, parameters, curve.interface, field_data, start, end,
                partner_data=fields[-1] if curve.projection else None, projection=curve.projection)
            title += f' ({curve.interface} side)'
            if curve.projection:
                info = projected_info(info, curve.projection)
                title += f', {curve.projection} part'
        else:
            distance, samples, centre_x, centre_y = line_profile(field_data.values, field_data.extent, start, end,
                                                                 field_data.x_centers, field_data.y_centers)
        return _CurveProfile(start, end, distance, samples, (centre_x, centre_y), info,
                             field_plane_height(field_data), title)

    def _brightest_frequency(self, key, physics, feature, n_frequencies):
        """Index of the largest max DFT intensity of the monitor quantity (from the summary digest, else fetched per
        frequency); 0 if none is finite."""
        maxima = (self._data.summary.get('results') or {}).get(key, {}).get('intensity_max')
        if not isinstance(maxima, list) or len(maxima) != n_frequencies:
            maxima = _dft_max_intensity(self.session, physics, feature, n_frequencies)
        values = np.array([math.nan if value is None else float(value) for value in maxima], dtype=np.float64)
        finite = np.isfinite(values)
        return int(np.argmax(np.where(finite, values, -np.inf))) if finite.any() else 0

    def _unstacked_dft_keys(self):
        return [key for key, result in self._data.results.items()
                if result.get('kind') == 'dft_fields' and 'intensity_all' not in result]

    def _stack_dft(self, key):
        """Host float64 (n_f, mx, my) DFT intensity of the monitor quantity at every frequency into the cached
        result's 'intensity_all'."""
        result = self._data.results[key]
        physics, _, feature = key.partition('.')
        result['intensity_all'] = _dft_intensity_stack(self.session, physics, feature, result)


# ---- selftest ----
class _SelftestPlan:
    """The tests of one selftest call in run order, for its progress events: 'begin' per phase (n_run, n_reused),
    'start' per test (1-based index over every phase, last_wall_time, remaining = the summed last run times of it and
    every later test, n_untimed of those never timed, time_estimate of its runs or None), then its finished record
    (with 'time_estimate': before / n_checked_runs / split); a raising callback is dropped."""

    def __init__(self, progress, cache):
        self.callback = progress
        self.cache = cache
        self.phases = []     # (label, n_run, n_reused)
        self.items = []      # (section, key, name, last seconds or None) of every phase in run order
        self.index = 0
        self.failure = None  # warning text of a dropped callback, taken by the phase that saw it

    def add_phase(self, label, tests, n_reused):
        """Plan a phase of tests [(cache section, key, name)] in run order -> its phase number."""
        self.phases.append((label, len(tests), n_reused))
        for section, key, name in tests:
            last = None if self.cache is None else self.cache.last_wall_time(section, key)
            self.items.append((section, key, name, last))
        return len(self.phases) - 1

    def begin(self, phase):
        label, n_run, n_reused = self.phases[phase]
        self.emit({'event': 'begin', 'phase': label, 'n_run': n_run, 'n_reused': n_reused,
                   'n_total': len(self.items)})

    def start(self, time_estimate=None):
        """'start' event of the next planned test (time_estimate: _specs_time_estimate of its runs)."""
        later = [item[3] for item in self.items[self.index:]]
        name, last = self.items[self.index][2:]
        self.index += 1
        self.emit({'event': 'start', 'index': self.index, 'n_total': len(self.items), 'name': name,
                   'last_wall_time': last, 'remaining': sum(seconds for seconds in later if seconds is not None),
                   'n_untimed': sum(1 for seconds in later if seconds is None), 'time_estimate': time_estimate})

    def finish(self, seconds, event):
        """Keep the started test's run time for later estimates, then emit its record (or 'skipped' event)."""
        section, key = self.items[self.index - 1][:2]
        if self.cache is not None:
            self.cache.record_wall_time(section, key, seconds)
        self.emit(event)

    def emit(self, event):
        if self.callback is None:
            return
        try:
            self.callback(event)
        except Exception as exc:   # e.g. a closed stdout pipe: must not end the selftest (as _Collector.emit)
            self.callback = None
            self.failure = f'progress callback failed: {_error_text(exc)}; progress disabled'

    def take_warnings(self):
        failure, self.failure = self.failure, None
        return [] if failure is None else [failure]


def selftest(out_dir=None, names=None, progress=None, heavy=False, reset=False) -> dict:
    """Built-in CPU cases, then the registered cases, never timed first, then by last wall time (names: only those,
    built-ins included only when named; else every built-in and by heavy: False light, True heavy, None all, which
    then also runs the selftest_examples run) -> report dict; out_dir -> the subset's
    report files + run directories (also partial on an interrupt) + CACHE_FILE: an unnamed case that passed with
    unchanged dependencies is reused, not rerun (reset True: rerun all); progress(event): 'begin' per phase, 'start'
    per test, then its case or example record (_SelftestPlan). Estimates price on an out-of-date stored machine profile
    too (marked outdated); only the time_profile_current case checks it."""
    registry = get_registry()
    builtin_names, cases, subset = _selected_cases(registry, names, heavy)
    n_skipped = len(registry.selftests()) - len(cases)
    out_dir = None if out_dir is None else Path(out_dir)
    start = time.perf_counter()
    records, warnings, reused = [], [], []
    all_cases_ran = False
    available_backends = _available_backends()
    cache = _selftest_cache(out_dir, available_backends)
    cases = _untimed_then_fastest(cases, cache, 'cases', lambda case: case.name)
    reuse = cache is not None and not reset and subset != 'named'
    reusable = {}   # case name -> its reused record
    for case in cases if reuse else ():
        record = cache.reusable_record('cases', case.name)
        if record is not None:
            reusable[case.name] = record
    builtins = tuple((name, run_builtin) for name, run_builtin in zip(
        BUILTIN_SELFTESTS, (_templates_validate_case, _examples_validate_case, _api_reference_case,
                            _import_contracts_case, _structure_index_case)) if name in builtin_names)
    plan = _SelftestPlan(progress, cache)
    phase = plan.add_phase(f'{subset} cases', [('cases', name, name) for name, _ in builtins]
                           + [('cases', case.name, case.name) for case in cases if case.name not in reusable],
                           len(reusable))
    examples_plan = _plan_examples(None, cache, cache is not None and not reset, plan) if subset == 'all' else None
    skipped_backend = []   # cases whose every run needs a backend this machine lacks (no usable GPU)
    interrupt = None   # the KeyboardInterrupt in flight: gets the partial report as .partial_report (CLI counts)
    plan.begin(phase)
    try:
        for _, run_builtin in builtins:
            plan.start()
            record = run_builtin(registry)
            records.append(record)
            plan.finish(record['wall_time'], record)
        for case in cases:
            if case.name in reusable:
                reused.append(case.name)
                records.append(reusable[case.name])
                continue
            before = _specs_time_estimate(_case_estimate_specs(case, available_backends, registry), registry,
                                          save_arrays=out_dir is not None)
            plan.start(before)
            case_start = time.perf_counter()
            from .fem.gmsh3d import default_disk_cache_off   # entered before the trace: no gmsh disk hit hides code
            with default_disk_cache_off(), DependencyTrace() as trace:
                record = _registered_case(case, out_dir, available_backends, registry)
            if record is not None:
                run_records = [*record['runs'].values(), *(run for variant in (record.get('variants') or {}).values()
                                                           for run in variant['runs'].values())]
                record['time_estimate'] = _time_estimate_after(before, run_records, record['wall_time'])
                _set_worst_deviations(record, run_records)
            if record is None:
                skipped_backend.append(case.name)
                plan.finish(time.perf_counter() - case_start,
                            {'event': 'skipped', 'name': case.name,
                             'message': 'needs the gpu backend (cupy and a CUDA device), not usable here'})
                continue
            if cache is not None:
                cache.store('cases', case.name, record, trace.units())
            records.append(record)
            plan.finish(record['wall_time'], record)
        all_cases_ran = True
    except KeyboardInterrupt as exc:
        interrupt = exc
        raise
    finally:   # an interrupt (KeyboardInterrupt) still writes the cases already run
        warnings.extend(plan.take_warnings())
        if cache is not None:
            cache.write()
        if not all_cases_ran:
            warnings.append(f'selftest interrupted after {len(records)} cases; the report is partial')
        if skipped_backend:
            warnings.append(f'{len(skipped_backend)} cases skipped: they need the gpu backend (cupy and a CUDA '
                            f'device), not usable here')
        variants_only = [record['name'] for record in records if record.get('variants_only')]
        if variants_only:
            warnings.append(f"{len(variants_only)} cases judged by their variants only (their own gpu runs skipped): "
                            f"{', '.join(variants_only)}")
        counts = _case_counts(records)
        report = {'open_sol_version': __version__, 'passed': all_cases_ran and counts['n_passed'] == len(records),
                  'n_cases': len(records), **counts, 'subset': subset, 'n_skipped': n_skipped + len(skipped_backend),
                  'skipped_backend': skipped_backend, 'n_reused': len(reused), 'reused': reused,
                  'backends': list(available_backends),
                  'wall_time': time.perf_counter() - start, 'device': _device_record(), 'warnings': warnings,
                  'cases': records}
        report['report_files'] = ([] if out_dir is None
                                  else _write_selftest_reports(out_dir, report, registry, all_cases_ran))
        if interrupt is not None:
            interrupt.partial_report = report
    if examples_plan is not None:   # everything: then the examples (the case reports are already written)
        _add_example_run(report, out_dir, plan, examples_plan, cache)
    return report


def _selftest_cache(out_dir, available_backends):
    """SelftestCache of out_dir/CACHE_FILE for this machine's environment; None without out_dir."""
    if out_dir is None:
        return None
    device = _device_record()
    return SelftestCache(out_dir / CACHE_FILE, environment_key(available_backends, device and device['name']))


def _add_example_run(report, out_dir, plan, examples_plan, cache):
    """selftest all mode: the planned examples run once -> report 'examples' (its counts), its report file appended;
    passed only when the examples run passed too."""
    count_keys = ('passed', 'n_examples', 'n_passed', 'n_failed', 'n_error', 'n_no_tolerance', 'n_skipped_backend',
                  'n_reused', 'wall_time', 'warnings')
    try:
        examples_report = _run_examples(out_dir, examples_plan, plan, cache, time.perf_counter())
    except KeyboardInterrupt as interrupt:   # the CLI prints the case counts and the partial examples counts
        examples_report = interrupt.partial_report
        report['examples'] = {key: examples_report[key] for key in count_keys}
        report['report_files'] = report['report_files'] + examples_report['report_files']
        report['passed'] = False
        interrupt.partial_report = report
        raise
    report['examples'] = {key: examples_report[key] for key in count_keys}
    report['report_files'] = report['report_files'] + examples_report['report_files']
    report['passed'] = report['passed'] and examples_report['passed']


def _write_selftest_reports(out_dir, report, registry, all_cases_ran):
    """Write the report to its subset's file ('all' -> the light and the heavy file, each with the built-ins, its own
    cases and counts) -> the written file names."""
    if report['subset'] != 'all':
        name = SELFTEST_REPORT_FILES[report['subset']]
        _write_report_keeping_timings(out_dir / name, report)
        return [name]
    heavy_names = {case.name for case in registry.selftests() if case.heavy}
    names = []
    for subset, keep_heavy in (('light', False), ('heavy', True)):
        records = [record for record in report['cases']
                   if record['name'] in BUILTIN_SELFTESTS or (record['name'] in heavy_names) == keep_heavy]
        counts = _case_counts(records)
        n_registered = sum(1 for case in registry.selftests() if case.heavy == keep_heavy)
        skipped_backend = [name for name in report['skipped_backend'] if (name in heavy_names) == keep_heavy]
        reused = [name for name in report['reused'] if (name in heavy_names) == keep_heavy]
        part = dict(report, subset=subset, cases=records, n_cases=len(records), **counts,
                    passed=all_cases_ran and counts['n_passed'] == len(records),
                    n_skipped=len(registry.selftests()) - n_registered + len(skipped_backend),
                    skipped_backend=skipped_backend, n_reused=len(reused), reused=reused)
        names.append(SELFTEST_REPORT_FILES[subset])
        _write_report_keeping_timings(out_dir / names[-1], part)
    return names


def _write_report_keeping_timings(path, report):
    """Write a selftest / examples report; an existing one first leaves its timing summary in <stem>_prev.json."""
    if path.exists():
        try:
            previous_timings = _report_timings(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:   # an unreadable old report never stops the write
            previous_timings = {'source': path.name, 'error': f'{type(exc).__name__}: {exc}'}
        write_json(path.with_name(f'{path.stem}_prev.json'), previous_timings)
    write_json(path, report)


def _report_timings(path):
    """Timing summary of a report file: file time, version, subset, counts, wall_time, device and per case (or
    example) {status, wall_time, reused}."""
    old_report = read_json(path)
    reused_names = set(old_report.get('reused') or [])
    records = old_report['cases'] if 'cases' in old_report else old_report.get('examples', [])
    case_timings = {}
    for record in records:
        name = record.get('name', record.get('example'))
        case_timings[name] = {'status': record.get('status'), 'wall_time': record.get('wall_time'),
                              'reused': name in reused_names}
    summary_keys = ('open_sol_version', 'subset', 'passed', 'n_cases', 'n_examples', 'n_passed',
                    'n_failed', 'n_error', 'n_reused', 'wall_time', 'device')
    file_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(path.stat().st_mtime))
    return {'source': path.name, 'written': file_time, **{key: old_report[key] for key in summary_keys
                                                          if key in old_report}, 'cases': case_timings}


def _untimed_then_fastest(items, cache, section, key_of):
    """items in run order: never timed first, then by the cache's last wall time (ties keep the given order); no
    cache: as given."""
    if cache is None:
        return list(items)
    seconds = [cache.last_wall_time(section, key_of(item)) for item in items]
    order = sorted(range(len(items)), key=lambda index: (seconds[index] is not None, seconds[index] or 0.0))
    return [items[index] for index in order]


def _case_counts(records):
    """{n_passed, n_failed, n_error} of selftest case records by status (as the analytic report counts)."""
    statuses = [record['status'] for record in records]
    return {'n_passed': statuses.count('passed'), 'n_failed': statuses.count('failed'),
            'n_error': len(statuses) - statuses.count('passed') - statuses.count('failed')}


def _selected_cases(registry, names, heavy):
    """(built-in names to run, registered SelfTestCases in registration order, report subset): only the named ones
    ('named'), else every built-in and heavy False / True / None -> 'light' / 'heavy' / 'all'; KeyError for unknown."""
    cases = registry.selftests()
    if names is None:
        if heavy is None:
            return BUILTIN_SELFTESTS, cases, 'all'
        if not isinstance(heavy, bool):
            raise TypeError(f'heavy must be False, True or None, got {heavy!r}')
        return BUILTIN_SELFTESTS, [case for case in cases if case.heavy == heavy], 'heavy' if heavy else 'light'
    names = [names] if isinstance(names, str) else list(names)
    registered = [case.name for case in cases]
    unknown = [name for name in names if name not in registered and name not in BUILTIN_SELFTESTS]
    if unknown:
        raise KeyError(f"unknown selftest {', '.join(map(str, unknown))}; available: "
                       f"{', '.join([*BUILTIN_SELFTESTS, *registered])}")
    return ([name for name in BUILTIN_SELFTESTS if name in names], [case for case in cases if case.name in names],
            'named')


def _case_record(name, label, description):
    """Selftest report case entry before it ran (status 'error' until set)."""
    return {'name': name, 'label': label, 'description': description, 'status': 'error', 'measured': None,
            'expected': None, 'error': None, 'tolerance': None, 'details': {}, 'message': '', 'wall_time': 0.0,
            'runs': {}}


def _set_outcome(record, measured, expected, error, tolerance, details):
    """Fill a case record from a comparison (values made JSON-able: a TypeError here makes the case an 'error');
    passed when the error is finite and <= tolerance."""
    passed = math.isfinite(error) and error <= tolerance
    record.update(status='passed' if passed else 'failed', measured=json_safe(measured),
                  expected=json_safe(expected), error=error, tolerance=tolerance, details=json_safe(details),
                  message='' if passed else f'error {error:.4g} is not <= tolerance {tolerance:.4g}')


def _set_error(record, exc):
    """Mark a case record 'error' (call inside the except block: it formats the current traceback)."""
    record.update(status='error', message=_error_text(exc), details={'traceback': traceback.format_exc()})


def _builtin_case(name, label, description):
    """Decorator of a built-in case body(registry, record) -> run(registry) returning its record: a fresh _case_record,
    wall_time set."""
    def decorate(body):
        def run_case(registry):
            start = time.perf_counter()
            record = _case_record(name, label, description)
            try:
                body(registry, record)
            except Exception as exc:   # an 'error' case (describe's doc-check RuntimeError too); the others still run
                _set_error(record, exc)
            record['wall_time'] = time.perf_counter() - start
            return record
        return run_case
    return decorate


@_builtin_case('templates_validate', 'Templates validate',
               'Every template of model.templates.template_entries() validates without errors (CPU). '
               'error = number of validation errors (a failing factory counts as one), tolerance 0.')
def _templates_validate_case(registry, record):
    """Built-in case: every template (built-in and registered) validates without errors on the CPU."""
    details = {}
    for entry in template_entries(registry):
        try:
            issues = validate(entry.factory(), registry)
        except Exception as exc:   # a crashing factory is a template error, not a selftest crash
            issues = [Issue('error', ('model',), f'the template factory failed: {_error_text(exc)}')]
        errors = [_issue_record(issue) for issue in issues if issue.level == 'error']
        if errors:
            details[entry.key] = errors
    n_errors = sum(len(errors) for errors in details.values())
    _set_outcome(record, n_errors, 0, float(n_errors), 0.0, details)
    if n_errors:
        record['message'] = f"validation errors in the templates {', '.join(details)}"


def _example_model_paths():
    """Sorted examples/**/*.json, skipping CLI run directories ('*_out') and hidden folders (the GUI catalog's rule)."""
    model_paths = []
    for folder, subfolder_names, file_names in os.walk(EXAMPLES_DIR):
        subfolder_names[:] = [name for name in subfolder_names if not (name.endswith('_out') or name.startswith('.'))]
        model_paths.extend(Path(folder) / name for name in file_names if name.endswith('.json'))
    return sorted(model_paths)


@_builtin_case('examples_validate', 'Examples validate',
               'Every model file under examples/ (GUI Examples menu) loads and validates without errors '
               '(CPU). error = number of failing files, tolerance 0.')
def _examples_validate_case(registry, record):
    """Built-in case: every examples/**/*.json loads and validates without errors (passes without an examples
    folder, e.g. an installed package)."""
    details = {}
    model_paths = _example_model_paths() if EXAMPLES_DIR.is_dir() else []
    for path in model_paths:
        relative_path = path.relative_to(EXAMPLES_DIR).as_posix()
        try:
            issues = validate(load_model(path), registry)
        except Exception as exc:   # a file that fails to load is a failing example, not a selftest crash
            issues = [Issue('error', ('model',), f'loading failed: {_error_text(exc)}')]
        errors = [_issue_record(issue) for issue in issues if issue.level == 'error']
        if errors:
            details[relative_path] = errors
    _set_outcome(record, len(details), 0, float(len(details)), 0.0, {'n_files': len(model_paths), 'failing': details})
    if details:
        record['message'] = f"examples with load or validation errors: {', '.join(details)}"


@_builtin_case('api_reference_current', 'API reference current',
               f'{REFERENCE_FILE} and {TEMPLATES_FILE} (what agents read) equal the text describe '
               'generates from the registry. error = number of differing lines (both files), tolerance 0.')
def _api_reference_case(registry, record):
    """Built-in case: api_reference.md and api_templates.md equal describe's reference_files() texts (line endings,
    trailing newlines aside)."""
    details, messages, n_differing = {}, [], 0
    for name, text in reference_files(registry).items():
        path = API_REFERENCE_PATH.parent / name
        file_details = details[name] = {'path': str(path)}
        if not path.is_file():
            file_details['missing'] = True
            messages.append(f'{name} is missing')
            n_differing += len(_normalized_lines(text))
            continue
        n_file, first = _first_difference(_normalized_lines(path.read_text(encoding='utf-8')), _normalized_lines(text))
        n_differing += n_file
        if n_file:
            file_details.update(n_differing=n_file, **first)
            messages.append(f"{name} differs from the generated text at line {first['first_line']}: file "
                            f"{first['file_line']!r}, generated {first['generated_line']!r}")
    _set_outcome(record, n_differing, 0, float(n_differing), 0.0, details)
    if messages:
        record['message'] = '; '.join(messages + [_REGENERATE_REFERENCE])


@_builtin_case('import_contracts', 'Import contracts',
               'Every import of the package keeps the layering / lazy-import rules of '
               'model/import_contracts.py, and no rule there is stale (an importer pattern matching no '
               'module, an exception no import uses). error = violations + stale rules, tolerance 0.')
def _import_contracts_case(registry, record):
    """Built-in case: every package import keeps the import contracts of model.import_contracts (AST scan, CPU)."""
    report = check_import_contracts()
    n_violations = sum(len(found) for found in report['violations'].values())
    n_problems = n_violations + len(report['stale'])
    _set_outcome(record, n_problems, 0, float(n_problems), 0.0, report)
    messages = [f"{name} ({rule_text(name)}): {'; '.join(found[:3])}{' ...' if len(found) > 3 else ''}"
                for name, found in report['violations'].items()]
    messages += [f'stale rule {text}' for text in report['stale']]
    if messages:
        record['message'] = ' | '.join(messages)


@_builtin_case('structure_index_current', 'Structure index current',
               f'{STRUCTURE_INDEX_PATH.name} (the module index agents read) equals the text generated from '
               'the module docstrings. error = number of differing lines, tolerance 0.')
def _structure_index_case(registry, record):
    """Built-in case: structure_index.md equals the text generated from the module docstrings (passes outside a
    repository checkout)."""
    if not CONVENTIONS_PATH.is_file():   # an installed package: no repository documents to keep current
        _set_outcome(record, 0, 0, 0.0, 0.0, {'note': 'not a repository checkout'})
        return
    saved = (_normalized_lines(STRUCTURE_INDEX_PATH.read_text(encoding='utf-8'))
             if STRUCTURE_INDEX_PATH.is_file() else [])
    n_differing, first = _first_difference(saved, _normalized_lines(structure_index_text()))
    details = {'path': str(STRUCTURE_INDEX_PATH), 'n_differing': n_differing, **first}
    _set_outcome(record, n_differing, 0, float(n_differing), 0.0, details)
    if n_differing:
        record['message'] = (f"{STRUCTURE_INDEX_PATH.name} differs from the generated text at line "
                             f"{details['first_line']}; {_REGENERATE_REFERENCE}")


def _first_difference(saved_lines, generated_lines):
    """(number of differing lines, {first_line (1-based), file_line, generated_line (clipped)} of the first one or {})
    of two _normalized_lines lists."""
    pairs = list(itertools.zip_longest(saved_lines, generated_lines))
    differing = [index for index, (saved_line, generated_line) in enumerate(pairs) if saved_line != generated_line]
    if not differing:
        return 0, {}
    file_line, generated_line = (_clipped_line(line) for line in pairs[differing[0]])
    return len(differing), {'first_line': differing[0] + 1, 'file_line': file_line, 'generated_line': generated_line}


def _normalized_lines(text):
    return text.replace('\r\n', '\n').rstrip('\n').split('\n')


def _clipped_line(line):
    return '<no line>' if line is None else line[:_MAX_LINE_CHARS]


def _available_backends():
    """('cpu', 'gpu') with a usable GPU (cupy and a CUDA device: one cached driver query), else ('cpu',)."""
    from .core.backend import gpu_available
    return ('cpu', 'gpu') if gpu_available() else ('cpu',)


def _selftest_run_backend(spec):
    """'cpu' for a selftest spec on solver_defaults.backend 'cpu', else 'gpu' (an 'auto' case spec is a GPU run)."""
    return 'cpu' if spec.solver_defaults.backend == 'cpu' else 'gpu'


def _registered_case(case, out_dir, available_backends=('cpu', 'gpu'), registry=None):
    """Run every spec of a registered SelfTestCase through run(), then its check(), then its variants. Runs on a
    backend not in available_backends are skipped (check gets the rest); a case whose own runs all are skipped is
    judged by its variants that ran (record 'variants_only'); None when nothing of it can run here."""
    start = time.perf_counter()
    record = _case_record(case.name, case.label, case.description)
    own_runs_skipped = False
    try:
        specs = case.specs()
        case_backends = case.backends or (tuple(dict.fromkeys(_selftest_run_backend(spec) for spec in specs.values()))
                                          or ('cpu',))
        own_runs_skipped = not set(case_backends) & set(available_backends)
        if own_runs_skipped:
            record['runs'] = {run_name: _skipped_gpu_run() for run_name in specs}
        else:
            _run_and_check(case, specs, None if out_dir is None else out_dir / case.name, available_backends, record)
    except Exception as exc:   # a broken case is reported as 'error'; the other cases still run
        _set_error(record, exc)
    if own_runs_skipped and not case.variants:
        return None
    if case.variants:
        registry = get_registry() if registry is None else registry
        record['variants'] = {variant.name: _variant_record(case, variant, out_dir, available_backends, registry)
                              for variant in case.variants}
        statuses = {name: variant['status'] for name, variant in record['variants'].items()}
        if own_runs_skipped and not _judge_by_variants(record, statuses):
            return None
        worst = 'error' if 'error' in statuses.values() else 'failed' if 'failed' in statuses.values() else None
        if worst is not None and record['status'] == 'passed':
            record.update(status=worst, message='variants not passed: ' + ', '.join(
                f'{name} ({status})' for name, status in statuses.items() if status in ('failed', 'error')))
    record['wall_time'] = time.perf_counter() - start
    return record


def _skipped_gpu_run():
    """Run record of a selftest run skipped for want of a usable GPU."""
    return {'status': 'skipped', 'wall_time': 0.0, 'out_dir': None, 'timing': None, 'vram': None, 'counts': None,
            'message': 'gpu run skipped: no usable GPU (cupy and a CUDA device)'}


def _judge_by_variants(record, statuses):
    """Case record whose own (gpu) runs were skipped: take the outcome of its first ran variant that did not pass (else
    its first ran one) -> False when no variant ran either."""
    ran = [name for name, status in statuses.items() if status != 'skipped']
    if not ran:
        return False
    shown = next((name for name in ran if statuses[name] != 'passed'), ran[0])
    variant = record['variants'][shown]
    record.update({key: variant[key] for key in ('status', 'measured', 'expected', 'error', 'tolerance', 'details')},
                  variants_only=True,
                  message=f"own gpu runs skipped (no usable GPU): judged by the variants {', '.join(ran)}; "
                          f"comparison of '{shown}'" + (f": {variant['message']}" if variant['message'] else ''))
    return True


def _run_and_check(case, specs, case_dir, available_backends, record, tolerance=None):
    """Run specs (skipping runs off available_backends), then case.check() into record (tolerance None = the
    check's own); the Results are closed afterwards, also on a raise."""
    results_by_name = {}
    try:
        for run_name, spec in specs.items():
            if _selftest_run_backend(spec) not in available_backends:
                record['runs'][run_name] = _skipped_gpu_run()
                continue
            run_dir = None if case_dir is None else case_dir / run_name
            results, run_record = _selftest_run(spec, run_dir, case.max_wall_time)
            record['runs'][run_name] = run_record
            if results is not None:
                results_by_name[run_name] = results
            if run_record['status'] != 'completed':
                record.update(status='failed', message=f"run '{run_name}' ended with status "
                                                       f"'{run_record['status']}': {run_record['message']}")
                return
        outcome = case.check(results_by_name)
        missing = [key for key in _CHECK_KEYS if key not in outcome]
        if missing:
            raise ValueError(f"check() of '{case.name}' returned no {', '.join(missing)}")
        _set_outcome(record, outcome['measured'], outcome['expected'], float(outcome['error']),
                     float(outcome['tolerance'] if tolerance is None else tolerance), outcome['details'])
    finally:
        for results in results_by_name.values():
            results._close(copy_to_host=False)   # nothing is read after the check (the run directories hold the rest)


def _variant_record(case, variant, out_dir, available_backends, registry):
    """Report entry of one SelfTestVariant: the case's runs moved by _variant_spec, judged by its own tolerance."""
    start = time.perf_counter()
    record = {'status': 'error', 'backend': variant.backend, 'precision': variant.precision, 'measured': None,
              'expected': None, 'error': None, 'tolerance': variant.tolerance, 'details': {}, 'message': '',
              'wall_time': 0.0, 'runs': {}}
    try:
        specs = {run_name: _variant_spec(spec, variant, registry) for run_name, spec in case.specs().items()}
        if not specs:
            raise ValueError(f"variant '{variant.name}' of '{case.name}': the case has no runs to repeat")
        if not {_selftest_run_backend(spec) for spec in specs.values()} & set(available_backends):
            record.update(status='skipped', message='gpu variant skipped: no usable GPU (cupy and a CUDA device)')
        else:
            case_dir = None if out_dir is None else out_dir / case.name / f'variant_{variant.name}'
            _run_and_check(case, specs, case_dir, available_backends, record, variant.tolerance)
    except Exception as exc:   # reported as an 'error' variant; the case's other variants still run
        _set_error(record, exc)
    record['wall_time'] = time.perf_counter() - start
    return record


def _variant_spec(spec, variant, registry):
    """Copy of a selftest spec on the variant's backend; with a precision, every role it applies to gets its own
    solver node (the assigned node's or the Automatic type and settings, precision replaced; orphaned nodes dropped)."""
    spec = copy.deepcopy(spec)
    if variant.backend is not None:
        spec.solver_defaults.backend = variant.backend
    if variant.precision is None:
        return spec
    backend = solver_backend(spec)
    nodes = {node.name: node for node in spec.solvers}
    replaced = set()
    for physics_spec in spec.physics:
        for role, solver_role in registry.physics_type(physics_spec.type).solver_roles.items():
            if not solver_role.used_in(spec.study.type) or (variant.roles and role not in variant.roles):
                continue
            node = nodes.get(physics_spec.solvers.get(role))
            type_name, settings = ((node.type, dict(node.settings)) if node is not None
                                   else solver_role.automatic(backend, solver_grid(spec, physics_spec),
                                                              spec.study.type))
            solver_type = registry.solver_type(type_name)
            if not any(param.name == 'precision' for param in solver_type.params):
                if variant.roles and variant.precision not in solver_type.precisions:
                    raise ValueError(f"variant '{variant.name}': the {role} solver of physics '{physics_spec.name}' "
                                     f"({type_name}) has no precision {variant.precision!r}")
                continue
            node_name = f'{variant.name}_{physics_spec.name}_{role}'
            spec.solvers.append(SolverSpec(node_name, type_name, dict(settings, precision=variant.precision)))
            if node is not None:
                replaced.add(node.name)
            physics_spec.solvers[role] = node_name
    still_used = {name for physics_spec in spec.physics for name in physics_spec.solvers.values()}
    spec.solvers = [node for node in spec.solvers if node.name not in replaced - still_used]
    return spec


def _selftest_run(spec, run_dir, max_wall_time):
    """(Results or None, run record) of one selftest run; any failure raised by run() (no out_dir: ModelError,
    RunError or a build's own exception) is a 'failed' run."""
    start = time.perf_counter()
    try:
        with warnings.catch_warnings():   # a multi-run case keeps its earlier Results live on purpose (log line only)
            warnings.filterwarnings('ignore', message=_LIVE_RESULTS_WARNING_PATTERN, category=RuntimeWarning)
            results = run(spec, out_dir=run_dir, max_wall_time=max_wall_time)
    except Exception as exc:   # text only (run() already cleared the frames of the exceptions it re-raises)
        return None, {'status': 'failed', 'wall_time': time.perf_counter() - start, 'message': _error_text(exc),
                      'out_dir': None, 'timing': None, 'vram': None, 'counts': None}
    run_seconds = time.perf_counter() - start
    return results, {'status': results.status, 'wall_time': run_seconds,
                     'message': results._data.summary.get('message', ''),
                     'out_dir': None if run_dir is None else str(run_dir),
                     'timing': _run_timing(results._data.summary, run_seconds),
                     'vram': _run_vram(results._data.summary, _selftest_run_backend(spec)),
                     'counts': _run_counts(results._data.summary)}


def _run_timing(summary, run_seconds):
    """Selftest run 'timing' from its summary.json: build / study seconds (every build and run of a sweep, refinement
    or parity-sector study summed), output seconds (run() after the study: summary, host copies, files = run_seconds -
    summary wall_time), cells, ticks, cell ticks per second (time_dependent only), the summary 'solvers' record with
    its work counters (a performance baseline for one machine) and the run-time check against the estimate (None
    without a machine profile)."""
    build = summary.get('build') or {}
    study_seconds, ticks, n_cells = summary.get('study_seconds'), summary.get('ticks'), build.get('n_cells')
    time_stepped = (summary.get('model') or {}).get('study_type') == 'time_dependent'
    cell_ticks_per_second = (n_cells * ticks / study_seconds
                             if time_stepped and study_seconds and ticks and n_cells else None)
    before_output = summary.get('wall_time')
    build_seconds = summary.get('build_seconds_total')
    return {'build_seconds': build.get('build_seconds') if build_seconds is None else build_seconds,
            'study_seconds': study_seconds,
            'output_seconds': None if before_output is None else max(run_seconds - before_output, 0.0),
            'n_cells': n_cells, 'ticks': ticks, 'cell_ticks_per_second': cell_ticks_per_second,
            'solvers': summary.get('solvers') or {}, 'time_check': summary.get('time_check')}


def _run_vram(summary, backend):
    """Selftest run 'vram' from its summary.json: {kept, build_peak, run, external (cuDSS outside the pool):
    {measured_bytes, estimated_bytes, deviation, verdict (builder.estimate_verdict)} or None, deviation, part, verdict,
    beyond_floor (of the worst part: _vram_rank)}; None on cpu / unestimated."""
    build = summary.get('build') or {}
    estimate = build.get('gpu_estimate')
    if backend == 'cpu' or not estimate:   # the cpu ledger traces host numpy: not VRAM
        return None

    def part(measured_bytes, estimated_bytes, deviation, unseen_bytes=0):
        if measured_bytes is None or estimated_bytes is None or deviation is None:
            return None
        return {'measured_bytes': int(measured_bytes), 'estimated_bytes': int(estimated_bytes),
                'unseen_bytes': int(unseen_bytes or 0), 'deviation': deviation}

    breakdown = build.get('gpu_breakdown') or []
    parts = {'kept': part(sum(size for _, size in breakdown) if breakdown else None, estimate.get('gpu_steady'),
                          estimate.get('steady_deviation')),
             'build_peak': part(build.get('gpu_peak_bytes'), estimate.get('gpu_build_peak'),
                                estimate.get('peak_deviation'), estimate.get('peak_unseen_bytes')),
             'run': summary.get('run_memory'), 'external': summary.get('external_memory')}
    if all(value is None for value in parts.values()):
        return None
    for value in parts.values():
        if value is not None:   # a pool mark misses up to unseen_bytes (free blocks stranded before it started)
            unseen_bytes = value.get('unseen_bytes') or 0
            value['verdict'] = estimate_verdict(value['measured_bytes'], value['estimated_bytes'], unseen_bytes)
            value['beyond_floor'] = abs(estimate_gap(value['measured_bytes'], value['estimated_bytes'],
                                                     unseen_bytes)) > ESTIMATE_FLOOR_BYTES
    worst_name, worst = max(((name, value) for name, value in parts.items() if value is not None),
                            key=lambda item: _vram_rank(item[1]))
    return dict(parts, deviation=worst['deviation'], part=worst_name, verdict=worst['verdict'],
                beyond_floor=worst['beyond_floor'])


_VRAM_PART_NAMES = ('kept', 'build_peak', 'run', 'external')


def _vram_rank(part):
    """Sort key of a vram part (or run vram): an 'over' / 'unaccounted' verdict first, then a difference beyond
    ESTIMATE_FLOOR_BYTES, then the larger |deviation|."""
    return bool(part.get('verdict')), bool(part.get('beyond_floor')), abs(part['deviation'])


def _run_counts(summary):
    """Selftest run 'counts' from its summary time_check: {deviation (measured / predicted - 1 of the item with a counter
    that is off the most, signed; 0.0 when none is off by more than 1), item (its label)}; None without counted items."""
    items = [item for item in ((summary.get('time_check') or {}).get('items') or ())
             if item.get('count_ratio') is not None]
    if not items:
        return None
    differing = [item for item in items if abs(item['measured_count'] - item['predicted_count']) > 1.0]
    if not differing:
        return {'deviation': 0.0, 'item': None}
    worst = max(differing, key=lambda item: abs(item['count_ratio'] - 1.0))
    return {'deviation': worst['count_ratio'] - 1.0, 'item': worst['label']}


def _set_worst_deviations(record, run_records):
    """Case / example record vram_deviation / vram_part / vram_verdict (the worst run vram: _vram_rank), vram_max_bytes
    (largest measured part over the runs), count_deviation and count_item (the largest by size); None when no run has one."""
    worst_vram = max((run['vram'] for run in run_records if run.get('vram') and run['vram']['deviation'] is not None),
                     key=_vram_rank, default=None)
    measured_part_bytes = [part['measured_bytes'] for run in run_records if run.get('vram')
                           for part in (run['vram'].get(name) for name in _VRAM_PART_NAMES) if part is not None]
    record['vram_max_bytes'] = max(measured_part_bytes, default=None)
    worst_counts = max((run['counts'] for run in run_records if run.get('counts')),
                       key=lambda part: abs(part['deviation']), default=None)
    record['vram_deviation'] = None if worst_vram is None else worst_vram['deviation']
    record['vram_part'] = None if worst_vram is None else worst_vram.get('part')
    record['vram_verdict'] = None if worst_vram is None else worst_vram.get('verdict')
    record['count_deviation'] = None if worst_counts is None else worst_counts['deviation']
    record['count_item'] = None if worst_counts is None else worst_counts['item']


def _case_estimate_specs(case, available_backends, registry):
    """The specs a registered case will run here (its own and its variants' on available backends) for the estimate;
    [] when its specs() raises (the run reports that)."""
    try:
        specs = list(case.specs().values())
        specs += [_variant_spec(spec, variant, registry) for variant in case.variants for spec in list(specs)]
    except Exception:   # the case run raises it again as the case's 'error'
        return []
    return [spec for spec in specs if _selftest_run_backend(spec) in available_backends]


_TIME_PHASES = ('build', 'study', 'output')
_SECONDS_KEYS = ('seconds_low', 'seconds_guess', 'seconds_high')


def _specs_time_estimate(specs, registry, save_arrays=True):
    """Run-time estimate of a selftest case / example before it runs: the summed estimates of its specs on the stored
    machine profiles -> {seconds_low, seconds_guess, seconds_high (build + study + output), phases {phase: the same
    keys}, n_runs, n_estimated, message, not_estimated, outdated (' (outdated gpu profile)' or '')}; None without
    specs. A failing estimate never stops a selftest (message)."""
    if not specs:
        return None
    phases = {phase: dict.fromkeys(_SECONDS_KEYS, 0.0) for phase in _TIME_PHASES}
    n_estimated, messages, not_estimated, outdated = 0, [], [], ''
    for spec in specs:
        try:
            parameters = evaluate_parameters(spec.parameters, errors={})
            meshes = estimate_meshes(spec, parameters, registry)
            runs = (estimated_runs(spec, parameters, registry, meshes)   # each run's meshes once, for both estimates
                    if stored_profile(solver_backend(spec))[0] is not None else None)
            estimate, message = estimate_time(spec, parameters, registry, meshes, runs=runs)
            parts = ({} if estimate is None
                     else dict(estimate_overheads(spec, parameters, registry, meshes, save_arrays=save_arrays,
                                                  runs=runs), study=estimate))
        except Exception as exc:   # a model error is the run's to report
            estimate, message = None, _error_text(exc)
        if estimate is None:
            messages.append(message)
            continue
        n_estimated += 1
        for phase, part in parts.items():
            for key in _SECONDS_KEYS:
                phases[phase][key] += getattr(part, key)
        not_estimated.extend(estimate.not_estimated)
        outdated = outdated or estimate.outdated_text()
    totals = {key: sum(phase[key] for phase in phases.values()) for key in _SECONDS_KEYS}
    return dict(totals, phases=phases, n_runs=len(specs), n_estimated=n_estimated,
                message=messages[0] if messages else '', not_estimated=not_estimated, outdated=outdated)


def _study_work_at_build(time_check):
    """Seconds of a run's counter-timed study work done at build (time_check seconds_at_build, e.g. a factorisation in
    finalize); 0.0 without a time_check."""
    return float((time_check or {}).get('seconds_at_build') or 0.0)


def _time_estimate_after(before, run_records, total_seconds):
    """Record 'time_estimate' of a finished case / example: the estimate before it ran and the time split {total,
    build / study / output / other: {seconds, estimate}, study_work_at_build} over its runs (study estimate: at the
    measured counts of the runs with a summary time_check; counter-timed study work done at build moves from build to
    study seconds, so both sides price the same work; build / output: the guesses before; other = the rest, never
    estimated)."""
    timings = [run.get('timing') or {} for run in run_records]
    checks = [timing['time_check'] for timing in timings if timing.get('time_check')]
    phases = (before or {}).get('phases') or {}

    def measured(key):
        return sum(timing.get(key) or 0.0 for timing in timings)

    split = {'total': total_seconds}
    for phase in _TIME_PHASES:
        guess = phases.get(phase, {}).get('seconds_guess') if before and before['n_estimated'] else None
        split[phase] = {'seconds': measured(f'{phase}_seconds'), 'estimate': guess}
    at_build = sum(_study_work_at_build(check) for check in checks)
    split['build']['seconds'] = max(split['build']['seconds'] - at_build, 0.0)
    split['study']['seconds'] += at_build
    split['study_work_at_build'] = at_build
    split['study']['estimate'] = (sum(group['estimated_seconds'] for check in checks for group in check['groups'])
                                  if checks else None)
    split['other'] = {'seconds': max(total_seconds - sum(split[phase]['seconds'] for phase in _TIME_PHASES), 0.0),
                      'estimate': None}
    return {'before': before, 'n_checked_runs': len(checks), 'split': split}


def _device_record(backend=None):
    """{device_id, name, kind, device_total, device_free, budget} [bytes] (summary.json memory names) of the GPU, or of
    the host for backend 'cpu' (budget None); None when the query fails."""
    if backend == 'cpu':
        info, budget = gpu_memory.cpu_device_info(), None
    else:
        try:
            info = gpu_memory.device_info()
        except Exception:   # a report field: a failing query must not cost the report
            return None
        budget = gpu_memory.budget_bytes()
    return {'device_id': info.device_id, 'name': info.name, 'kind': info.kind, 'device_total': info.total_bytes,
            'device_free': info.free_bytes, 'budget': budget}


# ---- analytic examples (selftest --examples) ----
def selftest_examples(out_dir=None, names=None, progress=None, reset=False) -> dict:
    """run() every examples/**/*.json with a derived reference or an analytic curve (names: only these, relative
    paths or stems) once, on its own backend (one setting backend 'gpu' is skipped without a usable GPU) -> the
    ANALYTIC_REPORT_FILE dict; out_dir -> that file + examples/<path>/ run directories (also partial on an interrupt)
    + CACHE_FILE: without names an example that passed with unchanged file and dependencies is reused (reset True:
    rerun all); progress(event): 'begin', then per example run 'start' and its record (_SelftestPlan; dropped with a
    warning if it raises). Estimates price on an out-of-date stored machine profile too."""
    out_dir = None if out_dir is None else Path(out_dir)
    start = time.perf_counter()
    cache = _selftest_cache(out_dir, _available_backends())
    plan = _SelftestPlan(progress, cache)
    examples_plan = _plan_examples(names, cache, cache is not None and not reset and names is None, plan)
    return _run_examples(out_dir, examples_plan, plan, cache, start)


def _plan_examples(names, cache, reuse, plan):
    """Load and sort the selftest examples (those with an analytic comparison; a named one without: KeyError) of the
    examples run (backend 'gpu' without a usable GPU: skipped_backend; cache hits: reused) and add its phase to plan
    -> the plan dict of _run_examples."""
    model_paths = _selected_example_paths(names)
    gpu_usable = 'gpu' in _available_backends()
    entries, skipped_backend, named_without_reference = [], [], []
    for path in model_paths:
        relative_path = path.relative_to(EXAMPLES_DIR).as_posix()
        entry = {'path': path, 'relative_path': relative_path, 'spec': None, 'load_error': None, 'reused': None,
                 'cache_key': relative_path, 'model_hash': None}
        try:
            entry['spec'] = load_model(path)
        except Exception as exc:   # an unloadable example is an 'error' entry, not a selftest crash
            entry['load_error'] = failed_example_record(relative_path, None, f'loading failed: {_error_text(exc)}', 0.0)
            entry['load_error']['backend'] = None
            entries.append(entry)
            continue
        if not has_analytic_comparison(entry['spec']):   # not a selftest example
            if names is not None:
                named_without_reference.append(relative_path)
            continue
        if entry['spec'].solver_defaults.backend == 'gpu' and not gpu_usable:
            skipped_backend.append(relative_path)
            continue
        entry['model_hash'] = file_hash(path)
        if reuse:
            entry['reused'] = cache.reusable_record('examples', entry['cache_key'], entry['model_hash'])
        entries.append(entry)
    if named_without_reference:
        raise KeyError(f"not a selftest example (no derived value with a reference, no analytic curve): "
                       f"{', '.join(named_without_reference)}; run it with: python -m open_sol run <model file>")
    entries = _untimed_then_fastest(entries, cache, 'examples', lambda entry: entry['cache_key'])
    tests = [('examples', entry['cache_key'], entry['relative_path']) for entry in entries if entry['reused'] is None]
    phase = plan.add_phase('examples', tests, len(entries) - len(tests))
    return {'phase': phase, 'entries': entries, 'skipped_backend': skipped_backend}


def _run_examples(out_dir, examples_plan, plan, cache, start):
    """Run the examples of a _plan_examples plan (reused records kept) -> the ANALYTIC_REPORT_FILE dict; out_dir ->
    that file + run directories (also partial on an interrupt) + the cache file."""
    records, reused, report_warnings = [], [], []
    skipped_backend = examples_plan['skipped_backend']
    all_ran = False
    interrupt = None   # the KeyboardInterrupt in flight: gets the partial report as .partial_report (CLI counts)
    plan.begin(examples_plan['phase'])
    try:
        for entry in examples_plan['entries']:
            if entry['reused'] is not None:
                reused.append(entry['relative_path'])
                records.append(entry['reused'])
                continue
            record = entry['load_error']
            if record is not None:
                plan.start()
            else:
                before = _specs_time_estimate([entry['spec'].copy()], get_registry(), save_arrays=False)
                plan.start(before)
                from .fem.gmsh3d import default_disk_cache_off   # entered before the trace (no gmsh disk hits)
                with default_disk_cache_off(), DependencyTrace() as trace:
                    record = _example_run_record(entry['path'], entry['relative_path'], entry['spec'], out_dir,
                                                 time.perf_counter())
                record['time_estimate'] = _time_estimate_after(before, [record], record['wall_time'])
                _set_worst_deviations(record, [record])
                if cache is not None:
                    cache.store('examples', entry['cache_key'], record, trace.units(), entry['model_hash'])
            records.append(record)
            plan.finish(record['wall_time'], record)
        all_ran = True
    except KeyboardInterrupt as exc:
        interrupt = exc
        raise
    finally:   # an interrupt (KeyboardInterrupt) still writes the examples already run
        report_warnings.extend(plan.take_warnings())
        if cache is not None:
            cache.write()
        if not all_ran:
            report_warnings.append(f'interrupted after {len(records)} examples; the report is partial')
        if skipped_backend:
            report_warnings.append(f'{len(skipped_backend)} examples skipped: they set backend gpu, and there is no '
                                   f'usable GPU (cupy and a CUDA device) here')
        report = analytic_report(records, time.perf_counter() - start, all_ran, report_warnings,
                                 _device_record(), __version__, skipped_backend)
        report.update(n_reused=len(reused), reused=reused)
        report['report_files'] = [] if out_dir is None else [ANALYTIC_REPORT_FILE]
        if out_dir is not None:
            _write_report_keeping_timings(out_dir / report['report_files'][0], report)
        if interrupt is not None:
            interrupt.partial_report = report
    return report


def _selected_example_paths(names):
    """examples/**/*.json (the examples_validate rule), only those named (relative path with or without .json, or
    stem) when names is given; KeyError for a name matching none."""
    model_paths = _example_model_paths() if EXAMPLES_DIR.is_dir() else []
    if names is None:
        return model_paths
    names = [names] if isinstance(names, str) else list(names)

    def keys(path):
        relative_path = path.relative_to(EXAMPLES_DIR).as_posix()
        return {relative_path, relative_path[:-len('.json')], path.stem}

    unknown = [name for name in names if not any(name in keys(path) for path in model_paths)]
    if unknown:
        raise KeyError(f"no example {', '.join(map(repr, unknown))} under {EXAMPLES_DIR} ({len(model_paths)} model "
                       f"files; name one by its relative path, e.g. em/dielectric_scattering, or its stem)")
    return [path for path in model_paths if any(name in keys(path) for name in names)]


def _example_run_record(path, relative_path, spec, out_dir, start):
    """Report entry of one example: run() on its own backend into out_dir/examples/<path>/ (no arrays, no figures),
    then its summary compared with the spec's tolerances; a raising run is an 'error' entry."""
    run_dir = None if out_dir is None else out_dir / 'examples' / relative_path[:-len('.json')]
    run_backend = solver_backend(spec)
    run_start = time.perf_counter()
    try:
        results = run(path, out_dir=run_dir, save_arrays=False)
    except Exception as exc:   # text only (run() cleared the frames of what it re-raises)
        record = failed_example_record(relative_path, spec.title, _error_text(exc), time.perf_counter() - start)
    else:
        run_seconds = time.perf_counter() - run_start
        summary = results.summary()
        results._close(copy_to_host=False)   # free the GPU before the next example; the run directory holds the rest
        record = example_record(relative_path, spec, summary, time.perf_counter() - start,
                                None if run_dir is None else str(run_dir))
        record['timing'] = _run_timing(summary, run_seconds)
        record['vram'] = _run_vram(summary, run_backend)
        record['counts'] = _run_counts(summary)
    record['backend'] = run_backend
    return record


# ---- shared helpers ----
def _as_spec(spec) -> ModelSpec:
    """Fresh ModelSpec of a ModelSpec (copy), dict or JSON model file path; ValueError for invalid content, OSError
    for an unreadable file, TypeError for anything else."""
    if isinstance(spec, ModelSpec):
        return spec.copy()
    if isinstance(spec, dict):
        return ModelSpec.from_dict(spec)
    if isinstance(spec, (str, os.PathLike)):
        return load_model(spec)
    raise TypeError(f'spec must be a ModelSpec, a dict or a model file path, got {type(spec).__name__}')


def _apply_backend(spec, backend):
    """Set solver_defaults.backend unless backend is None (validation reports a value outside SOLVER_BACKENDS)."""
    if backend is not None:
        spec.solver_defaults.backend = backend


def _apply_parameters(spec, parameters):
    """Set the expressions of {name: str | number} in spec (numbers as repr(float)); KeyError listing the known names
    before anything changes, TypeError for another value type."""
    if not parameters:
        return
    known_names = [parameter.name for parameter in spec.parameters]
    unknown_names = [name for name in parameters if name not in known_names]
    if unknown_names:
        raise KeyError(f"no parameter named {', '.join(map(repr, unknown_names))}; known: "
                       f"{', '.join(known_names) or 'none'}")
    for name, expression in parameters.items():
        if isinstance(expression, bool) or not isinstance(expression, (str, int, float)):
            raise TypeError(f'parameter {name!r}: the override must be an expression string or a number, got '
                            f'{expression!r}')
        spec.parameter(name).expression = expression if isinstance(expression, str) else repr(float(expression))


def _model_file(spec):
    """The model file path text when spec was given as a path, else None."""
    return os.fspath(spec) if isinstance(spec, (str, os.PathLike)) else None


def _result_key_parts(physics, feature):
    """(physics, feature) of result(physics, feature) or result('physics.feature')."""
    if feature is not None:
        return physics, feature
    physics_name, dot, feature_name = str(physics).partition('.')
    if not dot or not physics_name or not feature_name:
        raise KeyError(f"result key {physics!r} is not 'physics.feature'; pass (physics, feature) or such a key")
    return physics_name, feature_name


def _host_field_record(session, physics, component):
    """Host FieldRecord of one output field on the native window cells of its physics' mesh, with the window cell
    edges; polar / spherical: the containing-cell image with its uniform pixel edges over the sector's drawing-plane
    box; a triangle mesh: its raster pixels over the window plus the triangulation (node tables in results.npz)."""
    field_data = session.field(component, physics=physics)   # 3D: the mid-height z cut
    if field_data.x_centers is None:   # mapped image / triangle raster: pixels uniform over extent
        x_min, x_max, y_min, y_max = field_data.extent
        n_u, n_v = field_data.values.shape
        x_edges, y_edges = np.linspace(x_min, x_max, n_u + 1), np.linspace(y_min, y_max, n_v + 1)
    else:
        domain = session.domain_of(field_data.physics)
        window_x, window_y = domain.physical_window()[:2]
        all_x_edges, all_y_edges = domain.cell_edges()[:2]
        x_edges = np.array(all_x_edges[window_x.start:window_x.stop + 1])
        y_edges = np.array(all_y_edges[window_y.start:window_y.stop + 1])
    return FieldRecord(physics=physics, component=component, values=field_data.values,
                       x_edges=x_edges, y_edges=y_edges,
                       info=field_data.info, time=field_data.time, time_label=field_data.time_label,
                       plane=field_data.plane, mesh=field_data.mesh, triangulation=field_data.triangulation)


def _primary_field_keys(session):
    """'physics.component' of each physics' primary output field (its first one when none is marked primary)."""
    keys = []
    for physics in session.physics:
        info = primary_field_info(session.renderer(physics).field_infos.values())
        if info is not None:
            keys.append(f'{physics}.{info.name}')
    return keys


def _dft_max_intensity(session, physics, feature, n_frequencies):
    """Max DFT intensity of the monitor quantity per frequency, fetched one frequency at a time (no stacked copy; a
    box has >= 1 cell)."""
    return [float(session.feature_result(physics, feature, freq_index=index)['intensity'].max())
            for index in range(n_frequencies)]


def _dft_intensity_stack(session, physics, feature, result):
    """Host float32 (n_f, mx, my) DFT intensity of the monitor quantity at every frequency of the dft_fields result
    (the solver's own precision: float64 doubled results.npz)."""
    n_x, n_y = result['shape']
    stack = np.empty((len(result['frequency']), n_x, n_y), dtype=np.float32)
    for index in range(stack.shape[0]):
        stack[index] = session.feature_result(physics, feature, freq_index=index)['intensity']
    return stack


def _issue_record(issue):
    """JSON record of an Issue: level, path (list of str), node label, message."""
    return {'level': issue.level, 'path': [str(part) for part in issue.path], 'node': node_label(issue.path),
            'message': issue.message}


def _mesh_record(mesh):
    """check.json 'mesh' block of a MeshEstimate / TriangleMeshEstimate / TetrahedronMeshEstimate (None when the
    estimate failed); an element mesh has nx / ny null and its element / node / edge (tetrahedra: face) counts."""
    if mesh is None:
        return None
    record = {'name': mesh.name, 'kind': mesh.kind, 'n_cells': mesh.n_cells, 'dx_min': mesh.dx_min,
              'dx_max': mesh.dx_max, 'dt': mesh.dt, 'npml_sides': dict(mesh.npml_sides), 'uniform': mesh.uniform,
              'note': mesh.note, 'wavelength_min': mesh.wavelength_min, 'wavelength_source': mesh.wavelength_source,
              'auto_rule': mesh.auto_rule, 'custom_axes': list(mesh.custom_axes),
              'line_blocks': {axis: list(layout) for axis, layout in mesh.line_blocks.items()},
              'line_block_hint': line_block_hint(mesh)}
    if mesh.kind == 'tensor':
        return {'nx': mesh.nx, 'ny': mesh.ny, 'nz': mesh.nz, **record}
    return {'nx': None, 'ny': None, 'nz': 1, **record, 'n_elements': mesh.n_elements, 'n_nodes': mesh.n_nodes,
            'n_edges': mesh.n_edges, 'pml_widths': dict(mesh.sizing.pml_widths),
            **({'n_faces': mesh.n_faces} if mesh.kind == 'tetrahedra' else {})}


def _mesh_records(spec, parameters, meshes, mesh_errors, by_mesh):
    """check.json 'meshes': per MeshSpec {name, kind, mode, shape ([nx, ny(, nz)] or null), elements (triangles /
    tetrahedra, else null), cells, dx_min, dx_max, max_ratio, physics, used, gpu_bytes, host_bytes (null without an
    estimate), error}."""
    records = []
    for mesh_spec in spec.meshes:
        mesh = meshes.get(mesh_spec.name)
        try:
            max_ratio = mesh_values(mesh_spec, parameters)['max_ratio']
        except (ValueError, KeyError):   # reported by validate
            max_ratio = None
        part = by_mesh.get(mesh_spec.name)
        records.append({
            'name': mesh_spec.name, 'kind': mesh_spec.kind, 'mode': mesh_spec.mode,
            'shape': (None if mesh is None or mesh.kind != 'tensor'
                      else [mesh.nx, mesh.ny] + ([mesh.nz] if mesh.z_edges is not None else [])),
            'elements': None if mesh is None or mesh.kind == 'tensor' else mesh.n_elements,
            'cells': None if mesh is None else mesh.n_cells,
            'dx_min': None if mesh is None else mesh.dx_min, 'dx_max': None if mesh is None else mesh.dx_max,
            'max_ratio': max_ratio, 'physics': [physics.name for physics in spec.physics_on(mesh_spec.name)],
            'used': mesh_spec.name in meshes or mesh_spec.name in mesh_errors,
            'gpu_bytes': None if part is None else part.gpu_steady, 'host_bytes': None if part is None else part.host,
            'error': mesh_errors.get(mesh_spec.name)})
    return records


def _build_record(build_summary):
    """summary.json 'build' block of a BuildSummary (None before any successful build)."""
    if build_summary is None:
        return None
    return {'nx': build_summary.nx, 'ny': build_summary.ny, 'nz': build_summary.nz,
            'n_cells': build_summary.n_cells, 'dt': build_summary.dt, 'dx_min': build_summary.dx_min, 'dx_max': build_summary.dx_max,
            'npml_sides': dict(build_summary.npml_sides), 'build_seconds': build_summary.build_seconds,
            'gpu_bytes': build_summary.gpu_bytes, 'session_gpu_bytes': build_summary.session_gpu_bytes,
            'gpu_breakdown': [[owner, size] for owner, size in build_summary.gpu_breakdown],
            'stage_seconds': [[owner, seconds] for owner, seconds in build_summary.stage_seconds],
            'gpu_peak_bytes': build_summary.gpu_peak_bytes, 'gpu_budget': build_summary.gpu_budget,
            'gpu_estimate': build_summary.gpu_estimate}


def _memory_record(build_summary):
    """MemoryStatus.as_dict() + the last build's GPU bytes, or None when the driver query fails."""
    try:
        record = gpu_memory.memory_status().as_dict()
    except Exception:   # a report field: a failing query must not cost the summary
        return None
    record['build_gpu_bytes'] = None if build_summary is None else build_summary.gpu_bytes
    record['session_gpu_bytes'] = None if build_summary is None else build_summary.session_gpu_bytes
    return record


def _built_text(summary):
    """Log line of a Built event (session_gpu_bytes when known, else the process-wide pool use)."""
    time_step = 'no time stepping' if summary.dt is None else f'dt {summary.dt:.4g} s'
    gpu_bytes = summary.gpu_bytes if summary.session_gpu_bytes is None else summary.session_gpu_bytes
    return (f'built {_cells_text(summary.nx, summary.ny, summary.nz, summary.n_cells)}, {time_step}, '
            f'{summary.build_seconds:.2f} s, GPU {gpu_memory.format_bytes(gpu_bytes)}')


def _cells_text(nx, ny, nz, n_cells):
    """'nx x ny = n cells', 3D (nz > 1) 'nx x ny x nz = n cells' (also the CLI's)."""
    counts = f'{nx} x {ny}' if nz is None or nz <= 1 else f'{nx} x {ny} x {nz}'
    return f'{counts} = {n_cells:,} cells'


def _log_line(start, level, message):
    """log.txt line '[<elapsed s>] [<level>] <message>' (elapsed since start, a perf_counter value)."""
    return f'[{time.perf_counter() - start:.2f}] [{level}] {message}'


def _error_text(exc):
    return f'{type(exc).__name__}: {error_message(exc)}'
