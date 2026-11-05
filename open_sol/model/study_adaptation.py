"""Mesh refinement levels of one study (S7.7, StudySpec.adaptation): build and run the study per level, error
indicator, stop rule, the next level's meshes ('adaptive' tensor bands / triangle size points, or the 'uniform'
sequence) and the convergence report (summary 'convergence', convergence.json) with a Richardson estimate over the
last three levels."""
import math
from dataclasses import dataclass, replace

import numpy as np

from ..core import gpu_memory, host_memory
from ..core.backend import to_host
from ..core.error_indicators import interpolation_indicator, relative_field
from ..core.renderer import primary_field_info
from .expressions import evaluate_parameters
from .geometry import resolve_geometry
from .mesh_refinement import (adaptive_mesh_spec, frozen_mesh_spec, refined_axes, uniform_mesh_spec, window_cells,
                              window_widths)
from .meshing import backend_memory, estimate_memory, estimate_mesh
from .schema import solver_backend
from .session import live_snapshot_index
from .solver_hooks import ErrorIndicatorSolver
from .study import (Cancelled, LogEvent, StudyFailed, StudyStarted, _StageProgress, _failure_message, _fail_sweep_setup,
                    _record_failure, _run_built, build_session, release_session, summed_runs)

# stop reasons of the convergence report ('converged' alone means converged)
STOP_REASONS = ('converged', 'max_levels', 'max_cells', 'memory', 'no_refinement', 'not_finite', 'cancelled',
                'failed')
_RICHARDSON_MAX_ORDER = 64.0   # largest order in cells the Richardson fit searches
_RICHARDSON_BISECTIONS = 200


@dataclass
class _Setup:
    """Resolved StudySpec.adaptation of a run: the spec's AdaptationSpec, base parameters, refined mesh names and the
    model dimension (the 'uniform' size factor growth^(-1 / dimensions))."""
    adaptation: object
    parameters: dict
    meshes: list
    dimensions: int


def run_adaptation(spec, registry, build, emit, control, service, on_session):
    """Build and run the study on the model's meshes, then on refined meshes until the target derived value converges
    (or max_levels / max_cells / the memory budget stops it) -> the last level's summary + 'convergence' (its session
    stays live), or None after a failure of level 0 (StudyFailed emitted)."""
    try:
        setup = _setup(spec, registry)
    except Exception as exc:   # text only; reported through the build path with the node paths
        _fail_sweep_setup(spec, registry, build, emit, on_session, _record_failure(exc), what='mesh refinement')
        return None
    adaptation = setup.adaptation
    stages = _StageProgress(emit, [_stage_name(level) for level in range(adaptation.max_levels + 1)])
    level_spec, levels, kept_snapshots, estimates = spec, [], [], None
    last_summary, last_spec, session, previous = None, spec, None, None
    stop_reason, message, started = 'max_levels', '', False
    level_runs = []   # [(label, summary)] of every computed level (summed_runs)
    for level in range(adaptation.max_levels + 1):
        if level > 0:
            try:
                control.checkpoint(service)
            except Cancelled:
                stop_reason, message = 'cancelled', f'cancelled before level {level}'
                break
            emit(LogEvent('info', f'mesh refinement level {level}: {_cells_text(estimates, setup.meshes)}'))
        level_emit = stages.stage_emit(_stage_name(level))
        release_session(on_session, previous, emit if previous is not None else None)
        previous = None
        session, failure = build_session(level_spec, level_emit, on_session, build=build, registry=registry)
        if failure is not None:
            if level == 0:
                emit(StudyFailed(f'build failed: {failure.message}', failure.traceback,
                                 out_of_memory=failure.out_of_memory,
                                 build_memory=failure.build_memory if failure.out_of_memory else None))
                return None
            stop_reason, message = 'failed', f'level {level}: build failed: {failure.message}'
            break
        if not started:
            emit(StudyStarted())
            started = True
        if level == 0:   # a failing first run fails the study like a single run (run_study reports it)
            estimates = {mesh_spec.name: estimate_mesh(level_spec, setup.parameters, registry, mesh_spec.name)
                         for mesh_spec in level_spec.used_meshes()}   # after the build: it validated the spec
            emit(LogEvent('info', f'mesh refinement level 0: {_cells_text(estimates, setup.meshes)}'))
            summary = _run_built(session, level_spec, registry, level_emit, control, service)
        else:
            run_failure = None
            try:
                summary = _run_built(session, level_spec, registry, level_emit, control, service)
            except Exception as exc:   # text only: the traceback holds the run's arrays
                run_failure = _record_failure(exc)
            if run_failure is not None:
                release_session(on_session, session, emit)   # a half-run level must not pose as the results
                session = None
                stop_reason, message = 'failed', f'level {level}: {run_failure[0]}'
                break
        level_runs.append((f'level {level}', summary))
        levels.append(_level_record(level, estimates, setup.meshes, summary, adaptation.target, levels, session))
        last_summary, last_spec, previous = summary, level_spec, session
        if summary['status'] == 'cancelled':
            stop_reason, message = 'cancelled', f'level {level} was cancelled'
            break
        stop_reason, message = level_stop([record['value'] for record in levels], adaptation.tolerance, level,
                                          adaptation.max_levels)
        if stop_reason is None:
            next_spec, next_estimates, stop_reason, message = _next_level(session, level_spec, setup, estimates,
                                                                          registry, emit)
        if stop_reason is not None:
            break
        if adaptation.keep_levels:
            kept_snapshots += [replace(snapshot, label=f'level {level}' + (f': {snapshot.label}' if snapshot.label
                                                                            else ''), live=False)
                               for snapshot in session.snapshots]
        level_spec, estimates = next_spec, next_estimates
    summary = dict(last_summary)   # level 0 ran: its failures returned or raised above
    report = convergence_report(adaptation, setup, levels, stop_reason, message, last_spec)
    summary['convergence'] = report
    summary.update(summed_runs(level_runs))
    if previous is None:   # the level after the last computed one failed: its session (and snapshots) are gone
        summary['snapshot_times'], summary['snapshot_labels'], summary['live_snapshot'] = [], [], None
    elif kept_snapshots:
        previous.snapshots[:0] = kept_snapshots
        summary['snapshot_times'] = [float(snapshot.time) for snapshot in previous.snapshots]
        summary['snapshot_labels'] = [snapshot.label for snapshot in previous.snapshots]
        summary['live_snapshot'] = live_snapshot_index(previous.snapshots)
    if stop_reason == 'cancelled':
        summary['status'] = 'cancelled'
    emit(LogEvent('info' if report['converged'] else 'warning', f"mesh refinement: {report['message']}"))
    summary['sweep'], summary['sweep_index'] = None, None
    return summary


def _setup(spec, registry):
    """_Setup of spec.study.adaptation; ValueError when its meshes cannot be refined that way."""
    adaptation = spec.study.adaptation
    parameters = evaluate_parameters(spec.parameters)
    meshes = list(adaptation.meshes) or [mesh_spec.name for mesh_spec in spec.used_meshes()]
    if adaptation.method == 'adaptive':
        refined_axes(spec, parameters)   # ValueError in 3D
    return _Setup(adaptation, parameters, meshes, spec.dimensions)


def _stage_name(level):
    return f'Mesh level {level}'


def _cells_text(estimates, meshes):
    """'mesh: 40,000 cells' per refined mesh (the others named 'fixed')."""
    return ', '.join(f"{name}: {estimate.n_cells:,} {'elements' if estimate.kind != 'tensor' else 'cells'}"
                     f"{'' if name in meshes else ' (fixed)'}" for name, estimate in estimates.items())


def _level_record(level, estimates, meshes, summary, target, previous_levels, session):
    """Convergence-report row of one level: cells (padding included) and window cells per mesh (triangles: the built
    elements), refined_cells (window cells of the refined meshes: the Richardson N), the target value and its relative
    change, every derived value, study seconds and the estimated gpu_steady of the build."""
    value = float(summary['derived'].get(target, math.nan))
    previous = previous_levels[-1]['value'] if previous_levels else None
    estimate_record = (session.build_memory or {}).get('estimate') or {}
    cells, window = _built_counts(session, estimates)
    return {'level': level, 'cells': cells, 'window_cells': window, 'n_cells': int(sum(cells.values())),
            'refined_cells': int(sum(window[name] for name in meshes if name in window)),
            'value': value, 'relative_change': None if previous is None else relative_change(previous, value),
            'derived': {name: float(item) for name, item in summary['derived'].items()},
            'study_seconds': float(summary.get('study_seconds') or 0.0),
            'gpu_steady': estimate_record.get('gpu_steady'), 'status': summary['status']}


def _next_level(session, level_spec, setup, estimates, registry, emit):
    """(next level's spec, its {mesh: estimate}, None, '') or (None, None, stop reason, message): refined meshes, then
    the growth, max_cells and memory-budget checks of that spec (CPU estimates)."""
    adaptation = setup.adaptation
    try:
        next_spec = level_spec.copy()
        shapes = resolve_geometry(level_spec, setup.parameters, errors={})
        for name in setup.meshes:
            frozen = frozen_mesh_spec(level_spec, level_spec.mesh_by_name(name), setup.parameters, registry, shapes)
            if adaptation.method == 'uniform':
                refined = uniform_mesh_spec(frozen, adaptation.growth ** (-1.0 / setup.dimensions))
            else:
                parts = mesh_indicator(session, level_spec, name, estimates[name], registry, emit)
                triangle_mesh, order = _built_triangles(session, level_spec, name)
                adapted = adaptive_mesh_spec(level_spec, frozen, estimates[name], parts, adaptation.growth,
                                             adaptation.anisotropic, setup.parameters, registry, shapes,
                                             triangle_mesh=triangle_mesh, element_order=order)
                if adapted is None:
                    emit(LogEvent('warning', f"mesh refinement: the error indicator of mesh '{name}' is zero "
                                             'everywhere; it is not refined'))
                    continue
                refined = adapted[0]
            next_spec.meshes = [refined if mesh_spec.name == name else mesh_spec for mesh_spec in next_spec.meshes]
        next_estimates = {mesh_spec.name: estimate_mesh(next_spec, setup.parameters, registry, mesh_spec.name)
                          for mesh_spec in next_spec.used_meshes()}
        memory = estimate_memory(next_spec, setup.parameters, registry, mesh_estimate=next_estimates)
    except Exception as exc:
        if gpu_memory.is_out_of_memory(exc):
            raise
        return None, None, 'failed', f'the next level could not be planned: {_failure_message(exc)}'
    current = sum(estimate.n_cells for estimate in estimates.values())
    planned = sum(estimate.n_cells for estimate in next_estimates.values())
    stop_reason, message = next_level_stop(current, planned, adaptation.max_cells,
                                           _memory_problem(level_spec, memory))
    return (next_spec, next_estimates, None, '') if stop_reason is None else (None, None, stop_reason, message)


def _built_triangles(session, spec, mesh_name) -> tuple:
    """(fem TriangleMesh, smallest element order of its physics' solvers (default 2)) of a built triangle mesh; (None,
    2) for a tensor mesh."""
    domain = session.domains.get(mesh_name)
    if domain is None or domain.grid_kind != 'triangles':
        return None, 2
    orders = [int(getattr(session.physics[physics.name], 'order', 2)) for physics in spec.physics_on(mesh_name)
              if physics.name in session.physics]
    return domain.mesh, min(orders, default=2)


def _built_counts(session, estimates) -> tuple:
    """({mesh: cells}, {mesh: window cells}) of a level: the estimates' counts, the built elements on triangle /
    tetrahedron meshes (their estimate is a size-field integral)."""
    cells, window = {}, {}
    for name, estimate in estimates.items():
        domain = session.domains.get(name) if session is not None else None
        if domain is not None and domain.grid_kind != 'tensor':
            cells[name] = int(domain.mesh.n_elements)
            window[name] = int(np.count_nonzero(domain.window_elements()))
        else:
            cells[name], window[name] = int(estimate.n_cells), window_cells(estimate)
    return cells, window


def _memory_problem(spec, memory):
    """Text when the estimate exceeds the GPU budget (gpu backend with a budget set) or the available host RAM (cpu
    backend: device arrays included, meshing.backend_memory), else ''."""
    budget = gpu_memory.budget_bytes()
    memory = backend_memory(spec, memory)
    if solver_backend(spec) == 'gpu' and budget is not None and memory.device_peak > budget:
        return (f'its GPU estimate {gpu_memory.format_bytes(memory.device_peak)} exceeds the budget '
                f'{gpu_memory.format_bytes(budget)}')
    available = host_memory.available_bytes()
    if available is not None and memory.host > available:
        return (f'its host estimate {gpu_memory.format_bytes(memory.host)} exceeds the available RAM '
                f'{gpu_memory.format_bytes(available)}')
    return ''


# ---- error indicator ----
def mesh_indicator(session, spec, mesh_name, estimate, registry, emit):
    """eta^2 of one mesh, tensor: parts (2, *window shape), triangles: per element (n_elements,) of the built mesh; per
    physics on it its solver's error_indicator() (optional hook, same layout; the only source on triangles) or the
    interpolation indicator of its PhysicsType.indicator_fields (() = the primary output field),
    each physics scaled to a unit total; a physics that gives none is skipped (warning)."""
    if estimate.kind == 'triangles':
        widths, shape = None, (session.domains[mesh_name].mesh.n_elements,)
    else:
        widths = window_widths(estimate)
        shape = (2,) + tuple(axis_widths.size for axis_widths in widths)
    parts = np.zeros(shape)
    for physics_spec in spec.physics_on(mesh_name):
        solver = session.physics.get(physics_spec.name)
        if solver is None:
            continue
        try:
            own = _physics_indicator(session, solver, physics_spec, registry, estimate, widths, shape)
        except Exception as exc:   # one physics without an indicator must not stop the refinement
            if gpu_memory.is_out_of_memory(exc):
                raise
            emit(LogEvent('warning', f"mesh refinement: no error indicator of '{physics_spec.name}' "
                                     f'({_failure_message(exc)}); it does not steer the refinement'))
            continue
        weight = float(own.sum())
        if math.isfinite(weight) and weight > 0.0:
            parts += own / weight
    return parts


def _physics_indicator(session, solver, physics_spec, registry, estimate, widths, shape):
    """eta^2 of one physics in the mesh_indicator layout `shape` (widths None on triangles)."""
    if isinstance(solver, ErrorIndicatorSolver):
        own = np.asarray(to_host(solver.error_indicator()), dtype=np.float64)
        if own.shape != shape:
            raise ValueError(f'error_indicator() has shape {own.shape}, expected {shape}')
        return np.nan_to_num(own, nan=0.0, posinf=0.0, neginf=0.0)
    if widths is None:
        raise ValueError('it has no error indicator on triangle meshes')
    renderer = session.renderer(physics_spec.name)
    names = registry.physics_type(physics_spec.type).indicator_fields
    if not names:
        primary = primary_field_info(renderer.field_infos.values())
        if primary is None:
            raise ValueError('it has no output field')
        names = (primary.name,)
    own = np.zeros(shape)
    for component in names:
        if component in renderer.field_infos:
            values = renderer.physical_field(component)
        else:   # a state field: full grid -> window
            sides = estimate.npml_sides
            state = to_host(solver.state()[component])
            values = np.asarray(state)[sides.get('x_lo', 0):state.shape[0] - sides.get('x_hi', 0),
                                       sides.get('y_lo', 0):state.shape[1] - sides.get('y_hi', 0)]
        values = np.asarray(values, dtype=np.float64)
        if values.shape != shape[1:]:
            raise ValueError(f'field {component} has shape {values.shape}, the mesh window {shape[1:]}')
        own += interpolation_indicator(relative_field(values), widths)
    return own


# ---- stop rule and report (pure) ----
def relative_change(previous, current) -> float:
    """|current - previous| / |current| (0 when both are 0, inf when only current is 0; NaN propagates)."""
    difference = abs(current - previous)
    if current == 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / abs(current)


def level_stop(values, tolerance, level, max_levels):
    """(stop reason or None, message) after `level` from the target values of every level so far: converged when
    |v_k - v_k-1| <= tolerance |v_k|, not_finite, max_levels at the last level."""
    value = values[-1]
    if not math.isfinite(value):
        return 'not_finite', f'the target value is {value} at level {level}; refinement stopped'
    change = relative_change(values[-2], value) if len(values) > 1 else None
    if change is not None and abs(value - values[-2]) <= tolerance * abs(value):
        return 'converged', (f'converged at level {level}: relative change {change:.3g} <= tolerance '
                             f'{tolerance:.3g}')
    if level >= max_levels:
        last = '' if change is None else f' (last relative change {change:.3g})'
        return 'max_levels', (f'not converged after {max_levels} refinement level{"s" if max_levels != 1 else ""}'
                              f'{last}; tolerance {tolerance:.3g}')
    return None, ''


def next_level_stop(current_cells, planned_cells, max_cells, memory_problem=''):
    """(stop reason or None, message) of a planned next level: no_refinement when it does not grow, max_cells above
    the cap, memory when memory_problem names one."""
    if planned_cells <= current_cells:
        return 'no_refinement', (f'the next level would not add cells ({planned_cells:,} vs {current_cells:,}); '
                                 'refinement stopped, not converged')
    if max_cells is not None and planned_cells > max_cells:
        return 'max_cells', (f'the next level ({planned_cells:,} cells) would exceed max_cells {max_cells:,}; not '
                             'converged')
    if memory_problem:
        return 'memory', f'the next level ({planned_cells:,} cells) does not fit: {memory_problem}; not converged'
    return None, ''


def richardson(cells, values, dimensions=2):
    """(record, '') of v = v_inf + C N^(-q) through three levels (cells N growing, finite values): order_cells q,
    order q x dimensions (in cell size), extrapolated v_inf, error_estimate |v_3 - v_inf| (+ relative), coefficient C;
    (None, reason) when the changes alternate in sign or do not shrink."""
    cells = [float(count) for count in cells]
    values = [float(value) for value in values]
    if len(cells) != 3 or not 0.0 < cells[0] < cells[1] < cells[2] or not all(map(math.isfinite, values)):
        return None, 'needs three levels with growing cell counts and finite values'
    first, second = values[1] - values[0], values[2] - values[1]
    record = {'cells': cells, 'values': values, 'dimensions': int(dimensions)}
    if second == 0.0:
        record.update(order_cells=None, order=None, extrapolated=values[2], error_estimate=0.0,
                      relative_error_estimate=0.0, coefficient=0.0)
        return record, ''
    if first == 0.0 or (first > 0.0) != (second > 0.0):
        return None, 'the changes between the last three levels alternate in sign (not in the asymptotic range)'
    a, b = cells[1] / cells[0], cells[2] / cells[0]
    ratio = first / second

    def change_ratio(order):
        return (1.0 - a ** -order) / (a ** -order - b ** -order)

    if ratio <= math.log(a) / math.log(b / a):
        return None, 'the change does not shrink with the cell count (order <= 0)'
    low, high = 0.0, 1.0
    while change_ratio(high) < ratio and high < _RICHARDSON_MAX_ORDER:
        high *= 2.0
    if change_ratio(high) < ratio:
        return None, f'the fitted order in cells exceeds {_RICHARDSON_MAX_ORDER:g}'
    for _ in range(_RICHARDSON_BISECTIONS):
        middle = 0.5 * (low + high)
        if change_ratio(middle) < ratio:
            low = middle
        else:
            high = middle
        if high - low <= 1e-14 * high:
            break
    order = 0.5 * (low + high)
    scale = second / (b ** -order - a ** -order)        # v = v_inf + scale (N / N_1)^-q
    extrapolated = values[2] - scale * b ** -order
    error = abs(values[2] - extrapolated)
    log_coefficient = math.log(abs(scale)) + order * math.log(cells[0])
    record.update(order_cells=order, order=order * dimensions, extrapolated=extrapolated, error_estimate=error,
                  relative_error_estimate=error / abs(extrapolated) if extrapolated != 0.0 else math.inf,
                  coefficient=math.copysign(math.exp(log_coefficient) if log_coefficient < 700.0 else math.inf, scale))
    return record, ''


def cells_for_error(record, relative_error) -> float:
    """Cell count N at which a richardson() record's |C| N^(-q) reaches relative_error x |v_inf| (inf without an
    order)."""
    order = record.get('order_cells')
    if not order or record['extrapolated'] == 0.0 or record['coefficient'] == 0.0:
        return math.inf
    log_cells = (math.log(abs(record['coefficient'])) - math.log(relative_error * abs(record['extrapolated']))) / order
    return math.exp(min(log_cells, 700.0))


def convergence_report(adaptation, setup, levels, stop_reason, message, last_spec) -> dict:
    """summary 'convergence' / convergence.json: settings, levels, Richardson over the last three completed levels,
    converged, stop_reason, message and the last level's refined MeshSpecs (JSON)."""
    usable = [record for record in levels if record['status'] == 'completed' and math.isfinite(record['value'])]
    fit, note = richardson([record['refined_cells'] for record in usable[-3:]],
                           [record['value'] for record in usable[-3:]], setup.dimensions)
    if fit is not None:
        fit['levels'] = [record['level'] for record in usable[-3:]]
    mesh_dicts = last_spec.to_dict()['meshes']
    if fit is not None and fit['order_cells'] is not None:
        message = (f"{message}; Richardson: {fit['extrapolated']:.8g} +- {fit['error_estimate']:.3g} (order "
                   f"{fit['order_cells']:.3g} in cells)")
    return {'target': adaptation.target, 'method': adaptation.method, 'tolerance': float(adaptation.tolerance),
            'max_levels': int(adaptation.max_levels), 'growth': float(adaptation.growth),
            'max_cells': adaptation.max_cells, 'anisotropic': bool(adaptation.anisotropic), 'meshes': list(setup.meshes),
            'levels': levels, 'richardson': fit, 'richardson_note': note,
            'converged': stop_reason == 'converged', 'stop_reason': stop_reason, 'message': message,
            'adapted_meshes': [mesh for mesh in mesh_dicts if mesh.get('name') in setup.meshes]}
