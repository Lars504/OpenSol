"""Parity-sector runs of one study (symmetry 2.6, study setting parity_sectors): one build and run per symmetric /
antisymmetric combination of the model's mirror faces (per sweep value with a parameter sweep), merged into one
summary and one snapshot list."""
import math
from dataclasses import dataclass, replace

import numpy as np

from ..core.mirror import (MIRROR_SIGNS, combine_sector_fields, combine_sector_port_totals, combine_sector_result,
                           combine_sector_totals, field_parity, mirror_faces, sector_specs, unfold_field)
from .analytic import derived_curve_result, derived_references, is_curve_source, result_value
from .expressions import ExpressionError, compile_expression, evaluate_scalar, referenced_names
from .session import Snapshot, earlier_derived_names, live_snapshot_index
from .study import (LogEvent, StudyFailed, StudyStarted, SweepStep, SweepValueDone, _KeptSweepFields,
                    _append_derived_row, _fail_sweep_setup, _feature_results, _record_failure, _run_built,
                    build_session, release_session, summed_runs, sweep_table_copy, sweep_value_spec, sweep_values_of)

EIGEN_STUDIES = ('mode_analysis', 'eigenfrequency')   # numbered solutions merged sector by sector, nothing combined
_COUNT_KEYS = ('n_modes', 'n_resonances')             # summary counts of the eigen studies (sum over the sectors)
# source-free operations that return a field-free expression's value (field reductions of a constant, 'expression')
_CONSTANT_OPERATIONS = ('mean', 'max', 'min', 'at', 'expression')
_PHASE_PARITY_PREFIX = 'phase_'   # = core.mirror's FieldInfo.parity prefix of phasor arguments


@dataclass
class _SectorRuns:
    """Outcome of the sector loop of one spec: runs [(label, sector spec, summary, feature results, snapshots)], the
    session of the last built sector (live; a failed run's too) or None, status, failure (message, traceback,
    out_of_memory, build_memory) or None, the number of sectors and whether StudyStarted was emitted."""
    runs: list
    session: object
    status: str
    failure: tuple | None
    n_sectors: int
    started: bool


def run_parity_sectors(spec, registry, build, emit, control, service, on_session):
    """Build and run every parity sector of `spec` (the model's own mirror signs last, so its session stays live;
    with a sweep per sweep value) -> the last sector's summary with merged snapshots, combined derived values and the
    'parity_sectors' table (+ 'sweep'), or None after a failure (StudyFailed emitted)."""
    try:
        ordered_sector_specs(spec)   # ValueError without a mirror face
        sweep = None if spec.study.sweep is None else sweep_values_of(spec)
    except Exception as exc:   # text only; reported through the build path with the node paths
        _fail_sweep_setup(spec, registry, build, emit, on_session, _record_failure(exc), what='parity sector')
        return None
    if sweep is not None:
        return _run_sector_sweep(spec, *sweep, registry, build, emit, control, service, on_session)
    outcome = _run_sectors(spec, registry, build, emit, control, service, on_session, None, False)
    if outcome.failure is not None:
        message, failure_traceback, out_of_memory, build_memory = outcome.failure
        if outcome.session is not None and out_of_memory:   # as run_study: an OOM closes the session that ran out
            build_memory = getattr(outcome.session, 'build_memory', None)
            release_session(on_session, outcome.session, emit)
        emit(StudyFailed(message, failure_traceback, out_of_memory=out_of_memory,
                         session_live=outcome.session is not None and not out_of_memory,
                         build_memory=build_memory if out_of_memory else None))
        return None
    summary = _merge_runs(spec, outcome, emit)
    summary['sweep'], summary['sweep_index'] = None, None
    return summary


def _run_sectors(spec, registry, build, emit, control, service, on_session, previous_session, started):
    """Build and run every ordered_sector_specs copy of spec, one session at a time (previous_session, a live session
    of an earlier sweep value or None, is closed before the first build) -> _SectorRuns; a failed build or run stops."""
    sectors = ordered_sector_specs(spec)
    runs, unavailable, status = [], set(), 'completed'
    for index, (label, sector_spec) in enumerate(sectors):
        if index > 0 and control.checkpoint_cancelled(service):
            status = 'cancelled'
            break
        release_session(on_session, previous_session, emit if previous_session is not None else None)
        previous_session = None
        emit(LogEvent('info', f'parity sector {index + 1}/{len(sectors)}: {label}'))
        session, failure = build_session(sector_spec, emit, on_session, build=build, registry=registry)
        if failure is not None:
            return _SectorRuns(runs, None, status, (f'parity sector {label}: build failed: {failure.message}',
                                                    failure.traceback, failure.out_of_memory,
                                                    failure.build_memory if failure.out_of_memory else None),
                               len(sectors), started)
        if not started:
            emit(StudyStarted())
            started = True
        run_failure = None   # (message, traceback text, out_of_memory): the exception itself is not kept
        try:
            summary = _run_built(session, sector_spec, registry, emit, control, service)
            cancelled = summary['status'] == 'cancelled'
            results = {} if cancelled else _feature_results(session, emit, unavailable)
        except Exception as exc:
            run_failure = _record_failure(exc)
        if run_failure is not None:
            message, failure_traceback, out_of_memory = run_failure
            return _SectorRuns(runs, session, status, (f'parity sector {label}: {message}', failure_traceback,
                                                       out_of_memory, None), len(sectors), started)
        runs.append((label, sector_spec, summary, results, list(session.snapshots)))
        previous_session = session
        if cancelled:
            status = 'cancelled'
            break
    return _SectorRuns(runs, previous_session, status, None, len(sectors), started)


def _merge_runs(spec, outcome, emit) -> dict:
    """Summary of a sector loop that ran: the last run's, its live session's snapshots replaced by the merged ones,
    the combined derived values and the 'parity_sectors' table."""
    eigen = spec.study.type in EIGEN_STUDIES
    runs, session = outcome.runs, outcome.session
    complete = outcome.status == 'completed' and len(runs) == outcome.n_sectors
    snapshots, snapshot_sectors = _merged_snapshots(runs, eigen, complete, emit)
    session.snapshots[:] = snapshots
    summary = dict(runs[-1][2])
    summary.update(_merged_summary(spec, session, runs, eigen, complete, emit))
    if outcome.status != 'completed':
        summary['status'] = outcome.status
    summary['snapshot_times'] = [float(snapshot.time) for snapshot in snapshots]
    summary['snapshot_labels'] = [snapshot.label for snapshot in snapshots]
    summary['live_snapshot'] = live_snapshot_index(snapshots)
    summary['parity_sectors']['snapshot_sectors'] = snapshot_sectors
    return summary


def _run_sector_sweep(spec, parameter, sweep_values, registry, build, emit, control, service, on_session):
    """Every sweep value's parity sectors (cold starts) -> the last computed value's merged summary + 'sweep' (derived =
    the combined values, results = the last sector's, sector_derived = the per-sector values; failed values NaN / {}),
    or None (StudyFailed) when no value was computed."""
    derived_names = [derived.name for derived in spec.derived_values]
    reference_names = [derived.name for derived in spec.derived_values if derived.reference is not None]
    table = {'parameter': parameter, 'values': [], 'derived': {name: [] for name in derived_names},
             'reference': {name: [] for name in reference_names}, 'error': {name: [] for name in reference_names},
             'results': [], 'failed': [], 'converged': [], 'sector_derived': [], 'derived_series': []}
    kept_fields = _KeptSweepFields(emit) if spec.study.sweep.keep_fields else None
    if kept_fields is not None:
        table['snapshots'] = []   # per value: its merged host Snapshots ([] for a failed value)
    last_summary, last_index, previous_session, started, status = None, None, None, False, 'completed'
    first_failure = None   # (message, traceback text, out_of_memory) of the first failed value
    value_runs = []        # [(label, merged summary)] of every computed value (summed_runs)
    for index, value in enumerate(sweep_values):
        if index > 0 and control.checkpoint_cancelled(service):
            status = 'cancelled'
            break
        emit(SweepStep(index, len(sweep_values), parameter, value))
        value_spec = sweep_value_spec(spec, parameter, value)
        outcome = _run_sectors(value_spec, registry, build, emit, control, service, on_session, previous_session,
                               started)
        started, previous_session = outcome.started, outcome.session
        summary = None
        if outcome.failure is not None:
            message, failure_traceback, out_of_memory, _ = outcome.failure
            first_failure = first_failure or (message, failure_traceback, out_of_memory)
            emit(LogEvent('error', f'sweep {parameter} = {value:g}: {message}; its results are NaN'))
            table['failed'].append({'index': index, 'value': value, 'message': message})
            if outcome.session is not None:   # a failed run's session: closed now (an OOM frees the GPU)
                release_session(on_session, outcome.session, emit)
                previous_session = None
        else:
            summary = _merge_runs(value_spec, outcome, emit)
            last_summary, last_index = summary, index
            value_runs.append((f'{parameter} = {value:g}', summary))
            if summary['status'] == 'cancelled':   # partial run: not a sweep point
                status = 'cancelled'
                break
        sectors = None if summary is None else summary['parity_sectors']
        derived = {} if summary is None else summary['derived']
        fixed_point = None if summary is None else summary.get('fixed_point')
        table['values'].append(value)
        table['converged'].append(None if fixed_point is None else fixed_point['converged'])
        _append_derived_row(table, derived_names, reference_names, spec, value_spec, derived)
        table['results'].append({} if not sectors or not sectors['results'] else dict(sectors['results'][-1]))
        table['sector_derived'].append({} if sectors is None else {name: list(values)
                                                                   for name, values in sectors['derived'].items()})
        table['derived_series'].append(None)   # sector runs: no combined derived series
        if kept_fields is not None:
            table['snapshots'].append([] if summary is None else kept_fields.take(outcome.session.snapshots))
        emit(SweepValueDone(index, len(sweep_values), parameter, value, sweep_table_copy(table)))
    if last_summary is None:
        if status == 'cancelled':
            emit(StudyFailed('the sweep was cancelled before any value was computed', '',
                             session_live=previous_session is not None))
        else:
            message, failure_traceback, out_of_memory = first_failure
            emit(StudyFailed(f"no value of the sweep over '{parameter}' could be computed; first error: {message}",
                             failure_traceback, out_of_memory=out_of_memory))
        return None
    if previous_session is None:   # the last value failed: last_summary's snapshots went with its session
        last_summary['snapshot_times'] = []
        last_summary['snapshot_labels'] = []
        last_summary['live_snapshot'] = None
    last_summary.update(summed_runs(value_runs))
    last_summary['status'] = status
    last_summary['sweep'], last_summary['sweep_index'] = table, last_index
    return last_summary


def ordered_sector_specs(spec) -> list:
    """[(label, spec copy with parity_sectors off)] of core.mirror.sector_specs, the combination of the model's own
    mirror signs (when every physics agrees on them) last."""
    faces = mirror_faces(spec)
    sign_names = {sign: name for name, sign in MIRROR_SIGNS.items()}
    own_label = ', '.join(f'{face} {sign_names[faces[face]]}' for face in sorted(faces))
    sectors = list(sector_specs(spec).items())
    sectors.sort(key=lambda item: item[0] == own_label)   # stable: the own combination moves to the end
    for _, sector in sectors:
        sector.study.settings['parity_sectors'] = False
    return sectors


def _unfolded(fields, sector_spec):
    """{key: FieldData} unfolded over the sector's mirror faces (a field without a parity stays as it is)."""
    unfolded = {}
    for key, field_data in fields.items():
        try:
            unfolded[key] = unfold_field(field_data, mirror_faces(sector_spec, field_data.physics))
        except ValueError:   # no parity declared: kept on the drawn half
            unfolded[key] = field_data
    return unfolded


def _combined(field_sets):
    """{key: combine_sector_fields} over the sectors' unfolded {key: FieldData} of every linear signed field."""
    combined = {}
    for key in field_sets[0]:
        try:
            combined[key] = combine_sector_fields([fields[key] for fields in field_sets])
        except (KeyError, ValueError):   # unsigned / phase fields do not superpose
            continue
    return combined


def _combined_segments(segment_sets):
    """{curve name: LineProfile} of the drawn source alone on the drawn half: the mean of the sectors' segment
    profiles of every linear signed field (the drawn-half part of combine_sector_fields)."""
    combined = {}
    for name, first in segment_sets[0].items():
        profiles = [segments.get(name) for segments in segment_sets]
        try:
            parity = field_parity(first.info)
        except ValueError:   # no parity declared: not combined
            continue
        if (parity == 'even' or parity.startswith('phase_') or None in profiles
                or any(profile.values.shape != first.values.shape for profile in profiles)):
            continue
        combined[name] = replace(first, values=np.mean([profile.values for profile in profiles], axis=0))
    return combined


def _merged_snapshots(runs, eigen, complete, emit):
    """(snapshots, sector label per snapshot ('' = combined)): eigen studies every sector's snapshots in sector order
    (labels carry the parity); driven studies, when every sector ran, one combined snapshot per snapshot index first,
    then the sector snapshots labelled '<label> [<sector>]'; every sector field unfolded where it has a parity."""
    unfolded = []
    for label, sector_spec, _, _, snapshots in runs:
        unfolded.append([Snapshot(time=snapshot.time, fields=_unfolded(snapshot.fields, sector_spec),
                                  label=snapshot.label if eigen else _sector_label(snapshot.label, label),
                                  cuts=[_unfolded(cut, sector_spec) for cut in snapshot.cuts],
                                  segments=dict(snapshot.segments),   # segments: drawn half, not unfolded
                                  live=snapshot.live)   # the last sector's: the live session's state
                         for snapshot in snapshots])
    merged, sectors = [], []
    counts = {len(snapshots) for snapshots in unfolded}
    if not eigen and complete and len(counts) == 1:
        skipped = set()
        for index, original in enumerate(runs[0][4]):
            group = [snapshots[index] for snapshots in unfolded]
            fields = _combined([snapshot.fields for snapshot in group])
            skipped.update(key for key in group[0].fields if key not in fields)
            if not fields:
                continue
            cuts = [_combined([snapshot.cuts[cut] for snapshot in group]) for cut in range(len(group[0].cuts))]
            merged.append(Snapshot(time=original.time, fields=fields, label=original.label or 'combined', cuts=cuts,
                                   segments=_combined_segments([snapshot.segments for snapshot in group])))
            sectors.append('')
        if skipped:
            emit(LogEvent('info', f"parity sectors: {', '.join(sorted(skipped))} not combined (only linear signed "
                                  'fields superpose: Re / Im phasors, raw components); see the sector snapshots'))
    for (label, _, _, _, _), snapshots in zip(runs, unfolded):
        merged.extend(snapshots)
        sectors.extend([label] * len(snapshots))
    return merged, sectors


def _sector_label(label, sector):
    """'<label> [<sector>]' of a driven sector snapshot ('sector [<sector>]' for an unlabelled one)."""
    return f'{label} [{sector}]' if label else f'sector [{sector}]'


def _merged_summary(spec, session, runs, eigen, complete, emit) -> dict:
    """Summary entries of the merged run: derived (complete driven runs: per value its sector rule, _combined_derived;
    else NaN) with references / errors, summed ticks / study_seconds / time_check / eigen counts, and the
    'parity_sectors' table."""
    names = [derived.name for derived in spec.derived_values]
    per_sector = {name: [float(summary['derived'].get(name, math.nan)) for _, _, summary, _, _ in runs]
                  for name in names}
    faces = sorted(mirror_faces(spec))
    sector_results = [results for _, _, _, results, _ in runs]
    combined_results = _combined_results(sector_results, len(faces)) if complete and not eigen else {}
    if names and complete and not eigen:
        derived, rules = _combined_derived(spec, session, per_sector, combined_results)
        recomputed = _recompute_value_expressions(spec, session.parameters, derived)
        by_rule = {rule: [name for name in names if rules[name] == rule and name not in recomputed]
                   for rule in ('quadratic', 'linear', 'result', None)}
        texts = [f"{label}: {', '.join(by_rule[rule])}" for rule, label in (
            ('quadratic', f'quadratic totals (sector mean / {len(runs)}, regions symmetric about every mirror plane)'),
            ('linear', 'linear (sector mean)'), ('result', 'from the combined feature results'))
            if by_rule[rule]]
        if recomputed:
            texts.append(f"recomputed from the combined values: {', '.join(recomputed)}")
        emit(LogEvent('info', 'parity sectors: derived values combined as ' + '; '.join(texts)
                      + '; per-sector values: summary parity_sectors.derived'))
        if by_rule[None]:
            emit(LogEvent('warning', f"parity sectors: no sector combination for {', '.join(by_rule[None])} (NaN: max "
                                     '/ min / mode volume, phases, or neither linear in signed fields nor quadratic; '
                                     'a feature result without a combination rule); see summary parity_sectors.derived'))
    else:
        derived = {name: math.nan for name in names}
        if names:
            reason = 'an eigen study' if eigen else 'not every sector ran'
            emit(LogEvent('warning', 'parity sectors: derived values are per sector here (summary parity_sectors.'
                                     f'derived); the combined ones are NaN ({reason})'))
    references, errors, messages = derived_references(spec.derived_values, session.parameters, derived)
    for message in messages:
        emit(LogEvent('warning', message))
    merged = {'derived': derived, 'derived_reference': references, 'derived_error': errors,
              'derived_series': None,   # per sector only: the combination is not formed along the study axis
              **summed_runs((label, summary) for label, _, summary, _, _ in runs)}
    for key in _COUNT_KEYS:
        if key in runs[-1][2]:
            merged[key] = sum(int(summary.get(key) or 0) for _, _, summary, _, _ in runs)
    merged['parity_sectors'] = {
        'faces': faces, 'labels': [label for label, _, _, _, _ in runs],
        'status': [summary['status'] for _, _, summary, _, _ in runs], 'combined': complete and not eigen,
        'derived': per_sector, 'results': sector_results, 'combined_results': combined_results}
    return merged


def _combined_derived(spec, session, per_sector, combined_results):
    """({name: combined value}, {name: rule}) of complete driven sector runs: a feature-sourced value from the combined
    feature results ('result'), a field value by _field_sector_rule ('linear': sector mean, 'quadratic': mean / 2^k),
    NaN where no rule applies (rule None)."""
    derived, rules = {}, {}
    for derived_spec in spec.derived_values:
        name, values = derived_spec.name, per_sector[derived_spec.name]
        if derived_spec.source is not None:
            rules[name] = 'result'
            derived[name] = _combined_source_value(spec, session, derived_spec, combined_results, derived)
            if not math.isfinite(derived[name]):
                rules[name] = None
            continue
        rules[name] = _field_sector_rule(derived_spec, session)
        derived[name] = (float(np.mean(values)) if rules[name] == 'linear' else
                         float(combine_sector_totals(values)) if rules[name] == 'quadratic' else math.nan)
    return derived, rules


def _combined_source_value(spec, session, derived_spec, combined_results, earlier) -> float:
    """A feature-sourced derived value (or one of a derived curve of them) from the combined feature results; NaN
    when one is not combined or the value fails."""
    def fetch(key):
        return combined_results[key]   # KeyError: no combination rule for that result
    try:
        if is_curve_source(derived_spec.source):
            result = derived_curve_result(spec.derived_curve(derived_spec.source), fetch, session.parameters)
        else:
            result = fetch(derived_spec.source)
        return float(result_value(result, derived_spec, {**earlier, **session.parameters}))
    except Exception:   # any failure of user input -> NaN (reported with the uncombined values)
        return math.nan


_LINEAR_SECTOR_OPERATIONS = ('integral', 'mean', 'at')   # the reduction of the mean field = the mean of reductions
_QUADRATIC_SECTOR_OPERATIONS = ('integral', 'mean')     # cross terms of the sectors vanish over symmetric regions
_PROBE_SAMPLES = 8
_PROBE_RTOL = 1e-9


def _field_sector_rule(derived_spec, session):
    """'linear', 'quadratic' or None for a field derived value (source None), probed numerically on random field
    values: no field read -> 'linear' (equal in every sector); signed fields only: odd and additive -> 'linear', an even
    quadratic form -> 'quadratic'; unsigned (even) fields only: additive -> 'quadratic'; phases, a mix, scalar outputs
    or another form -> None. The operation must fit the rule (_LINEAR / _QUADRATIC_SECTOR_OPERATIONS)."""
    try:
        compiled = compile_expression(derived_spec.expression)
    except ExpressionError:
        return None
    kinds = {}
    for name in compiled.names:
        if name in session._scalar_names:
            return None
        if name not in session._field_names:
            continue
        physics, field_name = session._field_names[name]
        try:
            parity = field_parity(session.renderer(physics).field_info(field_name))
        except (KeyError, ValueError, RuntimeError):   # unknown field, no parity, closed session: no rule
            return None
        if parity.startswith(_PHASE_PARITY_PREFIX):
            return None
        kinds[name] = 'unsigned' if parity == 'even' else 'signed'
    if not kinds:
        return 'linear'
    if len(set(kinds.values())) > 1:
        return None
    rng = np.random.default_rng(20261104)
    constants = {name: float(rng.uniform(0.5, 1.5)) for name in compiled.names if name not in kinds}
    constants.update({name: session.parameters[name] for name in compiled.names if name in session.parameters})
    signed = 'signed' in kinds.values()

    def draw():
        return {name: rng.standard_normal(_PROBE_SAMPLES) if signed else rng.uniform(0.5, 2.0, _PROBE_SAMPLES)
                for name in kinds}

    def f(fields, scale=1.0):
        try:
            with np.errstate(all='ignore'):
                values = compiled.evaluate({**constants, **{name: scale * value for name, value in fields.items()}},
                                           xp=np)
            return np.broadcast_to(np.asarray(values, dtype=np.float64), (_PROBE_SAMPLES,))
        except Exception:   # the expression does not evaluate on probe values: no rule
            return None

    def close(first, second):
        if first is None or second is None or not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
            return False
        scale = max(float(np.max(np.abs(first))), float(np.max(np.abs(second))))
        return float(np.max(np.abs(first - second))) <= _PROBE_RTOL * scale

    u, v = draw(), draw()
    f_u, f_v = f(u), f(v)
    added = {name: u[name] + v[name] for name in kinds}
    f_sum = f(added)
    additive = f_u is not None and f_v is not None and close(f_sum, f_u + f_v)
    if signed and additive and close(f(u, -1.0), -f_u):
        rule = 'linear'
    elif not signed and additive and close(f(u, 2.0), 2.0 * f_u):
        rule = 'quadratic'
    elif (signed and f_u is not None and f_v is not None and close(f(u, -1.0), f_u) and close(f(u, 2.0), 4.0 * f_u)
          and close(f_sum + f({name: u[name] - v[name] for name in kinds}), 2.0 * (f_u + f_v))):
        rule = 'quadratic'
    else:
        return None
    operations = _LINEAR_SECTOR_OPERATIONS if rule == 'linear' else _QUADRATIC_SECTOR_OPERATIONS
    return rule if derived_spec.operation in operations else None


def _recompute_value_expressions(spec, parameters, derived) -> list:
    """Re-evaluate in order, from the combined values, every derived value that reads no field (source None,
    operation mean / max / min / at / expression, an expression of parameters and earlier derived values: ratios such
    as a Purcell factor) instead of combining it; derived is updated in place (NaN when it does not evaluate). -> their
    names."""
    recomputed = []
    for derived_spec in spec.derived_values:
        if derived_spec.source is not None or derived_spec.operation not in _CONSTANT_OPERATIONS:
            continue
        try:
            names = referenced_names(derived_spec.expression)
        except ExpressionError:
            continue
        earlier = earlier_derived_names(spec, derived_spec)
        if not names <= set(parameters) | earlier:
            continue
        namespace = {**{name: derived[name] for name in earlier}, **parameters}   # parameters win, as in evaluate
        try:
            derived[derived_spec.name] = evaluate_scalar(derived_spec.expression, namespace)
        except (ExpressionError, ZeroDivisionError):   # non-finite, e.g. from a NaN earlier value
            derived[derived_spec.name] = math.nan
        recomputed.append(derived_spec.name)
    return recomputed


def _combined_results(sector_results, n_faces) -> dict:
    """{key: core.mirror.combine_sector_result} of every feature result every sector has and a rule combines; port
    incident powers / fractions then summed per physics (combine_sector_port_totals)."""
    keys = [key for key in sector_results[0] if all(key in results for results in sector_results[1:])]
    combined = {}
    for key in keys:
        try:
            result = combine_sector_result([results[key] for results in sector_results], n_faces)
        except (ValueError, TypeError):   # shapes differ between the sectors: none combined
            result = None
        if result is not None:
            combined[key] = result
    try:
        combine_sector_port_totals(combined)
    except (ValueError, TypeError):   # ports recorded different frequencies: each keeps its own incident power
        pass
    return combined
