"""Analytic example check (selftest --examples, CPU): per-example records of the derived-value, analytic-curve and
derived-curve errors against their tolerances, flags and the ranked report; api.selftest_examples runs the examples."""
import math

from .analytic import parameters_with
from .expressions import ExpressionError, evaluate_scalar
from .schema import MESH_KIND_DEFAULTS

ANALYTIC_REPORT_FILE = 'selftest_report_analytic.json'   # each example on its own backend
EXACT_ERROR = 1e-9   # below this a simulated quantity matches its reference suspiciously well
EXAMPLE_STATUSES = ('passed', 'failed', 'error', 'no_tolerance')
FLAGS = ('over_tolerance', 'nonfinite', 'not_compared', 'suspicious_exact', 'sweep_not_converging')
_FAILING_FLAGS = ('over_tolerance', 'nonfinite', 'not_compared')   # fail an entry that has a tolerance


def analytic_report_file(backend=None):
    """Report file name of an examples run: ANALYTIC_REPORT_FILE (own backends, None), else
    selftest_report_analytic_<backend>.json."""
    return ANALYTIC_REPORT_FILE if backend is None else f'selftest_report_analytic_{backend}.json'


def has_analytic_comparison(spec):
    """True when spec has a derived value or derived curve with a reference, or an analytic curve."""
    return (any(item.reference is not None for item in spec.derived_values + spec.derived_curves)
            or bool(spec.analytic_curves))


def failed_example_record(name, title, message, wall_time, out_dir=None, run_status=None):
    """Report entry of an example that did not complete (load failure, run raised or status other than completed)."""
    return {'example': name, 'title': title, 'status': 'error', 'run_status': run_status, 'message': message,
            'wall_time': wall_time, 'out_dir': out_dir, 'worst_ratio': None, 'flags': [], 'derived': [],
            'curves': []}


def example_record(name, spec, summary, wall_time, out_dir=None):
    """Report entry of one example run (summary: its summary.json dict): every derived value with a reference, every
    analytic curve and every derived curve with a reference (in 'curves', kind 'derived_curve') with error, tolerance,
    ratio and flags; status passed / failed / error / no_tolerance."""
    run_status = summary.get('status')
    if run_status != 'completed':
        return failed_example_record(name, spec.title, f"run status {run_status}: {summary.get('message') or ''}",
                                     wall_time, out_dir, run_status)
    sweep = summary.get('sweep')
    order = _refinement_order(spec, sweep)
    derived = [_derived_entry(item, summary, sweep, order) for item in spec.derived_values
               if item.reference is not None]
    curves = [_curve_entry(curve, summary, sweep, order) for curve in spec.analytic_curves]
    curves += [_derived_curve_entry(curve, summary) for curve in spec.derived_curves if curve.reference is not None]
    entries = derived + curves
    toleranced = [entry for entry in entries if entry['tolerance'] is not None]
    failing = [entry for entry in toleranced if any(flag in _FAILING_FLAGS for flag in entry['flags'])]
    status = 'failed' if failing else 'passed' if toleranced else 'no_tolerance'
    message = '; '.join(f"{entry['kind']} {entry['name']}: error {_text(entry['error'])} vs tolerance "
                        f"{_text(entry['tolerance'])} ({', '.join(entry['flags'])})" for entry in failing)
    ratios = [entry['ratio'] for entry in toleranced]
    return {'example': name, 'title': spec.title, 'status': status, 'run_status': run_status,
            'message': message or (summary.get('message') or ''), 'wall_time': wall_time, 'out_dir': out_dir,
            'worst_ratio': _worst(ratios) if ratios else None,
            'flags': [flag for flag in FLAGS if any(flag in entry['flags'] for entry in entries)],
            'derived': derived, 'curves': curves}


def analytic_report(records, skipped, wall_time, all_ran, warnings, device, version, backend=None,
                    skipped_backend=()):
    """analytic_report_file(backend) content: counts, the examples without a comparison (and those not run on the
    forced backend), the toleranced entries ranked by error / tolerance (non-finite first), the others ranked by
    error, and every example record."""
    counts = {status: sum(1 for record in records if record['status'] == status) for status in EXAMPLE_STATUSES}
    ranked, untoleranced = [], []
    for record in records:
        for entry in record['derived'] + record['curves']:
            row = {'example': record['example'], 'kind': entry['kind'], 'name': entry['name'], 'error': entry['error'],
                   'tolerance': entry['tolerance'], 'ratio': entry['ratio'], 'flags': entry['flags']}
            (untoleranced if entry['tolerance'] is None else ranked).append(row)
    ranked.sort(key=lambda row: _rank_key(row['ratio']))
    untoleranced.sort(key=lambda row: _rank_key(row['error']))
    return {'open_sol_version': version, 'passed': all_ran and counts['failed'] == 0 and counts['error'] == 0,
            'backend': backend, 'n_examples': len(records), 'n_passed': counts['passed'], 'n_failed': counts['failed'],
            'n_error': counts['error'], 'n_no_tolerance': counts['no_tolerance'], 'n_skipped': len(skipped),
            'skipped': list(skipped), 'n_skipped_backend': len(skipped_backend),
            'skipped_backend': list(skipped_backend), 'wall_time': wall_time, 'device': device, 'warnings': list(warnings),
            'ranking': ranked, 'without_tolerance': untoleranced, 'examples': records}


# ---- entries ----
def _derived_entry(derived, summary, sweep, order):
    """Entry of a derived value with a reference; a sweep: value / reference / error per sweep value, error = the
    worst of them."""
    name = derived.name
    if sweep is not None and name in (sweep.get('error') or {}):
        values, references = sweep['derived'].get(name, []), sweep['reference'].get(name, [])
        errors = [float(error) for error in sweep['error'][name]]
        error = _worst(errors)
    else:
        values = float(summary.get('derived', {}).get(name, math.nan))
        references = float(summary.get('derived_reference', {}).get(name, math.nan))
        errors = None
        error = float(summary.get('derived_error', {}).get(name, math.nan))
    reference_list = references if isinstance(references, list) else [references]
    flags = _flags(error, derived.tolerance, exact_possible=any(reference != 0.0 for reference in reference_list))
    if errors is not None and _not_converging(errors, order):
        flags.append('sweep_not_converging')
    entry = {'kind': 'derived', 'name': name, 'value': values, 'reference': references, 'error': error,
             'tolerance': derived.tolerance, 'ratio': _ratio(error, derived.tolerance), 'flags': flags}
    if errors is not None:
        entry['per_value_error'] = errors
    return entry


def _curve_entry(curve, summary, sweep, order):
    """Entry of an analytic curve from summary curve_error (feature targets in a sweep: per value, error = the worst
    value's max)."""
    per_value = None if sweep is None else (sweep.get('curve_error') or {}).get(curve.name)
    records = per_value if per_value else [(summary.get('curve_error') or {}).get(curve.name)
                                           or {'max': math.nan, 'rms': math.nan, 'n_points': 0,
                                               'note': 'no curve_error in the run summary'}]
    entry = _compared_entry('curve', curve.name, records, curve.tolerance, order if per_value else None)
    entry.update(target=curve.target, series=curve.series)
    if per_value:
        entry['per_value_error'] = [float(record['max']) for record in records]
    return entry


def _derived_curve_entry(curve, summary):
    """Entry of a derived curve with a reference from summary derived_curves (final results only, also in a sweep)."""
    record = (summary.get('derived_curves') or {}).get(curve.name) or {}
    error_record = record.get('curve_error') or {
        'max': math.nan, 'rms': math.nan, 'n_points': 0,
        'note': record.get('error') or 'no derived curve curve_error in the run summary'}
    return _compared_entry('derived_curve', curve.name, [error_record], curve.tolerance, None)


def _compared_entry(kind, name, records, tolerance, order):
    """Entry of curve_error records (one, or one per sweep value): error = the worst max, flags (order: refinement
    order of per-value records, else None)."""
    errors = [float(record['max']) for record in records]
    error = _worst(errors)
    notes = list(dict.fromkeys(record['note'] for record in records if record.get('note')))
    compared = all(int(record['n_points']) > 0 for record in records)
    flags = _flags(error, tolerance, exact_possible=True) if compared else ['not_compared']
    if _not_converging(errors, order):
        flags.append('sweep_not_converging')
    return {'kind': kind, 'name': name, 'error': error, 'rms': _worst([float(record['rms']) for record in records]),
            'n_points': sum(int(record['n_points']) for record in records), 'tolerance': tolerance,
            'ratio': _ratio(error, tolerance), 'flags': flags, 'note': '; '.join(notes)}


def _flags(error, tolerance, exact_possible):
    """over_tolerance / nonfinite / suspicious_exact of one error (exact_possible False: a zero reference, where a
    tiny absolute error is expected)."""
    if not math.isfinite(error):
        return ['nonfinite']
    flags = []
    if tolerance is not None and error > tolerance:
        flags.append('over_tolerance')
    if exact_possible and error < EXACT_ERROR:
        flags.append('suspicious_exact')
    return flags


# ---- mesh-refinement sweeps ----
def _refinement_order(spec, sweep):
    """Sweep value indices from the coarsest to the finest mesh when the swept parameter changes a used mesh's cell
    size (mesh dx, auto mode: 1 / points_per_wavelength; ordered by the first used mesh's), else None."""
    if sweep is None or spec.study.sweep is None or len(sweep.get('values') or []) < 2:
        return None
    sizes = []
    for value in sweep['values']:
        parameters = parameters_with(spec, sweep['parameter'], value)
        try:
            sizes.append(tuple(_cell_size(mesh_spec, parameters) for mesh_spec in spec.used_meshes()))
        except (ExpressionError, ZeroDivisionError, TypeError):
            return None
    if len(set(sizes)) < 2:
        return None
    return sorted(range(len(sizes)), key=lambda index: [-size for size in sizes[index]])


def _cell_size(mesh_spec, parameters):
    """Cell size measure of a MeshSpec over the parameters: dx, auto mode 1 / points_per_wavelength (null: the kind
    default)."""
    if mesh_spec.mode != 'auto':
        return evaluate_scalar(mesh_spec.dx, parameters)
    points = mesh_spec.points_per_wavelength
    if points is None:
        points = MESH_KIND_DEFAULTS.get(mesh_spec.kind, MESH_KIND_DEFAULTS['tensor'])['points_per_wavelength']
    return 1.0 / evaluate_scalar(points, parameters)


def _not_converging(errors, order):
    """True when a mesh-refinement sweep's error at the finest mesh is not below the one at the coarsest (finite
    errors only; never when every error is at round-off level, below EXACT_ERROR)."""
    if order is None:
        return False
    ordered = [errors[index] for index in order if index < len(errors) and math.isfinite(errors[index])]
    return len(ordered) >= 2 and ordered[-1] >= ordered[0] and max(ordered) >= EXACT_ERROR


# ---- numbers ----
def _worst(values):
    """Largest value; NaN when there is none or any is not finite."""
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        return math.nan
    return max(values)


def _ratio(error, tolerance):
    """error / tolerance (tolerance 0: 0 for a zero error, else inf); None without a tolerance."""
    if tolerance is None:
        return None
    if not math.isfinite(error):
        return math.nan
    if tolerance == 0:
        return 0.0 if error == 0.0 else math.inf
    return error / tolerance


def _rank_key(number):
    """Sort key: non-finite (or None) first, then descending; cached records hold 'nan' / 'inf' as strings."""
    number = None if number is None else float(number)
    if number is None or not math.isfinite(number):
        return (0, 0.0)
    return (1, -number)


def _text(number):
    return 'n/a' if number is None else f'{number:.4g}'
