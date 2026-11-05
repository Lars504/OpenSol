"""S7.8 analysis (CPU): errors vs cells / seconds per accuracy case, power-law fits and cells needed for target errors,
PCG iteration thresholds, line_block speed-ups, per-physics totals and the gate verdicts of mesh_study_plan.md
-> tables/*.csv, figures/*.png, mesh_study_report.json.
References: analytic (coax, disc_heater: the model's derived references; disk: exact cylinder resonance), else a
Richardson extrapolation of the three finest uniform runs (rib, strip_heater; the fit order is reported)."""
import csv
import json
import math
import statistics

import numpy as np

from .. import __version__
from . import accuracy, audit, line_block_study, pcg_study, per_physics
from .common import COMPLETED, finite, read_records, record_file

REPORT_FILE = 'mesh_study_report.json'
TARGET_ERRORS = (1e-2, 1e-3, 1e-4)
GATE_TARGET = 1e-3
QUANTITIES = {   # case -> (derived values analysed, reference kind)
    'coax': (('C_per_m', 'W_field'), 'model'),
    'disc_heater': (('theta_centre', 'heater_mean_rise'), 'model'),
    'rib': (('n_eff_1',), 'richardson'),
    'strip_heater': (('dT_core', 'dT_heater'), 'richardson'),
    'disk': (('lambda_pair', 'Q_1', 'pair_split'), 'disk'),
}
DISK_RADIUS, DISK_INDEX = 1e-6, math.sqrt(12.11)
PCG_SWEEP_KEYS = {'contrast': 'contrast', 'ratio': 'max_ratio', 'aspect': 'aspect', 'outer': 'extent',
                  'metric': 'b_over_a'}
JACOBI_STALL_FACTOR = 3.0   # multigrid mandatory where Jacobi needs > this x its baseline iterations
TENSOR_UNIFORM_FAMILIES = ('uniform', 'conformal', 'subpixel')
REFINED_PREFIXES = ('graded_box', 'anisotropic')


# ---- references ----
def richardson(records, name, family='uniform'):
    """(v_inf, p) from the three finest completed runs of family (v = v_inf + C h^p); (finest value, None) when the
    sequence is not monotone or the fit fails."""
    points = sorted((record['h'], record['derived'][name]) for record in records
                    if record['family'] == family and record.get('status') == COMPLETED
                    and finite(record['derived'].get(name)))
    if len(points) < 3:
        return (points[0][1], None) if points else (None, None)
    (h1, v1), (h2, v2), (h3, v3) = points[:3]
    if (v1 - v2) * (v2 - v3) <= 0:
        return v1, None
    from scipy.optimize import brentq
    target = (v1 - v2) / (v2 - v3)

    def mismatch(p):
        return (h1 ** p - h2 ** p) / (h2 ** p - h3 ** p) - target
    try:
        p = brentq(mismatch, 0.2, 8.0)
    except ValueError:
        return v1, None
    return v1 - (v1 - v2) / (h1 ** p - h2 ** p) * h1 ** p, p


def disk_reference(wavelength, q_factor):
    """(lambda_exact, Q_exact, m) of the exact E_z dielectric-cylinder resonance nearest the measured one."""
    from scipy import optimize, special
    k_guess = 2 * math.pi / wavelength * (1 + 0.5j / max(q_factor, 1.0))
    best = None
    for order in range(1, 41):
        def mismatch(k, m=order):
            inside, outside = DISK_INDEX * k * DISK_RADIUS, k * DISK_RADIUS
            return (DISK_INDEX * special.jvp(m, inside) / special.jv(m, inside)
                    - special.h2vp(m, outside) / special.hankel2(m, outside))
        try:
            root = optimize.newton(mismatch, k_guess, tol=1e-14, maxiter=200)
        except (RuntimeError, OverflowError, ZeroDivisionError):
            continue
        if abs(mismatch(root)) > 1e-6 or abs(root - k_guess) > 0.02 * abs(k_guess):
            continue
        if best is None or abs(root - k_guess) < abs(best[0] - k_guess):
            best = (root, order)
    if best is None:
        return None, None, None
    root, order = best
    return 2 * math.pi / root.real, root.real / (2 * abs(root.imag)), order


def add_disk_quantities(records):
    """lambda_pair = mean of rows 1, 2 when they are the staircase-split pair (within 0.2 %), pair_split = their
    relative difference (exact: 0)."""
    for record in records:
        derived = record.get('derived') or {}
        lambda_1, lambda_2 = derived.get('lambda_1'), derived.get('lambda_2')
        if finite(lambda_1) and finite(lambda_2) and abs(lambda_1 - lambda_2) < 2e-3 * lambda_1:
            derived['lambda_pair'] = 0.5 * (lambda_1 + lambda_2)
            derived['pair_split'] = abs(lambda_1 - lambda_2) / lambda_1
        elif finite(lambda_1):
            derived['lambda_pair'] = lambda_1


def errors_of(case, records):
    """{quantity: [(record, relative error)]} and the reference notes."""
    names, kind = QUANTITIES[case]
    notes = []
    if kind == 'disk':
        add_disk_quantities(records)
        finest = min((record for record in records if record.get('status') == COMPLETED
                      and finite(record['derived'].get('lambda_pair'))), key=lambda record: record['h'], default=None)
        if finest is None:
            return {}, ['disk: no completed run']
        lambda_exact, q_exact, order = disk_reference(finest['derived']['lambda_pair'],
                                                      finest['derived'].get('Q_1') or 1e3)
        notes.append(f'disk: exact resonance m = {order}, lambda = {lambda_exact}, Q = {q_exact}')
        references = {'lambda_pair': lambda_exact, 'Q_1': q_exact, 'pair_split': None}
    elif kind == 'richardson':
        references = {}
        for name in names:
            value, order = richardson(records, name)
            references[name] = value
            notes.append(f'{case} {name}: Richardson reference {value} (order p = {order}'
                         + (', not monotone: finest value used)' if order is None else ')'))
    else:
        references = None
    result = {}
    for name in names:
        pairs = []
        for record in records:
            value = (record.get('derived') or {}).get(name)
            if record.get('status') != COMPLETED or not finite(value):
                continue
            if name == 'pair_split':
                pairs.append((record, value))
                continue
            reference = (record.get('derived_reference') or {}).get(name) if references is None else references[name]
            if finite(reference) and reference != 0:
                pairs.append((record, abs(value / reference - 1.0)))
        result[name] = pairs
    return result, notes


# ---- fits ----
def power_fit(pairs, x_key):
    """(c, p) of log err = c - p log x over the points with err > 0, or None."""
    points = [(record[x_key], error) for record, error in pairs if finite(record.get(x_key)) and error > 0]
    if len(points) < 2:
        return None
    slope, intercept = np.polyfit(np.log([x for x, _ in points]), np.log([error for _, error in points]), 1)
    return intercept, -slope


def needed(fit, target):
    if fit is None or fit[1] <= 0:
        return None
    return math.exp((fit[0] - math.log(target)) / fit[1])


def _new_figure(n_axes=1, width=6.0):
    from matplotlib.figure import Figure
    figure = Figure(figsize=(width * n_axes if n_axes > 1 else width, 4.2))
    return figure, [figure.add_subplot(1, n_axes, index + 1) for index in range(n_axes)]


def plot_case(options, case, quantity, pairs):
    figure, axes = _new_figure(2, 5.5)
    for family in sorted({record['family'] for record, _ in pairs}):
        points = sorted(((record['n_cells'], record.get('study_seconds'), error)
                         for record, error in pairs if record['family'] == family and error > 0),
                        key=lambda point: point[0] or 0)
        if not points:
            continue
        cells, seconds, errors = zip(*points)
        axes[0].loglog(cells, errors, 'o-', label=family)
        timed = [(s, e) for s, e in zip(seconds, errors) if finite(s) and s > 0]
        if timed:
            axes[1].loglog(*zip(*timed), 'o-', label=family)
    axes[0].set_xlabel('cells (incl. PML; prototypes: cells / triangles)')
    axes[1].set_xlabel('study seconds (prototypes: CPU wall time)')
    for axis in axes:
        axis.set_ylabel(f'relative error of {quantity}' if quantity != 'pair_split' else 'pair split (exact 0)')
        axis.grid(True, which='both', alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f'{case}: {quantity}')
    figure.tight_layout()
    options.figures_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(options.figures_dir / f'accuracy_{case}_{quantity}.png', dpi=120)


# ---- parts ----
def analyse_accuracy(options):
    """(gain rows, reference notes, problem runs); writes accuracy_table.csv / accuracy_gain.csv and the figures."""
    table_rows, gain_rows, notes, problems = [], [], [], []
    for case in QUANTITIES:
        records = read_records(record_file(options, 'accuracy', case))
        if not records:
            notes.append(f'{case}: no records')
            continue
        problems += _problem_runs(records)
        by_quantity, case_notes = errors_of(case, records)
        notes += case_notes
        for quantity, pairs in by_quantity.items():
            if not pairs:
                continue
            plot_case(options, case, quantity, pairs)
            for record, error in pairs:
                table_rows.append({'case': case, 'quantity': quantity, 'family': record['family'], 'h': record['h'],
                                   'n_cells': record['n_cells'], 'n_unknowns': record.get('n_unknowns'),
                                   'study_seconds': record.get('study_seconds'), 'error': error,
                                   'value': record['derived'][quantity]})
            rows = []
            for family in sorted({record['family'] for record, _ in pairs}):
                family_pairs = [(record, error) for record, error in pairs if record['family'] == family]
                fit, time_fit = power_fit(family_pairs, 'n_cells'), power_fit(family_pairs, 'study_seconds')
                row = {'case': case, 'quantity': quantity, 'family': family, 'n_runs': len(family_pairs),
                       'order_in_cells': None if fit is None else fit[1],
                       'max_cells_measured': max(record['n_cells'] or 0 for record, _ in family_pairs),
                       'min_error_measured': min(error for _, error in family_pairs)}
                for target in TARGET_ERRORS:
                    row[f'cells_at_{target:g}'] = needed(fit, target)
                    row[f'seconds_at_{target:g}'] = needed(time_fit, target)
                rows.append(row)
            uniform = next((row for row in rows if row['family'] == 'uniform'), {})
            for row in rows:
                for target in TARGET_ERRORS:
                    cells, reference = row[f'cells_at_{target:g}'], uniform.get(f'cells_at_{target:g}')
                    row[f'cell_gain_at_{target:g}'] = reference / cells if cells and reference else None
            gain_rows += rows
    write_csv(options.tables_dir / 'accuracy_table.csv', table_rows)
    write_csv(options.tables_dir / 'accuracy_gain.csv', gain_rows)
    return gain_rows, notes, problems


def pcg_counters(record):
    """The pcg counters (cg_iterations, solves, unconverged_solves) of the run's linear role."""
    for physics_record in (record.get('solvers') or {}).values():
        for role in (physics_record.get('roles') or {}).values():
            counters = role.get('counters') or {}
            if 'cg_iterations' in counters:
                return counters
    return {}


def analyse_pcg(options):
    """(rows, thresholds, problem runs); writes pcg_table.csv and pcg_<family>.png."""
    rows, problems = [], []
    for case in pcg_study.CASES:
        records = read_records(record_file(options, 'pcg', case))
        problems += _problem_runs(records)
        for record in records:
            counters = pcg_counters(record)
            iterations, solves = counters.get('cg_iterations'), counters.get('solves') or 1
            longest = max(record.get('nx') or 0, record.get('ny') or 0)
            sweep = record.get('sweep') or {}
            rows.append({'model': case, 'family': record['family'], 'preconditioner': sweep.get('preconditioner'),
                         'x': sweep.get(PCG_SWEEP_KEYS.get(record['family'], '')), **sweep,
                         'status': record.get('status'), 'n_cells': record.get('n_cells'), 'nx': record.get('nx'),
                         'ny': record.get('ny'), 'dx_min': record.get('dx_min'), 'dx_max': record.get('dx_max'),
                         'iterations_per_solve': None if iterations is None else iterations / solves,
                         'per_longest_axis_cell': (None if iterations is None or not longest
                                                   else iterations / solves / longest),
                         'unconverged': counters.get('unconverged_solves'),
                         'study_seconds': record.get('study_seconds')})
    if not rows:
        return [], [], problems
    write_csv(options.tables_dir / 'pcg_table.csv', rows)
    thresholds = []
    for model in pcg_study.CASES:
        for family in PCG_SWEEP_KEYS:
            series = {preconditioner: sorted((row for row in rows if row['model'] == model
                                              and row['family'] == family and row['preconditioner'] == preconditioner
                                              and finite(row['x'])), key=lambda row: row['x'])
                      for preconditioner in pcg_study.PRECONDITIONERS}
            jacobi = [row for row in series['jacobi'] if row['iterations_per_solve']]
            if not jacobi:
                continue
            baseline = jacobi[0]['iterations_per_solve']
            stall = next((row['x'] for row in jacobi if row['iterations_per_solve'] > JACOBI_STALL_FACTOR * baseline
                          or (row['unconverged'] or 0) > 0), None)
            multigrid = {row['x']: row for row in series['multigrid']}
            last = jacobi[-1]
            partner = multigrid.get(last['x'])
            speedup = (last['study_seconds'] / partner['study_seconds']
                       if partner and finite(partner.get('study_seconds')) and partner['study_seconds'] > 0
                       and finite(last.get('study_seconds')) else None)
            thresholds.append({'model': model, 'family': family, 'sweep_key': PCG_SWEEP_KEYS[family],
                               'baseline_iterations': baseline, 'jacobi_stall_from': stall,
                               'largest_x': last['x'], 'jacobi_iterations_at_largest': last['iterations_per_solve'],
                               'multigrid_iterations_at_largest': None if partner is None
                               else partner['iterations_per_solve'],
                               'multigrid_speedup_at_largest': speedup})
    write_csv(options.tables_dir / 'pcg_thresholds.csv', thresholds)
    for family, key in PCG_SWEEP_KEYS.items():
        figure, (axis,) = _new_figure()
        for model in pcg_study.CASES:
            for preconditioner in pcg_study.PRECONDITIONERS:
                points = sorted((row['x'], row['iterations_per_solve']) for row in rows
                                if row['model'] == model and row['family'] == family
                                and row['preconditioner'] == preconditioner and finite(row['x'])
                                and row['iterations_per_solve'])
                if points:
                    axis.loglog(*zip(*points), 'o-' if preconditioner == 'jacobi' else 's--',
                                label=f'{model} {preconditioner}')
        if axis.lines:
            axis.set_xlabel(key)
            axis.set_ylabel('CG iterations per solve (tolerance 1e-10)')
            axis.grid(True, which='both', alpha=0.3)
            axis.legend(fontsize=8)
            figure.tight_layout()
            options.figures_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(options.figures_dir / f'pcg_{family}.png', dpi=120)
    return rows, thresholds, problems


def analyse_line_block(options):
    """(rows: per mesh family and h the sparse_direct / line_block seconds and their ratio, problem runs)."""
    records = read_records(record_file(options, line_block_study.CASE))
    by_key = {}
    for record in records:
        key = (record['family'], record['h'])
        by_key.setdefault(key, {})[(record.get('sweep') or {}).get('solver')] = record
    rows = []
    for (family, h), solvers in sorted(by_key.items()):
        direct, blocks = solvers.get('sparse_direct'), solvers.get('line_block')
        seconds = {name: record.get('study_seconds') for name, record in solvers.items()
                   if record.get('status') == COMPLETED}
        speedup = (seconds['sparse_direct'] / seconds['line_block']
                   if finite(seconds.get('sparse_direct')) and finite(seconds.get('line_block'))
                   and seconds['line_block'] > 0 else None)
        any_record = direct or blocks
        rows.append({'family': family, 'h': h, 'n_cells': any_record.get('n_cells'), 'nx': any_record.get('nx'),
                     'ny': any_record.get('ny'), 'line_block_hint': any_record.get('line_block_hint'),
                     'sparse_direct_seconds': seconds.get('sparse_direct'),
                     'line_block_seconds': seconds.get('line_block'), 'speedup': speedup,
                     'phase_axis_difference': _difference(direct, blocks, 'phase_axis')})
    write_csv(options.tables_dir / 'line_block_table.csv', rows)
    return rows, _problem_runs(records)


def _difference(first, second, name):
    values = [(record or {}).get('derived', {}).get(name) for record in (first, second)]
    return abs(values[0] - values[1]) if all(finite(value) for value in values) else None


def per_physics_totals(options):
    """[{model, shared_gpu, own_gpu, ratio}] of the TOTAL rows of per_physics.csv."""
    path = options.tables_dir / per_physics.TABLE_FILE
    if not path.exists():
        return []
    totals = []
    with open(path, encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            if row['physics'] != 'TOTAL':
                continue
            shared, own = _float(row['gpu_on_shared_mesh']), _float(row['gpu_on_own_mesh'])
            totals.append({'model': row['model'], 'shared_gpu': shared, 'own_gpu': own,
                           'ratio': own / shared if shared and own else None})
    return totals


def audit_summary(options):
    path = options.records_dir / audit.SUMMARY_FILE
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as handle:
        areas = json.load(handle)['areas']
    physics_hits = sum(value['hits'] for area, value in areas.items() if area.startswith('physics:'))
    total_hits = sum(value['hits'] for value in areas.values())
    return {'physics_hits': physics_hits, 'total_hits': total_hits,
            'physics_share': physics_hits / total_hits if total_hits else None, 'areas': areas}


# ---- gates ----
def _gain_row(gain_rows, case, quantity, family):
    return next((row for row in gain_rows if row['case'] == case and row['quantity'] == quantity
                 and row['family'] == family), None)


def _cells(gain_rows, case, quantity, family):
    row = _gain_row(gain_rows, case, quantity, family)
    return None if row is None else row.get(f'cells_at_{GATE_TARGET:g}')


def _verdict(value, go, marginal):
    if value is None:
        return 'not measured'
    return 'go' if value >= go else 'marginal' if value >= marginal else 'no-go'


def gates(gain_rows, thresholds, line_rows, totals, audit_data):
    """Gate records {gate, measured, rule, verdict} of mesh_study_plan.md."""
    records = []
    ratios = [total['ratio'] for total in totals if total['ratio'] is not None]
    median = statistics.median(ratios) if ratios else None
    records.append({'gate': 'S7.2 per-physics meshes', 'measured': {'median_own_over_shared_gpu': median,
                                                                    'n_models': len(ratios)},
                    'rule': 'go when the median own / shared GPU estimate <= 0.5 (marginal <= 0.8)',
                    'verdict': _verdict(None if median is None else 1.0 / median if median > 0 else math.inf,
                                        2.0, 1.25)})
    tensor_cells = [cells for cells in (_cells(gain_rows, 'coax', 'C_per_m', family)
                                        for family in TENSOR_UNIFORM_FAMILIES) if cells]
    best_tensor = min(tensor_cells) if tensor_cells else None
    quadtree = _cells(gain_rows, 'coax', 'C_per_m', 'quadtree')
    local_gain = best_tensor / quadtree if best_tensor and quadtree else None
    share = None if audit_data is None else audit_data['physics_share']
    order = ('S7.3 first (blocks keep every physics kernel)' if share is None or share >= 0.3
             else 'S7.4 directly (most grid assumptions sit in shared code both options change)')
    records.append({'gate': 'S7.3 / S7.4 local refinement',
                    'measured': {'coax_best_tensor_over_quadtree_cells': local_gain, 'audit_physics_share': share,
                                 'order_if_go': order},
                    'rule': f'go when best tensor (uniform, conformal) / quadtree cells at {GATE_TARGET:g} >= 3 '
                            '(marginal >= 1.5)',
                    'verdict': _verdict(local_gain, 3.0, 1.5)})
    fem = _cells(gain_rows, 'coax', 'C_per_m', 'fem_delaunay')
    fem_gain = best_tensor / fem if best_tensor and fem else None
    rib_orders = [2.0 * row['order_in_cells'] for row in gain_rows
                  if row['case'] == 'rib' and row['order_in_cells'] is not None]
    rib_order = max(rib_orders) if rib_orders else None
    if fem_gain is None or rib_order is None:
        fem_verdict = 'not measured'
    else:
        n_true = int(fem_gain >= 3.0) + int(rib_order < 1.5)
        fem_verdict = ('no-go', 'marginal', 'go')[n_true]
    records.append({'gate': 'S7.5 FEM triangles',
                    'measured': {'coax_best_tensor_over_fem_delaunay_cells': fem_gain,
                                 'rib_best_order_in_h': rib_order},
                    'rule': 'go when both: tensor / FEM cells >= 3 and the best rib order in h < 1.5 (one: marginal); '
                            'the mesher licence stays a desk item',
                    'verdict': fem_verdict})
    case_gains = {}
    for row in gain_rows:
        gain = row.get(f'cell_gain_at_{GATE_TARGET:g}')
        if row['family'].startswith(REFINED_PREFIXES) and gain is not None:
            case_gains[row['case']] = max(case_gains.get(row['case'], 0.0), gain)
    n_cases = sum(gain >= 2.0 for gain in case_gains.values())
    records.append({'gate': 'S7.7 adaptive refinement', 'measured': {'best_refined_gain_per_case': case_gains,
                                                                     'cases_with_gain_ge_2': n_cases},
                    'rule': 'go when >= 2 cases gain >= 2x cells from refinement (1: marginal); the convergence '
                            'report half of S7.7 is go regardless',
                    'verdict': 'not measured' if not case_gains else _verdict(n_cases, 2, 1)})
    stalls = {f"{row['model']} {row['family']}": row['jacobi_stall_from'] for row in thresholds}
    records.append({'gate': 'S7.0 multigrid (informational)', 'measured': stalls,
                    'rule': f'Jacobi > {JACOBI_STALL_FACTOR:g}x its baseline iterations or unconverged from this '
                            'sweep value on: multigrid mandatory there',
                    'verdict': 'not measured' if not thresholds else 'informational'})
    records.append({'gate': 'line_block (informational)',
                    'measured': {f"{row['family']} h {row['h']:.3g}": row['speedup'] for row in line_rows},
                    'rule': 'sparse_direct / line_block study seconds per mesh: checks the refine-the-line-axis rule',
                    'verdict': 'not measured' if not line_rows else 'informational'})
    return records


# ---- driver ----
def run(options):
    gain_rows, notes, problems = analyse_accuracy(options)
    pcg_rows, thresholds, pcg_problems = analyse_pcg(options)
    line_rows, line_problems = analyse_line_block(options)
    totals = per_physics_totals(options)
    audit_data = audit_summary(options)
    gate_records = gates(gain_rows, thresholds, line_rows, totals, audit_data)
    report = {'open_sol_version': __version__, 'target_errors': list(TARGET_ERRORS), 'gate_target': GATE_TARGET,
              'gates': gate_records, 'reference_notes': notes, 'accuracy_gain': gain_rows,
              'pcg_thresholds': thresholds, 'line_block': line_rows, 'per_physics_totals': totals,
              'audit': audit_data, 'problem_runs': problems + pcg_problems + line_problems,
              'n_pcg_runs': len(pcg_rows), 'accuracy_cases': list(accuracy.CASES)}
    options.out_dir.mkdir(parents=True, exist_ok=True)
    with open(options.out_dir / REPORT_FILE, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, default=str)
    for note in notes:
        print(note)
    for row in gain_rows:
        gains = ', '.join(f"{target:g}: {_fmt(row.get(f'cells_at_{target:g}'))} cells "
                          f"(x{_fmt(row.get(f'cell_gain_at_{target:g}'))})" for target in TARGET_ERRORS)
        print(f"{row['case']:<13} {row['quantity']:<17} {row['family']:<21} order {_fmt(row['order_in_cells'])}  {gains}")
    for row in thresholds:
        print(f"pcg {row['model']:<14} {row['family']:<9} baseline {_fmt(row['baseline_iterations'])} it, Jacobi stalls "
              f"from {row['sweep_key']} = {_fmt(row['jacobi_stall_from'])}, multigrid x{_fmt(row['multigrid_speedup_at_largest'])} "
              f"faster at {_fmt(row['largest_x'])}")
    for problem in report['problem_runs']:
        print(f'problem run: {problem}')
    print('\nGates:')
    for gate in gate_records:
        print(f"  {gate['gate']:<34} {gate['verdict']:<14} {json.dumps(gate['measured'], default=str)}")
    print(f'written {options.out_dir / REPORT_FILE}')


def _problem_runs(records):
    """Text of each run that did not complete, fell back to a uniform mesh or warned at check time."""
    problems = []
    for record in records:
        if record.get('status') != COMPLETED:
            problems.append(f"{record.get('case')} {record['label']}: {record.get('status')} "
                            f"{record.get('message') or record.get('check_issues') or ''}")
        elif record.get('mesh_note'):
            problems.append(f"{record.get('case')} {record['label']}: mesh note {record['mesh_note']}")
    return problems


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'written {path}')


def _float(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _fmt(value):
    return '-' if value is None else f'{value:.3g}' if isinstance(value, (int, float)) else str(value)
