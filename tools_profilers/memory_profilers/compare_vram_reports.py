"""Compare the measured GPU memory of two selftest output folders (e.g. main vs a branch), run by run and part by part;
writes out/vram_compare<tag>.md + .json. Run: python compare_vram_reports.py <base selftest out> <new selftest out>"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # memory_profilers -> tools_profilers -> repository
OUT_DIR = Path(__file__).resolve().parent / 'out'
PART_NAMES = ('kept', 'build_peak', 'run', 'external')   # api._VRAM_PART_NAMES
# no open_sol import at module level: gpu_selftests imports this module after loading another checkout's open_sol


def report_files(out_dir):
    """The selftest reports of a folder, oldest first (a later report's record of the same run wins); *_prev skipped."""
    files = [path for path in Path(out_dir).glob('selftest_report_*.json') if not path.stem.endswith('_prev')]
    if not files:
        raise FileNotFoundError(f'no selftest_report_*.json in {out_dir}')
    return sorted(files, key=lambda path: path.stat().st_mtime)


def run_entries(report):
    """(run key, run record holding 'vram' and 'status') of every case run, variant run and example of one report."""
    for case in report.get('cases') or ():
        for run_name, run in (case.get('runs') or {}).items():
            yield f"{case['name']} / {run_name}", run
        for variant_name, variant in (case.get('variants') or {}).items():
            for run_name, run in (variant.get('runs') or {}).items():
                yield f"{case['name']} [{variant_name}] / {run_name}", run
    for example in report.get('examples') or ():
        yield f"example {example['example']} [{example.get('backend') or report.get('backend')}]", example


def physics_types(run_dir_text, report_dir):
    """Sorted physics types of a run from its run directory's model.json (a relative run dir is the selftest's cwd,
    taken as the report folder's parent)."""
    if not run_dir_text:
        return ['(no run directory)']
    run_dir = Path(run_dir_text)
    model_file = (run_dir if run_dir.is_absolute() else Path(report_dir).parent / run_dir) / 'model.json'
    if not model_file.exists():
        return ['(no model.json)']
    spec = json.loads(model_file.read_text(encoding='utf-8'))
    return sorted({physics['type'] for physics in spec.get('physics') or ()})


def used_solvers(solvers_record):
    """Sorted 'role type (back end)' labels of the solver roles a run used (a counter above 0, or no counters)."""
    labels = set()
    for physics in (solvers_record or {}).values():
        for role in (physics.get('roles') or {}).values():
            counters = role.get('counters') or {}
            numbers = [value for value in counters.values()
                       if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if counters and not any(value != 0 for value in numbers):
                continue   # declared but never solved with (e.g. the linear role of an eigen study)
            back_end = counters.get('back_end') or (role.get('settings') or {}).get('back_end')
            labels.add(role['type'] + (f' ({back_end})' if back_end else ''))
    return sorted(labels) or ['(no solver role)']


def measured_runs(out_dir):
    """{run key: {'status', 'file', 'physics', 'solvers', part: measured bytes or None}} of the runs with a GPU vram
    record."""
    runs = {}
    for path in report_files(out_dir):
        report = json.loads(path.read_text(encoding='utf-8'))
        for key, run in run_entries(report):
            vram = run.get('vram')
            if not vram:   # cpu run, skipped, failed before a build, or unestimated
                continue
            entry = {'status': run.get('status'), 'file': path.name,
                     'physics': physics_types(run.get('out_dir'), out_dir),
                     'solvers': used_solvers((run.get('timing') or {}).get('solvers'))}
            for part in PART_NAMES:
                value = vram.get(part)
                entry[part] = None if value is None else int(value['measured_bytes'])
            runs[key] = entry
    return runs


def peak_part(entry):
    """(part, bytes) of the largest measured part of a run (the selftest record's vram_max_bytes), (None, None) if none."""
    measured = [(part, entry[part]) for part in PART_NAMES if entry[part] is not None]
    return max(measured, key=lambda item: item[1]) if measured else (None, None)


def mb(n):
    return '' if n is None else f'{n / 2 ** 20:.2f}'


def percent(base_bytes, new_bytes):
    if base_bytes is None or new_bytes is None or base_bytes == 0:
        return ''
    return f'{100.0 * (new_bytes - base_bytes) / base_bytes:+.1f}'


def delta(base_bytes, new_bytes):
    return None if base_bytes is None or new_bytes is None else new_bytes - base_bytes


def compare(base_runs, new_runs, floor_bytes):
    """Rows of the runs in both folders: peak (largest part) and per-part changes, ranked by |change|."""
    peak_rows, part_rows = [], []
    for key in sorted(base_runs.keys() & new_runs.keys()):
        base, new = base_runs[key], new_runs[key]
        base_part, base_peak = peak_part(base)
        new_part, new_peak = peak_part(new)
        peak_rows.append({'run': key, 'base_part': base_part, 'base_bytes': base_peak, 'new_part': new_part,
                          'new_bytes': new_peak, 'delta_bytes': delta(base_peak, new_peak),
                          'base_status': base['status'], 'new_status': new['status'],
                          'physics': new['physics'], 'solvers': new['solvers'],
                          'base_physics': base['physics'], 'base_solvers': base['solvers']})
        for part in PART_NAMES:
            change = delta(base[part], new[part])
            appeared = (base[part] is None) != (new[part] is None)
            if appeared or (change is not None and abs(change) > floor_bytes):
                part_rows.append({'run': key, 'part': part, 'base_bytes': base[part], 'new_bytes': new[part],
                                  'delta_bytes': change})
    rank = lambda row: -abs(row['delta_bytes'] or 0)
    return sorted(peak_rows, key=rank), sorted(part_rows, key=rank)


def breakdown(peak_rows, field, floor_bytes):
    """Per label of `field` ('physics' / 'solvers', the new side's): runs, higher / lower beyond the floor, summed
    peaks; a run with several labels counts in each."""
    groups = {}
    for row in peak_rows:
        for label in row[field]:
            group = groups.setdefault(label, {'label': label, 'runs': 0, 'higher': 0, 'lower': 0, 'base_bytes': 0,
                                              'new_bytes': 0, 'max_rise_bytes': 0, 'max_rise_run': ''})
            group['runs'] += 1
            change = row['delta_bytes'] or 0
            group['higher'] += change > floor_bytes
            group['lower'] += change < -floor_bytes
            group['base_bytes'] += row['base_bytes'] or 0
            group['new_bytes'] += row['new_bytes'] or 0
            if change > group['max_rise_bytes']:
                group['max_rise_bytes'], group['max_rise_run'] = change, row['run']
    return sorted(groups.values(), key=lambda group: -abs(group['new_bytes'] - group['base_bytes']))


BREAKDOWN_HEADERS = ('runs', 'higher', 'lower', 'base MB', 'new MB', 'change MB', 'change %', 'largest rise MB')


def breakdown_cells(group):
    """Text cells of a breakdown row after its label, in BREAKDOWN_HEADERS order."""
    return [str(group['runs']), str(group['higher']), str(group['lower']), mb(group['base_bytes']),
            mb(group['new_bytes']), mb(group['new_bytes'] - group['base_bytes']),
            percent(group['base_bytes'], group['new_bytes']), mb(group['max_rise_bytes']) if group['higher'] else '']


def breakdown_lines(title, groups):
    """Markdown table of a breakdown, numbers right-aligned, the run of each largest rise in its own column."""
    lines = ['', f'## By {title} (summed peaks of the runs using it, ranked by |change|)', '',
             f"| {title} | {' | '.join(BREAKDOWN_HEADERS)} | largest rise in |",
             '|---|' + '---:|' * len(BREAKDOWN_HEADERS) + '---|']
    for group in groups:
        rise_run = group['max_rise_run'] if group['higher'] else ''
        lines.append(f"| {group['label']} | {' | '.join(breakdown_cells(group))} | {rise_run} |")
    return lines


def aligned_table(title, groups, limit=None):
    """Console text of a breakdown: label left, numbers right-aligned in padded columns."""
    rows = [[group['label'], *breakdown_cells(group)] for group in groups[:limit]]
    headers = [title, *BREAKDOWN_HEADERS]
    widths = [max(len(row[column]) for row in [headers, *rows]) for column in range(len(headers))]
    def line(cells):
        return '  '.join(cell.ljust(widths[0]) if column == 0 else cell.rjust(widths[column])
                         for column, cell in enumerate(cells))
    return '\n'.join([line(headers), '  '.join('-' * width for width in widths), *map(line, rows)])


def labels_text(row):
    """'physics; solvers' of the new side, with the base side's when they differ."""
    text = f"{', '.join(row['physics'])}; {', '.join(row['solvers'])}"
    if (row['base_physics'], row['base_solvers']) != (row['physics'], row['solvers']):
        text += f" (base: {', '.join(row['base_physics'])}; {', '.join(row['base_solvers'])})"
    return text


def report_text(base_dir, new_dir, base_runs, new_runs, peak_rows, part_rows, floor_bytes):
    beyond = [row for row in peak_rows if row['delta_bytes'] is not None and abs(row['delta_bytes']) > floor_bytes]
    higher = [row for row in beyond if row['delta_bytes'] > 0]
    total_base = sum(row['base_bytes'] or 0 for row in peak_rows)
    total_new = sum(row['new_bytes'] or 0 for row in peak_rows)
    lines = [f'# VRAM compare: {base_dir} (base) vs {new_dir} (new)', '',
             f'- runs with a GPU vram record: base {len(base_runs)}, new {len(new_runs)}, in both {len(peak_rows)}',
             f'- peak (largest measured part) changed by more than {mb(floor_bytes)} MB: {len(beyond)} runs '
             f'({len(higher)} higher, {len(beyond) - len(higher)} lower)',
             f'- summed peaks of the common runs: base {mb(total_base)} MB, new {mb(total_new)} MB '
             f'({percent(total_base, total_new)} %)',
             '- measured parts as in the selftest record: kept = build ledger breakdown, build_peak = pool high water '
             "of the build, run = reserved-based pool high water of the study, external = cuDSS's own plan estimate "
             '(not memGetInfo; vram_profile.py for device bytes)',
             '- solvers: the solver roles a run used (a work counter above 0), labelled type (back end)']
    lines += breakdown_lines('solver', breakdown(peak_rows, 'solvers', floor_bytes))
    lines += breakdown_lines('physics type', breakdown(peak_rows, 'physics', floor_bytes))
    lines += ['', '## Peak per run (largest measured part), ranked by |change|', '',
              '| run | base MB (part) | new MB (part) | change MB | change % | physics; solvers | status base / new |',
              '|---|---|---|---|---|---|---|']
    for row in peak_rows:
        status = '' if row['base_status'] == row['new_status'] else f"{row['base_status']} / {row['new_status']}"
        lines.append(f"| {row['run']} | {mb(row['base_bytes'])} ({row['base_part']}) | {mb(row['new_bytes'])} "
                     f"({row['new_part']}) | {mb(row['delta_bytes'])} | {percent(row['base_bytes'], row['new_bytes'])} "
                     f"| {labels_text(row)} | {status} |")
    lines += ['', f'## Parts changed by more than {mb(floor_bytes)} MB (or present on one side only)', '',
              '| run | part | base MB | new MB | change MB | change % |', '|---|---|---|---|---|---|']
    for row in part_rows:
        lines.append(f"| {row['run']} | {row['part']} | {mb(row['base_bytes'])} | {mb(row['new_bytes'])} | "
                     f"{mb(row['delta_bytes'])} | {percent(row['base_bytes'], row['new_bytes'])} |")
    for title, only in (('base', sorted(base_runs.keys() - new_runs.keys())),
                        ('new', sorted(new_runs.keys() - base_runs.keys()))):
        lines += ['', f'## Only in {title} ({len(only)})', '']
        lines += [f'- {key}: peak {mb(peak_part((base_runs if title == "base" else new_runs)[key])[1])} MB'
                  for key in only] or ['- (none)']
    return '\n'.join(lines) + '\n'


def main():
    sys.path.insert(0, str(ROOT))
    from open_sol.model.builder import ESTIMATE_FLOOR_BYTES
    parser = argparse.ArgumentParser(description='Compare measured GPU memory of two selftest output folders.')
    parser.add_argument('base', help='selftest --out folder of the reference checkout (e.g. main)')
    parser.add_argument('new', help='selftest --out folder of the changed checkout (e.g. the branch)')
    parser.add_argument('--floor-mb', type=float, default=ESTIMATE_FLOOR_BYTES / 2 ** 20,
                        help='part changes up to this are not listed (default: builder.ESTIMATE_FLOOR_BYTES)')
    parser.add_argument('--tag', default='', help='suffix of the report names (out/vram_compare<tag>.md)')
    args = parser.parse_args()
    floor_bytes = int(args.floor_mb * 2 ** 20)
    base_runs, new_runs = measured_runs(args.base), measured_runs(args.new)
    if not base_runs.keys() & new_runs.keys():
        raise ValueError(f'no run with a GPU vram record is in both {args.base} and {args.new}')
    peak_rows, part_rows = compare(base_runs, new_runs, floor_bytes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = report_text(args.base, args.new, base_runs, new_runs, peak_rows, part_rows, floor_bytes)
    (OUT_DIR / f'vram_compare{args.tag}.md').write_text(text, encoding='utf-8')
    (OUT_DIR / f'vram_compare{args.tag}.json').write_text(
        json.dumps({'base': str(args.base), 'new': str(args.new), 'floor_bytes': floor_bytes, 'peaks': peak_rows,
                    'parts': part_rows, 'by_solver': breakdown(peak_rows, 'solvers', floor_bytes),
                    'by_physics': breakdown(peak_rows, 'physics', floor_bytes),
                    'only_base': sorted(base_runs.keys() - new_runs.keys()),
                    'only_new': sorted(new_runs.keys() - base_runs.keys())}, indent=1), encoding='utf-8')
    print('\n'.join(text.splitlines()[2:5]))   # console: the summary counts, then aligned breakdowns
    print(f"\nby solver (summed peaks of the runs using it; a run counts in each of its solvers)\n"
          f"{aligned_table('solver', breakdown(peak_rows, 'solvers', floor_bytes))}")
    print(f"\nby physics type (the 10 largest changes)\n"
          f"{aligned_table('physics type', breakdown(peak_rows, 'physics', floor_bytes), limit=10)}\n")
    print(f"wrote {OUT_DIR / f'vram_compare{args.tag}.md'}")


if __name__ == '__main__':
    main()
