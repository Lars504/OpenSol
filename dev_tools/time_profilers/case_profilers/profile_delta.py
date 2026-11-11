"""Per-part changes between two profiles of one command: reads two runs.json (default: the two newest
out/history/<timestamp>/<folder>.runs.json copies run_all_profilers.py keeps) and prints, per run, the parts whose
exclusive time changed by more than --min-ms and --min-percent, largest first.
Run (from any folder): python profile_delta.py solid_mechanics_3d_nonlinear__gpu [--before PATH] [--after PATH]
[--min-ms 20] [--min-percent 30]"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HISTORY = HERE / 'out' / 'history'
DEFAULT_MIN_MS = 20.0        # smaller changes are below what one run resolves
DEFAULT_MIN_PERCENT = 30.0   # run-to-run noise of a part on a busy machine


def history_runs(folder) -> list:
    """Paths of the history copies '<timestamp>/<folder>.runs.json', oldest first."""
    return sorted(HISTORY.glob(f'*/{folder}.runs.json'), key=lambda path: path.parent.name)


def _exclusive_ms(run) -> dict:
    """{part label: exclusive ms} of one run record of runs.json."""
    return {label: 1e3 * part['exclusive_seconds'] for label, part in (run.get('parts') or {}).items()}


def _calls(run, label) -> int:
    return int(((run.get('parts') or {}).get(label) or {}).get('calls', 0))


def delta_text(before, after, min_ms=DEFAULT_MIN_MS, min_percent=DEFAULT_MIN_PERCENT) -> str:
    """Per run of `after` (runs.json dicts): its wall before -> after and the parts whose exclusive ms changed by at
    least min_ms and min_percent of the larger value; new and dropped runs are named."""
    lines = []
    for name, new in after.items():
        if name not in before:
            lines.append(f"== {name}: new run, wall {new['wall_seconds']:.2f} s")
            continue
        old = before[name]
        lines.append(f"== {name}: wall {old['wall_seconds']:.2f} -> {new['wall_seconds']:.2f} s")
        old_ms, new_ms = _exclusive_ms(old), _exclusive_ms(new)
        rows = []
        for label in set(old_ms) | set(new_ms):
            first, second = old_ms.get(label, 0.0), new_ms.get(label, 0.0)
            change = second - first
            if abs(change) >= min_ms and abs(change) >= 0.01 * min_percent * max(first, second):
                rows.append((change, label, first, second))
        for change, label, first, second in sorted(rows, key=lambda row: (-abs(row[0]), row[1])):
            lines.append(f'  {label:<62} {first:9.1f} -> {second:9.1f} ms ({change:+9.1f}), '
                         f'calls {_calls(old, label)} -> {_calls(new, label)}')
        if not rows:
            lines.append(f'  no part changed by {min_ms:g} ms and {min_percent:g} %')
    lines += [f'== {name}: run dropped' for name in before if name not in after]
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', help="report folder name, e.g. 'solid_mechanics_3d_nonlinear__gpu'")
    parser.add_argument('--before', type=Path, help='runs.json of the earlier profile (default: second newest copy)')
    parser.add_argument('--after', type=Path, help='runs.json of the later profile (default: newest copy)')
    parser.add_argument('--min-ms', type=float, default=DEFAULT_MIN_MS)
    parser.add_argument('--min-percent', type=float, default=DEFAULT_MIN_PERCENT)
    options = parser.parse_args()
    kept = history_runs(options.folder)
    after = options.after or (kept[-1] if kept else None)
    before = options.before or next((path for path in reversed(kept) if path != after), None)
    if before is None or after is None:
        raise SystemExit(f'{options.folder}: needs two runs.json (history holds {len(kept)}; run_all_profilers.py '
                         'keeps one per run from now on, or pass --before / --after)')
    print(f'before: {before}\nafter:  {after}')
    print(delta_text(json.loads(Path(before).read_text(encoding='utf-8')),
                     json.loads(Path(after).read_text(encoding='utf-8')), options.min_ms, options.min_percent))


if __name__ == '__main__':
    main()
