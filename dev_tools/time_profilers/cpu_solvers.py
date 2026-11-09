"""CPU solver profiler (dev tooling, not part of open_sol): wall time per phase, peak RSS and accuracy of every CPU
solver choice open_sol offers, on representative problems assembled by open_sol's own code, one child process per
(problem, size, solver); JSON records + markdown tables. Run from the repository root:
python -m dev_tools.time_profilers.cpu_solvers [--quick] [--problems ...] [--solvers ...] [--threads N]."""
# The parent enumerates (solver_catalog: registry + solver_misfit), never factorises; each case runs in a fresh
# `python -m dev_tools.time_profilers.case_runner` (clean peak RSS, a hard per-case timeout, a crash or the OOM killer
# costs one case), with OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS set from --threads before the child
# starts (thread pools read them at library load). The baseline (host SuperLU) of every problem always runs: the
# speed-ups and eigenvalue differences are against it.
import argparse
import dataclasses
import datetime
import fnmatch
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:   # run as a script (python dev_tools/time_profilers/cpu_solvers.py)
    sys.path.insert(0, REPO_ROOT)

from dev_tools.time_profilers import report   # noqa: E402
from dev_tools.time_profilers.case_runner import RESULT_MARKER, THREAD_VARIABLES   # noqa: E402
from dev_tools.time_profilers.problems import PROBLEMS   # noqa: E402
from dev_tools.time_profilers.solver_catalog import cpu_candidates   # noqa: E402

CHILD_MODULE = 'dev_tools.time_profilers.case_runner'
DEFAULT_REPEAT = 3
DEFAULT_TIMEOUT = 600.0        # seconds per case (all its repeats)
QUICK_TIMEOUT = 120.0
DEFAULT_OUT = 'cpu_solver_profiles'   # relative to the current directory
STDERR_TAIL_CHARS = 2000


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='python -m dev_tools.time_profilers.cpu_solvers',
        description='Compare every CPU solver open_sol offers (sparse direct SuperLU / MUMPS / PARDISO incl. mixed '
                    'precision, pcg, banded, line_block, shift-invert eigen factorizations) on problems assembled by '
                    'open_sol: wall time per phase, peak RSS, accuracy; one child process per case.')
    parser.add_argument('--problems', nargs='+', choices=sorted(PROBLEMS), metavar='PROBLEM',
                        help=f"problem classes (default: all): {', '.join(PROBLEMS)}")
    parser.add_argument('--sizes', nargs='+', metavar='SIZES',
                        help="size knobs: '64,128' for every selected problem, or 'heat_3d=16,24' per problem "
                             '(default: each problem\'s sizes, --quick its quick sizes)')
    parser.add_argument('--solvers', nargs='+', metavar='PATTERN',
                        help="fnmatch patterns of solver labels, e.g. 'sparse_direct:*' 'pcg:double/*' '*mumps*' "
                             '(default: all; the SuperLU baseline always runs)')
    parser.add_argument('--repeat', type=int, help=f'repeats per case (default {DEFAULT_REPEAT}, --quick 1)')
    parser.add_argument('--threads', type=int, help='set OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS in '
                                                    'every child (default: leave the environment as it is)')
    parser.add_argument('--timeout', type=float, help=f'seconds per case (default {DEFAULT_TIMEOUT:g}, --quick '
                                                      f'{QUICK_TIMEOUT:g})')
    parser.add_argument('--quick', action='store_true', help='small sizes, one repeat, shorter timeout')
    parser.add_argument('--out', default=DEFAULT_OUT, help=f'output folder for the JSON / markdown (default '
                                                           f'./{DEFAULT_OUT})')
    parser.add_argument('--list', action='store_true', help='list problems and solver choices, run nothing')
    args = parser.parse_args(argv)
    if args.repeat is not None and args.repeat < 1:
        parser.error('--repeat must be >= 1')
    if args.threads is not None and args.threads < 1:
        parser.error('--threads must be >= 1')
    return args, parser


def _sizes(args, parser, names) -> dict:
    """{problem: sizes} from --sizes (global 'a,b' items and 'problem=a,b' items), else the defaults."""
    chosen = {name: (PROBLEMS[name].quick_sizes if args.quick else PROBLEMS[name].sizes) for name in names}
    if not args.sizes:
        return chosen
    common, specific = [], {}
    try:
        for item in args.sizes:
            if '=' in item:
                name, values = item.split('=', 1)
                if name not in PROBLEMS:
                    parser.error(f"--sizes: unknown problem '{name}'")
                specific[name] = tuple(int(value) for value in values.split(',') if value)
            else:
                common.extend(int(value) for value in item.split(',') if value)
    except ValueError:
        parser.error(f'--sizes: sizes are integers, got {args.sizes}')
    for name in names:
        if name in specific:
            chosen[name] = specific[name]
        elif common:
            chosen[name] = tuple(common)
    return chosen


def _selected(candidates, patterns):
    """Candidates matching any pattern (all without patterns), the baseline always."""
    if not patterns:
        return list(candidates)
    return [candidate for candidate in candidates
            if candidate.baseline or any(fnmatch.fnmatchcase(candidate.label, pattern) for pattern in patterns)]


def _child_environment(threads):
    environment = dict(os.environ)
    if threads is not None:
        for name in THREAD_VARIABLES:
            environment[name] = str(threads)
    return environment


def _tail(text) -> str:
    text = (text or '').strip()
    return text[-STDERR_TAIL_CHARS:] if text else ''


def run_child(case, timeout, environment) -> dict:
    """Result of one case in a fresh child (case_runner): its result line, or status timeout / killed (negative
    return code: a signal, SIGKILL usually the out-of-memory killer) / crashed (no result line)."""
    started = time.perf_counter()
    try:
        completed = subprocess.run([sys.executable, '-m', CHILD_MODULE], input=json.dumps(case), capture_output=True,
                                   text=True, timeout=timeout, cwd=REPO_ROOT, env=environment)
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'reason': f'no result within {timeout:g} s (child killed)', 'repeats': [],
                'child_seconds': time.perf_counter() - started}
    result_lines = [line for line in completed.stdout.splitlines() if line.startswith(RESULT_MARKER)]
    if result_lines:
        result = json.loads(result_lines[-1][len(RESULT_MARKER):])
    elif completed.returncode < 0:
        result = {'status': 'killed', 'reason': f'child ended by signal {-completed.returncode} (9: SIGKILL, usually '
                                                'the out-of-memory killer)', 'repeats': []}
    else:
        result = {'status': 'crashed', 'reason': _tail(completed.stderr) or f'exit code {completed.returncode}',
                  'repeats': []}
    result['child_seconds'] = time.perf_counter() - started
    result['returncode'] = completed.returncode
    if completed.returncode != 0 or result.get('status') != 'ok':
        result['stderr_tail'] = _tail(completed.stderr)
    return result


def _print_list(names, sizes, catalog) -> None:
    for name in names:
        problem_class = PROBLEMS[name]
        candidates, excluded = catalog[name]
        print(f'{name}: {problem_class.description}')
        print(f'  {problem_class.physics_type} / {problem_class.role} / {problem_class.study} / grid '
              f'{problem_class.grid}; sizes {sizes[name]} ({problem_class.size_text})')
        for candidate in candidates:
            marks = ''.join((' [baseline]' if candidate.baseline else '', ' [auto]' if candidate.automatic else ''))
            state = 'available' if candidate.available else f'unavailable: {candidate.reason}'
            print(f'    {candidate.label}{marks}: {state}')
        for entry in excluded:
            print(f"    (not offered) {entry['label']}: {entry['reason']}")


def _record(problem_name, size, candidate, result) -> dict:
    record = {'problem': problem_name, 'size': size, 'solver': candidate.label, 'solver_type': candidate.solver_type,
              'values': candidate.values, 'automatic': candidate.automatic, 'baseline': candidate.baseline}
    record.update(result)
    record['summary'] = report.summarise(record)
    return record


def _add_eigen_differences(records) -> None:
    """eigen_difference of every eigen record against the baseline record of its problem and size."""
    baselines = {(record['problem'], record['size']): record for record in records if record['baseline']}
    for record in records:
        reference = baselines.get((record['problem'], record['size']))
        if PROBLEMS[record['problem']].role == 'eigen' and reference is not None and record.get('repeats'):
            record['eigen_difference'] = report.eigen_difference(record, reference)


def main(argv=None) -> int:
    args, parser = _parse_args(argv)
    names = args.problems or list(PROBLEMS)
    sizes = _sizes(args, parser, names)
    repeat = args.repeat or (1 if args.quick else DEFAULT_REPEAT)
    timeout = args.timeout or (QUICK_TIMEOUT if args.quick else DEFAULT_TIMEOUT)
    catalog = {name: cpu_candidates(PROBLEMS[name]) for name in names}
    if args.list:
        _print_list(names, sizes, catalog)
        return 0
    selected = {name: _selected(catalog[name][0], args.solvers) for name in names}
    environment = _child_environment(args.threads)
    machine = report.machine_info(args.threads)
    records = []
    for name in names:
        for size in sizes[name]:
            for candidate in selected[name]:
                prefix = f'{name} n={size} {candidate.label}'
                if not candidate.available:
                    records.append(_record(name, size, candidate, {'status': 'skipped', 'reason': candidate.reason,
                                                                   'repeats': []}))
                    print(f'{prefix}: skipped ({candidate.reason})', flush=True)
                    continue
                case = {'problem': name, 'size': size, 'solver_type': candidate.solver_type,
                        'values': candidate.values, 'repeat': repeat}
                record = _record(name, size, candidate, run_child(case, timeout, environment))
                records.append(record)
                total = record['summary'].get('total', {}).get('median')
                timing = '' if total is None else f' {total:.3g} s'
                print(f"{prefix}:{timing} {record['status']}"
                      + (f" ({record.get('reason')})" if record.get('reason') and record['status'] != 'ok' else ''),
                      flush=True)
    _add_eigen_differences(records)
    problem_classes = {name: PROBLEMS[name] for name in names}
    text = report.markdown(problem_classes, selected, records, sizes, machine)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, f'cpu_solvers_{stamp}.json')
    markdown_path = os.path.join(args.out, f'cpu_solvers_{stamp}.md')
    document = {'tool': 'dev_tools.time_profilers.cpu_solvers', 'created': datetime.datetime.now().isoformat(),
                'machine': machine,
                'settings': {'problems': names, 'sizes': sizes, 'solvers': args.solvers, 'repeat': repeat,
                             'threads': args.threads, 'timeout': timeout, 'quick': args.quick},
                'problems': {name: {'description': PROBLEMS[name].description,
                                    'physics_type': PROBLEMS[name].physics_type, 'role': PROBLEMS[name].role,
                                    'study': PROBLEMS[name].study, 'grid': PROBLEMS[name].grid,
                                    'size_text': PROBLEMS[name].size_text} for name in names},
                'solvers': {name: [dataclasses.asdict(candidate) for candidate in catalog[name][0]] for name in names},
                'not_offered': {name: catalog[name][1] for name in names},
                'records': records}
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(document, handle, indent=1, default=str)
    with open(markdown_path, 'w', encoding='utf-8') as handle:
        handle.write(text)
    print()
    print(text)
    print(f'records: {json_path}\ntables: {markdown_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
