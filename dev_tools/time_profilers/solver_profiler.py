"""Solver profiler (dev tooling, not part of open_sol): wall time per phase, peak RSS and accuracy of every solver
choice open_sol offers on the cpu (and with --backends, the gpu) backend, on representative problems assembled by
open_sol's own code and, with --models, on whole model files run by api.run; one child process per case; JSON records
+ markdown tables. Run from the repository root:
python -m dev_tools.time_profilers.solver_profiler [--quick] [--problems ...] [--backends cpu gpu] [--models FILE ...]."""
# The parent enumerates (solver_catalog: registry + solver_misfit per backend), never factorises and never makes a
# CUDA context; each case runs in a fresh `python -m dev_tools.time_profilers.case_runner` (clean peak RSS, a hard
# per-case timeout, a crash or the OOM killer costs one case), with OMP_NUM_THREADS / MKL_NUM_THREADS /
# OPENBLAS_NUM_THREADS set from --threads before the child starts (thread pools read them at library load). The
# baseline (host SuperLU on the cpu backend, else the cpu Automatic) of every problem always runs: the speed-ups and
# eigenvalue / derived-value differences are against it, whichever backends are asked for.
import argparse
import dataclasses
import datetime
import fnmatch
import json
import os
import re
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:   # run as a script (python dev_tools/time_profilers/solver_profiler.py)
    sys.path.insert(0, REPO_ROOT)

from dev_tools.time_profilers import report   # noqa: E402
from dev_tools.time_profilers.case_runner import PARTIAL_MARKER, RESULT_MARKER, THREAD_VARIABLES   # noqa: E402
from dev_tools.time_profilers.problems import PROBLEMS   # noqa: E402
from dev_tools.time_profilers.solver_catalog import BACKENDS, solver_candidates   # noqa: E402

CHILD_MODULE = 'dev_tools.time_profilers.case_runner'
DEFAULT_REPEAT = 3
DEFAULT_TIMEOUT = 600.0        # seconds per case (all its repeats)
QUICK_TIMEOUT = 120.0
DEFAULT_OUT = 'solver_profiles_out'   # relative to the current directory; *_out/ is git-ignored
STDERR_TAIL_CHARS = 2000


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        prog='python -m dev_tools.time_profilers.solver_profiler',
        description='Compare every solver open_sol offers (sparse direct SuperLU / MUMPS / PARDISO / cuDSS incl. mixed '
                    'precision, pcg, banded, line_block, shift-invert eigen factorizations) on problems assembled by '
                    'open_sol and on whole model files: wall time per phase, peak RSS, accuracy; one child process per '
                    'case.')
    parser.add_argument('--problems', nargs='+', choices=sorted(PROBLEMS), metavar='PROBLEM',
                        help=f"synthetic problem classes (default: all, none with --models): {', '.join(PROBLEMS)}")
    parser.add_argument('--sizes', nargs='+', metavar='SIZES',
                        help="size knobs: '64,128' for every selected problem, or 'heat_3d=16,24' per problem "
                             '(default: each problem\'s sizes, --quick its quick sizes)')
    parser.add_argument('--backends', nargs='+', choices=BACKENDS, default=['cpu'],
                        help="backends whose solver choices run (default cpu; 'gpu' adds cuDSS, gpu pcg / line_block / "
                             'banded and the gpu runs of the host libraries; the cpu baseline always runs)')
    parser.add_argument('--models', nargs='+', metavar='FILE',
                        help='model JSON files run whole by api.run (validate, mesh, build, study, output) with each '
                             'solver choice assigned to one physics role (default: every physics and role its study '
                             'uses)')
    parser.add_argument('--physics', nargs='+', metavar='NAME', help='--models: only these physics names')
    parser.add_argument('--roles', nargs='+', metavar='ROLE', help='--models: only these solver roles (linear, eigen, '
                                                                   'shifted, stepper)')
    parser.add_argument('--scan', metavar='NAME=V1,V2', help='--models: one column per value of this model parameter '
                                                             '(e.g. a mesh resolution), default the model as written')
    parser.add_argument('--set', action='append', default=[], metavar='NAME=EXPR',
                        help='--models: parameter override applied to every run')
    parser.add_argument('--solvers', nargs='+', metavar='PATTERN',
                        help="fnmatch patterns of solver labels, e.g. 'sparse_direct:*' 'pcg:double/*' '*cudss*' "
                             '(default: all; the baseline always runs)')
    parser.add_argument('--repeat', type=int, help=f'warm repeats per case at least (default {DEFAULT_REPEAT}, '
                                                   '--quick 1; tiny cases repeat until 0.5 s of warm time)')
    parser.add_argument('--no-warmup', action='store_true', help='skip the cold repeat (library loading, first '
                                                                 'touch, kernel compiles) that runs first')
    parser.add_argument('--threads', type=int, help='set OMP_NUM_THREADS / MKL_NUM_THREADS / OPENBLAS_NUM_THREADS in '
                                                    'every child (default: leave the environment as it is)')
    parser.add_argument('--timeout', type=float, help=f'seconds per case (default {DEFAULT_TIMEOUT:g}, --quick '
                                                      f'{QUICK_TIMEOUT:g})')
    parser.add_argument('--quick', action='store_true', help='small sizes, one repeat, shorter timeout')
    parser.add_argument('--out', default=DEFAULT_OUT, help=f'output folder for the JSON / markdown (and the model '
                                                           f'run folders; default ./{DEFAULT_OUT})')
    parser.add_argument('--list', action='store_true', help='list problems and solver choices, run nothing')
    args = parser.parse_args(argv)
    if args.repeat is not None and args.repeat < 1:
        parser.error('--repeat must be >= 1')
    if args.threads is not None and args.threads < 1:
        parser.error('--threads must be >= 1')
    if not args.models and (args.physics or args.roles or args.scan or args.set):
        parser.error('--physics / --roles / --scan / --set need --models')
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


def _assignments(items, parser, option) -> dict:
    """{name: text} of NAME=TEXT items."""
    pairs = {}
    for item in items:
        name, separator, text = item.partition('=')
        if not separator or not name.strip():
            parser.error(f'{option} needs NAME=VALUE, got {item!r}')
        pairs[name.strip()] = text.strip()
    return pairs


def _model_targets(args, parser) -> dict:
    """{name: ModelTarget} of --models (every physics / role, or those named by --physics / --roles)."""
    from dev_tools.time_profilers.model_cases import model_targets
    scan = None
    if args.scan:
        (name, text), = _assignments([args.scan], parser, '--scan').items()
        values = tuple(value.strip() for value in text.split(',') if value.strip())
        if not values:
            parser.error('--scan needs at least one value')
        scan = (name, values)
    parameters = _assignments(args.set, parser, '--set')
    targets = {}
    for path in args.models:
        try:
            found = model_targets(os.path.abspath(path), args.physics, args.roles, scan, parameters)
        except (OSError, ValueError) as error:
            parser.error(f'--models {path}: {error}')
        if not found:
            print(f'{path}: no physics role its study uses matches --physics / --roles', flush=True)
        targets.update((target.name, target) for target in found)
    return targets


def _selected(candidates, patterns, backends):
    """Candidates on the asked backends matching any pattern (all without patterns), the baseline always."""
    return [candidate for candidate in candidates
            if candidate.baseline or (candidate.backend in backends and (
                not patterns or any(fnmatch.fnmatchcase(candidate.label, pattern) for pattern in patterns)))]


def _child_environment(threads):
    environment = dict(os.environ)
    if threads is not None:
        for name in THREAD_VARIABLES:
            environment[name] = str(threads)
    return environment


def _tail(text) -> str:
    text = (text or '').strip()
    return text[-STDERR_TAIL_CHARS:] if text else ''


def _text(output) -> str:
    """Child output as text (TimeoutExpired carries bytes on POSIX even with text=True)."""
    if isinstance(output, bytes):
        return output.decode('utf-8', errors='replace')
    return output or ''


def partial_result(stdout) -> dict:
    """{assembly fields..., 'repeats': finished repeats} from the PARTIAL_MARKER lines of a child's output (a line
    cut off by the kill is skipped)."""
    result = {'repeats': []}
    for line in _text(stdout).splitlines():
        if not line.startswith(PARTIAL_MARKER):
            continue
        try:
            entry = json.loads(line[len(PARTIAL_MARKER):])
        except json.JSONDecodeError:
            continue
        if entry.get('kind') == 'assembly':
            result.update(entry['fields'])
        elif entry.get('kind') == 'repeat':
            result['repeats'].append(entry['repeat'])
    return result


def run_child(case, timeout, environment) -> dict:
    """Result of one case in a fresh child (case_runner): its result line, or status timeout / killed (negative
    return code: a signal, SIGKILL usually the out-of-memory killer) / crashed (no result line), these three with
    the repeats the child finished before it ended."""
    started = time.perf_counter()
    try:
        completed = subprocess.run([sys.executable, '-m', CHILD_MODULE], input=json.dumps(case), capture_output=True,
                                   text=True, timeout=timeout, cwd=REPO_ROOT, env=environment)
    except subprocess.TimeoutExpired as expired:
        result = partial_result(expired.stdout)
        result.update(status='timeout', reason=f'no result within {timeout:g} s (child killed after '
                                               f"{len(result['repeats'])} finished repeats)",
                      child_seconds=time.perf_counter() - started, stderr_tail=_tail(_text(expired.stderr)))
        return result
    result_lines = [line for line in completed.stdout.splitlines() if line.startswith(RESULT_MARKER)]
    if result_lines:
        result = json.loads(result_lines[-1][len(RESULT_MARKER):])
    elif completed.returncode < 0:
        result = partial_result(completed.stdout)
        result.update(status='killed', reason=f'child ended by signal {-completed.returncode} (9: SIGKILL, usually '
                                              'the out-of-memory killer)')
    else:
        result = partial_result(completed.stdout)
        result.update(status='crashed', reason=_tail(completed.stderr) or f'exit code {completed.returncode}')
    result['child_seconds'] = time.perf_counter() - started
    result['returncode'] = completed.returncode
    if completed.returncode != 0 or result.get('status') != 'ok':
        result['stderr_tail'] = _tail(completed.stderr)
    return result


def _print_list(targets, sizes, catalog) -> None:
    for name, target in targets.items():
        candidates, excluded = catalog[name]
        print(f'{name}: {target.description}')
        print(f'  {target.physics_type} / {target.role} / {target.study} / grid {target.grid}; sizes {sizes[name]} '
              f'({target.size_text})')
        for candidate in candidates:
            marks = ''.join((' [baseline]' if candidate.baseline else '', ' [auto]' if candidate.automatic else ''))
            state = 'available' if candidate.available else f'unavailable: {candidate.reason}'
            print(f'    {candidate.row}{marks}: {state}')
        for entry in excluded:
            row = entry['label'] if entry['backend'] == 'cpu' else f"{entry['label']} @{entry['backend']}"
            print(f"    (not offered) {row}: {entry['reason']}")


def _record(target, size, candidate, result) -> dict:
    record = {'problem': target.name, 'kind': target.kind, 'size': size, 'solver': candidate.label,
              'row': candidate.row, 'backend': candidate.backend, 'solver_type': candidate.solver_type,
              'values': candidate.values, 'automatic': candidate.automatic, 'baseline': candidate.baseline}
    if target.kind == 'model':
        record['physics'] = target.physics_name
    record.update(result)
    record['backend'] = candidate.backend
    record['summary'] = report.summarise(record)
    return record


def _add_differences(records, targets) -> None:
    """eigen_difference of every synthetic eigen record and derived_difference of every model record against the
    baseline record of its problem and size."""
    baselines = {(record['problem'], record['size']): record for record in records if record['baseline']}
    for record in records:
        reference = baselines.get((record['problem'], record['size']))
        if reference is None or not record.get('repeats') or record['baseline']:
            continue
        if record['kind'] == 'model':
            record['derived_difference'] = report.derived_difference(record, reference)
        elif targets[record['problem']].role == 'eigen':
            record['eigen_difference'] = report.eigen_difference(record, reference)


def _folder_name(text) -> str:
    return re.sub(r'[^A-Za-z0-9_.=-]+', '_', str(text)).strip('_') or 'case'


def _case(target, size, candidate, args, repeat, run_root) -> dict:
    case = {'solver_type': candidate.solver_type, 'values': candidate.values, 'backend': candidate.backend,
            'repeat': repeat, 'warmup': not args.no_warmup, 'size': size}
    if target.kind != 'model':
        case['problem'] = target.name
        return case
    case.update(kind='model', path=target.path, physics=target.physics_name, role=target.role,
                scan_parameter=target.scan_parameter, parameters=target.parameters,
                run_dir=os.path.join(run_root, _folder_name(target.name), _folder_name(candidate.row),
                                     _folder_name('as_written' if size is None else size)))
    return case


def main(argv=None) -> int:
    args, parser = _parse_args(argv)
    names = args.problems or ([] if args.models else list(PROBLEMS))
    targets = {name: PROBLEMS[name] for name in names}
    sizes = _sizes(args, parser, names)
    if args.models:
        model_targets = _model_targets(args, parser)
        targets.update(model_targets)
        sizes.update((name, target.sizes) for name, target in model_targets.items())
    if not targets:
        parser.error('nothing to profile')
    repeat = args.repeat or (1 if args.quick else DEFAULT_REPEAT)
    timeout = args.timeout or (QUICK_TIMEOUT if args.quick else DEFAULT_TIMEOUT)
    catalogue_backends = tuple(backend for backend in BACKENDS if backend == 'cpu' or backend in args.backends)
    catalog = {name: solver_candidates(target, catalogue_backends) for name, target in targets.items()}
    if args.list:
        _print_list(targets, sizes, catalog)
        return 0
    selected = {name: _selected(catalog[name][0], args.solvers, args.backends) for name in targets}
    environment = _child_environment(args.threads)
    machine = report.machine_info(args.threads)
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.abspath(args.out)   # the children run in REPO_ROOT
    run_root = os.path.join(out_dir, 'model_runs', stamp)
    records = []
    for name, target in targets.items():
        for size in sizes[name]:
            for candidate in selected[name]:
                prefix = f"{name} {'as written' if size is None else f'n={size}'} {candidate.row}"
                if not candidate.available:
                    records.append(_record(target, size, candidate, {'status': 'skipped', 'reason': candidate.reason,
                                                                     'repeats': []}))
                    print(f'{prefix}: skipped ({candidate.reason})', flush=True)
                    continue
                case = _case(target, size, candidate, args, repeat, run_root)
                record = _record(target, size, candidate, run_child(case, timeout, environment))
                records.append(record)
                total = record['summary'].get('total', {}).get('median')
                timing = '' if total is None else f' {total:.3g} s{report.spread_text(record, "total")}'
                print(f"{prefix}:{timing} {record['status']}"
                      + (f" ({record.get('reason')})" if record.get('reason') and record['status'] != 'ok' else ''),
                      flush=True)
    _add_differences(records, targets)
    report.add_gpu(machine, records)
    text = report.markdown(targets, selected, records, sizes, machine)
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, f'solver_profile_{stamp}.json')
    markdown_path = os.path.join(out_dir, f'solver_profile_{stamp}.md')
    document = {'tool': 'dev_tools.time_profilers.solver_profiler', 'created': datetime.datetime.now().isoformat(),
                'machine': machine,
                'settings': {'problems': names, 'models': args.models, 'physics': args.physics, 'roles': args.roles,
                             'scan': args.scan, 'set': args.set, 'backends': args.backends, 'sizes': sizes,
                             'solvers': args.solvers, 'repeat': repeat, 'warmup': not args.no_warmup,
                             'threads': args.threads, 'timeout': timeout, 'quick': args.quick},
                'problems': {name: {'kind': target.kind, 'description': target.description,
                                    'physics_type': target.physics_type, 'role': target.role, 'study': target.study,
                                    'grid': target.grid, 'size_text': target.size_text} for name, target in
                             targets.items()},
                'solvers': {name: [dict(dataclasses.asdict(candidate), row=candidate.row)
                                   for candidate in catalog[name][0]] for name in targets},
                'not_offered': {name: catalog[name][1] for name in targets},
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
