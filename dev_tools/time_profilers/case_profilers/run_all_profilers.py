"""Run every profiler_solver_*.py command in turn, once per chosen solver (host libraries on the cpu backend, cuDSS on
the gpu, the others per --backend; a failing one is logged, the rest continue) -> out/run_all.log; each report is also
kept in out/history/<timestamp>/ (with the points file as it was before the run). Each finished command updates its
point in out/profile_points.json and the plot out/profile_points.html (a self-contained page: run wall, command wall,
GPU peak and RAM peak per command; one line per solver and backend for the first complete profile, the previous and
the current one; a ring on the solver Automatic picks on the gpu, worked out by the harness from each case's roles
whichever solver was run; the page chooses the panels, solvers, profiles, backends and commands it draws).
A command is killed after --timeout s (default 60); one that was profiling is plotted at the '>timeout' tick.
Run (from any folder): python run_all_profilers.py [--only SUBSTRING ...] [--pardiso --mumps-mixed ...]
[--backend gpu cpu] [--timeout S] [--new-first] [--plot-only]"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
OUT = HERE / 'out'
POINTS_PATH = OUT / 'profile_points.json'
PLOT_PATH = OUT / 'profile_points.html'
PAGE_TEMPLATE = HERE / 'profile_points_page.html'   # read from here by every suite (example_profilers too)
PAGE_DATA_MARKER = '__PROFILE_DATA__'
SERIES = ('first', 'previous', 'current')   # first: the first complete profile (all commands, no --only)
BACKENDS = ('gpu', 'cpu')
DEFAULT_TIMEOUT = 60.0   # seconds per command (subprocess killed); the harness's api.run has no limit of its own
NOTHING_PROFILED = 'nothing profiled'   # the harness's --compare-only line when no run takes the solver
NEVER_FACTORISED = 'never factorised'   # log mark of a run whose compared solver did no factorisation (not counted)
TIMEOUT_CAP = 1.5        # a timed-out point is drawn at this many times the largest timeout shown


def _compared(setting, value, backends=None):
    """Solver variant of one --compare value: only the runs a role of which takes it are profiled and counted;
    backends: the fixed backends it runs on (None: those of --backend)."""
    return {'arguments': ('--compare', f'{setting}={value}', '--compare-only'), 'token': f'{setting}={value}',
            'backends': backends}


GPU, CPU = ('gpu',), ('cpu',)
# solver variants, one --<name> flag each (none given: auto); the order fixes each one's plot colour. The host
# libraries (SuperLU, PARDISO, MUMPS) run on the cpu backend only and cuDSS on the gpu only; a gpu model may still use a
# host library where its Automatic picks one (auto @gpu, e.g. SuperLU for some FDFD / membrane cases)
SOLVERS = {
    'auto': {'arguments': (), 'token': None, 'backends': None},   # the cases' solvers as written (mostly Automatic)
    'cudss': _compared('back_end', 'cudss', GPU),
    'cudss_mixed': _compared('back_end', 'cudss_mixed', GPU),
    'scipy_splu': _compared('back_end', 'scipy_splu', CPU),
    'pardiso': _compared('back_end', 'pardiso', CPU),
    'pardiso_mixed': _compared('back_end', 'pardiso_mixed', CPU),
    'mumps': _compared('back_end', 'mumps', CPU),
    'mumps_mixed': _compared('back_end', 'mumps_mixed', CPU),
    'multigrid': _compared('preconditioner', 'multigrid'),
    'jacobi': _compared('preconditioner', 'jacobi'),
}
# categorical palette slots in fixed order (8 hues), then two neutrals for the preconditioner variants: (light, dark)
# steps of each, one per SOLVERS entry (the page draws a solver without a slot, e.g. a renamed one, in text colour)
SOLVER_COLORS = (('#2a78d6', '#3987e5'), ('#eb6834', '#d95926'), ('#1baf7a', '#199e70'), ('#eda100', '#c98500'),
                 ('#e87ba4', '#d55181'), ('#008300', '#0ca30c'), ('#6250d6', '#9085e9'), ('#e34948', '#e66767'),
                 ('#52514e', '#c3c2b7'), ('#a3a29c', '#7c7b76'))
COMMANDS = (
    'pressure_acoustics', 'pressure_acoustics_tetrahedra',
    'em_fdfd_triangles', 'em_modes_triangles',
    'em_fdfd_3d', 'em_fdfd_3d_double_factor', 'em_fdfd_3d_tetrahedra', 'em_fdfd_3d_tetrahedra_double_factor',
    'em_eme_3d',
    'heat_fem --kind triangles', 'heat_fem --kind tetrahedra', 'heat_fem --kind steady',
    'magnetostatics_3d --kind tensor', 'magnetostatics_3d --kind tetrahedra', 'magnetostatics_3d --kind sphere',
    'magnetostatics_3d --kind coil',
    'solid_mechanics_triangles', 'solid_mechanics_3d_tensor --warmup',
    'solid_mechanics_3d_tetrahedra --warmup', 'solid_mechanics_3d_nonlinear --warmup',   # cuDSS load out of run 1
    'membrane_plate',
)
PLOT_TITLE = 'case profilers per command'


def command_line(entry) -> list:
    """[script path, arguments...] of one COMMANDS entry (a suite reusing main replaces this, COMMANDS, OUT,
    POINTS_PATH, PLOT_PATH and PLOT_TITLE: example_profilers/run_all_examples.py)."""
    script, *arguments = entry.split()
    return [str(HERE / f'profiler_solver_{script}.py'), *arguments]


def variant_label(solver, backend) -> str:
    return f'{solver} @{backend}'


def command_point(runs_path, command_seconds, started, token):
    """({run_seconds (sum of the counted runs' walls), command_seconds, gpu_peak_bytes / host_peak_bytes (largest over
    them), runs, when} or None, names of the token's runs not counted) of one command; counted: the runs whose name
    holds token (all for None) and in which a role taking the token's value factorised (a matrix-free eigen solve
    never does: that run measures another solver). None when runs.json is missing, older than the command, has no
    counted run or one of the token's runs did not complete (a part not run would shorten the wall)."""
    if not runs_path.is_file() or runs_path.stat().st_mtime < started:
        return None, []
    named = {name: run for name, run in json.loads(runs_path.read_text(encoding='utf-8')).items()
             if token is None or token in name}
    value = None if token is None else token.partition('=')[2]

    def idle(run):   # every role holding the compared value reports 0 factorisations (None: no counter, counted)
        taken = [choice for choice in run.get('solvers') or ()
                 if value is not None and value in (choice.get('back_end'), choice.get('preconditioner'))]
        return bool(taken) and all(choice.get('factorizations') == 0 for choice in taken)
    completed = {name: run for name, run in named.items() if run.get('status') == 'completed'}
    runs = [run for run in completed.values() if not idle(run)]
    not_completed = ([f"{name} ({run.get('status')})" for name, run in named.items() if name not in completed]
                     + [f'{name} ({value} {NEVER_FACTORISED})' for name, run in completed.items() if idle(run)])
    if not runs or len(completed) < len(named):
        return None, not_completed

    def largest(key):
        values = [run[key] for run in runs if run.get(key) is not None]
        return max(values) if values else None
    def names(field):   # the back ends / preconditioners of the counted runs' solver records under field
        return sorted({choice[key] for run in runs for choice in run.get(field) or ()
                       for key in ('back_end', 'preconditioner') if choice.get(key)})
    # chosen: what the runs used; automatic: what Automatic picks on the gpu for the same runs (ringed in the plot)
    return {'run_seconds': sum(run['wall_seconds'] for run in runs), 'command_seconds': command_seconds,
            'gpu_peak_bytes': largest('gpu_peak_bytes'), 'host_peak_bytes': largest('host_peak_bytes'),
            'runs': len(runs), 'chosen': names('solvers'), 'automatic': names('automatic'),
            'when': time.strftime('%Y-%m-%d %H:%M:%S')}, not_completed


def used_solvers(runs_path, started) -> str | None:
    """'+'-joined solvers the command's runs used per role (back end, else preconditioner, else solver type), or None
    when runs.json is missing or older than the command."""
    if not runs_path.is_file() or runs_path.stat().st_mtime < started:
        return None
    names = {choice.get('back_end') or choice.get('preconditioner') or choice.get('type')
             for run in json.loads(runs_path.read_text(encoding='utf-8')).values()
             for choice in run.get('solvers') or ()}
    return '+'.join(sorted(name for name in names if name)) or None


def timed_out_point(command_seconds, timeout):
    """Point of a command killed at the timeout: no run wall or memory, drawn at the timeout tick."""
    return {'run_seconds': None, 'timed_out': True, 'timeout_seconds': timeout, 'command_seconds': command_seconds,
            'gpu_peak_bytes': None, 'host_peak_bytes': None, 'runs': None, 'when': time.strftime('%Y-%m-%d %H:%M:%S')}


def _text(output) -> str:
    """Captured output of a killed command as text (bytes off Windows, str on it, None when there was none)."""
    if output is None:
        return ''
    return output.decode(errors='replace') if isinstance(output, bytes) else output


def load_points():
    """{variant: {series: {command: point}}} of POINTS_PATH (empty when there is none yet). Never drops points: an
    unreadable file raises (restore it from out/history/<run>/profile_points_before.json), the first one-solver layout
    {series: {command: point}} is read as 'auto @gpu'."""
    if not POINTS_PATH.is_file():
        return {}
    try:
        points = json.loads(POINTS_PATH.read_text(encoding='utf-8'))
    except ValueError as error:
        raise SystemExit(f'{POINTS_PATH} is unreadable ({error}): restore it from a profile_points_before.json in '
                         f'{OUT / "history"} before running again') from None
    if any(series in points for series in SERIES):
        points = {variant_label('auto', 'gpu'): points}
    return {variant: {series: dict(by_series.get(series, {})) for series in SERIES}
            for variant, by_series in points.items()}


def save_points(points):
    """Write POINTS_PATH through a sibling file and a replace, so a killed run never leaves half a file."""
    POINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    partial = POINTS_PATH.with_name(POINTS_PATH.name + '.partial')
    partial.write_text(json.dumps(points, indent=1), encoding='utf-8')
    os.replace(partial, POINTS_PATH)


def record_point(points, variant, entry, point, first=False):
    """The command's current point of this variant becomes its previous one, the new point its current one; first:
    also its first point (a complete profile and no first one yet, or --new-first)."""
    by_series = points.setdefault(variant, {series: {} for series in SERIES})
    if entry in by_series['current']:
        by_series['previous'][entry] = by_series['current'][entry]
    by_series['current'][entry] = point
    if first:
        by_series['first'][entry] = point


def auto_picks(points):
    """{(command, variant)} to ring: per command, every solver Automatic picks on the gpu (the 'automatic' of any
    variant's current point; an older auto @gpu point without it: what it used), on its variant on the gpu, else on
    its own fixed backend (a host library Automatic picks on a gpu model: its @cpu line)."""
    picked = {}   # command -> solver names Automatic picks
    for variant, by_series in points.items():
        for entry, point in by_series['current'].items():
            names = point.get('automatic')
            if names is None and variant == variant_label('auto', 'gpu'):
                names = point.get('chosen')
            picked.setdefault(entry, set()).update(names or ())
    picks = set()
    for entry, names in picked.items():
        for used in names:
            if used not in SOLVERS or used == 'auto':
                continue
            candidates = [variant_label(used, backend) for backend in SOLVERS[used]['backends'] or BACKENDS]
            target = next((name for name in candidates if name in points and entry in points[name]['current']), None)
            if target is not None:
                picks.add((entry, target))
    return picks


def _variant_order(variant):
    solver, _, backend = variant.partition(' @')
    return (list(SOLVERS).index(solver) if solver in SOLVERS else len(SOLVERS), backend, variant)


def plot_points(points, path):
    """Write the profile page (PAGE_TEMPLATE filled with the points; self-contained, opens in a browser) to path and
    return the path written (None: nothing measured yet, or not saved). On the page: panels run wall s (log), command
    wall s, GPU peak MiB, RAM peak MiB sharing the command axis; one colour per solver, one marker per backend, one
    line style per series (first dotted, previous dashed, current solid), a ring on the solver Automatic picks; which
    panels, solvers, series, backends and commands are drawn is chosen there (hover / table for the values)."""
    measured = {entry for by_series in points.values() for series in SERIES for entry in by_series[series]}
    entries = [entry for entry in COMMANDS if entry in measured] + sorted(measured - set(COMMANDS))   # dropped last
    if not entries:
        return None
    if len(SOLVERS) > len(SOLVER_COLORS):
        raise ValueError(f'{len(SOLVERS)} SOLVERS but {len(SOLVER_COLORS)} SOLVER_COLORS slots: give the new solver '
                         'a colour (never a cycled one)')
    timeouts = [point['timeout_seconds'] for by_series in points.values() for series in SERIES
                for point in by_series[series].values() if point.get('timed_out')]
    largest_timeout = max(timeouts, default=None)
    first_when = min((point['when'] for by_series in points.values() for point in by_series['first'].values()),
                     default='none yet')
    variants = []
    for variant in sorted(points, key=_variant_order):
        solver, _, backend = variant.partition(' @')
        variants.append({'name': variant, 'solver': solver, 'backend': backend,
                         'slot': list(SOLVERS).index(solver) if solver in SOLVERS else None,   # None: text colour
                         'points': {series: points[variant][series] for series in SERIES}})
    data = {'title': PLOT_TITLE, 'first_when': first_when, 'written': time.strftime('%Y-%m-%d %H:%M:%S'),
            'commands': entries, 'dropped': sorted(measured - set(COMMANDS)), 'colors': SOLVER_COLORS,
            'timeout': None if largest_timeout is None else {'largest': largest_timeout,
                                                             'cap': TIMEOUT_CAP * largest_timeout},
            'variants': variants, 'picks': sorted([entry, variant] for entry, variant in auto_picks(points))}
    template = PAGE_TEMPLATE.read_text(encoding='utf-8')
    if template.count(PAGE_DATA_MARKER) != 1:
        raise ValueError(f'{PAGE_TEMPLATE} must hold {PAGE_DATA_MARKER} exactly once')
    payload = json.dumps(data, allow_nan=False, separators=(',', ':')).replace('</', '<\\/')   # no early </script>
    return _save_page(template.replace(PAGE_DATA_MARKER, payload), path)


def _save_page(text, path):
    """Write a timestamped sibling, then replace path with it; a path that cannot be replaced (Windows: held open)
    keeps the new page beside it instead of failing the whole run."""
    fresh = path.with_name(f'{path.stem}_{time.strftime("%Y%m%d_%H%M%S")}{path.suffix}')
    try:
        fresh.parent.mkdir(parents=True, exist_ok=True)
        fresh.write_text(text, encoding='utf-8')
    except OSError as error:
        print(f'plot not saved ({error}); redraw with --plot-only', flush=True)
        return None
    try:
        os.replace(fresh, path)
    except OSError as error:
        print(f'{path} is held open ({error}): the new plot is {fresh}', flush=True)
        return fresh
    return path


def main(description=None):
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument('--only', nargs='*', default=None, help='run only commands containing one of these')
    for solver, variant in SOLVERS.items():
        parser.add_argument(f'--{solver.replace("_", "-")}', dest=f'solver_{solver}', action='store_true',
                            help=f'profile with {" ".join(variant["arguments"][:2]) or "the cases as written"}'
                                 + (f' (backend {variant["backends"][0]})' if variant['backends'] else ''))
    parser.add_argument('--backend', nargs='+', default=['gpu'], choices=BACKENDS,
                        help='backends of auto / multigrid / jacobi (default gpu); the host libraries always run on '
                             'the cpu, cuDSS on the gpu')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'seconds per command before it is killed (default {DEFAULT_TIMEOUT:g}; 0: no limit); a '
                             'killed command that was profiling is plotted at the timeout tick')
    parser.add_argument('--new-first', action='store_true',
                        help='this complete profile replaces the first one (commands that fail keep theirs)')
    parser.add_argument('--plot-only', action='store_true', help='redraw the plot from profile_points.json, run nothing')
    options = parser.parse_args()
    points = load_points()
    if options.plot_only:
        print(f'plot: {plot_points(points, PLOT_PATH)}')
        return
    solvers = [solver for solver in SOLVERS if getattr(options, f'solver_{solver}')] or ['auto']
    variants = [(solver, backend) for solver in solvers for backend in SOLVERS[solver]['backends'] or options.backend]
    commands = [entry for entry in COMMANDS if not options.only or any(text in entry for text in options.only)]
    complete = not options.only
    log_path = OUT / 'run_all.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history_folder = OUT / 'history' / time.strftime('%Y-%m-%d_%H%M%S')
    history_folder.mkdir(parents=True, exist_ok=True)
    if POINTS_PATH.is_file():   # the points as they were before this run, to restore from
        shutil.copyfile(POINTS_PATH, history_folder / 'profile_points_before.json')
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")}: {len(commands)} commands x '
                  f'{[variant_label(*variant) for variant in variants]}, reports kept in {history_folder} =====\n')
        for solver, backend in variants:
            label = variant_label(solver, backend)
            for entry in commands:
                command = [sys.executable, *command_line(entry), '--backend', backend, *SOLVERS[solver]['arguments']]
                print(f'>>> {entry} [{label}]', flush=True)
                started = time.time()
                start = time.perf_counter()
                try:
                    completed = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True,
                                               timeout=options.timeout or None)
                    return_code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
                except subprocess.TimeoutExpired as expired:   # run() has killed the command
                    return_code, stdout, stderr = None, _text(expired.stdout), _text(expired.stderr)
                wall = time.perf_counter() - start
                report = [line for line in stdout.splitlines() if line.startswith('report: ')]
                point, not_completed, shown = None, [], label   # shown: auto replaced by the solvers it used
                if return_code is None:
                    if NOTHING_PROFILED not in stdout:   # it was profiling this solver when killed
                        point = timed_out_point(wall, options.timeout)
                elif report:
                    report_path = Path(report[-1][len('report: '):].strip())
                    if report_path.is_file():   # the folder name holds the backend and --compare settings
                        shutil.copyfile(report_path, history_folder / f'{report_path.parent.name}.txt')
                    runs_path = report_path.parent / 'runs.json'   # per-part seconds: what profile_delta.py compares
                    if runs_path.is_file() and runs_path.stat().st_mtime >= started:
                        shutil.copyfile(runs_path, history_folder / f'{report_path.parent.name}.runs.json')
                    used = used_solvers(runs_path, started) if solver == 'auto' else None
                    if used:
                        shown = variant_label(used, backend)
                    if return_code == 0:
                        point, not_completed = command_point(report_path.parent / 'runs.json', wall, started,
                                                             SOLVERS[solver]['token'])
                if point is not None:   # saved at once: a later command that hangs or crashes loses nothing
                    first = complete and (options.new_first or entry not in points.get(label, {}).get('first', {}))
                    record_point(points, label, entry, point, first=first)
                    save_points(points)
                elif not_completed and all(NEVER_FACTORISED in text for text in not_completed):
                    # every run of this solver measured another one: its earlier points of the command go too
                    if [series for series in SERIES if points.get(label, {}).get(series, {}).pop(entry, None)]:
                        save_points(points)
                tail ='' if return_code in (0, None) else '\n    ' + '\n    '.join(stderr.splitlines()[-8:])
                if return_code is None:
                    reported = (f'timed out after {options.timeout:g} s'
                                + (' (plotted at the timeout tick)' if point is not None else ''))
                else:   # pass: the script exited 0 and every run of the solver completed
                    run_failed = any(NEVER_FACTORISED not in text for text in not_completed)
                    verdict = (f'FAIL (exit {return_code})' if return_code else
                               'FAIL (a run did not complete: not recorded, the plot keeps the earlier point)'
                               if run_failed else 'pass')
                    reported = (f'{verdict}, {wall:.1f} s, '
                                + (report[-1] if report else 'no report (no run takes this solver, or it failed)'))
                failed_runs = f', not counted: {", ".join(not_completed)}' if not_completed else ''
                line = f'{entry} [{shown}]: {reported}{failed_runs}{tail}'
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
    print(f'log: {log_path}\npoints: {POINTS_PATH}\nplot: {plot_points(points, PLOT_PATH)}')


if __name__ == '__main__':
    main()
