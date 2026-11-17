"""Run every profiler_solver_*.py command in turn, once per chosen solver (host libraries on the cpu backend, cuDSS on
the gpu, the others per --backend; a failing one is logged, the rest continue) -> out/run_all.log; each report is also
kept in out/history/<timestamp>/ (with the points file as it was before the run). Each finished command updates its
point in out/profile_points.json and the plot out/profile_points.png (run wall, GPU peak and RAM peak per command; one
line per solver and backend for the first complete profile, the previous and the current one; a ring on the solver
Automatic picks on the gpu, worked out by the harness from each case's roles whichever solver was run).
A command is killed after --timeout s (default 60); one that was profiling is plotted at the '>timeout' tick.
A command whose last complete profile still stands (harness --if-changed: same case specs and options, no changed unit
of the solver code it ran) is not rerun and keeps its points; --reset, --new-first and --only (named commands) profile
every command they run. run_all remembers each kept outcome (out/run_all_memory.json) with a snapshot of the
sources: while no .py / .json under open_sol/, dev_tools/ or examples/ changed, or only solver code (harness
PROFILED_CODE) in units the kept profile never ran, the command is not even started; the others are checked
together first (check_profiles.py, one process) and only those found changed are started. A
command killed at the timeout keeps its completed runs (harness pieces): the next run_all runs only the rest.
Each command's runs are checked by the check() of the selftest case they came from (harness solution check): the
counts go on its log line, a failed / error group is logged under it and listed again at the end (points still recorded).
Run (from any folder): python run_all_profilers.py [--only SUBSTRING ...] [--pardiso --mumps-mixed ...]
[--backend gpu cpu] [--timeout S] [--reset] [--new-first] [--plot-only]"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
OUT = HERE / 'out'
POINTS_PATH = OUT / 'profile_points.json'
PLOT_PATH = OUT / 'profile_points.png'
SERIES = ('first', 'previous', 'current')   # first: the first complete profile (all commands, no --only)
BACKENDS = ('gpu', 'cpu')
DEFAULT_TIMEOUT = 60.0   # seconds per command (subprocess killed); the harness's api.run has no limit of its own
NOTHING_PROFILED = 'nothing profiled'   # the harness's --compare-only line when no run takes the solver
NEVER_FACTORISED = 'never factorised'   # log mark of a run whose compared solver did no factorisation (not counted)
UNCHANGED = 'unchanged since '   # the harness's UNCHANGED_MARKER line (--if-changed): its kept profile stands
KEPT_REPORT = 'kept report: '    # the harness's KEPT_REPORT_MARKER line: that profile's report.txt
SOLUTION_CHECK = 'solution check: '   # the harness's SOLUTION_CHECK_MARKER lines: '<status> <group>: <detail>'
CHECK_NOT_PASSED = ('failed', 'error')
MEMORY_FILE = 'run_all_memory.json'   # in OUT: per command line, its kept outcome and the sources it stands for
# folders under REPOSITORY whose .py / .json bytes make the source fingerprint (out, *_out, __pycache__, hidden skipped)
FINGERPRINT_FOLDERS = ('open_sol', 'dev_tools', 'examples')
FINGERPRINT_SUFFIXES = ('.py', '.json')
_file_digests = {}   # path -> ((mtime_ns, size), sha1): a file is read again only after it changed
_unit_hashes = {}    # (path, sha1) -> {unit name: hash} of a changed solver file (parsed once per run_all)
_loaded = {}         # file stem -> module loaded on its own (selftest_cache, profiler_harness: no open_sol import)
BATCH_CHECK_MIN_SECONDS = 120.0   # timeout floor of the one-process check of all commands (check_profiles.py)
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
# categorical palette slots in fixed order (8 hues), then two neutrals for the preconditioner variants
SOLVER_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948',
                 '#52514e', '#a3a29c')
BACKEND_MARKERS = {'gpu': 'o', 'cpu': 's'}
SERIES_STYLES = {'first': dict(linestyle=':', linewidth=1.2, markersize=5, markerfacecolor='none'),
                 'previous': dict(linestyle='--', linewidth=1.2, markersize=4),
                 'current': dict(linestyle='-', linewidth=2.0, markersize=6)}
# ring around the current point of the solver variant Automatic picks on the gpu (auto_picks)
PICK_RING = dict(s=220, facecolors='none', edgecolors='#0b0b0b', linewidths=1.5, zorder=5)

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


def check_summary(stdout):
    """('; solution check <count> <status>, ...' or '', [lines of the failed / error groups]) of a command's output."""
    lines = [line[len(SOLUTION_CHECK):] for line in stdout.splitlines() if line.startswith(SOLUTION_CHECK)]
    statuses = [line.partition(' ')[0] for line in lines]
    counts = ', '.join(f'{statuses.count(status)} {status.upper() if status in CHECK_NOT_PASSED else status}'
                       for status in dict.fromkeys(statuses))
    return (f'; solution check {counts}' if lines else '',
            [line for line, status in zip(lines, statuses) if status in CHECK_NOT_PASSED])


def source_snapshot():
    """(snapshot id, {repository path: sha1} of every FINGERPRINT_FOLDERS source file, environment text: interpreter,
    numpy / scipy / cupy versions, machine); an equal id -> nothing a command reads has changed."""
    files = {}
    for name in FINGERPRINT_FOLDERS:
        for folder, subfolder_names, file_names in os.walk(REPOSITORY / name):
            subfolder_names[:] = sorted(sub for sub in subfolder_names if sub not in ('__pycache__', 'out')
                                        and not sub.endswith('_out') and not sub.startswith('.'))
            for file_name in sorted(file_names):
                if not file_name.endswith(FINGERPRINT_SUFFIXES):
                    continue
                path = os.path.join(folder, file_name)
                status = os.stat(path)
                known = _file_digests.get(path)
                if known is None or known[0] != (status.st_mtime_ns, status.st_size):
                    with open(path, 'rb') as handle:
                        known = _file_digests[path] = ((status.st_mtime_ns, status.st_size),
                                                       hashlib.sha1(handle.read()).hexdigest())
                files[os.path.relpath(path, REPOSITORY).replace(os.sep, '/')] = known[1]
    versions = []
    for module_name in ('numpy', 'scipy', 'cupy'):
        try:
            versions.append(importlib.metadata.version(module_name))
        except importlib.metadata.PackageNotFoundError:
            versions.append(None)
    environment = json.dumps([sys.executable, sys.version, versions, platform.node()])
    snapshot = hashlib.sha1(json.dumps(sorted(files.items())).encode('utf-8')).hexdigest()
    return snapshot, files, environment


def load_memory() -> dict:
    """{'commands': {command line: {snapshot, environment, lines, cache, cache_time}}, 'snapshots': {snapshot id:
    files}} of OUT / MEMORY_FILE (empty when missing, unreadable or of an older layout: the memory only saves time,
    the harness check stands behind it)."""
    try:
        memory = json.loads((OUT / MEMORY_FILE).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        memory = {}
    if not isinstance(memory.get('commands'), dict) or not isinstance(memory.get('snapshots'), dict):
        memory = {'commands': {}, 'snapshots': {}}
    return memory


def save_memory(memory):
    """Write the memory (snapshots no command refers to dropped) through a sibling file and a replace."""
    used = {record.get('snapshot') for record in memory['commands'].values()}
    memory['snapshots'] = {key: files for key, files in memory['snapshots'].items() if key in used}
    path = OUT / MEMORY_FILE
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(memory), encoding='utf-8')
    os.replace(partial, path)


def _standalone(stem, path):
    """Module of one file loaded on its own (cached): selftest_cache's unit hashing and the harness's PROFILED_CODE
    rule, without importing open_sol (seconds) into run_all."""
    if stem not in _loaded:
        spec = importlib.util.spec_from_file_location(f'_run_all_{stem}', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module   # as an import does (dataclasses look their module up)
        spec.loader.exec_module(module)
        _loaded[stem] = module
    return _loaded[stem]


def _solver_edits_stand(changed, files, cache_units):
    """True when every changed repository path is a PROFILED_CODE .py file of open_sol (the harness's solver code,
    no case spec source) and no unit of cache_units (the kept profile's) in those files has another hash now."""
    hasher = _standalone('selftest_cache', REPOSITORY / 'open_sol' / 'model' / 'selftest_cache.py')
    harness = _standalone('profiler_harness', HERE / 'profiler_harness.py')
    by_file = {}
    for unit_id, digest in cache_units.items():
        package_path, _, name = unit_id.partition('::')
        by_file.setdefault(package_path, []).append((name, digest))
    for path in changed:
        full_path = str(REPOSITORY / path)
        package_path = hasher._package_path(full_path) if path.startswith('open_sol/') else None
        if package_path is None or not harness._profiled_path(package_path):
            return False   # a case, harness, example or plumbing source: only the harness can judge it
        units = by_file.get(package_path)
        if not units:
            continue   # solver code this profile never ran
        key = (path, files.get(path))
        if key not in _unit_hashes:
            _unit_hashes[key] = hasher._file_units(full_path).hashes if key[1] is not None else {}
        if any(_unit_hashes[key].get(name) != digest for name, digest in units):
            return False
    return True


def remembered_lines(record, snapshot, files, environment, snapshots):
    """(the command's kept stdout lines, how) when its memory record stands, else None: its kept profile's
    profile_cache.json still complete and the same (not replaced by a run by hand), the same environment, and the same
    sources or only solver-code edits that leave every unit the profile ran unchanged (_solver_edits_stand)."""
    if record is None or record.get('environment') != environment:
        return None
    cache = None
    if record.get('cache') is not None:
        try:
            cache = json.loads(Path(record['cache']).read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None
        if cache.get('units') is None or cache.get('time') != record.get('cache_time'):
            return None
    if record.get('snapshot') == snapshot:
        return record['lines'], 'remembered'
    earlier = snapshots.get(record.get('snapshot'))
    if earlier is None or cache is None:   # a nothing-profiled outcome has no units to compare: the harness decides
        return None
    changed = sorted(path for path in earlier.keys() | files.keys() if earlier.get(path) != files.get(path))
    if not _solver_edits_stand(changed, files, cache['units']):
        return None
    return record['lines'], f'remembered: {len(changed)} solver file(s) edited, no unit it ran'


def precheck(commands, memory):
    """{memory key: (stdout lines, how)} of the commands {key: command} whose kept outcome stands: from the memory, the
    rest by one check_profiles.py process (the harness --if-changed check of all of them, open_sol loaded once);
    memory refreshed and saved. A failed batch leaves its commands to check themselves."""
    snapshot, files, environment = source_snapshot()
    standing, batch = {}, {}
    for key, command in commands.items():
        remembered = remembered_lines(memory['commands'].get(key), snapshot, files, environment, memory['snapshots'])
        if remembered is None:
            batch[key] = command[1:]
            continue
        standing[key] = remembered
        memory['commands'][key]['snapshot'] = snapshot   # it stands for these sources too
        memory['snapshots'][snapshot] = files
    if batch:
        print(f'checking {len(batch)} of {len(commands)} commands in one process (solver code, case specs, '
              'options) ...', flush=True)
        commands_path, results_path = OUT / 'check_commands.json', OUT / 'check_results.json'
        commands_path.write_text(json.dumps(batch, indent=1), encoding='utf-8')
        started, start = time.time(), time.perf_counter()
        try:   # a check is a spec build, no run: a hang (>= 10 s per command) leaves them to check themselves
            completed = subprocess.run([sys.executable, str(HERE / 'check_profiles.py'), str(commands_path),
                                        str(results_path)], cwd=REPOSITORY, capture_output=True, text=True,
                                       timeout=max(BATCH_CHECK_MIN_SECONDS, 10.0 * len(batch)))
            return_code, stderr = completed.returncode, completed.stderr
        except subprocess.TimeoutExpired as expired:
            return_code, stderr = None, _text(expired.stderr)
        seconds = time.perf_counter() - start
        try:   # never a results file of an earlier batch
            results = (json.loads(results_path.read_text(encoding='utf-8'))
                       if results_path.stat().st_mtime >= started else {})
        except (OSError, ValueError):
            results = {}
        if return_code != 0 or not results:
            print(f'batch check failed ({"timed out" if return_code is None else f"exit {return_code}"}): each '
                  'command checks itself\n    ' + '\n    '.join(stderr.splitlines()[-8:]), flush=True)
        broken = [f'{key}: {result["error"]}' for key, result in results.items() if result.get('error')]
        if broken:   # those commands run on their own and show the error in full
            print('batch check could not judge:\n    ' + '\n    '.join(broken), flush=True)
        for key, result in results.items():
            if result.get('lines'):
                standing[key] = (result['lines'], None)   # its time is in the one summary line below
                record = memory_record('\n'.join(result['lines']), 0, snapshot, environment)
                if record is not None:
                    memory['commands'][key] = record
                    memory['snapshots'][snapshot] = files
        print(f'batch check: {sum(key in standing for key in batch)} of {len(batch)} unchanged ({seconds:.1f} s)',
              flush=True)
    save_memory(memory)
    return standing


def memory_record(stdout, return_code, snapshot, environment):
    """Memory record of a finished command whose outcome stands until a source changes: a kept profile (harness
    unchanged line), a complete profile (its profile_cache.json filled) or no run taking the solver; else None."""
    if return_code != 0:
        return None
    lines = stdout.splitlines()
    checks = [line for line in lines if line.startswith(SOLUTION_CHECK)]
    kept = next((line[len(KEPT_REPORT):].strip() for line in lines if line.startswith(KEPT_REPORT)), None)
    report = next((line[len('report: '):].strip() for line in reversed(lines) if line.startswith('report: ')), None)
    if kept is None and report is None:
        nothing = [line for line in lines if NOTHING_PROFILED in line]
        return ({'snapshot': snapshot, 'environment': environment, 'lines': nothing, 'cache': None} if nothing
                else None)
    report_path = Path(kept or report)
    cache_path = report_path.parent / 'profile_cache.json'
    try:
        cache = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if cache.get('units') is None:   # incomplete, failed or --no-cprofile: never kept
        return None
    return {'snapshot': snapshot, 'environment': environment, 'cache': str(cache_path), 'cache_time': cache.get('time'),
            'lines': [f"{UNCHANGED}{cache.get('time')}: remembered by run_all", f'{KEPT_REPORT}{report_path}',
                      *checks]}


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
    """Three panels sharing the command axis (run wall s, log; GPU peak MiB; RAM peak MiB): one colour per solver,
    one marker per backend, one line style per series (first dotted, previous dashed, current solid), a ring on the
    solver Automatic picks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    measured = {entry for by_series in points.values() for series in SERIES for entry in by_series[series]}
    entries = [entry for entry in COMMANDS if entry in measured] + sorted(measured - set(COMMANDS))   # dropped last
    if not entries:
        return None
    variants = sorted(points, key=_variant_order)
    picks = {(entry, variant) for entry, variant in auto_picks(points) if entry in entries}

    def color(variant):
        solver = variant.partition(' @')[0]
        return SOLVER_COLORS[list(SOLVERS).index(solver) % len(SOLVER_COLORS)] if solver in SOLVERS else '#0b0b0b'
    panels = (('run_seconds', 1.0, 'run wall (s)', True), ('gpu_peak_bytes', 2.0 ** -20, 'GPU peak (MiB)', False),
              ('host_peak_bytes', 2.0 ** -20, 'RAM peak (MiB)', False))
    timeouts = [point['timeout_seconds'] for by_series in points.values() for series in SERIES
                for entry, point in by_series[series].items() if point.get('timed_out') and entry in entries]
    largest_timeout = max(timeouts, default=None)
    cap = None if largest_timeout is None else TIMEOUT_CAP * largest_timeout   # height of every timed-out point

    def value(point, key, variant):
        if point is None or (key == 'gpu_peak_bytes' and variant.endswith(' @cpu')):   # cpu: VRAM not measured
            return None
        return cap if key == 'run_seconds' and point.get('timed_out') else point.get(key)
    figure, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(max(11.0, 0.3 * len(entries)), 11.0))
    positions = list(range(len(entries)))
    for axis, (key, scale, label, logarithmic) in zip(axes, panels):
        for variant in variants:
            marker = BACKEND_MARKERS.get(variant.partition(' @')[2], 'o')
            for series in SERIES:
                values = [value(points[variant][series].get(entry), key, variant) for entry in entries]
                if all(item is None for item in values):
                    continue
                heights = [math.nan if item is None else item * scale for item in values]   # gaps break the line
                axis.plot(positions, heights, color=color(variant), marker=marker, **SERIES_STYLES[series])
        for entry, variant in picks:   # a ring on the current point of the solver Automatic picks
            height = value(points[variant]['current'].get(entry), key, variant)
            if height is not None:
                axis.scatter([entries.index(entry)], [height * scale], **PICK_RING)
        axis.set_ylabel(label)
        if logarithmic:
            axis.set_yscale('log')
        if key == 'run_seconds' and cap is not None:   # one tick '>timeout' above the timeout line
            from matplotlib.ticker import NullFormatter
            bottom = axis.get_ylim()[0]
            axis.set_ylim(bottom, cap * 1.3)
            ticks = [tick for tick in axis.get_yticks() if bottom <= tick < largest_timeout]
            axis.set_yticks(ticks + [cap])
            axis.set_yticklabels([f'{tick:g}' for tick in ticks] + [f'>{largest_timeout:g}'])
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.axhline(largest_timeout, color='#c3c2b7', linewidth=0.8)
        axis.grid(True, color='#e4e3df', linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    first_when = min((point['when'] for by_series in points.values() for point in by_series['first'].values()),
                     default='none yet')
    axes[0].set_title(f'{PLOT_TITLE} (first complete profile {first_when}; GPU peak: pool high-water '
                      '+ cuDSS factors; RAM peak: the process)', fontsize=10, loc='left')
    variant_handles = [Line2D([], [], color=color(variant), marker=BACKEND_MARKERS.get(variant.partition(' @')[2], 'o'),
                              linewidth=2.0, label=variant) for variant in variants]
    series_handles = [Line2D([], [], color='#52514e', label=series, **SERIES_STYLES[series]) for series in SERIES]
    series_handles.append(Line2D([], [], linestyle='none', marker='o', markersize=13, markerfacecolor='none',
                                 markeredgecolor=PICK_RING['edgecolors'], markeredgewidth=PICK_RING['linewidths'],
                                 label="Automatic's pick on the gpu (current)"))
    solver_legend = axes[0].legend(handles=variant_handles, loc='upper left', bbox_to_anchor=(1.0, 1.0),
                                   frameon=False, title='solver @backend')
    axes[0].add_artist(solver_legend)
    axes[1].legend(handles=series_handles, loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False,
                   title='profile')
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels(entries, rotation=90, fontsize=8)
    figure.tight_layout()
    saved = _save_figure(figure, path)
    plt.close(figure)
    return saved


def _save_figure(figure, path):
    """Save to a timestamped sibling, then replace path with it; a path a viewer holds open (Windows: EINVAL /
    PermissionError) keeps the new plot beside it instead of failing the whole run."""
    fresh = path.with_name(f'{path.stem}_{time.strftime("%Y%m%d_%H%M%S")}{path.suffix}')
    try:
        figure.savefig(fresh, dpi=120, bbox_inches='tight')
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
    parser.add_argument('--only', nargs='*', default=None, help='run only commands containing one of these (always '
                                                                     'profiled, unchanged or not)')
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
    parser.add_argument('--reset', action='store_true',
                        help='profile every command, also those whose solver code and case are unchanged since their '
                             'last complete profile (needed after an edit outside the harness PROFILED_CODE that '
                             'changes timings: model plumbing, the harness itself)')
    parser.add_argument('--new-first', action='store_true',
                        help='this complete profile replaces the first one (commands that fail keep theirs; implies '
                             '--reset)')
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
    # named commands (--only) always profile, as a named selftest run reruns its cases
    reuse = () if options.reset or options.new_first or options.only else ('--if-changed',)
    log_path = OUT / 'run_all.log'
    log_path.parent.mkdir(parents=True, exist_ok=True)
    history_folder = OUT / 'history' / time.strftime('%Y-%m-%d_%H%M%S')
    history_folder.mkdir(parents=True, exist_ok=True)
    if POINTS_PATH.is_file():   # the points as they were before this run, to restore from
        shutil.copyfile(POINTS_PATH, history_folder / 'profile_points_before.json')
    checks_not_passed = []   # 'command [variant]: group line' of every failed / error solution check
    memory = load_memory() if reuse else {'commands': {}, 'snapshots': {}}
    # every command line (script and arguments: entry, backend, solver and reuse; the memory key from command[1:])
    command_lines = {(solver, backend, entry): [sys.executable, *command_line(entry), '--backend', backend,
                                                 *SOLVERS[solver]['arguments'], *reuse]
                     for solver, backend in variants for entry in commands}
    # the kept outcomes, decided before anything runs: memory first, the rest in one batch process
    standing = precheck({' '.join(command[1:]): command for command in command_lines.values()}, memory) if reuse else {}
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n===== {time.strftime("%Y-%m-%d %H:%M:%S")}: {len(commands)} commands x '
                  f'{[variant_label(*variant) for variant in variants]}, reports kept in {history_folder} =====\n')
        for solver, backend in variants:
            label = variant_label(solver, backend)
            for entry in commands:
                command = command_lines[(solver, backend, entry)]
                print(f'>>> {entry} [{label}]', flush=True)
                started = time.time()
                start = time.perf_counter()
                memory_key = ' '.join(command[1:])
                remembered = standing.get(memory_key)
                if remembered is not None:   # nothing it depends on changed: not started
                    lines, checking = remembered
                    return_code, stdout, stderr = 0, '\n'.join(lines) + '\n', ''
                else:
                    if reuse:   # taken before the run: an edit during it counts
                        snapshot, files, environment = source_snapshot()
                    try:
                        completed = subprocess.run(command, cwd=REPOSITORY, capture_output=True, text=True,
                                                   timeout=options.timeout or None)
                        return_code, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
                    except subprocess.TimeoutExpired as expired:   # run() has killed the command
                        return_code, stdout, stderr = None, _text(expired.stdout), _text(expired.stderr)
                    if reuse:   # saved at once, as the points: a later command that hangs loses nothing
                        record = memory_record(stdout, return_code, snapshot, environment)
                        if record is not None or memory['commands'].pop(memory_key, None) is not None:
                            if record is not None:
                                memory['commands'][memory_key] = record
                                memory['snapshots'][snapshot] = files
                            save_memory(memory)
                wall = time.perf_counter() - start
                if remembered is None:
                    checking = f'{wall:.1f} s check'
                checked, not_passed = check_summary(stdout)   # the profiled runs against their selftest checks
                checks_not_passed += [f'{entry} [{label}]: {text}' for text in not_passed]
                check_tail = ''.join(f'\n    solution check {text}' for text in not_passed)
                since = next((line[len(UNCHANGED):].partition(': ')[0] for line in stdout.splitlines()
                              if line.startswith(UNCHANGED)), None) if return_code == 0 else None
                if since is not None:   # the kept profile stands: its points stay, nothing copied to the history
                    earlier_point = points.get(label, {}).get('current', {}).get(entry)
                    kept = next((line[len(KEPT_REPORT):].strip() for line in stdout.splitlines()
                                 if line.startswith(KEPT_REPORT)), None)
                    runs_path = None if kept is None else Path(kept).parent / 'runs.json'
                    used = used_solvers(runs_path, 0.0) if solver == 'auto' and runs_path is not None else None
                    first = complete and entry not in points.get(label, {}).get('first', {})
                    if earlier_point is None and runs_path is not None:   # profiled outside run_all: its point now
                        kept_point, _ = command_point(runs_path, None, 0.0, SOLVERS[solver]['token'])
                        if kept_point is not None:
                            record_point(points, label, entry, kept_point, first=first)
                            save_points(points)
                    elif first and earlier_point is not None:   # unchanged code and case: it counts for this profile
                        points[label]['first'][entry] = earlier_point
                        save_points(points)
                    kept_wall = ('' if earlier_point is None or earlier_point.get('command_seconds') is None
                                 else f" (kept {earlier_point['command_seconds']:.1f} s)")
                    line = (f'{entry} [{variant_label(used, backend) if used else label}]: unchanged since {since}, '
                            f'not rerun{f" ({checking})" if checking else ""}{kept_wall}{checked}{check_tail}')
                    print(line, flush=True)
                    log.write(line + '\n')
                    log.flush()
                    continue
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
                line = f'{entry} [{shown}]: {reported}{failed_runs}{checked}{check_tail}{tail}'
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
        if checks_not_passed:   # repeated at the end: a profile of a wrong solution must not pass unseen
            line = (f'SOLUTION CHECK NOT PASSED ({len(checks_not_passed)}; the points are recorded):\n    '
                    + '\n    '.join(checks_not_passed))
            print(line, flush=True)
            log.write(line + '\n')
    print(f'log: {log_path}\npoints: {POINTS_PATH}\nplot: {plot_points(points, PLOT_PATH)}')


if __name__ == '__main__':
    main()
