"""Run every profiler_solver_*.py command in turn, once per chosen solver group (SOLVERS; no flag: auto, per
--backend; a failing one is logged, the rest continue) -> out/run_all.log; each report is also
kept in out/history/<timestamp>/ (with the points file as it was before the run). Each finished command updates its
point in out/profile_points.json and the plot out/profile_points.png (run wall, GPU peak and RAM peak per command; one
line per solver and backend for the first complete profile, the previous and the current one; a ring on the solver
Automatic picks on the gpu, worked out by the harness from each case's roles whichever solver was run). A log line
gives the command wall and, in brackets, that of the command's previous recorded point: '15.2 s (14.9 s)'.
A command stops after --timeout s (default 60; the harness's --time-limit: the report of what was timed until then is
kept, its log line says 'partial'; killed REPORT_GRACE s later if a library call holds it); one that was profiling is
plotted at the '>timeout' tick.
Run (from any folder): python run_all_profilers.py [--only SUBSTRING ...]
[--cudss --cudss-mixed --scipy --pardiso --mumps | --all]
[--backend gpu cpu] [--timeout S] [--new-first] [--plot-only]"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
OUT = HERE / 'out'
POINTS_PATH = OUT / 'profile_points.json'
PLOT_PATH = OUT / 'profile_points.png'
SERIES = ('first', 'previous', 'current')   # first: the first complete profile (all commands, no --only)
BACKENDS = ('gpu', 'cpu')
DEFAULT_TIMEOUT = 60.0   # seconds per command: the harness's --time-limit (it stops and writes a partial report)
REPORT_GRACE = 30.0      # seconds after the timeout before the command is killed (a long library call, the report)
NOTHING_PROFILED = 'nothing profiled'   # the harness's --compare-only line when no run takes the solver
PLANNED_GROUPS = 'solver groups: '   # the harness's line before the build: the groups the runs' roles take
PARTIAL_REPORT = 'partial report'       # the harness's PARTIAL_REPORT_MARKER line: stopped at its time limit
NEVER_FACTORISED = 'never factorised'   # log mark of a run whose compared solver did no factorisation (not counted)
TIMEOUT_CAP = 1.5        # a timed-out point is drawn at this many times the largest timeout shown


def _compared(value, backends):
    """Host library group: --compare back_end=value --compare-only, so only the runs a role of which takes it are
    profiled and counted (and only those in which it factorised: back_ends); backends: its default backend (an
    explicit --backend: any, a gpu model factorising on the host)."""
    return {'arguments': ('--compare', f'back_end={value}', '--compare-only'), 'token': f'back_end={value}',
            'backends': backends, 'runs_on': BACKENDS, 'back_ends': (value,)}


def _whole(value, backends, precision=None, runs_on=BACKENDS):
    """Array-library group: every run on its backend (at the precision), the roles taking back_end on value; all runs
    counted. backends: default; runs_on: those an explicit --backend may pick."""
    return {'arguments': (('--precision', precision) if precision else ()) + ('--compare', f'back_end={value}'),
            'token': None, 'backends': backends, 'runs_on': runs_on, 'back_ends': ()}


GPU, CPU = ('gpu',), ('cpu',)
# the five solver groups (harness solver_group: every solver is in one), one --<name> flag each, --all: the five; the
# order fixes each one's plot colour. cudss: cuDSS + cupy double (pcg, explicit kernels, ... at double precision);
# cudss_mixed: cuDSS mixed + cupy single; scipy: host SuperLU + numpy (the cpu backend); pardiso / mumps: their host
# library, cpu backend, only the runs a role of which takes it. No flag: auto, whose log line names the groups it used.
# backends: each one's default; an explicit --backend runs the host groups on it too (a gpu model factorising on the
# host), never cuDSS on the cpu
SOLVERS = {
    # the cases' solvers as written; run_all_examples.py adds --automatic-solvers (every role on Automatic)
    'auto': {'arguments': (), 'token': None, 'backends': GPU, 'runs_on': BACKENDS, 'back_ends': ()},
    'cudss': _whole('cudss', GPU, 'double', runs_on=GPU),
    'cudss_mixed': _whole('cudss_mixed', GPU, 'single', runs_on=GPU),
    'scipy': _whole('scipy_splu', CPU),
    'pardiso': _compared('pardiso', CPU),
    'mumps': _compared('mumps', CPU),
}
GROUPS = tuple(solver for solver in SOLVERS if solver != 'auto')
# {solver group: COMMANDS entries it never runs} (a suite may replace it): a variant whose answer is wrong is no
# timing point (its earlier points are dropped)
SOLVER_EXCLUDED_COMMANDS = {}
# variants of an older points file under the name that now covers them (load_points)
RENAMED_VARIANTS = {'scipy_splu @cpu': 'scipy @cpu', 'numpy @cpu': 'scipy @cpu'}
# solver names of older runs.json (no 'groups') by group
LEGACY_GROUPS = {'cudss': 'cudss', 'cudss_mixed': 'cudss_mixed', 'scipy_splu': 'scipy', 'superlu': 'scipy',
                 'pardiso': 'pardiso', 'pardiso_mixed': 'pardiso', 'mumps': 'mumps', 'mumps_mixed': 'mumps'}
# categorical palette slots in fixed order (8 hues)
SOLVER_COLORS = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
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


def command_point(runs_path, command_seconds, started, token, back_ends=()):
    """({run_seconds (sum of the counted runs' walls), command_seconds, gpu_peak_bytes / host_peak_bytes (largest over
    them), runs, run_walls / run_gpu_peaks / run_host_peaks ({run: value} of the counted runs, by base_run name: the
    plot's split columns), when} or None, names of the token's runs not counted) of one command; counted: the runs
    whose name holds token (all for None) that completed and in which a role taking the token's value factorised (a
    matrix-free eigen solve never does: that run measures another solver). None when runs.json is missing, older than
    the command or has no counted run; a run not counted is a task this solver did not do (its own plot column)."""
    if not runs_path.is_file() or runs_path.stat().st_mtime < started:
        return None, []
    named = {name: run for name, run in json.loads(runs_path.read_text(encoding='utf-8')).items()
             if token is None or token in name}
    values = set(back_ends)   # the back end names the compared value reports (none: every completed run counts)

    def idle(run):   # every role holding the compared value reports 0 factorisations (None: no counter, counted)
        taken = [choice for choice in run.get('solvers') or ()
                 if {choice.get('back_end'), choice.get('preconditioner')} & values]
        return bool(taken) and all(choice.get('factorizations') == 0 for choice in taken)
    completed = {name: run for name, run in named.items() if run.get('status') == 'completed'}
    counted = {name: run for name, run in completed.items() if not idle(run)}
    runs = list(counted.values())
    not_completed = ([f"{name} ({run.get('status')})" for name, run in named.items() if name not in completed]
                     + [f"{name} ({'/'.join(sorted(values))} {NEVER_FACTORISED})"
                        for name, run in completed.items() if idle(run)])
    if not runs:
        return None, not_completed

    def largest(key):
        values = [run[key] for run in runs if run.get(key) is not None]
        return max(values) if values else None
    def per_run(key):
        return {base_run(name): run.get(key) for name, run in counted.items()}
    # chosen: the groups the runs used; automatic: the groups Automatic picks on the gpu (ringed in the plot)
    return {'run_seconds': sum(run['wall_seconds'] for run in runs), 'command_seconds': command_seconds,
            'gpu_peak_bytes': largest('gpu_peak_bytes'), 'host_peak_bytes': largest('host_peak_bytes'),
            'runs': len(runs), 'run_walls': per_run('wall_seconds'), 'run_gpu_peaks': per_run('gpu_peak_bytes'),
            'run_host_peaks': per_run('host_peak_bytes'),
            'run_automatic': {base_run(name): run_groups([run], 'automatic_groups', 'automatic')
                              for name, run in counted.items()},
            'chosen': run_groups(runs, 'groups', 'solvers'),
            'automatic': run_groups(runs, 'automatic_groups', 'automatic'),
            'when': time.strftime('%Y-%m-%d %H:%M:%S')}, not_completed


def base_run(name) -> str:
    """Run name without the harness's ' SETTING=value' --compare suffix (a repeat's '_N' kept): the same task under
    every solver."""
    match = re.fullmatch(r'(.*?) \S+?=\S+?(_\d+)?', name)
    return match.group(1) + (match.group(2) or '') if match else name


def run_groups(runs, field, legacy_field) -> list:
    """Sorted solver groups of runs (the harness's runs.json `field`; an older one without it: the LEGACY_GROUPS of
    its `legacy_field` back ends)."""
    groups = set()
    for run in runs:
        if field in run:
            groups.update(run[field])
        else:
            groups.update(LEGACY_GROUPS[choice['back_end']] for choice in run.get(legacy_field) or ()
                          if choice.get('back_end') in LEGACY_GROUPS)
    return sorted(groups)


def used_solvers(runs_path, started) -> str | None:
    """'+'-joined solver groups the command's runs used, or None when runs.json is missing or older than the
    command."""
    if not runs_path.is_file() or runs_path.stat().st_mtime < started:
        return None
    runs = json.loads(runs_path.read_text(encoding='utf-8')).values()
    return '+'.join(run_groups(runs, 'groups', 'solvers')) or None


def timed_out_point(command_seconds, timeout):
    """Point of a command killed at the timeout: no run wall or memory, drawn at the timeout tick."""
    return {'run_seconds': None, 'timed_out': True, 'timeout_seconds': timeout, 'command_seconds': command_seconds,
            'gpu_peak_bytes': None, 'host_peak_bytes': None, 'runs': None, 'when': time.strftime('%Y-%m-%d %H:%M:%S')}


def previous_wall_text(point, prefix='') -> str:
    """' (14.9 s)': the command wall of the previous recorded point of a command (' (timed out)' for a stopped one),
    '' without one."""
    if point is None or point.get('command_seconds') is None:
        return ''
    return f' ({prefix}timed out)' if point.get('timed_out') else f" ({prefix}{point['command_seconds']:.1f} s)"


def run_command(command, timeout, on_line):
    """(return code, None when killed after `timeout` s; stdout; stderr) of `command` run from the repository;
    on_line sees each stdout line as it arrives (the harness flushes its lines)."""
    process = subprocess.Popen(command, cwd=REPOSITORY, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               errors='replace')
    stderr_lines = []
    stderr_reader = threading.Thread(target=lambda: stderr_lines.extend(process.stderr), daemon=True)
    stderr_reader.start()
    killed = threading.Event()

    def kill():
        killed.set()
        process.kill()
    timer = threading.Timer(timeout, kill) if timeout else None
    if timer is not None:
        timer.start()
    stdout_lines = []
    try:
        for line in process.stdout:
            stdout_lines.append(line)
            on_line(line)
        process.wait()
    finally:
        if timer is not None:
            timer.cancel()
        if process.poll() is None:   # run_all itself stopped (Ctrl+C): the command goes with it
            process.kill()
    stderr_reader.join()
    return (None if killed.is_set() else process.returncode), ''.join(stdout_lines), ''.join(stderr_lines)


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
    loaded = {}
    for variant, by_series in points.items():   # a renamed variant merges into its new name (the new name's wins)
        merged = loaded.setdefault(RENAMED_VARIANTS.get(variant, variant), {series: {} for series in SERIES})
        for series in SERIES:
            merged[series] = ({**by_series.get(series, {}), **merged[series]} if variant in RENAMED_VARIANTS
                              else {**merged[series], **by_series.get(series, {})})
    return loaded


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


def auto_picks(points, columns) -> dict:
    """{column index: variant} to ring, one per column: the groups Automatic picks on the gpu for the column's runs
    (any variant's current point); one group: its line, several (roles in different groups) or that line not run: the
    auto @gpu point, which ran that mix, when it used those groups."""
    auto = variant_label('auto', 'gpu')
    # one source per column: auto @gpu (Automatic itself), else a variant that left the model's precision alone (the
    # cudss / cudss_mixed runs' --precision changes what Automatic would pick)
    def precision_forced(variant):
        return '--precision' in SOLVERS[variant.partition(' @')[0]]['arguments']
    sources = [auto] + sorted((variant for variant in points if variant != auto),
                              key=lambda variant: (precision_forced(variant), _variant_order(variant)))
    def covers(point, runs, field='run_walls'):   # the point holds `field` for every run of the column
        return runs is None or all(run in (point.get(field) or {}) for run in runs)
    picks = {}
    for index, (entry, runs) in enumerate(columns):
        candidates = [points[variant]['current'][entry] for variant in sources
                      if variant in points and entry in points[variant]['current']
                      and not points[variant]['current'][entry].get('timed_out')]
        # a split column: first a point with per-run picks for its runs, then one that measured them (older points,
        # e.g. an auto @gpu point from before per-run fields, hold whole-command back end names only)
        candidates.sort(key=lambda point: (not covers(point, runs, 'run_automatic'), not covers(point, runs)))
        point = candidates[0] if candidates else {}
        per_run = point.get('run_automatic')
        names = ([name for run in runs for name in per_run[run]] if runs is not None and covers(point, runs,
                                                                                                'run_automatic')
                 else point.get('automatic') or ())
        groups = {LEGACY_GROUPS.get(name, name) for name in names}   # older points hold back end names
        if len(groups) == 1 and next(iter(groups)) in GROUPS:
            group = next(iter(groups))
            # gpu first (BACKENDS order): Automatic's pick on the gpu model, else the group's other line; only a
            # line that measured the column's runs can carry its ring
            labels = [variant_label(group, backend) for backend in SOLVERS[group]['runs_on']]
            target = next((label for label in labels if label in points and entry in points[label]['current']
                           and covers(points[label]['current'][entry], runs)), None)
            if target is not None:
                picks[index] = target
                continue
        auto_point = points.get(auto, {}).get('current', {}).get(entry)
        auto_used = {LEGACY_GROUPS.get(name, name) for name in (auto_point or {}).get('chosen') or ()}
        if auto_point is not None and groups <= auto_used and covers(auto_point, runs):
            picks[index] = auto
    return picks


def _variant_order(variant):
    solver, _, backend = variant.partition(' @')
    return list(SOLVERS).index(solver), backend


def plotted_variants(points) -> dict:
    """The points of the lines the plot draws: the auto pick and the five groups on each backend they run on; other
    variants of the points file (older solver flags) stay in it, not drawn."""
    labels = {variant_label(solver, backend) for solver, variant in SOLVERS.items() for backend in variant['runs_on']}
    return {variant: by_series for variant, by_series in points.items() if variant in labels}


RUN_FIELDS = {'run_seconds': 'run_walls', 'gpu_peak_bytes': 'run_gpu_peaks', 'host_peak_bytes': 'run_host_peaks'}


def plot_columns(points, entries) -> list:
    """[(command, runs)]: runs None for a command every solver measured on the same runs (current points), else one
    column per set of runs the same solvers measured (sorted run names), so a solver that takes only part of a
    command's tasks (or failed one) is compared with the others on that part alone. A point without per-run walls
    (older, timed out) does not split a command."""
    columns = []
    for entry in entries:
        measured_by = {}   # run -> the variants whose current point measured it
        for variant, by_series in points.items():
            for run in (by_series['current'].get(entry) or {}).get('run_walls') or ():
                measured_by.setdefault(run, set()).add(variant)
        tasks = {}
        for run, variants in measured_by.items():
            tasks.setdefault(frozenset(variants), []).append(run)
        if len(tasks) <= 1:
            columns.append((entry, None))
            continue
        columns += [(entry, tuple(sorted(runs))) for runs in sorted(tasks.values(), key=lambda runs: (-len(runs),
                                                                                                    sorted(runs)))]
    return columns


def plot_points(points, path):
    """Three panels sharing the column axis (plot_columns: a command, or one of its task sets; run wall s, log; GPU
    peak MiB; RAM peak MiB): one colour per solver, one marker per backend, one line style per series (first dotted,
    previous dashed, current solid), a ring on the solver Automatic picks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    points = plotted_variants(points)
    measured = {entry for by_series in points.values() for series in SERIES for entry in by_series[series]}
    entries = [entry for entry in COMMANDS if entry in measured]   # a command dropped from COMMANDS is not drawn
    if not entries:
        return None
    variants = sorted(points, key=_variant_order)
    columns = plot_columns(points, entries)
    picks = auto_picks(points, columns)
    labels = [entry if runs is None else f'{entry}: {"+".join(runs)}' for entry, runs in columns]
    unringed = [label for index, label in enumerate(labels) if index not in picks]
    if unringed:   # mostly an auto @gpu point from before per-run fields: a no-flag run gives the ring
        print('no Automatic ring (no point holds its per-run pick on a line that measured it; rerun with no solver '
              f'flag, unless Automatic fails it): {", ".join(unringed)}')

    def color(variant):
        return SOLVER_COLORS[list(SOLVERS).index(variant.partition(' @')[0]) % len(SOLVER_COLORS)]
    panels = (('run_seconds', 1.0, 'run wall (s)', True), ('gpu_peak_bytes', 2.0 ** -20, 'GPU peak (MiB)', False),
              ('host_peak_bytes', 2.0 ** -20, 'RAM peak (MiB)', False))
    timeouts = [point['timeout_seconds'] for by_series in points.values() for series in SERIES
                for entry, point in by_series[series].items() if point.get('timed_out') and entry in entries]
    largest_timeout = max(timeouts, default=None)
    cap = None if largest_timeout is None else TIMEOUT_CAP * largest_timeout   # height of every timed-out point

    def value(point, key, variant, runs):
        if point is None or (key == 'gpu_peak_bytes' and variant.endswith(' @cpu')):   # cpu: VRAM not measured
            return None
        if point.get('timed_out'):
            return cap if key == 'run_seconds' else None
        if runs is None:
            return point.get(key)
        per_run = point.get(RUN_FIELDS[key]) or {}   # a split column: only a point that measured all its runs
        found = [per_run.get(run) for run in runs]
        if any(item is None for item in found):
            return None
        return sum(found) if key == 'run_seconds' else max(found)
    figure, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(max(11.0, 0.3 * len(columns)), 11.0))
    positions = list(range(len(columns)))
    for axis, (key, scale, label, logarithmic) in zip(axes, panels):
        for variant in variants:
            marker = BACKEND_MARKERS.get(variant.partition(' @')[2], 'o')
            for series in SERIES:
                values = [value(points[variant][series].get(entry), key, variant, runs) for entry, runs in columns]
                if all(item is None for item in values):
                    continue
                heights = [math.nan if item is None else item * scale for item in values]   # gaps break the line
                axis.plot(positions, heights, color=color(variant), marker=marker, **SERIES_STYLES[series])
        for position, variant in picks.items():   # one ring per column, on the current point of Automatic's pick
            entry, runs = columns[position]
            height = value(points[variant]['current'].get(entry), key, variant, runs)
            if height is not None:
                axis.scatter([position], [height * scale], **PICK_RING)
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
    axes[-1].set_xticklabels(labels, rotation=90, fontsize=8)
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
    parser.add_argument('--only', nargs='*', default=None, help='run only commands containing one of these')
    for solver, variant in SOLVERS.items():
        parser.add_argument(f'--{solver.replace("_", "-")}', dest=f'solver_{solver}', action='store_true',
                            help=f'profile with {" ".join(variant["arguments"]) or "the cases as written"}'
                                 + f' (default backend {variant["backends"][0]})')
    parser.add_argument('--all', action='store_true', help=f'the five solver groups in turn: {", ".join(GROUPS)}')
    parser.add_argument('--backend', nargs='+', default=None, choices=BACKENDS,
                        help='backends to run on (default: each flag its own, auto gpu, the host groups cpu); cudss '
                             '/ cudss_mixed run on the gpu only (skipped for a cpu-only --backend)')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f'seconds per command before it stops with a partial report (default '
                             f'{DEFAULT_TIMEOUT:g}; 0: no limit; killed {REPORT_GRACE:g} s later); a stopped command '
                             'that was profiling is plotted at the timeout tick')
    parser.add_argument('--new-first', action='store_true',
                        help='this complete profile replaces the first one (commands that fail keep theirs)')
    parser.add_argument('--plot-only', action='store_true', help='redraw the plot from profile_points.json, run nothing')
    options = parser.parse_args()
    points = load_points()
    if options.plot_only:
        print(f'plot: {plot_points(points, PLOT_PATH)}')
        return
    solvers = (list(GROUPS) if options.all
               else [solver for solver in SOLVERS if getattr(options, f'solver_{solver}')] or ['auto'])
    variants = [(solver, backend) for solver in solvers
                for backend in (SOLVERS[solver]['backends'] if options.backend is None else
                                [backend for backend in options.backend if backend in SOLVERS[solver]['runs_on']])]
    skipped = [solver for solver in solvers if solver not in {solver for solver, _ in variants}]
    if skipped:
        print(f'--backend {" ".join(options.backend)}: {", ".join(skipped)} run on the gpu only; skipped', flush=True)
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
                if entry in SOLVER_EXCLUDED_COMMANDS.get(solver, ()):
                    print(f'>>> {entry} [{label}]: excluded (SOLVER_EXCLUDED_COMMANDS)', flush=True)
                    if [series for series in SERIES if points.get(label, {}).get(series, {}).pop(entry, None)]:
                        save_points(points)
                    continue
                command = [sys.executable, *command_line(entry), '--backend', backend, *SOLVERS[solver]['arguments'],
                           *(('--time-limit', f'{options.timeout:g}') if options.timeout else ())]
                if solver != 'auto':   # auto: '>>>' once the harness names the groups its runs take
                    print(f'>>> {entry} [{label}]', flush=True)

                def announce(line, entry=entry, backend=backend, solver=solver):
                    if solver == 'auto' and line.startswith(PLANNED_GROUPS):
                        groups = line[len(PLANNED_GROUPS):].strip() or solver
                        print(f'>>> {entry} [{variant_label(groups, backend)}]', flush=True)
                started = time.time()
                start = time.perf_counter()
                return_code, stdout, stderr = run_command(
                    command, options.timeout + REPORT_GRACE if options.timeout else None, announce)
                wall = time.perf_counter() - start
                report = [line for line in stdout.splitlines() if line.startswith('report: ')]
                partial = return_code is not None and any(line.startswith(PARTIAL_REPORT)
                                                          for line in stdout.splitlines())
                point, not_completed, shown = None, [], label   # shown: auto replaced by the solvers it used
                untaken = return_code == 0 and NOTHING_PROFILED in stdout
                earlier_point = points.get(label, {}).get('current', {}).get(entry)   # read before it is replaced
                if return_code is None or partial:
                    if NOTHING_PROFILED not in stdout:   # it was profiling this solver when stopped
                        point = timed_out_point(wall, options.timeout)
                if return_code is not None and report:
                    report_path = Path(report[-1][len('report: '):].strip())
                    if report_path.is_file():   # the folder name holds the backend and --compare settings
                        shutil.copyfile(report_path, history_folder / f'{report_path.parent.name}.txt')
                    runs_path = report_path.parent / 'runs.json'   # per-part seconds: what profile_delta.py compares
                    if runs_path.is_file() and runs_path.stat().st_mtime >= started:
                        shutil.copyfile(runs_path, history_folder / f'{report_path.parent.name}.runs.json')
                    used = used_solvers(runs_path, started) if solver == 'auto' else None
                    if used:
                        shown = variant_label(used, backend)
                    if return_code == 0 and not partial:
                        point, not_completed = command_point(report_path.parent / 'runs.json', wall, started,
                                                             SOLVERS[solver]['token'], SOLVERS[solver]['back_ends'])
                if point is not None:   # saved at once: a later command that hangs or crashes loses nothing
                    first = complete and (options.new_first or entry not in points.get(label, {}).get('first', {}))
                    record_point(points, label, entry, point, first=first)
                    save_points(points)
                elif untaken or (not_completed and all(NEVER_FACTORISED in text for text in not_completed)):
                    # no run takes this solver (found from the specs, before any build) or every run of it measured
                    # another one: its earlier points of the command go too
                    if [series for series in SERIES if points.get(label, {}).get(series, {}).pop(entry, None)]:
                        save_points(points)
                tail ='' if return_code in (0, None) else '\n    ' + '\n    '.join(stderr.splitlines()[-8:])
                if return_code is None or partial:
                    reported = (f'timed out after {options.timeout:g} s'
                                + previous_wall_text(earlier_point, prefix='previous ')
                                + (' (plotted at the timeout tick)' if point is not None else '')
                                + (f', partial {report[-1]}' if partial and report else
                                   ', killed: no report' if return_code is None else ''))
                elif untaken:   # the harness found no role taking the value in any spec and built nothing
                    reported = f'skipped, {wall:.1f} s: no run takes this solver'
                else:   # pass: the script exited 0 and every run of the solver completed
                    run_failed = any(NEVER_FACTORISED not in text for text in not_completed)
                    verdict = (f'FAIL (exit {return_code})' if return_code else
                               'FAIL (no report)' if not report else
                               'FAIL (a run did not complete: the plot splits it off as a task this solver did not do)'
                               if run_failed else 'pass')
                    reported = (f'{verdict}, {wall:.1f} s{previous_wall_text(earlier_point)}'
                                + (f', {report[-1]}' if report else ''))
                failed_runs = f', not counted: {", ".join(not_completed)}' if not_completed else ''
                line = f'{entry} [{shown}]: {reported}{failed_runs}{tail}'
                print(line, flush=True)
                log.write(line + '\n')
                log.flush()
    print(f'log: {log_path}\npoints: {POINTS_PATH}\nplot: {plot_points(points, PLOT_PATH)}')


if __name__ == '__main__':
    main()
