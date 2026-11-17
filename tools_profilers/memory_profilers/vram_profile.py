"""VRAM profiler (dev tooling, not part of open_sol): runs named selftest cases in-process under a cupy pool hook and
cuDSS phase wrappers; writes out/<case><tag>.md + .json (estimate parts vs measured, sites at each peak).
Run (from any folder, gpu): python vram_profile.py em_bor_cudss_driven [more case names] [--tag _x --depth 3]"""
import argparse
import functools
import gc
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # memory_profilers -> tools_profilers -> repository
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(__file__).resolve().parent / 'out'
SNAPSHOT_SLACK_BYTES = 64 * 1024   # a stage peak re-snapshots once it grew this much past its last snapshot
TRIGGER_DEPTH = 8                  # open_sol frames of the allocation that reached a peak

import cupy as cp                                          # noqa: E402
from cupy.cuda import memory_hook                          # noqa: E402
from nvmath.sparse.advanced import DirectSolver            # noqa: E402
from open_sol import api                                   # noqa: E402
from open_sol.model import study as study_module           # noqa: E402
from open_sol.model.builder import estimate_gap, estimate_verdict   # noqa: E402
from open_sol.solvers import cudss_factor                  # noqa: E402

PACKAGE = os.path.normcase(str(ROOT / 'open_sol')) + os.sep
SOLVER_FILES = ('solvers/cudss_factor.py', 'solvers/sparse_direct.py')
_PACKAGE_PATHS = {}   # code file name -> path inside open_sol ('/' separated) or None


def _package_path(code_filename):
    """Path of a code file relative to open_sol, None outside the package (cached per file name)."""
    relative = _PACKAGE_PATHS.get(code_filename, False)
    if relative is False:   # normcase keeps the length on Windows, so the slice keeps the file's own case
        inside = os.path.normcase(code_filename).startswith(PACKAGE)
        relative = code_filename[len(PACKAGE):].replace('\\', '/') if inside else None
        _PACKAGE_PATHS[code_filename] = relative
    return relative


def frames_text(frame, depth, skip_solvers=False):
    """'file:line function < ...' of the first `depth` open_sol frames from `frame` outwards."""
    parts = []
    while frame is not None and len(parts) < depth:
        relative = _package_path(frame.f_code.co_filename)
        if relative is not None and not (skip_solvers and relative.endswith(SOLVER_FILES)):
            parts.append(f'{relative}:{frame.f_lineno} {frame.f_code.co_name}')
        frame = frame.f_back
    return ' < '.join(parts) or '(outside open_sol)'


def mb(n):
    return 'n/a' if n is None else f'{n / 2 ** 20:.3f}'


class PoolTrace(memory_hook.MemoryHook):
    """Live pool allocations by site (used bytes, 512 B rounded), a by-site snapshot at each stage's peak (taken at the
    first free after it, so exact within SNAPSHOT_SLACK_BYTES) and nested region marks (start, peak, end)."""
    name = 'PoolTrace'

    def __init__(self, site_depth=3):
        self.site_depth = site_depth
        self.live = {}       # pointer -> (bytes, site key, run index)
        self.current = 0     # pool used bytes seen by the hook
        self.run_index = 0
        self.stage = 'other'
        self.marks = []      # open regions: [start, peak]
        self.start_run()

    def start_run(self):
        self.run_index += 1
        self.earlier = self.current        # arrays of earlier runs still live (decreases as they are freed)
        self.site_live = defaultdict(int)  # '[stage] site' -> live bytes of this run
        self.stage_peaks = {}              # stage -> {'bytes', 'snapshot_bytes', 'snapshot'}
        self.pending = None                # (stage, trigger) of a stage peak not snapshotted yet
        self.regions = defaultdict(lambda: {'calls': 0, 'max_peak': 0, 'max_net': 0, 'sum_net': 0})

    def run_bytes(self):
        return self.current - self.earlier

    def malloc_postprocess(self, **kwargs):
        size = int(kwargs['mem_size'])
        frame = sys._getframe(1)
        key = f'[{self.stage}] {frames_text(frame, self.site_depth)}'
        self.live[kwargs['mem_ptr']] = (size, key, self.run_index)
        self.site_live[key] += size
        self.current += size
        if self.marks and self.current > self.marks[-1][1]:
            self.marks[-1][1] = self.current
        peak = self.stage_peaks.get(self.stage)
        if peak is None:
            peak = self.stage_peaks[self.stage] = {'bytes': 0, 'snapshot_bytes': -SNAPSHOT_SLACK_BYTES, 'snapshot': None}
        run_bytes = self.run_bytes()
        if run_bytes > peak['bytes']:
            peak['bytes'] = run_bytes
            if self.pending is not None or run_bytes >= peak['snapshot_bytes'] + SNAPSHOT_SLACK_BYTES:
                self.pending = (self.stage, frames_text(frame, TRIGGER_DEPTH))

    def free_postprocess(self, **kwargs):
        entry = self.live.pop(kwargs.get('mem_ptr'), None)
        if entry is None:   # allocated before the hook was installed
            return
        self.flush()        # the bytes live right before this free are the peak's
        size, key, run_index = entry
        self.current -= size
        if run_index != self.run_index:
            self.earlier -= size
        elif self.site_live[key] == size:
            del self.site_live[key]
        else:
            self.site_live[key] -= size

    def flush(self):
        """Snapshot a pending stage peak (on a free, a stage change and the run's end)."""
        if self.pending is None:
            return
        stage, trigger = self.pending
        self.pending = None
        peak = self.stage_peaks[stage]
        peak['snapshot_bytes'] = self.run_bytes()
        peak['snapshot'] = {'bytes': self.run_bytes(), 'earlier_runs': self.earlier, 'trigger': trigger,
                            'by_site': dict(self.site_live)}

    def set_stage(self, stage):
        self.flush()
        previous, self.stage = self.stage, stage
        return previous

    def enter(self):
        self.marks.append([self.current, self.current])

    def leave(self):
        start, peak = self.marks.pop()
        if self.marks:
            self.marks[-1][1] = max(self.marks[-1][1], peak)
        return start, peak, self.current


TRACE = PoolTrace()
PLANS, BUILDS, RUNS = [], [], []
CUDSS = {'live': 0, 'high_water': 0}   # measured cuDSS bytes outside the pool: summed phase deltas of live solvers


def outside_pool_bytes():
    """Device bytes in use outside the cupy pool: memGetInfo used minus the pool's reserved bytes (synchronised)."""
    cp.cuda.Device().synchronize()
    free, total = cp.cuda.runtime.memGetInfo()
    return (total - free) - cp.get_default_memory_pool().total_bytes()


def region(label, function):
    """function wrapped in a pool region mark."""
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        TRACE.enter()
        try:
            return function(*args, **kwargs)
        finally:
            start, peak, end = TRACE.leave()
            stats = TRACE.regions[label]
            stats['calls'] += 1
            stats['max_peak'] = max(stats['max_peak'], peak - start)
            stats['max_net'] = max(stats['max_net'], end - start)
            stats['sum_net'] += end - start
    return wrapped


# --- nvmath DirectSolver phases (no cupy pool traffic inside them, so the outside-pool delta is cuDSS's own) ----------
NVMATH_PHASES = ('__init__', 'plan', 'factorize', 'solve', 'free')
SOLVER_PHASES = {}   # id(DirectSolver) -> phase record (not on the solver: its free() clears its attributes)
BOOK_ERRORS = []     # profiler failures inside a wrapped call: recorded, never raised into the run


def _new_phases():
    return {'init': 0, 'plan': 0, 'factorize_first': None, 'factorize': 0, 'factorize_calls': 0, 'solve_first': None,
            'solve': 0, 'solve_calls': 0, 'freed': None, 'held': 0}


def _book(solver, phase, before):
    """Add the outside-pool delta since `before` to the solver's phase record and the live cuDSS counter."""
    try:
        delta = outside_pool_bytes() - before
    except Exception:   # free from __del__ at interpreter shutdown: the CUDA context may be gone
        return
    name = 'init' if phase == '__init__' else phase
    if name == 'init':   # a fresh solver (an id reused after an earlier solver's free gets a new record)
        SOLVER_PHASES[id(solver)] = _new_phases()
    phases = SOLVER_PHASES.setdefault(id(solver), _new_phases())
    if name == 'free':
        phases['freed'] = (phases['freed'] or 0) - delta
    else:
        phases[name] += delta
        if name in ('factorize', 'solve'):
            phases[f'{name}_calls'] += 1
            if phases[f'{name}_first'] is None:
                phases[f'{name}_first'] = delta
    phases['held'] += delta
    CUDSS['live'] += delta
    CUDSS['high_water'] = max(CUDSS['high_water'], CUDSS['live'])


def _phase_wrapper(phase, function):
    @functools.wraps(function)
    def wrapped(self, *args, **kwargs):
        before = outside_pool_bytes()
        try:
            return function(self, *args, **kwargs)
        finally:
            try:
                _book(self, phase, before)
            except Exception as error:   # a profiler bug must not change the run's outcome
                BOOK_ERRORS.append(f'{phase}: {type(error).__name__}: {error}')
    return wrapped


for _phase in NVMATH_PHASES:
    setattr(DirectSolver, _phase, _phase_wrapper(_phase, getattr(DirectSolver, _phase)))


# --- open_sol cuDSS wrappers ------------------------------------------------------------------------------------------
Factorization = cudss_factor.CudssFactorization
CUDSS_REGIONS = ('refactorize', 'refactorize_device_values', 'refactorize_device_lower', 'solve', '_device_index_maps',
                 '_device_asymmetry', '_plan', '_device_plan')
_missing = [name for name in ('_plan_solver', '_factorize_and_probe', *CUDSS_REGIONS) if not hasattr(Factorization, name)]
if _missing:
    raise AttributeError(f'vram_profile: CudssFactorization has no {", ".join(_missing)} (renamed? update the profiler '
                         'and its sub_structure_conventions.md)')
original_plan_solver = Factorization._plan_solver
original_factorize = Factorization._factorize_and_probe


def plan_solver(self, kind, started):
    original_plan_solver(self, kind, started)
    pattern = self._pattern
    plan = {'run': TRACE.run_index, 'caller': frames_text(sys._getframe(1), 4, skip_solvers=True),
            'n': self.n_unknowns, 'stored_entries': int(self._factor_matrix.nnz),
            'full_entries': None if pattern is None else int(pattern[1].size),
            'value_dtype': str(cp.dtype(self.value_dtype)), 'factor_dtype': str(cp.dtype(self._factor_dtype)),
            'kind': kind, 'ordering_reused': bool(self.last_ordering_reused),
            'cudss_permanent': self.permanent_bytes, 'cudss_peak': self.peak_bytes,
            'phases': SOLVER_PHASES.get(id(self._solver)), 'lu_entries': None, 'factorizations': 0}
    self._vram_plan = plan
    PLANS.append(plan)


def factorize_and_probe(self, defer=False):
    try:
        return original_factorize(self, defer)
    finally:
        plan = getattr(self, '_vram_plan', None)
        if plan is not None:
            if plan['run'] != TRACE.run_index and not any(entry is plan for entry in PLANS):
                PLANS.append(plan)   # planned in an earlier run, refactorised in this one
            plan['factorizations'] += 1
            plan['lu_entries'] = self.lu_entries


Factorization._plan_solver = plan_solver
Factorization._factorize_and_probe = factorize_and_probe
for _method in CUDSS_REGIONS:
    setattr(Factorization, _method, region(f'cudss {_method}', getattr(Factorization, _method)))


# --- build / study wrappers -------------------------------------------------------------------------------------------
original_build_session = study_module.build_session
original_run_built = study_module._run_built


def build_session(*args, **kwargs):
    previous = TRACE.set_stage('build')
    TRACE.enter()
    outcome = (None, None)
    try:
        outcome = original_build_session(*args, **kwargs)
        return outcome
    finally:
        start, peak, end = TRACE.leave()
        TRACE.set_stage(previous)
        BUILDS.append({'built': outcome[0] is not None, 'pool_used_at_start': start, 'peak_above_start': peak - start,
                       'kept': end - start, 'studies': []})


def run_built(*args, **kwargs):
    previous = TRACE.set_stage('study')
    TRACE.enter()
    try:
        return original_run_built(*args, **kwargs)
    finally:
        start, peak, end = TRACE.leave()
        TRACE.set_stage(previous)
        if not BUILDS:
            BUILDS.append({'built': True, 'pool_used_at_start': start, 'peak_above_start': 0, 'kept': 0, 'studies': []})
        build = BUILDS[-1]
        build['studies'].append({'start_above_build': start - build['pool_used_at_start'],
                                 'peak_above_start': peak - start, 'net': end - start})


# every loaded module holding them by name (a module imported later reads the patched study attributes)
for _module in list(sys.modules.values()):
    if getattr(_module, '__name__', '').startswith('open_sol'):
        if getattr(_module, 'build_session', None) is original_build_session:
            _module.build_session = build_session
        if getattr(_module, '_run_built', None) is original_run_built:
            _module._run_built = run_built


# --- selftest run wrapper ---------------------------------------------------------------------------------------------
original_selftest_run = api._selftest_run
original_registered_case = api._registered_case
STARTED = {}


def _begin_run():
    PLANS.clear()
    BUILDS.clear()
    TRACE.start_run()
    STARTED.update(seconds=time.perf_counter(), cudss_live=CUDSS['live'])
    CUDSS['high_water'] = CUDSS['live']


def _end_run(name, status, summary, vram):
    TRACE.flush()
    build = summary.get('build') or {}
    cudss_at_start = STARTED['cudss_live']
    RUNS.append({'name': name, 'seconds': time.perf_counter() - STARTED['seconds'], 'status': status,
                 'pool_earlier_runs': TRACE.earlier, 'pool_end': TRACE.run_bytes(),
                 'stage_peaks': {stage: dict(peak) for stage, peak in TRACE.stage_peaks.items()},
                 'regions': {label: dict(stats) for label, stats in TRACE.regions.items()},
                 'builds': [dict(entry) for entry in BUILDS],
                 'plans': list(PLANS),   # the same dicts: frees after this run (Results closed later) still land
                 'cudss_live_at_start': cudss_at_start, 'cudss_high_water': CUDSS['high_water'] - cudss_at_start,
                 'estimate': build.get('gpu_estimate'), 'ledger_peak': build.get('gpu_peak_bytes'),
                 'ledger_kept': sum(size for _, size in build['gpu_breakdown']) if build.get('gpu_breakdown') else None,
                 'run_memory': summary.get('run_memory'), 'external_memory': summary.get('external_memory'),
                 'open_sol_vram': vram})


def selftest_run(spec, run_dir, max_wall_time):
    _begin_run()
    results, record = original_selftest_run(spec, run_dir, max_wall_time)
    summary = {}
    if run_dir is not None and (Path(run_dir) / 'summary.json').exists():
        summary = json.loads((Path(run_dir) / 'summary.json').read_text())
    try:
        run_name = Path(run_dir).relative_to(OUT_DIR / 'runs').as_posix()
    except (TypeError, ValueError):
        run_name = str(run_dir)
    _end_run(run_name, record.get('status'), summary, record.get('vram'))
    return results, record


def registered_case(case, *args, **kwargs):
    """A case without model runs (specs {}: its check builds the solvers itself) is profiled as one 'direct' run."""
    runs_before = len(RUNS)
    _begin_run()
    record = original_registered_case(case, *args, **kwargs)
    if len(RUNS) == runs_before:
        _end_run(f'{case.name} (direct check, no model run)', (record or {}).get('status', 'skipped'), {}, None)
    return record


api._selftest_run = selftest_run
api._registered_case = registered_case


# --- report -----------------------------------------------------------------------------------------------------------
def measured_parts(run):
    """{part: (profiler measured bytes, what it is)} over every build / study of the run (max: sweeps build per value)."""
    builds = [entry for entry in run['builds'] if entry['built']]
    studies = [study for entry in builds for study in entry['studies']]
    return {
        'kept': (max((entry['kept'] for entry in builds), default=None), 'pool used at build end - at build start'),
        'build_peak': (max((entry['peak_above_start'] for entry in builds), default=None),
                       'pool used peak during build - at build start'),
        'run': (max((study['start_above_build'] + study['peak_above_start'] for study in studies), default=None),
                'pool used peak during study - at build start'),
        'external': ((run['external_memory'] or {}).get('measured_bytes'),
                     "cuDSS's own plan estimates, live high water (open_sol counter)"),
        'external_device': (run['cudss_high_water'] if run['plans'] else None,
                            'device bytes of the nvmath DirectSolver calls (memGetInfo deltas, 2 MiB steps on WDDM), '
                            'live high water')}


def estimated_parts(estimate):
    if not estimate or estimate.get('gpu_steady') is None:
        return {}
    return {'kept': estimate['gpu_steady'], 'build_peak': estimate.get('gpu_build_peak'),
            'run': estimate['gpu_steady'] + (estimate.get('gpu_run') or 0),
            'external': estimate.get('external_peak'), 'external_device': estimate.get('external_peak')}


def verdict_rows(run):
    """[part, estimate, measured, gap, verdict, open_sol measured, open_sol verdict, measured as] per estimate part."""
    estimates, own = estimated_parts(run['estimate']), run['open_sol_vram'] or {}
    rows = []
    for part, (measured, meaning) in measured_parts(run).items():
        estimated = estimates.get(part)
        judged = measured is not None and estimated is not None
        gap = estimate_gap(measured, estimated) if judged else None
        verdict = (estimate_verdict(measured, estimated) or 'ok') if judged else 'n/a'
        own_part = own.get('external' if part.startswith('external') else part)
        own_part = own_part if isinstance(own_part, dict) else {}
        rows.append([part, estimated, measured, gap, verdict, own_part.get('measured_bytes'),
                     (own_part.get('verdict') or 'ok') if own_part else 'n/a', meaning])
    return rows


def verdict_table(run):
    lines = ['| part | estimate MB | measured MB | held - estimate MB | verdict | open_sol measured MB | open_sol verdict '
             '| measured as |', '|---|---|---|---|---|---|---|---|']
    for part, estimated, measured, gap, verdict, own_measured, own_verdict, meaning in verdict_rows(run):
        lines.append(f'| {part} | {mb(estimated)} | {mb(measured)} | {mb(gap)} | {verdict} | {mb(own_measured)} | '
                     f'{own_verdict} | {meaning} |')
    return lines


def _items_text(rows, phased=False):
    lines = []
    for row in rows or ():
        lines.append(f'  - [{row[0]}] {row[1]}: {mb(row[2])} MB' if phased else f'  - {row[0]}: {mb(row[1])} MB')
    return lines or ['  - (none)']


def _fill_columns(plan):
    """lu entries per n log2 n and per n^(4/3), cuDSS permanent bytes per lu entry."""
    n, lu = plan['n'], plan['lu_entries']
    if not lu or n < 2:
        return 'n/a | n/a | n/a'
    return (f"{lu / (n * math.log2(n)):.3f} | {lu / n ** (4.0 / 3.0):.3f} | "
            f"{(plan['cudss_permanent'] or 0) / lu:.1f}")


def _phase_columns(plan):
    """init | plan | first factorize | later factorizes | first solve | later solves | freed | held MB."""
    phases = plan['phases']
    if phases is None:
        return ' | '.join(['n/a'] * 8)
    later_factorize = phases['factorize'] - (phases['factorize_first'] or 0)
    later_solve = phases['solve'] - (phases['solve_first'] or 0)
    return ' | '.join([mb(phases['init']), mb(phases['plan']), mb(phases['factorize_first']), mb(later_factorize),
                       mb(phases['solve_first']), mb(later_solve), mb(phases['freed']), mb(phases['held'])])


def run_text(run, top_sites):
    estimate = run['estimate'] or {}
    lines = [f"## {run['name']}  ({run['status']}, {run['seconds']:.1f} s)", '',
             '### Estimate parts vs measured (the estimate should sit just below the held bytes)', '',
             *verdict_table(run), '',
             "- pool: used bytes from the hook (512 B rounded, no cached blocks); open_sol's run part is reserved-based "
             "(pool reserved high water - free blocks stranded before the build)",
             f"- open_sol build ledger: kept {mb(run['ledger_kept'])} MB, peak {mb(run['ledger_peak'])} MB",
             f"- arrays of earlier runs live at the end: {mb(run['pool_earlier_runs'])} MB; this run's arrays still "
             f"held when it returned: {mb(run['pool_end'])} MB; measured cuDSS bytes of earlier runs at the start: "
             f"{mb(run['cudss_live_at_start'])} MB", '']
    if len(run['builds']) > 1 or any(len(entry['studies']) > 1 for entry in run['builds']):
        lines += ['### Builds and studies (MB)', '',
                  '| build | built | peak above start | kept | study: start above build | study peak above its start | '
                  'study net |', '|---|---|---|---|---|---|---|']
        for index, entry in enumerate(run['builds'], 1):
            for study in entry['studies'] or [None]:
                study_text = ('| | |' if study is None else
                              f"{mb(study['start_above_build'])} | {mb(study['peak_above_start'])} | {mb(study['net'])} |")
                lines.append(f"| {index} | {entry['built']} | {mb(entry['peak_above_start'])} | {mb(entry['kept'])} | "
                             f"{study_text}")
        lines.append('')
    lines += ['### Estimate',
              f"- gpu_steady {mb(estimate.get('gpu_steady'))}, gpu_build_peak {mb(estimate.get('gpu_build_peak'))}, "
              f"gpu_run {mb(estimate.get('gpu_run'))}, gpu_external {mb(estimate.get('gpu_external'))}, "
              f"external_extra {mb(estimate.get('external_extra'))}, external_peak {mb(estimate.get('external_peak'))} MB",
              '- kept items:', *_items_text(estimate.get('items')),
              '- run items:', *_items_text(estimate.get('run_items')),
              '- transient items (phase, label):', *_items_text(estimate.get('transient_items'), phased=True),
              '- external items:', *_items_text(estimate.get('external_items')), '']
    if run['plans']:
        lines += ['### cuDSS plans (device MB per nvmath DirectSolver call: memGetInfo minus pool, synchronised; freed = '
                  'returned at free, held = left after every booked call)', '',
                  '| caller | n | stored nnz | full nnz | value / factor | kind | ordering reused | cuDSS permanent | '
                  'cuDSS peak | init | plan | factorize 1st | factorize later | solve 1st | solve later | freed | held | '
                  'lu_nnz | lu / n log2 n | lu / n^(4/3) | B / lu entry | factorizations |', '|' + '---|' * 22]
        for plan in run['plans']:
            lines.append(f"| {plan['caller']} | {plan['n']} | {plan['stored_entries']} | {plan['full_entries']} | "
                         f"{plan['value_dtype']} / {plan['factor_dtype']} | {plan['kind']} | {plan['ordering_reused']} | "
                         f"{mb(plan['cudss_permanent'])} | {mb(plan['cudss_peak'])} | {_phase_columns(plan)} | "
                         f"{plan['lu_entries']} | {_fill_columns(plan)} | {plan['factorizations']} |")
        lines.append('')
    if run['regions']:
        lines += ['### Pool regions (peak / net over the call, MB)', '',
                  '| region | calls | max peak | max net | summed net |', '|---|---|---|---|---|']
        for label, stats in sorted(run['regions'].items()):
            lines.append(f"| {label} | {stats['calls']} | {mb(stats['max_peak'])} | {mb(stats['max_net'])} | "
                         f"{mb(stats['sum_net'])} |")
        lines.append('')
    for stage, peak in run['stage_peaks'].items():
        snapshot = peak['snapshot']
        if not snapshot:
            continue
        rows = sorted(snapshot['by_site'].items(), key=lambda item: -item[1])
        lines += [f"### Live pool bytes at the {stage} peak ({mb(snapshot['bytes'])} MB of this run, exact peak "
                  f"{mb(peak['bytes'])} MB; earlier runs' arrays {mb(snapshot['earlier_runs'])} MB)", '',
                  f"Peak allocation: `{snapshot['trigger']}`", '', '| MB | [stage] site |', '|---|---|']
        for site, size in rows[:top_sites]:
            lines.append(f'| {mb(size)} | `{site}` |')
        rest = sum(size for _, size in rows[top_sites:])
        if rest:
            lines.append(f'| {mb(rest)} | ({len(rows) - top_sites} more sites) |')
        lines.append('')
    return '\n'.join(lines)


def console_text(run):
    lines = [f"{run['name']} ({run['status']}):"]
    for part, estimated, measured, gap, verdict, _, own_verdict, _ in verdict_rows(run):
        lines.append(f'  {part:16s} est {mb(estimated):>10s}  meas {mb(measured):>10s}  gap {mb(gap):>10s} MB  '
                     f'{verdict} (open_sol: {own_verdict})')
    for plan in run['plans']:
        lines.append(f"  cuDSS n={plan['n']}: permanent {mb(plan['cudss_permanent'])} MB; init | plan | factorize 1st "
                     f"| factorize later | solve 1st | solve later | freed | held: {_phase_columns(plan)}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='VRAM profile of named selftest cases (gpu).')
    parser.add_argument('cases', nargs='*', default=['em_bor_cudss_driven'])
    parser.add_argument('--depth', type=int, default=3, help='open_sol frames per allocation site (default 3)')
    parser.add_argument('--top', type=int, default=25, help='sites listed per peak table (default 25)')
    parser.add_argument('--tag', default='', help='suffix of the report names (out/<case><tag>.md), keeps earlier ones')
    args = parser.parse_args()
    TRACE.site_depth = args.depth
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRACE:
        for name in args.cases:
            RUNS.clear()
            api.selftest(out_dir=OUT_DIR / 'runs', names=[name])
            gc.collect()   # factors of closed Results freed now, so their 'freed' bytes reach the report
            text = f'# VRAM profile: {name}\n\n' + '\n'.join(run_text(run, args.top) for run in RUNS)
            (OUT_DIR / f'{name}{args.tag}.md').write_text(text, encoding='utf-8')
            (OUT_DIR / f'{name}{args.tag}.json').write_text(json.dumps(RUNS, indent=1, default=str), encoding='utf-8')
            if BOOK_ERRORS:
                text += '\n## Profiler errors (inside wrapped calls, not raised)\n\n' + '\n'.join(
                    f'- {line}' for line in BOOK_ERRORS) + '\n'
                (OUT_DIR / f'{name}{args.tag}.md').write_text(text, encoding='utf-8')
                print(f'{len(BOOK_ERRORS)} profiler errors (see the report): {BOOK_ERRORS[0]}')
                BOOK_ERRORS.clear()
            print('\n'.join(console_text(run) for run in RUNS) or f'{name}: no runs (unknown case or cpu only?)')
            print(f'wrote {OUT_DIR / (name + args.tag + ".md")}')


if __name__ == '__main__':
    main()
