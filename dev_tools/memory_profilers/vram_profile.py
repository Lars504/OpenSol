"""VRAM profiler (dev tooling, not part of open_sol): runs named selftest cases in-process with a cupy pool hook and
cuDSS wrappers; writes out/<case>.md + .json next to this script (pool peak by allocation site, cuDSS plans, estimate,
and a reconciliation: the estimate's big items held together should sit just below the measured bytes).
Run (from any folder, gpu): python vram_profile.py em_bor_cudss_driven [more case names]"""
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # memory_profilers -> dev_tools -> repository
sys.path.insert(0, str(ROOT))
OUT_DIR = Path(__file__).resolve().parent / 'out'
SNAPSHOT_STEP_BYTES = 128 * 1024   # a new peak-time snapshot once the run peak grew this much
SITE_DEPTH = 3                     # open_sol frames per allocation site
TOP_SITES = 25

import cupy as cp                                  # noqa: E402
from cupy.cuda import memory_hook                  # noqa: E402
from open_sol import api                           # noqa: E402
from open_sol.model import study as study_module   # noqa: E402
from open_sol.model.builder import ESTIMATE_SMALL_ITEMS, estimate_verdict   # noqa: E402
from open_sol.solvers import cudss_factor          # noqa: E402

PACKAGE = os.path.normcase(str(ROOT / 'open_sol'))
SOLVER_FILES = ('solvers' + os.sep + 'cudss_factor.py', 'solvers' + os.sep + 'sparse_direct.py')


def _frames(start, depth, skip_solvers=False):
    parts, frame = [], start
    while frame is not None and len(parts) < depth:
        name = os.path.normcase(frame.f_code.co_filename)
        if name.startswith(PACKAGE) and not (skip_solvers and name.endswith(SOLVER_FILES)):
            parts.append(f'{os.path.relpath(name, PACKAGE)}:{frame.f_lineno} {frame.f_code.co_name}'.replace('\\', '/'))
        frame = frame.f_back
    return ' < '.join(parts) or '(outside open_sol)'


def mb(n):
    return 'n/a' if n is None else f'{n / 2 ** 20:.3f}'


class PoolTrace(memory_hook.MemoryHook):
    """Live pool allocations (pointer -> bytes, site, run, stage), the run peak and a by-site snapshot near it."""
    name = 'PoolTrace'

    def __init__(self):
        self.live = {}
        self.current = 0
        self.run_index = 0
        self.stage = 'other'
        self.region_peak = 0
        self.start_run()

    def start_run(self):
        self.run_index += 1
        self.run_start = self.current
        self.run_peak = self.current
        self.snapshot_at = -1
        self.snapshot = None
        self.build_peak = self.current
        self.build_snapshot_at = -1
        self.build_snapshot = None
        self.region_stats = defaultdict(lambda: {'calls': 0, 'max_peak': 0, 'max_net': 0, 'sum_net': 0})

    def malloc_postprocess(self, **kwargs):
        size = int(kwargs['mem_size'])
        self.live[kwargs['mem_ptr']] = (size, _frames(sys._getframe(1), SITE_DEPTH), self.run_index, self.stage)
        self.current += size
        self.region_peak = max(self.region_peak, self.current)
        if self.current > self.run_peak:
            self.run_peak = self.current
            if self.run_peak >= self.snapshot_at + SNAPSHOT_STEP_BYTES:
                self.snapshot, self.snapshot_at = self._snapshot(self.run_peak), self.run_peak
        if self.stage == 'build' and self.current > self.build_peak:
            self.build_peak = self.current
            if self.build_peak >= self.build_snapshot_at + SNAPSHOT_STEP_BYTES:
                self.build_snapshot, self.build_snapshot_at = self._snapshot(self.build_peak), self.build_peak

    def free_postprocess(self, **kwargs):
        entry = self.live.pop(kwargs.get('mem_ptr'), None)
        if entry is not None:
            self.current -= entry[0]

    def _snapshot(self, peak):
        by_site = defaultdict(int)
        earlier = 0
        for size, site, run_index, stage in self.live.values():
            if run_index == self.run_index:
                by_site[f'[{stage}] {site}'] += size
            else:
                earlier += size
        return {'peak': peak - self.run_start, 'stage': self.stage, 'earlier_runs': earlier,
                'trigger': _frames(sys._getframe(2), 8), 'by_site': dict(by_site)}


TRACE = PoolTrace()
PLANS = []
OUTSIDE = {'baseline': 0, 'high_water': 0}


def _sync():
    cp.cuda.Device().synchronize()


def _outside_pool_bytes():
    """Device bytes in use outside the cupy pool (cudaMemGetInfo minus the pool's reserved bytes), sampled."""
    _sync()
    free, total = cp.cuda.runtime.memGetInfo()
    return (total - free) - cp.get_default_memory_pool().total_bytes()


def _sample_outside():
    OUTSIDE['high_water'] = max(OUTSIDE['high_water'], _outside_pool_bytes() - OUTSIDE['baseline'])


def region(label, function):
    def wrapped(*args, **kwargs):
        saved, start = TRACE.region_peak, TRACE.current
        TRACE.region_peak = TRACE.current
        try:
            return function(*args, **kwargs)
        finally:
            stats = TRACE.region_stats[label]
            stats['calls'] += 1
            stats['max_peak'] = max(stats['max_peak'], TRACE.region_peak - start)
            stats['max_net'] = max(stats['max_net'], TRACE.current - start)
            stats['sum_net'] += TRACE.current - start
            TRACE.region_peak = max(saved, TRACE.region_peak)
    return wrapped


def staged(stage, function):
    def wrapped(*args, **kwargs):
        saved, TRACE.stage = TRACE.stage, stage
        try:
            return function(*args, **kwargs)
        finally:
            TRACE.stage = saved
    return wrapped


Factorization = cudss_factor.CudssFactorization
original_plan_solver = Factorization._plan_solver
original_factorize = Factorization._factorize_and_probe


def plan_solver(self, kind, started):
    before = _outside_pool_bytes()
    original_plan_solver(self, kind, started)
    after = _outside_pool_bytes()
    pattern = getattr(self, '_pattern', None)
    PLANS.append({'object': id(self), 'caller': _frames(sys._getframe(1), 4, skip_solvers=True),
                  'n': self.n_unknowns, 'stored_entries': int(self._factor_matrix.nnz),
                  'full_entries': None if pattern is None else int(pattern[1].size),
                  'value_dtype': str(cp.dtype(self.value_dtype)), 'factor_dtype': str(cp.dtype(self._factor_dtype)),
                  'kind': kind, 'cudss_permanent': self.permanent_bytes, 'cudss_peak': self.peak_bytes,
                  'plan_outside_drop': after - before, 'factorize_outside_drop': None, 'lu_entries': None,
                  'factorizations': 0})
    _sample_outside()


def factorize_and_probe(self, defer=False):
    before = _outside_pool_bytes()
    try:
        return original_factorize(self, defer)
    finally:
        after = _outside_pool_bytes()
        for plan in reversed(PLANS):
            if plan['object'] == id(self):
                plan['factorizations'] += 1
                plan['lu_entries'] = self.lu_entries
                if plan['factorize_outside_drop'] is None:
                    plan['factorize_outside_drop'] = after - before
                break
        _sample_outside()


Factorization._plan_solver = plan_solver
Factorization._factorize_and_probe = factorize_and_probe
for method in ('refactorize', 'refactorize_device_values', 'refactorize_device_lower', 'solve', '_device_index_maps',
               '_device_asymmetry', '_plan', '_device_plan'):
    setattr(Factorization, method, region(f'cudss {method}', getattr(Factorization, method)))
_build_session = staged('build', study_module.build_session)
_run_built = staged('study', region('study run', study_module._run_built))
for module_name in ('study', 'study_adaptation', 'study_estimation', 'study_sectors'):
    module = __import__(f'open_sol.model.{module_name}', fromlist=['_'])
    module.build_session, module._run_built = _build_session, _run_built

RUNS = []
original_selftest_run = api._selftest_run


def selftest_run(spec, run_dir, max_wall_time):
    del PLANS[:]
    _sync()
    TRACE.start_run()
    OUTSIDE['baseline'], OUTSIDE['high_water'] = _outside_pool_bytes(), 0
    started = time.perf_counter()
    results, record = original_selftest_run(spec, run_dir, max_wall_time)
    _sample_outside()
    summary = {}
    if run_dir is not None and (Path(run_dir) / 'summary.json').exists():
        summary = json.loads((Path(run_dir) / 'summary.json').read_text())
    try:
        run_name = Path(run_dir).relative_to(OUT_DIR / 'runs').as_posix()
    except (TypeError, ValueError):
        run_name = str(run_dir)
    RUNS.append({'name': run_name,
                 'seconds': time.perf_counter() - started, 'status': record.get('status'),
                 'pool_at_start': TRACE.run_start, 'pool_peak': TRACE.run_peak - TRACE.run_start,
                 'pool_end': TRACE.current - TRACE.run_start, 'snapshot': TRACE.snapshot,
                 'build_snapshot': TRACE.build_snapshot,
                 'regions': {label: dict(stats) for label, stats in TRACE.region_stats.items()},
                 'plans': [dict(plan) for plan in PLANS], 'outside_high_water': OUTSIDE['high_water'],
                 'estimate': (summary.get('build') or {}).get('gpu_estimate'),
                 'build_ledger_peak': (summary.get('build') or {}).get('gpu_peak_bytes'),
                 'run_memory': summary.get('run_memory'), 'external_memory': summary.get('external_memory')})
    return results, record


api._selftest_run = selftest_run


def _items_text(rows, phased=False):
    lines = []
    for row in rows or ():
        if phased:
            lines.append(f'  - [{row[0]}] {row[1]}: {mb(row[2])} MB')
        else:
            lines.append(f'  - {row[0]}: {mb(row[1])} MB')
    return lines or ['  - (none)']


def _reconcile_line(label, measured, estimated):
    if measured is None or estimated is None:
        return f'- {label}: n/a'
    verdict = estimate_verdict(measured, estimated) or f'ok (within the {ESTIMATE_SMALL_ITEMS:.0%} small-item margin)'
    share = (measured - estimated) / max(estimated, 1)
    return (f'- {label}: estimate {mb(estimated)} MB, measured {mb(measured)} MB, held beyond the estimate '
            f'{mb(measured - estimated)} MB ({share:+.0%}): {verdict}')


def reconciliation_lines(run, estimate):
    """Estimate (kept + run + largest transient phase; external) vs measured, judged as builder.estimate_verdict."""
    pool_estimate = (None if estimate.get('gpu_steady') is None   # the run's pool peak may fall in its build
                     else max(estimate['gpu_steady'] + (estimate.get('gpu_run') or 0),
                              estimate.get('gpu_build_peak') or 0))
    external_estimate = (None if estimate.get('gpu_external') is None
                         else estimate['gpu_external'] + (estimate.get('external_extra') or 0))
    largest_plan = max((plan['cudss_permanent'] or 0 for plan in run['plans']), default=None)
    return ['### Reconciliation (big items held together: should sit just below the measured bytes)',
            _reconcile_line('pool, used-bytes peak of build + run (hook) vs max(build peak, kept + run)',
                            run['pool_peak'] or None,   # a cpu run allocates nothing in the pool
                            pool_estimate),
            _reconcile_line('outside the pool, largest cuDSS permanent', largest_plan, external_estimate),
            _reconcile_line('outside the pool, real device bytes (memGetInfo high water)',
                            run['outside_high_water'] if run['plans'] else None, external_estimate),
            f"- pool bytes still held when the run returned: {mb(run['pool_end'])} MB (arrays outliving the run)", '']


def _fill_columns(plan):
    """lu entries per n log2 n and per n^(4/3), cuDSS bytes per lu entry (the fill law and the storage rule apart)."""
    n, lu = plan['n'], plan['lu_entries']
    if not lu or n < 2:
        return 'n/a | n/a | n/a'
    return (f"{lu / (n * math.log2(n)):.3f} | {lu / n ** (4.0 / 3.0):.3f} | "
            f"{(plan['cudss_permanent'] or 0) / lu:.1f}")


def run_text(run):
    estimate = run['estimate'] or {}
    lines = [f"## {run['name']}  ({run['status']}, {run['seconds']:.1f} s)", '',
             *reconciliation_lines(run, estimate),
             '### Measured (profiler)',
             f"- pool used at run start (earlier runs): {mb(run['pool_at_start'])} MB",
             f"- pool used-bytes peak of this run (hook, no block rounding): {mb(run['pool_peak'])} MB; "
             f"left at the end: {mb(run['pool_end'])} MB",
             f"- outside-pool device bytes high water (memGetInfo, sampled at plan / factorize): "
             f"{mb(run['outside_high_water'])} MB",
             f"- summary run_memory: {run['run_memory']}", f"- summary external_memory: {run['external_memory']}",
             f"- build ledger peak: {mb(run['build_ledger_peak'])} MB", '',
             '### Estimate',
             f"- gpu_steady {mb(estimate.get('gpu_steady'))}, gpu_build_peak {mb(estimate.get('gpu_build_peak'))}, "
             f"gpu_run {mb(estimate.get('gpu_run'))}, gpu_external {mb(estimate.get('gpu_external'))}, "
             f"external_extra {mb(estimate.get('external_extra'))} MB",
             '- kept items:', *_items_text(estimate.get('items')),
             '- run items:', *_items_text(estimate.get('run_items')),
             '- transient items (phase, label):', *_items_text(estimate.get('transient_items'), phased=True),
             '- external items:', *_items_text(estimate.get('external_items')), '']
    if run['plans']:
        lines += ['### cuDSS plans', '',
                  '| caller | n | stored nnz | full nnz | value / factor | kind | cuDSS permanent MB | cuDSS peak MB | '
                  'plan drop MB | factorize drop MB | lu_nnz | lu / n log2 n | lu / n^(4/3) | B / lu entry | '
                  'factorizations |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for plan in run['plans']:
            lines.append(f"| {plan['caller']} | {plan['n']} | {plan['stored_entries']} | {plan['full_entries']} | "
                         f"{plan['value_dtype']} / {plan['factor_dtype']} | {plan['kind']} | "
                         f"{mb(plan['cudss_permanent'])} | {mb(plan['cudss_peak'])} | {mb(plan['plan_outside_drop'])} | "
                         f"{mb(plan['factorize_outside_drop'])} | {plan['lu_entries']} | {_fill_columns(plan)} | "
                         f"{plan['factorizations']} |")
        lines.append('')
    if run['regions']:
        lines += ['### Pool regions (peak / net over the call, MB)', '',
                  '| region | calls | max peak | max net | summed net |', '|---|---|---|---|---|']
        for label, stats in sorted(run['regions'].items()):
            lines.append(f"| {label} | {stats['calls']} | {mb(stats['max_peak'])} | {mb(stats['max_net'])} | "
                         f"{mb(stats['sum_net'])} |")
        lines.append('')
    for title, snapshot in (('build peak', run.get('build_snapshot')), ('run peak', run['snapshot'])):
        if not snapshot:
            continue
        rows = sorted(snapshot['by_site'].items(), key=lambda item: -item[1])
        lines += [f"### Live pool bytes near the {title} ({mb(snapshot['peak'])} MB over the run start, stage "
                  f"{snapshot['stage']}; earlier runs' arrays {mb(snapshot['earlier_runs'])} MB)", '',
                  f"Peak allocation: `{snapshot['trigger']}`", '', '| MB | [stage] site |', '|---|---|']
        for site, size in rows[:TOP_SITES]:
            lines.append(f'| {mb(size)} | `{site}` |')
        rest = sum(size for _, size in rows[TOP_SITES:])
        if rest:
            lines.append(f'| {mb(rest)} | ({len(rows) - TOP_SITES} more sites) |')
        lines.append('')
    return '\n'.join(lines)


def main(names):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TRACE:
        for name in names:
            del RUNS[:]
            api.selftest(out_dir=OUT_DIR / 'runs', names=[name])
            text = f'# VRAM profile: {name}\n\n' + '\n'.join(run_text(run) for run in RUNS)
            (OUT_DIR / f'{name}.md').write_text(text, encoding='utf-8')
            (OUT_DIR / f'{name}.json').write_text(json.dumps(RUNS, indent=1, default=str), encoding='utf-8')
            print(f'wrote {OUT_DIR / (name + ".md")}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['em_bor_cudss_driven'])
