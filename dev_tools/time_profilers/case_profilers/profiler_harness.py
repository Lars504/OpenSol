"""Shared harness of the profiler_solver_*.py scripts: runs selftest case specs with GPU-synchronised part timers and
cProfile, then writes out/<name>/report.txt (parts, untimed rest, host hot spots, sync / transfer sites)."""
import argparse
import cProfile
import functools
import importlib
import inspect
import io
import json
import os
import pstats
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]   # case_profilers -> time_profilers -> dev_tools -> repository
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
OUT_ROOT = HERE / 'out'
PACKAGE_PATTERN = '(?i)' + re.escape(str(REPOSITORY / 'open_sol') + os.sep)
# cProfile entries whose callers are listed: device -> host reads, explicit syncs, host -> device uploads (cupy
# asarray / array live in from_data.py); implicit syncs of float(cupy scalar) are slot calls cProfile cannot see
SYNC_PATTERN = (r"'get' of 'cupy|asnumpy|'item' of 'cupy|'tolist' of 'cupy|synchronize|from_data\.py|"
                r"cupyx[\\/]scipy[\\/]sparse.*\(get\)")
N_ROWS = 45
N_CALLER_TARGETS = 15

# parts every profiler times: (target 'module:qualname', label)
STUDY_PARTS = (
    ('open_sol.model.study:run_study', 'study: run_study (incl. build)'),
    ('open_sol.model.builder:build', 'build: builder.build'),
    ('open_sol.model.meshing:make_domain', 'build: meshing.make_domain'),
    ('open_sol.model.meshing:element_mesh', 'build: meshing.element_mesh (triangle / tet mesher)'),
    ('open_sol.io.results_io:save_results', 'output: save_results'),
)
_ROLE_TARGETS = (
    'open_sol.solvers.sparse_direct:SparseDirectSolver.factorize',
    'open_sol.solvers.sparse_direct:SparseDirectSolver.factorize_host',
    'open_sol.solvers.sparse_direct:SparseDirectSolver.device_pencil',
    'open_sol.solvers.sparse_direct:SparseDirectSolver.factorize_pencil',
    'open_sol.solvers.sparse_direct:SparseDirectSolver.factorize_device_csr',
    'open_sol.solvers.sparse_direct:SparseDirectSolver.solve_recycled',
    'open_sol.solvers.sparse_direct:SparseDirectSolver.solve_recycled_host',
    'open_sol.solvers.sparse_direct:Factorization.solve',
    'open_sol.solvers.sparse_direct:Factorization.solve_host',
    'open_sol.solvers.sparse_direct:DiffusionDirectSolver.factorize',
    'open_sol.solvers.sparse_direct:DiffusionDirectSolver.factorize_host_operator',
    'open_sol.solvers.sparse_direct:DiffusionDirectSolver.factorize_device_values',
    'open_sol.solvers.sparse_direct:DiffusionDirectSolver.solve',
    'open_sol.solvers.cudss_factor:CudssFactorization.solve',
    'open_sol.solvers.cudss_factor:CudssFactorization.solve_host',
    'open_sol.solvers.cudss_factor:CudssFactorization.refactorize',
    'open_sol.solvers.cudss_factor:CudssFactorization.refactorize_device_values',
    'open_sol.solvers.cudss_factor:CudssFactorization.refactorize_device_lower',
    'open_sol.solvers.cudss_factor:CudssFactorization._plan',
    'open_sol.solvers.pcg:PCGSolver.factorize',
    'open_sol.solvers.pcg:PCGSolver.solve',
    'open_sol.solvers.banded:BandedSolver.factorize',
    'open_sol.solvers.banded:BandedSolver.solve',
    'open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver.factorize_host',
    'open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver.eigsh_shifted',
    'open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver.eigs_shifted',
    'open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver.eigs',
    'open_sol.solvers.arpack_shift_invert:ArpackShiftInvertSolver.eigsh_inverse',
    'open_sol.solvers.line_block:LineBlockSolver.factorize',
    'open_sol.solvers.line_block:LineBlockSolver.factorize_operators',
    'open_sol.solvers.line_block:LineBlockSolver.solve_shifted',
    'open_sol.solvers.line_block:LineBlockFactor.solve',
)
# solver-role runtimes: parts nobody calls stay out of the table
SOLVER_ROLE_PARTS = tuple((target, 'role: ' + target.split(':', 1)[1]) for target in _ROLE_TARGETS)


def _no_sync():
    pass


def _gpu_synchronizer():
    """Device synchronise when a GPU answers, else a no-op."""
    from open_sol.core.backend import gpu_available, synchronize
    if not gpu_available():
        return _no_sync
    return functools.partial(synchronize, 'gpu')


def _resolve(target):
    """(owner, attribute name, raw attribute) of 'module:qualname'; class attributes looked up along the MRO."""
    module_name, separator, qualname = target.partition(':')
    if not separator or not qualname:
        raise ValueError("expected 'package.module:qualname'")
    owner = importlib.import_module(module_name)
    *path, name = qualname.split('.')
    for part in path:
        owner = getattr(owner, part)
    if isinstance(owner, type):
        for klass in owner.__mro__:
            if name in vars(klass):
                return owner, name, vars(klass)[name]
        raise AttributeError(f'{owner.__name__} has no attribute {name!r}')
    return owner, name, getattr(owner, name)


class PartTimers:
    """Calls, inclusive / exclusive wall and inclusive process-CPU seconds per wrapped callable; the GPU is
    synchronised at entry and exit, so queued kernels count in the part that launched them."""

    def __init__(self, synchronize=True):
        self.records = {}      # label -> [calls, inclusive s, exclusive s, inclusive process-CPU s]
        self.missing = []      # targets that did not resolve (listed in the report, never raised)
        self._frames = []      # child wall seconds of each active call
        self._depth = {}       # label -> active calls (recursion: inclusive counted at the outermost only)
        self._patches = []     # (owner, name, original or None = delete on restore)
        self._synchronize = _gpu_synchronizer() if synchronize else _no_sync

    def wrap(self, target, label=None):
        """Time target 'package.module:Class.method' or 'package.module:function' (every open_sol module binding the
        same function object too)."""
        label = label or target.split(':', 1)[-1]
        try:
            owner, name, raw = _resolve(target)
        except (ImportError, AttributeError, ValueError) as exc:
            self.missing.append(f'{target}: {type(exc).__name__}: {exc}')
            return
        function = raw.__func__ if isinstance(raw, (staticmethod, classmethod)) else (
            raw.fget if isinstance(raw, property) else raw)
        if not callable(function):
            self.missing.append(f'{target}: not callable ({type(raw).__name__})')
            return
        if hasattr(function, '_profiler_label'):
            self.missing.append(f"{target}: already timed as '{function._profiler_label}'")
            return
        timed = self._timed(label, function)
        if isinstance(raw, staticmethod):
            replacement = staticmethod(timed)
        elif isinstance(raw, classmethod):
            replacement = classmethod(timed)
        elif isinstance(raw, property):
            replacement = property(timed, raw.fset, raw.fdel, raw.__doc__)
        else:
            replacement = timed
        self._patch(owner, name, replacement)
        if isinstance(owner, type):
            return
        for module_name, module in list(sys.modules.items()):
            if module is owner or module is None or not module_name.startswith('open_sol'):
                continue
            for key, value in list(getattr(module, '__dict__', {}).items()):
                if value is raw:
                    self._patch(module, key, replacement)

    def _patch(self, owner, name, replacement):
        had_own = not isinstance(owner, type) or name in vars(owner)
        self._patches.append((owner, name, getattr(owner, name) if not isinstance(owner, type) else
                              (vars(owner)[name] if had_own else None)))
        setattr(owner, name, replacement)

    def restore(self):
        """Undo every patch (last first)."""
        for owner, name, original in reversed(self._patches):
            if original is None:
                delattr(owner, name)
            else:
                setattr(owner, name, original)
        self._patches.clear()

    def _timed(self, label, function):
        if inspect.isgeneratorfunction(function):
            @functools.wraps(function)
            def timed(*args, **kwargs):   # creation and every next() step are timed (calls count the steps)
                generator = self._call(label, function, args, kwargs)
                while True:
                    try:
                        value = self._call(label, next, (generator,), {})
                    except StopIteration as stop:
                        return stop.value
                    yield value
        else:
            @functools.wraps(function)
            def timed(*args, **kwargs):
                return self._call(label, function, args, kwargs)
        timed._profiler_label = label
        return timed

    def _call(self, label, function, args, kwargs):
        record = self.records.setdefault(label, [0, 0.0, 0.0, 0.0])
        outermost = not self._depth.get(label)
        self._synchronize()
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        frame = [0.0]
        self._frames.append(frame)
        self._depth[label] = self._depth.get(label, 0) + 1
        try:
            return function(*args, **kwargs)
        finally:
            self._synchronize()
            wall = time.perf_counter() - start_wall
            cpu = time.process_time() - start_cpu
            self._frames.pop()
            self._depth[label] -= 1
            if self._frames:
                self._frames[-1][0] += wall
            record[0] += 1
            record[2] += wall - frame[0]
            if outermost:
                record[1] += wall
                record[3] += cpu

    def table(self, total):
        """Parts by exclusive seconds, then the untimed rest of total."""
        total = max(total, 1e-12)
        lines = [f'{"part":64s} {"calls":>7s} {"incl s":>9s} {"excl s":>9s} {"excl %":>7s} {"cpu/wall":>8s}']
        timed = 0.0
        for label, (calls, inclusive, exclusive, cpu) in sorted(self.records.items(), key=lambda item: -item[1][2]):
            timed += exclusive
            share = cpu / inclusive if inclusive > 0.0 else 0.0
            lines.append(f'{label[:64]:64s} {calls:7d} {inclusive:9.3f} {exclusive:9.3f} '
                         f'{100.0 * exclusive / total:6.1f}% {share:8.2f}')
        rest = total - timed
        lines.append(f'{"untimed rest (run wall - timed exclusive)":64s} {"":7s} {"":9s} {rest:9.3f} '
                     f'{100.0 * rest / total:6.1f}%')
        return '\n'.join(lines)


def case_specs(case_name, runs=None, variant=None):
    """{run name: ModelSpec} of a registered selftest case (runs: only those names; variant: that SelfTestVariant's
    moved specs, as api.selftest runs them)."""
    from open_sol import api
    registry = api.get_registry()
    cases = {case.name: case for case in registry.selftests()}
    if case_name not in cases:
        raise KeyError(f'no selftest case {case_name!r}')
    case = cases[case_name]
    specs = case.specs()
    if variant:
        variants = {entry.name: entry for entry in case.variants}
        if variant not in variants:
            raise KeyError(f"case {case_name!r} has no variant {variant!r} (has: {', '.join(variants) or 'none'})")
        specs = {name: api._variant_spec(spec, variants[variant], registry) for name, spec in specs.items()}
    if runs:
        unknown = [name for name in runs if name not in specs]
        if unknown:
            raise KeyError(f"case {case_name!r} has no runs {unknown} (has: {', '.join(specs)})")
        specs = {name: specs[name] for name in runs}
    return specs


def _run_one(spec, run_dir, backend, parameters, save_arrays):
    """api.run of one spec into run_dir -> its summary.json dict (status from the Results when no summary)."""
    from open_sol import api
    results = api.run(spec, out_dir=run_dir, backend=backend, parameters=parameters or None, save_arrays=save_arrays)
    try:
        status = results.status
    finally:
        results.close()
    summary_path = run_dir / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    summary.setdefault('status', status)
    return summary


def _run_text(label, wall, summary):
    """Run header line, the build stages above 50 ms and the solver records of summary.json."""
    def seconds(key):
        value = summary.get(key)
        return f'{value:.2f}' if isinstance(value, (int, float)) else '?'
    message = f" {summary['message']}" if summary.get('message') else ''
    lines = [f"run {label}: status {summary.get('status')}, wall {wall:.2f} s (summary wall {seconds('wall_time')} s, "
             f"build {seconds('build_seconds_total')} s, study {seconds('study_seconds')} s){message}"]
    stages = [(name, seconds) for name, seconds in (summary.get('build') or {}).get('stage_seconds', [])
              if seconds >= 0.05]
    if stages:
        lines.append('  build stages >= 0.05 s: ' + ', '.join(f'{name} {seconds:.2f}' for name, seconds in stages))
    if summary.get('solvers'):
        lines.append('  solvers: ' + json.dumps(summary['solvers'], default=str))
    return '\n'.join(lines)


def _cprofile_text(profiler):
    """Own / cumulative time of open_sol functions, own time of all, callers of the top non-open_sol host functions
    and of the sync / transfer entries."""
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stream.write('===== own time (tottime), open_sol functions: python-level host work incl. numpy operators =====\n')
    stats.sort_stats('tottime').print_stats(PACKAGE_PATTERN, N_ROWS)
    stream.write('===== cumulative, open_sol functions =====\n')
    stats.sort_stats('cumulative').print_stats(PACKAGE_PATTERN, N_ROWS)
    stream.write('===== own time, all functions =====\n')
    stats.sort_stats('tottime').print_stats(N_ROWS)
    package = re.compile(PACKAGE_PATTERN)
    harness = str(HERE)
    outside = [(entry[2], function) for function, entry in stats.stats.items()
               if not package.search(pstats.func_std_string(function))
               and not pstats.func_std_string(function).startswith(harness)]
    outside.sort(key=lambda item: -item[0])
    stream.write(f'===== callers of the {N_CALLER_TARGETS} largest own-time functions outside open_sol =====\n')
    for _, function in outside[:N_CALLER_TARGETS]:
        stats.print_callers(re.escape(pstats.func_std_string(function)))
    stream.write('===== device syncs and transfers (callers; the harness synchronize is its own part timing) =====\n')
    stats.print_callers(SYNC_PATTERN)
    return stream.getvalue()


def wrapped_timers(parts, synchronize=True):
    """PartTimers with every part wrapped. Call before anything builds the registry (case_specs, api.run): StudyType /
    FeatureType callbacks keep the function objects of registration time, so later wraps would never be called."""
    timers = PartTimers(synchronize=synchronize)
    for part in parts:
        timers.wrap(*part) if isinstance(part, tuple) else timers.wrap(part)
    return timers


def profile_runs(name, specs, parts=(), backend='gpu', repeat=1, warmup=False, use_cprofile=True, synchronize=True,
                 parameters=None, save_arrays=True, note='', timers=None, warmup_specs=None):
    """Run every spec (repeat times, after an optional untimed warm-up) under the part timers and cProfile; writes
    out/<name>/report.txt + profile.prof and returns the report path. timers: from wrapped_timers (else parts are
    wrapped here, too late for registry-held callbacks when the registry exists already). warmup_specs: {run: spec} the
    warm-up runs instead of specs (small runs compiling the same kernels)."""
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if timers is None:
        timers = wrapped_timers(parts, synchronize)
    if warmup:
        for run_name, spec in (specs if warmup_specs is None else warmup_specs).items():
            _run_one(spec, out_dir / f'warmup_{run_name}', backend, parameters, save_arrays)
        timers.records.clear()
    profiler = cProfile.Profile() if use_cprofile else None
    run_texts, total = [], 0.0
    try:
        for index in range(repeat):
            for run_name, spec in specs.items():
                label = run_name if repeat == 1 else f'{run_name}_{index + 1}'
                start = time.perf_counter()
                if profiler is not None:
                    profiler.enable()
                try:
                    summary = _run_one(spec, out_dir / label, backend, parameters, save_arrays)
                finally:
                    if profiler is not None:
                        profiler.disable()
                wall = time.perf_counter() - start
                total += wall
                run_texts.append(_run_text(label, wall, summary))
                print(run_texts[-1].splitlines()[0], flush=True)
    finally:
        timers.restore()
    header = (f'profiler {name}: backend {backend}, repeat {repeat}, warmup {warmup}, cprofile {use_cprofile}, '
              f'gpu sync at part edges {synchronize}, total run wall {total:.2f} s\n'
              'Exclusive = own wall minus timed sub-parts; cpu/wall = process CPU / inclusive wall (~1 host-bound, '
              '> 1 threaded BLAS, ~0 waiting on the GPU; Windows CPU clock ticks ~16 ms). cProfile inflates python-heavy '
              'parts: compare with --no-cprofile.\n')
    sections = [header + (note + '\n' if note else ''), '\n'.join(run_texts), '', '===== parts =====',
                timers.table(total)]
    if timers.missing:
        sections += ['', '===== parts that did not resolve (fix the target) =====', '\n'.join(timers.missing)]
    if profiler is not None:
        profiler.dump_stats(str(out_dir / 'profile.prof'))
        sections += ['', _cprofile_text(profiler)]
    report_path = out_dir / 'report.txt'
    report_path.write_text('\n'.join(sections) + '\n', encoding='utf-8')
    print(f'\n===== parts =====\n{timers.table(total)}\nreport: {report_path}', flush=True)
    return report_path


def _parameters(assignments):
    """{name: expression} of NAME=EXPR strings."""
    parameters = {}
    for text in assignments:
        name, separator, expression = text.partition('=')
        if not separator:
            raise SystemExit(f'--set needs NAME=EXPR, got {text!r}')
        parameters[name.strip()] = expression.strip()
    return parameters


def main(name, parts, description, case=None, runs=None, variant=None, specs=None, spec_edit=None, backend='gpu',
         repeat=1, note='', warmup_specs=None):
    """CLI of one profiler: the case's specs (or specs() -> {run: spec}), each passed through spec_edit(spec) when
    given, profiled by profile_runs; warmup_specs() -> {run: spec}: the --warmup runs of the profiler's own specs
    (default: every profiled spec once)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--case', default=case, help='selftest case whose specs run (default: the profiler\'s)')
    parser.add_argument('--runs', nargs='*', default=runs, help='only these run names of the case')
    parser.add_argument('--variant', default=variant, help='a SelfTestVariant name of the case')
    parser.add_argument('--backend', default=backend, choices=('gpu', 'cpu'))
    parser.add_argument('--repeat', type=int, default=repeat)
    parser.add_argument('--warmup', action='store_true', help='one untimed run first (kernel compiles, caches)')
    parser.add_argument('--no-cprofile', action='store_true', help='part timers only (no cProfile overhead)')
    parser.add_argument('--no-sync', action='store_true', help='no GPU synchronise at part edges')
    parser.add_argument('--no-arrays', action='store_true', help='api.run save_arrays=False')
    parser.add_argument('--set', action='append', default=[], metavar='NAME=EXPR', help='model parameter override')
    options = parser.parse_args()
    timers = wrapped_timers(parts, synchronize=not options.no_sync)   # before the registry exists
    own_specs = specs is not None and options.case == case
    if own_specs:
        run_specs = specs()
        if options.runs:
            run_specs = {key: run_specs[key] for key in options.runs}
    else:
        run_specs = case_specs(options.case, options.runs, options.variant)
    if spec_edit is not None:
        run_specs = {key: spec_edit(spec) for key, spec in run_specs.items()}
    small_warmup = warmup_specs() if own_specs and warmup_specs is not None and options.warmup else None
    if small_warmup is not None and spec_edit is not None:
        small_warmup = {key: spec_edit(spec) for key, spec in small_warmup.items()}
    source = f'specs of {name}' if specs is not None and options.case == case else f'case {options.case}'
    return profile_runs(name, run_specs, parts, backend=options.backend, repeat=options.repeat, warmup=options.warmup,
                        use_cprofile=not options.no_cprofile, synchronize=not options.no_sync,
                        parameters=_parameters(options.set), save_arrays=not options.no_arrays,
                        note=f'{source}, runs {list(run_specs)}' + (f'; {note}' if note else '')
                        + (f'; warm-up runs {list(small_warmup)}' if small_warmup is not None else ''),
                        timers=timers, warmup_specs=small_warmup)
