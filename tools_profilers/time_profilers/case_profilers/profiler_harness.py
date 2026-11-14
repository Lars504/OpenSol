"""Shared harness of the profiler_solver_*.py scripts: runs selftest case specs with GPU-synchronised part timers and
cProfile, then writes out/<name>/report.txt (parts, untimed rest, host hot spots, sync / transfer sites) and runs.md
(one column per run, e.g. per --compare value)."""
import _thread
import argparse
import cProfile
import copy
import functools
import importlib
import inspect
import io
import json
import os
import pstats
import re
import sys
import threading
import time
from pathlib import Path

PROCESS_START = time.time()   # --time-limit counts from here (the run_all timeout counts the whole command)
HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]   # case_profilers -> time_profilers -> tools_profilers -> repository
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
# no on-disk gmsh cache (open_sol's default folder is on for api.run): a profile times the mesher, never a disk read
os.environ['OPEN_SOL_MESH_CACHE'] = ''
OUT_ROOT = HERE / 'out'
PACKAGE_PATTERN = '(?i)' + re.escape(str(REPOSITORY / 'open_sol') + os.sep)
# cProfile entries whose callers are listed: device -> host reads, explicit syncs, host -> device uploads (cupy
# asarray / array live in from_data.py); implicit syncs of float(cupy scalar) are slot calls cProfile cannot see
SYNC_PATTERN = (r"'get' of 'cupy|asnumpy|'item' of 'cupy|'tolist' of 'cupy|synchronize|from_data\.py|"
                r"cupyx[\\/]scipy[\\/]sparse.*\(get\)")
SYNC_NOISE_PATTERN = r'multiprocessing[\\/]synchronize\.py'   # matches 'synchronize' by its file name only
N_ROWS = 45
N_CALLER_TARGETS = 15
CALLER_MIN_SECONDS = 0.001   # callers of an outside function below this (own and cumulative) fold into one line
# one record per part: [calls, inclusive s, exclusive s, inclusive process-CPU s, device wait s (own exit synchronize)]
EMPTY_RECORD = (0, 0.0, 0.0, 0.0, 0.0)
# stdout line of a profile cut short (--time-limit or Ctrl+C): its report holds the parts timed until then
# (run_all_profilers reads it: keep the text the same there)
PARTIAL_REPORT_MARKER = 'partial report'


class _TimeLimit:
    """Interrupts the main thread (KeyboardInterrupt) when `limit` s since PROCESS_START have passed; None: never."""

    def __init__(self, limit):
        self.fired = False
        self._lock = threading.Lock()
        self._armed = limit is not None
        self._timer = None
        if self._armed:
            self._timer = threading.Timer(max(float(limit) - (time.time() - PROCESS_START), 0.0), self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self):
        with self._lock:
            if self._armed:
                self.fired = True
                _thread.interrupt_main()   # raised at the main thread's next bytecode (after a running C call)

    def disarm(self):
        with self._lock:
            self._armed = False
        if self._timer is not None:
            self._timer.cancel()

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
    'open_sol.solvers.sparse_direct:SparseDirectSolver.pencil',
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
    # cuDSS phases inside refactorize / factorize_pencil (once per factorisation or solve, so the edge syncs are cheap)
    'open_sol.solvers.cudss_factor:CudssFactorization._equilibrated',
    'open_sol.solvers.cudss_factor:equilibrate_values',
    'open_sol.solvers.cudss_factor:CudssFactorization._fused_refresh',
    'open_sol.solvers.cudss_factor:CudssFactorization._plan_solver',
    'open_sol.solvers.cudss_factor:CudssFactorization._device_plan',
    'open_sol.solvers.cudss_factor:CudssFactorization._upload_values',
    'open_sol.solvers.cudss_factor:CudssFactorization._factorize_and_probe',
    'open_sol.solvers.cudss_factor:CudssFactorization._gmres_refine',
    'open_sol.solvers.cudss_factor:relative_asymmetry',
    'open_sol.solvers.cudss_factor:_lower_triangle',
    'open_sol.solvers.sparse_direct:DevicePencil.__init__',
    'open_sol.solvers.sparse_direct:DevicePencil.combined',
    'open_sol.solvers.sparse_direct:HostPencil.__init__',
    'open_sol.solvers.sparse_direct:HostPencil.combined',
    # host direct (PARDISO / MUMPS, double and mixed): open_sol's side, then the library calls of the adapters
    # (adapter __init__ = analysis, factor / refactor = numeric factorisation, solve = one library solve)
    'open_sol.solvers.host_direct:HostDirectFactorization.refactorize',
    'open_sol.solvers.host_direct:HostDirectFactorization.refactorize_values',
    'open_sol.solvers.host_direct:HostDirectFactorization._equilibrated',
    'open_sol.solvers.host_direct:HostDirectFactorization._equilibrated_values',
    'open_sol.solvers.host_direct:HostDirectFactorization._checked_solve',
    'open_sol.solvers.host_direct:HostDirectFactorization._factorize',
    'open_sol.solvers.host_direct:HostDirectFactorization._probe_factor',
    'open_sol.solvers.host_direct:HostDirectFactorization.solve',
    'open_sol.solvers.host_direct:HostDirectFactorization.solve_host',
    'open_sol.solvers.host_direct:HostDirectFactorization.solve_preconditioned',
    'open_sol.solvers.host_direct:HostDirectFactorization.solve_preconditioned_host',
    'open_sol.solvers.host_direct:_PardisoFactor.__init__',
    'open_sol.solvers.host_direct:_PardisoFactor.factor',
    'open_sol.solvers.host_direct:_PardisoFactor.refactor',
    'open_sol.solvers.host_direct:_PardisoFactor.solve',
    'open_sol.solvers.host_direct:_MumpsFactor.__init__',
    'open_sol.solvers.host_direct:_MumpsFactor.factor',
    'open_sol.solvers.host_direct:_MumpsFactor.refactor',
    'open_sol.solvers.host_direct:_MumpsFactor.solve',
    # multigrid PCG: set-up phases once per factorize; precondition is one V-cycle per CG iteration (10-40 per solve,
    # each far dearer than the edge syncs)
    'open_sol.solvers.multigrid:MultigridPreconditioner.setup',
    'open_sol.solvers.multigrid:aggregate_strong_links',
    'open_sol.solvers.multigrid:device_aggregate_strong_links',
    'open_sol.solvers.multigrid:MultigridPreconditioner._build_level',
    'open_sol.solvers.multigrid:MultigridPreconditioner._coarsest_solver',
    'open_sol.solvers.multigrid:MultigridPreconditioner.precondition',
)
# solver-role runtimes: parts nobody calls stay out of the table
SOLVER_ROLE_PARTS = tuple((target, 'role: ' + target.split(':', 1)[1]) for target in _ROLE_TARGETS)
# library clocks: after a call of the target, the float seconds attribute its instance set is a row of its own (a
# sub-part: out of the target's exclusive time); the attribute must be a new value, so a call that raised first is
# not counted twice
LIBRARY_CLOCKS = {
    'open_sol.solvers.cudss_factor:CudssFactorization._factorize_and_probe': (
        ('factorize_seconds', 'library: cuDSS numeric factorize (open_sol host clock)'),),
}


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
    """Calls, inclusive / exclusive wall, inclusive process-CPU and device-wait seconds per wrapped callable; the GPU is
    synchronised at entry and exit, so queued kernels count in the part that launched them."""

    def __init__(self, synchronize=True):
        self.records = {}      # label -> list(EMPTY_RECORD) layout
        self.missing = []      # targets that did not resolve (listed in the report, never raised)
        self._frames = []      # child wall seconds of each active call
        self._depth = {}       # label -> active calls (recursion: inclusive counted at the outermost only)
        self._patches = []     # (owner, name, original or None = delete on restore)
        self.library_labels = set()   # rows read from LIBRARY_CLOCKS (no process-CPU reading)
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
        timed = self._timed(label, function, LIBRARY_CLOCKS.get(target, ()))
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

    def _timed(self, label, function, clocks=()):
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
                return self._call(label, function, args, kwargs, clocks)
        timed._profiler_label = label
        return timed

    def _call(self, label, function, args, kwargs, clocks=()):
        record = self.records.setdefault(label, list(EMPTY_RECORD))
        outermost = not self._depth.get(label)
        instance = args[0] if clocks and args else None
        clocks_before = [getattr(instance, attribute, None) for attribute, _ in clocks] if instance is not None else []
        self._synchronize()
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        frame = [0.0]
        self._frames.append(frame)
        self._depth[label] = self._depth.get(label, 0) + 1
        try:
            return function(*args, **kwargs)
        finally:
            # the exit synchronize waits for device work this call queued and no timed sub-part waited for already
            sync_start = time.perf_counter()
            self._synchronize()
            sync_end = time.perf_counter()
            wall = sync_end - start_wall
            cpu = time.process_time() - start_cpu
            record[4] += sync_end - sync_start
            self._frames.pop()
            self._depth[label] -= 1
            for (attribute, clock_label), previous in zip(clocks, clocks_before):
                seconds = getattr(instance, attribute, None)
                if isinstance(seconds, float) and seconds is not previous:
                    clock = self.records.setdefault(clock_label, list(EMPTY_RECORD))
                    clock[0] += 1
                    clock[1] += seconds
                    clock[2] += seconds
                    frame[0] += seconds   # a sub-part of this call
                    self.library_labels.add(clock_label)
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
        lines = [f'{"part":64s} {"calls":>7s} {"incl s":>9s} {"excl s":>9s} {"excl %":>7s} {"cpu/wall":>8s} '
                 f'{"wait s":>8s}']
        timed = 0.0
        for label, (calls, inclusive, exclusive, cpu, wait) in sorted(self.records.items(),
                                                                       key=lambda item: -item[1][2]):
            timed += exclusive
            library = label in self.library_labels
            share = '-' if library else f'{cpu / inclusive if inclusive > 0.0 else 0.0:.2f}'
            waited = '-' if library or self._synchronize is _no_sync else f'{wait:.3f}'
            lines.append(f'{label[:64]:64s} {calls:7d} {inclusive:9.3f} {exclusive:9.3f} '
                         f'{100.0 * exclusive / total:6.1f}% {share:>8s} {waited:>8s}')
        rest = total - timed
        lines.append(f'{"untimed rest (run wall - timed exclusive)":64s} {"":7s} {"":9s} {rest:9.3f} '
                     f'{100.0 * rest / total:6.1f}%')
        return '\n'.join(lines)

    def snapshot(self):
        """Copy of the records (per-run parts are the difference of two snapshots)."""
        return {label: list(record) for label, record in self.records.items()}


def _records_since(records, before):
    """{part: record (EMPTY_RECORD layout)} added to records since the snapshot `before`."""
    added = {}
    for label, record in records.items():
        previous = before.get(label, EMPTY_RECORD)
        if record[0] > previous[0]:
            added[label] = [record[0] - previous[0]] + [now - then for now, then in zip(record[1:], previous[1:])]
    return added


def run_columns_table(run_parts, run_walls):
    """Markdown table with one column per run, cells 'inclusive ms per call x calls' (compare variants side by side)."""
    names = sorted({label for parts in run_parts.values() for label in parts})
    lines = ['| part (ms per call x calls) | ' + ' | '.join(run_parts) + ' |', '|' + '---|' * (len(run_parts) + 1)]
    for label in names:
        cells = [f'{1e3 * parts[label][1] / parts[label][0]:.1f} x {parts[label][0]}' if label in parts else '-'
                 for parts in run_parts.values()]
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')
    lines.append('| run wall s | ' + ' | '.join(f'{run_walls[run]:.2f}' for run in run_parts) + ' |')
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


# --compare: the direct-factor setting is back_end on sparse_direct and factorization on arpack_shift_invert (the
# pair schema.auto_back_end_values resolves alike), so either name sets both
SETTING_ALIASES = {'back_end': ('back_end', 'factorization'), 'factorization': ('factorization', 'back_end')}
# run_all's five solver groups: a library back end by its package, every other solver (open_sol's own pcg, explicit
# kernels, line_block, ...) by array library and precision: cupy double -> cudss, cupy single -> cudss_mixed, numpy
# (the cpu backend) -> scipy
LIBRARY_GROUPS = {'cudss': 'cudss', 'cudss_mixed': 'cudss_mixed', 'cusolver_qr': 'cudss', 'pardiso': 'pardiso',
                  'pardiso_mixed': 'pardiso', 'mumps': 'mumps', 'mumps_mixed': 'mumps', 'mumps_blr': 'mumps',
                  'scipy_splu': 'scipy', 'superlu': 'scipy'}
SINGLE_PRECISIONS = ('single', 'mixed', 'half_storage')


def solver_group(back_end, precision, backend) -> str:
    """run_all group of one role's solver: its library back end's package, else numpy on the cpu (scipy) or cupy by
    precision (single / mixed: cudss_mixed, double or unknown: cudss)."""
    if back_end in LIBRARY_GROUPS:
        return LIBRARY_GROUPS[back_end]
    if backend == 'cpu':
        return 'scipy'
    return 'cudss_mixed' if precision in SINGLE_PRECISIONS else 'cudss'


def run_groups(choices, spec, backend) -> list:
    """Sorted groups of a run's solver choices; a run without solver roles (EME): its backend's group at the model's
    solver_defaults precision."""
    groups = {choice['group'] for choice in choices}
    if not groups:
        groups = {solver_group(None, getattr(spec.solver_defaults, 'precision', 'auto'), backend)}
    return sorted(groups)


def _compared_variant(spec, setting, value, backend, registry, excluded=()):
    """(spec copy, [(physics, role), ...] changed): every solver role the study uses (but `excluded`) whose solver type
    has `setting` (or its alias) and whose setting_choices accept `value` (an opt-in value, e.g. MUMPS / PARDISO, only
    where they list it: roles.solver_misfit) gets a new node with it (over its node's settings, or its Automatic type
    and settings on `backend`; a node whose type lacks the setting, e.g. an example's exact pcg node under back_end,
    takes the Automatic type and settings when that one has it)."""
    from open_sol.model.schema import solver_grid
    from open_sol.model.spec import SolverSpec
    variant = copy.deepcopy(spec)
    nodes = {node.name: node for node in variant.solvers}
    changed = []

    def setting_key(type_name):
        names = {param.name for param in registry.solver_type(type_name).params}
        return next((name for name in SETTING_ALIASES.get(setting, (setting,)) if name in names), None)
    for physics_spec in variant.physics:
        physics_type = registry.physics_type(physics_spec.type)
        for role, solver_role in physics_type.solver_roles.items():
            if not solver_role.used_in(variant.study.type) or (physics_spec.name, role) in excluded:
                continue
            node = nodes.get((physics_spec.solvers or {}).get(role) or '')
            automatic = solver_role.automatic(backend, solver_grid(variant, physics_spec), variant.study.type)
            type_name, settings = (node.type, node.settings) if node is not None else automatic
            if setting_key(type_name) is None:
                type_name, settings = automatic
            solver_type = registry.solver_type(type_name)
            key = setting_key(type_name)
            if key is None or value not in solver_role.setting_choices.get(key, (value,)):
                continue
            if value in solver_type.opt_in_settings.get(key, ()) and key not in solver_role.setting_choices:
                continue
            physics_spec.solvers[role] = _compared_node_name(physics_spec.name, role)
            variant.solvers.append(SolverSpec(physics_spec.solvers[role], type_name, {**dict(settings), key: value}))
            changed.append((physics_spec.name, role))
    return variant, changed


def automatic_solvers(spec):
    """Spec copy in which every physics role runs Automatic: its node assignment dropped (the nodes stay, unreferenced),
    so the model resolves type, back end and preconditioner as it does with no node."""
    variant = copy.deepcopy(spec)
    for physics_spec in variant.physics:
        if physics_spec.solvers:
            physics_spec.solvers.clear()
    return variant


def with_precision(spec, precision):
    """Spec copy with solver_defaults.precision set (every solver offering it and setting none of its own takes it)."""
    variant = copy.deepcopy(spec)
    variant.solver_defaults.precision = precision
    return variant


def _compared_node_name(physics_name, role) -> str:
    return f'compared_{physics_name}_{role}'


def _refused_roles(variant, changed, backend, registry) -> set:
    """The (physics, role) of `changed` whose compared node an error of the variant's validation names: a physics that
    refuses the value in its own validate (the 1D membrane's banded host factor takes only auto / scipy_splu)."""
    from open_sol.model.validate import validate
    checked = copy.deepcopy(variant)
    checked.solver_defaults.backend = backend
    errors = [issue.message for issue in validate(checked, registry) if issue.level == 'error']
    return {entry for entry in changed if any(f"'{_compared_node_name(*entry)}'" in message for message in errors)}


def _run_key(variant, spec) -> str:
    """JSON text of a compared variant without the solver nodes the comparison displaced (named by a physics of spec,
    by none of the variant): equal for two runs that solve the same model under the compared value."""

    def named(model):
        return {name for physics_spec in model.physics for name in (physics_spec.solvers or {}).values()}
    displaced = named(spec) - named(variant)
    kept = copy.copy(variant)
    kept.solvers = [node for node in variant.solvers if node.name not in displaced]
    return json.dumps(kept.to_dict(), sort_keys=True, default=str)


def setting_variants(specs, assignment, backend, keep_unchanged=True):
    """{'<run> NAME=value': spec copy} per spec and value of assignment 'NAME=v1,v2,...' (_compared_variant; a role whose
    physics refuses the value in validation keeps its node); a value no
    role of a run accepts is skipped, a run no value applies to runs once as written (keep_unchanged) or is dropped, a
    run that under a value is the same model as an earlier one (it differed only by the node the value replaces) is
    dropped (all printed). Values stay strings (choice settings such as back_end)."""
    from open_sol.model.schema import get_registry
    setting, separator, values_text = assignment.partition('=')
    values = [value.strip() for value in values_text.split(',') if value.strip()]
    if not separator or not setting.strip() or not values:
        raise SystemExit(f'--compare needs NAME=v1,v2,..., got {assignment!r}')
    setting = setting.strip()
    registry = get_registry()
    variants, first_with_key = {}, {}
    for run_name, spec in specs.items():
        skipped = []
        for value in values:
            variant, changed = _compared_variant(spec, setting, value, backend, registry)
            refused = _refused_roles(variant, changed, backend, registry) if changed else set()
            if refused:
                print(f"--compare: run {run_name!r}: {', '.join('.'.join(entry) for entry in sorted(refused))} refuse "
                      f'{setting}={value} in validation; left as written', flush=True)
                variant, changed = _compared_variant(spec, setting, value, backend, registry, excluded=refused)
            if not changed:
                skipped.append(value)
                continue
            label = f'{run_name} {setting}={value}'
            twin = first_with_key.setdefault(_run_key(variant, spec), label)
            if twin != label:
                print(f'--compare: run {label!r} is the same model as {twin!r}; dropped', flush=True)
                continue
            variants[label] = variant
        if len(skipped) == len(values):
            if keep_unchanged:
                variants[run_name] = spec
            print(f'--compare: run {run_name!r}: no solver role takes {setting} {values}; '
                  + ('runs as written' if keep_unchanged else 'dropped (--compare-only)'), flush=True)
        elif skipped:
            print(f'--compare: run {run_name!r}: no solver role takes {setting} {skipped}; those values skipped',
                  flush=True)
    return variants


def _scaled(value, factor):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value / factor
    return f'({value}) / {factor}'


def refined(spec, factor):
    """Copy of spec with every mesh size (dx, h_min, refinement / band dx) divided by factor."""
    spec = copy.deepcopy(spec)
    for mesh in spec.meshes:
        mesh.dx = _scaled(mesh.dx, factor)
        mesh.h_min = _scaled(mesh.h_min, factor)
        for refinement in mesh.refinements:
            for name in ('dx', 'dx_x', 'dx_y', 'dx_z'):
                setattr(refinement, name, _scaled(getattr(refinement, name), factor))
        for axis in mesh.axes:
            for band in axis.bands:
                band.dx = _scaled(band.dx, factor)
    return spec


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


def _run_memory(summary, backend):
    """{gpu_peak_bytes: the larger of the build peak and the run's pool high-water + cuDSS factors outside the pool
    (None on the cpu backend: not measured), host_peak_bytes: this process's peak working set / RSS so far} of one run."""
    from tools_profilers.time_profilers.case_runner import current_and_peak_rss
    host_peak = current_and_peak_rss()[1]
    if backend == 'cpu':   # build.gpu_peak_bytes there is the tracemalloc peak of host numpy, not device memory
        return {'gpu_peak_bytes': None, 'host_peak_bytes': host_peak}
    build_peak = (summary.get('build') or {}).get('gpu_peak_bytes')
    pool_peak = (summary.get('run_memory') or {}).get('measured_bytes')
    external_peak = (summary.get('external_memory') or {}).get('measured_bytes') or 0
    run_peak = None if pool_peak is None else pool_peak + external_peak
    known = [value for value in (build_peak, run_peak) if value is not None]
    return {'gpu_peak_bytes': max(known) if known else None, 'host_peak_bytes': host_peak}


def _run_choices(spec, summary, backend):
    """[{role 'physics.role', type, back_end, preconditioner, factorizations}] the run's solvers used: back_end from
    the runtime counters (sparse_direct), else the setting with 'auto' resolved as schema.auto_back_end_values does;
    factorizations: the role's counter (eigen: eigen_factorizations; None without one), 0 = the role never factorised
    (a matrix-free eigen solve: run_all does not count such a run for a compared back end)."""
    from open_sol.model.schema import auto_back_end_values, get_registry, solver_grid
    from open_sol.solvers.sparse_direct import expected_back_end
    registry = get_registry()
    physics_specs = {physics_spec.name: physics_spec for physics_spec in spec.physics}
    choices = []
    for physics_name, record in (summary.get('solvers') or {}).items():
        for role, solver in (record.get('roles') or {}).items():
            settings = solver.get('settings') or {}
            key = 'back_end' if 'back_end' in settings else 'factorization'
            counters = solver.get('counters') or {}
            back_end = counters.get('back_end') or settings.get(key)
            physics_spec = physics_specs.get(physics_name)
            if back_end == 'auto' and physics_spec is not None:
                solver_role = registry.physics_type(physics_spec.type).solver_roles.get(role)
                if solver_role is not None:
                    back_end = auto_back_end_values({key: 'auto'}, solver_role, backend,
                                                    solver_grid(spec, physics_spec))[key]
            if back_end in ('auto', 'superlu'):   # host SuperLU: named as the run_all flag names it
                back_end = expected_back_end('auto', backend)
            precision = _solver_precision(registry, solver.get('type'), settings)
            choices.append({'role': f'{physics_name}.{role}', 'type': solver.get('type'), 'back_end': back_end,
                            'preconditioner': settings.get('preconditioner'), 'precision': precision,
                            'group': solver_group(back_end, precision, backend),
                            'factorizations': counters.get('factorizations', counters.get('eigen_factorizations'))})
    return choices


def _solver_precision(registry, type_name, settings):
    """A role's precision: its 'precision' setting, else its solver type's only (first) one, None for an unknown type."""
    if settings.get('precision'):
        return settings['precision']
    try:
        return registry.solver_type(type_name).precisions[0]
    except (KeyError, TypeError):
        return None


AUTOMATIC_BACKEND = 'gpu'   # the backend whose Automatic the run_all plot rings (the model default)
PLANNED_GROUPS = 'solver groups: '   # stdout line before the build: the groups the runs' roles take (run_all reads it)


def _automatic_choices(spec, backend=AUTOMATIC_BACKEND, spec_nodes=False):
    """[{role 'physics.role', type, back_end, preconditioner}] Automatic resolves to for every role the study uses on
    `backend`, whatever nodes the spec has (spec_nodes: a role's node instead, the run's choices foreseen before the
    build): back_end 'auto' resolved as the model does (cuDSS where the role lists it, else host SuperLU
    'scipy_splu'); a runtime cuDSS refusal (fallback to SuperLU) is not seen here."""
    from open_sol.model.schema import (auto_back_end_values, fill_defaults, get_registry, solver_grid,
                                       with_default_precision)
    from open_sol.solvers.sparse_direct import expected_back_end
    registry = get_registry()
    nodes = {node.name: node for node in spec.solvers} if spec_nodes else {}
    choices = []
    for physics_spec in spec.physics:
        for role, solver_role in registry.physics_type(physics_spec.type).solver_roles.items():
            if not solver_role.used_in(spec.study.type):
                continue
            grid = solver_grid(spec, physics_spec)
            node = nodes.get((physics_spec.solvers or {}).get(role) or '')
            type_name, settings = ((node.type, dict(node.settings)) if node is not None else
                                   solver_role.automatic(backend, grid, spec.study.type))   # study_defaults too
            solver_type = registry.solver_type(type_name)
            settings = with_default_precision(spec, solver_type, settings, solver_role)   # solver_defaults.precision
            values = auto_back_end_values(fill_defaults(solver_type.params, settings), solver_role, backend, grid)
            back_end = values.get('back_end', values.get('factorization'))
            if back_end in ('auto', 'superlu'):
                back_end = expected_back_end('auto', backend)
            precision = _solver_precision(registry, type_name, values)
            choices.append({'role': f'{physics_spec.name}.{role}', 'type': type_name, 'back_end': back_end,
                            'preconditioner': values.get('preconditioner'), 'precision': precision,
                            'group': solver_group(back_end, precision, backend)})
    return choices


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


def _caller_lines(stats, function, min_seconds) -> list:
    """Lines of one cProfile entry and its callers (calls, own s, cumulative s of the calls from each, largest first);
    callers below min_seconds in both fold into one count line."""
    _, calls, own, cumulative, callers = stats.stats[function]
    lines = [f'{pstats.func_std_string(function)}: {calls} calls, own {own:.3f} s, cumulative {cumulative:.3f} s']
    rows = []
    for caller, value in callers.items():   # cProfile: (calls, primitive calls, own s, cumulative s) from this caller
        caller_calls, caller_own, caller_cumulative = ((value[0], value[2], value[3]) if isinstance(value, tuple)
                                                       else (value, 0.0, 0.0))
        rows.append((max(caller_own, caller_cumulative), caller_calls, caller_own, caller_cumulative,
                     pstats.func_std_string(caller)))
    rows.sort(key=lambda row: (-row[0], row[4]))
    shown = [row for row in rows if row[0] >= min_seconds]
    lines += [f'  <- {caller_calls:>8} {caller_own:8.3f} {caller_cumulative:8.3f}  {text}'
              for _, caller_calls, caller_own, caller_cumulative, text in shown]
    folded = rows[len(shown):]
    if folded:
        lines.append(f'  <- {sum(row[1] for row in folded):>8} calls from {len(folded)} callers below '
                     f'{1e3 * min_seconds:g} ms each')
    return lines


def _cprofile_text(profiler):
    """Own / cumulative time of open_sol functions, own time of all, callers of the top non-open_sol host functions
    (those above CALLER_MIN_SECONDS) and every caller of the sync / transfer entries."""
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
    stream.write(f'===== callers of the {N_CALLER_TARGETS} largest own-time functions outside open_sol '
                 '(<- calls, own s, cumulative s, caller) =====\n')
    for _, function in outside[:N_CALLER_TARGETS]:
        stream.write('\n'.join(_caller_lines(stats, function, CALLER_MIN_SECONDS)) + '\n')
    stream.write('===== device syncs and transfers (every caller; the harness synchronize is its own part timing; '
                 'syncdetect._declare_synchronize callers = implicit syncs) =====\n')
    sync, noise = re.compile(SYNC_PATTERN), re.compile(SYNC_NOISE_PATTERN)
    sites = [(entry[2], function) for function, entry in stats.stats.items()
             if sync.search(pstats.func_std_string(function)) and not noise.search(pstats.func_std_string(function))]
    for _, function in sorted(sites, key=lambda item: (-item[0], pstats.func_std_string(item[1]))):
        stream.write('\n'.join(_caller_lines(stats, function, 0.0)) + '\n')
    return stream.getvalue()


def wrapped_timers(parts, synchronize=True):
    """PartTimers with every part wrapped. Call before anything builds the registry (case_specs, api.run): StudyType /
    FeatureType callbacks keep the function objects of registration time, so later wraps would never be called."""
    timers = PartTimers(synchronize=synchronize)
    for part in parts:
        timers.wrap(*part) if isinstance(part, tuple) else timers.wrap(part)
    return timers


def _interrupted_summary(run_dir, status):
    """summary.json api.run wrote after the interrupt (its minimal report), status replaced; {} without one."""
    summary_path = run_dir / 'summary.json'
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.is_file() else {}
    except ValueError:
        summary = {}
    summary['status'] = status
    return summary


def profile_runs(name, specs, parts=(), backend='gpu', repeat=1, warmup=False, use_cprofile=True, synchronize=True,
                 parameters=None, save_arrays=True, note='', timers=None, warmup_specs=None, time_limit=None):
    """Run every spec (repeat times, after an optional untimed warm-up) under the part timers and cProfile; writes
    out/<name>/report.txt + profile.prof and returns the report path. timers: from wrapped_timers (else parts are
    wrapped here, too late for registry-held callbacks when the registry exists already). warmup_specs: {run: spec} the
    warm-up runs instead of specs (small runs compiling the same kernels). time_limit (s since PROCESS_START) or Ctrl+C:
    the run going on is stopped (status 'timed_out' / 'cancelled') and the report holds what was timed until then."""
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    if timers is None:
        timers = wrapped_timers(parts, synchronize)
    profiler = cProfile.Profile() if use_cprofile else None
    run_texts, total = [], 0.0
    run_parts, run_walls, run_memory = {}, {}, {}   # per run: its parts (records difference), wall and memory peaks
    planned = sorted({group for spec in specs.values()
                      for group in run_groups(_automatic_choices(spec, backend, spec_nodes=True), spec, backend)})
    print(f'{PLANNED_GROUPS}{"+".join(planned)}', flush=True)   # run_all's '>>>' line for the auto pick
    limit = _TimeLimit(time_limit)
    stopped = None   # text of where an interrupt stopped the profile
    try:
        try:
            if warmup:
                stopped = 'during the warm-up (nothing timed)'
                for run_name, spec in (specs if warmup_specs is None else warmup_specs).items():
                    _run_one(spec, out_dir / f'warmup_{run_name}', backend, parameters, save_arrays)
                timers.records.clear()
                stopped = None
            for index in range(repeat):
                for run_name, spec in specs.items():
                    label = run_name if repeat == 1 else f'{run_name}_{index + 1}'
                    before = timers.snapshot()
                    start = time.perf_counter()
                    if profiler is not None:
                        profiler.enable()
                    try:
                        summary = _run_one(spec, out_dir / label, backend, parameters, save_arrays)
                    except KeyboardInterrupt:   # the timed parts unwound through their finally: recorded so far
                        stopped = f'during run {label}'
                        summary = _interrupted_summary(out_dir / label, 'timed_out' if limit.fired else 'cancelled')
                    finally:
                        if profiler is not None:
                            profiler.disable()
                    wall = time.perf_counter() - start
                    total += wall
                    run_parts[label], run_walls[label] = _records_since(timers.records, before), wall
                    choices, automatic = _run_choices(spec, summary, backend), _automatic_choices(spec)
                    run_memory[label] = {'status': summary.get('status'), **_run_memory(summary, backend),
                                         'solvers': choices, 'automatic': automatic,
                                         'groups': run_groups(choices, spec, backend),
                                         'automatic_groups': run_groups(automatic, spec, AUTOMATIC_BACKEND)}
                    run_texts.append(_run_text(label, wall, summary))
                    print(run_texts[-1].splitlines()[0], flush=True)
                    if stopped is not None:
                        break
                if stopped is not None:
                    break
        finally:
            limit.disarm()
    except KeyboardInterrupt:   # between runs (bookkeeping) or in the warm-up
        stopped = stopped or 'between runs'
        if not run_walls:   # the warm-up's parts are not part of the profile
            timers.records.clear()
    finally:
        if profiler is not None:
            profiler.disable()
        timers.restore()
    if stopped is not None:
        cause = f'time limit {time_limit:g} s reached' if limit.fired else 'interrupted (Ctrl+C)'
        skipped = [run_name if repeat == 1 else f'{run_name}_{index + 1}' for index in range(repeat)
                   for run_name in specs if (run_name if repeat == 1 else f'{run_name}_{index + 1}') not in run_walls]
        note = (f'{PARTIAL_REPORT_MARKER}: {cause} {stopped}; the parts below are those timed until then (a part '
                f"running at the interrupt counts up to it); not run: {', '.join(skipped) or 'none'}"
                + (f'\n{note}' if note else ''))
        print(note.splitlines()[0], flush=True)
    header = (f'profiler {name}: backend {backend}, repeat {repeat}, warmup {warmup}, cprofile {use_cprofile}, '
              f'gpu sync at part edges {synchronize}, total run wall {total:.2f} s\n'
              'Exclusive = own wall minus timed sub-parts; cpu/wall = process CPU / inclusive wall (> 1 threaded BLAS; '
              'a blocking device sync spins a core, so ~1 also while waiting on the GPU; Windows CPU clock ticks '
              '~16 ms); wait s = seconds of the part\'s own exit synchronize = only the device work still queued at '
              'its exit: ~0 on host-bound parts and also on device-bound parts that end in a host read (a solve '
              'reading its residual waits inside, not at exit). cProfile inflates python-heavy parts: compare with '
              '--no-cprofile.\n')
    sections = [header + (note + '\n' if note else ''), '\n'.join(run_texts), '', '===== parts =====',
                timers.table(total)]
    columns = run_columns_table(run_parts, run_walls)
    (out_dir / 'runs.md').write_text(columns + '\n', encoding='utf-8')
    (out_dir / 'runs.json').write_text(json.dumps(
        {run: {'wall_seconds': run_walls[run], **run_memory[run],
               'parts': {label: {'calls': calls, 'inclusive_seconds': inclusive, 'exclusive_seconds': exclusive,
                                 'ms_per_call': 1e3 * inclusive / calls, 'device_wait_seconds': wait}
                         for label, (calls, inclusive, exclusive, _, wait) in sorted(parts.items())}}
         for run, parts in run_parts.items()}, indent=1), encoding='utf-8')
    if len(run_parts) > 1:
        sections += ['', '===== parts per run (inclusive ms per call x calls; also runs.md / runs.json) =====', columns]
    if timers.missing:
        sections += ['', '===== parts that did not resolve (fix the target) =====', '\n'.join(timers.missing)]
    if profiler is not None:
        profiler.dump_stats(str(out_dir / 'profile.prof'))
        sections += ['', _cprofile_text(profiler)]
    report_path = out_dir / 'report.txt'
    report_path.write_text('\n'.join(sections) + '\n', encoding='utf-8')
    print(f'\n===== parts =====\n{timers.table(total)}\nreport: {report_path}', flush=True)
    return report_path


def output_name(name, backend, compare=None, compare_only=False, assignments=(), automatic=False, precision=None):
    """Folder name of one profile: name + the settings that change its results (backend, --automatic-solvers,
    --precision, --compare, --set), e.g. 'membrane_film__gpu__back_end-pardiso__only', so runs of different solvers
    never overwrite each other."""
    parts = ([name, backend] + (['automatic'] if automatic else []) + ([precision] if precision else [])
             + ([compare] + (['only'] if compare_only else []) if compare else []) + list(assignments))
    return '__'.join(re.sub(r'[^A-Za-z0-9_.+-]', '-', part.replace(',', '+')) for part in parts)


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
    parser.add_argument('--automatic-solvers', action='store_true',
                        help='drop every solver node assignment: each role runs what Automatic picks (run_all: no flag)')
    parser.add_argument('--precision', choices=('single', 'double', 'mixed'), default=None,
                        help='solver_defaults.precision of every run (each solver offering it and not setting its own)')
    parser.add_argument('--compare', metavar='NAME=v1,v2', help='run every spec once per value of solver setting NAME '
                        '(e.g. back_end=cudss,cudss_mixed): one column per run in runs.md')
    parser.add_argument('--compare-only', action='store_true',
                        help='drop the runs no --compare value applies to (instead of running them as written)')
    parser.add_argument('--time-limit', type=float, default=None, metavar='S',
                        help='stop the run going on S s after the start of the process and write the report of what '
                             'was timed until then (a long library call ends first; Ctrl+C does the same)')
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
    if options.automatic_solvers:
        run_specs = {key: automatic_solvers(spec) for key, spec in run_specs.items()}
    if options.precision:
        run_specs = {key: with_precision(spec, options.precision) for key, spec in run_specs.items()}
    if options.compare:
        run_specs = setting_variants(run_specs, options.compare, options.backend,
                                     keep_unchanged=not options.compare_only)
        if not run_specs:
            print(f'--compare-only: no run of {name} takes {options.compare}; nothing profiled', flush=True)
            return None
    small_warmup = warmup_specs() if own_specs and warmup_specs is not None and options.warmup else None
    if small_warmup is not None and spec_edit is not None:
        small_warmup = {key: spec_edit(spec) for key, spec in small_warmup.items()}
    source = f'specs of {name}' if specs is not None and options.case == case else f'case {options.case}'
    folder = output_name(name, options.backend, options.compare, options.compare_only, options.set,
                         options.automatic_solvers, options.precision)
    return profile_runs(folder, run_specs, parts, backend=options.backend, repeat=options.repeat, warmup=options.warmup,
                        use_cprofile=not options.no_cprofile, synchronize=not options.no_sync,
                        parameters=_parameters(options.set), save_arrays=not options.no_arrays,
                        note=f'{source}, runs {list(run_specs)}' + (f'; {note}' if note else '')
                        + (f'; warm-up runs {list(small_warmup)}' if small_warmup is not None else ''),
                        timers=timers, warmup_specs=small_warmup, time_limit=options.time_limit or None)
