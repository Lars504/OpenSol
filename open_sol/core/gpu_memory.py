"""GPU memory of the cupy pool and the device: budget, status, release, out-of-memory detection and per-object
accounting. GPU functions run on the GPU-owning thread only; format_bytes is the only one the GUI thread calls."""
import gc
import importlib
import time
import tracemalloc
import traceback
import types
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from . import host_memory
from .backend import cp, cupy_installed

__all__ = [
    'LEAK_TOLERANCE_BYTES', 'DEFAULT_BUDGET_FRACTION', 'DEFAULT_HEADROOM_BYTES', 'DeviceInfo', 'MemoryStatus',
    'device_info', 'memory_status', 'cpu_device_info', 'cpu_memory_status', 'pool_used_bytes', 'pool_reserved_bytes', 'set_budget', 'budget_bytes', 'release_all',
    'is_out_of_memory', 'out_of_memory_message', 'drop_exception_frames', 'owned_bytes', 'format_bytes',
    'MemoryLedger', 'ledger_text',
]

LEAK_TOLERANCE_BYTES = 1 << 20          # pool use left above a closed session's baseline reported as a leak
DEFAULT_BUDGET_FRACTION = 0.9
DEFAULT_HEADROOM_BYTES = 512 * 2**20    # left outside the pool: CUDA context, kernels, library workspaces

_MAX_CHAIN_LINKS = 10
# texts of CUDA runtime / driver and cuSPARSE / cuBLAS / cuFFT / cuSOLVER allocation failures
_OUT_OF_MEMORY_MARKERS = ('cudaErrorMemoryAllocation', 'ALLOC_FAILED', 'CUDA_ERROR_OUT_OF_MEMORY')
_CUPY_PACKAGES = ('cupy', 'cupy_backends', 'cupyx')
_OUT_OF_MEMORY_REMEDIES = ('Reduce the cell count (mesh dx, domain size), DFT / far-field boxes or their '
                           'n_wavelengths, or close other GPU programs')
# never walked by owned_bytes (bound methods: their __self__ belongs to another owner)
_LEAF_TYPES = (str, bytes, bytearray, int, float, complex, type(None), range, memoryview, np.ndarray, np.generic,
               type, types.ModuleType, types.BuiltinFunctionType, types.MethodType)

_budget = None             # pool limit [bytes] applied by set_budget in this process
_budget_settings = None    # (fraction, headroom_bytes) of that set_budget call


@dataclass(frozen=True)
class DeviceInfo:
    """The current CUDA device: id, name and memory [bytes] when queried; kind 'cpu' (cpu_device_info): no usable
    GPU, id -1 and the host RAM instead."""
    device_id: int
    name: str
    total_bytes: int
    free_bytes: int
    kind: str = 'gpu'


@dataclass(frozen=True)
class MemoryStatus:
    """Cupy pool and device memory [bytes] at one moment; the pool counters cover the whole process."""
    pool_used: int          # live cupy arrays
    pool_reserved: int      # held by the pool: used + cached free blocks
    device_free: int
    device_total: int
    budget: int | None      # pool limit set by set_budget in this process, None before
    host_available: int | None = None   # available host RAM (host_memory.available_bytes), None when unknown
    kind: str = 'gpu'       # 'cpu' (cpu_memory_status): device_free / device_total are the host RAM, pool 0

    @property
    def other(self) -> int:
        """Device memory outside the pool: CUDA context, kernels, library workspaces, other processes (cpu: the RAM
        other programs use)."""
        return max(0, self.device_total - self.device_free - self.pool_reserved)

    def as_dict(self) -> dict:
        """The seven fields plus 'other' (ints; budget and host_available may be None, kind a str)."""
        return {'pool_used': int(self.pool_used), 'pool_reserved': int(self.pool_reserved),
                'device_free': int(self.device_free), 'device_total': int(self.device_total),
                'budget': None if self.budget is None else int(self.budget), 'other': int(self.other),
                'host_available': None if self.host_available is None else int(self.host_available),
                'kind': self.kind}


def device_info() -> DeviceInfo:
    """Id, name and free / total memory of the current device (driver query)."""
    device = cp.cuda.Device()
    name = cp.cuda.runtime.getDeviceProperties(device.id)['name']
    if isinstance(name, bytes):
        name = name.decode('utf-8', errors='replace')
    free_bytes, total_bytes = device.mem_info
    return DeviceInfo(int(device.id), str(name), int(total_bytes), int(free_bytes))


def cpu_device_info() -> DeviceInfo:
    """DeviceInfo of a machine without a usable GPU (kind 'cpu'): the host RAM (0 when unknown); no driver call."""
    return DeviceInfo(-1, 'CPU (numpy / scipy)', int(host_memory.total_bytes() or 0),
                      int(host_memory.available_bytes() or 0), kind='cpu')


def cpu_memory_status() -> MemoryStatus:
    """MemoryStatus of the cpu backend (kind 'cpu'): host RAM as the device memory, empty pool, no budget."""
    available = host_memory.available_bytes()
    return MemoryStatus(0, 0, int(available or 0), int(host_memory.total_bytes() or 0), None, available, kind='cpu')


def memory_status() -> MemoryStatus:
    """Pool counters (host side), one driver query of the device memory and the available host RAM; call at most
    about once per second."""
    pool = cp.get_default_memory_pool()
    pool_used, pool_reserved = int(pool.used_bytes()), int(pool.total_bytes())
    device_free, device_total = cp.cuda.Device().mem_info
    return MemoryStatus(pool_used, pool_reserved, int(device_free), int(device_total), _budget,
                        host_memory.available_bytes())


class _NoPool:
    """Pool counters without cupy (CPU backend machines): always 0."""

    @staticmethod
    def used_bytes():
        return 0

    @staticmethod
    def total_bytes():
        return 0


def _pool():
    """cupy's default memory pool, or _NoPool when cupy did not import."""
    return cp.get_default_memory_pool() if cupy_installed() else _NoPool


class _TracedHost:
    """Pool-like counters of the cpu backend's MemoryLedger: tracemalloc current / peak bytes (numpy registers its
    array buffers there; Python objects count too, malloc'd library memory such as SuperLU factors does not)."""

    @staticmethod
    def used_bytes():
        return tracemalloc.get_traced_memory()[0]

    @staticmethod
    def total_bytes():
        return tracemalloc.get_traced_memory()[1]


# imported before a cpu ledger starts tracing, so their module objects are not booked on the first build's stages
_CPU_BUILD_MODULES = ('scipy.sparse', 'scipy.sparse.linalg', 'scipy.ndimage')


def pool_used_bytes() -> int:
    """Bytes of live cupy arrays in the default pool (host counter, no driver call; 0 without cupy)."""
    return int(_pool().used_bytes())


def pool_reserved_bytes() -> int:
    """Bytes the default pool holds (live arrays + cached free blocks: its high-water mark since the last
    free_all_blocks; host counter, no driver call; 0 without cupy)."""
    return int(_pool().total_bytes())


def set_budget(fraction=DEFAULT_BUDGET_FRACTION, headroom_bytes=DEFAULT_HEADROOM_BYTES) -> int:
    """Limit the default pool to min(fraction * device total, free + pool reserved - headroom) bytes, so an
    over-sized model raises OutOfMemoryError instead of spilling into shared system memory; returns the limit."""
    global _budget, _budget_settings
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f'fraction must be in (0, 1], got {fraction}')
    if not headroom_bytes >= 0:
        raise ValueError(f'headroom_bytes must be >= 0, got {headroom_bytes}')
    pool = cp.get_default_memory_pool()
    free_bytes, total_bytes = cp.cuda.Device().mem_info
    reserved_bytes = int(pool.total_bytes())   # the pool's cached blocks are available to it
    budget = int(min(fraction * total_bytes, free_bytes + reserved_bytes - headroom_bytes))
    if budget <= 0:   # set_limit(size=0) would mean 'unlimited'
        raise RuntimeError(f'GPU has only {format_bytes(free_bytes)} free of {format_bytes(total_bytes)} '
                           f'(headroom {format_bytes(headroom_bytes)}): close other GPU programs')
    pool.set_limit(size=budget)
    _budget = budget
    _budget_settings = (float(fraction), int(headroom_bytes))
    return budget


def budget_bytes() -> int | None:
    """Pool limit of the last set_budget call in this process, None before."""
    return _budget


def release_all(collect_garbage=True) -> None:
    """Collect garbage (cycles may hold arrays), clear the cuFFT plan cache when it holds plans (their work areas live
    in the pool), then return the cached blocks of the default and pinned pools to the device (without cupy: gc only)."""
    if collect_garbage:
        gc.collect()
    if not cupy_installed():
        return
    from cupy.fft import config as fft_config
    plan_cache = fft_config.get_plan_cache()
    if plan_cache.get_curr_size() > 0:   # clear() of even an empty cache runs a full gc.collect (~30 ms) in cupy
        plan_cache.clear()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


_FROZEN_BASELINE = {'on': False}   # freeze_collect_baseline froze the gc and thaw_collect_baseline has not yet run


def freeze_collect_baseline() -> None:
    """gc.freeze right after a full collect (a sweep's first release): the objects alive then (modules, registry, spec)
    leave the cyclic gc, so the release_all collects after it scan only objects made since (a sweep value's session);
    nothing when the gc is frozen already (selftest_cache's freeze is never undone here)."""
    if not _FROZEN_BASELINE['on'] and gc.get_freeze_count() == 0:
        gc.freeze()
        _FROZEN_BASELINE['on'] = True


def thaw_collect_baseline() -> None:
    """Undo freeze_collect_baseline (the frozen objects rejoin the oldest generation); nothing when it did not freeze."""
    if _FROZEN_BASELINE['on']:
        _FROZEN_BASELINE['on'] = False
        gc.unfreeze()


def is_out_of_memory(exc) -> bool:
    """True when exc or a cause / context link is a GPU allocation failure (cupy pool or CUDA library); a host
    MemoryError is an ordinary failure."""
    return any(_is_gpu_allocation_error(link) for link in _exception_chain(exc))


def out_of_memory_message(exc) -> str:
    """'out of GPU memory: <innermost GPU error>; <budget of device memory>. <remedies>' (the numbers are left out
    when the driver query fails)."""
    gpu_errors = [link for link in _exception_chain(exc) if _is_gpu_allocation_error(link)]
    cause = gpu_errors[-1] if gpu_errors else exc
    message = f"out of GPU memory: {_exception_text(cause).rstrip('. ') or type(cause).__name__}"
    try:
        status = memory_status()
    except Exception:   # the driver query may fail after a CUDA error: no numbers then
        status = None
    if status is not None:
        message += f'; {_budget_text(status)}'
    return f'{message}. {_OUT_OF_MEMORY_REMEDIES}'


def drop_exception_frames(exc) -> None:
    """Clear the locals of the finished frames in the tracebacks of exc and its cause / context links, so a kept
    exception holds no GPU arrays (str(exc) and already formatted tracebacks are unaffected)."""
    for link in _exception_chain(exc):
        if link.__traceback__ is not None:
            traceback.clear_frames(link.__traceback__)   # skips frames that are still executing


def owned_bytes(obj, seen=None, max_depth=6, backend='gpu') -> int:
    """Bytes of the run-backend arrays (cupy allocations on 'gpu', numpy buffers on 'cpu') reachable from obj within
    max_depth references; each buffer counts once, its id going into `seen` (share it across owners)."""
    if seen is None:
        seen = set()
    count_host = backend == 'cpu'
    visited = set()   # ids of the objects already walked in this call
    total = 0
    pending = [(obj, 0)]
    while pending:
        item, depth = pending.pop()
        if count_host and isinstance(item, np.ndarray):
            buffer_owner = item   # a view (slice, broadcast_to) counts its base array once
            while isinstance(buffer_owner.base, np.ndarray):
                buffer_owner = buffer_owner.base
            if id(buffer_owner) not in seen:
                seen.add(id(buffer_owner))
                total += int(buffer_owner.nbytes)
            continue
        if not count_host and isinstance(item, cp.ndarray):
            memory = item.data.mem
            if id(memory) not in seen:
                seen.add(id(memory))
                total += int(getattr(memory, 'size', item.nbytes))
            continue
        if depth >= max_depth or isinstance(item, _LEAF_TYPES) or id(item) in visited:
            continue
        visited.add(id(item))
        pending.extend((child, depth + 1) for child in _references(item))
    return total


class MemoryLedger:
    """Pool accounting of one build: the used-bytes change and wall time of each tracked stage and the pool's
    high-water mark. peak_bytes = largest pool reserved size at a stage exit minus the used and free bytes at creation
    (block rounding: high; stranded free blocks the build reuses: low), at least the used-bytes growth.
    backend 'cpu': the same over host bytes traced by tracemalloc (started here unless already on; stop() ends it)."""

    def __init__(self, backend='gpu'):
        self.kind = backend
        self._started_tracing = False
        if backend == 'cpu':
            self._start_tracing()
        pool = self._counters()
        self.start_used_bytes = int(pool.used_bytes())   # other live sessions' arrays
        # free pool bytes free_all_blocks could not return (chunk remainders split around arrays of open Results):
        # not this build's, though its arrays may reuse them
        self.start_free_bytes = int(pool.total_bytes()) - self.start_used_bytes
        self.budget = _budget if backend == 'gpu' else None
        self.entries = []             # [(owner, pool used-bytes change over its stage)]
        self.stage_seconds = []       # [(owner, wall seconds)]: the tracked stages plus add_time phases
        self.peak_bytes = 0
        self.failed_owner = None      # owner of the stage an exception left, None while none did
        self._active_owner = None     # owner of the stage being tracked (nesting would book its bytes twice)

    def _start_tracing(self):
        for module_name in _CPU_BUILD_MODULES:
            try:
                importlib.import_module(module_name)
            except ImportError:   # no scipy: the build reports it
                pass
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            self._started_tracing = True
        tracemalloc.reset_peak()

    def _counters(self):
        return _TracedHost if self.kind == 'cpu' else _pool()

    def stop(self) -> None:
        """End the tracemalloc tracing this ledger started (cpu; idempotent, nothing on the gpu)."""
        if self._started_tracing and tracemalloc.is_tracing():
            tracemalloc.stop()
        self._started_tracing = False

    @contextmanager
    def track(self, owner):
        """Record the pool used-bytes change of the enclosed stage under owner, also when it raises; RuntimeError
        when called inside another track (nesting would count the inner bytes twice)."""
        if self._active_owner is not None:
            raise RuntimeError(f"MemoryLedger.track('{owner}') inside track('{self._active_owner}'): stages must not "
                               'nest')
        pool = self._counters()
        used_before = int(pool.used_bytes())
        completed = False
        self._active_owner = str(owner)
        started = time.perf_counter()
        try:
            yield
            completed = True
        finally:
            self._active_owner = None
            self.stage_seconds.append((str(owner), time.perf_counter() - started))
            self.entries.append((str(owner), int(pool.used_bytes()) - used_before))
            self._update_peak(pool)
            if not completed and self.failed_owner is None:
                self.failed_owner = str(owner)

    def add_time(self, owner, seconds) -> None:
        """Book the wall time of an untracked build phase (validation, estimates) under owner."""
        self.stage_seconds.append((str(owner), float(seconds)))

    def finish(self, owner='other (untracked)'):
        """Book the used bytes no stage recorded under owner (only when nonzero), so the entries sum to the change
        since creation; updates the peak."""
        pool = self._counters()
        untracked_bytes = int(pool.used_bytes()) - self.start_used_bytes - self.total_bytes
        if untracked_bytes:
            self.entries.append((str(owner), untracked_bytes))
        self._update_peak(pool)

    @property
    def total_bytes(self) -> int:
        """Sum of the entries: the build's net pool use so far."""
        return sum(size for _, size in self.entries)

    def as_record(self) -> dict:
        """JSON-ready {breakdown [[owner, bytes]], seconds [[owner, wall s]], total_bytes, peak_bytes, budget,
        failed_owner, kind ('gpu' pool / 'cpu' traced host bytes)} (CPU data only)."""
        return {'breakdown': [[owner, int(size)] for owner, size in self.entries],
                'seconds': [[owner, round(float(seconds), 3)] for owner, seconds in self.stage_seconds],
                'total_bytes': self.total_bytes,
                'peak_bytes': int(self.peak_bytes), 'start_free_bytes': int(self.start_free_bytes),
                'budget': None if self.budget is None else int(self.budget),
                'failed_owner': self.failed_owner, 'kind': self.kind}

    def _update_peak(self, pool):
        self.peak_bytes = max(self.peak_bytes,
                              int(pool.total_bytes()) - self.start_used_bytes - self.start_free_bytes,
                              int(pool.used_bytes()) - self.start_used_bytes)


def ledger_text(record) -> str:
    """One-line text of a MemoryLedger.as_record() dict (pure; an 'estimate' {gpu_steady, gpu_build_peak} entry, when
    present, is appended)."""
    stages = ', '.join(f"{owner} {format_bytes(size)}" + (' (failed here)' if owner == record.get('failed_owner')
                                                          else '')
                       for owner, size in record.get('breakdown', ()))
    memory = 'Host memory (traced numpy)' if record.get('kind') == 'cpu' else 'GPU memory'
    text = (f"{memory} by build stage: {stages or 'none'}; total {format_bytes(record.get('total_bytes'))}, "
            f"build peak {format_bytes(record.get('peak_bytes'))} of budget {format_bytes(record.get('budget'))}")
    estimate = record.get('estimate')
    if estimate:
        text += (f"; estimated {format_bytes(estimate.get('gpu_steady'))} kept, "
                 f"{format_bytes(estimate.get('gpu_build_peak'))} build peak")
    return text


STAGE_TIME_FLOOR = 0.05   # s: shorter build stages are left out of ledger_time_text


def ledger_time_text(record, title='Build time by stage') -> str:
    """One-line text of the 'seconds' of a MemoryLedger.as_record() dict, stages under STAGE_TIME_FLOOR left out."""
    stages = record.get('seconds') or ()
    shown = ', '.join(f'{owner} {seconds:.2f} s' for owner, seconds in stages if seconds >= STAGE_TIME_FLOOR)
    total = sum(seconds for _, seconds in stages)
    return (f"{title}: {shown or 'none'}; total {total:.2f} s (stages under {STAGE_TIME_FLOOR:g} s "
            'not listed)')


def format_bytes(n) -> str:
    """'512 B', '12.3 KiB', '45.6 MiB', '7.80 GiB' (1024 based); None -> 'n/a'."""
    if n is None:
        return 'n/a'
    n = int(n)
    size, sign = abs(n), '-' if n < 0 else ''
    if size < 1024:
        return f'{n} B'
    if size < 1024**2:
        return f'{sign}{size / 1024:.1f} KiB'
    if size < 1024**3:
        return f'{sign}{size / 1024**2:.1f} MiB'
    return f'{sign}{size / 1024**3:.2f} GiB'


# ---- helpers ----
def _exception_chain(exc):
    """[exc, its __cause__ (else __context__), ...]: at most 1 + 10 links, cycle- and None-safe."""
    chain, seen = [], set()
    link = exc
    while link is not None and id(link) not in seen and len(chain) <= _MAX_CHAIN_LINKS:
        seen.add(id(link))
        chain.append(link)
        link = link.__cause__ if link.__cause__ is not None else link.__context__
    return chain


def _is_gpu_allocation_error(exc):
    """cupy's OutOfMemoryError, or a cupy CUDA / library error whose text names an allocation failure."""
    if isinstance(exc, cp.cuda.memory.OutOfMemoryError):
        return True
    package = (type(exc).__module__ or '').split('.')[0]
    if package not in _CUPY_PACKAGES:
        return False
    text = _exception_text(exc)
    return any(marker in text for marker in _OUT_OF_MEMORY_MARKERS)


def _exception_text(exc):
    """str(exc), or '' when str() itself fails."""
    try:
        return str(exc)
    except Exception:
        return ''


def _budget_text(status):
    """'GPU budget X of Y device memory (how set_budget chose it)', or the device memory when no budget is set."""
    if status.budget is None or _budget_settings is None:
        return (f'no GPU budget set; {format_bytes(status.device_total)} device memory, '
                f'{format_bytes(status.device_free)} free')
    fraction, headroom_bytes = _budget_settings
    return (f'GPU budget {format_bytes(status.budget)} of {format_bytes(status.device_total)} device memory '
            f"(set_budget: the smaller of {fraction:.0%} of the device and its free memory plus the pool's cached "
            f'blocks minus {format_bytes(headroom_bytes)} headroom)')


def _references(item):
    """Objects item refers to directly: mapping values, container items, closure cells and defaults, attributes."""
    if isinstance(item, Mapping):
        return list(item.values())
    if isinstance(item, (list, tuple, set, frozenset, deque)):
        return list(item)
    if isinstance(item, types.FunctionType):
        children = list(item.__defaults__ or ())
        for cell in item.__closure__ or ():
            try:
                children.append(cell.cell_contents)
            except ValueError:   # empty cell
                pass
        return children
    children = []
    try:
        children.extend(vars(item).values())
    except TypeError:   # no __dict__
        pass
    for cls in type(item).__mro__:
        slots = cls.__dict__.get('__slots__', ())
        for name in (slots,) if isinstance(slots, str) else slots:
            if name in ('__dict__', '__weakref__'):
                continue
            try:
                children.append(getattr(item, name))
            except Exception:   # unset slot, or a name-mangled private slot
                pass
    return children
