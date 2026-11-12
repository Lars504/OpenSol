"""Child process of the solver profiler: reads one case (problem, size, solver type and values, backend, repeats; or
a whole model, model_cases) as JSON on stdin, assembles the problem, runs a cold repeat and the warm repeats
(runners / model_cases) and prints one partial line per step (PARTIAL_MARKER) and one result line (RESULT_MARKER +
JSON) on stdout. Run by solver_profiler as `python -m tools_profilers.time_profilers.case_runner` from the repository root."""
# Peak RSS of the solver phase: on Linux the kernel's high-water mark VmHWM is reset after the assembly (writing 5 to
# /proc/self/clear_refs), so peak - baseline is exact. Elsewhere the peak cannot be reset (Windows: working set /
# peak working set of psapi GetProcessMemoryInfo): when the solver phase raised the process peak above the assembly's,
# peak - baseline is still exact; otherwise only the bound "at most assembly peak - baseline" is known. RSS counts
# every allocation (MUMPS / PARDISO / SuperLU included; tracemalloc would miss the C ones). Factors are released
# between repeats, so the peak is that of the largest repeat.
# Partial lines (the assembly, then each finished repeat) let the parent keep finished repeats of a case it kills.
import functools
import gc
import json
import os
import sys
import time
import traceback

RESULT_MARKER = 'OPEN_SOL_PROFILE_RESULT '
PARTIAL_MARKER = 'OPEN_SOL_PROFILE_PARTIAL '
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
MIN_WARM_SECONDS = 0.5   # warm repeats continue past the requested count until their summed wall time reaches this
MAX_REPEATS = 50         # warm repeats of a tiny case at most
_PROC_STATUS = '/proc/self/status'
_PROC_CLEAR_REFS = '/proc/self/clear_refs'
# rss_method keys (report.MEMORY_METHODS holds their legend texts)
RSS_RESET, RSS_RAISED, RSS_BOUND, RSS_PROCESS = 'reset', 'raised', 'bound', 'process'


def _status_bytes(field):
    """Bytes of a kB field of /proc/self/status (VmRSS, VmHWM), None off Linux or when unreadable."""
    try:
        with open(_PROC_STATUS, encoding='ascii') as handle:
            for line in handle:
                if line.startswith(field + ':'):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _ru_maxrss_bytes():
    """ru_maxrss of this process in bytes (Linux reports KiB, macOS bytes), None without the resource module."""
    try:
        import resource
    except ImportError:   # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak) if sys.platform == 'darwin' else int(peak) * 1024


@functools.lru_cache(maxsize=1)
def _windows_memory_reader():
    """Callable -> (working set, peak working set) bytes of this process (psapi GetProcessMemoryInfo), None off
    Windows or when the call cannot be set up."""
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    try:
        kernel32 = ctypes.WinDLL('kernel32')   # own instances: the argtypes below do not touch ctypes.windll
        psapi = ctypes.WinDLL('psapi')
    except OSError:
        return None
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    process = kernel32.GetCurrentProcess()

    def read():
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(ProcessMemoryCounters)
        if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize), int(counters.PeakWorkingSetSize)
    return read


def current_and_peak_rss():
    """(current RSS, process peak RSS) in bytes: Windows working set / peak working set, Linux VmRSS / VmHWM, else
    (None, ru_maxrss); None where unknown."""
    reader = _windows_memory_reader()
    if reader is not None:
        values = reader()
        if values is not None:
            return values
    peak = _status_bytes('VmHWM')
    return _status_bytes('VmRSS'), peak if peak is not None else _ru_maxrss_bytes()


class PeakMemory:
    """Peak RSS of the solver phase: start() after the assembly (baseline, high-water mark reset where possible),
    finish() -> {'rss_baseline_bytes', 'rss_peak_bytes', 'rss_solver_bytes' (exact or None),
    'rss_solver_bound_bytes' (upper bound when only that is known), 'rss_method' (RSS_* key)}."""

    def __init__(self):
        self.baseline = None
        self.peak_before = None
        self.reset = False

    def start(self) -> None:
        gc.collect()
        self.baseline, self.peak_before = current_and_peak_rss()
        try:
            with open(_PROC_CLEAR_REFS, 'w', encoding='ascii') as handle:
                handle.write('5')   # reset VmHWM to the current RSS (Linux >= 4.0)
            self.reset = self.baseline is not None
        except OSError:
            self.reset = False

    def finish(self) -> dict:
        _, peak = current_and_peak_rss()
        record = {'rss_baseline_bytes': self.baseline, 'rss_peak_bytes': peak, 'rss_solver_bytes': None,
                  'rss_solver_bound_bytes': None, 'rss_method': RSS_PROCESS}
        if self.baseline is None or peak is None:
            return record
        if self.reset:
            record.update(rss_solver_bytes=max(peak - self.baseline, 0), rss_method=RSS_RESET)
        elif self.peak_before is not None and peak > self.peak_before:
            record.update(rss_solver_bytes=max(peak - self.baseline, 0), rss_method=RSS_RAISED)
        elif self.peak_before is not None:
            record.update(rss_solver_bound_bytes=max(self.peak_before - self.baseline, 0), rss_method=RSS_BOUND)
        return record


def _blas_threads():
    """threadpoolctl's view of the loaded BLAS / OpenMP pools (None when threadpoolctl is not installed)."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    try:
        return [{key: pool.get(key) for key in ('user_api', 'internal_api', 'num_threads', 'version')}
                for pool in threadpool_info()]
    except Exception:   # informational only
        return None


def classify(error) -> str:
    """Status of a solver exception: refused_memory (a MUMPS / PARDISO memory refusal), out_of_memory (MemoryError),
    refused (ValueError: size rules, bandwidth, not positive definite, a missing wrapper), failed (RuntimeError:
    singular, accuracy, CG breakdown), error (anything else)."""
    from open_sol.solvers.host_direct import HostDirectMemoryError
    if isinstance(error, HostDirectMemoryError):
        return 'refused_memory'
    if isinstance(error, MemoryError):
        return 'out_of_memory'
    if isinstance(error, ValueError):
        return 'refused'
    if isinstance(error, RuntimeError):
        return 'failed'
    return 'error'


def _jsonable(value):
    """json.dumps default: numpy scalars / arrays as Python values, anything else as its text."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def _emit(marker, payload) -> None:
    sys.stdout.write(marker + json.dumps(payload, default=_jsonable) + '\n')
    sys.stdout.flush()


def _timed_repeat(repeat_function, cold) -> dict:
    """One repeat with its whole wall and process CPU time (all threads; factor, solve, refactor, residual; a model
    case: the whole api.run)."""
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    repeat = repeat_function()
    repeat['wall_seconds'] = time.perf_counter() - wall_started
    repeat['cpu_seconds'] = time.process_time() - cpu_started
    repeat['cold'] = cold
    return repeat


def _prepare_problem(case):
    """(assembly fields, repeat()) of a synthetic case: the problem assembled on the host and moved to the backend."""
    from tools_profilers.time_profilers import problems, runners
    started = time.perf_counter()
    problem = problems.build(case['problem'], case['size'])
    assembled = time.perf_counter()
    arrays = runners.backend_arrays(problem, case.get('backend', 'cpu'))
    fields = {'assembly_seconds': assembled - started, 'transfer_seconds': time.perf_counter() - assembled,
              'n_unknowns': problem.n_unknowns, 'nnz': problem.nnz}
    runner = runners.RUNNERS[case['solver_type']]
    return fields, lambda: runner(problem, case['values'], arrays)


def run_case(case) -> dict:
    """Result record of one case dict {'problem', 'size', 'solver_type', 'values', 'backend', 'repeat', 'warmup'}
    (kind 'model': model_cases.prepare's keys): a cold repeat first (warmup), then at least `repeat` warm ones, more
    while their summed wall time is below MIN_WARM_SECONDS."""
    from tools_profilers.time_profilers import model_cases
    result = {'threads_env': {name: os.environ.get(name) for name in THREAD_VARIABLES}, 'cpu_count': os.cpu_count(),
              'backend': case.get('backend', 'cpu'), 'repeats': []}
    try:
        assembly, repeat_function = (model_cases.prepare(case) if case.get('kind') == 'model'
                                     else _prepare_problem(case))
    except Exception as error:   # recorded
        result.update(status='error', reason=f'assembly: {type(error).__name__}: {error}',
                      traceback=traceback.format_exc(limit=8))
        return result
    if result['backend'] == 'gpu':   # the device of the run (the parent makes no CUDA context)
        try:
            from open_sol.core.gpu_memory import device_info
            info = device_info()
            assembly['gpu'] = {'name': info.name, 'total_bytes': info.total_bytes}
        except Exception:   # informational only
            pass
    result.update(assembly)
    _emit(PARTIAL_MARKER, {'kind': 'assembly', 'fields': assembly})
    wanted = int(case['repeat'])
    pending_cold = bool(case.get('warmup', True))
    memory = PeakMemory()
    memory.start()
    status = 'ok'
    warm_count, warm_seconds = 0, 0.0
    try:
        while warm_count < wanted or (warm_seconds < MIN_WARM_SECONDS and warm_count < MAX_REPEATS):
            repeat = _timed_repeat(repeat_function, pending_cold)
            result['repeats'].append(repeat)
            _emit(PARTIAL_MARKER, {'kind': 'repeat', 'repeat': repeat})
            if repeat['status'] != 'ok':
                status = repeat['status']
                if repeat.get('reason'):
                    result['reason'] = repeat['reason']
                if status == 'failed':   # a failed model run fails again: no more repeats
                    break
            if not pending_cold:
                warm_count += 1
                warm_seconds += repeat['wall_seconds']
            pending_cold = False
            gc.collect()
    except Exception as error:   # recorded, never re-raised: the parent reads one result line
        status = classify(error)
        result['reason'] = f'{type(error).__name__}: {error}'
        result['traceback'] = traceback.format_exc(limit=8)
    result.update(memory.finish())
    result['status'] = status
    result['blas_pools'] = _blas_threads()
    return result


def main() -> int:
    case = json.loads(sys.stdin.read())
    try:
        result = run_case(case)
    except Exception as error:   # import errors and the like: still one result line
        result = {'status': 'error', 'reason': f'{type(error).__name__}: {error}',
                  'traceback': traceback.format_exc(limit=8), 'repeats': []}
    _emit(RESULT_MARKER, result)
    return 0


if __name__ == '__main__':
    sys.exit(main())
