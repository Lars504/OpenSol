"""Child process of the CPU solver profiler: reads one case (problem, size, solver type and values, repeats) as JSON
on stdin, assembles the problem, runs the repeats (runners) and prints one result line (RESULT_MARKER + JSON) on
stdout: per-repeat phases, residuals and details, peak RSS, status. Run by cpu_solvers as
`python -m dev_tools.time_profilers.case_runner` from the repository root."""
# Peak RSS: on Linux the kernel's high-water mark VmHWM is reset after the assembly (writing 5 to
# /proc/self/clear_refs), so rss_peak_bytes - rss_baseline_bytes is what the solver phase added (MUMPS / PARDISO /
# SuperLU memory included: RSS counts every allocation, tracemalloc would miss the C ones); elsewhere ru_maxrss of
# the whole process (imports and assembly included) and rss_solver_bytes None. Factors are released between repeats,
# so the peak is that of the largest repeat.
import gc
import json
import os
import sys
import time
import traceback

RESULT_MARKER = 'OPEN_SOL_PROFILE_RESULT '
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
_PROC_STATUS = '/proc/self/status'
_PROC_CLEAR_REFS = '/proc/self/clear_refs'


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


class PeakMemory:
    """Peak RSS of the solver phase: start() after the assembly (baseline, high-water mark reset where possible),
    finish() -> {'rss_baseline_bytes', 'rss_peak_bytes', 'rss_solver_bytes', 'rss_method'}."""

    def __init__(self):
        self.baseline = None
        self.reset = False

    def start(self) -> None:
        gc.collect()
        self.baseline = _status_bytes('VmRSS')
        try:
            with open(_PROC_CLEAR_REFS, 'w', encoding='ascii') as handle:
                handle.write('5')   # reset VmHWM to the current RSS (Linux >= 4.0)
            self.reset = self.baseline is not None
        except OSError:
            self.reset = False

    def finish(self) -> dict:
        if self.reset:
            peak = _status_bytes('VmHWM')
            if peak is not None:
                return {'rss_baseline_bytes': self.baseline, 'rss_peak_bytes': peak,
                        'rss_solver_bytes': max(peak - self.baseline, 0),
                        'rss_method': 'VmHWM reset after the assembly'}
        return {'rss_baseline_bytes': self.baseline, 'rss_peak_bytes': _status_bytes('VmHWM') or _ru_maxrss_bytes(),
                'rss_solver_bytes': None, 'rss_method': 'whole-process peak (imports and assembly included)'}


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


def run_case(case) -> dict:
    """Result record of one case dict {'problem', 'size', 'solver_type', 'values', 'repeat'}."""
    from dev_tools.time_profilers import problems, runners
    result = {'threads_env': {name: os.environ.get(name) for name in THREAD_VARIABLES}, 'cpu_count': os.cpu_count(),
              'repeats': []}
    started = time.perf_counter()
    try:
        problem = problems.build(case['problem'], case['size'])
    except Exception as error:   # recorded
        result.update(status='error', reason=f'assembly: {type(error).__name__}: {error}')
        return result
    result.update(assembly_seconds=time.perf_counter() - started, n_unknowns=problem.n_unknowns, nnz=problem.nnz)
    runner = runners.RUNNERS[case['solver_type']]
    memory = PeakMemory()
    memory.start()
    status = 'ok'
    try:
        for _ in range(int(case['repeat'])):
            repeat = runner(problem, case['values'])
            result['repeats'].append(repeat)
            if repeat['status'] != 'ok':
                status = repeat['status']
            gc.collect()
    except Exception as error:   # recorded, never re-raised: the parent reads one result line
        status = classify(error)
        result['reason'] = f'{type(error).__name__}: {error}'
        result['traceback'] = traceback.format_exc(limit=8)
    result.update(memory.finish())
    result['status'] = status
    result['blas_pools'] = _blas_threads()
    return result


def _jsonable(value):
    """json.dumps default: numpy scalars / arrays as Python values, anything else as its text."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


def main() -> int:
    case = json.loads(sys.stdin.read())
    try:
        result = run_case(case)
    except Exception as error:   # import errors and the like: still one result line
        result = {'status': 'error', 'reason': f'{type(error).__name__}: {error}',
                  'traceback': traceback.format_exc(limit=8), 'repeats': []}
    sys.stdout.write(RESULT_MARKER + json.dumps(result, default=_jsonable) + '\n')
    sys.stdout.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
