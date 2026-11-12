"""Machine profile of the run-time estimate (time_estimation_plan section 3): microbenchmarks per backend, stored as
measured (size, seconds) curves in the user's cache folder and priced by log-log interpolation. Loading is GUI-thread
safe (json only); run_profile runs the benchmarks (gpu: on the GPU-owning thread)."""
import datetime
import hashlib
import importlib.util
import json
import math
import os
import platform
import time
from dataclasses import dataclass, field

PROFILE_DIR_VARIABLE = 'OPEN_SOL_PROFILE_DIR'   # overrides the folder (default <repository>/time_profiles)
PROFILE_FOLDER = 'time_profiles'                # next to the open_sol package (git-ignored: machine-specific)
PROFILE_VERSION = 1
QUICK_SCALE = 0.1          # --quick: largest benchmark sizes x this (curves still span their decades)
CUDSS_DISTRIBUTIONS = ('nvidia-cudss-cu13', 'nvidia-cudss-cu12')   # cuDSS wheels nvmath loads (profile versions)
CUDSS_AXES = (64, 128, 256, 512)   # cells per axis of the 2D 5-point cuDSS benchmarks (quick: the first three; larger
                                   # sizes extrapolate along the 256 -> 512 slope, at least linear)


@dataclass(frozen=True)
class MachineProfile:
    """Seconds of the op catalogue's primitives on one backend: curves {name: ((size, seconds), ...)} sorted by size,
    scalars {name: seconds} ('launch', 'sync', 'python_call'), what they were measured on, and the wall seconds each
    profiling task took (task_seconds {progress text: seconds}; compared on the next profile run). outdated: loaded
    although load_profile rejects it (other benchmark code / library versions; never stored)."""
    backend: str
    device: str
    date: str
    code_hash: str
    versions: dict
    curves: dict = field(default_factory=dict)
    scalars: dict = field(default_factory=dict)
    quick: bool = False
    task_seconds: dict = field(default_factory=dict)
    outdated: bool = False

    @property
    def key(self) -> str:
        return f'{self.backend}: {self.device}'

    def has(self, name) -> bool:
        return name in self.curves or name in self.scalars

    def scalar(self, name) -> float:
        if name not in self.scalars:
            raise KeyError(name)
        return float(self.scalars[name])

    def seconds(self, name, size) -> float:
        """Seconds at `size` on curve `name` by log-log interpolation; below the first point its seconds (latency), above
        the last the slope of the last segment (at least linear)."""
        if name not in self.curves:
            raise KeyError(name)
        points = self.curves[name]
        size = float(size)
        if size <= points[0][0]:
            return float(points[0][1])
        if len(points) == 1:
            return float(points[0][1]) * size / points[0][0]
        for (size_low, seconds_low), (size_high, seconds_high) in zip(points, points[1:]):
            if size <= size_high:
                break
        slope = math.log(seconds_high / seconds_low) / math.log(size_high / size_low)
        if size > size_high:
            slope = max(slope, 1.0)
        return seconds_low * (size / size_low) ** slope

    def record(self) -> dict:
        """{key, backend, date, quick, outdated} (check.json / summary.json; the run keeps no curves)."""
        return {'key': self.key, 'backend': self.backend, 'date': self.date, 'quick': self.quick,
                'outdated': self.outdated}

    def as_dict(self) -> dict:
        return {'version': PROFILE_VERSION, 'backend': self.backend, 'device': self.device, 'date': self.date,
                'code_hash': self.code_hash, 'versions': self.versions, 'quick': self.quick,
                'curves': {name: [list(point) for point in points] for name, points in self.curves.items()},
                'scalars': dict(self.scalars), 'task_seconds': dict(self.task_seconds)}


def profile_dir() -> str:
    package_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.environ.get(PROFILE_DIR_VARIABLE) or os.path.join(package_parent, PROFILE_FOLDER)


def profile_path(backend) -> str:
    return os.path.join(profile_dir(), f'time_profile_{backend}.json')


# the code the curves time beside this module: the cuDSS curves run solvers.sparse_direct / cudss_factor, so a change
# there (host / device routes, refresh kernels, probes) outdates them too (any edit of those files does)
_TIMED_SOURCES = (os.path.join('..', 'solvers', 'sparse_direct.py'), os.path.join('..', 'solvers', 'cudss_factor.py'))


def code_hash() -> str:
    """Hash of this module's source and of the solver sources its curves time (_TIMED_SOURCES): a changed benchmark or
    timed solver invalidates stored profiles."""
    digest = hashlib.sha1()
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.abspath(__file__), *(os.path.normpath(os.path.join(here, name)) for name in _TIMED_SOURCES)):
        with open(path, 'rb') as source:
            digest.update(source.read())
    return digest.hexdigest()[:12]


def library_versions(backend) -> dict:
    """numpy / scipy versions + numba on the cpu (its twins run only there, 'fused_numba') / cupy, nvmath and the cuDSS
    wheel on the gpu (the cudss_* curves); no driver call."""
    import numpy
    versions = {'python': platform.python_version(), 'numpy': numpy.__version__}
    for name in ('scipy',) + (('cupy', 'nvmath') if backend == 'gpu' else ('numba',)):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    if backend == 'gpu':
        versions['cudss'] = _cudss_wheel_version()
    return versions


def _cudss_wheel_version():
    """Version of the installed nvidia-cudss wheel (package metadata only, nothing loaded), None without one."""
    from importlib import metadata
    for distribution in CUDSS_DISTRIBUTIONS:
        try:
            return f'{distribution} {metadata.version(distribution)}'
        except metadata.PackageNotFoundError:
            continue
    return None


def load_profile(backend):
    """(MachineProfile or None, reason): None when there is no file, it is unreadable, or its benchmark code / library
    versions differ from this installation (re-run `python -m open_sol profile`; the reason names the changed
    libraries, e.g. numba installed after the cpu profile)."""
    path = profile_path(backend)
    if not os.path.exists(path):
        return None, f'no machine profile for the {backend} backend'
    try:
        with open(path, encoding='utf-8') as handle:
            data = json.load(handle)
        if data.get('version') != PROFILE_VERSION or data.get('code_hash') != code_hash():
            return None, f'the {backend} machine profile is out of date (measured by other benchmark code)'
        stored, installed = data.get('versions') or {}, library_versions(backend)
        if stored != installed:
            changed = ', '.join(f'{name} {stored.get(name)} -> {installed.get(name)}'
                                for name in sorted(set(stored) | set(installed)) if stored.get(name) != installed.get(name))
            return None, f'the {backend} machine profile is out of date (measured with other library versions: {changed})'
        return profile_from_dict(data), ''
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, f'the machine profile {path} is unreadable ({exc})'


def profile_from_dict(data) -> MachineProfile:
    """MachineProfile of an as_dict record (no version checks)."""
    return MachineProfile(data['backend'], data['device'], data['date'], data['code_hash'], data['versions'],
                          {name: tuple(tuple(point) for point in points) for name, points in data['curves'].items()},
                          dict(data['scalars']), bool(data.get('quick')), dict(data.get('task_seconds') or {}))


def _previous_task_seconds(backend):
    """({task: seconds}, 'date[, quick]') of the stored profile file, whatever its code / versions; ({}, '') without."""
    try:
        with open(profile_path(backend), encoding='utf-8') as handle:
            data = json.load(handle)
        return dict(data.get('task_seconds') or {}), f"{data.get('date')}{', quick' if data.get('quick') else ''}"
    except (OSError, ValueError, TypeError, AttributeError):
        return {}, ''


def _task_line(name, seconds, previous_seconds, previous_note=''):
    previous = '-' if previous_seconds is None else f'{previous_seconds:.3g} s'
    return f"{name}: {seconds:.3g} s (previous {previous}{f' on {previous_note}' if previous_note else ''})"


def save_profile(profile) -> str:
    os.makedirs(profile_dir(), exist_ok=True)
    path = profile_path(profile.backend)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(profile.as_dict(), handle, indent=1)
    return path


# --- benchmarks ---------------------------------------------------------------------------------------------------

MAX_BATCH_CALLS = 1 << 16


def _best_seconds(run, synchronize, repeats=5, minimum_seconds=0.02):
    """Best seconds of one run() call: a doubling probe batch to minimum_seconds / 8 sizes the batch to last about
    minimum_seconds (no x4 jumps in wall time), then best of the probe and `repeats` - 1 such batches."""
    def batch_seconds(calls):
        start = time.perf_counter()
        for _ in range(calls):
            run()
        synchronize()
        return time.perf_counter() - start

    run()
    synchronize()
    calls = 1
    while True:
        elapsed = batch_seconds(calls)
        if elapsed >= minimum_seconds / 8 or calls >= MAX_BATCH_CALLS:
            break
        calls *= 2
    best = elapsed / calls
    batch_calls = min(max(math.ceil(minimum_seconds / max(best, 1e-12)), 1), MAX_BATCH_CALLS)
    for _ in range(repeats - 1):
        best = min(best, batch_seconds(batch_calls) / batch_calls)
    return best


def _sizes(smallest, largest):
    """Log-spaced sizes (x ~3.16) from smallest to largest, both included."""
    sizes, size = [], float(smallest)
    while size < largest * 0.99:
        sizes.append(int(size))
        size *= math.sqrt(10.0)
    return sizes + [int(largest)]


def _device_name(backend):
    if backend == 'cpu':
        return platform.processor() or platform.machine() or 'cpu'
    from .backend import cp
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    name = properties['name']
    return name.decode() if isinstance(name, bytes) else str(name)


def _fused_kernel(backend, xp, use_numba=False):
    """run(arrays) of the 5-array fused benchmark (3 read, 2 written) on the backend: cupy ElementwiseKernel, numpy
    expressions with temporaries (as the numpy twins), or a parallel numba loop."""
    if backend == 'gpu':
        kernel = xp.ElementwiseKernel('float32 a, float32 b, float32 c', 'float32 d, float32 e',
                                      'd = a * b + c; e = a - b * c', 'open_sol_profile_fused')
        return lambda a, b, c, d, e: kernel(a, b, c, d, e)
    if use_numba:
        import numba

        @numba.njit(parallel=True, cache=False)
        def fused(a, b, c, d, e):
            for index in numba.prange(a.size):
                d[index] = a[index] * b[index] + c[index]
                e[index] = a[index] - b[index] * c[index]
        return fused

    def fused_numpy(a, b, c, d, e):
        d[...] = a * b + c
        e[...] = a - b * c
    return fused_numpy


ARITHMETIC_STEPS = 512   # FMA pairs per element of the arithmetic benchmark (4 flops each: compute-bound, ~12 B moved)


def _arithmetic_kernel(xp):
    """run(a, b, out) of the gpu arithmetic benchmark: two dependent FMA chains of ARITHMETIC_STEPS per element, in the
    arrays' real type (bounded values: 0.25 is a fixed point)."""
    kernel = xp.ElementwiseKernel('T a, T b', 'T c', f'''
        T x = a, y = b;
        for (int step = 0; step < {ARITHMETIC_STEPS}; ++step) {{ x = x * y + a; y = y * x + b; }}
        c = x + y;''', 'open_sol_profile_arithmetic')
    return lambda a, b, out: kernel(a, b, out)


def _device_csr_product(matrix):
    """vector -> A vector of the pcg gpu CSR kernel (the solvers' product, into one owned vector)."""
    from ..solvers.pcg import DeviceCsrProduct
    return DeviceCsrProduct(matrix)


def _laplacian(n_per_axis, dimensions, dtype):
    """Host scipy CSR of a shifted 2D 5-point / 3D 7-point Laplacian (complex: an imaginary shift, as Helmholtz)."""
    import numpy as np
    import scipy.sparse
    line = scipy.sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(n_per_axis, n_per_axis))
    identity = scipy.sparse.identity(n_per_axis)
    if dimensions == 2:
        operator = scipy.sparse.kron(line, identity) + scipy.sparse.kron(identity, line)
    else:
        operator = (scipy.sparse.kron(scipy.sparse.kron(line, identity), identity)
                    + scipy.sparse.kron(scipy.sparse.kron(identity, line), identity)
                    + scipy.sparse.kron(scipy.sparse.kron(identity, identity), line))
    shift = 0.1j if np.dtype(dtype).kind == 'c' else 0.1
    operator = operator + shift * scipy.sparse.identity(operator.shape[0])
    return operator.tocsr().astype(dtype)


def run_profile(backend, quick=False, progress=None, save=True) -> MachineProfile:
    """Measure the profile of the backend ('cpu' / 'gpu'; ~30 s, + ~10 s of cuDSS curves with nvmath; quick ~5 s) and
    save it (save False: a self-test keeps the stored one); progress(text) per finished task with its wall seconds and
    those of the stored profile, then the totals."""
    import numpy as np
    from .backend import array_module, synchronize as backend_synchronize
    from .kernels import numba_available
    xp = array_module(backend)
    previous_task_seconds, previous_note = _previous_task_seconds(backend)
    task_seconds, running_task = {}, {}

    def synchronize():
        backend_synchronize(backend)

    def finish_task():
        if running_task:
            name, started = running_task.pop('name'), running_task.pop('started')
            task_seconds[name] = time.perf_counter() - started
            if progress is not None:
                progress(_task_line(name, task_seconds[name], previous_task_seconds.get(name)))

    def report(text):
        finish_task()
        running_task.update(name=text, started=time.perf_counter())

    scale = QUICK_SCALE if quick else 1.0
    curves, scalars = {}, {}
    largest = int((1e8 if backend == 'gpu' else 2e7) * scale)

    report('launch / sync latency')
    fused = _fused_kernel(backend, xp)
    tiny_arrays = [xp.zeros(1, dtype=np.float32) for _ in range(5)]
    launch = _best_seconds(lambda: fused(*tiny_arrays), lambda: None, minimum_seconds=0.05)
    launch_sync = _best_seconds(lambda: (fused(*tiny_arrays), synchronize()), lambda: None, minimum_seconds=0.05)
    scalars['launch'] = launch
    scalars['sync'] = max(launch_sync - launch, 0.0) if backend == 'gpu' else 0.0

    def call(a, b, c, d, e):
        return a
    scalars['python_call'] = _best_seconds(lambda: call(1, 2, 3, 4, 5), lambda: None, minimum_seconds=0.05)

    def fused_curve(kernel):
        points = []
        for elements in _sizes(1e3, largest):
            arrays = [xp.full(elements, 1.0 + index, dtype=np.float32) for index in range(5)]
            points.append((5.0 * 4 * elements, _best_seconds(lambda: kernel(*arrays), synchronize)))
            del arrays
        return tuple(points)

    report('fused elementwise kernel')
    curves['fused'] = fused_curve(fused)
    if backend == 'cpu' and numba_available():
        report('fused elementwise kernel (numba)')
        try:
            curves['fused_numba'] = fused_curve(_fused_kernel(backend, xp, use_numba=True))
        except Exception:   # numba typing / compile failure: the engines fall back to numpy too
            pass

    if backend == 'gpu':   # arithmetic-bound kernels (Elementwise.flops): float64 runs at 1/32-1/64 rate on consumer GPUs
        arithmetic = _arithmetic_kernel(xp)
        for dtype, bits in ((np.float32, 32), (np.float64, 64)):
            report(f'float{bits} arithmetic')
            points = []
            for elements in _sizes(1e3, int(1e6 * scale)):
                a, b = xp.full(elements, 0.25, dtype=dtype), xp.full(elements, 0.25, dtype=dtype)
                out = xp.empty(elements, dtype=dtype)
                points.append((4.0 * ARITHMETIC_STEPS * elements,
                               _best_seconds(lambda: arithmetic(a, b, out), synchronize)))
                del a, b, out
            curves[f'arith_float{bits}'] = tuple(points)

    report('dot')
    # gpu: solvers.pcg's multi-block vector_dot (a ReductionKernel runs on one block: ~8x slower at 1e6 values, so the
    # te.Reduction of other one-block reductions is underpriced by this curve); cpu: np.dot
    if backend == 'gpu':
        from ..solvers.pcg import device_vector_dot as dot
    else:
        dot = np.dot
    points = []
    for elements in _sizes(1e3, largest):
        a, b = xp.ones(elements, dtype=np.float64), xp.ones(elements, dtype=np.float64)
        points.append((2.0 * 8 * elements, _best_seconds(lambda: dot(a, b), synchronize)))
        del a, b
    curves['dot'] = tuple(points)

    report('sparse matrix-vector product')
    from .backend import sparse_module
    sparse = sparse_module(backend)
    points = []
    for n_per_axis in (12, 26, 56, 120, 160) if not quick else (12, 26, 56):
        host = _laplacian(n_per_axis, 3, np.float64)
        matrix = sparse.csr_matrix(host) if backend == 'gpu' else host
        vector = xp.ones(host.shape[0], dtype=np.float64)
        rows, nnz = host.shape[0], host.nnz
        # gpu: the solvers' own CSR RawKernel (pcg / multigrid products), not cupyx's per-call descriptor setup
        product = _device_csr_product(matrix) if backend == 'gpu' else matrix.dot
        points.append((nnz * 12.0 + rows * 20.0, _best_seconds(lambda: product(vector), synchronize)))
        del matrix, vector, host, product
    curves['spmv'] = tuple(points)

    if backend == 'gpu':
        report('host <-> device transfer')
        points = []
        for elements in _sizes(1e3, largest // 4):
            host = np.ones(elements, dtype=np.float32)
            device = xp.asarray(host)
            seconds = 0.5 * (_best_seconds(lambda: xp.asarray(host), synchronize)
                             + _best_seconds(lambda: device.get(), synchronize))
            points.append((4.0 * elements, seconds))
            del host, device
        curves['transfer'] = tuple(points)

    def gemm_curve(module, sync):
        points = []
        for batch in (1, 8, 64, 512) if not quick else (1, 16, 128):
            blocks = module.ones((batch, 64, 64), dtype=np.float64)
            points.append((2.0 * batch * 64 ** 3, _best_seconds(lambda: module.matmul(blocks, blocks), sync)))
            del blocks
        return tuple(points)

    report('dense matrix products')
    curves['host_gemm'] = gemm_curve(np, lambda: None)
    curves['gemm'] = gemm_curve(xp, synchronize) if backend == 'gpu' else curves['host_gemm']

    report('host stream')
    points = []
    for elements in _sizes(1e3, int(2e7 * scale)):
        source, target = np.ones(elements), np.empty(elements)
        points.append((16.0 * elements, _best_seconds(lambda: np.copyto(target, source), lambda: None)))
        del source, target
    curves['host_stream'] = tuple(points)

    from ..solvers.sparse_direct import _host_superlu   # the FDFD / diffusion SuperLU options
    for dimensions, axes, dtype, suffix in ((2, (64, 128, 256, 512), np.float64, ''),
                                            (2, (64, 128, 256), np.complex128, '_complex'),
                                            (3, (10, 15, 22, 30), np.float64, '')):
        report(f'host SuperLU {dimensions}D{suffix.replace("_", " ")}')
        lu_points, solve_points = [], []
        for n_per_axis in axes[:-1] if quick else axes:
            csc = _laplacian(n_per_axis, dimensions, dtype).tocsc()
            unknowns = csc.shape[0]
            kept = {}

            def factorize():
                kept['factor'] = _host_superlu(csc)
            # a warm-up call (first-call imports / allocations), then the best of 2
            lu_points.append((float(unknowns), _best_seconds(factorize, lambda: None, repeats=2, minimum_seconds=0.0)))
            factor = kept['factor']
            rhs = np.ones(unknowns, dtype=dtype)
            solve_points.append((float(unknowns), _best_seconds(lambda: factor.solve(rhs), lambda: None, repeats=3)))
            del factor, csc
        curves[f'host_lu_{dimensions}d{suffix}'] = tuple(lu_points)
        curves[f'host_solve_{dimensions}d{suffix}'] = tuple(solve_points)

    if backend == 'gpu' and importlib.util.find_spec('nvmath') is not None:   # = cudss_factor.cudss_available()
        curves.update(_cudss_curves(quick, report))
    finish_task()
    if progress is not None:
        progress(_task_line('total', sum(task_seconds.values()),
                            sum(previous_task_seconds.values()) if previous_task_seconds else None, previous_note))

    profile = MachineProfile(backend, _device_name(backend), datetime.date.today().isoformat(), code_hash(),
                             library_versions(backend), _monotone(curves), scalars, bool(quick), task_seconds)
    if save:
        save_profile(profile)
    return profile


def _cudss_curves(quick, report) -> dict:
    """cuDSS 2D curves on complex symmetric 5-point Laplacians (+0.1j), called as the engines do
    (sparse_direct.cudss_factorization, symmetric): plan, refactorisation on the kept plan (host path, and the device
    path of SparseDirectSolver.factorize_pencil: '_device'), solve_host; {} on failure."""
    import numpy as np
    from ..solvers.sparse_direct import DevicePencil, SparseDirectSolver, cudss_factorization
    curves = {}
    try:
        for back_end, suffix in (('cudss', ''), ('cudss_mixed', '_mixed')):
            report(f"cuDSS 2D{' mixed' if suffix else ''}")
            plan_points, factor_points, device_points, solve_points = [], [], [], []
            pencil_solver = SparseDirectSolver(back_end, backend='gpu')
            for n_per_axis in CUDSS_AXES[:3] if quick else CUDSS_AXES:
                host = _laplacian(n_per_axis, 2, np.complex128)
                host.sum_duplicates()   # canonical: the kept plan's pattern is the pencil's
                unknowns = float(host.shape[0])
                factor, plans = None, []
                # fresh factors re-plan: the first warms up (library load), the plan curve takes the best of 2 (double)
                for _ in range(2 if not suffix else 1):
                    if factor is not None:
                        factor.release()
                    factor = cudss_factorization(host, back_end, symmetric=True)
                    plans.append(factor.plan_seconds)
                if not suffix:
                    plan_points.append((unknowns, min(plans)))
                kept = {'factor': factor}

                def refactorize():
                    kept['factor'] = cudss_factorization(host, back_end, previous=kept['factor'], symmetric=True)
                factor_points.append((unknowns, _best_seconds(refactorize, lambda: None, repeats=2,
                                                              minimum_seconds=0.0)))
                # device path on the same plan: terms (Laplacian, identity) summed with (1, 0.1j) on the device
                identity = (np.repeat(np.arange(host.shape[0]), np.diff(host.indptr)) == host.indices).astype(np.float64)
                pencil = DevicePencil(host.indptr, host.indices, [host.data.real, identity], symmetric=True)

                def refactorize_device():
                    kept['factor'] = pencil_solver.factorize_pencil(pencil, (1.0, 0.1j), previous=kept['factor'])
                device_points.append((unknowns, _best_seconds(refactorize_device, lambda: None, repeats=2,
                                                              minimum_seconds=0.0)))
                del pencil
                rhs = np.ones(host.shape[0], dtype=np.complex128)
                solve_points.append((unknowns, _best_seconds(lambda: kept['factor'].solve_host(rhs), lambda: None,
                                                             repeats=3)))
                kept['factor'].release()   # its device memory lies outside the cupy pool
            if plan_points:
                curves['cudss_plan_2d'] = tuple(plan_points)
            curves[f'cudss_factor_2d{suffix}'] = tuple(factor_points)
            curves[f'cudss_factor_2d_device{suffix}'] = tuple(device_points)
            curves[f'cudss_solve_2d{suffix}'] = tuple(solve_points)
    except Exception:   # cuDSS missing a library / out of device memory: its items stay not estimated
        return {}
    return curves


def _monotone(curves):
    """Curves with seconds made non-decreasing in size by clamping each point down to the next larger one (timer
    noise and one-off overheads only ever lengthen a measurement, so a slow small point never lifts the larger ones)."""
    result = {}
    for name, points in curves.items():
        fixed, running = [], math.inf
        for size, seconds in reversed(sorted(points)):
            running = max(min(running, float(seconds)), 1e-9)
            fixed.append((float(size), running))
        result[name] = tuple(reversed(fixed))
    return result


def profile_text(profile) -> str:
    """A few lines naming the device and headline numbers (launch, sync, stream bandwidth, gpu float32 / float64
    arithmetic rates)."""
    stream = profile.curves.get('fused', ())
    bandwidth = stream[-1][0] / stream[-1][1] / 1e9 if stream else float('nan')
    rates = ''.join(f', float{bits} {points[-1][0] / points[-1][1] / 1e9:.3g} GFLOP/s'
                    for bits in (32, 64) for points in (profile.curves.get(f'arith_float{bits}', ()),) if points)
    return (f'{profile.key} ({profile.date}{", quick" if profile.quick else ""}): launch '
            f"{profile.scalars.get('launch', 0.0) * 1e6:.3g} us, sync {profile.scalars.get('sync', 0.0) * 1e6:.3g} us, "
            f'fused-kernel bandwidth {bandwidth:.3g} GB/s{rates}')
