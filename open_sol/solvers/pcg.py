"""'pcg' LinearSolver runtime: preconditioned CG on an assembled CSR or matrix-free operator, device-side scalars, the
residual read on the host every check_interval iterations; 'mixed' = float32 CG corrections (a float32 CSR copy, or
the caller's float32 matrix-free callbacks) refined against the float64 residual. Backend 'gpu' (cupy / cupyx CSR,
fused kernels built lazily, opt-in CUDA graphs of whole check blocks) or 'cpu' (numpy / scipy CSR, the numpy twins of
the kernels, BLAS axpy and scipy's CSR kernel into preallocated vectors, numba loops on short vectors); cupy and scipy
only inside functions."""
import math
import time
import warnings

import numpy as np

BACKENDS = ('gpu', 'cpu')

CG_CHECK_INTERVAL = 16                 # iterations between host reads of the residual norm
# ... with the multigrid preconditioner: 10-40 iterations per solve, each a V-cycle far dearer than the host read, so
# a 16-block overshot by ~8 iterations (20-40 % of a solve)
MULTIGRID_CHECK_INTERVAL = 2
# cpu backend: no sync, the read costs one dot (~5 % of a Jacobi iteration): a short block keeps the overshoot of
# warm-started solves (10-30 iterations) small without paying a dot per iteration on long cold solves; multigrid 1
CPU_CG_CHECK_INTERVAL = 4
# precision -> dtype of rhs / solution / operator (the system the physics assembles); 'mixed' iterates in float32
PRECISION_DTYPES = {'single': np.float32, 'double': np.float64, 'mixed': np.float64}
# dtype-name precision of the script constructors (DiffusionSystem, ElasticSystem, the physics classes)
DTYPE_PRECISIONS = {'float32': 'single', 'float64': 'double'}
SOLVER_DTYPE_NAMES = {'single': 'float32', 'double': 'float64', 'mixed': 'float64'}   # system dtype per precision
FLOAT32_TOLERANCE_FLOOR = 1e-6         # single-precision CG stalls near this relative residual
# mixed (one float32 CG with residual replacement): the float64 residual replaces the float32 recurrence one whenever
# that fell by this factor since the last replacement ("reliable updates"; the usual single / double choice keeps the
# recurrence drift far above the float32 floor); a replacement that does not shrink ||r|| by MIXED_STALL_RATIO while
# the recurrence did stops the solve (float32 operator floor reached)
MIXED_REPLACEMENT_RATIO = 0.1
MIXED_STALL_RATIO = 0.5
# a replacement keeps the direction only while the true residual norm is within this factor of the recurrence one;
# beyond it CG restarts from the true residual (the pre-replacement scheme, always safe)
MIXED_DRIFT_LIMIT = 2.0
CG_ITERATIONS_PER_AXIS_CELL = 10       # Jacobi-PCG cold start: ~10 iterations per cell along the longest window axis
AUTO_CG_MAX_ITERATIONS_FLOOR = 5000
# run-time estimate, a priori (never fitted to run counters: a deviation is a lead on the solver or the model):
# multigrid PCG reduces the residual by this factor per iteration whatever the mesh (textbook smoothed aggregation
# V(1,1) on a diffusion operator), so a cold solve takes ln(1 / tol) / ln(1 / factor) iterations; Jacobi PCG takes the
# CG bound sqrt(kappa) ln(2 / tol) / 2 of the condition number its physics estimates (SolverWork.jacobi_condition),
# at most SolverWork.distinct_modes (finite termination). Guess: warm-started solves (time steps, fixed-point and
# nonlinear re-solves) as a fraction of a cold one (iterations and mixed residual replacements)
MULTIGRID_RESIDUAL_REDUCTION = 0.2
WARM_START_ITERATION_FRACTION = 0.25
# Stop reference of a solve: 'rhs' = ||b|| (steady solves: cheap warm-started re-solves); 'initial_residual' = ||b -
# A x0|| of the warm start (implicit time steps: x0 = previous step, so the tolerance is relative to the step's change;
# against ||b|| the capacity term C/dt u_old dominates and CG stops before the step moves: PHY-1)
STOP_REFERENCES = ('rhs', 'initial_residual')
ROUNDOFF_FLOOR_EPSILONS = 16           # 'initial_residual' never asks for less than this x eps(dtype) x ||b||

# 'multigrid' on an assembled CSR binds the Jacobi inverse diagonal where a hierarchy cannot pay (counter
# multigrid_jacobi_binds; residual read every CG_CHECK_INTERVAL): (a) the setup found no strong link on a fine level
# above multigrid.COARSEST_DENSE_LIMIT (C/dt- or loss-dominated rows: negative couplings < FLOOR_ROW_SUM_RATIO a_ii, so
# Jacobi kappa <= ~3; the hierarchy would be 8 l1-Jacobi sweeps of the full operator per iteration); (b) fewer than
# MULTIGRID_MIN_UNKNOWNS unknowns on the solver's first factorize, where one setup costs more than a Jacobi solve: no
# setup there, the hierarchy is built at the binding's second solve (a fixed operator solved many times) or at the next
# factorize (sweeps, nonlinear loops: then rebound, a rebind + solve is ~4x below a Jacobi solve). A Jacobi solve still
# open after CG_ITERATIONS_PER_AXIS_CELL x sqrt(n) iterations (more than a uniform 2D mesh needs: graded cells,
# contrasts) continues on the hierarchy, and that solver binds no Jacobi again.
# MULTIGRID_MIN_UNKNOWNS: the Jacobi / multigrid crossover of setup + one solve (solver_profiler heat_2d, gpu, warm:
# Jacobi 13.1 / multigrid 16.5 ms at 4096 unknowns, 21.1 / 18.0 ms at 1e4: crossover ~6.5e3), kept below it so the
# outer-graded heated-dot selftest (~5.6e3 cells) keeps testing the hierarchy (<= 3 ms per solve lost up to 6.5e3)
MULTIGRID_JACOBI_BINDS = True          # False: always the hierarchy
MULTIGRID_MIN_UNKNOWNS = 4000
# A same-pattern factorize of a 'multigrid' CSR keeps the earlier hierarchy (lagged setup; counter
# multigrid_lagged_binds): level 0 takes the new operator and its smoother weights (the V-cycle stays SPD), while the
# last solve on it took at most LAGGED_SETUP_ITERATION_RATIO x the first solve after its setup. A kept solve stops at
# that ratio x the fresh count + LAGGED_SETUP_SLACK and continues from its iterate on a fresh setup (a setup costs
# ~100 V-cycle iterations, so up to 2x a solve's iterations is still cheaper)
LAGGED_MULTIGRID_SETUP = True          # False: a setup per factorize
LAGGED_SETUP_ITERATION_RATIO = 2.0
LAGGED_SETUP_SLACK = 4

# gpu, callers opting in (factorize capture_blocks): each full check block (no host sync inside: alpha / beta / rho are
# device scalars, every launch a RawKernel or cupy kernel) runs once eagerly per (operator, solution, length), then as
# one captured CUDA graph (speed plan A3: 8 launches of ~12 us per block-Jacobi iteration were the whole iteration at
# 6e3 nodes). Graphs hold the callbacks' array pointers: the caller drops them (release_block_graphs) whenever those
# arrays change; factorize drops them too.
CG_BLOCK_GRAPHS = True       # False: launch every block eagerly
MAX_BLOCK_GRAPHS = 4         # kept graphs; a further key clears them

_KERNELS = {}
_HOST_AXPY = {}   # dtype -> scipy BLAS axpy of the cpu backend (fetched on first use)
# cpu: below this length dots and axpys run as serial numba loops (when numba imports): threaded BLAS spent ~20 us per
# call on 1.7e4-long vectors (solid_mechanics_3d profile, 1.35 s of 6.35 s), its thread start-up, not the arithmetic
HOST_SMALL_VECTOR_LIMIT = 100_000

# gpu reductions to one scalar: a cp.ReductionKernel runs those on ONE thread block (RTX 3050: 25 GB/s float64 against
# ~200 GB/s, 636 vs 84 us at 1e6 values), so the CG dots are two launches instead: a grid-stride kernel writing one
# partial sum per block (at most REDUCTION_BLOCKS_PER_SM x the SM count), then one block summing the partials in index
# order. Deterministic for a given size and device (no atomics). The Jacobi dot r . D^-1 r is fused into the solution
# update (x += alpha p, r -= alpha q in the same pass). Inputs must be C-contiguous arrays of the reduction dtype.
REDUCTION_THREADS = 256          # threads per block; the shared array of block_sum below has this size
REDUCTION_BLOCKS_PER_SM = 8
_REDUCTION_SOURCE = r'''
__device__ real_t block_sum(real_t sum) {
    __shared__ real_t partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    return partial[0];
}

extern "C" __global__ void pcg_dot_partials(const real_t* a, const real_t* b, real_t* partials, long long n) {
    real_t sum = 0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)blockDim.x * gridDim.x)
        sum += a[i] * b[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

// rho = r . z with z = r / diag(A) (Jacobi), never stored
extern "C" __global__ void pcg_jacobi_dot_partials(const real_t* residual, const real_t* inverse_diagonal,
                                                   real_t* partials, long long n) {
    real_t sum = 0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)blockDim.x * gridDim.x)
        sum += residual[i] * residual[i] * inverse_diagonal[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

// x += alpha p, r -= alpha q with alpha = rho / (p . q) (0 once p . q vanishes), then the Jacobi dot of the new r
extern "C" __global__ void pcg_update_jacobi_partials(const real_t* direction, const real_t* product, const real_t* rho,
        const real_t* direction_dot_product, const real_t* inverse_diagonal, real_t* solution, real_t* residual,
        real_t* partials, long long n) {
    const real_t alpha = *direction_dot_product != (real_t)0 ? *rho / *direction_dot_product : (real_t)0;
    real_t sum = 0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)blockDim.x * gridDim.x) {
        solution[i] += alpha * direction[i];
        const real_t updated = residual[i] - alpha * product[i];
        residual[i] = updated;
        sum += updated * updated * inverse_diagonal[i];
    }
    sum = block_sum(sum);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

extern "C" __global__ void pcg_sum_partials(const real_t* partials, real_t* total, int n_partials) {
    real_t sum = 0;
    for (int i = threadIdx.x; i < n_partials; i += blockDim.x)
        sum += partials[i];
    sum = block_sum(sum);
    if (threadIdx.x == 0) *total = sum;
}
'''
_REDUCTIONS = {}   # (device id, dtype) -> _DeviceReductions
_REDUCTION_TYPES = {np.dtype(np.float32): 'float', np.dtype(np.float64): 'double'}

# gpu CSR products by RawKernels (the CG product of _MatrixOperator, the multigrid cycle): LANES threads per row (the
# smallest power of two in CSR_LANE_CHOICES >= the mean row length) sum the row's products in a fixed order and reduce
# them by warp shuffles (deterministic). No cuSPARSE descriptors, workspace or output allocation per call (cupyx spmv
# spent ~60 us of host work per product, at 4e4 unknowns far more than the product), and capturable into a CUDA graph
# (cupy refuses cuSPARSE calls during stream capture). int32 indptr / indices (cupyx CSR); sums in the output type.
CSR_THREADS = 256
CSR_LANE_CHOICES = (2, 4, 8, 16, 32)
CSR_ROW_SOURCE = r'''
template<int LANES, typename Acc, typename M, typename X>
__device__ __forceinline__ Acc csr_row_sum(const int* __restrict__ indptr, const int* __restrict__ indices,
                                           const M* __restrict__ data, const X* __restrict__ x,
                                           const X* __restrict__ x_scale, int row, bool active) {
    Acc sum = (Acc)0;
    if (active) {
        const int stop = indptr[row + 1];
        for (int k = indptr[row] + (int)(threadIdx.x & (LANES - 1)); k < stop; k += LANES) {
            const int column = indices[k];
            const X value = x_scale ? (X)(x_scale[column] * x[column]) : x[column];
            sum += (Acc)data[k] * (Acc)value;
        }
    }
    for (int offset = LANES / 2; offset > 0; offset >>= 1)   // every lane of the warp takes part
        sum += __shfl_down_sync(0xffffffffu, sum, offset, LANES);
    return sum;
}

#define CSR_ROW \
    const long long thread = blockIdx.x * (long long)blockDim.x + threadIdx.x; \
    const int row = (int)(thread / LANES); \
    const bool active = row < n_rows; \
    const bool first_lane = (threadIdx.x & (LANES - 1)) == 0;

// y = A x, summed in Y
template<int LANES, typename M, typename X, typename Y>
__global__ void csr_product(int n_rows, const int* indptr, const int* indices, const M* data, const X* x, Y* y) {
    CSR_ROW
    const Y sum = csr_row_sum<LANES, Y, M, X>(indptr, indices, data, x, (const X*)0, row, active);
    if (active && first_lane) y[row] = sum;
}
'''
_TEMPLATE_MODULES = {}   # (device id, module key) -> {name expression: cupy RawFunction}
# cupy's disk cache skips modules with name_expressions (cupy/cuda/compiler.py: it recompiles to get the lowered C++
# names), so every process paid ~1-1.7 s of NVRTC for the CSR / multigrid templates. They are kept on disk here instead:
# the binary plus its {name expression: lowered name} map in cupy's cache directory, keyed on the source, the names, the
# options, cupy / NVRTC versions and the device architecture.
TEMPLATE_CACHE_OPTIONS = ('-ftz=true',)   # what cupy's RawModule adds, so the kernels compile as before
TEMPLATE_CACHE_PREFIX = 'open_sol_templates_'


def device_template_functions(key, source, name_expressions) -> dict:
    """{name expression: cupy function} of one module of templated kernels, compiled once per device and key and kept
    on disk across processes (_cached_template_binary)."""
    import cupy as cp
    cache_key = (cp.cuda.Device().id, key)
    if cache_key not in _TEMPLATE_MODULES:
        names = tuple(name_expressions)
        binary_path, lowered_names = _cached_template_binary(source, names)
        module = cp.RawModule(path=binary_path)
        _TEMPLATE_MODULES[cache_key] = {name: module.get_function(lowered_names[name]) for name in names}
    return _TEMPLATE_MODULES[cache_key]


def _cached_template_binary(source, names) -> tuple:
    """(binary path, {name expression: lowered name}) of source compiled by NVRTC for the current device, read from the
    cupy cache directory when an earlier process compiled the same thing, else compiled and written there."""
    import hashlib
    import json
    import os
    import cupy as cp
    from cupy.cuda import compiler
    from cupy_backends.cuda.libs import nvrtc
    signature = json.dumps([source, list(names), list(TEMPLATE_CACHE_OPTIONS), cp.__version__,
                            list(nvrtc.getVersion()), str(cp.cuda.Device().compute_capability)])
    cache_dir = compiler.get_cache_dir()
    stem = os.path.join(cache_dir, TEMPLATE_CACHE_PREFIX + hashlib.sha1(signature.encode('utf-8')).hexdigest())
    binary_path, names_path = stem + '.bin', stem + '.json'
    if os.path.exists(binary_path) and os.path.exists(names_path):   # the names file is written last
        with open(names_path, encoding='utf-8') as file:
            lowered_names = json.load(file)
        if all(name in lowered_names for name in names):
            return binary_path, lowered_names
    binary, lowered_names = compiler.compile_using_nvrtc(source, TEMPLATE_CACHE_OPTIONS, name_expressions=names)
    os.makedirs(cache_dir, exist_ok=True)
    _write_replacing(binary_path, binary.encode('utf-8') if isinstance(binary, str) else bytes(binary))
    _write_replacing(names_path, json.dumps(dict(lowered_names)).encode('utf-8'))
    return binary_path, dict(lowered_names)


def _write_replacing(path, data):
    """Write bytes to path through a temporary file in the same folder and os.replace (no half-written cache file)."""
    import os
    import tempfile
    handle, temporary_path = tempfile.mkstemp(dir=os.path.dirname(path), suffix='.tmp')
    with os.fdopen(handle, 'wb') as file:
        file.write(data)
    os.replace(temporary_path, path)


def csr_lanes(n_rows, nnz) -> int:
    """Threads per row of the gpu CSR kernels: the smallest CSR_LANE_CHOICES entry >= the mean row length."""
    mean_row = nnz / max(int(n_rows), 1)
    return next((lanes for lanes in CSR_LANE_CHOICES if lanes >= mean_row), CSR_LANE_CHOICES[-1])


def csr_grid(n_rows, lanes) -> tuple:
    """Grid of a CSR kernel with `lanes` threads per row in CSR_THREADS blocks."""
    return (max(1, -(-int(n_rows) * int(lanes) // CSR_THREADS)),)


def type_name(dtype) -> str:
    """C type of a float32 / float64 gpu kernel argument (ValueError otherwise)."""
    name = _REDUCTION_TYPES.get(np.dtype(dtype))
    if name is None:
        raise ValueError(f'gpu kernels take float32 / float64 arrays, got {np.dtype(dtype).name}')
    return name


def _csr_product_functions() -> dict:
    names = [f'csr_product<{lanes}, {t}, {t}, {t}>' for lanes in CSR_LANE_CHOICES for t in ('float', 'double')]
    return device_template_functions('pcg_csr_product', CSR_ROW_SOURCE, names)


class DeviceCsrProduct:
    """gpu y = A x of one cupyx CSR (float32 / float64, int32 indices) by csr_product: call -> one owned vector,
    apply(vector, out) -> the caller's (both capturable in a CUDA graph, unlike a cupyx SpMV)."""

    def __init__(self, matrix):
        import cupy as cp
        if matrix.indptr.dtype != np.int32 or matrix.indices.dtype != np.int32:
            raise TypeError(f'gpu CSR products need int32 indices, got {matrix.indptr.dtype} / {matrix.indices.dtype}')
        self.matrix = matrix
        self.dtype = np.dtype(matrix.dtype)
        n_rows = int(matrix.shape[0])
        lanes = csr_lanes(n_rows, int(matrix.nnz))
        t = type_name(self.dtype)
        self._function = _csr_product_functions()[f'csr_product<{lanes}, {t}, {t}, {t}>']
        self._grid = csr_grid(n_rows, lanes)
        self._n_rows = np.int32(n_rows)
        self.out = cp.empty(n_rows, dtype=self.dtype)

    def __call__(self, vector):
        return self.apply(vector, self.out)

    def apply(self, vector, out):
        """out = A vector (C-contiguous of the matrix dtype, not overlapping vector); returns out."""
        for name, array in (('vector', vector), ('out', out)):
            if array.dtype != self.dtype or not array.flags.c_contiguous:
                raise ValueError(f'gpu CSR product: {name} must be C-contiguous {self.dtype.name}, got {array.dtype}')
        matrix = self.matrix   # read per call: callers write new values into matrix.data in place
        self._function(self._grid, (CSR_THREADS,), (self._n_rows, matrix.indptr, matrix.indices, matrix.data, vector,
                                                     out))
        return out


class _DeviceReductions:
    """The gpu reductions of one dtype (module header) and their partial-sum buffer, reused by every call (calls are
    stream-ordered on the GPU-owning thread); each writes a 0-d device scalar, no host sync."""

    def __init__(self, dtype):
        import cupy as cp
        self._cp = cp
        self.dtype = np.dtype(dtype)
        module = cp.RawModule(code=_REDUCTION_SOURCE.replace('real_t', _REDUCTION_TYPES[self.dtype]))
        self._functions = {name: module.get_function(f'pcg_{name}')
                           for name in ('dot_partials', 'jacobi_dot_partials', 'update_jacobi_partials',
                                        'sum_partials')}
        self.max_blocks = REDUCTION_BLOCKS_PER_SM * int(cp.cuda.Device().attributes['MultiProcessorCount'])
        self._partials = cp.empty(self.max_blocks, dtype=self.dtype)

    def _reduce(self, name, arrays, out):
        """Stage 1 (`name` over the arrays, the first one's length) and stage 2 into out (a new 0-d array if None)."""
        n_values = int(arrays[0].size)
        for array in arrays:
            if array.dtype != self.dtype or not array.flags.c_contiguous:
                raise ValueError(f'pcg reduction: arrays must be C-contiguous {self.dtype.name}, got {array.dtype} '
                                 f'(contiguous {array.flags.c_contiguous})')
        if out is None:
            out = self._cp.empty((), dtype=self.dtype)
        elif out.dtype != self.dtype:
            raise ValueError(f'pcg reduction: out must be {self.dtype.name}, got {out.dtype}')
        blocks = max(1, min(self.max_blocks, -(-n_values // REDUCTION_THREADS)))
        self._functions[name]((blocks,), (REDUCTION_THREADS,), (*arrays, self._partials, np.int64(n_values)))
        self._functions['sum_partials']((1,), (REDUCTION_THREADS,), (self._partials, out, np.int32(blocks)))
        return out

    def dot(self, a, b, out=None):
        return self._reduce('dot_partials', (a, b), out)

    def jacobi_dot(self, residual, inverse_diagonal, out=None):
        return self._reduce('jacobi_dot_partials', (residual, inverse_diagonal), out)

    def update_jacobi_dot(self, direction, product, rho, direction_dot_product, inverse_diagonal, solution, residual,
                          out=None):
        return self._reduce('update_jacobi_partials', (direction, product, rho, direction_dot_product,
                                                       inverse_diagonal, solution, residual), out)


def _device_reductions(dtype) -> _DeviceReductions:
    import cupy as cp
    key = (cp.cuda.Device().id, np.dtype(dtype))
    if key not in _REDUCTIONS:
        _REDUCTIONS[key] = _DeviceReductions(dtype)
    return _REDUCTIONS[key]


def device_vector_dot(a, b, out=None):
    """a . b of two C-contiguous cupy vectors of one dtype as a 0-d device scalar (multi-block; out reused if given)."""
    return _device_reductions(a.dtype).dot(a, b, out)


def _device_jacobi_residual_dot(residual, inverse_diagonal, out=None):
    return _device_reductions(residual.dtype).jacobi_dot(residual, inverse_diagonal, out)


def _device_update_solution_jacobi_dot(direction, product, rho, direction_dot_product, inverse_diagonal, solution,
                                       residual, out=None):
    return _device_reductions(residual.dtype).update_jacobi_dot(direction, product, rho, direction_dot_product,
                                                                inverse_diagonal, solution, residual, out)


def _small_dot_numba(numba):
    @numba.njit(cache=True)
    def dot(a, b):
        total = 0.0
        for index in range(a.size):
            total += a[index] * b[index]
        return total
    return dot


def _small_axpy_numba(numba):
    @numba.njit(cache=True)
    def axpy(alpha, x, y):
        for index in range(y.size):
            y[index] += alpha * x[index]
    return axpy


def _host_small(name, a, b):
    """The short-vector twin 'dot' / 'axpy' (core.kernels.host_twin: a serial numba loop, the BLAS form without numba)
    for C-contiguous host vectors a, b shorter than HOST_SMALL_VECTOR_LIMIT, else its BLAS form."""
    blas, factory = _HOST_SMALL[name]
    if a.size >= HOST_SMALL_VECTOR_LIMIT or not (a.flags.c_contiguous and b.flags.c_contiguous):
        return blas
    from ..core.kernels import host_twin
    return host_twin(f'pcg.small_{name}', blas, factory)


def _host_axpy(alpha, x, y):
    """y += alpha x in place: a serial numba loop on short vectors, else BLAS axpy."""
    _host_small('axpy', x, y)(float(alpha), x, y)


def _blas_axpy(alpha, x, y):
    """y += alpha x in place by BLAS axpy (one pass, no temporary; scipy imported on first use)."""
    axpy = _HOST_AXPY.get(y.dtype)
    if axpy is None:
        from scipy.linalg.blas import get_blas_funcs
        axpy = _HOST_AXPY[y.dtype] = get_blas_funcs('axpy', dtype=y.dtype)
    updated = axpy(x, y, a=alpha)
    if updated is not y:   # f2py copied y (not contiguous): write the result back
        y[...] = updated


def host_csr_matvec():
    """scipy's in-place CSR product kernel y += A x (private sparsetools; also the multigrid cycle's), None when this
    scipy has none."""
    try:
        from scipy.sparse import _sparsetools
    except ImportError:
        return None
    return getattr(_sparsetools, 'csr_matvec', None)


# name -> (BLAS form, numba factory) of the short-vector twins (_host_small)
_HOST_SMALL = {'dot': (np.dot, _small_dot_numba), 'axpy': (_blas_axpy, _small_axpy_numba)}


def _host_vector_dot(a, b, out=None):
    total = _host_small('dot', a, b)(a, b)
    if out is None:
        return total
    out[...] = total
    return out


def _host_update_solution(direction, product, rho, direction_dot_product, solution, residual):
    if direction_dot_product == 0:   # exact convergence: alpha = 0
        return
    alpha = float(rho / direction_dot_product)
    _host_axpy(alpha, direction, solution)
    _host_axpy(-alpha, product, residual)


def _host_update_direction(preconditioned, rho_new, rho_old, direction):
    beta = rho_new / rho_old if rho_old != 0 else np.zeros((), direction.dtype)
    direction *= beta
    direction += preconditioned


# numpy twins of the fused kernels (same call signatures, outputs written in place); the cpu Jacobi preconditioner
# forms z = r / diag(A) once per iteration in the dead product vector instead of the two Jacobi kernels
_HOST_KERNELS = {'vector_dot': _host_vector_dot, 'update_solution': _host_update_solution,
                 'update_direction': _host_update_direction}


def _kernels(backend='gpu'):
    """CG kernels of the backend: the multi-block reductions (float32 / float64) and fused cupy elementwise kernels
    (dtype-generic T, compiled on first use), or their numpy twins."""
    if backend == 'cpu':
        return _HOST_KERNELS
    if not _KERNELS:
        import cupy as cp
        _KERNELS['vector_dot'] = device_vector_dot
        _KERNELS['jacobi_residual_dot'] = _device_jacobi_residual_dot
        _KERNELS['update_solution_jacobi_dot'] = _device_update_solution_jacobi_dot
        # x += alpha p, r -= alpha q with alpha = rho / (p . q) (0 once p . q vanishes: exact convergence)
        _KERNELS['update_solution'] = cp.ElementwiseKernel(
            'T direction, T product, T rho, T direction_dot_product', 'T solution, T residual',
            'T alpha = direction_dot_product != (T)0 ? rho / direction_dot_product : (T)0;'
            'solution += alpha * direction; residual -= alpha * product;',
            'pcg_update_solution')
        # p = z + beta p with beta = rho_new / rho_old
        _KERNELS['update_direction'] = cp.ElementwiseKernel(
            'T preconditioned, T rho_new, T rho_old', 'T direction',
            'T beta = rho_old != (T)0 ? rho_new / rho_old : (T)0;'
            'direction = preconditioned + beta * direction;',
            'pcg_update_direction')
        _KERNELS['update_direction_jacobi'] = cp.ElementwiseKernel(
            'T residual, T inverse_diagonal, T rho_new, T rho_old', 'T direction',
            'T beta = rho_old != (T)0 ? rho_new / rho_old : (T)0;'
            'direction = inverse_diagonal * residual + beta * direction;',
            'pcg_update_direction_jacobi')
    return _KERNELS


# --- CPU helpers (no cupy: estimators, validation) ---------------------------------------------------------------

def auto_max_iterations(longest_window_axis) -> int:
    """Iteration limit of max_iterations = 0: twice the Jacobi-PCG cold-start estimate, at least the floor."""
    return max(AUTO_CG_MAX_ITERATIONS_FLOOR, 2 * CG_ITERATIONS_PER_AXIS_CELL * int(longest_window_axis))


def resolve_max_iterations(max_iterations, domain) -> int:
    """max_iterations setting -> iteration limit on the domain's physical window (0 = auto; a triangle / tetrahedron
    mesh counts 2 nodes (P2) per mean window element size along its longest window side)."""
    if max_iterations:
        return int(max_iterations)
    grid_kind = getattr(domain, 'grid_kind', 'tensor')
    if grid_kind in ('triangles', 'tetrahedra'):
        n_window_elements = max(int(domain.window_elements().sum()), 1)
        sides = ((domain.size_x, domain.size_y) if grid_kind == 'triangles'
                 else (domain.size_x, domain.size_y, domain.size_z))
        mean_element_size = (math.prod(sides) / n_window_elements) ** (1.0 / len(sides))
        return auto_max_iterations(2 * math.ceil(max(sides) / mean_element_size))
    return auto_max_iterations(max(len(range(*window.indices(count)))   # 3D: z included
                                   for window, count in zip(domain.physical_window(), domain.materials.shape)))


def workspace_bytes(n_unknowns, precision, jacobi=True, csr_entries=0, fused=False) -> int:
    """Device bytes of the PCGSolver workspace (residual, direction, product (matrix-free workspace, the gpu CSR
    operator's or the cpu workspace's), + preconditioned unless Jacobi, + the second direction of a fused iteration);
    mixed: those in float32 plus the float64 residual and product, float32 correction, (Jacobi) inverse diagonal and the
    float32 copy of the csr_entries operator values. No solution vector; a matrix-free mixed caller counts its own
    float32 operator copy (and a fused caller its partial sums)."""
    n_vectors = 3 + int(not jacobi) + int(fused)
    if precision == 'mixed':   # + the two float32 / float64 start-norm slots
        return (n_vectors * int(n_unknowns) * 4 + 5 * 4 + 2 * 8
                + (8 + 8 + 4 + 4 * int(jacobi)) * int(n_unknowns) + 4 * int(csr_entries))
    item_bytes = np.dtype(PRECISION_DTYPES[precision]).itemsize
    return n_vectors * int(n_unknowns) * item_bytes + 5 * item_bytes   # 3 CG scalars + 2 start norms


def mixed_extra_bytes(n_unknowns, csr_entries) -> int:
    """Device bytes a 'mixed' Jacobi CSR solve adds over 'double' (estimators that count a double workspace)."""
    return workspace_bytes(n_unknowns, 'mixed', csr_entries=csr_entries) - workspace_bytes(n_unknowns, 'double')


def stop_threshold_squared(tolerance, rhs_squared, initial_residual_squared, reference, dtype):
    """(threshold ||r||^2 at which CG stops, reference norm ||b|| or ||r0||) for the stop reference."""
    if reference == 'rhs':
        return tolerance ** 2 * rhs_squared, math.sqrt(rhs_squared)
    if reference != 'initial_residual':
        raise ValueError(f"unknown stop reference {reference!r}; valid: {', '.join(STOP_REFERENCES)}")
    floor = ROUNDOFF_FLOOR_EPSILONS * float(np.finfo(dtype).eps)
    return (max(tolerance ** 2 * initial_residual_squared, floor ** 2 * rhs_squared),
            math.sqrt(initial_residual_squared))


# --- operators and preconditioners --------------------------------------------------------------------------------

def _array_module(backend):
    from ..core.backend import array_module
    return array_module(backend)


class _MatrixOperator:
    """Assembled CSR operator: gpu = the csr_product RawKernel into one owned product vector (overwritten by the next
    product), out unused; cpu = scipy's CSR kernel into out (the workspace product vector), scipy's allocating product
    when out is None or the dtypes differ."""

    matrix_free = False

    def __init__(self, matrix, backend='gpu'):
        self.matrix = matrix
        self.n_unknowns = int(matrix.shape[0])
        self._device_product = None
        self._host_matvec = None
        if backend == 'gpu':
            matrix.sum_duplicates()   # canonical rows (a no-op when already canonical)
            self._device_product = DeviceCsrProduct(matrix)
        elif matrix.format == 'csr':
            self._host_matvec = host_csr_matvec()

    def product(self, vector, out):
        if self._device_product is not None:
            return self._device_product(vector)
        matrix = self.matrix
        if (self._host_matvec is not None and out is not None and matrix.dtype == vector.dtype == out.dtype
                and vector.flags.c_contiguous and out.flags.c_contiguous):
            out.fill(0)   # the kernel accumulates y += A x
            self._host_matvec(matrix.shape[0], matrix.shape[1], matrix.indptr, matrix.indices, matrix.data, vector,
                              out)
            return out
        return matrix.dot(vector)


class _FunctionOperator:
    """Matrix-free operator: apply(vector, out) writes A vector into the preallocated out."""

    matrix_free = True

    def __init__(self, apply, n_unknowns):
        self.apply = apply
        self.n_unknowns = int(n_unknowns)

    def product(self, vector, out):
        self.apply(vector, out)
        return out


class _JacobiPreconditioner:
    """z = r / diag(A): gpu fused into the solution update + dot and the direction kernel (z never stored); cpu formed
    once per iteration in the workspace product vector, dead after the solution update. update() does the whole
    iteration after p . q: solution / residual update, rho_new, direction."""

    jacobi = True

    def __init__(self, inverse_diagonal, backend='gpu'):
        self.inverse_diagonal = (inverse_diagonal if backend == 'cpu'
                                 else _array_module(backend).ascontiguousarray(inverse_diagonal))
        self.backend = backend

    def start(self, residual, rho, direction, workspace):
        if self.backend == 'cpu':
            np.multiply(self.inverse_diagonal, residual, out=direction)
            _host_vector_dot(residual, direction, out=rho)
            return
        _kernels(self.backend)['jacobi_residual_dot'](residual, self.inverse_diagonal, out=rho)
        _array_module(self.backend).multiply(self.inverse_diagonal, residual, out=direction)

    def update(self, product, direction_dot_product, rho_old, rho_new, solution, workspace):
        residual, direction = workspace.residual, workspace.direction
        if self.backend == 'cpu':
            _host_update_solution(direction, product, rho_old, direction_dot_product, solution, residual)
            preconditioned = np.multiply(self.inverse_diagonal, residual, out=workspace.product)
            _host_vector_dot(residual, preconditioned, out=rho_new)
            _host_update_direction(preconditioned, rho_new, rho_old, direction)
            return
        kernels = _kernels(self.backend)
        kernels['update_solution_jacobi_dot'](direction, product, rho_old, direction_dot_product,
                                              self.inverse_diagonal, solution, residual, out=rho_new)
        kernels['update_direction_jacobi'](residual, self.inverse_diagonal, rho_new, rho_old, direction)


class _CallbackPreconditioner:
    """precondition(residual, out) writes z = M^-1 r into the workspace's preconditioned vector (block Jacobi, ...)."""

    jacobi = False

    def __init__(self, precondition, backend='gpu'):
        self.precondition = precondition
        self.backend = backend

    def start(self, residual, rho, direction, workspace):
        self.precondition(residual, workspace.preconditioned)
        _kernels(self.backend)['vector_dot'](residual, workspace.preconditioned, out=rho)
        _array_module(self.backend).copyto(direction, workspace.preconditioned)

    def update(self, product, direction_dot_product, rho_old, rho_new, solution, workspace):
        residual, direction = workspace.residual, workspace.direction
        kernels = _kernels(self.backend)
        kernels['update_solution'](direction, product, rho_old, direction_dot_product, solution, residual)
        self.precondition(residual, workspace.preconditioned)
        kernels['vector_dot'](residual, workspace.preconditioned, out=rho_new)
        kernels['update_direction'](workspace.preconditioned, rho_new, rho_old, direction)


class PCGWorkspace:
    """Preallocated CG vectors of one system size and dtype (product: matrix-free or cpu (CSR product, Jacobi z);
    preconditioned unless Jacobi; next_direction: a fused iteration's second direction) and the three device
    scalars."""

    __slots__ = ('n_unknowns', 'dtype', 'residual', 'direction', 'next_direction', 'product', 'preconditioned',
                 'scalars', 'start_norms', 'start_slots')

    def __init__(self, n_unknowns, dtype, matrix_free, jacobi, backend='gpu', fused=False):
        cp = _array_module(backend)
        self.n_unknowns = int(n_unknowns)
        self.dtype = np.dtype(dtype)
        self.residual = cp.zeros(self.n_unknowns, dtype=self.dtype)
        self.direction = cp.zeros(self.n_unknowns, dtype=self.dtype)
        self.next_direction = cp.zeros(self.n_unknowns, dtype=self.dtype) if fused else None
        self.product = cp.zeros(self.n_unknowns, dtype=self.dtype) if matrix_free or backend == 'cpu' else None
        self.preconditioned = None if jacobi else cp.zeros(self.n_unknowns, dtype=self.dtype)
        self.scalars = tuple(cp.zeros((), dtype=self.dtype) for _ in range(3))
        self.start_norms = cp.zeros(2, dtype=self.dtype)   # ||b||^2, ||r0||^2 of a solve start: one read
        self.start_slots = (self.start_norms[0, ...], self.start_norms[1, ...])   # 0-d views (reduction outs)

    def fits(self, n_unknowns, dtype, matrix_free, jacobi, fused=False) -> bool:
        return (self.n_unknowns == n_unknowns and self.dtype == np.dtype(dtype)
                and (self.product is not None or not matrix_free) and (self.preconditioned is not None or jacobi)
                and (self.next_direction is not None or not fused))


class _BlockGraphs:
    """gpu CUDA graphs of whole CG check blocks (CG_BLOCK_GRAPHS): per key one eager run (kernels loaded), then one
    capture, replayed as a single launch; a failed capture warns and keeps every block eager. participants: objects
    whose `inside_capture` flag is set during a capture (a multigrid V-cycle then launches into the block's graph)."""

    def __init__(self, participants=()):
        import cupy as cp
        self._cp = cp
        self.enabled = CG_BLOCK_GRAPHS
        self._graphs = {}   # key -> captured graph, or None after the eager run
        self.n_graphs = 0   # captures made
        self._participants = tuple(participants)

    def clear(self):
        self._graphs.clear()

    def run(self, key, block):
        """block(): the block's launches (no host sync, no allocation); eagerly, or replayed from its graph."""
        if not self.enabled:
            block()
            return
        if key not in self._graphs:
            if len(self._graphs) >= MAX_BLOCK_GRAPHS:
                self._graphs.clear()
            self._graphs[key] = None
            block()
            return
        graph = self._graphs[key]
        if graph is None:
            graph = self._capture(block)
            if graph is None:
                block()
                return
            self._graphs[key] = graph
        graph.launch(self._cp.cuda.get_current_stream())

    def _capture(self, block):
        """CUDA graph of block() (None, with a warning and eager blocks from now on, when capture fails)."""
        stream = self._cp.cuda.Stream(non_blocking=True)
        try:
            with stream:
                stream.begin_capture()
                for participant in self._participants:
                    participant.inside_capture = True
                try:
                    block()
                finally:
                    for participant in self._participants:
                        participant.inside_capture = False
                    graph = stream.end_capture()
        except Exception as error:   # capture unsupported here: the same kernels launched one by one
            self.enabled = False
            self._graphs.clear()
            warnings.warn(f'pcg: CUDA graph capture of a CG block failed ({type(error).__name__}: {error}); '
                          'launching its kernels one by one from now on', RuntimeWarning, stacklevel=4)
            return None
        self.n_graphs += 1
        return graph


# --- runtime object -----------------------------------------------------------------------------------------------

class _MixedWorkspace:
    """Float64 side of a 'mixed' solve: the replacing residual, the float32 correction (the CG's own solution, folded
    into x at each replacement) and, matrix-free, the float64 product."""

    __slots__ = ('n_unknowns', 'residual', 'correction', 'product', 'start_norms', 'start_slots')

    def __init__(self, n_unknowns, matrix_free, backend='gpu'):
        cp = _array_module(backend)
        self.n_unknowns = int(n_unknowns)
        self.residual = cp.zeros(self.n_unknowns, dtype=np.float64)
        self.correction = cp.zeros(self.n_unknowns, dtype=np.float32)
        self.product = cp.zeros(self.n_unknowns, dtype=np.float64) if matrix_free else None
        self.start_norms = cp.zeros(2, dtype=np.float64)   # ||b||^2, ||r0||^2 of a solve start: one read
        self.start_slots = (self.start_norms[0, ...], self.start_norms[1, ...])   # 0-d views (reduction outs)

    def fits(self, n_unknowns, matrix_free) -> bool:
        return self.n_unknowns == n_unknowns and (self.product is not None or not matrix_free)


# 'jacobi': the physics' diagonal (block) preconditioner; 'multigrid': solvers.multigrid V-cycles built from an
# assembled scalar CSR operator (DiffusionSystem; not mixed, not a block preconditioner), or a matrix-free caller's own
# V-cycle callback (ElasticSystem3D: physics.mechanics.multigrid3d; mixed too)
PRECONDITIONERS = ('jacobi', 'multigrid')
# fused_iteration protocol (factorize; ElasticSystem3D: kernels3d.FusedBrickIteration), device scalars as 0-d arrays,
# no host sync and no allocation (CUDA graph capture): start(residual, preconditioned, direction, product, scratch,
# rho): z = M^-1 r, rho = r . z, direction = z, product = A direction; step(solution, residual, preconditioned,
# direction, next_direction, product, rho, rho_next): alpha = rho / (p . q of the last product), x += alpha p, r -=
# alpha q, z = M^-1 r, rho_next = r . z, next_direction = z + (rho_next / rho) direction, product = A next_direction;
# rescale(direction, product, factor) (mixed residual replacement): direction, product and its p . q partials x factor
FUSED_ITERATION_PROTOCOL = ('start', 'step', 'rescale')


class PCGSolver:
    """LinearSolver of the 'pcg' type for SPD systems: factorize(operator, preconditioner) binds the operator and
    builds / takes the preconditioner, solve(rhs, solution) warm-starts from solution; keeps the convergence record."""

    host_bytes = 0

    def __init__(self, precision='double', tolerance=1e-8, max_iterations=AUTO_CG_MAX_ITERATIONS_FLOOR,
                 compute_residual=False, label=None, backend='gpu', preconditioner='jacobi'):
        if precision not in PRECISION_DTYPES:
            raise ValueError(f"precision must be one of {', '.join(PRECISION_DTYPES)}, got {precision!r}")
        if preconditioner not in PRECONDITIONERS:
            raise ValueError(f"preconditioner must be one of {', '.join(PRECONDITIONERS)}, got {preconditioner!r}")
        self.preconditioner = preconditioner   # mixed + multigrid: only a matrix-free caller's own V-cycle (_factorize)
        self.preconditioner_counters = None   # () -> dict: a caller-built preconditioner's work (multigrid3d levels)
        self._multigrid = None              # multigrid: solvers.multigrid.MultigridPreconditioner, built on factorize
        if not tolerance > 0.0 or int(max_iterations) < 1:
            raise ValueError(f'need tolerance > 0 and max_iterations >= 1, got tolerance={tolerance}, '
                             f'max_iterations={max_iterations}')
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, got {backend!r}")
        self.backend = backend   # 'gpu': cupy vectors, cupyx CSR; 'cpu': numpy vectors, scipy CSR
        self.precision = precision
        self.dtype = np.dtype(PRECISION_DTYPES[precision])   # rhs / solution / operator dtype
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.compute_residual = bool(compute_residual)
        self.label = label if label is not None else (lambda: 'pcg')   # () -> message prefix (physics names late)
        self._operator = None
        self._preconditioner = None
        self._workspace = None
        self._inner_operator = None         # mixed: float32 CSR sharing the operator's indices
        self._mixed = None                  # mixed: _MixedWorkspace
        self._fused = None                  # the caller's fused_iteration of the CG loop (mixed: the float32 one)
        self._block_graphs = None           # gpu, capture_blocks: _BlockGraphs
        self._capture_blocks = False        # the last factorize's capture_blocks (a hierarchy bound at solve)
        self._multigrid_bound = False       # the bound preconditioner is self._multigrid.precondition
        # 'multigrid' on a CSR (module header): how the binding may change at solve. 'jacobi_small' (no setup yet),
        # 'jacobi_no_links' (Jacobi on a smoother-only hierarchy), 'lagged' (a kept hierarchy), None
        self._binding = None
        self._binding_solves = 0            # solves on the current binding
        self._jacobi_binds_off = False      # a Jacobi bind missed its budget: always the hierarchy from now on
        self._fresh_iterations = None       # CG iterations of the first solve after the last fresh setup
        self._hierarchy_iterations = None   # CG iterations of the last solve on the own hierarchy
        self._solve_setup_seconds = 0.0     # setups run inside the current solve (factorize seconds, not solve)
        self._last_stop = None              # (threshold ||r||^2, reference norm) of the current solve's first attempt
        self.n_jacobi_binds = 0             # 'multigrid' factorizes that bound the Jacobi inverse diagonal
        self.n_jacobi_fallbacks = 0         # Jacobi solves that missed their budget and continued on the hierarchy
        self.n_lagged_fallbacks = 0         # kept-hierarchy solves that missed their cap and continued on a fresh one
        self.n_solves = 0
        self.last_iters = 0
        self.total_iters = 0
        self.total_refinements = 0          # mixed: outer float64 residual refreshes
        self.last_converged = False
        self.last_residual = float('nan')   # ||b - A x|| / reference norm of the last solve (see solve)
        self.last_warning = None            # message of the latest non-converged solve (never cleared)
        self.n_unconverged = 0
        self._non_finite = False            # the last solve stopped on a NaN / inf residual
        self.n_factorizations = 0
        self.total_factorize_seconds = 0.0  # host wall time (multigrid setups sync; a Jacobi bind is launch time)
        self.total_solve_seconds = 0.0      # host wall time: a solve ends in a residual read-back (synced)

    def factorize(self, operator, preconditioner=None, n_unknowns=None, single_operator=None,
                  single_preconditioner=None, fused_iteration=None, single_fused_iteration=None,
                  capture_blocks=False) -> None:
        """Bind a CSR operator of the backend (cupyx / scipy) or a matrix-free apply(vector, out) (then n_unknowns
        required); preconditioner None (Jacobi from the CSR diagonal), an inverse diagonal or precondition(residual, out).
        Mixed matrix-free: single_operator / single_preconditioner = their float32 twins for the inner CG.
        fused_iteration (single_: the mixed inner one): an object whose start / step run the whole CG iteration of the
        matrix-free operator and callback preconditioner in a few fused launches (FUSED_ITERATION_PROTOCOL).
        capture_blocks: gpu check blocks replayed as CUDA graphs (the caller calls release_block_graphs whenever the
        arrays its callbacks read are replaced)."""
        start = time.perf_counter()
        self._fused = None
        self._multigrid_bound = False   # _factorize: the bound preconditioner is self._multigrid's V-cycle
        self._binding, self._binding_solves = None, 0
        self._factorize(operator, preconditioner, n_unknowns, single_operator, single_preconditioner, fused_iteration,
                        single_fused_iteration)
        # an own multigrid hierarchy (rebuilt or rebound only here or by _bind_hierarchy, which make new _BlockGraphs,
        # so none replays its old vectors) joins
        self._capture_blocks = bool(capture_blocks)
        participants = (self._multigrid,) if self._multigrid_bound else ()
        self._block_graphs = (_BlockGraphs(participants) if capture_blocks and self.backend == 'gpu' else None)
        self.total_factorize_seconds += time.perf_counter() - start
        self.n_factorizations += 1

    def release_block_graphs(self) -> None:
        """Drop the captured CG blocks (they hold the pointers of the arrays the callbacks read at capture)."""
        if self._block_graphs is not None:
            self._block_graphs.clear()

    def _factorize(self, operator, preconditioner, n_unknowns, single_operator, single_preconditioner, fused_iteration,
                   single_fused_iteration):
        if not callable(operator):
            self._require_backend_array('operator', operator.data)
        if not (preconditioner is None or callable(preconditioner)):
            self._require_backend_array('inverse diagonal', preconditioner)
        fused = single_fused_iteration if self.precision == 'mixed' else fused_iteration
        if fused is not None and not (callable(operator) and callable(preconditioner)
                                      or self.precision == 'mixed' and callable(operator)):
            raise TypeError(f'{self.label()}: a fused iteration needs a matrix-free operator and a precondition '
                            'callback')
        self._fused = fused
        if self.precision == 'mixed':
            if callable(operator):
                self._factorize_mixed_matrix_free(operator, n_unknowns, single_operator, single_preconditioner)
            else:
                self._factorize_mixed(operator, preconditioner)
            return
        if callable(operator):
            if n_unknowns is None:
                raise ValueError(f'{self.label()}: a matrix-free operator needs n_unknowns')
            self._operator = _FunctionOperator(operator, n_unknowns)
        else:
            self._operator = _MatrixOperator(operator, self.backend)
        if self.preconditioner == 'multigrid':
            preconditioner = self._multigrid_preconditioner(operator, preconditioner)
            self._multigrid_bound = (callable(preconditioner) and self._multigrid is not None
                                     and preconditioner == self._multigrid.precondition)
        if preconditioner is None:
            if self._operator.matrix_free:
                raise TypeError(f'{self.label()}: a matrix-free operator needs a preconditioner (its inverse diagonal '
                                'or a precondition callback)')
            preconditioner = 1.0 / operator.diagonal()
        if callable(preconditioner):
            self._preconditioner = _CallbackPreconditioner(preconditioner, self.backend)
        else:
            if preconditioner.dtype != self.dtype or preconditioner.shape != (self._operator.n_unknowns,):
                raise ValueError(f'{self.label()}: inverse diagonal must be a flat {self.dtype.name} array of '
                                 f'{self._operator.n_unknowns} values, got {preconditioner.dtype} {preconditioner.shape}')
            self._preconditioner = _JacobiPreconditioner(preconditioner, self.backend)
        shape_key = (self._operator.n_unknowns, self.dtype, self._operator.matrix_free, self._preconditioner.jacobi)
        fused = self._fused is not None
        if self._workspace is None or not self._workspace.fits(*shape_key, fused=fused):
            self._workspace = PCGWorkspace(*shape_key, backend=self.backend, fused=fused)

    def _multigrid_preconditioner(self, operator, preconditioner):
        """precondition callback of the multigrid hierarchy of a CSR operator (set up, or kept: lagged setup), or the
        inverse diagonal (preconditioner; None: from the diagonal) where a hierarchy cannot pay (module header). A
        matrix-free operator brings its own multigrid callback (ElasticSystem3D: multigrid3d), used as given; a block
        preconditioner of a CSR raises TypeError."""
        if callable(operator) and callable(preconditioner):
            return preconditioner
        if callable(operator) or callable(preconditioner):
            raise TypeError(f'{self.label()}: the multigrid preconditioner needs an assembled scalar CSR operator '
                            '(heat, electric_currents, electrostatics, magnetostatics), not a matrix-free operator '
                            'or a block preconditioner')
        if operator.dtype != self.dtype:
            raise ValueError(f'{self.label()}: operator must be {self.dtype.name}, got {operator.dtype}')
        jacobi_binds = MULTIGRID_JACOBI_BINDS and not self._jacobi_binds_off
        if jacobi_binds and self.n_factorizations == 0 and operator.shape[0] < MULTIGRID_MIN_UNKNOWNS:
            return self._bind_jacobi('jacobi_small', preconditioner)   # a refactorize (sweep, nonlinear) sets up
        if self._multigrid is None:
            from .multigrid import MultigridPreconditioner
            self._multigrid = MultigridPreconditioner(self.backend)
        if self._lag_keeps(operator):
            self._multigrid.rebind(operator)
            self._binding = 'lagged'
            return self._multigrid.precondition
        self._multigrid.setup(operator)
        self._fresh_iterations = self._hierarchy_iterations = None
        if jacobi_binds and self._multigrid.smoother_only:
            return self._bind_jacobi('jacobi_no_links', preconditioner)
        return self._multigrid.precondition

    def _bind_jacobi(self, binding, inverse_diagonal):
        """The Jacobi bind of 'multigrid' (binding 'jacobi_small' / 'jacobi_no_links'); the kept hierarchy's lag
        record is dropped."""
        self._binding = binding
        self.n_jacobi_binds += 1
        self._fresh_iterations = self._hierarchy_iterations = None
        return inverse_diagonal

    def _lag_keeps(self, operator) -> bool:
        """Lagged setup: the hierarchy bound last stays for this same-pattern operator while the last solve on it took
        at most LAGGED_SETUP_ITERATION_RATIO x the first solve after its setup."""
        return (LAGGED_MULTIGRID_SETUP and self._fresh_iterations is not None
                and self._hierarchy_iterations is not None
                and self._hierarchy_iterations <= LAGGED_SETUP_ITERATION_RATIO * self._fresh_iterations
                and self._multigrid.can_rebind(operator))

    def _require_backend_array(self, name, array):
        """TypeError unless `array` lives on this solver's backend (numpy on the cpu, cupy on the gpu)."""
        on_host = isinstance(array, np.ndarray)
        if on_host != (self.backend == 'cpu'):
            expected = 'a numpy / scipy' if self.backend == 'cpu' else 'a cupy / cupyx'
            raise TypeError(f'{self.label()}: the {self.backend} backend needs {expected} {name}, got '
                            f'{type(array).__module__}.{type(array).__name__}')

    def _factorize_mixed(self, operator, inverse_diagonal):
        """Mixed CSR: a float64 CSR operator and Jacobi only (inverse diagonal or None); keeps a float32 copy of the
        values (shared indices) and of the inverse diagonal for the inner CG."""
        from ..core.backend import sparse_module
        cp = _array_module(self.backend)
        sparse = sparse_module(self.backend)
        if self.preconditioner == 'multigrid':
            raise ValueError(f"{self.label()}: the multigrid preconditioner of a CSR operator needs precision 'single' "
                             "or 'double' (mixed iterates on a Jacobi float32 operator copy)")
        if callable(inverse_diagonal):
            raise TypeError(f'{self.label()}: mixed precision on a CSR operator needs a Jacobi preconditioner (a block '
                            'preconditioner: single or double, or a matrix-free operator with float32 callbacks)')
        if operator.dtype != np.float64:
            raise ValueError(f'{self.label()}: mixed precision needs a float64 operator, got {operator.dtype}')
        self._operator = _MatrixOperator(operator, self.backend)
        n_unknowns = self._operator.n_unknowns
        if inverse_diagonal is None:
            inverse_diagonal = 1.0 / operator.diagonal()
        if inverse_diagonal.shape != (n_unknowns,):
            raise ValueError(f'{self.label()}: inverse diagonal must be a flat array of {n_unknowns} values, got '
                             f'{inverse_diagonal.shape}')
        self._inner_operator = _MatrixOperator(sparse.csr_matrix(
            (operator.data.astype(np.float32), operator.indices, operator.indptr), shape=operator.shape), self.backend)
        self._preconditioner = _JacobiPreconditioner(cp.asarray(inverse_diagonal, dtype=np.float32), self.backend)
        self._bind_mixed_workspaces(n_unknowns, matrix_free=False, jacobi=True)

    def _factorize_mixed_matrix_free(self, operator, n_unknowns, single_operator, single_preconditioner):
        """Mixed matrix-free: the float64 apply refreshes the outer residual, the caller's float32 apply / precondition
        callbacks (its own float32 operator copy) run the inner CG."""
        if n_unknowns is None:
            raise ValueError(f'{self.label()}: a matrix-free operator needs n_unknowns')
        if not (callable(single_operator) and callable(single_preconditioner)):
            raise TypeError(f'{self.label()}: mixed precision on a matrix-free operator needs its float32 apply and '
                            'precondition callbacks (single_operator, single_preconditioner)')
        self._operator = _FunctionOperator(operator, n_unknowns)
        self._inner_operator = _FunctionOperator(single_operator, n_unknowns)
        self._preconditioner = _CallbackPreconditioner(single_preconditioner, self.backend)
        self._bind_mixed_workspaces(self._operator.n_unknowns, matrix_free=True, jacobi=False)

    def _bind_mixed_workspaces(self, n_unknowns, matrix_free, jacobi):
        """(Re)allocate the float32 inner workspace and the float64 outer vectors when their size or kinds changed."""
        shape_key = (n_unknowns, np.float32, matrix_free, jacobi)
        fused = self._fused is not None
        if self._workspace is None or not self._workspace.fits(*shape_key, fused=fused):
            self._workspace = PCGWorkspace(*shape_key, backend=self.backend, fused=fused)
        if self._mixed is None or not self._mixed.fits(n_unknowns, matrix_free):
            self._mixed = _MixedWorkspace(n_unknowns, matrix_free, self.backend)

    def solve(self, rhs, solution, reference='rhs') -> bool:
        """PCG on A x = rhs from `solution` (updated in place) until ||r|| <= tolerance x the reference norm
        (STOP_REFERENCES); records the convergence and returns whether it converged."""
        if self._operator is None:
            raise RuntimeError(f'{self.label()}: factorize before solve')
        for name, vector in (('rhs', rhs), ('solution', solution)):
            if vector.dtype != self.dtype or vector.shape != (self._operator.n_unknowns,):
                raise ValueError(f'{self.label()}: {name} must be a flat {self.dtype.name} array of '
                                 f'{self._operator.n_unknowns} values, got {vector.dtype} {vector.shape}')
            self._require_backend_array(name, vector)
            if self.backend == 'gpu' and not vector.flags.c_contiguous:   # the gpu reductions read raw pointers
                raise ValueError(f'{self.label()}: {name} must be C-contiguous on the gpu backend')
        start = time.perf_counter()
        self._non_finite = False   # set by the loops when a residual norm is NaN / inf: they stop at once
        self._solve_setup_seconds = 0.0
        iterate = self._iterate_mixed if self.precision == 'mixed' else self._iterate_binding
        converged, iterations, reference_norm = iterate(rhs, solution, reference)
        self._record_convergence(converged, iterations, rhs, solution, reference_norm)
        self.total_solve_seconds += time.perf_counter() - start - self._solve_setup_seconds
        return converged

    def _iterate_binding(self, rhs, solution, reference):
        """_iterate on the current binding (module header): 'jacobi_small' sets the hierarchy up at the binding's second
        solve; a Jacobi bind or a kept hierarchy iterates within its budget, then continues from its iterate on the
        (fresh) hierarchy. Returns (converged, iterations, reference norm)."""
        binding = self._binding
        if binding == 'jacobi_small' and self._binding_solves:   # the operator is solved again: the setup pays
            self._bind_hierarchy(setup=True)
            binding = None
        self._binding_solves += 1
        budget = self._binding_budget(binding)
        limited = budget is not None and budget < self.max_iterations
        converged, iterations, reference_norm = self._iterate(rhs, solution, reference,
                                                              budget if limited else self.max_iterations)
        if limited and not converged and not self._non_finite:
            if binding == 'lagged':
                self.n_lagged_fallbacks += 1
            else:   # graded cells / contrasts: Jacobi needs more than a uniform mesh, so keep the hierarchy
                self.n_jacobi_fallbacks += 1
                self._jacobi_binds_off = True
            self._bind_hierarchy(setup=binding != 'jacobi_no_links')
            converged, more, _ = self._iterate(rhs, solution, reference, self.max_iterations - iterations,
                                               stop=self._last_stop)
            iterations += more
            return converged, iterations, reference_norm   # no lag record: the next factorize sets up afresh
        if self._multigrid_bound and binding in (None, 'lagged') and iterations > 0:
            if self._fresh_iterations is None and binding is None:
                self._fresh_iterations = iterations
            self._hierarchy_iterations = iterations
        return converged, iterations, reference_norm

    def _binding_budget(self, binding):
        """Iteration budget of a solve on the binding (None: max_iterations), a multiple of the check interval: a
        Jacobi bind CG_ITERATIONS_PER_AXIS_CELL x sqrt(n), a kept hierarchy LAGGED_SETUP_ITERATION_RATIO x the fresh
        solve's + LAGGED_SETUP_SLACK."""
        if binding in ('jacobi_small', 'jacobi_no_links'):
            iterations = CG_ITERATIONS_PER_AXIS_CELL * math.sqrt(self._operator.n_unknowns)
        elif binding == 'lagged' and self._fresh_iterations is not None:
            iterations = LAGGED_SETUP_ITERATION_RATIO * self._fresh_iterations + LAGGED_SETUP_SLACK
        else:
            return None
        interval = self._check_interval()
        return interval * max(1, math.ceil(iterations / interval))

    def _bind_hierarchy(self, setup) -> None:
        """Inside a solve: bind the own hierarchy of the bound CSR (set up first when `setup`; its time is factorize
        time) with new block graphs; the workspace gains the preconditioned vector."""
        started = time.perf_counter()
        if self._multigrid is None:
            from .multigrid import MultigridPreconditioner
            self._multigrid = MultigridPreconditioner(self.backend)
        if setup:
            self._multigrid.setup(self._operator.matrix)
        self._fresh_iterations = self._hierarchy_iterations = None
        self._binding = None
        self._preconditioner = _CallbackPreconditioner(self._multigrid.precondition, self.backend)
        shape_key = (self._operator.n_unknowns, self.dtype, False, False)
        if not self._workspace.fits(*shape_key):
            self._workspace = PCGWorkspace(*shape_key, backend=self.backend)
        self._multigrid_bound = True
        if self._capture_blocks and self.backend == 'gpu':
            self._block_graphs = _BlockGraphs((self._multigrid,))
        elapsed = time.perf_counter() - started
        self.total_factorize_seconds += elapsed
        self._solve_setup_seconds += elapsed

    def _check_interval(self) -> int:
        """CG iterations between residual reads of the bound preconditioner: the 'multigrid' interval only for a
        callback bound under that setting (a Jacobi bind of 'multigrid' reads like Jacobi)."""
        setting = 'jacobi' if self._preconditioner.jacobi else self.preconditioner
        return check_interval(setting, self.backend)

    def _iterate(self, rhs, solution, reference, max_iterations, stop=None):
        """The CG loop of at most max_iterations; stop = (threshold ||r||^2, reference norm) of an earlier attempt of
        this solve, else from ||b|| / ||r0|| (both read at once); returns (converged, iterations, reference norm)."""
        cp = _array_module(self.backend)
        vector_dot = _kernels(self.backend)['vector_dot']
        operator, workspace = self._operator, self._workspace
        residual = workspace.residual
        vector_dot(rhs, rhs, out=workspace.start_slots[0])
        cp.subtract(rhs, operator.product(solution, workspace.product), out=residual)
        vector_dot(residual, residual, out=workspace.start_slots[1])
        rhs_squared, initial_residual_squared = workspace.start_norms.tolist()   # one host read
        if rhs_squared == 0.0:   # b = 0: the solution of the nonsingular system is 0
            solution.fill(0)
            return True, 0, 0.0
        if stop is None:
            stop = stop_threshold_squared(self.tolerance, rhs_squared, initial_residual_squared, reference, self.dtype)
        self._last_stop = stop
        threshold_squared, reference_norm = stop
        converged = initial_residual_squared <= threshold_squared
        iterations = 0
        if not math.isfinite(initial_residual_squared):   # NaN / inf rhs, material or start: nothing to iterate
            self._non_finite = True
        elif not converged:
            converged, iterations = self._cg_loop(operator, solution, threshold_squared, max_iterations)
        return converged, iterations, reference_norm

    def _cg_loop(self, operator, solution, threshold_squared, max_iterations):
        """PCG from the residual already in the workspace until ||r||^2 <= threshold_squared; (converged, iterations).
        Blocks of an even length start with the scalars (and fused directions) in their initial order, so a captured
        block replays from any block start."""
        vector_dot = _kernels(self.backend)['vector_dot']
        residual = self._workspace.residual
        self._cg_start()
        interval = self._check_interval()
        iterations, converged, parity = 0, False, 0
        while not converged and iterations < max_iterations:
            n_block = min(interval, max_iterations - iterations)
            parity = self._cg_block(operator, solution, n_block, parity)
            iterations += n_block
            residual_squared = float(vector_dot(residual, residual))
            if not math.isfinite(residual_squared):   # a collapsed preconditioner / NaN operator: stop now
                self._non_finite = True
                break
            converged = residual_squared <= threshold_squared
        return converged, iterations

    def _cg_start(self):
        """From the workspace residual: z0, rho = r0 . z0, p0 = z0 (fused: and q0 = A p0 with its p . q partials)."""
        workspace = self._workspace
        if self._fused is not None:
            self._fused.start(workspace.residual, workspace.preconditioned, workspace.direction, workspace.product,
                              workspace.scalars[1], workspace.scalars[0])
        else:
            self._preconditioner.start(workspace.residual, workspace.scalars[0], workspace.direction, workspace)

    def _cg_block(self, operator, solution, n_block, parity) -> int:
        """n_block CG iterations from parity (a full even block from parity 0 replays its CUDA graph); the new parity."""
        fused, graphs = self._fused, self._block_graphs
        run_block = self._fused_block if fused is not None else self._plain_block
        if graphs is not None and n_block % 2 == 0 and parity == 0:
            graphs.run((id(operator), id(fused), solution.data.ptr, solution.size, n_block),
                       lambda: run_block(operator, solution, n_block, 0))
        else:
            run_block(operator, solution, n_block, parity)
        return (parity + n_block) % 2

    def _plain_block(self, operator, solution, n_block, parity):
        """n_block CG iterations: product, p . q, then the preconditioner's update (solution / residual, rho_next = r .
        z, direction; Jacobi on the gpu: one fused pass for the first two); parity 1 starts with rho in scalars[1]."""
        vector_dot = _kernels(self.backend)['vector_dot']
        preconditioner, workspace = self._preconditioner, self._workspace
        rho, rho_next, direction_dot_product = workspace.scalars
        if parity:
            rho, rho_next = rho_next, rho
        for _ in range(n_block):
            product = operator.product(workspace.direction, workspace.product)
            vector_dot(workspace.direction, product, out=direction_dot_product)
            preconditioner.update(product, direction_dot_product, rho, rho_next, solution, workspace)
            rho, rho_next = rho_next, rho

    def _fused_block(self, operator, solution, n_block, parity):
        """n_block CG iterations of the caller's fused_iteration: step = x += alpha p, r -= alpha q, z = M^-1 r, rho_next
        = r . z, then the next direction (the other direction buffer), its product and p . q; parity 1 starts from
        scalars[1] / next_direction."""
        workspace = self._workspace
        directions, scalars = (workspace.direction, workspace.next_direction), workspace.scalars
        for step in range(n_block):
            current = (parity + step) % 2
            self._fused.step(solution, workspace.residual, workspace.preconditioned, directions[current],
                             directions[1 - current], workspace.product, scalars[current], scalars[1 - current])

    def _iterate_mixed(self, rhs, solution, reference):
        """Mixed: one float32 PCG on A d = r / s (r = b - A x in float64, s = ||r||: unit scale, so float32 dots of small
        residuals cannot underflow) with residual replacement: when the recurrence residual fell by
        MIXED_REPLACEMENT_RATIO (or met the stop), x += s d, r is recomputed in float64 and replaces it (direction kept
        within MIXED_DRIFT_LIMIT, else a restart). Stops at the float64 reference, the iteration limit, or a stall."""
        vector_dot = _kernels(self.backend)['vector_dot']
        mixed, inner = self._mixed, self._workspace
        vector_dot(rhs, rhs, out=mixed.start_slots[0])
        mixed.correction.fill(0)
        self._fold_correction(rhs, solution, 1.0, out=mixed.start_slots[1])
        rhs_squared, residual_squared = mixed.start_norms.tolist()   # one host read
        if rhs_squared == 0.0:
            solution.fill(0)
            return True, 0, 0.0
        threshold_squared, reference_norm = stop_threshold_squared(
            self.tolerance, rhs_squared, residual_squared, reference, self.dtype)
        converged = residual_squared <= threshold_squared
        if not math.isfinite(residual_squared):
            self._non_finite = True
        if converged or self._non_finite:
            return converged, 0, reference_norm
        scale = math.sqrt(residual_squared)
        _array_module(self.backend).multiply(mixed.residual, 1.0 / scale, out=inner.residual)   # float32, unit norm
        if not float(vector_dot(inner.residual, inner.residual)) > 0.0:   # nothing representable to correct
            return False, 0, reference_norm
        self._cg_start()
        interval = self._check_interval()
        iterations, refinements, parity, pending = 0, 0, 0, False
        replaced_squared = 1.0   # unit-scale recurrence residual at the last replacement
        while iterations < self.max_iterations:
            n_block = min(interval, self.max_iterations - iterations)
            parity = self._cg_block(self._inner_operator, mixed.correction, n_block, parity)
            iterations += n_block
            pending = True
            inner_squared = float(vector_dot(inner.residual, inner.residual))
            if not math.isfinite(inner_squared):
                self._non_finite = True
                break
            met = inner_squared <= threshold_squared / residual_squared
            if not met and inner_squared > MIXED_REPLACEMENT_RATIO ** 2 * replaced_squared:
                continue
            updated_squared = self._fold_correction(rhs, solution, scale)
            refinements, pending = refinements + 1, False
            if not math.isfinite(updated_squared):
                self._non_finite = True
                break
            converged = updated_squared <= threshold_squared
            # float32 floor: the true residual no longer follows a recurrence that kept shrinking
            stalled = (not updated_squared < MIXED_STALL_RATIO ** 2 * residual_squared
                       and inner_squared <= MIXED_STALL_RATIO ** 2 * replaced_squared)
            if converged or stalled:
                break
            new_scale = math.sqrt(updated_squared)
            # true / recurrence residual norm in the old unit: a drifted recurrence (float32 operator error on a stiff
            # or nearly singular system) leaves the direction meaningless, so CG restarts from the true residual
            recurrence_norm = math.sqrt(inner_squared)
            drift = new_scale / (scale * recurrence_norm) if recurrence_norm > 0.0 else math.inf
            keep_direction = 1.0 / MIXED_DRIFT_LIMIT <= drift <= MIXED_DRIFT_LIMIT
            parity = self._replace_inner_residual(new_scale, scale / new_scale, parity, keep_direction)
            scale, residual_squared, replaced_squared = new_scale, updated_squared, 1.0
        if pending:   # the iteration limit (or a non-finite read) left a correction outside x
            residual_squared = self._fold_correction(rhs, solution, scale)
            refinements += 1
            converged = residual_squared <= threshold_squared
        self.total_refinements += refinements
        return converged, iterations, reference_norm

    def _fold_correction(self, rhs, solution, scale, out=None):
        """x += scale d (the float32 correction, then zeroed), r = b - A x in float64; returns ||r||^2 (out: written
        into that 0-d float64 array instead, no read, returns None)."""
        cp = _array_module(self.backend)
        mixed = self._mixed
        mixed.correction *= np.float32(scale)   # into the units of the residual (float32 range)
        cp.add(solution, mixed.correction, out=solution)
        mixed.correction.fill(0)
        cp.subtract(rhs, self._operator.product(solution, mixed.product), out=mixed.residual)
        squared = _kernels(self.backend)['vector_dot'](mixed.residual, mixed.residual, out=out)
        return None if out is not None else float(squared)

    def _replace_inner_residual(self, scale, factor, parity, keep_direction) -> int:
        """Float32 residual := r / scale. keep_direction: the recurrence state (current direction, rho; fused: its
        product and p . q too) x factor (rho x factor^2) into that unit, the CG coefficients stay consistent (van der
        Vorst-Ye residual replacement); else CG restarts from r. Returns the parity to continue from."""
        inner = self._workspace
        _array_module(self.backend).multiply(self._mixed.residual, 1.0 / scale, out=inner.residual)
        if not keep_direction:
            self._cg_start()   # p = z, rho = r . z in the first slot
            return 0
        if self._fused is not None:
            self._fused.rescale((inner.direction, inner.next_direction)[parity], inner.product, factor)
        else:
            inner.direction *= np.float32(factor)
        rho = inner.scalars[parity]   # 0-d device array inside a tuple: scale in place, no tuple item assignment
        rho *= np.float32(factor * factor)
        return parity

    def relative_residual(self, rhs, solution, reference_norm) -> float:
        """||rhs - A solution|| / reference_norm (absolute when it is 0), recomputed through the operator; overwrites the
        workspace residual."""
        cp = _array_module(self.backend)
        workspace = self._mixed if self.precision == 'mixed' else self._workspace
        cp.subtract(rhs, self._operator.product(solution, workspace.product), out=workspace.residual)
        residual = workspace.residual
        residual_norm = float(cp.linalg.norm(residual))
        return residual_norm / reference_norm if reference_norm > 0.0 else residual_norm

    def _record_convergence(self, converged, iterations, rhs, solution, reference_norm):
        self.n_solves += 1
        self.last_iters = iterations
        self.total_iters += iterations
        self.last_converged = bool(converged)
        if self.compute_residual or not self.last_converged:
            self.last_residual = self.relative_residual(rhs, solution, reference_norm)
        else:
            self.last_residual = float('nan')
        if not self.last_converged:
            self.n_unconverged += 1
            if self._non_finite:
                self.last_warning = (f'{self.label()}: CG stopped after {iterations} iterations on a non-finite '
                                     'residual (NaN / inf in the right-hand side, the materials or the preconditioner)')
            else:
                self.last_warning = (f'{self.label()}: CG did not converge in {iterations} iterations (relative '
                                     f'residual {self.last_residual:.2e}, tol {self.tolerance:.1e}, precision '
                                     f'{self.precision})')

    def work_counters(self) -> dict:
        """Machine-independent work of every solve so far (summary 'solvers' record, selftest timing); mixed adds its
        float64 residual refreshes, multigrid its setups / aggregations / levels, size and setup wall times."""
        counters = {'solves': self.n_solves, 'cg_iterations': self.total_iters,
                    'unconverged_solves': self.n_unconverged, 'factorizations': self.n_factorizations,
                    'solve_seconds': self.total_solve_seconds, 'factorize_seconds': self.total_factorize_seconds}
        if self.precision == 'mixed':
            counters['refinements'] = self.total_refinements
        if self._multigrid is not None or self.n_jacobi_binds:   # the summary's 'multigrid' that ran Jacobi
            counters.update({'multigrid_jacobi_binds': self.n_jacobi_binds,
                             'multigrid_jacobi_fallbacks': self.n_jacobi_fallbacks,
                             'multigrid_lagged_fallbacks': self.n_lagged_fallbacks})
        if self._multigrid is not None:
            counters.update(self._multigrid.work_counters())
        if self.preconditioner_counters is not None:
            counters.update(self.preconditioner_counters())
        return counters


def check_interval(preconditioner, backend='gpu') -> int:
    """CG iterations between host residual reads for a preconditioner setting value on the backend."""
    if backend == 'cpu':
        return 1 if preconditioner == 'multigrid' else CPU_CG_CHECK_INTERVAL
    return MULTIGRID_CHECK_INTERVAL if preconditioner == 'multigrid' else CG_CHECK_INTERVAL


def bound_preconditioner(values, work) -> str:
    """The preconditioner a pcg role runs (time recipe): 'jacobi' for the own CSR 'multigrid' below
    MULTIGRID_MIN_UNKNOWNS with one factorization and one solve (PCGSolver binds Jacobi there; the no-strong-link
    bind depends on values, so it is priced as multigrid), else the setting."""
    preconditioner = values.get('preconditioner')
    if (preconditioner == 'multigrid' and MULTIGRID_JACOBI_BINDS and not (work.preconditioner_ops or work.setup_ops)
            and work.unknowns < MULTIGRID_MIN_UNKNOWNS and not work.per_step and not work.factor_per_step
            and work.factorizations <= 1 and work.solves <= 1):
        return 'jacobi'
    return preconditioner


def cold_solve_iterations(values, work, fraction=1.0) -> int:
    """A priori CG iterations of a cold solve to the node's tolerance (times fraction: a warm start shrinks the
    residual, not the modes in it, so it applies before the cap), rounded up to the residual check block: multigrid
    from MULTIGRID_RESIDUAL_REDUCTION, Jacobi from work.jacobi_condition (0: CG_ITERATIONS_PER_AXIS_CELL x the longest
    axis), both at most work.distinct_modes (0: no cap)."""
    tolerance = float(values.get('tolerance') or 1e-8)
    if values.get('precision') == 'single':
        tolerance = max(tolerance, FLOAT32_TOLERANCE_FLOOR)
    preconditioner = bound_preconditioner(values, work)
    if preconditioner == 'multigrid':
        iterations = math.log(1.0 / tolerance) / math.log(1.0 / MULTIGRID_RESIDUAL_REDUCTION)
    elif work.jacobi_condition > 0.0:
        iterations = 0.5 * math.sqrt(work.jacobi_condition) * math.log(2.0 / tolerance)
    else:
        iterations = CG_ITERATIONS_PER_AXIS_CELL * max(int(work.longest_axis), 1)
    iterations *= fraction
    if work.distinct_modes > 0.0:
        iterations = min(iterations, work.distinct_modes)
    interval = check_interval(preconditioner, work.backend)
    return interval * math.ceil(max(iterations, 1.0) / interval)


def time_estimate(values, work):
    """Run-time recipe of a pcg role (core.time_estimate.SolverWork): CG iterations (a priori per cold solve, warm
    starts a fraction), per-solve overhead and setups; ops only."""
    from ..core import time_estimate as te
    precision = values.get('precision', 'double')
    n_unknowns = float(work.unknowns)
    item_bytes = 4 if precision in ('single', 'mixed') else 8   # mixed iterates in float32
    preconditioner = bound_preconditioner(values, work)
    multigrid = preconditioner == 'multigrid'
    product = work.product_ops or (te.Spmv(n_unknowns, work.nnz, item_bytes),)
    update = (te.elementwise(n_unknowns, 4, 2, item_bytes),)   # x += alpha p, r -= alpha q
    # preconditioner.start of a solve: z0 = M^-1 r0, rho = r0 . z0, the first direction (Jacobi: z0 into it)
    start = (te.Reduction(n_unknowns, item_bytes), te.elementwise(n_unknowns, 2, 1, item_bytes))
    if work.preconditioner_ops:
        precondition = work.preconditioner_ops + (te.Reduction(n_unknowns, item_bytes),
                                                  te.elementwise(n_unknowns, 3, 1, item_bytes))
        start = work.preconditioner_ops + (te.Reduction(n_unknowns, item_bytes),
                                           te.elementwise(n_unknowns, 1, 1, item_bytes))
    elif multigrid:
        from .multigrid import v_cycle_ops
        precondition = v_cycle_ops(n_unknowns, work.nnz, item_bytes, work.backend) + (
            te.Reduction(n_unknowns, item_bytes), te.elementwise(n_unknowns, 3, 1, item_bytes))
        start = v_cycle_ops(n_unknowns, work.nnz, item_bytes, work.backend) + (
            te.Reduction(n_unknowns, item_bytes), te.elementwise(n_unknowns, 1, 1, item_bytes))
    elif work.backend == 'cpu':   # Jacobi: its dot and the direction update
        precondition = (te.Reduction(n_unknowns, item_bytes), te.elementwise(n_unknowns, 3, 1, item_bytes))
    else:   # gpu Jacobi: the update and r . z in one pass (+ the partial-sum launch), z fused into the direction update
        update = (te.elementwise(n_unknowns, 5, 2, item_bytes, launches=2),)
        precondition = (te.elementwise(n_unknowns, 3, 1, item_bytes),)
    iteration = product + (te.Reduction(n_unknowns, item_bytes),) + update + precondition
    if work.iteration_ops:   # the physics' fused iteration (fused_iteration: product, dots, updates in its launches)
        iteration = work.iteration_ops
        start = work.iteration_ops
    iteration = iteration + ((1.0 / check_interval(preconditioner, work.backend),
                              te.Reduction(n_unknowns, item_bytes, read_back=True)),)
    # ||b||^2 and ||r0||^2 (read back together), r0 = b - A x0, then the preconditioner start (the solve's first
    # iteration follows only when r0 is above the stop threshold: counted always, an upper bound)
    per_solve = product + (te.Reduction(n_unknowns, item_bytes, read_back=True),
                           te.Reduction(n_unknowns, item_bytes),
                           te.elementwise(n_unknowns, 2, 1, item_bytes)) + start
    limit = int(values.get('max_iterations') or 0) or auto_max_iterations(work.longest_axis)
    cold = min(cold_solve_iterations(values, work), limit)
    warm = min(cold_solve_iterations(values, work, WARM_START_ITERATION_FRACTION), limit)
    solves = float(work.solves)
    cold_solves = min(float(work.cold_solves), solves) if work.warm_start else solves
    total = cold_solves * cold + (solves - cold_solves) * warm
    items = [te.TimeItem('CG iterations', 'iteration', total, 'guess', iteration,
                         limit=solves * limit, per_step=work.per_step, counter=work.counter('cg_iterations'),
                         seconds_counter=work.counter('solve_seconds')),
             te.TimeItem('CG solves', 'solve', solves, work.solves_kind, per_solve, per_step=work.per_step,
                         counter=work.counter('solves'), seconds_counter=work.counter('solve_seconds'))]
    if precision == 'mixed':   # the float64 residual refresh: the matrix-free double product, else a CSR Spmv
        refinement = (work.refinement_ops or (te.Spmv(n_unknowns, work.nnz, 8),)) + (
            te.Reduction(n_unknowns, 8, read_back=True), te.elementwise(n_unknowns, 4, 3, 8))
        # one replacement per MIXED_REPLACEMENT_RATIO of the tolerance (warm: a fraction), at most one per check block
        tolerance = float(values.get('tolerance') or 1e-8)
        cold_replacements = max(1, math.ceil(math.log(tolerance) / math.log(MIXED_REPLACEMENT_RATIO)))
        warm_replacements = max(1, math.ceil(WARM_START_ITERATION_FRACTION * cold_replacements))
        replacements = cold_solves * cold_replacements + (solves - cold_solves) * warm_replacements
        per_solve_limit = limit // check_interval(preconditioner, work.backend) + 1
        items.append(te.TimeItem('mixed CG refinements', 'refinement', replacements, 'guess',
                                 refinement, limit=solves * per_solve_limit, per_step=work.per_step,
                                 counter=work.counter('refinements'), seconds_counter=work.counter('solve_seconds')))
    if work.setup_ops:   # a caller-built preconditioner's setup (solid_mechanics_3d geometric multigrid)
        setup = work.setup_ops
    elif multigrid:
        from .multigrid import setup_ops
        setup = setup_ops(n_unknowns, work.nnz, item_bytes, work.backend)
    else:
        setup = (te.elementwise(n_unknowns, 1, 1, item_bytes),)
    count_name, seconds_name = work.setup_counters or ('factorizations', 'factorize_seconds')
    items.append(te.TimeItem('CG setup', 'setup', float(work.factorizations), work.factorizations_kind, setup,
                             per_step=work.factor_per_step, counter=work.counter(count_name),
                             seconds_counter=work.counter(seconds_name)))
    return te.TimeEstimate(tuple(items))
