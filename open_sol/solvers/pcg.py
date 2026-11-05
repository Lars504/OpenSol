"""'pcg' LinearSolver runtime: preconditioned CG on an assembled CSR or matrix-free operator, device-side scalars, the
residual read on the host every check_interval iterations; 'mixed' = float32 CG corrections (a float32 CSR copy, or
the caller's float32 matrix-free callbacks) refined against the float64 residual. Backend 'gpu' (cupy / cupyx CSR,
fused kernels built lazily) or 'cpu' (numpy / scipy CSR, the numpy twins of the kernels, BLAS axpy and scipy's CSR
kernel into preallocated vectors); cupy and scipy only inside functions."""
import math
import time

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
# mixed: each float32 inner CG reduces the float64 residual by this factor at most (well above the float32 floor);
# an outer step that does not shrink ||r|| by MIXED_STALL_RATIO stops the solve (float32 operator floor reached)
MIXED_INNER_TOLERANCE = 1e-4
MIXED_STALL_RATIO = 0.5
MIXED_MAX_REFINEMENTS = 20
CG_ITERATIONS_PER_AXIS_CELL = 10       # Jacobi-PCG cold start: ~10 iterations per cell along the longest window axis
AUTO_CG_MAX_ITERATIONS_FLOOR = 5000
# run-time estimate, a priori (never fitted to run counters: a deviation is a lead on the solver or the model):
# multigrid PCG reduces the residual by this factor per iteration whatever the mesh (textbook smoothed aggregation
# V(1,1) on a diffusion operator), so a cold solve takes ln(1 / tol) / ln(1 / factor) iterations; Jacobi PCG takes the
# CG bound sqrt(kappa) ln(2 / tol) / 2 of the condition number its physics estimates (SolverWork.jacobi_condition),
# at most SolverWork.distinct_modes (finite termination). Guesses: warm-started solves (time steps, fixed-point and
# nonlinear re-solves) as a fraction of a cold one, mixed outer refinements per solve
MULTIGRID_RESIDUAL_REDUCTION = 0.2
WARM_START_ITERATION_FRACTION = 0.25
MIXED_REFINEMENTS_GUESS = 3
# Stop reference of a solve: 'rhs' = ||b|| (steady solves: cheap warm-started re-solves); 'initial_residual' = ||b -
# A x0|| of the warm start (implicit time steps: x0 = previous step, so the tolerance is relative to the step's change;
# against ||b|| the capacity term C/dt u_old dominates and CG stops before the step moves: PHY-1)
STOP_REFERENCES = ('rhs', 'initial_residual')
ROUNDOFF_FLOOR_EPSILONS = 16           # 'initial_residual' never asks for less than this x eps(dtype) x ||b||

_KERNELS = {}
_HOST_AXPY = {}   # dtype -> scipy BLAS axpy of the cpu backend (fetched on first use)

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


class _DeviceCsrProduct:
    """gpu y = A x of one cupyx CSR (float32 / float64, int32 indices) into one owned vector by csr_product."""

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
        if vector.dtype != self.dtype or not vector.flags.c_contiguous:
            raise ValueError(f'gpu CSR product: the vector must be C-contiguous {self.dtype.name}, got {vector.dtype}')
        matrix = self.matrix   # read per call: callers write new values into matrix.data in place
        self._function(self._grid, (CSR_THREADS,), (self._n_rows, matrix.indptr, matrix.indices, matrix.data, vector,
                                                     self.out))
        return self.out


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


def _host_axpy(alpha, x, y):
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


def _host_vector_dot(a, b, out=None):
    total = np.dot(a, b)
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


def workspace_bytes(n_unknowns, precision, jacobi=True, csr_entries=0) -> int:
    """Device bytes of the PCGSolver workspace (residual, direction, product (matrix-free workspace, the gpu CSR
    operator's or the cpu workspace's), + preconditioned unless Jacobi); mixed: those in float32 plus the float64 residual and product, float32
    correction, (Jacobi) inverse diagonal and the float32 copy of the csr_entries operator values. No solution vector;
    a matrix-free mixed caller counts its own float32 operator copy."""
    n_vectors = 3 + int(not jacobi)
    if precision == 'mixed':
        return (n_vectors * int(n_unknowns) * 4 + 3 * 4
                + (8 + 8 + 4 + 4 * int(jacobi)) * int(n_unknowns) + 4 * int(csr_entries))
    item_bytes = np.dtype(PRECISION_DTYPES[precision]).itemsize
    return n_vectors * int(n_unknowns) * item_bytes + 3 * item_bytes


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
            self._device_product = _DeviceCsrProduct(matrix)
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
            rho[...] = np.dot(residual, direction)
            return
        _kernels(self.backend)['jacobi_residual_dot'](residual, self.inverse_diagonal, out=rho)
        _array_module(self.backend).multiply(self.inverse_diagonal, residual, out=direction)

    def update(self, product, direction_dot_product, rho_old, rho_new, solution, workspace):
        residual, direction = workspace.residual, workspace.direction
        if self.backend == 'cpu':
            _host_update_solution(direction, product, rho_old, direction_dot_product, solution, residual)
            preconditioned = np.multiply(self.inverse_diagonal, residual, out=workspace.product)
            rho_new[...] = np.dot(residual, preconditioned)
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
    preconditioned unless Jacobi) and the three device scalars."""

    __slots__ = ('n_unknowns', 'dtype', 'residual', 'direction', 'product', 'preconditioned', 'scalars')

    def __init__(self, n_unknowns, dtype, matrix_free, jacobi, backend='gpu'):
        cp = _array_module(backend)
        self.n_unknowns = int(n_unknowns)
        self.dtype = np.dtype(dtype)
        self.residual = cp.zeros(self.n_unknowns, dtype=self.dtype)
        self.direction = cp.zeros(self.n_unknowns, dtype=self.dtype)
        self.product = cp.zeros(self.n_unknowns, dtype=self.dtype) if matrix_free or backend == 'cpu' else None
        self.preconditioned = None if jacobi else cp.zeros(self.n_unknowns, dtype=self.dtype)
        self.scalars = tuple(cp.zeros((), dtype=self.dtype) for _ in range(3))

    def fits(self, n_unknowns, dtype, matrix_free, jacobi) -> bool:
        return (self.n_unknowns == n_unknowns and self.dtype == np.dtype(dtype)
                and (self.product is not None or not matrix_free) and (self.preconditioned is not None or jacobi))


# --- runtime object -----------------------------------------------------------------------------------------------

class _MixedWorkspace:
    """Outer vectors of a 'mixed' solve: float64 residual, float32 correction (the inner CG solution) and, matrix-free,
    the float64 product."""

    __slots__ = ('n_unknowns', 'residual', 'correction', 'product')

    def __init__(self, n_unknowns, matrix_free, backend='gpu'):
        cp = _array_module(backend)
        self.n_unknowns = int(n_unknowns)
        self.residual = cp.zeros(self.n_unknowns, dtype=np.float64)
        self.correction = cp.zeros(self.n_unknowns, dtype=np.float32)
        self.product = cp.zeros(self.n_unknowns, dtype=np.float64) if matrix_free else None

    def fits(self, n_unknowns, matrix_free) -> bool:
        return self.n_unknowns == n_unknowns and (self.product is not None or not matrix_free)


# 'jacobi': the physics' diagonal (block) preconditioner; 'multigrid': solvers.multigrid V-cycles built from an
# assembled scalar CSR operator (DiffusionSystem; not mixed, not matrix-free, not a block preconditioner)
PRECONDITIONERS = ('jacobi', 'multigrid')


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
        if preconditioner == 'multigrid' and precision == 'mixed':
            raise ValueError("the multigrid preconditioner needs precision 'single' or 'double' (mixed iterates on "
                             'a Jacobi float32 operator copy)')
        self.preconditioner = preconditioner
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
                  single_preconditioner=None) -> None:
        """Bind a CSR operator of the backend (cupyx / scipy) or a matrix-free apply(vector, out) (then n_unknowns
        required); preconditioner None (Jacobi from the CSR diagonal), an inverse diagonal or precondition(residual, out).
        Mixed matrix-free: single_operator / single_preconditioner = their float32 twins for the inner CG."""
        start = time.perf_counter()
        self._factorize(operator, preconditioner, n_unknowns, single_operator, single_preconditioner)
        self.total_factorize_seconds += time.perf_counter() - start
        self.n_factorizations += 1

    def _factorize(self, operator, preconditioner, n_unknowns, single_operator, single_preconditioner):
        if not callable(operator):
            self._require_backend_array('operator', operator.data)
        if not (preconditioner is None or callable(preconditioner)):
            self._require_backend_array('inverse diagonal', preconditioner)
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
        if self._workspace is None or not self._workspace.fits(*shape_key):
            self._workspace = PCGWorkspace(*shape_key, backend=self.backend)

    def _multigrid_preconditioner(self, operator, preconditioner):
        """precondition callback of the (re)built multigrid hierarchy of a CSR operator; the inverse diagonal a
        DiffusionSystem passes is not needed (block preconditioners and matrix-free operators raise TypeError)."""
        if callable(operator) or callable(preconditioner):
            raise TypeError(f'{self.label()}: the multigrid preconditioner needs an assembled scalar CSR operator '
                            '(heat, electric_currents, electrostatics, magnetostatics), not a matrix-free operator '
                            'or a block preconditioner')
        if operator.dtype != self.dtype:
            raise ValueError(f'{self.label()}: operator must be {self.dtype.name}, got {operator.dtype}')
        if self._multigrid is None:
            from .multigrid import MultigridPreconditioner
            self._multigrid = MultigridPreconditioner(self.backend)
        self._multigrid.setup(operator)
        return self._multigrid.precondition

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
        if self._workspace is None or not self._workspace.fits(*shape_key):
            self._workspace = PCGWorkspace(*shape_key, backend=self.backend)
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
        iterate = self._iterate_mixed if self.precision == 'mixed' else self._iterate
        converged, iterations, reference_norm = iterate(rhs, solution, reference)
        self._record_convergence(converged, iterations, rhs, solution, reference_norm)
        self.total_solve_seconds += time.perf_counter() - start
        return converged

    def _iterate(self, rhs, solution, reference):
        """The CG loop; returns (converged, iterations, reference norm)."""
        cp = _array_module(self.backend)
        vector_dot = _kernels(self.backend)['vector_dot']
        operator, workspace = self._operator, self._workspace
        residual = workspace.residual
        rhs_squared = float(vector_dot(rhs, rhs))
        if rhs_squared == 0.0:   # b = 0: the solution of the nonsingular system is 0
            solution.fill(0)
            return True, 0, 0.0
        cp.subtract(rhs, operator.product(solution, workspace.product), out=residual)
        initial_residual_squared = float(vector_dot(residual, residual))
        threshold_squared, reference_norm = stop_threshold_squared(
            self.tolerance, rhs_squared, initial_residual_squared, reference, self.dtype)
        converged = initial_residual_squared <= threshold_squared
        iterations = 0
        if not math.isfinite(initial_residual_squared):   # NaN / inf rhs, material or start: nothing to iterate
            self._non_finite = True
        elif not converged:
            converged, iterations = self._cg_loop(operator, solution, threshold_squared, self.max_iterations)
        return converged, iterations, reference_norm

    def _cg_loop(self, operator, solution, threshold_squared, max_iterations):
        """PCG from the residual already in the workspace until ||r||^2 <= threshold_squared; (converged, iterations)."""
        vector_dot = _kernels(self.backend)['vector_dot']
        preconditioner, workspace = self._preconditioner, self._workspace
        residual, direction = workspace.residual, workspace.direction
        rho, rho_next, direction_dot_product = workspace.scalars
        preconditioner.start(residual, rho, direction, workspace)
        interval = check_interval(self.preconditioner, self.backend)
        iterations, converged = 0, False
        while not converged and iterations < max_iterations:
            n_block = min(interval, max_iterations - iterations)
            for _ in range(n_block):
                product = operator.product(direction, workspace.product)
                vector_dot(direction, product, out=direction_dot_product)
                # solution / residual update, rho_next = r . z, direction (Jacobi: one fused pass for the first two)
                preconditioner.update(product, direction_dot_product, rho, rho_next, solution, workspace)
                rho, rho_next = rho_next, rho
            iterations += n_block
            residual_squared = float(vector_dot(residual, residual))
            if not math.isfinite(residual_squared):   # a collapsed preconditioner / NaN operator: stop now
                self._non_finite = True
                break
            converged = residual_squared <= threshold_squared
        return converged, iterations

    def _iterate_mixed(self, rhs, solution, reference):
        """Mixed: float64 residual r = b - A x, float32 PCG on A d = r / ||r|| (unit scale: float32 dots of small
        residuals cannot underflow; to MIXED_INNER_TOLERANCE, or less when that suffices), x += ||r|| d; repeated until
        the float64 residual meets the stop reference, the iteration limit, or an outer step that no longer shrinks
        ||r|| by MIXED_STALL_RATIO."""
        cp = _array_module(self.backend)
        vector_dot = _kernels(self.backend)['vector_dot']
        mixed, inner = self._mixed, self._workspace
        rhs_squared = float(vector_dot(rhs, rhs))
        if rhs_squared == 0.0:
            solution.fill(0)
            return True, 0, 0.0
        cp.subtract(rhs, self._operator.product(solution, mixed.product), out=mixed.residual)
        residual_squared = float(vector_dot(mixed.residual, mixed.residual))
        threshold_squared, reference_norm = stop_threshold_squared(
            self.tolerance, rhs_squared, residual_squared, reference, self.dtype)
        iterations, refinements = 0, 0
        converged = residual_squared <= threshold_squared
        if not math.isfinite(residual_squared):
            self._non_finite = True
        while (not converged and not self._non_finite and iterations < self.max_iterations
               and refinements < MIXED_MAX_REFINEMENTS):
            inner_tolerance = max(MIXED_INNER_TOLERANCE, 0.5 * math.sqrt(threshold_squared / residual_squared))
            residual_norm = math.sqrt(residual_squared)
            cp.multiply(mixed.residual, 1.0 / residual_norm, out=inner.residual)   # float64 -> float32, unit norm
            mixed.correction.fill(0)
            inner_squared = float(vector_dot(inner.residual, inner.residual))
            if not inner_squared > 0.0:   # non-finite residual: nothing to correct
                break
            _, inner_iterations = self._cg_loop(self._inner_operator, mixed.correction,
                                                inner_tolerance ** 2 * inner_squared, self.max_iterations - iterations)
            iterations += inner_iterations
            refinements += 1
            mixed.correction *= np.float32(residual_norm)   # back to the units of the residual (float32 range)
            cp.add(solution, mixed.correction, out=solution)
            cp.subtract(rhs, self._operator.product(solution, mixed.product), out=mixed.residual)
            updated_squared = float(vector_dot(mixed.residual, mixed.residual))
            stalled = not updated_squared < MIXED_STALL_RATIO ** 2 * residual_squared
            residual_squared = updated_squared
            converged = residual_squared <= threshold_squared
            if stalled:
                break
        self.total_refinements += refinements
        return converged, iterations, reference_norm

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
        if self._multigrid is not None:
            counters.update(self._multigrid.work_counters())
        return counters


def check_interval(preconditioner, backend='gpu') -> int:
    """CG iterations between host residual reads for a preconditioner setting value on the backend."""
    if backend == 'cpu':
        return 1 if preconditioner == 'multigrid' else CPU_CG_CHECK_INTERVAL
    return MULTIGRID_CHECK_INTERVAL if preconditioner == 'multigrid' else CG_CHECK_INTERVAL


def cold_solve_iterations(values, work, fraction=1.0) -> int:
    """A priori CG iterations of a cold solve to the node's tolerance (times fraction: a warm start shrinks the
    residual, not the modes in it, so it applies before the cap), rounded up to the residual check block: multigrid
    from MULTIGRID_RESIDUAL_REDUCTION, Jacobi from work.jacobi_condition (0: CG_ITERATIONS_PER_AXIS_CELL x the longest
    axis), both at most work.distinct_modes (0: no cap)."""
    tolerance = float(values.get('tolerance') or 1e-8)
    if values.get('precision') == 'single':
        tolerance = max(tolerance, FLOAT32_TOLERANCE_FLOOR)
    preconditioner = values.get('preconditioner')
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
    multigrid = values.get('preconditioner') == 'multigrid'
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
    iteration = product + (te.Reduction(n_unknowns, item_bytes),) + update + precondition \
        + ((1.0 / check_interval(values.get('preconditioner'), work.backend),
            te.Reduction(n_unknowns, item_bytes, read_back=True)),)
    # ||b||^2 and ||r0||^2 read back, r0 = b - A x0, then the preconditioner start (the solve's first iteration
    # follows only when r0 is above the stop threshold: counted always, an upper bound)
    per_solve = product + (te.Reduction(n_unknowns, item_bytes, read_back=True),
                           te.Reduction(n_unknowns, item_bytes, read_back=True),
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
        items.append(te.TimeItem('mixed CG refinements', 'refinement', solves * MIXED_REFINEMENTS_GUESS, 'guess',
                                 refinement, limit=solves * MIXED_MAX_REFINEMENTS, per_step=work.per_step,
                                 counter=work.counter('refinements'), seconds_counter=work.counter('solve_seconds')))
    if multigrid:
        from .multigrid import setup_ops
        setup = setup_ops(n_unknowns, work.nnz, item_bytes)
    else:
        setup = (te.elementwise(n_unknowns, 1, 1, item_bytes),)
    items.append(te.TimeItem('CG setup', 'setup', float(work.factorizations), work.factorizations_kind, setup,
                             per_step=work.factor_per_step, counter=work.counter('factorizations'),
                             seconds_counter=work.counter('factorize_seconds')))
    return te.TimeEstimate(tuple(items))
