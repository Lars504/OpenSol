"""Smoothed-aggregation algebraic multigrid: the 'multigrid' preconditioner of the pcg solver (solver_plan S7.0) on
assembled SPD CSR operators (DiffusionSystem), one symmetric V(1,1)-cycle per application, hierarchy in float32.
Aggregation on the host (numpy), Galerkin products on the backend (cupy / cupyx or numpy / scipy); gpu cycles by
fused RawKernels replayed as CUDA graphs, cpu cycles by scipy's CSR kernel into level vectors."""
import time
import warnings

import numpy as np

from .cudss_factor import csr_rows
from .pcg import host_csr_matvec

# -a_ij >= theta sqrt(a_ii a_jj) is a strong link; weaker ones (across a 3:1 cell, into a 100x poorer conductor) are not
# coarsened together, so graded / stretched cells and contrasts semi-coarsen along their strong direction
STRENGTH_THRESHOLD = 0.1
# a row whose negative couplings sum to at least this x a_ii keeps its largest one as a strong link even below the
# threshold (strong_links: spread-thin Galerkin coarse rows); below it the diagonal dominates (a large C/dt or loss)
# and the smoother alone converges, so no link is forced (0: every row, inf: none)
FLOOR_ROW_SUM_RATIO = 0.5
# l1-Jacobi weight SMOOTHER_WEIGHT / sum_j |a_ij| (rows without off-diagonals: 1 / a_ii, exact): convergent for every SPD A
# (2 M - A = M_l1 / 2 + (M_l1 - A) > 0), 2/3-damped Jacobi on an interior Laplacian row; also the prolongator smoothing
SMOOTHER_WEIGHT = 4.0 / 3.0
COARSEST_UNKNOWNS = 400          # no further coarsening at or below this size
COARSEST_DENSE_LIMIT = 4000      # coarsest level by a dense pseudo-inverse up to this size, else by smoothing sweeps
COARSEST_SMOOTHING_SWEEPS = 8    # the fallback when coarsening stalls above COARSEST_DENSE_LIMIT
MIN_COARSENING = 0.8             # stop coarsening when a level would keep more than this fraction of its unknowns
MAX_LEVELS = 25
MIS_MAX_ROUNDS = 100             # distance-2 independent set rounds (O(log n) in practice); leftovers become roots
PSEUDO_INVERSE_CUTOFF = 1e-12    # coarsest eigenvalues below this x the largest are dropped
AGGREGATION_SEED = 0             # tie-break order of the independent set: deterministic hierarchies
# CPU estimate (2D 5-point and 3D 7-point smoothed aggregation, operator complexity ~1.5-2): stored CSR entries per
# fine unknown (tentative T, prolongation P, restriction R, coarse operators) and vectors per fine unknown (weights,
# residual, coarse solutions, the PCG preconditioned vector)
# (filtered smoothing: 11.96 measured on a 2.16e6-cell 3D mesh graded 2.5 um - 2 mm, operator complexity 1.81)
HIERARCHY_ENTRIES_PER_UNKNOWN = 12
SETUP_ENTRIES_PER_UNKNOWN = 8    # transient stacked P blocks and coarse products per fine unknown
HIERARCHY_VECTORS_PER_UNKNOWN = 4
HIERARCHY_ROWS_PER_UNKNOWN = 4   # CSR indptr rows of T, P, R and the coarse operators per fine unknown
# host aggregation of the fine operator: its copy plus int64 / bool temporaries (+16: the symmetrised int64 edge keys
# of strong_links and their sort)
HOST_BYTES_PER_ENTRY = 56
HOST_BYTES_PER_UNKNOWN = 80
# Galerkin products in row blocks of at most this many partial products: one cuSPARSE SpGEMM over a 5e6-row 3D operator
# asked for a 6.5 GB workspace; PRODUCT_WORKSPACE_BYTES is one full block's (~32 bytes per partial product)
PRODUCT_BLOCK_PRODUCTS = 1 << 23
PRODUCT_BYTES_PER_PRODUCT = 32
PRODUCT_WORKSPACE_BYTES = PRODUCT_BYTES_PER_PRODUCT * PRODUCT_BLOCK_PRODUCTS
# share of the n entries_per_row^2 bound a small operator's whole A @ P takes (magnet3d_magnetized_sphere coarse,
# 149k unknowns at 7 per row: the bound gave ~2x the measured setup peak; the capped fine case is unaffected)
SETUP_PRODUCT_SHARE = 0.5
# _filtered_rows peak per entry of its row block (at most PRODUCT_BLOCK_PRODUCTS entries: A T has one product per
# entry): int64 row index, float64 diagonal product and its gather, strength bools; a block that is not the whole
# operator adds its own copy (value + int32 index: setup_extra_bytes)
FILTER_BYTES_PER_ENTRY = 28
# T, P, R, the coarse operators and every coarse-level vector, whatever the fine precision: the cycle is SPD for any
# symmetric A_c (l1 weights from the same A_c, coarsest pseudo-inverse keeps positive eigenvalues only), so the
# float32 rounding changes the preconditioner, not CG's float64 answer
HIERARCHY_DTYPE = np.float32
HIERARCHY_VALUE_BYTES = 4


def hierarchy_bytes(n_unknowns, value_bytes, entries_per_row) -> int:
    """Device bytes of the hierarchy of an n_unknowns operator with entries_per_row (CPU): HIERARCHY_DTYPE matrices,
    vectors (value_bytes: the fine precision, conservative) and the cached nonzero pattern of the fine operator."""
    n_unknowns = int(n_unknowns)
    return int(n_unknowns * (HIERARCHY_ENTRIES_PER_UNKNOWN * (HIERARCHY_VALUE_BYTES + 4)
                             + HIERARCHY_ROWS_PER_UNKNOWN * 4 + HIERARCHY_VECTORS_PER_UNKNOWN * value_bytes
                             + HIERARCHY_VALUE_BYTES + entries_per_row))


def setup_extra_bytes(n_unknowns, value_bytes, entries_per_row) -> int:
    """Transient device bytes of one setup on top of hierarchy_bytes: the stacked P blocks and coarse products, the l1
    row sums and the block planning (4-byte pattern data per entry), plus the largest of the fine level's three steps
    on one row block, which follow each other (_build_level): the strength filter, the smoothing product A_F T on the
    filtered block it returns, the Galerkin product (a small operator's whole A @ P: n entries_per_row^2 bounds it)."""
    n_unknowns, entries_per_row = int(n_unknowns), int(entries_per_row)
    per_unknown = (SETUP_ENTRIES_PER_UNKNOWN * (HIERARCHY_VALUE_BYTES + 4) + 2 * value_bytes
                   + 4 * entries_per_row + 12)
    entries = n_unknowns * entries_per_row
    block_entries = min(entries, PRODUCT_BLOCK_PRODUCTS)
    sliced = block_entries < entries   # a row slice is a copy, the whole operator is taken as it is (_rows)
    strength_filter = (FILTER_BYTES_PER_ENTRY + ((value_bytes + 4) if sliced else 0)) * block_entries
    # the filter's temporaries are gone: its block (values, int32 columns), the HIERARCHY_DTYPE copy, one partial
    # product per entry (T holds one entry per row)
    smoothing = (value_bytes + 4 + HIERARCHY_VALUE_BYTES + PRODUCT_BYTES_PER_PRODUCT) * block_entries
    # A @ P on the block's HIERARCHY_DTYPE copy (a slice: with its own columns); P rows hold no more entries than A's
    # (SETUP_PRODUCT_SHARE of that bound)
    products = SETUP_PRODUCT_SHARE * n_unknowns * entries_per_row ** 2
    galerkin = (min(PRODUCT_WORKSPACE_BYTES, int(PRODUCT_BYTES_PER_PRODUCT * products))
                + (HIERARCHY_VALUE_BYTES + (4 if sliced else 0)) * block_entries)
    return n_unknowns * per_unknown + max(strength_filter, smoothing, galerkin)


def aggregation_host_bytes(n_unknowns, entries_per_row) -> int:
    """Transient host bytes of one aggregation of the fine operator."""
    return int(n_unknowns) * (HOST_BYTES_PER_ENTRY * int(entries_per_row) + HOST_BYTES_PER_UNKNOWN)


# run-time recipe (CPU, uncalibrated: compare with the pcg factorize_seconds / solve_seconds of summary.json): unknowns
# kept per coarsening step, operator complexity, host passes over the aggregation bytes, device passes over the setup
# entries (Galerkin products, smoothing of P)
COARSENING_RATIO_GUESS = 0.2
OPERATOR_COMPLEXITY_GUESS = 1.8
AGGREGATION_HOST_PASSES = 10
SETUP_DEVICE_PASSES = 20


def level_count_guess(n_unknowns) -> int:
    """Levels of a hierarchy of n_unknowns (the coarsest included) at COARSENING_RATIO_GUESS."""
    levels, size = 1, float(n_unknowns)
    while size > COARSEST_UNKNOWNS and levels < MAX_LEVELS:
        size *= COARSENING_RATIO_GUESS
        levels += 1
    return levels


def v_cycle_ops(n_unknowns, nnz, value_bytes, backend='gpu') -> tuple:
    """Ops of one V(1,1)-cycle (core.time_estimate): two operator products per level (the levels' entries =
    OPERATOR_COMPLEXITY_GUESS x nnz; gpu: fused with the smoothing), restriction / prolongation, the smoothing vector
    passes and the power-of-two scaling; gpu: one CUDA graph replay of the whole cycle (_DeviceCycle) on the host."""
    from ..core import time_estimate as te
    levels = level_count_guess(n_unknowns)
    level_rows = n_unknowns / (1.0 - COARSENING_RATIO_GUESS)
    transfer_entries = HIERARCHY_ENTRIES_PER_UNKNOWN * n_unknowns / 4.0
    products = ((2.0, te.Spmv(level_rows, OPERATOR_COMPLEXITY_GUESS * nnz, HIERARCHY_VALUE_BYTES)),
                (2.0, te.Spmv(level_rows, transfer_entries, HIERARCHY_VALUE_BYTES)))
    if backend == 'gpu':   # per level the correction add; abs-max partials, exponent, scale in and out
        return products + (te.elementwise(level_rows, 2, 1, value_bytes, launches=levels),
                           te.elementwise(n_unknowns, 3, 2, value_bytes, launches=4), te.PythonCalls(4))
    return products + (te.elementwise(level_rows, 12, 6, value_bytes, launches=6 * levels),
                       te.PythonCalls(10 * levels))


def setup_ops(n_unknowns, nnz, value_bytes) -> tuple:
    """Ops of one setup: host aggregation of the fine pattern (a reused pattern skips it: counted as if not),
    transfers, Galerkin products on the backend, the coarsest dense pseudo-inverse."""
    from ..core import time_estimate as te
    entries_per_row = nnz / max(n_unknowns, 1.0)
    host_bytes = aggregation_host_bytes(n_unknowns, max(int(round(entries_per_row)), 1))
    return (te.HostStream(AGGREGATION_HOST_PASSES * host_bytes), te.Transfer(nnz * (value_bytes + 4)),
            te.Elementwise(SETUP_DEVICE_PASSES * SETUP_ENTRIES_PER_UNKNOWN * n_unknowns * (HIERARCHY_VALUE_BYTES + 4),
                           0.0, launches=50 * level_count_guess(n_unknowns)),
            te.DenseFlops(float(COARSEST_UNKNOWNS) ** 3 * 4.0, host=True))


# --- host aggregation (numpy) ---------------------------------------------------------------------------------------

def _row_max(edge_values, strong_indptr, fill):
    """Per row the maximum of edge_values over its strong links (row-sorted edges), `fill` for rows without any."""
    n_rows = strong_indptr.size - 1
    result = np.full(n_rows, fill, dtype=edge_values.dtype)
    nonempty = np.diff(strong_indptr) > 0
    if edge_values.size:
        result[nonempty] = np.maximum.reduceat(edge_values, strong_indptr[:-1][nonempty])
    return result


def strong_links(indptr, indices, data, n_unknowns):
    """(strong_indptr, strong_columns) of the symmetric strength graph of a host CSR operator: off-diagonal a_ij < 0
    (both diagonals > 0) with -a_ij >= STRENGTH_THRESHOLD sqrt(a_ii a_jj), plus the largest such coupling of each row
    whose couplings sum to >= FLOOR_ROW_SUM_RATIO a_ii, every edge kept both ways."""
    indptr = np.asarray(indptr, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    data = np.asarray(data, dtype=np.float64)
    rows = csr_rows(indptr)
    on_diagonal = rows == indices
    diagonal = np.bincount(rows[on_diagonal], weights=data[on_diagonal], minlength=n_unknowns)
    candidate = ~on_diagonal & (data < 0.0)
    candidate_rows, candidate_columns = rows[candidate], indices[candidate]   # row-sorted (CSR order)
    scale = diagonal[candidate_rows] * diagonal[candidate_columns]
    coupling = np.where(scale > 0.0, -data[candidate], 0.0)
    strong = (coupling > 0.0) & (coupling >= STRENGTH_THRESHOLD * np.sqrt(np.maximum(scale, 0.0)))
    # Floor: a row with spread couplings keeps its largest one. A 3D Galerkin coarse operator spreads each row over
    # dozens of couplings, all below the threshold: without it ~98 % of the level-1 rows of a 7-point Laplacian had no
    # strong link, stayed out of P and got no coarse correction (PCG 49 iterations at 60^3, 26 with it; an anisotropic
    # operator keeps its threshold-made semi-coarsening). Not on diagonally dominated rows: forced there it coarsened
    # the C/dt-dominated cube_cooling_3d (8082 iterations over 1600 steps against 3237)
    candidate_indptr = np.zeros(n_unknowns + 1, dtype=np.int64)
    candidate_indptr[1:] = np.cumsum(np.bincount(candidate_rows, minlength=n_unknowns))
    spread = np.bincount(candidate_rows, weights=coupling, minlength=n_unknowns) >= FLOOR_ROW_SUM_RATIO * diagonal
    strong |= ((coupling > 0.0) & (coupling >= _row_max(coupling, candidate_indptr, 0.0)[candidate_rows])
               & spread[candidate_rows])
    edges = np.unique(np.concatenate((candidate_rows[strong] * n_unknowns + candidate_columns[strong],
                                      candidate_columns[strong] * n_unknowns + candidate_rows[strong])))
    strong_rows, strong_columns = np.divmod(edges, n_unknowns)   # sorted by row, then column
    strong_indptr = np.zeros(n_unknowns + 1, dtype=np.int64)
    strong_indptr[1:] = np.cumsum(np.bincount(strong_rows, minlength=n_unknowns))
    return strong_indptr, strong_columns


def _distance_two_roots(strong_indptr, strong_columns, n_unknowns):
    """Bool roots of a maximal distance-2 independent set (Luby-style rounds on random keys) of the strength graph;
    rows without strong links are never roots."""
    rank = np.random.default_rng(AGGREGATION_SEED).permutation(n_unknowns).astype(np.int64)
    state = np.zeros(n_unknowns, dtype=np.int64)            # 0 undecided, 1 root, -1 out
    state[np.diff(strong_indptr) == 0] = -1
    for _ in range(MIS_MAX_ROUNDS):
        undecided = state == 0
        if not undecided.any():
            break
        key = (state + 1) * n_unknowns + rank              # root > undecided > out, unique within each state
        best = key
        for _hop in range(2):
            best = np.maximum(best, _row_max(best[strong_columns], strong_indptr, -1))
        state[undecided & (best == key)] = 1                # largest undecided key within distance 2
        state[undecided & (best != key) & (best >= 2 * n_unknowns)] = -1   # a root within distance 2
    state[state == 0] = 1
    return state == 1


def aggregate_strong_links(indptr, indices, data, n_unknowns):
    """(aggregate index per unknown (-1: no strong link, left to the smoother), aggregate count) of a host CSR
    operator: distance-2 independent roots, each neighbour joins an adjacent root, the rest a neighbour's aggregate."""
    strong_indptr, strong_columns = strong_links(indptr, indices, data, n_unknowns)
    isolated = np.diff(strong_indptr) == 0
    roots = _distance_two_roots(strong_indptr, strong_columns, n_unknowns)
    aggregate = np.full(n_unknowns, -1, dtype=np.int64)
    n_aggregates = int(np.count_nonzero(roots))
    aggregate[roots] = np.arange(n_aggregates, dtype=np.int64)
    for joins_roots_only in (True, False):
        offered = aggregate[strong_columns]
        if joins_roots_only:
            offered = np.where(roots[strong_columns], offered, -1)
        candidate = _row_max(offered, strong_indptr, -1)
        join = (aggregate < 0) & ~isolated & (candidate >= 0)
        aggregate[join] = candidate[join]
    leftover = (aggregate < 0) & ~isolated                  # none after a complete independent set
    aggregate[leftover] = n_aggregates + np.arange(np.count_nonzero(leftover), dtype=np.int64)
    return aggregate, n_aggregates + int(np.count_nonzero(leftover))


# --- hierarchy and cycle (backend) -----------------------------------------------------------------------------------

class _Level:
    """One fine level: its operator (not owned at level 0), smoother weights, P, R = P^T and its work vectors (the
    residual, the coarse right-hand side R r and coarse solution: the cycle's products go into them, no allocation); on
    the cpu a float64 fine level also keeps a HIERARCHY_DTYPE copy of its residual for the restriction (the gpu
    restriction kernel reads the float64 residual)."""

    __slots__ = ('operator', 'weights', 'prolongation', 'restriction', 'residual', 'residual_low', 'coarse_solution',
                 'coarse_rhs')

    def __init__(self, operator, weights, prolongation, restriction, xp, host=False):
        self.operator = operator
        self.weights = weights
        self.prolongation = prolongation
        self.restriction = restriction
        self.residual = xp.zeros(operator.shape[0], dtype=operator.dtype)
        self.residual_low = (None if operator.dtype == prolongation.dtype or not host
                             else xp.zeros(operator.shape[0], dtype=prolongation.dtype))
        self.coarse_solution = xp.zeros(prolongation.shape[1], dtype=prolongation.dtype)
        self.coarse_rhs = xp.zeros(prolongation.shape[1], dtype=prolongation.dtype)

    def low_residual(self, xp):
        """The level residual in the hierarchy dtype (a float64 level casts it into residual_low first)."""
        if self.residual_low is None:
            return self.residual
        xp.copyto(self.residual_low, self.residual, casting='same_kind')
        return self.residual_low

    def prolongation_target(self):
        """Fine-size hierarchy-dtype vector free while P e is added (residual_low, else the residual)."""
        return self.residual if self.residual_low is None else self.residual_low


# gpu cycle (_DeviceCycle): fused RawKernels into the preallocated level vectors, 5 launches per level (pre-smoothing
# with its residual, restriction, prolongation add, smoothing correction, its add), 4 for the power-of-two scaling and
# the coarsest solve; each launch plan (one per (residual, out) pair: PCG passes the same workspace vectors every time)
# is captured once into a CUDA graph and replayed as one launch. At 4e4 unknowns the cupy / cupyx cycle was launch-bound
# (~1.5 ms of host work per V-cycle against microseconds of device work; heat_2d profile).
CUDA_GRAPH_CYCLES = True        # False: replay the launch plans eagerly (no capture)
GRAPH_AFTER_EAGER_CYCLES = 1    # eager applications of a plan before its capture (every kernel loaded once first)
MAX_CYCLE_PLANS = 4             # kept (residual, out) plans; a further pair clears them
_CYCLE_SOURCE = r'''
#define GRID_STRIDE(i, n) \
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (n); i += (long long)blockDim.x * gridDim.x)

__device__ __forceinline__ float magnitude(float x) { return fabsf(x); }
__device__ __forceinline__ double magnitude(double x) { return fabs(x); }
__device__ __forceinline__ float times_power_of_two(float x, int e) { return ldexpf(x, e); }
__device__ __forceinline__ double times_power_of_two(double x, int e) { return ldexp(x, e); }

template<typename T>
__device__ T block_max(T value) {   // blockDim.x == 256
    __shared__ T partial[256];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride && partial[threadIdx.x + stride] > partial[threadIdx.x])
            partial[threadIdx.x] = partial[threadIdx.x + stride];
        __syncthreads();
    }
    return partial[0];
}

// pre-smoothing from zero: solution = w rhs, residual = rhs - A solution
template<int LANES, typename T>
__global__ void mg_presmooth(int n_rows, const int* indptr, const int* indices, const T* data, const T* weights,
                             const T* rhs, T* solution, T* residual) {
    CSR_ROW
    const T sum = csr_row_sum<LANES, T, T, T>(indptr, indices, data, rhs, weights, row, active);
    if (active && first_lane) {
        solution[row] = weights[row] * rhs[row];
        residual[row] = rhs[row] - sum;
    }
}

// smoothing correction = w (rhs - A solution)
template<int LANES, typename T>
__global__ void mg_correction(int n_rows, const int* indptr, const int* indices, const T* data, const T* weights,
                              const T* rhs, const T* solution, T* correction) {
    CSR_ROW
    const T sum = csr_row_sum<LANES, T, T, T>(indptr, indices, data, solution, (const T*)0, row, active);
    if (active && first_lane) correction[row] = weights[row] * (rhs[row] - sum);
}

// y += P e, the product summed in the hierarchy type M
template<int LANES, typename M, typename Y>
__global__ void mg_add_product(int n_rows, const int* indptr, const int* indices, const M* data, const M* x, Y* y) {
    CSR_ROW
    const M sum = csr_row_sum<LANES, M, M, M>(indptr, indices, data, x, (const M*)0, row, active);
    if (active && first_lane) y[row] += (Y)sum;
}

template<typename T>
__global__ void mg_add(long long n, const T* source, T* target) {
    GRID_STRIDE(i, n) target[i] += source[i];
}

template<typename T>
__global__ void mg_scale(long long n, const T* weights, const T* rhs, T* solution) {
    GRID_STRIDE(i, n) solution[i] = weights[i] * rhs[i];
}

// coarsest y = M x of a row-major n x n matrix, one warp per row
template<typename T>
__global__ void mg_dense_product(int n, const T* matrix, const T* x, T* y) {
    const long long thread = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const int row = (int)(thread / 32);
    const int lane = threadIdx.x & 31;
    T sum = (T)0;
    if (row < n)
        for (int column = lane; column < n; column += 32) sum += matrix[(long long)row * n + column] * x[column];
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (row < n && lane == 0) y[row] = sum;
}

// max |x| per block (grid-stride), then one block: the frexp exponent e of the largest (max = m 2^e, m in [0.5, 1);
// 0 for x = 0)
template<typename T>
__global__ void mg_abs_max_partials(long long n, const T* x, T* partials) {
    T largest = (T)0;
    GRID_STRIDE(i, n) {
        const T value = magnitude(x[i]);
        if (value > largest) largest = value;
    }
    largest = block_max(largest);
    if (threadIdx.x == 0) partials[blockIdx.x] = largest;
}

template<typename T>
__global__ void mg_exponent(const T* partials, int n_partials, int* exponent) {
    T largest = (T)0;
    for (int i = threadIdx.x; i < n_partials; i += blockDim.x)
        if (partials[i] > largest) largest = partials[i];
    largest = block_max(largest);
    if (threadIdx.x == 0) {
        int e = 0;
        frexp((double)largest, &e);
        *exponent = e;
    }
}

// y = x 2^(sign e)
template<typename T>
__global__ void mg_scale_exponent(long long n, const T* x, const int* exponent, int sign, T* y) {
    const int e = sign * *exponent;
    GRID_STRIDE(i, n) y[i] = times_power_of_two(x[i], e);
}
'''


def _cycle_functions() -> dict:
    """{name expression: function} of the cycle kernels (one module per device, compiled on first use)."""
    from .pcg import CSR_LANE_CHOICES, CSR_ROW_SOURCE, device_template_functions
    hierarchy = 'float'
    names = []
    for lanes in CSR_LANE_CHOICES:
        for t in ('float', 'double'):
            names += [f'csr_product<{lanes}, {hierarchy}, {t}, {hierarchy}>', f'mg_presmooth<{lanes}, {t}>',
                      f'mg_correction<{lanes}, {t}>', f'mg_add_product<{lanes}, {hierarchy}, {t}>']
    for t in ('float', 'double'):
        names += [f'mg_{name}<{t}>' for name in ('add', 'scale', 'dense_product', 'abs_max_partials', 'exponent',
                                                 'scale_exponent')]
    return device_template_functions('multigrid_cycle', CSR_ROW_SOURCE + _CYCLE_SOURCE, names)


class _DeviceCycle:
    """gpu precondition of one built hierarchy: per (residual, out) pair a launch plan of fused RawKernels into the
    level vectors, replayed eagerly or as one captured CUDA graph (CUDA_GRAPH_CYCLES)."""

    def __init__(self, levels, coarsest, scaled_residual):
        import cupy as cp
        from .pcg import CSR_THREADS, REDUCTION_BLOCKS_PER_SM
        self._cp = cp
        self._levels = levels
        self._coarsest = coarsest
        self._scaled = scaled_residual
        self._functions = _cycle_functions()
        self._threads = CSR_THREADS
        self._max_blocks = REDUCTION_BLOCKS_PER_SM * int(cp.cuda.Device().attributes['MultiProcessorCount'])
        self._partial_blocks = self._grid_stride_blocks(scaled_residual.size)
        self._partials = cp.zeros(self._partial_blocks, dtype=scaled_residual.dtype)
        self._exponent = cp.zeros((), dtype=np.int32)
        self._plans = {}           # (residual pointer, out pointer, size) -> [launches, eager uses, graph]
        self.use_graphs = CUDA_GRAPH_CYCLES
        self.n_graphs = 0          # plans captured into a CUDA graph

    def apply(self, residual, out):
        """out = the scaled V-cycle of residual (C-contiguous vectors of the fine dtype)."""
        key = (residual.data.ptr, out.data.ptr, residual.size)
        plan = self._plans.get(key)
        if plan is None:
            if len(self._plans) >= MAX_CYCLE_PLANS:
                self._plans.clear()
            plan = self._plans[key] = [self._plan(residual, out), 0, None]
        launches, uses, graph = plan
        if graph is None and self.use_graphs and uses >= GRAPH_AFTER_EAGER_CYCLES:
            graph = plan[2] = self._capture(launches)
        if graph is not None:
            graph.launch(self._cp.cuda.get_current_stream())
            return
        for function, grid, block, arguments in launches:
            function(grid, block, arguments)
        plan[1] = uses + 1

    def _capture(self, launches):
        """CUDA graph of a launch plan (None, with a warning and eager plans from now on, when capture fails)."""
        cp = self._cp
        stream = cp.cuda.Stream(non_blocking=True)
        try:
            with stream:
                stream.begin_capture()
                try:
                    for function, grid, block, arguments in launches:
                        function(grid, block, arguments)
                finally:
                    graph = stream.end_capture()
        except Exception as error:   # capture unsupported here: the same kernels launched one by one
            self.use_graphs = False
            warnings.warn(f'multigrid: CUDA graph capture of the V-cycle failed ({type(error).__name__}: {error}); '
                          'launching its kernels one by one from now on', RuntimeWarning, stacklevel=3)
            return None
        self.n_graphs += 1
        return graph

    def _plan(self, residual, out):
        """[(function, grid, block, arguments)] of one application: scale r by 2^-e (max |r| in [0.5, 1)), the cycle
        into out, out x 2^e."""
        dtype = self._scaled.dtype
        for name, vector in (('residual', residual), ('out', out)):
            if vector.dtype != dtype or vector.shape != self._scaled.shape or not vector.flags.c_contiguous:
                raise ValueError(f'multigrid: {name} must be a C-contiguous {dtype.name} vector of '
                                 f'{self._scaled.size} values, got {vector.dtype} {vector.shape}')
        from .pcg import type_name
        t = type_name(dtype)
        n = self._scaled.size
        launches = [(self._functions[f'mg_abs_max_partials<{t}>'], (self._partial_blocks,), (self._threads,),
                     (np.int64(n), residual, self._partials)),
                    (self._functions[f'mg_exponent<{t}>'], (1,), (self._threads,),
                     (self._partials, np.int32(self._partial_blocks), self._exponent)),
                    self._elementwise('scale_exponent', residual, (residual, self._exponent, np.int32(-1),
                                                                   self._scaled))]
        self._cycle_launches(0, self._scaled, out, launches)
        launches.append(self._elementwise('scale_exponent', out, (out, self._exponent, np.int32(1), out)))
        return launches

    def _cycle_launches(self, depth, rhs, solution, launches):
        if depth == len(self._levels):
            self._coarsest_launches(rhs, solution, launches)
            return
        level = self._levels[depth]
        hierarchy = 'float'
        from .pcg import type_name
        t = type_name(level.operator.dtype)
        _require_dtypes(level, rhs, solution)
        launches.append(self._csr(f'mg_presmooth<{{lanes}}, {t}>', level.operator,
                                  (level.weights, rhs, solution, level.residual)))
        launches.append(self._csr(f'csr_product<{{lanes}}, {hierarchy}, {t}, {hierarchy}>', level.restriction,
                                  (level.residual, level.coarse_rhs)))
        self._cycle_launches(depth + 1, level.coarse_rhs, level.coarse_solution, launches)
        launches.append(self._csr(f'mg_add_product<{{lanes}}, {hierarchy}, {t}>', level.prolongation,
                                  (level.coarse_solution, solution)))
        launches.append(self._csr(f'mg_correction<{{lanes}}, {t}>', level.operator,
                                  (level.weights, rhs, solution, level.residual)))
        launches.append(self._elementwise('add', solution, (level.residual, solution)))

    def _coarsest_launches(self, rhs, solution, launches):
        from .pcg import type_name
        t = type_name(rhs.dtype)
        if self._coarsest[0] == 'dense':
            inverse = self._coarsest[1]
            n_unknowns = int(inverse.shape[0])
            if n_unknowns:
                launches.append((self._functions[f'mg_dense_product<{t}>'],
                                 (max(1, -(-n_unknowns * 32 // self._threads)),), (self._threads,),
                                 (np.int32(n_unknowns), inverse, rhs, solution)))
            return
        _, operator, weights, residual = self._coarsest
        launches.append(self._elementwise('scale', solution, (weights, rhs, solution)))
        for _ in range(COARSEST_SMOOTHING_SWEEPS - 1):
            launches.append(self._csr(f'mg_correction<{{lanes}}, {t}>', operator, (weights, rhs, solution, residual)))
            launches.append(self._elementwise('add', solution, (residual, solution)))

    def _csr(self, name_pattern, matrix, vectors):
        """Launch of a CSR row kernel (name_pattern with '{lanes}') over matrix's rows: (n_rows, indptr, indices,
        data, *vectors)."""
        from .pcg import csr_grid, csr_lanes
        n_rows = int(matrix.shape[0])
        lanes = csr_lanes(n_rows, int(matrix.nnz))
        if matrix.indptr.dtype != np.int32 or matrix.indices.dtype != np.int32:
            raise TypeError(f'multigrid: gpu cycle CSR needs int32 indices, got {matrix.indices.dtype}')
        return (self._functions[name_pattern.format(lanes=lanes)], csr_grid(n_rows, lanes), (self._threads,),
                (np.int32(n_rows), matrix.indptr, matrix.indices, matrix.data) + tuple(vectors))

    def _elementwise(self, name, vector, arguments):
        """Grid-stride launch of mg_<name> over the vector's size in its dtype: (n, *arguments)."""
        from .pcg import type_name
        return (self._functions[f'mg_{name}<{type_name(vector.dtype)}>'],
                (self._grid_stride_blocks(vector.size),), (self._threads,), (np.int64(vector.size),) + tuple(arguments))

    def _grid_stride_blocks(self, n_values) -> int:
        return max(1, min(self._max_blocks, -(-int(n_values) // self._threads)))


def _require_dtypes(level, rhs, solution):
    """ValueError unless a level's vectors and matrices have the dtypes the gpu cycle kernels read."""
    fine = np.dtype(level.operator.dtype)
    expected = ((level.weights, fine), (rhs, fine), (solution, fine), (level.residual, fine),
                (level.coarse_rhs, HIERARCHY_DTYPE), (level.coarse_solution, HIERARCHY_DTYPE),
                (level.restriction.data, HIERARCHY_DTYPE), (level.prolongation.data, HIERARCHY_DTYPE))
    for array, dtype in expected:
        if np.dtype(array.dtype) != np.dtype(dtype) or not array.flags.c_contiguous:
            raise ValueError(f'multigrid: gpu cycle arrays must be C-contiguous {np.dtype(dtype).name}, got '
                             f'{array.dtype}')


class MultigridPreconditioner:
    """SA-AMG hierarchy of one CSR operator: setup(operator) (re)builds it, precondition(residual, out) applies one
    V-cycle; aggregates stay while the operator's nonzero pattern does, so value-only setups need no host transfer."""

    def __init__(self, backend='gpu'):
        from ..core.backend import array_module, sparse_module
        self.backend = backend
        self._xp = array_module(backend)
        self._sparse = sparse_module(backend)
        self._host_matvec = host_csr_matvec() if backend == 'cpu' else None   # cpu cycle products into level vectors
        self._levels = []
        self._aggregates = []        # host (aggregate, count) per coarsening, kept while the pattern stays
        self._tentatives = []        # backend tentative prolongators T of those aggregates
        self._pattern = None         # (indptr, indices, nonzero mask) the aggregates were built from
        self._coarsest = None        # ('dense', inverse) or ('smooth', operator, weights, residual)
        self._scaled_residual = None # fine-level copy of the residual scaled to max |r| in [0.5, 1) (precondition)
        self._device_cycle = None    # gpu: _DeviceCycle of the current hierarchy
        self.n_cycle_graphs = 0      # gpu: launch plans captured into CUDA graphs, all setups
        self.n_setups = 0
        self.n_aggregations = 0
        self.device_bytes = 0                # last setup's hierarchy, the fine operator excluded (_record_size)
        self.entries_per_unknown = 0.0       # stored T, P, R and coarse-operator entries per fine unknown
        self.operator_complexity = 1.0       # entries of every level operator / the fine operator's
        self.setup_seconds = {'aggregation': 0.0, 'products': 0.0, 'coarsest': 0.0}   # last setup's wall time

    @property
    def n_levels(self) -> int:
        """Levels of the current hierarchy, the coarsest included (1: no coarsening)."""
        return len(self._levels) + 1

    def setup(self, operator) -> None:
        """Build the hierarchy of a square backend CSR operator; re-aggregates only when its nonzero pattern changed."""
        from ..core.backend import synchronize, to_host
        if operator.shape[0] != operator.shape[1]:
            raise ValueError(f'multigrid needs a square operator, got shape {operator.shape}')
        seconds = {'aggregation': 0.0, 'products': 0.0, 'coarsest': 0.0}
        if self._device_cycle is not None:   # its plans and graphs hold the old hierarchy's vectors
            self.n_cycle_graphs += self._device_cycle.n_graphs
            self._device_cycle = None
        nonzero = operator.data != 0
        reuse = self._pattern_matches(operator, nonzero)
        if not reuse:
            self._aggregates, self._tentatives = [], []
            self._pattern = (operator.indptr, operator.indices, nonzero)
            self.n_aggregations += 1
        self._levels = []                  # drop the old hierarchy before building the new one
        current, depth = operator, 0
        while True:
            n_unknowns = current.shape[0]
            if reuse:
                if depth == len(self._aggregates):
                    break
            else:
                if n_unknowns <= COARSEST_UNKNOWNS or depth + 1 >= MAX_LEVELS:
                    break
                started = time.perf_counter()
                aggregate, n_aggregates = aggregate_strong_links(
                    to_host(current.indptr), to_host(current.indices), to_host(current.data), n_unknowns)
                seconds['aggregation'] += time.perf_counter() - started
                if n_aggregates == 0 or n_aggregates > MIN_COARSENING * n_unknowns:
                    break
                self._aggregates.append((aggregate, n_aggregates))
            started = time.perf_counter()
            level, current = self._build_level(current, depth)
            synchronize(self.backend)      # the products' device time lands here, not in the next aggregation
            seconds['products'] += time.perf_counter() - started
            self._levels.append(level)
            depth += 1
        started = time.perf_counter()
        self._coarsest = self._coarsest_solver(current)
        synchronize(self.backend)
        seconds['coarsest'] = time.perf_counter() - started
        if (self._scaled_residual is None or self._scaled_residual.shape != (operator.shape[0],)
                or self._scaled_residual.dtype != operator.dtype):
            self._scaled_residual = self._xp.zeros(operator.shape[0], dtype=operator.dtype)
        if self.backend == 'gpu':
            self._device_cycle = _DeviceCycle(self._levels, self._coarsest, self._scaled_residual)
        self.setup_seconds = seconds
        self._record_size(operator)
        self.n_setups += 1

    def _record_size(self, operator):
        """device_bytes / entries_per_unknown / operator_complexity of the hierarchy just built (the counters that
        calibrate hierarchy_bytes)."""
        coarse_operators = [level.operator for level in self._levels[1:]]
        coarsest_vectors = [self._coarsest[1]] if self._coarsest[0] == 'dense' else list(self._coarsest[2:])
        if self._coarsest[0] == 'smooth' and self._levels:
            coarse_operators.append(self._coarsest[1])
        matrices = [matrix for level in self._levels for matrix in (level.prolongation, level.restriction)]
        matrices += list(self._tentatives) + coarse_operators
        vectors = [vector for level in self._levels
                   for vector in (level.weights, level.residual, level.residual_low, level.coarse_solution,
                                  level.coarse_rhs)
                   if vector is not None] + coarsest_vectors + [self._scaled_residual]
        self.device_bytes = (sum(int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
                                 for matrix in matrices)
                             + sum(int(vector.nbytes) for vector in vectors) + int(self._pattern[2].nbytes))
        self.entries_per_unknown = sum(int(matrix.nnz) for matrix in matrices) / max(operator.shape[0], 1)
        self.operator_complexity = ((int(operator.nnz) + sum(int(matrix.nnz) for matrix in coarse_operators))
                                    / max(int(operator.nnz), 1))

    def _pattern_matches(self, operator, nonzero) -> bool:
        if self._pattern is None:
            return False
        indptr, indices, cached_nonzero = self._pattern
        xp = self._xp
        if indptr.shape != operator.indptr.shape or indices.shape != operator.indices.shape:
            return False
        return all(cached is current or bool(xp.array_equal(cached, current))
                   for cached, current in ((indptr, operator.indptr), (indices, operator.indices),
                                           (cached_nonzero, nonzero)))

    def _tentative(self, depth, n_unknowns, dtype):
        """Backend tentative prolongator of the cached aggregates at depth (columns normalised to unit length)."""
        if depth < len(self._tentatives) and self._tentatives[depth].dtype == dtype:
            return self._tentatives[depth]
        xp = self._xp
        aggregate, n_aggregates = self._aggregates[depth]
        member = aggregate >= 0
        sizes = np.bincount(aggregate[member], minlength=n_aggregates)
        indptr = np.zeros(n_unknowns + 1, dtype=np.int32)
        indptr[1:] = np.cumsum(member)
        tentative = self._sparse.csr_matrix(
            (xp.asarray(1.0 / np.sqrt(sizes[aggregate[member]]), dtype=dtype),
             xp.asarray(aggregate[member].astype(np.int32)), xp.asarray(indptr)), shape=(n_unknowns, n_aggregates))
        del self._tentatives[depth:]
        self._tentatives.append(tentative)
        return tentative

    def _row_l1(self, matrix, dtype):
        """(rows,) sum_j |a_ij| of a CSR block (one product with a ones vector of `dtype`)."""
        xp = self._xp
        absolute = self._sparse.csr_matrix((xp.abs(matrix.data), matrix.indices, matrix.indptr), shape=matrix.shape)
        return absolute.dot(xp.ones(matrix.shape[1], dtype=dtype))

    def _smoother_weights(self, operator):
        """l1-Jacobi weights SMOOTHER_WEIGHT / sum_j |a_ij|, 1 / |a_ii| on rows without off-diagonals, 0 on empty rows."""
        xp = self._xp
        row_l1 = self._row_l1(operator, operator.dtype)
        off_diagonal_l1 = row_l1 - xp.abs(operator.diagonal())
        weights = xp.where(off_diagonal_l1 > 0, SMOOTHER_WEIGHT, 1.0) / xp.where(row_l1 > 0, row_l1, 1.0)
        return xp.where(row_l1 > 0, weights, 0.0).astype(operator.dtype)

    def _build_level(self, operator, depth):
        """(_Level, coarse operator) of operator with the depth's aggregates: P = (I - W_F A_F) T (A_F: the filtered
        operator, `_filtered_rows`), R = P^T, A_c = R A P (Galerkin with the full A, so SPD) in HIERARCHY_DTYPE, in
        row blocks b (P_b = T_b - W_F,b A_F,b T, A_c = sum_b P_b^T (A_b P)) so no full A T or A P is ever held."""
        n_unknowns = operator.shape[0]
        tentative = self._tentative(depth, n_unknowns, HIERARCHY_DTYPE)
        weights = self._smoother_weights(operator)   # the cycle's smoother: the full operator
        diagonal = operator.diagonal()
        blocks = []
        for start, stop in self._row_blocks(operator, tentative.indptr):
            filtered, filtered_weights = self._filtered_rows(operator, start, stop, diagonal)
            blocks.append((self._rows(tentative, start, stop)
                           - self._diagonal(filtered_weights) @ (self._low(filtered) @ tentative)).tocsr())
            del filtered, filtered_weights
        del diagonal
        prolongation = self._stack_rows(blocks, tentative.shape[1])
        del blocks
        xp = self._xp
        restriction = None
        # binary-counter sum of the block contributions: merging only partial sums of equal block counts costs
        # O(nnz log blocks); a running sum recopies the whole coarse operator per block (297 s of a 303 s 3D build)
        partial_sums = []   # [(partial sum, number of blocks in it)], counts strictly decreasing
        for start, stop in self._row_blocks(operator, prolongation.indptr):
            block_product = self._low(self._rows(operator, start, stop)) @ prolongation   # A_b P
            # R_b (A_b P) takes nnz(P_i) nnz((A P)_i) partial products per row, more than A_b P's nnz(A_i) nnz(P_i),
            # so it runs in its own sub-blocks (the workspace setup_extra_bytes books). carey2026 E2 (5.6e6 cells)
            # still raises INSUFFICIENT_RESOURCES in this SpGEMM at <= 2^23 products: not a product-count limit
            outer_products = (xp.diff(prolongation.indptr[start:stop + 1]).astype(np.int64)
                              * xp.diff(block_product.indptr).astype(np.int64))
            for sub_start, sub_stop in self._blocks_of(outer_products):
                first, last = start + sub_start, start + sub_stop
                if first == 0 and last == n_unknowns:   # a single block's left factor is R itself (no 2nd transpose)
                    restriction = left = prolongation.T.tocsr()
                else:
                    left = prolongation[first:last].T.tocsr()
                partial_sum = left @ self._rows(block_product, sub_start, sub_stop)
                del left
                n_blocks = 1
                while partial_sums and partial_sums[-1][1] == n_blocks:
                    earlier_sum, earlier_blocks = partial_sums.pop()
                    partial_sum, n_blocks = earlier_sum + partial_sum, earlier_blocks + n_blocks
                partial_sums.append((partial_sum, n_blocks))
            del block_product, outer_products
        coarse_operator = partial_sums.pop()[0]
        while partial_sums:
            coarse_operator = partial_sums.pop()[0] + coarse_operator
        if restriction is None:
            restriction = prolongation.T.tocsr()
        return (_Level(operator, weights, prolongation, restriction, self._xp, host=self.backend == 'cpu'),
                coarse_operator.tocsr())

    def _row_blocks(self, left, right_indptr):
        """Host [(start, stop)] row blocks of left @ right (right's CSR indptr) with about PRODUCT_BLOCK_PRODUCTS
        partial products each (a longer single row stays one block)."""
        xp = self._xp
        # per-row counts as one float32 SpMV of left's pattern (exact: rows hold far fewer than 2^24 products), so the
        # only entry-sized temporary is the 4-byte pattern data, not int64 per-entry cumulative sums
        pattern = self._sparse.csr_matrix((xp.ones(left.nnz, dtype=np.float32), left.indices, left.indptr),
                                          shape=left.shape)
        products_per_row = pattern.dot(xp.diff(right_indptr).astype(np.float32))
        del pattern
        return self._blocks_of(products_per_row)

    def _blocks_of(self, products_per_row):
        """Host [(start, stop)] row blocks of about PRODUCT_BLOCK_PRODUCTS products each from backend per-row counts
        (a longer single row stays one block)."""
        from ..core.backend import to_host
        xp = self._xp
        n_rows = products_per_row.shape[0]
        row_products = xp.zeros(n_rows + 1, dtype=np.int64)
        xp.cumsum(products_per_row.astype(np.int64), out=row_products[1:])
        targets = xp.arange(PRODUCT_BLOCK_PRODUCTS, int(row_products[-1]), PRODUCT_BLOCK_PRODUCTS, dtype=np.int64)
        boundaries = np.asarray(to_host(xp.searchsorted(row_products, targets, side='right') - 1), dtype=np.int64)
        boundaries = np.unique(np.concatenate(([0], boundaries, [n_rows])))
        return list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))

    def _filtered_rows(self, operator, start, stop, diagonal):
        """(rows start:stop of A_F, their HIERARCHY_DTYPE l1 weights): the prolongator-smoothing operator keeps the
        diagonal and the strong links of aggregate_strong_links and lumps every weak entry into the diagonal (row sums
        kept), so P spreads only along strong links (unfiltered: operator complexity 9.4, a 292 s setup in 3D)."""
        # array operations only: cuSPARSE csr2csr_compress (eliminate_zeros) failed here with INSUFFICIENT_RESOURCES;
        # entry-sized temporaries in place where possible (FILTER_BYTES_PER_ENTRY: the block is the whole fine level
        # of most 2D / 3D operators)
        xp = self._xp
        block = self._rows(operator, start, stop)
        n_rows = stop - start
        columns = block.indices
        values = block.data
        local_rows = xp.searchsorted(block.indptr[1:], xp.arange(values.size, dtype=block.indptr.dtype), side='right')
        on_diagonal = (columns - start) == local_rows   # the global row of an entry is start + its local row
        scale = diagonal[start:stop][local_rows]
        scale *= diagonal[columns]
        del local_rows
        threshold = xp.maximum(scale, 0.0)   # strong: a_ij <= -theta sqrt(a_ii a_jj) (a_ij < 0, a_ii a_jj > 0)
        xp.sqrt(threshold, out=threshold)
        threshold *= -STRENGTH_THRESHOLD
        strong = (values < 0) & (scale > 0) & (values <= threshold)
        strong |= on_diagonal
        del scale, threshold
        weak = self._sparse.csr_matrix((xp.where(strong, 0.0, values).astype(values.dtype, copy=False), columns,
                                        block.indptr), shape=block.shape)
        lumped = weak.dot(xp.ones(operator.shape[1], dtype=values.dtype))   # per row: the dropped entries' sum
        del weak
        kept_per_entry = xp.concatenate((xp.zeros(1, dtype=np.int32), xp.cumsum(strong, dtype=np.int32)))
        kept_values = values[strong]   # the diagonal entries (strong) take their row's lumped sum below
        diagonal_entries = xp.flatnonzero(on_diagonal)
        kept_values[kept_per_entry[diagonal_entries]] += lumped[columns[diagonal_entries] - start]
        del on_diagonal, diagonal_entries
        filtered = self._sparse.csr_matrix(   # only stored strong entries: weak zeros would still be products
            (kept_values.astype(values.dtype, copy=False), columns[strong], kept_per_entry[block.indptr]),
            shape=(n_rows, operator.shape[1]))
        # l1 weights SMOOTHER_WEIGHT / sum_j |a_F,ij| (a row block: its diagonal is diagonal[start:stop] + lumped, not
        # (i, i)); 0 on rows without a strong off-diagonal, so P = T there. A fresh aggregation leaves such rows out of
        # every aggregate (T row 0), but a value-only setup reuses aggregates whose links may since have turned weak (a
        # growing diagonal), and the exact weight 1 / a_ii gave (I - W A_F) T = T - T = 0 on those rows: the coarse
        # operators collapsed to rounding level (4e6-unknown Laplacian, diagonal x (1 + u): coarsest pseudo-inverse
        # overflowed float32, CG never converged)
        row_l1 = self._row_l1(filtered, values.dtype)
        off_diagonal_l1 = row_l1 - xp.abs(diagonal[start:stop] + lumped)
        weights = xp.where(off_diagonal_l1 > 0, SMOOTHER_WEIGHT / xp.where(row_l1 > 0, row_l1, 1.0), 0.0)
        return filtered, weights.astype(HIERARCHY_DTYPE)

    @staticmethod
    def _rows(matrix, start, stop):
        """Rows start:stop of a CSR: the matrix itself when that is all of it (a full-range slice copies it)."""
        return matrix if start == 0 and stop == matrix.shape[0] else matrix[start:stop]

    @staticmethod
    def _low(matrix):
        """A CSR block in HIERARCHY_DTYPE (itself when it already is)."""
        return matrix if matrix.dtype == HIERARCHY_DTYPE else matrix.astype(HIERARCHY_DTYPE)

    def _diagonal(self, values):
        """Square diagonal CSR of a backend vector."""
        xp = self._xp
        n_rows = values.shape[0]
        return self._sparse.csr_matrix((values, xp.arange(n_rows, dtype=np.int32),
                                        xp.arange(n_rows + 1, dtype=np.int32)), shape=(n_rows, n_rows))

    def _stack_rows(self, blocks, n_columns):
        """One CSR of consecutive CSR row blocks (n_columns each)."""
        if len(blocks) == 1:
            return blocks[0]
        xp = self._xp
        offsets = np.cumsum([0] + [block.nnz for block in blocks])
        indptr = xp.concatenate([block.indptr[:-1] + int(offset) for block, offset in zip(blocks, offsets)]
                                + [xp.asarray([offsets[-1]], dtype=blocks[0].indptr.dtype)])
        return self._sparse.csr_matrix(
            (xp.concatenate([block.data for block in blocks]), xp.concatenate([block.indices for block in blocks]),
             indptr), shape=(sum(block.shape[0] for block in blocks), n_columns))

    def _coarsest_solver(self, operator):
        xp = self._xp
        n_unknowns = operator.shape[0]
        if n_unknowns <= COARSEST_DENSE_LIMIT:
            from ..core.backend import to_host
            dense = np.asarray(to_host(operator.toarray()), dtype=np.float64)
            dense = 0.5 * (dense + dense.T)
            inverse = np.zeros_like(dense)
            if n_unknowns:
                eigenvalues, vectors = np.linalg.eigh(dense)
                keep = eigenvalues > PSEUDO_INVERSE_CUTOFF * max(float(eigenvalues.max()), 0.0)
                inverse = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
            with np.errstate(over='ignore'):
                low = inverse.astype(operator.dtype)
            if not np.isfinite(low).all():   # a collapsed hierarchy: every cycle would return inf / NaN
                raise RuntimeError(f'multigrid: the coarsest-level pseudo-inverse ({n_unknowns} unknowns) overflows '
                                   f'{np.dtype(operator.dtype).name} (largest coarsest eigenvalue '
                                   f'{float(eigenvalues.max()):.3g}): the coarse operators collapsed')
            return ('dense', xp.asarray(low))
        return ('smooth', operator, self._smoother_weights(operator), xp.zeros(n_unknowns, dtype=operator.dtype))

    def precondition(self, residual, out) -> None:
        """out = one V(1,1)-cycle applied to residual (from a zero guess); a fixed SPD linear map for CG. The cycle runs
        on r / 2^e with max |r| in [0.5, 1) (exact, M is linear): tiny float64 residuals underflow the float32 levels."""
        if self._coarsest is None:
            raise RuntimeError('multigrid: setup before precondition')
        if self._device_cycle is not None:   # gpu: the fused kernels / CUDA graph of the same cycle
            self._device_cycle.apply(residual, out)
            return
        xp = self._xp
        scaled = self._scaled_residual
        xp.abs(residual, out=scaled)
        exponent = xp.frexp(scaled.max())[1]               # device scalar on the gpu: no host sync; 0 for r = 0
        xp.ldexp(residual, -exponent, out=scaled)
        self._cycle(0, scaled, out)
        xp.ldexp(out, exponent, out=out)

    def _cycle(self, depth, rhs, solution):
        xp = self._xp
        if depth == len(self._levels):
            self._solve_coarsest(rhs, solution)
            return
        level = self._levels[depth]
        product = self._product
        xp.multiply(level.weights, rhs, out=solution)                       # pre-smoothing from 0
        xp.subtract(rhs, product(level.operator, solution, level.residual), out=level.residual)
        coarse_rhs = product(level.restriction, level.low_residual(xp), level.coarse_rhs)
        self._cycle(depth + 1, coarse_rhs, level.coarse_solution)
        # float32 into a float64 level too
        solution += product(level.prolongation, level.coarse_solution, level.prolongation_target())
        xp.subtract(rhs, product(level.operator, solution, level.residual), out=level.residual)   # post-smoothing
        xp.multiply(level.weights, level.residual, out=level.residual)
        solution += level.residual

    def _product(self, matrix, vector, out):
        """matrix @ vector of the cpu cycle: written into `out` by scipy's CSR kernel (no allocation; equal dtypes),
        else the allocating product (the gpu cycle is _DeviceCycle)."""
        if (self._host_matvec is None or out is None or matrix.format != 'csr'
                or not matrix.dtype == vector.dtype == out.dtype):
            return matrix.dot(vector)
        out.fill(0)   # the kernel accumulates y += A x
        self._host_matvec(matrix.shape[0], matrix.shape[1], matrix.indptr, matrix.indices, matrix.data, vector, out)
        return out

    def _solve_coarsest(self, rhs, solution):
        xp = self._xp
        if self._coarsest[0] == 'dense':
            xp.dot(self._coarsest[1], rhs, out=solution)
            return
        _, operator, weights, residual = self._coarsest
        xp.multiply(weights, rhs, out=solution)
        for _ in range(COARSEST_SMOOTHING_SWEEPS - 1):
            xp.subtract(rhs, self._product(operator, solution, residual), out=residual)
            xp.multiply(weights, residual, out=residual)
            solution += residual

    def work_counters(self) -> dict:
        """Setups, host aggregations and the current level count (merged into the pcg counters)."""
        return {'multigrid_setups': self.n_setups, 'multigrid_aggregations': self.n_aggregations,
                'multigrid_levels': self.n_levels, 'multigrid_device_bytes': self.device_bytes,
                'multigrid_entries_per_unknown': round(self.entries_per_unknown, 2),
                'multigrid_operator_complexity': round(self.operator_complexity, 3),
                'multigrid_cycle_graphs': self.n_cycle_graphs + (self._device_cycle.n_graphs
                                                                 if self._device_cycle is not None else 0),
                **{f'multigrid_{phase}_seconds': round(seconds, 3) for phase, seconds in self.setup_seconds.items()}}
