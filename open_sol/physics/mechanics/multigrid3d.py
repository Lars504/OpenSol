"""Geometric multigrid V-cycle preconditioner of the matrix-free 3D brick elasticity (ElasticSystem3D, speed plan A2):
2:1 coarsened tensor levels on kernels3d.BrickKernels, separable trilinear transfers, Chebyshev block-Jacobi smoothing
and a dense coarsest pseudo-inverse; every launch a core.kernels twin, no host sync and no allocation inside a cycle."""
# Design: .temp_scratchpad/solidmech3d_multigrid_design.md. Coarse levels are rediscretised (not Galerkin): pair sums of
# the widths per axis (an odd count keeps its last cell single), volume-mean Lame values (void children count 0, so a
# half-void coarse cell is half as stiff; void only when every child is), held dofs by injection at coincident nodes
# (holds at other fine nodes are lost there; the prolonged correction is masked by the fine free dofs, and a coarse
# rigid motion that results falls into the pseudo-inverse's dropped null space). Pre- and post-smoothing are the same
# Chebyshev polynomial in B^-1 A and R = P^T, so the cycle is symmetric (plain PCG).
import itertools
import time

import numpy as np

from ...core.backend import to_host
from ...core.kernels import real_raw_twin
from .kernels3d import BrickKernels

COARSEST_DOFS = 1500        # no further coarsening at or below this many dofs (dense eigh of the coarsest operator)
MAX_LEVELS = 16
CHEBYSHEV_DEGREE = 2        # smoothing steps before and after the coarse correction
CHEBYSHEV_LOWER = 0.1       # Chebyshev interval [lower, upper] x the estimated largest eigenvalue of B^-1 A
CHEBYSHEV_UPPER = 1.15      # (power iteration from below: the margin covers its shortfall)
POWER_ITERATIONS = 15
PSEUDO_INVERSE_CUTOFF = 1e-12   # coarsest eigenvalues below this x the largest are dropped (rigid / lost-hold modes)
# a multigrid solve is capped at the block-Jacobi iteration estimate in V-cycles (ElasticSystem3D.multigrid_budget,
# within [FALLBACK_MIN_ITERATIONS, FALLBACK_ITERATIONS]); one that does not converge within it switches the system to
# block Jacobi (_guarded_solve; typical multigrid solves take 12-40)
FALLBACK_ITERATIONS = 200
FALLBACK_MIN_ITERATIONS = 60
# fine-level products of one V-cycle (pre / post Chebyshev and the residual) = its cost in block-Jacobi iterations
# (measured 4.4: 3.1 ms per cycle against 0.7 ms per fused Jacobi iteration on 2e5 nodes)
CYCLE_COST_IN_JACOBI_ITERATIONS = 2 * CHEBYSHEV_DEGREE + 1
_POWER_SEED = 20251111

_BLOCK = 256
_RESIDUAL, _SMOOTH, _ASSIGN, _ADD = 1, 2, 4, 8   # mg3_chebyshev mode bits

_SOURCES = {
    # fine (n0, n1, n2, 3) = trilinear prolongation along `axis` of coarse (n_coarse along it): fine node f between
    # coarse left[f] and left[f] + 1 at weight[f]; mode 1 zeroes the fixed dofs of the output
    'mg3_axis_prolong': r"""
extern "C" __global__
void mg3_axis_prolong(const real_t* __restrict__ coarse, real_t* __restrict__ fine, const int* __restrict__ left,
                      const real_t* __restrict__ weight, const bool* __restrict__ free_dofs,
                      int n0, int n1, int n2, int axis, int n_coarse, int mode)
{
    const idx_t index = (idx_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (idx_t)n0 * n1 * n2 * 3) return;
    const int component = (int)(index % 3);
    idx_t rest = index / 3;
    int c[3];
    c[2] = (int)(rest % n2); rest /= n2;
    c[1] = (int)(rest % n1);
    c[0] = (int)(rest / n1);
    int dims[3] = {n0, n1, n2};
    dims[axis] = n_coarse;
    const int f = c[axis];
    const int l = left[f];
    const real_t t = weight[f];
    c[axis] = l;
    real_t value = ((real_t)1 - t) * coarse[(((idx_t)c[0] * dims[1] + c[1]) * dims[2] + c[2]) * 3 + component];
    if (t != (real_t)0) {
        c[axis] = l + 1;
        value += t * coarse[(((idx_t)c[0] * dims[1] + c[1]) * dims[2] + c[2]) * 3 + component];
    }
    fine[index] = (mode == 1 && !free_dofs[index]) ? (real_t)0 : value;
}
""",
    # coarse (n0, n1, n2, 3) = transpose of mg3_axis_prolong along `axis` of fine (n_fine along it), gathered over the
    # fine nodes fine_node[c] - 1 .. + 1; masked zeroes the fixed dofs of the output
    'mg3_axis_restrict': r"""
extern "C" __global__
void mg3_axis_restrict(const real_t* __restrict__ fine, real_t* __restrict__ coarse, const int* __restrict__ left,
                       const real_t* __restrict__ weight, const int* __restrict__ fine_node,
                       const bool* __restrict__ free_dofs, int n0, int n1, int n2, int axis, int n_fine, int masked)
{
    const idx_t index = (idx_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (idx_t)n0 * n1 * n2 * 3) return;
    const int component = (int)(index % 3);
    idx_t rest = index / 3;
    int c[3];
    c[2] = (int)(rest % n2); rest /= n2;
    c[1] = (int)(rest % n1);
    c[0] = (int)(rest / n1);
    int dims[3] = {n0, n1, n2};
    dims[axis] = n_fine;
    const int own = c[axis];
    const int centre = fine_node[own];
    real_t sum = (real_t)0;
    for (int f = centre - 1; f <= centre + 1; ++f) {
        if (f < 0 || f >= n_fine) continue;
        const int l = left[f];
        const real_t t = weight[f];
        const real_t w = l == own ? (real_t)1 - t : (l == own - 1 ? t : (real_t)0);
        if (w == (real_t)0) continue;
        c[axis] = f;
        sum += w * fine[(((idx_t)c[0] * dims[1] + c[1]) * dims[2] + c[2]) * 3 + component];
    }
    coarse[index] = (masked && !free_dofs[index]) ? (real_t)0 : sum;
}
""",
    # one Chebyshev step per node: r -= q (RESIDUAL), d = c1 d + c2 B^-1 r (SMOOTH), x = d (ASSIGN) or x += d (SMOOTH /
    # ADD)
    'mg3_chebyshev': r"""
extern "C" __global__
void mg3_chebyshev(real_t* __restrict__ residual, const real_t* __restrict__ product,
                   const real_t* __restrict__ inverse, real_t* __restrict__ direction, real_t* __restrict__ solution,
                   real_t c1, real_t c2, int mode, int n_nodes)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;
    const idx_t dof = 3 * (idx_t)node;
    real_t r[3], d[3];
    for (int a = 0; a < 3; ++a) {
        r[a] = residual[dof + a];
        if (mode & 1) {
            r[a] -= product[dof + a];
            residual[dof + a] = r[a];
        }
    }
    if (mode & 2) {
        const real_t* b = inverse + 6 * (idx_t)node;
        const real_t z[3] = {b[0] * r[0] + b[3] * r[1] + b[4] * r[2],
                             b[3] * r[0] + b[1] * r[1] + b[5] * r[2],
                             b[4] * r[0] + b[5] * r[1] + b[2] * r[2]};
        for (int a = 0; a < 3; ++a) {
            d[a] = c2 * z[a] + (c1 != (real_t)0 ? c1 * direction[dof + a] : (real_t)0);
            direction[dof + a] = d[a];
        }
    } else if (mode & 8) {
        for (int a = 0; a < 3; ++a) d[a] = direction[dof + a];
    }
    if (mode & 10)
        for (int a = 0; a < 3; ++a) solution[dof + a] = (mode & 4) ? d[a] : solution[dof + a] + d[a];
}
""",
    # coarsest y = M x of a row-major n x n matrix, one warp per row
    'mg3_dense_product': r"""
extern "C" __global__
void mg3_dense_product(const real_t* __restrict__ matrix, const real_t* __restrict__ x, real_t* __restrict__ y, int n)
{
    const long long thread = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const int row = (int)(thread / 32);
    const int lane = threadIdx.x & 31;
    real_t sum = (real_t)0;
    if (row < n)
        for (int column = lane; column < n; column += 32) sum += matrix[(long long)row * n + column] * x[column];
    for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (row < n && lane == 0) y[row] = sum;
}
""",
}

# ---- numpy twins (the kernels' args tuples, outputs written in place) ----
def _axis_shape(n0, n1, n2, axis, count):
    shape = [int(n0), int(n1), int(n2)]
    shape[int(axis)] = int(count)
    return shape


def _along_axis(values, axis):
    shape = [1, 1, 1, 1]
    shape[axis] = values.size
    return values.reshape(shape)


def _axis_prolong_host(coarse, fine, left, weight, free_dofs, n0, n1, n2, axis, n_coarse, mode):
    axis = int(axis)
    source = coarse.reshape(*_axis_shape(n0, n1, n2, axis, n_coarse), 3)
    upper_index = np.minimum(left + 1, int(n_coarse) - 1)
    t = _along_axis(weight, axis)
    value = ((1 - t) * np.take(source, left, axis=axis) + t * np.take(source, upper_index, axis=axis)).reshape(-1)
    fine[...] = np.where(free_dofs, value, 0) if int(mode) == 1 else value


def _axis_restrict_host(fine, coarse, left, weight, fine_node, free_dofs, n0, n1, n2, axis, n_fine, masked):
    axis, n_fine = int(axis), int(n_fine)
    source = fine.reshape(*_axis_shape(n0, n1, n2, axis, n_fine), 3)
    own = np.arange(fine_node.size)
    result = np.zeros((int(n0), int(n1), int(n2), 3), dtype=coarse.dtype)
    for offset in (-1, 0, 1):
        candidate = fine_node + offset
        clipped = np.clip(candidate, 0, n_fine - 1)
        l, t = left[clipped], weight[clipped]
        w = np.where(l == own, 1 - t, np.where(l == own - 1, t, 0)) * ((candidate >= 0) & (candidate < n_fine))
        result += _along_axis(w.astype(coarse.dtype), axis) * np.take(source, clipped, axis=axis)
    flat = result.reshape(-1)
    coarse[...] = np.where(free_dofs, flat, 0) if int(masked) else flat


def _chebyshev_host(residual, product, inverse, direction, solution, c1, c2, mode, n_nodes):
    mode = int(mode)
    if mode & _RESIDUAL:
        residual -= product
    if mode & _SMOOTH:
        b = inverse.reshape(-1, 6)
        r = residual.reshape(-1, 3)
        z = np.stack((b[:, 0] * r[:, 0] + b[:, 3] * r[:, 1] + b[:, 4] * r[:, 2],
                      b[:, 3] * r[:, 0] + b[:, 1] * r[:, 1] + b[:, 5] * r[:, 2],
                      b[:, 4] * r[:, 0] + b[:, 5] * r[:, 1] + b[:, 2] * r[:, 2]), axis=1).reshape(-1)
        if c1 != 0:
            direction *= c1
            direction += c2 * z
        else:
            np.multiply(z, c2, out=direction)
    if mode & (_SMOOTH | _ADD):
        if mode & _ASSIGN:
            solution[...] = direction
        else:
            solution += direction


def _dense_product_host(matrix, x, y, n):
    np.matmul(matrix.reshape(int(n), int(n)), x, out=y)


# cupy: the RawKernel compiled per real type of args[0] (core.kernels.RealRawKernel)
_TWINS = {name: real_raw_twin(f'mechanics3d_multigrid.{name}', 'typedef long long idx_t;\n' + _SOURCES[name], name, host)
          for name, host in (('mg3_axis_prolong', _axis_prolong_host), ('mg3_axis_restrict', _axis_restrict_host),
                             ('mg3_chebyshev', _chebyshev_host), ('mg3_dense_product', _dense_product_host))}


def _grid(count):
    return ((int(count) + _BLOCK - 1) // _BLOCK,)


# ---- hierarchy shapes (shared with the estimators) ----
def _node_count(element_shape) -> int:
    return int(np.prod([n + 1 for n in element_shape]))


def level_element_shapes(element_shape) -> list:
    """Element shapes of every level, the fine one first: each axis halved (odd counts round up) until the dofs are <=
    COARSEST_DOFS, nothing coarsens or MAX_LEVELS."""
    shapes = [tuple(int(n) for n in element_shape)]
    while len(shapes) < MAX_LEVELS and 3 * _node_count(shapes[-1]) > COARSEST_DOFS:
        coarse = tuple((n + 1) // 2 for n in shapes[-1])
        if coarse == shapes[-1]:
            break
        shapes.append(coarse)
    return shapes


def _transfer_sizes(fine_shape, coarse_shape) -> tuple:
    """Values of the two intermediate arrays of one separable transfer: (F0, C1, C2, 3) and (F0, F1, C2, 3) nodes."""
    f0, f1, _ = (n + 1 for n in fine_shape)
    _, c1, c2 = (n + 1 for n in coarse_shape)
    return 3 * f0 * c1 * c2, 3 * f0 * f1 * c2


def hierarchy_bytes(element_shape, item_bytes) -> int:
    """Device bytes BrickMultigrid holds beside the system: level vectors (fine: residual, direction, product; coarse
    + solution), coarse Lame values, inverse blocks and free masks, the transfer intermediates and the coarsest dense
    inverse (axis tables and widths negligible)."""
    shapes = level_element_shapes(element_shape)
    total = 0
    for index, shape in enumerate(shapes):
        nodes, elements = _node_count(shape), int(np.prod(shape))
        dofs = 3 * nodes
        total += (3 if index == 0 else 4) * dofs * item_bytes
        if index > 0:
            total += (2 * elements + 6 * nodes) * item_bytes + dofs
        if index + 1 < len(shapes):
            total += sum(_transfer_sizes(shape, shapes[index + 1])) * item_bytes
    coarsest = 3 * _node_count(shapes[-1])
    return total + coarsest * coarsest * item_bytes


def coarsest_host_bytes(element_shape) -> int:
    """Host bytes of the coarsest build (float64 probed matrix, its free block, eigenvectors and the inverse)."""
    coarsest = 3 * _node_count(level_element_shapes(element_shape)[-1])
    return 4 * coarsest * coarsest * 8


def coarse_axis(widths) -> tuple:
    """(coarse widths, fine node index of every coarse node) of one axis (host float64 widths): pairs of cells summed,
    an odd count keeps its last cell single."""
    count = widths.size
    pairs = count // 2
    coarse = widths[:2 * pairs].reshape(pairs, 2).sum(axis=1)
    fine_nodes = np.arange(0, 2 * pairs + 1, 2)
    if count % 2:
        coarse = np.append(coarse, widths[-1])
        fine_nodes = np.append(fine_nodes, count)
    return coarse, fine_nodes


def prolongation_table(fine_widths, fine_nodes) -> tuple:
    """(left coarse node (int32), weight of left + 1) of every fine node along one axis: linear interpolation in the
    node coordinates (graded widths: not 1/2)."""
    coordinates = np.concatenate(([0.0], np.cumsum(fine_widths)))
    coarse_coordinates = coordinates[fine_nodes]
    left = np.searchsorted(fine_nodes, np.arange(coordinates.size), side='right') - 1
    left = np.minimum(left, fine_nodes.size - 2)
    weight = (coordinates - coarse_coordinates[left]) / (coarse_coordinates[left + 1] - coarse_coordinates[left])
    return left.astype(np.int32), weight


class _Level:
    """Arrays of one level: operator (Lame values, widths, free dofs, reduced inverse blocks; level 0 the system's),
    work vectors, Chebyshev coefficients and the transfer to the next coarser level."""

    def __init__(self, element_shape, dtype, xp, coarse):
        self.element_shape = tuple(element_shape)
        self.node_shape = tuple(n + 1 for n in element_shape)
        self.n_nodes = _node_count(element_shape)
        self.n_dofs = 3 * self.n_nodes
        self.n_nodes_i32 = np.int32(self.n_nodes)
        self.kernels = (BrickKernels(dtype, element_shape, (0, 0, 0), element_shape) if coarse else None)
        self.residual = xp.zeros(self.n_dofs, dtype=dtype)
        self.direction = xp.zeros(self.n_dofs, dtype=dtype)
        self.product = xp.zeros(self.n_dofs, dtype=dtype)
        self.solution = xp.zeros(self.n_dofs, dtype=dtype) if coarse else None   # level 0: the caller's out
        self.free = xp.zeros(self.n_dofs, dtype=xp.bool_) if coarse else None
        self.inverse_blocks = xp.zeros((self.n_nodes, 6), dtype=dtype) if coarse else None
        self.lame_lambda = self.mu = self.widths = None
        self.widths_host = None
        self.coefficients = ()   # Chebyshev (c1, c2) per step, numpy scalars of the level dtype
        self.transfer = None     # _Transfer to the next coarser level

    def product_of(self, vector, out):
        self.kernels.product(vector, out, self.lame_lambda, self.mu, self.widths, self.free, masked=True)


class _Transfer:
    """Per-axis tables (device) and the two intermediate arrays of the transfers between a level and the next."""

    def __init__(self, fine_shape, coarse_shape, dtype, xp):
        self.fine_nodes = tuple(n + 1 for n in fine_shape)
        self.coarse_nodes = tuple(n + 1 for n in coarse_shape)
        first, second = _transfer_sizes(fine_shape, coarse_shape)
        self.first = xp.zeros(first, dtype=dtype)     # (F0, C1, C2, 3)
        self.second = xp.zeros(second, dtype=dtype)   # (F0, F1, C2, 3)
        self.tables = None   # per axis (left int32, weight, fine node of each coarse node int32)


class BrickMultigrid:
    """precondition(residual, out) = one V(CHEBYSHEV_DEGREE, CHEBYSHEV_DEGREE)-cycle on the level hierarchy of an
    ElasticSystem3D window (level arrays allocated here once; build() after every assembly)."""

    def __init__(self, element_shape, dtype, xp):
        self.dtype = np.dtype(dtype)
        self.xp = xp
        self.shapes = level_element_shapes(element_shape)
        self.levels = [_Level(shape, self.dtype, xp, coarse=index > 0) for index, shape in enumerate(self.shapes)]
        for fine, coarse in zip(self.levels[:-1], self.levels[1:]):
            fine.transfer = _Transfer(fine.element_shape, coarse.element_shape, self.dtype, xp)
        coarsest = self.levels[-1]
        self._dense = None
        self._dense_n = np.int32(coarsest.n_dofs)
        self._dense_grid = ((32 * coarsest.n_dofs + _BLOCK - 1) // _BLOCK,)
        self.n_builds = 0
        self.fallbacks = 0          # ElasticSystem3D: solves after which the system switched to block Jacobi
        self.coarsest_kept = 0      # coarsest eigenvalues kept by the pseudo-inverse of the last build
        self.build_seconds = 0.0

    # --- build ------------------------------------------------------------------
    def build(self, lame_lambda, mu, free, inverse_blocks, widths, widths_host, kernels) -> None:
        """Hierarchy of the latest assembly: level 0 = the system's arrays in the cycle dtype (lame_lambda, mu,
        reduced inverse blocks, device widths, free dofs, its BrickKernels; widths_host float64 per axis), then per
        coarser level Lame means, injected holds, blocks, Chebyshev bounds and the coarsest dense inverse."""
        start = time.perf_counter()
        xp = self.xp
        top = self.levels[0]
        top.lame_lambda, top.mu, top.inverse_blocks, top.widths, top.free = lame_lambda, mu, inverse_blocks, widths, free
        top.kernels = kernels
        top.widths_host = tuple(np.asarray(values, dtype=np.float64) for values in widths_host)
        volume = self._volume(top.widths_host)
        solid = mu > 0
        lambda_volume = xp.where(solid, lame_lambda, 0).astype(np.float64) * volume
        mu_volume = xp.where(solid, mu, 0).astype(np.float64) * volume
        for fine, coarse in zip(self.levels[:-1], self.levels[1:]):
            tables = []
            coarse_widths = []
            for axis in range(3):
                widths_axis, fine_nodes = coarse_axis(fine.widths_host[axis])
                left, weight = prolongation_table(fine.widths_host[axis], fine_nodes)
                tables.append((xp.asarray(left), xp.asarray(weight, dtype=self.dtype),
                               xp.asarray(fine_nodes.astype(np.int32))))
                coarse_widths.append(widths_axis)
            fine.transfer.tables = tuple(tables)
            coarse.widths_host = tuple(coarse_widths)
            coarse.widths = tuple(xp.asarray(values, dtype=self.dtype) for values in coarse_widths)
            for axis in range(3):
                lambda_volume = _pair_sums(lambda_volume, axis, xp)
                mu_volume = _pair_sums(mu_volume, axis, xp)
                volume = _pair_sums(volume, axis, xp)
            coarse.lame_lambda = xp.ascontiguousarray((lambda_volume / volume).astype(self.dtype))
            coarse.mu = xp.ascontiguousarray((mu_volume / volume).astype(self.dtype))
            injected = fine.free.reshape(*fine.node_shape, 3)
            for axis, (_, _, fine_nodes) in enumerate(tables):
                injected = xp.take(injected, fine_nodes, axis=axis)
            coarse.kernels.diagonal_blocks(coarse.inverse_blocks, coarse.lame_lambda, coarse.mu, coarse.widths)
            unconnected = (coarse.inverse_blocks[:, :3] <= 0).reshape(-1)   # dofs touching no solid coarse element
            xp.logical_and(injected.reshape(-1), ~unconnected, out=coarse.free)
            coarse.kernels.reduce_invert_blocks(coarse.inverse_blocks, coarse.free)
        del lambda_volume, mu_volume, volume
        for level in self.levels[:-1]:
            level.coefficients = self._chebyshev_coefficients(self._largest_eigenvalue(level))
        self._dense = None
        self._dense = self._dense_inverse(self.levels[-1])
        self.n_builds += 1
        self.build_seconds += time.perf_counter() - start

    def _volume(self, widths_host):
        wx, wy, wz = widths_host
        return self.xp.asarray(wx[:, None, None] * wy[None, :, None] * wz[None, None, :])

    def _largest_eigenvalue(self, level) -> float:
        """Power-iteration estimate (from below) of the largest eigenvalue of B^-1 A on the free dofs: the A-inner
        product Rayleigh quotient (z . w) / (v . w), w = A v, z = B^-1 w; the scalars stay on the backend, one host read
        per level (a zero vector stays zero, a NaN one keeps the last estimate)."""
        xp = self.xp
        vector, image, preconditioned = level.direction, level.product, level.residual
        start = np.random.default_rng(_POWER_SEED).standard_normal(level.n_dofs)
        vector[...] = xp.asarray(start, dtype=self.dtype)
        xp.multiply(vector, level.free, out=vector)
        tiny, tiny64 = self.dtype.type(np.finfo(self.dtype).tiny), np.finfo(np.float64).tiny
        estimate = xp.zeros((), dtype=np.float64)
        for _ in range(POWER_ITERATIONS):
            scale = 1.0 / xp.sqrt(xp.maximum(xp.dot(vector, vector), tiny))
            xp.multiply(vector, xp.asarray(scale, dtype=self.dtype), out=vector)
            level.product_of(vector, image)
            level.kernels.precondition(level.inverse_blocks, image, preconditioned)
            curvature = xp.asarray(xp.dot(vector, image), dtype=np.float64)
            quotient = xp.asarray(xp.dot(preconditioned, image), dtype=np.float64) / xp.maximum(curvature, tiny64)
            estimate = xp.where(curvature > 0.0, quotient, estimate)
            xp.copyto(vector, preconditioned)
        estimate = float(estimate)
        return estimate if estimate > 0.0 else 1.0

    def _chebyshev_coefficients(self, largest) -> tuple:
        """(c1, c2) per smoothing step of the Chebyshev iteration on [CHEBYSHEV_LOWER, CHEBYSHEV_UPPER] x largest."""
        lower, upper = CHEBYSHEV_LOWER * largest, CHEBYSHEV_UPPER * largest
        theta, delta = 0.5 * (upper + lower), 0.5 * (upper - lower)
        sigma = theta / delta
        rho = 1.0 / sigma
        real = self.dtype.type
        coefficients = [(real(0.0), real(1.0 / theta))]
        for _ in range(1, CHEBYSHEV_DEGREE):
            rho_next = 1.0 / (2.0 * sigma - rho)
            coefficients.append((real(rho_next * rho), real(2.0 * rho_next / delta)))
            rho = rho_next
        return tuple(coefficients)

    def _dense_inverse(self, level):
        """Pseudo-inverse (cycle dtype, on the backend) of the coarsest masked operator, probed with 27 node colours x 3
        components (colour nodes 3 apart: their 27-node stencils never overlap), eigh of the free block on the host."""
        xp = self.xp
        n0, n1, n2 = level.node_shape
        n = level.n_dofs
        free = np.asarray(to_host(level.free), dtype=bool)
        node_i, node_j, node_k = (index.reshape(-1) for index in np.indices(level.node_shape))
        colours = (node_i % 3) * 9 + (node_j % 3) * 3 + (node_k % 3)
        dense = np.zeros((n, n))
        offsets = np.array(tuple(itertools.product((-1, 0, 1), repeat=3)), dtype=np.int64)   # (27, 3) node offsets
        components = np.arange(3)
        for colour in range(27):
            nodes = np.flatnonzero(colours == colour)
            for component in range(3):
                columns = 3 * nodes + component
                columns = columns[free[columns]]
                if not columns.size:
                    continue
                probe = np.zeros(n)
                probe[columns] = 1.0
                level.direction[...] = xp.asarray(probe, dtype=self.dtype)
                level.product_of(level.direction, level.product)
                image = np.asarray(to_host(level.product), dtype=np.float64)
                owner = columns // 3   # every stencil node (27 offsets) x component of every probed column at once
                mi = node_i[owner][:, None] + offsets[:, 0]
                mj = node_j[owner][:, None] + offsets[:, 1]
                mk = node_k[owner][:, None] + offsets[:, 2]
                inside = (mi >= 0) & (mi < n0) & (mj >= 0) & (mj < n1) & (mk >= 0) & (mk < n2)   # (c, 27)
                neighbour = ((mi * n1 + mj) * n2 + mk)[inside]
                rows = (3 * neighbour[:, None] + components).reshape(-1)
                probed = np.repeat(np.broadcast_to(columns[:, None], inside.shape)[inside], 3)
                dense[rows, probed] = image[rows]
        kept = np.flatnonzero(free)
        inverse = np.zeros((n, n))
        self.coarsest_kept = 0
        if kept.size:
            block = dense[np.ix_(kept, kept)]
            block = 0.5 * (block + block.T)
            eigenvalues, vectors = np.linalg.eigh(block)
            keep = eigenvalues > PSEUDO_INVERSE_CUTOFF * max(float(eigenvalues.max()), 0.0)
            self.coarsest_kept = int(np.count_nonzero(keep))
            inverse[np.ix_(kept, kept)] = (vectors[:, keep] / eigenvalues[keep]) @ vectors[:, keep].T
        with np.errstate(over='ignore'):
            low = inverse.astype(self.dtype)
        if not np.isfinite(low).all():
            raise RuntimeError(f'multigrid (solid_mechanics_3d): the coarsest pseudo-inverse ({n} dofs) overflows '
                               f'{self.dtype.name}: the coarse operator collapsed')
        return xp.asarray(low)

    # --- cycle ------------------------------------------------------------------
    def precondition(self, residual, out) -> None:
        """out = V-cycle(residual) (pcg precondition callback; residual is not modified)."""
        top = self.levels[0]
        self.xp.copyto(top.residual, residual)
        self._cycle(0, out)

    def _cycle(self, index, solution):
        level = self.levels[index]
        if index == len(self.levels) - 1:
            _TWINS['mg3_dense_product'](self._dense_grid, (_BLOCK,),
                                        (self._dense, level.residual, solution, self._dense_n))
            return
        coarse = self.levels[index + 1]
        self._smooth(level, solution, first=True)
        self._restrict(level, coarse)
        self._cycle(index + 1, coarse.solution)
        self._prolong(level, coarse)
        level.product_of(level.direction, level.product)
        self._chebyshev(level, solution, _RESIDUAL | _ADD, (self.dtype.type(0), self.dtype.type(0)))
        self._smooth(level, solution, first=False)

    def _chebyshev(self, level, solution, mode, coefficients):
        c1, c2 = coefficients
        _TWINS['mg3_chebyshev'](_grid(level.n_nodes), (_BLOCK,),
                                (level.residual, level.product, level.inverse_blocks, level.direction, solution, c1,
                                 c2, np.int32(mode), level.n_nodes_i32))

    def _smooth(self, level, solution, first):
        """Chebyshev steps on level.residual into solution (first: solution assigned, the residual updated after the
        last step for the restriction; else added to)."""
        steps = level.coefficients
        self._chebyshev(level, solution, _SMOOTH | (_ASSIGN if first else 0), steps[0])
        for coefficients in steps[1:]:
            level.product_of(level.direction, level.product)
            self._chebyshev(level, solution, _RESIDUAL | _SMOOTH, coefficients)
        if first:
            level.product_of(level.direction, level.product)
            self._chebyshev(level, solution, _RESIDUAL, steps[0])

    def _restrict(self, fine, coarse):
        """coarse.residual = P^T fine.residual (axis 2, 1, 0), fixed coarse dofs zeroed."""
        transfer = fine.transfer
        f0, f1, f2 = transfer.fine_nodes
        c0, c1, c2 = transfer.coarse_nodes
        passes = ((2, fine.residual, transfer.second, (f0, f1, c2), f2, 0),
                  (1, transfer.second, transfer.first, (f0, c1, c2), f1, 0),
                  (0, transfer.first, coarse.residual, (c0, c1, c2), f0, 1))
        for axis, source, target, dims, n_fine, masked in passes:
            left, weight, fine_nodes = transfer.tables[axis]
            _TWINS['mg3_axis_restrict'](_grid(3 * int(np.prod(dims))), (_BLOCK,),
                                        (source, target, left, weight, fine_nodes, coarse.free,
                                         *(np.int32(n) for n in dims), np.int32(axis), np.int32(n_fine),
                                         np.int32(masked)))

    def _prolong(self, fine, coarse):
        """fine.direction = P coarse.solution (axis 0, 1, 2), fixed fine dofs zeroed."""
        transfer = fine.transfer
        f0, f1, f2 = transfer.fine_nodes
        c0, c1, c2 = transfer.coarse_nodes
        passes = ((0, coarse.solution, transfer.first, (f0, c1, c2), c0, 0),
                  (1, transfer.first, transfer.second, (f0, f1, c2), c1, 0),
                  (2, transfer.second, fine.direction, (f0, f1, f2), c2, 1))
        for axis, source, target, dims, n_coarse, mode in passes:
            left, weight, _ = transfer.tables[axis]
            _TWINS['mg3_axis_prolong'](_grid(3 * int(np.prod(dims))), (_BLOCK,),
                                       (source, target, left, weight, fine.free, *(np.int32(n) for n in dims),
                                        np.int32(axis), np.int32(n_coarse), np.int32(mode)))

    def work_counters(self) -> dict:
        """Hierarchy record for the pcg role runtime's counters (levels, coarsest dofs, builds, fallbacks)."""
        return {'multigrid_levels': len(self.levels), 'multigrid_coarsest_dofs': int(self.levels[-1].n_dofs),
                'multigrid_builds': self.n_builds, 'multigrid_fallbacks': self.fallbacks,
                'multigrid_build_seconds': self.build_seconds}


def _pair_sums(values, axis, xp):
    """Sums of cell pairs along axis (an odd count keeps its last cell), as coarse_axis pairs the widths."""
    count = values.shape[axis]
    pairs = count // 2

    def part(selection):
        index = [slice(None)] * values.ndim
        index[axis] = selection
        return values[tuple(index)]
    summed = part(slice(0, 2 * pairs, 2)) + part(slice(1, 2 * pairs, 2))
    if count % 2:
        summed = xp.concatenate((summed, part(slice(count - 1, count))), axis=axis)
    return summed
