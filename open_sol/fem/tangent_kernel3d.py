"""Per-element total-Lagrangian force and tangent of P1 / P2 tetrahedra (next-gains R1): one core.kernels raw twin
'fem.element_tangent_3d' = a CUDA block per element (rule-point operators in shared memory), a numba prange over the
elements, and the batched numpy / cupy formulas of fem.elasticity3d as the reference twin. numpy at module level."""
# Arguments (every array C-contiguous float64, numpy or cupy): gradients (m, q, n, 3) physical basis gradients at the
# rule points, weights (m, q), local (m, n, 3) element displacements, material (m, 6, 6) C, fixed (m, q, 6) S0 or a
# (1, 1, 1) dummy, outputs force (m, 3 n) and blocks (m, 3 n, 3 n) (a (1, 1, 1) dummy when want_tangent is 0), then
# int32 n_elements, has_fixed, want_tangent. Per rule point: H = grad u, E = Green-Lagrange (engineering shear), S = C E
# + S0, B + B_NL rows = G_a,j F_ci (+ G_a,i F_cj off the diagonal), F = I + H; force = sum w (B + B_NL)^T S, tangent =
# sum w (B + B_NL)^T C (B + B_NL) + (sum w grad N_a . S grad N_b) x I3 (lower triangle computed, mirrored). Voigt order
# = elasticity3d.VOIGT_PAIRS (xx, yy, zz, yz, xz, xy). The CUDA kernel keeps q x 6 x 3 n operators twice in static
# shared memory: P2 on the 14-point rule 46.8 kB, under KERNEL_SHARED_LIMIT; a larger rule (a degree argument of 6+)
# runs the batched cupy formulas instead (kernel_fits).
import numpy as np

from ..core.kernels import raw_twin

KERNEL_THREADS = 256                 # threads per element block
KERNEL_SHARED_LIMIT = 48 * 1024      # static shared memory per block
_VOIGT_FIRST = np.array([0, 1, 2, 1, 0, 0], dtype=np.int64)    # elasticity3d.VOIGT_PAIRS
_VOIGT_SECOND = np.array([0, 1, 2, 2, 2, 1], dtype=np.int64)


def shared_bytes(n_local, points) -> int:
    """Static shared bytes of the CUDA kernel for n_local nodes and `points` rule points."""
    columns = 3 * int(n_local)
    points = int(points)
    return 8 * (points * (3 * n_local + 1 + 9 + 6 + 2 * 6 * columns) + columns + 36 + n_local * n_local)


def kernel_fits(n_local, points) -> bool:
    """True when the CUDA kernel's shared arrays fit one block (else the batched cupy formulas run)."""
    return shared_bytes(n_local, points) <= KERNEL_SHARED_LIMIT


def element_flops(n_local, points, want_tangent=True) -> float:
    """A priori arithmetic per element of one call: H, E, S, B + B_NL and the force; with the tangent also w C (B +
    B_NL), the scalar geometric blocks and the lower triangle of the element block (the estimators' count)."""
    n_local, points = int(n_local), int(points)
    columns = 3 * n_local
    force = points * (18 * n_local + 108 + 18 * columns) + 14 * points * columns
    if not want_tangent:
        return float(force)
    tangent = (13 * points * 6 * columns + 26 * points * n_local ** 2
               + 12 * points * columns * (columns + 1) // 2)
    return float(force + tangent)


def element_bytes(n_local, want_tangent=True) -> int:
    """Device bytes per element of one call (force, blocks, gathered displacements): the kernel route's chunk
    temporaries."""
    columns = 3 * int(n_local)
    return 8 * (columns + (columns ** 2 if want_tangent else 0) + columns)


_SOURCE = r"""
#define NL __NL__
#define NQ __NQ__
#define ND (3 * NL)
__constant__ int VOIGT_FIRST[6] = {0, 1, 2, 1, 0, 0};
__constant__ int VOIGT_SECOND[6] = {0, 1, 2, 2, 2, 1};

extern "C" __global__ void element_tangent_3d(
    const double* __restrict__ gradients, const double* __restrict__ weights, const double* __restrict__ local,
    const double* __restrict__ material, const double* __restrict__ fixed, double* __restrict__ force,
    double* __restrict__ blocks, const int n_elements, const int has_fixed, const int want_tangent)
{
    __shared__ double G[NQ * NL * 3];
    __shared__ double U[NL * 3];
    __shared__ double C[36];
    __shared__ double W[NQ];
    __shared__ double H[NQ * 9];
    __shared__ double S[NQ * 6];
    __shared__ double GEO[NL * NL];
    __shared__ double OP[NQ * 6 * ND];
    __shared__ double CB[NQ * 6 * ND];
    const long long e = blockIdx.x;
    if (e >= n_elements) return;   // uniform per block
    const int t = threadIdx.x, nt = blockDim.x;
    for (int i = t; i < NQ * NL * 3; i += nt) G[i] = gradients[e * (NQ * NL * 3) + i];
    for (int i = t; i < NL * 3; i += nt) U[i] = local[e * (NL * 3) + i];
    for (int i = t; i < 36; i += nt) C[i] = material[e * 36 + i];
    for (int i = t; i < NQ; i += nt) W[i] = weights[e * NQ + i];
    __syncthreads();
    for (int i = t; i < NQ * 9; i += nt) {   // H[q][c][d] = d u_c / d x_d
        const int q = i / 9, c = (i / 3) % 3, d = i % 3;
        double h = 0.0;
        for (int a = 0; a < NL; ++a) h += G[(q * NL + a) * 3 + d] * U[a * 3 + c];
        H[i] = h;
    }
    __syncthreads();
    for (int q = t; q < NQ; q += nt) {   // E (H + H^T + H^T H) / 2, engineering shear; S = C E + S0
        const double* h = H + q * 9;
        double strain[6];
        for (int k = 0; k < 6; ++k) {
            const int fi = VOIGT_FIRST[k], fj = VOIGT_SECOND[k];
            const double quadratic = h[fi] * h[fj] + h[3 + fi] * h[3 + fj] + h[6 + fi] * h[6 + fj];
            strain[k] = (fi == fj) ? h[4 * fi] + 0.5 * quadratic : h[3 * fi + fj] + h[3 * fj + fi] + quadratic;
        }
        for (int k = 0; k < 6; ++k) {
            double s = 0.0;
            for (int l = 0; l < 6; ++l) s += C[k * 6 + l] * strain[l];
            S[q * 6 + k] = has_fixed ? s + fixed[(e * NQ + q) * 6 + k] : s;
        }
    }
    __syncthreads();
    for (int i = t; i < NQ * 6 * ND; i += nt) {   // (B + B_NL)[q][k][3 a + c]
        const int q = i / (6 * ND), k = (i / ND) % 6, r = i % ND, a = r / 3, c = r % 3;
        const int fi = VOIGT_FIRST[k], fj = VOIGT_SECOND[k];
        const double* g = G + (q * NL + a) * 3;
        const double* h = H + q * 9;
        const double fci = h[3 * c + fi] + (c == fi ? 1.0 : 0.0);
        OP[i] = (fi == fj) ? g[fi] * fci : g[fj] * fci + g[fi] * (h[3 * c + fj] + (c == fj ? 1.0 : 0.0));
    }
    __syncthreads();
    for (int r = t; r < ND; r += nt) {
        double total = 0.0;
        for (int q = 0; q < NQ; ++q) {
            double part = 0.0;
            for (int k = 0; k < 6; ++k) part += OP[(q * 6 + k) * ND + r] * S[q * 6 + k];
            total += W[q] * part;
        }
        force[e * ND + r] = total;
    }
    if (!want_tangent) return;   // a kernel argument: uniform per block
    for (int i = t; i < NQ * 6 * ND; i += nt) {   // w C (B + B_NL)
        const int q = i / (6 * ND), k = (i / ND) % 6, s = i % ND;
        double v = 0.0;
        for (int l = 0; l < 6; ++l) v += C[k * 6 + l] * OP[(q * 6 + l) * ND + s];
        CB[i] = W[q] * v;
    }
    for (int i = t; i < NL * NL; i += nt) {   // sum w grad N_a . S grad N_b
        const int a = i / NL, b = i % NL;
        double total = 0.0;
        for (int q = 0; q < NQ; ++q) {
            const double* ga = G + (q * NL + a) * 3;
            const double* gb = G + (q * NL + b) * 3;
            double part = 0.0;
            for (int k = 0; k < 6; ++k) {
                const int fi = VOIGT_FIRST[k], fj = VOIGT_SECOND[k];
                part += S[q * 6 + k] * ((fi == fj) ? ga[fi] * gb[fi] : ga[fi] * gb[fj] + ga[fj] * gb[fi]);
            }
            total += W[q] * part;
        }
        GEO[i] = total;
    }
    __syncthreads();
    double* out = blocks + e * (ND * ND);
    for (int i = t; i < ND * (ND + 1) / 2; i += nt) {   // lower-triangle entry i = r (r + 1) / 2 + s, s <= r
        int r = (int)((sqrt(8.0 * i + 1.0) - 1.0) * 0.5);
        while (r * (r + 1) / 2 > i) --r;
        while ((r + 1) * (r + 2) / 2 <= i) ++r;
        const int s = i - r * (r + 1) / 2;
        double total = 0.0;
        for (int j = 0; j < NQ * 6; ++j) total += OP[j * ND + r] * CB[j * ND + s];
        if (r % 3 == s % 3) total += GEO[(r / 3) * NL + s / 3];
        out[r * ND + s] = total;
        out[s * ND + r] = total;
    }
}
"""

_KERNELS = {}   # (n_local, points) -> cupy RawKernel, compiled on first use


def _kernel(n_local, points):
    kernel = _KERNELS.get((n_local, points))
    if kernel is None:
        import cupy as cp
        source = _SOURCE.replace('__NL__', str(int(n_local))).replace('__NQ__', str(int(points)))
        kernel = cp.RawKernel(source, 'element_tangent_3d')
        _KERNELS[(n_local, points)] = kernel
    return kernel


def _launch(grid, block, args):
    """RawKernel launch compiled for the (n_local, points) of args[0] (the gradients)."""
    _, points, n_local, _ = args[0].shape
    _kernel(int(n_local), int(points))(grid, block, args)


def _element_tangent_host(gradients, weights, local, material, fixed, force, blocks, n_elements, has_fixed,
                          want_tangent):
    """numpy twin (also the cupy route of a rule too large for the kernel): the batched elasticity3d formulas."""
    from .elasticity3d import _green_lagrange_terms_3d, _internal_force_blocks_3d, _tangent_blocks_3d
    operator, stress = _green_lagrange_terms_3d(gradients, local, material, fixed if int(has_fixed) else None)
    force[...] = _internal_force_blocks_3d(operator, stress, weights)
    if int(want_tangent):
        blocks[...] = _tangent_blocks_3d(gradients, weights, operator, stress, material)


def _element_tangent_numba(numba):
    """numba twin: prange over the elements, the CUDA kernel's passes on per-element arrays (Voigt tables: module
    globals, frozen at compile time)."""
    @numba.njit(parallel=True, cache=True)
    def kernel(gradients, weights, local, material, fixed, force, blocks, n_elements, n_points, n_local, has_fixed,
               want_tangent):
        columns = 3 * n_local
        for prange_e in numba.prange(n_elements):
            e = np.int64(prange_e)
            gradient = np.empty(9)
            strain = np.empty(6)
            stress = np.empty((n_points, 6))
            operator = np.empty((n_points, 6, columns))
            for q in range(n_points):
                base = (e * n_points + q) * n_local * 3
                for c in range(3):
                    for d in range(3):
                        acc = 0.0
                        for a in range(n_local):
                            acc += gradients[base + 3 * a + d] * local[(e * n_local + a) * 3 + c]
                        gradient[3 * c + d] = acc
                for k in range(6):
                    i, j = _VOIGT_FIRST[k], _VOIGT_SECOND[k]
                    quadratic = (gradient[i] * gradient[j] + gradient[3 + i] * gradient[3 + j]
                                 + gradient[6 + i] * gradient[6 + j])
                    if i == j:
                        strain[k] = gradient[4 * i] + 0.5 * quadratic
                    else:
                        strain[k] = gradient[3 * i + j] + gradient[3 * j + i] + quadratic
                for k in range(6):
                    value = 0.0
                    for l in range(6):
                        value += material[e * 36 + 6 * k + l] * strain[l]
                    if has_fixed:
                        value += fixed[(e * n_points + q) * 6 + k]
                    stress[q, k] = value
                for k in range(6):
                    i, j = _VOIGT_FIRST[k], _VOIGT_SECOND[k]
                    for a in range(n_local):
                        gi, gj = gradients[base + 3 * a + i], gradients[base + 3 * a + j]
                        for c in range(3):
                            fci = gradient[3 * c + i] + (1.0 if c == i else 0.0)
                            if i == j:
                                operator[q, k, 3 * a + c] = gi * fci
                            else:
                                fcj = gradient[3 * c + j] + (1.0 if c == j else 0.0)
                                operator[q, k, 3 * a + c] = gj * fci + gi * fcj
            for r in range(columns):
                total = 0.0
                for q in range(n_points):
                    part = 0.0
                    for k in range(6):
                        part += operator[q, k, r] * stress[q, k]
                    total += weights[e * n_points + q] * part
                force[e * columns + r] = total
            if want_tangent:
                weighted = np.empty((n_points, 6, columns))
                for q in range(n_points):
                    w = weights[e * n_points + q]
                    for k in range(6):
                        for s in range(columns):
                            value = 0.0
                            for l in range(6):
                                value += material[e * 36 + 6 * k + l] * operator[q, l, s]
                            weighted[q, k, s] = w * value
                geometric = np.empty((n_local, n_local))
                for a in range(n_local):
                    for b in range(n_local):
                        total = 0.0
                        for q in range(n_points):
                            base = (e * n_points + q) * n_local * 3
                            part = 0.0
                            for k in range(6):
                                i, j = _VOIGT_FIRST[k], _VOIGT_SECOND[k]
                                ga_i, ga_j = gradients[base + 3 * a + i], gradients[base + 3 * a + j]
                                gb_i, gb_j = gradients[base + 3 * b + i], gradients[base + 3 * b + j]
                                if i == j:
                                    part += stress[q, k] * ga_i * gb_i
                                else:
                                    part += stress[q, k] * (ga_i * gb_j + ga_j * gb_i)
                            total += weights[e * n_points + q] * part
                        geometric[a, b] = total
                out = e * columns * columns
                for r in range(columns):
                    for s in range(r + 1):
                        total = 0.0
                        for q in range(n_points):
                            for k in range(6):
                                total += operator[q, k, r] * weighted[q, k, s]
                        if r % 3 == s % 3:
                            total += geometric[r // 3, s // 3]
                        blocks[out + r * columns + s] = total
                        blocks[out + s * columns + r] = total

    def element_tangent_3d(gradients, weights, local, material, fixed, force, blocks, n_elements, has_fixed,
                           want_tangent):
        _, points, n_local, _ = gradients.shape
        kernel(gradients.reshape(-1), weights.reshape(-1), local.reshape(-1), material.reshape(-1), fixed.reshape(-1),
               force.reshape(-1), blocks.reshape(-1), int(n_elements), int(points), int(n_local), int(has_fixed),
               int(want_tangent))
    return element_tangent_3d


ELEMENT_TANGENT_3D = raw_twin('fem.element_tangent_3d', _launch, _element_tangent_host,
                              numba=_element_tangent_numba)


def element_force_tangent_3d(gradients, weights, local, material, fixed_stress=None, want_tangent=True):
    """(element forces (m, 3 n), element tangents (m, 3 n, 3 n) or None) of one chunk on the arrays' module: the CUDA
    kernel / numba twin, the batched formulas for a cupy rule too large for the kernel (kernel_fits)."""
    xp = np if isinstance(gradients, np.ndarray) else _cupy()
    m, points, n_local, _ = gradients.shape
    columns = 3 * n_local
    force = xp.empty((m, columns))
    blocks = xp.empty((m, columns, columns)) if want_tangent else xp.empty((1, 1, 1))
    fixed = xp.empty((1, 1, 1)) if fixed_stress is None else xp.ascontiguousarray(fixed_stress, dtype=np.float64)
    args = (xp.ascontiguousarray(gradients, dtype=np.float64), xp.ascontiguousarray(weights, dtype=np.float64),
            xp.ascontiguousarray(local, dtype=np.float64), xp.ascontiguousarray(material, dtype=np.float64), fixed,
            force, blocks, np.int32(m), np.int32(fixed_stress is not None), np.int32(bool(want_tangent)))
    if m:
        if xp is np or kernel_fits(n_local, points):
            ELEMENT_TANGENT_3D((m,), (KERNEL_THREADS,), args)
        else:
            _element_tangent_host(*args)
    return force, (blocks if want_tangent else None)


def _cupy():
    from ..core.backend import cp
    return cp
