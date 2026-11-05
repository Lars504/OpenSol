"""RawKernels of the matrix-free 3D elasticity (trilinear bricks on the window cells, nodal (u_x, u_y, u_z)), templated
on float / double and built lazily on first use, each a core.kernels twin with a numpy host version (the cpu backend;
numba twin of the stiffness product); this module imports no cupy at module level."""
import itertools
import math

import numpy as np

from ...core.kernels import raw_twin

# Cell-centre outputs in the argument order of elastic_outputs (the 3D field class keeps this order).
OUTPUT_ARRAYS_3D = ('u_x', 'u_y', 'u_z', 'u_abs',
                    'strain_xx', 'strain_yy', 'strain_zz', 'strain_xy', 'strain_xz', 'strain_yz',
                    'elastic_strain_xx', 'elastic_strain_yy', 'elastic_strain_zz',
                    'stress_xx', 'stress_yy', 'stress_zz', 'stress_xy', 'stress_xz', 'stress_yz', 'von_mises')

_BLOCK = 256
_REAL_TYPES = {'float32': 'float', 'float64': 'double'}
_MASK_ON, _MASK_OFF = np.int32(1), np.int32(0)

# Layout: elements (ex, ey, ez) = window cells, element e = (ei ey + ej) ez + ek; nodes (ex + 1, ey + 1, ez + 1),
# node = (i nyn + j) nzn + k, dof = 3 node + component; lambda = mu = 0 marks a void element (skipped). Full-grid
# float32 inputs / outputs are indexed through the window origin (ox, oy, oz) and the full ny, nz.
_PREAMBLE = r"""
typedef long long idx_t;

__device__ __forceinline__ real_t slope_sign(int local) { return local ? (real_t)1 : (real_t)-1; }

__device__ __forceinline__ idx_t full_index(int ei, int ej, int ek, int ox, int oy, int oz, int ny_full, int nz_full)
{
    return ((idx_t)(ox + ei) * ny_full + (oy + ej)) * nz_full + (oz + ek);
}

__device__ __forceinline__ void node_ijk(int node, int nyn, int nzn, int* i, int* j, int* k)
{
    *k = node % nzn;
    const int rest = node / nzn;
    *j = rest % nyn;
    *i = rest / nyn;
}

// K[a][b] = block (component a of local node (kx, ky, kz), component b of local node (lx, ly, lz)) of the exact brick
// stiffness on wx x wy x wz: lambda I_ab + mu I_ba + delta_ab mu sum_c I_cc with I_ab = int dN_k/dx_a dN_l/dx_b dV.
__device__ __forceinline__ void brick_block(
    int kx, int ky, int kz, int lx, int ly, int lz, real_t wx, real_t wy, real_t wz,
    real_t lam, real_t mu, real_t K[3][3])
{
    const real_t sk[3] = {slope_sign(kx), slope_sign(ky), slope_sign(kz)};
    const real_t sl[3] = {slope_sign(lx), slope_sign(ly), slope_sign(lz)};
    const real_t w[3] = {wx, wy, wz};
    const int same[3] = {kx == lx, ky == ly, kz == lz};
    real_t D[3], M[3];   // 1D integrals: N_k' N_l' = s_k s_l / w, N_k N_l = w (1 + delta) / 6
    for (int d = 0; d < 3; ++d) {
        D[d] = sk[d] * sl[d] / w[d];
        M[d] = w[d] * (real_t)(1 + same[d]) / (real_t)6;
    }
    real_t I[3][3];
    I[0][0] = D[0] * M[1] * M[2];
    I[1][1] = M[0] * D[1] * M[2];
    I[2][2] = M[0] * M[1] * D[2];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            if (a != b) I[a][b] = (real_t)0.25 * sk[a] * sl[b] * M[3 - a - b];   // (s_ka / 2)(s_lb / 2) M_c
    const real_t trace = I[0][0] + I[1][1] + I[2][2];
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            K[a][b] = lam * I[a][b] + mu * I[b][a] + (a == b ? mu * trace : (real_t)0);
}

// brick_block's 1D factors of local node k against both local nodes l = 0, 1 per axis (once per element, not per
// pair): sk = slope sign of k, D[d][l] = s_k s_l / w, M[d][l] = w (1 + delta_kl) / 6.
__device__ __forceinline__ void axis_tables(
    int kx, int ky, int kz, real_t wx, real_t wy, real_t wz, real_t sk[3], real_t D[3][2], real_t M[3][2])
{
    const int k[3] = {kx, ky, kz};
    const real_t w[3] = {wx, wy, wz};
    for (int d = 0; d < 3; ++d) {
        sk[d] = slope_sign(k[d]);
        const real_t slope = sk[d] / w[d];
        const real_t sixth = w[d] / (real_t)6;
        for (int l = 0; l < 2; ++l) {
            D[d][l] = slope_sign(l) * slope;
            M[d][l] = sixth * (real_t)(1 + (l == k[d]));
        }
    }
}

// acc += K u_l with K = brick_block(k, l) from the axis_tables of k (K never formed).
__device__ __forceinline__ void add_table_block(
    int lx, int ly, int lz, const real_t sk[3], const real_t D[3][2], const real_t M[3][2], real_t lam, real_t mu,
    real_t u0, real_t u1, real_t u2, real_t acc[3])
{
    const real_t m0 = M[0][lx], m1 = M[1][ly], m2 = M[2][lz];
    const real_t i00 = D[0][lx] * m1 * m2, i11 = m0 * D[1][ly] * m2, i22 = m0 * m1 * D[2][lz];
    const real_t h0 = (real_t)0.25 * slope_sign(lx), h1 = (real_t)0.25 * slope_sign(ly);
    const real_t h2 = (real_t)0.25 * slope_sign(lz);
    // I_ab (a != b) = (s_ka / 2)(s_lb / 2) M_c, c the third axis
    const real_t i01 = sk[0] * h1 * m2, i10 = sk[1] * h0 * m2, i02 = sk[0] * h2 * m1, i20 = sk[2] * h0 * m1;
    const real_t i12 = sk[1] * h2 * m0, i21 = sk[2] * h1 * m0;
    const real_t trace = mu * (i00 + i11 + i22);
    const real_t shear = lam + mu;
    acc[0] += (shear * i00 + trace) * u0 + (lam * i01 + mu * i10) * u1 + (lam * i02 + mu * i20) * u2;
    acc[1] += (lam * i10 + mu * i01) * u0 + (shear * i11 + trace) * u1 + (lam * i12 + mu * i21) * u2;
    acc[2] += (lam * i20 + mu * i02) * u0 + (lam * i21 + mu * i12) * u1 + (shear * i22 + trace) * u2;
}
"""

_SOURCES = {
    # y = K u per node (gather over the <= 8 adjacent solid elements); use_mask zeroes the rows of fixed dofs.
    'elastic_product': r"""
extern "C" __global__
void elastic_product(
    const real_t* __restrict__ u, real_t* __restrict__ y,
    const real_t* __restrict__ lam, const real_t* __restrict__ mu,
    const real_t* __restrict__ wx, const real_t* __restrict__ wy, const real_t* __restrict__ wz,
    const bool* __restrict__ free_dofs, int use_mask, int ex, int ey, int ez)
{
    const int nyn = ey + 1, nzn = ez + 1;
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= (ex + 1) * nyn * nzn) return;
    int i, j, k;
    node_ijk(node, nyn, nzn, &i, &j, &k);
    real_t acc[3] = {(real_t)0, (real_t)0, (real_t)0};
    for (int di = 0; di < 2; ++di) {
        const int ei = i - 1 + di;
        if (ei < 0 || ei >= ex) continue;
        for (int dj = 0; dj < 2; ++dj) {
            const int ej = j - 1 + dj;
            if (ej < 0 || ej >= ey) continue;
            for (int dk = 0; dk < 2; ++dk) {
                const int ek = k - 1 + dk;
                if (ek < 0 || ek >= ez) continue;
                const int e = (ei * ey + ej) * ez + ek;
                const real_t m = mu[e];
                if (m <= (real_t)0) continue;
                const real_t l = lam[e];
                real_t sk[3], D[3][2], M[3][2];   // per element: no per-pair divisions (float64 rate on consumer GPUs)
                axis_tables(1 - di, 1 - dj, 1 - dk, wx[ei], wy[ej], wz[ek], sk, D, M);
                for (int lx = 0; lx < 2; ++lx)
                    for (int ly = 0; ly < 2; ++ly)
                        for (int lz = 0; lz < 2; ++lz) {
                            const int nb = 3 * (((ei + lx) * nyn + (ej + ly)) * nzn + (ek + lz));
                            add_table_block(lx, ly, lz, sk, D, M, l, m, u[nb], u[nb + 1], u[nb + 2], acc);
                        }
            }
        }
    }
    for (int a = 0; a < 3; ++a) {
        const int dof = 3 * node + a;
        y[dof] = (use_mask && !free_dofs[dof]) ? (real_t)0 : acc[a];
    }
}
""",
    # y = M u per node of the consistent trilinear mass (rho N_k N_l per component; rho <= 0 elements skipped), 0 on
    # the rows of fixed dofs.
    'elastic_mass_product': r"""
extern "C" __global__
void elastic_mass_product(
    const real_t* __restrict__ u, real_t* __restrict__ y, const real_t* __restrict__ rho,
    const real_t* __restrict__ wx, const real_t* __restrict__ wy, const real_t* __restrict__ wz,
    const bool* __restrict__ free_dofs, int ex, int ey, int ez)
{
    const int nyn = ey + 1, nzn = ez + 1;
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= (ex + 1) * nyn * nzn) return;
    int i, j, k;
    node_ijk(node, nyn, nzn, &i, &j, &k);
    real_t acc[3] = {(real_t)0, (real_t)0, (real_t)0};
    for (int di = 0; di < 2; ++di) {
        const int ei = i - 1 + di;
        if (ei < 0 || ei >= ex) continue;
        for (int dj = 0; dj < 2; ++dj) {
            const int ej = j - 1 + dj;
            if (ej < 0 || ej >= ey) continue;
            for (int dk = 0; dk < 2; ++dk) {
                const int ek = k - 1 + dk;
                if (ek < 0 || ek >= ez) continue;
                const int e = (ei * ey + ej) * ez + ek;
                const real_t density = rho[e];
                if (density <= (real_t)0) continue;
                const real_t sixth = (real_t)1 / (real_t)6;
                for (int lx = 0; lx < 2; ++lx)
                    for (int ly = 0; ly < 2; ++ly)
                        for (int lz = 0; lz < 2; ++lz) {
                            // 1D N_k N_l = w (1 + delta) / 6; local node of this node is (1 - di, 1 - dj, 1 - dk)
                            const real_t m = density * wx[ei] * (real_t)(1 + (lx == 1 - di)) * sixth
                                             * wy[ej] * (real_t)(1 + (ly == 1 - dj)) * sixth
                                             * wz[ek] * (real_t)(1 + (lz == 1 - dk)) * sixth;
                            const int nb = 3 * (((ei + lx) * nyn + (ej + ly)) * nzn + (ek + lz));
                            for (int a = 0; a < 3; ++a) acc[a] += m * u[nb + a];
                        }
            }
        }
    }
    for (int a = 0; a < 3; ++a) {
        const int dof = 3 * node + a;
        y[dof] = free_dofs[dof] ? acc[a] : (real_t)0;
    }
}
""",
    # Own 3 x 3 block of every node (xx, yy, zz, xy, xz, yz) of the full stiffness.
    'elastic_diagonal_blocks': r"""
extern "C" __global__
void elastic_diagonal_blocks(
    real_t* __restrict__ blocks, const real_t* __restrict__ lam, const real_t* __restrict__ mu,
    const real_t* __restrict__ wx, const real_t* __restrict__ wy, const real_t* __restrict__ wz,
    int ex, int ey, int ez)
{
    const int nyn = ey + 1, nzn = ez + 1;
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= (ex + 1) * nyn * nzn) return;
    int i, j, k;
    node_ijk(node, nyn, nzn, &i, &j, &k);
    real_t acc[6] = {(real_t)0, (real_t)0, (real_t)0, (real_t)0, (real_t)0, (real_t)0};
    for (int di = 0; di < 2; ++di) {
        const int ei = i - 1 + di;
        if (ei < 0 || ei >= ex) continue;
        for (int dj = 0; dj < 2; ++dj) {
            const int ej = j - 1 + dj;
            if (ej < 0 || ej >= ey) continue;
            for (int dk = 0; dk < 2; ++dk) {
                const int ek = k - 1 + dk;
                if (ek < 0 || ek >= ez) continue;
                const int e = (ei * ey + ej) * ez + ek;
                const real_t m = mu[e];
                if (m <= (real_t)0) continue;
                real_t K[3][3];
                brick_block(1 - di, 1 - dj, 1 - dk, 1 - di, 1 - dj, 1 - dk, wx[ei], wy[ej], wz[ek], lam[e], m, K);
                acc[0] += K[0][0]; acc[1] += K[1][1]; acc[2] += K[2][2];
                acc[3] += K[0][1]; acc[4] += K[0][2]; acc[5] += K[1][2];
            }
        }
    }
    const idx_t base = 6 * (idx_t)node;
    for (int n = 0; n < 6; ++n) blocks[base + n] = acc[n];
}
""",
    # In place: B -> (D_free B D_free + D_fixed)^-1 per node; diagonal inverse when the reduced block is not SPD.
    'reduce_invert_blocks': r"""
extern "C" __global__
void reduce_invert_blocks(real_t* __restrict__ blocks, const bool* __restrict__ free_dofs, int n_nodes)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;
    real_t* b = blocks + 6 * (idx_t)node;
    const bool f0 = free_dofs[3 * node], f1 = free_dofs[3 * node + 1], f2 = free_dofs[3 * node + 2];
    const real_t one = (real_t)1, zero = (real_t)0;
    const real_t b00 = f0 ? b[0] : one, b11 = f1 ? b[1] : one, b22 = f2 ? b[2] : one;
    const real_t b01 = (f0 && f1) ? b[3] : zero, b02 = (f0 && f2) ? b[4] : zero, b12 = (f1 && f2) ? b[5] : zero;
    const real_t c00 = b11 * b22 - b12 * b12, c11 = b00 * b22 - b02 * b02, c22 = b00 * b11 - b01 * b01;
    const real_t c01 = b02 * b12 - b01 * b22, c02 = b01 * b12 - b02 * b11, c12 = b01 * b02 - b00 * b12;
    const real_t det = b00 * c00 + b01 * c01 + b02 * c02;
    if (b00 > zero && c22 > zero && det > zero) {
        b[0] = c00 / det; b[1] = c11 / det; b[2] = c22 / det;
        b[3] = c01 / det; b[4] = c02 / det; b[5] = c12 / det;
    } else {
        b[0] = b00 > zero ? one / b00 : one;
        b[1] = b11 > zero ? one / b11 : one;
        b[2] = b22 > zero ? one / b22 : one;
        b[3] = zero; b[4] = zero; b[5] = zero;
    }
}
""",
    # z = B_solve^-1 r per node (symmetric inverse blocks xx, yy, zz, xy, xz, yz).
    'block_precondition3': r"""
extern "C" __global__
void block_precondition3(
    const real_t* __restrict__ inverse, const real_t* __restrict__ r, real_t* __restrict__ z, int n_nodes)
{
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= n_nodes) return;
    const real_t* b = inverse + 6 * (idx_t)node;
    const int dof = 3 * node;
    const real_t r0 = r[dof], r1 = r[dof + 1], r2 = r[dof + 2];
    z[dof] = b[0] * r0 + b[3] * r1 + b[4] * r2;
    z[dof + 1] = b[3] * r0 + b[1] * r1 + b[5] * r2;
    z[dof + 2] = b[4] * r0 + b[5] * r1 + b[2] * r2;
}
""",
    # Total nodal load [N] = applied + per adjacent solid element: thermal beta_a dT s_ka A_perp_a / 4 (lower-side
    # nodes -, upper-side +) and body force (coupled + applied) times volume / 8.
    'elastic_load': r"""
extern "C" __global__
void elastic_load(
    real_t* __restrict__ load, const real_t* __restrict__ applied_load,
    const real_t* __restrict__ lam, const real_t* __restrict__ mu,
    const float* __restrict__ alpha_xx, const float* __restrict__ alpha_yy, const float* __restrict__ alpha_zz,
    const float* __restrict__ rise, const float* __restrict__ rise_ext,
    const float* __restrict__ body_x, const float* __restrict__ body_y, const float* __restrict__ body_z,
    const real_t* __restrict__ applied_body,
    const real_t* __restrict__ wx, const real_t* __restrict__ wy, const real_t* __restrict__ wz,
    int ex, int ey, int ez, int ox, int oy, int oz, int ny_full, int nz_full)
{
    const int nyn = ey + 1, nzn = ez + 1;
    const int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= (ex + 1) * nyn * nzn) return;
    int i, j, k;
    node_ijk(node, nyn, nzn, &i, &j, &k);
    const int n_elements = ex * ey * ez;
    real_t acc[3] = {applied_load[3 * node], applied_load[3 * node + 1], applied_load[3 * node + 2]};
    for (int di = 0; di < 2; ++di) {
        const int ei = i - 1 + di;
        if (ei < 0 || ei >= ex) continue;
        for (int dj = 0; dj < 2; ++dj) {
            const int ej = j - 1 + dj;
            if (ej < 0 || ej >= ey) continue;
            for (int dk = 0; dk < 2; ++dk) {
                const int ek = k - 1 + dk;
                if (ek < 0 || ek >= ez) continue;
                const int e = (ei * ey + ej) * ez + ek;
                const real_t m = mu[e];
                if (m <= (real_t)0) continue;
                const real_t l = lam[e];
                const idx_t g = full_index(ei, ej, ek, ox, oy, oz, ny_full, nz_full);
                const real_t a_w = wx[ei], b_w = wy[ej], c_w = wz[ek];
                const real_t area[3] = {b_w * c_w, a_w * c_w, a_w * b_w};
                const real_t sign[3] = {di ? (real_t)-1 : (real_t)1, dj ? (real_t)-1 : (real_t)1,
                                        dk ? (real_t)-1 : (real_t)1};
                const real_t alpha[3] = {(real_t)alpha_xx[g], (real_t)alpha_yy[g], (real_t)alpha_zz[g]};
                const real_t alpha_sum = alpha[0] + alpha[1] + alpha[2];
                const real_t dT = (real_t)rise[g] + (real_t)rise_ext[g];
                const real_t body[3] = {(real_t)body_x[g] + applied_body[e],
                                        (real_t)body_y[g] + applied_body[n_elements + e],
                                        (real_t)body_z[g] + applied_body[2 * n_elements + e]};
                const real_t eighth_volume = (real_t)0.125 * a_w * b_w * c_w;
                for (int a = 0; a < 3; ++a) {
                    const real_t beta = (real_t)2 * m * alpha[a] + l * alpha_sum;   // (C alpha)_aa per kelvin
                    acc[a] += (real_t)0.25 * sign[a] * beta * dT * area[a] + eighth_volume * body[a];
                }
            }
        }
    }
    for (int a = 0; a < 3; ++a) load[3 * node + a] = acc[a];
}
""",
    # Per element: centre u (8-corner mean), centre gradients, strain, elastic strain, stress, von Mises into the
    # full-grid float32 outputs (OUTPUT_ARRAYS_3D order); 0 in void elements.
    'elastic_outputs': r"""
extern "C" __global__
void elastic_outputs(
    const real_t* __restrict__ u, const real_t* __restrict__ lam, const real_t* __restrict__ mu,
    const float* __restrict__ alpha_xx, const float* __restrict__ alpha_yy, const float* __restrict__ alpha_zz,
    const float* __restrict__ rise, const float* __restrict__ rise_ext,
    const real_t* __restrict__ wx, const real_t* __restrict__ wy, const real_t* __restrict__ wz,
    float* u_x, float* u_y, float* u_z, float* u_abs,
    float* strain_xx, float* strain_yy, float* strain_zz, float* strain_xy, float* strain_xz, float* strain_yz,
    float* elastic_xx, float* elastic_yy, float* elastic_zz,
    float* stress_xx, float* stress_yy, float* stress_zz, float* stress_xy, float* stress_xz, float* stress_yz,
    float* von_mises, int thermal,
    int ex, int ey, int ez, int ox, int oy, int oz, int ny_full, int nz_full)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= ex * ey * ez) return;
    const int ek = e % ez, rest = e / ez, ej = rest % ey, ei = rest / ey;
    const idx_t g = full_index(ei, ej, ek, ox, oy, oz, ny_full, nz_full);
    float* outputs[20] = {u_x, u_y, u_z, u_abs, strain_xx, strain_yy, strain_zz, strain_xy, strain_xz, strain_yz,
                          elastic_xx, elastic_yy, elastic_zz, stress_xx, stress_yy, stress_zz, stress_xy, stress_xz,
                          stress_yz, von_mises};
    const real_t m = mu[e];
    if (m <= (real_t)0) {
        for (int n = 0; n < 20; ++n) outputs[n][g] = 0.0f;
        return;
    }
    const int nyn = ey + 1, nzn = ez + 1;
    real_t centre[3] = {(real_t)0, (real_t)0, (real_t)0};
    real_t grad[3][3] = {{(real_t)0, (real_t)0, (real_t)0}, {(real_t)0, (real_t)0, (real_t)0},
                         {(real_t)0, (real_t)0, (real_t)0}};   // grad[a][d] = du_a / dx_d
    for (int cx = 0; cx < 2; ++cx)
        for (int cy = 0; cy < 2; ++cy)
            for (int cz = 0; cz < 2; ++cz) {
                const int nb = 3 * (((ei + cx) * nyn + (ej + cy)) * nzn + (ek + cz));
                const real_t s[3] = {slope_sign(cx), slope_sign(cy), slope_sign(cz)};
                for (int a = 0; a < 3; ++a) {
                    const real_t value = u[nb + a];
                    centre[a] += value;
                    for (int d = 0; d < 3; ++d) grad[a][d] += s[d] * value;
                }
            }
    const real_t w[3] = {wx[ei], wy[ej], wz[ek]};
    for (int a = 0; a < 3; ++a) {
        centre[a] *= (real_t)0.125;
        for (int d = 0; d < 3; ++d) grad[a][d] /= (real_t)4 * w[d];   // mean of the 4 edge differences
    }
    const real_t eps_xx = grad[0][0], eps_yy = grad[1][1], eps_zz = grad[2][2];
    const real_t eps_xy = (real_t)0.5 * (grad[0][1] + grad[1][0]);   // tensor shear strains
    const real_t eps_xz = (real_t)0.5 * (grad[0][2] + grad[2][0]);
    const real_t eps_yz = (real_t)0.5 * (grad[1][2] + grad[2][1]);
    const real_t l = lam[e];
    const real_t alpha[3] = {(real_t)alpha_xx[g], (real_t)alpha_yy[g], (real_t)alpha_zz[g]};
    const real_t alpha_sum = alpha[0] + alpha[1] + alpha[2];
    const real_t dT = thermal ? (real_t)rise[g] + (real_t)rise_ext[g] : (real_t)0;   // 0: a mode shape
    const real_t volumetric = l * (eps_xx + eps_yy + eps_zz);
    const real_t thermal_x = ((real_t)2 * m * alpha[0] + l * alpha_sum) * dT;
    const real_t thermal_y = ((real_t)2 * m * alpha[1] + l * alpha_sum) * dT;
    const real_t thermal_z = ((real_t)2 * m * alpha[2] + l * alpha_sum) * dT;
    const real_t s_xx = volumetric + (real_t)2 * m * eps_xx - thermal_x;
    const real_t s_yy = volumetric + (real_t)2 * m * eps_yy - thermal_y;
    const real_t s_zz = volumetric + (real_t)2 * m * eps_zz - thermal_z;
    const real_t s_xy = (real_t)2 * m * eps_xy, s_xz = (real_t)2 * m * eps_xz, s_yz = (real_t)2 * m * eps_yz;
    const real_t mises = sqrt((real_t)0.5 * ((s_xx - s_yy) * (s_xx - s_yy) + (s_yy - s_zz) * (s_yy - s_zz)
                                             + (s_zz - s_xx) * (s_zz - s_xx))
                              + (real_t)3 * (s_xy * s_xy + s_xz * s_xz + s_yz * s_yz));
    const real_t values[20] = {
        centre[0], centre[1], centre[2],
        sqrt(centre[0] * centre[0] + centre[1] * centre[1] + centre[2] * centre[2]),
        eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz,
        eps_xx - alpha[0] * dT, eps_yy - alpha[1] * dT, eps_zz - alpha[2] * dT,
        s_xx, s_yy, s_zz, s_xy, s_xz, s_yz, mises};
    for (int n = 0; n < 20; ++n) outputs[n][g] = (float)values[n];
}
""",
}

_KERNELS = {}   # (name, C type) -> cupy RawKernel, compiled on first use


def _kernel(name, dtype):
    real = _REAL_TYPES[np.dtype(dtype).name]
    kernel = _KERNELS.get((name, real))
    if kernel is None:
        import cupy as cp
        kernel = cp.RawKernel(f'typedef {real} real_t;\n' + _PREAMBLE + _SOURCES[name], name)
        _KERNELS[(name, real)] = kernel
    return kernel


def _gpu_launcher(name):
    """kernel(grid, block, args) of the RawKernel compiled for the real type of args[0] (every kernel's first array)."""
    def launch(grid, block, args):
        _kernel(name, args[0].dtype)(grid, block, args)
    return launch


# ---- numpy twins (core.kernels raw_twin: the kernel's args tuple, outputs written in place, the solve dtype) ----
# Element integrals by 2-point Gauss per axis (exact for the trilinear products, so equal to the CUDA closed forms up to
# rounding), through separable differences / interpolations of the node array. A change to a CUDA source above changes
# its twin here; the cpu self-test solid_mechanics_3d_cpu (the exact 3D cases on the twins) catches a wrong twin.
_GAUSS = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))   # abscissae on [0, 1], weight 1/2 each
_GAUSS_POINTS = tuple(itertools.product(_GAUSS, repeat=3))
_CORNERS = tuple(itertools.product((0, 1), repeat=3))


def _lower(values, axis):
    index = [slice(None)] * values.ndim
    index[axis] = slice(None, -1)
    return values[tuple(index)]


def _upper(values, axis):
    index = [slice(None)] * values.ndim
    index[axis] = slice(1, None)
    return values[tuple(index)]


def _interpolate(values, axis, t):
    """Linear interpolation at local coordinate t between neighbours along axis (length - 1)."""
    return (1.0 - t) * _lower(values, axis) + t * _upper(values, axis)


def _interpolate_adjoint(values, axis, t):
    """Transpose of _interpolate (length + 1 along axis)."""
    shape = list(values.shape)
    shape[axis] += 1
    out = np.zeros(shape, dtype=values.dtype)
    lower, upper = _lower(out, axis), _upper(out, axis)
    lower += (1.0 - t) * values
    upper += t * values
    return out


def _along(widths, axis, ndim):
    """1D widths reshaped to broadcast along axis of an ndim array."""
    shape = [1] * ndim
    shape[axis] = widths.size
    return widths.reshape(shape)


def _element_lame_host(lame_lambda, mu, element_shape):
    """(solid, lambda, mu) per element with void (mu <= 0) elements zeroed, as the kernels skip them."""
    shear = mu.reshape(element_shape)
    solid = shear > 0
    return solid, np.where(solid, lame_lambda.reshape(element_shape), 0), np.where(solid, shear, 0)


def _gradients_at(differences, point):
    """(du/dx, du/dy, du/dz) (each (ex, ey, ez, 3)) at a local point from the per-axis node differences / width."""
    gradients = []
    for axis, value in enumerate(differences):
        for other in range(3):
            if other != axis:
                value = _interpolate(value, other, point[other])
        gradients.append(value)
    return gradients


def _elastic_product_host(u, y, lame_lambda, mu, wx, wy, wz, free_dofs, use_mask, ex, ey, ez):
    ex, ey, ez = int(ex), int(ey), int(ez)
    nodes = u.reshape(ex + 1, ey + 1, ez + 1, 3)
    _, lam, shear = _element_lame_host(lame_lambda, mu, (ex, ey, ez))
    widths = (wx, wy, wz)
    weight = (0.125 * wx[:, None, None] * wy[None, :, None] * wz[None, None, :])[..., None]   # Gauss weight x volume
    differences = [(_upper(nodes, axis) - _lower(nodes, axis)) / _along(widths[axis], axis, 4) for axis in range(3)]
    gathered = [np.zeros_like(values) for values in differences]
    for point in _GAUSS_POINTS:
        gx, gy, gz = _gradients_at(differences, point)
        lam_trace = lam * (gx[..., 0] + gy[..., 1] + gz[..., 2])
        s_xy = shear * (gy[..., 0] + gx[..., 1])
        s_xz = shear * (gz[..., 0] + gx[..., 2])
        s_yz = shear * (gz[..., 1] + gy[..., 2])
        # stresses[d][..., a] = sigma_ad: y_(node, a) = sum_d int sigma_ad dN_node / dx_d
        stresses = (np.stack((lam_trace + 2.0 * shear * gx[..., 0], s_xy, s_xz), axis=-1),
                    np.stack((s_xy, lam_trace + 2.0 * shear * gy[..., 1], s_yz), axis=-1),
                    np.stack((s_xz, s_yz, lam_trace + 2.0 * shear * gz[..., 2]), axis=-1))
        for axis in range(3):
            value = stresses[axis] * weight
            for other in range(3):
                if other != axis:
                    value = _interpolate_adjoint(value, other, point[other])
            gathered[axis] += value
    result = np.zeros_like(nodes)
    for axis in range(3):
        scaled = gathered[axis] / _along(widths[axis], axis, 4)
        lower, upper = _lower(result, axis), _upper(result, axis)
        lower -= scaled
        upper += scaled
    flat = result.reshape(-1)
    if int(use_mask):
        flat[~free_dofs] = 0
    y[...] = flat


def _elastic_product_numba(numba):
    """numba twin of elastic_product: the CUDA per-node gather with the closed-form brick blocks (brick_block written
    out per entry), parallel over the node x index; the solve dtype of u / y."""
    @numba.njit(parallel=True, cache=False)
    def kernel(u, y, lame_lambda, mu, wx, wy, wz, free_dofs, use_mask, ex, ey, ez):
        nyn, nzn = ey + 1, ez + 1
        for prange_i in numba.prange(ex + 1):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            for j in range(nyn):
                for k in range(nzn):
                    acc0, acc1, acc2 = 0.0, 0.0, 0.0
                    for di in range(2):
                        ei = i - 1 + di
                        if ei < 0 or ei >= ex:
                            continue
                        sk0 = 1.0 if di == 0 else -1.0   # slope sign of this node, local index 1 - di
                        w0 = wx[ei]
                        for dj in range(2):
                            ej = j - 1 + dj
                            if ej < 0 or ej >= ey:
                                continue
                            sk1 = 1.0 if dj == 0 else -1.0
                            w1 = wy[ej]
                            for dk in range(2):
                                ek = k - 1 + dk
                                if ek < 0 or ek >= ez:
                                    continue
                                e = (ei * ey + ej) * ez + ek
                                shear = mu[e]
                                if shear <= 0.0:
                                    continue
                                lam = lame_lambda[e]
                                sk2 = 1.0 if dk == 0 else -1.0
                                w2 = wz[ek]
                                for lx in range(2):
                                    sl0 = 1.0 if lx == 1 else -1.0
                                    m0 = w0 * (2.0 if lx == 1 - di else 1.0) / 6.0   # 1D N_k N_l
                                    d0 = sk0 * sl0 / w0                                # 1D N_k' N_l'
                                    for ly in range(2):
                                        sl1 = 1.0 if ly == 1 else -1.0
                                        m1 = w1 * (2.0 if ly == 1 - dj else 1.0) / 6.0
                                        d1 = sk1 * sl1 / w1
                                        for lz in range(2):
                                            sl2 = 1.0 if lz == 1 else -1.0
                                            m2 = w2 * (2.0 if lz == 1 - dk else 1.0) / 6.0
                                            d2 = sk2 * sl2 / w2
                                            i00, i11, i22 = d0 * m1 * m2, m0 * d1 * m2, m0 * m1 * d2
                                            trace = i00 + i11 + i22
                                            i01, i10 = 0.25 * sk0 * sl1 * m2, 0.25 * sk1 * sl0 * m2
                                            i02, i20 = 0.25 * sk0 * sl2 * m1, 0.25 * sk2 * sl0 * m1
                                            i12, i21 = 0.25 * sk1 * sl2 * m0, 0.25 * sk2 * sl1 * m0
                                            nb = 3 * (((ei + lx) * nyn + (ej + ly)) * nzn + (ek + lz))
                                            u0, u1, u2 = u[nb], u[nb + 1], u[nb + 2]
                                            acc0 += (((lam + shear) * i00 + shear * trace) * u0
                                                     + (lam * i01 + shear * i10) * u1 + (lam * i02 + shear * i20) * u2)
                                            acc1 += ((lam * i10 + shear * i01) * u0
                                                     + ((lam + shear) * i11 + shear * trace) * u1
                                                     + (lam * i12 + shear * i21) * u2)
                                            acc2 += ((lam * i20 + shear * i02) * u0 + (lam * i21 + shear * i12) * u1
                                                     + ((lam + shear) * i22 + shear * trace) * u2)
                    dof = 3 * ((i * nyn + j) * nzn + k)
                    masked = use_mask != 0
                    y[dof] = 0.0 if masked and not free_dofs[dof] else acc0
                    y[dof + 1] = 0.0 if masked and not free_dofs[dof + 1] else acc1
                    y[dof + 2] = 0.0 if masked and not free_dofs[dof + 2] else acc2

    def elastic_product(u, y, lame_lambda, mu, wx, wy, wz, free_dofs, use_mask, ex, ey, ez):
        kernel(u.reshape(-1), y.reshape(-1), lame_lambda.reshape(-1), mu.reshape(-1), wx, wy, wz,
               free_dofs.reshape(-1), use_mask, ex, ey, ez)
    return elastic_product


def _elastic_mass_product_host(u, y, rho, wx, wy, wz, free_dofs, ex, ey, ez):
    ex, ey, ez = int(ex), int(ey), int(ez)
    nodes = u.reshape(ex + 1, ey + 1, ez + 1, 3)
    density = rho.reshape(ex, ey, ez)
    weight = (np.where(density > 0, density, 0)
              * (0.125 * wx[:, None, None] * wy[None, :, None] * wz[None, None, :]))[..., None]
    result = np.zeros_like(nodes)
    for point in _GAUSS_POINTS:
        value = nodes
        for axis in range(3):
            value = _interpolate(value, axis, point[axis])
        value = value * weight
        for axis in range(3):
            value = _interpolate_adjoint(value, axis, point[axis])
        result += value
    flat = result.reshape(-1)
    flat[~free_dofs] = 0
    y[...] = flat


def _elastic_diagonal_blocks_host(blocks, lame_lambda, mu, wx, wy, wz, ex, ey, ez):
    ex, ey, ez = int(ex), int(ey), int(ez)
    _, lam, shear = _element_lame_host(lame_lambda, mu, (ex, ey, ez))
    a_w, b_w, c_w = wx[:, None, None], wy[None, :, None], wz[None, None, :]
    # own-node integrals: I_aa = (1 / w_a)(w_b / 3)(w_c / 3); I_ab = s_a s_b w_c / 12 (a != b, c the third axis)
    i_xx, i_yy, i_zz = b_w * c_w / (9.0 * a_w), a_w * c_w / (9.0 * b_w), a_w * b_w / (9.0 * c_w)
    trace = i_xx + i_yy + i_zz
    lam_mu = lam + shear
    diagonal = (lam_mu * i_xx + shear * trace, lam_mu * i_yy + shear * trace, lam_mu * i_zz + shear * trace)
    off_diagonal = (lam_mu * (c_w / 12.0), lam_mu * (b_w / 12.0), lam_mu * (a_w / 12.0))   # xy, xz, yz
    node_blocks = blocks.reshape(ex + 1, ey + 1, ez + 1, 6)
    node_blocks[...] = 0
    for corner in _CORNERS:
        sign = tuple(1.0 if offset else -1.0 for offset in corner)
        target = node_blocks[corner[0]:corner[0] + ex, corner[1]:corner[1] + ey, corner[2]:corner[2] + ez]
        for entry in range(3):
            target[..., entry] += diagonal[entry]
        target[..., 3] += sign[0] * sign[1] * off_diagonal[0]
        target[..., 4] += sign[0] * sign[2] * off_diagonal[1]
        target[..., 5] += sign[1] * sign[2] * off_diagonal[2]


def _reduce_invert_blocks_host(blocks, free_dofs, n_nodes):
    b = blocks.reshape(-1, 6)
    free = free_dofs.reshape(-1, 3)
    f0, f1, f2 = free[:, 0], free[:, 1], free[:, 2]
    b00, b11, b22 = (np.where(flag, b[:, index], 1) for flag, index in ((f0, 0), (f1, 1), (f2, 2)))
    b01 = np.where(f0 & f1, b[:, 3], 0)
    b02 = np.where(f0 & f2, b[:, 4], 0)
    b12 = np.where(f1 & f2, b[:, 5], 0)
    c00, c11, c22 = b11 * b22 - b12 * b12, b00 * b22 - b02 * b02, b00 * b11 - b01 * b01
    c01, c02, c12 = b02 * b12 - b01 * b22, b01 * b12 - b02 * b11, b01 * b02 - b00 * b12
    det = b00 * c00 + b01 * c01 + b02 * c02
    spd = (b00 > 0) & (c22 > 0) & (det > 0)
    safe_det = np.where(spd, det, 1)
    inverse = np.stack((c00, c11, c22, c01, c02, c12), axis=1) / safe_det[:, None]
    zero = np.zeros_like(b00)
    fallback = np.stack(tuple(np.where(value > 0, 1 / np.where(value > 0, value, 1), 1) for value in (b00, b11, b22))
                        + (zero, zero, zero), axis=1)
    b[...] = np.where(spd[:, None], inverse, fallback)


def _block_precondition3_host(inverse, residual, out, n_nodes):
    b = inverse.reshape(-1, 6)
    r = residual.reshape(-1, 3)
    r0, r1, r2 = r[:, 0], r[:, 1], r[:, 2]
    z = np.stack((b[:, 0] * r0 + b[:, 3] * r1 + b[:, 4] * r2,
                  b[:, 3] * r0 + b[:, 1] * r1 + b[:, 5] * r2,
                  b[:, 4] * r0 + b[:, 5] * r1 + b[:, 2] * r2), axis=1)
    out[...] = z.reshape(-1)


def _window_values_host(full, window, dtype):
    return full[window].astype(dtype)


def _elastic_load_host(out, applied_load, lame_lambda, mu, alpha_xx, alpha_yy, alpha_zz, rise, rise_ext, body_x, body_y,
                       body_z, applied_body, wx, wy, wz, ex, ey, ez, ox, oy, oz, ny_full, nz_full):
    ex, ey, ez, ox, oy, oz = (int(value) for value in (ex, ey, ez, ox, oy, oz))
    dtype = out.dtype
    window = (slice(ox, ox + ex), slice(oy, oy + ey), slice(oz, oz + ez))
    solid, lam, shear = _element_lame_host(lame_lambda, mu, (ex, ey, ez))
    alphas = tuple(_window_values_host(alpha, window, dtype) for alpha in (alpha_xx, alpha_yy, alpha_zz))
    alpha_sum = alphas[0] + alphas[1] + alphas[2]
    temperature_change = _window_values_host(rise, window, dtype) + _window_values_host(rise_ext, window, dtype)
    applied = applied_body.reshape(3, ex, ey, ez)
    bodies = tuple(_window_values_host(body, window, dtype) + applied[axis]
                   for axis, body in enumerate((body_x, body_y, body_z)))
    a_w, b_w, c_w = wx[:, None, None], wy[None, :, None], wz[None, None, :]
    areas = (b_w * c_w, a_w * c_w, a_w * b_w)
    eighth_volume = 0.125 * a_w * b_w * c_w
    thermal = tuple(np.where(solid, 0.25 * (2.0 * shear * alphas[axis] + lam * alpha_sum) * temperature_change
                             * areas[axis], 0) for axis in range(3))
    body = tuple(np.where(solid, eighth_volume * bodies[axis], 0) for axis in range(3))
    load = applied_load.reshape(ex + 1, ey + 1, ez + 1, 3).copy()
    for corner in _CORNERS:   # lower-side nodes of an element take -, upper-side + of the thermal term
        target = load[corner[0]:corner[0] + ex, corner[1]:corner[1] + ey, corner[2]:corner[2] + ez]
        for axis in range(3):
            target[..., axis] += (1.0 if corner[axis] else -1.0) * thermal[axis] + body[axis]
    out[...] = load.reshape(-1)


def _elastic_outputs_host(u, lame_lambda, mu, alpha_xx, alpha_yy, alpha_zz, rise, rise_ext, wx, wy, wz, *rest):
    outputs = rest[:len(OUTPUT_ARRAYS_3D)]
    thermal, ex, ey, ez, ox, oy, oz, _, _ = (int(value) for value in rest[len(OUTPUT_ARRAYS_3D):])
    dtype = u.dtype
    nodes = u.reshape(ex + 1, ey + 1, ez + 1, 3)
    window = (slice(ox, ox + ex), slice(oy, oy + ey), slice(oz, oz + ez))
    solid, lam, shear = _element_lame_host(lame_lambda, mu, (ex, ey, ez))
    widths = (wx, wy, wz)
    centre = nodes
    for axis in range(3):
        centre = _interpolate(centre, axis, 0.5)   # 8-corner mean
    differences = [(_upper(nodes, axis) - _lower(nodes, axis)) / _along(widths[axis], axis, 4) for axis in range(3)]
    gx, gy, gz = _gradients_at(differences, (0.5, 0.5, 0.5))   # mean of the 4 edge differences
    eps_xx, eps_yy, eps_zz = gx[..., 0], gy[..., 1], gz[..., 2]
    eps_xy = 0.5 * (gy[..., 0] + gx[..., 1])   # tensor shear strains
    eps_xz = 0.5 * (gz[..., 0] + gx[..., 2])
    eps_yz = 0.5 * (gz[..., 1] + gy[..., 2])
    alphas = tuple(_window_values_host(alpha, window, dtype) for alpha in (alpha_xx, alpha_yy, alpha_zz))
    alpha_sum = alphas[0] + alphas[1] + alphas[2]
    temperature_change = (_window_values_host(rise, window, dtype) + _window_values_host(rise_ext, window, dtype)
                          if thermal else dtype.type(0))   # 0: a mode shape
    volumetric = lam * (eps_xx + eps_yy + eps_zz)
    s_xx, s_yy, s_zz = (volumetric + 2.0 * shear * strain - (2.0 * shear * alpha + lam * alpha_sum) * temperature_change
                        for strain, alpha in zip((eps_xx, eps_yy, eps_zz), alphas))
    s_xy, s_xz, s_yz = 2.0 * shear * eps_xy, 2.0 * shear * eps_xz, 2.0 * shear * eps_yz
    mises = np.sqrt(0.5 * ((s_xx - s_yy) ** 2 + (s_yy - s_zz) ** 2 + (s_zz - s_xx) ** 2)
                    + 3.0 * (s_xy ** 2 + s_xz ** 2 + s_yz ** 2))
    values = (centre[..., 0], centre[..., 1], centre[..., 2],
              np.sqrt(centre[..., 0] ** 2 + centre[..., 1] ** 2 + centre[..., 2] ** 2),
              eps_xx, eps_yy, eps_zz, eps_xy, eps_xz, eps_yz,
              eps_xx - alphas[0] * temperature_change, eps_yy - alphas[1] * temperature_change,
              eps_zz - alphas[2] * temperature_change,
              s_xx, s_yy, s_zz, s_xy, s_xz, s_yz, mises)
    for output, value in zip(outputs, values):
        output[window] = np.where(solid, value, 0).astype(np.float32)


_HOST_TWINS = {'elastic_product': _elastic_product_host, 'elastic_mass_product': _elastic_mass_product_host,
               'elastic_diagonal_blocks': _elastic_diagonal_blocks_host,
               'reduce_invert_blocks': _reduce_invert_blocks_host, 'block_precondition3': _block_precondition3_host,
               'elastic_load': _elastic_load_host, 'elastic_outputs': _elastic_outputs_host}
# name -> numba factory: the stiffness product only (once per CG iteration; the rest runs per assembly / solve)
_NUMBA_TWINS = {'elastic_product': _elastic_product_numba}
# name -> core.kernels twin: the cupy RawKernel (per real type) for cupy arguments, the numba / numpy twin for numpy ones
_TWINS = {name: raw_twin(f'mechanics3d.{name}', _gpu_launcher(name), host, numba=_NUMBA_TWINS.get(name))
          for name, host in _HOST_TWINS.items()}


class BrickKernels:
    """Launchers of the 3D elasticity kernels for one window (element shape, origin in the full grid) and solve dtype;
    every array argument is a C-contiguous array of one backend (cupy: the RawKernel, numpy: its twin; full-grid inputs /
    outputs float32)."""

    def __init__(self, dtype, element_shape, window_origin, grid_shape):
        self.dtype = np.dtype(dtype)
        self.grid_shape = tuple(int(n) for n in grid_shape)
        ex, ey, ez = (int(n) for n in element_shape)
        self.n_nodes = (ex + 1) * (ey + 1) * (ez + 1)
        self.n_elements = ex * ey * ez
        self._element_args = tuple(np.int32(n) for n in (ex, ey, ez))
        self._window_args = tuple(np.int32(n) for n in (*window_origin, self.grid_shape[1], self.grid_shape[2]))
        self._n_nodes_i32 = np.int32(self.n_nodes)
        self._node_blocks = ((self.n_nodes + _BLOCK - 1) // _BLOCK,)
        self._element_blocks = ((self.n_elements + _BLOCK - 1) // _BLOCK,)

    def _launch(self, name, blocks, args):
        _TWINS[name](blocks, (_BLOCK,), args)

    def _full_grid(self, array, label):
        if array.dtype != np.float32 or tuple(array.shape) != self.grid_shape or not array.flags.c_contiguous:
            raise ValueError(f'{label} must be a C-contiguous float32 full-grid {self.grid_shape} array, got '
                             f'{array.dtype} {tuple(array.shape)}')
        return array

    def product(self, u, out, lame_lambda, mu, widths, free_dofs, masked: bool):
        """out = K u (masked: 0 on the rows of fixed dofs, i.e. A_solve u for u = 0 on them)."""
        self._launch('elastic_product', self._node_blocks,
                     (u, out, lame_lambda, mu, *widths, free_dofs, _MASK_ON if masked else _MASK_OFF,
                      *self._element_args))

    def mass_product(self, u, out, density, widths, free_dofs):
        """out = M u of the consistent mass of the element densities [kg/m^3] (<= 0: void), 0 on fixed rows."""
        self._launch('elastic_mass_product', self._node_blocks,
                     (u, out, density, *widths, free_dofs, *self._element_args))

    def diagonal_blocks(self, blocks, lame_lambda, mu, widths):
        """blocks (n_nodes, 6) = each node's own 3 x 3 stiffness block (xx, yy, zz, xy, xz, yz)."""
        self._launch('elastic_diagonal_blocks', self._node_blocks,
                     (blocks, lame_lambda, mu, *widths, *self._element_args))

    def reduce_invert_blocks(self, blocks, free_dofs):
        """In place: reduce each block to the free dofs (1 on fixed diagonals) and invert it."""
        self._launch('reduce_invert_blocks', self._node_blocks, (blocks, free_dofs, self._n_nodes_i32))

    def precondition(self, inverse_blocks, residual, out):
        self._launch('block_precondition3', self._node_blocks, (inverse_blocks, residual, out, self._n_nodes_i32))

    def load(self, out, applied_load, lame_lambda, mu, alphas, temperature_rise, temperature_rise_ext, body_forces,
             applied_body_force, widths):
        """out = applied nodal load + thermal load + body forces (coupled full-grid + applied per element) [N]."""
        full = [self._full_grid(array, label) for array, label in zip(
            (*alphas, temperature_rise, temperature_rise_ext, *body_forces),
            ('alpha_xx', 'alpha_yy', 'alpha_zz', 'temperature_rise', 'temperature_rise_ext',
             'body_force_x', 'body_force_y', 'body_force_z'))]
        self._launch('elastic_load', self._node_blocks,
                     (out, applied_load, lame_lambda, mu, *full, applied_body_force, *widths,
                      *self._element_args, *self._window_args))

    def outputs(self, output_arrays, unknown, lame_lambda, mu, alphas, temperature_rise, temperature_rise_ext,
                widths, thermal=True):
        """Write the 20 cell-centre outputs (full-grid float32, OUTPUT_ARRAYS_3D order) of the window elements;
        thermal False: dT = 0 (a mode shape)."""
        if len(output_arrays) != len(OUTPUT_ARRAYS_3D):
            raise ValueError(f'need {len(OUTPUT_ARRAYS_3D)} output arrays ({", ".join(OUTPUT_ARRAYS_3D)}), got '
                             f'{len(output_arrays)}')
        outputs = [self._full_grid(array, name) for array, name in zip(output_arrays, OUTPUT_ARRAYS_3D)]
        inputs = [self._full_grid(array, label) for array, label in zip(
            (*alphas, temperature_rise, temperature_rise_ext),
            ('alpha_xx', 'alpha_yy', 'alpha_zz', 'temperature_rise', 'temperature_rise_ext'))]
        self._launch('elastic_outputs', self._element_blocks,
                     (unknown, lame_lambda, mu, *inputs, *widths, *outputs, _MASK_ON if thermal else _MASK_OFF,
                      *self._element_args, *self._window_args))
