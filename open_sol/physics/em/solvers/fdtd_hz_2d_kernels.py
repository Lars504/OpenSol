"""Fused kernels of the 2D H_z-polarisation Yee engine (H_z at cell centres, E_x on hi-y faces, E_y on hi-x faces);
the thread mapping, launch config, H-coefficient and pole E-correction kernels are shared with the E_z engine, the
face-weighted Drude J / Lorentz P updates are H_z's own. Each kernel is a core.kernels twin (numpy twin of the same
call; numba twins for the two Yee updates): a change to a cupy source changes its twin. Stepper precisions as the E_z
kernels (fdtd_ez_2d_kernels header): real_t / T = the field dtype, materials and q_accum float32."""
import numpy as np

from ....core.backend import cp
from ....core.kernels import real_raw_twin, twin

from .fdtd_ez_2d_kernels import launch, h_coefficients, lorentz_e_correction, drude_e_correction, q_accum_je_half
from ..yee_grid import face_partner


__all__ = [
    'launch', 'h_coefficients', 'lorentz_p_update', 'lorentz_e_correction',
    'update_hz_kernel', 'update_exy_kernel', 'e_face_coefficients', 'drude_j_update', 'drude_e_correction',
    'drude_j_update_power', 'lorentz_p_update_power', 'q_accum_je_half', 'q_accum_hz',
]


# mu dHz/dt = dEx/dy - dEy/dx - sigma_m Hz at the centre of cell (i, j): E_x[i, j] / E_x[i, j-1] sit on its hi / lo y
# faces (spacing dy_j -> inv_dy_e[j]), E_y[i, j] / E_y[i-1, j] on its hi / lo x faces (inv_dx_e[i]). The j = 0 / i = 0
# lo-face samples wrap to E_x[i, ny-1] / E_y[nx-1, j] on periodic axes and read 0 otherwise (tangential E = 0 half a
# cell beyond the outermost centres, PEC, like the never-updated last face row / column). da / db from h_coefficients;
# CPML kappa is folded into inv_d*_e. Bloch wraps (lo samples): this * wrap_cos - partner * wrap_sin, as the E_z
# kernels (fdtd_ez_2d_kernels header; a real-only engine passes partner = this, cos 1, sin 0).
_UPDATE_HZ_SOURCE = r"""
extern "C" __global__
void update_hz_kernel(
    real_t* __restrict__ hz,
    const real_t* __restrict__ ex,
    const real_t* __restrict__ ey,
    const real_t* __restrict__ da,
    const real_t* __restrict__ db,
    const real_t* __restrict__ inv_dx_e,
    const real_t* __restrict__ inv_dy_e,
    int nx, int ny, int periodic_x, int periodic_y,
    const real_t* __restrict__ ex_partner,
    const real_t* __restrict__ ey_partner,
    real_t wrap_cos_x, real_t wrap_sin_x, real_t wrap_cos_y, real_t wrap_sin_y)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    real_t curl = 0;
    if (j >= 1)          curl += (ex[idx] - ex[idx - 1])             * inv_dy_e[j];
    else if (periodic_y) {
        int wrapped = idx + ny - 1;
        curl += (ex[idx] - (ex[wrapped] * wrap_cos_y - ex_partner[wrapped] * wrap_sin_y)) * inv_dy_e[j];
    }
    else                 curl += ex[idx]                             * inv_dy_e[j];
    if (i >= 1)          curl -= (ey[idx] - ey[idx - ny])            * inv_dx_e[i];
    else if (periodic_x) {
        int wrapped = (nx - 1) * ny + j;
        curl -= (ey[idx] - (ey[wrapped] * wrap_cos_x - ey_partner[wrapped] * wrap_sin_x)) * inv_dx_e[i];
    }
    else                 curl -= ey[idx]                             * inv_dx_e[i];
    hz[idx] = da[idx] * hz[idx] + db[idx] * curl;
}
"""


def _update_hz_host(hz, ex, ey, da, db, inv_dx_e, inv_dy_e, nx, ny, periodic_x, periodic_y, ex_partner, ey_partner,
                    wrap_cos_x, wrap_sin_x, wrap_cos_y, wrap_sin_y):
    curl = np.empty_like(hz)
    curl[:, 1:] = (ex[:, 1:] - ex[:, :-1]) * inv_dy_e[1:]
    if periodic_y:
        curl[:, 0] = (ex[:, 0] - (ex[:, -1] * wrap_cos_y - ex_partner[:, -1] * wrap_sin_y)) * inv_dy_e[0]
    else:
        curl[:, 0] = ex[:, 0] * inv_dy_e[0]
    curl[1:] -= (ey[1:] - ey[:-1]) * inv_dx_e[1:, None]
    if periodic_x:
        curl[0] -= (ey[0] - (ey[-1] * wrap_cos_x - ey_partner[-1] * wrap_sin_x)) * inv_dx_e[0]
    else:
        curl[0] -= ey[0] * inv_dx_e[0]
    hz *= da
    curl *= db
    hz += curl


def _update_hz_numba(numba):
    @numba.njit(parallel=True, cache=True)
    def kernel(hz, ex, ey, da, db, inv_dx_e, inv_dy_e, nx, ny, periodic_x, periodic_y, ex_partner, ey_partner,
               wrap_cos_x, wrap_sin_x, wrap_cos_y, wrap_sin_y):
        for prange_i in numba.prange(nx):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            for j in range(ny):
                if j >= 1:
                    curl = (ex[i, j] - ex[i, j - 1]) * inv_dy_e[j]
                elif periodic_y:
                    curl = (ex[i, j] - (ex[i, ny - 1] * wrap_cos_y - ex_partner[i, ny - 1] * wrap_sin_y)) * inv_dy_e[j]
                else:
                    curl = ex[i, j] * inv_dy_e[j]
                if i >= 1:
                    curl -= (ey[i, j] - ey[i - 1, j]) * inv_dx_e[i]
                elif periodic_x:
                    curl -= (ey[i, j] - (ey[nx - 1, j] * wrap_cos_x - ey_partner[nx - 1, j] * wrap_sin_x)) * inv_dx_e[i]
                else:
                    curl -= ey[i, j] * inv_dx_e[i]
                hz[i, j] = da[i, j] * hz[i, j] + db[i, j] * curl
    return kernel


update_hz_kernel = real_raw_twin('em2d.update_hz', _UPDATE_HZ_SOURCE, 'update_hz_kernel', _update_hz_host,
                                 numba=_update_hz_numba)


# eps_x dEx/dt = dHz/dy - sigma_x Ex on the hi-y face of cell (i, j): (H_z[i, j+1] - H_z[i, j]) inv_dy_h[j].
# eps_y dEy/dt = -dHz/dx - sigma_y Ey on the hi-x face: -(H_z[i+1, j] - H_z[i, j]) inv_dx_h[i].
# The last face row / column wraps to H_z of cell 0 on periodic axes (inv_d*_h[n-1] = 1 / wrap spacing) and is never
# updated otherwise (stays 0). ca_* / cb_* from e_face_coefficients; CPML kappa is folded into inv_d*_h. Bloch wraps
# (hi samples): this * wrap_cos + partner * wrap_sin.
_UPDATE_EXY_SOURCE = r"""
extern "C" __global__
void update_exy_kernel(
    real_t* __restrict__ ex,
    real_t* __restrict__ ey,
    const real_t* __restrict__ hz,
    const real_t* __restrict__ ca_x,
    const real_t* __restrict__ cb_x,
    const real_t* __restrict__ ca_y,
    const real_t* __restrict__ cb_y,
    const real_t* __restrict__ inv_dx_h,
    const real_t* __restrict__ inv_dy_h,
    int nx, int ny, int periodic_x, int periodic_y,
    const real_t* __restrict__ hz_partner,
    real_t wrap_cos_x, real_t wrap_sin_x, real_t wrap_cos_y, real_t wrap_sin_y)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    real_t h = hz[idx];
    if (j < ny - 1)      ex[idx] = ca_x[idx] * ex[idx] + cb_x[idx] * (hz[idx + 1]  - h) * inv_dy_h[j];
    else if (periodic_y) ex[idx] = ca_x[idx] * ex[idx] + cb_x[idx]
                                   * (hz[i * ny] * wrap_cos_y + hz_partner[i * ny] * wrap_sin_y - h) * inv_dy_h[j];
    if (i < nx - 1)      ey[idx] = ca_y[idx] * ey[idx] - cb_y[idx] * (hz[idx + ny] - h) * inv_dx_h[i];
    else if (periodic_x) ey[idx] = ca_y[idx] * ey[idx] - cb_y[idx]
                                   * (hz[j] * wrap_cos_x + hz_partner[j] * wrap_sin_x - h) * inv_dx_h[i];
}
"""


def _update_exy_host(ex, ey, hz, ca_x, cb_x, ca_y, cb_y, inv_dx_h, inv_dy_h, nx, ny, periodic_x, periodic_y,
                     hz_partner, wrap_cos_x, wrap_sin_x, wrap_cos_y, wrap_sin_y):
    ex[:, :-1] = ca_x[:, :-1] * ex[:, :-1] + cb_x[:, :-1] * (hz[:, 1:] - hz[:, :-1]) * inv_dy_h[:-1]
    if periodic_y:
        ex[:, -1] = ca_x[:, -1] * ex[:, -1] + cb_x[:, -1] * (
            hz[:, 0] * wrap_cos_y + hz_partner[:, 0] * wrap_sin_y - hz[:, -1]) * inv_dy_h[-1]
    ey[:-1] = ca_y[:-1] * ey[:-1] - cb_y[:-1] * (hz[1:] - hz[:-1]) * inv_dx_h[:-1, None]
    if periodic_x:
        ey[-1] = ca_y[-1] * ey[-1] - cb_y[-1] * (
            hz[0] * wrap_cos_x + hz_partner[0] * wrap_sin_x - hz[-1]) * inv_dx_h[-1]


def _update_exy_numba(numba):
    @numba.njit(parallel=True, cache=True)
    def kernel(ex, ey, hz, ca_x, cb_x, ca_y, cb_y, inv_dx_h, inv_dy_h, nx, ny, periodic_x, periodic_y,
               hz_partner, wrap_cos_x, wrap_sin_x, wrap_cos_y, wrap_sin_y):
        for prange_i in numba.prange(nx):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            for j in range(ny):
                h = hz[i, j]
                if j < ny - 1:
                    ex[i, j] = ca_x[i, j] * ex[i, j] + cb_x[i, j] * (hz[i, j + 1] - h) * inv_dy_h[j]
                elif periodic_y:
                    ex[i, j] = ca_x[i, j] * ex[i, j] + cb_x[i, j] * (
                        hz[i, 0] * wrap_cos_y + hz_partner[i, 0] * wrap_sin_y - h) * inv_dy_h[j]
                if i < nx - 1:
                    ey[i, j] = ca_y[i, j] * ey[i, j] - cb_y[i, j] * (hz[i + 1, j] - h) * inv_dx_h[i]
                elif periodic_x:
                    ey[i, j] = ca_y[i, j] * ey[i, j] - cb_y[i, j] * (
                        hz[0, j] * wrap_cos_x + hz_partner[0, j] * wrap_sin_x - h) * inv_dx_h[i]
    return kernel


update_exy_kernel = real_raw_twin('em2d.update_exy', _UPDATE_EXY_SOURCE, 'update_exy_kernel', _update_exy_host,
                                  numba=_update_exy_numba)


# Both E face sets in one launch, no temporaries. Face eps_rel / sigma = yee_grid.face_average of the cell arrays:
# a_self lower + a_partner upper with the yee_grid.face_shares weights, products and sum rounded separately (no FMA)
# like face_average's cupy ops (em_operators_consistency compares the two). E_x[i, j] pairs cell (i, j) with (i, j+1),
# E_y[i, j] with (i+1, j); the last column / row pairs with cell 0 on periodic axes, else with itself. Then the
# semi-implicit coefficients of eps dE/dt = curl H - sigma E - J: eps = eps_rel eps0 [F/m], k = sigma dt / (2 eps),
# ca = (1 - k) / (1 + k), cb = (dt / eps) / (1 + k) (also the Drude correction factor). Materials are float32, the face
# shares, coefficients, dt and eps0 real_t (the face mean in real_t, rounded like face_average in that dtype).
_E_FACE_COEFFICIENTS_SOURCE = r"""
__device__ __forceinline__ real_t face_value(real_t self_value, real_t partner_value, real_t lower_share,
                                             real_t upper_share)
{
#if REAL_IS_DOUBLE
    return __dadd_rn(__dmul_rn(self_value, lower_share), __dmul_rn(partner_value, upper_share));
#else
    return __fadd_rn(__fmul_rn(self_value, lower_share), __fmul_rn(partner_value, upper_share));
#endif
}

__device__ __forceinline__ void store_face_coefficients(
    real_t eps_rel, real_t sigma, real_t dt, real_t eps0, real_t* ca, real_t* cb, real_t* eps_abs, int idx)
{
    real_t eps = eps_rel * eps0;
    real_t k = sigma * dt / ((real_t)2 * eps);
    real_t denom = (real_t)1 + k;
    ca[idx] = ((real_t)1 - k) / denom;
    cb[idx] = (dt / eps) / denom;
    eps_abs[idx] = eps;
}

extern "C" __global__
void e_face_coefficients(
    real_t* __restrict__ ca_x,
    real_t* __restrict__ cb_x,
    real_t* __restrict__ eps_x,
    real_t* __restrict__ ca_y,
    real_t* __restrict__ cb_y,
    real_t* __restrict__ eps_y,
    const float* __restrict__ eps_xx,
    const float* __restrict__ eps_yy,
    const float* __restrict__ sigma_e,
    const real_t* __restrict__ lower_share_y,
    const real_t* __restrict__ upper_share_y,
    const real_t* __restrict__ lower_share_x,
    const real_t* __restrict__ upper_share_x,
    int nx, int ny, int periodic_x, int periodic_y, real_t dt, real_t eps0)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    int partner_j = (j < ny - 1) ? j + 1 : (periodic_y ? 0 : j);
    int partner_x = i * ny + partner_j;
    store_face_coefficients(
        face_value(eps_xx[idx], eps_xx[partner_x], lower_share_y[j], upper_share_y[j]),
        face_value(sigma_e[idx], sigma_e[partner_x], lower_share_y[j], upper_share_y[j]),
        dt, eps0, ca_x, cb_x, eps_x, idx);
    int partner_i = (i < nx - 1) ? i + 1 : (periodic_x ? 0 : i);
    int partner_y = partner_i * ny + j;
    store_face_coefficients(
        face_value(eps_yy[idx], eps_yy[partner_y], lower_share_x[i], upper_share_x[i]),
        face_value(sigma_e[idx], sigma_e[partner_y], lower_share_x[i], upper_share_x[i]),
        dt, eps0, ca_y, cb_y, eps_y, idx);
}
"""


def _partner_index(n, periodic):
    """Upper partner cell of each hi face along an axis of n cells (yee_grid.face_partner of the indices)."""
    return face_partner(np.arange(int(n)), 0, periodic)


def _store_face_coefficients_host(eps_rel, sigma, dt, eps0, ca, cb, eps_abs):
    eps = eps_rel * eps0
    k = sigma * dt / (2.0 * eps)
    denom = 1.0 + k
    ca[...] = (1.0 - k) / denom
    cb[...] = (dt / eps) / denom
    eps_abs[...] = eps


def _e_face_coefficients_host(ca_x, cb_x, eps_x, ca_y, cb_y, eps_y, eps_xx, eps_yy, sigma_e, lower_share_y,
                              upper_share_y, lower_share_x, upper_share_x, nx, ny, periodic_x, periodic_y, dt, eps0):
    partner_j = _partner_index(ny, periodic_y)
    partner_i = _partner_index(nx, periodic_x)
    eps_xx, eps_yy, sigma_e = (array.astype(ca_x.dtype, copy=False) for array in (eps_xx, eps_yy, sigma_e))
    _store_face_coefficients_host(eps_xx * lower_share_y + eps_xx[:, partner_j] * upper_share_y,
                                  sigma_e * lower_share_y + sigma_e[:, partner_j] * upper_share_y,
                                  dt, eps0, ca_x, cb_x, eps_x)
    lower_x, upper_x = lower_share_x[:, None], upper_share_x[:, None]
    _store_face_coefficients_host(eps_yy * lower_x + eps_yy[partner_i] * upper_x,
                                  sigma_e * lower_x + sigma_e[partner_i] * upper_x,
                                  dt, eps0, ca_y, cb_y, eps_y)


e_face_coefficients = real_raw_twin('em2d.e_face_coefficients', _E_FACE_COEFFICIENTS_SOURCE, 'e_face_coefficients',
                                    _e_face_coefficients_host)


# Drude ADE with J on the half steps (derivation, velocity form and the shared drude_e_correction in
# fdtd_ez_2d_kernels): J^{n+1/2} = J^{n-1/2} + damping J^{n-1/2} + drive E^n from the pre-update face E, then E^{n+1} =
# E_pred - cb J^{n+1/2} after the Yee update; the region's face weight (PEC keep folded in, shared by every pole on the
# region) scales the scalar drive.
def _drude_j_update_host(damping, drive, weight, e, current):
    current += damping * current + drive * weight * e


drude_j_update = twin('em2d.hz_drude_j_update', cp.ElementwiseKernel(
    'T damping, T drive, T weight, T e',
    'T current',
    'current += damping * current + drive * weight * e',
    'hz_drude_j_update'), _drude_j_update_host)


# Lorentz velocity form of fdtd_ez_2d_kernels.lorentz_p_update with the scalar drive scaled by the region's face weight
# (weight 0 keeps p and dp pinned at 0).
def _lorentz_p_update_host(damping, restoring, drive, weight, e, p, dp):
    dp += damping * dp - restoring * p + drive * weight * e
    p += dp


lorentz_p_update = twin('em2d.hz_lorentz_p_update', cp.ElementwiseKernel(
    'T damping, T restoring, T drive, T weight, T e',
    'T p, T dp',
    '''
    dp += damping * dp - restoring * p + drive * weight * e;
    p += dp;
    ''',
    'hz_lorentz_p_update'), _lorentz_p_update_host)


# Face pole power density [W/m^3] on a pole box, added to the full-grid face totals q_accum_hz splits, as two halves of
# the time-centred work (E^n half fused into the pole update below, E^{n+1} half via the shared q_accum_je_half after
# every E correction): Drude J^{n+1/2} (E^n + E^{n+1}) / 2 (scale 1), Lorentz (P^{n+1} - P^n) / dt (E^n + E^{n+1}) / 2
# (scale 1 / dt, dp from lorentz_p_update). Pairing with E^{n+1} alone biases the loss (see fdtd_ez_2d_kernels).
# Fused pole update + E^n power half (one launch instead of two while accumulating), the same arithmetic as
# q_accum_je_half (half_inv_dt = 0.5 / dt).
def _drude_j_update_power_host(damping, drive, weight, e, current, power):
    _drude_j_update_host(damping, drive, weight, e, current)
    power += 0.5 * current * e


drude_j_update_power = twin('em2d.hz_drude_j_update_power', cp.ElementwiseKernel(
    'T damping, T drive, T weight, T e',
    'T current, T power',
    '''
    current += damping * current + drive * weight * e;
    power += (T)0.5 * current * e;
    ''',
    'hz_drude_j_update_power'), _drude_j_update_power_host)


def _lorentz_p_update_power_host(damping, restoring, drive, weight, half_inv_dt, e, p, dp, power):
    _lorentz_p_update_host(damping, restoring, drive, weight, e, p, dp)
    power += dp * half_inv_dt * e


lorentz_p_update_power = twin('em2d.hz_lorentz_p_update_power', cp.ElementwiseKernel(
    'T damping, T restoring, T drive, T weight, T half_inv_dt, T e',
    'T p, T dp, T power',
    '''
    dp += damping * dp - restoring * p + drive * weight * e;
    p += dp;
    power += dp * half_inv_dt * e;
    ''',
    'hz_lorentz_p_update_power'), _lorentz_p_update_power_host)


# Loss accounting q += dt * (cell share of sigma_f E_f^2 + P_f) over the up to four E faces of cell (i, j), P_f = the
# face pole power total: X_hi = ex[i, j] (partner cell (i, j+1)), X_lo = ex[i, j-1], Y_hi = ey[i, j] (partner
# (i+1, j)), Y_lo = ey[i-1, j]; lo faces wrap on periodic axes and are absent at index 0 otherwise. The face's dual
# cell covers w_self / 2 of each adjacent cell (h_f = (w_self + w_partner) / 2 across the face), and sigma_f / pole
# weights are the matching width-weighted face_average, so its ohmic loss splits exactly as 0.5 sigma_cell E_f^2 per
# cell; its pole power density P_f gives the cell m_self h_f / (m_self w_self + m_partner w_partner) (pole membership m;
# 0.5 when neither cell has a pole), i.e. the face energy by pole-covered dual area. A non-periodic last face (its own
# partner, the cell itself) goes wholly to the cell, so every face's loss is assigned exactly once. has_pole_power = 0:
# no poles (pole_power_x / pole_power_y / pole_cell are dummies and never read). cell_dx / cell_dy = cell widths [m].
# q float32 (q_accum); fields, pole power totals, widths and dt real_t (dtype from ex: argument 1). Runs on the cell
# box (i0, j0, box_nx, box_ny) that can receive loss (FDTDEngine2D_Hz._loss_box_launch); the twin adds on it alone.
_Q_ACCUM_HZ_SOURCE = r"""
__device__ __forceinline__ real_t pole_share(
    const bool* pole_cell, int self_idx, int partner_idx, real_t w_self, real_t w_partner)
{
    real_t m_self = pole_cell[self_idx] ? (real_t)1 : (real_t)0;
    real_t m_partner = pole_cell[partner_idx] ? (real_t)1 : (real_t)0;
    real_t covered_width = m_self * w_self + m_partner * w_partner;
    return (covered_width > (real_t)0) ? m_self * (real_t)0.5 * (w_self + w_partner) / covered_width : (real_t)0.5;
}

// Power density [W/m^3] one face assigns to cell self_idx; own_face = non-periodic last face (partner = the cell);
// w_self / w_partner = the two cells' widths across the face.
__device__ __forceinline__ real_t face_loss(
    real_t e, const real_t* pole_power, int face_idx, real_t half_sigma,
    const bool* pole_cell, int self_idx, int partner_idx, real_t w_self, real_t w_partner, bool own_face,
    int has_pole_power)
{
    real_t loss = (own_face ? (real_t)2 : (real_t)1) * half_sigma * e * e;
    if (has_pole_power) {
        real_t share = own_face ? (real_t)1 : pole_share(pole_cell, self_idx, partner_idx, w_self, w_partner);
        loss += share * pole_power[face_idx];
    }
    return loss;
}

extern "C" __global__
void q_accum_hz(
    float* __restrict__ q,
    const real_t* __restrict__ ex,
    const real_t* __restrict__ ey,
    const real_t* __restrict__ pole_power_x,
    const real_t* __restrict__ pole_power_y,
    const float* __restrict__ sigma_e,
    const bool* __restrict__ pole_cell,
    const real_t* __restrict__ cell_dx,
    const real_t* __restrict__ cell_dy,
    int nx, int ny, int periodic_x, int periodic_y, int has_pole_power, real_t dt,
    int i0, int j0, int box_nx, int box_ny)
{
    int box_j = blockIdx.x * blockDim.x + threadIdx.x;
    int box_i = blockIdx.y * blockDim.y + threadIdx.y;
    if (box_i >= box_nx || box_j >= box_ny) return;
    int i = i0 + box_i;
    int j = j0 + box_j;
    int idx = i * ny + j;
    real_t half_sigma = (real_t)0.5 * sigma_e[idx];
    real_t power = 0;

    // X_hi: ex[i, j], partner (i, j+1); the last column wraps to (i, 0) or is the cell's own face.
    bool own_x = (j == ny - 1) && !periodic_y;
    int partner_j = (j < ny - 1) ? j + 1 : (own_x ? j : 0);
    power += face_loss(ex[idx], pole_power_x, idx, half_sigma, pole_cell, idx, i * ny + partner_j,
                       cell_dy[j], cell_dy[partner_j], own_x, has_pole_power);
    // X_lo: ex[i, j-1] (wraps to ex[i, ny-1]); the face index equals its partner cell's index.
    if (j >= 1 || periodic_y) {
        int lo_j = (j >= 1) ? j - 1 : ny - 1;
        int lo_x = i * ny + lo_j;
        power += face_loss(ex[lo_x], pole_power_x, lo_x, half_sigma, pole_cell, idx, lo_x,
                           cell_dy[j], cell_dy[lo_j], false, has_pole_power);
    }
    // Y_hi: ey[i, j], partner (i+1, j); the last row wraps to (0, j) or is the cell's own face.
    bool own_y = (i == nx - 1) && !periodic_x;
    int partner_i = (i < nx - 1) ? i + 1 : (own_y ? i : 0);
    power += face_loss(ey[idx], pole_power_y, idx, half_sigma, pole_cell, idx, partner_i * ny + j,
                       cell_dx[i], cell_dx[partner_i], own_y, has_pole_power);
    // Y_lo: ey[i-1, j] (wraps to ey[nx-1, j]).
    if (i >= 1 || periodic_x) {
        int lo_i = (i >= 1) ? i - 1 : nx - 1;
        int lo_y = lo_i * ny + j;
        power += face_loss(ey[lo_y], pole_power_y, lo_y, half_sigma, pole_cell, idx, lo_y,
                           cell_dx[i], cell_dx[lo_i], false, has_pole_power);
    }
    q[idx] += (float)(power * dt);
}
"""


def _pole_share_host(self_member, partner_member, self_width, partner_width):
    """pole_share of the cuda source in the widths' dtype: m_self (w_self + w_partner) / 2 / covered width, 0.5 when
    uncovered."""
    real = self_width.dtype.type
    m_self = self_member.astype(self_width.dtype)
    m_partner = partner_member.astype(self_width.dtype)
    covered = m_self * self_width + m_partner * partner_width
    safe = np.where(covered > 0.0, covered, real(1.0))
    return np.where(covered > 0.0, m_self * 0.5 * (self_width + partner_width) / safe, real(0.5))


def _q_accum_hz_host(q, ex, ey, pole_power_x, pole_power_y, sigma_e, pole_cell, cell_dx, cell_dy, nx, ny,
                     periodic_x, periodic_y, has_pole_power, dt, i0, j0, box_nx, box_ny):
    nx, ny = int(nx), int(ny)
    half_sigma = 0.5 * sigma_e.astype(ex.dtype, copy=False)
    dx_rows, dy_columns = cell_dx[:, None], cell_dy[None, :]
    # hi faces: partner j + 1 / i + 1; the non-periodic last face is the cell's own (factor 2, share 1)
    partner_j, partner_i = _partner_index(ny, periodic_y), _partner_index(nx, periodic_x)
    own_x = np.zeros(ny, dtype=bool)
    own_y = np.zeros((nx, 1), dtype=bool)
    if not periodic_y:
        own_x[-1] = True
    if not periodic_x:
        own_y[-1] = True
    power = np.where(own_x, np.float32(2.0), np.float32(1.0)) * half_sigma * ex * ex
    power += np.where(own_y, np.float32(2.0), np.float32(1.0)) * half_sigma * ey * ey
    # lo faces ex[:, j - 1] / ey[i - 1]: column / row 0 wraps on a periodic axis, else has none
    lo_ex = np.roll(ex, 1, axis=1)
    lo_ey = np.roll(ey, 1, axis=0)
    lo_x_valid = np.ones(ny, dtype=bool)
    lo_y_valid = np.ones((nx, 1), dtype=bool)
    if not periodic_y:
        lo_x_valid[0] = False
    if not periodic_x:
        lo_y_valid[0] = False
    power += np.where(lo_x_valid, half_sigma * lo_ex * lo_ex, np.float32(0.0))
    power += np.where(lo_y_valid, half_sigma * lo_ey * lo_ey, np.float32(0.0))
    if has_pole_power:
        lo_cell_x = np.roll(pole_cell, 1, axis=1)
        lo_cell_y = np.roll(pole_cell, 1, axis=0)
        share = np.where(own_x, np.float32(1.0),
                         _pole_share_host(pole_cell, pole_cell[:, partner_j], dy_columns, cell_dy[partner_j][None, :]))
        power += share * pole_power_x
        share = np.where(own_y, np.float32(1.0),
                         _pole_share_host(pole_cell, pole_cell[partner_i], dx_rows, cell_dx[partner_i][:, None]))
        power += share * pole_power_y
        lo_share = _pole_share_host(pole_cell, lo_cell_x, dy_columns, np.roll(cell_dy, 1)[None, :])
        power += np.where(lo_x_valid, lo_share * np.roll(pole_power_x, 1, axis=1), np.float32(0.0))
        lo_share = _pole_share_host(pole_cell, lo_cell_y, dx_rows, np.roll(cell_dx, 1)[:, None])
        power += np.where(lo_y_valid, lo_share * np.roll(pole_power_y, 1, axis=0), np.float32(0.0))
    box = (slice(int(i0), int(i0) + int(box_nx)), slice(int(j0), int(j0) + int(box_ny)))
    q[box] += (power * dt)[box]


q_accum_hz = real_raw_twin('em2d.q_accum_hz', _Q_ACCUM_HZ_SOURCE, 'q_accum_hz', _q_accum_hz_host, dtype_argument=1)
