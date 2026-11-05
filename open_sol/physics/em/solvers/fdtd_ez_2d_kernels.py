"""Fused kernels of the 2D E_z-polarisation Yee engine, each a core.kernels twin (cupy kernel + numpy twin of the same
call; numba twins for the two Yee updates): the numpy twins below every cupy source must follow any change to it.
Stepper precisions (explicit_kernels single / double): RawKernels are RealRawKernels (real_t compiled per field
dtype), ElementwiseKernels take the field dtype as T (materials stay float32, q_accum float32); scalars are passed in
the engine's real type."""
import numpy as np

from ....core.backend import cp
from ....core.kernels import real_raw_twin, twin


_BLOCK = (32, 8, 1)


# Periodic axes (periodic_x / periodic_y = 1) wrap the Yee stencil: the last H row/column differences
# Ez against cell 0 (inv_d*_h[n-1] = 1 / wrap spacing) and Ez of cell 0 against the last H row/column,
# so the period is exactly the nx (ny) cells of the grid. Non-periodic axes keep the last H row/column at 0 and read
# the missing lo face of cell 0 as 0: tangential H = 0 half a cell beyond the outermost centres (PMC) on both sides.
# Bloch phase (both engines, one field set per launch): a wrapped sample of the far edge is F e^{-j k L} across the hi
# edge and F e^{+j k L} across the lo edge; per real set that is this * wrap_cos + partner * wrap_sin (hi) and
# this * wrap_cos - partner * wrap_sin (lo), wrap_sin = +sin(k L) for the real set, -sin(k L) for the imaginary one.
# A real-only engine passes partner = this, cos 1, sin 0 (exact: the k = 0 results are unchanged bit for bit).
_UPDATE_H_SOURCE = r"""
extern "C" __global__
void update_h_kernel(
    const real_t* __restrict__ ez,
    real_t* __restrict__ hx,
    real_t* __restrict__ hy,
    const real_t* __restrict__ da,
    const real_t* __restrict__ db,
    const real_t* __restrict__ inv_dx_h,
    const real_t* __restrict__ inv_dy_h,
    int nx, int ny, int periodic_x, int periodic_y,
    const real_t* __restrict__ ez_partner,
    real_t wrap_cos_x, real_t wrap_sin_x, real_t wrap_cos_y, real_t wrap_sin_y)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    real_t e = ez[idx];
    if (j < ny - 1)      hx[idx] = da[idx] * hx[idx] - db[idx] * (ez[idx + 1]  - e) * inv_dy_h[j];
    else if (periodic_y) hx[idx] = da[idx] * hx[idx] - db[idx]
                                   * (ez[i * ny] * wrap_cos_y + ez_partner[i * ny] * wrap_sin_y - e) * inv_dy_h[j];
    if (i < nx - 1)      hy[idx] = da[idx] * hy[idx] + db[idx] * (ez[idx + ny] - e) * inv_dx_h[i];
    else if (periodic_x) hy[idx] = da[idx] * hy[idx] + db[idx]
                                   * (ez[j] * wrap_cos_x + ez_partner[j] * wrap_sin_x - e) * inv_dx_h[i];
}
"""


def _update_h_host(ez, hx, hy, da, db, inv_dx_h, inv_dy_h, nx, ny, periodic_x, periodic_y, ez_partner,
                   wrap_cos_x, wrap_sin_x, wrap_cos_y, wrap_sin_y):
    hx[:, :-1] = da[:, :-1] * hx[:, :-1] - db[:, :-1] * (ez[:, 1:] - ez[:, :-1]) * inv_dy_h[:-1]
    if periodic_y:
        hx[:, -1] = da[:, -1] * hx[:, -1] - db[:, -1] * (
            ez[:, 0] * wrap_cos_y + ez_partner[:, 0] * wrap_sin_y - ez[:, -1]) * inv_dy_h[-1]
    hy[:-1] = da[:-1] * hy[:-1] + db[:-1] * (ez[1:] - ez[:-1]) * inv_dx_h[:-1, None]
    if periodic_x:
        hy[-1] = da[-1] * hy[-1] + db[-1] * (ez[0] * wrap_cos_x + ez_partner[0] * wrap_sin_x - ez[-1]) * inv_dx_h[-1]


def _update_h_numba(numba):
    @numba.njit(parallel=True, cache=True)
    def kernel(ez, hx, hy, da, db, inv_dx_h, inv_dy_h, nx, ny, periodic_x, periodic_y, ez_partner,
               wrap_cos_x, wrap_sin_x, wrap_cos_y, wrap_sin_y):
        for prange_i in numba.prange(nx):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            for j in range(ny):
                e = ez[i, j]
                if j < ny - 1:
                    hx[i, j] = da[i, j] * hx[i, j] - db[i, j] * (ez[i, j + 1] - e) * inv_dy_h[j]
                elif periodic_y:
                    hx[i, j] = da[i, j] * hx[i, j] - db[i, j] * (
                        ez[i, 0] * wrap_cos_y + ez_partner[i, 0] * wrap_sin_y - e) * inv_dy_h[j]
                if i < nx - 1:
                    hy[i, j] = da[i, j] * hy[i, j] + db[i, j] * (ez[i + 1, j] - e) * inv_dx_h[i]
                elif periodic_x:
                    hy[i, j] = da[i, j] * hy[i, j] + db[i, j] * (
                        ez[0, j] * wrap_cos_x + ez_partner[0, j] * wrap_sin_x - e) * inv_dx_h[i]
    return kernel


update_h_kernel = real_raw_twin('em2d.update_h', _UPDATE_H_SOURCE, 'update_h_kernel', _update_h_host,
                                numba=_update_h_numba)


_UPDATE_E_SOURCE = r"""
extern "C" __global__
void update_e_kernel(
    real_t* __restrict__ ez,
    const real_t* __restrict__ hx,
    const real_t* __restrict__ hy,
    const real_t* __restrict__ ca,
    const real_t* __restrict__ cb,
    const real_t* __restrict__ inv_dx_e,
    const real_t* __restrict__ inv_dy_e,
    int nx, int ny, int periodic_x, int periodic_y,
    const real_t* __restrict__ hx_partner,
    const real_t* __restrict__ hy_partner,
    real_t wrap_cos_x, real_t wrap_sin_x, real_t wrap_cos_y, real_t wrap_sin_y)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    real_t curl = 0;
    if (i >= 1)          curl += (hy[idx] - hy[idx - ny])            * inv_dx_e[i];
    else if (periodic_x) {
        int wrapped = (nx - 1) * ny + j;
        curl += (hy[idx] - (hy[wrapped] * wrap_cos_x - hy_partner[wrapped] * wrap_sin_x)) * inv_dx_e[i];
    }
    else                 curl += hy[idx]                             * inv_dx_e[i];
    if (j >= 1)          curl -= (hx[idx] - hx[idx - 1])             * inv_dy_e[j];
    else if (periodic_y) {
        int wrapped = idx + ny - 1;
        curl -= (hx[idx] - (hx[wrapped] * wrap_cos_y - hx_partner[wrapped] * wrap_sin_y)) * inv_dy_e[j];
    }
    else                 curl -= hx[idx]                             * inv_dy_e[j];
    ez[idx] = ca[idx] * ez[idx] + cb[idx] * curl;
}
"""


def _update_e_host(ez, hx, hy, ca, cb, inv_dx_e, inv_dy_e, nx, ny, periodic_x, periodic_y, hx_partner, hy_partner,
                   wrap_cos_x, wrap_sin_x, wrap_cos_y, wrap_sin_y):
    curl = np.empty_like(ez)
    curl[1:] = (hy[1:] - hy[:-1]) * inv_dx_e[1:, None]
    if periodic_x:
        curl[0] = (hy[0] - (hy[-1] * wrap_cos_x - hy_partner[-1] * wrap_sin_x)) * inv_dx_e[0]
    else:
        curl[0] = hy[0] * inv_dx_e[0]
    curl[:, 1:] -= (hx[:, 1:] - hx[:, :-1]) * inv_dy_e[1:]
    if periodic_y:
        curl[:, 0] -= (hx[:, 0] - (hx[:, -1] * wrap_cos_y - hx_partner[:, -1] * wrap_sin_y)) * inv_dy_e[0]
    else:
        curl[:, 0] -= hx[:, 0] * inv_dy_e[0]
    ez *= ca
    curl *= cb
    ez += curl


def _update_e_numba(numba):
    @numba.njit(parallel=True, cache=True)
    def kernel(ez, hx, hy, ca, cb, inv_dx_e, inv_dy_e, nx, ny, periodic_x, periodic_y, hx_partner, hy_partner,
               wrap_cos_x, wrap_sin_x, wrap_cos_y, wrap_sin_y):
        for prange_i in numba.prange(nx):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            for j in range(ny):
                if i >= 1:
                    curl = (hy[i, j] - hy[i - 1, j]) * inv_dx_e[i]
                elif periodic_x:
                    curl = (hy[i, j] - (hy[nx - 1, j] * wrap_cos_x - hy_partner[nx - 1, j] * wrap_sin_x)) * inv_dx_e[i]
                else:
                    curl = hy[i, j] * inv_dx_e[i]
                if j >= 1:
                    curl -= (hx[i, j] - hx[i, j - 1]) * inv_dy_e[j]
                elif periodic_y:
                    curl -= (hx[i, j] - (hx[i, ny - 1] * wrap_cos_y - hx_partner[i, ny - 1] * wrap_sin_y)) * inv_dy_e[j]
                else:
                    curl -= hx[i, j] * inv_dy_e[j]
                ez[i, j] = ca[i, j] * ez[i, j] + cb[i, j] * curl
    return kernel


update_e_kernel = real_raw_twin('em2d.update_e', _UPDATE_E_SOURCE, 'update_e_kernel', _update_e_host,
                                numba=_update_e_numba)


def launch(nx: int, ny: int):
    bx, by, _ = _BLOCK
    grid = ((ny + bx - 1) // bx, (nx + by - 1) // by, 1)
    return grid, _BLOCK


# ---- Fused dispersive-pole / loss-accounting kernels -----------------------
# ElementwiseKernels read AND write their output args, so each pole update is
# one launch with no temporaries (the numpy twins write the same outputs in place).

# Drude ADE with J on the half steps (leap-frog of dJ/dt + gamma J = eps0 wp^2 E, like the Lorentz P update), both
# polarisations: J^{n+1/2} = c1 J^{n-1/2} + c2 E^n from the pre-update E, then E^{n+1} = E_pred - cb J^{n+1/2} after
# the Yee update (cb carries the semi-implicit sigma_e factor, so conduction and a pole on one cell stay consistent).
# Velocity form like Lorentz (DrudePole.ade_coefficients): J += damping J + drive E with damping = c1 - 1, since a
# float32 c1 ~ 1 keeps gamma dt only to ~6e-8 absolute (weakly damped metals at fine dt). Every 2D / 3D Drude kernel
# (H_z faces, em3d drude_j_weighted) uses it.
# Accurate to O((omega dt)^2); its 0D amplification determinant c1 <= 1 is necessary, not sufficient (trace 2 -
# (wp dt)^2 / eps_inf): the alternating (z = -1) mode needs eps_inf mu_r >= (C0 dt)^2 (1/dx^2 + 1/dy^2) + mu_r (wp
# dt)^2 / 4 per cell, for any gamma and sigma_e (EMPhysics._check_courant_stability). (A J update from the predictor E^{n+1} corrected with dt /
# (2 eps) (J^{n+1} + J^n) has determinant c1 + c2 dt / (2 eps) instead and grows whenever wp^2 dt > 2 eps_inf gamma.)
# E_z sums every pole's J^{n+1/2} into jz_total, which the correction, the J.E loss and state()['jz_total'] read;
# runs on the pole's bounding box, mask and PEC keep folded into the drive array (damping scalar).
def _drude_j_update_total_host(damping, drive, ez, jz, jz_total):
    jz += damping * jz + drive * ez
    jz_total += jz


drude_j_update_total = twin('em2d.drude_j_update_total', cp.ElementwiseKernel(
    'T damping, T drive, T ez',
    'T jz, T jz_total',
    '''
    jz += damping * jz + drive * ez;
    jz_total += jz;
    ''',
    'drude_j_update_total'), _drude_j_update_total_host)


# Single Drude pole (its box is the whole Drude union box): jz_total = jz, so no zero fill precedes it.
def _drude_j_update_store_host(damping, drive, ez, jz, jz_total):
    jz += damping * jz + drive * ez
    jz_total[...] = jz


drude_j_update_store = twin('em2d.drude_j_update_store', cp.ElementwiseKernel(
    'T damping, T drive, T ez',
    'T jz, T jz_total',
    '''
    jz += damping * jz + drive * ez;
    jz_total = jz;
    ''',
    'drude_j_update_store'), _drude_j_update_store_host)


# Shared with the H_z engine (per face box there, with the summed jz_total on cells here) and em3d.
def _drude_e_correction_host(current, cb, e):
    e -= cb * current


drude_e_correction = twin('em2d.drude_e_correction', cp.ElementwiseKernel(
    'T current, T cb',
    'T e',
    'e -= cb * current',
    'drude_e_correction'), _drude_e_correction_host)


# Lorentz ADE in velocity form (LorentzPole.ade_coefficients): dp = P^n - P^{n-1} becomes P^{n+1} - P^n, then
# P^{n+1} = P^n + dp. The textbook a1 P^n + a2 P^{n-1} keeps the restoring term only in 1 - a1 - a2 = w0^2 dt^2 /
# denom, which float32 a1 ~ 2 rounds by ~1e-3 for IR / phonon poles (w0 dt ~ 1e-2); here restoring is its own factor.
def _lorentz_p_update_host(damping, restoring, drive, ez, pz, dpz):
    dpz += damping * dpz - restoring * pz + drive * ez
    pz += dpz


lorentz_p_update = twin('em2d.lorentz_p_update', cp.ElementwiseKernel(
    'T damping, T restoring, T drive, T ez',
    'T pz, T dpz',
    '''
    dpz += damping * dpz - restoring * pz + drive * ez;
    pz += dpz;
    ''',
    'lorentz_p_update'), _lorentz_p_update_host)


# Lorentz E correction E^{n+1} = E_pred - (cb / dt) (P^{n+1} - P^n): from eps dE/dt + sigma_e E + dP/dt = curl H with
# the same semi-implicit sigma_e factor as the Yee update (cb / dt = 1 / (eps (1 + k))), so a lossy cell keeps chi.
# dp = P^{n+1} - P^n straight from lorentz_p_update (no float32 difference of two P buffers).
def _lorentz_e_correction_host(dp, cb, inv_dt, e):
    e -= cb * inv_dt * dp


lorentz_e_correction = twin('em2d.lorentz_e_correction', cp.ElementwiseKernel(
    'T dp, T cb, T inv_dt',
    'T e',
    'e -= cb * inv_dt * dp',
    'lorentz_e_correction'), _lorentz_e_correction_host)


# Drude work [J/m^3] is the time-centred J^{n+1/2} (E^n + E^{n+1}) / 2 dt as two q_accum_je_half calls (on the Drude
# union box): with E^n right after the J update and with E^{n+1} after every correction. J^{n+1/2} E^{n+1} alone adds
# (dt / 2) J dE/dt, -omega^2 dt / (2 gamma) of the loss per cycle. Shared with the H_z engine's face pole power
# densities [W/m^3] (scale 1 for Drude J, 1 / dt for Lorentz dp); here scale = dt. q: float32 q_accum (Q) or the
# real-typed hz pole power totals.
def _q_accum_je_half_host(current, e, scale, q):
    q += (0.5 * scale) * current * e


q_accum_je_half = twin('em2d.q_accum_je_half', cp.ElementwiseKernel(
    'T current, T e, T scale',
    'Q q',
    'q += (Q)((T)0.5 * scale * current * e)',
    'q_accum_je_half'), _q_accum_je_half_host)


# Ohmic sigma (E^{n+1})^2 dt (needs no centring), on the bounding box of the sigma_e > 0 cells (float32 sigma_e, q).
def _q_accum_ohmic_host(sigma, ez, dt, q):
    q += dt * sigma * ez * ez


q_accum_ohmic = twin('em2d.q_accum_ohmic', cp.ElementwiseKernel(
    'float32 sigma, T ez, T dt',
    'float32 q',
    'q += (float)(dt * sigma * ez * ez)',
    'q_accum_ohmic'), _q_accum_ohmic_host)


# Lorentz work [J/m^3] is the time-centred (P^{n+1} - P^n) (E^n + E^{n+1}) / 2 as two halves, both from the dPz buffer:
# with E^n right after lorentz_p_update and with E^{n+1} after every correction.
# (P^{n+1} - P^n) E^{n+1} alone counts reactive power ~ tan(omega dt / 2) Re(chi) as loss: ~6x for a weakly damped
# dielectric at omega dt ~ 0.04.
def _q_accum_lorentz_half_host(dp, ez, q):
    q += 0.5 * dp * ez


q_accum_lorentz_half = twin('em2d.q_accum_lorentz_half', cp.ElementwiseKernel(
    'T dp, T ez',
    'float32 q',
    'q += (float)((T)0.5 * dp * ez)',
    'q_accum_lorentz_half'), _q_accum_lorentz_half_host)


# Semi-implicit cell coefficients of eps dE/dt = curl H - sigma E - J: eps = eps_r eps0 [F/m], k = sigma dt / (2 eps),
# ca = (1 - k) / (1 + k), cb = (dt / eps) / (1 + k) (also the Drude and Lorentz correction factor). float32 material
# arrays, coefficients and scalars in T (the twin lifts the materials first: numpy keeps float32 arrays float32 with a
# float64 scalar).
def _e_coefficients_host(sigma, eps_r, dt, eps0, ca, cb):
    eps = eps_r.astype(ca.dtype, copy=False) * eps0
    k = sigma.astype(ca.dtype, copy=False) * dt / (2.0 * eps)
    denom = 1.0 + k
    ca[...] = (1.0 - k) / denom
    cb[...] = (dt / eps) / denom


e_coefficients = twin('em2d.e_coefficients', cp.ElementwiseKernel(
    'float32 sigma, float32 eps_r, T dt, T eps0',
    'T ca, T cb',
    '''
    T eps = eps_r * eps0;
    T k = sigma * dt / ((T)2 * eps);
    T denom = (T)1 + k;
    ca = ((T)1 - k) / denom;
    cb = (dt / eps) / denom;
    ''',
    'e_coefficients'), _e_coefficients_host)


def _h_coefficients_host(sigma_m, mu_r, dt, mu0, da, db):
    mu = mu_r.astype(da.dtype, copy=False) * mu0
    k = sigma_m.astype(da.dtype, copy=False) * dt / (2.0 * mu)
    denom = 1.0 + k
    da[...] = (1.0 - k) / denom
    db[...] = (dt / mu) / denom


h_coefficients = twin('em2d.h_coefficients', cp.ElementwiseKernel(
    'float32 sigma_m, float32 mu_r, T dt, T mu0',
    'T da, T db',
    '''
    T mu = mu_r * mu0;
    T k = sigma_m * dt / ((T)2 * mu);
    T denom = (T)1 + k;
    da = ((T)1 - k) / denom;
    db = (dt / mu) / denom;
    ''',
    'h_coefficients'), _h_coefficients_host)
