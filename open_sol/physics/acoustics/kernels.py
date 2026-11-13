"""Fused GPU kernels of the 2D velocity-stress staggered-grid FDTD (plane strain, in-plane motion, or axisymmetric (r, z)):
velocity and stress updates with the CPML memory variables, coefficient rebuild, source scatters, the thermoelastic /
pressure / strain helpers, the monoclinic c16 / c26 stress and the piezoelectric stress / polarisation / bound charge.
Every kernel is a core.kernels twin (numpy twin of the same call on the cpu backend, numba twins of the single-precision
velocity / stress updates); half_storage kernels stay gpu."""
import numpy as np

from ...core.backend import cp
from ...core.half_storage import STORED_ACCESS, storage_preamble
from ...core.kernels import raw_twin, twin

_BLOCK = (32, 8, 1)


# ---- host twin helpers (numpy, float32 fields (nx, ny); 1D spacings per row / column) ----
def _shifted(values, axis, step, periodic):
    """values at index + step (+-1) along axis: the wrapped row / column when periodic, else 0 past the grid edge."""
    if periodic:
        return np.roll(values, -step, axis=axis)
    shifted = np.zeros_like(values)
    if axis == 0:
        if step > 0:
            shifted[:-1] = values[1:]
        else:
            shifted[1:] = values[:-1]
    elif step > 0:
        shifted[:, :-1] = values[:, 1:]
    else:
        shifted[:, 1:] = values[:, :-1]
    return shifted


def _flat(array):
    """1D view of a C-contiguous array (flat-index scatters write through it)."""
    if not array.flags.c_contiguous:
        raise ValueError('acoustic scatter twin: the field array must be C-contiguous')
    return array.reshape(-1)


def _stretch_x(derivative, psi, profiles, row, slab_x):
    """CPML on the x-slab rows (slab_x >= 0): psi = b psi + a d, d += psi; profile rows `row` = b, `row` + 1 = a."""
    rows = np.flatnonzero(slab_x >= 0)
    if rows.size:
        slab = slab_x[rows]
        updated = profiles[row, rows][:, None] * psi[slab] + profiles[row + 1, rows][:, None] * derivative[rows]
        psi[slab] = updated
        derivative[rows] += updated


def _stretch_y(derivative, psi, profiles, row, slab_y):
    """The y twin of _stretch_x on the slab columns (psi slabs (nx, n_slab_y))."""
    columns = np.flatnonzero(slab_y >= 0)
    if columns.size:
        slab = slab_y[columns]
        updated = (profiles[row, columns][None, :] * psi[:, slab]
                   + profiles[row + 1, columns][None, :] * derivative[:, columns])
        psi[:, slab] = updated
        derivative[:, columns] += updated

# Staggering (as the EM Yee kernels: idx = i * ny + j, i along x): sxx, syy at cell centres, vx on the hi x face, vy on
# the hi y face, sxy on the hi-hi corner. A non-periodic axis reads the missing lo neighbour as 0 (v_n = 0 and sxy = 0
# on the lo outer face) and pins the hi outer face / corner through zero coefficients (va = vb = 0, c66c = 0): a slip
# wall (normal velocity and shear stress 0) on every non-periodic outer face. Periodic axes wrap (period n cells).
# CPML: slab_x[i] >= 0 marks the rows of the x absorbing layers (psi arrays (n_slab_x, ny)), slab_y[j] the columns of
# the y layers (psi arrays (nx, n_slab_y)); pml_x / pml_y rows = (b at cell centres, a at cell centres, b at hi faces,
# a at hi faces), a derivative d becomes d + psi with psi = b psi + a d (kappa = 1, alpha = 0).
_NEIGHBOURS = r"""
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    int ip = (i + 1 < nx) ? idx + ny : (periodic_x ? j : -1);
    int im = (i >= 1) ? idx - ny : (periodic_x ? (nx - 1) * ny + j : -1);
    int jp = (j + 1 < ny) ? idx + 1 : (periodic_y ? i * ny : -1);
    int jm = (j >= 1) ? idx - 1 : (periodic_y ? idx + ny - 1 : -1);
"""

# Axisymmetric (r, z) variants (x = r from the axis on the lo edge of cell 0, y = z; compiled with AXISYMMETRIC): the
# hoop stress stt at the cell centres; radial[k * nx + i] = the rows of axisymmetric.kernel_rows. Velocities take the
# conservative (1/r) d(r s_rr)/dr - s_tt / r and (1/r) d(r s_rz)/dr, the stresses the hoop strain rate e_tt = (v_lo +
# v_hi) / (2 r_c) with c_tt = c11, c_rt = c_zt = c12 (exact for isotropic solids and fluids): the discrete adjoint pair
# of the ring-volume energy, so the scheme stays energy conserving. The CPML stretches the radial derivatives only.
_AXISYMMETRIC = '#define AXISYMMETRIC\n'

# Stepper precisions (field_kernels): the fields are field_t (core.half_storage: float, or fp16 under half_storage, read
# by FLOAD / written by FSTORE, arithmetic in float) in stored units: stress / stress_scale, velocity / velocity_scale.
# stress_to_velocity = stress_scale / velocity_scale scales the stress divergence in the velocity update,
# velocity_to_stress its inverse the strain rate in the stress update (both exactly 1 in single). The CPML psi slabs,
# the coefficients and the strain accumulator stay float32 (psi in stored units, the strain physical).
_VELOCITY_SOURCE = r"""
#ifdef AXISYMMETRIC
#define HOOP_STRESS_ARG const field_t* __restrict__ stt,
#define RADIAL_ARG const float* __restrict__ radial,
#else
#define HOOP_STRESS_ARG
#define RADIAL_ARG
#endif
extern "C" __global__
void acoustic_update_velocity(
    field_t* __restrict__ vx, field_t* __restrict__ vy,
    const field_t* __restrict__ sxx, const field_t* __restrict__ syy, const field_t* __restrict__ sxy,
    HOOP_STRESS_ARG
    const float* __restrict__ va_x, const float* __restrict__ vb_x,
    const float* __restrict__ va_y, const float* __restrict__ vb_y,
    const float* __restrict__ inv_dx_e, const float* __restrict__ inv_dx_h,
    const float* __restrict__ inv_dy_e, const float* __restrict__ inv_dy_h,
    const int* __restrict__ slab_x, const int* __restrict__ slab_y,
    const float* __restrict__ pml_x, const float* __restrict__ pml_y,
    RADIAL_ARG
    float* __restrict__ psi_x_sxx, float* __restrict__ psi_x_sxy,
    float* __restrict__ psi_y_sxy, float* __restrict__ psi_y_syy,
    float stress_to_velocity, int nx, int ny, int n_slab_y, int periodic_x, int periodic_y)
{""" + _NEIGHBOURS + r"""
    float s_xx = FLOAD(sxx[idx]);
    float s_yy = FLOAD(syy[idx]);
    float s_xy = FLOAD(sxy[idx]);
    float s_xy_lo = (im >= 0 ? FLOAD(sxy[im]) : 0.0f);
    float s_xx_hi = (ip >= 0 ? FLOAD(sxx[ip]) : 0.0f);
#ifdef AXISYMMETRIC
    float dsxx_dx = (radial[nx + i] * s_xx_hi - radial[i] * s_xx) * inv_dx_h[i];
    float dsxy_dx = (radial[5 * nx + i] * s_xy - radial[4 * nx + i] * s_xy_lo) * inv_dx_e[i];
    float hoop_force = radial[2 * nx + i] * FLOAD(stt[idx]) + radial[3 * nx + i] * (ip >= 0 ? FLOAD(stt[ip]) : 0.0f);
#else
    float dsxx_dx = (s_xx_hi - s_xx) * inv_dx_h[i];
    float dsxy_dx = (s_xy - s_xy_lo) * inv_dx_e[i];
    float hoop_force = 0.0f;
#endif
    float dsxy_dy = (s_xy - (jm >= 0 ? FLOAD(sxy[jm]) : 0.0f)) * inv_dy_e[j];
    float dsyy_dy = ((jp >= 0 ? FLOAD(syy[jp]) : 0.0f) - s_yy) * inv_dy_h[j];
    int kx = slab_x[i];
    if (kx >= 0) {
        int p = kx * ny + j;
        float psi = pml_x[2 * nx + i] * psi_x_sxx[p] + pml_x[3 * nx + i] * dsxx_dx;
        psi_x_sxx[p] = psi;
        dsxx_dx += psi;
        psi = pml_x[i] * psi_x_sxy[p] + pml_x[nx + i] * dsxy_dx;
        psi_x_sxy[p] = psi;
        dsxy_dx += psi;
    }
    int ky = slab_y[j];
    if (ky >= 0) {
        int p = i * n_slab_y + ky;
        float psi = pml_y[j] * psi_y_sxy[p] + pml_y[ny + j] * dsxy_dy;
        psi_y_sxy[p] = psi;
        dsxy_dy += psi;
        psi = pml_y[2 * ny + j] * psi_y_syy[p] + pml_y[3 * ny + j] * dsyy_dy;
        psi_y_syy[p] = psi;
        dsyy_dy += psi;
    }
    vx[idx] = FSTORE(va_x[idx] * FLOAD(vx[idx]) + vb_x[idx] * stress_to_velocity * (dsxx_dx + dsxy_dy - hoop_force));
    vy[idx] = FSTORE(va_y[idx] * FLOAD(vy[idx]) + vb_y[idx] * stress_to_velocity * (dsxy_dx + dsyy_dy));
}
"""


_STRESS_SOURCE = r"""
#ifdef AXISYMMETRIC
#define HOOP_STRESS_ARG field_t* __restrict__ stt,
#define RADIAL_ARG const float* __restrict__ radial,
#else
#define HOOP_STRESS_ARG
#define RADIAL_ARG
#endif
#ifdef STRAIN_ACCUMULATOR
#define STRAIN_ARGS float* __restrict__ strain_xx, float* __restrict__ strain_yy, float* __restrict__ shear_strain,
#else
#define STRAIN_ARGS
#endif
#ifdef MONOCLINIC
#define RATE_ARGS float* __restrict__ rate_xx, float* __restrict__ rate_yy, float* __restrict__ rate_xy,
#else
#define RATE_ARGS
#endif
extern "C" __global__
void acoustic_update_stress(
    field_t* __restrict__ sxx, field_t* __restrict__ syy, field_t* __restrict__ sxy,
    HOOP_STRESS_ARG
    const field_t* __restrict__ vx, const field_t* __restrict__ vy,
    const float* __restrict__ c11, const float* __restrict__ c12, const float* __restrict__ c22,
    const float* __restrict__ c66c,
    STRAIN_ARGS
    RATE_ARGS
    const float* __restrict__ inv_dx_e, const float* __restrict__ inv_dx_h,
    const float* __restrict__ inv_dy_e, const float* __restrict__ inv_dy_h,
    const int* __restrict__ slab_x, const int* __restrict__ slab_y,
    const float* __restrict__ pml_x, const float* __restrict__ pml_y,
    RADIAL_ARG
    float* __restrict__ psi_x_vx, float* __restrict__ psi_x_vy,
    float* __restrict__ psi_y_vy, float* __restrict__ psi_y_vx,
    float dt, float velocity_to_stress,
    const float* __restrict__ gruneisen, const float* __restrict__ heating, float thermal_scale,
    int nx, int ny, int n_slab_y, int periodic_x, int periodic_y)
{""" + _NEIGHBOURS + r"""
    float v_x = FLOAD(vx[idx]);
    float v_y = FLOAD(vy[idx]);
    float v_x_lo = (im >= 0 ? FLOAD(vx[im]) : 0.0f);
    float dvx_dx = (v_x - v_x_lo) * inv_dx_e[i];
    float dvy_dy = (v_y - (jm >= 0 ? FLOAD(vy[jm]) : 0.0f)) * inv_dy_e[j];
    float dvy_dx = ((ip >= 0 ? FLOAD(vy[ip]) : 0.0f) - v_y) * inv_dx_h[i];
    float dvx_dy = ((jp >= 0 ? FLOAD(vx[jp]) : 0.0f) - v_x) * inv_dy_h[j];
    int kx = slab_x[i];
    if (kx >= 0) {
        int p = kx * ny + j;
        float psi = pml_x[i] * psi_x_vx[p] + pml_x[nx + i] * dvx_dx;
        psi_x_vx[p] = psi;
        dvx_dx += psi;
        psi = pml_x[2 * nx + i] * psi_x_vy[p] + pml_x[3 * nx + i] * dvy_dx;
        psi_x_vy[p] = psi;
        dvy_dx += psi;
    }
    int ky = slab_y[j];
    if (ky >= 0) {
        int p = i * n_slab_y + ky;
        float psi = pml_y[j] * psi_y_vy[p] + pml_y[ny + j] * dvy_dy;
        psi_y_vy[p] = psi;
        dvy_dy += psi;
        psi = pml_y[2 * ny + j] * psi_y_vx[p] + pml_y[3 * ny + j] * dvx_dy;
        psi_y_vx[p] = psi;
        dvx_dy += psi;
    }
    float dt_stress = dt * velocity_to_stress;
    // coupled heating slot (thermal_scale = dt x envelope / stress_scale, 0 = none): d(sxx)/dt = d(syy)/dt (= d(stt)/dt)
    // = -gruneisen q, after the stiffness update as the separate pass it replaces
    float thermal = (thermal_scale != 0.0f) ? thermal_scale * gruneisen[idx] * heating[idx] : 0.0f;
#ifdef AXISYMMETRIC
    float hoop = radial[6 * nx + i] * (v_x + v_x_lo);
    float lame = c12[idx];
    sxx[idx] = FSTORE(FLOAD(sxx[idx]) + dt_stress * (c11[idx] * dvx_dx + lame * (dvy_dy + hoop)) - thermal);
    syy[idx] = FSTORE(FLOAD(syy[idx]) + dt_stress * (lame * (dvx_dx + hoop) + c22[idx] * dvy_dy) - thermal);
    stt[idx] = FSTORE(FLOAD(stt[idx]) + dt_stress * (lame * (dvx_dx + dvy_dy) + c11[idx] * hoop) - thermal);
#else
    sxx[idx] = FSTORE(FLOAD(sxx[idx]) + dt_stress * (c11[idx] * dvx_dx + c12[idx] * dvy_dy) - thermal);
    syy[idx] = FSTORE(FLOAD(syy[idx]) + dt_stress * (c12[idx] * dvx_dx + c22[idx] * dvy_dy) - thermal);
#endif
    sxy[idx] = FSTORE(FLOAD(sxy[idx]) + dt_stress * c66c[idx] * (dvx_dy + dvy_dx));
#ifdef STRAIN_ACCUMULATOR
    strain_xx[idx] += dt * dvx_dx;
    strain_yy[idx] += dt * dvy_dy;
    if (c66c[idx] > 0.0f) shear_strain[idx] += dt * (dvx_dy + dvy_dx);
#endif
#ifdef MONOCLINIC
    rate_xx[idx] = dvx_dx;
    rate_yy[idx] = dvy_dy;
    rate_xy[idx] = (c66c[idx] > 0.0f) ? dvx_dy + dvy_dx : 0.0f;
#endif
}
"""
# The planar piezoelectric / monoclinic variants ('stress_strain*', single only: half_storage refuses piezo and c16 /
# c26 arrays) also accumulate the total strain S = integral of the strain rate (centres xx, yy; the engineering shear 2
# S_xy on the corners that carry shear, c66c > 0, else it stays 0), args after c66c. MONOCLINIC also writes this step's
# (CPML-stretched) strain rates, the shear rate 0 on corners without shear, for monoclinic_stress (args after those).

# In-plane monoclinic stiffness c16 / c26 (T_xx += c16 g, T_yy += c26 g, T_xy += c16 e_xx + c26 e_yy, g = the engineering
# shear rate), after the stress update from the rates it wrote: a cell takes c16 / c26 x the mean shear rate of its 4
# corners (lo corners missing on a non-periodic lo edge read 0), a corner with shear (c66c > 0) the mean over its 4
# cells of c16 e_xx + c26 e_yy. The two averages are adjoint (the cross energy sum_cells (c16 e_xx + c26 e_yy) g_mean),
# so the scheme conserves energy and is stable for a positive definite [[c11, c12, c16], [c12, c22, c26], [c16, c26,
# c66]] (Schur complement: the averaged coupling only shrinks); dt_stress = dt x velocity_to_stress.
_MONOCLINIC_SOURCE = r"""
extern "C" __global__
void acoustic_monoclinic_stress(
    field_t* __restrict__ sxx, field_t* __restrict__ syy, field_t* __restrict__ sxy,
    const float* __restrict__ rate_xx, const float* __restrict__ rate_yy, const float* __restrict__ rate_xy,
    const float* __restrict__ c16, const float* __restrict__ c26, const float* __restrict__ c66c,
    float dt_stress, int nx, int ny, int periodic_x, int periodic_y)
{""" + _NEIGHBOURS + r"""
    float m16 = c16[idx], m26 = c26[idx];
    if (m16 != 0.0f || m26 != 0.0f) {
        int i_lo = (i >= 1) ? i - 1 : (periodic_x ? nx - 1 : -1);
        int j_lo = (j >= 1) ? j - 1 : (periodic_y ? ny - 1 : -1);
        float shear = rate_xy[idx];
        if (im >= 0) shear += rate_xy[im];
        if (jm >= 0) shear += rate_xy[jm];
        if (i_lo >= 0 && j_lo >= 0) shear += rate_xy[i_lo * ny + j_lo];
        shear *= 0.25f * dt_stress;
        sxx[idx] = FSTORE(FLOAD(sxx[idx]) + m16 * shear);
        syy[idx] = FSTORE(FLOAD(syy[idx]) + m26 * shear);
    }
    if (c66c[idx] > 0.0f && ip >= 0 && jp >= 0) {
        int ipjp = ((i + 1 < nx) ? i + 1 : 0) * ny + ((j + 1 < ny) ? j + 1 : 0);
        float drive = c16[idx] * rate_xx[idx] + c26[idx] * rate_yy[idx]
                    + c16[ip] * rate_xx[ip] + c26[ip] * rate_yy[ip]
                    + c16[jp] * rate_xx[jp] + c26[jp] * rate_yy[jp]
                    + c16[ipjp] * rate_xx[ipjp] + c26[ipjp] * rate_yy[ipjp];
        sxy[idx] = FSTORE(FLOAD(sxy[idx]) + 0.25f * dt_stress * drive);
    }
}
"""


def _velocity_arguments(args, axisymmetric):
    """The velocity kernel's args as one fixed tuple (vx, vy, sxx, syy, sxy, stt, va_x .. pml_y, radial, psi_x_sxx ..
    periodic_y): stt and radial None in planar."""
    k = 6 if axisymmetric else 5
    radial = args[k + 12] if axisymmetric else None
    return (*args[:5], args[5] if axisymmetric else None, *args[k:k + 12], radial,
            *args[k + 12 + (1 if axisymmetric else 0):])


def _stress_arguments(args, axisymmetric, strain_accumulator, monoclinic):
    """The stress kernel's args as one fixed tuple (sxx, syy, sxy, stt, vx .. c66c, strain_xx, strain_yy, shear_strain,
    rate_xx, rate_yy, rate_xy, inv_dx_e .. pml_y, radial, psi_x_vx .. periodic_y): absent entries None."""
    k = 4 if axisymmetric else 3
    velocities_and_moduli = args[k:k + 6]
    k += 6
    strain = args[k:k + 3] if strain_accumulator else (None, None, None)
    k += 3 if strain_accumulator else 0
    rates = args[k:k + 3] if monoclinic else (None, None, None)
    k += 3 if monoclinic else 0
    common = args[k:k + 8]
    k += 8
    radial = args[k] if axisymmetric else None
    k += 1 if axisymmetric else 0
    return (*args[:3], args[3] if axisymmetric else None, *velocities_and_moduli, *strain, *rates, *common, radial,
            *args[k:])


def _velocity_host(axisymmetric):
    """numpy twin of acoustic_update_velocity (args of the planar or the AXISYMMETRIC variant)."""
    def update_velocity(*args):
        (vx, vy, sxx, syy, sxy, stt, va_x, vb_x, va_y, vb_y, inv_dx_e, inv_dx_h, inv_dy_e, inv_dy_h, slab_x, slab_y,
         pml_x, pml_y, radial, psi_x_sxx, psi_x_sxy, psi_y_sxy, psi_y_syy, stress_to_velocity, _, _, _, periodic_x,
         periodic_y) = _velocity_arguments(args, axisymmetric)
        periodic_x, periodic_y = bool(periodic_x), bool(periodic_y)
        s_xx_hi = _shifted(sxx, 0, 1, periodic_x)
        s_xy_lo = _shifted(sxy, 0, -1, periodic_x)
        if axisymmetric:
            row = lambda index: radial[index][:, None]   # noqa: E731
            dsxx_dx = (row(1) * s_xx_hi - row(0) * sxx) * inv_dx_h[:, None]
            dsxy_dx = (row(5) * sxy - row(4) * s_xy_lo) * inv_dx_e[:, None]
            hoop_force = row(2) * stt + row(3) * _shifted(stt, 0, 1, periodic_x)
        else:
            dsxx_dx = (s_xx_hi - sxx) * inv_dx_h[:, None]
            dsxy_dx = (sxy - s_xy_lo) * inv_dx_e[:, None]
        dsxy_dy = (sxy - _shifted(sxy, 1, -1, periodic_y)) * inv_dy_e[None, :]
        dsyy_dy = (_shifted(syy, 1, 1, periodic_y) - syy) * inv_dy_h[None, :]
        _stretch_x(dsxx_dx, psi_x_sxx, pml_x, 2, slab_x)
        _stretch_x(dsxy_dx, psi_x_sxy, pml_x, 0, slab_x)
        _stretch_y(dsxy_dy, psi_y_sxy, pml_y, 0, slab_y)
        _stretch_y(dsyy_dy, psi_y_syy, pml_y, 2, slab_y)
        force_x = dsxx_dx + dsxy_dy
        if axisymmetric:
            force_x -= hoop_force
        vx[...] = va_x * vx + vb_x * stress_to_velocity * force_x
        vy[...] = va_y * vy + vb_y * stress_to_velocity * (dsxy_dx + dsyy_dy)
    return update_velocity


def _stress_host(axisymmetric, strain_accumulator=False, monoclinic=False):
    """numpy twin of acoustic_update_stress (planar, AXISYMMETRIC, STRAIN_ACCUMULATOR and MONOCLINIC argument sets)."""
    def update_stress(*args):
        (sxx, syy, sxy, stt, vx, vy, c11, c12, c22, c66c, strain_xx, strain_yy, shear_strain, rate_xx, rate_yy,
         rate_xy, inv_dx_e, inv_dx_h, inv_dy_e, inv_dy_h, slab_x, slab_y, pml_x, pml_y, radial, psi_x_vx, psi_x_vy,
         psi_y_vy, psi_y_vx, dt, velocity_to_stress, gruneisen, heating, thermal_scale, _, _, _, periodic_x,
         periodic_y) = _stress_arguments(args, axisymmetric, strain_accumulator, monoclinic)
        periodic_x, periodic_y = bool(periodic_x), bool(periodic_y)
        v_x_lo = _shifted(vx, 0, -1, periodic_x)
        dvx_dx = (vx - v_x_lo) * inv_dx_e[:, None]
        dvy_dy = (vy - _shifted(vy, 1, -1, periodic_y)) * inv_dy_e[None, :]
        dvy_dx = (_shifted(vy, 0, 1, periodic_x) - vy) * inv_dx_h[:, None]
        dvx_dy = (_shifted(vx, 1, 1, periodic_y) - vx) * inv_dy_h[None, :]
        _stretch_x(dvx_dx, psi_x_vx, pml_x, 0, slab_x)
        _stretch_x(dvy_dx, psi_x_vy, pml_x, 2, slab_x)
        _stretch_y(dvy_dy, psi_y_vy, pml_y, 0, slab_y)
        _stretch_y(dvx_dy, psi_y_vx, pml_y, 2, slab_y)
        dt_stress = dt * velocity_to_stress
        if axisymmetric:
            hoop = radial[6][:, None] * (vx + v_x_lo)
            sxx += dt_stress * (c11 * dvx_dx + c12 * (dvy_dy + hoop))
            syy += dt_stress * (c12 * (dvx_dx + hoop) + c22 * dvy_dy)
            stt += dt_stress * (c12 * (dvx_dx + dvy_dy) + c11 * hoop)
        else:
            sxx += dt_stress * (c11 * dvx_dx + c12 * dvy_dy)
            syy += dt_stress * (c12 * dvx_dx + c22 * dvy_dy)
        if thermal_scale != 0.0:   # the coupled heating slot
            thermal = thermal_scale * gruneisen * heating
            sxx -= thermal
            syy -= thermal
            if axisymmetric:
                stt -= thermal
        shear_rate = dvx_dy + dvy_dx
        sxy += dt_stress * c66c * shear_rate
        has_shear = c66c > 0.0
        if strain_accumulator:
            strain_xx += dt * dvx_dx
            strain_yy += dt * dvy_dy
            shear_strain[has_shear] += dt * shear_rate[has_shear]
        if monoclinic:
            rate_xx[...] = dvx_dx
            rate_yy[...] = dvy_dy
            rate_xy[...] = np.where(has_shear, shear_rate, np.float32(0.0))
    return update_stress


# numba twins of the single-precision velocity / stress updates: per-cell loops transcribing the CUDA sources (prange
# over i, fields (nx, ny) read by [i, j]); the variant flags are compile-time closure constants, and an absent array
# (planar stt / radial, strain, rates) is passed as a dummy of the same type that the frozen flags never touch.
# cache=True: numba keys closures by their cell values (one cache entry per variant), so only the first process after
# an edit of this file compiles them (~0.5-1 s per variant and signature otherwise in every cpu process).
def _velocity_numba(axisymmetric):
    def factory(numba):
        @numba.njit(parallel=True, cache=True)
        def kernel(vx, vy, sxx, syy, sxy, stt, va_x, vb_x, va_y, vb_y, inv_dx_e, inv_dx_h, inv_dy_e, inv_dy_h, slab_x,
                   slab_y, pml_x, pml_y, radial, psi_x_sxx, psi_x_sxy, psi_y_sxy, psi_y_syy, stress_to_velocity, nx, ny,
                   periodic_x, periodic_y):
            for prange_i in numba.prange(nx):
                i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
                has_ip = i + 1 < nx or periodic_x != 0
                has_im = i >= 1 or periodic_x != 0
                ip = i + 1 if i + 1 < nx else 0
                im = i - 1 if i >= 1 else nx - 1
                kx = slab_x[i]
                for j in range(ny):
                    has_jp = j + 1 < ny or periodic_y != 0
                    has_jm = j >= 1 or periodic_y != 0
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j >= 1 else ny - 1
                    s_xx, s_yy, s_xy = sxx[i, j], syy[i, j], sxy[i, j]
                    s_xy_lo = sxy[im, j] if has_im else 0.0
                    s_xx_hi = sxx[ip, j] if has_ip else 0.0
                    if axisymmetric:
                        dsxx_dx = (radial[1, i] * s_xx_hi - radial[0, i] * s_xx) * inv_dx_h[i]
                        dsxy_dx = (radial[5, i] * s_xy - radial[4, i] * s_xy_lo) * inv_dx_e[i]
                        hoop_force = radial[2, i] * stt[i, j] + radial[3, i] * (stt[ip, j] if has_ip else 0.0)
                    else:
                        dsxx_dx = (s_xx_hi - s_xx) * inv_dx_h[i]
                        dsxy_dx = (s_xy - s_xy_lo) * inv_dx_e[i]
                        hoop_force = 0.0
                    dsxy_dy = (s_xy - (sxy[i, jm] if has_jm else 0.0)) * inv_dy_e[j]
                    dsyy_dy = ((syy[i, jp] if has_jp else 0.0) - s_yy) * inv_dy_h[j]
                    if kx >= 0:
                        psi = pml_x[2, i] * psi_x_sxx[kx, j] + pml_x[3, i] * dsxx_dx
                        psi_x_sxx[kx, j] = psi
                        dsxx_dx += psi
                        psi = pml_x[0, i] * psi_x_sxy[kx, j] + pml_x[1, i] * dsxy_dx
                        psi_x_sxy[kx, j] = psi
                        dsxy_dx += psi
                    ky = slab_y[j]
                    if ky >= 0:
                        psi = pml_y[0, j] * psi_y_sxy[i, ky] + pml_y[1, j] * dsxy_dy
                        psi_y_sxy[i, ky] = psi
                        dsxy_dy += psi
                        psi = pml_y[2, j] * psi_y_syy[i, ky] + pml_y[3, j] * dsyy_dy
                        psi_y_syy[i, ky] = psi
                        dsyy_dy += psi
                    vx[i, j] = va_x[i, j] * vx[i, j] + vb_x[i, j] * stress_to_velocity * (dsxx_dx + dsxy_dy - hoop_force)
                    vy[i, j] = va_y[i, j] * vy[i, j] + vb_y[i, j] * stress_to_velocity * (dsxy_dx + dsyy_dy)

        def update_velocity(*args):
            (vx, vy, sxx, syy, sxy, stt, va_x, vb_x, va_y, vb_y, inv_dx_e, inv_dx_h, inv_dy_e, inv_dy_h, slab_x,
             slab_y, pml_x, pml_y, radial, psi_x_sxx, psi_x_sxy, psi_y_sxy, psi_y_syy, stress_to_velocity, nx, ny, _,
             periodic_x, periodic_y) = _velocity_arguments(args, axisymmetric)
            kernel(vx, vy, sxx, syy, sxy, sxx if stt is None else stt, va_x, vb_x, va_y, vb_y, inv_dx_e, inv_dx_h,
                   inv_dy_e, inv_dy_h, slab_x, slab_y, pml_x, pml_y, pml_x if radial is None else radial, psi_x_sxx,
                   psi_x_sxy, psi_y_sxy, psi_y_syy, stress_to_velocity, nx, ny, periodic_x, periodic_y)
        return update_velocity
    return factory


def _stress_numba(axisymmetric, strain_accumulator=False, monoclinic=False):
    def factory(numba):
        @numba.njit(parallel=True, cache=True)
        def kernel(sxx, syy, sxy, stt, vx, vy, c11, c12, c22, c66c, strain_xx, strain_yy, shear_strain, rate_xx,
                   rate_yy, rate_xy, inv_dx_e, inv_dx_h, inv_dy_e, inv_dy_h, slab_x, slab_y, pml_x, pml_y, radial,
                   psi_x_vx, psi_x_vy, psi_y_vy, psi_y_vx, dt, velocity_to_stress, gruneisen, heating, thermal_scale, nx,
                   ny, periodic_x, periodic_y):
            dt_stress = dt * velocity_to_stress
            heated = thermal_scale != 0.0
            for prange_i in numba.prange(nx):
                i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
                has_ip = i + 1 < nx or periodic_x != 0
                has_im = i >= 1 or periodic_x != 0
                ip = i + 1 if i + 1 < nx else 0
                im = i - 1 if i >= 1 else nx - 1
                kx = slab_x[i]
                for j in range(ny):
                    has_jp = j + 1 < ny or periodic_y != 0
                    has_jm = j >= 1 or periodic_y != 0
                    jp = j + 1 if j + 1 < ny else 0
                    jm = j - 1 if j >= 1 else ny - 1
                    v_x, v_y = vx[i, j], vy[i, j]
                    v_x_lo = vx[im, j] if has_im else 0.0
                    dvx_dx = (v_x - v_x_lo) * inv_dx_e[i]
                    dvy_dy = (v_y - (vy[i, jm] if has_jm else 0.0)) * inv_dy_e[j]
                    dvy_dx = ((vy[ip, j] if has_ip else 0.0) - v_y) * inv_dx_h[i]
                    dvx_dy = ((vx[i, jp] if has_jp else 0.0) - v_x) * inv_dy_h[j]
                    if kx >= 0:
                        psi = pml_x[0, i] * psi_x_vx[kx, j] + pml_x[1, i] * dvx_dx
                        psi_x_vx[kx, j] = psi
                        dvx_dx += psi
                        psi = pml_x[2, i] * psi_x_vy[kx, j] + pml_x[3, i] * dvy_dx
                        psi_x_vy[kx, j] = psi
                        dvy_dx += psi
                    ky = slab_y[j]
                    if ky >= 0:
                        psi = pml_y[0, j] * psi_y_vy[i, ky] + pml_y[1, j] * dvy_dy
                        psi_y_vy[i, ky] = psi
                        dvy_dy += psi
                        psi = pml_y[2, j] * psi_y_vx[i, ky] + pml_y[3, j] * dvx_dy
                        psi_y_vx[i, ky] = psi
                        dvx_dy += psi
                    if axisymmetric:
                        hoop = radial[6, i] * (v_x + v_x_lo)
                        lame = c12[i, j]
                        sxx[i, j] += dt_stress * (c11[i, j] * dvx_dx + lame * (dvy_dy + hoop))
                        syy[i, j] += dt_stress * (lame * (dvx_dx + hoop) + c22[i, j] * dvy_dy)
                        stt[i, j] += dt_stress * (lame * (dvx_dx + dvy_dy) + c11[i, j] * hoop)
                    else:
                        sxx[i, j] += dt_stress * (c11[i, j] * dvx_dx + c12[i, j] * dvy_dy)
                        syy[i, j] += dt_stress * (c12[i, j] * dvx_dx + c22[i, j] * dvy_dy)
                    if heated:   # the coupled heating slot
                        thermal = thermal_scale * gruneisen[i, j] * heating[i, j]
                        sxx[i, j] -= thermal
                        syy[i, j] -= thermal
                        if axisymmetric:
                            stt[i, j] -= thermal
                    shear_rate = dvx_dy + dvy_dx
                    sxy[i, j] += dt_stress * c66c[i, j] * shear_rate
                    has_shear = c66c[i, j] > 0.0
                    if strain_accumulator:
                        strain_xx[i, j] += dt * dvx_dx
                        strain_yy[i, j] += dt * dvy_dy
                        if has_shear:
                            shear_strain[i, j] += dt * shear_rate
                    if monoclinic:
                        rate_xx[i, j] = dvx_dx
                        rate_yy[i, j] = dvy_dy
                        rate_xy[i, j] = shear_rate if has_shear else 0.0

        def update_stress(*args):
            (sxx, syy, sxy, stt, vx, vy, c11, c12, c22, c66c, strain_xx, strain_yy, shear_strain, rate_xx, rate_yy,
             rate_xy, inv_dx_e, inv_dx_h, inv_dy_e, inv_dy_h, slab_x, slab_y, pml_x, pml_y, radial, psi_x_vx, psi_x_vy,
             psi_y_vy, psi_y_vx, dt, velocity_to_stress, gruneisen, heating, thermal_scale, nx, ny, _, periodic_x,
             periodic_y) = _stress_arguments(args, axisymmetric, strain_accumulator, monoclinic)
            strain_xx, strain_yy, shear_strain, rate_xx, rate_yy, rate_xy = (
                sxx if value is None else value for value in (strain_xx, strain_yy, shear_strain, rate_xx, rate_yy,
                                                               rate_xy))
            if thermal_scale == 0.0:   # the dummies of an absent heating slot: 2D arrays the kernel never reads
                gruneisen = heating = sxx
            kernel(sxx, syy, sxy, sxx if stt is None else stt, vx, vy, c11, c12, c22, c66c, strain_xx, strain_yy,
                   shear_strain, rate_xx, rate_yy, rate_xy, inv_dx_e, inv_dx_h, inv_dy_e, inv_dy_h, slab_x, slab_y,
                   pml_x, pml_y, pml_x if radial is None else radial, psi_x_vx, psi_x_vy, psi_y_vy, psi_y_vx, dt,
                   velocity_to_stress, gruneisen, heating, thermal_scale, nx, ny, periodic_x, periodic_y)
        return update_stress
    return factory


def _monoclinic_stress_host(sxx, syy, sxy, rate_xx, rate_yy, rate_xy, c16, c26, c66c, dt_stress, nx, ny, periodic_x,
                            periodic_y):
    """numpy twin of acoustic_monoclinic_stress."""
    periodic_x, periodic_y = bool(periodic_x), bool(periodic_y)
    shear = _corner_sum(rate_xy, periodic_x, periodic_y, -1) * (np.float32(0.25) * dt_stress)
    active = (c16 != 0.0) | (c26 != 0.0)
    sxx[active] += (c16 * shear)[active]
    syy[active] += (c26 * shear)[active]
    drive = _corner_sum(c16 * rate_xx + c26 * rate_yy, periodic_x, periodic_y, 1)
    corners = c66c > 0.0
    if not periodic_x:
        corners[-1, :] = False
    if not periodic_y:
        corners[:, -1] = False
    sxy[corners] += (np.float32(0.25) * dt_stress * drive)[corners]


def _isotropic_moduli_host(rho, sound_speed, shear_speed, youngs, poisson):
    """(longitudinal, lame_lambda, shear) [Pa] per cell: from the speeds, else youngs / poisson, else 0."""
    with np.errstate(all='ignore'):
        speed_longitudinal = rho * sound_speed * sound_speed
        speed_shear = rho * shear_speed * shear_speed
        youngs_shear = youngs / (2.0 * (1.0 + poisson))
        youngs_lambda = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    from_speeds = sound_speed > 0.0
    from_youngs = ~from_speeds & (youngs > 0.0)
    zero = np.float32(0.0)

    def pick(by_speeds, by_youngs):
        return np.where(from_speeds, by_speeds, np.where(from_youngs, by_youngs, zero))
    return (pick(speed_longitudinal, youngs_lambda + 2.0 * youngs_shear),
            pick(speed_longitudinal - 2.0 * speed_shear, youngs_lambda), pick(speed_shear, youngs_shear))


def _face_coefficients_host(rho, damping, keep, has_keep, axis, periodic, share_lo, share_hi, pinned, dt, va, vb):
    """va / vb of the hi faces along axis (0 where pinned, on a non-periodic hi edge, without density or rigid)."""
    shape = (-1, 1) if axis == 0 else (1, -1)
    lo, hi = share_lo.reshape(shape), share_hi.reshape(shape)
    face_rho = lo * rho + hi * _shifted(rho, axis, 1, periodic)
    valid = (face_rho > 0.0) & ~pinned
    if not periodic:
        inside = np.ones(rho.shape[axis], dtype=bool)
        inside[-1] = False
        valid &= inside.reshape(shape)
    if has_keep:
        valid &= (keep * _shifted(keep, axis, 1, periodic)) != 0.0
    k = np.float32(0.5) * (lo * damping + hi * _shifted(damping, axis, 1, periodic)) * dt
    zero = np.float32(0.0)
    with np.errstate(all='ignore'):
        va[...] = np.where(valid, (1.0 - k) / (1.0 + k), zero)
        vb[...] = np.where(valid, dt / (face_rho * (1.0 + k)), zero)


def _coefficients_host(rho, sound_speed, shear_speed, youngs, poisson, m11, m12, m22, m66, damping, keep, has_keep,
                       share_x_lo, share_x_hi, share_y_lo, share_y_hi, c11, c12, c22, c66c, va_x, vb_x, va_y, vb_y,
                       dt, nx, ny, periodic_x, periodic_y, pin_x_lo, pin_x_hi, pin_y_lo, pin_y_hi):
    """numpy twin of acoustic_coefficients."""
    periodic_x, periodic_y = bool(periodic_x), bool(periodic_y)
    longitudinal, lame_lambda, shear = _isotropic_moduli_host(rho, sound_speed, shear_speed, youngs, poisson)
    c11[...] = np.where(np.isnan(m11), longitudinal, m11)
    c12[...] = np.where(np.isnan(m12), lame_lambda, m12)
    c22[...] = np.where(np.isnan(m22), longitudinal, m22)
    g0 = np.where(np.isnan(m66), shear, m66)
    g1 = _shifted(g0, 0, 1, periodic_x)          # missing neighbours read 0: no shear corner there
    g2 = _shifted(g0, 1, 1, periodic_y)
    g3 = _shifted(g1, 1, 1, periodic_y)
    positive = (g0 > 0.0) & (g1 > 0.0) & (g2 > 0.0) & (g3 > 0.0)
    with np.errstate(all='ignore'):
        corner = np.where(positive, 4.0 / (1.0 / g0 + 1.0 / g1 + 1.0 / g2 + 1.0 / g3), np.float32(0.0))
    rows, columns = np.arange(int(nx))[:, None], np.arange(int(ny))[None, :]
    pinned_x = (rows == int(pin_x_lo)) | (rows == int(pin_x_hi))
    pinned_y = (columns == int(pin_y_lo)) | (columns == int(pin_y_hi))
    c66c[...] = np.where(pinned_x | pinned_y, np.float32(0.0), corner)
    _face_coefficients_host(rho, damping, keep, has_keep, 0, periodic_x, share_x_lo, share_x_hi, pinned_x, dt, va_x,
                            vb_x)
    _face_coefficients_host(rho, damping, keep, has_keep, 1, periodic_y, share_y_lo, share_y_hi, pinned_y, dt, va_y,
                            vb_y)


# Coefficients from the material arrays, one thread per cell (no temporaries: safe on per-step material updates).
# Isotropic moduli: from the speeds when sound_speed > 0 (c11 / c22 = rho sound_speed^2, c12 = rho (sound_speed^2 - 2
# shear_speed^2), shear modulus rho shear_speed^2), else from youngs_modulus > 0 / poisson_ratio (the solid_mechanics
# vocabulary: c11 = c22 = lambda + 2 mu, c12 = lambda, shear modulus mu), else 0; a c11 / c12 / c22 / c66 array holding
# a number (NaN = unset) overrides its entry. c66c = harmonic mean of the shear
# moduli of the 4 cells around the corner (0 when any is <= 0: fluid-solid and vacuum corners carry no shear; 0 on a
# non-periodic hi row / column). Faces: density and damping are the share-weighted means of the two cells (exact
# mass of the dual cell: width shares, axisymmetric x faces the ring-volume shares); va = (1 - k) / (1 + k), vb = dt / (rho_f (1 + k)), k = damping_f dt / 2; va = vb = 0 (pinned
# velocity) without density, on a rigid cell (keep 0) or on a non-periodic hi outer face. pin_x_lo / _hi (row) and
# pin_y_lo / _hi (column) >= 0 = a slip wall on the hi face of that row / column (a 'rigid' window face of a padded
# grid): its normal velocity and the shear corners on it are 0. Host twin: _coefficients_host.
coefficients_kernel = raw_twin('acoustics.coefficients', cp.RawKernel(r"""
__device__ __forceinline__ void isotropic_moduli(
    const float* rho, const float* sound_speed, const float* shear_speed, const float* youngs, const float* poisson,
    int k, float* longitudinal, float* lame_lambda, float* shear)
{
    float vp = sound_speed[k];
    if (vp > 0.0f) {
        float r = rho[k];
        float vs = shear_speed[k];
        *longitudinal = r * vp * vp;
        *shear = r * vs * vs;
        *lame_lambda = *longitudinal - 2.0f * *shear;
    } else if (youngs[k] > 0.0f) {
        float e = youngs[k];
        float nu = poisson[k];
        *shear = e / (2.0f * (1.0f + nu));
        *lame_lambda = e * nu / ((1.0f + nu) * (1.0f - 2.0f * nu));
        *longitudinal = *lame_lambda + 2.0f * *shear;
    } else {
        *longitudinal = 0.0f;
        *lame_lambda = 0.0f;
        *shear = 0.0f;
    }
}

__device__ __forceinline__ float shear_modulus(
    const float* rho, const float* sound_speed, const float* shear_speed, const float* youngs, const float* poisson,
    const float* m66, int k)
{
    float m = m66[k];
    if (m == m) return m;
    float longitudinal, lame_lambda, shear;
    isotropic_moduli(rho, sound_speed, shear_speed, youngs, poisson, k, &longitudinal, &lame_lambda, &shear);
    return shear;
}

__device__ __forceinline__ void face_coefficients(
    const float* rho, const float* damping, const float* keep, int has_keep, int idx, int other,
    float share_lo, float share_hi, float dt, float* va, float* vb)
{
    if (other < 0) { *va = 0.0f; *vb = 0.0f; return; }
    float rho_f = share_lo * rho[idx] + share_hi * rho[other];
    float keep_f = has_keep ? keep[idx] * keep[other] : 1.0f;
    if (!(rho_f > 0.0f) || keep_f == 0.0f) { *va = 0.0f; *vb = 0.0f; return; }
    float k = 0.5f * (share_lo * damping[idx] + share_hi * damping[other]) * dt;
    *va = (1.0f - k) / (1.0f + k);
    *vb = dt / (rho_f * (1.0f + k));
}

extern "C" __global__
void acoustic_coefficients(
    const float* __restrict__ rho, const float* __restrict__ sound_speed, const float* __restrict__ shear_speed,
    const float* __restrict__ youngs, const float* __restrict__ poisson,
    const float* __restrict__ m11, const float* __restrict__ m12, const float* __restrict__ m22,
    const float* __restrict__ m66, const float* __restrict__ damping,
    const float* __restrict__ keep, int has_keep,
    const float* __restrict__ share_x_lo, const float* __restrict__ share_x_hi,
    const float* __restrict__ share_y_lo, const float* __restrict__ share_y_hi,
    float* __restrict__ c11, float* __restrict__ c12, float* __restrict__ c22, float* __restrict__ c66c,
    float* __restrict__ va_x, float* __restrict__ vb_x, float* __restrict__ va_y, float* __restrict__ vb_y,
    float dt, int nx, int ny, int periodic_x, int periodic_y,
    int pin_x_lo, int pin_x_hi, int pin_y_lo, int pin_y_hi)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    int ip = (i + 1 < nx) ? idx + ny : (periodic_x ? j : -1);
    int jp = (j + 1 < ny) ? idx + 1 : (periodic_y ? i * ny : -1);
    bool pinned_x = (i == pin_x_lo || i == pin_x_hi);
    bool pinned_y = (j == pin_y_lo || j == pin_y_hi);
    int ipjp = (ip >= 0 && jp >= 0) ? ((i + 1 < nx) ? i + 1 : 0) * ny + ((j + 1 < ny) ? j + 1 : 0) : -1;

    float longitudinal, lame_lambda, shear;
    isotropic_moduli(rho, sound_speed, shear_speed, youngs, poisson, idx, &longitudinal, &lame_lambda, &shear);
    float a = m11[idx], b = m12[idx], c = m22[idx];
    c11[idx] = (a != a) ? longitudinal : a;
    c12[idx] = (b != b) ? lame_lambda : b;
    c22[idx] = (c != c) ? longitudinal : c;

    float corner = 0.0f;
    if (ipjp >= 0) {
        float g0 = shear_modulus(rho, sound_speed, shear_speed, youngs, poisson, m66, idx);
        float g1 = shear_modulus(rho, sound_speed, shear_speed, youngs, poisson, m66, ip);
        float g2 = shear_modulus(rho, sound_speed, shear_speed, youngs, poisson, m66, jp);
        float g3 = shear_modulus(rho, sound_speed, shear_speed, youngs, poisson, m66, ipjp);
        if (g0 > 0.0f && g1 > 0.0f && g2 > 0.0f && g3 > 0.0f)
            corner = 4.0f / (1.0f / g0 + 1.0f / g1 + 1.0f / g2 + 1.0f / g3);
    }
    c66c[idx] = (pinned_x || pinned_y) ? 0.0f : corner;

    face_coefficients(rho, damping, keep, has_keep, idx, pinned_x ? -1 : ip, share_x_lo[i], share_x_hi[i], dt,
                      va_x + idx, vb_x + idx);
    face_coefficients(rho, damping, keep, has_keep, idx, pinned_y ? -1 : jp, share_y_lo[j], share_y_hi[j], dt,
                      va_y + idx, vb_y + idx);
}
""", "acoustic_coefficients"), _coefficients_host)


# Stability stiffness S [Pa] of a cell (time step c_bound^2 = S / rho_face): the largest eigenvalue of the acoustic
# tensor at the corner wavenumbers (pi / dx, pi / dy), [[c11 wx + c66 wy, |c12 + c66| sqrt(wx wy)], [., c66 wx + c22
# wy]], wx = (1/dx^2) / (1/dx^2 + 1/dy^2), wy = 1 - wx (isotropic: c11; square cells: lambda_max(M) / 2), at least
# max(c11, c22) and 0; NaN when an input is not finite. shear = the largest c66c of the cell's corners. c16 / c26 (the
# material's monoclinic terms; 0 scalars without them) enter as the absolute-value bound (diagonals + 2 |c16| s, + 2
# |c26| s, off-diagonal |c16| wx + |c12 + c66| s + |c26| wy, s = sqrt(wx wy)): it dominates the symbol entrywise at
# every wavenumber (the corner / cell averages scale c16 / c26 by cos(kx dx / 2) cos(ky dy / 2) <= 1), so it is a
# bound, not the exact limit. Two-way piezo: the stiffened c^D = c^E + e^T eps^-1 e (eps = eps0 eps_dc, inverse_eps0 =
# 1 / eps0; e = 0, eps = 1 scalars otherwise), its c16 / c26 added to the material ones. Mirrored on the host by
# definition.stability_stiffness / piezo_stiffening.
_STIFFENING_PREAMBLE = r'''
__device__ __forceinline__ double piezo_stiffening(double i11, double i12, double i22, double a1, double b1, double a2,
                                                   double b2)
{
    return i11 * a1 * a2 + i12 * (a1 * b2 + b1 * a2) + i22 * b1 * b2;
}
'''


def _stability_stiffness_host(c11, c12, c22, shear, c16, c26, e11, e12, e16, e21, e22, e26, eps_xx, eps_yy, eps_xy,
                              inverse_eps0, weight_x, weight_y):
    """numpy twin of acoustic_stability_stiffness (float64 arithmetic, float32 result of the broadcast shape)."""
    c11, c12, c22, shear, c16, c26, e11, e12, e16, e21, e22, e26, eps_xx, eps_yy, eps_xy = (
        np.asarray(value, dtype=np.float64) for value in (c11, c12, c22, shear, c16, c26, e11, e12, e16, e21, e22,
                                                            e26, eps_xx, eps_yy, eps_xy))
    weight_x, weight_y = float(weight_x), float(weight_y)
    determinant = eps_xx * eps_yy - eps_xy * eps_xy
    with np.errstate(all='ignore'):
        scale = np.where(determinant > 0.0, float(inverse_eps0) / determinant, 0.0)
        i11, i12, i22 = eps_yy * scale, -eps_xy * scale, eps_xx * scale

        def stiffening(a1, b1, a2, b2):
            return i11 * a1 * a2 + i12 * (a1 * b2 + b1 * a2) + i22 * b1 * b2
        s11 = c11 + stiffening(e11, e21, e11, e21)
        s12 = c12 + stiffening(e11, e21, e12, e22)
        s22 = c22 + stiffening(e12, e22, e12, e22)
        s66 = shear + stiffening(e16, e26, e16, e26)
        s16 = np.abs(c16 + stiffening(e11, e21, e16, e26))
        s26 = np.abs(c26 + stiffening(e12, e22, e16, e26))
        s = np.sqrt(weight_x * weight_y)
        p = s11 * weight_x + s66 * weight_y + 2.0 * s16 * s
        q = s66 * weight_x + s22 * weight_y + 2.0 * s26 * s
        r = s16 * weight_x + np.abs(s12 + s66) * s + s26 * weight_y
        h = 0.5 * (p - q)
        largest = np.maximum(np.maximum(0.5 * (p + q) + np.sqrt(h * h + r * r), np.maximum(s11, s22)), 0.0)
        total = s11 + s12 + s22 + s66 + s16 + s26
    return np.asarray(np.where(np.isfinite(total), largest, np.nan), dtype=np.float32)


stability_stiffness = twin('acoustics.stability_stiffness', cp.ElementwiseKernel(
    'float32 c11, float32 c12, float32 c22, float32 shear, float32 c16, float32 c26, float32 e11, float32 e12, '
    'float32 e16, float32 e21, '
    'float32 e22, float32 e26, float32 eps_xx, float32 eps_yy, float32 eps_xy, float64 inverse_eps0, '
    'float64 weight_x, float64 weight_y',
    'float32 stiffness',
    '''
    const double det = (double)eps_xx * eps_yy - (double)eps_xy * eps_xy;
    const double scale = det > 0.0 ? inverse_eps0 / det : 0.0;
    const double i11 = eps_yy * scale, i12 = -eps_xy * scale, i22 = eps_xx * scale;
    const double s11 = c11 + piezo_stiffening(i11, i12, i22, e11, e21, e11, e21);
    const double s12 = c12 + piezo_stiffening(i11, i12, i22, e11, e21, e12, e22);
    const double s22 = c22 + piezo_stiffening(i11, i12, i22, e12, e22, e12, e22);
    const double s66 = shear + piezo_stiffening(i11, i12, i22, e16, e26, e16, e26);
    const double s16 = fabs(c16 + piezo_stiffening(i11, i12, i22, e11, e21, e16, e26));
    const double s26 = fabs(c26 + piezo_stiffening(i11, i12, i22, e12, e22, e16, e26));
    const double s = sqrt(weight_x * weight_y);
    const double p = s11 * weight_x + s66 * weight_y + 2.0 * s16 * s;
    const double q = s66 * weight_x + s22 * weight_y + 2.0 * s26 * s;
    const double r = s16 * weight_x + fabs(s12 + s66) * s + s26 * weight_y;
    const double h = 0.5 * (p - q);
    double largest = 0.5 * (p + q) + sqrt(h * h + r * r);
    if (s11 > largest) largest = s11;
    if (s22 > largest) largest = s22;
    if (largest < 0.0) largest = 0.0;
    const double total = s11 + s12 + s22 + s66 + s16 + s26;
    stiffness = isfinite(total) ? (float)largest : (float)(total - total);
    ''',
    'acoustic_stability_stiffness', preamble=_STIFFENING_PREAMBLE), _stability_stiffness_host)


# The scatters' waveform value: a scalar argument, or (VALUE_TABLE: pulse tables, fixed arguments for the step graphs)
# table[step - table_start] at the device step counter, nothing written outside the table or where it holds 0 (the
# steps the eager injection skips).
_VALUE_TABLE = r"""
#define VALUE_PARAMETERS const float* table, const int* step_counter, int table_start, int table_length
#define LOAD_VALUE const int table_step = step_counter[0] - table_start; \
    if (table_step < 0 || table_step >= table_length) return; \
    const float value = table[table_step]; \
    if (value == 0.0f) return;
"""
_SCALAR_VALUE = r"""
#ifndef VALUE_PARAMETERS
#define VALUE_PARAMETERS float value
#define LOAD_VALUE
#endif
"""

# field[index[k]] += coefficient[index[k]] * weight[k] * value (velocity sources: coefficient = vb of the face; value in
# stored units). Single: atomicAdd; half_storage (HALF_FIELDS) a plain read-modify-write, so the indices of one scatter
# must be distinct (the solver's face entries are).
_SCATTER_SOURCE = _SCALAR_VALUE + r"""
extern "C" __global__
void acoustic_scatter_add(field_t* field, const float* coefficient, const int* index, const float* weight, int n,
                          VALUE_PARAMETERS)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    LOAD_VALUE
    int c = index[k];
#ifdef HALF_FIELDS
    field[c] = FSTORE(FLOAD(field[c]) + coefficient[c] * weight[k] * value);
#else
    atomicAdd(field + c, coefficient[c] * weight[k] * value);
#endif
}
"""

# Volume injection: an isotropic in-plane strain rate -q / 2 per axis, so sxx -= (c11 + c12) / 2 * q dt and syy -=
# (c12 + c22) / 2 * q dt; value = dt * waveform (/ stress_scale), weight = volume rate / cell area (a fluid: p += K q
# dt). Axisymmetric: an isotropic 3D strain rate -q / 3 per axis (c_tt = c11, c_rt = c_zt = c12), so a fluid again gets
# p += K q dt. One cell per source (distinct indices).
_VOLUME_SCATTER_SOURCE = _SCALAR_VALUE + r"""
#ifdef HALF_FIELDS
#define ADD_STRESS(field, c, value) field[c] = FSTORE(FLOAD(field[c]) + (value))
#else
#define ADD_STRESS(field, c, value) atomicAdd(field + c, value)
#endif
extern "C" __global__
void acoustic_volume_scatter(field_t* sxx, field_t* syy,
#ifdef AXISYMMETRIC
                             field_t* stt,
#endif
                             const float* c11, const float* c12, const float* c22,
                             const int* index, const float* weight, int n, VALUE_PARAMETERS)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;
    LOAD_VALUE
    int c = index[k];
#ifdef AXISYMMETRIC
    float rate = weight[k] * value / 3.0f;
    float lame = c12[c];
    ADD_STRESS(sxx, c, -(c11[c] + 2.0f * lame) * rate);
    ADD_STRESS(syy, c, -(c22[c] + 2.0f * lame) * rate);
    ADD_STRESS(stt, c, -(c11[c] + 2.0f * lame) * rate);
#else
    float rate = weight[k] * value;
    ADD_STRESS(sxx, c, -0.5f * (c11[c] + c12[c]) * rate);
    ADD_STRESS(syy, c, -0.5f * (c12[c] + c22[c]) * rate);
#endif
}
"""


# Every probe of a physics in one launch per step: samples[row, k] = the stored-unit value of probe k at its cell (codes
# PROBE_RECORD_CODES: a field, or the pressure -(sxx + syy) / 2, axisymmetric -(srr + szz + stt) / 3, as
# pressure_from_stress with scale 1); stt is a dummy (sxx) in planar physics, whose probes never read it. The row is a
# device counter the kernel advances (fixed arguments: the launch can sit in a step graph), so it runs as ONE block
# (PROBE_RECORD_LAUNCH): every thread reads the row before thread 0 advances it.
PROBE_RECORD_CODES = {'vx': 0, 'vy': 1, 'sxx': 2, 'syy': 3, 'sxy': 4, 'stt': 5, 'pressure': 6,
                      'pressure_axisymmetric': 7}
PROBE_RECORD_LAUNCH = ((1, 1, 1), (128, 1, 1))
_PROBE_RECORD_SOURCE = r"""
extern "C" __global__
void acoustic_probe_record(const field_t* vx, const field_t* vy, const field_t* sxx, const field_t* syy,
                           const field_t* sxy, const field_t* stt, const int* cells, const int* codes, float* samples,
                           int* row_counter, int n_probes)
{
    const long long row = row_counter[0];
    for (int k = threadIdx.x; k < n_probes; k += blockDim.x) {
        int c = cells[k];
        float value;
        switch (codes[k]) {
            case 0: value = FLOAD(vx[c]); break;
            case 1: value = FLOAD(vy[c]); break;
            case 2: value = FLOAD(sxx[c]); break;
            case 3: value = FLOAD(syy[c]); break;
            case 4: value = FLOAD(sxy[c]); break;
            case 5: value = FLOAD(stt[c]); break;
            case 6: value = -0.5f * ((float)FLOAD(sxx[c]) + (float)FLOAD(syy[c])) * 1.0f; break;
            default: value = -((float)FLOAD(sxx[c]) + (float)FLOAD(syy[c]) + (float)FLOAD(stt[c])) / 3.0f * 1.0f;
        }
        samples[row * n_probes + k] = value;
    }
    __syncthreads();
    if (threadIdx.x == 0) row_counter[0] = (int)(row + 1);
}
"""


def _probe_record_host(vx, vy, sxx, syy, sxy, stt, cells, codes, samples, row_counter, n_probes):
    """numpy twin of acoustic_probe_record (a loop over the few probes; float32 arithmetic as _pressure_host)."""
    fields = tuple(_flat(field) for field in (vx, vy, sxx, syy, sxy, stt))
    target = _flat(samples)
    row = int(row_counter[0])
    row_counter[0] = row + 1
    base = row * int(n_probes)
    for k in range(int(n_probes)):
        cell, code = int(cells[k]), int(codes[k])
        if code <= 5:
            value = np.float32(fields[code][cell])
        elif code == 6:
            value = np.float32(-0.5) * (np.float32(fields[2][cell]) + np.float32(fields[3][cell])) * np.float32(1.0)
        else:
            value = (-(np.float32(fields[2][cell]) + np.float32(fields[3][cell]) + np.float32(fields[5][cell]))
                     / np.float32(3.0) * np.float32(1.0))
        target[base + k] = value


def _scatter_add_host(field, coefficient, index, weight, n, value):
    """numpy twin of acoustic_scatter_add (np.add.at: repeated indices add like atomicAdd)."""
    cells = index[:int(n)]
    np.add.at(_flat(field), cells, _flat(coefficient)[cells] * weight[:int(n)] * value)


def _table_value(table, step_counter, table_start, table_length):
    """The float32 value a VALUE_TABLE kernel reads at the step counter; None outside the table or for 0 (no write)."""
    step = int(step_counter[0]) - int(table_start)
    if not 0 <= step < int(table_length):
        return None
    value = np.float32(table[step])
    return None if value == 0 else value


def _scatter_table_host(field, coefficient, index, weight, n, table, step_counter, table_start, table_length):
    """numpy twin of acoustic_scatter_add with VALUE_TABLE."""
    value = _table_value(table, step_counter, table_start, table_length)
    if value is not None:
        _scatter_add_host(field, coefficient, index, weight, n, value)


def _volume_scatter_table_host(axisymmetric):
    """numpy twin of acoustic_volume_scatter with VALUE_TABLE (planar or AXISYMMETRIC argument set)."""
    plain = _volume_scatter_host(axisymmetric)

    def volume_scatter_table(*args):
        value = _table_value(*args[-4:])
        if value is not None:
            plain(*args[:-4], value)
    return volume_scatter_table


# Device step counter of the pulse tables: +1 per step (the last launch of a step that has tables; one thread).
STEP_COUNT_LAUNCH = ((1, 1, 1), (1, 1, 1))
_STEP_COUNT_SOURCE = r"""
extern "C" __global__
void acoustic_step_count(int* step_counter)
{
    step_counter[0] += 1;
}
"""


def _step_count_host(step_counter):
    """numpy twin of acoustic_step_count."""
    step_counter[0] += 1


def _volume_scatter_host(axisymmetric):
    """numpy twin of acoustic_volume_scatter (planar or AXISYMMETRIC argument set)."""
    def volume_scatter(*args):
        sxx, syy = args[:2]
        stt = args[2] if axisymmetric else None
        c11, c12, c22, index, weight, n, value = args[3 if axisymmetric else 2:]
        cells = index[:int(n)]
        a, b, c = _flat(c11)[cells], _flat(c12)[cells], _flat(c22)[cells]
        if axisymmetric:
            rate = weight[:int(n)] * value / np.float32(3.0)
            np.add.at(_flat(sxx), cells, -(a + 2.0 * b) * rate)
            np.add.at(_flat(syy), cells, -(c + 2.0 * b) * rate)
            np.add.at(_flat(stt), cells, -(a + 2.0 * b) * rate)
        else:
            rate = weight[:int(n)] * value
            np.add.at(_flat(sxx), cells, np.float32(-0.5) * (a + b) * rate)
            np.add.at(_flat(syy), cells, np.float32(-0.5) * (b + c) * rate)
    return volume_scatter


# key -> (defines, source, kernel name) of the kernels touching the stepped fields, compiled per stepper precision
_FIELD_SOURCES = {
    'velocity': ('', _VELOCITY_SOURCE, 'acoustic_update_velocity'),
    'velocity_axisymmetric': (_AXISYMMETRIC, _VELOCITY_SOURCE, 'acoustic_update_velocity'),
    'stress': ('', _STRESS_SOURCE, 'acoustic_update_stress'),
    'stress_axisymmetric': (_AXISYMMETRIC, _STRESS_SOURCE, 'acoustic_update_stress'),
    'stress_strain': ('#define STRAIN_ACCUMULATOR\n', _STRESS_SOURCE, 'acoustic_update_stress'),
    'stress_strain_monoclinic': ('#define STRAIN_ACCUMULATOR\n#define MONOCLINIC\n', _STRESS_SOURCE,
                                 'acoustic_update_stress'),
    'monoclinic_stress': ('', _MONOCLINIC_SOURCE, 'acoustic_monoclinic_stress'),
    'scatter': ('', _SCATTER_SOURCE, 'acoustic_scatter_add'),
    'volume_scatter': ('', _VOLUME_SCATTER_SOURCE, 'acoustic_volume_scatter'),
    'volume_scatter_axisymmetric': (_AXISYMMETRIC, _VOLUME_SCATTER_SOURCE, 'acoustic_volume_scatter'),
    'scatter_table': (_VALUE_TABLE, _SCATTER_SOURCE, 'acoustic_scatter_add'),
    'volume_scatter_table': (_VALUE_TABLE, _VOLUME_SCATTER_SOURCE, 'acoustic_volume_scatter'),
    'volume_scatter_table_axisymmetric': (_AXISYMMETRIC + _VALUE_TABLE, _VOLUME_SCATTER_SOURCE,
                                          'acoustic_volume_scatter'),
    'probe_record': ('', _PROBE_RECORD_SOURCE, 'acoustic_probe_record'),
    'step_count': ('', _STEP_COUNT_SOURCE, 'acoustic_step_count'),
}
# key -> numpy twin of the single-precision field kernels (half_storage has none: gpu only)
_FIELD_HOST_TWINS = {
    'velocity': _velocity_host(False),
    'velocity_axisymmetric': _velocity_host(True),
    'stress': _stress_host(False),
    'stress_axisymmetric': _stress_host(True),
    'stress_strain': _stress_host(False, strain_accumulator=True),
    'stress_strain_monoclinic': _stress_host(False, strain_accumulator=True, monoclinic=True),
    'monoclinic_stress': _monoclinic_stress_host,
    'scatter': _scatter_add_host,
    'volume_scatter': _volume_scatter_host(False),
    'volume_scatter_axisymmetric': _volume_scatter_host(True),
    'scatter_table': _scatter_table_host,
    'volume_scatter_table': _volume_scatter_table_host(False),
    'volume_scatter_table_axisymmetric': _volume_scatter_table_host(True),
    'probe_record': _probe_record_host,
    'step_count': _step_count_host,
}
# key -> numba factory of the per-step velocity / stress updates (single only; the rest stays numpy)
_FIELD_NUMBA_TWINS = {
    'velocity': _velocity_numba(False),
    'velocity_axisymmetric': _velocity_numba(True),
    'stress': _stress_numba(False),
    'stress_axisymmetric': _stress_numba(True),
    'stress_strain': _stress_numba(False, strain_accumulator=True),
    'stress_strain_monoclinic': _stress_numba(False, strain_accumulator=True, monoclinic=True),
}
_FIELD_KERNELS = {}


def field_kernels(precision='single') -> dict:
    """{key: kernel} of the field kernels (_FIELD_SOURCES) for a stepper precision ('single': core.kernels twins
    'acoustics.<key>'; 'half_storage': gpu RawKernels), built on first use (compiled at the first launch)."""
    if precision not in _FIELD_KERNELS:
        preamble = storage_preamble(precision, 'field_t', 'FLOAD', 'FSTORE')
        if precision == 'half_storage':
            preamble += '#define HALF_FIELDS\n'
        kernels = {key: cp.RawKernel(preamble + defines + source, name)
                   for key, (defines, source, name) in _FIELD_SOURCES.items()}
        if precision == 'single':
            kernels = {key: raw_twin(f'acoustics.{key}', kernel, _FIELD_HOST_TWINS[key],
                                     numba=_FIELD_NUMBA_TWINS.get(key))
                       for key, kernel in kernels.items()}
        _FIELD_KERNELS[precision] = kernels
    return _FIELD_KERNELS[precision]


# Thermoelastic stress of a heating rate q [W/m^3]: d(sxx)/dt = d(syy)/dt = -gruneisen q (p rises by gruneisen q dt);
# scale = dt * envelope / stress_scale (stored units); T = float32, or the uint16 bits of the float16 fields under
# half_storage (callers pass core.half_storage.half_bits: cupy's float16 casts do not compile everywhere).
def _thermoelastic_stress_host(gruneisen, heating, scale, sxx, syy):
    change = scale * gruneisen * heating
    sxx -= change
    syy -= change


def _thermoelastic_hoop_stress_host(gruneisen, heating, scale, stt):
    stt -= scale * gruneisen * heating


thermoelastic_stress = twin('acoustics.thermoelastic_stress', cp.ElementwiseKernel(
    'float32 gruneisen, float32 heating, float32 scale',
    'T sxx, T syy',
    '''
    float d = scale * gruneisen * heating;
    stored_set(sxx, stored_load(sxx) - d);
    stored_set(syy, stored_load(syy) - d);
    ''',
    'acoustic_thermoelastic_stress', preamble=STORED_ACCESS), _thermoelastic_stress_host)

# The hoop part of the same source (axisymmetric: d(stt)/dt = -gruneisen q too).
thermoelastic_hoop_stress = twin('acoustics.thermoelastic_hoop_stress', cp.ElementwiseKernel(
    'float32 gruneisen, float32 heating, float32 scale',
    'T stt',
    'stored_set(stt, stored_load(stt) - scale * gruneisen * heating)',
    'acoustic_thermoelastic_hoop_stress', preamble=STORED_ACCESS), _thermoelastic_hoop_stress_host)


def _stored_to_physical_host(stored, scale, value):
    value[...] = stored.astype(np.float32) * scale


# Physical float32 value of a stored field: value = stored * scale (half_storage readers: state(), renderer).
stored_to_physical = twin('acoustics.stored_to_physical', cp.ElementwiseKernel(
    'T stored, float32 scale',
    'float32 value',
    'value = (float)stored * scale',
    'acoustic_stored_to_physical'), _stored_to_physical_host)


def _piezo_component_stress_host(field, e_xx, e_yy, e_xy, last, sxx, syy, shear):
    change = field - last
    last[...] = field
    sxx -= e_xx * change
    syy -= e_yy * change
    shear[...] = e_xy * change


# Piezoelectric stress of the change of one electric field component E_k at the cell centres (T = c S - e^T E): dE =
# E_k - E_last, sxx -= e_k1 dE, syy -= e_k2 dE, shear = e_k6 dE (to the corners by piezo_shear_stress_kernel); E_last
# becomes E_k.
piezo_component_stress = twin('acoustics.piezo_component_stress', cp.ElementwiseKernel(
    'float32 field, float32 e_xx, float32 e_yy, float32 e_xy',
    'float32 last, float32 sxx, float32 syy, float32 shear',
    '''
    const float change = field - last;
    last = field;
    sxx -= e_xx * change;
    syy -= e_yy * change;
    shear = e_xy * change;
    ''',
    'acoustic_piezo_component_stress'), _piezo_component_stress_host)


def _corner_sum(values, periodic_x, periodic_y, step):
    """values + its x, y and diagonal neighbours at index + step (+1: the hi cells of a corner, -1: the lo corners of a
    cell); missing neighbours on a non-periodic edge read 0."""
    along_x = _shifted(values, 0, step, periodic_x)
    return values + along_x + _shifted(values, 1, step, periodic_y) + _shifted(along_x, 1, step, periodic_y)


def _piezo_shear_stress_host(shear, c66c, sxy, nx, ny, periodic_x, periodic_y):
    periodic_x, periodic_y = bool(periodic_x), bool(periodic_y)
    corners = c66c > 0.0
    if not periodic_x:
        corners[-1, :] = False
    if not periodic_y:
        corners[:, -1] = False
    sxy[corners] -= (np.float32(0.25) * _corner_sum(shear, periodic_x, periodic_y, 1))[corners]


# sxy (hi-hi corner) -= the mean of the shear drive of its 4 cells, only where the corner carries shear (c66c > 0: not
# on fluid / vacuum corners or pinned slip-wall corners, which also covers the non-periodic hi edges).
piezo_shear_stress_kernel = raw_twin('acoustics.piezo_shear_stress', cp.RawKernel(r"""
extern "C" __global__
void acoustic_piezo_shear_stress(const float* __restrict__ shear, const float* __restrict__ c66c,
                                 float* __restrict__ sxy, int nx, int ny, int periodic_x, int periodic_y)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    if (!(c66c[idx] > 0.0f)) return;
    int ip = (i + 1 < nx) ? idx + ny : (periodic_x ? j : -1);
    int jp = (j + 1 < ny) ? idx + 1 : (periodic_y ? i * ny : -1);
    if (ip < 0 || jp < 0) return;
    int ipjp = ((i + 1 < nx) ? i + 1 : 0) * ny + ((j + 1 < ny) ? j + 1 : 0);
    sxy[idx] -= 0.25f * (shear[idx] + shear[ip] + shear[jp] + shear[ipjp]);
}
""", "acoustic_piezo_shear_stress"), _piezo_shear_stress_host)


def _piezo_polarization_host(strain_xx, strain_yy, shear_strain, e11, e12, e16, e21, e22, e26, polarization_x,
                             polarization_y, nx, ny, periodic_x, periodic_y):
    shear = np.float32(0.25) * _corner_sum(shear_strain, bool(periodic_x), bool(periodic_y), -1)
    polarization_x[...] = e11 * strain_xx + e12 * strain_yy + e16 * shear
    polarization_y[...] = e21 * strain_xx + e22 * strain_yy + e26 * shear


# Piezoelectric polarisation P = e S [C/m^2] at the cell centres from the strain accumulator: the engineering shear is
# the mean of the cell's 4 corners (lo corners missing on a non-periodic lo edge read 0): the adjoint of the 4-cell
# mean of piezo_shear_stress_kernel, so the two-way coupling stays symmetric.
piezo_polarization_kernel = raw_twin('acoustics.piezo_polarization', cp.RawKernel(r"""
extern "C" __global__
void acoustic_piezo_polarization(
    const float* __restrict__ strain_xx, const float* __restrict__ strain_yy, const float* __restrict__ shear_strain,
    const float* __restrict__ e11, const float* __restrict__ e12, const float* __restrict__ e16,
    const float* __restrict__ e21, const float* __restrict__ e22, const float* __restrict__ e26,
    float* __restrict__ polarization_x, float* __restrict__ polarization_y,
    int nx, int ny, int periodic_x, int periodic_y)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    int i_lo = (i >= 1) ? i - 1 : (periodic_x ? nx - 1 : -1);
    int j_lo = (j >= 1) ? j - 1 : (periodic_y ? ny - 1 : -1);
    float shear = shear_strain[idx];
    if (i_lo >= 0) shear += shear_strain[i_lo * ny + j];
    if (j_lo >= 0) shear += shear_strain[i * ny + j_lo];
    if (i_lo >= 0 && j_lo >= 0) shear += shear_strain[i_lo * ny + j_lo];
    shear *= 0.25f;
    float s_xx = strain_xx[idx], s_yy = strain_yy[idx];
    polarization_x[idx] = e11[idx] * s_xx + e12[idx] * s_yy + e16[idx] * shear;
    polarization_y[idx] = e21[idx] * s_xx + e22[idx] * s_yy + e26[idx] * shear;
}
""", "acoustic_piezo_polarization"), _piezo_polarization_host)


def _window_face_values(polarization, axis, own_lo, own_hi):
    """(lo, hi) face values of the window cells along axis: two-cell means inside, the cell's own value on an owned
    window edge face, else 0."""
    zero = np.float32(0.0)
    lo, hi = np.empty_like(polarization), np.empty_like(polarization)
    if axis == 0:
        mean = np.float32(0.5) * (polarization[1:] + polarization[:-1])
        lo[1:], hi[:-1] = mean, mean
        lo[0] = polarization[0] if own_lo else zero
        hi[-1] = polarization[-1] if own_hi else zero
    else:
        mean = np.float32(0.5) * (polarization[:, 1:] + polarization[:, :-1])
        lo[:, 1:], hi[:, :-1] = mean, mean
        lo[:, 0] = polarization[:, 0] if own_lo else zero
        hi[:, -1] = polarization[:, -1] if own_hi else zero
    return lo, hi


def _piezo_bound_charge_host(polarization_x, polarization_y, inv_dx, inv_dy, charge, nx, ny, x0, x1, y0, y1, own_x_lo,
                             own_x_hi, own_y_lo, own_y_hi):
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    charge[...] = 0.0
    if x1 <= x0 or y1 <= y0:
        return
    x_lo, x_hi = _window_face_values(polarization_x[x0:x1, y0:y1], 0, own_x_lo, own_x_hi)
    y_lo, y_hi = _window_face_values(polarization_y[x0:x1, y0:y1], 1, own_y_lo, own_y_hi)
    charge[x0:x1, y0:y1] = -((x_hi - x_lo) * inv_dx[x0:x1, None] + (y_hi - y_lo) * inv_dy[None, y0:y1])


# Bound charge rho_b = -div P [C/m^3] on the window cells [x0, x1) x [y0, y1) in face form: an interior face takes the
# mean P of its two cells; a window edge face the cell's own P when own_<face> (a Dirichlet electrode: its surface
# charge lands on the conductor), else 0 (an insulating / mirror face). 0 outside the window.
piezo_bound_charge_kernel = raw_twin('acoustics.piezo_bound_charge', cp.RawKernel(r"""
extern "C" __global__
void acoustic_piezo_bound_charge(
    const float* __restrict__ polarization_x, const float* __restrict__ polarization_y,
    const float* __restrict__ inv_dx, const float* __restrict__ inv_dy, float* __restrict__ charge,
    int nx, int ny, int x0, int x1, int y0, int y1, int own_x_lo, int own_x_hi, int own_y_lo, int own_y_hi)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    if (i < x0 || i >= x1 || j < y0 || j >= y1) { charge[idx] = 0.0f; return; }
    float px = polarization_x[idx];
    float py = polarization_y[idx];
    float x_lo = (i > x0) ? 0.5f * (px + polarization_x[idx - ny]) : (own_x_lo ? px : 0.0f);
    float x_hi = (i + 1 < x1) ? 0.5f * (px + polarization_x[idx + ny]) : (own_x_hi ? px : 0.0f);
    float y_lo = (j > y0) ? 0.5f * (py + polarization_y[idx - 1]) : (own_y_lo ? py : 0.0f);
    float y_hi = (j + 1 < y1) ? 0.5f * (py + polarization_y[idx + 1]) : (own_y_hi ? py : 0.0f);
    charge[idx] = -((x_hi - x_lo) * inv_dx[i] + (y_hi - y_lo) * inv_dy[j]);
}
""", "acoustic_piezo_bound_charge"), _piezo_bound_charge_host)


def _pressure_host(sxx, syy, scale, p):
    p[...] = np.float32(-0.5) * (sxx.astype(np.float32) + syy) * scale


def _pressure_axisymmetric_host(sxx, syy, stt, scale, p):
    p[...] = -(sxx.astype(np.float32) + syy + stt) / np.float32(3.0) * scale


# Mean in-plane pressure p = -(sxx + syy) / 2 (exactly the pressure of a fluid); stresses of the field dtype T, scale =
# stress_scale (1 for the stored-unit pressure of a probe).
pressure_from_stress = twin('acoustics.pressure_from_stress', cp.ElementwiseKernel(
    'T sxx, T syy, float32 scale',
    'float32 p',
    'p = -0.5f * ((float)sxx + (float)syy) * scale',
    'acoustic_pressure_from_stress'), _pressure_host)

# Axisymmetric mean stress pressure p = -(srr + szz + stt) / 3.
pressure_from_stress_axisymmetric = twin('acoustics.pressure_from_stress_axisymmetric', cp.ElementwiseKernel(
    'T sxx, T syy, T stt, float32 scale',
    'float32 p',
    'p = -((float)sxx + (float)syy + (float)stt) / 3.0f * scale',
    'acoustic_pressure_from_stress_axisymmetric'), _pressure_axisymmetric_host)


def _elastic_strain_host(stored_xx, stored_yy, c11, c12, c22, scale, strain_xx, strain_yy):
    sxx, syy = stored_xx.astype(np.float32) * scale, stored_yy.astype(np.float32) * scale
    determinant = c11 * c22 - c12 * c12
    solid = determinant > np.float32(1e-6) * c11 * c22
    fluid = ~solid & (c11 + c12 > 0.0)
    zero = np.float32(0.0)
    with np.errstate(all='ignore'):
        volumetric = np.float32(0.5) * (sxx + syy) / (c11 + c12)
        strain_xx[...] = np.where(solid, (c22 * sxx - c12 * syy) / determinant, np.where(fluid, volumetric, zero))
        strain_yy[...] = np.where(solid, (c11 * syy - c12 * sxx) / determinant, np.where(fluid, volumetric, zero))


def _elastic_strain_axisymmetric_host(stored_xx, stored_yy, stored_tt, c11, c12, c22, scale, strain_xx, strain_yy):
    sxx, syy, stt = (stored.astype(np.float32) * scale for stored in (stored_xx, stored_yy, stored_tt))
    a, b, c = c11, c12, c22
    m00, m01, m02 = c * a - b * b, b * b - b * a, b * b - c * b
    m11, m12 = a * a - b * b, b * b - a * b
    determinant = a * m00 + b * m01 + b * m02
    total = 2.0 * a + 6.0 * b + c
    solid = determinant > np.float32(1e-6) * a * c * a
    fluid = ~solid & (total > 0.0)
    zero = np.float32(0.0)
    with np.errstate(all='ignore'):
        volumetric = (sxx + syy + stt) / total
        strain_xx[...] = np.where(solid, (m00 * sxx + m01 * syy + m02 * stt) / determinant,
                                  np.where(fluid, volumetric, zero))
        strain_yy[...] = np.where(solid, (m01 * sxx + m11 * syy + m12 * stt) / determinant,
                                  np.where(fluid, volumetric, zero))


# In-plane elastic strain of the centre stresses (plane strain, the stress-producing part: thermoelastic stress counts
# as -C e_thermal): the inverse of [[c11, c12], [c12, c22]] in a solid; a fluid (det ~ 0) determines only the
# volumetric strain, split evenly (exx = eyy = (sxx + syy) / (2 (c11 + c12))); 0 without stiffness. Stresses of the
# field dtype T times scale (stress_scale).
elastic_strain_from_stress = twin('acoustics.elastic_strain_from_stress', cp.ElementwiseKernel(
    'T stored_xx, T stored_yy, float32 c11, float32 c12, float32 c22, float32 scale',
    'float32 strain_xx, float32 strain_yy',
    '''
    const float sxx = (float)stored_xx * scale, syy = (float)stored_yy * scale;
    const float det = c11 * c22 - c12 * c12;
    if (det > 1e-6f * c11 * c22) {
        strain_xx = (c22 * sxx - c12 * syy) / det;
        strain_yy = (c11 * syy - c12 * sxx) / det;
    } else if (c11 + c12 > 0.0f) {
        strain_xx = 0.5f * (sxx + syy) / (c11 + c12);
        strain_yy = strain_xx;
    } else {
        strain_xx = 0.0f;
        strain_yy = 0.0f;
    }
    ''',
    'acoustic_elastic_strain_from_stress'), _elastic_strain_host)


# Axisymmetric: (e_rr, e_zz) of the 3x3 normal block [[c11, c12, c12], [c12, c22, c12], [c12, c12, c11]] inverted on
# (srr, szz, stt) by cofactors; a fluid (det ~ 0) gets the volumetric strain split in thirds, (srr + szz + stt) / (sum
# of the 9 entries) each; 0 without stiffness.
elastic_strain_from_stress_axisymmetric = twin(
    'acoustics.elastic_strain_from_stress_axisymmetric', cp.ElementwiseKernel(
    'T stored_xx, T stored_yy, T stored_tt, float32 c11, float32 c12, float32 c22, float32 scale',
    'float32 strain_xx, float32 strain_yy',
    '''
    const float sxx = (float)stored_xx * scale, syy = (float)stored_yy * scale, stt = (float)stored_tt * scale;
    const float a = c11, b = c12, c = c22;
    const float m00 = c * a - b * b, m01 = b * b - b * a, m02 = b * b - c * b;   // cofactor rows 0 and 1
    const float m11 = a * a - b * b, m12 = b * b - a * b;
    const float det = a * m00 + b * m01 + b * m02;
    const float total = 2.0f * a + 6.0f * b + c;
    if (det > 1e-6f * a * c * a) {
        strain_xx = (m00 * sxx + m01 * syy + m02 * stt) / det;
        strain_yy = (m01 * sxx + m11 * syy + m12 * stt) / det;
    } else if (total > 0.0f) {
        strain_xx = (sxx + syy + stt) / total;
        strain_yy = strain_xx;
    } else {
        strain_xx = 0.0f;
        strain_yy = 0.0f;
    }
    ''',
    'acoustic_elastic_strain_from_stress_axisymmetric'), _elastic_strain_axisymmetric_host)


# Engineering shear strain on the corners = sxy / c66c (0 on corners without shear stiffness); sxy of the field dtype T
# times scale (stress_scale).
def _corner_shear_strain_host(stored_xy, c66c, scale, shear_strain):
    with np.errstate(all='ignore'):
        shear_strain[...] = np.where(c66c > 0.0, stored_xy.astype(np.float32) * scale / c66c, np.float32(0.0))


corner_shear_strain = twin('acoustics.corner_shear_strain', cp.ElementwiseKernel(
    'T stored_xy, float32 c66c, float32 scale',
    'float32 shear_strain',
    'shear_strain = (c66c > 0.0f) ? (float)stored_xy * scale / c66c : 0.0f',
    'acoustic_corner_shear_strain'), _corner_shear_strain_host)


def _centre_shear_strain_host(corner, centre, nx, ny, periodic_x, periodic_y):
    centre[...] = np.float32(0.25) * _corner_sum(corner, bool(periodic_x), bool(periodic_y), -1)


# Centre value of a hi-hi corner quantity: the mean of the cell's 4 corners, missing lo corners on a non-periodic lo edge
# read 0 (the averaging of piezo_polarization_kernel).
centre_shear_strain_kernel = raw_twin('acoustics.centre_shear_strain', cp.RawKernel(r"""
extern "C" __global__
void acoustic_centre_shear_strain(const float* __restrict__ corner, float* __restrict__ centre, int nx, int ny,
                                  int periodic_x, int periodic_y)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = i * ny + j;
    int i_lo = (i >= 1) ? i - 1 : (periodic_x ? nx - 1 : -1);
    int j_lo = (j >= 1) ? j - 1 : (periodic_y ? ny - 1 : -1);
    float sum = corner[idx];
    if (i_lo >= 0) sum += corner[i_lo * ny + j];
    if (j_lo >= 0) sum += corner[i * ny + j_lo];
    if (i_lo >= 0 && j_lo >= 0) sum += corner[i_lo * ny + j_lo];
    centre[idx] = 0.25f * sum;
}
""", "acoustic_centre_shear_strain"), _centre_shear_strain_host)


def launch(nx: int, ny: int):
    """(grid, block) of the cell kernels (x of the block along j, the contiguous axis)."""
    bx, by, _ = _BLOCK
    return ((ny + bx - 1) // bx, (nx + by - 1) // by, 1), _BLOCK


def scatter_launch(n: int):
    """(grid, block) of the scatter kernels over n entries."""
    return ((max(int(n), 1) + 127) // 128, 1, 1), (128, 1, 1)
