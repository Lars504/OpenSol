"""D2Q9 lattice-Boltzmann kernels of FluidPhysics: TRT collision (magic parameter 3/16), He-Luo incompressible
equilibrium, Guo forcing, shifted populations g = f - w, pull streaming with the face conditions fused in; every kernel a
core.kernels twin (numpy twin of the same call on the cpu backend; numba twin of the 'single' step)."""
import numpy as np

from ...core.backend import array_module_of, cp
from ...core.half_storage import STORED_ACCESS, storage_preamble
from ...core.kernels import raw_twin, twin

MAGIC_LAMBDA = 3.0 / 16.0   # TRT (tau+ - 1/2)(tau- - 1/2): halfway bounce-back exact for Poiseuille flow
N_DIRECTIONS = 9
BLOCK_SIZE = 256
# Face condition codes shared with the kernel (faces in core.constants.FACES order).
FACE_WALL, FACE_SLIP, FACE_VELOCITY, FACE_PRESSURE, FACE_PERIODIC = 0, 1, 2, 3, 4
FACE_KIND_NAMES = {FACE_WALL: 'wall', FACE_SLIP: 'slip', FACE_VELOCITY: 'velocity', FACE_PRESSURE: 'pressure',
                   FACE_PERIODIC: 'periodic'}
# Lattice directions (i along x): rest, +x, +y, -x, -y, (+x+y), (-x+y), (-x-y), (+x-y); weights of the rest equilibrium.
DIRECTION_X = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
DIRECTION_Y = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
WEIGHTS = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

# One lattice step per launch: gather the post-collision populations g* = f* - w of the previous step (pull), with
# halfway bounce-back off solid neighbours and the face rules, then collide and write g* of this step plus the
# macroscopic lattice velocity (j + F/2, rho0 = 1) and density deviation rho - 1. Face values (face, component,
# position): component 0 / 1 = lattice velocity x / y of a wall or velocity face, 2 = density deviation of a pressure
# face. A corner diagonal takes the y face's rule when that face is a wall or slip face, else the x face's.
# Pressure faces: anti-bounce-back f_q = -f*_qbar + 2 e+_q(rho_w, u_b) - 2 Lambda+ (c_q . grad)(3 w_q c_q . u); the last
# (TRT second-order) term uses the previous step's velocity gradient at the link midpoint r - c_q / 2 (half a cell
# along the face for a diagonal), read from the neighbours' post-collision populations (velocity arrays are rewritten
# in place); without it a sheared face (Poiseuille inlet) injects spurious momentum.
# Populations are pop_t (core.half_storage.storage_preamble: float, or fp16 x POPULATION_SCALE under half_storage),
# read by POP_LOAD and written by POP_STORE; every other array and all arithmetic stay float32.
# BOUZIDI 1 (solver_defaults.conformal_boundaries): a solid neighbour link of a cell with wall_row >= 0 takes the
# linear interpolated bounce-back at the wall fraction delta of that link (bouzidi.py): delta >= 1/2 mixes f*_i and
# f*_q of the cell, delta < 1/2 f*_i of the cell and of its upstream fluid neighbour n + c_q (halfway without one).
# The weights sum to 1, so the shifted populations g = f - w need no correction.
# Speed: the bulk direction loop is unrolled (literal tables fold, g[] stays in registers); the face rules and the
# Bouzidi links run out of line (__noinline__, __constant__ tables) so their dynamic indexing never spills a bulk cell
# to local memory; periodic wraps compare-and-add, no integer modulo. Same arithmetic as the numpy / numba twins.
_STEP_SOURCE = r'''
// direction tables of the out-of-line boundary helpers (dynamic q; the bulk loop keeps literal tables)
__constant__ int LINK_X[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant__ int LINK_Y[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__constant__ int LINK_OPPOSITE[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
__constant__ int LINK_MIRROR_X[9] = {0, 3, 2, 1, 4, 6, 5, 8, 7};
__constant__ int LINK_MIRROR_Y[9] = {0, 1, 4, 3, 2, 8, 7, 6, 5};
__constant__ float LINK_WEIGHT[9] = {4.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f,
                                     1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f};

// lattice velocity u = sum c g* - F/2 of window cell (i, j) from its post-collision populations; false outside / solid
__device__ bool neighbour_velocity(const pop_t* g_in, const bool* fluid, const float* force_x, const float* force_y,
                                   int i, int j, const int wx, const int wy, const bool periodic_x,
                                   const bool periodic_y, float* u)
{
    if (periodic_x) i = ((i % wx) + wx) % wx;
    if (periodic_y) j = ((j % wy) + wy) % wy;
    if (i < 0 || i >= wx || j < 0 || j >= wy) return false;
    const int s = i * wy + j;
    if (!fluid[s]) return false;
    const int n_cells = wx * wy;
    const pop_t* p = g_in + s;
    float g[9];
    for (int q = 1; q < 9; ++q) g[q] = POP_LOAD(p[q * n_cells]);
    u[0] = g[1] - g[3] + g[5] - g[6] - g[7] + g[8] - 0.5f * force_x[s];
    u[1] = g[2] - g[4] + g[5] + g[6] - g[7] - g[8] - 0.5f * force_y[s];
    return true;
}

// velocity gradient {du_x/dx, du_y/dx, du_x/dy, du_y/dy} of outer face `face`, half a cell from boundary cell (i, j)
// towards `side` (-1 / 0 / +1 along the face: the link midpoint; at the cell the curvature d_t^2 u_n costs ~1 % of a
// Poiseuille flow rate). Side links difference the two rows (normal derivative averaged over both); side 0 or no fluid
// side neighbour: central at the cell (one-sided second order next to solids / window ends). Normal derivative of the
// tangential component one-sided inward, the normal one from continuity (d_n u_n = -d_t u_t).
__device__ void face_velocity_gradient(const pop_t* g_in, const bool* fluid, const float* force_x,
                                       const float* force_y, const int face, const int i, const int j,
                                       const int side, const float* u0, const int wx, const int wy,
                                       const bool periodic_x, const bool periodic_y, float* gradient)
{
    const bool x_face = face < 2;
    const int ti = x_face ? 0 : 1, tj = x_face ? 1 : 0;
    const int inward = (face % 2 == 0) ? 1 : -1;
    const int ni = x_face ? inward : 0, nj = x_face ? 0 : inward;
    float up[2], um[2], u2[2], u_side[2], u_inward[2], tangential[2] = {0.f, 0.f}, normal[2] = {0.f, 0.f};
    const bool has_side = side != 0 && neighbour_velocity(g_in, fluid, force_x, force_y, i + side * ti,
                                                          j + side * tj, wx, wy, periodic_x, periodic_y, u_side);
    const bool has_plus = !has_side && neighbour_velocity(g_in, fluid, force_x, force_y, i + ti, j + tj, wx, wy,
                                                          periodic_x, periodic_y, up);
    const bool has_minus = !has_side && neighbour_velocity(g_in, fluid, force_x, force_y, i - ti, j - tj, wx, wy,
                                                           periodic_x, periodic_y, um);
    if (has_side) {
        for (int c = 0; c < 2; ++c) tangential[c] = side * (u_side[c] - u0[c]);
    } else if (has_plus && has_minus) {
        for (int c = 0; c < 2; ++c) tangential[c] = 0.5f * (up[c] - um[c]);
    } else if (has_plus) {
        if (neighbour_velocity(g_in, fluid, force_x, force_y, i + 2 * ti, j + 2 * tj, wx, wy,
                               periodic_x, periodic_y, u2)) {
            for (int c = 0; c < 2; ++c) tangential[c] = 0.5f * (-3.f * u0[c] + 4.f * up[c] - u2[c]);
        } else {
            for (int c = 0; c < 2; ++c) tangential[c] = up[c] - u0[c];
        }
    } else if (has_minus) {
        if (neighbour_velocity(g_in, fluid, force_x, force_y, i - 2 * ti, j - 2 * tj, wx, wy,
                               periodic_x, periodic_y, u2)) {
            for (int c = 0; c < 2; ++c) tangential[c] = 0.5f * (3.f * u0[c] - 4.f * um[c] + u2[c]);
        } else {
            for (int c = 0; c < 2; ++c) tangential[c] = u0[c] - um[c];
        }
    }
    if (neighbour_velocity(g_in, fluid, force_x, force_y, i + ni, j + nj, wx, wy, periodic_x, periodic_y, u_inward)) {
        for (int c = 0; c < 2; ++c) normal[c] = inward * (u_inward[c] - u0[c]);
        if (has_side && neighbour_velocity(g_in, fluid, force_x, force_y, i + ni + side * ti, j + nj + side * tj,
                                           wx, wy, periodic_x, periodic_y, u2)) {
            for (int c = 0; c < 2; ++c) normal[c] = 0.5f * (normal[c] + inward * (u2[c] - u_side[c]));
        }
    }
    if (x_face) {
        gradient[0] = -tangential[1]; gradient[1] = normal[1]; gradient[2] = tangential[0]; gradient[3] = tangential[1];
    } else {
        gradient[0] = tangential[0]; gradient[1] = tangential[1]; gradient[2] = normal[0]; gradient[3] = -tangential[0];
    }
}

// index wrapped across a periodic axis of `size` cells (callers step at most one cell outside: no integer modulo)
__device__ __forceinline__ int wrap_once(const int index, const int size)
{
    return index < 0 ? index + size : (index >= size ? index - size : index);
}

// g*_q of fluid cell n = (i, j) whose source cell (si, sj) (wrapped on periodic axes) lies outside the window: the rule
// of the face the link crosses
__device__ __noinline__ float face_link(const pop_t* __restrict__ g_in, const float* __restrict__ velocity_x,
                                        const float* __restrict__ velocity_y, const bool* __restrict__ fluid,
                                        const float* __restrict__ force_x, const float* __restrict__ force_y,
                                        const int* __restrict__ face_kind, const float* __restrict__ face_values,
                                        const float wp, const int q, const int i, const int j, const int si,
                                        const int sj, const int n, const int wx, const int wy, const int face_length,
                                        const bool periodic_x, const bool periodic_y)
{
    const int n_cells = wx * wy;
    const bool out_x = si < 0 || si >= wx;
    const bool out_y = sj < 0 || sj >= wy;
    int face;
    if (out_x && out_y) {
        const int face_y = sj < 0 ? 2 : 3;
        face = face_kind[face_y] <= 1 ? face_y : (si < 0 ? 0 : 1);
    } else if (out_x) {
        face = si < 0 ? 0 : 1;
    } else {
        face = sj < 0 ? 2 : 3;
    }
    const int kind = face_kind[face];
    const int position = face < 2 ? j : i;
    const float* values = face_values + face * 3 * face_length;
    const int link_x = LINK_X[q];
    const int link_y = LINK_Y[q];
    const float weight = LINK_WEIGHT[q];
    if (kind == 1) {
        // specular: the mirrored population of the ghost cell's mirror image, the upstream cell along the face
        // (cell n itself for the normal link; n where that cell is outside the window or solid)
        const int mi = face < 2 ? i : si;
        const int mj = face < 2 ? sj : j;
        const bool upstream = mi >= 0 && mi < wx && mj >= 0 && mj < wy && fluid[mi * wy + mj];
        const int mirrored = face < 2 ? LINK_MIRROR_X[q] : LINK_MIRROR_Y[q];
        return POP_LOAD(g_in[mirrored * n_cells + (upstream ? mi * wy + mj : n)]);
    }
    if (kind == 3) {
        // gradient at the link midpoint r - c_q / 2 (tangential offset -c_t / 2), du_x/dx, du_y/dx, du_x/dy, du_y/dy;
        // velocity_x / y[n] still hold the previous step (only this thread rewrites them, after the gather)
        const float old_x = velocity_x[n];
        const float old_y = velocity_y[n];
        const float old_velocity[2] = {old_x, old_y};
        float gradient[4];
        face_velocity_gradient(g_in, fluid, force_x, force_y, face, i, j, face < 2 ? -link_y : -link_x,
                               old_velocity, wx, wy, periodic_x, periodic_y, gradient);
        const float cu = link_x * old_x + link_y * old_y;
        const float directional_shear = link_x * link_x * gradient[0] + link_x * link_y * (gradient[1] + gradient[2])
                                        + link_y * link_y * gradient[3];
        return -POP_LOAD(g_in[LINK_OPPOSITE[q] * n_cells + n])
               + 2.f * weight * (values[2 * face_length + position] + 4.5f * cu * cu
                                 - 1.5f * (old_x * old_x + old_y * old_y))
               - 6.f * (1.f / wp - 0.5f) * weight * directional_shear;
    }
    const float cu = link_x * values[position] + link_y * values[face_length + position];
    return POP_LOAD(g_in[LINK_OPPOSITE[q] * n_cells + n]) + 6.f * weight * cu;
}

#if BOUZIDI
// linear interpolated bounce-back of the solid link n -> n - c_q, wall at delta of the link (bouzidi.py); halfway
// (bounce) without a wall row, or for delta < 1/2 without an upstream fluid cell
__device__ __noinline__ float bouzidi_link(const pop_t* __restrict__ g_in, const bool* __restrict__ fluid,
                                           const int* __restrict__ wall_row, const float* __restrict__ wall_fraction,
                                           const int q, const int i, const int j, const int n, const int wx,
                                           const int wy, const bool periodic_x, const bool periodic_y,
                                           const float bounce)
{
    const int row = wall_row[n];
    if (row < 0) return bounce;
    const int n_cells = wx * wy;
    const float delta = wall_fraction[row * 8 + q - 1];
    if (delta >= 0.5f) return (0.5f / delta) * bounce + (1.f - 0.5f / delta) * POP_LOAD(g_in[q * n_cells + n]);
    int ui = i + LINK_X[q];
    int uj = j + LINK_Y[q];
    if (periodic_x) ui = wrap_once(ui, wx);
    if (periodic_y) uj = wrap_once(uj, wy);
    if (ui >= 0 && ui < wx && uj >= 0 && uj < wy && fluid[ui * wy + uj])
        return 2.f * delta * bounce + (1.f - 2.f * delta) * POP_LOAD(g_in[LINK_OPPOSITE[q] * n_cells + ui * wy + uj]);
    return bounce;
}
#endif

extern "C" __global__
void lbm_trt_step(const pop_t* __restrict__ g_in, pop_t* __restrict__ g_out,
                  float* __restrict__ velocity_x, float* __restrict__ velocity_y,
                  float* __restrict__ density_deviation, const bool* __restrict__ fluid,
                  const float* __restrict__ omega_plus, const float* __restrict__ omega_minus,
                  const float* __restrict__ force_x, const float* __restrict__ force_y,
                  const int* __restrict__ face_kind, const float* __restrict__ face_values,
                  const int wx, const int wy, const int face_length,
                  const int* __restrict__ wall_row, const float* __restrict__ wall_fraction)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    const int n_cells = wx * wy;
    if (n >= n_cells || !fluid[n]) return;
    const int i = n / wy;
    const int j = n - i * wy;
    // literal tables: indexed by the unrolled q only, so they fold into immediates (no local memory)
    const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    const int opposite[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    const float w[9] = {4.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f, 1.f / 9.f,
                        1.f / 36.f, 1.f / 36.f, 1.f / 36.f, 1.f / 36.f};
    const bool periodic_x = face_kind[0] == 4;
    const bool periodic_y = face_kind[2] == 4;
    const float wp = omega_plus[n];
    const float wm = omega_minus[n];
    float g[9];
#pragma unroll
    for (int q = 0; q < 9; ++q) {
        int si = i - cx[q];
        int sj = j - cy[q];
        if (periodic_x) si = wrap_once(si, wx);
        if (periodic_y) sj = wrap_once(sj, wy);
        if (si >= 0 && si < wx && sj >= 0 && sj < wy) {
            const int s = si * wy + sj;
            if (fluid[s]) {
                g[q] = POP_LOAD(g_in[q * n_cells + s]);
            } else {
                const float bounce = POP_LOAD(g_in[opposite[q] * n_cells + n]);
#if BOUZIDI
                g[q] = bouzidi_link(g_in, fluid, wall_row, wall_fraction, q, i, j, n, wx, wy, periodic_x, periodic_y,
                                    bounce);
#else
                g[q] = bounce;   // halfway bounce-back
#endif
            }
        } else {
            g[q] = face_link(g_in, velocity_x, velocity_y, fluid, force_x, force_y, face_kind, face_values, wp, q, i,
                             j, si, sj, n, wx, wy, face_length, periodic_x, periodic_y);
        }
    }
    float deviation = 0.f, momentum_x = 0.f, momentum_y = 0.f;
#pragma unroll
    for (int q = 0; q < 9; ++q) {
        deviation += g[q];
        momentum_x += cx[q] * g[q];
        momentum_y += cy[q] * g[q];
    }
    const float fx = force_x[n];
    const float fy = force_y[n];
    const float ux = momentum_x + 0.5f * fx;
    const float uy = momentum_y + 0.5f * fy;
    const float u_squared = ux * ux + uy * uy;
    const float u_dot_force = ux * fx + uy * fy;
    const float force_scale_plus = 1.f - 0.5f * wp;
    const float force_scale_minus = 1.f - 0.5f * wm;
    g_out[n] = POP_STORE(g[0] - wp * (g[0] - w[0] * (deviation - 1.5f * u_squared))
                         + force_scale_plus * w[0] * (-3.f * u_dot_force));
    const int pair_first[4] = {1, 2, 5, 6};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int q = pair_first[k];
        const int o = opposite[q];
        const float cu = cx[q] * ux + cy[q] * uy;
        const float cf = cx[q] * fx + cy[q] * fy;
        const float equilibrium_plus = w[q] * (deviation + 4.5f * cu * cu - 1.5f * u_squared);
        const float equilibrium_minus = w[q] * 3.f * cu;
        const float relax_plus = -wp * (0.5f * (g[q] + g[o]) - equilibrium_plus)
                                 + force_scale_plus * w[q] * (9.f * cu * cf - 3.f * u_dot_force);
        const float relax_minus = -wm * (0.5f * (g[q] - g[o]) - equilibrium_minus)
                                  + force_scale_minus * w[q] * 3.f * cf;
        g_out[q * n_cells + n] = POP_STORE(g[q] + relax_plus + relax_minus);
        g_out[o * n_cells + n] = POP_STORE(g[o] + relax_plus - relax_minus);
    }
    velocity_x[n] = ux;
    velocity_y[n] = uy;
    density_deviation[n] = deviation;
}
'''

_STEP_KERNELS = {}
# half_storage: populations stored as fp16 x this power of two (|g| <~ 0.1 -> stored <~ 1e2, far below the fp16 max;
# keeps g down to ~6e-8 normal)
POPULATION_SCALE = 1024.0
POPULATION_DTYPES = {'single': np.float32, 'half_storage': np.float16}

# ---- numpy twin of lbm_trt_step (same arguments; Python ints / float32 tables keep the arithmetic float32) ----
_CX = tuple(int(value) for value in DIRECTION_X)
_CY = tuple(int(value) for value in DIRECTION_Y)
_OPPOSITE = (0, 3, 4, 1, 2, 7, 8, 5, 6)
_MIRROR_X = (0, 3, 2, 1, 4, 6, 5, 8, 7)
_MIRROR_Y = (0, 1, 4, 3, 2, 8, 7, 6, 5)
_WEIGHTS32 = tuple(np.float32(value) for value in WEIGHTS)
_PAIR_FIRST = (1, 2, 5, 6)
_ZERO = np.float32(0.0)


def _padded(array, periodic_x, periodic_y):
    """array (..., wx, wy) padded by one cell on both window axes: wrapped on a periodic axis, 0 / False else."""
    lead = [(0, 0)] * (array.ndim - 2)
    array = np.pad(array, lead + [(1, 1), (0, 0)], mode='wrap' if periodic_x else 'constant')
    return np.pad(array, lead + [(0, 0), (1, 1)], mode='wrap' if periodic_y else 'constant')


def _shifted(padded, di, dj, wx, wy):
    """Window view of a _padded array at offset (di, dj): element (i, j) is the value of cell (i + di, j + dj)."""
    return padded[..., 1 + di:1 + di + wx, 1 + dj:1 + dj + wy]


def _neighbour_velocity_host(lattice_velocity, fluid, i, j, wx, wy, periodic_x, periodic_y):
    """(valid, u_x, u_y) at window cells (i, j) (element arrays) of the kernel's neighbour_velocity: valid False outside
    the window or in solids (their values are those of a clamped cell)."""
    if periodic_x:
        i = i % wx
    if periodic_y:
        j = j % wy
    inside = (i >= 0) & (i < wx) & (j >= 0) & (j < wy)
    clamped_i, clamped_j = np.clip(i, 0, wx - 1), np.clip(j, 0, wy - 1)
    valid = inside & fluid[clamped_i, clamped_j]
    return valid, lattice_velocity[0][clamped_i, clamped_j], lattice_velocity[1][clamped_i, clamped_j]


def _face_velocity_gradient_host(lattice_velocity, fluid, face, i, j, side, old_x, old_y, wx, wy, periodic_x,
                                 periodic_y):
    """(du_x/dx, du_y/dx, du_x/dy, du_y/dy) float32 element arrays of the kernel's face_velocity_gradient."""
    def at(di, dj):
        return _neighbour_velocity_host(lattice_velocity, fluid, i + di, j + dj, wx, wy, periodic_x, periodic_y)
    x_face = face < 2
    ti, tj = np.where(x_face, 0, 1), np.where(x_face, 1, 0)
    inward = np.where(face % 2 == 0, 1, -1)
    ni, nj = np.where(x_face, inward, 0), np.where(x_face, 0, inward)
    side_ok, *u_side = at(side * ti, side * tj)
    has_side = (side != 0) & side_ok
    plus_ok, *u_plus = at(ti, tj)
    minus_ok, *u_minus = at(-ti, -tj)
    has_plus, has_minus = ~has_side & plus_ok, ~has_side & minus_ok
    plus2_ok, *u_plus2 = at(2 * ti, 2 * tj)
    minus2_ok, *u_minus2 = at(-2 * ti, -2 * tj)
    inward_ok, *u_inward = at(ni, nj)
    corner_ok, *u_corner = at(ni + side * ti, nj + side * tj)
    side_f, inward_f = side.astype(np.float32), inward.astype(np.float32)
    tangential, normal = [], []
    for c, u0 in enumerate((old_x, old_y)):
        # the kernel's if / else chain: side link, central, one-sided second order, one-sided first order
        tangential.append(np.select(
            [has_side, has_plus & has_minus, has_plus & plus2_ok, has_plus, has_minus & minus2_ok, has_minus],
            [side_f * (u_side[c] - u0), 0.5 * (u_plus[c] - u_minus[c]),
             0.5 * (-3.0 * u0 + 4.0 * u_plus[c] - u_plus2[c]), u_plus[c] - u0,
             0.5 * (3.0 * u0 - 4.0 * u_minus[c] + u_minus2[c]), u0 - u_minus[c]], _ZERO))
        value = np.where(inward_ok, inward_f * (u_inward[c] - u0), _ZERO)
        normal.append(np.where(inward_ok & has_side & corner_ok,
                               0.5 * (value + inward_f * (u_corner[c] - u_side[c])), value))
    return (np.where(x_face, -tangential[1], tangential[0]), np.where(x_face, normal[1], tangential[1]),
            np.where(x_face, tangential[0], normal[0]), np.where(x_face, tangential[1], -tangential[0]))


def _face_link_values(q, populations, fluid, lattice_velocity, kinds, face_values, velocity_x, velocity_y,
                      omega_plus, i, j, wx, wy, periodic_x, periodic_y):
    """Gathered g_q of fluid cells (i, j) whose link q leaves the window: the kernel's face rules (wall / velocity,
    slip, pressure with the TRT shear term)."""
    cx, cy, opposite, weight = _CX[q], _CY[q], _OPPOSITE[q], _WEIGHTS32[q]
    source_i, source_j = i - cx, j - cy
    if periodic_x:
        source_i = (source_i + wx) % wx
    if periodic_y:
        source_j = (source_j + wy) % wy
    out_x = (source_i < 0) | (source_i >= wx)
    out_y = (source_j < 0) | (source_j >= wy)
    face_x, face_y = np.where(source_i < 0, 0, 1), np.where(source_j < 0, 2, 3)
    face = np.where(out_x & out_y, np.where(kinds[face_y] <= FACE_SLIP, face_y, face_x),
                    np.where(out_x, face_x, face_y))
    kind = kinds[face]
    x_face = face < 2
    position = np.where(x_face, j, i)
    bounce = populations[opposite, i, j]
    # wall / velocity faces: moving bounce-back with the face velocity
    values = bounce + 6.0 * weight * (cx * face_values[face, 0, position] + cy * face_values[face, 1, position])
    # slip: the mirrored population of the upstream cell along the face (own cell where that is outside / solid)
    mirror_i, mirror_j = np.where(x_face, i, source_i), np.where(x_face, source_j, j)
    upstream = (mirror_i >= 0) & (mirror_i < wx) & (mirror_j >= 0) & (mirror_j < wy)
    upstream &= fluid[np.clip(mirror_i, 0, wx - 1), np.clip(mirror_j, 0, wy - 1)]
    mirrored = np.where(x_face, _MIRROR_X[q], _MIRROR_Y[q])
    values = np.where(kind == FACE_SLIP, populations[mirrored, np.where(upstream, mirror_i, i),
                                                     np.where(upstream, mirror_j, j)], values)
    pressure = np.nonzero(kind == FACE_PRESSURE)[0]
    if pressure.size:
        cell_i, cell_j, pressure_face = i[pressure], j[pressure], face[pressure]
        side = np.where(pressure_face < 2, -cy, -cx)   # link midpoint offset along the face
        old_x, old_y = velocity_x[cell_i, cell_j], velocity_y[cell_i, cell_j]
        gradient = _face_velocity_gradient_host(lattice_velocity, fluid, pressure_face, cell_i, cell_j, side, old_x,
                                                old_y, wx, wy, periodic_x, periodic_y)
        cu = cx * old_x + cy * old_y
        shear = cx * cx * gradient[0] + cx * cy * (gradient[1] + gradient[2]) + cy * cy * gradient[3]
        values[pressure] = (-bounce[pressure]
                            + 2.0 * weight * (face_values[pressure_face, 2, position[pressure]] + 4.5 * cu * cu
                                              - 1.5 * (old_x * old_x + old_y * old_y))
                            - 6.0 * (1.0 / omega_plus[cell_i, cell_j] - 0.5) * weight * shear)
    return values


def _step_host(precision, bouzidi):
    """numpy twin of lbm_trt_step at a stepper precision (stored = g x POPULATION_SCALE under half_storage)."""
    scale = POPULATION_SCALE if precision == 'half_storage' else 1.0

    def lbm_trt_step_host(g_in, g_out, velocity_x, velocity_y, density_deviation, fluid, omega_plus, omega_minus,
                          force_x, force_y, face_kind, face_values, wx, wy, face_length, wall_row, wall_fraction):
        wx, wy = int(wx), int(wy)
        kinds = np.asarray(face_kind, dtype=np.int64)
        periodic_x, periodic_y = bool(kinds[0] == FACE_PERIODIC), bool(kinds[2] == FACE_PERIODIC)
        populations = g_in if scale == 1.0 else g_in.astype(np.float32) * np.float32(1.0 / scale)
        lattice_velocity = None
        if (kinds == FACE_PRESSURE).any():   # neighbour velocities of the pressure-face shear term
            g = populations
            lattice_velocity = (g[1] - g[3] + g[5] - g[6] - g[7] + g[8] - 0.5 * force_x,
                                g[2] - g[4] + g[5] + g[6] - g[7] - g[8] - 0.5 * force_y)
        padded = _padded(populations, periodic_x, periodic_y)
        padded_fluid = _padded(fluid, periodic_x, periodic_y)
        if bouzidi:
            has_row = wall_row >= 0
            link_fractions = wall_fraction.reshape(-1, N_DIRECTIONS - 1)[np.maximum(wall_row, 0)]
        rows, columns = np.arange(wx), np.arange(wy)
        gathered = np.empty((N_DIRECTIONS, wx, wy), dtype=np.float32)
        for q in range(N_DIRECTIONS):
            cx, cy, opposite = _CX[q], _CY[q], _OPPOSITE[q]
            bounce = populations[opposite]
            wall = bounce   # halfway bounce-back off a solid in-window neighbour
            if bouzidi and q:
                delta = link_fractions[..., q - 1]
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    far = (0.5 / delta) * bounce + (1.0 - 0.5 / delta) * populations[q]
                upstream = _shifted(padded[opposite], cx, cy, wx, wy)
                near = np.where(_shifted(padded_fluid, cx, cy, wx, wy),
                                2.0 * delta * bounce + (1.0 - 2.0 * delta) * upstream, bounce)
                wall = np.where(has_row, np.where(delta >= 0.5, far, near), bounce)
            gathered[q] = np.where(_shifted(padded_fluid, -cx, -cy, wx, wy), _shifted(padded[q], -cx, -cy, wx, wy),
                                   wall)
            out_rows = np.zeros(wx, dtype=bool) if periodic_x else (rows - cx < 0) | (rows - cx >= wx)
            out_columns = np.zeros(wy, dtype=bool) if periodic_y else (columns - cy < 0) | (columns - cy >= wy)
            face_i, face_j = np.nonzero(fluid & (out_rows[:, None] | out_columns[None, :]))
            if face_i.size:
                gathered[q, face_i, face_j] = _face_link_values(
                    q, populations, fluid, lattice_velocity, kinds, face_values, velocity_x, velocity_y, omega_plus,
                    face_i, face_j, wx, wy, periodic_x, periodic_y)
        g = gathered
        deviation = g[0]
        for q in range(1, N_DIRECTIONS):
            deviation = deviation + g[q]
        ux = g[1] - g[3] + g[5] - g[6] - g[7] + g[8] + 0.5 * force_x
        uy = g[2] - g[4] + g[5] + g[6] - g[7] - g[8] + 0.5 * force_y
        u_squared = ux * ux + uy * uy
        u_dot_force = ux * force_x + uy * force_y
        wp, wm = omega_plus, omega_minus
        force_scale_plus = 1.0 - 0.5 * wp
        force_scale_minus = 1.0 - 0.5 * wm

        def store(q, value):
            if scale != 1.0:
                value = (value * np.float32(scale)).astype(g_out.dtype)
            g_out[q][fluid] = value[fluid]

        w0 = _WEIGHTS32[0]
        store(0, g[0] - wp * (g[0] - w0 * (deviation - 1.5 * u_squared))
              + force_scale_plus * w0 * (-3.0 * u_dot_force))
        for q in _PAIR_FIRST:
            o, cx, cy, weight = _OPPOSITE[q], _CX[q], _CY[q], _WEIGHTS32[q]
            cu = cx * ux + cy * uy
            cf = cx * force_x + cy * force_y
            equilibrium_plus = weight * (deviation + 4.5 * cu * cu - 1.5 * u_squared)
            equilibrium_minus = weight * 3.0 * cu
            relax_plus = (-wp * (0.5 * (g[q] + g[o]) - equilibrium_plus)
                          + force_scale_plus * weight * (9.0 * cu * cf - 3.0 * u_dot_force))
            relax_minus = -wm * (0.5 * (g[q] - g[o]) - equilibrium_minus) + force_scale_minus * weight * 3.0 * cf
            store(q, g[q] + relax_plus + relax_minus)
            store(o, g[o] + relax_plus - relax_minus)
        velocity_x[fluid] = ux[fluid]
        velocity_y[fluid] = uy[fluid]
        density_deviation[fluid] = deviation[fluid]

    return lbm_trt_step_host


def _step_numba(bouzidi):
    """numba factory of lbm_trt_step at precision 'single': a per-cell loop transcribing the CUDA source (float32
    constants and tables, so the arithmetic stays float32), prange over i."""
    def factory(numba):
        cx_int, cy_int = np.array(_CX, dtype=np.int64), np.array(_CY, dtype=np.int64)
        cx_real, cy_real = cx_int.astype(np.float32), cy_int.astype(np.float32)
        opposite_table = np.array(_OPPOSITE, dtype=np.int64)
        mirror_x_table, mirror_y_table = np.array(_MIRROR_X, dtype=np.int64), np.array(_MIRROR_Y, dtype=np.int64)
        weight_table = np.array(_WEIGHTS32, dtype=np.float32)
        pair_table = np.array(_PAIR_FIRST, dtype=np.int64)
        zero, half, one, two = np.float32(0.0), np.float32(0.5), np.float32(1.0), np.float32(2.0)
        three, four, six, nine = np.float32(3.0), np.float32(4.0), np.float32(6.0), np.float32(9.0)
        c45, c15 = np.float32(4.5), np.float32(1.5)
        bouzidi_links = bool(bouzidi)

        @numba.njit(cache=False)
        def neighbour_velocity(g_in, fluid, force_x, force_y, i, j, wx, wy, periodic_x, periodic_y):
            if periodic_x:
                i = ((i % wx) + wx) % wx
            if periodic_y:
                j = ((j % wy) + wy) % wy
            if i < 0 or i >= wx or j < 0 or j >= wy:
                return False, zero, zero
            if not fluid[i, j]:
                return False, zero, zero
            ux = (g_in[1, i, j] - g_in[3, i, j] + g_in[5, i, j] - g_in[6, i, j] - g_in[7, i, j] + g_in[8, i, j]
                  - half * force_x[i, j])
            uy = (g_in[2, i, j] - g_in[4, i, j] + g_in[5, i, j] + g_in[6, i, j] - g_in[7, i, j] - g_in[8, i, j]
                  - half * force_y[i, j])
            return True, ux, uy

        @numba.njit(cache=False)
        def face_velocity_gradient(g_in, fluid, force_x, force_y, face, i, j, side, u0x, u0y, wx, wy, periodic_x,
                                   periodic_y):
            x_face = face < 2
            ti = 0 if x_face else 1
            tj = 1 if x_face else 0
            inward = 1 if face % 2 == 0 else -1
            ni = inward if x_face else 0
            nj = 0 if x_face else inward
            side_f, inward_f = np.float32(side), np.float32(inward)
            has_side, side_x, side_y = False, zero, zero
            if side != 0:
                has_side, side_x, side_y = neighbour_velocity(g_in, fluid, force_x, force_y, i + side * ti,
                                                              j + side * tj, wx, wy, periodic_x, periodic_y)
            has_plus, plus_x, plus_y = False, zero, zero
            has_minus, minus_x, minus_y = False, zero, zero
            if not has_side:
                has_plus, plus_x, plus_y = neighbour_velocity(g_in, fluid, force_x, force_y, i + ti, j + tj, wx, wy,
                                                              periodic_x, periodic_y)
                has_minus, minus_x, minus_y = neighbour_velocity(g_in, fluid, force_x, force_y, i - ti, j - tj, wx,
                                                                 wy, periodic_x, periodic_y)
            tangential_x, tangential_y, normal_x, normal_y = zero, zero, zero, zero
            if has_side:
                tangential_x, tangential_y = side_f * (side_x - u0x), side_f * (side_y - u0y)
            elif has_plus and has_minus:
                tangential_x, tangential_y = half * (plus_x - minus_x), half * (plus_y - minus_y)
            elif has_plus:
                ok, far_x, far_y = neighbour_velocity(g_in, fluid, force_x, force_y, i + 2 * ti, j + 2 * tj, wx, wy,
                                                      periodic_x, periodic_y)
                if ok:
                    tangential_x = half * (-three * u0x + four * plus_x - far_x)
                    tangential_y = half * (-three * u0y + four * plus_y - far_y)
                else:
                    tangential_x, tangential_y = plus_x - u0x, plus_y - u0y
            elif has_minus:
                ok, far_x, far_y = neighbour_velocity(g_in, fluid, force_x, force_y, i - 2 * ti, j - 2 * tj, wx, wy,
                                                      periodic_x, periodic_y)
                if ok:
                    tangential_x = half * (three * u0x - four * minus_x + far_x)
                    tangential_y = half * (three * u0y - four * minus_y + far_y)
                else:
                    tangential_x, tangential_y = u0x - minus_x, u0y - minus_y
            ok, inward_x, inward_y = neighbour_velocity(g_in, fluid, force_x, force_y, i + ni, j + nj, wx, wy,
                                                        periodic_x, periodic_y)
            if ok:
                normal_x, normal_y = inward_f * (inward_x - u0x), inward_f * (inward_y - u0y)
                if has_side:
                    corner_ok, corner_x, corner_y = neighbour_velocity(
                        g_in, fluid, force_x, force_y, i + ni + side * ti, j + nj + side * tj, wx, wy, periodic_x,
                        periodic_y)
                    if corner_ok:
                        normal_x = half * (normal_x + inward_f * (corner_x - side_x))
                        normal_y = half * (normal_y + inward_f * (corner_y - side_y))
            if x_face:
                return -tangential_y, normal_y, tangential_x, tangential_y
            return tangential_x, tangential_y, normal_x, -tangential_x

        @numba.njit(parallel=True, cache=False)
        def kernel(g_in, g_out, velocity_x, velocity_y, density_deviation, fluid, omega_plus, omega_minus, force_x,
                   force_y, face_kind, face_values, wx, wy, wall_row, wall_fraction):
            periodic_x = face_kind[0] == FACE_PERIODIC
            periodic_y = face_kind[2] == FACE_PERIODIC
            for prange_i in numba.prange(wx):
                i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
                g = np.empty(N_DIRECTIONS, dtype=np.float32)
                for j in range(wy):
                    if not fluid[i, j]:
                        continue
                    old_x, old_y = velocity_x[i, j], velocity_y[i, j]
                    wp, wm = omega_plus[i, j], omega_minus[i, j]
                    for q in range(N_DIRECTIONS):
                        si, sj = i - cx_int[q], j - cy_int[q]
                        if periodic_x:
                            si = (si + wx) % wx
                        if periodic_y:
                            sj = (sj + wy) % wy
                        out_x = si < 0 or si >= wx
                        out_y = sj < 0 or sj >= wy
                        opposite = opposite_table[q]
                        if not out_x and not out_y:
                            if fluid[si, sj]:
                                g[q] = g_in[q, si, sj]
                                continue
                            bounce = g_in[opposite, i, j]
                            value = bounce   # halfway bounce-back
                            if bouzidi_links:
                                row = wall_row[i * wy + j]
                                if row >= 0:
                                    delta = wall_fraction[row * (N_DIRECTIONS - 1) + q - 1]
                                    if delta >= half:
                                        value = (half / delta) * bounce + (one - half / delta) * g_in[q, i, j]
                                    else:
                                        ui, uj = i + cx_int[q], j + cy_int[q]
                                        if periodic_x:
                                            ui = (ui + wx) % wx
                                        if periodic_y:
                                            uj = (uj + wy) % wy
                                        if ui >= 0 and ui < wx and uj >= 0 and uj < wy and fluid[ui, uj]:
                                            value = two * delta * bounce + (one - two * delta) * g_in[opposite, ui, uj]
                            g[q] = value
                            continue
                        if out_x and out_y:
                            face_y = 2 if sj < 0 else 3
                            face = face_y if face_kind[face_y] <= FACE_SLIP else (0 if si < 0 else 1)
                        elif out_x:
                            face = 0 if si < 0 else 1
                        else:
                            face = 2 if sj < 0 else 3
                        kind = face_kind[face]
                        position = j if face < 2 else i
                        if kind == FACE_SLIP:
                            mi = i if face < 2 else si
                            mj = sj if face < 2 else j
                            mirrored = mirror_x_table[q] if face < 2 else mirror_y_table[q]
                            if mi >= 0 and mi < wx and mj >= 0 and mj < wy and fluid[mi, mj]:
                                g[q] = g_in[mirrored, mi, mj]
                            else:
                                g[q] = g_in[mirrored, i, j]
                        elif kind == FACE_PRESSURE:
                            side = -cy_int[q] if face < 2 else -cx_int[q]
                            gradient_0, gradient_1, gradient_2, gradient_3 = face_velocity_gradient(
                                g_in, fluid, force_x, force_y, face, i, j, side, old_x, old_y, wx, wy, periodic_x,
                                periodic_y)
                            cxq, cyq, weight = cx_real[q], cy_real[q], weight_table[q]
                            cu = cxq * old_x + cyq * old_y
                            shear = (cxq * cxq * gradient_0 + cxq * cyq * (gradient_1 + gradient_2)
                                     + cyq * cyq * gradient_3)
                            g[q] = (-g_in[opposite, i, j]
                                    + two * weight * (face_values[face, 2, position] + c45 * cu * cu
                                                      - c15 * (old_x * old_x + old_y * old_y))
                                    - six * (one / wp - half) * weight * shear)
                        else:
                            cu = cx_real[q] * face_values[face, 0, position] + cy_real[q] * face_values[face, 1, position]
                            g[q] = g_in[opposite, i, j] + six * weight_table[q] * cu
                    deviation = g[0]
                    for q in range(1, N_DIRECTIONS):
                        deviation += g[q]
                    fx, fy = force_x[i, j], force_y[i, j]
                    ux = g[1] - g[3] + g[5] - g[6] - g[7] + g[8] + half * fx
                    uy = g[2] - g[4] + g[5] + g[6] - g[7] - g[8] + half * fy
                    u_squared = ux * ux + uy * uy
                    u_dot_force = ux * fx + uy * fy
                    force_scale_plus = one - half * wp
                    force_scale_minus = one - half * wm
                    w0 = weight_table[0]
                    g_out[0, i, j] = (g[0] - wp * (g[0] - w0 * (deviation - c15 * u_squared))
                                      + force_scale_plus * w0 * (-three * u_dot_force))
                    for k in range(4):
                        q = pair_table[k]
                        o = opposite_table[q]
                        cxq, cyq, weight = cx_real[q], cy_real[q], weight_table[q]
                        cu = cxq * ux + cyq * uy
                        cf = cxq * fx + cyq * fy
                        equilibrium_plus = weight * (deviation + c45 * cu * cu - c15 * u_squared)
                        equilibrium_minus = weight * three * cu
                        relax_plus = (-wp * (half * (g[q] + g[o]) - equilibrium_plus)
                                      + force_scale_plus * weight * (nine * cu * cf - three * u_dot_force))
                        relax_minus = (-wm * (half * (g[q] - g[o]) - equilibrium_minus)
                                       + force_scale_minus * weight * three * cf)
                        g_out[q, i, j] = g[q] + relax_plus + relax_minus
                        g_out[o, i, j] = g[o] + relax_plus - relax_minus
                    velocity_x[i, j] = ux
                    velocity_y[i, j] = uy
                    density_deviation[i, j] = deviation

        def lbm_trt_step_numba(g_in, g_out, velocity_x, velocity_y, density_deviation, fluid, omega_plus, omega_minus,
                               force_x, force_y, face_kind, face_values, wx, wy, face_length, wall_row, wall_fraction):
            # flat wall tables: the (wx, wy) rows / (n_rows, 8) fractions and the 1D halfway stand-ins alike
            kernel(g_in, g_out, velocity_x, velocity_y, density_deviation, fluid, omega_plus, omega_minus, force_x,
                   force_y, face_kind, face_values, int(wx), int(wy), wall_row.reshape(-1), wall_fraction.reshape(-1))
        return lbm_trt_step_numba
    return factory


def step_kernel(precision='single', bouzidi=False):
    """lbm_trt_step twin (core.kernels raw_twin: RawKernel + numpy twin, numba twin in 'single') of a stepper precision
    ('single' | 'half_storage'), with the interpolated wall links read (bouzidi; else wall_row / wall_fraction are
    ignored), built and registered on first use."""
    key = (precision, bool(bouzidi))
    if key not in _STEP_KERNELS:
        preamble = storage_preamble(precision, 'pop_t', 'POP_LOAD', 'POP_STORE',
                                    scale=POPULATION_SCALE if precision == 'half_storage' else 1.0)
        gpu_kernel = cp.RawKernel(f'#define BOUZIDI {int(bool(bouzidi))}\n' + preamble + _STEP_SOURCE, 'lbm_trt_step')
        # numba for 'single' only: half_storage stays gpu-only (explicit_kernels backend_misfit), numba has no CPU fp16
        _STEP_KERNELS[key] = raw_twin(f"fluid.lbm_trt_step_{precision}{'_bouzidi' if bouzidi else ''}", gpu_kernel,
                                      _step_host(precision, bouzidi),
                                      numba=_step_numba(bouzidi) if precision == 'single' else None)
    return _STEP_KERNELS[key]


# Lattice -> SI on the window (outputs may be strided views of the full-grid fields): u = u_lat dx/dt, pressure =
# (rho - 1) c_s^2 rho_SI (dx/dt)^2 (pressure_scale = (dx/dt)^2 / 3), all 0 in solid cells.
_to_physical_gpu = cp.ElementwiseKernel(
    'float32 lattice_x, float32 lattice_y, float32 deviation, bool fluid, float32 mass_density, '
    'float32 velocity_scale, float32 pressure_scale',
    'float32 velocity_x, float32 velocity_y, float32 speed, float32 pressure',
    '''
    if (fluid) {
        velocity_x = lattice_x * velocity_scale;
        velocity_y = lattice_y * velocity_scale;
        speed = sqrtf(velocity_x * velocity_x + velocity_y * velocity_y);
        pressure = deviation * pressure_scale * mass_density;
    } else {
        velocity_x = 0.0f; velocity_y = 0.0f; speed = 0.0f; pressure = 0.0f;
    }
    ''',
    'fluid_lattice_to_physical')


def _to_physical_host(lattice_x, lattice_y, deviation, fluid, mass_density, velocity_scale, pressure_scale,
                      velocity_x, velocity_y, speed, pressure):
    scaled_x, scaled_y = lattice_x * velocity_scale, lattice_y * velocity_scale
    velocity_x[...] = np.where(fluid, scaled_x, _ZERO)
    velocity_y[...] = np.where(fluid, scaled_y, _ZERO)
    speed[...] = np.where(fluid, np.sqrt(scaled_x * scaled_x + scaled_y * scaled_y), _ZERO)
    pressure[...] = np.where(fluid, deviation * pressure_scale * mass_density, _ZERO)


to_physical = twin('fluid.lattice_to_physical', _to_physical_gpu, _to_physical_host)

# SI force density [N/m^3] -> lattice force (acceleration dt^2 / dx): scale = dt^2 / dx; 0 in solid cells.
_force_to_lattice_gpu = cp.ElementwiseKernel(
    'float32 coupled, float32 applied, bool fluid, float32 mass_density, float32 scale',
    'float32 lattice_force',
    'lattice_force = fluid ? (coupled + applied) * scale / mass_density : 0.0f',
    'fluid_force_to_lattice')


def _force_to_lattice_host(coupled, applied, fluid, mass_density, scale, lattice_force):
    density = np.where(fluid, mass_density, np.float32(1.0))
    lattice_force[...] = np.where(fluid, (coupled + applied) * scale / density, _ZERO)


force_to_lattice = twin('fluid.force_to_lattice', _force_to_lattice_gpu, _force_to_lattice_host)


# Host-built populations into the stored buffer: stored = g x scale; T = float32, or the uint16 bits of the float16
# populations under half_storage (core.half_storage.half_bits; cupy's float16 casts do not compile everywhere).
def _store_populations_host(values, scale, stored):
    stored[...] = values * scale


store_populations = twin('fluid.store_populations', cp.ElementwiseKernel(
    'float32 value, float32 scale', 'T stored', 'stored_set(stored, value * scale)', 'fluid_store_populations',
    preamble=STORED_ACCESS), _store_populations_host)


def equilibrium_populations(deviation, lattice_x, lattice_y):
    """Shifted He-Luo equilibrium g_q = w_q (rho - 1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u^2), shape (9, *field shape), on the
    inputs' array module."""
    populations = []
    u_squared = lattice_x * lattice_x + lattice_y * lattice_y
    for q in range(N_DIRECTIONS):
        cu = float(DIRECTION_X[q]) * lattice_x + float(DIRECTION_Y[q]) * lattice_y
        populations.append(float(WEIGHTS[q]) * (deviation + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_squared))
    return array_module_of(deviation).stack(populations)
