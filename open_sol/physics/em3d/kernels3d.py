"""Fused kernels of the 3D Yee engine (layout and conventions: engine3d header): H and E updates with the
material-index coefficient lookup, CPML psi updates on the layer slabs, the ohmic J.E loss and the sparse conformal-PEC
H corrections (per stepper precision: field_kernels), face cb of a box and the weighted pole updates (float32 fields
only). Each is a core.kernels twin
(numpy twin of the same call below its cupy source, numba twins of the single-precision field kernels): a change to a
cupy source changes its twins."""
import weakref

import numpy as np

from ...core.backend import cp

from ...core.constants import EPS0, MU0
from ...core.half_storage import storage_preamble
from ...core.kernels import raw_twin, twin
from .boundaries3d import along

_BLOCK = (32, 4, 2)   # CUDA block (x, y, z) = threads along the grid's (z, y, x): z is the contiguous axis
_EPS0_F32 = np.float32(EPS0)
_MU0_F32 = np.float32(MU0)
_ONE = np.float32(1.0)
_TWO = np.float32(2.0)

# Shared device helpers (idx_t = 64-bit flat index: `long` is 32 bit on Windows). E coefficients are computed per step
# from the uint16 material index and the small per-id tables (eps_x / eps_y / eps_z per E component, sigma_e, mu_r,
# sigma_m): a face takes the width-weighted mean (wl, wu = yee_grid.face_shares) of its two cells' table of its own
# component; a PEC id has eps < 0: a face between two PEC cells is pinned (ca = cb = 0), a face between a PEC and
# another cell takes the other cell's values (normal E on a conductor is free). H edges take their lower cell's mu_r and
# sigma_m (edge_coefficients: da, db of the semi-implicit magnetic loss).
_PREAMBLE = f"""
#define EPS0F {float(EPS0)!r}f
#define MU0F {float(MU0)!r}f
typedef unsigned short mat_t;
typedef long long idx_t;

__device__ __forceinline__ idx_t hi_offset(int index, int n, int periodic, idx_t stride)
{{
    if (index + 1 < n) return stride;
    return periodic ? -(idx_t)(n - 1) * stride : (idx_t)0;
}}

__device__ __forceinline__ void edge_coefficients(
    const mat_t* __restrict__ mat, const float* __restrict__ t_mu, const float* __restrict__ t_sm,
    idx_t idx, float dt, float* da, float* db)
{{
    mat_t id = mat[idx];
    float mu = t_mu[id] * MU0F;
    float k = t_sm[id] * dt / (2.0f * mu);
    float den = 1.0f + k;
    *da = (1.0f - k) / den;
    *db = (dt / mu) / den;
}}

// Subpixel smoothing (igor 1.5, subpixel_smoothing3d): fid = uint16 face override id of the lower cell (0 none),
// t_face = (ids + 1) x 3 averaged eps per component (0 keeps the mean); use_face 0 = off (fid / t_face unread).
#define FACE_PARAMS const mat_t* __restrict__ fid, const float* __restrict__ t_face, int use_face
#define FACE_ARGS fid, t_face, use_face

__device__ __forceinline__ void face_coefficients(
    const mat_t* __restrict__ mat, const float* __restrict__ t_eps, const float* __restrict__ t_sig, FACE_PARAMS,
    int comp, idx_t a, idx_t b, float wa, float wb, float dt, float* ca, float* cb)
{{
    mat_t ia = mat[a], ib = mat[b];
    float ea = t_eps[ia], eb = t_eps[ib];
    float sa = t_sig[ia], sb = t_sig[ib];
    if (ea < 0.0f && eb < 0.0f) {{ *ca = 0.0f; *cb = 0.0f; return; }}
    bool touches_pec = ea < 0.0f || eb < 0.0f;
    if (ea < 0.0f) {{ ea = eb; sa = sb; }}
    else if (eb < 0.0f) {{ eb = ea; sb = sa; }}
    float eps = (wa * ea + wb * eb) * EPS0F;
    if (use_face && !touches_pec) {{
        mat_t face = fid[a];
        if (face) {{
            float smoothed = t_face[3 * (idx_t)face + comp];
            if (smoothed > 0.0f) eps = smoothed * EPS0F;
        }}
    }}
    float sig = wa * sa + wb * sb;
    float k = sig * dt / (2.0f * eps);
    float den = 1.0f + k;
    *ca = (1.0f - k) / den;
    *cb = (dt / eps) / den;
}}

__device__ __forceinline__ void component_coefficients(
    int comp, int i, int j, int k, idx_t idx, int nx, int ny, int nz, int px, int py, int pz,
    const mat_t* __restrict__ mat, const float* __restrict__ t_ex, const float* __restrict__ t_ey,
    const float* __restrict__ t_ez, const float* __restrict__ t_sig, FACE_PARAMS,
    const float* __restrict__ wlx, const float* __restrict__ wux,
    const float* __restrict__ wly, const float* __restrict__ wuy,
    const float* __restrict__ wlz, const float* __restrict__ wuz, float dt, float* ca, float* cb)
{{
    idx_t sy = nz, sx = (idx_t)ny * nz;
    if (comp == 0)
        face_coefficients(mat, t_ex, t_sig, FACE_ARGS, 0, idx, idx + hi_offset(i, nx, px, sx), wlx[i], wux[i], dt,
                          ca, cb);
    else if (comp == 1)
        face_coefficients(mat, t_ey, t_sig, FACE_ARGS, 1, idx, idx + hi_offset(j, ny, py, sy), wly[j], wuy[j], dt,
                          ca, cb);
    else
        face_coefficients(mat, t_ez, t_sig, FACE_ARGS, 2, idx, idx + hi_offset(k, nz, pz, (idx_t)1), wlz[k], wuz[k],
                          dt, ca, cb);
}}
"""


# ---- host twin helpers (numpy, float32 arithmetic as the device helpers above) ----
def _row_shape(axis):
    """Broadcast shape of a 1D row array along axis."""
    shape = [1, 1, 1]
    shape[axis] = -1
    return tuple(shape)


def _hi_difference(field, partner, axis, cosine, sine):
    """float32 F[hi neighbour] - F along axis; the last row reads the wrap c F[0] + s F_partner[0] (wrap_load)."""
    n = field.shape[axis]
    out = np.empty(field.shape, dtype=np.float32)
    np.subtract(field[along(axis, slice(1, n))], field[along(axis, slice(0, n - 1))],
                out=out[along(axis, slice(0, n - 1))])
    first, last = along(axis, slice(0, 1)), along(axis, slice(n - 1, n))
    out[last] = cosine * field[first] + sine * partner[first] - field[last]
    return out


def _lo_difference(field, partner, axis, periodic, cosine, sine):
    """float32 F - F[lo neighbour] along axis; row 0 reads the wrap c F[n-1] + s F_partner[n-1] on a periodic axis,
    nothing otherwise (the kernels' lx / ly / lz)."""
    n = field.shape[axis]
    out = np.empty(field.shape, dtype=np.float32)
    np.subtract(field[along(axis, slice(1, n))], field[along(axis, slice(0, n - 1))],
                out=out[along(axis, slice(1, n))])
    first, last = along(axis, slice(0, 1)), along(axis, slice(n - 1, n))
    if periodic:
        out[first] = field[first] - (cosine * field[last] + sine * partner[last])
    else:
        out[first] = field[first]
    return out


def pec_face_values(value_lower, value_upper, pec_lower, pec_upper):
    """(lower, upper) cell values of hi faces under the PEC rule of face_coefficients: a face touching one PEC cell takes
    the other cell's value on both sides; two PEC cells keep theirs (the caller pins that face)."""
    only_upper = pec_upper & ~pec_lower
    return np.where(pec_lower, value_upper, value_lower), np.where(only_upper, value_lower, value_upper)


def _edge_tables(t_mu, t_sm, dt):
    """(da, db) float32 per material id (edge_coefficients)."""
    mu = t_mu * _MU0_F32
    loss = t_sm * dt / (_TWO * mu)
    denominator = _ONE + loss
    return (_ONE - loss) / denominator, (dt / mu) / denominator


def _face_coefficients(mat, t_eps, t_sig, fid, t_face, use_face, comp, origin, extents, lower_share, upper_share,
                       periodic, dt):
    """(ca, cb) float32 of E component comp on the box at origin (face_coefficients: the cell and its hi partner
    along comp, PEC rules, subpixel face override)."""
    comp = int(comp)
    n = mat.shape[comp]
    box = tuple(slice(int(start), int(start) + int(count)) for start, count in zip(origin, extents))
    rows = np.arange(box[comp].start, box[comp].stop)
    partners = rows + 1
    wrapped = partners >= n
    partners[wrapped] = 0 if periodic else rows[wrapped]
    partner_box = list(box)
    partner_box[comp] = partners
    ids_a, ids_b = mat[box], mat[tuple(partner_box)]
    eps_a, eps_b, sig_a, sig_b = t_eps[ids_a], t_eps[ids_b], t_sig[ids_a], t_sig[ids_b]
    pec_a, pec_b = eps_a < 0, eps_b < 0
    eps_a, eps_b = pec_face_values(eps_a, eps_b, pec_a, pec_b)
    sig_a, sig_b = pec_face_values(sig_a, sig_b, pec_a, pec_b)
    weight_a = lower_share[rows].reshape(_row_shape(comp))
    weight_b = upper_share[rows].reshape(_row_shape(comp))
    eps = (weight_a * eps_a + weight_b * eps_b) * _EPS0_F32
    if int(use_face):
        face = fid[box]
        smoothed = t_face[3 * face.astype(np.int64) + comp]
        eps = np.where((face != 0) & (smoothed > 0) & ~(pec_a | pec_b), smoothed * _EPS0_F32, eps)
    sig = weight_a * sig_a + weight_b * sig_b
    loss = sig * dt / (_TWO * eps)
    denominator = _ONE + loss
    ca = (_ONE - loss) / denominator
    cb = (dt / eps) / denominator
    pinned = pec_a & pec_b
    ca[pinned] = 0.0
    cb[pinned] = 0.0
    return ca, cb


# Host twin coefficient cache: the face ca / cb, edge da / db and curl-factor rows the device recomputes per step in
# registers cost the numpy twins ~10 full-grid gathers / passes per component, so they are cached under the identity of
# their input arrays (weak references: a replaced or freed array misses, its entries are dropped). Valid because those
# inputs change only in engine3d.rebuild_materials, which writes new table arrays every time; ~32 B per cell of host
# memory on the cpu backend (numba twins do not use it).
_HOST_CACHE = {}


def _drop_dead_entries(_reference=None, cache=_HOST_CACHE):
    """Remove cache entries whose input arrays were freed (weakref callback; the cache is bound as a default since
    interpreter shutdown sets module globals to None before the arrays die)."""
    for key, (references, _) in list(cache.items()):
        if any(reference() is None for reference in references):
            cache.pop(key, None)


def _host_cached(arrays, scalars, compute):
    """compute() cached under the identity of the numpy `arrays` and the hashable `scalars`."""
    key = (tuple(id(array) for array in arrays), scalars)
    entry = _HOST_CACHE.get(key)
    if entry is not None and all(reference() is array for reference, array in zip(entry[0], arrays)):
        return entry[1]
    value = compute()
    _HOST_CACHE[key] = (tuple(weakref.ref(array, _drop_dead_entries) for array in arrays), value)
    return value


def _cached_face_coefficients(mat, t_eps, t_sig, fid, t_face, use_face, comp, origin, extents, lower_share,
                              upper_share, periodic, dt):
    """_face_coefficients through the host cache (same arguments)."""
    scalars = ('face', int(use_face), int(comp), tuple(int(value) for value in origin),
               tuple(int(value) for value in extents), int(periodic), float(dt))
    return _host_cached((mat, t_eps, t_sig, fid, t_face, lower_share, upper_share), scalars,
                        lambda: _face_coefficients(mat, t_eps, t_sig, fid, t_face, use_face, comp, origin, extents,
                                                   lower_share, upper_share, periodic, dt))


def _cached_edge_coefficients(mat, t_mu, t_sm, dt, e_to_h, rows=None):
    """(da, db x e_to_h) float32 of the H edges (of the `rows` index tuple; None = the whole grid), host cached."""
    def compute():
        da_table, db_table = _edge_tables(t_mu, t_sm, dt)
        ids = mat if rows is None else mat[rows]
        return da_table[ids], db_table[ids] * e_to_h
    scalars = ('edge', float(dt), float(e_to_h),
               None if rows is None else tuple((key.start, key.stop) for key in rows))
    return _host_cached((mat, t_mu, t_sm), scalars, compute)


def _row_factor(values, axis):
    """1D float32 per-row curl factor broadcast along axis, None when every entry is 1 (Cartesian: x 1 is exact)."""
    return _host_cached((values,), ('row', int(axis)),
                        lambda: None if bool(np.all(values == 1.0)) else _axis_row(values, axis))


def _times(values, factor):
    """values x factor, values itself for factor None."""
    return values if factor is None else values * factor


def _face_scalar(eps_a, eps_b, sig_a, sig_b, weight_a, weight_b, dt, eps0, smoothed):
    """(ca, cb) of one face (numba twins; smoothed <= 0 = no subpixel override)."""
    if eps_a < 0.0 and eps_b < 0.0:
        return 0.0, 0.0
    touches_pec = eps_a < 0.0 or eps_b < 0.0
    if eps_a < 0.0:
        eps_a = eps_b
        sig_a = sig_b
    elif eps_b < 0.0:
        eps_b = eps_a
        sig_b = sig_a
    eps = (weight_a * eps_a + weight_b * eps_b) * eps0
    if smoothed > 0.0 and not touches_pec:
        eps = smoothed * eps0
    loss = (weight_a * sig_a + weight_b * sig_b) * dt / (2.0 * eps)
    return (1.0 - loss) / (1.0 + loss), (dt / eps) / (1.0 + loss)


def _publish_numba_face_scalar(numba):
    """Builds once the module global _numba_face_scalar (numba _face_scalar) the E numba twins call."""
    # a global, not a closure cell: numba's cache key pickles the closure cells and a dispatcher pickles with a
    # per-process uuid, so a kernel closing over one recompiled in every process (structure_conventions)
    if '_numba_face_scalar' not in globals():
        globals()['_numba_face_scalar'] = numba.njit(cache=True)(_face_scalar)


# H on the edges: H_x (centre x, hi y, hi z), H_y (hi x, centre y, hi z), H_z (hi x, hi y, centre z); mu_r and
# sigma_m of the lower cell (i, j, k). The hi neighbour wraps; on non-periodic axes ih*[n-1] = 0, so the last edge row
# of that axis couples only to the normal E of the same row, a closed subsystem the interior never drives: H stays 0
# there (PMC at the grid edge) unless a source injects into that row.
# Field kernels (H / E updates, CPML, ohmic loss) are compiled per stepper precision (field_kernels): field_t is float
# ('single') or fp16 storage ('half_storage', core.half_storage: FLOAD / FSTORE convert, arithmetic in float). The
# fields are stored in units E / e_scale and H / h_scale (engine3d: both 1 in single, h_scale = e_scale / eta0 in
# half_storage, so stored E and H are alike in size); e_to_h = e_scale / h_scale scales the curl of stored E in the H
# update, h_to_e = 1 / e_to_h the curl of stored H in the E update (exactly 1 in single: unchanged arithmetic).
# Bloch k != 0 (symmetry 2.8, engine3d): two field sets, real and imaginary, each stepped by these kernels; a sample
# read across a periodic wrap mixes in the partner set: value = c F + s F_partner with (c, s) = (cos kL, +-sin kL) per
# axis (H reads hi wraps: s = +sin for the real set, -sin for the imaginary one; E reads lo wraps: the opposite signs).
# A real-only engine passes partner = the set itself, c = 1, s = 0: 1 F + 0 F = F, bit-identical.
# Cylindrical (r, phi, z) sectors (symmetry 2.21; x = r, y = phi [rad]): the integral form of the curls folds the radii
# into per-i 1D factors (engine3d: all 1 on a Cartesian grid). H: H_r's phi term x 1 / r_c (rce / irce = r_c, 1 / r_c
# of the cell centres), H_z's (r_c E_phi) radial difference and phi term x 1 / r_link (irl: the mean radius between
# centres i and i + 1, i.e. the H_z face's exact mean radius). E: E_r's phi term x 1 / r_f[i + 1] (irfh), E_z's (r_f
# H_phi) radial difference (rfh = r_f[i + 1], the hi face of cell i) and phi term x 1 / r_c (irce). The CPML psi of the
# second component along an axis takes the same weights (sw source weight per grid row, of out factor per row; 1 off
# the r axis): the stretched-coordinate PML of d/dr only (no complex-r stretch of the 1 / r factors: approximate).
_WRAP_LOAD = r"""
__device__ __forceinline__ float wrap_load(const field_t* __restrict__ f, const field_t* __restrict__ partner, idx_t o,
                                           bool wrapped, float c, float s)
{
    float value = FLOAD(f[o]);
    return wrapped ? c * value + s * FLOAD(partner[o]) : value;
}
"""

_UPDATE_H_SOURCE = _WRAP_LOAD + r"""
extern "C" __global__
void update_h_3d(
    field_t* __restrict__ hx, field_t* __restrict__ hy, field_t* __restrict__ hz,
    const field_t* __restrict__ ex, const field_t* __restrict__ ey, const field_t* __restrict__ ez,
    const field_t* __restrict__ pex, const field_t* __restrict__ pey, const field_t* __restrict__ pez,
    const mat_t* __restrict__ mat, const float* __restrict__ t_mu, const float* __restrict__ t_sm,
    const float* __restrict__ ihx, const float* __restrict__ ihy, const float* __restrict__ ihz,
    const float* __restrict__ rce, const float* __restrict__ irce, const float* __restrict__ irl,
    int nx, int ny, int nz, float dt, float e_to_h,
    float wcx, float wsx, float wcy, float wsy, float wcz, float wsz)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    idx_t sy = nz, sx = (idx_t)ny * nz;
    idx_t idx = (idx_t)i * sx + (idx_t)j * sy + k;
    bool wx = i + 1 == nx, wy = j + 1 == ny, wz = k + 1 == nz;
    int ip = wx ? 0 : i + 1;
    idx_t dxp = wx ? -(idx_t)(nx - 1) * sx : sx;
    idx_t dyp = wy ? -(idx_t)(ny - 1) * sy : sy;
    idx_t dzp = wz ? -(idx_t)(nz - 1) : (idx_t)1;
    float da, db;
    edge_coefficients(mat, t_mu, t_sm, idx, dt, &da, &db);
    db *= e_to_h;
    float ex0 = FLOAD(ex[idx]), ey0 = FLOAD(ey[idx]), ez0 = FLOAD(ez[idx]);
    hx[idx] = FSTORE(da * FLOAD(hx[idx])
                     - db * ((wrap_load(ez, pez, idx + dyp, wy, wcy, wsy) - ez0) * ihy[j] * irce[i]
                             - (wrap_load(ey, pey, idx + dzp, wz, wcz, wsz) - ey0) * ihz[k]));
    hy[idx] = FSTORE(da * FLOAD(hy[idx])
                     - db * ((wrap_load(ex, pex, idx + dzp, wz, wcz, wsz) - ex0) * ihz[k]
                             - (wrap_load(ez, pez, idx + dxp, wx, wcx, wsx) - ez0) * ihx[i]));
    hz[idx] = FSTORE(da * FLOAD(hz[idx])
                     - db * ((rce[ip] * wrap_load(ey, pey, idx + dxp, wx, wcx, wsx) - rce[i] * ey0) * ihx[i]
                             - (wrap_load(ex, pex, idx + dyp, wy, wcy, wsy) - ex0) * ihy[j]) * irl[i]);
}
"""


def _axis_row(values, axis):
    """1D float32 per-row factors broadcast along axis of a 3D array."""
    return np.asarray(values, dtype=np.float32).reshape(_row_shape(axis))


def _update_h_host(hx, hy, hz, ex, ey, ez, pex, pey, pez, mat, t_mu, t_sm, ihx, ihy, ihz, rce, irce, irl,
                   nx, ny, nz, dt, e_to_h, wcx, wsx, wcy, wsy, wcz, wsz):
    da, db = _cached_edge_coefficients(mat, t_mu, t_sm, dt, e_to_h)
    ihx, ihy, ihz = ihx[:, None, None], ihy[None, :, None], ihz[None, None, :]
    rce, irce, irl = (_row_factor(values, 0) for values in (rce, irce, irl))
    hx[...] = da * hx - db * (_times(_hi_difference(ez, pez, 1, wcy, wsy) * ihy, irce)
                              - _hi_difference(ey, pey, 2, wcz, wsz) * ihz)
    hy[...] = da * hy - db * (_hi_difference(ex, pex, 2, wcz, wsz) * ihz - _hi_difference(ez, pez, 0, wcx, wsx) * ihx)
    ey_r = _times(ey, rce)
    hz[...] = da * hz - db * _times(_hi_difference(ey_r, ey_r if pey is ey else _times(pey, rce), 0, wcx, wsx) * ihx
                                    - _hi_difference(ex, pex, 1, wcy, wsy) * ihy, irl)


def _update_h_numba(numba):
    mu0 = float(MU0)

    @numba.njit(parallel=True, cache=True)
    def kernel(hx, hy, hz, ex, ey, ez, pex, pey, pez, mat, t_mu, t_sm, ihx, ihy, ihz, rce, irce, irl, nx, ny, nz, dt,
               e_to_h, wcx, wsx, wcy, wsy, wcz, wsz):
        # per-material float64 da / db once per call: the cell loop only gathers them (no divisions per cell)
        table_da = np.empty(t_mu.shape[0], dtype=np.float64)
        table_db = np.empty(t_mu.shape[0], dtype=np.float64)
        for material in range(t_mu.shape[0]):
            mu = t_mu[material] * mu0
            loss = t_sm[material] * dt / (2.0 * mu)
            table_da[material] = (1.0 - loss) / (1.0 + loss)
            table_db[material] = (dt / mu) / (1.0 + loss) * e_to_h
        for prange_i in numba.prange(nx):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            wx = i + 1 == nx
            ip = 0 if wx else i + 1
            for j in range(ny):
                wy = j + 1 == ny
                jp = 0 if wy else j + 1
                for k in range(nz):
                    wz = k + 1 == nz
                    kp = 0 if wz else k + 1
                    m = mat[i, j, k]
                    da = table_da[m]
                    db = table_db[m]
                    ex0, ey0, ez0 = ex[i, j, k], ey[i, j, k], ez[i, j, k]
                    ex_y = (wcy * ex[i, jp, k] + wsy * pex[i, jp, k]) if wy else ex[i, jp, k]
                    ex_z = (wcz * ex[i, j, kp] + wsz * pex[i, j, kp]) if wz else ex[i, j, kp]
                    ey_x = (wcx * ey[ip, j, k] + wsx * pey[ip, j, k]) if wx else ey[ip, j, k]
                    ey_z = (wcz * ey[i, j, kp] + wsz * pey[i, j, kp]) if wz else ey[i, j, kp]
                    ez_x = (wcx * ez[ip, j, k] + wsx * pez[ip, j, k]) if wx else ez[ip, j, k]
                    ez_y = (wcy * ez[i, jp, k] + wsy * pez[i, jp, k]) if wy else ez[i, jp, k]
                    hx[i, j, k] = da * hx[i, j, k] - db * ((ez_y - ez0) * ihy[j] * irce[i] - (ey_z - ey0) * ihz[k])
                    hy[i, j, k] = da * hy[i, j, k] - db * ((ex_z - ex0) * ihz[k] - (ez_x - ez0) * ihx[i])
                    hz[i, j, k] = da * hz[i, j, k] - db * ((rce[ip] * ey_x - rce[i] * ey0) * ihx[i]
                                                           - (ex_y - ex0) * ihy[j]) * irl[i]
    return kernel


# E on the hi faces: E_x on the hi-x face, E_y on the hi-y face, E_z on the hi-z face of cell (i, j, k). The missing lo
# H of cell 0 reads 0 on non-periodic axes (tangential H = 0 half a cell beyond the outermost centres: PMC).
_UPDATE_E_SOURCE = _WRAP_LOAD + r"""
extern "C" __global__
void update_e_3d(
    field_t* __restrict__ ex, field_t* __restrict__ ey, field_t* __restrict__ ez,
    const field_t* __restrict__ hx, const field_t* __restrict__ hy, const field_t* __restrict__ hz,
    const field_t* __restrict__ phx, const field_t* __restrict__ phy, const field_t* __restrict__ phz,
    const mat_t* __restrict__ mat, const float* __restrict__ t_ex, const float* __restrict__ t_ey,
    const float* __restrict__ t_ez, const float* __restrict__ t_sig, FACE_PARAMS,
    const float* __restrict__ iex, const float* __restrict__ iey, const float* __restrict__ iez,
    const float* __restrict__ rfh, const float* __restrict__ irfh, const float* __restrict__ irce,
    const float* __restrict__ wlx, const float* __restrict__ wux,
    const float* __restrict__ wly, const float* __restrict__ wuy,
    const float* __restrict__ wlz, const float* __restrict__ wuz,
    int nx, int ny, int nz, int px, int py, int pz, float dt, float h_to_e,
    float wcx, float wsx, float wcy, float wsy, float wcz, float wsz)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz) return;
    idx_t sy = nz, sx = (idx_t)ny * nz;
    idx_t idx = (idx_t)i * sx + (idx_t)j * sy + k;
    bool wx = i == 0, wy = j == 0, wz = k == 0;
    int im = wx ? nx - 1 : i - 1;
    idx_t dxm = wx ? (idx_t)(nx - 1) * sx : -sx;
    idx_t dym = wy ? (idx_t)(ny - 1) * sy : -sy;
    idx_t dzm = wz ? (idx_t)(nz - 1) : -(idx_t)1;
    float lx = (i > 0 || px) ? 1.0f : 0.0f;
    float ly = (j > 0 || py) ? 1.0f : 0.0f;
    float lz = (k > 0 || pz) ? 1.0f : 0.0f;
    float hx0 = FLOAD(hx[idx]), hy0 = FLOAD(hy[idx]), hz0 = FLOAD(hz[idx]);
    float ca, cb;
    face_coefficients(mat, t_ex, t_sig, FACE_ARGS, 0, idx, idx + hi_offset(i, nx, px, sx), wlx[i], wux[i], dt, &ca,
                      &cb);
    ex[idx] = FSTORE(ca * FLOAD(ex[idx]) + cb * h_to_e * ((hz0 - ly * wrap_load(hz, phz, idx + dym, wy, wcy, wsy))
                                                          * iey[j] * irfh[i]
                                                          - (hy0 - lz * wrap_load(hy, phy, idx + dzm, wz, wcz, wsz))
                                                          * iez[k]));
    face_coefficients(mat, t_ey, t_sig, FACE_ARGS, 1, idx, idx + hi_offset(j, ny, py, sy), wly[j], wuy[j], dt, &ca,
                      &cb);
    ey[idx] = FSTORE(ca * FLOAD(ey[idx]) + cb * h_to_e * ((hx0 - lz * wrap_load(hx, phx, idx + dzm, wz, wcz, wsz))
                                                          * iez[k]
                                                          - (hz0 - lx * wrap_load(hz, phz, idx + dxm, wx, wcx, wsx))
                                                          * iex[i]));
    face_coefficients(mat, t_ez, t_sig, FACE_ARGS, 2, idx, idx + hi_offset(k, nz, pz, (idx_t)1), wlz[k], wuz[k], dt,
                      &ca, &cb);
    ez[idx] = FSTORE(ca * FLOAD(ez[idx]) + cb * h_to_e * ((rfh[i] * hy0
                                                           - lx * rfh[im] * wrap_load(hy, phy, idx + dxm, wx, wcx, wsx))
                                                          * iex[i]
                                                          - (hx0 - ly * wrap_load(hx, phx, idx + dym, wy, wcy, wsy))
                                                          * iey[j]) * irce[i]);
}
"""


def _update_e_host(ex, ey, ez, hx, hy, hz, phx, phy, phz, mat, t_ex, t_ey, t_ez, t_sig, fid, t_face, use_face,
                   iex, iey, iez, rfh, irfh, irce, wlx, wux, wly, wuy, wlz, wuz, nx, ny, nz, px, py, pz, dt, h_to_e,
                   wcx, wsx, wcy, wsy, wcz, wsz):
    shares = ((wlx, wux), (wly, wuy), (wlz, wuz))
    periodic = (px, py, pz)
    tables = (t_ex, t_ey, t_ez)
    iex, iey, iez = iex[:, None, None], iey[None, :, None], iez[None, None, :]
    rfh, irfh, irce = (_row_factor(values, 0) for values in (rfh, irfh, irce))

    def update(comp, field, curl):
        ca, cb = _cached_face_coefficients(mat, tables[comp], t_sig, fid, t_face, use_face, comp, (0, 0, 0),
                                           mat.shape, *shares[comp], periodic[comp], dt)
        field[...] = ca * field + cb * h_to_e * curl
    update(0, ex, _times(_lo_difference(hz, phz, 1, py, wcy, wsy) * iey, irfh)
           - _lo_difference(hy, phy, 2, pz, wcz, wsz) * iez)
    update(1, ey, _lo_difference(hx, phx, 2, pz, wcz, wsz) * iez - _lo_difference(hz, phz, 0, px, wcx, wsx) * iex)
    hy_r = _times(hy, rfh)
    update(2, ez, _times(_lo_difference(hy_r, hy_r if phy is hy else _times(phy, rfh), 0, px, wcx, wsx) * iex
                         - _lo_difference(hx, phx, 1, py, wcy, wsy) * iey, irce))


def _update_e_numba(numba):
    eps0 = float(EPS0)
    _publish_numba_face_scalar(numba)

    @numba.njit(parallel=True, cache=True)
    def kernel(ex, ey, ez, hx, hy, hz, phx, phy, phz, mat, t_ex, t_ey, t_ez, t_sig, fid, t_face, use_face,
               iex, iey, iez, rfh, irfh, irce, wlx, wux, wly, wuy, wlz, wuz, nx, ny, nz, px, py, pz, dt, h_to_e,
               wcx, wsx, wcy, wsy, wcz, wsz):
        for prange_i in numba.prange(nx):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            im = nx - 1 if i == 0 else i - 1
            ip = i + 1 if i + 1 < nx else (0 if px != 0 else i)
            lx = 1.0 if (i > 0 or px != 0) else 0.0
            for j in range(ny):
                jm = ny - 1 if j == 0 else j - 1
                jp = j + 1 if j + 1 < ny else (0 if py != 0 else j)
                ly = 1.0 if (j > 0 or py != 0) else 0.0
                for k in range(nz):
                    km = nz - 1 if k == 0 else k - 1
                    kp = k + 1 if k + 1 < nz else (0 if pz != 0 else k)
                    lz = 1.0 if (k > 0 or pz != 0) else 0.0
                    m = mat[i, j, k]
                    hx0, hy0, hz0 = hx[i, j, k], hy[i, j, k], hz[i, j, k]
                    hx_y = (wcy * hx[i, jm, k] + wsy * phx[i, jm, k]) if j == 0 else hx[i, jm, k]
                    hx_z = (wcz * hx[i, j, km] + wsz * phx[i, j, km]) if k == 0 else hx[i, j, km]
                    hy_x = (wcx * hy[im, j, k] + wsx * phy[im, j, k]) if i == 0 else hy[im, j, k]
                    hy_z = (wcz * hy[i, j, km] + wsz * phy[i, j, km]) if k == 0 else hy[i, j, km]
                    hz_x = (wcx * hz[im, j, k] + wsx * phz[im, j, k]) if i == 0 else hz[im, j, k]
                    hz_y = (wcy * hz[i, jm, k] + wsy * phz[i, jm, k]) if j == 0 else hz[i, jm, k]
                    face_index = 3 * fid[i, j, k] if use_face != 0 else 0
                    smoothed_x = t_face[face_index] if face_index != 0 else 0.0
                    smoothed_y = t_face[face_index + 1] if face_index != 0 else 0.0
                    smoothed_z = t_face[face_index + 2] if face_index != 0 else 0.0
                    ca, cb = _numba_face_scalar(t_ex[m], t_ex[mat[ip, j, k]], t_sig[m], t_sig[mat[ip, j, k]], wlx[i],
                                                wux[i], dt, eps0, smoothed_x)
                    ex[i, j, k] = ca * ex[i, j, k] + cb * h_to_e * ((hz0 - ly * hz_y) * iey[j] * irfh[i]
                                                                    - (hy0 - lz * hy_z) * iez[k])
                    ca, cb = _numba_face_scalar(t_ey[m], t_ey[mat[i, jp, k]], t_sig[m], t_sig[mat[i, jp, k]], wly[j],
                                                wuy[j], dt, eps0, smoothed_y)
                    ey[i, j, k] = ca * ey[i, j, k] + cb * h_to_e * ((hx0 - lz * hx_z) * iez[k]
                                                                    - (hz0 - lx * hz_x) * iex[i])
                    ca, cb = _numba_face_scalar(t_ez[m], t_ez[mat[i, j, kp]], t_sig[m], t_sig[mat[i, j, kp]], wlz[k],
                                                wuz[k], dt, eps0, smoothed_z)
                    ez[i, j, k] = ca * ez[i, j, k] + cb * h_to_e * ((rfh[i] * hy0 - lx * rfh[im] * hy_x) * iex[i]
                                                                    - (hx0 - ly * hx_y) * iey[j]) * irce[i]
    return kernel


# CPML on one layer slab (axis dimension = n_rows, first grid row row0): psi = b psi + a dF (bulk spacing), then
# field += sign * coefficient * psi, run after the Yee kernel (which never writes its own curl inputs). H rows (hi
# faces) difference F[row + 1] - F[row], E rows (centres) F[row] - F[row - 1]; the coefficient is the live db / cb
# (times e_to_h / h_to_e: psi holds differences of the stored fields, float32). The second component's difference is
# weighted, (sw[g + 1] F[g + 1] - sw[g] F[g]) of[g] at grid row g (E: sw[g] F[g] - sw[g - 1] F[g - 1]): the cylindrical
# r factors on the r axis (update header), all 1 elsewhere.
_CPML_H_SOURCE = r"""
extern "C" __global__
void cpml_h_3d(
    field_t* __restrict__ h_first, field_t* __restrict__ h_second,
    const field_t* __restrict__ e_first, const field_t* __restrict__ e_second,
    float* __restrict__ psi_first, float* __restrict__ psi_second,
    const float* __restrict__ b, const float* __restrict__ a, const float* __restrict__ inv,
    const float* __restrict__ sw, const float* __restrict__ of,
    const mat_t* __restrict__ mat, const float* __restrict__ t_mu, const float* __restrict__ t_sm,
    int axis, int row0, int n_rows, int nx, int ny, int nz, float sign_first, float sign_second, float dt,
    float e_to_h)
{
    int d0 = (axis == 0) ? n_rows : nx, d1 = (axis == 1) ? n_rows : ny, d2 = (axis == 2) ? n_rows : nz;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= d0 || j >= d1 || k >= d2) return;
    idx_t p = ((idx_t)i * d1 + j) * d2 + k;
    int r = (axis == 0) ? i : (axis == 1) ? j : k;
    int gi = i + ((axis == 0) ? row0 : 0), gj = j + ((axis == 1) ? row0 : 0), gk = k + ((axis == 2) ? row0 : 0);
    int g = r + row0;
    idx_t sy = nz, sx = (idx_t)ny * nz;
    idx_t idx = (idx_t)gi * sx + (idx_t)gj * sy + gk;
    idx_t stride = (axis == 0) ? sx : (axis == 1) ? sy : (idx_t)1;
    float da, db;
    edge_coefficients(mat, t_mu, t_sm, idx, dt, &da, &db);
    db *= e_to_h;
    float br = b[r], ar = a[r] * inv[r];
    float pf = br * psi_first[p] + ar * (FLOAD(e_first[idx + stride]) - FLOAD(e_first[idx]));
    psi_first[p] = pf;
    h_first[idx] = FSTORE(FLOAD(h_first[idx]) + sign_first * db * pf);
    float ps = br * psi_second[p]
               + ar * (sw[g + 1] * FLOAD(e_second[idx + stride]) - sw[g] * FLOAD(e_second[idx])) * of[g];
    psi_second[p] = ps;
    h_second[idx] = FSTORE(FLOAD(h_second[idx]) + sign_second * db * ps);
}
"""


def _slab_terms(b, a, inv, axis):
    """(b, a inv) of a slab's rows broadcast along axis."""
    return b.reshape(_row_shape(axis)), (a * inv).reshape(_row_shape(axis))


def _cpml_h_host(h_first, h_second, e_first, e_second, psi_first, psi_second, b, a, inv, sw, of, mat, t_mu, t_sm,
                 axis, row0, n_rows, nx, ny, nz, sign_first, sign_second, dt, e_to_h):
    axis, row0, n_rows = int(axis), int(row0), int(n_rows)
    rows = along(axis, slice(row0, row0 + n_rows))
    upper = along(axis, slice(row0 + 1, row0 + n_rows + 1))
    decay, gain = _slab_terms(b, a, inv, axis)
    db = _cached_edge_coefficients(mat, t_mu, t_sm, dt, e_to_h, rows)[1]
    psi_first[...] = decay * psi_first + gain * (e_first[upper] - e_first[rows])
    h_first[rows] += sign_first * db * psi_first
    weight_upper = _axis_row(sw[row0 + 1:row0 + n_rows + 1], axis)
    weight_rows, factor = _axis_row(sw[row0:row0 + n_rows], axis), _axis_row(of[row0:row0 + n_rows], axis)
    psi_second[...] = (decay * psi_second
                       + gain * (weight_upper * e_second[upper] - weight_rows * e_second[rows]) * factor)
    h_second[rows] += sign_second * db * psi_second


def _cpml_h_numba(numba):
    mu0 = float(MU0)

    @numba.njit(parallel=True, cache=True)
    def kernel(h_first, h_second, e_first, e_second, psi_first, psi_second, b, a, inv, sw, of, mat, t_mu, t_sm, axis,
               row0, n_rows, nx, ny, nz, sign_first, sign_second, dt, e_to_h):
        d0 = n_rows if axis == 0 else nx
        d1 = n_rows if axis == 1 else ny
        d2 = n_rows if axis == 2 else nz
        step_x = 1 if axis == 0 else 0
        step_y = 1 if axis == 1 else 0
        step_z = 1 if axis == 2 else 0
        for prange_i in numba.prange(d0):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            gi = i + np.int64(row0) * step_x
            for j in range(d1):
                gj = j + np.int64(row0) * step_y
                for k in range(d2):
                    gk = k + np.int64(row0) * step_z
                    r = i if axis == 0 else (j if axis == 1 else k)
                    m = mat[gi, gj, gk]
                    mu = t_mu[m] * mu0
                    db = (dt / mu) / (1.0 + t_sm[m] * dt / (2.0 * mu)) * e_to_h
                    decay = b[r]
                    gain = a[r] * inv[r]
                    hi_i, hi_j, hi_k = gi + step_x, gj + step_y, gk + step_z
                    g = r + np.int64(row0)
                    psi = decay * psi_first[i, j, k] + gain * (e_first[hi_i, hi_j, hi_k] - e_first[gi, gj, gk])
                    psi_first[i, j, k] = psi
                    h_first[gi, gj, gk] += sign_first * db * psi
                    psi = decay * psi_second[i, j, k] + gain * (sw[g + 1] * e_second[hi_i, hi_j, hi_k]
                                                                - sw[g] * e_second[gi, gj, gk]) * of[g]
                    psi_second[i, j, k] = psi
                    h_second[gi, gj, gk] += sign_second * db * psi
    return kernel


_CPML_E_SOURCE = r"""
extern "C" __global__
void cpml_e_3d(
    field_t* __restrict__ e_first, field_t* __restrict__ e_second,
    const field_t* __restrict__ h_first, const field_t* __restrict__ h_second,
    float* __restrict__ psi_first, float* __restrict__ psi_second,
    const float* __restrict__ b, const float* __restrict__ a, const float* __restrict__ inv,
    const float* __restrict__ sw, const float* __restrict__ of,
    const mat_t* __restrict__ mat, const float* __restrict__ t_ex, const float* __restrict__ t_ey,
    const float* __restrict__ t_ez, const float* __restrict__ t_sig, FACE_PARAMS,
    const float* __restrict__ wlx, const float* __restrict__ wux,
    const float* __restrict__ wly, const float* __restrict__ wuy,
    const float* __restrict__ wlz, const float* __restrict__ wuz,
    int comp_first, int comp_second, int axis, int row0, int n_rows, int nx, int ny, int nz,
    int px, int py, int pz, float sign_first, float sign_second, float dt, float h_to_e)
{
    int d0 = (axis == 0) ? n_rows : nx, d1 = (axis == 1) ? n_rows : ny, d2 = (axis == 2) ? n_rows : nz;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= d0 || j >= d1 || k >= d2) return;
    idx_t p = ((idx_t)i * d1 + j) * d2 + k;
    int r = (axis == 0) ? i : (axis == 1) ? j : k;
    int g = r + row0;
    int gi = i + ((axis == 0) ? row0 : 0), gj = j + ((axis == 1) ? row0 : 0), gk = k + ((axis == 2) ? row0 : 0);
    idx_t sy = nz, sx = (idx_t)ny * nz;
    idx_t idx = (idx_t)gi * sx + (idx_t)gj * sy + gk;
    idx_t stride = (axis == 0) ? sx : (axis == 1) ? sy : (idx_t)1;
    float br = b[r], ar = a[r] * inv[r];
    float ca, cb;
    float pf = br * psi_first[p] + ar * (FLOAD(h_first[idx]) - FLOAD(h_first[idx - stride]));
    psi_first[p] = pf;
    component_coefficients(comp_first, gi, gj, gk, idx, nx, ny, nz, px, py, pz, mat, t_ex, t_ey, t_ez, t_sig,
                           FACE_ARGS, wlx, wux, wly, wuy, wlz, wuz, dt, &ca, &cb);
    e_first[idx] = FSTORE(FLOAD(e_first[idx]) + sign_first * cb * h_to_e * pf);
    float ps = br * psi_second[p]
               + ar * (sw[g] * FLOAD(h_second[idx]) - sw[g - 1] * FLOAD(h_second[idx - stride])) * of[g];
    psi_second[p] = ps;
    component_coefficients(comp_second, gi, gj, gk, idx, nx, ny, nz, px, py, pz, mat, t_ex, t_ey, t_ez, t_sig,
                           FACE_ARGS, wlx, wux, wly, wuy, wlz, wuz, dt, &ca, &cb);
    e_second[idx] = FSTORE(FLOAD(e_second[idx]) + sign_second * cb * h_to_e * ps);
}
"""


def _cpml_e_host(e_first, e_second, h_first, h_second, psi_first, psi_second, b, a, inv, sw, of, mat, t_ex, t_ey,
                 t_ez, t_sig, fid, t_face, use_face, wlx, wux, wly, wuy, wlz, wuz, comp_first, comp_second, axis, row0,
                 n_rows, nx, ny, nz, px, py, pz, sign_first, sign_second, dt, h_to_e):
    axis, row0, n_rows = int(axis), int(row0), int(n_rows)
    rows = along(axis, slice(row0, row0 + n_rows))
    lower = along(axis, slice(row0 - 1, row0 + n_rows - 1))
    decay, gain = _slab_terms(b, a, inv, axis)
    origin, extents = [0, 0, 0], list(mat.shape)
    origin[axis], extents[axis] = row0, n_rows
    tables, shares, periodic = (t_ex, t_ey, t_ez), ((wlx, wux), (wly, wuy), (wlz, wuz)), (px, py, pz)
    weight_rows, factor = _axis_row(sw[row0:row0 + n_rows], axis), _axis_row(of[row0:row0 + n_rows], axis)
    weight_lower = _axis_row(sw[row0 - 1:row0 + n_rows - 1], axis)
    for e, h, psi, sign, comp, weighted in ((e_first, h_first, psi_first, sign_first, int(comp_first), False),
                                            (e_second, h_second, psi_second, sign_second, int(comp_second), True)):
        difference = ((weight_rows * h[rows] - weight_lower * h[lower]) * factor if weighted
                      else h[rows] - h[lower])
        psi[...] = decay * psi + gain * difference
        _, cb = _cached_face_coefficients(mat, tables[comp], t_sig, fid, t_face, use_face, comp, origin, extents,
                                          *shares[comp], periodic[comp], dt)
        e[rows] += sign * cb * h_to_e * psi


def _cpml_e_numba(numba):
    eps0 = float(EPS0)
    _publish_numba_face_scalar(numba)

    @numba.njit(cache=True)
    def component_cb(comp, gi, gj, gk, nx, ny, nz, px, py, pz, mat, t_ex, t_ey, t_ez, t_sig, fid, t_face, use_face,
                     wlx, wux, wly, wuy, wlz, wuz, dt):
        """cb of E component comp at cell (gi, gj, gk) (component_coefficients)."""
        m = mat[gi, gj, gk]
        if comp == 0:
            partner = mat[gi + 1 if gi + 1 < nx else (0 if px != 0 else gi), gj, gk]
            eps_a, eps_b, weight_a, weight_b = t_ex[m], t_ex[partner], wlx[gi], wux[gi]
        elif comp == 1:
            partner = mat[gi, gj + 1 if gj + 1 < ny else (0 if py != 0 else gj), gk]
            eps_a, eps_b, weight_a, weight_b = t_ey[m], t_ey[partner], wly[gj], wuy[gj]
        else:
            partner = mat[gi, gj, gk + 1 if gk + 1 < nz else (0 if pz != 0 else gk)]
            eps_a, eps_b, weight_a, weight_b = t_ez[m], t_ez[partner], wlz[gk], wuz[gk]
        face_index = 3 * fid[gi, gj, gk] if use_face != 0 else 0
        smoothed = t_face[face_index + comp] if face_index != 0 else 0.0
        return _numba_face_scalar(eps_a, eps_b, t_sig[m], t_sig[partner], weight_a, weight_b, dt, eps0, smoothed)[1]

    globals()['_numba_cpml_component_cb'] = component_cb   # a global for the kernel, not a closure cell

    @numba.njit(parallel=True, cache=True)
    def kernel(e_first, e_second, h_first, h_second, psi_first, psi_second, b, a, inv, sw, of, mat, t_ex, t_ey, t_ez,
               t_sig, fid, t_face, use_face, wlx, wux, wly, wuy, wlz, wuz, comp_first, comp_second, axis, row0, n_rows,
               nx, ny, nz, px, py, pz, sign_first, sign_second, dt, h_to_e):
        d0 = n_rows if axis == 0 else nx
        d1 = n_rows if axis == 1 else ny
        d2 = n_rows if axis == 2 else nz
        step_x = 1 if axis == 0 else 0
        step_y = 1 if axis == 1 else 0
        step_z = 1 if axis == 2 else 0
        for prange_i in numba.prange(d0):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            gi = i + np.int64(row0) * step_x
            for j in range(d1):
                gj = j + np.int64(row0) * step_y
                for k in range(d2):
                    gk = k + np.int64(row0) * step_z
                    r = i if axis == 0 else (j if axis == 1 else k)
                    decay = b[r]
                    gain = a[r] * inv[r]
                    lo_i, lo_j, lo_k = gi - step_x, gj - step_y, gk - step_z
                    psi = decay * psi_first[i, j, k] + gain * (h_first[gi, gj, gk] - h_first[lo_i, lo_j, lo_k])
                    psi_first[i, j, k] = psi
                    cb = _numba_cpml_component_cb(comp_first, gi, gj, gk, nx, ny, nz, px, py, pz, mat, t_ex, t_ey,
                                                  t_ez, t_sig, fid, t_face, use_face, wlx, wux, wly, wuy, wlz, wuz, dt)
                    e_first[gi, gj, gk] += sign_first * cb * h_to_e * psi
                    g = r + np.int64(row0)
                    psi = decay * psi_second[i, j, k] + gain * (sw[g] * h_second[gi, gj, gk]
                                                                - sw[g - 1] * h_second[lo_i, lo_j, lo_k]) * of[g]
                    psi_second[i, j, k] = psi
                    cb = _numba_cpml_component_cb(comp_second, gi, gj, gk, nx, ny, nz, px, py, pz, mat, t_ex, t_ey,
                                                  t_ez, t_sig, fid, t_face, use_face, wlx, wux, wly, wuy, wlz, wuz, dt)
                    e_second[gi, gj, gk] += sign_second * cb * h_to_e * psi
    return kernel


# cb of E component `comp` on the cells of a box (origin i0, j0, k0, shape bx, by, bz) into a contiguous float32 box
# array: the live Yee coefficient of the pole corrections and the dipole calibration (no host sync).
_box_face_cb_gpu = cp.RawKernel(_PREAMBLE + r"""
extern "C" __global__
void box_face_cb_3d(
    float* __restrict__ out, int comp, int i0, int j0, int k0, int bx, int by, int bz,
    const mat_t* __restrict__ mat, const float* __restrict__ t_ex, const float* __restrict__ t_ey,
    const float* __restrict__ t_ez, const float* __restrict__ t_sig, FACE_PARAMS,
    const float* __restrict__ wlx, const float* __restrict__ wux,
    const float* __restrict__ wly, const float* __restrict__ wuy,
    const float* __restrict__ wlz, const float* __restrict__ wuz,
    int nx, int ny, int nz, int px, int py, int pz, float dt)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bx || j >= by || k >= bz) return;
    int gi = i0 + i, gj = j0 + j, gk = k0 + k;
    idx_t idx = ((idx_t)gi * ny + gj) * nz + gk;
    float ca, cb;
    component_coefficients(comp, gi, gj, gk, idx, nx, ny, nz, px, py, pz, mat, t_ex, t_ey, t_ez, t_sig,
                           FACE_ARGS, wlx, wux, wly, wuy, wlz, wuz, dt, &ca, &cb);
    out[((idx_t)i * by + j) * bz + k] = cb;
}
""", "box_face_cb_3d")


def _box_face_cb_host(out, comp, i0, j0, k0, bx, by, bz, mat, t_ex, t_ey, t_ez, t_sig, fid, t_face, use_face,
                      wlx, wux, wly, wuy, wlz, wuz, nx, ny, nz, px, py, pz, dt):
    comp = int(comp)
    shares = ((wlx, wux), (wly, wuy), (wlz, wuz))[comp]
    _, cb = _face_coefficients(mat, (t_ex, t_ey, t_ez)[comp], t_sig, fid, t_face, use_face, comp, (i0, j0, k0),
                               (bx, by, bz), *shares, (px, py, pz)[comp], dt)
    out[...] = cb.reshape(out.shape)


box_face_cb_3d = raw_twin('em3d.box_face_cb', _box_face_cb_gpu, _box_face_cb_host)


# Ohmic J.E loss of the cells of a box (origin, shape) into q [J/m^3]: each E face's sigma E^2 V_face splits between
# its two cells by their half-cell conductance, so a cell gets 0.5 sigma_cell (E_hi^2 + E_lo^2) per component (its own
# hi face and the hi face of its lower neighbour; none below cell 0 of a non-periodic axis), half_dt = dt / 2. The
# split sums exactly to the width-weighted face sigma of the update, except on faces next to PEC (the PEC half is 0).
# scale = dt / 2 x e_scale^2 (stored E -> V/m).
_OHMIC_LOSS_SOURCE = r"""
__device__ __forceinline__ float squared(float value) { return value * value; }

extern "C" __global__
void ohmic_loss_3d(
    float* __restrict__ q, const field_t* __restrict__ ex, const field_t* __restrict__ ey,
    const field_t* __restrict__ ez, const mat_t* __restrict__ mat, const float* __restrict__ t_sig,
    int i0, int j0, int k0, int bx, int by, int bz, int nx, int ny, int nz, int px, int py, int pz, float scale)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i = blockIdx.z * blockDim.z + threadIdx.z;
    if (i >= bx || j >= by || k >= bz) return;
    int gi = i0 + i, gj = j0 + j, gk = k0 + k;
    idx_t sy = nz, sx = (idx_t)ny * nz;
    idx_t idx = (idx_t)gi * sx + (idx_t)gj * sy + gk;
    float sigma = t_sig[mat[idx]];
    if (sigma == 0.0f) return;
    float e2 = squared(FLOAD(ex[idx])) + squared(FLOAD(ey[idx])) + squared(FLOAD(ez[idx]));
    if (gi > 0 || px) { idx_t o = idx + ((gi > 0) ? -sx : (idx_t)(nx - 1) * sx); e2 += squared(FLOAD(ex[o])); }
    if (gj > 0 || py) { idx_t o = idx + ((gj > 0) ? -sy : (idx_t)(ny - 1) * sy); e2 += squared(FLOAD(ey[o])); }
    if (gk > 0 || pz) { idx_t o = idx + ((gk > 0) ? -(idx_t)1 : (idx_t)(nz - 1)); e2 += squared(FLOAD(ez[o])); }
    q[idx] += scale * sigma * e2;
}
"""


def _ohmic_loss_host(q, ex, ey, ez, mat, t_sig, i0, j0, k0, bx, by, bz, nx, ny, nz, px, py, pz, scale):
    origin, extents = (int(i0), int(j0), int(k0)), (int(bx), int(by), int(bz))
    box = tuple(slice(start, start + count) for start, count in zip(origin, extents))

    def squared(field, key):
        values = field[key].astype(np.float32, copy=False)
        return values * values
    sigma = t_sig[mat[box]]
    e2 = squared(ex, box) + squared(ey, box) + squared(ez, box)
    for axis, (field, periodic) in enumerate(((ex, px), (ey, py), (ez, pz))):   # the lower neighbour's own face
        rows = np.arange(origin[axis], origin[axis] + extents[axis]) - 1
        present = rows >= 0
        if periodic:
            rows[~present] = mat.shape[axis] - 1
            present[:] = True
        else:
            rows[~present] = 0
        key = list(box)
        key[axis] = rows
        e2 += np.where(present.reshape(_row_shape(axis)), squared(field, tuple(key)), np.float32(0.0))
    q[box] += scale * sigma * e2


def _ohmic_loss_numba(numba):
    @numba.njit(parallel=True, cache=True)
    def kernel(q, ex, ey, ez, mat, t_sig, i0, j0, k0, bx, by, bz, nx, ny, nz, px, py, pz, scale):
        for prange_i in numba.prange(bx):
            i = np.int64(prange_i)   # the parfor gufunc types the prange index uint64: mixed with int64 -> float64
            gi = i0 + i
            lo_i = gi - 1 if gi > 0 else nx - 1
            for j in range(by):
                gj = j0 + j
                lo_j = gj - 1 if gj > 0 else ny - 1
                for k in range(bz):
                    gk = k0 + k
                    sigma = t_sig[mat[gi, gj, gk]]
                    if sigma == 0.0:
                        continue
                    lo_k = gk - 1 if gk > 0 else nz - 1
                    e2 = ex[gi, gj, gk] ** 2 + ey[gi, gj, gk] ** 2 + ez[gi, gj, gk] ** 2
                    if gi > 0 or px != 0:
                        e2 += ex[lo_i, gj, gk] ** 2
                    if gj > 0 or py != 0:
                        e2 += ey[gi, lo_j, gk] ** 2
                    if gk > 0 or pz != 0:
                        e2 += ez[gi, gj, lo_k] ** 2
                    q[gi, gj, gk] += scale * sigma * e2
    return kernel


# Conformal PEC (solver_plan S7.6, conformal_pec3d): one thread per corrected H face, right after the Yee H kernel of
# each field set; face n = H component c of the flat cell face_cell[n] gains db x the curl of its weights - 1 (E_b at
# the hi / lo a corner, E_a at the hi / lo b corner, (a, b) the next two axes cyclically), so with the Yee term its
# curl is (1 / A_eff) sum +-l E over the free edge lengths (Dey-Mittra). Faces never touch the grid edge (no wraps).
_CONFORMAL_H_SOURCE = r"""
extern "C" __global__
void conformal_h_3d(
    field_t* __restrict__ hx, field_t* __restrict__ hy, field_t* __restrict__ hz,
    const field_t* __restrict__ ex, const field_t* __restrict__ ey, const field_t* __restrict__ ez,
    const mat_t* __restrict__ mat, const float* __restrict__ t_mu, const float* __restrict__ t_sm,
    const float* __restrict__ ihx, const float* __restrict__ ihy, const float* __restrict__ ihz,
    const idx_t* __restrict__ face_cell, const int* __restrict__ face_axis, const float* __restrict__ face_extra,
    int n_faces, int ny, int nz, float dt, float e_to_h)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= n_faces) return;
    idx_t sy = nz, sx = (idx_t)ny * nz;
    idx_t idx = face_cell[n];
    int i = (int)(idx / sx);
    int j = (int)((idx - (idx_t)i * sx) / sy);
    int k = (int)(idx - (idx_t)i * sx - (idx_t)j * sy);
    float da, db;
    edge_coefficients(mat, t_mu, t_sm, idx, dt, &da, &db);
    db *= e_to_h;
    const float* w = face_extra + 4 * (idx_t)n;
    int c = face_axis[n];
    if (c == 0) {
        float curl = (w[0] * FLOAD(ez[idx + sy]) - w[1] * FLOAD(ez[idx])) * ihy[j]
                   - (w[2] * FLOAD(ey[idx + 1]) - w[3] * FLOAD(ey[idx])) * ihz[k];
        hx[idx] = FSTORE(FLOAD(hx[idx]) - db * curl);
    } else if (c == 1) {
        float curl = (w[0] * FLOAD(ex[idx + 1]) - w[1] * FLOAD(ex[idx])) * ihz[k]
                   - (w[2] * FLOAD(ez[idx + sx]) - w[3] * FLOAD(ez[idx])) * ihx[i];
        hy[idx] = FSTORE(FLOAD(hy[idx]) - db * curl);
    } else {
        float curl = (w[0] * FLOAD(ey[idx + sx]) - w[1] * FLOAD(ey[idx])) * ihx[i]
                   - (w[2] * FLOAD(ex[idx + sy]) - w[3] * FLOAD(ex[idx])) * ihy[j];
        hz[idx] = FSTORE(FLOAD(hz[idx]) - db * curl);
    }
}
"""


def _conformal_h_host(hx, hy, hz, ex, ey, ez, mat, t_mu, t_sm, ihx, ihy, ihz, face_cell, face_axis, face_extra,
                      n_faces, ny, nz, dt, e_to_h):
    ny, nz = int(ny), int(nz)

    def layout():   # per H component: its faces' cells, neighbour cells, weights, curl factors and db (host cached)
        strides = (ny * nz, nz, 1)
        db_table = _edge_tables(t_mu, t_sm, dt)[1]
        i_index, rest = np.divmod(face_cell, ny * nz)
        j_index, k_index = np.divmod(rest, nz)
        indices, inverse = (i_index, j_index, k_index), (ihx, ihy, ihz)
        components = []
        for component in range(3):
            on_axis = face_axis == component
            if not on_axis.any():
                continue
            axis_a, axis_b = (component + 1) % 3, (component + 2) % 3
            cells, weights = face_cell[on_axis], face_extra[on_axis]
            components.append((component, axis_a, axis_b, cells, cells + strides[axis_a], cells + strides[axis_b],
                               tuple(weights[:, column].copy() for column in range(4)),
                               inverse[axis_a][indices[axis_a][on_axis]], inverse[axis_b][indices[axis_b][on_axis]],
                               db_table[mat.reshape(-1)[cells]] * e_to_h))
        return components
    h_fields, e_fields = (hx, hy, hz), (ex, ey, ez)
    for (component, axis_a, axis_b, cells, cells_a, cells_b, (w0, w1, w2, w3), inverse_a, inverse_b,
         db) in _host_cached((face_cell, face_axis, face_extra, ihx, ihy, ihz, mat, t_mu, t_sm),
                             ('conformal', ny, nz, float(dt), float(e_to_h)), layout):
        e_a, e_b = e_fields[axis_a].reshape(-1), e_fields[axis_b].reshape(-1)
        curl = ((w0 * e_b[cells_a] - w1 * e_b[cells]) * inverse_a - (w2 * e_a[cells_b] - w3 * e_a[cells]) * inverse_b)
        np.subtract.at(h_fields[component].reshape(-1), cells, db * curl)


_FIELD_SOURCES = {'update_h_3d': _UPDATE_H_SOURCE, 'update_e_3d': _UPDATE_E_SOURCE, 'cpml_h_3d': _CPML_H_SOURCE,
                  'cpml_e_3d': _CPML_E_SOURCE, 'ohmic_loss_3d': _OHMIC_LOSS_SOURCE,
                  'conformal_h_3d': _CONFORMAL_H_SOURCE}
# name -> (numpy twin, numba factory or None); the host twins compute in float32 (the cpu backend refuses
# half_storage, solvers.explicit_kernels backend_misfit), numba only in single precision (the sparse conformal
# correction has none: a few faces)
_FIELD_HOST_TWINS = {'update_h_3d': (_update_h_host, _update_h_numba), 'update_e_3d': (_update_e_host, _update_e_numba),
                     'cpml_h_3d': (_cpml_h_host, _cpml_h_numba), 'cpml_e_3d': (_cpml_e_host, _cpml_e_numba),
                     'ohmic_loss_3d': (_ohmic_loss_host, _ohmic_loss_numba),
                     'conformal_h_3d': (_conformal_h_host, None)}
_FIELD_KERNELS = {}


def field_kernels(precision='single') -> dict:
    """{name: raw KernelTwin} of the field kernels for a stepper precision ('single' | 'half_storage'; built on first
    use, compiled at the first launch)."""
    if precision not in _FIELD_KERNELS:
        preamble = _PREAMBLE + storage_preamble(precision, 'field_t', 'FLOAD', 'FSTORE')
        kernels = {}
        for name, source in _FIELD_SOURCES.items():
            host, numba_factory = _FIELD_HOST_TWINS[name]
            kernels[name] = raw_twin(f'em3d.{name}.{precision}', cp.RawKernel(preamble + source, name), host,
                                     numba=numba_factory if precision == 'single' else None)
        _FIELD_KERNELS[precision] = kernels
    return _FIELD_KERNELS[precision]


# Dispersive poles on a region box (2D leap-frog forms, materials.DrudePole / LorentzPole.ade_coefficients), the drive
# times the face weight w (width-weighted share of the region's cells at that face): J from E^n before the Yee update;
# the corrections E -= cb J and E -= (cb / dt) dP (fdtd_ez_2d_kernels.drude_e_correction / lorentz_e_correction) after.
# Drude in the 2D velocity form J += damping J + drive w E (damping = c1 - 1, fdtd_ez_2d_kernels).
def _drude_j_weighted_host(damping, drive, w, e, j):
    j += damping * j + drive * w * e


drude_j_weighted = twin('em3d.drude_j_weighted', cp.ElementwiseKernel(
    'float32 damping, float32 drive, float32 w, float32 e',
    'float32 j',
    'j += damping * j + drive * w * e',
    'em3d_drude_j_weighted'), _drude_j_weighted_host)


def _lorentz_p_weighted_host(damping, restoring, drive, w, e, p, dp):
    dp += damping * dp - restoring * p + drive * w * e
    p += dp


lorentz_p_weighted = twin('em3d.lorentz_p_weighted', cp.ElementwiseKernel(
    'float32 damping, float32 restoring, float32 drive, float32 w, float32 e',
    'float32 p, float32 dp',
    '''
    dp += damping * dp - restoring * p + drive * w * e;
    p += dp;
    ''',
    'em3d_lorentz_p_weighted'), _lorentz_p_weighted_host)


# Half of the time-centred pole work on a face box [J/m^3]: scale = dt / 2 with the Drude J, 1 / 2 with the Lorentz
# dP = P^{n+1} - P^n; called with E^n after the pole drive and with E^{n+1} at the end of the step (2D convention).
def _pole_work_half_host(scale, current, e, work):
    work += scale * current * e


pole_work_half = twin('em3d.pole_work_half', cp.ElementwiseKernel(
    'float32 scale, float32 current, float32 e',
    'float32 work',
    'work += scale * current * e',
    'em3d_pole_work_half'), _pole_work_half_host)


def launch(dims):
    """(grid, block) of a kernel over (d0, d1, d2) = (x, y, z) extents (threads along z fastest)."""
    threads_z, threads_y, threads_x = _BLOCK
    return ((int(dims[2]) + threads_z - 1) // threads_z, (int(dims[1]) + threads_y - 1) // threads_y,
            (int(dims[0]) + threads_x - 1) // threads_x), _BLOCK
