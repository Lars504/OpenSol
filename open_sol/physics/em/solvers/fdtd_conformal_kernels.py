"""Sparse conformal-PEC corrections of the 2D FDTD H updates (physics/em/conformal_pec): launched right after the Yee
H kernel of each field set, on the listed faces (E_z) / cells (H_z) only, with the same db and curl spacings; core.kernels
twins (numpy twins of the same call), real_t = the field dtype (stepper single / double)."""
import numpy as np

from ....core.kernels import real_raw_twin

_BLOCK = 256

# E_z: H on face k (lower cell idx, axis 0 = hi-x face H_y, 1 = hi-y face H_x) gains extra[k] x its Yee curl term, so
# the wall (E_z = 0 in the PEC cell) sits at distance d / (1 + extra) along the link. Faces never wrap.
_CONFORMAL_H_EZ_SOURCE = r"""
extern "C" __global__
void conformal_h_ez(const real_t* __restrict__ ez, real_t* __restrict__ hx, real_t* __restrict__ hy,
                    const real_t* __restrict__ db, const real_t* __restrict__ inv_dx_h,
                    const real_t* __restrict__ inv_dy_h,
                    const int* __restrict__ face_cell, const int* __restrict__ face_axis,
                    const real_t* __restrict__ face_extra, int n_faces, int ny)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_faces) return;
    int idx = face_cell[k];
    int i = idx / ny;
    int j = idx - i * ny;
    real_t e = ez[idx];
    if (face_axis[k] == 0) hy[idx] += face_extra[k] * db[idx] * (ez[idx + ny] - e) * inv_dx_h[i];
    else                   hx[idx] -= face_extra[k] * db[idx] * (ez[idx + 1] - e) * inv_dy_h[j];
}
"""


def _conformal_h_ez_host(ez, hx, hy, db, inv_dx_h, inv_dy_h, face_cell, face_axis, face_extra, n_faces, ny):
    ny = int(ny)
    ez_flat, db_flat = ez.reshape(-1), db.reshape(-1)
    i_index, j_index = np.divmod(face_cell, ny)
    for axis, target, step, inverse_spacing, index, sign in ((0, hy, ny, inv_dx_h, i_index, 1.0),
                                                              (1, hx, 1, inv_dy_h, j_index, -1.0)):
        on_axis = face_axis == axis
        cells = face_cell[on_axis]
        if cells.size:
            update = (face_extra[on_axis] * db_flat[cells] * (ez_flat[cells + step] - ez_flat[cells])
                      * inverse_spacing[index[on_axis]])
            target.reshape(-1)[cells] += update.dtype.type(sign) * update   # faces unique per axis: no add.at


conformal_h_ez = real_raw_twin('em2d.conformal_h_ez', _CONFORMAL_H_EZ_SOURCE, 'conformal_h_ez', _conformal_h_ez_host)

# H_z: cut cell k (flat idx, never on the grid edge) adds db x the curl of its weights - 1 (E_x hi / lo, E_y hi / lo):
# with the Yee term the curl is (1 / A_eff) sum +-l E over the free face lengths l (Dey-Mittra).
_CONFORMAL_HZ_SOURCE = r"""
extern "C" __global__
void conformal_hz(real_t* __restrict__ hz, const real_t* __restrict__ ex, const real_t* __restrict__ ey,
                  const real_t* __restrict__ db, const real_t* __restrict__ inv_dx_e,
                  const real_t* __restrict__ inv_dy_e,
                  const int* __restrict__ cell, const real_t* __restrict__ cell_extra, int n_cells, int ny)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_cells) return;
    int idx = cell[k];
    int i = idx / ny;
    int j = idx - i * ny;
    const real_t* w = cell_extra + 4 * k;
    real_t curl = (w[0] * ex[idx] - w[1] * ex[idx - 1]) * inv_dy_e[j]
                - (w[2] * ey[idx] - w[3] * ey[idx - ny]) * inv_dx_e[i];
    hz[idx] += db[idx] * curl;
}
"""


def _conformal_hz_host(hz, ex, ey, db, inv_dx_e, inv_dy_e, cell, cell_extra, n_cells, ny):
    ny = int(ny)
    ex_flat, ey_flat = ex.reshape(-1), ey.reshape(-1)
    i_index, j_index = np.divmod(cell, ny)
    curl = ((cell_extra[:, 0] * ex_flat[cell] - cell_extra[:, 1] * ex_flat[cell - 1]) * inv_dy_e[j_index]
            - (cell_extra[:, 2] * ey_flat[cell] - cell_extra[:, 3] * ey_flat[cell - ny]) * inv_dx_e[i_index])
    hz.reshape(-1)[cell] += db.reshape(-1)[cell] * curl   # the cut cells are unique: no add.at


conformal_hz = real_raw_twin('em2d.conformal_hz', _CONFORMAL_HZ_SOURCE, 'conformal_hz', _conformal_hz_host)


def sparse_launch(count: int):
    """(grid, block, int32 count) of a one-thread-per-entry launch."""
    return ((count + _BLOCK - 1) // _BLOCK,), (_BLOCK,), np.int32(count)
