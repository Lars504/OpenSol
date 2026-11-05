"""Shared Yee spacing of the FDTD kernels and frequency-domain operators, and the face-material helpers that DEFINE the
H_z face material (E_x / E_y faces); every engine, FDFD and mode solver must use them (xp passed explicitly)."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0


def grid_cell_widths(grid):
    """(dx_cells (nx,), dy_cells (ny,)) float64 widths [m] the kernels use: dx_arr / dy_arr on graded grids, else the
    float32 grid.dx / grid.dy (NOT the requested float64 dx of Domain.cell_widths())."""
    if hasattr(grid, 'dx_arr'):
        return np.asarray(grid.dx_arr, dtype=np.float64), np.asarray(grid.dy_arr, dtype=np.float64)
    return np.full(grid.nx, float(grid.dx)), np.full(grid.ny, float(grid.dy))


@dataclass(frozen=True, eq=False)
class YeeInverseSpacings:
    """Float64 inverse spacings [1/m]: *_e = 1/width of cell i (derivatives at cell centres: E_z, H_z updates), *_h =
    1/mean(w_i, w_(i+1) mod n) (derivatives at hi faces: H_x/H_y, E_x/E_y updates; last entry = periodic wrap)."""
    inv_dx_e: np.ndarray
    inv_dx_h: np.ndarray
    inv_dy_e: np.ndarray
    inv_dy_h: np.ndarray


def _axis_inverse_spacings(cell_widths, axis_name):
    """(1/width, 1/hi-face spacing) of one axis in float64; ValueError for non-1D, empty, non-finite or non-positive
    widths."""
    widths = np.asarray(cell_widths, dtype=np.float64)
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError(f"{axis_name} cell widths must be a non-empty 1D array, got shape {widths.shape}")
    if not (np.all(np.isfinite(widths)) and np.all(widths > 0.0)):
        raise ValueError(f"{axis_name} cell widths must be finite and > 0 [m], got min {float(np.min(widths))!r}")
    face_spacing = 0.5 * (widths + np.roll(widths, -1))
    return 1.0 / widths, 1.0 / face_spacing


def yee_inverse_spacings(dx_cells, dy_cells):
    """YeeInverseSpacings of the (nx,) / (ny,) cell widths [m] (the arithmetic the FDTD kernels' arrays come from)."""
    inv_dx_e, inv_dx_h = _axis_inverse_spacings(dx_cells, 'x')
    inv_dy_e, inv_dy_h = _axis_inverse_spacings(dy_cells, 'y')
    return YeeInverseSpacings(inv_dx_e=inv_dx_e, inv_dx_h=inv_dx_h, inv_dy_e=inv_dy_e, inv_dy_h=inv_dy_h)


def cpml_grading_cell_size(domain):
    """PML cell size [m] of the CPML sigma grading: mesh dx_base on graded grids (from_mesh grades the PML toward it),
    else grid.dx; a padding-free Domain.from_widths (no mesh_config, no CPML cells): its finest x cell."""
    if domain.non_uniform:
        if domain.mesh_config is None:
            return float(np.min(np.asarray(domain.grid.dx_arr)))
        return float(domain.mesh_config.dx_base)
    return float(domain.grid.dx)


def sheet_launch_factor(index_squared, omega, normal_spacing, dt=None, xp=np):
    """complex cos(kappa/2), sin(kappa/2) = sqrt(index_squared) D sin(omega dt/2) / (C0 dt) (dt None, FDFD: omega D /
    (2 C0)), kappa = k_num D across a sheet: a Yee sheet current K launches eta K / (2 cos(kappa/2)) per side."""
    if dt is None:
        vacuum_half_phase = 0.5 * float(omega) * float(normal_spacing) / C0
    else:
        vacuum_half_phase = float(normal_spacing) * math.sin(0.5 * float(omega) * float(dt)) / (C0 * float(dt))
    return xp.sqrt(1.0 - xp.asarray(index_squared, dtype=xp.complex128) * vacuum_half_phase ** 2)


# ---- face material (H_z polarisation) -----------------------------------------------------------------------------
# E_x of cell (i, j) sits on its hi-y face (axis 1), E_y on its hi-x face (axis 0). A face value is the dual-cell mean
# of its two cells, weighted by the half widths w_k / 2, w_(k+1) / 2 its dual cell covers (tangential E: exact energy
# split, so the 0.5 sigma_cell E_f^2 loss share per cell is exact); the last row / column takes the periodic partner
# (cell 0) or the cell itself. A face is PEC when either adjacent cell is. These helpers are THE definition: FDTD,
# FDFD and mode solvers must not re-derive it (the fused H_z e_face_coefficients kernel reuses face_shares and is
# checked against face_average by em_operators_consistency).

def face_partner(cell_array, axis, periodic, xp=np):
    """New array of the cell across each hi face along `axis`: a[k+1]; the last entry is a[0] (periodic) or a[last]."""
    if periodic:
        return xp.roll(cell_array, -1, axis=axis)
    following = [slice(None)] * cell_array.ndim
    following[axis] = slice(1, None)
    last = [slice(None)] * cell_array.ndim
    last[axis] = slice(-1, None)
    return xp.concatenate((cell_array[tuple(following)], cell_array[tuple(last)]), axis=axis)


# The reverse direction (hi-face values -> cell centres): cell k = plain mean of its hi face k and lo face k - 1; cell 0
# takes the wrapped last face (periodic; a Bloch k != 0 phasor times e^{+j k L}: the face one period down) or its own hi
# face. fields.collocate_to_centres is its float32 fast path (real FDTD fields, no Bloch factor).

def lower_face(index, periodic):
    """Face index the centre mean of cell `index` pairs with its hi face: index - 1; cell 0 gives -1 (periodic: the last
    face by negative indexing; a Bloch phasor needs e^{+j k L} there) or 0 (its own hi face)."""
    return index - 1 if (index >= 1 or periodic) else index


def lower_face_partner(face_array, axis, periodic, xp=np, wrap_factor=1.0):
    """New array of each cell's lo-face value along `axis`: a[k-1]; the first entry is a[last] (periodic; times
    wrap_factor = e^{+j k L} on a Bloch axis, complex arrays only) or a[0]."""
    if periodic:
        partner = xp.roll(face_array, 1, axis=axis)
        if wrap_factor != 1.0:
            first = [slice(None)] * face_array.ndim
            first[axis] = 0
            partner[tuple(first)] *= wrap_factor
        return partner
    first = [slice(None)] * face_array.ndim
    first[axis] = slice(0, 1)
    preceding = [slice(None)] * face_array.ndim
    preceding[axis] = slice(None, -1)
    return xp.concatenate((face_array[tuple(first)], face_array[tuple(preceding)]), axis=axis)


def cell_row(array, axis, index):
    """View of the cells (or hi faces) at `index` along `axis` of an (nx, ny) array (the FDFD wall rows)."""
    return array[index, :] if axis == 0 else array[:, index]


def face_shares(cell_widths, periodic, real_dtype):
    """Host (n,) (lower, upper) weights of the hi-face mean: w_k / (w_k + w_(k+1)), w_(k+1) / (w_k + w_(k+1)) in
    `real_dtype`; uniform widths give exactly 0.5 / 0.5 (the plain mean bit for bit)."""
    widths = np.asarray(cell_widths, dtype=np.float64)
    partner_widths = face_partner(widths, 0, periodic)
    return ((widths / (widths + partner_widths)).astype(real_dtype),
            (partner_widths / (widths + partner_widths)).astype(real_dtype))


def face_average(cell_array, axis, periodic, cell_widths, xp=np):
    """Hi-face value along `axis` = (a_k w_k + a_(k+1) w_(k+1)) / (w_k + w_(k+1)) in the array's (float or complex) dtype;
    `cell_widths` = the (n,) widths [m] along `axis` (grid_cell_widths); axis 1 -> E_x faces, axis 0 -> E_y faces."""
    if not np.issubdtype(cell_array.dtype, np.inexact):
        raise TypeError(f"face_average needs a float or complex array, got dtype {cell_array.dtype}")
    widths = np.asarray(cell_widths, dtype=np.float64)
    if widths.shape != (cell_array.shape[axis],):
        raise ValueError(f"face_average needs {cell_array.shape[axis]} cell widths along axis {axis}, got shape "
                         f"{widths.shape}")
    broadcast_shape = [1] * cell_array.ndim
    broadcast_shape[axis] = widths.size
    lower_share, upper_share = face_shares(widths, periodic, np.finfo(cell_array.dtype).dtype)
    lower_share = xp.asarray(lower_share.reshape(broadcast_shape))
    upper_share = xp.asarray(upper_share.reshape(broadcast_shape))
    return cell_array * lower_share + face_partner(cell_array, axis, periodic, xp) * upper_share


def face_pec(cell_mask, axis, periodic, xp=np):
    """Bool hi-face mask along `axis`: a face is PEC when either adjacent cell is (axis 1 -> E_x faces, axis 0 -> E_y
    faces)."""
    if cell_mask.dtype != np.bool_:
        raise TypeError(f"face_pec needs a bool cell mask, got dtype {cell_mask.dtype}")
    return cell_mask | face_partner(cell_mask, axis, periodic, xp)
