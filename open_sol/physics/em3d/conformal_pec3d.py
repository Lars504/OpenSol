"""Conformal PEC of em_fdtd_3d (solver_plan S7.6; host numpy / scipy, built once at finalize): Dey-Mittra on the H
faces, each cut face (the square between four cell centres) takes the free lengths of its four E edges over its free
area, floored per face so the unchanged time step stays stable (the smallest cut never sets dt)."""
# Geometry (engine3d layout): E_d of cell q is the link from centre q to centre q + e_d (the 2D staircase rule: pinned
# when both centres are PEC); H_c of cell p sits on the face between the centres p, p + e_a, p + e_b, p + e_a + e_b,
# (a, b) = the next two axes after c cyclically, with the curl (E_b[p + e_a] - E_b[p]) / D_a - (E_a[p + e_b] - E_a[p]) / D_b
# (D = dual spacings). Dey-Mittra: (1 / A_eff) sum +-l E over the free edge lengths l, i.e. weight w = (A / A_eff) l / L
# per edge (L its full length, A the face area). Cut faces = faces with a conformal-PEC and a free corner and no other
# PEC corner (pec walls, staircase regions), inside the physical window and off the grid edge (no wraps); edges with
# one conformal and one free end are bisected from the free end, others keep their staircase length (a cut no centre
# sees stays staircased).
# Stability floor (3D Courant): the face's Gershgorin row (vacuum) G(A) = (1 / A) sum_e l_e (2 sum_{o != d_e} 1 / w_o)
# may not exceed its uncut row G_full scaled to the time step, row_scale STABILITY_FRACTION^2 (2 / (C0 dt))^2 G_full /
# lambda_cfl (lambda_cfl = 4 (1 / w_c^2 + 1 / D_a^2 + 1 / D_b^2) the face's local CFL eigenvalue): A_eff = max(A_free,
# min(A, stable area)). In 3D the row comparison is a heuristic (Gershgorin overestimates the full curl-curl, 16 vs
# 12 / d^2 on cubes; an E edge shared by several floored faces adds them up: a PEC sphere reached 1.01 x the leap-frog
# limit at row_scale 1), so the build measures the largest eigenvalue of the vacuum operator on a band of
# STABILITY_BAND_CELLS around the corrected faces and lowers row_scale until it is <= STABILITY_FRACTION^2 of the limit
# (row_scale 0 after _FLOOR_ROUNDS: A_eff = A, weights l / L <= 1, at most the staircase operator).
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, FACES_3D
from ...core.cut_cells import rectangle_free_areas, segment_crossings

STABILITY_FRACTION = 0.99   # the corrected operator's largest eigenvalue stays <= this fraction^2 of the leap-frog limit
STABILITY_BAND_CELLS = 4    # cells around the corrected faces in the eigenvalue check (E = 0 beyond; the box margin)
_FLOOR_ROUNDS = 6           # row_scale tightenings by the measured eigenvalue before row_scale 0
_FLOOR_MARGIN = 0.98        # a tightening aims this fraction below the target
_EIGEN_TOLERANCE = 1e-3     # ARPACK relative tolerance of the largest eigenvalue
_EIGEN_BASIS = 20           # ARPACK basis vectors (host bytes 8 x this per band E edge)
_EIGEN_ITERATIONS = 5000
_DENSE_EDGES = 400          # band operators up to this many E edges: dense eigenvalues
_EXTRA_ZERO = 1e-7          # |weight - 1| below this on every edge: the face needs no correction
# the four E edges of an H_c face as (edge axis 'b' or 'a', start corner (step along a, step along b)), in the order
# of the kernel weights: E_b at the hi / lo a corner, E_a at the hi / lo b corner
_FACE_EDGES = (('b', (1, 0)), ('b', (0, 0)), ('a', (0, 1)), ('a', (0, 0)))


@dataclass(frozen=True, eq=False)
class ConformalPEC3D:
    """Host arrays of the corrected H faces: face_cell (int64 flat (i ny + j) nz + k of the face's lower corner cell
    p), face_axis (int32 H component 0 / 1 / 2), face_extra (float32 (n, 4) weight - 1 of its E edges in the H curl,
    order of _FACE_EDGES), area_fraction (float64 free area / face area), floored (bool: the stability floor raised
    A_eff above the free area); row_scale and stability_ratio (the band operator's largest eigenvalue x dt^2 / 4)."""
    face_cell: np.ndarray
    face_axis: np.ndarray
    face_extra: np.ndarray
    area_fraction: np.ndarray
    floored: np.ndarray
    row_scale: float = 1.0
    stability_ratio: float = 0.0

    def summary(self) -> str:
        """One line for logs: corrected faces, the smallest free area, how many the stability floor limited."""
        return (f'conformal PEC (3D): {self.face_cell.size} cut H faces (free area min '
                f'{float(np.min(self.area_fraction, initial=1.0)):.3g}, {int(np.count_nonzero(self.floored))} limited '
                f'by the stability floor at row scale {self.row_scale:.3g}; largest eigenvalue x dt^2 / 4 '
                f'{self.stability_ratio:.4g})')


@dataclass(frozen=True, eq=False)
class _FaceCuts:
    """Geometry of one H component's cut faces (row_scale independent): grid cells, edge fractions (n, 4), the
    Gershgorin numerators (full / cut), face areas, local CFL eigenvalues and free area fractions."""
    component: int
    cells: tuple
    fractions: np.ndarray
    numerator_full: np.ndarray
    numerator_cut: np.ndarray
    face_area: np.ndarray
    lambda_cfl: np.ndarray
    area: np.ndarray


def _empty():
    return ConformalPEC3D(np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros((0, 4), np.float32), np.zeros(0),
                          np.zeros(0, dtype=bool))


def build_conformal_pec_3d(widths, centres, window, origin, pec, conformal, inside, dt) -> ConformalPEC3D:
    """ConformalPEC3D of the H faces whose corners lie in a host box at grid cell `origin`: pec = every PEC cell,
    conformal = the cells of the drawn regions inside(x, y, z) (host bool box arrays); widths / centres = per-axis
    float64 cell widths / centres [m] of the whole grid, window = the physical-window slices, dt the engine step.
    The box should reach STABILITY_BAND_CELLS + 3 cells past the conformal cells (else the check's band is clipped)."""
    pec = np.asarray(pec, dtype=bool)
    conformal = np.asarray(conformal, dtype=bool) & pec
    grid_shape = tuple(axis_widths.size for axis_widths in widths)
    leapfrog_limit = (STABILITY_FRACTION * 2.0 / (C0 * float(dt))) ** 2
    cuts = [_component_cuts(component, widths, centres, window, origin, pec, conformal, inside)
            for component in range(3)]
    cuts = [cut for cut in cuts if cut is not None]
    if not cuts:
        return _empty()
    target = STABILITY_FRACTION ** 2
    row_scale = 1.0
    for round_index in range(_FLOOR_ROUNDS + 1):
        if round_index == _FLOOR_ROUNDS:
            row_scale = 0.0
        parts = [_corrected_faces(cut, row_scale * leapfrog_limit) for cut in cuts]
        cells, axes, extra, area, floored = (np.concatenate(values, axis=-1 if index == 0 else 0)
                                             for index, values in enumerate(zip(*parts)))
        ratio = _largest_ratio(widths, origin, pec, cells, axes, extra, dt)
        if ratio <= target or row_scale == 0.0:
            break
        row_scale *= (_FLOOR_MARGIN * target / ratio) if math.isfinite(ratio) else 0.0
    if cells.shape[1] == 0:
        return _empty()
    flat = (cells[0] * grid_shape[1] + cells[1]) * grid_shape[2] + cells[2]
    return ConformalPEC3D(flat.astype(np.int64), axes.astype(np.int32), extra.astype(np.float32), area, floored,
                          float(row_scale), float(ratio))


def _component_cuts(component, widths, centres, window, origin, pec, conformal, inside):
    """_FaceCuts of the H_component faces with a conformal and a free corner (outline tests), None when none."""
    axis_a, axis_b = (component + 1) % 3, (component + 2) % 3
    box_shape = pec.shape
    corner_ranges = []   # local lower-corner cells p with all four corners in the box and the window
    for axis in range(3):
        first = max(window[axis].start - origin[axis], 0)
        stop = min(window[axis].stop - origin[axis], box_shape[axis]) - (1 if axis in (axis_a, axis_b) else 0)
        corner_ranges.append(slice(first, max(first, stop)))

    def corner_view(array, step_a, step_b):
        key = list(corner_ranges)
        for axis, step in ((axis_a, step_a), (axis_b, step_b)):
            key[axis] = slice(corner_ranges[axis].start + step, corner_ranges[axis].stop + step)
        return array[tuple(key)]
    corners = ((0, 0), (1, 0), (0, 1), (1, 1))
    other_pec = pec & ~conformal
    any_conformal = np.zeros(corner_view(pec, 0, 0).shape, dtype=bool)
    any_free = np.zeros_like(any_conformal)
    any_other = np.zeros_like(any_conformal)
    for steps in corners:
        any_conformal |= corner_view(conformal, *steps)
        any_free |= ~corner_view(pec, *steps)
        any_other |= corner_view(other_pec, *steps)
    local = np.nonzero(any_conformal & any_free & ~any_other)
    if local[0].size == 0:
        return None
    local = tuple(indices + corner_ranges[axis].start for axis, indices in enumerate(local))
    cells = tuple(indices + origin[axis] for axis, indices in enumerate(local))
    count = cells[0].size

    def shifted(indices, step_a, step_b):
        moved = list(indices)
        moved[axis_a] = moved[axis_a] + step_a
        moved[axis_b] = moved[axis_b] + step_b
        return tuple(moved)

    def point(grid_cells):
        return tuple(np.asarray(axis_centres, dtype=np.float64)[index] for axis_centres, index in
                     zip(centres, grid_cells))

    def stepped(indices, axis):
        moved = list(indices)
        moved[axis] = moved[axis] + 1
        return tuple(moved)

    fractions = np.ones((count, 4))
    numerator_full = np.zeros(count)
    numerator_cut = np.zeros(count)
    for slot, (edge_name, steps) in enumerate(_FACE_EDGES):
        edge_axis = axis_b if edge_name == 'b' else axis_a
        start_local = shifted(local, *steps)
        end_local = stepped(start_local, edge_axis)
        start_pec, end_pec = pec[start_local], pec[end_local]
        fraction = np.where(start_pec & end_pec, 0.0, 1.0)
        located = (conformal[start_local] & ~end_pec) | (conformal[end_local] & ~start_pec)   # one end in the metal
        edge_cells = shifted(cells, *steps)   # E cell = the edge's start corner
        if located.any():
            start_point, end_point = point(edge_cells), point(stepped(edge_cells, edge_axis))
            free_first = ~start_pec[located]
            free_end = tuple(np.where(free_first, start[located], end[located])
                             for start, end in zip(start_point, end_point))
            wall_end = tuple(np.where(free_first, end[located], start[located])
                             for start, end in zip(start_point, end_point))
            start_inside = np.asarray(inside(*free_end), dtype=bool)
            straddles = ~start_inside & np.asarray(inside(*wall_end), dtype=bool)
            values = np.ones(int(np.count_nonzero(located)))   # no straddle seen by the test: the staircase link
            if straddles.any():
                values[straddles] = segment_crossings(inside, tuple(value[straddles] for value in free_end),
                                                      tuple(value[straddles] for value in wall_end),
                                                      start_inside[straddles])
            fraction[located] = np.clip(values, 0.0, 1.0)
        fractions[:, slot] = fraction
        # Gershgorin term of the edge: full length L x the curl factors of the E update, 2 sum_{o != edge} 1 / w_o
        edge_widths = widths[edge_axis]
        length = 0.5 * (edge_widths[edge_cells[edge_axis]] + edge_widths[edge_cells[edge_axis] + 1])
        curl_factor = sum(2.0 / widths[other][edge_cells[other]] for other in range(3) if other != edge_axis)
        numerator_full += length * curl_factor
        numerator_cut += fraction * length * curl_factor

    dual_a, dual_b = (0.5 * (widths[axis][cells[axis]] + widths[axis][cells[axis] + 1]) for axis in (axis_a, axis_b))
    face_area = dual_a * dual_b
    lambda_cfl = 4.0 * (1.0 / widths[component][cells[component]] ** 2 + 1.0 / dual_a ** 2 + 1.0 / dual_b ** 2)

    lo = point(cells)
    hi = list(lo)
    for axis in (axis_a, axis_b):
        hi[axis] = np.asarray(centres[axis], dtype=np.float64)[cells[axis] + 1]
    rows_along_a = np.abs(fractions[:, 0] - fractions[:, 1]) >= np.abs(fractions[:, 2] - fractions[:, 3])
    area = np.clip(rectangle_free_areas(inside, lo, tuple(hi), axis_a, axis_b, rows_along_a), 0.0, 1.0)
    return _FaceCuts(component, cells, fractions, numerator_full, numerator_cut, face_area, lambda_cfl, area)


def _corrected_faces(cut, row_limit):
    """(grid cells (3, n) int64, H components, extras (n, 4), area fractions, floored) of the corrected faces of one
    component with the floor's row allowance row_limit x G_full / lambda_cfl (0: A_eff = A)."""
    face_area = cut.face_area
    if row_limit > 0.0:
        allowed_row = row_limit * (cut.numerator_full / face_area) / cut.lambda_cfl
        ceiling = np.minimum(face_area, cut.numerator_cut / allowed_row)
    else:
        ceiling = face_area
    effective_area = np.maximum(cut.area * face_area, ceiling)
    active = (cut.numerator_cut > 0.0) & (effective_area > 0.0)
    weights = (face_area / np.where(active, effective_area, 1.0))[:, None] * cut.fractions
    extra = np.where(cut.fractions > 0.0, weights - 1.0, 0.0)
    corrected = active & np.any(np.abs(extra) > _EXTRA_ZERO, axis=1)
    floored = effective_area > cut.area * face_area * (1.0 + 1e-9)
    return (np.stack([indices[corrected] for indices in cut.cells]).astype(np.int64),
            np.full(int(np.count_nonzero(corrected)), cut.component), extra[corrected], cut.area[corrected],
            floored[corrected])


def _largest_ratio(widths, origin, pec, cells, axes, extra, dt) -> float:
    """Largest eigenvalue x (C0 dt / 2)^2 of the vacuum E operator C0^2 curl_h curl_e (operators3d; H rows weighted 1 +
    extra on the corrected faces) on the live E edges within STABILITY_BAND_CELLS of the corrected faces, E = 0 beyond (a
    principal submatrix: modes above the Yee band decay within a few cells of the cuts); host scipy, inf when ARPACK
    finds none."""
    from scipy import ndimage
    from scipy.sparse.linalg import ArpackNoConvergence, eigs
    from .boundaries3d import BoundaryConfig3D
    from .operators3d import weighted_curl_e, yee_operators_3d
    if cells.shape[1] == 0:
        return 0.0
    box_shape = np.asarray(pec.shape)
    local = cells - np.asarray(origin, dtype=np.int64)[:, None]
    lower = np.maximum(local.min(axis=1) - STABILITY_BAND_CELLS - 1, 0)
    upper = np.minimum(local.max(axis=1) + STABILITY_BAND_CELLS + 3, box_shape)
    shape = tuple(int(value) for value in upper - lower)
    sub_pec = pec[tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))].reshape(-1)
    face_flat = np.ravel_multi_index(tuple(local - lower[:, None]), shape)
    seed = np.zeros(int(np.prod(shape)), dtype=bool)
    seed[face_flat] = True
    cube = np.ones((3, 3, 3), dtype=bool)
    edge_band = ndimage.binary_dilation(seed.reshape(shape), structure=cube, iterations=STABILITY_BAND_CELLS)
    strides = (shape[1] * shape[2], shape[2], 1)
    first_cell = [int(start) + int(lo) for start, lo in zip(origin, lower)]   # grid index of the sub-box's cell 0
    sub_widths = tuple(np.asarray(widths[axis], dtype=np.float64)[first_cell[axis]:first_cell[axis] + shape[axis]]
                       for axis in range(3))
    # pmc faces: non-periodic sub-box axes without CPML (the band never reaches wraps, ghosts or PML)
    ops = yee_operators_3d(sub_widths, BoundaryConfig3D({face: 'pmc' for face in FACES_3D}), (1.0,) * 3)
    curl_e = weighted_curl_e(ops, face_flat, axes, extra)
    n_cells = int(np.prod(shape))
    band_flat = np.flatnonzero(edge_band)
    band_index = np.unravel_index(band_flat, shape)
    live = []
    for component in range(3):   # live E_c edges q -> q + e_c inside the sub-box (the curl rows of their H faces too)
        flat = band_flat[band_index[component] + 1 < shape[component]]
        flat = flat[~(sub_pec[flat] & sub_pec[flat + strides[component]])]
        live.append(component * n_cells + flat)
    live = np.concatenate(live)
    if live.size == 0:
        return 0.0
    band_operator = (C0 * float(dt) / 2.0) ** 2 * (ops.curl_h[live] @ curl_e.tocsc()[:, live])
    if live.size <= _DENSE_EDGES:
        return float(np.max(np.abs(np.linalg.eigvals(band_operator.toarray()))))
    try:
        values = eigs(band_operator, k=1, which='LM', ncv=_EIGEN_BASIS, tol=_EIGEN_TOLERANCE,
                      maxiter=_EIGEN_ITERATIONS, v0=np.random.default_rng(0).standard_normal(live.size),
                      return_eigenvectors=False)
    except ArpackNoConvergence as exc:   # the Ritz values found so far
        values = exc.eigenvalues
    return float(np.max(np.abs(values))) if np.size(values) else math.inf
