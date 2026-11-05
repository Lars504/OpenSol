"""Conformal PEC of the 2D FDTD engines and the 2D FDFD assembly (solver_plan S7.6; host numpy, built once): E_z gets
the exact wall distance on each H link to a PEC cell, H_z the free face lengths and cell areas (Dey-Mittra). FDTD: each
limited per cell so the cell keeps its Gershgorin bound below the leap-frog limit (the smallest cut never sets the time
step); FDFD: floored at CONDITIONING_FLOOR only, the free area entering as the mass (the operator stays symmetric)."""
from dataclasses import dataclass, replace

import numpy as np

from ...core.constants import C0
from ...core.cut_cells import DUAL_CUT_FLOOR, crossing_fractions, free_areas, free_fractions
from .yee_grid import face_pec, grid_cell_widths

STABILITY_FRACTION = 0.99   # a corrected cell's Gershgorin bound stays <= this fraction^2 of (2 / dt)^2 (vacuum)
FULL_FACE = 1e-6            # free face fractions within this of 0 / 1 count as wholly PEC / wholly free
CONDITIONING_FLOOR = DUAL_CUT_FLOOR   # FDFD: smallest E_z wall fraction / H_z free area fraction (no dt, as em_modes)
_FLOOR_STEPS = 50           # bisection steps of a cell's E_z wall-distance floor
_EXTRA_ZERO = 1e-7          # |weight - 1| below this on every face: the H_z cell needs no correction


@dataclass(frozen=True, eq=False)
class ConformalPEC:
    """Host arrays of one polarisation. ez: face_cell (int32 flat lower cell i * ny + j), face_axis (int32: 0 = hi-x
    face, H_y; 1 = hi-y face, H_x), face_extra (float32 factor - 1 of that face's H curl term), wall_fraction (float64
    wall distance / centre spacing before the stability floor). hz: cell (int32 flat), cell_extra (float32 (n, 4)
    weight - 1 of E_x hi / lo and E_y hi / lo in the H_z curl), keep_x / keep_y (float32 (nx, ny) E face keep masks,
    0 = PEC), area_fraction (float64 free area / cell area of the corrected cells)."""
    polarization: str
    face_cell: np.ndarray = None
    face_axis: np.ndarray = None
    face_extra: np.ndarray = None
    wall_fraction: np.ndarray = None
    cell: np.ndarray = None
    cell_extra: np.ndarray = None
    keep_x: np.ndarray = None
    keep_y: np.ndarray = None
    area_fraction: np.ndarray = None

    def summary(self) -> str:
        """One line for logs: corrected faces / cells and how many the stability floor limited."""
        if self.polarization == 'ez':
            floored = int(np.sum(1.0 / (1.0 + self.face_extra.astype(np.float64)) > self.wall_fraction + 1e-6))
            return (f'conformal PEC (E_z): {self.face_cell.size} H faces at the exact wall distance, {floored} '
                    'limited by the stability floor')
        return (f'conformal PEC (H_z): {self.cell.size} cut cells (free area min '
                f'{float(np.min(self.area_fraction, initial=1.0)):.3g})')


def build_conformal_pec(domain, polarization, pec, conformal, inside, dt, periodic=(False, False)) -> ConformalPEC:
    """ConformalPEC of a Domain: pec = every PEC cell, conformal = the cells of the drawn region inside(x, y) (host
    bool (nx, ny)); geometry from Domain.cell_edges, spacings from yee_grid.grid_cell_widths."""
    x_edges, y_edges = (np.asarray(edges, dtype=np.float64) for edges in domain.cell_edges())
    cell_dx, cell_dy = grid_cell_widths(domain.grid)
    window = domain.physical_window()
    if polarization == 'ez':
        return ez_wall_faces(x_edges, y_edges, cell_dx, cell_dy, pec, conformal, inside, dt, window)
    return hz_cut_cells(x_edges, y_edges, cell_dx, cell_dy, conformal, inside, dt, window, periodic)


def _leapfrog_limit(dt):
    """(STABILITY_FRACTION 2 / (C0 dt))^2: the largest stable Gershgorin row of the vacuum operator, times a margin."""
    return (STABILITY_FRACTION * 2.0 / (C0 * dt)) ** 2


def _ez_wall_links(x_edges, y_edges, pec, conformal, inside, window):
    """(axis, lower i, lower j, free i, free j, t) of every face between a free window cell and a conformal cell
    (axis 0 = hi-x face, 1 = hi-y face of the lower cell; t = wall distance / centre spacing from the free centre)."""
    nx, ny = pec.shape
    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    in_window = np.zeros((nx, ny), dtype=bool)
    in_window[window] = True
    free = ~pec & in_window
    parts = []
    for axis, (di, dj) in ((0, (1, 0)), (1, (0, 1))):
        lower, upper = (slice(0, nx - di), slice(0, ny - dj)), (slice(di, nx), slice(dj, ny))
        both = in_window[lower] & in_window[upper]
        for free_lower, chosen in ((True, free[lower] & conformal[upper] & both),
                                   (False, free[upper] & conformal[lower] & both)):
            li, lj = np.nonzero(chosen)
            free_i, free_j = (li, lj) if free_lower else (li + di, lj + dj)
            wall_i, wall_j = (li + di, lj + dj) if free_lower else (li, lj)
            parts.append((np.full(li.size, axis), li, lj, free_i, free_j, wall_i, wall_j))
    axis, li, lj, free_i, free_j, wall_i, wall_j = (np.concatenate(values) for values in zip(*parts))
    fraction = np.ones(axis.size)
    if axis.size:
        start_x, start_y = x_centres[free_i], y_centres[free_j]
        end_x, end_y = x_centres[wall_i], y_centres[wall_j]
        start_inside = np.asarray(inside(start_x, start_y), dtype=bool)
        straddles = ~start_inside & np.asarray(inside(end_x, end_y), dtype=bool)
        if straddles.any():
            fraction[straddles] = crossing_fractions(inside, start_x[straddles], start_y[straddles], end_x[straddles],
                                                     end_y[straddles], start_inside[straddles])
    return axis, li, lj, free_i, free_j, np.clip(fraction, 0.0, 1.0)


def ez_wall_faces(x_edges, y_edges, cell_dx, cell_dy, pec, conformal, inside, dt, window) -> ConformalPEC:
    """E_z: every face between a free window cell and a conformal cell puts E_z = 0 at the wall distance t d of its
    link (H curl term x 1 / t_eff); t_eff = max(t, floor) with the free cell's floor keeping its operator row
    sum_e c_e l_e / (d_e A) <= the leap-frog limit (c_e = 2 free neighbour, 1 PEC, 1 / t_eff conformal)."""
    nx, ny = pec.shape
    axis, li, lj, free_i, free_j, fraction = _ez_wall_links(x_edges, y_edges, pec, conformal, inside, window)
    if axis.size == 0:
        return ConformalPEC('ez', face_cell=np.zeros(0, np.int32), face_axis=np.zeros(0, np.int32),
                            face_extra=np.zeros(0, np.float32), wall_fraction=np.zeros(0))

    # per free cell: operator row of its fixed links, the budget left for its conformal links
    owner, inverse = np.unique(free_i * ny + free_j, return_inverse=True)
    owner_i, owner_j = np.divmod(owner, ny)
    fixed = np.zeros(owner.size)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbour_i, neighbour_j = owner_i + di, owner_j + dj
        on_grid = (neighbour_i >= 0) & (neighbour_i < nx) & (neighbour_j >= 0) & (neighbour_j < ny)
        clipped_i, clipped_j = np.clip(neighbour_i, 0, nx - 1), np.clip(neighbour_j, 0, ny - 1)
        if di:
            length, spacing = cell_dy[owner_j], 0.5 * (cell_dx[owner_i] + cell_dx[clipped_i])
        else:
            length, spacing = cell_dx[owner_i], 0.5 * (cell_dy[owner_j] + cell_dy[clipped_j])
        fixed += np.where(on_grid & pec[clipped_i, clipped_j], 1.0, 2.0) * length / spacing
    link_length = np.where(axis == 0, cell_dy[free_j], cell_dx[free_i])
    link_spacing = np.where(axis == 0, 0.5 * (cell_dx[li] + cell_dx[np.minimum(li + 1, nx - 1)]),
                            0.5 * (cell_dy[lj] + cell_dy[np.minimum(lj + 1, ny - 1)]))   # both branches evaluate
    conductance = link_length / link_spacing
    fixed -= np.bincount(inverse, weights=conductance, minlength=owner.size)   # the conformal links were counted as PEC
    budget = _leapfrog_limit(dt) * cell_dx[owner_i] * cell_dy[owner_j] - fixed

    def row(floor):
        return np.bincount(inverse, weights=conductance / np.maximum(fraction, floor[inverse]), minlength=owner.size)
    lower, upper = np.zeros(owner.size), np.ones(owner.size)
    for _ in range(_FLOOR_STEPS):
        middle = 0.5 * (lower + upper)
        fits = row(middle) <= budget
        upper = np.where(fits, middle, upper)
        lower = np.where(fits, lower, middle)
    floor = np.where(row(np.zeros(owner.size)) <= budget, 0.0, upper)
    effective = np.maximum(fraction, floor[inverse])
    return ConformalPEC('ez', face_cell=(li * ny + lj).astype(np.int32), face_axis=axis.astype(np.int32),
                        face_extra=(1.0 / effective - 1.0).astype(np.float32), wall_fraction=fraction)


def _hz_cut_geometry(x_edges, y_edges, conformal, inside, window, periodic):
    """(keep_x, keep_y float32 (nx, ny) E_x / E_y face keeps, 0 = PEC, ci, cj, fractions (n, 4) free fractions of E_x
    hi / lo and E_y hi / lo, area (n,) free area / cell area) of the cut window cells off the grid edge."""
    nx, ny = conformal.shape
    keep_x = (~face_pec(conformal, 1, periodic[1])).astype(np.float32)
    keep_y = (~face_pec(conformal, 0, periodic[0])).astype(np.float32)
    padded = np.pad(conformal, 1, mode='edge')
    any_inside = np.zeros((nx, ny), dtype=bool)
    any_outside = np.zeros((nx, ny), dtype=bool)
    for di in range(3):
        for dj in range(3):
            view = padded[di:di + nx, dj:dj + ny]
            any_inside |= view
            any_outside |= ~view
    interior = np.zeros((nx, ny), dtype=bool)
    x_start, x_stop, _ = window[0].indices(nx)
    y_start, y_stop, _ = window[1].indices(ny)
    interior[max(x_start, 1):min(x_stop, nx - 1), max(y_start, 1):min(y_stop, ny - 1)] = True
    ci, cj = np.nonzero(any_inside & any_outside & interior)
    if ci.size == 0:
        return keep_x, keep_y, ci, cj, np.zeros((0, 4)), np.zeros(0)

    x_lo, x_hi, y_lo, y_hi = x_edges[ci], x_edges[ci + 1], y_edges[cj], y_edges[cj + 1]
    fractions = np.stack([free_fractions(inside, x_lo, y_hi, x_hi, y_hi),    # E_x on the hi-y face
                          free_fractions(inside, x_lo, y_lo, x_hi, y_lo),    # E_x on the lo-y face
                          free_fractions(inside, x_hi, y_lo, x_hi, y_hi),    # E_y on the hi-x face
                          free_fractions(inside, x_lo, y_lo, x_lo, y_hi)], axis=1)   # E_y on the lo-x face
    fractions = np.where(fractions < FULL_FACE, 0.0, np.where(fractions > 1.0 - FULL_FACE, 1.0, fractions))
    keep_x[ci, cj] = fractions[:, 0] > 0.0
    keep_x[ci, cj - 1] = fractions[:, 1] > 0.0
    keep_y[ci, cj] = fractions[:, 2] > 0.0
    keep_y[ci - 1, cj] = fractions[:, 3] > 0.0
    rows_along_x = np.abs(fractions[:, 3] - fractions[:, 2]) >= np.abs(fractions[:, 1] - fractions[:, 0])
    area = np.clip(free_areas(inside, x_lo, x_hi, y_lo, y_hi, rows_along_x), 0.0, 1.0)
    return keep_x, keep_y, ci, cj, fractions, area


def hz_cut_cells(x_edges, y_edges, cell_dx, cell_dy, conformal, inside, dt, window, periodic) -> ConformalPEC:
    """H_z (Dey-Mittra): E faces are PEC only where wholly inside the region; every cut window cell takes its free
    face lengths l and area A_free, H_z curl (1 / A_eff) sum +-l E, A_eff = max(A_free, min(A, (C0 dt / 0.99)^2
    sum l / (2 d))) so its row stays within the leap-frog limit (d = the face's dual spacing)."""
    keep_x, keep_y, ci, cj, fractions, area = _hz_cut_geometry(x_edges, y_edges, conformal, inside, window, periodic)
    if ci.size == 0:
        return ConformalPEC('hz', cell=np.zeros(0, np.int32), cell_extra=np.zeros((0, 4), np.float32),
                            keep_x=keep_x, keep_y=keep_y, area_fraction=np.zeros(0))
    ny = conformal.shape[1]
    cell_area = cell_dx[ci] * cell_dy[cj]
    lengths = np.stack([cell_dx[ci], cell_dx[ci], cell_dy[cj], cell_dy[cj]], axis=1)
    duals = np.stack([0.5 * (cell_dy[cj] + cell_dy[cj + 1]), 0.5 * (cell_dy[cj - 1] + cell_dy[cj]),
                      0.5 * (cell_dx[ci] + cell_dx[ci + 1]), 0.5 * (cell_dx[ci - 1] + cell_dx[ci])], axis=1)
    row_sum = np.sum(fractions * lengths / duals, axis=1)
    stable_area = 2.0 * row_sum / _leapfrog_limit(dt)          # A_eff giving exactly the limit
    effective_area = np.maximum(area * cell_area, np.minimum(cell_area, stable_area))
    active = (row_sum > 0.0) & (effective_area > 0.0)
    weights = (cell_area / np.where(active, effective_area, 1.0))[:, None] * fractions
    extra = np.where(fractions > 0.0, weights - 1.0, 0.0)
    corrected = active & np.any(np.abs(extra) > _EXTRA_ZERO, axis=1)
    return ConformalPEC('hz', cell=(ci * ny + cj)[corrected].astype(np.int32),
                        cell_extra=extra[corrected].astype(np.float32), keep_x=keep_x, keep_y=keep_y,
                        area_fraction=area[corrected])


# ---- FDFD (physics/em/solvers/fdfd_2d): the same geometry as operator weights, no time step ----
@dataclass(frozen=True, eq=False)
class ConformalFDFD:
    """(nx, ny) arrays of the FDFD conformal PEC of one polarisation (host numpy; on_backend for the run backend). ez:
    face_scale_hi_x / _y = 1 / max(t, CONDITIONING_FLOOR) on the H faces of the wall links, 1 elsewhere. hz:
    fraction_hi_x / _y = free length / length of the E_y / E_x faces (0 = PEC), area_fraction = free area / cell area of
    the cut cells (>= CONDITIONING_FLOOR; 1 elsewhere), fixed_cells = PEC cells without a free face (H_z = 0 rows)."""
    polarization: str
    face_scale_hi_x: object = None
    face_scale_hi_y: object = None
    fraction_hi_x: object = None
    fraction_hi_y: object = None
    area_fraction: object = None
    fixed_cells: object = None

    def on_backend(self, xp):
        """A copy with every array moved to the array module xp (numpy or cupy)."""
        names = (('face_scale_hi_x', 'face_scale_hi_y') if self.polarization == 'ez'
                 else ('fraction_hi_x', 'fraction_hi_y', 'area_fraction', 'fixed_cells'))
        return replace(self, **{name: xp.asarray(getattr(self, name)) for name in names})


def build_conformal_fdfd(domain, polarization, pec, conformal, inside, periodic=(False, False)) -> ConformalFDFD:
    """Host ConformalFDFD of a Domain: pec = every PEC cell, conformal = the cells of the drawn region inside(x, y)
    (host bool (nx, ny)); geometry from Domain.cell_edges."""
    x_edges, y_edges = (np.asarray(edges, dtype=np.float64) for edges in domain.cell_edges())
    window = domain.physical_window()
    if polarization == 'ez':
        return fdfd_ez_wall_faces(x_edges, y_edges, pec, conformal, inside, window)
    return fdfd_hz_cut_cells(x_edges, y_edges, pec, conformal, inside, window, periodic)


def fdfd_ez_wall_faces(x_edges, y_edges, pec, conformal, inside, window) -> ConformalFDFD:
    """E_z: the H face of each link from a free window cell into a conformal cell carries 1 / max(t, floor), E_z = 0 at
    t d; the PEC end is pinned, so only the free cell's diagonal changes (the operator stays symmetric)."""
    axis, li, lj, _, _, fraction = _ez_wall_links(x_edges, y_edges, pec, conformal, inside, window)
    scales = (np.ones(pec.shape), np.ones(pec.shape))
    for face_axis, scale in enumerate(scales):
        chosen = axis == face_axis
        scale[li[chosen], lj[chosen]] = 1.0 / np.maximum(fraction[chosen], CONDITIONING_FLOOR)
    return ConformalFDFD('ez', face_scale_hi_x=scales[0], face_scale_hi_y=scales[1])


def fdfd_hz_cut_cells(x_edges, y_edges, pec, conformal, inside, window, periodic) -> ConformalFDFD:
    """H_z (Dey-Mittra of hz_cut_cells without dt): operator face weights x the free length fraction l / L of each E
    face, the k0^2 mu mass x the free area fraction of each cut cell; one fraction per face, so the area-scaled operator
    (K - k0^2 diag(A_free)) stays complex symmetric."""
    keep_x, keep_y, ci, cj, fractions, area = _hz_cut_geometry(x_edges, y_edges, conformal, inside, window, periodic)
    fraction_hi_y = keep_x.astype(np.float64)   # E_x faces
    fraction_hi_x = keep_y.astype(np.float64)   # E_y faces
    area_fraction = np.ones(conformal.shape)
    if ci.size:   # a face shared by two cut cells gets the same fraction from both (same segment)
        fraction_hi_y[ci, cj] = fractions[:, 0]
        fraction_hi_y[ci, cj - 1] = fractions[:, 1]
        fraction_hi_x[ci, cj] = fractions[:, 2]
        fraction_hi_x[ci - 1, cj] = fractions[:, 3]
        area_fraction[ci, cj] = np.maximum(area, CONDITIONING_FLOOR)
    has_free_face = (_cell_has_open_face(fraction_hi_x > 0.0, 0, periodic[0])
                     | _cell_has_open_face(fraction_hi_y > 0.0, 1, periodic[1]))
    fixed_cells = (pec & ~conformal) | (conformal & ~has_free_face)
    area_fraction[fixed_cells] = 1.0
    return ConformalFDFD('hz', fraction_hi_x=fraction_hi_x, fraction_hi_y=fraction_hi_y, area_fraction=area_fraction,
                         fixed_cells=fixed_cells)


def _cell_has_open_face(open_faces, axis, periodic):
    """Bool (nx, ny): the cell's hi or lo face along axis is open (non-periodic: the never-updated last hi face and the
    missing lo face of cell 0 do not count)."""
    if periodic:
        return open_faces | np.roll(open_faces, 1, axis=axis)
    upper = open_faces.copy()
    last = [slice(None), slice(None)]
    last[axis] = -1
    upper[tuple(last)] = False
    lower = np.zeros_like(upper)
    target, source = [slice(None), slice(None)], [slice(None), slice(None)]
    target[axis], source[axis] = slice(1, None), slice(None, -1)
    lower[tuple(target)] = upper[tuple(source)]
    return upper | lower
