"""Exact boundary geometry of the conformal options (solver_plan S7.6) from a point-membership test inside(x, y) -> bool
(segment_* / rectangle_free_areas: any dimension, inside(x, y, z) in 3D): crossing fractions along segments, free
(outside) fractions of segments and of axis-aligned cells, the held-edge fractions and the insulated-wall apertures /
merged volume fractions of a DiffusionSystem, the outline pieces of perimeter fluxes, the inside area fractions of
cells / rectangles (FDTD pole weights) and the dual-cell cuts of the em_modes conformal PEC; host numpy only. On a
polar / spherical window the coordinates are the Domain's logical (r - radius_min, angle - angle_min): fractions are
taken along the logical links / faces / cells, outline lengths in the local physical frame."""
from dataclasses import dataclass

import numpy as np

BISECTION_STEPS = 40      # a crossing is located to 2^-40 of its segment
EDGE_SAMPLES = 16         # sub-intervals per segment searched for crossings: features thinner than 1/16 of it are missed
AREA_ROWS = 32            # midpoint rows of the free-area quadrature (rows across the boundary: O(1 / rows^2) at kinks)
SEGMENT_CHUNK = 1 << 16   # segments per vectorised batch (bounds the (segments, samples) temporaries)
# Insulated walls: a face between two open cells keeps at least this aperture (never closes a link the staircase keeps:
# conditioning and the anchor graph stay), a conducting cut cell at least this volume fraction.
APERTURE_FLOOR = 0.02
VOLUME_FLOOR = 0.02
# Outline pieces: every cell is tested on its faces inset by this share of its width (an outline on a cell face then
# shows as a face piece, a shape ending on the window edge has none there); cell pieces shorter than OUTLINE_FLOOR of
# the cell diagonal and face pieces below FACE_OUTLINE_FLOOR of their face are dropped; cell chords are refined by
# OUTLINE_REFINEMENT_LEVELS rounds of bisector points (a smooth arc's chord deficit / 4 per round).
EDGE_INSET = 1e-9
OUTLINE_FLOOR = 1e-9
FACE_OUTLINE_FLOOR = 1e-6
OUTLINE_REFINEMENT_LEVELS = 2
# em_modes conformal PEC: smallest free fraction of a cut centre link / dual cell (conditioning of the eigen solve).
DUAL_CUT_FLOOR = 0.02
# Neighbour offsets (di, dj) of a cell in the sliver host search: left, right, below, above.
_FOUR_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL_NEIGHBOURS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def segment_crossings(inside, starts, ends, start_inside):
    """float64 fraction t in [0, 1] from start to end where inside(*point) changes (bisection; each end must differ
    from its start): starts / ends = tuples of coordinate arrays (any dimension), start_inside = the start memberships."""
    starts = tuple(np.asarray(value, dtype=np.float64) for value in starts)
    steps = tuple(np.asarray(end, dtype=np.float64) - start for start, end in zip(starts, ends))
    start_inside = np.asarray(start_inside, dtype=bool)
    shape = np.broadcast(*starts, *steps).shape
    lower = np.zeros(shape)
    upper = np.ones(shape)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        points = tuple(start + middle * step for start, step in zip(starts, steps))
        same = np.asarray(inside(*points), dtype=bool) == start_inside
        lower = np.where(same, middle, lower)
        upper = np.where(same, upper, middle)
    return 0.5 * (lower + upper)


def crossing_fractions(inside, start_x, start_y, end_x, end_y, start_inside):
    """segment_crossings of 2D segments (inside(x, y))."""
    return segment_crossings(inside, (start_x, start_y), (end_x, end_y), start_inside)


def segment_free_fractions(inside, starts, ends, samples=EDGE_SAMPLES):
    """float64 fraction of each segment start -> end (tuples of coordinate arrays, any dimension) lying outside the
    region (inside False): `samples` equal sub-intervals, each with at most one crossing located by bisection."""
    starts = tuple(np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel() for value in starts)
    ends = tuple(np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel() for value in ends)
    count = starts[0].size
    result = np.empty(count)
    positions = np.linspace(0.0, 1.0, samples + 1)
    for begin in range(0, count, SEGMENT_CHUNK):
        part = slice(begin, begin + SEGMENT_CHUNK)
        points = tuple(start[part, None] + positions[None, :] * (end[part] - start[part])[:, None]
                       for start, end in zip(starts, ends))
        member = np.asarray(inside(*points), dtype=bool)
        first, last = member[:, :-1], member[:, 1:]
        free = (~first & ~last).astype(np.float64)
        rows, columns = np.nonzero(first != last)
        if rows.size:
            crossing = segment_crossings(inside, tuple(point[rows, columns] for point in points),
                                         tuple(point[rows, columns + 1] for point in points), first[rows, columns])
            free[rows, columns] = np.where(first[rows, columns], 1.0 - crossing, crossing)
        result[part] = free.mean(axis=1)
    return result


def free_fractions(inside, start_x, start_y, end_x, end_y, samples=EDGE_SAMPLES):
    """segment_free_fractions of 2D segments (inside(x, y))."""
    return segment_free_fractions(inside, (start_x, start_y), (end_x, end_y), samples)


def rectangle_free_areas(inside, lo, hi, axis_a, axis_b, rows_along_a, rows=AREA_ROWS):
    """float64 free (outside) area fraction of axis-aligned rectangles lo..hi (tuples of coordinate arrays, any
    dimension; lo = hi off the plane axes axis_a, axis_b) by midpoint quadrature of segment_free_fractions over `rows`
    segments: along axis_a where rows_along_a (rows across a steep boundary), else along axis_b."""
    lo = tuple(np.atleast_1d(np.asarray(value, dtype=np.float64)) for value in lo)
    hi = tuple(np.atleast_1d(np.asarray(value, dtype=np.float64)) for value in hi)
    count = lo[0].size
    along_a = np.broadcast_to(np.asarray(rows_along_a, dtype=bool), (count,))[:, None]
    offsets = (np.arange(rows) + 0.5) / rows
    starts, ends = [], []
    for axis, (low, high) in enumerate(zip(lo, hi)):
        if axis in (axis_a, axis_b):
            across = low[:, None] + offsets[None, :] * (high - low)[:, None]   # row positions across this axis
            segment_axis = along_a if axis == axis_a else ~along_a            # rows run along this axis
            start, end = np.where(segment_axis, low[:, None], across), np.where(segment_axis, high[:, None], across)
        else:
            start = end = np.broadcast_to(low[:, None], (count, rows))
        starts.append(start.ravel())
        ends.append(end.ravel())
    return segment_free_fractions(inside, starts, ends).reshape(count, rows).mean(axis=1)


def free_areas(inside, x_lo, x_hi, y_lo, y_hi, rows_along_x, rows=AREA_ROWS):
    """float64 free (outside) area fraction of each axis-aligned 2D cell (rectangle_free_areas): rows along x where
    rows_along_x (boundary steeper than 45 degrees), else along y."""
    return rectangle_free_areas(inside, (x_lo, y_lo), (x_hi, y_hi), 0, 1, rows_along_x, rows)


def held_edge_fractions(region, held, x_centres, y_centres, inside):
    """(t_x, located_x, t_y, located_y) of the interior faces (nx-1, ny) / (nx, ny-1) between a `region` cell and a
    cell not `held` (host bool windows): t = distance of the drawn edge of inside(x, y) from the free centre over the
    centre spacing, located only where the link leaves the region (free centre outside, held centre inside)."""
    results = []
    for axis in (0, 1):
        lower = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        upper = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        fraction = np.full(region[lower].shape, np.nan)
        located = np.zeros(region[lower].shape, dtype=bool)
        step_i, step_j = (1, 0) if axis == 0 else (0, 1)
        for held_lower, chosen in ((True, region[lower] & ~held[upper]), (False, region[upper] & ~held[lower])):
            face_i, face_j = np.nonzero(chosen)
            free_i, free_j = (face_i + step_i, face_j + step_j) if held_lower else (face_i, face_j)
            held_i, held_j = (face_i, face_j) if held_lower else (face_i + step_i, face_j + step_j)
            start_x, start_y, end_x, end_y = x_centres[free_i], y_centres[free_j], x_centres[held_i], y_centres[held_j]
            start_inside = np.asarray(inside(start_x, start_y), dtype=bool)
            straddles = ~start_inside & np.asarray(inside(end_x, end_y), dtype=bool)
            if straddles.any():
                fraction[face_i[straddles], face_j[straddles]] = crossing_fractions(
                    inside, start_x[straddles], start_y[straddles], end_x[straddles], end_y[straddles],
                    start_inside[straddles])
                located[face_i[straddles], face_j[straddles]] = True
        results += [fraction, located]
    return tuple(results)


def locate_held_edges(system, region_window, inside):
    """Cut cells of a 2D core.fv_diffusion.DiffusionSystem: the drawn edge of the just-held region (window mask) on
    every link from a free cell into it, stored by system.set_held_wall (host bisection, one transfer each way)."""
    from .backend import to_host
    region = np.asarray(to_host(region_window), dtype=bool)
    held = np.asarray(to_host(system.fixed_user), dtype=bool)
    x_centres, y_centres = (np.asarray(centres, dtype=np.float64)[window]
                            for centres, window in zip(system.domain.cell_centers(), system.window))
    fraction_x, located_x, fraction_y, located_y = held_edge_fractions(region, held, x_centres, y_centres, inside)
    system.set_held_wall(fraction_x, fraction_y, located_x, located_y)


def _dilated(mask, diagonal):
    """Bool mask grown by one cell into its 8 (diagonal) or 4 axis neighbours."""
    n_x, n_y = mask.shape
    padded = np.pad(mask, 1)
    grown = mask.copy()
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di or dj) and (diagonal or not (di and dj)):
                grown |= padded[1 + di:1 + di + n_x, 1 + dj:1 + dj + n_y]
    return grown


def insulated_wall_geometry(open_cells, free_cells, x_edges, y_edges, cell_areas, insulating):
    """(aperture_x (nx-1, ny), aperture_y (nx, ny-1), volume_fraction (nx, ny)) host float64 of the curved insulated
    walls inside insulating(x, y), None when no free cell borders a closed one. open_cells: conducting or held, free_cells:
    conducting and not held (host bool windows), cell_areas the metric volumes. Faces between open cells near a cut keep
    their free fraction (>= APERTURE_FLOOR); a free cut cell (a closed one among its 8 neighbours) gets its free area
    (>= VOLUME_FLOOR) plus the free slivers of closed cells merged into it across their most open face."""
    open_cells = np.asarray(open_cells, dtype=bool)
    free_cells = np.asarray(free_cells, dtype=bool)
    closed = ~open_cells
    wall = free_cells & _dilated(closed, True)
    if not wall.any():
        return None
    sliver = closed & _dilated(free_cells, False)
    touched = wall | sliver
    x_edges, y_edges = np.asarray(x_edges, dtype=np.float64), np.asarray(y_edges, dtype=np.float64)
    cell_areas = np.asarray(cell_areas, dtype=np.float64)
    n_x, n_y = open_cells.shape

    fraction_x = np.ones((n_x - 1, n_y))   # free fraction of the interior faces touching a cut cell
    face_x = touched[:-1, :] | touched[1:, :]
    i, j = np.nonzero(face_x)
    if i.size:
        fraction_x[i, j] = free_fractions(insulating, x_edges[i + 1], y_edges[j], x_edges[i + 1], y_edges[j + 1])
    fraction_y = np.ones((n_x, n_y - 1))
    face_y = touched[:, :-1] | touched[:, 1:]
    i, j = np.nonzero(face_y)
    if i.size:
        fraction_y[i, j] = free_fractions(insulating, x_edges[i], y_edges[j + 1], x_edges[i + 1], y_edges[j + 1])
    aperture_x = np.where(face_x & open_cells[:-1, :] & open_cells[1:, :], np.maximum(fraction_x, APERTURE_FLOOR), 1.0)
    aperture_y = np.where(face_y & open_cells[:, :-1] & open_cells[:, 1:], np.maximum(fraction_y, APERTURE_FLOOR), 1.0)

    # free area of every touched cell; rows along x where its two x faces differ more than its y faces (steep wall)
    padded_x = np.ones((n_x + 1, n_y))
    padded_x[1:-1, :] = fraction_x
    padded_y = np.ones((n_x, n_y + 1))
    padded_y[:, 1:-1] = fraction_y
    rows_along_x = np.abs(padded_x[:-1, :] - padded_x[1:, :]) >= np.abs(padded_y[:, :-1] - padded_y[:, 1:])
    i, j = np.nonzero(touched)
    free_area = np.zeros((n_x, n_y))
    free_area[i, j] = free_areas(insulating, x_edges[i], x_edges[i + 1], y_edges[j], y_edges[j + 1],
                                 rows_along_x[i, j])
    volume_fraction = np.where(wall, np.maximum(free_area, VOLUME_FLOOR), 1.0)

    # small-cell merging: a closed cut cell's free sliver joins its free 4-neighbour across the most open face
    i, j = np.nonzero(sliver & (free_area > 0.0))
    scores = np.full((len(_FOUR_NEIGHBOURS), i.size), -1.0)
    for slot, (di, dj) in enumerate(_FOUR_NEIGHBOURS):
        if (di and n_x < 2) or (dj and n_y < 2):
            continue
        neighbour_i, neighbour_j = i + di, j + dj
        in_window = (neighbour_i >= 0) & (neighbour_i < n_x) & (neighbour_j >= 0) & (neighbour_j < n_y)
        clipped_i, clipped_j = np.clip(neighbour_i, 0, n_x - 1), np.clip(neighbour_j, 0, n_y - 1)
        if di:
            face = fraction_x[np.clip(np.minimum(i, neighbour_i), 0, n_x - 2), j]
        else:
            face = fraction_y[i, np.clip(np.minimum(j, neighbour_j), 0, n_y - 2)]
        scores[slot] = np.where(in_window & free_cells[clipped_i, clipped_j], face, -1.0)
    best = np.argmax(scores, axis=0)
    hosted = scores[best, np.arange(i.size)] >= 0.0   # every sliver borders a free cell by construction
    offsets = np.asarray(_FOUR_NEIGHBOURS, dtype=np.int64)[best]
    host_i, host_j = i + offsets[:, 0], j + offsets[:, 1]
    i, j, host_i, host_j = i[hosted], j[hosted], host_i[hosted], host_j[hosted]
    volume_fraction += np.bincount(np.ravel_multi_index((host_i, host_j), volume_fraction.shape),
                                   free_area[i, j] * cell_areas[i, j] / cell_areas[host_i, host_j],
                                   minlength=volume_fraction.size).reshape(volume_fraction.shape)
    return aperture_x, aperture_y, volume_fraction


def locate_insulated_walls(system, conducting_window, material_index):
    """insulated_wall_geometry of a 2D core.fv_diffusion.DiffusionSystem's curved insulated walls (None = none):
    closed = neither conducting (bool window) nor held, insulating(x, y) = a point of a material whose cells (by centre)
    are mostly closed (material_index(x, y): model.boundary_locator, duck-typed); host work."""
    from .backend import to_host
    held = np.asarray(to_host(system.fixed_user), dtype=bool)
    conducting = np.asarray(to_host(conducting_window), dtype=bool)
    open_cells = conducting | held
    free_cells = conducting & ~held
    if open_cells.all() or not free_cells.any():
        return None
    x_edges, y_edges = (np.asarray(edges, dtype=np.float64)[window.start:window.stop + 1]
                        for edges, window in zip(system.domain.cell_edges(), system.window))
    x_centres, y_centres = 0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:])
    index = np.asarray(material_index(x_centres[:, None], y_centres[None, :])).astype(np.int64).ravel()
    counts = np.bincount(index)
    open_votes = np.bincount(index, weights=open_cells.ravel(), minlength=counts.size)
    open_material = (open_votes > 0.5 * counts) | (counts == 0)   # a material at no centre counts as open

    def insulating(x, y):
        material = np.asarray(material_index(x, y)).astype(np.int64)
        return (material < open_material.size) & ~open_material[np.minimum(material, open_material.size - 1)]
    return insulated_wall_geometry(open_cells, free_cells, x_edges, y_edges, system.metric.volume.full(), insulating)


@dataclass(frozen=True, eq=False)
class OutlinePieces:
    """The drawn outline of a region in pieces (host float64 arrays): owning window cell, length [m] (cell piece: its
    chord refined by bisector points; face piece: the part of a cell face the outline runs along), unit outward normal
    (grid frame: radial / angular on a polar window), a point of the piece in window coordinates (cell piece: mean of
    its cell-boundary transitions) and the distance [m] from the owner's centre to the piece along the normal (>= 0)."""
    owner_i: np.ndarray
    owner_j: np.ndarray
    length: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    point_x: np.ndarray
    point_y: np.ndarray
    offset: np.ndarray


def _empty_pieces():
    empty = np.zeros(0)
    return OutlinePieces(np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), *([empty] * 6))


def _scaled_test(inside, y_scale, owner):
    """inside(x, y) of points given as (x, y_scale[owner] y) (the local physical frame of a polar piece)."""
    def test(x, scaled_y):
        return inside(x, scaled_y / y_scale[owner])
    return test


def _refined_length(inside, start_x, start_y, end_x, end_y, normal_x, normal_y, bounds, levels, y_scale=None):
    """Outline length between two of its points in a cell: the chord refined `levels` times by the outline point on
    each sub-chord's perpendicular bisector (bisection from its midpoint toward the outline, within the cell bounds
    (left, right, bottom, top)); a sub-chord whose bisector meets no crossing stays straight. y_scale (per piece):
    lengths in the frame (x, y_scale y) (polar: the piece radius), normals given in that frame."""
    left, right, bottom, top = bounds
    n_pieces = start_x.size
    if y_scale is not None:   # work in the scaled frame (x, s y), testing points mapped back
        start_y, end_y, bottom, top = start_y * y_scale, end_y * y_scale, bottom * y_scale, top * y_scale
    owner = np.arange(n_pieces)
    for _ in range(levels):
        test = inside if y_scale is None else _scaled_test(inside, y_scale, owner)
        step_x, step_y = end_x - start_x, end_y - start_y
        chord = np.hypot(step_x, step_y)
        safe_chord = np.where(chord > 0.0, chord, 1.0)
        perpendicular_x, perpendicular_y = step_y / safe_chord, -step_x / safe_chord
        flip = perpendicular_x * normal_x[owner] + perpendicular_y * normal_y[owner] < 0.0   # on the outward side
        perpendicular_x = np.where(flip, -perpendicular_x, perpendicular_x)
        perpendicular_y = np.where(flip, -perpendicular_y, perpendicular_y)
        middle_x, middle_y = 0.5 * (start_x + end_x), 0.5 * (start_y + end_y)
        middle_inside = np.asarray(test(middle_x, middle_y), dtype=bool)
        direction = np.where(middle_inside, 1.0, -1.0)   # a midpoint inside the region: the outline lies outward
        search_x, search_y = direction * perpendicular_x, direction * perpendicular_y
        reach = np.full(chord.shape, np.inf)
        for search, middle, low, high in ((search_x, middle_x, left[owner], right[owner]),
                                          (search_y, middle_y, bottom[owner], top[owner])):
            safe = np.where(search != 0.0, search, 1.0)
            reach = np.minimum(reach, np.where(search > 0.0, (high - middle) / safe,
                                               np.where(search < 0.0, (low - middle) / safe, np.inf)))
        distance = np.clip(np.minimum(chord, reach), 0.0, None)
        far_x, far_y = middle_x + distance * search_x, middle_y + distance * search_y
        found = (distance > 0.0) & (np.asarray(test(far_x, far_y), dtype=bool) != middle_inside)
        if not found.any():
            break
        found_test = inside if y_scale is None else _scaled_test(inside, y_scale, owner[found])
        fraction = crossing_fractions(found_test, middle_x[found], middle_y[found], far_x[found], far_y[found],
                                      middle_inside[found])
        point_x = middle_x[found] + fraction * distance[found] * search_x[found]
        point_y = middle_y[found] + fraction * distance[found] * search_y[found]
        start_x, start_y, end_x, end_y = (
            np.concatenate((start_x[~found], start_x[found], point_x)),
            np.concatenate((start_y[~found], start_y[found], point_y)),
            np.concatenate((end_x[~found], point_x, end_x[found])),
            np.concatenate((end_y[~found], point_y, end_y[found])))
        owner = np.concatenate((owner[~found], owner[found], owner[found]))
    return np.bincount(owner, weights=np.hypot(end_x - start_x, end_y - start_y), minlength=n_pieces)


def near_centre_changes(region):
    """Bool mask of the cells within one 8-neighbour of a centre-membership change of region (host bool window): the
    candidate cut cells of a drawn outline."""
    region = np.asarray(region, dtype=bool)
    changes = np.zeros(region.shape, dtype=bool)
    differs_x = region[:-1, :] != region[1:, :]
    differs_y = region[:, :-1] != region[:, 1:]
    changes[:-1, :] |= differs_x
    changes[1:, :] |= differs_x
    changes[:, :-1] |= differs_y
    changes[:, 1:] |= differs_y
    return _dilated(changes, True)


def rectangle_inside_fractions(inside, x_lo, x_hi, y_lo, y_hi):
    """float64 inside (region) area fraction of axis-aligned 2D rectangles: free_areas with rows along x where the two
    constant-x sides differ more in free fraction than the two constant-y sides (rows across the steeper boundary)."""
    left = free_fractions(inside, x_lo, y_lo, x_lo, y_hi)
    right = free_fractions(inside, x_hi, y_lo, x_hi, y_hi)
    bottom = free_fractions(inside, x_lo, y_lo, x_hi, y_lo)
    top = free_fractions(inside, x_lo, y_hi, x_hi, y_hi)
    rows_along_x = np.abs(left - right) >= np.abs(bottom - top)
    return 1.0 - np.clip(free_areas(inside, x_lo, x_hi, y_lo, y_hi, rows_along_x), 0.0, 1.0)


def outline_pieces(region, x_edges, y_edges, inside, refinement_levels=OUTLINE_REFINEMENT_LEVELS, radius_min=None):
    """OutlinePieces of the outline of inside(x, y) near the centre-membership changes of region (host bool window: the
    cells by centre; candidates within one 8-neighbour of a change), every cell tested on its faces inset by EDGE_INSET:
    a cell piece = -sum of the faces' outward normals x inside lengths (the chord; refined by bisector points when it
    has two ends), a face piece = the inside-length difference of the two inset faces along a shared face (outlines on
    cell faces: exact). Owners: the cell when in region, else the region 4-neighbour across the most inside face (a
    diagonal one last; none: dropped); a face piece's more inside cell (else the other; neither in region: dropped).
    radius_min (polar / spherical window: x = r - radius_min, y an angle [rad]): lengths [m], normals (radial, angular)
    and offsets in the local frame (dr, r d_angle) at each piece's radius, so arcs about the origin are exact."""
    region = np.asarray(region, dtype=bool)
    n_x, n_y = region.shape
    x_edges, y_edges = np.asarray(x_edges, dtype=np.float64), np.asarray(y_edges, dtype=np.float64)
    i, j = np.nonzero(near_centre_changes(region))
    if i.size == 0:
        return _empty_pieces()
    polar = radius_min is not None
    width_x, width_y = x_edges[i + 1] - x_edges[i], y_edges[j + 1] - y_edges[j]
    left, right = x_edges[i] + EDGE_INSET * width_x, x_edges[i + 1] - EDGE_INSET * width_x
    bottom, top = y_edges[j] + EDGE_INSET * width_y, y_edges[j + 1] - EDGE_INSET * width_y

    # the four inset faces counter-clockwise (bottom, right, top, left), each from its start corner to the next corner
    corner_x = np.stack((left, right, right, left))
    corner_y = np.stack((bottom, bottom, top, top))
    end_x, end_y = np.roll(corner_x, -1, axis=0), np.roll(corner_y, -1, axis=0)
    free = free_fractions(inside, corner_x.ravel(), corner_y.ravel(), end_x.ravel(), end_y.ravel()).reshape(4, i.size)
    logical_length = np.stack((right - left, top - bottom, right - left, top - bottom))
    face_scale = np.ones((4, i.size))   # physical / logical length of each face (polar: the arcs' radii)
    if polar:
        face_scale[1], face_scale[3] = radius_min + x_edges[i + 1], radius_min + x_edges[i]
    face_length = logical_length * face_scale
    inside_length = (1.0 - free) * face_length   # physical [m]

    # transitions of the boundary walk: crossings inside the cut faces, corners between faces of differing states
    start_inside = np.asarray(inside(corner_x, corner_y), dtype=bool)
    cut = (free > 0.0) & (free < 1.0)
    start_state = np.where(free <= 0.0, True, np.where(free >= 1.0, False, start_inside))
    end_state = np.where(free <= 0.0, True, np.where(free >= 1.0, False, ~start_inside))
    crossing = np.where(start_inside, 1.0 - free, free)
    transition_x = np.concatenate((corner_x + crossing * (end_x - corner_x), corner_x))   # 4 crossings, 4 corners
    transition_y = np.concatenate((corner_y + crossing * (end_y - corner_y), corner_y))
    valid = np.concatenate((cut, np.roll(end_state, 1, axis=0) != start_state))
    count = np.count_nonzero(valid, axis=0)
    point_x = np.where(count > 0, np.sum(np.where(valid, transition_x, 0.0), axis=0) / np.maximum(count, 1),
                       0.5 * (left + right))
    point_y = np.where(count > 0, np.sum(np.where(valid, transition_y, 0.0), axis=0) / np.maximum(count, 1),
                       0.5 * (bottom + top))
    # the chord in the local frame (x, s y), s = the piece radius on a polar window (1 on a Cartesian one)
    scale = radius_min + point_x if polar else np.ones(i.size)
    logical_inside = (1.0 - free) * logical_length
    vector_x = scale * (logical_inside[3] - logical_inside[1])   # -(sum of outward face normals x inside lengths)
    vector_y = logical_inside[0] - logical_inside[2]
    length = np.hypot(vector_x, vector_y)
    safe_length = np.where(length > 0.0, length, 1.0)
    normal_x, normal_y = vector_x / safe_length, vector_y / safe_length
    two_ends = np.flatnonzero((count == 2) & (length > 0.0))
    if refinement_levels > 0 and two_ends.size:
        first, second = np.argsort(~valid[:, two_ends], axis=0, kind='stable')[:2]
        length[two_ends] = _refined_length(
            inside, transition_x[first, two_ends], transition_y[first, two_ends], transition_x[second, two_ends],
            transition_y[second, two_ends], normal_x[two_ends], normal_y[two_ends],
            (left[two_ends], right[two_ends], bottom[two_ends], top[two_ends]), refinement_levels,
            y_scale=scale[two_ends] if polar else None)

    # cell pieces: owned by the cell in region, else by the region neighbour across the face with the longest inside
    face_of_neighbour = {(-1, 0): 3, (1, 0): 1, (0, -1): 0, (0, 1): 2}
    scores = np.full((len(_FOUR_NEIGHBOURS) + len(_DIAGONAL_NEIGHBOURS), i.size), -2.0)
    for slot, (di, dj) in enumerate(_FOUR_NEIGHBOURS + _DIAGONAL_NEIGHBOURS):
        neighbour_i, neighbour_j = i + di, j + dj
        in_window = (neighbour_i >= 0) & (neighbour_i < n_x) & (neighbour_j >= 0) & (neighbour_j < n_y)
        member = in_window & region[np.clip(neighbour_i, 0, n_x - 1), np.clip(neighbour_j, 0, n_y - 1)]
        score = inside_length[face_of_neighbour[(di, dj)]] if (di, dj) in face_of_neighbour else -1.0
        scores[slot] = np.where(member, score, -2.0)
    best = np.argmax(scores, axis=0)
    shifts = np.asarray(_FOUR_NEIGHBOURS + _DIAGONAL_NEIGHBOURS, dtype=np.int64)[best]
    own = region[i, j]
    keep = ((length > OUTLINE_FLOOR * np.hypot(width_x, scale * width_y))
            & (own | (scores[best, np.arange(i.size)] > -2.0)))
    parts = [(np.where(own, i, i + shifts[:, 0])[keep], np.where(own, j, j + shifts[:, 1])[keep], length[keep],
              normal_x[keep], normal_y[keep], point_x[keep], point_y[keep])]

    # face pieces: an outline along a shared face shows as differing inside lengths of the two inset faces
    slot_of = np.full(region.shape, -1, dtype=np.int64)
    slot_of[i, j] = np.arange(i.size)
    # the shared face as the lower cell's right / top face and the upper cell's left / bottom face
    for di, dj, lower_cell_face, upper_cell_face in ((1, 0, 1, 3), (0, 1, 2, 0)):
        lower = np.flatnonzero((i + di < n_x) & (j + dj < n_y))
        upper = slot_of[i[lower] + di, j[lower] + dj]
        lower, upper = lower[upper >= 0], upper[upper >= 0]
        difference = inside_length[lower_cell_face, lower] - inside_length[upper_cell_face, upper]
        found = np.abs(difference) > FACE_OUTLINE_FLOOR * face_length[lower_cell_face, lower]
        lower, upper, difference = lower[found], upper[found], difference[found]
        inner, outer = np.where(difference > 0.0, lower, upper), np.where(difference > 0.0, upper, lower)
        inner_member, outer_member = region[i[inner], j[inner]], region[i[outer], j[outer]]
        owner = np.where(inner_member, inner, outer)[inner_member | outer_member]
        member = inner_member | outer_member
        sign = np.where(difference > 0.0, 1.0, -1.0)[member]   # outward: from the more inside cell across the face
        face_x = x_edges[i[lower] + 1] if di else 0.5 * (x_edges[i[lower]] + x_edges[i[lower] + 1])
        face_y = y_edges[j[lower] + 1] if dj else 0.5 * (y_edges[j[lower]] + y_edges[j[lower] + 1])
        parts.append((i[owner], j[owner], np.abs(difference)[member], sign * di, sign * dj, face_x[member],
                      face_y[member]))

    owner_i, owner_j, length, normal_x, normal_y, point_x, point_y = (np.concatenate(values) for values in zip(*parts))
    owner_i, owner_j = owner_i.astype(np.int64), owner_j.astype(np.int64)
    normal_x, normal_y = normal_x.astype(np.float64), normal_y.astype(np.float64)
    centre_x = 0.5 * (x_edges[owner_i] + x_edges[owner_i + 1])
    centre_y = 0.5 * (y_edges[owner_j] + y_edges[owner_j + 1])
    piece_scale = radius_min + point_x if polar else 1.0
    offset = np.maximum(normal_x * (point_x - centre_x) + normal_y * piece_scale * (point_y - centre_y), 0.0)
    return OutlinePieces(owner_i, owner_j, length, normal_x, normal_y, point_x, point_y, offset)


def outline_cell_sums(pieces, values, shape):
    """float64 (nx, ny) window array of per-piece values summed into their owner cells."""
    size = int(np.prod(shape))
    return np.bincount(np.ravel_multi_index((pieces.owner_i, pieces.owner_j), shape),
                       np.asarray(values, dtype=np.float64), minlength=size).reshape(shape)


def dual_cell_cuts(x_centres, y_centres, solid, inside):
    """(link_x, link_y, area) host float64 (nx, ny) of the drawn solid inside(x, y) on the dual grid of the cell centres:
    link_x[i, j] = free (outside) fraction of the link from centre (i, j) to (i + 1, j) whose centre memberships (solid,
    host bool) differ, link_y likewise along y, area[i, j] = free fraction of the square between centres i..i+1,
    j..j+1 with mixed corners; floored at DUAL_CUT_FLOOR, 1 elsewhere (last row / column: no in-grid link)."""
    solid = np.asarray(solid, dtype=bool)
    x_centres, y_centres = np.asarray(x_centres, dtype=np.float64), np.asarray(y_centres, dtype=np.float64)
    n_x, n_y = solid.shape
    links = []
    raw = []
    for axis in (0, 1):
        fraction = np.ones((n_x, n_y))
        lower = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        upper = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        i, j = np.nonzero(solid[lower] != solid[upper])
        if i.size:
            end_i, end_j = (i + 1, j) if axis == 0 else (i, j + 1)
            fraction[i, j] = free_fractions(inside, x_centres[i], y_centres[j], x_centres[end_i], y_centres[end_j])
        # quadrature-direction heuristic only: links inside the solid are not free (the returned links keep 1 there)
        raw_fraction = fraction.copy()
        raw_fraction[lower] = np.where(solid[lower] & solid[upper], 0.0, fraction[lower])
        raw.append(raw_fraction)
        links.append(np.where((fraction > 0.0) & (fraction < 1.0), np.maximum(fraction, DUAL_CUT_FLOOR), 1.0))
    area = np.ones((n_x, n_y))
    corners = solid[:-1, :-1].astype(np.int8) + solid[1:, :-1] + solid[:-1, 1:] + solid[1:, 1:]
    i, j = np.nonzero((corners > 0) & (corners < 4))
    if i.size:
        link_x, link_y = raw
        # rows along x where the square's two vertical sides (y links) differ more than its horizontal ones (x links)
        rows_along_x = np.abs(link_y[i, j] - link_y[i + 1, j]) >= np.abs(link_x[i, j] - link_x[i, j + 1])
        free = free_areas(inside, x_centres[i], x_centres[i + 1], y_centres[j], y_centres[j + 1], rows_along_x)
        area[i, j] = np.clip(free, DUAL_CUT_FLOOR, 1.0)
    return links[0], links[1], area


def union_test(tests):
    """inside(*coordinates) of the union of several membership tests (2D or 3D)."""
    tests = list(tests)

    def inside(*coordinates):
        result = np.asarray(tests[0](*coordinates), dtype=bool)
        for test in tests[1:]:
            result = result | np.asarray(test(*coordinates), dtype=bool)
        return result
    return inside
