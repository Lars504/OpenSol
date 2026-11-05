"""Wall links of the interpolated (Bouzidi, linear) bounce-back of laminar_flow (solver_plan S7.6; host numpy): per
fluid window cell next to a solid cell, the wall position along each lattice link from the drawn geometry."""
import numpy as np

from ...core.cut_cells import crossing_fractions
from .lattice import DIRECTION_X, DIRECTION_Y, N_DIRECTIONS

HALFWAY = 0.5   # link fraction of the plain halfway bounce-back (links without a located wall keep it)


def _wrapped_test(solid, periodic_spans):
    """solid(x, y) with the coordinates of each periodic axis (span (lo, hi)) wrapped into [lo, hi)."""
    if all(span is None for span in periodic_spans):
        return solid

    def wrap(value, span):
        # lo + mod(-tiny, period) rounds to hi itself, outside a shape drawn up to the window edge (x < x1): -> lo
        wrapped = span[0] + np.mod(np.asarray(value, dtype=np.float64) - span[0], span[1] - span[0])
        return np.where(wrapped >= span[1], span[0], wrapped)

    def test(*coordinates):
        return solid(*(value if span is None else wrap(value, span)
                       for value, span in zip(coordinates, periodic_spans)))
    return test


def wall_link_fractions(fluid, x_centres, y_centres, solid, periodic_spans=(None, None)):
    """(int32 (wx, wy) row per window cell, -1 without a solid link; float32 (rows, 8) wall fraction delta of links
    q = 1..8: wall distance from the fluid centre toward its solid neighbour n - c_q over the link length, HALFWAY on
    links without a solid in-window neighbour or without a crossing). solid(x, y) -> bool is the drawn solid region;
    periodic_spans: per axis the window (lo, hi) edges of a periodic axis (links wrap across it, as the kernel), else
    None."""
    wx, wy = fluid.shape
    sizes = (wx, wy)
    links = []
    for q in range(1, N_DIRECTIONS):
        cx, cy = int(DIRECTION_X[q]), int(DIRECTION_Y[q])
        i, j = np.nonzero(fluid)
        source = [i - cx, j - cy]   # pull streaming: population q arrives from n - c_q
        shift = [np.zeros(i.size), np.zeros(i.size)]   # unwrapped neighbour position - wrapped one, per axis [m]
        in_window = np.ones(i.size, dtype=bool)
        for axis, span in enumerate(periodic_spans):
            if span is None:
                in_window &= (source[axis] >= 0) & (source[axis] < sizes[axis])
            else:
                shift[axis] = np.floor_divide(source[axis], sizes[axis]) * (span[1] - span[0])
                source[axis] = np.mod(source[axis], sizes[axis])
        i, j = i[in_window], j[in_window]
        source_i, source_j = source[0][in_window], source[1][in_window]
        shift_x, shift_y = shift[0][in_window], shift[1][in_window]
        walled = ~fluid[source_i, source_j]
        links.append((np.full(int(walled.sum()), q), i[walled], j[walled], source_i[walled], source_j[walled],
                      shift_x[walled], shift_y[walled]))
    direction, i, j, source_i, source_j, shift_x, shift_y = (np.concatenate(values) for values in zip(*links))
    rows = np.full((wx, wy), -1, dtype=np.int32)
    if direction.size == 0:
        return rows, np.zeros((0, N_DIRECTIONS - 1), dtype=np.float32)
    cell = i * wy + j
    owners = np.unique(cell)
    rows.ravel()[owners] = np.arange(owners.size, dtype=np.int32)
    solid = _wrapped_test(solid, periodic_spans)
    start_x, start_y = x_centres[i], y_centres[j]
    end_x, end_y = x_centres[source_i] + shift_x, y_centres[source_j] + shift_y
    start_solid = np.asarray(solid(start_x, start_y), dtype=bool)
    straddles = ~start_solid & np.asarray(solid(end_x, end_y), dtype=bool)
    delta = np.full(direction.size, HALFWAY)
    if straddles.any():
        delta[straddles] = crossing_fractions(solid, start_x[straddles], start_y[straddles], end_x[straddles],
                                              end_y[straddles], start_solid[straddles])
    fractions = np.full((owners.size, N_DIRECTIONS - 1), HALFWAY, dtype=np.float32)
    fractions[rows.ravel()[cell], direction - 1] = np.clip(delta, 0.0, 1.0)
    return rows, fractions
