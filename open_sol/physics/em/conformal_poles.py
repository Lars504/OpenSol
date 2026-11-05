"""Conformal pole weights of dispersive (Drude / Lorentz) materials in the 2D FDTD engines (solver_plan S7.6; host
numpy, built once per material region): the pole current of a cell / E face is scaled by the share of its control area
inside the drawn material instead of the staircase 0 / 1. E_z: the cell itself (E_z is tangential to every in-plane
interface, so the arithmetic mean f eps_metal + (1 - f) eps_out is the exact first-order cut-cell permittivity); H_z:
the dual cell of each E_x / E_y face as two half cells, the same width weighting as yee_grid.face_average (exact for an
E tangential to the interface, the staircase's arithmetic mean for a normal one). The background eps stays the cell
mask's (subpixel smoothing averages it)."""
from dataclasses import dataclass

import numpy as np

from ...core.cut_cells import near_centre_changes, rectangle_inside_fractions
from .yee_grid import face_partner, face_shares, grid_cell_widths


@dataclass(frozen=True, eq=False)
class ConformalPoleWeights:
    """Host float64 (nx, ny) pole weights (0..1) of one region: ez `cell` (E_z cells); hz `face_x` (E_x on the hi-y
    faces) / `face_y` (E_y on the hi-x faces); `stability` = the per-cell weight of the Courant check (ez the cell
    weight; hz the largest half-cell share of the cell, which bounds every face weight next to it the way the
    face-averaged eps is bounded by the cell eps)."""
    polarization: str
    stability: np.ndarray
    cell: np.ndarray = None
    face_x: np.ndarray = None
    face_y: np.ndarray = None

    def summary(self) -> str:
        """One line for logs: weighted cells / faces strictly between 0 and 1."""
        arrays = (self.cell,) if self.polarization == 'ez' else (self.face_x, self.face_y)
        count = sum(int(np.count_nonzero((array > 0.0) & (array < 1.0))) for array in arrays)
        return f'conformal poles ({self.polarization}): {count} cut {"cells" if self.polarization == "ez" else "faces"}'


def conformal_pole_weights(domain, polarization, region, inside, periodic=(False, False)) -> ConformalPoleWeights:
    """ConformalPoleWeights of a dispersive region of a Domain: region = its cells by centre (host bool (nx, ny)),
    inside(x, y) its drawn outline; window cells within one 8-neighbour of a centre change take their inside area
    fractions, every other cell (PML ring included) the mask."""
    region = np.asarray(region, dtype=bool)
    n_x, n_y = region.shape
    x_edges, y_edges = (np.asarray(edges, dtype=np.float64) for edges in domain.cell_edges())
    window = domain.physical_window()
    candidates = np.zeros((n_x, n_y), dtype=bool)
    candidates[window] = near_centre_changes(region[window])
    ci, cj = np.nonzero(candidates)
    mask = region.astype(np.float64)
    x_lo, x_hi, y_lo, y_hi = x_edges[ci], x_edges[ci + 1], y_edges[cj], y_edges[cj + 1]

    if polarization == 'ez':
        cell = mask.copy()
        if ci.size:
            cell[ci, cj] = rectangle_inside_fractions(inside, x_lo, x_hi, y_lo, y_hi)
        return ConformalPoleWeights('ez', stability=cell, cell=cell)

    # hz: inside fractions of the four half cells (x lo / hi, y lo / hi) of every candidate cell
    x_mid, y_mid = 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi)
    halves = {name: mask.copy() for name in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    if ci.size:
        for name, bounds in (('x_lo', (x_lo, x_mid, y_lo, y_hi)), ('x_hi', (x_mid, x_hi, y_lo, y_hi)),
                             ('y_lo', (x_lo, x_hi, y_lo, y_mid)), ('y_hi', (x_lo, x_hi, y_mid, y_hi))):
            halves[name][ci, cj] = rectangle_inside_fractions(inside, *bounds)
    cell_dx, cell_dy = grid_cell_widths(domain.grid)
    faces = []
    for axis, widths, lower_half, upper_half in ((1, cell_dy, 'y_hi', 'y_lo'), (0, cell_dx, 'x_hi', 'x_lo')):
        lower_share, upper_share = face_shares(widths, periodic[axis], np.float64)
        shape = (1, -1) if axis == 1 else (-1, 1)
        faces.append(halves[lower_half] * lower_share.reshape(shape)
                     + face_partner(halves[upper_half], axis, periodic[axis]) * upper_share.reshape(shape))
    stability = np.maximum.reduce([halves[name] for name in ('x_lo', 'x_hi', 'y_lo', 'y_hi')])
    return ConformalPoleWeights('hz', stability=stability, face_x=faces[0], face_y=faces[1])
