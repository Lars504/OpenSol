"""Density design regions of the optimization study (C7): rho in [0, 1] per design cell -> area-weighted conic filter
-> tanh projection -> eps = eps_min + rho_bar (eps_max - eps_min), and the transposed chain rule of a dJ/d eps per cell.
DesignMap is host numpy only; DesignRegion keeps the cropped run-backend mask and writes the permittivity arrays."""
import math

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE


def _filter_entries(cell_i, cell_j, x_centres, y_centres, x_widths, y_widths, radius):
    """COO (rows, cols, weights) of the filter over the design cells: W_ab = w_ab A_b / sum_b w_ab A_b with w_ab =
    max(0, radius - |c_a - c_b|) and A_b the cell area (rows sum to 1); radius <= 0 gives the identity."""
    n_cells = cell_i.size
    own = np.arange(n_cells)
    if not radius > 0.0:
        return own, own.copy(), np.ones(n_cells)
    i_start, j_start = int(cell_i.min()), int(cell_j.min())
    box_shape = (int(cell_i.max()) - i_start + 1, int(cell_j.max()) - j_start + 1)
    lookup = np.full(box_shape, -1, dtype=np.int64)   # box cell -> design index
    local_i, local_j = cell_i - i_start, cell_j - j_start
    lookup[local_i, local_j] = own
    # index offsets reaching `radius` with the finest box cells (an upper bound on graded meshes)
    reach_i = min(int(math.ceil(radius / float(x_widths[i_start:i_start + box_shape[0]].min()))), box_shape[0] - 1)
    reach_j = min(int(math.ceil(radius / float(y_widths[j_start:j_start + box_shape[1]].min()))), box_shape[1] - 1)
    rows, cols, weights = [], [], []
    for offset_i in range(-reach_i, reach_i + 1):
        neighbour_i = local_i + offset_i
        inside_i = (neighbour_i >= 0) & (neighbour_i < box_shape[0])
        for offset_j in range(-reach_j, reach_j + 1):
            neighbour_j = local_j + offset_j
            inside = inside_i & (neighbour_j >= 0) & (neighbour_j < box_shape[1])
            source = own[inside]
            neighbour = lookup[neighbour_i[inside], neighbour_j[inside]]
            present = neighbour >= 0
            source, neighbour = source[present], neighbour[present]
            distance = np.hypot(x_centres[cell_i[neighbour]] - x_centres[cell_i[source]],
                                y_centres[cell_j[neighbour]] - y_centres[cell_j[source]])
            weight = (radius - distance) * x_widths[cell_i[neighbour]] * y_widths[cell_j[neighbour]]
            keep = weight > 0.0
            rows.append(source[keep])
            cols.append(neighbour[keep])
            weights.append(weight[keep])
    rows, cols, weights = np.concatenate(rows), np.concatenate(cols), np.concatenate(weights)
    weights /= np.bincount(rows, weights, minlength=n_cells)[rows]   # the own cell (distance 0) keeps every row > 0
    return rows, cols, weights


class DesignMap:
    """Host map of design densities rho (float64, one per design cell (cell_i[k], cell_j[k]) of the 1D centre / width
    arrays) to the permittivity, and of a per-cell dJ/d eps back to dJ/d rho."""

    def __init__(self, cell_i, cell_j, x_centres, y_centres, x_widths, y_widths, eps_min, eps_max, filter_radius=0.0,
                 projection_beta=0.0, projection_eta=0.5):
        for label, value in (('eps_min', eps_min), ('eps_max', eps_max), ('filter_radius', filter_radius),
                             ('projection_beta', projection_beta), ('projection_eta', projection_eta)):
            if not math.isfinite(float(value)):
                raise ValueError(f'design region {label} must be finite, got {value!r}')
        if float(projection_beta) < 0.0 or not 0.0 <= float(projection_eta) <= 1.0:
            raise ValueError(f'design region projection_beta must be >= 0 and projection_eta in [0, 1], got '
                             f'{projection_beta!r}, {projection_eta!r}')
        cell_i = np.asarray(cell_i, dtype=np.int64)
        cell_j = np.asarray(cell_j, dtype=np.int64)
        if cell_i.size == 0 or cell_i.shape != cell_j.shape:
            raise ValueError('design region needs at least one cell (equal-length cell index arrays)')
        self.n_cells = int(cell_i.size)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
        self.projection_beta = float(projection_beta)
        self.projection_eta = float(projection_eta)
        self._rows, self._cols, self._weights = _filter_entries(
            cell_i, cell_j, np.asarray(x_centres, dtype=np.float64), np.asarray(y_centres, dtype=np.float64),
            np.asarray(x_widths, dtype=np.float64), np.asarray(y_widths, dtype=np.float64), float(filter_radius))

    def filtered(self, density):
        """Filtered density rho_tilde = W rho."""
        return np.bincount(self._rows, self._weights * density[self._cols], minlength=self.n_cells)

    def projected(self, filtered_density):
        """(rho_bar, d rho_bar / d rho_tilde) of the tanh projection; beta 0 = identity."""
        beta, eta = self.projection_beta, self.projection_eta
        if beta == 0.0:
            return filtered_density, np.ones_like(filtered_density)
        denominator = math.tanh(beta * eta) + math.tanh(beta * (1.0 - eta))
        steep = np.tanh(beta * (filtered_density - eta))
        return (math.tanh(beta * eta) + steep) / denominator, beta * (1.0 - steep ** 2) / denominator

    def physical_density(self, density):
        """rho_bar (filtered, projected) of densities rho."""
        return self.projected(self.filtered(density))[0]

    def permittivity(self, density):
        """eps = eps_min + rho_bar (eps_max - eps_min) per design cell."""
        return self.eps_min + self.physical_density(density) * (self.eps_max - self.eps_min)

    def density_gradient(self, density, eps_gradient):
        """dJ/d rho of a per-cell dJ/d eps at densities rho (interpolation, projection, then the transposed filter)."""
        _, slope = self.projected(self.filtered(density))
        filtered_gradient = eps_gradient * (self.eps_max - self.eps_min) * slope
        return np.bincount(self._cols, self._weights * filtered_gradient[self._rows], minlength=self.n_cells)


class DesignRegion:
    """One design_region of an FDFDPhysics / EMEPhysics: window cells of a mask, its DesignMap, the densities (host
    float64) written as eps into `array_names`, and the per-iteration history the optimization study records."""

    def __init__(self, domain, mask, array_names, eps_min, eps_max, initial_density=0.5, filter_radius=0.0,
                 projection_beta=0.0, projection_eta=0.5, objective_unit='W/m'):
        initial_density = float(initial_density)
        if not 0.0 <= initial_density <= 1.0:
            raise ValueError(f'design region initial_density must be in [0, 1], got {initial_density!r}')
        self._xp = domain.xp
        window = domain.physical_window()
        window_mask = np.zeros(mask.shape, dtype=bool)
        window_mask[window] = to_host(mask[window])   # one transfer at build (gpu)
        cell_i, cell_j = np.nonzero(window_mask)
        if cell_i.size == 0:
            raise ValueError('design region selects no cell of the physical window')
        self.box = (slice(int(cell_i.min()), int(cell_i.max()) + 1), slice(int(cell_j.min()), int(cell_j.max()) + 1))
        self.host_mask = window_mask[self.box].copy()   # numpy bool, box-shaped; C order = the density order
        self.box_mask = self._xp.asarray(self.host_mask)   # run-backend bool, box-shaped
        self.objective_unit = str(objective_unit)   # unit of J in the result: 'W/m' (FDFD), '1' (em_eme fractions)
        self.array_names = tuple(array_names)
        self.n_cells = int(cell_i.size)
        x_centres, y_centres = domain.cell_centers()
        x_widths, y_widths = domain.cell_widths()
        self.map = DesignMap(cell_i, cell_j, x_centres, y_centres, x_widths, y_widths, eps_min, eps_max,
                             filter_radius, projection_beta, projection_eta)
        self.density = np.full(self.n_cells, initial_density)
        self.history = []   # (iteration, objective J [W/m], grayness) per recorded iterate
        self._materials = domain.materials

    def overlaps(self, other) -> bool:
        """True when both regions write a common array on a common cell."""
        if not set(self.array_names).intersection(other.array_names):
            return False
        start = [max(mine.start, theirs.start) for mine, theirs in zip(self.box, other.box)]
        stop = [min(mine.stop, theirs.stop) for mine, theirs in zip(self.box, other.box)]
        if any(lo >= hi for lo, hi in zip(start, stop)):
            return False
        mine = self.host_mask[start[0] - self.box[0].start:stop[0] - self.box[0].start,
                              start[1] - self.box[1].start:stop[1] - self.box[1].start]
        theirs = other.host_mask[start[0] - other.box[0].start:stop[0] - other.box[0].start,
                                 start[1] - other.box[1].start:stop[1] - other.box[1].start]
        return bool((mine & theirs).any())

    def _box_values(self, values):
        """Run-backend float32 box array holding `values` (one per design cell) on the mask, 0 elsewhere (one
        transfer on the gpu)."""
        box_values = np.zeros(self.host_mask.shape, dtype=DTYPE)
        box_values[self.host_mask] = values
        return self._xp.asarray(box_values)

    def write(self):
        """Write eps of the current densities into every array of array_names on the design cells (the caller calls
        materials_changed)."""
        box_values = self._box_values(self.map.permittivity(self.density))
        for name in self.array_names:
            self._materials.assign_window(name, self.box, self.box_mask, box_values)

    def set_density(self, density):
        """Store densities rho (copied, one per design cell) and write their eps."""
        density = np.array(density, dtype=np.float64)
        if density.shape != (self.n_cells,):
            raise ValueError(f'design region expects {self.n_cells} densities, got shape {density.shape}')
        self.density = density
        self.write()

    def set_projection_beta(self, beta):
        """Change the projection steepness (optimization beta continuation); the next set_density writes its eps."""
        beta = float(beta)
        if not (math.isfinite(beta) and beta >= 0.0):
            raise ValueError(f'design region projection_beta must be finite and >= 0, got {beta!r}')
        self.map.projection_beta = beta

    def gather(self, cell_values):
        """Host float64 values of a full-grid run-backend array on the design cells (density order)."""
        return to_host(cell_values[self.box])[self.host_mask].astype(np.float64)

    def density_gradient(self, eps_gradient):
        """dJ/d rho of a dJ/d eps per design cell at the current densities."""
        return self.map.density_gradient(self.density, eps_gradient)

    def physical_density(self):
        """Host rho_bar (filtered, projected) of the current densities."""
        return self.map.physical_density(self.density)

    def paint_density(self, cell_array):
        """Write rho_bar into a full-grid run-backend float32 array on the design cells."""
        self._xp.copyto(cell_array[self.box], self._box_values(self.physical_density()), where=self.box_mask)

    def record(self, iteration, objective):
        """Append (iteration, J [objective_unit], grayness = mean 4 rho_bar (1 - rho_bar)) to the history."""
        physical = self.physical_density()
        self.history.append((int(iteration), float(objective), float(np.mean(4.0 * physical * (1.0 - physical)))))


def paint_regions(regions, shape, xp):
    """float32 `shape` array of the projected density rho_bar of every DesignRegion, 0 elsewhere (fresh array): the
    design_density() of the FDFD, EME and NLFD physics."""
    density = xp.zeros(shape, dtype=DTYPE)
    for region in regions:
        region.paint_density(density)
    return density


def design_region_result(region) -> dict:
    """'xy' result of the recorded iterates versus iteration (0 = initial design): series objective [region
    objective_unit] and grayness [1]; extras of the current design. Never raises: without an iterate one point + note."""
    physical = region.physical_density()
    grayness_now = float(np.mean(4.0 * physical * (1.0 - physical)))
    history = region.history or [(0, float('nan'), grayness_now)]
    iterations, objectives, grayness = (np.array(column, dtype=np.float64) for column in zip(*history))
    note = '' if region.history else 'no iteration recorded: only an optimization study records J'
    return {'kind': 'xy', 'x': iterations, 'x_label': 'Iteration', 'x_unit': '1', 'y_label': 'Objective J',
            'series': {'objective': objectives, 'grayness': grayness},
            'series_units': {'objective': region.objective_unit, 'grayness': '1'}, 'n_cells': region.n_cells,
            'eps_min': region.map.eps_min, 'eps_max': region.map.eps_max,
            'mean_density': float(physical.mean()), 'final_grayness': grayness_now, 'note': note}
