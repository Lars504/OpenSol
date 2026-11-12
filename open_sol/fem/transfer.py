"""Transfers between tensor grids and triangle meshes (mesh plan 2.9 / 1.4), the GridTransfer interface
(apply(array, out=None)) over one host CSR. Kinds: 'interpolate' (fields: bilinear from window cell centres, FE
evaluation; clamped into the source window = edge extension into padding / the PML ring) and 'conservative'
(densities: the window integral is preserved exactly; tensor padding and ring elements get 0). Host numpy (scipy
inside functions); core.backend only inside FEMTransfer.apply (device copies of the matrix)."""
import math

import numpy as np

from .assembly import CsrAccumulator, _weights, element_areas
from .elements import element_geometry, lagrange_values
from .mesher import MERGE_TOLERANCE
from .quadrature import subdivided_rule
from .sampling import locator

TRANSFER_KINDS = ('interpolate', 'conservative')
MAX_SUBDIVISIONS = 16        # sub-triangles per element side of the conservative sample rules
SAMPLE_CHUNK = 1 << 20       # quadrature samples per vectorised block


class FEMTransfer:
    """Linear map source -> target: target.ravel() = matrix @ source.ravel() (host float64 CSR, shapes in the
    attributes); apply() runs on the array's backend with the matrix in the array's float / complex precision."""

    def __init__(self, kind, matrix, source_shape, target_shape):
        if kind not in TRANSFER_KINDS:
            raise ValueError(f'FEMTransfer: kind must be one of {TRANSFER_KINDS}, got {kind!r}')
        self.kind = kind
        self.matrix = matrix.tocsr()
        self.source_shape = tuple(source_shape)
        self.target_shape = tuple(target_shape)
        self._copies = {}   # (backend, dtype) -> matrix on that backend

    @property
    def nnz(self) -> int:
        return int(self.matrix.nnz)

    def apply(self, array, out=None):
        """Target-shaped array of the transferred source array (numpy or cupy); written into out when given."""
        from ..core.backend import array_module_of, sparse_module

        xp = array_module_of(array)
        if tuple(array.shape) != self.source_shape:
            raise ValueError(f'FEMTransfer.apply: source shape {tuple(array.shape)} != {self.source_shape}')
        dtype = np.result_type(array.dtype, np.float32)
        backend = 'cpu' if xp is np else 'gpu'
        matrix = self._copies.get((backend, dtype))
        if matrix is None:
            host = self.matrix.astype(dtype)
            matrix = host if backend == 'cpu' else sparse_module('gpu').csr_matrix(host)
            self._copies[(backend, dtype)] = matrix
        result = (matrix @ array.reshape(-1).astype(dtype, copy=False)).reshape(self.target_shape)
        if out is None:
            return result
        out[...] = result
        return out


# ---- tensor grid helpers ----
class _TensorWindow:
    """Full-grid edges (nx + 1, ny + 1) [m, window coordinates] and the window slices of a tensor grid."""

    def __init__(self, x_edges, y_edges, window):
        self.x_edges = np.asarray(x_edges, dtype=np.float64)
        self.y_edges = np.asarray(y_edges, dtype=np.float64)
        self.nx, self.ny = len(self.x_edges) - 1, len(self.y_edges) - 1
        window_x, window_y = window
        self.ox, self.oy = window_x.start or 0, window_y.start or 0
        stop_x = self.nx if window_x.stop is None else window_x.stop
        stop_y = self.ny if window_y.stop is None else window_y.stop
        self.wx_edges = self.x_edges[self.ox:stop_x + 1]
        self.wy_edges = self.y_edges[self.oy:stop_y + 1]
        self.wx_centres = 0.5 * (self.wx_edges[:-1] + self.wx_edges[1:])
        self.wy_centres = 0.5 * (self.wy_edges[:-1] + self.wy_edges[1:])
        self.n_window = (len(self.wx_centres), len(self.wy_centres))

    @property
    def shape(self):
        return self.nx, self.ny

    def full_index(self, window_x_index, window_y_index):
        return (window_x_index + self.ox) * self.ny + (window_y_index + self.oy)

    def cell_of(self, x, y):
        """Window cell (ix, iy) holding points (an edge point goes to the upper cell, clamped to the window)."""
        ix = np.clip(np.searchsorted(self.wx_edges, x, 'right') - 1, 0, self.n_window[0] - 1)
        iy = np.clip(np.searchsorted(self.wy_edges, y, 'right') - 1, 0, self.n_window[1] - 1)
        return ix, iy

    def cell_measures(self, axisymmetric):
        """Float64 (wx, wy) window cell areas, or int r dr dy for axisymmetric (r, z)."""
        widths_y = np.diff(self.wy_edges)
        if axisymmetric:
            return (0.5 * np.diff(self.wx_edges ** 2))[:, None] * widths_y[None, :]
        return np.diff(self.wx_edges)[:, None] * widths_y[None, :]

    def clamped_centres(self):
        """(nx * ny, 2) full-grid cell centres with padding cells moved to the nearest window centre."""
        centres_x = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        centres_y = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        centres_x = np.clip(centres_x, self.wx_centres[0], self.wx_centres[-1])
        centres_y = np.clip(centres_y, self.wy_centres[0], self.wy_centres[-1])
        grid_x, grid_y = np.meshgrid(centres_x, centres_y, indexing='ij')
        return np.column_stack([grid_x.ravel(), grid_y.ravel()])


def _axis_weights(centres, values):
    """(lower index, upper index, lower weight, upper weight) of linear interpolation between centres, clamped."""
    if len(centres) == 1:
        zeros = np.zeros(len(values), dtype=np.int64)
        return zeros, zeros, np.ones(len(values)), np.zeros(len(values))
    clamped = np.clip(values, centres[0], centres[-1])
    lower = np.clip(np.searchsorted(centres, clamped, 'right') - 1, 0, len(centres) - 2)
    fraction = (clamped - centres[lower]) / (centres[lower + 1] - centres[lower])
    return lower, lower + 1, 1.0 - fraction, fraction


def _csr(rows, cols, values, shape):
    from scipy.sparse import coo_matrix

    if not len(rows):
        return coo_matrix(shape).tocsr()
    return coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=shape).tocsr()


# ---- triangle helpers ----
def _require(kind, target=None):
    if kind not in TRANSFER_KINDS:
        raise ValueError(f'transfer kind must be one of {TRANSFER_KINDS}, got {kind!r}')
    if kind == 'conservative' and target is not None and target.order != 0:
        raise ValueError('a conservative transfer onto triangles needs a P0 (per element) target')


def _window_positions(dofs):
    """Positions (into dofs.elements) of the dof map's elements inside the window (not in the PML ring)."""
    return np.flatnonzero(dofs.mesh.pml_faces[dofs.elements] == 0)


def _covers_window(dofs):
    return bool((dofs.element_position[dofs.mesh.pml_faces == 0] >= 0).all())


def _longest_edge(mesh, elements):
    corners = mesh.points[mesh.triangles[elements]]
    lengths = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)
    return float(lengths.max()) if len(lengths) else 0.0


def _subdivisions(element_size, sample_size):
    if not sample_size > 0.0:
        return 1
    return int(np.clip(math.ceil(element_size / sample_size), 1, MAX_SUBDIVISIONS))


def _samples(dofs, positions, degree, subdivisions, axisymmetric):
    """Yield (positions, points (m, q, 2), weights w det J [x r] (m, q), reference points (q, 2)) per chunk."""
    rule = subdivided_rule(degree, subdivisions)
    chunk = max(1, SAMPLE_CHUNK // len(rule.weights))
    for start in range(0, len(positions), chunk):
        part = positions[start:start + chunk]
        geometry = element_geometry(dofs.mesh, dofs.elements[part], rule.points, dofs.geometry_order)
        yield part, geometry.points, _weights(geometry, rule, axisymmetric), rule.points


def _basis_rows(dofs, points):
    """(point indices, dofs (k, n), basis values (k, n)) of points [m] inside the dof map's elements (clamped into the
    mesh window first)."""
    x0, y0, x1, y1 = dofs.mesh.window
    inset = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
    clamped = np.column_stack([np.clip(points[:, 0], x0 + inset, x1 - inset),
                               np.clip(points[:, 1], y0 + inset, y1 - inset)])
    location = locator(dofs.mesh, dofs.geometry_order).locate(clamped)
    position = np.where(location.found, dofs.element_position[np.maximum(location.element, 0)], -1)
    inside = np.flatnonzero(position >= 0)
    values = lagrange_values(dofs.order, location.reference[inside])
    return inside, dofs.element_dofs[position[inside]], values


def _evaluation_matrix(dofs, points, n_rows, row_of_point=None):
    """CSR (n_rows, dofs.n_dofs) evaluating a Lagrange field at points (row = point index, or row_of_point[point])."""
    inside, element_dofs, values = _basis_rows(dofs, points)
    rows = inside if row_of_point is None else row_of_point[inside]
    rows = np.broadcast_to(rows[:, None], element_dofs.shape).ravel()
    return _csr([rows], [element_dofs.ravel()], [values.ravel()], (n_rows, dofs.n_dofs))


def _lagrange_check(dofs, role):
    if dofs.family != 'lagrange':
        raise ValueError(f'transfer {role}: Lagrange (P0 / P1 / P2) dof maps only, got {dofs.family}')


# ---- public factories ----
def tensor_to_triangles(kind, x_edges, y_edges, window, target, axisymmetric=False) -> FEMTransfer:
    """Transfer from a tensor grid (full-grid edges, window = (slice x, slice y) of its physical cells) to the
    target Lagrange dofs: interpolate = bilinear from window centres at the dof points; conservative (P0 target) =
    sampled overlaps, scaled per cell so each window cell's integral lands on the elements exactly."""
    _require(kind, target=target)
    _lagrange_check(target, 'target')
    grid = _TensorWindow(x_edges, y_edges, window)
    if kind == 'interpolate':
        points = target.dof_points()
        lower_x, upper_x, weight_lx, weight_ux = _axis_weights(grid.wx_centres, points[:, 0])
        lower_y, upper_y, weight_ly, weight_uy = _axis_weights(grid.wy_centres, points[:, 1])
        rows = np.tile(np.arange(target.n_dofs), 4)
        cols = np.concatenate([grid.full_index(ix, iy) for ix, iy in ((lower_x, lower_y), (upper_x, lower_y),
                                                                      (lower_x, upper_y), (upper_x, upper_y))])
        values = np.concatenate([weight_lx * weight_ly, weight_ux * weight_ly, weight_lx * weight_uy,
                                 weight_ux * weight_uy])
        matrix = _csr([rows], [cols], [values], (target.n_dofs, grid.nx * grid.ny))
        return FEMTransfer(kind, matrix, grid.shape, (target.n_dofs,))
    positions = _window_positions(target)
    n_window = grid.n_window[0] * grid.n_window[1]
    smallest_cell = min(float(np.diff(grid.wx_edges).min()), float(np.diff(grid.wy_edges).min()))
    subdivisions = _subdivisions(_longest_edge(target.mesh, target.elements[positions]), smallest_cell)
    rows, cols, values = [], [], []
    for part, points, weights, _ in _samples(target, positions, 2, subdivisions, axisymmetric):
        ix, iy = grid.cell_of(points[..., 0].ravel(), points[..., 1].ravel())
        rows.append(np.repeat(target.element_dofs[part, 0], weights.shape[1]))
        cols.append(ix * grid.n_window[1] + iy)
        values.append(weights.ravel())
    overlap = _csr(rows, cols, values, (target.n_dofs, n_window))
    if _covers_window(target):
        overlap = _normalise_columns(overlap, target, grid, axisymmetric)
    areas = np.zeros(target.n_dofs)
    areas[target.element_dofs[:, 0]] = element_areas(target, axisymmetric)
    coo = overlap.tocoo()
    full_cols = grid.full_index(coo.col // grid.n_window[1], coo.col % grid.n_window[1])
    matrix = _csr([coo.row], [full_cols], [coo.data / areas[coo.row]], (target.n_dofs, grid.nx * grid.ny))
    return FEMTransfer(kind, matrix, grid.shape, (target.n_dofs,))


def _normalise_columns(overlap, target, grid, axisymmetric):
    """Scale each window cell column to the cell's exact measure; cells no sample reached go whole to the element
    holding their centre."""
    from scipy.sparse import diags

    measures = grid.cell_measures(axisymmetric).ravel()
    sums = np.asarray(overlap.sum(axis=0)).ravel()
    scale = np.divide(measures, sums, out=np.zeros_like(measures), where=sums > 0.0)
    overlap = (overlap @ diags(scale)).tocsr()
    empty = np.flatnonzero(sums <= 0.0)
    if len(empty):
        ix, iy = empty // grid.n_window[1], empty % grid.n_window[1]
        centres = np.column_stack([grid.wx_centres[ix], grid.wy_centres[iy]])
        element = locator(target.mesh, target.geometry_order).locate(centres).element
        position = np.where(element >= 0, target.element_position[np.maximum(element, 0)], -1)
        hit = position >= 0
        extra = _csr([target.element_dofs[position[hit], 0]], [empty[hit]], [measures[empty[hit]]], overlap.shape)
        overlap = (overlap + extra).tocsr()
    return overlap


def triangles_to_tensor(kind, source, x_edges, y_edges, window, axisymmetric=False) -> FEMTransfer:
    """Transfer from source Lagrange dofs to a tensor grid: interpolate = FE evaluation at the (clamped) cell centres;
    conservative = degree-4 samples of the source density on its window elements binned into window cells (the window
    integral exact)."""
    _require(kind)
    _lagrange_check(source, 'source')
    grid = _TensorWindow(x_edges, y_edges, window)
    n_cells = grid.nx * grid.ny
    if kind == 'interpolate':
        matrix = _evaluation_matrix(source, grid.clamped_centres(), n_cells)
        return FEMTransfer(kind, matrix, (source.n_dofs,), grid.shape)
    positions = _window_positions(source)
    measures = grid.cell_measures(axisymmetric)
    smallest_cell = min(float(np.diff(grid.wx_edges).min()), float(np.diff(grid.wy_edges).min()))
    subdivisions = _subdivisions(_longest_edge(source.mesh, source.elements[positions]), smallest_cell)
    matrix = CsrAccumulator((n_cells, source.n_dofs))   # COO parts converted in batches, not re-added per chunk
    degree = 6 if axisymmetric else 4   # P2 density x det J (x r): exact on curved elements
    for part, points, weights, reference in _samples(source, positions, degree, subdivisions, axisymmetric):
        ix, iy = grid.cell_of(points[..., 0].ravel(), points[..., 1].ravel())
        basis = lagrange_values(source.order, reference)                        # (q, n)
        local = weights[:, :, None] * basis[None, :, :] / measures[ix, iy].reshape(weights.shape)[:, :, None]
        rows = np.broadcast_to(grid.full_index(ix, iy).reshape(weights.shape)[:, :, None], local.shape)
        cols = np.broadcast_to(source.element_dofs[part][:, None, :], local.shape)
        matrix.add_entries(rows, cols, local)
    return FEMTransfer(kind, matrix.result(), (source.n_dofs,), grid.shape)


def triangles_to_triangles(kind, source, target, axisymmetric=False) -> FEMTransfer:
    """Transfer between Lagrange dofs of two triangle meshes: interpolate = source evaluated at the target dof points;
    conservative (P0 target) = target element means of the source density at sub-quadrature points (total to
    quadrature accuracy: the meshes do not nest)."""
    _require(kind, target=target)
    _lagrange_check(source, 'source')
    _lagrange_check(target, 'target')
    if kind == 'interpolate':
        matrix = _evaluation_matrix(source, target.dof_points(), target.n_dofs)
        return FEMTransfer(kind, matrix, (source.n_dofs,), (target.n_dofs,))
    positions = _window_positions(target)
    source_edges = source.mesh.points[source.mesh.edges]
    shortest_source = float(np.linalg.norm(source_edges[:, 1] - source_edges[:, 0], axis=1).min())
    subdivisions = _subdivisions(_longest_edge(target.mesh, target.elements[positions]), shortest_source)
    areas = np.zeros(target.n_dofs)
    areas[target.element_dofs[:, 0]] = element_areas(target, axisymmetric)
    matrix = CsrAccumulator((target.n_dofs, source.n_dofs))
    for part, points, weights, _ in _samples(target, positions, 2, subdivisions, axisymmetric):
        sample_rows = np.repeat(target.element_dofs[part, 0], weights.shape[1])
        inside, element_dofs, values = _basis_rows(source, points.reshape(-1, 2))
        scale = (weights.ravel()[inside] / areas[sample_rows[inside]])[:, None] * values
        rows = np.broadcast_to(sample_rows[inside][:, None], element_dofs.shape)
        matrix.add_entries(rows, element_dofs, scale)
    return FEMTransfer(kind, matrix.result(), (source.n_dofs,), (target.n_dofs,))
