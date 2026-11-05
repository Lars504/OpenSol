"""S7.2 grid transfers between two tensor grids spanning one physical window (per-physics meshes): one sparse matrix
per axis, applied separably on the run backend. 'interpolate' (fields): linear between the source window centres,
clamped (edge extension into padding; exact for linear fields). 'conservative' (densities): overlap of the metric
measure / target cell measure, so the window integral is kept exactly by the float64 host weights (the applied float32
matrices to ~1e-7 relative); target padding cells get 0."""
import numpy as np

from .backend import array_module_of, sparse_module
from .constants import DTYPE

TRANSFER_KINDS = ('interpolate', 'conservative')
# per-axis 1D measure of the cell metric (core.metric): plain length, r dr, r^2 dr (r = origin + logical x) and
# sin(theta) dtheta (theta = origin + logical y)
AXIS_MEASURES = ('length', 'radial', 'radial_squared', 'sine')


def axis_measures(coordinates, axisymmetric, dimensions):
    """((measure, origin), ...) per axis of a window (core.metric.Coordinates): r dr along x of (r, z) and (r, phi)
    windows, r^2 dr and sin(theta) dtheta on spherical (r, theta) ones, length elsewhere (z of a sector too)."""
    if coordinates.polar and axisymmetric:
        measures = [('radial_squared', coordinates.radius_min), ('sine', coordinates.angle_min)]
    elif coordinates.polar or axisymmetric:
        measures = [('radial', coordinates.radius_min), ('length', 0.0)]
    else:
        measures = [('length', 0.0), ('length', 0.0)]
    return tuple(measures + [('length', 0.0)] * (int(dimensions) - 2))


def interval_measure(lower, upper, measure):
    """float64 measure of intervals [lower, upper] (logical coordinates) under one AXIS_MEASURES entry."""
    kind, origin = measure
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    width = upper - lower
    if kind == 'length':
        return width
    if kind == 'radial':
        return width * (origin + 0.5 * (lower + upper))
    if kind == 'radial_squared':
        radius_lower, radius_upper = origin + lower, origin + upper
        return width * (radius_lower * radius_lower + radius_lower * radius_upper + radius_upper * radius_upper) / 3.0
    if kind == 'sine':   # cos(a) - cos(b) without cancellation
        return 2.0 * np.sin(origin + 0.5 * (lower + upper)) * np.sin(0.5 * width)
    raise ValueError(f"axis measure must be one of {', '.join(AXIS_MEASURES)}, got {kind!r}")


def conservative_axis(source_edges, source_window, target_edges, target_window, measure):
    """(rows, columns, weights) float64 host entries of one axis: each piece of the merged window edges maps source
    cell -> target cell with weight measure(piece) / measure(target cell); the target window ends are snapped onto the
    source's (float32 width sums differ by ulps), so the weights of every source cell sum to its measure exactly."""
    source = np.asarray(source_edges[source_window.start:source_window.stop + 1], dtype=np.float64)
    target = np.array(target_edges[target_window.start:target_window.stop + 1], dtype=np.float64)
    target[0], target[-1] = source[0], source[-1]
    merged = np.union1d(source, target)
    lower, upper = merged[:-1], merged[1:]
    middle = 0.5 * (lower + upper)
    source_cell = np.clip(np.searchsorted(source, middle, side='right') - 1, 0, source.size - 2)
    target_cell = np.clip(np.searchsorted(target, middle, side='right') - 1, 0, target.size - 2)
    weights = (interval_measure(lower, upper, measure)
               / interval_measure(target[target_cell], target[target_cell + 1], measure))
    return target_window.start + target_cell, source_window.start + source_cell, weights


def interpolation_axis(source_centres, source_window, target_centres):
    """(rows, columns, weights) float64 host entries of one axis: every target cell (padding included) linear between
    the two source window centres around it, clamped to the first / last one (one entry when the source window has one
    cell)."""
    centres = np.asarray(source_centres[source_window], dtype=np.float64)
    positions = np.asarray(target_centres, dtype=np.float64)
    rows = np.arange(positions.size)
    if centres.size == 1:
        return rows, np.full(positions.size, source_window.start), np.ones(positions.size)
    positions = np.clip(positions, centres[0], centres[-1])
    left = np.clip(np.searchsorted(centres, positions, side='right') - 1, 0, centres.size - 2)
    fraction = (positions - centres[left]) / (centres[left + 1] - centres[left])
    return (np.concatenate((rows, rows)), source_window.start + np.concatenate((left, left + 1)),
            np.concatenate((1.0 - fraction, fraction)))


class GridTransfer:
    """Separable transfer of full-grid float32 arrays from a source to a target tensor grid of one window: per axis a
    float32 CSR matrix (n_target, n_source) on the backend ('gpu' cupyx, 'cpu' scipy), the float64 host originals in
    `host_matrices` (exact weights, self-tests). apply(array, out=None) -> target-shaped float32 array."""

    def __init__(self, kind, source_edges, source_windows, target_edges, target_windows, measures=None,
                 backend='cpu'):
        if kind not in TRANSFER_KINDS:
            raise ValueError(f"grid transfer kind must be one of {', '.join(TRANSFER_KINDS)}, got {kind!r}")
        if not (len(source_edges) == len(target_edges) == len(source_windows) == len(target_windows)):
            raise ValueError('grid transfer: source and target need the same number of axes')
        measures = tuple(measures) if measures is not None else (('length', 0.0),) * len(source_edges)
        import scipy.sparse as host_sparse   # host assembly of the 1D matrices (scipy is required)
        self.kind = kind
        self.backend = backend
        self.source_shape = tuple(len(edges) - 1 for edges in source_edges)
        self.target_shape = tuple(len(edges) - 1 for edges in target_edges)
        self.host_matrices = []
        for axis, (source, source_window, target, target_window) in enumerate(
                zip(source_edges, source_windows, target_edges, target_windows)):
            if kind == 'conservative':
                rows, columns, weights = conservative_axis(source, source_window, target, target_window,
                                                           measures[axis])
            else:
                source_centres = 0.5 * (np.asarray(source[:-1]) + np.asarray(source[1:]))
                target_centres = 0.5 * (np.asarray(target[:-1]) + np.asarray(target[1:]))
                rows, columns, weights = interpolation_axis(source_centres, source_window, target_centres)
            matrix = host_sparse.csr_matrix((weights, (rows, columns)),
                                            shape=(self.target_shape[axis], self.source_shape[axis]))
            matrix.sum_duplicates()
            self.host_matrices.append(matrix)
        device_sparse = sparse_module(backend)
        if backend == 'cpu':
            self._matrices = [matrix.astype(DTYPE) for matrix in self.host_matrices]
        else:
            from .backend import cp
            self._matrices = [device_sparse.csr_matrix(
                (cp.asarray(matrix.data, dtype=DTYPE), cp.asarray(matrix.indices, dtype=np.int32),
                 cp.asarray(matrix.indptr, dtype=np.int32)), shape=matrix.shape) for matrix in self.host_matrices]

    @classmethod
    def between(cls, kind, source_domain, target_domain, axisymmetric=False):
        """Transfer between two Domains / Domain3Ds of one model (edges in one frame: the window's lower corner); the
        measures from the target's coordinates and axisymmetric (r weights of an (r, z) physics); the target's
        backend."""
        measures = axis_measures(target_domain.coordinates, axisymmetric, target_domain.dimensions)
        return cls(kind, source_domain.cell_edges(), source_domain.physical_window(), target_domain.cell_edges(),
                   target_domain.physical_window(), measures, backend=target_domain.backend.name)

    def apply(self, array, out=None):
        """Target-shaped float32 transfer of a full-grid source array (real; float64 is rounded first), into out when
        given (returned); two (3D: three) sparse products along the axes."""
        if tuple(array.shape) != self.source_shape:
            raise ValueError(f'grid transfer: array of shape {tuple(array.shape)}, expected the source grid '
                             f'{self.source_shape}')
        xp = array_module_of(array)
        if xp.iscomplexobj(array):
            raise ValueError('grid transfer: complex arrays are not transferred (real fields and densities only)')
        values = array if array.dtype == DTYPE else array.astype(DTYPE)
        for axis, matrix in enumerate(self._matrices):
            moved = xp.moveaxis(values, axis, 0)
            rest = moved.shape[1:]
            product = matrix @ xp.ascontiguousarray(moved).reshape(moved.shape[0], -1)
            values = xp.moveaxis(xp.asarray(product).reshape((matrix.shape[0],) + rest), 0, axis)
        if out is None:
            return xp.ascontiguousarray(values, dtype=DTYPE)
        out[...] = values
        return out
