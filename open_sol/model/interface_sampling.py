"""Interface-aware field sampling (igor 1.8; host numpy, GUI-thread safe): samples along the outline of domain objects
with outward normals, and one-sided values read from the cells of one side only, extrapolated onto the outline.
Windows are Cartesian or polar / spherical (CellFrame: logical cells, drawing-plane outlines)."""
import math
import re
from dataclasses import dataclass, replace

import numpy as np

from .geometry import (bounding_box, domain_map, expand_entities, outline, primitive_names, rasterize, rasterize_window,
                       resolve_geometry)
from .schema import INTERFACE_PROJECTIONS

# Each sample reads the side at two depths along the normal (in local cell sizes h): a weighted least-squares plane
# through the side cells of a 4 x 4 block around the depth point, using only cells whose 8 neighbours lie on that side
# too (so no cell-centre value mixes the two materials); then it extrapolates linearly onto the outline: v = 2.5 v(1.5
# h) - 1.5 v(2.5 h). Exact for fields linear on the side, error ~1.9 (k_n h)^2 for a wave of normal wavenumber k_n.
# The planes are fitted in the drawing plane (the cells' drawing-plane centres), so polar cells stay exact too.
INTERFACE_DEPTHS = (1.5, 2.5)
_EXTRAPOLATION = (INTERFACE_DEPTHS[1] / (INTERFACE_DEPTHS[1] - INTERFACE_DEPTHS[0]),
                  -INTERFACE_DEPTHS[0] / (INTERFACE_DEPTHS[1] - INTERFACE_DEPTHS[0]))
SAMPLE_SPACING = 0.5              # outline sample spacing [finest window cell]
MAX_SAMPLES = 1_000_000           # outline samples of one selection
MAX_LINE_SAMPLES = 4000           # interface line profiles: two samples per finest cell, at most this many
_BLOCK = 4                        # cells per axis of the least-squares block around a depth point
_BLOCK_SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))   # tried in order
#                                   until a fit is well posed: near a convex corner the centred block holds one
#                                   eroded column / row only (a degenerate plane)
_CONDITION_FLOOR = 1e-6           # smallest normal-matrix determinant / (trace / 3)^3: else the depth point drops
_SIDE_TEST = 0.1                  # inside / outside test offset along the normal [sample spacing]; must exceed the
#                                   chord sag of the midpoints (<= 0.077 spacing^2 / R) or curved outlines drop out
_SIDE_SIGNS = {'inside': -1.0, 'outside': 1.0}   # depth direction relative to the outward normal
_X_COMPONENT = re.compile(r'(?<![xyz])x(?![a-z0-9])')   # the axis letter of an x component name ('ex_re', 'E_x')


@dataclass(frozen=True, eq=False)
class CellFrame:
    """Cells interface sampling reads: 1D logical centres (Cartesian: x / y [m]); Cartesian box = window (x0, x1, y0,
    y1) [m]; on a polar / spherical window its DomainMap, the (nx, ny) drawing-plane centres and the shift added to
    DomainMap.to_logical (native FieldData centres: r and the angle themselves); basis = the DomainMap whose grid axes
    (radial, angular) the field components follow, None = x / y."""
    x_centres: np.ndarray
    y_centres: np.ndarray
    box: tuple | None = None
    domain_map: object = None
    mapped_x: np.ndarray | None = None
    mapped_y: np.ndarray | None = None
    logical_shift: tuple = (0.0, 0.0)
    basis: object = None

    @property
    def mapped(self):
        return self.domain_map is not None

    def inside(self, points):
        """Bool (n,) of drawing points (n, 2) strictly inside the Cartesian box, or inside the polar window."""
        if self.mapped:
            return np.asarray(self.domain_map.contains(points[:, 0], points[:, 1]), dtype=bool)
        x0, x1, y0, y1 = self.box
        return (points[:, 0] > x0) & (points[:, 0] < x1) & (points[:, 1] > y0) & (points[:, 1] < y1)

    def logical(self, points):
        """(n, 2) positions in the centres' coordinates of drawing points (n, 2)."""
        if not self.mapped:
            return points
        x_logical, y_logical = self.domain_map.to_logical(points[:, 0], points[:, 1])
        return np.column_stack([x_logical + self.logical_shift[0], y_logical + self.logical_shift[1]])

    def positions(self, cells_x, cells_y):
        """Drawing-plane (x, y) [m] of window cells (index arrays of one shape)."""
        if not self.mapped:
            return self.x_centres[cells_x], self.y_centres[cells_y]
        return self.mapped_x[cells_x, cells_y], self.mapped_y[cells_x, cells_y]

    def finest(self):
        """Smallest cell width [m] (polar: dr and the arc r d angle at the outermost centres)."""
        widths = [float(np.diff(self.x_centres).min()), float(np.diff(self.y_centres).min())]
        if self.mapped:
            widths[1] *= self._radius(self.x_centres[-1])
        return min(widths)

    def local_sizes(self, points):
        """Local cell size h (n,) [m] at drawing points (n, 2): the larger cell width there (polar: dr, r d angle)."""
        logical = self.logical(points)
        columns = [np.clip(np.searchsorted(centres, coordinates, side='right') - 1, 0, centres.size - 2)
                   for centres, coordinates in ((self.x_centres, logical[:, 0]), (self.y_centres, logical[:, 1]))]
        width_x = self.x_centres[columns[0] + 1] - self.x_centres[columns[0]]
        width_y = self.y_centres[columns[1] + 1] - self.y_centres[columns[1]]
        if self.mapped:
            width_y = width_y * np.maximum(self._radius(logical[:, 0]), 0.0)
        return np.maximum(width_x, width_y)

    def side_cells(self, name, resolved, side):
        """Host bool (nx, ny) of the cells on `side` of object name (float32 centres, as the build masks)."""
        if self.mapped:
            inside = np.asarray(rasterize(name, resolved, self.mapped_x.astype(np.float32),
                                          self.mapped_y.astype(np.float32), np), dtype=bool)
        else:
            window, cropped = rasterize_window(name, resolved, self.x_centres.astype(np.float32),
                                               self.y_centres.astype(np.float32), np)
            inside = np.zeros((self.x_centres.size, self.y_centres.size), dtype=bool)
            inside[window] = np.asarray(cropped, dtype=bool)
        return inside if side == 'inside' else ~inside

    def grid_components(self, points, vectors):
        """(n, 2) components of drawing-plane vectors (n, 2) at points along the field components' axes."""
        if self.basis is None:
            return vectors
        radial = points / np.maximum(np.hypot(points[:, 0], points[:, 1]), 1e-300)[:, None]
        turn = 1.0 if self.basis.axisymmetric else -1.0   # angular unit: theta from +y (spherical), phi from +x
        angular = np.column_stack([turn * radial[:, 1], -turn * radial[:, 0]])
        return np.column_stack([np.sum(vectors * radial, axis=1), np.sum(vectors * angular, axis=1)])

    def _radius(self, logical_x):
        return logical_x - self.logical_shift[0] + self.domain_map.radius_min


@dataclass(frozen=True, eq=False)
class OneSidedStencil:
    """n samples: points (n, 2) [m], outward unit normals (n, 2), outline length each stands for (n,) [m], linear
    stencils (window cell indices, coefficients: (n, 2, 16)) of the plane fits at the two INTERFACE_DEPTHS; usable
    (n,): both depths found a well-posed fit; grid_normals / grid_tangents (n, 2): normal and tangent (normal turned
    +90 deg) along the field components' axes (radial / angular on a polar grid)."""
    points: np.ndarray
    normals: np.ndarray
    lengths: np.ndarray
    cells_x: np.ndarray
    cells_y: np.ndarray
    weights: np.ndarray
    usable: np.ndarray
    grid_normals: np.ndarray
    grid_tangents: np.ndarray

    def one_sided(self, gathered):
        """float64 (n,) values on the outline from the stencil cell values gathered (n, 2, 16); NaN where not
        usable."""
        gathered = np.where(self.weights != 0.0, np.asarray(gathered, dtype=np.float64), 0.0)
        depth_values = np.sum(self.weights * gathered, axis=2)
        values = _EXTRAPOLATION[0] * depth_values[:, 0] + _EXTRAPOLATION[1] * depth_values[:, 1]
        return np.where(self.usable, values, np.nan)

    def projected(self, x_values, y_values, projection):
        """Normal / tangential part (INTERFACE_PROJECTIONS) of a vector from its one-sided x / y component values."""
        direction = self.grid_normals if projection == 'normal' else self.grid_tangents
        return direction[:, 0] * x_values + direction[:, 1] * y_values

    def variables(self):
        """Per-sample expression variables: x, y [m] and schema.INTERFACE_VARIABLES along the field components' axes."""
        return {'x': self.points[:, 0], 'y': self.points[:, 1],
                'normal_x': self.grid_normals[:, 0], 'normal_y': self.grid_normals[:, 1],
                'tangent_x': self.grid_tangents[:, 0], 'tangent_y': self.grid_tangents[:, 1]}


def window_frame(x_centres, y_centres, box=None, window_map=None, logical_shift=(0.0, 0.0), basis=None):
    """CellFrame of host 1D window centres: Cartesian (box = window (x0, x1, y0, y1) [m]; basis: the DomainMap of
    grid-frame components on a resampled polar image) or the logical cells of a polar / spherical DomainMap."""
    x_centres, y_centres = _checked_centres(x_centres, y_centres)
    if window_map is None or not window_map.mapped:
        return CellFrame(x_centres, y_centres, box=tuple(float(value) for value in box), basis=basis)
    shape = (x_centres.size, y_centres.size)
    mapped_x, mapped_y = (np.ascontiguousarray(np.broadcast_to(np.asarray(values, dtype=np.float64), shape))
                          for values in window_map.to_physical(x_centres[:, None] - logical_shift[0],
                                                               y_centres[None, :] - logical_shift[1]))
    return CellFrame(x_centres, y_centres, domain_map=window_map, mapped_x=mapped_x, mapped_y=mapped_y,
                     logical_shift=tuple(float(value) for value in logical_shift), basis=window_map)


def partner_component(component):
    """y component name of the x component of an in-plane vector ('ex_re' -> 'ey_re', 'E_x' -> 'E_y': its one x
    that is an axis letter, not part of a word or tensor index), None when it has none."""
    matches = list(_X_COMPONENT.finditer(component))
    if len(matches) != 1:
        return None
    index = matches[0].start()
    return f'{component[:index]}y{component[index + 1:]}'


def line_curve_field_keys(curve):
    """'physics.component' keys a line AnalyticCurveSpec reads: its field, plus the y component when projected."""
    if not curve.projection:
        return [curve.field]
    physics, _, component = curve.field.partition('.')
    return [curve.field, f'{physics}.{partner_component(component)}']


def projected_info(info, projection):
    """FieldInfo of the projected profile of an x component (label names the part, signed)."""
    return replace(info, label=f'{info.label}: {projection} part', signed=True)


def interface_selection(spec, entities):
    """The 'interface' SelectionSpec when entities is exactly its name (a derived-value selection), else None."""
    if isinstance(entities, (list, tuple)) and len(entities) == 1:
        for selection in spec.selections:
            if selection.name == entities[0] and selection.kind == 'interface':
                return selection
    return None


def interface_objects(spec, selection):
    """Domain object names of an 'interface' selection (nested selections expanded)."""
    return [name for name in expand_entities(spec, selection.entities) if name != 'all']


def interface_stencil(spec, selection, resolved, frame):
    """OneSidedStencil of an 'interface' selection on the window cells of a CellFrame: the outline samples of each of
    its objects, read from selection.side; ValueError when none is usable."""
    spacing = SAMPLE_SPACING * frame.finest()
    parts = []
    for name in interface_objects(spec, selection):
        points, normals, lengths = object_outline(name, resolved, spacing, frame.inside)
        side_cells = frame.side_cells(name, resolved, selection.side)
        parts.append(_stencil(points, normals, lengths, selection.side, frame, side_cells))
    stencil = _concatenated(parts)
    if not stencil.usable.any():
        raise ValueError(f"interface selection '{selection.name}': no outline sample has cells of the "
                         f"{selection.side} side {INTERFACE_DEPTHS[1]:g} cells deep (outlines on the window edge are "
                         'skipped): refine the mesh near the objects')
    return stencil


def object_outline(name, resolved, spacing, inside):
    """(points (n, 2), outward unit normals (n, 2), lengths (n,)) of samples at most `spacing` [m] apart along the
    outline of domain object name (composites: the parts of their inputs' outlines on the result's boundary); samples
    whose two sides are not both inside(points (n, 2)) -> bool (n,) (e.g. CellFrame.inside) are dropped."""
    pieces = [_resampled(polyline, spacing) for primitive in primitive_names(name, resolved)
              for polyline in outline(primitive, resolved, n_segments=_outline_segments(primitive, resolved, spacing))]
    if not pieces:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty(0)
    points, normals, lengths = (np.concatenate([piece[index] for piece in pieces]) for index in range(3))
    offset = _SIDE_TEST * spacing
    behind, ahead = points - offset * normals, points + offset * normals
    inside_behind = np.asarray(rasterize(name, resolved, behind[:, 0], behind[:, 1], np), dtype=bool)
    inside_ahead = np.asarray(rasterize(name, resolved, ahead[:, 0], ahead[:, 1], np), dtype=bool)
    keep = (inside_behind != inside_ahead) & inside(behind) & inside(ahead)
    normals = np.where((inside_ahead & ~inside_behind)[:, None], -normals, normals)
    return points[keep], normals[keep], lengths[keep]


def interface_line_profile(spec, parameters, interface_name, field_data, start, end, partner_data=None,
                           projection=''):
    """(distance from start [m], one-sided samples, (x, y) sample positions [m]) of a host 2D FieldData along start ->
    end: each sample reads the side of 'interface' selection interface_name with the outward normal of the nearest
    outline sample of its objects, extrapolated to the sample itself (NaN where no side cell is near). projection
    'normal' / 'tangential': field_data is the x component, partner_data the y one (partner_component), sampled alike
    and projected on that normal / its tangent. Native polar / spherical FieldData (mapped_centres) are read on their
    cells; a resampled polar image as pixels (its NaN outside the sector skipped)."""
    selection = next((item for item in spec.selections if item.name == interface_name), None)
    if selection is None or selection.kind != 'interface':
        raise ValueError(f"no interface selection named {interface_name!r}")
    if projection and projection not in INTERFACE_PROJECTIONS:
        raise ValueError(f"projection must be '' or one of {', '.join(INTERFACE_PROJECTIONS)}, got {projection!r}")
    if projection and partner_data is None:
        raise ValueError(f"projection '{projection}' needs the y component's FieldData (partner_data)")
    values = np.asarray(field_data.values, dtype=np.float64)
    frame = _field_frame(spec, parameters, field_data, values.shape)
    readable = np.isfinite(values)
    partner_values = None
    if projection:
        partner_values = np.asarray(partner_data.values, dtype=np.float64)
        if partner_values.shape != values.shape:
            raise ValueError(f'the y component has {partner_values.shape} cells, the x component {values.shape}')
        readable &= np.isfinite(partner_values)
    finest = frame.finest()
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    length = math.dist(start, end)
    if not length > 0.0:
        raise ValueError('the line has zero length: give distinct start and end points')
    n_samples = int(np.clip(2.0 * length / finest, 2, MAX_LINE_SAMPLES))
    distance = np.linspace(0.0, length, n_samples)
    positions = start + (distance / length)[:, None] * (end - start)
    resolved = resolve_geometry(spec, parameters)
    names = interface_objects(spec, selection)
    outlines = [object_outline(name, resolved, SAMPLE_SPACING * finest, frame.inside) for name in names]
    owners = np.concatenate([np.full(points.shape[0], index) for index, (points, _, _) in enumerate(outlines)])
    if owners.size == 0:
        raise ValueError(f"interface selection '{interface_name}' has no outline inside the field window")
    outline_points = np.concatenate([points for points, _, _ in outlines])
    outline_normals = np.concatenate([normals for _, normals, _ in outlines])
    from scipy.spatial import cKDTree   # no (samples x outline) distance blocks: ~GBs at large outlines
    nearest = np.asarray(cKDTree(outline_points).query(positions)[1], dtype=np.int64)
    samples = np.full(n_samples, np.nan)
    for index, name in enumerate(names):
        mine = np.flatnonzero(owners[nearest] == index)
        if not mine.size:
            continue
        side_cells = frame.side_cells(name, resolved, selection.side) & readable
        stencil = _stencil(positions[mine], outline_normals[nearest[mine]], np.zeros(mine.size), selection.side,
                           frame, side_cells)
        one_sided = stencil.one_sided(values[stencil.cells_x, stencil.cells_y])
        if projection:
            one_sided = stencil.projected(one_sided, stencil.one_sided(
                partner_values[stencil.cells_x, stencil.cells_y]), projection)
        samples[mine] = one_sided
    return distance, samples, (positions[:, 0], positions[:, 1])


# ---- helpers ----
def _checked_centres(x_centres, y_centres):
    centres = tuple(np.asarray(axis, dtype=np.float64) for axis in (x_centres, y_centres))
    if min(axis.size for axis in centres) < _BLOCK:
        raise ValueError(f'interface sampling needs at least {_BLOCK} window cells along x and y')
    return centres


def _field_frame(spec, parameters, field_data, shape):
    """CellFrame of a host 2D FieldData: native polar / spherical cells (mapped_centres set), else its centres (or
    uniform cells over its extent), with the grid-frame basis of a polar model's resampled image."""
    window_map = domain_map(spec, parameters)
    if getattr(field_data, 'mapped_centres', None) is not None:   # centres r [m] and the angle [rad] themselves
        return window_frame(field_data.x_centers, field_data.y_centers, window_map=window_map,
                            logical_shift=(window_map.radius_min, window_map.angle_min))
    if field_data.x_centers is not None and field_data.y_centers is not None:
        x_centres, y_centres = field_data.x_centers, field_data.y_centers
    else:
        x0, x1, y0, y1 = (float(value) for value in field_data.extent)
        x_centres = x0 + (np.arange(shape[0]) + 0.5) * (x1 - x0) / shape[0]
        y_centres = y0 + (np.arange(shape[1]) + 0.5) * (y1 - y0) / shape[1]
    return window_frame(x_centres, y_centres, box=field_data.extent,
                        basis=window_map if window_map.mapped else None)


def _outline_segments(name, resolved, spacing):
    """Segments of a circle / ellipse outline no longer than about `spacing` (its bounding-box perimeter bound)."""
    x0, y0, x1, y1 = bounding_box(name, resolved)
    return int(min(max(64, math.ceil(2.0 * ((x1 - x0) + (y1 - y0)) / spacing)), MAX_SAMPLES))


def _resampled(polyline, spacing):
    """(midpoints, unit normals (tangent turned -90 deg), lengths) of the polyline cut into pieces <= spacing."""
    starts, vectors = polyline[:-1], np.diff(polyline, axis=0)
    segment_lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    kept = segment_lengths > 0.0
    starts, vectors, segment_lengths = starts[kept], vectors[kept], segment_lengths[kept]
    counts = np.maximum(1, np.ceil(segment_lengths / spacing)).astype(np.int64)
    if int(counts.sum()) > MAX_SAMPLES:
        raise ValueError(f'an interface outline needs {int(counts.sum()):,} samples (at most {MAX_SAMPLES:,}): '
                         'coarsen the mesh or select a smaller object')
    segment = np.repeat(np.arange(counts.size), counts)
    within = np.arange(segment.size) - np.repeat(np.cumsum(counts) - counts, counts)
    fraction = (within + 0.5) / counts[segment]
    points = starts[segment] + fraction[:, None] * vectors[segment]
    tangents = vectors[segment] / segment_lengths[segment, None]
    return points, np.column_stack([tangents[:, 1], -tangents[:, 0]]), segment_lengths[segment] / counts[segment]


def _eroded(cells):
    """Cells whose 8 neighbours are cells too (edge replicated: the window edge does not erode)."""
    padded = np.pad(cells, 1, mode='edge')
    eroded = cells.copy()
    nx, ny = cells.shape
    for offset_x in range(3):
        for offset_y in range(3):
            eroded &= padded[offset_x:offset_x + nx, offset_y:offset_y + ny]
    return eroded


def _block_cells(coordinates, centres, shift=0):
    """(n, _BLOCK) window cell indices along one axis of the block around each coordinate, moved by shift cells
    (clamped to the window), and whether the coordinate lies inside the window cells."""
    lowest = np.clip(np.searchsorted(centres, coordinates, side='right') - _BLOCK // 2 + shift, 0,
                     centres.size - _BLOCK)
    inside = ((coordinates >= centres[0] - 0.5 * (centres[1] - centres[0]))
              & (coordinates <= centres[-1] + 0.5 * (centres[-1] - centres[-2])))
    return lowest[:, None] + np.arange(_BLOCK)[None, :], inside


def _stencil(points, normals, lengths, side, frame, side_cells):
    """OneSidedStencil of samples at drawing points with outward normals, reading `side` (side_cells: host bool
    window of the frame): per depth point the value of the weighted least-squares plane (weights exp(-(r / h)^2),
    drawing-plane offsets r) through the usable side cells of its (first well-posed) block, as cell coefficients."""
    usable_cells = _eroded(side_cells)
    n_samples = points.shape[0]
    sizes = frame.local_sizes(points)   # local cell size h [m]
    n_cells = _BLOCK * _BLOCK
    cells_x = np.zeros((n_samples, 2, n_cells), dtype=np.int64)
    cells_y = np.zeros((n_samples, 2, n_cells), dtype=np.int64)
    weights = np.zeros((n_samples, 2, n_cells))
    usable = np.ones(n_samples, dtype=bool)
    for depth_index, depth in enumerate(INTERFACE_DEPTHS):
        probes = points + (_SIDE_SIGNS[side] * depth * sizes)[:, None] * normals
        logical = frame.logical(probes)
        posed = np.zeros(n_samples, dtype=bool)
        for shift in _BLOCK_SHIFTS:   # centred block for all, shifted ones where it is degenerate (convex corners)
            retry = np.flatnonzero(~posed)
            if not retry.size:
                break
            cell_x, cell_y, coefficients, fitted = _plane_fit(probes[retry], logical[retry], sizes[retry], frame,
                                                              usable_cells, shift)
            taken = fitted | (shift == _BLOCK_SHIFTS[0])
            rows = retry[taken]
            cells_x[rows, depth_index], cells_y[rows, depth_index] = cell_x[taken], cell_y[taken]
            weights[rows, depth_index] = coefficients[taken]
            posed[rows] = fitted[taken]
        usable &= posed
    tangents = np.column_stack([-normals[:, 1], normals[:, 0]])
    return OneSidedStencil(points, normals, lengths, cells_x, cells_y, weights, usable,
                           frame.grid_components(points, normals), frame.grid_components(points, tangents))


def _plane_fit(probes, logical, sizes, frame, usable_cells, shift):
    """(cells_x, cells_y (n, 16), coefficients (n, 16), posed (n,)) of the weighted least-squares plane value at
    drawing points probes through the usable cells of their block moved by shift = (cells x, cells y)."""
    block_x, inside_x = _block_cells(logical[:, 0], frame.x_centres, shift[0])
    block_y, inside_y = _block_cells(logical[:, 1], frame.y_centres, shift[1])
    cell_x, cell_y = np.repeat(block_x, _BLOCK, axis=1), np.tile(block_y, (1, _BLOCK))   # (n, 16)
    centre_x, centre_y = frame.positions(cell_x, cell_y)
    offset_x = (centre_x - probes[:, :1]) / sizes[:, None]
    offset_y = (centre_y - probes[:, 1:]) / sizes[:, None]
    fit_weights = np.where(usable_cells[cell_x, cell_y], np.exp(-(offset_x ** 2 + offset_y ** 2)), 0.0)
    # the symmetric normal matrix M of the design (1, dx / h, dy / h) written out (no per-point LAPACK calls)
    s_0, s_x, s_y = fit_weights.sum(axis=1), (fit_weights * offset_x).sum(axis=1), (fit_weights * offset_y).sum(axis=1)
    s_xx = (fit_weights * offset_x * offset_x).sum(axis=1)
    s_xy = (fit_weights * offset_x * offset_y).sum(axis=1)
    s_yy = (fit_weights * offset_y * offset_y).sum(axis=1)
    cofactor_0 = s_xx * s_yy - s_xy * s_xy   # first adjugate column: M^-1 e_1 = (c_0, c_x, c_y) / det
    cofactor_x = s_xy * s_y - s_x * s_yy
    cofactor_y = s_x * s_xy - s_xx * s_y
    determinant = s_0 * cofactor_0 + s_x * cofactor_x + s_y * cofactor_y
    scale = (s_0 + s_xx + s_yy) / 3.0
    posed = (determinant > _CONDITION_FLOOR * scale ** 3) & (scale > 0.0) & inside_x & inside_y
    inverse_det = np.where(posed, 1.0 / np.where(posed, determinant, 1.0), 0.0)
    coefficients = fit_weights * (cofactor_0[:, None] + cofactor_x[:, None] * offset_x
                                  + cofactor_y[:, None] * offset_y) * inverse_det[:, None]
    return cell_x, cell_y, coefficients, posed


_STENCIL_FIELDS = ('points', 'normals', 'lengths', 'cells_x', 'cells_y', 'weights', 'usable', 'grid_normals',
                   'grid_tangents')


def _concatenated(parts):
    """One OneSidedStencil of several (in order)."""
    if not parts:
        empty = np.empty((0, 2, _BLOCK * _BLOCK))
        return OneSidedStencil(np.empty((0, 2)), np.empty((0, 2)), np.empty(0), empty.astype(np.int64),
                               empty.astype(np.int64), empty, np.empty(0, dtype=bool), np.empty((0, 2)),
                               np.empty((0, 2)))
    return OneSidedStencil(*(np.concatenate([getattr(part, name) for part in parts]) for name in _STENCIL_FIELDS))
