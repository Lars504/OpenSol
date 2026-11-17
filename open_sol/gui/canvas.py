"""ModelCanvas: geometry, mesh and field views of a ModelDocument (pyqtgraph) with editable ROIs and draw tools.
GUI thread only: CPU geometry (numpy), document.mesh_estimate(name) of the active mesh, triangle line sets of the
mesh_preview store and FieldData from the worker; never cupy."""
import html
import math
from dataclasses import dataclass, field, replace

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ..core.constants import FACES
from ..core.domain import containing_cell
from ..core.mesh import has_uniform_spacing
from ..model.expressions import ExpressionError, error_message
from ..model.geometry import (GEOMETRY_KINDS, PRISM_KINDS, bounding_box, consumed_objects, domain_map,
                              domain_size_z, expand_entities, is_array_copy, line_orientation, material_preview,
                              outline, point_z,
                              preview_pixel_centers, rasterize, rasterize_window, resolve_geometry, section_mask,
                              vertical_section, z_extent)
from ..model.interface_sampling import interface_objects, interface_selection, object_outline
from ..model.schema import FREE_MESH_KINDS, fill_defaults, solver_backend
from ..model.spec import outer_faces
from ..model.study import sweep_values_of
from ..viz.common import (color_limits, containing_cells, edges_from_centers, length_unit, line_profile,
                          tetrahedron_mesh_cut)
from ..viz.visualization import DARK_CMAPS
from .document import COMPUTED_RESULTS_NODE, MESHES_NODE, harmonic_channel_polarizations, mesh_of_node
from .forms import display_label, face_cell_counts_text, format_quantity, rich_tooltip, wrapped_label
from .helpers import (coupling_mesh_text, field_mesh_name, memory_estimate_text, mesh_badge_html, mesh_color,
                      mesh_summary_text, several_meshes)
from .mesh_preview import AUTO_PREVIEW_MAX_ELEMENTS, AUTO_PREVIEW_MAX_TETRAHEDRA
from .plot_controls import PlotPanel, log_tick_strings

TOOLS = ('select', 'rectangle', 'circle', 'ellipse', 'polygon', 'point', 'line', 'profile')
_DRAG_TOOLS = frozenset({'rectangle', 'circle', 'ellipse', 'line'})
# tree node (first path element) -> the view selecting it shows; ('mesh',) matches every ('mesh', name) node
_NODE_VIEWS = {MESHES_NODE: 'mesh', ('mesh',): 'mesh', COMPUTED_RESULTS_NODE: 'field'}
# Params that place an editable object; an expression in any of them locks the ROI
_POSITION_PARAMS = {'rectangle': ('x0', 'y0', 'x1', 'y1'), 'circle': ('cx', 'cy', 'r'),
                    'ellipse': ('cx', 'cy', 'a', 'b', 'angle'), 'polygon': ('points',), 'point': ('x', 'y'),
                    'line': ('x0', 'y0', 'x1', 'y1')}
DRAGGABLE_KINDS = frozenset(_POSITION_PARAMS)   # geometry kinds the canvas can move (settings panel drag hint)

_MATERIAL_COLORS = ('#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462', '#b3de69', '#fccde5',
                    '#bc80bd', '#ccebc5', '#ffed6f', '#d9d9d9')
_CATEGORY_COLORS = {'source': '#d62728', 'probe': '#2ca02c', 'monitor': '#ff7f0e',
                    'boundary_condition': '#1f77b4', 'domain_condition': '#9467bd'}
_CATEGORY_SYMBOLS = {'source': 'star', 'probe': 'o'}
_HIGHLIGHT_COLOR = '#ffa000'
_HOVER_COLOR = '#00a8d8'                # hovered settings selection entry / model tree node (dashed)
_ROI_COLOR = '#e6550d'
_DRAW_COLOR = '#0050c8'
_PROFILE_COLOR = '#c000c0'
_FACE_LABEL_COLOR = '#204a87'
_CUT_LINE_COLOR = '#7a3db8'
_LAYER_SHADE_RGBA = (0, 80, 200, 30)
_SWEEP_ERRORS = (ValueError, KeyError, TypeError, ArithmeticError)   # a sweep value spec / estimate that fails
_OVER_BUDGET_STYLE = 'color:#b02a37;'

# 3D views: Top = the xy layers, Front / Side = vertical cuts at y = const / x = const (scene axes (u, v))
_VIEWS = (('xy', 'Top (xy)'), ('xz', 'Front (xz at y)'), ('yz', 'Side (yz at x)'))
_VIEW_AXES = {'xy': ('x', 'y'), 'xz': ('x', 'z'), 'yz': ('y', 'z')}
_VIEW_CUT_AXIS = {'xz': 'y', 'yz': 'x'}            # side view -> axis its cut plane is normal to
_VIEW_NAMES = {'xz': 'Front', 'yz': 'Side'}
_VIEW_FACES = {'xy': FACES, 'xz': ('x_lo', 'x_hi', 'z_lo', 'z_hi'),   # outer faces on the (lo u, hi u, lo v, hi v)
               'yz': ('y_lo', 'y_hi', 'z_lo', 'z_hi')}                 # edges of the view
_SIDE_VIEW_EDIT_HINT = 'Front / Side views select and highlight only: draw and move shapes in the Top (xy) view.'

_HIT_TOLERANCE_PX = 6          # click distance to a face / outline (boundary selections, outline picking)
_NEAREST_TOLERANCE_PX = 12     # click distance to a point / line object
_MIN_DRAG_PX = 3               # shorter rubber-band drags are ignored
_SNAP_TAN = math.tan(math.radians(10.0))   # lines within 10 deg of an axis snap to it (Shift = free)
_SCENE_DECIMALS = 3            # edited / drawn coordinates round to 1e-3 scene units, finer when a pixel is smaller
_MATERIAL_PREVIEW_CELLS = 400
_INTERFACE_OUTLINE_SAMPLES = 2000   # interface selection outlines: sample spacing = plane size / this
_INTERFACE_BAND_PX = 6              # width of the shaded read side of an interface selection [preview pixels]
_GRID_LINES_MAX = 300          # mesh grid lines drawn per axis
_DISPLAY_PIXELS_MAX = 4_000_000
_DETAIL_DELAY_MS = 60          # zoom / pan settle time before the field detail image is resampled
_INFO_BOX_MIN_LINES = 2        # text lines always kept below the plot (banner, mesh badge, legend; more scroll)
_MIN_FRAME_FRACTION = 0.5      # fitted view frame: each side at least this fraction of the largest it can be
_COLORBAR_AXIS_WIDTH = 64     # px of the colour bar's tick labels, fixed so new levels do not resize the plot
_DETAIL_MARGIN = 0.25          # detail image: the visible range padded by this fraction of the view per side
_NATIVE_PROFILE_SAMPLES = 20_000   # drawn line profiles (exact field cells): at most this many samples


# ---- module helpers ----
def _field_view(field_data):
    """Canvas view of a FieldData's cut: 'xz' for a y cut, 'yz' for an x cut, else 'xy' (z cuts, 2D, no field)."""
    plane = getattr(field_data, 'plane', None)
    axis = None if plane is None else plane[0]
    return next((view for view, cut_axis in _VIEW_CUT_AXIS.items() if cut_axis == axis), 'xy')


def _polyline_distance(x, y, polyline):
    """Shortest distance from (x, y) to an (N, 2) polyline (a single vertex is a point)."""
    if len(polyline) == 1:
        return math.hypot(x - polyline[0, 0], y - polyline[0, 1])
    starts, ends = polyline[:-1], polyline[1:]
    segment = ends - starts
    offset = np.array([x, y]) - starts
    length_squared = np.maximum(np.einsum('ij,ij->i', segment, segment), 1e-300)
    fraction = np.clip(np.einsum('ij,ij->i', offset, segment) / length_squared, 0.0, 1.0)
    closest = starts + fraction[:, None] * segment
    return float(np.min(np.hypot(closest[:, 0] - x, closest[:, 1] - y)))


def _face_segments(size_u, size_v, faces=FACES):
    """{outer face: (2, 2) end points [m]} of a size_u x size_v view; faces = names of its (lo u, hi u, lo v, hi v)
    edges (_VIEW_FACES)."""
    lo_u, hi_u, lo_v, hi_v = faces
    return {lo_u: np.array([[0.0, 0.0], [0.0, size_v]]), hi_u: np.array([[size_u, 0.0], [size_u, size_v]]),
            lo_v: np.array([[0.0, 0.0], [size_u, 0.0]]), hi_v: np.array([[0.0, size_v], [size_u, size_v]])}


def _split_markers(polylines):
    """(polylines of >= 2 vertices, [vertex (2,)] of the one-vertex ones: points, line crossings of a side view)."""
    curves = [polyline for polyline in polylines if len(polyline) > 1]
    return curves, [polyline[0] for polyline in polylines if len(polyline) == 1]


def _side_band(mask, side, width=_INTERFACE_BAND_PX):
    """Pixels on `side` ('inside' | 'outside') of a bool preview mask within `width` 4-neighbour steps of its edge."""
    region = mask if side == 'inside' else ~mask
    near = ~region
    for _ in range(width):
        grown = near.copy()
        grown[1:, :] |= near[:-1, :]
        grown[:-1, :] |= near[1:, :]
        grown[:, 1:] |= near[:, :-1]
        grown[:, :-1] |= near[:, 1:]
        near = grown
    return region & near


def _decimate_uniform(values):
    """values nearest-sampled down to at most _DISPLAY_PIXELS_MAX cells (same extent); unchanged when smaller."""
    n_x, n_y = values.shape
    shrink = math.sqrt(min(1.0, _DISPLAY_PIXELS_MAX / max(1, n_x * n_y)))
    if shrink >= 1.0:
        return values
    n_x_display, n_y_display = max(1, int(n_x * shrink)), max(1, int(n_y * shrink))
    ix = ((np.arange(n_x_display) + 0.5) * (n_x / n_x_display)).astype(np.int64)
    iy = ((np.arange(n_y_display) + 0.5) * (n_y / n_y_display)).astype(np.int64)
    return values[np.ix_(ix, iy)]


def _centred_span(centers, span):
    """(lo, hi) of uniform cell centres: first - half spacing, last + half spacing; span with fewer than 2 centres."""
    if centers.size < 2:
        return tuple(span)
    half = 0.5 * (centers[-1] - centers[0]) / (centers.size - 1)
    return float(centers[0] - half), float(centers[-1] + half)


@dataclass
class _NativeCells:
    """Native cells of a field on non-uniform centres: values (nx, ny), centres and the cell edges rebuilt from them
    (viz.common.edges_from_centers, outer edges = the extent) [m], smallest centre spacing per axis (the extent's size
    for a single cell)."""
    values: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    finest_x: float
    finest_y: float

    def cell_indices(self, x, y):
        """(ix, iy) of the cells containing x, y [m] (arrays or scalars): viz.common.containing_cells, the rule of
        line_profile and probes (an edge reads the upper cell), clamped to the cells."""
        return containing_cells(x, self.x_edges), containing_cells(y, self.y_edges)

    def resample(self, x0, x1, y0, y1, n_x, n_y):
        """Containing-cell values on n_x x n_y uniform pixels over (x0, x1, y0, y1) [m]."""
        x_display = x0 + (np.arange(n_x) + 0.5) * ((x1 - x0) / n_x)
        y_display = y0 + (np.arange(n_y) + 0.5) * ((y1 - y0) / n_y)
        return self.values[np.ix_(*self.cell_indices(x_display, y_display))]


def _finest_cell(centers, lo, hi):
    """Smallest centre spacing among the cells overlapping [lo, hi] [m] (inf for a single cell)."""
    first = max(0, int(np.searchsorted(centers, lo)) - 1)
    last = min(centers.size - 1, int(np.searchsorted(centers, hi)))
    spacing = np.diff(centers[first:last + 1])
    return float(spacing.min()) if spacing.size else math.inf


def _display_grid(field_data):
    """(values on a uniform display grid of at most _DISPLAY_PIXELS_MAX cells, extent [m], _NativeCells or None):
    float64 stays float64 (DFT maps ~1e-35 would underflow float32), others float32; non-uniform cell centres are
    resampled nearest (and kept as _NativeCells for the zoomed detail image, hover and profiles), large uniform grids
    nearest-decimated; uniform centres span centre +- half a spacing (downsampled DFT maps sit on their sample
    centres, not spread over the box edges)."""
    values = np.asarray(field_data.values)
    values = values.astype(np.float64 if values.dtype == np.float64 else np.float32, copy=False)
    if values.ndim != 2:
        raise ValueError(f'field values must be 2D (nx, ny), got shape {values.shape}')
    extent = tuple(float(value) for value in field_data.extent)
    x_centers, y_centers = field_data.x_centers, field_data.y_centers
    if x_centers is None or y_centers is None:
        return _decimate_uniform(values), extent, None
    x_centers = np.asarray(x_centers, dtype=np.float64)
    y_centers = np.asarray(y_centers, dtype=np.float64)
    if (x_centers.size, y_centers.size) != values.shape:
        raise ValueError(f'field centres ({x_centers.size}, {y_centers.size}) do not match values {values.shape}')
    x_uniform = x_centers.size < 3 or has_uniform_spacing(np.diff(x_centers))
    y_uniform = y_centers.size < 3 or has_uniform_spacing(np.diff(y_centers))
    if x_uniform and y_uniform:
        return (_decimate_uniform(values), (*_centred_span(x_centers, extent[:2]), *_centred_span(y_centers, extent[2:])),
                None)
    x0, x1, y0, y1 = extent
    native = _NativeCells(values, x_centers, y_centers, edges_from_centers(x_centers, x0, x1),
                          edges_from_centers(y_centers, y0, y1),
                          float(np.diff(x_centers).min()) if x_centers.size > 1 else x1 - x0,
                          float(np.diff(y_centers).min()) if y_centers.size > 1 else y1 - y0)
    n_x = max(values.shape[0], math.ceil((x1 - x0) / native.finest_x))
    n_y = max(values.shape[1], math.ceil((y1 - y0) / native.finest_y))
    shrink = math.sqrt(min(1.0, _DISPLAY_PIXELS_MAX / (n_x * n_y)))
    n_x, n_y = max(1, int(n_x * shrink)), max(1, int(n_y * shrink))
    return native.resample(x0, x1, y0, y1, n_x, n_y), extent, native


def _uses_expressions(params, kind):
    """True when a position param of the object is an expression (text) and the ROI must stay locked."""
    for name in _POSITION_PARAMS[kind]:
        value = params.get(name)
        if isinstance(value, str):
            return True
        if name == 'points' and isinstance(value, (list, tuple)):
            if any(isinstance(coordinate, str) for point in value if isinstance(point, (list, tuple))
                   for coordinate in point):
                return True
    return False


def _enabled_features(spec, physics, physics_type):
    """Yield (feature, feature_type, entities) of the physics' enabled features; unresolvable ones are skipped."""
    for feature in physics.features:
        if not feature.enabled:
            continue
        try:
            feature_type = physics_type.feature_type(feature.type)
            entities = expand_entities(spec, feature.selection)
        except (KeyError, ValueError, TypeError):
            continue
        yield feature, feature_type, entities


def _snap_segment(anchor, moving):
    """`moving` snapped so anchor -> moving is horizontal / vertical when within 10 degrees of it."""
    delta_x, delta_y = moving[0] - anchor[0], moving[1] - anchor[1]
    if abs(delta_y) <= _SNAP_TAN * abs(delta_x):
        return moving[0], anchor[1]
    if abs(delta_x) <= _SNAP_TAN * abs(delta_y):
        return anchor[0], moving[1]
    return moving


def _ellipse_polyline(cx, cy, semi_x, semi_y, n_segments=96):
    theta = np.linspace(0.0, 2.0 * np.pi, n_segments + 1)
    return np.column_stack([cx + semi_x * np.cos(theta), cy + semi_y * np.sin(theta)])


def _layer_bounds(shapes, size_z):
    """[(z_lo, z_hi)] layer slabs of a 3D model between its objects' z breakpoints (prism / box / cylinder ends,
    sphere poles) inside [0, size_z]."""
    breakpoints = {0.0, float(size_z)}
    for name, shape in shapes.items():
        kind = GEOMETRY_KINDS[shape.kind]
        if kind.entity_kind == 'domain' and not kind.composite:
            breakpoints.update(float(z) for z in z_extent(name, shapes) if math.isfinite(z) and 0.0 < z < size_z)
    edges = sorted(breakpoints)
    return [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:]) if hi - lo > 1e-9 * size_z]


def _absent_from_layer(name, shapes, layer, size_z, outlines):
    """True when a primitive, point or line has nothing in the layer slab (composites always show); a domain
    primitive's outline at the layer's mid-height goes into outlines (the top view draws it, not computed twice)."""
    shape = shapes[name]
    kind = GEOMETRY_KINDS[shape.kind]
    if kind.composite:
        return False
    if kind.entity_kind in ('point', 'line'):
        height = point_z(shape.values, size_z)
        return not (layer[0] <= height < layer[1] or (height == layer[1] == size_z))
    drawn = outlines[name] = outline(name, shapes, z=0.5 * (layer[0] + layer[1]))
    return not drawn


def _material_color(index):
    return _MATERIAL_COLORS[index % len(_MATERIAL_COLORS)]


def _rgba(color, alpha=255):
    qcolor = QtGui.QColor(color)
    return qcolor.red(), qcolor.green(), qcolor.blue(), alpha


# ---- evaluated model ----
@dataclass(eq=False)
class _CellFrame:
    """Full-grid cell edges / centres [m] (physical frame) and the window cell ranges of a MeshEstimate; the z
    fields (graded z included) only in 3D."""
    x_edges: np.ndarray
    y_edges: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    x_window: tuple          # (first, last) physical cell index along x
    y_window: tuple
    z_edges: np.ndarray | None = None
    z_centers: np.ndarray | None = None
    z_window: tuple | None = None

    @classmethod
    def from_estimate(cls, mesh):
        x_edges = np.asarray(mesh.x_edges, dtype=np.float64)    # same edges as Domain.cell_edges at build time
        y_edges = np.asarray(mesh.y_edges, dtype=np.float64)
        sides = mesh.npml_sides
        frame = cls(x_edges, y_edges, 0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1]),
                    (sides['x_lo'], x_edges.size - sides['x_hi'] - 2), (sides['y_lo'], y_edges.size - sides['y_hi'] - 2))
        if mesh.z_edges is not None:
            z_edges = np.asarray(mesh.z_edges, dtype=np.float64)
            frame.z_edges, frame.z_centers = z_edges, 0.5 * (z_edges[1:] + z_edges[:-1])
            frame.z_window = (sides.get('z_lo', 0), z_edges.size - sides.get('z_hi', 0) - 2)
        return frame

    def edges(self, axis):
        return getattr(self, f'{axis}_edges')

    def centers(self, axis):
        return getattr(self, f'{axis}_centers')

    def window(self, axis):
        return getattr(self, f'{axis}_window')

    def cell(self, axis, coordinate):
        return containing_cell(coordinate, self.edges(axis), *self.window(axis))

    def cell_x(self, x):
        return self.cell('x', x)

    def cell_y(self, y):
        return self.cell('y', y)

    def window_extent(self, axes=('x', 'y')):
        """(u0, u1, v0, v1) [m] of the window along the scene axes (u, v)."""
        (u_edges, (u_first, u_last)), (v_edges, (v_first, v_last)) = ((self.edges(axis), self.window(axis))
                                                                        for axis in axes)
        return u_edges[u_first], u_edges[u_last + 1], v_edges[v_first], v_edges[v_last + 1]


@dataclass(eq=False)
class _EvaluatedModel:
    """CPU evaluation of the document at one revision (SI units); size_x is None when the domain size failed."""
    revision: int
    sweep_value: tuple | None = None                 # ((parameter, value), ...) it was evaluated at (canvas sliders)
    size_x: float | None = None
    size_y: float | None = None
    scale: float = 1e-6                 # metres per scene unit
    unit: str = 'µm'
    shapes: dict = field(default_factory=dict)       # name -> ResolvedShape, spec order
    consumed: set = field(default_factory=set)
    outlines: dict = field(default_factory=dict)     # name -> list of (N, 2) polylines [m]
    problems: list = field(default_factory=list)
    mesh_name: str | None = None                     # the active mesh (ModelCanvas._active_mesh_name)
    mesh: object | None = None                       # its MeshEstimate / TriangleMeshEstimate
    cells: _CellFrame | None = None
    preview: tuple | None = None                     # material_preview(...) result
    size_z: float | None = None                      # 3D models: evaluated size_z [m]; None in 2D
    layers: list = field(default_factory=list)       # 3D: [(z_lo, z_hi)] layer slabs between object z breakpoints
    layer: tuple | None = None                       # 3D: the shown layer (one of layers), None = all (footprints)
    hidden: set = field(default_factory=set)         # 3D: objects absent from the shown layer / cut (no outline, no hits)
    view: str = 'xy'                                 # 3D: 'xy' (layers), 'xz' (cut y = cut), 'yz' (cut x = cut)
    cut: float | None = None                         # side views: cut-plane position [m] along cut_axis
    _preview_masks: dict = field(default_factory=dict)
    _preview_centres: tuple | None = None             # 1D (u, v) pixel centres of the preview
    domain_map: object = None                         # 2D polar / spherical domains: geometry.DomainMap, else None

    def plane_extent(self):
        """(u0, u1, v0, v1) [m] of the shown plane: (0, width, 0, height), a polar / spherical sector's box."""
        if self.domain_map is not None:
            return self.domain_map.extent()
        size_u, size_v = self.plane_size()
        return 0.0, size_u, 0.0, size_v

    def face_segments(self):
        """{outer face: (N, 2) polyline [m]} of the shown plane's faces (arcs / rays on a polar sector; rays joined by
        an angle_wrap are no faces: not drawn, picked or highlighted)."""
        if self.domain_map is not None:
            polylines = self.domain_map.face_polylines()
            for face in self.domain_map.wrapped_faces():
                del polylines[face]
            return polylines
        return _face_segments(*self.plane_size(), self.plane_faces)

    def inside_domain(self, u, v):
        """True when the scene point (u, v) [m] lies in the shown plane's domain."""
        if self.domain_map is not None:
            return bool(self.domain_map.contains(u, v))
        size_u, size_v = self.plane_size()
        return 0.0 <= u <= size_u and 0.0 <= v <= size_v

    @property
    def layer_z(self):
        """Mid-height [m] of the shown layer, None when no layer is selected (2D or all layers)."""
        return None if self.layer is None else 0.5 * (self.layer[0] + self.layer[1])

    @property
    def scene_axes(self):
        """Model axes ('x', 'y', 'z') along the scene's (u, v): ('x', 'y') in the top view."""
        return _VIEW_AXES[self.view]

    @property
    def cut_axis(self):
        """Side views: the axis the cut plane is normal to ('y' for xz, 'x' for yz); None in the top view."""
        return _VIEW_CUT_AXIS.get(self.view)

    @property
    def plane_faces(self):
        """Outer faces on the view's (lo u, hi u, lo v, hi v) edges."""
        return _VIEW_FACES[self.view]

    def plane_size(self):
        """(width, height) [m] of the shown plane: (size_x, size_y), (size_x, size_z) or (size_y, size_z)."""
        sizes = {'x': self.size_x, 'y': self.size_y, 'z': self.size_z}
        return tuple(sizes[axis] for axis in self.scene_axes)

    def scene_point(self, u, v):
        """(x, y, z) [m] of the scene point (u, v) [m]; z is the shown layer's mid-height (or None) in the top view."""
        if self.view == 'xy':
            return u, v, self.layer_z
        return (u, self.cut, v) if self.view == 'xz' else (self.cut, u, v)

    def plane_box(self, name):
        """(u0, v0, u1, v1) [m] of a domain object in the shown plane (side views: z clamped to the domain)."""
        x0, y0, x1, y1 = bounding_box(name, self.shapes)
        if self.view == 'xy':
            return x0, y0, x1, y1
        z_lo, z_hi = z_extent(name, self.shapes)
        z_lo, z_hi = max(z_lo, 0.0), min(z_hi, self.size_z)
        return (x0, z_lo, x1, z_hi) if self.view == 'xz' else (y0, z_lo, y1, z_hi)

    def entity_kind(self, name):
        return GEOMETRY_KINDS[self.shapes[name].kind].entity_kind

    def objects(self, entity_kind, include_consumed=True):
        """Pickable objects of an entity kind (array copies never: no spec object, their array is picked)."""
        return [name for name in self.shapes if self.entity_kind(name) == entity_kind
                and (include_consumed or name not in self.consumed) and name not in self.hidden
                and not is_array_copy(name)]

    def contains(self, name, u, v):
        """True when scene point (u, v) [m] is inside the domain object (top view: its cross-section at the shown layer,
        its xy footprint on all layers)."""
        if self.view == 'xy' and self.layer_z is None:
            return bool(rasterize(name, self.shapes, np.array([[u]]), np.array([[v]]), np)[0, 0])
        x, y, z = self.scene_point(u, v)
        _, cropped = rasterize_window(name, self.shapes, np.array([x]), np.array([y]), np, z)
        return bool(cropped.any())

    def near_boundary(self, name, u, v, tolerance):
        """True when the object's inside/outside changes within `tolerance` of (u, v) (composites have no outline)."""
        if self.view != 'xy' or self.layer_z is not None:
            inside = np.array([self.contains(name, probe_u, probe_v) for probe_u, probe_v in
                               ((u, v), (u - tolerance, v), (u + tolerance, v), (u, v - tolerance), (u, v + tolerance))])
            return bool(inside.any() and not inside.all())
        probes_x = np.array([[u, u - tolerance, u + tolerance, u, u]])
        probes_y = np.array([[v, v, v, v - tolerance, v + tolerance]])
        inside = rasterize(name, self.shapes, probes_x, probes_y, np)[0]
        return bool(inside.any() and not inside.all())

    def preview_mask(self, name):
        """Bool mask of a domain object on the material-preview grid (None without a preview)."""
        if self.preview is None:
            return None
        if name not in self._preview_masks:
            index_image, _, extent = self.preview
            if self._preview_centres is None:
                self._preview_centres = preview_pixel_centers(extent, index_image.shape)
            try:
                if self.view != 'xy':
                    self._preview_masks[name] = section_mask(name, self.shapes, (self.cut_axis, self.cut),
                                                             *self._preview_centres)
                    return self._preview_masks[name]
                window, cropped = rasterize_window(name, self.shapes, *self._preview_centres, np, self.layer_z)
                mask = np.zeros(index_image.shape, dtype=bool)
                mask[window] = cropped
                self._preview_masks[name] = mask
            except (ValueError, KeyError):
                self._preview_masks[name] = None
        return self._preview_masks[name]


class _PassiveScatter(pg.ScatterPlotItem):
    """Marker overlay that lets clicks through (ScatterPlotItem accepts clicks on its spots, so the ViewBox
    would never see a click on a glyph, a highlight or a polygon vertex)."""

    def mouseClickEvent(self, ev):
        ev.ignore()


class _CanvasViewBox(pg.ViewBox):
    """ViewBox routing left clicks and left drags of the draw tools to the canvas (pan / menu otherwise)."""

    def __init__(self, canvas):
        super().__init__()
        self._canvas = canvas

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            self._canvas._handle_click(self.mapSceneToView(ev.scenePos()), ev.double(), ev.modifiers())
        else:
            super().mouseClickEvent(ev)

    def mouseDragEvent(self, ev, axis=None):
        if axis is None and ev.button() == QtCore.Qt.MouseButton.LeftButton and self._canvas.tool in _DRAG_TOOLS:
            ev.accept()
            self._canvas._handle_drag(self.mapSceneToView(ev.buttonDownScenePos()),
                                      self.mapSceneToView(ev.scenePos()), ev.isStart(), ev.isFinish(), ev.modifiers())
        else:
            super().mouseDragEvent(ev, axis)


def _uniform_tick_strings(values, scale, spacing):
    """Colour-bar tick labels in one format: all fixed-point, or all scientific with the largest tick's exponent."""
    scaled = [value * scale for value in values]
    largest = max((abs(value) for value in scaled), default=0.0)
    step = abs(spacing * scale)
    if largest == 0.0 or step == 0.0:
        return [f'{value:g}' for value in scaled]
    if 1e-3 <= largest < 1e4:
        exponent = 0
    else:
        exponent = math.floor(math.log10(largest))
    unit = 10.0 ** exponent
    places = max(0, math.ceil(-math.log10(step / unit) - 1e-9))
    suffix = f'e{exponent}' if exponent else ''
    # a float-noise zero tick reads '0', not '-0.0e-5'
    return ['0' if abs(value) < 1e-6 * step else f'{value / unit:.{places}f}{suffix}' for value in scaled]


class _PlotArea(QtWidgets.QWidget):
    """Layout-free holder of the plot and the text box below it, placed by ModelCanvas._place_plot (so a child's
    size never sets the canvas minimum)."""
    resized = QtCore.Signal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()


def _nearest_index(values, value) -> int:
    """Index of the entry of values closest to value."""
    return min(range(len(values)), key=lambda position: abs(values[position] - value))


def _sweep_axes(spec) -> tuple:
    """((parameter, (values...)), ...) of the study's swept parameters, one canvas slider each; () without a sweep or
    when its values fail (validation reports why)."""
    if spec.study.sweep is None:
        return ()
    try:
        parameter, values = sweep_values_of(spec)
    except _SWEEP_ERRORS:
        return ()
    return ((parameter, tuple(float(value) for value in values)),)


class ModelCanvas(QtWidgets.QWidget):
    """Central view: geometry (materials, PML, labels, glyphs, editable ROI), mesh, or a FieldData with colour bar."""

    cursor_info = QtCore.Signal(str)
    tool_changed = QtCore.Signal(str)
    profile_changed = QtCore.Signal(object)
    edit_refused = QtCore.Signal(str)   # a refused document edit (also in the banner); hints stay banner-only
    # field view, 3D: View / cut edited -> (cut axis 'x' | 'y' | 'z', position [m] or None = keep / middle); the
    # Results panel's cut plane follows and the field it fetches sets the view
    field_cut_requested = QtCore.Signal(str, object)

    def __init__(self, document, registry, parent=None):
        super().__init__(parent)
        self._document = document
        self._registry = registry
        self.mode = 'geometry'                 # 'geometry' | 'mesh' | 'field'
        self._node_view = None                 # view chosen by the selected Mesh / Results node (leaving it: geometry)
        self._tool = 'select'
        self._current_field = None
        self._display = None                   # (values, extent [m]) of the current field on a uniform grid
        self._native = None                    # _NativeCells of a non-uniform current field (detail, hover, profile)
        self._field_levels = (0.0, 1.0)        # colour levels of the shown field (the detail image uses them too)
        self._field_log_floor = None           # log colour scale: smallest value shown (levels are log10), else None
        self._colormap_name = None             # None -> FieldInfo.cmap
        self._symmetric = None                 # None -> FieldInfo.signed
        self._color_range = None               # values whose colour limits replace the field's own (replay), or None
        self._colormaps = {}
        self._colorbar_cmap = None
        self._outlines_dark = False            # object outlines drawn light (over a dark field colormap)
        self._refresh_pending = False
        self._full_redraw_needed = True        # False: only the selection changed since the last _redraw
        self._edit_message = ''
        self._roi_hint = ''
        self._base_legend = []                 # legend lines of the last _redraw without the ROI hint
        self._view_key = None
        self._view_sweep_value = None          # sweep values of the last fit (_reset_view_if_needed)
        self._box_aspect = None                # plot frame height / width of the last fit (None: fill the area)
        self._overlay_items = []
        self._highlight_items = []             # selection highlight overlays (rebuilt by _redraw_selection)
        self._hover_items = []                 # hover highlight overlays (rebuilt by _redraw_hover)
        self._hover_request = ([], False)      # (entities, fill domains) of the hovered settings entry / tree node
        self._roi = None
        self._roi_target = None                # (object name, kind) of the editable ROI
        self._profile_roi = None
        self._rubber_band = None
        self._drag_cancelled = False
        self._polygon_vertices = []            # scene coordinates of the polygon being drawn
        self._polygon_preview = None           # (edge curve, vertex markers) while a polygon is drawn

        self._banner = wrapped_label(text_format=QtCore.Qt.TextFormat.PlainText)
        self._banner.setStyleSheet('background:#fff3cd; color:#664d03; border:1px solid #ffe69c; padding:4px;')
        self._banner.hide()
        self._legend = wrapped_label(text_format=QtCore.Qt.TextFormat.RichText)
        self._legend.setStyleSheet('color:#333; padding:2px;')
        # names the active mesh (mesh view; every view with several meshes)
        self._mesh_badge = wrapped_label(text_format=QtCore.Qt.TextFormat.RichText)
        self._mesh_badge.hide()
        self._show_field_mesh = False          # the field view draws its field's mesh lines (Results 'Show mesh')
        self._results_meshes = {}              # mesh refinement: {mesh name: estimate} of the results' last level
        self._results_level_text = ''          # 'mesh refinement level k' of _results_meshes ('' without)
        # sweep sliders (_shown_sweep_value: the running value, else in the field view the results' value, else the
        # slider choices; None = the document's parameters)
        self._running_sweep_value = None       # (parameter, value) of the running sweep's value (set_sweep_value)
        self._results_sweep_value = None       # (parameter, value) the shown results belong to (field view only)
        self._sweep_choice = {}                # parameter -> chosen sweep value index (missing = the model value)
        self._sweep_axes = ()                  # ((parameter, (values...)), ...) the sliders show (_sweep_axes)
        self._sweep_rows = {}                  # parameter -> (slider, value label)
        self._sweep_box = QtWidgets.QWidget()
        self._sweep_layout = QtWidgets.QGridLayout(self._sweep_box)
        self._sweep_layout.setContentsMargins(4, 2, 4, 2)
        self._sweep_layout.setColumnStretch(1, 1)
        self._sweep_mesh_button = QtWidgets.QPushButton('Build mesh')
        self._sweep_mesh_button.setToolTip(rich_tooltip(
            'Mesh the shown mesh at the chosen sweep values (the model values until a slider is moved) and show it '
            'in the mesh view. Its real element count then sets the memory estimate of this value.'))
        self._sweep_mesh_button.clicked.connect(self._build_sweep_mesh)
        self._sweep_reset_button = QtWidgets.QPushButton('Model values')
        self._sweep_reset_button.setToolTip(rich_tooltip('Show the model again with its own parameter values.'))
        self._sweep_reset_button.clicked.connect(self._reset_sweep_choice)
        self._sweep_buttons = QtWidgets.QWidget()   # kept across row rebuilds (_rebuild_sweep_rows)
        buttons_layout = QtWidgets.QHBoxLayout(self._sweep_buttons)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.addWidget(self._sweep_reset_button)
        buttons_layout.addWidget(self._sweep_mesh_button)
        self._sweep_status = wrapped_label()
        self._sweep_box.hide()
        self._mesh_store = None               # mesh_preview.ElementMeshStore: built / previewed element meshes
        self._tetrahedron_cut_cache = None    # (data id, axis, position, scene axes) -> cut segments of the last draw
        self._layer = None                     # 3D: (z_lo, z_hi) of the edited layer, None = all layers
        self._view = 'xy'                      # 3D: chosen view (_VIEWS); the field view follows the field's cut
        self._last_side_view = None            # side view whose cut line the top view draws (None: none chosen yet)
        self._cuts = {}                        # cut axis 'x' / 'y' -> chosen cut position [m] (missing: mid-domain)
        self._stretch_z = False                # side views: z scaled to fill the view (aspect unlocked)
        self._view_combo = QtWidgets.QComboBox()
        for view, label in _VIEWS:
            self._view_combo.addItem(label, view)
        self._view_combo.setToolTip(rich_tooltip(
            '3D models: Top shows one layer of the stack (or every footprint) and is where shapes are drawn and '
            'moved. Front and Side show the vertical cut through the model at the chosen y or x: select and highlight '
            'there; z faces can be clicked. Showing a result field, it chooses the cut plane of the Results panel.'))
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        self._layer_label = QtWidgets.QLabel('Layer:')
        self._layer_combo = QtWidgets.QComboBox()
        self._layer_combo.setToolTip(rich_tooltip(
            '3D models: the layer of the stack shown and edited. Objects drawn while a layer is chosen get its bottom '
            'and top height; "All layers" shows every object\'s footprint and draws full-height objects.'))
        self._layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        self._cut_label = QtWidgets.QLabel()
        self._cut_spin = QtWidgets.QDoubleSpinBox()
        self._cut_spin.setDecimals(4)
        self._cut_spin.setKeyboardTracking(False)
        self._cut_spin.setToolTip(rich_tooltip('Position of the vertical cut shown. The arrows step one mesh cell; in '
                                               'the Top view the cut is the dashed line, which can be dragged.'))
        self._cut_spin.valueChanged.connect(self._on_cut_changed)
        self._stretch_check = QtWidgets.QCheckBox('Stretch z')
        self._stretch_check.setToolTip(rich_tooltip('Scale the height to fill the view, so thin layers become '
                                                    'visible. The z axis is then not to scale with the horizontal '
                                                    'axis.'))
        self._stretch_check.toggled.connect(self._on_stretch_toggled)
        self._view_row = QtWidgets.QWidget()
        view_layout = QtWidgets.QHBoxLayout(self._view_row)
        view_layout.setContentsMargins(4, 0, 4, 0)
        view_layout.addWidget(QtWidgets.QLabel('View:'))
        view_layout.addWidget(self._view_combo)
        view_layout.addWidget(self._layer_label)
        view_layout.addWidget(self._layer_combo, 1)
        for widget in (self._cut_label, self._cut_spin, self._stretch_check):
            view_layout.addWidget(widget)
        view_layout.addStretch()
        self._view_row.hide()

        self._graphics = pg.GraphicsLayoutWidget()
        self._view_box = _CanvasViewBox(self)
        self._plot = self._graphics.addPlot(row=0, col=0, viewBox=self._view_box)
        # never locked: the fit gives equal scales where the frame allows (_place_plot); a wheel over an axis zooms it
        self._plot.setAspectLocked(False)
        title_label = self._plot.titleLabel
        fit_title = title_label.updateMin

        def fit_title_height_only():
            fit_title()
            title_label.setMinimumWidth(0)   # a long title may overflow its row, never widen the plot off screen

        title_label.updateMin = fit_title_height_only
        self._material_image = pg.ImageItem(axisOrder='col-major')
        self._field_image = pg.ImageItem(axisOrder='col-major')
        self._highlight_image = pg.ImageItem(axisOrder='col-major')
        self._hover_image = pg.ImageItem(axisOrder='col-major')
        for item, z_value in ((self._material_image, -20), (self._field_image, -10), (self._hover_image, 13),
                              (self._highlight_image, 14)):
            item.setZValue(z_value)
            item.hide()
            self._plot.addItem(item)
        # native cells of a graded field resampled to the zoomed range, over the whole-window image (no view bounds:
        # the A button still fits the whole field)
        self._detail_image = pg.ImageItem(axisOrder='col-major')
        self._detail_image.setZValue(-9)
        self._detail_image.hide()
        self._plot.addItem(self._detail_image, ignoreBounds=True)
        self._colorbar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=self._colormap('viridis'),
                                         interactive=False, width=16)
        self._colorbar.setImageItem([self._field_image, self._detail_image])
        # always in the layout (transparent without a field), so showing it never resizes or shifts the plot
        self._colorbar.axis.setWidth(_COLORBAR_AXIS_WIDTH)
        colorbar_axis = self._colorbar.axis   # log colour scale: levels are log10 values, labels 10ⁿ
        colorbar_axis.tickStrings = lambda values, scale, spacing: (
            log_tick_strings(values, scale) if colorbar_axis.logMode else _uniform_tick_strings(values, scale, spacing))
        self._graphics.addItem(self._colorbar, row=0, col=1)
        self._colorbar.setOpacity(0.0)
        self._view_box.sigResized.connect(self._align_colorbar)
        self._colorbar.vb.sigResized.connect(self._align_colorbar)
        # Axes / Labels / Exports of the figure (shown in the Results Field tab); it replaces the plot's axes
        self.plot_panel = PlotPanel(self._graphics, title='Figure', settings_key='canvas_plot_style',
                                    plot_item=self._plot, embed_plot=False, curves_tab=False, log_axes=False,
                                    colorbar=self._colorbar, scene_export=True)
        self.plot_panel.axis_controls.color_changed.connect(self._update_field_image)
        metrics = self.fontMetrics()   # fixed tick-label width: zooming never resizes the frame
        self._plot.getAxis('left').setWidth(metrics.horizontalAdvance('-0000.00') + metrics.height() + 14)
        self._detail_timer = QtCore.QTimer(self)
        self._detail_timer.setSingleShot(True)
        self._detail_timer.setInterval(_DETAIL_DELAY_MS)
        self._detail_timer.timeout.connect(self._update_field_detail)
        self._view_box.sigRangeChanged.connect(self._schedule_field_detail)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self._view_row)
        # plot frame tight to the fitted range at the top, the text box in the rest below: text never moves the plot
        self._plot_area = _PlotArea()
        layout.addWidget(self._plot_area, 1)
        self._graphics.setParent(self._plot_area)
        info_widget = QtWidgets.QWidget()
        info_layout = QtWidgets.QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(2)
        for label in (self._banner, self._mesh_badge, self._legend, self._sweep_box):
            info_layout.addWidget(label)
        info_layout.addStretch()
        self._info_box = QtWidgets.QScrollArea(self._plot_area)
        self._info_box.setWidgetResizable(True)
        self._info_box.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self._info_box.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._info_box.setWidget(info_widget)
        self._info_min_height = metrics.lineSpacing() * _INFO_BOX_MIN_LINES + 8
        self._plot_area.resized.connect(self._place_plot)
        self._view_box.sigResized.connect(lambda _view_box: self._place_plot())

        self._mouse_proxy = pg.SignalProxy(self._graphics.scene().sigMouseMoved, rateLimit=30,
                                           slot=self._on_mouse_moved)
        for keys, slot in (('Return', self._finish_polygon), ('Enter', self._finish_polygon),
                           ('Esc', self._on_escape)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)

        document.replaced.connect(self._on_document_replaced)
        document.changed.connect(self._on_document_changed)
        document.node_selected.connect(self._on_node_selected)
        document.geometry_group_changed.connect(lambda _names: self._schedule_refresh(selection_only=True))
        document.issues_invalidated.connect(self._update_sweep_status)   # budgets / built counts move the memory line
        self._model = self._evaluate()
        self._schedule_refresh()

    # ---- public API ----
    @property
    def tool(self):
        return self._tool

    def set_hover_entities(self, entities, fill_domains=False):
        """Highlight `entities` dashed beside the selection (a hovered settings selection entry; [] clears it)."""
        request = (list(entities), bool(fill_domains))
        if request != self._hover_request:
            self._hover_request = request
            self._redraw_hover()

    def set_hover_node(self, node):
        """Hover highlight of what selecting `node` would highlight (a hovered model tree node; None clears it)."""
        self.set_hover_entities(*(self._highlight_request(node) if node else ([], False)))

    @property
    def current_field(self):
        """FieldData shown in the field view (None before the first show_field)."""
        return self._current_field

    def set_tool(self, tool):
        """Activate a tool (TOOLS); 'profile' needs a field and switches to the field view, a draw tool to the geometry
        view. tool_changed fires when
        the tool changes or a request is refused (so tool actions resync)."""
        if tool not in TOOLS:
            raise ValueError(f'unknown canvas tool {tool!r}; valid: {", ".join(TOOLS)}')
        requested = tool
        self._cancel_drawing()
        if tool == 'profile' and self._current_field is None:
            self._set_edit_message('The line profile needs a field: show a result first.')
            tool = 'select'
        if tool != 'profile':
            self._remove_profile_roi()
        changed = tool != self._tool or tool != requested
        self._tool = tool
        cursor = QtCore.Qt.CursorShape.CrossCursor if tool in _DRAG_TOOLS or tool in ('polygon', 'point') \
            else QtCore.Qt.CursorShape.ArrowCursor
        self._graphics.viewport().setCursor(cursor)
        if tool == 'profile':
            if self.mode != 'field':
                self.show_field()
            self._ensure_profile_roi()
        elif tool != 'select' and self.mode != 'geometry':   # drawing happens in the geometry view
            self._set_mode('geometry')
        if changed:
            self.tool_changed.emit(tool)

    def show_geometry(self):
        self._set_mode('geometry')

    def show_mesh(self):
        self._set_mode('mesh')

    def show_field(self, field_data=None, color_range=None):
        """Show field_data (a CPU FieldData) in the field view; None re-shows the current field. color_range (e.g.
        [min, max] over every replay frame) fixes the colour limits for this field_data."""
        if field_data is not None:
            values, extent, self._native = _display_grid(field_data)
            self._display = (values, extent)
            self._current_field = field_data
            self._color_range = None if color_range is None else np.asarray(color_range)
        if self._current_field is None:
            self._set_edit_message('No field to show yet: build / compute, then pick a field in the Results panel.')
            return
        if (self.mode != 'field' or not self._field_image.isVisible() or self._outlines_dark != self._dark_field()
                or self._view_is_stale()):
            self._set_mode('field')      # full redraw: image visibility, colour bar, outline colour, cut view
        else:
            self._update_field_image()
        if self._profile_roi is not None:
            self._emit_profile()

    def set_colormap(self, name):
        """Colormap override (matplotlib name) for fields; None or '' restores each field's own colormap."""
        if name:
            self._colormap(name)
        self._colormap_name = name or None
        if self.mode == 'field':
            self._redraw()

    def set_symmetric(self, flag):
        """Symmetric colour limits about 0 (bool); None restores each field's FieldInfo.signed."""
        self._symmetric = None if flag is None else bool(flag)
        if self.mode == 'field':
            self._update_field_image()

    def set_show_field_mesh(self, flag):
        """Draw the cell lines of the shown field's own mesh over it (field view; triangles: the built mesh)."""
        self._show_field_mesh = bool(flag)
        if self.mode == 'field':
            self._redraw()

    def set_results_meshes(self, estimates, level_text=''):
        """Meshes the results were computed on where they are not the model's (mesh refinement: {mesh name: estimate}
        of the last level) and the level text naming them in the field title / badge; ({}, '') = the model's."""
        if not estimates and not self._results_meshes and level_text == self._results_level_text:
            return   # nothing to change (estimates hold arrays: never compared)
        self._results_meshes = dict(estimates)
        self._results_level_text = level_text
        if self.mode == 'field':
            self._redraw()

    def set_sweep_value(self, parameter=None, value=None):
        """Show the model at a running sweep's value (`parameter` at `value`; the sliders follow it, disabled); None =
        the run ended: back to the sliders' choice."""
        running = None if parameter is None else (parameter, float(value))
        if running != self._running_sweep_value:
            self._running_sweep_value = running
            self._update_sweep_labels()
            self._schedule_refresh()

    def set_results_sweep_value(self, parameter=None, value=None):
        """Evaluate the field view's geometry with `parameter` at the sweep value of the shown results (a finished
        sweep); a running sweep's value (set_sweep_value) wins; None = the document's parameters."""
        sweep_value = None if parameter is None else (parameter, float(value))
        if sweep_value != self._results_sweep_value:
            self._results_sweep_value = sweep_value
            if self.mode == 'field':
                self._schedule_refresh()

    def _shown_sweep_value(self):
        """((parameter, value), ...) the canvas shows: the running sweep's, else in the field view the shown results'
        sweep value, else the sliders' (_sweep_value); None = the document's parameters."""
        if self._running_sweep_value is None and self.mode == 'field' and self._results_sweep_value is not None:
            return (self._results_sweep_value,)
        return self._sweep_value

    def set_mesh_store(self, store):
        """Source of the elements drawn in the mesh view and by 'Show mesh' (mesh_preview.ElementMeshStore)."""
        self._mesh_store = store
        store.changed.connect(self._on_mesh_store_changed)

    def _on_mesh_store_changed(self, mesh_name):
        if mesh_name == self._model.mesh_name:
            self._update_sweep_status()
            if self.mode != 'geometry':
                self._schedule_refresh()

    # ---- sweep sliders (one per swept parameter, above the legend) ----
    @property
    def _sweep_value(self):
        """((parameter, value), ...) the canvas shows: the running sweep's value, else the sliders' chosen values; None
        = the document's parameters."""
        if self._running_sweep_value is not None:
            return (self._running_sweep_value,)
        pairs = tuple((parameter, values[self._sweep_choice[parameter]]) for parameter, values in self._sweep_axes
                      if parameter in self._sweep_choice)
        return pairs or None

    def _value_estimates(self):
        """document.SweepValueEstimates of the shown sweep values; None at the document's parameters or when the values
        cannot be set (a running sweep's parameter gone)."""
        values = self._shown_sweep_value()
        if values is None:
            return None
        estimates = self._document.at_sweep_values(values)
        try:
            estimates.spec
        except _SWEEP_ERRORS:
            return None
        return estimates

    def _sync_sweep_controls(self):
        """Rebuild the slider rows when the swept parameters or their values changed; drop gone choices."""
        axes = _sweep_axes(self._document.spec)
        counts = dict(axes)
        self._sweep_choice = {parameter: index for parameter, index in self._sweep_choice.items()
                              if parameter in counts and index < len(counts[parameter])}
        if axes != self._sweep_axes:
            self._sweep_axes = axes
            self._rebuild_sweep_rows()
        self._sweep_box.setVisible(bool(axes))

    def _rebuild_sweep_rows(self):
        layout = self._sweep_layout
        while layout.count():
            widget = layout.takeAt(0).widget()
            if widget is not None and widget not in (self._sweep_buttons, self._sweep_status):
                widget.blockSignals(True)
                widget.deleteLater()
        self._sweep_rows = {}
        label_width = self.fontMetrics().horizontalAdvance('running value 88 of 88: -8.88888e-88')
        for row, (parameter, values) in enumerate(self._sweep_axes):
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            slider.setRange(0, len(values) - 1)   # one tick per sweep value; the model value is no position
            slider.setPageStep(1)
            slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
            slider.setToolTip(rich_tooltip(
                f'Show the geometry, materials and mesh with {parameter} at one of its sweep values (Model values: '
                'back to the model). Only the view changes: the model, its validation and Compute keep the model '
                'value.'))
            slider.valueChanged.connect(lambda index, parameter=parameter: self._on_sweep_slider(parameter, index))
            value_label = QtWidgets.QLabel()
            value_label.setMinimumWidth(label_width)
            layout.addWidget(QtWidgets.QLabel(f'Sweep {parameter}:'), row, 0)
            layout.addWidget(slider, row, 1)
            layout.addWidget(value_label, row, 2)
            self._sweep_rows[parameter] = (slider, value_label)
        row = len(self._sweep_axes)
        layout.addWidget(self._sweep_buttons, row, 0)
        layout.addWidget(self._sweep_status, row, 1, 1, 2)
        self._update_sweep_labels()

    def _update_sweep_labels(self):
        """Value texts and handles of the sliders (at the model value: the nearest sweep value's position); a running
        sweep moves its parameter's slider and disables them all."""
        running = self._running_sweep_value
        base = self._document.parameters()
        for parameter, values in self._sweep_axes:
            slider, label = self._sweep_rows[parameter]
            slider.setEnabled(running is None)
            index = self._sweep_choice.get(parameter)
            if running is not None and running[0] == parameter:
                index = _nearest_index(values, running[1])
                text = f'running value {index + 1} of {len(values)}: {running[1]:.6g}'
            elif index is None:
                model_value = base.get(parameter)
                numeric = isinstance(model_value, (int, float))
                index = _nearest_index(values, model_value) if numeric else 0
                text = f'model value: {model_value:.6g}' if numeric else 'model value'
            else:
                text = f'value {index + 1} of {len(values)}: {values[index]:.6g}'
            slider.blockSignals(True)
            slider.setValue(index)
            slider.blockSignals(False)
            label.setText(text)
        self._sweep_reset_button.setEnabled(running is None and bool(self._sweep_choice))

    def _on_sweep_slider(self, parameter, index):
        if index == self._sweep_choice.get(parameter):
            return
        self._sweep_choice[parameter] = index
        self._update_sweep_labels()
        self._schedule_refresh()

    def _reset_sweep_choice(self):
        if self._sweep_choice:
            self._sweep_choice = {}
            self._update_sweep_labels()
            self._schedule_refresh()

    def _build_sweep_mesh(self):
        """Build mesh at the shown sweep values (the mesh store's preview) and switch to the mesh view."""
        if self._mesh_store is None or self._model.mesh_name is None:
            return
        refusal = self._mesh_store.request_preview(self._model.mesh_name, log=True, estimates=self._value_estimates())
        if refusal:
            self._set_edit_message(refusal)
        elif self.mode != 'mesh':
            self._set_mode('mesh')
        self._update_sweep_status()

    def _update_sweep_status(self):
        """Build mesh button and the size / memory line of the active mesh at the shown sweep values."""
        if not self._sweep_axes:
            return
        estimates = self._value_estimates()
        source = self._document if estimates is None else estimates
        name = self._model.mesh_name
        try:
            mesh, problem = source.mesh_estimate(name)
            memory = source.memory_estimate() if mesh is not None else None
            verdict = source.memory_verdict() if memory is not None else None
            backend = solver_backend(source.spec)
        except _SWEEP_ERRORS as exc:
            mesh, problem = None, error_message(exc)
        element = mesh is not None and mesh.kind in FREE_MESH_KINDS and self._mesh_store is not None
        self._sweep_mesh_button.setVisible(element)
        if element:
            self._sweep_mesh_button.setEnabled(self._running_sweep_value is None
                                               and not self._mesh_store.pending(name, estimates))
        where = f"Mesh '{name}'" if several_meshes(self._document.spec) else 'Mesh'
        if mesh is None:
            self._sweep_status.setText(f'{where}: {problem}')
            self._sweep_status.setStyleSheet(_OVER_BUDGET_STYLE)
            return
        if mesh.kind in FREE_MESH_KINDS:
            noun = 'tetrahedra' if mesh.kind == 'tetrahedra' else 'triangles'
            size = (f'{mesh.n_elements:,} {noun} (built)' if mesh.counts_built else
                    f'about {mesh.n_elements:,} {noun} (from the sizes; Build mesh counts them)')
        else:
            size = f'{int(mesh.n_cells):,} cells'
        over = verdict is not None and verdict.over
        memory_text = memory_estimate_text(memory, source.memory_budget, source.host_memory_budget, backend)
        self._sweep_status.setText(f'{where}: {size}. {memory_text}' + (': over the limit.' if over else '.'))
        self._sweep_status.setStyleSheet(_OVER_BUDGET_STYLE if over else '')

    # ---- document signals ----
    def _on_document_replaced(self):
        """New / opened model: drop the old model's field, view fit and shape in progress (changed('all') redraws)."""
        self._current_field = None
        self._display = None
        self._native = None
        self._detail_image.hide()
        self._color_range = None
        self._view_key = None
        self._running_sweep_value = None
        self._results_sweep_value = None
        self._sweep_choice = {}
        self._cancel_drawing()
        if self._tool == 'profile':
            self.set_tool('select')
        self._remove_profile_roi()
        self._node_view = None               # the document selects ('model',) without node_selected
        if self.mode == 'field':
            self.mode = 'geometry'
        self._view, self._last_side_view, self._cuts = 'xy', None, {}

    def _on_document_changed(self, section, origin):
        self._edit_message = ''
        if section in ('model', 'results') and self._model.revision == self._document.revision - 1:
            # title / description / result settings: shapes, preview and mesh still hold; only the selection overlays
            # (a derived value's selection) and the title follow the edit
            self._model.revision = self._document.revision
            if self.mode == 'geometry' and self._model.size_x is not None:
                self._plot.setTitle(self._geometry_title(self._model))
            self._schedule_refresh(selection_only=True)
            return
        self._schedule_refresh()

    def _on_node_selected(self, node_path):
        """The Mesh / Results node switches to the mesh / field view (Results without a field: the field shown later);
        another node then shows the geometry view unless the view was changed in between."""
        node_view = _NODE_VIEWS.get(tuple(node_path[:1]))
        if node_view is not None:
            switch = self.mode != node_view and (node_view != 'field' or self._current_field is not None)
            if switch:
                self._set_mode(node_view, deferred=True)
            self._node_view = node_view
            if switch:
                return
        elif self._node_view is not None:
            self._set_mode('geometry', deferred=True)
            return
        self._schedule_refresh(selection_only=True)

    def _schedule_refresh(self, selection_only=False):
        if not selection_only:
            self._full_redraw_needed = True
        if not self._refresh_pending:
            self._refresh_pending = True
            QtCore.QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        self._refresh_pending = False
        if self._model.revision != self._document.revision or self._view_is_stale():
            self._model = self._evaluate()
            self._full_redraw_needed = True
        if self._full_redraw_needed:
            self._redraw()
        else:
            self._redraw_selection()

    def _active_mesh_name(self):
        """Mesh the canvas shows: the shown field's in the field view; else the selected mesh node's, the selected
        physics / feature's or a selected coupling's target mesh; else the default mesh (None without meshes)."""
        spec = self._document.spec
        if self.mode == 'field' and self._current_field is not None:
            name = field_mesh_name(self._current_field, spec)
            if any(mesh_spec.name == name for mesh_spec in spec.meshes):
                return name
        return mesh_of_node(spec, self._document.current_node) or self._document.default_mesh_name()

    def _view_is_stale(self):
        """True when the model was evaluated for another mesh than the active one or another sweep value, or (3D) for
        another view than the mode shows (the field view: the view and cut of the shown field's plane, _field_view)."""
        if self._model.mesh_name != self._active_mesh_name() or self._model.sweep_value != self._shown_sweep_value():
            return True
        if self._model.size_z is None:
            return False
        if self.mode != 'field':
            return self._model.view != self._view
        view = _field_view(self._current_field)
        return self._model.view != view or (view != 'xy' and self._model.cut != float(self._current_field.plane[1]))

    def _shown_axes(self, model):
        """Scene (u, v) axis names: the shown field's axes in the field view (a 3D side cut left on screen after the
        model became 2D keeps its own), else the evaluated view's."""
        if self.mode == 'field' and self._current_field is not None:
            return tuple(self._current_field.axes)
        return model.scene_axes

    def _set_mode(self, mode, deferred=False):
        """Switch view; a switch to another view than the Mesh / Results node chose ends that node's auto switch.
        deferred: redraw via the timer."""
        if mode != self._node_view:
            self._node_view = None
        if mode != 'field':
            self._remove_profile_roi()
            if self._tool == 'profile':
                self.set_tool('select')
        if mode != 'geometry' and self._tool not in ('select', 'profile'):   # draw tools belong to the geometry view
            self.set_tool('select')
        self.mode = mode
        if deferred:
            self._schedule_refresh()
        else:
            if self._view_is_stale():
                self._model = self._evaluate()
            self._redraw()

    def _set_edit_message(self, text):
        self._edit_message = text
        self._update_banner()

    def _report_refused_edit(self, text):
        """A refused document edit: the banner, plus edit_refused (status bar and Messages tab, as the other panels)."""
        self._set_edit_message(text)
        self.edit_refused.emit(text)

    # ---- evaluation (CPU) ----
    def _evaluate(self):
        """Evaluate the document's geometry, mesh estimate and material preview at the shown sweep values
        (_shown_sweep_value); problems go to the banner."""
        document = self._document
        self._sync_sweep_controls()
        model = _EvaluatedModel(revision=document.revision, sweep_value=self._shown_sweep_value(),
                                mesh_name=self._active_mesh_name())
        estimates = self._value_estimates()
        if model.sweep_value is not None and estimates is None:
            model.problems.append('The sweep value cannot be set in this model: the model values are shown.')
        source = document if estimates is None else estimates
        spec = source.spec
        parameters = source.parameters()
        model.problems.extend(f'Parameter {name!r}: {message}'
                              for name, message in source.evaluation_errors().items())
        try:
            window_map = domain_map(spec, parameters)
        except ValueError as exc:            # ExpressionError included
            model.problems.append(f'Domain: {exc}')
            if self._display is not None:
                x0, x1, y0, y1 = self._display[1]
                model.scale, model.unit = length_unit(max(x1 - x0, y1 - y0))
            return model
        if window_map.mapped:   # the canvas draws the sector in the drawing plane: sizes of its box [m]
            model.domain_map = window_map
            x0, x1, y0, y1 = window_map.extent()
            size_x, size_y = x1 - x0, y1 - y0
        else:
            size_x, size_y = window_map.size_x, window_map.size_y
        model.size_x, model.size_y = size_x, size_y
        model.scale, model.unit = length_unit(max(size_x, size_y))
        geometry_errors = {}
        model.shapes = resolve_geometry(spec, parameters, errors=geometry_errors)
        model.problems.extend(f'Geometry {name!r}: {message}' for name, message in geometry_errors.items())
        model.consumed = consumed_objects(spec) | {name for name in model.shapes if is_array_copy(name)}
        mesh, mesh_error = source.mesh_estimate(model.mesh_name)
        if mesh_error:
            model.problems.append(f"Mesh '{model.mesh_name}': {mesh_error}" if several_meshes(spec)
                                  else f'Mesh: {mesh_error}')
        if mesh is not None:
            model.mesh = mesh
            # a polar / spherical mesh has logical (r, angle) cells, a triangle mesh none: no rectilinear cell frame
            # (snapping, readout)
            model.cells = (_CellFrame.from_estimate(mesh) if mesh.kind == 'tensor' and mesh.domain_map is None
                           else None)
        self._evaluate_layers(model, spec, parameters)
        cut = None
        if model.view == 'xy':
            for name in model.shapes:
                if name in model.hidden:
                    model.outlines[name] = []
                elif name not in model.outlines:   # _absent_from_layer drew the layer's primitives already
                    model.outlines[name] = outline(name, model.shapes, z=model.layer_z)
        else:
            self._evaluate_section(model)
            cut = (model.cut_axis, model.cut)
        try:
            model.preview = material_preview(spec, parameters, max_cells=_MATERIAL_PREVIEW_CELLS, shapes=model.shapes,
                                             z=model.layer_z if cut is None else None, cut=cut)
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            model.problems.append(f'Material preview: {error_message(exc)}')
        return model

    def _evaluate_layers(self, model, spec, parameters):
        """3D: size_z, the layer slabs, the shown layer (kept while it still exists), the view and, in the top view,
        the objects absent from the layer; then the layer selector is resynced. 2D: back to the top view."""
        try:
            model.size_z = domain_size_z(spec, parameters)
        except ValueError as exc:
            model.problems.append(f'Domain size_z: {exc}')
        if model.size_z is None:
            self._view = 'xy'
        else:
            model.layers = _layer_bounds(model.shapes, model.size_z)
            model.layer = self._layer if self._layer in model.layers else None
            model.view = _field_view(self._current_field) if self.mode == 'field' else self._view
            if model.layer is not None and model.view == 'xy':
                model.hidden = {name for name in model.shapes
                                if _absent_from_layer(name, model.shapes, model.layer, model.size_z, model.outlines)}
        self._layer = model.layer
        self._sync_layer_combo(model)

    def _cut_position(self, model, axis):
        """Cut position [m] along axis 'x' / 'y': the chosen one while inside the domain, else mid-domain."""
        size_axis = model.size_x if axis == 'x' else model.size_y
        chosen = self._cuts.get(axis)
        return chosen if chosen is not None and 0.0 <= chosen <= size_axis else 0.5 * size_axis

    def _cut_tolerance(self, model):
        """Half the width [m] of the mesh cell holding the cut (points / lines in it show in the side view), else
        half a preview pixel."""
        axis = model.cut_axis
        if model.cells is not None:
            edges = model.cells.edges(axis)
            index = model.cells.cell(axis, model.cut)
            return 0.5 * float(edges[index + 1] - edges[index])
        return 0.5 * (model.size_x if axis == 'x' else model.size_y) / _MATERIAL_PREVIEW_CELLS

    def _evaluate_section(self, model):
        """Side views: the cut position (field view: the shown side cut's), every object's section outline and the
        objects the cut misses (composites have no outline and always show)."""
        model.cut = (float(self._current_field.plane[1]) if self.mode == 'field'
                     else self._cut_position(model, model.cut_axis))
        tolerance = self._cut_tolerance(model)
        for name, shape in model.shapes.items():
            model.outlines[name] = vertical_section(name, model.shapes, (model.cut_axis, model.cut), model.size_z,
                                                    tolerance)
            if not model.outlines[name] and not GEOMETRY_KINDS[shape.kind].composite:
                model.hidden.add(name)

    def _sync_view_row(self, model):
        """3D view row: View combo and cut position (the field's cut in the field view), the Layer combo in the top
        view, the cut position and Stretch z in side views (signals blocked); hidden in 2D."""
        self._view_row.setVisible(model.size_z is not None)
        if model.size_z is None:
            return
        side = model.view != 'xy'
        self._view_combo.blockSignals(True)
        self._view_combo.setCurrentIndex([view for view, _ in _VIEWS].index(model.view))
        self._view_combo.blockSignals(False)
        for widget in (self._layer_label, self._layer_combo):
            widget.setVisible(not side)
        for widget in (self._cut_label, self._cut_spin, self._stretch_check):
            widget.setVisible(side)
        if not side:
            return
        axis = model.cut_axis
        self._cut_label.setText(f'at {axis} =')
        self._cut_spin.blockSignals(True)
        try:
            self._cut_spin.setSuffix(f' {model.unit}')
            self._cut_spin.setRange(0.0, (model.size_x if axis == 'x' else model.size_y) / model.scale)
            self._cut_spin.setSingleStep(2.0 * self._cut_tolerance(model) / model.scale)
            self._cut_spin.setValue(model.cut / model.scale)
        finally:
            self._cut_spin.blockSignals(False)

    def _on_view_changed(self, index):
        """Top / Front / Side chosen: re-evaluate outlines, hidden objects and the preview for that view (side views
        return to the select tool: they do not draw). Field view: request that view's cut from the Results panel."""
        self._view = self._view_combo.itemData(index)
        if self._view != 'xy':
            self._last_side_view = self._view
            if self._tool not in ('select', 'profile'):
                self.set_tool('select')
        if self.mode == 'field':
            cut_axis = _VIEW_CUT_AXIS.get(self._view)
            self._request_field_cut('z' if cut_axis is None else cut_axis,
                                    None if cut_axis is None else self._cut_position(self._model, cut_axis))
            return
        self._cancel_drawing()
        self._model = self._evaluate()
        self._redraw()

    def _on_cut_changed(self, value):
        """Cut position edited (scene units): re-evaluate the side view there (field view: request that cut)."""
        axis = self._model.cut_axis
        if axis is None:
            return
        self._cuts[axis] = float(value) * self._model.scale
        if self.mode == 'field':
            self._request_field_cut(axis, self._cuts[axis])
            return
        self._model = self._evaluate()
        self._redraw()

    def _request_field_cut(self, axis, position):
        """Field view: ask the Results panel for the cut (axis, position [m] or None); the row shows the shown field's
        cut until the fetched field replaces it."""
        self._sync_view_row(self._model)
        self.field_cut_requested.emit(axis, position)

    def _on_cut_line_moved(self, axis, line):
        """The top view's dashed cut line was dragged: the side view of that axis cuts there next."""
        self._cuts[axis] = float(line.value()) * self._model.scale

    def _on_stretch_toggled(self, checked):
        self._stretch_z = bool(checked)
        self._redraw()

    def _pick_layer(self, z):
        """Side views: make the layer slab holding height z [m] the Top view's layer (shaded here)."""
        model = self._model
        layer = next((bounds for bounds in model.layers if bounds[0] <= z < bounds[1]), None)
        if layer is None or layer == model.layer:
            return
        self._layer = model.layer = layer
        self._sync_layer_combo(model)
        self._redraw()

    def _sync_layer_combo(self, model):
        """Layer selector entries: 'All layers' + one per layer slab (signals blocked)."""
        scale, unit = length_unit(model.size_z) if model.size_z is not None else (1.0, 'm')
        self._layer_combo.blockSignals(True)
        try:
            self._layer_combo.clear()
            self._layer_combo.addItem('All layers (footprints, full-height drawing)', None)
            for lo, hi in model.layers:
                self._layer_combo.addItem(f'z {lo / scale:.4g} to {hi / scale:.4g} {unit}', (lo, hi))
            index = model.layers.index(model.layer) + 1 if model.layer is not None else 0
            self._layer_combo.setCurrentIndex(index)
        finally:
            self._layer_combo.blockSignals(False)

    def _on_layer_changed(self, index):
        """Show / edit another layer: re-evaluate the outlines, hidden objects and the material preview."""
        layer = self._layer_combo.itemData(index)
        self._layer = tuple(layer) if layer is not None else None
        self._cancel_drawing()
        self._model = self._evaluate()
        self._full_redraw_needed = True
        self._redraw()

    # ---- drawing ----
    def _redraw(self):
        """Rebuild every overlay for the current mode (deferred; called from _refresh / mode changes), then the
        figure style of plot_panel."""
        self._draw_view()
        self._update_sweep_status()
        self.plot_panel.apply()

    def _draw_view(self):
        model = self._model
        self._full_redraw_needed = False
        self._clear_overlay()
        self._update_banner()
        self._outlines_dark = self.mode == 'field' and self._dark_field()
        legend = []
        valid = model.size_x is not None
        self._material_image.setVisible(valid and self.mode == 'geometry' and model.preview is not None)
        field_visible = self.mode == 'field' and self._current_field is not None
        self._field_image.setVisible(field_visible)
        if not field_visible:
            self._detail_image.hide()
        self._set_colorbar_visible(field_visible)
        self._highlight_image.hide()
        self._hover_image.hide()
        self._sync_view_row(model)
        u_axis, v_axis = self._shown_axes(model)
        stretched = model.view != 'xy' and self._stretch_z
        self._plot.setLabel('bottom', f'{u_axis} ({model.unit})')
        self._plot.setLabel('left', f'{v_axis} ({model.unit}, stretched)' if stretched else f'{v_axis} ({model.unit})')
        self._update_mesh_badge(model)
        if self.mode == 'field':
            self._update_field_image()
            legend.append('Hover for values; Results panel: colormap, snapshots, line profile, show mesh.')
        self._base_legend = legend
        if not valid:
            self._remove_roi()
            if self.mode != 'field':
                self._plot.setTitle('')
            self._set_legend(legend)
            return
        if self.mode == 'geometry':
            self._plot.setTitle(self._geometry_title(model))
            legend.extend(self._draw_materials(model))
            legend.extend(self._draw_pml(model))
            self._draw_face_labels(model)
            self._draw_feature_glyphs(model)
        elif self.mode == 'mesh':
            mesh_title = f" '{html.escape(str(model.mesh_name))}'" if several_meshes(self._document.spec) else ''
            self._plot.setTitle(f'Mesh{mesh_title}{self._view_title(model)}{self._sweep_title(model)}')
            legend.extend(self._draw_mesh(model))
        elif (self.mode == 'field' and self._show_field_mesh and self._current_field is not None
              and self._shown_axes(model) == model.scene_axes):
            legend.extend(self._draw_field_mesh(model))
        if self.mode != 'field' and model.size_z is not None:
            legend.append(self._draw_view_links(model))
        if self._shown_axes(model) != model.scene_axes:   # a 3D side cut of an earlier build on a now-2D model
            legend.append('The field is a side cut of the earlier 3D model: no outlines (build / compute again).')
            self._set_legend(legend)
            self._reset_view_if_needed(model)
            return
        self._draw_domain_outline(model)
        self._draw_object_outlines(model)
        self._draw_selection(model)
        self._draw_hover(model)
        self._reset_view_if_needed(model)

    @staticmethod
    def _view_title(model):
        """' (front view, y = 1.5 µm)' title suffix of a side view, '' in the top view."""
        if model.view == 'xy':
            return ''
        cut_text = f'{model.cut_axis} = {model.cut / model.scale:.4g} {model.unit}'
        return f' ({_VIEW_NAMES[model.view].lower()} view, {cut_text})'

    def _draw_view_links(self, model):
        """3D cross-links: the top view's dashed, draggable line of the last side view's cut; a side view's shaded
        Top-view layer. Returns the legend line."""
        if model.view == 'xy':
            if self._last_side_view is None:
                return ''
            axis = _VIEW_CUT_AXIS[self._last_side_view]
            size_axis = model.size_x if axis == 'x' else model.size_y
            line = pg.InfiniteLine(pos=self._cut_position(model, axis) / model.scale, angle=90 if axis == 'x' else 0,
                                   movable=True, bounds=(0.0, size_axis / model.scale),
                                   pen=pg.mkPen(_CUT_LINE_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine),
                                   hoverPen=pg.mkPen(_CUT_LINE_COLOR, width=3))
            line.sigPositionChangeFinished.connect(lambda moved, axis=axis: self._on_cut_line_moved(axis, moved))
            self._add_overlay(line, 13)
            return (f'Dashed purple line: the cut of the {_VIEW_NAMES[self._last_side_view]} view (drag it to move '
                    'the cut).')
        if model.layer is not None:
            item = QtWidgets.QGraphicsRectItem(self._scene_rect((0.0, model.plane_size()[0], *model.layer)))
            item.setPen(pg.mkPen(None))
            item.setBrush(pg.mkBrush(*_LAYER_SHADE_RGBA))
            self._add_overlay(item, -6)
        return (f'{_VIEW_NAMES[model.view]} view: the objects cut by the plane {model.cut_axis} = '
                f'{model.cut / model.scale:.4g} {model.unit}. {_SIDE_VIEW_EDIT_HINT} Double-click a layer to make it '
                "the Top view's layer (shaded blue).")

    def _geometry_title(self, model):
        return (f'{html.escape(str(self._document.spec.title))} — geometry{self._view_title(model)}'
                f'{self._sweep_title(model)}')

    @staticmethod
    def _sweep_title(model):
        """' at w = 1.2e-06' title suffix of the shown sweep values, '' at the model values."""
        if model.sweep_value is None:
            return ''
        return ' at ' + html.escape(', '.join(f'{name} = {value:.6g}' for name, value in model.sweep_value))

    def _redraw_selection(self):
        """Redraw only the selection-dependent items (highlight, ROI, legend hint) at an unchanged revision and mode."""
        self._clear_overlay(self._highlight_items)
        self._highlight_image.hide()
        self._update_mesh_badge(self._model)   # a selected coupling names its mesh transfer
        if self._model.size_x is not None and self._shown_axes(self._model) == self._model.scene_axes:
            self._draw_selection(self._model)

    def _redraw_hover(self):
        """Redraw only the hover highlight (a hovered settings entry / tree node changed)."""
        self._clear_overlay(self._hover_items)
        self._hover_image.hide()
        if self._model.revision != self._document.revision:   # the pending full redraw draws it
            return
        if self._model.size_x is not None and self._shown_axes(self._model) == self._model.scene_axes:
            self._draw_hover(self._model)

    def _draw_selection(self, model):
        self._draw_highlight(model)
        self._sync_roi(model)
        hint = [html.escape(self._roi_hint)] if self._roi_hint else []
        self._set_legend(self._base_legend + hint)

    def _set_legend(self, parts):
        self._legend.setText('<br>'.join(part for part in parts if part))
        self._legend.setVisible(any(parts))

    def _update_mesh_badge(self, model):
        """Badge naming the active mesh in its colour: always in the mesh view, in the geometry / field views when the
        model has several meshes; a selected coupling adds its mesh transfer."""
        spec = self._document.spec
        several = several_meshes(spec)
        if model.mesh_name is None or not (self.mode == 'mesh' or several):
            self._mesh_badge.hide()
            return
        mesh_spec = next((mesh_spec for mesh_spec in spec.meshes if mesh_spec.name == model.mesh_name), None)
        users = [physics.name for physics in spec.physics_on(model.mesh_name)]
        adapted = self._results_meshes.get(model.mesh_name) if self.mode == 'field' else None
        summary = (mesh_summary_text(mesh_spec, model.mesh if adapted is None else adapted)
                   if mesh_spec is not None else '')
        detail = '; '.join(part for part in (summary, f"used by {', '.join(users)}" if users else 'unused')
                           if part)
        if self.mode == 'field':
            where = 'this mesh' if adapted is None else f'this mesh, refined ({self._results_level_text})'
            detail = f'the field lives on {where}; {detail}'
        elif self.mode == 'geometry':
            detail = f'absorbing layer and cell snapping of this mesh; {detail}'
        lines = [mesh_badge_html(spec, model.mesh_name, detail)]
        path = tuple(self._document.current_node or ())
        if self.mode != 'field' and len(path) == 2 and path[0] == 'coupling' and several:
            try:
                transfer = coupling_mesh_text(spec, self._registry, spec.coupling(path[1]))
            except KeyError:
                transfer = ''
            if transfer:
                lines.append(f'Coupling {html.escape(path[1])}: {html.escape(transfer)}')
        color = mesh_color(spec, model.mesh_name)
        self._mesh_badge.setStyleSheet(f'QLabel {{ border-left: 4px solid {color}; padding: 2px 4px; }}')
        self._mesh_badge.setText('<br>'.join(lines))
        self._mesh_badge.show()

    def _update_banner(self):
        lines = [self._edit_message] if self._edit_message else []
        problems = self._model.problems if self._model is not None else []
        lines.extend(problems[:4])
        if len(problems) > 4:
            lines.append(f'… and {len(problems) - 4} more problems (see the Validation tab)')
        self._banner.setText('\n'.join(lines))
        self._banner.setVisible(bool(lines))

    def _clear_overlay(self, items=None):
        """Remove the given overlay list (None: every overlay, highlights and hover included) from the plot."""
        lists = [self._overlay_items, self._highlight_items, self._hover_items] if items is None else [items]
        for item_list in lists:
            for item in item_list:
                self._plot.removeItem(item)
            item_list.clear()

    def _add_overlay(self, item, z_value, items=None):
        """Add item to the plot and to `items` (None: the general overlay list)."""
        item.setZValue(z_value)
        self._plot.addItem(item, ignoreBounds=True)
        (self._overlay_items if items is None else items).append(item)
        return item

    def _scene_rect(self, extent):
        """QRectF in scene units of an (x0, x1, y0, y1) extent [m]."""
        x0, x1, y0, y1 = (value / self._model.scale for value in extent)
        return QtCore.QRectF(x0, y0, x1 - x0, y1 - y0)

    def _polyline_item(self, polylines, pen):
        """One PlotCurveItem drawing (N, 2) polylines [m] (NaN-separated, scene units)."""
        inverse_scale = 1.0 / self._model.scale
        xs, ys = [], []
        for polyline in polylines:
            polyline = np.asarray(polyline, dtype=np.float64)
            xs.extend((polyline[:, 0] * inverse_scale, [np.nan]))
            ys.extend((polyline[:, 1] * inverse_scale, [np.nan]))
        return pg.PlotCurveItem(x=np.concatenate(xs), y=np.concatenate(ys), pen=pen, connect='finite')

    def _text_item(self, text, color, position, anchor, angle=0):
        item = pg.TextItem(text, color=color, anchor=anchor, angle=angle, fill=pg.mkBrush(255, 255, 255, 170))
        item.setPos(position[0] / self._model.scale, position[1] / self._model.scale)
        return item

    def _colormap(self, name):
        """Cached pyqtgraph ColorMap of a matplotlib colormap name; ValueError for unknown names."""
        if name not in self._colormaps:
            try:
                colormap = pg.colormap.get(name, source='matplotlib')
            except (ValueError, KeyError, AttributeError) as exc:
                raise ValueError(f'unknown colormap {name!r}: {exc}') from None
            if colormap is None:   # pyqtgraph returns None when matplotlib cannot be imported
                raise ValueError(f'colormap {name!r} needs matplotlib, which could not be imported')
            self._colormaps[name] = colormap
        return self._colormaps[name]

    def _set_colorbar_visible(self, visible):
        self._colorbar.setOpacity(1.0 if visible else 0.0)

    def _align_colorbar(self, *_):
        """Colour bar margins so its bar spans exactly the view box (not the title row / bottom axis beside it)."""
        view_rect = self._view_box.sceneBoundingRect()
        bar_rect = self._colorbar.vb.sceneBoundingRect()
        if view_rect.height() <= 1 or bar_rect.height() <= 1:
            return
        left, top, right, bottom = self._colorbar.layout.getContentsMargins()
        aligned_top = max(top + view_rect.top() - bar_rect.top(), 0.0)
        aligned_bottom = max(bottom + bar_rect.bottom() - view_rect.bottom(), 0.0)
        if abs(aligned_top - top) > 0.5 or abs(aligned_bottom - bottom) > 0.5:   # converges: no resize loop
            self._colorbar.layout.setContentsMargins(left, aligned_top, right, aligned_bottom)

    def _draw_materials(self, model):
        """Categorical material image; returns legend lines."""
        if model.preview is None:
            return []
        index_image, names, extent = model.preview
        lut = np.zeros((len(names) + 1, 4), dtype=np.uint8)
        lut[0] = (255, 255, 255, 0)
        for index in range(len(names)):
            lut[index + 1] = _rgba(_material_color(index))
        self._material_image.setImage(lut[index_image], autoLevels=False, levels=(0, 255),
                                      rect=self._scene_rect(extent))
        if not names:
            return ['No materials (empty cells are vacuum-like).']
        swatches = ' '.join(f'<span style="background-color:{_material_color(index)}">&nbsp;&nbsp;&nbsp;</span> '
                            f'{html.escape(name)}' for index, name in enumerate(names))
        uncovered = ' · white: no material' if bool((index_image == 0).any()) else ''
        return [f'Materials: {swatches}{uncovered}']

    def _draw_outer_rings(self, inner, outer) -> bool:
        """Grey rectangles of the outer box (u0, u1, v0, v1) [m] beyond the inner one (u strips full height, v strips
        the inner width): an absorbing ring / shell; True when one was drawn."""
        inner_u0, inner_u1, inner_v0, inner_v1 = inner
        outer_u0, outer_u1, outer_v0, outer_v1 = outer
        rings = [ring for ring, present in (((outer_u0, inner_u0, outer_v0, outer_v1), outer_u0 < inner_u0),
                                            ((inner_u1, outer_u1, outer_v0, outer_v1), outer_u1 > inner_u1),
                                            ((inner_u0, inner_u1, outer_v0, inner_v0), outer_v0 < inner_v0),
                                            ((inner_u0, inner_u1, inner_v1, outer_v1), outer_v1 > inner_v1)) if present]
        for ring in rings:
            item = QtWidgets.QGraphicsRectItem(self._scene_rect(ring))
            item.setPen(pg.mkPen(None))
            item.setBrush(pg.mkBrush(110, 110, 110, 55))
            self._add_overlay(item, -5)
        return bool(rings)

    def _draw_pml(self, model):
        """Grey PML ring of the shown plane from the mesh estimate; returns legend lines."""
        if model.mesh is not None and model.mesh.kind == 'triangles':
            return self._draw_triangle_ring(model)
        if model.mesh is not None and model.mesh.kind == 'tetrahedra':
            return self._draw_tetrahedra_shell(model)
        cells = model.cells
        if cells is None or (model.view != 'xy' and cells.z_edges is None):
            return []
        u_edges, v_edges = (cells.edges(axis) for axis in model.scene_axes)
        self._draw_outer_rings(cells.window_extent(model.scene_axes),
                               (u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]))
        sides = model.mesh.npml_sides
        if not any(sides.values()):
            return []
        return ['Absorbing layer (grey ring): ' + face_cell_counts_text(sides) + ' cells']

    def _draw_triangle_ring(self, model):
        """Grey absorbing layer ring of a triangle mesh estimate (its extent beyond the window); legend lines."""
        if not self._draw_outer_rings((0.0, model.size_x, 0.0, model.size_y), model.mesh.extent):
            return []
        return [f'Absorbing layer (grey ring): {model.mesh.sizing.pml_layers} element layers']

    @staticmethod
    def _tetrahedra_plane_extent(model):
        """(u0, u1, v0, v1) [m] of a tetrahedron mesh estimate's box (window + PML shell) in the shown plane."""
        extent = model.mesh.extent   # (x0, x1, y0, y1, z0, z1)
        u_index, v_index = ('xyz'.index(axis) for axis in model.scene_axes)
        return extent[2 * u_index], extent[2 * u_index + 1], extent[2 * v_index], extent[2 * v_index + 1]

    def _draw_tetrahedra_shell(self, model):
        """Grey PML shell of a tetrahedron mesh estimate (its box beyond the window) in the shown plane; legend
        lines."""
        size_u, size_v = model.plane_size()
        if not self._draw_outer_rings((0.0, size_u, 0.0, size_v), self._tetrahedra_plane_extent(model)):
            return []
        return ['Absorbing layer (grey shell): meshed with tetrahedra like the window']

    def _element_mesh_status(self, name, noun, appear, max_auto) -> str:
        """The mesh view's status line of an element mesh not drawn yet: meshing (pending or auto preview), failed,
        outdated or above max_auto estimated elements (noun 'triangles' / 'tetrahedra'; appear: where they show)."""
        store, estimates = self._mesh_store, self._value_estimates()
        failure = store.failure(name, estimates) if store is not None else ''
        auto_preview = store is not None and self.mode == 'mesh' and store.request_auto_preview(name, estimates)
        if store is not None and (store.pending(name, estimates) or auto_preview):
            return f'Meshing… the {noun} appear here{appear} when the mesher is done.'
        if failure:
            return f'Meshing failed: {failure}'
        if estimates is not None:
            return (f'Above {max_auto:,} estimated elements the mesh view does not mesh by itself: press Build mesh '
                    f'beside the sweep sliders to see the {noun} of this sweep value.')
        if store is not None and store.outdated(name):
            return (f'The {noun} made earlier no longer match the model: press Build mesh on the mesh form (or build '
                    'the model) to see the current ones.')
        return (f'Above {max_auto:,} estimated elements the mesh view does not mesh by itself: press Build mesh on the '
                f'mesh form (or build the model) to see the {noun}.')

    def _draw_tetrahedron_mesh(self, model):
        """Tetrahedron mesh view: the built mesh (else a preview of the same inputs; the mesh view asks for one itself
        up to AUTO_PREVIEW_MAX_TETRAHEDRA) cut by the shown plane; without either the estimate. Legend lines."""
        mesh, name, store = model.mesh, model.mesh_name, self._mesh_store
        drawing = store.drawing(name, self._value_estimates()) if store is not None else None
        if drawing is not None:
            return self._draw_tetrahedron_cut(model, drawing)
        how = ('x-y triangles stacked in z layers, each prism split into 3' if mesh.sizing.sweep_axis
               else 'cubes of the maximum size, each split into 6' if mesh.sizing.mesher == 'builtin' else 'gmsh')
        lines = [f'Tetrahedron mesh estimate: about {int(mesh.n_elements):,} elements, size '
                 f"{format_quantity(float(mesh.dx_min), 'm')} to {format_quantity(float(mesh.dx_max), 'm')} ({how})."]
        lines.append(self._element_mesh_status(name, 'tetrahedra', ' (cut by the shown plane)',
                                               AUTO_PREVIEW_MAX_TETRAHEDRA))
        return lines

    def _tetrahedron_cut_plane(self, model):
        """(normal axis, position [m]) of the plane tetrahedra are cut by: the shown field's cut in the field view,
        else the shown layer's mid-height (all layers: the domain's) or the side view's cut."""
        field = self._current_field
        if self.mode == 'field' and field is not None and field.plane is not None:
            return field.plane[0], float(field.plane[1])
        if model.view == 'xy':
            return 'z', float(model.layer_z if model.layer_z is not None else 0.5 * model.size_z)
        return model.cut_axis, float(model.cut)

    def _draw_tetrahedron_cut(self, model, drawing):
        """Outlines of the tetrahedra's cross-sections with the shown plane (PML shell ones fainter) and the dashed
        window outline; legend lines."""
        data = drawing.lines
        axis, position = self._tetrahedron_cut_plane(model)
        key = (id(data), axis, position, tuple(model.scene_axes))
        if self._tetrahedron_cut_cache is None or self._tetrahedron_cut_cache[0] != key:
            self._tetrahedron_cut_cache = (key, tetrahedron_mesh_cut(data, axis, position, model.scene_axes))
        segments, shell_segments = self._tetrahedron_cut_cache[1]
        u_index, v_index = ('xyz'.index(name) for name in model.scene_axes)
        source = self._draw_element_edges(model, drawing, ((shell_segments, 45), (segments, 90)),
                                          (data.window[u_index], data.window[3 + u_index],
                                           data.window[v_index], data.window[3 + v_index]))
        lines = [f'{data.n_elements:,} tetrahedra, {data.n_points:,} nodes ({source}), cut at {axis} = '
                 f"{format_quantity(position, 'm')}; smallest radius ratio {data.smallest_quality:.2g}"]
        if data.n_poor:
            lines.append(f'{data.n_poor:,} tetrahedra are slivers (radius ratio below 0.1 and an obtuse dihedral angle).')
        return lines

    def _face_texts(self):
        """{face: ['physics: condition', ...]} from bc_<face> settings and boundary features on outer faces (3D: z faces
        too)."""
        spec = self._document.spec
        faces = outer_faces(spec)
        texts = {face: [] for face in faces}
        for physics in spec.physics:
            try:
                physics_type = self._registry.physics_type(physics.type)
            except KeyError:
                continue
            settings = fill_defaults(physics_type.settings, physics.settings if isinstance(physics.settings, dict)
                                     else {})
            settings_by_name = {param.name: param for param in physics_type.settings}
            claimed = set()
            for face in faces:
                bc_param = settings_by_name.get(f'bc_{face}')
                if bc_param is not None:   # the settings form's drop-down text
                    value = settings.get(f'bc_{face}')
                    label = dict(bc_param.choice_labels).get(value, value) if isinstance(value, str) else value
                    texts[face].append(f'{physics.name}: {label}')
                    claimed.add(face)
            for feature, feature_type, entities in _enabled_features(spec, physics, physics_type):
                if feature_type.selection_kind != 'boundary':
                    continue
                for face in faces:
                    if face in entities:
                        texts[face].append(f'{physics.name}: {feature.name} ({feature_type.label})')
                        claimed.add(face)
            if any(feature_type.selection_kind == 'boundary' for feature_type in physics_type.features):
                for face in faces:
                    if face not in claimed:
                        texts[face].append(f'{physics.name}: default')
        return texts

    def _draw_face_labels(self, model):
        """Labels of the shown plane's four outer faces (side views: z_lo / z_hi below / above)."""
        faces = model.plane_faces
        face_segments = model.face_segments()
        # (lo u, hi u, lo v, hi v) edge -> (anchor, angle): text at the face midpoint, just inside the face (a polar
        # sector: centred on the middle of its arc / ray, unrotated)
        placements = (dict.fromkeys(faces, ((0.5, 0.5), 0)) if model.domain_map is not None else
                      dict(zip(faces, (((0.5, 0.0), 90), ((0.5, 1.0), 90), ((0.5, 1.0), 0), ((0.5, 0.0), 0)))))
        face_texts = self._face_texts()
        for face in faces:
            texts = face_texts.get(face)
            if texts and face in face_segments:   # wrapped polar rays are no faces
                anchor, angle = placements[face]
                segment = face_segments[face]
                position = segment.mean(axis=0) if len(segment) == 2 else segment[len(segment) // 2]
                self._add_overlay(self._text_item(' | '.join(texts), _FACE_LABEL_COLOR, position, anchor, angle), 12)

    def _draw_feature_glyphs(self, model):
        spec = self._document.spec
        for physics in spec.physics:
            try:
                physics_type = self._registry.physics_type(physics.type)
            except KeyError:
                continue
            for feature, feature_type, entities in _enabled_features(spec, physics, physics_type):
                color =_CATEGORY_COLORS.get(feature_type.category, '#555555')
                feature_type = self._channel_glyph_type(physics, feature, feature_type)
                for entity in entities:
                    shape = model.shapes.get(entity)
                    if shape is None or entity in model.hidden:   # 3D: only objects in the shown layer / cut
                        continue
                    if model.view == 'xy':
                        self._draw_glyph(model, feature_type, shape, color, feature.name)
                    else:
                        self._draw_section_glyph(model, feature_type, shape, color, feature.name)

    def _channel_glyph_type(self, physics, feature, feature_type):
        """em_fdfd_nonlinear channel features: the em_fdfd_ez / _hz FeatureType of the channel's polarisation (its
        acts_on_hi_face places the glyph); any other feature: feature_type."""
        params = feature.params if isinstance(feature.params, dict) else {}
        polarization = harmonic_channel_polarizations(physics).get(params.get('channel'))
        if polarization is None:
            return feature_type
        try:
            return self._registry.physics_type(f'em_fdfd_{polarization}').feature_type(feature_type.name)
        except KeyError:
            return feature_type

    def _draw_section_glyph(self, model, feature_type, shape, color, label):
        """Side views: glyph of one feature on the object's section (points and line crossings as markers, lines in
        the cut as segments; no cell snapping)."""
        curves, markers = _split_markers(model.outlines.get(shape.name, []))
        entity_kind = GEOMETRY_KINDS[shape.kind].entity_kind
        if markers:
            marker_scene = np.asarray(markers) / model.scale
            self._add_overlay(_PassiveScatter(x=marker_scene[:, 0], y=marker_scene[:, 1], size=14,
                                              symbol=_CATEGORY_SYMBOLS.get(feature_type.category, 'd'),
                                              brush=pg.mkBrush(color), pen=pg.mkPen('w', width=1)), 10)
        if curves:
            solid = entity_kind == 'line' and feature_type.category == 'monitor'
            style = QtCore.Qt.PenStyle.SolidLine if solid else QtCore.Qt.PenStyle.DashLine
            pen = pg.mkPen(color, width=3 if entity_kind == 'line' else 2, style=style)
            self._add_overlay(self._polyline_item(curves, pen), 10)
        if entity_kind == 'domain':
            u0, _, _, v1 = model.plane_box(shape.name)
            anchor_point = (u0, v1)
        elif curves or markers:
            anchor_point = tuple(curves[0][0]) if curves else tuple(markers[0])
        else:
            return
        self._add_overlay(self._text_item(label, color, anchor_point, (0.0, 1.0)), 11)

    def _draw_glyph(self, model, feature_type, shape, color, label):
        """Glyph of one feature on one object, on the cells the solver uses (half-open spans / boxes)."""
        cells = model.cells
        values = shape.values
        entity_kind = GEOMETRY_KINDS[shape.kind].entity_kind
        if entity_kind == 'point':
            x, y = values['x'], values['y']
            if cells is not None:
                x, y = cells.x_centers[cells.cell_x(x)], cells.y_centers[cells.cell_y(y)]
            item = _PassiveScatter(x=[x / model.scale], y=[y / model.scale], size=14,
                                   symbol=_CATEGORY_SYMBOLS.get(feature_type.category, 'd'),
                                   brush=pg.mkBrush(color), pen=pg.mkPen('w', width=1))
            self._add_overlay(item, 10)
            anchor_point = (x, y)
        elif entity_kind == 'line':
            is_monitor = feature_type.category == 'monitor'
            on_hi_face = is_monitor or feature_type.acts_on_hi_face
            segment = self._line_glyph(model, values, on_hi_face)
            style = QtCore.Qt.PenStyle.SolidLine if is_monitor else QtCore.Qt.PenStyle.DashLine
            self._add_overlay(self._polyline_item([segment], pg.mkPen(color, width=3, style=style)), 10)
            anchor_point = tuple(segment[0])
        else:
            if shape.kind == 'rectangle' and feature_type.category in ('source', 'monitor'):
                polylines = [self._cell_box(model, values)]
                style = QtCore.Qt.PenStyle.DashDotLine
            else:
                polylines = model.outlines.get(shape.name, [])
                style = QtCore.Qt.PenStyle.DashLine
            if polylines:
                self._add_overlay(self._polyline_item(polylines, pg.mkPen(color, width=2, style=style)), 10)
            x0, y0, x1, y1 = bounding_box(shape.name, model.shapes)
            anchor_point = (x0, y1)
        self._add_overlay(self._text_item(label, color, anchor_point, (0.0, 1.0)), 11)

    @staticmethod
    def _line_glyph(model, values, on_hi_face):
        """(2, 2) segment [m] of a line feature: its cell span through the line cell's centre, or that cell's hi
        face (monitors, hi-face sources); the raw segment without a mesh or when not axis-aligned."""
        cells = model.cells
        raw = np.array([[values['x0'], values['y0']], [values['x1'], values['y1']]], dtype=np.float64)
        orientation = line_orientation(values, model.size_x, model.size_y)
        if cells is None or orientation is None:
            return raw
        if orientation == 'horizontal':
            lo, hi = sorted((values['x0'], values['x1']))
            j = cells.cell_y(0.5 * (values['y0'] + values['y1']))
            y = cells.y_edges[j + 1] if on_hi_face else cells.y_centers[j]
            return np.array([[cells.x_edges[cells.cell_x(lo)], y], [cells.x_edges[cells.cell_x(hi) + 1], y]])
        lo, hi = sorted((values['y0'], values['y1']))
        i = cells.cell_x(0.5 * (values['x0'] + values['x1']))
        x = cells.x_edges[i + 1] if on_hi_face else cells.x_centers[i]
        return np.array([[x, cells.y_edges[cells.cell_y(lo)]], [x, cells.y_edges[cells.cell_y(hi) + 1]]])

    @staticmethod
    def _cell_box(model, values):
        """Closed outline [m] of a rectangle as the cell-edge box [cell(x0), cell(x1)+1) x [cell(y0), cell(y1)+1)."""
        x0, y0, x1, y1 = values['x0'], values['y0'], values['x1'], values['y1']
        cells = model.cells
        if cells is not None:
            x0, x1 = cells.x_edges[cells.cell_x(x0)], cells.x_edges[cells.cell_x(x1) + 1]
            y0, y1 = cells.y_edges[cells.cell_y(y0)], cells.y_edges[cells.cell_y(y1) + 1]
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64)

    def _draw_domain_outline(self, model):
        if model.domain_map is not None and model.domain_map.wrapped_faces():
            domain = list(model.face_segments().values())   # the arcs only: the joined rays are no boundary
        elif model.domain_map is not None:
            domain = [model.domain_map.outline()]
        else:
            size_u, size_v = model.plane_size()
            domain = [np.array([[0.0, 0.0], [size_u, 0.0], [size_u, size_v], [0.0, size_v], [0.0, 0.0]])]
        self._add_overlay(self._polyline_item(domain, pg.mkPen('k', width=1.5)), 4)

    def _draw_object_outlines(self, model):
        """Non-interactive outlines of every object (consumed composite inputs dashed, composites labelled); one-vertex
        outlines (points, side-view line crossings) as markers."""
        main_color = (255, 255, 255, 210) if self._outlines_dark else (40, 40, 40, 220)
        main, consumed, points = [], [], []
        for name, polylines in model.outlines.items():   # empty: hidden, or a 3D object off the shown layer / cut
            kind = model.shapes[name].kind
            curves, markers = _split_markers(polylines)
            points.extend(markers)
            (consumed if name in model.consumed else main).extend(curves)
            if GEOMETRY_KINDS[kind].composite and self.mode == 'geometry' and name not in model.hidden:
                u0, v0, u1, v1 = model.plane_box(name)
                self._add_overlay(self._text_item(name, '#404040', (0.5 * (u0 + u1), 0.5 * (v0 + v1)),
                                                  (0.5, 0.5)), 6)
        if main:
            self._add_overlay(self._polyline_item(main, pg.mkPen(main_color, width=1.5)), 5)
        if consumed:
            pen = pg.mkPen(main_color[:3] + (130,), width=1, style=QtCore.Qt.PenStyle.DashLine)
            self._add_overlay(self._polyline_item(consumed, pen), 5)
        if points:
            points = np.asarray(points) / model.scale
            self._add_overlay(_PassiveScatter(x=points[:, 0], y=points[:, 1], symbol='+', size=11,
                                              pen=pg.mkPen(main_color, width=1.5)), 5)

    def _dark_field(self):
        """True when the shown field's colormap (override or FieldInfo.cmap) is dark."""
        field_cmap = self._current_field.info.cmap if self._current_field is not None else ''
        return (self._colormap_name or field_cmap) in DARK_CMAPS

    def _highlight_request(self, node=None):
        """(entities, fill domain objects) of `node` (None: the current node, or every shift-click selected object): its
        selection, or the geometry object itself."""
        if node is None and self._document.geometry_group:
            return list(self._document.geometry_group), True
        node = self._document.current_node if node is None else node
        if not node:
            return [], False
        if node[0] == 'geometry' and len(node) == 2:
            return [node[1]], True
        try:
            kind = self._document.selection_kind(node)
            if kind is None:
                return [], False
            return list(self._document.selection_of(node)), kind == 'domain'
        except (KeyError, ValueError, TypeError):
            return [], False

    def _draw_highlight(self, model):
        """Highlight the current node's selection: faces, outlines, points and a translucent domain fill."""
        entities, fill_domains = self._highlight_request()
        if not entities:
            return
        interface = self._highlight_interface(entities)
        if interface is not None and model.view == 'xy' and model.size_z is None:
            self._draw_interface_highlight(model, interface)
            return
        self._draw_entities(model, entities, fill_domains, pg.mkPen(_HIGHLIGHT_COLOR, width=4), self._highlight_items,
                            self._highlight_image, _HIGHLIGHT_COLOR)

    def _draw_hover(self, model):
        """Hover highlight of a settings / tree entry (`set_hover_entities`): dashed, its own items and fill image."""
        entities, fill_domains = self._hover_request
        if entities:
            self._draw_entities(model, entities, fill_domains,
                                pg.mkPen(_HOVER_COLOR, width=3, style=QtCore.Qt.PenStyle.DashLine),
                                self._hover_items, self._hover_image, _HOVER_COLOR)

    def _draw_entities(self, model, entities, fill_domains, pen, items, image, fill_color):
        """Faces, outlines, points (circled) and, with fill_domains, a translucent fill of the entities (named
        selections expanded) into the overlay list `items` and the mask ImageItem `image`."""
        try:
            expanded = expand_entities(self._document.spec, entities)
        except (ValueError, TypeError):
            return
        face_segments = model.face_segments()
        polylines, points, mask = [], [], None
        for entity in expanded:
            if entity in face_segments:
                polylines.append(face_segments[entity])
                continue
            if entity == 'all':
                if model.preview is not None and model.domain_map is not None:   # the sector's pixels
                    pixel_u, pixel_v = preview_pixel_centers(model.preview[2], model.preview[0].shape)
                    mask = model.domain_map.contains(pixel_u[:, None], pixel_v[None, :])
                elif model.preview is not None:
                    mask = np.ones(model.preview[0].shape, dtype=bool)
                continue
            shape = model.shapes.get(entity)
            if shape is None or entity in model.hidden:   # 3D: absent from the shown layer, as outlines and hits
                continue
            entity_kind = GEOMETRY_KINDS[shape.kind].entity_kind
            if entity_kind == 'point' and model.view == 'xy':   # on the object's own marker (its stored x, y)
                points.append((shape.values['x'], shape.values['y']))
                continue
            curves, markers = _split_markers(model.outlines.get(entity, []))
            polylines.extend(curves)
            points.extend(markers)
            if entity_kind == 'domain' and fill_domains:
                object_mask = model.preview_mask(entity)
                if object_mask is not None:
                    mask = object_mask.copy() if mask is None else (mask | object_mask)
        if polylines:
            self._add_overlay(self._polyline_item(polylines, pen), 15, items)
        if points:
            points = np.asarray(points) / model.scale
            self._add_overlay(_PassiveScatter(x=points[:, 0], y=points[:, 1], symbol='o', size=22,
                                              brush=pg.mkBrush(None), pen=pg.mkPen(fill_color, width=3)), 15, items)
        self._show_highlight_mask(model, mask, image, fill_color)

    def _show_highlight_mask(self, model, mask, image=None, color=_HIGHLIGHT_COLOR):
        """Translucent fill of a bool mask on the material-preview grid into `image` (None: the selection highlight
        image; mask None / empty: nothing)."""
        image = self._highlight_image if image is None else image
        if mask is not None and mask.any():
            rgba = np.zeros(mask.shape + (4,), dtype=np.uint8)
            rgba[mask] = _rgba(color, 80)
            image.setImage(rgba, autoLevels=False, levels=(0, 255), rect=self._scene_rect(model.preview[2]))
            image.show()

    def _highlight_interface(self, entities):
        """The 'interface' SelectionSpec the current node highlights (the selection itself, or a derived value over
        exactly one), else None."""
        node = self._document.current_node
        spec = self._document.spec
        if node and node[0] == 'selection' and len(node) == 2:
            selection = next((item for item in spec.selections if item.name == node[1]), None)
            return selection if selection is not None and selection.kind == 'interface' else None
        return interface_selection(spec, entities)

    def _draw_interface_highlight(self, model, selection):
        """An interface selection as interface sampling reads it: its objects' outlines (composites: the result's
        boundary; window-edge parts skipped) and a translucent band on the side it reads."""
        try:
            objects = interface_objects(self._document.spec, selection)
        except (ValueError, TypeError):
            return
        u0, u1, v0, v1 = model.plane_extent()
        spacing = max(u1 - u0, v1 - v0) / _INTERFACE_OUTLINE_SAMPLES
        if model.domain_map is not None:
            def inside(points):
                return np.asarray(model.domain_map.contains(points[:, 0], points[:, 1]), dtype=bool)
        else:
            def inside(points):
                return (points[:, 0] > u0) & (points[:, 0] < u1) & (points[:, 1] > v0) & (points[:, 1] < v1)
        segments, band = [], None
        for name in objects:
            shape = model.shapes.get(name)
            if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
                continue
            try:
                points, normals, lengths = object_outline(name, model.shapes, spacing, inside)
            except (ValueError, KeyError):
                continue
            half_tangents = np.column_stack([-normals[:, 1], normals[:, 0]]) * (0.5 * lengths)[:, None]
            segments.append(np.stack([points - half_tangents, points + half_tangents], axis=1))   # (n, 2 ends, 2)
            object_mask = model.preview_mask(name)
            if object_mask is not None:
                side_band = _side_band(object_mask, selection.side)
                band = side_band if band is None else band | side_band
        if segments:
            ends = np.concatenate(segments) / model.scale
            self._add_overlay(pg.PlotCurveItem(x=ends[:, :, 0].ravel(), y=ends[:, :, 1].ravel(), connect='pairs',
                                               pen=pg.mkPen(_HIGHLIGHT_COLOR, width=4)), 15, self._highlight_items)
        if band is not None and model.domain_map is not None:   # the sector's pixels only
            pixel_u, pixel_v = preview_pixel_centers(model.preview[2], model.preview[0].shape)
            band &= model.domain_map.contains(pixel_u[:, None], pixel_v[None, :])
        self._show_highlight_mask(model, band)

    def _draw_field_mesh(self, model):
        """'Show mesh' in the field view: the worker session's triangles of the field's mesh when it has them, else
        the tensor cells of the results' last mesh refinement level (set_results_meshes), else the document's current
        mesh estimate; one legend line."""
        drawing = self._mesh_store.built_drawing(model.mesh_name) if self._mesh_store is not None else None
        if drawing is not None and drawing.kind == 'tetrahedra':
            return self._draw_tetrahedron_cut(model, drawing)[:1]
        if drawing is not None:
            return self._draw_triangle_lines(model, drawing)[:1]
        adapted = self._results_meshes.get(model.mesh_name)
        if adapted is not None and adapted.kind == 'tensor':
            cells = _CellFrame.from_estimate(adapted) if adapted.domain_map is None else None
            lines = self._draw_mesh(replace(model, mesh=adapted, cells=cells))[:1]
            return [f'{html.escape(self._results_level_text.capitalize())}: {line}' for line in lines]
        return self._draw_mesh(model)[:1]

    def _draw_mesh(self, model):
        """Grid lines of the shown plane over the full grid (PML included; side views: the graded z cells) and the
        window outline; triangle meshes: the built / previewed triangles (else the estimate); legend lines."""
        cells, mesh = model.cells, model.mesh
        if mesh is not None and mesh.kind == 'triangles':
            return self._draw_triangle_mesh(model)
        if mesh is not None and mesh.kind == 'tetrahedra':
            return self._draw_tetrahedron_mesh(model)
        if mesh is not None and model.domain_map is not None:
            return self._draw_mapped_mesh(model)
        if cells is None or (model.view != 'xy' and cells.z_edges is None):
            return ['Mesh unavailable (see the message above).']
        inverse_scale = 1.0 / model.scale
        u_axis, v_axis = model.scene_axes
        u_edges, v_edges = cells.edges(u_axis), cells.edges(v_axis)
        u_step = max(1, math.ceil(u_edges.size / _GRID_LINES_MAX))
        v_step = max(1, math.ceil(v_edges.size / _GRID_LINES_MAX))
        u_lines = np.unique(np.append(u_edges[::u_step], u_edges[-1]))
        v_lines = np.unique(np.append(v_edges[::v_step], v_edges[-1]))
        full_u = [u_edges[0], u_edges[-1]]
        full_v = [v_edges[0], v_edges[-1]]
        line_u = np.concatenate([np.repeat(u_lines, 2), np.tile(full_u, v_lines.size)]) * inverse_scale
        line_v = np.concatenate([np.tile(full_v, u_lines.size), np.repeat(v_lines, 2)]) * inverse_scale
        self._add_overlay(pg.PlotCurveItem(x=line_u, y=line_v, connect='pairs', pen=pg.mkPen((0, 0, 0, 80))), -4)
        self._add_window_outline(*cells.window_extent(model.scene_axes))

        # wording of the Mesh node's estimate (settings panel)
        dt_text = 'no time step' if mesh.dt is None else f"time step {format_quantity(float(mesh.dt), 's')}"
        z_count = f' × {mesh.nz}' if mesh.dz_cells is not None else ''
        lines = [f'{mesh.nx} × {mesh.ny}{z_count} = {int(mesh.n_cells):,} cells (absorbing layer included), '
                 f'{"uniform" if mesh.uniform else "graded"}; cell size '
                 f"{format_quantity(float(mesh.dx_min), 'm')} to {format_quantity(float(mesh.dx_max), 'm')}; "
                 f'{dt_text}']
        if mesh.dz_cells is not None:   # 3D: the top view draws the xy plan, Front / Side the z cells
            drawn = ('Front / Side views draw them' if model.view == 'xy' else
                     f'drawn: the {u_axis} and z cell edges of the cut {model.cut_axis} = '
                     f'{model.cut / model.scale:.4g} {model.unit}')
            lines.append(f'z: {mesh.nz} cells (absorbing layer included), cell size '
                         f"{format_quantity(float(mesh.dz_cells.min()), 'm')} to "
                         f"{format_quantity(float(mesh.dz_cells.max()), 'm')}; {drawn}")
        if u_step > 1 or v_step > 1:
            lines.append(f'Grid lines thinned: every {u_step}th {u_axis} edge, every {v_step}th {v_axis} edge.')
        return lines

    def _draw_triangle_mesh(self, model):
        """Triangle mesh view: the built mesh while the model still gives it, else a preview of the same inputs (the
        mesh view asks for one itself up to AUTO_PREVIEW_MAX_ELEMENTS); without either the estimate. Legend lines."""
        mesh, name, store = model.mesh, model.mesh_name, self._mesh_store
        drawing = store.drawing(name, self._value_estimates()) if store is not None else None
        if drawing is not None:
            return self._draw_triangle_lines(model, drawing)
        lines = [f'Triangle mesh estimate: about {int(mesh.n_elements):,} elements, size '
                 f"{format_quantity(float(mesh.dx_min), 'm')} to {format_quantity(float(mesh.dx_max), 'm')}."]
        lines.append(self._element_mesh_status(name, 'triangles', '', AUTO_PREVIEW_MAX_ELEMENTS))
        return lines

    def _add_window_outline(self, u0, u1, v0, v1):
        """Dashed outline of the window rectangle."""
        outline = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1], [u0, v0]])
        self._add_overlay(self._polyline_item([outline], pg.mkPen('#606060', width=1,
                                                                  style=QtCore.Qt.PenStyle.DashLine)), -3)

    def _draw_element_edges(self, model, drawing, edge_sets, window_extent):
        """Element edge pairs of a MeshDrawing (edge_sets: (pairs, alpha), drawn in order) and the dashed window outline
        (window_extent: u0, u1, v0, v1); returns the text of where the drawing comes from."""
        inverse_scale = 1.0 / model.scale
        for pairs, alpha in edge_sets:
            if len(pairs):
                self._add_overlay(pg.PlotCurveItem(x=pairs[:, 0] * inverse_scale, y=pairs[:, 1] * inverse_scale,
                                                   connect='pairs', pen=pg.mkPen((0, 0, 0, alpha))), -4)
        self._add_window_outline(*window_extent)
        if drawing.source == 'built':
            return 'built with the model'
        return f'Build mesh preview, {drawing.seconds:.2g} s' if drawing.seconds is not None else 'Build mesh preview'

    def _draw_triangle_lines(self, model, drawing):
        """Element edges of a MeshDrawing (absorbing-ring edges fainter) and the dashed window outline; legend lines."""
        mesh_lines = drawing.lines
        x0, y0, x1, y1 = mesh_lines.window
        source = self._draw_element_edges(model, drawing, ((mesh_lines.ring_segments, 45), (mesh_lines.segments, 90)),
                                          (x0, x1, y0, y1))
        quality = mesh_lines.quality
        angle = quality.get('min_angle')
        angle_text = (f'; smallest angle {float(angle):.1f}° (outside the absorbing ring)'
                      if isinstance(angle, (int, float)) and math.isfinite(angle) else '')
        lines = [f'{mesh_lines.n_elements:,} triangles, {mesh_lines.n_points:,} nodes ({source}){angle_text}']
        below = int(quality.get('below_quality_angle') or 0)
        if below > 0:
            lines.append(f'{below:,} triangles have a smallest angle below the quality limit (small input angles '
                         'of the geometry, or the mesher stopped at the minimum element size).')
        if mesh_lines.thinned:
            lines.append(f'Only region outlines and boundaries are drawn: {mesh_lines.n_edges:,} edges are too many '
                         'to draw.')
        return lines

    def _draw_mapped_mesh(self, model):
        """Polar / spherical mesh: the arcs (r edges) and rays (angle edges) of the logical grid; legend lines."""
        mesh, window_map = model.mesh, model.domain_map
        r_step = max(1, math.ceil(mesh.x_edges.size / _GRID_LINES_MAX))
        angle_step = max(1, math.ceil(mesh.y_edges.size / _GRID_LINES_MAX))
        r_lines = np.unique(np.append(mesh.x_edges[::r_step], mesh.x_edges[-1]))
        angle_lines = np.unique(np.append(mesh.y_edges[::angle_step], mesh.y_edges[-1]))
        arc_angles = np.linspace(0.0, window_map.size_y, 2 + int(64 * window_map.size_y / math.pi))
        polylines = [np.column_stack(window_map.to_physical(np.full_like(arc_angles, radius), arc_angles))
                     for radius in r_lines]
        polylines += [np.column_stack(window_map.to_physical(np.array([0.0, window_map.size_x]), np.full(2, angle)))
                      for angle in angle_lines]
        self._add_overlay(self._polyline_item(polylines, pg.mkPen((0, 0, 0, 80))), -4)
        lines = [f'{mesh.nx} (r) × {mesh.ny} (angle) = {int(mesh.n_cells):,} {window_map.kind} cells; cell size '
                 f"{format_quantity(float(mesh.dx_min), 'm')} to {format_quantity(float(mesh.dx_max), 'm')} "
                 '(radial and along the arcs)']
        if r_step > 1 or angle_step > 1:
            lines.append(f'Grid lines thinned: every {r_step}th arc, every {angle_step}th ray.')
        return lines

    def _update_field_image(self):
        field_data = self._current_field
        if field_data is None:
            return
        values, extent = self._display
        info = field_data.info
        symmetric = info.signed if self._symmetric is None else self._symmetric
        limit_values = self._native.values if self._native is not None else values   # every native cell counts
        low, high = color_limits(limit_values if self._color_range is None else self._color_range, symmetric)
        self.plot_panel.axis_controls.show_color_range(low, high)
        fixed_low, fixed_high = self.plot_panel.axis_controls.color_limits()   # the Figure box's colour row
        chosen_low = low if fixed_low is None else fixed_low
        chosen_high = high if fixed_high is None else fixed_high
        if chosen_low < chosen_high:
            low, high = chosen_low, chosen_high
        elif fixed_low is not None or fixed_high is not None:
            self.plot_panel.axis_controls.mark_color_limits_invalid()
        log_scale = self.plot_panel.axis_controls.color_log() and high > 0.0
        self._field_log_floor = None
        if log_scale:
            if low <= 0.0:   # start at the field's smallest positive value, at least a decade below the highest
                field_values = np.asarray(limit_values, dtype=np.float64)
                positive = field_values[np.isfinite(field_values) & (field_values > 0.0)]
                low = float(positive.min()) if positive.size else high * 1e-6
                low = min(low, high * 0.1)
            self._field_log_floor = low
            low, high = math.log10(low), math.log10(high)
        self._colorbar.axis.setLogMode(log_scale)
        self._field_levels = (low, high)
        self._field_image.setImage(self._color_values(values), autoLevels=False, levels=(low, high),
                                   rect=self._scene_rect(extent))
        cmap_name = self._colormap_name or info.cmap
        if cmap_name != self._colorbar_cmap:
            self._colorbar.setColorMap(self._colormap(cmap_name))
            self._colorbar_cmap = cmap_name
        self._colorbar.rounding = 10.0 ** math.floor(math.log10(high - low)) / 100.0
        self._colorbar.setLevels((low, high))
        unit = f' ({info.unit})' if info.unit and info.unit != '1' else ''
        time_text = f'  |  {display_label(field_data.time_label)}' if field_data.time_label else ''
        spec = self._document.spec
        mesh_name = field_mesh_name(field_data, spec) if several_meshes(spec) else ''
        mesh_text = f"  |  mesh '{mesh_name}'" if mesh_name else ''
        # mesh refinement: the session's state and last-level snapshots (kept earlier levels are labelled 'level k: ')
        level_text = (f'  |  {self._results_level_text}' if self._results_level_text
                      and not str(field_data.time_label or '').startswith('level ') else '')
        self._plot.setTitle(html.escape(f'{field_data.physics}: {info.label}{unit}{time_text}{mesh_text}{level_text}'))
        self._reset_view_if_needed(self._model)
        self._update_field_detail()   # at once: a stale detail of the previous field must not linger
        self.plot_panel.apply()       # the new title in the figure style

    def _color_values(self, values):
        """Image values of field values on the colour scale: as they are, or log10 with values below the log floor
        (zero, negative) at the floor."""
        if self._field_log_floor is None:
            return values
        return np.log10(np.maximum(np.asarray(values, dtype=np.float64), self._field_log_floor))

    def _schedule_field_detail(self, *args):
        """View range changed: resample the detail image once the zoom / pan settles (field view, native cells)."""
        if self.mode == 'field' and self._native is not None:
            self._detail_timer.start()
        elif self._detail_image.isVisible():
            self._detail_image.hide()

    def _update_field_detail(self):
        """Non-uniform field: its native cells resampled onto the visible range (padded by _DETAIL_MARGIN) at the
        coarser of half the finest visible cell and a screen pixel, drawn over the whole-window image; hidden where
        that image already resolves the visible cells."""
        native = self._native
        if self.mode != 'field' or native is None or self._display is None or not self._field_image.isVisible():
            self._detail_image.hide()
            return
        values, (x0, x1, y0, y1) = self._display
        scale = self._model.scale
        (u_lo, u_hi), (v_lo, v_hi) = ((lo * scale, hi * scale) for lo, hi in self._view_box.viewRange())
        lo_x, hi_x, lo_y, hi_y = max(x0, u_lo), min(x1, u_hi), max(y0, v_lo), min(y1, v_hi)
        if lo_x >= hi_x or lo_y >= hi_y:
            self._detail_image.hide()
            return
        ratio = self._graphics.devicePixelRatioF()   # physical screen pixels per logical one
        screen_x, screen_y = (size * scale / ratio for size in self._view_box.viewPixelSize())
        base_x, base_y = (x1 - x0) / values.shape[0], (y1 - y0) / values.shape[1]
        pixel_x = min(base_x, max(0.5 * _finest_cell(native.x_centers, lo_x, hi_x), screen_x))
        pixel_y = min(base_y, max(0.5 * _finest_cell(native.y_centers, lo_y, hi_y), screen_y))
        if pixel_x >= base_x and pixel_y >= base_y:   # the whole-window image resolves this range already
            self._detail_image.hide()
            return
        margin_x, margin_y = _DETAIL_MARGIN * (u_hi - u_lo), _DETAIL_MARGIN * (v_hi - v_lo)
        lo_x, hi_x = max(x0, lo_x - margin_x), min(x1, hi_x + margin_x)
        lo_y, hi_y = max(y0, lo_y - margin_y), min(y1, hi_y + margin_y)
        n_x, n_y = max(1, math.ceil((hi_x - lo_x) / pixel_x)), max(1, math.ceil((hi_y - lo_y) / pixel_y))
        shrink = math.sqrt(min(1.0, _DISPLAY_PIXELS_MAX / (n_x * n_y)))
        n_x, n_y = max(1, int(n_x * shrink)), max(1, int(n_y * shrink))
        self._detail_image.setImage(self._color_values(native.resample(lo_x, hi_x, lo_y, hi_y, n_x, n_y)),
                                    autoLevels=False,
                                    levels=self._field_levels, rect=self._scene_rect((lo_x, hi_x, lo_y, hi_y)))
        self._detail_image.show()

    # ---- view range ----
    def _reset_view_if_needed(self, model):
        """Fit the view when the view plane, domain size, scale or stretch changed, or the geometry / mesh view's PML
        cell counts / triangle ring changed since its last draw; a mode switch, a new field or another sweep value
        (sliders, running sweep) keeps the user's zoom (the A button fits)."""
        field_mode = self.mode == 'field'
        field_extent = self._display[1] if field_mode and self._display is not None else None
        plane_extent = model.plane_extent() if model.size_x is not None else None
        stretched = model.view != 'xy' and self._stretch_z
        ring_extent = None   # element meshes: the absorbing ring's / shell's outer edge
        if not field_mode and model.mesh is not None and model.mesh.kind == 'triangles':
            ring_extent = tuple(model.mesh.extent)
        elif not field_mode and model.mesh is not None and model.mesh.kind == 'tetrahedra':
            ring_extent = self._tetrahedra_plane_extent(model)
        plane_key = (model.view, plane_extent, model.scale, stretched)
        pml_cells = tuple(sorted(model.mesh.npml_sides.items())) if model.mesh is not None else None
        mode_detail = None if field_mode else (pml_cells, ring_extent)
        fitted_plane_key, fitted_detail = self._view_key or (None, None)
        if self._view_key is not None and model.sweep_value != self._view_sweep_value:
            # another sweep value (slider, running sweep) keeps the zoom like a mode switch; a new length unit rescales
            # the shown range so the same region stays in view
            self._view_sweep_value = model.sweep_value
            fitted_scale = fitted_plane_key[2]
            if fitted_scale != model.scale:
                factor = fitted_scale / model.scale
                (x0, x1), (y0, y1) = self._view_box.viewRange()
                self._plot.setRange(xRange=(x0 * factor, x1 * factor), yRange=(y0 * factor, y1 * factor), padding=0)
            self._view_key = (plane_key, fitted_detail if mode_detail is None else mode_detail)
            return
        if plane_key == fitted_plane_key:
            if mode_detail is not None:
                self._view_key = (plane_key, mode_detail)
            if fitted_detail is None or mode_detail is None or fitted_detail == mode_detail:
                return                   # first draw of this mode on the fitted plane, or nothing moved: keep the zoom
        # a fit from the field view leaves no detail: the next geometry / mesh draw records it without a fit
        key = (plane_key, mode_detail)
        boxes = []
        if plane_extent is not None:
            boxes.append(plane_extent)
        if ring_extent is not None:
            boxes.append(ring_extent)
        cells = model.cells
        if self.mode != 'field' and cells is not None and (model.view == 'xy' or cells.z_edges is not None):
            u_edges, v_edges = (cells.edges(axis) for axis in model.scene_axes)
            boxes.append((u_edges[0], u_edges[-1], v_edges[0], v_edges[-1]))
        if field_extent is not None:
            boxes.append(field_extent)
        if not boxes:
            return
        self._view_key = key
        self._view_sweep_value = model.sweep_value
        x0, x1 = min(box[0] for box in boxes) / model.scale, max(box[1] for box in boxes) / model.scale
        y0, y1 = min(box[2] for box in boxes) / model.scale, max(box[3] for box in boxes) / model.scale
        self._box_aspect = None if stretched or x1 <= x0 or y1 <= y0 else (y1 - y0) / (x1 - x0)
        self._place_plot()
        self._plot.setRange(xRange=(x0, x1), yRange=(y0, y1), padding=0.04)

    def _place_plot(self):
        """Plot frame at the top of the plot area, tight to `_box_aspect` (the whole area without one) but each view
        side at least _MIN_FRAME_FRACTION of its largest (unequal scales then); the text box takes the rest below."""
        area_width, area_height = self._plot_area.width(), self._plot_area.height()
        spacing = 2
        max_height = max(area_height - self._info_min_height - spacing, 1)
        width, height = area_width, max_height
        view_rect = self._view_box.sceneBoundingRect()
        if self._box_aspect is not None and view_rect.width() > 1 and view_rect.height() > 1:
            extra_width = self._graphics.width() - view_rect.width()     # axes, title and colour bar around the view
            extra_height = self._graphics.height() - view_rect.height()
            largest_view_width = max(area_width - extra_width, 1.0)
            largest_view_height = max(max_height - extra_height, 1.0)
            view_width = largest_view_width
            view_height = view_width * self._box_aspect
            if view_height > largest_view_height:
                view_height = largest_view_height
                view_width = view_height / self._box_aspect
            view_width = max(view_width, _MIN_FRAME_FRACTION * largest_view_width)
            view_height = max(view_height, _MIN_FRAME_FRACTION * largest_view_height)
            width = min(round(view_width + extra_width), area_width)
            height = min(round(view_height + extra_height), max_height)
        old = self._graphics.geometry()
        left = (area_width - width) // 2
        # 1 px tolerance: the measured margins round, so a re-placement after the view's resize settles
        if (abs(old.left() - left) > 1 or abs(old.width() - width) > 1 or abs(old.height() - height) > 1
                or old.top() != 0):
            self._graphics.setGeometry(left, 0, width, height)
            old = self._graphics.geometry()
        self._info_box.setGeometry(0, old.height() + spacing, area_width, max(area_height - old.height() - spacing, 0))

    # ---- editable ROI ----
    def _sync_roi(self, model):
        """Editable ROI for the current ('geometry', name) node, recreated from the spec values on every redraw (never
        during a drag: the document changes only on change-finished)."""
        self._roi_hint = ''
        target_shape = None
        node = self._document.current_node
        if self.mode != 'field' and node and node[0] == 'geometry' and len(node) == 2:
            shape = model.shapes.get(node[1])
            if shape is not None and model.view != 'xy':
                self._roi_hint = _SIDE_VIEW_EDIT_HINT
            elif shape is not None and shape.kind in _POSITION_PARAMS:
                try:
                    params = self._document.spec.geometry_object(shape.name).params
                except KeyError:
                    params = None
                if isinstance(params, dict) and _uses_expressions(params, shape.kind):
                    self._roi_hint = f"'{shape.name}' is positioned by expressions: edit it in the settings panel."
                elif shape.values.get('axis', 'z') != 'z':   # its x, y are not the drawn plane's
                    self._roi_hint = (f"'{shape.name}' is extruded along {shape.values['axis']}: edit it in the "
                                      f"settings panel.")
                elif isinstance(params, dict):
                    target_shape = shape
        self._remove_roi()
        if target_shape is not None:
            self._create_roi(target_shape)

    def _create_roi(self, shape):
        inverse_scale = 1.0 / self._model.scale
        values = {name: value * inverse_scale for name, value in shape.values.items()
                  if name not in ('points', 'angle', 'axis', 'sidewall_angle')
                  and value is not None}   # unset 3D z params are None
        pen = pg.mkPen(_ROI_COLOR, width=2)
        kind = shape.kind
        if kind == 'rectangle':
            roi = pg.RectROI((values['x0'], values['y0']), (values['x1'] - values['x0'], values['y1'] - values['y0']),
                             pen=pen)
            roi.addScaleHandle([0.0, 0.0], [1.0, 1.0])
            roi.addScaleHandle([1.0, 0.0], [0.0, 1.0])
            roi.addScaleHandle([0.0, 1.0], [1.0, 0.0])
        elif kind == 'circle':
            radius = values['r']
            roi = pg.CircleROI((values['cx'] - radius, values['cy'] - radius), (2 * radius, 2 * radius), pen=pen)
        elif kind == 'ellipse':
            roi = pg.EllipseROI(self._ellipse_origin(values, shape.values['angle']),
                                (2 * values['a'], 2 * values['b']), angle=shape.values['angle'], pen=pen)
        elif kind == 'polygon':
            roi = pg.PolyLineROI([(x * inverse_scale, y * inverse_scale) for x, y in shape.values['points']],
                                 closed=True, pen=pen)
        elif kind == 'line':
            roi = pg.LineSegmentROI([(values['x0'], values['y0']), (values['x1'], values['y1'])], pen=pen)
        else:
            roi = pg.TargetItem(pos=(values['x'], values['y']), size=16, movable=True, pen=pen)
        if kind == 'point':
            roi.sigPositionChangeFinished.connect(self._commit_roi)
        else:
            roi.sigRegionChangeFinished.connect(self._commit_roi)
        roi.setZValue(20)
        self._plot.addItem(roi, ignoreBounds=True)
        self._roi = roi
        self._roi_target = (shape.name, kind)

    @staticmethod
    def _ellipse_origin(scene_values, angle_degrees):
        """Scene position of an ellipse ROI's local origin: centre - R(angle) (a, b)."""
        angle = math.radians(angle_degrees)
        semi_a, semi_b = scene_values['a'], scene_values['b']
        return (scene_values['cx'] - (semi_a * math.cos(angle) - semi_b * math.sin(angle)),
                scene_values['cy'] - (semi_a * math.sin(angle) + semi_b * math.cos(angle)))

    def _remove_roi(self):
        if self._roi is not None:
            self._roi.blockSignals(True)       # no late change-finished commit from a removed ROI
            self._plot.removeItem(self._roi)
        self._roi = None
        self._roi_target = None

    def _commit_roi(self, *args):
        """One document.mutate('geometry', ...) with the ROI's position (see _roi_params; on change-finished only)."""
        if self._roi is None:
            return
        name, kind = self._roi_target
        try:
            params = self._roi_params(name, kind)
        except ValueError as exc:
            self._report_refused_edit(f'Edit {name}: {error_message(exc)}')
            self._schedule_refresh()
            return

        def apply(spec):
            spec.geometry_object(name).params.update(params)

        try:
            self._document.mutate('geometry', apply, label=f'Edit {name}', origin=self)
        except (ValueError, KeyError, ExpressionError) as exc:
            self._report_refused_edit(f'Edit {name}: {error_message(exc)}')
        self._schedule_refresh()      # snaps the ROI to the stored (rounded) values

    def _roi_params(self, name, kind):
        """Spec params [m] read back from the ROI: a param within half a pixel of its stored value keeps it, a moved one
        is rounded by _to_si (a drag never re-rounds what it did not move). ValueError on a degenerate shape."""
        roi = self._roi
        scale = self._model.scale
        previous = self._model.shapes.get(name)
        stored = dict(previous.values) if previous is not None else {}
        half_pixel = 0.5 * max(self._view_box.viewPixelSize())

        def si(scene_value, stored_si=None, axis=None):
            if stored_si is not None and abs(scene_value - stored_si / scale) <= half_pixel:
                return float(stored_si)
            return self._to_si(scene_value, axis)

        if kind == 'rectangle':
            position, size = roi.pos(), roi.size()
            x0, x1 = sorted((position.x(), position.x() + size.x()))
            y0, y1 = sorted((position.y(), position.y() + size.y()))
            if previous is not None:
                stored['x0'], stored['x1'] = sorted((stored['x0'], stored['x1']))
                stored['y0'], stored['y1'] = sorted((stored['y0'], stored['y1']))
            params = {'x0': si(x0, stored.get('x0'), 'x'), 'y0': si(y0, stored.get('y0'), 'y'),
                      'x1': si(x1, stored.get('x1'), 'x'), 'y1': si(y1, stored.get('y1'), 'y')}
            if params['x0'] == params['x1'] or params['y0'] == params['y1']:
                raise ValueError('rectangle has zero width or height')
            return params
        if kind in ('circle', 'ellipse'):
            size = roi.size()
            center = roi.mapToParent(QtCore.QPointF(0.5 * size.x(), 0.5 * size.y()))
            center_params = {'cx': si(center.x(), stored.get('cx')), 'cy': si(center.y(), stored.get('cy'))}
            if kind == 'circle':
                params = {**center_params, 'r': si(0.5 * abs(size.x()), stored.get('r'))}
                if not params['r'] > 0:
                    raise ValueError('circle radius is zero')
                return params
            angle = (roi.angle() + 180.0) % 360.0 - 180.0
            stored_angle = stored.get('angle')
            if stored_angle is not None and abs(angle - ((stored_angle + 180.0) % 360.0 - 180.0)) <= 1e-9:
                angle_param = float(stored_angle)
            else:
                angle_param = round(angle, _SCENE_DECIMALS) + 0.0
            params = {**center_params, 'a': si(0.5 * abs(size.x()), stored.get('a')),
                      'b': si(0.5 * abs(size.y()), stored.get('b')), 'angle': angle_param}
            if not (params['a'] > 0 and params['b'] > 0):
                raise ValueError('ellipse semi-axis is zero')
            return params
        if kind == 'polygon':
            vertices = [roi.mapToParent(handle.pos()) for handle in roi.getHandles()]
            if len(vertices) < 3:
                raise ValueError('polygon needs at least 3 vertices')
            stored_points = stored.get('points') or []
            if len(stored_points) == len(vertices):   # same vertices: per coordinate
                return {'points': [[si(vertex.x(), x, 'x'), si(vertex.y(), y, 'y')]
                                   for vertex, (x, y) in zip(vertices, stored_points)]}
            points = []   # a vertex was added / removed: keep whole stored vertices that did not move
            for vertex in vertices:
                kept = next(([float(x), float(y)] for x, y in stored_points
                             if abs(vertex.x() - x / scale) <= half_pixel and abs(vertex.y() - y / scale) <= half_pixel),
                            None)
                points.append(kept if kept is not None
                              else [self._to_si(vertex.x(), 'x'), self._to_si(vertex.y(), 'y')])
            return {'points': points}
        if kind == 'line':
            start, end = (roi.mapToParent(handle.pos()) for handle in roi.getHandles()[:2])
            start, end = (start.x(), start.y()), (end.x(), end.y())
            free = bool(QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
            if not free and previous is not None:
                old_start = (stored['x0'] / scale, stored['y0'] / scale)
                old_end = (stored['x1'] / scale, stored['y1'] / scale)
                # the endpoint that moved less is the anchor; the dragged one snaps
                if math.dist(start, old_start) <= math.dist(end, old_end):
                    end = _snap_segment(start, end)
                else:
                    start = _snap_segment(end, start)
            params = {'x0': si(start[0], stored.get('x0')), 'y0': si(start[1], stored.get('y0')),
                      'x1': si(end[0], stored.get('x1')), 'y1': si(end[1], stored.get('y1'))}
            if (params['x0'], params['y0']) == (params['x1'], params['y1']):
                raise ValueError('line has zero length')
            return params
        position = roi.pos()
        return {'x': si(position.x(), stored.get('x')), 'y': si(position.y(), stored.get('y'))}

    def _to_si(self, scene_value, axis=None):
        """Scene coordinate -> metres with a clean decimal representation, rounded to the finer of 1e-3 scene units and
        one significant digit of the screen pixel size (so zooming in allows finer shapes). With axis 'x' / 'y' / 'z'
        (a rectangle / polygon / prism edge) a value on a uniform mesh's cell centre moves up one rounding step: no
        float32 tie."""
        pixel = min(self._view_box.viewPixelSize())
        decimals = _SCENE_DECIMALS
        if math.isfinite(pixel) and pixel > 0:
            decimals = max(decimals, -math.floor(math.log10(pixel)))
        exponent = int(round(math.log10(self._model.scale)))
        rounded = round(float(scene_value), decimals)
        value = float(f'{rounded:.{decimals}f}e{exponent}') + 0.0
        if axis is not None and self._on_cell_centre(value, axis, 10.0 ** (exponent - decimals)):
            value = float(f'{rounded + 10.0 ** -decimals:.{decimals}f}e{exponent}') + 0.0
        return value

    def _on_cell_centre(self, value, axis, step):
        """True when `value` [m] lies within 1e-3 `step` of a cell centre along axis 'x' / 'y' / 'z' of a uniform
        mesh."""
        model = self._model
        if model.cells is None or not model.mesh.uniform:
            return False
        centres = model.cells.centers(axis)
        if centres is None:   # z of a 2D mesh
            return False
        index = int(np.searchsorted(centres, value))
        nearest = centres[max(0, index - 1):index + 1]
        return nearest.size > 0 and float(np.min(np.abs(nearest - value))) <= 1e-3 * step

    # ---- mouse: clicks, hit tests, draw tools ----
    def _handle_click(self, view_position, double, modifiers=QtCore.Qt.KeyboardModifier.NoModifier):
        if self._model.size_x is None:
            return
        tool = self._tool
        side_view = self._model.view != 'xy'
        if side_view and tool in ('point', 'polygon'):
            self._set_edit_message(_SIDE_VIEW_EDIT_HINT)
        elif tool == 'point':
            self._add_object('point', {'x': self._to_si(view_position.x()), 'y': self._to_si(view_position.y())})
        elif tool == 'polygon':
            if double:
                self._finish_polygon()
            else:
                self._polygon_vertices.append((view_position.x(), view_position.y()))
                self._update_polygon_preview(None)
        elif tool == 'select' and self.mode != 'field':
            # a double click must not toggle twice (side views: it picks the layer); field-view and profile clicks
            # never edit the model
            if double:
                if side_view:
                    self._pick_layer(view_position.y() * self._model.scale)
                return
            control = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
            shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
            self._select_at(view_position.x() * self._model.scale, view_position.y() * self._model.scale, control,
                            shift)

    def _select_at(self, x, y, control=False, shift=False):
        """With a selecting node current: toggle the eligible entity hit (Ctrl: only 'all'; Ctrl elsewhere selects the
        object instead, a miss only explains). Without one: select the object under (x, y) (scene point [m]: (u, v)
        of a side view); Shift adds it to / removes it from the selected objects."""
        document = self._document
        node = document.current_node
        try:
            wanted_kind = document.selection_kind(node) if node else None
        except (KeyError, ValueError):
            wanted_kind = None
        if wanted_kind is not None:
            entity = self._hit_entity(x, y, wanted_kind, self._feature_type_of(node))
            if entity is not None and (entity == 'all') == control:
                try:
                    document.toggle_selection_entity(node, entity)
                except (ValueError, KeyError, ExpressionError) as exc:
                    self._report_refused_edit(f'Toggle {entity}: {error_message(exc)}')
                return
            if not control:
                self._set_edit_message("Ctrl+click on empty space toggles 'all' (the whole domain)."
                                       if entity == 'all' else
                                       f'No eligible {wanted_kind} here; Ctrl+click selects the object.')
                return
        name = self._hit_object(x, y)
        if name is not None:
            if self._edit_message:
                self._set_edit_message('')     # drop a stale miss hint
            if shift and wanted_kind is None:
                document.toggle_geometry_group(name)
            else:
                document.select(('geometry', name))

    def _tolerance(self, pixels):
        """`pixels` screen pixels in metres at the current zoom."""
        pixel_width, pixel_height = self._view_box.viewPixelSize()
        return pixels * max(pixel_width, pixel_height) * self._model.scale

    def _nearest_outline(self, x, y, names, tolerance):
        """Object whose outline passes within tolerance of (x, y), nearest (later on ties); composites by probing."""
        model = self._model
        best_name, best_distance = None, tolerance
        for name in names:
            polylines = model.outlines.get(name, [])
            if polylines:
                distance = min(_polyline_distance(x, y, polyline) for polyline in polylines)
            elif GEOMETRY_KINDS[model.shapes[name].kind].composite and model.near_boundary(name, x, y, tolerance):
                distance = tolerance
            else:
                continue
            if distance <= best_distance:
                best_name, best_distance = name, distance
        return best_name

    def _feature_type_of(self, node):
        """FeatureType of a ('feature', physics, name) node, else None (its object kinds / face rules limit hits)."""
        if not node or node[0] != 'feature' or len(node) != 3:
            return None
        try:
            physics = self._document.spec.physics_by_name(node[1])
            return self._registry.physics_type(physics.type).feature_type(physics.feature(node[2]).type)
        except KeyError:
            return None

    def _hit_entity(self, x, y, kind, feature_type=None):
        """Entity of selection kind `kind` under (x, y) [m] that feature_type accepts (None: any), or None."""
        model = self._model
        object_kinds = feature_type.object_kinds if feature_type is not None else ()

        def eligible(names):
            return [name for name in names if not object_kinds or model.shapes[name].kind in object_kinds]

        if kind == 'boundary':
            tolerance = self._tolerance(_HIT_TOLERANCE_PX)
            if feature_type is None or feature_type.allow_outer_faces:
                distances = {face: _polyline_distance(x, y, segment)
                             for face, segment in model.face_segments().items()}
                face = min(distances, key=distances.get)
                if distances[face] <= tolerance:
                    return face
            if feature_type is not None and not feature_type.allow_interior:
                return None
            return self._nearest_outline(x, y, eligible(model.objects('domain', include_consumed=False)), tolerance)
        if kind == 'domain':
            if not model.inside_domain(x, y):
                return None
            for name in reversed(eligible(model.objects('domain', include_consumed=False))):
                if model.contains(name, x, y):
                    return name
            return None if object_kinds else 'all'
        if kind in ('point', 'line'):
            return self._nearest_outline(x, y, eligible(model.objects(kind)), self._tolerance(_NEAREST_TOLERANCE_PX))
        return None

    def _hit_object(self, x, y):
        """Geometry object under (x, y) [m]: nearest point/line, else nearest outline, else last containing object."""
        model = self._model
        name = self._nearest_outline(x, y, model.objects('point') + model.objects('line'),
                                     self._tolerance(_NEAREST_TOLERANCE_PX))
        if name is None:
            name = self._nearest_outline(x, y, model.objects('domain'), self._tolerance(_HIT_TOLERANCE_PX))
        if name is None:
            name = next((candidate for candidate in reversed(model.objects('domain'))
                         if model.contains(candidate, x, y)), None)
        return name

    def _handle_drag(self, start, current, is_start, is_finish, modifiers):
        """Rubber band of the rectangle / circle / ellipse / line tools; adds the object when the drag ends."""
        if is_start:
            self._drag_cancelled = False       # a cancel only applies to the drag it interrupted
        if self._drag_cancelled or self._model.size_x is None:
            self._drag_cancelled = self._drag_cancelled and not is_finish
            return
        if self._model.view != 'xy':   # side views do not draw (v1)
            if is_start:
                self._set_edit_message(_SIDE_VIEW_EDIT_HINT)
            return
        kind = self._tool
        free = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)
        start, current = (start.x(), start.y()), (current.x(), current.y())
        params, preview = self._drawn_shape(kind, start, current, free)
        if self._rubber_band is None:
            self._rubber_band = pg.PlotCurveItem(pen=pg.mkPen(_DRAW_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine))
            self._rubber_band.setZValue(30)
            self._plot.addItem(self._rubber_band, ignoreBounds=True)
        self._rubber_band.setData(x=preview[:, 0], y=preview[:, 1])
        if not is_finish:
            return
        self._remove_rubber_band()
        pixel_width, pixel_height = self._view_box.viewPixelSize()
        dragged_pixels = max(abs(current[0] - start[0]) / pixel_width, abs(current[1] - start[1]) / pixel_height)
        if params is not None and dragged_pixels >= _MIN_DRAG_PX:
            self._add_object(kind, params)

    def _drawn_shape(self, kind, start, end, free):
        """(params [m] or None when degenerate, (N, 2) scene preview) of a rubber-band drag start -> end."""
        si = self._to_si
        if kind == 'rectangle':
            x0, x1 = sorted((start[0], end[0]))
            y0, y1 = sorted((start[1], end[1]))
            params = {'x0': si(x0, 'x'), 'y0': si(y0, 'y'), 'x1': si(x1, 'x'), 'y1': si(y1, 'y')}
            preview = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
            valid = params['x0'] != params['x1'] and params['y0'] != params['y1']
        elif kind == 'circle':
            radius = math.dist(start, end)
            params = {'cx': si(start[0]), 'cy': si(start[1]), 'r': si(radius)}
            preview = _ellipse_polyline(start[0], start[1], radius, radius)
            valid = params['r'] > 0
        elif kind == 'ellipse':
            center = (0.5 * (start[0] + end[0]), 0.5 * (start[1] + end[1]))
            semi_a, semi_b = 0.5 * abs(end[0] - start[0]), 0.5 * abs(end[1] - start[1])
            params = {'cx': si(center[0]), 'cy': si(center[1]), 'a': si(semi_a), 'b': si(semi_b), 'angle': 0.0}
            preview = _ellipse_polyline(center[0], center[1], semi_a, semi_b)
            valid = params['a'] > 0 and params['b'] > 0
        else:   # line
            end = end if free else _snap_segment(start, end)
            params = {'x0': si(start[0]), 'y0': si(start[1]), 'x1': si(end[0]), 'y1': si(end[1])}
            preview = np.array([start, end])
            valid = (params['x0'], params['y0']) != (params['x1'], params['y1'])
        return (params if valid else None), preview

    def _add_object(self, kind, params):
        layer = self._model.layer
        if layer is not None:   # 3D: drawn on the shown layer (exact bounds: rounding would split the layer)
            if kind in PRISM_KINDS:
                params = {**params, 'z0': layer[0], 'z1': layer[1]}
            elif kind in ('point', 'line'):
                params = {**params, 'z': self._model.layer_z}
        try:
            self._document.add_geometry(kind, params)
        except (ValueError, KeyError, ExpressionError) as exc:
            kind_text = GEOMETRY_KINDS[kind].label.lower() if kind in GEOMETRY_KINDS else kind
            self._report_refused_edit(f'Add {kind_text}: {error_message(exc)}')
            return
        self.set_tool('select')

    def _update_polygon_preview(self, cursor):
        """Polyline through the polygon vertices (plus the cursor position while hovering)."""
        points = self._polygon_vertices + ([cursor] if cursor is not None else [])
        if not points:
            return
        if self._polygon_preview is None:   # (dashed edges, vertex markers); the markers must not eat clicks
            self._polygon_preview = (
                pg.PlotCurveItem(pen=pg.mkPen(_DRAW_COLOR, width=2, style=QtCore.Qt.PenStyle.DashLine)),
                _PassiveScatter(symbol='o', size=6, brush=pg.mkBrush(_DRAW_COLOR), pen=pg.mkPen(None)))
            for item in self._polygon_preview:
                item.setZValue(30)
                self._plot.addItem(item, ignoreBounds=True)
        points = np.asarray(points, dtype=np.float64)
        for item in self._polygon_preview:
            item.setData(x=points[:, 0], y=points[:, 1])

    def _finish_polygon(self):
        """Close the polygon being drawn (double click / Enter) and add it."""
        if self._tool != 'polygon' or not self._polygon_vertices:
            return
        points = []
        for x, y in self._polygon_vertices:
            point = [self._to_si(x, 'x'), self._to_si(y, 'y')]
            if not points or point != points[-1]:
                points.append(point)
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        self._cancel_drawing()
        if len(points) < 3:
            self._set_edit_message('A polygon needs at least 3 distinct vertices.')
            return
        self._add_object('polygon', {'points': points})

    def _on_escape(self):
        """Esc cancels the shape being drawn, else returns to the select tool."""
        if self._polygon_vertices or self._rubber_band is not None:
            self._cancel_drawing()
        elif self._tool != 'select':
            self.set_tool('select')

    def _cancel_drawing(self):
        if self._rubber_band is not None:
            self._drag_cancelled = True        # ignore the rest of the drag in progress
        self._polygon_vertices = []
        if self._polygon_preview is not None:
            for item in self._polygon_preview:
                self._plot.removeItem(item)
            self._polygon_preview = None
        self._remove_rubber_band()

    def _remove_rubber_band(self):
        if self._rubber_band is not None:
            self._plot.removeItem(self._rubber_band)
            self._rubber_band = None

    # ---- line profile ----
    def _ensure_profile_roi(self):
        if self._profile_roi is not None or self._display is None:
            return
        x0, x1, y0, y1 = (value / self._model.scale for value in self._display[1])
        y_middle, margin = 0.5 * (y0 + y1), 0.1 * (x1 - x0)
        roi = pg.LineSegmentROI([(x0 + margin, y_middle), (x1 - margin, y_middle)],
                                pen=pg.mkPen(_PROFILE_COLOR, width=2))
        roi.setZValue(25)
        roi.sigRegionChanged.connect(self._emit_profile)
        self._plot.addItem(roi, ignoreBounds=True)
        self._profile_roi = roi
        self._emit_profile()

    def _remove_profile_roi(self):
        if self._profile_roi is not None:
            self._profile_roi.blockSignals(True)
            self._plot.removeItem(self._profile_roi)
            self._profile_roi = None

    def _emit_profile(self, *args):
        """profile_changed({'distance' [m], 'values', 'info', 'start', 'end'}) along the profile line."""
        if self._profile_roi is None or self._display is None:
            return
        start, end = (self._profile_roi.mapToParent(handle.pos()) for handle in self._profile_roi.getHandles()[:2])
        scale = self._model.scale
        start, end = (start.x() * scale, start.y() * scale), (end.x() * scale, end.y() * scale)
        field_data = self._current_field
        native = self._native   # the field's own cells (never the decimated display grid), viz.common's rule
        values, centres = ((native.values, (native.x_centers, native.y_centers)) if native is not None
                           else (np.asarray(field_data.values), (None, None)))
        distance, samples = line_profile(values, self._display[1], start, end, *centres,
                                         max_samples=_NATIVE_PROFILE_SAMPLES)[:2]
        resampled = getattr(field_data, 'native_shape', None) is not None   # a field resampled before it got here
        self.profile_changed.emit({'distance': distance, 'values': samples, 'info': field_data.info,
                                   'start': start, 'end': end, 'resampled': resampled})

    # ---- hover readout ----
    def _on_mouse_moved(self, event_args):
        scene_position = event_args[0]
        if not self._view_box.sceneBoundingRect().contains(scene_position):
            return
        view_position = self._view_box.mapSceneToView(scene_position)
        if self._tool == 'polygon' and self._polygon_vertices:
            self._update_polygon_preview((view_position.x(), view_position.y()))
        model = self._model
        x, y = view_position.x() * model.scale, view_position.y() * model.scale   # (u, v) in side views
        u_axis, v_axis = self._shown_axes(model)
        text = f'{u_axis} = {view_position.x():.4g} {model.unit}, {v_axis} = {view_position.y():.4g} {model.unit}'
        detail = ''
        if self.mode == 'field' and self._display is not None:
            detail = self._field_readout(x, y)
        elif self.mode == 'mesh' and model.cells is not None:
            detail = self._mesh_readout(x, y)
        elif self.mode == 'geometry' and model.preview is not None:
            detail = self._material_readout(x, y)
        self.cursor_info.emit(f'{text}   {detail}' if detail else text)

    def _field_readout(self, x, y):
        values, (x0, x1, y0, y1) = self._display
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return ''
        native = self._native
        if native is not None:   # the native cell under the cursor
            ix, iy = native.cell_indices(x, y)
            value = native.values[int(ix), int(iy)]
        else:
            n_x, n_y = values.shape
            value = values[min(int((x - x0) / (x1 - x0) * n_x), n_x - 1), min(int((y - y0) / (y1 - y0) * n_y), n_y - 1)]
        info = self._current_field.info
        unit = f' {info.unit}' if info.unit and info.unit != '1' else ''
        return f'{info.label} = {float(value):.4g}{unit}'

    def _mesh_readout(self, u, v):
        """Cell indices, region and widths under scene point (u, v) [m] of the shown plane."""
        model = self._model
        cells = model.cells
        u_axis, v_axis = model.scene_axes
        if v_axis == 'z' and cells.z_edges is None:
            return ''
        u_edges, v_edges = cells.edges(u_axis), cells.edges(v_axis)
        i = int(np.searchsorted(u_edges, u, side='right')) - 1
        j = int(np.searchsorted(v_edges, v, side='right')) - 1
        if not (0 <= i < u_edges.size - 1 and 0 <= j < v_edges.size - 1):
            return ''
        width, height = u_edges[i + 1] - u_edges[i], v_edges[j + 1] - v_edges[j]
        cell_scale, cell_unit = length_unit(max(width, height))
        (u_first, u_last), (v_first, v_last) = cells.window(u_axis), cells.window(v_axis)
        region = 'window' if u_first <= i <= u_last and v_first <= j <= v_last else 'absorbing layer'
        index_text = f'({i}, {j})' if model.view == 'xy' else f'({u_axis} {i}, {v_axis} {j})'
        return f'cell {index_text} {region}: {width / cell_scale:.4g} × {height / cell_scale:.4g} {cell_unit}'

    def _material_readout(self, u, v):
        index_image, names, (u0, u1, v0, v1) = self._model.preview
        if not (u0 <= u < u1 and v0 <= v < v1):
            return ''
        n_u, n_v = index_image.shape
        index = int(index_image[int((u - u0) / (u1 - u0) * n_u), int((v - v0) / (v1 - v0) * n_v)])
        return f'material: {names[index - 1]}' if index > 0 else 'material: none'
