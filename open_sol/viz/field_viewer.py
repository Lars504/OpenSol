"""Self-contained HTML field viewer: scalar fields on 2D maps or 3D cut planes, turned and coloured in the browser.

`write_field_viewer` packs `ViewerFrame`s (one or more `ViewerLayer`s per frame: a grid of cell-centre samples or a
triangulation, each a 2D map or a cut plane placed in 3D) into `field_viewer.html`, a single page with no network use
(hand-written WebGL2, data embedded as base64, values int16-quantised per frame). Frames are time snapshots (the page
plays them), or labelled states (wavelengths, modes, DFT frequencies: a slider, no playback). `layer_from_field`
reads a FieldData duck-typed (viz never imports api or model); api.Results.export_html collects the frames. An
optional `ViewerGeometry` adds the model objects: their outlines on every layer and, in 3D, their edges and surfaces;
`ViewerMirror`s let the page draw the mirror images of a half / quarter model (a Mirror button). `figure_settings`
checks the Plot settings a page opens with (font sizes, colour and axis limits, colour bar and label places).
"""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .mode_player import _b64, length_scale

TEMPLATE_FILE = 'field_viewer.html'
DATA_MARKER = '/*__FIELD_DATA__*/null'
FORMAT = 'open_sol.field_viewer'
FORMAT_VERSION = 3                    # 2: length_scale, figure (Plot settings); 3: inward directions in line rows
FRAME_KINDS = ('time', 'labelled')    # time: snapshots the page can play; labelled: wavelengths, modes, frequencies
MAX_BYTES = 24_000_000                # quantised values of every frame and layer (the page is ~4/3 of this, base64)
MAX_FRAMES = 200
MAX_LAYER_CELLS = 400_000             # samples per grid layer after the stride
MIN_LAYER_CELLS = 62_500              # a stride never takes the largest grid layer below this (frames are thinned)
MAX_STRIDE = 16
_Q_SPAN = 65534                       # int16 codes -32767 .. 32767 map lo .. hi; -32768 = no value (NaN)
_Q_NAN = -32768
_Q_LOW = -_Q_SPAN // 2                # lowest value code (log codes: at or below the floor, decoded as the frame's lo)
LOG_DECADES = 9                       # log codes span hi * 10^-9 .. hi (step ~0.03 % of the value)
LOG_CODE_RATIO = 1e-3                 # non-negative frames with lo <= this * hi get log codes (intensities, |E|)
_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}
_TIME_UNITS = ((1e-15, 'fs'), (1e-12, 'ps'), (1e-9, 'ns'), (1e-6, 'µs'), (1e-3, 'ms'), (1.0, 's'))
MATERIAL_COLORS = ('#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                   '#bcbd22', '#17becf')   # matplotlib tab10, as viz.scene_figure's default material colours
NO_MATERIAL_COLOR = '#b0b0b0'
GLYPH_ARM = 0.012                     # half length of a point object's cross, fraction of the largest layer span
MAX_GEOMETRY_BYTES = 8_000_000        # packed outline segments and surface triangles (float32)
MAX_VOLUME_BYTES = 16_000_000         # float16 volume samples of every frame (3D fields; the page ~4/3 of this)
MAX_VOLUME_SAMPLES = 1_100_000        # lattice samples per frame (~103^3: a ray-march stays interactive)
MIN_VOLUME_SAMPLES = 24_000           # many frames: the lattice is never coarser than this (~29^3)
_UNIFORM_TOLERANCE = 1e-3             # relative spread of the centre steps below which an axis counts as uniform
_OUTER_FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')
# Plot settings of the page (its panel edits the same keys; field_viewer.html FIGURE_CHOICES / FIGURE_RANGES)
FIGURE_CHOICES = {
    'title': ('top_left', 'top_center', 'top_right', 'off'),                     # heading: frame label and field
    'colorbar': ('bottom_right', 'bottom_left', 'top_right', 'top_left', 'right', 'left', 'top', 'bottom', 'off'),
    'horizontal_axis': ('bottom', 'top'),                                        # axis labels of a face-on map
    'vertical_axis': ('left', 'right'),
    'scale_bar': ('off', 'on'),                                                  # on: in place of the axes
    'scale_bar_color': ('auto', 'white', 'black'),
    'legend_color': ('auto', 'white', 'black'),                                  # material legend text
    'background': ('theme', 'white'),
    'export_scale': (1, 2, 4),                                                   # PNG pixels per screen pixel
}
FIGURE_RANGES = {'font_size': (6.0, 40.0), 'title_size': (8.0, 72.0),            # CSS px
                 'colorbar_length': (0.1, 1.0)}                                  # fraction of the view side
FIGURE_LIMITS = ('color', 'x', 'y', 'z')   # <name>_min / <name>_max: colour limits (field unit), axis limits (m)
FIGURE_LENGTHS = ('scale_bar_length',)     # positive lengths (m)


@dataclass(eq=False)
class ViewerLayer:
    """One sampled surface of a frame. values (nu, nv) on the u_centres x v_centres grid (m) along axes, or with
    triangulation (nodes (n, 2) in (u, v) m, triangles (k, 3)) values (n,) per node; plane (normal axis, position m)
    places it in 3D, None = a 2D map at z = 0; extent (u0, u1, v0, v1) m clips its object outlines (None: from
    the samples)."""
    values: np.ndarray
    u_centres: np.ndarray | None = None
    v_centres: np.ndarray | None = None
    axes: tuple = ('x', 'y')
    plane: tuple | None = None
    triangulation: tuple | None = None   # (nodes (n, 2) float, triangles (k, 3) int)
    extent: tuple | None = None


@dataclass(eq=False)
class ViewerGeometry:
    """Model objects drawn with the field (duck-typed parts, coordinates m): outlines = one list per layer of frame 0
    of items with polylines [(N, 3)], material (index | None), solid (False: dashed); solids (3D) items with faces
    [(N, 3) planar polygons], edges [(M, 3)], material, solid; glyphs items with vertices (1 point | 2 line, 3);
    box = the 3D domain size (outlined and fitted by the page) or None."""
    material_names: list
    outlines: list
    solids: list = field(default_factory=list)
    glyphs: list = field(default_factory=list)
    box: tuple | None = None


@dataclass(eq=False)
class ViewerVolume:
    """A 3D field on a lattice of cell centres (m): values (nx, ny, nz), NaN = no value (outside an element mesh);
    the page ray-marches it (isosurfaces, nested shells, a glowing cloud, movable slices)."""
    values: np.ndarray
    centres: tuple                       # (x, y, z) 1D arrays [m], ascending


@dataclass(eq=False)
class ViewerFrame:
    """One state: its layers (the same surfaces in every frame) and its readout label; volume: the whole 3D field
    of that state (ViewerVolume, every frame or none)."""
    layers: list
    label: str = ''
    time: float | None = None
    volume: 'ViewerVolume | None' = None


@dataclass(eq=False)
class ViewerQuantity:
    label: str
    unit: str = ''
    signed: bool = False                 # True: diverging colour maps, limits symmetric about 0
    phase: bool = False                  # an angle in rad on (-pi, pi]: an odd mirror image shifts it by pi


@dataclass(eq=False)
class ViewerMirror:
    """A mirror plane of the model the page can unfold across: face 'x_lo' ... 'z_hi', position (m) of the plane on
    its axis, value_sign of the field's image (+1 even, -1 odd: negated, a phase shifted by pi)."""
    face: str
    position: float
    value_sign: int = 1


def _centres(centres, low, high, count):
    """Cell-centre coordinates: the given ones, else uniform cells over [low, high]."""
    if centres is not None:
        return np.asarray(centres, dtype=np.float64).reshape(-1)
    return low + (np.arange(count) + 0.5) * (high - low) / count


def layer_from_field(field_data) -> ViewerLayer:
    """ViewerLayer of a FieldData (duck-typed): its display triangulation when it has one, else its window cells
    (uniform over extent when x_centers / y_centers are None); a 3D cut keeps its plane and in-plane axes."""
    axes = tuple(getattr(field_data, 'axes', ('x', 'y')))
    plane = getattr(field_data, 'plane', None)
    plane = None if plane is None else (str(plane[0]), float(plane[1]))
    triangulation = getattr(field_data, 'triangulation', None)
    if triangulation is not None:
        nodes, triangles, node_values = triangulation
        nodes, triangles, node_values = _clipped_triangulation(
            np.asarray(nodes, dtype=np.float64).reshape(-1, 2), np.asarray(triangles).reshape(-1, 3),
            np.asarray(node_values, dtype=np.float64).reshape(-1), field_data.extent)
        return ViewerLayer(values=node_values, axes=axes, plane=plane, triangulation=(nodes, triangles),
                           extent=tuple(float(edge) for edge in field_data.extent))
    values = np.asarray(field_data.values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError(f'field {field_data.physics}.{field_data.component}: a map needs at least 2 x 2 samples, '
                         f'got shape {values.shape} (a 1D field is a curve: export a figure)')
    u0, u1, v0, v1 = (float(edge) for edge in field_data.extent)
    return ViewerLayer(values=values, axes=axes, plane=plane, extent=(u0, u1, v0, v1),
                       u_centres=_centres(field_data.x_centers, u0, u1, values.shape[0]),
                       v_centres=_centres(field_data.y_centers, v0, v1, values.shape[1]))


def _clipped_triangulation(nodes, triangles, node_values, extent):
    """The triangles whose centroid lies inside extent (u0, u1, v0, v1): the window, without the absorbing ring
    the display triangulation also covers (as visualization._draw_triangles clips it); unused nodes dropped."""
    u0, u1, v0, v1 = (float(edge) for edge in extent)
    centroids = nodes[triangles].mean(axis=1)
    inside = ((centroids[:, 0] >= u0) & (centroids[:, 0] <= u1) & (centroids[:, 1] >= v0)
              & (centroids[:, 1] <= v1))
    triangles = triangles[inside]
    if not len(triangles):
        raise ValueError('no display triangle lies inside the field window')
    used = np.unique(triangles)
    renumber = np.full(len(nodes), -1, dtype=np.int64)
    renumber[used] = np.arange(used.size)
    return nodes[used], renumber[triangles], node_values[used]


def time_label(seconds) -> str:
    """'t = 1.25 ps' of a time in s."""
    seconds = float(seconds)
    magnitude = abs(seconds)
    for scale, unit in _TIME_UNITS:
        if magnitude < 1000 * scale:
            return f't = {seconds / scale:.4g} {unit}'
    return f't = {seconds:.4g} s'


# ---- packing ----
def _check_layers(frames):
    """Every frame holds the same surfaces: same count, kinds, planes and sample counts (fails loudly otherwise)."""
    first = frames[0].layers
    if not first:
        raise ValueError('a viewer frame needs at least one layer')
    for index, frame in enumerate(frames):
        if len(frame.layers) != len(first):
            raise ValueError(f'frame {index} has {len(frame.layers)} layers, frame 0 has {len(first)}')
        for layer, reference in zip(frame.layers, first):
            same = ((layer.triangulation is None) == (reference.triangulation is None)
                    and layer.plane == reference.plane and tuple(layer.axes) == tuple(reference.axes)
                    and np.shape(layer.values) == np.shape(reference.values))
            if not same:
                raise ValueError(f'frame {index}: a layer differs from frame 0 (kind, plane, axes or sample count '
                                 f'{np.shape(layer.values)} vs {np.shape(reference.values)}): one page shows one set '
                                 'of surfaces')


def _stride(layers, n_frames, max_bytes):
    """Grid stride (1 .. MAX_STRIDE): the smallest keeping every grid layer within MAX_LAYER_CELLS and all frames
    within max_bytes, but never so coarse that the largest grid layer drops below MIN_LAYER_CELLS (frames are thinned
    instead); 1 without grid layers (triangulations are never strided)."""
    grids = [layer for layer in layers if layer.triangulation is None]
    if not grids:
        return 1
    for stride in range(1, MAX_STRIDE + 1):
        if any(_layer_samples(layer, stride) > MAX_LAYER_CELLS for layer in grids):
            continue
        if 2 * sum(_layer_samples(layer, stride) for layer in layers) * n_frames <= max_bytes:
            return stride
        if max(_layer_samples(layer, stride + 1) for layer in grids) < MIN_LAYER_CELLS:
            return stride
    if any(_layer_samples(layer, MAX_STRIDE) > MAX_LAYER_CELLS for layer in grids):
        raise ValueError(f'a grid layer keeps more than {MAX_LAYER_CELLS} samples even at stride {MAX_STRIDE}: '
                         'export a coarser field')
    return MAX_STRIDE


def _layer_samples(layer, stride):
    if layer.triangulation is not None:
        return int(np.size(layer.values))
    nu, nv = np.shape(layer.values)
    return (-(-nu // stride)) * (-(-nv // stride))


def _block_mean(values, stride, axes):
    """Mean over stride-wide blocks along the given axes (the last block partial, NaN ignored; NaN where a block has
    no finite value)."""
    values = np.asarray(values, dtype=np.float64)
    pad = [(0, (-size) % stride if axis in axes else 0) for axis, size in enumerate(values.shape)]
    values = np.pad(values, pad, constant_values=np.nan)
    shape, block_axes = [], []
    for axis, size in enumerate(values.shape):
        if axis in axes:
            shape.append(size // stride)
            block_axes.append(len(shape))           # (n0, s, n1, s): block axes 1 and 3
            shape.append(stride)
        else:
            shape.append(size)
    blocks = values.reshape(shape)
    block_axes = tuple(block_axes)
    finite = np.isfinite(blocks)
    count = finite.sum(axis=block_axes)
    total = np.where(finite, blocks, 0.0).sum(axis=block_axes)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _strided(layer, stride):
    """Grid layer averaged over stride x stride blocks (no aliasing of oscillating fields, edge cells kept; centres
    = block mean centres); triangulations unchanged."""
    if layer.triangulation is not None or stride == 1:
        return layer
    return ViewerLayer(values=_block_mean(layer.values, stride, (0, 1)),
                       u_centres=_block_mean(layer.u_centres, stride, (0,)),
                       v_centres=_block_mean(layer.v_centres, stride, (0,)), axes=layer.axes, plane=layer.plane,
                       extent=layer.extent)


def _log_floor(low, high):
    """log10 of the log-code floor of a frame over [low, high], None for linear codes (signed values, or a range
    not reaching down to ~0, e.g. temperatures 300 .. 310 K, whose small differences log codes would blur)."""
    if low < 0 or high <= 0 or low > LOG_CODE_RATIO * high:
        return None
    return math.log10(high) - LOG_DECADES


def _quantised(values, low, high, log_low=None):
    """int16 codes of values: linear over [low, high], or with log_low log10-spaced over [10^log_low, high] (values
    at or below the floor -> _Q_LOW, decoded as low); NaN / inf -> _Q_NAN."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    if log_low is not None:
        span = math.log10(high) - log_low
        with np.errstate(divide='ignore', invalid='ignore'):
            scaled = (np.log10(np.where(finite & (values > 0), values, 1.0)) - log_low) / span
        codes = np.clip(np.round(scaled * _Q_SPAN) + _Q_LOW, _Q_LOW + 1, _Q_LOW + _Q_SPAN)
        codes = np.where(finite & (values > 10.0 ** log_low), codes, _Q_LOW)
    else:
        span = high - low
        scaled = np.zeros_like(values) if span <= 0 else (np.where(finite, values, low) - low) / span
        codes = np.round(np.clip(scaled, 0.0, 1.0) * _Q_SPAN) + _Q_LOW
    codes = codes.astype('<i2')
    codes[~finite] = _Q_NAN
    return codes


def _layer_geometry(layer, scale):
    """JSON of a layer's surface (coordinates divided by scale)."""
    normal = None if layer.plane is None else layer.plane[0]
    position = None if layer.plane is None else layer.plane[1] / scale
    entry = {'axes': list(layer.axes), 'normal': normal, 'position': position,
             'name': '' if layer.plane is None else f'{layer.plane[0]} = {position:.4g}'}
    for axis in (*layer.axes, *(() if normal is None else (normal,))):
        if axis not in _AXIS_INDEX:
            raise ValueError(f'layer axis {axis!r} is not x, y or z (native polar cells cannot be placed: export the '
                             'image over the sector box)')
    if layer.triangulation is not None:
        nodes, triangles = layer.triangulation
        nodes = np.asarray(nodes, dtype=np.float64).reshape(-1, 2)
        triangles = np.asarray(triangles).reshape(-1, 3)
        if triangles.size and (triangles.min() < 0 or triangles.max() >= len(nodes)):
            raise ValueError('triangulation indices out of range')
        if len(nodes) != np.size(layer.values):
            raise ValueError(f'triangulation has {len(nodes)} nodes but {np.size(layer.values)} values')
        entry.update(kind='tri', count=len(nodes), nodes=_b64((nodes / scale).astype('<f4')),
                     triangles=_b64(triangles.astype('<u4')))
    else:
        nu, nv = np.shape(layer.values)
        if len(layer.u_centres) != nu or len(layer.v_centres) != nv:
            raise ValueError(f'grid centres ({len(layer.u_centres)}, {len(layer.v_centres)}) do not match values '
                             f'{(nu, nv)}')
        entry.update(kind='grid', nu=int(nu), nv=int(nv), u=_b64((layer.u_centres / scale).astype('<f4')),
                     v=_b64((layer.v_centres / scale).astype('<f4')))
    return entry


def _layer_span(layer):
    """Largest extent (m) of a layer's coordinates."""
    if layer.triangulation is not None:
        nodes = np.asarray(layer.triangulation[0], dtype=np.float64).reshape(-1, 2)
        spans = np.ptp(nodes, axis=0) if len(nodes) else np.zeros(2)
        reach = np.abs(nodes).max() if len(nodes) else 0.0
    else:
        spans = [np.ptp(layer.u_centres), np.ptp(layer.v_centres)]
        reach = max(np.abs(layer.u_centres).max(), np.abs(layer.v_centres).max())
    position = 0.0 if layer.plane is None else abs(layer.plane[1])
    return max(float(max(spans)), float(reach), position)


def _finite_range(arrays):
    """(min, max) of the finite values, None when there are none."""
    low, high = math.inf, -math.inf
    for values in arrays:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = min(low, float(finite.min())), max(high, float(finite.max()))
    return (low, high) if low <= high else None


def _evenly(count, keep):
    """Indices of keep items spread evenly over range(count), first and last kept."""
    if count <= keep:
        return list(range(count))
    return [int(index) for index in np.unique(np.round(np.linspace(0, count - 1, keep)).astype(int))]


@dataclass(eq=False)
class FramePlan:
    """Which frames a page keeps and the grid stride: frames[i] for i in indices, every stride-th grid sample."""
    indices: list
    stride: int
    total: int

    @property
    def notes(self):
        notes = [f'{len(self.indices)} of {self.total} frames'] if len(self.indices) < self.total else []
        return notes + ([f'cells averaged in {self.stride} x {self.stride} blocks'] if self.stride > 1 else [])


def plan_frames(layers, n_frames, max_bytes=MAX_BYTES, max_frames=MAX_FRAMES) -> FramePlan:
    """FramePlan fitting n_frames frames of these layers (one frame's, unstrided) into max_bytes of int16 values and
    max_frames: the smallest grid stride that fits, then evenly thinned frames; ValueError when one frame does not
    fit at MAX_STRIDE."""
    if n_frames < 1:
        raise ValueError('nothing to show: no field frames')
    geometry = sum(np.size(layer.values) * 8 + np.size(layer.triangulation[1]) * 4 for layer in layers
                   if layer.triangulation is not None)
    if geometry > max_bytes:
        raise ValueError(f'the display triangulation needs {geometry} bytes, above max_bytes {max_bytes}')
    keep = min(n_frames, max_frames)
    stride = _stride(layers, keep, max_bytes)
    per_frame = 2 * sum(_layer_samples(layer, stride) for layer in layers)
    if per_frame > max_bytes:
        raise ValueError(f'one frame needs {per_frame} bytes at stride {stride}, above max_bytes {max_bytes}: export '
                         'a coarser field')
    keep = min(keep, int(max_bytes // per_frame))
    return FramePlan(indices=_evenly(n_frames, keep), stride=stride, total=n_frames)


def strided_layer(layer, stride) -> ViewerLayer:
    """layer with every stride-th grid sample along u and v (triangulations unchanged)."""
    return _strided(layer, stride)


# ---- model objects ----
def _layer_extent(layer):
    """(u0, u1, v0, v1) m of a layer: its extent, else the sample range (grid: plus half an end cell)."""
    if layer.extent is not None:
        return tuple(float(edge) for edge in layer.extent)
    if layer.triangulation is not None:
        nodes = np.asarray(layer.triangulation[0], dtype=np.float64).reshape(-1, 2)
        return float(nodes[:, 0].min()), float(nodes[:, 0].max()), float(nodes[:, 1].min()), float(nodes[:, 1].max())
    bounds = []
    for centres in (np.asarray(layer.u_centres, dtype=np.float64), np.asarray(layer.v_centres, dtype=np.float64)):
        bounds += [centres[0] - 0.5 * (centres[1] - centres[0]), centres[-1] + 0.5 * (centres[-1] - centres[-2])]
    return tuple(float(edge) for edge in bounds)


def _segments(polylines, sides=None):
    """float64 (n, 10) rows (start xyz, end xyz, arc length at the start, unit direction into the object) of (N, 3)
    polylines (m); sides: `_ring_sides` of the polylines, None = zeros (lines drawn centred on their segments)."""
    rows = []
    for index, polyline in enumerate(polylines):
        points = np.asarray(polyline, dtype=np.float64).reshape(-1, 3)
        if len(points) < 2:
            continue
        lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
        starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
        inward = np.zeros((len(points) - 1, 3)) if sides is None else sides[index]
        rows.append(np.column_stack([points[:-1], points[1:], starts, inward]))
    return np.vstack(rows) if rows else np.empty((0, 10))


def _ring_holds(ring, point):
    """True when the closed (N, 2) ring holds the point (even-odd rule)."""
    low, high = ring[:-1], ring[1:]
    crossing = (low[:, 1] > point[1]) != (high[:, 1] > point[1])
    with np.errstate(divide='ignore', invalid='ignore'):
        at = low[:, 0] + (point[1] - low[:, 1]) * (high[:, 0] - low[:, 0]) / (high[:, 1] - low[:, 1])
    return bool(np.count_nonzero(crossing & (point[0] < at)) % 2)


def _ring_sides(polylines, layer):
    """Per polyline of one object its (N - 1, 3) unit directions from each segment into the object, in the layer
    plane: closed rings by their turning sense, reversed for a ring inside another ring of the object (a hole); open
    polylines zeros."""
    first, second = _AXIS_INDEX[layer.axes[0]], _AXIS_INDEX[layer.axes[1]]
    rings = [np.asarray(polyline, dtype=np.float64).reshape(-1, 3)[:, [first, second]] for polyline in polylines]
    closed = [len(ring) > 3 and float(np.abs(ring[0] - ring[-1]).max()) <= 1e-9 * float(np.ptp(ring, axis=0).max())
              for ring in rings]
    sides = []
    for index, ring in enumerate(rings):
        inward = np.zeros((max(len(ring) - 1, 0), 3))
        if closed[index]:
            step = np.diff(ring, axis=0)
            signed_area = 0.5 * float(np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1]))
            around = sum(_ring_holds(rings[other], ring[0]) for other in range(len(rings))
                         if other != index and closed[other])
            sense = (1.0 if signed_area > 0.0 else -1.0) * (-1.0 if around % 2 else 1.0)
            length = np.linalg.norm(step, axis=1)
            length[length == 0.0] = 1.0
            inward[:, first] = -sense * step[:, 1] / length   # left of a counter-clockwise ring's direction
            inward[:, second] = sense * step[:, 0] / length
        sides.append(inward)
    return sides


def _clipped_segments(segments, layer):
    """Segment rows clipped to the layer's extent in its (u, v) axes (Liang-Barsky, edges inclusive); misses and
    zero-length pieces dropped, arc lengths moved with the start, inward directions kept."""
    start, delta = segments[:, 0:3], segments[:, 3:6] - segments[:, 0:3]
    t_low, t_high = np.zeros(len(segments)), np.ones(len(segments))
    u0, u1, v0, v1 = _layer_extent(layer)
    for axis, low, high in ((_AXIS_INDEX[layer.axes[0]], u0, u1), (_AXIS_INDEX[layer.axes[1]], v0, v1)):
        step, origin = delta[:, axis], start[:, axis]
        inside = (origin >= low) & (origin <= high)
        moving = step != 0.0
        with np.errstate(divide='ignore', invalid='ignore'):
            to_low, to_high = (low - origin) / step, (high - origin) / step
        t_low = np.maximum(t_low, np.where(moving, np.minimum(to_low, to_high), np.where(inside, -np.inf, np.inf)))
        t_high = np.minimum(t_high, np.where(moving, np.maximum(to_low, to_high), np.where(inside, np.inf, -np.inf)))
    keep = (t_high > t_low) & np.any(delta != 0.0, axis=1)
    start, delta, t_low, t_high = start[keep], delta[keep], t_low[keep], t_high[keep]
    arc = segments[keep, 6] + t_low * np.linalg.norm(delta, axis=1)
    return np.column_stack([start + t_low[:, None] * delta, start + t_high[:, None] * delta, arc,
                            segments[keep, 7:]])


def _glyph_segments(glyphs, arm, three_d):
    """Segment rows of point / line objects: a line its segment, a point a cross of half length arm (m) along x, y
    (and z in 3D)."""
    polylines = []
    for glyph in glyphs:
        vertices = np.asarray(glyph.vertices, dtype=np.float64).reshape(-1, 3)
        if len(vertices) > 1:
            polylines.append(vertices)
            continue
        for axis in range(3 if three_d else 2):
            offset = np.zeros(3)
            offset[axis] = arm
            polylines.append(np.vstack([vertices[0] - offset, vertices[0] + offset]))
    return _segments(polylines)


def _polygon_triangles(polygon):
    """float64 (k, 3, 3) triangles of a planar simple polygon (N, 3): a fan when convex, else ear clipping in the
    plane of its largest normal component."""
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 3)
    if len(points) > 3 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        return np.empty((0, 3, 3))
    normal = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)   # Newell
    first, second = (axis for axis in range(3) if axis != int(np.argmax(np.abs(normal))))
    flat = points[:, [first, second]]
    order = list(range(len(points)))
    if np.sum(flat[:, 0] * np.roll(flat[:, 1], -1) - np.roll(flat[:, 0], -1) * flat[:, 1]) < 0.0:
        order.reverse()   # counter-clockwise in the projection

    def turn(a, b, c):
        return (flat[b, 0] - flat[a, 0]) * (flat[c, 1] - flat[b, 1]) - (flat[b, 1] - flat[a, 1]) * (flat[c, 0] - flat[b, 0])

    tolerance = 1e-12 * float(np.ptp(flat, axis=0).max()) ** 2
    count = len(order)
    if all(turn(order[k - 1], order[k], order[(k + 1) % count]) >= -tolerance for k in range(count)):
        fan = [(order[0], order[k], order[k + 1]) for k in range(1, count - 1)]
        return points[np.array(fan)]
    triangles = []
    while len(order) > 3:
        for k in range(len(order)):
            a, b, c = order[k - 1], order[k], order[(k + 1) % len(order)]
            if turn(a, b, c) <= tolerance:
                continue   # reflex or degenerate corner
            if any(turn(a, b, p) > tolerance and turn(b, c, p) > tolerance and turn(c, a, p) > tolerance
                   for p in order if p not in (a, b, c)):
                continue   # another vertex inside: not an ear
            triangles.append((a, b, c))
            order.pop(k)
            break
        else:
            break          # no ear (a self-touching polygon): fan the rest
    triangles += [(order[0], order[k], order[k + 1]) for k in range(1, len(order) - 1)]
    return points[np.array(triangles)]


def _geometry_data(geometry, layers, scale):
    """JSON of the model objects: materials, line groups (kind 'outline' per layer, 'edge', 'glyph'; segment rows
    float32: positions and arc length in length units, then the unit direction into the object, zeros = none) and
    surface triangles per material (3D solids)."""
    if len(geometry.outlines) != len(layers):
        raise ValueError(f'geometry has outlines for {len(geometry.outlines)} layers, the frames have {len(layers)}')
    lines, faces, used, total = [], [], set(), 0

    def add_lines(kind, layer, material, solid, rows):
        nonlocal total
        if len(rows):
            packed = rows.copy()
            packed[:, :7] /= scale   # the inward directions stay unit vectors
            lines.append({'kind': kind, 'layer': layer, 'material': material, 'dashed': not solid,
                          'count': int(len(rows)), 'segments': _b64(packed.astype('<f4'))})
            total += rows.size * 4
            if material is not None:
                used.add(material)

    for index, (layer, items) in enumerate(zip(layers, geometry.outlines)):
        groups = {}
        for item in items:
            # solid objects know their inside (the page draws material-coloured outlines on that side, so two
            # objects meeting on a line show both colours); subtracted / intersected inputs stay centred
            sides = _ring_sides(item.polylines, layer) if item.solid else None
            groups.setdefault((item.material, bool(item.solid)), []).append(_segments(item.polylines, sides))
        for (material, solid), rows in groups.items():
            add_lines('outline', index, material, solid, _clipped_segments(np.vstack(rows), layer))
    groups, triangles = {}, {}
    for solid in geometry.solids:
        groups.setdefault((solid.material, bool(solid.solid)), []).extend(solid.edges)
        if solid.solid and solid.faces:
            triangles.setdefault(solid.material, []).extend(_polygon_triangles(face) for face in solid.faces)
    for (material, solid), polylines in groups.items():
        add_lines('edge', None, material, solid, _segments(polylines))
    for material, parts in triangles.items():
        corners = np.concatenate(parts).reshape(-1, 3) if parts else np.empty((0, 3))
        if len(corners):
            faces.append({'material': material, 'count': int(len(corners)),
                          'positions': _b64((corners / scale).astype('<f4'))})
            total += corners.size * 4
            if material is not None:
                used.add(material)
    if geometry.glyphs:
        extents = [_layer_extent(layer) for layer in layers]
        arm = GLYPH_ARM * max(max(u1 - u0, v1 - v0) for u0, u1, v0, v1 in extents)
        three_d = any(layer.plane is not None for layer in layers)
        add_lines('glyph', None, None, True, _glyph_segments(geometry.glyphs, arm, three_d))
    if total > MAX_GEOMETRY_BYTES:
        raise ValueError(f'the model objects need {total} bytes on the page, above {MAX_GEOMETRY_BYTES}: export '
                         'without geometry')
    materials = [{'index': index, 'name': str(name), 'color': MATERIAL_COLORS[index % len(MATERIAL_COLORS)]}
                 for index, name in enumerate(geometry.material_names) if index in used]
    return {'materials': materials, 'no_material_color': NO_MATERIAL_COLOR, 'lines': lines, 'faces': faces,
            'box': None if geometry.box is None else [float(size) / scale for size in geometry.box]}


# ---- mirror planes ----
def mirror_plane(spans, face):
    """Position (m) of the outer face 'x_lo' ... on its axis: the extent edge of the spans [(axes, extent (u0, u1,
    v0, v1))] holding that axis (the window edges), None when none does; ValueError when they disagree."""
    if face not in _OUTER_FACES:
        raise ValueError(f'mirror face {face!r} is not one of {_OUTER_FACES}')
    axis, side = face.split('_')
    edges, reach = [], 0.0
    for axes, extent in spans:
        extent = [float(edge) for edge in extent]
        reach = max(reach, extent[1] - extent[0], extent[3] - extent[2], *(abs(edge) for edge in extent))
        if axis in axes:
            edges.append(extent[2 * list(axes).index(axis) + (side == 'hi')])
    if not edges:
        return None
    if max(edges) - min(edges) > 1e-6 * reach:
        raise ValueError(f'the layers end at different {face} edges {edges}: no single mirror plane')
    return edges[0]


def snap_to_mirrors(frames, mirrors):
    """Move every layer normal to a mirror's axis that lies no further from the plane than the nearest sample row of
    the grid layers along that axis (the cut next to the symmetry plane) onto the plane, in every frame (in place):
    its object outlines are then the plane's sections and the page draws it once (its image is itself)."""
    first = frames[0].layers if frames else []
    for mirror in mirrors:
        axis, position = mirror.face[0], float(mirror.position)
        offsets = [float(np.min(np.abs(np.asarray(layer.u_centres if layer.axes[0] == axis else layer.v_centres,
                                                  dtype=np.float64) - position)))
                   for layer in first if layer.triangulation is None and axis in layer.axes]
        if not offsets:
            continue
        reach = 1.01 * min(offsets)
        for index, layer in enumerate(first):
            if layer.plane is not None and layer.plane[0] == axis and abs(layer.plane[1] - position) <= reach:
                for frame in frames:
                    frame.layers[index].plane = (axis, position)


def _plane_pads(layer, mirrors, tolerance):
    """(axis index, side, position, value_sign) of the mirror planes a grid layer's extent ends on but its sample
    centres do not reach ([] for triangulations, whose nodes lie on the window edge)."""
    if layer.triangulation is not None:
        return []
    extent = _layer_extent(layer)
    pads = []
    for mirror in mirrors:
        axis, side = mirror.face.split('_')
        if axis not in layer.axes:
            continue
        index = list(layer.axes).index(axis)
        centres = np.asarray(layer.u_centres if index == 0 else layer.v_centres, dtype=np.float64)
        end = centres[0] if side == 'lo' else centres[-1]
        edge = extent[2 * index + (side == 'hi')]
        if abs(edge - mirror.position) <= tolerance < abs(end - mirror.position):
            pads.append((index, side, float(mirror.position), int(mirror.value_sign)))
    return pads


def _plane_padded(layer, pads, phase):
    """Grid layer with a sample row on each padded mirror plane (even image: the nearest row, odd: 0, a phase keeps
    the nearest), so the map and its mirror image meet without a one-cell gap."""
    values = np.asarray(layer.values, dtype=np.float64)
    centres = [np.asarray(layer.u_centres, dtype=np.float64), np.asarray(layer.v_centres, dtype=np.float64)]
    for index, side, position, sign in pads:
        nearest = np.take(values, [0 if side == 'lo' else -1], axis=index)
        row = nearest if sign > 0 or phase else np.where(np.isfinite(nearest), 0.0, np.nan)
        values = np.concatenate((row, values) if side == 'lo' else (values, row), axis=index)
        centres[index] = np.concatenate(([position], centres[index]) if side == 'lo' else (centres[index], [position]))
    return ViewerLayer(values=values, u_centres=centres[0], v_centres=centres[1], axes=layer.axes, plane=layer.plane,
                       extent=layer.extent)


def _mirror_padded_frames(frames, mirrors, phase):
    """frames with their grid layers padded onto the mirror planes (pads decided on frame 0, the same surfaces in
    every frame)."""
    if not mirrors:
        return frames
    tolerance = 1e-6 * max(_layer_span(layer) for layer in frames[0].layers)
    pads = [_plane_pads(layer, mirrors, tolerance) for layer in frames[0].layers]
    if not any(pads):
        return frames
    return [ViewerFrame([_plane_padded(layer, pad, phase) if pad else layer for layer, pad in zip(frame.layers, pads)],
                        frame.label, frame.time, frame.volume) for frame in frames]


def _mirror_data(mirrors, scale):
    """JSON of the mirror planes (positions in length units); ValueError on an unknown face, a repeated one or a
    sign other than +-1."""
    faces = [mirror.face for mirror in mirrors]
    if len(set(faces)) != len(faces):
        raise ValueError(f'mirror faces repeat: {faces}')
    data = []
    for mirror in mirrors:
        if mirror.face not in _OUTER_FACES or int(mirror.value_sign) not in (1, -1):
            raise ValueError(f'mirror {mirror.face!r} sign {mirror.value_sign!r}: faces {_OUTER_FACES}, signs +1 / -1')
        data.append({'face': mirror.face, 'axis': mirror.face[0], 'position': float(mirror.position) / scale,
                     'sign': int(mirror.value_sign)})
    return data


# ---- 3D volumes (ray-marched by the page) ----
def _volume_stride(shape, n_frames, max_bytes):
    """Common lattice stride of a volume: the smallest keeping one frame within MAX_VOLUME_SAMPLES and all frames
    within max_bytes (float16), but never coarser than MIN_VOLUME_SAMPLES allows."""
    target = min(MAX_VOLUME_SAMPLES, max(MIN_VOLUME_SAMPLES, max_bytes // (2 * max(n_frames, 1))))
    stride = 1
    while math.prod(-(-size // stride) for size in shape) > target and stride < max(shape):
        stride += 1
    return stride


def reduced_volume(volume, n_frames, min_step=None) -> ViewerVolume:
    """volume averaged in the blocks a page of n_frames volumes takes (the stride viewer_data would choose), so a
    caller collecting many frames holds small float32 lattices, not full fields; unchanged when it already fits.
    min_step (m): blocks at least this long on the finest axis (an element mesh sampled on a finer lattice: its
    piecewise values would give stair-stepped, speckled isosurfaces; block means over the element size do not)."""
    values = np.asarray(volume.values)
    stride = _volume_stride(values.shape, min(n_frames, MAX_FRAMES), MAX_VOLUME_BYTES)
    if min_step is not None:
        finest = min(float(np.min(np.diff(np.asarray(axis, dtype=np.float64)))) for axis in volume.centres)
        stride = max(stride, min(int(min_step // finest), min(values.shape) // 2))
    if stride == 1:
        return volume
    return ViewerVolume(values=_block_mean(values, stride, (0, 1, 2)).astype(np.float32),
                        centres=tuple(_block_mean(np.asarray(axis, dtype=np.float64), stride, (0,))
                                      for axis in volume.centres))


def _resample_axis(values, centres, targets, axis):
    """values linearly interpolated along axis from the ascending centres onto targets (NaN-aware: a target takes
    the finite neighbours by weight, NaN when they hold less than half of it)."""
    upper = np.clip(np.searchsorted(centres, targets, side='right'), 1, len(centres) - 1)
    lower = upper - 1
    width = centres[upper] - centres[lower]
    weight = np.clip((targets - centres[lower]) / np.where(width > 0, width, 1.0), 0.0, 1.0)
    shape = [1] * values.ndim
    shape[axis] = -1
    weight = weight.reshape(shape)
    low, high = np.take(values, lower, axis=axis), np.take(values, upper, axis=axis)
    low_ok, high_ok = np.isfinite(low), np.isfinite(high)
    w_low, w_high = (1.0 - weight) * low_ok, weight * high_ok
    total = w_low + w_high
    with np.errstate(invalid='ignore', divide='ignore'):
        mixed = (w_low * np.where(low_ok, low, 0.0) + w_high * np.where(high_ok, high, 0.0)) / total
    return np.where(total >= 0.5, mixed, np.nan)


def _volume_lattice(volume, stride):
    """(values (mx, my, mz) float64, lo (3,), hi (3,)) of a ViewerVolume on a uniform lattice: stride^3 block means
    (no aliasing of oscillating fields), then a graded axis resampled linearly onto as many uniform centres between
    its first and last centre (a 3D texture is uniform)."""
    values = np.asarray(volume.values, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 2:
        raise ValueError(f'a viewer volume needs at least 2 x 2 x 2 samples, got shape {values.shape}')
    centres = [np.asarray(axis, dtype=np.float64).reshape(-1) for axis in volume.centres]
    if len(centres) != 3 or any(len(axis) != size for axis, size in zip(centres, values.shape)):
        raise ValueError(f'volume centres {[len(axis) for axis in centres]} do not match its values {values.shape}')
    if any(np.any(np.diff(axis) <= 0) for axis in centres):
        raise ValueError('volume centres must be strictly ascending along every axis')
    if stride > 1:
        values = _block_mean(values, stride, (0, 1, 2))
        centres = [_block_mean(axis, stride, (0,)) for axis in centres]
    for axis, axis_centres in enumerate(centres):
        steps = np.diff(axis_centres)
        if steps.size and float(np.ptp(steps)) > _UNIFORM_TOLERANCE * float(steps.mean()):
            targets = np.linspace(axis_centres[0], axis_centres[-1], len(axis_centres))
            values = _resample_axis(values, axis_centres, targets, axis)
            centres[axis] = targets
    return values, np.array([axis[0] for axis in centres]), np.array([axis[-1] for axis in centres])


def _volume_code_frame(low, high, log_low, signed):
    """(offset, scale) of a frame's volume samples: the page reads value = offset + scale * r from the float16 r.
    Linear frames r = (v - lo) / (hi - lo) in [0, 1]; log-coded frames r = v / hi (relative precision down to the
    floor of the log scale); signed fields r = v / max |v| in [-1, 1]."""
    if signed:
        return 0.0, max(abs(low), abs(high)) or 1.0
    if log_low is not None:
        return 0.0, high or 1.0
    return low, (high - low) or 1.0


def _volume_bytes(values, offset, scale, masked):
    """base64 float16 texels in texture order (x fastest, then y, then z): r, or (r, finite) pairs when masked."""
    finite = np.isfinite(values)
    codes = np.where(finite, (values - offset) / scale, 0.0).transpose(2, 1, 0)
    if masked:
        codes = np.stack([codes, finite.transpose(2, 1, 0).astype(np.float64)], axis=-1)
    return _b64(codes.astype('<f2'))


def figure_settings(figure) -> dict:
    """The checked Plot settings {key: value} of a page (FIGURE_CHOICES / _RANGES / _LIMITS / _LENGTHS keys; None
    values = automatic, dropped); ValueError for an unknown key, a value outside its choices / range or min >= max."""
    figure = dict(figure or {})
    limit_keys = tuple(f'{name}_{end}' for name in FIGURE_LIMITS for end in ('min', 'max'))
    known = (*FIGURE_CHOICES, *FIGURE_RANGES, *limit_keys, *FIGURE_LENGTHS)
    unknown = sorted(set(figure) - set(known))
    if unknown:
        raise ValueError(f"unknown figure setting(s) {', '.join(map(repr, unknown))}; known: {', '.join(known)}")
    checked = {}
    for key, value in figure.items():
        if value is None:
            continue
        if key in FIGURE_CHOICES:
            if isinstance(value, bool) or value not in FIGURE_CHOICES[key]:
                raise ValueError(f'figure setting {key} = {value!r} is not one of {FIGURE_CHOICES[key]}')
            checked[key] = value if isinstance(value, str) else int(value)
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f'figure setting {key} = {value!r} is not a number') from None
        if not math.isfinite(number):
            raise ValueError(f'figure setting {key} = {value!r} is not finite')
        if key in FIGURE_RANGES and not FIGURE_RANGES[key][0] <= number <= FIGURE_RANGES[key][1]:
            raise ValueError(f'figure setting {key} = {number:g} is outside {FIGURE_RANGES[key]}')
        if key in FIGURE_LENGTHS and not number > 0.0:
            raise ValueError(f'figure setting {key} = {number:g} is not a positive length')
        checked[key] = number
    for name in FIGURE_LIMITS:
        low, high = checked.get(f'{name}_min'), checked.get(f'{name}_max')
        if low is not None and high is not None and not low < high:
            raise ValueError(f'figure settings {name}_min = {low:g} is not below {name}_max = {high:g}')
    return checked


def viewer_data(frames, quantity, title='', subtitle='', frame_kind='labelled', view=None, plan=None,
                max_bytes=MAX_BYTES, max_frames=MAX_FRAMES, geometry=None, mirrors=None, figure=None):
    """The JSON-able dict the page reads. frames [ViewerFrame] (one surface set throughout), quantity ViewerQuantity,
    frame_kind 'time' | 'labelled', view (azimuth, elevation) degrees or None, geometry ViewerGeometry or None,
    mirrors [ViewerMirror] the planes the page can unfold across (a Mirror button; grid layers ending on one get a
    sample row on it). plan None: frames are thinned and grid layers strided here (plan_frames over max_bytes /
    max_frames); a plan: frames already follow it (picked and strided by the caller). The subtitle names any
    thinning or stride. figure: the page's Plot settings (`figure_settings` keys) it opens with, over the ones the
    browser remembers."""
    if not frames:
        raise ValueError('nothing to show: no field frames')
    if frame_kind not in FRAME_KINDS:
        raise ValueError(f'frame_kind {frame_kind!r} is not one of {FRAME_KINDS}')
    figure = figure_settings(figure)
    _check_layers(frames)
    volumes = [frame.volume for frame in frames]
    if any(volume is not None for volume in volumes) and any(volume is None for volume in volumes):
        raise ValueError('every frame needs a volume, or none does')
    if plan is None:
        plan = plan_frames(frames[0].layers, len(frames), max_bytes, max_frames)
        frames = [ViewerFrame([_strided(layer, plan.stride) for layer in frames[index].layers], frames[index].label,
                              frames[index].time, frames[index].volume) for index in plan.indices]
    elif len(frames) != len(plan.indices):
        raise ValueError(f'{len(frames)} frames for a plan of {len(plan.indices)}')
    mirrors = list(mirrors or [])
    frames = _mirror_padded_frames(frames, mirrors, bool(quantity.phase))
    first = frames[0].layers
    span = max(_layer_span(layer) for layer in first)
    if geometry is not None and geometry.box is not None:
        span = max(span, *(float(size) for size in geometry.box))
    scale, unit = length_scale(span or 1.0)
    lattices, volume_entry = [None] * len(frames), None
    if frames[0].volume is not None:
        if quantity.phase:
            raise ValueError('a phase field has no volume view (its values wrap at +-pi): pass frames without volume')
        shape = np.shape(frames[0].volume.values)
        if any(np.shape(frame.volume.values) != shape for frame in frames):
            raise ValueError('the frame volumes differ in shape: one page shows one lattice')
        stride = _volume_stride(shape, len(frames), MAX_VOLUME_BYTES)
        lattices = [_volume_lattice(frame.volume, stride) for frame in frames]
        values0, lo, hi = lattices[0]
        masked = any(not np.all(np.isfinite(values)) for values, _, _ in lattices)
        volume_entry = {'n': [int(size) for size in values0.shape], 'lo': (lo / scale).tolist(),
                        'hi': (hi / scale).tolist(), 'masked': bool(masked),
                        'stride': int(stride)}
    packed, ranges = [], []
    for frame, lattice in zip(frames, lattices):
        frame_range = _finite_range([*(layer.values for layer in frame.layers),
                                     *(() if lattice is None else (lattice[0],))])
        if frame_range is not None:
            ranges.append(frame_range)
        low, high = frame_range or (0.0, 0.0)   # no value at all: every code is _Q_NAN
        log_low = _log_floor(low, high)
        entry = {'label': frame.label, 'time': None if frame.time is None else float(frame.time),
                 'lo': low, 'hi': high, 'empty': frame_range is None, 'log_lo': log_low,
                 'values': [_b64(_quantised(layer.values, low, high, log_low)) for layer in frame.layers]}
        if lattice is not None:
            offset, code_scale = _volume_code_frame(low, high, log_low, bool(quantity.signed))
            entry.update(volume=_volume_bytes(lattice[0], offset, code_scale, volume_entry['masked']),
                         volume_offset=offset, volume_scale=code_scale)
        packed.append(entry)
    if not ranges:
        raise ValueError('the field has no finite value in any frame')
    low, high = min(lo for lo, _ in ranges), max(hi for _, hi in ranges)
    return {
        'format': FORMAT, 'version': FORMAT_VERSION,
        'title': title,
        'subtitle': ' · '.join(part for part in (subtitle, *plan.notes) if part)
        + ('' if volume_entry is None or volume_entry['stride'] == 1
           else f" · volume cells averaged in {volume_entry['stride']}^3 blocks"),
        'dimensions': 3 if any(layer.plane is not None for layer in first) else 2,
        'length_unit': unit, 'length_scale': float(scale),   # m per length unit of the page
        'figure': figure or None,
        'quantity': {'label': quantity.label, 'unit': quantity.unit, 'signed': bool(quantity.signed),
                     'phase': bool(quantity.phase)},
        'layers': [_layer_geometry(layer, scale) for layer in first],
        'frames': packed, 'frame_kind': frame_kind if len(packed) > 1 else 'single',
        'range': [low, high],
        'view': None if view is None else [float(view[0]), float(view[1])],
        'geometry': None if geometry is None else _geometry_data(geometry, first, scale),
        'mirrors': _mirror_data(mirrors, scale),
        'volume': volume_entry,
    }


def render_html(data):
    """The page text with data embedded ('</' escaped so the JSON cannot close its script element)."""
    template = (Path(__file__).resolve().parent / TEMPLATE_FILE).read_text(encoding='utf-8')
    if DATA_MARKER not in template:
        raise RuntimeError(f'{TEMPLATE_FILE} has no data marker')
    text = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    return template.replace(DATA_MARKER, text, 1)


def write_field_viewer(path, frames, quantity, **options) -> str:
    """Write the viewer page (arguments of `viewer_data`); returns the path (.html added when missing)."""
    path = Path(path)
    if path.suffix.lower() not in ('.html', '.htm'):
        path = path.with_suffix('.html')
    text = render_html(viewer_data(frames, quantity, **options))
    partial = path.with_name(path.name + '.partial')
    partial.write_text(text, encoding='utf-8')
    partial.replace(path)
    return str(path)
