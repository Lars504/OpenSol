"""Self-contained HTML field viewer: scalar fields on 2D maps or 3D cut planes, turned and coloured in the browser.

`write_field_viewer` packs `ViewerFrame`s (one or more `ViewerLayer`s per frame: a grid of cell-centre samples or a
triangulation, each a 2D map or a cut plane placed in 3D) into `field_viewer.html`, a single page with no network use
(hand-written WebGL2, data embedded as base64, values int16-quantised per frame). Frames are time snapshots (the page
plays them), or labelled states (wavelengths, modes, DFT frequencies: a slider, no playback). `layer_from_field`
reads a FieldData duck-typed (viz never imports api or model); api.Results.export_html collects the frames.
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
FORMAT_VERSION = 1
FRAME_KINDS = ('time', 'labelled')    # time: snapshots the page can play; labelled: wavelengths, modes, frequencies
MAX_BYTES = 24_000_000                # quantised values of every frame and layer (the page is ~4/3 of this, base64)
MAX_FRAMES = 200
MAX_LAYER_CELLS = 400_000             # samples per grid layer after the stride
MIN_LAYER_CELLS = 62_500              # a stride never takes the largest grid layer below this (frames are thinned)
MAX_STRIDE = 16
_Q_SPAN = 65534                       # int16 codes -32767 .. 32767 map lo .. hi; -32768 = no value (NaN)
_Q_NAN = -32768
_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}
_TIME_UNITS = ((1e-15, 'fs'), (1e-12, 'ps'), (1e-9, 'ns'), (1e-6, 'µs'), (1e-3, 'ms'), (1.0, 's'))


@dataclass(eq=False)
class ViewerLayer:
    """One sampled surface of a frame. values (nu, nv) on the u_centres x v_centres grid (m) along axes, or with
    triangulation (nodes (n, 2) in (u, v) m, triangles (k, 3)) values (n,) per node; plane (normal axis, position m)
    places it in 3D, None = a 2D map at z = 0."""
    values: np.ndarray
    u_centres: np.ndarray | None = None
    v_centres: np.ndarray | None = None
    axes: tuple = ('x', 'y')
    plane: tuple | None = None
    triangulation: tuple | None = None   # (nodes (n, 2) float, triangles (k, 3) int)


@dataclass(eq=False)
class ViewerFrame:
    """One state: its layers (the same surfaces in every frame) and its readout label."""
    layers: list
    label: str = ''
    time: float | None = None


@dataclass(eq=False)
class ViewerQuantity:
    label: str
    unit: str = ''
    signed: bool = False                 # True: diverging colour maps, limits symmetric about 0


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
        return ViewerLayer(values=np.asarray(node_values, dtype=np.float64).reshape(-1), axes=axes, plane=plane,
                           triangulation=(np.asarray(nodes, dtype=np.float64), np.asarray(triangles)))
    values = np.asarray(field_data.values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f'field {field_data.physics}.{field_data.component}: a map needs 2D values, got shape '
                         f'{values.shape}')
    u0, u1, v0, v1 = (float(edge) for edge in field_data.extent)
    return ViewerLayer(values=values, axes=axes, plane=plane,
                       u_centres=_centres(field_data.x_centers, u0, u1, values.shape[0]),
                       v_centres=_centres(field_data.y_centers, v0, v1, values.shape[1]))


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
    return MAX_STRIDE


def _layer_samples(layer, stride):
    if layer.triangulation is not None:
        return int(np.size(layer.values))
    nu, nv = np.shape(layer.values)
    return (-(-nu // stride)) * (-(-nv // stride))


def _strided(layer, stride):
    if layer.triangulation is not None or stride == 1:
        return layer
    return ViewerLayer(values=np.asarray(layer.values)[::stride, ::stride], u_centres=layer.u_centres[::stride],
                       v_centres=layer.v_centres[::stride], axes=layer.axes, plane=layer.plane)


def _quantised(values, low, high):
    """int16 codes of values over [low, high]; NaN / inf -> _Q_NAN."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    span = high - low
    scaled = np.zeros_like(values) if span <= 0 else (np.where(finite, values, low) - low) / span
    codes = np.round(np.clip(scaled, 0.0, 1.0) * _Q_SPAN) - _Q_SPAN // 2
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
        return notes + ([f'every {self.stride}th cell'] if self.stride > 1 else [])


def plan_frames(layers, n_frames, max_bytes=MAX_BYTES, max_frames=MAX_FRAMES) -> FramePlan:
    """FramePlan fitting n_frames frames of these layers (one frame's, unstrided) into max_bytes of int16 values and
    max_frames: the smallest grid stride that fits, then evenly thinned frames; ValueError when one frame does not
    fit at MAX_STRIDE."""
    if n_frames < 1:
        raise ValueError('nothing to show: no field frames')
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


def viewer_data(frames, quantity, title='', subtitle='', frame_kind='labelled', view=None, plan=None,
                max_bytes=MAX_BYTES, max_frames=MAX_FRAMES):
    """The JSON-able dict the page reads. frames [ViewerFrame] (one surface set throughout), quantity ViewerQuantity,
    frame_kind 'time' | 'labelled', view (azimuth, elevation) degrees or None. plan None: frames are thinned and grid
    layers strided here (plan_frames over max_bytes / max_frames); a plan: frames already follow it (picked and
    strided by the caller, e.g. loaded one frequency at a time). The subtitle names any thinning or stride."""
    if not frames:
        raise ValueError('nothing to show: no field frames')
    if frame_kind not in FRAME_KINDS:
        raise ValueError(f'frame_kind {frame_kind!r} is not one of {FRAME_KINDS}')
    _check_layers(frames)
    if plan is None:
        plan = plan_frames(frames[0].layers, len(frames), max_bytes, max_frames)
        frames = [ViewerFrame([_strided(layer, plan.stride) for layer in frames[index].layers], frames[index].label,
                              frames[index].time) for index in plan.indices]
    elif len(frames) != len(plan.indices):
        raise ValueError(f'{len(frames)} frames for a plan of {len(plan.indices)}')
    first = frames[0].layers
    scale, unit = length_scale(max(_layer_span(layer) for layer in first) or 1.0)
    packed, ranges = [], []
    for frame in frames:
        frame_range = _finite_range(layer.values for layer in frame.layers)
        if frame_range is not None:
            ranges.append(frame_range)
        low, high = frame_range or (0.0, 0.0)   # no value at all: every code is _Q_NAN
        packed.append({'label': frame.label, 'time': None if frame.time is None else float(frame.time),
                       'lo': low, 'hi': high, 'empty': frame_range is None,
                       'values': [_b64(_quantised(layer.values, low, high)) for layer in frame.layers]})
    if not ranges:
        raise ValueError('the field has no finite value in any frame')
    low, high = min(lo for lo, _ in ranges), max(hi for _, hi in ranges)
    return {
        'format': FORMAT, 'version': FORMAT_VERSION,
        'title': title, 'subtitle': ' · '.join(part for part in (subtitle, *plan.notes) if part),
        'dimensions': 3 if any(layer.plane is not None for layer in first) else 2,
        'length_unit': unit,
        'quantity': {'label': quantity.label, 'unit': quantity.unit, 'signed': bool(quantity.signed)},
        'layers': [_layer_geometry(layer, scale) for layer in first],
        'frames': packed, 'frame_kind': frame_kind if len(packed) > 1 else 'single',
        'range': [low, high],
        'view': None if view is None else [float(view[0]), float(view[1])],
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
