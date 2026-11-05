"""numpy-only plot helpers shared by viz and the GUI (no matplotlib, no Qt): axis units, colour limits, safe ratios, dB
normalisation, the probe FFT, line profiles of native fields, the line sets of triangle meshes and plane cuts of
tetrahedron meshes."""
import math
from dataclasses import dataclass

import numpy as np

_RATIO_THRESHOLD = 1e-30   # relative: dt-weighted flux spectra are ~1e-40 in absolute terms
_MAX_PROFILE_SAMPLES = 4000
MESH_LINES_MAX_EDGES = 400_000   # triangle_mesh_lines: above this, only region / boundary / curved edges are kept


def length_unit(size_m):
    """(scale [m], unit) for lengths of about size_m: m, mm, µm or nm."""
    if size_m >= 1.0:
        return 1.0, 'm'
    if size_m >= 1e-3:
        return 1e-3, 'mm'
    if size_m >= 1e-6:
        return 1e-6, 'µm'
    return 1e-9, 'nm'


def time_unit(t_s):
    """(scale [s], unit) for times of about t_s: s, ms, µs, ns, ps or fs."""
    for scale, unit in ((1.0, 's'), (1e-3, 'ms'), (1e-6, 'µs'), (1e-9, 'ns'), (1e-12, 'ps')):
        if t_s >= scale:
            return scale, unit
    return 1e-15, 'fs'


def color_limits(values, signed, vmax=None):
    """(vmin, vmax): symmetric about 0 for signed fields, the finite data range otherwise; never a zero span."""
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if signed:
        if vmax is None:
            vmax = float(np.max(np.abs(finite))) if finite.size else 0.0
        vmax = abs(float(vmax)) or 1.0
        return -vmax, vmax
    vmin = float(finite.min()) if finite.size else 0.0
    vmax = float(vmax) if vmax is not None else (float(finite.max()) if finite.size else vmin)
    if not vmax > vmin:
        vmax = vmin + (abs(vmin) * 1e-3 or 1.0)
    return vmin, vmax


def safe_ratio(values, reference):
    """values / reference, NaN where |reference| <= 1e-30 of its largest finite magnitude."""
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    finite_magnitudes = np.abs(reference[np.isfinite(reference)])
    peak = float(finite_magnitudes.max()) if finite_magnitudes.size else 0.0
    threshold = _RATIO_THRESHOLD * max(peak, 1e-300)
    return values / np.where(np.abs(reference) > threshold, reference, np.nan)


def normalised_db(power, floor_db=-300.0):
    """(power / peak, its dB value floored at floor_db); peak over the finite samples (a NaN angle stays NaN); power
    kept as is when the peak is not > 0."""
    power = np.asarray(power, dtype=np.float64)
    finite = power[np.isfinite(power)]
    peak = float(finite.max()) if finite.size else 0.0
    normalised = power / peak if peak > 0 else power
    return normalised, 10.0 * np.log10(np.maximum(normalised, 10.0 ** (floor_db / 10.0)))


def probe_spectrum(times, values, remove_mean=False):
    """(frequency [Hz], |FFT| * dt) without the DC bin of a series resampled onto its median step (off-stride flush
    samples); None below 4 samples. Shared by the Visualizer and the GUI Probes tab."""
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size < 4:
        return None
    steps = np.diff(times)
    step = float(np.median(steps))
    if not step > 0:
        return None
    if float(np.max(np.abs(steps - step))) > 1e-3 * step:
        uniform_times = times[0] + step * np.arange(int((times[-1] - times[0]) / step) + 1)
        values = np.interp(uniform_times, times, values)
    if remove_mean:
        values = values - values.mean()
    magnitude = np.abs(np.fft.rfft(values)) * step
    frequency = np.fft.rfftfreq(values.size, step)
    return frequency[1:], magnitude[1:]


def edges_from_centers(centers, low_edge, high_edge):
    """Cell edges from the cell centres: e[k+1] = 2 c[k] - e[k] from the lowest edge (exact for midpoint centres;
    rounding errors do not grow), the last edge pinned to high_edge."""
    centers = np.asarray(centers, dtype=np.float64)
    edges = np.empty(centers.size + 1)
    edges[0] = low_edge
    for index, center in enumerate(centers):
        edges[index + 1] = 2.0 * center - edges[index]
    edges[-1] = high_edge
    return edges


def containing_cells(coords, edges):
    """Index of the cell containing each coordinate, as core.domain.containing_cell: 1e-9-cell snap, an edge goes
    to the upper cell, clamped to the cells (also the GUI canvas's native-cell lookup)."""
    snap = 1e-9 * float(np.diff(edges).min())
    return np.clip(np.searchsorted(edges, coords + snap, side='right') - 1, 0, edges.size - 2)


def line_profile(values, extent, start, end, x_centers=None, y_centers=None, max_samples=_MAX_PROFILE_SAMPLES):
    """(distance from start [m], containing-cell values, centre x, centre y [m] of the sampled cells; NaN outside
    extent) along start -> end [m], two samples per finest cell (at most max_samples); a sample on a cell edge reads the
    upper cell like pos_to_cell; x_centers / y_centers = native (graded) cell centres [m], None = uniform cells over
    extent."""
    values = np.asarray(values)
    x_low, x_high, y_low, y_high = (float(edge) for edge in extent)
    n_x, n_y = values.shape
    if x_centers is None:
        x_centers = x_low + (np.arange(n_x) + 0.5) * ((x_high - x_low) / n_x)
    if y_centers is None:
        y_centers = y_low + (np.arange(n_y) + 0.5) * ((y_high - y_low) / n_y)
    x_centers = np.asarray(x_centers, dtype=np.float64)
    y_centers = np.asarray(y_centers, dtype=np.float64)
    x_edges = edges_from_centers(x_centers, x_low, x_high)
    y_edges = edges_from_centers(y_centers, y_low, y_high)
    finest_cell = min(float(np.diff(x_edges).min()), float(np.diff(y_edges).min()))
    length = math.hypot(end[0] - start[0], end[1] - start[1])
    n_samples = int(np.clip(2.0 * length / finest_cell, 2, max_samples)) if finest_cell > 0 else 2
    fraction = np.linspace(0.0, 1.0, n_samples)
    xs = start[0] + fraction * (end[0] - start[0])
    ys = start[1] + fraction * (end[1] - start[1])
    inside = (xs >= x_low) & (xs <= x_high) & (ys >= y_low) & (ys <= y_high)
    ix = containing_cells(xs, x_edges)
    iy = containing_cells(ys, y_edges)
    samples, centre_x, centre_y = (np.full(n_samples, np.nan) for _ in range(3))
    samples[inside] = values[ix[inside], iy[inside]]
    centre_x[inside], centre_y[inside] = x_centers[ix[inside]], y_centers[iy[inside]]
    return fraction * length, samples, centre_x, centre_y


# ---- triangle meshes (mesh plan T6) ----
@dataclass(frozen=True, eq=False)
class TriangleMeshLines:
    """Host line sets of a triangle mesh for drawing: segment end-point pairs [m] (rows 2i, 2i + 1 = one segment;
    curved edges as two chords through their P2 midside node), the absorbing-ring edges apart."""
    segments: np.ndarray        # float64 (2k, 2) element edges with an element inside the window
    ring_segments: np.ndarray   # float64 (2m, 2) edges whose elements all lie in the absorbing ring
    n_elements: int
    n_points: int
    n_edges: int
    window: tuple               # (x0, y0, x1, y1) [m] of the physical window
    extent: tuple               # (x0, x1, y0, y1) [m] of every node (absorbing ring included)
    quality: dict               # fem MeshQuality.as_dict() ({} without one)
    digest: str
    thinned: bool               # above max_edges: only region-boundary, outer and curved edges kept


def triangle_mesh_lines(mesh, max_edges=MESH_LINES_MAX_EDGES) -> TriangleMeshLines:
    """TriangleMeshLines of a fem TriangleMesh (duck-typed: points, edges, edge_elements, edge_curve, midside_points,
    pml_faces, region, window, quality, digest)."""
    points = np.asarray(mesh.points, dtype=np.float64)
    edges = np.asarray(mesh.edges, dtype=np.int64)
    edge_elements = np.asarray(mesh.edge_elements, dtype=np.int64)
    curved = np.asarray(mesh.edge_curve) >= 0
    midside = np.asarray(mesh.midside_points, dtype=np.float64)
    in_ring = np.asarray(mesh.pml_faces) != 0
    first, second = edge_elements[:, 0], edge_elements[:, 1]
    outer = second < 0
    neighbour = np.where(outer, first, second)
    ring_edge = in_ring[first] & in_ring[neighbour]
    keep = np.ones(len(edges), dtype=bool)
    thinned = len(edges) > max_edges
    if thinned:
        region = np.asarray(mesh.region)
        keep = outer | curved | (region[first] != region[neighbour]) | (in_ring[first] != in_ring[neighbour])

    def pairs(selected):
        indices = np.flatnonzero(selected)
        straight, bent = indices[~curved[indices]], indices[curved[indices]]
        blocks = [np.stack([points[edges[straight, 0]], points[edges[straight, 1]]], axis=1),
                  np.stack([points[edges[bent, 0]], midside[bent]], axis=1),
                  np.stack([midside[bent], points[edges[bent, 1]]], axis=1)]
        return np.concatenate(blocks).reshape(-1, 2)

    quality = getattr(mesh, 'quality', None)
    return TriangleMeshLines(
        segments=pairs(keep & ~ring_edge), ring_segments=pairs(keep & ring_edge), n_elements=int(len(mesh.triangles)),
        n_points=int(len(points)), n_edges=int(len(edges)), window=tuple(float(value) for value in mesh.window),
        extent=(float(points[:, 0].min()), float(points[:, 0].max()), float(points[:, 1].min()),
                float(points[:, 1].max())),
        quality=quality.as_dict() if quality is not None else {}, digest=str(getattr(mesh, 'digest', '')),
        thinned=thinned)


# ---- tetrahedron meshes (igor3d_plan T4b) ----
_AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}
_PLANE_NUDGE = 1e-9   # x the longest box side: nodes this close to a cut plane count as above it


@dataclass(frozen=True, eq=False)
class TetrahedronMeshData:
    """Host nodes and elements of a tetrahedron mesh for drawing plane cuts (tetrahedron_mesh_cut) and its counts."""
    points: np.ndarray          # float64 (n, 3) [m]
    tets: np.ndarray            # int64 (m, 4)
    shell: np.ndarray           # bool (m,): centroid outside the window (PML shell)
    n_edges: int
    window: tuple               # (x0, y0, z0, x1, y1, z1) [m] of the physical window
    extent: tuple               # (x0, x1, y0, y1, z0, z1) [m] of every node (PML shell included)
    smallest_quality: float     # smallest radius ratio 3 r_in / r_circ
    n_poor: int                 # elements below fem.mesh3d.POOR_RADIUS_RATIO
    digest: str

    @property
    def n_elements(self) -> int:
        return int(len(self.tets))

    @property
    def n_points(self) -> int:
        return int(len(self.points))


def tetrahedron_mesh_data(mesh) -> TetrahedronMeshData:
    """TetrahedronMeshData of a fem TetrahedronMesh (duck-typed: points, tets, edges, window, quality, digest)."""
    points = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.tets, dtype=np.int64)
    low, high = np.asarray(mesh.window[:3], dtype=np.float64), np.asarray(mesh.window[3:], dtype=np.float64)
    slack = _PLANE_NUDGE * float((high - low).max())
    centroids = points[tets].mean(axis=1)
    shell = ~((centroids >= low - slack) & (centroids <= high + slack)).all(axis=1)
    return TetrahedronMeshData(
        points=points, tets=tets, shell=shell, n_edges=int(len(mesh.edges)),
        window=tuple(float(value) for value in mesh.window),
        extent=tuple(float(value) for axis in range(3) for value in (points[:, axis].min(), points[:, axis].max())),
        smallest_quality=float(mesh.quality.smallest), n_poor=int(mesh.quality.n_poor),
        digest=str(getattr(mesh, 'digest', '')))


def tetrahedron_mesh_cut(data, axis, position, plane_axes) -> tuple:
    """(window segments (2k, 2), shell segments (2m, 2)) [m]: the outline of every element's cross-section with the
    plane normal to axis ('x' | 'y' | 'z') at position, in the plane's (u, v) = plane_axes coordinates (rows 2i, 2i + 1
    = one segment; edges shared by neighbouring cross-sections drawn once)."""
    normal = _AXIS_INDEX[axis]
    in_plane = [_AXIS_INDEX[name] for name in plane_axes]
    sides = np.asarray(data.extent[1::2]) - np.asarray(data.extent[0::2])
    offset = data.points[:, normal] - float(position)
    nudge = _PLANE_NUDGE * float(sides.max())
    offset = np.where(np.abs(offset) < nudge, nudge, offset)
    above = offset[data.tets] > 0.0
    count = above.sum(axis=1)
    crossing = (count > 0) & (count < 4)
    tets, above, count, shell = data.tets[crossing], above[crossing], count[crossing], data.shell[crossing]

    def crossing_point(rows, first, second):
        """(len(rows), 2) in-plane point where element edge (column first, column second) meets the plane."""
        node_a, node_b = tets[rows, first], tets[rows, second]
        fraction = offset[node_a] / (offset[node_a] - offset[node_b])
        point = data.points[node_a] + fraction[:, None] * (data.points[node_b] - data.points[node_a])
        return point[:, in_plane]

    pieces = []   # (segments (k, 2 ends, 2), shell flags (k,))
    single = np.flatnonzero(count != 2)   # one node apart: a triangle cross-section
    if single.size:
        lone_key = above[single] ^ (count[single] == 3)[:, None]   # True at the node on its own side
        order = np.argsort(lone_key, axis=1, kind='stable')
        corners = [crossing_point(single, order[:, 3], order[:, k]) for k in range(3)]
        for k in range(3):
            pieces.append((np.stack([corners[k], corners[(k + 1) % 3]], axis=1), shell[single]))
    double = np.flatnonzero(count == 2)   # two and two: a quadrilateral a1-b1, a1-b2, a2-b2, a2-b1
    if double.size:
        order = np.argsort(~above[double], axis=1, kind='stable')   # the two above first
        corners = [crossing_point(double, order[:, a], order[:, b]) for a, b in ((0, 2), (0, 3), (1, 3), (1, 2))]
        for k in range(4):
            pieces.append((np.stack([corners[k], corners[(k + 1) % 4]], axis=1), shell[double]))
    if not pieces:
        empty = np.zeros((0, 2))
        return empty, empty
    segments = np.concatenate([piece[0] for piece in pieces])
    in_shell = np.concatenate([piece[1] for piece in pieces])
    ends = np.round(segments / (nudge * 1e3)).astype(np.int64)   # (k, 2 ends, 2) on a 1e-6 x side lattice
    swap = (ends[:, 0, 0] > ends[:, 1, 0]) | ((ends[:, 0, 0] == ends[:, 1, 0]) & (ends[:, 0, 1] > ends[:, 1, 1]))
    ends[swap] = ends[swap][:, ::-1]   # one end order per segment: shared edges compare equal
    _, first = np.unique(ends.reshape(-1, 4), axis=0, return_index=True)
    segments, in_shell = segments[first], in_shell[first]
    return segments[~in_shell].reshape(-1, 2), segments[in_shell].reshape(-1, 2)
