"""Own conforming Delaunay mesher (solver_plan S7.5, decision D8): size field, PSLG from straight pieces and closed
curves, Ruppert / Shewchuk refinement on scipy.spatial.Delaunay (rebuilt, or updated locally in rounds adding few
nodes), smoothing, the PML layer ring, mirror images of a half mesh, periodic window sides (matching vertices,
paired segment splits).
Host numpy, scipy inside functions; deterministic (no jitter): the same input gives the same mesh."""
import itertools
import math
from dataclasses import dataclass, replace

import numpy as np

from .mesh import FACE_BITS, FULL_TURN, PSLG, signed_areas

MIN_ANGLE = 28.0             # deg: triangles below it are refined (circumcentre insertion)
SMALL_INPUT_ANGLE = 60.0     # deg: input angles below it are accepted and flagged (concentric-shell splitting)
MAX_ROUNDS = 400             # refinement rounds before the mesher gives up (RuntimeError)
SMOOTHING_PASSES = 3
MERGE_TOLERANCE = 1e-9       # relative to the window's longest side: coincident vertices / breakpoints
_BLOCK_ROWS = 256            # pairwise piece tests per block
_CURVE_CROSSING_SAMPLES = 1024   # chords of a curve when locating its crossings with another curve
_NEWTON_STEPS = 30
_SHELL_TOLERANCE = 0.1       # relative distance mismatch of two points on concentric shells around an acute apex
_CURVATURE_SAMPLES = 720     # ellipse curvature-size samples per turn
_EDGE_KEY_BASE = 1 << 31     # edge key = lower node x base + higher node: stable across rounds (nodes only appended)
_SPACED_PARALLEL_ROUNDS = 16   # vectorised greedy rounds of _spaced before the sequential tail
_LOCAL_UPDATE_SHARE = 0.25   # a round adding at most this share of the nodes updates the triangulation locally
_CAVITY_SLACK = 1e-7         # circumradius enlargement of the local update's cavity test (near co-circular triangles)
_QHULL_OPTIONS = 'Qbb Qc Qz Qx Q12'   # scipy's 2D default + Qx (exact pre-merges): ~0.8x on degenerate inputs
_QHULL_FRAME = True          # Qhull with a far triangle added: thousands of collinear hull nodes made it 20-40x slower
_FRAME_RADIUS = 4.0          # frame circumradius / half the points' longest box side (its incircle holds the box)
_FRAME_DIRECTIONS = np.column_stack([np.cos(np.radians([90.0, 210.0, 330.0])),
                                     np.sin(np.radians([90.0, 210.0, 330.0]))])


@dataclass(frozen=True, eq=False)
class RawMesh:
    """Mesher output before tagging: nodes (PSLG vertices first, in PSLG order), ccw elements, constraint edges."""
    points: np.ndarray               # float64 (n, 2) [m]
    triangles: np.ndarray            # int32 (m, 3)
    constraint_edges: np.ndarray     # int32 (k, 2) node pairs of the split PSLG segments
    constraint_segment: np.ndarray   # int32 (k,) PSLG segment of each
    rounds: int


class SizeField:
    """Element size h(x) [m] = min(h_max, region sizes inside their regions, h_k + (growth - 1) d_k(x) over sampled
    sources), floored at h_min; call with (n, 2) points [m]."""

    def __init__(self, h_max, growth, h_min=0.0):
        if not h_max > 0.0:
            raise ValueError(f'size field: maximum element size must be > 0, got {h_max!r}')
        if not growth > 1.0:
            raise ValueError(f'size field: growth rate must be > 1, got {growth!r}')
        self.h_max = float(h_max)
        self.growth = float(growth)
        self.h_min = float(h_min)
        self._regions = []   # (inside(points) -> bool, h)
        self._sources = {}   # h -> [(n, 2) points]
        self._trees = None

    def add_region(self, inside, h):
        """Size h inside the region inside(points) -> bool; its outline should also be added as a source."""
        if h < self.h_max:
            self._regions.append((inside, float(h)))

    def add_points(self, points, h):
        """Distance source of size h at points."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if h < self.h_max and len(points):
            self._sources.setdefault(float(h), []).append(points)
            self._trees = None

    def add_polyline(self, polyline, h):
        """Distance source of size h along the (n, 2) polyline, sampled every h / 4."""
        polyline = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
        if h >= self.h_max or len(polyline) < 2:
            self.add_points(polyline, h)
            return
        counts = np.maximum(1, np.ceil(4.0 * np.linalg.norm(np.diff(polyline, axis=0), axis=1) / h)).astype(np.int64)
        segment = np.repeat(np.arange(len(counts)), counts)
        steps = np.arange(1, int(counts.sum()) + 1) - np.repeat(np.cumsum(counts) - counts, counts)   # 1 .. count
        fractions = (steps / counts[segment])[:, None]
        start, end = polyline[segment], polyline[segment + 1]
        self.add_points(np.concatenate([polyline[:1], start + fractions * (end - start)]), h)

    def add_curve(self, curve, curvature_factor, h_min):
        """Curvature sources of a curve (an arc piece: over its range): h = max(curvature_factor x radius of curvature,
        h_min) along it (ellipse sizes rounded down to steps of 2^(1/8) so the sources stay few)."""
        closed = curve.is_closed
        count = _CURVATURE_SAMPLES if closed else max(2, int(math.ceil(_CURVATURE_SAMPLES * curve.span / FULL_TURN)))
        t = curve.t0 + np.linspace(0.0, curve.span, count, endpoint=not closed)
        sizes = np.maximum(curvature_factor * curve.curvature_radius(t), h_min)
        if curve.is_circle:
            self.add_polyline(np.vstack([curve.points(t), curve.points(t[:1])]) if closed else curve.points(t),
                              float(sizes[0]))
            return
        steps = 2.0 ** (np.floor(8.0 * np.log2(sizes)) / 8.0)
        points = curve.points(t)
        for size in np.unique(steps):
            self.add_points(points[steps == size], float(size))

    def __call__(self, points) -> np.ndarray:
        from scipy.spatial import cKDTree

        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        sizes = np.full(len(points), self.h_max)
        for inside, size in self._regions:
            mask = np.asarray(inside(points), dtype=bool)
            sizes[mask] = np.minimum(sizes[mask], size)
        if self._trees is None:
            self._trees = [(size, cKDTree(np.concatenate(arrays))) for size, arrays in sorted(self._sources.items())]
        slope = self.growth - 1.0
        for size, tree in self._trees:   # ascending sizes: a source lowers only points still above its size
            active = np.flatnonzero(sizes > size)
            if not len(active):
                break
            reach = (float(sizes[active].max()) - size) / slope
            distance, _ = tree.query(points[active], distance_upper_bound=reach)
            sizes[active] = np.minimum(sizes[active], size + slope * distance)
        return np.maximum(sizes, self.h_min)


# ---- PSLG ----
def _curve_pairs(curves, tolerance) -> list:
    """(first, second) index pairs, first < second, of curves whose bounding circles (centre, max(a, b)) come within
    tolerance: the candidates of _curve_curve_points (a k-d tree query instead of every pair: arrays of holes)."""
    if len(curves) < 2:
        return []
    from scipy.spatial import cKDTree
    centres = np.array([(curve.cx, curve.cy) for curve in curves], dtype=np.float64)
    radii = np.array([max(curve.a, curve.b) for curve in curves], dtype=np.float64)
    pairs = cKDTree(centres).query_pairs(2.0 * float(radii.max()) + tolerance, output_type='ndarray')
    if not len(pairs):
        return []
    distance = np.hypot(*(centres[pairs[:, 0]] - centres[pairs[:, 1]]).T)
    pairs = pairs[distance <= radii[pairs[:, 0]] + radii[pairs[:, 1]] + tolerance]
    return sorted((int(min(pair)), int(max(pair))) for pair in pairs)


def build_pslg(window, size_field, *, straight=(), curves=(), points=None, curvature_factor=0.3, h_min=0.0,
               periodic_axes=()) -> PSLG:
    """PSLG of the window rectangle (edges tagged x_lo / x_hi / y_lo / y_hi) plus straight pieces
    [(x0, y0, x1, y1, tags)], curves [(Curve, tags)] (closed or arc pieces: arcs on one whole curve are merged, their
    ends become vertices, crossings off every arc are ignored) and points {name: (x, y)}: every crossing inserted,
    pieces clipped to the window, each sampled by the size field (curves also h <= curvature_factor x radius);
    periodic_axes ('x' / 'y'): opposite window sides get matching vertices (periodic_pslg)."""
    if periodic_axes:
        return periodic_pslg(build_pslg(window, size_field, straight=straight, curves=curves, points=points,
                                        curvature_factor=curvature_factor, h_min=h_min), periodic_axes)
    x0, y0, x1, y1 = (float(value) for value in window)
    scale = max(x1 - x0, y1 - y0)
    tolerance = MERGE_TOLERANCE * scale
    lines = [(x0, y0, x1, y0, ('y_lo',)), (x1, y0, x1, y1, ('x_hi',)), (x1, y1, x0, y1, ('y_hi',)),
             (x0, y1, x0, y0, ('x_lo',))]
    lines += [tuple(float(value) for value in piece[:4]) + (tuple(piece[4]),) for piece in straight]
    lines = [line for line in lines if math.hypot(line[2] - line[0], line[3] - line[1]) > tolerance]
    starts = np.array([line[:2] for line in lines], dtype=np.float64)
    ends = np.array([line[2:4] for line in lines], dtype=np.float64)
    line_tags = [line[4] for line in lines]
    curve_list, curve_tags, curve_arcs = _merge_curves(curves, tolerance)
    slacks = [tolerance / min(curve.a, curve.b) for curve in curve_list]   # parameter widening of the arc ends [rad]
    inside_window = _window_test(window, tolerance)
    point_items = [(name, (float(px), float(py))) for name, (px, py) in (points or {}).items()
                   if inside_window(np.array([[px, py]]))[0]]
    point_xy = np.array([xy for _, xy in point_items], dtype=np.float64).reshape(-1, 2)

    line_breaks = [[] for _ in lines]
    curve_breaks = [[] for _ in curve_list]
    _line_line_breaks(starts, ends, tolerance, line_breaks)
    _endpoint_breaks(starts, ends, np.vstack([starts, ends, point_xy]), tolerance, line_breaks)
    for curve_index, curve in enumerate(curve_list):
        _line_curve_breaks(starts, ends, curve, curve_index, tolerance, line_breaks, curve_breaks,
                           curve_arcs[curve_index])
        for xy in point_xy:
            t = float(curve.parameters(xy[None])[0])
            if (np.linalg.norm(curve.project(xy[None])[0] - xy) <= tolerance
                    and _on_arcs(curve_arcs[curve_index], t, slacks[curve_index])):
                curve_breaks[curve_index].append((t, tuple(xy)))
    for first, second in _curve_pairs(curve_list, tolerance):   # only curves whose bounding circles meet
        for xy in _curve_curve_points(curve_list[first], curve_list[second], tolerance):
            t_first = float(curve_list[first].parameters(xy[None])[0])
            t_second = float(curve_list[second].parameters(xy[None])[0])
            if (_on_arcs(curve_arcs[first], t_first, slacks[first])
                    and _on_arcs(curve_arcs[second], t_second, slacks[second])):
                curve_breaks[first].append((t_first, tuple(xy)))
                curve_breaks[second].append((t_second, tuple(xy)))

    chains = []   # (float64 (k, 2) sampled points, curve index or -1, tags)
    for index in range(len(lines)):
        for start, end in _line_pieces(starts[index], ends[index], line_breaks[index], tolerance):
            if inside_window(0.5 * (start + end)[None])[0]:
                chains.append((_sample_line(start, end, size_field), -1, line_tags[index]))
    for index, curve in enumerate(curve_list):
        for t_start, t_end, start, end in _curve_pieces(curve, curve_breaks[index], tolerance, curve_arcs[index]):
            if inside_window(curve.points([0.5 * (t_start + t_end)]))[0]:
                chains.append((_sample_arc(curve, t_start, t_end, start, end, size_field, curvature_factor, h_min),
                               index, curve_tags[index]))
    return _assemble_pslg(window, chains, point_items, curve_list, tolerance)


def _window_test(window, tolerance):
    x0, y0, x1, y1 = window
    return lambda xy: ((xy[:, 0] >= x0 - tolerance) & (xy[:, 0] <= x1 + tolerance)
                       & (xy[:, 1] >= y0 - tolerance) & (xy[:, 1] <= y1 + tolerance))


def _merge_curves(curves, tolerance) -> tuple:
    """(whole curves, tags, arcs) with equal curves merged (Curve.key; tags joined), first occurrence order; arcs per
    curve: None when an input closes it, else the input arc pieces re-parameterised on that whole curve."""
    merged = {}
    for curve, tags in curves:
        key = curve.key(tolerance)
        if key in merged:
            merged[key][1] = tuple(sorted(set(merged[key][1]) | set(tags)))
        else:
            merged[key] = [curve.whole, tuple(tags), []]
        entry = merged[key]
        if curve.is_closed:
            entry[2] = None
        elif entry[2] is not None:   # its start point's parameter on the whole curve (circle angles may differ)
            start = float(entry[0].parameters(curve.points([curve.t0]))[0])
            entry[2].append(replace(entry[0], t0=start, span=curve.span))
    values = list(merged.values())
    return ([value[0] for value in values], [value[1] for value in values],
            [None if value[2] is None else tuple(value[2]) for value in values])


def _on_arcs(arcs, t, slack) -> bool:
    """True when parameter t [rad] lies on one of the arcs (None: the closed curve), their ends widened by slack."""
    return arcs is None or any(bool(arc.covers(t, slack)) for arc in arcs)


def _cross(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _line_line_breaks(starts, ends, tolerance, line_breaks):
    """Crossings of every pair of non-parallel straight pieces (an end within tolerance snaps to that end)."""
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1)
    n_lines = len(starts)
    for block in range(0, n_lines, _BLOCK_ROWS):
        rows = np.arange(block, min(n_lines, block + _BLOCK_ROWS))
        offsets = starts[None, :, :] - starts[rows, None, :]
        denominator = _cross(directions[rows, None, :], directions[None, :, :])
        with np.errstate(divide='ignore', invalid='ignore'):
            t_first = _cross(offsets, directions[None, :, :]) / denominator
            t_second = _cross(offsets, directions[rows, None, :]) / denominator
        eps_first = (tolerance / lengths[rows])[:, None]
        eps_second = (tolerance / lengths)[None, :]
        hit = ((np.arange(n_lines)[None, :] > rows[:, None])
               & (np.abs(denominator) > 1e-12 * lengths[rows, None] * lengths[None, :])
               & (t_first >= -eps_first) & (t_first <= 1.0 + eps_first)
               & (t_second >= -eps_second) & (t_second <= 1.0 + eps_second))
        for row, column in zip(*np.nonzero(hit)):
            first = int(rows[row])
            t_a, t_b = float(t_first[row, column]), float(t_second[row, column])
            ea, eb = float(eps_first[row, 0]), float(eps_second[0, column])
            if t_a <= ea:
                xy = starts[first]
            elif t_a >= 1.0 - ea:
                xy = ends[first]
            elif t_b <= eb:
                xy = starts[column]
            elif t_b >= 1.0 - eb:
                xy = ends[column]
            else:
                xy = starts[first] + t_a * directions[first]
            line_breaks[first].append((min(max(t_a, 0.0), 1.0), tuple(xy)))
            line_breaks[int(column)].append((min(max(t_b, 0.0), 1.0), tuple(xy)))


def _endpoint_breaks(starts, ends, candidates, tolerance, line_breaks):
    """Candidate points (piece ends, point entities) lying inside a straight piece: T-junctions, collinear overlaps."""
    directions = ends - starts
    lengths = np.linalg.norm(directions, axis=1)
    units = directions / lengths[:, None]
    for block in range(0, len(candidates), _BLOCK_ROWS):
        chunk = candidates[block:block + _BLOCK_ROWS]
        offsets = chunk[:, None, :] - starts[None, :, :]
        along = (offsets * units[None, :, :]).sum(axis=2)
        across = np.abs(_cross(units[None, :, :], offsets))
        hit = (across <= tolerance) & (along > tolerance) & (along < lengths[None, :] - tolerance)
        for row, column in zip(*np.nonzero(hit)):
            line_breaks[int(column)].append((float(along[row, column] / lengths[column]), tuple(chunk[row])))


def _line_curve_breaks(starts, ends, curve, curve_index, tolerance, line_breaks, curve_breaks, arcs=None):
    """Crossings (tangencies within tolerance: one point) of the straight pieces with one curve, in its unit frame;
    arcs (_merge_curves): crossings off every arc are skipped."""
    slack = tolerance / min(curve.a, curve.b)
    local = [np.column_stack(curve.local(points)) / np.array([curve.a, curve.b]) for points in (starts, ends)]
    unit_start, direction = local[0], local[1] - local[0]
    alpha = (direction ** 2).sum(axis=1)
    beta = (unit_start * direction).sum(axis=1)
    gamma = (unit_start ** 2).sum(axis=1) - 1.0
    discriminant = beta ** 2 - alpha * gamma
    unit_tolerance = tolerance / min(curve.a, curve.b)
    lengths = np.linalg.norm(ends - starts, axis=1)
    for index in np.flatnonzero(discriminant >= -2.0 * unit_tolerance * alpha):
        if abs(discriminant[index]) <= 2.0 * unit_tolerance * alpha[index]:
            roots = [-beta[index] / alpha[index]]
        else:
            root = math.sqrt(discriminant[index])
            roots = [(-beta[index] - root) / alpha[index], (-beta[index] + root) / alpha[index]]
        eps = tolerance / lengths[index]
        for s in roots:
            if not -eps <= s <= 1.0 + eps:
                continue
            xy = (starts[index] if s <= eps else ends[index] if s >= 1.0 - eps
                  else starts[index] + s * (ends[index] - starts[index]))
            t = float(curve.parameters(np.asarray(xy)[None])[0])
            if not _on_arcs(arcs, t, slack):
                continue
            line_breaks[index].append((min(max(float(s), 0.0), 1.0), tuple(xy)))
            curve_breaks[curve_index].append((t, tuple(xy)))


def _curve_curve_points(first, second, tolerance) -> list:
    """Crossing points of two curves: circles in closed form, else chords of the first against the second refined by
    Newton on both parameters; tangencies that do not converge are skipped."""
    distance = math.hypot(second.cx - first.cx, second.cy - first.cy)
    if distance > max(first.a, first.b) + max(second.a, second.b) + tolerance:
        return []
    if first.is_circle and second.is_circle:
        r_first, r_second = first.a, second.a
        if (distance <= tolerance or distance > r_first + r_second + tolerance
                or distance < abs(r_first - r_second) - tolerance):
            return []
        along = (r_first ** 2 - r_second ** 2 + distance ** 2) / (2.0 * distance)
        half_chord = math.sqrt(max(r_first ** 2 - along ** 2, 0.0))
        unit = np.array([second.cx - first.cx, second.cy - first.cy]) / distance
        base = np.array([first.cx, first.cy]) + along * unit
        normal = np.array([-unit[1], unit[0]])
        return [base] if half_chord <= tolerance else [base + half_chord * normal, base - half_chord * normal]
    t = np.linspace(0.0, 2.0 * math.pi, _CURVE_CROSSING_SAMPLES + 1)
    chord_points = first.points(t)
    line_breaks = [[] for _ in range(_CURVE_CROSSING_SAMPLES)]
    curve_breaks = [[]]
    _line_curve_breaks(chord_points[:-1], chord_points[1:], second, 0, tolerance, line_breaks, curve_breaks)
    if not curve_breaks[0]:
        return []
    guesses = np.array([xy for _, xy in curve_breaks[0]])
    t_first, t_second = first.parameters(guesses), second.parameters(guesses)
    for _ in range(_NEWTON_STEPS):
        mismatch = first.points(t_first) - second.points(t_second)
        tangent_first, tangent_second = first.tangents(t_first), -second.tangents(t_second)
        determinant = _cross(tangent_first, tangent_second)
        safe = np.where(np.abs(determinant) > 0.0, determinant, 1.0)
        t_first = t_first - _cross(mismatch, tangent_second) / safe
        t_second = t_second - _cross(tangent_first, mismatch) / safe
    crossings = first.points(t_first)
    converged = np.linalg.norm(crossings - second.points(t_second), axis=1) <= tolerance
    found = []
    for xy in crossings[converged]:
        if all(np.linalg.norm(xy - other) > tolerance for other in found):
            found.append(xy)
    return found


def _line_pieces(start, end, breaks, tolerance) -> list:
    """Consecutive (start, end) float64 pairs of a straight piece cut at its breakpoints (ends exact)."""
    length = float(np.linalg.norm(end - start))
    interior = sorted((s, xy) for s, xy in breaks if tolerance < s * length < length - tolerance)
    nodes, last_s = [start], 0.0
    for s, xy in interior:
        if (s - last_s) * length > tolerance:
            nodes.append(np.array(xy, dtype=np.float64))
            last_s = s
    if np.linalg.norm(end - nodes[-1]) <= tolerance and len(nodes) > 1:
        nodes.pop()
    nodes.append(end)
    return list(zip(nodes[:-1], nodes[1:]))


def _curve_pieces(curve, breaks, tolerance, arcs=None) -> list:
    """(t_start, t_end, start xy, end xy) arcs of a closed curve between its breakpoints (none: one full turn); arcs
    (_merge_curves): their ends are breakpoints too and only pieces on an arc are kept."""
    turn = 2.0 * math.pi
    if arcs is not None:
        ends = [arc.t0 + offset for arc in arcs for offset in (0.0, arc.span)]
        breaks = list(breaks) + [(t, tuple(xy)) for t, xy in zip(ends, curve.points(ends))]
        return [piece for piece in _curve_pieces(curve, breaks, tolerance)
                if _on_arcs(arcs, 0.5 * (piece[0] + piece[1]), 0.0)]
    if not breaks:
        start = curve.points([0.0])[0]
        return [(0.0, turn, start, start)]
    radius = max(curve.a, curve.b)
    kept = []
    for t, xy in sorted((t % turn, xy) for t, xy in breaks):
        if not kept or (t - kept[-1][0]) * radius > tolerance:
            kept.append((t, np.array(xy, dtype=np.float64)))
    if len(kept) > 1 and (kept[0][0] + turn - kept[-1][0]) * radius <= tolerance:
        kept.pop()
    pieces = []
    for index, (t, xy) in enumerate(kept):
        next_t, next_xy = kept[(index + 1) % len(kept)]
        pieces.append((t, next_t if index + 1 < len(kept) else next_t + turn, xy, next_xy))
    return pieces


def _sample_counts(parameters, density) -> np.ndarray:
    """Interior parameters placing ceil(integral density) equal shares along the dense parameter grid."""
    widths = np.diff(parameters)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * widths)])
    total = float(cumulative[-1])
    count = max(1, int(math.ceil(total - 1e-6)))
    return np.interp(np.arange(1, count) * (total / count), cumulative, parameters)


def _dense_count(length, smallest_size) -> int:
    """Dense grid points of a piece; the -1e-9 keeps an integer ratio (0.3 / 0.06) from rounding up on one copy of an
    overlapping piece only (ulp-different ends), which would sample the copies apart."""
    return int(min(50_001, max(65, math.ceil(16.0 * length / smallest_size - 1e-9) + 1)))


def _sample_line(start, end, size_field) -> np.ndarray:
    """Points start .. end along a straight piece at the size field's spacing (direction canonical)."""
    if (end[0], end[1]) < (start[0], start[1]):
        return _sample_line(end, start, size_field)[::-1]
    length = float(np.linalg.norm(end - start))
    probe = np.linspace(0.0, 1.0, 65)
    smallest = float(size_field(start + probe[:, None] * (end - start)).min())
    s = np.linspace(0.0, 1.0, _dense_count(length, smallest))
    density = length / size_field(start + s[:, None] * (end - start))
    interior = _sample_counts(s, density)
    return np.vstack([start, start + interior[:, None] * (end - start), end])


def _sample_arc(curve, t_start, t_end, start, end, size_field, curvature_factor, h_min) -> np.ndarray:
    """Points start .. end along an arc, spacing <= min(h, max(curvature_factor x radius of curvature, h_min))."""
    def allowed(t):
        return np.minimum(size_field(curve.points(t)),
                          np.maximum(curvature_factor * curve.curvature_radius(t), h_min))

    probe = np.linspace(t_start, t_end, 65)
    speed = np.linalg.norm(curve.tangents(probe), axis=1)
    length = float(np.sum(0.5 * (speed[1:] + speed[:-1]) * np.diff(probe)))
    t = np.linspace(t_start, t_end, _dense_count(length, float(allowed(probe).min())))
    density = np.linalg.norm(curve.tangents(t), axis=1) / allowed(t)
    interior = _sample_counts(t, density)
    return np.vstack([start, curve.points(interior), end])


def _assemble_pslg(window, chains, point_items, curve_list, tolerance) -> PSLG:
    """Merge coincident chain points, dedup segments (tags joined) and flag small input angles."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    blocks = [np.array([xy for _, xy in point_items], dtype=np.float64).reshape(-1, 2)]
    blocks += [chain for chain, _, _ in chains]
    coordinates = np.concatenate(blocks)
    pairs = cKDTree(coordinates).query_pairs(tolerance, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(coordinates),) * 2)
    _, component = connected_components(graph, directed=False)
    first_of_component = np.full(component.max(initial=-1) + 1, len(coordinates))
    np.minimum.at(first_of_component, component, np.arange(len(coordinates)))
    representative = first_of_component[component]
    used, new_index = np.unique(representative, return_inverse=True)
    vertices = coordinates[used]

    segments, curve_of, tags_of, piece_of, piece_ends = {}, {}, {}, {}, []
    offset = len(point_items)
    for piece, (chain, curve_index, tags) in enumerate(chains):
        nodes = new_index[offset:offset + len(chain)]
        offset += len(chain)
        piece_ends.append((int(nodes[0]), int(nodes[-1])))
        for a, b in zip(nodes[:-1], nodes[1:]):
            if a == b:
                continue
            key = (int(min(a, b)), int(max(a, b)))
            if key in segments:
                tags_of[key] = tuple(sorted(set(tags_of[key]) | set(tags)))
                curve_of[key] = max(curve_of[key], curve_index)
            else:
                segments[key] = (int(a), int(b))
                curve_of[key], tags_of[key], piece_of[key] = curve_index, tuple(tags), piece
    keys = list(segments)
    segment_array = np.array([segments[key] for key in keys], dtype=np.int32).reshape(-1, 2)
    small_angles, acute = _small_angles(vertices, segment_array)
    return PSLG(vertices=vertices, segments=segment_array,
                segment_curve=np.array([curve_of[key] for key in keys], dtype=np.int32),
                segment_tags=tuple(tags_of[key] for key in keys),
                segment_piece=np.array([piece_of[key] for key in keys], dtype=np.int32),
                piece_ends=np.array(piece_ends, dtype=np.int32).reshape(-1, 2), curves=tuple(curve_list),
                vertex_tags={name: int(new_index[index]) for index, (name, _) in enumerate(point_items)},
                window=tuple(float(value) for value in window), small_angles=tuple(small_angles),
                acute_vertices=np.array(sorted(acute), dtype=np.int32))


def _small_angles(vertices, segments) -> tuple:
    """([(x, y, angle_deg)], {vertex}) of vertices where two consecutive incident segments meet below 60 deg."""
    ends = np.concatenate([segments, segments[:, ::-1]])
    directions = vertices[ends[:, 1]] - vertices[ends[:, 0]]
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    order = np.lexsort((angles, ends[:, 0]))
    found, acute = [], set()
    vertex_of, angle_of = ends[order, 0], angles[order]
    boundaries = np.flatnonzero(np.diff(vertex_of)) + 1
    for group in np.split(np.arange(len(order)), boundaries):
        if len(group) < 2:
            continue
        group_angles = angle_of[group]
        gaps = np.diff(np.concatenate([group_angles, group_angles[:1] + 2.0 * math.pi]))
        smallest = math.degrees(float(gaps.min()))
        if smallest < SMALL_INPUT_ANGLE:
            vertex = int(vertex_of[group[0]])
            acute.add(vertex)
            found.append((float(vertices[vertex, 0]), float(vertices[vertex, 1]), smallest))
    return found, acute


# ---- periodic window sides (new_modules_plan G3) ----
def periodic_pslg(pslg, axes) -> PSLG:
    """PSLG whose window sides of each periodic axis ('x' / 'y') carry matching vertices: every side vertex gets a
    partner at the same position along the opposite side (that side's segment split there, tags / pieces kept;
    positions within MERGE_TOLERANCE merged onto the lo side's); ValueError for a curved segment along such a side."""
    requested = set(axes)
    if requested - {'x', 'y'}:
        raise ValueError(f"periodic_pslg: axes must be 'x' / 'y', got {sorted(requested)}")
    axes = tuple(name for name in ('x', 'y') if name in requested)
    x0, y0, x1, y1 = pslg.window
    tolerance = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
    vertices = np.array(pslg.vertices, dtype=np.float64)
    segments = pslg.segments.astype(np.int64)
    segment_curve, segment_piece = pslg.segment_curve.astype(np.int64), pslg.segment_piece.astype(np.int64)
    segment_tags = list(pslg.segment_tags)
    for name in axes:
        axis = 'xy'.index(name)
        other = 1 - axis
        side_values = (float(pslg.window[axis]), float(pslg.window[axis + 2]))
        sides = [np.flatnonzero(np.all(np.abs(vertices[segments][:, :, axis] - value) <= tolerance, axis=1))
                 for value in side_values]
        for side, face in zip(sides, (f'{name}_lo', f'{name}_hi')):
            if (segment_curve[side] >= 0).any():
                raise ValueError(f'periodic {name}: a circle / ellipse runs along the {face} window side; move it off '
                                 'the side (or make the axis non-periodic)')
        nodes = [np.unique(segments[side].ravel()) for side in sides]
        along = np.concatenate([vertices[nodes[0], other], vertices[nodes[1], other]])
        owner = np.concatenate([np.zeros(len(nodes[0]), dtype=np.int64), np.ones(len(nodes[1]), dtype=np.int64)])
        node_of = np.concatenate(nodes)
        order = np.lexsort((owner, along))
        cluster = np.concatenate([[0], np.cumsum(np.diff(along[order]) > tolerance)])
        inserted = ([], [])   # per side: along positions of new vertices
        for members in np.split(order, np.flatnonzero(np.diff(cluster)) + 1):
            per_side = [members[owner[members] == side] for side in (0, 1)]
            if max(len(per_side[0]), len(per_side[1])) > 1:
                raise ValueError(f'periodic {name}: two window-side vertices closer than the merge tolerance at '
                                 f"{'xy'[other]} = {float(along[members[0]]):.6g} m")
            value = float(along[per_side[0][0]] if len(per_side[0]) else along[per_side[1][0]])
            for side in (0, 1):
                if len(per_side[side]):
                    vertices[node_of[per_side[side][0]], other] = value
                else:
                    inserted[side].append(value)
        for side in (0, 1):
            if not inserted[side]:
                continue
            values = np.array(sorted(inserted[side]), dtype=np.float64)
            first_new = len(vertices)
            new_vertices = np.empty((len(values), 2))
            new_vertices[:, axis], new_vertices[:, other] = side_values[side], values
            vertices = np.vstack([vertices, new_vertices])
            rows, extra = [], []
            for segment in sides[side]:
                a, b = (int(node) for node in segments[segment])
                low, high = sorted((vertices[a, other], vertices[b, other]))
                inside = np.flatnonzero((values > low) & (values < high))
                if not len(inside):
                    continue
                if vertices[a, other] > vertices[b, other]:
                    inside = inside[::-1]
                chain = [a] + [first_new + int(index) for index in inside] + [b]
                segments[segment] = chain[:2]
                rows += [chain[k:k + 2] for k in range(1, len(chain) - 1)]
                extra += [segment] * (len(chain) - 2)
            if rows:
                extra = np.array(extra, dtype=np.int64)
                segments = np.vstack([segments, np.array(rows, dtype=np.int64)])
                segment_curve = np.concatenate([segment_curve, segment_curve[extra]])
                segment_piece = np.concatenate([segment_piece, segment_piece[extra]])
                segment_tags += [segment_tags[index] for index in extra]
    return replace(pslg, vertices=vertices, segments=segments.astype(np.int32),
                   segment_curve=segment_curve.astype(np.int32), segment_tags=tuple(segment_tags),
                   segment_piece=segment_piece.astype(np.int32), periodic_axes=axes)


def _periodic_pairs(points, segments, axes, extent, tolerance) -> tuple:
    """(partner (m,) int64 segment on the opposite side or -1, side_axis (m,) int64 or -1, side_value (m,) float64) of
    the window-side segments of the periodic axes in normalised coordinates; ValueError when the sides do not match."""
    n_segments = len(segments)
    partner = np.full(n_segments, -1, dtype=np.int64)
    side_axis = np.full(n_segments, -1, dtype=np.int64)
    side_value = np.zeros(n_segments)
    ends = points[segments]   # (m, 2 nodes, 2 coordinates)
    for name in axes:
        axis = 'xy'.index(name)
        other = 1 - axis
        sides, ranges = [], []
        for value in (0.0, float(extent[axis])):
            on_side = np.flatnonzero(np.all(np.abs(ends[:, :, axis] - value) <= tolerance, axis=1))
            span = np.sort(ends[on_side][:, :, other], axis=1)
            order = np.argsort(span[:, 0], kind='stable')
            sides.append(on_side[order])
            ranges.append(span[order])
            side_axis[on_side], side_value[on_side] = axis, value
        if len(sides[0]) != len(sides[1]) or not len(sides[0]) or np.abs(ranges[0] - ranges[1]).max() > tolerance:
            raise ValueError(f'periodic {name}: the {name}_lo and {name}_hi window sides are split differently '
                             f'({len(sides[0])} / {len(sides[1])} segments); build the PSLG with periodic_axes')
        partner[sides[0]], partner[sides[1]] = sides[1], sides[0]
    return partner, side_axis, side_value


# ---- refinement ----
def mesh_pslg(pslg, size_field, *, h_min, min_angle=MIN_ANGLE, smoothing_passes=SMOOTHING_PASSES) -> RawMesh:
    """Conforming Delaunay refinement of the PSLG: encroached segments split (concentric shells at acute input
    vertices), circumcentres of triangles below min_angle or with an edge > h(midpoint) inserted in batches; then
    Laplacian smoothing of free nodes, a move kept only where the local smallest angle grows. Window sides of
    pslg.periodic_axes split in pairs (same position on both sides), so their nodes match."""
    x0, y0, x1, y1 = pslg.window
    origin = np.array([x0, y0])
    scale = max(x1 - x0, y1 - y0)
    extent = np.array([x1 - x0, y1 - y0]) / scale
    points = (pslg.vertices - origin) / scale
    segments = pslg.segments.astype(np.int64)
    origin_segment = np.arange(len(segments), dtype=np.int64)
    acute = np.zeros(len(points), dtype=bool)
    acute[pslg.acute_vertices] = True
    vertex_piece = np.full(len(points), -1, dtype=np.int64)
    piece_end = np.zeros(len(points), dtype=bool)
    piece_end[pslg.piece_ends.ravel()] = True
    for column in (0, 1):
        interior = ~piece_end[segments[:, column]]
        vertex_piece[segments[interior, column]] = pslg.segment_piece[interior]

    def sizes(normalised):
        return size_field(normalised * scale + origin) / scale

    state = _RefineState(points, segments, origin_segment, acute, vertex_piece, pslg, scale, origin, extent,
                         h_min / scale, math.radians(min_angle), sizes)
    if pslg.periodic_axes:
        state.partner, state.side_axis, state.side_value = _periodic_pairs(points, segments, pslg.periodic_axes,
                                                                           extent, MERGE_TOLERANCE)
    for rounds in range(1, MAX_ROUNDS + 1):
        if not state.round():
            break
    else:
        raise RuntimeError(f'triangle mesher: no convergence after {MAX_ROUNDS} rounds ({len(state.points)} nodes); '
                           f'check for tiny geometry features or raise the minimum element size')
    fixed = np.zeros(len(state.points), dtype=bool)
    fixed[:len(pslg.vertices)] = True
    fixed[state.segments.ravel()] = True
    smoothed = smooth(state.points, state.triangles, fixed, smoothing_passes)
    physical = smoothed * scale + origin
    physical[:len(pslg.vertices)] = pslg.vertices
    snap = MERGE_TOLERANCE * scale
    for axis, (low, high) in enumerate(((x0, x1), (y0, y1))):
        physical[np.abs(physical[:, axis] - low) <= snap, axis] = low
        physical[np.abs(physical[:, axis] - high) <= snap, axis] = high
    return RawMesh(points=physical, triangles=state.triangles.astype(np.int32),
                   constraint_edges=state.segments.astype(np.int32),
                   constraint_segment=state.origin_segment.astype(np.int32), rounds=rounds)


class _RefineState:
    """Working arrays of mesh_pslg in normalised coordinates (window corner 0, longest side 1)."""

    def __init__(self, points, segments, origin_segment, acute, vertex_piece, pslg, scale, origin, extent, h_min,
                 min_angle, sizes):
        self.points, self.segments, self.origin_segment = points, segments, origin_segment
        self.acute, self.vertex_piece, self.pslg = acute, vertex_piece, pslg
        self.scale, self.origin, self.extent = scale, origin, extent
        self.h_min, self.sin_min_angle, self.sizes = h_min, math.sin(min_angle), sizes
        self.triangles = None
        self.partner = self.side_axis = self.side_value = None   # periodic window-side segments (_periodic_pairs)
        self._triangulated = 0   # nodes in self.triangles
        self._size_keys, self._size_values = np.zeros(0, dtype=np.int64), np.zeros(0)   # last round's edge sizes
        self._circles = None   # (centres, radii) of self.triangles from a local update, else None (computed on use)
        self._quality = None   # (bad, size_bad) of self.triangles once _bad_triangles ran this round
        self._carried = None   # (bad, size_bad) of the rows a local update kept (the first rows of self.triangles)

    def round(self) -> bool:
        """One Delaunay update plus splits / insertions; False when nothing was left to do."""
        self.triangles = self._triangulate()
        edges = _mesh_edges(self.triangles)
        encroached = self._encroached_segments(edges)
        if encroached.any():
            self._split(np.flatnonzero(encroached))
            return True
        bad, size_bad = self._bad_triangles(edges)
        if not bad.any():
            return False
        centres, radii = _circumcentres(self.points, self.triangles[bad])
        bad_index = np.flatnonzero(bad)
        order = np.argsort(-radii, kind='stable')
        centres, radii, bad_index = centres[order], radii[order], bad_index[order]
        hits, split = self._candidate_encroachment(centres, size_bad[bad_index])
        inside = np.all((centres >= -1e-12) & (centres <= self.extent + 1e-12), axis=1)
        keep = ~hits & inside
        accepted = _spaced(centres[keep], 0.5 * radii[keep])
        new_points = centres[keep][accepted]
        if not (len(new_points) or len(split)):
            return False
        if len(split):
            self._split(split)
        if len(new_points):
            self.points = np.vstack([self.points, new_points])
            self.acute = np.concatenate([self.acute, np.zeros(len(new_points), dtype=bool)])
            self.vertex_piece = np.concatenate([self.vertex_piece, np.full(len(new_points), -1, dtype=np.int64)])
        return True

    def _triangulate(self) -> np.ndarray:
        """Delaunay triangles of the nodes: updated locally when few are new since the last round (kept rows first, their
        circumcircles and last round's quality flags carried), else Qhull."""
        n_points, new = len(self.points), len(self.points) - self._triangulated
        previous_quality, self._quality, self._carried = self._quality, None, None
        update = None
        if self.triangles is not None and 0 < new <= _LOCAL_UPDATE_SHARE * n_points:
            update = _local_delaunay(self.points, self.triangles, self._triangulated, self.extent, self._circles)
        self._triangulated = n_points
        if update is None:
            self._circles = None
            return _delaunay(self.points)
        triangles, kept, self._circles = update
        if previous_quality is not None:
            self._carried = tuple(values[kept] for values in previous_quality)
        return triangles

    def _encroached_segments(self, edges) -> np.ndarray:
        """Segments that are no Delaunay edge or (at least h_min long) see an adjacent apex at more than 90 deg."""
        segment_keys = (np.minimum(self.segments[:, 0], self.segments[:, 1]) * _EDGE_KEY_BASE
                        + np.maximum(self.segments[:, 0], self.segments[:, 1]))
        position = np.minimum(np.searchsorted(edges.keys, segment_keys), len(edges.keys) - 1)
        present = edges.keys[position] == segment_keys
        on_segment = np.zeros(len(edges.keys), dtype=bool)
        on_segment[position[present]] = True
        local = np.flatnonzero(on_segment[edges.local])   # apex tests only along present segments
        starts, ends, apexes = edges.starts[local], edges.ends[local], edges.apexes[local]
        to_start = self.points[starts] - self.points[apexes]
        to_end = self.points[ends] - self.points[apexes]
        length_squared = ((self.points[ends] - self.points[starts]) ** 2).sum(axis=1)
        obtuse_local = (to_start * to_end).sum(axis=1) < -1e-12 * length_squared
        obtuse = np.bincount(edges.local[local[obtuse_local]], minlength=len(edges.keys)) > 0
        # a present segment below h_min stays even when an apex sees it obtuse (stops split cascades near small angles)
        splittable = present & (np.linalg.norm(self.points[self.segments[:, 1]] - self.points[self.segments[:, 0]],
                                               axis=1) >= self.h_min)
        return ~present | (splittable & obtuse[position])

    def _bad_triangles(self, edges) -> tuple:
        """(bad, size_bad) bool per triangle: smallest angle below the target (unless skipped at small input angles
        or below h_min) or an edge longer than h at its midpoint. Rows a local update kept carry last round's flags
        (same nodes, edges and edge sizes); only the new rows are evaluated."""
        n_triangles = len(self.triangles)
        sizes = self._edge_sizes(edges)   # every round: keeps the per-edge size table current
        first = 0 if self._carried is None else len(self._carried[0])
        triangles = self.triangles[first:]
        local_edges = edges.local[(np.arange(3)[:, None] * n_triangles + np.arange(first, n_triangles)).T]
        lengths = np.linalg.norm(self.points[edges.high[local_edges]] - self.points[edges.low[local_edges]], axis=2)
        size_bad = (lengths / sizes[local_edges]).max(axis=1) > 1.0
        area = signed_areas(self.points, triangles)
        skinny = 2.0 * area < self.sin_min_angle * _median_of_three(lengths) * lengths.max(axis=1)
        shortest = np.argmin(lengths, axis=1)
        rows = np.arange(len(triangles))
        node_p = triangles[rows, (shortest + 1) % 3]
        node_q = triangles[rows, (shortest + 2) % 3]
        skip = (lengths[rows, shortest] < self.h_min) | self._on_shells(node_p, node_q)
        bad = size_bad | (skinny & ~skip)
        if first:
            bad = np.concatenate([self._carried[0], bad])
            size_bad = np.concatenate([self._carried[1], size_bad])
        self._quality = (bad, size_bad)
        return bad, size_bad

    def _edge_sizes(self, edges) -> np.ndarray:
        """Size field at the edge midpoints; edges of the last round keep their value (nodes never move), so only new
        edges are evaluated."""
        sizes = np.empty(len(edges.keys))
        known = np.zeros(len(edges.keys), dtype=bool)
        if len(self._size_keys):
            position = np.minimum(np.searchsorted(self._size_keys, edges.keys), len(self._size_keys) - 1)
            known = self._size_keys[position] == edges.keys
            sizes[known] = self._size_values[position[known]]
        new = ~known
        if new.any():
            sizes[new] = self.sizes(0.5 * (self.points[edges.low[new]] + self.points[edges.high[new]]))
        self._size_keys, self._size_values = edges.keys, sizes
        return sizes

    def _on_shells(self, node_p, node_q) -> np.ndarray:
        """True where p and q lie on two pieces meeting at an acute input vertex, at equal distance from it."""
        piece_p, piece_q = self.vertex_piece[node_p], self.vertex_piece[node_q]
        result = np.zeros(len(node_p), dtype=bool)
        both = (piece_p >= 0) & (piece_q >= 0) & (piece_p != piece_q)
        if not both.any():
            return result
        ends_p = self.pslg.piece_ends[piece_p[both]]
        ends_q = self.pslg.piece_ends[piece_q[both]]
        p, q = self.points[node_p[both]], self.points[node_q[both]]
        found = np.zeros(int(both.sum()), dtype=bool)
        for i in (0, 1):
            for j in (0, 1):
                apex = ends_p[:, i]
                common = (apex == ends_q[:, j]) & self.acute[apex]
                distance_p = np.linalg.norm(p - self.points[apex], axis=1)
                distance_q = np.linalg.norm(q - self.points[apex], axis=1)
                found |= common & (np.abs(distance_p - distance_q)
                                   <= _SHELL_TOLERANCE * np.minimum(distance_p, distance_q))
        result[both] = found
        return result

    def _candidate_encroachment(self, centres, size_bad) -> tuple:
        """(bool per centre encroaching a segment, segment indices to split instead)."""
        from scipy.spatial import cKDTree

        starts, ends = self.points[self.segments[:, 0]], self.points[self.segments[:, 1]]
        middles = 0.5 * (starts + ends)
        halves = 0.5 * np.linalg.norm(ends - starts, axis=1)
        segments, candidates = _ball_pairs(cKDTree(centres), middles, halves)   # centres in each diametral circle
        distances = np.linalg.norm(centres[candidates] - middles[segments], axis=1)
        inside = distances < halves[segments] * (1.0 - 1e-9)
        hits = np.zeros(len(centres), dtype=bool)
        hits[candidates[inside]] = True
        allowed = inside & (size_bad[candidates] | (2.0 * halves[segments] > self.h_min))
        return hits, np.unique(segments[allowed])

    def _split(self, which):
        """Split segments `which`: midpoint, or a concentric-shell point when exactly one end is an acute input vertex
        (straight and curved alike, so both sides of a small angle share the shells); curved ones projected; a periodic
        window-side segment is split together with its partner at the same position."""
        which = np.asarray(which, dtype=np.int64)
        if self.partner is not None:
            partners = self.partner[which]
            which = np.union1d(which, partners[partners >= 0])
        first, second = self.segments[which, 0], self.segments[which, 1]
        start, end = self.points[first], self.points[second]
        length = np.linalg.norm(end - start, axis=1)
        new_points = 0.5 * (start + end)
        curve_of = self.pslg.segment_curve[self.origin_segment[which]]
        shell_at_start = self.acute[first] & ~self.acute[second]
        shell_at_end = self.acute[second] & ~self.acute[first]
        shell = shell_at_start | shell_at_end
        if shell.any():
            apex = np.where(shell_at_start[:, None], start, end)[shell]
            other = np.where(shell_at_start[:, None], end, start)[shell]
            distance = 2.0 ** np.round(np.log2(0.5 * length[shell]))
            new_points[shell] = apex + (distance / length[shell])[:, None] * (other - apex)
        for curve_index in np.unique(curve_of[curve_of >= 0]):
            on_curve = curve_of == curve_index
            physical = new_points[on_curve] * self.scale + self.origin
            new_points[on_curve] = (self.pslg.curves[curve_index].project(physical) - self.origin) / self.scale
        pairs = self._matched_split_points(which, new_points, shell) if self.partner is not None else None
        new_index = np.arange(len(self.points), len(self.points) + len(which))
        n_segments = len(self.segments)
        self.points = np.vstack([self.points, new_points])
        self.acute = np.concatenate([self.acute, np.zeros(len(which), dtype=bool)])
        self.vertex_piece = np.concatenate([self.vertex_piece,
                                            self.pslg.segment_piece[self.origin_segment[which]].astype(np.int64)])
        self.segments[which, 1] = new_index
        self.segments = np.vstack([self.segments, np.column_stack([new_index, second])])
        self.origin_segment = np.concatenate([self.origin_segment, self.origin_segment[which]])
        if pairs is not None:
            self._pair_halves(which, first, second, n_segments, pairs)

    def _matched_split_points(self, which, new_points, shell) -> tuple:
        """(a, b) positions in `which` of each periodic segment pair (once); the split point of one (the one with a
        concentric-shell point when only it has one, else a) copied to the other along its side, in place."""
        positions = np.flatnonzero(self.partner[which] >= 0)
        mates = np.searchsorted(which, self.partner[which[positions]])
        lead = positions < mates
        a, b = positions[lead], mates[lead]
        swap = shell[b] & ~shell[a]
        master, slave = np.where(swap, b, a), np.where(swap, a, b)
        new_points[slave] = new_points[master]
        for rows in (master, slave):   # the across coordinate exactly on the row's own side
            new_points[rows, self.side_axis[which[rows]]] = self.side_value[which[rows]]
        return a, b

    def _pair_halves(self, which, first, second, n_segments, pairs):
        """Partners of the halves of split periodic pairs: segment s of `which` (position k) keeps (first, new), its
        new half n_segments + k runs (new, second); halves pair by their position along the side."""
        n_new = len(which)
        self.partner = np.concatenate([self.partner, np.full(n_new, -1, dtype=np.int64)])
        self.side_axis = np.concatenate([self.side_axis, self.side_axis[which]])
        self.side_value = np.concatenate([self.side_value, self.side_value[which]])
        a, b = pairs
        if not len(a):
            return
        other = 1 - self.side_axis[which[a]]
        along_first_a = self.points[first[a], other]
        same = (np.abs(along_first_a - self.points[first[b], other])
                <= np.abs(along_first_a - self.points[second[b], other]))
        keep_a, keep_b = which[a], which[b]          # (first, new) halves
        half_a, half_b = n_segments + a, n_segments + b   # (new, second) halves
        mate_of_keep_a = np.where(same, keep_b, half_b)
        mate_of_half_a = np.where(same, half_b, keep_b)
        self.partner[keep_a], self.partner[mate_of_keep_a] = mate_of_keep_a, keep_a
        self.partner[half_a], self.partner[mate_of_half_a] = mate_of_half_a, half_a


def _delaunay(points, cover_hull=True) -> np.ndarray:
    """Int64 (m, 3) counter-clockwise Delaunay triangles (Qhull with a far frame when _QHULL_FRAME, else or when the
    framed result does not cover the hull plain Qhull; cover_hull=False: hull triangles may be missing)."""
    triangles = _framed_simplices(points, cover_hull) if _QHULL_FRAME else None
    if triangles is None:
        triangles = _qhull_simplices(points)
    area = signed_areas(points, triangles)
    flat = np.abs(area) <= 1e-12 * np.abs(area).max()
    triangles = triangles[~flat]
    clockwise = area[~flat] < 0.0
    triangles[clockwise] = triangles[clockwise][:, [0, 2, 1]]
    return triangles


def _qhull_simplices(points) -> np.ndarray:
    """Int64 Qhull Delaunay simplices (any orientation, flat ones included); RuntimeError when Qhull drops a point."""
    from scipy.spatial import Delaunay

    triangulation = Delaunay(points, qhull_options=_QHULL_OPTIONS)
    if len(triangulation.coplanar):
        raise RuntimeError(f'triangle mesher: Qhull dropped {len(triangulation.coplanar)} coincident or degenerate '
                           f'points')
    return triangulation.simplices.astype(np.int64)


def _framed_simplices(points, cover_hull=True):
    """Qhull simplices of points plus a far triangle (no input point on the hull), the frame's triangles dropped; None
    when Qhull fails or (cover_hull) the rest does not cover the points' convex hull (area, 1e-9 relative)."""
    from scipy.spatial import ConvexHull

    low, high = points.min(axis=0), points.max(axis=0)
    frame = 0.5 * (low + high) + _FRAME_RADIUS * 0.5 * float((high - low).max()) * _FRAME_DIRECTIONS
    try:
        simplices = _qhull_simplices(np.vstack([points, frame]))
        hull_area = ConvexHull(points).volume if cover_hull else None
    except Exception:   # RuntimeError (dropped point) or QhullError (degenerate input): plain Qhull decides
        return None
    simplices = simplices[(simplices < len(points)).all(axis=1)]
    if not cover_hull:
        return simplices
    covered = np.abs(signed_areas(points, simplices)).sum()
    return simplices if abs(covered - hull_area) <= 1e-9 * hull_area else None


def _local_delaunay(points, triangles, first_new, extent, circles=None):
    """(triangles, kept, (centres, radii)) after appending points[first_new:] to the triangulation `triangles` (circles:
    its circumcircles, None = computed): the cavity (triangles whose slightly enlarged circumcircle holds a new node)
    re-triangulated by Qhull on its nodes plus the new ones; kept rows first, kept = bool per old triangle; None when
    the patch does not tile the cavity exactly (the caller rebuilds)."""
    from scipy.spatial import cKDTree

    centres, radii = _circumcentres(points, triangles) if circles is None else circles
    reach = radii * (1.0 + _CAVITY_SLACK)
    new_points = points[first_new:]
    cavity = np.zeros(len(triangles), dtype=bool)
    classes = np.floor(np.log2(reach)).astype(np.int64)   # radius classes: one ball-query bound 2^(class + 1) each
    for radius_class in np.unique(classes):
        members = np.flatnonzero(classes == radius_class)
        rows, columns = _ball_pairs(cKDTree(centres[members]), new_points,
                                    np.full(len(new_points), 2.0 ** (radius_class + 1)))
        candidates = members[columns]
        cavity[candidates[np.linalg.norm(new_points[rows] - centres[candidates], axis=1) < reach[candidates]]] = True
    cavity_triangles = triangles[cavity]
    patch_nodes = np.unique(np.concatenate([cavity_triangles.ravel(), np.arange(first_new, len(points))]))
    try:   # a far-frame patch can miss triangles on the patch hull: _tiles_cavity checks the cavity is covered
        patch = patch_nodes[_delaunay(points[patch_nodes], cover_hull=False)]
    except RuntimeError:
        return None
    # patch triangles inside the cavity: those with a new node, plus enlarged-cavity triangles that stay Delaunay
    has_new = (patch >= first_new).any(axis=1)
    replacement = patch[has_new | _same_rows(patch, cavity_triangles, len(points))]
    if not _tiles_cavity(cavity_triangles, replacement, points, extent):
        return None
    kept = ~cavity
    replacement_centres, replacement_radii = _circumcentres(points, replacement)
    return (np.vstack([triangles[kept], replacement]), kept,
            (np.vstack([centres[kept], replacement_centres]), np.concatenate([radii[kept], replacement_radii])))


def _same_rows(triangles, reference, base) -> np.ndarray:
    """Bool per triangle: its node set is a row of reference (int64 keys below ~2.1e6 nodes, else tuples)."""
    ordered, reference = np.sort(triangles, axis=1), np.sort(reference, axis=1)
    if base ** 3 < 2 ** 63:
        def keys(rows):
            return (rows[:, 0] * base + rows[:, 1]) * base + rows[:, 2]
        return np.isin(keys(ordered), keys(reference))
    known = set(map(tuple, reference.tolist()))
    return np.fromiter((row in known for row in map(tuple, ordered.tolist())), dtype=bool, count=len(ordered))


def _tiles_cavity(cavity, replacement, points, extent) -> bool:
    """True when the ccw replacement has unique directed edges, the cavity's area and its boundary away from the
    window sides (new nodes on a window side split the hull edges there)."""
    base = len(points)

    def directed(triangles):
        starts, ends = triangles.ravel(), triangles[:, [1, 2, 0]].ravel()
        return starts * base + ends, ends * base + starts

    def boundary(triangles):
        forward, backward = directed(triangles)
        keys = forward[~np.isin(forward, backward)]
        starts, ends = points[keys // base], points[keys % base]
        on_side = np.zeros(len(keys), dtype=bool)
        for axis in (0, 1):
            for value in (0.0, extent[axis]):
                on_side |= (np.abs(starts[:, axis] - value) <= 1e-12) & (np.abs(ends[:, axis] - value) <= 1e-12)
        return np.sort(keys[~on_side])

    forward, _ = directed(replacement)
    if len(np.unique(forward)) != len(forward) or not np.array_equal(boundary(replacement), boundary(cavity)):
        return False
    replacement_areas, cavity_area = signed_areas(points, replacement), signed_areas(points, cavity).sum()
    return bool((replacement_areas > 0.0).all() and abs(replacement_areas.sum() - cavity_area) <= 1e-9 * cavity_area)


@dataclass(frozen=True, eq=False)
class _Edges:
    """Unique edges of a triangulation and its 3 m local edges (edge k of triangle e at k * m + e)."""
    keys: np.ndarray     # int64 (k,) sorted: low x _EDGE_KEY_BASE + high
    low: np.ndarray      # int64 (k,) lower node
    high: np.ndarray     # int64 (k,) higher node
    starts: np.ndarray   # int64 (3 m,) local edge nodes and the opposite apex
    ends: np.ndarray
    apexes: np.ndarray
    local: np.ndarray    # int64 (3 m,) unique edge of each local edge


def _mesh_edges(triangles) -> _Edges:
    starts = np.concatenate([triangles[:, (k + 1) % 3] for k in range(3)])
    ends = np.concatenate([triangles[:, (k + 2) % 3] for k in range(3)])
    apexes = np.concatenate([triangles[:, k] for k in range(3)])
    keys, local = np.unique(np.minimum(starts, ends) * _EDGE_KEY_BASE + np.maximum(starts, ends), return_inverse=True)
    return _Edges(keys=keys, low=keys // _EDGE_KEY_BASE, high=keys % _EDGE_KEY_BASE, starts=starts, ends=ends,
                  apexes=apexes, local=local.ravel())


def _ball_pairs(tree, points, radii) -> tuple:
    """(rows, columns) int64: the tree points (columns) within radii[row] of points[row], grouped by row."""
    neighbours = tree.query_ball_point(points, r=radii)
    counts = np.fromiter(map(len, neighbours), dtype=np.int64, count=len(neighbours))
    columns = np.fromiter(itertools.chain.from_iterable(neighbours), dtype=np.int64, count=int(counts.sum()))
    return np.repeat(np.arange(len(points), dtype=np.int64), counts), columns


def _circumcentres(points, triangles) -> tuple:
    """(centres (m, 2), radii (m,)) of triangles."""
    corner = points[triangles[:, 0]]
    b = points[triangles[:, 1]] - corner
    c = points[triangles[:, 2]] - corner
    denominator = 2.0 * _cross(b, c)
    b_squared, c_squared = (b ** 2).sum(axis=1), (c ** 2).sum(axis=1)
    offset = np.column_stack([(c[:, 1] * b_squared - b[:, 1] * c_squared) / denominator,
                              (b[:, 0] * c_squared - c[:, 0] * b_squared) / denominator])
    return corner + offset, np.linalg.norm(offset, axis=1)


def _spaced(centres, spacing) -> np.ndarray:
    """Greedy bool selection in the given (priority) order: a centre is kept unless a kept earlier one lies within
    its spacing. Vectorised rounds decide every centre whose earlier neighbours are decided (the same result)."""
    from scipy.spatial import cKDTree

    n_centres = len(centres)
    if not n_centres:
        return np.zeros(0, dtype=bool)
    rows, columns = _ball_pairs(cKDTree(centres), centres, spacing)
    earlier = columns < rows
    rows, columns = rows[earlier], columns[earlier]
    state = np.zeros(n_centres, dtype=np.int8)   # 0 open, 1 kept, -1 dropped
    for _ in range(_SPACED_PARALLEL_ROUNDS):
        undecided = state[rows] == 0
        rows, columns = rows[undecided], columns[undecided]
        open_centres = state == 0
        if not open_centres.any():
            break
        waiting = np.bincount(rows[state[columns] >= 0], minlength=n_centres) > 0   # an earlier one open or kept
        state[open_centres & ~waiting] = 1
        blocked = np.bincount(rows[state[columns] == 1], minlength=n_centres) > 0
        state[(state == 0) & blocked] = -1
    bounds = np.searchsorted(rows, np.arange(n_centres + 1))   # long priority chains: the sequential greedy
    for index in np.flatnonzero(state == 0):
        state[index] = -1 if (state[columns[bounds[index]:bounds[index + 1]]] == 1).any() else 1
    return state == 1


# ---- smoothing ----
def _smallest_angle_sines(points, triangles) -> np.ndarray:
    """Signed sine of the smallest angle per triangle (2 area / the two longer edges; <= 0 inverted or flat): it grows
    with the smallest angle, which is at most 60 deg, without the arctangents of element_angles."""
    corners = points[triangles]
    squared = ((corners[:, [1, 2, 0]] - corners[:, [2, 0, 1]]) ** 2).sum(axis=2)
    twice_area = _cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return twice_area / np.sqrt(_median_of_three(squared) * squared.max(axis=1))


def _median_of_three(values) -> np.ndarray:
    """Middle value per row of an (m, 3) array (exact: no arithmetic, the sorted middle column)."""
    first, second, third = values[:, 0], values[:, 1], values[:, 2]
    return np.maximum(np.minimum(first, second), np.minimum(np.maximum(first, second), third))


def smooth(points, triangles, fixed, passes) -> np.ndarray:
    """Laplacian smoothing of the free nodes (not fixed); a node moves only when the smallest angle of its elements
    grows with every accepted move applied (connectivity unchanged, so conformity is kept)."""
    n_points = len(points)
    pairs = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    keys = np.unique(pairs[:, 0].astype(np.int64) * n_points + pairs[:, 1])   # 1D keys: no np.unique(axis=0)
    pairs = np.column_stack([keys // n_points, keys % n_points])
    degree = np.bincount(pairs.ravel(), minlength=n_points).astype(np.float64)
    free = ~fixed & (degree > 0)
    node_minimum = _node_minimum(triangles, n_points)
    for _ in range(passes):
        target = np.empty_like(points)
        for axis in (0, 1):
            target[:, axis] = (np.bincount(pairs[:, 0], weights=points[pairs[:, 1], axis], minlength=n_points)
                               + np.bincount(pairs[:, 1], weights=points[pairs[:, 0], axis], minlength=n_points))
        target /= np.maximum(degree, 1.0)[:, None]
        current = _smallest_angle_sines(points, triangles)
        old_local = node_minimum(current)
        moving = free.copy()
        trial = np.where(moving[:, None], target, points)
        trial_sines = current.copy()
        changed = moving[triangles].any(axis=1)   # only these triangles change shape
        for _ in range(20):
            # trial_sines = sines of the trial nodes on every triangle: re-evaluated only where a node changed since the
            # last trial (per-row arithmetic, so bit-identical to evaluating every moved triangle again: same meshes)
            trial_sines[changed] = _smallest_angle_sines(trial, triangles[changed])
            new_local = node_minimum(trial_sines)
            worse = moving & (new_local <= old_local + 1e-11)   # ~1e-9 deg
            if not worse.any():
                break
            moving &= ~worse
            trial[worse] = points[worse]
            changed = worse[triangles].any(axis=1)
        else:
            moving[:] = False
        points = np.where(moving[:, None], target, points)
    return points


def _node_minimum(triangles, n_points):
    """f(value per triangle) -> float64 (n_points,) minimum over each node's triangles (inf at nodes of none)."""
    corners = triangles.ravel()
    order = np.argsort(corners, kind='stable')
    sorted_nodes = corners[order]
    starts = np.flatnonzero(np.concatenate([[True], sorted_nodes[1:] != sorted_nodes[:-1]]))
    owners, triangle_of = sorted_nodes[starts], order // 3

    def minimum(values):
        result = np.full(n_points, np.inf)
        result[owners] = np.minimum.reduceat(values[triangle_of], starts)
        return result
    return minimum


# ---- PML layer ring ----
def add_layer_ring(points, triangles, window, thickness, n_layers) -> tuple:
    """(points, triangles, pml_faces uint8): n_layers structured layers outside the window on every face with
    thickness[face] > 0 [m], extruded from the window-edge nodes (corner blocks where two such faces meet; the x
    faces' strips hold them), each quad split into two triangles."""
    x0, y0, x1, y1 = window
    tolerance = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
    widths = {face: float(thickness.get(face) or 0.0) for face in FACE_BITS}
    pml_faces = np.zeros(len(triangles), dtype=np.uint8)
    if not any(widths.values()):
        return points, triangles, pml_faces
    n_layers = int(n_layers)
    offsets = {face: width * np.arange(1, n_layers + 1) / n_layers for face, width in widths.items()}
    edge_values = {'x_lo': x0, 'x_hi': x1, 'y_lo': y0, 'y_hi': y1}

    def key(x, y):
        return (round(x / tolerance), round(y / tolerance))

    on_window_edge = np.zeros(len(points), dtype=bool)
    for face, value in edge_values.items():
        on_window_edge |= np.abs(points[:, 0 if face[0] == 'x' else 1] - value) <= tolerance
    index = {key(float(px), float(py)): int(node) for node, (px, py) in zip(np.flatnonzero(on_window_edge),
                                                                        points[on_window_edge])}
    new_points, ring_triangles, ring_bits = [], [], []

    def node(x, y):
        node_key = key(x, y)
        if node_key not in index:
            index[node_key] = len(points) + len(new_points)
            new_points.append((x, y))
        return index[node_key]

    for face, width in widths.items():
        if not width:
            continue
        axis = 0 if face[0] == 'x' else 1
        value = edge_values[face]
        outward = -1.0 if face.endswith('lo') else 1.0
        along = np.unique(points[np.abs(points[:, axis] - value) <= tolerance, 1 - axis])
        if axis == 0:
            along = np.concatenate([(y0 - offsets['y_lo'][::-1]) if widths['y_lo'] else [], along,
                                    (y1 + offsets['y_hi']) if widths['y_hi'] else []])
        across = [value] + [value + outward * offset for offset in offsets[face]]
        grid = [[node(c, a) if axis == 0 else node(a, c) for a in along] for c in across]
        for layer in range(n_layers):
            for j in range(len(along) - 1):
                a, b = grid[layer][j], grid[layer + 1][j]
                c, d = grid[layer + 1][j + 1], grid[layer][j + 1]
                bits = FACE_BITS[face]
                if axis == 0 and along[j + 1] <= y0 + tolerance:
                    bits |= FACE_BITS['y_lo']
                if axis == 0 and along[j] >= y1 - tolerance:
                    bits |= FACE_BITS['y_hi']
                ring_triangles += [(a, b, c), (a, c, d)]
                ring_bits += [bits, bits]
    all_points = np.vstack([points, np.array(new_points, dtype=np.float64).reshape(-1, 2)])
    all_triangles = np.vstack([triangles, np.array(ring_triangles, dtype=triangles.dtype).reshape(-1, 3)])
    return all_points, all_triangles, np.concatenate([pml_faces, np.array(ring_bits, dtype=np.uint8)])


# ---- mirror symmetry (new_modules_plan G2) ----
def reflect_mesh(points, triangles, pml_faces, axis, plane) -> tuple:
    """(points, triangles, pml_faces, image) of a mesh on one side of the line coordinate[axis] = plane plus its
    mirror image: nodes within MERGE_TOLERANCE x the node extent of the line snapped onto it and shared, image
    elements with reversed node order (ccw kept), ring bits lo / hi swapped along axis; image (n_old,) int64 = the
    node index of each old node's mirror image (itself on the line)."""
    points = np.array(points, dtype=np.float64)
    triangles = np.asarray(triangles)
    extent = float((points.max(axis=0) - points.min(axis=0)).max())
    on_line = np.abs(points[:, axis] - plane) <= MERGE_TOLERANCE * extent
    points[on_line, axis] = plane
    off_line = np.flatnonzero(~on_line)
    image = np.empty(len(points), dtype=np.int64)
    image[on_line] = np.flatnonzero(on_line)
    image[off_line] = len(points) + np.arange(len(off_line))
    mirrored = points[off_line].copy()
    mirrored[:, axis] = 2.0 * plane - mirrored[:, axis]
    low_bit, high_bit = (FACE_BITS['x_lo'], FACE_BITS['x_hi']) if axis == 0 else (FACE_BITS['y_lo'], FACE_BITS['y_hi'])
    bits = np.asarray(pml_faces, dtype=np.int64)
    swapped = ((bits & ~(low_bit | high_bit)) | np.where(bits & low_bit, high_bit, 0)
               | np.where(bits & high_bit, low_bit, 0))
    image_triangles = image[triangles][:, ::-1].astype(triangles.dtype)
    return (np.vstack([points, mirrored]), np.vstack([triangles, image_triangles]),
            np.concatenate([bits, swapped]).astype(np.uint8), image)
