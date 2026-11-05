"""Triangle mesh data of the FEM path (solver_plan S7.5): analytic boundary curves, the mesher input (PSLG) and the
frozen TriangleMesh. Host numpy only; coordinates in m with the origin at the physical-domain corner."""
import hashlib
import math
from dataclasses import dataclass

import numpy as np

FACE_BITS = {'x_lo': 1, 'x_hi': 2, 'y_lo': 4, 'y_hi': 8}   # TriangleMesh.pml_faces bits of the PML layer ring
PERIODIC_TOLERANCE = 1e-9   # x the longest node-extent side: nodes this close to a side lie on it (periodic pairs)
FULL_TURN = 2.0 * math.pi   # Curve.span of a closed curve
_NEWTON_STEPS = 40                                         # ellipse nearest-point iterations (quadratic convergence)


@dataclass(frozen=True)
class Curve:
    """Analytic boundary curve: ellipse with semi-axes a, b about (cx, cy), semi-axis a turned by angle [rad] from +x;
    a circle has a == b. Parameter t [rad]: point (a cos t, b sin t) in the turned frame. An arc piece covers t0 .. t0 +
    span (counter-clockwise in t, 0 < span <= 2 pi); span 2 pi (default) = the closed curve."""
    cx: float
    cy: float
    a: float
    b: float
    angle: float = 0.0
    t0: float = 0.0
    span: float = FULL_TURN

    def __post_init__(self):
        if not 0.0 < self.span <= FULL_TURN:
            raise ValueError(f'Curve: arc span must lie in (0, 2 pi] rad, got {self.span!r}')

    @property
    def is_circle(self) -> bool:
        return self.a == self.b

    @property
    def is_closed(self) -> bool:
        return self.span >= FULL_TURN

    @property
    def whole(self) -> 'Curve':
        """The closed curve this arc lies on (itself when closed)."""
        return self if self.is_closed else Curve(self.cx, self.cy, self.a, self.b, self.angle)

    def arc_points(self, count) -> np.ndarray:
        """Float64 (count + 1, 2) points at count equal parameter steps over the arc (closed: last = first)."""
        points = self.points(self.t0 + np.linspace(0.0, self.span, int(count) + 1))
        if self.is_closed:
            points[-1] = points[0]
        return points

    def covers(self, t, slack=0.0) -> np.ndarray:
        """Bool per parameter t [rad] (any turn) on the arc, its ends widened by slack [rad]."""
        offset = np.mod(np.asarray(t, dtype=np.float64) - self.t0, FULL_TURN)
        return self.is_closed | (offset <= self.span + slack) | (offset >= FULL_TURN - slack)

    def _turn(self, local_u, local_v):
        cos_angle, sin_angle = math.cos(self.angle), math.sin(self.angle)
        return np.column_stack([local_u * cos_angle - local_v * sin_angle, local_u * sin_angle + local_v * cos_angle])

    def points(self, t) -> np.ndarray:
        """Float64 (n, 2) curve points at parameters t."""
        t = np.asarray(t, dtype=np.float64)
        return self._turn(self.a * np.cos(t), self.b * np.sin(t)) + np.array([self.cx, self.cy])

    def tangents(self, t) -> np.ndarray:
        """Float64 (n, 2) derivatives dP/dt at parameters t."""
        t = np.asarray(t, dtype=np.float64)
        return self._turn(-self.a * np.sin(t), self.b * np.cos(t))

    def curvature_radius(self, t) -> np.ndarray:
        """Radius of curvature [m] at parameters t (r for a circle)."""
        t = np.asarray(t, dtype=np.float64)
        if self.is_circle:
            return np.full(t.shape, float(self.a))
        speed_squared = (self.a * np.sin(t)) ** 2 + (self.b * np.cos(t)) ** 2
        return speed_squared ** 1.5 / (self.a * self.b)

    def local(self, points) -> tuple:
        """(u, v) of points in the curve's turned frame about its centre (not scaled)."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        offset_x, offset_y = points[:, 0] - self.cx, points[:, 1] - self.cy
        cos_angle, sin_angle = math.cos(self.angle), math.sin(self.angle)
        return offset_x * cos_angle + offset_y * sin_angle, offset_y * cos_angle - offset_x * sin_angle

    def parameters(self, points) -> np.ndarray:
        """Parameter t in [0, 2 pi) of the nearest curve point to each point (ellipse: Newton on the angle)."""
        local_u, local_v = self.local(points)
        t = np.arctan2(self.a * local_v, self.b * local_u)
        if not self.is_circle:
            a_squared_minus_b_squared = self.a ** 2 - self.b ** 2
            for _ in range(_NEWTON_STEPS):
                sin_t, cos_t = np.sin(t), np.cos(t)
                slope = (-a_squared_minus_b_squared * sin_t * cos_t + local_u * self.a * sin_t
                         - local_v * self.b * cos_t)
                curvature = (-a_squared_minus_b_squared * np.cos(2.0 * t) + local_u * self.a * cos_t
                             + local_v * self.b * sin_t)
                step = np.where(curvature > 0.0, slope / np.where(curvature > 0.0, curvature, 1.0), 0.0)
                t = t - np.clip(step, -0.5, 0.5)
        return np.mod(t, 2.0 * math.pi)

    def project(self, points) -> np.ndarray:
        """Float64 (n, 2) nearest curve points (circle radial, ellipse by Newton on the angle)."""
        return self.points(self.parameters(points))

    def key(self, tolerance) -> tuple:
        """Rounded identity of the whole curve for merging equal curves (arc range and a circle's angle ignored, an
        ellipse's taken modulo pi)."""
        angle = 0.0 if self.is_circle else math.fmod(self.angle, math.pi) % math.pi
        values = (self.cx, self.cy, self.a, self.b)
        return tuple(round(value / tolerance) for value in values) + (round(angle * 1e9),)


@dataclass(frozen=True, eq=False)
class PSLG:
    """Mesher input: vertices, constraint segments (chords of a curve carry its index), their tags and pieces."""
    vertices: np.ndarray        # float64 (n, 2) [m]
    segments: np.ndarray        # int32 (m, 2) vertex pairs
    segment_curve: np.ndarray   # int32 (m,) index into curves, -1 straight
    segment_tags: tuple         # (m,) tuples of names (outer faces, line entities, object names)
    segment_piece: np.ndarray   # int32 (m,) index of the intersection-free input piece the segment samples
    piece_ends: np.ndarray      # int32 (n_pieces, 2) end vertices of every piece (a closed curve: twice the same)
    curves: tuple               # closed Curves (arc inputs merged onto the whole curve they lie on)
    vertex_tags: dict           # point entity name -> vertex index
    window: tuple               # (x0, y0, x1, y1) [m] of the meshed rectangle
    small_angles: tuple         # ((x, y, angle_deg), ...) vertices where two segments meet below 60 deg
    acute_vertices: np.ndarray  # int32 indices of those vertices
    periodic_axes: tuple = ()   # 'x' / 'y' whose opposite window sides carry matching vertices (mesher.periodic_pslg)


@dataclass(frozen=True, eq=False)
class TriangleMesh:
    """Conforming triangle mesh of a 2D model: nodes, ccw elements, unique edges with signs, tags, curved edges."""
    points: np.ndarray          # float64 (n_points, 2) [m]; the PML ring lies outside the window
    triangles: np.ndarray       # int32 (n_elements, 3), counter-clockwise
    region: np.ndarray          # int32 (n_elements,) index into region_names of the last domain entity holding the
    region_names: tuple         #   element centroid (PML ring: the centroid clamped to the window), -1 none
    edges: np.ndarray           # int32 (n_edges, 2) node pairs, lower index first
    element_edges: np.ndarray   # int32 (n_elements, 3): local edge k = nodes (k + 1, k + 2) mod 3, opposite node k
    edge_signs: np.ndarray      # int8 (n_elements, 3): +1 when local edge k runs from the lower to the higher index
    edge_elements: np.ndarray   # int32 (n_edges, 2) adjacent elements, the second -1 on the outer boundary
    edge_curve: np.ndarray      # int32 (n_edges,) curve index of an edge on a curved boundary, -1 straight
    curves: tuple               # Curve
    midside_points: np.ndarray  # float64 (n_edges, 2) P2 midside nodes, on the exact curve for curved edges
    edge_tags: dict             # name -> int32 sorted edge indices (outer faces, object outlines, line entities)
    point_tags: dict            # point entity name -> node index
    pml_faces: np.ndarray       # uint8 (n_elements,) FACE_BITS of the PML layer ring holding the element, 0 inside
    window: tuple               # (x0, y0, x1, y1) [m] of the physical window
    quality: object             # quality.MeshQuality
    digest: str                 # sha256 of points and triangles: the same model gives the same mesh
    periodic_axes: tuple = ()   # 'x' / 'y' meshed with matching opposite window sides (periodic_node_partners)

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_elements(self) -> int:
        return len(self.triangles)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def centroids(self) -> np.ndarray:
        return self.points[self.triangles].mean(axis=1)

    def areas(self) -> np.ndarray:
        return signed_areas(self.points, self.triangles)

    def boundary_edges(self) -> np.ndarray:
        """Int edge indices on the outer boundary."""
        return np.flatnonzero(self.edge_elements[:, 1] < 0)


def signed_areas(points, triangles) -> np.ndarray:
    """Float64 signed areas (> 0 counter-clockwise) of the triangles."""
    corner_a, corner_b, corner_c = (points[triangles[:, k]] for k in range(3))
    return 0.5 * ((corner_b[:, 0] - corner_a[:, 0]) * (corner_c[:, 1] - corner_a[:, 1])
                  - (corner_b[:, 1] - corner_a[:, 1]) * (corner_c[:, 0] - corner_a[:, 0]))


def side_members(pairs, member) -> tuple:
    """(first, second) bool of an element mask on the two sides of entities (pairs (k, 2): edge_elements /
    face_elements rows, -1 = no element there, which counts as non-member)."""
    member = np.asarray(member, dtype=bool)
    pairs = np.asarray(pairs)
    return member[pairs[:, 0]], np.where(pairs[:, 1] >= 0, member[np.maximum(pairs[:, 1], 0)], False)


def one_sided(pairs, member) -> np.ndarray:
    """Bool per entity: exactly one side is a member element (side_members)."""
    first, second = side_members(pairs, member)
    return first != second


def edge_keys(pairs, n_points) -> np.ndarray:
    """Int64 key lower * n_points + higher of node pairs (any order)."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return np.minimum(pairs[:, 0], pairs[:, 1]) * n_points + np.maximum(pairs[:, 0], pairs[:, 1])


def connectivity(triangles, n_points) -> tuple:
    """(edges, element_edges, edge_signs, edge_elements) of a triangle list; ValueError on an edge of > 2 elements."""
    n_elements = len(triangles)
    local_starts = np.concatenate([triangles[:, (k + 1) % 3] for k in range(3)]).astype(np.int64)
    local_ends = np.concatenate([triangles[:, (k + 2) % 3] for k in range(3)]).astype(np.int64)
    keys = np.minimum(local_starts, local_ends) * n_points + np.maximum(local_starts, local_ends)
    unique_keys, first_index, inverse, counts = np.unique(keys, return_index=True, return_inverse=True,
                                                          return_counts=True)
    if counts.max(initial=0) > 2:
        raise ValueError(f'triangle mesh: {int((counts > 2).sum())} edges are shared by more than two elements')
    edges = np.column_stack([unique_keys // n_points, unique_keys % n_points]).astype(np.int32)
    element_edges = inverse.reshape(3, n_elements).T.astype(np.int32)
    edge_signs = np.where(local_starts < local_ends, 1, -1).reshape(3, n_elements).T.astype(np.int8)
    last_index = len(keys) - 1 - np.unique(keys[::-1], return_index=True)[1]
    edge_elements = np.full((len(edges), 2), -1, dtype=np.int32)
    edge_elements[:, 0] = first_index % n_elements
    shared = counts == 2
    edge_elements[shared, 1] = last_index[shared] % n_elements
    return edges, element_edges, edge_signs, edge_elements


def edge_indices(edges, n_points, pairs) -> np.ndarray:
    """Int32 indices into edges (sorted keys, as from connectivity) of node pairs; ValueError for a pair not in it."""
    keys = edge_keys(edges, n_points)
    wanted = edge_keys(pairs, n_points)
    found = np.searchsorted(keys, wanted)
    found = np.minimum(found, max(len(keys) - 1, 0))
    if len(wanted) and (not len(keys) or not np.array_equal(keys[found], wanted)):
        raise ValueError('triangle mesh: a constraint segment is not an edge of the mesh')
    return found.astype(np.int32)


def build_triangle_mesh(points, triangles, *, curves=(), curved_pairs=None, curved_ids=None, tagged_pairs=None,
                        outline_masks=None, point_tags=None, region=None, region_names=(), pml_faces=None, window,
                        quality=None, periodic_axes=()) -> TriangleMesh:
    """TriangleMesh from nodes and elements (orientation fixed to ccw): curved_pairs (k, 2) node pairs on curve
    curved_ids. edge_tags: the outer box sides as FACE_BITS names, tagged_pairs {name: (k, 2) node pairs}, and per
    outline_masks {name: bool per element} the edges where membership changes (outer edges of member elements)."""
    periodic_axes = tuple(periodic_axes)
    points = np.ascontiguousarray(points, dtype=np.float64)
    triangles = np.array(triangles, dtype=np.int32)
    areas = signed_areas(points, triangles)
    if not (np.abs(areas) > 0.0).all():
        raise ValueError(f'triangle mesh: {int((areas == 0.0).sum())} elements have zero area')
    clockwise = areas < 0.0
    triangles[clockwise] = triangles[clockwise][:, [0, 2, 1]]
    n_points = len(points)
    edges, element_edges, edge_signs, edge_elements = connectivity(triangles, n_points)
    edge_curve = np.full(len(edges), -1, dtype=np.int32)
    midside = 0.5 * (points[edges[:, 0]] + points[edges[:, 1]])
    if curved_pairs is not None and len(curved_pairs):
        curved_edges = edge_indices(edges, n_points, curved_pairs)
        edge_curve[curved_edges] = np.asarray(curved_ids, dtype=np.int32)
        for curve_index, curve in enumerate(curves):
            on_curve = np.flatnonzero(edge_curve == curve_index)
            if len(on_curve):
                midside[on_curve] = curve.project(midside[on_curve])
    tags = _outer_face_tags(points, edges, edge_elements)
    for name, pairs in (tagged_pairs or {}).items():
        indices = edge_indices(edges, n_points, pairs)
        tags[name] = np.union1d(tags.get(name, np.zeros(0, dtype=np.int32)), indices).astype(np.int32)
    outer = edge_elements[:, 1] < 0
    for name, member in (outline_masks or {}).items():
        member = np.asarray(member, dtype=bool)
        first = member[edge_elements[:, 0]]
        second = np.where(outer, False, member[np.maximum(edge_elements[:, 1], 0)])
        changes = np.flatnonzero(first != second)
        tags[name] = np.union1d(tags.get(name, np.zeros(0, dtype=np.int32)), changes).astype(np.int32)
    n_elements = len(triangles)
    region = np.full(n_elements, -1, dtype=np.int32) if region is None else np.asarray(region, dtype=np.int32)
    pml_faces = (np.zeros(n_elements, dtype=np.uint8) if pml_faces is None
                 else np.asarray(pml_faces, dtype=np.uint8))
    mesh = TriangleMesh(points=points, triangles=triangles, region=region, region_names=tuple(region_names),
                        edges=edges, element_edges=element_edges, edge_signs=edge_signs, edge_elements=edge_elements,
                        edge_curve=edge_curve, curves=tuple(curves), midside_points=midside, edge_tags=tags,
                        point_tags=dict(point_tags or {}), pml_faces=pml_faces, window=tuple(window),
                        quality=quality, digest=mesh_digest(points, triangles), periodic_axes=periodic_axes)
    for name in periodic_axes:   # a mesh declared periodic must be: fail at build, not in the solver
        periodic_node_partners(mesh, 'xy'.index(name))
    return mesh


def periodic_node_partners(mesh, axis) -> np.ndarray:
    """Int64 (n_points,): for each node on the hi side of the node extent along axis (0 x / 1 y) the node on the lo
    side at the same other coordinate, -1 elsewhere; ValueError when the two sides' nodes do not match."""
    points = mesh.points
    low, high = points.min(axis=0), points.max(axis=0)
    slack = PERIODIC_TOLERANCE * float((high - low).max())
    lo_nodes = np.flatnonzero(np.abs(points[:, axis] - low[axis]) <= slack)
    hi_nodes = np.flatnonzero(np.abs(points[:, axis] - high[axis]) <= slack)
    name, other = 'xy'[axis], 1 - axis
    if len(lo_nodes) != len(hi_nodes) or len(lo_nodes) == 0:
        raise ValueError(f'periodic {name}: {len(lo_nodes)} nodes on the {name}_lo side, {len(hi_nodes)} on '
                         f'{name}_hi; the mesh is not periodic (mesh it with periodic_axes, without an absorbing ring '
                         'on that axis)')
    lo_nodes = lo_nodes[np.argsort(points[lo_nodes, other], kind='stable')]
    hi_nodes = hi_nodes[np.argsort(points[hi_nodes, other], kind='stable')]
    mismatch = np.abs(points[lo_nodes, other] - points[hi_nodes, other]) > slack
    if mismatch.any():
        raise ValueError(f'periodic {name}: {int(mismatch.sum())} {name}_hi nodes have no {name}_lo node at the same '
                         'position; the mesh is not periodic')
    partners = np.full(len(points), -1, dtype=np.int64)
    partners[hi_nodes] = lo_nodes
    return partners


def _outer_face_tags(points, edges, edge_elements) -> dict:
    """{face: int32 edge indices} of the outer-boundary edges on each side of the nodes' bounding box."""
    low, high = points.min(axis=0), points.max(axis=0)
    tolerance = 1e-9 * float((high - low).max())
    outer = np.flatnonzero(edge_elements[:, 1] < 0)
    ends = points[edges[outer]]   # (k, 2 nodes, 2 coordinates)
    tags = {}
    for face in FACE_BITS:
        axis = 0 if face[0] == 'x' else 1
        value = low[axis] if face.endswith('lo') else high[axis]
        on_side = np.all(np.abs(ends[:, :, axis] - value) <= tolerance, axis=1)
        tags[face] = outer[on_side].astype(np.int32)
    return tags


def mesh_digest(points, triangles) -> str:
    """Sha256 hex of the node coordinates and element nodes."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(points, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(triangles, dtype=np.int32).tobytes())
    return digest.hexdigest()
