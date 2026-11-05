"""Tetrahedron mesh data (igor3d_plan section 7, T4): the frozen TetrahedronMesh with ascending element nodes, unique
edges / faces and their element maps, outer-face and object-interface tags, region tags and quality; window / interface
face selections; periodic node partners; the structured box mesh (Kuhn split of a tensor grid, conforming and
gmsh-free: the CPU self-tests).
Host numpy; coordinates in m."""
import hashlib
import itertools
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from .elements3d import LOCAL_EDGES, LOCAL_FACES
from .mesh import one_sided

OUTER_FACES_3D = ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')
PLANE_TOLERANCE = 1e-9   # x the longest bounding-box side: nodes this close to a box side lie on it
POOR_RADIUS_RATIO = 0.1   # first value (igor3d_plan 7.2 T4): tune from the T5 measurements
OBTUSE_DIHEDRAL_COS = -1e-6   # cos of a dihedral angle below this = obtuse (orthoschemes: exactly 90 deg at most)
DEGENERATE_VOLUME_RTOL = 1e-13   # 6 |V| / longest edge^3 at or below which an element counts as flat (refused)
ENTITY_MAJORITY = 0.9     # share of a mesher volume's elements whose common tag / membership it takes throughout
_KUHN_PATHS = tuple(itertools.permutations(range(3)))   # 6 tetrahedra per cube: monotone corner paths 000 -> 111


@dataclass(frozen=True, eq=False)
class TetrahedronQuality:
    """Radius ratio 3 r_in / r_circ per element (1 regular, -> 0 sliver): minimum, mean and n_poor = elements below
    POOR_RADIUS_RATIO with an obtuse dihedral angle (slivers / caps; graded Kuhn orthoschemes, all <= 90 deg, are not)."""
    radius_ratio: np.ndarray
    smallest: float
    mean: float
    n_poor: int


@dataclass(frozen=True, eq=False)
class TetrahedronMesh:
    """Conforming tetrahedron mesh: nodes, elements with ascending node indices, unique edges / faces, tags."""
    points: np.ndarray          # float64 (n_points, 3) [m]; a PML shell lies outside the window
    tets: np.ndarray            # int32 (n_elements, 4), ascending per row (fem.elements3d orientation)
    region: np.ndarray          # int32 (n_elements,) index into region_names (last solid holding the centroid
    region_names: tuple         #   clamped into the window), -1 none
    edges: np.ndarray           # int32 (n_edges, 2), lower node first
    element_edges: np.ndarray   # int32 (n_elements, 6) in elements3d.LOCAL_EDGES order
    faces: np.ndarray           # int32 (n_faces, 3), ascending
    element_faces: np.ndarray   # int32 (n_elements, 4) in elements3d.LOCAL_FACES order
    face_elements: np.ndarray   # int32 (n_faces, 2), the second -1 on the outer boundary
    face_tags: dict             # outer face name -> int32 sorted boundary faces on that side of the bounding box
    window: tuple               # (x0, y0, z0, x1, y1, z1) [m] of the physical window (PML shell outside)
    quality: TetrahedronQuality
    digest: str
    entities: np.ndarray | None = None   # int64 (n_elements,) mesher volume (gmsh fragment) per element; None: one
    # domain object name -> int32 sorted interface faces (membership changes across them, outer boundary faces of
    # member elements included; model.tetrahedral_meshing); None: no objects tagged
    object_faces: dict | None = None

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.tets.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bounds(self) -> tuple:
        """(x0, y0, z0, x1, y1, z1) of the nodes (the window plus any PML shell)."""
        return tuple(float(value) for value in (*self.points.min(axis=0), *self.points.max(axis=0)))

    def centroids(self, elements=None) -> np.ndarray:
        """(m, 3) element centroids (all elements by default; a copy of the cached array, so callers may write)."""
        if elements is None:
            return self._centroids.copy()
        return self.points[self.tets[elements]].mean(axis=1)

    @cached_property
    def _centroids(self) -> np.ndarray:
        """Read-only (n_elements, 3) centroids of every element (centroids() hands out copies)."""
        centroids = self.points[self.tets].mean(axis=1)
        centroids.setflags(write=False)
        return centroids

    def volumes(self, elements=None) -> np.ndarray:
        """(m,) element volumes [m^3]."""
        tets = self.tets if elements is None else self.tets[elements]
        corners = self.points[tets]
        return np.abs(span_adjugate(corners[:, 1:, :] - corners[:, :1, :])[0]) / 6.0

    def in_window(self, points, tolerance=None) -> np.ndarray:
        """Bool per point (n, 3): inside the closed window (tolerance default PLANE_TOLERANCE x the longest side)."""
        points = np.asarray(points, dtype=np.float64)
        low, high = np.array(self.window[:3]), np.array(self.window[3:])
        slack = PLANE_TOLERANCE * float((high - low).max()) if tolerance is None else float(tolerance)
        return ((points >= low - slack) & (points <= high + slack)).all(axis=-1)

    def clamped_into_window(self, points) -> np.ndarray:
        """(n, 3) points clamped into the window (the PML rule: materials extruded into the shell)."""
        return np.clip(np.asarray(points, dtype=np.float64), self.window[:3], self.window[3:])

    def inset_centroids(self) -> np.ndarray:
        """(n_elements, 3) centroids clamped just inside the window (PLANE_TOLERANCE x the longest side): the point of
        every element membership test (builder masks, object_faces, TetrahedronDomain.host_centroids); cached,
        read-only."""
        return self._inset_centroids

    @cached_property
    def _inset_centroids(self) -> np.ndarray:
        low, high = np.array(self.window[:3]), np.array(self.window[3:])
        inset = PLANE_TOLERANCE * float((high - low).max())
        centroids = np.clip(self.centroids(), low + inset, high - inset)
        centroids.setflags(write=False)
        return centroids

    @cached_property
    def _entity_groups(self) -> tuple:
        """(entity index per element (m,), elements per entity) of the mesher volumes (entities not None)."""
        _, index, sizes = np.unique(self.entities, return_inverse=True, return_counts=True)
        return index.reshape(-1), sizes

    def window_elements(self) -> np.ndarray:
        """Bool per element: centroid inside the closed window (False in a PML shell); cached, read-only."""
        return self._window_elements

    @cached_property
    def _window_elements(self) -> np.ndarray:
        inside = self.in_window(self._centroids)
        inside.setflags(write=False)
        return inside

    def by_entity_majority(self, values) -> np.ndarray:
        """Per-element tags / bool memberships with every mesher volume whose values agree on >= ENTITY_MAJORITY of
        its elements set to that value throughout (centroid tests misread elements next to faceted curved
        interfaces); unchanged without entities."""
        return values if self.entities is None else entity_majority(values, self.entities, self._entity_groups)


def sliver_warnings(mesh) -> list:
    """[] or the one finalize warning of a solver on a tetrahedron mesh with slivers (quality.n_poor > 0)."""
    quality = mesh.quality
    if not quality.n_poor:
        return []
    return [f'{quality.n_poor:,} of {mesh.n_elements:,} tetrahedra have a radius ratio below {POOR_RADIUS_RATIO:g} '
            f'and an obtuse dihedral angle (smallest ratio {quality.smallest:.3g}): such slivers worsen the '
            'conditioning; check thin gaps in the geometry or set h_min']


def entity_majority(values, entities, groups=None) -> np.ndarray:
    """Copy of per-element integer / bool values: each entity (entities (m,)) whose most common value covers >=
    ENTITY_MAJORITY of its elements takes it throughout; groups = TetrahedronMesh._entity_groups of entities when cached."""
    values = np.asarray(values)
    result = values.copy()
    entity_index, sizes = groups if groups is not None else np.unique(entities, return_inverse=True,
                                                                      return_counts=True)[1:]
    entity_index = np.asarray(entity_index).reshape(-1)
    if values.dtype == np.bool_:   # memberships: no sort
        kinds, value_index = np.array([False, True]), values.astype(np.intp)
    else:
        kinds, value_index = np.unique(values, return_inverse=True)
    n_entities, n_kinds = len(sizes), len(kinds)
    # (entity, value) counts; argmax = the most common value, the smallest on a tie (never one reaching the majority)
    table = np.bincount(entity_index * n_kinds + value_index.reshape(-1),
                        minlength=n_entities * n_kinds).reshape(n_entities, n_kinds)
    best_kind = table.argmax(axis=1)
    agrees = table[np.arange(n_entities), best_kind] >= ENTITY_MAJORITY * sizes
    take = agrees[entity_index]
    result[take] = kinds[best_kind[entity_index[take]]]
    return result


def interface_faces(mesh, member) -> np.ndarray:
    """Sorted int64 faces with exactly one member element (member: bool per element; the outer side of a boundary face
    counts as non-member): an object's interface, the 2D object-outline edge tags' analogue."""
    return np.flatnonzero(one_sided(mesh.face_elements, member)).astype(np.int64)


def window_faces(mesh, face) -> np.ndarray:
    """Sorted int64 faces of the window on an outer face ('x_lo' ... 'z_hi'): interface faces of the window elements
    with all three nodes on that window plane (the PML shell's inner side when there is one)."""
    axis = 'xyz'.index(face[0])
    plane = mesh.window[axis + (3 if face.endswith('hi') else 0)]
    slack = PLANE_TOLERANCE * float(np.max(np.array(mesh.window[3:]) - np.array(mesh.window[:3])))
    candidates = interface_faces(mesh, mesh.window_elements())
    on_plane = (np.abs(mesh.points[mesh.faces[candidates]][:, :, axis] - plane) <= slack).all(axis=1)
    return candidates[on_plane]


def periodic_node_partners(mesh, axis) -> np.ndarray:
    """int64 (n_points,): for each node on the hi side of the node extent along axis (0 / 1 / 2) the node on the lo
    side at the same other two coordinates, -1 elsewhere; ValueError when the two sides' nodes do not match."""
    from scipy.spatial import cKDTree
    low, high = mesh.points.min(axis=0), mesh.points.max(axis=0)
    slack = PLANE_TOLERANCE * float((high - low).max())
    lo_nodes = np.flatnonzero(np.abs(mesh.points[:, axis] - low[axis]) <= slack)
    hi_nodes = np.flatnonzero(np.abs(mesh.points[:, axis] - high[axis]) <= slack)
    others = [other for other in range(3) if other != axis]
    name = 'xyz'[axis]
    if len(lo_nodes) != len(hi_nodes) or len(lo_nodes) == 0:
        raise ValueError(f'periodic {name}: {len(lo_nodes)} nodes on the {name}_lo side, {len(hi_nodes)} on {name}_hi; '
                         'the mesh is not periodic (periodic faces need matching sides, e.g. the builtin Kuhn split)')
    distance, nearest = cKDTree(mesh.points[lo_nodes][:, others]).query(mesh.points[hi_nodes][:, others])
    if (distance > slack).any() or len(np.unique(nearest)) != len(hi_nodes):
        raise ValueError(f'periodic {name}: {int((distance > slack).sum())} {name}_hi nodes have no {name}_lo node at '
                         'the same position; the mesh is not periodic')
    partners = np.full(mesh.n_points, -1, dtype=np.int64)
    partners[hi_nodes] = lo_nodes[nearest]
    return partners


def _unique_rows(rows):
    """(unique rows, inverse (len(rows),)) of a non-negative int64 (k, d) array, rows in lexicographic order (as
    np.unique(axis=0)): one int64 key per row (base max + 1) and a 1D unique while base^d fits int64, else axis=0."""
    rows = np.asarray(rows, dtype=np.int64)
    base = int(rows.max()) + 1 if rows.size else 1
    if rows.size and int(rows.min()) >= 0 and base ** rows.shape[1] < 2 ** 63:
        keys = rows[:, 0].copy()
        for column in range(1, rows.shape[1]):
            keys *= base
            keys += rows[:, column]
        unique_keys, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        return rows[first], np.asarray(inverse).reshape(-1)
    unique, inverse = np.unique(rows, axis=0, return_inverse=True)
    return unique, np.asarray(inverse).reshape(-1)


def span_adjugate(spans) -> tuple:
    """(det (m,), adjugate (m, 3, 3)) of (m, 3, 3) row stacks (rows a, b, c) by cross products, no per-matrix LAPACK:
    the adjugate columns are b x c, c x a, a x b, so spans @ adjugate = det I and the inverse is adjugate / det."""
    a, b, c = spans[:, 0], spans[:, 1], spans[:, 2]
    bc, ca, ab = np.cross(b, c), np.cross(c, a), np.cross(a, b)
    return np.einsum('mi,mi->m', a, bc), np.stack([bc, ca, ab], axis=2)


def tetrahedron_quality(points, tets) -> TetrahedronQuality:
    """Radius ratios 3 r_in / r_circ (r_in = 3 V / face area sum, circumcentre from the 3 x 3 bisector system)."""
    corners = points[tets]
    edges = corners[:, 1:, :] - corners[:, :1, :]   # (m, 3, 3) rows x_k - x_0
    determinant, adjugate = span_adjugate(edges)
    volume = np.abs(determinant) / 6.0
    area = np.zeros(len(tets))
    for a, b, c in LOCAL_FACES:
        area += 0.5 * np.linalg.norm(np.cross(corners[:, b] - corners[:, a], corners[:, c] - corners[:, a]), axis=1)
    inradius = 3.0 * volume / area
    # circumcentre relative to x_0: 2 E c = |e_k|^2, c = adj |e|^2 / (2 det)
    centre = np.einsum('mij,mj->mi', adjugate, (edges ** 2).sum(axis=2)) / (2.0 * determinant)[:, None]
    circumradius = np.linalg.norm(centre, axis=1)
    ratio = 3.0 * inradius / circumradius
    low = np.flatnonzero(ratio < POOR_RADIUS_RATIO)
    n_poor = int(np.count_nonzero(largest_dihedral_cosines(corners[low]) < OBTUSE_DIHEDRAL_COS)) if len(low) else 0
    return TetrahedronQuality(radius_ratio=ratio, smallest=float(ratio.min(initial=1.0)),
                              mean=float(ratio.mean()) if len(ratio) else 1.0, n_poor=n_poor)


def largest_dihedral_cosines(corners) -> np.ndarray:
    """cos of the largest dihedral angle per element of corners (m, 4, 3) [m]: the smallest -cos between the barycentric
    gradients of a vertex pair (< 0: obtuse; a Kuhn orthoscheme reaches 0 exactly, a sliver about -1)."""
    spans = corners[:, 1:, :] - corners[:, :1, :]                     # rows x_k - x_0
    determinant, adjugate = span_adjugate(spans)
    gradients = adjugate / determinant[:, None, None]                  # column k = grad lambda_k (k = 1..3)
    grads = np.concatenate([-gradients.sum(axis=2, keepdims=True), gradients], axis=2).transpose(0, 2, 1)
    unit = grads / np.linalg.norm(grads, axis=2, keepdims=True)
    cosines = [-(unit[:, a] * unit[:, b]).sum(axis=1) for a in range(4) for b in range(a + 1, 4)]
    return np.min(cosines, axis=0) if len(corners) else np.zeros(0)


def build_tetrahedron_mesh(points, tets, window=None, region=None, region_names=(), entities=None) -> TetrahedronMesh:
    """TetrahedronMesh of nodes (n, 3) [m] and elements (m, 4): element nodes sorted ascending, edges / faces unique,
    outer faces tagged by the bounding-box side their three nodes lie on; window None = the bounding box; region (m,)
    indices into region_names (None: -1); entities (m,) the mesher volume per element (None: one)."""
    points = np.ascontiguousarray(points, dtype=np.float64)
    tets = np.sort(np.asarray(tets, dtype=np.int64), axis=1)
    n_elements = len(tets)
    if tets.ndim != 2 or tets.shape[1] != 4 or n_elements == 0:
        raise ValueError(f'build_tetrahedron_mesh needs (m, 4) elements, got shape {tets.shape}')
    if (tets[:, 1:] == tets[:, :-1]).any():
        raise ValueError('build_tetrahedron_mesh: an element repeats a node')
    spans = points[tets[:, 1:]] - points[tets[:, :1]]   # (m, 3, 3); tetrahedron_quality's solve fails on flat ones
    longest = np.linalg.norm(spans, axis=2).max(axis=1)
    flat = np.abs(span_adjugate(spans)[0]) <= DEGENERATE_VOLUME_RTOL * longest ** 3
    if flat.any():
        raise ValueError(f'build_tetrahedron_mesh: {int(flat.sum())} element(s) have (near) zero volume, e.g. element '
                         f'{int(np.flatnonzero(flat)[0])}')
    del spans, longest, flat
    edge_rows = np.concatenate([tets[:, [a, b]] for a, b in LOCAL_EDGES])
    edges, edge_inverse = _unique_rows(edge_rows)
    element_edges = edge_inverse.reshape(len(LOCAL_EDGES), n_elements).T
    face_rows = np.concatenate([tets[:, list(face)] for face in LOCAL_FACES])
    faces, face_inverse = _unique_rows(face_rows)
    element_faces = face_inverse.reshape(len(LOCAL_FACES), n_elements).T
    counts = np.bincount(face_inverse, minlength=len(faces))
    if (counts > 2).any():
        raise ValueError(f'build_tetrahedron_mesh: {int((counts > 2).sum())} faces belong to more than two elements')
    face_elements = np.full((len(faces), 2), -1, dtype=np.int64)
    owner = np.tile(np.arange(n_elements), len(LOCAL_FACES))
    order = np.argsort(face_inverse, kind='stable')
    sorted_faces = face_inverse[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = sorted_faces[1:] != sorted_faces[:-1]
    face_elements[sorted_faces[first], 0] = owner[order[first]]
    face_elements[sorted_faces[~first], 1] = owner[order[~first]]
    low, high = points.min(axis=0), points.max(axis=0)
    slack = PLANE_TOLERANCE * float((high - low).max())
    boundary = np.flatnonzero(face_elements[:, 1] < 0)
    face_points = points[faces[boundary]]   # (k, 3, 3)
    face_tags = {}
    for axis in range(3):
        for side, value in (('lo', low[axis]), ('hi', high[axis])):
            on_side = (np.abs(face_points[:, :, axis] - value) <= slack).all(axis=1)
            face_tags[f"{'xyz'[axis]}_{side}"] = boundary[on_side].astype(np.int32)
    window = tuple(float(value) for value in ((*low, *high) if window is None else window))
    region = (np.full(n_elements, -1, dtype=np.int32) if region is None
              else np.asarray(region, dtype=np.int32).reshape(n_elements))
    digest = hashlib.sha256(points.tobytes() + tets.astype(np.int64).tobytes()).hexdigest()
    return TetrahedronMesh(points=points, tets=tets.astype(np.int32), region=region, region_names=tuple(region_names),
                           edges=edges.astype(np.int32), element_edges=element_edges.astype(np.int32),
                           faces=faces.astype(np.int32), element_faces=element_faces.astype(np.int32),
                           face_elements=face_elements.astype(np.int32), face_tags=face_tags, window=window,
                           quality=tetrahedron_quality(points, tets), digest=digest,
                           entities=None if entities is None else np.asarray(entities, dtype=np.int64).reshape(-1))


def region_tags(points, solids) -> np.ndarray:
    """int32 index of the last solid (objects with contains(points) -> bool) holding each point, -1 none."""
    region = np.full(len(points), -1, dtype=np.int32)
    for index, solid in enumerate(solids):
        region[np.asarray(solid.contains(points), dtype=bool)] = index
    return region


def structured_box_mesh(x_edges, y_edges, z_edges, window=None, solids=()) -> TetrahedronMesh:
    """Kuhn split (6 tetrahedra per cell along the monotone paths 000 -> 111, conforming between cells) of the tensor
    grid of these increasing edge coordinates [m]; window None = the grid box (cells outside a given window form a PML
    shell); regions by `solids` (objects with name / contains) at centroids clamped into the window."""
    axes = [np.asarray(edges, dtype=np.float64) for edges in (x_edges, y_edges, z_edges)]
    if any(edges.ndim != 1 or len(edges) < 2 or not (np.diff(edges) > 0.0).all() for edges in axes):
        raise ValueError('structured_box_mesh needs increasing 1D edge coordinates with >= 2 entries per axis')
    sizes = [len(edges) for edges in axes]
    grid = np.meshgrid(*axes, indexing='ij')
    points = np.column_stack([coordinate.ravel() for coordinate in grid])
    cells = np.stack(np.meshgrid(*(np.arange(size - 1) for size in sizes), indexing='ij'), axis=-1).reshape(-1, 3)

    def node(offset):
        corner = cells + np.asarray(offset)
        return (corner[:, 0] * sizes[1] + corner[:, 1]) * sizes[2] + corner[:, 2]

    tets = []
    for path in _KUHN_PATHS:
        offset = [0, 0, 0]
        nodes = [node(offset)]
        for axis in path:
            offset[axis] = 1
            nodes.append(node(offset))
        tets.append(np.stack(nodes, axis=1))
    tets = np.concatenate(tets)
    mesh_window = tuple(float(value) for value in ((*(edges[0] for edges in axes), *(edges[-1] for edges in axes))
                                                   if window is None else window))
    region = None
    if solids:
        centroids = np.clip(points[tets].mean(axis=1), mesh_window[:3], mesh_window[3:])
        region = region_tags(centroids, solids)
    return build_tetrahedron_mesh(points, tets, window=mesh_window, region=region,
                                  region_names=tuple(solid.name for solid in solids))
