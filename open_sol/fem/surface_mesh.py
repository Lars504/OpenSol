"""Planar triangle meshes of flat tetrahedron boundary faces (new_modules_plan S6): surface_triangle_mesh builds a
TriangleMesh with node / edge / face maps back into the TetrahedronMesh from faces on one coordinate plane,
normal_displacement transfers a P1 / P2 displacement onto it, lagrange_field_function evaluates such a field at
quadrature points and boundary_edges_where selects rim edges. Host numpy."""
# Coordinates: the two in-plane axes of the tetrahedron mesh in cyclic order (normal z: (x, y); x: (y, z); y: (z, x)),
# not shifted, so model-frame expressions (r = hypot(x, y)) apply unchanged; window = the node bounding box, no PML ring,
# no curved edges. Surface nodes are the face nodes in ascending tetrahedron numbering, so a surface edge's lower node is
# its tet edge's lower node. Surface element k is face `faces[k]`. The surface full Lagrange numbering (nodes, then
# n_points + edge) maps dof by dof onto the tet numbering (nodes, then n_points + edge: `tet_dofs`); both sides are
# affine, so the trace of a tet P1 / P2 field is exactly the surface P1 / P2 field (no interpolation).
# Normal: n_out = outward_sign e_axis points out of the solid element beside each face (one sign for every face: one
# gap); w = -u . n_out is the gap-opening displacement of a gap on the outward side (physics/mechanics/squeeze_film).
from dataclasses import dataclass

import numpy as np

from .elements import _inverse_2x2, lagrange_values
from .mesh import build_triangle_mesh, side_members
from .mesh3d import PLANE_TOLERANCE
from .quality import mesh_quality

IN_PLANE_AXES = {0: (1, 2), 1: (2, 0), 2: (0, 1)}   # surface (x, y) = these tet axes (cyclic: right-handed with axis)
VECTOR_COMPONENTS = 3                               # interleaved tet displacement dofs 3 d + component


@dataclass(frozen=True, eq=False)
class SurfaceMesh:
    """TriangleMesh of selected flat tetrahedron faces with its maps into the TetrahedronMesh (module header)."""
    mesh: object                # fem.mesh.TriangleMesh in the in-plane coordinates [m]
    tet_mesh: object            # the source fem.mesh3d.TetrahedronMesh
    axis: int                   # normal axis 0 / 1 / 2
    plane: float                # coordinate of the faces along axis [m]
    outward_sign: float         # +1 / -1: the solid's outward normal = outward_sign e_axis
    nodes: np.ndarray           # int64 (n_points,) tet node of each surface node (ascending)
    edges: np.ndarray           # int64 (n_edges,) tet edge of each surface edge
    faces: np.ndarray           # int64 (n_elements,) tet face of each surface element
    solid_elements: np.ndarray  # int64 (n_elements,) tet element beside each face on the solid side

    @property
    def in_plane_axes(self) -> tuple:
        return IN_PLANE_AXES[self.axis]

    def tet_dofs(self, order) -> np.ndarray:
        """Int64 tet full-numbering scalar dof of each surface full-numbering Lagrange dof (P1: nodes; P2: + edges)."""
        if order == 1:
            return self.nodes.copy()
        if order == 2:
            return np.concatenate([self.nodes, self.tet_mesh.n_points + self.edges])
        raise ValueError(f'SurfaceMesh.tet_dofs: order must be 1 or 2, got {order!r}')


def surface_triangle_mesh(tet_mesh, faces, solid_elements=None, *, axis=None) -> SurfaceMesh:
    """SurfaceMesh of tet faces on one plane axis = const (axis None: from the face normals); the solid side of each face
    is its element (outer faces) or the solid_elements (bool per element) one; ValueError for curved / mixed selections."""
    faces = np.unique(np.asarray(faces, dtype=np.int64).ravel())
    if not faces.size:
        raise ValueError('surface_triangle_mesh: no faces selected')
    if faces[0] < 0 or faces[-1] >= tet_mesh.n_faces:
        raise ValueError(f'surface_triangle_mesh: face indices must lie in [0, {tet_mesh.n_faces})')
    face_nodes = tet_mesh.faces[faces].astype(np.int64)   # (k, 3)
    corners = tet_mesh.points[face_nodes]                  # (k, 3, 3)
    if axis is None:
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        axis = int(np.argmax(np.abs(normals).sum(axis=0)))
    elif axis not in IN_PLANE_AXES:
        raise ValueError(f'surface_triangle_mesh: axis must be 0, 1 or 2, got {axis!r}')
    name = 'xyz'[axis]
    bounds = np.array(tet_mesh.bounds)
    slack = PLANE_TOLERANCE * float((bounds[3:] - bounds[:3]).max())
    plane = float(np.mean(corners[:, :, axis]))
    offset = float(np.abs(corners[:, :, axis] - plane).max())
    if offset > slack:
        raise ValueError(f'surface_triangle_mesh: the faces do not lie on one {name} = const plane (largest offset '
                         f'{offset:.3g} m, tolerance {slack:.3g} m); select the flat faces of one side')
    pairs = tet_mesh.face_elements[faces].astype(np.int64)
    if solid_elements is None:
        if (pairs[:, 1] >= 0).any():
            raise ValueError(f'surface_triangle_mesh: {int((pairs[:, 1] >= 0).sum())} interior faces selected; pass '
                             'solid_elements (bool per element) to tell the solid side')
        owners = pairs[:, 0]
    else:
        solid = np.asarray(solid_elements, dtype=bool)
        if solid.shape != (tet_mesh.n_elements,):
            raise ValueError(f'surface_triangle_mesh: solid_elements has shape {solid.shape}, expected '
                             f'({tet_mesh.n_elements},)')
        first, second = side_members(pairs, solid)
        if (first == second).any():
            raise ValueError(f'surface_triangle_mesh: {int((first == second).sum())} faces have a solid element on '
                             'both or neither side')
        owners = np.where(first, pairs[:, 0], pairs[:, 1])
    signs = np.sign(plane - tet_mesh.centroids(owners)[:, axis])
    if not (np.all(signs > 0.0) or np.all(signs < 0.0)):
        raise ValueError(f'surface_triangle_mesh: the solid lies on both sides of the {name} = {plane:.6g} m plane; '
                         'select the faces of one side (one gap)')
    nodes, inverse = np.unique(face_nodes.ravel(), return_inverse=True)
    triangles = np.asarray(inverse).reshape(-1).reshape(face_nodes.shape)
    points = np.ascontiguousarray(tet_mesh.points[nodes][:, list(IN_PLANE_AXES[axis])])
    window = tuple(float(value) for value in (*points.min(axis=0), *points.max(axis=0)))
    mesh = build_triangle_mesh(points, triangles, window=window,
                               quality=mesh_quality(points, triangles, mesher='tetrahedron_faces'))
    n_tet_points = tet_mesh.n_points
    tet_keys = tet_mesh.edges[:, 0].astype(np.int64) * n_tet_points + tet_mesh.edges[:, 1]
    ends = nodes[mesh.edges.astype(np.int64)]   # ascending node maps: lower surface node -> lower tet node
    wanted = ends[:, 0] * n_tet_points + ends[:, 1]
    order = np.argsort(tet_keys, kind='stable')
    found = order[np.minimum(np.searchsorted(tet_keys, wanted, sorter=order), len(order) - 1)]
    if not np.array_equal(tet_keys[found], wanted):
        raise RuntimeError('surface_triangle_mesh: a surface edge is no tetrahedron edge')
    return SurfaceMesh(mesh=mesh, tet_mesh=tet_mesh, axis=axis, plane=plane, outward_sign=float(signs[0]), nodes=nodes,
                       edges=found.astype(np.int64), faces=faces, solid_elements=owners)


def normal_displacement(surface, displacement, order=None) -> np.ndarray:
    """Gap-opening w = -u . n_out [m] on the surface full Lagrange numbering of a tet displacement (interleaved 3 d +
    component or (n_dofs, 3); P1 / P2 by length, real or complex); order None = the field's, 1 = its node values."""
    tet_mesh = surface.tet_mesh
    values = np.asarray(displacement)
    if values.ndim == 1:
        if values.size % VECTOR_COMPONENTS:
            raise ValueError(f'normal_displacement: {values.size} interleaved values are no multiple of 3')
        values = values.reshape(-1, VECTOR_COMPONENTS)
    elif values.ndim != 2 or values.shape[1] != VECTOR_COMPONENTS:
        raise ValueError(f'normal_displacement: displacement shape {values.shape}, expected (3 n,) or (n, 3)')
    field_order = {tet_mesh.n_points: 1, tet_mesh.n_points + tet_mesh.n_edges: 2}.get(len(values))
    if field_order is None:
        raise ValueError(f'normal_displacement: {len(values)} displacement dofs match neither P1 '
                         f'({tet_mesh.n_points}) nor P2 ({tet_mesh.n_points + tet_mesh.n_edges}) of the tet mesh')
    order = field_order if order is None else int(order)
    if order not in (1, 2) or order > field_order:
        raise ValueError(f'normal_displacement: order {order!r} from a P{field_order} field')
    return -surface.outward_sign * values[surface.tet_dofs(order), surface.axis]


def lagrange_field_function(mesh, coefficients):
    """Callable (points (m, q, 2), mesh elements (m,)) -> (m, q) of full-mesh Lagrange P1 / P2 coefficients (order by
    length) on an affine TriangleMesh: a fem.assembly coefficient / source form."""
    coefficients = np.asarray(coefficients)
    order = ({mesh.n_points: 1, mesh.n_points + mesh.n_edges: 2}.get(coefficients.shape[0])
             if coefficients.ndim == 1 else None)
    if order is None:
        raise ValueError(f'lagrange_field_function: coefficients of shape {coefficients.shape} match neither P1 '
                         f'({mesh.n_points},) nor P2 ({mesh.n_points + mesh.n_edges},) of the mesh')
    if (mesh.edge_curve >= 0).any():
        raise ValueError('lagrange_field_function: the mesh has curved edges (affine elements only)')
    triangles = mesh.triangles.astype(np.int64)
    element_ids = triangles if order == 1 else np.concatenate(
        [triangles, mesh.n_points + mesh.element_edges.astype(np.int64)], axis=1)
    corners = mesh.points[triangles]                                                        # (n, 3, 2)
    jacobian = np.stack([corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]], axis=2)   # columns
    determinant = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
    inverse = _inverse_2x2(jacobian, determinant[:, None, None])   # once per mesh: no 2 x 2 solve per point
    origin = corners[:, 0]

    def at_points(points, elements):
        elements = np.asarray(elements, dtype=np.int64)
        offsets = np.asarray(points, dtype=np.float64) - origin[elements][:, None, :]      # (m, q, 2)
        reference = np.einsum('mij,mqj->mqi', inverse[elements], offsets)                   # (m, q, 2)
        basis = lagrange_values(order, reference)                                           # (m, q, n)
        return np.einsum('mqn,mn->mq', basis, coefficients[element_ids[elements]])
    return at_points


def boundary_edges_where(mesh, select=None) -> np.ndarray:
    """Int64 outer-boundary edges of a TriangleMesh whose two end nodes satisfy select(points (k, 2)) -> bool (k,)
    (None: every boundary edge), e.g. the vented rims of a squeeze-film footprint."""
    edges = mesh.boundary_edges().astype(np.int64)
    if select is None or not len(edges):
        return edges
    ends = mesh.edges[edges]
    keep = (np.asarray(select(mesh.points[ends[:, 0]]), dtype=bool)
            & np.asarray(select(mesh.points[ends[:, 1]]), dtype=bool))
    return edges[keep]
