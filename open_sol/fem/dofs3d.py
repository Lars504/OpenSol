"""Degree-of-freedom maps on a TetrahedronMesh (igor3d_plan section 7, T5): Lagrange P1 / P2 and Nedelec first kind
degree 1 / 2 over the whole mesh, a dof = its mesh entity; periodic dof partners and the Bloch masters / phases (em3d,
the tetrahedra solid). Host numpy."""
# Numbering: Lagrange P0 element, P1 node, P2 node or n_points + edge; Nedelec degree 1 edge, degree 2 2 edge + slot (slot 0 =
# L_lo W, 1 = L_hi W), then 2 n_edges + 2 face + slot (fem.elements3d face functions). Element subsets (PEC regions)
# are the assembler's business: every mesh entity keeps its dof, unused ones are pinned by the caller.
from dataclasses import dataclass, field

import numpy as np

from .elements3d import LAGRANGE_COUNTS_3D, LOCAL_EDGES, LOCAL_FACES, NEDELEC_COUNTS_3D
from .dofs import bloch_master_map
from .mesh3d import periodic_node_partners

_LOCAL_EDGE_OF = {pair: index for index, pair in enumerate(LOCAL_EDGES)}
_FACE_EDGE_TABLE = np.array([[_LOCAL_EDGE_OF[(a, b)], _LOCAL_EDGE_OF[(a, c)], _LOCAL_EDGE_OF[(b, c)]]
                             for a, b, c in LOCAL_FACES])   # local edges of local face k


@dataclass(frozen=True, eq=False)
class DofMap3D:
    """Dofs of one element family on every element: element_dofs[e] in the local order of fem.elements3d."""
    mesh: object
    family: str                   # 'lagrange' | 'nedelec'
    order: int                    # Lagrange 0 / 1 / 2, Nedelec degree 1 / 2
    element_dofs: np.ndarray      # int64 (n_elements, n_local)
    n_dofs: int
    face_quadrature: dict = field(default_factory=dict, init=False, repr=False)   # assembly3d's face term cache

    @property
    def n_local(self) -> int:
        return int(self.element_dofs.shape[1])

    def face_dofs(self, faces) -> np.ndarray:
        """Sorted int64 dofs with a tangential (Nedelec) / nonzero (Lagrange) trace on mesh faces: their nodes, edges
        (and P2 edge nodes / degree-2 face dofs)."""
        mesh = self.mesh
        faces = np.asarray(faces, dtype=np.int64)
        if faces.size == 0:
            return np.zeros(0, dtype=np.int64)
        if self.family == 'lagrange' and self.order == 0:
            raise ValueError('face_dofs: a P0 map has no face trace')
        face_edges = _face_edges(mesh, faces)
        if self.family == 'lagrange':
            ids = [mesh.faces[faces].ravel().astype(np.int64)]
            if self.order == 2:
                ids.append(mesh.n_points + face_edges.ravel())
        elif self.order == 1:
            ids = [face_edges.ravel()]
        else:
            ids = [2 * face_edges.ravel(), 2 * face_edges.ravel() + 1,
                   2 * mesh.n_edges + 2 * faces, 2 * mesh.n_edges + 2 * faces + 1]
        return np.unique(np.concatenate(ids))

    def face_local_dofs(self, faces) -> np.ndarray:
        """(k, 3 | 6) int64 Lagrange P1 / P2 dofs of mesh faces in the face-local order: the face nodes ascending (a,
        b, c), then (P2) the edges ab, ac, bc (fem.assembly3d face integrals)."""
        if self.family != 'lagrange' or self.order not in (1, 2):
            raise ValueError('face_local_dofs needs a Lagrange P1 / P2 map')
        faces = np.asarray(faces, dtype=np.int64)
        nodes = self.mesh.faces[faces].astype(np.int64)
        if self.order == 1:
            return nodes
        return np.concatenate([nodes, self.mesh.n_points + _face_edges(self.mesh, faces)], axis=1)

    def element_dofs_of(self, elements) -> np.ndarray:
        """Sorted int64 dofs of mesh elements (int indices or a bool mask)."""
        return np.unique(self.element_dofs[np.asarray(elements)].ravel())

    def dof_points(self) -> np.ndarray:
        """(n_dofs, 3) entity centres: nodes, edge midpoints, face centroids (P0: element centroids)."""
        mesh = self.mesh
        if self.family == 'lagrange' and self.order == 0:
            return mesh.centroids()
        edge_centres = mesh.points[mesh.edges].mean(axis=1)
        if self.family == 'lagrange':
            return mesh.points.copy() if self.order == 1 else np.concatenate([mesh.points, edge_centres])
        if self.order == 1:
            return edge_centres
        face_centres = mesh.points[mesh.faces].mean(axis=1)
        return np.concatenate([np.repeat(edge_centres, 2, axis=0), np.repeat(face_centres, 2, axis=0)])


def _face_edges(mesh, faces) -> np.ndarray:
    """(k, 3) int64 mesh edges of mesh faces (through one adjacent element's local maps)."""
    owners = mesh.face_elements[faces, 0].astype(np.int64)
    local_face = np.argmax(mesh.element_faces[owners] == np.asarray(faces)[:, None], axis=1)
    return mesh.element_edges[owners[:, None], _FACE_EDGE_TABLE[local_face]].astype(np.int64)


def _entity_lookup(keys, wanted, what) -> np.ndarray:
    """int64 indices of `wanted` int64 keys among the entity keys; ValueError naming `what` when one is missing."""
    order = np.argsort(keys, kind='stable')
    position = np.minimum(np.searchsorted(keys, wanted, sorter=order), max(len(keys) - 1, 0))
    found = order[position]
    missing = int((keys[found] != wanted).sum())
    if missing:
        raise ValueError(f'periodic dofs: {missing} {what}s of the hi side have no partner on the lo side; the two sides '
                         'are meshed differently')
    return found.astype(np.int64)


def periodic_dof_partners(dofs, node_partners) -> np.ndarray:
    """int64 (n_dofs,): the dof of the partner entity of every dof on the hi side (node_partners of
    fem.mesh3d.periodic_node_partners), -1 elsewhere; ValueError when a partner edge / face is missing or (Nedelec) the
    pairs do not keep the node order (orientation-reversing pairs would need dof signs; Lagrange dofs have none)."""
    mesh = dofs.mesh
    if dofs.family == 'lagrange' and dofs.order == 0:
        raise ValueError('periodic_dof_partners: a P0 map has no face trace')
    node_partners = np.asarray(node_partners, dtype=np.int64)
    hi_nodes = np.flatnonzero(node_partners >= 0)
    if dofs.family != 'lagrange' and (np.diff(node_partners[hi_nodes]) <= 0).any():
        raise ValueError('periodic_dof_partners: the periodic node pairs do not keep the node order (the hi side is '
                         'numbered unlike the lo side), so edge / face dofs would need signs')
    n_points, n_edges = mesh.n_points, mesh.n_edges
    edges = mesh.edges.astype(np.int64)
    edge_keys = edges[:, 0] * n_points + edges[:, 1]
    hi_edges = np.flatnonzero((node_partners[edges] >= 0).all(axis=1))
    paired = np.sort(node_partners[edges[hi_edges]], axis=1)   # a no-op when the pairs keep the node order
    edge_partners = _entity_lookup(edge_keys, paired[:, 0] * n_points + paired[:, 1], 'edge')
    partners = np.full(dofs.n_dofs, -1, dtype=np.int64)
    if dofs.family == 'lagrange':
        partners[hi_nodes] = node_partners[hi_nodes]
        if dofs.order == 2:
            partners[n_points + hi_edges] = n_points + edge_partners
        return partners
    if dofs.order == 1:
        partners[hi_edges] = edge_partners
        return partners
    faces = mesh.faces.astype(np.int64)
    face_keys = _entity_lookup(edge_keys, faces[:, 0] * n_points + faces[:, 1], 'edge') * n_points + faces[:, 2]
    hi_faces = np.flatnonzero((node_partners[faces] >= 0).all(axis=1))
    paired = node_partners[faces[hi_faces]]
    paired_edges = _entity_lookup(edge_keys, paired[:, 0] * n_points + paired[:, 1], 'edge')
    face_partners = _entity_lookup(face_keys, paired_edges * n_points + paired[:, 2], 'face')
    for slot in (0, 1):
        partners[2 * hi_edges + slot] = 2 * edge_partners + slot
        partners[2 * n_edges + 2 * hi_faces + slot] = 2 * n_edges + 2 * face_partners + slot
    return partners


def bloch_masters(dofs, wavenumbers) -> tuple:
    """dofs.bloch_master_map of a tetrahedron DofMap3D's Bloch-periodic axes {axis 0 / 1 / 2: k [rad/m]}; ValueError
    when the mesh sides do not match."""
    bounds = dofs.mesh.bounds
    return bloch_master_map(dofs.n_dofs, [bounds[3 + axis] - bounds[axis] for axis in range(3)], wavenumbers,
                            lambda axis: periodic_dof_partners(dofs, periodic_node_partners(dofs.mesh, axis)))


def lagrange_dofs_3d(mesh, order) -> DofMap3D:
    """Lagrange P0 (one dof per element: per-element state, D12) / P1 / P2 dofs of every element."""
    if order not in LAGRANGE_COUNTS_3D:
        raise ValueError(f'lagrange_dofs_3d: order must be 0, 1 or 2, got {order!r}')
    if order == 0:
        return DofMap3D(mesh=mesh, family='lagrange', order=0,
                        element_dofs=np.arange(mesh.n_elements, dtype=np.int64)[:, None], n_dofs=mesh.n_elements)
    local = mesh.tets.astype(np.int64)
    n_dofs = mesh.n_points
    if order == 2:
        local = np.concatenate([local, mesh.n_points + mesh.element_edges.astype(np.int64)], axis=1)
        n_dofs += mesh.n_edges
    return DofMap3D(mesh=mesh, family='lagrange', order=order, element_dofs=local, n_dofs=int(n_dofs))


def nedelec_dofs_3d(mesh, degree) -> DofMap3D:
    """Nedelec first-kind dofs of degree 1 (one per edge) or 2 (two per edge, two per face) of every element."""
    if degree not in NEDELEC_COUNTS_3D:
        raise ValueError(f'nedelec_dofs_3d: degree must be 1 or 2, got {degree!r}')
    element_edges = mesh.element_edges.astype(np.int64)
    if degree == 1:
        return DofMap3D(mesh=mesh, family='nedelec', order=1, element_dofs=element_edges, n_dofs=mesh.n_edges)
    edge_dofs = np.stack([2 * element_edges, 2 * element_edges + 1], axis=2).reshape(mesh.n_elements, 12)
    element_faces = mesh.element_faces.astype(np.int64)
    face_dofs = np.stack([2 * mesh.n_edges + 2 * element_faces, 2 * mesh.n_edges + 2 * element_faces + 1],
                         axis=2).reshape(mesh.n_elements, 8)
    return DofMap3D(mesh=mesh, family='nedelec', order=2, element_dofs=np.concatenate([edge_dofs, face_dofs], axis=1),
                    n_dofs=int(2 * mesh.n_edges + 2 * mesh.n_faces))
