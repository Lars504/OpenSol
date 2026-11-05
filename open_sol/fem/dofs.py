"""Degree-of-freedom maps of the FEM path (mesh plan 2.4): Lagrange P0 / P1 / P2 and Nedelec degree 1 / 2 on all
elements or an element subset (a region of the mesh), global numbering compressed to the subset; periodic dof
partners, Bloch masters / phases and the Bloch prolongation (new_modules_plan G3, the 2D twins of fem.dofs3d). Host
numpy."""
# Bloch convention (as fem.dofs3d / em3d): each hi-side dof stands for its lo-side partner times e^{-j k L} per crossed
# period, u(x + L) = u(x) e^{-j k L}; x = P x_r, reduced operators P^H A P (Hermitian for Hermitian A).
from dataclasses import dataclass, field

import numpy as np

from .elements import LAGRANGE_COUNTS, NEDELEC_COUNTS, element_geometry
from .mesh import edge_keys, one_sided, periodic_node_partners


@dataclass(frozen=True, eq=False)
class DofMap:
    """Dofs of one element family on mesh elements `elements` (sorted int64 mesh indices); element_dofs[p] are the
    dofs of elements[p] in the local order of fem.elements. global_ids: the mesh entity of each dof (Lagrange P0:
    element, P1: node, P2: node or n_points + edge; Nedelec: 2 edge + slot / edge, then interior)."""
    mesh: object
    family: str                   # 'lagrange' | 'nedelec'
    order: int                    # Lagrange 0 / 1 / 2, Nedelec degree 1 / 2
    geometry_order: int           # 1 straight elements, 2 isoparametric on curved edges
    elements: np.ndarray          # int64 (m,)
    element_dofs: np.ndarray      # int64 (m, n_local)
    global_ids: np.ndarray        # int64 (n_dofs,)
    element_position: np.ndarray  # int64 (n_elements,): position in `elements`, -1 outside the subset
    edge_quadrature: dict = field(default_factory=dict, init=False, repr=False)   # assembly's edge rule cache
    element_quadrature: dict = field(default_factory=dict, init=False, repr=False)   # per-element integral cache

    @property
    def n_dofs(self) -> int:
        return len(self.global_ids)

    @property
    def n_local(self) -> int:
        return self.element_dofs.shape[1]

    @property
    def full(self) -> bool:
        """True when every mesh element is in the subset."""
        return len(self.elements) == self.mesh.n_elements

    def boundary_edges(self) -> np.ndarray:
        """Int64 mesh edges adjacent to exactly one subset element (the outer boundary of the subset)."""
        return np.flatnonzero(one_sided(self.mesh.edge_elements, self.element_position >= 0))

    def edge_dofs(self, edges) -> np.ndarray:
        """Sorted int64 dofs whose support touches mesh edges `edges` along them (Lagrange: end nodes and the P2 edge
        node; Nedelec: the edge's tangential dofs); entities outside the subset are skipped. P0: none."""
        edges = np.asarray(edges, dtype=np.int64)
        mesh = self.mesh
        if self.family == 'lagrange':
            if self.order == 0:
                return np.zeros(0, dtype=np.int64)
            ids = [mesh.edges[edges].ravel()]
            if self.order == 2:
                ids.append(mesh.n_points + edges)
        else:
            ids = [edges] if self.order == 1 else [2 * edges, 2 * edges + 1]
        return self._dofs_of(np.concatenate(ids))

    def node_dofs(self, nodes) -> np.ndarray:
        """Sorted int64 Lagrange dofs of mesh nodes (P1 / P2); nodes outside the subset skipped."""
        if self.family != 'lagrange' or self.order == 0:
            raise ValueError('node_dofs: only P1 / P2 Lagrange maps have node dofs')
        return self._dofs_of(np.asarray(nodes, dtype=np.int64))

    def _dofs_of(self, ids) -> np.ndarray:
        found = np.searchsorted(self.global_ids, ids)
        found = np.minimum(found, max(self.n_dofs - 1, 0))
        hit = self.global_ids[found] == ids if self.n_dofs else np.zeros(len(ids), dtype=bool)
        return np.unique(found[hit])

    def dof_points(self) -> np.ndarray:
        """Float64 (n_dofs, 2) interpolation points (Lagrange): nodes, P2 edge nodes (on the curve for geometry order
        2, chord midpoints for 1), P0 the mapped element centroid; Nedelec: edge midpoints / element centroids."""
        mesh = self.mesh
        if self.family == 'lagrange' and self.order == 0:
            geometry = element_geometry(mesh, self.elements, np.array([[1.0 / 3.0, 1.0 / 3.0]]),
                                        self.geometry_order)
            return geometry.points[:, 0, :]
        points = np.empty((self.n_dofs, 2))
        ids = self.global_ids
        if self.family == 'lagrange':
            is_node = ids < mesh.n_points
            points[is_node] = mesh.points[ids[is_node]]
            edges = ids[~is_node] - mesh.n_points
            points[~is_node] = self._edge_points(edges)
            return points
        n_edge_dofs = mesh.n_edges * (1 if self.order == 1 else 2)
        on_edge = ids < n_edge_dofs
        points[on_edge] = self._edge_points(ids[on_edge] // (1 if self.order == 1 else 2))
        interior_elements = (ids[~on_edge] - n_edge_dofs) // 2
        points[~on_edge] = mesh.points[mesh.triangles[interior_elements]].mean(axis=1)
        return points

    def _edge_points(self, edges):
        mesh = self.mesh
        if self.geometry_order == 2:
            return mesh.midside_points[edges]
        return 0.5 * (mesh.points[mesh.edges[edges, 0]] + mesh.points[mesh.edges[edges, 1]])


def _subset(mesh, elements):
    if elements is None:
        return np.arange(mesh.n_elements, dtype=np.int64)
    elements = np.asarray(elements)
    if elements.dtype == bool:
        if elements.shape != (mesh.n_elements,):
            raise ValueError(f'dof map: element mask has shape {elements.shape}, expected ({mesh.n_elements},)')
        return np.flatnonzero(elements).astype(np.int64)
    return np.unique(elements.astype(np.int64))


def _compress(mesh, family, order, geometry_order, elements, local_ids) -> DofMap:
    """DofMap numbering the global ids of local_ids (m, n) consecutively in increasing id order."""
    global_ids, inverse = np.unique(local_ids, return_inverse=True)
    position = np.full(mesh.n_elements, -1, dtype=np.int64)
    position[elements] = np.arange(len(elements))
    return DofMap(mesh=mesh, family=family, order=order, geometry_order=geometry_order, elements=elements,
                  element_dofs=inverse.reshape(local_ids.shape).astype(np.int64), global_ids=global_ids,
                  element_position=position)


def lagrange_dofs(mesh, order, elements=None, geometry_order=None) -> DofMap:
    """Lagrange P0 / P1 / P2 dofs on mesh elements (None = all, int indices or a bool mask); geometry_order None =
    automatic (2 for P0 / P2, 1 for P1: decision D6)."""
    if order not in LAGRANGE_COUNTS:
        raise ValueError(f'lagrange_dofs: order must be 0, 1 or 2, got {order!r}')
    geometry_order = (1 if order == 1 else 2) if geometry_order is None else int(geometry_order)
    elements = _subset(mesh, elements)
    if order == 0:
        local_ids = elements[:, None]
    else:
        local_ids = mesh.triangles[elements].astype(np.int64)
        if order == 2:
            local_ids = np.concatenate([local_ids, mesh.n_points + mesh.element_edges[elements].astype(np.int64)],
                                       axis=1)
    return _compress(mesh, 'lagrange', order, geometry_order, elements, local_ids)


def nedelec_dofs(mesh, degree, elements=None, geometry_order=None) -> DofMap:
    """Nedelec first-kind dofs of degree 1 (one per edge) or 2 (two per edge, lower-node function first, two per
    element) on mesh elements; geometry_order None = the degree."""
    if degree not in NEDELEC_COUNTS:
        raise ValueError(f'nedelec_dofs: degree must be 1 or 2, got {degree!r}')
    geometry_order = int(degree if geometry_order is None else geometry_order)
    elements = _subset(mesh, elements)
    element_edges = mesh.element_edges[elements].astype(np.int64)
    if degree == 1:
        local_ids = element_edges
    else:
        edge_ids = np.stack([2 * element_edges, 2 * element_edges + 1], axis=2).reshape(len(elements), 6)
        interior = 2 * mesh.n_edges + 2 * elements[:, None] + np.arange(2)[None, :]
        local_ids = np.concatenate([edge_ids, interior], axis=1)
    return _compress(mesh, 'nedelec', degree, geometry_order, elements, local_ids)


# ---- periodic sides and Bloch cells (new_modules_plan G3) ----
def periodic_dof_partners(dofs, node_partners) -> np.ndarray:
    """Int64 (n_dofs,): the dof of the partner entity of every dof on the hi side (node_partners of
    fem.mesh.periodic_node_partners), -1 elsewhere; Lagrange P1 / P2 only (their dofs carry no sign); ValueError for a
    missing partner edge or a partner outside the map's element subset."""
    if dofs.family != 'lagrange' or dofs.order == 0:
        raise ValueError('periodic_dof_partners: Lagrange P1 / P2 maps only (P0 has no side trace, Nedelec edge dofs '
                         'would need signs)')
    mesh = dofs.mesh
    n_points = mesh.n_points
    node_partners = np.asarray(node_partners, dtype=np.int64)
    entity_partners = np.full(n_points + (mesh.n_edges if dofs.order == 2 else 0), -1, dtype=np.int64)
    entity_partners[:n_points] = node_partners
    if dofs.order == 2:
        edges = mesh.edges.astype(np.int64)
        hi_edges = np.flatnonzero((node_partners[edges] >= 0).all(axis=1))   # edges along the hi side
        keys = edge_keys(edges, n_points)   # ascending: connectivity numbers edges by key
        wanted = edge_keys(node_partners[edges[hi_edges]], n_points)
        found = np.minimum(np.searchsorted(keys, wanted), max(len(keys) - 1, 0))
        missing = int((keys[found] != wanted).sum()) if len(wanted) else 0
        if missing:
            raise ValueError(f'periodic dofs: {missing} edges of the hi side have no partner edge on the lo side; the '
                             'two sides are meshed differently')
        entity_partners[n_points + hi_edges] = n_points + found
    ids = dofs.global_ids
    partner_ids = entity_partners[ids]
    on_hi = np.flatnonzero(partner_ids >= 0)
    position = np.minimum(np.searchsorted(ids, partner_ids[on_hi]), max(dofs.n_dofs - 1, 0))
    missing = int((ids[position] != partner_ids[on_hi]).sum()) if len(on_hi) else 0
    if missing:
        raise ValueError(f'periodic dofs: {missing} hi-side dofs have their partner outside the dof map elements (the '
                         'element subset does not reach the lo side alike)')
    partners = np.full(dofs.n_dofs, -1, dtype=np.int64)
    partners[on_hi] = position
    return partners


def bloch_master_map(n_dofs, extents, wavenumbers, partners_of) -> tuple:
    """(master (n_dofs,) int64 = the dof each dof stands for (itself off the hi sides), phase (n_dofs,) complex128 with
    u = phase x u_master) of the Bloch-periodic axes {axis: k [rad/m]}: e^{-j k L} per crossed period (L = extents[axis],
    the node extent; a phase within 1e-15 of real is made real); partners_of(axis) = the dof partners of that axis
    (2D and 3D)."""
    master = np.arange(n_dofs, dtype=np.int64)
    phase = np.ones(n_dofs, dtype=np.complex128)
    for axis, wavenumber in sorted(wavenumbers.items()):   # an image on another axis' hi side is mapped again there
        partners = partners_of(axis)
        step = np.exp(-1j * float(wavenumber) * float(extents[axis]))
        if abs(step.imag) <= 1e-15:   # k L a multiple of pi: keep the pencil real
            step = complex(np.sign(step.real), 0.0)
        crossing = partners[master] >= 0
        phase[crossing] *= step
        master[crossing] = partners[master[crossing]]
    return master, phase


def bloch_masters(dofs, wavenumbers) -> tuple:
    """bloch_master_map of a triangle DofMap's Bloch-periodic axes {axis 0 / 1: k [rad/m]}; ValueError when the mesh
    sides do not match."""
    points = dofs.mesh.points
    return bloch_master_map(dofs.n_dofs, points.max(axis=0) - points.min(axis=0), wavenumbers,
                            lambda axis: periodic_dof_partners(dofs, periodic_node_partners(dofs.mesh, axis)))


def bloch_prolongation(master, phase, free=None) -> tuple:
    """(columns (n_r,) int64: the free masters, host CSR P (n, n_r)) with x = P x_r: each dof standing for a free
    master carries its phase (float64 P when every phase is real), the rest 0; a master is free when every dof it
    stands for is (free None = every dof)."""
    import scipy.sparse

    master = np.asarray(master, dtype=np.int64)
    free = np.ones(master.size, dtype=bool) if free is None else np.asarray(free, dtype=bool)
    columns = np.setdiff1d(master[free], master[~free])
    column_of = np.full(master.size, -1, dtype=np.int64)
    column_of[columns] = np.arange(columns.size)
    rows = np.flatnonzero(column_of[master] >= 0)
    values = np.asarray(phase)[rows]
    if np.iscomplexobj(values) and not np.any(values.imag):
        values = np.ascontiguousarray(values.real)
    prolongation = scipy.sparse.csr_matrix((values, (rows, column_of[master[rows]])),
                                           shape=(master.size, columns.size))
    return columns, prolongation
