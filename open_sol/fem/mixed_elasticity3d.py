"""Taylor-Hood u-p pieces of the elasticity on tetrahedra: P1 pressure spaces per element group and the coupling /
pressure mass blocks of a P2 displacement DofMap3D, the pressure at element points and its nodal display values (the
fem/mixed_elasticity.py analogue; its pressure_regions, embedded and saddle_coupling are dimension-free). Host numpy,
scipy inside functions."""
# Weak form, signs and block layout: fem/mixed_elasticity.py header (B_ij = int N_p,i div N_u,j dV, M_p,g = int N_p N_p
# dV over group g; [[A_uu, -B^T], [-B, -M_p / K*]] in x = [u; p]). Affine tetrahedra: div N_u = the sum of the normal
# Voigt rows of strain_operator_3d (d N / d x on the x component, ...), built directly from the Piola gradients; the
# rule of the elastic stiffness (degree 2 (order - 1)) integrates both blocks exactly. The pressure is P1 on the group's
# nodes, continuous inside a group, duplicated on nodes shared by groups; group g's dofs are offsets[g] + the index of
# its node in nodes[g].
from dataclasses import dataclass

import numpy as np

from .assembly import CsrAccumulator
from .assembly3d import _rule_degree, _volume_weights, element_chunks_3d
from .elasticity import _basis_product_blocks
from .elasticity3d import COMPONENTS, vector_dofs_3d
from .elements3d import lagrange_basis_3d, piola_vectors_3d
from .quadrature3d import tetrahedron_rule

PRESSURE_LOCAL_3D = 4   # P1 pressure dofs per tetrahedron (its corners, mesh.tets order)


@dataclass(frozen=True, eq=False)
class PressureSpace3D:
    """P1 pressure dofs per element group on a TetrahedronMesh: `elements` (int64 sorted mesh elements per group),
    `nodes` (int64 sorted mesh nodes per group), `offsets` int64 (n_groups + 1,) and `element_table` int64
    (n_mesh_elements, 4) global pressure dofs per mesh element (-1 outside every group; read-only)."""
    mesh: object
    elements: tuple
    nodes: tuple
    offsets: np.ndarray
    element_table: np.ndarray

    @property
    def n_dofs(self) -> int:
        return int(self.offsets[-1])

    @property
    def n_groups(self) -> int:
        return len(self.elements)

    def group_element_dofs(self, group) -> np.ndarray:
        """Int64 (m_g, 4) global pressure dofs of group `group`'s elements."""
        return self.element_table[self.elements[group]]

    def dof_points(self) -> np.ndarray:
        """Float64 (n_dofs, 3) node positions of the pressure dofs."""
        return self.mesh.points[np.concatenate(self.nodes)] if self.nodes else np.zeros((0, 3))


def pressure_space_3d(mesh, groups, n_groups) -> PressureSpace3D:
    """PressureSpace3D of the groups (int64 per mesh element, -1 = none); ValueError for an empty group."""
    groups = np.asarray(groups, dtype=np.int64)
    if groups.shape != (mesh.n_elements,):
        raise ValueError(f'pressure_space_3d: groups has shape {groups.shape}, expected ({mesh.n_elements},)')
    elements, nodes, offsets = [], [], [0]
    # booked (estimate): element_table int64 (n_mesh_elements, 4) + per group its elements and nodes (int64), host,
    # kept with the harmonic operators
    table = np.full((mesh.n_elements, PRESSURE_LOCAL_3D), -1, dtype=np.int64)
    for group in range(int(n_groups)):
        members = np.flatnonzero(groups == group).astype(np.int64)
        if not members.size:
            raise ValueError(f'pressure_space_3d: group {group} has no element')
        group_nodes, inverse = np.unique(mesh.tets[members].astype(np.int64), return_inverse=True)
        table[members] = offsets[-1] + np.asarray(inverse, dtype=np.int64).reshape(-1, PRESSURE_LOCAL_3D)
        elements.append(members)
        nodes.append(group_nodes)
        offsets.append(offsets[-1] + group_nodes.size)
    table.flags.writeable = False
    return PressureSpace3D(mesh, tuple(elements), tuple(nodes), np.asarray(offsets, dtype=np.int64), table)


def assemble_pressure_terms_3d(dofs, space, degree=None) -> tuple:
    """(B (n_p, 3 n_u) CSR, [M_p,g (n_p, n_p) CSR per group]) of a full-mesh vector Lagrange DofMap3D (P2; u rows
    interleaved 3 d + component) and a PressureSpace3D on its mesh: per group its element chunks (element_chunks_3d),
    the Piola gradients once per chunk, a CsrAccumulator for B and one per group for M_p (never kept_pattern: group
    patterns would evict K's kept block pattern). Rule degree default 2 (order - 1), as the elastic stiffness."""
    if dofs.family != 'lagrange' or dofs.order not in (1, 2):
        raise ValueError('assemble_pressure_terms_3d needs a Lagrange P1 / P2 DofMap3D')
    if space.mesh is not dofs.mesh:
        raise ValueError('assemble_pressure_terms_3d: the pressure space and the displacement map need one mesh')
    mesh = dofs.mesh
    n_u, n_p = COMPONENTS * dofs.n_dofs, space.n_dofs
    rule = tetrahedron_rule(_rule_degree(2 * dofs.order - 2, degree))
    _, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    pressure_basis = lagrange_basis_3d(1, rule.points, gradients=False)[0]   # (q, 4), the corners in mesh.tets order
    # booked (estimate): COO batches of B ((4 x 3 n_local) float64 + 2 int32 per element) and of each M_p,g (16 float64
    # + 2 int32 per element) up to COO_BATCH_ENTRIES each, then the CSRs (float64 + int32 indices), host
    coupling = CsrAccumulator((n_p, n_u))
    masses = []
    for group, members in enumerate(space.elements):
        mass = CsrAccumulator((n_p, n_p))
        for chunk, geometry in element_chunks_3d(mesh, members):
            gradients = piola_vectors_3d(reference_gradients, geometry)                  # (m, q, n, 3)
            weights = _volume_weights(geometry, rule)                                    # (m, q)
            count, points, n_local, _ = gradients.shape
            # div N: component c of the vector basis function of node a has d N_a / d x_c (rows 0-2 of the Voigt B)
            weighted = (gradients * weights[:, :, None, None]).reshape(count, points, n_local * COMPONENTS)
            blocks = pressure_basis.T @ weighted.transpose(1, 0, 2).reshape(points, count * n_local * COMPONENTS)
            blocks = blocks.reshape(PRESSURE_LOCAL_3D, count, n_local * COMPONENTS).transpose(1, 0, 2)
            rows = space.element_table[chunk]
            coupling.add(rows, blocks, vector_dofs_3d(dofs.element_dofs[chunk]))
            mass.add(rows, _basis_product_blocks(pressure_basis, weights))
        masses.append(mass.result())
    return coupling.result(), masses


def pressure_at_3d(space, pressure, elements, reference_points) -> np.ndarray:
    """(m, q) pressure (dtype of `pressure`, (n_p,)) at reference points (q, 3) of the mesh elements `elements` (sorted
    int64; 0 on elements in no group): the P1 basis times the element's pressure dofs (one GEMM)."""
    pressure = np.asarray(pressure)
    if pressure.shape != (space.n_dofs,):
        raise ValueError(f'pressure_at_3d: pressure has shape {pressure.shape}, expected ({space.n_dofs},)')
    table = space.element_table[np.asarray(elements, dtype=np.int64)]
    basis = lagrange_basis_3d(1, np.asarray(reference_points, dtype=np.float64).reshape(-1, 3), gradients=False)[0]
    local = np.where(table >= 0, pressure[np.maximum(table, 0)], 0.0)              # (m, 4)
    return local @ basis.T


def nodal_pressure_3d(full_dofs, space, pressure) -> np.ndarray:
    """(n_dofs,) display values of the pressure (dtype of `pressure`) on the full-mesh P1 / P2 DofMap3D: at a node the
    mean over the groups meeting there, at a P2 edge node the mean of its end nodes on edges of grouped elements; 0
    elsewhere."""
    mesh = full_dofs.mesh
    pressure = np.asarray(pressure)
    dtype = np.result_type(pressure, np.float64)
    sums = np.zeros(mesh.n_points, dtype=dtype)
    counts = np.zeros(mesh.n_points)
    on_group_edges = np.zeros(mesh.n_edges, dtype=bool)
    for group, nodes in enumerate(space.nodes):   # unique within a group: a plain fancy += (no np.add.at)
        sums[nodes] += pressure[int(space.offsets[group]):int(space.offsets[group + 1])]
        counts[nodes] += 1.0
        on_group_edges[mesh.element_edges[space.elements[group]].ravel()] = True
    node_values = sums / np.maximum(counts, 1.0)
    result = np.zeros(full_dofs.n_dofs, dtype=dtype)
    result[:mesh.n_points] = node_values   # full-mesh numbering: nodes first, then (P2) n_points + edge
    if full_dofs.n_dofs > mesh.n_points:
        ends = mesh.edges.astype(np.int64)
        result[mesh.n_points:] = np.where(on_group_edges, 0.5 * (node_values[ends[:, 0]] + node_values[ends[:, 1]]),
                                          0.0)
    return result
