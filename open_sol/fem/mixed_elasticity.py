"""Taylor-Hood u-p pieces of the 2D elasticity: P1 pressure spaces per element group and the coupling / pressure mass
blocks of a P2 displacement map (planar plane strain, axisymmetric rings, azimuthal order m >= 1), the axis pressure
dofs, the pressure at element points and its nodal display values, the connected pressure regions of a union of groups
and the index shifts that place the blocks in the [u; p] numbering. Host numpy, scipy inside functions."""
# Weak form (exp(+j omega t), p > 0 in compression, sigma = 2 G* dev eps - p I, p = -K* tr eps): for every v, q
#   int 2 G* dev eps(u) : eps(v) dV - int p tr eps(v) dV - omega^2 int rho u . v dV = int f . v dV   (momentum)
#   -int q (tr eps(u) + p / K*) dV = 0                                                              (volume)
# so with B_ij = int N_p,i tr eps(N_u,j) dV and M_p,ij = int N_p,i N_p,j dV the system in x = [u; p] is the complex
# symmetric [[A_uu, -B^T], [-B, -M_p / K*]] x = [f; 0] (the (u, p) block of the momentum row is -B^T, the (p, u) block
# of the volume row -B: the same matrix transposed). An incompressible group has 1 / K* = 0 (a zero pressure block).
# dV carries the fem.elasticity ring weights (never fem.assembly's x r): planar w det J, axisymmetric x 2 pi r (RING),
# order m >= 1 x pi r (AZIMUTHAL_RING: p = P cos(m phi) like u_r, u_z, the phi integral of cos^2 is pi), taken from
# _operator_at / _azimuthal_terms with the stiffness's own rules; tr eps = the sum of the normal rows of the
# stiffness's strain operator: planar xx + yy (plane strain: eps_zz = 0), axisymmetric rr + zz + tt (tt = u_r / r),
# order m: rr + zz + tt with tt = (u_r + d u_phi / d phi) / r = (U_r + m U_phi) cos(m phi) / r (rows 0-2 of
# azimuthal_strain_operator itself, u_phi = U_phi sin(m phi): the solid's sign convention, so tr eps ~ cos(m phi)). The
# pressure is P1 on the displacement map's geometry (isoparametric P2 on curved edges), continuous inside a group and
# duplicated on nodes shared by groups (the pressure jumps there): group g's dofs are offsets[g] + its map's numbering.
from dataclasses import dataclass

import numpy as np

from .assembly import _to_csr, append_blocks
from .dofs import lagrange_dofs
from .elasticity import (AZIMUTHAL_COMPONENTS, COMPONENTS, _azimuthal_terms, _basis_product_blocks,
                         _check_azimuthal_order, _groups, _operator_at, azimuthal_strain_operator, component_dofs)
from .elements import lagrange_basis

PRESSURE_LOCAL = 3   # P1 pressure dofs per triangle (its corners, mesh.triangles order)


@dataclass(frozen=True, eq=False)
class PressureSpace:
    """P1 pressure dofs per element group: `maps` one fem.dofs.DofMap per group (lagrange_dofs order 1 on the group's
    elements, the displacement map's geometry order), `offsets` int64 (n_groups + 1,) (group g's dofs are offsets[g] +
    its map's numbering), `element_table` int64 (n_mesh_elements, 3) global pressure dofs per mesh element (-1 outside
    every group; read-only)."""
    maps: tuple
    offsets: np.ndarray
    element_table: np.ndarray

    @property
    def n_dofs(self) -> int:
        return int(self.offsets[-1])

    @property
    def n_groups(self) -> int:
        return len(self.maps)

    def group_element_dofs(self, group) -> np.ndarray:
        """Int64 (m_g, 3) global pressure dofs of group `group`'s elements."""
        return int(self.offsets[group]) + self.maps[group].element_dofs

    def dof_points(self) -> np.ndarray:
        """Float64 (n_dofs, 2) node positions of the pressure dofs."""
        points = np.empty((self.n_dofs, 2))
        for group, dofs in enumerate(self.maps):
            points[int(self.offsets[group]):int(self.offsets[group + 1])] = dofs.mesh.points[dofs.global_ids]
        return points

    def dof_nodes(self) -> np.ndarray:
        """Int64 (n_dofs,) mesh node of every pressure dof (a node shared by groups appears once per group)."""
        if not self.maps:
            return np.zeros(0, dtype=np.int64)
        return np.concatenate([dofs.global_ids for dofs in self.maps]).astype(np.int64)


def pressure_space(displacement_dofs, groups, n_groups) -> PressureSpace:
    """PressureSpace of the groups (int64 per mesh element, -1 = none) on a displacement DofMap holding every
    grouped element; ValueError for an empty group or a grouped element off the displacement map."""
    mesh = displacement_dofs.mesh
    groups = np.asarray(groups, dtype=np.int64)
    if groups.shape != (mesh.n_elements,):
        raise ValueError(f'pressure_space: groups has shape {groups.shape}, expected ({mesh.n_elements},)')
    outside = (groups >= 0) & (displacement_dofs.element_position < 0)
    if outside.any():
        raise ValueError(f'pressure_space: {int(outside.sum())} grouped element(s) are not on the displacement map')
    maps, offsets = [], [0]
    # booked (estimate): element_table int64 (n_mesh_elements, 3) + per group a P1 DofMap (element_dofs int64 (m_g, 3),
    # global_ids int64 (n_g,), element_position int64 (n_mesh_elements,)), host, kept with the harmonic operators
    table = np.full((mesh.n_elements, PRESSURE_LOCAL), -1, dtype=np.int64)
    for group in range(int(n_groups)):
        mask = groups == group
        if not mask.any():
            raise ValueError(f'pressure_space: group {group} has no element')
        dofs = lagrange_dofs(mesh, 1, elements=mask, geometry_order=displacement_dofs.geometry_order)
        table[dofs.elements] = offsets[-1] + dofs.element_dofs
        maps.append(dofs)
        offsets.append(offsets[-1] + dofs.n_dofs)
    table.flags.writeable = False
    return PressureSpace(tuple(maps), np.asarray(offsets, dtype=np.int64), table)


def assemble_pressure_terms(displacement_dofs, space, *, axisymmetric=False, azimuthal_order=0) -> tuple:
    """(B (n_p, c n_u) CSR, [M_p,g (n_p, n_p) CSR per group]) on the displacement map's rules per element chunk (module
    header: B_ij = int N_p,i tr eps(N_u,j) dV, M_p,g = int N_p N_p dV over group g; c = 2, or 3 at an azimuthal order
    m >= 1); explicit zeros kept (fem.assembly._to_csr). Rules as assemble_elastic_stiffness (planar max(2 order - 2,
    2), axisymmetric AXISYMMETRIC_DEGREE, curved elements CURVED_DEGREE) / assemble_azimuthal_stiffness: exact for the
    polynomial integrands on straight elements."""
    dofs = displacement_dofs
    order = _check_azimuthal_order(azimuthal_order) if azimuthal_order else 0
    components = AZIMUTHAL_COMPONENTS if order else COMPONENTS
    ring = axisymmetric or bool(order)
    n_u, n_p = components * dofs.n_dofs, space.n_dofs
    table = space.element_table[dofs.elements]   # (m_u, 3) per displacement-map element position
    group_of = np.full(len(dofs.elements), -1, dtype=np.int64)
    for group, pressure_dofs in enumerate(space.maps):
        group_of[dofs.element_position[pressure_dofs.elements]] = group
    # booked (estimate): COO of B ((3 x c n_local) float64 + 2 int64 per element) and of every M_p,g (9 float64 + 2
    # int64 per element) held until the CSRs (float64 data + int32 indices; B ~ n_p x c x the P2 row entries around a
    # vertex, M_p,g ~ the P1 pattern of the group), host
    coupling = ([], [], [])
    masses = [([], [], []) for _ in space.maps]
    degree = 0 if order else max(2 * dofs.order - 2, 2)
    for positions, rule in _groups(dofs, ring, degree):
        positions = positions[group_of[positions] >= 0]
        if not positions.size:
            continue
        pressure_basis = lagrange_basis(1, rule.points)[0]   # (q, 3), the corners in mesh.triangles order
        if order:
            _, basis, gradients, weights, radius = _azimuthal_terms(dofs, positions, rule, None, None)
            operator = azimuthal_strain_operator(basis, gradients, radius, order)
        else:
            _, operator, weights = _operator_at(dofs, positions, rule, axisymmetric)
        normal_rows = 3 if ring else 2   # rr, zz, tt (rows 0-2) / xx, yy: their sum is tr eps of each basis vector
        trace = operator[..., 0, :] + operator[..., 1, :]
        if normal_rows == 3:
            trace = trace + operator[..., 2, :]
        weighted = trace * weights[:, :, None]                                    # (m, q, c n_local)
        count, points, width = weighted.shape
        # sum_q N_p,a(q) w tr eps(N_u,j)(q): one GEMM over every element of the chunk (no per-element matmul)
        blocks = (pressure_basis.T @ weighted.transpose(1, 0, 2).reshape(points, count * width))
        blocks = blocks.reshape(PRESSURE_LOCAL, count, width).transpose(1, 0, 2)  # (m, 3, c n_local)
        rows = table[positions]
        append_blocks(*coupling, rows, blocks, component_dofs(dofs.element_dofs[positions], components))
        mass_blocks = _basis_product_blocks(pressure_basis, weights)              # (m, 3, 3)
        chunk_groups = group_of[positions]
        for group in np.unique(chunk_groups):
            selection = chunk_groups == group
            append_blocks(*masses[int(group)], rows[selection], mass_blocks[selection])
    return (_to_csr(*coupling, (n_p, n_u)),
            [_to_csr(*lists, (n_p, n_p)) for lists in masses])


def axis_pressure_dofs(space, nodes) -> np.ndarray:
    """Sorted int64 pressure dofs on the mesh nodes `nodes` (the axis r = 0 of an axisymmetric solid: held at 0 for an
    azimuthal order m >= 1, where a cos(m phi) pressure is regular on the axis only when it vanishes there)."""
    nodes = np.asarray(nodes, dtype=np.int64)
    found = [int(space.offsets[group]) + np.flatnonzero(np.isin(dofs.global_ids, nodes))
             for group, dofs in enumerate(space.maps)]
    return np.concatenate(found) if found else np.zeros(0, dtype=np.int64)


def pressure_at(displacement_dofs, space, pressure, reference_points) -> np.ndarray:
    """(m_u, q) pressure (dtype of `pressure`, (n_p,)) at reference points (q, 2) of every displacement-map element (0
    on elements in no group): the P1 basis times the element's pressure dofs (one GEMM)."""
    pressure = np.asarray(pressure)
    if pressure.shape != (space.n_dofs,):
        raise ValueError(f'pressure_at: pressure has shape {pressure.shape}, expected ({space.n_dofs},)')
    table = space.element_table[displacement_dofs.elements]
    basis = lagrange_basis(1, np.asarray(reference_points, dtype=np.float64))[0]   # (q, 3)
    local = np.where(table >= 0, pressure[np.maximum(table, 0)], 0.0)              # (m_u, 3)
    return local @ basis.T


def nodal_pressure(full_dofs, space, pressure) -> np.ndarray:
    """(n_global,) display values of the pressure (dtype of `pressure`) on the full-mesh P1 / P2 Lagrange map: at a node
    the mean over the groups meeting there (a jump between groups smears over one element), at a P2 edge node the mean
    of its end nodes on edges of grouped elements; 0 elsewhere."""
    mesh = full_dofs.mesh
    pressure = np.asarray(pressure)
    dtype = np.result_type(pressure, np.float64)
    sums = np.zeros(mesh.n_points, dtype=dtype)
    counts = np.zeros(mesh.n_points)
    on_group_edges = np.zeros(mesh.n_edges, dtype=bool)
    for group, dofs in enumerate(space.maps):
        ids = dofs.global_ids   # unique within a group: a plain fancy += (no np.add.at)
        sums[ids] += pressure[int(space.offsets[group]):int(space.offsets[group + 1])]
        counts[ids] += 1.0
        on_group_edges[mesh.element_edges[dofs.elements].ravel()] = True
    node_values = sums / np.maximum(counts, 1.0)
    ids = full_dofs.global_ids
    is_node = ids < mesh.n_points
    result = np.zeros(full_dofs.n_dofs, dtype=dtype)
    result[is_node] = node_values[ids[is_node]]
    if not is_node.all():
        edges = ids[~is_node] - mesh.n_points
        ends = mesh.edges[edges].astype(np.int64)
        result[~is_node] = np.where(on_group_edges[edges], 0.5 * (node_values[ends[:, 0]] + node_values[ends[:, 1]]),
                                    0.0)
    return result


def pressure_regions(space, groups) -> list:
    """[int64 sorted global pressure dofs] of each connected region of the union of the groups `groups` (an iterable
    of group indices): elements joined through their pressure nodes, and groups joined where they share a mesh node
    (their duplicated dofs there are one region: a pressure constant over two touching groups is one mode). space a
    PressureSpace or fem.mixed_elasticity3d.PressureSpace3D (group_element_dofs, offsets, dof_nodes)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    groups = sorted({int(group) for group in groups})
    if not groups:
        return []
    # the union's dofs, ascending (group blocks are contiguous and in offset order), and their local numbering
    members = np.concatenate([np.arange(int(space.offsets[group]), int(space.offsets[group + 1]), dtype=np.int64)
                              for group in groups])
    local_of = np.full(space.n_dofs, -1, dtype=np.int64)
    local_of[members] = np.arange(members.size, dtype=np.int64)
    element_dofs = [local_of[space.group_element_dofs(group)] for group in groups]
    # every element dof joined to its element's first dof
    first = [np.broadcast_to(local[:, :1], local.shape).ravel() for local in element_dofs]
    second = [local.ravel() for local in element_dofs]
    # every dof joined to the first union dof at its mesh node (the other groups' duplicates there)
    nodes = space.dof_nodes()[members]
    order = np.argsort(nodes, kind='stable')
    starts = np.flatnonzero(np.r_[True, nodes[order][1:] != nodes[order][:-1]])
    lead = order[np.repeat(starts, np.diff(np.r_[starts, order.size]))]
    rows, columns = np.concatenate(first + [lead]), np.concatenate(second + [order])
    graph = coo_matrix((np.ones(rows.size, dtype=np.int8), (rows, columns)), shape=(members.size, members.size))
    n_regions, labels = connected_components(graph, directed=False)
    by_label = np.argsort(labels, kind='stable')   # stable: ascending dofs within a region
    split = np.split(by_label, np.cumsum(np.bincount(labels, minlength=n_regions))[:-1])
    return [members[part] for part in split]


def embedded(matrix, n_total, row_offset=0, column_offset=0):
    """Square CSR (n_total, n_total) holding `matrix` at (row_offset, column_offset) (an index shift of its CSR arrays:
    every stored entry kept, explicit zeros too; no COO round trip, no bmat)."""
    import scipy.sparse
    matrix = matrix.tocsr()
    n_rows, n_columns = matrix.shape
    n_total, row_offset, column_offset = int(n_total), int(row_offset), int(column_offset)
    if row_offset < 0 or column_offset < 0 or row_offset + n_rows > n_total or column_offset + n_columns > n_total:
        raise ValueError(f'embedded: a ({n_rows}, {n_columns}) block at ({row_offset}, {column_offset}) does not fit '
                         f'({n_total}, {n_total})')
    index_dtype = np.int32 if max(n_total, matrix.nnz) < 2 ** 31 else np.int64
    indptr = np.concatenate([np.zeros(row_offset, dtype=np.int64), matrix.indptr.astype(np.int64),
                             np.full(n_total - row_offset - n_rows, matrix.nnz, dtype=np.int64)])
    indices = matrix.indices.astype(np.int64) + column_offset
    # booked (estimate): a copy of the block (data + int32 indices) at its new place, host
    return scipy.sparse.csr_matrix((matrix.data.copy(), indices.astype(index_dtype), indptr.astype(index_dtype)),
                                   shape=(n_total, n_total))


def saddle_coupling(coupling, n_u):
    """Symmetric CSR [[0, -B^T], [-B, 0]] over n_u + n_p of the coupling B (n_p, n_u) (module header signs), explicit
    zeros kept (fem.assembly.pattern_sum)."""
    from .assembly import pattern_sum
    coupling = coupling.tocsr()
    n_p, columns = coupling.shape
    if columns != int(n_u):
        raise ValueError(f'saddle_coupling: B has {columns} columns, expected n_u = {n_u}')
    n_total = int(n_u) + n_p
    # booked (estimate): -B and -B^T embedded (2 copies of B: float64 + int32 indices) before their pattern sum (the
    # saddle CSR: 2 x nnz(B) entries), host
    return pattern_sum([embedded(-coupling, n_total, row_offset=n_u),
                        embedded((-coupling.T).tocsr(), n_total, column_offset=n_u)])
