"""Complex drives of the vector elasticity on triangle meshes (frequency_response loads from fields given at quadrature
points): int B^T sigma dV of a complex stress per point (planar, axisymmetric m = 0 and azimuthal order m >= 1 with the
cos / sin Voigt rows) and int N t ds of a complex edge traction, with the fem.elasticity rules and ring weights; host
numpy."""
# Use (the spontaneous Brillouin drive, physics/mechanics/brillouin.py): a caller samples its fields at the points of
# drive_points / edge_drive_points, forms a complex stress (rows, n) or traction (k, q, d) there and assembles the
# generalised force w with w^T u = sum_points weight sigma . eps(u) (+ sum weight t . u on edges) for EVERY u of the map
# (non-conjugated: the transpose of the strain operator, not an inner product).
# Rows and weights are those of fem.elasticity, taken from its helpers (never re-derived here): planar (xx, yy, g =
# 2 eps_xy) with weight w det J; axisymmetric m = 0 (rr, zz, tt, g_rz) x 2 pi r (RING, whole rings); azimuthal order
# m >= 1 (rr, zz, tt, g_rz: cos(m phi) parts; g_zt, g_rt: sin(m phi) parts) x pi r (AZIMUTHAL_RING), dofs 3 d +
# (U_r, U_z, U_phi). So a drive at order m pairs with the solid's K_m: a stress field whose (rr, zz, tt, rz) rows are
# s_n(r, z) cos(m phi) and whose (zt, rt) rows are s_s(r, z) sin(m phi) does the work w^T U on the displacement u_r =
# U_r cos(m phi), u_z = U_z cos(m phi), u_phi = U_phi sin(m phi) when stress = (s_n, s_s). The odd parity (normal rows
# ~ sin, shear rows ~ cos) is the even solve rotated by pi / (2 m): its drive is (s_n, -s_s) in these rows (u_r = U_r
# sin, u_phi = -U_phi cos; self-test fem_elastic_drive_azimuthal_quadrature).
# Points: every element chunk of assembly._element_groups at DRIVE_DEGREE (the largest Dunavant rule: the Brillouin
# integrand oscillates on lambda / (2 n); curved elements at CURVED_DEGREE if higher), flattened (group, element,
# point); the same chunks and points as elasticity.assemble_stress_load at the axisymmetric degree. B per chunk:
# elasticity._form_operator_at, the stiffness groups' own form; a drive assembled on the same points several times
# (keep_operator: one per scan position) keeps it on the DrivePoints. Edge points: indicators.edge_sides (side 0 =
# mesh.edge_elements[:, 0], unit normals out of it, points and weights) on the edges with a map element; the basis of
# assembly._edge_quadrature at the same edges and degree evaluates at the same points (both take gauss_line(degree (+ 2
# on geometry order 2)) and elements.edge_geometry of the map's geometry order).
from dataclasses import dataclass, field

import numpy as np

from .assembly import _edge_quadrature, _edge_sides, _element_groups
from .elasticity import (AZIMUTHAL_COMPONENTS, AZIMUTHAL_RING, AZIMUTHAL_ROWS, COMPONENTS, RING, _azimuthal_terms,
                         _check_azimuthal_order, _form_operator_at, _weights, component_dofs, strain_rows)
from .elements import element_geometry
from .fixed_pattern import scatter_add
from .indicators import edge_sides
from .quadrature import TRIANGLE_DEGREES

DRIVE_DEGREE = TRIANGLE_DEGREES[-1]   # element and edge rule degree of a drive (oscillating integrands)


def _read_only(array) -> np.ndarray:
    array = np.ascontiguousarray(array)
    array.flags.writeable = False
    return array


def _form(axisymmetric, azimuthal_order) -> tuple:
    """(axisymmetric, order) checked: an order m >= 1 needs the axisymmetric form."""
    order = _check_azimuthal_order(azimuthal_order) if azimuthal_order else 0
    if order and not axisymmetric:
        raise ValueError(f'elastic drive: azimuthal order {order} needs axisymmetric=True')
    return bool(axisymmetric), order


def _components(order) -> int:
    return AZIMUTHAL_COMPONENTS if order else COMPONENTS


def _rows(axisymmetric, order) -> int:
    return AZIMUTHAL_ROWS if order else strain_rows(axisymmetric)


def _ring_factor(axisymmetric, order, radius):
    """Ring weight of edge points: 1 planar, 2 pi r (m = 0), pi r (m >= 1) (fem.elasticity's RING / AZIMUTHAL_RING)."""
    if not axisymmetric:
        return 1.0
    return (AZIMUTHAL_RING if order else RING) * radius


@dataclass(frozen=True, eq=False)
class DrivePoints:
    """Quadrature points of a stress drive on one solid dof map: per element chunk (rule group) the positions into
    dofs.elements and its TriangleRule; flattened in the order (group, element, point) the physical points, the
    weights (w det J; x 2 pi r axisymmetric; x pi r at an azimuthal order m >= 1) and the mesh element of each point.
    Arrays are read-only (shared by every scan position)."""
    dofs: object
    axisymmetric: bool
    azimuthal_order: int
    groups: tuple            # ((positions int64 (m,), TriangleRule), ...)
    offsets: np.ndarray      # int64 (n_groups + 1,): start of each group in the flat arrays
    points: np.ndarray       # (n, 2) [m]
    weights: np.ndarray      # (n,) [m^2] planar per unit depth, [m^3] axisymmetric (ring included)
    elements: np.ndarray     # int64 (n,) mesh element of each point
    # kept by assemble_stress_drive(keep_operator=True) on first use, per group: (B float64 (m, q, rows, c n_local),
    # vector dofs int64 (m, c n_local)), i.e. n_points x rows x components x n_local x 8 B + n_elements x components x
    # n_local x 8 B (n_local 3 P1 / 6 P2; rows x components 3 x 2 planar, 4 x 2 m = 0, 6 x 3 m >= 1); dropped with the
    # points
    kept: list = field(default_factory=list, init=False, repr=False)

    @property
    def n_points(self) -> int:
        return len(self.weights)

    @property
    def rows(self) -> int:
        """Voigt rows of the stress: 3 planar, 4 axisymmetric m = 0, 6 at m >= 1."""
        return _rows(self.axisymmetric, self.azimuthal_order)

    @property
    def components(self) -> int:
        """Displacement components per scalar dof (2, or 3 at m >= 1): the drive has components x n_dofs entries."""
        return _components(self.azimuthal_order)


@dataclass(frozen=True, eq=False)
class EdgeDrivePoints:
    """Gauss points of a traction drive on mesh edges of one solid dof map (edges without a map element dropped):
    side elements (mesh.edge_elements: side 0, side 1 or -1), physical points, unit normals out of side 0 and weights
    w |dx/dt| (x 2 pi r axisymmetric, x pi r at m >= 1). Arrays are read-only."""
    dofs: object
    axisymmetric: bool
    azimuthal_order: int
    edges: np.ndarray        # int64 (k,) mesh edges
    sides: np.ndarray        # int64 (k, 2) side 0 / side 1 mesh elements (-1: an outer edge)
    points: np.ndarray       # (k, q, 2) [m]
    normals: np.ndarray      # (k, q, 2) unit, out of side 0
    weights: np.ndarray      # (k, q) [m] planar per unit depth, [m^2] axisymmetric (ring included)

    @property
    def components(self) -> int:
        return _components(self.azimuthal_order)


def _element_mask(dofs, elements):
    """Bool per mesh element of a drive selection (None = every map element): a bool mask or mesh element indices."""
    if elements is None:
        return None
    elements = np.asarray(elements)
    n_elements = dofs.mesh.n_elements
    if elements.dtype == bool:
        if elements.shape != (n_elements,):
            raise ValueError(f'drive_points: element mask has shape {elements.shape}, expected ({n_elements},)')
        return elements
    indices = elements.astype(np.int64).ravel()
    if indices.size and (indices.min() < 0 or indices.max() >= n_elements):
        raise ValueError(f'drive_points: element indices outside 0..{n_elements - 1}')
    mask = np.zeros(n_elements, dtype=bool)
    mask[indices] = True
    return mask


def _point_geometry(dofs, positions, rule, axisymmetric, order) -> tuple:
    """(physical points (m, q, 2), weights (m, q)) of fem.elasticity's form at a rule (those of
    elasticity._form_operator_at, without building B)."""
    if order:
        geometry, _, _, weights, _ = _azimuthal_terms(dofs, positions, rule, None, None)
        return geometry.points, weights
    geometry = element_geometry(dofs.mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
    return geometry.points, _weights(geometry, rule, axisymmetric)


def drive_points(dofs, *, axisymmetric=False, azimuthal_order=0, elements=None) -> DrivePoints:
    """DrivePoints of the map's elements inside `elements` (bool per mesh element or mesh element indices; None =
    every map element) at the rule of DRIVE_DEGREE (curved elements >= CURVED_DEGREE). ValueError when the selection
    holds no map element (a drive with no points is a wiring error)."""
    axisymmetric, order = _form(axisymmetric, azimuthal_order)
    mask = _element_mask(dofs, elements)
    groups, points, weights, element_ids, offsets = [], [], [], [], [0]
    for positions, rule in _element_groups(dofs, DRIVE_DEGREE):
        if mask is not None:
            positions = positions[mask[dofs.elements[positions]]]
            if not positions.size:
                continue
        group_points, group_weights = _point_geometry(dofs, positions, rule, axisymmetric, order)
        groups.append((_read_only(positions), rule))
        points.append(group_points.reshape(-1, 2))
        weights.append(group_weights.reshape(-1))
        element_ids.append(np.repeat(dofs.elements[positions], len(rule.weights)))
        offsets.append(offsets[-1] + group_weights.size)
    if not groups:
        raise ValueError('drive_points: the drive selection holds no element of the solid dof map')
    return DrivePoints(dofs=dofs, axisymmetric=axisymmetric, azimuthal_order=order, groups=tuple(groups),
                       offsets=_read_only(np.asarray(offsets, dtype=np.int64)),
                       points=_read_only(np.concatenate(points)), weights=_read_only(np.concatenate(weights)),
                       elements=_read_only(np.concatenate(element_ids).astype(np.int64)))


def _checked_map(dofs, points) -> None:
    """ValueError when the (edge) drive points belong to another dof map."""
    if points.dofs is not dofs:
        raise ValueError('elastic drive: the points belong to another dof map (rebuild them after a re-assembly)')


def _operators(dofs, points):
    """(B (m, q, k, c n), vector dofs (m, c n)) per group of DrivePoints, one group at a time (module header)."""
    for positions, rule in points.groups:
        operator = _form_operator_at(dofs, positions, rule, points.axisymmetric, points.azimuthal_order)[1]
        yield operator, component_dofs(dofs.element_dofs[positions], points.components)


def assemble_stress_drive(dofs, points, stress, *, keep_operator=False) -> np.ndarray:
    """Complex128 (c n,) of int B^T sigma dV (c = points.components, the map's vector numbering c d + component) for
    a complex stress (points.rows, points.n_points) at the DrivePoints in their Voigt rows (module header);
    keep_operator: build B of every group once and keep it on the points for the next drives (DrivePoints.kept;
    else B is built per group and dropped). ValueError for points of another map."""
    _checked_map(dofs, points)
    rows, components = points.rows, points.components
    stress = np.asarray(stress)
    if stress.shape != (rows, points.n_points):
        raise ValueError(f'assemble_stress_drive: stress has shape {stress.shape}, expected ({rows}, '
                         f'{points.n_points}) (Voigt rows x drive points)')
    if not np.isfinite(stress).all():
        raise ValueError('assemble_stress_drive: the stress holds non-finite values')
    if keep_operator and not points.kept:
        built = list(_operators(dofs, points))   # whole, so an interrupted build keeps nothing
        points.kept.extend(built)
        del built
    operators = points.kept or _operators(dofs, points)
    result = np.zeros(components * dofs.n_dofs, dtype=np.complex128)
    for index, ((positions, rule), (operator, vector_dofs)) in enumerate(zip(points.groups, operators)):
        start, stop = int(points.offsets[index]), int(points.offsets[index + 1])
        count, n_rule = len(positions), len(rule.weights)
        weighted = (stress[:, start:stop] * points.weights[start:stop]).T.reshape(count, n_rule, rows)
        local = np.einsum('mqki,mqk->mi', operator, weighted)
        scatter_add(result, vector_dofs, local)
    return result


def edge_drive_points(dofs, edges, *, axisymmetric=False, azimuthal_order=0) -> EdgeDrivePoints:
    """EdgeDrivePoints of mesh edges (int64; those without an element of the map dropped) at the Gauss rule of
    DRIVE_DEGREE (+ 2 on geometry order 2): sides, points, weights and normals by indicators.edge_sides with side 0 =
    mesh.edge_elements[:, 0] (the points of assembly._edge_quadrature: module header)."""
    axisymmetric, order = _form(axisymmetric, azimuthal_order)
    kept = _edge_sides(dofs, np.asarray(edges, dtype=np.int64).ravel())[0]
    mesh = dofs.mesh
    sides = edge_sides(mesh, kept, np.ones(mesh.n_elements, dtype=bool), dofs.geometry_order, DRIVE_DEGREE)
    weights = sides.weights * _ring_factor(axisymmetric, order, sides.points[..., 0])
    return EdgeDrivePoints(dofs=dofs, axisymmetric=axisymmetric, azimuthal_order=order, edges=_read_only(kept),
                           sides=_read_only(sides.elements), points=_read_only(sides.points),
                           normals=_read_only(sides.normals), weights=_read_only(weights))


def assemble_edge_drive(dofs, edge_points, traction) -> np.ndarray:
    """Complex128 (c n,) of int N t ds over the EdgeDrivePoints of a complex traction (k, q, d) [Pa] at their points:
    d = 2 in-plane (t_x, t_y) / (t_r, t_z) (at m >= 1 t_phi = 0), or d = 3 (t_r, t_z, t_phi) at m >= 1 (the cos /
    cos / sin amplitudes of the order). ValueError for points of another map."""
    _checked_map(dofs, edge_points)
    order = edge_points.azimuthal_order
    components = edge_points.components
    traction = np.asarray(traction)
    shape = edge_points.weights.shape
    if traction.shape[:2] != shape or traction.ndim != 3 or traction.shape[2] not in ((2, 3) if order else (2,)):
        raise ValueError(f'assemble_edge_drive: traction has shape {traction.shape}, expected {shape + (2,)}'
                         + (f' or {shape + (3,)}' if order else ''))
    if not np.isfinite(traction).all():
        raise ValueError('assemble_edge_drive: the traction holds non-finite values')
    result = np.zeros(components * dofs.n_dofs, dtype=np.complex128)
    if not len(edge_points.edges):
        return result
    _, element_dofs, values, _, _, _ = _edge_quadrature(dofs, edge_points.edges, DRIVE_DEGREE)
    local = np.einsum('kqn,kqd,kq->knd', values, traction, edge_points.weights)
    if local.shape[2] != components:   # in-plane traction at m >= 1: no U_phi part
        padded = np.zeros(local.shape[:2] + (components,), dtype=local.dtype)
        padded[..., :local.shape[2]] = local
        local = padded
    scatter_add(result, component_dofs(element_dofs, components), local.reshape(len(local), -1))
    return result
