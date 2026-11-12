"""Plate bending of the membrane on Lagrange P2 triangles (new_modules_plan S7): the C0 interior penalty form of
D del^4 w on the isoparametric P2 map (Hessian volume term, normal-moment / slope-jump edge terms, Nitsche slope holds
on clamped edges), its fem.pml stretched form and the Bloch seam edges of a periodic cell. Host numpy, scipy inside
functions."""
# Energy sum_K int M(w) : H(v) dA with M(w) = D [(1 - nu) H(w) + nu tr H(w) I] (D = bending_stiffness (1 + j eta),
# nu = plate_poisson_ratio), minus sum_e int {M_nn(w)} [dv/dn] + {M_nn(v)} [dw/dn] ds, plus sigma_e int [dw/dn] [dv/dn]
# ds; [dv/dn] = the sum of both sides' outward normal derivatives, {.} the side mean (a one-sided edge: the side
# itself). Interior edges of the active set carry the edge terms; a clamped edge carries them one-sided on each active
# side (dw/dn = 0 weakly, Nitsche); free and simply supported edges none (natural moment condition). sigma_e =
# PLATE_PENALTY x D of the stiffer side / the smaller side height 2 A / |e| (coercive above 6 on straight P2: Hessians
# constant there). Consistent for a C1 exact solution: energy error O(h), eigenvalues O(h^2) on P2. Chosen over Morley
# (does not converge for the tension term) and DKT (no w field for the tension / mass terms): the P2 Lagrange space of
# the membrane stays, with its holes, curved outlines, Bloch masters and probes. Unconjugated (complex symmetric).
# Bloch seam (periodic axis, phase p = e^{-j k L}): an element A on the hi side sees across the seam the copy of the
# lo-side partner element B carrying p u_B, so the full-numbering seam matrix holds S_AA, S_AB p, S_BA conj(p), S_BB and
# its reduction P^H S P is the Galerkin term of the Bloch cell.
# PML (plan F1; fem.pml stretch s_i(x_i), d/dx_i -> (1 / s_i) d/dx_i, dA -> J dA, J = s_x s_y): the same form in the
# stretched coordinates, unconjugated (complex symmetric). Stretched Hessian H~_ij = H_ij / (s_i s_j) - delta_ij (s_i' /
# s_i^3) dw/dx_i, volume int M~(w) : H~(v) J dA with M~ = M of H~; integrating it by parts per element gives the edge
# moment J n . S^-1 M~ S^-1 n against the physical slope jump [dv/dn] (tangential parts cancel across an edge), and the
# penalty is the analytic continuation of sigma_e [dw/dn~][dv/dn~] ds~ with the stretched height h~ = s_n h: sigma_e J
# (n . S^-2 n)^2 (= J / s_n^4 on axis-aligned edges). Ring elements take PML_EXTRA_DEGREE more quadrature (as
# fem.harmonic). Every function takes `stretch` = a fem.pml.PMLStretch or None (identical to the plain form).
# Straight elements (affine map) have constant Hessians G^T H_ref G (G = J^-1): their volume term is one point (area x
# H : M) and their edge sides take the Hessians once per element instead of per point (the same values up to round-off;
# the batched per-point matmuls / einsums of p2_derivatives were most of the assembly). The stretch is 1 (s' = 0) inside
# the window, so only PML ring elements and edges touching the ring take the stretched form.
import numpy as np

from ...fem.assembly import _to_csr
from ...fem.elements import REFERENCE_GRADIENTS, curved_elements, lagrange_basis, lagrange_values, physical_gradients
from ...fem.harmonic import PML_EXTRA_DEGREE
from ...fem.indicators import edge_sides
from ...fem.mesh import PERIODIC_TOLERANCE, edge_keys, periodic_node_partners
from ...fem.quadrature import CURVED_DEGREE, TRIANGLE_DEGREES, triangle_rule

PLATE_PENALTY = 16.0     # sigma_e x side height / D (> 6 on straight P2 elements; first estimate)
EDGE_DEGREE = 4          # Gauss degree along edges (+2 on curved geometry inside edge_sides)
CHUNK_ITEMS = 16384      # elements / edges per evaluation block


def _reference_hessians() -> np.ndarray:
    """(6, 2, 2) constant reference Hessians of the P2 basis in fem.elements' local order (corners, then midsides)."""
    gradients = REFERENCE_GRADIENTS
    hessians = np.zeros((6, 2, 2))
    for corner in range(3):
        hessians[corner] = 4.0 * np.outer(gradients[corner], gradients[corner])
    for edge in range(3):
        first, second = (edge + 1) % 3, (edge + 2) % 3
        hessians[3 + edge] = 4.0 * (np.outer(gradients[first], gradients[second])
                                    + np.outer(gradients[second], gradients[first]))
    return hessians


P2_REFERENCE_HESSIANS = _reference_hessians()


def straight_geometry(mesh, elements) -> tuple:
    """(inverse Jacobians G = J^-1 (m, 2, 2), det J (m,)) of the affine corner map of mesh elements (m,)."""
    corners = mesh.points[mesh.triangles[np.asarray(elements, dtype=np.int64)]]
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]     # J columns dx / dxi, dx / deta
    determinant = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
    if not (determinant > 0.0).all():
        raise ValueError(f'plate bending: {int((determinant <= 0.0).sum())} elements have a non-positive Jacobian')
    inverse = np.empty((len(corners), 2, 2))
    inverse[:, 0, 0] = second[:, 1] / determinant
    inverse[:, 0, 1] = -second[:, 0] / determinant
    inverse[:, 1, 0] = -first[:, 1] / determinant
    inverse[:, 1, 1] = first[:, 0] / determinant
    return inverse, determinant


def straight_hessians(inverse) -> np.ndarray:
    """(m, 6, 2, 2) constant physical Hessians H_cf = sum_de G_dc H_ref,de G_ef of the P2 basis on straight elements
    with inverse Jacobians G (m, 2, 2) (written out per component: no batched matmul)."""
    hessians = np.empty((len(inverse), 6, 2, 2))
    for c in range(2):
        for f in range(c, 2):
            component = np.zeros((len(inverse), 6))
            for d in range(2):
                for e in range(2):
                    component += (inverse[:, d, c] * inverse[:, e, f])[:, None] * P2_REFERENCE_HESSIANS[None, :, d, e]
            hessians[:, :, c, f] = component
            hessians[:, :, f, c] = component
    return hessians


def p2_derivatives(mesh, elements, reference_points) -> tuple:
    """(physical gradients (m, q, 6, 2), Hessians (m, q, 6, 2, 2), det J (m, q)) of the isoparametric P2 basis (corners +
    mesh.midside_points: affine on straight elements) of mesh elements (m,) at reference points (q, 2) or (m, q, 2)."""
    elements = np.asarray(elements, dtype=np.int64)
    reference_points = np.asarray(reference_points, dtype=np.float64)   # (q, 2) shared: broadcast by the matmuls
    nodes = np.concatenate([mesh.points[mesh.triangles[elements]],
                            mesh.midside_points[mesh.element_edges[elements]]], axis=1)   # (m, 6, 2)
    # about corner 0: basis gradients and reference Hessians sum to 0, so J and the map Hessians are cancelling sums;
    # absolute coordinates left a round-off residue ~eps |x| (relative ~1e-13 in the Hessians of straight elements)
    nodes = nodes - nodes[:, :1]
    _, reference_gradients = lagrange_basis(2, reference_points)                          # ([m,] q, 6, 2)
    jacobian = np.matmul(nodes.swapaxes(1, 2)[:, None], reference_gradients)             # (m, q, 2, 2)
    determinant = jacobian[..., 0, 0] * jacobian[..., 1, 1] - jacobian[..., 0, 1] * jacobian[..., 1, 0]
    if not (determinant > 0.0).all():
        raise ValueError(f'plate bending: {int((determinant <= 0.0).any(axis=-1).sum())} elements have a non-positive '
                         'Jacobian (folded curved element: refine near the curve)')
    inverse = np.empty_like(jacobian)
    inverse[..., 0, 0] = jacobian[..., 1, 1] / determinant
    inverse[..., 1, 1] = jacobian[..., 0, 0] / determinant
    inverse[..., 0, 1] = -jacobian[..., 0, 1] / determinant
    inverse[..., 1, 0] = -jacobian[..., 1, 0] / determinant
    gradients = physical_gradients(reference_gradients, inverse)                         # (m, q, 6, 2)
    # d^2 x_k / dxi_d dxi_e per element, (m, k, d e)
    map_hessians = nodes.swapaxes(1, 2) @ P2_REFERENCE_HESSIANS.reshape(len(P2_REFERENCE_HESSIANS), -1)
    corrected = P2_REFERENCE_HESSIANS[None, None] - np.matmul(gradients, map_hessians[:, None]).reshape(
        gradients.shape[:3] + (2, 2))
    hessians = np.einsum('mqdc,mqade,mqef->mqacf', inverse, corrected, inverse, optimize=True)
    return gradients, hessians, determinant


def p2_points(mesh, elements, reference_points) -> np.ndarray:
    """(m, q, 2) physical points [m] of the isoparametric P2 map of mesh elements (m,) at reference points (q, 2) or
    (m, q, 2) (as p2_derivatives)."""
    elements = np.asarray(elements, dtype=np.int64)
    nodes = np.concatenate([mesh.points[mesh.triangles[elements]],
                            mesh.midside_points[mesh.element_edges[elements]]], axis=1)   # (m, 6, 2)
    return lagrange_values(2, np.asarray(reference_points, dtype=np.float64)) @ nodes     # ([m,] q, 6) @ (m, 6, 2)


def stretch_terms(stretch, points):
    """(1 / s (m, q, 2), s' / s^3 (m, q, 2), J = s_x s_y (m, q)) of a fem.pml.PMLStretch at points (m, q, 2) [m]; None
    without a stretch (stretch None)."""
    if stretch is None:
        return None
    stretches = stretch.stretches(points)
    inverse = 1.0 / stretches
    return inverse, stretch.stretch_derivatives(points) * inverse ** 3, np.prod(stretches, axis=-1)


def stretched_hessians(gradients, hessians, terms) -> np.ndarray:
    """(m, q, 6, 2, 2) H~_ij = H_ij / (s_i s_j) - delta_ij (s_i' / s_i^3) dphi/dx_i of basis gradients (m, q, 6, 2) and
    Hessians at points with stretch_terms `terms`; the Hessians unchanged when terms is None."""
    if terms is None:
        return hessians
    inverse, curvature, _ = terms
    stretched = hessians * inverse[:, :, None, :, None] * inverse[:, :, None, None, :]
    diagonal = np.arange(2)
    stretched[..., diagonal, diagonal] -= curvature[:, :, None, :] * gradients
    return stretched


def _moments(hessians, rigidity, poisson) -> np.ndarray:
    """(m, q, 6, 2, 2) M(phi_a) = D [(1 - nu) H + nu tr(H) I] with D, nu per element (m,)."""
    trace = hessians[..., 0, 0] + hessians[..., 1, 1]
    nu = np.asarray(poisson, dtype=np.float64)[:, None, None, None, None]
    moments = (1.0 - nu) * hessians + nu * trace[..., None, None] * np.eye(2)
    return np.asarray(rigidity)[:, None, None, None, None] * moments


def _check_inputs(dofs, rigidity, poisson) -> tuple:
    if dofs.family != 'lagrange' or dofs.order != 2:
        raise ValueError('plate bending needs a P2 Lagrange dof map (discretization quadratic)')
    n_elements = dofs.mesh.n_elements
    rigidity = np.asarray(rigidity)
    poisson = np.broadcast_to(np.asarray(poisson, dtype=np.float64), (n_elements,))
    if rigidity.shape != (n_elements,):
        raise ValueError(f'plate bending: rigidity must have one value per mesh element ({n_elements}), got '
                         f'{rigidity.shape}')
    return rigidity, poisson


def _straight_areas(mesh, elements) -> np.ndarray:
    corners = mesh.points[mesh.triangles[elements]]
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    return 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])


def side_values(mesh, elements, reference, normals, rigidity, poisson, terms=None) -> tuple:
    """(outward normal derivatives (k, q, 6), normal moments (k, q, 6)) of the P2 basis of elements (k,) at their
    reference points (k, q, 2) with unit normals (k, q, 2) out of those elements; the moment is M_nn, or with
    stretch_terms `terms` of the edge points the stretched J n . S^-1 M~ S^-1 n (module header)."""
    elements = np.asarray(elements, dtype=np.int64)
    if terms is not None:
        return _isoparametric_side_values(mesh, elements, reference, normals, rigidity, poisson, terms)
    straight = ~curved_elements(mesh, elements)
    if straight.all():
        return _straight_side_values(mesh, elements, reference, normals, rigidity, poisson)
    if not straight.any():
        return _isoparametric_side_values(mesh, elements, reference, normals, rigidity, poisson, None)
    slopes = np.empty(normals.shape[:2] + (6,))
    moments = np.empty(normals.shape[:2] + (6,), dtype=np.result_type(rigidity, np.float64))
    for rows, evaluate in ((straight, _straight_side_values), (~straight, _isoparametric_side_values)):
        slopes[rows], moments[rows] = evaluate(mesh, elements[rows], reference[rows], normals[rows], rigidity, poisson)
    return slopes, moments


def _straight_side_values(mesh, elements, reference, normals, rigidity, poisson) -> tuple:
    """side_values of straight elements without a stretch: gradients per point, Hessians once per element."""
    inverse, _ = straight_geometry(mesh, elements)
    _, reference_gradients = lagrange_basis(2, reference)                                # (k, q, 6, 2)
    gradients = physical_gradients(reference_gradients, inverse[:, None])                # (k, q, 6, 2)
    normal_x, normal_y = normals[..., 0, None], normals[..., 1, None]                    # (k, q, 1)
    slopes = gradients[..., 0] * normal_x + gradients[..., 1] * normal_y
    hessians = straight_hessians(inverse)[:, None]                                       # (k, 1, 6, 2, 2)
    normal_curvature = (hessians[..., 0, 0] * (normal_x * normal_x) + 2.0 * hessians[..., 0, 1] * (normal_x * normal_y)
                        + hessians[..., 1, 1] * (normal_y * normal_y))
    trace = hessians[..., 0, 0] + hessians[..., 1, 1]
    nu = poisson[elements][:, None, None]
    moments = (1.0 - nu) * normal_curvature + nu * trace * (normal_x * normal_x + normal_y * normal_y)
    return slopes, rigidity[elements][:, None, None] * moments


def _isoparametric_side_values(mesh, elements, reference, normals, rigidity, poisson, terms=None) -> tuple:
    """side_values on the isoparametric P2 map (curved elements, stretched edges): everything per point."""
    gradients, hessians, _ = p2_derivatives(mesh, elements, reference)
    slopes = np.matmul(gradients, normals[..., None])[..., 0]
    hessians = stretched_hessians(gradients, hessians, terms)
    directions = normals if terms is None else normals * terms[0]
    normal_curvature = np.einsum('kqacf,kqc,kqf->kqa', hessians, directions, directions, optimize=True)
    trace = hessians[..., 0, 0] + hessians[..., 1, 1]
    nu = poisson[elements][:, None, None]
    moments = (1.0 - nu) * normal_curvature + nu * trace * np.einsum('kqc,kqc->kq', directions, directions)[..., None]
    if terms is not None:
        moments = moments * terms[2][..., None]
    return slopes, rigidity[elements][:, None, None] * moments


def penalty_scale(terms, normals):
    """(k, q) stretched penalty factor J (n . S^-2 n)^2 of stretch_terms at edge points with unit normals (k, q, 2);
    None without a stretch (terms None)."""
    if terms is None:
        return None
    directions = normals * terms[0]
    return terms[2] * np.einsum('kqc,kqc->kq', directions, directions) ** 2


def _point_blocks(test_values, trial_values, weights) -> np.ndarray:
    """(m, n, n) sum over points and trailing components of weights (m, q) x test_i . trial_j, values (m, q, n, ...):
    weights folded into the test values, then one batched matmul over (point, component)."""
    count, points, n = test_values.shape[:3]
    test = np.multiply(test_values.reshape(count, points, n, -1).swapaxes(-1, -2), weights[:, :, None, None],
                       order='C').reshape(count, -1, n)
    trial = np.ascontiguousarray(trial_values.reshape(count, points, n, -1).swapaxes(-1, -2)).reshape(count, -1, n)
    return test.swapaxes(1, 2) @ trial


def edge_blocks(slopes, moments, weights, penalty, scale=None) -> np.ndarray:
    """(k, 6 s, 6 s) edge matrices of s sides (lists of (k, q, 6) per side): -{M_nn(w)} [dv/dn] - {M_nn(v)} [dw/dn] +
    sigma [dw/dn] [dv/dn] (test rows, trial columns); scale (k, q) = penalty_scale of a stretch (None: 1)."""
    jumps = np.concatenate(slopes, axis=-1)
    means = np.concatenate(moments, axis=-1) / float(len(moments))
    consistency = _point_blocks(jumps, means, weights)
    penalty_weights = weights if scale is None else weights * scale
    return (-consistency - np.swapaxes(consistency, 1, 2)
            + _point_blocks(jumps, jumps, penalty[:, None] * penalty_weights))


def _penalty(mesh, elements_per_side, lengths, rigidity) -> np.ndarray:
    """sigma_e (k,) = PLATE_PENALTY x D of the side with the largest |D| / the smallest side height 2 A / |e|."""
    rigidities = np.stack([rigidity[elements] for elements in elements_per_side], axis=1)          # (k, s)
    heights = np.stack([2.0 * _straight_areas(mesh, elements) / lengths for elements in elements_per_side], axis=1)
    stiffest = np.take_along_axis(rigidities, np.argmax(np.abs(rigidities), axis=1)[:, None], axis=1)[:, 0]
    return PLATE_PENALTY * stiffest / heights.min(axis=1)


def _local_dofs(dofs, elements) -> np.ndarray:
    positions = dofs.element_position[elements]
    if (positions < 0).any():
        raise ValueError('plate bending: an edge side element lies outside the dof map')
    return dofs.element_dofs[positions]


def _append(rows, cols, values, local, blocks) -> None:
    if local.max(initial=0) < 2 ** 31:   # int32 COO indices: half the index traffic of the CSR conversion
        local = local.astype(np.int32)
    rows.append(np.broadcast_to(local[:, :, None], blocks.shape).ravel())
    cols.append(np.broadcast_to(local[:, None, :], blocks.shape).ravel())
    values.append(blocks)


def _volume_groups(mesh, elements, stretch) -> tuple:
    """((element subset, rule degree, curved, in the ring), ...): straight 2, curved CURVED_DEGREE, + PML_EXTRA_DEGREE
    on PML ring elements (pml_faces != 0) when stretched (only those take the stretch: s = 1 inside the window)."""
    curved = curved_elements(mesh, elements)
    ring = (mesh.pml_faces[elements] != 0) if stretch is not None else np.zeros(len(elements), dtype=bool)
    groups = []
    for in_ring in (False, True):
        for is_curved, degree in ((False, 2), (True, CURVED_DEGREE)):
            mask = (curved == is_curved) & (ring == in_ring)
            if mask.any():
                extra = PML_EXTRA_DEGREE if in_ring else 0
                groups.append((elements[mask], min(degree + extra, TRIANGLE_DEGREES[-1]), is_curved, in_ring))
    return tuple(groups)


def _ring_parts(items, touches_ring) -> tuple:
    """((items, stretched), ...): the items not touching the PML ring (taken unstretched) and those touching it;
    touches_ring None (no stretch): all unstretched."""
    if touches_ring is None:
        return ((items, False),)
    return tuple((items[mask], stretched) for mask, stretched in ((~touches_ring, False), (touches_ring, True))
                 if mask.any())


def _volume_terms(mesh, chunk, rule, curved, stretch) -> tuple:
    """(Hessians (m, q, 6, 2, 2), weights (m, q)) of the volume term on elements chunk (m,): one point of constant
    Hessians on straight unstretched elements (area x H : M), else the rule's points (stretched with stretch)."""
    if not curved and stretch is None:
        inverse, determinant = straight_geometry(mesh, chunk)
        return straight_hessians(inverse)[:, None], (determinant * float(rule.weights.sum()))[:, None]
    gradients, hessians, determinant = p2_derivatives(mesh, chunk, rule.points)
    weights = determinant * rule.weights[None, :]
    if stretch is not None:
        terms = stretch_terms(stretch, p2_points(mesh, chunk, rule.points))
        hessians = stretched_hessians(gradients, hessians, terms)
        weights = weights * terms[2]
    return hessians, weights


def assemble_plate_bending(dofs, rigidity, poisson=0.0, clamped_edges=(), stretch=None):
    """CSR (n_dofs, n_dofs) in the DofMap's numbering (P2 Lagrange on its elements, the active set) of the C0 interior
    penalty bending form (module header): rigidity D (n_elements,) [N m], real or complex D (1 + j eta); poisson nu
    (number or (n_elements,)); clamped_edges = mesh edges whose slope dw/dn is held (Nitsche, each active side);
    stretch = a fem.pml.PMLStretch of the mesh's ring (complex symmetric stretched form) or None."""
    rigidity, poisson = _check_inputs(dofs, rigidity, poisson)
    mesh = dofs.mesh
    inside = dofs.element_position >= 0
    rows, cols, values = [], [], []
    for selected, degree, curved, in_ring in _volume_groups(mesh, dofs.elements, stretch):
        rule = triangle_rule(degree)
        for start in range(0, len(selected), CHUNK_ITEMS):
            chunk = selected[start:start + CHUNK_ITEMS]
            hessians, weights = _volume_terms(mesh, chunk, rule, curved, stretch if in_ring else None)
            moments = _moments(hessians, rigidity[chunk], poisson[chunk])
            _append(rows, cols, values, _local_dofs(dofs, chunk), _point_blocks(hessians, moments, weights))
    ring = (mesh.pml_faces != 0) if stretch is not None else None
    clamped = np.unique(np.asarray(clamped_edges, dtype=np.int64))
    pairs = mesh.edge_elements.astype(np.int64)
    two_sided = (pairs[:, 1] >= 0) & inside[pairs[:, 0]] & inside[np.maximum(pairs[:, 1], 0)]
    two_sided[clamped] = False
    interior = np.flatnonzero(two_sided)
    touches = None if ring is None else ring[pairs[interior, 0]] | ring[pairs[interior, 1]]
    for part, stretched in _ring_parts(interior, touches):
        for start in range(0, len(part), CHUNK_ITEMS):
            sides = edge_sides(mesh, part[start:start + CHUNK_ITEMS], inside, 2, EDGE_DEGREE)
            first, second = sides.elements[:, 0], sides.elements[:, 1]
            terms = stretch_terms(stretch, sides.points) if stretched else None
            side_0 = side_values(mesh, first, sides.reference_points(0), sides.normals, rigidity, poisson, terms)
            side_1 = side_values(mesh, second, sides.reference_points(1), -sides.normals, rigidity, poisson, terms)
            penalty = _penalty(mesh, (first, second), sides.lengths, rigidity)
            blocks = edge_blocks([side_0[0], side_1[0]], [side_0[1], side_1[1]], sides.weights, penalty,
                                 penalty_scale(terms, sides.normals))
            _append(rows, cols, values, np.concatenate([_local_dofs(dofs, first), _local_dofs(dofs, second)],
                                                       axis=1), blocks)
    if len(clamped):
        sides = edge_sides(mesh, clamped, inside, 2, EDGE_DEGREE)
        for side, sign in ((0, 1.0), (1, -1.0)):
            rows_of_side = np.flatnonzero(sides.elements[:, side] >= 0)
            touches = None if ring is None else ring[sides.elements[rows_of_side, side]]
            for part, stretched in _ring_parts(rows_of_side, touches):
                for start in range(0, len(part), CHUNK_ITEMS):
                    chunk = part[start:start + CHUNK_ITEMS]
                    elements_of_side = sides.elements[chunk, side]
                    normals = sign * sides.normals[chunk]
                    terms = stretch_terms(stretch, sides.points[chunk]) if stretched else None
                    slopes, moments = side_values(mesh, elements_of_side, sides.reference_points(side, chunk),
                                                  normals, rigidity, poisson, terms)
                    penalty = _penalty(mesh, (elements_of_side,), sides.lengths[chunk], rigidity)
                    _append(rows, cols, values, _local_dofs(dofs, elements_of_side),
                            edge_blocks([slopes], [moments], sides.weights[chunk], penalty,
                                        penalty_scale(terms, normals)))
    return _to_csr(rows, cols, values, (dofs.n_dofs, dofs.n_dofs))


def seam_edge_pairs(mesh, axis, inside) -> tuple:
    """(hi-side edges, their lo-side partner edges, reversed (k,) bool) of the periodic seam of axis 0 / 1 whose two
    elements are both in the bool element set inside; reversed: the partner edge runs the other way (lower -> higher
    node), so its Gauss points come in reverse order."""
    partners = periodic_node_partners(mesh, axis)
    edges = mesh.edges.astype(np.int64)
    hi_edges = np.flatnonzero((partners[edges] >= 0).all(axis=1))
    images = partners[edges[hi_edges]]                       # (k, 2) lo-side nodes of (lower, higher)
    keys = edge_keys(edges, mesh.n_points)                   # ascending: edges are numbered by key
    wanted = edge_keys(images, mesh.n_points)
    found = np.minimum(np.searchsorted(keys, wanted), max(len(keys) - 1, 0))
    if len(wanted) and (keys[found] != wanted).any():
        raise ValueError(f"plate bending: {int((keys[found] != wanted).sum())} seam edges of periodic {'xy'[axis]} "
                         'have no partner edge (the two sides are meshed differently)')
    reversed_order = images[:, 0] > images[:, 1]
    element_hi = mesh.edge_elements[hi_edges, 0].astype(np.int64)
    element_lo = mesh.edge_elements[found, 0].astype(np.int64)
    keep = inside[element_hi] & inside[element_lo]
    return hi_edges[keep], found[keep].astype(np.int64), reversed_order[keep]


def seam_phase(mesh, axis, wavenumber) -> complex:
    """e^{-j k L} of the node extent L along axis (as fem.dofs.bloch_masters: a phase within 1e-15 of real is real)."""
    points = mesh.points
    extent = float(points[:, axis].max() - points[:, axis].min())
    step = complex(np.exp(-1j * float(wavenumber) * extent))
    if abs(step.imag) <= 1e-15:
        step = complex(np.sign(step.real), 0.0)
    return step


def assemble_seam_bending(dofs, rigidity, poisson, axis, phase, stretch=None):
    """Complex CSR (n_dofs, n_dofs) in the DofMap's numbering of the bending edge terms across the periodic seam of
    axis 0 / 1 with the Bloch step phase = e^{-j k L} (module header); reduce it with the Bloch prolongation P^H S P.
    stretch: the PMLStretch of rings on the other axis' faces (never on the periodic faces) or None."""
    rigidity, poisson = _check_inputs(dofs, rigidity, poisson)
    mesh = dofs.mesh
    inside = dofs.element_position >= 0
    hi_edges, lo_edges, reversed_order = seam_edge_pairs(mesh, axis, inside)
    rows, cols, values = [], [], []
    phase = complex(phase)
    factors = np.ones((12, 12), dtype=np.complex128)
    factors[:6, 6:] = phase            # test A, trial B' = phase u_B
    factors[6:, :6] = np.conj(phase)   # test B' = conj(phase) v_B
    for start in range(0, len(hi_edges), CHUNK_ITEMS):
        hi = edge_sides(mesh, hi_edges[start:start + CHUNK_ITEMS], inside, 2, EDGE_DEGREE)
        lo = edge_sides(mesh, lo_edges[start:start + CHUNK_ITEMS], inside, 2, EDGE_DEGREE)
        flip = reversed_order[start:start + CHUNK_ITEMS]
        if len(hi.edges) != len(lo.edges):
            raise ValueError('plate bending: seam edges lost an active side (the cell is not periodic)')
        first, second = hi.elements[:, 0], lo.elements[:, 0]
        terms_a = stretch_terms(stretch, hi.points)
        side_a = side_values(mesh, first, hi.reference_points(0), hi.normals, rigidity, poisson, terms_a)
        side_b = side_values(mesh, second, lo.reference_points(0), lo.normals, rigidity, poisson,
                             stretch_terms(stretch, lo.points))
        side_b = tuple(np.where(flip[:, None, None], part[:, ::-1, :], part) for part in side_b)
        tolerance = 1e3 * PERIODIC_TOLERANCE * float(np.ptp(mesh.points, axis=0).max())
        other = 1 - axis
        lo_points = np.where(flip[:, None], lo.points[:, ::-1, other], lo.points[:, :, other])
        if np.abs(lo_points - hi.points[:, :, other]).max(initial=0.0) > tolerance:
            raise ValueError('plate bending: seam Gauss points do not match across the periodic sides')
        penalty = _penalty(mesh, (first, second), hi.lengths, rigidity)
        blocks = edge_blocks([side_a[0], side_b[0]], [side_a[1], side_b[1]], hi.weights, penalty,
                             penalty_scale(terms_a, hi.normals)) * factors
        _append(rows, cols, values, np.concatenate([_local_dofs(dofs, first), _local_dofs(dofs, second)], axis=1),
                blocks)
    return _to_csr(rows, cols, values, (dofs.n_dofs, dofs.n_dofs)).astype(np.complex128)
