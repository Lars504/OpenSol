"""Reference elements of the FEM path (mesh plan 2.4): Lagrange P0 / P1 / P2, Nedelec first kind degree 1 / 2 with
the global edge orientation (lower -> higher node index), and the element geometry (affine, or isoparametric P2 on
elements with a curved edge; covariant Piola for edge elements). Host numpy; reference triangle (0, 0), (1, 0),
(0, 1); local node k, local edge k = nodes (k + 1, k + 2) mod 3 (fem.mesh); P2 local node 3 + k = midside of edge k."""
from dataclasses import dataclass

import numpy as np

# d L_k / d(xi, eta) of the barycentric coordinates L0 = 1 - xi - eta, L1 = xi, L2 = eta
REFERENCE_GRADIENTS = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
REFERENCE_VERTICES = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
LAGRANGE_COUNTS = {0: 1, 1: 3, 2: 6}
NEDELEC_COUNTS = {1: 3, 2: 8}
_INTERIOR_EDGES = (0, 1)   # Nedelec degree 2 interior functions L_k W_(k+1, k+2) of local edges 0 and 1


def barycentric(points) -> np.ndarray:
    """(..., 3) barycentric coordinates of reference points (..., 2)."""
    points = np.asarray(points, dtype=np.float64)
    return np.stack([1.0 - points[..., 0] - points[..., 1], points[..., 0], points[..., 1]], axis=-1)


def _lagrange_values(order, bary):
    """(..., n) order 0 / 1 / 2 Lagrange values of barycentric coordinates (..., 3) (order 1 returns bary itself)."""
    lead = bary.shape[:-1]
    if order == 0:
        return np.ones(lead + (1,))
    if order == 1:
        return bary
    if order != 2:
        raise ValueError(f'lagrange_basis: order must be 0, 1 or 2, got {order!r}')
    values = np.empty(lead + (6,))
    values[..., :3] = bary * (2.0 * bary - 1.0)
    for k in range(3):
        values[..., 3 + k] = 4.0 * bary[..., (k + 1) % 3] * bary[..., (k + 2) % 3]
    return values


def lagrange_values(order, points) -> np.ndarray:
    """(..., n) values of the order 0 / 1 / 2 Lagrange basis at reference points (..., 2): lagrange_basis without the
    gradients (mass / load blocks, field reads)."""
    return _lagrange_values(order, barycentric(points))


def lagrange_basis(order, points) -> tuple:
    """(values (..., n), reference gradients (..., n, 2)) of the order 0 / 1 / 2 Lagrange basis at reference points
    (..., 2)."""
    bary = barycentric(points)
    values = _lagrange_values(order, bary)
    lead = bary.shape[:-1]
    if order == 0:
        return values, np.zeros(lead + (1, 2))
    gradients_l = np.broadcast_to(REFERENCE_GRADIENTS, lead + (3, 2))
    if order == 1:
        return values, np.array(gradients_l)
    gradients = np.empty(lead + (6, 2))
    gradients[..., :3, :] = (4.0 * bary - 1.0)[..., None] * gradients_l
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3
        gradients[..., 3 + k, :] = 4.0 * (bary[..., b, None] * REFERENCE_GRADIENTS[a]
                                          + bary[..., a, None] * REFERENCE_GRADIENTS[b])
    return values, gradients


def _cross(first, second):
    """2D cross product first_x second_y - first_y second_x over the last axis."""
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _whitney(bary, a, b):
    """(W_ab (..., 2), curl W_ab scalar) of W_ab = L_a grad L_b - L_b grad L_a in reference coordinates."""
    vector = bary[..., a, None] * REFERENCE_GRADIENTS[b] - bary[..., b, None] * REFERENCE_GRADIENTS[a]
    return vector, 2.0 * _cross(REFERENCE_GRADIENTS[a], REFERENCE_GRADIENTS[b])


def nedelec_basis(degree, points, edge_signs) -> tuple:
    """(reference vectors (..., n, 2), reference curls (..., n)) of the Nedelec first-kind basis of degree 1 (Whitney,
    edge k) or 2 (edge k: L_lo W, L_hi W of W = W_(lo, hi), lo / hi = lower / higher global node; then L_0 W_12,
    L_1 W_20) at reference points (..., 2); edge_signs (..., 3) broadcastable: +1 when local edge k runs lower ->
    higher. Physical values by piola_vectors, curls / det J."""
    bary = barycentric(points)
    signs = np.asarray(edge_signs, dtype=np.float64)
    lead = np.broadcast_shapes(bary.shape[:-1], signs.shape[:-1])
    bary = np.broadcast_to(bary, lead + (3,))
    signs = np.broadcast_to(signs, lead + (3,))
    count = NEDELEC_COUNTS.get(degree)
    if count is None:
        raise ValueError(f'nedelec_basis: degree must be 1 or 2, got {degree!r}')
    vectors = np.empty(lead + (count, 2))
    curls = np.empty(lead + (count,))
    for k in range(3):
        a, b = (k + 1) % 3, (k + 2) % 3
        whitney, whitney_curl = _whitney(bary, a, b)
        sign = signs[..., k]
        if degree == 1:
            vectors[..., k, :] = sign[..., None] * whitney
            curls[..., k] = sign * whitney_curl
            continue
        # L_c W_ab and curl(L_c W_ab) = grad L_c x W_ab + L_c curl W_ab
        along_a = bary[..., a, None] * whitney
        along_b = bary[..., b, None] * whitney
        curl_a = _cross(REFERENCE_GRADIENTS[a], whitney) + bary[..., a] * whitney_curl
        curl_b = _cross(REFERENCE_GRADIENTS[b], whitney) + bary[..., b] * whitney_curl
        forward = sign > 0.0   # lo = a: (L_a W_ab, L_b W_ab); else lo = b, W_lohi = -W_ab: (-L_b W_ab, -L_a W_ab)
        vectors[..., 2 * k, :] = np.where(forward[..., None], along_a, -along_b)
        vectors[..., 2 * k + 1, :] = np.where(forward[..., None], along_b, -along_a)
        curls[..., 2 * k] = np.where(forward, curl_a, -curl_b)
        curls[..., 2 * k + 1] = np.where(forward, curl_b, -curl_a)
    if degree == 2:
        for slot, k in enumerate(_INTERIOR_EDGES):
            a, b = (k + 1) % 3, (k + 2) % 3
            whitney, whitney_curl = _whitney(bary, a, b)
            vectors[..., 6 + slot, :] = bary[..., k, None] * whitney
            curls[..., 6 + slot] = _cross(REFERENCE_GRADIENTS[k], whitney) + bary[..., k] * whitney_curl
    return vectors, curls


def physical_gradients(reference_gradients, inverse) -> np.ndarray:
    """grad_x = J^-T grad_xi: reference (m, q, n, 2) or (q, n, 2) with inverse Jacobians (m, q, 2, 2) -> (m, q, n, 2)
    (rows times J^-1, written out per component: a batched 2x2 matmul loops per point); also covariant Piola."""
    reference_gradients = np.asarray(reference_gradients)
    inverse = np.asarray(inverse)
    along_xi, along_eta = reference_gradients[..., 0], reference_gradients[..., 1]          # (..., n)
    lead = np.broadcast_shapes(along_xi.shape, inverse.shape[:-2] + (1,))
    result = np.empty(lead + (2,), dtype=np.result_type(reference_gradients, inverse))
    for column in range(2):
        component = result[..., column]
        np.multiply(along_xi, inverse[..., 0, column, None], out=component)
        component += along_eta * inverse[..., 1, column, None]
    return result


piola_vectors = physical_gradients


@dataclass(frozen=True, eq=False)
class ElementGeometry:
    """Map of m elements at q reference points: physical points, the determinant of the Jacobian J[c, d] = dx_c /
    dxi_d (> 0, ccw elements) and its inverse (J itself is not kept: no reader)."""
    points: np.ndarray        # (m, q, 2) [m]
    determinant: np.ndarray   # (m, q)
    inverse: np.ndarray       # (m, q, 2, 2)


def curved_elements(mesh, elements=None) -> np.ndarray:
    """Bool per element (of `elements`, None = all): True when an edge lies on a curved boundary."""
    element_edges = mesh.element_edges if elements is None else mesh.element_edges[elements]
    return (mesh.edge_curve[element_edges] >= 0).any(axis=1)


def element_geometry(mesh, elements, reference_points, order=2, check=True) -> ElementGeometry:
    """Geometry of mesh elements (int indices) at reference points (q, 2) shared, or (m, q, 2) per element: affine,
    except order 2 maps elements with a curved edge isoparametrically (P2 on corners + mesh.midside_points).
    ValueError when a curved element folds (det J <= 0) unless check is False (point location outside elements)."""
    elements = np.asarray(elements, dtype=np.int64)
    reference_points = np.asarray(reference_points, dtype=np.float64)   # (q, 2) shared, or (m, q, 2)
    per_element = reference_points.ndim == 3
    corners = mesh.points[mesh.triangles[elements]]   # (m, 3, 2)
    # elementwise, not batched matmuls (those loop per element / point): points = sum_k L_k x_k, affine J columns
    # x_1 - x_0 and x_2 - x_0 (= sum_k x_k grad L_k), det and inverse once per element, then copied to every point
    bary = barycentric(reference_points)                                         # (q, 3) or (m, q, 3)
    points = bary[..., 0, None] * corners[:, None, 0, :]
    points += bary[..., 1, None] * corners[:, None, 1, :]
    points += bary[..., 2, None] * corners[:, None, 2, :]                         # (m, q, 2)
    affine = np.empty((len(elements), 2, 2))
    affine[:, :, 0] = corners[:, 1, :] - corners[:, 0, :]
    affine[:, :, 1] = corners[:, 2, :] - corners[:, 0, :]
    affine_determinant = affine[:, 0, 0] * affine[:, 1, 1] - affine[:, 0, 1] * affine[:, 1, 0]
    affine_inverse = _inverse_2x2(affine, affine_determinant[:, None, None])
    point_count = points.shape[1]
    determinant = np.empty(points.shape[:2])
    determinant[...] = affine_determinant[:, None]
    inverse = np.empty(points.shape + (2,))
    inverse[...] = affine_inverse[:, None]
    if order == 2:
        curved = np.flatnonzero(curved_elements(mesh, elements))
        if len(curved):
            nodes = np.concatenate([corners[curved],
                                    mesh.midside_points[mesh.element_edges[elements[curved]]]], axis=1)   # (k, 6, 2)
            values2, gradients2 = lagrange_basis(2, reference_points[curved] if per_element else reference_points)
            points[curved] = values2 @ nodes
            if per_element:   # (k, q, 6, 2): one (2, 6) @ (6, 2 q) product per element
                stacked = gradients2.transpose(0, 2, 1, 3).reshape(len(curved), 6, 2 * point_count)
                curved_jacobian = (nodes.swapaxes(1, 2) @ stacked).reshape(len(curved), 2, point_count, 2)
            else:             # (q, 6, 2) shared: one GEMM over all curved elements
                curved_jacobian = np.tensordot(nodes, gradients2, axes=([1], [1]))   # (k, 2, q, 2)
            curved_jacobian = curved_jacobian.transpose(0, 2, 1, 3)                    # (k, q, 2, 2)
            curved_determinant = (curved_jacobian[..., 0, 0] * curved_jacobian[..., 1, 1]
                                  - curved_jacobian[..., 0, 1] * curved_jacobian[..., 1, 0])
            determinant[curved] = curved_determinant
            inverse[curved] = _inverse_2x2(curved_jacobian, curved_determinant[..., None, None])
    if check and not (determinant > 0.0).all():
        raise ValueError(f'element geometry: {int((determinant <= 0.0).any(axis=1).sum())} elements have a '
                         'non-positive Jacobian (folded curved element: refine near the curve)')
    return ElementGeometry(points, determinant, inverse)


def _inverse_2x2(matrices, determinant) -> np.ndarray:
    """(..., 2, 2) inverses of 2x2 matrices given their determinants broadcastable to (..., 1, 1)."""
    inverse = np.empty_like(matrices)
    inverse[..., 0, 0] = matrices[..., 1, 1]
    inverse[..., 1, 1] = matrices[..., 0, 0]
    inverse[..., 0, 1] = -matrices[..., 0, 1]
    inverse[..., 1, 0] = -matrices[..., 1, 0]
    inverse /= determinant
    return inverse


def edge_reference_points(edge_signs_of_local, local_edge, t) -> np.ndarray:
    """(k, q, 2) reference points of parameters t (q,) along local edge local_edge (k,) of their elements, t running
    from the edge's lower to its higher global node (edge_signs_of_local (k,) = the element's sign of that edge)."""
    local_edge = np.asarray(local_edge, dtype=np.int64)
    forward = np.asarray(edge_signs_of_local) > 0
    first = np.where(forward, (local_edge + 1) % 3, (local_edge + 2) % 3)
    second = np.where(forward, (local_edge + 2) % 3, (local_edge + 1) % 3)
    start = REFERENCE_VERTICES[first][:, None, :]
    end = REFERENCE_VERTICES[second][:, None, :]
    t = np.asarray(t, dtype=np.float64)[None, :, None]
    return start + t * (end - start)


def edge_geometry(mesh, edges, t, order=2) -> tuple:
    """(points (k, q, 2), line Jacobians |dx/dt| (k, q), unit tangents (k, q, 2)) of mesh edges at parameters t (q,)
    from the lower to the higher node; order 2 follows curved edges (P2 through mesh.midside_points)."""
    edges = np.asarray(edges, dtype=np.int64)
    t = np.asarray(t, dtype=np.float64)
    start = mesh.points[mesh.edges[edges, 0]][:, None, :]
    end = mesh.points[mesh.edges[edges, 1]][:, None, :]
    points = start + t[None, :, None] * (end - start)
    derivative = np.broadcast_to(end - start, points.shape).copy()
    if order == 2:
        curved = mesh.edge_curve[edges] >= 0
        if curved.any():
            middle = mesh.midside_points[edges[curved]][:, None, :]
            first, second = start[curved], end[curved]
            shape_start, shape_end, shape_middle = (1.0 - t) * (1.0 - 2.0 * t), t * (2.0 * t - 1.0), 4.0 * t * (1.0 - t)
            slope_start, slope_end, slope_middle = 4.0 * t - 3.0, 4.0 * t - 1.0, 4.0 - 8.0 * t
            points[curved] = (shape_start[None, :, None] * first + shape_end[None, :, None] * second
                              + shape_middle[None, :, None] * middle)
            derivative[curved] = (slope_start[None, :, None] * first + slope_end[None, :, None] * second
                                  + slope_middle[None, :, None] * middle)
    lengths = np.linalg.norm(derivative, axis=-1)
    return points, lengths, derivative / lengths[..., None]
