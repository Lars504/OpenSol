"""Reference elements on tetrahedra (igor3d_plan section 7, T5): Lagrange P1 / P2 and Nedelec first kind degree 1 / 2,
affine element geometry (covariant Piola for edge elements, contravariant for their curls). Host numpy; reference
tetrahedron (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)."""
# Orientation: a TetrahedronMesh stores every element's nodes in ascending global order (fem.mesh3d), so local edge
# (a, b), a < b, runs from the lower to the higher global node and a local face (a, b, c) lists its nodes ascending in
# both of its elements: the Nedelec functions below need no signs and agree between neighbours. Degree 2 (20 dofs):
# per local edge (a, b) L_a W_ab, L_b W_ab; per local face (a, b, c) L_a W_bc, L_b W_ca (zero tangential trace on
# every edge and on the other faces; L_c W_ab is their dependent third). W_ab = L_a grad L_b - L_b grad L_a, curl
# W_ab = 2 grad L_a x grad L_b, curl (L_c W_ab) = grad L_c x W_ab + L_c curl W_ab. A negative det J (ascending nodes
# need not be right-handed) is fine: integrals take |det J|, curls divide by the signed det J.
from dataclasses import dataclass

import numpy as np

# d L_k / d(xi, eta, zeta) of L0 = 1 - xi - eta - zeta, L1 = xi, L2 = eta, L3 = zeta
REFERENCE_GRADIENTS_3D = np.array([[-1.0, -1.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
REFERENCE_VERTICES_3D = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
LOCAL_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
LOCAL_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))   # face k opposite local node k, nodes ascending
LAGRANGE_COUNTS_3D = {0: 1, 1: 4, 2: 10}
NEDELEC_COUNTS_3D = {1: 6, 2: 20}
# lagrange_basis_3d memoises point sets up to LAGRANGE_MEMO_POINTS points (rules, local dof points, centroids, probes),
# the latest LAGRANGE_MEMO_ENTRIES of them; larger sets (located samples) are computed per call
LAGRANGE_MEMO_POINTS = 1024
LAGRANGE_MEMO_ENTRIES = 256
_LAGRANGE_MEMO = {}   # (order, shape, point bytes) -> read-only (values, gradients); insertion order = recency


def barycentric_3d(points) -> np.ndarray:
    """(..., 4) barycentric coordinates of reference points (..., 3)."""
    points = np.asarray(points, dtype=np.float64)
    return np.concatenate([1.0 - points.sum(axis=-1, keepdims=True), points], axis=-1)


def lagrange_basis_3d(order, points, gradients=True) -> tuple:
    """(values (..., n), reference gradients (..., n, 3); None with gradients False) of the P0 / P1 / P2 basis at
    reference points (..., 3); P2 local nodes: corners, then 4 + e = midpoint of LOCAL_EDGES[e]; P0 = the element
    indicator. Sets up to LAGRANGE_MEMO_POINTS points are memoised and returned READ-ONLY."""
    # the memo lookup and the formulas stay in this one function: a memo hit still runs (and traces) this code unit
    points = np.asarray(points, dtype=np.float64)
    want_gradients = bool(gradients)
    key = None
    if points.size <= 3 * LAGRANGE_MEMO_POINTS:
        key = (order, points.shape, points.tobytes(), want_gradients)
        found = _LAGRANGE_MEMO.pop(key, None)
        if found is not None:
            _LAGRANGE_MEMO[key] = found   # most recent last
            return found
    bary = barycentric_3d(points)
    lead = bary.shape[:-1]
    gradients_l = np.broadcast_to(REFERENCE_GRADIENTS_3D, lead + (4, 3)) if want_gradients else None
    basis_gradients = None
    if order == 0:
        values = np.ones(lead + (1,))
        if want_gradients:
            basis_gradients = np.zeros(lead + (1, 3))
    elif order == 1:
        values = bary
        if want_gradients:
            basis_gradients = np.array(gradients_l)
    elif order == 2:
        values = np.empty(lead + (10,))
        values[..., :4] = bary * (2.0 * bary - 1.0)
        for edge, (a, b) in enumerate(LOCAL_EDGES):
            values[..., 4 + edge] = 4.0 * bary[..., a] * bary[..., b]
        if want_gradients:
            basis_gradients = np.empty(lead + (10, 3))
            basis_gradients[..., :4, :] = (4.0 * bary - 1.0)[..., None] * gradients_l
            for edge, (a, b) in enumerate(LOCAL_EDGES):
                basis_gradients[..., 4 + edge, :] = 4.0 * (bary[..., b, None] * REFERENCE_GRADIENTS_3D[a]
                                                           + bary[..., a, None] * REFERENCE_GRADIENTS_3D[b])
    else:
        raise ValueError(f'lagrange_basis_3d: order must be 0, 1 or 2, got {order!r}')
    if key is not None:
        values.flags.writeable = False
        if basis_gradients is not None:
            basis_gradients.flags.writeable = False
        _LAGRANGE_MEMO[key] = (values, basis_gradients)
        if len(_LAGRANGE_MEMO) > LAGRANGE_MEMO_ENTRIES:
            del _LAGRANGE_MEMO[next(iter(_LAGRANGE_MEMO))]
    return values, basis_gradients


def _whitney(bary, a, b):
    """(W_ab (..., 3), curl W_ab (3,)) in reference coordinates."""
    vector = bary[..., a, None] * REFERENCE_GRADIENTS_3D[b] - bary[..., b, None] * REFERENCE_GRADIENTS_3D[a]
    return vector, 2.0 * np.cross(REFERENCE_GRADIENTS_3D[a], REFERENCE_GRADIENTS_3D[b])


def _weighted_whitney(bary, c, a, b):
    """(L_c W_ab (..., 3), its curl (..., 3)) in reference coordinates."""
    whitney, whitney_curl = _whitney(bary, a, b)
    curl = np.cross(REFERENCE_GRADIENTS_3D[c], whitney) + bary[..., c, None] * whitney_curl
    return bary[..., c, None] * whitney, curl


def nedelec_basis_3d(degree, points) -> tuple:
    """(reference vectors (..., n, 3), reference curls (..., n, 3)) of the Nedelec first-kind basis of degree 1
    (Whitney, LOCAL_EDGES order) or 2 (module header order: 2 per edge, then 2 per face) at reference points (..., 3).
    Physical values by piola_vectors_3d, curls by piola_curls_3d."""
    count = NEDELEC_COUNTS_3D.get(degree)
    if count is None:
        raise ValueError(f'nedelec_basis_3d: degree must be 1 or 2, got {degree!r}')
    bary = barycentric_3d(points)
    lead = bary.shape[:-1]
    vectors = np.empty(lead + (count, 3))
    curls = np.empty(lead + (count, 3))
    for edge, (a, b) in enumerate(LOCAL_EDGES):
        if degree == 1:
            vectors[..., edge, :], curls[..., edge, :] = _whitney(bary, a, b)
            continue
        vectors[..., 2 * edge, :], curls[..., 2 * edge, :] = _weighted_whitney(bary, a, a, b)
        vectors[..., 2 * edge + 1, :], curls[..., 2 * edge + 1, :] = _weighted_whitney(bary, b, a, b)
    if degree == 2:
        for face, (a, b, c) in enumerate(LOCAL_FACES):
            vectors[..., 12 + 2 * face, :], curls[..., 12 + 2 * face, :] = _weighted_whitney(bary, a, b, c)
            vectors[..., 13 + 2 * face, :], curls[..., 13 + 2 * face, :] = _weighted_whitney(bary, b, c, a)
    return vectors, curls


@dataclass(frozen=True, eq=False)
class TetrahedronGeometry:
    """Affine map of m elements: corner 0 (m, 3), Jacobian J[c, d] = dx_c / dxi_d (m, 3, 3), signed determinant (m,)
    and inverse (m, 3, 3)."""
    origin: np.ndarray
    jacobian: np.ndarray
    determinant: np.ndarray
    inverse: np.ndarray

    def points(self, reference_points) -> np.ndarray:
        """(m, q, 3) physical points of reference points (q, 3) shared, or (m, q, 3) per element."""
        reference_points = np.asarray(reference_points, dtype=np.float64)
        if reference_points.ndim == 3:
            return self.origin[:, None, :] + np.einsum('mqd,mcd->mqc', reference_points, self.jacobian)
        return self.origin[:, None, :] + _rows_times(reference_points, self.jacobian.swapaxes(1, 2))


def _rows_times(rows, matrices) -> np.ndarray:
    """(m, ..., c) = rows (..., k) @ matrices (m, k, c): rows (q, k) / (q, n, k) shared by every element, one GEMM
    (np.matmul would broadcast to m x points tiny products); per-element rows (m, q, n, k) take the batched einsum."""
    if rows.ndim == 4:
        return np.einsum('mqnk,mkc->mqnc', rows, matrices)
    flat = rows.reshape(-1, rows.shape[-1])
    product = np.tensordot(flat, matrices, axes=([1], [1]))                     # (p, m, c)
    return np.ascontiguousarray(product.transpose(1, 0, 2)).reshape((len(matrices),) + rows.shape[:-1]
                                                                    + (matrices.shape[-1],))


def tetrahedron_geometry(mesh, elements) -> TetrahedronGeometry:
    """TetrahedronGeometry of mesh elements (int indices); ValueError on a degenerate element."""
    elements = np.asarray(elements, dtype=np.int64)
    corners = mesh.points[mesh.tets[elements]]   # (m, 4, 3)
    jacobian = np.transpose(corners[:, 1:, :] - corners[:, :1, :], (0, 2, 1))   # columns x_k - x_0
    # closed-form cofactors (np.linalg.det / inv run LAPACK once per 3x3 element): adjugate rows = cross products of
    # the Jacobian columns, det = column_0 . (column_1 x column_2)
    column_0, column_1, column_2 = jacobian[:, :, 0], jacobian[:, :, 1], jacobian[:, :, 2]
    adjugate = np.empty_like(jacobian)
    adjugate[:, 0, :] = np.cross(column_1, column_2)
    adjugate[:, 1, :] = np.cross(column_2, column_0)
    adjugate[:, 2, :] = np.cross(column_0, column_1)
    determinant = np.einsum('mc,mc->m', column_0, adjugate[:, 0, :])
    scale = np.abs(jacobian).max(axis=(1, 2)) ** 3
    if not (np.abs(determinant) > 1e-14 * scale).all():
        raise ValueError(f'tetrahedron geometry: {int((np.abs(determinant) <= 1e-14 * scale).sum())} degenerate '
                         'elements (zero volume)')
    return TetrahedronGeometry(origin=corners[:, 0, :], jacobian=jacobian, determinant=determinant,
                               inverse=adjugate / determinant[:, None, None])


def piola_vectors_3d(reference_vectors, geometry) -> np.ndarray:
    """Covariant Piola J^-T v: reference (q, n, 3) -> physical (m, q, n, 3) (also Lagrange gradients; rows times
    J^-1: one GEMM over all elements)."""
    return _rows_times(np.asarray(reference_vectors), geometry.inverse)


def piola_curls_3d(reference_curls, geometry) -> np.ndarray:
    """Contravariant Piola J c / det J of edge-element curls: reference (q, n, 3) -> physical (m, q, n, 3)."""
    curls = _rows_times(np.asarray(reference_curls), geometry.jacobian.swapaxes(1, 2))
    curls /= geometry.determinant.reshape((-1,) + (1,) * (curls.ndim - 1))
    return curls


def reference_coordinates(geometry, points) -> np.ndarray:
    """(m, 3) reference coordinates J^-1 (x - x0) of one physical point (m, 3) per element."""
    return np.einsum('mij,mj->mi', geometry.inverse, np.asarray(points, dtype=np.float64) - geometry.origin)
