"""Point location and field evaluation on a TetrahedronMesh (igor3d_plan section 7, T7): the containing element of
physical points by a k-d tree over element centroids, Lagrange fields (values and gradients) and Nedelec fields (values
and curls) there, Lagrange values at element centroids. Host numpy, scipy inside functions."""
import numpy as np

from .elements3d import lagrange_basis_3d, nedelec_basis_3d, reference_coordinates, tetrahedron_geometry

FIRST_CANDIDATES = 8          # nearest centroids of the first k-d tree pass (most points of regular meshes)
LOCATE_CANDIDATES = 24        # nearest centroids tried per point (sliver-rich meshes may need more)
FALLBACK_CANDIDATES = 128     # last k-d tree pass for the points the earlier ones missed
INSIDE_TOLERANCE = 1e-10      # barycentric slack: points on shared faces / the outer boundary count as inside


class PointLocator3D:
    """Containing element of physical points: the first of the nearest centroids (passes of FIRST_CANDIDATES,
    LOCATE_CANDIDATES, then FALLBACK_CANDIDATES, each over the misses of the last, nearest first) whose barycentric
    coordinates are all >= -INSIDE_TOLERANCE, then the elements whose bounding box holds the point (graded / sliver
    meshes: a large element's centroid lies far away); -1 when none holds it."""

    def __init__(self, mesh, candidates=LOCATE_CANDIDATES):
        from scipy.spatial import cKDTree
        self.mesh = mesh
        self.candidates = int(min(candidates, mesh.n_elements))
        self._tree = cKDTree(mesh.centroids())
        self._boxes = None   # (lower (n, 3), upper (n, 3)) element bounding boxes, built at the first brute-force pass

    def locate(self, points) -> tuple:
        """(elements (p,) int64, -1 outside; reference coordinates (p, 3), NaN outside) of points (p, 3) [m]."""
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        elements = np.full(len(points), -1, dtype=np.int64)
        reference = np.full((len(points), 3), np.nan)
        missed = np.arange(len(points))
        tried = 0   # nearest centroids already tried by every missed point
        passes = (min(FIRST_CANDIDATES, self.candidates), self.candidates,
                  min(FALLBACK_CANDIDATES, self.mesh.n_elements))
        for count in passes:
            if missed.size == 0 or count <= tried:
                continue
            _, nearest = self._tree.query(points[missed], k=count)
            nearest = np.asarray(nearest).reshape(len(missed), -1)[:, tried:]
            self._try_candidates(points, missed, nearest, elements, reference)
            missed = np.flatnonzero(elements < 0)
            tried = count
        if missed.size:
            self._brute_force(points, missed, elements, reference)
        return elements, reference

    def _try_candidates(self, points, rows, nearest, elements, reference) -> None:
        """Fill elements / reference of `rows` from their candidate columns nearest (len(rows), k), first hit wins."""
        for column in range(nearest.shape[1]):
            open_positions = np.flatnonzero(elements[rows] < 0)
            if open_positions.size == 0:
                return
            open_rows = rows[open_positions]
            candidates = nearest[open_positions, column].astype(np.int64)
            self._accept(points, open_rows, candidates, elements, reference)

    def _accept(self, points, rows, candidates, elements, reference) -> None:
        """Write the candidates (one per row) that hold their point."""
        local = reference_coordinates(tetrahedron_geometry(self.mesh, candidates), points[rows])
        smallest = np.minimum(local.min(axis=1), 1.0 - local.sum(axis=1))
        hit = smallest >= -INSIDE_TOLERANCE
        elements[rows[hit]] = candidates[hit]
        reference[rows[hit]] = local[hit]

    def _brute_force(self, points, rows, elements, reference) -> None:
        """Test every element whose bounding box holds the point (rows outside the node box skipped: truly outside)."""
        mesh = self.mesh
        slack = INSIDE_TOLERANCE * float(np.ptp(mesh.points, axis=0).max())
        inside_box = ((points[rows] >= mesh.points.min(axis=0) - slack)
                      & (points[rows] <= mesh.points.max(axis=0) + slack)).all(axis=1)
        rows = rows[inside_box]
        if rows.size == 0:
            return
        if self._boxes is None:
            corners = mesh.points[mesh.tets]
            self._boxes = (corners.min(axis=1), corners.max(axis=1))
        lower, upper = self._boxes
        for row in rows:
            candidates = np.flatnonzero(((lower <= points[row] + slack) & (upper >= points[row] - slack)).all(axis=1))
            if candidates.size:
                self._accept(points, np.full(candidates.size, row), candidates, elements, reference)


def evaluate_lagrange_3d(dofs, coefficients, elements, reference, gradients=True) -> tuple:
    """(values (p,), gradients (p, 3) or None with gradients False) of the Lagrange P0 / P1 / P2 field with dof
    coefficients (n_dofs,) at located points (elements (p,), reference (p, 3) from PointLocator3D.locate); NaN where the
    element is -1."""
    coefficients = np.asarray(coefficients)
    elements = np.asarray(elements, dtype=np.int64)
    dtype = np.result_type(coefficients.dtype, np.float64)
    values = np.full(len(elements), np.nan, dtype=dtype)
    point_gradients = np.full((len(elements), 3), np.nan, dtype=dtype) if gradients else None
    found = np.flatnonzero(elements >= 0)
    if found.size == 0:
        return values, point_gradients
    basis, reference_gradients = lagrange_basis_3d(dofs.order, reference[found])   # (p, n), (p, n, 3)
    local = coefficients[dofs.element_dofs[elements[found]]]   # (p, n)
    values[found] = np.einsum('pn,pn->p', local, basis)
    if gradients:
        geometry = tetrahedron_geometry(dofs.mesh, elements[found])
        point_gradients[found] = np.einsum('pn,pji,pnj->pi', local, geometry.inverse, reference_gradients)
    return values, point_gradients


def centroid_values_3d(dofs, coefficients, elements=None) -> np.ndarray:
    """(m,) Lagrange field values at element reference centroids (all elements by default): the per-element state of
    D12, no point location needed."""
    basis, _ = lagrange_basis_3d(dofs.order, np.full((1, 3), 0.25))
    local = dofs.element_dofs if elements is None else dofs.element_dofs[elements]
    return np.asarray(coefficients)[local] @ basis[0]


def evaluate_nedelec_3d(dofs, coefficients, elements, reference) -> tuple:
    """(values (p, 3), curls (p, 3)) of the Nedelec field with dof coefficients (n_dofs,) at located points (elements
    (p,), reference (p, 3) from PointLocator3D.locate); NaN where the element is -1."""
    coefficients = np.asarray(coefficients)
    elements = np.asarray(elements, dtype=np.int64)
    dtype = np.result_type(coefficients.dtype, np.float64)
    values = np.full((len(elements), 3), np.nan, dtype=dtype)
    curls = np.full((len(elements), 3), np.nan, dtype=dtype)
    found = np.flatnonzero(elements >= 0)
    if found.size == 0:
        return values, curls
    geometry = tetrahedron_geometry(dofs.mesh, elements[found])
    reference_vectors, reference_curls = nedelec_basis_3d(dofs.order, reference[found])   # (p, n, 3)
    vectors = np.einsum('pji,pnj->pni', geometry.inverse, reference_vectors)
    curl_vectors = (np.einsum('pij,pnj->pni', geometry.jacobian, reference_curls)
                    / geometry.determinant[:, None, None])
    local = coefficients[dofs.element_dofs[elements[found]]]   # (p, n)
    values[found] = np.einsum('pn,pni->pi', local, vectors)
    curls[found] = np.einsum('pn,pni->pi', local, curl_vectors)
    return values, curls
