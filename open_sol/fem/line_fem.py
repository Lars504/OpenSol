"""1D line finite elements on a line's cell edges: P1 / P2 Lagrange and Hermite cubic (w, w') dof maps, element
matrices of any derivative pair with complex coefficients, loads, point terms, held ends, sampling, band solve and eigen.

Dofs ascend along the line (unlike the 2D "nodes, then edges" numbering), so every operator is banded: P1 node k = k;
P2 node k = 2 k, midpoint of element e = 2 e + 1; Hermite w at node k = 2 k, w' (= dw/dx, physical) = 2 k + 1. Local
dofs per element: P1 (left, right), P2 (left, mid, right), Hermite (w_L, w'_L, w_R, w'_R).

Coefficients: a number; an (n_elements,) array (one value per element = per cell of a 1D model); or a callable
f(x (m, q), elements (m,)) -> (m, q) at the quadrature points. Complex values give complex matrices, unconjugated
(complex symmetric for equal derivative orders). Host numpy, scipy inside functions."""
from dataclasses import dataclass

import numpy as np

from .assembly import CHUNK_ELEMENTS, DirichletSystem, _to_csr, eliminate_dirichlet, pattern_sum, restrict
from .fixed_pattern import scatter_add
from .quadrature import gauss_line

__all__ = ('LINE_FAMILIES', 'LineDofMap', 'line_dofs', 'assemble_line_matrix', 'assemble_line_stiffness',
           'assemble_line_mass', 'assemble_line_bending', 'assemble_line_load', 'assemble_point_load',
           'assemble_point_matrix', 'locate_on_line', 'evaluate_line', 'element_means', 'interpolate_line',
           'dof_points', 'node_dofs', 'end_dofs', 'nodes_at', 'active_dofs', 'band_storage', 'band_solve',
           'solve_line_system', 'line_eigenpairs', 'DirichletSystem', 'eliminate_dirichlet', 'pattern_sum', 'restrict')

LINE_FAMILIES = {'lagrange': (1, 2), 'hermite': (3,)}   # family -> allowed orders (= polynomial degree)
POSITION_TOLERANCE = 1e-9    # x line length: points this far outside the line are clamped onto it, farther ones miss
DENSE_EIGEN_DOFS = 200       # line_eigenpairs solves densely up to this many kept dofs (ARPACK above)

# Reference bases on t in [0, 1] (x = x_left + h t) as ascending power coefficients, in local dof order; Hermite slope
# functions are per unit h (the physical slope basis is h times them).
_BASIS_COEFFICIENTS = {
    ('lagrange', 1): ((1.0, -1.0), (0.0, 1.0)),
    ('lagrange', 2): ((1.0, -3.0, 2.0), (0.0, 4.0, -4.0), (0.0, -1.0, 2.0)),
    ('hermite', 3): ((1.0, 0.0, -3.0, 2.0), (0.0, 1.0, -2.0, 1.0), (0.0, 0.0, 3.0, -2.0), (0.0, 0.0, -1.0, 1.0)),
}
_FACES = ('x_lo', 'x_hi')


@dataclass(frozen=True, eq=False)
class LineDofMap:
    """Dofs of one element family on a line: edges (n_elements + 1,) [m] strictly ascending, element_dofs
    (n_elements, n_local) int64 (local order in the module header), n_dofs."""
    edges: np.ndarray
    family: str
    order: int
    element_dofs: np.ndarray
    n_dofs: int

    @property
    def n_elements(self) -> int:
        return len(self.edges) - 1

    @property
    def n_nodes(self) -> int:
        return len(self.edges)

    @property
    def n_local(self) -> int:
        return self.element_dofs.shape[1]

    @property
    def widths(self) -> np.ndarray:
        """Float64 (n_elements,) element lengths [m]."""
        return np.diff(self.edges)

    @property
    def length(self) -> float:
        return float(self.edges[-1] - self.edges[0])


def line_dofs(edges, family='lagrange', order=1) -> LineDofMap:
    """LineDofMap on the cell edges of a line (a 1D model: `Domain.cell_edges()[0]`, padding-free): 'lagrange' order 1
    / 2 or 'hermite' order 3 (C1 cubic, value and slope per node)."""
    if family not in LINE_FAMILIES or int(order) not in LINE_FAMILIES[family]:
        raise ValueError(f"line_dofs: family / order must be one of {LINE_FAMILIES}, got {family!r} / {order!r}")
    order = int(order)
    edges = np.array(edges, dtype=np.float64).ravel()
    if edges.size < 2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0.0):
        raise ValueError('line_dofs: edges must be >= 2 finite, strictly ascending coordinates')
    edges.flags.writeable = False
    n_elements = edges.size - 1
    first = np.arange(n_elements, dtype=np.int64)
    if family == 'lagrange' and order == 1:
        element_dofs = np.stack([first, first + 1], axis=1)
        n_dofs = n_elements + 1
    elif family == 'lagrange':
        element_dofs = np.stack([2 * first, 2 * first + 1, 2 * first + 2], axis=1)
        n_dofs = 2 * n_elements + 1
    else:
        element_dofs = np.stack([2 * first, 2 * first + 1, 2 * first + 2, 2 * first + 3], axis=1)
        n_dofs = 2 * (n_elements + 1)
    element_dofs.flags.writeable = False
    return LineDofMap(edges, family, order, element_dofs, int(n_dofs))


# ---- bases and coefficients ----
def _reference_table(dofs, t, derivative):
    """(..., n_local) d^derivative / dt^derivative of the reference basis at t (any shape)."""
    from numpy.polynomial import polynomial
    t = np.asarray(t, dtype=np.float64)
    columns = [polynomial.polyval(t, polynomial.polyder(np.array(coefficients), int(derivative)))
               * np.ones_like(t) for coefficients in _BASIS_COEFFICIENTS[(dofs.family, dofs.order)]]
    return np.stack(columns, axis=-1)


def _physical_basis(dofs, table, widths, derivative):
    """Physical d^derivative / dx^derivative basis from a reference table (..., n_local) and element widths broadcast to
    table[..., 0] (Hermite slope columns scaled by h)."""
    widths = np.asarray(widths, dtype=np.float64)[..., None]
    values = table / widths ** int(derivative)
    if dofs.family == 'hermite':
        values[..., 1::2] *= widths
    return values


def _check_derivative(dofs, derivative):
    derivative = int(derivative)
    if derivative < 0:
        raise ValueError(f'line_fem: derivative order must be >= 0, got {derivative}')
    if derivative >= 2 and dofs.family != 'hermite':
        raise ValueError('line_fem: second derivatives need Hermite (C1) elements; Lagrange elements are only C0')
    return derivative


def _element_list(dofs, elements):
    """Sorted unique int64 element indices of None (all), a bool mask (n_elements,) or indices."""
    if elements is None:
        return np.arange(dofs.n_elements, dtype=np.int64)
    elements = np.asarray(elements)
    if elements.dtype == bool:
        if elements.shape != (dofs.n_elements,):
            raise ValueError(f'line_fem: element mask must have shape ({dofs.n_elements},), got {elements.shape}')
        return np.flatnonzero(elements).astype(np.int64)
    elements = np.unique(elements.astype(np.int64).ravel())
    if elements.size and (elements[0] < 0 or elements[-1] >= dofs.n_elements):
        raise ValueError(f'line_fem: element indices must lie in [0, {dofs.n_elements})')
    return elements


def _coefficient_at(coefficient, x, elements, n_elements):
    """(m, q) coefficient values at the points x (m, q) of the elements (m,)."""
    if callable(coefficient):
        values = np.asarray(coefficient(x, elements))
    elif np.ndim(coefficient) == 0:
        values = np.asarray(coefficient)
    else:
        values = np.asarray(coefficient)
        if values.shape != (n_elements,):
            raise ValueError(f'line_fem: a coefficient array must have shape ({n_elements},) (one value per element), '
                             f'got {values.shape}')
        values = values[elements][:, None]
    return np.broadcast_to(values, x.shape)


def _rule_degree(dofs, derivatives, coefficient, degree):
    """Gauss degree: given, else exact for a per-element constant coefficient (+ 2 for a callable)."""
    if degree is not None:
        return int(degree)
    polynomial_degree = sum(max(dofs.order - derivative, 0) for derivative in derivatives)
    return polynomial_degree + (2 if callable(coefficient) else 0)


# ---- matrices and loads ----
def assemble_line_matrix(dofs, coefficient=1.0, test_derivative=0, trial_derivative=0, elements=None,
                         degree=None):
    """CSR (n_dofs, n_dofs) of int c(x) d^i phi_test d^j phi_trial dx over the elements (i, j = the derivative orders;
    unconjugated); rows of dofs no selected element touches stay empty."""
    test_derivative = _check_derivative(dofs, test_derivative)
    trial_derivative = _check_derivative(dofs, trial_derivative)
    selected = _element_list(dofs, elements)
    t, weights = gauss_line(_rule_degree(dofs, (test_derivative, trial_derivative), coefficient, degree))
    test_table = _reference_table(dofs, t, test_derivative)[None]     # (1, q, n)
    trial_table = _reference_table(dofs, t, trial_derivative)[None]
    n_local = dofs.n_local
    rows, cols, values = [], [], []
    for start in range(0, selected.size, CHUNK_ELEMENTS):
        chunk = selected[start:start + CHUNK_ELEMENTS]
        widths = dofs.widths[chunk]
        x = dofs.edges[chunk][:, None] + widths[:, None] * t[None, :]
        weighted = _coefficient_at(coefficient, x, chunk, dofs.n_elements) * (weights[None, :] * widths[:, None])
        test = _physical_basis(dofs, test_table, widths[:, None], test_derivative)       # (m, q, n)
        trial = _physical_basis(dofs, trial_table, widths[:, None], trial_derivative)
        blocks = np.einsum('mq,mqa,mqb->mab', weighted, test, trial)
        local = dofs.element_dofs[chunk]
        rows.append(np.repeat(local, n_local, axis=1).ravel())
        cols.append(np.tile(local, (1, n_local)).ravel())
        values.append(blocks)
    return _to_csr(rows, cols, values, (dofs.n_dofs, dofs.n_dofs))


def assemble_line_stiffness(dofs, coefficient=1.0, elements=None, degree=None):
    """CSR of int c u' v' dx (tension / conduction)."""
    return assemble_line_matrix(dofs, coefficient, 1, 1, elements, degree)


def assemble_line_mass(dofs, coefficient=1.0, elements=None, degree=None):
    """CSR of int c u v dx (consistent mass)."""
    return assemble_line_matrix(dofs, coefficient, 0, 0, elements, degree)


def assemble_line_bending(dofs, coefficient=1.0, elements=None, degree=None):
    """CSR of int c u'' v'' dx (bending stiffness EI / D; Hermite only)."""
    return assemble_line_matrix(dofs, coefficient, 2, 2, elements, degree)


def assemble_line_load(dofs, density, test_derivative=0, elements=None, degree=None) -> np.ndarray:
    """(n_dofs,) vector of int f(x) d^i phi dx over the elements (density: coefficient forms of the module header)."""
    test_derivative = _check_derivative(dofs, test_derivative)
    selected = _element_list(dofs, elements)
    t, weights = gauss_line(_rule_degree(dofs, (test_derivative,), density, degree))
    table = _reference_table(dofs, t, test_derivative)[None]
    widths = dofs.widths[selected]
    x = dofs.edges[selected][:, None] + widths[:, None] * t[None, :]
    weighted = _coefficient_at(density, x, selected, dofs.n_elements) * (weights[None, :] * widths[:, None])
    local_loads = np.einsum('mq,mqa->ma', weighted,
                            _physical_basis(dofs, table, widths[:, None], test_derivative))
    load = np.zeros(dofs.n_dofs, dtype=np.result_type(local_loads, np.float64))
    scatter_add(load, dofs.element_dofs[selected], local_loads)
    return load


def locate_on_line(dofs, positions):
    """(elements (k,) int64, t (k,) float64) of points [m]: the containing element (a node goes to the upper
    element, the last node to the last), clamped within POSITION_TOLERANCE x length; element -1 beyond."""
    positions = np.asarray(positions, dtype=np.float64).ravel()
    tolerance = POSITION_TOLERANCE * dofs.length
    outside = (positions < dofs.edges[0] - tolerance) | (positions > dofs.edges[-1] + tolerance) | ~np.isfinite(
        positions)
    clamped = np.clip(np.nan_to_num(positions, nan=dofs.edges[0]), dofs.edges[0], dofs.edges[-1])
    elements = np.clip(np.searchsorted(dofs.edges, clamped, side='right') - 1, 0, dofs.n_elements - 1)
    t = np.clip((clamped - dofs.edges[elements]) / dofs.widths[elements], 0.0, 1.0)
    elements = np.where(outside, -1, elements).astype(np.int64)
    return elements, t


def _point_basis(dofs, positions, derivative):
    """(elements (k,), basis (k, n_local)) of located points; ValueError for a point off the line."""
    derivative = _check_derivative(dofs, derivative)
    elements, t = locate_on_line(dofs, positions)
    if np.any(elements < 0):
        raise ValueError(f'line_fem: point(s) outside the line [{dofs.edges[0]}, {dofs.edges[-1]}] m')
    basis = _physical_basis(dofs, _reference_table(dofs, t, derivative), dofs.widths[elements], derivative)
    return elements, basis


def assemble_point_load(dofs, positions, values, test_derivative=0) -> np.ndarray:
    """(n_dofs,) vector sum_k v_k d^i phi(x_k): point forces (i = 0) or, on Hermite, point moments (i = 1)."""
    elements, basis = _point_basis(dofs, positions, test_derivative)
    values = np.broadcast_to(np.asarray(values), elements.shape)
    load = np.zeros(dofs.n_dofs, dtype=np.result_type(values, np.float64))
    scatter_add(load, dofs.element_dofs[elements], values[:, None] * basis)
    return load


def assemble_point_matrix(dofs, positions, values, test_derivative=0, trial_derivative=0):
    """CSR of sum_k v_k d^i phi(x_k) d^j phi(x_k): point springs / masses, end Robin, impedance or DtN terms."""
    elements, test = _point_basis(dofs, positions, test_derivative)
    _elements, trial = _point_basis(dofs, positions, trial_derivative)
    values = np.broadcast_to(np.asarray(values), elements.shape)
    blocks = values[:, None, None] * test[:, :, None] * trial[:, None, :]
    local = dofs.element_dofs[elements]
    n_local = dofs.n_local
    return _to_csr([np.repeat(local, n_local, axis=1).ravel()], [np.tile(local, (1, n_local)).ravel()], [blocks],
                   (dofs.n_dofs, dofs.n_dofs))


# ---- sampling ----
def evaluate_line(dofs, coefficients, positions, derivative=0) -> np.ndarray:
    """(k,) d^derivative u / dx^derivative of u = sum coefficients phi at points [m] (NaN off the line); at a node the
    upper element's value (derivatives of Lagrange fields jump there)."""
    derivative = _check_derivative(dofs, derivative)
    coefficients = np.asarray(coefficients)
    if coefficients.shape != (dofs.n_dofs,):
        raise ValueError(f'evaluate_line: coefficients must have shape ({dofs.n_dofs},), got {coefficients.shape}')
    elements, t = locate_on_line(dofs, positions)
    inside = elements >= 0
    safe = np.where(inside, elements, 0)
    basis = _physical_basis(dofs, _reference_table(dofs, t, derivative), dofs.widths[safe], derivative)
    values = np.einsum('ka,ka->k', basis, coefficients[dofs.element_dofs[safe]])
    values = values.astype(np.result_type(values, np.float64))
    values[~inside] = np.nan
    return values


def line_evaluation_matrix(dofs, positions, derivative=0):
    """CSR (k, n_dofs) E with E @ coefficients = evaluate_line(dofs, coefficients, positions, derivative) (empty rows
    off the line, where evaluate_line gives NaN): points evaluated many times locate once."""
    import scipy.sparse
    derivative = _check_derivative(dofs, derivative)
    elements, t = locate_on_line(dofs, positions)
    inside = np.flatnonzero(elements >= 0)
    basis = _physical_basis(dofs, _reference_table(dofs, t[inside], derivative), dofs.widths[elements[inside]],
                            derivative)
    rows = np.repeat(inside, basis.shape[1])
    return scipy.sparse.csr_matrix((basis.ravel(), (rows, dofs.element_dofs[elements[inside]].ravel())),
                                   shape=(elements.size, dofs.n_dofs))


def element_means(dofs, coefficients, derivative=0, elements=None) -> np.ndarray:
    """(m,) mean of d^derivative u / dx^derivative over each selected element (exact Gauss rule): per-element state."""
    derivative = _check_derivative(dofs, derivative)
    coefficients = np.asarray(coefficients)
    selected = _element_list(dofs, elements)
    t, weights = gauss_line(max(dofs.order - derivative, 0))
    table = _reference_table(dofs, t, derivative)[None]
    basis = _physical_basis(dofs, table, dofs.widths[selected][:, None], derivative)     # (m, q, n)
    return np.einsum('q,mqa,ma->m', weights, basis, coefficients[dofs.element_dofs[selected]])


def dof_points(dofs) -> np.ndarray:
    """(n_dofs,) positions [m] of the dofs (Hermite: the node of both its value and slope dof)."""
    if dofs.family == 'hermite':
        return np.repeat(dofs.edges, 2)
    if dofs.order == 1:
        return dofs.edges.copy()
    points = np.empty(dofs.n_dofs)
    points[0::2] = dofs.edges
    points[1::2] = 0.5 * (dofs.edges[:-1] + dofs.edges[1:])
    return points


def interpolate_line(dofs, function, slope_function=None) -> np.ndarray:
    """Coefficients of the interpolant of f(x) (vectorised): Lagrange nodal values; Hermite needs slope_function
    f'(x) for the slope dofs."""
    if dofs.family != 'hermite':
        points = dof_points(dofs)
        return np.asarray(function(points)) * np.ones(points.size)
    if slope_function is None:
        raise ValueError('interpolate_line: Hermite dofs need slope_function (df/dx)')
    values = np.asarray(function(dofs.edges)) * np.ones(dofs.n_nodes)
    slopes = np.asarray(slope_function(dofs.edges)) * np.ones(dofs.n_nodes)
    coefficients = np.empty(dofs.n_dofs, dtype=np.result_type(values, slopes))
    coefficients[0::2] = values
    coefficients[1::2] = slopes
    return coefficients


# ---- held dofs ----
def node_dofs(dofs, nodes, value=True, slope=False) -> np.ndarray:
    """Sorted int64 dofs at node indices (into edges): the value dof and, Hermite only, the slope dof w'."""
    nodes = np.asarray(nodes, dtype=np.int64).ravel()
    if nodes.size and (nodes.min() < 0 or nodes.max() >= dofs.n_nodes):
        raise ValueError(f'node_dofs: node indices must lie in [0, {dofs.n_nodes})')
    if slope and dofs.family != 'hermite':
        raise ValueError('node_dofs: slope dofs exist on Hermite elements only')
    stride = 1 if (dofs.family == 'lagrange' and dofs.order == 1) else 2
    parts = []
    if value:
        parts.append(stride * nodes)
    if slope:
        parts.append(2 * nodes + 1)
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)


def end_dofs(dofs, face, value=True, slope=False) -> np.ndarray:
    """Dofs of a line end 'x_lo' / 'x_hi' (held / clamped ends: slope=True also holds w' on Hermite)."""
    if face not in _FACES:
        raise ValueError(f"end_dofs: face must be one of {_FACES}, got {face!r}")
    return node_dofs(dofs, [0 if face == 'x_lo' else dofs.n_nodes - 1], value, slope)


def nodes_at(dofs, positions, tolerance=POSITION_TOLERANCE) -> np.ndarray:
    """int64 node indices at positions [m]; ValueError when no node lies within tolerance x length (held points)."""
    positions = np.asarray(positions, dtype=np.float64).ravel()
    nearest = np.clip(np.searchsorted(dofs.edges, positions), 1, dofs.n_nodes - 1)
    nearest = np.where(np.abs(dofs.edges[nearest - 1] - positions) <= np.abs(dofs.edges[nearest] - positions),
                       nearest - 1, nearest)
    missing = np.abs(dofs.edges[nearest] - positions) > tolerance * dofs.length
    if np.any(missing):
        raise ValueError(f'nodes_at: no node at x = {positions[missing].tolist()} m')
    return nearest.astype(np.int64)


def active_dofs(dofs, elements=None) -> np.ndarray:
    """bool (n_dofs,) dofs touched by the selected elements (the rest belong to excluded elements: hold them)."""
    mask = np.zeros(dofs.n_dofs, dtype=bool)
    mask[dofs.element_dofs[_element_list(dofs, elements)].ravel()] = True
    return mask


# ---- solves ----
def band_storage(matrix):
    """((lower, upper), bands) of a square sparse matrix in LAPACK band storage (bands[upper + i - j, j] = a_ij,
    duplicates summed): the scipy.linalg.solve_banded layout."""
    coo = matrix.tocoo()
    n_rows = matrix.shape[0]
    if matrix.shape != (n_rows, n_rows):
        raise ValueError(f'band_storage: square matrix needed, got shape {matrix.shape}')
    offsets = coo.col.astype(np.int64) - coo.row.astype(np.int64)
    lower = int(max(0, -offsets.min())) if offsets.size else 0
    upper = int(max(0, offsets.max())) if offsets.size else 0
    bands = np.zeros((lower + upper + 1, n_rows), dtype=np.result_type(coo.data, np.float64))
    scatter_add(bands.reshape(-1), (upper - offsets) * n_rows + coo.col, coo.data)   # a view: summed in place
    return (lower, upper), bands


def _equilibration(matrix) -> np.ndarray:
    """(n,) symmetric diagonal scale 1 / sqrt(|a_ii|) (1 where a_ii = 0): w and w' rows of Hermite operators differ by
    ~1 / h^2 in SI units."""
    diagonal = np.abs(matrix.diagonal())
    scale = np.ones(diagonal.size)
    nonzero = diagonal > 0.0
    scale[nonzero] = 1.0 / np.sqrt(diagonal[nonzero])
    return scale


def _scaled(matrix, scale):
    """CSR S A S of a diagonal scale S."""
    from scipy.sparse import diags
    return (diags(scale) @ matrix @ diags(scale)).tocsr()


def band_solve(matrix, rhs) -> np.ndarray:
    """Solution of matrix x = rhs by a banded LU (LAPACK gbsv: complex non-Hermitian allowed) of the symmetrically
    equilibrated matrix; refactorises per call; LinAlgError when singular (e.g. a dof no element touches left free)."""
    import scipy.linalg
    scale = _equilibration(matrix)
    (lower, upper), bands = band_storage(_scaled(matrix, scale))
    rhs = np.asarray(rhs)
    bands = bands.astype(np.result_type(bands, rhs))
    return scale * scipy.linalg.solve_banded((lower, upper), bands, (scale * rhs).astype(bands.dtype),
                                             check_finite=False)


def solve_line_system(matrix, rhs, fixed=(), values=0.0) -> np.ndarray:
    """Full (n_dofs,) solution of matrix u = rhs with the fixed dofs held at values (number or per dof), by
    eliminate_dirichlet + band_solve (the reduced matrix keeps the band)."""
    system = eliminate_dirichlet(matrix, rhs, np.asarray(fixed, dtype=np.int64), values)
    if system.free.size == 0:
        return system.expand(np.zeros(0))
    return system.expand(band_solve(system.matrix, system.rhs))


def line_eigenpairs(stiffness, mass, keep, count, target=0.0):
    """(eigenvalues (count,), vectors (n_dofs, count)) of K u = lambda M u on the kept dofs (bool mask or indices),
    nearest to target (lambda = omega^2 for K - omega^2 M), sorted by |lambda - target|; removed dofs are 0 in the
    vectors. Dense up to DENSE_EIGEN_DOFS kept dofs, else ARPACK shift-invert (eigsh when K and M are real symmetric,
    else eigs, whose M must be real symmetric positive semi-definite); both on the equilibrated pencil S K S, S M S."""
    import scipy.linalg
    import scipy.sparse.linalg
    n_dofs = stiffness.shape[0]
    keep = np.asarray(keep)
    kept = np.flatnonzero(keep) if keep.dtype == bool else np.unique(keep.astype(np.int64))
    reduced_stiffness = restrict(stiffness, kept)
    scale = _equilibration(reduced_stiffness)
    reduced_stiffness, reduced_mass = _scaled(reduced_stiffness, scale), _scaled(restrict(mass, kept), scale)
    count = int(count)
    if not 0 < count <= kept.size:
        raise ValueError(f'line_eigenpairs: count must lie in [1, {kept.size}] (the kept dofs), got {count}')
    real = not (np.iscomplexobj(reduced_stiffness.data) or np.iscomplexobj(reduced_mass.data)
                or np.iscomplexobj(target))

    def symmetric(matrix):
        scale = float(np.abs(matrix.data).max(initial=0.0))
        return float(np.abs((matrix - matrix.T).data).max(initial=0.0)) <= 1e-12 * max(scale, 1e-300)
    hermitian = real and symmetric(reduced_stiffness) and symmetric(reduced_mass)
    if kept.size <= max(DENSE_EIGEN_DOFS, 2 * count + 2):
        if hermitian:
            eigenvalues, vectors = scipy.linalg.eigh(reduced_stiffness.toarray(), reduced_mass.toarray())
        else:
            eigenvalues, vectors = scipy.linalg.eig(reduced_stiffness.toarray(), reduced_mass.toarray())
    elif hermitian:
        eigenvalues, vectors = scipy.sparse.linalg.eigsh(reduced_stiffness.tocsc(), k=count,
                                                         M=reduced_mass.tocsc(), sigma=float(target), which='LM')
    else:
        eigenvalues, vectors = scipy.sparse.linalg.eigs(reduced_stiffness.tocsc().astype(np.complex128), k=count,
                                                        M=reduced_mass.tocsc(), sigma=target, which='LM')
    chosen = np.argsort(np.abs(eigenvalues - target), kind='stable')[:count]
    full = np.zeros((n_dofs, count), dtype=vectors.dtype)
    full[kept] = scale[:, None] * vectors[:, chosen]
    return eigenvalues[chosen], full
