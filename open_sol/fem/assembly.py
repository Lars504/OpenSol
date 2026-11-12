"""Vectorised FEM assembly (mesh plan 2.4): element blocks by batched matmul / einsum over quadrature points in element
chunks, COO -> scipy CSR; Lagrange stiffness / mass / load, Nedelec curl-curl / mass / gradient coupling, edge (Robin / flux) terms,
Dirichlet elimination with lifting, integrals and L2 errors. Host numpy, scipy inside functions.

Coefficients: a number; an array over mesh elements (n_elements,) scalar, (n_elements, 2) diagonal or
(n_elements, 2, 2) tensor; or a callable f(points (m, q, 2), elements (m,)) -> (m, q) or (m, q, 2, 2) values at the
quadrature points (PML stretch tensors, position-dependent sources). Complex values give complex matrices."""
import functools
import hashlib
import weakref
from dataclasses import dataclass

import numpy as np

from .elements import (curved_elements, edge_geometry, edge_reference_points, element_geometry, lagrange_basis,
                       lagrange_values, nedelec_basis, physical_gradients, piola_vectors)
from .fixed_pattern import scatter_add, scatter_rows
from .quadrature import CURVED_DEGREE, TRIANGLE_DEGREES, gauss_line, triangle_rule

CHUNK_ELEMENTS = 16384   # elements per element-block chunk (bounds the (m, q, n, n) temporaries)
EDGE_QUADRATURE_ENTRIES = 32   # edge rules kept per DofMap (edge set x degree), then the cache is cleared
COO_BATCH_ENTRIES = 20_000_000   # CsrAccumulator: COO entries per conversion to CSR
# Square assemblies over one dof map's element rows (2D Lagrange stiffness / mass, assembly3d's Lagrange terms):
# PATTERN_REUSE_CALLS calls on one row list keep its slot map (fixed_pattern.FixedPatternCsr), later ones sum blocks
# by bincount instead of COO -> CSR (same canonical pattern, explicit zeros kept)
PATTERN_REUSE_CALLS = 4
PATTERN_MAX_SLOT_BYTES = 1 << 28   # largest kept slot map (int32 per block entry); above it COO -> CSR every time
_PATTERNS = weakref.WeakKeyDictionary()   # DofMap -> [rows digest, calls, FixedPatternCsr | None, element_dofs]


# ---- coefficients and element groups ----
def _coefficient_values(coefficient, points, elements, tensor):
    """Coefficient at quadrature points: (m, q) scalar, or (m, q, 2, 2) when tensor (scalars become c I)."""
    if callable(coefficient):
        values = np.asarray(coefficient(points, elements))
    elif np.ndim(coefficient) == 0:
        values = np.asarray(coefficient)
        return values * np.eye(2) if tensor else values
    else:
        values = np.asarray(coefficient)[elements]
        if values.ndim == 1:
            values = values[:, None]
        elif values.ndim == 2 and values.shape[1] == 2:   # diagonal (xx, yy) per element
            diagonal = np.zeros(values.shape[:1] + (2, 2), dtype=values.dtype)
            diagonal[:, 0, 0], diagonal[:, 1, 1] = values[:, 0], values[:, 1]
            values = diagonal[:, None]
        else:
            values = values[:, None]
    scalar = values.ndim <= 2
    if tensor and scalar:
        return values[..., None, None] * np.eye(2)
    if not tensor and not scalar:
        raise ValueError('assembly: a tensor coefficient was given where a scalar one is expected')
    return values


def _element_groups(dofs, degree):
    """Yield (positions into dofs.elements, rule) per element chunk (never empty): straight elements at `degree`,
    curved ones (when the geometry order is 2) at max(degree, CURVED_DEGREE)."""
    for positions, rule, _ in _element_groups_by_kind(dofs, degree):
        yield positions, rule


def _element_groups_by_kind(dofs, degree):
    """_element_groups with a third item: True for the chunks of curved elements (False: affine)."""
    if dofs.geometry_order == 2:
        curved = curved_elements(dofs.mesh, dofs.elements)
    else:
        curved = np.zeros(len(dofs.elements), dtype=bool)
    for mask, rule_degree, is_curved in ((~curved, degree, False), (curved, max(degree, CURVED_DEGREE), True)):
        positions = np.flatnonzero(mask)
        rule = triangle_rule(rule_degree)
        for start in range(0, len(positions), CHUNK_ELEMENTS):
            yield positions[start:start + CHUNK_ELEMENTS], rule, is_curved


def _weights(geometry, rule, axisymmetric):
    """(m, q) quadrature weight x det J (x r = x for axisymmetric (r, z) integrals)."""
    weights = rule.weights[None, :] * geometry.determinant
    return weights * geometry.points[..., 0] if axisymmetric else weights


def _lagrange_at(dofs, geometry, rule):
    """(values (q, n), physical gradients (m, q, n, 2)) of the dof map's Lagrange basis."""
    values, gradients = lagrange_basis(dofs.order, rule.points)
    return values, physical_gradients(gradients, geometry.inverse)


def _nedelec_at(dofs, positions, geometry, rule):
    """(physical vectors (m, q, n, 2), physical curls (m, q, n)) of the dof map's Nedelec basis (covariant Piola)."""
    signs = dofs.mesh.edge_signs[dofs.elements[positions]][:, None, :]
    vectors, curls = nedelec_basis(dofs.order, rule.points, signs)
    return piola_vectors(vectors, geometry.inverse), curls / geometry.determinant[..., None]


def _tensor_blocks(test_vectors, trial_vectors, tensor, weights):
    """(m, n_test, n_trial) sum over points of weights (m, q) x test_i . (tensor trial_j), vectors (m, q, n, d), tensor
    broadcastable to (m, q, d, d): weights folded into the test vectors, then one batched matmul over (point, component)
    (dimension-free: assembly3d uses it too)."""
    tensor = np.asarray(tensor)
    dimension = trial_vectors.shape[-1]
    trial_rows = trial_vectors.swapaxes(-1, -2)                                     # (m, q, d, n_trial) view
    lead = np.broadcast_shapes(tensor.shape[:-2], trial_rows.shape[:-2])
    mapped = np.empty(lead + (dimension, trial_rows.shape[-1]), dtype=np.result_type(tensor, trial_vectors))
    for row in range(dimension):   # tensor @ trial per component: a batched d x d matmul loops per point
        mapped_row = mapped[..., row, :]
        np.multiply(tensor[..., row, 0, None], trial_rows[..., 0, :], out=mapped_row)
        for column in range(1, dimension):
            mapped_row += tensor[..., row, column, None] * trial_rows[..., column, :]
    weighted = np.multiply(test_vectors.swapaxes(-1, -2), weights[..., None, None], order='C')   # (m, q, 2, n_test)
    count = len(weights)
    return weighted.reshape(count, -1, weighted.shape[-1]).swapaxes(1, 2) @ mapped.reshape(count, -1, mapped.shape[-1])


def append_blocks(rows, cols, values, row_dofs, blocks, column_dofs=None) -> None:
    """Append element blocks (m, r, c) at row dofs (m, r) and column dofs (m, c) (None: the row dofs) to the COO lists
    of _to_csr."""
    column_dofs = row_dofs if column_dofs is None else column_dofs
    rows.append(np.broadcast_to(row_dofs[:, :, None], blocks.shape).ravel())
    cols.append(np.broadcast_to(column_dofs[:, None, :], blocks.shape).ravel())
    values.append(blocks)


def _to_csr(rows, cols, values, shape):
    """Canonical CSR of COO lists (duplicates summed by tocsr, explicit zeros kept); int32 indices below 2^31."""
    from scipy.sparse import coo_matrix

    if not values:
        return coo_matrix(shape).tocsr()
    index_dtype = np.int32 if max(shape) < 2 ** 31 else np.int64
    data = np.concatenate([block.ravel() for block in values])
    rows = np.concatenate(rows).astype(index_dtype, copy=False)
    cols = np.concatenate(cols).astype(index_dtype, copy=False)
    return coo_matrix((data, (rows, cols)), shape=shape).tocsr()   # tocsr sums duplicates: canonical already


def kept_block_pattern(dofs, rows):
    """FixedPatternCsr of the square blocks over dofs.element_dofs[rows] (rows in block order) once PATTERN_REUSE_CALLS
    assemblies asked for that row list (one list per dof map, kept for the map's lifetime; slot map <=
    PATTERN_MAX_SLOT_BYTES), else None: a one-off build stays COO -> CSR (building the slot map costs ~3-4 of them)."""
    from .fixed_pattern import FixedPatternCsr
    digest = hashlib.blake2b(np.ascontiguousarray(rows), digest_size=16).hexdigest()
    try:
        state = _PATTERNS.get(dofs)
    except TypeError:   # not weak-referenceable: no reuse
        return None
    if state is None or state[0] != digest or state[3] is not dofs.element_dofs:
        state = [digest, 0, None, dofs.element_dofs]
        _PATTERNS[dofs] = state
    state[1] += 1
    if state[2] is None and state[1] >= PATTERN_REUSE_CALLS:
        local = dofs.element_dofs[rows]
        if 4 * local.size * local.shape[1] <= PATTERN_MAX_SLOT_BYTES:
            state[2] = FixedPatternCsr(dofs.n_dofs, [local])
    return state[2]


def blocks_csr(dofs, rows, blocks):
    """Canonical CSR (n_dofs, n_dofs) of square element blocks (m, n, n) over dofs.element_dofs[rows]: summed on the
    kept slot map of that row list (kept_block_pattern), else by COO -> CSR; explicit zeros kept either way."""
    import scipy.sparse
    shape = (dofs.n_dofs, dofs.n_dofs)
    pattern = kept_block_pattern(dofs, rows)
    if pattern is not None:
        return scipy.sparse.csr_matrix((pattern.summed(0, blocks), pattern.indices.copy(), pattern.indptr.copy()),
                                       shape=shape)
    local = dofs.element_dofs[rows]
    return _to_csr([np.broadcast_to(local[:, :, None], blocks.shape).ravel()],
                   [np.broadcast_to(local[:, None, :], blocks.shape).ravel()], [blocks], shape)


class CsrAccumulator:
    """Sum of COO element blocks (or raw entries) into one CSR, converting every COO_BATCH_ENTRIES entries (bounded
    peak; batches are summed with `+`, so exact-zero sums drop: not for a pattern refreshed in place)."""

    def __init__(self, shape):
        self.shape = shape
        self.total = None
        self._rows, self._columns, self._values, self._count = [], [], [], 0
        self._index_dtype = np.int32 if max(shape) < 2 ** 31 else np.int64

    def add(self, rows, blocks, columns=None) -> None:
        """Element blocks (m, r, c) at element row dofs (m, r) and column dofs (m, c) (None = the row dofs)."""
        columns = rows if columns is None else columns
        self.add_entries(np.broadcast_to(rows[:, :, None], blocks.shape),
                         np.broadcast_to(columns[:, None, :], blocks.shape), blocks)

    def add_entries(self, rows, columns, values) -> None:
        """COO entries of same-shape row / column / value arrays (duplicates summed)."""
        self._rows.append(np.asarray(rows).astype(self._index_dtype).ravel())
        self._columns.append(np.asarray(columns).astype(self._index_dtype).ravel())
        self._values.append(np.asarray(values).ravel())
        self._count += self._values[-1].size
        if self._count >= COO_BATCH_ENTRIES:
            self._flush()

    def _flush(self) -> None:
        import scipy.sparse
        if not self._values:
            return
        batch = scipy.sparse.coo_matrix((np.concatenate(self._values), (np.concatenate(self._rows),
                                                                         np.concatenate(self._columns))),
                                        shape=self.shape).tocsr()
        self.total = batch if self.total is None else self.total + batch
        self._rows, self._columns, self._values, self._count = [], [], [], 0

    def result(self):
        import scipy.sparse
        self._flush()
        return scipy.sparse.csr_matrix(self.shape) if self.total is None else self.total


def pattern_sum(matrices):
    """CSR sum of same-shape sparse matrices keeping every stored entry (explicit zeros too): scipy `a + b` drops exact
    zero results, so its pattern would depend on the values (dimension-free)."""
    parts = [matrix.tocoo() for matrix in matrices]
    return _to_csr([part.row for part in parts], [part.col for part in parts], [part.data for part in parts],
                   parts[0].shape)


def congruence(transform, matrix):
    """CSR T^T A T of host sparse matrices: entry by entry when T is diagonal (one stored entry per row, on the
    diagonal: a scaling; A's pattern kept, explicit zeros too, no sparse products), else the sparse products."""
    matrix, transform = matrix.tocsr(), transform.tocsr()
    n_rows = transform.shape[0]
    diagonal = (transform.shape == matrix.shape and transform.nnz == n_rows
                and np.array_equal(transform.indptr, np.arange(n_rows + 1))
                and np.array_equal(transform.indices, np.arange(n_rows)))
    if not diagonal:
        return (transform.T @ matrix @ transform).tocsr()
    scale = transform.data
    data = matrix.data.astype(np.result_type(matrix.data, scale))   # a copy: the caller's matrix stays as it is
    data *= np.repeat(scale, np.diff(matrix.indptr))
    data *= scale[matrix.indices]
    return type(matrix)((data, matrix.indices.copy(), matrix.indptr.copy()), shape=matrix.shape)


def _assemble(test, trial, degree, block, elements=None, affine=None):
    """CSR (test.n_dofs, trial.n_dofs) of element blocks block(positions, geometry, rule) -> (m, n_test, n_trial);
    elements: bool per dof map element integrated (None = all); affine(positions) -> blocks of the straight chunks
    (element-constant coefficients: reference matrices, no per-point geometry). A square term (test is trial) sums
    on the kept slot map of its element rows (blocks_csr)."""
    if test.mesh is not trial.mesh or not np.array_equal(test.elements, trial.elements):
        raise ValueError('assembly: test and trial dof maps must share mesh and elements')
    rows, cols, values, order = [], [], [], []
    for positions, rule, is_curved in _element_groups_by_kind(test, degree):
        if elements is not None:
            positions = positions[elements[positions]]
            if not positions.size:
                continue
        if affine is not None and not is_curved:
            local = affine(positions)
        else:
            geometry = element_geometry(test.mesh, test.elements[positions], rule.points, test.geometry_order)
            local = block(positions, geometry, rule)
        if test is trial:
            order.append(positions)
            values.append(local)
        else:
            append_blocks(rows, cols, values, test.element_dofs[positions], local, trial.element_dofs[positions])
    if test is trial:
        if not values:
            return _to_csr([], [], [], (test.n_dofs, test.n_dofs))
        return blocks_csr(test, np.concatenate(order), np.concatenate(values))
    return _to_csr(rows, cols, values, (test.n_dofs, trial.n_dofs))


def _affine_geometry(dofs, positions) -> tuple:
    """(J^-1 (m, 2, 2), det J (m,)) of straight elements at their centroid (the affine map: equal at every point)."""
    geometry = element_geometry(dofs.mesh, dofs.elements[positions], np.array([[1.0 / 3.0, 1.0 / 3.0]]),
                                dofs.geometry_order)
    return geometry.inverse[:, 0], geometry.determinant[:, 0]


@functools.lru_cache(maxsize=None)
def _reference_products(order, degree, kind) -> np.ndarray:
    """Read-only reference integrals at triangle_rule(degree): 'mass' (n n,) sum_q w phi_i phi_j; 'stiffness' (4, n n)
    sum_q w dphi_i/dxi_a dphi_j/dxi_b in row 2 a + b."""
    rule = triangle_rule(degree)
    values, gradients = lagrange_basis(order, rule.points)   # (q, n), (q, n, 2)
    if kind == 'mass':
        products = np.einsum('q,qi,qj->ij', rule.weights, values, values).ravel()
    else:
        products = np.einsum('q,qia,qjb->abij', rule.weights, gradients, gradients).reshape(4, -1)
    products.flags.writeable = False
    return products


def _element_constant(coefficient) -> bool:
    """A number or per-element values (not a callable): straight elements take the reference matrices."""
    return not callable(coefficient)


def _degree(dofs, extra=0):
    """Default rule degree: 2 x the polynomial order of the basis (+ extra: axisymmetric r, coefficient order),
    capped at the highest available triangle rule (P2 axisymmetric loads ask for 7)."""
    return min(max(2 * dofs.order + extra, 2), TRIANGLE_DEGREES[-1])


# ---- Lagrange ----
def assemble_stiffness(dofs, coefficient=1.0, *, degree=None, axisymmetric=False, elements=None):
    """CSR of int (K grad phi_j) . grad phi_i dA (x r when axisymmetric); K scalar / diagonal / tensor; elements: bool
    per dof map element integrated (None = all)."""
    degree = _degree(dofs, int(axisymmetric) - 2) if degree is None else degree

    def block(positions, geometry, rule):
        _, gradients = _lagrange_at(dofs, geometry, rule)
        values = _coefficient_values(coefficient, geometry.points, dofs.elements[positions], tensor=True)
        return _tensor_blocks(gradients, gradients, values, _weights(geometry, rule, axisymmetric))

    def affine(positions):   # det J (J^-1 K J^-T)_ab against the 4 reference gradient products
        inverse, determinant = _affine_geometry(dofs, positions)
        tensor = np.asarray(_coefficient_values(coefficient, None, dofs.elements[positions], tensor=True))
        tensor = np.broadcast_to(tensor[:, 0] if tensor.ndim == 4 else tensor, (len(positions), 2, 2))
        mapped = np.einsum('mab,mbc,mdc->mad', inverse, tensor, inverse) * determinant[:, None, None]
        n_local = dofs.n_local
        return (mapped.reshape(len(positions), 4) @ _reference_products(dofs.order, int(degree), 'stiffness')
                ).reshape(len(positions), n_local, n_local)
    return _assemble(dofs, dofs, degree, block, elements,
                     affine if _element_constant(coefficient) and not axisymmetric else None)


def assemble_mass(dofs, coefficient=1.0, *, degree=None, axisymmetric=False, elements=None):
    """CSR of int c phi_i phi_j dA (x r when axisymmetric); elements: bool per dof map element integrated (None =
    all)."""
    degree = _degree(dofs, int(axisymmetric)) if degree is None else degree

    def block(positions, geometry, rule):
        values = lagrange_values(dofs.order, rule.points)
        scale = _coefficient_values(coefficient, geometry.points, dofs.elements[positions], tensor=False)
        weights = np.broadcast_to(scale, geometry.determinant.shape) * _weights(geometry, rule, axisymmetric)
        products = (values[:, :, None] * values[:, None, :]).reshape(len(values), -1)   # (q, n n)
        return (weights @ products).reshape(len(weights), values.shape[1], values.shape[1])

    def affine(positions):   # c det J x the reference mass
        _, determinant = _affine_geometry(dofs, positions)
        scale = np.asarray(_coefficient_values(coefficient, None, dofs.elements[positions], tensor=False))
        scale = np.broadcast_to(scale.reshape(-1) if scale.ndim else scale, (len(positions),)) * determinant
        n_local = dofs.n_local
        return (scale[:, None] * _reference_products(dofs.order, int(degree), 'mass')[None, :]).reshape(
            len(positions), n_local, n_local)
    return _assemble(dofs, dofs, degree, block, elements,
                     affine if _element_constant(coefficient) and not axisymmetric else None)


def _cached(dofs, key, build):
    """dofs.element_quadrature[key], built once by build() and made read-only (the map's geometry never changes)."""
    found = dofs.element_quadrature.get(key)
    if found is None:
        found = build()
        for array in (found if isinstance(found, tuple) else (found,)):
            array.flags.writeable = False
        dofs.element_quadrature[key] = found
    return found


def element_basis_integrals(dofs, degree, axisymmetric=False) -> tuple:
    """(measure (m,), int phi_n (m, n)) per element of a Lagrange map (x r when axisymmetric) at the rule of `degree`
    (curved elements: max(degree, CURVED_DEGREE)); cached read-only on the map."""
    def build():
        measure = np.zeros(len(dofs.elements))
        integrals = np.zeros((len(dofs.elements), dofs.n_local))
        for positions, rule in _element_groups(dofs, degree):
            geometry = element_geometry(dofs.mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
            weights = _weights(geometry, rule, axisymmetric)
            measure[positions] = weights.sum(axis=1)
            integrals[positions] = weights @ lagrange_values(dofs.order, rule.points)
        return measure, integrals
    return _cached(dofs, ('basis', int(degree), bool(axisymmetric)), build)


def element_gradient_integrals(dofs, degree, axisymmetric=False) -> np.ndarray:
    """int grad phi_n (m, n, 2) per element of a Lagrange map (x r when axisymmetric), rule as element_basis_integrals;
    cached read-only on the map (the element-mean gradient of a field is then one small einsum)."""
    def build():
        integrals = np.zeros((len(dofs.elements), dofs.n_local, 2))
        for positions, rule in _element_groups(dofs, degree):
            geometry = element_geometry(dofs.mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
            _, gradients = _lagrange_at(dofs, geometry, rule)
            integrals[positions] = np.einsum('mqnc,mq->mnc', gradients, _weights(geometry, rule, axisymmetric))
        return integrals
    return _cached(dofs, ('gradient', int(degree), bool(axisymmetric)), build)


def scatter_local(n_dofs, element_dofs, local) -> np.ndarray:
    """Vector (n_dofs,) of the per-element local values (m, n) summed into their dofs (fixed_pattern.scatter_add into
    zeros); complex local values give a complex vector."""
    local = np.asarray(local)
    result = np.zeros(n_dofs, dtype=np.complex128 if np.iscomplexobj(local) else np.float64)
    scatter_add(result, element_dofs, local)
    return result


def assemble_load(dofs, source=1.0, *, degree=None, axisymmetric=False, elements=None) -> np.ndarray:
    """Vector of int f phi_i dA (x r when axisymmetric); f scalar per element / at quadrature points; elements: bool per
    dofs element integrated (None = all; a source that is 0 elsewhere skips those). A number or per-element f takes the
    cached element_basis_integrals (no geometry per call: per-step / per-frequency loads)."""
    degree = _degree(dofs, int(axisymmetric) + 2) if degree is None else degree
    if not callable(source) and np.ndim(source) <= 1:
        coefficient = np.asarray(source)
        per_element = coefficient[dofs.elements] if coefficient.ndim == 1 else np.broadcast_to(
            coefficient, dofs.elements.shape)
        _, integrals = element_basis_integrals(dofs, degree, axisymmetric)
        local = integrals * per_element[:, None]
        if elements is not None:
            local = np.where(np.asarray(elements, dtype=bool)[:, None], local, 0.0)
        return scatter_local(dofs.n_dofs, dofs.element_dofs, local)
    result = None
    for positions, rule in _element_groups(dofs, degree):
        if elements is not None:
            positions = positions[elements[positions]]
            if not positions.size:
                continue
        geometry = element_geometry(dofs.mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
        values = lagrange_values(dofs.order, rule.points)
        scale = _coefficient_values(source, geometry.points, dofs.elements[positions], tensor=False)
        weights = np.broadcast_to(scale, geometry.determinant.shape) * _weights(geometry, rule, axisymmetric)
        local = weights @ values
        if result is None:
            result = np.zeros(dofs.n_dofs, dtype=local.dtype)
        elif local.dtype != result.dtype:
            result = result.astype(np.result_type(result, local))
        scatter_add(result, dofs.element_dofs[positions], local)
    return np.zeros(dofs.n_dofs) if result is None else result


# ---- Nedelec ----
def assemble_curl_curl(dofs, coefficient=1.0, *, degree=None):
    """CSR of int c curl N_i curl N_j dA (c scalar: 1 / mu_z, or the PML s_x s_y / s_z ratio at points)."""
    degree = max(2 * dofs.order - 2, 2) if degree is None else degree

    def block(positions, geometry, rule):
        _, curls = _nedelec_at(dofs, positions, geometry, rule)
        scale = _coefficient_values(coefficient, geometry.points, dofs.elements[positions], tensor=False)
        weights = np.broadcast_to(scale, geometry.determinant.shape) * _weights(geometry, rule, False)
        return (curls * weights[..., None]).swapaxes(1, 2) @ curls
    return _assemble(dofs, dofs, degree, block)


def assemble_vector_mass(dofs, coefficient=1.0, *, degree=None):
    """CSR of int (T N_j) . N_i dA, T scalar / diagonal / tensor (eps_t, 1 / mu_t, PML tensors)."""
    degree = 2 * dofs.order if degree is None else degree

    def block(positions, geometry, rule):
        vectors, _ = _nedelec_at(dofs, positions, geometry, rule)
        values = _coefficient_values(coefficient, geometry.points, dofs.elements[positions], tensor=True)
        return _tensor_blocks(vectors, vectors, values, _weights(geometry, rule, False))
    return _assemble(dofs, dofs, degree, block)


def assemble_gradient_coupling(nedelec, lagrange, coefficient=1.0, *, degree=None):
    """CSR (nedelec.n_dofs, lagrange.n_dofs) of int (T grad phi_j) . N_i dA (the Lee-Sun-Cendes E_t / E_z block)."""
    degree = nedelec.order + max(lagrange.order - 1, 0) if degree is None else degree
    if nedelec.geometry_order != lagrange.geometry_order:
        raise ValueError('assemble_gradient_coupling: both dof maps need the same geometry order')

    def block(positions, geometry, rule):
        vectors, _ = _nedelec_at(nedelec, positions, geometry, rule)
        _, gradients = _lagrange_at(lagrange, geometry, rule)
        values = _coefficient_values(coefficient, geometry.points, nedelec.elements[positions], tensor=True)
        return _tensor_blocks(vectors, gradients, values, _weights(geometry, rule, False))
    return _assemble(nedelec, lagrange, max(degree, 2), block)


# ---- edges (natural / Robin / flux terms) ----
def _edge_sides(dofs, edges):
    """(edges, positions of their subset element, local edge index, element sign of it) of mesh edges with a subset
    element (the first one found; an interior edge of the subset is taken from its first element)."""
    edges = np.asarray(edges, dtype=np.int64)
    mesh = dofs.mesh
    pairs = mesh.edge_elements[edges]
    first_inside = dofs.element_position[pairs[:, 0]] >= 0
    element = np.where(first_inside, pairs[:, 0], pairs[:, 1])
    keep = element >= 0
    keep[keep] = dofs.element_position[element[keep]] >= 0
    edges, element = edges[keep], element[keep]
    local = np.argmax(mesh.element_edges[element] == edges[:, None], axis=1)
    signs = mesh.edge_signs[element, local]
    return edges, dofs.element_position[element], local, signs


def _edge_quadrature(dofs, edges, degree):
    """(kept edges (k,), element dofs (k, n), basis values (k, q, n), Gauss weights (q,), |dx/dt| (k, q), points (k, q,
    2)) of mesh edges at the edge rule of `degree`; kept per (edges, degree) in `dofs.edge_quadrature`, read-only."""
    edges = np.ascontiguousarray(edges, dtype=np.int64)
    key = (int(degree), hashlib.blake2b(edges.tobytes(), digest_size=16).digest())
    cache = dofs.edge_quadrature
    found = cache.get(key)
    if found is None:
        if len(cache) >= EDGE_QUADRATURE_ENTRIES:
            cache.clear()
        kept, positions, local, signs = _edge_sides(dofs, edges)
        t, weights = gauss_line(degree + (2 if dofs.geometry_order == 2 else 0))
        points, lengths, _ = edge_geometry(dofs.mesh, kept, t, dofs.geometry_order)
        values = lagrange_values(dofs.order, edge_reference_points(signs, local, t))
        owned = (kept, dofs.element_dofs[positions], values, lengths, points)
        for array in owned:
            array.flags.writeable = False
        found = cache[key] = owned[:3] + (weights,) + owned[3:]
    return found


def _edge_terms(dofs, edges, coefficient, degree, axisymmetric):
    """(element dofs (k, n), basis values (k, q, n), weights (k, q) incl. the coefficient) on mesh edges."""
    edges, element_dofs, values, weights, lengths, points = _edge_quadrature(dofs, edges, degree)
    if callable(coefficient):
        scale = np.asarray(coefficient(points, edges))
    elif np.ndim(coefficient) == 0:
        scale = np.asarray(coefficient)
    else:   # per mesh edge
        scale = np.asarray(coefficient)[edges][:, None]
    line_weights = np.broadcast_to(scale, lengths.shape) * weights[None, :] * lengths
    if axisymmetric:
        line_weights = line_weights * points[..., 0]
    return element_dofs, values, line_weights


def assemble_edge_mass(dofs, edges, coefficient=1.0, *, degree=None, axisymmetric=False):
    """CSR of int_edges h phi_i phi_j ds (Robin / convection h; coefficient number, per mesh edge or callable
    f(points (k, q, 2), edges (k,)))."""
    degree = 2 * dofs.order + int(axisymmetric) if degree is None else degree
    element_dofs, values, weights = _edge_terms(dofs, edges, coefficient, degree, axisymmetric)
    rows, cols, blocks = [], [], []
    append_blocks(rows, cols, blocks, element_dofs, np.einsum('kqi,kqj,kq->kij', values, values, weights))
    return _to_csr(rows, cols, blocks, (dofs.n_dofs, dofs.n_dofs))


def assemble_edge_load(dofs, edges, source=1.0, *, degree=None, axisymmetric=False) -> np.ndarray:
    """Vector of int_edges g phi_i ds (inward flux / Robin h T_ext; source as assemble_edge_mass's coefficient)."""
    degree = dofs.order + 2 + int(axisymmetric) if degree is None else degree
    element_dofs, values, weights = _edge_terms(dofs, edges, source, degree, axisymmetric)
    local = np.einsum('kqi,kq->ki', values, weights)
    return scatter_local(dofs.n_dofs, element_dofs, local)


# ---- Dirichlet ----
@dataclass(frozen=True, eq=False)
class DirichletSystem:
    """matrix u_free = rhs after eliminating fixed dofs (values lifted into rhs); expand() gives the full vector."""
    matrix: object          # scipy CSR (n_free, n_free)
    rhs: np.ndarray         # (n_free,)
    free: np.ndarray        # int64 free dof indices
    fixed: np.ndarray       # int64 fixed dof indices
    fixed_values: np.ndarray
    n_dofs: int

    def expand(self, free_solution) -> np.ndarray:
        free_solution = np.asarray(free_solution)
        full = np.zeros(self.n_dofs, dtype=np.result_type(free_solution, self.fixed_values))
        full[self.free] = free_solution
        full[self.fixed] = self.fixed_values
        return full


def eliminate_dirichlet(matrix, rhs, fixed, values=0.0) -> DirichletSystem:
    """Eliminate fixed dofs (values number or per fixed dof) from matrix u = rhs and lift them into the rhs."""
    from scipy.sparse import csr_matrix

    matrix = csr_matrix(matrix)
    n_dofs = matrix.shape[0]
    given = np.asarray(fixed, dtype=np.int64).ravel()
    fixed, first = np.unique(given, return_index=True)   # values follow their dofs through the sort / dedup
    fixed_values = np.broadcast_to(np.asarray(values), given.shape)[first].copy()
    free_mask = np.ones(n_dofs, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    lifted = np.zeros(n_dofs, dtype=np.result_type(matrix.dtype, fixed_values))
    lifted[fixed] = fixed_values
    reduced_rhs = (np.asarray(rhs) - matrix @ lifted)[free]
    return DirichletSystem(matrix[free][:, free].tocsr(), reduced_rhs, free, fixed, fixed_values, n_dofs)


def restrict(matrix, keep):
    """CSR matrix[keep][:, keep] (eigenproblems with essential conditions)."""
    from scipy.sparse import csr_matrix

    keep = np.asarray(keep)
    keep = np.flatnonzero(keep) if keep.dtype == bool else keep
    return csr_matrix(matrix)[keep][:, keep].tocsr()


# ---- interpolation, integrals, errors ----
def interpolate(dofs, function) -> np.ndarray:
    """Nodal interpolant of f(x, y) (vectorised over float64 arrays) at the Lagrange dof points."""
    if dofs.family != 'lagrange':
        raise ValueError('interpolate: Lagrange dof maps only')
    points = dofs.dof_points()
    return np.asarray(function(points[:, 0], points[:, 1])) * np.ones(len(points))


def element_areas(dofs, axisymmetric=False) -> np.ndarray:
    """Float64 (m,) areas (x r: ring cross-section moments) of the dof map's elements in its geometry order."""
    result = np.zeros(len(dofs.elements))
    for positions, rule in _element_groups(dofs, 2 + int(axisymmetric)):
        geometry = element_geometry(dofs.mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
        result[positions] = _weights(geometry, rule, axisymmetric).sum(axis=1)
    return result


def nodal_average(dofs, local_values) -> np.ndarray:
    """(n_dofs, ...) mean over the map's elements sharing each dof of per-element values at their local dof points
    (m, n_local, ...): the smoothed (continuous) field of a discontinuous one such as strain, stress or B."""
    return element_dof_mean(dofs.n_dofs, dofs.element_dofs, local_values)


def element_dof_mean(n_dofs, element_dofs, local_values) -> np.ndarray:
    """(n_dofs, ...) mean of local values (m, n_local, ...) over the elements (element_dofs (m, n_local)) sharing each
    dof; dofs no element touches read 0 (nodal_average, 2D and 3D)."""
    local_values = np.asarray(local_values, dtype=np.float64)
    trailing = local_values.shape[2:]
    flat_dofs = np.asarray(element_dofs).ravel()
    counts = np.bincount(flat_dofs, minlength=n_dofs).astype(np.float64)
    result = scatter_rows(n_dofs, flat_dofs, local_values.reshape((-1,) + trailing))
    return result / np.maximum(counts, 1.0).reshape((-1,) + (1,) * len(trailing))


def integrate(dofs, coefficients, *, axisymmetric=False):
    """int u_h dA (x r when axisymmetric) of a Lagrange field (coefficients per dof)."""
    return assemble_load(dofs, 1.0, axisymmetric=axisymmetric) @ np.asarray(coefficients)


# ---- affine P1 box-method weights (edge-averaged FE / Scharfetter-Gummel on triangles: drift_diffusion) ----
def _element_indices(mesh, elements):
    """Int64 element indices of None (all), indices or a bool mask."""
    if elements is None:
        return np.arange(mesh.n_elements)
    elements = np.asarray(elements)
    return np.flatnonzero(elements) if elements.dtype == bool else elements.astype(np.int64)


def p1_gradients(mesh, elements=None) -> tuple:
    """(gradients (m, 3, 2) of the affine P1 basis, areas (m,)) of mesh elements (None = all, indices or a bool mask) on
    their corner nodes (curved edges as chords)."""
    corners = mesh.points[mesh.triangles[_element_indices(mesh, elements)]]   # (m, 3, 2), counter-clockwise
    edges = np.roll(corners, -2, axis=1) - np.roll(corners, -1, axis=1)        # local edge k: node k + 1 -> k + 2
    areas = 0.5 * ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
                   - (corners[:, 2, 0] - corners[:, 0, 0]) * (corners[:, 1, 1] - corners[:, 0, 1]))
    gradients = np.stack([-edges[..., 1], edges[..., 0]], axis=-1) / (2.0 * areas[:, None, None])
    return gradients, areas


def p1_edge_weights(mesh, elements=None, coefficient=None, axisymmetric=False) -> np.ndarray:
    """(m, 3) affine P1 stiffness weights w_k = -K_(k+1)(k+2) of local edge k (cot / 2 of the angle at node k for a unit
    coefficient); coefficient None / (m,) / (m, 2) diagonal / (m, 2, 2); x the centroid r when axisymmetric (per
    radian). Sum over a node's edges of w (u_node - u_other) = (K u) at the node."""
    indices = _element_indices(mesh, elements)
    gradients, areas = p1_gradients(mesh, indices)
    count = len(areas)
    if coefficient is None:
        flux = gradients
    else:
        values = np.asarray(coefficient, dtype=np.float64)
        if values.shape == (count,):
            flux = gradients * values[:, None, None]
        elif values.shape == (count, 2):
            flux = gradients * values[:, None, :]
        elif values.shape == (count, 2, 2):
            flux = np.einsum('mcd,mkd->mkc', values, gradients)
        else:
            raise ValueError(f'p1_edge_weights: coefficient shape {values.shape} fits none of ({count},), ({count}, 2), '
                             f'({count}, 2, 2)')
    weights = -areas[:, None] * np.einsum('mkc,mkc->mk', np.roll(gradients, -1, axis=1), np.roll(flux, -2, axis=1))
    if axisymmetric:
        weights = weights * mesh.points[mesh.triangles[indices]][..., 0].mean(axis=1)[:, None]
    return weights


def p1_lumped_weights(mesh, elements=None, axisymmetric=False) -> np.ndarray:
    """(m, 3) lumped P1 mass (box-volume share) of each corner: area / 3, or int r phi_k dA = area (r_k + 3 r_c) / 12
    when axisymmetric (per radian)."""
    indices = _element_indices(mesh, elements)
    _, areas = p1_gradients(mesh, indices)
    if not axisymmetric:
        return np.repeat(areas[:, None] / 3.0, 3, axis=1)
    radii = mesh.points[mesh.triangles[indices]][..., 0]
    return areas[:, None] * (radii + 3.0 * radii.mean(axis=1)[:, None]) / 12.0


def l2_error(dofs, coefficients, exact, *, degree=CURVED_DEGREE) -> float:
    """sqrt(int (u_h - u)^2 dA) of a Lagrange field against exact(x, y) on the dof map's (mapped) elements."""
    coefficients = np.asarray(coefficients)
    total = 0.0
    for positions, rule in _element_groups(dofs, degree):
        geometry = element_geometry(dofs.mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
        values = lagrange_values(dofs.order, rule.points)
        approximate =np.einsum('qn,mn->mq', values, coefficients[dofs.element_dofs[positions]])
        difference = approximate - exact(geometry.points[..., 0], geometry.points[..., 1])
        total += float((np.abs(difference) ** 2 * _weights(geometry, rule, False)).sum())
    return float(np.sqrt(total))
