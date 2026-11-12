"""Assembly on a TetrahedronMesh (igor3d_plan section 7, T6): the Nedelec curl-curl and vector mass matrices with
diagonal-tensor coefficients at quadrature points (PML stretch tensors, diagonal anisotropy); Lagrange P1 / P2
stiffness (full 3 x 3 tensors), mass, volume / point / gradient loads, face (Robin / Neumann) masses and loads, element
integrals, field values at rule points; Dirichlet elimination (fem.assembly's, dimension-free). Affine elements. Host
numpy, scipy inside functions."""
# Coefficients: a number, per-element values over the WHOLE mesh ((n_elements,) isotropic, (n_elements, 3) diagonal,
# Lagrange stiffness also (n_elements, 3, 3) full), or a callable f(points (m, q, 3), mesh elements (m,)) -> (m, q) /
# (m, q, 3) / (m, q, 3, 3) at the quadrature points. Face coefficients: a number, per mesh face (n_faces,), or f(points
# (k, q, 3), faces (k,)) -> (k, q). The default rule degree 2 x degree is exact for affine elements with
# element-constant coefficients; a varying coefficient (PML, x / y / z expressions) wants 2 more. Unconjugated
# products: complex coefficients give complex symmetric matrices. `elements` = None (all), indices or a bool mask; the
# matrices keep the full dof numbering (rows of dofs no listed element touches stay empty).
# Lagrange stiffness / mass (S4): an element-constant coefficient (number / per-element values) takes the affine
# reference form (one (m, 9) @ (9, n n) product, or c |det J| x the reference mass), only callables (and the varying
# elements of a SplitCoefficient: the PML shell) are integrated point by point. A dof map's element set assembled
# PATTERN_REUSE_CALLS times keeps its slot map (fixed_pattern.FixedPatternCsr): later terms and re-assemblies sum
# blocks by bincount instead of COO -> CSR (same canonical pattern, explicit zeros kept).
import functools
import hashlib
from dataclasses import dataclass

import numpy as np

from .assembly import (PATTERN_REUSE_CALLS, DirichletSystem, _tensor_blocks, _to_csr, append_blocks,   # noqa: F401
                       blocks_csr, eliminate_dirichlet, restrict)   # DirichletSystem, eliminate_dirichlet, restrict,
#                                                                     PATTERN_REUSE_CALLS re-exported
from .elements3d import (lagrange_basis_3d, nedelec_basis_3d, piola_curls_3d, piola_vectors_3d,
                         tetrahedron_geometry)
from .fixed_pattern import scatter_add
from .quadrature3d import TETRAHEDRON_MAX_DEGREE, tetrahedron_rule, triangle_face_rule

CHUNK_ELEMENTS = 1000   # elements per vectorised block: (chunk, q <= 64, 20, 3) complex arrays stay <= ~60 MB
FACE_QUADRATURE_ENTRIES = 32      # face sets kept per DofMap3D (_face_entities), then the cache is cleared


def element_chunks_3d(mesh, elements):
    """Yield (chunk, tetrahedron_geometry of it) per CHUNK_ELEMENTS block of an element index list."""
    for start in range(0, len(elements), CHUNK_ELEMENTS):
        chunk = elements[start:start + CHUNK_ELEMENTS]
        yield chunk, tetrahedron_geometry(mesh, chunk)


def _element_list(mesh, elements) -> np.ndarray:
    if elements is None:
        return np.arange(mesh.n_elements, dtype=np.int64)
    elements = np.asarray(elements)
    if elements.dtype == bool:
        if elements.shape != (mesh.n_elements,):
            raise ValueError(f'element mask has shape {elements.shape}, expected ({mesh.n_elements},)')
        return np.flatnonzero(elements).astype(np.int64)
    return np.unique(elements.astype(np.int64))


def coefficient_at(coefficient, points, elements, n_elements, n_points) -> np.ndarray:
    """(m, q, 3) diagonal-tensor values of a coefficient (module header forms) on elements (m,) at q points
    (points (m, q, 3), only read by callables)."""
    if callable(coefficient):
        values = np.asarray(coefficient(points, elements))
    else:
        values = np.asarray(coefficient)
        if values.ndim == 0:
            values = np.broadcast_to(values, (len(elements), n_points))
        elif values.shape[0] == n_elements and values.ndim in (1, 2):
            values = values[elements]
            values = np.broadcast_to(values[:, None] if values.ndim == 1 else values[:, None, :],
                                     (len(elements), n_points) + values.shape[1:])
        else:
            raise ValueError(f'coefficient of shape {values.shape}: expected a number, ({n_elements},), '
                             f'({n_elements}, 3) or a callable')
    if values.ndim == 2:
        values = np.repeat(values[..., None], 3, axis=2)
    if values.shape != (len(elements), n_points, 3):
        raise ValueError(f'coefficient values have shape {values.shape}, expected ({len(elements)}, {n_points}[, 3])')
    return values


def _component_blocks(test_vectors, trial_vectors) -> np.ndarray:
    """(m, n_test, n_trial) sum over points and components of test_i . trial_j, vectors (m, q, n, 3): one batched
    matmul over (point, component)."""
    count = len(test_vectors)
    test = np.ascontiguousarray(test_vectors.swapaxes(-1, -2)).reshape(count, -1, test_vectors.shape[2])
    trial = np.ascontiguousarray(trial_vectors.swapaxes(-1, -2)).reshape(count, -1, trial_vectors.shape[2])
    return test.swapaxes(1, 2) @ trial


def assemble_nedelec_3d(dofs, curl_coefficient=None, mass_coefficient=None, elements=None, degree=None) -> tuple:
    """(K, M) scipy CSR (n_dofs, n_dofs): K = int curl W_i . nu curl W_j, M = int W_i . eps W_j over `elements` (None =
    all, indices or a bool mask) with diagonal-tensor coefficients nu = curl_coefficient, eps = mass_coefficient (None
    skips that matrix: None returned); rule degree default 2 x the Nedelec degree. Element-constant coefficients take
    the affine reference form (J^T nu J / |det J| resp. |det J| J^-1 eps J^-T against 9 reference products); callables
    are integrated point by point."""
    import scipy.sparse
    if dofs.family != 'nedelec':
        raise ValueError('assemble_nedelec_3d needs a Nedelec dof map')
    mesh = dofs.mesh
    elements = _element_list(mesh, elements)
    rule_degree = 2 * dofs.order if degree is None else int(degree)
    rule = tetrahedron_rule(rule_degree)
    n_points = len(rule.weights)
    reference_vectors, reference_curls = nedelec_basis_3d(dofs.order, rule.points)
    needs_points = callable(curl_coefficient) or callable(mass_coefficient)
    n_local = dofs.n_local
    rows, columns, curl_blocks, mass_blocks = [], [], [], []
    for chunk, geometry in element_chunks_3d(mesh, elements):
        weights = np.abs(geometry.determinant)[:, None] * rule.weights[None, :]   # (m, q)
        points = geometry.points(rule.points) if needs_points else None
        local = dofs.element_dofs[chunk]
        rows.append(np.repeat(local, n_local, axis=1).ravel())
        columns.append(np.tile(local, (1, n_local)).ravel())
        if curl_coefficient is not None and not callable(curl_coefficient):
            nu = coefficient_at(curl_coefficient, None, chunk, mesh.n_elements, 1)[:, 0]   # (m, 3)
            jacobian = geometry.jacobian
            mapped = (np.einsum('mca,mc,mcb->mab', jacobian, nu, jacobian)
                      / np.abs(geometry.determinant)[:, None, None])
            curl_blocks.append((mapped.reshape(len(chunk), 9)
                                @ _nedelec_reference_products(dofs.order, rule_degree, 'curl')).ravel())
        elif curl_coefficient is not None:
            curls = piola_curls_3d(reference_curls, geometry)
            nu = coefficient_at(curl_coefficient, points, chunk, mesh.n_elements, n_points) * weights[..., None]
            curl_blocks.append(_component_blocks(curls * nu[:, :, None, :], curls).ravel())
        if mass_coefficient is not None and not callable(mass_coefficient):
            eps = coefficient_at(mass_coefficient, None, chunk, mesh.n_elements, 1)[:, 0]   # (m, 3)
            inverse = geometry.inverse
            mapped = (np.einsum('mac,mc,mbc->mab', inverse, eps, inverse)
                      * np.abs(geometry.determinant)[:, None, None])
            mass_blocks.append((mapped.reshape(len(chunk), 9)
                                @ _nedelec_reference_products(dofs.order, rule_degree, 'mass')).ravel())
        elif mass_coefficient is not None:
            vectors = piola_vectors_3d(reference_vectors, geometry)
            eps = coefficient_at(mass_coefficient, points, chunk, mesh.n_elements, n_points) * weights[..., None]
            mass_blocks.append(_component_blocks(vectors * eps[:, :, None, :], vectors).ravel())
    rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    columns = np.concatenate(columns) if columns else np.zeros(0, dtype=np.int64)
    shape = (dofs.n_dofs, dofs.n_dofs)

    def matrix(blocks):
        if not blocks:
            return scipy.sparse.csr_matrix(shape)
        return _to_csr([rows], [columns], [np.concatenate(blocks)], shape)

    return (None if curl_coefficient is None else matrix(curl_blocks),
            None if mass_coefficient is None else matrix(mass_blocks))


@functools.lru_cache(maxsize=None)
def _nedelec_reference_products(order, degree, kind) -> np.ndarray:
    """Read-only (9, n n) reference integrals at tetrahedron_rule(degree), row 3 a + b: 'mass' sum_q w W_i[a] W_j[b],
    'curl' sum_q w curl W_i[a] curl W_j[b] (no signs: ascending node orientation)."""
    rule = tetrahedron_rule(degree)
    vectors, curls = nedelec_basis_3d(order, rule.points)   # (q, n, 3) each
    basis = vectors if kind == 'mass' else curls
    products = np.einsum('q,qia,qjb->abij', rule.weights, basis, basis).reshape(9, -1)
    products.flags.writeable = False
    return products


# ---- Lagrange coefficients ----
def scalar_coefficient_at(coefficient, points, elements, n_elements, n_points) -> np.ndarray:
    """(m, q) values of a scalar coefficient (number, (n_elements,) or callable) on elements (m,) at q points (points
    (m, q, 3), only read by callables)."""
    if callable(coefficient):
        values = np.asarray(coefficient(points, elements))
    else:
        values = np.asarray(coefficient)
        if values.ndim == 1 and values.shape[0] == n_elements:
            values = values[elements][:, None]
        elif values.ndim != 0:
            raise ValueError(f'scalar coefficient of shape {values.shape}: expected a number, ({n_elements},) or a '
                             'callable')
    return np.broadcast_to(values, (len(elements), n_points))


def tensor_coefficient_at(coefficient, points, elements, n_elements, n_points) -> np.ndarray:
    """(m, q, 3, 3) values of a tensor coefficient (number, (n_elements,) isotropic, (n_elements, 3) diagonal,
    (n_elements, 3, 3) full or a callable giving (m, q[, 3[, 3]])) on elements (m,) at q points."""
    if callable(coefficient):
        values = np.asarray(coefficient(points, elements))
        lead = 2
    else:
        values = np.asarray(coefficient)
        lead = 0
        if values.ndim >= 1:
            if values.shape[0] != n_elements or values.shape[1:] not in ((), (3,), (3, 3)):
                raise ValueError(f'tensor coefficient of shape {values.shape}: expected a number, ({n_elements},), '
                                 f'({n_elements}, 3), ({n_elements}, 3, 3) or a callable')
            values = values[elements][:, None]
            lead = 2
    trailing = values.shape[lead:]
    if trailing == ():
        values = values[..., None, None] * np.eye(3)
    elif trailing == (3,):
        values = values[..., :, None] * np.eye(3)
    elif trailing != (3, 3):
        raise ValueError(f'tensor coefficient values of shape {values.shape}: expected (m, q[, 3[, 3]])')
    return np.broadcast_to(values, (len(elements), n_points, 3, 3))


def _require_lagrange(dofs, what, orders=(1, 2)) -> None:
    if dofs.family != 'lagrange' or dofs.order not in orders:
        raise ValueError(f"{what} needs a Lagrange dof map of order {' / '.join(map(str, orders))}")


def _rule_degree(default, degree) -> int:
    return min(max(int(default if degree is None else degree), 1), TETRAHEDRON_MAX_DEGREE)


@dataclass(frozen=True, eq=False)
class SplitCoefficient:
    """Coefficient equal to `constant` (a number / per-element form of the module header, never a callable) except on
    the `varying` elements (bool (n_elements,)), where the callable `function` gives it at the rule points (a PML
    shell): the Lagrange stiffness / mass integrate the constant part by affine reference matrices and only the varying
    elements by quadrature."""
    constant: object
    function: object
    varying: np.ndarray


@functools.lru_cache(maxsize=None)
def _reference_products(order, degree, kind) -> np.ndarray:
    """Read-only reference integrals at tetrahedron_rule(degree): 'mass' (n n,) sum_q w phi_i phi_j; 'stiffness' (9,
    n n) sum_q w dphi_i/dxi_a dphi_j/dxi_b in row 3 a + b."""
    rule = tetrahedron_rule(degree)
    values, gradients = lagrange_basis_3d(order, rule.points)   # (q, n), (q, n, 3)
    if kind == 'mass':
        products = np.einsum('q,qi,qj->ij', rule.weights, values, values).ravel()
    else:
        products = np.einsum('q,qia,qjb->abij', rule.weights, gradients, gradients).reshape(9, -1)
    products.flags.writeable = False
    return products


def _assemble_lagrange(dofs, elements, parts):
    """CSR (n_dofs, n_dofs) of element blocks over the listed elements: parts = [(selected: bool over the element list
    or None = all, block(chunk) -> (m, n, n))]; summed on the kept slot map of the element set
    (assembly.blocks_csr), else by COO -> CSR. Either way canonical, explicit zeros kept (the pattern never depends on
    the values)."""
    import scipy.sparse
    elements = _element_list(dofs.mesh, elements)
    shape = (dofs.n_dofs, dofs.n_dofs)
    if not len(elements):
        return scipy.sparse.csr_matrix(shape)
    order, blocks = [], []
    for selected, block in parts:
        positions = np.arange(len(elements)) if selected is None else np.flatnonzero(selected)
        for start in range(0, len(positions), CHUNK_ELEMENTS):
            chunk = positions[start:start + CHUNK_ELEMENTS]
            order.append(chunk)
            blocks.append(block(elements[chunk]))
    blocks = np.concatenate(blocks)
    if len(parts) > 1:   # back to element-list order
        ordered = np.empty_like(blocks)
        ordered[np.concatenate(order)] = blocks
        blocks = ordered
    return blocks_csr(dofs, elements, blocks)


def _volume_weights(geometry, rule) -> np.ndarray:
    """(m, q) quadrature weights x |det J|."""
    return np.abs(geometry.determinant)[:, None] * rule.weights[None, :]


def _coefficient_parts(coefficient, elements, affine, quadrature) -> list:
    """_assemble_lagrange parts of a coefficient: a SplitCoefficient's constant elements by affine(constant), its
    varying ones by quadrature(function); a callable by quadrature; any other form by affine."""
    if isinstance(coefficient, SplitCoefficient):
        if callable(coefficient.constant):
            raise ValueError('SplitCoefficient.constant must be a number or per-element values, not a callable')
        varying = np.asarray(coefficient.varying, dtype=bool)[elements]
        return [(~varying, affine(coefficient.constant)), (varying, quadrature(coefficient.function))]
    if callable(coefficient):
        return [(None, quadrature(coefficient))]
    return [(None, affine(coefficient))]


# ---- Lagrange volume terms ----
def assemble_lagrange_stiffness_3d(dofs, coefficient=1.0, elements=None, degree=None):
    """CSR (n_dofs, n_dofs) of int grad phi_i . C grad phi_j dV over `elements`, C scalar / diagonal / full 3 x 3, a
    callable or a SplitCoefficient (module header forms); rule degree default 2 (order - 1) (+2 for a callable that
    varies). Element-constant C: affine reference matrices (|det J| J^-1 C J^-T against 9 reference products)."""
    _require_lagrange(dofs, 'assemble_lagrange_stiffness_3d')
    mesh, n_local = dofs.mesh, dofs.n_local
    elements = _element_list(mesh, elements)
    rule_degree = _rule_degree(2 * dofs.order - 2, degree)
    rule = tetrahedron_rule(rule_degree)

    def affine(constant):
        products = _reference_products(dofs.order, rule_degree, 'stiffness')

        def block(chunk):
            geometry = tetrahedron_geometry(mesh, chunk)
            tensor = tensor_coefficient_at(constant, None, chunk, mesh.n_elements, 1)[:, 0]   # (m, 3, 3)
            mapped = np.einsum('mab,mbc,mdc->mad', geometry.inverse, tensor, geometry.inverse)
            mapped = mapped * np.abs(geometry.determinant)[:, None, None]
            return (mapped.reshape(len(chunk), 9) @ products).reshape(len(chunk), n_local, n_local)
        return block

    def quadrature(function):
        _, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)

        def block(chunk):
            geometry = tetrahedron_geometry(mesh, chunk)
            gradients = piola_vectors_3d(reference_gradients, geometry)   # (m, q, n, 3)
            points = geometry.points(rule.points) if callable(function) else None
            tensor = tensor_coefficient_at(function, points, chunk, mesh.n_elements, len(rule.weights))
            return _tensor_blocks(gradients, gradients, tensor, _volume_weights(geometry, rule))
        return block
    return _assemble_lagrange(dofs, elements, _coefficient_parts(coefficient, elements, affine, quadrature))


def assemble_lagrange_mass_3d(dofs, coefficient=1.0, elements=None, degree=None):
    """CSR (n_dofs, n_dofs) of int c phi_i phi_j dV over `elements` (P0 / P1 / P2; c scalar, a callable or a
    SplitCoefficient); rule degree default 2 x order. Element-constant c: c |det J| x the reference mass."""
    _require_lagrange(dofs, 'assemble_lagrange_mass_3d', (0, 1, 2))
    mesh, n_local = dofs.mesh, dofs.n_local
    elements = _element_list(mesh, elements)
    rule_degree = _rule_degree(2 * dofs.order, degree)
    rule = tetrahedron_rule(rule_degree)

    def affine(constant):
        products = _reference_products(dofs.order, rule_degree, 'mass')

        def block(chunk):
            geometry = tetrahedron_geometry(mesh, chunk)
            scale = scalar_coefficient_at(constant, None, chunk, mesh.n_elements, 1)[:, 0]
            scale = scale * np.abs(geometry.determinant)
            return (scale[:, None] * products[None, :]).reshape(len(chunk), n_local, n_local)
        return block

    def quadrature(function):
        values, _ = lagrange_basis_3d(dofs.order, rule.points)   # (q, n)
        products = (values[:, :, None] * values[:, None, :]).reshape(len(values), -1)   # (q, n n)

        def block(chunk):
            geometry = tetrahedron_geometry(mesh, chunk)
            points = geometry.points(rule.points) if callable(function) else None
            scale = scalar_coefficient_at(function, points, chunk, mesh.n_elements, len(rule.weights))
            return ((scale * _volume_weights(geometry, rule)) @ products).reshape(len(chunk), n_local, n_local)
        return block
    return _assemble_lagrange(dofs, elements, _coefficient_parts(coefficient, elements, affine, quadrature))


def assemble_lagrange_load_3d(dofs, source=1.0, elements=None, degree=None) -> np.ndarray:
    """(n_dofs,) int f phi_i dV over `elements` (P0 / P1 / P2; f scalar per element or at points); rule degree default
    order + 2."""
    _require_lagrange(dofs, 'assemble_lagrange_load_3d', (0, 1, 2))
    mesh = dofs.mesh
    elements = _element_list(mesh, elements)
    rule = tetrahedron_rule(_rule_degree(dofs.order + 2, degree))
    values, _ = lagrange_basis_3d(dofs.order, rule.points)
    result = np.zeros(dofs.n_dofs)
    for chunk, geometry in element_chunks_3d(mesh, elements):
        points = geometry.points(rule.points) if callable(source) else None
        scale = scalar_coefficient_at(source, points, chunk, mesh.n_elements, len(rule.weights))
        local = (scale * _volume_weights(geometry, rule)) @ values
        if local.dtype != result.dtype:
            result = result.astype(np.result_type(result, local))
        scatter_add(result, dofs.element_dofs[chunk], local)
    return result


def assemble_point_load_3d(dofs, elements, reference, values) -> np.ndarray:
    """(n_dofs,) sum_p values_p phi_i(x_p) of point sources at located points (elements (p,) >= 0, reference (p, 3)
    from PointLocator3D.locate): the exact load of a point source."""
    _require_lagrange(dofs, 'assemble_point_load_3d', (0, 1, 2))
    elements = np.asarray(elements, dtype=np.int64)
    if (elements < 0).any():
        raise ValueError('assemble_point_load_3d: a point lies outside the mesh (element -1)')
    basis, _ = lagrange_basis_3d(dofs.order, np.asarray(reference, dtype=np.float64).reshape(-1, 3))   # (p, n)
    weights = np.broadcast_to(np.asarray(values), elements.shape)
    result = np.zeros(dofs.n_dofs, dtype=np.result_type(weights.dtype, np.float64))
    scatter_add(result, dofs.element_dofs[elements], basis * weights[:, None])
    return result


def assemble_lagrange_gradient_load_3d(dofs, vector, elements=None, degree=None) -> np.ndarray:
    """(n_dofs,) int grad phi_i . g dV over `elements` (the weak divergence of a vector field: drives of a reduced
    potential); g (n_elements, 3) per element, (m, q, 3) at the tetrahedron_rule(degree) points of the listed elements
    (rule_values_3d order) or a callable f(points (m, q, 3), elements) -> (m, q, 3); rule degree default the order."""
    _require_lagrange(dofs, 'assemble_lagrange_gradient_load_3d')
    mesh = dofs.mesh
    elements = _element_list(mesh, elements)
    rule = tetrahedron_rule(_rule_degree(dofs.order, degree))
    n_points = len(rule.weights)
    _, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    given = None if callable(vector) else np.asarray(vector)
    if given is not None and given.shape not in ((mesh.n_elements, 3), (len(elements), n_points, 3)):
        raise ValueError(f'gradient load vector of shape {given.shape}: expected ({mesh.n_elements}, 3), '
                         f'({len(elements)}, {n_points}, 3) or a callable')
    result = np.zeros(dofs.n_dofs)
    chunk_offset = 0   # position of the chunk in `elements` (rows of a per-point given)
    for chunk, geometry in element_chunks_3d(mesh, elements):
        if given is None:
            values = np.asarray(vector(geometry.points(rule.points), chunk))
        elif given.ndim == 2:
            values = np.broadcast_to(given[chunk][:, None, :], (len(chunk), n_points, 3))
        else:
            values = given[chunk_offset:chunk_offset + len(chunk)]
        chunk_offset += len(chunk)
        gradients = piola_vectors_3d(reference_gradients, geometry)   # (m, q, n, 3)
        local = np.einsum('mqni,mqi,mq->mn', gradients, values, _volume_weights(geometry, rule))
        if local.dtype != result.dtype:
            result = result.astype(np.result_type(result, local))
        scatter_add(result, dofs.element_dofs[chunk], local)
    return result


def rule_values_3d(dofs, coefficients, elements=None, degree=2) -> tuple:
    """(values, derivatives, weights) of a field at the tetrahedron_rule(degree) points of `elements` (sorted, as the
    assemblers list them; callers chunk large meshes): Lagrange values (m, q) and gradients (m, q, 3), Nedelec vectors
    and curls (m, q, 3); weights (m, q) = rule weight x |det J|."""
    mesh = dofs.mesh
    elements = _element_list(mesh, elements)
    rule = tetrahedron_rule(_rule_degree(degree, None))
    local = np.asarray(coefficients)[dofs.element_dofs[elements]]   # (m, n)
    geometry = tetrahedron_geometry(mesh, elements)
    weights = _volume_weights(geometry, rule)
    if dofs.family == 'lagrange':
        basis, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
        values = np.einsum('qn,mn->mq', basis, local)
        gradients = np.einsum('mqni,mn->mqi', piola_vectors_3d(reference_gradients, geometry), local)
        return values, gradients, weights
    reference_vectors, reference_curls = nedelec_basis_3d(dofs.order, rule.points)
    vectors = np.einsum('mqni,mn->mqi', piola_vectors_3d(reference_vectors, geometry), local)
    curls = np.einsum('mqni,mn->mqi', piola_curls_3d(reference_curls, geometry), local)
    return vectors, curls, weights


# ---- Lagrange face terms (Robin / Neumann on outer or interface faces) ----
def _face_basis(order, bary) -> np.ndarray:
    """(q, 3 | 6) face-local P1 / P2 values at face barycentric points (q, 3) (DofMap3D.face_local_dofs order)."""
    if order == 1:
        return np.array(bary)
    first, second, third = bary.T
    return np.column_stack([first * (2.0 * first - 1.0), second * (2.0 * second - 1.0), third * (2.0 * third - 1.0),
                            4.0 * first * second, 4.0 * first * third, 4.0 * second * third])


def _face_entities(dofs, faces) -> tuple:
    """(face-local dofs (k, n), areas (k,)) of mesh faces, kept per face set in `dofs.face_quadrature` (read-only):
    the per-solve Neumann / Robin face loads of a fixed face set skip the edge lookup."""
    key = hashlib.blake2b(faces.tobytes(), digest_size=16).digest()
    cache = dofs.face_quadrature
    found = cache.get(key)
    if found is None:
        if len(cache) >= FACE_QUADRATURE_ENTRIES:
            cache.clear()
        found = (dofs.face_local_dofs(faces), face_areas_3d(dofs.mesh, faces))
        for array in found:
            array.flags.writeable = False
        cache[key] = found
    return found


def _face_terms(dofs, faces, coefficient, degree):
    """(face-local dofs (k, n), basis (q, n), weights (k, q) = rule weight x area x coefficient) of mesh faces."""
    _require_lagrange(dofs, 'face assembly')
    mesh = dofs.mesh
    faces = np.ascontiguousarray(faces, dtype=np.int64).reshape(-1)
    bary, rule_weights = triangle_face_rule(degree)
    local, area = _face_entities(dofs, faces)
    if callable(coefficient):
        points = np.einsum('qi,kic->kqc', bary, mesh.points[mesh.faces[faces]])
        scale = np.asarray(coefficient(points, faces))
    else:
        scale = np.asarray(coefficient)
        if scale.ndim == 1 and scale.shape[0] == mesh.n_faces:
            scale = scale[faces][:, None]
        elif scale.ndim != 0:
            raise ValueError(f'face coefficient of shape {scale.shape}: expected a number, ({mesh.n_faces},) or a '
                             'callable')
    weights = np.broadcast_to(scale, (len(faces), len(rule_weights))) * (area[:, None] * rule_weights[None, :])
    return local, _face_basis(dofs.order, bary), weights


def assemble_face_mass_3d(dofs, faces, coefficient=1.0, degree=None):
    """CSR (n_dofs, n_dofs) of int_faces h phi_i phi_j dA (Robin / convection h; face coefficient forms of the module
    header); rule degree default 2 x order."""
    local, basis, weights = _face_terms(dofs, faces, coefficient, 2 * dofs.order if degree is None else degree)
    rows, columns, blocks = [], [], []
    append_blocks(rows, columns, blocks, local, np.einsum('qi,qj,kq->kij', basis, basis, weights))
    return _to_csr(rows, columns, blocks, (dofs.n_dofs, dofs.n_dofs))


def assemble_face_load_3d(dofs, faces, source=1.0, degree=None) -> np.ndarray:
    """(n_dofs,) int_faces g phi_i dA (inward flux, Robin h T_ext; source as assemble_face_mass_3d's coefficient); rule
    degree default order + 2."""
    local, basis, weights = _face_terms(dofs, faces, source, dofs.order + 2 if degree is None else degree)
    values = np.einsum('qi,kq->ki', basis, weights)
    result = np.zeros(dofs.n_dofs, dtype=values.dtype)
    scatter_add(result, local, values)
    return result


def face_areas_3d(mesh, faces) -> np.ndarray:
    """(k,) areas [m^2] of mesh faces."""
    corners = mesh.points[mesh.faces[np.asarray(faces, dtype=np.int64)]]
    return 0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)


# ---- integrals ----
def element_basis_integrals_3d(dofs, elements, degree=None) -> tuple:
    """(measure (m,), int phi_n (m, n), int grad phi_n (m, n, 3)) of a Lagrange map on mesh elements (int (m,)) at the
    rule of `degree` (default 2 x order). Affine: |det J| x the reference rule sums (J^-1 is constant per element), so no
    per-point geometry: cheap enough per call that nothing is cached."""
    _require_lagrange(dofs, 'element_basis_integrals_3d', (0, 1, 2))
    rule = tetrahedron_rule(_rule_degree(2 * dofs.order, degree))
    basis, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    volume = np.zeros(0)
    gradients = np.zeros((0, basis.shape[1], 3))
    if len(elements):
        geometry = tetrahedron_geometry(dofs.mesh, elements)
        volume = np.abs(geometry.determinant)
        gradient_sums = np.einsum('q,qnk->nk', rule.weights, reference_gradients)   # (n, 3)
        gradients = volume[:, None, None] * piola_vectors_3d(gradient_sums, geometry)
    return volume * rule.weights.sum(), volume[:, None] * (rule.weights @ basis)[None, :], gradients


def element_integrals_3d(dofs, coefficients, elements=None, tensor=None, degree=None) -> dict:
    """Per mesh element (0 off `elements`) of a Lagrange field (coefficients (n_dofs,)): 'measure' (volume), 'value'
    (int u), 'gradient' (n_elements, 3) (int grad u) and, with tensor (n_elements, 3, 3), 'dissipation' (int grad u . C
    grad u); rule degree default 2 x order."""
    _require_lagrange(dofs, 'element_integrals_3d', (0, 1, 2))
    mesh = dofs.mesh
    count = mesh.n_elements
    coefficients = np.asarray(coefficients)
    result = {'measure': np.zeros(count), 'value': np.zeros(count), 'gradient': np.zeros((count, 3))}
    elements = _element_list(mesh, elements)
    if tensor is None:   # linear in the field: affine per-element integrals (no per-point geometry)
        measure, value_integrals, gradient_integrals = element_basis_integrals_3d(dofs, elements, degree)
        local = coefficients[dofs.element_dofs[elements]]   # (m, n)
        result['measure'][elements] = measure
        result['value'][elements] = np.einsum('mn,mn->m', value_integrals, local)
        result['gradient'][elements] = np.einsum('mnc,mn->mc', gradient_integrals, local)
        return result
    result['dissipation'] = np.zeros(count)
    rule = tetrahedron_rule(_rule_degree(2 * dofs.order, degree))
    basis, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    for chunk, geometry in element_chunks_3d(mesh, elements):
        weights = _volume_weights(geometry, rule)
        local = coefficients[dofs.element_dofs[chunk]]   # (m, n)
        point_gradients = np.einsum('mqni,mn->mqi', piola_vectors_3d(reference_gradients, geometry), local)
        result['measure'][chunk] = weights.sum(axis=1)
        result['value'][chunk] = np.einsum('qn,mn,mq->m', basis, local, weights)
        result['gradient'][chunk] = np.einsum('mqi,mq->mi', point_gradients, weights)
        result['dissipation'][chunk] = np.einsum('mqi,mij,mqj,mq->m', point_gradients, tensor[chunk],
                                                 point_gradients, weights)
    return result
