"""Linear elasticity on tetrahedron meshes (the fem/elasticity.py analogue, igor3d_plan section 7): vector Lagrange P1 /
P2 with interleaved dofs (3 d + component) on a full-mesh scalar DofMap3D, Voigt rows (xx, yy, zz, yz, xz, xy;
engineering shear), stiffness of a per-element 6 x 6 Voigt matrix (isotropic or anisotropic c_ij), the piezoelectric
coupling block to a scalar potential on the same map, consistent mass, stress / body / face-traction loads, element
strains / scalar gradients and their nodal average over an element subset. Prestress pieces (new_modules_plan K1): the
geometric stiffness K_sigma (PML stretch capable), initial-stress force, stress sampling of a fem.elasticity StressField
and the total-Lagrangian Green-Lagrange internal force / tangent (the K6 Newton of physics/mechanics/solver_fem3d; its
iterations on TangentAssembler3D: fixed pattern, cached rule gradients; device=True runs them on cupy). Affine
elements; host numpy, scipy / core.backend inside functions."""
# The matrices keep the full vector numbering (3 x the scalar map's dofs); rows of dofs no listed element touches stay
# empty, callers restrict to their free dofs (as fem.assembly3d). The elastic stiffness sums its blocks into the block
# pattern of fem.fixed_pattern.kept_pattern (scalar pattern x dense 3 x 3 blocks, kept per mesh and element set; slots
# searched per chunk, no kept slot map), the same base TangentAssembler3D groups; the geometric stiffness sums its
# scalar blocks x I3 into that pattern too while it is kept (the pencil K + K_sigma is then a same-pattern sum); the other
# matrices sum COO into CSR in batches (fem.assembly.CsrAccumulator) so the assembly peak stays bounded (P2: 900
# entries per element). Symmetric
# element products (B^T C B, the tangent) multiply only the column groups on and below the diagonal
# (_symmetric_operator_blocks, SYMMETRIC_COLUMN_GROUPS) and mirror the rest.
import numpy as np

from .assembly import CsrAccumulator, element_dof_mean
from .assembly3d import (CHUNK_ELEMENTS, _element_list, _rule_degree, _volume_weights, assemble_face_load_3d,
                         assemble_lagrange_load_3d, assemble_lagrange_mass_3d, element_chunks_3d)
from .elasticity import _fixed_field, as_stress_field, component_dofs, fixed_stress_at, stress_degree
from .elements3d import LOCAL_EDGES, REFERENCE_VERTICES_3D, lagrange_basis_3d, piola_vectors_3d, tetrahedron_geometry
from .fixed_pattern import kept_pattern, kept_pattern_if_any, scatter_add
from .quadrature3d import tetrahedron_rule
from .tangent_kernel3d import element_bytes, element_force_tangent_3d, kernel_fits

COMPONENTS = 3            # u_x, u_y, u_z per scalar dof
STRAIN_ROWS = 6           # Voigt xx, yy, zz, yz, xz, xy (engineering shear 2 strain_ij)
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))   # Voigt row -> tensor indices
STIFFNESS_NAMES = tuple(f'c{row}{column}' for row in range(1, 7) for column in range(row, 7))   # 21 upper entries
CENTROID_3D = np.full((1, 3), 0.25)
# elements per GEMM of the point evaluations (element_strains_3d, element_gradients_3d; P2 vector at 10 points: three
# float64 temporaries of ~24 MB) and per tetrahedron_geometry call of element_inverse_jacobians_3d
POINT_CHUNK_ELEMENTS = 2 ** 15
# host bytes up to which TangentAssembler3D keeps the rule gradients and weights (P2: ~6.7 kB per element); above it
# they are recomputed per call (the scatter map is always kept)
TANGENT_GRADIENT_CACHE_BYTES = 2 * 2 ** 30
# pool bytes up to which a device TangentAssembler3D keeps them on the GPU (above: host gradients uploaded per chunk
# and call); its other device caches (scatter slots, dofs, C, S0, pattern) are always kept
TANGENT_DEVICE_GRADIENT_CACHE_BYTES = 2 ** 30
# a device TangentAssembler3D sizes its chunks so one chunk's temporaries (tangent_chunk_element_bytes per element)
# stay within these pool bytes and within this share of the pool budget left at its construction
TANGENT_DEVICE_CHUNK_BYTES = 2 ** 30
TANGENT_DEVICE_CHUNK_SHARE = 0.25
# column groups of the symmetric element products: (groups + 1) / (2 groups) of the full matmul flops, groups launches
SYMMETRIC_COLUMN_GROUPS = 3
# elements from which assemble_elastic_stiffness_3d runs the numba element kernel although numba is not imported yet
# (import + cache load ~1 s against ~10 us per P2 element of batched numpy products saved; a guess)
ELEMENT_KERNEL_COLD_ELEMENTS = 50_000


def tangent_chunk_element_bytes(n_local, points, element_kernel=False) -> int:
    """Device bytes per element of one tangent chunk's temporaries. element_kernel (tangent_kernel3d): force, blocks and
    gathered displacements; else B, B_NL and their sum, C (B + B_NL), the weighted transposed copy and the
    column-group copies (q x 6 x 3 n each), strain / stress rows, blocks and lower products."""
    if element_kernel:
        return element_bytes(n_local)
    columns = COMPONENTS * int(n_local)
    return 8 * (int(points) * (5 * STRAIN_ROWS * columns + 21) + 2 * columns ** 2)


def device_tangent_chunk_elements(n_local, points, available=None, element_kernel=False) -> int:
    """Elements per device tangent chunk: its temporaries within TANGENT_DEVICE_CHUNK_BYTES and, with `available` pool
    bytes, within TANGENT_DEVICE_CHUNK_SHARE of them (at least one element)."""
    limit = TANGENT_DEVICE_CHUNK_BYTES
    if available is not None:
        limit = min(limit, int(TANGENT_DEVICE_CHUNK_SHARE * max(int(available), 0)))
    return max(1, limit // tangent_chunk_element_bytes(n_local, points, element_kernel))


def _available_pool_bytes():
    """Pool budget minus the live pool bytes (host counters), None without a budget (scripts)."""
    from ..core import gpu_memory
    budget = gpu_memory.budget_bytes()
    return None if budget is None else budget - gpu_memory.pool_used_bytes()


def local_dof_points_3d(order) -> np.ndarray:
    """(n_local, 3) reference points of the Lagrange local dofs: corners, then (P2) the midpoints of LOCAL_EDGES."""
    if order == 1:
        return REFERENCE_VERTICES_3D.copy()
    if order != 2:
        raise ValueError(f'local_dof_points_3d: order must be 1 or 2, got {order!r}')
    midpoints = [0.5 * (REFERENCE_VERTICES_3D[a] + REFERENCE_VERTICES_3D[b]) for a, b in LOCAL_EDGES]
    return np.concatenate([REFERENCE_VERTICES_3D, np.array(midpoints)])


def vector_dofs_3d(element_dofs) -> np.ndarray:
    """(m, 3 n) interleaved vector dofs 3 d + component of scalar element dofs (m, n)."""
    return component_dofs(element_dofs, COMPONENTS)


def isotropic_matrix_3d(lame_lambda, mu) -> np.ndarray:
    """(n, 6, 6) Voigt stiffness of per-element Lame coefficients (n,) [Pa]."""
    lame_lambda = np.asarray(lame_lambda, dtype=np.float64).reshape(-1)
    mu = np.broadcast_to(np.asarray(mu, dtype=np.float64), lame_lambda.shape)
    matrix = np.zeros((lame_lambda.size, STRAIN_ROWS, STRAIN_ROWS))
    matrix[:, :3, :3] = lame_lambda[:, None, None]
    for row in range(3):
        matrix[:, row, row] += 2.0 * mu
        matrix[:, row + 3, row + 3] = mu
    return matrix


def voigt_matrix_3d(coefficients, n_elements) -> np.ndarray:
    """(n_elements, 6, 6) symmetric Voigt stiffness from {c_IJ (STIFFNESS_NAMES): number or (n_elements,)}; missing
    entries are 0."""
    matrix = np.zeros((int(n_elements), STRAIN_ROWS, STRAIN_ROWS))
    for name, value in coefficients.items():
        if name not in STIFFNESS_NAMES:
            raise ValueError(f'voigt_matrix_3d: unknown stiffness entry {name!r}; expected c_IJ with 1 <= I <= J <= 6')
        row, column = int(name[1]) - 1, int(name[2]) - 1
        matrix[:, row, column] = matrix[:, column, row] = np.asarray(value, dtype=np.float64)
    return matrix


def _array_module(array):
    """numpy, or cupy for a cupy array (the device TangentAssembler3D; core.backend imported here only)."""
    if isinstance(array, np.ndarray):
        return np
    from ..core.backend import array_module_of
    return array_module_of(array)


def strain_operator_3d(gradients) -> np.ndarray:
    """B (m, q, 6, 3 n): Voigt strain of each vector basis function from physical gradients (m, q, n, 3) (numpy or
    cupy)."""
    m, q, n, _ = gradients.shape
    operator = _array_module(gradients).zeros((m, q, STRAIN_ROWS, COMPONENTS * n), dtype=gradients.dtype)
    for row, (first, second) in enumerate(VOIGT_PAIRS):
        operator[..., row, first::COMPONENTS] += gradients[..., second]
        if first != second:
            operator[..., row, second::COMPONENTS] += gradients[..., first]
    return operator


def _require_vector(dofs, what) -> None:
    if dofs.family != 'lagrange' or dofs.order not in (1, 2):
        raise ValueError(f'{what} needs a Lagrange P1 / P2 DofMap3D')


# ---- matrices ----
def _stretched(gradients, weights, stretch, geometry, rule, chunk) -> tuple:
    """(gradients / s per axis, weights x s_x s_y s_z) at the rule points of a PML stretch callable(points (m, q, 3),
    elements) -> complex (m, q, 3); unchanged without one."""
    if stretch is None:
        return gradients, weights
    stretches = np.asarray(stretch(geometry.points(rule.points), chunk), dtype=np.complex128)
    return gradients / stretches[:, :, None, :], weights * np.prod(stretches, axis=-1)


def _stiffness_element_kernel(n_elements, stretch, element_kernel) -> bool:
    """True when assemble_elastic_stiffness_3d runs the numba element kernel (tangent_kernel3d at u = 0, no S0): a real
    stiffness, numba available and loaded already or n_elements >= ELEMENT_KERNEL_COLD_ELEMENTS (element_kernel None),
    or forced (True; False: the batched numpy products)."""
    if stretch is not None or element_kernel is False:
        return False
    import sys
    if element_kernel is None and 'numba' not in sys.modules and n_elements < ELEMENT_KERNEL_COLD_ELEMENTS:
        return False
    from .tangent_kernel3d import ELEMENT_TANGENT_3D
    return ELEMENT_TANGENT_3D.host_kind == 'numba'


def assemble_elastic_stiffness_3d(dofs, material, elements=None, degree=None, stretch=None, element_kernel=None):
    """CSR (3 n, 3 n) of int B^T C B dV over `elements` (None / indices / bool mask); material (n_elements, 6, 6)
    Voigt stiffness per mesh element (isotropic_matrix_3d / voigt_matrix_3d); rule degree default 2 (order - 1), + 2
    with a PML stretch (_stretched: complex symmetric, d/dx_i -> d/dx_i / s_i, dV -> s_x s_y s_z dV). element_kernel:
    _stiffness_element_kernel (None: by numba and size)."""
    _require_vector(dofs, 'assemble_elastic_stiffness_3d')
    mesh = dofs.mesh
    material = _material_3d(material, mesh.n_elements)
    elements = _element_list(mesh, elements)
    n_vector = COMPONENTS * dofs.n_dofs
    if not len(elements):
        import scipy.sparse
        return scipy.sparse.csr_matrix((n_vector, n_vector))
    rule = tetrahedron_rule(_rule_degree(2 * dofs.order - 2 + (2 if stretch is not None else 0), degree))
    _, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    pattern = kept_pattern(mesh, n_vector, dofs.element_dofs[elements], block=COMPONENTS)
    data = np.zeros(pattern.nnz, dtype=np.float64 if stretch is None else np.complex128)
    kernel = _stiffness_element_kernel(len(elements), stretch, element_kernel)
    zero_displacement = np.zeros((CHUNK_ELEMENTS, dofs.n_local, COMPONENTS)) if kernel else None
    for chunk, geometry in element_chunks_3d(mesh, elements):
        gradients, weights = _stretched(piola_vectors_3d(reference_gradients, geometry), _volume_weights(geometry, rule),
                                        stretch, geometry, rule, chunk)
        if kernel:   # the tangent at u = 0 without S0 = B^T C B (prange over the chunk's elements)
            blocks = element_force_tangent_3d(gradients, weights, zero_displacement[:len(chunk)], material[chunk])[1]
        else:
            operator = strain_operator_3d(gradients)                                   # (m, q, 6, 3 n)
            stress_operator = np.matmul(material[chunk][:, None], operator)            # C B
            blocks = _symmetric_operator_blocks(operator, stress_operator, weights)
        # chunk rows of the element dofs the kept pattern was built from (its digest): no slot check needed
        pattern.add_elements(data, dofs.element_dofs[chunk], blocks, check=False)
    return pattern.full(data, copy_pattern=True)


def assemble_piezo_coupling_3d(dofs, piezo, elements=None, degree=None, stretch=None):
    """CSR (3 n, n) of int B^T e^T grad N dV over `elements` (u rows interleaved 3 d + component, potential columns on the
    same scalar map): the K_uphi block of the electric enthalpy grad(phi) . e S; piezo (n_elements, 3, 6) e_iJ [C/m^2]
    per mesh element (i = field x / y / z, J = Voigt); rule degree default 2 (order - 1), + 2 with a PML stretch."""
    _require_vector(dofs, 'assemble_piezo_coupling_3d')
    mesh = dofs.mesh
    piezo = np.asarray(piezo, dtype=np.float64)
    if piezo.shape != (mesh.n_elements, 3, STRAIN_ROWS):
        raise ValueError(f'piezo has shape {piezo.shape}, expected ({mesh.n_elements}, 3, 6)')
    elements = _element_list(mesh, elements)
    rule = tetrahedron_rule(_rule_degree(2 * dofs.order - 2 + (2 if stretch is not None else 0), degree))
    _, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    accumulator = CsrAccumulator((COMPONENTS * dofs.n_dofs, dofs.n_dofs))
    for chunk, geometry in element_chunks_3d(mesh, elements):
        gradients, weights = _stretched(piola_vectors_3d(reference_gradients, geometry), _volume_weights(geometry, rule),
                                        stretch, geometry, rule, chunk)                # (m, q, n, 3)
        operator = strain_operator_3d(gradients)                                       # (m, q, 6, 3 n)
        # e^T grad N: (m, q, 6, n)
        stress_per_gradient = np.matmul(piezo[chunk].swapaxes(1, 2)[:, None], gradients.swapaxes(-1, -2))
        local = dofs.element_dofs[chunk]
        accumulator.add(vector_dofs_3d(local), _operator_blocks(operator, stress_per_gradient, weights), local)
    return accumulator.result()


def assemble_elastic_mass_3d(dofs, density, elements=None):
    """CSR (3 n, 3 n) consistent mass int rho N_a N_b dV, the same block for every component (density per mesh element
    or a number)."""
    from scipy.sparse import identity, kron
    _require_vector(dofs, 'assemble_elastic_mass_3d')
    return kron(assemble_lagrange_mass_3d(dofs, density, elements=elements), identity(COMPONENTS), format='csr')


# ---- loads ----
def assemble_stress_load_3d(dofs, stress, elements=None) -> np.ndarray:
    """Vector (3 n,) of int B^T s dV for a stress s constant per element ((n_elements, 6) Voigt rows; the thermal load
    beta dT)."""
    _require_vector(dofs, 'assemble_stress_load_3d')
    mesh = dofs.mesh
    stress = np.asarray(stress, dtype=np.float64)
    if stress.shape != (mesh.n_elements, STRAIN_ROWS):
        raise ValueError(f'stress has shape {stress.shape}, expected ({mesh.n_elements}, 6)')
    elements = _element_list(mesh, elements)
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    if not stress[elements].any():   # no thermal / fixed stress on the elements (a frequency sweep without dT)
        return result
    rule = tetrahedron_rule(_rule_degree(dofs.order - 1, None))
    _, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    for chunk, geometry in element_chunks_3d(mesh, elements):
        operator = strain_operator_3d(piola_vectors_3d(reference_gradients, geometry))
        local = np.einsum('mqki,mk,mq->mi', operator, stress[chunk], _volume_weights(geometry, rule))
        scatter_add(result, vector_dofs_3d(dofs.element_dofs[chunk]), local)
    return result


def assemble_body_load_3d(dofs, forces, elements=None) -> np.ndarray:
    """Vector (3 n,) of int f N dV of a body force density [N/m^3]: forces = (f_x, f_y, f_z), each a number, per mesh
    element (n_elements,) or a callable of fem.assembly3d.assemble_lagrange_load_3d."""
    _require_vector(dofs, 'assemble_body_load_3d')
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    for component, force in enumerate(forces):
        if not callable(force) and not np.any(force):   # a zero component (number or per-element array) adds nothing
            continue
        result[component::COMPONENTS] = assemble_lagrange_load_3d(dofs, force, elements=elements)
    return result


def assemble_face_traction_3d(dofs, faces, traction) -> np.ndarray:
    """Vector (3 n,) of int_faces t N dA of a traction [Pa] on mesh faces (outer or interface): traction = (t_x, t_y,
    t_z), each a number, per mesh face or a face callable of fem.assembly3d.assemble_face_load_3d."""
    _require_vector(dofs, 'assemble_face_traction_3d')
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1)
    if not faces.size:
        return result
    for component, value in enumerate(traction):
        if np.ndim(value) == 0 and not callable(value) and float(value) == 0.0:
            continue
        result[component::COMPONENTS] = assemble_face_load_3d(dofs, faces, value)
    return result


# ---- strains and recovery ----
def _module_name(xp) -> str:
    return getattr(xp, '__name__', type(xp).__name__)


def element_inverse_jacobians_3d(dofs, xp=np):
    """(n_elements, 3, 3) float64 J^-1 of every mesh element on module xp (numpy / cupy), cached on the dof map
    (DofMap3D.element_geometry; host copy read-only)."""
    key = ('inverse_jacobians', _module_name(xp))
    found = dofs.element_geometry.get(key)
    if found is None:
        if xp is np:
            mesh = dofs.mesh
            found = np.empty((mesh.n_elements, 3, 3))
            for start in range(0, mesh.n_elements, POINT_CHUNK_ELEMENTS):
                chunk = np.arange(start, min(start + POINT_CHUNK_ELEMENTS, mesh.n_elements))
                found[chunk] = tetrahedron_geometry(mesh, chunk).inverse
            found.flags.writeable = False
        else:
            found = xp.asarray(element_inverse_jacobians_3d(dofs))
        dofs.element_geometry[key] = found
    return found


def _element_dofs_on(dofs, xp):
    """dofs.element_dofs on module xp (a device copy cached on the dof map)."""
    if xp is np:
        return dofs.element_dofs
    key = ('element_dofs', _module_name(xp))
    found = dofs.element_geometry.get(key)
    if found is None:
        found = dofs.element_geometry[key] = xp.asarray(dofs.element_dofs)
    return found


def _point_derivative_chunks(dofs, field, reference_points, elements):
    """Yield (rows, values (c, q, k), derivatives (c, k, q, 3) = d f_k / d x_d) of a Lagrange field (n_dofs, k) (numpy
    or cupy) at reference points (q, 3) per chunk of the element list: one GEMM per chunk with the reference tables,
    then grad = J^-T grad_ref by components from the cached J^-1."""
    xp = _array_module(field)
    components = field.shape[1]
    basis, reference_gradients = lagrange_basis_3d(dofs.order, reference_points)    # (q, n), (q, n, 3)
    q, n = basis.shape
    basis_columns = xp.asarray(np.ascontiguousarray(basis.T))                          # (n, q)
    gradient_columns = xp.asarray(reference_gradients.transpose(1, 0, 2).reshape(n, 3 * q))   # (n, q 3)
    inverses = element_inverse_jacobians_3d(dofs, xp)
    element_dofs = _element_dofs_on(dofs, xp)
    element_index = elements if xp is np else xp.asarray(elements)
    for start in range(0, len(elements), POINT_CHUNK_ELEMENTS):
        rows = slice(start, start + POINT_CHUNK_ELEMENTS)
        chunk = element_index[rows]
        count = int(chunk.shape[0])
        local = field[element_dofs[chunk]].transpose(0, 2, 1).reshape(count * components, n)   # (c k, n)
        values = (local @ basis_columns).reshape(count, components, q).transpose(0, 2, 1)
        reference = (local @ gradient_columns).reshape(count, components, q, 3)        # d f_k / d xi_a
        inverse = inverses[chunk]                                                       # (c, 3, 3) [a, d]
        derivatives = reference[..., 0, None] * inverse[:, None, None, 0, :]
        for axis in (1, 2):
            derivatives += reference[..., axis, None] * inverse[:, None, None, axis, :]
        yield rows, values, derivatives


def element_strains_3d(dofs, displacement, reference_points, elements=None, green_lagrange=False) -> tuple:
    """(displacement (m, q, 3), Voigt strains (m, q, 6), engineering shear; Green-Lagrange with green_lagrange) of the
    listed elements (sorted; None = all) at reference points (q, 3) from the nodal displacement (n_dofs, 3), on the
    displacement's module (numpy or cupy)."""
    _require_vector(dofs, 'element_strains_3d')
    xp = _array_module(displacement)
    elements = _element_list(dofs.mesh, elements)
    displacement = xp.asarray(displacement, dtype=np.float64).reshape(dofs.n_dofs, COMPONENTS)
    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    m, q = len(elements), len(reference_points)
    values = xp.empty((m, q, COMPONENTS))
    strains = xp.empty((m, q, STRAIN_ROWS))
    for rows, chunk_values, derivative in _point_derivative_chunks(dofs, displacement, reference_points, elements):
        values[rows] = chunk_values
        if green_lagrange:
            strains[rows] = _voigt_of_gradient(derivative.transpose(0, 2, 1, 3), green_lagrange=True)
            continue
        for row, (first, second) in enumerate(VOIGT_PAIRS):   # derivative[e, c, q, d] = d u_c / d x_d
            strains[rows, :, row] = (derivative[:, first, :, first] if first == second
                                     else derivative[:, first, :, second] + derivative[:, second, :, first])
    return values, strains


def element_gradients_3d(dofs, coefficients, reference_points, elements=None) -> tuple:
    """(values (m, q), gradients (m, q, 3)) of a scalar Lagrange field (coefficients (n_dofs,)) on the listed elements
    (sorted; None = all) at reference points (q, 3): the potential and -E of a piezoelectric solve."""
    _require_vector(dofs, 'element_gradients_3d')
    elements = _element_list(dofs.mesh, elements)
    coefficients = np.asarray(coefficients, dtype=np.float64).reshape(dofs.n_dofs, 1)
    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    m, q = len(elements), len(reference_points)
    values = np.empty((m, q))
    gradients = np.empty((m, q, 3))
    for rows, chunk_values, derivative in _point_derivative_chunks(dofs, coefficients, reference_points, elements):
        values[rows] = chunk_values[..., 0]
        gradients[rows] = derivative[:, 0]
    return values, gradients


def nodal_average_3d(dofs, local_values, elements=None) -> np.ndarray:
    """(n_dofs, ...) mean over the listed elements (sorted; None = all) sharing each dof of their values at the local
    dof points (m, n_local, ...): the smoothed field of strain / stress; dofs no listed element touches read 0."""
    return element_dof_mean(dofs.n_dofs, dofs.element_dofs[_element_list(dofs.mesh, elements)], local_values)


# ---- prestress (new_modules_plan K1): geometric stiffness, initial-stress force, stress sampling ----
def voigt_tensor_3d(voigt) -> np.ndarray:
    """(..., 3, 3) symmetric tensor of Voigt stress rows (..., 6) (xx, yy, zz, yz, xz, xy); numpy or cupy."""
    xp = _array_module(voigt)
    voigt = xp.asarray(voigt)
    tensor = xp.empty(voigt.shape[:-1] + (3, 3), dtype=voigt.dtype)
    for row, (first, second) in enumerate(VOIGT_PAIRS):
        tensor[..., first, second] = tensor[..., second, first] = voigt[..., row]
    return tensor


def _material_3d(material, n_elements) -> np.ndarray:
    material = np.asarray(material, dtype=np.float64)
    if material.shape != (n_elements, STRAIN_ROWS, STRAIN_ROWS):
        raise ValueError(f'material has shape {material.shape}, expected ({n_elements}, 6, 6)')
    return material


def _voigt_of_gradient(gradient, green_lagrange=False) -> np.ndarray:
    """(m, q, 6) engineering Voigt strain of displacement gradients H (m, q, 3, 3), H[c, d] = d u_c / d x_d: linear,
    or Green-Lagrange (H + H^T + H^T H) / 2 (numpy or cupy)."""
    xp = _array_module(gradient)
    product = xp.einsum('mqci,mqcj->mqij', gradient, gradient) if green_lagrange else None
    strains = xp.empty(gradient.shape[:2] + (STRAIN_ROWS,), dtype=gradient.dtype)
    for row, (first, second) in enumerate(VOIGT_PAIRS):
        if first == second:
            strains[..., row] = gradient[..., first, first]
            if green_lagrange:
                strains[..., row] += 0.5 * product[..., first, first]
        else:
            strains[..., row] = gradient[..., first, second] + gradient[..., second, first]
            if green_lagrange:
                strains[..., row] += product[..., first, second]
    return strains


def _stress_at_3d(field, dofs, chunk, points, gradients) -> np.ndarray:
    """(m, q, 6) stress of a StressField at physical points (m, q, 3) of mesh elements chunk with the physical basis
    gradients (m, q, n, 3) there (the displacement part from the unstretched gradients: a real stress)."""
    mesh = dofs.mesh
    stress = fixed_stress_at(field, points, chunk, mesh.n_elements, STRAIN_ROWS)
    if field.displacement is not None:
        displacement = np.asarray(field.displacement, dtype=np.float64).reshape(dofs.n_dofs, COMPONENTS)
        gradient = np.einsum('mqnd,mnc->mqcd', gradients, displacement[dofs.element_dofs[chunk]])
        material = _material_3d(field.material, mesh.n_elements)[chunk]
        stress = stress + np.einsum('mkl,mql->mqk', material, _voigt_of_gradient(gradient))
    return stress


def _operator_blocks(operator, product, weights) -> np.ndarray:
    """(m, r, c) sum over points and rows k of w B_ki P_kj: operator B (m, q, k, r), product P (m, q, k, c), weights (m,
    q) (one batched matmul over (point, row); numpy or cupy)."""
    xp = _array_module(operator)
    m = operator.shape[0]
    weighted = (operator * weights[:, :, None, None]).reshape(m, -1, operator.shape[-1])
    return xp.matmul(weighted.transpose(0, 2, 1), product.reshape(m, -1, product.shape[-1]))


def _symmetric_operator_blocks(operator, product, weights, column_groups=None) -> np.ndarray:
    """(m, c, c) _operator_blocks of a symmetric product (P = C B, C symmetric): per group of columns only the rows
    from that group on are multiplied, the rest mirrored (SYMMETRIC_COLUMN_GROUPS; 1 = the full product)."""
    xp = _array_module(operator)
    groups = SYMMETRIC_COLUMN_GROUPS if column_groups is None else int(column_groups)
    m, columns = operator.shape[0], operator.shape[-1]
    weighted = (operator * weights[:, :, None, None]).reshape(m, -1, columns)
    weighted = xp.ascontiguousarray(weighted.transpose(0, 2, 1))                        # (m, c, q k)
    product = product.reshape(m, -1, columns)                                            # (m, q k, c)
    blocks = xp.empty((m, columns, columns), dtype=xp.result_type(weighted, product))
    bounds = [round(columns * group / groups) for group in range(groups + 1)]
    for first, last in zip(bounds[:-1], bounds[1:]):
        lower = xp.matmul(weighted[:, first:], product[:, :, first:last])               # rows first.., columns of the group
        blocks[:, first:, first:last] = lower
        blocks[:, first:last, last:] = lower[:, last - first:].transpose(0, 2, 1)        # mirrored above the group
    return blocks


def _scalar_geometric_blocks(gradients, stress, weights) -> np.ndarray:
    """(m, n, n) blocks of int grad N_a . sigma grad N_b w of the stress (m, q, 6) at the rule points (numpy or cupy)."""
    xp = _array_module(gradients)
    transposed = gradients.swapaxes(-1, -2)                                   # (m, q, 3, n)
    return _operator_blocks(xp.ascontiguousarray(transposed), xp.matmul(voigt_tensor_3d(stress), transposed), weights)


def assemble_geometric_stiffness_3d(dofs, stress, elements=None, degree=None, stretch=None):
    """CSR (3 n, 3 n) K_sigma = (int grad N_a . sigma grad N_b dV) x I3 over `elements`; stress a fem.elasticity
    StressField or its prescribed forms; PML stretch as assemble_elastic_stiffness_3d (complex symmetric, rule + 2).
    On the kept pattern of a K of the same elements (assemble_elastic_stiffness_3d before) K_sigma takes K's pattern
    (explicit zeros off the component diagonal), so K + K_sigma (pattern_sum) sums values only."""
    from scipy.sparse import identity, kron
    _require_vector(dofs, 'assemble_geometric_stiffness_3d')
    field = as_stress_field(stress)
    mesh = dofs.mesh
    elements = _element_list(mesh, elements)
    default = 2 * dofs.order - 2 + stress_degree(field, dofs.order) + (2 if stretch is not None else 0)
    rule = tetrahedron_rule(_rule_degree(default, degree))
    _, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    pattern = (kept_pattern_if_any(mesh, COMPONENTS * dofs.n_dofs, dofs.element_dofs[elements], block=COMPONENTS)
               if len(elements) else None)
    if pattern is not None:
        data = np.zeros(pattern.nnz, dtype=np.float64 if stretch is None else np.complex128)
    accumulator = CsrAccumulator((dofs.n_dofs, dofs.n_dofs))
    for chunk, geometry in element_chunks_3d(mesh, elements):
        physical = piola_vectors_3d(reference_gradients, geometry)
        stress_values = _stress_at_3d(field, dofs, chunk, geometry.points(rule.points), physical)
        gradients, weights = _stretched(physical, _volume_weights(geometry, rule), stretch, geometry, rule, chunk)
        blocks = _scalar_geometric_blocks(gradients, stress_values, weights)
        if pattern is None:
            accumulator.add(np.asarray(dofs.element_dofs[chunk], dtype=np.int64), blocks)
        else:   # element dofs of the pattern's own build: no slot check needed
            pattern.add_elements(data, dofs.element_dofs[chunk], _component_identity_blocks(blocks), check=False)
    if pattern is not None:
        return pattern.full(data, copy_pattern=True)
    return kron(accumulator.result(), identity(COMPONENTS), format='csr')


def _component_identity_blocks(blocks) -> np.ndarray:
    """(m, 3 r, 3 r) blocks x I3 on the interleaved vector dofs (3 d + component) of scalar blocks (m, r, r)."""
    m, size, _ = blocks.shape
    expanded = np.zeros((m, size, COMPONENTS, size, COMPONENTS), dtype=blocks.dtype)
    for component in range(COMPONENTS):
        expanded[:, :, component, :, component] = blocks
    return expanded.reshape(m, COMPONENTS * size, COMPONENTS * size)


def initial_stress_force_3d(dofs, stress, elements=None, degree=None) -> np.ndarray:
    """Vector (3 n,) of -int B^T sigma0 dV over `elements`: the initial-stress load (a stationary K u = f + this relaxes
    it, total stress sigma0 + C eps); stress forms as assemble_geometric_stiffness_3d."""
    _require_vector(dofs, 'initial_stress_force_3d')
    field = as_stress_field(stress)
    mesh = dofs.mesh
    elements = _element_list(mesh, elements)
    rule = tetrahedron_rule(_rule_degree(dofs.order - 1 + stress_degree(field, dofs.order), degree))
    _, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    for chunk, geometry in element_chunks_3d(mesh, elements):
        gradients = piola_vectors_3d(reference_gradients, geometry)
        stress_values = _stress_at_3d(field, dofs, chunk, geometry.points(rule.points), gradients)
        local = np.einsum('mqki,mqk,mq->mi', strain_operator_3d(gradients), stress_values,
                          _volume_weights(geometry, rule))
        if np.iscomplexobj(local) and not np.iscomplexobj(result):
            result = result.astype(np.complex128)
        scatter_add(result, vector_dofs_3d(dofs.element_dofs[chunk]), -local)
    return result


def sample_stress_3d(dofs, stress, reference_points, elements=None) -> np.ndarray:
    """(m, q, 6) stress of a StressField (or its prescribed forms) on the listed elements (sorted; None = all) at
    reference points (q, 3): sigma0 + C eps(u) - beta dT, the total stress output of a prestressed solve."""
    _require_vector(dofs, 'sample_stress_3d')
    field = as_stress_field(stress)
    mesh = dofs.mesh
    elements = _element_list(mesh, elements)
    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 3)
    _, reference_gradients = lagrange_basis_3d(dofs.order, reference_points)
    parts = []
    for chunk, geometry in element_chunks_3d(mesh, elements):
        parts.append(_stress_at_3d(field, dofs, chunk, geometry.points(reference_points),
                                   piola_vectors_3d(reference_gradients, geometry)))
    return np.concatenate(parts) if parts else np.zeros((0, len(reference_points), STRAIN_ROWS))


# ---- Green-Lagrange pieces of the total-Lagrangian form (K6: solver_fem3d's geometrically nonlinear Newton) ----
def nonlinear_strain_operator_3d(gradients, displacement_gradient) -> np.ndarray:
    """B_NL (m, q, 6, 3 n) with delta E = (B + B_NL) delta u at the displacement gradient H (m, q, 3, 3) (H[c, d] = d u_c
    / d x_d) from physical gradients (m, q, n, 3); engineering shear rows (numpy or cupy)."""
    m, q, n, _ = gradients.shape
    gradient = displacement_gradient
    operator = _array_module(gradients).zeros((m, q, STRAIN_ROWS, n, COMPONENTS),
                                              dtype=np.result_type(gradients.dtype, gradient.dtype))
    for row, (first, second) in enumerate(VOIGT_PAIRS):
        operator[..., row, :, :] = gradients[..., :, second, None] * gradient[..., None, :, first]
        if first != second:
            operator[..., row, :, :] += gradients[..., :, first, None] * gradient[..., None, :, second]
    return operator.reshape(m, q, STRAIN_ROWS, COMPONENTS * n)


def green_lagrange_strains_3d(dofs, displacement, reference_points, elements=None) -> np.ndarray:
    """(m, q, 6) Green-Lagrange strain (H + H^T + H^T H) / 2 (engineering shear 2 E_ij) of the listed elements (sorted;
    None = all) at reference points (q, 3) from the nodal displacement (n_dofs, 3) (element_strains_3d's evaluation)."""
    return element_strains_3d(dofs, displacement, reference_points, elements=elements, green_lagrange=True)[1]


def _green_lagrange_terms_3d(gradients, local_displacement, material_chunk, fixed_stress):
    """(B + B_NL (m, q, 6, 3 n), S (m, q, 6)) at the rule points from physical gradients (m, q, n, 3), element
    displacements (m, n, 3), C (m, 6, 6) and S0 (m, q, 6) or None: S = C E(u) + S0 (numpy or cupy, one module)."""
    xp = _array_module(gradients)
    gradient = xp.einsum('mqnd,mnc->mqcd', gradients, local_displacement)
    stress = xp.einsum('mkl,mql->mqk', material_chunk, _voigt_of_gradient(gradient, green_lagrange=True))
    if fixed_stress is not None:
        stress = stress + fixed_stress
    return strain_operator_3d(gradients) + nonlinear_strain_operator_3d(gradients, gradient), stress


def _internal_force_blocks_3d(operator, stress, weights) -> np.ndarray:
    """(m, 3 n) element internal forces int (B + B_NL)^T S dV (numpy or cupy)."""
    return _array_module(operator).einsum('mqki,mqk,mq->mi', operator, stress, weights)


def _tangent_blocks_3d(gradients, weights, operator, stress, material_chunk, column_groups=None) -> np.ndarray:
    """(m, 3 n, 3 n) element tangents int (B + B_NL)^T C (B + B_NL) dV + K_sigma(S): the symmetric product of
    _symmetric_operator_blocks, the geometric term added per component through a strided view (numpy or cupy)."""
    xp = _array_module(operator)
    stress_operator = xp.matmul(material_chunk[:, None], operator)                    # C (B + B_NL)
    blocks = _symmetric_operator_blocks(operator, stress_operator, weights, column_groups)
    del stress_operator
    scalar = _scalar_geometric_blocks(gradients, stress, weights)                     # (m, n, n)
    m, n = scalar.shape[:2]
    interleaved = blocks.reshape(m, n, COMPONENTS, n, COMPONENTS)                     # a view: rows / columns 3 a + c
    for component in range(COMPONENTS):
        interleaved[:, :, component, :, component] += scalar
    return blocks


def _green_lagrange_chunks_3d(dofs, displacement, material, elements, degree, initial_stress):
    """Yield (chunk, gradients, weights, B + B_NL (m, q, 6, 3 n), S (m, q, 6), C (m, 6, 6)) on the total-Lagrangian rule
    (default 4 (order - 1)): S = C E(u) + S0."""
    _require_vector(dofs, 'the Green-Lagrange terms')
    mesh = dofs.mesh
    material = _material_3d(material, mesh.n_elements)
    fixed = _fixed_field(initial_stress)
    displacement = np.asarray(displacement, dtype=np.float64).reshape(dofs.n_dofs, COMPONENTS)
    elements = _element_list(mesh, elements)
    rule = tetrahedron_rule(_rule_degree(4 * (dofs.order - 1), degree))
    _, reference_gradients = lagrange_basis_3d(dofs.order, rule.points)
    for chunk, geometry in element_chunks_3d(mesh, elements):
        gradients = piola_vectors_3d(reference_gradients, geometry)
        fixed_stress = (None if fixed is None else
                        fixed_stress_at(fixed, geometry.points(rule.points), chunk, mesh.n_elements, STRAIN_ROWS))
        operator, stress = _green_lagrange_terms_3d(gradients, displacement[dofs.element_dofs[chunk]], material[chunk],
                                                    fixed_stress)
        yield chunk, gradients, _volume_weights(geometry, rule), operator, stress, material[chunk]


def internal_force_3d(dofs, displacement, material, elements=None, degree=None, initial_stress=None) -> np.ndarray:
    """Vector (3 n,) total-Lagrangian internal force int (B + B_NL(u))^T S dV, S = C E(u) + S0 (second Piola-Kirchhoff;
    S0 prescribed / thermal forms); = K u - initial_stress_force_3d(S0) to first order in u."""
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    for chunk, _, weights, operator, stress, _ in _green_lagrange_chunks_3d(dofs, displacement, material, elements,
                                                                            degree, initial_stress):
        local = _internal_force_blocks_3d(operator, stress, weights)
        scatter_add(result, vector_dofs_3d(dofs.element_dofs[chunk]), local)
    return result


def assemble_tangent_stiffness_3d(dofs, displacement, material, elements=None, degree=None, initial_stress=None):
    """CSR (3 n, 3 n) d internal_force_3d / d u = int (B + B_NL)^T C (B + B_NL) dV + K_sigma(S) on the same rule
    (symmetric); at u = 0 the elastic stiffness + assemble_geometric_stiffness_3d(S0)."""
    return internal_force_and_tangent_3d(dofs, displacement, material, elements, degree, initial_stress)[1]


def internal_force_and_tangent_3d(dofs, displacement, material, elements=None, degree=None, initial_stress=None):
    """(internal_force_3d vector, assemble_tangent_stiffness_3d CSR) of one displacement in one pass over the elements
    (a Newton iteration needs both). One-off COO assembly of the full (unmirrored) element products: the self-test
    reference of TangentAssembler3D; repeated Newton iterations use TangentAssembler3D."""
    force = np.zeros(COMPONENTS * dofs.n_dofs)
    accumulator = CsrAccumulator((COMPONENTS * dofs.n_dofs, COMPONENTS * dofs.n_dofs))
    for chunk, gradients, weights, operator, stress, material_chunk in _green_lagrange_chunks_3d(
            dofs, displacement, material, elements, degree, initial_stress):
        local_dofs = vector_dofs_3d(dofs.element_dofs[chunk])
        scatter_add(force, local_dofs, _internal_force_blocks_3d(operator, stress, weights))
        accumulator.add(local_dofs, _tangent_blocks_3d(gradients, weights, operator, stress, material_chunk,
                                                       column_groups=1))
    return force, accumulator.result()


class TangentAssembler3D:
    """Total-Lagrangian internal force and tangent of one solid assembly on a fixed CSR pattern
    (fixed_pattern.kept_pattern: per mesh, element set, chunks and free mask): rule gradients / weights cached per chunk
    (while within cache_bytes), per call only values; the full and free-free reduced CSRs share the cached index arrays
    (pattern constant by construction). device=True (plan G5): the same formulas on cupy (gradients, weights, C, dofs
    and S0 on the GPU, blocks summed by fixed_pattern.DevicePatternCsr) in chunks sized from the pool budget
    (device_tangent_chunk_elements), the force returned on the host, tangent data kept on the device. element_kernel
    (default): per-element force / tangent by fem.tangent_kernel3d (CUDA block per element, numba / numpy twins);
    False, or a device rule too large for the kernel: the batched formulas."""

    def __init__(self, dofs, material, elements=None, free=None, degree=None, cache_bytes=None, device=False,
                 element_kernel=True):
        _require_vector(dofs, 'TangentAssembler3D')
        mesh = dofs.mesh
        self.dofs = dofs
        self.device = bool(device)
        if cache_bytes is None:
            cache_bytes = TANGENT_DEVICE_GRADIENT_CACHE_BYTES if self.device else TANGENT_GRADIENT_CACHE_BYTES
        self.material = _material_3d(material, mesh.n_elements)
        elements = _element_list(mesh, elements)
        self.rule = tetrahedron_rule(_rule_degree(4 * (dofs.order - 1), degree))
        _, self._reference_gradients = lagrange_basis_3d(dofs.order, self.rule.points)
        self.n_vector = COMPONENTS * dofs.n_dofs
        points = len(self.rule.weights)
        self.element_kernel = bool(element_kernel) and (not self.device or kernel_fits(dofs.n_local, points))
        gradient_bytes = 8 * len(elements) * points * (3 * dofs.n_local + 1)
        self.gradients_cached = gradient_bytes <= cache_bytes
        self.chunk_elements = CHUNK_ELEMENTS   # host; device: from the pool budget (T3 a)
        if self.device:
            available = _available_pool_bytes()
            if available is not None:   # less the kept gradients and int32 scatter slots allocated below
                available -= ((gradient_bytes if self.gradients_cached else 0)
                              + 4 * len(elements) * (COMPONENTS * dofs.n_local) ** 2)
            self.chunk_elements = device_tangent_chunk_elements(dofs.n_local, points, available, self.element_kernel)
        step = self.chunk_elements
        self.chunks = [elements[start:start + step] for start in range(0, len(elements), step)]
        self._vector_dofs = [vector_dofs_3d(dofs.element_dofs[chunk]) for chunk in self.chunks]
        # the block pattern of the element set, kept per mesh with these groups and free mask (later sweep values)
        self.pattern = kept_pattern(mesh, self.n_vector, dofs.element_dofs[elements], block=COMPONENTS,
                                    groups=[dofs.element_dofs[chunk] for chunk in self.chunks], free=free)
        self._fixed_cache = None   # device: (initial_stress object, {chunk index: device S0}) of the latest S0
        self.device_pattern = None
        self.device_bytes = 0      # pool bytes of the device caches (device=True)
        if self.device:
            from ..core.backend import cp
            self.xp = cp
            self.device_pattern = self.pattern.to_device()
            index_dtype = np.int32 if self.n_vector < 2 ** 31 else np.int64
            self._element_dofs = [cp.asarray(np.asarray(dofs.element_dofs[chunk]).astype(index_dtype))
                                  for chunk in self.chunks]
            self._scatter_dofs = [cp.asarray(local.astype(index_dtype).ravel()) for local in self._vector_dofs]
            self._materials = [cp.asarray(self.material[chunk]) for chunk in self.chunks]
            self.device_bytes = (self.device_pattern.device_bytes
                                 + sum(int(array.nbytes) for array in self._element_dofs + self._scatter_dofs
                                       + self._materials)
                                 + (gradient_bytes if self.gradients_cached else 0))
        else:
            self.xp = np
        self._geometry = [self._chunk_geometry(chunk) for chunk in self.chunks] if self.gradients_cached else None
        self.cached_bytes = (self.pattern.index_bytes + sum(local.nbytes for local in self._vector_dofs)
                             + (gradient_bytes if self.gradients_cached and not self.device else 0))

    def _chunk_geometry(self, chunk):
        """(physical gradients (m, q, n, 3), weights (m, q)) of mesh elements chunk on the rule (device: uploaded)."""
        geometry = tetrahedron_geometry(self.dofs.mesh, chunk)
        gradients, weights = piola_vectors_3d(self._reference_gradients, geometry), _volume_weights(geometry, self.rule)
        if self.device:
            return self.xp.asarray(gradients), self.xp.asarray(weights)
        return gradients, weights

    def _host_fixed_stress(self, chunk, fixed):
        """Host S0 (m, q, 6) of mesh elements chunk at the rule points."""
        mesh = self.dofs.mesh
        shape = (len(chunk), len(self.rule.weights), 3)   # points are read only by a callable S0
        points = (tetrahedron_geometry(mesh, chunk).points(self.rule.points) if callable(fixed.prescribed)
                  else np.broadcast_to(0.0, shape))
        return fixed_stress_at(fixed, points, chunk, mesh.n_elements, STRAIN_ROWS)

    def _chunk_fixed_stress(self, index, fixed, initial_stress):
        """S0 of chunk index on the assembler's module; the device keeps it per initial_stress object (a Newton solve
        passes one object: a new S0 value needs a new object)."""
        chunk = self.chunks[index]
        if not self.device:
            return self._host_fixed_stress(chunk, fixed)
        if self._fixed_cache is None or self._fixed_cache[0] is not initial_stress:
            self._fixed_cache = (initial_stress, {})
        stored = self._fixed_cache[1]
        if index not in stored:
            stored[index] = self.xp.asarray(np.ascontiguousarray(self._host_fixed_stress(chunk, fixed),
                                                                 dtype=np.float64))
        return stored[index]

    def _chunk_inputs(self, index, displacement, fixed, initial_stress):
        """(gradients, weights, element displacements (m, n, 3), C, S0 or None) of chunk index (assembler's module)."""
        chunk = self.chunks[index]
        gradients, weights = self._geometry[index] if self.gradients_cached else self._chunk_geometry(chunk)
        fixed_stress = None if fixed is None else self._chunk_fixed_stress(index, fixed, initial_stress)
        if self.device:
            material_chunk, local = self._materials[index], displacement[self._element_dofs[index]]
        else:
            material_chunk, local = self.material[chunk], displacement[self.dofs.element_dofs[chunk]]
        return gradients, weights, local, material_chunk, fixed_stress

    def _chunk_blocks(self, index, displacement, fixed, initial_stress, want_tangent=True):
        """(element forces (m, 3 n), element tangents (m, 3 n, 3 n) or None) of chunk index at the displacement."""
        gradients, weights, local, material_chunk, fixed_stress = self._chunk_inputs(index, displacement, fixed,
                                                                                     initial_stress)
        if self.element_kernel:
            return element_force_tangent_3d(gradients, weights, local, material_chunk, fixed_stress, want_tangent)
        operator, stress = _green_lagrange_terms_3d(gradients, local, material_chunk, fixed_stress)
        forces = _internal_force_blocks_3d(operator, stress, weights)
        if not want_tangent:
            return forces, None
        return forces, _tangent_blocks_3d(gradients, weights, operator, stress, material_chunk)

    def _displacement(self, displacement):
        """(n_dofs, 3) float64 displacement on the assembler's module (device: one upload)."""
        return self.xp.asarray(displacement, dtype=np.float64).reshape(self.dofs.n_dofs, COMPONENTS)

    def _add_force(self, force, index, local) -> None:
        """force += element forces local (m, 3 n) of chunk index (host bincount / device scatter_add)."""
        if self.device:
            import cupyx
            cupyx.scatter_add(force, self._scatter_dofs[index], local.ravel())
        else:
            scatter_add(force, self._vector_dofs[index], local)

    def _host(self, array):
        return self.xp.asnumpy(array) if self.device else array

    def force(self, displacement, initial_stress=None) -> np.ndarray:
        """Host internal_force_3d (3 n,) of the assembler's elements at the displacement (initial_stress: S0)."""
        displacement = self._displacement(displacement)
        fixed = _fixed_field(initial_stress)
        force = self.xp.zeros(self.n_vector)
        for index in range(len(self.chunks)):
            self._add_force(force, index, self._chunk_blocks(index, displacement, fixed, initial_stress,
                                                             want_tangent=False)[0])
        return self._host(force)

    def force_and_tangent(self, displacement, initial_stress=None) -> tuple:
        """(host internal force (3 n,), tangent data (nnz,): host, or cupy with device) in one pass; CSRs by `full` /
        `reduced`."""
        displacement = self._displacement(displacement)
        fixed = _fixed_field(initial_stress)
        force = self.xp.zeros(self.n_vector)
        pattern = self.device_pattern if self.device else self.pattern
        data = pattern.zeros()
        for index in range(len(self.chunks)):
            forces, blocks = self._chunk_blocks(index, displacement, fixed, initial_stress)
            self._add_force(force, index, forces)
            pattern.add(data, index, blocks)
            del forces, blocks
        return self._host(force), data

    def _on_device(self, data) -> bool:
        return self.device and not isinstance(data, np.ndarray)

    def full(self, data, copy_pattern=False):
        """CSR (3 n, 3 n) of tangent data: host scipy (FixedPatternCsr.full), cupyx for device data."""
        if self._on_device(data):
            return self.device_pattern.full(data, copy_pattern=copy_pattern)
        return self.pattern.full(data, copy_pattern=copy_pattern)

    def reduced(self, data):
        """Canonical CSR of the free rows / columns of tangent data (the constructor's free mask): host scipy, cupyx for
        device data."""
        if self._on_device(data):
            return self.device_pattern.reduced(data)
        return self.pattern.reduced(data)

    def reduced_values(self, data):
        """Values of the free rows / columns of tangent data (same module) in the order of the host reduced pattern
        `pattern.reduced_indptr` / `reduced_indices`."""
        if self._on_device(data):
            return self.device_pattern.reduced_values(data)
        return data[self.pattern.reduced_take]

    def full_product(self, data, vector) -> np.ndarray:
        """Host (3 n,) full tangent of data times a host vector (device data: the product on the device)."""
        if self._on_device(data):
            return self.xp.asnumpy(self.full(data) @ self.xp.asarray(vector, dtype=np.float64))
        return self.full(data) @ vector

    def tangent(self, displacement, initial_stress=None):
        """Host assemble_tangent_stiffness_3d on the fixed pattern (own index copies: a pencil may be modified)."""
        data = self._host(self.force_and_tangent(displacement, initial_stress)[1])
        return self.pattern.full(data, copy_pattern=True)
