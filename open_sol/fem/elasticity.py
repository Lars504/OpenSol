"""Linear elasticity on triangle meshes (mesh plan 2.7, stage T5): vector Lagrange P1 / P2 with interleaved dofs
(2 d + component), planar Voigt rows (xx, yy, g = 2 strain_xy) or axisymmetric (r, z) rows (rr, zz, tt, g) with the
hoop strain u_r / r and whole-ring weights 2 pi r; stiffness, consistent mass, stress / body / edge loads and element
strains (nodal recovery: fem.assembly.nodal_average). Prestress pieces (new_modules_plan K1): StressField, geometric
stiffness K_sigma, initial-stress force, stress sampling and the total-Lagrangian Green-Lagrange internal force /
tangent (geometric nonlinearity, K6; Newton iterations on TangentAssembler, a fixed pattern). PML coordinate stretches (stretch= of the stiffness / mass / K_sigma, K5) and the
azimuthal order m >= 1 form (U_r, U_z, U_phi) of the axisymmetric solid (K7). Host numpy, scipy inside functions (via
fem.assembly)."""
# Stretch (the elastic PML, fem.pml): stretch(points (m, q, 2), elements) -> complex (m, q, 2) (s_x, s_y); gradients
# d/dx_i -> d/dx_i / s_i, weights x s_x s_y, axisymmetric also x radial_ratio(points) = r~ / r and the hoop N / r~;
# unconjugated (complex symmetric), rule + 2 (planar). Azimuthal order m >= 1: u_r = U_r cos(m phi), u_z = U_z cos(m
# phi), u_phi = U_phi sin(m phi); Voigt rows rr = U_r,r, zz = U_z,z, tt = (U_r + m U_phi) / r, g_rz = U_r,z + U_z,r (cos
# parts), g_zt = U_phi,z - m U_z / r, g_rt = -m U_r / r + U_phi,r - U_phi / r (sin parts); the phi integrals of cos^2 and
# sin^2 are both pi (AZIMUTHAL_RING), dofs 3 d + component.
import math
from dataclasses import dataclass

import numpy as np

from .assembly import (CHUNK_ELEMENTS, _element_groups, _tensor_blocks, _to_csr, append_blocks, assemble_edge_load,
                       assemble_load, assemble_mass)
from .elements import REFERENCE_VERTICES, element_geometry, lagrange_basis, physical_gradients
from .fixed_pattern import FixedPatternCsr, scatter_add
from .quadrature import TRIANGLE_DEGREES

COMPONENTS = 2                 # u_x (u_r), u_y (u_z) per scalar dof
RING = 2.0 * math.pi           # axisymmetric integrals are whole rings
AZIMUTHAL_COMPONENTS = 3       # (U_r, U_z, U_phi) per scalar dof of an azimuthal order m >= 1
AZIMUTHAL_RING = math.pi       # = core.metric.ring_angle(m >= 1): the phi integral of cos^2 (m phi) and sin^2 (m phi)
AZIMUTHAL_ROWS = 6             # Voigt rows rr, zz, tt, g_rz (cos parts), g_zt, g_rt (sin parts) of an order m >= 1
AXIS_TOLERANCE = 1e-12         # r <= this x the mesh extent: on the axis, the hoop strain is d u_r / d r there
AXISYMMETRIC_DEGREE = 6        # rule of the rational hoop integrands (N_a N_b / r)
_PLANAR_ROWS = (('c11', 'c12', 'c16'), ('c12', 'c22', 'c26'), ('c16', 'c26', 'c66'))
_AXISYMMETRIC_ROWS = (('c11', 'c12', 'c13', None), ('c12', 'c22', 'c23', None), ('c13', 'c23', 'c33', None),
                      (None, None, None, 'c66'))
CENTROID = np.array([[1.0 / 3.0, 1.0 / 3.0]])


def strain_rows(axisymmetric) -> int:
    """Voigt rows of the strain: 3 planar (xx, yy, g), 4 axisymmetric (rr, zz, tt, g)."""
    return 4 if axisymmetric else 3


def local_dof_points(order) -> np.ndarray:
    """(n_local, 2) reference points of the Lagrange local dofs: corners, then (P2) the midsides of local edges 0, 1, 2."""
    if order == 1:
        return REFERENCE_VERTICES.copy()
    if order != 2:
        raise ValueError(f'local_dof_points: order must be 1 or 2, got {order!r}')
    midsides = [0.5 * (REFERENCE_VERTICES[(k + 1) % 3] + REFERENCE_VERTICES[(k + 2) % 3]) for k in range(3)]
    return np.concatenate([REFERENCE_VERTICES, np.array(midsides)])


def vector_dofs(element_dofs) -> np.ndarray:
    """(m, 2 n) interleaved vector dofs 2 d + component of scalar element dofs (m, n)."""
    return component_dofs(element_dofs, COMPONENTS)


def material_matrix(coefficients, elements, axisymmetric) -> np.ndarray:
    """(m, k, k) Voigt stiffness of mesh elements from {c_ij: per mesh element array or number} (missing c16 / c26 =
    0; axisymmetric: orthotropic with the hoop c13 / c23 / c33)."""
    rows = _AXISYMMETRIC_ROWS if axisymmetric else _PLANAR_ROWS
    matrix = np.zeros((len(elements), len(rows), len(rows)))
    for a, row in enumerate(rows):
        for b, name in enumerate(row):
            value = None if name is None else coefficients.get(name)
            if value is not None:
                matrix[:, a, b] = np.asarray(value, dtype=np.float64)[elements] if np.ndim(value) else float(value)
    return matrix


def strain_operator(values, gradients, points, axisymmetric) -> np.ndarray:
    """B (m, q, k, 2 n): Voigt strain of each vector basis function (values (q, n), physical gradients (m, q, n, 2),
    points (m, q, 2) for the hoop row N / r; complex stretched gradients / radius give a complex B)."""
    m, q, n, _ = gradients.shape
    k = strain_rows(axisymmetric)
    operator = np.zeros((m, q, k, COMPONENTS * n), dtype=np.result_type(gradients, points))
    operator[..., 0, 0::2] = gradients[..., 0]
    operator[..., 1, 1::2] = gradients[..., 1]
    operator[..., k - 1, 0::2] = gradients[..., 1]
    operator[..., k - 1, 1::2] = gradients[..., 0]
    if axisymmetric:
        operator[..., 2, 0::2] = values[None, :, :] / points[..., 0:1]
    return operator


def _weights(geometry, rule, axisymmetric):
    """(m, q) w det J (x 2 pi r: whole rings)."""
    weights = rule.weights[None, :] * geometry.determinant
    return weights * (RING * geometry.points[..., 0]) if axisymmetric else weights


def _groups(dofs, axisymmetric, planar_degree):
    """Element groups of fem.assembly (non-empty chunks) with the elasticity rule degree."""
    return _element_groups(dofs, AXISYMMETRIC_DEGREE if axisymmetric else planar_degree)


def _stretched(geometry, gradients, weights, elements, stretch, radial_ratio, axisymmetric):
    """(gradients / s per axis, weights x s_x s_y (x r~ / r axisymmetric), points with the stretched hoop radius r~ in
    column 0) of a PML stretch at the rule points (module header); stretch None: unchanged."""
    if stretch is None:
        return gradients, weights, geometry.points
    stretches = np.asarray(stretch(geometry.points, elements))
    gradients = gradients / stretches[:, :, None, :]
    weights = weights * stretches[..., 0] * stretches[..., 1]
    points = geometry.points
    if axisymmetric and radial_ratio is not None:
        ratio = np.asarray(radial_ratio(geometry.points))
        weights = weights * ratio
        points = points.astype(np.complex128)
        points[..., 0] = points[..., 0] * ratio
    return gradients, weights, points


def _operator_at(dofs, positions, rule, axisymmetric, stretch=None, radial_ratio=None):
    """(geometry, B, weights w det J (x 2 pi r)) of the dof map's elements at positions on a rule (a PML stretch:
    stretched B and weights)."""
    geometry = element_geometry(dofs.mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
    basis, reference_gradients = lagrange_basis(dofs.order, rule.points)
    gradients = physical_gradients(reference_gradients, geometry.inverse)
    gradients, weights, points = _stretched(geometry, gradients, _weights(geometry, rule, axisymmetric),
                                            dofs.elements[positions], stretch, radial_ratio, axisymmetric)
    return geometry, strain_operator(basis, gradients, points, axisymmetric), weights


def _material_blocks(operator, material, weights) -> np.ndarray:
    """(m, r, r) sum over points of w B^T C B: operator B (m, q, k, r), material C (m, k, k), weights (m, q); C B and
    the sum over (point, row) as batched matmuls."""
    count = len(operator)
    stress_operator = np.matmul(material[:, None], operator)
    weighted = (operator * weights[:, :, None, None]).reshape(count, -1, operator.shape[-1])
    return weighted.transpose(0, 2, 1) @ stress_operator.reshape(count, -1, stress_operator.shape[-1])


def _basis_product_blocks(basis, weights) -> np.ndarray:
    """(m, n, n) sum over points of w N_a N_b: basis (q, n), weights (m, q) @ the (q, n n) basis products."""
    products = (basis[:, :, None] * basis[:, None, :]).reshape(len(basis), -1)
    return (weights @ products).reshape(len(weights), basis.shape[1], basis.shape[1])


def _stretch_degree(stretch) -> int:
    """Extra rule degree of a stretched integrand (the s profile is quadratic in the depth)."""
    return 0 if stretch is None else 2


# ---- matrices ----
def assemble_elastic_stiffness(dofs, coefficients, *, axisymmetric=False, stretch=None, radial_ratio=None):
    """CSR (2 n, 2 n) of int B^T C B dA (axisymmetric: dV = 2 pi r dA) on the dof map's elements; coefficients
    {c11, c12, c22, c66, c16, c26 (+ c13, c23, c33 hoop)} per mesh element; stretch / radial_ratio: the PML form
    (module header, complex symmetric)."""
    n_vector = COMPONENTS * dofs.n_dofs
    rows, cols, values = [], [], []
    for positions, rule in _groups(dofs, axisymmetric, max(2 * dofs.order - 2, 2) + _stretch_degree(stretch)):
        _, operator, weights = _operator_at(dofs, positions, rule, axisymmetric, stretch, radial_ratio)
        local = _material_blocks(operator, material_matrix(coefficients, dofs.elements[positions], axisymmetric),
                                 weights)
        _append_blocks(rows, cols, values, dofs, positions, local)
    return _to_csr(rows, cols, values, (n_vector, n_vector))


def _scalar_mass(dofs, density, axisymmetric, stretch, radial_ratio):
    """CSR int rho N_a N_b dA (x r axisymmetric, no ring angle); a PML stretch weights rho by s_x s_y (x r~ / r)."""
    if stretch is None:
        return assemble_mass(dofs, density, axisymmetric=axisymmetric)
    density = np.asarray(density, dtype=np.float64)

    def stretched_density(points, elements):
        values = density[elements][:, None] if density.ndim else density
        factor = np.prod(np.asarray(stretch(points, elements)), axis=-1)
        if axisymmetric and radial_ratio is not None:
            factor = factor * np.asarray(radial_ratio(points))
        return values * factor
    degree = min(2 * dofs.order + int(axisymmetric) + _stretch_degree(stretch), TRIANGLE_DEGREES[-1])
    return assemble_mass(dofs, stretched_density, degree=degree, axisymmetric=axisymmetric)


def assemble_elastic_mass(dofs, density, *, axisymmetric=False, stretch=None, radial_ratio=None):
    """CSR (2 n, 2 n) consistent mass int rho N_a N_b dV, the same block for both components (density per mesh
    element or a number); stretch / radial_ratio: the PML form (complex symmetric)."""
    from scipy.sparse import identity, kron

    scalar = _scalar_mass(dofs, density, axisymmetric, stretch, radial_ratio)
    if axisymmetric:
        scalar = scalar * RING
    return kron(scalar, identity(COMPONENTS), format='csr')


# ---- loads ----
def assemble_stress_load(dofs, stress, *, axisymmetric=False, weight=None) -> np.ndarray:
    """Vector (2 n,) of int B^T s w dV for a stress s constant per element ((n_elements, k) in the Voigt rows; the
    thermal load beta dT) times weight w(points (m, q, 2)) -> (m, q), at most linear (None = 1)."""
    stress = np.asarray(stress, dtype=np.float64)
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    if not stress[dofs.elements].any():   # no thermal / fixed stress on the map (a frequency sweep without dT)
        return result
    for positions, rule in _groups(dofs, axisymmetric, max(2 * dofs.order - 1, 2)):
        geometry, operator, weights = _operator_at(dofs, positions, rule, axisymmetric)
        if weight is not None:
            weights = weights * weight(geometry.points)
        local = np.einsum('mqki,mk,mq->mi', operator, stress[dofs.elements[positions]], weights)
        scatter_add(result, vector_dofs(dofs.element_dofs[positions]), local)
    return result


def assemble_body_load(dofs, force_x, force_y, *, axisymmetric=False) -> np.ndarray:
    """Vector (2 n,) of int f N dV of a body force density (per mesh element or numbers) [N/m^3]."""
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    scale = RING if axisymmetric else 1.0
    degree = min(2 * dofs.order + 2 + int(axisymmetric), AXISYMMETRIC_DEGREE)
    for component, force in enumerate((force_x, force_y)):
        if not callable(force) and not np.any(force):   # as assemble_body_load_3d: a zero component adds nothing
            continue
        result[component::COMPONENTS] = scale * assemble_load(dofs, force, degree=degree, axisymmetric=axisymmetric)
    return result


def assemble_edge_traction(dofs, edges, traction_x, traction_y, *, axisymmetric=False) -> np.ndarray:
    """Vector (2 n,) of int_edges t N ds (axisymmetric: x 2 pi r) of a traction [Pa] on mesh edges (numbers, per mesh
    edge or callables of fem.assembly.assemble_edge_load); edges without an element of the map add nothing."""
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    edges = np.asarray(edges, dtype=np.int64)
    if not len(edges):
        return result
    scale = RING if axisymmetric else 1.0
    for component, traction in enumerate((traction_x, traction_y)):
        result[component::COMPONENTS] = scale * assemble_edge_load(dofs, edges, traction, axisymmetric=axisymmetric)
    return result


# ---- strains and recovery ----
def element_strains(dofs, displacement, reference_points, *, axisymmetric=False) -> tuple:
    """(displacement (m, q, 2), Voigt strains (m, q, k)) of every element of the map at reference points (q, 2) from
    nodal displacement (n, 2); hoop strain u_r / r, on the axis d u_r / d r (its limit when u_r(0) = 0)."""
    displacement = np.asarray(displacement, dtype=np.float64).reshape(dofs.n_dofs, COMPONENTS)
    reference_points = np.asarray(reference_points, dtype=np.float64)
    basis, reference_gradients = lagrange_basis(dofs.order, reference_points)
    m, q = len(dofs.elements), len(reference_points)
    k = strain_rows(axisymmetric)
    values = np.empty((m, q, COMPONENTS))
    strains = np.empty((m, q, k))
    axis_radius = AXIS_TOLERANCE * float(np.abs(dofs.mesh.points).max())
    for start in range(0, m, CHUNK_ELEMENTS):
        chunk = slice(start, start + CHUNK_ELEMENTS)
        geometry = element_geometry(dofs.mesh, dofs.elements[chunk], reference_points, dofs.geometry_order)
        gradients = physical_gradients(reference_gradients, geometry.inverse)   # (c, q, n, 2)
        local = displacement[dofs.element_dofs[chunk]]                           # (c, n, 2)
        values[chunk] = np.einsum('qn,mnc->mqc', basis, local)
        derivative = np.einsum('mqnd,mnc->mqcd', gradients, local)               # d u_c / d x_d
        strains[chunk, :, 0] = derivative[..., 0, 0]
        strains[chunk, :, 1] = derivative[..., 1, 1]
        strains[chunk, :, k - 1] = derivative[..., 0, 1] + derivative[..., 1, 0]
        if axisymmetric:
            radius = geometry.points[..., 0]
            on_axis = radius <= axis_radius
            strains[chunk, :, 2] = np.where(on_axis, derivative[..., 0, 0],
                                            values[chunk, :, 0] / np.where(on_axis, 1.0, radius))
    return values, strains


# ---- prestress (new_modules_plan K1): stress fields, geometric stiffness, initial-stress force ----
@dataclass(frozen=True, eq=False)
class StressField:
    """Stress [Pa] in the Voigt rows of its module, evaluated by the assemblers at their own points: prescribed + C
    eps(displacement) - thermal (sigma = C eps - beta dT, as the solvers' outputs); a None part adds nothing."""
    prescribed: object = None     # (n_elements, k) per mesh element, or f(points (m, q, d), elements (m,)) -> (m, q, k)
    material: object = None       # needed with displacement: 2D {c_ij} of material_matrix, 3D (n_elements, 6, 6)
    displacement: object = None   # nodal (n_dofs, d) (or flat) on the assembling dof map, e.g. the static solve
    thermal: object = None        # (n_elements, k) beta dT per mesh element, subtracted

    def __post_init__(self):
        if self.displacement is not None and self.material is None:
            raise ValueError('StressField: a displacement needs its material')


def as_stress_field(stress) -> StressField:
    """StressField of a StressField, or of a per-element (n_elements, k) array / callable taken as the prescribed part."""
    if stress is None:
        raise ValueError('as_stress_field: no stress given')
    return stress if isinstance(stress, StressField) else StressField(prescribed=stress)


def stress_degree(field, order) -> int:
    """Polynomial degree of a StressField inside an element: order - 1 with a displacement, else 0 (a varying
    callable wants an explicit rule degree)."""
    return order - 1 if field.displacement is not None else 0


def _element_rows(values, elements, n_elements, rows, what) -> np.ndarray:
    """(m, 1, rows) per-element values of mesh elements (m,) from an (n_elements, rows) array."""
    values = np.asarray(values)
    if values.shape != (n_elements, rows):
        raise ValueError(f'{what} has shape {values.shape}, expected ({n_elements}, {rows})')
    return values[elements][:, None, :]


def fixed_stress_at(field, points, elements, n_elements, rows) -> np.ndarray:
    """(m, q, rows) prescribed - thermal part of a StressField at physical points (m, q, d) of mesh elements (m,)."""
    m, q = points.shape[:2]
    stress = np.zeros((m, q, rows))
    if field.prescribed is not None:
        if callable(field.prescribed):
            values = np.asarray(field.prescribed(points, elements))
            if values.shape != (m, q, rows):
                raise ValueError(f'prescribed stress callable returned shape {values.shape}, expected {(m, q, rows)}')
            stress = stress + values
        else:
            stress = stress + _element_rows(field.prescribed, elements, n_elements, rows, 'prescribed stress')
    if field.thermal is not None:
        stress = stress - _element_rows(field.thermal, elements, n_elements, rows, 'thermal stress')
    return stress


def _prestress_degree(axisymmetric, default, degree) -> int:
    """Triangle rule degree: given, else AXISYMMETRIC_DEGREE (axisymmetric) / max(default, 2); capped at the largest
    Dunavant rule."""
    if degree is None:
        degree = AXISYMMETRIC_DEGREE if axisymmetric else max(int(default), 2)
    return min(int(degree), TRIANGLE_DEGREES[-1])


def _rule_values(dofs, degree, axisymmetric):
    """Yield (positions, geometry, basis (q, n), physical gradients (m, q, n, 2), weights (m, q)) per element group."""
    for positions, rule in _element_groups(dofs, degree):
        geometry = element_geometry(dofs.mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
        basis, reference_gradients = lagrange_basis(dofs.order, rule.points)
        yield (positions, geometry, basis, physical_gradients(reference_gradients, geometry.inverse),
               _weights(geometry, rule, axisymmetric))


def _stress_at(field, dofs, positions, geometry, operator, axisymmetric) -> np.ndarray:
    """(m, q, k) stress of a StressField at the rule points of the map's elements at positions (operator = B there,
    needed only with a displacement)."""
    elements = dofs.elements[positions]
    stress = fixed_stress_at(field, geometry.points, elements, dofs.mesh.n_elements, strain_rows(axisymmetric))
    if field.displacement is not None:
        displacement = np.asarray(field.displacement, dtype=np.float64).reshape(-1)
        local = displacement[vector_dofs(dofs.element_dofs[positions])]               # (m, 2 n)
        strain = np.einsum('mqki,mi->mqk', operator, local)
        stress = stress + np.einsum('mkl,mql->mqk', material_matrix(field.material, elements, axisymmetric), strain)
    return stress


def _geometric_blocks(basis, gradients, points, stress, weights, axisymmetric) -> np.ndarray:
    """(m, 2 n, 2 n) element blocks of int grad N_a . sigma grad N_b w for both components (+ sigma_tt N_a N_b / r^2
    on u_r, axisymmetric) of the stress (m, q, k) at the rule points."""
    k = stress.shape[-1]
    tensor = np.stack([np.stack([stress[..., 0], stress[..., k - 1]], axis=-1),
                       np.stack([stress[..., k - 1], stress[..., 1]], axis=-1)], axis=-2)   # (m, q, 2, 2)
    scalar = _tensor_blocks(gradients, gradients, tensor, weights)
    m, n = scalar.shape[:2]
    blocks = np.zeros((m, COMPONENTS * n, COMPONENTS * n), dtype=scalar.dtype)
    blocks[:, 0::2, 0::2] = scalar
    blocks[:, 1::2, 1::2] = scalar
    if axisymmetric:
        hoop = stress[..., 2] / points[..., 0] ** 2
        blocks[:, 0::2, 0::2] += _basis_product_blocks(basis, hoop * weights)
    return blocks


def _append_blocks(rows, cols, values, dofs, positions, blocks) -> None:
    append_blocks(rows, cols, values, vector_dofs(dofs.element_dofs[positions]), blocks)


def assemble_geometric_stiffness(dofs, stress, *, axisymmetric=False, degree=None, stretch=None, radial_ratio=None):
    """CSR (2 n, 2 n) K_sigma = int grad N_a . sigma grad N_b dV per component (axisymmetric + sigma_tt N_a N_b / r^2 on
    u_r, dV = 2 pi r dA; planar per unit thickness as the stiffness); stress a StressField or its prescribed forms;
    stretch / radial_ratio: the PML form (the stress evaluated unstretched, the gradients and weights stretched)."""
    field = as_stress_field(stress)
    rule_degree = _prestress_degree(axisymmetric, 2 * dofs.order - 2 + stress_degree(field, dofs.order)
                                    + _stretch_degree(stretch), degree)
    n_vector = COMPONENTS * dofs.n_dofs
    rows, cols, values = [], [], []
    for positions, geometry, basis, gradients, weights in _rule_values(dofs, rule_degree, axisymmetric):
        operator = (strain_operator(basis, gradients, geometry.points, axisymmetric)
                    if field.displacement is not None else None)
        stress_values = _stress_at(field, dofs, positions, geometry, operator, axisymmetric)
        gradients, weights, points = _stretched(geometry, gradients, weights, dofs.elements[positions], stretch,
                                                radial_ratio, axisymmetric)
        blocks = _geometric_blocks(basis, gradients, points, stress_values, weights, axisymmetric)
        _append_blocks(rows, cols, values, dofs, positions, blocks)
    return _to_csr(rows, cols, values, (n_vector, n_vector))


def initial_stress_force(dofs, stress, *, axisymmetric=False, degree=None) -> np.ndarray:
    """Vector (2 n,) of -int B^T sigma0 dV: the load of an initial stress (a stationary K u = f + this relaxes it, total
    stress sigma0 + C eps); stress a StressField or its prescribed forms; uniform sigma0 = -(its boundary tractions)."""
    field = as_stress_field(stress)
    rule_degree = _prestress_degree(axisymmetric, 2 * dofs.order - 1 + stress_degree(field, dofs.order), degree)
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    for positions, geometry, basis, gradients, weights in _rule_values(dofs, rule_degree, axisymmetric):
        operator = strain_operator(basis, gradients, geometry.points, axisymmetric)
        stress_values = _stress_at(field, dofs, positions, geometry, operator, axisymmetric)
        local = np.einsum('mqki,mqk,mq->mi', operator, stress_values, weights)
        if np.iscomplexobj(local) and not np.iscomplexobj(result):
            result = result.astype(np.complex128)
        scatter_add(result, vector_dofs(dofs.element_dofs[positions]), -local)
    return result


def sample_stress(dofs, stress, reference_points, *, axisymmetric=False) -> np.ndarray:
    """(m, q, k) stress of a StressField (or its prescribed forms) at reference points (q, 2) of every element of the
    map: sigma0 + C eps(u) - beta dT (hoop strain on the axis as element_strains), the total stress output."""
    field = as_stress_field(stress)
    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2)
    elements = dofs.elements
    rows = strain_rows(axisymmetric)
    parts = []
    for start in range(0, len(elements), CHUNK_ELEMENTS):
        chunk = elements[start:start + CHUNK_ELEMENTS]
        geometry = element_geometry(dofs.mesh, chunk, reference_points, dofs.geometry_order)
        parts.append(fixed_stress_at(field, geometry.points, chunk, dofs.mesh.n_elements, rows))
    result = np.concatenate(parts) if parts else np.zeros((0, len(reference_points), rows))
    if field.displacement is not None:
        _, strains = element_strains(dofs, field.displacement, reference_points, axisymmetric=axisymmetric)
        result = result + np.einsum('mkl,mql->mqk', material_matrix(field.material, elements, axisymmetric), strains)
    return result


# ---- Green-Lagrange pieces of the total-Lagrangian form (geometric nonlinearity of the triangle solid, K6) ----
def _green_lagrange_at(basis, gradients, points, local, axisymmetric, axis_radius=0.0) -> tuple:
    """(E (m, q, k) Voigt, H (m, q, 2, 2) with H[c, d] = d u_c / d x_d, hoop u_r / r (m, q) or None) of element
    displacements local (m, n, 2); on the axis (r <= axis_radius) the hoop ratio is d u_r / d r."""
    gradient = np.einsum('mqnd,mnc->mqcd', gradients, local)
    k = strain_rows(axisymmetric)
    strains = np.empty(gradient.shape[:2] + (k,))
    strains[..., 0] = gradient[..., 0, 0] + 0.5 * (gradient[..., 0, 0] ** 2 + gradient[..., 1, 0] ** 2)
    strains[..., 1] = gradient[..., 1, 1] + 0.5 * (gradient[..., 0, 1] ** 2 + gradient[..., 1, 1] ** 2)
    strains[..., k - 1] = (gradient[..., 0, 1] + gradient[..., 1, 0] + gradient[..., 0, 0] * gradient[..., 0, 1]
                           + gradient[..., 1, 0] * gradient[..., 1, 1])
    hoop = None
    if axisymmetric:
        radius = points[..., 0]
        on_axis = radius <= axis_radius
        hoop = np.where(on_axis, gradient[..., 0, 0],
                        np.einsum('qn,mn->mq', basis, local[..., 0]) / np.where(on_axis, 1.0, radius))
        strains[..., 2] = hoop + 0.5 * hoop ** 2
    return strains, gradient, hoop


def nonlinear_strain_operator(values, gradients, points, displacement_gradient, hoop=None, *,
                              axisymmetric=False) -> np.ndarray:
    """B_NL (m, q, k, 2 n) with delta E = (B + B_NL) delta u at the displacement gradient H (m, q, 2, 2) (H[c, d] = d u_c
    / d x_d; values (q, n), gradients (m, q, n, 2), points (m, q, 2), hoop u_r / r (m, q) axisymmetric)."""
    m, q, n, _ = gradients.shape
    k = strain_rows(axisymmetric)
    gradient = displacement_gradient
    operator = np.zeros((m, q, k, n, COMPONENTS), dtype=np.result_type(gradients, gradient))
    operator[..., 0, :, :] = gradients[..., :, 0, None] * gradient[..., None, :, 0]
    operator[..., 1, :, :] = gradients[..., :, 1, None] * gradient[..., None, :, 1]
    operator[..., k - 1, :, :] = (gradients[..., :, 1, None] * gradient[..., None, :, 0]
                                  + gradients[..., :, 0, None] * gradient[..., None, :, 1])
    if axisymmetric:
        operator[..., 2, :, 0] = values[None, :, :] * (hoop / points[..., 0])[..., None]
    return operator.reshape(m, q, k, COMPONENTS * n)


def green_lagrange_strains(dofs, displacement, reference_points, *, axisymmetric=False) -> np.ndarray:
    """(m, q, k) Green-Lagrange strain (H + H^T + H^T H) / 2 in the Voigt rows (engineering shear 2 E_xy; hoop u_r / r +
    (u_r / r)^2 / 2, d u_r / d r on the axis) of every element of the map at reference points (q, 2)."""
    displacement = np.asarray(displacement, dtype=np.float64).reshape(dofs.n_dofs, COMPONENTS)
    reference_points = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2)
    basis, reference_gradients = lagrange_basis(dofs.order, reference_points)
    axis_radius = AXIS_TOLERANCE * float(np.abs(dofs.mesh.points).max())
    m = len(dofs.elements)
    strains = np.empty((m, len(reference_points), strain_rows(axisymmetric)))
    for start in range(0, m, CHUNK_ELEMENTS):
        chunk = slice(start, start + CHUNK_ELEMENTS)
        geometry = element_geometry(dofs.mesh, dofs.elements[chunk], reference_points, dofs.geometry_order)
        gradients = physical_gradients(reference_gradients, geometry.inverse)
        strains[chunk], _, _ = _green_lagrange_at(basis, gradients, geometry.points,
                                                  displacement[dofs.element_dofs[chunk]], axisymmetric, axis_radius)
    return strains


def _fixed_field(initial_stress):
    """StressField of a total-Lagrangian initial stress S0 (prescribed / thermal only), None for none."""
    if initial_stress is None:
        return None
    field = as_stress_field(initial_stress)
    if field.displacement is not None:
        raise ValueError('initial_stress of the total-Lagrangian form must not carry a displacement')
    return field


def _green_lagrange_group(basis, gradients, points, local, material, fixed_stress, axisymmetric) -> tuple:
    """(B + B_NL (m, q, k, 2 n), S (m, q, k)) of one element group from element displacements local (m, n, 2), C (m,
    k, k) and S0 (m, q, k) or None: S = C E(u) + S0."""
    strains, gradient, hoop = _green_lagrange_at(basis, gradients, points, local, axisymmetric)
    stress = np.einsum('mkl,mql->mqk', material, strains)
    if fixed_stress is not None:
        stress = stress + fixed_stress
    operator = (strain_operator(basis, gradients, points, axisymmetric)
                + nonlinear_strain_operator(basis, gradients, points, gradient, hoop, axisymmetric=axisymmetric))
    return operator, stress


def _tangent_blocks(basis, gradients, points, weights, operator, stress, material, axisymmetric) -> np.ndarray:
    """(m, 2 n, 2 n) element tangents int (B + B_NL)^T C (B + B_NL) dV + K_sigma(S)."""
    return _material_blocks(operator, material, weights) + _geometric_blocks(basis, gradients, points, stress, weights, axisymmetric)


def _green_lagrange_terms(dofs, displacement, coefficients, axisymmetric, initial_stress, degree):
    """Yield (positions, geometry, basis, gradients, weights, B + B_NL (m, q, k, 2 n), S (m, q, k), C (m, k, k)) on the
    total-Lagrangian rule (default 4 (order - 1), axisymmetric 6): S = C E(u) + S0."""
    fixed = _fixed_field(initial_stress)
    displacement = np.asarray(displacement, dtype=np.float64).reshape(dofs.n_dofs, COMPONENTS)
    rows = strain_rows(axisymmetric)
    rule_degree = _prestress_degree(axisymmetric, 4 * (dofs.order - 1), degree)
    for positions, geometry, basis, gradients, weights in _rule_values(dofs, rule_degree, axisymmetric):
        elements = dofs.elements[positions]
        material = material_matrix(coefficients, elements, axisymmetric)
        fixed_stress = (None if fixed is None
                        else fixed_stress_at(fixed, geometry.points, elements, dofs.mesh.n_elements, rows))
        operator, stress = _green_lagrange_group(basis, gradients, geometry.points,
                                                 displacement[dofs.element_dofs[positions]], material, fixed_stress,
                                                 axisymmetric)
        yield positions, geometry, basis, gradients, weights, operator, stress, material


def internal_force(dofs, displacement, coefficients, *, axisymmetric=False, initial_stress=None,
                   degree=None) -> np.ndarray:
    """Vector (2 n,) total-Lagrangian internal force int (B + B_NL(u))^T S dV, S = C E(u) + S0 (second Piola-Kirchhoff;
    initial_stress S0 in the prescribed / thermal forms); K u - initial_stress_force(S0) to first order in u."""
    result = np.zeros(COMPONENTS * dofs.n_dofs)
    for positions, _, _, _, weights, operator, stress, _ in _green_lagrange_terms(
            dofs, displacement, coefficients, axisymmetric, initial_stress, degree):
        local = np.einsum('mqki,mqk,mq->mi', operator, stress, weights)
        scatter_add(result, vector_dofs(dofs.element_dofs[positions]), local)
    return result


def assemble_tangent_stiffness(dofs, displacement, coefficients, *, axisymmetric=False, initial_stress=None,
                               degree=None):
    """CSR (2 n, 2 n) d internal_force / d u = int (B + B_NL)^T C (B + B_NL) dV + K_sigma(S) on the same rule
    (symmetric); at u = 0 it is the elastic stiffness + assemble_geometric_stiffness(S0) (a self-test reference: the
    tangent of internal_force_and_tangent)."""
    return internal_force_and_tangent(dofs, displacement, coefficients, axisymmetric=axisymmetric,
                                      initial_stress=initial_stress, degree=degree)[1]


def internal_force_and_tangent(dofs, displacement, coefficients, *, axisymmetric=False, initial_stress=None,
                               degree=None) -> tuple:
    """(internal_force, assemble_tangent_stiffness) of one pass over the total-Lagrangian rule (a Newton iteration)."""
    n_vector = COMPONENTS * dofs.n_dofs
    force = np.zeros(n_vector)
    rows, cols, values = [], [], []
    for positions, geometry, basis, gradients, weights, operator, stress, material in _green_lagrange_terms(
            dofs, displacement, coefficients, axisymmetric, initial_stress, degree):
        local = np.einsum('mqki,mqk,mq->mi', operator, stress, weights)
        scatter_add(force, vector_dofs(dofs.element_dofs[positions]), local)
        blocks = _tangent_blocks(basis, gradients, geometry.points, weights, operator, stress, material, axisymmetric)
        _append_blocks(rows, cols, values, dofs, positions, blocks)
    return force, _to_csr(rows, cols, values, (n_vector, n_vector))


class TangentAssembler:
    """Total-Lagrangian internal force and tangent of one solid map (the 2D twin of elasticity3d.TangentAssembler3D):
    rule points, basis, gradients, weights and C cached per element group, values summed on a fixed CSR pattern
    (fixed_pattern.FixedPatternCsr) with its free-free restriction."""

    def __init__(self, dofs, coefficients, *, axisymmetric=False, free=None, degree=None):
        self.dofs = dofs
        self.axisymmetric = bool(axisymmetric)
        self.n_vector = COMPONENTS * dofs.n_dofs
        rule_degree = _prestress_degree(self.axisymmetric, 4 * (dofs.order - 1), degree)
        self.groups = []   # (positions, points, basis, gradients, weights, C, vector dofs) per element group
        for positions, geometry, basis, gradients, weights in _rule_values(dofs, rule_degree, self.axisymmetric):
            elements = dofs.elements[positions]
            self.groups.append((positions, geometry.points, basis, gradients, weights,
                                material_matrix(coefficients, elements, self.axisymmetric),
                                vector_dofs(dofs.element_dofs[positions])))
        self.pattern = FixedPatternCsr(self.n_vector, [group[-1] for group in self.groups], free=free)

    def _terms(self, displacement, initial_stress):
        """Yield (group index, B + B_NL, S) of every group at the displacement (S0 = initial_stress)."""
        fixed = _fixed_field(initial_stress)
        displacement = np.asarray(displacement, dtype=np.float64).reshape(self.dofs.n_dofs, COMPONENTS)
        rows = strain_rows(self.axisymmetric)
        for index, (positions, points, basis, gradients, _, material, _) in enumerate(self.groups):
            fixed_stress = (None if fixed is None else
                            fixed_stress_at(fixed, points, self.dofs.elements[positions], self.dofs.mesh.n_elements,
                                            rows))
            operator, stress = _green_lagrange_group(basis, gradients, points,
                                                     displacement[self.dofs.element_dofs[positions]], material,
                                                     fixed_stress, self.axisymmetric)
            yield index, operator, stress

    def force(self, displacement, initial_stress=None) -> np.ndarray:
        """internal_force (2 n,) at the displacement."""
        force = np.zeros(self.n_vector)
        for index, operator, stress in self._terms(displacement, initial_stress):
            weights, local_dofs = self.groups[index][4], self.groups[index][6]
            scatter_add(force, local_dofs, np.einsum('mqki,mqk,mq->mi', operator, stress, weights))
        return force

    def force_and_tangent(self, displacement, initial_stress=None) -> tuple:
        """(internal force (2 n,), tangent data (nnz,)) in one pass; CSRs by `full` / `reduced`."""
        force = np.zeros(self.n_vector)
        data = self.pattern.zeros()
        for index, operator, stress in self._terms(displacement, initial_stress):
            _, points, basis, gradients, weights, material, local_dofs = self.groups[index]
            scatter_add(force, local_dofs, np.einsum('mqki,mqk,mq->mi', operator, stress, weights))
            self.pattern.add(data, index, _tangent_blocks(basis, gradients, points, weights, operator, stress, material,
                                                          self.axisymmetric))
        return force, data

    def full(self, data, copy_pattern=False):
        """Host CSR (2 n, 2 n) of tangent data (FixedPatternCsr.full)."""
        return self.pattern.full(data, copy_pattern=copy_pattern)

    def reduced(self, data):
        """Host canonical CSR of the free rows / columns of tangent data."""
        return self.pattern.reduced(data)

    def tangent(self, displacement, initial_stress=None):
        """assemble_tangent_stiffness on the fixed pattern (own index copies)."""
        return self.full(self.force_and_tangent(displacement, initial_stress)[1], copy_pattern=True)


# ---- azimuthal order m >= 1 of the axisymmetric solid (new_modules_plan K7; module header) ----
def _check_azimuthal_order(order) -> int:
    if isinstance(order, bool) or int(order) != order or int(order) < 1:
        raise ValueError(f'azimuthal order must be an integer >= 1 (0 is the axisymmetric form), got {order!r}')
    return int(order)


def component_dofs(element_dofs, components) -> np.ndarray:
    """(m, c n) interleaved dofs c d + component of scalar element dofs (m, n) (c = components)."""
    element_dofs = np.asarray(element_dofs, dtype=np.int64)
    return (components * element_dofs[:, :, None] + np.arange(components)).reshape(len(element_dofs), -1)


def azimuthal_material_matrix(coefficients, elements) -> np.ndarray:
    """(m, 6, 6) Voigt stiffness of the order-m rows: the orthotropic (rr, zz, tt, rz) block of material_matrix plus
    c44 (zt) and c55 (rt), each defaulting to c66 (isotropic)."""
    matrix = np.zeros((len(elements), AZIMUTHAL_ROWS, AZIMUTHAL_ROWS))
    matrix[:, :4, :4] = material_matrix(coefficients, elements, True)
    for row, name in ((4, 'c44'), (5, 'c55')):
        value = coefficients.get(name, coefficients.get('c66'))
        if value is not None:
            matrix[:, row, row] = np.asarray(value, dtype=np.float64)[elements] if np.ndim(value) else float(value)
    return matrix


def azimuthal_strain_operator(values, gradients, radius, order) -> np.ndarray:
    """B (m, q, 6, 3 n) of the order-m amplitudes (U_r, U_z, U_phi) (module header rows; values (q, n), physical
    gradients (m, q, n, 2), radius (m, q): r, or the stretched r~ of a PML)."""
    m, q, n, _ = gradients.shape
    operator = np.zeros((m, q, AZIMUTHAL_ROWS, AZIMUTHAL_COMPONENTS * n), dtype=np.result_type(gradients, radius))
    radial, axial, azimuthal = slice(0, None, 3), slice(1, None, 3), slice(2, None, 3)
    over_radius = values[None, :, :] / radius[..., None]   # N / r
    d_r, d_z = gradients[..., 0], gradients[..., 1]
    operator[..., 0, radial] = d_r
    operator[..., 1, axial] = d_z
    operator[..., 2, radial] = over_radius
    operator[..., 2, azimuthal] = order * over_radius
    operator[..., 3, radial] = d_z
    operator[..., 3, axial] = d_r
    operator[..., 4, azimuthal] = d_z
    operator[..., 4, axial] = -order * over_radius
    operator[..., 5, radial] = -order * over_radius
    operator[..., 5, azimuthal] = d_r - over_radius
    return operator


def _azimuthal_terms(dofs, positions, rule, stretch, radial_ratio):
    """(geometry, basis (q, n), gradients (m, q, n, 2), weights pi r w det J (m, q), radius (m, q)) of the order-m form
    at a rule (a PML stretch: stretched gradients, weights and r~)."""
    geometry = element_geometry(dofs.mesh, dofs.elements[positions], rule.points, dofs.geometry_order)
    basis, reference_gradients = lagrange_basis(dofs.order, rule.points)
    gradients = physical_gradients(reference_gradients, geometry.inverse)
    weights = rule.weights[None, :] * geometry.determinant * (AZIMUTHAL_RING * geometry.points[..., 0])
    gradients, weights, points = _stretched(geometry, gradients, weights, dofs.elements[positions], stretch,
                                            radial_ratio, True)
    return geometry, basis, gradients, weights, points[..., 0]


def _append_component_blocks(rows, cols, values, dofs, positions, blocks, components) -> None:
    append_blocks(rows, cols, values, component_dofs(dofs.element_dofs[positions], components), blocks)


def assemble_azimuthal_stiffness(dofs, coefficients, order, *, stretch=None, radial_ratio=None):
    """CSR (3 n, 3 n) of pi int B^T C B r dA of azimuthal order `order` >= 1 (coefficients as the axisymmetric
    material_matrix, + optional c44 / c55; stretch / radial_ratio: the PML form)."""
    order = _check_azimuthal_order(order)
    n_vector = AZIMUTHAL_COMPONENTS * dofs.n_dofs
    rows, cols, values = [], [], []
    for positions, rule in _groups(dofs, True, 0):
        _, basis, gradients, weights, radius = _azimuthal_terms(dofs, positions, rule, stretch, radial_ratio)
        operator = azimuthal_strain_operator(basis, gradients, radius, order)
        local = _material_blocks(operator, azimuthal_material_matrix(coefficients, dofs.elements[positions]), weights)
        _append_component_blocks(rows, cols, values, dofs, positions, local, AZIMUTHAL_COMPONENTS)
    return _to_csr(rows, cols, values, (n_vector, n_vector))


def assemble_azimuthal_mass(dofs, density, *, stretch=None, radial_ratio=None):
    """CSR (3 n, 3 n) consistent mass pi int rho N_a N_b r dA of the three order-m amplitudes (stretch: the PML
    form)."""
    from scipy.sparse import identity, kron

    scalar = _scalar_mass(dofs, density, True, stretch, radial_ratio) * AZIMUTHAL_RING
    return kron(scalar, identity(AZIMUTHAL_COMPONENTS), format='csr')


def _azimuthal_geometric_blocks(basis, gradients, radius, stress, weights, order) -> np.ndarray:
    """(m, 3 n, 3 n) blocks of sigma : (H^T H) of the order-m field: per component grad N . sigma grad N (rr, zz, rz),
    plus sigma_tt / r^2 [(m^2 + 1)(U_r^2 + U_phi^2) + 4 m U_r U_phi + m^2 U_z^2]."""
    tensor = np.stack([np.stack([stress[..., 0], stress[..., 3]], axis=-1),
                       np.stack([stress[..., 3], stress[..., 1]], axis=-1)], axis=-2)   # (m, q, 2, 2)
    scalar = _tensor_blocks(gradients, gradients, tensor, weights)
    hoop = _basis_product_blocks(basis, stress[..., 2] / radius ** 2 * weights)
    m, n = scalar.shape[:2]
    blocks = np.zeros((m, AZIMUTHAL_COMPONENTS * n, AZIMUTHAL_COMPONENTS * n), dtype=np.result_type(scalar, hoop))
    radial, axial, azimuthal = slice(0, None, 3), slice(1, None, 3), slice(2, None, 3)
    squared = order * order
    blocks[:, radial, radial] = scalar + (squared + 1.0) * hoop
    blocks[:, axial, axial] = scalar + squared * hoop
    blocks[:, azimuthal, azimuthal] = scalar + (squared + 1.0) * hoop
    blocks[:, radial, azimuthal] = 2.0 * order * hoop
    blocks[:, azimuthal, radial] = 2.0 * order * hoop
    return blocks


def assemble_azimuthal_geometric_stiffness(dofs, stress, order, *, degree=None, stretch=None, radial_ratio=None):
    """CSR (3 n, 3 n) K_sigma of azimuthal order `order` >= 1 for an axisymmetric prestress (rows rr, zz, tt, rz; a
    StressField without displacement or its prescribed forms; stretch: the PML form, stress unstretched)."""
    order = _check_azimuthal_order(order)
    field = as_stress_field(stress)
    if field.displacement is not None:
        raise ValueError('assemble_azimuthal_geometric_stiffness takes a prescribed (axisymmetric) stress only')
    rule_degree = _prestress_degree(True, 0, degree)
    n_vector = AZIMUTHAL_COMPONENTS * dofs.n_dofs
    rows, cols, values = [], [], []
    for positions, rule in _element_groups(dofs, rule_degree):
        geometry, basis, gradients, weights, radius = _azimuthal_terms(dofs, positions, rule, stretch, radial_ratio)
        stress_values = fixed_stress_at(field, geometry.points, dofs.elements[positions], dofs.mesh.n_elements,
                                        strain_rows(True))
        blocks = _azimuthal_geometric_blocks(basis, gradients, radius, stress_values, weights, order)
        _append_component_blocks(rows, cols, values, dofs, positions, blocks, AZIMUTHAL_COMPONENTS)
    return _to_csr(rows, cols, values, (n_vector, n_vector))


def azimuthal_element_strains(dofs, displacement, reference_points, order) -> tuple:
    """(amplitudes (m, q, 3), Voigt strains (m, q, 6), module header rows) of every element of the map at reference
    points (q, 2) from nodal (U_r, U_z, U_phi) (n, 3) of azimuthal order >= 1; on the axis each X / r is its limit d X /
    d r (X(0) = 0 there by the axis rule)."""
    order = _check_azimuthal_order(order)
    displacement = np.asarray(displacement, dtype=np.float64).reshape(dofs.n_dofs, AZIMUTHAL_COMPONENTS)
    reference_points = np.asarray(reference_points, dtype=np.float64)
    basis, reference_gradients = lagrange_basis(dofs.order, reference_points)
    m, q = len(dofs.elements), len(reference_points)
    values = np.empty((m, q, AZIMUTHAL_COMPONENTS))
    strains = np.empty((m, q, AZIMUTHAL_ROWS))
    axis_radius = AXIS_TOLERANCE * float(np.abs(dofs.mesh.points).max())
    for start in range(0, m, CHUNK_ELEMENTS):
        chunk = slice(start, start + CHUNK_ELEMENTS)
        geometry = element_geometry(dofs.mesh, dofs.elements[chunk], reference_points, dofs.geometry_order)
        gradients = physical_gradients(reference_gradients, geometry.inverse)
        local = displacement[dofs.element_dofs[chunk]]                     # (c, n, 3)
        amplitudes = np.einsum('qn,mnc->mqc', basis, local)
        derivative = np.einsum('mqnd,mnc->mqcd', gradients, local)         # d U_c / d x_d
        radius = geometry.points[..., 0]
        on_axis = radius <= axis_radius
        safe = np.where(on_axis, 1.0, radius)

        def over_radius(value, limit):
            return np.where(on_axis, limit, value / safe)
        u_r, u_z, u_phi = amplitudes[..., 0], amplitudes[..., 1], amplitudes[..., 2]
        values[chunk] = amplitudes
        strains[chunk, :, 0] = derivative[..., 0, 0]
        strains[chunk, :, 1] = derivative[..., 1, 1]
        strains[chunk, :, 2] = over_radius(u_r + order * u_phi, derivative[..., 0, 0] + order * derivative[..., 2, 0])
        strains[chunk, :, 3] = derivative[..., 0, 1] + derivative[..., 1, 0]
        strains[chunk, :, 4] = derivative[..., 2, 1] - order * over_radius(u_z, derivative[..., 1, 0])
        strains[chunk, :, 5] = (derivative[..., 2, 0] - over_radius(order * u_r + u_phi, order * derivative[..., 0, 0]
                                                                    + derivative[..., 2, 0]))
    return values, strains
