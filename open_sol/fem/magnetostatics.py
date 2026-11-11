"""Magnetostatics on triangle meshes (magnetic plan section 10, stage 5): the curl operator Q of Lagrange P1 / P2 A in
B-space (planar A_z: B = (dA/dy, -dA/dx); axisymmetric A_phi: B = (-dA/dz, dA/dr + A / r), weights r per radian),
cached at the quadrature points of a dof map's elements (curl_quadrature), B at those points, vectors int Q^T v dV
(residual of H, load of nu B_r), the tangent int Q^T D Q dV and integrals of point values. W, R and K_t share the rule
(decision 3: Newton stays quadratic and the energy line search consistent); magnetic_fields_ac adds the mass and loads
on the same rule (complex values allowed). Host numpy, scipy inside functions."""
from dataclasses import dataclass

import numpy as np

from .assembly import _element_groups, _tensor_blocks, _to_csr, _weights, append_blocks, scatter_local
from .elements import element_geometry, lagrange_basis, physical_gradients

PLANAR_DEGREES = {1: 2, 2: 4}   # rule degree per Lagrange order (P2: |B| varies within the element); +1 axisymmetric


@dataclass(frozen=True, eq=False)
class CurlBlock:
    """One element chunk of a dof map at its rule: mesh elements (m,), their dofs (m, n), weights w det J (x r when
    axisymmetric: per radian) (m, q), points (m, q, 2), the Lagrange values (q, n) and the curl operator Q
    (m, q, 2, n)."""
    elements: np.ndarray
    element_dofs: np.ndarray
    weights: np.ndarray
    points: np.ndarray
    basis: np.ndarray
    operator: np.ndarray


def curl_operator(values, gradients, points, axisymmetric, axis_radius=0.0) -> np.ndarray:
    """Q (m, q, 2, n): B of each Lagrange basis function (values (q, n), physical gradients (m, q, n, 2), points
    (m, q, 2)); axisymmetric points with r <= axis_radius take the axis limit A / r -> dA/dr (A = 0 on the axis)."""
    m, q, n, _ = gradients.shape
    operator = np.empty((m, q, 2, n))
    if not axisymmetric:
        operator[..., 0, :] = gradients[..., 1]
        operator[..., 1, :] = -gradients[..., 0]
        return operator
    radius = points[..., 0:1]
    on_axis = radius <= axis_radius
    operator[..., 0, :] = -gradients[..., 1]
    operator[..., 1, :] = gradients[..., 0] + np.where(on_axis, gradients[..., 0],
                                                       values[None, :, :] / np.where(on_axis, 1.0, radius))
    return operator


def curl_quadrature(dofs, axisymmetric) -> tuple:
    """CurlBlocks of every element of a Lagrange dof map: Dunavant degree 2 (P1) / 4 (P2), +1 axisymmetric, curved
    elements at least CURVED_DEGREE (fem.assembly element groups)."""
    degree = PLANAR_DEGREES[dofs.order] + int(axisymmetric)
    blocks = []
    for positions, rule in _element_groups(dofs, degree):
        elements = dofs.elements[positions]
        geometry = element_geometry(dofs.mesh, elements, rule.points, dofs.geometry_order)
        values, reference_gradients = lagrange_basis(dofs.order, rule.points)
        gradients = physical_gradients(reference_gradients, geometry.inverse)
        operator = curl_operator(values, gradients, geometry.points, axisymmetric)
        blocks.append(CurlBlock(elements, dofs.element_dofs[positions], _weights(geometry, rule, axisymmetric),
                                geometry.points, values, operator))
    return tuple(blocks)


def local_flux_density(dofs, coefficients, reference_points, axisymmetric, axis_radius) -> np.ndarray:
    """B (m, p, 2) of every element of the map at reference points (p, 2) (the local dof points of nodal recovery;
    axis points: the A / r limit)."""
    values, reference_gradients = lagrange_basis(dofs.order, reference_points)
    result = np.empty((len(dofs.elements), len(reference_points), 2),
                      dtype=np.result_type(np.asarray(coefficients).dtype, np.float64))
    for positions, _ in _element_groups(dofs, 2):
        geometry = element_geometry(dofs.mesh, dofs.elements[positions], reference_points, dofs.geometry_order)
        gradients = physical_gradients(reference_gradients, geometry.inverse)
        operator = curl_operator(values, gradients, geometry.points, axisymmetric, axis_radius)
        result[positions] = np.einsum('mqcn,mn->mqc', operator, coefficients[dofs.element_dofs[positions]])
    return result


def quadrature_values(blocks, coefficients) -> list:
    """A (m, q) per block of dof coefficients a (n_dofs,)."""
    return [np.einsum('qn,mn->mq', block.basis, coefficients[block.element_dofs]) for block in blocks]


def quadrature_flux_density(blocks, coefficients) -> list:
    """B (m, q, 2) per block of dof coefficients a (n_dofs,)."""
    return [np.einsum('mqcn,mn->mqc', block.operator, coefficients[block.element_dofs]) for block in blocks]


def curl_vector(blocks, vectors, n_dofs) -> np.ndarray:
    """(n_dofs,) int Q^T v dV of point vectors v (m, q, 2) per block (the internal force of H, the load of nu B_r)."""
    result = np.zeros(n_dofs)
    for block, vector in zip(blocks, vectors):
        count = len(block.elements)
        weighted = (np.asarray(vector) * block.weights[..., None]).reshape(count, 1, -1)    # (m, 1, q 2)
        local = (weighted @ block.operator.reshape(count, -1, block.operator.shape[-1]))[:, 0]   # (m, n)
        result += scatter_local(n_dofs, block.element_dofs, local)
    return result


def curl_tangent_blocks(blocks, tangents) -> list:
    """Element blocks (m, n, n) of int Q^T D Q dV per block for point tensors D (m, q, 2, 2) (fem.assembly
    _tensor_blocks over the curl vectors: no per-point matmul)."""
    return [_tensor_blocks(block.operator.swapaxes(-1, -2), block.operator.swapaxes(-1, -2), tangent, block.weights)
            for block, tangent in zip(blocks, tangents)]


def assemble_curl_tangent(blocks, tangents, n_dofs):
    """CSR (n_dofs, n_dofs) int Q^T D Q dV of point tensors D (m, q, 2, 2) per block (symmetric D: symmetric CSR)."""
    rows, cols, values = [], [], []
    for block, local in zip(blocks, curl_tangent_blocks(blocks, tangents)):
        append_blocks(rows, cols, values, block.element_dofs, local)
    return _to_csr(rows, cols, values, (n_dofs, n_dofs))


def quadrature_integral(blocks, values):
    """int v dV (per radian when axisymmetric) of point values v (m, q) per block: float, complex for complex values."""
    total = sum(np.einsum('mq,mq->', value, block.weights) for block, value in zip(blocks, values))
    return complex(total) if np.iscomplexobj(total) else float(total)


def _value_dtype(values):
    """float64, or complex128 when a point-value array is complex."""
    return np.result_type(np.float64, *(np.asarray(value).dtype for value in values))


def block_mass(blocks, coefficients, n_dofs):
    """CSR (n_dofs, n_dofs) int c phi_i phi_j dV of per-element coefficients c (n_elements,) on the blocks' rule (the
    eddy-current mass sharing the curl tangent's quadrature)."""
    coefficients = np.asarray(coefficients)
    rows, cols, values = [], [], []
    for block in blocks:
        local = np.einsum('qi,qj,mq->mij', block.basis, block.basis,
                          block.weights * coefficients[block.elements][:, None])
        append_blocks(rows, cols, values, block.element_dofs, local)
    return _to_csr(rows, cols, values, (n_dofs, n_dofs))


def block_load(blocks, values, n_dofs) -> np.ndarray:
    """(n_dofs,) int v phi_i dV of point values v (m, q) per block, real or complex (sources, conductor rows)."""
    result = np.zeros(n_dofs, dtype=_value_dtype(values))
    for block, value in zip(blocks, values):
        local = (value * block.weights) @ block.basis   # (m, q) @ (q, n): one GEMM
        result += scatter_local(n_dofs, block.element_dofs, local)
    return result


def element_integrals(blocks, values, n_elements) -> np.ndarray:
    """(n_elements, ...) int_e v dV per mesh element of point values v (m, q, ...) per block (0 off the blocks; complex
    for complex values)."""
    trailing = values[0].shape[2:] if values else ()
    result = np.zeros((n_elements,) + trailing, dtype=_value_dtype(values))
    for block, value in zip(blocks, values):
        result[block.elements] = np.einsum('mq...,mq->m...', value, block.weights)
    return result
