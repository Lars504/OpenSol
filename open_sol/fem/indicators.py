"""Error indicators on triangle meshes (mesh plan A3, S7.7): eta^2 per element of a scalar Lagrange solution (element
residual + Kelly edge jumps of the normal flux C grad u . n, energy-norm weighted, P1 / P2) and the edge sampling of
both sides of an edge that the em_modes jumps (tangential H, normal D) share. Host numpy."""
import math
from dataclasses import dataclass

import numpy as np

from .elements import (REFERENCE_GRADIENTS, _inverse_2x2, curved_elements, edge_geometry, edge_reference_points,
                       element_geometry, lagrange_basis, nedelec_basis, physical_gradients, piola_vectors)
from .fixed_pattern import scatter_add
from .quadrature import CURVED_DEGREE, gauss_line, triangle_rule

EDGE_DEGREE = 4          # Gauss rule along edges (jumps of P2 fluxes are linear: squared quadratic)
RESIDUAL_DEGREE = 4      # element rule of the residual integral (CURVED_DEGREE on curved elements)
CHUNK_ITEMS = 32768      # edges / elements per evaluation block
# edge condition codes of diffusion_indicator
_NONE, _NEUMANN, _ROBIN, _DIRICHLET = 0, 1, 2, 3


def element_sizes(mesh) -> np.ndarray:
    """Float64 (n_elements,) equilateral-equivalent edge length sqrt(4 A / sqrt(3)) of the straight elements [m]."""
    corners = mesh.points[mesh.triangles]
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    area = 0.5 * np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    return np.sqrt(4.0 * area / math.sqrt(3.0))


def element_diameters(mesh) -> np.ndarray:
    """Float64 (n_elements,) longest straight edge [m]."""
    corners = mesh.points[mesh.triangles]
    return np.max(np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2), axis=1)


@dataclass(frozen=True, eq=False)
class EdgeSides:
    """Mesh edges sampled at Gauss points with the elements on both sides (second -1: one-sided, an outline of the
    element set): local edge indices, points, weights |dx/dt| w (sum = edge length), unit normals out of side 0."""
    edges: np.ndarray        # int64 (k,)
    elements: np.ndarray     # int64 (k, 2)
    local: np.ndarray        # int64 (k, 2) local edge index of the edge in each side's element (-1 without)
    signs: np.ndarray        # int8 (k, 2) mesh.edge_signs of that local edge (0 without)
    t: np.ndarray            # (q,) parameters from the lower to the higher node
    points: np.ndarray       # (k, q, 2) [m]
    weights: np.ndarray      # (k, q) [m]
    normals: np.ndarray      # (k, q, 2)
    lengths: np.ndarray      # (k,) [m]

    @property
    def two_sided(self) -> np.ndarray:
        return self.elements[:, 1] >= 0

    def reference_points(self, side, rows=None) -> np.ndarray:
        """(k, q, 2) reference points of the sample points in the element of `side` (0 / 1) of the rows (None = all;
        rows of a missing side are invalid)."""
        rows = np.arange(len(self.edges)) if rows is None else np.asarray(rows, dtype=np.int64)
        return edge_reference_points(self.signs[rows, side], np.maximum(self.local[rows, side], 0), self.t)

    def select(self, rows) -> 'EdgeSides':
        """The rows `rows` (int indices or bool mask) as an EdgeSides."""
        rows = np.flatnonzero(rows) if np.asarray(rows).dtype == bool else np.asarray(rows, dtype=np.int64)
        return EdgeSides(edges=self.edges[rows], elements=self.elements[rows], local=self.local[rows],
                         signs=self.signs[rows], t=self.t, points=self.points[rows], weights=self.weights[rows],
                         normals=self.normals[rows], lengths=self.lengths[rows])


def edge_sides(mesh, edges, inside, geometry_order=2, degree=EDGE_DEGREE) -> EdgeSides:
    """EdgeSides of mesh edges with at least one element in the bool element set `inside` (others dropped): side 0 an
    element of the set, side 1 the other one when it is in the set too; geometry_order 2 follows curved edges."""
    edges = np.asarray(edges, dtype=np.int64)
    inside = np.asarray(inside, dtype=bool)
    pairs = mesh.edge_elements[edges].astype(np.int64)
    first_in = inside[pairs[:, 0]]
    second_in = (pairs[:, 1] >= 0) & inside[np.maximum(pairs[:, 1], 0)]
    keep = first_in | second_in
    edges, pairs, first_in, second_in = edges[keep], pairs[keep], first_in[keep], second_in[keep]
    side0 = np.where(first_in, pairs[:, 0], pairs[:, 1])
    side1 = np.where(first_in & second_in, pairs[:, 1], -1)
    elements = np.stack([side0, side1], axis=1)
    local = np.full(elements.shape, -1, dtype=np.int64)
    signs = np.zeros(elements.shape, dtype=np.int8)
    for side in (0, 1):
        present = elements[:, side] >= 0
        local[present, side] = np.argmax(mesh.element_edges[elements[present, side]] == edges[present, None], axis=1)
        signs[present, side] = mesh.edge_signs[elements[present, side], local[present, side]]
    t, gauss_weights = gauss_line(degree + (2 if geometry_order == 2 else 0))
    points, line_jacobians, tangents = edge_geometry(mesh, edges, t, order=geometry_order)
    weights = gauss_weights[None, :] * line_jacobians
    normals = np.stack([tangents[..., 1], -tangents[..., 0]], axis=-1)
    centroids = mesh.points[mesh.triangles[side0]].mean(axis=1)
    middle = points.mean(axis=1)
    outward = np.einsum('kc,kc->k', middle - centroids, normals.mean(axis=1)) >= 0.0
    normals = np.where(outward[:, None, None], normals, -normals)
    return EdgeSides(edges=edges, elements=elements, local=local, signs=signs, t=t, points=points, weights=weights,
                     normals=normals, lengths=weights.sum(axis=1))


def lagrange_at(dofs, coefficients, elements, reference_points) -> tuple:
    """(values (k, q), physical gradients (k, q, 2)) of a Lagrange field (coefficients in the DofMap's numbering, real or
    complex) at per-element reference points (k, q, 2) of mesh elements (k,) of the map."""
    elements = np.asarray(elements, dtype=np.int64)
    positions = dofs.element_position[elements]
    if (positions < 0).any():
        raise ValueError('lagrange_at: elements outside the dof map')
    local = np.asarray(coefficients)[dofs.element_dofs[positions]]
    values, reference_gradients = lagrange_basis(dofs.order, reference_points)
    geometry = element_geometry(dofs.mesh, elements, reference_points, dofs.geometry_order)
    gradients = physical_gradients(reference_gradients, geometry.inverse)
    return np.einsum('kqn,kn->kq', values, local), np.einsum('kqnc,kn->kqc', gradients, local)


def nedelec_at(dofs, coefficients, elements, reference_points) -> tuple:
    """(physical vectors (k, q, 2), curls (k, q)) of a Nedelec field (coefficients in the DofMap's numbering) at
    per-element reference points (k, q, 2) of mesh elements (k,) of the map."""
    elements = np.asarray(elements, dtype=np.int64)
    positions = dofs.element_position[elements]
    if (positions < 0).any():
        raise ValueError('nedelec_at: elements outside the dof map')
    local = np.asarray(coefficients)[dofs.element_dofs[positions]]
    mesh = dofs.mesh
    vectors, curls = nedelec_basis(dofs.order, reference_points, mesh.edge_signs[elements][:, None, :])
    geometry = element_geometry(mesh, elements, reference_points, dofs.geometry_order)
    vectors = piola_vectors(vectors, geometry.inverse)
    curls = curls / geometry.determinant[..., None]
    return np.einsum('kqnc,kn->kqc', vectors, local), np.einsum('kqn,kn->kq', curls, local)


def add_edge_terms(eta_squared, sides, terms, compliance=None) -> None:
    """eta_squared (n_elements,) += h_E x share x compliance x terms per edge (terms (k,) = int_E |jump|^2 ds, or (k, 2)
    per side), share 1/2 to each side of a two-sided edge, 1 to a one-sided one; compliance None = 1, else per
    element."""
    terms = np.asarray(terms, dtype=np.float64)
    two_sided = sides.two_sided
    for side in (0, 1):
        present = sides.elements[:, side] >= 0
        elements = sides.elements[present, side]
        side_terms = terms[present, side] if terms.ndim == 2 else terms[present]
        share = np.where(two_sided[present], 0.5, 1.0) * sides.lengths[present] * side_terms
        if compliance is not None:
            share = share * np.asarray(compliance, dtype=np.float64)[elements]
        scatter_add(eta_squared, elements, share)


# ---- scalar diffusion (fem.diffusion.FEMDiffusionSystem) ----
def diffusion_indicator(system, tensor, source_density=None) -> np.ndarray:
    """eta^2 per mesh element (float64 (n_elements,), 0 off the free active ones) of the last solve of a
    FEMDiffusionSystem with coefficient tensors (n_elements, 2, 2): h_K^2 / c_K int_K R^2 with R = f + a (u_ext - u) +
    div(C grad u), plus h_E / c_K int_E J^2 over its edges (1/2 per side of an interior edge), J = (C grad u . n) jump
    minus outline line sources; Neumann / Robin / insulated edges one-sided, Dirichlet and held edges none; c_K = the
    mean diagonal of C; r-weighted when axisymmetric. Point sources are not counted."""
    dofs = system.dofs
    if dofs is None or system.n_solves == 0:
        raise RuntimeError(f'{system.label()}: the error indicator needs a solve first')
    mesh = system.mesh
    tensor = np.asarray(tensor, dtype=np.float64)
    count = mesh.n_elements
    held = np.zeros(count, dtype=bool)
    for mask, _, _ in system.held:
        held |= mask
    free = np.zeros(count, dtype=bool)
    free[dofs.elements] = True
    free &= ~held
    diagonal = 0.5 * (tensor[:, 0, 0] + tensor[:, 1, 1])
    compliance = np.where(diagonal > 0.0, 1.0 / np.where(diagonal > 0.0, diagonal, 1.0), 0.0)
    u = system.values[dofs.global_ids]
    density = (np.zeros(count) if source_density is None
               else np.broadcast_to(np.asarray(source_density, dtype=np.float64), (count,)))
    eta_squared = np.zeros(count)
    _add_residuals(eta_squared, system, tensor, u, density, np.flatnonzero(free), compliance)
    _add_flux_jumps(eta_squared, system, tensor, u, free, held, compliance)
    return eta_squared


def _add_residuals(eta_squared, system, tensor, u, density, elements, compliance) -> None:
    """eta_squared += h_K^2 / c_K int_K R^2 (r dA axisymmetric) on the free elements."""
    mesh, dofs = system.mesh, system.dofs
    curved = (curved_elements(mesh, elements) if system.geometry_order == 2
              else np.zeros(len(elements), dtype=bool))
    diameters = element_diameters(mesh)
    for group, degree in ((~curved, RESIDUAL_DEGREE), (curved, CURVED_DEGREE)):
        chosen = elements[group]
        rule = triangle_rule(degree)
        basis, reference_gradients = lagrange_basis(dofs.order, rule.points)   # the shared rule: one basis
        for start in range(0, chosen.size, CHUNK_ITEMS):
            chunk = chosen[start:start + CHUNK_ITEMS]
            geometry = element_geometry(mesh, chunk, rule.points, dofs.geometry_order)   # values and weights alike
            local = u[dofs.element_dofs[dofs.element_position[chunk]]]
            values = np.einsum('qn,kn->kq', basis, local)
            gradients = np.einsum('kqnc,kn->kqc', physical_gradients(reference_gradients, geometry.inverse), local)
            weights = rule.weights[None, :] * geometry.determinant
            residual = (density[chunk, None] + system.loss_drive[chunk, None]
                        - system.loss_coefficient[chunk, None] * values)
            if dofs.order == 2:
                residual = residual + _divergence_p2(mesh, dofs, u, chunk, tensor)[:, None]
            if system.axisymmetric:
                radius = geometry.points[..., 0]
                flux_r = np.einsum('kc,kqc->kq', tensor[chunk, 0, :], gradients)
                residual = residual + flux_r / np.where(radius > 0.0, radius, np.inf)
                weights = weights * radius
            integral = np.sum(residual ** 2 * weights, axis=1)
            eta_squared[chunk] += diameters[chunk] ** 2 * compliance[chunk] * integral   # chunk indices are unique


def _divergence_p2(mesh, dofs, u, elements, tensor) -> np.ndarray:
    """div(C grad u) per element of a P2 field from the constant Hessians of its basis on the affine corner map (exact
    on straight elements, the chord map on curved ones)."""
    corners = mesh.points[mesh.triangles[elements]]
    jacobian = np.stack([corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]], axis=2)   # dx_c / dxi_d
    determinant = jacobian[:, 0, 0] * jacobian[:, 1, 1] - jacobian[:, 0, 1] * jacobian[:, 1, 0]
    inverse = _inverse_2x2(jacobian, determinant[:, None, None])   # closed form: no LAPACK loop over 2x2 stacks
    gradients = np.einsum('kdc,nd->knc', inverse, REFERENCE_GRADIENTS)   # grad L_n = J^-T grad_xi L_n
    local = u[dofs.element_dofs[dofs.element_position[elements]]]
    coefficient = tensor[elements]

    def contraction(first, second):   # g_a . C g_b
        return np.einsum('kc,kcd,kd->k', gradients[:, first], coefficient, gradients[:, second])

    result = np.zeros(len(elements))
    for k in range(3):   # corner L_k (2 L_k - 1): tr(C 4 g_k g_k^T); midside 4 L_a L_b: tr(C 4 (g_a g_b^T + g_b g_a^T))
        a, b = (k + 1) % 3, (k + 2) % 3
        result += 4.0 * local[:, k] * contraction(k, k) + 8.0 * local[:, 3 + k] * contraction(a, b)
    return result


def _on_edges(value, edges):
    """A condition value (number or per mesh edge array) on the listed edges."""
    value = np.asarray(value, dtype=np.float64)
    return value if value.ndim == 0 else value[edges]


def _edge_conditions(system) -> tuple:
    """Per mesh edge (code, neumann value, robin h, robin external) of the face and outline conditions (later
    sequence wins)."""
    n_edges = system.mesh.n_edges
    code = np.full(n_edges, _NONE, dtype=np.int8)
    value, h, external = np.zeros(n_edges), np.zeros(n_edges), np.zeros(n_edges)
    entries = [(sequence, system.face_edges(face), condition)
               for face, (condition, sequence) in system.face_conditions.items()]
    entries += [(sequence, edges, condition) for edges, condition, sequence in system.edge_conditions]
    for _, edges, condition in sorted(entries, key=lambda entry: entry[0]):
        if not len(edges):
            continue
        kind = {'neumann': _NEUMANN, 'robin': _ROBIN, 'dirichlet': _DIRICHLET}[condition.kind]
        code[edges] = kind
        value[edges] = condition.value if kind == _NEUMANN else 0.0
        h[edges] = _on_edges(condition.h, edges) if kind == _ROBIN else 0.0   # numbers or per mesh edge
        external[edges] = _on_edges(condition.external, edges) if kind == _ROBIN else 0.0
    return code, value, h, external


def _add_flux_jumps(eta_squared, system, tensor, u, free, held, compliance) -> None:
    """eta_squared += h_E / c_K int_E J^2 (r ds axisymmetric) on the edges of the free elements (see
    diffusion_indicator)."""
    mesh, dofs = system.mesh, system.dofs
    code, value, h, external = _edge_conditions(system)
    candidates = np.unique(mesh.element_edges[free])
    for start in range(0, candidates.size, CHUNK_ITEMS):
        sides = edge_sides(mesh, candidates[start:start + CHUNK_ITEMS], free, dofs.geometry_order)
        pairs = mesh.edge_elements[sides.edges]
        other = np.where(pairs[:, 0] == sides.elements[:, 0], pairs[:, 1], pairs[:, 0])
        next_to_held = (other >= 0) & held[np.maximum(other, 0)]
        keep = (code[sides.edges] != _DIRICHLET) & ~(~sides.two_sided & next_to_held)
        if not keep.any():
            continue
        sides = sides.select(keep)
        edges = sides.edges
        values, gradients = lagrange_at(dofs, u, sides.elements[:, 0], sides.reference_points(0))
        jump = np.einsum('kqc,kcd,kqd->kq', sides.normals, tensor[sides.elements[:, 0]], gradients)
        two_sided = sides.two_sided
        if two_sided.any():
            rows = np.flatnonzero(two_sided)
            _, other_gradients = lagrange_at(dofs, u, sides.elements[rows, 1], sides.reference_points(1, rows))
            jump[rows] -= np.einsum('kqc,kcd,kqd->kq', sides.normals[rows], tensor[sides.elements[rows, 1]],
                                    other_gradients)
        line_source = value[edges, None] + h[edges, None] * (external[edges, None] - values)
        jump = jump - line_source
        weights = sides.weights * sides.points[..., 0] if system.axisymmetric else sides.weights
        add_edge_terms(eta_squared, sides, np.sum(jump ** 2 * weights, axis=1), compliance)
