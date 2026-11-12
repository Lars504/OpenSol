"""Per-element total-Lagrangian force and tangent of planar P1 / P2 triangles (the 2D twin of fem.tangent_kernel3d): one
core.kernels host twin 'fem.element_tangent_2d' = a numba prange over the elements, with the batched numpy formulas of
fem.elasticity as the reference twin. Host only (the 2D tangent is a host CSR); numpy at module level."""
# Arguments (every array C-contiguous float64): gradients (m, q, n, 2) physical basis gradients at the rule points,
# weights (m, q), local (m, n, 2) element displacements, material (m, 3, 3) C, fixed (m, q, 3) S0 or a (1, 1, 1) dummy,
# outputs force (m, 2 n) and blocks (m, 2 n, 2 n) (a (1, 1, 1) dummy when want_tangent is 0), then n_elements,
# has_fixed, want_tangent. Per rule point: H = grad u, E = Green-Lagrange (Voigt xx, yy, engineering shear), S = C E +
# S0, B + B_NL rows = G_a,j F_ci (+ G_a,i F_cj on the shear row), F = I + H; force = sum w (B + B_NL)^T S, tangent = sum
# w (B + B_NL)^T C (B + B_NL) + (sum w grad N_a . S grad N_b) x I2 (lower triangle computed, mirrored). Axisymmetric
# (hoop row) and azimuthal forms keep the numpy formulas (TangentAssembler).
import sys

import numpy as np

from ..core.kernels import host_twin

# numba runs the kernel when the process already loaded numba or the call has at least this many elements (the first
# parallel numba call of a process costs ~0.2 s, a first compile ~0.5 s: 3D speed plan N6 / lessons)
NUMBA_MIN_ELEMENTS = 20000
_VOIGT_FIRST = np.array([0, 1, 0], dtype=np.int64)    # xx, yy, g = (x, y)
_VOIGT_SECOND = np.array([0, 1, 1], dtype=np.int64)


def _element_tangent_host(gradients, weights, local, material, fixed, force, blocks, n_elements, has_fixed,
                          want_tangent):
    """numpy twin: the batched fem.elasticity formulas (planar, points unused there)."""
    from .elasticity import _green_lagrange_group, _tangent_blocks
    m, q, n, _ = gradients.shape
    basis = np.zeros((q, n))   # read only by the axisymmetric hoop terms
    points = np.ones((m, q, 2))
    operator, stress = _green_lagrange_group(basis, gradients, points, local, material,
                                             fixed if int(has_fixed) else None, False)
    force[...] = np.einsum('mqki,mqk,mq->mi', operator, stress, weights)
    if int(want_tangent):
        blocks[...] = _tangent_blocks(basis, gradients, points, weights, operator, stress, material, False)


def _element_tangent_numba(numba):
    """numba twin: prange over the elements, per-element arrays (Voigt tables: module globals, frozen at compile)."""
    @numba.njit(parallel=True, cache=True)
    def kernel(gradients, weights, local, material, fixed, force, blocks, n_elements, n_points, n_local, has_fixed,
               want_tangent):
        columns = 2 * n_local
        for prange_e in numba.prange(n_elements):
            e = np.int64(prange_e)
            gradient = np.empty(4)
            strain = np.empty(3)
            stress = np.empty((n_points, 3))
            operator = np.empty((n_points, 3, columns))
            for q in range(n_points):
                base = (e * n_points + q) * n_local * 2
                for c in range(2):
                    for d in range(2):
                        acc = 0.0
                        for a in range(n_local):
                            acc += gradients[base + 2 * a + d] * local[(e * n_local + a) * 2 + c]
                        gradient[2 * c + d] = acc
                for k in range(3):
                    i, j = _VOIGT_FIRST[k], _VOIGT_SECOND[k]
                    quadratic = gradient[i] * gradient[j] + gradient[2 + i] * gradient[2 + j]
                    if i == j:
                        strain[k] = gradient[3 * i] + 0.5 * quadratic
                    else:
                        strain[k] = gradient[2 * i + j] + gradient[2 * j + i] + quadratic
                for k in range(3):
                    value = 0.0
                    for l in range(3):
                        value += material[e * 9 + 3 * k + l] * strain[l]
                    if has_fixed:
                        value += fixed[(e * n_points + q) * 3 + k]
                    stress[q, k] = value
                for k in range(3):
                    i, j = _VOIGT_FIRST[k], _VOIGT_SECOND[k]
                    for a in range(n_local):
                        gi, gj = gradients[base + 2 * a + i], gradients[base + 2 * a + j]
                        for c in range(2):
                            fci = gradient[2 * c + i] + (1.0 if c == i else 0.0)
                            if i == j:
                                operator[q, k, 2 * a + c] = gi * fci
                            else:
                                fcj = gradient[2 * c + j] + (1.0 if c == j else 0.0)
                                operator[q, k, 2 * a + c] = gj * fci + gi * fcj
            for r in range(columns):
                total = 0.0
                for q in range(n_points):
                    part = 0.0
                    for k in range(3):
                        part += operator[q, k, r] * stress[q, k]
                    total += weights[e * n_points + q] * part
                force[e * columns + r] = total
            if want_tangent:
                weighted = np.empty((n_points, 3, columns))
                for q in range(n_points):
                    w = weights[e * n_points + q]
                    for k in range(3):
                        for s in range(columns):
                            value = 0.0
                            for l in range(3):
                                value += material[e * 9 + 3 * k + l] * operator[q, l, s]
                            weighted[q, k, s] = w * value
                geometric = np.empty((n_local, n_local))
                for a in range(n_local):
                    for b in range(n_local):
                        total = 0.0
                        for q in range(n_points):
                            base = (e * n_points + q) * n_local * 2
                            ga_x, ga_y = gradients[base + 2 * a], gradients[base + 2 * a + 1]
                            gb_x, gb_y = gradients[base + 2 * b], gradients[base + 2 * b + 1]
                            part = (stress[q, 0] * ga_x * gb_x + stress[q, 1] * ga_y * gb_y
                                    + stress[q, 2] * (ga_x * gb_y + ga_y * gb_x))
                            total += weights[e * n_points + q] * part
                        geometric[a, b] = total
                out = e * columns * columns
                for r in range(columns):
                    for s in range(r + 1):
                        total = 0.0
                        for q in range(n_points):
                            for k in range(3):
                                total += operator[q, k, r] * weighted[q, k, s]
                        if r % 2 == s % 2:
                            total += geometric[r // 2, s // 2]
                        blocks[out + r * columns + s] = total
                        blocks[out + s * columns + r] = total

    def element_tangent_2d(gradients, weights, local, material, fixed, force, blocks, n_elements, has_fixed,
                           want_tangent):
        _, points, n_local, _ = gradients.shape
        kernel(gradients.reshape(-1), weights.reshape(-1), local.reshape(-1), material.reshape(-1), fixed.reshape(-1),
               force.reshape(-1), blocks.reshape(-1), int(n_elements), int(points), int(n_local), int(has_fixed),
               int(want_tangent))
    return element_tangent_2d


ELEMENT_TANGENT_2D = host_twin('fem.element_tangent_2d', _element_tangent_host, _element_tangent_numba)


def use_kernel(n_elements) -> bool:
    """True when the numba twin should run a call of n_elements: numba already imported by this process (cpu runs of
    numba twins; checked before host_kind, which imports it) or the call large enough to pay its first-call cost, and
    the twin resolves to numba."""
    if not ('numba' in sys.modules or int(n_elements) >= NUMBA_MIN_ELEMENTS):
        return False
    return ELEMENT_TANGENT_2D.host_kind == 'numba'


def element_force_tangent_2d(gradients, weights, local, material, fixed_stress=None, want_tangent=True):
    """(element forces (m, 2 n), element tangents (m, 2 n, 2 n) or None) of one planar group: the numba twin (when
    use_kernel) or the batched numpy formulas (TangentAssembler calls it only when use_kernel)."""
    m, points, n_local, _ = gradients.shape
    columns = 2 * n_local
    force = np.empty((m, columns))
    blocks = np.empty((m, columns, columns)) if want_tangent else np.empty((1, 1, 1))
    fixed = np.empty((1, 1, 1)) if fixed_stress is None else np.ascontiguousarray(fixed_stress, dtype=np.float64)
    args = (np.ascontiguousarray(gradients, dtype=np.float64), np.ascontiguousarray(weights, dtype=np.float64),
            np.ascontiguousarray(local, dtype=np.float64), np.ascontiguousarray(material, dtype=np.float64), fixed,
            force, blocks, np.int32(m), np.int32(fixed_stress is not None), np.int32(bool(want_tangent)))
    if m:
        if use_kernel(m):
            ELEMENT_TANGENT_2D(*args)
        else:
            _element_tangent_host(*args)
    return force, (blocks if want_tangent else None)
