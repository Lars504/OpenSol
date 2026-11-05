"""Quadrature on the reference tetrahedron (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1): the centroid (degree 1), the
symmetric 4-point rule (degree 2), symmetric positive rules of 14 points (degree 5: requests 4-5) and 24 points (degree
6) and collapsed (Stroud conical product) Gauss-Jacobi rules for degree 3 and 7-8. Host numpy; scipy inside functions."""
# Conical product: xi = u, eta = (1 - u) v, zeta = (1 - u) (1 - v) w on the unit cube, Jacobian (1 - u)^2 (1 - v); a
# polynomial of total degree d stays degree <= d in each of u, v, w, so n = ceil((d + 1) / 2) Gauss-Jacobi points per
# direction (weights (1 - u)^2, (1 - v), 1) integrate it exactly with positive weights (n^3 points: 8 / 64 at degree 3 /
# 7-8). Not symmetric: fine for assembly, where only exactness matters. Degrees 4-6 use the fully symmetric rules below
# (S4: 14 / 24 points instead of 27 / 64; the PML-stretched P2 forms run at degree 4 / 6), checked for exactness on
# their first use.
import functools
import math
from dataclasses import dataclass

import numpy as np

TETRAHEDRON_MAX_DEGREE = 8
_SYMMETRIC_A = (5.0 - math.sqrt(5.0)) / 20.0   # degree-2 rule: barycentric (1 - 3a, a, a, a) and its permutations
# Symmetric rules as orbits of barycentric points, weights summing to the reference volume 1/6: ('aaab', a, weight) =
# the 4 points (a, a, a, 1 - 3a); ('aabb', a, weight) = the 6 points (a, a, 1/2 - a, 1/2 - a); ('aabc', (a, b),
# weight) = the 12 points (a, a, b, 1 - 2a - b).
_SYMMETRIC_RULES = {
    5: (('aaab', 0.0927352503108912264, 0.01224884051939365826),    # 14 points, positive (Walkington / Keast)
        ('aaab', 0.3108859192633006097, 0.01878132095300264180),
        ('aabb', 0.4544962958743503986, 0.007091003462846911073)),
    6: (('aaab', 0.2146028712591516840, 0.006653791709694645070),   # 24 points, positive (Keast)
        ('aaab', 0.04067395853461135312, 0.001679535175886775961),
        ('aaab', 0.3223378901422757045, 0.009226196923942399350),
        ('aabc', (0.06366100187501750110, 0.2696723314583158716), 0.008035714285714285714)),
}
_SYMMETRIC_RTOL = 1e-12   # relative monomial error a symmetric rule must stay under (checked on first use)


@dataclass(frozen=True, eq=False)
class TetrahedronRule:
    """Reference points (q, 3) = (xi, eta, zeta) and weights (q,) summing to the reference volume 1/6."""
    degree: int
    points: np.ndarray
    weights: np.ndarray


def _frozen(degree, points, weights) -> TetrahedronRule:
    points = np.ascontiguousarray(points, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    points.flags.writeable = False
    weights.flags.writeable = False
    return TetrahedronRule(degree, points, weights)


def _jacobi_on_unit_interval(count, alpha):
    """(nodes, weights) on [0, 1] of int f(u) (1 - u)^alpha du, exact for degree 2 count - 1."""
    from scipy.special import roots_jacobi
    nodes, weights = roots_jacobi(count, alpha, 0.0)
    return 0.5 * (nodes + 1.0), weights / 2.0 ** (alpha + 1)


@functools.lru_cache(maxsize=None)
def tetrahedron_rule(degree) -> TetrahedronRule:
    """Rule exact for polynomials of `degree` (1..TETRAHEDRON_MAX_DEGREE; ValueError above)."""
    degree = max(int(degree), 1)
    if degree > TETRAHEDRON_MAX_DEGREE:
        raise ValueError(f'tetrahedron quadrature: degree {degree} > {TETRAHEDRON_MAX_DEGREE} is not available')
    if degree == 1:
        return _frozen(1, [[0.25, 0.25, 0.25]], [1.0 / 6.0])
    if degree == 2:
        a, b = _SYMMETRIC_A, 1.0 - 3.0 * _SYMMETRIC_A
        points = [[a, a, a], [b, a, a], [a, b, a], [a, a, b]]   # (xi, eta, zeta) = (L1, L2, L3)
        return _frozen(2, points, [1.0 / 24.0] * 4)
    if degree in (4, 5, 6):
        return _symmetric_rule(5 if degree < 6 else 6)
    count = int(math.ceil((degree + 1) / 2.0))
    u, weight_u = _jacobi_on_unit_interval(count, 2.0)
    v, weight_v = _jacobi_on_unit_interval(count, 1.0)
    w, weight_w = _jacobi_on_unit_interval(count, 0.0)
    grid_u, grid_v, grid_w = np.meshgrid(u, v, w, indexing='ij')
    weights = (weight_u[:, None, None] * weight_v[None, :, None] * weight_w[None, None, :]).ravel()
    xi = grid_u.ravel()
    eta = ((1.0 - grid_u) * grid_v).ravel()
    zeta = ((1.0 - grid_u) * (1.0 - grid_v) * grid_w).ravel()
    return _frozen(2 * count - 1, np.column_stack([xi, eta, zeta]), weights)


def _symmetric_rule(degree) -> TetrahedronRule:
    """The _SYMMETRIC_RULES rule of `degree` as (xi, eta, zeta) = (L1, L2, L3); ValueError when a monomial of total
    degree <= degree misses its exact integral a! b! c! / (a + b + c + 3)! by more than _SYMMETRIC_RTOL."""
    import itertools
    barycentric, weights = [], []
    for orbit, values, weight in _SYMMETRIC_RULES[degree]:
        if orbit == 'aaab':
            base = (values, values, values, 1.0 - 3.0 * values)
        elif orbit == 'aabb':
            base = (values, values, 0.5 - values, 0.5 - values)
        else:
            a, b = values
            base = (a, a, b, 1.0 - 2.0 * a - b)
        for point in sorted(set(itertools.permutations(base))):
            barycentric.append(point)
            weights.append(weight)
    points = np.array(barycentric)[:, 1:]
    weights = np.array(weights)
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            for c in range(degree + 1 - a - b):
                exact = math.factorial(a) * math.factorial(b) * math.factorial(c) / math.factorial(a + b + c + 3)
                value = float(np.sum(weights * points[:, 0] ** a * points[:, 1] ** b * points[:, 2] ** c))
                if abs(value / exact - 1.0) > _SYMMETRIC_RTOL:
                    raise ValueError(f'tetrahedron quadrature: the {len(weights)}-point rule misses x^{a} y^{b} z^{c} '
                                     f'by {abs(value / exact - 1.0):.3g} (relative)')
    return _frozen(degree, points, weights)


@functools.lru_cache(maxsize=None)
def triangle_face_rule(degree) -> tuple:
    """(barycentric (q, 3) on a face, weights (q,) summing to 1): the fem.quadrature Dunavant rule (face integrals
    = area x sum)."""
    from .quadrature import triangle_rule
    rule = triangle_rule(min(max(int(degree), 1), 6))
    bary = np.column_stack([1.0 - rule.points[:, 0] - rule.points[:, 1], rule.points[:, 0], rule.points[:, 1]])
    weights = 2.0 * rule.weights
    bary.flags.writeable = False
    weights.flags.writeable = False
    return bary, weights
