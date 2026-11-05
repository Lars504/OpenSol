"""Quadrature of the FEM path: Dunavant rules of degree 2 / 4 / 5 / 6 on the reference triangle (0, 0), (1, 0), (0, 1),
their s x s subdivisions (transfers), and Gauss-Legendre on [0, 1] (edges). Host numpy."""
import functools
import math
from dataclasses import dataclass

import numpy as np

TRIANGLE_DEGREES = (2, 4, 5, 6)   # available Dunavant rules
CURVED_DEGREE = 6              # rule on elements with a curved (isoparametric) edge

# Dunavant orbits: ('s3', weight) = the centroid; ('s21', a, weight) = barycentric (a, a, 1 - 2a) and its 3 permutations; ('s111', a, b, weight) =
# (a, b, 1 - a - b) and its 6 permutations; weights sum to 1 over the orbit points
_DUNAVANT = {
    2: (('s21', 1.0 / 6.0, 1.0 / 3.0),),
    4: (('s21', 0.44594849091596488632, 0.22338158967801146570),
        ('s21', 0.09157621350977074346, 0.10995174365532186764)),
    5: (('s3', 0.225),   # 7 points, positive weights (axisymmetric P2 mass, PML stiffness at degree 3 + 2)
        ('s21', 0.47014206410511508977, 0.13239415278850618074),
        ('s21', 0.10128650732345633880, 0.12593918054482715260)),
    6: (('s21', 0.24928674517091042129, 0.11678627572637936603),
        ('s21', 0.06308901449150222834, 0.05084490637020681692),
        ('s111', 0.05314504984481694735, 0.31035245103378440542, 0.08285107561837357519)),
}


@dataclass(frozen=True, eq=False)
class TriangleRule:
    """Reference points (q, 2) = (xi, eta) and weights (q,) summing to the reference area 1/2."""
    degree: int
    points: np.ndarray
    weights: np.ndarray


@functools.lru_cache(maxsize=None)
def triangle_rule(degree) -> TriangleRule:
    """Smallest Dunavant rule integrating polynomials of `degree` exactly (2, 4, 5 or 6; ValueError above 6)."""
    chosen = next((available for available in TRIANGLE_DEGREES if available >= max(int(degree), 1)), None)
    if chosen is None:
        raise ValueError(f'triangle quadrature: degree {degree} > {TRIANGLE_DEGREES[-1]} is not available')
    barycentric, weights = [], []
    for orbit in _DUNAVANT[chosen]:
        if orbit[0] == 's3':
            _, weight = orbit
            triples = [(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)]
        elif orbit[0] == 's21':
            _, a, weight = orbit
            triples = [(a, a, 1.0 - 2.0 * a), (a, 1.0 - 2.0 * a, a), (1.0 - 2.0 * a, a, a)]
        else:
            _, a, b, weight = orbit
            c = 1.0 - a - b
            triples = [(a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a)]
        barycentric += triples
        weights += [weight] * len(triples)
    barycentric = np.array(barycentric, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    weights = 0.5 * weights / weights.sum()
    points = np.ascontiguousarray(barycentric[:, 1:])   # (xi, eta) = (L1, L2)
    points.flags.writeable = False
    weights.flags.writeable = False
    return TriangleRule(chosen, points, weights)


@functools.lru_cache(maxsize=None)
def subdivided_rule(degree, subdivisions) -> TriangleRule:
    """The degree rule on each of the subdivisions^2 congruent sub-triangles of the reference triangle (dense sample
    points with the same weight sum 1/2)."""
    base = triangle_rule(degree)
    count = int(subdivisions)
    if count < 1:
        raise ValueError(f'subdivided_rule: subdivisions must be >= 1, got {subdivisions!r}')
    if count == 1:
        return base
    origins, first_axes, second_axes = [], [], []
    for i in range(count):
        for j in range(count - i):
            origins.append((i, j))
            first_axes.append((1.0, 0.0))
            second_axes.append((0.0, 1.0))
            if i + j < count - 1:   # the downward sub-triangle (i + 1, j), (i + 1, j + 1), (i, j + 1)
                origins.append((i + 1, j))
                first_axes.append((0.0, 1.0))
                second_axes.append((-1.0, 1.0))
    origins = np.array(origins, dtype=np.float64)[:, None, :] / count
    first_axes = np.array(first_axes)[:, None, :] / count
    second_axes = np.array(second_axes)[:, None, :] / count
    points = (origins + base.points[None, :, 0:1] * first_axes + base.points[None, :, 1:2] * second_axes)
    weights = np.broadcast_to(base.weights / count ** 2, points.shape[:2])
    points = np.ascontiguousarray(points.reshape(-1, 2))
    weights = np.ascontiguousarray(weights.reshape(-1))
    points.flags.writeable = False
    weights.flags.writeable = False
    return TriangleRule(base.degree, points, weights)


@functools.lru_cache(maxsize=None)
def gauss_line(degree) -> tuple:
    """(t (q,), weights (q,)) of the Gauss-Legendre rule on [0, 1] exact for polynomials of `degree` (weights sum 1)."""
    count = max(1, int(math.ceil((int(degree) + 1) / 2.0)))
    nodes, weights = np.polynomial.legendre.leggauss(count)
    t = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    t.flags.writeable = False
    weights.flags.writeable = False
    return t, weights
