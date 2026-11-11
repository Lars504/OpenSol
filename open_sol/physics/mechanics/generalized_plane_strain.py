"""Generalized plane strain of the planar 2D solid (symmetry plan 2.4, the usual form eps_zz = e0 + e1 x + e2 y): the
axial force and the two bending moments fix the free terms by superposition of unit-term homogeneous solves; every
load and condition integral is exact for bilinear u and element-constant coefficients. Backend arrays (numpy / cupy)."""
from dataclasses import dataclass

import numpy as np
from ...core.backend import array_module_of

from .system import LOCAL_NODES

# bending setting -> the free terms of eps_zz (0: uniform e0, 1: e1 x, 2: e2 y); the others are held at 0
BENDING_TERMS = {'xy': (0, 1, 2), 'x': (0, 1), 'y': (0, 2), 'none': (0,)}
BENDING_CHOICES = tuple(BENDING_TERMS)
# (power of x, power of y) of each term h_k: h_0 = 1, h_1 = x - x_ref, h_2 = y - y_ref
_TERM_POWERS = ((0, 0), (1, 0), (0, 1))
_SLOPE_SIGN = (-1.0, 1.0)            # sign of d/ds of the lower / upper 1D linear shape function
_DEFINITE_TOLERANCE = 1e-10          # smallest eigenvalue of the diagonally scaled condition matrix


class TermGeometry:
    """Window element data of the terms h_k about the reference point (the window centre): integrals of s^p per axis
    (float64) and the per-element mean of h / shape-weighted integrals of h (solve dtype) of the consistent loads."""

    def __init__(self, system):
        xp = system.xp
        x_nodes = np.concatenate(([0.0], np.cumsum(system.dx_window)))   # [m] from the window corner (= drawing x)
        y_nodes = np.concatenate(([0.0], np.cumsum(system.dy_window)))
        self.reference = (0.5 * float(x_nodes[-1]), 0.5 * float(y_nodes[-1]))
        self.xp = xp
        self.element_shape = system.element_shape
        self._axes = []
        for nodes, reference, shape in ((x_nodes, self.reference[0], (-1, 1)), (y_nodes, self.reference[1], (1, -1))):
            lower, upper = nodes[:-1] - reference, nodes[1:] - reference
            width = upper - lower
            powers = tuple(xp.asarray(((upper ** (p + 1) - lower ** (p + 1)) / (p + 1)).reshape(shape))
                           for p in range(3))   # integral of s^p over each element
            # per linear flag: (mean of h along the axis, integrals of N_lower h and N_upper h)
            moments = ((np.ones_like(width), (0.5 * width, 0.5 * width)),
                       (0.5 * (lower + upper), (width * (2.0 * lower + upper) / 6.0, width * (lower + 2.0 * upper) / 6.0)))
            weights = tuple((xp.asarray(mean.reshape(shape), dtype=system.dtype),
                             tuple(xp.asarray(node.reshape(shape), dtype=system.dtype) for node in node_moments))
                            for mean, node_moments in moments)
            self._axes.append((powers, weights))

    def integral(self, values, *terms) -> float:
        """Sum over elements of the element-constant values times the integral of the product of the terms' h."""
        xp = self.xp
        power_x = sum(_TERM_POWERS[term][0] for term in terms)
        power_y = sum(_TERM_POWERS[term][1] for term in terms)
        (powers_x, _), (powers_y, _) = self._axes
        return float(xp.sum(xp.asarray(values, dtype=xp.float64) * powers_x[power_x] * powers_y[power_y]))

    def add_load(self, load_nodes, stress_x, stress_y, stress_xy, term) -> None:
        """Add the consistent nodal forces (integral of B^T sigma h dA) of element stresses (stress_x, stress_y,
        stress_xy) [Pa] times h_term to a (nx_nodes, ny_nodes, 2) load."""
        linear_x, linear_y = _TERM_POWERS[term]
        (_, weights_x), (_, weights_y) = self._axes
        mean_x, node_x = weights_x[linear_x]
        mean_y, node_y = weights_y[linear_y]
        n_x, n_y = self.element_shape
        for k_x, k_y in LOCAL_NODES:
            along_x = _SLOPE_SIGN[k_x] * mean_x * node_y[k_y]   # integral of dN_k/dx h
            along_y = _SLOPE_SIGN[k_y] * node_x[k_x] * mean_y   # integral of dN_k/dy h
            target = load_nodes[k_x:k_x + n_x, k_y:k_y + n_y]
            target[..., 0] += stress_x * along_x + stress_xy * along_y
            target[..., 1] += stress_y * along_y + stress_xy * along_x

    def strain_field(self, coefficients):
        """(n_x, n_y) eps_zz at the element centres of the coefficients (e0, e1, e2) about the reference point."""
        (_, weights_x), (_, weights_y) = self._axes
        centre_x, centre_y = weights_x[1][0], weights_y[1][0]   # python floats keep the solve dtype
        return float(coefficients[0]) + float(coefficients[1]) * centre_x + float(coefficients[2]) * centre_y


@dataclass(frozen=True, eq=False)
class UnitStrainSolves:
    """Homogeneous solves of the free unit eps_zz terms of one assembly: flat loads f_k and solutions u_k (solve dtype),
    the condition matrix S_mk = integral c33 h_k h_m dA - u_k . f_m (float64) and the TermGeometry."""
    terms: tuple
    loads: tuple
    solutions: tuple
    matrix: np.ndarray
    geometry: TermGeometry


def _dot(first, second) -> float:
    """float64 dot product of two flat backend vectors."""
    xp = array_module_of(first)
    return float(xp.dot(first.astype(xp.float64), second.astype(xp.float64)))


def unit_strain_solves(system, coupling, c33, terms) -> UnitStrainSolves:
    """Unit-term solves K u_k = f_k (held dofs 0) of the loads f_k = -integral B^T (c13, c23, c36) h_k dA (coupling =
    the element (c13, c23, c36) [Pa]: in-plane stress per unit eps_zz) and the condition matrix; ValueError when it is
    not positive definite (no solid cells, or a section without extent along a free bending term)."""
    xp = system.xp
    geometry = TermGeometry(system)
    c13, c23, c36 = coupling
    loads, solutions = [], []
    for term in terms:
        load = xp.zeros((*system.node_shape, 2), dtype=system.dtype)
        geometry.add_load(load, -c13, -c23, -c36, term)
        solution = xp.zeros(system.n_dofs, dtype=system.dtype)
        system.solve_homogeneous(load, solution)
        loads.append(load.reshape(-1))
        solutions.append(solution)
    matrix = condition_matrix(terms, loads, solutions, lambda term_k, term_m: geometry.integral(c33, term_k, term_m),
                              _dot, system.label(), 'cells')
    return UnitStrainSolves(tuple(terms), tuple(loads), tuple(solutions), matrix, geometry)


def condition_matrix(terms, loads, solutions, integral, dot, label, pieces) -> np.ndarray:
    """S_mk = integral(term_k, term_m) (int c33 h_k h_m dA) - dot(u_k, f_m) of the unit-term solves (tensor and FEM
    solids); ValueError naming `label` when it is not positive definite (no solid `pieces`, or dependent terms)."""
    size = len(terms)
    matrix = np.empty((size, size))
    for m, term_m in enumerate(terms):
        for k, term_k in enumerate(terms):
            matrix[m, k] = integral(term_k, term_m) - dot(solutions[k], loads[m])
    symmetric = 0.5 * (matrix + matrix.T)
    diagonal = np.diag(symmetric)
    if not np.all(diagonal > 0.0):
        raise ValueError(f'{label}: generalized plane strain needs solid {pieces} (axial stiffness per unit eps_zz term '
                         f'{diagonal.tolist()})')
    scale = 1.0 / np.sqrt(diagonal)
    if float(np.linalg.eigvalsh(symmetric * scale[:, None] * scale[None, :])[0]) <= _DEFINITE_TOLERANCE:
        raise ValueError(f'{label}: the generalized plane-strain terms {terms} (0 uniform, 1 along x, 2 along y) are '
                         'not independent on this section: set bending to fewer terms')
    return matrix


def strain_term_values(matrix, terms, loads, targets, thermal_integral, unknown_dot) -> np.ndarray:
    """The unit-term amplitudes e_k: S e = target_m + int beta_3 dT h_m dA (thermal_integral(term)) + u_0 . f_m
    (unknown_dot(load))."""
    right_hand = np.array([targets[term] + thermal_integral(term) + unknown_dot(load)
                           for term, load in zip(terms, loads)])
    return np.linalg.solve(matrix, right_hand)


def condition_targets(geometry, axial_force, moment_x, moment_y):
    """(integral sigma_zz h_k dA for k = 0, 1, 2) of F [N] and the moments about the origin (x = y = 0 of the drawing):
    M_x = integral y sigma_zz dA, M_y = -integral x sigma_zz dA [N m]."""
    x_ref, y_ref = geometry.reference
    return (axial_force, -moment_y - x_ref * axial_force, moment_x - y_ref * axial_force)


def solve_strain_terms(system, solves, thermal_stress_z, axial_force, moment_x, moment_y):
    """Superpose u = u_0 + sum e_k u_k on system.unknown (u_0 = its current eps_zz = 0 solve) and e_k f_k on
    system.load_full so the conditions hold (thermal_stress_z = beta_3 dT per element); returns the (3,) coefficients
    about the reference point (held terms 0)."""
    geometry = solves.geometry
    targets = condition_targets(geometry, axial_force, moment_x, moment_y)
    values = strain_term_values(solves.matrix, solves.terms, solves.loads, targets,
                                lambda term: geometry.integral(thermal_stress_z, term),
                                lambda load: _dot(system.unknown, load))
    coefficients = np.zeros(3)
    for term, value, load, solution in zip(solves.terms, values, solves.loads, solves.solutions):
        coefficients[term] = value
        system.unknown += system.dtype(value) * solution
        system.load_full = system.load_full + system.dtype(value) * load
    return coefficients


def terms_about_origin(geometry, coefficients):
    """(e0, e1, e2) of eps_zz = e0 + e1 x + e2 y with x, y from the window corner (the drawing coordinates)."""
    x_ref, y_ref = geometry.reference
    return (float(coefficients[0] - coefficients[1] * x_ref - coefficients[2] * y_ref), float(coefficients[1]),
            float(coefficients[2]))
