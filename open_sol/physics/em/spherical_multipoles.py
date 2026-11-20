"""Spherically symmetric frequency-domain EM (em_fdfd_sphere): the radial equation of one multipole term (family 'er' /
'hr', order l) on the radial cells of a 1D spherical domain, its exact outgoing-wave boundary, plane-wave (Mie) and
dipole sources, field recovery and the resonance search; host numpy, scipy inside functions only."""
# Fields ~ exp(+j omega t). Vector spherical harmonics with unit mean square over the sphere: Y^ = sqrt(4 pi) Y_lm,
# Psi^ = sqrt(4 pi / L) r grad Y_lm, Phi^ = r^ x Psi^, L = l (l + 1), so the sphere mean of |F|^2 of a sum of terms is
# the sum of their |radial functions|^2 (different (l, m, family) are orthogonal on the sphere) and a field integral
# over the whole-shell metric volumes 4 pi r^2 dr of |E|^2 radial functions is the 3D integral. Curls:
#   curl(f Y^ r^) = -(sqrt(L) / r) f Phi^,  curl(f Psi^) = (1 / r)(r f)' Phi^,
#   curl(f Phi^)  = -(sqrt(L) / r) f Y^ r^ - (1 / r)(r f)' Psi^.
# Families (named by the radial field that exists, as ez / hz name the out-of-plane one):
#   'hr' (magnetic multipoles, Mie b_l, E_r = 0): E = E_t Phi^, y = r E_t; H_r = sqrt(L) y / (j w mu0 mu r^2),
#        H_t (along Psi^) = y' / (j w mu0 mu r);  p = q = 1 / mu, m = eps.
#   'er' (electric multipoles, Mie a_l, H_r = 0): H = H_t Phi^, y = r H_t; E_r = -sqrt(L) y / (j w eps0 eps r^2),
#        E_t (along Psi^) = -y' / (j w eps0 eps r);  p = q = 1 / eps, m = mu.
# Both: -(p y')' + L q y / r^2 = k0^2 m y, the 3D weak form divided by 4 pi (curl-curl of E for 'hr', of H for 'er').
# Discretisation: y on the radial cell edges (nodes) r_n, P1 between them: stiffness sum_cells (p_i / h_i)(y_i+1 -
# y_i)^2 with p the cell value, the L q y^2 / r^2 term and the mass lumped on the dual cells (half cells on each side;
# dual integrals of q and m: the normal B_r / D_r sees the mean of 1 / mu or 1 / eps, the tangential fields the mean
# of eps or mu), 1 / r_n^2 at the node (the second-order three-point scheme). Complex symmetric tridiagonal operator
# A = K - k0^2 M on the free nodes. Second order in h.
# Boundaries: the sphere centre (radius_min 0) holds y = 0 (y ~ r^(l+1)). pec: tangential E = 0, i.e. y = 0 for 'hr'
# and the natural end (1 / eps) y' = 0 for 'er'; pmc the other way round. A pec cell: 'hr' holds y = 0 on its nodes,
# 'er' drops it (p = q = m = 0 there; a node inside metal leaves the unknowns, a metal surface node is the natural
# end). 'radiation' (outer face): the last cell's medium continues to infinity with the exact outgoing wave y ~
# xi_l(k r), xi = x h_l^(2)(x) (the Riccati-Hankel function of exp(+j omega t)): the weak form's boundary term p y' v
# at R becomes K_NN += B = -p k D(k R), D = xi' / xi. No PML and no reflection at any l; for an incident regular wave
# alpha psi_l(k r) (psi = x j_l(x)) the boundary also gets the source j p k alpha / xi(k R) (Wronskian psi' xi - psi
# xi' = j). Plane wave E0 (x-polarised, along +z): per order the regular wave amplitude alpha = E0 sqrt((2 l + 1) / 2)
# / k ('hr') or / (eta k) ('er'), both m = +-1 terms in one solve (their sphere-RMS sum); the outgoing coefficient c =
# (y_N - alpha psi) / xi gives the Mie coefficient -c / alpha (the complex conjugate of Bohren & Huffman's a_l / b_l:
# exp(+j omega t)) and C_sca = 2 pi / k^2 (2 l + 1) |a|^2, C_ext = 2 pi / k^2 (2 l + 1) Re a per family.
# Dipole I d [A m] at radius r_s (on the polar axis): radial couples to 'er' m = 0 (sum_m |Y^|^2 = 2 l + 1), tangential
# (along the plane-wave polarisation) to both families m = +-1 (sum_m |Psi^ . t|^2 = sum_m |Phi^ . t|^2 = (2 l + 1) /
# 2); one solve per family and orientation with the coupling g = sqrt of that sum is the sphere-RMS sum of its m terms.
# Weak right-hand sides: 'hr' -j w mu0 I d g v(r_s) / (4 pi r_s); 'er' (I d / (4 pi eps_s)) l(v), l(v) = -g sqrt(L)
# v(r_s) / r_s^2 (radial) or -g v'(r_s) / r_s (tangential), v(r_s) by the P1 interpolation of the cell holding r_s,
# v'(r_s) by the linear interpolation of the two nearest cell slopes (three nodes: second order anywhere, the central
# difference on a node; the one-cell slope would be first order). Emitted power -1/2 Re(E . (I d)*) with E . n = g y(r_s) / r_s ('hr') or l(y) / (j w eps0 eps_s) ('er', the
# self-field -J / (j w eps0 eps) dropped: reactive in a lossless source cell); for a lossless model it equals the
# outgoing power 2 pi Im(y y'*) / (w mu0 mu) ('hr') or / (w eps0 eps) ('er') exactly (the discrete Poynting theorem of
# the complex symmetric A).
# Resonances: shift-invert about k_t^2 of the pencil with the radiation term frozen at k_t (the eigen role), then each
# candidate refined with the exact radiation term at its own complex k by Rayleigh-functional inverse iteration
# (x^T T(k) x = 0 for k, x <- T(k)^-1 T'(k) x; T complex symmetric: cubic convergence); materials stay frozen at the
# target (sigma_e, poles), as in the other EM eigen-solves.
import math
import time
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, EPS0, ETA0, MU0

SPHERE_FAMILIES = ('er', 'hr')
FAMILY_TEXTS = {'er': 'electric', 'hr': 'magnetic'}
INNER_FACES = ('centre', 'pec', 'pmc')
OUTER_FACES = ('radiation', 'pec', 'pmc')
DIPOLE_ORIENTATIONS = ('radial', 'tangential')
SPHERE_FIELD_COMPONENTS = ('er', 'et', 'hr', 'ht')   # sphere-RMS radial functions of E_r, E_t, H_r, H_t
COMPONENT_UNITS = {'er': 'V/m', 'et': 'V/m', 'hr': 'A/m', 'ht': 'A/m'}
LOSSLESS_RELATIVE_IMAG = 1e-12    # |Im| below this times |Re| of eps / mu: lossless (plane-wave cross sections)
REFINE_MAX_ITERATIONS = 40        # Rayleigh-functional inverse iterations per resonance candidate
REFINE_TOLERANCE = 1e-12          # relative k change that ends the refinement
REFINE_RESIDUAL_LIMIT = 1e-6      # relative residual |T x| / (|T|_diag |x|) above which a refined candidate is dropped
DUPLICATE_RELATIVE = 1e-8         # candidates refined to within this relative k are one resonance
_SCALAR_NEWTON_STEPS = 30
_START_VECTOR_SEED = 6            # deterministic ARPACK start vector


@dataclass(frozen=True, eq=False)
class SphereGrid:
    """Radial cells of a 1D spherical domain: node radii r_n [m] (n + 1,), from radius_min outward."""
    edges: np.ndarray

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def centres(self) -> np.ndarray:
        return 0.5 * (self.edges[:-1] + self.edges[1:])

    @property
    def n_cells(self) -> int:
        return int(self.edges.size - 1)


def sphere_grid(radius_min, widths) -> SphereGrid:
    """SphereGrid of the radial cell widths [m] from radius_min [m] (>= 0); ValueError for non-positive widths."""
    widths = np.asarray(widths, dtype=np.float64).ravel()
    if widths.size < 2 or not np.all(widths > 0.0):
        raise ValueError(f'a spherical grid needs >= 2 radial cells of positive width, got {widths.size}')
    if not (math.isfinite(radius_min) and radius_min >= 0.0):
        raise ValueError(f'radius_min must be finite and >= 0, got {radius_min!r}')
    edges = float(radius_min) + np.concatenate(([0.0], np.cumsum(widths)))
    edges[0] = float(radius_min)
    return SphereGrid(edges=edges)


@dataclass(frozen=True, eq=False)
class SphereMaterials:
    """Complex relative permittivity / permeability per radial cell at one omega (loss and poles included) and the
    host bool PEC cells."""
    eps: np.ndarray
    mu: np.ndarray
    pec: np.ndarray


# ---- Riccati-Bessel functions ----
def riccati_hankel(order, x):
    """(xi, D = xi' / xi, dD/dx) of the outgoing Riccati-Hankel function xi_l(x) = x h_l^(2)(x) (exp(+j omega t): xi ~
    j^(l+1) exp(-j x)), x complex != 0, l >= 1: upward recurrence of the ratio R_l = xi_(l-1) / xi_l (stable for the
    dominant xi), log xi summed (xi may overflow to inf for x << l: then 1 / xi = 0). dD/dx = L / x^2 - 1 - D^2 (the
    Riccati-Bessel equation)."""
    order = int(order)
    if order < 1:
        raise ValueError(f'multipole order must be >= 1, got {order}')
    x = complex(x)
    if x == 0:
        raise ValueError('riccati_hankel needs x != 0')
    log_xi = complex(np.log(1j * np.exp(-1j * x)))          # xi_0 = j exp(-j x)
    ratio = -1j * x / (x - 1j)                                # R_1 = xi_0 / xi_1, xi_1 = -exp(-j x)(1 - j / x)
    for level in range(1, order + 1):
        log_xi -= complex(np.log(ratio))
        if level < order:
            ratio = 1.0 / ((2 * level + 1) / x - ratio)
    log_derivative = ratio - order / x
    with np.errstate(over='ignore'):
        xi = complex(np.exp(log_xi)) if log_xi.real < 700.0 else complex(math.inf, 0.0)
    derivative_of_d = order * (order + 1) / x ** 2 - 1.0 - log_derivative ** 2
    return xi, log_derivative, derivative_of_d


def riccati_bessel(order, x):
    """(psi, psi') of the regular Riccati-Bessel function psi_l(x) = x j_l(x) (scipy spherical_jn; x real or
    complex)."""
    from scipy.special import spherical_jn
    j_value = spherical_jn(int(order), x)
    j_derivative = spherical_jn(int(order), x, derivative=True)
    return complex(x * j_value), complex(j_value + x * j_derivative)


# ---- assembly ----
@dataclass(frozen=True, eq=False)
class SphereSystem:
    """The radial operator of one multipole term at omega: stiffness (radiation term included) / base_stiffness
    (without) / mass on the free nodes (scipy CSR complex128), the free node indices, the radiation boundary (position
    of the outer node among the free ones, its p and index of the exterior medium) or None."""
    grid: SphereGrid
    materials: SphereMaterials
    family: str
    order: int
    omega: float
    free: np.ndarray
    base_stiffness: object
    stiffness: object
    mass: object
    boundary_row: int | None
    exterior: tuple | None    # (p_ext, eps_ext, mu_ext) of the radiation boundary, else None

    @property
    def angular(self) -> float:
        """L = l (l + 1)."""
        return float(self.order * (self.order + 1))

    @property
    def k0(self) -> float:
        return self.omega / C0

    @property
    def radiation(self) -> bool:
        return self.exterior is not None

    @property
    def n_unknowns(self) -> int:
        return int(self.free.size)

    @property
    def operator(self):
        """A = K - k0^2 M (radiation term at omega)."""
        return (self.stiffness - (self.k0 ** 2) * self.mass).tocsr()

    def exterior_index(self) -> complex:
        """Refractive index sqrt(eps_ext mu_ext) of the exterior medium (Re >= 0, Im <= 0 when lossy)."""
        _, eps_ext, mu_ext = self.exterior
        return complex(np.sqrt(complex(eps_ext) * complex(mu_ext)))

    def radiation_term(self, k):
        """(B, dB/dk) of the outgoing-wave term K_NN += B = -p_ext k_e D(k_e R), k_e = k n_ext, at a complex vacuum
        wave number k [1/m]."""
        p_ext = self.exterior[0]
        index = self.exterior_index()
        x = complex(k) * index * float(self.grid.edges[-1])
        _, log_derivative, derivative_of_d = riccati_hankel(self.order, x)
        value = -p_ext * complex(k) * index * log_derivative
        derivative = -p_ext * index * (log_derivative + x * derivative_of_d)
        return value, derivative

    def operator_at(self, k):
        """T(k) = K_base + B(k) e_N e_N^T - k^2 M (materials frozen at omega) at a complex vacuum wave number k."""
        import scipy.sparse
        operator = self.base_stiffness - (complex(k) ** 2) * self.mass
        if self.radiation:
            value, _ = self.radiation_term(k)
            row = self.boundary_row
            operator = operator + scipy.sparse.csr_matrix(([value], ([row], [row])), shape=operator.shape)
        return operator.tocsr()

    def full(self, values) -> np.ndarray:
        """complex128 (n + 1,) node values of a free-node vector (held nodes 0)."""
        result = np.zeros(self.grid.edges.size, dtype=np.complex128)
        result[self.free] = values
        return result

    def restricted(self, node_values) -> np.ndarray:
        """Free-node part of a (n + 1,) node vector."""
        return np.asarray(node_values, dtype=np.complex128)[self.free]


def _family_coefficients(family, materials):
    """(p, q, m) per cell of a family ('hr': 1 / mu, 1 / mu, eps; 'er': 1 / eps, 1 / eps, mu), 0 in PEC cells."""
    eps = np.asarray(materials.eps, dtype=np.complex128)
    mu = np.asarray(materials.mu, dtype=np.complex128)
    pec = np.asarray(materials.pec, dtype=bool)
    if family == 'hr':
        p, m = 1.0 / np.where(pec, 1.0, mu), eps
    elif family == 'er':
        p, m = 1.0 / np.where(pec, 1.0, eps), mu
    else:
        raise ValueError(f"multipole family must be one of {', '.join(SPHERE_FAMILIES)}, got {family!r}")
    zero = np.zeros_like(p)
    return np.where(pec, zero, p), np.where(pec, zero, p), np.where(pec, zero, m)


def assemble_sphere(grid, materials, family, order, omega, inner='centre', outer='radiation') -> SphereSystem:
    """SphereSystem of one term (family, order l >= 1) at omega [rad/s] with the inner / outer face conditions
    (INNER_FACES / OUTER_FACES; 'centre' needs radius_min 0 and is required there); ValueError for a radiation face on a
    PEC last cell, a centre at radius_min > 0 or no free node."""
    import scipy.sparse
    order = int(order)
    if order < 1:
        raise ValueError(f'multipole order must be >= 1, got {order}')
    if inner not in INNER_FACES or outer not in OUTER_FACES:
        raise ValueError(f'inner face must be one of {INNER_FACES}, outer one of {OUTER_FACES}; got {inner!r}, '
                         f'{outer!r}')
    omega = float(omega)
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f'omega must be finite and > 0 [rad/s], got {omega!r}')
    edges = grid.edges
    n_cells = grid.n_cells
    if (inner == 'centre') != (edges[0] == 0.0):
        raise ValueError(f"inner face '{inner}' at radius_min = {edges[0]:.6g} m: the sphere centre (radius_min 0) "
                         "takes 'centre' and only it; a shell (radius_min > 0) takes pec or pmc")
    pec = np.asarray(materials.pec, dtype=bool).ravel()
    if pec.size != n_cells or np.asarray(materials.eps).size != n_cells or np.asarray(materials.mu).size != n_cells:
        raise ValueError(f'materials must hold one value per radial cell ({n_cells})')
    widths = grid.widths
    p, q, m = _family_coefficients(family, materials)
    half = 0.5 * widths
    q_dual = np.zeros(n_cells + 1, dtype=np.complex128)
    m_dual = np.zeros(n_cells + 1, dtype=np.complex128)
    measure = np.zeros(n_cells + 1, dtype=np.float64)
    for target, values in ((q_dual, half * q), (m_dual, half * m), (measure, np.where(pec, 0.0, half))):
        target[:-1] += values
        target[1:] += values
    held = np.zeros(n_cells + 1, dtype=bool)
    if family == 'hr':   # tangential E = 0 on every metal surface
        held[:-1] |= pec
        held[1:] |= pec
    else:                # nodes inside metal leave the unknowns; metal surfaces are the natural end
        held |= measure == 0.0
    if inner == 'centre' or (inner == 'pec') == (family == 'hr'):
        held[0] = True
    exterior = None
    if outer == 'radiation':
        if pec[-1]:
            raise ValueError("outer face 'radiation': the last radial cell is PEC (pec_domain); the exterior medium "
                             'continues the last cell to infinity, so use bc_x_hi pec for a metal wall')
        eps_ext, mu_ext = complex(materials.eps[-1]), complex(materials.mu[-1])
        exterior = ((1.0 / mu_ext) if family == 'hr' else (1.0 / eps_ext), eps_ext, mu_ext)
    elif (outer == 'pec') == (family == 'hr'):
        held[-1] = True
    free = np.flatnonzero(~held)
    if free.size == 0:
        raise ValueError(f'multipole term {family} l = {order}: every node is held (all cells PEC?)')
    angular = float(order * (order + 1))
    with np.errstate(divide='ignore', invalid='ignore'):
        node_term = np.where(edges > 0.0, angular * q_dual / edges ** 2, 0.0)
    coupling = p / widths
    diagonal = node_term.astype(np.complex128)
    diagonal[:-1] += coupling
    diagonal[1:] += coupling
    base = scipy.sparse.diags([-coupling, diagonal, -coupling], [-1, 0, 1], format='csr')
    base = base[free][:, free].tocsr()
    mass = scipy.sparse.diags(m_dual[free], 0, format='csr').astype(np.complex128)
    boundary_row = None
    stiffness = base
    if exterior is not None:
        boundary_row = int(np.flatnonzero(free == n_cells)[0])
        system = SphereSystem(grid, materials, family, order, omega, free, base, base, mass, boundary_row, exterior)
        value, _ = system.radiation_term(omega / C0)
        stiffness = (base + scipy.sparse.csr_matrix(([value], ([boundary_row], [boundary_row])),
                                                    shape=base.shape)).tocsr()
    return SphereSystem(grid, materials, family, order, omega, free, base, stiffness, mass, boundary_row, exterior)


# ---- sources ----
@dataclass(frozen=True, eq=False)
class _SourcePoint:
    """A point at radius [m]: the cell holding it, P1 value weights on its two nodes, and second-order y' weights on
    three nodes (the two nearest cell slopes, which live at the cell midpoints, interpolated linearly to the radius)."""
    cell: int
    value_nodes: tuple
    value_weights: tuple
    slope_nodes: tuple
    slope_weights: tuple

    def value(self, node_values) -> complex:
        return sum(weight * node_values[node] for node, weight in zip(self.value_nodes, self.value_weights))

    def slope(self, node_values) -> complex:
        return sum(weight * node_values[node] for node, weight in zip(self.slope_nodes, self.slope_weights))


def _source_point(grid, radius, label) -> _SourcePoint:
    """_SourcePoint of a point at radius [m] inside the grid. The one-cell P1 slope would sit the source at that cell's
    midpoint, a first-order error h / (2 r_s) in the tangential 'er' coupling (2e-3 at 1 nm, r_s 0.2 um)."""
    edges = grid.edges
    if not edges[0] < radius <= edges[-1] or radius <= 0.0:
        raise ValueError(f'{label}: radius {radius:.6g} m must lie in ({edges[0]:.6g}, {edges[-1]:.6g}] m and be > 0')
    cell = int(np.clip(np.searchsorted(edges, radius, side='right') - 1, 0, grid.n_cells - 1))
    widths = grid.widths
    upper = (radius - edges[cell]) / widths[cell]
    midpoints = grid.centres
    # slope cells (lo, lo + 1) with midpoints bracketing the radius; clamped at the ends (linear extrapolation)
    lo = int(np.clip(np.searchsorted(midpoints, radius, side='right') - 1, 0, grid.n_cells - 2))
    share = (radius - midpoints[lo]) / (midpoints[lo + 1] - midpoints[lo])
    slope_lo, slope_hi = (1.0 - share) / widths[lo], share / widths[lo + 1]
    return _SourcePoint(cell=cell, value_nodes=(cell, cell + 1), value_weights=(1.0 - upper, upper),
                        slope_nodes=(lo, lo + 1, lo + 2),
                        slope_weights=(-slope_lo, slope_lo - slope_hi, slope_hi))


def multipole_coupling(order, orientation) -> float:
    """g = sqrt(sum over m of |angular function . n|^2) of a dipole on the polar axis: radial sqrt(2 l + 1) (Y^, the
    'er' family only), tangential sqrt((2 l + 1) / 2) (Psi^ / Phi^, both families)."""
    if orientation == 'radial':
        return math.sqrt(2.0 * order + 1.0)
    if orientation == 'tangential':
        return math.sqrt((2.0 * order + 1.0) / 2.0)
    raise ValueError(f"dipole orientation must be one of {', '.join(DIPOLE_ORIENTATIONS)}, got {orientation!r}")


@dataclass(frozen=True, eq=False)
class SphereDipole:
    """A point current element of moment I d [A m] (complex: the phasor) at radius [m] on the polar axis, radial or
    tangential (parallel to the plane-wave polarisation)."""
    radius: float
    orientation: str
    moment: complex

    def node_rhs(self, system) -> np.ndarray:
        """complex128 (n + 1,) weak right-hand side on the nodes of `system` (zero for a radial dipole in 'hr')."""
        rhs = np.zeros(system.grid.edges.size, dtype=np.complex128)
        if system.family == 'hr' and self.orientation == 'radial':
            return rhs
        point = _source_point(system.grid, self.radius, 'dipole source')
        coupling = multipole_coupling(system.order, self.orientation)
        nodes, weights = point.value_nodes, point.value_weights
        if system.family == 'hr':
            factor = -1j * system.omega * MU0 * self.moment * coupling / (4.0 * math.pi * self.radius)
        else:
            eps_source = _source_permittivity(system, point.cell)
            if self.orientation == 'radial':
                factor = (-self.moment * coupling * math.sqrt(system.angular)
                          / (4.0 * math.pi * eps_source * self.radius ** 2))
            else:
                factor = -self.moment * coupling / (4.0 * math.pi * eps_source * self.radius)
                nodes, weights = point.slope_nodes, point.slope_weights
        for node, weight in zip(nodes, weights):
            rhs[node] += factor * weight
        return rhs

    def emitted_power(self, system, node_values) -> float:
        """Time-averaged power [W] this element delivers, -1/2 Re(E . (I d)*), summed over its m terms (module
        header)."""
        if system.family == 'hr' and self.orientation == 'radial':
            return 0.0
        point = _source_point(system.grid, self.radius, 'dipole source')
        coupling = multipole_coupling(system.order, self.orientation)
        if system.family == 'hr':
            field = coupling * point.value(node_values) / self.radius
        else:
            eps_source = _source_permittivity(system, point.cell)
            if self.orientation == 'radial':
                curl = -coupling * math.sqrt(system.angular) * point.value(node_values) / self.radius ** 2
            else:
                curl = -coupling * point.slope(node_values) / self.radius
            field = curl / (1j * system.omega * EPS0 * eps_source)
        return float(-0.5 * (np.conj(self.moment) * field).real)


def _source_permittivity(system, cell) -> complex:
    """Complex eps of the cell holding a source; ValueError in a PEC cell."""
    if bool(system.materials.pec[cell]):
        raise ValueError(f'dipole source in the PEC cell {cell}: move it out of the metal')
    return complex(system.materials.eps[cell])


def homogeneous_dipole_power(moment, omega, eps, mu) -> float:
    """eta k^2 |I d|^2 / (12 pi) [W] of a current element in an unbounded lossless medium (eps, mu real parts)."""
    eps, mu = float(np.real(eps)), float(np.real(mu))
    if eps <= 0.0 or mu <= 0.0:
        return math.nan
    k = omega / C0 * math.sqrt(eps * mu)
    return ETA0 * math.sqrt(mu / eps) * k ** 2 * abs(complex(moment)) ** 2 / (12.0 * math.pi)


def _lossless_exterior(system, what):
    """(k_ext [1/m], real eps_ext, real mu_ext) of a radiation boundary; ValueError without one or for a lossy
    exterior (no conserved far-field flux)."""
    if not system.radiation:
        raise ValueError(f"{what} needs the outer face 'radiation' (an open exterior)")
    _, eps_ext, mu_ext = system.exterior
    for name, value in (('eps', eps_ext), ('mu', mu_ext)):
        if abs(value.imag) > LOSSLESS_RELATIVE_IMAG * abs(value.real) or value.real <= 0.0:
            raise ValueError(f'{what}: the exterior medium (the last radial cell, continued to infinity) must be '
                             f'lossless with positive {name}, got {name} = {value:.6g}')
    eps_ext, mu_ext = eps_ext.real, mu_ext.real
    return system.k0 * math.sqrt(eps_ext * mu_ext), eps_ext, mu_ext


def plane_wave_amplitude(system, amplitude) -> complex:
    """alpha of the regular wave alpha psi_l(k r) in y units for a plane wave of amplitude E0 [V/m] (both m = +-1 of
    this term, sphere-RMS)."""
    k_ext, eps_ext, mu_ext = _lossless_exterior(system, 'plane_wave')
    rms = complex(amplitude) * math.sqrt((2.0 * system.order + 1.0) / 2.0)
    if system.family == 'hr':
        return rms / k_ext
    return rms / (ETA0 * math.sqrt(mu_ext / eps_ext) * k_ext)


def plane_wave_rhs(system, alpha) -> np.ndarray:
    """complex128 (n + 1,) node right-hand side of an incident regular wave alpha psi_l: j p k alpha / xi(k R) on the
    outer node."""
    k_ext, _, _ = _lossless_exterior(system, 'plane_wave')
    xi, _, _ = riccati_hankel(system.order, k_ext * float(system.grid.edges[-1]))
    rhs = np.zeros(system.grid.edges.size, dtype=np.complex128)
    rhs[-1] = 1j * system.exterior[0] * k_ext * alpha / xi
    return rhs


def outgoing_coefficient(system, node_values, alpha=0.0) -> complex:
    """c of the outgoing part c xi_l(k r) at the radiation boundary: (y_N - alpha psi_l(k R)) / xi_l(k R) (0 when xi
    overflows)."""
    k_ext, _, _ = _lossless_exterior(system, 'the outgoing wave')
    x = k_ext * float(system.grid.edges[-1])
    xi, _, _ = riccati_hankel(system.order, x)
    regular = riccati_bessel(system.order, x)[0] if alpha != 0.0 else 0.0
    return complex((node_values[-1] - alpha * regular) / xi)


def outgoing_power(system, coefficient) -> float:
    """Time-averaged power [W] of the outgoing wave c xi_l(k r) of a term (both families: 2 pi k |c|^2 / (omega mu0 mu)
    or / (omega eps0 eps), from Im(xi xi'*) = 1)."""
    k_ext, eps_ext, mu_ext = _lossless_exterior(system, 'the outgoing power')
    material = MU0 * mu_ext if system.family == 'hr' else EPS0 * eps_ext
    return float(2.0 * math.pi * k_ext * abs(coefficient) ** 2 / (system.omega * material))


def mie_cross_sections(system, coefficient) -> tuple:
    """(C_sca, C_ext) [m^2] of one term from its Mie coefficient a (= -c / alpha): 2 pi / k^2 (2 l + 1) (|a|^2,
    Re a)."""
    k_ext, _, _ = _lossless_exterior(system, 'plane_wave')
    weight = 2.0 * math.pi * (2.0 * system.order + 1.0) / k_ext ** 2
    return weight * abs(coefficient) ** 2, weight * coefficient.real


# ---- fields ----
def centre_fields(system, node_values, omega=None) -> dict:
    """complex128 (n,) radial functions er, et, hr, ht at the cell centres (module header; y and y' of the cell's two
    nodes), 0 in PEC cells; omega [rad/s] (complex for a resonance) defaults to the system's."""
    omega = system.omega if omega is None else complex(omega)
    grid = system.grid
    values = np.asarray(node_values, dtype=np.complex128)
    centre_value = 0.5 * (values[:-1] + values[1:])
    slope = np.diff(values) / grid.widths
    radius = grid.centres
    root = math.sqrt(system.angular)
    pec = np.asarray(system.materials.pec, dtype=bool)
    zero = np.zeros(grid.n_cells, dtype=np.complex128)
    if system.family == 'hr':
        mu = np.where(pec, 1.0, system.materials.mu)
        fields = {'er': zero, 'et': centre_value / radius,
                  'hr': root * centre_value / (1j * omega * MU0 * mu * radius ** 2),
                  'ht': slope / (1j * omega * MU0 * mu * radius)}
    else:
        eps = np.where(pec, 1.0, system.materials.eps)
        fields = {'er': -root * centre_value / (1j * omega * EPS0 * eps * radius ** 2),
                  'et': -slope / (1j * omega * EPS0 * eps * radius),
                  'hr': zero, 'ht': centre_value / radius}
    return {name: np.where(pec, 0.0, value) for name, value in fields.items()}


def absorbed_power_density(system, fields, omega=None) -> np.ndarray:
    """float64 (n,) sphere mean of 1/2 omega eps0 (-Im eps) |E|^2 [W/m^3] of one term's centre fields."""
    omega = system.omega if omega is None else float(np.real(omega))
    loss = -np.imag(np.asarray(system.materials.eps, dtype=np.complex128))
    return 0.5 * omega * EPS0 * loss * (np.abs(fields['er']) ** 2 + np.abs(fields['et']) ** 2)


# ---- resonances ----
@dataclass(frozen=True, eq=False)
class SphereCandidate:
    """One resonance of a term: complex omega [rad/s] (Im > 0 decays), the node values (n + 1,), the relative residual
    of the refinement (0 for a closed system: exact for the frozen pencil) and its iterations."""
    omega: complex
    node_values: np.ndarray
    residual: float
    iterations: int


def refine_resonance(system, k, vector, factorize):
    """(k, free vector, iterations, converged) of Rayleigh-functional inverse iteration on T(k) (operator_at) from
    (k, vector); factorize(host CSR) -> factor with solve_host."""
    vector = np.asarray(vector, dtype=np.complex128)
    row = system.boundary_row
    mass, base = system.mass, system.base_stiffness
    k = complex(k)
    for iteration in range(1, REFINE_MAX_ITERATIONS + 1):
        stiffness_product = complex(vector @ (base @ vector))
        mass_product = complex(vector @ (mass @ vector))
        boundary = complex(vector[row]) ** 2
        k_new = k
        for _ in range(_SCALAR_NEWTON_STEPS):
            value, derivative = system.radiation_term(k_new)
            residual = stiffness_product + value * boundary - k_new ** 2 * mass_product
            slope = derivative * boundary - 2.0 * k_new * mass_product
            if slope == 0:
                break
            step = residual / slope
            k_new -= step
            if abs(step) <= 1e-15 * abs(k_new):
                break
        change = abs(k_new - k)
        k = k_new
        if change <= REFINE_TOLERANCE * abs(k) and iteration > 1:
            return k, vector, iteration, True
        _, derivative = system.radiation_term(k)
        tangent = -2.0 * k * (mass @ vector)
        tangent[row] += derivative * vector[row]
        factor = None
        try:
            factor = factorize(system.operator_at(k))
            update = factor.solve_host(tangent)
        except RuntimeError:   # T(k) singular to working precision: k is the eigenvalue
            return k, vector, iteration, True
        finally:
            release = getattr(factor, 'release', None)
            if release is not None:
                release()
        if not np.all(np.isfinite(update)) or not np.any(update):
            return k, vector, iteration, change <= 1e3 * REFINE_TOLERANCE * abs(k)
        vector = update / update[int(np.argmax(np.abs(update)))]
    return k, vector, REFINE_MAX_ITERATIONS, False


def relative_residual(system, k, vector) -> float:
    """|T(k) x| / (max |diag T| |x|) of a resonance candidate."""
    operator = system.operator_at(k)
    scale = float(np.max(np.abs(operator.diagonal()))) * float(np.linalg.norm(vector))
    return float(np.linalg.norm(operator @ vector)) / scale if scale > 0.0 else math.inf


def solve_term_resonances(system, n_resonances, eigen_solver):
    """(candidates nearest the target omega = system.omega, nearest first, at most n_resonances; info dict) of one
    term: shift-invert about k_t^2 of (stiffness at k_t, mass) by eigen_solver.eigs_shifted (2 n + 10 eigenpairs), then
    refine_resonance with the exact radiation term (radiation boundary only; closed systems are exact for the frozen
    pencil). Candidates with Re k <= 0, unconverged or with a residual above REFINE_RESIDUAL_LIMIT are dropped, refined
    duplicates merged."""
    n_free = system.n_unknowns
    n_wanted = min(2 * int(n_resonances) + 10, n_free - 2)
    if n_wanted < 1:
        raise ValueError(f'{system.family} l = {system.order}: {n_free} free radial unknowns are too few for an eigen '
                         'solve; refine the mesh')
    k_target = system.k0
    rng = np.random.default_rng(_START_VECTOR_SEED)
    start = rng.standard_normal(n_free) + 1j * rng.standard_normal(n_free)
    started = time.perf_counter()
    eigenvalues, vectors, applications, converged, factor = eigen_solver.eigs_shifted(
        system.stiffness, system.mass, k_target ** 2, n_wanted, start_vector=start)
    release = getattr(factor, 'release', None)
    if release is not None:
        release()
    eigen_seconds = time.perf_counter() - started
    candidates, dropped, iterations_total = [], 0, 0
    refine_started = time.perf_counter()
    for index, eigenvalue in enumerate(eigenvalues):
        if not np.isfinite(eigenvalue):
            continue
        k = complex(np.sqrt(complex(eigenvalue)))
        if k.real <= 0.0:
            continue
        vector = np.asarray(vectors[:, index], dtype=np.complex128)
        iterations, ok = 0, True
        if system.radiation:
            k, vector, iterations, ok = refine_resonance(system, k, vector, eigen_solver.factorize_host)
            iterations_total += iterations
        residual = relative_residual(system, k, vector) if system.radiation else 0.0
        if not ok or k.real <= 0.0 or not residual <= REFINE_RESIDUAL_LIMIT:
            dropped += 1
            continue
        if any(abs(k - other.omega / C0) <= DUPLICATE_RELATIVE * abs(k) for other in candidates):
            continue
        candidates.append(SphereCandidate(omega=k * C0, node_values=system.full(vector), residual=residual,
                                          iterations=iterations))
    candidates.sort(key=lambda candidate: abs(candidate.omega - system.omega))
    info = {'n_unknowns': n_free, 'eigenvalues_requested': n_wanted, 'eigenvalues_converged': int(len(eigenvalues)),
            'arpack_converged': bool(converged), 'operator_applications': int(applications),
            'eigen_seconds': eigen_seconds, 'refine_seconds': time.perf_counter() - refine_started,
            'refine_iterations': iterations_total, 'candidates_dropped': dropped}
    return candidates[:int(n_resonances)], info
