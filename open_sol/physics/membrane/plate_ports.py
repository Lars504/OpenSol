"""Plate ports of the membrane (plan F2): numeric strip modes of a straight tensioned Kirchhoff plate guide from the 1D
Hermite strip (fem.line_fem), the exact Dirichlet-to-Neumann map of that semi-discrete strip (reciprocity-symmetric),
the incident load and the modal port powers. Host numpy, scipy inside functions."""
# exp(+j omega t); d = outward distance beyond the port face (the FE normal n), s = the coordinate along it, the guide
# outside straight and uniform: T_nn, T_ss (1 + j eta), rho_s, D (1 + j eta), nu (port medium = mean of the port's
# elements). Strip: Hermite cubic w(s) on the port's edge nodes, walls by kind ('clamped' w = 0 and w' = 0 by Nitsche
# as the 2D clamped edges (K0 += D wall_nitsche), 'supported' w = 0, 'symmetry' w' = 0, 'free' nothing). Writing w(d, s) = sum_i W_i(d) N_i(s), the plate energy over the exterior is
# int W''^T K2 V'' + W'^T K1 V' + W^T K0 V dd + boundary terms with
#   K2 = D M,  K1 = T_nn M + 2 (1 - nu) D S - nu D (C + C^T),  K0 = D B + T_ss S - rho_s omega^2 M
# (M = int N N, S = int N' N', B = int N'' N'', C_ij = int N_i N_j''; for held walls K1 = T_nn M + 2 D S). Modes W =
# phi exp(-j beta d), lambda = beta^2: (K0 + lambda K1 + lambda^2 K2) phi = 0, 2 n roots (n free strip dofs), solved
# densely through the companion pencil per frequency; outgoing branch Im beta <= 0 (a real beta takes the sign of
# positive power flux); numbered by cut-off = Re lambda descending (propagating modes first). The exterior boundary form
# at d = 0 is b(W, V) = sum_k a_k [r_k^w V + r_k^t V'] with
#   r_k^w = j beta_k phi_k^T (lambda_k K2 + K1 + nu D C),  r_k^t = phi_k^T (lambda_k K2 - nu D C^T)
# (V, V' = the traces v and dv/dn of the test function). The traces of mode k are t_k = (phi_k, -j beta_k phi_k), so
# with all 2 n modes the amplitudes of a trace pair t are a = Phi^-1 t and the DtN term added to the operator is
# t(v)^T Z t(w), Z = R^T Phi^-1 (R rows r_k = (r_k^w, r_k^t)). The QEP orthogonality phi_k^T (K1 + (lambda_k + lambda_l)
# K2) phi_l = 0 makes R Phi symmetric (reciprocity), so Z is complex symmetric (kept so by symmetrising round-off).
# FE traces enter through the L2 projection Pi onto the free strip space (w trace and one-sided dn w of the port
# elements; real), so the block on the port elements' P2 dofs is Pi^T Z Pi. An incident mode m of amplitude A (traces
# t_m^- = (phi_m, +j beta_m phi_m), weights r_m^- = (-r_m^w, r_m^t)) adds the load A (Z t_m^- - r_m^-). Outgoing
# amplitudes b = Phi^-1 (t - A t_m^-), power P_k = (omega / 2) Im(r_k . conj t_k) |b_k|^2 [W] (the exterior flux; only
# real-beta modes carry power, and their cross terms vanish), so the discrete power balance is exact (lossless).
# Scaled unknowns: Hermite slope dofs x ell (the mean strip width; basis / ell, since the line_fem slope basis is O(h))
# and the dn w traces x ell, lambda in units 1 / ell^2.
from dataclasses import dataclass

import numpy as np

from ...fem.elements import lagrange_values
from ...fem.indicators import edge_sides
from ...fem.line_fem import assemble_line_matrix, end_dofs, evaluate_line, line_dofs
from .bending_fem import PLATE_PENALTY, p2_derivatives
from .ports import PORT_UNIFORM_RTOL, PortBase

WALL_KINDS = ('clamped', 'supported', 'symmetry', 'free')
# clamped: w held, w' by Nitsche (PlateStrip.wall_nitsche) as the 2D clamped edges, so an unresolved boundary layer
# sqrt(D / T) behaves alike on both sides of the port
_HELD = {'clamped': (True, False), 'supported': (True, False), 'symmetry': (False, True), 'free': (False, False)}
PLATE_PORT_DEGREE = 6        # Gauss degree along the port edges (Hermite cubic x P2 value: exact on straight edges)
REAL_BETA_RTOL = 1e-9        # |Im beta| below this x |beta|: a propagating mode (direction from its power flux)
STRIP_TWIN_RTOL = 1e-12      # two port strips whose matrices agree to this share their strip modes (round-off only)


@dataclass(frozen=True, eq=False)
class PlateStrip:
    """Hermite cubic strip across a plate port: line dofs on its nodes s [m], the free dofs (walls held by kind), their
    basis scale (1 for w, 1 / ell for w': unknowns w, ell w') and the scaled dense free-block matrices mass int N N,
    slope int N' N', curvature int N'' N'', coupling C_ij = int N_i N_j'', wall_nitsche (x D: clamped wall slopes)."""
    dofs: object
    free: np.ndarray
    scale: np.ndarray
    length_scale: float
    value_rows: np.ndarray       # rows of the free numbering that are w dofs
    mass: np.ndarray
    slope: np.ndarray
    curvature: np.ndarray
    coupling: np.ndarray
    wall_nitsche: np.ndarray

    @property
    def size(self) -> int:
        return int(self.free.size)

    def basis_at(self, positions) -> np.ndarray:
        """(len(positions), size) scaled free basis N_f(s) x scale_f at s [m] on the strip."""
        basis = _free_basis(self.dofs, self.free, positions)
        if np.isnan(basis).any():
            raise ValueError('plate port: a trace point lies off the strip')
        return basis * self.scale[None, :]


def _free_basis(dofs, free, positions, derivative=0) -> np.ndarray:
    """(len(positions), len(free)) d^derivative N_f / ds^derivative of the free dofs at s [m] (a node: the upper
    element, the last node the last element; NaN off the strip)."""
    positions = np.asarray(positions, dtype=np.float64).ravel()
    basis = np.empty((positions.size, free.size))
    unit = np.zeros(dofs.n_dofs)
    for column, dof in enumerate(free):
        unit[dof] = 1.0
        basis[:, column] = evaluate_line(dofs, unit, positions, derivative)
        unit[dof] = 0.0
    return basis


def _wall_nitsche(dofs, free, scale, wall_kinds) -> np.ndarray:
    """(n, n) scaled K0 part / D of the clamped walls: bending_fem's one-sided Nitsche edge terms on the wall line,
    -n (c g^T + g c^T) + (PLATE_PENALTY / width) g g^T with g = N', c = N'' at the wall node, n = -1 / +1 its outward
    sign along s (M_nn = D w_ss there: w is held; no d derivatives, so the DtN weights are unchanged)."""
    terms = np.zeros((free.size, free.size))
    for kind, node, sign, width in zip(wall_kinds, (dofs.edges[0], dofs.edges[-1]), (-1.0, 1.0),
                                       (dofs.widths[0], dofs.widths[-1])):
        if kind != 'clamped':
            continue
        slope = _free_basis(dofs, free, [node], 1)[0] * scale
        curvature = _free_basis(dofs, free, [node], 2)[0] * scale
        terms += (-sign * (np.outer(curvature, slope) + np.outer(slope, curvature))
                  + (PLATE_PENALTY / float(width)) * np.outer(slope, slope))
    return terms


def plate_strip(nodes, wall_kinds) -> PlateStrip:
    """PlateStrip on the nodes s [m] (sorted, unique) with wall_kinds (at the first node, at the last) of WALL_KINDS."""
    for kind in wall_kinds:
        if kind not in WALL_KINDS:
            raise ValueError(f"plate port: wall kind must be one of {', '.join(WALL_KINDS)}, got {kind!r}")
    dofs = line_dofs(np.unique(np.asarray(nodes, dtype=np.float64)), 'hermite', 3)
    held = np.concatenate([end_dofs(dofs, face, *_HELD[kind]) for face, kind in zip(('x_lo', 'x_hi'), wall_kinds)])
    free = np.setdiff1d(np.arange(dofs.n_dofs), held).astype(np.int64)
    if not free.size:
        raise ValueError('plate port: the strip has no free dofs (a port needs at least two edges)')
    length_scale = float(np.mean(dofs.widths))
    # unknowns w and ell w': the line_fem slope basis is h x O(1), so x ell would leave its blocks ~h^4 below w's
    scale = np.where(free % 2 == 1, 1.0 / length_scale, 1.0)

    def matrix(test, trial):
        dense = assemble_line_matrix(dofs, 1.0, test, trial).toarray()[np.ix_(free, free)]
        return scale[:, None] * dense * scale[None, :]
    return PlateStrip(dofs, free, scale, length_scale, np.flatnonzero(free % 2 == 0), matrix(0, 0), matrix(1, 1),
                      matrix(2, 2), matrix(0, 2), _wall_nitsche(dofs, free, scale, wall_kinds))


@dataclass(frozen=True, eq=False)
class StripModes:
    """The 2 n strip modes of one angular frequency [rad/s] in cut-off order: lambda = beta^2 [1/m^2], beta [rad/m],
    traces Phi (2 n, 2 n) and weights R (2 n, 2 n) per column (scaled traces, header), flux (2 n,) = outward power per
    |amplitude|^2 [W/m^2] (power_factor 1), the LU of Phi and the symmetric DtN matrix Z on scaled traces."""
    omega: float
    squared: np.ndarray
    beta: np.ndarray
    traces: np.ndarray
    weights: np.ndarray
    flux: np.ndarray
    factor: tuple
    impedance: np.ndarray

    @property
    def count(self) -> int:
        return int(self.beta.size)

    def incoming(self, index) -> tuple:
        """(traces, weights) of the incoming partner of mode `index` (0-based): beta -> -beta."""
        size = self.traces.shape[0] // 2
        traces, weights = self.traces[:, index].copy(), self.weights[:, index].copy()
        traces[size:] *= -1.0
        weights[:size] *= -1.0
        return traces, weights

    def inflow(self, index) -> float:
        """Power [W] per |amplitude|^2 the incoming partner of mode `index` carries into the domain (power_factor 1)."""
        traces, weights = self.incoming(index)
        return float(-0.5 * self.omega * np.imag(np.sum(weights * np.conj(traces))))


def _normalized(shapes, value_rows) -> np.ndarray:
    """Mode shapes scaled to peak |w| 1 with the first w dof reaching half the peak (in s order) real positive."""
    values = shapes[value_rows]
    magnitudes = np.abs(values)
    peaks = magnitudes.max(axis=0)
    if not (peaks > 0.0).all():
        raise ValueError('plate port: a strip mode has no displacement (slopes only)')
    first = np.argmax(magnitudes >= 0.5 * peaks[None, :], axis=0)
    lead = values[first, np.arange(values.shape[1])]
    return shapes / ((lead / np.abs(lead)) * peaks)[None, :]


def _mode_traces(strip, stiffness, coupling_weight, squared, beta, shapes, omega) -> tuple:
    """(weights R (2 n, m), traces Phi (2 n, m), flux (m,)) of mode shapes (n, m) (header formulas, scaled)."""
    second, first = stiffness
    ell = strip.length_scale
    curvature_part = (second @ shapes) * squared[None, :]
    shear = curvature_part + first @ shapes + coupling_weight * (strip.coupling.T @ shapes)
    moment = curvature_part - coupling_weight * (strip.coupling @ shapes)
    weights = np.vstack([(1j * beta)[None, :] * shear, moment / ell])
    traces = np.vstack([shapes, ell * (-1j * beta)[None, :] * shapes])
    flux = 0.5 * omega * np.imag(np.sum(weights * np.conj(traces), axis=0))
    return weights, traces, flux


def strip_modes(strip, medium, omega) -> StripModes:
    """StripModes of a PlateStrip for medium (T_nn, T_ss (1 + j eta) [N/m], rho_s [kg/m^2], D (1 + j eta) [N m], nu) at
    omega [rad/s] (companion pencil, dense eig)."""
    import scipy.linalg
    tension_normal, tension_transverse, density, rigidity, poisson = medium
    if rigidity == 0.0:
        raise ValueError('plate port: the bending stiffness at the port is 0 (a film port takes the analytic modes)')
    omega = float(omega)
    mass, slope = strip.mass, strip.slope
    coupling_weight = poisson * rigidity
    second = rigidity * mass
    first = (tension_normal * mass + 2.0 * (1.0 - poisson) * rigidity * slope
             - coupling_weight * (strip.coupling + strip.coupling.T))
    zeroth = (rigidity * (strip.curvature + strip.wall_nitsche) + tension_transverse * slope
              - density * omega ** 2 * mass)
    ell, size = strip.length_scale, strip.size
    scaled_first, scaled_second = first / ell ** 2, second / ell ** 4     # lambda ell^2 = mu, dimensionless
    norm = max(float(np.abs(zeroth).max()), float(np.abs(scaled_first).max()), float(np.abs(scaled_second).max()))
    identity, zero = np.eye(size), np.zeros((size, size))
    left = np.block([[zero, identity], [-zeroth / norm, -scaled_first / norm]])
    right = np.block([[identity, zero], [zero, scaled_second / norm]])
    mu, vectors = scipy.linalg.eig(left, right)
    finite = np.isfinite(mu)
    if int(finite.sum()) != 2 * size:
        raise ValueError(f'plate port: the strip pencil has {int((~finite).sum())} infinite roots (singular bending '
                         'mass block)')
    mu, vectors = mu[finite], vectors[:, finite]
    small = np.abs(mu) <= 1.0
    shapes = np.where(small[None, :], vectors[:size], vectors[size:] / np.where(small, 1.0, mu)[None, :])
    squared = mu / ell ** 2
    order = np.lexsort((squared.imag, -squared.real))
    squared, shapes = squared[order], _normalized(shapes[:, order], strip.value_rows)
    beta = np.sqrt(squared.astype(np.complex128))
    beta = np.where(beta.imag > 0.0, -beta, beta)
    stiffness = (second, first)
    weights, traces, flux = _mode_traces(strip, stiffness, coupling_weight, squared, beta, shapes, omega)
    backward = (np.abs(beta.imag) <= REAL_BETA_RTOL * np.abs(beta)) & (flux < 0.0)
    if backward.any():   # a real beta whose flux points inward: the outgoing wave is the other sign
        beta = np.where(backward, -beta, beta)
        weights, traces, flux = _mode_traces(strip, stiffness, coupling_weight, squared, beta, shapes, omega)
    factor = scipy.linalg.lu_factor(traces)
    impedance = scipy.linalg.lu_solve(factor, weights.T, trans=1).T    # Z = R^T Phi^-1 (R^T = weights)
    impedance = 0.5 * (impedance + impedance.T)
    return StripModes(omega, squared, beta, traces, weights, flux, factor, impedance)


def _same_strip(first, second) -> bool:
    """True when two PlateStrips give the same strip_modes: equal free / value rows, matrices and ell within round-off
    (translated nodes of the same spacing differ by round-off only)."""
    if first.size != second.size or not np.array_equal(first.value_rows, second.value_rows):
        return False
    if not np.isclose(first.length_scale, second.length_scale, rtol=STRIP_TWIN_RTOL, atol=0.0):
        return False
    return all(np.allclose(getattr(first, name), getattr(second, name), rtol=STRIP_TWIN_RTOL,
                           atol=STRIP_TWIN_RTOL * float(np.abs(getattr(first, name)).max(initial=0.0)))
               for name in ('mass', 'slope', 'curvature', 'coupling', 'wall_nitsche'))


def link_twin_strips(ports) -> None:
    """Point each assembled PlatePort of ports at the first earlier one with the same strip (its twin), whose StripModes
    it reuses at an equal medium and omega (both ends of a straight guide: one dense eig per frequency, not two)."""
    plate_ports = [port for port in ports if isinstance(port, PlatePort)]
    for index, port in enumerate(plate_ports):
        port.twin = next((other for other in plate_ports[:index]
                          if other.twin is None and _same_strip(other.strip, port.strip)), None)


class PlatePort(PortBase):
    """Runtime of a port on a plate guide (bending_stiffness > 0 at its face): MembranePort's interface (positions,
    terms, outputs, medium, result extras) on the numeric strip modes (header); the DtN holds every strip mode,
    mode_count sets the reported modes and the incident range (numbered by cut-off on the drawn guide)."""

    def __init__(self, name, face, edges, start, width, mode_count, wall_kinds, incidents=(),
                 tension_tolerance=PORT_UNIFORM_RTOL, mirror_walls=(None, None)):
        # incidents numbered by cut-off on the drawn guide; positions = every system dof of the port elements
        self._init_guide(name, face, edges, start, width, mode_count, incidents, tension_tolerance, mirror_walls)
        self.wall_kinds = tuple(str(kind) for kind in wall_kinds)
        self.free_walls = tuple(kind == 'symmetry' for kind in self.wall_kinds)
        self.drive_scale = 1.0
        self.strip = None
        self._trace_map = None      # real (2 n, len(positions)): scaled strip traces (w, ell dn w) of the FE dofs
        self.medium = None          # (T_nn, T_ss (1 + j eta), rho_s, D (1 + j eta), nu)
        self._cached = None         # StripModes of the last omega
        self.twin = None            # an earlier PlatePort with the same strip matrices (link_twin_strips)

    def assemble(self, dofs) -> None:
        """Positions (every system dof of the port's elements: dn v is one-sided) and the trace map Pi: L2 projection of
        the w trace and of dn w (outward) onto the free strip space on the port's edge nodes."""
        mesh = dofs.mesh
        inside = dofs.element_position >= 0
        sides = edge_sides(mesh, self.edges, inside, 2, PLATE_PORT_DEGREE)
        if len(sides.edges) != len(self.edges) or (sides.elements[:, 1] >= 0).any():
            raise ValueError(f"port '{self.name}': every port edge needs exactly one active element")
        elements = sides.elements[:, 0]
        local = dofs.element_dofs[dofs.element_position[elements]]                  # (k, 6)
        reference = sides.reference_points(0)
        values = lagrange_values(2, reference)                                       # (k, q, 6)
        gradients, _, _ = p2_derivatives(mesh, elements, reference)
        slopes = np.einsum('kqac,kqc->kqa', gradients, sides.normals)              # outward dn of the basis
        nodes = np.unique(mesh.points[mesh.edges[self.edges]][..., self.tangent_axis])
        self.strip = plate_strip(nodes, self.wall_kinds)
        basis = self.strip.basis_at(sides.points[..., self.tangent_axis])             # (k q, n)
        positions, inverse = np.unique(local, return_inverse=True)
        inverse = inverse.reshape(local.shape)
        count, points = len(elements), sides.points.shape[1]
        rows = np.broadcast_to(np.arange(count * points).reshape(count, points, 1), values.shape)
        columns = np.broadcast_to(inverse[:, None, :], values.shape)
        weighted = sides.weights[:, :, None]
        flat = (rows * positions.size + columns).ravel()   # bincount over the flat (point, position) index
        shape = (count * points, positions.size)
        trace_w = np.bincount(flat, weights=(values * weighted).ravel(), minlength=shape[0] * shape[1]).reshape(shape)
        trace_t = np.bincount(flat, weights=(slopes * weighted).ravel(), minlength=shape[0] * shape[1]).reshape(shape)
        mass = self.strip.mass
        projected_w = np.linalg.solve(mass, basis.T @ trace_w)
        projected_t = np.linalg.solve(mass, basis.T @ trace_t)
        self._trace_map = np.vstack([projected_w, self.strip.length_scale * projected_t])
        self.positions = positions.astype(np.int64)
        self._cached = None

    def set_medium(self, tension_normal, tension_transverse, density, rigidity=0.0, poisson=0.0) -> None:
        medium = (complex(tension_normal), complex(tension_transverse), float(density), complex(rigidity),
                  float(poisson))
        if medium != self.medium:
            self._cached = None
        self.medium = medium

    def modes(self, omega) -> StripModes:
        """StripModes at omega [rad/s] (the last one cached)."""
        if self.medium is None or self.strip is None:
            raise RuntimeError(f"port '{self.name}': medium or strip not set (coefficients not applied yet)")
        omega = float(omega)
        if self._cached is None or self._cached.omega != omega:
            twin = self.twin
            if (twin is not None and twin.medium == self.medium and twin._cached is not None
                    and twin._cached.omega == omega):
                modes = twin._cached   # the same strip matrices and medium: the same modes (one dense eig per guide)
            else:
                modes = strip_modes(self.strip, self.medium, omega)
            if modes.count < self.mode_count:
                raise ValueError(f"port '{self.name}': mode_count {self.mode_count} exceeds the {modes.count} strip "
                                 'modes of the port (two per free strip dof)')
            self._cached = modes
        return self._cached

    def terms(self, omega) -> tuple:
        """(dense (k, k) DtN block Pi^T Z Pi on positions, (k,) incident load) at omega [rad/s]."""
        modes = self.modes(omega)
        trace_map = self._trace_map
        block = trace_map.T @ modes.impedance @ trace_map
        load = np.zeros(trace_map.shape[0], dtype=np.complex128)
        for number, amplitude in self.incidents:
            traces, weights = modes.incoming(number - 1)
            load += amplitude * (modes.impedance @ traces - weights)
        return block, trace_map.T @ load

    def outputs(self, omega, solution) -> dict:
        """{'beta', 'total', 'outgoing' (b_n), 'power' (P_n [W]), 'incident_power' [W]} of the first mode_count modes at
        omega [rad/s] (powers of the unfolded guide: power_factor)."""
        import scipy.linalg
        modes = self.modes(omega)
        traces = self._trace_map @ np.asarray(solution)[self.positions]
        incident_power = 0.0
        for number, amplitude in self.incidents:
            incoming, _ = modes.incoming(number - 1)
            traces = traces - amplitude * incoming
            incident_power += self.power_factor * modes.inflow(number - 1) * abs(amplitude) ** 2
        outgoing = scipy.linalg.lu_solve(modes.factor, traces)[:self.mode_count]
        total = outgoing.copy()
        for number, amplitude in self.incidents:
            total[number - 1] += amplitude
        power = self.power_factor * modes.flux[:self.mode_count] * np.abs(outgoing) ** 2
        return {'beta': modes.beta[:self.mode_count], 'total': total, 'outgoing': outgoing, 'power': power,
                'incident_power': float(incident_power)}
