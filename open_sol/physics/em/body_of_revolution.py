"""Body-of-revolution (BOR) FDFD of the (r, z) half plane, fields ~ exp(j omega t - j m phi) (symmetry plan 2.3):
full-vector finite-integration Yee operator, driven solve and shift-invert resonances; host numpy / scipy only."""
# Grid: x = r, y = z. The axis r = 0 is the lo edge of grid cell 0 (no PML side there); edge k = lo edge of cell k,
# edge n = the hi grid edge. Unknowns (3D cylindrical Yee cell projected, as cross_section_modes with x -> r, y -> z,
# z -> phi): E_r on (edge k, centre j), u = -j E_phi on the centres, E_z on (centre i, edge l). With u instead of
# E_phi every curl entry is real (d/dphi = -j m), so the operator is complex symmetric (real symmetric when lossless
# and PML-free) and (E_r, u, E_z) of a lossless resonance is real up to one phase: the standing wave E_r, E_z ~ cos(m
# phi), E_phi ~ u sin(m phi). curl E = (-j a_r, a_phi, j a_z) with
#   a_r(centre i, edge l)  = (m / r_i) E_z + d_z u
#   a_z(edge k, centre j)  = (r_k u_k - r_{k-1} u_{k-1}) / (rbar_k L_k) + (m / rbar_k) E_r
#   a_phi(edge k, edge l)  = d_z E_r - d_r E_z
# L_k = dual length of edge k (centre k-1 .. centre k; an end edge ends on itself), rbar_k its mean radius, so
# rbar_k L_k = integral r dr over it (the disc r_0^2 / 2 at the axis). A = C^T diag(V_a / mu) C - k0^2 diag(V_E eps),
# V = ring volume per radian of each point (cell values: r_i w_i h_j; edge ones with L / rbar). Discrete gradients (E
# = grad phi, u = -(m / r) phi) lie in the kernel of C exactly: their resonances sit at omega = 0 (the eigen-solve's
# squared Cayley map maps them to 0, never wanted; any round-off copy is dropped).
# Axis (edge 0 of r): E_r, H_z, H_phi sit there, E_phi, E_z, H_r start half a cell off it. The dual face of the axis E_r
# (normal to r at r = 0) and the dual edge of the axis H_phi (an arc at r = 0) have zero measure, so in the finite-
# integration form neither is an unknown at any order: m = 0 keeps H_z (a_z over the disc face), |m| >= 1 drops all
# three. At |m| = 1 the massless axis E_r row reduces to a_z(axis) = 0, i.e. the regularity slaving E_r(0) = -u_0 / m
# (= j E_phi(r_0) / m, second order); H_phi(0) = -j m H_r(r_0) likewise. yee_fields fills both recovered values in, and
# a source term on the axis E_r moves to the u row of cell 0 as -b / m (point_current_rhs). Giving the axis E_r the
# ring volume r_0^2 / 2 instead (an earlier version) made cell 0's discrete Gauss row see a face of area r_0 / 2 at
# r = 0, a spurious axis charge that distorted the near-axis field at every resolution (8/9 of the power of an
# axis-cell 'r' source). |m| >= 2: the fields vanish on the axis as r^(|m| - 1), the recovered values are 0.
# Walls: every wall is the outer grid edge. pec: tangential E = 0 there (the centre value beyond reads 0: a half-cell
# difference); the normal E on the edge is an unknown with half a dual volume. pmc and the natural end beyond a cpml
# layer: tangential H = 0, normal E = 0 (edge unknowns and edge H rows dropped). pec / pmc on the z_lo window edge are
# the antisymmetric / symmetric mirror of a structure symmetric about that plane.
# PML: the complex stretch s of the cpml_axis_profiles centre rows (operators.pml_stretch) integrated into complex
# coordinates (widths w s, centres and edges as running sums), r~ in every metric factor (cylindrical PML). Without
# omega (BORAssembly(..., None)) the metric stays real: the operator of the FDTD twin fdtd_bor (its own CPML).
# Orders: only the m / r entries of C depend on m, C = C0 + m C1, so A(m) = K0 + m K1 + m^2 K2 - k0^2 M with K0 = C0^T W
# C0, K1 = C0^T W C1 + its transpose, K2 = C1^T W C1. The active sets depend on m only through the axis rules (classes m
# = 0, |m| >= 1), so BORAssembly (materials, metric, C0, C1 at one omega) builds the K of each class once and
# BORAssembly.system(m) of any order is a sparse sum (an m sweep reads the materials and assembles once).
# Materials: diagonal eps (eps_rr at E_r, eps_phiphi at E_phi, eps_zz at E_z: rotationally invariant), eps / mu at an
# edge point = the ring-volume (r) / length (z) weighted mean of its non-PEC cells (PEC cells hold no field; all its
# cells when every one is PEC); an E point is PEC when every cell it touches is (cell E_phi: its cell). Sources (FDTD
# sign, as the 2D FDFD): b = +j omega mu0 V J on E_r / E_z rows,
# +omega mu0 V J_phi on u rows. Recovery: E_phi = j u, H = curl E / (-j omega mu0 mu).
# Flux: for unknowns B inside a cut, Im sum_B conj(x) (A x) = Im sum_B conj(x) b reduces (real C, W in lossless
# PML-free cells) to the a points straddling the cut, so pi / (omega mu0) Im(x_B^H b_B) [W] = the outward power
# pi sum [w Re(E_t H_t*)] with E of the cells inside and H on the cut (box_flux): exactly conserved. The recovered axis
# E_r / H_phi are no unknowns, so the axis edge carries no flux weight (the disc [0, r_0] of a z face is already
# accounted for by the balance); the far field keeps them as samples of the regular field.
# Far field: equivalent currents J = n x H, M = -n x E on a closed (r, z) box, phi' integral of exp(-j m phi') by
# Jacobi-Anger, G_n = 2 pi j^n J_n(k rho sin theta) (observer at phi = 0, vacuum outside).
# Driven order sums (an off-axis source = sum_m of its exp(-j m phi) Fourier terms): with P = diag(+1 on E_r, -1 on u,
# +1 on E_z) the phi -> -phi mirror gives C(-m) P = P_a C(m) (P_a = -1 on a_r, a_z), so A(-m) = P A(m) P and one factor
# of A(m) serves both orders: x_-m = P A(m)^-1 P b, its fields those of +m with E_phi, H_r, H_z negated. The 3D field
# is sum_m F_m exp(-j m phi); powers through surfaces of revolution add per order (the phi integral is orthogonal), far-
# field amplitudes add with exp(-j m phi) per direction (order_sum_far_field).
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0, ETA0, MU0
from .boundaries import BCType, cpml_axis_profiles
from .eigenfrequency import EigenSolve, Resonance, solve_resonances
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, ArpackShiftInvertSolver
from ...solvers.host_direct import HOST_DIRECT_FACTORIZATIONS, factor_label
from ...solvers.sparse_direct import size_issue
from .operators import pml_stretch

BOR_SOURCE_COMPONENTS = ('r', 'phi', 'z')
BOR_FIELD_COMPONENTS = ('er', 'ephi', 'ez', 'hr', 'hphi', 'hz')   # centre phasors of centre_fields
_WALL_KINDS = {BCType.PEC: 'pec', BCType.ANTISYMMETRIC: 'pec', BCType.PMC: 'pmc', BCType.SYMMETRIC: 'pmc',
               BCType.CPML: 'pmc'}   # cpml: the grid edge beyond the layer is a natural (pmc) end
_SIZE_REMEDY = ('coarsen the mesh or shrink the (r, z) domain (a pec / pmc face on the equatorial plane of a '
                'symmetric structure halves it)')


@dataclass(frozen=True, eq=False)
class BORMaterials:
    """Host cell values at one omega: complex128 (nr, nz) relative eps (E_phi, and every component left None), eps_rr
    (E_r) and eps_zz (E_z), mu (sigma and poles included, Im < 0 lossy), bool (nr, nz) pec."""
    eps: np.ndarray
    mu: np.ndarray
    pec: np.ndarray
    eps_rr: np.ndarray = None
    eps_zz: np.ndarray = None

    def component(self, name):
        """complex128 cell eps of component 'r' | 'phi' | 'z'."""
        value = {'r': self.eps_rr, 'phi': None, 'z': self.eps_zz}[name]
        return np.asarray(self.eps if value is None else value, dtype=np.complex128)


@dataclass(frozen=True, eq=False)
class _Axis:
    """One axis of n cells: complex (stretched) centres, widths, edge dual lengths and mean coordinates; real twins
    (energy weights, sources); each edge's real share of cell k-1 / k (ring volume on r); wall kinds; window masks."""
    centres: np.ndarray
    widths: np.ndarray
    edge_lengths: np.ndarray
    edge_means: np.ndarray
    real_centres: np.ndarray
    real_widths: np.ndarray
    real_edge_lengths: np.ndarray
    real_edge_means: np.ndarray
    lower_shares: np.ndarray
    upper_shares: np.ndarray
    lo_kind: str
    hi_kind: str
    window_centres: np.ndarray
    window_edges: np.ndarray


@dataclass(frozen=True, eq=False)
class BORSystem:
    """Assembled operator of one omega and azimuthal order on the active unknowns (E_r, u, E_z flattened in that
    order, each (r index, z index) row-major), with what the solves, fields and sources need."""
    omega: float
    azimuthal_order: int
    shape: tuple                # (nr, nz) cells
    operator: object            # scipy CSR complex128 (n, n): K - k0^2 diag(mass)
    mass: np.ndarray            # complex128 (n,) V eps of the active unknowns (B of the eigenproblem)
    curl: object                # scipy CSR (n_a, n): a = curl @ x on every a point (inactive rows still filled)
    a_active: np.ndarray        # bool (n_a,) a points of the operator (others are 0 by the wall / axis rules)
    a_mu: np.ndarray            # complex128 (n_a,) relative mu at the a points
    a_weights: np.ndarray       # complex128 (n_a,) V_a / mu (0 on inactive points): operator = C^T diag(this) C - k0^2 M
    unknown_index: np.ndarray   # int (n,) positions of the active unknowns in the full E vector
    full_size: int
    energy_weights: np.ndarray  # float64 (n,) real ring volumes per radian (|x|^2 weights)
    in_window: np.ndarray       # bool (n,) unknowns inside the physical window
    cell_eps: dict              # 'r' | 'phi' | 'z' -> complex128 (nr, nz) cell eps of that component (absorbed power)
    r_centres: np.ndarray       # float64 (nr,) real radii of the cell centres [m]
    r_widths: np.ndarray        # float64 (nr,)
    z_widths: np.ndarray        # float64 (nz,)
    r_axis: _Axis               # r layout (real twins: flux and far-field weights)
    z_axis: _Axis

    @property
    def n_unknowns(self) -> int:
        return int(self.unknown_index.size)


@dataclass(frozen=True, eq=False)
class BORResonance(Resonance):
    """eigenfrequency.Resonance of one azimuthal order (field = active solution vector of that order's system)."""
    azimuthal_order: int = 0

    def text(self):
        """'m = 15, lambda = 1536.54 nm, Q = 1023'."""
        return f'm = {self.azimuthal_order}, {super().text()}'


def size_issue_for(nr, nz):
    """(level, text) of the host SuperLU size rule for 3 nr nz unknowns (sparse_direct.size_issue), None when fine; CPU."""
    return size_issue('scipy_splu', nr, nz, False, False, unknowns_per_cell=3, remedy=_SIZE_REMEDY)


def _coordinates(widths):
    """(centres, edge dual lengths, edge mean coordinates, edges, dual segment lower / upper ends) from cell widths
    (real or complex), the first edge at 0."""
    edges = np.concatenate((np.zeros(1, dtype=widths.dtype), np.cumsum(widths)))
    centres = edges[:-1] + 0.5 * widths
    lower = np.concatenate((edges[:1], centres))   # dual segment of edge k: lower .. upper
    upper = np.concatenate((centres, edges[-1:]))
    return centres, upper - lower, 0.5 * (lower + upper), edges, lower, upper


def _axis(widths, profile, omega, lo_kind, hi_kind, window, radial):
    """_Axis of float64 widths with the stretch of a CPMLAxisProfile at omega (None: unstretched, the real metric);
    window = slice of the window cells."""
    stretch = 1.0 if omega is None else pml_stretch(profile.sigma_e, profile.kappa_e, profile.alpha_e, omega)
    centres, edge_lengths, edge_means, _, _, _ = _coordinates(widths * stretch)
    real_centres, real_edge_lengths, real_edge_means, edges, lower, upper = _coordinates(widths)
    measure = (lambda values: 0.5 * values ** 2) if radial else (lambda values: values)
    window_centres = np.zeros(widths.size, dtype=bool)
    window_centres[window] = True
    window_edges = np.zeros(widths.size + 1, dtype=bool)
    window_edges[window.start:window.stop + 1] = True
    return _Axis(centres=centres, widths=widths * stretch, edge_lengths=edge_lengths, edge_means=edge_means,
                 real_centres=real_centres, real_widths=widths, real_edge_lengths=real_edge_lengths,
                 real_edge_means=real_edge_means, lower_shares=measure(edges) - measure(lower),
                 upper_shares=measure(upper) - measure(edges), lo_kind=lo_kind, hi_kind=hi_kind,
                 window_centres=window_centres, window_edges=window_edges)


def _edge_sum(values, axis, axis_layout):
    """(n + 1) edge values along `axis` of cell values: each edge's shares of its cells k-1 and k times their values."""
    moved = np.moveaxis(np.asarray(values), axis, 0)
    padding = np.zeros((1,) + moved.shape[1:], dtype=moved.dtype)
    lower = np.concatenate((padding, moved))   # cell k-1 of edge k (row 0 has none)
    upper = np.concatenate((moved, padding))   # cell k of edge k (row n has none)
    shape = (-1,) + (1,) * (moved.ndim - 1)
    return np.moveaxis(axis_layout.lower_shares.reshape(shape) * lower
                       + axis_layout.upper_shares.reshape(shape) * upper, 0, axis)


def _point_mean(values, keep, axes):
    """Share-weighted mean of cell values onto the edge points of axes = ((axis, _Axis), ...) over the cells where
    `keep` (non-PEC) holds; over all of a point's cells when it holds on none."""
    values = np.asarray(values)
    weights = np.asarray(keep, dtype=np.float64)
    kept_sum, kept_weight, plain_sum, plain_weight = values * weights, weights, values, np.ones(values.shape)
    for axis, layout in axes:
        kept_sum, kept_weight = _edge_sum(kept_sum, axis, layout), _edge_sum(kept_weight, axis, layout)
        plain_sum, plain_weight = _edge_sum(plain_sum, axis, layout), _edge_sum(plain_weight, axis, layout)
    some_kept = kept_weight > 0.0
    return np.where(some_kept, kept_sum / np.where(some_kept, kept_weight, 1.0), plain_sum / plain_weight)


def _edge_all(mask, axis):
    """(n + 1) edge mask along `axis`: True when every cell touching the edge is True (an end edge: its one cell)."""
    moved = np.moveaxis(np.asarray(mask, dtype=bool), axis, 0)
    padding = np.ones((1,) + moved.shape[1:], dtype=bool)
    return np.moveaxis(np.concatenate((padding, moved)) & np.concatenate((moved, padding)), 0, axis)


def _difference(n_cells, column_factors, denominators):
    """(n + 1, n) CSR: row k = (c_k f_k - c_{k-1} f_{k-1}) / den_k, f beyond the ends = 0."""
    import scipy.sparse
    rows = np.concatenate((np.arange(n_cells), np.arange(1, n_cells + 1)))
    columns = np.concatenate((np.arange(n_cells), np.arange(n_cells)))
    data = np.concatenate((column_factors / denominators[:-1], -column_factors / denominators[1:]))
    return scipy.sparse.csr_matrix((data.astype(np.complex128), (rows, columns)), shape=(n_cells + 1, n_cells))


def _wall_kind(face_type, face):
    """'pec' / 'pmc' of a BCType on an r_hi / z face; ValueError for periodic / bloch."""
    kind = _WALL_KINDS.get(face_type)
    if kind is None:
        raise ValueError(f"body-of-revolution {face}: {face_type.value} is not supported (cpml, pec or pmc; the axis "
                         f"is x_lo)")
    return kind


def _full_offsets(shape):
    """(E_r size, u size, E_z size) of the full E vector of an (nr, nz) grid."""
    nr, nz = shape
    return (nr + 1) * nz, nr * nz, nr * (nz + 1)


def axis_class(azimuthal_order) -> int:
    """Axis-rule class of an integer order (module header): 0 (m = 0), 1 (|m| >= 1); ValueError else."""
    if isinstance(azimuthal_order, bool) or int(azimuthal_order) != azimuthal_order:
        raise ValueError(f'azimuthal_order must be an integer, got {azimuthal_order!r}')
    return min(abs(int(azimuthal_order)), 1)


def bor_row_scale(system):
    """cuDSS / eigen row scale (|diag K| + k0^2 |M|)^-1/2 of a BORSystem (diag A = diag K - k0^2 M nearly cancels near
    a resonance, so the parts' moduli; 1 on an empty row)."""
    k0_squared = (system.omega / C0) ** 2
    scale = np.abs(system.operator.diagonal() + k0_squared * system.mass) + k0_squared * np.abs(system.mass)
    return 1.0 / np.sqrt(np.where(scale > 0.0, scale, 1.0))


def distinct_orders(orders, label, non_negative=False) -> tuple:
    """Tuple of distinct integer azimuthal orders (None / empty: ()); ValueError naming `label` for a non-integer, a
    repeated or (non_negative) a negative order."""
    orders = tuple(orders or ())
    for order in orders:
        axis_class(order)   # integer check
    orders = tuple(int(order) for order in orders)
    if len(set(orders)) != len(orders) or (non_negative and any(order < 0 for order in orders)):
        rule = 'distinct integers >= 0 (each m > 0 with -m)' if non_negative else 'distinct'
        raise ValueError(f'{label} must be {rule}, got {orders}')
    return orders


def _regular_axis(azimuthal_order) -> bool:
    """True at |m| = 1: the axis E_r / H_phi are recovered from u / H_r of cell 0 (module header)."""
    return abs(int(azimuthal_order)) == 1


@dataclass(frozen=True, eq=False)
class _ClassOperators:
    """Active sets and order-polynomial parts of one axis class (0: m = 0, 1: |m| >= 1): A(m) = k0_part + m k1_part +
    m^2 k2_part - k0^2 M, curl(m) = curl_static + m curl_order (k1 / k2 None for class 0)."""
    unknown_index: np.ndarray
    a_active: np.ndarray
    a_weights: np.ndarray
    curl_static: object
    curl_order: object
    k0_part: object
    k1_part: object
    k2_part: object


class BORAssembly:
    """m-independent parts of the BOR operator at one omega (axes, point materials, ring volumes, curl split C0 + m C1);
    system(m) returns the BORSystem of any azimuthal order, the K parts built once per axis class (module header)."""

    def __init__(self, cell_widths, boundary_config, pml_dx, materials, omega, window=None):
        """cell_widths = (dr_cells, dz_cells) [m] (x = r from the x_lo grid edge, which is the axis), the r_hi / z faces
        of boundary_config (pml_dx = CPML grading cell [m]), BORMaterials, omega [rad/s] (None: the real, unstretched
        metric of the time-domain twin fdtd_bor, whose CPML stretches its derivatives itself; system() and
        order_family() then refuse); window = (slice_r, slice_z) of the physical window (None: the whole grid)."""
        import scipy.sparse
        if omega is not None:
            omega = float(omega)
            if not (math.isfinite(omega) and omega > 0.0):
                raise ValueError(f'BORAssembly needs a finite angular frequency omega > 0 [rad/s], got {omega!r}')
        dr_cells, dz_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
        nr, nz = dr_cells.size, dz_cells.size
        shape = (nr, nz)
        for name in ('eps', 'mu', 'pec', 'eps_rr', 'eps_zz'):
            value = getattr(materials, name)
            if value is not None and np.shape(value) != shape:
                raise ValueError(f'BORMaterials {name} shape {np.shape(value)} does not match the {nr} x {nz} cells')
        if boundary_config.cpml_sides['x_lo'] or boundary_config.periodic_x:
            raise ValueError('body-of-revolution x_lo is the axis r = 0: it must be no cpml or periodic face')
        if boundary_config.periodic_y:
            raise ValueError('body-of-revolution: a periodic z axis is not supported (cpml, pec or pmc faces)')
        window = (slice(0, nr), slice(0, nz)) if window is None else window
        r_axis = _axis(dr_cells, cpml_axis_profiles(boundary_config, nr, 'x', pml_dx), omega, 'axis',
                       _wall_kind(boundary_config.x_hi, 'x_hi'), window[0], radial=True)
        z_axis = _axis(dz_cells, cpml_axis_profiles(boundary_config, nz, 'y', pml_dx), omega,
                       _wall_kind(boundary_config.y_lo, 'y_lo'), _wall_kind(boundary_config.y_hi, 'y_hi'), window[1],
                       radial=False)
        pec = np.asarray(materials.pec, dtype=bool)
        cell_eps = {name: materials.component(name) for name in ('r', 'phi', 'z')}
        mu = np.asarray(materials.mu, dtype=np.complex128)
        for name, values in (('eps_rr', cell_eps['r']), ('eps', cell_eps['phi']), ('eps_zz', cell_eps['z']),
                             ('mu', mu)):
            if not (np.all(np.isfinite(values[~pec])) and np.all(values[~pec] != 0)):
                raise ValueError(f'BORMaterials {name} must be finite and non-zero on non-PEC cells')
        self.omega, self.shape, self.pec = omega, shape, pec
        self.r_axis, self.z_axis = r_axis, z_axis
        self.cell_eps, self.dr_cells, self.dz_cells = cell_eps, dr_cells, dz_cells

        # point materials: means over the non-PEC cells (module header)
        self.a_mu = self.a_point_means(mu)

        # ring volumes per radian (E_r / a_z points, cells, E_z / a_r points, a_phi points): complex (stretched) for
        # the operator, real for the energy weights
        def volumes(centres_r, widths_r, edge_means_r, edge_lengths_r, widths_z, edge_lengths_z):
            cell_r, edge_r = centres_r * widths_r, edge_means_r * edge_lengths_r
            return (np.outer(edge_r, widths_z), np.outer(cell_r, widths_z), np.outer(cell_r, edge_lengths_z),
                    np.outer(edge_r, edge_lengths_z))

        v_edge_r, v_cell, v_edge_z, v_corner = volumes(r_axis.centres, r_axis.widths, r_axis.edge_means,
                                                       r_axis.edge_lengths, z_axis.widths, z_axis.edge_lengths)
        w_edge_r, w_cell, w_edge_z, _ = volumes(r_axis.real_centres, r_axis.real_widths, r_axis.real_edge_means,
                                                r_axis.real_edge_lengths, z_axis.real_widths, z_axis.real_edge_lengths)
        self.a_volumes = np.concatenate((v_edge_z.ravel(), v_edge_r.ravel(), v_corner.ravel()))
        self.full_mass = (np.concatenate((v_edge_r.ravel(), v_cell.ravel(), v_edge_z.ravel()))
                          * self.e_point_means(cell_eps['r'], cell_eps['phi'], cell_eps['z']))
        self.full_energy_weights = np.abs(np.concatenate((w_edge_r.ravel(), w_cell.ravel(), w_edge_z.ravel())))
        self.full_in_window = np.concatenate((np.outer(r_axis.window_edges, z_axis.window_centres).ravel(),
                                              np.outer(r_axis.window_centres, z_axis.window_centres).ravel(),
                                              np.outer(r_axis.window_centres, z_axis.window_edges).ravel()))

        # curl C = C0 + m C1: rows a_r (nr, nz+1), a_z (nr+1, nz), a_phi (nr+1, nz+1); columns E_r (nr+1, nz), u (nr,
        # nz), E_z (nr, nz+1); C1 holds the m / r entries at m = 1
        identity, kron, diags = scipy.sparse.identity, scipy.sparse.kron, scipy.sparse.diags
        n_er, n_u, n_ez = _full_offsets(shape)
        n_ar, n_az, n_aphi = nr * (nz + 1), (nr + 1) * nz, (nr + 1) * (nz + 1)

        def zeros(n_rows, n_columns):
            return scipy.sparse.csr_matrix((n_rows, n_columns), dtype=np.complex128)

        gradient_z = _difference(nz, np.ones(nz), z_axis.edge_lengths)
        gradient_r = _difference(nr, np.ones(nr), r_axis.edge_lengths)
        metric_gradient_r = _difference(nr, r_axis.centres, r_axis.edge_means * r_axis.edge_lengths)
        self._curl_static = scipy.sparse.bmat([
            [zeros(n_ar, n_er), kron(identity(nr), gradient_z), zeros(n_ar, n_ez)],
            [zeros(n_az, n_er), kron(metric_gradient_r, identity(nz)), zeros(n_az, n_ez)],
            [kron(identity(nr + 1), gradient_z), zeros(n_aphi, n_u), -kron(gradient_r, identity(nz + 1))],
        ], format='csc')
        self._curl_order = scipy.sparse.bmat([
            [zeros(n_ar, n_er), zeros(n_ar, n_u), kron(diags(1.0 / r_axis.centres), identity(nz + 1))],
            [kron(diags(1.0 / r_axis.edge_means), identity(nz)), zeros(n_az, n_u), zeros(n_az, n_ez)],
            [zeros(n_aphi, n_er), zeros(n_aphi, n_u), zeros(n_aphi, n_ez)],
        ], format='csc')
        self._classes = {}

    def e_point_means(self, r_values, phi_values, z_values):
        """Full E layout (E_r, E_phi, E_z points) of cell values per component (eps_rr, eps_phiphi, eps_zz or a
        conductivity): the cell value at E_phi, the ring-volume / length weighted mean over non-PEC cells at E_r / E_z."""
        free = ~self.pec
        return np.concatenate((_point_mean(r_values, free, ((0, self.r_axis),)).ravel(),
                               np.asarray(phi_values).ravel(),
                               _point_mean(z_values, free, ((1, self.z_axis),)).ravel()))

    def a_point_means(self, values):
        """Full a layout (H_r, H_z, H_phi points) of cell values (mu, a magnetic conductivity): means over the non-PEC
        cells around each point."""
        free = ~self.pec
        return np.concatenate((_point_mean(values, free, ((1, self.z_axis),)).ravel(),
                               _point_mean(values, free, ((0, self.r_axis),)).ravel(),
                               _point_mean(values, free, ((0, self.r_axis), (1, self.z_axis))).ravel()))

    def curl_parts(self, azimuthal_order):
        """(C0, C1 on the full E and a layouts with the inactive columns and rows of the order's axis class zeroed
        (scipy CSR complex128; curl = C0 + m C1), bool active E mask, bool active a mask): fdtd_bor's operator."""
        import scipy.sparse
        parts = self._class_operators(axis_class(azimuthal_order))
        active_e = np.zeros(self.full_mass.size, dtype=bool)
        active_e[parts.unknown_index] = True
        rows = scipy.sparse.diags(parts.a_active.astype(np.float64))
        columns = scipy.sparse.diags(active_e.astype(np.float64))
        return ((rows @ self._curl_static @ columns).tocsr(), (rows @ self._curl_order @ columns).tocsr(), active_e,
                parts.a_active)

    def _require_omega(self):
        if self.omega is None:
            raise ValueError('BORAssembly built without omega (the real time-domain metric) has no frequency-domain '
                             'system')

    def _class_operators(self, class_index):
        """_ClassOperators of axis class 0 | 1, built on first use."""
        import scipy.sparse
        if class_index in self._classes:
            return self._classes[class_index]
        nr, nz = self.shape
        # active edges: interior always; an end edge only on a pec wall; on the axis only the m = 0 H_z disc face
        # (module header: the axis E_r and H_phi have zero dual measure at every order)
        z_edges = np.ones(nz + 1, dtype=bool)
        z_edges[0], z_edges[-1] = self.z_axis.lo_kind == 'pec', self.z_axis.hi_kind == 'pec'

        def r_edges(on_axis):
            edges = np.ones(nr + 1, dtype=bool)
            edges[0], edges[-1] = on_axis, self.r_axis.hi_kind == 'pec'
            return edges

        er_r_edges, az_r_edges, aphi_r_edges = r_edges(False), r_edges(class_index == 0), r_edges(False)
        active_e = np.concatenate(((np.outer(er_r_edges, np.ones(nz, dtype=bool)) & ~_edge_all(self.pec, 0)).ravel(),
                                   (~self.pec).ravel(),
                                   (np.outer(np.ones(nr, dtype=bool), z_edges) & ~_edge_all(self.pec, 1)).ravel()))
        a_active = np.concatenate((np.outer(np.ones(nr, dtype=bool), z_edges).ravel(),
                                   np.outer(az_r_edges, np.ones(nz, dtype=bool)).ravel(),
                                   np.outer(aphi_r_edges, z_edges).ravel()))
        unknown_index = np.flatnonzero(active_e)
        a_weights = self.a_volumes / self.a_mu * a_active
        weights = scipy.sparse.diags(a_weights)
        curl_static = self._curl_static[:, unknown_index].tocsr()
        weighted_static = weights @ curl_static
        k0_part = (curl_static.T @ weighted_static).tocsr()
        curl_order = k1_part = k2_part = None
        if class_index > 0:
            curl_order = self._curl_order[:, unknown_index].tocsr()
            weighted_order = weights @ curl_order
            cross = curl_static.T @ weighted_order
            k1_part = (cross + cross.T).tocsr()
            k2_part = (curl_order.T @ weighted_order).tocsr()
        operators = _ClassOperators(unknown_index=unknown_index, a_active=a_active, a_weights=a_weights,
                                    curl_static=curl_static, curl_order=curl_order, k0_part=k0_part,
                                    k1_part=k1_part, k2_part=k2_part)
        self._classes[class_index] = operators
        return operators

    def system(self, azimuthal_order) -> 'BORSystem':
        """BORSystem of an integer azimuthal order: K0 + m K1 + m^2 K2 - k0^2 M of its axis class."""
        import scipy.sparse
        self._require_omega()
        class_index = axis_class(azimuthal_order)
        order = int(azimuthal_order)
        parts = self._class_operators(class_index)
        unknown_index = parts.unknown_index
        mass = self.full_mass[unknown_index]
        stiffness, curl = parts.k0_part, parts.curl_static
        if order != 0:
            stiffness = parts.k0_part + order * parts.k1_part + order ** 2 * parts.k2_part
            curl = (parts.curl_static + order * parts.curl_order).tocsr()
        operator = (stiffness - (self.omega / C0) ** 2 * scipy.sparse.diags(mass)).tocsr()
        return BORSystem(omega=self.omega, azimuthal_order=order, shape=self.shape, operator=operator, mass=mass,
                         curl=curl, a_active=parts.a_active, a_mu=self.a_mu, a_weights=parts.a_weights,
                         unknown_index=unknown_index, full_size=int(self.full_mass.size),
                         energy_weights=self.full_energy_weights[unknown_index],
                         in_window=self.full_in_window[unknown_index], cell_eps=self.cell_eps,
                         r_centres=self.r_axis.real_centres, r_widths=self.dr_cells, z_widths=self.dz_cells,
                         r_axis=self.r_axis, z_axis=self.z_axis)

    def order_family(self, class_index):
        """(K0 - k0^2 diag(mass), (K1, K2) (() for class 0), unknown_index) of axis class 0 | 1 (scipy CSR
        complex128): A(m) = base + m K1 + m^2 K2 for every order m of the class (the line-block shift family)."""
        import scipy.sparse
        self._require_omega()
        parts = self._class_operators(class_index)
        mass = self.full_mass[parts.unknown_index]
        base =(parts.k0_part - (self.omega / C0) ** 2 * scipy.sparse.diags(mass)).tocsr()
        terms = () if class_index == 0 else (parts.k1_part, parts.k2_part)
        return base, terms, parts.unknown_index


def assemble_bor(cell_widths, boundary_config, pml_dx, materials, omega, azimuthal_order, window=None):
    """BORSystem at omega [rad/s] and one azimuthal order: BORAssembly(...).system(azimuthal_order) (arguments as
    BORAssembly)."""
    axis_class(azimuthal_order)   # the integer check before any assembly work
    return BORAssembly(cell_widths, boundary_config, pml_dx, materials, omega, window=window).system(azimuthal_order)


def point_current_rhs(system, ix, iz, component, current, order=None):
    """Active rhs (complex128 (n,)) of a current element in cell (ix, iz), FDTD sign, ~ exp(-j m phi): 'phi' J_phi =
    I / (w h) (a loop of current I at m = 0), 'z' / 'r' the current I through the cell's ring cross-section 2 pi r w /
    2 pi r h (m = 0: total current; the axis cell's 'z' is an on-axis element of moment I h), split on its two edges.
    An 'r' half on the axis edge moves to the u row of cell 0 as -b / m at |m| = 1 (the regular transverse element:
    module header) and is lost at other orders (no field there). order: the m of that move (default the system's; the
    -m partner of an order sum passes -m, whose active set is the same)."""
    if component not in BOR_SOURCE_COMPONENTS:
        raise ValueError(f'source component must be one of {BOR_SOURCE_COMPONENTS}, got {component!r}')
    nr, nz = system.shape
    ix, iz = int(ix), int(iz)
    if not (0 <= ix < nr and 0 <= iz < nz):
        raise ValueError(f'source cell ({ix}, {iz}) is outside the {nr} x {nz} grid')
    n_er, n_u, _ = _full_offsets(system.shape)
    full = np.zeros(system.full_size, dtype=np.complex128)
    omega_mu = system.omega * MU0
    current = complex(current)
    if component == 'phi':
        full[n_er + ix * nz + iz] = omega_mu * system.r_centres[ix] * current
    elif component == 'z':
        for edge in (iz, iz + 1):
            full[n_er + n_u + ix * (nz + 1) + edge] += 1j * omega_mu * current * system.z_widths[iz] / (4.0 * math.pi)
    else:
        half = 1j * omega_mu * current * system.r_widths[ix] / (4.0 * math.pi)
        for edge in (ix, ix + 1):
            full[edge * nz + iz] += half
        axis_order = system.azimuthal_order if order is None else int(order)
        if axis_class(axis_order) != axis_class(system.azimuthal_order):
            raise ValueError(f'point_current_rhs order {axis_order} is not of the axis class of the system order '
                             f'{system.azimuthal_order}')
        if ix == 0 and _regular_axis(axis_order):
            full[n_er + iz] -= half / axis_order
    return full[system.unknown_index]


def yee_fields(system, solution):
    """complex128 phasors on their Yee points of an active solution vector (inactive points 0; at |m| = 1 the axis E_r
    / H_phi recovered from cell 0, module header): er (nr+1, nz), ephi (nr, nz), ez (nr, nz+1), hr (nr, nz+1), hz
    (nr+1, nz), hphi (nr+1, nz+1)."""
    nr, nz = system.shape
    n_er, n_u, _ = _full_offsets(system.shape)
    solution = np.asarray(solution, dtype=np.complex128)
    full = np.zeros(system.full_size, dtype=np.complex128)
    full[system.unknown_index] = solution
    h_points = (system.curl @ solution) * system.a_active / (system.omega * MU0 * system.a_mu)
    n_ar, n_az = nr * (nz + 1), (nr + 1) * nz
    yee = {'er': full[:n_er].reshape(nr + 1, nz), 'ephi': 1j * full[n_er:n_er + n_u].reshape(nr, nz),
           'ez': full[n_er + n_u:].reshape(nr, nz + 1), 'hr': h_points[:n_ar].reshape(nr, nz + 1),
           'hz': -h_points[n_ar:n_ar + n_az].reshape(nr + 1, nz),
           'hphi': 1j * h_points[n_ar + n_az:].reshape(nr + 1, nz + 1)}
    if _regular_axis(system.azimuthal_order):
        order = system.azimuthal_order
        yee['er'][0] = 1j * yee['ephi'][0] / order
        yee['hphi'][0] = -1j * order * yee['hr'][0]
    return yee


def centre_fields(system, solution):
    """(dict er, ephi, ez, hr, hphi, hz of complex128 (nr, nz) phasors at the cell centres (edge values averaged),
    dict 'r' | 'phi' | 'z' of float64 (nr, nz) cell mean |E_c|^2 over that component's points) of a solution."""
    return yee_centres(yee_fields(system, solution))


def yee_centres(yee):
    """centre_fields of a yee_fields dict (numpy or cupy, phasors or real time-domain samples): edge values averaged
    onto the cell centres, and the per-component cell mean |E_c|^2."""
    er, ez, hr, hz, hphi = (yee[name] for name in ('er', 'ez', 'hr', 'hz', 'hphi'))
    fields = {'er': 0.5 * (er[:-1] + er[1:]), 'ephi': yee['ephi'], 'ez': 0.5 * (ez[:, :-1] + ez[:, 1:]),
              'hr': 0.5 * (hr[:, :-1] + hr[:, 1:]), 'hz': 0.5 * (hz[:-1] + hz[1:]),
              'hphi': 0.25 * (hphi[:-1, :-1] + hphi[1:, :-1] + hphi[:-1, 1:] + hphi[1:, 1:])}
    e_squared = {'r': 0.5 * (np.abs(er[:-1]) ** 2 + np.abs(er[1:]) ** 2), 'phi': np.abs(yee['ephi']) ** 2,
                 'z': 0.5 * (np.abs(ez[:, :-1]) ** 2 + np.abs(ez[:, 1:]) ** 2)}
    return fields, e_squared


# ---- flux (module header): E of the cells inside the cut, H on it; per radian x 2 pi, x 1/2 time average ----
def span_edge_weights(axis_layout, lo, hi):
    """float64 (hi - lo + 1,) real weights of edges lo..hi inside the cell span [lo, hi): each edge's dual share in the
    span (ring area per radian along r, length along z)."""
    edges = np.arange(lo, hi + 1)
    return (axis_layout.upper_shares[edges] * (edges < hi) + axis_layout.lower_shares[edges] * (edges > lo))


def _z_face_power(system, fields, edge, row, cells, edge_weights):
    """pi sum [A_k Re(E_r H_phi*) - r w Re(E_phi H_r*)] [W] toward +z through z-edge `edge` over r cells [lo, hi): E of
    `row`, H on the edge, E_r / H_phi edges lo..hi weighted by edge_weights."""
    lo, hi = cells
    r_axis = system.r_axis
    cell_areas = r_axis.real_centres[lo:hi] * r_axis.real_widths[lo:hi]
    radial = np.sum(edge_weights * (fields['er'][lo:hi + 1, row] * np.conj(fields['hphi'][lo:hi + 1, edge])).real)
    azimuthal = np.sum(cell_areas * (fields['ephi'][lo:hi, row] * np.conj(fields['hr'][lo:hi, edge])).real)
    return math.pi * float(radial - azimuthal)


def _r_face_power(system, fields, edge, column, cells, edge_weights):
    """pi [r_c sum h Re(E_phi H_z*) - rbar_k sum L Re(E_z H_phi*)] [W] toward +r through r-edge k = `edge` over z cells
    [lo, hi): E of cell column c = `column` (whose centre radius r_c weights E_phi), H on the edge."""
    lo, hi = cells
    r_axis, z_axis = system.r_axis, system.z_axis
    azimuthal = r_axis.real_centres[column] * np.sum(
        z_axis.real_widths[lo:hi] * (fields['ephi'][column, lo:hi] * np.conj(fields['hz'][edge, lo:hi])).real)
    axial = r_axis.real_edge_means[edge] * np.sum(
        edge_weights * (fields['ez'][column, lo:hi + 1] * np.conj(fields['hphi'][edge, lo:hi + 1])).real)
    return math.pi * float(azimuthal - axial)


def line_flux(system, fields, normal, line_cell, cells):
    """Power [W] through the hi face of line cell `line_cell` along `normal` 'z' (an annulus over r cells [lo, hi))
    or 'r' (a cylinder over z cells [lo, hi)), > 0 toward +normal: E of the line cells, H on the face (the exactly
    conserved pairing; span ends by their share); ValueError when the face is the outer grid edge."""
    if normal not in ('z', 'r'):
        raise ValueError(f"line_flux normal must be 'z' or 'r', got {normal!r}")
    n_normal = system.shape[1] if normal == 'z' else system.shape[0]
    if not 0 <= line_cell < n_normal - 1:
        raise ValueError(f'flux line cell {line_cell} along {normal}: its hi face must be an interior edge of the '
                         f'{n_normal} cells')
    lo, hi = cells
    if normal == 'z':
        edge_weights = span_edge_weights(system.r_axis, lo, hi)
        if lo == 0:
            edge_weights[0] = 0.0   # the axis E_r / H_phi are recovered values, no flux unknowns (module header)
        return _z_face_power(system, fields, line_cell + 1, line_cell, cells, edge_weights)
    return _r_face_power(system, fields, line_cell + 1, line_cell, cells, span_edge_weights(system.z_axis, lo, hi))


def box_flux(system, fields, r_cells, z_cells):
    """Power [W] out of the cell box [r_lo, r_hi) x [z_lo, z_hi) (an inner face only for r_lo > 0): the exact discrete
    flux, = source_power of the sources inside when the faces lie in lossless, PML-free cells (module header)."""
    (i0, i1), (j0, j1) = r_cells, z_cells
    r_axis, z_axis = system.r_axis, system.z_axis
    # E_r / E_z unknowns inside: interior edges with their full dual shares (the axis edge r_lo = 0 holds no unknown:
    # its E_r / H_phi are recovered values, module header)
    r_weights = r_axis.real_edge_means[i0:i1 + 1] * r_axis.real_edge_lengths[i0:i1 + 1]
    r_weights[0] = r_weights[-1] = 0.0
    z_weights = z_axis.real_edge_lengths[j0:j1 + 1].copy()
    z_weights[0] = z_weights[-1] = 0.0
    power = (_z_face_power(system, fields, j1, j1 - 1, r_cells, r_weights)
             - _z_face_power(system, fields, j0, j0, r_cells, r_weights)
             + _r_face_power(system, fields, i1, i1 - 1, z_cells, z_weights))
    if i0 > 0:
        power -= _r_face_power(system, fields, i0, i0, z_cells, z_weights)
    return power


def source_power(system, solution, rhs):
    """Time-averaged power [W] delivered by the sources of `rhs` (FDTD sign of point_current_rhs, 2 pi ring weights):
    pi / (omega mu0) Im(x^H b)."""
    return math.pi / (system.omega * MU0) * float(np.vdot(solution, rhs).imag)


# ---- driven order sums (module header) ----
_MIRROR_ODD_COMPONENTS = ('ephi', 'hr', 'hz')   # negated by the phi -> -phi mirror


def order_parity_signs(system):
    """float64 (n,) diagonal of P on the active unknowns (+1 E_r, -1 u, +1 E_z): A(-m) = P A(m) P."""
    n_er, n_u, _ = _full_offsets(system.shape)
    index = system.unknown_index
    return np.where((index >= n_er) & (index < n_er + n_u), -1.0, 1.0)


def mirrored_yee(yee):
    """yee_fields at order -m of x_-m = P y from those of y at +m: ephi, hr, hz negated (the phi -> -phi mirror)."""
    return {name: -values if name in _MIRROR_ODD_COMPONENTS else values for name, values in yee.items()}


def order_pair_fields(system, rhs, solve_host, minus_rhs=None):
    """[(signed order, yee_fields)] of a source term rhs (active) at m = system.azimuthal_order >= 0 and, for m > 0, of
    minus_rhs (its -m term; None = the same as rhs) at -m by the same factor (solve_host(b) solves system.operator x =
    b): x_-m = P A(m)^-1 P minus_rhs, one solve when P minus_rhs = +-rhs (e.g. sources all r / z or all phi)."""
    order = system.azimuthal_order
    if order < 0:
        raise ValueError(f'order_pair_fields needs m >= 0 (its -m partner is derived), got {order}')
    plus = yee_fields(system, solve_host(rhs))
    terms = [(order, plus)]
    if order == 0:
        return terms
    mirrored_rhs = order_parity_signs(system) * (rhs if minus_rhs is None else minus_rhs)
    if np.array_equal(mirrored_rhs, rhs):
        partner = plus
    elif np.array_equal(mirrored_rhs, -rhs):
        partner = {name: -values for name, values in plus.items()}
    else:
        partner = yee_fields(system, solve_host(mirrored_rhs))
    terms.append((-order, mirrored_yee(partner)))
    return terms


# ---- far field (module header) ----
def _interpolated(values, index, widths, axis):
    """Linear interpolation along `axis` of centre values onto the edge `index` between cells index-1 and index."""
    lower, upper = np.take(values, index - 1, axis=axis), np.take(values, index, axis=axis)
    w_lower, w_upper = widths[index - 1], widths[index]
    return (lower * w_upper + upper * w_lower) / (w_lower + w_upper)


def _surface_currents(normal, e_field, h_field):
    """(J, M) complex (n, 3) cylindrical (r, phi, z) of J = n x H, M = -n x E on a face of normal '+z' | '-z' | '+r'
    | '-r' (field arrays (n, 3) in the same components)."""
    sign = 1.0 if normal[0] == '+' else -1.0
    current_j, current_m = np.zeros_like(h_field), np.zeros_like(e_field)
    if normal[1] == 'z':     # z x r = phi, z x phi = -r
        current_j[:, 0], current_j[:, 1] = -h_field[:, 1], h_field[:, 0]
        current_m[:, 0], current_m[:, 1] = e_field[:, 1], -e_field[:, 0]
    else:                    # r x phi = z, r x z = -phi
        current_j[:, 1], current_j[:, 2] = -h_field[:, 2], h_field[:, 1]
        current_m[:, 1], current_m[:, 2] = e_field[:, 2], -e_field[:, 1]
    return sign * current_j, sign * current_m


def far_field_samples(system, fields, r_cells, z_cells, mirror_sign=None):
    """Equivalent-current samples (rho, z [m], weight [m^2 per radian], J, M complex (n, 3) cylindrical) on the box of
    cells [r_lo, r_hi) x [z_lo, z_hi): its z faces, outer (and for r_lo > 0 inner) cylinder, tangential E interpolated
    onto the face, H on its Yee points. mirror_sign +1 / -1 (pmc / pec wall on grid edge 0 = the box bottom): the
    bottom face is replaced by the mirror images of the others (E_r, E_phi, H_z even for pmc, odd for pec)."""
    (i0, i1), (j0, j1) = r_cells, z_cells
    nr, nz = system.shape
    if not (0 <= i0 < i1 <= nr - 1 and 0 <= j0 < j1 <= nz - 1):
        raise ValueError(f'far-field box cells [{i0}, {i1}) x [{j0}, {j1}) need one cell beyond r_hi and z_hi '
                         f'inside the {nr} x {nz} grid')
    if mirror_sign is None and j0 < 1:
        raise ValueError(f'far-field box cells [{j0}, {j1}) along z need one cell below the box (or a pec / pmc z_lo '
                         'wall at the box bottom)')
    r_axis, z_axis = system.r_axis, system.z_axis
    r_edges = np.concatenate(([0.0], np.cumsum(r_axis.real_widths)))
    z_edges = np.concatenate(([0.0], np.cumsum(z_axis.real_widths)))
    faces = []   # (normal, rho, z, weight, E (n, 3), H (n, 3))

    def z_face(edge, normal):
        e_r = _interpolated(fields['er'][i0:i1 + 1], edge, z_axis.real_widths, 1)
        e_phi = _interpolated(fields['ephi'][i0:i1], edge, z_axis.real_widths, 1)
        zeros_edges, zeros_cells = np.zeros(i1 - i0 + 1, complex), np.zeros(i1 - i0, complex)
        faces.append((normal, r_edges[i0:i1 + 1], np.full(i1 - i0 + 1, z_edges[edge]),
                      span_edge_weights(r_axis, i0, i1), np.stack((e_r, zeros_edges, zeros_edges), axis=1),
                      np.stack((zeros_edges, fields['hphi'][i0:i1 + 1, edge], zeros_edges), axis=1)))
        faces.append((normal, r_axis.real_centres[i0:i1], np.full(i1 - i0, z_edges[edge]),
                      r_axis.real_centres[i0:i1] * r_axis.real_widths[i0:i1],
                      np.stack((zeros_cells, e_phi, zeros_cells), axis=1),
                      np.stack((fields['hr'][i0:i1, edge], zeros_cells, zeros_cells), axis=1)))

    def r_face(edge, normal):
        radius = r_edges[edge]
        e_phi = _interpolated(fields['ephi'][:, j0:j1], edge, r_axis.real_widths, 0)
        e_z = _interpolated(fields['ez'][:, j0:j1 + 1], edge, r_axis.real_widths, 0)
        zeros_cells, zeros_edges = np.zeros(j1 - j0, complex), np.zeros(j1 - j0 + 1, complex)
        faces.append((normal, np.full(j1 - j0, radius), z_axis.real_centres[j0:j1],
                      radius * z_axis.real_widths[j0:j1], np.stack((zeros_cells, e_phi, zeros_cells), axis=1),
                      np.stack((zeros_cells, zeros_cells, fields['hz'][edge, j0:j1]), axis=1)))
        faces.append((normal, np.full(j1 - j0 + 1, radius), z_edges[j0:j1 + 1],
                      radius * span_edge_weights(z_axis, j0, j1), np.stack((zeros_edges, zeros_edges, e_z), axis=1),
                      np.stack((zeros_edges, fields['hphi'][edge, j0:j1 + 1], zeros_edges), axis=1)))

    z_face(j1, '+z')
    r_face(i1, '+r')
    if i0 > 0:
        r_face(i0, '-r')
    if mirror_sign is None:
        z_face(j0, '-z')
    else:   # images through z = 0: E_t (r, phi) x s, E_z x -s; H_t x -s, H_z x s
        e_parity, h_parity = np.array([1.0, 1.0, -1.0]) * mirror_sign, np.array([-1.0, -1.0, 1.0]) * mirror_sign
        faces.extend([({'+z': '-z', '-z': '+z'}.get(normal, normal), rho, -z, weight, e_field * e_parity,
                       h_field * h_parity) for normal, rho, z, weight, e_field, h_field in list(faces)])
    currents = [_surface_currents(normal, e_field, h_field) for normal, _, _, _, e_field, h_field in faces]
    return (np.concatenate([face[1] for face in faces]), np.concatenate([face[2] for face in faces]),
            np.concatenate([face[3] for face in faces]), np.concatenate([pair[0] for pair in currents]),
            np.concatenate([pair[1] for pair in currents]))


class _RadiationKernel:
    """Order-free factors of the radiation integrals of one far_field_samples geometry toward theta: the z phase times
    the weights, and J_|n|(k rho sin theta) evaluated once per |n| (the Bessel calls dominate the far field)."""

    def __init__(self, wavenumber, theta, samples_rho, samples_z, weights):
        self.argument = wavenumber * np.outer(np.sin(theta), samples_rho)
        self.phase = np.exp(1j * wavenumber * np.outer(np.cos(theta), samples_z)) * weights
        self.cos_theta, self.sin_theta = np.cos(theta), np.sin(theta)
        self._bessel = {}

    def phi_integral(self, n):
        """Integral of exp(-j n phi') exp(j a cos phi') over phi' = 2 pi j^n J_n(a); J_-n = (-1)^n J_n."""
        import scipy.special
        magnitude = abs(n)
        if magnitude not in self._bessel:
            self._bessel[magnitude] = scipy.special.jv(magnitude, self.argument)
        sign = -1.0 if n < 0 and magnitude % 2 else 1.0
        return (2.0 * math.pi * sign * 1j ** n) * self._bessel[magnitude]


def _radiation_integrals(order, kernel, currents):
    """[(N_theta, N_phi) complex (n_theta,)] per current of `currents` (cylindrical (n, 3), ~ exp(-j m phi')) toward
    (theta, phi = 0): sum weight exp(j k z cos theta) times the phi' integral of the current times exp(j k rho sin
    theta cos phi')."""
    lower, middle, upper = (kernel.phi_integral(n) for n in (order - 1, order, order + 1))
    phased_cosine = kernel.phase * (0.5 * (lower + upper))   # cos phi' weighted
    phased_sine = kernel.phase * ((lower - upper) / 2j)      # sin phi' weighted
    phased_middle = kernel.phase * middle
    integrals = []
    for current in currents:
        n_theta = (kernel.cos_theta * (phased_cosine @ current[:, 0] - phased_sine @ current[:, 1])
                   - kernel.sin_theta * (phased_middle @ current[:, 2]))
        integrals.append((n_theta, phased_sine @ current[:, 0] + phased_cosine @ current[:, 1]))
    return integrals


def far_field_amplitudes(order, omega, samples, theta):
    """(E_theta, E_phi) complex (n_theta,) far-field amplitudes of far_field_samples at azimuthal order `order` toward
    polar angles theta [rad from +z] at phi = 0, scaled so that |amplitude|^2 is that polarisation's W/sr (toward phi:
    times exp(-j m phi) in the local theta / phi basis)."""
    wavenumber = float(omega) / C0
    kernel = _RadiationKernel(wavenumber, np.asarray(theta, dtype=np.float64), *samples[:3])
    return _kernel_amplitudes(order, wavenumber, samples, kernel)


def _kernel_amplitudes(order, wavenumber, samples, kernel):
    """far_field_amplitudes on a _RadiationKernel of the samples' geometry."""
    (n_theta, n_phi), (l_theta, l_phi) = _radiation_integrals(int(order), kernel, samples[3:])
    root_scale = wavenumber / (4.0 * math.sqrt(2.0) * math.pi * math.sqrt(float(ETA0)))   # sqrt(k^2 / (32 pi^2 eta0))
    return root_scale * (float(ETA0) * n_theta + l_phi), root_scale * (float(ETA0) * n_phi - l_theta)


def far_field_power(order, omega, samples, theta):
    """(total, theta-polarised, phi-polarised) radiated power per steradian [W/sr] = r^2 |E|^2 / (2 eta0) toward polar
    angles theta [rad from +z] of far_field_samples (vacuum outside the box; independent of phi)."""
    amplitude_theta, amplitude_phi = far_field_amplitudes(order, omega, samples, theta)
    power_theta, power_phi = np.abs(amplitude_theta) ** 2, np.abs(amplitude_phi) ** 2
    return power_theta + power_phi, power_theta, power_phi


def order_sum_far_field(order_samples, omega, theta, phi):
    """(total, theta-, phi-polarised W/sr toward (theta, phi) of the summed field sum_m F_m exp(-j m phi), {|m|: the phi
    mean W/sr of orders +m and -m}) of order_samples = [(signed order, far_field_samples)]; phi broadcasts to theta."""
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.broadcast_to(np.asarray(phi, dtype=np.float64), theta.shape)
    wavenumber = float(omega) / C0
    total_theta = np.zeros(theta.shape, dtype=np.complex128)
    total_phi = np.zeros(theta.shape, dtype=np.complex128)
    order_means = {}
    kernels = {}   # one per sample geometry (rho, z, weight): the terms of one box share its Bessel values
    for order, samples in order_samples:
        geometry = b''.join(np.ascontiguousarray(part, dtype=np.float64).tobytes() for part in samples[:3])
        if geometry not in kernels:
            kernels[geometry] = _RadiationKernel(wavenumber, theta, *samples[:3])
        amplitude_theta, amplitude_phi = _kernel_amplitudes(order, wavenumber, samples, kernels[geometry])
        rotation = np.exp(-1j * order * phi)
        total_theta += amplitude_theta * rotation
        total_phi += amplitude_phi * rotation
        key = abs(int(order))
        order_means[key] = order_means.get(key, 0.0) + np.abs(amplitude_theta) ** 2 + np.abs(amplitude_phi) ** 2
    power_theta, power_phi = np.abs(total_theta) ** 2, np.abs(total_phi) ** 2
    return power_theta + power_phi, power_theta, power_phi, order_means


def solve_bor_resonances(system, n_resonances, eigen_solver=None, factorization=None):
    """eigenfrequency.EigenSolve of up to n_resonances BORResonances nearest system.omega: squared-Cayley shift-invert
    (eigenfrequency.solve_resonances) by eigen_solver (solvers.arpack_shift_invert, None = default settings) on
    `factorization` of system.operator (solve_host, host_bytes, description, e.g. a line-block factor; None = an own
    SuperLU / cuDSS / MUMPS / PARDISO factor at k_t^2 (1 + SHIFT_RELATIVE_OFFSET)); fields = active solution vectors, 1
    real at their largest |component|. The FIT operator is complex symmetric (D = I): a cudss / mumps / pardiso node
    factors an LDL^T, a mixed one (cudss_mixed, mumps_mixed, pardiso_mixed) adds the float64 Rayleigh polish."""
    eigen_solver = ArpackShiftInvertSolver() if eigen_solver is None else eigen_solver
    warnings = []
    # host SuperLU size rule: only for an own SuperLU / MUMPS / PARDISO factor (a passed line-block factor has no host LU)
    if factorization is None and eigen_solver.factorization not in CUDSS_FACTORIZATIONS:
        issue = size_issue_for(*system.shape)
        if issue is not None:
            if issue[0] == 'error':
                raise ValueError(f'eigenfrequency solve: {issue[1]}')
            warnings.append(issue[1])
    if system.n_unknowns < 4:
        raise ValueError(f'solve_bor_resonances: only {system.n_unknowns} unknowns remain outside the PEC / wall points')
    # solve_resonances' A = L + k_t^2 M' with L = K, M' = -M: its own factor is of K - s M, a passed one of A itself
    solve = solve_resonances(system.operator, -system.mass, system.omega, system.energy_weights, system.in_window,
                             n_resonances, spectral_map='squared_cayley', warnings=warnings,
                             eigen_solver=eigen_solver, factorization=factorization, symmetric=True,
                             equilibration=bor_row_scale(system))
    return order_eigen_solve(solve, system.azimuthal_order, eigen_solver, factorization)


def order_eigen_solve(solve, order, eigen_solver, factorization=None):
    """EigenSolve of one azimuthal order from an eigenfrequency.solve_resonances one (Yee or triangles): BORResonances
    and the info keys merge_order_solves reads (azimuthal_order(s), factor_kind, factor_text)."""
    info = dict(solve.info)
    if factorization is not None:
        kind = getattr(factorization, 'description', 'external')
    elif eigen_solver.factorization in CUDSS_FACTORIZATIONS:
        kind = f'cuDSS {eigen_solver.factorization}'
    elif eigen_solver.factorization in HOST_DIRECT_FACTORIZATIONS:
        kind = f'{factor_label(eigen_solver.factorization)} {eigen_solver.factorization}'
    else:
        kind = 'SuperLU'
    info.update(azimuthal_order=int(order), azimuthal_orders=[int(order)], factor_kind=kind,
                factor_host_bytes=int(info['factor_host_bytes'] or 0))
    # an own cuDSS / MUMPS / PARDISO factor keeps solve_resonances' text
    if factorization is not None or 'factor_text' not in info:
        info['factor_text'] = _factor_text([kind], info['factor_seconds'], info['factor_host_bytes'])
    return EigenSolve(resonances=[_order_resonance(resonance, order) for resonance in solve.resonances], info=info)


def _order_resonance(resonance, order):
    """BORResonance of an eigenfrequency.Resonance at azimuthal order `order`."""
    return BORResonance(omega=resonance.omega, wavelength=resonance.wavelength, q_factor=resonance.q_factor,
                        pml_fraction=resonance.pml_fraction, field=resonance.field, azimuthal_order=int(order))


def _factor_text(kinds, seconds, host_bytes):
    """'SuperLU factorisation 0.31 s, L + U factors 12 MiB of host memory' (EigenSolve.summary factor part; 'factors'
    without a SuperLU kind, e.g. MUMPS / PARDISO)."""
    kinds = list(dict.fromkeys(kinds))
    text = f"{' / '.join(kinds)} factorisation {seconds:.3g} s"
    factors = 'L + U factors' if 'SuperLU' in kinds else 'factors'
    return text + (f', {factors} {host_bytes / 2 ** 20:.4g} MiB of host memory' if host_bytes else '')


_SUMMED_INFO = ('n_requested', 'factor_seconds', 'eigen_seconds', 'seconds', 'operator_applications',
                'eigenvalues_requested', 'eigenvalues_converged', 'pml_resonances_dropped')


def merge_order_solves(solves, warnings=()):
    """One EigenSolve of the per-order solves (in their order): resonances concatenated (order by order, nearest the
    target first within each), counts and seconds summed, warnings prefixed with 'm = <order>: ' (+ `warnings`)."""
    if len(solves) == 1 and not warnings:
        return solves[0]
    infos = [solve.info for solve in solves]
    info = {key: sum(part[key] for part in infos) for key in _SUMMED_INFO}
    orders = [part['azimuthal_order'] for part in infos]
    info.update(n_unknowns=max(part['n_unknowns'] for part in infos), target_wavelength=infos[0]['target_wavelength'],
                factor_host_bytes=max(part['factor_host_bytes'] for part in infos), azimuthal_order=orders[-1],
                azimuthal_orders=orders, factor_kind=' / '.join(dict.fromkeys(part['factor_kind'] for part in infos)),
                per_order=[{key: value for key, value in part.items() if key != 'warnings'} for part in infos],
                warnings=[f"m = {part['azimuthal_order']}: {text}" for part in infos for text in part['warnings']]
                + list(warnings))
    info['factor_text'] = (f"azimuthal orders m = {', '.join(map(str, orders))}: "
                           + _factor_text([part['factor_kind'] for part in infos], info['factor_seconds'],
                                          info['factor_host_bytes']))
    return EigenSolve(resonances=[resonance for solve in solves for resonance in solve.resonances], info=info)
