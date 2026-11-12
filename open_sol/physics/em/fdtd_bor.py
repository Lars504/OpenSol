"""Body-of-revolution FDTD (symmetry plan 2.3): leap-frog stepping of one azimuthal order m on the (r, z) half plane
with the em_fdfd_bor finite-integration curl (body_of_revolution, real metric), CPML (cylindrical along r), Drude /
Lorentz poles, current sources and DFT spectra; operators host-built (scipy), stepped on numpy or cupy."""
# State: x = the full E layout of body_of_revolution (E_r (nr+1, nz), u (nr, nz), E_z (nr, nz+1); inactive points of
# the order's axis class and PEC points held 0, except the |m| = 1 axis E_r, written after every E update as the
# regularity slaving E_r(0) = -u_0 / m of body_of_revolution; it feeds no update) and h = H on the a points (H_r (nr,
# nz+1), H_z (nr+1, nz), H_phi (nr+1, nz+1); the |m| = 1 axis H_phi = m H_r(cell 0) likewise after every H update),
# both real: the standing wave E_r, E_z, H_phi ~ cos(m phi) and E_phi = u, H_r, H_z ~ sin(m phi) (m = 0: the
# decoupled TM fields E_r, E_z, H_phi and TE fields E_phi, H_r, H_z themselves). With the curl a = C x of the module
# header of body_of_revolution and S = -1 on the a_r rows (h_r = -a_r variable), leap-frog (x^n at n dt, h at
# (n + 1/2) dt, sources at (n + 1/2) dt):
#   mu0 mu (h^{n+1/2} - h^{n-1/2}) / dt = -S C x^n
#   eps0 eps (x^{n+1} - x^n) / dt = V^-1 C^T S V_a h^{n+1/2} + q / V      (sigma_e / sigma_m semi-implicit)
# so eps0 mu0 M x'' + C^T W C x = mu0 q' with the em_fdfd_bor M = V eps, W = V_a / mu: DFT phasors (x sampled at n dt,
# h and q at (n + 1/2) dt) solve the em_fdfd_bor system at omega_tilde = (2 / dt) sin(omega dt / 2) exactly in
# lossless PML-free cells, and an em_fdfd_bor resonance omega rings at (2 / dt) asin(omega dt / 2).
# Phasors in the em_fdfd_bor convention (phasor_yee): E_r = X_er, E_phi = j X_u, E_z = X_ez, H_r = j H_r, H_z = j H_z,
# H_phi = H_phi of the DFTs.
# Sources (q per E point; the em_fdfd_bor rhs is j omega mu0 q): 'z' / 'r' current I s(t) through the cell's ring
# cross-section as point_current_rhs (I h / (4 pi) or I w / (4 pi) on each of its two edges; the 'r' half on the axis
# edge moves to u of cell 0 as -q / m at |m| = 1), 'phi' q_u = r I s(t), i.e. the em_fdfd_bor phi current at phase
# +90 deg (the u row carries -j J_phi).
# z CPML: the d_z terms (C0 entries: the a_r rows and the a_phi rows' E_r columns, and their duals in the E rows) get
# 1 / kappa and a psi convolution per row with the cpml_axis_profiles profile at the row's position (edges = 'h' rows,
# edge k = H row k - 1; centres = 'e' rows).
# r_hi CPML (cylindrical): the em_fdfd_bor complex radius stretch at alpha = 0, r~ = integral of kappa + sigma / (j omega
# eps0) dr of the centre profile (widths w s, centres / edges as running sums). Real part: the assembly is built on the
# r widths kappa w (stretched_assembly: rho = real stretched radius, w', L' stretched widths / edge lengths). Imaginary
# part: per r point a single pole, 1 / w~ = P(nu) / w', 1 / L~ = P(gamma) / L', 1 / r~ = P(beta) / rho with P(p) =
# j omega / (j omega + p) (rates nu = sigma / (eps0 kappa), gamma = Xi / L', beta = Omega / rho; Omega, Xi = the running
# sums of sigma w / eps0 at centres / edge means and edge lengths), and the radius inside the metric difference r~ u =
# rho (u + beta int u dt) (a_z rows) and r~bar H_phi = rho_bar (H_phi + beta_bar int H_phi dt) (E_z rows). Per layer row
# out = P_outer [P_inner (A (src + beta int src)) + C src] with A = its d_r part, C = its m / r part (outer pole 1 / r~ on
# a_r, a_z, E_r, E_z rows; inner pole 1 / L~ on a_z, a_phi, 1 / w~ on u, E_z rows), each pole a CPML-style recursion
# (psi = b psi + (b - 1) T, b = exp(-p dt), out = T + psi), the integral centred (dt / 2 of the current sample). Exact
# for the continuum (a coordinate stretch); pml_alpha_max acts on the z layers only.
# Poles (velocity form as the 2D engines): per Drude pole J += damping J + drive w E^n, per Lorentz pole dP += damping dP
# - restoring P^n + drive w E^n, P += dP, both from E^n before the E update, then E^{n+1} -= g J (Drude) / (g / dt) dP
# (Lorentz) with g = the E gain dt / (eps0 eps (1 + loss)); w = the point's e_point_means share of the pole cells (all
# three components, as em_fdfd_bor). The stepped pole is the continuous one at omega_tilde with gamma cos(omega dt / 2)
# (leapfrog_susceptibility: exact for the float32 coefficients), so DFT phasors solve em_fdfd_bor at omega_tilde with
# eps + leapfrog_susceptibility exactly.
# Time step: dt = courant x 2 / omega_max, omega_max of the vacuum operator (eps = mu_r = 1, the PEC of the build) at the
# order, so cells with eps >= 1, mu_r >= 1 are stable (|m| >= 2: the m / r terms near the axis set dt ~ dr / (c |m|));
# with poles eps - sum w P >= (dt / limit)^2 at every point (P = (omega_p dt)^2 / 4, delta_eps (omega_0 dt)^2 / (4 -
# (omega_0 dt)^2): the z = -1 bound of the 2D engines).
# Driven order sums (BOROrderSum, the em_fdfd_bor driven_orders twin): a point current at phi = 0 of moment I d is sum_m
# of its exp(-j m phi) terms (ring current I, or I d / (2 pi r) for phi, per term); the +m and -m terms of a real field
# add to 2 cos(m phi). Family 'rz' = the stepped standing wave itself (E_r, E_z, H_phi ~ cos(m phi)) carries r / z
# currents (stepper current 2 I); family 'phi' = the same wave a quarter period on (E_r, E_z, H_phi ~ sin(m phi), E_phi,
# H_r, H_z ~ -cos(m phi)) carries phi currents (stepper current -2 I d / (2 pi r)); m = 0 one stepper (current I, I d /
# (2 pi r)) holds both. One stepper per (order, family) on one shared assembly, in lockstep at courant x the smallest
# order limit. DFT phasors X of a member give the travelling terms (order_term_fields): m = 0 X itself, m > 0
# Y = phasor_yee(X) / 2 at +m and its phi mirror at -m, times j / -j for family 'phi' (the quarter-period shift
# exp(+-j pi / 2)); so the summed terms are the physical phasors of the 3D field, as em_fdfd_bor's order sum (m = 0: no
# phasor_yee j on E_phi, H_r, H_z).
# The phi = 0 field (shown fields, probes, harmonic phasors) = sum of the members with phi_zero_weights.
# Lockstep batch (_LockstepBatch): every stepper (a single order: a batch of one) steps inside a batch of steppers of one
# grid and dt, whose x / h are the rows of stacked (k, n) arrays: per half step one CSR product of the members'
# block-diagonal main, z CPML driver, r_hi derivative and m / r rows (one thread per row on cupy), the r_hi integral
# product, the layer poles and one fused update; poles, current injections (merged per waveform), the |m| = 1 axis
# slaving and each DFT group (stacked over its members) one launch each, so the launch count does not grow with k.
import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ...core.backend import cp
from ...core.constants import DTYPE, C0, EPS0, MU0
from ...core.sources import sampled_waveform_spectrum
from .body_of_revolution import BOR_SOURCE_COMPONENTS, BORAssembly, _regular_axis, axis_class, mirrored_yee
from .boundaries import cpml_axis_profiles
from .materials import DrudePole, LorentzPole

_EIGEN_TOLERANCE = 1e-6          # eigsh tolerance of the largest operator eigenvalue (its Ritz value lies below it)
_STABILITY_MARGIN = 1e-5         # omega_max x (1 + this) covers that tolerance
_DENSE_LIMIT = 400               # active E points up to which the largest eigenvalue comes from a dense eigvalsh
_UNIT_SLACK = 1e-6               # point eps / mu below 1 - this break the vacuum time-step bound
_NEUTRAL_PROFILE = (0.0, 1.0, 0.0)   # sigma, kappa, alpha of a row without that derivative
QUADRATURE_COMPONENTS = ('ephi', 'hr', 'hz')   # the sin(m phi) components: em_fdfd_bor phasor = j x the DFT phasor


# ---- layouts ----
def layout_sizes(shape):
    """((E_r, u, E_z) sizes, (H_r, H_z, H_phi) sizes) of the full E and a layouts of an (nr, nz) grid."""
    nr, nz = shape
    return ((nr + 1) * nz, nr * nz, nr * (nz + 1)), (nr * (nz + 1), (nr + 1) * nz, (nr + 1) * (nz + 1))


def layout_views(e_values, a_values, shape):
    """yee_fields-style dict of 2D views of full-layout vectors: er, ephi (u), ez of e_values, hr, hz, hphi of
    a_values."""
    nr, nz = shape
    (n_er, n_u, _), (n_ar, n_az, _) = layout_sizes(shape)
    return {'er': e_values[:n_er].reshape(nr + 1, nz), 'ephi': e_values[n_er:n_er + n_u].reshape(nr, nz),
            'ez': e_values[n_er + n_u:].reshape(nr, nz + 1), 'hr': a_values[:n_ar].reshape(nr, nz + 1),
            'hz': a_values[n_ar:n_ar + n_az].reshape(nr + 1, nz), 'hphi': a_values[n_ar + n_az:].reshape(nr + 1, nz + 1)}


def phasor_yee(e_values, a_values, shape):
    """body_of_revolution.yee_fields dict (em_fdfd_bor phasor convention) of full-layout DFT phasors: j on the
    sin(m phi) components E_phi, H_r, H_z (module header)."""
    return {name: 1j * values if name in QUADRATURE_COMPONENTS else values
            for name, values in layout_views(e_values, a_values, shape).items()}


def point_index(shape, selections):
    """(E index, a index) int64 full-layout positions of {component: (slice_r, slice_z)} (layout_views names)."""
    (n_er, n_u, n_ez), (n_ar, n_az, n_aphi) = layout_sizes(shape)
    e_mask, a_mask = np.zeros(n_er + n_u + n_ez, dtype=bool), np.zeros(n_ar + n_az + n_aphi, dtype=bool)
    views = layout_views(e_mask, a_mask, shape)
    for name, cells in selections.items():
        views[name][cells] = True
    return np.flatnonzero(e_mask), np.flatnonzero(a_mask)


# ---- CPML rows ----
def _profile_rows(profile, kind):
    """(sigma, kappa, alpha) float64 of a CPMLAxisProfile at the cell centres (kind 'e', n rows) or the edges (kind
    'h', n + 1 rows: edge k = H row k - 1, edge 0 unstretched)."""
    rows = [getattr(profile, f'{name}_{kind}') for name in ('sigma', 'kappa', 'alpha')]
    if kind == 'h':
        rows = [np.concatenate(([neutral], row)) for row, neutral in zip(rows, _NEUTRAL_PROFILE)]
    return rows


def _layout_profile(blocks, profile):
    """(sigma, kappa, alpha) float64 per row of a layout of blocks ((n_r, n_z, kind | None), ...) of a z profile; kind
    'e' / 'h' = the d_z sits at cell centres / edges along z, None = no d_z in that block."""
    parts = ([], [], [])
    for n_r, n_z, kind in blocks:
        if kind is None:
            values = [np.full(n_r * n_z, neutral) for neutral in _NEUTRAL_PROFILE]
        else:
            values = [np.tile(row, n_r) for row in _profile_rows(profile, kind)]
        for part, value in zip(parts, values):
            part.append(value)
    return tuple(np.concatenate(part) for part in parts)


def _recursion(sigma, kappa, alpha, dt):
    """CPML psi recursion (a, b) float64 (the 2D formula of boundaries._cpml_recursion): psi = b psi + a d(field)."""
    b = np.exp(-(sigma / kappa + alpha) * float(dt) / EPS0)
    denominator = sigma * kappa + kappa ** 2 * alpha
    a = np.divide(sigma * (b - 1.0), denominator, out=np.zeros_like(denominator), where=denominator > 1e-30)
    return a, b


# ---- cylindrical r_hi CPML (module header) ----
def _running_metric(widths):
    """(centres, edge means, edge lengths) of cell widths from 0 (body_of_revolution._coordinates layout)."""
    edges = np.concatenate((np.zeros(1), np.cumsum(widths)))
    centres = edges[:-1] + 0.5 * widths
    lower, upper = np.concatenate((edges[:1], centres)), np.concatenate((centres, edges[-1:]))
    return centres, 0.5 * (lower + upper), upper - lower


def stretched_assembly(cell_widths, boundary_config, pml_dx, materials, window=None):
    """Real-metric BORAssembly of fdtd_bor: r widths kappa w (the real part of the r_hi radius stretch, module header),
    z widths as given; identical to BORAssembly(..., None) without an r_hi cpml."""
    dr_cells, dz_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    profile = cpml_axis_profiles(boundary_config, dr_cells.size, 'x', pml_dx)
    return BORAssembly((profile.kappa_e * dr_cells, dz_cells), boundary_config, pml_dx, materials, None,
                       window=window)


def _radial_rates(dr_cells, profile, nz):
    """{'h': (inner, outer, source integral weights), 'e': (...)} float64 pole rates [1/s] per row of the a / E layout
    (integral weights per source column: u on the H side, H_phi on the E side) of the r_hi layer, None without one."""
    if not np.any(profile.sigma_e > 0.0):
        return None
    stretched, conducting = profile.kappa_e * dr_cells, profile.sigma_e * dr_cells / EPS0
    radius, edge_radius, edge_length = _running_metric(stretched)
    reach, edge_reach, edge_extra = _running_metric(conducting)
    cell_rate, edge_rate = conducting / stretched, edge_extra / edge_length            # 1 / w~, 1 / L~
    radius_rate, edge_radius_rate = reach / radius, edge_reach / edge_radius          # 1 / r~ at centres / edges
    centre_zero, edge_zero = np.zeros(dr_cells.size), np.zeros(dr_cells.size + 1)

    def layout(*blocks):   # (values along r, n_z) blocks, r index major
        return np.concatenate([np.repeat(values, n_z) for values, n_z in blocks])
    h_rows = (layout((centre_zero, nz + 1), (edge_rate, nz), (edge_rate, nz + 1)),
              layout((radius_rate, nz + 1), (edge_radius_rate, nz), (edge_zero, nz + 1)),
              layout((edge_zero, nz), (radius_rate, nz), (centre_zero, nz + 1)))
    e_rows = (layout((edge_zero, nz), (cell_rate, nz), (cell_rate, nz + 1)),
              layout((edge_radius_rate, nz), (centre_zero, nz), (radius_rate, nz + 1)),
              layout((centre_zero, nz + 1), (edge_zero, nz), (edge_radius_rate, nz + 1)))
    return {'h': h_rows, 'e': e_rows}


def _pole_recursion(rates, dt):
    """(a, b) float64 of the single-pole recursion psi = b psi + a T, out = T + psi: j omega / (j omega + rate)."""
    b = np.exp(-np.asarray(rates, dtype=np.float64) * float(dt))
    return b - 1.0, b


# ---- dispersive poles (module header) ----
def leapfrog_susceptibility(pole, omega, dt):
    """complex128 chi(omega) of a DrudePole / LorentzPole as stepped (velocity form, its float32 ade_coefficients at
    dt), the eps term of em_fdfd_bor at omega_tilde = (2 / dt) sin(omega dt / 2); omega [rad/s] float or array."""
    omega = np.asarray(omega, dtype=np.float64)
    dt = float(dt)
    tilde = 2.0 / dt * np.sin(0.5 * omega * dt)
    later, earlier = np.exp(0.5j * omega * dt), np.exp(-0.5j * omega * dt)
    if isinstance(pole, DrudePole):
        damping, drive = (float(value) for value in pole.ade_coefficients(dt))
        chi = drive / (1j * tilde * EPS0 * (later - (1.0 + damping) * earlier))
    else:
        damping, restoring, drive = (float(value) for value in pole.ade_coefficients(dt))
        step = later ** 2 - 1.0
        chi = drive / (EPS0 * (step * (1.0 - (1.0 + damping) * earlier ** 2) + restoring))
    return complex(chi) if np.ndim(chi) == 0 else chi


def _derivative_split(curl_static, shape):
    """(d_z part, d_r part) float64 CSR of the full-layout static curl C0 (module header)."""
    import scipy.sparse
    (n_er, _, _), (n_ar, n_az, _) = layout_sizes(shape)
    matrix = curl_static.tocoo()
    rows, columns, data = matrix.row, matrix.col, matrix.data.real
    along_z = (rows < n_ar) | ((rows >= n_ar + n_az) & (columns < n_er))

    def part(mask):
        return scipy.sparse.csr_matrix((data[mask], (rows[mask], columns[mask])), shape=matrix.shape)
    return part(along_z), part(~along_z)


def stable_time_step(curl, e_volumes, a_volumes, e_active, warm_start=None):
    """Leap-frog limit 2 / omega_max [s] of eps0 mu0 V x'' = -C^T V_a C x on the active E points (vacuum):
    omega_max^2 = C0^2 lambda_max(V^-1/2 C^T V_a C V^-1/2), dense or by eigsh (with _STABILITY_MARGIN); warm_start: a
    dict whose 'vector' (full E layout) seeds eigsh and receives its eigenvector (the orders of one order sum)."""
    import scipy.sparse
    import scipy.sparse.linalg
    index = np.flatnonzero(e_active)
    if index.size == 0:
        raise ValueError('body-of-revolution FDTD: no active E point is left (every cell PEC?)')
    weighted = (scipy.sparse.diags(np.sqrt(a_volumes)) @ curl.tocsc()[:, index]
                @ scipy.sparse.diags(1.0 / np.sqrt(e_volumes[index]))).tocsr()
    if index.size <= _DENSE_LIMIT:
        dense = weighted.toarray()
        largest = float(np.linalg.eigvalsh(dense.T @ dense)[-1])
    else:
        operator = scipy.sparse.linalg.LinearOperator((index.size, index.size), dtype=np.float64,
                                                      matvec=lambda vector: weighted.T @ (weighted @ vector))
        previous = None if warm_start is None else warm_start.get('vector')
        start = None if previous is None else previous[index]
        if start is not None and not np.any(start):
            start = None
        values, vectors = scipy.sparse.linalg.eigsh(operator, k=1, which='LA', tol=_EIGEN_TOLERANCE, v0=start)
        largest = float(values[0])
        if warm_start is not None:   # the next order's operator differs by its order term: a near eigenvector
            full = np.zeros(e_active.size, dtype=np.float64)
            full[index] = vectors[:, 0]
            warm_start['vector'] = full
    if not (math.isfinite(largest) and largest > 0.0):
        raise ValueError(f'body-of-revolution FDTD: the largest operator eigenvalue is {largest!r}: no time step')
    return 2.0 / (C0 * math.sqrt(largest) * (1.0 + _STABILITY_MARGIN))


def order_time_step_limit(assembly, order, warm_start=None) -> float:
    """Leap-frog limit 2 / omega_max [s] of azimuthal order `order` on a real-metric BORAssembly (stable_time_step,
    warm_start passed on)."""
    curl_static, curl_order, e_active, _ = assembly.curl_parts(int(order))
    return stable_time_step((curl_static + int(order) * curl_order).real, np.asarray(assembly.full_energy_weights,
                                                                                     dtype=np.float64),
                            np.abs(assembly.a_volumes), e_active, warm_start=warm_start)


# ---- device kernels (numpy twins next to them) ----
_UPDATE = cp.ElementwiseKernel(
    'float32 decay, float32 gain, float32 curl', 'float32 field', 'field = decay * field + gain * curl',
    'bor_fdtd_update')
_UPDATE_PML = cp.ElementwiseKernel(
    'float32 decay, float32 gain, float32 curl, float32 drive, float32 a, float32 b', 'float32 psi, float32 field',
    'psi = b * psi + a * drive; field = decay * field + gain * (curl + psi)', 'bor_fdtd_update_pml')
_RADIAL_POLES = cp.ElementwiseKernel(
    'int64 row, float32 inner, float32 integral, float32 order, float32 a_inner, float32 b_inner, float32 a_outer, '
    'float32 b_outer',
    'float32 psi_inner, float32 psi_outer, raw float32 curl',
    'float total = inner + integral; psi_inner = b_inner * psi_inner + a_inner * total; '
    'float term = total + psi_inner + order; psi_outer = b_outer * psi_outer + a_outer * term; '
    'curl[row] += term + psi_outer',
    'bor_fdtd_radial_poles')
_RADIAL_INTEGRATE = cp.ElementwiseKernel(
    'int64 column, float32 interval, raw float32 source', 'float32 integrated',
    'integrated += interval * source[column]', 'bor_fdtd_radial_integrate')
_DRUDE_STEP = cp.ElementwiseKernel(
    'int64 point, float32 damping, float32 drive, raw float32 field', 'float32 current',
    'current += damping * current + drive * field[point]', 'bor_fdtd_drude_step')
_LORENTZ_STEP = cp.ElementwiseKernel(
    'int64 point, float32 damping, float32 restoring, float32 drive, raw float32 field',
    'float32 polarization, float32 change',
    'change += damping * change - restoring * polarization + drive * field[point]; polarization += change',
    'bor_fdtd_lorentz_step')
_POLE_CORRECT = cp.ElementwiseKernel(
    'int64 point, float32 weight, float32 value', 'raw float32 field', 'field[point] -= weight * value',
    'bor_fdtd_pole_correct')
_FIELD_INJECT = cp.ElementwiseKernel(
    'int64 point, float32 weight_re, float32 weight_im, float32 signal, float32 quadrature', 'raw float32 field',
    'field[point] += weight_re * signal + weight_im * quadrature', 'bor_fdtd_field_inject')
_POINT_ADD = cp.ElementwiseKernel(
    'int64 point, float32 weight, float32 value', 'raw float32 field', 'field[point] += weight * value',
    'bor_fdtd_point_add')
_AXIS_COPY = cp.ElementwiseKernel(
    'int64 target, int64 source, float32 factor', 'raw float32 field', 'field[target] = factor * field[source]',
    'bor_fdtd_axis_copy')
# DFT of stacked member rows: total (k, n_f, n) += phase (n_f) x values[offset (k) + point (n)] (values flat)
_ACCUMULATE_DOUBLE = cp.ElementwiseKernel(
    'complex128 phase, int64 point, int64 offset, raw float32 values', 'complex128 total',
    'total += phase * (double)values[offset + point]', 'bor_fdtd_dft_double')
_ACCUMULATE_SINGLE = cp.ElementwiseKernel(
    'complex128 phase, int64 point, int64 offset, raw float32 values', 'complex64 total',
    'total += complex<float>(phase * (double)values[offset + point])', 'bor_fdtd_dft_single')
_CSR_PRODUCT = cp.RawKernel(r'''
extern "C" __global__ void bor_fdtd_csr_product(const int* row_starts, const int* columns, const float* values,
                                                 const float* vector, float* out, const int n_rows) {
    const int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= n_rows) return;
    float total = 0.0f;
    for (int entry = row_starts[row]; entry < row_starts[row + 1]; ++entry) {
        total += values[entry] * vector[columns[entry]];
    }
    out[row] = total;
}
''', 'bor_fdtd_csr_product')
_CSR_BLOCK = 256   # threads per block of _CSR_PRODUCT (one row each)


def _update_host(field, decay, gain, curl, pml):
    """numpy twin of _UPDATE / _UPDATE_PML (pml = (drive, a, b, psi) of the z CPML or None); overwrites the curl work
    buffer."""
    if pml is not None:
        drive, a, b, psi = pml
        psi *= b
        psi += a * drive
        curl += psi
    field *= decay
    field += gain * curl


def _radial_host(rows, inner, integral, order, a_inner, b_inner, a_outer, b_outer, psi_inner, psi_outer, curl):
    """numpy twin of _RADIAL_POLES (rows distinct)."""
    total = inner + integral
    psi_inner *= b_inner
    psi_inner += a_inner * total
    term = total + psi_inner + order
    psi_outer *= b_outer
    psi_outer += a_outer * term
    curl[rows] += term + psi_outer


def _accumulate_stack(total, phase, points, offsets, rows, values, flat_values):
    """total (k, n_f, n) += phase (n_f,) x values[rows][:, points] of stacked member rows values (k_all, n_values);
    cupy: the kernels above on flat_values at offsets = rows x n_values (numpy twin)."""
    if isinstance(total, np.ndarray):
        total += phase[None, :, None] * values[rows][:, points][:, None, :]
        return
    kernel = _ACCUMULATE_DOUBLE if total.dtype == np.complex128 else _ACCUMULATE_SINGLE
    kernel(phase[None, :, None], points[None, None, :], offsets[:, None, None], flat_values, total)


class _SparseProduct:
    """out = matrix @ vector of a host-built CSR as float32: scipy on numpy, one thread per row on cupy
    (_CSR_PRODUCT: no cuSPARSE descriptors or buffers per call)."""

    def __init__(self, matrix, xp):
        matrix = matrix.tocsr()
        matrix.eliminate_zeros()
        matrix = matrix.astype(np.float32)
        self.nnz, self.n_rows = int(matrix.nnz), int(matrix.shape[0])
        if xp is np:
            self.matrix = matrix
            return
        if self.nnz >= 2 ** 31:
            raise ValueError(f'BOR FDTD: a curl product of {self.nnz} entries exceeds the int32 CSR index')
        self.matrix = None
        self.row_starts = xp.asarray(matrix.indptr.astype(np.int32))
        self.columns = xp.asarray(matrix.indices.astype(np.int32))
        self.values = xp.asarray(matrix.data)
        self._n_rows_i32 = np.int32(self.n_rows)
        self._blocks = ((self.n_rows + _CSR_BLOCK - 1) // _CSR_BLOCK,)

    def apply(self, vector, out):
        if self.matrix is not None:
            out[...] = self.matrix @ vector
        elif self.n_rows:
            _CSR_PRODUCT(self._blocks, (_CSR_BLOCK,), (self.row_starts, self.columns, self.values, vector, out,
                                                       self._n_rows_i32))


def _block_diagonal(blocks, shapes):
    """Block-diagonal float64 CSR of host sparse blocks with (rows, columns) shapes (None: an all-zero block)."""
    import scipy.sparse
    rows, columns, values = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)], [np.zeros(0)]
    row_offset = column_offset = 0
    for block, (n_rows, n_columns) in zip(blocks, shapes):
        if block is not None:
            if tuple(block.shape) != (n_rows, n_columns):
                raise ValueError(f'BOR FDTD batch block of shape {block.shape} differs from {(n_rows, n_columns)}')
            entries = scipy.sparse.coo_matrix(block)
            rows.append(entries.row.astype(np.int64) + row_offset)
            columns.append(entries.col.astype(np.int64) + column_offset)
            values.append(np.asarray(entries.data, dtype=np.float64))
        row_offset += n_rows
        column_offset += n_columns
    return scipy.sparse.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
                                   shape=(row_offset, column_offset))


@dataclass(frozen=True, eq=False)
class _RadialPlan:
    """Host r_hi layer rows of one half step of one stepper (module header): int64 rows, d_r CSR (+ dt / 2 of the
    current sample's radius integral), integral CSR over the source columns (None: none), m / r CSR (None: none) and
    the float64 pole recursion (a_inner, b_inner, a_outer, b_outer) per row."""
    rows: np.ndarray
    derivative: object
    integral: object
    columns: np.ndarray
    order: object
    coefficients: tuple


@dataclass(frozen=True, eq=False)
class _SidePlan:
    """Host operators of one half step (E or H) of one stepper: main CSR (size x source_size), z CPML (driver CSR, a,
    b; None without a z CPML face) and the r_hi layer (_RadialPlan, None without one)."""
    main: object
    z_pml: object
    radial: object
    size: int
    source_size: int


def _radial_plan(rows, r_part, order_part, inner_rates, outer_rates, integral_weights, dt):
    """_RadialPlan of the layer rows of a side: r_part / order_part = its full d_r / m / r CSR parts, per-row rates
    [1/s] of the inner / outer poles, integral weights [1/s] per source column (0: no radius integral)."""
    import scipy.sparse
    derivative = r_part.tocsr()[rows]
    weighted = derivative @ scipy.sparse.diags(integral_weights)
    columns = np.flatnonzero(integral_weights > 0.0)
    order = order_part.tocsr()[rows]
    order.eliminate_zeros()
    return _RadialPlan(rows=rows.astype(np.int64), derivative=(derivative + (0.5 * dt) * weighted).tocsr(),
                       integral=weighted.tocsc()[:, columns].tocsr() if columns.size else None,
                       columns=columns.astype(np.int64), order=order if order.nnz else None,
                       coefficients=(*_pole_recursion(inner_rates, dt), *_pole_recursion(outer_rates, dt)))


class _BatchRadial:
    """The r_hi layer rows of one half step of every batch member (module header): their d_r part A (+ the centred
    radius integral of the integral-weighted source columns) through the inner pole, plus their m / r part C, through
    the outer pole, added into the side's stacked curl buffer (global rows / columns: member offset + own index)."""

    def __init__(self, plans, size, source_size, derivative_values, order_values, xp, dt):
        """plans = per member _RadialPlan or None; derivative_values / order_values = the side product's output views of
        the layer rows (member order)."""
        rows, columns, blocks, shapes = [], [], [], []
        coefficients = ([], [], [], [])
        for position, plan in enumerate(plans):
            if plan is None:
                continue
            rows.append(position * size + plan.rows)
            columns.append(position * source_size + plan.columns)
            blocks.append(plan.integral)
            shapes.append((plan.rows.size, plan.columns.size))
            for store, values in zip(coefficients, plan.coefficients):
                store.append(values)
        self.rows = np.concatenate(rows).astype(np.int64)
        self.columns = np.concatenate(columns).astype(np.int64)
        self._integral = _SparseProduct(_block_diagonal(blocks, shapes), xp) if self.columns.size else None
        self.device_rows = xp.asarray(self.rows)
        self.device_columns = xp.asarray(self.columns)
        self.values = (derivative_values, xp.zeros(self.rows.size, dtype=DTYPE), order_values)
        self.integrated = xp.zeros(self.columns.size, dtype=DTYPE)   # int source dt up to the previous sample's end
        self.coefficients = tuple(xp.asarray(np.concatenate(store), dtype=DTYPE) for store in coefficients)
        self.psi = (xp.zeros(self.rows.size, dtype=DTYPE), xp.zeros(self.rows.size, dtype=DTYPE))
        self.interval = DTYPE(dt)

    def add(self, source, curl):
        """curl[rows] += the stretched layer terms (d_r / m / r values from the side product), then the radius integral
        takes the source sample."""
        if self._integral is not None:
            self._integral.apply(self.integrated, self.values[1])
        if isinstance(curl, np.ndarray):
            _radial_host(self.rows, *self.values, *self.coefficients, *self.psi, curl)
            if self.columns.size:
                self.integrated += self.interval * source[self.columns]
            return
        _RADIAL_POLES(self.device_rows, *self.values, *self.coefficients, *self.psi, curl)
        if self.columns.size:
            _RADIAL_INTEGRATE(self.device_columns, self.interval, source, self.integrated)


class _BatchSide:
    """One half step (E or H) of every batch member (module header 'Lockstep batch'): one CSR product of the stacked
    block-diagonal main, z CPML driver, r_hi d_r and m / r rows into one buffer (curl, psi drive, layer values), the
    r_hi layer (_BatchRadial), decay / gain per member row and the fused update."""

    def __init__(self, plans, xp, dt):
        import scipy.sparse
        size, source_size = plans[0].size, plans[0].source_size
        if any((plan.size, plan.source_size) != (size, source_size) for plan in plans):
            raise ValueError('BOR FDTD lockstep batch members need one grid')
        z_pml = plans[0].z_pml is not None
        if any((plan.z_pml is not None) != z_pml for plan in plans):
            raise ValueError('BOR FDTD lockstep batch members need the same z CPML faces')
        main_shapes = [(size, source_size)] * len(plans)
        blocks = [_block_diagonal([plan.main for plan in plans], main_shapes)]
        if z_pml:
            blocks.append(_block_diagonal([plan.z_pml[0] for plan in plans], main_shapes))
        radial_plans = [plan.radial for plan in plans]
        layer_shapes = [(0 if radial is None else radial.rows.size, source_size) for radial in radial_plans]
        n_layer = sum(rows for rows, _ in layer_shapes)
        if n_layer:
            blocks.append(_block_diagonal([None if radial is None else radial.derivative for radial in radial_plans],
                                          layer_shapes))
            blocks.append(_block_diagonal([None if radial is None else radial.order for radial in radial_plans],
                                          layer_shapes))
        self._product = _SparseProduct(scipy.sparse.vstack(blocks, format='csr'), xp)
        n_main = len(plans) * size
        self._out = xp.zeros(n_main * (1 + int(z_pml)) + 2 * n_layer, dtype=DTYPE)
        self.curl = self._out[:n_main]
        self.decay = xp.ones(n_main, dtype=DTYPE)
        self.gain = xp.zeros(n_main, dtype=DTYPE)
        offset = n_main
        self.pml = None
        if z_pml:
            a, b = (np.concatenate([plan.z_pml[part] for plan in plans]) for part in (1, 2))
            self.pml = (self._out[offset:offset + n_main], xp.asarray(a, dtype=DTYPE), xp.asarray(b, dtype=DTYPE),
                        xp.zeros(n_main, dtype=DTYPE))
            offset += n_main
        self.radial = None
        if n_layer:
            self.radial = _BatchRadial(radial_plans, size, source_size, self._out[offset:offset + n_layer],
                                       self._out[offset + n_layer:], xp, dt)

    def advance(self, source, field):
        """field = decay field + gain (main source + r_hi layer terms + psi) on the flat stacked vectors."""
        self._product.apply(source, self._out)
        if self.radial is not None:
            self.radial.add(source, self.curl)
        if isinstance(field, np.ndarray):
            _update_host(field, self.decay, self.gain, self.curl, self.pml)
        elif self.pml is None:
            _UPDATE(self.decay, self.gain, self.curl, field)
        else:
            _UPDATE_PML(self.decay, self.gain, self.curl, *self.pml, field)


class _PolePoints:
    """One Drude / Lorentz pole on the E points its cells touch in one stepper (module header): host point index,
    point weights w, recursion scalars, float64 drive w x drive and correction weights g (Drude) / g / dt (Lorentz);
    stepped by the batch's _BatchPole."""

    def __init__(self, pole, indices, weights, dt):
        if not isinstance(pole, (DrudePole, LorentzPole)):
            raise TypeError(f'BOR FDTD poles must be DrudePole or LorentzPole, got {type(pole).__name__}')
        self.pole = pole
        self.drude = isinstance(pole, DrudePole)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        coefficients = pole.ade_coefficients(dt)
        self.damping = DTYPE(coefficients[0])
        self.restoring = None if self.drude else DTYPE(coefficients[1])
        self.drive = float(coefficients[-1]) * self.weights
        self.correction = None
        self.dt = float(dt)

    def set_correction(self, gain_e):
        """Correction weights from the host float64 E gain dt / (eps0 eps (1 + loss)) of the full layout."""
        self.correction = gain_e[self.indices] / (1.0 if self.drude else self.dt)


class _BatchPole:
    """One Drude / Lorentz pole on the E points of every batch member that has it (flat indices member offset + own
    index): device buffers (J; P and dP), drive and correction from the members' _PolePoints."""

    def __init__(self, member_points, n_e, xp):
        """member_points = [(member position, _PolePoints)] of one pole; n_e = E points per member."""
        first = member_points[0][1]
        self.drude, self.damping, self.restoring = first.drude, first.damping, first.restoring
        self._members = member_points
        self.xp = xp
        self.indices = np.concatenate([position * n_e + points.indices for position, points in member_points])
        self.device_index = xp.asarray(self.indices)
        self.drive = xp.asarray(np.concatenate([points.drive for _, points in member_points]), dtype=DTYPE)
        self.current = xp.zeros(self.indices.size, dtype=DTYPE)                              # J or P
        self.change = None if self.drude else xp.zeros(self.indices.size, dtype=DTYPE)      # dP
        self.correction = None

    def upload(self):
        """Device correction weights from the members' host ones (after a coefficient refresh)."""
        self.correction = self.xp.asarray(np.concatenate([points.correction for _, points in self._members]),
                                          dtype=DTYPE)

    def update(self, x):
        """J^{n+1/2} (Drude) or P^{n+1}, dP (Lorentz) from E^n = x (flat stacked) before the E update."""
        if isinstance(x, np.ndarray):
            sample = x[self.indices]
            if self.drude:
                self.current += self.damping * self.current + self.drive * sample
            else:
                self.change += self.damping * self.change - self.restoring * self.current + self.drive * sample
                self.current += self.change
        elif self.drude:
            _DRUDE_STEP(self.device_index, self.damping, self.drive, x, self.current)
        else:
            _LORENTZ_STEP(self.device_index, self.damping, self.restoring, self.drive, x, self.current, self.change)

    def correct(self, x):
        """E^{n+1} -= g J (Drude) or (g / dt) dP (Lorentz) after the E update."""
        value = self.current if self.drude else self.change
        if isinstance(x, np.ndarray):
            x[self.indices] -= self.correction * value
        else:
            _POLE_CORRECT(self.device_index, self.correction, value, x)


class _Injection:
    """A current source on its E points: host indices and q per point [A m] (module header), waveform temporal; host
    float64 weights gain q / V, set by the stepper's coefficient refresh (the batch merges them per waveform)."""

    def __init__(self, indices, charges, temporal):
        self.indices = np.asarray(indices, dtype=np.int64)
        self.charges = np.asarray(charges, dtype=np.float64)
        self.temporal = temporal
        self.weights = None


class _FieldInjection:
    """A phasor-shaped injection into the H ('h') or E ('e') update rows `indices` (fdtd_bor_ports' TFSF terms): the
    update term phasor per row, gain-weighted (host complex128 weights, set by the coefficient refresh), applied by the
    batch as Re s(t) + Im s_q(t) of the waveform and its quadrature."""

    def __init__(self, side, indices, phasor, temporal):
        if side not in ('h', 'e'):
            raise ValueError(f"field injection side must be 'h' or 'e', got {side!r}")
        self.side = side
        self.indices = np.asarray(indices, dtype=np.int64)
        self.phasor = np.asarray(phasor, dtype=np.complex128)
        self.temporal = temporal
        self.weights = None


def source_spectrum(temporal, dt, frequencies, n_steps):
    """S(omega) = sum_{n < n_steps} s((n + 1/2) dt) exp(-j omega (n + 1/2) dt) dt (the injection times), complex128 at
    frequencies [Hz]."""
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    if n_steps < 1:
        return np.zeros(frequencies.shape, dtype=np.complex128)
    shifted = sampled_waveform_spectrum(lambda times: temporal(np.asarray(times) + 0.5 * dt), dt, frequencies, 0,
                                        n_steps - 1)
    return shifted * np.exp(-1j * math.pi * frequencies * dt)


class BORSpectra:
    """Running DFT sum exp(-j omega t) dt of the x entries e_index and h entries a_index (None: all) of one stepper at
    frequencies [Hz] (x at its time n dt, h at (n - 1/2) dt; module header): e_spectra (n_f, n_e) / a_spectra complex
    on xp, views of its batch's _SpectraGroup, which accumulates them."""

    def __init__(self, frequencies, sizes, host_e_index, host_a_index, e_spectra, a_spectra):
        self.frequencies = frequencies
        self.sizes = sizes
        self.host_e_index = host_e_index
        self.host_a_index = host_a_index
        self.e_spectra = e_spectra
        self.a_spectra = a_spectra

    def scale(self, factors):
        """Multiply the spectra of frequency k by factors[k] in place (factors (n_f,) complex on xp)."""
        self.e_spectra *= factors[:, None]
        self.a_spectra *= factors[:, None]

    def full_phasors(self, index, scale=1.0):
        """(E, a) complex128 host full-layout vectors of frequency `index` times scale (unrecorded entries 0)."""
        parts = []
        for spectra, host_index, size in ((self.e_spectra, self.host_e_index, self.sizes[0]),
                                          (self.a_spectra, self.host_a_index, self.sizes[1])):
            values = np.asarray(_host(spectra[index]), dtype=np.complex128) * scale
            if host_index is None:
                parts.append(values)
                continue
            full = np.zeros(size, dtype=np.complex128)
            full[host_index] = values
            parts.append(full)
        return tuple(parts)


def _host(array):
    """numpy view / copy of a numpy or cupy array."""
    return array if isinstance(array, np.ndarray) else array.get()


class _SpectraGroup:
    """DFT sums of the x entries e_index and h entries a_index (None: all) of the batch members at `positions`, from
    step next_step on, stacked (k, n_f, n) per side in complex `dtype` (one launch per side and step); `views` = the
    members' BORSpectra."""

    def __init__(self, positions, frequencies, dt, xp, sizes, next_step, e_index=None, a_index=None,
                 dtype=np.complex128):
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
        if not (frequencies.size and np.all(np.isfinite(frequencies)) and np.all(frequencies > 0)):
            raise ValueError(f'BOR FDTD spectra need finite frequencies > 0 [Hz], got {frequencies!r}')
        omega = 2.0 * math.pi * frequencies
        self.positions = np.asarray(positions, dtype=np.int64)
        host_indices = tuple(None if index is None else np.asarray(index, dtype=np.int64)
                             for index in (e_index, a_index))
        self._sides = []   # (stack, phase view, device points, device offsets) of x, then h
        stacks = []
        for host_index, size in zip(host_indices, sizes):
            points = np.arange(size, dtype=np.int64) if host_index is None else host_index
            stack = xp.zeros((self.positions.size, omega.size, points.size), dtype=dtype)
            stacks.append(stack)
            self._sides.append([stack, None, xp.asarray(points), xp.asarray(self.positions * size)])
        # complex128 phases of the next samples (device recursion: ~1e-16 per step): row 0 x, row 1 h
        self._phases = xp.asarray(np.stack((np.exp(-1j * omega * next_step * dt) * dt,
                                            np.exp(-1j * omega * (next_step - 0.5) * dt) * dt)))
        self._rotation = xp.asarray(np.exp(-1j * omega * dt))[None, :]
        for row, side in enumerate(self._sides):
            side[1] = self._phases[row]
        self.views = [BORSpectra(frequencies, sizes, *host_indices, stacks[0][member], stacks[1][member])
                      for member in range(self.positions.size)]

    def accumulate(self, stacked_x, flat_x, stacked_h, flat_h):
        """Add the samples of the step that just ended (x at n dt, h at (n - 1/2) dt) of the stacked member rows."""
        for (stack, phase, points, offsets), values, flat_values in zip(self._sides, (stacked_x, stacked_h),
                                                                        (flat_x, flat_h)):
            _accumulate_stack(stack, phase, points, offsets, self.positions, values, flat_values)
        self._phases *= self._rotation


class _LockstepBatch:
    """Stacked state of BORTimeSteppers of one grid and time step stepped in lockstep (module header 'Lockstep batch'):
    x (k, n_e) / h (k, n_a) whose rows are the members' x / h, one _BatchSide per half step, the poles, current and
    field injections, |m| = 1 axis slaving and DFT groups of all members, each in one launch."""

    def __init__(self, steppers):
        first = steppers[0]
        self.members = tuple(steppers)
        if any(stepper.shape != first.shape or stepper.dt != first.dt or stepper.xp is not first.xp
               or len(stepper._pole_slots) != len(first._pole_slots) for stepper in steppers):
            raise ValueError('BOR FDTD lockstep batch members need one grid, time step, backend and pole list')
        self.xp = xp = first.xp
        self.shape, self.sizes, self.dt = first.shape, first.sizes, first.dt
        n_e, n_a = self.sizes
        self.x = xp.zeros((len(steppers), n_e), dtype=DTYPE)
        self.h = xp.zeros((len(steppers), n_a), dtype=DTYPE)
        self._x_flat, self._h_flat = self.x.reshape(-1), self.h.reshape(-1)
        self.n_step = 0
        self._h_side = _BatchSide([stepper._plans['h'] for stepper in steppers], xp, self.dt)
        self._e_side = _BatchSide([stepper._plans['e'] for stepper in steppers], xp, self.dt)
        self._poles = []
        for slot in range(len(first._pole_slots)):
            member_points = [(position, stepper._pole_slots[slot]) for position, stepper in enumerate(steppers)
                             if stepper._pole_slots[slot] is not None]
            if member_points:
                self._poles.append(_BatchPole(member_points, n_e, xp))
        self._slaving = self._axis_slaving()
        self._spectra = []
        self._injection_groups = ()
        self._field_injections = ()
        for position, stepper in enumerate(steppers):
            stepper._bind(self, position)
        self.upload()

    def _axis_slaving(self):
        """{'e' | 'a': (flat targets, flat sources, float32 factors) of the |m| = 1 members, None without}."""
        parts = {'e': ([], [], []), 'a': ([], [], [])}
        n_e, n_a = self.sizes
        for position, stepper in enumerate(self.members):
            if stepper._axis_slaving is None:
                continue
            e_axis, e_cell, e_factor, a_axis, a_cell, a_factor = stepper._axis_slaving
            for kind, size, targets, sources, factor in (('e', n_e, e_axis, e_cell, e_factor),
                                                         ('a', n_a, a_axis, a_cell, a_factor)):
                store = parts[kind]
                store[0].append(position * size + targets)
                store[1].append(position * size + sources)
                store[2].append(np.full(targets.size, factor))
        xp = self.xp
        return {kind: None if not store[0] else (xp.asarray(np.concatenate(store[0])),
                                                 xp.asarray(np.concatenate(store[1])),
                                                 xp.asarray(np.concatenate(store[2]), dtype=DTYPE))
                for kind, store in parts.items()}

    def upload(self):
        """Copy the members' host coefficients (decay / gain, pole corrections, injection / field-injection weights)."""
        xp = self.xp
        for side, name in ((self._h_side, 'h'), (self._e_side, 'e')):
            side.decay[...] = xp.asarray(np.concatenate([stepper._decay[name] for stepper in self.members]),
                                         dtype=DTYPE)
            side.gain[...] = xp.asarray(np.concatenate([stepper._gain_h if name == 'h' else stepper._gain_e
                                                        for stepper in self.members]), dtype=DTYPE)
        for pole in self._poles:
            pole.upload()
        self.injections_changed()

    def injections_changed(self):
        """Rebuild the device injections: current sources merged per waveform (duplicate points summed), field
        injections one each, on flat indices."""
        xp = self.xp
        n_e, n_a = self.sizes
        groups = {}
        for position, stepper in enumerate(self.members):
            for injection in stepper._injections:
                entry = groups.setdefault(id(injection.temporal), (injection.temporal, [], []))
                entry[1].append(position * n_e + injection.indices)
                entry[2].append(injection.weights)
        merged = []
        for temporal, indices, weights in groups.values():
            points, inverse = np.unique(np.concatenate(indices), return_inverse=True)
            summed = np.bincount(inverse.ravel(), weights=np.concatenate(weights), minlength=points.size)
            merged.append((temporal, xp.asarray(points), xp.asarray(summed, dtype=DTYPE)))
        self._injection_groups = tuple(merged)
        field = []
        for position, stepper in enumerate(self.members):
            for injection in stepper._field_injections:
                offset = position * (n_a if injection.side == 'h' else n_e)
                field.append((injection, xp.asarray(offset + injection.indices),
                              xp.asarray(injection.weights.real, dtype=DTYPE),
                              xp.asarray(injection.weights.imag, dtype=DTYPE)))
        self._field_injections = tuple(field)

    def add_spectra(self, positions, frequencies, e_index=None, a_index=None, dtype=np.complex128):
        """[BORSpectra] of the members at `positions` from the next step on (_SpectraGroup arguments)."""
        group = _SpectraGroup(positions, frequencies, self.dt, self.xp, self.sizes, self.n_step + 1, e_index=e_index,
                              a_index=a_index, dtype=dtype)
        self._spectra.append(group)
        return group.views

    def _apply_field_injections(self, side, field, time):
        for injection, index, weight_re, weight_im in self._field_injections:
            if injection.side != side:
                continue
            signal, quadrature = DTYPE(injection.temporal(time)), DTYPE(injection.temporal.quadrature(time))
            if self.xp is np:
                field[index] += weight_re * signal + weight_im * quadrature
            else:
                _FIELD_INJECT(index, weight_re, weight_im, signal, quadrature, field)

    def _slave(self, kind, field):
        """|m| = 1 axis recovery of the flat field (module header): field[target] = factor field[source]."""
        slaving = self._slaving[kind]
        if slaving is None:
            return
        targets, sources, factors = slaving
        if self.xp is np:
            field[targets] = factors * field[sources]
        else:
            _AXIS_COPY(targets, sources, factors, field)

    def step(self):
        """One leap-frog step of every member: h to (n + 1/2) dt (its field injections at n dt), the poles from x^n, x
        to (n + 1) dt (pole corrections, field injections and the sources at (n + 1/2) dt), then the spectra."""
        source_time = (self.n_step + 0.5) * self.dt
        x, h = self._x_flat, self._h_flat
        self._h_side.advance(x, h)
        self._apply_field_injections('h', h, self.n_step * self.dt)
        self._slave('a', h)
        for pole in self._poles:
            pole.update(x)
        self._e_side.advance(h, x)
        for pole in self._poles:
            pole.correct(x)
        self._apply_field_injections('e', x, source_time)
        for temporal, points, weights in self._injection_groups:
            value = float(temporal(source_time))
            if value == 0.0:
                continue
            if self.xp is np:
                x[points] += weights * DTYPE(value)
            else:
                _POINT_ADD(points, weights, DTYPE(value), x)
        self._slave('e', x)
        self.n_step += 1
        for group in self._spectra:
            group.accumulate(self.x, x, self.h, h)


class BORTimeStepper:
    """Leap-frog fields of one azimuthal order on an (nr, nz) grid (module header): full-layout E vector x and a-point
    H vector h on xp (rows of its _LockstepBatch), host operators / coefficients, Drude / Lorentz poles, current
    injections and DFT spectra; host-built from a real-metric stretched_assembly (kept for coefficient refreshes and the
    flux / far-field weights)."""

    order_sum = False   # BOROrderSum: True

    def __init__(self, cell_widths, boundary_config, pml_dx, materials, azimuthal_order, courant=0.99, sigma_e=None,
                 sigma_m=None, window=None, xp=np, assembly=None, time_step=None, time_step_limit=None, poles=(),
                 own_batch=True):
        """cell_widths = (dr_cells, dz_cells) [m] (r from the axis), the r_hi / z faces of boundary_config (cpml, pec,
        pmc; pml_dx = CPML grading cell [m]), BORMaterials with real eps / mu (>= 1), integer order, courant in (0, 1]
        of the leap-frog limit, cell conductivities sigma_e [S/m] / sigma_m [Ohm/m] (None: 0); assembly = the
        stretched_assembly of the same grid and PEC to share (None: built here), time_step [s] replaces courant x the
        order's limit (an order sum's lockstep dt: the caller keeps it at or below time_step_limit [s], the order's
        limit, None: assumed equal); poles = [(DrudePole | LorentzPole, float64 (nr, nz) cell share 0..1)], fixed;
        own_batch False = a member whose _LockstepBatch the caller builds (BOROrderSum): no device state until then."""
        import scipy.sparse
        axis_class(azimuthal_order)
        courant = float(courant)
        if not 0.0 < courant <= 1.0:
            raise ValueError(f'BOR FDTD courant must lie in (0, 1], got {courant!r}')
        self.xp = xp
        self.order = int(azimuthal_order)
        self.orders = (self.order,)
        if assembly is None:
            assembly = stretched_assembly(cell_widths, boundary_config, pml_dx, materials, window=window)
        elif assembly.omega is not None:
            raise ValueError('BORTimeStepper needs the real-metric assembly (stretched_assembly)')
        self.assembly = assembly
        self.shape = assembly.shape
        curl_static, curl_order, self.e_active, self.a_active = assembly.curl_parts(self.order)
        self.e_volumes = np.asarray(assembly.full_energy_weights, dtype=np.float64)
        self.a_volumes = np.abs(assembly.a_volumes)
        e_sizes, a_sizes = layout_sizes(self.shape)
        self.sizes = (sum(e_sizes), sum(a_sizes))
        order_part = (self.order * curl_order).real.tocsr()
        z_part, r_part = _derivative_split(curl_static, self.shape)
        if time_step is None:
            self.dt = courant * stable_time_step(z_part + r_part + order_part, self.e_volumes, self.a_volumes,
                                                 self.e_active)
            self.stability_ratio = courant   # dt / the order's limit (the pole bound)
        else:
            self.dt = float(time_step)
            if not (math.isfinite(self.dt) and self.dt > 0.0):
                raise ValueError(f'BOR FDTD time_step must be finite and > 0 [s], got {time_step!r}')
            self.stability_ratio = 1.0 if time_step_limit is None else self.dt / float(time_step_limit)

        # operators (module header): H rows S C, E rows V^-1 C^T S V_a; z CPML (1 / kappa, psi drivers on its rows);
        # the r_hi layer rows through _RadialPlan (stepped by the batch's _BatchSide)
        diags = scipy.sparse.diags
        nr, nz = self.shape
        sign = np.ones(self.sizes[1])
        sign[:a_sizes[0]] = -1.0
        to_h = diags(sign)
        dual_scale, a_scale = diags(self.e_active / self.e_volumes), diags(self.a_volumes)

        def dual(part):
            return (dual_scale @ part.T @ to_h @ a_scale).tocsr()

        profile_z = cpml_axis_profiles(boundary_config, nz, 'y', pml_dx)
        h_z = _layout_profile(((nr, nz + 1, 'h'), (nr + 1, nz, None), (nr + 1, nz + 1, 'h')), profile_z)
        e_z = _layout_profile(((nr + 1, nz, 'e'), (nr, nz, 'e'), (nr, nz + 1, None)), profile_z)
        rates = _radial_rates(np.asarray(cell_widths[0], dtype=np.float64),
                              cpml_axis_profiles(boundary_config, nr, 'x', pml_dx), nz)
        h_parts = (to_h @ z_part, to_h @ r_part, to_h @ order_part)
        e_parts = (dual(z_part), dual(r_part), dual(order_part))
        self._plans = {'h': self._side(h_parts, h_z, None if rates is None else rates['h'], self.sizes[1]),
                       'e': self._side(e_parts, e_z, None if rates is None else rates['e'], self.sizes[0])}

        # |m| = 1 axis recovery (module header): E_r(0, j) = -u(0, j) / m, H_phi(0, l) = m H_r(0, l) (host indices)
        self._axis_slaving = None
        if _regular_axis(self.order):
            e_axis, _ = point_index(self.shape, {'er': (0, slice(None))})
            e_cell, _ = point_index(self.shape, {'ephi': (0, slice(None))})
            _, a_axis = point_index(self.shape, {'hphi': (0, slice(None))})
            _, a_cell = point_index(self.shape, {'hr': (0, slice(None))})
            self._axis_slaving = (e_axis, e_cell, DTYPE(-1.0 / self.order), a_axis, a_cell, DTYPE(self.order))
        self._batch = None       # _LockstepBatch holding x / h (rows `_position`) and stepping them
        self._position = 0
        self._owns_batch = bool(own_batch)
        self.x = self.h = self.fields = None   # device rows and 2D views (probes), set by _bind
        self._injections = []
        self._field_injections = []   # _FieldInjection of mode ports
        self._gain_e = None    # host float64 E gain dt / (eps0 eps (1 + loss)) (injection and pole weights)
        self._gain_h = None    # host float64 H gain -dt / (mu0 mu (1 + loss)) (field injection weights)
        self._decay = None     # {'e' | 'h': host float64 decay (1 - loss) / (1 + loss)}
        self.port_operators = (h_parts[0].tocsr(), e_parts[0].tocsr())   # host d_z parts S C_z, V^-1 C_z^T S V_a
        self._energy_e = self._energy_a = None
        self._pole_slots = [self._pole_points(pole, share) for pole, share in poles]   # None: no point of this order
        self._poles = [points for points in self._pole_slots if points is not None]
        self.refresh_coefficients(materials, sigma_e, sigma_m)
        if self._owns_batch:
            _LockstepBatch([self])

    def _bind(self, batch, position):
        """Rows `position` of the batch's stacked x / h become this stepper's x / h (called by _LockstepBatch)."""
        self._batch, self._position = batch, int(position)
        self.x, self.h = batch.x[self._position], batch.h[self._position]
        self.fields = SimpleNamespace(**layout_views(self.x, self.h, self.shape))   # 2D views (probes)

    @property
    def n_step(self) -> int:
        return 0 if self._batch is None else self._batch.n_step

    def _side(self, parts, profile_z, radial_rates, size):
        """_SidePlan of (d_z, d_r, m / r) parts with the per-row (sigma, kappa, alpha) along z and the r_hi layer rates
        (inner, outer, source integral weights; None without an r_hi cpml)."""
        import scipy.sparse
        z_part, r_part, order_part = parts
        radial_part = (r_part + order_part).tocsr()
        keep = np.ones(size)
        radial = None
        if radial_rates is not None:
            inner, outer, integral = radial_rates
            has_entries = np.asarray(abs(radial_part).sum(axis=1)).ravel() > 0.0
            rows = np.flatnonzero(((inner > 0.0) | (outer > 0.0)) & has_entries)
            if rows.size:
                keep[rows] = 0.0
                radial = _radial_plan(rows, r_part, order_part, inner[rows], outer[rows], integral, self.dt)
        main = (scipy.sparse.diags(1.0 / profile_z[1]) @ z_part + scipy.sparse.diags(keep) @ radial_part).tocsr()
        z_pml = None
        if np.any(profile_z[0] > 0.0):
            a, b = _recursion(*profile_z, self.dt)
            z_pml = ((scipy.sparse.diags((a != 0.0).astype(np.float64)) @ z_part).tocsr(), a, b)
        return _SidePlan(main=main, z_pml=z_pml, radial=radial, size=int(size), source_size=int(main.shape[1]))

    def _pole_points(self, pole, share):
        """_PolePoints of a pole on its float64 (nr, nz) cell share (the active E points it touches), None without."""
        share = np.asarray(share, dtype=np.float64)
        if share.shape != self.shape:
            raise ValueError(f'BOR FDTD pole cell share shape {share.shape} does not match the {self.shape} cells')
        weights = np.asarray(self.assembly.e_point_means(share, share, share), dtype=np.float64) * self.e_active
        indices = np.flatnonzero(weights > 0.0)
        return _PolePoints(pole, indices, weights[indices], self.dt) if indices.size else None

    @property
    def time(self) -> float:
        """Time [s] of x (h lags it by dt / 2)."""
        return self.n_step * self.dt

    def refresh_coefficients(self, materials, sigma_e=None, sigma_m=None):
        """Update coefficients from BORMaterials (real eps components, mu; the PEC of the build stays) and cell
        conductivities sigma_e [S/m], sigma_m [Ohm/m] (None: 0); ValueError for a point eps or mu below 1 or a pole
        bound eps - sum w P below (dt / limit)^2 (module header). An order-sum member refreshes the host values only
        (BOROrderSum uploads its batch once)."""
        assembly, xp, dt = self.assembly, self.xp, self.dt
        eps = assembly.e_point_means(*(materials.component(name).real for name in ('r', 'phi', 'z'))).real
        mu = assembly.a_point_means(np.asarray(materials.mu).real).real
        for name, values, active in (('permittivity', eps, self.e_active), ('permeability', mu, self.a_active)):
            if active.any() and float(np.min(values[active])) < 1.0 - _UNIT_SLACK:
                raise ValueError(f'em_fdtd_bor: a relative {name} below 1 ({float(np.min(values[active])):.4g}) at '
                                 'a field point breaks the time step (computed for eps = mu_r = 1)')
        if self._poles and self.e_active.any():
            from . import _pole_step_term   # the 2D engines' z = -1 pole term (package init done by then)
            pole_terms = np.zeros(self.sizes[0])
            for points in self._poles:
                pole_terms[points.indices] += points.weights * _pole_step_term(points.pole, dt)
            margin = eps - pole_terms - self.stability_ratio ** 2
            worst = int(np.argmin(np.where(self.e_active, margin, np.inf)))
            if margin[worst] < -_UNIT_SLACK:
                raise ValueError(
                    f'em_fdtd_bor: the dispersive poles break the time step at a field point: eps - P = '
                    f'{eps[worst] - pole_terms[worst]:.4g} < (time step / its limit)^2 = {self.stability_ratio ** 2:.4g} '
                    '(P = (omega_p dt)^2 / 4 per Drude pole, delta_eps (omega_0 dt)^2 / (4 - (omega_0 dt)^2) per '
                    'Lorentz pole); refine the mesh or lower courant_factor (both shrink dt), or raise eps_inf')
        conductivity = (np.zeros(self.sizes[0]) if sigma_e is None
                        else assembly.e_point_means(sigma_e, sigma_e, sigma_e).real)
        loss = conductivity * dt / (2.0 * EPS0 * eps)
        self._gain_e = dt / (EPS0 * eps * (1.0 + loss)) * self.e_active
        magnetic = np.zeros(self.sizes[1]) if sigma_m is None else assembly.a_point_means(sigma_m).real
        magnetic_loss = magnetic * dt / (2.0 * MU0 * mu)
        self._gain_h = -dt / (MU0 * mu * (1.0 + magnetic_loss)) * self.a_active
        self._decay = {'e': (1.0 - loss) / (1.0 + loss), 'h': (1.0 - magnetic_loss) / (1.0 + magnetic_loss)}
        self._energy_e = xp.asarray(0.5 * EPS0 * eps * self.e_volumes * self.e_active)
        self._energy_a = xp.asarray(0.5 * MU0 * mu * self.a_volumes * self.a_active)
        for injection in self._injections:
            self._set_weights(injection)
        for injection in self._field_injections:
            self._set_field_weights(injection)
        for points in self._poles:
            points.set_correction(self._gain_e)
        if self._batch is not None and self._owns_batch:
            self._batch.upload()

    def _set_weights(self, injection):
        injection.weights = self._gain_e[injection.indices] * injection.charges / self.e_volumes[injection.indices]

    def _set_field_weights(self, injection):
        gain = self._gain_h if injection.side == 'h' else self._gain_e
        injection.weights = gain[injection.indices] * injection.phasor

    def add_field_injection(self, side, indices, phasor, temporal):
        """_FieldInjection of update-term phasors [field units / s x dt / gain] on the 'h' (a layout) or 'e' (E layout)
        rows `indices`, applied as gain x (Re phasor s(t) + Im phasor s_q(t)) right after that update (h at the E time
        n dt, e at the H time (n + 1/2) dt); temporal needs a quadrature (fdtd_bor_ports)."""
        if not hasattr(temporal, 'quadrature'):
            raise TypeError(f'a field injection needs a waveform with a quadrature signal (gaussian or cw), got '
                            f'{type(temporal).__name__}')
        injection = _FieldInjection(side, indices, phasor, temporal)
        self._set_field_weights(injection)
        self._field_injections.append(injection)
        self._batch.injections_changed()
        return injection

    def add_current(self, ix, iz, component, current, temporal):
        """Current element of `component` 'r' | 'phi' | 'z' in cell (ix, iz): peak current [A] times temporal(t)
        (point_current_rhs normalisation incl. the |m| = 1 axis transfer; 'phi' = the em_fdfd_bor phi current at
        phase +90 deg) -> _Injection."""
        if component not in BOR_SOURCE_COMPONENTS:
            raise ValueError(f'source component must be one of {BOR_SOURCE_COMPONENTS}, got {component!r}')
        nr, nz = self.shape
        ix, iz = int(ix), int(iz)
        if not (0 <= ix < nr and 0 <= iz < nz):
            raise ValueError(f'source cell ({ix}, {iz}) is outside the {nr} x {nz} grid')
        (n_er, n_u, _), _ = layout_sizes(self.shape)
        current = float(current)
        if component == 'phi':
            indices, charges = [n_er + ix * nz + iz], [current * self.assembly.r_axis.real_centres[ix]]
        elif component == 'z':
            indices = [n_er + n_u + ix * (nz + 1) + edge for edge in (iz, iz + 1)]
            charges = [current * self.assembly.dz_cells[iz] / (4.0 * math.pi)] * 2
        else:
            half = current * self.assembly.dr_cells[ix] / (4.0 * math.pi)
            indices, charges = [edge * nz + iz for edge in (ix, ix + 1)], [half, half]
            if ix == 0 and _regular_axis(self.order):
                indices.append(n_er + iz)
                charges.append(-half / self.order)
        injection = _Injection(indices, charges, temporal)
        self._set_weights(injection)
        self._injections.append(injection)
        self._batch.injections_changed()
        return injection

    def add_spectra(self, frequencies, e_index=None, a_index=None, dtype=np.complex128):
        """BORSpectra of x / h entries (None: all) at frequencies [Hz] from the next step on, accumulated by the batch
        step."""
        return self._batch.add_spectra([self._position], frequencies, e_index=e_index, a_index=a_index, dtype=dtype)[0]

    def step(self):
        """One leap-frog step (_LockstepBatch.step); an order-sum member steps only through BOROrderSum.step."""
        if not self._owns_batch:
            raise ValueError('BORTimeStepper: an order-sum member steps in lockstep through BOROrderSum.step')
        self._batch.step()

    def yee_views(self):
        """layout_views of the current x and h (real time-domain samples on their Yee points)."""
        return layout_views(self.x, self.h, self.shape)

    def probe_value(self, component, cell):
        """0-d sample of a layout_views component at its (r, z) index (no host sync)."""
        return getattr(self.fields, component)[cell]

    def spectra_terms(self, spectra, index, scale=1.0):
        """[(order, phasor_yee dict)] host phasors of frequency `index` of a BORSpectra times scale (em_fdfd_bor
        convention, module header)."""
        return [(self.order, phasor_yee(*spectra.full_phasors(index, scale), self.shape))]

    def harmonic_yee(self, spectra, index):
        """phasor_yee dict on xp of frequency `index` of a full-layout BORSpectra."""
        return phasor_yee(spectra.e_spectra[index], spectra.a_spectra[index], self.shape)

    def field_energy(self) -> float:
        """0.5 sum (eps0 eps V x^2 + mu0 mu V_a h^2) [J per radian] over every point (a decay measure; h lags dt / 2)."""
        xp = self.xp
        return float(xp.dot(self._energy_e, self.x.astype(np.float64) ** 2)
                     + xp.dot(self._energy_a, self.h.astype(np.float64) ** 2))


# ---- driven order sums (module header) ----
FAMILY_RZ = 'rz'     # the stepped standing wave (E_r, E_z, H_phi ~ cos(m phi)): r / z currents at phi = 0; all at m = 0
FAMILY_PHI = 'phi'   # a quarter period on (E_phi, H_r, H_z ~ -cos(m phi)): phi currents at phi = 0
_COSINE_COMPONENTS = ('er', 'ez', 'hphi')   # cos(m phi) components of the stepped standing wave


def order_sum_families(order, components):
    """Families (FAMILY_RZ / FAMILY_PHI) of the steppers order m needs for point currents of `components`: m = 0 one
    (FAMILY_RZ holds every component), m > 0 FAMILY_RZ for r / z and FAMILY_PHI for phi currents (none: FAMILY_RZ)."""
    components = set(components)
    if int(order) == 0:
        return (FAMILY_RZ,)
    families = [FAMILY_RZ] if components & {'r', 'z'} or not components else []
    if 'phi' in components:
        families.append(FAMILY_PHI)
    return tuple(families)


def order_sum_current(order, family, component, current, radius, width) -> float:
    """Stepper current [A] of member (order, family) for a point current element at phi = 0 of moment I d (d = cell
    height for 'z', width w for 'r' / 'phi'; radius = cell centre r): the term I (r, z) or I w / (2 pi r) (phi) at m = 0,
    2 x it on the family carrying the component at m > 0 (FAMILY_PHI with sign -1), else 0 (module header)."""
    term = float(current) * (float(width) / (2.0 * math.pi * float(radius)) if component == 'phi' else 1.0)
    if int(order) == 0:
        return term
    if component == 'phi':
        return -2.0 * term if family == FAMILY_PHI else 0.0
    return 2.0 * term if family == FAMILY_RZ else 0.0


def phi_zero_weights(order, family):
    """{layout_views component: weight} of a member's real fields in the phi = 0 field (module header)."""
    names = ('er', 'ephi', 'ez', 'hr', 'hz', 'hphi')
    if int(order) == 0:
        return {name: 1.0 for name in names}
    if family == FAMILY_RZ:
        return {name: 1.0 if name in _COSINE_COMPONENTS else 0.0 for name in names}
    return {name: 0.0 if name in _COSINE_COMPONENTS else -1.0 for name in names}


def order_term_fields(order, family, e_values, a_values, shape):
    """[(signed order, yee_fields-style dict)] travelling exp(-j m phi) terms (physical phasors) of a member's full-layout
    DFT phasors (module header): m = 0 the views, m > 0 Y = phasor_yee / 2 at +m, its phi mirror at -m, x j / -j for
    FAMILY_PHI."""
    if int(order) == 0:
        return [(0, layout_views(e_values, a_values, shape))]
    half = {name: 0.5 * values for name, values in phasor_yee(e_values, a_values, shape).items()}
    partner = mirrored_yee(half)
    if family == FAMILY_PHI:
        half = {name: 1j * values for name, values in half.items()}
        partner = {name: -1j * values for name, values in partner.items()}
    return [(int(order), half), (-int(order), partner)]


def combine_terms(term_lists):
    """[(signed order, summed yee dict)] of term lists [(signed order, yee dict)], equal signed orders added."""
    combined = {}
    for terms in term_lists:
        for order, fields in terms:
            if order in combined:
                combined[order] = {name: combined[order][name] + values for name, values in fields.items()}
            else:
                combined[order] = dict(fields)
    return list(combined.items())


@dataclass(frozen=True, eq=False)
class OrderMember:
    """One stepper of a driven order sum: azimuthal order m >= 0, family FAMILY_RZ | FAMILY_PHI, BORTimeStepper."""
    order: int
    family: str
    stepper: BORTimeStepper


class OrderSumSpectra:
    """BORSpectra of every member of a BOROrderSum (members_spectra = [(OrderMember, BORSpectra)])."""

    def __init__(self, members_spectra):
        self.members_spectra = list(members_spectra)
        self.frequencies = self.members_spectra[0][1].frequencies

    def scale(self, factors):
        for _, spectra in self.members_spectra:
            spectra.scale(factors)


class BOROrderSum:
    """Driven order sum of point currents at phi = 0 (module header): one BORTimeStepper per (order, family) on one shared
    real-metric assembly, in lockstep at courant x the smallest order limit; the BORTimeStepper engine interface (dt,
    n_step, time, step, add_current, add_spectra, spectra_terms, harmonic_yee, probe_value, yee_views, field_energy,
    refresh_coefficients) over the summed 3D field (phi = 0 for fields and probes)."""
    order_sum = True

    def __init__(self, cell_widths, boundary_config, pml_dx, materials, orders, components, courant=0.99, sigma_e=None,
                 sigma_m=None, window=None, xp=np, poles=()):
        """orders = distinct integers >= 0 (each m > 0 stands for +m and -m), components = the point-current components
        the members must carry (order_sum_families); the other arguments as BORTimeStepper."""
        orders = tuple(int(order) for order in orders)
        if not orders or any(order < 0 for order in orders) or len(set(orders)) != len(orders):
            raise ValueError(f'BOR FDTD order sum needs distinct integer orders >= 0, got {orders}')
        courant = float(courant)
        if not 0.0 < courant <= 1.0:
            raise ValueError(f'BOR FDTD courant must lie in (0, 1], got {courant!r}')
        self.xp = xp
        self.orders = orders
        self.assembly = stretched_assembly(cell_widths, boundary_config, pml_dx, materials, window=window)
        self.shape = self.assembly.shape
        warm_start = {}   # each order's eigsh starts from the previous order's top eigenvector
        limits = {order: order_time_step_limit(self.assembly, order, warm_start) for order in sorted(orders)}
        self.dt = courant * min(limits.values())
        self.members = tuple(
            OrderMember(order, family, BORTimeStepper(cell_widths, boundary_config, pml_dx, materials, order,
                                                      sigma_e=sigma_e, sigma_m=sigma_m, window=window, xp=xp,
                                                      assembly=self.assembly, time_step=self.dt,
                                                      time_step_limit=limits[order], poles=poles, own_batch=False))
            for order in orders for family in order_sum_families(order, components))
        self._batch = _LockstepBatch([member.stepper for member in self.members])   # every member in one launch
        self._weights = [(member, phi_zero_weights(member.order, member.family)) for member in self.members]

    @property
    def n_step(self) -> int:
        return self.members[0].stepper.n_step

    @property
    def time(self) -> float:
        """Time [s] of x (h lags it by dt / 2)."""
        return self.members[0].stepper.time

    def refresh_coefficients(self, materials, sigma_e=None, sigma_m=None):
        for member in self.members:
            member.stepper.refresh_coefficients(materials, sigma_e, sigma_m)
        self._batch.upload()

    def add_current(self, ix, iz, component, current, temporal):
        """Point current element at phi = 0 (moment I d, order_sum_current) in cell (ix, iz) -> [_Injection] of the
        members carrying it; ValueError when an order has no member for the component (add sources before the build)."""
        if component not in BOR_SOURCE_COMPONENTS:
            raise ValueError(f'source component must be one of {BOR_SOURCE_COMPONENTS}, got {component!r}')
        nr, nz = self.shape
        if not (0 <= int(ix) < nr and 0 <= int(iz) < nz):
            raise ValueError(f'source cell ({ix}, {iz}) is outside the {nr} x {nz} grid')
        radius, width = self.assembly.r_axis.real_centres[int(ix)], self.assembly.dr_cells[int(ix)]
        injections, carried = [], set()
        for member in self.members:
            value = order_sum_current(member.order, member.family, component, current, radius, width)
            if value != 0.0:
                injections.append(member.stepper.add_current(ix, iz, component, value, temporal))
                carried.add(member.order)
        missing = sorted(set(self.orders) - carried)
        if missing and float(current) != 0.0:
            raise ValueError(f"BOR FDTD order sum: orders {missing} have no stepper for a '{component}' current (the "
                             'members follow the source components at the build: add sources before it)')
        return injections

    def add_spectra(self, frequencies, e_index=None, a_index=None, dtype=np.complex128):
        """OrderSumSpectra of every member (BORTimeStepper.add_spectra arguments; one stacked _SpectraGroup)."""
        views = self._batch.add_spectra(range(len(self.members)), frequencies, e_index=e_index, a_index=a_index,
                                        dtype=dtype)
        return OrderSumSpectra(zip(self.members, views))

    def step(self):
        """One lockstep leap-frog step of every member (_LockstepBatch.step)."""
        self._batch.step()

    def spectra_terms(self, spectra, index, scale=1.0):
        """[(signed order, yee dict)] host physical phasors of the summed field's exp(-j m phi) terms at frequency
        `index` of an OrderSumSpectra times scale (order_term_fields per member, combine_terms)."""
        return combine_terms(order_term_fields(member.order, member.family, *member_spectra.full_phasors(index, scale),
                                               self.shape) for member, member_spectra in spectra.members_spectra)

    def _phi_zero_sum(self, views_of):
        """phi = 0 sum of views_of(member) (layout_views-style dicts) with phi_zero_weights."""
        total = {}
        for member, weights in self._weights:
            views = views_of(member)
            for name, weight in weights.items():
                if weight == 0.0:
                    continue
                part = weight * views[name]
                total[name] = part if name not in total else total[name] + part
        for name, values in views_of(self.members[0]).items():   # components no member carries
            if name not in total:
                total[name] = self.xp.zeros_like(values)
        return total

    def harmonic_yee(self, spectra, index):
        """Physical phi = 0 phasors (dict on xp, layout_views names) of frequency `index` of a full-layout
        OrderSumSpectra."""
        by_member = {id(member): member_spectra for member, member_spectra in spectra.members_spectra}

        def views(member):
            member_spectra = by_member[id(member)]
            return layout_views(member_spectra.e_spectra[index], member_spectra.a_spectra[index], self.shape)
        return self._phi_zero_sum(views)

    def yee_views(self):
        """phi = 0 real fields (new arrays, layout_views names) of the summed field."""
        return self._phi_zero_sum(lambda member: member.stepper.yee_views())

    def probe_value(self, component, cell):
        """0-d phi = 0 sample of the summed field (no host sync)."""
        value = None
        for member, weights in self._weights:
            weight = weights[component]
            if weight != 0.0:
                part = weight * member.stepper.probe_value(component, cell)
                value = part if value is None else value + part
        return self.members[0].stepper.probe_value(component, cell) * 0.0 if value is None else value

    def field_energy(self) -> float:
        """Sum of the members' field_energy (a decay measure)."""
        return float(sum(member.stepper.field_energy() for member in self.members))


def stepper_bytes(shape, order, z_pml_rows, radial_rows):
    """(device bytes, host bytes) estimate of a BORTimeStepper of an (nr, nz) grid at `order`: float32 vectors, CSR
    products (main curls; z psi drivers on z_pml_rows = E + a rows of the z CPML layers, 0 without), the r_hi layer
    rows (radial_rows = E + a rows, 0 without), host assembly."""
    e_sizes, a_sizes = layout_sizes(shape)
    n_e, n_a = sum(e_sizes), sum(a_sizes)
    vectors = (4 + 4 * bool(z_pml_rows)) * (n_e + n_a)   # field, decay, gain, curl (+ drive, a, b, psi)
    curl_nnz = 2 * a_sizes[0] + 2 * a_sizes[1] + 4 * a_sizes[2] + (a_sizes[0] + a_sizes[1]) * (order != 0)
    csr_entry, index = 8, 4   # float32 data + int32 column; int32 row pointer
    products = 2 * curl_nnz * csr_entry + (n_e + n_a + 2) * index
    if z_pml_rows:   # two drivers: full row pointers, ~2 entries per stretched row
        products += (n_e + n_a + 4) * index + 2 * z_pml_rows * csr_entry
    # r_hi layer: 9 float32 + int64 row per row, ~4 CSR entries + 3 row pointers (derivative, integral, order products)
    radial = radial_rows * (9 * 4 + 8 + 4 * csr_entry + 3 * index + 2 * 4)   # + the integral and its column (~1 / 3)
    device = 4 * vectors + products + radial + 8 * (n_e + n_a)   # + float64 energy weights
    host = 40 * 24 * n_e + 16 * (n_e + n_a)   # kept assembly (complex128 curl split, class operators; rough)
    return device, host
