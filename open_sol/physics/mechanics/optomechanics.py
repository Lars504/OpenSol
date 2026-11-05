"""Vacuum optomechanical coupling g0 between one optical mode and the solid vibration modes of a 2D model (host numpy):
moving-boundary face sums and the photoelastic overlap of the nodal mode shape, effective mass and zero-point motion."""
# Conventions (optomechanics_plan.md): exp(+j omega t) peak phasors; Q = nodal mode shape on the window nodes. Faces lie
# between window cells a (lo) and b (hi) along x or y, Delta eps = eps_a - eps_b (Q_d > 0 moves a into b), Q_d of a
# face = the mean of its two corner nodes (the exact edge mean of bilinear u). Moving boundary per face (Johnson et al.,
# PRE 65, 066611): Q_d [eps0 Delta eps |E_par|^2 - Delta(1/eps_nn) |D_n|^2 / eps0] l_f; photoelastic per cell:
# eps0 E* Delta eps E dA, Delta eps_ii = -eps_ii^2 (p11 S_ii + p12 (S_jj + S_kk)) of the element-centre strain S of Q
# (solid cells only; plane strain S_zz = S_xz = S_yz = 0) plus the shear term Delta eps_xy = -eps_xx eps_yy 2 p44 S_xy
# (Voigt: Delta(1/eps)_6 = p44 x the engineering shear; unset p44 = (p11 - p12) / 2, isotropic), summed as
# eps0 Delta eps_xy 2 Re(E_x* E_y) dA with the in-plane E plain-averaged to the cell centres (ez: no E_x, E_y, no term).
# The diagonal-eps optical operators have no off-diagonal entry, so the shear term is the continuum form, not the exact
# discrete perturbation the diagonal terms are. General form (crystal cells, `inverse_permittivity_changes`): Nye
# Delta(1/eps)_I = p_IJ S_J (Voigt I, J = xx, yy, zz, yz, xz, xy; engineering shear S_4..6 = 2 S_ij, B_I tensor
# components), Delta eps_ij = -eps_ii eps_jj Delta(1/eps)_ij; isotropic cells p_ii = p11, p_ij = p12, shear rows p44;
# a crystal region's rotated (6, 6) p (materials.crystal_tensors, model axes) adds xz / yz rows even for plane strain,
# paired with E_z as E* . Delta eps E: 2 Re(E_x* E_z) Delta eps_xz forward, 2 i Im(E_x* E_z) backward.
# Subpixel smoothing: every face jump and photoelastic eps is the unsmoothed staircase cell eps (the build's material
# assignment; ez subtracts the smoothing deltas from the system eps), fields, D_n and the mode energy are the solve's.
# In-plane ez resonance: g_om = -(omega / 2) Re[(I_mb + I_pe) / sum eps0 eps |E_z|^2 dA] over the window cells.
# In-plane hz resonance (E_x on the hi-y, E_y on the hi-x faces): the same with the magnetic energy sum mu0 mu |H_z|^2 dA
# (= the electric one at a discrete resonance: exact Rayleigh quotient); a material face takes its tangential E on the
# face itself and |D_n|^2 as the mean over the four nearest normal-E faces; I_pe sums eps0 |E_f|^2 (face_average of the
# cell Delta eps) over the E faces with their dual areas, the exact first-order change of the hz operator.
# Cross-section mode (1 W, the whole mode): d beta / d alpha = omega m (I_mb + I_pe) / (4 P), m = 2 per mirror face;
# a ring at fixed azimuthal order: g_om = -(C0 / n_g) Re(d beta / d alpha).
# g0 = g_om x_zpf, x_zpf = sqrt(hbar / (2 m_eff Omega)), m_eff = L m Q^T M Q with Q scaled to |Q| = 1 at the reference
# node (the peak node, or the solid node nearest a point): g0 does not depend on that choice.
# Forward intramodal Brillouin gain of a cross-section mode (k_z = 0 vibration, pump and Stokes in the same optical
# mode): G_B = 2 Q_m (d beta / d alpha)^2 / (omega m' Omega^2) [1/(W m)], m' = m_eff / L (from the driven damped
# oscillator with the optical force per length (P / omega) d beta / d alpha; = 4 g0^2 L Q_m / (hbar omega Omega v_g^2)).
# Backward Brillouin scattering (cross-section mode, guided vibration at k_z = 2 Re beta): the Stokes mode is the
# counter-propagating twin E_s = (E_x, E_y, -E_z) of the pump, the phonon u ~ exp(i (k_z z - Omega t)) = guided_modes'
# (u_x, u_y, -i w); phase matching drives the Stokes by Delta eps of the conjugated optical-convention strain, which
# is the guided_modes complex strain: S_xx, S_yy, S_xy real, S_zz = k_z w, S_xz = i T_xz, S_yz = i T_yz with T_xz =
# (k_z u_x - w_x) / 2 (T_yz alike). kappa = omega m (I_mb + I_pe) / (4 P) with E_s* . Delta eps E_p: the E_z
# terms of both sums flip sign, Delta eps_zz gains p11 S_zz, Delta eps_xx / _yy p12 S_zz, and Delta eps_xz = -eps_xx
# eps_zz 2 p44 S_xz pairs (E_x* E_z - E_z* E_x) (yz alike). G_B = 2 Q_m |kappa|^2 / (omega m' Omega^2), m' = L^-1 m_eff
# with the 3-component mass; a bulk plane wave gives the classical g_B = 2 pi n^7 p12^2 / (c lambda^2 rho v Delta nu)
# per area. The g_om / g0 columns then hold the forward-backward coupling rate -(C0 / n_g) Re kappa (x x_zpf).
import math

import numpy as np

from ...core.constants import C0, EPS0, HBAR

NORMALISATIONS = ('max', 'point')
BOR_POLARIZATION = 'bor'   # = physics/em solvers.fdfd_bor.BOR_POLARIZATION: em_fdfd_bor optics (optomechanics_fem)
SCATTERING = ('forward', 'backward')   # optomechanical_coupling: k_z = 0 plane strain, or k_z = 2 beta guided modes
OPTOMECHANICS_TABLE_COLUMNS = ('mechanical_mode', 'frequency', 'effective_mass', 'zero_point_motion',
                               'g_om_moving_boundary', 'g_om_photoelastic', 'g_om', 'dn_eff_dx',
                               'g0_moving_boundary', 'g0_photoelastic', 'g0', 'brillouin_gain')
OPTOMECHANICS_TABLE_UNITS = ('1', 'Hz', 'kg', 'm', 'rad/s/m', 'rad/s/m', 'rad/s/m', '1/m', 'Hz', 'Hz', 'Hz', '1/(W m)')
_MIRROR_NAMES = ('symmetric', 'antisymmetric')   # BCType values of the EM mirror faces


# ---- face helpers (numpy or cupy arrays; shared with optical_force) ----
def face_mean(values, axis):
    """Mean of the two cells of every interior face along axis (0: x faces (n_x - 1, n_y), 1: y faces)."""
    if axis == 0:
        return 0.5 * (values[:-1, :] + values[1:, :])
    return 0.5 * (values[:, :-1] + values[:, 1:])


def face_jump(values, axis):
    """values_lo - values_hi across every interior face along axis (the material that moves in for Q_d > 0)."""
    if axis == 0:
        return values[:-1, :] - values[1:, :]
    return values[:, :-1] - values[:, 1:]


# ---- mechanical side ----
def face_displacements(nodal):
    """(Q_x on the interior x faces (n_x - 1, n_y), Q_y on the interior y faces (n_x, n_y - 1)) of nodal (n_x + 1,
    n_y + 1, 2) [m]: the mean of each face's two corner nodes."""
    return (0.5 * (nodal[1:-1, :-1, 0] + nodal[1:-1, 1:, 0]),
            0.5 * (nodal[:-1, 1:-1, 1] + nodal[1:, 1:-1, 1]))


def element_strains(nodal, dx, dy):
    """(S_xx, S_yy, S_xy) at the element centres (tensor shear) of nodal (n_x + 1, n_y + 1, 2) on widths dx, dy [m]."""
    south_west, south_east = nodal[:-1, :-1], nodal[1:, :-1]
    north_east, north_west = nodal[1:, 1:], nodal[:-1, 1:]
    d_dx = ((south_east - south_west) + (north_east - north_west)) / (2.0 * np.asarray(dx)[:, None, None])
    d_dy = ((north_west - south_west) + (north_east - south_east)) / (2.0 * np.asarray(dy)[None, :, None])
    return d_dx[..., 0], d_dy[..., 1], 0.5 * (d_dy[..., 0] + d_dx[..., 1])


def guided_axial_strains(nodal, dx, dy, axial_wavenumber):
    """(S_zz, T_xz, T_yz) at the element centres of a guided nodal (u_x, u_y, w) (n_x + 1, n_y + 1, 3) at k_z [rad/m]:
    S_zz = k_z w, T_xz = (k_z u_x - w_x) / 2, T_yz = (k_z u_y - w_y) / 2 (complex shears S_xz = i T_xz, S_yz = i T_yz)."""
    south_west, south_east = nodal[:-1, :-1], nodal[1:, :-1]
    north_east, north_west = nodal[1:, 1:], nodal[:-1, 1:]
    centre = 0.25 * (south_west + south_east + north_east + north_west)
    d_dx = ((south_east - south_west) + (north_east - north_west)) / (2.0 * np.asarray(dx)[:, None, None])
    d_dy = ((north_west - south_west) + (north_east - south_east)) / (2.0 * np.asarray(dy)[None, :, None])
    wavenumber = float(axial_wavenumber)
    return (wavenumber * centre[..., 2], 0.5 * (wavenumber * centre[..., 0] - d_dx[..., 2]),
            0.5 * (wavenumber * centre[..., 1] - d_dy[..., 2]))


def photoelastic_changes(eps_xx, eps_yy, eps_zz, p11, p12, strain_xx, strain_yy, strain_zz=0.0):
    """(Delta eps_xx, _yy, _zz) = -eps_ii^2 (p11 S_ii + p12 (S_jj + S_kk)) (diagonal eps; plane strain: S_zz = 0)."""
    return (-eps_xx ** 2 * (p11 * strain_xx + p12 * (strain_yy + strain_zz)),
            -eps_yy ** 2 * (p11 * strain_yy + p12 * (strain_xx + strain_zz)),
            -eps_zz ** 2 * (p11 * strain_zz + p12 * (strain_xx + strain_yy)))


def shear_photoelastic_coefficient(p11, p12, p44=None):
    """p44 per cell: the given values, the isotropic (p11 - p12) / 2 where p44 is None or NaN (unset)."""
    isotropic = 0.5 * (np.asarray(p11, dtype=np.float64) - np.asarray(p12, dtype=np.float64))
    if p44 is None:
        return isotropic
    p44 = np.asarray(p44, dtype=np.float64)
    return np.where(np.isnan(p44), isotropic, p44)


def inverse_permittivity_changes(voigt_strains, p11, p12, p44, crystal_regions=()):
    """(6, n_x, n_y) complex Delta(1/eps)_I = p_IJ S_J of the Voigt strains (S_xx, S_yy, S_zz, 2 S_yz, 2 S_xz, 2 S_xy;
    arrays or scalars): isotropic p11 / p12 / p44 cells, a (window mask, (6, 6) p) crystal region its full tensor."""
    s_xx, s_yy, s_zz, s_yz, s_xz, s_xy = voigt_strains
    rows = (p11 * s_xx + p12 * (s_yy + s_zz), p11 * s_yy + p12 * (s_xx + s_zz), p11 * s_zz + p12 * (s_xx + s_yy),
            p44 * s_yz, p44 * s_xz, p44 * s_xy)
    changes = np.stack(np.broadcast_arrays(*rows)).astype(np.complex128)
    if crystal_regions:
        strains = np.stack(np.broadcast_arrays(*voigt_strains, changes[0])[:6]).astype(np.complex128)
        for mask, tensor in crystal_regions:
            changes[:, mask] = np.asarray(tensor, dtype=np.float64) @ strains[:, mask]
    return changes


def permittivity_changes(eps_xx, eps_yy, eps_zz, inverse_changes):
    """(Delta eps_xx, _yy, _zz, _yz, _xz, _xy) = -eps_ii eps_jj Delta(1/eps)_ij of diagonal eps and Voigt-ordered
    Delta(1/eps) rows."""
    inverse_xx, inverse_yy, inverse_zz, inverse_yz, inverse_xz, inverse_xy = inverse_changes
    return (-eps_xx ** 2 * inverse_xx, -eps_yy ** 2 * inverse_yy, -eps_zz ** 2 * inverse_zz,
            -eps_yy * eps_zz * inverse_yz, -eps_xx * eps_zz * inverse_xz, -eps_xx * eps_yy * inverse_xy)


def nearest_solid_node(solid, x_nodes, y_nodes, x, y):
    """(i, j) of the window node touching a solid cell nearest (x, y) [m]; ValueError without solid cells."""
    touching = np.zeros((solid.shape[0] + 1, solid.shape[1] + 1), dtype=bool)
    touching[:-1, :-1] |= solid
    touching[1:, :-1] |= solid
    touching[:-1, 1:] |= solid
    touching[1:, 1:] |= solid
    if not touching.any():
        raise ValueError('the solid has no cell with youngs_modulus or c11 > 0: no reference node')
    distance = (np.asarray(x_nodes)[:, None] - x) ** 2 + (np.asarray(y_nodes)[None, :] - y) ** 2
    distance = np.where(touching, distance, np.inf)
    return np.unravel_index(int(np.argmin(distance)), distance.shape)


# ---- optical side ----
class OpticalOverlap:
    """Optical side of g0 on the window cells: moving-boundary weights per interior face (I_mb = sum Q_x face_x + sum
    Q_y face_y), the cell eps (xx, yy, zz) the photoelastic change uses, photoelastic(d_xx, d_yy, d_zz, d_xy, d_xz,
    d_yz) -> I_pe, g_om = Re(scale I) [rad/s/m] and, for a cross-section mode, d beta / d alpha = Re(beta_scale
    I) [1/m^2] (backward: kappa = beta_scale I between the mode and its counter-propagating twin)."""

    def __init__(self, face_x, face_y, cell_eps, photoelastic, scale, omega, mirror_factor, beta_scale=None,
                 n_eff=None, n_group=None, backward=False):
        self.backward = bool(backward)
        self.face_x = face_x
        self.face_y = face_y
        self.cell_eps = cell_eps
        self.photoelastic = photoelastic
        self.scale = scale
        self.omega = float(omega)
        self.mirror_factor = float(mirror_factor)
        self.beta_scale = beta_scale
        self.n_eff = n_eff
        self.n_group = n_group


def in_plane_ez_overlap(field, eps, dx, dy, omega, mirror_factor=1.0, staircase_eps=None):
    """OpticalOverlap of an in-plane E_z resonance: window cell field / complex eps of the solve (n_x, n_y), widths [m],
    omega [rad/s]; mirror_factor only multiplies m_eff (numerator and denominator both lose the mirrored half);
    staircase_eps = the unsmoothed cell eps of the face jumps / photoelastic changes (None: eps)."""
    field = np.asarray(field, dtype=np.complex128)
    eps = np.asarray(eps, dtype=np.complex128)
    staircase = eps if staircase_eps is None else np.asarray(staircase_eps, dtype=np.complex128)
    dx, dy = np.asarray(dx, dtype=np.float64), np.asarray(dy, dtype=np.float64)
    intensity = EPS0 * np.abs(field) ** 2
    weight = intensity * np.outer(dx, dy)
    norm = complex(np.sum(eps * weight))
    if norm == 0.0:
        raise ValueError('the optical resonance field vanishes on the window')
    face_x = face_mean(intensity, 0) * face_jump(staircase, 0) * dy[None, :]
    face_y = face_mean(intensity, 1) * face_jump(staircase, 1) * dx[:, None]

    def photoelastic(change_xx, change_yy, change_zz, change_xy, change_xz=None, change_yz=None):
        """sum eps0 |E_z|^2 Delta eps_zz dA (no in-plane E: the other changes pair with nothing)."""
        return complex(np.sum(change_zz * weight))

    return OpticalOverlap(face_x, face_y, (staircase, staircase, staircase), photoelastic, -0.5 * float(omega) / norm,
                          omega, mirror_factor)


def _lower_faces(face_values, axis, periodic):
    """Each cell's lo-face value along axis (a[k - 1]): the last face wraps in (periodic), else cell 0 reads 0, the
    missing lo face of the FDFD operator."""
    if periodic:
        return np.roll(face_values, 1, axis=axis)
    shifted = np.zeros_like(face_values)
    if axis == 0:
        shifted[1:, :] = face_values[:-1, :]
    else:
        shifted[:, 1:] = face_values[:, :-1]
    return shifted


def in_plane_hz_overlap(field, ex_faces, ey_faces, eps_xx, eps_yy, eps_ex_faces, eps_ey_faces, mu, cell_widths,
                        window, periodic, omega, mirror_factor=1.0):
    """OpticalOverlap of an in-plane H_z resonance from full-grid host arrays: H_z cells, E_x on the hi-y and E_y on the
    hi-x faces [V/m at omega], cell eps_xx / eps_yy, eps_xx at the E_x and eps_yy at the E_y faces, mu cells, the
    (dx_cells, dy_cells) widths [m], window slices, (periodic_x, periodic_y), omega [rad/s]."""
    from ..em.yee_grid import face_average, face_partner
    field, ex_faces, ey_faces, eps_xx, eps_yy, eps_ex_faces, eps_ey_faces, mu = (
        np.asarray(values, dtype=np.complex128)
        for values in (field, ex_faces, ey_faces, eps_xx, eps_yy, eps_ex_faces, eps_ey_faces, mu))
    dx_cells, dy_cells = (np.asarray(widths, dtype=np.float64) for widths in cell_widths)
    window_x, window_y = window
    periodic_x, periodic_y = periodic
    # magnetic energy over the window; mu0 = 1 / (eps0 C0^2), the constants the curl and the operator's k0 use
    norm = complex(np.sum(mu[window_x, window_y] * np.abs(field[window_x, window_y]) ** 2
                          * np.outer(dx_cells[window_x], dy_cells[window_y]))) / (EPS0 * C0 ** 2)
    if norm == 0.0:
        raise ValueError('the optical resonance field vanishes on the window')
    abs_ex, abs_ey = np.abs(ex_faces) ** 2, np.abs(ey_faces) ** 2
    normal_x = np.abs(eps_ex_faces * ex_faces) ** 2   # |D_x / eps0|^2 on the E_x faces
    normal_y = np.abs(eps_ey_faces * ey_faces) ** 2
    centre_normal_x = 0.5 * (normal_x + _lower_faces(normal_x, 1, periodic_y))   # mean of the cell's two y faces
    centre_normal_y = 0.5 * (normal_y + _lower_faces(normal_y, 0, periodic_x))

    # x faces between window cells i and i + 1: E_y tangential on the face itself, D_x the mean of the 4 E_x faces
    lo, hi = slice(window_x.start, window_x.stop - 1), slice(window_x.start + 1, window_x.stop)
    tangential = (eps_yy[lo, window_y] - eps_yy[hi, window_y]) * abs_ey[lo, window_y]
    normal = (1.0 / eps_xx[lo, window_y] - 1.0 / eps_xx[hi, window_y]) * 0.5 * (
        centre_normal_x[lo, window_y] + centre_normal_x[hi, window_y])
    face_x = EPS0 * (tangential - normal) * dy_cells[window_y][None, :]

    # y faces between cells j and j + 1: E_x tangential on the face itself, D_y the mean of the 4 E_y faces
    lo, hi = slice(window_y.start, window_y.stop - 1), slice(window_y.start + 1, window_y.stop)
    tangential = (eps_xx[window_x, lo] - eps_xx[window_x, hi]) * abs_ex[window_x, lo]
    normal = (1.0 / eps_yy[window_x, lo] - 1.0 / eps_yy[window_x, hi]) * 0.5 * (
        centre_normal_y[window_x, lo] + centre_normal_y[window_x, hi])
    face_y = EPS0 * (tangential - normal) * dx_cells[window_x][:, None]

    dual_x = 0.5 * (dx_cells + face_partner(dx_cells, 0, periodic_x))   # widths of the hi-x face dual cells
    dual_y = 0.5 * (dy_cells + face_partner(dy_cells, 0, periodic_y))
    weight_ex = EPS0 * abs_ex * np.outer(dx_cells, dual_y)
    weight_ey = EPS0 * abs_ey * np.outer(dual_x, dy_cells)
    centre_ex = 0.5 * (ex_faces + _lower_faces(ex_faces, 1, periodic_y))   # plain centre means (missing lo face 0)
    centre_ey = 0.5 * (ey_faces + _lower_faces(ey_faces, 0, periodic_x))
    weight_xy = (EPS0 * 2.0 * (np.conj(centre_ex) * centre_ey).real)[window_x, window_y] * np.outer(
        dx_cells[window_x], dy_cells[window_y])

    def photoelastic(change_xx, change_yy, change_zz, change_xy, change_xz=None, change_yz=None):
        """sum eps0 |E_f|^2 (face_average of the window cell Delta eps) dA_f over the E_x / E_y faces, plus the centre
        shear term eps0 Delta eps_xy 2 Re(E_x* E_y) dA (no E_z: the zz / xz / yz changes pair with nothing)."""
        total = complex(np.sum(change_xy * weight_xy))
        for change, weight, axis, periodic_axis, widths in ((change_xx, weight_ex, 1, periodic_y, dy_cells),
                                                            (change_yy, weight_ey, 0, periodic_x, dx_cells)):
            full = np.zeros(field.shape, dtype=np.complex128)
            full[window_x, window_y] = change
            total += complex(np.sum(face_average(full, axis, periodic_axis, widths) * weight))
        return total

    cell_eps = (eps_xx[window_x, window_y], eps_yy[window_x, window_y], eps_yy[window_x, window_y])   # zz unused
    return OpticalOverlap(face_x, face_y, cell_eps, photoelastic, -0.5 * float(omega) / norm, omega, mirror_factor)


def cross_section_overlap(mode, points, window, n_group=None, backward=False):
    """OpticalOverlap of a cross_section_modes.CrossSectionMode with its ModePointMaterials on the window slices
    (full-grid fields, faces between window cells); n_group None or NaN: g_om NaN (d beta / d alpha stays); backward:
    the coupling to the counter-propagating twin (E_x, E_y, -E_z), photoelastic with the xz / yz shear changes."""
    from ..em.yee_grid import face_average, lower_face_partner
    window_x, window_y = window
    periodic_x, periodic_y = points.periodic
    z_sign = -1.0 if backward else 1.0   # E_s,z* E_p,z of the backward Stokes twin
    abs_x, abs_y, abs_z = (np.abs(np.asarray(values)) ** 2 for values in (mode.ex, mode.ey, mode.ez))
    abs_z = z_sign * abs_z
    centre_x = 0.5 * (abs_x + lower_face_partner(abs_x, 0, periodic_x))   # |E_x|^2 at the centres
    centre_y = 0.5 * (abs_y + lower_face_partner(abs_y, 1, periodic_y))

    # x faces between cells i and i + 1 of the window: E_y, E_z tangential, E_x normal (on that face)
    lo, hi = slice(window_x.start, window_x.stop - 1), slice(window_x.start + 1, window_x.stop)
    # bend factors (ModePointMaterials): transverse eps_xx / eps_yy and longitudinal eps_zz, at the face's own x
    factor, z_factor = points.face_factor[lo][:, None], points.z_face_factor[lo][:, None]
    tangential = factor * (points.cell_eps_yy[lo, window_y] - points.cell_eps_yy[hi, window_y]) * 0.5 * (
        centre_y[lo, window_y] + centre_y[hi, window_y])
    tangential = tangential + z_factor * (points.cell_eps_zz[lo, window_y] - points.cell_eps_zz[hi, window_y]) * (
        0.5 * (abs_z[lo, window_y] + abs_z[hi, window_y]))
    normal = (1.0 / points.cell_eps_xx[lo, window_y] - 1.0 / points.cell_eps_xx[hi, window_y]) * np.abs(
        points.eps_x_faces[lo, window_y] * mode.ex[lo, window_y]) ** 2
    face_x = EPS0 * (tangential - normal / factor) * points.y_centre_length[window_y][None, :]

    # y faces between cells j and j + 1: E_x, E_z tangential, E_y normal
    lo, hi = slice(window_y.start, window_y.stop - 1), slice(window_y.start + 1, window_y.stop)
    factor, z_factor = points.centre_factor[window_x][:, None], points.z_centre_factor[window_x][:, None]
    tangential = factor * (points.cell_eps_xx[window_x, lo] - points.cell_eps_xx[window_x, hi]) * 0.5 * (
        centre_x[window_x, lo] + centre_x[window_x, hi])
    tangential = tangential + z_factor * (points.cell_eps_zz[window_x, lo] - points.cell_eps_zz[window_x, hi]) * (
        0.5 * (abs_z[window_x, lo] + abs_z[window_x, hi]))
    normal = (1.0 / points.cell_eps_yy[window_x, lo] - 1.0 / points.cell_eps_yy[window_x, hi]) * np.abs(
        points.eps_y_faces[window_x, lo] * mode.ey[window_x, lo]) ** 2
    face_y = EPS0 * (tangential - normal / factor) * points.x_centre_length[window_x][:, None]

    shape = points.cell_eps_zz.shape
    weight_x = EPS0 * abs_x * mode.x_face_area * points.face_factor[:, None]
    weight_y = EPS0 * abs_y * mode.y_face_area * points.centre_factor[:, None]
    weight_z = EPS0 * abs_z * points.centre_area * points.z_centre_factor[:, None]
    ex, ey = np.asarray(mode.ex), np.asarray(mode.ey)
    centre_ex = 0.5 * (ex + lower_face_partner(ex, 0, periodic_x))   # plain centre means
    centre_ey = 0.5 * (ey + lower_face_partner(ey, 1, periodic_y))
    centre_cross = (np.conj(centre_ex) * centre_ey).real   # Re(E_x* E_y) at the centres
    weight_xy = (EPS0 * 2.0 * centre_cross * points.centre_area * points.centre_factor[:, None])[window_x, window_y]
    # E_s* . Delta eps_xz E_p of the xz pair: forward (E_s = E_p) Delta eps_xz 2 Re(E_x* E_z), backward (E_s,z = -E_z)
    # Delta eps_xz (E_x* E_z - E_z* E_x) = Delta eps_xz 2 i Im(E_x* E_z); yz alike
    ez = np.asarray(mode.ez)

    def pair(transverse):
        product = np.conj(transverse) * ez
        return 2.0j * product.imag if backward else 2.0 * product.real

    # bend map of an xz entry: sqrt of the eps_xx and eps_zz factors (exact model s * (1 / s), conformal s^2)
    shear_area = points.centre_area * np.sqrt(points.centre_factor * points.z_centre_factor)[:, None]
    weight_xz = (EPS0 * pair(centre_ex) * shear_area)[window_x, window_y]
    weight_yz = (EPS0 * pair(centre_ey) * shear_area)[window_x, window_y]

    def photoelastic(change_xx, change_yy, change_zz, change_xy, change_xz=None, change_yz=None):
        """sum eps0 |E_i|^2 (face-averaged, bend-mapped Delta eps_ii) dA of window cell changes (ghost-filled), plus the
        centre shear terms eps0 Delta eps_xy 2 Re(E_x* E_y) dA and the xz / yz pairs with E_z (bend-mapped)."""
        total = complex(np.sum(change_xy * weight_xy))
        if change_xz is not None:
            total += complex(np.sum(change_xz * weight_xz))
        if change_yz is not None:
            total += complex(np.sum(change_yz * weight_yz))
        for change, weight, axis, periodic, widths in (
                (change_xx, weight_x, 0, periodic_x, points.dx_cells),
                (change_yy, weight_y, 1, periodic_y, points.dy_cells),
                (change_zz, weight_z, None, None, None)):
            full = np.zeros(shape, dtype=np.complex128)
            full[window_x, window_y] = change
            full = points.ghost_filled(full)
            if axis is not None:
                full = face_average(full, axis, periodic, widths)
            total += complex(np.sum(full * weight))
        return total

    power = float(mode.power)
    if power == 0.0:
        raise ValueError('the optical mode carries no power (evanescent): no perturbation formula')
    beta_scale = float(mode.omega) * float(mode.mirror_factor) / (4.0 * power)
    usable_group = n_group is not None and math.isfinite(n_group) and n_group != 0.0
    scale = -C0 / n_group * beta_scale if usable_group else math.nan
    cell_eps = tuple(values[window_x, window_y] for values in (points.cell_eps_xx, points.cell_eps_yy,
                                                                points.cell_eps_zz))
    return OpticalOverlap(face_x, face_y, cell_eps, photoelastic, scale, mode.omega, mode.mirror_factor,
                          beta_scale=beta_scale, n_eff=complex(mode.n_eff), n_group=n_group, backward=backward)


def _host(array):
    """numpy array of a numpy (as is) or cupy (one transfer) array: core.backend.to_host, the Backends rule."""
    from ...core.backend import to_host
    return np.asarray(to_host(array))


def _unsmoothed_cell_eps(optical, cell_eps):
    """Host full-grid ez cell eps without the subpixel-smoothing deltas (the build's staircase; sigma / poles keep
    their values); None when the physics does not smooth (its system eps is the staircase)."""
    smoothing = getattr(optical, 'subpixel_smoothing', None)
    if smoothing is None:
        return None
    from ...core.backend import array_module_of
    deltas = array_module_of(cell_eps).zeros_like(cell_eps)
    smoothing.add_deltas('cells', deltas, optical.domain.materials)
    return _host(cell_eps - deltas)


def _cross_section_mode(optical, number):
    """CrossSectionMode `number` (1-based) of a solved em_modes physics; ValueError when it was not found or Bloch k !=
    0 (the overlaps collocate without the wrap phase; definition_modes refuses it too)."""
    if optical.boundary_config.has_bloch_phase:
        raise ValueError(f"optomechanics needs the modes of '{optical.name}' at Bloch k = 0 (bloch_kx / bloch_ky 0)")
    modes = optical.modes
    if not 1 <= number <= len(modes):
        raise ValueError(f"optical mode {number} of '{optical.name}' does not exist: its mode solve found "
                         f'{len(modes)} mode(s) (raise n_modes or set n_eff_guess)')
    return modes[number - 1]


def backward_axial_wavenumber(optical, number):
    """k_z = 2 Re beta [rad/m] of cross-section mode `number` of a solved em_modes physics: the phonon wavenumber of
    backward Brillouin scattering into its counter-propagating twin; ValueError for another type or Re beta <= 0."""
    if not hasattr(optical, 'modes'):
        raise ValueError(f"backward Brillouin scattering needs an em_modes optical physics, '{optical.name}' is not one")
    beta = complex(_cross_section_mode(optical, number).beta).real
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError(f"optical mode {number} of '{optical.name}' has Re beta = {beta:g} 1/m: no guided wave to "
                         'phase-match a backward phonon')
    return 2.0 * beta


def optical_overlap(optical, number, window, backward=False):
    """OpticalOverlap of mode / resonance `number` (1-based) of a solved em_modes, em_fdfd_ez or em_fdfd_hz physics
    (duck-typed: `modes`, else `eigen_solve`) on the window slices (backward: em_modes only); em_fdfd_bor (polarization
    'bor'): optomechanics_fem.BOROpticalSamples (window unused); ValueError when it was not found. hz: the physics then
    shows resonance `number` unless another one was selected before (none to restore)."""
    if getattr(optical, 'polarization', None) == BOR_POLARIZATION:
        if backward:
            raise ValueError(f"backward Brillouin scattering needs an em_modes optical physics, '{optical.name}' is not "
                             'one')
        from .optomechanics_fem import bor_optical_samples
        return bor_optical_samples(optical, number)
    if hasattr(optical, 'modes'):
        mode = _cross_section_mode(optical, number)
        n_group = None if mode.n_group is None else complex(mode.n_group).real
        return cross_section_overlap(mode, optical.point_materials(), window, n_group, backward)
    if backward:
        raise ValueError(f"backward Brillouin scattering needs an em_modes optical physics, '{optical.name}' is not one")
    solve = getattr(optical, 'eigen_solve', None)
    if solve is None:
        raise ValueError(f"optical physics '{optical.name}' has no eigenfrequency solve")
    if not 1 <= number <= len(solve.resonances):
        raise ValueError(f"optical resonance {number} of '{optical.name}' does not exist: the eigen-solve found "
                         f'{len(solve.resonances)} (raise n_optical_resonances)')
    resonance = solve.resonances[number - 1]
    omega = complex(resonance.omega).real
    cell_widths = optical.domain.cell_widths()
    n_mirrors = sum(getattr(optical.boundary_config, face).value in _MIRROR_NAMES
                    for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'))
    system = optical.system
    if optical.polarization == 'ez':
        staircase = _unsmoothed_cell_eps(optical, system.cell_eps)
        return in_plane_ez_overlap(np.asarray(resonance.field)[window], _host(system.cell_eps)[window],
                                   cell_widths[0][window[0]], cell_widths[1][window[1]], omega, float(2 ** n_mirrors),
                                   None if staircase is None else staircase[window])
    # hz: E faces of the shown resonance (wall rows filled), from the curl at the assembly omega -> at the resonance
    shown = optical.selected_resonance
    if shown != number:
        optical.select_resonance(number)
    ex_faces, ey_faces = (_host(values) * (system.omega / omega) for values in optical.face_fields())
    field = _host(optical.solution)
    if shown is not None and shown != number:
        optical.select_resonance(shown)
    return in_plane_hz_overlap(field, ex_faces, ey_faces, _host(system.cell_eps_xx), _host(system.cell_eps_yy),
                               _host(system.eps_hi_y_faces), _host(system.eps_hi_x_faces), _host(system.cell_mu),
                               cell_widths, window, (system.periodic_x, system.periodic_y), omega,
                               float(2 ** n_mirrors))


# ---- g0 ----
def overlap_integrals(overlap, nodal, solid, p11, p12, dx, dy, p44=None, axial_wavenumber=None, crystal_regions=()):
    """(I_mb, I_pe) complex of the nodal shape (n_x + 1, n_y + 1, 2) [m] (guided: (u_x, u_y, w), 3 components at k_z =
    axial_wavenumber, backward overlaps only): face sum of the moving boundary and the photoelastic overlap of its strain
    on the solid cells (bool (n_x, n_y)) with p11 / p12 / p44 (n_x, n_y; p44 None or NaN cells = isotropic), the cells
    of each (window mask, (6, 6) Voigt p in model axes) crystal region with that full tensor."""
    if (axial_wavenumber is not None) != overlap.backward:
        raise ValueError('guided (k_z = 2 beta) vibration modes pair with the backward overlap only, plane-strain modes '
                         'with the forward one')
    q_x, q_y = face_displacements(nodal)
    moving = complex(np.sum(q_x * overlap.face_x) + np.sum(q_y * overlap.face_y))
    strain_xx, strain_yy, strain_xy = element_strains(nodal, dx, dy)
    strain_zz = engineering_yz = engineering_xz = 0.0   # plane strain
    if axial_wavenumber is not None:   # complex shears S_xz = i T_xz, S_yz = i T_yz
        strain_zz, tension_xz, tension_yz = guided_axial_strains(nodal, dx, dy, axial_wavenumber)
        engineering_xz, engineering_yz = 2.0j * tension_xz, 2.0j * tension_yz
    voigt_strains = [np.where(solid, values, 0.0) for values in
                     (strain_xx, strain_yy, strain_zz, engineering_yz, engineering_xz, 2.0 * strain_xy)]
    inverse = inverse_permittivity_changes(voigt_strains, p11, p12, shear_photoelastic_coefficient(p11, p12, p44),
                                           crystal_regions)
    change_xx, change_yy, change_zz, change_yz, change_xz, change_xy = permittivity_changes(*overlap.cell_eps, inverse)
    return moving, overlap.photoelastic(change_xx, change_yy, change_zz, change_xy, change_xz, change_yz)


def scaled_mode_shape(number, displacement, node_shape, reference_node=None, components=2):
    """Nodal (n_x + 1, n_y + 1, components) shape of mechanical mode `number` (flat host dofs, any scale) scaled to |Q|
    = 1 at reference_node (None = the peak node); ValueError when that node does not move."""
    nodal = np.asarray(displacement, dtype=np.float64).reshape(*node_shape, components)
    norms = np.sqrt(np.sum(nodal ** 2, axis=2))
    reference = float(np.max(norms)) if reference_node is None else float(norms[tuple(reference_node)])
    if not reference > 0.0:
        raise ValueError(f'mechanical mode {number} does not move the reference node: choose another reference point')
    return nodal / reference


def moving_boundary_density(overlap, scaled, dx, dy):
    """om_integrand: window (n_x, n_y) density [rad/s/m^3] of the moving-boundary g_om of the scaled nodal shape, each
    face's term split half / half onto its two cells over their area (sum x area = g_om_moving_boundary)."""
    q_x, q_y = face_displacements(scaled)
    term_x = (overlap.scale * q_x * overlap.face_x).real
    term_y = (overlap.scale * q_y * overlap.face_y).real
    density = np.zeros((term_x.shape[0] + 1, term_x.shape[1]))
    density[:-1, :] += 0.5 * term_x
    density[1:, :] += 0.5 * term_x
    density[:, :-1] += 0.5 * term_y
    density[:, 1:] += 0.5 * term_y
    return density / np.outer(dx, dy)


def mode_coupling_row(number, overlap, omega_mechanical, displacement, node_shape, mass, solid, p11, p12, dx, dy,
                      length, reference_node=None, moving_boundary=True, photoelastic=True, p44=None,
                      quality_factor=None, axial_wavenumber=None, crystal_regions=(), scaled=None):
    """OPTOMECHANICS_TABLE_COLUMNS row of one mechanical mode: displacement = flat host nodal dofs [m] (any scale; 3
    per node for a guided mode at k_z = axial_wavenumber, backward overlap), mass = host sparse consistent mass of the
    solve [kg/m], length [m]; reference_node None = the peak node; quality_factor = mechanical Q_m of the Brillouin gain
    (None, or an in-plane resonance: NaN); crystal_regions as overlap_integrals; scaled = the caller's
    scaled_mode_shape of this mode (None: computed here)."""
    components = 2 if axial_wavenumber is None else 3
    if scaled is None:
        scaled = scaled_mode_shape(number, displacement, node_shape, reference_node, components)
    moving, photo = overlap_integrals(overlap, scaled, solid, p11, p12, dx, dy, p44, axial_wavenumber, crystal_regions)
    moving = moving if moving_boundary else 0.0j
    photo = photo if photoelastic else 0.0j
    flat = scaled.reshape(-1)
    effective_mass = float(length) * overlap.mirror_factor * float(flat @ (mass @ flat))
    zero_point = math.sqrt(HBAR / (2.0 * effective_mass * float(omega_mechanical)))
    rate_moving, rate_photo = (overlap.scale * moving).real, (overlap.scale * photo).real
    rate = rate_moving + rate_photo
    gain = math.nan
    if overlap.beta_scale is None:
        index_rate = math.nan
    else:
        kappa = overlap.beta_scale * (moving + photo)   # d beta / d alpha, backward: the twin coupling [1/m^2]
        beta_rate = kappa.real
        index_rate = beta_rate * C0 / overlap.omega     # d n_eff / d alpha
        if quality_factor is not None:
            mass_per_length = effective_mass / float(length)
            strength = abs(kappa) ** 2 if overlap.backward else beta_rate ** 2
            gain = (2.0 * float(quality_factor) * strength
                    / (overlap.omega * mass_per_length * float(omega_mechanical) ** 2))
    to_hz = zero_point / (2.0 * math.pi)
    return [number, float(omega_mechanical) / (2.0 * math.pi), effective_mass, zero_point, rate_moving, rate_photo,
            rate, index_rate, rate_moving * to_hz, rate_photo * to_hz, rate * to_hz, gain]


class OptomechanicalCoupling:
    """Runtime of an optomechanical_coupling feature: its settings and the 'table' of the last evaluate (the
    optomechanics study calls it after the optical and the mechanical eigen-solve)."""

    def __init__(self, name, optical_physics, optical_mode, length, moving_boundary=True, photoelastic=True,
                 reference_point=None, quality_factor=None, scattering='forward', crystal_regions=(),
                 prescribed_normal_displacement=None, axis_point=(0.0, 0.0, 0.0)):
        if scattering not in SCATTERING:
            raise ValueError(f"scattering must be one of {', '.join(SCATTERING)}, got {scattering!r}")
        self.name = name
        # em_fdfd_bor optics (optomechanics_fem): q . n [m] of the solid's outward motion for the table's prescribed_g_om
        # (None: none), and the 3D point of the body-of-revolution axis for a tetrahedra solid
        self.prescribed_normal_displacement = (None if prescribed_normal_displacement is None
                                               else float(prescribed_normal_displacement))
        self.axis_point = tuple(float(value) for value in axis_point)
        # (host full-grid bool mask, (6, 6) Voigt p in model axes) of the crystal materials: their full photoelastic tensor
        self.crystal_regions = tuple(crystal_regions)
        self.backward = scattering == 'backward'   # guided vibration modes at k_z = 2 Re beta, the twin overlap
        self.quality_factor = quality_factor   # mechanical Q_m of the Brillouin gain column (None: NaN)
        self.optical_physics = optical_physics
        self.optical_mode = int(optical_mode)
        self.length = float(length)
        self.moving_boundary = bool(moving_boundary)
        self.photoelastic = bool(photoelastic)
        self.reference_point = reference_point   # (x, y) [m] or None (peak node)
        self.table = None
        self.integrands = None   # per mechanical mode: host window moving_boundary_density of the last evaluate

    def axial_wavenumber(self, optical):
        """k_z [rad/m] the vibration modes need: 2 Re beta of the optical mode (backward), None (forward: plane strain)."""
        return backward_axial_wavenumber(optical, self.optical_mode) if self.backward else None

    def evaluate(self, optical, solid):
        """Table (and om_integrand densities) of every mode of solid's last eigenfrequency solve against the optical
        mode of `optical` (backward: the solid's guided modes at k_z = 2 Re beta); em_fdfd_bor optics or a finite-element
        solid: optomechanics_fem.axisymmetric_coupling_table (no om_integrand)."""
        self.table = self.integrands = None
        data = solid.modal_overlap_data()
        if (getattr(optical, 'polarization', None) == BOR_POLARIZATION
                or data.get('kind', 'tensor') != 'tensor'):
            from .optomechanics_fem import axisymmetric_coupling_table
            self.table = axisymmetric_coupling_table(self, optical, solid, data)
            return self.table
        wavenumber = data['axial_wavenumber']
        expected = self.axial_wavenumber(optical)
        if (wavenumber is None) != (expected is None) or (
                expected is not None and not math.isclose(wavenumber, expected, rel_tol=1e-12)):
            raise ValueError(f"optomechanical_coupling '{self.name}': the vibration modes were solved at k_z = "
                             f'{wavenumber} rad/m, its {"backward" if self.backward else "forward"} scattering needs '
                             f'{expected} (None = plane strain)')
        overlap = optical_overlap(optical, self.optical_mode, data['window'], self.backward)
        reference = None
        if self.reference_point is not None:
            reference = nearest_solid_node(data['solid'], data['x_nodes'], data['y_nodes'], *self.reference_point)
        resonances = solid.eigen_solve.resonances
        components = 2 if wavenumber is None else 3
        crystal_regions = [(mask[data['window']], tensor) for mask, tensor in self.crystal_regions]
        shapes = [scaled_mode_shape(number, resonance.displacement, data['node_shape'], reference, components)
                  for number, resonance in enumerate(resonances, start=1)]   # once per mode: rows and integrands
        rows = [mode_coupling_row(number, overlap, resonance.omega, resonance.displacement, data['node_shape'],
                                  data['mass'], data['solid'], data['p11'], data['p12'], data['dx'], data['dy'],
                                  self.length, reference, self.moving_boundary, self.photoelastic, data['p44'],
                                  self.quality_factor, wavenumber, crystal_regions, scaled=shape)
                for number, (resonance, shape) in enumerate(zip(resonances, shapes), start=1)]
        self.integrands = [moving_boundary_density(overlap, shape, data['dx'], data['dy']) for shape in shapes]
        omega = overlap.omega
        self.table = {
            'kind': 'table', 'columns': list(OPTOMECHANICS_TABLE_COLUMNS), 'units': list(OPTOMECHANICS_TABLE_UNITS),
            'rows': rows, 'optical_physics': self.optical_physics, 'optical_mode': self.optical_mode,
            'optical_frequency': omega / (2.0 * math.pi), 'optical_wavelength': 2.0 * math.pi * C0 / omega,
            'n_eff': None if overlap.n_eff is None else overlap.n_eff.real, 'n_group': overlap.n_group,
            'length': self.length, 'mirror_factor': overlap.mirror_factor,
            'scattering': 'backward' if self.backward else 'forward', 'axial_wavenumber': wavenumber,
            'normalisation': 'max' if reference is None else 'point',
            'reference_node': None if reference is None else [int(index) for index in reference]}
        return self.table


def optomechanical_table_result(coupling) -> dict:
    """'table' of the last optomechanics evaluate; ValueError before one."""
    if coupling.table is None:
        raise ValueError('not solved yet: the optomechanical coupling table needs an optomechanics study')
    return coupling.table


def shown_integrand(solid):
    """Host float32 om_integrand of the mechanical mode the solid shows (its first evaluated coupling); ValueError
    outside an optomechanics study or before its mechanical solve."""
    coupling = next((coupling for coupling in solid.optomechanical_couplings if coupling.integrands is not None), None)
    number = solid.selected_resonance
    if coupling is None or number is None or not 1 <= number <= len(coupling.integrands):
        raise ValueError('om_integrand needs an optomechanics study whose mechanical mode solve is shown')
    return coupling.integrands[number - 1].astype(np.float32)
