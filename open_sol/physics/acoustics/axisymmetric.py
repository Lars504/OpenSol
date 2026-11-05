"""Host (numpy) radial geometry of the axisymmetric (r, z) velocity-stress scheme: the radius rows the fused kernels
read, the ring-volume face density shares and the radial stability factor (no cupy: definition.py reads the factor)."""
import numpy as np

KERNEL_ROWS = 7   # rows of `kernel_rows` (radial[k * nx + i] in kernels.py)


def _radii(widths):
    """(widths, following widths, face radii R_i of the hi faces, centre radii c_i, following centre radii c_i+1)
    float64; the axis is the lo edge of cell 0, the last cell is followed by its mirror (the pinned outer wall)."""
    widths = np.asarray(widths, dtype=np.float64)
    faces = np.cumsum(widths)
    following_widths = np.append(widths[1:], widths[-1])
    return widths, following_widths, faces, faces - 0.5 * widths, faces + 0.5 * following_widths


def kernel_rows(widths):
    """(KERNEL_ROWS, n) float64: c_i / R_i and c_i+1 / R_i (the (1/r) d(r s_rr)/dr weights at the v_r face), width share
    lo / R_i and hi / R_i (hoop force -(s_tt) / r at that face), R_i-1 / c_i and R_i / c_i ((1/r) d(r s_rz)/dr at the
    v_z face) and 0.5 / c_i (hoop strain rate (v_lo + v_hi) / (2 c_i))."""
    widths, following_widths, faces, centres, following_centres = _radii(widths)
    pair = widths + following_widths
    lower_faces = np.concatenate(([0.0], faces[:-1]))
    return np.stack((centres / faces, following_centres / faces, widths / pair / faces, following_widths / pair / faces,
                     lower_faces / centres, faces / centres, 0.5 / centres))


def density_shares(widths):
    """(lo, hi) shares of the x-face density: the ring volumes pi (R^2 - c_i^2) and pi (c_i+1^2 - R^2) of the two
    half cells over their sum (the exact mass of the dual ring)."""
    _, _, faces, centres, following_centres = _radii(widths)
    total = following_centres ** 2 - centres ** 2
    return (faces ** 2 - centres ** 2) / total, (following_centres ** 2 - faces ** 2) / total


def _row_bound(diagonal, upper):
    """Largest Gershgorin row sum of the symmetric tridiagonal matrix (diagonal, upper off-diagonal)."""
    rows = np.abs(diagonal)
    rows[:-1] += np.abs(upper)
    rows[1:] += np.abs(upper)
    return float(rows.max())


def _normal_bound(p, q, cell_weight, face_mass):
    """Gershgorin bound of W^1/2 A M^-1 A^T W^1/2 for a cell strain A v = p_i v_i + q_i v_i-1 of the x-face velocities
    (the axis face and the pinned outer face carry none)."""
    inverse_mass = 1.0 / face_mass
    inverse_mass[-1] = 0.0
    lower_inverse = np.concatenate(([0.0], inverse_mass[:-1]))
    diagonal = cell_weight * (p ** 2 * inverse_mass + q ** 2 * lower_inverse)
    upper = np.sqrt(cell_weight[:-1] * cell_weight[1:]) * p[:-1] * q[1:] * inverse_mass[:-1]
    return _row_bound(diagonal, upper)


def _shear_bound(dual_widths, corner_weight, cell_mass):
    """Gershgorin bound for the corner strain (v_i+1 - v_i) / h_i of the cell-centred (y-face) velocities; the outer
    corner is pinned."""
    corner_weight = corner_weight.copy()
    corner_weight[-1] = 0.0
    inverse_mass = 1.0 / cell_mass
    following_inverse = np.append(inverse_mass[1:], 0.0)
    diagonal = corner_weight / dual_widths ** 2 * (inverse_mass + following_inverse)
    upper = np.sqrt(corner_weight[:-1] * corner_weight[1:]) / (dual_widths[:-1] * dual_widths[1:]) * inverse_mass[1:]
    return _row_bound(diagonal, upper)


def radial_stability_factor(widths):
    """F >= 1 scaling 1 / dx_min^2 in the time-step bound: the largest ratio of the axisymmetric to the planar
    Gershgorin bound of the radial operators (normal strains e_rr +- e_tt, shear e_rz); ~1.047 on uniform cells."""
    widths, following_widths, faces, centres, _ = _radii(widths)
    dual_widths = 0.5 * (widths + following_widths)
    planar_normal = _normal_bound(1.0 / widths, -1.0 / widths, widths, dual_widths)
    hoop = 0.5 / centres
    ring_normal = max(_normal_bound(1.0 / widths + sign * hoop, -1.0 / widths + sign * hoop, centres * widths,
                                    faces * dual_widths) for sign in (1.0, -1.0))
    planar_shear = _shear_bound(dual_widths, dual_widths, widths)
    ring_shear = _shear_bound(dual_widths, faces * dual_widths, centres * widths)
    return max(1.0, ring_normal / planar_normal, ring_shear / planar_shear)
