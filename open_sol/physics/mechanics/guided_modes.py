"""2.5D guided vibration modes of a cross-section (x, y) of a structure long along z, u ~ exp(i (k_z z - omega t)): the
three-component Q1 stiffness / mass of the eigenfrequency study (host numpy / scipy) and the k_z = 0 antiplane support
check. modal.solve_modes solves the pencil."""
# Unknowns per node (u_x, u_y, w), the physical displacement u = Re[(u_x, u_y, -i w) exp(i phi)] = (u_x cos phi, u_y cos
# phi, w sin phi), phi = k_z z - omega t. The diagonal unitary u_z = -i w turns the complex Hermitian operator into a
# real symmetric one, exactly for isotropic cells (no normal-shear coupling in C, so only even powers of i remain):
# strains S_xx = u_x,x, S_yy = u_y,y, S_zz = k w, gamma_xy = u_x,y + u_y,x, gamma_xz = -i (w,x - k u_x), gamma_yz = -i
# (w,y - k u_y); energy density c11 (S_xx^2 + S_yy^2 + S_zz^2) + 2 lambda (S_xx S_yy + S_xx S_zz + S_yy S_zz) + mu
# (gamma_xy^2 + |gamma_xz|^2 + |gamma_yz|^2), c11 = lambda + 2 mu. Element integrals are exact for bilinear N on a x b
# rectangles (as system.stiffness_stencil): int N_k,x N_l,x = s_kx s_lx (1 + delta_y) b / (6 a), int N_k,x N_l,y = s_kx
# s_ly / 4, int N_k N_l,x = s_lx (1 + delta_y) b / 12, int N_k N_l = (1 + delta_x)(1 + delta_y) a b / 36. k_z = 0 splits
# into the plane-strain block (= stiffness_stencil) and the antiplane block mu (w,x^2 + w,y^2); k_z > 0 leaves no
# zero-energy motion (S_zz = k w and gamma_xz, gamma_yz vanish only for u = 0), so a free body is positive definite.
import numpy as np

from .system import LOCAL_NODES, ElasticPattern, mass_stencil

GUIDED_COMPONENTS = 3
_SLOPE_SIGN = (-1.0, 1.0)   # derivative sign of the lower / upper 1D linear shape function (as system)


def guided_stiffness_stencil(lame_lambda, mu, dx_elements, dy_elements, axial_wavenumber, xp=np):
    """(nx_nodes, ny_nodes, 3, 3, 3, 3) float64 stencil [N/m per m of depth] of the real symmetric 2.5D Q1 stiffness of
    window element Lame coefficients (n_x, n_y) [Pa] on widths dx / dy [m] at k_z = axial_wavenumber [rad/m]."""
    n_x, n_y = mu.shape
    stencil = xp.zeros((n_x + 1, n_y + 1, GUIDED_COMPONENTS, 3, 3, GUIDED_COMPONENTS), dtype=xp.float64)
    wavenumber = float(axial_wavenumber)
    longitudinal = lame_lambda + 2.0 * mu
    width_x, width_y = dx_elements[:, None], dy_elements[None, :]
    ratio_x, ratio_y = width_y / (6.0 * width_x), width_x / (6.0 * width_y)   # b / (6 a), a / (6 b)
    slope_x, slope_y = width_y / 12.0, width_x / 12.0   # int N_k N_l,x = s_lx (1 + delta_y) b / 12
    element_mass = width_x * width_y / 36.0
    axial_squared = wavenumber ** 2
    for k_x, k_y in LOCAL_NODES:
        target = stencil[k_x:k_x + n_x, k_y:k_y + n_y]
        for l_x, l_y in LOCAL_NODES:
            s_kx, s_ky, s_lx, s_ly = (_SLOPE_SIGN[index] for index in (k_x, k_y, l_x, l_y))
            mass_x = 2.0 if k_x == l_x else 1.0   # 1D mass factor (1 + delta)
            mass_y = 2.0 if k_y == l_y else 1.0
            di, dj = l_x - k_x + 1, l_y - k_y + 1
            gradient_xx = s_kx * s_lx * mass_y * ratio_x   # int N_k,x N_l,x
            gradient_yy = s_ky * s_ly * mass_x * ratio_y
            gradient_xy = 0.25 * s_kx * s_ly                # int N_k,x N_l,y
            gradient_yx = 0.25 * s_ky * s_lx
            overlap = mass_x * mass_y * element_mass        # int N_k N_l
            trial_x = s_lx * mass_y * slope_x               # int N_k N_l,x (test k, trial l)
            test_x = s_kx * mass_y * slope_x                # int N_k,x N_l
            trial_y = s_ly * mass_x * slope_y
            test_y = s_ky * mass_x * slope_y
            target[:, :, 0, di, dj, 0] += longitudinal * gradient_xx + mu * (gradient_yy + axial_squared * overlap)
            target[:, :, 1, di, dj, 1] += longitudinal * gradient_yy + mu * (gradient_xx + axial_squared * overlap)
            target[:, :, 2, di, dj, 2] += mu * (gradient_xx + gradient_yy) + longitudinal * axial_squared * overlap
            target[:, :, 0, di, dj, 1] += lame_lambda * gradient_xy + mu * gradient_yx
            target[:, :, 1, di, dj, 0] += lame_lambda * gradient_yx + mu * gradient_xy
            # lambda S_zz S_xx and the mu gamma_xz cross term -2 mu k u_x w,x (the y pair alike)
            target[:, :, 0, di, dj, 2] += wavenumber * (lame_lambda * test_x - mu * trial_x)
            target[:, :, 2, di, dj, 0] += wavenumber * (lame_lambda * trial_x - mu * test_x)
            target[:, :, 1, di, dj, 2] += wavenumber * (lame_lambda * test_y - mu * trial_y)
            target[:, :, 2, di, dj, 1] += wavenumber * (lame_lambda * trial_y - mu * test_y)
    return stencil


def guided_operators(lame_lambda, mu, density, dx_elements, dy_elements, axial_wavenumber):
    """(K, M) host scipy CSR [N/m, kg/m per m of depth] over every dof (dof = 3 node + component, components (u_x, u_y,
    w)) of window element host float64 Lame coefficients, density [kg/m^3] and widths [m] at k_z [rad/m]."""
    import scipy.sparse
    n_x, n_y = mu.shape
    pattern = ElasticPattern(n_x + 1, n_y + 1, components=GUIDED_COMPONENTS, xp=np)
    dx_elements = np.asarray(dx_elements, dtype=np.float64)
    dy_elements = np.asarray(dy_elements, dtype=np.float64)
    stiffness = pattern.csr(guided_stiffness_stencil(lame_lambda, mu, dx_elements, dy_elements, axial_wavenumber),
                            scipy.sparse)
    mass = pattern.csr(mass_stencil(density, dx_elements, dy_elements, np.float64, xp=np,
                                    components=GUIDED_COMPONENTS), scipy.sparse)
    return stiffness, mass


def guided_free_dofs(stiffness, held_in_plane, held_axial):
    """Host bool flat free dofs: positive stiffness diagonal (the node touches a solid element) and not held; held_in_plane
    (nx_nodes, ny_nodes, 2) and held_axial (nx_nodes, ny_nodes) host bool."""
    held = np.concatenate((np.asarray(held_in_plane, dtype=bool), np.asarray(held_axial, dtype=bool)[..., None]),
                          axis=-1).reshape(-1)
    return (stiffness.diagonal() > 0.0) & ~held


def check_antiplane_supports(solid, held_axial, label):
    """k_z = 0: ValueError when a solid region (8-connected window cells, host bool) holds no u_z node (held_axial host
    bool nodes), since its antiplane block then has the rigid translation along z."""
    import scipy.ndimage
    labels, n_regions = scipy.ndimage.label(np.asarray(solid, dtype=bool), structure=np.ones((3, 3), dtype=bool))
    if not n_regions:
        return
    padded = np.zeros((labels.shape[0] + 2, labels.shape[1] + 2), dtype=labels.dtype)
    padded[1:-1, 1:-1] = labels
    node_labels = np.maximum(np.maximum(padded[:-1, :-1], padded[1:, :-1]), np.maximum(padded[:-1, 1:], padded[1:, 1:]))
    held_regions = set(np.unique(node_labels[np.asarray(held_axial, dtype=bool) & (node_labels > 0)]).tolist())
    missing = [region for region in range(1, int(n_regions) + 1) if region not in held_regions]
    if missing:
        raise ValueError(f'{label()}: at axial_wavenumber 0, {len(missing)} of {int(n_regions)} solid region(s) hold no '
                         'u_z, so their antiplane (u_z) block has a rigid translation along z and the solve is '
                         'singular: hold u_z on every region with a Fixed or Fixed (domain) support (Roller and '
                         'Displacement leave u_z free), or set axial_wavenumber > 0')
