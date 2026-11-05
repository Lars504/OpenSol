"""Library materials LITHIUM_NIOBATE_Z_CUT / _X_CUT: uniaxial lithium niobate films at 1.55 um (n_o / n_e along the
optic axis), thermal, elastic and DC data, Pockels r on the x cut."""
from ._base import Material


LITHIUM_NIOBATE_Z_CUT = Material(
    name='lithium_niobate_z_cut',
    # n_o = 2.211 in plane (E_x, E_y), n_e = 2.138 along the optic axis z (E_z)
    optical=dict(eps_r=4.889, eps_zz=4.571),
    # alpha: a-axis (in plane), alpha_zz: c-axis along the optic axis z (solid_mechanics only)
    thermal=dict(k=4.6, rho=4640.0, cp_=628.0, dndT_xx=3.0e-6, dndT_yy=3.0e-6, dndT_zz=3.3e-5, alpha=1.5e-5,
                 alpha_zz=7.5e-6),
    mechanical=dict(youngs_modulus=1.7e11, poisson_ratio=0.25),   # rough isotropic estimate of the trigonal crystal
    # clamped eps_a^S (in plane for a z-cut film; unclamped eps_a^T ~85); eps_c ~28 along the optic axis
    electrical=dict(eps_dc=43.0),
    description='Z-cut lithium niobate film at 1.55 um and room temperature, optic axis z (out of plane): eps_zz = '
                'n_e^2 (n_e = 2.138) is seen by E_z (em_fdtd_ez, em_fdfd_ez), and the isotropic xy plane has eps_r = '
                'n_o^2 (n_o = 2.211), seen by E_x / E_y (em_fdtd_hz, em_fdfd_hz). An eps_r override therefore does '
                'not change E_z: set eps_zz. Lossless and non-dispersive. dndT_zz is dn_e/dT, dndT_xx = dndT_yy '
                'dn_o/dT (approximate). For an x-cut film (optic axis along x) use lithium_niobate_x_cut. k and '
                'alpha are a-axis values, alpha_zz the c-axis expansion (solid_mechanics only); '
                'alpha is the free crystal (a film bonded to Si follows the substrate: set alpha and alpha_zz to '
                '2.6e-6; an alpha override alone keeps the library alpha_zz). eps_dc = '
                'eps_a, the in-plane z-cut permittivity, clamped (eps^S, valid above the acoustic resonances, as '
                'for RF / electro-optic electrodes; the unclamped DC eps_a of a free crystal is ~85: override eps_dc '
                'for that case). No Pockels coefficients: an in-plane applied field of a z-cut film acts through '
                'r22 with opposite signs on n_xx and n_yy, which the diagonal r_xx / r_yy / r_zz of '
                'electrostatics_em_pockels cannot express.',
    summary='Z-cut lithium niobate film at 1550 nm: refractive index 2.138 for the electric field along z (out of '
            'the plane), 2.211 in the plane.\n\n'
            "To change the out-of-plane value, edit 'Relative permittivity, z (out of plane)', not 'Relative "
            "permittivity'.",
)

LITHIUM_NIOBATE_X_CUT = Material(
    name='lithium_niobate_x_cut',
    # optic axis (crystal z) along model x: n_e = 2.138 for E_x, n_o = 2.211 for E_y and E_z
    optical=dict(eps_r=4.889, eps_xx=4.571),
    # alpha: a-axis (y, z), alpha_xx: c-axis along the optic axis x
    thermal=dict(k=4.6, rho=4640.0, cp_=628.0, dndT_xx=3.3e-5, dndT_yy=3.0e-6, dndT_zz=3.0e-6, alpha=1.5e-5,
                 alpha_xx=7.5e-6),
    mechanical=dict(youngs_modulus=1.7e11, poisson_ratio=0.25),   # rough isotropic estimate of the trigonal crystal
    # clamped eps_c along the optic axis, eps_a across it; clamped Pockels r33^S on the optic axis, r13^S across it
    # [m/V] (the crystal_tensors values, consistent with the clamped eps; unclamped r^T: r13 10, r33 32.2 pm/V)
    electrical=dict(eps_dc=43.0, eps_dc_xx=28.0, r=8.6e-12, r_xx=30.8e-12),
    description='X-cut lithium niobate film at 1.55 um and room temperature in top view, optic axis (crystal z) along '
                'model x: eps_xx = n_e^2 is seen by E_x (em_fdtd_hz, em_fdfd_hz, em_modes), eps_r = n_o^2 by E_y '
                'and E_z. Lossless and non-dispersive. dndT_xx is dn_e/dT, dndT_yy = dndT_zz dn_o/dT '
                '(approximate); k and alpha are a-axis values, alpha_xx the c-axis expansion along x, both the free '
                'crystal (a film bonded to Si follows the substrate: set alpha and alpha_xx; an alpha override alone '
                'keeps the library alpha_xx). eps_dc_xx = eps_c and eps_dc = eps_a are clamped values (eps^S, as for RF / '
                'electro-optic electrodes). Pockels coefficients for an applied field along the optic axis '
                "(electrostatics_em_pockels with field_component 'x'): r_xx = r33 (n_e, E_x of the H_z types), r = "
                'r13 on r_yy and r_zz (n_o), both clamped (r^S, consistent with the clamped eps_dc; the unclamped '
                'r13 / r33 of a free crystal at low frequency are 10 / 32.2 pm/V). An applied field along y acts '
                'through r22 / r51 terms, not modelled '
                "(keep field_component 'x').",
    summary='X-cut lithium niobate film at 1550 nm, optic axis along x: refractive index 2.138 for the electric field '
            'along x, 2.211 along y and z.\n\n'
            'Carries the electro-optic (Pockels) coefficients for a field applied along x, as between electrodes on '
            'either side of a waveguide running along y.',
)
