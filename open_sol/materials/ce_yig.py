"""Library material CE_YIG: cerium-substituted yttrium iron garnet (Ce:YIG) films at 1.55 um, the magneto-optic garnet
of integrated optical isolators (index and loss; its gyration in the description)."""
from ._base import Material


CE_YIG = Material(
    name='ce_yig',
    # n = 2.22, film loss ~40 dB/cm: sigma_e = 2 n kappa omega eps0 with kappa = alpha lambda / (4 pi)
    optical=dict(eps_r=4.93, sigma_e=5.4),
    thermal=dict(k=6.0, rho=5200.0, cp_=570.0),
    description='Polycrystalline Ce:YIG (Ce1Y2Fe5O12) film at 1.55 um and room temperature, as sputtered or pulsed-'
                'laser deposited on silicon or silicon nitride and saturated by a small in-plane field, non-'
                'dispersive. Magneto-optic data are not material properties (the magnetisation direction is the '
                "model's): set the gyration g on the material, |g| ~ 0.009 from its Faraday rotation theta_F ~ -4500 "
                'deg/cm (g = theta_F lambda n / pi in magnitude; films range ~ -2000 to -6000 deg/cm, and g changes '
                'sign with the magnetisation): gyration_z (em_fdfd_hz, magnetisation out of the plane) or '
                'gyration_x / gyration_y (em_modes, transverse magnetisation, the non-reciprocal phase shift). The '
                'loss (~40 dB/cm, sigma_e) varies a lot between films (~20-60 dB/cm). Values are typical figures '
                'from the integrated-isolator literature (Bi et al., Nature Photonics 5, 758 (2011); Zhang et al., '
                'Optica 6, 473 (2019)), unverified: check against the film at hand. Thermal data rough (YIG bulk).',
    summary='Cerium-doped yttrium iron garnet film at 1550 nm, the magneto-optic material of on-chip optical '
            'isolators: refractive index 2.22 and some loss. Set its magneto-optic strength (gyration) yourself, '
            'about 0.009, along the magnetisation direction of your device.',
)
