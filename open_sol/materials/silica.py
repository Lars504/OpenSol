"""Library material SILICA: fused silica at 1.55 um (n = 1.444, lossless, non-dispersive) with thermal, elastic,
photoelastic and DC data."""
from ._base import Material


SILICA = Material(
    name='silica',
    optical=dict(eps_r=2.085),   # n = 1.444 at 1.55 um
    thermal=dict(k=1.38, rho=2200.0, cp_=746.0, dndT=1.0e-5, alpha=5.5e-7),
    mechanical=dict(youngs_modulus=7.3e10, poisson_ratio=0.17, photoelastic_p11=0.121, photoelastic_p12=0.270),
    electrical=dict(eps_dc=3.9),
    description='Fused silica (SiO2 glass) at 1.55 um and room temperature (n = 1.444), lossless and '
                'non-dispersive, so every wavelength sees n = 1.444 (about 1.46 in the visible). An insulator (no '
                'sigma_dc). Elastic data of bulk fused silica (deposited oxide films are often softer, 60-70 GPa).',
    summary='Fused silica glass at 1550 nm: refractive index 1.444 at every wavelength, no loss, an electrical '
            'insulator.',
)
