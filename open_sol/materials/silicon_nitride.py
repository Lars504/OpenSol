"""Library material SILICON_NITRIDE: stoichiometric LPCVD silicon nitride (Si3N4) film at 1.55 um (n = 1.996, lossless,
non-dispersive) with thermal, elastic and DC data."""
from ._base import Material


SILICON_NITRIDE = Material(
    name='silicon_nitride',
    optical=dict(eps_r=3.984),   # n = 1.996 at 1.55 um
    thermal=dict(k=10.0, rho=3100.0, cp_=700.0, dndT=2.45e-5, alpha=2.3e-6),
    mechanical=dict(youngs_modulus=2.5e11, poisson_ratio=0.23),
    electrical=dict(eps_dc=7.5),
    description='Stoichiometric LPCVD silicon nitride (Si3N4), n = 1.996 lossless (about 2.02 in the visible). Film '
                "data vary with the deposition: Young's modulus 200-300 GPa, density 2900-3200 kg/m^3, alpha "
                '1.5-3.3e-6 1/K and above all k (thin films ~2-15 W/(m K), bulk ceramic 15-30): set k and alpha for '
                'your film when e.g. thermoelastic damping matters. The ~1 GPa tensile deposition stress is not '
                'included: set initial_stress_* for stressed membranes and strings.',
    summary='Silicon nitride film at 1550 nm: refractive index 1.996 at every wavelength, no loss, an electrical '
            'insulator.\n\nThermal and elastic data are typical film values: check the conductivity and the '
            'built-in stress of your film.',
)
