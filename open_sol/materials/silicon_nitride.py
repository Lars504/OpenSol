"""Library material SILICON_NITRIDE: stoichiometric LPCVD silicon nitride (Si3N4) film at 1.55 um (n = 1.996, lossless,
non-dispersive) with thermal, elastic and DC data."""
from ._base import Material


SILICON_NITRIDE = Material(
    name='silicon_nitride',
    optical=dict(eps_r=3.984),   # n = 1.996 at 1.55 um
    thermal=dict(k=10.0, rho=3100.0, cp_=700.0, dndT=2.45e-5, alpha=2.3e-6),
    mechanical=dict(youngs_modulus=2.5e11, poisson_ratio=0.23),
    electrical=dict(eps_dc=7.5),
    description='Stoichiometric silicon nitride (Si3N4) as deposited by LPCVD for integrated photonics and '
                'nano-mechanical resonators, at 1.55 um and room temperature: lossless and non-dispersive (every '
                'wavelength sees n = 1.996; about 2.02 in the visible). Film data vary with the deposition: Young\'s '
                'modulus 200-300 GPa, density 2900-3200 kg/m^3, thermal expansion 1.5-3.3e-6 1/K (bulk ceramic the '
                'upper end), and above all the thermal conductivity (thin films measure about 2-15 W/(m K), bulk '
                'ceramic 15-30): set k (and alpha) for your film when thermal effects such as thermoelastic damping '
                'matter. The large tensile deposition stress of stoichiometric films (about 1 GPa) is not included: '
                'set initial_stress_* for stressed membranes and strings. An insulator (no sigma_dc).',
    summary='Silicon nitride film at 1550 nm: refractive index 1.996 at every wavelength, no loss, an electrical '
            'insulator.\n\nThermal and elastic data are typical film values: check the conductivity and the '
            'built-in stress of your film.',
)
