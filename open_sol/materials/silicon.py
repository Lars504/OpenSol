"""Library material SILICON: crystalline silicon at 1.55 um (n = 3.476) with thermal, elastic, photoelastic and
drift-diffusion data and its Green 2008 n, k table."""
from ._base import Material
from .nk_data import SILICON_GREEN_2008


SILICON = Material(
    name='silicon',
    optical=dict(eps_r=12.08),   # n = 3.476 at 1.55 um
    thermal=dict(k=130.0, rho=2329.0, cp_=700.0, dndT=1.8e-4, alpha=2.6e-6),
    # elastic: isotropic (Hill) average of the cubic crystal; photoelastic p11 / p12 / p44 near 1.55 um, cube axes along
    # x, y, z
    mechanical=dict(youngs_modulus=1.65e11, poisson_ratio=0.22, photoelastic_p11=-0.094, photoelastic_p12=0.017,
                    photoelastic_p44=-0.051),
    # drift_diffusion: n_i at 300 K, low-doping mobilities, radiative B (lifetimes are device-specific: unset)
    electrical=dict(eps_dc=11.7, n_intrinsic=1.0e16, mobility_n=0.135, mobility_p=0.048,
                    radiative_recombination=1.1e-20),
    description='Crystalline silicon at 1.55 um and room temperature (n = 3.476), lossless and non-dispersive, so '
                'every wavelength sees n = 3.476 (real silicon absorbs below about 1.1 um). An nk_fit dispersion '
                'entry gives dispersive, absorbing silicon instead (table: Green 2008, intrinsic, 300 K, 250-1450 '
                'nm, band-to-band absorption only). Intrinsic: no sigma_dc (doped silicon sets its own, e.g. 1e4 '
                'S/m). Thermal conductivity: the common engineering value, below the ~148 W/(m K) of pure bulk '
                'silicon at 300 K (doped silicon, thin device layers and higher temperatures run lower). Elastic '
                'data: an isotropic average of the cubic crystal (E is 130 GPa along <100>, 169 GPa '
                'along <110>). Drift-diffusion data: 300 K, mobilities of lightly doped silicon (no doping or field '
                'dependence).',
    summary='Crystalline silicon at 1550 nm: refractive index 3.476 at every wavelength and no loss; real silicon '
            'absorbs below about 1100 nm.\n\n'
            'For absorbing, dispersive silicon (250 to 1450 nm) add a tabulated-data fit pole. Undoped: set an '
            'electrical conductivity for doped silicon.',
    nk_table=SILICON_GREEN_2008,
)
