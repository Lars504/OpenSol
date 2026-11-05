"""Library material GOLD: a Drude pole fitted to Johnson & Christy (about 0.6-2 um), thermal, elastic and
DC-conductivity data and its n, k table."""
from ._base import Material
from .nk_data import GOLD_JOHNSON_CHRISTY
from ..physics.em.materials import DrudePole


GOLD = Material(
    name='gold',
    em=DrudePole(omega_p=1.37e16, gamma=1.02e14, eps_inf=9.84),
    thermal=dict(k=317.0, rho=19300.0, cp_=129.0, dndT=0.0, alpha=1.42e-5),
    mechanical=dict(youngs_modulus=7.9e10, poisson_ratio=0.42),
    electrical=dict(sigma_dc=4.1e7),
    description='Gold for EM, heat and electric currents (room temperature). The Drude pole is a fit to Johnson & '
                'Christy 1972 valid about 0.6-2 um: eps about -116 - 10.6j at 1.55 um (Johnson & Christy about '
                '-114 - 11j; evaporated films in Olmon 2012 have |Im eps| about 7, single crystals about 4). The '
                'far-IR / DC damping (about 4e13 rad/s) would underestimate the near-IR loss 2-3x. No interband '
                'absorption, so least accurate below about 550 nm (|Im eps| up to ~5x low). Resolve the field decay '
                'into the metal (about 25 nm) with several cells. Its table (Johnson & Christy 1972, 188-1937 nm) '
                'serves an nk_fit dispersion entry, whose Lorentz poles add the interband absorption.',
    summary='Gold for light, heat and electric currents; the optical data fit about 600 to 2000 nm.\n\n'
            'Fields decay within about 25 nm inside it: mesh that depth with several cells. For shorter '
            'wavelengths add a tabulated-data fit pole (measured data, 188 to 1937 nm).',
    nk_table=GOLD_JOHNSON_CHRISTY,
)
