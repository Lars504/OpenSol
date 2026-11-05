"""Library material ALUMINUM: a near-IR Drude pole (fit at 1.55 um), thermal, elastic and DC-conductivity data, and its
Rakic 1995 n, k table."""
from ._base import Material
from .nk_data import ALUMINUM_RAKIC_1995
from ..physics.em.materials import DrudePole


ALUMINUM = Material(
    name='aluminum',
    em=DrudePole(omega_p=1.94e16, gamma=2.46e14, eps_inf=1.0),
    thermal=dict(k=237.0, rho=2700.0, cp_=897.0, dndT=0.0, alpha=2.31e-5),
    mechanical=dict(youngs_modulus=7.0e10, poisson_ratio=0.35),
    electrical=dict(sigma_dc=3.8e7),
    description='Aluminium (library name aluminum) for EM, heat and electric currents (room temperature). The '
                'effective Drude pole is fitted to Rakic 1998 at 1.55 um: eps -244 - 49.6j (Rakic about -243 - 49j). '
                'It folds the 1.5 eV (0.83 um) interband band into the pole, so it is a near-IR fit only: in the '
                'visible / UV it is wrong (a broadband free-electron pole, Ordal omega_p 2.24e16, gamma 1.22e14, '
                'gives -335 - 34j at 1.55 um instead). For other bands use an nk_fit dispersion entry on its table '
                '(Rakic 1995 evaporated films, 207-1771 nm; a Lorentz pole takes the 800 nm interband band).',
    summary='Aluminium for light, heat and electric currents.\n\n'
            'Its default optical data fit wavelengths near 1550 nm only and are wrong in the visible and '
            'ultraviolet; there add a tabulated-data fit pole (measured data, 207 to 1771 nm).',
    nk_table=ALUMINUM_RAKIC_1995,
)
