"""Library material SILVER: a free-electron Drude pole, thermal, elastic and DC-conductivity data and its Johnson &
Christy n, k table."""
from ._base import Material
from .nk_data import SILVER_JOHNSON_CHRISTY
from ..physics.em.materials import DrudePole


SILVER = Material(
    name='silver',
    em=DrudePole(omega_p=1.37e16, gamma=2.73e13, eps_inf=3.7),
    thermal=dict(k=429.0, rho=10490.0, cp_=235.0, dndT=0.0, alpha=1.89e-5),
    mechanical=dict(youngs_modulus=8.3e10, poisson_ratio=0.37),
    electrical=dict(sigma_dc=6.3e7),
    description='Silver for EM, heat and electric currents (room temperature): one Drude pole without interband '
                'terms, a common free-electron fit whose eps_inf absorbs the interband background (its original '
                'source is not recorded). At 1.55 um it gives eps -123 - 2.9j against Johnson & Christy -129 - 3.3j '
                '(real part 5 %, loss 13 % low); visible / near-IR only, wrong below the interband edge near 320 nm. '
                'Tabulated n, k (Johnson & Christy 1972, 188-1937 nm) for an nk_fit dispersion entry, which fits '
                'Drude + Lorentz poles over a chosen band.',
    summary='Silver for light, heat and electric currents, with a simple free-electron optical model.\n\n'
            'Measured optical data (188 to 1937 nm) can replace it: add a tabulated-data fit pole.',
    nk_table=SILVER_JOHNSON_CHRISTY,
)
