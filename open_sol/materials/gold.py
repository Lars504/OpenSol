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
    description='Gold for EM, heat and electric currents (room temperature). The Drude pole fits Johnson & Christy '
                '1972 over about 0.6-2 um (eps ~ -116 - 10.6j at 1.55 um; evaporated films have |Im eps| ~7, single '
                'crystals ~4); no interband absorption, so below ~550 nm |Im eps| is up to ~5x low. Resolve the '
                '~25 nm field decay into the metal with several cells. Its table (Johnson & Christy 1972, 188-1937 '
                'nm) serves an nk_fit entry, whose Lorentz poles add the interband absorption.',
    summary='Gold for light, heat and electric currents; the optical data fit about 600 to 2000 nm.\n\n'
            'Fields decay within about 25 nm inside it: mesh that depth with several cells. For shorter '
            'wavelengths add a tabulated-data fit pole (measured data, 188 to 1937 nm).',
    nk_table=GOLD_JOHNSON_CHRISTY,
)
