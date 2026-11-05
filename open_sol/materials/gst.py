"""Library materials GST_225_AMORPHOUS / _CRYSTALLINE: Ge2Sb2Te5 phase-change alloy, eps_r and sigma_e at 1.55 um from
physics.em.phase_change.GST_225, plus thermal data."""
import math

from ._base import Material
from ..core.constants import C0, EPS0
from ..physics.em.phase_change import GST_225, GST_225_REFERENCES


def _optical(phase):
    """eps_r = n^2 - k^2 and sigma_e = 2 n k omega eps0 of one GST-225 phase at its 1.55 um reference wavelength."""
    n, k = GST_225[f'n_{phase}'], GST_225[f'k_{phase}']
    omega = 2.0 * math.pi * C0 / GST_225['wavelength']
    return dict(eps_r=round(n * n - k * k, 4), sigma_e=round(2.0 * n * k * omega * EPS0, 1))


_THERMAL_TEXT = ('Thermal data: room-temperature conductivity of the phase (Lyeo et al., Appl. Phys. Lett. 89, 151904 '
                 '(2006)), density (Njoroge et al., J. Vac. Sci. Technol. A 20, 230 (2002)), a typical specific heat; '
                 'no latent heat, no temperature dependence.')
_OPTICAL_TEXT = ('Optical data at 1.55 um only, absorption as sigma_e (exact there, sigma_e = 2 n k omega eps0); '
                 'non-dispersive.')

GST_225_AMORPHOUS = Material(
    name='gst_225_amorphous',
    optical=_optical('amorphous'),
    thermal=dict(k=0.19, rho=5870.0, cp_=210.0),
    description=f'Ge2Sb2Te5 phase-change alloy, amorphous (as deposited). {_OPTICAL_TEXT} {_THERMAL_TEXT} For switching '
                f'give its domains a phase_change_medium feature (its defaults are these optical constants plus the '
                f'crystalline ones and the crystallisation kinetics). {GST_225_REFERENCES}',
    summary='Ge2Sb2Te5 phase-change alloy in its amorphous state, for light at 1550 nm and for heat.\n\n'
            'To let it crystallise or melt as it heats, add a phase-change medium feature to its domains.',
)

GST_225_CRYSTALLINE = Material(
    name='gst_225_crystalline',
    optical=_optical('crystalline'),
    thermal=dict(k=0.57, rho=6270.0, cp_=210.0),
    description=f'Ge2Sb2Te5 phase-change alloy, crystalline (face-centred cubic, the phase a short anneal or SET pulse '
                f'gives). {_OPTICAL_TEXT} {_THERMAL_TEXT} {GST_225_REFERENCES}',
    summary='Ge2Sb2Te5 phase-change alloy in its crystalline state, for light at 1550 nm and for heat. It absorbs '
            'strongly at 1550 nm.',
)
