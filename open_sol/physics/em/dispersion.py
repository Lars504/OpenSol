"""Complex permittivity eps(omega) of the FDTD material arrays in the exp(+j omega t) convention (lossy: Im < 0), one
component at a time; H_z face values come from yee_grid.face_average of eps, sigma_e and the mask weights."""
import math

import numpy as np

from ...core.constants import EPS0


def complex_permittivity(eps_background, sigma_e, poles_with_masks, omega, xp=np):
    """eps_background - j sigma_e/(omega eps0) + sum chi_k(omega) mask_k (complex128 xp array, relative); mask None =
    everywhere. eps_background is the array FDTD steps (Drude eps_inf written; Lorentz eps_r = eps_inf)."""
    omega = float(omega)
    if not (math.isfinite(omega) and omega > 0.0):
        raise ValueError(f"complex_permittivity needs a finite angular frequency omega > 0 [rad/s], got {omega!r}")
    permittivity = (xp.asarray(eps_background, dtype=xp.complex128)
                    - 1j * xp.asarray(sigma_e, dtype=xp.float64) / (omega * EPS0))
    for pole, mask in poles_with_masks:
        weight = 1.0 if mask is None else xp.asarray(mask, dtype=xp.float64)
        permittivity = permittivity + complex(pole.susceptibility(omega)) * weight
    return permittivity
