"""PotentialFields2D: full-grid potential V, field E, flux (J or D), source slots, Joule heating and the
electrolyte ion space charge of PotentialPhysics."""
import numpy as np

from ...core.constants import DTYPE


class PotentialFields2D:
    """Full-grid float32 potential V [V], field E_x / E_y [V/m], face-averaged flux (J [A/m^2] or D [C/m^2]),
    coupled source, applied source_ext, (currents only) the Joule heating density q_joule [W/m^3] and
    (electrostatics only) the electrolyte space charge rho_ion [C/m^3]; xp: the Domain's array module."""

    def __init__(self, nx: int, ny: int, with_joule: bool, with_ions: bool, xp=np):
        self.nx = nx
        self.ny = ny
        shape = (nx, ny)
        self.V = xp.zeros(shape, dtype=DTYPE)
        self.E_x = xp.zeros(shape, dtype=DTYPE)
        self.E_y = xp.zeros(shape, dtype=DTYPE)
        self.flux_x = xp.zeros(shape, dtype=DTYPE)
        self.flux_y = xp.zeros(shape, dtype=DTYPE)
        self.source = xp.zeros(shape, dtype=DTYPE)       # coupled slot (overwritten per inject_source)
        self.source_ext = xp.zeros(shape, dtype=DTYPE)   # applied by features (sources add)
        self.q_joule = xp.zeros(shape, dtype=DTYPE) if with_joule else None
        self.rho_ion = xp.zeros(shape, dtype=DTYPE) if with_ions else None
