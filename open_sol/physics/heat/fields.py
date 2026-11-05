"""TemperatureField2D / TemperatureField3D: full-grid temperature T [K], coupled heat source Q and applied heat source
Q_ext [W/m^3] of HeatPhysics; heat_state, the state() vocabulary of every heat solver."""
import numpy as np

from ...core.constants import DTYPE


# MaterialGrid arrays every heat state() exposes (conduction, capacity, thermo-optic and expansion inputs)
STATE_MATERIALS = ('k_xx', 'k_yy', 'k_zz', 'k_xy', 'rho', 'cp', 'n0_zz', 'dndT_xx', 'dndT_yy', 'dndT_zz', 'alpha_xx',
                   'alpha_yy')


def heat_state(T, T0, Q, Q_ext, materials) -> dict:
    """state() of a heat solver: T, the reference T0 (0-d run-backend array), Q, Q_ext and STATE_MATERIALS."""
    return {'T': T, 'T0': T0, 'Q': Q, 'Q_ext': Q_ext, **{name: getattr(materials, name) for name in STATE_MATERIALS}}


class TemperatureField2D:
    """Full-grid temperature T [K], coupled heat source Q and applied heat source Q_ext [W/m^3] (xp: the Domain's
    array module, numpy or cupy)."""

    def __init__(self, nx: int, ny: int, T0: float = 293.15, xp=np):
        self.nx = nx
        self.ny = ny
        self.T = xp.full((nx, ny), DTYPE(T0), dtype=DTYPE)
        self.Q = xp.zeros((nx, ny), dtype=DTYPE)
        self.Q_ext = xp.zeros((nx, ny), dtype=DTYPE)


class TemperatureField3D:
    """Full-grid (nx, ny, nz) temperature T [K], coupled heat source Q and applied heat source Q_ext [W/m^3]."""

    def __init__(self, nx: int, ny: int, nz: int, T0: float = 293.15, xp=np):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.T = xp.full((nx, ny, nz), DTYPE(T0), dtype=DTYPE)
        self.Q = xp.zeros((nx, ny, nz), dtype=DTYPE)
        self.Q_ext = xp.zeros((nx, ny, nz), dtype=DTYPE)
