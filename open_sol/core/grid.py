"""Grid2D (uniform square cells, EM Courant time step) and the courant_time_step / cell-count / cell-edge helpers."""
import numpy as np

from .constants import DTYPE, C0

DEFAULT_COURANT_FACTOR = 0.99   # fraction of the 2D CFL limit


def courant_time_step(dx_min: float, dy_min: float, courant_factor: float = None):
    """EM time step (DTYPE) = courant_factor * 2D CFL limit of the finest cells; factor in (0, 1], None = 0.99."""
    if courant_factor is None:
        courant_factor = DEFAULT_COURANT_FACTOR
    if not 0.0 < float(courant_factor) <= 1.0:
        raise ValueError(f"courant must be a fraction of the 2D CFL limit in (0, 1], got {courant_factor} "
                         f"(it is not C0 dt / dx: 0.99 means dt = 0.99 dx / (C0 sqrt(2)) on square cells)")
    inverse_spacing_squared = 1.0 / float(dx_min) ** 2 + 1.0 / float(dy_min) ** 2
    return DTYPE(float(courant_factor) / (C0 * np.sqrt(inverse_spacing_squared)))


def uniform_cell_count(length: float, dx: float) -> int:
    """Cells of width dx covering length; the -1e-9 removes float round-up (4e-6 / 20e-9 = 200.00000000000003)."""
    return int(np.ceil(length / dx - 1e-9))


def edges_from_widths(widths, origin_index: int, uniform: bool):
    """float64 edges (n+1,) of cell widths with edge origin_index at 0; uniform uses (k - origin) * widths[0]."""
    widths = np.asarray(widths, dtype=np.float64)
    if uniform:
        return (np.arange(widths.size + 1) - origin_index) * float(widths[0])
    edges = np.concatenate(([0.0], np.cumsum(widths)))
    return edges - edges[origin_index]


class Grid2D:
    """Uniform square-cell grid; dt is the EM Courant step (courant = fraction of the 2D CFL limit, default 0.99)."""

    def __init__(self, size_x: float, size_y: float, dx: float, courant: float = None,
                 verbose: bool = True):
        if not dx > 0:
            raise ValueError(f"Grid2D needs dx > 0, got {dx}")
        self.dx = DTYPE(dx)
        self.dy = DTYPE(dx)

        self.nx = uniform_cell_count(size_x, dx)
        self.ny = uniform_cell_count(size_y, dx)
        if self.nx < 1 or self.ny < 1:
            raise ValueError(f"Grid2D has no cells: size {size_x} x {size_y} m, dx {dx} m")

        self.size_x = self.nx * self.dx
        self.size_y = self.ny * self.dy

        self.dt = courant_time_step(dx, dx, courant)
        self.courant = DTYPE(DEFAULT_COURANT_FACTOR if courant is None else courant)

        if verbose:
            print(f"Grid2D: {self.nx} x {self.ny} cells, "
                  f"dx={self.dx*1e9:.1f} nm, dt={self.dt*1e18:.3f} as, "
                  f"domain={self.size_x*1e6:.2f} x {self.size_y*1e6:.2f} µm")

    @property
    def shape(self):
        return (self.nx, self.ny)
