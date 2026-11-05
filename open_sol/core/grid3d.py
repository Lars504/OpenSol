"""3D rectilinear grid (B1): per-axis float32 cell widths, PML included; dt from the 3D CFL limit."""
import numpy as np

from .constants import DTYPE, C0
from .grid import DEFAULT_COURANT_FACTOR, uniform_cell_count


def courant_time_step_3d(dx_min: float, dy_min: float, dz_min: float, courant_factor: float = None):
    """EM time step (DTYPE) = courant_factor * 3D CFL limit of the finest cells; factor in (0, 1], None = 0.99."""
    if courant_factor is None:
        courant_factor = DEFAULT_COURANT_FACTOR
    if not 0.0 < float(courant_factor) <= 1.0:
        raise ValueError(f"courant must be a fraction of the 3D CFL limit in (0, 1], got {courant_factor} "
                         f"(0.99 means dt = 0.99 dx / (C0 sqrt(3)) on cubic cells)")
    inverse_spacing_squared = 1.0 / float(dx_min) ** 2 + 1.0 / float(dy_min) ** 2 + 1.0 / float(dz_min) ** 2
    return DTYPE(float(courant_factor) / (C0 * np.sqrt(inverse_spacing_squared)))


def finest_physical_widths_3d(dx_arr, dy_arr, dz_arr, radius_first_face=None) -> tuple:
    """(dx, dy, dz) finest physical widths [m] of the CFL limit; a cylindrical (r, phi, z) sector (radius_first_face =
    r at grid x edge 0, y widths in rad, symmetry 2.21) takes the arc r_c[0] dphi_min of the innermost centres."""
    finest = [float(np.min(np.asarray(widths, dtype=np.float64))) for widths in (dx_arr, dy_arr, dz_arr)]
    if radius_first_face is not None:
        finest[1] *= float(radius_first_face) + 0.5 * float(np.asarray(dx_arr, dtype=np.float64)[0])
    return tuple(finest)


class Grid3D:
    """Rectilinear grid from per-cell widths (nx,), (ny,), (nz,); arrays are (nx, ny, nz) C-order, so the flat
    index is (i * ny + j) * nz + k (z contiguous). dt = courant x 3D CFL limit of the finest cells (a polar Domain3D
    switches it to the sector's physical widths: use_polar_time_step)."""

    def __init__(self, dx_arr, dy_arr, dz_arr, courant: float = None):
        self.dx_arr = np.asarray(dx_arr, dtype=DTYPE)
        self.dy_arr = np.asarray(dy_arr, dtype=DTYPE)
        self.dz_arr = np.asarray(dz_arr, dtype=DTYPE)
        self.nx, self.ny, self.nz = len(self.dx_arr), len(self.dy_arr), len(self.dz_arr)
        if min(self.nx, self.ny, self.nz) < 1:
            raise ValueError(f"Grid3D has no cells: {self.nx} x {self.ny} x {self.nz}")
        # total extent including PML; cell coordinates come from Domain3D.cell_edges / cell_centers
        self.size_x = float(np.sum(self.dx_arr, dtype=np.float64))
        self.size_y = float(np.sum(self.dy_arr, dtype=np.float64))
        self.size_z = float(np.sum(self.dz_arr, dtype=np.float64))
        self.dx_min, self.dy_min, self.dz_min = (float(np.min(widths))
                                                 for widths in (self.dx_arr, self.dy_arr, self.dz_arr))
        self.dx, self.dy, self.dz = DTYPE(self.dx_min), DTYPE(self.dy_min), DTYPE(self.dz_min)
        self.dt = courant_time_step_3d(self.dx_min, self.dy_min, self.dz_min, courant)
        self.courant = DTYPE(DEFAULT_COURANT_FACTOR if courant is None else courant)
        self._courant_factor = courant
        self.radius_first_face = None   # polar (symmetry 2.21): r [m] at grid x edge 0, y widths in rad

    def use_polar_time_step(self, radius_first_face: float) -> None:
        """y widths are angles [rad] of a cylindrical sector whose grid x edge 0 lies at radius_first_face [m]: dt =
        courant x the 3D CFL limit of finest_physical_widths_3d (the innermost arc)."""
        self.radius_first_face = float(radius_first_face)
        finest = finest_physical_widths_3d(self.dx_arr, self.dy_arr, self.dz_arr, self.radius_first_face)
        self.dt = courant_time_step_3d(*finest, self._courant_factor)

    @classmethod
    def uniform(cls, total_x: float, total_y: float, total_z: float, dx: float, courant: float = None):
        """Cubic cells of width dx covering the totals (PML included; core.grid.uniform_cell_count per axis)."""
        if not dx > 0:
            raise ValueError(f"Grid3D needs dx > 0, got {dx}")
        counts = [uniform_cell_count(total, dx) for total in (total_x, total_y, total_z)]
        return cls(*(np.full(count, dx, dtype=DTYPE) for count in counts), courant=courant)

    @property
    def shape(self):
        return (self.nx, self.ny, self.nz)

    @property
    def n_cells(self) -> int:
        return self.nx * self.ny * self.nz
