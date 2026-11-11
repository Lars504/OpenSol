"""3D Domain (B1): Grid3D + MaterialGrid (nx, ny, nz) + per-face PML padding over FACES_3D; the 2D Domain's
coordinate rules per axis (physical origin at the window's lower corner, float64 read-only widths / edges / centres)."""
import numpy as np

from .constants import FACES_3D
from .domain import TensorDomainBase
from .grid3d import Grid3D
from .material_grid import MaterialGrid
from .metric import CARTESIAN, Coordinates


class Domain3D(TensorDomainBase):
    """Grid3D + MaterialGrid (nx, ny, nz) + npml_sides over FACES_3D; dx = finest cell width over the three axes.
    coordinates (core.metric.Coordinates): Cartesian, or 'polar' = a cylindrical (r, phi, z) sector (symmetry 2.21:
    x = r - radius_min [m], y = phi - angle_min [rad] logical, as on a polar 2D Domain; grid.dt then is the sector's CFL
    step; DiffusionSystem and the em_fdtd_3d engine read it)."""

    dimensions = 3
    _AXES = ('x', 'y', 'z')
    grid_kind = 'tensor'   # as core.domain.Domain
    mesh_name = ''         # S7.2: the model mesh it was built from (model.builder sets it)

    def __init__(self, grid: Grid3D, materials: MaterialGrid, boundary_config=None, *, npml_sides: dict,
                 size_x: float, size_y: float, size_z: float, uniform_dx: float = None,
                 coordinates: Coordinates = CARTESIAN):
        if not isinstance(coordinates, Coordinates):
            raise TypeError(f"coordinates must be a core.metric.Coordinates, got {type(coordinates).__name__}")
        if coordinates.line:
            raise ValueError('a 1D line window is a 2D Domain one cell across, not a Domain3D')
        self.coordinates = coordinates
        self.grid = grid
        self.materials = materials
        self.backend = materials.backend   # core.backend.Backend of the run (Domain.backend)
        self.boundary_config = boundary_config
        self.npml_sides = {face: int(npml_sides.get(face, 0)) for face in FACES_3D}
        self.ox, self.oy, self.oz = (self.npml_sides[f'{axis}_lo'] for axis in self._AXES)
        self.size_x, self.size_y, self.size_z = float(size_x), float(size_y), float(size_z)
        self.uniform_dx = None if uniform_dx is None else float(uniform_dx)   # set -> cubic cells of this width
        self.non_uniform = self.uniform_dx is None
        self.dx = self.uniform_dx if self.uniform_dx is not None else min(grid.dx_min, grid.dy_min, grid.dz_min)
        for axis, count in zip(self._AXES, grid.shape):
            if count - self.npml_sides[f'{axis}_lo'] - self.npml_sides[f'{axis}_hi'] < 1:
                raise ValueError(f"Domain3D has no physical cells along {axis}: grid {grid.shape}, "
                                 f"PML sides {self.npml_sides}")
        if coordinates.polar:   # grid.dt from the sector's physical widths (the innermost arc r_c[0] dphi)
            radius_first_face = coordinates.radius_min - float(np.sum(grid.dx_arr[:self.ox], dtype=np.float64))
            if not radius_first_face >= -1e-9 * float(grid.dx_arr[0]):   # round-off of the padding sum
                raise ValueError(f'polar Domain3D: the first cell face lies at r = {radius_first_face:.6g} m < 0 '
                                 f'(radius_min {coordinates.radius_min:.6g} m is smaller than the x_lo padding); radii '
                                 'must stay positive in the whole grid')
            grid.use_polar_time_step(radius_first_face)
        self._coordinate_cache = {}

    @classmethod
    def uniform(cls, size_x: float, size_y: float, size_z: float, dx: float, npml_sides: dict = None,
                courant: float = None, boundary_config=None):
        """Cubic cells of width dx; npml_sides (FACES_3D keys, missing = 0) pad the window with dx cells."""
        sides = {face: int((npml_sides or {}).get(face, 0)) for face in FACES_3D}
        totals = [size + (sides[f'{axis}_lo'] + sides[f'{axis}_hi']) * dx
                  for axis, size in zip('xyz', (size_x, size_y, size_z))]
        grid = Grid3D.uniform(*totals, dx, courant=courant)
        return cls(grid, MaterialGrid(grid.nx, grid.ny, grid.nz), boundary_config, npml_sides=sides,
                   size_x=size_x, size_y=size_y, size_z=size_z, uniform_dx=dx)

    @classmethod
    def from_widths(cls, dx_arr, dy_arr, dz_arr, npml_sides: dict, size_x: float, size_y: float, size_z: float,
                    courant: float = None, boundary_config=None, coordinates: Coordinates = CARTESIAN,
                    backend: str = None):
        """Rectilinear (graded) grid from full per-axis widths, PML cells included (polar: y widths in rad)."""
        grid = Grid3D(dx_arr, dy_arr, dz_arr, courant=courant)
        return cls(grid, MaterialGrid(grid.nx, grid.ny, grid.nz, backend=backend), boundary_config,
                   npml_sides=npml_sides, size_x=size_x, size_y=size_y, size_z=size_z, coordinates=coordinates)

    def _origins(self) -> tuple:
        return self.ox, self.oy, self.oz

    def _grid_shape(self) -> tuple:
        return tuple(self.grid.shape)

    def _uniform_width(self):
        return self.uniform_dx

    def _graded_widths(self) -> tuple:
        return self.grid.dx_arr, self.grid.dy_arr, self.grid.dz_arr

    def window_shape(self):
        return tuple(window.stop - window.start for window in self.physical_window())

    def plane_extent(self, axis='z'):
        """(u_min, u_max, v_min, v_max) of the window in a cut normal to axis (FieldData.extent): (x, y) for 'z',
        (x, z) for 'y', (y, z) for 'x'."""
        if axis not in self._AXES:
            raise ValueError(f"plane axis must be one of {self._AXES}, got {axis!r}")
        extent = self.physical_extent()
        return tuple(value for index, name in enumerate(self._AXES) if name != axis
                     for value in extent[2 * index:2 * index + 2])

    # ------------------------------------------------------- coords -> cells
    def pos_to_cell(self, x: float, y: float, z: float):
        return self.pos_to_cell_x(x), self.pos_to_cell_y(y), self.pos_to_cell_z(z)

    def pos_to_cell_x(self, x: float):
        return self._containing(0, x)

    def pos_to_cell_y(self, y: float):
        return self._containing(1, y)

    def pos_to_cell_z(self, z: float):
        """Index of the cell containing physical z (edge -> upper cell), clamped to the physical window."""
        return self._containing(2, z)

    def validate_physical_pos(self, x: float, y: float, z: float, label: str = ""):
        """Raise unless (x, y, z) lies in the physical domain, edges included within 1e-9 of the largest side."""
        self._validate_position((x, y, z), label)
