"""Domain: a 2D tensor grid (uniform or graded) with its MaterialGrid, per-side PML padding, cartesian or polar window
coordinates, cell masks and the run backend."""
import math

import numpy as np

from .constants import DTYPE, FACES
from .grid import Grid2D, edges_from_widths
from .mesh import MeshConfig, NonUniformGrid2D
from .material_grid import MaterialGrid
from .metric import CARTESIAN, Coordinates


class TensorDomainBase:
    """Per-axis cell geometry shared by Domain (2D) and core.domain3d.Domain3D: physical origin at the window's lower
    corner, float64 read-only widths / edges / centres, coordinate grids, containing cells. Subclasses set grid,
    materials, backend, npml_sides, the size_<axis> values and implement _origins / _grid_shape / _uniform_width /
    _graded_widths."""

    _AXES = ('x', 'y')

    def _origins(self) -> tuple:
        raise NotImplementedError

    def _grid_shape(self) -> tuple:
        raise NotImplementedError

    def _uniform_width(self):
        """Width [m] of every cell on a uniform grid, None on a graded one."""
        raise NotImplementedError

    def _graded_widths(self) -> tuple:
        raise NotImplementedError

    def _sizes(self) -> tuple:
        return tuple(getattr(self, f'size_{axis}') for axis in self._AXES)

    @property
    def xp(self):
        """Array module of the run backend (numpy or cupy)."""
        return self.backend.xp

    def use_backend(self, name: str) -> None:
        """Move an array-free domain to backend name ('gpu' / 'cpu'); ValueError once material arrays exist."""
        self.materials.use_backend(name)
        self.backend = self.materials.backend
        self._coordinate_cache.clear()

    # ------------------------------------------------------------ cell geometry

    def physical_window(self):
        """One slice per axis of the cells inside the physical domain (PML ring / shell excluded)."""
        sides = self.npml_sides
        return tuple(slice(sides[f'{axis}_lo'], count - sides[f'{axis}_hi'])
                     for axis, count in zip(self._AXES, self._grid_shape()))

    def cell_widths(self):
        """Per-axis float64 numpy widths (n,) of ALL cells (read-only)."""
        if self._widths is None:
            uniform_width = self._uniform_width()
            if uniform_width is not None:
                widths = tuple(np.full(count, float(uniform_width)) for count in self._grid_shape())
            else:
                widths = tuple(np.array(axis_widths, dtype=np.float64) for axis_widths in self._graded_widths())
            self._widths = tuple(_read_only(axis_widths) for axis_widths in widths)
        return self._widths

    def cell_edges(self):
        """Per-axis float64 numpy edges (n+1,) in physical coords (PML edges < 0 or > size)."""
        if self._edges is None:
            uniform = self._uniform_width() is not None
            self._edges = tuple(_read_only(edges_from_widths(widths, origin, uniform))
                                for widths, origin in zip(self.cell_widths(), self._origins()))
        return self._edges

    def cell_centers(self):
        """Per-axis float64 numpy cell centres in physical coords (read-only)."""
        if self._centers is None:
            self._centers = tuple(_read_only(0.5 * (edges[:-1] + edges[1:])) for edges in self.cell_edges())
        return self._centers

    def physical_extent(self):
        """(x_min, x_max, y_min, y_max[, z_min, z_max]) of the physical window edges: 0 to the window's width sum."""
        extent = []
        for edges, window in zip(self.cell_edges(), self.physical_window()):
            extent += [0.0, float(edges[window.stop])]
        return tuple(extent)

    def host_centres(self, clamp_to_physical: bool = True):
        """Per-axis 1D float32 numpy centres, PML cells clamped to the nearest window centre: the rasterisation centres
        of every build mask."""
        centres = []
        for axis_centres, window in zip(self.cell_centers(), self.physical_window()):
            if clamp_to_physical:
                axis_centres = np.clip(axis_centres, axis_centres[window.start], axis_centres[window.stop - 1])
            centres.append(axis_centres.astype(DTYPE))
        return tuple(centres)

    def coordinate_grids(self, clamp_to_physical: bool = True):
        """Cached float32 (X, Y[, Z]) cell-centre grids (backend arrays) as zero-stride full-grid views of the 1D
        centres (never write into them); clamped: PML cells take the nearest window centre."""
        key = bool(clamp_to_physical)
        grids = self._coordinate_cache.get(key)
        if grids is None:
            shape = self._grid_shape()
            xp = self.xp
            grids = tuple(xp.broadcast_to(xp.asarray(centres).reshape([-1 if other == axis else 1
                                                                       for other in range(len(shape))]), shape)
                          for axis, centres in enumerate(self.host_centres(key)))
            self._coordinate_cache[key] = grids
        return grids

    def release_coordinate_grids(self):
        """Drop the cached coordinate grids (recomputed on the next coordinate_grids call)."""
        self._coordinate_cache.clear()

    # ------------------------------------------------------- coords -> cells

    def _containing(self, axis_index, coord):
        """Index of the cell containing physical coord along an axis (edge -> upper cell), clamped to the window."""
        window = self.physical_window()[axis_index]
        return containing_cell(coord, self.cell_edges()[axis_index], window.start, window.stop - 1)

    def _validate_position(self, position, label):
        """Raise unless position lies in the physical domain, edges included within 1e-9 of the largest side."""
        sizes = self._sizes()
        edge_tolerance = 1e-9 * max(sizes)   # a sum of lengths can overshoot an edge by an ulp
        if not all(-edge_tolerance <= value <= size + edge_tolerance for value, size in zip(position, sizes)):
            coordinates = ', '.join(f'{value*1e6:.10g}' for value in position)
            extents = ' x '.join(f'{size*1e6:.10g}' for size in sizes)
            raise ValueError(f"{label} position ({coordinates}) µm outside physical domain ({extents}) µm")


class Domain(TensorDomainBase):
    """Grid + MaterialGrid + per-side PML padding; physical coords have their origin at the window's lower corner.
    coordinates (core.metric.Coordinates): the in-plane grid; on a polar one x = r - radius_min [m] and y = angle -
    angle_min [rad] are logical (widths, edges, centres, masks), and only physics whose coordinate_grids list it run."""

    dimensions = 2   # core.domain3d.Domain3D: 3
    grid_kind = 'tensor'   # S7.5: a triangle-mesh domain (fem) has 'triangles'
    mesh_name = ''         # S7.2: the model mesh it was built from (model.builder sets it; '' in scripts)

    def __init__(self, grid, materials: MaterialGrid, boundary_config=None,
                 *, ox: int = 0, oy: int = 0,
                 size_x: float, size_y: float,
                 dx: float = None,
                 non_uniform: bool = False, mesh_config=None,
                 npml_sides: dict = None, coordinates: Coordinates = CARTESIAN, backend: str = None):
        if not isinstance(coordinates, Coordinates):
            raise TypeError(f"coordinates must be a core.metric.Coordinates, got {type(coordinates).__name__}")
        self.coordinates = coordinates
        if backend is not None:
            materials.use_backend(backend)
        self.backend = materials.backend   # core.backend.Backend of the run: every array lives on backend.xp
        self.grid = grid
        self.materials = materials
        self.boundary_config = boundary_config
        self.ox = ox
        self.oy = oy
        self.size_x = float(size_x)   # requested physical (window) size, PML excluded
        self.size_y = float(size_y)
        self.dx = dx if dx is not None else float(grid.dx)
        self.non_uniform = non_uniform
        self.mesh_config = mesh_config

        if npml_sides is None:
            npml_sides = {'x_lo': ox, 'x_hi': ox, 'y_lo': oy, 'y_hi': oy}
        if npml_sides['x_lo'] != ox or npml_sides['y_lo'] != oy:
            raise ValueError(f"Domain origin ({ox}, {oy}) disagrees with npml_sides {npml_sides}")
        self.npml_sides = dict(npml_sides)
        if grid.nx - npml_sides['x_lo'] - npml_sides['x_hi'] < 1 or \
                grid.ny - npml_sides['y_lo'] - npml_sides['y_hi'] < 1:
            raise ValueError(f"Domain has no physical cells: grid {grid.nx} x {grid.ny}, "
                             f"PML sides {npml_sides}")

        self._widths = None
        self._edges = None
        self._centers = None
        self._coordinate_cache = {}

    @classmethod
    def uniform(cls, size_x: float, size_y: float, dx: float,
                npml: int = 15, courant: float = None, boundary_config=None,
                verbose: bool = True, backend: str = None):
        if boundary_config is not None:
            sides = boundary_config.npml_sides
        else:
            sides = {face: npml for face in FACES}

        total_x = size_x + (sides['x_lo'] + sides['x_hi']) * dx
        total_y = size_y + (sides['y_lo'] + sides['y_hi']) * dx

        grid = Grid2D(total_x, total_y, dx, courant, verbose=verbose)
        materials = MaterialGrid(grid.nx, grid.ny, backend=backend)

        return cls(
            grid, materials, boundary_config=boundary_config,
            ox=sides['x_lo'], oy=sides['y_lo'],
            size_x=size_x, size_y=size_y, dx=dx,
            non_uniform=False, npml_sides=sides,
        )

    @classmethod
    def from_mesh(cls, mesh_config: MeshConfig, npml: int = 15,
                  boundary_config=None, courant: float = None, verbose: bool = True, backend: str = None):
        if boundary_config is not None:
            sides = boundary_config.npml_sides
        else:
            sides = {face: npml for face in FACES}

        # mirror ghosts (EM BoundaryConfig.ghost_faces) take the window edge cell's width
        grid = mesh_config.build(pml_dx=mesh_config.dx_base, npml_sides=sides, courant=courant, verbose=verbose,
                                 ghost_faces=getattr(boundary_config, 'ghost_faces', ()))
        materials = MaterialGrid(grid.nx, grid.ny, backend=backend)

        return cls(
            grid, materials, boundary_config=boundary_config,
            ox=sides['x_lo'], oy=sides['y_lo'],
            size_x=mesh_config.size_x, size_y=mesh_config.size_y,
            dx=float(grid.dx_min),
            non_uniform=True, mesh_config=mesh_config, npml_sides=sides,
        )

    @classmethod
    def from_widths(cls, dx_cells, dy_cells, coordinates: Coordinates = CARTESIAN, courant: float = None,
                    backend: str = None):
        """Padding-free Domain of explicit window cell widths (x [m]; y [m], or [rad] on a polar grid)."""
        grid = NonUniformGrid2D(dx_cells, dy_cells, courant=courant, verbose=False)
        return cls(grid, MaterialGrid(grid.nx, grid.ny, backend=backend), size_x=grid.size_x, size_y=grid.size_y,
                   non_uniform=True, npml_sides={face: 0 for face in FACES}, coordinates=coordinates)

    def _origins(self) -> tuple:
        return self.ox, self.oy

    def _grid_shape(self) -> tuple:
        return self.grid.nx, self.grid.ny

    def _uniform_width(self):
        return None if self.non_uniform else float(self.dx)

    def _graded_widths(self) -> tuple:
        return self.grid.dx_arr, self.grid.dy_arr

    # ------------------------------------------------------- coords -> cells

    def pos_to_cell(self, x: float, y: float):
        return self.pos_to_cell_x(x), self.pos_to_cell_y(y)

    def pos_to_cell_x(self, x: float):
        """Index of the cell containing physical x, clamped to the physical window."""
        return self._containing(0, x)

    def pos_to_cell_y(self, y: float):
        """Index of the cell containing physical y, clamped to the physical window."""
        return self._containing(1, y)

    def validate_physical_pos(self, x: float, y: float, label: str = ""):
        """Raise unless (x, y) lies in the physical domain, edges included within 1e-9 of the larger side."""
        self._validate_position((x, y), label)

    # ------------------------------------------------------------- masks

    def rect_mask(self, x0: float, y0: float, x1: float, y1: float):
        """Bool backend mask of cells whose centre lies in [x_min, x_max) x [y_min, y_max) (corners in any order)."""
        X, Y = self.coordinate_grids(False)
        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)
        return (X >= x_min) & (X < x_max) & (Y >= y_min) & (Y < y_max)

    def circle_mask(self, cx: float, cy: float, radius: float):
        """Bool backend mask of cells whose centre lies within `radius` of (cx, cy) (boundary included)."""
        if not radius >= 0:
            raise ValueError(f"circle radius must be >= 0, got {radius}")
        X, Y = self.coordinate_grids(False)
        return (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2


def _read_only(array):
    array.flags.writeable = False
    return array


def containing_cell(coord, edges, first, last):
    """Cell index i with edges[i] <= coord < edges[i+1] (1e-9-cell snap at edges), clamped to [first, last]."""
    coord = float(coord)
    if not math.isfinite(coord):
        raise ValueError(f"non-finite coordinate {coord}")
    snap = 1e-9 * (edges[first + 1] - edges[first])
    index = int(np.searchsorted(edges, coord + snap, side='right')) - 1
    return min(max(index, first), last)
