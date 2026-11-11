"""TetrahedronDomain (igor3d_plan T4b): the Domain of a 'tetrahedra' mesh: TetrahedronMesh + per-element MaterialGrid
(shape (n_elements,)) on the run backend, plus the uniform sample lattice over the window that the Session's 3D reads
(cut planes, volumes, cut positions) sample fields on. Tensor-only accessors (cell_widths, physical_window, ...) do not
exist here: physics using them are refused by PhysicsType.grids."""
import hashlib

import numpy as np

from ..core.constants import DTYPE
from ..core.material_grid import MaterialGrid
from ..core.metric import CARTESIAN
from .sampling3d import PointLocator3D

SAMPLES_PER_LONGEST_SIDE = 96   # sample lattice of the Session's 3D reads: points along the longest window side
_MIN_SAMPLES = 8                # ... at least this many along every axis
LOCATE_CACHE_POINTS = SAMPLES_PER_LONGEST_SIDE ** 3   # located points kept per domain (one full lattice, ~28 MB)


class TetrahedronDomain:
    """Tetrahedron mesh + MaterialGrid of one value per element (decision D12 as triangles); coordinates in m from the
    window corner, the PML shell outside the window."""

    grid_kind = 'tetrahedra'   # core Domain / Domain3D: 'tensor'
    mesh_name = ''             # the model mesh it was built from (model.builder sets it)
    dimensions = 3

    def __init__(self, mesh, materials: MaterialGrid = None, *, backend: str = None):
        self.mesh = mesh
        self.materials = MaterialGrid(mesh.n_elements, backend=backend) if materials is None else materials
        if self.materials.shape != (mesh.n_elements,):
            raise ValueError(f'TetrahedronDomain: MaterialGrid shape {self.materials.shape} != ({mesh.n_elements},)')
        if backend is not None:
            self.materials.use_backend(backend)
        self.backend = self.materials.backend
        self.coordinates = CARTESIAN
        self.boundary_config = None
        x0, y0, z0, x1, y1, z1 = mesh.window
        self.size_x, self.size_y, self.size_z = float(x1 - x0), float(y1 - y0), float(z1 - z0)
        lengths = np.linalg.norm(mesh.points[mesh.edges[:, 1]] - mesh.points[mesh.edges[:, 0]], axis=1)
        self.dx = float(lengths.min())   # smallest element edge [m]
        self.edge_range = (float(lengths.min()), float(lengths.max()))
        longest = max(self.size_x, self.size_y, self.size_z)
        centres = []
        for low, side in zip(mesh.window[:3], (self.size_x, self.size_y, self.size_z)):
            count = max(_MIN_SAMPLES, int(round(SAMPLES_PER_LONGEST_SIDE * side / longest)))
            centres.append(low + (np.arange(count) + 0.5) * (side / count))
        self.sample_centres = tuple(centres)   # float64 uniform sample centres per axis over the window [m]
        self._coordinate_cache = {}
        self._locator = None
        self._located = {}         # points digest -> read-only (elements, reference) of locate()
        self._located_points = 0   # points held in _located

    @property
    def xp(self):
        """Array module of the run backend (numpy or cupy)."""
        return self.backend.xp

    @property
    def n_elements(self) -> int:
        return self.mesh.n_elements

    def use_backend(self, name: str) -> None:
        """Move an array-free domain to backend name ('gpu' / 'cpu'); ValueError once material arrays exist."""
        self.materials.use_backend(name)
        self.backend = self.materials.backend
        self._coordinate_cache.clear()

    def physical_extent(self):
        """(x_min, x_max, y_min, y_max, z_min, z_max) of the window (the PML shell excluded)."""
        x0, y0, z0, x1, y1, z1 = self.mesh.window
        return float(x0), float(x1), float(y0), float(y1), float(z0), float(z1)

    def window_elements(self) -> np.ndarray:
        """Bool per element: True when its centroid lies in the window (not in the PML shell)."""
        return self.mesh.window_elements()

    def host_centroids(self, clamp_to_physical: bool = True) -> np.ndarray:
        """Float64 (n_elements, 3) element centroids; clamped: shell centroids moved just inside the window (the tensor
        PML rule: materials, masks and x / y / z expressions extruded into the shell; TetrahedronMesh.inset_centroids)."""
        return self.mesh.inset_centroids() if clamp_to_physical else self.mesh.centroids()

    def coordinate_grids(self, clamp_to_physical: bool = True):
        """Cached float32 backend (x, y, z) arrays (n_elements,) of the element centroids: the per-element analogue of
        Domain3D.coordinate_grids (field expressions of x, y, z; never write into them)."""
        key = bool(clamp_to_physical)
        grids = self._coordinate_cache.get(key)
        if grids is None:
            centroids = self.host_centroids(key)
            grids = tuple(self.xp.asarray(centroids[:, axis], dtype=DTYPE) for axis in range(3))
            self._coordinate_cache[key] = grids
        return grids

    def release_coordinate_grids(self):
        self._coordinate_cache.clear()

    def locate(self, points) -> tuple:
        """(elements (p,) int64, -1 outside; reference coordinates (p, 3)) of points (p, 3) [m], read-only arrays
        remembered per point set (lattice planes come again for every component) up to LOCATE_CACHE_POINTS."""
        points = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 3)
        key = hashlib.blake2b(points.tobytes(), digest_size=16).digest()
        located = self._located.get(key)
        if located is not None:
            return located
        if self._locator is None:
            self._locator = PointLocator3D(self.mesh)
        located = self._locator.locate(points)
        for array in located:
            array.flags.writeable = False
        if len(points) <= LOCATE_CACHE_POINTS:
            if self._located_points + len(points) > LOCATE_CACHE_POINTS:
                self._located.clear()
                self._located_points = 0
            self._located[key] = located
            self._located_points += len(points)
        return located

    def sample_points(self, axis, index) -> tuple:
        """((n_u, n_v, 3) float64 points [m] of the sample lattice plane normal to axis (0 / 1 / 2) at sample `index`,
        (u axis, v axis)) with u, v the other axes in order."""
        others = tuple(other for other in range(3) if other != axis)
        u_centres, v_centres = (self.sample_centres[other] for other in others)
        points = np.empty((len(u_centres), len(v_centres), 3))
        points[..., axis] = self.sample_centres[axis][int(index)]
        points[..., others[0]] = u_centres[:, None]
        points[..., others[1]] = v_centres[None, :]
        return points, others
