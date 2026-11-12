"""TriangleDomain (mesh plan 2.5): the Domain of a 'triangles' mesh: TriangleMesh + per-element MaterialGrid
(shape (n_elements,)) on the run backend. Tensor-only accessors (cell_widths, cell_edges, pos_to_cell, ...) do not
exist here: physics using them are refused by PhysicsType.grids."""
import hashlib

import numpy as np

from ..core.constants import DTYPE
from ..core.material_grid import MaterialGrid
from ..core.metric import CARTESIAN
from .assembly import element_areas
from .dofs import lagrange_dofs
from .mesher import MERGE_TOLERANCE
from .sampling import Location, LocatedPoints, locator, quadrature_samples

LOCATE_CACHE_POINTS = 2 ** 21   # located points + sampler point equivalents kept per domain (~150 MB host with P2)


def _digest(array) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(array).tobytes(), digest_size=16).digest()


class TriangleDomain:
    """Triangle mesh + MaterialGrid of one value per element (decision D12); coordinates in m from the window corner,
    the PML layer ring outside it (mesh.pml_faces)."""

    grid_kind = 'triangles'   # core Domain / Domain3D: 'tensor'
    mesh_name = ''            # S7.2: the model mesh it was built from (model.builder sets it)
    dimensions = 2

    def __init__(self, mesh, materials: MaterialGrid = None, *, backend: str = None):
        self.mesh = mesh
        self.materials = MaterialGrid(mesh.n_elements, backend=backend) if materials is None else materials
        if self.materials.shape != (mesh.n_elements,):
            raise ValueError(f'TriangleDomain: MaterialGrid shape {self.materials.shape} != ({mesh.n_elements},)')
        if backend is not None:
            self.materials.use_backend(backend)
        self.backend = self.materials.backend
        self.coordinates = CARTESIAN
        self.boundary_config = None
        x0, y0, x1, y1 = mesh.window
        self.size_x = float(x1 - x0)
        self.size_y = float(y1 - y0)
        lengths = np.linalg.norm(mesh.points[mesh.edges[:, 1]] - mesh.points[mesh.edges[:, 0]], axis=1)
        self.dx = float(lengths.min())   # smallest element edge [m]
        self._coordinate_cache = {}
        self._located = {}         # (kind, geometry order, ..., digest) -> located() / quadrature_located() result
        self._located_points = 0   # point equivalents held in _located (located sets + sampling operators)
        self._identity = {}        # (id(read-only points), geometry order) -> (points, _located key)

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
        """(x_min, x_max, y_min, y_max) of the window (the PML ring excluded)."""
        return 0.0, self.size_x, 0.0, self.size_y

    def window_elements(self) -> np.ndarray:
        """Bool per element: True inside the window (not in the PML layer ring)."""
        return self.mesh.pml_faces == 0

    def host_centroids(self, clamp_to_physical: bool = True) -> np.ndarray:
        """Float64 (n_elements, 2) element centroids (straight); clamped: ring centroids moved just inside the window
        (the tensor PML rule: materials, masks and x / y expressions extruded into the ring)."""
        centroids = self.mesh.centroids()
        if clamp_to_physical:
            x0, y0, x1, y1 = self.mesh.window
            inset = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)   # as model.triangle_meshing.element_masks
            centroids[:, 0] = np.clip(centroids[:, 0], x0 + inset, x1 - inset)
            centroids[:, 1] = np.clip(centroids[:, 1], y0 + inset, y1 - inset)
        return centroids

    def coordinate_grids(self, clamp_to_physical: bool = True):
        """Cached float32 backend (x, y) arrays (n_elements,) of the element centroids: the per-element analogue of
        Domain.coordinate_grids (field expressions of x, y; never write into them)."""
        key = bool(clamp_to_physical)
        grids = self._coordinate_cache.get(key)
        if grids is None:
            centroids = self.host_centroids(key)
            grids = tuple(self.xp.asarray(centroids[:, axis], dtype=DTYPE) for axis in (0, 1))
            self._coordinate_cache[key] = grids
        return grids

    def release_coordinate_grids(self):
        self._coordinate_cache.clear()

    def element_areas(self, geometry_order: int = 2) -> np.ndarray:
        """Float64 (n_elements,) areas [m^2] (geometry order 2: curved elements exact to the P2 map)."""
        return element_areas(lagrange_dofs(self.mesh, 0, geometry_order=geometry_order))

    def locate(self, points, geometry_order: int = 2) -> np.ndarray:
        """Read-only int64 containing element of each (n, 2) point [m], -1 outside the mesh."""
        return self.located(points, geometry_order).location.element

    def located(self, points, geometry_order: int = 2) -> LocatedPoints:
        """LocatedPoints of (n, 2) points [m] (read-only; their basis and sampling operators shared by every read),
        remembered per point set up to LOCATE_CACHE_POINTS: Session rasters and point reads repeat for every field and
        snapshot. A read-only array seen before is matched by identity (no hash)."""
        order = int(geometry_order)
        held = self._identity.get((id(points), order))
        if held is not None and held[0] is points and held[1] in self._located:
            return self._located[held[1]]
        host = np.ascontiguousarray(points, dtype=np.float64).reshape(-1, 2)
        key = ('points', order, _digest(host))
        located = self._located.get(key)
        if located is None:
            located = LocatedPoints(locator(self.mesh, order).locate(host), charge=self._charge)
            self._remember(key, located, len(host))
        if isinstance(points, np.ndarray) and not points.flags.writeable and key in self._located:
            self._identity[(id(points), order)] = (points, key)   # the entry holds the array: its id stays unique
        return located

    def quadrature_located(self, degree, elements, geometry_order: int = 2) -> tuple:
        """(QuadratureSamples (read-only), their LocatedPoints) of the elements selected by bool `elements`, remembered
        per (degree, selection) as located(): derived values of one selection share them."""
        elements = np.asarray(elements, dtype=bool)
        key = ('quadrature', int(geometry_order), int(degree), _digest(elements))
        remembered = self._located.get(key)
        if remembered is None:
            samples = quadrature_samples(self.mesh, degree, elements, geometry_order)
            for array in (samples.x, samples.y, samples.weight):
                array.flags.writeable = False
            remembered = (samples, LocatedPoints(Location(samples.element, samples.reference), charge=self._charge))
            self._remember(key, remembered, len(samples.x))
        return remembered

    def _remember(self, key, value, count):
        """Keep a located set of count points; the whole store is cleared when it would pass LOCATE_CACHE_POINTS."""
        if count > LOCATE_CACHE_POINTS:
            return
        self._charge(count)
        self._located[key] = value

    def _charge(self, count):
        """Count point equivalents held by the store (located sets, their sampling operators); clear it when it would
        pass LOCATE_CACHE_POINTS (objects a caller holds stay valid)."""
        if self._located_points + count > LOCATE_CACHE_POINTS:
            self._located.clear()
            self._identity.clear()
            self._located_points = 0
        self._located_points += count

    def edge_tag(self, name) -> np.ndarray:
        """Int32 mesh edges tagged name (outer face, object outline, line entity); empty when untagged."""
        return self.mesh.edge_tags.get(name, np.zeros(0, dtype=np.int32))
