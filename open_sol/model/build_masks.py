"""MeshMasks: the grid-kind part of a BuildContext (domain-object masks, coordinates of field expressions, boundary
and point lookups), one class per kind: Cartesian / mapped 2D windows, 3D tensor boxes / sectors, triangles,
tetrahedra."""
import numpy as np

from ..core.backend import to_host
from ..core.constants import DTYPE
from .geometry import (GEOMETRY_KINDS, bounding_box, box_window, perimeter_mask as geometry_perimeter_mask,
                       perimeter_weights as geometry_perimeter_weights, rasterize, rasterize_box, rasterize_mapped,
                       rasterize_mapped_box, rasterize_window)


def mesh_masks(ctx):
    """The MeshMasks of ctx's mesh: by grid kind, a tensor grid also by dimensions and DomainMap (polar / spherical
    windows, 1D lines and cylindrical sectors are mapped)."""
    kind = ctx.domain.grid_kind
    if kind != 'tensor':
        return ELEMENT_MASKS[kind](ctx)
    if ctx.domain.dimensions == 3:
        return SectorMasks(ctx) if ctx.domain_map.mapped else BoxMasks(ctx)
    return MappedMasks(ctx) if ctx.domain_map.mapped else CartesianMasks(ctx)


class MeshMasks:
    """Grid-kind part of the BuildContext `ctx` of one mesh; BuildContext's public mask / coordinate / boundary methods
    delegate here. A lookup the kind lacks raises ValueError."""
    paints_by_slabs = False   # True on 3D tensor grids: material_index per z-slab, box masks, plane-wise expressions

    def __init__(self, ctx):
        self.ctx = ctx
        self.domain = ctx.domain
        self.xp = ctx.xp
        self._object_masks = {}   # name -> (window slices, cropped mask)

    def release(self):
        """Drop the cached masks and coordinate grids."""
        self._object_masks.clear()

    def _domain_object(self, name):
        shape = self.ctx._shape(name)
        if GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
            raise ValueError(f'{name!r} is a {shape.kind}, not a domain object')
        return shape

    # ---- domain-object masks ----
    def object_mask(self, name):
        """(window slices, cropped mask) of a domain object as domain_mask unions them (window masks; 3D tensor grids:
        box masks)."""
        return self.object_window_mask(name)

    def object_window_mask(self, name):
        """Cached (window slices, cupy bool mask cropped to it) of a domain object (read-only by convention)."""
        cached = self._object_masks.get(name)
        if cached is None:
            self._domain_object(name)
            cached = self._object_masks[name] = self._window_mask(name)
        return cached

    def _window_mask(self, name):
        raise NotImplementedError

    def object_box_mask(self, name):
        raise ValueError('object_box_mask works on 3D tensor grids only; 2D builds and element meshes use '
                         'object_window_mask')

    def region_window(self, objects, is_all):
        """3D tensor grids: full-grid window slices of a material region (BuildContext.region_window_mask)."""
        raise ValueError('region_window_mask works on 3D tensor grids only')

    # ---- coordinates ----
    def coordinate_grids(self):
        """2D float32 full-grid (x, y) cell centres [m] in the drawing plane (field expressions, masks): the Domain's
        clamped zero-stride grids."""
        return self.domain.coordinate_grids(clamp_to_physical=True)[:2]

    def field_coordinates(self):
        """{'x', 'y'(, 'z')} of a field expression BuildContext.eval_field evaluates at once (every cell or element)."""
        return dict(zip(('x', 'y'), self.coordinate_grids()))

    def dynamic_coordinates(self):
        """{'x', 'y'(, 'z')} zero-stride grids a state- / time-dependent expression captures (2D: x, y)."""
        return dict(zip(('x', 'y'), self.coordinate_grids()))

    def plane_coordinates(self, window_x=slice(None), window_y=slice(None)):
        raise ValueError('plane_coordinates works on 3D tensor grids only')

    # ---- boundaries and points ----
    def boundary_edges(self, selection):
        raise ValueError(f'boundary_edges works on triangle meshes only (this mesh is {self.domain.grid_kind})')

    def boundary_faces(self, selection):
        raise ValueError(f'boundary_faces works on tetrahedron meshes only (this mesh is {self.domain.grid_kind})')

    def point_node(self, name):
        raise ValueError(f'point_node works on triangle meshes only (this mesh is {self.domain.grid_kind})')

    def perimeter_mask(self, objects):
        raise ValueError('perimeter_mask works on 2D tensor meshes only; on a triangle mesh use boundary_edges / '
                         'point_node')

    def perimeter_weights(self, objects):
        raise ValueError('perimeter_weights works on 2D tensor meshes only; on a triangle mesh use boundary_edges / '
                         'point_node')


class CartesianMasks(MeshMasks):
    """2D Cartesian tensor windows: masks on the float32 clamped cell centres (geometry.rasterize_window), perimeter
    masks / weights of object unions."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self._host_centres = None   # numpy float32 clamped (x, y) cell centres of the rasterisation
        self._perimeter_masks = {}
        self._perimeter_weights = {}
        self._window_unions = {}

    def release(self):
        super().release()
        self._host_centres = None
        self._perimeter_masks.clear()
        self._perimeter_weights.clear()
        self._window_unions.clear()

    def _window_mask(self, name):
        if self._host_centres is None:
            X, Y = self.domain.coordinate_grids(clamp_to_physical=True)
            self._host_centres = (np.array(to_host(X[:, 0])), np.array(to_host(Y[0, :])))
        return rasterize_window(name, self.ctx.resolved_geometry, *self._host_centres, self.xp)

    def perimeter_mask(self, objects):
        """Cached cupy bool full-grid mask of the union's perimeter cells, window cells only (computed on the union's
        occupied box plus a 2-cell halo: exact, the stencils reach 2 cells)."""
        key = tuple(objects)
        mask = self._perimeter_masks.get(key)
        if mask is None:
            mask = self.xp.zeros((self.domain.grid.nx, self.domain.grid.ny), dtype=np.bool_)
            crop = self._perimeter_crop(objects)
            if crop is not None:
                in_window = mask[self.domain.physical_window()]   # view
                in_window[crop] = geometry_perimeter_mask(self._window_union(objects)[crop], self.xp)
            self._perimeter_masks[key] = mask
        return mask

    def perimeter_weights(self, objects):
        """Cached cupy float32 full-grid perimeter length / cell area [1/m] of the union (geometry.perimeter_weights),
        window cells only (on the union's occupied box plus a 2-cell halo, as perimeter_mask)."""
        key = tuple(objects)
        weights = self._perimeter_weights.get(key)
        if weights is None:
            window_x, window_y = self.domain.physical_window()
            dx_cells, dy_cells = self.domain.cell_widths()
            weights = self.xp.zeros((self.domain.grid.nx, self.domain.grid.ny), dtype=DTYPE)
            crop = self._perimeter_crop(objects)
            if crop is not None:
                crop_x, crop_y = crop
                in_window = weights[window_x, window_y]   # view
                in_window[crop] = geometry_perimeter_weights(
                    self._window_union(objects)[crop], dx_cells[window_x][crop_x], dy_cells[window_y][crop_y], self.xp)
            self._perimeter_weights[key] = weights
        return weights

    def _perimeter_crop(self, objects):
        """(slice x, slice y) of the window union's occupied box widened by 2 cells (clipped to the window): outside
        it every cell is False and no perimeter stencil reaches in; None for an empty union (one host sync)."""
        union = self._window_union(objects)
        crop = []
        for axis in (1, 0):   # rows (x): any along y; columns (y): any along x
            occupied = to_host(self.xp.flatnonzero(union.any(axis=axis)))
            if not occupied.size:
                return None
            crop.append(slice(max(int(occupied[0]) - 2, 0), min(int(occupied[-1]) + 3, union.shape[1 - axis])))
        return tuple(crop)

    def _window_union(self, objects):
        """Cached window view of the union of domain objects' masks (shared by perimeter_mask / perimeter_weights)."""
        key = tuple(objects)
        union = self._window_unions.get(key)
        if union is None:
            union = self.xp.zeros((self.domain.grid.nx, self.domain.grid.ny), dtype=np.bool_)
            for name in objects:
                window, cropped = self.object_window_mask(name)
                union[window] |= cropped
            union = union[self.domain.physical_window()]
            self._window_unions[key] = union
        return union


class _MappedGrids:
    """Mixin of the mapped tensor grids (polar / spherical windows, 1D lines, cylindrical sectors): the drawing-plane
    centres of the (nx, ny) cells, cached."""
    _mapped_grids = None

    def release(self):
        super().release()
        self._mapped_grids = None

    def coordinate_grids(self):
        """2D float32 full-grid drawing-plane (x, y) cell centres [m] (DomainMap.to_physical; no PML there)."""
        if self._mapped_grids is None:
            x_centres, y_centres = self.domain.cell_centers()[:2]
            x_mapped, y_mapped = self.ctx.domain_map.to_physical(x_centres[:, None], y_centres[None, :])
            self._mapped_grids = tuple(self.xp.asarray(values, dtype=DTYPE) for values in (x_mapped, y_mapped))
        return self._mapped_grids


class MappedMasks(_MappedGrids, CartesianMasks):
    """2D polar / spherical windows and 1D lines: masks on the drawing-plane centres of the cells
    (geometry.rasterize_mapped)."""

    def _window_mask(self, name):
        """rasterize_mapped tested only on the cells of the object's DomainMap.logical_bounds (an exact upper bound,
        widened by one cell)."""
        shapes = self.ctx.resolved_geometry
        x_mapped, y_mapped = self.coordinate_grids()
        crop = self._logical_crop(name)
        if crop is None:
            return rasterize_mapped(name, shapes, x_mapped, y_mapped, self.xp)
        window, mask = rasterize_mapped(name, shapes, x_mapped[crop], y_mapped[crop], self.xp)
        if not mask.size:
            return window, mask
        shifted = tuple(slice(part.start + offset.start, part.stop + offset.start)
                        for part, offset in zip(window, crop))
        return shifted, mask

    def _logical_crop(self, name):
        """(slice x, slice y) of the cells whose logical centres lie within the logical bounds of the object's
        drawing-plane bounding box, one cell wider; None for a non-finite box."""
        box = bounding_box(name, self.ctx.resolved_geometry)
        if not all(np.isfinite(value) for value in box):
            return None
        x_lo, x_hi, y_lo, y_hi = self.ctx.domain_map.logical_bounds(box)
        crop = []
        for centres, low, high in zip(self.domain.cell_centers()[:2], (x_lo, y_lo), (x_hi, y_hi)):
            start = max(int(np.searchsorted(centres, low, side='left')) - 1, 0)
            stop = min(int(np.searchsorted(centres, high, side='right')) + 1, centres.size)
            crop.append(slice(start, max(start, stop)))
        return tuple(crop)


class BoxMasks(MeshMasks):
    """3D Cartesian tensor grids: box masks per object (geometry.rasterize_box, one 2D raster per cross-section),
    material_index painted per z-slab, x / y expressions as one (nx, ny, 1) layer."""
    paints_by_slabs = True

    def object_mask(self, name):
        return self.object_box_mask(name)

    def object_window_mask(self, name):
        raise ValueError('object_window_mask is 2D only; 3D builds use object_box_mask')

    def object_box_mask(self, name):
        """Cached ((slice x, slice y, slice z), cupy bool mask cropped to the object's bounding-box cells; read-only by
        convention) on Domain3D.host_centres."""
        cached = self._object_masks.get(name)
        if cached is None:
            self._domain_object(name)
            cached = self._object_masks[name] = self._box_mask(name)
        return cached

    def _box_mask(self, name):
        return rasterize_box(name, self.ctx.resolved_geometry, *self.domain.host_centres(), self.xp)

    def region_window(self, objects, is_all):
        """The union of the objects' bounding boxes (the full grid for an 'all' material)."""
        if is_all or not objects:
            return tuple(slice(0, count) for count in self.domain.materials.shape)
        centres = self.domain.host_centres()
        boxes = [box_window(name, self.ctx.resolved_geometry, *centres) for name in objects]
        return tuple(slice(min(box[axis].start for box in boxes), max(box[axis].stop for box in boxes))
                     for axis in range(3))

    def field_coordinates(self):
        """xy centres with a length-1 z axis: the result stays one xy layer."""
        return dict(zip(('x', 'y'), (values[:, :, None] for values in self.plane_coordinates())))

    def dynamic_coordinates(self):
        return dict(zip(('x', 'y', 'z'), self.domain.coordinate_grids(clamp_to_physical=True)))

    def plane_coordinates(self, window_x=slice(None), window_y=slice(None)):
        """Device float32 drawing-plane (x, y) [m] of the cells of a full-grid (window x, window y) box, broadcastable
        to (bx, by): the clamped 1D centres."""
        x_centres, y_centres, _ = self.domain.host_centres()
        return self.xp.asarray(x_centres[window_x])[:, None], self.xp.asarray(y_centres[window_y])[None, :]


class SectorMasks(_MappedGrids, BoxMasks):
    """3D cylindrical (r, phi, z) sectors: box masks on the drawing-plane centres per cross-section, region windows =
    the whole grid (logical boxes need the mapped rasters)."""

    def _box_mask(self, name):
        return rasterize_mapped_box(name, self.ctx.resolved_geometry, *self.coordinate_grids(),
                                    self.domain.host_centres()[2], self.xp)

    def region_window(self, objects, is_all):
        return tuple(slice(0, count) for count in self.domain.materials.shape)

    def dynamic_coordinates(self):
        """Drawing-plane (nx, ny, 1) layers and the axial z."""
        coordinates = dict(zip(('x', 'y'), (values[:, :, None] for values in self.coordinate_grids())))
        coordinates['z'] = self.domain.coordinate_grids(clamp_to_physical=True)[2]
        return coordinates

    def plane_coordinates(self, window_x=slice(None), window_y=slice(None)):
        """The cropped mapped centres."""
        x_mapped, y_mapped = self.coordinate_grids()
        return x_mapped[window_x, window_y], y_mapped[window_x, window_y]


class TriangleMasks(MeshMasks):
    """Triangle meshes: element masks ((slice of all elements,), mask) by the float64 clamped centroids (the mesh's
    region tag rule), boundary edges and point nodes from the mesh tags."""

    def _window_mask(self, name):
        centroids = self.domain.host_centroids(clamp_to_physical=True)
        inside = np.asarray(rasterize(name, self.ctx.resolved_geometry, centroids[:, 0], centroids[:, 1], np),
                            dtype=np.bool_)
        return (slice(0, len(inside)),), self.xp.asarray(inside)

    def boundary_edges(self, selection):
        """Sorted int64 mesh edges of a boundary / line selection (outer faces, domain object outlines, line entities;
        the mesh's edge tags)."""
        tagged = [self.domain.edge_tag(name) for name in selection.outer_faces + selection.objects]
        return np.unique(np.concatenate(tagged).astype(np.int64)) if tagged else np.zeros(0, dtype=np.int64)

    def point_node(self, name):
        """The mesh node of a point object (the mesher puts one there); ValueError for a point outside the window."""
        self.ctx._shape(name, 'point')
        node = self.domain.mesh.point_tags.get(name)
        if node is None:
            raise ValueError(f'point {name!r} lies outside the meshed window: it has no mesh node')
        return int(node)


class TetrahedraMasks(MeshMasks):
    """Tetrahedron meshes: element masks by majority per mesher volume (tetrahedral_meshing.element_membership, faceted
    curved interfaces), x / y / z of the clamped centroids, boundary faces from the window faces and object
    interfaces."""

    def _window_mask(self, name):
        from .tetrahedral_meshing import element_membership
        inside = element_membership(self.domain.mesh, name, self.ctx.resolved_geometry)
        return (slice(0, len(inside)),), self.xp.asarray(inside)

    def field_coordinates(self):
        """The clamped element centroids."""
        return dict(zip(('x', 'y', 'z'), self.domain.coordinate_grids(clamp_to_physical=True)))

    def dynamic_coordinates(self):
        return dict(zip(('x', 'y', 'z'), self.domain.coordinate_grids(clamp_to_physical=True)))

    def boundary_faces(self, selection):
        """Sorted int64 mesh faces of a boundary selection: outer faces = the window faces on that side
        (fem.mesh3d.window_faces), domain objects = their interface faces (mesh.object_faces)."""
        from ..fem.mesh3d import window_faces
        mesh = self.domain.mesh
        tagged = [window_faces(mesh, face) for face in selection.outer_faces]
        for name in selection.objects:
            faces = (mesh.object_faces or {}).get(name)
            if faces is None:
                raise ValueError(f'{name!r} has no interface faces on this tetrahedron mesh (not a domain object)')
            tagged.append(faces)
        return np.unique(np.concatenate(tagged).astype(np.int64)) if tagged else np.zeros(0, dtype=np.int64)


ELEMENT_MASKS = {'triangles': TriangleMasks, 'tetrahedra': TetrahedraMasks}   # grid kind -> MeshMasks class
