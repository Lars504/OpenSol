"""TriangleAccess: the FieldAccess of triangle-mesh Session views (S7.5: renderers provide fem_field; host numpy
rasters, quadrature-sample derived values)."""
import math

import numpy as np

from ..core.backend import to_host
from ..core.constants import DTYPE, FACES_3D
from ..core.mirror import mirror_faces, touch_factor
from .field_access import ElementAccess, mode_volume_ratio
from .field_data import FieldData
from .interface_sampling import interface_selection

TRIANGLE_FIELD_PIXELS = (65_536, 4_000_000)   # triangle-mesh field(): raster pixels, 4 per element within these bounds
KEPT_RASTERS = 4   # raster / window point sets kept per access (oldest dropped)


def _read_only_points(x_centres, y_centres) -> np.ndarray:
    """Read-only float64 (nx ny, 2) points of the (x, y) centre grid, x outer (indexing 'ij')."""
    points = np.empty((len(x_centres), len(y_centres), 2))
    points[..., 0] = np.asarray(x_centres, dtype=np.float64)[:, None]
    points[..., 1] = np.asarray(y_centres, dtype=np.float64)[None, :]
    points = points.reshape(-1, 2)
    points.flags.writeable = False
    return points


class TriangleAccess(ElementAccess):
    """Triangle meshes: fields as FE values at raster pixel centres plus the display triangulation; derived values
    from degree-4 quadrature samples of the window elements."""

    def __init__(self, view):
        super().__init__(view)
        self._display_triangulations = {}   # (id(mesh), order) -> read-only (nodes, triangles) shared by every FieldData
        self._raster_points = {}   # (n_u, n_v) -> read-only (n_u n_v, 2) pixel centres of field() / display_field()
        self._window_points = {}   # (x centres, y centres bytes) -> read-only window centres of values_on_window

    def window_shape(self):
        return (self.window_element_count(),)

    def size_range(self):
        mesh = self.domain.mesh
        lengths = np.linalg.norm(mesh.points[mesh.edges[:, 1]] - mesh.points[mesh.edges[:, 0]], axis=1)
        return float(lengths.min()), float(lengths.max())

    def mesh_arrays(self):
        """{kind, points, triangles, pml_faces}."""
        mesh = self.domain.mesh
        return {'kind': 'triangles', 'points': np.array(mesh.points, dtype=np.float64),
                'triangles': np.array(mesh.triangles, dtype=np.int32),
                'pml_faces': np.array(mesh.pml_faces, dtype=np.uint8)}

    # ---- fields ----
    def field(self, physics, component, plane, native):
        """A raster of 4 pixels per element (within TRIANGLE_FIELD_PIXELS) plus the triangulation."""
        low, high = TRIANGLE_FIELD_PIXELS
        return self._field_data(physics, component, self.view.renderer(physics),
                                int(min(high, max(low, 4 * self.domain.n_elements))), plane)

    def display_field(self, physics, component, max_pixels, plane):
        return self._field_data(physics, component, self.view.renderer(physics), int(max_pixels), plane)

    def fem_field(self, renderer, component):
        """(Lagrange DofMap, host float64 coefficients) of a field of a physics on this triangle mesh."""
        if not hasattr(renderer, 'fem_field'):
            raise ValueError(f"field '{component}': a renderer on a triangle mesh must provide fem_field(component) "
                             '-> (Lagrange DofMap, coefficients) (mesh plan T3 / T4)')
        dofs, coefficients = renderer.fem_field(component)
        if dofs.family != 'lagrange':
            raise ValueError(f"field '{component}': fem_field must give a Lagrange (P0 / P1 / P2) map, got "
                             f'{dofs.family}')
        coefficients = np.asarray(to_host(coefficients))
        if np.iscomplexobj(coefficients):
            raise ValueError(f"field '{component}': fem_field gave complex coefficients; display fields are real")
        return dofs, coefficients.astype(np.float64, copy=False)

    def _field_data(self, physics, component, renderer, max_pixels, plane):
        """FieldData of a triangle-mesh field: uniform pixels over the window (at most max_pixels), each the FE value
        at its centre (NaN outside the field's elements), plus the display triangulation with node values (P1 / P2)."""
        if plane is not None:
            raise ValueError(f'plane={plane!r} needs a 3D model; triangle meshes are 2D')
        if max_pixels < 1:
            raise ValueError(f'max_pixels must be >= 1, got {max_pixels}')
        from ..fem.sampling import display_triangulation, evaluate as fem_evaluate, node_values
        dofs, coefficients = self.fem_field(renderer, component)
        x_min, x_max, y_min, y_max = self.domain.physical_extent()
        width, height = x_max - x_min, y_max - y_min
        pixel = math.sqrt(width * height / max_pixels)
        n_u, n_v = max(1, int(width / pixel)), max(1, int(height / pixel))
        if n_u * n_v > max_pixels:   # float rounding guard
            n_u = max(1, max_pixels // n_v)
        points = self._raster_points.get((n_u, n_v))
        if points is None:   # read-only and kept: the domain matches it by identity (no meshgrid, no hash per read)
            x_centres = x_min + (np.arange(n_u) + 0.5) * (width / n_u)
            y_centres = y_min + (np.arange(n_v) + 0.5) * (height / n_v)
            points = _read_only_points(x_centres, y_centres)
            self._raster_points[(n_u, n_v)] = points
            while len(self._raster_points) > KEPT_RASTERS:
                self._raster_points.pop(next(iter(self._raster_points)))
        values = fem_evaluate(dofs, coefficients, points, location=self._located(dofs, points)).reshape(n_u, n_v)
        triangulation = None
        if dofs.order > 0:   # node values = the coefficients: locating the nodes of a graded mesh took seconds
            key = (id(dofs.mesh), dofs.order)
            if key not in self._display_triangulations:   # the mesh lives as long as this access (domain)
                nodes, triangles = display_triangulation(dofs.mesh, dofs.order)
                nodes, triangles = np.array(nodes, dtype=np.float64), np.array(triangles, dtype=np.int32)
                nodes.setflags(write=False)
                triangles.setflags(write=False)
                self._display_triangulations[key] = (nodes, triangles)
            triangulation = (*self._display_triangulations[key],
                             np.asarray(node_values(dofs, coefficients), dtype=np.float32))
        return FieldData(physics, component, np.ascontiguousarray(values, dtype=np.float32),
                         (x_min, x_max, y_min, y_max), renderer.field_info(component), self.view.time,
                         renderer.time_label(), mesh=self.domain.mesh_name, triangulation=triangulation)

    # ---- derived values ----
    def evaluate(self, compiled, selection, operation, point, earlier):
        """Degree-4 quadrature samples of the window elements (or the selected ones, by centroid as the build masks):
        mean / integral weighted by w det J (axisymmetric: x ring_angle r), max / min over the samples, 'at' the value
        at the point; integrals doubled per mirror face the selection touches. Not yet: interface selections, max /
        min over mirror images."""
        view = self.view
        if interface_selection(view.spec, selection) is not None:
            raise ValueError('interface selections are not evaluated on triangle meshes yet')
        if operation == 'at':
            return float(self._values(compiled, np.array([self._point_xy(point)]), earlier)[0])
        elements = self.selected_elements(selection)
        if not elements.any():
            raise ValueError(f'selection {list(selection)} covers no element of the physical window')
        samples, located = self.domain.quadrature_located(4, elements)
        values = self._values(compiled, np.column_stack([samples.x, samples.y]), earlier, located)
        faces = mirror_faces(view.spec)
        touched = self._touched_faces(elements)
        wants_extreme = operation in ('max', 'min') or (operation == 'mode_volume' and not point)
        if wants_extreme and any(face in touched for face in faces):
            raise ValueError('max / min over the mirror images of a triangle mesh are not evaluated yet')
        if operation in ('max', 'min'):
            return float(values.max() if operation == 'max' else values.min())
        weights = samples.weight
        if view._has_axisymmetric_solver(compiled.names):
            weights = weights * view._ring_angle(compiled.names) * samples.x   # (r, z): x = r from the window's x_lo edge
        integral = float(np.sum(values * weights))
        if operation == 'mean':
            return integral / float(np.sum(weights))
        integral *= touch_factor(faces, touched)
        if operation == 'integral':
            return integral
        reference = (float(self._values(compiled, np.array([self._point_xy(point)]), earlier)[0]) if point
                     else float(values.max()))
        return mode_volume_ratio(integral, reference)

    def _values(self, compiled, points, earlier, location=None):
        """float64 (n,) values of the compiled expression at (n, 2) points [m] of this triangle mesh (location: their
        fem LocatedPoints, reused for this mesh's fields)."""
        view = self.view
        names = compiled.names
        namespace = view._plain_namespace(names, earlier)
        for axis, name in enumerate(('x', 'y')):
            if name in names:
                namespace[name] = points[:, axis]
        states = {}
        for name in names:
            if name in namespace or name not in view._field_names:
                continue
            physics, field_name = view._field_names[name]
            source = view._physics_view(physics)
            namespace[name] = source.access.values_at(physics, field_name, points,
                                                      location if source is view else None, states)
        values = np.asarray(compiled.evaluate(namespace, xp=np), dtype=np.float64)
        return np.broadcast_to(values, (len(points),))

    def values_at(self, physics, field_name, points, location, states):
        """FE evaluation (at the quadrature LocatedPoints when given: geometry order 2 fields, and order 1 fields when no
        sample lies in a curved element, where the P2 map is the affine one; else located anew)."""
        from ..fem.elements import curved_elements
        from ..fem.sampling import evaluate as fem_evaluate
        dofs, coefficients = self.fem_field(self.view.renderer(physics), field_name)
        mesh = self.domain.mesh
        if location is not None and (dofs.geometry_order == 2 or (   # quadrature_located's order 2 map
                dofs.mesh is mesh and not curved_elements(mesh)[location.location.element].any())):
            return fem_evaluate(dofs, coefficients, location=location)
        return fem_evaluate(dofs, coefficients, points, location=self._located(dofs, points))

    def values_on_window(self, target, physics, field_name, states):
        """FE evaluation at the window centres of a Cartesian 2D tensor target."""
        from ..fem.sampling import evaluate as fem_evaluate
        x_centres, y_centres = target.cartesian_centres(f"field '{physics}.{field_name}' on a triangle mesh")
        dofs, coefficients = self.fem_field(self.view.renderer(physics), field_name)
        x_centres = np.ascontiguousarray(x_centres, dtype=np.float64)
        y_centres = np.ascontiguousarray(y_centres, dtype=np.float64)
        key = (x_centres.tobytes(), y_centres.tobytes())   # the 1D centres: cheap to key, unlike the raster
        points = self._window_points.get(key)
        if points is None:
            points = _read_only_points(x_centres, y_centres)
            self._window_points[key] = points
            while len(self._window_points) > KEPT_RASTERS:
                self._window_points.pop(next(iter(self._window_points)))
        values = fem_evaluate(dofs, coefficients, points, location=self._located(dofs, points))
        return target.domain.xp.asarray(values.reshape(len(x_centres), len(y_centres)), dtype=DTYPE)

    def _located(self, dofs, points):
        """The domain's remembered LocatedPoints of points (basis shared across fields and reads); None (evaluate
        locates anew) for a dof map on another mesh."""
        return self.domain.located(points, dofs.geometry_order) if dofs.mesh is self.domain.mesh else None

    def _point_xy(self, point):
        """(x, y) [m] of point object `point` (the 'at' and mode_volume point)."""
        if not point:
            raise ValueError("operation 'at' over fields needs a point object (DerivedValueSpec.point)")
        return self.view.context.point(point)

    def _member_elements(self, names):
        """By centroid (triangle_meshing.element_masks, the build-mask rule)."""
        from .triangle_meshing import element_masks
        mesh = self.domain.mesh
        selected = np.zeros(mesh.n_elements, dtype=bool)
        for mask in element_masks(self.view.context.resolved_geometry, names, mesh.window, mesh.centroids()).values():
            selected |= mask
        return selected

    def _touched_faces(self, elements):
        """Outer faces with a tagged boundary edge of a selected element (mirror faces carry no PML ring, so their
        edges are on the window edge)."""
        mesh = self.domain.mesh
        touched = set()
        for face in FACES_3D[:4]:
            owners = mesh.edge_elements[self.domain.edge_tag(face)].ravel()
            owners = owners[owners >= 0]
            if owners.size and bool(elements[owners].any()):
                touched.add(face)
        return touched
