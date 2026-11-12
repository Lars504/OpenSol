"""TetrahedraAccess: the FieldAccess of tetrahedron-mesh Session views (igor3d_plan T4b: renderers provide
sample(component, points); host numpy reads on the mesh's sample lattice, quadrature-point derived values)."""
import math

import numpy as np

from ..core.constants import FACES_3D
from ..core.mirror import mirror_faces, touch_factor
from .field_access import ElementAccess, cut_plane, mode_volume_ratio
from .field_data import COORDINATE_AXES, PLANE_AXES, FieldData, LineProfile, VolumeData
from .interface_sampling import interface_selection

TETRAHEDRA_PROFILE_SAMPLES = 100_000   # segment_profile: largest number of samples
# evaluate(): quadrature degree (27 points per element: exact for |E|^2 of quadratic Nedelec elements) and elements per
# renderer sample call (bounds the (points, 20, 3) basis temporaries of a sample)
TETRAHEDRA_DERIVED_DEGREE = 4
TETRAHEDRA_DERIVED_CHUNK = 2048


class TetrahedraAccess(ElementAccess):
    """Tetrahedron meshes: cuts, volumes and profiles sampled by the renderer on the mesh's sample lattice; derived
    values at quadrature points of the window elements."""

    def window_shape(self):
        """The sample lattice (the cells of its 3D reads)."""
        return tuple(len(centres) for centres in self.domain.sample_centres)

    def size_range(self):
        return self.domain.edge_range

    def mesh_arrays(self):
        """{kind, points, tets}."""
        return {'kind': 'tetrahedra', 'points': np.array(self.domain.mesh.points, dtype=np.float64),
                'tets': np.array(self.domain.mesh.tets, dtype=np.int32)}

    # ---- fields ----
    def field(self, physics, component, plane, native):
        return self._field_data(physics, component, self.view.renderer(physics), plane)

    def display_field(self, physics, component, max_pixels, plane):
        return self._field_data(physics, component, self.view.renderer(physics), plane)

    def _plane(self, renderer, component, axis, index) -> np.ndarray:
        """Host float32 (n_u, n_v) samples of the lattice plane normal to axis ('x' | 'y' | 'z') at lattice index
        `index` (NaN outside the elements)."""
        if not hasattr(renderer, 'sample'):
            raise ValueError(f"field '{component}': a renderer on a tetrahedron mesh must provide sample(component, "
                             'points) (igor3d_plan T4b)')
        points, shape = self.domain.sample_plane(COORDINATE_AXES.index(axis), index)   # read-only, kept per plane
        values = np.asarray(renderer.sample(component, points), dtype=np.float64)
        return values.reshape(shape).astype(np.float32)

    def _field_data(self, physics, component, renderer, plane):
        """FieldData of a tetrahedron-mesh field: the sample-lattice cut normal to plane[0] at the lattice centre
        nearest plane[1] [m] (None = mid-height z), uniform pixels over the window plane."""
        axis, position = cut_plane(plane, self.domain.size_z)
        centres = self.domain.sample_centres[COORDINATE_AXES.index(axis)]
        index = int(np.argmin(np.abs(centres - float(position))))
        values = self._plane(renderer, component, axis, index)
        extent = self.domain.physical_extent()   # (x0, x1, y0, y1, z0, z1)
        u_index, v_index = (COORDINATE_AXES.index(name) for name in PLANE_AXES[axis])
        return FieldData(physics, component, values, (extent[2 * u_index], extent[2 * u_index + 1],
                                                      extent[2 * v_index], extent[2 * v_index + 1]),
                         renderer.field_info(component), self.view.time, renderer.time_label(),
                         plane=(axis, float(centres[index])), axes=PLANE_AXES[axis], mesh=self.domain.mesh_name)

    def segment_profile(self, physics, component, start, end, sampling):
        """The field at points spaced half a sample-lattice step (at least 2, at most TETRAHEDRA_PROFILE_SAMPLES), NaN
        outside the window."""
        renderer = self.view.renderer(physics)
        length = math.dist(start, end)
        spacing = min(float(centres[1] - centres[0]) for centres in self.domain.sample_centres)
        count = int(min(TETRAHEDRA_PROFILE_SAMPLES, max(2, math.ceil(2.0 * length / spacing) + 1)))
        distance = np.linspace(0.0, length, count)
        fractions = distance / length if length > 0.0 else np.zeros(count)
        points = np.asarray(start) + fractions[:, None] * (np.asarray(end) - np.asarray(start))
        values = np.asarray(renderer.sample(component, points), dtype=np.float64)
        outside = ~self.domain.mesh.in_window(points)
        values[outside] = np.nan
        points[outside] = np.nan
        return LineProfile(physics, component, distance, values, tuple(points.T.copy()), start, end,
                           renderer.field_info(component), self.view.time, renderer.time_label())

    def volume(self, physics, component):
        """The sample lattice, one z plane at a time."""
        renderer = self.view.renderer(physics)
        window_shape = self.window_shape()
        values = np.empty(window_shape, dtype=np.float32)
        for layer in range(window_shape[2]):
            values[:, :, layer] = self._plane(renderer, component, 'z', layer)
        centres = tuple(np.array(axis_centres) for axis_centres in self.domain.sample_centres)
        return VolumeData(physics, component, values, centres, renderer.field_info(component), self.view.time,
                          renderer.time_label())

    def cut_centers(self, axis):
        return tuple(float(value) for value in self.domain.sample_centres[COORDINATE_AXES.index(axis)])

    # ---- derived values ----
    def evaluate(self, compiled, selection, operation, point, earlier):
        """The expression at degree-TETRAHEDRA_DERIVED_DEGREE quadrature points of the window elements (or the
        selected ones, build-mask rule) through the renderers' sample(component, points), in element chunks: mean /
        integral weighted by w |det J|, max / min over the points, 'at' the value at the point; integrals doubled per
        mirror face the selection touches. Not yet: interface selections, max / min over mirror images, fields of
        physics on other meshes."""
        from ..fem.elements3d import tetrahedron_geometry
        from ..fem.quadrature3d import tetrahedron_rule
        if interface_selection(self.view.spec, selection) is not None:
            raise ValueError('interface selections are not evaluated on tetrahedron meshes yet')
        if operation == 'at':
            return float(self._values(compiled, np.array([self._point_xyz(point)]), earlier)[0])
        elements = np.flatnonzero(self.selected_elements(selection))
        if elements.size == 0:
            raise ValueError(f'selection {list(selection)} covers no element of the physical window')
        faces = mirror_faces(self.view.spec)
        touched = self._touched_faces(elements)
        wants_extreme = operation in ('max', 'min') or (operation == 'mode_volume' and not point)
        if wants_extreme and any(face in touched for face in faces):
            raise ValueError('max / min over the mirror images of a tetrahedron mesh are not evaluated yet (give '
                             'mode_volume a point object)')
        rule = tetrahedron_rule(TETRAHEDRA_DERIVED_DEGREE)
        smallest = operation == 'min'
        extreme, integral, volume = (math.inf if smallest else -math.inf), 0.0, 0.0
        for start in range(0, elements.size, TETRAHEDRA_DERIVED_CHUNK):
            geometry = tetrahedron_geometry(self.domain.mesh, elements[start:start + TETRAHEDRA_DERIVED_CHUNK])
            weights = (np.abs(geometry.determinant)[:, None] * rule.weights[None, :]).ravel()
            values = self._values(compiled, geometry.points(rule.points).reshape(-1, 3), earlier)
            missing = int(np.isnan(values).sum())
            if missing:
                raise ValueError(f'{missing} quadrature points of the tetrahedron mesh evaluate to NaN (point location '
                                 'missed their element, or the expression is undefined there)')
            if wants_extreme:
                extreme = min(extreme, float(values.min())) if smallest else max(extreme, float(values.max()))
            integral += float(np.sum(values * weights))
            volume += float(np.sum(weights))
        if operation in ('max', 'min'):
            return extreme
        if operation == 'mean':
            return integral / volume
        integral *= touch_factor(faces, touched)
        if operation == 'integral':
            return integral
        reference = (float(self._values(compiled, np.array([self._point_xyz(point)]), earlier)[0]) if point
                     else extreme)
        return mode_volume_ratio(integral, reference)

    def _values(self, compiled, points, earlier):
        """float64 (n,) values of the compiled expression at (n, 3) points [m] of this tetrahedron mesh: x / y / z and
        the fields of its physics by renderer.sample (NaN outside the mesh)."""
        view = self.view
        names = compiled.names
        namespace = view._plain_namespace(names, earlier)
        for axis, name in enumerate(COORDINATE_AXES):
            if name in names:
                namespace[name] = points[:, axis]
        for name in names:
            if name in namespace or name not in view._field_names:
                continue
            physics, field_name = view._field_names[name]
            if view._physics_view(physics) is not view:
                raise ValueError(f"field '{name}': derived values on a tetrahedron mesh read the fields of its own "
                                 'physics only (no cross-mesh transfers to tetrahedra)')
            namespace[name] = np.asarray(view.renderer(physics).sample(field_name, points), dtype=np.float64)
        values = np.asarray(compiled.evaluate(namespace, xp=np), dtype=np.float64)
        return np.broadcast_to(values, (len(points),))

    def _point_xyz(self, point):
        """(x, y, z) [m] of point object `point` in a 3D model (the 'at' and mode_volume point)."""
        if not point:
            raise ValueError("operation 'at' over fields needs a point object (DerivedValueSpec.point)")
        return self.view.context.point_position(point)

    def _member_elements(self, names):
        """Majority per mesher volume (tetrahedral_meshing.element_membership, the build-mask rule)."""
        from .tetrahedral_meshing import element_membership
        mesh = self.domain.mesh
        resolved = self.view.context.resolved_geometry
        selected = np.zeros(mesh.n_elements, dtype=bool)
        for name in names:
            selected |= element_membership(mesh, name, resolved)
        return selected

    def _touched_faces(self, elements):
        """Outer faces holding a boundary face of one of the elements (int indices); mirror faces carry no PML shell,
        so their face tags lie on the window plane."""
        mesh = self.domain.mesh
        selected = np.zeros(mesh.n_elements, dtype=bool)
        selected[elements] = True
        touched = set()
        for face in FACES_3D:
            owners = mesh.face_elements[mesh.face_tags[face], 0]
            if owners.size and bool(selected[owners].any()):
                touched.add(face)
        return touched
