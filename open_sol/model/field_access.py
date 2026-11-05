"""FieldAccess: the grid-kind part of the Session reads of one mesh view (fields, cuts, profiles, volumes, derived-value
reductions), implemented per kind by field_access_tensor, field_access_triangles and field_access_tetrahedra."""
import math

import numpy as np

from .field_data import PLANE_AXES
from .geometry import GEOMETRY_KINDS, expand_entities
from .spec import OUTER_FACES_3D, RESERVED_ENTITY_NAMES

DISPLAY_MAX_PIXELS = 250_000   # Session.display_field default (also the canvas fallback)


class FieldAccess:
    """Grid-kind part of the reads of one Session mesh view; `view` (that Session or mesh view) gives renderers,
    solvers, spec, context and the mesh-free namespace. Session checks the arguments and picks the view, then calls
    here; a read the kind lacks raises ValueError."""

    def __init__(self, view):
        self.view = view
        self.domain = view.domain
        self.domain_map = view.domain_map

    def _unsupported(self, what):
        raise ValueError(f'{what} is not available on a {self.domain.grid_kind} mesh')

    # ---- description ----
    def window_shape(self) -> tuple:
        """Window cells (wx, wy(, wz)); a triangle mesh: (window elements,); a tetrahedron mesh: its sample lattice."""
        raise NotImplementedError

    def size_range(self) -> tuple:
        """(dx_min, dx_max) [m]: physical cell lengths, on element meshes the element edge lengths."""
        raise NotImplementedError

    def mesh_record(self) -> dict:
        """{kind, shape (cells incl. padding; None on element meshes), elements (element meshes, else None), cells,
        window_cells} of Session.mesh_records."""
        raise NotImplementedError

    def mesh_arrays(self) -> dict:
        """Host arrays of this mesh for meshes.npz (Session.mesh_arrays)."""
        raise NotImplementedError

    # ---- fields ----
    def field(self, physics, component, plane, native):
        """FieldData on the native cells (Session.field)."""
        raise NotImplementedError

    def display_field(self, physics, component, max_pixels, plane):
        """FieldData on a uniform display grid of at most max_pixels pixels (Session.display_field)."""
        raise NotImplementedError

    def canvas_field(self, physics, component, plane, max_cells):
        """FieldData for the GUI canvas: the display field (TensorAccess: native graded cells within max_cells)."""
        return self.display_field(physics, component, DISPLAY_MAX_PIXELS, plane)

    def segment_profile(self, physics, component, start, end, sampling):
        """LineProfile along a 3D segment (Session.segment_profile checked the arguments)."""
        self._unsupported('segment_profile')

    def volume(self, physics, component):
        """VolumeData of a 3D field (Session.volume checked dimensions and size)."""
        self._unsupported('volume')

    def cut_centers(self, axis) -> tuple:
        """Positions [m] along axis of the cuts field() can show; () where it cuts along no such axis."""
        return ()

    # ---- derived values ----
    def evaluate(self, compiled, selection, operation, point, earlier) -> float:
        """Reduce a compiled expression over the window or a selection (Session.evaluate checked the operation)."""
        raise NotImplementedError

    def values_at(self, physics, field_name, points, location, states):
        """float64 values of a field of a physics on this mesh at (n, 2) points [m] (triangle-mesh evaluations)."""
        self._unsupported(f"reading field '{physics}.{field_name}' at triangle-mesh points")

    def values_on_window(self, target, physics, field_name, states):
        """A field of a physics on this mesh on the tensor window of TensorAccess `target` (derived values, D3)."""
        self._unsupported(f"reading field '{physics}.{field_name}' on another mesh")

    def selected_objects(self, selection):
        """Domain object names of a derived-value selection (reserved names dropped); None = the whole window (empty
        selection or one containing 'all'). ValueError for outer faces or names that are no built domain object."""
        entities = list(selection or ())
        if not entities:
            return None
        expanded = expand_entities(self.view.spec, entities)
        if 'all' in expanded:
            return None
        faces = [name for name in expanded if name in OUTER_FACES_3D]
        if faces:
            raise ValueError(f'derived values need domain entities; {faces} are outer faces')
        resolved = self.view.context.resolved_geometry
        names = []
        for name in expanded:
            if name in RESERVED_ENTITY_NAMES:
                continue
            shape = resolved.get(name)
            if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
                raise ValueError(f'{name!r} is not a built domain object')
            names.append(name)
        return names


class ElementAccess(FieldAccess):
    """Triangle / tetrahedron meshes: element counts and element selections (the kind's build-mask rule)."""

    def __init__(self, view):
        super().__init__(view)
        self._memberships = {}   # tuple(object names) -> read-only bool per element (every series point reuses it)

    def window_element_count(self) -> int:
        return int(np.count_nonzero(self.domain.window_elements()))

    def mesh_record(self):
        shape = self.domain.materials.shape
        return {'kind': self.domain.grid_kind, 'shape': None, 'elements': int(shape[0]),
                'cells': int(math.prod(shape)), 'window_cells': self.window_element_count()}

    def selected_elements(self, selection):
        """Bool per element: window elements in the selection's domain objects (_member_elements); every window
        element without a selection."""
        window = self.domain.window_elements()
        names = self.selected_objects(selection)
        if names is None:
            return window
        key = tuple(names)
        if key not in self._memberships:
            members = self._member_elements(names)
            members.setflags(write=False)
            self._memberships[key] = members
        return self._memberships[key] & window

    def _member_elements(self, names):
        """Bool per element: inside one of the named domain objects (the kind's build-mask rule)."""
        raise NotImplementedError


def cut_plane(plane, size_z):
    """The cut of a 3D read: None -> mid-height ('z', size_z / 2); ValueError unless (axis 'x' | 'y' | 'z',
    position in m)."""
    if plane is None:
        return 'z', 0.5 * size_z
    if not (isinstance(plane, (tuple, list)) and len(plane) == 2 and plane[0] in PLANE_AXES):
        raise ValueError(f"plane must be (axis 'x' | 'y' | 'z', position in m), got {plane!r}")
    return plane


def mode_volume_ratio(integral, reference) -> float:
    """integral / reference of operation 'mode_volume'; ValueError unless the reference (maximum or value at the
    point) is > 0."""
    if not reference > 0.0:
        raise ValueError(f'mode_volume: the reference value {reference:g} (maximum or value at the point) must be > 0; '
                         'give an energy density such as eps_r * e_intensity')
    return integral / reference
