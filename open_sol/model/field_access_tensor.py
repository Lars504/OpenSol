"""TensorAccess (Cartesian 2D / 3D windows), MappedAccess (polar / spherical windows, cylindrical sectors) and
LineAccess (1D models): the FieldAccess of tensor-grid Session views."""
import itertools
import math

import numpy as np

from ..core.backend import to_host
from ..core.constants import DTYPE, FACES_3D
from ..core.grid_transfer import GridTransfer
from ..core.metric import window_metric, window_metric_3d
from ..core.mirror import field_parity, image_sign, mirror_faces, touch_factor
from .analytic import segment_cells, segment_trilinear_weights
from .field_access import DISPLAY_MAX_PIXELS, FieldAccess, cut_plane, mode_volume_ratio
from .field_data import COORDINATE_AXES, PLANE_AXES, FieldData, LineProfile, VolumeData, host_float32
from .geometry import point_z, rasterize_box, rasterize_mapped, rasterize_mapped_box, rasterize_window
from .interface_sampling import interface_selection, interface_stencil, window_frame
from .schema import INTERFACE_OPERATIONS, INTERFACE_VARIABLES

# polar / spherical field(): containing-cell pixels, 4 per window cell within these bounds
MAPPED_FIELD_PIXELS = (1_000_000, 16_000_000)


class TensorAccess(FieldAccess):
    """Cartesian tensor windows (2D, 3D): window-cell fields with their centres, z / side cuts, segment profiles,
    volumes and the area / volume-weighted reductions of evaluate (3D one z plane at a time, SCL-26)."""

    def __init__(self, view):
        super().__init__(view)
        self._display_indices = {}   # display-grid key -> cached window-cell indices (device; 'mapped': host)

    # ---- description ----
    def window_shape(self):
        return tuple(window.stop - window.start for window in self.domain.physical_window())

    def size_range(self):
        widths = self._physical_widths(self.domain.cell_widths())   # 3D: (dx, dy, dz) cells
        return float(min(axis.min() for axis in widths)), float(max(axis.max() for axis in widths))

    def _physical_widths(self, widths):
        """The per-axis cell lengths [m] among the Domain cell widths (Cartesian: all of them)."""
        return widths

    def mesh_record(self):
        shape = self.domain.materials.shape
        return {'kind': self.domain.grid_kind, 'shape': list(shape), 'elements': None, 'cells': int(math.prod(shape)),
                'window_cells': int(math.prod(self.window_shape()))}

    def mesh_arrays(self):
        """{kind, x_edges, y_edges(, z_edges)}: the window cell edges in Domain coordinates (logical r - radius_min /
        angle on polar windows)."""
        return {'kind': 'tensor', **{f'{axis}_edges': np.array(axis_edges, dtype=np.float64)
                                     for axis, axis_edges in zip(COORDINATE_AXES, window_edges(self.domain))}}

    # ---- fields ----
    def field(self, physics, component, plane, native):
        renderer = self.view.renderer(physics)
        device_values, plane, axes = self._plane_field(renderer, component, plane)
        return self._native_field(physics, component, renderer, device_values, plane, axes, native)

    def _native_field(self, physics, component, renderer, device_values, plane, axes, native):
        """FieldData of the window values on their cells with the cell centres (native changes nothing here)."""
        windows, centres = self.domain.physical_window(), self.domain.cell_centers()
        u_index, v_index = (COORDINATE_AXES.index(axis) for axis in axes)
        return FieldData(physics, component, host_float32(device_values), self._plane_extent(plane),
                         renderer.field_info(component), self.view.time, renderer.time_label(),
                         x_centers=np.array(centres[u_index][windows[u_index]]),
                         y_centers=np.array(centres[v_index][windows[v_index]]), plane=plane, axes=axes,
                         mesh=self.domain.mesh_name)

    def display_field(self, physics, component, max_pixels, plane):
        renderer = self.view.renderer(physics)
        device_values, plane, axes = self._plane_field(renderer, component, plane)
        return self._display_data(physics, component, renderer, device_values, int(max_pixels), plane, axes)

    def _display_data(self, physics, component, renderer, device_values, max_pixels, plane, axes):
        """FieldData on a uniform display grid: the window values as they are when uniform within max_pixels, else
        the nearest-cell resampling (native_shape set)."""
        n_window_u, n_window_v = device_values.shape
        native_shape = None
        if not self.domain.non_uniform and n_window_u * n_window_v <= max_pixels:
            values = host_float32(device_values)
        else:
            u_index, v_index = self._display_index_arrays(max_pixels, axes)
            values = host_float32(device_values[u_index[:, None], v_index[None, :]])
            native_shape = (int(n_window_u), int(n_window_v))
        return FieldData(physics, component, values, self._plane_extent(plane), renderer.field_info(component),
                         self.view.time, renderer.time_label(), native_shape=native_shape, plane=plane, axes=axes,
                         mesh=self.domain.mesh_name)

    def canvas_field(self, physics, component, plane, max_cells):
        """Graded Cartesian planes of at most max_cells cells: the native cells (the canvas resamples them to the
        zoomed range), else the display field."""
        if self.domain.non_uniform:
            field_data = self.field(physics, component, plane, False)
            if field_data.values.size <= max_cells:
                return field_data
        return self.display_field(physics, component, DISPLAY_MAX_PIXELS, plane)

    def segment_profile(self, physics, component, start, end, sampling):
        """Containing window cell of each sample (segment_cells) or the trilinear value between the 8 surrounding
        cell centres; z planes via device_plane when the renderer has it (no volume temporaries)."""
        renderer = self.view.renderer(physics)
        windows, all_edges, all_centres = (self.domain.physical_window(), self.domain.cell_edges(),
                                           self.domain.cell_centers())
        edges_in_window = [edges[window.start:window.stop + 1] for edges, window in zip(all_edges, windows)]
        distance, inside, cells = segment_cells(start, end, edges_in_window)
        values = np.full(distance.size, np.nan)
        centres = tuple(np.full(distance.size, np.nan) for _ in range(3))
        if inside.any():
            plane = self._plane_source(renderer, component)
            if sampling == 'trilinear':
                window_centres = [centres_all[window] for centres_all, window in zip(all_centres, windows)]
                values[inside] = self._trilinear_samples(plane, segment_trilinear_weights(
                    start, end, distance[inside], window_centres))
                for axis_centres, low, high in zip(centres, start, end):
                    axis_centres[inside] = low + distance[inside] / math.dist(start, end) * (high - low)
            else:
                values[inside] = self._nearest_samples(plane, [axis_cells[inside] for axis_cells in cells])
                for axis_centres, centres_all, window, axis_cells in zip(centres, all_centres, windows, cells):
                    axis_centres[inside] = centres_all[window.start + axis_cells[inside]]
        return LineProfile(physics, component, distance, values, centres, start, end, renderer.field_info(component),
                           self.view.time, renderer.time_label())

    def _nearest_samples(self, plane, cells):
        """float64 values of the window cells (x, y, z index arrays), one device plane per z layer."""
        x_cells, y_cells, z_cells = cells
        gathered = np.empty(x_cells.size, dtype=np.float64)
        xp = self.domain.xp
        for layer in np.unique(z_cells):
            in_layer = np.flatnonzero(z_cells == layer)
            layer_values = plane('z', int(layer))[xp.asarray(x_cells[in_layer]), xp.asarray(y_cells[in_layer])]
            gathered[in_layer] = to_host(layer_values)
        return gathered

    def _trilinear_samples(self, plane, weights):
        """float64 trilinear values of segment_trilinear_weights' samples: per z layer one device plane and one
        gather of every corner value it holds."""
        (x_low, x_high, x_weight), (y_low, y_high, y_weight), (z_low, z_high, z_weight) = weights
        n_samples = x_low.size
        rows = np.arange(n_samples)
        xp = self.domain.xp
        total = np.zeros(n_samples, dtype=np.float64)
        corners = [(x_index, y_index, x_part * y_part)
                   for x_index, x_part in ((x_low, 1.0 - x_weight), (x_high, x_weight))
                   for y_index, y_part in ((y_low, 1.0 - y_weight), (y_high, y_weight))]
        for layer in np.unique(np.concatenate([z_low, z_high])):
            picks = []   # (sample rows, x cells, y cells, weights) of the corners in this layer
            for z_index, z_part in ((z_low, 1.0 - z_weight), (z_high, z_weight)):
                in_layer = z_index == layer
                for x_index, y_index, xy_part in corners:
                    picks.append((rows[in_layer], x_index[in_layer], y_index[in_layer], (xy_part * z_part)[in_layer]))
            sample_rows, x_cells, y_cells, corner_weights = (np.concatenate(parts) for parts in zip(*picks))
            corner_values = to_host(plane('z', int(layer))[xp.asarray(x_cells), xp.asarray(y_cells)])
            total += np.bincount(sample_rows, corner_weights * np.asarray(corner_values, dtype=np.float64),
                                 minlength=total.size)
        return total

    def volume(self, physics, component):
        """The window cells fetched one z plane at a time (device_plane when the renderer has it)."""
        renderer = self.view.renderer(physics)
        plane = self._plane_source(renderer, component)
        window_shape = self.window_shape()
        values = np.empty(window_shape, dtype=np.float32)
        for layer in range(window_shape[2]):
            values[:, :, layer] = to_host(plane('z', layer))
        return VolumeData(physics, component, values, self._volume_centres(), renderer.field_info(component),
                          self.view.time, renderer.time_label())

    def _volume_centres(self):
        return tuple(np.array(axis_centres[window]) for axis_centres, window
                     in zip(self.domain.cell_centers(), self.domain.physical_window()))

    def cut_centers(self, axis):
        axis_index = COORDINATE_AXES.index(axis)
        window = self.domain.physical_window()[axis_index]
        return tuple(float(value) for value in self.domain.cell_centers()[axis_index][window])

    def _window_field(self, renderer, component):
        values = renderer.device_field(component)
        if values.shape != self.window_shape():
            raise ValueError(f"renderer field '{component}' has shape {values.shape}, expected the physical "
                             f"window {self.window_shape()}")
        return values

    def _plane_field(self, renderer, component, plane):
        """(device (wu, wv) values, plane, axes) of a field: 2D the window field (plane must be None); 3D the cut
        normal to plane[0] at the window cell containing plane[1] [m] (None = mid-height z) via the renderer's
        optional device_plane(component, axis, index_window), else a slice of device_field(component)."""
        if self.domain.dimensions != 3:
            if plane is not None:
                raise ValueError(f'plane={plane!r} needs a 3D model (domain size_z); this model is 2D')
            return self._window_field(renderer, component), None, PLANE_AXES['z']
        plane = cut_plane(plane, self.domain.size_z)
        axis = plane[0]
        if self.domain_map.mapped and axis != 'z':
            raise ValueError(f"plane={plane!r}: a cylindrical (r, phi, z) sector is cut at constant z only "
                             "(plane=('z', height)); Session.volume gives every (r, phi, z) cell")
        axis_index = COORDINATE_AXES.index(axis)
        containing = (self.domain.pos_to_cell_x, self.domain.pos_to_cell_y, self.domain.pos_to_cell_z)[axis_index]
        cell = containing(float(plane[1]))
        index_window = cell - self.domain.physical_window()[axis_index].start
        centre = float(self.domain.cell_centers()[axis_index][cell])
        return self._plane_source(renderer, component)(axis, index_window), (axis, centre), PLANE_AXES[axis]

    def _plane_source(self, renderer, component):
        """plane(axis, index_window) -> device window plane of a 3D field (shape-checked): the renderer's optional
        device_plane, else slices of one device_field(component) volume (kept for the block of a
        Session.shared_plane_reads: every cut of one snapshot record from one volume)."""
        root = self.view._parent or self.view
        shared = root._plane_reads
        key = (self.domain.mesh_name, id(renderer), component)
        if shared is not None and key in shared:
            return shared[key][1]
        plane = self._new_plane_source(renderer, component)
        if shared is not None:
            shared[key] = (renderer, plane)   # the renderer held: its id stays unique while cached
        return plane

    def _new_plane_source(self, renderer, component):
        window_shape = self.window_shape()
        if hasattr(renderer, 'device_plane'):
            def plane(axis, index_window):
                values = renderer.device_plane(component, axis, index_window)
                expected = tuple(count for index, count in enumerate(window_shape)
                                 if index != COORDINATE_AXES.index(axis))
                if values.shape != expected:
                    raise ValueError(f"renderer plane of '{component}' has shape {values.shape}, expected {expected}")
                return values
            return plane
        volume = self._window_field(renderer, component)

        def plane(axis, index_window):
            cut = [slice(None)] * 3
            cut[COORDINATE_AXES.index(axis)] = index_window
            return volume[tuple(cut)]
        return plane

    def _plane_extent(self, plane):
        """FieldData extent (u0, u1, v0, v1): 2D physical_extent, 3D Domain3D.plane_extent of the cut's axis."""
        if self.domain.dimensions == 3:
            return tuple(self.domain.plane_extent(plane[0]))
        return tuple(self.domain.physical_extent())

    def _display_index_arrays(self, max_pixels, axes):
        """Cached device (u, v) window-cell indices of a uniform display grid of at most max_pixels pixels over the
        window axes `axes`."""
        key = (max_pixels, tuple(axes))
        indices = self._display_indices.get(key)
        if indices is not None:
            return indices
        if max_pixels < 1:
            raise ValueError(f'max_pixels must be >= 1, got {max_pixels}')
        windows, all_edges = self.domain.physical_window(), self.domain.cell_edges()
        u_edges, v_edges = (all_edges[index][windows[index].start:windows[index].stop + 1]
                            for index in (COORDINATE_AXES.index(axis) for axis in axes))
        width, height = float(u_edges[-1] - u_edges[0]), float(v_edges[-1] - v_edges[0])
        finest = float(min(np.diff(u_edges).min(), np.diff(v_edges).min()))
        pixel = max(finest, math.sqrt(width * height / max_pixels))
        n_pixels_u = max(1, int(width / pixel))
        n_pixels_v = max(1, int(height / pixel))
        if n_pixels_u * n_pixels_v > max_pixels:   # float rounding guard
            n_pixels_u = max(1, max_pixels // n_pixels_v)
        indices = (_nearest_cells(u_edges, n_pixels_u, self.domain.xp),
                   _nearest_cells(v_edges, n_pixels_v, self.domain.xp))
        self._display_indices[key] = indices
        return indices

    # ---- derived values ----
    def evaluate(self, compiled, selection, operation, point, earlier):
        """Area-weighted reductions over the window cells (axisymmetric solvers: ring volumes 2 pi r_c dx dy; 3D one
        z plane at a time); an 'interface' selection reduces one-sided outline samples (_evaluate_interface)."""
        interface = interface_selection(self.view.spec, selection)
        if interface is not None:
            return self._evaluate_interface(compiled, interface, operation, earlier)
        window_shape = self.window_shape()
        xp = self.domain.xp
        namespace = self.evaluation_namespace(compiled.names, earlier)
        by_planes = self._reduces_by_planes()
        if operation == 'at':
            if by_planes:
                return self._plane_value_at_point(compiled, namespace, point)
            return self._value_at_point(xp.broadcast_to(xp.asarray(compiled.evaluate(namespace, xp=xp)),
                                                        window_shape), point)
        mask = self._selection_mask(selection, window_shape)
        if mask is not None and not bool(mask.any()):
            raise ValueError(f'selection {list(selection)} covers no cell of the physical window')
        if by_planes:
            return self._evaluate_planes(compiled, namespace, mask, operation, point)
        values = xp.broadcast_to(xp.asarray(compiled.evaluate(namespace, xp=xp)), window_shape)
        if operation in ('max', 'min'):
            return self._window_extreme(compiled, namespace, values, mask, operation)
        cell_area = self._cell_measures(window_shape, compiled.names)
        weighted = values * cell_area
        if mask is not None:
            weighted = xp.where(mask, weighted, 0.0)
            cell_area = xp.where(mask, cell_area, 0.0)
        integral = float(weighted.sum())
        if operation == 'mean':
            return integral / float(cell_area.sum())
        integral *= touch_factor(mirror_faces(self.view.spec), self._touched_faces(mask))
        if operation == 'integral':
            return integral
        if point:   # mode_volume, the field-at-point definition
            reference = self._value_at_point(values, point)
        else:
            reference = self._window_extreme(compiled, namespace, values, mask, 'max')
        return mode_volume_ratio(integral, reference)

    def _reduces_by_planes(self):
        """True when evaluate reduces one z plane at a time (Cartesian 3D)."""
        return self.domain.dimensions == 3

    def _cell_measures(self, window_shape, names=()):
        """Device cell areas (wx, wy) of a 2D window: the outer product of the widths, times the ring weights when an
        axisymmetric solver provides the fields `names` read (Session._axisymmetric_solvers)."""
        xp = self.domain.xp
        cell_area = None
        for axis, (window, widths) in enumerate(zip(self.domain.physical_window(), self.domain.cell_widths())):
            axis_widths = xp.asarray(widths[window]).reshape([-1 if other == axis else 1
                                                              for other in range(len(window_shape))])
            cell_area = axis_widths if cell_area is None else cell_area * axis_widths
        ring_weights = self._ring_weights(window_shape, names)
        return cell_area if ring_weights is None else cell_area * ring_weights

    def _ring_weights(self, window_shape, names=()):
        """ring_angle r_c (wx, 1) device float64 of a 2D model with an axisymmetric solver (geometry 'axisymmetric':
        x = r from the window's x_lo edge, as the (r, z) core.metric.WindowMetric), else None."""
        if len(window_shape) != 2 or not self.view._has_axisymmetric_solver(names):
            return None
        window_x = self.domain.physical_window()[0]
        x_edges = self.domain.cell_edges()[0]
        radii = 0.5 * (x_edges[window_x.start:window_x.stop] + x_edges[window_x.start + 1:window_x.stop + 1]) \
            - x_edges[window_x.start]
        return self.domain.xp.asarray(self.view._ring_angle(names) * radii)[:, None]

    def _window_extreme(self, compiled, namespace, values, mask, operation) -> float:
        """'max' / 'min' of window-shaped `values` (the compiled expression on namespace) under mask (None: all) and of
        the expression over the mirror images the selection reaches (_image_namespaces): the unfolded extreme."""
        xp = self.domain.xp
        fill = -math.inf if operation == 'max' else math.inf
        extremes = []
        images = self._image_namespaces(compiled.names, namespace, mask)
        for image_values in [values] + [xp.broadcast_to(xp.asarray(compiled.evaluate(image, xp=xp)), values.shape)
                                        for image in images]:
            masked = image_values if mask is None else xp.where(mask, image_values, fill)
            extremes.append(masked.max() if operation == 'max' else masked.min())
        stacked = xp.stack([xp.asarray(extreme) for extreme in extremes])   # one host sync
        return float(stacked.max() if operation == 'max' else stacked.min())

    def _image_namespaces(self, names, namespace, mask) -> list:
        """Namespaces of the mirror images of the window (or of a selection reaching mirror planes; symmetry 2.5) for
        extremes over the unfolded model: one per non-empty set of the reached mirror faces, each field times its
        image sign (core.mirror.image_sign of its FieldInfo parity; a phase gains pi), coordinates reflected about the
        planes; images equal to the drawn half are left out. ValueError for a field of unknown parity, coordinates on
        a polar / spherical window or a coordinate of an axis mirrored on both faces (the unfolding repeats)."""
        spec = self.view.spec
        faces = mirror_faces(spec)
        reached = self._touched_faces(mask)
        touched = [face for face in sorted(faces) if face in reached]
        if not touched:
            return []
        xp = self.domain.xp
        coordinates = [axis for axis in COORDINATE_AXES if axis in names and axis in namespace
                       and any(face[0] == axis for face in touched)]
        if coordinates and self.domain_map.mapped:
            raise ValueError('max / min over mirror images: coordinates x, y on a polar / spherical window are not '
                             'supported')
        for axis in coordinates:
            if f'{axis}_lo' in touched and f'{axis}_hi' in touched:
                raise ValueError(f'max / min of an expression of {axis} with mirror planes on both {axis} faces: the '
                                 'unfolded model repeats without end, so its extreme is not defined here')
        windows, edges = self.domain.physical_window(), self.domain.cell_edges()
        planes = {}   # face -> plane position [m] (the window edge)
        for face in touched:
            axis_index = COORDINATE_AXES.index(face[0])
            window = windows[axis_index]
            planes[face] = float(edges[axis_index][window.start if face.endswith('lo') else window.stop])
        field_names = self.view._field_names
        field_signs = {}   # name -> ({face: image sign}, phase)
        for name in names:
            if name in COORDINATE_AXES or name not in field_names or not isinstance(namespace.get(name), xp.ndarray):
                continue
            physics, field_name = field_names[name]
            info = self.view.renderer(physics).field_infos.get(field_name)
            if info is None:
                raise ValueError(f"max / min over mirror images: field '{name}' has no renderer description, so its "
                                 'mirror parity is unknown')
            parity = field_parity(info)
            physics_faces = mirror_faces(spec, physics)
            field_signs[name] = ({face: image_sign(parity, face[0], physics_faces.get(face, 1)) for face in touched},
                                 parity.startswith('phase_'))
        images = []
        for count in range(1, len(touched) + 1):
            for subset in itertools.combinations(touched, count):
                image, changed = dict(namespace), False
                for name, (signs, phase) in field_signs.items():
                    if math.prod(signs[face] for face in subset) < 0:
                        value = namespace[name]
                        image[name] = xp.where(value > 0, value - math.pi, value + math.pi) if phase else -value
                        changed = True
                for face in subset:
                    if face[0] in coordinates:
                        image[face[0]] = 2.0 * planes[face] - namespace[face[0]]
                        changed = True
                if changed:
                    images.append(image)
        return images

    def _value_at_point(self, values, point):
        """float of window-shaped device values in the window cell containing point object `point` (pos_to_cell)."""
        return float(values[self._point_window_index(point)])

    def _point_window_index(self, point):
        """Window cell index tuple (2D (i, j), 3D (i, j, k)) containing point object `point` (pos_to_cell, clamped)."""
        if not point:
            raise ValueError("operation 'at' over fields needs a point object (DerivedValueSpec.point)")
        context = self.view.context
        coordinates = list(context.point(point))
        if self.domain.dimensions == 3:
            coordinates.append(point_z(context.resolved_geometry[point].values, self.domain.size_z))
        locate = (self.domain.pos_to_cell_x, self.domain.pos_to_cell_y, getattr(self.domain, 'pos_to_cell_z', None))
        return tuple(min(max(locate[axis](value) - window.start, 0), window.stop - window.start - 1)
                     for axis, (value, window) in enumerate(zip(coordinates, self.domain.physical_window())))

    # ---- 3D reductions, one z plane of the expression at a time (SCL-26) ----
    @staticmethod
    def _plane_namespace(namespace, layer):
        """Namespace of window z layer `layer`: 3D arrays sliced to their (.., ..) plane (length-1 z axes to index 0)."""
        return {name: (value[:, :, layer if value.shape[2] > 1 else 0]
                       if getattr(value, 'ndim', 0) == 3 else value)
                for name, value in namespace.items()}

    def _plane_values(self, compiled, namespace, layer):
        """Device (wx, wy) values of the compiled expression on window z layer `layer`."""
        xp = self.domain.xp
        plane_shape = self.window_shape()[:2]
        return xp.broadcast_to(xp.asarray(compiled.evaluate(self._plane_namespace(namespace, layer), xp=xp)),
                               plane_shape)

    def _plane_value_at_point(self, compiled, namespace, point):
        """3D 'at': the expression on the z layer holding point object `point`, read at its cell."""
        i, j, k = self._point_window_index(point)
        return float(self._plane_values(compiled, namespace, k)[i, j])

    def _evaluate_planes(self, compiled, namespace, mask, operation, point):
        """3D evaluate() reductions over the window volume (or the selection mask) one z plane at a time, so no
        full-grid float64 temporaries appear (values, cell volumes, their product and masked copies)."""
        xp = self.domain.xp
        windows, widths = self.domain.physical_window(), self.domain.cell_widths()
        area_xy = xp.asarray(widths[0][windows[0]])[:, None] * xp.asarray(widths[1][windows[1]])[None, :]
        z_widths = widths[2][windows[2]]
        wants_extreme = operation in ('max', 'min') or (operation == 'mode_volume' and not point)
        wants_integral = operation not in ('max', 'min')
        fill = math.inf if operation == 'min' else -math.inf
        images = self._image_namespaces(compiled.names, namespace, mask) if wants_extreme else []
        extremes, integral, volume = [], 0.0, 0.0
        for layer, z_width in enumerate(z_widths):
            values = self._plane_values(compiled, namespace, layer)
            selected = None if mask is None else mask[:, :, layer]
            if wants_extreme:   # the drawn part and its mirror images (the unfolded extreme)
                for plane_values in [values] + [self._plane_values(compiled, image, layer) for image in images]:
                    masked = plane_values if selected is None else xp.where(selected, plane_values, fill)
                    extremes.append(masked.min() if operation == 'min' else masked.max())
            if wants_integral:
                weighted, areas = values * area_xy, area_xy
                if selected is not None:
                    weighted, areas = xp.where(selected, weighted, 0.0), xp.where(selected, area_xy, 0.0)
                integral = integral + weighted.sum() * float(z_width)   # device scalars: one sync at the end
                volume = volume + areas.sum() * float(z_width)
        if wants_extreme:
            stacked = xp.stack(extremes)
            extreme = float(stacked.min() if operation == 'min' else stacked.max())
            if operation in ('max', 'min'):
                return extreme
        integral = float(integral)
        if operation == 'mean':
            return integral / float(volume)
        integral *= touch_factor(mirror_faces(self.view.spec), self._touched_faces(mask))
        if operation == 'integral':
            return integral
        reference = self._plane_value_at_point(compiled, namespace, point) if point else extreme
        return mode_volume_ratio(integral, reference)

    def _evaluate_interface(self, compiled, selection, operation, earlier):
        """Reduce a compiled expression over the one-sided outline samples of an 'interface' selection (2D models):
        fields read from selection.side (interface_sampling), x, y and the normal / tangent per sample (polar /
        spherical: along the radial / angular field components); mean / integral weigh by outline length
        (axisymmetric: times 2 pi r), integral per unit z length doubled per mirror face within a cell of a usable
        sample."""
        if self.domain.dimensions != 2:
            raise ValueError(f"interface selection '{selection.name}' needs a 2D model")
        if operation not in INTERFACE_OPERATIONS:
            raise ValueError(f"operation {operation!r} over an interface selection; valid: "
                             f"{', '.join(INTERFACE_OPERATIONS)}")
        xp = self.domain.xp
        window_shape = self.window_shape()
        windows, all_edges = self.domain.physical_window(), self.domain.cell_edges()
        centres = [np.asarray(axis_centres[window], dtype=np.float64)
                   for axis_centres, window in zip(self.domain.cell_centers(), windows)]
        box = tuple(float(edges[bound]) for edges, window in zip(all_edges, windows)
                    for bound in (window.start, window.stop))   # logical on a polar window (no padding there)
        mapped = self.domain_map.mapped
        frame = window_frame(*centres, window_map=self.domain_map) if mapped else window_frame(*centres, box=box)
        stencil = interface_stencil(self.view.spec, selection, self.view.context.resolved_geometry, frame)
        cells = (xp.asarray(stencil.cells_x), xp.asarray(stencil.cells_y))
        namespace = self.evaluation_namespace(
            [name for name in compiled.names if name not in INTERFACE_VARIABLES and name not in ('x', 'y')], earlier)
        samples = {name: (stencil.one_sided(to_host(xp.broadcast_to(value, window_shape)[cells]))
                          if isinstance(value, xp.ndarray) and value.ndim else value)
                   for name, value in namespace.items()}
        samples.update(stencil.variables())   # they shadow parameters
        values = np.broadcast_to(np.asarray(compiled.evaluate(samples, xp=np), dtype=np.float64),
                                 stencil.lengths.shape)
        kept = stencil.usable & np.isfinite(values)
        if not kept.any():
            raise ValueError(f"interface selection '{selection.name}': no usable outline sample has a finite value")
        values = values[kept]
        if operation in ('max', 'min'):
            return float(values.max() if operation == 'max' else values.min())
        weights = stencil.lengths[kept]
        if self.view._has_axisymmetric_solver(compiled.names):
            axis_distance = stencil.points[kept, 0] - (0.0 if mapped else box[0])   # (r, z): from the x_lo edge
            weights = weights * self.view._ring_angle(compiled.names) * axis_distance
        if operation == 'mean':
            return float(np.sum(values * weights) / np.sum(weights))
        reaches = [1.5 * float(np.diff(axis).min()) for axis in centres]   # per logical axis on a polar window
        if not mapped:
            reaches = [min(reaches)] * 2
        points = frame.logical(stencil.points[kept])
        touched = {face for face, axis, edge in (('x_lo', 0, box[0]), ('x_hi', 0, box[1]), ('y_lo', 1, box[2]),
                                                 ('y_hi', 1, box[3]))
                   if bool(np.any(np.abs(points[:, axis] - edge) <= reaches[axis]))}
        return float(np.sum(values * weights)) * touch_factor(mirror_faces(self.view.spec), touched)

    def _touched_faces(self, mask):
        """Outer faces whose window edge layer a selection mask (None = the whole window) reaches."""
        faces = FACES_3D[:2 * len(self.window_shape())]
        if mask is None:
            return set(faces)
        touched = set()
        for face in faces:
            axis = 'xyz'.index(face[0])
            layer = [slice(None)] * mask.ndim
            layer[axis] = 0 if face.endswith('lo') else -1
            if bool(mask[tuple(layer)].any()):
                touched.add(face)
        return touched

    def evaluation_namespace(self, names, earlier=None):
        """Window-shaped namespace: the view's mesh-free namespace, x / y (3D: z) cell centres, state arrays
        (windowed) and renderer fields; fields of physics on other meshes from their access (values_on_window)."""
        view = self.view
        namespace = view._plain_namespace(names, earlier)
        if 'x' in names or 'y' in names:
            namespace['x'], namespace['y'] = self._centre_coordinates()
        if 'z' in names and self.domain.dimensions == 3:   # validation rejects z in 2D models
            window_z = self.domain.physical_window()[2]
            namespace['z'] = self.domain.xp.asarray(self.domain.cell_centers()[2][window_z])[None, None, :]
        states = {}
        for name in names:
            if name in namespace or name not in view._field_names:
                continue   # unknown names are reported by Expression.evaluate
            physics, field_name = view._field_names[name]
            source = view._physics_view(physics)
            if source is view:
                namespace[name] = self.window_values(physics, field_name, states)
            else:
                namespace[name] = source.access.values_on_window(self, physics, field_name, states)
        return namespace

    def _centre_coordinates(self):
        """Device x, y cell-centre arrays broadcastable to the window (3D: along z too)."""
        window_x, window_y = self.domain.physical_window()[:2]
        x_centers, y_centers = self.domain.cell_centers()[:2]
        trailing = (None,) * (self.domain.dimensions - 2)   # 3D: broadcast along z
        return (self.domain.xp.asarray(x_centers[window_x])[(slice(None), None) + trailing],
                self.domain.xp.asarray(y_centers[window_y])[(None, slice(None)) + trailing])

    def window_values(self, physics, field_name, states):
        """Device window array of a field of a physics on this mesh: its state entry (windowed when full-grid; other
        state values as they are), else the renderer's; states caches state() per physics, Session.shared_field_reads
        the arrays per (mesh, physics, field) on 2D windows (3D volumes are not kept: device memory)."""
        root = self.view._parent or self.view
        shared = root._field_reads if self.domain.dimensions == 2 else None
        key = (self.domain.mesh_name, physics, field_name)
        if shared is not None and key in shared:
            return shared[key]
        value = self._read_window_values(physics, field_name, states)
        if shared is not None:
            shared[key] = value
        return value

    def _read_window_values(self, physics, field_name, states):
        if physics not in states:
            states[physics] = self.view._solver(physics).state()
        state = states[physics]
        if field_name not in state:
            return self._window_field(self.view.renderer(physics), field_name)
        value = state[field_name]
        if isinstance(value, self.domain.xp.ndarray) and value.shape == self.domain.materials.shape:
            return value[self.domain.physical_window()]
        return value

    def values_on_window(self, target, physics, field_name, states):
        """Linear interpolation between this window's centres onto target's (clamped; a cached GridTransfer, D2);
        non-array state values as they are."""
        values = self.window_values(physics, field_name, states)
        if not isinstance(values, self.domain.xp.ndarray) or tuple(values.shape) != self.window_shape():
            return values
        root = target.view._parent or target.view
        key = (self.domain.mesh_name, target.domain.mesh_name)
        transfer = root._window_transfers.get(key)
        if transfer is None:
            source_edges, target_edges = window_edges(self.domain), window_edges(target.domain)
            transfer = root._window_transfers[key] = GridTransfer(
                'interpolate', source_edges, [slice(0, len(edges) - 1) for edges in source_edges], target_edges,
                [slice(0, len(edges) - 1) for edges in target_edges], backend=target.domain.backend.name)
        return transfer.apply(values)

    def cartesian_centres(self, what):
        """Host float64 (x, y) window centres of a Cartesian 2D window; ValueError naming `what` otherwise."""
        if self.domain.dimensions != 2:
            raise ValueError(f'{what}: a triangle mesh reads Cartesian 2D windows only')
        return tuple(np.asarray(centres[window], dtype=np.float64)
                     for centres, window in zip(self.domain.cell_centers(), self.domain.physical_window()))

    def values_at(self, physics, field_name, points, location, states):
        """Clamped bilinear between the window centres of a Cartesian 2D window; non-array state values as they
        are."""
        x_centres, y_centres = self.cartesian_centres(f"field '{physics}.{field_name}'")
        values = self.window_values(physics, field_name, states)
        if not isinstance(values, self.domain.xp.ndarray) or tuple(values.shape) != self.window_shape():
            return values
        return _bilinear(x_centres, y_centres, np.asarray(to_host(values), dtype=np.float64), points)

    def _selection_mask(self, selection, window_shape):
        """Window-shaped bool mask of the selection's domain objects (the build-mask rasterisation); None for the
        whole window."""
        names = self.selected_objects(selection)
        if names is None:
            return None
        resolved = self.view.context.resolved_geometry
        xp = self.domain.xp
        window_centres, rasterize_object = self._mask_centres(window_shape)
        mask = xp.zeros(window_shape, dtype=np.bool_)
        for name in names:
            object_window, cropped = rasterize_object(name, resolved, *window_centres, xp)
            mask[object_window] |= cropped
        return mask

    def _mask_centres(self, window_shape):
        """(window centres, rasterize function) of the selection masks: the build's float32 cell centres (clamping is
        a no-op inside the window); 3D adds z."""
        window_centres = [centres[window].astype(DTYPE)
                          for centres, window in zip(self.domain.cell_centers(), self.domain.physical_window())]
        return window_centres, (rasterize_box if len(window_centres) == 3 else rasterize_window)


class MappedAccess(TensorAccess):
    """Polar / spherical windows and cylindrical (r, phi, z) sectors: fields as containing-cell images over the
    drawing-plane box (native: the logical cells), z cuts only, reductions over the core.metric sector volumes."""

    canvas_field = FieldAccess.canvas_field

    def _physical_widths(self, widths):
        """dr and r_c d_angle (a sector: and dz)."""
        return (widths[0], np.outer(*self._arc_factors())) + tuple(widths[2:])

    def _native_field(self, physics, component, renderer, device_values, plane, axes, native):
        """native: the logical cells themselves (_logical_field_data); else the containing-cell image at 4 pixels per
        window cell within MAPPED_FIELD_PIXELS."""
        if native:
            return self._logical_field_data(physics, component, renderer, device_values, plane)
        low, high = MAPPED_FIELD_PIXELS
        return self._mapped_field_data(physics, component, renderer, device_values,
                                       int(min(high, max(low, 4 * device_values.size))), plane)

    def _display_data(self, physics, component, renderer, device_values, max_pixels, plane, axes):
        return self._mapped_field_data(physics, component, renderer, device_values, max_pixels, plane)

    def segment_profile(self, physics, component, start, end, sampling):
        raise ValueError('segment_profile samples Cartesian 3D models; on a cylindrical (r, phi, z) sector sample '
                         "a line in a z cut (field(plane=('z', height)))")

    def _volume_centres(self):
        """r [m], phi [rad], z [m] (logical r - radius_min, phi - angle_min shifted back)."""
        centres = super()._volume_centres()
        return centres[0] + self.domain_map.radius_min, centres[1] + self.domain_map.angle_min, centres[2]

    def cut_centers(self, axis):
        return super().cut_centers(axis) if axis == 'z' else ()   # a sector is cut at constant z only

    def _reduces_by_planes(self):
        return False

    def _cell_measures(self, window_shape, names=()):
        """Exact sector areas / whole-shell volumes (core.metric)."""
        return self.domain.xp.asarray(self._window_metric().volume.full())

    def _centre_coordinates(self):
        """Drawing-plane centres (no PML there); a sector: broadcast along z."""
        trailing = (None,) * (self.domain.dimensions - 2)
        return tuple(self.domain.xp.asarray(values)[(Ellipsis,) + trailing] for values in self._mapped_centres())

    def cartesian_centres(self, what):
        raise ValueError(f'{what}: a triangle mesh reads Cartesian 2D windows only')

    def _mask_centres(self, window_shape):
        """The build's float32 drawing-plane centres (a sector: + host float32 z centres)."""
        window_centres = [self.domain.xp.asarray(values, dtype=DTYPE) for values in self._mapped_centres()]
        if len(window_shape) == 3:
            window_centres.append(self.domain.host_centres()[2])
            return window_centres, rasterize_mapped_box
        return window_centres, rasterize_mapped

    def _mapped_centres(self):
        """Host float64 (wx, wy) drawing-plane (x, y) cell centres [m] of a polar / spherical window (a cylindrical
        sector: of its (r, phi) plane)."""
        x_centres, y_centres = self.domain.cell_centers()[:2]
        return self.domain_map.to_physical(x_centres[:, None], y_centres[None, :])

    def _window_metric(self):
        """core.metric.WindowMetric of the polar / spherical window (sector areas, whole-shell volumes); a cylindrical
        sector: its WindowMetric3D (volumes r_c dr dphi dz)."""
        widths = [axis_widths[window] for axis_widths, window in zip(self.domain.cell_widths(),
                                                                     self.domain.physical_window())]
        if self.domain.dimensions == 3:
            return window_metric_3d(*widths, self.domain.coordinates)
        return window_metric(*widths, self.domain.coordinates, self.domain_map.axisymmetric)

    def _arc_factors(self):
        """(along r, along the angle) 1D host factors of the window's angular cell lengths r_c d_angle [m]."""
        metric = self._window_metric()
        arc = metric.length[1] if self.domain.dimensions == 3 else metric.y_length
        return arc.along_x, arc.along_y

    def _mapped_field_data(self, physics, component, renderer, device_values, max_pixels, plane=None):
        """FieldData of window values on a polar / spherical grid (a sector: its z cut `plane`): square pixels over the
        sector's drawing-plane box, each the window cell containing its centre, NaN outside the sector (host indices
        cached per max_pixels; the image is gathered on the host from the window cells, so a run allocates no
        pixel-sized device arrays: at least 4 pixels per cell)."""
        key = ('mapped', max_pixels)
        if key not in self._display_indices:
            if max_pixels < 1:
                raise ValueError(f'max_pixels must be >= 1, got {max_pixels}')
            x_min, x_max, y_min, y_max = self.domain_map.extent()
            width, height = x_max - x_min, y_max - y_min
            arc_along_r, arc_along_angle = self._arc_factors()   # r_c d_angle
            finest = float(min(self.domain.cell_widths()[0].min(), arc_along_r.min() * arc_along_angle.min()))
            pixel = max(finest, math.sqrt(width * height / max_pixels))
            n_u, n_v = max(1, int(width / pixel)), max(1, int(height / pixel))
            if n_u * n_v > max_pixels:   # float rounding guard
                n_u = max(1, max_pixels // n_v)
            x_logical, y_logical = self.domain_map.to_logical(
                (x_min + (np.arange(n_u) + 0.5) * (width / n_u))[:, None],
                (y_min + (np.arange(n_v) + 0.5) * (height / n_v))[None, :])
            x_edges, y_edges = self.domain.cell_edges()[:2]   # no padding: the window starts at 0
            inside = ((x_logical >= x_edges[0]) & (x_logical <= x_edges[-1])
                      & (y_logical >= y_edges[0]) & (y_logical <= y_edges[-1]))
            cells_x, cells_y = (np.clip(np.searchsorted(edges, coordinate, side='right') - 1, 0, edges.size - 2)
                                for edges, coordinate in ((x_edges, x_logical), (y_edges, y_logical)))
            flat = np.where(inside, cells_x * (y_edges.size - 1) + cells_y, 0).astype(np.int64)
            self._display_indices[key] = (flat.ravel(), inside.ravel(), (n_u, n_v))
        flat, inside, shape = self._display_indices[key]
        cell_values = np.ravel(host_float32(device_values))
        values = np.where(inside, cell_values[flat], np.float32(np.nan)).reshape(shape)
        return FieldData(physics, component, values, self.domain_map.extent(),
                         renderer.field_info(component), self.view.time, renderer.time_label(),
                         native_shape=tuple(int(count) for count in device_values.shape), plane=plane,
                         mesh=self.domain.mesh_name)

    def _logical_field_data(self, physics, component, renderer, device_values, plane=None):
        """FieldData of window values on the polar / spherical cells themselves (a sector: its z cut `plane`): axes
        ('r', 'phi' | 'theta'), centres r [m] and angle [rad], extent of the cell edges, mapped_centres the
        drawing-plane centres."""
        window_map = self.domain_map
        x_edges, y_edges = self.domain.cell_edges()[:2]   # no padding: the window starts at 0
        x_centres, y_centres = self.domain.cell_centers()[:2]
        extent = (window_map.radius_min + float(x_edges[0]), window_map.radius_min + float(x_edges[-1]),
                  window_map.angle_min + float(y_edges[0]), window_map.angle_min + float(y_edges[-1]))
        return FieldData(physics, component, host_float32(device_values), extent, renderer.field_info(component),
                         self.view.time, renderer.time_label(),
                         x_centers=window_map.radius_min + np.array(x_centres, dtype=np.float64),
                         y_centers=window_map.angle_min + np.array(y_centres, dtype=np.float64),
                         axes=('r', 'theta' if window_map.axisymmetric else 'phi'), plane=plane,
                         mapped_centres=tuple(np.array(centres, dtype=np.float64)
                                              for centres in self._mapped_centres()), mesh=self.domain.mesh_name)


class LineAccess(MappedAccess):
    """1D models (DomainMap.line): values (wx, 1) on the cells along the drawing's x axis, native or not."""

    def _physical_widths(self, widths):
        return widths[:1]   # only the cells along x are lengths

    def _native_field(self, physics, component, renderer, device_values, plane, axes, native):
        return self._line_field_data(physics, component, renderer, device_values)

    def _display_data(self, physics, component, renderer, device_values, max_pixels, plane, axes):
        return self._line_field_data(physics, component, renderer, device_values)

    def _line_field_data(self, physics, component, renderer, device_values):
        """FieldData of a 1D model: values (wx, 1) on its cells, x_centers the drawing-plane x (slab x, or r) [m],
        y_centers [0] (the x axis), extent the DomainMap strip, so line profiles along y = 0 read the cells."""
        x_centres, _ = self._mapped_centres()
        return FieldData(physics, component, host_float32(device_values), self.domain_map.extent(),
                         renderer.field_info(component), self.view.time, renderer.time_label(),
                         x_centers=np.array(x_centres[:, 0], dtype=np.float64), y_centers=np.zeros(1),
                         mesh=self.domain.mesh_name)


def window_edges(domain):
    """Float64 window cell edges per axis of a tensor Domain / Domain3D (Domain coordinates)."""
    return [np.asarray(edges[window.start:window.stop + 1], dtype=np.float64)
            for edges, window in zip(domain.cell_edges(), domain.physical_window())]


def _bilinear(x_centres, y_centres, values, points):
    """float64 clamped bilinear interpolation of (wx, wy) cell-centre values at (n, 2) points [m]."""
    parts = []
    for centres, positions in ((x_centres, points[:, 0]), (y_centres, points[:, 1])):
        if centres.size == 1:
            zeros = np.zeros(len(positions), dtype=np.int64)
            parts.append((zeros, zeros, np.zeros(len(positions))))
            continue
        positions = np.clip(positions, centres[0], centres[-1])
        low = np.clip(np.searchsorted(centres, positions, side='right') - 1, 0, centres.size - 2)
        parts.append((low, low + 1, (positions - centres[low]) / (centres[low + 1] - centres[low])))
    (x_low, x_high, x_part), (y_low, y_high, y_part) = parts
    return ((1.0 - x_part) * ((1.0 - y_part) * values[x_low, y_low] + y_part * values[x_low, y_high])
            + x_part * ((1.0 - y_part) * values[x_high, y_low] + y_part * values[x_high, y_high]))


def _nearest_cells(edges, n_pixels, xp):
    """Device (xp) int64 index of the cell containing each of n_pixels uniform pixel centres over [edges[0],
    edges[-1]]."""
    span = float(edges[-1] - edges[0])
    pixel_centers = edges[0] + (np.arange(n_pixels) + 0.5) * (span / n_pixels)
    cells = np.searchsorted(edges, pixel_centers, side='right') - 1
    return xp.asarray(np.clip(cells, 0, len(edges) - 2).astype(np.int64))
