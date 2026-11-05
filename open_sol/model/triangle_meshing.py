"""Triangle meshes of 'triangles' MeshSpecs (solver_plan S7.5 2.2): PSLG and size field from the resolved geometry
(the only reader of shape internals for the mesher), the fem mesher (built-in or gmsh), the PML layer ring, and
element regions / boundary edges / points tagged by the builder's rasterizer rules. Host numpy only."""
import math
import time

import numpy as np

from ..fem.mesh import Curve, build_triangle_mesh
from ..fem.mesher import MERGE_TOLERANCE, SizeField, add_layer_ring, build_pslg, mesh_pslg, reflect_mesh
from ..fem.quality import mesh_quality
from .geometry import (GEOMETRY_KINDS, UNION_KINDS, bounding_box, domain_entities, domain_size, is_plain,
                       primitive_names, rasterize, resolve_geometry, rounded_corners)
from .schema import get_registry

_OUTLINE_SPACING = 0.25   # sampled outline sources of a sized object: spacing / its size
_CULLED_PIECES = 4        # unions of at least this many inputs test each input only near its bounding box


def triangulate(spec, parameters, registry=None, mesh=None, *, shapes=None, sizing=None, log=None):
    """fem.mesh.TriangleMesh of a 'triangles' mesh (its name; None = meshing.default_mesh_name): sizes from
    meshing.triangle_sizing (it refuses models that are not 2D Cartesian) unless given, shapes a strict
    resolve_geometry unless given; ValueError on model errors, RuntimeError when the mesher fails. log(level, text)
    gets the stage times (the time_estimate mesh items: mesher, element tagging)."""
    started = time.perf_counter()
    shapes = resolve_geometry(spec, parameters) if shapes is None else shapes
    if sizing is None:
        from .meshing import default_mesh_name, triangle_sizing

        mesh_spec = spec.mesh_by_name(default_mesh_name(spec) if mesh is None else mesh)
        registry = registry if registry is not None else get_registry()
        sizing = triangle_sizing(spec, mesh_spec, parameters, registry, shapes)
    size_x, size_y = domain_size(spec, parameters)
    window = (0.0, 0.0, float(size_x), float(size_y))
    periodic_axes = _periodic_axes(sizing)
    parts = (_mirrored_parts(shapes, window, sizing, periodic_axes) if sizing.mirror_axes
             else _plain_parts(shapes, window, sizing, periodic_axes))
    points, triangles, pml_faces = parts['points'], parts['triangles'], parts['pml_faces']
    meshed = time.perf_counter()
    names = [name for name in domain_entities(spec) if name != 'all' and name in shapes]
    masks = element_masks(shapes, names, window, points[triangles].mean(axis=1))
    region = np.full(len(triangles), -1, dtype=np.int32)
    for index, name in enumerate(names):
        region[masks[name]] = index
    tagged = time.perf_counter()
    quality = mesh_quality(points, triangles, pml_faces == 0, size_field=parts['size_field'],
                           small_input_angles=parts['small_angles'], rounds=parts['rounds'], mesher=sizing.mesher)
    built = build_triangle_mesh(
        points, triangles, curves=parts['curves'], curved_pairs=parts['curved_pairs'], curved_ids=parts['curved_ids'],
        tagged_pairs=parts['line_pairs'], outline_masks=masks, point_tags=parts['point_tags'], region=region,
        region_names=names, pml_faces=pml_faces, window=window, quality=quality, periodic_axes=periodic_axes)
    if log is not None:
        finished = time.perf_counter()
        log('info', f"mesh '{mesh or ''}': {len(triangles):,} triangles ({len(points):,} nodes, {parts['rounds']} "
                    f'refinement rounds, {sizing.mesher}) in {finished - started:.2f} s: mesher {meshed - started:.2f} '
                    f's, element tagging ({len(names)} objects) {tagged - meshed:.2f} s, quality + mesh arrays '
                    f'{finished - tagged:.2f} s')
    return built


def _periodic_axes(sizing) -> tuple:
    """'x' / 'y' of sizing.periodic_axes (new_modules_plan G3: matching opposite window sides); ValueError when one
    carries an absorbing ring (a periodic side is the cell boundary)."""
    periodic_axes = tuple(axis for axis in ('x', 'y') if axis in tuple(getattr(sizing, 'periodic_axes', ())))
    for axis in periodic_axes:
        widths = [float(sizing.pml_widths.get(f'{axis}_{side}') or 0.0) for side in ('lo', 'hi')]
        if any(widths):
            raise ValueError(f'triangle mesh: periodic {axis} needs no absorbing ring on {axis}_lo / {axis}_hi (now '
                             f'{widths[0]:g} m / {widths[1]:g} m); give those faces a non-absorbing condition')
    return periodic_axes


def _run_mesher(pslg, size_field, sizing):
    """RawMesh of the PSLG by the sizing's mesher (built-in or gmsh)."""
    if sizing.mesher == 'gmsh':
        from ..fem.gmsh_io import mesh_with_gmsh

        return mesh_with_gmsh(pslg, size_field)
    return mesh_pslg(pslg, size_field, h_min=sizing.h_min)


def _plain_parts(shapes, window, sizing, periodic_axes=(), size_field=None) -> dict:
    """Mesh pieces of the whole window: nodes, elements, ring bits, curves and curved pairs, line pairs by the PSLG
    segment tags, point tags by the PSLG vertices, the size field, small input angles, refinement rounds, constraint
    edges; the window sides of periodic_axes meshed with matching nodes; size_field None = mesher_size_field (swept
    prisms pass their own)."""
    size_field = mesher_size_field(sizing, shapes) if size_field is None else size_field
    pslg = model_pslg(shapes, window, size_field, sizing, periodic_axes=periodic_axes)
    raw = _run_mesher(pslg, size_field, sizing)
    points, triangles, pml_faces = add_layer_ring(raw.points, raw.triangles, window, sizing.pml_widths,
                                                  sizing.pml_layers)
    segment_tags = [pslg.segment_tags[segment] for segment in raw.constraint_segment]
    line_pairs = {}
    for pair, tags in zip(raw.constraint_edges, segment_tags):
        for tag in tags:
            if tag in shapes and GEOMETRY_KINDS[shapes[tag].kind].entity_kind == 'line':
                line_pairs.setdefault(tag, []).append(pair)
    curve_of = pslg.segment_curve[raw.constraint_segment]
    return {'points': points, 'triangles': triangles, 'pml_faces': pml_faces, 'curves': pslg.curves,
            'curved_pairs': raw.constraint_edges[curve_of >= 0], 'curved_ids': curve_of[curve_of >= 0],
            'line_pairs': {name: np.array(pairs) for name, pairs in line_pairs.items()},
            'point_tags': pslg.vertex_tags, 'size_field': size_field, 'small_angles': pslg.small_angles,
            'rounds': raw.rounds, 'constraint_edges': raw.constraint_edges}


def _mirrored_parts(shapes, window, sizing, periodic_axes=()) -> dict:
    """_plain_parts of a mirror-symmetric mesh (new_modules_plan G2): the half (quarter) window below the centre
    line(s) of sizing.mirror_axes meshed with a symmetric size field and the PSLG of the geometry plus its mirror
    images, the PML ring on its outer faces, then mirrored per axis (fem.mesher.reflect_mesh; curves mapped to their
    images); line and point tags found geometrically on the whole mesh. A periodic mirror axis matches by the
    reflection itself; the other periodic axis is matched in the half mesh."""
    planes = [(axis, 0.5 * (window[axis] + window[axis + 2])) for axis in ('xy'.index(name) for name in
                                                                         sizing.mirror_axes)]
    size_field = _symmetric_size_field(mesher_size_field(sizing, shapes), planes)
    half, widths = list(window), dict(sizing.pml_widths)
    for axis, plane in planes:
        half[axis + 2] = plane
        widths[f"{'xy'[axis]}_hi"] = 0.0   # the mirror line: the image of the lo-side ring takes its place
    half = tuple(half)
    pslg = model_pslg(shapes, half, size_field, sizing, planes=planes,
                      periodic_axes=tuple(axis for axis in periodic_axes if axis not in sizing.mirror_axes))
    raw = _run_mesher(pslg, size_field, sizing)
    points, triangles, pml_faces = add_layer_ring(raw.points, raw.triangles, half, widths, sizing.pml_layers)
    tolerance = MERGE_TOLERANCE * max(window[2] - window[0], window[3] - window[1])
    curves = list(pslg.curves)
    pairs = raw.constraint_edges.astype(np.int64)
    curve_of = pslg.segment_curve[raw.constraint_segment].astype(np.int64)
    for axis, plane in planes:
        curve_image = np.array([_curve_index(curves, mirrored_curve(curve, axis, plane), tolerance)
                                for curve in list(curves)], dtype=np.int64)
        points, triangles, pml_faces, image = reflect_mesh(points, triangles, pml_faces, axis, plane)
        pairs = np.concatenate([pairs, image[pairs]])
        image_curves = curve_of.copy()
        image_curves[curve_of >= 0] = curve_image[curve_of[curve_of >= 0]]
        curve_of = np.concatenate([curve_of, image_curves])
    _, first = np.unique(np.sort(pairs, axis=1), axis=0, return_index=True)
    first = np.sort(first)
    pairs, curve_of = pairs[first], curve_of[first]
    return {'points': points, 'triangles': triangles, 'pml_faces': pml_faces, 'curves': tuple(curves),
            'curved_pairs': pairs[curve_of >= 0], 'curved_ids': curve_of[curve_of >= 0],
            'line_pairs': _line_pairs_on(shapes, points, pairs, tolerance),
            'point_tags': _point_tags_at(shapes, window, points, tolerance), 'size_field': size_field,
            'small_angles': pslg.small_angles, 'rounds': raw.rounds}


def mirrored_curve(curve, axis, plane) -> Curve:
    """fem Curve of the mirror image of a curve about the line coordinate[axis] = plane (axis 0: x, 1: y); the image
    of parameter t is -t, so an arc t0 .. t0 + span maps to -(t0 + span) .. -t0."""
    t0 = curve.t0 if curve.is_closed else -(curve.t0 + curve.span)
    if axis == 0:
        return Curve(2.0 * plane - curve.cx, curve.cy, curve.a, curve.b, math.pi - curve.angle, t0, curve.span)
    return Curve(curve.cx, 2.0 * plane - curve.cy, curve.a, curve.b, -curve.angle, t0, curve.span)


def _curve_index(curves, curve, tolerance) -> int:
    """Index of the curve equal to `curve` (Curve.key) in the list, appended when missing."""
    key = curve.key(tolerance)
    for index, other in enumerate(curves):
        if other.key(tolerance) == key:
            return index
    curves.append(curve)
    return len(curves) - 1


def _reflected(points, axis, plane) -> np.ndarray:
    """Copy of float64 (n, 2) points mirrored about the line coordinate[axis] = plane."""
    mirrored = np.array(points, dtype=np.float64).reshape(-1, 2)
    mirrored[:, axis] = 2.0 * plane - mirrored[:, axis]
    return mirrored


def _symmetric_size_field(size_field, planes):
    """Callable (n, 2) points -> the smallest size_field value over each point and its mirror images."""
    def sizes(points):
        images = [np.asarray(points, dtype=np.float64).reshape(-1, 2)]
        for axis, plane in planes:
            images += [_reflected(image, axis, plane) for image in images]
        return np.min([size_field(image) for image in images], axis=0)
    return sizes


def _line_pairs_on(shapes, points, pairs, tolerance) -> dict:
    """{line entity: (k, 2) constraint node pairs with both nodes on its segment (within tolerance)}."""
    tagged = {}
    for name, shape in shapes.items():
        if GEOMETRY_KINDS[shape.kind].entity_kind != 'line':
            continue
        values = shape.values
        start = np.array([values['x0'], values['y0']], dtype=np.float64)
        along = np.array([values['x1'], values['y1']], dtype=np.float64) - start
        length_squared = float(along @ along)
        on_segment = []
        for column in (0, 1):
            offset = points[pairs[:, column]] - start
            fraction = np.clip(offset @ along / length_squared, 0.0, 1.0) if length_squared > 0.0 else 0.0
            on_segment.append(np.linalg.norm(offset - np.multiply.outer(fraction, along), axis=1) <= tolerance)
        members = pairs[on_segment[0] & on_segment[1]]
        if len(members):
            tagged[name] = members
    return tagged


def _point_tags_at(shapes, window, points, tolerance) -> dict:
    """{point entity: node} of the point entities inside the window (within tolerance) that are mesh nodes."""
    from scipy.spatial import cKDTree

    tree = cKDTree(points)
    x0, y0, x1, y1 = window
    tags = {}
    for name, shape in shapes.items():
        if GEOMETRY_KINDS[shape.kind].entity_kind != 'point':
            continue
        x, y = float(shape.values['x']), float(shape.values['y'])
        if not (x0 - tolerance <= x <= x1 + tolerance and y0 - tolerance <= y <= y1 + tolerance):
            continue
        distance, node = tree.query([x, y])
        if distance <= 10.0 * tolerance:
            tags[name] = int(node)
    return tags


def mirror_mismatch(spec, shapes, sizes, axis):
    """(objects, mismatched sample fraction) where the material objects drawn are not mirror symmetric about the
    window centre line normal to axis 0 (x) / 1 (y) (symmetry_check's sampling and paint rules, one-sample tolerance),
    None when they are."""
    import itertools

    from .geometry import expand_entities
    from .symmetry_check import SAMPLES_PER_SIDE_2D, object_masks, paint, sample_centres

    selections = []
    for material in spec.materials:
        try:
            selections.append(expand_entities(spec, material.selection))
        except (ValueError, TypeError):   # reported by the selection checks
            selections.append([])
    names = sorted({entity for entities in selections for entity in entities
                    if entity in shapes and GEOMETRY_KINDS[shapes[entity].kind].entity_kind == 'domain'})
    if not names:
        return None
    half = [float(size) for size in sizes]
    half[axis] *= 0.5
    centres = sample_centres(f"{'xy'[axis]}_hi", tuple(half), SAMPLES_PER_SIDE_2D)   # pairs k <-> 2 n - 1 - k
    masks = object_masks(shapes, names, centres)
    shape = tuple(axis_centres.size for axis_centres in centres)
    labels = paint(selections, masks, shape, set(names))
    mirrored = np.pad(np.flip(labels, axis=axis), 1, mode='edge')
    matched = np.zeros(shape, dtype=bool)
    for offsets in itertools.product((0, 1, 2), repeat=2):   # staircase ties: one sample either way
        matched |= labels == mirrored[tuple(slice(offset, offset + size) for offset, size in zip(offsets, shape))]
    bad = ~matched
    if not bad.any():
        return None
    either = bad | np.flip(bad, axis=axis)
    return [name for name in names if (masks[name] & either).any()], float(bad.mean())


def element_masks(shapes, names, window, centroids) -> dict:
    """{name: bool per element} of domain objects at the element centroids (geometry.rasterize, the build-mask rules);
    centroids outside the window (PML ring) are clamped just inside it, like the tensor PML cells."""
    x0, y0, x1, y1 = window
    inset = MERGE_TOLERANCE * max(x1 - x0, y1 - y0)
    clamped = np.column_stack([np.clip(centroids[:, 0], x0 + inset, x1 - inset),
                               np.clip(centroids[:, 1], y0 + inset, y1 - inset)])
    return {name: contained(name, shapes, clamped) for name in names}


def contained(name, shapes, points) -> np.ndarray:
    """Bool per (n, 2) point of geometry.rasterize(name); a union of many inputs tests each input only at the points
    in its bounding box (a KD tree of the points), the same result."""
    pieces = _union_pieces(name, shapes)
    if len(pieces) < _CULLED_PIECES:
        return np.asarray(rasterize(name, shapes, points[:, 0], points[:, 1], np), dtype=bool)
    from scipy.spatial import cKDTree

    boxes = np.array([bounding_box(piece, shapes) for piece in pieces], dtype=np.float64)
    centres = 0.5 * (boxes[:, :2] + boxes[:, 2:])
    half_widths = 0.5 * (boxes[:, 2:] - boxes[:, :2]).max(axis=1)
    radii = half_widths + 1e-9 * (np.abs(centres).max(axis=1) + half_widths)   # never loses a point on the box
    inside = np.zeros(len(points), dtype=bool)
    for piece, nearby in zip(pieces, cKDTree(points).query_ball_point(centres, r=radii, p=np.inf)):
        index = np.asarray(nearby, dtype=np.int64)
        index = index[~inside[index]]
        if len(index):
            inside[index] = np.asarray(rasterize(piece, shapes, points[index, 0], points[index, 1], np), dtype=bool)
    return inside


def _union_pieces(name, shapes) -> list:
    """Names whose rasterize masks OR to name's: the inputs of (nested) unions and the copies of arrays, else
    [name]."""
    shape = shapes[name]
    if shape.kind not in UNION_KINDS or not is_plain(shape):
        return [name]
    return [piece for item in shape.values['inputs'] for piece in _union_pieces(item, shapes)]


def mesher_size_field(sizing, shapes) -> SizeField:
    """fem SizeField of a meshing.TriangleSizing, the mesher's form of meshing.triangle_size_field (the raster of the
    estimate): object sizes inside / along / at their objects, local curvature sizes (ellipses per point, the
    estimate takes the smallest radius; rounded-rectangle corners along their arcs only, the estimate all round), size
    points, floored at h_min."""
    size_field = SizeField(sizing.h_max, sizing.growth, sizing.h_min)
    for name, size in sizing.object_sizes.items():
        shape = shapes.get(name)
        if shape is None:
            continue
        entity_kind = GEOMETRY_KINDS[shape.kind].entity_kind
        values = shape.values
        if entity_kind == 'domain':
            size_field.add_region(_inside(name, shapes), size)
            for primitive in _primitives(name, shapes):
                for polyline in _outline_polylines(primitive, size):
                    size_field.add_polyline(polyline, size)
        elif entity_kind == 'line':
            size_field.add_polyline([[values['x0'], values['y0']], [values['x1'], values['y1']]], size)
        else:
            size_field.add_points([[values['x'], values['y']]], size)
    for shape in shapes.values():
        for curve in _shape_curves(shape):
            size_field.add_curve(curve, sizing.curvature_factor, sizing.h_min)
    size_points = sizing.size_points
    for size in np.unique(size_points[:, 2]):   # adapted sizes (MeshSpec.size_points, quantised per octave)
        size_field.add_points(size_points[size_points[:, 2] == size, :2], float(size))
    return size_field


def model_pslg(shapes, window, size_field, sizing, planes=(), periodic_axes=()):
    """fem PSLG of the window plus the outline of every 2D domain primitive (composite inputs and array copies, which
    resolve_geometry lists as shapes of their own, included), line entities
    as tagged constraint segments and point entities as tagged vertices; planes [(axis, coordinate)]: their mirror
    images are added too (untagged; the half window of a mirrored mesh then conforms to both sides); periodic_axes
    ('x' / 'y'): matching vertices on the opposite window sides (fem.mesher.periodic_pslg)."""
    straight, curves, points = [], [], {}
    for name, shape in shapes.items():
        kind = GEOMETRY_KINDS[shape.kind]
        values = shape.values
        if kind.entity_kind == 'point':
            points[name] = (values['x'], values['y'])
        elif kind.entity_kind == 'line':
            straight.append((values['x0'], values['y0'], values['x1'], values['y1'], (name,)))
        elif not kind.composite:
            sides, shape_curves = _outline_parts(shape)
            curves += [(curve, ()) for curve in shape_curves]
            straight += [(*start, *end, ()) for start, end in sides]
    for axis, plane in planes:
        images = [(*_reflected([piece[:2], piece[2:4]], axis, plane).ravel(), ()) for piece in straight]
        straight += images
        curves += [(mirrored_curve(curve, axis, plane), ()) for curve, _ in curves]
        points.update({(name, f"mirror_{'xy'[axis]}"): tuple(_reflected([xy], axis, plane)[0])
                       for name, xy in list(points.items())})
    return build_pslg(window, size_field, straight=straight, curves=curves, points=points,
                      curvature_factor=sizing.curvature_factor, h_min=sizing.h_min, periodic_axes=periodic_axes)


def _primitives(name, shapes) -> list:
    """Plain domain shapes under a domain object (geometry.primitive_names)."""
    return [shapes[primitive] for primitive in primitive_names(name, shapes)]


def _curve(shape):
    """Closed fem Curve of a circle / ellipse, None for other kinds."""
    values = shape.values
    if shape.kind == 'circle':
        return Curve(values['cx'], values['cy'], values['r'], values['r'])
    if shape.kind == 'ellipse':
        return Curve(values['cx'], values['cy'], values['a'], values['b'], math.radians(values['angle']))
    return None


def _shape_curves(shape) -> list:
    """fem Curves of a primitive's outline: a circle / ellipse closed, a rounded rectangle's four quarter-circle arc
    pieces (none for r = 0), [] for other kinds."""
    curve = _curve(shape)
    if curve is not None:
        return [curve]
    if shape.kind == 'rounded_rectangle' and shape.values['r'] > 0.0:
        radius = shape.values['r']
        return [Curve(cx, cy, radius, radius, 0.0, start, 0.5 * math.pi)
                for cx, cy, start in rounded_corners(shape.values)]
    return []


def _corners(shape) -> list:
    """Corner (x, y) tuples of a rectangle / polygon (a rounded rectangle: its arc ends, two per corner) in order (the
    last connects back to the first)."""
    values = shape.values
    if shape.kind == 'rectangle' or (shape.kind == 'rounded_rectangle' and not values['r'] > 0.0):
        x0, y0, x1, y1 = values['x0'], values['y0'], values['x1'], values['y1']
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    if shape.kind == 'rounded_rectangle':   # (x1 - r, y0), (x1, y0 + r), (x1, y1 - r), ... counter-clockwise
        x0, y0, x1, y1, radius = values['x0'], values['y0'], values['x1'], values['y1'], values['r']
        return [(x1 - radius, y0), (x1, y0 + radius), (x1, y1 - radius), (x1 - radius, y1), (x0 + radius, y1),
                (x0, y1 - radius), (x0, y0 + radius), (x0 + radius, y0)]
    if shape.kind == 'polygon':
        return [(float(px), float(py)) for px, py in values['points']]
    raise ValueError(f"triangle mesh: geometry kind {shape.kind!r} ('{shape.name}') has no 2D outline")


def _outline_parts(shape) -> tuple:
    """(straight sides [((x, y), (x, y))], fem Curves) of a primitive's outline: a rounded rectangle's sides run
    between its arc ends (zero-length sides of a stadium / circle dropped)."""
    curves = _shape_curves(shape)
    if curves and shape.kind != 'rounded_rectangle':
        return [], curves
    corners = _corners(shape)
    pairs = list(zip(corners, corners[1:] + corners[:1]))
    if curves:   # rounded: the sides are every other pair (arc end -> next arc start); the rest are the arcs
        pairs = pairs[1::2]
    return [(start, end) for start, end in pairs if start != end], curves


def _outline_polylines(shape, size) -> list:
    """Closed float64 polylines of a primitive's outline, curves sampled every _OUTLINE_SPACING x size."""
    curve = _curve(shape)
    if curve is None and shape.kind == 'rounded_rectangle' and shape.values['r'] > 0.0:
        quarter = 0.5 * math.pi * shape.values['r']
        count = max(4, int(math.ceil(quarter / (_OUTLINE_SPACING * size))))
        polyline = np.concatenate([arc.arc_points(count) for arc in _shape_curves(shape)])
        return [np.vstack([polyline, polyline[:1]])]
    if curve is None:
        corners = _corners(shape)
        return [np.array(corners + corners[:1], dtype=np.float64)]
    perimeter = 2.0 * math.pi * max(curve.a, curve.b)
    count = max(16, int(math.ceil(perimeter / (_OUTLINE_SPACING * size))))
    return [curve.points(np.linspace(0.0, 2.0 * math.pi, count + 1))]


def _inside(name, shapes):
    return lambda points: contained(name, shapes, points)
