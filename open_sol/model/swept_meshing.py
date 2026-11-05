"""Swept prism-layer tetrahedra of 'tetrahedra' MeshSpecs with sweep_axis 'z' (experiment plan D5): the xy footprint
meshed by the 2D triangle mesher, extruded through graded z layers and split into tetrahedra (fem.prism_layers); the
sizing rules, the element-count estimate and the entity tags. Host numpy only."""
# In-plane sizes are the triangle rules (dx = h_max, refinement dx inside / along their objects' footprints over the
# whole height, curvature_factor, h_min, the physics' edge sizes as outline sizes); the layers are the z part of
# meshing.graded_window_edges (keep_uniform=False: object z faces always snapped) plus the PML shell cells of dx; the
# x / y PML ring is the triangle ring of pml_layers. Vertical-walled primitives (boxes, z cylinders, untapered z prisms,
# untapered x / y rectangles) conform; spheres, other x / y prisms and tapers are tagged per element (a staircase).
import math
import time
from dataclasses import dataclass, replace

import numpy as np

from ..core.constants import FACES
from ..fem.mesh3d import PLANE_TOLERANCE
from ..fem.prism_layers import periodic_order_key, swept_counts, swept_tetrahedron_mesh
from .geometry import (GEOMETRY_KINDS, PRISM_KINDS, ResolvedShape, _footprint, contains_points_3d, domain_size,
                       domain_size_z, is_plain, primitive_names, rasterize_window, z_extent)

SWEPT_FACES = ('z_lo', 'z_hi')   # the face_sizes a swept mesh takes (as z layer bands)


@dataclass(frozen=True)
class Footprint:
    """The xy footprint of a 3D model's shapes for the 2D mesher: sized = every shape with 3D primitives as plain 2D
    ones (box -> rectangle, z cylinder / sphere -> circle, x / y prisms and tapers -> their footprint); outlined = the
    primitives with vertical side walls (their outlines go into the PSLG); unswept = the other primitives."""
    sized: dict
    outlined: dict
    unswept: tuple


def footprint_shapes(shapes) -> Footprint:
    """Footprint of resolved 3D shapes (composites, points and lines kept as they are in `sized`)."""
    sized, outlined, unswept = {}, {}, []
    for name, shape in shapes.items():
        kind = GEOMETRY_KINDS[shape.kind]
        if kind.entity_kind != 'domain' or kind.composite:
            sized[name] = shape
            continue
        flat, vertical = _footprint_shape(shape)
        sized[name] = flat
        if vertical:
            outlined[name] = flat
        else:
            unswept.append(name)
    return Footprint(sized, outlined, tuple(unswept))


def _footprint_shape(shape) -> tuple:
    """(plain 2D ResolvedShape of a 3D primitive's xy footprint, True when its side walls are vertical)."""
    values = shape.values
    if shape.kind == 'box':
        return ResolvedShape(shape.name, 'rectangle', {key: values[key] for key in ('x0', 'y0', 'x1', 'y1')}), True
    if shape.kind in ('cylinder_z', 'sphere'):   # a sphere: its equator disc, sized only
        return (ResolvedShape(shape.name, 'circle', {key: values[key] for key in ('cx', 'cy', 'r')}),
                shape.kind == 'cylinder_z')
    if shape.kind in PRISM_KINDS and not is_plain(shape):
        rectangle = shape.kind == 'rectangle' or (shape.kind == 'rounded_rectangle' and not values['r'] > 0.0)
        return _footprint(shape), rectangle and not values.get('sidewall_angle')
    return shape, True   # an untapered z prism as drawn


def footprint_sizing(sizing):
    """meshing.TriangleSizing of the footprint of a swept TetrahedronSizing (x / y PML ring and periodic axes)."""
    from .meshing import TriangleSizing
    return TriangleSizing(
        h_max=sizing.h_max, h_min=sizing.h_min, growth=sizing.growth, curvature_factor=sizing.curvature_factor,
        object_sizes=dict(sizing.object_sizes),
        pml_widths={face: float(sizing.pml_widths.get(face) or 0.0) for face in FACES},
        pml_layers=int(sizing.pml_layers), mesher=sizing.mesher,
        periodic_axes=tuple(axis for axis in sizing.periodic_axes if axis in ('x', 'y')))


def footprint_size_field(triangle_sizing, footprint, edge_sizes):
    """fem SizeField of the footprint: triangle_meshing.mesher_size_field plus each edge-size object's primitive
    outlines at its edge size (a physics' mesh_edge_sizes act along the vertical walls)."""
    from .triangle_meshing import _outline_polylines, _primitives, mesher_size_field
    size_field = mesher_size_field(triangle_sizing, footprint.sized)
    for name, size in edge_sizes.items():
        shape = footprint.sized.get(name)
        if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
            continue
        for primitive in _primitives(name, footprint.sized):
            for polyline in _outline_polylines(primitive, size):
                size_field.add_polyline(polyline, size)
    return size_field


# ---- sizing rules (meshing._element_sizing) ----
def swept_mesh_problem(mesh_spec, three_d) -> str:
    """'' when a MeshSpec with sweep_axis 'z' fits swept prism layers, else the first problem (not a tetrahedron mesh,
    size_points, x / y axes options, face sizes off the z faces)."""
    from ..core.constants import FACES_3D
    if not three_d or mesh_spec.kind != 'tetrahedra':
        return "sweep_axis 'z' applies to tetrahedron meshes only (3D Cartesian models); set it to 'none'"
    if mesh_spec.size_points:
        return ("size_points do not apply to swept prism layers (sweep_axis 'z': in-plane sizes come from dx and the "
                "refinements, layers from the z grading); remove them or set sweep_axis 'none'")
    in_plane = sorted({axis_spec.axis for axis_spec in mesh_spec.axes if axis_spec.axis != 'z'})
    if in_plane:
        return (f"mesh axes options for {' / '.join(map(str, in_plane))} do not apply to swept prism layers (only z "
                'is graded, into the layers; in-plane sizes come from dx and the refinements); remove them')
    for index, entry in enumerate(mesh_spec.face_sizes):
        face = entry.entity if entry.entity in FACES_3D else entry.face
        if face not in SWEPT_FACES:
            return (f'face size {index + 1} ({entry.entity!r}): swept prism layers take face sizes on z faces only '
                    "(z_lo / z_hi, or an object's face z_lo / z_hi: bands of thin layers); use a refinement for "
                    'in-plane sizes')
    return ''


def swept_refinement_problem(refinement) -> str:
    """'' unless a meshing._Refinement gives dx_x / dx_y unlike its dx (the in-plane size is one size)."""
    for axis in ('x', 'y'):
        if axis in refinement.axis_dx and not math.isclose(refinement.axis_dx[axis], refinement.dx, rel_tol=1e-9):
            return (f'dx_{axis} must equal dx on swept prism layers (the triangle mesher takes one in-plane size); '
                    'give dx for the in-plane size and dx_z for the layer thickness')
    return ''


def layer_edges(spec, mesh_spec, parameters, shapes, sizing) -> np.ndarray:
    """Float64 z node coordinates [m] of the prism layers: meshing.graded_window_edges of z (keep_uniform=False, so
    object z faces are always layer planes) plus the PML shell cells of h_max."""
    from .meshing import graded_window_edges, shell_axis_edges
    window = graded_window_edges(spec, mesh_spec, parameters, shapes, sizing, axes='z', keep_uniform=False)['z']
    return shell_axis_edges(window, sizing.h_max, sizing.pml_widths['z_lo'], sizing.pml_widths['z_hi'])


# ---- mesher (tetrahedral_meshing.tetrahedralize) ----
def swept_tetrahedra(spec, parameters, shapes, sizing, names, region_of, timings):
    """fem TetrahedronMesh of a swept sizing: the footprint (triangle_meshing._plain_parts on the outlined primitives,
    footprint_size_field, periodic x / y sides matched), extruded through sizing.layer_edges and split (periodic order
    key on periodic x / y); entities by _swept_entities, regions = region_of at clamped centroids with the entity
    majority; timings gets footprint_triangles / prism_split / region_tags seconds."""
    from .triangle_meshing import _periodic_axes, _plain_parts
    started = time.perf_counter()
    size_x, size_y = (float(length) for length in domain_size(spec, parameters))
    size_z = float(domain_size_z(spec, parameters))
    footprint = footprint_shapes(shapes)
    triangle_sizing = footprint_sizing(sizing)
    periodic = _periodic_axes(triangle_sizing)
    parts = _plain_parts(footprint.outlined, (0.0, 0.0, size_x, size_y), triangle_sizing, periodic,
                         size_field=footprint_size_field(triangle_sizing, footprint, sizing.edge_sizes))
    meshed = time.perf_counter()
    z_edges = np.asarray(sizing.layer_edges, dtype=np.float64)
    order_key = periodic_order_key(parts['points'], periodic) if periodic else None
    raw = swept_tetrahedron_mesh(parts['points'], parts['triangles'], z_edges,
                                 window=(0.0, 0.0, 0.0, size_x, size_y, size_z), order_key=order_key)
    split = time.perf_counter()
    raw = replace(raw, entities=_swept_entities(raw, shapes, footprint, parts, z_edges, size_z, names))
    region = raw.by_entity_majority(region_of(raw.inset_centroids()))
    timings.update(footprint_triangles=meshed - started, prism_split=split - meshed,
                   region_tags=time.perf_counter() - split)
    return replace(raw, region=np.asarray(region, dtype=np.int32), region_names=tuple(names))


def unswept_primitives(shapes, footprint, z_edges, size_z) -> tuple:
    """(set of primitive names tagged per element: no vertical walls, or a z face inside the window that is no layer
    plane, sorted z planes [m] bounding the entity bands: 0, size_z and the conforming z faces)."""
    slack = PLANE_TOLERANCE * max(float(size_z), float(np.ptp(z_edges)))
    unswept, planes = set(footprint.unswept), [0.0, float(size_z)]
    for name in footprint.outlined:
        faces = [float(value) for value in z_extent(name, shapes)
                 if math.isfinite(value) and slack < value < size_z - slack]
        if all(float(np.min(np.abs(z_edges - value))) <= slack for value in faces):
            planes += faces
        else:   # e.g. a film thinner than h_max / 4 whose faces merged into one snap plane
            unswept.add(name)
    return unswept, np.unique(planes)


def _swept_entities(mesh, shapes, footprint, parts, z_edges, size_z, names) -> np.ndarray:
    """int64 mesher volume per element: (PSLG face of its triangle, z band between conforming planes, membership of
    each region object holding an unswept primitive): the majority then squares faceted curved walls, while objects
    tagged per element keep their own tags."""
    triangles = np.asarray(parts['triangles'], dtype=np.int64)
    n_triangles = len(triangles)
    component = _pslg_faces(triangles, parts['constraint_edges'], parts['pml_faces'])
    layer, triangle = np.divmod(np.arange(mesh.n_elements, dtype=np.int64) // 3, n_triangles)
    unswept, planes = unswept_primitives(shapes, footprint, z_edges, size_z)
    band = np.searchsorted(planes, 0.5 * (z_edges[1:] + z_edges[:-1]))
    columns = [component[triangle], band[layer]]
    held = [name for name in names if unswept.intersection(primitive_names(name, shapes))]
    if held:
        inset = mesh.inset_centroids()
        columns += [contains_points_3d(name, shapes, *inset.T).astype(np.int64) for name in held]
    return _row_ids(columns)


def _row_ids(columns) -> np.ndarray:
    """int64 index of each row of the non-negative int64 columns among the distinct rows in lexicographic order (as
    np.unique(axis=0, return_inverse=True)): a mixed-radix key and a 1D unique while it fits in int64."""
    radices = [int(column.max()) + 1 if column.size else 1 for column in columns]
    if math.prod(radices) >= 2 ** 62:   # too many held objects: the row sort
        _, entity = np.unique(np.column_stack(columns), axis=0, return_inverse=True)
        return np.asarray(entity, dtype=np.int64).reshape(-1)
    key = np.zeros(len(columns[0]), dtype=np.int64)
    for column, radix in zip(columns, radices):
        key = key * radix + column
    _, entity = np.unique(key, return_inverse=True)
    return np.asarray(entity, dtype=np.int64).reshape(-1)


def _pslg_faces(triangles, constraint_edges, pml_faces) -> np.ndarray:
    """int64 per triangle: the connected region across non-constraint edges (a face of the PSLG); ring triangles
    (pml_faces != 0) each on their own."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n_triangles = len(triangles)
    n_points = int(triangles.max()) + 1
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(n_triangles, dtype=np.int64), 3)
    keys = edges[:, 0] * n_points + edges[:, 1]
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    shared = np.flatnonzero(sorted_keys[1:] == sorted_keys[:-1])
    first, second = owner[order[shared]], owner[order[shared + 1]]
    constraint = np.sort(np.asarray(constraint_edges, dtype=np.int64).reshape(-1, 2), axis=1)
    blocked = np.isin(sorted_keys[shared], constraint[:, 0] * n_points + constraint[:, 1])
    ring = np.asarray(pml_faces) != 0
    keep = ~blocked & ~ring[first] & ~ring[second]
    graph = coo_matrix((np.ones(int(keep.sum())), (first[keep], second[keep])), shape=(n_triangles, n_triangles))
    _, labels = connected_components(graph, directed=False)
    return np.asarray(labels, dtype=np.int64)


# ---- estimate (meshing._tetrahedron_estimate) ----
def swept_estimate(spec, parameters, shapes, sizing) -> tuple:
    """(n_elements, n_nodes, n_edges, smallest size [m], raster centres [x, y, z], elements per raster cell) of a swept
    mesh: footprint triangles by meshing.triangle_counts (as meshing._triangle_estimate, with the edge sizes), times
    3 per layer of sizing.layer_edges (exact); nodes / edges by fem.prism_layers.swept_counts from the disc Euler
    counts; elements spread over (pixel, layer)."""
    from .meshing import _thin_source_density, _thin_sources, triangle_counts, triangle_raster, triangle_size_field
    size_x, size_y = (float(length) for length in domain_size(spec, parameters))
    footprint = footprint_shapes(shapes)
    triangle_sizing = footprint_sizing(sizing)
    centres, pixel_sides = triangle_raster(size_x, size_y)
    # objects thinner than a pixel (and points / axis lines) by their pixel means, as meshing._triangle_estimate
    thin = _thin_sources(triangle_sizing, footprint.sized, pixel_sides)
    raster_sizing = replace(triangle_sizing, object_sizes={name: size for name, size in
                                                           triangle_sizing.object_sizes.items() if name not in thin})
    size = _with_edge_sizes(triangle_size_field(raster_sizing, footprint.sized, *centres), sizing, footprint,
                            centres, pixel_sides)
    density = 4.0 / (math.sqrt(3.0) * size ** 2)
    if thin:
        density = np.maximum(density, _thin_source_density(triangle_sizing, thin, centres, pixel_sides))
    counted_2d = triangle_counts(density, pixel_sides, triangle_sizing.pml_widths, triangle_sizing.pml_layers)
    per_pixel, along, corner_columns = counted_2d['per_pixel'], counted_2d['along'], counted_2d['corner_columns']
    n_triangles = max(int(math.ceil(counted_2d['window'])) + counted_2d['ring'], 2)
    n_boundary = sum(along.values()) + 2 * corner_columns   # outer boundary edges (the ring keeps the side counts)
    n_points = (n_triangles + n_boundary) // 2 + 1          # Euler of a triangulated disc: n = (t + b) / 2 + 1
    n_edges = (3 * n_triangles + n_boundary) // 2
    z_edges = np.asarray(sizing.layer_edges, dtype=np.float64)
    n_layers = len(z_edges) - 1
    counted = swept_counts(n_triangles, n_points, n_edges, n_layers)
    smallest = float(min(size.min(), np.diff(z_edges).min(), *(thin_size for _, thin_size in thin.values())))
    window_triangles = max(float(per_pixel.sum()), 1e-300)
    elements = per_pixel[:, :, None] * np.full(n_layers, counted['elements'] / (window_triangles * n_layers))
    raster = [centres[0], centres[1], 0.5 * (z_edges[1:] + z_edges[:-1])]
    return counted['elements'], counted['nodes'], counted['edges'], smallest, raster, elements


def _with_edge_sizes(size, sizing, footprint, centres, pixel_sides) -> np.ndarray:
    """The raster size field with each edge-size object's primitive outlines as sources (footprint_size_field's twin:
    s + (growth - 1) x distance to the rasterised outline; one distance transform per edge size), floored at h_min."""
    if not sizing.edge_sizes:
        return size
    from scipy.ndimage import binary_erosion

    from .meshing import SizedUnions
    from .triangle_meshing import _primitives
    x_centres, y_centres = (np.asarray(axis_centres, dtype=np.float64) for axis_centres in centres)
    slope = sizing.growth - 1.0
    size = size.copy()

    def add_source(distance, edge_size):
        np.minimum(size, edge_size + slope * distance, out=size)

    outlines = SizedUnions(size.shape)
    for name, edge_size in sizing.edge_sizes.items():
        shape = footprint.sized.get(name)
        if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
            continue
        for primitive in _primitives(name, footprint.sized):
            window, mask = rasterize_window(primitive.name, {primitive.name: primitive}, x_centres, y_centres, np)
            if mask.any():
                outlines.add(edge_size, window, mask & ~binary_erosion(mask))
    outlines.apply(add_source, pixel_sides)
    return np.maximum(size, sizing.h_min)
