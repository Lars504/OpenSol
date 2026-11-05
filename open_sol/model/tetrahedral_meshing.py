"""Tetrahedron meshes of 'tetrahedra' MeshSpecs (igor3d_plan T4b): the model's 3D primitives as gmsh solids (the only
reader of shape internals for the 3D mesher), the gmsh mesher or the builtin Kuhn split, element regions tagged by the
builder's 3D inclusion rules (geometry.contains_points_3d at clamped centroids, majority per gmsh volume:
TetrahedronMesh.by_entity_majority, as the builder's object masks), object interface face tags (mesh.object_faces from
element_membership, the builder's mask rule). Host numpy only."""
# gmsh solids: spheres, boxes, z cylinders, untapered z-prism circles (cylinders), rectangles of any axis (boxes) and
# untapered z rounded rectangles (two cross boxes + four corner cylinders, fragmented together like any overlap),
# composite inputs and array copies included (as triangle_meshing's outlines: resolve_geometry lists every copy as a
# shape), so their surfaces are element faces. Other primitives
# (ellipses, polygons, x / y circles and rounded rectangles, tapers) are not passed: their elements are tagged per
# centroid (a staircase of element size; refine them). Open prism ends run SOLID_OVERHANG beyond the meshed box (the fragments outside are
# dropped). Sizes: a plain object's refinement size applies to its solids, a composite's to the composite only
# (sized_regions: a gmsh3d.SizedRegion field tree, so a hole or the subtracted part stays at h_max; inputs without a gmsh
# solid drop out of the tree), an edge size (a physics' mesh_edge_sizes: electrode edges) to the edges of every
# primitive under the object (hole rims are edges of the difference); the other primitives have no gmsh edges.
# RefinementSpec.where 'edges' sizes join those edge sizes (meshing._element_sizing), 'surface' sizes become
# sized_surfaces (gmsh3d: size + slope x |distance to the object's surface|, plain objects and composites alike).
# MeshSpec.face_sizes (TetrahedronSizing.face_sizes: window faces, object bounding-box faces) and size_points
# [x, y, z, h] act through gmsh fields for any object kind; the builtin Kuhn split takes the sizing's graded axis edges
# (meshing.builtin_axis_edges: refinements, axes options and face_sizes along whole axes, flat object faces snapped).
# MeshSpec.sweep_axis 'z' dispatches to model.swept_meshing (the 2D footprint swept through z layers, plan D5).
import time
from dataclasses import replace

import numpy as np

from ..fem.gmsh3d import Solid, SizedRegion, mesh_solids_with_gmsh
from ..fem.mesh3d import interface_faces, structured_box_mesh
from .geometry import (GEOMETRY_KINDS, composite_inputs, contains_points_3d, domain_entities, domain_size,
                       domain_size_z, primitive_names, resolve_geometry, rounded_corners)
from .schema import get_registry

SOLID_OVERHANG = 0.1   # x the longest box side: how far an open prism end reaches beyond the box
_SLIVER = 1e-9         # x a rounded rectangle's longer side: narrower cross boxes / closer corner centres are dropped


def tetrahedralize(spec, parameters, registry=None, mesh=None, *, shapes=None, sizing=None, log=None):
    """fem.mesh3d.TetrahedronMesh of a 'tetrahedra' mesh (its name; None = meshing.default_mesh_name): sizes from
    meshing.tetrahedron_sizing unless given, shapes a strict resolve_geometry unless given; regions = the model's domain
    objects (domain_entities order; the last holding an element wins), object_faces = each object's interface faces;
    gmsh meshes the sizing's periodic axes with periodic surfaces (the builtin Kuhn split is periodic already).
    log(level, text) gets one info line of the stage times. ValueError on model errors, RuntimeError when gmsh fails."""
    from .meshing import default_mesh_name, sizing_axis_edges, tetrahedron_box, tetrahedron_sizing

    shapes = resolve_geometry(spec, parameters) if shapes is None else shapes
    if sizing is None:
        mesh_spec = spec.mesh_by_name(default_mesh_name(spec) if mesh is None else mesh)
        registry = registry if registry is not None else get_registry()
        sizing = tetrahedron_sizing(spec, mesh_spec, parameters, registry, shapes)
    sizes = (*domain_size(spec, parameters), domain_size_z(spec, parameters))
    window = (0.0, 0.0, 0.0, *(float(size) for size in sizes))
    names = [name for name in domain_entities(spec) if name != 'all' and name in shapes]

    def region_of(centroids):
        region = np.full(len(centroids), -1, dtype=np.int32)
        for index, name in enumerate(names):
            region[contains_points_3d(name, shapes, *np.asarray(centroids, dtype=np.float64).T)] = index
        return region

    widths = sizing.pml_widths
    started = time.perf_counter()
    timings = {}
    if getattr(sizing, 'sweep_axis', ''):   # swept prism layers (plan D5): the 2D mesher (builtin / gmsh) in-plane
        from .swept_meshing import swept_tetrahedra
        built = swept_tetrahedra(spec, parameters, shapes, sizing, names, region_of, timings)
    elif sizing.mesher == 'gmsh':
        low, high = tetrahedron_box(spec, parameters, sizing)
        built = mesh_solids_with_gmsh(window, model_solids(shapes, sizing, low, high), sizing.h_max,
                                      pml_widths=widths, h_min=sizing.h_min, growth=sizing.growth,
                                      curvature_factor=sizing.curvature_factor, region_of=region_of,
                                      region_names=tuple(names), periodic_axes=sizing.periodic_axes,
                                      sized_faces=sizing.face_sizes, size_points=sizing.size_points,
                                      sized_regions=sized_regions(shapes, sizing, low, high), timings=timings,
                                      sized_surfaces=sized_surfaces(shapes, sizing, low, high))
    else:
        raw = structured_box_mesh(*sizing_axis_edges(sizing, sizes), window=window)   # meshing.builtin_axis_edges
        split = time.perf_counter()
        built = replace(raw, region=region_of(raw.inset_centroids()), region_names=tuple(names))
        timings.update(kuhn_split=split - started, region_tags=time.perf_counter() - split)
    faces_started = time.perf_counter()
    object_faces = {name: interface_faces(built, element_membership(built, name, shapes)).astype(np.int32)
                    for name in names}
    timings['object_faces'] = time.perf_counter() - faces_started
    if log is not None:
        label = default_mesh_name(spec) if mesh is None else mesh
        log('info', _timing_text(label, built.n_elements, time.perf_counter() - started, timings))
    return replace(built, object_faces=object_faces)


def _timing_text(mesh_name, n_elements, total, timings) -> str:
    """Info line of the tetrahedralize stage times (gmsh or Kuhn split, region tags, mesh tables, object faces)."""
    sources = {'gmsh': '', 'disk': ' (read from the on-disk mesh cache)', 'memory': ' (kept in memory)'}
    parts = []
    if 'gmsh' in timings:
        parts.append(f"gmsh {timings['gmsh']:.1f} s{sources.get(timings.get('source', 'gmsh'), '')}")
    labels = (('kuhn_split', 'Kuhn split'), ('footprint_triangles', 'footprint triangles'),
              ('prism_split', 'prism split'), ('region_tags', 'region tags'), ('mesh_tables', 'mesh tables'),
              ('object_faces', 'object faces'))
    parts += [f'{label} {timings[key]:.1f} s' for key, label in labels if key in timings]
    return f"mesh '{mesh_name}': {n_elements:,} tetrahedra in {total:.1f} s ({', '.join(parts)})"


def element_membership(mesh, name, shapes) -> np.ndarray:
    """Bool per element of a domain object: geometry.contains_points_3d at the inset centroids, majority per mesher
    volume (TetrahedronMesh.by_entity_majority); the one rule of the builder's masks and the object_faces tags."""
    return np.asarray(mesh.by_entity_majority(contains_points_3d(name, shapes, *mesh.inset_centroids().T)), dtype=bool)


def model_solids(shapes, sizing, low, high) -> list:
    """fem.gmsh3d Solids of the plain 3D domain primitives (module header), composite inputs and array copies
    included; each takes its own refinement size (composites: sized_regions) and the smallest edge size of the objects
    holding it (0: none)."""
    overhang = SOLID_OVERHANG * float(np.max(high - low))
    edge_sizes = _primitive_sizes(shapes, sizing.edge_sizes)
    solids = []
    for name, shape in shapes.items():
        kind = GEOMETRY_KINDS[shape.kind]
        if kind.entity_kind != 'domain' or kind.composite:
            continue
        for solid_kind, solid_values in _solids(shape, low - overhang, high + overhang):
            solids.append(Solid(name, solid_kind, solid_values, float(sizing.object_sizes.get(name, 0.0)),
                                float(edge_sizes.get(name, 0.0))))
    return solids


def sized_regions(shapes, sizing, low, high) -> tuple:
    """fem.gmsh3d SizedRegions of the refined composite domain objects (union, intersection, difference, array): the
    size holds inside the composite only, not over its inputs' whole volumes (plan D1)."""
    overhang = SOLID_OVERHANG * float(np.max(high - low))
    regions = []
    for name, size in sizing.object_sizes.items():
        shape = shapes.get(name)
        if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
            continue
        if GEOMETRY_KINDS[shape.kind].composite:
            region = _sized_region(name, shapes, float(size), low - overhang, high + overhang)
            if region is not None:
                regions.append(region)
    return tuple(regions)


def sized_surfaces(shapes, sizing, low, high) -> tuple:
    """fem.gmsh3d SizedRegions (plain objects too) of sizing.surface_sizes (RefinementSpec.where 'surface', plan D3b):
    the size holds at the object's surface, growing to both sides; objects without a gmsh solid drop out."""
    overhang = SOLID_OVERHANG * float(np.max(high - low))
    regions = []
    for name, size in sizing.surface_sizes.items():
        shape = shapes.get(name)
        if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
            continue
        region = _sized_region(name, shapes, float(size), low - overhang, high + overhang)
        if region is not None:
            regions.append(region)
    return tuple(regions)


def primitives_without_solids(name, shapes, edges=False) -> list:
    """Names of the plain domain primitives under a domain object that gmsh gets no solid of (tagged per element, so
    no size field of theirs); edges=True also lists those without gmsh edges (spheres). [] for other objects."""
    shape = shapes.get(name)
    if shape is None or GEOMETRY_KINDS[shape.kind].entity_kind != 'domain':
        return []
    corner = np.zeros(3)
    return [primitive for primitive in primitive_names(name, shapes)
            if not _solids(shapes[primitive], corner, corner + 1.0)
            or (edges and all(kind == 'sphere' for kind, _ in _solids(shapes[primitive], corner, corner + 1.0)))]


def _sized_region(name, shapes, size, low, high):
    """SizedRegion of a domain object (array copies as a union), None when no part of it has a gmsh solid; inputs
    without one drop out (a dropped subtracted input leaves its part refined)."""
    shape = shapes.get(name)
    if shape is None:
        return None
    kind = GEOMETRY_KINDS[shape.kind]
    if not kind.composite:
        leaves = [SizedRegion('solid', size, solid=Solid(name, solid_kind, values))
                  for solid_kind, values in _solids(shape, low, high)]
        return _joined_region('union', leaves, size)
    children = [_sized_region(item, shapes, size, low, high) for item in composite_inputs(kind, shape.values)]
    if shape.kind == 'difference':
        if not children or children[0] is None:
            return None
        return _joined_region('difference', [children[0], *(child for child in children[1:] if child is not None)],
                              size)
    joined = 'intersection' if shape.kind == 'intersection' else 'union'
    return _joined_region(joined, [child for child in children if child is not None], size)


def _joined_region(kind, children, size):
    """SizedRegion of kind over children (the child itself when one, None when none)."""
    if not children:
        return None
    return children[0] if len(children) == 1 else SizedRegion(kind, size, children=tuple(children))


def _primitive_sizes(shapes, object_sizes) -> dict:
    """Primitive name -> the smallest size of the domain objects (object_sizes: object -> size) holding it."""
    sizes = {}
    for name, size in object_sizes.items():
        if name in shapes and GEOMETRY_KINDS[shapes[name].kind].entity_kind == 'domain':
            for primitive in primitive_names(name, shapes):
                sizes[primitive] = min(sizes.get(primitive, np.inf), size)
    return sizes


def _solids(shape, low, high) -> list:
    """[(gmsh3d kind, values)] of a primitive gmsh meshes conformally ([] for the others): one solid, or for an
    untapered z rounded rectangle (r > 0) the union of its two cross boxes (zero-width ones dropped) and the four corner
    cylinders (coincident ones once); open prism ends at low / high."""
    values = shape.values
    if shape.kind != 'rounded_rectangle' or not values['r'] > 0.0:
        solid = _solid(shape, low, high)
        return [] if solid is None else [solid]
    if values.get('sidewall_angle') or values.get('axis', 'z') != 'z':
        return []   # gmsh3d cylinders stand along z only: x / y rounded prisms and tapers are tagged per element
    start = values['z0'] if values.get('z0') is not None else float(low[2])
    end = values['z1'] if values.get('z1') is not None else float(high[2])
    x0, y0, x1, y1, radius = values['x0'], values['y0'], values['x1'], values['y1'], values['r']
    sliver = _SLIVER * max(x1 - x0, y1 - y0)   # a stadium's zero-width box / coincident corner centres
    solids = [('box', box) for box in ((x0 + radius, y0, start, x1 - radius, y1, end),
                                       (x0, y0 + radius, start, x1, y1 - radius, end))
              if box[3] - box[0] > sliver and box[4] - box[1] > sliver]
    centres = []
    for cx, cy, _ in rounded_corners(values):
        if all(abs(cx - other_x) > sliver or abs(cy - other_y) > sliver for other_x, other_y in centres):
            centres.append((cx, cy))
            solids.append(('cylinder', (cx, cy, start, radius, end - start)))
    return solids


def _solid(shape, low, high):
    """(gmsh3d kind, values) of a primitive gmsh meshes conformally, None for the others; open prism ends at low /
    high (a rounded rectangle with r = 0 is a rectangle)."""
    values = shape.values
    if shape.kind == 'sphere':
        return 'sphere', (values['cx'], values['cy'], values['cz'], values['r'])
    if shape.kind == 'box':
        return 'box', tuple(values[key] for key in ('x0', 'y0', 'z0', 'x1', 'y1', 'z1'))
    axis = values.get('axis', 'z')
    if values.get('sidewall_angle'):
        return None
    along_index = 'xyz'.index(axis)
    start = values['z0'] if values.get('z0') is not None else float(low[along_index])
    end = values['z1'] if values.get('z1') is not None else float(high[along_index])
    if shape.kind in ('cylinder_z', 'circle') and axis == 'z':
        return 'cylinder', (values['cx'], values['cy'], start, values['r'], end - start)
    if shape.kind not in ('rectangle', 'rounded_rectangle'):   # rounded here only with r = 0 (_solids)
        return None
    # the rectangle's (x, y) are model axes geometry.plane_axes(axis): (x, y) for z, (y, z) for x, (x, z) for y
    first, second = {'z': (0, 1), 'x': (1, 2), 'y': (0, 2)}[axis]
    corner_low, corner_high = [0.0] * 3, [0.0] * 3
    corner_low[first], corner_high[first] = values['x0'], values['x1']
    corner_low[second], corner_high[second] = values['y0'], values['y1']
    corner_low[along_index], corner_high[along_index] = start, end
    return 'box', (*corner_low, *corner_high)
