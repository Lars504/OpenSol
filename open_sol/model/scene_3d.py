"""3D preview scene of a 3D model (view_3d C; numpy only, GUI-thread safe): surface faces of every domain object in
its material's paint order, point / line glyphs and highlights; drawn by viz.scene_figure (no model import there).
Also the object outlines on one plane (a 2D model's drawing plane or a 3D cut) for the field web page."""
from dataclasses import dataclass, field

import numpy as np

from .geometry import (GEOMETRY_KINDS, UNION_KINDS, composite_inputs, consumed_objects, domain_size, domain_size_z,
                       expand_entities, is_array_copy, material_paint_order, material_selections, outline, point_z,
                       resolve_geometry, surface_faces, vertical_section)
from .spec import outer_faces

_FACE_BUDGET = 6000          # automatic n_segments keeps about this many faces over all round objects
_SEGMENTS_RANGE = (12, 48)   # automatic n_segments bounds


@dataclass(eq=False)
class SceneSolid:
    """Surface of one primitive (a composite's inputs carry the composite's name and material)."""
    name: str
    faces: list                # float64 (N, 3) polygons [m], clipped to the domain box
    edges: list                # float64 (M, 3) rim / corner polylines [m]
    material: int | None       # index into Scene3D.material_names of the winning material, None = none
    solid: bool = True         # False: subtracted / intersected composite input, drawn as dashed edges only
    highlighted: bool = False


@dataclass(eq=False)
class SceneGlyph:
    """A point (1, 3) or line (2, 3) object [m] with the category of the first enabled feature using it ('' none)."""
    name: str
    vertices: np.ndarray
    category: str = ''
    highlighted: bool = False


@dataclass(eq=False)
class Scene3D:
    size: tuple                                          # (size_x, size_y, size_z) [m] of the physical domain
    material_names: list = field(default_factory=list)
    background: list = field(default_factory=list)       # indices of materials selecting 'all'
    solids: list = field(default_factory=list)
    glyphs: list = field(default_factory=list)
    highlighted_faces: list = field(default_factory=list)   # outer face names; 'all' = the whole domain
    problems: list = field(default_factory=list)          # objects that failed to resolve or to build faces


def build_scene(spec, parameters, registry=None, highlight=(), n_segments=None) -> Scene3D:
    """Scene3D of a 3D model; ValueError for a 2D model or a bad domain size. highlight = entity names (selections
    expanded here); registry (optional) gives glyph categories; n_segments None = automatic from the object count."""
    size_x, size_y = domain_size(spec, parameters)
    size_z = domain_size_z(spec, parameters)
    if size_z is None:
        raise ValueError('the 3D preview needs a 3D model (domain size_z set)')
    scene = Scene3D(size=(size_x, size_y, size_z), material_names=[material.name for material in spec.materials])
    errors = {}
    shapes = resolve_geometry(spec, parameters, errors=errors)
    scene.problems.extend(f'{name}: {message}' for name, message in errors.items())
    try:
        highlighted = set(expand_entities(spec, highlight))
    except (ValueError, TypeError):
        highlighted = set()
    scene.highlighted_faces = [face for face in (*outer_faces(spec), 'all') if face in highlighted]
    owners, scene.background = _material_owners(spec)
    leaves = _domain_leaves(spec, shapes)
    if n_segments is None:
        n_round = sum(shapes[leaf].kind in ('circle', 'ellipse', 'cylinder_z', 'sphere')
                      or (shapes[leaf].kind == 'rounded_rectangle' and shapes[leaf].values['r'] > 0.0)
                      for _, leaf, _ in leaves)
        n_segments = int(np.clip(_FACE_BUDGET // max(1, n_round), *_SEGMENTS_RANGE))
    for top, leaf, solid in leaves:
        try:
            faces, edges = surface_faces(leaf, shapes, scene.size, n_segments)
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            scene.problems.append(f'{leaf}: {exc}')
            continue
        if faces or edges:
            scene.solids.append(SceneSolid(top, faces, edges, owners.get(top), solid,
                                           top in highlighted or leaf in highlighted))
    categories = _glyph_categories(spec, registry) if registry is not None else {}
    for name, shape in shapes.items():
        entity_kind = GEOMETRY_KINDS[shape.kind].entity_kind
        if entity_kind in ('point', 'line'):
            values = shape.values
            height = point_z(values, size_z)
            ends = [('x', 'y')] if entity_kind == 'point' else [('x0', 'y0'), ('x1', 'y1')]
            vertices = np.array([[values[x_key], values[y_key], height] for x_key, y_key in ends], dtype=np.float64)
            scene.glyphs.append(SceneGlyph(name, vertices, categories.get(name, ''), name in highlighted))
    return scene


def object_cut_targets(scene, full_fraction=0.99) -> dict:
    """{axis: middle [m] of the material objects' extent inside the domain along it}, leaving out on each axis the
    objects spanning >= full_fraction of the domain there (substrates, guides running through); axes with none
    left are missing (callers take the window middle)."""
    boxes = []
    for solid in scene.solids:
        if solid.solid and solid.faces and solid.material is not None:
            points = np.concatenate([np.asarray(face, dtype=np.float64).reshape(-1, 3) for face in solid.faces])
            boxes.append((points.min(axis=0), points.max(axis=0)))
    targets = {}
    for index, axis in enumerate('xyz'):
        kept = [(low[index], high[index]) for low, high in boxes
                if high[index] - low[index] < full_fraction * scene.size[index]]
        if kept:
            targets[axis] = 0.5 * (min(low for low, _ in kept) + max(high for _, high in kept))
    return targets


@dataclass(eq=False)
class PlaneOutline:
    """Outline polylines of one primitive on a plane, float64 (N, 3) [m] (2D models: z = 0); name = its top object."""
    name: str
    polylines: list
    material: int | None       # index into spec.materials of the winning material, None = none
    solid: bool = True         # False: subtracted / intersected composite input (drawn dashed)


def plane_outlines(spec, parameters, plane=None, n_segments=128) -> tuple:
    """([PlaneOutline] of every domain primitive crossing the plane, [SceneGlyph] of the point / line objects (2D
    models only), problems) on plane None = a 2D model's drawing plane, (axis, position [m]) a cut of a 3D model."""
    size_z = domain_size_z(spec, parameters)
    if (plane is None) != (size_z is None):
        raise ValueError('plane outlines: a 2D model takes plane None, a 3D model an (axis, position) cut, got '
                         f'{plane!r}')
    errors = {}
    shapes = resolve_geometry(spec, parameters, errors=errors)
    problems = [f'{name}: {message}' for name, message in errors.items()]
    owners, _ = _material_owners(spec)
    outlines = []
    for top, leaf, solid in _domain_leaves(spec, shapes):
        try:
            polylines = _plane_polylines(leaf, shapes, plane, size_z, n_segments)
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            problems.append(f'{leaf}: {exc}')
            continue
        if polylines:
            outlines.append(PlaneOutline(top, polylines, owners.get(top), solid))
    glyphs = []
    if plane is None:
        for name, shape in shapes.items():
            if GEOMETRY_KINDS[shape.kind].entity_kind in ('point', 'line'):
                vertices = outline(name, shapes)[0]
                glyphs.append(SceneGlyph(name, np.column_stack([vertices, np.zeros(len(vertices))])))
    return outlines, glyphs, problems


def _plane_polylines(name, shapes, plane, size_z, n_segments) -> list:
    """(N, 3) polylines of a primitive on the plane (None: the 2D drawing plane at z = 0; a z cut: its cross-section
    there; an x / y cut: geometry.vertical_section, whose u is the other horizontal axis)."""
    if plane is None or plane[0] == 'z':
        height = 0.0 if plane is None else float(plane[1])
        polylines = outline(name, shapes, n_segments, z=None if plane is None else height)
        return [np.column_stack([polyline, np.full(len(polyline), height)]) for polyline in polylines]
    axis, position = plane[0], float(plane[1])
    lifted = []
    for polyline in vertical_section(name, shapes, (axis, position), size_z, n_segments=n_segments):
        points = np.empty((len(polyline), 3), dtype=np.float64)
        points[:, 0 if axis == 'y' else 1] = polyline[:, 0]
        points[:, 1 if axis == 'y' else 0] = position
        points[:, 2] = polyline[:, 1]
        lifted.append(points)
    return lifted


def _domain_leaves(spec, shapes) -> list:
    """[(top object, primitive, solid)] of every drawn domain object (array copies are leaves of their array)."""
    consumed = consumed_objects(spec)
    tops = [name for name, shape in shapes.items() if GEOMETRY_KINDS[shape.kind].entity_kind == 'domain'
            and name not in consumed and not is_array_copy(name)]
    return [(top, leaf, solid) for top in tops for leaf, solid in _leaves(top, shapes, True)]


def _material_owners(spec) -> tuple:
    """({object name: index of the material that wins it (last in material_paint_order)}, [background indices])."""
    expanded = material_selections(spec)
    owners, background = {}, []
    for index in material_paint_order(expanded):
        if 'all' in expanded[index]:
            background.append(index)
        owners.update((entity, index) for entity in expanded[index] if entity != 'all')
    return owners, background


def _leaves(name, shapes, solid) -> list:
    """[(primitive name, solid)] of a domain object: union inputs keep `solid`, a difference's base keeps it and its
    subtracted objects are not solid, intersection inputs are not solid (their overlap has no exact surface here)."""
    shape = shapes[name]
    kind = GEOMETRY_KINDS[shape.kind]
    if not kind.composite:
        return [(name, solid)]
    if shape.kind == 'difference':
        base = [(shape.values['base'], solid)] if shape.values.get('base') else []
        children = base + [(item, False) for item in shape.values.get('subtract') or ()]
    else:
        children = [(item, solid and shape.kind in UNION_KINDS) for item in composite_inputs(kind, shape.values)]
    return [leaf for child, child_solid in children for leaf in _leaves(child, shapes, child_solid)]


def _glyph_categories(spec, registry) -> dict:
    """{point / line object name: FeatureType.category of the first enabled feature selecting it}."""
    categories = {}
    for physics in spec.physics:
        try:
            physics_type = registry.physics_type(physics.type)
        except KeyError:
            continue
        for feature in physics.features:
            if not feature.enabled:
                continue
            try:
                category = physics_type.feature_type(feature.type).category
                entities = expand_entities(spec, feature.selection)
            except (KeyError, ValueError, TypeError):
                continue
            for entity in entities:
                categories.setdefault(entity, category)
    return categories
