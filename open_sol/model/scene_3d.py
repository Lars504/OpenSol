"""3D preview scene of a 3D model (view_3d C; numpy only, GUI-thread safe): surface faces of every domain object in
its material's paint order, point / line glyphs and highlights; drawn by viz.scene_figure (no model import there)."""
from dataclasses import dataclass, field

import numpy as np

from .geometry import (GEOMETRY_KINDS, UNION_KINDS, composite_inputs, consumed_objects, domain_size, domain_size_z,
                       expand_entities, is_array_copy, material_paint_order, material_selections, point_z,
                       resolve_geometry, surface_faces)
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
    consumed = consumed_objects(spec)
    tops = [name for name, shape in shapes.items() if GEOMETRY_KINDS[shape.kind].entity_kind == 'domain'
            and name not in consumed and not is_array_copy(name)]   # copies are leaves of their array
    leaves = [(top, leaf, solid) for top in tops for leaf, solid in _leaves(top, shapes, True)]
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
