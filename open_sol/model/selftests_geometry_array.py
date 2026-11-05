"""Self-tests of the array geometry kind (CPU, specs() empty, check({}) directly): resolved copies, bucketed rasters
against the union of the copies, validation messages, 3D copies, and a 3 x 2 array of circles meshed like the same six
circles written out (same mesh digest). Written, not run: tolerances are first estimates."""
import math

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import SubChecks
from .spec import DomainSpec, GeometrySpec, ModelSpec

_UM = 1e-6
_WINDOW = (2.0 * _UM, 1.4 * _UM)
_HOLE = {'cx': 0.4 * _UM, 'cy': 0.45 * _UM, 'r': 0.17 * _UM}
_PITCH = (0.6 * _UM, 0.5 * _UM)     # 3 x 2 holes at x 0.4 / 1.0 / 1.6 um, y 0.45 / 0.95 um
_HOLE_ARRAY = {'base': 'hole', 'nx': 3, 'ny': 2, 'pitch_x': _PITCH[0], 'pitch_y': _PITCH[1]}
_H = 0.08 * _UM                     # triangle mesh h_max [m]
_RASTER_PIXEL = 5e-9                # raster checks: centres (k + 0.37) pixel
_DOMAIN_3D = (1.2 * _UM, 1.0 * _UM, 0.8 * _UM)
_SECTION_MISMATCH = 1e-3            # rasterize_box (float32 centres) vs contains_points_3d (float64), share of cells
_AREA_RTOL = 1e-12                  # array vs written-out circles: the same elements


def _spec(objects, size=_WINDOW, size_z=None) -> ModelSpec:
    spec = ModelSpec(title='Self-test: geometry array',
                     domain=DomainSpec(size_x=size[0], size_y=size[1], size_z=size_z))
    spec.geometry = [GeometrySpec(name, kind, dict(params)) for name, kind, params in objects]
    return spec


def _shapes(objects, size=_WINDOW, size_z=None):
    from .geometry import resolve_geometry

    return resolve_geometry(_spec(objects, size, size_z), {})


def _union_of_copies(name, shapes, test):
    """OR of test(copy) over the array's resolved copies (the generic union the fast paths must equal)."""
    result = None
    for copy in shapes[name].values['inputs']:
        mask = np.asarray(test(copy), dtype=bool)
        result = mask if result is None else result | mask
    return result


# ---- resolution ----
def _resolve_checks(add):
    """Copy names / order / exact moved values, entities, bounding box, a composite base and a nested array."""
    from .geometry import bounding_box, consumed_objects, domain_entities, rasterize

    objects = [('hole', 'circle', _HOLE), ('holes', 'array', _HOLE_ARRAY)]
    spec = _spec(objects)
    shapes = _shapes(objects)
    expected = ['hole'] + [f'holes[{i},{j}]' for j in range(2) for i in range(3) if (i, j) != (0, 0)]
    add('copy_names', shapes['holes'].values['inputs'] == expected, shapes['holes'].values['inputs'])
    add('resolved_order', list(shapes) == ['hole', 'holes'] + expected[1:], list(shapes))
    moved = all(shapes[f'holes[{i},{j}]'].values['cx'] == _HOLE['cx'] + i * _PITCH[0]
                and shapes[f'holes[{i},{j}]'].values['cy'] == _HOLE['cy'] + j * _PITCH[1]
                and shapes[f'holes[{i},{j}]'].values['r'] == _HOLE['r']
                for j in range(2) for i in range(3) if (i, j) != (0, 0))
    add('copies_moved_exactly', moved)
    add('entities', domain_entities(spec) == ['all', 'holes'] and consumed_objects(spec) == {'hole'},
        {'domain': domain_entities(spec), 'consumed': sorted(consumed_objects(spec))})
    boxes = np.array([bounding_box(copy, shapes) for copy in expected])
    union_box = (boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max())
    add('bounding_box', tuple(bounding_box('holes', shapes)) == tuple(union_box), list(bounding_box('holes', shapes)))
    um = _UM
    nested_objects = [('disc', 'circle', {'cx': 0.2 * um, 'cy': 0.2 * um, 'r': 0.15 * um}),
                      ('core', 'circle', {'cx': 0.2 * um, 'cy': 0.2 * um, 'r': 0.07 * um}),
                      ('ring', 'difference', {'base': 'disc', 'subtract': ['core']}),
                      ('rings', 'array', {'base': 'ring', 'nx': 2, 'ny': 2, 'pitch_x': 0.4 * um, 'pitch_y': 0.4 * um}),
                      ('blocks', 'array', {'base': 'rings', 'nx': 2, 'ny': 1, 'pitch_x': 0.9 * um,
                                           'pitch_y': 1.0 * um})]
    nested = _shapes(nested_objects)
    corner = nested.get('disc@rings[1,1]@blocks[1,0]')
    add('nested_copy_moved', corner is not None and nested['blocks[1,0]'].kind == 'array'
        and corner.values['cx'] == (0.2 * um + 1 * (0.4 * um)) + 1 * (0.9 * um)
        and nested['rings[1,1]@blocks[1,0]'].values['subtract'] == ['core@rings[1,1]@blocks[1,0]'],
        None if corner is None else corner.values['cx'])
    x = (np.arange(400) + 0.37) * (_WINDOW[0] / 400)
    y = (np.arange(280) + 0.37) * (_WINDOW[1] / 280)
    X, Y = np.meshgrid(x, y, indexing='ij')
    expected_mask = np.zeros(X.shape, dtype=bool)
    for shift_x in (0.0, 0.4 * um, 0.9 * um, 1.3 * um):
        for shift_y in (0.0, 0.4 * um):
            distance_squared = (X - (0.2 * um + shift_x)) ** 2 + (Y - (0.2 * um + shift_y)) ** 2
            expected_mask |= (distance_squared <= (0.15 * um) ** 2) & (distance_squared > (0.07 * um) ** 2)
    nested_mask = np.asarray(rasterize('blocks', nested, X, Y, np), dtype=bool)
    mismatched = int((nested_mask != expected_mask).sum())
    add('nested_raster_closed_form', mismatched <= 4 and nested_mask.any(), mismatched)   # rounding at the rims only


# ---- bucketed rasters ----
def _raster_checks(add):
    """_array_mask (per-copy tests near each point) equals the union of the copies exactly on float64 and float32
    centres for rounded rectangles, overlapping rotated ellipses and polygons; rasterize_window and
    triangle_meshing.contained agree; the overlap warning."""
    from .geometry import _array_mask, array_overlap, rasterize, rasterize_window
    from .triangle_meshing import contained

    um = _UM
    size = (2.0 * um, 1.0 * um)
    objects = [('pad', 'rounded_rectangle', {'x0': 0.02 * um, 'y0': 0.03 * um, 'x1': 0.07 * um, 'y1': 0.06 * um,
                                             'r': 0.01 * um}),
               ('pads', 'array', {'base': 'pad', 'nx': 30, 'ny': 20, 'pitch_x': 0.065 * um, 'pitch_y': 0.045 * um}),
               ('grain', 'ellipse', {'cx': 0.1 * um, 'cy': 0.1 * um, 'a': 0.09 * um, 'b': 0.03 * um, 'angle': 30.0}),
               ('grains', 'array', {'base': 'grain', 'nx': 12, 'ny': 9, 'pitch_x': 0.15 * um, 'pitch_y': 0.1 * um}),
               ('tooth', 'polygon', {'points': [(0.01 * um, 0.01 * um), (0.2 * um, 0.02 * um),
                                                (0.05 * um, 0.1 * um)]}),
               ('teeth', 'array', {'base': 'tooth', 'nx': 8, 'ny': 8, 'pitch_x': 0.24 * um, 'pitch_y': 0.12 * um})]
    shapes = _shapes(objects, size=size)
    names = ('pads', 'grains', 'teeth')
    for dtype in (np.float64, np.float32):
        x = ((np.arange(int(round(size[0] / _RASTER_PIXEL))) + 0.37) * _RASTER_PIXEL).astype(dtype)
        y = ((np.arange(int(round(size[1] / _RASTER_PIXEL))) + 0.37) * _RASTER_PIXEL).astype(dtype)
        X, Y = np.meshgrid(x, y, indexing='ij')
        for name in names:
            label = f'{name}_{np.dtype(dtype).name}'
            union = _union_of_copies(name, shapes, lambda copy: rasterize(copy, shapes, X, Y, np))
            bucketed = _array_mask(shapes[name], shapes, X, Y, np)
            add(f'{label}_bucketed_equals_copies', bucketed is not None and union.any()
                and np.array_equal(bucketed, union), None if bucketed is None else int((bucketed != union).sum()))
            window, cropped = rasterize_window(name, shapes, x, y, np)
            windowed = np.zeros(X.shape, dtype=bool)
            windowed[window] = cropped
            add(f'{label}_window_equals_copies', np.array_equal(windowed, union), int((windowed != union).sum()))
    points = np.random.default_rng(3).uniform((0.0, 0.0), size, size=(20000, 2))
    for name in names:
        union = _union_of_copies(name, shapes, lambda copy: rasterize(copy, shapes, points[:, 0], points[:, 1], np))
        direct = np.asarray(rasterize(name, shapes, points[:, 0], points[:, 1], np), dtype=bool)
        add(f'{name}_contained_equals_copies', np.array_equal(contained(name, shapes, points), union)
            and np.array_equal(direct, union), int(union.sum()))
    warnings = {name: array_overlap(name, shapes) for name in names}
    add('overlap_warning', warnings['grains'] is not None and 'overlap along x' in warnings['grains']
        and warnings['pads'] is None and warnings['teeth'] is None, warnings)


# ---- validation ----
def _validation_checks(add):
    """Errors for nx 0, pitch 0, no base, a point base, nz in 2D, a base used twice; a warning for overlapping copies;
    nothing at a good array; nz > 1 without pitch_z refused at resolve (3D)."""
    from .validate import validate

    circle = dict(_HOLE)
    objects = [('b_good', 'circle', circle), ('good', 'array', dict(_HOLE_ARRAY, base='b_good')),
               ('b_close', 'circle', circle), ('close', 'array', dict(_HOLE_ARRAY, base='b_close', pitch_x=0.3 * _UM)),
               ('b_count', 'circle', circle), ('zero_count', 'array', dict(_HOLE_ARRAY, base='b_count', nx=0)),
               ('b_pitch', 'circle', circle), ('zero_pitch', 'array', dict(_HOLE_ARRAY, base='b_pitch', pitch_y=0.0)),
               ('no_base', 'array', dict(_HOLE_ARRAY, base='')),
               ('spot', 'point', {'x': 0.5 * _UM, 'y': 0.5 * _UM}),
               ('point_base', 'array', dict(_HOLE_ARRAY, base='spot')),
               ('b_z', 'circle', circle), ('stacked', 'array', dict(_HOLE_ARRAY, base='b_z', nz=2, pitch_z=0.1 * _UM)),
               ('b_twice', 'circle', circle), ('first', 'array', dict(_HOLE_ARRAY, base='b_twice')),
               ('second', 'array', dict(_HOLE_ARRAY, base='b_twice'))]
    issues = {}
    for issue in validate(_spec(objects)):
        if len(issue.path) == 2 and issue.path[0] == 'geometry':
            issues.setdefault(issue.path[1], []).append((issue.level, issue.message))

    def has(name, level, text):
        return any(found_level == level and text in message for found_level, message in issues.get(name, []))

    add('good_array_clean', 'good' not in issues, issues.get('good'))
    add('overlap_warned', has('close', 'warning', 'overlap'), issues.get('close'))
    add('zero_count_refused', has('zero_count', 'error', 'nx'), issues.get('zero_count'))
    add('zero_pitch_refused', has('zero_pitch', 'error', 'pitch_y'), issues.get('zero_pitch'))
    add('missing_base_refused', has('no_base', 'error', 'base'), issues.get('no_base'))
    add('point_base_refused', has('point_base', 'error', 'not a domain object'), issues.get('point_base'))
    add('nz_in_2d_refused', has('stacked', 'error', 'nz'), issues.get('stacked'))
    add('base_used_twice_refused', has('second', 'error', 'already an input'), issues.get('second'))
    try:
        _shapes([('post', 'circle', dict(_HOLE, z0=0.1 * _UM, z1=0.3 * _UM)),
                 ('posts', 'array', dict(_HOLE_ARRAY, base='post', nz=2))], size=_DOMAIN_3D[:2], size_z=_DOMAIN_3D[2])
        add('nz_without_pitch_z_refused', False, 'no error')
    except ValueError as exc:
        add('nz_without_pitch_z_refused', 'pitch_z' in str(exc), str(exc))


# ---- 3D ----
def _three_d_checks(add):
    """z copies: names, z extent, contains_points_3d (bucketed along z, and an open-ended base bucketed in xy) equal to
    the union of the copies, rasterize_box vs the point tests, one gmsh solid per copy."""
    from .geometry import contains_points_3d, rasterize_box, z_extent
    from .meshing import TetrahedronSizing
    from .tetrahedral_meshing import model_solids

    um = _UM
    # every post end on a 0.02 um cell edge (pitch_z 0.36): an end on a centre flips a whole layer between the float32
    # section compare and the float64 point test
    objects = [('post', 'circle', {'cx': 0.15 * um, 'cy': 0.15 * um, 'r': 0.08 * um, 'z0': 0.1 * um, 'z1': 0.3 * um}),
               ('posts', 'array', {'base': 'post', 'nx': 3, 'ny': 3, 'nz': 2, 'pitch_x': 0.35 * um,
                                   'pitch_y': 0.3 * um, 'pitch_z': 0.36 * um}),
               ('bar', 'rectangle', {'x0': 0.05 * um, 'y0': 0.7 * um, 'x1': 0.2 * um, 'y1': 0.75 * um}),
               ('bars', 'array', {'base': 'bar', 'nx': 4, 'ny': 3, 'pitch_x': 0.28 * um, 'pitch_y': 0.09 * um})]
    shapes = _shapes(objects, size=_DOMAIN_3D[:2], size_z=_DOMAIN_3D[2])
    inputs = shapes['posts'].values['inputs']
    add('z_copy_names', len(inputs) == 18 and inputs[9] == 'posts[0,0,1]' and inputs[-1] == 'posts[2,2,1]',
        inputs[9:11])
    extent = z_extent('posts', shapes)
    add('z_extent', extent == (0.1 * um, 0.3 * um + 1 * (0.36 * um)), list(extent))
    points = np.random.default_rng(5).uniform((0.0, 0.0, 0.0), _DOMAIN_3D, size=(30000, 3))
    for name in ('posts', 'bars'):
        union = _union_of_copies(name, shapes, lambda copy: contains_points_3d(copy, shapes, *points.T))
        inside = contains_points_3d(name, shapes, *points.T)
        add(f'{name}_points_equal_copies', np.array_equal(inside, union) and union.any(), int(union.sum()))
    centres = [((np.arange(int(round(length / (0.02 * um)))) + 0.5) * (0.02 * um)).astype(np.float32)
               for length in _DOMAIN_3D]
    X, Y, Z = np.meshgrid(*(axis.astype(np.float64) for axis in centres), indexing='ij')
    window, cropped = rasterize_box('posts', shapes, *centres, np)
    sections = np.zeros(X.shape, dtype=bool)
    sections[window] = cropped
    point_mask = contains_points_3d('posts', shapes, X.ravel(), Y.ravel(), Z.ravel()).reshape(X.shape)
    share = float((sections != point_mask).mean())
    add('sections_match_points', share <= _SECTION_MISMATCH and point_mask.any(), share)
    sizing = TetrahedronSizing(h_max=0.1 * um, h_min=0.001 * um, growth=1.3, curvature_factor=0.3, object_sizes={},
                               pml_widths={face: 0.0 for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')},
                               pml_layers=8, mesher='gmsh')
    solids = model_solids(shapes, sizing, np.zeros(3), np.array(_DOMAIN_3D))
    kinds = {}
    for solid in solids:
        owner = 'posts' if solid.name == 'post' or solid.name.startswith('posts[') else 'bars'
        kinds.setdefault(owner, []).append(solid.kind)
    add('one_solid_per_copy', kinds.get('posts') == ['cylinder'] * 18 and kinds.get('bars') == ['box'] * 12,
        {owner: len(found) for owner, found in kinds.items()})


def _check_geometry(results_by_name):
    checks = SubChecks()
    add = checks.add
    _resolve_checks(add)
    _raster_checks(add)
    _validation_checks(add)
    _three_d_checks(add)
    return checks.result()


# ---- triangle mesh: array vs the same circles written out ----
def _sizing():
    from .meshing import TriangleSizing

    return TriangleSizing(h_max=_H, h_min=_H / 100.0, growth=1.3, curvature_factor=0.3, object_sizes={},
                          pml_widths={face: 0.0 for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}, pml_layers=4,
                          mesher='builtin')


def _check_triangle_mesh(results_by_name):
    from .triangle_meshing import element_masks, triangulate

    checks = SubChecks()
    add = checks.add
    written_names = [f'c{i}{j}' for j in range(2) for i in range(3)]
    written = [(f'c{i}{j}', 'circle', dict(_HOLE, cx=_HOLE['cx'] + i * _PITCH[0], cy=_HOLE['cy'] + j * _PITCH[1]))
               for j in range(2) for i in range(3)]
    written_spec = _spec(written)
    array_spec = _spec([('hole', 'circle', _HOLE), ('holes', 'array', _HOLE_ARRAY)])
    written_mesh = triangulate(written_spec, {}, sizing=_sizing())
    array_mesh = triangulate(array_spec, {}, sizing=_sizing())
    add('same_mesh_digest', written_mesh.digest == array_mesh.digest,
        {'written_elements': int(written_mesh.n_elements), 'array_elements': int(array_mesh.n_elements)})
    add('array_is_one_region', list(array_mesh.region_names) == ['holes'], list(array_mesh.region_names))
    areas = np.abs(array_mesh.areas())
    array_area = float(areas[array_mesh.region == 0].sum())
    written_member = np.isin(written_mesh.region, [written_mesh.region_names.index(name) for name in written_names])
    written_area = float(np.abs(written_mesh.areas())[written_member].sum())
    exact = 6.0 * math.pi * _HOLE['r'] ** 2
    add('same_area', abs(array_area - written_area) <= _AREA_RTOL * exact and 0.95 * exact < array_area < exact,
        {'array': array_area, 'written': written_area, 'exact': exact})
    n_array_edges = len(array_mesh.edge_tags.get('holes', ()))
    n_written_edges = sum(len(written_mesh.edge_tags.get(name, ())) for name in written_names)
    add('outline_edges', n_array_edges == n_written_edges > 0, {'array': n_array_edges, 'written': n_written_edges})
    add('six_curves', len(array_mesh.curves) == 6, len(array_mesh.curves))
    window = (0.0, 0.0, *_WINDOW)
    centroids = array_mesh.points[array_mesh.triangles].mean(axis=1)
    array_masks = element_masks(_shapes([('hole', 'circle', _HOLE), ('holes', 'array', _HOLE_ARRAY)]), ['holes'],
                                window, centroids)
    written_masks = element_masks(_shapes(written), written_names, window, centroids)
    add('element_masks', np.array_equal(array_masks['holes'], np.logical_or.reduce(list(written_masks.values()))))
    return checks.result(elements=int(array_mesh.n_elements))


GEOMETRY_ARRAY_SELFTEST_CASES = (
    SelfTestCase(
        name='geometry_array',
        label='Array geometry: copies, rasters, validation, 3D',
        description=(
            'CPU checks of the array kind (model.geometry): a 3 x 2 array of circles resolves into copies named '
            "'holes[i,j]' (base first, i fastest) moved by exactly i pitch_x / j pitch_y, listed right after the "
            'array; only the array is a domain entity (the base consumed); its bounding box is that of the copies; a '
            'difference base and a nested array are copied whole (raster vs the closed-form annuli). The bucketed '
            'raster (each copy tested only near its points) equals the union of the copies exactly on float64 and '
            'float32 centres for 30 x 20 rounded rectangles, 12 x 9 overlapping rotated ellipses and 8 x 8 '
            'polygons, as do rasterize_window and triangle_meshing.contained; the overlap warning. validate: nx 0, '
            'pitch 0, no base, a point base, nz in 2D, a base used twice refused, overlapping copies warned. 3D: a '
            '3 x 3 x 2 post array (z extent, point tests bucketed along z equal the copies, rasterize_box vs '
            f'contains_points_3d share <= {_SECTION_MISMATCH:g}) and an open-ended bar array; one gmsh solid per '
            'copy. error = failed sub-checks, tolerance 0. Written, not run.'),
        specs=lambda: {},
        check=_check_geometry,
        max_wall_time=300.0),
    SelfTestCase(
        name='fem_array_triangle_mesh',
        label='Array triangle mesh: same mesh as the copies written out',
        description=(
            'CPU: triangulate (h 0.08 um, built-in mesher, no ring) of a 3 x 2 array of circles and of the same six '
            'circles written out: the same mesh digest (the meshers see every copy outline in the same order), the '
            'array is one region whose element area equals the six circle regions, as many outline edges, six '
            'curves, element masks equal. error = failed sub-checks, tolerance 0. Written, not run.'),
        specs=lambda: {},
        check=_check_triangle_mesh,
        max_wall_time=300.0),
)
