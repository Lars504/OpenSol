"""Self-tests of tetrahedron face / point size fields (new_modules_plan G1; CPU, specs() empty, check({}) directly):
the SizedFace distance, the gmsh MathEval / Ball rows against their numpy twins, the estimate's density, the MeshSpec
path and, when gmsh is found, element sizes near / far from a refined face and a size point."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from ..model.spec import DomainSpec, GeometrySpec, MeshFaceSizeSpec, MeshSpec, ModelSpec, RefinementSpec

_UM = 1e-6
_SIDE = 0.6 * _UM               # cube window [m]
_H_MAX = 0.1 * _UM
_GROWTH = 1.3
_FACE_SIZE = 0.03 * _UM         # element size at the z_hi face, within _BAND of it
_BAND = 0.03 * _UM
_POINT = (0.3 * _UM, 0.3 * _UM, 0.15 * _UM, 0.025 * _UM)   # size point [x, y, z, h]
_TWIN_RTOL = 1e-12              # MathEval text vs SizedFace.sizes, Ball rows vs size_point_function
_NEAR_FACTOR = 1.5              # gmsh: mean edge of elements within 2 / 3 band of the face <= this x its size (1.12)
_FAR_FACTOR = 0.5               # ... beyond every ramp >= this x h_max (1.0)
_POINT_FACTOR = 3.0             # ... the element holding the size point <= this x its size (2.57; 5.6 unrefined)


def _face():
    from .gmsh3d import SizedFace

    return SizedFace(2, _SIDE, (0.0, 0.0), (_SIDE, _SIDE), _FACE_SIZE, _BAND)


def _sizing(with_sources=True):
    """meshing.TetrahedronSizing of the cube (gmsh, no shell) with the z_hi face and the size point (or none)."""
    from ..model.meshing import TetrahedronSizing

    return TetrahedronSizing(
        h_max=_H_MAX, h_min=_H_MAX / 100.0, growth=_GROWTH, curvature_factor=0.3, object_sizes={},
        pml_widths={face: 0.0 for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')}, pml_layers=8,
        mesher='gmsh', size_points=np.array([_POINT]) if with_sources else np.zeros((0, 4)),
        face_sizes=(_face(),) if with_sources else ())


def _field_tree(geo):
    """(formulas, kinds, lists, background field number or None) parsed from a .geo text."""
    import re

    formulas = {int(number): text for number, text in re.findall(r'Field\[(\d+)\]\.F = "([^"]*)";', geo)}
    kinds = {int(number): kind for number, kind in re.findall(r'Field\[(\d+)\] = (\w+);', geo)}
    lists = {int(number): [int(k) for k in members.split(',')]
             for number, members in re.findall(r'Field\[(\d+)\]\.FieldsList = \{([^}]*)\};', geo)}
    background = re.search(r'Background Field = (\d+);', geo)
    return formulas, kinds, lists, int(background.group(1)) if background else None


def _field_value(tree, number, point, scale):
    """Normalised value of field `number` of a _field_tree at point [m] (origin 0, coordinates / scale): MathEval
    texts evaluated in Python (arithmetic and Sqrt), Min / Max fields over their lists."""
    formulas, kinds, lists, _ = tree
    if kinds[number] == 'MathEval':
        names = {'x': point[0] / scale, 'y': point[1] / scale, 'z': point[2] / scale}
        return eval(formulas[number], {'__builtins__': {}, 'Sqrt': math.sqrt}, names)   # noqa: S307 (own .geo)
    return (min if kinds[number] == 'Min' else max)(_field_value(tree, k, point, scale) for k in lists[number])


def _refusal_text(function) -> str:
    """str of the ValueError function() raises, '' when it raises none (other exceptions propagate)."""
    try:
        function()
    except ValueError as exc:
        return str(exc)
    return ''


def _distance_checks(add):
    """SizedFace.face_distance at points above, beside and diagonal to the rectangle."""
    from .gmsh3d import SizedFace

    um = _UM
    face = SizedFace(2, 0.5 * um, (0.0, 0.0), (1.0 * um, 1.0 * um), 0.05 * um, 0.0)
    points = np.array([[0.3, 0.4, 0.7], [1.3, 0.4, 0.5], [1.3, 1.4, 0.9], [-0.2, 0.5, 0.5]]) * um
    expected = np.array([0.2, 0.3, math.sqrt(0.09 + 0.16 + 0.16), 0.2]) * um
    error = float(np.max(np.abs(face.face_distance(points) - expected) / expected))
    add('sized_face_distance', error <= 1e-14, error)


def _geo_twin_checks(add):
    """The face's Max of two MathEval fields (evaluated in Python: arithmetic and Sqrt; the Max alone in the Min list,
    no field reference, every string within GEO_STRING_MAX, also for 17-digit coordinates) = SizedFace.sizes; the
    point's Ball (radius 0, linear ramp over Thickness) = meshing.size_point_function, at 64 points of the cube."""
    import re

    from ..model.meshing import size_point_function
    from .gmsh3d import GEO_STRING_MAX, SizedFace, geo_text

    bounds = (0.0, 0.0, 0.0, _SIDE, _SIDE, _SIDE)
    geo = geo_text(bounds, bounds, [], _H_MAX, _H_MAX / 100.0, 2.0 * _H_MAX, _GROWTH, sized_faces=(_face(),),
                   size_points=np.array([_POINT]))
    rows = geo.split('\n')
    formulas, kinds, lists, _ = _field_tree(geo)
    in_min = next((lists.get(number, []) for number, kind in kinds.items() if kind == 'Min'), [])
    maxima = [number for number in in_min if kinds.get(number) == 'Max']
    parts = lists.get(maxima[0], []) if len(maxima) == 1 else []
    add('geo_has_matheval', len(parts) == 2 and set(parts) == set(formulas) and not set(parts) & set(in_min)
        and not any(re.search(r'F\d', text) for text in formulas.values()), {'kinds': kinds, 'lists': lists})
    awkward = SizedFace(0, _SIDE / 3.0, (_SIDE / 7.0, _SIDE / 11.0), (_SIDE / 1.3, _SIDE / 1.7), _FACE_SIZE / 3.0,
                        _BAND / 3.0)
    awkward_geo = geo_text(bounds, bounds, [], _H_MAX, _H_MAX / 100.0, 2.0 * _H_MAX, _GROWTH, sized_faces=(awkward,))
    longest = max(len(text) for text in re.findall(r'"([^"]*)"', geo + awkward_geo))
    add('geo_strings_fit', longest <= GEO_STRING_MAX, longest)
    points = np.random.default_rng(7).uniform(0.0, _SIDE, size=(64, 3))
    scale = _SIDE

    def evaluate(x, y, z):
        names = {'x': x / scale, 'y': y / scale, 'z': z / scale}
        return max(eval(formulas[number], {'__builtins__': {}, 'Sqrt': math.sqrt}, names)   # noqa: S307 (own .geo)
                   for number in parts)

    if len(parts) == 2 and set(parts) <= set(formulas):
        values = np.array([evaluate(*point) for point in points]) * scale
        twin = _face().sizes(points, _GROWTH)
        error = float(np.max(np.abs(values - twin) / twin))
        add('matheval_equals_twin', error <= _TWIN_RTOL, error)
    ball = {}
    for row in rows:
        match = re.match(r'Field\[\d+\]\.(Radius|Thickness|XCenter|YCenter|ZCenter|VIn|VOut) = ([-0-9.e+]+);', row)
        if match:
            ball[match.group(1)] = float(match.group(2)) * scale
    add('geo_has_ball', {'Radius', 'Thickness', 'VIn', 'VOut'} <= set(ball) and 'Ball;' in geo, sorted(ball))
    if {'Thickness', 'VIn', 'VOut', 'XCenter', 'YCenter', 'ZCenter'} <= set(ball):
        distance = np.linalg.norm(points - np.array([ball['XCenter'], ball['YCenter'], ball['ZCenter']]), axis=1)
        values = np.minimum(ball['VIn'] + distance / ball['Thickness'] * (ball['VOut'] - ball['VIn']), ball['VOut'])
        twin = size_point_function(np.array([_POINT]), _GROWTH, _H_MAX)(points)
        error = float(np.max(np.abs(values - twin) / twin))
        add('ball_equals_twin', error <= _TWIN_RTOL, error)


def _density_checks(add):
    """meshing.tetrahedron_element_density on a 1 / 60 um voxel raster: h = the face size in the band, h_max beyond
    every ramp, finer near the size point; more elements with the sources than without."""
    from ..model.meshing import tetrahedron_element_density

    count = 60
    centres = [(np.arange(count) + 0.5) * (_SIDE / count)] * 3
    density, smallest = tetrahedron_element_density(_sizing(), {}, centres, _SIDE)
    plain, _ = tetrahedron_element_density(_sizing(False), {}, centres, _SIDE)
    voxel = _SIDE / count
    z = centres[2][None, None, :]
    x, y = np.meshgrid(centres[0], centres[1], indexing='ij')
    to_point = np.sqrt((x[:, :, None] - _POINT[0]) ** 2 + (y[:, :, None] - _POINT[1]) ** 2 + (z - _POINT[2]) ** 2)
    to_face = np.broadcast_to(_SIDE - z, density.shape)
    in_band = to_face <= _BAND - voxel
    ramp_face = _BAND + (_H_MAX - _FACE_SIZE) / (_GROWTH - 1.0) + voxel
    ramp_point = (_H_MAX - _POINT[3]) / (_GROWTH - 1.0) + voxel
    far = (to_face >= ramp_face) & (to_point >= ramp_point)
    band_error = float(np.max(np.abs(density[in_band] * _FACE_SIZE ** 3 - 1.0)))
    far_error = float(np.max(np.abs(density[far] * _H_MAX ** 3 - 1.0)))
    add('density_in_band', bool(in_band.any()) and band_error <= 1e-12, band_error)
    add('density_far', bool(far.any()) and far_error <= 1e-12, far_error)
    nearest = np.unravel_index(np.argmin(to_point), density.shape)
    add('density_at_point', density[nearest] > 10.0 * _H_MAX ** -3 and smallest < 2.0 * _POINT[3],
        {'density_ratio': float(density[nearest] * _H_MAX ** 3), 'smallest': smallest})
    add('density_counts_more', float(density.sum()) > 2.0 * float(plain.sum()),
        float(density.sum() / plain.sum()))
    return density, voxel


def _meshspec_checks(add):
    """MeshSpec.face_sizes / size_points through meshing.tetrahedron_sizing (window face, one object face, all six
    faces of an object clipped to the window) and estimate_mesh counting more elements with them."""
    from ..model.meshing import estimate_mesh, tetrahedron_sizing
    from ..model.schema import get_registry

    um = _UM
    registry = get_registry()
    spec = ModelSpec(title='Self-test: tetrahedron face sizes',
                     domain=DomainSpec(size_x=1.0 * um, size_y=1.0 * um, size_z=0.5 * um))
    spec.geometry = [GeometrySpec('slab', 'box', {'x0': 0.2 * um, 'y0': 0.2 * um, 'z0': 0.1 * um, 'x1': 0.8 * um,
                                                  'y1': 1.4 * um, 'z1': 0.2 * um})]
    plain_mesh = MeshSpec(name='tets', kind='tetrahedra', mode='uniform', dx=0.1 * um, mesher='gmsh')
    refined = MeshSpec(name='tets', kind='tetrahedra', mode='uniform', dx=0.1 * um, mesher='gmsh',
                       face_sizes=[MeshFaceSizeSpec('z_hi', 0.03 * um, distance=0.02 * um),
                                   MeshFaceSizeSpec('slab', 0.04 * um, face='z_lo'),
                                   MeshFaceSizeSpec('slab', 0.05 * um)],
                       size_points=[[0.5 * um, 0.5 * um, 0.25 * um, 0.02 * um]])
    spec.meshes = [refined]
    sizing = tetrahedron_sizing(spec, refined, {}, registry)
    faces = sizing.face_sizes
    top, slab_bottom = (faces[0], faces[1]) if len(faces) >= 2 else (None, None)
    add('meshspec_face_count', len(faces) == 8, len(faces))
    add('meshspec_window_face', top is not None and top.axis == 2 and math.isclose(top.coordinate, 0.5 * um)
        and top.low == (0.0, 0.0) and np.allclose(top.high, (1.0 * um, 1.0 * um)) and top.distance == 0.02 * um,
        None if top is None else repr(top))
    add('meshspec_object_face_clipped', slab_bottom is not None and math.isclose(slab_bottom.coordinate, 0.1 * um)
        and np.allclose(slab_bottom.low, (0.2 * um, 0.2 * um)) and np.allclose(slab_bottom.high, (0.8 * um, 1.0 * um)),
        None if slab_bottom is None else repr(slab_bottom))
    points = np.asarray(sizing.size_points)
    add('meshspec_size_point', points.shape == (1, 4) and 0.0 < points[0, 3] <= 0.02 * um, points.tolist())
    refined_count = estimate_mesh(spec, {}, registry, 'tets').n_elements
    spec.meshes = [plain_mesh]
    plain_count = estimate_mesh(spec, {}, registry, 'tets').n_elements
    add('meshspec_estimate_counts_more', refined_count > plain_count, {'refined': refined_count, 'plain': plain_count})


def _gmsh_checks(density, voxel, add):
    """With gmsh: the cube meshed with the face and the point; mean edge lengths near the face, beyond the ramps and
    at the point; the estimate / built element ratio recorded. Skipped (passes) without gmsh."""
    from ..model.meshing import TETRAHEDRON_FILL
    from .gmsh3d import mesh_solids_with_gmsh
    from .gmsh_io import find_gmsh

    if find_gmsh() is None:
        add('gmsh_face_sizes', True, 'skipped: gmsh not found (OPEN_SOL_GMSH or PATH)')
        return
    window = (0.0, 0.0, 0.0, _SIDE, _SIDE, _SIDE)
    mesh = mesh_solids_with_gmsh(window, [], _H_MAX, growth=_GROWTH, sized_faces=(_face(),),
                                 size_points=np.array([_POINT]))
    centroids, _, _, edges = _tet_stats(mesh)
    to_face = _SIDE - centroids[:, 2]
    to_point = np.linalg.norm(centroids - np.array(_POINT[:3]), axis=1)
    near = to_face <= 2.0 * _BAND / 3.0
    far = (to_face >= _BAND + (_H_MAX - _FACE_SIZE) / (_GROWTH - 1.0)) & (to_point >= (_H_MAX - _POINT[3])
                                                                         / (_GROWTH - 1.0))
    holding = int(np.argmin(to_point))
    estimate = TETRAHEDRON_FILL * 6.0 * math.sqrt(2.0) * voxel ** 3 * float(density.sum())
    values = {'near_mean_edge': float(edges[near].mean()) if near.any() else None,
              'far_mean_edge': float(edges[far].mean()) if far.any() else None,
              'point_element_edge': float(edges[holding]), 'elements': int(len(mesh.tets)),
              'estimate_over_built': estimate / len(mesh.tets)}
    add('gmsh_face_sizes', near.any() and far.any() and values['near_mean_edge'] <= _NEAR_FACTOR * _FACE_SIZE
        and values['far_mean_edge'] >= _FAR_FACTOR * _H_MAX
        and values['point_element_edge'] <= _POINT_FACTOR * _POINT[3], values)


def _check_size_fields(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    checks = SubChecks()
    add = checks.add

    _distance_checks(add)
    _geo_twin_checks(add)
    density, voxel = _density_checks(add)
    _meshspec_checks(add)
    _gmsh_checks(density, voxel, add)
    return checks_result(checks)


# ---- composite sizes (plan D1) and the on-disk gmsh cache (plan D4) ----
_RING_WINDOW = (0.0, 0.0, 0.0, 1.2 * _UM, 1.2 * _UM, 0.4 * _UM)
_RING_CENTRE = (0.6 * _UM, 0.6 * _UM)
_RING_OUTER = 0.5 * _UM
_RING_HOLE = 0.3 * _UM
_RING_SIZE = 0.05 * _UM          # element size inside the refined difference (hole centre: size + 0.3 x 0.3 um > h_max)
_RING_EDGE_FACTOR = 1.5          # gmsh: mean edge of ring elements <= this x the ring size
_HOLE_VOLUME_RATIO = 0.5         # gmsh: mean element volume in the hole core, refined / unrefined (sizing the inputs: ~0.125)
_ESTIMATE_FACTOR = 2.5           # gmsh: estimate / built elements within this factor (TETRAHEDRON_FILL re-check)


def _cylinder(name, radius, z0, height):
    from .gmsh3d import Solid

    return Solid(name, 'cylinder', (_RING_CENTRE[0], _RING_CENTRE[1], z0, radius, height))


def _leaf(solid):
    from .gmsh3d import SizedRegion

    return SizedRegion('solid', _RING_SIZE, solid=solid)


def _ring_point(radius, angle=0.3, z=0.2 * _UM):
    return np.array([_RING_CENTRE[0] + radius * math.cos(angle), _RING_CENTRE[1] + radius * math.sin(angle), z])


def _region_twin_checks(add):
    """SizedRegion.sizes of an annulus (cylinder minus cylinder): h_max at the hole centre, the size in the ring, the
    ramp inside the hole, continuous at the hole surface, the ramp outside up to h_max; a film hole's ends ignored
    (clip); a nested difference (De Morgan); the region bounds."""
    from .gmsh3d import Solid, SizedRegion

    um = _UM
    slope = _GROWTH - 1.0
    z0, height = -0.12 * um, 0.64 * um   # through the window: no end face near the mid plane

    def sizes(region, *points):
        return region.sizes(np.array(points), _GROWTH, _H_MAX)

    annulus = SizedRegion('difference', _RING_SIZE, children=(_leaf(_cylinder('outer', _RING_OUTER, z0, height)),
                                                               _leaf(_cylinder('hole', _RING_HOLE, z0, height))))
    centre = float(sizes(annulus, _ring_point(0.0))[0])
    add('region_hole_centre_h_max', abs(centre / _H_MAX - 1.0) <= _TWIN_RTOL, centre)
    ring = sizes(annulus, *(_ring_point(0.4 * um, angle) for angle in np.linspace(0.0, 6.0, 7)))
    ring_error = float(np.max(np.abs(ring / _RING_SIZE - 1.0)))
    add('region_ring_size', ring_error <= _TWIN_RTOL, ring_error)
    depth = 0.05 * um
    ramp = float(sizes(annulus, _ring_point(_RING_HOLE - depth))[0])
    ramp_error = abs(ramp / (_RING_SIZE + slope * depth) - 1.0)
    add('region_hole_ramp', ramp_error <= 1e-9, ramp_error)
    step = 1e-6 * _RING_HOLE
    below, above = (float(value) for value in sizes(annulus, _ring_point(_RING_HOLE - step),
                                                     _ring_point(_RING_HOLE + step)))
    add('region_continuous_at_hole', abs(below - above) <= 2.0 * slope * step + 1e-12 * _RING_SIZE, below - above)
    outside, far = (float(value) for value in sizes(annulus, _ring_point(0.55 * um), _ring_point(0.8 * um)))
    outside_error = abs(outside / (_RING_SIZE + slope * 0.05 * um) - 1.0)
    add('region_outside_ramp', outside_error <= 1e-9 and far == _H_MAX, {'outside': outside, 'far': far})
    film = SizedRegion('difference', _RING_SIZE, children=(_leaf(_cylinder('disc', _RING_OUTER, 0.1 * um, 0.1 * um)),
                                                            _leaf(_cylinder('hole', _RING_HOLE, 0.1 * um, 0.1 * um))))
    film_centre = float(sizes(film, _ring_point(0.0, z=0.15 * um))[0])   # without the clip: size + 0.3 x 0.05 um
    add('region_film_hole_ends_ignored', film_centre == _H_MAX, film_centre)
    slab = Solid('slab', 'box', (0.0, 0.0, 0.0, 1.2 * um, 1.2 * um, 0.4 * um))
    annular_cut = SizedRegion('difference', _RING_SIZE, children=(_leaf(_cylinder('cut', _RING_OUTER, z0, height)),
                                                                   _leaf(_cylinder('post', 0.2 * um, z0, height))))
    nested = SizedRegion('difference', _RING_SIZE, children=(_leaf(slab), annular_cut))
    post, gap = (float(value) for value in sizes(nested, _ring_point(0.1 * um), _ring_point(0.35 * um)))
    gap_expected = _RING_SIZE + slope * 0.15 * um
    add('region_nested_difference', abs(post / _RING_SIZE - 1.0) <= _TWIN_RTOL
        and abs(gap / gap_expected - 1.0) <= 1e-9, {'post': post, 'gap': gap, 'gap_expected': gap_expected})
    low, high = annulus.bounds()
    expected_low = np.array([_RING_CENTRE[0] - _RING_OUTER, _RING_CENTRE[1] - _RING_OUTER, z0])
    expected_high = np.array([_RING_CENTRE[0] + _RING_OUTER, _RING_CENTRE[1] + _RING_OUTER, z0 + height])
    add('region_bounds', np.allclose(low, expected_low, rtol=1e-12, atol=0.0)
        and np.allclose(high, expected_high, rtol=1e-12, atol=0.0), [low.tolist(), high.tolist()])


def _region_geo_checks(add):
    """geo_text of a box minus (a film hole, a box notch, a sphere pore): one Max root in the Min list over the base's
    Box field and one Min / MathEval per subtracted solid (faces at the base's box sides left out: 1, 5, 1 MathEvals),
    no field references, strings within GEO_STRING_MAX, the MathEvals (evaluated in Python) = size + slope x
    max(Solid.depth(clip), 0)."""
    import re

    from .gmsh3d import GEO_STRING_MAX, SizedRegion, Solid, geo_text

    um = _UM
    base = Solid('slab', 'box', (0.1 * um, 0.1 * um, 0.1 * um, 1.1 * um, 1.1 * um, 0.3 * um))
    subtracted = (_cylinder('hole', _RING_HOLE, 0.1 * um, 0.2 * um),   # ends on the slab faces: its side only
                  Solid('notch', 'box', (0.15 * um, 0.15 * um, 0.0, 0.35 * um, 0.35 * um, 0.25 * um)),   # z_lo beyond
                  Solid('pore', 'sphere', (0.85 * um, 0.85 * um, 0.2 * um, 0.1 * um)))
    region = SizedRegion('difference', _RING_SIZE, children=(_leaf(base), *(_leaf(solid) for solid in subtracted)))
    window = _RING_WINDOW
    geo = geo_text(window, window, [], _H_MAX, _H_MAX / 100.0, 2.0 * _H_MAX, _GROWTH, sized_regions=(region,))
    tree = _field_tree(geo)
    formulas, kinds, lists, background = tree
    in_min = lists.get(background, []) if background is not None else []
    root = in_min[0] if len(in_min) == 1 else None
    children = lists.get(root, []) if root is not None and kinds.get(root) == 'Max' else []

    def math_evals(number):
        if kinds.get(number) == 'MathEval':
            return 1
        return sum(math_evals(k) for k in lists.get(number, [])) if kinds.get(number) in ('Min', 'Max') else 0

    counts = [math_evals(number) for number in children[1:]]
    add('region_geo_tree', len(children) == 4 and kinds.get(children[0]) == 'Box' and counts == [1, 5, 1]
        and not any(re.search(r'F\d', text) for text in formulas.values()),
        {'kinds': kinds, 'lists': lists, 'counts': counts})
    longest = max((len(text) for text in re.findall(r'"([^"]*)"', geo)), default=0)
    add('region_geo_strings_fit', 0 < longest <= GEO_STRING_MAX, longest)
    scale = max(window[3] - window[0], window[4] - window[1], window[5] - window[2])   # origin: the window corner 0
    if len(children) == 4:
        points = np.random.default_rng(11).uniform(window[:3], window[3:], size=(64, 3))
        errors = []
        for number, solid in zip(children[1:], subtracted):
            values = np.array([_field_value(tree, number, point, scale) for point in points]) * scale
            twin = _RING_SIZE + (_GROWTH - 1.0) * np.maximum(solid.depth(points, base.bounds()), 0.0)
            errors.append(float(np.max(np.abs(values - twin) / twin)))
        add('region_matheval_equals_twin', max(errors) <= _TWIN_RTOL, errors)


def _ring_spec(refined, where='inside', title='Self-test: composite refinement'):
    """ModelSpec of the annulus 'ring' = 'outer' minus 'hole' (z cylinders through the window), gmsh tetrahedra, the
    ring refined to _RING_SIZE (RefinementSpec.where `where`) or not."""
    window = _RING_WINDOW
    spec = ModelSpec(title=title,
                     domain=DomainSpec(size_x=window[3], size_y=window[4], size_z=window[5]))
    cylinder = {'cx': _RING_CENTRE[0], 'cy': _RING_CENTRE[1], 'z0': 0.0, 'z1': window[5]}
    spec.geometry = [GeometrySpec('outer', 'cylinder_z', dict(cylinder, r=_RING_OUTER)),
                     GeometrySpec('hole', 'cylinder_z', dict(cylinder, r=_RING_HOLE)),
                     GeometrySpec('ring', 'difference', {'base': 'outer', 'subtract': ['hole']})]
    refinements = [RefinementSpec('ring', _RING_SIZE, where=where)] if refined else []
    spec.meshes = [MeshSpec(name='tets', kind='tetrahedra', mode='nonuniform', dx=_H_MAX, max_ratio=_GROWTH,
                            mesher='gmsh', refinements=refinements)]
    return spec


def _tet_stats(built):
    """(centroids (m, 3), their distance from the ring axis, volumes, mean edge lengths) per element of a mesh."""
    corners = built.points[built.tets]
    centroids = corners.mean(axis=1)
    radial = np.hypot(centroids[:, 0] - _RING_CENTRE[0], centroids[:, 1] - _RING_CENTRE[1])
    volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0
    edges = np.mean([np.linalg.norm(corners[:, a] - corners[:, b], axis=1)
                     for a in range(4) for b in range(a + 1, 4)], axis=0)
    return centroids, radial, volumes, edges


def _region_gmsh_checks(add):
    """With gmsh, the model path: the refined ring's hole core keeps about the unrefined element volume, the ring has
    the ring size, the estimate is within _ESTIMATE_FACTOR of the built count, one timing info line. Skipped without
    gmsh."""
    from ..model.meshing import TETRAHEDRON_FILL, estimate_mesh
    from ..model.schema import get_registry
    from ..model.tetrahedral_meshing import tetrahedralize
    from .gmsh_io import find_gmsh

    if find_gmsh() is None:
        add('gmsh_composite_sizes', True, 'skipped: gmsh not found (OPEN_SOL_GMSH or PATH)')
        return
    um = _UM
    registry = get_registry()
    lines = []
    spec = _ring_spec(True)
    mesh = tetrahedralize(spec, {}, registry, 'tets', log=lambda level, text: lines.append((level, text)))
    estimate = estimate_mesh(spec, {}, registry, 'tets').n_elements
    unrefined = tetrahedralize(_ring_spec(False), {}, registry, 'tets')
    centroids, radial, volumes, edges = _tet_stats(mesh)
    z = centroids[:, 2]
    _, plain_radial, plain_volumes, _ = _tet_stats(unrefined)
    core, plain_core = radial < 0.15 * um, plain_radial < 0.15 * um
    ring = ((radial > _RING_HOLE + 0.03 * um) & (radial < _RING_OUTER - 0.03 * um) & (z > 0.05 * um)
            & (z < _RING_WINDOW[5] - 0.05 * um))
    volume_ratio = float(volumes[core].mean() / plain_volumes[plain_core].mean()) if core.any() and plain_core.any() \
        else None
    add('gmsh_hole_not_refined', volume_ratio is not None and volume_ratio >= _HOLE_VOLUME_RATIO,
        {'core_volume_ratio': volume_ratio, 'elements': int(len(mesh.tets)), 'unrefined': int(len(unrefined.tets))})
    ring_edge = float(edges[ring].mean()) if ring.any() else None
    add('gmsh_ring_refined', ring_edge is not None and ring_edge <= _RING_EDGE_FACTOR * _RING_SIZE,
        {'ring_mean_edge': ring_edge, 'size': _RING_SIZE})
    ratio = estimate / len(mesh.tets)
    add('gmsh_estimate_matches', 1.0 / _ESTIMATE_FACTOR <= ratio <= _ESTIMATE_FACTOR,
        {'estimate_over_built': ratio, 'estimate': estimate, 'TETRAHEDRON_FILL': TETRAHEDRON_FILL})
    timing = [text for level, text in lines if level == 'info']
    add('gmsh_timing_logged', len(timing) == 1 and 'gmsh' in timing[0] and 'object faces' in timing[0], timing)


def _builtin_timing_checks(add):
    """tetrahedralize(log=) of a builtin mesh logs one info line with the Kuhn split and object faces times."""
    from ..model.schema import get_registry
    from ..model.tetrahedral_meshing import tetrahedralize

    um = _UM
    spec = ModelSpec(title='Self-test: tetrahedralize timing',
                     domain=DomainSpec(size_x=0.3 * um, size_y=0.3 * um, size_z=0.3 * um))
    spec.geometry = [GeometrySpec('cube', 'box', {'x0': 0.1 * um, 'y0': 0.1 * um, 'z0': 0.1 * um, 'x1': 0.2 * um,
                                                  'y1': 0.2 * um, 'z1': 0.2 * um})]
    spec.meshes = [MeshSpec(name='tets', kind='tetrahedra', mode='uniform', dx=0.1 * um, mesher='builtin')]
    lines = []
    tetrahedralize(spec, {}, get_registry(), 'tets', log=lambda level, text: lines.append((level, text)))
    add('builtin_timing_logged', len(lines) == 1 and lines[0][0] == 'info' and 'Kuhn split' in lines[0][1]
        and 'object faces' in lines[0][1], lines)


def _disk_cache_checks(add):
    """With gmsh and a temporary cache folder: the first mesh runs gmsh and writes one file, the second (memory cache
    cleared) comes from disk with the same digest / entities and no gmsh run, a third from memory, a corrupted file is
    re-meshed and rewritten. Skipped without gmsh."""
    import os
    import tempfile

    from . import gmsh3d
    from .gmsh_io import find_gmsh

    add('cache_folder_off', gmsh3d.mesh_cache_folder('') is None, gmsh3d.mesh_cache_folder(''))
    if find_gmsh() is None:
        add('gmsh_disk_cache', True, 'skipped: gmsh not found (OPEN_SOL_GMSH or PATH)')
        return
    um = _UM
    window = (0.0, 0.0, 0.0, 0.4 * um, 0.4 * um, 0.4 * um)
    solids = [gmsh3d.Solid('ball', 'sphere', (0.2 * um, 0.2 * um, 0.2 * um, 0.1 * um))]
    with tempfile.TemporaryDirectory(prefix='open_sol_mesh_cache_') as folder:
        def mesh_once(clear=True):
            if clear:
                with gmsh3d._GMSH_CACHE_LOCK:
                    gmsh3d._GMSH_CACHE.clear()
            timings, before = {}, gmsh3d.gmsh_run_count()
            built = gmsh3d.mesh_solids_with_gmsh(window, solids, 0.1 * um, growth=_GROWTH, disk_cache=folder,
                                                 timings=timings)
            return built, timings.get('source'), gmsh3d.gmsh_run_count() - before

        first, source, runs = mesh_once()
        files = sorted(name for name in os.listdir(folder) if name.endswith('.npz'))
        add('cache_first_runs_gmsh', source == 'gmsh' and runs == 1 and len(files) == 1,
            {'source': source, 'runs': runs, 'files': files})
        second, source, runs = mesh_once()
        same = (second.digest == first.digest and np.array_equal(second.entities, first.entities)
                and np.array_equal(second.region, first.region))
        add('cache_second_from_disk', source == 'disk' and runs == 0 and same,
            {'source': source, 'runs': runs, 'same_mesh': same})
        _, source, runs = mesh_once(clear=False)
        add('cache_third_from_memory', source == 'memory' and runs == 0, {'source': source, 'runs': runs})
        if files:
            path = os.path.join(folder, files[0])
            with open(path, 'wb') as handle:
                handle.write(b'not an npz file')
            fourth, source, runs = mesh_once()
            rewritten = gmsh3d._read_disk_mesh(path)
            add('cache_corrupt_file_remeshed', source == 'gmsh' and runs == 1 and rewritten is not None,
                {'source': source, 'runs': runs, 'rewritten': rewritten is not None,
                 'elements': [int(len(first.tets)), int(len(fourth.tets))]})


def _sub_checks(*groups):
    """Run the check groups (each add(name, passed, value)); error = number of failed sub-checks, tolerance 0."""
    checks = SubChecks()
    for group in groups:
        group(checks.add)
    return checks.result()


def _check_composite_sizes(results_by_name):
    return _sub_checks(_region_twin_checks, _region_geo_checks, _region_gmsh_checks)


def _check_mesh_cache(results_by_name):
    return _sub_checks(_builtin_timing_checks, _disk_cache_checks)


MESH_SIZES_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_tetrahedron_size_fields',
        label='Tetrahedron meshes: face and point size fields',
        description=(
            'CPU checks of MeshSpec.face_sizes / size_points on tetrahedra (new_modules_plan G1): SizedFace distances '
            '(1e-14); the gmsh .geo Max of two MathEval fields of a face (no field references, every string within '
            "gmsh's limit, evaluated in Python) and the radius-0 Ball of a size point equal their numpy twins "
            f'SizedFace.sizes / size_point_function ({_TWIN_RTOL:g}); the estimate density '
            'on a 1 / 60 um raster of a 0.6 um cube is the face size^-3 inside the band and h_max^-3 beyond every '
            'ramp (1e-12), finer at the point, > 2x the count without sources; tetrahedron_sizing of a window face, '
            'one object face and all six faces of an object clipped to the window, the size point [x, y, z, h], and '
            'estimate_mesh counting more elements. With gmsh: mean edge length of elements within 2 / 3 of the band '
            f'<= {_NEAR_FACTOR:g} x the face size, beyond the ramps >= {_FAR_FACTOR:g} h_max, the element holding '
            f'the point <= {_POINT_FACTOR:g} x its size (gmsh 4.15: 1.12, 1.0, 2.57; estimate / built ratio '
            'recorded), skipped without gmsh. error = failed sub-checks, tolerance 0.'),
        specs=lambda: {},
        check=_check_size_fields,
        max_wall_time=300.0),
    SelfTestCase(
        name='fem_tetrahedron_composite_sizes',
        label='Tetrahedron meshes: refinement of a composite sizes only the composite',
        description=(
            'Plan D1 (CPU, check({})): SizedRegion.sizes of a cylinder minus cylinder is h_max at the hole centre, the '
            'size in the ring, size + (growth - 1) x depth inside the hole and x distance outside (1e-9; cylinder '
            'leaves ramp, no Cylinder field step), continuous at the hole surface; a film hole whose ends lie on the film faces is not refined by them; a nested difference '
            'follows De Morgan; the bounds are the base\'s. The .geo of a box minus a film hole, a notch box and a '
            'sphere pore is one Max root over the base Box field and 1 / 5 / 1 MathEvals (faces at the base box left '
            f'out), no field references, strings within GEO_STRING_MAX, the MathEvals = the twin ({_TWIN_RTOL:g}). '
            'With gmsh (model path, refinement of a difference): the hole core keeps >= '
            f'{_HOLE_VOLUME_RATIO:g} of the unrefined mean element volume, ring elements <= {_RING_EDGE_FACTOR:g} x '
            f'the size, estimate / built within {_ESTIMATE_FACTOR:g}x (ratio recorded: TETRAHEDRON_FILL), one '
            'timing info line; skipped without gmsh. error = failed sub-checks, tolerance 0. Written, not run.'),
        specs=lambda: {},
        check=_check_composite_sizes,
        max_wall_time=300.0),
    SelfTestCase(
        name='fem_gmsh3d_disk_cache',
        label='Tetrahedron meshes: on-disk gmsh cache and stage timings',
        description=(
            'Plan D4 (CPU, check({})): tetrahedralize(log=) of a builtin mesh logs one info line of its stage times; '
            "mesh_cache_folder('') is off. With gmsh and a temporary cache folder: the first mesh runs gmsh and "
            'writes one .npz, the second (memory cache cleared) is read from disk without a gmsh run and has the same '
            'digest, entities and regions, a third comes from memory, a corrupted file is re-meshed and rewritten; '
            'skipped without gmsh. error = failed sub-checks, tolerance 0. Written, not run.'),
        specs=lambda: {},
        check=_check_mesh_cache,
        max_wall_time=300.0),
)
