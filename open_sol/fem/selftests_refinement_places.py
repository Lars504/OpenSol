"""Self-tests of RefinementSpec.where 'edges' / 'surface' on gmsh tetrahedra (plan D3 / D3b; CPU, specs() empty,
check({}) directly), the gmsh Cylinder field's half-length axis and the sliver count that spares graded orthoschemes;
gmsh meshes (skipped without gmsh) for the surface sizes and the cylinder field."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from ..model.spec import DomainSpec, GeometrySpec, MeshSpec, ModelSpec, RefinementSpec

_UM = 1e-6
_H_MAX = 0.1 * _UM
_GROWTH = 1.3
_SIZE = 0.02 * _UM               # refinement size of the pad / ring
_TWIN_RTOL = 1e-12               # analytic distances vs closed forms
_GEO_RTOL = 1e-9                 # MathEval texts (12 significant digits) vs the numpy twin
_WINDOW = (0.0, 0.0, 0.0, 1.2 * _UM, 1.2 * _UM, 0.4 * _UM)
_CENTRE = (0.6 * _UM, 0.6 * _UM)
_OUTER, _HOLE = 0.5 * _UM, 0.3 * _UM
_RING_SIZE = 0.05 * _UM
_NEAR_FACTOR = 1.5               # gmsh: mean edge of elements touching the hole surface <= this x the ring size
_CORE_VOLUME_RATIO = 0.5         # gmsh: hole-core mean element volume, surface-refined / unrefined
_ESTIMATE_FACTOR = 2.5           # gmsh: estimate / built elements within this factor (recorded)
_CYLINDER = (0.5 * _UM, 0.5 * _UM, 0.4 * _UM, 0.2 * _UM, 0.2 * _UM)   # sized z cylinder (cx, cy, z0, r, height)
_CYLINDER_SIZE = 0.03 * _UM
_ABOVE_FACTOR = 1.5              # gmsh: mean edge above the cylinder end (0.05 .. 0.1 um) >= this x its size


def _cylinder(name, radius, z0=-0.12 * _UM, height=0.64 * _UM):
    from .gmsh3d import Solid

    return Solid(name, 'cylinder', (_CENTRE[0], _CENTRE[1], z0, radius, height))


def _annulus(size=_RING_SIZE):
    """SizedRegion of the outer cylinder minus the hole (both through the window: the hole's ends are clipped)."""
    from .gmsh3d import SizedRegion

    return SizedRegion('difference', size, children=(SizedRegion('solid', size, solid=_cylinder('outer', _OUTER)),
                                                     SizedRegion('solid', size, solid=_cylinder('hole', _HOLE))))


def _ring_point(radius, angle=0.3, z=0.2 * _UM):
    return np.array([_CENTRE[0] + radius * math.cos(angle), _CENTRE[1] + radius * math.sin(angle), z])


def _surface_twin_checks(add):
    """SizedRegion.surface_distance of a box, a z cylinder and the annulus = closed-form distances to their surfaces
    (inside, outside, at corners); surface_sizes = size + slope x distance capped at h_max."""
    from .gmsh3d import SizedRegion, Solid

    um = _UM
    box = SizedRegion('solid', _SIZE, solid=Solid('box', 'box', (0.0, 0.0, 0.0, 1.0 * um, 1.0 * um, 0.4 * um)))
    box_cases = [((0.5, 0.5, 0.2), 0.2), ((0.1, 0.5, 0.2), 0.1), ((1.3, 0.5, 0.2), 0.3),
                 ((1.3, 1.4, 0.6), math.sqrt(0.09 + 0.16 + 0.04)), ((1.0, 0.3, 0.4), 0.0)]
    rod = SizedRegion('solid', _SIZE, solid=_cylinder('rod', 0.2 * um, 0.0, 0.4 * um))
    rod_cases = [(_ring_point(0.05 * um, z=0.2 * um), 0.15 * um), (_ring_point(0.1 * um, z=0.05 * um), 0.05 * um),
                 (_ring_point(0.5 * um, z=0.2 * um), 0.3 * um), (_ring_point(0.5 * um, z=0.8 * um), 0.5 * um)]
    ring_cases = [(_ring_point(0.0), _HOLE), (_ring_point(0.35 * um), 0.05 * um), (_ring_point(0.4 * um), 0.1 * um),
                  (_ring_point(0.6 * um), 0.1 * um), (_ring_point(_HOLE), 0.0)]
    errors = {}
    for label, region, cases, scale in (('box', box, box_cases, um), ('cylinder', rod, rod_cases, 1.0),
                                        ('annulus', _annulus(), ring_cases, 1.0)):
        points = np.array([np.asarray(point, dtype=np.float64) * scale for point, _ in cases])
        expected = np.array([distance for _, distance in cases]) * scale
        found = region.surface_distance(points)
        errors[label] = float(np.max(np.abs(found - expected)) / um)
    add('surface_distance_closed_form', max(errors.values()) <= _TWIN_RTOL, errors)
    points = np.array([_ring_point(radius) for radius in (0.0, 0.32 * um, 0.4 * um, 0.7 * um)])
    sizes = _annulus().surface_sizes(points, _GROWTH, _H_MAX)
    expected = np.minimum(_RING_SIZE + (_GROWTH - 1.0) * np.array([_HOLE, 0.02 * um, 0.1 * um, 0.2 * um]), _H_MAX)
    add('surface_sizes_ramp', float(np.max(np.abs(sizes / expected - 1.0))) <= 1e-9,
        {'sizes': sizes.tolist(), 'expected': expected.tolist()})


def _field_tree(geo):
    """(formulas, kinds, lists, background field number) parsed from a .geo text."""
    import re

    formulas = {int(number): text for number, text in re.findall(r'Field\[(\d+)\]\.F = "([^"]*)";', geo)}
    kinds = {int(number): kind for number, kind in re.findall(r'Field\[(\d+)\] = (\w+);', geo)}
    lists = {int(number): [int(k) for k in members.split(',')]
             for number, members in re.findall(r'Field\[(\d+)\]\.FieldsList = \{([^}]*)\};', geo)}
    background = re.search(r'Background Field = (\d+);', geo)
    return formulas, kinds, lists, int(background.group(1)) if background else None


def _surface_geo_checks(add):
    """geo_text of the annulus and a box as sized surfaces: one root each in the Min list, only MathEval / Min / Max
    fields below it (no Box / Ball / Cylinder step), no field references, strings within GEO_STRING_MAX; the tree
    evaluated in Python = SizedRegion.surface_sizes at 64 points."""
    import re

    from .gmsh3d import GEO_STRING_MAX, SizedRegion, Solid, geo_text

    um = _UM
    window = _WINDOW
    scale = max(window[3] - window[0], window[4] - window[1], window[5] - window[2])   # origin: the window corner 0
    box = SizedRegion('solid', _SIZE, solid=Solid('pad', 'box', (0.13 * um, 0.17 * um, 0.05 * um, 0.41 * um,
                                                                  0.37 * um, 0.31 * um)))
    regions = (_annulus(), box)
    geo = geo_text(window, window, [], _H_MAX, _H_MAX / 100.0, 2.0 * _H_MAX, _GROWTH, sized_surfaces=regions)
    formulas, kinds, lists, background = _field_tree(geo)
    roots = lists.get(background, []) if background is not None else []

    def below(number):
        return {number} | set().union(*(below(k) for k in lists.get(number, [])))

    used_kinds = {kinds.get(number) for root in roots for number in below(root)}
    add('surface_geo_tree', len(roots) == 2 and all(kinds.get(root) == 'Max' for root in roots)
        and used_kinds <= {'MathEval', 'Min', 'Max'} and not any(re.search(r'F\d', text) for text in formulas.values()),
        {'roots': roots, 'kinds': sorted(str(kind) for kind in used_kinds)})
    longest = max((len(text) for text in re.findall(r'"([^"]*)"', geo)), default=0)
    add('surface_geo_strings_fit', 0 < longest <= GEO_STRING_MAX, longest)

    def evaluate(number, point):
        if kinds[number] == 'MathEval':
            names = {'x': point[0] / scale, 'y': point[1] / scale, 'z': point[2] / scale}
            return eval(formulas[number], {'__builtins__': {}, 'Sqrt': math.sqrt}, names)   # noqa: S307 (own .geo)
        return (min if kinds[number] == 'Min' else max)(evaluate(k, point) for k in lists[number])

    if len(roots) == 2:
        points = np.random.default_rng(13).uniform(window[:3], window[3:], size=(64, 3))
        errors = []
        for root, region in zip(roots, regions):
            values = np.minimum(np.array([evaluate(root, point) for point in points]) * scale, _H_MAX)
            twin = region.surface_sizes(points, _GROWTH, _H_MAX)
            errors.append(float(np.max(np.abs(values - twin) / twin)))
        add('surface_matheval_equals_twin', max(errors) <= _GEO_RTOL, errors)


def _cylinder_axis_checks(add):
    """The Cylinder field of a sized z cylinder: ZAxis = half its height, centred at mid height; gmsh's rule
    |(X - centre) . axis| < |axis|^2 then agrees with Solid.contains along the axis (ends +- 10 % of the height)."""
    import re

    from .gmsh3d import Solid, geo_text

    solid = Solid('rod', 'cylinder', _CYLINDER, size=_CYLINDER_SIZE)
    window = (0.0, 0.0, 0.0, 1.0 * _UM, 1.0 * _UM, 1.0 * _UM)
    geo = geo_text(window, window, [solid], _H_MAX, _H_MAX / 100.0, 2.0 * _H_MAX, _GROWTH)
    settings = {name: float(value) * _UM for name, value in
                re.findall(r'Field\[\d+\]\.(XAxis|YAxis|ZAxis|XCenter|YCenter|ZCenter|Radius) = ([-0-9.e+]+);', geo)}
    cx, cy, z0, radius, height = _CYLINDER
    if not {'ZAxis', 'XCenter', 'YCenter', 'ZCenter', 'Radius'} <= set(settings):
        add('cylinder_field_half_axis', False, sorted(settings))
        return
    axis = np.array([settings.get('XAxis', 0.0), settings.get('YAxis', 0.0), settings['ZAxis']])
    centre = np.array([settings['XCenter'], settings['YCenter'], settings['ZCenter']])
    heights = z0 + height * np.array([-0.1, 0.1, 0.5, 0.9, 1.1])
    points = np.column_stack([np.full(5, cx), np.full(5, cy), heights])
    gmsh_inside = np.abs((points - centre) @ axis) < axis @ axis
    add('cylinder_field_half_axis', math.isclose(settings['ZAxis'], 0.5 * height, rel_tol=1e-9)
        and np.array_equal(gmsh_inside, solid.contains(points)),
        {'ZAxis_over_height': settings['ZAxis'] / height, 'gmsh_inside': gmsh_inside.tolist()})


def _sliver_count_checks(add):
    """tetrahedron_quality: a flat sliver counts as poor; a 20:1:1 Kuhn orthoscheme (radius ratio below the limit, every
    dihedral <= 90 deg) and a structured 20:1:1 brick mesh count none."""
    from .mesh3d import POOR_RADIUS_RATIO, largest_dihedral_cosines, structured_box_mesh, tetrahedron_quality

    sliver = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.02], [0.0, 1.0, 0.02]])
    needle = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [20.0, 1.0, 0.0], [20.0, 1.0, 1.0]])
    tets = np.array([[0, 1, 2, 3]])
    sliver_quality, needle_quality = tetrahedron_quality(sliver, tets), tetrahedron_quality(needle, tets)
    needle_cos = float(largest_dihedral_cosines(needle[None])[0])
    add('sliver_counted', sliver_quality.n_poor == 1, {'radius_ratio': sliver_quality.smallest})
    add('orthoscheme_not_counted', needle_quality.n_poor == 0 and needle_quality.smallest < POOR_RADIUS_RATIO
        and abs(needle_cos) <= 1e-12, {'radius_ratio': needle_quality.smallest, 'largest_dihedral_cos': needle_cos})
    graded = structured_box_mesh(np.linspace(0.0, 20.0, 3), np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    add('graded_bricks_not_counted', graded.quality.n_poor == 0 and graded.quality.smallest < POOR_RADIUS_RATIO,
        {'radius_ratio_min': graded.quality.smallest, 'n_poor': graded.quality.n_poor})


def _pad_spec(where=None, mesher='gmsh', kind='tetrahedra', target='pad'):
    """1 x 1 x 0.5 um window with a pad box and a small ball, one mesh refined at `target` (where None: unrefined)."""
    um = _UM
    spec = ModelSpec(title='Self-test: refinement places', domain=DomainSpec(size_x=1.0 * um, size_y=1.0 * um,
                                                                               size_z=0.5 * um))
    spec.geometry = [GeometrySpec('pad', 'box', {'x0': 0.3 * um, 'y0': 0.3 * um, 'z0': 0.1 * um, 'x1': 0.7 * um,
                                                 'y1': 0.7 * um, 'z1': 0.4 * um}),
                     GeometrySpec('ball', 'sphere', {'cx': 0.15 * um, 'cy': 0.15 * um, 'cz': 0.25 * um,
                                                     'r': 0.08 * um})]
    refinements = [] if where is None else [RefinementSpec(target, _SIZE, where=where)]
    spec.meshes = [MeshSpec(name='tets', kind=kind, mode='nonuniform', dx=_H_MAX, max_ratio=_GROWTH, mesher=mesher,
                            refinements=refinements)]
    return spec


def _sizing_checks(add):
    """tetrahedron_sizing routes 'edges' into edge_sizes (the physics' edge-size path) and 'surface' into surface_sizes;
    the gmsh estimate counts plain < edges < inside and plain < surface < inside; JSON round trip keeps where."""
    from ..model.meshing import estimate_mesh, tetrahedron_sizing
    from ..model.schema import get_registry

    registry = get_registry()
    counts, sizings = {}, {}
    for where in (None, 'inside', 'edges', 'surface'):
        spec = _pad_spec(where)
        sizings[where] = tetrahedron_sizing(spec, spec.meshes[0], {}, registry)
        counts[str(where)] = estimate_mesh(spec, {}, registry, 'tets').n_elements
    edges, surface = sizings['edges'], sizings['surface']
    add('edges_into_edge_sizes', edges.edge_sizes == {'pad': _SIZE} and not edges.object_sizes
        and not edges.surface_sizes, {'edge_sizes': edges.edge_sizes, 'object_sizes': edges.object_sizes})
    add('surface_into_surface_sizes', surface.surface_sizes == {'pad': _SIZE} and not surface.object_sizes
        and not surface.edge_sizes, {'surface_sizes': surface.surface_sizes})
    add('estimate_order', counts['None'] < counts['edges'] < counts['inside']
        and counts['None'] < counts['surface'] < counts['inside'], counts)
    spec = _pad_spec('surface')
    loaded = ModelSpec.from_dict(spec.to_dict())
    add('json_round_trip', loaded.meshes[0].refinements[0].where == 'surface', loaded.meshes[0].refinements[0].where)


def _refusal_checks(add):
    """where 'edges' / 'surface' refused by the builtin tetrahedra sizing and a 3D tensor estimate; validate: an error
    on builtin, a warning for a sphere's edges on gmsh, none about where for a gmsh box."""
    from ..model.meshing import estimate_mesh, tetrahedron_sizing
    from ..model.schema import get_registry
    from ..model.validate import validate

    registry = get_registry()
    refused = {}
    builtin = _pad_spec('edges', mesher='builtin')
    tensor = _pad_spec('surface', mesher='builtin', kind='tensor')
    for label, call in (('builtin', lambda: tetrahedron_sizing(builtin, builtin.meshes[0], {}, registry)),
                        ('tensor', lambda: estimate_mesh(tensor, {}, registry, 'tets'))):
        try:
            call()
            refused[label] = ''
        except ValueError as exc:
            refused[label] = str(exc)
    add('sizing_refuses', all("where '" in text for text in refused.values()), refused)

    def where_issues(spec, level):
        return [issue.message for issue in validate(spec, registry)
                if issue.level == level and "where '" in issue.message and tuple(issue.path[:1]) == ('mesh',)]

    builtin_errors = where_issues(builtin, 'error')
    sphere_warnings = where_issues(_pad_spec('edges', target='ball'), 'warning')
    box_issues = where_issues(_pad_spec('surface'), 'error') + where_issues(_pad_spec('surface'), 'warning')
    add('validate_where', bool(builtin_errors) and any('spheres have no edges' in text for text in sphere_warnings)
        and not box_issues, {'builtin': builtin_errors, 'sphere': sphere_warnings, 'box': box_issues})


def _gmsh_checks(add):
    """With gmsh: the annulus refined at its surface (model path) has fine elements at the hole surface, a coarse hole
    core, an estimate within _ESTIMATE_FACTOR; a sized z cylinder is coarse again 0.05 .. 0.1 um above its end (the
    half-length Cylinder axis). Skipped without gmsh."""
    from ..model.meshing import estimate_mesh
    from ..model.schema import get_registry
    from ..model.tetrahedral_meshing import tetrahedralize
    from .gmsh3d import Solid, mesh_solids_with_gmsh
    from .gmsh_io import find_gmsh

    if find_gmsh() is None:
        add('gmsh_surface_sizes', True, 'skipped: gmsh not found (OPEN_SOL_GMSH or PATH)')
        return
    um = _UM
    registry = get_registry()

    def ring_spec(where):
        window = _WINDOW
        spec = ModelSpec(title='Self-test: surface refinement',
                         domain=DomainSpec(size_x=window[3], size_y=window[4], size_z=window[5]))
        cylinder = {'cx': _CENTRE[0], 'cy': _CENTRE[1], 'z0': 0.0, 'z1': window[5]}
        spec.geometry = [GeometrySpec('outer', 'cylinder_z', dict(cylinder, r=_OUTER)),
                         GeometrySpec('hole', 'cylinder_z', dict(cylinder, r=_HOLE)),
                         GeometrySpec('ring', 'difference', {'base': 'outer', 'subtract': ['hole']})]
        refinements = [] if where is None else [RefinementSpec('ring', _RING_SIZE, where=where)]
        spec.meshes = [MeshSpec(name='tets', kind='tetrahedra', mode='nonuniform', dx=_H_MAX, max_ratio=_GROWTH,
                                mesher='gmsh', refinements=refinements)]
        return spec

    def stats(built):
        corners = built.points[built.tets]
        centroids = corners.mean(axis=1)
        radial = np.hypot(centroids[:, 0] - _CENTRE[0], centroids[:, 1] - _CENTRE[1])
        volumes = np.abs(np.linalg.det(corners[:, 1:] - corners[:, :1])) / 6.0
        edges = np.mean([np.linalg.norm(corners[:, a] - corners[:, b], axis=1)
                         for a in range(4) for b in range(a + 1, 4)], axis=0)
        node_radial = np.hypot(built.points[:, 0] - _CENTRE[0], built.points[:, 1] - _CENTRE[1])
        touching = (np.abs(node_radial - _HOLE) <= 1e-3 * _HOLE)[built.tets].sum(axis=1) >= 3
        return radial, volumes, edges, touching

    spec = ring_spec('surface')
    mesh = tetrahedralize(spec, {}, registry, 'tets')
    plain = tetrahedralize(ring_spec(None), {}, registry, 'tets')
    radial, volumes, edges, touching = stats(mesh)
    plain_radial, plain_volumes, _, _ = stats(plain)
    core, plain_core = radial < 0.05 * um, plain_radial < 0.05 * um
    near_edge = float(edges[touching].mean()) if touching.any() else None
    core_ratio = (float(volumes[core].mean() / plain_volumes[plain_core].mean())
                  if core.any() and plain_core.any() else None)
    estimate = estimate_mesh(spec, {}, registry, 'tets').n_elements
    ratio = estimate / len(mesh.tets)
    add('gmsh_surface_sizes', near_edge is not None and near_edge <= _NEAR_FACTOR * _RING_SIZE
        and core_ratio is not None and core_ratio >= _CORE_VOLUME_RATIO,
        {'hole_surface_mean_edge': near_edge, 'size': _RING_SIZE, 'core_volume_ratio': core_ratio,
         'elements': int(len(mesh.tets)), 'unrefined': int(len(plain.tets))})
    add('gmsh_surface_estimate', 1.0 / _ESTIMATE_FACTOR <= ratio <= _ESTIMATE_FACTOR,
        {'estimate_over_built': ratio, 'estimate': estimate})
    cx, cy, z0, radius, height = _CYLINDER
    window = (0.0, 0.0, 0.0, 1.0 * um, 1.0 * um, 1.0 * um)
    rod = Solid('rod', 'cylinder', _CYLINDER, size=_CYLINDER_SIZE)
    built = mesh_solids_with_gmsh(window, [rod], _H_MAX, growth=_GROWTH)
    corners = built.points[built.tets]
    centroids = corners.mean(axis=1)
    rod_edges = np.mean([np.linalg.norm(corners[:, a] - corners[:, b], axis=1)
                         for a in range(4) for b in range(a + 1, 4)], axis=0)
    axial = np.hypot(centroids[:, 0] - cx, centroids[:, 1] - cy) < 0.5 * radius
    top = z0 + height
    above = axial & (centroids[:, 2] > top + 0.05 * um) & (centroids[:, 2] < top + 0.1 * um)
    inside = axial & (centroids[:, 2] > z0 + 0.03 * um) & (centroids[:, 2] < top - 0.03 * um)
    above_edge = float(rod_edges[above].mean()) if above.any() else None
    inside_edge = float(rod_edges[inside].mean()) if inside.any() else None
    add('gmsh_cylinder_field_length', above_edge is not None and above_edge >= _ABOVE_FACTOR * _CYLINDER_SIZE,
        {'above_mean_edge': above_edge, 'inside_mean_edge': inside_edge, 'size': _CYLINDER_SIZE})


def _check_refinement_places(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    checks = SubChecks()
    add = checks.add

    for group in (_surface_twin_checks, _surface_geo_checks, _cylinder_axis_checks, _sliver_count_checks,
                  _sizing_checks, _refusal_checks, _gmsh_checks):
        group(add)
    return checks_result(checks)


REFINEMENT_PLACES_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_tetrahedron_refinement_places',
        label="Tetrahedron meshes: refinement along edges / at the surface, cylinder field, sliver count",
        description=(
            "Plans D3 / D3b (CPU, check({})): SizedRegion.surface_distance of a box, a z cylinder and an annulus = "
            f'closed-form distances to their surfaces ({_TWIN_RTOL:g} um), surface_sizes = size + (growth - 1) x '
            'distance capped at h_max; the .geo sized surfaces are one Max root each over MathEval / Min / Max only, '
            f'no field references, strings within GEO_STRING_MAX, evaluated in Python = the twin ({_GEO_RTOL:g}); '
            "the Cylinder field's ZAxis is half the height and gmsh's inside rule then matches Solid.contains; a "
            'flat sliver counts as poor, a 20:1:1 Kuhn orthoscheme and brick mesh (radius ratio below the limit) '
            "do not; tetrahedron_sizing puts where 'edges' into edge_sizes and 'surface' into surface_sizes, the "
            "gmsh estimate counts unrefined < edges / surface < inside, the JSON round trip keeps where; 'edges' "
            'refused by builtin tetrahedra and a tensor mesh (sizing and validate), a sphere warned. With gmsh: '
            f'the annulus refined at its surface has elements touching the hole surface <= {_NEAR_FACTOR:g} x the '
            f'size, a hole core >= {_CORE_VOLUME_RATIO:g} of the unrefined mean element volume, estimate / built '
            f'within {_ESTIMATE_FACTOR:g}x; a sized z cylinder has mean edges >= {_ABOVE_FACTOR:g} x its size 0.05 .. '
            '0.1 um above its end; skipped without gmsh. error = failed sub-checks, tolerance 0. Written, not run.'),
        specs=lambda: {},
        check=_check_refinement_places,
        max_wall_time=300.0),
)
