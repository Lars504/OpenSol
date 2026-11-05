"""Self-tests of the rounded_rectangle geometry kind and fem.mesh.Curve arc pieces (new_modules_plan G4; CPU, specs()
empty, check({}) directly): rasters, outlines, sections and 3D rules against closed forms, the arc PSLG, triangle meshes
following the corner arcs (P2 area) and the tetrahedra solids. Written, not run: tolerances are first estimates."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from ..model.spec import DomainSpec, GeometrySpec, ModelSpec

_UM = 1e-6
# 2D objects [m]: corners unsorted on purpose; a stadium (r = half its height); a square rounded into a circle
_ROUNDED = {'x0': 1.2 * _UM, 'y0': 0.9 * _UM, 'x1': 0.2 * _UM, 'y1': 0.3 * _UM, 'r': 0.15 * _UM}
_STADIUM = {'x0': 1.5 * _UM, 'y0': 0.2 * _UM, 'x1': 2.3 * _UM, 'y1': 0.5 * _UM, 'r': 0.15 * _UM}
_ROUND_SQUARE = {'x0': 1.6 * _UM, 'y0': 0.8 * _UM, 'x1': 2.0 * _UM, 'y1': 1.2 * _UM, 'r': 0.2 * _UM}
_WINDOW = (2.6 * _UM, 1.4 * _UM)
_H = 0.08 * _UM                 # triangle mesh h_max [m]
_PIXEL = 1e-9                   # raster checks: centres (k + 0.37) pixel never hit a corner coordinate
_RASTER_AREA_RTOL = 2e-3        # centre-count area vs the exact area
_OUTLINE_AREA_RTOL = 1e-5       # shoelace area of a 1024-segment outline
_ON_BOUNDARY = 1e-12            # x the longest window side: outline vertices / P2 midside nodes off the boundary
# region area + 2/3 chord x midside offset per curved outline edge: the parabola misses theta^5 r^2 / 960 per edge
# (theta <= 0.3 rad at curvature_factor 0.3), 1.7e-5 of a circle's area; P1 misses ~theta^2 / 6 ~ 1e-2
_P2_AREA_RTOL = 5e-5
_SECTION_MISMATCH = 1e-3        # 3D: rasterize_box (float32 centres) vs contains_points_3d (float64), share of cells
_GMSH_VOLUME_RTOL = 2e-2        # gmsh: member element volume vs the exact prism volume (faceted cylinders)
_DOMAIN_3D = (1.4 * _UM, 1.0 * _UM, 0.6 * _UM)


def _shapes(objects, size_z=None, size=(1.4 * _UM, 1.0 * _UM)):
    """resolve_geometry of [(name, kind, params)] in a 2D (or size_z: 3D) domain."""
    from ..model.geometry import resolve_geometry

    spec = ModelSpec(title='Self-test: rounded rectangle', domain=DomainSpec(size_x=size[0], size_y=size[1],
                                                                             size_z=size_z))
    spec.geometry = [GeometrySpec(name, kind, dict(params)) for name, kind, params in objects]
    return resolve_geometry(spec, {})


def _sorted(values) -> dict:
    x0, x1 = sorted((values['x0'], values['x1']))
    y0, y1 = sorted((values['y0'], values['y1']))
    return dict(values, x0=x0, y0=y0, x1=x1, y1=y1)


def _exact_area(values) -> float:
    box = _sorted(values)
    return (box['x1'] - box['x0']) * (box['y1'] - box['y0']) - (4.0 - math.pi) * box['r'] ** 2


def _boundary_distance(values, points) -> np.ndarray:
    """Signed distance [m] of (n, 2) points to the rounded rectangle's boundary (< 0 inside)."""
    box = _sorted(values)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centre = np.array([0.5 * (box['x0'] + box['x1']), 0.5 * (box['y0'] + box['y1'])])
    half = np.array([0.5 * (box['x1'] - box['x0']) - box['r'], 0.5 * (box['y1'] - box['y0']) - box['r']])
    beyond = np.abs(points - centre) - half
    outside = np.maximum(beyond, 0.0)
    return np.hypot(outside[:, 0], outside[:, 1]) + np.minimum(beyond.max(axis=1), 0.0) - box['r']


def _grid(size=(1.4 * _UM, 1.0 * _UM)):
    x = (np.arange(int(round(size[0] / _PIXEL))) + 0.37) * _PIXEL
    y = (np.arange(int(round(size[1] / _PIXEL))) + 0.37) * _PIXEL
    return np.meshgrid(x, y, indexing='ij')


# ---- geometry (model.geometry) ----
def _plane_checks(add):
    """Resolve rules, raster = union of 2 rectangles + 4 circles, r = 0 = rectangle, area, outline, cut intervals,
    inward / outward offsets and the per-point offset test."""
    from ..model.geometry import (_line_intervals, _offset_plane_inside, _plane_shape, axis_parallel_faces, outline,
                                  rasterize)

    shapes = _shapes([('rounded', 'rounded_rectangle', _ROUNDED)])
    shape = shapes['rounded']
    box = _sorted(_ROUNDED)
    add('resolve_sorts_corners', all(shape.values[key] == box[key] for key in ('x0', 'y0', 'x1', 'y1')),
        {key: shape.values[key] for key in ('x0', 'y0', 'x1', 'y1')})
    try:
        _shapes([('too_round', 'rounded_rectangle', dict(_ROUNDED, r=0.31 * _UM))])
        add('resolve_refuses_large_radius', False, 'no error')
    except ValueError as exc:
        add('resolve_refuses_large_radius', 'half the shorter side' in str(exc), str(exc))
    X, Y = _grid()
    mask = np.asarray(rasterize('rounded', shapes, X, Y, np), dtype=bool)
    radius = box['r']
    parts = [('wide', 'rectangle', {'x0': box['x0'] + radius, 'y0': box['y0'], 'x1': box['x1'] - radius,
                                    'y1': box['y1']}),
             ('tall', 'rectangle', {'x0': box['x0'], 'y0': box['y0'] + radius, 'x1': box['x1'],
                                    'y1': box['y1'] - radius})]
    parts += [(f'corner_{index}', 'circle', {'cx': cx, 'cy': cy, 'r': radius})
              for index, (cx, cy) in enumerate(((box['x0'] + radius, box['y0'] + radius),
                                                (box['x1'] - radius, box['y0'] + radius),
                                                (box['x1'] - radius, box['y1'] - radius),
                                                (box['x0'] + radius, box['y1'] - radius)))]
    union_shapes = _shapes(parts)
    union = np.zeros(X.shape, dtype=bool)
    for name in union_shapes:
        union |= np.asarray(rasterize(name, union_shapes, X, Y, np), dtype=bool)
    add('raster_equals_rectangles_and_circles', bool(np.array_equal(mask, union)), int((mask != union).sum()))
    area_error = abs(float(mask.sum()) * _PIXEL ** 2 / _exact_area(_ROUNDED) - 1.0)
    add('raster_area', area_error <= _RASTER_AREA_RTOL, area_error)
    sharp = _shapes([('sharp', 'rounded_rectangle', dict(_ROUNDED, r=0.0)), ('plain', 'rectangle',
                                                                           {key: _ROUNDED[key] for key in box
                                                                            if key != 'r'})])
    add('zero_radius_is_rectangle', bool(np.array_equal(np.asarray(rasterize('sharp', sharp, X, Y, np)),
                                                        np.asarray(rasterize('plain', sharp, X, Y, np)))))
    polyline = outline('rounded', shapes, n_segments=1024)[0]
    off = float(np.abs(_boundary_distance(_ROUNDED, polyline)).max())
    shoelace = 0.5 * abs(float(np.sum(polyline[:-1, 0] * polyline[1:, 1] - polyline[1:, 0] * polyline[:-1, 1])))
    add('outline_on_boundary', off <= _ON_BOUNDARY * _WINDOW[0] and np.array_equal(polyline[0], polyline[-1]), off)
    add('outline_area', abs(shoelace / _exact_area(_ROUNDED) - 1.0) <= _OUTLINE_AREA_RTOL, shoelace)
    ends, middles_inside = [], []
    for axis, positions in (('y', (0.3001, 0.33, 0.45, 0.6, 0.85, 0.8999)), ('x', (0.2001, 0.25, 0.7, 1.19))):
        for position in positions:
            for lo, hi in _line_intervals(shape, axis, position * _UM):
                pairs = [(lo, position * _UM), (hi, position * _UM)] if axis == 'y' else \
                    [(position * _UM, lo), (position * _UM, hi)]
                ends += pairs
                middle = np.mean(np.array(pairs), axis=0)
                middles_inside.append(bool(rasterize('rounded', shapes, middle[:1], middle[1:], np)[0]))
    ends_off = float(np.abs(_boundary_distance(_ROUNDED, ends)).max())
    add('cut_intervals_on_boundary', len(ends) == 20 and ends_off <= _ON_BOUNDARY * _WINDOW[0] and all(middles_inside),
        {'ends': len(ends), 'off': ends_off})
    radii = {f'{offset:g}': _plane_shape(shape, offset * _UM).values['r'] / _UM for offset in (0.05, 0.2, -0.05)}
    add('offset_radius', abs(radii['0.05'] - 0.1) < 1e-12 and radii['0.2'] == 0.0
        and abs(radii['-0.05'] - 0.2) < 1e-12, radii)
    plane = _plane_shape(shape)
    mismatches = {}
    for offset in (0.05, 0.2, -0.05):
        offset_shapes = {'offset': _plane_shape(shape, offset * _UM)}
        per_point = _offset_plane_inside(plane, np.full(X.shape, offset * _UM), X, Y)
        mismatches[f'{offset:g}'] = int((per_point != np.asarray(rasterize('offset', offset_shapes, X, Y, np))).sum())
    add('per_point_offset_matches_offset_shape', not any(mismatches.values()), mismatches)
    stadium = _shapes([('stadium', 'rounded_rectangle', _STADIUM)], size=_WINDOW)
    faces, stadium_faces = axis_parallel_faces('rounded', shapes), axis_parallel_faces('stadium', stadium)
    add('flat_side_faces', set(faces) >= {'x', 'y'} and 'x' not in stadium_faces and 'y' in stadium_faces,
        {'rounded': sorted(faces), 'stadium': sorted(stadium_faces)})


def _prism_checks(add):
    """3D: per-height sections (rasterize_box) vs per-point tests (contains_points_3d) of a tapered z prism whose top
    corners end sharp and an x extrusion; surface faces of tapers; a collapsing taper refused."""
    from ..model.geometry import contains_points_3d, rasterize_box, surface_faces

    um = _UM
    objects = [('tapered', 'rounded_rectangle', dict(_ROUNDED, r=0.08 * um, z0=0.1 * um, z1=0.5 * um,
                                                     sidewall_angle=15.0)),
               ('sideways', 'rounded_rectangle', {'x0': 0.2 * um, 'y0': 0.1 * um, 'x1': 0.8 * um, 'y1': 0.5 * um,
                                                  'r': 0.1 * um, 'axis': 'x', 'z0': 0.3 * um, 'z1': 1.1 * um}),
               ('sideways_tapered', 'rounded_rectangle', {'x0': 0.2 * um, 'y0': 0.1 * um, 'x1': 0.8 * um,
                                                          'y1': 0.5 * um, 'r': 0.1 * um, 'axis': 'x', 'z0': 0.3 * um,
                                                          'z1': 1.1 * um, 'sidewall_angle': 10.0})]
    shapes = _shapes(objects, size_z=_DOMAIN_3D[2], size=_DOMAIN_3D[:2])
    centres = [((np.arange(int(round(length / (0.02 * um)))) + 0.5) * (0.02 * um)).astype(np.float32)
               for length in _DOMAIN_3D]
    X, Y, Z = np.meshgrid(*(axis.astype(np.float64) for axis in centres), indexing='ij')
    shares = {}
    for name in ('tapered', 'sideways'):
        window, cropped = rasterize_box(name, shapes, *centres, np)
        sections = np.zeros(X.shape, dtype=bool)
        sections[window] = cropped
        points = contains_points_3d(name, shapes, X.ravel(), Y.ravel(), Z.ravel()).reshape(X.shape)
        shares[name] = float((sections != points).mean())
        add(f'{name}_sections_match_points', shares[name] <= _SECTION_MISMATCH and points.any(), shares[name])
    for name in ('tapered', 'sideways_tapered'):
        faces, edges = surface_faces(name, shapes, _DOMAIN_3D, n_segments=48)
        add(f'{name}_surface_faces', len(faces) > 2 and all(np.isfinite(face).all() for face in faces)
            and len(faces[0]) == len(faces[1]), {'faces': len(faces), 'edges': len(edges)})
    try:
        _shapes([('collapsing', 'rounded_rectangle', dict(_ROUNDED, z0=0.1 * um, z1=0.5 * um, sidewall_angle=60.0))],
                size_z=_DOMAIN_3D[2], size=_DOMAIN_3D[:2])
        add('collapsing_taper_refused', False, 'no error')
    except ValueError as exc:
        add('collapsing_taper_refused', 'collapses' in str(exc), str(exc))


def _check_geometry(results_by_name):
    checks = SubChecks()
    add = checks.add
    _plane_checks(add)
    _prism_checks(add)
    return checks_result(checks)


# ---- Curve arc pieces and the triangle mesh ----
def _curve_checks(add):
    """Curve arc API (flags, covers, arc_points, span check) and mirrored_curve images of circle / ellipse arcs."""
    from ..model.triangle_meshing import mirrored_curve
    from .mesh import Curve

    arc = Curve(1.0, 2.0, 0.5, 0.5, 0.0, -0.5 * math.pi, 0.5 * math.pi)
    add('curve_closed_flags', Curve(0.0, 0.0, 1.0, 1.0).is_closed and not arc.is_closed and arc.whole.is_closed)
    covered = arc.covers(np.array([-0.25, 0.25, 1.75, -0.5, 0.0]) * math.pi).tolist()
    add('curve_covers', covered == [True, False, True, True, True], covered)
    points = arc.arc_points(8)
    add('curve_arc_points', len(points) == 9 and np.allclose(points[0], (1.0, 1.5), atol=1e-15)
        and np.allclose(points[-1], (1.5, 2.0), atol=1e-15), points[[0, -1]].tolist())
    try:
        Curve(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        add('curve_refuses_zero_span', False)
    except ValueError:
        add('curve_refuses_zero_span', True)
    worst = 0.0
    for curve in (arc, Curve(1.0, 2.0, 0.5, 0.3, 0.4, 0.3, 2.0)):
        for axis, plane in ((0, 3.0), (1, -1.0)):
            image = mirrored_curve(curve, axis, plane).arc_points(16)
            expected = curve.arc_points(16)[::-1].copy()
            expected[:, axis] = 2.0 * plane - expected[:, axis]
            worst = max(worst, float(np.abs(image - expected).max()))
    add('mirrored_arc_images', worst <= 1e-12, worst)


def _pslg_checks(add):
    """build_pslg of an upper half circle (arc piece) closed by its diameter: a line crossing the absent lower half
    gets no vertex there, one crossing the arc does; curved segments stay on the upper half."""
    from scipy.spatial import cKDTree
    from .mesh import Curve
    from .mesher import SizeField, build_pslg

    um = _UM
    centre, radius = (1.0 * um, 1.0 * um), 0.4 * um
    arc = Curve(centre[0], centre[1], radius, radius, 0.0, 0.0, math.pi)
    straight = [(0.6 * um, 1.0 * um, 1.4 * um, 1.0 * um, ()), (0.3 * um, 0.8 * um, 1.7 * um, 0.8 * um, ()),
                (0.3 * um, 1.2 * um, 1.7 * um, 1.2 * um, ())]
    pslg = build_pslg((0.0, 0.0, 2.0 * um, 2.0 * um), SizeField(0.1 * um, 1.3, 0.001 * um), straight=straight,
                      curves=[(arc, ())], curvature_factor=0.3, h_min=0.001 * um)
    half_chord = math.sqrt(radius ** 2 - (0.2 * um) ** 2)
    tree = cKDTree(pslg.vertices)
    off_arc, _ = tree.query([(centre[0] - half_chord, 0.8 * um), (centre[0] + half_chord, 0.8 * um)])
    on_arc, _ = tree.query([(centre[0] - half_chord, 1.2 * um), (centre[0] + half_chord, 1.2 * um)])
    add('pslg_no_vertex_off_arc', float(off_arc.min()) > 1e-3 * um, off_arc.tolist())
    add('pslg_vertex_at_arc_crossing', float(on_arc.max()) <= 1e-9 * 2.0 * um, on_arc.tolist())
    curved = pslg.segments[pslg.segment_curve >= 0].ravel()
    heights = pslg.vertices[curved, 1] - centre[1]
    radial = np.abs(np.hypot(pslg.vertices[curved, 0] - centre[0], heights) - radius)
    add('pslg_curved_segments_on_arc', len(curved) > 0 and float(heights.min()) >= -1e-9 * 2.0 * um
        and float(radial.max()) <= 1e-9 * 2.0 * um, {'segments': len(curved) // 2, 'lowest': float(heights.min())})
    add('pslg_whole_curve', len(pslg.curves) == 1 and pslg.curves[0].is_closed)


def _sizing(mirror_axes=()):
    from ..model.meshing import TriangleSizing

    return TriangleSizing(h_max=_H, h_min=_H / 100.0, growth=1.3, curvature_factor=0.3, object_sizes={},
                          pml_widths={face: 0.0 for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}, pml_layers=4,
                          mesher='builtin', mirror_axes=tuple(mirror_axes))


def _mesh_spec(objects) -> ModelSpec:
    spec = ModelSpec(title='Self-test: rounded rectangle triangle mesh',
                     domain=DomainSpec(size_x=_WINDOW[0], size_y=_WINDOW[1]))
    spec.geometry = [GeometrySpec(name, 'rounded_rectangle', dict(values)) for name, values in objects]
    return spec


def _object_checks(mesh, name, values, n_curves, add, label=''):
    """Outline midside nodes on the exact boundary, curved edges on n_curves arcs, P1 area below the exact one and
    the P2 area (2/3 chord x midside offset per curved edge) within _P2_AREA_RTOL."""
    edges = mesh.edge_tags.get(name, np.zeros(0, dtype=np.int32))
    off = float(np.abs(_boundary_distance(values, mesh.midside_points[edges])).max()) if len(edges) else math.inf
    curved = edges[mesh.edge_curve[edges] >= 0]
    used = len(np.unique(mesh.edge_curve[curved]))
    add(f'{label}{name}_outline_on_boundary', off <= _ON_BOUNDARY * _WINDOW[0], off)
    add(f'{label}{name}_arcs', len(curved) > 0 and used == n_curves, {'curved_edges': len(curved), 'curves': used})
    member = mesh.region == mesh.region_names.index(name)
    p1_area = float(np.abs(mesh.areas()[member]).sum())
    ends = mesh.points[mesh.edges[curved]]
    chords = np.linalg.norm(ends[:, 1] - ends[:, 0], axis=1)
    offsets = np.linalg.norm(mesh.midside_points[curved] - ends.mean(axis=1), axis=1)
    p2_area = p1_area + float((2.0 / 3.0 * chords * offsets).sum())
    exact = _exact_area(values)
    p2_error = abs(p2_area / exact - 1.0)
    add(f'{label}{name}_p2_area', p1_area < exact and p2_error <= _P2_AREA_RTOL,
        {'p1_deficit': 1.0 - p1_area / exact, 'p2_error': p2_error})
    return curved


def _check_triangle_mesh(results_by_name):
    from ..model.triangle_meshing import triangulate

    checks = SubChecks()
    add = checks.add
    _curve_checks(add)
    _pslg_checks(add)
    objects = [('rounded', _ROUNDED), ('stadium', _STADIUM), ('round_square', _ROUND_SQUARE)]
    mesh = triangulate(_mesh_spec(objects), {}, sizing=_sizing())
    window_error = abs(float(mesh.areas().sum()) / (_WINDOW[0] * _WINDOW[1]) - 1.0)
    add('window_area', window_error <= 1e-12, window_error)
    for (name, values), n_curves in zip(objects, (4, 2, 1)):
        _object_checks(mesh, name, values, n_curves, add)
    centred = {'x0': 0.8 * _UM, 'y0': 0.4 * _UM, 'x1': 1.8 * _UM, 'y1': 1.0 * _UM, 'r': 0.2 * _UM}   # about x = 1.3
    mirrored = triangulate(_mesh_spec([('centred', centred)]), {}, sizing=_sizing(('x',)))
    curved = _object_checks(mirrored, 'centred', centred, 4, add, label='mirror_x_')
    middles = mirrored.midside_points[curved, 0]
    sides = (int((middles < 0.5 * _WINDOW[0]).sum()), int((middles > 0.5 * _WINDOW[0]).sum()))
    add('mirror_x_curved_edges_per_side', sides[0] == sides[1] > 0, sides)
    return checks_result(checks, elements=int(mesh.n_elements), mirrored_elements=int(mirrored.n_elements))


# ---- tetrahedra (model.tetrahedral_meshing) ----
def _check_tetrahedra(results_by_name):
    """model_solids per rounded prism (2 boxes + 4 cylinders; stadium 1 + 2; circle 1 cylinder; r = 0 a box; x axis /
    taper none), their union = contains_points_3d at random points, and with gmsh the member volume."""
    from ..model.geometry import contains_points_3d
    from ..model.meshing import TetrahedronSizing
    from ..model.tetrahedral_meshing import element_membership, model_solids, tetrahedralize
    from .gmsh_io import find_gmsh

    um = _UM
    checks = SubChecks()
    add = checks.add
    objects = {'plain': {'x0': 0.1 * um, 'y0': 0.1 * um, 'x1': 0.7 * um, 'y1': 0.5 * um, 'r': 0.1 * um,
                         'z0': 0.1 * um, 'z1': 0.5 * um},
               'stadium': {'x0': 0.8 * um, 'y0': 0.1 * um, 'x1': 1.3 * um, 'y1': 0.3 * um, 'r': 0.1 * um,
                           'z0': 0.1 * um, 'z1': 0.4 * um},
               'disc': {'x0': 0.8 * um, 'y0': 0.5 * um, 'x1': 1.2 * um, 'y1': 0.9 * um, 'r': 0.2 * um},
               'sharp': {'x0': 0.1 * um, 'y0': 0.6 * um, 'x1': 0.5 * um, 'y1': 0.9 * um, 'r': 0.0,
                         'z0': 0.2 * um, 'z1': 0.4 * um},
               'sideways': {'x0': 0.2 * um, 'y0': 0.1 * um, 'x1': 0.8 * um, 'y1': 0.5 * um, 'r': 0.1 * um,
                            'axis': 'x', 'z0': 0.3 * um, 'z1': 1.1 * um},
               'tapered': {'x0': 0.1 * um, 'y0': 0.1 * um, 'x1': 0.7 * um, 'y1': 0.5 * um, 'r': 0.1 * um,
                           'z0': 0.1 * um, 'z1': 0.5 * um, 'sidewall_angle': 5.0}}
    shapes = _shapes([(name, 'rounded_rectangle', values) for name, values in objects.items()],
                     size_z=_DOMAIN_3D[2], size=_DOMAIN_3D[:2])
    sizing = TetrahedronSizing(h_max=0.1 * um, h_min=0.001 * um, growth=1.3, curvature_factor=0.3, object_sizes={},
                               pml_widths={face: 0.0 for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')},
                               pml_layers=8, mesher='gmsh')
    low, high = np.zeros(3), np.array(_DOMAIN_3D)
    solids = model_solids(shapes, sizing, low, high)
    counts = {name: sorted(solid.kind for solid in solids if solid.name == name) for name in objects}
    expected = {'plain': ['box'] * 2 + ['cylinder'] * 4, 'stadium': ['box', 'cylinder', 'cylinder'],
                'disc': ['cylinder'], 'sharp': ['box'], 'sideways': [], 'tapered': []}
    add('solid_counts', counts == expected, counts)
    rng = np.random.default_rng(7)
    mismatches = {}
    for name in ('plain', 'stadium', 'disc', 'sharp'):
        points = rng.uniform(low, high, size=(20000, 3))
        union = np.zeros(len(points), dtype=bool)
        for solid in solids:
            if solid.name == name:
                union |= solid.contains(points)
        inside = contains_points_3d(name, shapes, *points.T)
        mismatches[name] = int((union != inside).sum())
        add(f'{name}_solids_union_matches', mismatches[name] == 0 and inside.any(), mismatches[name])
    if find_gmsh() is None:
        add('gmsh_member_volume', True, 'skipped: gmsh not found (OPEN_SOL_GMSH or PATH)')
        return checks_result(checks)
    spec = ModelSpec(title='Self-test: rounded rectangle tetrahedra',
                     domain=DomainSpec(size_x=_DOMAIN_3D[0], size_y=_DOMAIN_3D[1], size_z=_DOMAIN_3D[2]))
    spec.geometry = [GeometrySpec('plain', 'rounded_rectangle', dict(objects['plain']))]
    mesh = tetrahedralize(spec, {}, sizing=sizing)
    plain_shapes = _shapes([('plain', 'rounded_rectangle', objects['plain'])], size_z=_DOMAIN_3D[2],
                           size=_DOMAIN_3D[:2])
    member = element_membership(mesh, 'plain', plain_shapes)
    exact = _exact_area(objects['plain']) * 0.4 * um
    volume_error = abs(float(mesh.volumes(member).sum()) / exact - 1.0)
    add('gmsh_member_volume', volume_error <= _GMSH_VOLUME_RTOL, {'error': volume_error,
                                                                  'elements': int(mesh.tets.shape[0])})
    return checks_result(checks)


ROUNDED_RECTANGLE_SELFTEST_CASES = (
    SelfTestCase(
        name='geometry_rounded_rectangle',
        label='Rounded rectangle geometry: rasters, outlines, sections',
        description=(
            'CPU checks of the rounded_rectangle kind (new_modules_plan G4, model.geometry): corners sorted, r above '
            'half the shorter side refused; on 1 nm centres the raster equals the union of two rectangles and four '
            f'circles exactly, r = 0 equals the rectangle, the centre-count area is within {_RASTER_AREA_RTOL:g} of '
            'w h - (4 - pi) r^2; the 1024-segment outline lies on the boundary (1e-12 x side) with shoelace area '
            f'within {_OUTLINE_AREA_RTOL:g}; cut intervals end on the boundary; offsets give r - offset (0 once '
            'collapsed, larger outward) and the per-point offset test equals the offset shape; flat-side snap faces '
            '(none along a stadium\'s round ends). 3D: rasterize_box vs contains_points_3d of a 15 deg taper whose '
            f'corners end sharp and of an x extrusion (mismatch share <= {_SECTION_MISMATCH:g}), surface faces of '
            'tapers, a collapsing taper refused. error = failed sub-checks, tolerance 0. Written, not run.'),
        specs=lambda: {},
        check=_check_geometry,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem_rounded_rectangle_triangle_mesh',
        label='Rounded rectangle triangle mesh: corner arcs',
        description=(
            'CPU checks of fem.mesh.Curve arc pieces and their triangle meshes (G4): Curve flags / covers / '
            'arc_points / span check, mirrored_curve arc images; build_pslg of a half circle closed by its diameter '
            '(no vertex where a line crosses the absent half, one where it crosses the arc, curved segments on the '
            'arc, one whole curve). triangulate (h 0.08 um, no ring) of a rounded rectangle, a stadium and a square '
            'rounded into a circle: outline midside nodes on the exact boundary (1e-12 x side), curved edges on 4 / '
            f'2 / 1 merged curves, P1 area below the exact one and P2 area within {_P2_AREA_RTOL:g}; a mirror_x mesh '
            'of a centred rounded rectangle: the same checks and equal curved edges per side. error = failed '
            'sub-checks, tolerance 0. Written, not run.'),
        specs=lambda: {},
        check=_check_triangle_mesh,
        max_wall_time=300.0),
    SelfTestCase(
        name='fem_rounded_rectangle_tetrahedra',
        label='Rounded rectangle tetrahedra: box and cylinder solids',
        description=(
            'CPU checks of tetrahedral_meshing.model_solids for rounded_rectangle prisms (G4): 2 boxes + 4 z cylinders, '
            'a stadium 1 box + 2 cylinders, a circle 1 cylinder, r = 0 one box, x axis and tapers none; the union of '
            'Solid.contains equals contains_points_3d at 20000 random points each; with gmsh the member elements of '
            f'a meshed prism hold its exact volume within {_GMSH_VOLUME_RTOL:g} (skipped without gmsh). error = '
            'failed sub-checks, tolerance 0. Written, not run.'),
        specs=lambda: {},
        check=_check_tetrahedra,
        max_wall_time=300.0),
)
