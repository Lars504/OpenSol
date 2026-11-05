"""Self-test of the own triangle mesher (mesh plan stage T1, CPU; specs() is empty, check({}) meshes directly):
conformity, quality, areas, region tags, P2 curved-boundary convergence, size bound, determinism, the PML ring, the
MeshSpec path and, when gmsh is found, the gmsh variant."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from ..model.spec import DomainSpec, GeometrySpec, MeshSpec, ModelSpec, RefinementSpec

_UM = 1e-6
_SIZE = (3.0 * _UM, 2.0 * _UM)                 # window [m]
_H_MAX = 0.12 * _UM                            # maximum element size [m]
_RECT_SIZE = 0.05 * _UM                        # element size inside the refined rectangle [m]
_GROWTH = 1.3
_CURVATURE_FACTOR = 0.3
_DISC = (1.0 * _UM, 1.0 * _UM, 0.6 * _UM)      # circle in a 2 x 2 um window: cx, cy, r [m]
_DISC_SIZES = (0.12 * _UM, 0.06 * _UM)         # h of the P2 area pair (curvature size 0.18 um not binding); also
                                               # the fem pipe / transfer meshes (selftests_elements, shared by the cache)
_RING = (0.3 * _UM, 4)                         # PML ring thickness [m], layers
_MIN_ANGLE = 25.0                              # deg, away from small input angles
_SMALL_ANGLE_REACH = 3.0                       # excluded radius around a small input angle / h there
_SIZE_RATIO = 1.3                              # longest edge / h(midpoint)
_AREA_RTOL = 1e-12
_LENGTH_RTOL = 1e-9
_P2_GAIN = 12.0                                # P2 disc area error ratio per halving of h (order 4: 16)
_PERIMETER_RTOL = 1e-2                         # composite outline length vs exact (straight chords of the hole)


def mesher_spec() -> ModelSpec:
    """2D spec of the main mesh: refined rectangle, circle crossing a diamond, a plate with a hole (difference), an
    ellipse crossing the rectangle and a line, a 10.6 deg wedge, a point; one 'triangles' mesh."""
    um = _UM
    spec = ModelSpec(title='Self-test: triangle mesher', domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]))
    spec.geometry = [
        GeometrySpec('rect', 'rectangle', {'x0': 0.3 * um, 'y0': 0.3 * um, 'x1': 1.3 * um, 'y1': 1.1 * um}),
        GeometrySpec('disc', 'circle', {'cx': 2.1 * um, 'cy': 1.0 * um, 'r': 0.55 * um}),
        GeometrySpec('oval', 'ellipse', {'cx': 1.0 * um, 'cy': 1.35 * um, 'a': 0.45 * um, 'b': 0.25 * um,
                                         'angle': 20.0}),
        GeometrySpec('diamond', 'polygon', {'points': [(1.85 * um, 0.55 * um), (2.2 * um, 0.2 * um),
                                                       (2.55 * um, 0.55 * um), (2.2 * um, 0.9 * um)]}),
        GeometrySpec('wedge', 'polygon', {'points': [(0.2 * um, 1.8 * um), (1.0 * um, 1.8 * um),
                                                     (0.2 * um, 1.95 * um)]}),
        GeometrySpec('plate', 'rectangle', {'x0': 1.6 * um, 'y0': 1.15 * um, 'x1': 2.9 * um, 'y1': 1.95 * um}),
        GeometrySpec('hole', 'circle', {'cx': 2.5 * um, 'cy': 1.6 * um, 'r': 0.2 * um}),
        GeometrySpec('holey', 'difference', {'base': 'plate', 'subtract': ['hole']}),
        GeometrySpec('cut', 'line', {'x0': 0.2 * um, 'y0': 1.6 * um, 'x1': 2.8 * um, 'y1': 1.6 * um}),
        GeometrySpec('probe', 'point', {'x': 0.7 * um, 'y': 0.7 * um}),
    ]
    spec.meshes = [MeshSpec(name='triangles', kind='triangles', mode='nonuniform', dx=_H_MAX, max_ratio=_GROWTH,
                            refinements=[RefinementSpec('rect', _RECT_SIZE)])]
    return spec


def disc_spec(h_max) -> ModelSpec:
    """2 x 2 um window holding the circle _DISC on a uniform-size 'triangles' mesh."""
    cx, cy, r = _DISC
    spec = ModelSpec(title='Self-test: triangle disc', domain=DomainSpec(size_x=2.0 * _UM, size_y=2.0 * _UM))
    spec.geometry = [GeometrySpec('disc', 'circle', {'cx': cx, 'cy': cy, 'r': r})]
    spec.meshes = [MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=h_max)]
    return spec


def _sizing(h_max, object_sizes=None, pml=None, mesher='builtin'):
    """Explicit meshing.TriangleSizing (independent of the MeshSpec resolution)."""
    from ..model.meshing import TriangleSizing

    return TriangleSizing(h_max=h_max, h_min=h_max / 100.0, growth=_GROWTH, curvature_factor=_CURVATURE_FACTOR,
                          object_sizes=dict(object_sizes or {}), pml_widths=dict(pml or {}),
                          pml_layers=_RING[1], mesher=mesher)


_DISC_MESHES = {}   # (h_max, ring) -> TriangleMesh: the disc meshes shared read-only by every fem self-test


def disc_mesh(h_max, ring=False):
    """Cached TriangleMesh of disc_spec(h_max) (ring: the _RING PML layer ring on every face); callers never write it."""
    key = (float(h_max), bool(ring))
    if key not in _DISC_MESHES:
        from ..fem.mesh import FACE_BITS
        from ..model.triangle_meshing import triangulate

        pml = {face: _RING[0] for face in FACE_BITS} if ring else None
        _DISC_MESHES[key] = triangulate(disc_spec(h_max), {}, sizing=_sizing(h_max, pml=pml))
    return _DISC_MESHES[key]


def disc_region_mesh(h_max):
    """disc_mesh for checks on the disc region only: at _DISC_SIZES[0] the ringed mesh (ring = layers extruded from the
    window-edge nodes, no disc tag, no curved edge: the same disc elements, one build fewer)."""
    return disc_mesh(h_max, ring=h_max == _DISC_SIZES[0])


def _conformity_checks(registry, spec, add):
    """Fem-level pipeline on mesher_spec: constraint edges are mesh edges covering every PSLG segment, area, angles
    away from small input angles, size bound, region tags constant inside elements; the MeshSpec-path mesh has the
    digest of the raw mesh of the explicit sizing (determinism + equal sizing); returns that mesh (None on failure)."""
    from ..fem.mesh import mesh_digest
    from ..fem.mesher import mesh_pslg
    from ..model.geometry import resolve_geometry
    from ..model.triangle_meshing import mesher_size_field, model_pslg, triangulate

    shapes = resolve_geometry(spec, {})
    sizing = _sizing(_H_MAX, {'rect': _RECT_SIZE})
    window = (0.0, 0.0) + _SIZE
    size_field = mesher_size_field(sizing, shapes)
    pslg = model_pslg(shapes, window, size_field, sizing)
    raw = mesh_pslg(pslg, size_field, h_min=sizing.h_min)
    try:
        mesh = triangulate(spec, {}, registry, shapes=shapes)   # MeshSpec -> meshing.triangle_sizing
    except ValueError as exc:   # a constraint segment that is no mesh edge
        add('constraint_edges_are_mesh_edges', False, str(exc))
        return None
    add('constraint_edges_are_mesh_edges', True)
    add('same_mesh_twice', mesh.digest == mesh_digest(raw.points, raw.triangles), mesh.digest[:16])

    lengths = np.linalg.norm(raw.points[raw.constraint_edges[:, 0]] - raw.points[raw.constraint_edges[:, 1]], axis=1)
    covered = np.bincount(raw.constraint_segment, weights=lengths, minlength=len(pslg.segments))
    segment_lengths = np.linalg.norm(pslg.vertices[pslg.segments[:, 0]] - pslg.vertices[pslg.segments[:, 1]], axis=1)
    straight = pslg.segment_curve < 0
    length_error = float(np.max(np.abs(covered[straight] / segment_lengths[straight] - 1.0)))
    add('straight_segments_covered', length_error <= _LENGTH_RTOL and bool((covered > 0).all()), length_error)
    off_curve = 0.0
    curve_of = pslg.segment_curve[raw.constraint_segment]
    for curve_index, curve in enumerate(pslg.curves):
        nodes = np.unique(raw.constraint_edges[curve_of == curve_index])
        if len(nodes):
            off_curve = max(off_curve, float(np.max(np.linalg.norm(curve.project(raw.points[nodes])
                                                                   - raw.points[nodes], axis=1))))
    add('curve_nodes_on_curve', off_curve <= _LENGTH_RTOL * max(_SIZE), off_curve)

    area_error = abs(float(mesh.areas().sum()) / (_SIZE[0] * _SIZE[1]) - 1.0)
    add('areas_sum_to_window', area_error <= _AREA_RTOL, area_error)
    _angle_checks(mesh, pslg, size_field, add)
    add('edge_size_bound', mesh.quality.max_size_ratio <= _SIZE_RATIO, mesh.quality.max_size_ratio)
    _region_checks(mesh, shapes, window, add)
    return mesh


def _angle_checks(mesh, pslg, size_field, add):
    """Smallest angle >= _MIN_ANGLE over elements farther than _SMALL_ANGLE_REACH x h from every small input angle."""
    from ..fem.quality import element_angles

    smallest = element_angles(mesh.points, mesh.triangles).min(axis=1)
    near = np.zeros(mesh.n_elements, dtype=bool)
    for x, y, _ in pslg.small_angles:
        reach = _SMALL_ANGLE_REACH * float(size_field(np.array([[x, y]]))[0])
        distance = np.linalg.norm(mesh.points[mesh.triangles] - np.array([x, y]), axis=2).min(axis=1)
        near |= distance <= reach
    away = float(smallest[~near].min())
    add('min_angle_away_from_small_inputs', away >= _MIN_ANGLE,
        {'min_angle': away, 'small_input_angles': len(pslg.small_angles), 'excluded_elements': int(near.sum())})
    add('wedge_flagged_small', any(angle < 11.0 for _, _, angle in pslg.small_angles),
        [round(angle, 2) for _, _, angle in pslg.small_angles])


def _region_checks(mesh, shapes, window, add):
    """Object membership equal at each element's centroid and at its three inner sub-triangle centroids (no element
    straddles an object outline; elements with a curved edge skipped: the exact curve bulges past their chord);
    region tags = the last object holding the centroid; tags of lines / points / composite outline / faces."""
    from ..model.triangle_meshing import element_masks

    names = list(mesh.region_names)
    corners = mesh.points[mesh.triangles]
    centre_masks = element_masks(shapes, names, window, corners.mean(axis=1))
    straight = ~(mesh.edge_curve[mesh.element_edges] >= 0).any(axis=1)
    straddling = 0
    for k in range(3):
        weights = np.full(3, 1.0 / 6.0)
        weights[k] = 2.0 / 3.0
        sub_masks = element_masks(shapes, names, window, np.einsum('ekc,k->ec', corners, weights))
        straddling += sum(int((sub_masks[name] != centre_masks[name])[straight].sum()) for name in names)
    add('no_element_straddles_an_outline', straddling == 0, straddling)
    expected_region = np.full(mesh.n_elements, -1)
    for index, name in enumerate(names):
        expected_region[centre_masks[name]] = index
    add('region_tags_match_centroids', bool(np.array_equal(expected_region, mesh.region)), names)

    um = _UM
    cut_edges = mesh.edges[mesh.edge_tags.get('cut', np.zeros(0, dtype=np.int32))]
    cut_length = float(np.linalg.norm(mesh.points[cut_edges[:, 0]] - mesh.points[cut_edges[:, 1]], axis=1).sum())
    on_line = bool(len(cut_edges)) and bool(np.allclose(mesh.points[cut_edges.ravel(), 1], 1.6 * um, rtol=0,
                                                        atol=1e-9 * max(_SIZE)))
    add('line_edges_tagged', on_line and abs(cut_length / (2.6 * um) - 1.0) <= _LENGTH_RTOL, cut_length)
    probe = mesh.point_tags.get('probe')
    add('point_is_a_node', probe is not None and bool(np.allclose(mesh.points[probe], (0.7 * um, 0.7 * um),
                                                                   rtol=0, atol=1e-15)), probe)
    holey_edges = mesh.edges[mesh.edge_tags.get('holey', np.zeros(0, dtype=np.int32))]
    holey_length = float(np.linalg.norm(mesh.points[holey_edges[:, 0]] - mesh.points[holey_edges[:, 1]],
                                        axis=1).sum())
    exact = 2.0 * (1.3 + 0.8) * um + 2.0 * math.pi * 0.2 * um
    add('composite_outline_tagged', abs(holey_length / exact - 1.0) <= _PERIMETER_RTOL,
        {'length': holey_length, 'exact': exact})
    faces_ok = all(len(mesh.edge_tags.get(face, ())) for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi'))
    add('outer_faces_tagged', faces_ok, {face: len(mesh.edge_tags.get(face, ())) for face in ('x_lo', 'x_hi')})


def _disc_area(mesh) -> tuple:
    """(straight, P2) area of the disc region: element areas plus 2/3 x chord x midside offset per curved edge."""
    disc = mesh.region == mesh.region_names.index('disc')
    straight = float(mesh.areas()[disc].sum())
    curved = np.flatnonzero(mesh.edge_curve >= 0)
    starts, ends = mesh.points[mesh.edges[curved, 0]], mesh.points[mesh.edges[curved, 1]]
    chord_middles = 0.5 * (starts + ends)
    centre = np.array(_DISC[:2])
    offsets = np.linalg.norm(mesh.midside_points[curved] - chord_middles, axis=1)
    outward = np.where(np.linalg.norm(mesh.midside_points[curved] - centre, axis=1)
                       >= np.linalg.norm(chord_middles - centre, axis=1), 1.0, -1.0)
    chords = np.linalg.norm(ends - starts, axis=1)
    return straight, straight + float((2.0 / 3.0 * chords * offsets * outward).sum())


def _p2_and_ring_checks(add):
    """P2 disc area error falls >= _P2_GAIN x per halving of h; the PML ring's area, element count and corner bits."""
    from ..fem.mesh import FACE_BITS

    exact = math.pi * _DISC[2] ** 2
    errors = []
    for h_max in _DISC_SIZES:
        straight, curved = _disc_area(disc_region_mesh(h_max))
        errors.append((abs(straight / exact - 1.0), abs(curved / exact - 1.0)))
    gain = errors[0][1] / max(errors[1][1], 1e-300)
    add('p2_disc_area_order', gain >= _P2_GAIN, {'p2_errors': [e[1] for e in errors],
                                                 'straight_errors': [e[0] for e in errors], 'gain': gain})

    width, layers = _RING
    ringed = disc_mesh(_DISC_SIZES[0], ring=True)
    side = 2.0 * _UM
    area_error = abs(float(ringed.areas().sum()) / (side + 2.0 * width) ** 2 - 1.0)
    inside = ringed.points[(ringed.points >= 0.0).all(axis=1) & (ringed.points <= side).all(axis=1)]
    on_edge = {axis: (int((np.abs(inside[:, axis]) <= 1e-9 * side).sum()),
                      int((np.abs(inside[:, axis] - side) <= 1e-9 * side).sum())) for axis in (0, 1)}
    expected = 2 * layers * ((on_edge[0][0] - 1 + 2 * layers) + (on_edge[0][1] - 1 + 2 * layers)
                             + (on_edge[1][0] - 1) + (on_edge[1][1] - 1))
    ring_count = int((ringed.pml_faces > 0).sum())
    corners = int((ringed.pml_faces == FACE_BITS['x_lo'] | FACE_BITS['y_lo']).sum())
    add('pml_ring', area_error <= _AREA_RTOL and ring_count == expected and corners == 2 * layers * layers,
        {'area_error': area_error, 'ring_elements': ring_count, 'expected': expected, 'corner_elements': corners})


def _model_path_checks(mesh, add):
    """The MeshSpec-path mesh of _conformity_checks: elements in the refined rectangle obey its size."""
    if mesh is None:
        add('meshspec_refinement_size', False, 'no mesh (see constraint_edges_are_mesh_edges)')
        return
    in_rect = mesh.region == mesh.region_names.index('rect')
    corners = mesh.points[mesh.triangles[in_rect]]
    longest = max(float(np.linalg.norm(corners[:, (k + 1) % 3] - corners[:, k], axis=1).max()) for k in range(3))
    add('meshspec_refinement_size', longest <= _SIZE_RATIO * _RECT_SIZE and mesh.quality.mesher == 'builtin',
        {'longest_edge_in_rect': longest, 'refinement': _RECT_SIZE, 'elements': mesh.n_elements})


def _gmsh_checks(spec, add):
    """With gmsh found: the same model through gmsh (area, conformity of lines, no straddling); else skipped."""
    from ..fem.gmsh_io import find_gmsh
    from ..model.geometry import resolve_geometry
    from ..model.triangle_meshing import triangulate

    if find_gmsh() is None:
        add('gmsh_variant', True, 'skipped: gmsh not found (OPEN_SOL_GMSH or PATH)')
        return
    shapes = resolve_geometry(spec, {})
    mesh = triangulate(spec, {}, shapes=shapes, sizing=_sizing(_H_MAX, {'rect': _RECT_SIZE}, mesher='gmsh'))
    area_error = abs(float(mesh.areas().sum()) / (_SIZE[0] * _SIZE[1]) - 1.0)
    checks = {}
    _region_checks(mesh, shapes, (0.0, 0.0) + _SIZE, lambda name, passed, value=None: checks.update({name: passed}))
    add('gmsh_variant', area_error <= _AREA_RTOL and checks.get('no_element_straddles_an_outline', False),
        {'area_error': area_error, 'elements': mesh.n_elements, 'region_checks': checks})


def _check_mesher(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from ..model.schema import get_registry

    registry = get_registry()
    spec = mesher_spec()
    checks = SubChecks()
    add = checks.add

    mesh = _conformity_checks(registry, spec, add)
    _p2_and_ring_checks(add)
    _model_path_checks(mesh, add)
    _gmsh_checks(spec, add)
    return checks_result(checks)


MESHER_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_mesher_conformity',
        label='Triangle mesher: conformity, quality, curved boundaries',
        description='CPU checks of the own conforming Delaunay mesher (mesh plan T1) on a 3 x 2 um window with a '
                    'refined rectangle, a circle crossing a diamond and a holed plate (circle-circle crossing), an '
                    'ellipse crossing the rectangle and a line, a 10.6 deg wedge and a point (h_max 0.12 um, '
                    'growth 1.3): every split PSLG segment is a mesh edge, straight segments are covered to 1e-9 '
                    'and curved-segment nodes lie on their curve, element areas sum to the window area (1e-12), '
                    'smallest angle >= 25 deg farther than 3 h from small input angles (the wedge flagged), longest '
                    'edge <= 1.3 h(midpoint), membership constant at the centroid and three inner points of every '
                    'element without a curved edge, region tags = centroid tests, the line / point / composite-outline / outer-face tags, '
                    'checked on the MeshSpec-path mesh, whose digest equals the raw mesh of the explicit sizing '
                    '(determinism and MeshSpec -> the same sizing); the P2 disc area (midside nodes on the circle) '
                    'error falls >= 12x per halving of h (order 4); a 4-layer PML ring keeps the area (1e-12) with '
                    'the expected element and corner counts; a MeshSpec refinement bounds the rectangle elements '
                    '(1.3 x its size); the gmsh variant runs only when gmsh is found. error = number of failed '
                    'sub-checks, tolerance 0. Since 2026-10-20 the checks run on the MeshSpec-path mesh (no '
                    'explicit-sizing triangulate) and the disc meshes are shared (cached), not yet run that way.',
        specs=lambda: {},
        check=_check_mesher,
        max_wall_time=300.0),
)
