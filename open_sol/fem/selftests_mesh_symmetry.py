"""Self-tests of mirror-symmetric triangle meshes (new_modules_plan G2; CPU, specs() empty, check({}) meshes directly):
node / element / ring / curved-edge invariance under the mirror, tags, the MeshSpec path, the material symmetry check
and the em_fdfd_ez parity selection rule on a mirrored mesh. Written, not run: tolerances are first estimates."""
import math

import numpy as np

from ..core.constants import C0
from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from ..model.spec import DomainSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec

_UM = 1e-6
_WINDOW = (2.0 * _UM, 1.2 * _UM)
_H = 0.08 * _UM                 # maximum element size of the geometry checks [m]
_RING = (0.2 * _UM, 4)          # PML ring thickness [m], layers (every face)
_GROWTH = 1.3
_NODE_TOLERANCE = 1e-12         # x the longest window side: mirrored node / midside positions
_CURVE_TOLERANCE = 1e-12        # x the longest window side: curved midside nodes off their curve
_LENGTH_RTOL = 1e-9
_AREA_RTOL = 1e-12
# parity run: vacuum + eps 4 discs, cpml ring (kappa_max 1), quadratic, lambda 1 um, h 50 nm
_PARITY_H = 0.05 * _UM
_PARITY_WAVELENGTH = 1.0 * _UM
_PARITY_EPS = 4.0
_PARITY_TOLERANCE = 1e-10       # opposite-parity part / largest field over the sample pairs
_EVEN_SOURCE = (1.0 * _UM, 0.2 * _UM)    # on the mirror line x = 1 um
_ODD_SOURCES = ((0.6 * _UM, 0.2 * _UM), (1.4 * _UM, 0.2 * _UM))   # +I and -I at mirror images
_SAMPLE_X = (0.15, 0.35, 0.55, 0.75, 0.9)   # um, left half; their images are sampled too
_SAMPLE_Y = (0.15, 0.45, 0.75, 1.05)        # um


def _geometry():
    """Mirror-symmetric (about x = 1 um) objects: a disc on the line, a disc pair, a tilted ellipse pair (angles 30 /
    150 deg), a bar on the line, a line across it and three points (one on the line)."""
    um = _UM
    return [GeometrySpec('centre_disc', 'circle', {'cx': 1.0 * um, 'cy': 0.6 * um, 'r': 0.25 * um}),
            GeometrySpec('left_disc', 'circle', {'cx': 0.4 * um, 'cy': 0.4 * um, 'r': 0.15 * um}),
            GeometrySpec('right_disc', 'circle', {'cx': 1.6 * um, 'cy': 0.4 * um, 'r': 0.15 * um}),
            GeometrySpec('left_oval', 'ellipse', {'cx': 0.45 * um, 'cy': 0.9 * um, 'a': 0.2 * um, 'b': 0.1 * um,
                                                  'angle': 30.0}),
            GeometrySpec('right_oval', 'ellipse', {'cx': 1.55 * um, 'cy': 0.9 * um, 'a': 0.2 * um, 'b': 0.1 * um,
                                                   'angle': 150.0}),
            GeometrySpec('bar', 'rectangle', {'x0': 0.7 * um, 'y0': 0.05 * um, 'x1': 1.3 * um, 'y1': 0.15 * um}),
            GeometrySpec('cut', 'line', {'x0': 0.2 * um, 'y0': 1.1 * um, 'x1': 1.8 * um, 'y1': 1.1 * um}),
            GeometrySpec('p_left', 'point', {'x': 0.8 * um, 'y': 0.3 * um}),
            GeometrySpec('p_right', 'point', {'x': 1.2 * um, 'y': 0.3 * um}),
            GeometrySpec('p_mid', 'point', {'x': 1.0 * um, 'y': 1.0 * um})]


def _spec(materials=None, mirror_x=False, mirror_y=False, h_max=_H) -> ModelSpec:
    """2D spec of _geometry on one 'triangles' mesh (uniform h_max, the mirror flags)."""
    spec = ModelSpec(title='Self-test: mirror-symmetric triangle mesh',
                     domain=DomainSpec(size_x=_WINDOW[0], size_y=_WINDOW[1]))
    spec.geometry = _geometry()
    spec.materials = list(materials or [])
    spec.meshes = [MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=h_max, mirror_x=mirror_x,
                            mirror_y=mirror_y)]
    return spec


def _sizing(mirror_axes, h_max=_H):
    """Explicit meshing.TriangleSizing with the _RING ring on every face and the mirror axes."""
    from ..model.meshing import TriangleSizing

    return TriangleSizing(h_max=h_max, h_min=h_max / 100.0, growth=_GROWTH, curvature_factor=0.3, object_sizes={},
                          pml_widths={face: _RING[0] for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')},
                          pml_layers=_RING[1], mesher='builtin', mirror_axes=tuple(mirror_axes))


def _mirrored(points, axis, plane) -> np.ndarray:
    mirrored = np.array(points, dtype=np.float64).reshape(-1, 2)
    mirrored[:, axis] = 2.0 * plane - mirrored[:, axis]
    return mirrored


def _symmetry_checks(mesh, axis, label, add):
    """Nodes, elements, ring bits, curved midside nodes and outer-face tags invariant under the mirror about the
    window centre line normal to axis; the centre disc region mirrored."""
    from scipy.spatial import cKDTree
    from .mesh import FACE_BITS

    window = mesh.window
    plane = 0.5 * (window[axis] + window[axis + 2])
    scale = max(window[2] - window[0], window[3] - window[1])
    distance, partner = cKDTree(mesh.points).query(_mirrored(mesh.points, axis, plane))
    add(f'{label}_nodes_mirrored', float(distance.max()) <= _NODE_TOLERANCE * scale, float(distance.max()))
    original = np.sort(mesh.triangles, axis=1)
    mapped = np.sort(partner[mesh.triangles], axis=1)
    lookup = {tuple(row): index for index, row in enumerate(original.tolist())}
    element_partner = np.array([lookup.get(tuple(row), -1) for row in mapped.tolist()], dtype=np.int64)
    found = bool((element_partner >= 0).all())
    add(f'{label}_elements_mirrored', found, {'elements': int(mesh.n_elements),
                                              'unmatched': int((element_partner < 0).sum())})
    name = 'xy'[axis]
    low_bit, high_bit = FACE_BITS[f'{name}_lo'], FACE_BITS[f'{name}_hi']
    bits = mesh.pml_faces.astype(np.int64)
    swapped = (bits & ~(low_bit | high_bit)) | np.where(bits & low_bit, high_bit, 0) | np.where(bits & high_bit,
                                                                                              low_bit, 0)
    add(f'{label}_ring_bits_mirrored', found and bool(np.array_equal(swapped, bits[np.maximum(element_partner, 0)])),
        int((bits > 0).sum()))
    if 'centre_disc' in mesh.region_names:
        disc = mesh.region == mesh.region_names.index('centre_disc')
        add(f'{label}_region_mirrored', found and bool(np.array_equal(disc, disc[np.maximum(element_partner, 0)])),
            int(disc.sum()))
    curved = np.flatnonzero(mesh.edge_curve >= 0)
    middles = mesh.midside_points[curved]
    mid_distance, _ = cKDTree(middles).query(_mirrored(middles, axis, plane)) if len(curved) else (np.zeros(1), None)
    off_curve = max((float(np.max(np.linalg.norm(curve.project(middles[mesh.edge_curve[curved] == index])
                                                 - middles[mesh.edge_curve[curved] == index], axis=1)))
                     for index, curve in enumerate(mesh.curves) if (mesh.edge_curve[curved] == index).any()),
                    default=0.0)
    sides = (int((middles[:, axis] < plane).sum()), int((middles[:, axis] > plane).sum()))
    add(f'{label}_curved_edges_mirrored', len(curved) > 0 and sides[0] == sides[1]
        and float(mid_distance.max()) <= _NODE_TOLERANCE * scale and off_curve <= _CURVE_TOLERANCE * scale,
        {'curved_edges': int(len(curved)), 'per_side': sides, 'midside_mismatch': float(mid_distance.max()),
         'off_curve': off_curve})
    counts = (len(mesh.edge_tags.get(f'{name}_lo', ())), len(mesh.edge_tags.get(f'{name}_hi', ())))
    add(f'{label}_outer_faces_mirrored', counts[0] == counts[1] and counts[0] > 0, counts)
    return element_partner


def _tag_checks(mesh, add):
    """Window + ring area, the line tag length, the point tags at their exact coordinates."""
    width, _ = _RING
    area = (_WINDOW[0] + 2.0 * width) * (_WINDOW[1] + 2.0 * width)
    area_error = abs(float(mesh.areas().sum()) / area - 1.0)
    add('mirror_x_area', area_error <= _AREA_RTOL, area_error)
    cut_edges = mesh.edges[mesh.edge_tags.get('cut', np.zeros(0, dtype=np.int32))]
    cut_length = float(np.linalg.norm(mesh.points[cut_edges[:, 0]] - mesh.points[cut_edges[:, 1]], axis=1).sum())
    add('mirror_x_line_tagged', abs(cut_length / (1.6 * _UM) - 1.0) <= _LENGTH_RTOL, cut_length)
    expected = {'p_left': (0.8 * _UM, 0.3 * _UM), 'p_right': (1.2 * _UM, 0.3 * _UM), 'p_mid': (1.0 * _UM, 1.0 * _UM)}
    found = {name: mesh.point_tags.get(name) for name in expected}
    add('mirror_x_point_tags', all(node is not None and np.allclose(mesh.points[node], xy, rtol=0.0, atol=1e-15)
                                   for (name, xy), node in zip(expected.items(), found.values())), found)


def _model_path_checks(add):
    """MeshSpec mirror flags -> TriangleSizing.mirror_axes; mirror_mismatch None for symmetric materials, naming the
    object of an asymmetric one."""
    from ..model.geometry import resolve_geometry
    from ..model.meshing import triangle_sizing
    from ..model.schema import get_registry
    from ..model.triangle_meshing import mirror_mismatch

    registry = get_registry()
    spec = _spec(mirror_x=True, mirror_y=True)
    sizing = triangle_sizing(spec, spec.meshes[0], {}, registry)
    add('meshspec_mirror_axes', sizing.mirror_axes == ('x', 'y'), list(sizing.mirror_axes))
    symmetric = [MaterialSpec('glass', properties={'eps_r': 2.25}, selection=['left_disc', 'right_disc', 'centre_disc',
                                                                              'left_oval', 'right_oval', 'bar'])]
    shapes = resolve_geometry(_spec(symmetric), {})
    add('mirror_check_symmetric', mirror_mismatch(_spec(symmetric), shapes, _WINDOW, 0) is None)
    one_sided = [MaterialSpec('glass', properties={'eps_r': 2.25}, selection=['left_disc', 'centre_disc'])]
    mismatch = mirror_mismatch(_spec(one_sided), shapes, _WINDOW, 0)
    add('mirror_check_names_left_disc', mismatch is not None and 'left_disc' in mismatch[0], mismatch)


def _check_mirror_mesh(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from ..model.triangle_meshing import triangulate

    checks = SubChecks()
    add = checks.add

    spec = _spec()
    mesh = triangulate(spec, {}, sizing=_sizing(('x',)))
    _symmetry_checks(mesh, 0, 'mirror_x', add)
    _tag_checks(mesh, add)
    quarter = triangulate(spec, {}, sizing=_sizing(('x', 'y')))   # the geometry is not y-symmetric: images conform
    _symmetry_checks(quarter, 0, 'mirror_xy_x', add)
    _symmetry_checks(quarter, 1, 'mirror_xy_y', add)
    _model_path_checks(add)
    return checks_result(checks, elements=int(mesh.n_elements), quarter_elements=int(quarter.n_elements))


# ---- parity selection rule (em_fdfd_ez on the mirrored mesh) ----
def _parity_ratios(mirror_axes) -> dict:
    """{'even drive odd part', 'odd drive even part'}: largest opposite-parity part of E_z over the sample pairs /
    largest |E_z|, of a point current on the mirror line and of a +I / -I pair at mirror images."""
    import scipy.sparse.linalg
    from ..model.triangle_meshing import triangulate
    from ..physics.em.boundaries import BoundaryConfig
    from ..physics.em.fem_fdfd import (FEMFDFDMaterials, assemble_fem_fdfd, fem_fdfd_layout, point_fields,
                                       point_load)

    mesh = triangulate(_spec(), {}, sizing=_sizing(mirror_axes, _PARITY_H))
    config = BoundaryConfig('cpml', 'cpml', 'cpml', 'cpml')
    config.cpml_npml = _RING[1]
    config.cpml_kappa_max = 1.0
    config._validate()
    discs = [mesh.region_names.index(name) for name in ('centre_disc', 'left_disc', 'right_disc')]
    eps = np.where(np.isin(mesh.region, discs), _PARITY_EPS, 1.0).astype(np.complex128)
    materials = FEMFDFDMaterials(eps=eps, mu=np.ones(mesh.n_elements, dtype=np.complex128),
                                 pec=np.zeros(mesh.n_elements, dtype=bool))
    layout = fem_fdfd_layout(mesh, 2, config, 'ez')
    omega = 2.0 * math.pi * C0 / _PARITY_WAVELENGTH
    system = assemble_fem_fdfd(layout, materials, omega)
    factor = scipy.sparse.linalg.splu(system.operator.tocsc())
    plane = 0.5 * _WINDOW[0]
    samples = [(x * _UM, y * _UM) for x in _SAMPLE_X for y in _SAMPLE_Y]
    loads = {'even drive odd part': (point_load(layout, _EVEN_SOURCE, 1.0), -1.0),
             'odd drive even part': (point_load(layout, _ODD_SOURCES[0], 1.0)
                                     - point_load(layout, _ODD_SOURCES[1], 1.0), 1.0)}
    ratios = {}
    for name, (load, sign) in loads.items():
        coefficients = layout.coefficients(factor.solve(load))
        values = np.array([point_fields(system, coefficients, point)['ez'] for point in samples])
        images = np.array([point_fields(system, coefficients, (2.0 * plane - x, y))['ez'] for x, y in samples])
        largest = float(max(np.abs(values).max(), np.abs(images).max()))
        ratios[name] = float(np.abs(values + sign * images).max()) / max(largest, 1e-300)
    ratios['unknowns'] = layout.n_unknowns
    return ratios


def _check_parity(results_by_name):
    """Mirrored mesh: both opposite-parity ratios <= _PARITY_TOLERANCE; the plain mesh's ratios are recorded only."""
    mirrored = _parity_ratios(('x',))
    plain = _parity_ratios(())
    worst = max(mirrored['even drive odd part'], mirrored['odd drive even part'])
    return {'measured': worst, 'expected': 0.0, 'error': worst, 'tolerance': _PARITY_TOLERANCE,
            'details': {'mirrored_mesh': mirrored, 'plain_mesh_for_comparison': plain}}


MESH_SYMMETRY_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_mirror_triangle_mesh',
        label='Mirror-symmetric triangle mesh: invariance and tags',
        description=(
            'CPU checks of MeshSpec.mirror_x / mirror_y (new_modules_plan G2, triangle_meshing._mirrored_parts): a '
            '2 x 1.2 um window, symmetric about x = 1 um (a disc on the line, a disc pair, a 30 / 150 deg ellipse '
            'pair, a bar, a line, three points), h 0.08 um, a 0.2 um PML ring of 4 layers on every face. mirror_x: '
            'nodes, elements, ring bits, the centre disc region, curved midside nodes (equal counts per side, on '
            'their curves to 1e-12 x side) and the x_lo / x_hi tags invariant under x -> 2 um - x; area (1e-12), the '
            'line tag length, point tags at exact coordinates. mirror_x + mirror_y (the geometry is not y-symmetric: '
            'the quarter mesh conforms to the images too): invariant under both mirrors. MeshSpec path: '
            'triangle_sizing gives mirror_axes (x, y); mirror_mismatch is None for symmetric materials and names '
            'left_disc when only it and the centre disc carry the material. error = failed sub-checks, tolerance 0. '
            'Written, not run.'),
        specs=lambda: {},
        check=_check_mirror_mesh,
        max_wall_time=300.0),
    SelfTestCase(
        name='fem_mirror_parity_fdfd',
        label='Mirror-symmetric triangle mesh: em_fdfd_ez parity rule',
        description=(
            'CPU (physics/em/fem_fdfd directly, no model run): the fem_mirror_triangle_mesh geometry with eps 4 in '
            'the three discs, vacuum elsewhere, a cpml ring (kappa_max 1) on every face, quadratic, lambda 1 um, h '
            '50 nm, mirror_x mesh, one SuperLU factor. A point current on the mirror line (even drive) and a +I / -I '
            'pair at mirror images (odd drive): the opposite-parity part of E_z over 20 sample pairs (x, 2 um - x) / '
            f'the largest |E_z| <= {_PARITY_TOLERANCE:g} (rounding only; first estimate). The same ratios on the '
            'unmirrored mesh are recorded in details (the mesh asymmetry, ~1e-3 expected). Written, not run.'),
        specs=lambda: {},
        check=_check_parity,
        max_wall_time=300.0),
)
