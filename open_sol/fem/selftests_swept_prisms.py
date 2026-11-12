"""Self-tests of swept prism layers (experiment plan D5; CPU, no gmsh, specs() empty, check({}) directly): the
minimum-vertex split (conformity, counts, volume, a P2 patch test, periodic partners) and the model path (estimate vs
built, layer planes, conforming walls, refusals)."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from ..model.spec import (DomainSpec, GeometrySpec, MeshAxisSpec, MeshFaceSizeSpec, MeshSpec, ModelSpec,
                          RefinementSpec)
from .selftests_mesh_sizes3d import _refusal_text

_UM = 1e-6
_SQUARE = (1.0 * _UM, 0.8 * _UM)                     # footprint of the fem checks [m]
_SQUARE_CELLS = 6                                    # boundary nodes per side - 1
_Z_EDGES = np.array([0.0, 0.1, 0.25, 0.5]) * _UM     # graded layers
_EXACT_RTOL = 1e-10                                  # volumes, areas, the P2 patch test (host SuperLU)
# model checks: a 4 x 3 x 0.6 um box with a plate (z 0.2 .. 0.4 um, layers of 0.05 um), a z cylinder and a sphere
_SIZE = (4.0 * _UM, 3.0 * _UM, 0.6 * _UM)
_H = 0.25 * _UM
_LAYER = 0.05 * _UM
_PLATE = {'x0': 0.5 * _UM, 'y0': 0.5 * _UM, 'z0': 0.2 * _UM, 'x1': 2.5 * _UM, 'y1': 2.0 * _UM, 'z1': 0.4 * _UM}
_DISC = {'cx': 3.2 * _UM, 'cy': 2.2 * _UM, 'r': 0.5 * _UM, 'z0': 0.0, 'z1': _SIZE[2]}   # the whole height
_BALL = {'cx': 1.5 * _UM, 'cy': 2.5 * _UM, 'cz': 0.3 * _UM, 'r': 0.25 * _UM}
_ESTIMATE_RTOL = 0.3            # estimated vs built elements (the 2D TRIANGLE_FILL; first estimate)
_DISC_RTOL = 0.03               # disc membership volume vs pi r^2 h (faceted outline at curvature_factor 0.3)
_BALL_RTOL = 0.35               # sphere tagged per element (staircase of 0.25 um prisms)


def _square_triangles(periodic=False):
    """(points (n, 2), triangles (t, 3)) of the square: a grid of _SQUARE_CELLS^2 cells, interior nodes jittered by
    < 1 / 4 cell, each cell cut along a diagonal alternating with the cell parity; boundary nodes regular (opposite
    sides match)."""
    cells = _SQUARE_CELLS
    t = np.linspace(0.0, 1.0, cells + 1)
    grid = np.stack(np.meshgrid(t, t, indexing='ij'), axis=-1)   # (i, j) -> node i (cells + 1) + j
    jitter = np.random.default_rng(3 if periodic else 5).uniform(-0.2, 0.2, grid.shape) / cells
    grid[1:-1, 1:-1] += jitter[1:-1, 1:-1]
    points = grid.reshape(-1, 2) * np.array(_SQUARE)
    triangles = []
    for i in range(cells):
        for j in range(cells):
            a, b, c, d = (i * (cells + 1) + j, (i + 1) * (cells + 1) + j, (i + 1) * (cells + 1) + j + 1,
                          i * (cells + 1) + j + 1)
            triangles += [(a, b, c), (a, c, d)] if (i + j) % 2 == 0 else [(a, b, d), (b, c, d)]
    return points, np.array(triangles, dtype=np.int64)


def _unique_edge_count(triangles) -> int:
    edges = np.sort(np.concatenate([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]]), axis=1)
    return len(np.unique(edges, axis=0))


def _split_checks(add):
    """Counts, conformity (boundary faces = the box surface, all on its sides), volume, random order keys, refusal."""
    from .assembly3d import face_areas_3d
    from .prism_layers import split_prisms, swept_counts, swept_tetrahedron_mesh
    points, triangles = _square_triangles()
    mesh = swept_tetrahedron_mesh(points, triangles, _Z_EDGES)
    counts = swept_counts(len(triangles), len(points), _unique_edge_count(triangles), len(_Z_EDGES) - 1)
    built = {'elements': mesh.n_elements, 'nodes': mesh.n_points, 'edges': mesh.n_edges, 'faces': mesh.n_faces}
    add('counts', counts == built, {'formula': counts, 'built': built})
    size_x, size_y = _SQUARE
    height = float(_Z_EDGES[-1])
    volume = size_x * size_y * height
    add('volume', abs(float(mesh.volumes().sum()) / volume - 1.0) <= _EXACT_RTOL, float(mesh.volumes().sum()))
    boundary = np.flatnonzero(mesh.face_elements[:, 1] < 0)
    surface = 2.0 * (size_x * size_y + size_y * height + height * size_x)
    tagged = sum(len(faces) for faces in mesh.face_tags.values())
    add('conforming_surface', abs(float(face_areas_3d(mesh, boundary).sum()) / surface - 1.0) <= _EXACT_RTOL
        and tagged == len(boundary), {'boundary_faces': len(boundary), 'tagged': tagged})
    shuffled = swept_tetrahedron_mesh(points, triangles, _Z_EDGES,
                                      order_key=np.random.default_rng(11).permutation(len(points)))
    add('any_order_key_conforms', len(np.flatnonzero(shuffled.face_elements[:, 1] < 0)) == len(boundary),
        shuffled.n_faces)
    try:
        split_prisms(np.arange(6)[None, :], np.array([[0, 1, 2, 3, 4, 0]]))
        add('repeated_key_refused', False)
    except ValueError as exc:
        add('repeated_key_refused', True, str(exc))


def _patch_checks(add):
    """P2 on a swept mesh reproduces the harmonic quadratic u = x^2 + 2 y^2 - 3 z^2 + x y + y z (k = 1) from its
    boundary values exactly."""
    from .diffusion3d import FEMDiffusionSystem3D
    from .prism_layers import swept_tetrahedron_mesh
    points, triangles = _square_triangles()
    mesh = swept_tetrahedron_mesh(points, triangles, _Z_EDGES)
    system = FEMDiffusionSystem3D(mesh, 2)
    dof_points = system.full.dof_points() / _UM

    def exact(at):
        x, y, z = at.T
        return x ** 2 + 2.0 * y ** 2 - 3.0 * z ** 2 + x * y + y * z

    low, high = mesh.points.min(axis=0) / _UM, mesh.points.max(axis=0) / _UM
    on_side = ((np.abs(dof_points - low) <= 1e-9) | (np.abs(dof_points - high) <= 1e-9)).any(axis=1)
    fixed = np.flatnonzero(on_side)
    system.add_fixed_dofs(fixed, exact(dof_points[fixed]))
    system.assemble(np.broadcast_to(np.eye(3), (mesh.n_elements, 3, 3)))
    system.solve()
    scale = float(np.abs(exact(dof_points)).max())
    error = float(np.abs(system.values - exact(dof_points)).max()) / scale
    add('p2_patch_quadratic', error <= _EXACT_RTOL, {'error': error, 'free_dofs': int((~on_side).sum())})


def _periodic_checks(add):
    """periodic_order_key on x and y: node and P2 edge partners exist on all three axes (Lagrange periodic dofs)."""
    from .dofs3d import lagrange_dofs_3d, periodic_dof_partners
    from .mesh3d import periodic_node_partners
    from .prism_layers import periodic_order_key, swept_tetrahedron_mesh
    points, triangles = _square_triangles(periodic=True)
    mesh = swept_tetrahedron_mesh(points, triangles, _Z_EDGES, order_key=periodic_order_key(points, ('x', 'y')))
    dofs = lagrange_dofs_3d(mesh, 2)
    found = {}
    for axis in range(3):
        try:
            partners = periodic_dof_partners(dofs, periodic_node_partners(mesh, axis))
            found['xyz'[axis]] = int((partners >= 0).sum())
        except ValueError as exc:
            found['xyz'[axis]] = str(exc)
    add('periodic_p2_partners', all(isinstance(count, int) and count > 0 for count in found.values()), found)


def _model_spec(**mesh_changes):
    values = dict(name='swept', kind='tetrahedra', mode='nonuniform', dx=_H, mesher='builtin', sweep_axis='z',
                  refinements=[RefinementSpec('plate', _H, dx_z=_LAYER)])
    values.update(mesh_changes)
    spec = ModelSpec(title='Self-test: swept prism layers',
                     domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1], size_z=_SIZE[2]))
    spec.geometry = [GeometrySpec('plate', 'box', dict(_PLATE)), GeometrySpec('disc', 'cylinder_z', dict(_DISC)),
                     GeometrySpec('ball', 'sphere', dict(_BALL))]
    spec.meshes = [MeshSpec(**values)]
    return spec


def _model_checks(add, registry):
    """Estimate vs built (elements within the fill tolerance, layers exact), plate faces are mesh faces, disc and
    sphere volumes, the mesh key."""
    from ..model.geometry import resolve_geometry
    from ..model.meshing import element_mesh_key, estimate_mesh, tetrahedron_sizing
    from ..model.tetrahedral_meshing import element_membership, tetrahedralize
    from .assembly3d import face_areas_3d
    spec = _model_spec()
    estimate = estimate_mesh(spec, {}, registry, 'swept')
    mesh = tetrahedralize(spec, {}, registry, 'swept')
    layer_edges = np.asarray(estimate.sizing.layer_edges)
    built_planes = np.unique(np.round(mesh.points[:, 2] / _UM, 9))
    add('layers_exact', len(built_planes) == len(layer_edges)
        and np.allclose(built_planes, layer_edges / _UM, rtol=0.0, atol=1e-9), len(layer_edges) - 1)
    add('plate_faces_are_layers', all(np.min(np.abs(layer_edges - _PLATE[key])) <= 1e-9 * _SIZE[2]
                                      for key in ('z0', 'z1')), layer_edges.tolist())
    ratio = estimate.n_elements / mesh.n_elements
    add('estimate_elements', abs(ratio - 1.0) <= _ESTIMATE_RTOL,
        {'estimate': estimate.n_elements, 'built': mesh.n_elements, 'nodes': (estimate.n_nodes, mesh.n_points)})
    sides = [_PLATE[f'{axis}1'] - _PLATE[f'{axis}0'] for axis in 'xyz']
    plate_area = 2.0 * (sides[0] * sides[1] + sides[1] * sides[2] + sides[2] * sides[0])
    found_area = float(face_areas_3d(mesh, mesh.object_faces['plate']).sum())
    add('plate_faces_conform', abs(found_area / plate_area - 1.0) <= 1e-9, {'found': found_area, 'box': plate_area})
    shapes = resolve_geometry(spec, {})
    volumes = mesh.volumes()
    disc = float(volumes[element_membership(mesh, 'disc', shapes)].sum())
    disc_exact = math.pi * _DISC['r'] ** 2 * _SIZE[2]
    add('disc_volume', abs(disc / disc_exact - 1.0) <= _DISC_RTOL, {'found': disc, 'exact': disc_exact})
    ball = float(volumes[element_membership(mesh, 'ball', shapes)].sum())
    ball_exact = 4.0 / 3.0 * math.pi * _BALL['r'] ** 3
    add('sphere_tagged_per_element', abs(ball / ball_exact - 1.0) <= _BALL_RTOL, {'found': ball, 'exact': ball_exact})
    plain = _model_spec(sweep_axis='none')
    first = tetrahedron_sizing(spec, spec.meshes[0], {}, registry)
    second = tetrahedron_sizing(plain, plain.meshes[0], {}, registry)
    add('mesh_key_follows_sweep', element_mesh_key(first, _SIZE, {}) != element_mesh_key(second, _SIZE, {}),
        first.layer_edges_digest)


def _refusal_checks(add, registry):
    """size_points, x / y axes options, dx_x unlike dx, an x face size and where 'edges' raise with sweep_axis 'z'."""
    from ..model.meshing import tetrahedron_sizing
    cases = {'size_points': dict(size_points=[[1.0 * _UM, 1.0 * _UM, 0.3 * _UM, _LAYER]]),
             'x_axis_options': dict(axes=[MeshAxisSpec('x')]),
             'dx_x_unlike_dx': dict(refinements=[RefinementSpec('plate', _H, dx_x=0.5 * _H, dx_z=_LAYER)]),
             'x_face_size': dict(face_sizes=[MeshFaceSizeSpec('x_lo', _LAYER)]),
             'where_edges': dict(refinements=[RefinementSpec('plate', 0.5 * _H, where='edges')])}
    refused = {}
    for name, changes in cases.items():
        spec = _model_spec(**changes)
        refused[name] = _refusal_text(lambda: tetrahedron_sizing(spec, spec.meshes[0], {}, registry))
    allowed = _model_spec(face_sizes=[MeshFaceSizeSpec('plate', _LAYER, face='z_hi')],
                          axes=[MeshAxisSpec('z', edges=[0.3 * _UM])])
    try:
        tetrahedron_sizing(allowed, allowed.meshes[0], {}, registry)
        refused['z_options_allowed'] = 'accepted'
    except ValueError as exc:
        refused['z_options_allowed'] = ''
        refused['z_options_error'] = str(exc)
    add('refusals', all(refused.values()), refused)


def _check_split(results_by_name):
    checks = SubChecks()
    add = checks.add

    _split_checks(add)
    _patch_checks(add)
    _periodic_checks(add)
    return checks_result(checks)


def _check_model(results_by_name):
    from ..model.schema import get_registry
    checks = SubChecks()
    add = checks.add

    registry = get_registry()
    _model_checks(add, registry)
    _refusal_checks(add, registry)
    return checks_result(checks)


SWEPT_PRISMS_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_swept_prisms_split',
        label='Tetrahedron meshes: swept prism layers, the minimum-vertex split',
        description=(
            'CPU checks of fem.prism_layers (plan D5, no model): a 6 x 6 cell triangulation of a 1 x 0.8 um square '
            '(regular boundary nodes, jittered interior, alternating diagonals) swept through 3 graded layers (0.1 / '
            '0.15 / 0.25 um). '
            'Elements / nodes / edges / faces = swept_counts; volume = the box and the boundary faces = its '
            f'surface, all on its sides ({_EXACT_RTOL:g}: conforming); a random order key still conforms; a '
            'prism repeating a key is refused; P2 diffusion (k = 1) reproduces u = x^2 + 2 y^2 - 3 z^2 + x y + y z '
            f'from its boundary values within {_EXACT_RTOL:g}; with periodic_order_key on x and y the Lagrange P2 '
            'periodic partners exist on x, y and z. error = failed sub-checks, tolerance 0 (not yet run).'),
        specs=lambda: {},
        check=_check_split,
        max_wall_time=120.0),
    SelfTestCase(
        name='fem_swept_prisms_model',
        label='Tetrahedron meshes: swept prism layers of a model',
        description=(
            "CPU checks of MeshSpec.sweep_axis 'z' (plan D5, builtin 2D mesher): a 4 x 3 x 0.6 um box holding a "
            'plate box (z 0.2 .. 0.4 um, refinement dx_z 0.05 um), a z cylinder through the height and a sphere, dx '
            '0.25 um. The layer planes of the estimate are the built ones (exact) and hold the plate faces; '
            f'estimated elements within {_ESTIMATE_RTOL:g} of the built ones; the plate object_faces area = its '
            f'surface (1e-9: conforming walls and layers); disc volume within {_DISC_RTOL:g} of pi r^2 h, the '
            f'sphere (tagged per element) within {_BALL_RTOL:g}; the element-mesh key changes with sweep_axis; '
            'size_points, x axes options, dx_x unlike dx, an x_lo face size and where edges are refused, z face '
            'sizes and a z axis accepted. error = failed sub-checks, tolerance 0 (first estimates, not yet run).'),
        specs=lambda: {},
        check=_check_model,
        max_wall_time=180.0),
)
