"""Self-tests of the graded builtin tetrahedron mesher (plan D2; CPU, no gmsh, specs() empty, check({}) directly): the
exact estimate against the built Kuhn split, conforming object faces, the orthoscheme dihedral bound on a 10:1 graded
mesh, the uniform case unchanged, the z axis alone (swept prisms), the mesh key and the refusals."""
import math

import numpy as np

from ..model.schema import SelfTestCase
from ..model.selftest_helpers import SubChecks, checks_result
from ..model.spec import DomainSpec, GeometrySpec, MeshFaceSizeSpec, MeshSpec, ModelSpec, RefinementSpec
from .selftests_mesh_sizes3d import _refusal_text

_UM = 1e-6
_SIZE = (4.0 * _UM, 1.0 * _UM, 0.6 * _UM)
_H_MAX = 0.25 * _UM
_FINE = 0.025 * _UM             # bar refinement across y / z: 10:1 bricks along the bar
_END = 0.05 * _UM               # face size at x_lo / x_hi (clamped string ends)
_BAR = {'x0': 0.33 * _UM, 'y0': 0.41 * _UM, 'z0': 0.2 * _UM, 'x1': 3.67 * _UM, 'y1': 0.59 * _UM, 'z1': 0.3 * _UM}
_PAD = {'x0': 2.55 * _UM, 'y0': 0.1 * _UM, 'z0': 0.1 * _UM, 'x1': 3.45 * _UM, 'y1': 0.3 * _UM, 'z1': 0.45 * _UM}
_AREA_RTOL = 1e-9               # object_faces area vs the box surface (faces snapped exactly)
_DIHEDRAL_SLACK_DEG = 1e-6      # orthoschemes: every dihedral <= 90 deg
_MIN_ASPECT = 8.0               # the graded mesh holds bricks at least this elongated (else the bound tests little)
_GRADED_GAIN = 5.0              # graded elements x this < the uniform mesh at the finest size


def _spec(mesh_spec):
    """Model of the bar (refined across y / z only) and the unrefined pad on one tetrahedra mesh, no physics."""
    spec = ModelSpec(title='Self-test: graded builtin tetrahedra',
                     domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1], size_z=_SIZE[2]))
    spec.geometry = [GeometrySpec('bar', 'box', dict(_BAR)), GeometrySpec('pad', 'box', dict(_PAD))]
    spec.meshes = [mesh_spec]
    return spec


def _graded_mesh_spec(**changes):
    values = dict(name='tets', kind='tetrahedra', mode='nonuniform', dx=_H_MAX, mesher='builtin',
                  refinements=[RefinementSpec('bar', _H_MAX, dx_y=_FINE, dx_z=_FINE)],
                  face_sizes=[MeshFaceSizeSpec('x_lo', _END), MeshFaceSizeSpec('x_hi', _END)])
    values.update(changes)
    return MeshSpec(**values)


def _face_area(mesh, faces) -> float:
    corners = mesh.points[mesh.faces[faces]]
    return float(0.5 * np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
                                      axis=1).sum())


def _box_area(box) -> float:
    a, b, c = (box[f'{axis}1'] - box[f'{axis}0'] for axis in 'xyz')
    return 2.0 * (a * b + b * c + c * a)


def _largest_dihedral_deg(mesh) -> float:
    """Largest dihedral angle [deg] over the elements: pi - angle between barycentric gradients of each vertex pair."""
    corners = mesh.points[mesh.tets]
    spans = corners[:, 1:, :] - corners[:, :1, :]                    # rows p_k - p_0
    gradients = np.linalg.inv(spans)                                  # column k = grad lambda_k (k = 1..3)
    grads = np.concatenate([-gradients.sum(axis=2, keepdims=True), gradients], axis=2).transpose(0, 2, 1)
    unit = grads / np.linalg.norm(grads, axis=2, keepdims=True)
    smallest_cos = min(float((-(unit[:, a] * unit[:, b]).sum(axis=1)).min())
                       for a in range(4) for b in range(a + 1, 4))
    return math.degrees(math.acos(max(-1.0, min(1.0, smallest_cos))))


def _graded_checks(add, registry):
    """Estimate = built mesh, faces conform, dihedral bound, fewer elements than uniform at the fine size, z alone."""
    from ..model.geometry import resolve_geometry
    from ..model.meshing import estimate_mesh, graded_window_edges, tetrahedron_sizing
    from ..model.tetrahedral_meshing import tetrahedralize

    mesh_spec = _graded_mesh_spec()
    spec = _spec(mesh_spec)
    estimate = estimate_mesh(spec, {}, registry, 'tets')
    mesh = tetrahedralize(spec, {}, registry, 'tets')
    counts = {'elements': (estimate.n_elements, mesh.n_elements), 'nodes': (estimate.n_nodes, mesh.n_points),
              'edges': (estimate.n_edges, mesh.n_edges), 'faces': (estimate.n_faces, mesh.n_faces)}
    add('estimate_equals_built', all(planned == built for planned, built in counts.values()), counts)
    areas = {name: (_face_area(mesh, mesh.object_faces[name]), _box_area(box))
             for name, box in (('bar', _BAR), ('pad', _PAD))}
    add('object_faces_conform', all(abs(found - exact) <= _AREA_RTOL * exact for found, exact in areas.values()),
        areas)
    widths = [np.diff(edges) for edges in estimate.sizing.axis_edges]
    cells = np.stack(np.broadcast_arrays(widths[0][:, None, None], widths[1][None, :, None],
                                         widths[2][None, None, :]))
    aspect = float((cells.max(axis=0) / cells.min(axis=0)).max())
    largest = _largest_dihedral_deg(mesh)
    add('dihedral_at_most_90', aspect >= _MIN_ASPECT and largest <= 90.0 + _DIHEDRAL_SLACK_DEG,
        {'largest_dihedral_deg': largest, 'largest_brick_aspect': aspect, 'radius_ratio_min': mesh.quality.smallest})
    add('no_sliver_count', mesh.quality.n_poor == 0,   # orthoschemes never count as slivers (no obtuse dihedral)
        {'n_poor': mesh.quality.n_poor, 'radius_ratio_min': mesh.quality.smallest})
    uniform = estimate_mesh(_spec(MeshSpec(name='tets', kind='tetrahedra', mode='uniform', dx=_FINE)), {}, registry,
                            'tets')
    add('graded_fewer_elements', _GRADED_GAIN * estimate.n_elements < uniform.n_elements,
        {'graded': estimate.n_elements, 'uniform_fine': uniform.n_elements})
    shapes = resolve_geometry(spec, {})
    sizing = tetrahedron_sizing(spec, mesh_spec, {}, registry, shapes)
    z_alone = graded_window_edges(spec, mesh_spec, {}, shapes, sizing, axes='z')
    add('z_axis_alone', set(z_alone) == {'z'} and np.array_equal(z_alone['z'], sizing.axis_edges[2]),
        len(z_alone.get('z', ())))


def _uniform_and_key_checks(add, registry):
    """Uniform mode = the old equal cells (refinements ignored, no snaps); the mesh key follows a per-axis size."""
    from ..model.meshing import element_mesh_key, kuhn_axis_edges, shell_axis_edges, tetrahedron_sizing

    spec = _spec(_graded_mesh_spec(mode='uniform', face_sizes=[]))
    sizing = tetrahedron_sizing(spec, spec.meshes[0], {}, registry)
    expected = [kuhn_axis_edges(size, _H_MAX, 0.0, 0.0) for size in _SIZE]
    add('uniform_unchanged', len(sizing.axis_edges) == 3
        and all(np.array_equal(found, wanted) for found, wanted in zip(sizing.axis_edges, expected)),
        [len(edges) for edges in sizing.axis_edges])
    first = tetrahedron_sizing(_spec(_graded_mesh_spec()), _graded_mesh_spec(), {}, registry)
    finer = _graded_mesh_spec(refinements=[RefinementSpec('bar', _H_MAX, dx_y=0.5 * _FINE, dx_z=_FINE)])
    second = tetrahedron_sizing(_spec(finer), finer, {}, registry)
    add('mesh_key_follows_grading', element_mesh_key(first, _SIZE, {}) != element_mesh_key(second, _SIZE, {}),
        [first.axis_edges_digest, second.axis_edges_digest])
    shelled = shell_axis_edges(np.linspace(0.0, 1.0, 5), 0.25, 0.5, 0.75)
    add('shell_cells', len(shelled) == 5 + 2 + 3 and math.isclose(shelled[0], -0.5)
        and math.isclose(shelled[-1], 1.75), shelled.tolist())


def _refusal_checks(add, registry):
    """size_points refused with builtin; axes options and per-axis sizes still refused with gmsh."""
    from ..model.meshing import tetrahedron_sizing
    from ..model.spec import MeshAxisSpec

    cases = {'size_points_builtin': _graded_mesh_spec(size_points=[[1.0 * _UM, 0.5 * _UM, 0.3 * _UM, _FINE]]),
             'axes_gmsh': _graded_mesh_spec(mesher='gmsh', refinements=[], axes=[MeshAxisSpec('z')]),
             'per_axis_gmsh': _graded_mesh_spec(mesher='gmsh')}
    refused = {name: _refusal_text(lambda: tetrahedron_sizing(_spec(mesh_spec), mesh_spec, {}, registry))
               for name, mesh_spec in cases.items()}
    add('refusals', all(refused.values()), refused)


def _check_builtin_tetrahedra(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from ..model.schema import get_registry

    checks = SubChecks()
    add = checks.add

    registry = get_registry()
    _graded_checks(add, registry)
    _uniform_and_key_checks(add, registry)
    _refusal_checks(add, registry)
    return checks_result(checks)


BUILTIN_TETRAHEDRA_SELFTEST_CASES = (
    SelfTestCase(
        name='fem_builtin_tetrahedra_graded',
        label='Tetrahedron meshes: graded builtin Kuhn split',
        description=(
            'CPU checks of the graded builtin tetrahedra (plan D2, no gmsh): a 4 x 1 x 0.6 um box with a bar refined '
            f'across y / z to {_FINE / _UM:g} um (dx {_H_MAX / _UM:g} um, so 10:1 bricks) and face sizes '
            f'{_END / _UM:g} um at x_lo / x_hi, plus an unrefined pad box. The estimate equals the built mesh exactly '
            '(elements, nodes, edges, faces); the object_faces areas of both boxes equal their surfaces '
            f'({_AREA_RTOL:g}: refined and unrefined flat faces snapped); every dihedral angle <= 90 deg (Kuhn '
            f'orthoschemes) with bricks >= {_MIN_ASPECT:g}:1 and no element counted poor (mesh quality n_poor 0: no '
            f'sliver warning); > {_GRADED_GAIN:g}x fewer elements than uniform at the '
            'fine size; graded_window_edges of z alone = the sizing z edges (swept prisms); uniform mode = the old '
            'equal cells; the element-mesh key changes with a per-axis size; shell_axis_edges cell counts; '
            'size_points refused with builtin, axes / per-axis sizes refused with gmsh. error = failed sub-checks, '
            'tolerance 0.'),
        specs=lambda: {},
        check=_check_builtin_tetrahedra,
        max_wall_time=120.0),
)
