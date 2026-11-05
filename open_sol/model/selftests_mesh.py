"""CPU self-tests of the mesh spec: the S7.1 tensor-grid options (anisotropic refinements, coordinate bands, per-axis
max_ratio, custom edges / segment profiles (2D and z), outer grading) and the S7.2 / S7.5 / S7.7 spec (named meshes per
physics, their validation, the triangle size estimate, adaptation); mesh estimates and validation, plus builtin-mesher
triangle meshes for the element-count estimate."""
import math

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import SubChecks
from .spec import (AdaptationSpec, AnalyticCurveSpec, CouplingSpec, DerivedValueSpec, DomainSpec, GeometrySpec, MeshAxisSpec,
                   MeshBandSpec, MeshSegmentSpec, MeshSpec, ModelSpec, PhysicsSpec, RefinementSpec, StudySpec)

_SIZE = (10e-6, 4e-6)                # m
_BASE, _FINE, _BAND = 100e-9, 20e-9, 25e-9   # m
_RATIO, _BAND_RATIO = 1.15, 1.3
_RATIO_SLACK = 1.05                  # snap-node taper (core sub file) and float32 widths
_RESCALE_SLACK = 1.02                # an unsnapped axis is rescaled by up to half a base cell over its length
_OUTER = dict(size=1e-3, base=1e-6, fine=0.2e-6, outer_dx=100e-6, ratio=1.3)   # m / 1


def _spec(mesh, size=_SIZE, size_z=None, geometry=()):
    spec = ModelSpec(title='Self-test: S7.1 mesh options', domain=DomainSpec(size_x=size[0], size_y=size[1],
                                                                              size_z=size_z))
    spec.geometry = list(geometry)
    spec.mesh = mesh
    return spec


def _ratio(widths):
    ratios = widths[1:] / widths[:-1]
    return float(np.max(np.maximum(ratios, 1.0 / ratios))) if ratios.size else 1.0


def _check_mesh_options(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from ..core.mesh import segment_widths
    from .meshing import estimate_mesh, line_block_hint
    from .schema import get_registry
    from .validate import validate

    registry = get_registry()
    checks = SubChecks()
    add = checks.add

    waveguide = GeometrySpec('wg', 'rectangle', {'x0': 0.0, 'y0': 1.8e-6, 'x1': _SIZE[0], 'y1': 2.2e-6})
    aniso = _spec(MeshSpec(mode='nonuniform', dx=_BASE, max_ratio=_RATIO,
                           refinements=[RefinementSpec('wg', _FINE, 0.0, dx_x=_BASE)]), geometry=[waveguide])
    mesh = estimate_mesh(aniso, {}, registry)
    add('anisotropic_x_untouched', mesh.nx == round(_SIZE[0] / _BASE) and _ratio(mesh.dx_cells) < 1.0 + 1e-6,
        {'nx': mesh.nx})
    inside = mesh.dy_cells[(mesh.y_edges[:-1] >= 1.8e-6 - 1e-12) & (mesh.y_edges[1:] <= 2.2e-6 + 1e-12)]
    add('anisotropic_y_fine', inside.size == round(0.4e-6 / _FINE)
        and float(np.max(np.abs(inside / _FINE - 1.0))) < 1e-3, int(inside.size))
    add('line_block_hint', set(mesh.line_blocks) == {'x', 'y'} and 'cheaper' in line_block_hint(mesh),
        {axis: list(value) for axis, value in mesh.line_blocks.items()})

    band = _spec(MeshSpec(mode='nonuniform', dx=_BASE, max_ratio=_RATIO, axes=[
        MeshAxisSpec('x', max_ratio=_BAND_RATIO, bands=[MeshBandSpec(4e-6, 5e-6, _BAND)])]))
    mesh = estimate_mesh(band, {}, registry)
    in_band = mesh.dx_cells[(mesh.x_edges[:-1] >= 4e-6 + _BAND) & (mesh.x_edges[1:] <= 5e-6 - _BAND)]
    add('band_fine_inside', in_band.size > 0 and float(np.max(in_band)) <= _BAND * _RESCALE_SLACK, int(in_band.size))
    add('band_axis_ratio', _BAND_RATIO < _ratio(mesh.dx_cells) * 1.2 and _ratio(mesh.dx_cells)
        <= _RATIO_SLACK * _BAND_RATIO, _ratio(mesh.dx_cells))
    add('band_y_uniform', mesh.ny == round(_SIZE[1] / _BASE), mesh.ny)

    segments = [MeshSegmentSpec(20, 1e-6, 'geometric', 0.2), MeshSegmentSpec(40, 3e-6, 'tanh', 5.0),
                MeshSegmentSpec(20, None, 'geometric', 5.0)]
    custom = _spec(MeshSpec(mode='nonuniform', dx=_BASE, axes=[
        MeshAxisSpec('y', segments=segments), MeshAxisSpec('x', edges=[1e-6 * k for k in range(1, 10)])]))
    mesh = estimate_mesh(custom, {}, registry)
    dy, y_edges = mesh.dy_cells, mesh.y_edges
    add('segments_cells_and_ends', mesh.ny == 80 and abs(y_edges[20] - 1e-6) < 1e-12
        and abs(y_edges[60] - 3e-6) < 1e-12 and abs(y_edges[-1] - _SIZE[1]) < 1e-12, mesh.ny)
    add('geometric_ratio', abs(dy[19] / dy[0] - 0.2) < 1e-4 and abs(dy[79] / dy[60] - 5.0) < 1e-4,
        [float(dy[19] / dy[0]), float(dy[79] / dy[60])])
    tanh_piece = dy[20:60]
    add('tanh_ratio', abs(float(tanh_piece.max() / tanh_piece.min()) - 5.0) < 1e-3 and tanh_piece[0] < tanh_piece[20],
        float(tanh_piece.max() / tanh_piece.min()))
    add('custom_edges_completed', mesh.nx == 10 and _ratio(mesh.dx_cells) < 1.0 + 1e-6
        and set(mesh.custom_axes) == {'x', 'y'}, {'nx': mesh.nx, 'custom_axes': list(mesh.custom_axes)})
    widths = segment_widths(1.0, 31, 'tanh', 8.0)
    add('segment_widths_tanh', abs(widths.sum() - 1.0) < 1e-12 and abs(widths.max() / widths.min() - 8.0) < 1e-6
        and abs(widths[0] - widths[-1]) < 1e-12, float(widths.max() / widths.min()))

    size, base = _OUTER['size'], _OUTER['base']
    dot = GeometrySpec('dot', 'rectangle', {'x0': 0.5 * size - 1e-6, 'y0': 0.5 * size - 1e-6, 'x1': 0.5 * size + 1e-6,
                                            'y1': 0.5 * size + 1e-6})
    outer = _spec(MeshSpec(mode='nonuniform', dx=base, max_ratio=_RATIO, outer_dx=_OUTER['outer_dx'],
                           outer_ratio=_OUTER['ratio'], refinements=[RefinementSpec('dot', _OUTER['fine'])]),
                  size=(size, size), geometry=[dot])
    mesh = estimate_mesh(outer, {}, registry)
    dx_max = float(mesh.dx_cells.max())   # the end-cell rescale may stretch the outermost cells by ~10 %
    add('outer_grading_cells', mesh.nx < 0.2 * size / base and 0.5 * _OUTER['outer_dx'] < dx_max
        <= 1.25 * _OUTER['outer_dx'], {'nx': mesh.nx, 'dx_max': dx_max})
    add('outer_grading_ratio', _ratio(mesh.dx_cells) <= _RATIO_SLACK * _OUTER['ratio'], _ratio(mesh.dx_cells))

    stack = _spec(MeshSpec(mode='nonuniform', dx=_BASE, axes=[
        MeshAxisSpec('z', segments=[MeshSegmentSpec(12, None, 'tanh', 3.0)])]), size=(1e-6, 1e-6), size_z=1e-6)
    mesh = estimate_mesh(stack, {}, registry)
    add('custom_z_segments', mesh.nz == 12 and abs(float(mesh.dz_cells.sum()) - 1e-6) < 1e-12
        and abs(float(mesh.dz_cells.max() / mesh.dz_cells.min()) - 3.0) < 1e-3, mesh.nz)

    data = custom.to_dict()
    add('spec_round_trip', ModelSpec.from_dict(data).to_dict() == data)
    flat_z = _spec(MeshSpec(mode='nonuniform', dx=_BASE, axes=[MeshAxisSpec('z', edges=[1e-7])]))
    z_errors = [issue.message for issue in validate(flat_z, registry)
                if issue.level == 'error' and 'needs a 3D model' in issue.message]
    add('z_axis_rejected_in_2d', bool(z_errors), z_errors)

    return checks.result()


_MESH_FINE_DX, _MESH_COARSE_DX = 100e-9, 200e-9   # m: the two uniform meshes of the per-physics checks
_TRIANGLE_H = 250e-9                     # m: maximum element size of the triangle estimate checks


def _meshes_spec(physics, meshes, couplings=(), geometry=()):
    """2D 10 x 4 um model with the given physics, meshes and couplings."""
    spec = ModelSpec(title='Self-test: per-physics meshes', domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]))
    spec.meshes, spec.physics = list(meshes), list(physics)
    spec.couplings, spec.geometry = list(couplings), list(geometry)
    return spec


def _messages(issues, level, path, text):
    """Messages of the issues at path of the level that contain text."""
    return [issue.message for issue in issues if issue.level == level and issue.path == path and text in issue.message]


def _check_per_physics_spec(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from .. import api
    from .meshing import TRIANGLE_FILL, estimate_memory, estimate_mesh, estimate_meshes, triangle_sizing
    from .schema import get_registry, solver_grid
    from .validate import validate

    registry = get_registry()
    checks = SubChecks()
    add = checks.add

    one = _meshes_spec([PhysicsSpec('heat', 'heat')], [MeshSpec(mode='uniform', dx=_MESH_FINE_DX)])
    data = one.to_dict()
    data['mesh'] = data.pop('meshes')[0]
    loaded = ModelSpec.from_dict(data)
    add('legacy_mesh_key', loaded.to_dict() == one.to_dict() and len(loaded.meshes) == 1)
    try:
        ModelSpec.from_dict(dict(data, meshes=[data['mesh']]))
        add('legacy_and_meshes_rejected', False)
    except ValueError as exc:
        add('legacy_and_meshes_rejected', True, str(exc))

    split = _meshes_spec([PhysicsSpec('heat', 'heat', mesh='fine'), PhysicsSpec('currents', 'electric_currents',
                                                                                 mesh='coarse')],
                         [MeshSpec('fine', mode='uniform', dx=_MESH_FINE_DX),
                          MeshSpec('coarse', mode='uniform', dx=_MESH_COARSE_DX)])
    meshes = estimate_meshes(split, {}, registry)
    cells = {name: mesh.n_cells for name, mesh in meshes.items()}
    add('per_mesh_estimates', cells == {
        'fine': round(_SIZE[0] / _MESH_FINE_DX) * round(_SIZE[1] / _MESH_FINE_DX),
        'coarse': round(_SIZE[0] / _MESH_COARSE_DX) * round(_SIZE[1] / _MESH_COARSE_DX)}, cells)
    by_mesh = {}
    total = estimate_memory(split, {}, registry, meshes, by_mesh=by_mesh)
    add('memory_sum_of_parts', set(by_mesh) == {'fine', 'coarse'}
        and total.gpu_steady == sum(part.gpu_steady for part in by_mesh.values()),
        {'total': total.gpu_steady, 'parts': {name: part.gpu_steady for name, part in by_mesh.items()}})
    heat_alone = _meshes_spec([PhysicsSpec('heat', 'heat')], [MeshSpec('fine', mode='uniform', dx=_MESH_FINE_DX)])
    alone = estimate_memory(heat_alone, {}, registry)
    add('mesh_part_equals_alone', alone.gpu_steady == by_mesh['fine'].gpu_steady,
        [alone.gpu_steady, by_mesh['fine'].gpu_steady])
    issues = validate(split, registry)
    add('several_meshes_allowed', not _messages(issues, 'error', ('meshes',), ''),
        _messages(issues, 'error', ('meshes',), ''))

    em_split = _meshes_spec([PhysicsSpec('em', 'em_fdfd_ez'), PhysicsSpec('em2', 'em_fdfd_ez', mesh='other')],
                            [MeshSpec(mode='uniform', dx=_MESH_FINE_DX),
                             MeshSpec('other', mode='uniform', dx=_MESH_COARSE_DX)])
    add('split_em_rejected',
        bool(_messages(validate(em_split, registry), 'error', ('physics', 'em2'), 'EM physics share one mesh')))
    pockels = _meshes_spec([PhysicsSpec('em', 'em_fdfd_ez'), PhysicsSpec('es', 'electrostatics', mesh='other')],
                           [MeshSpec(mode='uniform', dx=_MESH_FINE_DX),
                            MeshSpec('other', mode='uniform', dx=_MESH_COARSE_DX)],
                           couplings=[CouplingSpec('pockels', 'electrostatics_em_pockels', 'es', 'em')])
    add('one_mesh_coupling_rejected', bool(_messages(validate(pockels, registry), 'error', ('coupling', 'pockels'),
                                                     'within one mesh only')))
    unknown = _meshes_spec([PhysicsSpec('heat', 'heat', mesh='nope')], [MeshSpec(mode='uniform', dx=_MESH_FINE_DX)])
    add('unknown_mesh_rejected', bool(_messages(validate(unknown, registry), 'error', ('physics', 'heat'),
                                                'does not exist')))
    unused = _meshes_spec([PhysicsSpec('heat', 'heat')], [MeshSpec(mode='uniform', dx=_MESH_FINE_DX),
                                                         MeshSpec('spare', mode='uniform', dx=_MESH_COARSE_DX)])
    add('unused_mesh_warning', bool(_messages(validate(unused, registry), 'warning', ('mesh', 'spare'),
                                              'no physics uses')))
    triangle_axes = _meshes_spec([], [MeshSpec(kind='triangles', mode='nonuniform', dx=_TRIANGLE_H,
                                               axes=[MeshAxisSpec('x', edges=[1e-6])])])
    add('axes_on_triangles_rejected', bool(_messages(validate(triangle_axes, registry), 'error', ('mesh', 'mesh'),
                                                     'tensor meshes only')))
    on_triangles = _meshes_spec([PhysicsSpec('fluid', 'laminar_flow')], [MeshSpec(kind='triangles', dx=_TRIANGLE_H)])
    add('physics_grids_checked', bool(_messages(validate(on_triangles, registry), 'error', ('physics', 'fluid'),
                                                'does not run on a triangles mesh'))
        and solver_grid(on_triangles, on_triangles.physics[0]) == 'triangles')

    plain = _meshes_spec([], [MeshSpec(kind='triangles', dx=_TRIANGLE_H)])
    estimate = estimate_mesh(plain, {}, registry)
    expected = TRIANGLE_FILL * 4.0 / math.sqrt(3.0) * _SIZE[0] * _SIZE[1] / _TRIANGLE_H ** 2
    add('triangle_count_uniform', estimate.kind == 'triangles' and abs(estimate.n_elements / expected - 1.0) < 1e-3,
        [estimate.n_elements, expected])
    sizing = triangle_sizing(plain, plain.meshes[0], {}, registry)
    add('triangle_h_min_default', math.isclose(sizing.h_min, _TRIANGLE_H / 100.0) and sizing.growth == 1.3,
        [sizing.h_min, sizing.growth])
    dot = GeometrySpec('dot', 'circle', {'cx': 5e-6, 'cy': 2e-6, 'r': 0.5e-6})
    refined = _meshes_spec([], [MeshSpec(kind='triangles', mode='nonuniform', dx=_TRIANGLE_H,
                                         refinements=[RefinementSpec('dot', _TRIANGLE_H / 4)])], geometry=[dot])
    refined_count = estimate_mesh(refined, {}, registry).n_elements
    add('triangle_count_refined', refined_count > 1.2 * estimate.n_elements, [refined_count, estimate.n_elements])

    report = api.check(split)
    add('check_json_meshes', [record['name'] for record in report['meshes']] == ['fine', 'coarse']
        and [record['cells'] for record in report['meshes']] == [cells['fine'], cells['coarse']]
        and report['transfers'] == [], report['meshes'])

    adapted = _meshes_spec([PhysicsSpec('heat', 'heat')], [MeshSpec(mode='uniform', dx=_MESH_FINE_DX)])
    adapted.study = StudySpec(type='stationary', adaptation=AdaptationSpec('T_unknown', method='uniform'))
    adapted.derived_values = [DerivedValueSpec('T_max', 'T', [], 'max')]
    data = adapted.to_dict()
    add('adaptation_round_trip', ModelSpec.from_dict(data).to_dict() == data)
    add('adaptation_target_checked', bool(_messages(validate(adapted, registry), 'error', ('study',),
                                                    'adaptation target')))
    adapted.study.adaptation.target = 'T_max'
    adapted.analytic_curves = [AnalyticCurveSpec('T_curve', 'sweep', '0', series='T_max')]
    adapted.rename_derived_value('T_max', 'T_peak')
    add('adaptation_target_renamed', adapted.study.adaptation.target == 'T_peak')
    add('sweep_curve_series_renamed', adapted.analytic_curves[0].series == 'T_peak')

    return checks.result()


_ESTIMATE_RTOL = 0.15   # triangle element estimate / builtin-mesher elements


def _check_triangle_estimate(results_by_name):
    """Estimate / built element ratios of the builtin mesher (uniform, refined circle, thin film, rib window and PML ring);
    measured = the largest |ratio - 1|."""
    from .expressions import evaluate_parameters
    from .meshing import estimate_mesh
    from .schema import get_registry
    from .selftests_adaptation import _rib_modes_spec
    from .triangle_meshing import triangulate

    registry = get_registry()
    plain = _meshes_spec([], [MeshSpec(kind='triangles', dx=_TRIANGLE_H)])
    dot = GeometrySpec('dot', 'circle', {'cx': 5e-6, 'cy': 2e-6, 'r': 0.5e-6})
    refined = _meshes_spec([], [MeshSpec(kind='triangles', mode='nonuniform', dx=_TRIANGLE_H,
                                         refinements=[RefinementSpec('dot', _TRIANGLE_H / 4)])], geometry=[dot])
    # a film thinner than a raster pixel (39 nm) between two pixel-centre rows, refined to its own thickness
    film = GeometrySpec('film', 'rectangle', {'x0': 0.0, 'y0': 2.005e-6, 'x1': _SIZE[0], 'y1': 2.015e-6})
    thin_film = _meshes_spec([], [MeshSpec(kind='triangles', mode='nonuniform', dx=_TRIANGLE_H,
                                           refinements=[RefinementSpec('film', 10e-9)])], geometry=[film])
    ringed = _rib_modes_spec(None)
    ringed.meshes = [MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=100e-9)]
    walled = ringed.copy()
    walled.physics[0].settings.update(bc_x_hi='pec', bc_y_lo='pec', bc_y_hi='pec')
    ratios, counts = {}, {}
    for name, spec in (('uniform', plain), ('refined_circle', refined), ('thin_film', thin_film)):
        built = triangulate(spec, {}, registry).n_elements
        estimate = estimate_mesh(spec, {}, registry).n_elements
        ratios[name], counts[name] = estimate / built, {'estimate': estimate, 'built': built}
    rib_parameters = evaluate_parameters(ringed.parameters)   # the rib geometry is written in its parameters
    ring_mesh = triangulate(ringed, rib_parameters, registry)
    built_window = int(np.sum(ring_mesh.pml_faces == 0))
    built_ring = ring_mesh.n_elements - built_window
    total = estimate_mesh(ringed, rib_parameters, registry).n_elements
    window = estimate_mesh(walled, rib_parameters, registry).n_elements   # same window raster, no ring
    ratios['rib_window'], counts['rib_window'] = window / built_window, {'estimate': window, 'built': built_window}
    ratios['rib_ring'] = (total - window) / max(built_ring, 1)
    counts['rib_ring'] = {'estimate': total - window, 'built': built_ring}
    error = max(abs(ratio - 1.0) for ratio in ratios.values())
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _ESTIMATE_RTOL,
            'details': {'ratios': ratios, 'counts': counts}}


MESH_SELFTEST_CASES = (
    SelfTestCase(
        name='mesh_tensor_options',
        label='Tensor mesh options: anisotropic, bands, custom axes, outer grading',
        description='CPU checks of the graded tensor-mesh options (no model run, no GPU) on mesh estimates of '
                    'physics-free models. A 10 x 4 um domain (base 100 nm): a waveguide band refined to 20 nm along y '
                    'only (dx_x = base: 100 equal x cells, 20 cells of 20 nm across the 0.4 um core); an x band '
                    '4-5 um at 25 nm with the x max_ratio 1.3 (cells <= 1.02 x 25 nm inside, ratio <= 1.05 x 1.3, y '
                    'untouched); custom y segments geometric 20 cells ratio 0.2 to 1 um, tanh 40 cells ratio 5 to 3 '
                    'um, geometric 20 cells ratio 5 to the end (exact ends and ratios) and custom x edges every 1 um '
                    '(0 and the end added: 10 cells); segment_widths tanh 31 cells ratio 8 (sum, ratio, symmetry). '
                    'A 1 x 1 mm domain (base 1 um) with a 2 um dot at 0.2 um and outer grading to 100 um at ratio '
                    '1.3: fewer than 200 cells per axis (1000 at the base size), largest cell 50-125 um, ratio <= '
                    '1.05 x 1.3. A 3D '
                    'z axis of 12 tanh cells (ratio 3), the spec round trip of the new fields and a z axis entry '
                    'rejected in 2D. error = number of failed sub-checks, tolerance 0. The anisotropic checks at '
                    '20 nm not yet confirmed by a run (a rebound constant made them test a uniform 100 nm grid).',
        specs=lambda: {},
        check=_check_mesh_options,
        max_wall_time=60.0),
    SelfTestCase(
        name='mesh_per_physics_spec',
        label='Per-physics meshes, triangle sizes and adaptation: spec and validation',
        description='CPU checks of the S7.2 / S7.5 / S7.7 spec (no build, no GPU) on a 10 x 4 um domain: a legacy '
                    '"mesh" object loads as meshes[0] (and with "meshes" too is refused); heat on a uniform 100 nm '
                    'mesh and electric_currents on a 200 nm one give per-mesh estimates of 4000 and 1000 cells, a '
                    'memory estimate equal to the sum of its per-mesh parts, the heat part equal to heat alone on '
                    'that mesh, and no error at ("meshes",) for several meshes; validation errors for two '
                    'em_fdfd_ez physics on different meshes, an electrostatics_em_pockels coupling across meshes, an '
                    'unknown mesh name, axes on a triangle mesh and laminar_flow (tensor only) on a triangle mesh '
                    '(solver_grid triangles); '
                    'an unused mesh warning; the triangle element estimate of a physics-free model (TRIANGLE_FILL 2.3 '
                    'x 4 / sqrt(3) x area / h^2 within 1e-3), h_min = h / 100, growth 1.3, and more elements with a '
                    'refined circle; api.check meshes records; an AdaptationSpec round trip, unknown target error and '
                    'target rename (also of a \'sweep\' analytic-curve series). error = number of failed sub-checks, '
                    'tolerance 0. The series rename check not yet confirmed by a run.',
        specs=lambda: {},
        check=_check_per_physics_spec,
        max_wall_time=60.0),
    SelfTestCase(
        name='mesh_triangle_estimate',
        label='Triangle meshes: element-count estimate vs the builtin mesher',
        description='CPU (no build, no GPU): the raster element estimate (meshing.TRIANGLE_FILL / TRIANGLE_EDGE_FILL) '
                    'against the elements fem.mesher builds for a physics-free 10 x 4 um window at h 250 nm, the same '
                    'with a 0.5 um circle refined to h / 4 (graded), the same with a 10 nm film across the window '
                    'refined to 10 nm (no raster pixel centre inside it: the pixel-mean band density), and the TFLN '
                    'rib of adaptation_rib_modes at h 100 nm with its cpml faces: window elements (estimate of the '
                    'pec-walled twin) and PML ring elements apart. measured = the largest |estimate / built - 1|; '
                    'passes at <= 0.15. The thin film not yet confirmed by a run.',
        specs=lambda: {},
        check=_check_triangle_estimate,
        max_wall_time=60.0),
)
