"""Self-test of the B1 3D model core (spec v2, 3D geometry, mesh estimate, dimension validation, per-slab material
painting on a Domain3D); no model run (specs() is empty, check({}) builds a bare Domain3D through builder internals)."""
import math

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import SubChecks
from .spec import DomainSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec

_CELL = 20e-9                        # m, uniform
_SIZE = (2e-6, 2e-6, 1e-6)           # m; 100 x 100 x 50 cells
_CELLS = (100, 100, 50)
_SLAB_TOP = 0.3e-6                   # m; slab z 0..0.3 um = 15 cells
_BALL = (1e-6, 1e-6, 0.6e-6, 0.3e-6)   # cx, cy, cz, r [m]
_PILLAR = (0.4e-6, 0.4e-6, 0.2e-6, 0.3e-6, 0.9e-6)   # cx, cy, r, z0, z1 [m]
_VOLUME_RTOL = 0.03                  # staircase of the 15-cell radius sphere, 10-cell radius cylinder (~0.6 %)
_RHO_ATOL = 1e-3                     # float32 painting of rho = 1 + x / 1 um


def core_3d_spec():
    """3D spec without physics: slab (rectangle z 0..0.3 um) under a sphere, a z cylinder; air / substrate / metal."""
    cx, cy, cz, radius = _BALL
    px, py, pillar_radius, pz0, pz1 = _PILLAR
    spec = ModelSpec(title='Self-test: B1 3D core', domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1],
                                                                      size_z=_SIZE[2]))
    spec.mesh = MeshSpec(mode='uniform', dx=_CELL)
    spec.geometry = [
        GeometrySpec('slab', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _SIZE[0], 'y1': _SIZE[1], 'z0': 0.0,
                                           'z1': _SLAB_TOP}),
        GeometrySpec('ball', 'sphere', {'cx': cx, 'cy': cy, 'cz': cz, 'r': radius}),
        GeometrySpec('pillar', 'cylinder_z', {'cx': px, 'cy': py, 'r': pillar_radius, 'z0': pz0, 'z1': pz1}),
    ]
    spec.materials = [MaterialSpec('air', properties={'k': 1.0, 'rho': 1.0}, selection=['all']),
                      MaterialSpec('substrate', properties={'k': 2.0, 'rho': '1 + x / 1e-6'}, selection=['slab']),
                      MaterialSpec('metal', properties={'k': 3.0}, selection=['ball', 'pillar'])]
    return spec


def _dimension_errors(spec, registry, physics_name):
    """Validation errors at ('physics', physics_name) about the model's dimensions."""
    from .validate import validate
    return [issue.message for issue in validate(spec, registry)
            if issue.level == 'error' and issue.path == ('physics', physics_name) and 'runs only in' in issue.message]


def _cpu_checks(registry, spec, shapes, add):
    """Spec v2 round trip / v1 load, sphere / slab / cylinder rasterisation, z_extent, runs, layer preview, mesh nz,
    dimension validation (heat 3D-capable, magnetostatics 2D only), z params rejected in 2D."""
    from .geometry import cross_section_runs, material_preview, rasterize_box, z_extent
    from .meshing import estimate_mesh
    from .validate import validate

    data = spec.to_dict()
    add('spec_v2_round_trip', ModelSpec.from_dict(data).to_dict() == data)
    v1_data = dict(data, version=1)
    v1_data['domain'] = {'size_x': _SIZE[0], 'size_y': _SIZE[1]}
    v1_spec = ModelSpec.from_dict(v1_data)
    add('spec_v1_loads_as_v2', v1_spec.version == 2 and not v1_spec.is_3d, v1_spec.version)

    centres = [((np.arange(n) + 0.5) * _CELL).astype(np.float32) for n in _CELLS]
    _, ball_mask = rasterize_box('ball', shapes, *centres, np)
    sphere_ratio = float(ball_mask.sum()) * _CELL ** 3 / (4.0 / 3.0 * math.pi * _BALL[3] ** 3)
    add('sphere_volume', abs(sphere_ratio - 1.0) < _VOLUME_RTOL, sphere_ratio)
    slab_window, slab_mask = rasterize_box('slab', shapes, *centres, np)
    slab_cells = int(slab_mask.sum())
    add('slab_cells', slab_cells == _CELLS[0] * _CELLS[1] * round(_SLAB_TOP / _CELL),
        {'cells': slab_cells, 'window': str(slab_window)})
    px, py, pillar_radius, pz0, pz1 = _PILLAR
    _, pillar_mask = rasterize_box('pillar', shapes, *centres, np)
    pillar_ratio = float(pillar_mask.sum()) * _CELL ** 3 / (math.pi * pillar_radius ** 2 * (pz1 - pz0))
    add('cylinder_volume', abs(pillar_ratio - 1.0) < _VOLUME_RTOL, pillar_ratio)
    pillar_extent = z_extent('pillar', shapes)
    add('z_extent', pillar_extent == (pz0, pz1), list(pillar_extent))
    runs = cross_section_runs(['slab', 'ball', 'pillar'], shapes, centres[2])
    add('cross_section_runs_cover_z', runs[0][0] == 0 and runs[-1][1] == _CELLS[2] and len(runs) > 3,
        {'n_runs': len(runs)})
    preview = material_preview(spec, {}, max_cells=100, shapes=shapes, z=_BALL[2])
    add('layer_preview_has_metal', bool((preview[0] == 3).any()))   # 1 + index of 'metal'

    mesh = estimate_mesh(spec, {}, registry)
    add('mesh_estimate_nz', mesh.nz == _CELLS[2] and mesh.n_cells == math.prod(_CELLS),
        {'nx': mesh.nx, 'ny': mesh.ny, 'nz': mesh.nz, 'n_cells': mesh.n_cells})

    with_heat = spec.copy()
    with_heat.physics = [PhysicsSpec('heat', 'heat')]
    heat_errors = _dimension_errors(with_heat, registry, 'heat')
    add('heat_accepted_in_3d', not heat_errors, heat_errors)
    with_magnetic = spec.copy()
    with_magnetic.physics = [PhysicsSpec('magnetic', 'magnetostatics')]   # 2D only (3D: magnetostatics_3d)
    magnetic_errors = _dimension_errors(with_magnetic, registry, 'magnetic')
    add('2d_physics_rejected_in_3d', bool(magnetic_errors), magnetic_errors)

    flat = spec.copy()
    flat.domain.size_z = None
    flat.geometry = [GeometrySpec('slab', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': 1e-6, 'y1': 1e-6, 'z0': 0.0})]
    flat.materials = []
    z_errors = [issue.message for issue in validate(flat, registry) if 'only used in 3D' in issue.message]
    add('z_param_rejected_in_2d', bool(z_errors), z_errors)


_GRADED_BASE, _GRADED_FINE, _GRADED_RATIO = 50e-9, 10e-9, 1.2   # m, m, 1: graded z around the pillar


def _graded_z_checks(registry, spec, add):
    """Graded z (igor 1.4) on the mesh estimate: a pillar refinement (margin 0) snaps z 0.3 / 0.9 um to cell edges,
    keeps its target inside, grades at <= max_ratio (+5 % taper slack) and coarsens outside."""
    from .meshing import estimate_mesh
    from .spec import RefinementSpec

    graded = spec.copy()
    graded.mesh = MeshSpec(mode='nonuniform', dx=_GRADED_BASE, max_ratio=_GRADED_RATIO,
                           refinements=[RefinementSpec('pillar', _GRADED_FINE, 0.0)])
    mesh = estimate_mesh(graded, {}, registry)
    edges, widths = mesh.z_edges, mesh.dz_cells
    _, _, _, pz0, pz1 = _PILLAR
    snap_misses = [float(np.min(np.abs(edges - z))) for z in (pz0, pz1)]
    add('graded_z_snaps_pillar_faces', max(snap_misses) < 1e-3 * _GRADED_FINE, snap_misses)
    inside = widths[(edges[:-1] >= pz0 - 1e-12) & (edges[1:] <= pz1 + 1e-12)]
    add('graded_z_fine_inside', inside.size > 0 and float(np.max(np.abs(inside / _GRADED_FINE - 1.0))) < 1e-3,
        {'cells': int(inside.size), 'min': float(inside.min(initial=np.inf)), 'max': float(inside.max(initial=0.0))})
    ratios = widths[1:] / widths[:-1]
    worst_ratio = float(np.max(np.maximum(ratios, 1.0 / ratios)))
    add('graded_z_ratio', worst_ratio <= 1.05 * _GRADED_RATIO, worst_ratio)
    add('graded_z_coarsens', float(widths.max()) > 2.0 * _GRADED_FINE and not mesh.uniform
        and mesh.nz < round(_SIZE[2] / _GRADED_FINE), {'nz': mesh.nz, 'dz_max': float(widths.max())})


def _box_values_checks(add):
    """igor 1.10 box-wise adds (numpy): BoxValues.add_masked_to / copy_masked_to on a box half outside the window
    equal the window_array forms; a box outside the window leaves the target unchanged."""
    from ..core.box_values import BoxValues

    rng = np.random.default_rng(7)
    window = (slice(2, 10), slice(1, 7), slice(0, 5))
    window_shape = (8, 6, 5)
    box = (slice(6, 13), slice(0, 4), slice(2, 5))
    values = BoxValues(box, rng.random((7, 4, 3)).astype(np.float32))
    mask_window = rng.random(window_shape) > 0.4
    start = rng.random(window_shape).astype(np.float32)
    added = start.copy()
    values.add_masked_to(added, mask_window, window)
    expected_added = start + np.where(mask_window, values.window_array(window), np.float32(0.0))
    copied = start.copy()
    values.copy_masked_to(copied, mask_window, window)
    inside_box = BoxValues(box, np.ones((7, 4, 3), dtype=np.float32)).window_array(window) > 0.0
    expected_copied = np.where(mask_window & inside_box, values.window_array(window), start)
    outside = BoxValues((slice(20, 22), slice(0, 2), slice(0, 2)), np.ones((2, 2, 2), dtype=np.float32))
    untouched = start.copy()
    outside.add_masked_to(untouched, mask_window, window)
    errors = {'add': float(np.max(np.abs(added - expected_added))),
              'copy': float(np.max(np.abs(copied - expected_copied))),
              'outside': float(np.max(np.abs(untouched - start)))}
    add('box_values_boxwise_adds', max(errors.values()) < 1e-6, errors)


def _gpu_checks(registry, spec, shapes, add):
    """Domain3D + heat material arrays painted per z-slab run: k per object, x-dependent rho, region window mask."""
    from .builder import BuildContext, _apply_static_materials
    from .meshing import make_domain, optional_material_arrays

    heat_spec = spec.copy()
    heat_spec.physics = [PhysicsSpec('heat', 'heat')]
    heat_type = registry.physics_type('heat')
    domain = make_domain(heat_spec, {}, registry, resolved_geometry=shapes)
    add('domain_is_3d', domain.dimensions == 3, domain.dimensions)
    domain.materials.add_properties(heat_type.material_schema)
    domain.materials.add_properties(optional_material_arrays(heat_spec, (heat_type,)))
    log_lines = []
    ctx = BuildContext(heat_spec, registry, {}, shapes, domain, lambda level, message: log_lines.append(
        f'{level}: {message}'))
    try:
        deferred = _apply_static_materials(ctx, [heat_type])
        k_xx = domain.materials.k_xx.get()
        rho = domain.materials.rho.get()
        # cells: ball centre, slab, air corner, pillar
        painted_k = [float(k_xx[50, 50, 30]), float(k_xx[50, 50, 5]), float(k_xx[90, 90, 45]),
                     float(k_xx[20, 20, 30])]
        add('painted_k_ball_slab_air_pillar', painted_k == [3.0, 2.0, 1.0, 3.0], painted_k)
        rho_slab, rho_air = float(rho[75, 10, 5]), float(rho[75, 10, 40])
        expected_rho_slab = 1.0 + (75 + 0.5) * _CELL / 1e-6
        add('x_dependent_rho_in_slab', abs(rho_slab - expected_rho_slab) < _RHO_ATOL and rho_air == 1.0,
            {'slab': rho_slab, 'expected_slab': expected_rho_slab, 'air': rho_air})
        add('no_deferred_updates', deferred == [], len(deferred))
        window, region_mask = ctx.region_window_mask(ctx.material_regions[2])
        region_cells, metal_cells = int(region_mask.sum()), int((k_xx == 3.0).sum())
        add('region_window_mask_of_metal', region_cells == metal_cells,
            {'mask_cells': region_cells, 'k3_cells': metal_cells, 'window': str(window),
             'build_log': log_lines[:5]})
    finally:
        ctx.release_caches()


def _check_3d_core(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from .geometry import resolve_geometry
    from .schema import get_registry

    registry = get_registry()
    spec = core_3d_spec()
    shapes = resolve_geometry(spec, {})
    checks = SubChecks()
    add = checks.add

    _cpu_checks(registry, spec, shapes, add)
    _graded_z_checks(registry, spec, add)
    _box_values_checks(add)
    _gpu_checks(registry, spec, shapes, add)
    return checks.result()


CORE_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='model_3d_core',
        label='3D model core: spec, geometry, mesh, validation, material painting',
        description='Checks of the B1 3D core (no model run; GPU for the material painting) on a 2 x 2 x 1 um '
                    'domain of 20 nm cells (100 x 100 x 50) with a slab (z 0..0.3 um), a sphere (r 0.3 um) and a z '
                    'cylinder (r 0.2 um, z 0.3..0.9 um): spec v2 round trip and v1 load as 2D v2, sphere and cylinder '
                    'staircase volumes within 3 %, exact slab cell count, z_extent, cross_section_runs covering z, the '
                    'layer material preview, the mesh estimate nz / n_cells, heat accepted and magnetostatics '
                    '(2D only) rejected in 3D, a z0 rejected in 2D, graded z of a pillar refinement (z faces on cell '
                    'edges, 10 nm inside, ratio <= max_ratio, coarser outside), box-wise BoxValues adds / copies '
                    'equal to their window-array forms; then a Domain3D with heat material '
                    'arrays painted '
                    'per z-slab run: k of each object, rho = 1 + x / 1 um in the slab (float32, 1e-3), no deferred '
                    'updates and the metal region_window_mask equal to the k = 3 cells. error = number of failed '
                    'sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_3d_core,
        max_wall_time=60.0, backends=('gpu',)),
)
