"""Self-tests of the S7.7 mesh refinement levels (mesh plan A1 / A2 / A3): CPU checks of the stop rule, Richardson fit,
report, indicators (interpolation, Kelly face jumps, energy-weighted EM, a cpu-backend heat solve; triangles: residual
/ flux jumps, edge-element jumps) and refined MeshSpecs (tensor bands, triangle size points), and the strip-heater /
disc-heater / rib-modes runs of uniform sequences and adaptive tensor bands, the L-shaped conductor and rib on
triangles."""
import math

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import SubChecks, with_exact_cg
from .spec import (AdaptationSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec,
                   ModelSpec, ParameterSpec, PhysicsSpec, SolverDefaultsSpec, StudySpec)

_STRIP_DX = 150e-9            # m: every strip-heater edge lies on this lattice (S7.8 mesh study model)
_STRIP_ORDER = 0.68           # S7.8 record: dT_heater error ~ N^-0.68 on uniform meshes (order in cells)
_STRIP_ORDER_TOLERANCE = 0.1
_ADAPTIVE_ERROR = 1e-3        # relative dT_heater error the adaptive and uniform cell counts are compared at
_ADAPTIVE_GAIN = 0.5          # adaptive cells / uniform cells at that error must not exceed this
# adaptive max_cells (checked on the planned estimate): the disc's sits just above _ADAPTIVE_GAIN x the ~4.6e5 uniform
# cells the error needs, so a finer level can never pass. The strip is capped below that bound, one level past the
# recorded crossing (1e-3 first reached at level 3, 2.9e5 cells, level 4 at 5.7e5 is refused): the measured value is
# unchanged, but the effective gate tightens to a crossing below the cap (gain <= ~0.27 vs measured 0.154). The rib
# cap stops after level 3 (~4.5e4 window cells, ~1.1e5 planned next), past 0.5 x its 5.95e4 reference cells
_STRIP_ADAPTIVE_MAX_CELLS, _DISC_ADAPTIVE_MAX_CELLS, _RIB_ADAPTIVE_MAX_CELLS = 400_000, 250_000, 75_000
_STRIP_UNIFORM_FIT_LEVELS = 3   # the gain reference: the Richardson fit of the uniform levels 0..2 (150 to 37.5 nm)
_SMALL_SIZE, _SMALL_DX = (4e-6, 2e-6), 100e-9   # m: the CPU refinement checks' domain and uniform cell
# A2 disc heater (S7.8 disc_heater.json): disc radius, heater layer = uniform base cell, held hemisphere radius [m]
_DISC_RADIUS, _DISC_LAYER, _DISC_OUTER = 1e-3, 5e-5, 1e-2
_DISC_BOX = _DISC_OUTER + 10 * _DISC_LAYER
# A2 rib modes (S7.8 rib_modes.json): uniform base cell [m]; the gain is read at the error of the finest uniform
# level (index 2, 12.5 nm, 5.95e4 cells, ~4e-5) under the three-level fit, not at a fixed 1e-5: that exact fit
# (1.907934) and the adaptive levels' own trend (1.907969 +- 6e-6) differ by ~1.8e-5 relative, so adaptive errors
# against it floor near 2e-5 (measured 4.0e-5 / 2.1e-5 at 1.9e4 / 4.5e4 cells; uniform levels thin the PML with dx,
# adaptive bands do not: a suspected source). A matched-level comparison barely moves with that reference offset
_RIB_DX, _RIB_REFERENCE_LEVEL = 50e-9, 2
_INDICATOR_CG_TOLERANCE = 1e-13   # float64 CG of the cpu heat indicator checks
# A3 triangles: indicator checks on a 1 x 1 um window (right half a second layer), adaptive spec checks
_FEM_SIDE, _FEM_H = 1e-6, 0.1e-6
_FEM_EXACT = 1e-18           # eta^2 total / its energy scale of exactly representable solutions
_FEM_P1_RATIO = (2.5, 6.5)   # P1 eta^2 total at h over h / 2 (energy error ~ h^2: 4)
_FEM_EM_RATIO = 6.0          # quadratic edge elements: eta^2 total at h over h / 2 at least this (h^4: 16)
# element sizes of that pair [m] (2 / 4 elements across the 90 nm slab height); 30 / 15 nm recorded ratio 25.1
_FEM_EM_SIZES = (45e-9, 22.5e-9)
_FEM_GROWTH_RANGE = (1.3, 2.4)   # adapted / old window elements at growth 1.7 (mesher spread)
# A3 L-shaped conductor (re-entrant corner, V ~ r^(2/3)) and the triangle rib
_LSHAPE_SIDE, _LSHAPE_H = 2e-6, 0.25e-6
_LSHAPE_ERROR, _LSHAPE_GAIN = 1e-4, 0.25
# adaptive max_cells just above _LSHAPE_GAIN x the ~5.5e3 uniform elements for 1e-4 (finer levels cannot pass; the
# 2e5 cap meshed and solved levels to 2.9e5 elements in ~450 s); it is checked on the planned element estimate
# (~9.3e2 for the passing 936-element level, ~3.5e3 for the next), so that next level is refused
_LSHAPE_ADAPTIVE_MAX_CELLS = 1_500
# A3 triangle rib on linear elements (discretization 'linear', n_eff error ~h^2): quadratic elements put the 200 nm
# base within 2e-6 of the converged n_eff, below the mesher's noise floor (a uniform P2 sequence alternates in sign:
# no Richardson fit), so no gain is measurable there. Base element [m]; the gain is read at the error of uniform
# level 2 (50 nm, ~1.2e4 elements) under the fit of levels 1..3 (the 200 nm base stays out of the fit); adaptive
# max_cells on the planned element estimate just above 0.5 x those elements; adaptive growth 1.5 (finer level steps
# than 2 for the interpolation between the bracketing levels). A smaller window with pec walls (a PML ring of
# pml_cells x h would thin with each uniform level).
_FEM_RIB_DX, _FEM_RIB_REFERENCE_LEVEL, _FEM_RIB_ADAPTIVE_MAX_CELLS = 200e-9, 2, 10_000
_FEM_RIB_ADAPTIVE_GROWTH = 1.5
_FEM_RIB_WINDOW = {'t_box': 1.0e-6, 'y_slab': 1.3e-6, 'y_film': 1.6e-6, 'Lx': 2.4e-6, 'Ly': 2.4e-6}   # m


def _strip_heater_spec(adaptation):
    """S7.8 planar strip heater (mesh_study/models/strip_heater.json): 2.1 x 0.15 um gold heater above a 0.6 x 0.3 um
    silicon core in 4.5 um of silica on a substrate at T_sub, air above, 60 x 15 um; uniform 150 nm cells, exact
    multigrid CG."""
    geometry = [GeometrySpec('oxide', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': 60e-6, 'y1': 4.5e-6}),
                GeometrySpec('heater', 'rectangle', {'x0': 28.95e-6, 'y0': 3.3e-6, 'x1': 31.05e-6, 'y1': 3.45e-6}),
                GeometrySpec('core', 'rectangle', {'x0': 29.7e-6, 'y0': 1.95e-6, 'x1': 30.3e-6, 'y1': 2.25e-6})]
    materials = [MaterialSpec('air', properties={'k': 0.026}, selection=['all']),
                 MaterialSpec('silica', properties={'k': 1.38}, selection=['oxide']),
                 MaterialSpec('silicon_core', properties={'k': 148.0}, selection=['core']),
                 MaterialSpec('gold_heater', properties={'k': 317.0}, selection=['heater'])]
    heat = PhysicsSpec('heat', 'heat', settings={'T0': 'T_sub'}, features=[
        FeatureSpec('substrate', 'temperature', ['y_lo'], {'temperature': 'T_sub'}),
        FeatureSpec('heater_power', 'heat_source', ['heater'], {'Q': 'P_heater / (2.1e-6 * 0.15e-6)'})])
    spec = ModelSpec(title='Self-test: strip heater mesh refinement levels',
                     parameters=[ParameterSpec('P_heater', 20.0), ParameterSpec('T_sub', 293.15)],
                     domain=DomainSpec(size_x=60e-6, size_y=15e-6), geometry=geometry, materials=materials,
                     physics=[heat], mesh=MeshSpec(mode='uniform', dx=_STRIP_DX),
                     study=StudySpec(type='stationary', adaptation=adaptation),
                     derived_values=[DerivedValueSpec('dT_core', 'T - T_sub', ['core'], 'mean'),
                                     DerivedValueSpec('dT_heater', 'T - T_sub', ['heater'], 'mean')])
    return with_exact_cg(spec, preconditioner='multigrid')


def _uniform_sequence(max_levels, target='dT_heater'):
    """AdaptationSpec of a uniform sequence halving every cell size per level, never stopping on the tolerance."""
    return AdaptationSpec(target, method='uniform', tolerance=1e-12, max_levels=max_levels, growth=4.0)


def _disc_heater_spec(adaptation):
    """S7.8 disc heater (mesh_study/models/disc_heater.json): axisymmetric heat, flux q into a 1 mm disc on a k = 0.2
    half-space held at T_s beyond the 10 mm hemisphere (conformal held edge); uniform 50 um cells, exact multigrid
    CG."""
    geometry = [GeometrySpec('heater', 'rectangle', {'x0': 0.0, 'y0': _DISC_BOX - _DISC_LAYER, 'x1': _DISC_RADIUS,
                                                     'y1': _DISC_BOX}),
                GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _DISC_BOX, 'y1': _DISC_BOX}),
                GeometrySpec('hemisphere', 'circle', {'cx': 0.0, 'cy': _DISC_BOX, 'r': _DISC_OUTER}),
                GeometrySpec('far_solid', 'difference', {'base': 'box', 'subtract': ['hemisphere']})]
    heat = PhysicsSpec('heat', 'heat', settings={'geometry': 'axisymmetric', 'T0': 'T_s'}, features=[
        FeatureSpec('far_field', 'fixed_temperature_domain', ['far_solid'], {'temperature': 'T_s'}),
        FeatureSpec('disc_flux', 'heat_source', ['heater'], {'Q': 'q_flux / 5e-5'})])
    spec = ModelSpec(title='Self-test: disc heater mesh refinement levels',
                     parameters=[ParameterSpec('q_flux', 1e4), ParameterSpec('T_s', 293.15)],
                     domain=DomainSpec(size_x=_DISC_BOX, size_y=_DISC_BOX), geometry=geometry,
                     materials=[MaterialSpec('solid', properties={'k': 0.2}, selection=['all'])],
                     physics=[heat], mesh=MeshSpec(mode='uniform', dx=_DISC_LAYER),
                     solver_defaults=SolverDefaultsSpec(conformal_boundaries=True),
                     study=StudySpec(type='stationary', adaptation=adaptation),
                     derived_values=[DerivedValueSpec('heater_mean_rise', 'T - T_s', ['heater'], 'mean')])
    return with_exact_cg(spec, preconditioner='multigrid')


def _rib_modes_spec(adaptation):
    """S7.8 TFLN rib (mesh_study/models/rib_modes.json, rib centre on the x_lo mirror, no snap objects): 600 nm x-cut
    film, 300 nm etch, 1.2 um top, 60 degree sidewalls, quasi-TE0 n_eff at 1550 nm, subpixel smoothing; 50 nm cells."""
    rib = [[0.0, 't_box'], ['x_foot', 't_box'], ['x_foot', 'y_slab'], ['x_top', 'y_film'], [0.0, 'y_film']]
    geometry = [GeometrySpec('top_region', 'rectangle', {'x0': 0.0, 'y0': 't_box', 'x1': 'Lx', 'y1': 'Ly'}),
                GeometrySpec('slab', 'rectangle', {'x0': 0.0, 'y0': 't_box', 'x1': 'Lx', 'y1': 'y_slab'}),
                GeometrySpec('rib', 'polygon', {'points': rib})]
    materials = [MaterialSpec('buried_oxide', library='silica', selection=['all']),
                 MaterialSpec('top_cladding', properties={'eps_r': 1.0}, selection=['top_region']),
                 MaterialSpec('lithium_niobate_x_cut', properties={'eps_xx': '2.138**2', 'eps_yy': '2.211**2',
                                                                   'eps_zz': '2.211**2'}, selection=['slab', 'rib'])]
    modes = PhysicsSpec('modes', 'em_modes', settings={
        'bc_x_lo': 'antisymmetric', 'bc_x_hi': 'cpml', 'bc_y_lo': 'cpml', 'bc_y_hi': 'cpml', 'pml_sigma_factor': 0.7,
        'wavelength': 1.55e-6, 'n_modes': 2, 'group_index': False},
        features=[FeatureSpec('table', 'mode_table')])
    parameters = [ParameterSpec('t_box', 1.5e-6), ParameterSpec('Lx', 3e-6), ParameterSpec('Ly', 3.1e-6),
                  ParameterSpec('y_slab', 1.8e-6), ParameterSpec('y_film', 2.1e-6), ParameterSpec('x_top', 0.6e-6),
                  ParameterSpec('x_foot', 'x_top + 0.3e-6 / tan(pi / 3)')]
    return ModelSpec(title='Self-test: rib waveguide mode refinement levels', parameters=parameters,
                     domain=DomainSpec(size_x='Lx', size_y='Ly'), geometry=geometry, materials=materials,
                     physics=[modes], mesh=MeshSpec(mode='uniform', dx=_RIB_DX),
                     solver_defaults=SolverDefaultsSpec(subpixel_smoothing=True),
                     study=StudySpec(type='mode_analysis', adaptation=adaptation),
                     derived_values=[DerivedValueSpec('n_eff_1', 'n_eff', [], 'at', source='modes.table', at=0)])


def _small_spec(mode):
    """Heat on a 4 x 2 um domain with a 0.2 x 0.1 um block at (1, 0.5) um (CPU mesh-estimate checks; auto mode
    refines the block: 25 nm < the 80 nm base)."""
    block = GeometrySpec('block', 'rectangle', {'x0': 1e-6, 'y0': 0.5e-6, 'x1': 1.2e-6, 'y1': 0.6e-6})
    return ModelSpec(title='Self-test: refined mesh specs', domain=DomainSpec(size_x=_SMALL_SIZE[0],
                                                                                size_y=_SMALL_SIZE[1]),
                     geometry=[block], materials=[MaterialSpec('fill', properties={'k': 1.0}, selection=['all']),
                                                  MaterialSpec('core', properties={'k': 10.0}, selection=['block'])],
                     physics=[PhysicsSpec('heat', 'heat')], mesh=MeshSpec(mode=mode, dx=_SMALL_DX))


def _check_stop_rules(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from ..core.error_indicators import interpolation_indicator
    from .geometry import resolve_geometry
    from .mesh_refinement import adaptive_mesh_spec, axis_bands, frozen_mesh_spec, uniform_mesh_spec
    from .meshing import estimate_mesh
    from .schema import get_registry
    from .study_adaptation import (STOP_REASONS, _Setup, cells_for_error, convergence_report, level_stop,
                                   next_level_stop, richardson)

    checks = SubChecks()
    add = checks.add

    cells = [1000.0, 1700.0, 2890.0]
    fit, _ = richardson(cells, [2.0 + 3.0 * count ** -0.7 for count in cells], 2)
    add('richardson_synthetic', fit is not None and abs(fit['order_cells'] - 0.7) < 1e-8
        and abs(fit['extrapolated'] - 2.0) < 1e-9 and abs(fit['coefficient'] - 3.0) < 1e-6 and abs(fit['order'] - 1.4)
        < 1e-7, None if fit is None else {key: fit[key] for key in ('order_cells', 'extrapolated', 'coefficient')})
    needed = cells_for_error(fit, 1e-3) if fit is not None else math.nan
    add('cells_for_error', abs(needed / (3.0 / 2e-3) ** (1.0 / 0.7) - 1.0) < 1e-6, needed)
    add('richardson_oscillating', richardson(cells, [1.0, 1.1, 1.05])[0] is None)
    add('richardson_not_shrinking', richardson(cells, [1.0, 2.0, 3.5])[0] is None)
    add('richardson_exact', richardson(cells, [1.0, 1.5, 1.5])[0]['extrapolated'] == 1.5)
    add('stop_converged', level_stop([1.0, 1.0005], 1e-3, 1, 2)[0] == 'converged')
    add('stop_max_levels', level_stop([1.0, 1.01], 1e-3, 2, 2)[0] == 'max_levels')
    add('stop_continue', level_stop([1.0, 1.01], 1e-3, 1, 2)[0] is None and level_stop([1.0], 1e-3, 0, 2)[0] is None)
    add('stop_not_finite', level_stop([1.0, math.nan], 1e-3, 1, 2)[0] == 'not_finite')
    add('next_no_refinement', next_level_stop(100, 100, None)[0] == 'no_refinement')
    add('next_max_cells', next_level_stop(100, 200, 150)[0] == 'max_cells')
    add('next_memory', next_level_stop(100, 170, None, 'too big')[0] == 'memory')
    add('next_continue', next_level_stop(100, 170, 1000)[0] is None)

    adaptation = AdaptationSpec('v', method='uniform')
    levels = [{'level': index, 'refined_cells': int(count), 'value': 2.0 + 3.0 * count ** -0.7, 'status': 'completed'}
              for index, count in enumerate(cells)]
    report = convergence_report(adaptation, _Setup(adaptation, {}, ['mesh'], 2), levels, 'max_levels', 'not converged',
                                ModelSpec())
    add('report_fields', {'target', 'method', 'levels', 'richardson', 'converged', 'stop_reason', 'message',
                          'adapted_meshes'} <= set(report) and report['converged'] is False
        and report['richardson']['levels'] == [0, 1, 2] and report['adapted_meshes'][0]['name'] == 'mesh'
        and report['stop_reason'] in STOP_REASONS, sorted(report))

    widths = np.array([1.0, 1.2, 1.5, 1.5, 1.2, 1.0, 0.8])
    centres = np.cumsum(widths) - 0.5 * widths
    quadratic = np.repeat((centres ** 2)[:, None], 3, axis=1)
    parts = interpolation_indicator(quadratic, [widths, np.ones(3)])
    expected = (widths[1:-1] ** 2 * 2.0) ** 2 * widths[1:-1]
    add('indicator_quadratic', np.allclose(parts[0][1:-1, 1], expected, rtol=1e-12) and not parts[1].any(),
        float(np.max(np.abs(parts[0][1:-1, 1] / expected - 1.0))))
    linear = np.repeat(centres[:, None], 3, axis=1)
    add('indicator_linear_zero', float(np.max(interpolation_indicator(linear, [widths, np.ones(3)]))) < 1e-24)

    edges = np.arange(31, dtype=np.float64)
    bands = axis_bands(np.array([5, 6, 7, 20]), np.ones(30), edges, 1.0)
    add('axis_bands', bands == [(3.0, 10.0, 0.5), (18.0, 23.0, 0.5)], bands)

    registry = get_registry()
    auto = _small_spec('auto')
    shapes = resolve_geometry(auto, {}, errors={})
    auto_estimate = estimate_mesh(auto, {}, registry)
    frozen_auto = auto.copy()
    frozen_auto.meshes = [frozen_mesh_spec(auto, auto.meshes[0], {}, registry, shapes)]
    frozen_estimate = estimate_mesh(frozen_auto, {}, registry)
    add('freeze_auto_same_grid', frozen_auto.meshes[0].mode == 'nonuniform'
        and frozen_estimate.dx_cells.shape == auto_estimate.dx_cells.shape
        and frozen_estimate.dy_cells.shape == auto_estimate.dy_cells.shape
        and np.allclose(frozen_estimate.dx_cells, auto_estimate.dx_cells, rtol=1e-12)
        and np.allclose(frozen_estimate.dy_cells, auto_estimate.dy_cells, rtol=1e-12),
        [auto_estimate.nx, auto_estimate.ny, frozen_estimate.nx, frozen_estimate.ny])

    uniform = _small_spec('uniform')
    base_estimate = estimate_mesh(uniform, {}, registry)
    frozen = frozen_mesh_spec(uniform, uniform.meshes[0], {}, registry, shapes)
    finer = uniform.copy()
    finer.meshes = [uniform_mesh_spec(frozen, 4.0 ** -0.5)]
    ratio = estimate_mesh(finer, {}, registry).n_cells / base_estimate.n_cells
    add('uniform_growth', abs(ratio - 4.0) < 0.05, ratio)

    peak = np.zeros((2, base_estimate.nx, base_estimate.ny))
    peak[:, 10, 5] = 1.0
    adapted = adaptive_mesh_spec(uniform, frozen, base_estimate, peak, 1.7, True, {}, registry, shapes)
    bands = {axis_spec.axis: [(band.lo, band.hi, band.dx) for band in axis_spec.bands]
             for axis_spec in adapted[0].axes} if adapted is not None else {}
    add('adaptive_bands_both_axes', adapted is not None and adapted[2] == 1
        and np.allclose(bands.get('x', [(0, 0, 0)])[0], (0.8e-6, 1.3e-6, 0.5 * _SMALL_DX), rtol=1e-9)
        and np.allclose(bands.get('y', [(0, 0, 0)])[0], (0.3e-6, 0.8e-6, 0.5 * _SMALL_DX), rtol=1e-9)
        and adapted[1].n_cells > base_estimate.n_cells and adapted[0].mode == 'nonuniform'
        and any(refinement.object == 'block' for refinement in adapted[0].refinements), bands)
    if adapted is not None:   # the snapped block faces stay cell edges of the graded grid
        x_edges = adapted[1].x_edges
        add('adaptive_snapped_faces', min(abs(x_edges - 1e-6)) < 1e-11 and min(abs(x_edges - 1.2e-6)) < 1e-11)
    peak[1, 10, 5] = 0.1
    anisotropic = adaptive_mesh_spec(uniform, frozen, base_estimate, peak, 1.7, True, {}, registry, shapes)
    add('adaptive_anisotropic_x_only', anisotropic is not None
        and {axis_spec.axis for axis_spec in anisotropic[0].axes if axis_spec.bands} == {'x'})

    return checks.result()


def _levels(results, name):
    """(cells, values, report) of one run's convergence report (refined window cells, target values per level)."""
    report = results[name].summary()['convergence']
    levels = [record for record in report['levels'] if record['status'] == 'completed']
    return [record['refined_cells'] for record in levels], [record['value'] for record in levels], report


def _check_uniform_richardson(results):
    """Order in cells of the finest three levels vs the S7.8 record; the extrapolation of the coarsest three must lie
    closer to the finest extrapolation than the finest level's own value does."""
    from .study_adaptation import richardson
    cells, values, report = _levels(results, 'uniform')
    finest = report['richardson']
    coarse, coarse_note = richardson(cells[:3], values[:3], 2) if len(cells) >= 4 else (None, 'fewer than 4 levels')
    details = {'cells': cells, 'values': values, 'finest_fit': finest, 'coarse_fit': coarse,
               'coarse_note': coarse_note, 'richardson_note': report['richardson_note']}
    if finest is None or coarse is None or finest['order_cells'] is None:
        return {'measured': math.nan, 'expected': _STRIP_ORDER, 'error': math.inf,
                'tolerance': _STRIP_ORDER_TOLERANCE, 'details': details}
    extrapolation_gap = abs(coarse['extrapolated'] - finest['extrapolated'])
    finest_gap = abs(values[-1] - finest['extrapolated'])
    details.update(extrapolation_gap=extrapolation_gap, finest_gap=finest_gap)
    order = finest['order_cells']
    return {'measured': order, 'expected': _STRIP_ORDER,
            'error': abs(order - _STRIP_ORDER) if extrapolation_gap < finest_gap else math.inf,
            'tolerance': _STRIP_ORDER_TOLERANCE, 'details': details}


def _cells_at_error(cells, errors, wanted):
    """Cells at which the levels' errors reach `wanted`: the first level at or below it, log-log interpolated from
    the level before (that level's own cells when it is the first or the errors do not shrink); None when none does."""
    index = next((level for level, error in enumerate(errors) if error <= wanted), None)
    if index is None:
        return None
    if index == 0 or not errors[index - 1] > errors[index] > 0.0:
        return float(cells[index])
    fraction = math.log(errors[index - 1] / wanted) / math.log(errors[index - 1] / errors[index])
    return float(cells[index - 1] * (cells[index] / cells[index - 1]) ** fraction)


def _adaptive_gain_check(relative_error, gain, uniform_level=None, uniform_fit_levels=None):
    """check(results): adaptive cells at relative_error of the uniform Richardson value (_cells_at_error over the
    adaptive levels), over the uniform cells that error needs (the uniform fit's C N^-q); passes at <= gain.
    uniform_level (an index into the completed uniform levels) takes that level's own error under the fit as the
    relative error instead (its cells are then the uniform cells when the fit runs through it). uniform_fit_levels n
    fits the coarsest n uniform levels instead of the report's finest three."""
    def check(results):
        from .study_adaptation import cells_for_error, richardson
        uniform_level_cells, uniform_values, uniform_report = _levels(results, 'uniform')
        cells, values, adaptive_report = _levels(results, 'adaptive')
        reference = (uniform_report['richardson'] if uniform_fit_levels is None else
                     richardson(uniform_level_cells[:uniform_fit_levels], uniform_values[:uniform_fit_levels], 2)[0])
        details = {'uniform_fit': reference, 'adaptive_cells': cells, 'adaptive_values': values,
                   'adaptive_stop': adaptive_report['stop_reason']}
        if reference is None or reference['order_cells'] is None:
            return {'measured': math.nan, 'expected': gain, 'error': math.inf, 'tolerance': gain, 'details': details}
        target = reference['extrapolated']
        wanted = relative_error if uniform_level is None else abs(uniform_values[uniform_level] / target - 1.0)
        errors = [abs(value / target - 1.0) for value in values]
        reached = _cells_at_error(cells, errors, wanted)
        uniform_cells = cells_for_error(reference, wanted) if wanted > 0.0 else math.inf
        details.update(relative_error=wanted, adaptive_errors=errors, uniform_cells=uniform_cells,
                       adaptive_cells_at_error=reached)
        ratio = math.inf if reached is None or not math.isfinite(uniform_cells) else reached / uniform_cells
        return {'measured': ratio, 'expected': gain, 'error': ratio, 'tolerance': gain, 'details': details}
    return check


_strip_gain_check = _adaptive_gain_check(_ADAPTIVE_ERROR, _ADAPTIVE_GAIN, uniform_fit_levels=_STRIP_UNIFORM_FIT_LEVELS)


def _check_strip_heater(results):
    """Uniform Richardson order (_check_uniform_richardson) and adaptive gain (fit of the coarsest three uniform
    levels); measured = error = the worse item in its own tolerance units, tolerance 1."""
    items = {'uniform_order': _check_uniform_richardson(results), 'adaptive_gain': _strip_gain_check(results)}
    normalized = {name: item['error'] / item['tolerance'] for name, item in items.items()}
    worst_item = max(normalized, key=lambda name: math.inf if math.isnan(normalized[name]) else normalized[name])
    return {'measured': normalized[worst_item], 'expected': 0.0, 'error': normalized[worst_item], 'tolerance': 1.0,
            'details': {'worst_item': worst_item, 'normalized': normalized, **items}}


def _cpu_heat(widths_x, widths_y, conductivity, source=0.0, x_lo_temperature=1.0):
    """Steady cpu-backend HeatPhysics (T0 = 0, float64 CG at _INDICATOR_CG_TOLERANCE) on a padding-free grid of these
    widths: conductivity (nx,) per column, uniform source [W/m^3], Dirichlet x_lo / x_hi at x_lo_temperature / 0 K."""
    from ..core.constants import FACES
    from ..core.domain import Domain
    from ..core.fv_diffusion import BCSide
    from ..core.material_grid import MaterialGrid
    from ..core.mesh import NonUniformGrid2D
    from ..physics.heat.solver import HeatPhysics
    grid = NonUniformGrid2D(np.asarray(widths_x, dtype=np.float64), np.asarray(widths_y, dtype=np.float64),
                            verbose=False)
    domain = Domain(grid, MaterialGrid(grid.nx, grid.ny, backend='cpu'), size_x=grid.size_x, size_y=grid.size_y,
                    non_uniform=True, npml_sides={face: 0 for face in FACES})
    heat = HeatPhysics(domain, T0=0.0, tol=_INDICATOR_CG_TOLERANCE, max_iter=20000, precision='float64')
    columns = np.ascontiguousarray(np.broadcast_to(np.asarray(conductivity, dtype=np.float32)[:, None],
                                                   (grid.nx, grid.ny)))
    for name in ('k_xx', 'k_yy'):
        domain.materials.assign(name, None, columns)
    heat.materials_changed(('k_xx', 'k_yy'))
    heat.set_face_bc('x_lo', BCSide('dirichlet', value=x_lo_temperature))
    heat.set_face_bc('x_hi', BCSide('dirichlet', value=0.0))
    if source:
        heat.add_heat_source(np.ones((grid.nx, grid.ny), dtype=bool), source)
    heat.step(0.0, math.inf)
    return heat


def _check_physics_indicators(_results):
    """A2 indicator sub-checks (pure numpy and cpu-backend heat solves); error = number of failed sub-checks."""
    from ..core.error_indicators import (energy_interpolation_indicator, face_jump_indicator, interpolation_indicator,
                                         relative_field)
    checks = SubChecks()
    add = checks.add

    # pure: the linear flux q = -k u' of u = x^2 on uniform cells: jump 2 k h per face, interior cells h^3 (2k)^2 A / k
    width, conductivity, area = 0.25, 3.0, 2.0
    centres = width * (np.arange(6) + 0.5)
    flux = np.repeat((-2.0 * conductivity * centres)[:, None], 2, axis=1)
    coupled_x, coupled_y = np.ones((5, 2), dtype=bool), np.ones((6, 1), dtype=bool)
    parts = face_jump_indicator((flux,), (np.zeros((6, 2)),), coupled_x, coupled_y, width, width, area, area,
                                1.0 / conductivity, 1.0 / conductivity)
    expected = width ** 3 * (2.0 * conductivity) ** 2 * area / conductivity
    add('jump_linear_flux', np.allclose(parts[0][1:-1], expected, rtol=1e-12)
        and np.allclose(parts[0][[0, -1]], 0.5 * expected, rtol=1e-12) and not parts[1].any(),
        float(np.max(np.abs(parts[0][1:-1] / expected - 1.0))))
    coupled_x[2, 0] = False   # an uncoupled face (held / non-conducting neighbour) adds nothing
    masked = face_jump_indicator((flux,), (np.zeros((6, 2)),), coupled_x, coupled_y, width, width, area, area,
                                 1.0 / conductivity, 1.0 / conductivity)
    add('jump_uncoupled_face', np.isclose(masked[0][2, 0], 0.5 * expected, rtol=1e-12)
        and np.isclose(masked[0][3, 0], 0.5 * expected, rtol=1e-12) and np.isclose(masked[0][2, 1], expected))
    # a complex two-component traction: |J|^2 summed over the components, per-cell compliance on each side
    first = np.array([[1.0 + 1.0j], [3.0 - 1.0j]])
    second = np.array([[0.5], [2.5]])
    cell_compliance = np.array([[2.0], [5.0]])
    no_y_faces = (np.zeros((2, 1)), np.zeros((2, 1)))
    traction = face_jump_indicator((first, second), no_y_faces, np.ones((1, 1), dtype=bool),
                                   np.ones((2, 0), dtype=bool), 0.5, 0.5, 3.0, 3.0, cell_compliance, cell_compliance)
    jump_squared = abs(2.0 - 2.0j) ** 2 + 2.0 ** 2
    expected_traction = 0.5 * 0.5 * jump_squared * 3.0 * cell_compliance[:, 0]
    add('jump_complex_traction', np.allclose(traction[0][:, 0], expected_traction, rtol=1e-12),
        traction[0][:, 0].tolist())
    # energy-weighted interpolation: a complex field = its real and imaginary parts, times the weight
    widths = [np.array([1.0, 1.2, 1.5, 1.5, 1.2]), np.ones(3)]
    quadratic = np.repeat(((np.cumsum(widths[0]) - 0.5 * widths[0]) ** 2)[:, None], 3, axis=1)
    weighted = energy_interpolation_indicator([(quadratic * (1.0 - 2.0j), 4.0), (quadratic, 1.0)], widths)
    add('energy_interpolation_weights', np.allclose(weighted, 21.0 * interpolation_indicator(quadratic, widths),
                                                    rtol=1e-12))

    # cpu heat: two layers (k 1 | 4) between Dirichlet faces, no source: constant flux, FV exact -> no jump; the
    # interpolation indicator of T flags the kink
    cell, layers = 0.125, np.where(np.arange(10) < 5, 1.0, 4.0)   # float32-exact widths (NonUniformGrid2D)
    widths_x, widths_y = np.full(10, cell), np.ones(3)
    layered = _cpu_heat(widths_x, widths_y, layers)
    flux_jumps = layered.error_indicator()
    flux = float(np.mean(np.asarray(layered.system.flux_at_centres()[0])))
    energy = flux ** 2 * float(np.sum(cell / layers)) * 3.0   # sum q^2 / k V over the 10 x 3 cells (dy = 1)
    kink = interpolation_indicator(relative_field(np.asarray(layered.system.unknown2d)), [widths_x, widths_y])
    add('heat_layered_no_jump', flux_jumps.shape == (2, 10, 3) and float(flux_jumps.max()) <= 1e-14 * energy,
        float(flux_jumps.max()) / energy)
    linear_cells = max(float(kink[0][1:3].max()), float(kink[0][7:9].max()), 1e-300)
    add('interpolation_flags_kink', float(kink[0][4:6].min()) > 1e6 * linear_cells,
        [float(kink[0][4:6].min()), linear_cells])
    # cpu heat, uniform k = 2 and source Q, both faces at 0 K: face flows exact by symmetry -> jumps Q h per face
    source = 1024.0
    parts = _cpu_heat(widths_x, widths_y, np.full(10, 2.0), source, x_lo_temperature=0.0).error_indicator()
    expected = cell ** 3 * source ** 2 * 1.0 / 2.0   # h^3 Q^2 dy / k per interior cell
    add('heat_source_jumps', np.allclose(parts[0][1:-1], expected, rtol=1e-8)
        and np.allclose(parts[0][[0, -1]], 0.5 * expected, rtol=1e-8)
        and float(parts[1].max()) <= 1e-16 * expected, float(np.max(np.abs(parts[0][1:-1] / expected - 1.0))))

    return checks.result()


# ---- A3: triangles ----
def _fem_layers_spec():
    """1 x 1 um window whose right half (x >= 0.5 um) is the object 'right' (a second layer)."""
    return ModelSpec(title='Self-test: triangle error indicators', domain=DomainSpec(size_x=_FEM_SIDE, size_y=_FEM_SIDE),
                     geometry=[GeometrySpec('right', 'rectangle', {'x0': 0.5 * _FEM_SIDE, 'y0': 0.0, 'x1': _FEM_SIDE,
                                                                   'y1': _FEM_SIDE})])


def _fem_mesh(size):
    """Triangle mesh of _fem_layers_spec with uniform element size `size` (explicit sizing, no ring)."""
    from .meshing import TriangleSizing
    from .triangle_meshing import triangulate
    sizing = TriangleSizing(h_max=size, h_min=size / 100.0, growth=1.3, curvature_factor=0.3, object_sizes={},
                            pml_widths={}, pml_layers=8, mesher='builtin')
    return triangulate(_fem_layers_spec(), {}, sizing=sizing)


def _fem_solve(mesh, order, conductivity, source=0.0, x_lo_value=0.0):
    """(FEMDiffusionSystem after one solve, tensor): conductivity (left, right), uniform source, Dirichlet x_lo at
    x_lo_value and x_hi at 0, insulated y faces."""
    from ..fem.diffusion import EdgeCondition, FEMDiffusionSystem, coefficient_tensor
    right = mesh.region == mesh.region_names.index('right')
    values = np.where(right, conductivity[1], conductivity[0])
    tensor = coefficient_tensor(values, values)
    system = FEMDiffusionSystem(mesh, order, None, backend='cpu')
    system.set_face_condition('x_lo', EdgeCondition('dirichlet', value=x_lo_value))
    system.set_face_condition('x_hi', EdgeCondition('dirichlet', value=0.0))
    system.assemble(tensor)
    system.solve(np.full(mesh.n_elements, float(source)))
    return system, tensor


def _fem_heat_spec():
    """Heat (k 1 | 4 layers) on a uniform triangles mesh (_FEM_H) of _fem_layers_spec, cpu backend."""
    spec = _fem_layers_spec()
    spec.materials = [MaterialSpec('left_layer', properties={'k': 1.0}, selection=['all']),
                      MaterialSpec('right_layer', properties={'k': 4.0}, selection=['right'])]
    spec.physics = [PhysicsSpec('heat', 'heat', features=[FeatureSpec('hot', 'temperature', ['x_lo'],
                                                                      {'temperature': 300.0})])]
    spec.meshes = [MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_FEM_H)]
    spec.solver_defaults = SolverDefaultsSpec(backend='cpu')
    return spec


def _check_triangle_parts(_results):
    """A3 sub-checks (no model run): exact solutions give no indicator, P1 / quadratic edge-element indicators shrink
    with h, adaptive triangle sizes, size_points rules; error = number of failed sub-checks."""
    from ..fem.indicators import diffusion_indicator, element_sizes
    from .geometry import resolve_geometry
    from .mesh_refinement import adaptive_mesh_spec, frozen_mesh_spec, uniform_mesh_spec
    from .meshing import estimate_mesh, size_points_problem
    from .schema import get_registry
    from .triangle_meshing import triangulate
    checks = SubChecks()
    add = checks.add

    mesh = _fem_mesh(_FEM_H)
    # piecewise linear across the k 1 | 4 layers (the interface is a mesh line): exact for P1 and P2
    flux = 1.0 / (0.5 * _FEM_SIDE / 1.0 + 0.5 * _FEM_SIDE / 4.0)   # |k du/dx| of the 1 K drop
    for order in (1, 2):
        system, tensor = _fem_solve(mesh, order, (1.0, 4.0), x_lo_value=1.0)
        total = float(diffusion_indicator(system, tensor).sum()) / (flux ** 2 * _FEM_SIDE ** 2)
        add(f'p{order}_layers_exact', total <= _FEM_EXACT, total)
    # uniform source between faces at 0: u = f x (L - x) / (2 k) is exact in P2 (Hessian residual and jumps vanish)
    source, conductivity = 1e12, 2.0
    scale = (source * _FEM_SIDE) ** 2 * _FEM_SIDE ** 2 / conductivity
    system, tensor = _fem_solve(mesh, 2, (conductivity, conductivity), source)
    quadratic_total = float(diffusion_indicator(system, tensor, np.full(mesh.n_elements, source)).sum()) / scale
    add('p2_parabola_exact', quadratic_total <= _FEM_EXACT, quadratic_total)
    totals = []
    for size in (_FEM_H, 0.5 * _FEM_H):
        fine_mesh = _fem_mesh(size)
        system, tensor = _fem_solve(fine_mesh, 1, (conductivity, conductivity), source)
        totals.append(float(diffusion_indicator(system, tensor, np.full(fine_mesh.n_elements, source)).sum()) / scale)
    ratio = totals[0] / totals[1] if totals[1] > 0.0 else math.inf
    add('p1_parabola_rate', _FEM_P1_RATIO[0] <= ratio <= _FEM_P1_RATIO[1] and totals[0] > 1e6 * quadratic_total,
        {'totals': totals, 'ratio': ratio})

    # quadratic edge elements on the walled TE slab of the em_modes self-tests: jumps shrink like h^4
    from ..physics.em.fem_modes import FEMModeMaterials, mode_error_indicator
    from ..physics.em.selftests_modes_fem import (_SLAB_INDICES, _SLAB_WAVELENGTH, _direct_solve, _sizing,
                                                  _slab_spec)
    em_totals = []
    for size in _FEM_EM_SIZES:
        solve, layout = _direct_solve(_slab_spec(), _sizing(size), {'core': _SLAB_INDICES[0]}, _SLAB_INDICES[1],
                                      {'x_lo': 'pec', 'x_hi': 'pec', 'y_lo': 'pec', 'y_hi': 'pec'}, _SLAB_WAVELENGTH,
                                      2, 2)
        slab_mesh = layout.mesh
        eps = np.full(slab_mesh.n_elements, _SLAB_INDICES[1] ** 2, dtype=np.complex128)
        eps[slab_mesh.region == slab_mesh.region_names.index('core')] = _SLAB_INDICES[0] ** 2
        materials = FEMModeMaterials(eps_xx=eps, eps_yy=eps, eps_zz=eps,
                                     mu=np.ones(slab_mesh.n_elements, dtype=np.complex128),
                                     pec=np.zeros(slab_mesh.n_elements, dtype=bool))
        values = mode_error_indicator(layout, solve.modes[0], materials)
        em_totals.append(float(values.sum()) if np.all(np.isfinite(values)) and values.min() >= 0.0 else math.nan)
    em_ratio = em_totals[0] / em_totals[1] if em_totals[1] > 0.0 else math.nan
    add('em_jumps_rate', em_ratio >= _FEM_EM_RATIO, {'totals': em_totals, 'ratio': em_ratio})

    # adaptive sizes: a Gaussian indicator around p refines there by ~growth in element count, far elements stay
    registry = get_registry()
    spec = _fem_heat_spec()
    shapes = resolve_geometry(spec, {}, errors={})
    heat_mesh = triangulate(spec, {}, registry)
    estimate = estimate_mesh(spec, {}, registry)
    frozen = frozen_mesh_spec(spec, spec.meshes[0], {}, registry, shapes)
    peak = np.array([0.3, 0.7]) * _FEM_SIDE
    centroids = heat_mesh.points[heat_mesh.triangles].mean(axis=1)
    distance = np.linalg.norm(centroids - peak, axis=1)
    indicator = np.exp(-(distance / (0.15 * _FEM_SIDE)) ** 2)
    adapted = adaptive_mesh_spec(spec, frozen, estimate, indicator, 1.7, True, {}, registry, shapes,
                                 triangle_mesh=heat_mesh, element_order=2)
    add('adaptive_size_points', adapted is not None and len(adapted[0].size_points) > 0
        and adapted[1].n_cells > estimate.n_cells, None if adapted is None else len(adapted[0].size_points))
    if adapted is not None:
        refined = spec.copy()
        refined.meshes = [adapted[0]]
        new_mesh = triangulate(refined, {}, registry)
        growth = new_mesh.n_elements / heat_mesh.n_elements
        add('adaptive_growth', _FEM_GROWTH_RANGE[0] <= growth <= _FEM_GROWTH_RANGE[1],
            {'old': heat_mesh.n_elements, 'new': new_mesh.n_elements, 'estimate_ratio':
             adapted[1].n_cells / estimate.n_cells})
        new_distance = np.linalg.norm(new_mesh.points[new_mesh.triangles].mean(axis=1) - peak, axis=1)
        old_sizes, new_sizes = element_sizes(heat_mesh), element_sizes(new_mesh)
        near = float(new_sizes[new_distance < 0.15 * _FEM_SIDE].mean() / old_sizes[distance < 0.15 * _FEM_SIDE].mean())
        far = float(new_sizes[new_distance > 0.7 * _FEM_SIDE].mean() / old_sizes[distance > 0.7 * _FEM_SIDE].mean())
        add('adaptive_local_sizes', near < 0.8 and 0.75 <= far <= 1.33, {'near_ratio': near, 'far_ratio': far})
        loaded = ModelSpec.from_dict(refined.to_dict())
        add('size_points_round_trip', loaded.meshes[0].size_points == [list(map(float, entry))
                                                                       for entry in adapted[0].size_points])
        scaled = uniform_mesh_spec(adapted[0], 0.5)
        add('size_points_uniform_scale', math.isclose(scaled.size_points[0][2], 0.5 * adapted[0].size_points[0][2]))
    add('size_points_problems', size_points_problem([]) == '' and size_points_problem([[0.0, 0.0, 1e-7]]) == ''
        and bool(size_points_problem([[0.0, 0.0, -1.0]])) and bool(size_points_problem([[0.0, 0.0]]))
        and bool(size_points_problem({'x': 1})))
    return checks.result()


def _lshape_spec(adaptation):
    """Electric currents (P2 triangles, cpu) in an L-shaped conductor: the 2 x 2 um square without its upper right
    quarter (sigma_dc 0 there), 1 V on the top face of the left arm, 0 V on the right face of the lower arm; the
    insulated re-entrant corner at (1, 1) um gives V ~ r^(2/3). Target: the terminal current I_top."""
    side, half = _LSHAPE_SIDE, 0.5 * _LSHAPE_SIDE
    currents = PhysicsSpec('currents', 'electric_currents', features=[
        FeatureSpec('top', 'voltage', ['y_hi'], {'voltage': 1.0}),
        FeatureSpec('right', 'voltage', ['x_hi'], {'voltage': 0.0})])
    return ModelSpec(title='Self-test: L-shaped conductor mesh refinement on triangles',
                     domain=DomainSpec(size_x=side, size_y=side),
                     geometry=[GeometrySpec('notch', 'rectangle', {'x0': half, 'y0': half, 'x1': side, 'y1': side})],
                     materials=[MaterialSpec('metal', properties={'sigma_dc': 1e6}, selection=['all']),
                                MaterialSpec('gap', properties={'sigma_dc': 0.0}, selection=['notch'])],
                     physics=[currents],
                     meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_LSHAPE_H)],
                     solver_defaults=SolverDefaultsSpec(backend='cpu'),
                     study=StudySpec(type='stationary', adaptation=adaptation),
                     derived_values=[DerivedValueSpec('I_top', 'currents_current_top', [], 'mean')])


def _fem_rib_spec(adaptation):
    """The A2 TFLN rib (_rib_modes_spec) in a 2.4 x 2.4 um window with pec walls on uniform 200 nm linear
    triangles, cpu backend."""
    spec = _rib_modes_spec(adaptation)
    for parameter in spec.parameters:
        parameter.expression = _FEM_RIB_WINDOW.get(parameter.name, parameter.expression)
    spec.meshes = [MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=_FEM_RIB_DX)]
    spec.solver_defaults = SolverDefaultsSpec(backend='cpu')
    settings = spec.physics[0].settings
    settings.update(bc_x_hi='pec', bc_y_lo='pec', bc_y_hi='pec', discretization='linear')
    del settings['pml_sigma_factor']
    return spec


ADAPTATION_SELFTEST_CASES = (
    SelfTestCase(
        name='adaptation_stop_rules',
        label='Mesh refinement levels: stop rule, Richardson, indicator and refined meshes',
        description='CPU checks of the S7.7 refinement loop parts (no model run): Richardson through v = 2 + 3 N^-0.7 '
                    '(order 0.7 in cells, 1.4 in h, v_inf 2, C 3; the cells for 1e-3), no fit for alternating or '
                    'growing changes, the exact case; the stop rule (converged at relative change <= tolerance, '
                    'max_levels, not_finite, continue) and the next-level checks (no growth, max_cells, memory); the '
                    'report fields; the interpolation indicator (h^2 d2u)^2 V exact for a quadratic on graded cells, '
                    'zero for a linear field; axis bands of marked cells (half width, 2-cell margins, merged runs); '
                    'a frozen auto heat mesh giving the same grid; a uniform level with growth 4 (cells x 4 within '
                    '0.05); adaptive bands around one marked cell on both axes (x only when the y part is 10x '
                    'smaller), the uniform mesh turned graded with its block faces kept as cell edges. error = number '
                    'of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_stop_rules),
    SelfTestCase(
        name='adaptation_tensor_strip_heater',
        label='Mesh refinement levels: uniform Richardson order and adaptive tensor bands on the strip heater',
        description='S7.8 strip heater (steady heat, gold heater above a silicon core in silica, multigrid CG) from '
                    'uniform 150 nm cells. Uniform sequence, growth 4 (cell size halved) over 4 levels (to 18.75 nm, '
                    '2.56e6 cells), target dT_heater: the Richardson order in cells of the finest three levels must '
                    'match the S7.8 record 0.68 within 0.1, and the extrapolation of the coarsest three must lie closer '
                    "to the finest extrapolation than the finest level's own value (error inf otherwise). Adaptive "
                    'bands (Kelly heat-flux jump indicator, growth 2, anisotropic, tolerance 1e-4, at most 8 levels and '
                    '4e5 planned cells: one level past the recorded crossing at 2.9e5) vs the Richardson fit C N^-q of '
                    'the coarsest three uniform levels, which gives the uniform cells (~1.46e6) for a 1e-3 relative '
                    'dT_heater error: the adaptive window cells at 1e-3 of that extrapolation (log-log interpolated '
                    'between the bracketing adaptive levels) / those uniform cells must be <= 0.5 (S7.8 graded gain 19x '
                    'is the ceiling; the cap makes the effective bound ~0.27). measured = error = the worse item in its '
                    'own tolerance units, tolerance 1. Tolerances are first estimates. Not yet confirmed by a run as '
                    'one case with the 4e5 cap.',
        specs=lambda: {'uniform': _strip_heater_spec(_uniform_sequence(3)),
                       'adaptive': _strip_heater_spec(AdaptationSpec('dT_heater', method='adaptive', tolerance=1e-4,
                                                                     max_levels=8, growth=2.0,
                                                                     max_cells=_STRIP_ADAPTIVE_MAX_CELLS))},
        check=_check_strip_heater, heavy=True, max_wall_time=1800.0),
    SelfTestCase(
        name='adaptation_physics_indicators',
        label='Mesh refinement levels: physics error indicators (flux / traction jumps, EM energy weights)',
        description='CPU checks of the mesh plan A2 indicators (no model run): Kelly face jumps 0.5 w |J|^2 A / c exact '
                    'for the linear flux of u = x^2 (interior cells, half at the ends, nothing on an uncoupled face), '
                    'a complex two-component traction with per-cell compliance; the energy-weighted interpolation of a '
                    'complex field = weight x (real + imaginary parts); cpu-backend heat: two layers (k 1 | 4) '
                    'between held faces give no flux jump (<= 1e-14 of the field energy) while the interpolation '
                    'indicator of T flags the kink cells (> 1e6 x the linear cells), and a uniform source Q between '
                    'faces at 0 K gives h^3 Q^2 dy / k per interior cell (1e-8). error = number of failed sub-checks, '
                    'tolerance 0.',
        specs=lambda: {},
        check=_check_physics_indicators),
    SelfTestCase(
        name='adaptation_flux_jump_disc_heater',
        label='Mesh refinement levels: flux-jump adaptive bands on the axisymmetric disc heater',
        description='S7.8 disc heater (axisymmetric heat, 1 mm disc flux into a half-space held beyond a 10 mm '
                    'hemisphere, conformal held edge, multigrid CG) from uniform 50 um cells: adaptive bands (Kelly '
                    'heat-flux jump indicator, growth 2, tolerance 1e-4, at most 8 levels and 2.5e5 cells: a finer '
                    'level cannot pass) vs a uniform sequence of 3 levels (growth 4) whose Richardson fit gives the '
                    'uniform cells (~4.6e5) for a 1e-3 relative heater_mean_rise error. measured = adaptive cells at '
                    '1e-3 of the uniform extrapolation (log-log interpolated between the bracketing adaptive levels) '
                    '/ those uniform cells; passes at <= 0.5 (S7.8 graded gain 6.8x). Tolerance is a first estimate.',
        specs=lambda: {'uniform': _disc_heater_spec(_uniform_sequence(2, 'heater_mean_rise')),
                       'adaptive': _disc_heater_spec(AdaptationSpec('heater_mean_rise', method='adaptive',
                                                                    tolerance=1e-4, max_levels=8, growth=2.0,
                                                                    max_cells=_DISC_ADAPTIVE_MAX_CELLS))},
        check=_adaptive_gain_check(_ADAPTIVE_ERROR, _ADAPTIVE_GAIN), heavy=True, max_wall_time=1800.0),
    SelfTestCase(
        name='adaptation_rib_modes',
        label='Mesh refinement levels: energy-weighted adaptive bands on a rib waveguide mode',
        description='S7.8 TFLN rib (em_modes, quasi-TE0 at 1550 nm, subpixel smoothing) from uniform 50 nm cells: '
                    'adaptive bands (energy-weighted interpolation indicator of the mode, growth 2, tolerance 1e-6, at '
                    'most 8 levels and 7.5e4 planned cells: stops after the ~4.5e4-cell level 3) vs a uniform sequence '
                    'of 3 levels (growth 4, to 12.5 nm, 5.95e4 cells) whose exact Richardson fit extrapolates n_eff_1. '
                    'measured = adaptive cells at the relative error of the 12.5 nm uniform level under that fit (~4e-5; '
                    'log-log interpolated between the bracketing adaptive levels) / those uniform cells; passes at <= '
                    '0.5 (S7.8 graded gain 2.5x; last run ~0.32 by the recorded levels). Not a fixed 1e-5: the fit and '
                    "the adaptive levels' own extrapolation differ by ~1.8e-5 relative, so 1e-5 of the fit is never "
                    'reached. Tolerances are first estimates.',
        specs=lambda: {'uniform': _rib_modes_spec(_uniform_sequence(2, 'n_eff_1')),
                       'adaptive': _rib_modes_spec(AdaptationSpec('n_eff_1', method='adaptive', tolerance=1e-6,
                                                                  max_levels=8, growth=2.0,
                                                                  max_cells=_RIB_ADAPTIVE_MAX_CELLS))},
        check=_adaptive_gain_check(None, _ADAPTIVE_GAIN, uniform_level=_RIB_REFERENCE_LEVEL), heavy=True,
        max_wall_time=3600.0),
    SelfTestCase(
        name='adaptation_triangle_parts',
        label='Mesh refinement levels on triangles: error indicators and adapted sizes',
        description='CPU checks of the mesh plan A3 parts (no model run) on a 1 x 1 um window with a k 1 | 4 layer '
                    'boundary as a mesh line: the residual / flux-jump indicator vanishes (<= 1e-18 of its energy '
                    'scale) for the piecewise-linear layered field (P1 and P2) and the source parabola (P2: Hessian '
                    'residual and jumps cancel exactly); P1 with the source falls 2.5 to 6.5x when h halves (h^2); the '
                    'em_modes edge-element jumps (tangential H, normal D) of the walled TE slab fall >= 6x when h halves '
                    '(quadratic, h^4); a Gaussian indicator on a uniform heat triangles mesh gives size points whose '
                    'rebuilt mesh has 1.3 to 2.4x the elements (growth 1.7), elements near the peak < 0.8x the old '
                    'size and far ones within 0.75 to 1.33x; size_points survive the JSON round trip, scale in a '
                    'uniform level and malformed entries are reported. error = number of failed sub-checks, '
                    'tolerance 0. The em pair at 45 / 22.5 nm not yet confirmed by a run.',
        specs=lambda: {},
        check=_check_triangle_parts),
    SelfTestCase(
        name='adaptation_fem_lshape',
        label='Mesh refinement levels on triangles: adaptive L-shaped conductor (corner singularity)',
        description='Electric currents (P2 triangles) in an L-shaped conductor (2 x 2 um square without its upper right '
                    'quarter, 1 V on the top of the left arm, 0 V on the right of the lower arm; V ~ r^(2/3) at the '
                    'insulated re-entrant corner) from uniform 250 nm elements: adaptive triangle sizes (flux-jump / '
                    'residual indicator, growth 2, tolerance 1e-8, at most 10 levels and 1500 elements: a finer level '
                    'cannot pass) vs a uniform sequence of 4 levels (growth 4) whose Richardson fit C N^-q (recorded q '
                    '1.28 for the current) gives the uniform elements (~5.5e3) for a 1e-4 relative I_top error. measured '
                    '= the adaptive window elements at 1e-4 of the uniform extrapolation (log-log interpolated '
                    'between the bracketing adaptive levels) / those uniform elements; passes at <= 0.25 '
                    '(optimal P2 adaptation converges like N^-2). Tolerances are first estimates.',
        specs=lambda: {'uniform': _lshape_spec(_uniform_sequence(3, 'I_top')),
                       'adaptive': _lshape_spec(AdaptationSpec('I_top', method='adaptive', tolerance=1e-8,
                                                               max_levels=10, growth=2.0,
                                                               max_cells=_LSHAPE_ADAPTIVE_MAX_CELLS))},
        check=_adaptive_gain_check(_LSHAPE_ERROR, _LSHAPE_GAIN), heavy=True, max_wall_time=1800.0),
    SelfTestCase(
        name='adaptation_fem_rib',
        label='Mesh refinement levels on triangles: adaptive rib waveguide mode',
        description='The adaptation_rib_modes TFLN rib (em_modes, quasi-TE0 at 1550 nm) in a 2.4 x 2.4 um window (1 um '
                    'box, 0.8 um air) with pec walls (no PML ring, whose width would shrink with each uniform level) on '
                    'linear triangles (discretization linear, n_eff error ~h^2: quadratic elements are within 2e-6 of '
                    'the converged value already at 200 nm, below the noise floor of the mesher, where a uniform '
                    'sequence alternates in sign) from uniform 200 nm elements: adaptive triangle sizes (tangential H '
                    '/ normal D jump indicator of the mode, growth 1.5, tolerance 1e-6, at most 8 levels and an '
                    'element estimate of 10,000: a finer level cannot pass) vs a uniform sequence of 4 levels (growth '
                    '4, to 25 nm) whose Richardson fit over the finest three extrapolates n_eff_1. measured = adaptive '
                    'elements at the relative error of the 50 nm uniform level (~1.2e4 elements; log-log interpolated '
                    'between the bracketing adaptive levels) / those uniform elements; passes at <= 0.5. Tolerances '
                    'are first estimates.',
        specs=lambda: {'uniform': _fem_rib_spec(_uniform_sequence(3, 'n_eff_1')),
                       'adaptive': _fem_rib_spec(AdaptationSpec('n_eff_1', method='adaptive', tolerance=1e-6,
                                                                max_levels=8, growth=_FEM_RIB_ADAPTIVE_GROWTH,
                                                                max_cells=_FEM_RIB_ADAPTIVE_MAX_CELLS))},
        check=_adaptive_gain_check(None, _ADAPTIVE_GAIN, uniform_level=_FEM_RIB_REFERENCE_LEVEL), heavy=True,
        max_wall_time=600.0),
)
