"""CPU self-tests of the run-time estimate (time_estimation_plan): the quick cpu machine profile (curves, interpolation,
round trip; never saved) and the recipes on a synthetic profile (pricing, count spans, study steps, sweeps, the
post-run time_check warnings, every template estimated without a failing recipe)."""
import math

from .schema import SelfTestCase
from .selftest_helpers import SubChecks
from .spec import DomainSpec, GeometrySpec, MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec, StudySpec, SweepSpec

# synthetic profile: every curve linear, so a price is size x rate exactly (log-log interpolation of a line)
_SECONDS_PER_BYTE = 1e-9
_SECONDS_PER_FLOP = 1e-10
_SECONDS_PER_KERNEL_FLOP = {32: 1e-12, 64: 64e-12}   # synthetic arith_float32 / _float64 (a consumer GPU's 1/64)
_SECONDS_PER_UNKNOWN = 1e-7
_DEVICE_FACTOR_RATIO = 0.25               # synthetic cudss_factor_2d_device[_mixed] over the host refactorisation curve
_SCALARS = {'launch': 1e-6, 'sync': 1e-5, 'python_call': 1e-7}
_PRICE_RTOL = 1e-9
_EXPECTED_CURVES = ('fused', 'dot', 'spmv', 'gemm', 'host_gemm', 'host_stream', 'host_lu_2d', 'host_solve_2d',
                    'host_lu_2d_complex', 'host_solve_2d_complex', 'host_lu_3d', 'host_solve_3d')
_BANDWIDTH_RANGE = (0.1e9, 1e12)          # B/s: a fused-kernel bandwidth outside it is a broken benchmark
_SIZE = (4e-6, 4e-6)                      # m
_DX = 50e-9                               # m
_DURATION = 1e-13                         # s: ~1000 Yee steps at dx 50 nm
_TRIANGLE_H = 0.5e-6                      # m: element size of the build-estimate triangle meshes (~150 triangles)


def _spec(physics_type, study):
    spec = ModelSpec(title='Self-test: run-time estimate', domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]))
    spec.meshes = [MeshSpec(mode='uniform', dx=_DX)]
    spec.physics = [PhysicsSpec('p', physics_type)]
    spec.study = study
    return spec


def _synthetic_profile():
    from ..core.time_profile import MachineProfile

    def line(rate):
        return (1.0, rate), (1e15, rate * 1e15)
    curves = {name: line(_SECONDS_PER_BYTE) for name in ('fused', 'fused_numba', 'spmv', 'dot', 'transfer',
                                                         'host_stream')}
    curves.update({name: line(_SECONDS_PER_FLOP) for name in ('gemm', 'host_gemm')})
    curves.update({f'arith_float{bits}': line(rate) for bits, rate in _SECONDS_PER_KERNEL_FLOP.items()})
    for dimensions in (2, 3):
        for kind in ('lu', 'solve'):
            curves[f'host_{kind}_{dimensions}d'] = line(_SECONDS_PER_UNKNOWN)
    for name in ('cudss_plan_2d', 'cudss_factor_2d', 'cudss_factor_2d_mixed', 'cudss_solve_2d', 'cudss_solve_2d_mixed'):
        curves[name] = line(_SECONDS_PER_UNKNOWN)
    for name in ('cudss_factor_2d_device', 'cudss_factor_2d_device_mixed'):
        curves[name] = line(_DEVICE_FACTOR_RATIO * _SECONDS_PER_UNKNOWN)
    return MachineProfile('cpu', 'synthetic', '2026-01-01', '', {}, curves, dict(_SCALARS))


def _cudss_2d_checks(add, te, profile):
    """A 2D cuDSS SolverWork (sparse_direct 'linear', arpack 'eigen') prices finite, fully, and grows with n; with
    device_values its factorisations take the device curve (the host one on a profile without it)."""
    from ..core.time_profile import MachineProfile
    from ..solvers import arpack_shift_invert, sparse_direct

    def priced(module, values, role, unknowns):
        work = te.SolverWork('p', role, unknowns=unknowns, nnz=12.0 * unknowns, value_bytes=16, dimensions=2,
                             solves=3.0, factorizations=3.0, eigenpairs=6, factor_scale=3.0, solve_scale=3.0)
        return module.time_estimate(values, work).priced(profile, 'gpu')

    for back_end in sparse_direct.CUDSS_BACK_ENDS:
        for module, values, role in ((sparse_direct, {'back_end': back_end}, 'linear'),
                                     (arpack_shift_invert, {'factorization': back_end}, 'eigen')):
            small, large = priced(module, values, role, 1e4), priced(module, values, role, 1e5)
            add(f'cudss_2d_{role}_{back_end}', not small.not_estimated and not large.not_estimated
                and _priced_well(small) and _priced_well(large) and small.items
                and 0.0 < small.seconds_guess < large.seconds_guess,
                {'small': [te.item_text(item) for item in small.items], 'large_seconds': large.seconds_guess,
                 'not_estimated': list(small.not_estimated)})

    def factor_seconds(back_end, device_values, priced_on):
        work = te.SolverWork('p', 'linear', unknowns=1e5, nnz=1.2e6, value_bytes=16, dimensions=2, solves=3.0,
                             factorizations=3.0, device_values=device_values)
        estimate = sparse_direct.time_estimate({'back_end': back_end}, work).priced(priced_on, 'gpu')
        return next(item.seconds_per_unit for item in estimate.items if item.label == 'cuDSS factorisations')
    without_device = MachineProfile(profile.backend, profile.device, profile.date, profile.code_hash, profile.versions,
                                    {name: points for name, points in profile.curves.items() if '_device' not in name},
                                    dict(profile.scalars))
    for back_end in sparse_direct.CUDSS_BACK_ENDS:   # device_values: the device curve, the host one when absent
        host, device = factor_seconds(back_end, False, profile), factor_seconds(back_end, True, profile)
        fallback = factor_seconds(back_end, True, without_device)
        add(f'cudss_2d_device_values_{back_end}', math.isclose(device, _DEVICE_FACTOR_RATIO * host, rel_tol=1e-9)
            and math.isclose(fallback, host, rel_tol=1e-9), [host, device, fallback])


def _arithmetic_checks(add, te, profile, bandwidth_seconds):
    """Elementwise flops: the gpu takes the slower of bytes and arithmetic (float64 at its own curve), the cpu and a
    profile without the arith curves price the bytes only."""
    from ..core.time_profile import MachineProfile

    def priced(flops, flop_bytes, on=profile, backend='gpu'):
        return te.price(te.Elementwise(3e6, 1e6, launches=4, flops=flops, flop_bytes=flop_bytes), on, backend)
    light, heavy = priced(1e6, 4), priced(1e9, 8)
    without = MachineProfile(profile.backend, profile.device, profile.date, profile.code_hash, profile.versions,
                             {name: points for name, points in profile.curves.items() if not name.startswith('arith_')},
                             dict(profile.scalars))
    add('price_elementwise_arithmetic', math.isclose(light, bandwidth_seconds, rel_tol=_PRICE_RTOL)
        and math.isclose(heavy, 1e9 * _SECONDS_PER_KERNEL_FLOP[64], rel_tol=_PRICE_RTOL)
        and math.isclose(priced(1e9, 8, backend='cpu'), bandwidth_seconds, rel_tol=_PRICE_RTOL)
        and math.isclose(priced(1e9, 8, on=without), bandwidth_seconds, rel_tol=_PRICE_RTOL),
        [light, heavy, bandwidth_seconds])


def _priced_well(estimate):
    """Every item priced finite and >= 0, low <= guess <= high."""
    return (all(item.seconds_per_unit is not None and math.isfinite(item.seconds_per_unit)
                and item.seconds_per_unit >= 0.0 for item in estimate.items)
            and 0.0 <= estimate.seconds_low <= estimate.seconds_guess <= estimate.seconds_high
            and math.isfinite(estimate.seconds_high))


def _check_quick_profile(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    import json

    from ..core.time_profile import profile_from_dict, profile_text, run_profile
    from .schema import get_registry
    from .time_estimate import estimate_time

    checks = SubChecks()
    add = checks.add
    profile = run_profile('cpu', quick=True, save=False)
    missing = [name for name in _EXPECTED_CURVES if name not in profile.curves]
    add('curves_present', not missing, {'missing': missing, 'curves': sorted(profile.curves)})
    bad_curves = {}
    for name, points in profile.curves.items():
        sizes, seconds = [size for size, _ in points], [value for _, value in points]
        if (len(points) < 2 or any(high <= low for low, high in zip(sizes, sizes[1:]))
                or any(high < low for low, high in zip(seconds, seconds[1:]))
                or not all(math.isfinite(value) and value > 0.0 for value in seconds)):
            bad_curves[name] = [list(point) for point in points]
    add('curves_monotone', not bad_curves, bad_curves)
    add('scalars', profile.scalar('launch') > 0.0 and profile.scalar('python_call') > 0.0
        and profile.scalar('sync') == 0.0, dict(profile.scalars))

    points = profile.curves['fused']
    at_points = [profile.seconds('fused', size) for size, _ in points]
    (size_low, seconds_low), (size_high, seconds_high) = points[0], points[1]
    middle = profile.seconds('fused', math.sqrt(size_low * size_high))
    last_size, last_seconds = points[-1]
    beyond = profile.seconds('fused', 10.0 * last_size)
    add('interpolation', all(math.isclose(value, stored, rel_tol=1e-12) for value, (_, stored) in zip(at_points, points))
        and seconds_low <= middle <= seconds_high and beyond >= 10.0 * last_seconds * (1.0 - 1e-12)
        and profile.seconds('fused', 0.5 * size_low) == seconds_low,
        {'middle': middle, 'beyond': beyond, 'last': [last_size, last_seconds]})
    bandwidth = last_size / last_seconds
    add('fused_bandwidth', _BANDWIDTH_RANGE[0] <= bandwidth <= _BANDWIDTH_RANGE[1], bandwidth)
    add('dict_round_trip', profile_from_dict(json.loads(json.dumps(profile.as_dict()))) == profile)
    add('profile_text', profile.key in profile_text(profile), profile_text(profile))

    spec = _spec('heat', StudySpec(type='stationary'))
    spec.solver_defaults.backend = 'cpu'
    estimate, message = estimate_time(spec, {}, get_registry(), profile=profile)
    add('heat_priced', estimate is not None and not estimate.not_estimated and _priced_well(estimate)
        and estimate.seconds_guess > 0.0,
        message or {'seconds_guess': estimate.seconds_guess, 'not_estimated': list(estimate.not_estimated)})
    return checks.result()


def _check_recipes(results_by_name):
    """Every sub-check pass / fail plus one per template whose estimate fails; tolerance 0."""
    from ..core import time_estimate as te
    from .expressions import evaluate_parameters
    from .schema import get_registry
    from .templates import template_entries
    from .time_estimate import estimate_overheads, estimate_time

    registry = get_registry()
    profile = _synthetic_profile()
    checks = SubChecks()
    add = checks.add

    fused = te.price(te.Elementwise(3e6, 1e6, launches=4), profile, 'gpu')
    add('price_elementwise', math.isclose(fused, 4 * 1e6 * _SECONDS_PER_BYTE, rel_tol=_PRICE_RTOL), fused)
    _arithmetic_checks(add, te, profile, fused)
    real = te.price(te.HostLU(1e4), profile, 'gpu')
    complex_lu = te.price(te.HostLU(1e4, complex=True), profile, 'gpu')
    add('price_complex_lu', math.isclose(complex_lu, te.COMPLEX_LU_FACTOR * real, rel_tol=_PRICE_RTOL),
        [real, complex_lu])
    add('price_sync_cpu', te.price(te.Sync(3), profile, 'cpu') == 0.0
        and math.isclose(te.price(te.Sync(3), profile, 'gpu'), 3 * _SCALARS['sync'], rel_tol=_PRICE_RTOL))
    guess = te.TimeItem('g', 'iteration', 10.0, 'guess', limit=20.0).counts()
    bounded = te.TimeItem('b', 'iteration', 10.0, 'bounded').counts()
    add('count_spans', guess == (10.0 / te.GUESS_RANGE, 10.0, 20.0) and bounded == (10.0 / te.GUESS_RANGE, 10.0, 10.0),
        [guess, bounded])

    _time_check_checks(add, te)
    _cudss_2d_checks(add, te, profile)

    heat = _spec('heat', StudySpec(type='stationary'))
    estimate, message = estimate_time(heat, {}, registry, profile=profile)
    add('heat_stationary', estimate is not None and not estimate.not_estimated and _priced_well(estimate)
        and any(item.label.startswith("physics 'p'") for item in estimate.items),
        message or [te.item_text(item) for item in estimate.items])
    swept = heat.copy()
    swept.parameters = [ParameterSpec('power', 1.0)]
    swept.study.sweep = SweepSpec('power', [1.0, 2.0, 3.0])
    whole, _ = estimate_time(swept, {}, registry, profile=profile)
    one, _ = estimate_time(swept, {}, registry, profile=profile, per_run=True)
    add('sweep_runs', whole is not None and one is not None
        and math.isclose(whole.seconds_guess, 3.0 * one.seconds_guess, rel_tol=_PRICE_RTOL)
        and not any(item.counter for item in whole.items),
        None if whole is None or one is None else [whole.seconds_guess, one.seconds_guess])
    # a swept mesh size: every value priced on its own mesh (the sum of the two fixed-dx runs)
    mesh_swept = heat.copy()
    mesh_swept.parameters = [ParameterSpec('dx_cell', _DX)]
    mesh_swept.meshes[0].dx = 'dx_cell'
    mesh_swept.study.sweep = SweepSpec('dx_cell', [_DX, 0.5 * _DX])
    fine = heat.copy()
    fine.meshes[0].dx = 0.5 * _DX
    mesh_whole, _ = estimate_time(mesh_swept, {}, registry, profile=profile)
    fine_one, _ = estimate_time(fine, {}, registry, profile=profile)
    add('sweep_runs_own_mesh', mesh_whole is not None and fine_one is not None and estimate is not None
        and math.isclose(mesh_whole.seconds_guess, estimate.seconds_guess + fine_one.seconds_guess,
                         rel_tol=1e-6),
        None if mesh_whole is None or fine_one is None or estimate is None
        else [mesh_whole.seconds_guess, estimate.seconds_guess, fine_one.seconds_guess])
    single, sweep_overheads = (estimate_overheads(spec, {}, registry, profile=profile) for spec in (heat, swept))
    add('overheads', set(single) == {'build', 'output'} and _priced_well(single['build'])
        and _priced_well(single['output']) and single['build'].seconds_guess > 0.0
        and math.isclose(sweep_overheads['build'].seconds_guess, 3.0 * single['build'].seconds_guess,
                         rel_tol=_PRICE_RTOL)
        and math.isclose(sweep_overheads['output'].seconds_guess, single['output'].seconds_guess, rel_tol=_PRICE_RTOL),
        {name: [te.item_text(item) for item in part.items] for name, part in single.items()})

    fdtd = _spec('em_fdtd_ez', StudySpec(type='time_dependent', settings={'duration': _DURATION}))
    estimate, message = estimate_time(fdtd, {}, registry, profile=profile)
    steps = [item for item in (estimate.items if estimate is not None else ()) if item.unit == 'step']
    expected_steps = round(_DURATION / estimate.dt_limit) if estimate is not None else None
    add('fdtd_steps', len(steps) == 1 and steps[0].count == expected_steps and steps[0].count_kind == 'exact'
        and steps[0].counter == ('ticks',) and _priced_well(estimate),
        message or {'steps': [te.item_text(item) for item in steps], 'expected': expected_steps})

    failures, unestimated = {}, {}
    for entry in template_entries(registry):
        try:
            spec = entry.factory()
            estimate, message = estimate_time(spec, evaluate_parameters(spec.parameters, errors={}), registry,
                                              profile=profile)
        except Exception as exc:   # a crashing estimate is this template's failure, not a selftest crash
            failures[entry.key] = f'{type(exc).__name__}: {exc}'
            continue
        if estimate is None:
            failures[entry.key] = message
            continue
        failed = [label for label in estimate.not_estimated if 'recipe failed' in label]
        if failed or not _priced_well(estimate):
            failures[entry.key] = failed or [te.item_text(item) for item in estimate.items]
        if estimate.not_estimated:
            unestimated[entry.key] = list(estimate.not_estimated)
    add('templates_estimated', not failures, failures)
    checks['templates_not_estimated'] = {'passed': True, 'value': unestimated}   # recorded, never failing
    return checks.result()


def _time_check_checks(add, te):
    """time_check / time_check_warnings on hand-made summaries: agreement, an exact count off, a slow group, noise."""
    linear = ('solvers', 'p', 'roles', 'linear', 'counters')
    estimate = te.TimeEstimate((
        te.TimeItem('stepper', 'step', 100.0, 'exact', counter=('ticks',), seconds_per_unit=0.01),
        te.TimeItem('CG iterations', 'iteration', 50.0, 'guess', counter=linear + ('iterations',),
                    seconds_counter=linear + ('solve_seconds',), seconds_per_unit=0.02)), profile={'key': 'synthetic'})

    def summary(study_seconds, ticks, iterations, solve_seconds):
        return {'study_seconds': study_seconds, 'ticks': ticks, 'solvers': {'p': {'roles': {'linear': {'counters': {
            'iterations': iterations, 'solve_seconds': solve_seconds}}}}}}

    agree = te.time_check(estimate, summary(3.2, 100, 60, 1.2))
    add('time_check_agree', not te.time_check_warnings(agree) and math.isclose(agree['total_ratio'], 1.6)
        and math.isclose(agree['unaccounted_seconds'], 1.0), te.time_check_warnings(agree))
    extra_steps = te.time_check_warnings(te.time_check(estimate, summary(3.2, 120, 60, 1.2)))
    add('time_check_exact_count', len(extra_steps) == 1 and 'ran 120' in extra_steps[0], extra_steps)
    slow = te.time_check_warnings(te.time_check(estimate, summary(14.0, 100, 60, 12.0)))
    add('time_check_slow_group', len(slow) == 1 and 'slower' in slow[0] and 'CG iterations' in slow[0], slow)
    noise = te.time_check_warnings(te.time_check(estimate, summary(0.3, 100, 500, 0.01)))
    add('time_check_below_noise', not noise, noise)


def _triangle_spec(mirror=False, mesher='builtin'):
    """Stationary heat on a 4 x 4 um triangle mesh (h 0.5 um) with a rectangle 'core' of width w centred in x."""
    spec = _spec('heat', StudySpec(type='stationary'))
    spec.parameters = [ParameterSpec('w', 1e-6), ParameterSpec('power', 1.0)]
    spec.geometry = [GeometrySpec('core', 'rectangle', {'x0': '2e-6 - w/2', 'y0': 1e-6, 'x1': '2e-6 + w/2',
                                                        'y1': 3e-6})]
    spec.meshes = [MeshSpec(kind='triangles', mode='uniform', dx=_TRIANGLE_H, mirror_x=mirror, mesher=mesher)]
    return spec


def _check_build_estimate(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from .. import api
    from . import meshing
    from .expressions import evaluate_parameters
    from .geometry import resolve_geometry
    from .schema import get_registry
    from .time_estimate import (GMSH_HOST_BYTES_PER_NODE, MESHER_CALLS_PER_ROUND, MESHER_HOST_BYTES_PER_NODE_ROUND,
                                build_time_text, estimate_overheads, mesher_rounds)

    registry, profile = get_registry(), _synthetic_profile()
    checks = SubChecks()
    add = checks.add
    base = _triangle_spec()
    mesh_name = base.meshes[0].name
    parameters = evaluate_parameters(base.parameters)
    base_mesh = meshing.estimate_mesh(base, parameters, registry)
    rounds = mesher_rounds(base_mesh.sizing, base_mesh.dx_min)   # uniform h: 1 (the initial triangulation)
    calls_seconds = rounds * MESHER_CALLS_PER_ROUND * _SCALARS['python_call']
    bytes_per_node = MESHER_HOST_BYTES_PER_NODE_ROUND * rounds

    def build_items(spec, ending, **options):
        build = estimate_overheads(spec, evaluate_parameters(spec.parameters), registry, profile=profile,
                                   **options)['build']
        return build, [item for item in build.items if item.label.endswith(ending)]

    def node_count(spec):
        return meshing.estimate_mesh(spec, evaluate_parameters(spec.parameters), registry).n_nodes

    single, mesher = build_items(base, 'triangle mesher')
    expected = bytes_per_node * node_count(base) * _SECONDS_PER_BYTE + calls_seconds
    tagging = [item for item in single.items if item.label == f"mesh '{mesh_name}': mesh tagging"]
    setup = [item for item in single.items if item.label == f"mesh '{mesh_name}': TriangleDomain set-up"]
    add('mesh_items', len(mesher) == 1 and mesher[0].count_kind == 'guess'
        and math.isclose(mesher[0].seconds_per_unit, expected, rel_tol=_PRICE_RTOL)
        and len(tagging) == 1 and tagging[0].count == 1.0 and len(setup) == 1 and _priced_well(single),
        [item.label for item in single.items])
    _, mirrored_mesher = build_items(_triangle_spec(mirror=True), 'triangle mesher')
    half = bytes_per_node * node_count(_triangle_spec(mirror=True)) / 2 * _SECONDS_PER_BYTE + calls_seconds
    add('mirror_meshes_half', len(mirrored_mesher) == 1
        and math.isclose(mirrored_mesher[0].seconds_per_unit, half, rel_tol=_PRICE_RTOL),
        [item.seconds_per_unit for item in mirrored_mesher] + [half])
    _, gmsh_mesher = build_items(_triangle_spec(mesher='gmsh'), 'triangle mesher')
    gmsh_expected = GMSH_HOST_BYTES_PER_NODE * node_count(_triangle_spec(mesher='gmsh')) * _SECONDS_PER_BYTE
    add('gmsh_mesher', len(gmsh_mesher) == 1
        and math.isclose(gmsh_mesher[0].seconds_per_unit, gmsh_expected, rel_tol=_PRICE_RTOL),
        [item.seconds_per_unit for item in gmsh_mesher] + [gmsh_expected])

    free = base.copy()   # a parameter no mesh setting or shape reads: one mesher run, a domain set-up per value
    free.study.sweep = SweepSpec('power', [1.0, 2.0, 3.0])
    _, free_mesher = build_items(free, 'triangle mesher')
    _, free_setup = build_items(free, 'TriangleDomain set-up')
    add('sweep_meshes_once', len(free_mesher) == 1 and len(free_setup) == 3,
        [item.label for item in free_mesher + free_setup])
    shaped = base.copy()   # the core width: every value meshes
    shaped.study.sweep = SweepSpec('w', [0.5e-6, 1e-6, 1.5e-6])
    _, shaped_mesher = build_items(shaped, 'triangle mesher')
    add('geometry_sweep_meshes_each', len(shaped_mesher) == 3, [item.label for item in shaped_mesher])
    key = meshing.element_mesh_key(meshing.estimate_mesh(base, parameters, registry).sizing,
                                   meshing.element_window_size(base, parameters, 'triangles'),
                                   resolve_geometry(base, parameters))
    _, kept_mesher = build_items(free, 'triangle mesher', per_run=True, kept_mesh_keys={mesh_name: key})
    _, stale_mesher = build_items(free, 'triangle mesher', per_run=True, kept_mesh_keys={mesh_name: 'another mesh'})
    add('kept_mesh_not_meshed', not kept_mesher and len(stale_mesher) == 1,
        [item.label for item in kept_mesher + stale_mesher])

    text = build_time_text(single, [['validation + mesh / memory estimates', 0.5], ['domain', 2.0], ['physics', 1.0]],
                           {mesh_name: 'domain'})
    add('build_log_text', text.startswith('Build estimate: domain estimated ') and 'took 2 s' in text
        and 'whole build estimated' in text and 'took 3.5 s' in text, text)

    was_on = meshing.remembers_element_meshes()
    with meshing.element_meshes_kept():
        first, first_reused = meshing.element_mesh(base, parameters, registry)
        again, again_reused = meshing.element_mesh(base, dict(parameters, power=2.0), registry)
        kept_names = list(meshing.kept_element_mesh_keys())
        with meshing.element_meshes_kept():   # nested: stays on after the inner block
            pass
        still_on = meshing.remembers_element_meshes()
        wider, wider_reused = meshing.element_mesh(base, dict(parameters, w=1.5e-6), registry)
    add('mesh_cache_reuse', not first_reused and again_reused and again is first and not wider_reused
        and wider is not first and kept_names == [mesh_name] and still_on,
        {'reused': [first_reused, again_reused, wider_reused], 'kept': kept_names, 'still_on': still_on})
    add('mesh_cache_restored', meshing.remembers_element_meshes() == was_on
        and (was_on or meshing.kept_element_mesh_keys() == {}), was_on)

    record = api.check(base).get('time_estimate')   # stored profile: no record without one
    build_record = (record or {}).get('build')
    add('check_json_build', record is None or (build_record is not None and any(
        item['label'].endswith('triangle mesher') for item in build_record['items'])),
        None if build_record is None else [item['label'] for item in build_record['items']])
    return checks.result()


def _check_profiles_current(results_by_name):
    """Per available backend: a stored machine profile measured by this benchmark code and these library versions."""
    from ..core.backend import gpu_available
    from ..core.time_profile import load_profile

    checks = SubChecks()
    add = checks.add
    for backend in ['cpu'] + (['gpu'] if gpu_available() else []):
        profile, reason = load_profile(backend)
        add(f'{backend}_profile_current', profile is not None,
            reason + ': run `python -m open_sol profile`' if profile is None else profile.record())
    return checks.result()


TIME_SELFTEST_CASES = (
    SelfTestCase(
        name='time_profile_current',
        label='Run-time estimate: stored machine profiles are current',
        description='CPU (no model run): core.time_profile.load_profile per available backend (cpu, and gpu with a '
                    'CUDA device) finds time_profiles/time_profile_<backend>.json measured by this time_profile.py '
                    'and these library versions. The only selftest that judges it: every other case and example '
                    'prices its estimates on the stored profile even when out of date ("(outdated <backend> profile)"). '
                    'Fails until `python -m open_sol profile` is rerun. error = number of backends without a current '
                    'profile, tolerance 0.',
        specs=lambda: {},
        check=_check_profiles_current,
        max_wall_time=30.0),
    SelfTestCase(
        name='time_profile_quick_cpu',
        label='Run-time estimate: quick cpu machine profile',
        description='CPU (no model run): core.time_profile.run_profile("cpu", quick=True, save=False), so the stored '
                    'profile is kept. Checks: the fused / dot / spmv / gemm / host stream / host SuperLU 2D, 2D '
                    'complex and 3D curves exist with increasing sizes and non-decreasing positive seconds; launch '
                    'and python_call > 0, sync 0 on the cpu; seconds() returns the stored points, lies between '
                    'neighbours at a geometric midpoint, grows at least linearly past the last point and is the '
                    'first point below it; the fused bandwidth is 0.1 GB/s - 1 TB/s; the JSON round trip; a '
                    'stationary heat model on a 4 x 4 um, 50 nm mesh prices finite with nothing left unestimated. '
                    'error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_quick_profile,
        max_wall_time=300.0),
    SelfTestCase(
        name='time_estimate_recipes',
        label='Run-time estimate: pricing, study counts, time_check and every template',
        description='CPU (no model run) on a synthetic profile with linear curves (1 ns / byte, 0.1 ns / flop, '
                    '0.1 us / SuperLU unknown): an Elementwise of 4 launches costs 4 x its bytes / 4 x rate, a '
                    'complex HostLU without a complex curve COMPLEX_LU_FACTOR x the real one, Sync nothing on the cpu; '
                    'guess / bounded count spans; time_check on hand-made summaries (agreement: no warning, total '
                    'ratio 1.6; an exact step count of 120 for 100: one warning; a solve-seconds group 10x slow: one '
                    'warning; groups under MIN_CHECK_SECONDS: none); a 2D cuDSS SolverWork (sparse_direct linear and '
                    'arpack eigen, cudss and cudss_mixed, on the synthetic cudss_* curves) fully priced and growing '
                    'from 1e4 to 1e5 unknowns, its device_values factorisations on the device curve (0.25x) and on '
                    'the host curve when the profile lacks it; stationary heat on a 4 x 4 um, 50 nm mesh fully '
                    'estimated, a 3-value sweep 3x one run with the counters cleared; its build / output overheads '
                    'priced, the sweep building 3x and writing once; em_fdtd_ez over 0.1 ps: one '
                    'exact step item of round(duration / dt) counted by ticks; every template estimates without an '
                    'exception or a failing recipe (unestimated work is recorded only). error = number of failed '
                    'sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_recipes,
        max_wall_time=300.0),
    SelfTestCase(
        name='build_estimate_triangles',
        label='Build-time estimate: triangle mesher items and the in-process mesh cache',
        description='CPU (no model run) on the synthetic profile: stationary heat on a 4 x 4 um triangle mesh (h 0.5 '
                    'um, a centred rectangle of width w). estimate_overheads build items: one guessed triangle mesher '
                    'item priced MESHER_HOST_BYTES_PER_NODE_ROUND x nodes x mesher_rounds + mesher_rounds x '
                    'MESHER_CALLS_PER_ROUND calls, one mesh tagging item over 1 object, one TriangleDomain set-up; mirror_x meshes half the '
                    'nodes, mesher gmsh priced by GMSH_HOST_BYTES_PER_NODE; a 3-value sweep of an unused parameter '
                    'meshes once (3 set-ups), a sweep of w meshes 3 times; per_run with the kept element_mesh_key '
                    'has no mesher item, with another key one; the "Build estimate" log text. meshing.'
                    'element_meshes_kept: the same mesh object reused for a changed unused parameter, a new mesh for '
                    'a changed w, nested blocks keep it on, the previous state restored after; api.check time_estimate '
                    'build lists the mesher (when a stored profile exists). error = number of failed sub-checks, '
                    'tolerance 0. Not yet confirmed by a run.',
        specs=lambda: {},
        check=_check_build_estimate,
        max_wall_time=120.0),
)
