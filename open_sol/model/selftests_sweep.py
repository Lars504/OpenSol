"""Sweep self-tests: the absorbing strip of coupling/selftests as a coupled frequency_domain run against its stationary
twin, a two-power sweep of it (derived series, kept fields), the warm-started load sweep of a nonlinear tetrahedral
string and CPU checks of the sweep overlays and the kept-snapshot round trip of results_io; registered by
study.register."""
import math
import tempfile

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import relative_error
from .spec import (DerivedValueSpec, DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, ParameterSpec,
                   PhysicsSpec, SolverDefaultsSpec, StudySpec, SweepSpec)

_STATIONARY = 'stationary'
_FREQUENCY_DOMAIN = 'frequency_domain'
_SWEEP = 'sweep'
_POWERS = (1.0, 2.0)                  # W/m of the mode source per sweep value (a linear problem: dndT = 0)
_WAVELENGTH_STEP = 1.01               # second study wavelength = 1.01 x the strip wavelength
_TWIN_TOLERANCE = 1e-5                # relative (rad for the phase): the same Gauss-Seidel sequence, float32 noise
_RATIO_TOLERANCE = 1e-4               # relative, power ratios of the linear sweep (float32 T: ~1e-6 of a 33 K rise)
# sweep_nonlinear_string_warm_start: a 20 um x 0.2 x 0.2 um Si3N4 string (sigma0 1 GPa, fixed ends, P2 builtin
# tetrahedra of 0.2 um, geometric nonlinearity) under a uniform f_z = q0; at the largest q0 the sag ~ 0.2 um and the
# stretching stiffens it by ~7 %. Three values, warm_start on / off.
_STRING_SIZE = (20e-6, 0.2e-6, 0.2e-6)
_STRING_LOADS = (2e12, 3e12, 4e12)    # N/m^3
_WARM_RUNS = ('warm', 'cold')
_WARM_DERIVED_TOLERANCE = 1e-6        # relative: float32 per-element outputs of solves equal to the Newton tolerance
# sweep_reuse_build_parity: that string, x_hi pulled by u_x = pull (a parameter depending on q0) and a top traction
# t_z = 0.25 q0 h besides the body force (every follower kind: body force, face load, held value); reuse on / off
_REUSE_RUNS = ('reuse', 'rebuild')
_PULL_PER_LOAD = 1e-21                # m per (N/m^3): pull = 2 .. 4 nm (axial strain 1e-4 .. 2e-4)
_REUSE_DERIVED_TOLERANCE = 1e-6       # relative: equal Newton solves from other starts, float32 element outputs


# ---- specs (built from the strip of coupling/selftests, a sibling selftests module) ----
def _feedback_specs():
    """'stationary' and 'frequency_domain' (its one wavelength, coupled: em_heat_joule reads the FDFD physics) of the
    heated strip (dndT 1e-3 /K)."""
    from ..coupling.selftests import _DNDT, _WAVELENGTH, _strip_spec
    coupled = _strip_spec(_FREQUENCY_DOMAIN, _DNDT)
    coupled.study.type = _FREQUENCY_DOMAIN
    coupled.study.settings['wavelengths'] = [_WAVELENGTH]
    return {_STATIONARY: _strip_spec(_STATIONARY, _DNDT), _FREQUENCY_DOMAIN: coupled}


def _power_sweep_specs():
    """The strip with dndT 0 at two wavelengths (coupled frequency_domain), mode-source power P_in swept over _POWERS
    with keep_fields."""
    from ..coupling.selftests import _WAVELENGTH, _strip_spec
    spec = _strip_spec('power sweep', 0.0)
    spec.parameters.append(ParameterSpec('P_in', repr(_POWERS[0])))
    spec.physics_by_name('em').feature('source').params['power'] = 'P_in'
    spec.study.type = _FREQUENCY_DOMAIN
    spec.study.settings['wavelengths'] = [_WAVELENGTH, _WAVELENGTH * _WAVELENGTH_STEP]
    spec.study.sweep = SweepSpec('P_in', list(_POWERS), keep_fields=True)
    return {_SWEEP: spec}


def _string_sweep_specs():
    """The nonlinear tetrahedral string swept over _STRING_LOADS, stationary study with warm_start on ('warm') and off
    ('cold')."""
    specs = {}
    size_x, size_y, size_z = _STRING_SIZE
    for run in _WARM_RUNS:
        spec = ModelSpec(
            title=f'Self-test: nonlinear tetrahedral string load sweep (warm start {run})',
            parameters=[ParameterSpec('q0', repr(_STRING_LOADS[0]))],
            domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
            materials=[MaterialSpec('string', properties={'youngs_modulus': 250e9, 'poisson_ratio': 0.23,
                                                          'initial_stress_xx': 1e9}, selection=['all'])],
            physics=[PhysicsSpec('solid', 'solid_mechanics_3d',
                                 settings={'discretization': 'quadratic', 'geometric_nonlinearity': True,
                                           'nonlinear_tolerance': 1e-10, 'nonlinear_max_iterations': 30},
                                 features=[FeatureSpec('ends', 'fixed', ['x_lo', 'x_hi']),
                                           FeatureSpec('load', 'body_force', ['all'],
                                                       {'f_x': 0.0, 'f_y': 0.0, 'f_z': 'q0'})])],
            meshes=[MeshSpec(name='tetrahedra', kind='tetrahedra', mode='uniform', dx=size_y, mesher='builtin')],
            solver_defaults=SolverDefaultsSpec(backend='cpu'),
            derived_values=[DerivedValueSpec('u_z_mean', 'u_z', [], 'mean'),
                            DerivedValueSpec('u_z_max', 'u_z', [], 'max')],
            study=StudySpec(type=_STATIONARY, settings={'warm_start': run == 'warm'}))
        spec.study.sweep = SweepSpec('q0', list(_STRING_LOADS))
        specs[run] = spec
    return specs


def _reuse_string_spec(reuse):
    """The nonlinear string of _string_sweep_specs (cold) with x_hi pulled by u_x = pull = _PULL_PER_LOAD q0 and a z_hi
    traction t_z = 0.25 q0 h, swept over _STRING_LOADS with reuse_build on or off."""
    spec = _string_sweep_specs()['cold']
    spec.title = f"Self-test: reused sweep build of a nonlinear tetrahedral string (reuse_build {reuse})"
    height = _STRING_SIZE[2]
    spec.parameters += [ParameterSpec('pull', f'{_PULL_PER_LOAD!r}*q0'), ParameterSpec('q_top', f'0.25*q0*{height!r}')]
    solid = spec.physics_by_name('solid')
    solid.features = [FeatureSpec('end_lo', 'fixed', ['x_lo']),
                      FeatureSpec('end_hi', 'displacement', ['x_hi'], {'u_x': 'pull', 'u_y': 0.0, 'u_z': 0.0}),
                      FeatureSpec('top', 'boundary_load', ['z_hi'], {'t_x': 0.0, 't_y': 0.0, 't_z': 'q_top'}),
                      FeatureSpec('load', 'body_force', ['all'], {'f_x': 0.0, 'f_y': 0.0, 'f_z': 'q0'})]
    spec.derived_values.append(DerivedValueSpec('u_x_mean', 'u_x', [], 'mean'))
    spec.study.sweep = SweepSpec('q0', list(_STRING_LOADS), reuse_build=reuse)
    return spec


def _reuse_sweep_specs():
    """The pulled, top-loaded string sweep with reuse_build on ('reuse') and off ('rebuild')."""
    return {run: _reuse_string_spec(run == 'reuse') for run in _REUSE_RUNS}


# ---- checks ----
def _phase_difference(results):
    """phase_positive at mode_down minus mode_up [rad] of the single solved wavelength."""
    phases = [float(np.asarray(results.result('em', feature)['series']['phase_positive'], dtype=np.float64)[-1])
              for feature in ('mode_down', 'mode_up')]
    return math.atan2(math.sin(phases[0] - phases[1]), math.cos(phases[0] - phases[1]))


def _check_feedback(results_by_name):
    """The coupled frequency_domain strip reproduces the stationary one: P_absorbed, T_max, the mode-monitor phase
    difference and the fixed-point iteration count; one wavelength fixed point, converged."""
    stationary, coupled = results_by_name[_STATIONARY], results_by_name[_FREQUENCY_DOMAIN]
    summaries = {name: results.summary() for name, results in results_by_name.items()}
    errors = {name: relative_error(float(summaries[_FREQUENCY_DOMAIN]['derived'][name]),
                              float(summaries[_STATIONARY]['derived'][name])) for name in ('P_absorbed', 'T_max')}
    phase_stationary, phase_coupled = _phase_difference(stationary), _phase_difference(coupled)
    errors['phase'] = abs(phase_coupled - phase_stationary)
    failed = []
    records = summaries[_FREQUENCY_DOMAIN].get('wavelength_fixed_points') or []
    if len(records) != 1 or not records[0]['converged']:
        failed.append(f'wavelength_fixed_points {records!r}: expected one converged record')
    iterations = {name: (summary.get('fixed_point') or {}).get('iterations') for name, summary in summaries.items()}
    if iterations[_STATIONARY] != iterations[_FREQUENCY_DOMAIN]:
        failed.append(f'fixed-point iterations differ: {iterations}')
    error = max(errors.values()) / _TWIN_TOLERANCE + len(failed)   # NaN propagates
    return {'measured': phase_coupled, 'expected': phase_stationary, 'error': error, 'tolerance': 1.0,
            'details': {'errors': errors, 'iterations': iterations, 'failed_checks': failed,
                        'derived': {name: summary['derived'] for name, summary in summaries.items()},
                        'warnings': {name: list(summary['warnings']) for name, summary in summaries.items()}}}


def _check_power_sweep(results_by_name):
    """Linear two-power sweep: P_absorbed and the T_max rise double, one result entry and a two-point derived series
    per value (its last point = the value's derived value), two kept snapshots per value whose T rise doubles."""
    from ..coupling.selftests import _T0
    results = results_by_name[_SWEEP]
    sweep = results.summary()['sweep']
    ratio = _POWERS[1] / _POWERS[0]
    derived = {name: [float(value) for value in values] for name, values in sweep['derived'].items()}
    rises = [value - _T0 for value in derived['T_max']]
    errors = {'P_absorbed_ratio': relative_error(derived['P_absorbed'][1] / derived['P_absorbed'][0], ratio),
              'T_rise_ratio': relative_error(rises[1] / rises[0], ratio)}
    failed = []
    if len(sweep['results']) != len(_POWERS) or not all('em.flux_up' in entry for entry in sweep['results']):
        failed.append('the sweep results do not hold em.flux_up once per value')
    series = sweep.get('derived_series') or []
    for index, xy in enumerate(series):
        if not xy or len(xy['x']) != 2:
            failed.append(f'value {index}: derived series {xy!r} has not 2 points')
            continue
        errors[f'series_last_{index}'] = relative_error(float(xy['series']['T_max'][-1]), derived['T_max'][index])
    if len(series) != len(_POWERS):
        failed.append(f'{len(series)} derived series for {len(_POWERS)} values')
    if [len(times) for times in sweep.get('snapshot_times') or []] != [2] * len(_POWERS):
        failed.append(f"kept snapshot counts {sweep.get('snapshot_times')!r}: expected 2 per value")
    else:
        field_rises = [float(np.nanmax(np.asarray(results.sweep_field(index, 'heat.T').values, dtype=np.float64)))
                       - _T0 for index in range(len(_POWERS))]
        errors['field_rise_ratio'] = relative_error(field_rises[1] / field_rises[0], ratio)
    error = max(errors.values()) / _RATIO_TOLERANCE + len(failed)
    return {'measured': rises[1] / rises[0], 'expected': ratio, 'error': error, 'tolerance': 1.0,
            'details': {'errors': errors, 'failed_checks': failed, 'derived': derived,
                        'converged': sweep.get('converged'), 'warnings': list(results.summary()['warnings'])}}


def _check_string_warm_start(results_by_name):
    """Warm vs cold string sweep: every derived value agrees, the last value (seeded by the extrapolation of the first
    two) needs fewer Newton iterations, both converged."""
    summaries = {run: results_by_name[run].summary() for run in _WARM_RUNS}
    derived = {run: {name: [float(value) for value in values] for name, values in summary['sweep']['derived'].items()}
               for run, summary in summaries.items()}
    errors, failed = {}, []
    for name, cold_values in derived['cold'].items():
        warm_values = derived['warm'][name]
        if len(warm_values) != len(_STRING_LOADS) or len(cold_values) != len(_STRING_LOADS):
            failed.append(f'{name}: {len(warm_values)} / {len(cold_values)} values for {len(_STRING_LOADS)} loads')
            continue
        errors[name] = max(relative_error(warm, cold) for warm, cold in zip(warm_values, cold_values))
    iterations = {run: ((summary.get('solvers') or {}).get('solid') or {}).get('counters', {}).get(
        'nonlinear_iterations') for run, summary in summaries.items()}
    if not all(isinstance(count, int) for count in iterations.values()):
        failed.append(f'no nonlinear_iterations counter in the last values: {iterations}')
    elif not iterations['warm'] < iterations['cold']:
        failed.append(f'the extrapolated warm start took {iterations["warm"]} Newton iterations, cold '
                      f'{iterations["cold"]}')
    for run, summary in summaries.items():
        if summary['sweep'].get('failed'):
            failed.append(f"{run}: failed values {summary['sweep']['failed']}")
    error = max(errors.values(), default=math.nan) / _WARM_DERIVED_TOLERANCE + len(failed)
    return {'measured': max(errors.values(), default=math.nan), 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'errors': errors, 'failed_checks': failed, 'iterations_last_value': iterations,
                        'derived': derived, 'warnings': {run: list(summary['warnings'])
                                                         for run, summary in summaries.items()}}}


def _check_reuse_parity(results_by_name):
    """Reused vs rebuilt string sweep: every derived value agrees, the sweep table counts 1 vs 3 builds, no value
    fails, and the derived values change between values (the followers re-set the loads)."""
    summaries = {run: results_by_name[run].summary() for run in _REUSE_RUNS}
    derived = {run: {name: [float(value) for value in values] for name, values in summary['sweep']['derived'].items()}
               for run, summary in summaries.items()}
    errors, failed = {}, []
    for name, rebuilt_values in derived['rebuild'].items():
        reused_values = derived['reuse'][name]
        if len(reused_values) != len(_STRING_LOADS) or len(rebuilt_values) != len(_STRING_LOADS):
            failed.append(f'{name}: {len(reused_values)} / {len(rebuilt_values)} values for {len(_STRING_LOADS)} loads')
            continue
        errors[name] = max(relative_error(reused, rebuilt) for reused, rebuilt in zip(reused_values, rebuilt_values))
        if len(set(reused_values)) != len(reused_values):
            failed.append(f'{name}: the reused build repeats a value {reused_values} (a load did not follow)')
    builds = {run: summary['sweep'].get('builds') for run, summary in summaries.items()}
    if builds != {'reuse': 1, 'rebuild': len(_STRING_LOADS)}:
        failed.append(f'builds {builds}: expected 1 reused, {len(_STRING_LOADS)} rebuilt')
    for run, summary in summaries.items():
        if summary['sweep'].get('failed'):
            failed.append(f"{run}: failed values {summary['sweep']['failed']}")
    error = max(errors.values(), default=math.nan) / _REUSE_DERIVED_TOLERANCE + len(failed)
    return {'measured': max(errors.values(), default=math.nan), 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'errors': errors, 'failed_checks': failed, 'builds': builds, 'derived': derived,
                        'warnings': {run: list(summary['warnings']) for run, summary in summaries.items()}}}


def _check_reuse_blockers(_results_by_name):
    """CPU: sweep_reuse.reuse_issues lists a swept parameter (or one depending on it) in an x-dependent material value,
    an unflagged physics setting, a mesh size, an x-dependent following load and a time_dependent study at their node
    paths, and nothing for loads, held values, a uniform (repainted) material value and derived values of parameters
    only; live_parameter_names follows dependencies transitively; validate reports the issues as errors."""
    from .schema import get_registry
    from .sweep_reuse import live_parameter_names, reuse_issues
    from .validate import validate
    registry = get_registry()

    def string(change=None):
        spec = _reuse_string_spec(True)
        spec.parameters.append(ParameterSpec('q_unrelated', '1.0'))
        spec.derived_values.append(DerivedValueSpec('q_scaled', 'q_top*u_z', [], 'mean'))
        if change is not None:
            change(spec)
        return spec

    def set_body_force(value):
        return lambda spec: spec.physics_by_name('solid').feature('load').params.__setitem__('f_z', value)

    cases = {
        'loads only': (None, set()),
        'material': (lambda spec: spec.materials[0].properties.__setitem__('youngs_modulus',
                                                                            '250e9 + 0*pull*x'),
                     {('material', 'string')}),
        'uniform material': (lambda spec: spec.materials[0].properties.__setitem__('youngs_modulus',
                                                                                    '250e9 + 0*pull'), set()),
        'physics setting': (lambda spec: spec.physics_by_name('solid').settings.__setitem__(
            'nonlinear_tolerance', '1e-10 + 0*q_top'), {('physics', 'solid')}),
        'mesh size': (lambda spec: setattr(spec.meshes[0], 'dx', f'{_STRING_SIZE[1]!r} + 0*q0'),
                      {('mesh', 'tetrahedra')}),
        'x-dependent load': (set_body_force('q0*x/1e-6'), {('feature', 'solid', 'load')}),
        'unrelated parameter': (lambda spec: spec.materials[0].properties.__setitem__(
            'youngs_modulus', '250e9*q_unrelated'), set()),
        'time_dependent study': (lambda spec: setattr(spec.study, 'type', 'time_dependent'), {('study',)}),
    }
    failed = []
    for name, (change, expected) in cases.items():
        spec = string(change)
        paths = {tuple(path) for path, _ in reuse_issues(spec, registry)}
        if paths != expected:
            failed.append(f'{name}: reuse issues at {sorted(paths)}, expected {sorted(expected)}')
        if name == 'material':
            errors = {tuple(issue.path) for issue in validate(spec, registry) if issue.level == 'error'
                      and 'reuse_build' in issue.message}
            if ('material', 'string') not in errors:
                failed.append(f'validate did not report the material reuse issue: {sorted(errors)}')
    live = live_parameter_names(string(), 'q0')
    if live != {'q0', 'pull', 'q_top'}:
        failed.append(f'live parameters {sorted(live)}, expected q0, pull, q_top')
    return {'measured': len(failed), 'expected': 0, 'error': float(len(failed)), 'tolerance': 0.0,
            'details': {'failed_checks': failed}}


def _check_overlay_helpers(_results_by_name):
    """CPU: analytic.sweep_overlay_xy / sweep_value_curve on synthetic flux, timeseries and xy results (own
    abscissas, series choice, iteration axis), and the results_io round trip of kept sweep snapshots and a NaN in a
    derived series."""
    from ..core.renderer import FieldInfo
    from ..io.results_io import ResultsData, load_results, save_results
    from .analytic import sweep_overlay_xy, sweep_value_curve
    from .session import FieldData, Snapshot
    failed = []
    wavelengths = [np.array([1.5e-6, 1.6e-6]), np.array([1.5e-6, 1.55e-6, 1.6e-6])]
    spectra = [{'kind': 'flux_spectrum', 'wavelength': grid, 'power': (index + 1.0) * np.ones(grid.size),
                'unit': 'W/m'} for index, grid in enumerate(wavelengths)]
    overlay = sweep_overlay_xy('P', [1.0, 2.0, 3.0], spectra + [None])   # a failed third value has no curve
    if overlay is None or list(overlay['series']) != ['P = 1', 'P = 2']:
        failed.append(f"flux overlay series {None if overlay is None else list(overlay['series'])!r}")
    elif list(overlay.get('series_x', {})) != ['P = 2'] or not np.array_equal(overlay['series_x']['P = 2'],
                                                                               wavelengths[1]):
        failed.append('the second spectrum lost its own wavelengths (series_x)')
    repeated = sweep_overlay_xy('P', [1.0, 1.0], spectra)
    if repeated is None or list(repeated['series']) != ['P = 1', 'P = 1 (2)']:
        failed.append(f"repeated value names {None if repeated is None else list(repeated['series'])!r}")
    iteration = sweep_value_curve({'kind': 'timeseries', 't': [0.0, 0.0, 0.0], 'values': [1.0, 2.0, 3.0]})
    if iteration is None or iteration['x_label'] != 'Iteration' or list(iteration['x']) != [1.0, 2.0, 3.0]:
        failed.append(f'iteration axis {iteration!r}')
    xy = {'kind': 'xy', 'x': [0.0, 1.0], 'series': {'a': [1.0, 2.0], 'b': [3.0, 4.0]}, 'series_units': {'b': 'W'}}
    picked = sweep_value_curve(xy, 'b')
    if picked is None or list(picked['y']) != [3.0, 4.0] or picked['unit'] != 'W':
        failed.append(f'xy series choice {picked!r}')
    if sweep_value_curve(xy, 'missing') is not None or sweep_value_curve(spectra[0], 'b') is not None:
        failed.append('a missing series did not give None')
    info = FieldInfo('T', 'Temperature', 'K', primary=True)
    snapshots = [[Snapshot(time=0.0, label=f'lambda = {1500 + 10 * k} nm', fields={'heat.T': FieldData(
        'heat', 'T', np.full((3, 2), 300.0 + 10 * value + k, dtype=np.float32), (0.0, 3e-6, 0.0, 2e-6), info, 0.0,
        None)}) for k in range(2)] for value in range(2)]
    series = {'kind': 'xy', 'x': [1.5e-6, 1.6e-6], 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Derived value',
              'series': {'T_max': [1.0, math.nan]}, 'series_units': {'T_max': ''}}
    data = ResultsData(summary={'status': 'completed', 'derived_series': series,
                                'sweep': {'parameter': 'P', 'values': [1.0, 2.0], 'derived_series': [series, None]}},
                       sweep_snapshots=snapshots)
    with tempfile.TemporaryDirectory() as directory:
        save_results(directory, data)
        loaded = load_results(directory)
    if [[snapshot.label for snapshot in value] for value in loaded.sweep_snapshots] != \
            [[snapshot.label for snapshot in value] for value in snapshots]:
        failed.append('kept sweep snapshot labels did not round-trip')
    elif not all(np.array_equal(kept.fields['heat.T'].values, original.fields['heat.T'].values)
                 for kept_value, original_value in zip(loaded.sweep_snapshots, snapshots)
                 for kept, original in zip(kept_value, original_value)):
        failed.append('kept sweep snapshot values did not round-trip')
    loaded_series = loaded.summary['sweep']['derived_series'][0]['series']['T_max']
    if not (isinstance(loaded_series[1], float) and math.isnan(loaded_series[1])):
        failed.append(f'derived series NaN came back as {loaded_series[1]!r}')
    return {'measured': len(failed), 'expected': 0, 'error': float(len(failed)), 'tolerance': 0.0,
            'details': {'failed_checks': failed}}


def _check_range_spacing(_results_by_name):
    """CPU: SweepSpec linear / log / descending / one-point ranges through sweep_values_of (exact end points), and a
    range sweep's JSON round trip."""
    from .expressions import evaluate_parameters
    from .spec import ModelSpec
    from .study import sweep_values_of
    parameters = [ParameterSpec('a_lat', '350*nm'), ParameterSpec('L_def', '1.5*a_lat')]
    lattice = evaluate_parameters(parameters)['a_lat']
    cases = {'linear': (SweepSpec('L_def', spacing='linear', lower='1.3*a_lat', upper='1.7*a_lat', count=5),
                        np.linspace(1.3 * lattice, 1.7 * lattice, 5)),
             'log': (SweepSpec('L_def', spacing='log', lower=1e-3, upper=10.0, count=5), np.geomspace(1e-3, 10.0, 5)),
             'descending': (SweepSpec('L_def', spacing='linear', lower=4, upper=1, count=4), np.array([4., 3., 2., 1.])),
             'one_point': (SweepSpec('L_def', spacing='log', lower='a_lat', upper=1.0, count=1), np.array([lattice])),
             'list': (SweepSpec('L_def', ['a_lat', 2.0], spacing='list', lower=0.0, upper=1.0), np.array([lattice, 2.]))}
    errors, failed = {}, []
    for name, (sweep, expected) in cases.items():
        spec = ModelSpec(parameters=list(parameters))
        spec.study.sweep = sweep
        swept = np.asarray(sweep_values_of(spec)[1], dtype=np.float64)
        if swept.shape != expected.shape:
            failed.append(f'{name}: {swept.size} values, expected {expected.size}')
            continue
        errors[name] = float(np.max(np.abs(swept / expected - 1.0)))
        if swept[0] != expected[0] or swept[-1] != expected[-1]:
            failed.append(f'{name}: end points {swept[0]!r}, {swept[-1]!r} are not exact')
    spec = ModelSpec(parameters=list(parameters))
    spec.study.sweep = cases['linear'][0]
    loaded = ModelSpec.from_dict(spec.to_dict()).study.sweep
    if loaded.point_values() != spec.study.sweep.point_values():
        failed.append(f'range sweep did not round-trip: {loaded!r}')
    error = max(errors.values(), default=0.0) / 1e-14 + len(failed)
    return {'measured': max(errors.values(), default=math.nan), 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'errors': errors, 'failed_checks': failed}}


SWEEP_SELFTEST_CASES = (
    SelfTestCase(
        name='frequency_domain_feedback_strip',
        label='Coupled frequency_domain: the absorbing strip at one wavelength reproduces its stationary twin',
        description='The heated absorbing strip of em_heat_thermo_optic_strip (em_fdfd_ez + heat, em_heat_joule + '
                    'heat_em_thermo_optic, dndT = 1e-3 /K, tolerance 2e-6) run twice: a stationary study, and a '
                    'frequency_domain study with the wavelength list [1.55 um] (the coupling reads the FDFD physics, so '
                    'the wavelength is a coupled stationary fixed point of every physics). Both run the same '
                    'Gauss-Seidel sequence (the study omega equals the wavelength setting), so P_absorbed, T_max and '
                    'the mode-monitor phase difference (mode_down - mode_up) must agree to float32 noise: error = the '
                    'largest relative difference (rad for the phase) / 1e-5 + the number of failed checks (one '
                    'converged wavelength_fixed_points record, equal fixed-point iteration counts), tolerance 1.',
        specs=_feedback_specs, check=_check_feedback),
    SelfTestCase(
        name='sweep_power_strip',
        label='Parameter sweep: two source powers of the coupled strip (derived series, kept fields)',
        description='The absorbing strip with dndT = 0 (a linear problem) in a coupled frequency_domain study at 1.55 '
                    'and 1.5655 um, its mode-source power P_in swept over [1, 2] W/m with keep_fields. P_absorbed and '
                    'the T_max rise (T_max - T0) must double (relative 1e-4: the float32 T resolves ~1e-6 of the 33 K '
                    'rise), likewise the largest rise of the kept final heat.T snapshot (Results.sweep_field); every '
                    'value has an em.flux_up result, a two-point derived series whose last T_max equals its derived '
                    'value (the last wavelength is the final state) and two kept snapshots. error = the largest '
                    'relative error / 1e-4 + the number of failed checks, tolerance 1.',
        specs=_power_sweep_specs, check=_check_power_sweep),
    SelfTestCase(
        name='sweep_nonlinear_string_warm_start',
        label='Parameter sweep: warm start of a geometrically nonlinear tetrahedral string (secant extrapolation)',
        description='A 20 um x 0.2 x 0.2 um Si3N4 string (sigma0 1 GPa, Fixed ends, P2 builtin tetrahedra of 0.2 um, '
                    'geometric_nonlinearity, Newton tolerance 1e-10, cpu) under a uniform body force f_z = q0 swept '
                    f'over {list(_STRING_LOADS)} N/m^3 in a stationary study, warm_start on and off: the mean and max '
                    f'u_z of every value agree within {_WARM_DERIVED_TOLERANCE:g} relative (float32 outputs), the last '
                    'value (seeded by the displacement extrapolated from the first two) takes fewer Newton iterations '
                    'than its cold solve (summary solvers counters), no value fails. error = the largest relative '
                    'difference / tolerance + the number of failed checks, tolerance 1 (first estimate, not yet run).',
        specs=_string_sweep_specs, check=_check_string_warm_start),
    SelfTestCase(
        name='sweep_reuse_build_parity',
        label='Parameter sweep: a reused build (reuse_build) of a nonlinear tetrahedral string equals the rebuilt sweep',
        description='The string of sweep_nonlinear_string_warm_start with x_lo Fixed, x_hi pulled by a Displacement u_x '
                    f'= pull = {_PULL_PER_LOAD:g} q0 (a parameter depending on the swept q0), a z_hi Boundary load t_z = '
                    f'0.25 q0 h and the body force f_z = q0, swept over {list(_STRING_LOADS)} N/m^3 (stationary, cold) '
                    'with reuse_build on and off: every derived value (mean / max u_z, mean u_x) agrees within '
                    f'{_REUSE_DERIVED_TOLERANCE:g} relative (Newton to 1e-10 from other starts, float32 outputs) and '
                    'changes from value to value, the sweep table counts 1 and 3 builds, no value fails. error = the '
                    'largest relative difference / tolerance + the number of failed checks, tolerance 1 (first '
                    'estimate, not yet run).',
        specs=_reuse_sweep_specs, check=_check_reuse_parity),
    SelfTestCase(
        name='sweep_reuse_blockers',
        label='Reused sweep builds: what the swept parameter may reach (CPU)',
        description='sweep_reuse.reuse_issues on the reuse_build string: loads, held values, a uniform Young\'s '
                    'modulus (repainted) and derived values of the swept q0 and of parameters depending on it give no '
                    'issue; q0 or a dependent in an x-dependent material property, an unflagged physics setting, the '
                    'mesh size, an x-dependent body force and a time_dependent study give one issue at the node path '
                    '(material, physics, mesh, feature, study); a parameter not '
                    'depending on q0 gives none; live_parameter_names = {q0, pull, q_top}; validate reports the '
                    'material case as an error. error = the number of failed checks, tolerance 0.',
        specs=lambda: {}, check=_check_reuse_blockers),
    SelfTestCase(
        name='sweep_overlay_helpers',
        label='Sweep overlays and kept-snapshot round trip (CPU)',
        description='analytic.sweep_overlay_xy of two flux spectra on different wavelength grids (series "P = 1", "P '
                    '= 2", the second on its own series_x; a repeated value gets " (2)"), sweep_value_curve of a '
                    'stationary timeseries (1-based iteration axis) and of an xy series choice (a missing series '
                    'gives None), and results_io save_results / load_results of two kept snapshots per sweep value '
                    '(labels and values exact) and a NaN in a summary derived series (back as a float NaN). error = '
                    'the number of failed checks, tolerance 0.',
        specs=lambda: {}, check=_check_overlay_helpers),
    SelfTestCase(
        name='sweep_range_spacing',
        label='Sweep ranges: linear, log, descending and one-point spacing (CPU)',
        description='SweepSpec spacing linear (1.3 a .. 1.7 a, 5 points, parameter expressions), log (1e-3 .. 10, 5 '
                    'points), a descending linear range (4 .. 1), a one-point range (the lower bound) and a custom '
                    'list, evaluated by study.sweep_values_of against numpy linspace / geomspace (end points exact), '
                    'plus the JSON round trip of a range sweep. error = the largest relative error / 1e-14 + the '
                    'number of failed checks, tolerance 1.',
        specs=lambda: {}, check=_check_range_spacing),
)
