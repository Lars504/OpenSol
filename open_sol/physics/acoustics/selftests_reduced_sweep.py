"""Self-tests of pressure_acoustics frequency sweeps with sweep_method adaptive (experiment plan A3 / A4): the plane-wave
duct of selftests_harmonic driven by a vibrating end ((j omega) loads: moment matching) and by a held pressure under a
background plane wave (non-polynomial load and held offsets: snapshots), each against its direct sweep. The same two
drives on a batched_direct 'linear' node (fem.harmonic plan_batched) against the plane-wave closed form."""
import math

import numpy as np

from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import solver_role
from ...model.spec import DomainSpec, FeatureSpec, MeshSpec, SolverSpec
from .selftests_harmonic import (_AIR_DENSITY, _AIR_SPEED, _DUCT_ELEMENT, _DUCT_LENGTH, _DUCT_PRESSURE, _DUCT_PROBE_X,
                                 _DUCT_VELOCITY, _DUCT_WIDTH, PHYSICS, _graded, _model, _probes, _wavenumber)

_SWEEP_HZ = (300.0, 900.0, 61)
_SWEEP_ROM_TOLERANCE = 1e-10
_SWEEP_TOLERANCE = 1e-6            # relative to the peak |p| of the direct sweep (first estimate)
_SWEEP_MAX_FACTORIZATIONS = 41     # rom_max_points + the verification
_BACKGROUND_AMPLITUDE = 0.5        # Pa: a +x plane wave (the total field does not depend on it)
_DRIVES = ('velocity', 'background')


def _sweep_spec(drive, method):
    middle = 0.5 * _DUCT_WIDTH
    geometry, probes = _probes([(x, middle) for x in _DUCT_PROBE_X])
    features = [FeatureSpec('end', 'impedance', ['x_hi'])]
    if drive == 'velocity':
        features.append(FeatureSpec('drive', 'vibrating_boundary', ['x_lo'], {'quantity': 'velocity',
                                                                              'velocity': _DUCT_VELOCITY}))
    else:
        features += [FeatureSpec('drive', 'prescribed_pressure', ['x_lo'], {'pressure': _DUCT_PRESSURE}),
                     FeatureSpec('incident', 'background_pressure_field', [], {
                         'wave': 'plane_wave', 'amplitude': _BACKGROUND_AMPLITUDE, 'direction': '+x',
                         'background_density': _AIR_DENSITY, 'background_sound_speed': _AIR_SPEED})]
    study = {'frequency_start': _SWEEP_HZ[0], 'frequency_stop': _SWEEP_HZ[1], 'n_frequencies': _SWEEP_HZ[2],
             'snapshots': 'none', 'sweep_method': method}
    if method == 'adaptive':
        study['rom_tolerance'] = _SWEEP_ROM_TOLERANCE
    mesh = MeshSpec(name='fluid', kind='triangles', mode='uniform', dx=_DUCT_ELEMENT)
    return _model(f'Self-test: plane-wave duct, {drive} drive, {method} frequency sweep',
                  DomainSpec(size_x=_DUCT_LENGTH, size_y=_DUCT_WIDTH), geometry, features + probes, mesh,
                  {'discretization': 'quadratic'}, study)


def _sweep_specs():
    return {f'{drive}_{method}': _sweep_spec(drive, method) for drive in _DRIVES for method in ('direct', 'adaptive')}


def _probe_series(results):
    """Complex (probes, frequencies) total pressure of every probe."""
    rows = []
    for index in range(len(_DUCT_PROBE_X)):
        series = results.result(PHYSICS, f'probe{index}')['series']
        rows.append(np.asarray(series['p_re'], dtype=np.float64) + 1j * np.asarray(series['p_im'], dtype=np.float64))
    return np.array(rows)


def _check_sweep(results_by_name):
    errors, tolerances, details = {}, {}, {}
    for drive in _DRIVES:
        direct = _probe_series(results_by_name[f'{drive}_direct'])
        adaptive_results = results_by_name[f'{drive}_adaptive']
        adaptive = _probe_series(adaptive_results)
        scale = float(np.max(np.abs(direct)))
        errors[f'{drive} points'], tolerances[f'{drive} points'] = abs(direct.shape[1] - _SWEEP_HZ[2]), 0.5
        errors[f'{drive} adaptive'] = (float(np.max(np.abs(adaptive - direct))) / scale if adaptive.shape == direct.shape
                                       else math.inf)
        tolerances[f'{drive} adaptive'] = _SWEEP_TOLERANCE
        record = ((adaptive_results.summary().get('frequency_sweep') or {}).get('physics') or {}).get(PHYSICS) or {}
        errors[f'{drive} factorizations'] = float(record.get('factorizations', math.inf))
        tolerances[f'{drive} factorizations'] = float(_SWEEP_MAX_FACTORIZATIONS)
        details[drive] = {'peak_p': scale, 'frequency_sweep': record}
    return _graded(errors, tolerances, details)


# ---- pressure_acoustics_batched_sweep: both drives of the duct above on a batched_direct 'linear' node (batch_size
# _BATCHED_SIZE: three solve_batch calls, the last padded) over _BATCHED_HZ; the impedance end absorbs exactly, so the
# total field is rho c v0 exp(-j k x) (velocity) / p0 exp(-j k x) (held pressure; the background only moves the
# scattered / background split) at every frequency. GPU natural (nvmath-python: else the cpu backend), variants cpu
# (the SuperLU loop of the same plan_batched path) and double ----
_BATCHED_HZ = (300.0, 900.0, 9)
_BATCHED_SIZE = 4
_BATCHED_NODE = 'batched_sweep'
_BATCHED_FIELD_TOLERANCE = 1e-3    # of the closed form's peak: P2 dispersion at 15 elements per wavelength (900 Hz)
_BATCHED_VARIANTS = (SelfTestVariant('cpu', tolerance=1.0, backend='cpu'),
                     SelfTestVariant('double', tolerance=1.0, precision='double', roles=('linear',)))


def _batched_specs():
    from ...solvers.cudss_factor import cudss_available   # find_spec only
    specs = {}
    for drive in _DRIVES:
        spec = _sweep_spec(drive, 'direct')
        spec.title = f'Self-test: plane-wave duct, {drive} drive, batched direct frequency sweep'
        spec.study.settings.update(frequency_start=_BATCHED_HZ[0], frequency_stop=_BATCHED_HZ[1],
                                   n_frequencies=_BATCHED_HZ[2])
        spec.solvers = [SolverSpec(_BATCHED_NODE, 'batched_direct', {'batch_size': _BATCHED_SIZE})]
        spec.physics[0].solvers = {'linear': _BATCHED_NODE}
        spec.solver_defaults.backend = 'gpu' if cudss_available() else 'cpu'
        specs[drive] = spec
    return specs


def _check_batched(results_by_name):
    frequencies = np.linspace(*_BATCHED_HZ[:2], _BATCHED_HZ[2])
    positions = np.array(_DUCT_PROBE_X)
    errors, tolerances, details = {}, {}, {}
    for drive, results in results_by_name.items():
        amplitude = _AIR_DENSITY * _AIR_SPEED * _DUCT_VELOCITY if drive == 'velocity' else _DUCT_PRESSURE
        expected = amplitude * np.exp(-1j * np.outer(positions, [_wavenumber(f) for f in frequencies]))
        measured = _probe_series(results)
        errors[f'{drive} field'] = (float(np.max(np.abs(measured - expected))) / amplitude
                                    if measured.shape == expected.shape else math.inf)
        tolerances[f'{drive} field'] = _BATCHED_FIELD_TOLERANCE
        role = solver_role(results, PHYSICS)
        counters = role.get('counters') or {}
        batches = -(-_BATCHED_HZ[2] // _BATCHED_SIZE)
        record_ok = (role.get('type') == 'batched_direct' and counters.get('batched_systems') == _BATCHED_HZ[2]
                     and counters.get('batches') == batches and counters.get('batch_rejected') == 0
                     and not counters.get('batch_unavailable'))
        errors[f'{drive} batches'], tolerances[f'{drive} batches'] = (0.0 if record_ok else 1.0), 0.5
        details[drive] = {'counters': {name: counters.get(name) for name in (
            'batches', 'batched_systems', 'batch_plans', 'batch_rejected', 'batch_refinement_steps', 'factorizations',
            'worst_batch_residual', 'batch_unavailable')}, 'backend': role.get('backend')}
    return _graded(errors, tolerances, details)


REDUCED_SWEEP_CASES = (
    SelfTestCase(
        name='pressure_acoustics_reduced_sweep',
        label='Pressure acoustics: adaptive frequency sweeps of a duct vs direct',
        description=(f'The plane-wave duct of pressure_acoustics_duct (P2 triangles, impedance x_hi) over '
                     f'{_SWEEP_HZ[0]:g} - {_SWEEP_HZ[1]:g} Hz ({_SWEEP_HZ[2]} frequencies, snapshots none), driven by a '
                     'vibrating x_lo ((j omega) load: moment matching) and by a held x_lo pressure under a +x '
                     f'background plane wave of {_BACKGROUND_AMPLITUDE:g} Pa (non-polynomial load and held offsets: '
                     'snapshots only): the probe pressures of sweep_method adaptive (rom_tolerance '
                     f'{_SWEEP_ROM_TOLERANCE:g}) against the direct sweep within {_SWEEP_TOLERANCE:g} of the peak, at '
                     f'most {_SWEEP_MAX_FACTORIZATIONS} factorisations. measured = error in tolerance units, tolerance '
                     '1 (first estimates, not yet run).'),
        specs=_sweep_specs, check=_check_sweep),
    SelfTestCase(
        name='pressure_acoustics_batched_sweep',
        label='Pressure acoustics: duct frequency sweep on the batched direct solver',
        description=(f'The plane-wave duct of pressure_acoustics_reduced_sweep (P2 triangles, impedance x_hi) over '
                     f'{_BATCHED_HZ[0]:g} - {_BATCHED_HZ[1]:g} Hz ({_BATCHED_HZ[2]} frequencies, sweep_method direct, '
                     "snapshots none) on a batched_direct 'linear' node with batch_size "
                     f'{_BATCHED_SIZE} (fem.harmonic plan_batched: three batches, the last padded on the kept plan): '
                     'driven by a vibrating x_lo ((j omega) load) and by a held x_lo pressure under a +x background '
                     'plane wave (non-polynomial load and held offsets per frequency, computed at batch time). Every '
                     'probe pressure against the absorbed plane wave rho c v0 exp(-j k x) / p0 exp(-j k x) within '
                     f'{_BATCHED_FIELD_TOLERANCE:g} of its amplitude; the linear record shows batched_direct with every '
                     f'frequency batched (batched_systems {_BATCHED_HZ[2]}, '
                     f'{-(-_BATCHED_HZ[2] // _BATCHED_SIZE)} batches, none rejected or unavailable). GPU backend '
                     '(precision mixed: one explicitly batched cuDSS plan; the cpu backend without nvmath-python), '
                     'variants cpu (the SuperLU loop of the same path) and double. measured = error in tolerance '
                     'units, tolerance 1. Written, not run.'),
        specs=_batched_specs, check=_check_batched, variants=_BATCHED_VARIANTS),
)
