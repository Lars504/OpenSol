"""Self-tests of pressure_acoustics frequency sweeps with sweep_method adaptive (experiment plan A3 / A4): the plane-wave
duct of selftests_harmonic driven by a vibrating end ((j omega) loads: moment matching) and by a held pressure under a
background plane wave (non-polynomial load and held offsets: snapshots), each against its direct sweep."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.spec import DomainSpec, FeatureSpec, MeshSpec
from .selftests_harmonic import (_AIR_DENSITY, _AIR_SPEED, _DUCT_ELEMENT, _DUCT_LENGTH, _DUCT_PRESSURE, _DUCT_PROBE_X,
                                 _DUCT_VELOCITY, _DUCT_WIDTH, PHYSICS, _graded, _model, _probes)

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
)
