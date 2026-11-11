"""Self-tests of the membrane frequency_response study with sweep_method adaptive / modal (fem.reduced_harmonic): the
lossy 1D string against its closed form with few factorisations, and a clamped rectangle window of eight resonances
against the direct sweep."""
import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cpu_defaults, float32_value, scaled_worst_result
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests import (_FILM, _MEMBRANE_TYPE, _PHYSICS, _RECTANGLE, _STRING_POINTS, _STRING_PROBE,
                        _STRING_RESPONSE_TOLERANCE, _probe_phasors, _string_closed_form, _string_spec, _triangles)

_ROM_TOLERANCE = 1e-10            # training residual: Q x residual stays far below the checked errors
_LINE_MAX_FACTORIZATIONS = 6      # expansions + the verification (plan A1)
# clamped 1 x 0.7 mm rectangle, eta = 1e-2 (Q = 100), point force off every nodal line of the lowest modes; the window
# 0.42 - 1.30 MHz holds f_11 .. f_13 (eight modes)
_WINDOW_HZ = (0.42e6, 1.30e6)
_WINDOW_POINTS = 201
_WINDOW_LOSS = float32_value(1e-2)
_WINDOW_H = _RECTANGLE[0] / 40.0
_WINDOW_DRIVE = (0.31, 0.27)      # fractions of a, b
_WINDOW_PROBE_POINT = (0.62, 0.41)
_WINDOW_PROBE = 'window_probe'
_WINDOW_MODAL_COUNT = 60          # every mode below about 3 MHz
_WINDOW_ADAPTIVE_TOLERANCE = 1e-6   # of the peak |w|, against the direct sweep
_WINDOW_MODAL_TOLERANCE = 5e-3      # truncated modes + static correction (first estimate)
_WINDOW_MAX_FACTORIZATIONS = 41     # rom_max_points + the verification


def _sweep_record(results):
    """The membrane's frequency_sweep info of a run's summary ({} without one)."""
    sweep = results.summary().get('frequency_sweep') or {}
    return (sweep.get('physics') or {}).get(_PHYSICS) or {}


# ---- membrane_line_loss_adaptive: membrane_line_loss's driven run with sweep_method adaptive ----
def _line_loss_specs():
    spec = _string_spec('response')
    spec.study.settings = {**spec.study.settings, 'sweep_method': 'adaptive', 'rom_tolerance': _ROM_TOLERANCE}
    return {'response': spec}


def _check_line_loss(results_by_name):
    results = results_by_name['response']
    frequencies, measured = _probe_phasors(results, _STRING_PROBE)
    expected = _string_closed_form(frequencies)
    record = _sweep_record(results)
    errors = {'response': float(np.max(np.abs(measured - expected))) / float(np.max(np.abs(expected))),
              'points': abs(len(frequencies) - _STRING_POINTS),
              'factorizations': float(record.get('factorizations', np.inf))}
    tolerances = {'response': _STRING_RESPONSE_TOLERANCE, 'points': 0.5,
                  'factorizations': float(_LINE_MAX_FACTORIZATIONS)}
    return scaled_worst_result(errors, tolerances, frequency_sweep=record)


# ---- membrane_rectangle_reduced_sweep: direct, adaptive and modal sweeps of one window ----
def _window_spec(method):
    size_x, size_y = _RECTANGLE
    settings = {'frequency_start': _WINDOW_HZ[0], 'frequency_stop': _WINDOW_HZ[1], 'n_frequencies': _WINDOW_POINTS,
                'snapshots': 'none', 'sweep_method': method}
    if method == 'adaptive':
        settings['rom_tolerance'] = _ROM_TOLERANCE
    elif method == 'modal':
        settings['modal_count'] = _WINDOW_MODAL_COUNT
    return ModelSpec(
        title=f'Self-test: clamped membrane rectangle window ({method} sweep)',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('drive', 'point', {'x': _WINDOW_DRIVE[0] * size_x, 'y': _WINDOW_DRIVE[1] * size_y}),
                  GeometrySpec('probe_point', 'point', {'x': _WINDOW_PROBE_POINT[0] * size_x,
                                                        'y': _WINDOW_PROBE_POINT[1] * size_y})],
        materials=[MaterialSpec('lossy_film', properties={**_FILM, 'loss_factor': _WINDOW_LOSS}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=[
            FeatureSpec('frame', 'clamped', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
            FeatureSpec('drive_force', 'point_force', ['drive'], {'force': 1e-6}),
            FeatureSpec(_WINDOW_PROBE, 'field_probe', ['probe_point'])])],
        meshes=[_triangles(_WINDOW_H)], solver_defaults=cpu_defaults(),
        study=StudySpec(type='frequency_response', settings=settings))


def _window_specs():
    return {method: _window_spec(method) for method in ('direct', 'adaptive', 'modal')}


def _check_window(results_by_name):
    frequencies, direct = _probe_phasors(results_by_name['direct'], _WINDOW_PROBE)
    scale = float(np.max(np.abs(direct)))
    errors, details = {'points': abs(len(frequencies) - _WINDOW_POINTS)}, {'peak_w': scale}
    for method in ('adaptive', 'modal'):
        _, measured = _probe_phasors(results_by_name[method], _WINDOW_PROBE)
        errors[method] = (float(np.max(np.abs(measured - direct))) / scale if measured.shape == direct.shape
                          else float('inf'))
        details[f'{method}_sweep'] = _sweep_record(results_by_name[method])
    errors['adaptive_factorizations'] = float(details['adaptive_sweep'].get('factorizations', np.inf))
    tolerances = {'points': 0.5, 'adaptive': _WINDOW_ADAPTIVE_TOLERANCE, 'modal': _WINDOW_MODAL_TOLERANCE,
                  'adaptive_factorizations': float(_WINDOW_MAX_FACTORIZATIONS)}
    return scaled_worst_result(errors, tolerances, **details)


MEMBRANE_REDUCED_SWEEP_CASES = (
    SelfTestCase(
        'membrane_line_loss_adaptive', 'Membrane: lossy string with an adaptive reduced sweep',
        "membrane_line_loss's driven run (clamped 1D string, eta = 1e-3, uniform pressure, "
        f'{_STRING_POINTS} frequencies around f_1) with sweep_method adaptive (rom_tolerance {_ROM_TOLERANCE:g}): the '
        'probe phasor at L / 2 against p / (rho_s omega^2) (1 / cos(k L / 2) - 1) '
        f'({_STRING_RESPONSE_TOLERANCE:g} of the peak) with at most {_LINE_MAX_FACTORIZATIONS} factorisations (summary '
        'frequency_sweep, the verification included).',
        _line_loss_specs, _check_line_loss),
    SelfTestCase(
        'membrane_rectangle_reduced_sweep', 'Membrane: rectangle window, adaptive and modal sweeps vs direct',
        'The clamped 1 x 0.7 mm film (eta = 1e-2) driven by an off-centre point force, quadratic triangles h = a / 40, '
        f'{_WINDOW_POINTS} frequencies over 0.42 - 1.30 MHz (eight modes): the probe phasor of sweep_method adaptive '
        f'(rom_tolerance {_ROM_TOLERANCE:g}) against the direct sweep to {_WINDOW_ADAPTIVE_TOLERANCE:g} of the peak with '
        f'at most {_WINDOW_MAX_FACTORIZATIONS} factorisations, and of modal ({_WINDOW_MODAL_COUNT} modes nearest the '
        f'band centre + static correction) to {_WINDOW_MODAL_TOLERANCE:g} (first estimates).',
        _window_specs, _check_window),
)
