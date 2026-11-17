"""Self-tests of the membrane frequency_response study with sweep_method adaptive / modal (fem.reduced_harmonic): the
lossy 1D string against its closed form with few factorisations, and a clamped rectangle window of eight resonances
against the direct sweep. A straight two-port guide swept on a batched_direct 'linear' node against its direct sweep
and full transmission."""
import numpy as np

from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import cpu_defaults, float32_value, scaled_worst_result, solver_role
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, ModelSpec, PhysicsSpec, SolverSpec,
                           SolverDefaultsSpec, StudySpec)
from .selftests import (_FILM, _GUIDE_H, _GUIDE_WIDTH, _MEMBRANE_TYPE, _PHYSICS, _RECTANGLE, _STRING_POINTS,
                        _STRING_PROBE, _STRING_RESPONSE_TOLERANCE, _cutoff_frequency, _probe_phasors,
                        _string_closed_form, _string_spec, _triangles)

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


# ---- membrane_guide_ports_batched: a straight guide W x 2 W (y faces clamped) between two ports (x_lo: incident mode
# 1), _PORTS_POINTS frequencies over 1.2 - 2.6 f_c (mode 2 propagates above 2 f_c; a straight guide converts none):
# full transmission of mode 1, and the batched_direct sweep (the port DtN blocks per frequency as the batch's extra
# blocks at their pattern slots) against the Automatic direct one. GPU natural (nvmath-python: else cpu), variant cpu.
_PORTS_HZ_FACTORS = (1.2, 2.6)
_PORTS_POINTS = 6
_PORTS_BATCH = 4                   # two solve_batch calls, the last padded
_PORTS_MODES = 3
_PORTS_AMPLITUDE = 1e-9            # m
_PORTS_NODE = 'batched_ports'
_PORTS_AGREEMENT_TOLERANCE = 1e-7  # |batched - direct| / A: both refined to a 1e-12 residual
_PORTS_TRANSMISSION_TOLERANCE = 1e-3   # |power_fraction_1 - 1| (P2 dispersion vs the exact port modes; first estimate)
_PORTS_RUNS = ('direct', 'batched')


def _ports_spec(run, backend):
    features = [FeatureSpec('walls', 'clamped', ['y_lo', 'y_hi']),
                FeatureSpec('in_port', 'port', ['x_lo'], {'mode_count': _PORTS_MODES, 'incident_mode': 1,
                                                         'amplitude': _PORTS_AMPLITUDE}),
                FeatureSpec('out_port', 'port', ['x_hi'], {'mode_count': _PORTS_MODES})]
    spec = ModelSpec(
        title=f'Self-test: straight membrane guide between two ports ({run} sweep)',
        domain=DomainSpec(size_x=2.0 * _GUIDE_WIDTH, size_y=_GUIDE_WIDTH),
        materials=[MaterialSpec('film', properties=dict(_FILM), selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=features)],
        meshes=[_triangles(_GUIDE_H)], solver_defaults=SolverDefaultsSpec(backend=backend),
        study=StudySpec(type='frequency_response', settings={
            'frequency_start': _PORTS_HZ_FACTORS[0] * _cutoff_frequency(),
            'frequency_stop': _PORTS_HZ_FACTORS[1] * _cutoff_frequency(), 'n_frequencies': _PORTS_POINTS,
            'snapshots': 'none'}))
    if run == 'batched':
        spec.solvers = [SolverSpec(_PORTS_NODE, 'batched_direct', {'batch_size': _PORTS_BATCH})]
        spec.physics[0].solvers = {'linear': _PORTS_NODE}
    return spec


def _ports_specs():
    from ...solvers.cudss_factor import cudss_available   # find_spec only
    backend = 'gpu' if cudss_available() else 'cpu'
    return {run: _ports_spec(run, backend) for run in _PORTS_RUNS}


def _port_amplitudes(results, port):
    series = results.result(_PHYSICS, port)['series']
    return np.asarray(series['mode_1_re'], dtype=np.float64) + 1j * np.asarray(series['mode_1_im'], dtype=np.float64)


def _check_ports(results_by_name):
    direct, batched = (results_by_name[run] for run in _PORTS_RUNS)
    errors, details = {}, {}
    for port in ('in_port', 'out_port'):
        expected, measured = _port_amplitudes(direct, port), _port_amplitudes(batched, port)
        errors[f'{port} agreement'] = (float(np.max(np.abs(measured - expected))) / _PORTS_AMPLITUDE
                                       if measured.shape == expected.shape == (_PORTS_POINTS,) else float('inf'))
    transmission = np.asarray(batched.result(_PHYSICS, 'out_port')['series']['power_fraction_1'], dtype=np.float64)
    errors['transmission'] = float(np.max(np.abs(transmission - 1.0))) if transmission.size else float('inf')
    counters = solver_role(batched, _PHYSICS).get('counters') or {}
    record_ok = (solver_role(batched, _PHYSICS).get('type') == 'batched_direct'
                 and counters.get('batched_systems') == _PORTS_POINTS and counters.get('batch_rejected') == 0
                 and counters.get('batches') == -(-_PORTS_POINTS // _PORTS_BATCH) and not counters.get('batch_unavailable'))
    errors['batched record'] = 0.0 if record_ok else 1.0
    tolerances = {'in_port agreement': _PORTS_AGREEMENT_TOLERANCE, 'out_port agreement': _PORTS_AGREEMENT_TOLERANCE,
                  'transmission': _PORTS_TRANSMISSION_TOLERANCE, 'batched record': 0.5}
    details['counters'] = {name: counters.get(name) for name in ('batches', 'batched_systems', 'batch_plans',
                                                                 'batch_rejected', 'factorizations',
                                                                 'worst_batch_residual', 'batch_unavailable')}
    return scaled_worst_result(errors, tolerances, transmission=transmission.tolist(), **details)


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
    SelfTestCase(
        'membrane_guide_ports_batched', 'Membrane: two-port guide swept on the batched direct solver',
        'A straight guide W x 2 W (W = 0.2 mm, y faces clamped, quadratic triangles h = W / 16) between two ports of '
        f'{_PORTS_MODES} modes (x_lo: incident mode 1 of {_PORTS_AMPLITUDE:g} m) at {_PORTS_POINTS} frequencies over '
        f'{_PORTS_HZ_FACTORS[0]:g} - {_PORTS_HZ_FACTORS[1]:g} f_c, sweep_method direct: the Automatic direct run '
        f"against a batched_direct 'linear' node with batch_size {_PORTS_BATCH} (fem.harmonic plan_batched; the port "
        'DtN blocks added per frequency at their pattern slots as the batch extras): the mode-1 amplitudes of both '
        f'ports agree to {_PORTS_AGREEMENT_TOLERANCE:g} A, the out_port power_fraction_1 is 1 within '
        f'{_PORTS_TRANSMISSION_TOLERANCE:g} (no conversion in a straight guide), the linear record shows every '
        'frequency batched (none rejected or unavailable). GPU backend (the cpu backend without nvmath-python), '
        'variant cpu (the SuperLU loop of the same path). Written, not run.',
        _ports_specs, _check_ports, variants=(SelfTestVariant('cpu', tolerance=1.0, backend='cpu'),)),
)
