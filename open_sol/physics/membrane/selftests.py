"""Self-tests of the membrane physics (new_modules_plan S4 / S7) against closed forms: rectangle and circular-drum modes
on triangles, a rectangle split by a clamped line, the 1D strip dispersion with an axial wavenumber, the lossy string (driven response, Lorentzian Q, complex
resonance), evanescent decay below cut-off, the PML-terminated guide, the 1D plate strip (Hermite), the simply
supported plate rectangle, its half models with mirror planes and the clamped plate strip Bloch cell (interior penalty),
port tunnelling / mode filter and
the em_fdfd_ez cross-check; plus the silicon nitride template."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import cpu_defaults, float32_value, scaled_worst_result
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, SolverDefaultsSpec, StudySpec)

_PHYSICS = 'membrane'
_MEMBRANE_TYPE = 'membrane'   # = definition.MEMBRANE_TYPE_NAME (selftests never import the definition)
_TRIANGLE_MESH = 'triangles'
_TABLE = 'modes'
_FREQUENCY_COLUMN, _Q_COLUMN = 1, 3   # resonance_table rows: number, frequency, frequency_imag, q_factor, pml_fraction

# silicon nitride film, 50 nm at 1 GPa: T = sigma h, rho_s = rho h (float32-exact: the MaterialGrid stores float32)
_TENSION = float32_value(50.0)              # N/m
_AREAL_DENSITY = float32_value(1.55e-4)     # kg/m^2
_FILM = {'tension': _TENSION, 'areal_density': _AREAL_DENSITY}


def _wave_speed():
    """c = sqrt(T / rho_s) [m/s] of the film."""
    return math.sqrt(_TENSION / _AREAL_DENSITY)


def _triangles(dx):
    return MeshSpec(name=_TRIANGLE_MESH, kind='triangles', mode='uniform', dx=dx)


def _table_frequencies(results, count, column=_FREQUENCY_COLUMN):
    """Sorted column values of the resonance table (ValueError with fewer than count rows)."""
    rows = results.result(_PHYSICS, _TABLE)['rows']
    if len(rows) < count:
        raise ValueError(f'expected {count} resonances in the table, got {len(rows)}')
    return sorted(float(row[column]) for row in rows)[:count]


def _probe_phasors(results, feature):
    """(frequencies [Hz], complex w) of a field_probe result."""
    result = results.result(_PHYSICS, feature)
    series = result['series']
    return (np.asarray(result['x'], dtype=np.float64),
            np.asarray(series['w_re'], dtype=np.float64) + 1j * np.asarray(series['w_im'], dtype=np.float64))


# ---- the template rectangle (checked in membrane_emfdfd_rectangle): a x b clamped rectangle, P2 triangles h = a / 80. Continuum f_mn =
# (c / 2) sqrt((m / a)^2 + (n / b)^2); P2 eigenvalue errors ~ (k h)^4 (~1e-6 at the 6th mode: first-estimate tolerance).
_RECTANGLE = (1.0e-3, 0.7e-3)
_RECTANGLE_H = _RECTANGLE[0] / 80.0
_RECTANGLE_MODES = 6
_RECTANGLE_TOLERANCE = 1e-5
RECTANGLE_LABEL = 'Silicon nitride membrane modes'
RECTANGLE_DESCRIPTION = (
    'A 1 x 0.7 mm silicon nitride membrane (50 nm film at 1 GPa stress: tension 50 N/m, areal density 1.55e-4 '
    'kg/m^2) clamped on its frame: the six lowest drum modes on quadratic triangles, f_mn = (c / 2) sqrt((m / a)^2 + '
    '(n / b)^2) with c = sqrt(T / rho_s) = 568 m/s (f_11 = 495 kHz), listed by the resonance table.')


def rectangle_modes_spec(backend='auto'):
    """The clamped silicon nitride rectangle (template 'silicon_nitride_membrane_modes'; self-test on the cpu)."""
    size_x, size_y = _RECTANGLE
    return ModelSpec(
        title='Membrane: silicon nitride rectangle modes', description=RECTANGLE_DESCRIPTION,
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        materials=[MaterialSpec('silicon_nitride_film', properties=dict(_FILM), selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=[
            FeatureSpec('frame', 'clamped', ['x_lo', 'x_hi', 'y_lo', 'y_hi']), FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[_triangles(_RECTANGLE_H)], solver_defaults=SolverDefaultsSpec(backend=backend),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _RECTANGLE_MODES}))


def _rectangle_frequencies():
    size_x, size_y = _RECTANGLE
    values = [0.5 * _wave_speed() * math.hypot(m / size_x, n / size_y) for m in range(1, 9) for n in range(1, 9)]
    return sorted(values)[:_RECTANGLE_MODES]


def _check_rectangle(results_by_name):
    results = results_by_name['rectangle']
    expected = _rectangle_frequencies()
    measured = _table_frequencies(results, _RECTANGLE_MODES)
    labels = results.summary().get('snapshot_labels') or []
    errors = {'frequency': max(abs(value / reference - 1.0) for value, reference in zip(measured, expected)),
              'snapshots': abs(len(labels) - _RECTANGLE_MODES)}
    return scaled_worst_result(errors, {'frequency': _RECTANGLE_TOLERANCE, 'snapshots': 0.5},
                               frequency_expected=expected, frequency_measured=measured, snapshot_labels=labels)


# ---- membrane_drum_modes: a circle of radius R clamped on its outline (curved: isoparametric P2), the square around it
# without material (holes, excluded); f = j_mn c / (2 pi R), degenerate pairs for m >= 1. First-estimate tolerance.
_DRUM_RADIUS = 0.5e-3
_DRUM_H = _DRUM_RADIUS / 20.0
_DRUM_ZEROS = (2.404825557695773, 3.831705970207512, 3.831705970207512, 5.135622301840683, 5.135622301840683,
               5.520078110286311)   # j_01, j_11 (x2), j_21 (x2), j_02
_DRUM_TOLERANCE = 1e-4


def _drum_spec():
    side = 2.2 * _DRUM_RADIUS
    centre = 0.5 * side
    return ModelSpec(
        title='Self-test: clamped circular membrane (Bessel modes)',
        domain=DomainSpec(size_x=side, size_y=side),
        geometry=[GeometrySpec('drum', 'circle', {'cx': centre, 'cy': centre, 'r': _DRUM_RADIUS})],
        materials=[MaterialSpec('film', properties=dict(_FILM), selection=['drum'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=[
            FeatureSpec('rim', 'clamped', ['drum']), FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[_triangles(_DRUM_H)], solver_defaults=cpu_defaults(),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': len(_DRUM_ZEROS)}))


def _check_drum(results_by_name):
    expected = [zero * _wave_speed() / (2.0 * math.pi * _DRUM_RADIUS) for zero in _DRUM_ZEROS]
    measured = _table_frequencies(results_by_name['drum'], len(expected))
    errors = {'frequency': max(abs(value / reference - 1.0) for value, reference in zip(measured, expected))}
    return scaled_worst_result(errors, {'frequency': _DRUM_TOLERANCE},
                               frequency_expected=expected, frequency_measured=measured)


# ---- membrane_line_clamp_strips: the template rectangle with a clamped_line across it at x = a1 (a zero-thickness
# wall from y_lo to y_hi, frame clamped): the modes are those of the two clamped rectangles a1 x b and (a - a1) x b
# together. P2 h = a / 80 (a1 = 32 h: the line on element edges either way); the sixth mode sits near the template's
# sixth, so the template tolerance doubled (first estimate).
_IRIS_X = 0.4e-3
_IRIS_MODES = 6
_IRIS_TOLERANCE = 2e-5


def _iris_spec():
    size_x, size_y = _RECTANGLE
    return ModelSpec(
        title='Self-test: membrane rectangle split by a clamped line',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('iris', 'line', {'x0': _IRIS_X, 'y0': 0.0, 'x1': _IRIS_X, 'y1': size_y})],
        materials=[MaterialSpec('film', properties=dict(_FILM), selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=[
            FeatureSpec('frame', 'clamped', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
            FeatureSpec('wall', 'clamped_line', ['iris']), FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[_triangles(_RECTANGLE_H)], solver_defaults=cpu_defaults(),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _IRIS_MODES}))


def _iris_frequencies():
    """The lowest f_mn of the clamped rectangles a1 x b and (a - a1) x b together [Hz]."""
    size_x, size_y = _RECTANGLE
    values = [0.5 * _wave_speed() * math.hypot(m / width, n / size_y)
              for width in (_IRIS_X, size_x - _IRIS_X) for m in range(1, 9) for n in range(1, 9)]
    return sorted(values)[:_IRIS_MODES]


def _check_iris(results_by_name):
    expected = _iris_frequencies()
    measured = _table_frequencies(results_by_name['iris'], _IRIS_MODES)
    errors = {'frequency': max(abs(value / reference - 1.0) for value, reference in zip(measured, expected))}
    return scaled_worst_result(errors, {'frequency': _IRIS_TOLERANCE},
                               frequency_expected=expected, frequency_measured=measured)


# ---- membrane_line_dispersion: a 1D strip of width W clamped at both ends with axial wavenumber k_y; branches
# omega_n = c sqrt((n pi / W)^2 + k_y^2). The k_y^2 term is exact (a mass-type term of the same Gram matrix), so only the
# P2 transverse eigenvalue errs (~(n pi h / W)^4 / 720, < 1e-9 at n = 4 on 200 cells).
_STRIP_WIDTH = 0.2e-3
_STRIP_CELLS = 200
_STRIP_WAVENUMBER = 1.5 * math.pi / _STRIP_WIDTH   # rad/m
_STRIP_MODES = 4
_STRIP_TOLERANCE = 1e-6


def _strip_spec():
    return ModelSpec(
        title='Self-test: 1D membrane strip dispersion (axial wavenumber)',
        domain=DomainSpec(size_x=_STRIP_WIDTH, size_y=None),
        materials=[MaterialSpec('film', properties=dict(_FILM), selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, settings={'axial_wavenumber': _STRIP_WAVENUMBER}, features=[
            FeatureSpec('ends', 'clamped', ['x_lo', 'x_hi']), FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[MeshSpec(mode='uniform', dx=_STRIP_WIDTH / _STRIP_CELLS)], solver_defaults=cpu_defaults(),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _STRIP_MODES}))


def _check_strip(results_by_name):
    expected = [_wave_speed() * math.hypot(n * math.pi / _STRIP_WIDTH, _STRIP_WAVENUMBER) / (2.0 * math.pi)
                for n in range(1, _STRIP_MODES + 1)]
    measured = _table_frequencies(results_by_name['strip'], _STRIP_MODES)
    errors = {'frequency': max(abs(value / reference - 1.0) for value, reference in zip(measured, expected))}
    return scaled_worst_result(errors, {'frequency': _STRIP_TOLERANCE},
                               frequency_expected=expected, frequency_measured=measured)


# ---- membrane_line_loss: a 1D string of length L clamped at both ends under uniform pressure p with loss factor eta.
# 'response': frequency response around f_1 = c / (2 L); w(L / 2) = p / (rho_s omega^2) (1 / cos(k L / 2) - 1), k = omega
# sqrt(rho_s / (T (1 + j eta))), at the probe (the P2 nodal error ~ (k h)^4 / eta is ~1e-8 of the peak); a lorentz_fit of
# |w|^2 gives Q = 1 / eta to O(eta) (line shape and off-resonant background: first-estimate 1e-2). 'modes': the complex
# resonance omega = omega_h sqrt(1 + j eta) exactly (K = (1 + j eta) K_0), so Q = Re s / (2 Im s), s = sqrt(1 + j eta).
# Closed forms use the meshed length (a 1D window stores float32 cell widths: L (1 - 2.5e-8), which 2 Q = 2000 lifts to
# 5e-5 of the peak near f_1).
_STRING_LENGTH = 1.0e-3
_STRING_CELLS = 200
_STRING_MESH_LENGTH = _STRING_CELLS * float32_value(_STRING_LENGTH / _STRING_CELLS)
_STRING_LOSS = float32_value(1e-3)
_STRING_PRESSURE = 1.0                 # Pa
_STRING_POINTS = 51
_STRING_SPAN = 2.5                     # frequencies f_1 (1 +- span eta)
_STRING_PROBE = 'middle_probe'
_STRING_RESPONSE_TOLERANCE = 1e-5      # relative to the largest |w|
_STRING_FIT_TOLERANCE = 1e-2           # relative Q of the Lorentzian fit
_STRING_EIGEN_TOLERANCE = 1e-6


def _string_first_frequency():
    return _wave_speed() / (2.0 * _STRING_MESH_LENGTH)


def _string_frequencies():
    return [float(_string_first_frequency() * (1.0 + shift))
            for shift in np.linspace(-_STRING_SPAN * _STRING_LOSS, _STRING_SPAN * _STRING_LOSS, _STRING_POINTS)]


def _string_spec(run):
    material = {**_FILM, 'loss_factor': _STRING_LOSS}
    features = [FeatureSpec('ends', 'clamped', ['x_lo', 'x_hi'])]
    if run == 'response':
        features += [FeatureSpec('load', 'pressure_load', ['all'], {'pressure': _STRING_PRESSURE}),
                     FeatureSpec(_STRING_PROBE, 'field_probe', ['middle'])]
        study = StudySpec(type='frequency_response', settings={'frequencies': _string_frequencies()})
        derived = [DerivedValueSpec('q_fit', 'w_abs*w_abs', [], 'lorentz_fit', source=f'{_PHYSICS}.{_STRING_PROBE}',
                                    fit='q')]
        settings = {}
    else:
        features.append(FeatureSpec(_TABLE, 'resonance_table'))
        study = StudySpec(type='eigenfrequency', settings={'n_resonances': 1})
        derived = []
        settings = {'target_frequency': _string_first_frequency()}
    return ModelSpec(
        title=f'Self-test: lossy 1D membrane string ({run})',
        domain=DomainSpec(size_x=_STRING_LENGTH, size_y=None),
        geometry=[GeometrySpec('middle', 'point', {'x': 0.5 * _STRING_LENGTH, 'y': 0.0})],
        materials=[MaterialSpec('lossy_film', properties=material, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, settings=settings, features=features)],
        meshes=[MeshSpec(mode='uniform', dx=_STRING_LENGTH / _STRING_CELLS)], solver_defaults=cpu_defaults(),
        study=study, derived_values=derived)


def _string_specs():
    return {'response': _string_spec('response'), 'modes': _string_spec('modes')}


def _string_closed_form(frequencies):
    omega = 2.0 * math.pi * np.asarray(frequencies, dtype=np.float64)
    wavenumber = omega * np.sqrt(_AREAL_DENSITY / (_TENSION * (1.0 + 1j * _STRING_LOSS)))
    return _STRING_PRESSURE / (_AREAL_DENSITY * omega ** 2) * (1.0 / np.cos(0.5 * wavenumber * _STRING_MESH_LENGTH) - 1.0)


def _check_string(results_by_name):
    response = results_by_name['response']
    frequencies, measured = _probe_phasors(response, _STRING_PROBE)
    expected = _string_closed_form(frequencies)
    scale = float(np.max(np.abs(expected)))
    shape_factor = complex(np.sqrt(1.0 + 1j * _STRING_LOSS))
    exact_q = shape_factor.real / (2.0 * shape_factor.imag)
    q_fit = float(response.summary()['derived']['q_fit'])
    modes = results_by_name['modes']
    rows = modes.result(_PHYSICS, _TABLE)['rows']
    eigen_frequency, eigen_q = float(rows[0][_FREQUENCY_COLUMN]), float(rows[0][_Q_COLUMN])
    expected_frequency = _string_first_frequency() * shape_factor.real
    errors = {'response': float(np.max(np.abs(measured - expected))) / scale,
              'points': abs(len(frequencies) - _STRING_POINTS),
              'lorentz_q': abs(q_fit / (1.0 / _STRING_LOSS) - 1.0),
              'eigen_q': abs(eigen_q / exact_q - 1.0),
              'eigen_frequency': abs(eigen_frequency / expected_frequency - 1.0)}
    tolerances = {'response': _STRING_RESPONSE_TOLERANCE, 'points': 0.5, 'lorentz_q': _STRING_FIT_TOLERANCE,
                  'eigen_q': _STRING_EIGEN_TOLERANCE, 'eigen_frequency': _STRING_EIGEN_TOLERANCE}
    return scaled_worst_result(errors, tolerances, q_fit=q_fit, eigen_q=eigen_q, exact_q=exact_q,
                               eigen_frequency=eigen_frequency, expected_frequency=expected_frequency,
                               peak_w=scale)


# ---- membrane_guide_evanescent: a strip guide of width W (y faces clamped, x_hi clamped at L = 5 W) driven by a line
# force on x_lo at half the first cut-off f_c = c / (2 W): mode 1 decays as sinh(kappa (L - x)), kappa = (pi / W) sqrt(1
# - 0.25) (mode 3 is 5e-6 of it at the probes). |w(2.5 W) / w(1.5 W)| on the centre line against that ratio.
_GUIDE_WIDTH = 0.2e-3
_GUIDE_H = _GUIDE_WIDTH / 16.0
_GUIDE_LENGTH = 5.0 * _GUIDE_WIDTH
_GUIDE_FORCE = 1.0                     # N/m
_EVANESCENT_FACTOR = 0.5               # drive frequency / f_c
_EVANESCENT_PROBES = {'near_probe': 1.5, 'far_probe': 2.5}   # x / W
_EVANESCENT_TOLERANCE = 1e-4           # relative ratio (first estimate: P2 h = W / 16)


def _cutoff_frequency():
    return _wave_speed() / (2.0 * _GUIDE_WIDTH)


def _guide_spec(title, length, frequency, probes, clamped_faces, settings=None):
    points = [GeometrySpec(name.removesuffix('_probe'), 'point', {'x': fraction * _GUIDE_WIDTH, 'y': 0.5 * _GUIDE_WIDTH})
              for name, fraction in probes.items()]
    features = [FeatureSpec('walls', 'clamped', list(clamped_faces)),
                FeatureSpec('drive', 'line_force', ['x_lo'], {'force': _GUIDE_FORCE})]
    features += [FeatureSpec(name, 'field_probe', [name.removesuffix('_probe')]) for name in probes]
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=length, size_y=_GUIDE_WIDTH), geometry=points,
        materials=[MaterialSpec('film', properties=dict(_FILM), selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, settings=dict(settings or {}), features=features)],
        meshes=[_triangles(_GUIDE_H)], solver_defaults=cpu_defaults(),
        study=StudySpec(type='frequency_response', settings={'frequencies': [frequency]}))


def _evanescent_specs():
    return {'guide': _guide_spec('Self-test: evanescent membrane guide below cut-off', _GUIDE_LENGTH,
                                 _EVANESCENT_FACTOR * _cutoff_frequency(), _EVANESCENT_PROBES,
                                 ('y_lo', 'y_hi', 'x_hi'))}


def _check_evanescent(results_by_name):
    results = results_by_name['guide']
    values = {name: _probe_phasors(results, name)[1][0] for name in _EVANESCENT_PROBES}
    decay = math.pi / _GUIDE_WIDTH * math.sqrt(1.0 - _EVANESCENT_FACTOR ** 2)
    x_near, x_far = (_EVANESCENT_PROBES[name] * _GUIDE_WIDTH for name in ('near_probe', 'far_probe'))
    expected = math.sinh(decay * (_GUIDE_LENGTH - x_far)) / math.sinh(decay * (_GUIDE_LENGTH - x_near))
    ratio = values['far_probe'] / values['near_probe']
    errors = {'ratio': abs(ratio - expected) / expected}
    return scaled_worst_result(errors, {'ratio': _EVANESCENT_TOLERANCE},
                               ratio_measured=[ratio.real, ratio.imag], ratio_expected=expected,
                               kappa=decay, w_near=[values['near_probe'].real, values['near_probe'].imag])


# ---- membrane_guide_pml: the same guide (window length 4 W) with a PML ring of 12 h_max on x_hi, driven at 1.5 f_c
# (mode 1 propagates, beta = (pi / W) sqrt(1.5^2 - 1); mode 3 evanescent): w = A exp(-j beta x) + B exp(+j beta x) on
# the centre line by least squares over four probes; |B / A| < 0.01 (-40 dB).
_PML_FACTOR = 1.5
_PML_PROBES = {'first_probe': 1.5, 'second_probe': 2.0, 'third_probe': 2.5, 'fourth_probe': 3.0}   # x / W
_PML_REFLECTION_LIMIT = 0.01


def _pml_specs():
    return {'guide': _guide_spec('Self-test: PML-terminated membrane guide', 4.0 * _GUIDE_WIDTH,
                                 _PML_FACTOR * _cutoff_frequency(), _PML_PROBES, ('y_lo', 'y_hi'),
                                 settings={'pml_x_hi': True, 'pml_cells': 12})}


def _check_pml(results_by_name):
    results = results_by_name['guide']
    beta = math.pi / _GUIDE_WIDTH * math.sqrt(_PML_FACTOR ** 2 - 1.0)
    positions = np.array([_PML_PROBES[name] * _GUIDE_WIDTH for name in _PML_PROBES])
    values = np.array([_probe_phasors(results, name)[1][0] for name in _PML_PROBES])
    basis = np.stack([np.exp(-1j * beta * positions), np.exp(1j * beta * positions)], axis=1)
    (forward, backward), *_ = np.linalg.lstsq(basis, values, rcond=None)
    reflection = abs(backward) / abs(forward)
    residual = float(np.linalg.norm(basis @ np.array([forward, backward]) - values) / np.linalg.norm(values))
    errors = {'reflection': reflection, 'two_wave_residual': residual}
    return scaled_worst_result(errors, {'reflection': _PML_REFLECTION_LIMIT, 'two_wave_residual': 1e-2},
                               forward_abs=abs(forward), backward_abs=abs(backward),
                               reflection_dB=20.0 * math.log10(max(reflection, 1e-300)))


# ---- exact clamped plate strip (romero2019 plan section 4): D w'''' - T w'' = rho_s omega^2 w on [0, W] clamped (w,
# w'), longitudinal wavenumber k: q1^2 - q2^2 = 2 k^2 + T / D; symmetric (odd n) q2 sin(q2 W / 2) + q1 cos(q2 W / 2)
# tanh(q1 W / 2) = 0, antisymmetric (even n) q1 sin(q2 W / 2) - q2 cos(q2 W / 2) tanh(q1 W / 2) = 0, root n in q2 in (n
# pi / W, (n + 1) pi / W); omega^2 = (T Q + D Q^2) / rho_s, Q = q2^2 + k^2 (the simply supported form with q2 = n pi / W).
def _plate_strip_frequency(number, wavenumber, rigidity, tension, density, width):
    """Exact f_n [Hz] of the clamped tensioned plate strip with longitudinal wavenumber k [rad/m]."""
    import scipy.optimize
    half = 0.5 * width
    offset = 2.0 * wavenumber ** 2 + tension / rigidity

    def mismatch(q2):
        q1 = math.sqrt(q2 * q2 + offset)
        if number % 2:
            return q2 * math.sin(q2 * half) + q1 * math.cos(q2 * half) * math.tanh(q1 * half)
        return q1 * math.sin(q2 * half) - q2 * math.cos(q2 * half) * math.tanh(q1 * half)
    step = math.pi / width
    q2 = scipy.optimize.brentq(mismatch, number * step, (number + 1) * step, xtol=1e-13 * step, rtol=1e-14,
                               maxiter=200)
    return _plate_frequency(q2 * q2 + wavenumber ** 2, rigidity, tension, density)


def _plate_frequency(squared_wavenumber, rigidity, tension, density):
    """f [Hz] of a plane plate wave: omega^2 = (T q^2 + D q^4) / rho_s."""
    return math.sqrt((tension * squared_wavenumber + rigidity * squared_wavenumber ** 2) / density) / (2.0 * math.pi)


# ---- membrane_plate_strip_dispersion (R-E1B, romero2019 plan section 6): 1D Hermite plate strip, W = 76 um, h = 80 nm
# silicon nitride at 1 GPa (E 250 GPa, nu 0.23): clamped at both ends, k_y = 0 and 5e4 rad/m, n = 1, 2 against the plan
# table (5 digits: 1e-3 as the plan) and the exact roots of the float32 material values (1e-5; boundary layer sqrt(D / T)
# = 0.38 um on 38 nm cells).
_PLATE_WIDTH = 76e-6
_PLATE_THICKNESS = 80e-9
_PLATE_TENSION = float32_value(1e9 * _PLATE_THICKNESS)                    # 80 N/m
_PLATE_AREAL_DENSITY = float32_value(3200.0 * _PLATE_THICKNESS)           # 2.56e-4 kg/m^2
_PLATE_RIGIDITY = float32_value(250e9 * _PLATE_THICKNESS ** 3 / (12.0 * (1.0 - 0.23 ** 2)))   # 1.126e-11 N m
_PLATE_CELLS = 2000
_PLATE_TABLE = ((0.0, (3.7149e6, 7.4324e6)), (5e4, (5.7971e6, 8.6646e6)))   # (k_y [rad/m], (f_1, f_2) [Hz])
_PLATE_TABLE_TOLERANCE = 1e-3
_PLATE_EXACT_TOLERANCE = 1e-5


def _plate_strip_specs():
    material = {'tension': _PLATE_TENSION, 'areal_density': _PLATE_AREAL_DENSITY,
                'bending_stiffness': _PLATE_RIGIDITY}
    specs = {}
    for index, (wavenumber, _) in enumerate(_PLATE_TABLE):
        specs[f'ky_{index}'] = ModelSpec(
            title=f'Self-test: 1D clamped plate strip, k_y = {wavenumber:g} rad/m',
            domain=DomainSpec(size_x=_PLATE_WIDTH, size_y=None),
            materials=[MaterialSpec('nitride_plate', properties=dict(material), selection=['all'])],
            physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, settings={'axial_wavenumber': wavenumber}, features=[
                FeatureSpec('ends', 'clamped', ['x_lo', 'x_hi']), FeatureSpec(_TABLE, 'resonance_table')])],
            meshes=[MeshSpec(mode='uniform', dx=_PLATE_WIDTH / _PLATE_CELLS)], solver_defaults=cpu_defaults(),
            study=StudySpec(type='eigenfrequency', settings={'n_resonances': 2}))
    return specs


def _check_plate_strip(results_by_name):
    errors, details = {'table': 0.0, 'exact': 0.0}, {}
    for index, (wavenumber, table) in enumerate(_PLATE_TABLE):
        measured = _table_frequencies(results_by_name[f'ky_{index}'], 2)
        exact = [_plate_strip_frequency(number, wavenumber, _PLATE_RIGIDITY, _PLATE_TENSION, _PLATE_AREAL_DENSITY,
                                        _PLATE_WIDTH) for number in (1, 2)]
        errors['table'] = max(errors['table'], *(abs(value / reference - 1.0) for value, reference in zip(measured, table)))
        errors['exact'] = max(errors['exact'], *(abs(value / reference - 1.0) for value, reference in zip(measured, exact)))
        details[f'k_y = {wavenumber:g}'] = {'measured_hz': measured, 'exact_hz': exact, 'table_hz': list(table)}
    return scaled_worst_result(errors, {'table': _PLATE_TABLE_TOLERANCE, 'exact': _PLATE_EXACT_TOLERANCE}, **details)


# ---- membrane_plate_supported_rectangle: the a x b rectangle simply supported (w = 0, moment free) with D = 2e-6 N m
# (bending ~ 6x the tension at the lowest modes): f_mn = sqrt((T Q + D Q^2) / rho_s) / 2 pi, Q = pi^2 ((m / a)^2 + (n /
# b)^2), exact for any Poisson ratio (M_nn = 0 on the edges). C0 interior penalty on P2: eigenvalue error O(h^2), so h =
# a / 30 and a / 60 are compared too (observed rate >= 1.5). First-estimate tolerances.
_SUPPORTED_RIGIDITY = float32_value(2e-6)
_SUPPORTED_MODES = 4
_SUPPORTED_DIVISIONS = {'coarse': 30, 'fine': 60}
_SUPPORTED_TOLERANCE = 5e-3
_SUPPORTED_MIN_RATE = 1.5


def _supported_specs():
    size_x, size_y = _RECTANGLE
    material = {**_FILM, 'bending_stiffness': _SUPPORTED_RIGIDITY, 'plate_poisson_ratio': float32_value(0.25)}
    return {name: ModelSpec(
        title=f'Self-test: simply supported tensioned plate rectangle (h = a / {divisions})',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        materials=[MaterialSpec('plate', properties=dict(material), selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=[
            FeatureSpec('edges', 'simply_supported', ['x_lo', 'x_hi', 'y_lo', 'y_hi']),
            FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[_triangles(size_x / divisions)], solver_defaults=cpu_defaults(),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _SUPPORTED_MODES}))
        for name, divisions in _SUPPORTED_DIVISIONS.items()}


def _supported_frequencies():
    size_x, size_y = _RECTANGLE
    values = [_plate_frequency(math.pi ** 2 * ((m / size_x) ** 2 + (n / size_y) ** 2), _SUPPORTED_RIGIDITY, _TENSION,
                               _AREAL_DENSITY) for m in range(1, 7) for n in range(1, 7)]
    return sorted(values)[:_SUPPORTED_MODES]


def _check_supported(results_by_name):
    expected = _supported_frequencies()
    measured = {name: _table_frequencies(results_by_name[name], _SUPPORTED_MODES) for name in _SUPPORTED_DIVISIONS}
    error = {name: max(abs(value / reference - 1.0) for value, reference in zip(values, expected))
             for name, values in measured.items()}
    rate = math.log(error['coarse'] / max(error['fine'], 1e-300)) / math.log(
        _SUPPORTED_DIVISIONS['fine'] / _SUPPORTED_DIVISIONS['coarse'])
    errors = {'frequency': error['fine'], 'rate_shortfall': max(_SUPPORTED_MIN_RATE - rate, 0.0)}
    return scaled_worst_result(errors, {'frequency': _SUPPORTED_TOLERANCE, 'rate_shortfall': 1e-12},
                               expected_hz=expected, measured_hz=measured, relative_errors=error, observed_rate=rate)


# ---- membrane_plate_mirror_half: the supported plate rectangle cut at x = a / 2 (x_lo, y_lo, y_hi simply supported on
# a / 2 x b, h = a / 60 as the fine run there), x_hi a mirror plane: symmetry (dw/dn = 0, w free) gives the modes of the
# full rectangle with m odd, antisymmetry (w = 0, M_nn = 0) those with m even (sin(m pi x / a) is even / odd about a /
# 2). Plus the 1D clamped plate strip of width W cut at W / 2 (Hermite, h as membrane_plate_strip_dispersion): symmetry
# (w' = 0) gives the roots n odd, antisymmetry (w = 0) n even. Same tolerances as the full models.
_MIRROR_MODES = 3
_MIRROR_FEATURES = {'even': 'symmetry', 'odd': 'antisymmetry'}
_MIRROR_STRIP_MODES = 2


def _mirror_specs():
    size_x, size_y = _RECTANGLE
    plate = {**_FILM, 'bending_stiffness': _SUPPORTED_RIGIDITY, 'plate_poisson_ratio': float32_value(0.25)}
    strip = {'tension': _PLATE_TENSION, 'areal_density': _PLATE_AREAL_DENSITY, 'bending_stiffness': _PLATE_RIGIDITY}
    specs = {}
    for parity, mirror in _MIRROR_FEATURES.items():
        specs[f'rectangle_{parity}'] = ModelSpec(
            title=f'Self-test: half supported plate rectangle, {mirror} plane on x_hi',
            domain=DomainSpec(size_x=0.5 * size_x, size_y=size_y),
            materials=[MaterialSpec('plate', properties=dict(plate), selection=['all'])],
            physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=[
                FeatureSpec('edges', 'simply_supported', ['x_lo', 'y_lo', 'y_hi']),
                FeatureSpec('mirror', mirror, ['x_hi']), FeatureSpec(_TABLE, 'resonance_table')])],
            meshes=[_triangles(size_x / _SUPPORTED_DIVISIONS['fine'])], solver_defaults=cpu_defaults(),
            study=StudySpec(type='eigenfrequency', settings={'n_resonances': _MIRROR_MODES}))
        specs[f'strip_{parity}'] = ModelSpec(
            title=f'Self-test: half 1D clamped plate strip, {mirror} end on x_hi',
            domain=DomainSpec(size_x=0.5 * _PLATE_WIDTH, size_y=None),
            materials=[MaterialSpec('nitride_plate', properties=dict(strip), selection=['all'])],
            physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=[
                FeatureSpec('end', 'clamped', ['x_lo']), FeatureSpec('mirror', mirror, ['x_hi']),
                FeatureSpec(_TABLE, 'resonance_table')])],
            meshes=[MeshSpec(mode='uniform', dx=_PLATE_WIDTH / _PLATE_CELLS)], solver_defaults=cpu_defaults(),
            study=StudySpec(type='eigenfrequency', settings={'n_resonances': _MIRROR_STRIP_MODES}))
    return specs


def _mirror_rectangle_frequencies(parity):
    """Lowest _MIRROR_MODES f_mn of the full supported rectangle with m odd ('even') or m even ('odd')."""
    size_x, size_y = _RECTANGLE
    first = 1 if parity == 'even' else 2
    values = [_plate_frequency(math.pi ** 2 * ((m / size_x) ** 2 + (n / size_y) ** 2), _SUPPORTED_RIGIDITY, _TENSION,
                               _AREAL_DENSITY) for m in range(first, 13, 2) for n in range(1, 7)]
    return sorted(values)[:_MIRROR_MODES]


def _check_mirror(results_by_name):
    errors, details = {}, {}
    for parity in _MIRROR_FEATURES:
        expected = _mirror_rectangle_frequencies(parity)
        measured = _table_frequencies(results_by_name[f'rectangle_{parity}'], _MIRROR_MODES)
        errors[f'rectangle_{parity}'] = max(abs(value / reference - 1.0) for value, reference in zip(measured, expected))
        details[f'rectangle_{parity}'] = {'measured_hz': measured, 'expected_hz': expected}
        first = 1 if parity == 'even' else 2
        exact = [_plate_strip_frequency(number, 0.0, _PLATE_RIGIDITY, _PLATE_TENSION, _PLATE_AREAL_DENSITY,
                                        _PLATE_WIDTH) for number in range(first, first + 2 * _MIRROR_STRIP_MODES, 2)]
        measured = _table_frequencies(results_by_name[f'strip_{parity}'], _MIRROR_STRIP_MODES)
        errors[f'strip_{parity}'] = max(abs(value / reference - 1.0) for value, reference in zip(measured, exact))
        details[f'strip_{parity}'] = {'measured_hz': measured, 'exact_hz': exact}
    tolerances = {name: _SUPPORTED_TOLERANCE if name.startswith('rectangle') else _PLATE_EXACT_TOLERANCE
                  for name in errors}
    return scaled_worst_result(errors, tolerances, **details)


# ---- membrane_plate_strip_bloch: a W x L Bloch cell periodic along y (bloch_ky = k), clamped on x_lo / x_hi, D = T
# (W / 10)^2: the 2D interior penalty plate (bending seam terms across the periodic sides, complex Hermitian pencil for k
# != 0) against the exact plate strip roots n = 1..3 at k = 0 and k = 3e4 rad/m. h = W / 100 (boundary layer W / 10).
_BLOCH_WIDTH = 100e-6
_BLOCH_PERIOD = 10e-6
_BLOCH_RIGIDITY = float32_value(5e-9)
_BLOCH_WAVENUMBERS = (0.0, 3e4)
_BLOCH_MODES = 3
_BLOCH_TOLERANCE = 2e-3


def _bloch_specs():
    material = {**_FILM, 'bending_stiffness': _BLOCH_RIGIDITY}
    return {f'ky_{index}': ModelSpec(
        title=f'Self-test: clamped plate strip Bloch cell, k_y = {wavenumber:g} rad/m',
        domain=DomainSpec(size_x=_BLOCH_WIDTH, size_y=_BLOCH_PERIOD),
        materials=[MaterialSpec('plate', properties=dict(material), selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, settings={'periodic_y': True, 'bloch_ky': wavenumber},
                             features=[FeatureSpec('walls', 'clamped', ['x_lo', 'x_hi']),
                                       FeatureSpec(_TABLE, 'resonance_table')])],
        meshes=[_triangles(_BLOCH_WIDTH / 100.0)], solver_defaults=cpu_defaults(),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _BLOCH_MODES}))
        for index, wavenumber in enumerate(_BLOCH_WAVENUMBERS)}


def _check_bloch(results_by_name):
    errors, details = {}, {}
    for index, wavenumber in enumerate(_BLOCH_WAVENUMBERS):
        measured = _table_frequencies(results_by_name[f'ky_{index}'], _BLOCH_MODES)
        exact = [_plate_strip_frequency(number, wavenumber, _BLOCH_RIGIDITY, _TENSION, _AREAL_DENSITY, _BLOCH_WIDTH)
                 for number in range(1, _BLOCH_MODES + 1)]
        errors[f'k_y = {wavenumber:g}'] = max(abs(value / reference - 1.0) for value, reference in zip(measured, exact))
        details[f'k_y = {wavenumber:g}'] = {'measured_hz': measured, 'exact_hz': exact}
    return scaled_worst_result(errors, {name: _BLOCH_TOLERANCE for name in errors}, **details)


# ---- membrane_port_tunnelling (mauranyapin2021 plan, E3 / E4): guide W1 = 78 um, barrier W2 = 44 um centred, L_b = 75
# um, c = 502.6 m/s, feeds of 30 um to the ports (10 modes each: the DtN is exact for them). 5.5 MHz, incident mode 1:
# T11 = 0.0493 (2 %, mode matching), T + R = 1 (exact discrete power balance) and the mode_projection of mode 1 in the
# input feed against A exp(-j beta y) + b exp(+j beta y) of the port amplitudes; 8.1 MHz, incident mode 2: T(2 -> 2) =
# 4.50e-8 within 1 dB. h = 1 um quadratic triangles (re-entrant clamped corners: first estimates).
_TUNNEL_GUIDE = 78e-6
_TUNNEL_BARRIER = 44e-6
_TUNNEL_LENGTH = 75e-6
_TUNNEL_FEED = 30e-6
_TUNNEL_TENSION = float32_value(80.0)
_TUNNEL_DENSITY = float32_value(80.0 / 502.6 ** 2)
_TUNNEL_H = 1e-6
_TUNNEL_AMPLITUDE = 1e-9
_TUNNEL_CASES = {'t11': (5.5e6, 1, 0.0493), 'filter': (8.1e6, 2, 4.50e-8)}   # frequency, incident mode, T(m -> m)
_TUNNEL_T11_TOLERANCE = 2e-2
_TUNNEL_BALANCE_TOLERANCE = 1e-6
_TUNNEL_FILTER_DB = 1.0
_TUNNEL_PROJECTION_TOLERANCE = 1e-3
_TUNNEL_PROJECTION = 'feed_modes'
_TUNNEL_POWER = 'feed_power'               # the same projection with result_axis 'frequency'
_TUNNEL_AVERAGE = (6e-6, 24e-6)            # its averaging window: 10 of the 14 positions (2 um apart from 2 um)
_TUNNEL_AVERAGE_POINTS = 10
_TUNNEL_POWER_TOLERANCE = 1e-12
_TUNNEL_HALF_TOLERANCE = 5e-3   # T11 of the half model vs the full one (different meshes; first estimate)


def _tunnel_spec(name, half=False):
    """The tunnelling model; half: x from 0 to W1 / 2 with the even mirror (symmetry) on x_hi, free-wall port profiles."""
    frequency, mode, _ = _TUNNEL_CASES[name]
    notch = 0.5 * (_TUNNEL_GUIDE - _TUNNEL_BARRIER)
    top = _TUNNEL_FEED + _TUNNEL_LENGTH
    width = 0.5 * _TUNNEL_GUIDE if half else _TUNNEL_GUIDE
    notch_boxes = {'left': {'x0': 0.0, 'y0': _TUNNEL_FEED, 'x1': notch, 'y1': top}}
    if not half:
        notch_boxes['right'] = {'x0': _TUNNEL_GUIDE - notch, 'y0': _TUNNEL_FEED, 'x1': _TUNNEL_GUIDE, 'y1': top}
    # the guide subtracts copies of the notches (composite inputs are consumed), so the notches stay selectable walls
    geometry = [GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': width, 'y1': top + _TUNNEL_FEED})]
    geometry += [GeometrySpec(f'notch_{side}{suffix}', 'rectangle', dict(box))
                 for side, box in notch_boxes.items() for suffix in ('', '_cut')]
    geometry.append(GeometrySpec('guide', 'difference',
                                 {'base': 'box', 'subtract': [f'notch_{side}_cut' for side in notch_boxes]}))
    walls = ['x_lo'] + ([] if half else ['x_hi']) + [f'notch_{side}' for side in notch_boxes]
    features = [FeatureSpec('walls', 'clamped', walls),
                FeatureSpec('in_port', 'port', ['y_lo'], {'mode_count': 10, 'incident_mode': mode,
                                                         'amplitude': _TUNNEL_AMPLITUDE}),
                FeatureSpec('out_port', 'port', ['y_hi'], {'mode_count': 10})]
    if half:
        features.append(FeatureSpec('mirror', 'symmetry', ['x_hi']))
    if name == 't11':
        projection = {'guide_axis': 'y', 'cross_start': 0.0, 'cross_end': width, 'start': 2e-6,
                      'end': _TUNNEL_FEED - 2e-6, 'points': 14, 'mode_count': 2}
        features.append(FeatureSpec(_TUNNEL_PROJECTION, 'mode_projection', [], dict(projection)))
        features.append(FeatureSpec(_TUNNEL_POWER, 'mode_projection', [], {
            **projection, 'result_axis': 'frequency', 'average_start': _TUNNEL_AVERAGE[0],
            'average_end': _TUNNEL_AVERAGE[1]}))
    return ModelSpec(
        title=f'Self-test: membrane port tunnelling ({name}{", half" if half else ""})',
        domain=DomainSpec(size_x=width, size_y=top + _TUNNEL_FEED), geometry=geometry,
        materials=[MaterialSpec('film', properties={'tension': _TUNNEL_TENSION, 'areal_density': _TUNNEL_DENSITY},
                                selection=['guide'])],
        physics=[PhysicsSpec(_PHYSICS, _MEMBRANE_TYPE, features=features)],
        meshes=[_triangles(_TUNNEL_H)], solver_defaults=cpu_defaults(),
        study=StudySpec(type='frequency_response',   # 'filter' also runs the study without snapshots
                        settings={'frequencies': [frequency], 'snapshots': 'all' if name == 't11' else 'none'}))


def _tunnel_specs():
    specs = {name: _tunnel_spec(name) for name in _TUNNEL_CASES}
    specs['t11_half'] = _tunnel_spec('t11', half=True)
    return specs


def _port_series(results, port, name):
    return float(np.asarray(results.result(_PHYSICS, port)['series'][name], dtype=np.float64)[0])


def _t11_errors(results, suffix=''):
    """({error name: value}, {detail name: value}) of one t11 run: T11 vs mode matching, the power balance, the mode-1
    projection in the input feed vs the port amplitudes and its frequency record."""
    errors, details = {}, {}
    transmission = _port_series(results, 'out_port', 'power_fraction_1')
    balance = sum(_port_series(results, port, f'power_fraction_{number}') for port in ('in_port', 'out_port')
                  for number in range(1, 11))
    errors[f't11{suffix}'] = abs(transmission / _TUNNEL_CASES['t11'][2] - 1.0)
    errors[f'power_balance{suffix}'] = abs(balance - 1.0)
    in_port = results.result(_PHYSICS, 'in_port')['series']
    reflected = complex(in_port['mode_1_re'][0], in_port['mode_1_im'][0])
    omega = 2.0 * math.pi * _TUNNEL_CASES['t11'][0]
    beta = math.sqrt(_TUNNEL_DENSITY * omega ** 2 / _TUNNEL_TENSION - (math.pi / _TUNNEL_GUIDE) ** 2)
    projection = results.result(_PHYSICS, _TUNNEL_PROJECTION)
    positions = np.asarray(projection['x'], dtype=np.float64)
    measured = (np.asarray(projection['series']['mode_1_re'], dtype=np.float64)
                + 1j * np.asarray(projection['series']['mode_1_im'], dtype=np.float64))
    expected = _TUNNEL_AMPLITUDE * np.exp(-1j * beta * positions) + reflected * np.exp(1j * beta * positions)
    errors[f'projection{suffix}'] = float(np.max(np.abs(measured - expected))) / _TUNNEL_AMPLITUDE
    frequency_series = projection['frequency_series']   # the per-frequency record: its one row = the shown profile
    recorded = (np.asarray(frequency_series['mode_1_re'], dtype=np.float64)[-1]
                + 1j * np.asarray(frequency_series['mode_1_im'], dtype=np.float64)[-1])
    errors[f'projection_record{suffix}'] = float(np.max(np.abs(recorded - measured))) / _TUNNEL_AMPLITUDE
    power_errors, power_details = _feed_power_errors(results, positions, frequency_series)
    errors.update({f'{item}{suffix}': value for item, value in power_errors.items()})
    details.update({f't11{suffix}': transmission, f'power_balance{suffix}': balance,
                    f'reflected_mode_1{suffix}': [reflected.real, reflected.imag], f'feed_power{suffix}': power_details})
    return errors, details


def _feed_power_errors(results, positions, frequency_series):
    """result_axis 'frequency' of the feed projection against the window mean of |a_n|^2 of its frequency_series rows
    (relative); the window point count as an error count."""
    power = results.result(_PHYSICS, _TUNNEL_POWER)
    inside = (positions >= _TUNNEL_AVERAGE[0] - 1e-12) & (positions <= _TUNNEL_AVERAGE[1] + 1e-12)
    expected = {}
    for number in (1, 2):
        rows = (np.asarray(frequency_series[f'mode_{number}_re'], dtype=np.float64)
                + 1j * np.asarray(frequency_series[f'mode_{number}_im'], dtype=np.float64))
        expected[f'mode_{number}_power'] = np.mean(np.abs(rows[:, inside]) ** 2, axis=1)
    expected['total_power'] = expected['mode_1_power'] + expected['mode_2_power']
    expected['mode_1_rms'] = np.sqrt(expected['mode_1_power'])
    scale = float(np.max(expected['total_power'])) or 1.0
    worst = max(float(np.max(np.abs(np.asarray(power['series'][name], dtype=np.float64) - values)))
                / (math.sqrt(scale) if name.endswith('_rms') else scale) for name, values in expected.items())
    frequencies = np.asarray(power['x'], dtype=np.float64)
    errors = {'feed_power': worst,
              'feed_power_axis': float(abs(frequencies[-1] / _TUNNEL_CASES['t11'][0] - 1.0)),
              'feed_power_points': float(abs(int(power['average_points']) - _TUNNEL_AVERAGE_POINTS)
                                         + abs(int(inside.sum()) - _TUNNEL_AVERAGE_POINTS))}
    return errors, {'mode_1_power': expected['mode_1_power'].tolist(), 'average_window': power['average_window'],
                    'average_points': power['average_points']}


def _check_tunnel(results_by_name):
    errors, details = _t11_errors(results_by_name['t11'])
    half = results_by_name['t11_half']
    half_errors, half_details = _t11_errors(half, '_half')
    errors.update(half_errors)
    details.update(half_details)
    errors['half_vs_full'] = abs(details['t11_half'] / details['t11'] - 1.0)
    walls = {feature: half.result(_PHYSICS, feature).get('free_walls')
             for feature in ('in_port', 'out_port', _TUNNEL_PROJECTION)}
    errors['half_free_walls'] = float(sum(list(value or ()) != [False, True] for value in walls.values()))
    filtered = _port_series(results_by_name['filter'], 'out_port', 'power_fraction_2')
    errors['filter_db'] = abs(10.0 * math.log10(max(filtered, 1e-300) / _TUNNEL_CASES['filter'][2]))
    details.update(half_free_walls=walls, filter_t22=filtered,
                   filter_t22_db=10.0 * math.log10(max(filtered, 1e-300)))
    tolerances = {'filter_db': _TUNNEL_FILTER_DB, 'half_vs_full': _TUNNEL_HALF_TOLERANCE, 'half_free_walls': 0.5}
    for suffix in ('', '_half'):
        tolerances.update({f't11{suffix}': _TUNNEL_T11_TOLERANCE, f'power_balance{suffix}': _TUNNEL_BALANCE_TOLERANCE,
                           f'projection{suffix}': _TUNNEL_PROJECTION_TOLERANCE,
                           f'projection_record{suffix}': _TUNNEL_PROJECTION_TOLERANCE,
                           f'feed_power{suffix}': _TUNNEL_POWER_TOLERANCE,
                           f'feed_power_axis{suffix}': _TUNNEL_POWER_TOLERANCE, f'feed_power_points{suffix}': 0.5})
    return scaled_worst_result(errors, tolerances, **details)


# ---- membrane_emfdfd_rectangle (plan S4 cross-check): the template rectangle on em_fdfd_ez (vacuum, PEC walls,
# quadratic, the same uniform triangles): both solve -lap u = (omega / c)^2 u with Dirichlet walls on one mesh, so f_n / c
# agree to the eigen-solve accuracy (1e-6 relative). The EM target wavelength lies 20 % above the lowest mode's.
_EM_PHYSICS = 'em'
_EM_TABLE = 'resonances'
_EM_TOLERANCE = 1e-6


def _em_rectangle_spec():
    size_x, size_y = _RECTANGLE
    lowest = 0.5 * C0 * math.hypot(1.0 / size_x, 1.0 / size_y)
    settings = {'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec',
                'wavelength': 1.2 * C0 / lowest, 'discretization': 'quadratic'}
    return ModelSpec(
        title='Self-test: em_fdfd_ez analogue of the membrane rectangle', domain=DomainSpec(size_x=size_x, size_y=size_y),
        physics=[PhysicsSpec(_EM_PHYSICS, 'em_fdfd_ez', settings=settings,
                             features=[FeatureSpec(_EM_TABLE, 'resonance_table')])],
        meshes=[_triangles(_RECTANGLE_H)], solver_defaults=cpu_defaults(),
        study=StudySpec(type='eigenfrequency', settings={'n_resonances': _RECTANGLE_MODES}))


def _check_em_rectangle(results_by_name):
    membrane = _table_frequencies(results_by_name['membrane'], _RECTANGLE_MODES)
    table = results_by_name['em'].result(_EM_PHYSICS, _EM_TABLE)
    column = list(table['columns']).index('frequency')
    rows = table['rows']
    if len(rows) < _RECTANGLE_MODES:
        raise ValueError(f'expected {_RECTANGLE_MODES} em resonances, got {len(rows)}')
    electromagnetic = sorted(float(row[column]) for row in rows)[:_RECTANGLE_MODES]
    ratios = [(value / C0) / (reference / _wave_speed()) for value, reference in zip(electromagnetic, membrane)]
    analytic = _check_rectangle({'rectangle': results_by_name['membrane']})['details']
    errors = {'scaled_frequency': max(abs(ratio - 1.0) for ratio in ratios), **analytic['errors']}
    return scaled_worst_result(errors, {'scaled_frequency': _EM_TOLERANCE, **analytic['tolerances']},
                               membrane_hz=membrane, em_hz=electromagnetic, ratios=ratios,
                               frequency_expected=analytic['frequency_expected'],
                               snapshot_labels=analytic['snapshot_labels'])


MEMBRANE_SELFTEST_CASES = (
    SelfTestCase(
        'membrane_drum_modes', 'Membrane: circular drum (Bessel zeros, curved outline)',
        'A circle of radius 0.5 mm clamped on its outline (isoparametric P2 on the curved edges, h = R / 20), the '
        'square around it a hole (no material: excluded elements): the six lowest frequencies against j_mn c / (2 pi '
        f'R) with j_01, j_11 (twice), j_21 (twice), j_02, to {_DRUM_TOLERANCE:g} relative (first estimate).',
        lambda: {'drum': _drum_spec()}, _check_drum),
    SelfTestCase(
        'membrane_line_clamp_strips', 'Membrane: clamped line across a rectangle (two clamped rectangles)',
        f'The clamped template rectangle with a clamped_line from (a1, 0) to (a1, b), a1 = {_IRIS_X * 1e3:g} mm, on '
        f'quadratic triangles h = a / 80: the {_IRIS_MODES} lowest frequencies against the f_mn of the clamped '
        f'rectangles a1 x b and (a - a1) x b together, to {_IRIS_TOLERANCE:g} relative (first estimate).',
        lambda: {'iris': _iris_spec()}, _check_iris),
    SelfTestCase(
        'membrane_line_dispersion', 'Membrane: 1D strip dispersion with an axial wavenumber',
        f'A 1D strip of width W clamped at both ends ({_STRIP_CELLS} P2 line cells) with axial_wavenumber k_y = 1.5 pi '
        '/ W: the four lowest frequencies against c sqrt((n pi / W)^2 + k_y^2) / (2 pi) to '
        f'{_STRIP_TOLERANCE:g} relative (the k_y^2 term is exact in the discretisation).',
        lambda: {'strip': _strip_spec()}, _check_strip),
    SelfTestCase(
        'membrane_line_loss', 'Membrane: lossy string (driven response, Lorentzian Q, complex resonance)',
        f'A clamped 1D string with loss_factor eta = 1e-3 under uniform pressure: the probe phasor at L / 2 over '
        f'{_STRING_POINTS} frequencies around f_1 against p / (rho_s omega^2) (1 / cos(k L / 2) - 1), k = omega '
        f'sqrt(rho_s / (T (1 + j eta))) ({_STRING_RESPONSE_TOLERANCE:g} of the peak), a lorentz_fit of |w|^2 against Q '
        f'= 1 / eta ({_STRING_FIT_TOLERANCE:g}: line shape and background), and the eigenfrequency resonance (complex '
        f'pencil) against Q = Re s / (2 Im s), s = sqrt(1 + j eta), and f_1 Re s ({_STRING_EIGEN_TOLERANCE:g}).',
        _string_specs, _check_string),
    SelfTestCase(
        'membrane_guide_evanescent', 'Membrane: evanescent decay below cut-off (P2 triangles)',
        'A strip guide of width W (clamped walls, clamped far end at 5 W) driven by a line force on x_lo at half the '
        'first cut-off: |w(2.5 W) / w(1.5 W)| on the centre line against sinh(kappa (L - x)) of mode 1, kappa = (pi / '
        f'W) sqrt(1 - 0.25), to {_EVANESCENT_TOLERANCE:g} relative (first estimate).',
        _evanescent_specs, _check_evanescent),
    SelfTestCase(
        'membrane_guide_pml', 'Membrane: PML-terminated guide reflection',
        'The guide with a PML ring (12 h_max, reflection 1e-6 at the drive frequency) on x_hi, driven at 1.5 f_c (one '
        'propagating mode): a two-wave least-squares fit of four centre-line probes, |A_b / A_f| below '
        f'{_PML_REFLECTION_LIMIT:g} (-40 dB) and the two-wave residual below 1e-2. Variant gpu: the same on the graphics '
        'card (cuDSS device pencil, device field publish).',
        _pml_specs, _check_pml, variants=(SelfTestVariant('gpu', tolerance=1.0, backend='gpu'),)),
    SelfTestCase(
        'membrane_plate_strip_dispersion', 'Membrane: 1D clamped plate strip (Hermite, bending + tension)',
        f'R-E1B: a 1D strip W = 76 um of 80 nm silicon nitride at 1 GPa (T = 80 N/m, D = E h^3 / 12 (1 - nu^2)), clamped '
        f'(w and w\') at both ends, {_PLATE_CELLS} Hermite cubic cells, k_y = 0 and 5e4 rad/m: f_1, f_2 against the '
        f'romero2019 plan table ({_PLATE_TABLE_TOLERANCE:g}) and the exact clamped plate strip roots '
        f'({_PLATE_EXACT_TOLERANCE:g}; first estimate).',
        _plate_strip_specs, _check_plate_strip),
    SelfTestCase(
        'membrane_plate_supported_rectangle', 'Membrane: simply supported plate rectangle (interior penalty P2)',
        'The 1 x 0.7 mm rectangle with bending_stiffness 2e-6 N m and plate_poisson_ratio 0.25, simply supported on its '
        f'frame: the {_SUPPORTED_MODES} lowest frequencies against sqrt((T Q + D Q^2) / rho_s) / 2 pi, Q = pi^2 ((m / a)^2 '
        f'+ (n / b)^2), on h = a / 60 ({_SUPPORTED_TOLERANCE:g}) with an observed rate >= {_SUPPORTED_MIN_RATE:g} from h = '
        'a / 30 (eigenvalues O(h^2); first estimates).',
        _supported_specs, _check_supported),
    SelfTestCase(
        'membrane_plate_mirror_half', 'Membrane: half plate models with symmetry / antisymmetry planes',
        'The simply supported plate rectangle of membrane_plate_supported_rectangle cut at x = a / 2 (h = a / 60), x_hi a '
        f'symmetry plane (slope held, w free) or an antisymmetry plane (w held): the {_MIRROR_MODES} lowest frequencies '
        f'against the full rectangle modes with m odd / m even ({_SUPPORTED_TOLERANCE:g}); the 1D clamped plate strip W '
        '= 76 um cut at W / 2 (Hermite): the two lowest against the exact roots n = 1, 3 / n = 2, 4 '
        f'({_PLATE_EXACT_TOLERANCE:g}). First estimates.',
        _mirror_specs, _check_mirror),
    SelfTestCase(
        'membrane_plate_strip_bloch', 'Membrane: clamped plate strip Bloch cell (bending across the periodic seam)',
        'A 100 x 10 um cell periodic along y (bloch_ky 0 and 3e4 rad/m: real and complex Hermitian pencils), clamped on '
        'x_lo / x_hi, D = T (W / 10)^2, quadratic triangles h = W / 100: the three lowest frequencies against the exact '
        f'clamped plate strip roots at k_y, {_BLOCH_TOLERANCE:g} relative (first estimate).',
        _bloch_specs, _check_bloch),
    SelfTestCase(
        'membrane_port_tunnelling', 'Membrane: port tunnelling and mode filter (DtN ports)',
        'M-E3 / M-E4: guide 78 um, centred barrier 44 um x 75 um (clamped steps), c = 502.6 m/s, ports with 10 modes 30 '
        f'um from the junctions: T(1 -> 1) at 5.5 MHz against 0.0493 ({_TUNNEL_T11_TOLERANCE:g}), the outgoing powers of '
        f'both ports summing to the incident one ({_TUNNEL_BALANCE_TOLERANCE:g}), the mode_projection of mode 1 in the input '
        f'feed against the port amplitudes ({_TUNNEL_PROJECTION_TOLERANCE:g} of A) and its result_axis frequency view '
        f'against the window mean of |a_n|^2 of the record ({_TUNNEL_POWER_TOLERANCE:g}); T(2 -> 2) at 8.1 MHz against 4.50e-8 '
        f'within {_TUNNEL_FILTER_DB:g} dB. The t11 run again as a half model (x_hi a symmetry plane at W1 / 2: free-wall '
        'port and projection profiles sin((n - 1/2) pi s / W)): the same checks, free_walls [False, True] on both ports '
        f'and the projection, T11 against the full model ({_TUNNEL_HALF_TOLERANCE:g}). h = 1 um quadratic (first '
        'estimates).',
        _tunnel_specs, _check_tunnel),
    SelfTestCase(
        'membrane_emfdfd_rectangle', 'Membrane: clamped rectangle modes and their em_fdfd_ez cross-check',
        f'The template rectangle (clamped {_RECTANGLE[0] * 1e3:g} x {_RECTANGLE[1] * 1e3:g} mm, T = 50 N/m, rho_s = '
        '1.55e-4 kg/m^2, quadratic triangles h = a / 80): the six lowest modes against f_mn = (c / 2) sqrt((m / a)^2 + '
        f'(n / b)^2), c = sqrt(T / rho_s), to {_RECTANGLE_TOLERANCE:g} relative (P2 eigenvalue error ~ (k h)^4) with one '
        'snapshot per mode; and against the same rectangle on em_fdfd_ez (vacuum, PEC walls, quadratic, the same '
        f'triangles): f_n / c of both agree to {_EM_TOLERANCE:g} relative (one discrete operator; plan S4 cross-check).',
        lambda: {'membrane': rectangle_modes_spec('cpu'), 'em': _em_rectangle_spec()}, _check_em_rectangle),
)
