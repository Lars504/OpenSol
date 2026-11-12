"""Self-tests of the frequency_response study of solid_mechanics (piezoelectric plan stage 5): the Butterworth-Van Dyke
fit round trip and its kappa_e extra (CPU) and the piezoelectric / elastic bar driven harmonically with losses against
the dense 1D linear-element system (exact: the y-uniform Q1 solution); no physics imports outside the checks."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float32_value, scaled_worst_result, worst_result
from ...model.spec import DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec
from .selftests_piezo import (_CELL, _CELLS, _DENSITY, _EPS_DC, _LENGTH, _PIEZO_E11, _POISSON_RATIO, _WIDTH,
                              _YOUNGS_MODULUS, _chain_frequencies, _moduli, _short_frequencies)

_PHYSICS = 'solid'
_SOLID_TYPE = 'solid_mechanics'       # = selftests.SOLID_TYPE_NAME
_STUDY = 'frequency_response'         # = model.study_frequency_response.STUDY_NAME

# ---- 1. solid_bvd_fit_round_trip (CPU) ----
_BVD = {'C0': 2.0e-12, 'C1': 9.0e-14, 'L1': 1.0e-4, 'R1': 15.0, 'tan_delta_0': 2e-3}
_ROUND_TRIP_TOLERANCE = 1e-7


def _bvd_band():
    """(f_s, f_p, 301 frequencies from 0.97 f_s to f_p + 0.03 f_s) [Hz] of the _BVD circuit."""
    f_s = 1.0 / (2.0 * math.pi * math.sqrt(_BVD['L1'] * _BVD['C1']))
    f_p = f_s * math.sqrt(1.0 + _BVD['C1'] / _BVD['C0'])
    return f_s, f_p, np.linspace(0.97 * f_s, f_p + 0.03 * f_s, 301)


def _check_bvd(results_by_name):
    """bvd_fit of the exact BVD admittance on 301 frequencies around the resonance recovers C0, C1, L1, R1, tan_delta_0
    and f_s / f_p / k_eff2; a 4-point band reports NaN and a fit_warning."""
    from .bvd import bvd_admittance, bvd_fit
    f_s, f_p, frequencies = _bvd_band()
    fit = bvd_fit(frequencies, bvd_admittance(frequencies, _BVD))
    errors = {name: abs(fit[name] / value - 1.0) for name, value in _BVD.items()}
    errors.update(f_s=abs(fit['f_s'] / f_s - 1.0), f_p=abs(fit['f_p'] / f_p - 1.0),
                  k_eff2=abs(fit['k_eff2'] / (_BVD['C1'] / (_BVD['C0'] + _BVD['C1'])) - 1.0),
                  fit_error=fit['fit_error'])
    narrow = bvd_fit(frequencies[:4], bvd_admittance(frequencies[:4], _BVD))
    errors['narrow band flagged'] = 0.0 if (math.isnan(narrow['C0']) and 'fit_warning' in narrow) else 1.0
    return worst_result(errors, _ROUND_TRIP_TOLERANCE, fit={key: fit[key] for key in fit if key != 'fit_warning'},
                        f_s=f_s, f_p=f_p)


# ---- 2. solid_bvd_kappa_e (CPU): kappa_e = Z / L1 through admittance_result ----
_LOAD_IMPEDANCE = 50.0   # Ohm


def _check_kappa(results_by_name):
    """admittance_result of exact BVD points (V = 1, Q = Y / (j omega)) with depth 1 (a 3D model): kappa_e = Z / L1 and
    omega_C0_Z = 2 pi f_s C0 Z of the fitted circuit; per m of depth (thickness None) both NaN with a kappa_e_warning."""
    from .bvd import bvd_admittance
    from .frequency_response import AdmittanceMonitor, FrequencyPoint, admittance_result
    f_s, _, frequencies = _bvd_band()
    admittance = bvd_admittance(frequencies, _BVD)
    points = [FrequencyPoint(float(frequency), {'drive': (1.0, complex(value / (2j * math.pi * frequency)))})
              for frequency, value in zip(frequencies, admittance)]
    monitor = AdmittanceMonitor('admittance', 'drive', None, None, _LOAD_IMPEDANCE)
    device = admittance_result(monitor, 'drive', points, 1.0)
    expected = {'kappa_e': _LOAD_IMPEDANCE / _BVD['L1'],
                'omega_C0_Z': 2.0 * math.pi * f_s * _BVD['C0'] * _LOAD_IMPEDANCE}
    errors = {name: abs(device[name] / value - 1.0) for name, value in expected.items()}
    per_depth = admittance_result(monitor, 'drive', points, None)
    errors['per-depth flagged'] = 0.0 if (math.isnan(per_depth['kappa_e']) and 'kappa_e_warning' in per_depth) \
        else 1.0
    return worst_result(errors, _ROUND_TRIP_TOLERANCE, expected=expected,
                        measured={name: device[name] for name in expected})


# ---- 3. solid_frequency_response_bar (model runs) ----
# The selftests_piezo bar (10 x 1 mm, 20 x 2 cells; rollers on y_lo / y_hi and x_lo): y-uniform fields, so the 2D Q1
# solution is W x the 1D linear-element one. Runs: 'piezo' (ground x_lo, drive x_hi at _DRIVE V, loss factor and loss
# tangent, admittance feature) on a band around mode 1 vs the dense complex 1D system [[K_uu (1 + j eta) - omega^2 M,
# K_uphi], [K_phiu, -K_phiphi (1 - j tan delta)]] (u_0 = 0, phi_0 = 0, phi_N = V; Q = -(phi row N), Y = j omega W Q /
# V), plus the BVD fit against the lossless short / open mode-1 frequencies; 'elastic' (no piezo: a traction on x_hi,
# loss factor) at three frequencies: Re u_x and u_abs = |u| per snapshot vs the 1D solution; 'heated' (elastic plus a
# uniform temperature_change): the harmonic drive holds no thermal load, so its u_x and stresses equal 'elastic' (D3).
_DRIVE = 10.0                 # V
_LOSS_FACTOR = float32_value(2e-3)    # float32-exact material values (as the selftests_piezo bar)
_LOSS_TANGENT = float32_value(1e-3)
_N_SWEEP = 201                # spacing ~1e-3 f: about 2 points per linewidth eta f at eta = 2e-3
_BAND_MARGIN = 0.05           # x f_short below the resonance and above the antiresonance (a dip above the peak)
_TRACTION = 1e6               # Pa on x_hi (elastic run)
_ELASTIC_FACTORS = (0.5, 0.98, 1.5)   # x the lowest chain frequency of the elastic bar
_ADMITTANCE_TOLERANCE = 1e-8  # relative complex Y (float64 host solves)
_FIELD_TOLERANCE = 1e-5       # float32 outputs, relative to the peak
_BVD_FREQUENCY_TOLERANCE = 5e-3   # the single-mode BVD of a distributed resonator vs its discrete frequencies
_BVD_COUPLING_TOLERANCE = 5e-2
_THERMAL_EXPANSION = float32_value(2.6e-6)   # 1/K (heated run)
_TEMPERATURE_CHANGE = 50.0                   # K: a static thermal stress the confined bar would carry
_HEATED_FIELDS = ('u_x', 'stress_xx', 'stress_yy', 'stress_zz')


def _lossless_mode_one():
    """(short (resonance), open (antiresonance)) [Hz] of the lossless mode 1 of the piezo bar."""
    c11, eps, piezo_e = _moduli()
    stiffened = c11 + piezo_e ** 2 / eps
    return _short_frequencies(stiffened, piezo_e ** 2 / (eps * _LENGTH))[0], _chain_frequencies(stiffened)[0]


def _sweep_band():
    """(start, stop) [Hz] around the lossless short (resonance) and open (antiresonance) mode 1 of the piezo bar."""
    short, open_ = _lossless_mode_one()
    return (1.0 - _BAND_MARGIN) * short, open_ + _BAND_MARGIN * short


def _bar_spec(run):
    properties = {'youngs_modulus': _YOUNGS_MODULUS, 'poisson_ratio': _POISSON_RATIO, 'rho': _DENSITY,
                  'mechanical_loss_factor': _LOSS_FACTOR}
    features = [FeatureSpec('rails', 'roller', ['y_lo', 'y_hi']), FeatureSpec('anchor', 'roller', ['x_lo'])]
    if run == 'heated':
        properties['alpha'] = _THERMAL_EXPANSION
        features.append(FeatureSpec('heating', 'temperature_change', ['all'], {'delta_T': _TEMPERATURE_CHANGE}))
    if run == 'piezo':
        properties.update(piezo_e11=_PIEZO_E11, eps_dc=_EPS_DC, dielectric_loss_tangent=_LOSS_TANGENT)
        features += [FeatureSpec('ground', 'electrode', ['x_lo'], {'connection': 'voltage', 'voltage': 0.0}),
                     FeatureSpec('drive', 'electrode', ['x_hi'], {'connection': 'voltage', 'voltage': _DRIVE}),
                     FeatureSpec('admittance', 'admittance')]
        start, stop = _sweep_band()
        study = {'frequency_start': start, 'frequency_stop': stop, 'n_frequencies': _N_SWEEP}
    else:
        features.append(FeatureSpec('push', 'boundary_load', ['x_hi'], {'t_x': _TRACTION, 't_y': 0.0}))
        first = _chain_frequencies(_moduli()[0])[0]
        study = {'frequencies': [factor * first for factor in _ELASTIC_FACTORS]}
    return ModelSpec(
        title=f'Self-test: bar driven harmonically, {run} (solid mechanics frequency response)',
        domain=DomainSpec(size_x=_LENGTH, size_y=_WIDTH),
        materials=[MaterialSpec('bar', properties=properties, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, _SOLID_TYPE, settings={'piezoelectric': run == 'piezo'}, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type=_STUDY, settings=study, stored_fields=[
            f'{_PHYSICS}.{name}' for name in dict.fromkeys(('u_x', 'u_abs') + _HEATED_FIELDS)]))


def _bar_specs():
    return {run: _bar_spec(run) for run in ('piezo', 'elastic', 'heated')}


def _bar_1d(omega, piezo, load=0.0, voltage=0.0):
    """Complex 1D solution (u nodes 0..N, phi nodes 0..N, Q of node N per unit area) of the bar: linear elements,
    consistent mass, u_0 = 0, phi_0 = 0, phi_N = voltage, load [Pa] on node N."""
    c11, eps, piezo_e = _moduli()
    n = _CELLS[0]
    stiffness = c11 * (1.0 + 1j * _LOSS_FACTOR)
    coupling = piezo_e if piezo else 0.0
    permittivity = eps * (1.0 - 1j * _LOSS_TANGENT)
    unit = np.array([[1.0, -1.0], [-1.0, 1.0]]) / _CELL
    k_uu, k_up, k_pp, mass = (np.zeros((n + 1, n + 1), dtype=np.complex128) for _ in range(4))
    for element in range(n):
        pair = np.ix_((element, element + 1), (element, element + 1))
        k_uu[pair] += stiffness * unit
        k_up[pair] += coupling * unit
        k_pp[pair] += permittivity * unit
        mass[pair] += _DENSITY * _CELL / 6.0 * np.array([[2.0, 1.0], [1.0, 2.0]])
    operator = np.block([[k_uu - omega ** 2 * mass, k_up], [k_up.T, -k_pp]])
    fixed = np.zeros(2 * (n + 1), dtype=np.complex128)
    fixed[2 * n + 1] = voltage   # phi_N
    rhs = np.zeros(2 * (n + 1), dtype=np.complex128)
    rhs[n] = load
    held = [0] + ([n + 1, 2 * n + 1] if piezo else list(range(n + 1, 2 * n + 2)))   # u_0, phi_0, phi_N (or every phi)
    free = [index for index in range(2 * (n + 1)) if index not in held]
    solution = fixed.copy()
    solution[free] = np.linalg.solve(operator[np.ix_(free, free)], (rhs - operator @ fixed)[free])
    product = operator @ solution
    return solution[:n + 1], solution[n + 1:], -product[2 * n + 1]


def _check_bar(results_by_name):
    """piezo: Y(f) (re / im series) against the 1D reference, the BVD fit against the lossless short / open mode-1
    frequencies and k_eff^2; elastic: Re u_x and u_abs of every snapshot against the 1D solution."""
    errors, tolerances, details = {}, {}, {}
    piezo = results_by_name['piezo'].result(_PHYSICS, 'admittance')
    frequencies = np.asarray(piezo['x'], dtype=np.float64)
    measured = np.asarray(piezo['series']['re']) + 1j * np.asarray(piezo['series']['im'])
    expected = np.array([2j * math.pi * frequency * _WIDTH * _bar_1d(2.0 * math.pi * frequency, True,
                                                                    voltage=_DRIVE)[2] / _DRIVE
                         for frequency in frequencies])
    errors['piezo Y'] = float(np.max(np.abs(measured - expected) / np.abs(expected)))
    tolerances['piezo Y'] = _ADMITTANCE_TOLERANCE
    if frequencies.size != _N_SWEEP:
        errors['piezo frequency count'] = abs(frequencies.size - _N_SWEEP)
        tolerances['piezo frequency count'] = 0.5
    short, open_ = _lossless_mode_one()
    errors['BVD f_s'] = abs(piezo['f_s'] / short - 1.0)
    errors['BVD f_p'] = abs(piezo['f_p'] / open_ - 1.0)
    tolerances['BVD f_s'] = tolerances['BVD f_p'] = _BVD_FREQUENCY_TOLERANCE
    errors['BVD k_eff2'] = abs(piezo['k_eff2'] / ((open_ ** 2 - short ** 2) / open_ ** 2) - 1.0)
    tolerances['BVD k_eff2'] = _BVD_COUPLING_TOLERANCE
    details['piezo'] = {'short': short, 'open': open_, 'fit': {key: piezo.get(key) for key in
                                                              ('f_s', 'f_p', 'k_eff2', 'Q_s', 'C0', 'C1', 'L1', 'R1',
                                                               'tan_delta_0', 'fit_error', 'fit_warning')}}
    elastic = results_by_name['elastic']
    first = _chain_frequencies(_moduli()[0])[0]
    for index, factor in enumerate(_ELASTIC_FACTORS):
        nodes, _, _ = _bar_1d(2.0 * math.pi * factor * first, False, load=_TRACTION)
        centres = 0.5 * (nodes[:-1] + nodes[1:])
        for name, values in (('u_x', centres.real), ('u_abs', np.abs(centres))):
            field = np.asarray(elastic.snapshot_field(index, f'{_PHYSICS}.{name}').values, dtype=np.float64)
            item = f'elastic {factor:g} f_1 {name}'
            errors[item] = float(np.max(np.abs(field - values[:, None]))) / float(np.max(np.abs(centres)))
            tolerances[item] = _FIELD_TOLERANCE
    heated = results_by_name['heated']
    for index, factor in enumerate(_ELASTIC_FACTORS):
        for name in _HEATED_FIELDS:
            plain = np.asarray(elastic.snapshot_field(index, f'{_PHYSICS}.{name}').values, dtype=np.float64)
            warm = np.asarray(heated.snapshot_field(index, f'{_PHYSICS}.{name}').values, dtype=np.float64)
            item = f'heated {factor:g} f_1 {name}'
            errors[item] = float(np.max(np.abs(warm - plain))) / max(float(np.max(np.abs(plain))), 1e-300)
            tolerances[item] = _FIELD_TOLERANCE
    return scaled_worst_result(errors, tolerances, **details)


FREQUENCY_RESPONSE_CASES = (
    SelfTestCase(
        name='solid_bvd_fit_round_trip',
        label='Solid mechanics: Butterworth-Van Dyke fit round trip (CPU)',
        description='Host check (specs {}) of physics/mechanics/bvd.py: the exact admittance of a BVD circuit (C0 '
                    f"{_BVD['C0']:g} F with tan delta {_BVD['tan_delta_0']:g}, R1 {_BVD['R1']:g} Ohm, L1 {_BVD['L1']:g} "
                    f"H, C1 {_BVD['C1']:g} F) on 301 frequencies from 0.97 f_s to f_p + 0.03 f_s is fitted back: every "
                    f'element, f_s, f_p and k_eff2 within {_ROUND_TRIP_TOLERANCE:g} (relative), fit_error alike; a '
                    'band of 4 frequencies gives NaN values and a fit_warning.',
        specs=lambda: {}, check=_check_bvd),
    SelfTestCase(
        name='solid_bvd_kappa_e',
        label='Solid mechanics: external coupling kappa_e of the admittance result (CPU)',
        description='Host check (specs {}) of frequency_response.admittance_result and bvd.external_coupling: the exact '
                    'admittance of the solid_bvd_fit_round_trip circuit as electrode points (V = 1 V, Q = Y / (j omega)) '
                    f'with depth 1 (the 3D case) gives kappa_e = Z / L1 and omega_C0_Z = 2 pi f_s C0 Z at Z = '
                    f'{_LOAD_IMPEDANCE:g} Ohm within {_ROUND_TRIP_TOLERANCE:g} (relative); per m of depth (thickness '
                    'None) both are NaN with a kappa_e_warning (first estimate, not yet run).',
        specs=lambda: {}, check=_check_kappa),
    SelfTestCase(
        name='solid_frequency_response_bar',
        label='Solid mechanics: harmonically driven bar, piezoelectric admittance and elastic response',
        description='frequency_response runs of the selftests_piezo bar (10 x 1 mm, 20 x 2 cells, rollers on y_lo / '
                    'y_hi and x_lo: y-uniform fields, the 2D Q1 solution is W x the 1D linear-element one) with '
                    f'mechanical_loss_factor {_LOSS_FACTOR:g}. piezo: ground x_lo, drive x_hi at {_DRIVE:g} V, '
                    f'dielectric_loss_tangent {_LOSS_TANGENT:g}, {_N_SWEEP} frequencies from {1.0 - _BAND_MARGIN:g} '
                    f'f_short to f_open + {_BAND_MARGIN:g} f_short (the Y comparison is exact per frequency; the '
                    'narrower band suits the single-mode BVD); the admittance series re / im against the dense complex '
                    '1D system (relative '
                    f'{_ADMITTANCE_TOLERANCE:g}) and its BVD fit f_s / f_p against the lossless discrete short / open '
                    f'mode-1 frequencies ({_BVD_FREQUENCY_TOLERANCE:g}), k_eff2 against (f_open^2 - f_short^2) / '
                    f'f_open^2 ({_BVD_COUPLING_TOLERANCE:g}). elastic (piezoelectric off): a {_TRACTION:g} Pa traction '
                    f"on x_hi at {', '.join(f'{factor:g}' for factor in _ELASTIC_FACTORS)} x the lowest chain frequency; "
                    'per snapshot the real part u_x and the amplitude u_abs against the 1D solution within '
                    f'{_FIELD_TOLERANCE:g} of its peak. heated: the elastic run with alpha {_THERMAL_EXPANSION:g} 1/K '
                    f'and a uniform temperature_change of {_TEMPERATURE_CHANGE:g} K (the harmonic drive holds no '
                    'thermal load): u_x and stress_xx / yy / zz per snapshot equal the elastic run within '
                    f'{_FIELD_TOLERANCE:g} of its peak. measured = error = the worst deviation in tolerance units, '
                    'tolerance 1: this pins the complex host solve, both loss models, the electrode charge and '
                    'current, the admittance sign and the snapshot outputs. Not yet run with 201 frequencies on the '
                    'narrower band (was 601 from 0.85 f_short to f_open + 0.15 f_short; the BVD fit start is the risk).',
        specs=_bar_specs, check=_check_bar, heavy=True),
)
