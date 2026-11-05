"""Self-tests of the post-processing calculus of model.analytic on synthetic 1D results (CPU, no model run): derived-curve
gradient / cumtrapz and references, windowed reductions, fits (two-term, Richardson), far field, harmonic probe record;
plus derived values of operation 'expression' over earlier ones (a 1D heat slab sweep on the CPU)."""
import copy
import math

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import SubChecks
from .spec import (DerivedCurveSpec, DerivedValueSpec, DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec,
                   ParameterSpec, PhysicsSpec, SolverDefaultsSpec, StudySpec, SweepSpec)

_SINE_X = np.linspace(0.0, 2.0, 401)                 # abscissa of the sine curve (h = 5e-3)
_CALCULUS_RTOL = 1e-4                                # second-order gradient / trapezoid at h = 5e-3: ~1e-5
_LINE_CENTER, _LINE_FWHM, _LINE_OFFSET = 1550e-9, 2e-9, 0.1   # Lorentzian line [m], [m], [1]
_LINE_X = np.linspace(1540e-9, 1560e-9, 401)         # 0.05 nm samples
_FIT_RTOL = 1e-6                                     # exact model functions: round-off only
_FWHM_RTOL = 2e-3                                    # linear half-height crossings on 0.05 nm samples
_BEAM_WIDTH = 0.2                                    # rad, full width at half maximum of the Gaussian far-field beam
_BEAM_ANGLES = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
_BEAM_RTOL = 1e-2                                    # 1 degree samples
# two_exp_fit: A exp(-kappa y) + B exp(kappa (y - L)) over a barrier of length L, fitted inside 15 um margins
_BARRIER_Y = np.linspace(0.0, 150e-6, 301)           # 0.5 um samples [m]
_BARRIER_LENGTH, _BARRIER_MARGIN = 150e-6, 15e-6
_BARRIER_DECAY, _BARRIER_A, _BARRIER_B = 27.3e-6, 1.0, 0.05   # 1 / kappa [m], amplitudes [1]
# two_wave_fit: A_f exp(-j beta y) + A_b exp(+j beta y) over about one guided wavelength
_GUIDE_Y = np.linspace(0.0, 400e-6, 201)             # 2 um samples [m]
_GUIDE_WAVELENGTH = 370e-6                           # 2 pi / beta [m]
_GUIDE_FORWARD, _GUIDE_BACKWARD = 1.3 * np.exp(0.3j), 0.2 * np.exp(-1.1j)
_PROBE_FREQUENCIES = np.linspace(1.0e6, 2.0e6, 101)  # Hz: a two-wave phasor along the frequency abscissa
_PROBE_PERIOD = 0.7e6                                # Hz, 2 pi / beta of that phasor
_TWO_TERM_RTOL = 1e-6                                # exact model functions: scan + golden-section refinement
# richardson: v(h) = 2 + 3 h^2 (step sizes) and v(N) = 5 - 4 N^-1.5 (cell counts, swept out of order)
_STEP_SIZES = np.array([0.4, 0.2, 0.1])
_CELL_COUNTS = (40.0, 10.0, 80.0, 20.0)
_RICHARDSON_RTOL = 1e-8
# multi_lorentz_fit: three partly overlapping Lorentzian lines [m] on an offset (0.01 nm samples)
_PEAKS_X = np.linspace(1545e-9, 1556e-9, 1101)
_PEAK_CENTERS, _PEAK_FWHMS = (1549.0e-9, 1550.0e-9, 1551.5e-9), (0.8e-9, 1.0e-9, 0.6e-9)
_PEAK_AMPLITUDES, _PEAKS_OFFSET = (1.0, 0.7, 0.5), 0.1
# pole_fit: two complex poles nu = f0 (1 + j / (2 Q)) with residues [Hz] on a linear complex background (6.5 Hz samples)
_POLE_FREQUENCIES = np.linspace(0.995e6, 1.008e6, 2001)
_POLE_CENTERS, _POLE_QS = (1.000e6, 1.003e6), (2000.0, 5000.0)
_POLE_RESIDUES = (250.0 * np.exp(0.4j), 60.0 * np.exp(-1.2j))
_CENTER_RTOL, _POLE_Q_RTOL = 1e-6, 1e-3


def _xy(x, **series):
    return {'kind': 'xy', 'x': np.asarray(x), 'x_label': 'x', 'x_unit': '1', 'y_label': '',
            'series': {name: np.asarray(values) for name, values in series.items()},
            'series_units': dict.fromkeys(series, '1')}


def _results():
    """Synthetic 1D results by 'physics.feature' key."""
    wrapped = np.angle(np.exp(1j * _BEAM_ANGLES))   # in (-pi, pi]: a beam centred on 0 rad
    beam = np.exp(-4.0 * math.log(2.0) * (wrapped / _BEAM_WIDTH) ** 2)
    line = _LINE_OFFSET + 1.0 / (1.0 + (2.0 * (_LINE_X - _LINE_CENTER) / _LINE_FWHM) ** 2)
    short_x = np.linspace(0.5, 1.5, 101)
    return {'p.sine': _xy(_SINE_X, s=np.sin(_SINE_X)),
            'p.short': _xy(short_x, z=short_x),
            'p.poly': _xy(_SINE_X, q=3.0 - 2.0 * _SINE_X + 0.5 * _SINE_X ** 2, e=2.0 * np.exp(-3.0 * _SINE_X)),
            'p.line': _xy(_LINE_X, T=line),
            'p.ff': {'kind': 'far_field', 'angles': _BEAM_ANGLES, 'power': beam[None, :],
                     'wavelength': np.array([1.55e-6]), 'frequency': np.array([2.998e8 / 1.55e-6]), 'unit': 'a.u.'}}


def _check_analytic_calculus(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from .analytic import derived_curve_plots, derived_curve_result, result_value

    results = _results()
    checks = SubChecks()
    add = checks.add_relative

    curves = [DerivedCurveSpec('slope', 'gradient(s)', {'s': 'p.sine:s'}, reference='cos(x)', tolerance=1e-3),
              DerivedCurveSpec('area', 'cumtrapz(s)', {'s': 'p.sine:s'}, reference='1 - cos(x)'),
              DerivedCurveSpec('overlap', 'cumtrapz(z)', {'s': 'p.sine:s', 'z': 'p.short:z'}, plot='overlap')]
    plots = {plot: (xy, errors) for plot, xy, errors in derived_curve_plots(curves, results.get, {})}
    for name in ('slope', 'area'):
        xy, errors = plots[name]
        record = xy['curve_error'][name] if xy is not None else {'max': math.nan}
        checks[f'{name}_curve_error'] = {'passed': bool(record['max'] <= _CALCULUS_RTOL), 'value': record,
                                         'errors': errors}
    overlap = derived_curve_result(curves[2], results.get, {})
    values = overlap['series']['overlap']
    inside = np.isfinite(values)
    first_x, last_x = float(overlap['x'][inside][0]), float(overlap['x'][inside][-1])
    add('cumtrapz_over_overlap', values[inside][-1], 0.5 * (last_x ** 2 - first_x ** 2), 1e-9)   # linear: exact
    add('cumtrapz_overlap_start', first_x, 0.5, 0.0101)   # the first curve point at or above 0.5 (h = 5e-3)

    def derived(source, expression, operation, **options):
        return result_value(results[source], DerivedValueSpec('v', expression, operation=operation, source=source,
                                                              **options), {})

    add('windowed_integral', derived('p.short', 'z', 'integral', window=[0.75, 1.25]), 0.5, 1e-12)
    add('windowed_max', derived('p.poly', 'q', 'max', window=[0.5, 1.0]), 3.0 - 1.0 + 0.125, 1e-12)
    add('polyfit_c2', derived('p.poly', 'q', 'polyfit', fit='c2', degree=2), 0.5, _FIT_RTOL)
    add('polyfit_c0', derived('p.poly', 'q', 'polyfit', fit='c0', degree=2), 3.0, _FIT_RTOL)
    add('exp_fit_rate', derived('p.poly', 'e', 'exp_fit', fit='rate'), -3.0, _FIT_RTOL)
    add('exp_fit_amplitude', derived('p.poly', 'e', 'exp_fit', fit='amplitude'), 2.0, _FIT_RTOL)
    add('lorentz_center', derived('p.line', 'T', 'lorentz_fit', fit='center'), _LINE_CENTER, _FIT_RTOL)
    add('lorentz_fwhm', derived('p.line', 'T', 'lorentz_fit', fit='fwhm'), _LINE_FWHM, _FIT_RTOL)
    add('lorentz_offset', derived('p.line', 'T', 'lorentz_fit', fit='offset'), _LINE_OFFSET, _FIT_RTOL)
    add('lorentz_q', derived('p.line', 'T', 'lorentz_fit', fit='q'), _LINE_CENTER / _LINE_FWHM, _FIT_RTOL)
    add('fwhm_of_line', derived('p.line', f'T - {_LINE_OFFSET!r}', 'fwhm'), _LINE_FWHM, _FWHM_RTOL)
    add('argmax_of_line', derived('p.line', 'T', 'argmax'), _LINE_CENTER, 1e-6)
    peak_angle = derived('p.ff', 'power', 'argmax')
    add('far_field_argmax_wraps', min(peak_angle, 2.0 * math.pi - peak_angle) + 1.0, 1.0, 1e-9)
    add('far_field_fwhm_wraps', derived('p.ff', 'power', 'fwhm'), _BEAM_WIDTH, _BEAM_RTOL)
    add('far_field_at_negative_angle', derived('p.ff', 'power_norm', 'at', at=-0.1),
        math.exp(-4.0 * math.log(2.0) * (0.1 / _BEAM_WIDTH) ** 2), _BEAM_RTOL)
    add('far_field_window_across_zero', derived('p.ff', 'power', 'integral', window=[-0.5, 0.5]),
        _BEAM_WIDTH * math.sqrt(math.pi / (4.0 * math.log(2.0))), 1e-4)
    return checks.result()


def _two_wave(y, period, forward, backward):
    """Complex two-wave phasor A_f exp(-j beta y) + A_b exp(+j beta y), beta = 2 pi / period."""
    beta = 2.0 * math.pi / period
    return forward * np.exp(-1j * beta * y) + backward * np.exp(1j * beta * y)


def _check_two_term_fits(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from ..core.harmonic_probe import HarmonicProbeRecord, harmonic_probe_result
    from .analytic import result_value, sweep_derived_xy, sweep_richardson

    checks = SubChecks()
    add, add_raises = checks.add_relative, checks.add_raises

    kappa = 1.0 / _BARRIER_DECAY
    barrier = (_BARRIER_A * np.exp(-kappa * _BARRIER_Y)
               + _BARRIER_B * np.exp(kappa * (_BARRIER_Y - _BARRIER_LENGTH)))
    guide = _two_wave(_GUIDE_Y, _GUIDE_WAVELENGTH, _GUIDE_FORWARD, _GUIDE_BACKWARD)
    record = HarmonicProbeRecord('w', unit='m')
    for frequency in _PROBE_FREQUENCIES[::-1]:   # recorded from the top: the result sorts them
        record.record(frequency, complex(_two_wave(frequency, _PROBE_PERIOD, _GUIDE_FORWARD, _GUIDE_BACKWARD)))
    results = {'p.barrier': _xy(_BARRIER_Y, a=barrier),
               'p.guide': _xy(_GUIDE_Y, a_re=guide.real, a_im=guide.imag, b_re=guide.real),
               'p.probe': harmonic_probe_result(record),
               'p.step': _xy(_STEP_SIZES, v=2.0 + 3.0 * _STEP_SIZES ** 2),
               'p.wiggle': _xy(_STEP_SIZES, v=np.array([1.0, 3.0, 2.0]))}

    def derived(source, expression, operation, **options):
        return result_value(results[source], DerivedValueSpec('v', expression, operation=operation, source=source,
                                                              **options), {})

    window = [_BARRIER_MARGIN, _BARRIER_LENGTH - _BARRIER_MARGIN]
    add('two_exp_kappa', derived('p.barrier', 'a', 'two_exp_fit', fit='kappa', window=window), kappa, _TWO_TERM_RTOL)
    add('two_exp_decay_length', derived('p.barrier', 'a', 'two_exp_fit', fit='decay_length', window=window),
        _BARRIER_DECAY, _TWO_TERM_RTOL)
    add('two_exp_amplitude_start', derived('p.barrier', 'a', 'two_exp_fit', fit='amplitude_start', window=window),
        _BARRIER_A * math.exp(-kappa * _BARRIER_MARGIN), _TWO_TERM_RTOL)
    add('two_exp_amplitude_end', derived('p.barrier', 'a', 'two_exp_fit', fit='amplitude_end', window=window),
        _BARRIER_B * math.exp(-kappa * _BARRIER_MARGIN), _TWO_TERM_RTOL)
    add('two_exp_r_squared', derived('p.barrier', 'a', 'two_exp_fit', fit='r_squared', window=window), 1.0, 1e-9)
    add('two_wave_beta', derived('p.guide', 'a_re', 'two_wave_fit', fit='beta'), 2.0 * math.pi / _GUIDE_WAVELENGTH,
        _TWO_TERM_RTOL)
    add('two_wave_forward_abs', derived('p.guide', 'a_re', 'two_wave_fit', fit='forward_abs'), abs(_GUIDE_FORWARD),
        _TWO_TERM_RTOL)
    add('two_wave_forward_phase', derived('p.guide', 'a_re', 'two_wave_fit', fit='forward_phase'),
        float(np.angle(_GUIDE_FORWARD)), _TWO_TERM_RTOL)
    add('two_wave_backward_phase', derived('p.guide', 'a_re', 'two_wave_fit', fit='backward_phase'),
        float(np.angle(_GUIDE_BACKWARD)), _TWO_TERM_RTOL)
    add('two_wave_reflection_abs', derived('p.guide', 'a_re', 'two_wave_fit', fit='reflection_abs'),
        abs(_GUIDE_BACKWARD) / abs(_GUIDE_FORWARD), _TWO_TERM_RTOL)
    add('two_wave_r_squared', derived('p.guide', 'a_re', 'two_wave_fit', fit='r_squared'), 1.0, 1e-9)
    add('two_wave_harmonic_probe_beta', derived('p.probe', 'w_re', 'two_wave_fit', fit='beta'),
        2.0 * math.pi / _PROBE_PERIOD, _TWO_TERM_RTOL)
    add('two_wave_harmonic_probe_backward_abs', derived('p.probe', 'w_re', 'two_wave_fit', fit='backward_abs'),
        abs(_GUIDE_BACKWARD), _TWO_TERM_RTOL)
    add_raises('two_wave_needs_re_name', lambda: derived('p.guide', 'a_re * 2', 'two_wave_fit', fit='beta'))
    add_raises('two_wave_needs_im_sibling', lambda: derived('p.guide', 'b_re', 'two_wave_fit', fit='beta'))
    add('richardson_step_extrapolated', derived('p.step', 'v', 'richardson', fit='extrapolated'), 2.0,
        _RICHARDSON_RTOL)
    add('richardson_step_order', derived('p.step', 'v', 'richardson', fit='order'), 2.0, _RICHARDSON_RTOL)
    add('richardson_step_coefficient', derived('p.step', 'v', 'richardson', fit='coefficient'), 3.0, 1e-7)
    add('richardson_step_error_estimate', derived('p.step', 'v', 'richardson', fit='error_estimate'),
        3.0 * 0.1 ** 2, 1e-6)
    add_raises('richardson_alternating_raises', lambda: derived('p.wiggle', 'v', 'richardson', fit='extrapolated'))
    counts = np.asarray(_CELL_COUNTS)
    derived_by_name = {'v': list(5.0 - 4.0 * counts ** -1.5), 'other': [None, 1.0, 'nan', 2.0]}
    sweep_record = sweep_richardson('n_cells', _CELL_COUNTS, derived_by_name, 'v')
    add('sweep_richardson_extrapolated', sweep_record['extrapolated'], 5.0, _RICHARDSON_RTOL)
    add('sweep_richardson_order', sweep_record['order'], -1.5, _RICHARDSON_RTOL)
    add('sweep_richardson_coefficient', sweep_record['coefficient'], -4.0, 1e-7)
    add('sweep_richardson_error_estimate', sweep_record['error_estimate'], 4.0 * 80.0 ** -1.5, 1e-6)
    sweep_xy = sweep_derived_xy('n_cells', _CELL_COUNTS, derived_by_name)
    add('sweep_derived_xy_sorted', float(np.sum(np.abs(sweep_xy['x'] - np.sort(counts)))), 0.0, 0.0)
    add('sweep_derived_xy_nan', float(np.isnan(sweep_xy['series']['other']).sum()), 2.0, 0.0)
    return checks.result()


def _check_multi_peak_fits(results_by_name):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from .analytic import fit_problem, fit_quantities, result_value

    checks = SubChecks()
    add, add_raises = checks.add_relative, checks.add_raises

    peaks = _PEAKS_OFFSET + sum(amplitude / (1.0 + (2.0 * (_PEAKS_X - center) / fwhm) ** 2)
                                for center, fwhm, amplitude in zip(_PEAK_CENTERS, _PEAK_FWHMS, _PEAK_AMPLITUDES))
    line = _LINE_OFFSET + 1.0 / (1.0 + (2.0 * (_LINE_X - _LINE_CENTER) / _LINE_FWHM) ** 2)
    poles = [center * (1.0 + 0.5j / q) for center, q in zip(_POLE_CENTERS, _POLE_QS)]
    response = (0.3 - 0.2j + (0.05 + 0.02j) * (_POLE_FREQUENCIES - 1.0e6) / 1.0e4
                + sum(residue / (_POLE_FREQUENCIES - pole) for residue, pole in zip(_POLE_RESIDUES, poles)))
    # the poles listed from the top: the record pairs w_re / w_im on the sorted abscissa
    results = {'p.peaks': _xy(_PEAKS_X, T=peaks), 'p.line': _xy(_LINE_X, T=line),
               'p.poles': _xy(_POLE_FREQUENCIES[::-1], w_re=response.real[::-1], w_im=response.imag[::-1])}

    def derived(source, expression, operation, **options):
        return result_value(results[source], DerivedValueSpec('v', expression, operation=operation, source=source,
                                                              **options), {})

    for k, center in enumerate(_PEAK_CENTERS, start=1):
        add(f'multi_lorentz_center_{k}', derived('p.peaks', 'T', 'multi_lorentz_fit', fit=f'center_{k}', degree=3),
            center, _CENTER_RTOL)
    add('multi_lorentz_fwhm_2', derived('p.peaks', 'T', 'multi_lorentz_fit', fit='fwhm_2', degree=3),
        _PEAK_FWHMS[1], _CENTER_RTOL)
    add('multi_lorentz_amplitude_3', derived('p.peaks', 'T', 'multi_lorentz_fit', fit='amplitude_3', degree=3),
        _PEAK_AMPLITUDES[2], _CENTER_RTOL)
    add('multi_lorentz_q_1', derived('p.peaks', 'T', 'multi_lorentz_fit', fit='q_1', degree=3),
        _PEAK_CENTERS[0] / _PEAK_FWHMS[0], _CENTER_RTOL)
    add('multi_lorentz_offset', derived('p.peaks', 'T', 'multi_lorentz_fit', fit='offset', degree=3), _PEAKS_OFFSET,
        _CENTER_RTOL)
    add('multi_lorentz_spacing', derived('p.peaks', 'T', 'multi_lorentz_fit', fit='spacing', degree=3),
        0.5 * (_PEAK_CENTERS[2] - _PEAK_CENTERS[0]), _CENTER_RTOL)   # least-squares slope of three points
    add('multi_lorentz_r_squared', derived('p.peaks', 'T', 'multi_lorentz_fit', fit='r_squared', degree=3), 1.0,
        1e-9)
    add('multi_lorentz_single_line', derived('p.line', 'T', 'multi_lorentz_fit', fit='center_1', degree=1),
        _LINE_CENTER, _CENTER_RTOL)
    for k, (center, q) in enumerate(zip(_POLE_CENTERS, _POLE_QS), start=1):
        add(f'pole_frequency_{k}', derived('p.poles', 'w_re', 'pole_fit', fit=f'frequency_{k}', degree=2), center,
            _CENTER_RTOL)
        add(f'pole_q_{k}', derived('p.poles', 'w_re', 'pole_fit', fit=f'q_{k}', degree=2), q, _POLE_Q_RTOL)
    add('pole_fwhm_2', derived('p.poles', 'w_re', 'pole_fit', fit='fwhm_2', degree=2),
        _POLE_CENTERS[1] / _POLE_QS[1], _POLE_Q_RTOL)
    add('pole_residue_abs_1', derived('p.poles', 'w_re', 'pole_fit', fit='residue_abs_1', degree=2),
        abs(_POLE_RESIDUES[0]), _POLE_Q_RTOL)
    add('pole_spacing', derived('p.poles', 'w_re', 'pole_fit', fit='spacing', degree=2),
        _POLE_CENTERS[1] - _POLE_CENTERS[0], 1e-4)
    add('pole_r_squared', derived('p.poles', 'w_re', 'pole_fit', fit='r_squared', degree=2), 1.0, 1e-9)
    expected_names = ('frequency_1', 'q_1', 'fwhm_1', 'residue_abs_1', 'frequency_2', 'q_2', 'fwhm_2',
                      'residue_abs_2', 'spacing', 'r_squared')
    checks['fit_quantities_pole_fit'] = {'passed': fit_quantities('pole_fit', 2) == expected_names,
                                         'value': list(fit_quantities('pole_fit', 2))}
    checks['fit_quantities_one_line_no_spacing'] = {
        'passed': 'spacing' not in fit_quantities('multi_lorentz_fit', 1),
        'value': list(fit_quantities('multi_lorentz_fit', 1))}
    checks['fit_problem_degree_0_refused'] = {'passed': fit_problem('multi_lorentz_fit', 'center_1', 0) is not None,
                                              'value': fit_problem('multi_lorentz_fit', 'center_1', 0)}
    add_raises('multi_lorentz_line_beyond_degree',
               lambda: derived('p.peaks', 'T', 'multi_lorentz_fit', fit='center_4', degree=3))
    add_raises('multi_lorentz_too_few_peaks',
               lambda: derived('p.peaks', 'T', 'multi_lorentz_fit', fit='center_1', degree=4))
    add_raises('pole_fit_needs_re_name', lambda: derived('p.poles', 'w_re * 2', 'pole_fit', fit='q_1', degree=2))
    return checks.result()


def _check_harmonic_probe_record(results_by_name):
    """core.harmonic_probe record / replace / series round trip (the check lives in core, which imports no model)."""
    from ..core.selftests_harmonic_probe import check_harmonic_probe_record
    checks = SubChecks()
    check_harmonic_probe_record(checks.add)
    return checks.result()


# operation 'expression': a 1D slab (100 cells) with a uniform heat source Q_in, swept; N_int / rho_int = Q_in exactly
_GLOBAL_RUN = 'global_expression'
_SLAB_LENGTH, _SLAB_CELL = 1e-3, 1e-5          # m
_SOURCE_DENSITIES = (1e8, 4e8)                 # W/m^3 (float32-exact): c_eff = sqrt(Q_in) = 1e4, 2e4
_GLOBAL_RTOL = 1e-5                            # float32 field values summed in the two integrals


def _global_derived_values():
    """Two field integrals, then two 'expression' values over them (and a parameter) in list order."""
    return [DerivedValueSpec('N_int', 'heat_Q_ext', operation='integral'),
            DerivedValueSpec('rho_int', '1', operation='integral'),
            DerivedValueSpec('c_eff', 'sqrt(N_int / rho_int)', operation='expression', reference='sqrt(Q_in)',
                             tolerance=_GLOBAL_RTOL),
            DerivedValueSpec('c_ratio', 'c_eff**2 / Q_in', operation='expression')]


def _global_expression_specs():
    """The heated 1D slab (x_hi held at 300 K), stationary on the CPU, Q_in swept over _SOURCE_DENSITIES."""
    spec = ModelSpec(
        title="Self-test: derived values of operation 'expression'",
        parameters=[ParameterSpec('Q_in', repr(_SOURCE_DENSITIES[0]))],
        domain=DomainSpec(size_x=_SLAB_LENGTH, size_y=None),
        materials=[MaterialSpec('slab', properties={'k': 1.0}, selection=['all'])],
        physics=[PhysicsSpec('heat', 'heat', settings={'geometry': 'planar', 'T0': 300.0}, features=[
            FeatureSpec('source', 'heat_source', ['all'], {'Q': 'Q_in'}),
            FeatureSpec('sink', 'temperature', ['x_hi'], {'temperature': 300.0})])],
        mesh=MeshSpec(mode='uniform', dx=_SLAB_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary', sweep=SweepSpec('Q_in', list(_SOURCE_DENSITIES))),
        derived_values=_global_derived_values())
    return {_GLOBAL_RUN: spec}


def _check_global_expression(results_by_name):
    """Run values and sweep table of the 'expression' values, their parity-sector recompute, and validation of
    forward / self / unknown / field / coordinate names, a selection and a source."""
    from .schema import get_registry
    from .study_sectors import _recompute_value_expressions
    from .validate import validate

    checks = SubChecks()
    add = checks.add_relative

    summary = results_by_name[_GLOBAL_RUN].summary()
    derived = summary['derived']
    add('c_eff_final', derived['c_eff'], math.sqrt(_SOURCE_DENSITIES[-1]), _GLOBAL_RTOL)
    add('c_ratio_final', derived['c_ratio'], 1.0, 2.0 * _GLOBAL_RTOL)
    add('c_eff_derived_error', summary['derived_error']['c_eff'], 0.0, _GLOBAL_RTOL)
    sweep = summary.get('sweep') or {}
    swept = list((sweep.get('derived') or {}).get('c_eff') or [])
    if len(swept) != len(_SOURCE_DENSITIES):
        checks['sweep_table'] = {'passed': False, 'value': swept}
    for index, (value, density) in enumerate(zip(swept, _SOURCE_DENSITIES)):
        add(f'sweep_c_eff_{index}', float(value), math.sqrt(density), _GLOBAL_RTOL)

    spec = _global_expression_specs()[_GLOBAL_RUN]
    combined = {'N_int': 9.0, 'rho_int': 1.0, 'c_eff': math.nan, 'c_ratio': math.nan}
    recomputed = _recompute_value_expressions(spec, {'Q_in': 9.0}, combined)
    checks['sector_recompute'] = {'passed': recomputed == ['c_eff', 'c_ratio'] and combined['c_eff'] == 3.0
                                  and combined['c_ratio'] == 1.0, 'value': {'names': recomputed, **combined}}

    registry = get_registry()

    def errors_at(variant, name):
        return [issue.message for issue in validate(variant, registry)
                if issue.level == 'error' and tuple(issue.path) == ('derived', name)]

    good = {name: errors_at(spec, name) for name in ('c_eff', 'c_ratio')}
    checks['valid_spec'] = {'passed': not any(good.values()), 'value': good}
    refusals = {'forward_reference': ('c_eff', {'expression': 'sqrt(N_int / rho_int) + 0*c_ratio'}, 'listed after'),
                'self_reference': ('c_eff', {'expression': 'c_eff + 1'}, 'itself'),
                'unknown_name': ('c_eff', {'expression': 'sqrt(N_unknown)'}, 'N_unknown'),
                'field_name': ('c_eff', {'expression': 'sqrt(heat_T)'}, 'field or coordinate'),
                'coordinate': ('c_eff', {'expression': 'x * N_int'}, 'field or coordinate'),
                'selection': ('c_eff', {'selection': ['all']}, 'reads no fields'),
                'source': ('c_eff', {'source': 'heat.source'}, 'not a source')}
    for check_name, (name, changes, text) in refusals.items():
        variant = copy.deepcopy(spec)
        for attribute, value in changes.items():
            setattr(variant.derived_value(name), attribute, value)
        messages = errors_at(variant, name)
        checks[f'refuses_{check_name}'] = {'passed': any(text in message for message in messages),
                                           'value': messages}
    return checks.result(derived=derived)


ANALYTIC_SELFTEST_CASES = (
    SelfTestCase(
        name='analytic_calculus',
        label='Post-processing calculus: derived-curve gradient / cumtrapz, fits, far-field reductions',
        description='CPU checks of model.analytic on synthetic results (no model run): gradient(sin) vs cos and '
                    'cumtrapz(sin) vs 1 - cos as derived-curve references (curve_error <= 1e-4 at h = 5e-3), cumtrapz '
                    'of an input interpolated onto a longer abscissa integrating over the overlap only, windowed '
                    'integral (interpolated edges) and max, polyfit / exp_fit / lorentz_fit of exact model curves '
                    '(1e-6), fwhm and argmax of a Lorentzian line, and a Gaussian far-field beam centred on 0 rad: '
                    'argmax, fwhm (1 %), at a negative angle and a window across 0 (wrap-around). error = number of '
                    'failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_analytic_calculus,
        max_wall_time=10.0),
    SelfTestCase(
        name='analytic_two_term_fits',
        label='Post-processing fits: two-exponential, complex two-wave, Richardson extrapolation',
        description='CPU checks of model.analytic on synthetic results (no model run): two_exp_fit of A exp(-kappa y) '
                    '+ B exp(kappa (y - L)) inside 15 um margins of a 150 um barrier (kappa, decay length, end '
                    'amplitudes, 1e-6), two_wave_fit of A_f exp(-j beta y) + A_b exp(j beta y) pairing a_re with '
                    'a_im (beta, |A_f|, phases, |A_b / A_f|, 1e-6) and over a core.harmonic_probe result (w_re / '
                    'w_im vs frequency), its refusals (an expression other than one _re name, a missing _im '
                    'sibling), richardson of v = 2 + 3 h^2 on three step sizes and, through sweep_richardson, of v = '
                    '5 - 4 N^-1.5 on four cell counts swept out of order (extrapolated, order, coefficient, 1e-8 / '
                    '1e-7), alternating changes refused. error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_two_term_fits,
        max_wall_time=10.0),
    SelfTestCase(
        name='analytic_multi_peak_fits',
        label='Post-processing fits: several Lorentzian peaks, complex resonance poles',
        description='CPU checks of model.analytic on synthetic results (no model run): multi_lorentz_fit of three '
                    'partly overlapping Lorentzian lines on an offset (centres, a width, an amplitude, q, offset, '
                    'spacing, 1e-6; R squared 1e-9) and of one line (degree 1), pole_fit of two complex poles at Q 2000 '
                    '/ 5000 with a linear complex background pairing w_re with w_im on a reversed abscissa '
                    '(frequencies 1e-6, Q / width / residue 1e-3, spacing 1e-4), the generated quantity names '
                    '(fit_quantities) and the refusals (degree 0, a line beyond the degree, more lines than maxima, '
                    'an expression other than one _re name). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_multi_peak_fits,
        max_wall_time=10.0),
    SelfTestCase(
        name='harmonic_probe_record',
        label='Harmonic probe record: per-frequency complex phasors and their xy series',
        description='CPU check of core.harmonic_probe (no model run): records out of frequency order, a repeated '
                    'frequency replaced (replace_same_frequency) or kept in record order, series <component>_re / '
                    '_im / _abs / _phase vs ascending frequency [Hz] rebuilding the phasors (round-off), units, and '
                    'the refusals (missing component, bare value for two components, frequency <= 0, vector '
                    'value, non-identifier name, recorded() before the first record). error = number of failed '
                    'sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_harmonic_probe_record,
        max_wall_time=10.0),
    SelfTestCase(
        name='derived_global_expression',
        label="Derived values of operation 'expression': global evaluation of earlier derived values",
        description="A 1D heat slab (1 mm, 100 cells, uniform source Q_in, CPU, stationary) with Q_in swept over 1e8 "
                    "and 4e8 W/m^3: N_int = integral of heat_Q_ext and rho_int = integral of 1, then c_eff = "
                    "sqrt(N_int / rho_int) and c_ratio = c_eff**2 / Q_in of operation 'expression' (no fields, list "
                    'order). c_eff vs sqrt(Q_in) in the summary, its derived_error against reference sqrt(Q_in) and '
                    'per sweep value in the sweep table, c_ratio vs 1 (relative 1e-5: float32 field integrals); '
                    'study_sectors._recompute_value_expressions re-evaluates both from combined values; validation '
                    'refuses a later derived value, itself, an unknown name, a field, a coordinate, a selection and a '
                    'source. error = number of failed sub-checks, tolerance 0.',
        specs=_global_expression_specs,
        check=_check_global_expression,
        max_wall_time=30.0),
)
