"""Self-test check of core.harmonic_probe (CPU, no model run): record / replace / keep-both / series round trip; the
SelfTestCase that runs it lives in model.selftests_analytic, which passes its SubChecks.add (core imports no model)."""
import math

import numpy as np

_PROBE_FREQUENCIES = (3.0e6, 1.0e6, 2.0e6)   # Hz, recorded out of order
_ROUND_OFF = 1e-15                           # complex128 values pass through unchanged


def _phasor(frequency, component):
    """Synthetic phasor of a component at frequency [Hz] (every quadrant over the three frequencies)."""
    scale = 1.0 if component == 'u' else -2.5
    return scale * complex(math.cos(frequency * 1e-6 * 2.1), math.sin(frequency * 1e-6 * 2.1)) * (1.0 + frequency
                                                                                                    * 1e-7)


def check_harmonic_probe_record(add):
    """Fill the sub-checks through add(name, passed, **details) (model's SubChecks.add)."""
    from .harmonic_probe import HarmonicProbeRecord, harmonic_probe_result

    def raises(function):
        try:
            function()
        except ValueError:
            return True
        return False

    record = HarmonicProbeRecord(('u', 'v'), unit='m', label='probe')
    add('before_first_record_raises', raises(record.recorded))
    for frequency in _PROBE_FREQUENCIES:
        record.record(frequency, {'u': np.complex128(_phasor(frequency, 'u')), 'v': _phasor(frequency, 'v')})
    record.record(2.0e6 * (1.0 + 1e-14), {'u': 7.0 + 1.0j, 'v': -1.0j})   # the same frequency: replaced
    add('replace_same_frequency', len(record) == 3, n_records=len(record))
    result = harmonic_probe_result(record)
    frequencies = np.asarray(result['x'])
    add('ascending_frequencies', np.array_equal(frequencies, np.sort(np.asarray(_PROBE_FREQUENCIES))),
        x=frequencies.tolist())
    expected_u = np.array([_phasor(1.0e6, 'u'), 7.0 + 1.0j, _phasor(3.0e6, 'u')])
    expected_v = np.array([_phasor(1.0e6, 'v'), -1.0j, _phasor(3.0e6, 'v')])
    for component, expected in (('u', expected_u), ('v', expected_v)):
        series = result['series']
        rebuilt = series[f'{component}_re'] + 1j * series[f'{component}_im']
        add(f'{component}_re_im_round_trip', np.max(np.abs(rebuilt - expected)) <= _ROUND_OFF * np.max(np.abs(expected)))
        polar = series[f'{component}_abs'] * np.exp(1j * series[f'{component}_phase'])
        add(f'{component}_abs_phase_round_trip',
            np.max(np.abs(polar - expected)) <= 1e3 * _ROUND_OFF * np.max(np.abs(expected)))
        add(f'{component}_phase_range', bool(np.all((series[f'{component}_phase'] > -math.pi)
                                                    & (series[f'{component}_phase'] <= math.pi))))
    add('series_order', list(result['series']) == ['u_re', 'u_im', 'u_abs', 'u_phase', 'v_re', 'v_im', 'v_abs',
                                                   'v_phase'], series=list(result['series']))
    add('series_units', result['series_units']['u_abs'] == 'm' and result['series_units']['v_phase'] == 'rad')
    add('result_axes', result['kind'] == 'xy' and result['x_unit'] == 'Hz' and result['components'] == ['u', 'v']
        and np.array_equal(result['frequency'], frequencies))
    keep_both = HarmonicProbeRecord('w')
    keep_both.record(5.0, 1.0)
    keep_both.record(5.0, 2.0, replace_same_frequency=False)
    keep_both.record(4.0, 3.0)
    kept_frequencies, kept_values = keep_both.recorded()
    add('keep_both_in_record_order', kept_frequencies.tolist() == [4.0, 5.0, 5.0]
        and kept_values['w'].tolist() == [3.0, 1.0, 2.0], values=kept_values['w'].real.tolist())
    add('index_of_latest', keep_both.index_of(5.0) == 1 and keep_both.index_of(6.0) is None)
    add('missing_component_raises', raises(lambda: record.record(1.0, {'u': 1.0})))
    add('bare_value_needs_one_component', raises(lambda: record.record(1.0, 1.0)))
    add('nonpositive_frequency_raises', raises(lambda: keep_both.record(0.0, 1.0)))
    add('vector_value_raises', raises(lambda: keep_both.record(1.0, np.ones(2))))
    add('component_name_must_be_identifier', raises(lambda: HarmonicProbeRecord(('u x',))))
    keep_both.clear()
    add('clear', len(keep_both) == 0 and raises(keep_both.recorded))
