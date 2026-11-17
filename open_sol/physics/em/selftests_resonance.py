"""Self-tests of the FDTD resonance_monitor (A6): core.resonance harmonic inversion of a synthetic damped-sinusoid
signal (CPU) and the ring-down of a lossy PEC cavity against its exact discrete Yee resonances, single-point, joint
and on a nodal line (GPU), plus the 'disk_resonator' template."""
import math

import numpy as np

from ...core.constants import EPS0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_centre
from ...model.spec import DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, StudySpec

_PHYSICS_NAME = 'em'
_PHYSICS_TYPE = 'em_fdtd_ez'
_RESONANCE_FEATURE = 'resonances'

# em_resonance_harmonic_inversion (CPU): (frequency [Hz], Q, amplitude, phase [rad]) in the band and outside it
_SYNTHETIC_DT = 5e-17                  # s
_SYNTHETIC_STEPS = 40000               # 2 ps
_SYNTHETIC_BAND = (1.8e14, 2.8e14)     # Hz
_SYNTHETIC_MODES = ((1.9e14, 50.0, 1.0, 0.3), (2.1e14, 500.0, 0.3, -1.2), (2.6e14, 5000.0, 0.05, 2.5))
_SYNTHETIC_OUT_OF_BAND = ((1.2e14, 100.0, 1.0, 1.0), (3.5e14, 30.0, 2.0, 0.0))
_SYNTHETIC_NOISE = 1e-6                # rms white noise, then float32 rounding
_SYNTHETIC_SEED = 20261012
_SYNTHETIC_TOLERANCE = 1e-3
# em_resonance_joint_fit (CPU): (frequency [Hz], Q, (amplitude, phase) per channel); each channel lacks one mode
_JOINT_MODES = ((1.9e14, 50.0, ((1.0, 0.3), (0.5, 1.0))), (2.1e14, 500.0, ((0.3, -1.2), (0.0, 0.0))),
                (2.6e14, 2000.0, ((0.0, 0.0), (0.2, 2.5))))

# em_resonance_nodal_line (GPU): eps_r 2.25, sigma_e 250 S/m filling a 1.0 x 0.6 um box of pec walls, 40 nm mesh;
# 25 x 15 cells (E_z pinned to 0 in the outermost cells: interior (1..23) x (1..13)), so TM21 (sin(2 pi i / 24))
# vanishes on column 12
_CAVITY_CELL = 40e-9                   # m
_NODAL_CELLS = (25, 15)
_CAVITY_EPS_R = 2.25
_CAVITY_SIGMA = 250.0                  # S/m: Q ~ 100 (TM11), ~134 (TM21)
_CAVITY_MODES = ((1, 1), (2, 1))       # (p, q) of sin(p pi i / (nx - 1)) sin(q pi j / (ny - 1)) in the band
_CAVITY_BAND = (1.0e-6, 2.0e-6)        # m: TM11 ~1.50 um, TM21 ~1.12 um; TM31 / TM12 (~0.86 um) lie outside
_CAVITY_SOURCE_CELL = (6, 4)           # off the nodal lines of TM11 and TM21 (TM21 crest in x)
_CAVITY_PROBE_CELL = (18, 10)          # off the nodal lines of TM11 and TM21 (TM21 crest in x)
_CAVITY_PULSE = {'waveform': 'gaussian', 'wavelength': 1.3e-6, 'bandwidth': 0.5, 'current': 1e-3}
_CAVITY_DURATION = 0.6e-12             # s: ~3.6 field e-folds of TM11
_CAVITY_TOLERANCE = 1e-3
_NODAL_POINT_CELL = (12, 5)            # on the TM21 nodal line, on the TM11 crest in x
_NODAL_MISS_TOLERANCE = 1e-2           # a 'single' row within this relative frequency of TM21 counts as found
_NODAL_RATIO_TOLERANCE = 1e-3          # TM21 amplitude at the nodal point / at the other point

# 'disk_resonator' template: silicon disk (n ~3.48) of radius 1 um in air, whispering-gallery modes near 1550 nm
DISK_RESONATOR_LABEL = 'Disk resonator ring-down'
DISK_RESONATOR_DESCRIPTION = (
    'EM (E_z): a 1.55 um gaussian pulse (bandwidth 0.2) from a line current inside a silicon disk (eps_r 12.11, radius '
    '1 um) in air excites its whispering-gallery modes; a resonance_monitor near the rim lists their wavelengths, Q '
    'and amplitudes over 1.45-1.65 um from the 2 ps ring-down.')


def _cavity_spec(title, cells, points, monitors):
    """Lossy dielectric filling a pec box of cells, point source pulse, resonance_monitors {feature: [points]} over
    the probe points {name: cell}."""
    size_x, size_y = (count * _CAVITY_CELL for count in cells)
    source_x, source_y = (cell_centre(index, _CAVITY_CELL) for index in _CAVITY_SOURCE_CELL)
    shortest, longest = _CAVITY_BAND
    monitor_params = {'component': 'ez', 'wavelength_min': shortest, 'wavelength_max': longest}
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('source_point', 'point', {'x': source_x, 'y': source_y})]
                 + [GeometrySpec(name, 'point', {'x': cell_centre(cell[0], _CAVITY_CELL),
                                                   'y': cell_centre(cell[1], _CAVITY_CELL)})
                    for name, cell in points.items()],
        materials=[MaterialSpec('lossy_dielectric', properties={'eps_r': _CAVITY_EPS_R, 'sigma_e': _CAVITY_SIGMA},
                                selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE,
                             settings={'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec'},
                             features=[FeatureSpec('source', 'point_source', ['source_point'], dict(_CAVITY_PULSE))]
                                      + [FeatureSpec(name, 'resonance_monitor', list(selection), dict(monitor_params))
                                         for name, selection in monitors.items()])],
        mesh=MeshSpec(mode='uniform', dx=_CAVITY_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _CAVITY_DURATION}))


def _nodal_specs():
    """'run': the cavity of _NODAL_CELLS; 'single' = one point on the TM21 nodal line, 'joint' = it plus another,
    _RESONANCE_FEATURE = the other point alone."""
    return {'run': _cavity_spec('Self-test: resonance finder with a point on a nodal line', _NODAL_CELLS,
                                {'nodal_point': _NODAL_POINT_CELL, 'probe_point': _CAVITY_PROBE_CELL},
                                {'single': ['nodal_point'], 'joint': ['nodal_point', 'probe_point'],
                                 _RESONANCE_FEATURE: ['probe_point']})}


def disk_resonator_spec():
    """Template 'disk_resonator': silicon disk in a 4 x 4 um CPML box, pulse inside near the rim, resonance finder."""
    return ModelSpec(
        title='Disk resonator ring-down',
        domain=DomainSpec(size_x=4e-6, size_y=4e-6),
        geometry=[GeometrySpec('disk', 'circle', {'cx': 2e-6, 'cy': 2e-6, 'r': 1e-6}),
                  GeometrySpec('source_point', 'point', {'x': 2.81e-6, 'y': 2.01e-6}),
                  GeometrySpec('probe_point', 'point', {'x': 1.45e-6, 'y': 2.57e-6})],
        materials=[MaterialSpec('silicon_disk', properties={'eps_r': 12.11}, selection=['disk'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPE, features=[
            FeatureSpec('source', 'point_source', ['source_point'],
                        {'waveform': 'gaussian', 'wavelength': 1.55e-6, 'bandwidth': 0.2, 'current': 1e-3}),
            FeatureSpec(_RESONANCE_FEATURE, 'resonance_monitor', ['probe_point'],
                        {'component': 'ez', 'wavelength_min': 1.45e-6, 'wavelength_max': 1.65e-6})])],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=StudySpec(type='time_dependent', settings={'duration': 2e-12}))


# ---- checks ----
def _ring_down(times, modes):
    """sum of a exp(-pi f t / Q) cos(2 pi f t + phi) over (f, Q, a, phi) modes."""
    return sum(amplitude * np.exp(-math.pi * frequency * times / quality) * np.cos(2.0 * math.pi * frequency * times
                                                                                    + phase)
               for frequency, quality, amplitude, phase in modes)


def _nearest_row(frequencies, frequency):
    """Index of the entry of `frequencies` closest to `frequency`; None when empty."""
    return int(np.argmin(np.abs(np.asarray(frequencies) - frequency))) if len(frequencies) else None


def _check_synthetic(results_by_name):
    """Recovered frequency, Q, amplitude (relative) and phase (rad) of each in-band mode; worst of all four."""
    from ...core.resonance import harmonic_inversion
    times = np.arange(_SYNTHETIC_STEPS) * _SYNTHETIC_DT
    signal = _ring_down(times, _SYNTHETIC_MODES + _SYNTHETIC_OUT_OF_BAND)
    signal = (signal + _SYNTHETIC_NOISE * np.random.default_rng(_SYNTHETIC_SEED).standard_normal(times.size))
    found = harmonic_inversion(signal.astype(np.float32), _SYNTHETIC_DT, *_SYNTHETIC_BAND)
    errors, per_mode = [], {}
    for frequency, quality, amplitude, phase in _SYNTHETIC_MODES:
        row = _nearest_row(found['frequency'], frequency)
        if row is None:
            errors.append(math.inf)
            per_mode[f'{frequency:.3g}'] = None
            continue
        mode_errors = {'frequency': abs(found['frequency'][row] / frequency - 1.0),
                       'Q': abs(found['Q'][row] / quality - 1.0),
                       'amplitude': abs(found['amplitude'][row] / amplitude - 1.0),
                       'phase': abs(float(np.angle(np.exp(1j * (found['phase'][row] - phase)))))}
        errors.append(max(mode_errors.values()))
        per_mode[f'{frequency:.3g}'] = {name: float(value) for name, value in mode_errors.items()}
    worst = int(np.argmax(errors))
    return {'measured': float(found['frequency'].size), 'expected': float(len(_SYNTHETIC_MODES)),
            'error': float(errors[worst]), 'tolerance': _SYNTHETIC_TOLERANCE,
            'details': {'worst_mode_frequency': _SYNTHETIC_MODES[worst][0], 'errors_by_mode': per_mode,
                        'n_rows': int(found['frequency'].size), 'sample_interval': found['sample_interval'],
                        'n_taps': found['n_taps'], 'n_samples': found['n_samples'],
                        'model_order': found['model_order'], 'note': found['note']}}


def _joint_traces(times):
    """(n_channels, n_steps) synthetic ring-downs of _JOINT_MODES (1e-6 rms noise each)."""
    rng = np.random.default_rng(_SYNTHETIC_SEED)
    channels = []
    for channel in range(len(_JOINT_MODES[0][2])):
        modes = [(frequency, quality, per_channel[channel][0], per_channel[channel][1])
                 for frequency, quality, per_channel in _JOINT_MODES]
        channels.append(_ring_down(times, modes) + _SYNTHETIC_NOISE * rng.standard_normal(times.size))
    return np.stack(channels)


def _check_joint_fit(results_by_name):
    """Joint fit of two channels, each missing one mode: every mode's f, Q, per-channel amplitude (relative; absent:
    / its largest amplitude) and phase; a single-channel fit of channel 0 must miss the mode absent there."""
    from ...core.resonance import harmonic_inversion
    times = np.arange(_SYNTHETIC_STEPS) * _SYNTHETIC_DT
    traces = _joint_traces(times).astype(np.float32)
    found = harmonic_inversion(traces, _SYNTHETIC_DT, *_SYNTHETIC_BAND)
    single = harmonic_inversion(traces[0], _SYNTHETIC_DT, *_SYNTHETIC_BAND)
    errors, per_mode = [], {}
    for frequency, quality, per_channel in _JOINT_MODES:
        row = _nearest_row(found['frequency'], frequency)
        if row is None or abs(found['frequency'][row] / frequency - 1.0) > _NODAL_MISS_TOLERANCE:
            errors.append(math.inf)
            per_mode[f'{frequency:.3g}'] = None
            continue
        largest = max(amplitude for amplitude, _ in per_channel)
        mode_errors = {'frequency': abs(found['frequency'][row] / frequency - 1.0),
                       'Q': abs(found['Q'][row] / quality - 1.0)}
        for channel, (amplitude, phase) in enumerate(per_channel):
            measured = float(found['amplitudes'][row][channel])
            mode_errors[f'amplitude_{channel}'] = abs(measured - amplitude) / (amplitude if amplitude else largest)
            if amplitude:
                mode_errors[f'phase_{channel}'] = abs(float(np.angle(np.exp(1j * (found['phases'][row][channel]
                                                                                   - phase)))))
        errors.append(max(mode_errors.values()))
        per_mode[f'{frequency:.3g}'] = {name: float(value) for name, value in mode_errors.items()}
    absent_frequency = next(frequency for frequency, _, per_channel in _JOINT_MODES if per_channel[0][0] == 0.0)
    single_row = _nearest_row(single['frequency'], absent_frequency)
    single_found = (single_row is not None
                    and abs(single['frequency'][single_row] / absent_frequency - 1.0) <= _NODAL_MISS_TOLERANCE)
    errors.append(1.0 if single_found else 0.0)
    worst = int(np.argmax(errors))
    return {'measured': float(found['frequency'].size), 'expected': float(len(_JOINT_MODES)),
            'error': float(errors[worst]), 'tolerance': _SYNTHETIC_TOLERANCE,
            'details': {'errors_by_mode': per_mode, 'single_channel_found_absent_mode': bool(single_found),
                        'single_channel_frequencies': [float(value) for value in single['frequency']],
                        'n_rows': int(found['frequency'].size), 'model_order': found['model_order'],
                        'note': found['note']}}


def _discrete_cavity_mode(p, q, dt, cells):
    """(frequency [Hz], Q) of cavity mode (p, q) from the Yee recursion with semi-implicit sigma_e: (eps/dt + sigma/2)
    z^2 - (2 eps/dt - dt kappa^2 / mu0) z + (eps/dt - sigma/2) = 0, kappa^2 the discrete Dirichlet Laplacian."""
    nx, ny = cells
    kappa_squared = (2.0 / _CAVITY_CELL) ** 2 * (math.sin(p * math.pi / (2 * (nx - 1))) ** 2
                                                 + math.sin(q * math.pi / (2 * (ny - 1))) ** 2)
    eps = EPS0 * _CAVITY_EPS_R
    roots = np.roots([eps / dt + 0.5 * _CAVITY_SIGMA, -2.0 * eps / dt + dt * kappa_squared / MU0,
                      eps / dt - 0.5 * _CAVITY_SIGMA])
    pole = roots[np.argmax(roots.imag)]
    frequency = float(np.angle(pole)) / (2.0 * math.pi * dt)
    decay_rate = -math.log(abs(pole)) / dt
    return frequency, math.pi * frequency / decay_rate


def _cavity_mode_errors(table):
    """(relative f / Q error per _CAVITY_MODES entry (inf when no row), per-mode details, rows, TM21 row index or
    None) of a resonance table of the _NODAL_CELLS cavity vs the exact discrete recursion."""
    columns = list(table['columns'])
    rows = [list(row) for row in table['rows']]
    frequencies = [row[columns.index('frequency')] for row in rows]
    errors, per_mode, tm21_row = [], {}, None
    for p, q in _CAVITY_MODES:
        expected_frequency, expected_q = _discrete_cavity_mode(p, q, float(table['dt']), _NODAL_CELLS)
        row = _nearest_row(frequencies, expected_frequency)
        if row is None:
            errors.append(math.inf)
            per_mode[f'TM{p}{q}'] = None
            continue
        if (p, q) == (2, 1):
            tm21_row = row
        measured_frequency, measured_q = rows[row][columns.index('frequency')], rows[row][columns.index('q_factor')]
        mode_error = max(abs(measured_frequency / expected_frequency - 1.0), abs(measured_q / expected_q - 1.0))
        errors.append(mode_error)
        per_mode[f'TM{p}{q}'] = {'frequency': measured_frequency, 'expected_frequency': expected_frequency,
                                 'Q': measured_q, 'expected_Q': expected_q, 'error': mode_error}
    return errors, per_mode, rows, tm21_row


def _check_nodal_line(results_by_name):
    """'joint' and the single-point _RESONANCE_FEATURE TM11 / TM21 vs the discrete recursion; 'single' (on the TM21
    node) must miss TM21 and the joint TM21 amplitude at the nodal point stay ~0; worst entry."""
    run = results_by_name['run']
    joint, single = run.result(_PHYSICS_NAME, 'joint'), run.result(_PHYSICS_NAME, 'single')
    probe_only = run.result(_PHYSICS_NAME, _RESONANCE_FEATURE)
    errors, per_mode, rows, tm21_row = _cavity_mode_errors(joint)
    probe_errors, probe_modes, probe_rows, _ = _cavity_mode_errors(probe_only)
    errors.extend(probe_errors)
    tm21_frequency = _discrete_cavity_mode(2, 1, float(single['dt']), _NODAL_CELLS)[0]
    single_frequencies = [row[list(single['columns']).index('frequency')] for row in single['rows']]
    single_row = _nearest_row(single_frequencies, tm21_frequency)
    single_found = (single_row is not None
                    and abs(single_frequencies[single_row] / tm21_frequency - 1.0) <= _NODAL_MISS_TOLERANCE)
    errors.append(1.0 if single_found else 0.0)
    ratio = math.inf
    if tm21_row is not None:
        nodal_index, other_index = (list(joint['points']).index(name) for name in ('nodal_point', 'probe_point'))
        amplitudes = joint['point_amplitudes'][tm21_row]
        ratio = amplitudes[nodal_index] / amplitudes[other_index] if amplitudes[other_index] > 0.0 else math.inf
    errors.append(ratio * _CAVITY_TOLERANCE / _NODAL_RATIO_TOLERANCE)   # scaled onto the f / Q tolerance
    worst = int(np.argmax(errors))
    return {'measured': float(len(rows)), 'expected': float(len(_CAVITY_MODES)), 'error': float(errors[worst]),
            'tolerance': _CAVITY_TOLERANCE,
            'details': {'modes': per_mode, 'probe_only_modes': probe_modes, 'single_found_TM21': bool(single_found),
                        'single_frequencies': single_frequencies, 'TM21_nodal_amplitude_ratio': ratio,
                        'worst_entry': ['TM11', 'TM21', 'probe_only TM11', 'probe_only TM21', 'single found TM21',
                                        'nodal ratio'][worst],
                        'n_rows': len(rows), 'probe_only_n_rows': len(probe_rows),
                        'probe_only_sample_interval': probe_only.get('sample_interval'),
                        'probe_only_n_samples': probe_only.get('n_samples'),
                        'probe_only_start_time': probe_only.get('start_time'),
                        'note': joint.get('note'), 'single_note': single.get('note'),
                        'probe_only_note': probe_only.get('note')}}


RESONANCE_SELFTEST_CASES = (
    SelfTestCase(
        name='em_resonance_harmonic_inversion',
        label='EM resonance finder: synthetic ring-down (CPU)',
        description='harmonic_inversion of a synthetic float32 signal (dt 5e-17 s, 40000 samples, 1e-6 rms white '
                    'noise): three damped sinusoids in the band 1.8-2.8e14 Hz (Q 50, 500, 5000; amplitudes 1, 0.3, '
                    '0.05) plus two strong modes outside it (1.2e14 and 3.5e14 Hz). Each in-band mode must come back '
                    'with its frequency, Q and amplitude (relative error) and phase (rad); error = the worst of all, '
                    'tolerance 1e-3: noise and the 100 dB anti-alias filter leave ~1e-5; a wrong filter-gain '
                    'correction, decimation alias or time reference shows up at >= 1e-2. measured / expected = the '
                    'number of rows / in-band modes (informational).',
        specs=lambda: {},
        check=_check_synthetic),
    SelfTestCase(
        name='em_resonance_joint_fit',
        label='EM resonance finder: joint fit of two probes (CPU)',
        description='harmonic_inversion of two synthetic float32 channels (dt 5e-17 s, 40000 samples, 1e-6 rms noise '
                    'each) sharing three modes in 1.8-2.8e14 Hz: 1.9e14 Hz (Q 50) in both, 2.1e14 Hz (Q 500) only in '
                    'channel 0, 2.6e14 Hz (Q 2000) only in channel 1 (zero amplitude = a probe on its nodal line). '
                    'The joint fit must return all three with f, Q, per-channel amplitude (relative; an absent mode: '
                    'its fitted amplitude / the largest one) and phase (rad); a single-channel fit of channel 0 must '
                    'miss the 2.6e14 Hz mode (error 1 when it finds it). error = the worst entry, tolerance 1e-3.',
        specs=lambda: {},
        check=_check_joint_fit),
    SelfTestCase(
        name='em_resonance_nodal_line',
        label='EM resonance finder: lossy pec cavity, point on a nodal line (FDTD)',
        description='em_fdtd_ez box of pec walls (25 x 15 cells of 40 nm, E_z pinned to 0 in the outermost cells) '
                    'filled with eps_r 2.25, sigma_e 250 S/m; a gaussian point-source pulse (1.3 um, bandwidth 0.5) '
                    'in cell (6, 4) rings the box for 0.6 ps (~3.6 TM11 field e-folds). TM21 ~ sin(2 pi i / 24) '
                    'vanishes on column 12. resonance_monitors (E_z, band 1-2 um): single = one point in cell (12, 5) '
                    'on that nodal line (and on the TM11 crest in x), joint = the same point plus cell (18, 10), '
                    f'{_RESONANCE_FEATURE} = cell (18, 10) alone (the single-channel fit path). joint and '
                    f'{_RESONANCE_FEATURE} must each find TM11 (~1.50 um, Q ~100) and TM21 (~1.12 um, Q ~134) within '
                    '1e-3 (f and Q) of the exact discrete modes of the Yee recursion with the semi-implicit '
                    'conductivity, (eps/dt + sigma/2) z^2 - (2 eps/dt - dt kappa^2/mu0) z + (eps/dt - sigma/2) = 0, '
                    'kappa^2 = (2/dx)^2 (sin^2(p pi / (2 (nx - 1))) + sin^2(q pi / (2 (ny - 1)))), f = arg(z) / (2 pi '
                    'dt), Q = pi f / gamma, gamma = -ln|z| / dt (exact at any resolution; TM31 / TM12 near 0.86 um '
                    'stay outside the band); tolerance 1e-3: float32 coefficients (ca rounding ~1e-4 of gamma) and '
                    'fit noise; a cavity one cell off moves f by ~4%, a wrong loss factor moves Q by far more. '
                    'single must have no row within 1% of TM21 (error 1 otherwise); the joint TM21 point_amplitudes '
                    'ratio nodal / other must stay below 1e-3 (float32 rounding leaves ~1e-7; scaled onto the 1e-3 '
                    'tolerance). error = the worst entry. 40 nm cells and 0.6 ps not yet confirmed by a run.',
        specs=_nodal_specs,
        check=_check_nodal_line),
)
