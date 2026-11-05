"""Harmonic inversion of time-stepped probe signals (the resonance_monitor of every time-stepped physics), host numpy
only: mix the band down, Kaiser anti-alias FIR, decimate, matrix pencil -> frequency, decay rate, Q, amplitude and
phase per resonance.

Model: s(n dt) = sum_k Re(c_k exp(s_k n dt)), s_k = -gamma_k + j omega_k, exp(+j omega t) convention, n = 0 the
reference sample. A FIR filter maps every exponential onto itself (amplitude times its transfer at the pole), so the
filter gain is divided out exactly at each fitted pole; only the first n_taps samples are spent on the filter.
Several probes (channels) share their poles: their Hankel matrices are stacked (each channel scaled to unit peak), and
the amplitudes are fitted per channel, so a mode on the nodal line of one probe is still found through the others.
"""
import math

import numpy as np

from .backend import to_host
from .constants import C0
from .memory_estimate import COMPLEX128_BYTES

_OVERSAMPLE = 8.0         # preferred decimated rate / band half-width W: pass |f| <= W, stop |f| >= rate - W
_MIN_OVERSAMPLE = 3.0     # coarsest decimated rate / W when the sample cap forces a coarser grid
MAX_SAMPLES = 2000        # decimated samples handed to the matrix pencil (the earliest ones when more exist)
MIN_SAMPLES = 16
_STOPBAND_DB = 100.0      # Kaiser anti-alias attenuation: an alias lands in the band at <= 1e-5 of its amplitude
_SINGULAR_TOLERANCE = 1e-6    # singular values above this fraction of the largest set the model order
_MAX_MODEL_ORDER = 200
MATCH_TOLERANCE = 1e-2    # |s_a - s_b| / omega between the two pencil sizes above which a pole counts as noise
AMPLITUDE_FLOOR = 1e-4    # in-band resonances below this fraction of the strongest one are dropped
_MAX_GROWTH = 2.0         # poles growing by more than this over the analysed window are dropped
BAND_KINDS = ('wavelength', 'frequency')   # band in vacuum wavelengths [m] (EM) or in frequencies [Hz]
# per analysed step: float64 trace, arange, complex128 mix / baseband and three FFT arrays of ~2x the length (a rough
# upper figure); the matrix pencil: Hankel <= MAX_SAMPLES^2 / 4 complex128 entries, x 3 for the SVD
_HOST_BYTES_PER_STEP = 160
_PENCIL_BYTES = 3 * COMPLEX128_BYTES * MAX_SAMPLES ** 2 // 4


def resonance_columns(band):
    """Result columns of a band kind (the eigenfrequency resonance_table names); the 'wavelength' column (vacuum C0 /
    frequency) only for a wavelength band."""
    if band not in BAND_KINDS:
        raise ValueError(f'band must be one of {BAND_KINDS}, got {band!r}')
    return (('resonance_number',) + (('wavelength',) if band == 'wavelength' else ())
            + ('frequency', 'frequency_imag', 'q_factor', 'amplitude', 'phase', 'error'))


def analysis_host_bytes(n_steps, n_channels=1):
    """Host bytes of one harmonic_inversion of n_channels n_steps-sample traces (memory estimate, rough upper
    figure)."""
    return int(n_channels) * (_HOST_BYTES_PER_STEP * int(n_steps) + _PENCIL_BYTES)


def decimation_interval(dt, frequency_min, frequency_max, n_available):
    """Steps m between decimated samples: rate 1 / (m dt) ~ 8 W (W the band half-width), coarser (down to 3 W) when
    n_available / m would exceed MAX_SAMPLES."""
    half_width = 0.5 * (frequency_max - frequency_min)
    preferred = max(1, int(1.0 / (_OVERSAMPLE * half_width * dt)))
    coarsest = max(1, int(1.0 / (_MIN_OVERSAMPLE * half_width * dt)))
    return min(max(preferred, -(-int(n_available) // MAX_SAMPLES)), coarsest)


def anti_alias_taps(dt, half_width, sample_interval):
    """Kaiser-windowed sinc low pass (unit DC gain) cutting at half the decimated rate: |f| <= half_width passes,
    |f| >= rate - half_width is attenuated by _STOPBAND_DB; [1.0] without decimation."""
    if sample_interval == 1:
        return np.ones(1)
    sample_rate = 1.0 / (sample_interval * dt)
    transition = sample_rate - 2.0 * half_width
    n_taps = int(math.ceil((_STOPBAND_DB - 7.95) / (2.285 * 2.0 * math.pi * transition * dt))) + 1
    cutoff = 0.5 * sample_rate * dt   # cycles per step
    offsets = np.arange(n_taps) - 0.5 * (n_taps - 1)
    taps = 2.0 * cutoff * np.sinc(2.0 * cutoff * offsets) * np.kaiser(n_taps, 0.1102 * (_STOPBAND_DB - 8.7))
    return taps / taps.sum()


def _valid_convolution(signal, taps):
    """y[k] = sum_j taps[j] signal[k + n_taps - 1 - j] for every k with a full window (FFT convolution)."""
    n_taps = taps.size
    if n_taps == 1:
        return signal * taps[0]
    n_fft = 1 << int(signal.size + n_taps - 2).bit_length()
    full = np.fft.ifft(np.fft.fft(signal, n_fft) * np.fft.fft(taps, n_fft))
    return full[n_taps - 1:signal.size]


def _pencil_poles(channels, pencil):
    """Matrix-pencil poles w_k shared by the rows of channels (channels[c, q] ~ sum b_ck w_k^q) with pencil parameter
    `pencil`; model order from the singular values of the stacked Hankel matrices."""
    hankel = np.concatenate([np.lib.stride_tricks.sliding_window_view(samples, pencil + 1)
                             for samples in channels])   # (n_channels (N - pencil), pencil + 1)
    _, singular, right = np.linalg.svd(hankel, full_matrices=False)
    if singular.size == 0 or not singular[0] > 0.0:
        return np.zeros(0, dtype=np.complex128)
    order = min(int(np.count_nonzero(singular > _SINGULAR_TOLERANCE * singular[0])), _MAX_MODEL_ORDER, pencil - 1)
    if order < 1:
        return np.zeros(0, dtype=np.complex128)
    rows = right[:order]   # signal row space = T W, W[k, c] = w_k^c: rows[:, 1:] = X rows[:, :-1], eig(X) = w
    shift = np.linalg.lstsq(rows[:, :-1].T, rows[:, 1:].T, rcond=None)[0]
    return np.linalg.eigvals(shift)


def harmonic_inversion(signal, dt, frequency_min, frequency_max):
    """Resonances of a real signal sampled every dt (n = 0 the reference time) in [frequency_min, frequency_max] Hz; a
    2D signal (n_channels, n_steps) holds several probes fitted with shared poles. Dict of float64 arrays frequency,
    decay_rate [1/s], Q, amplitude and phase [rad] (in the channel where the resonance is largest), error (descending
    amplitude), amplitudes / phases (n_rows, n_channels), plus sample_interval, n_taps, n_samples, model_order and note
    ('' or why nothing was analysed)."""
    signal = np.asarray(signal, dtype=np.float64)
    channels = signal[None, :] if signal.ndim == 1 else signal
    n_channels, n_steps = channels.shape
    dt = float(dt)
    half_width = 0.5 * (frequency_max - frequency_min)
    centre = 0.5 * (frequency_max + frequency_min)
    result = {name: np.zeros(0) for name in ('frequency', 'decay_rate', 'Q', 'amplitude', 'phase', 'error')}
    result.update(amplitudes=np.zeros((0, n_channels)), phases=np.zeros((0, n_channels)), sample_interval=0, n_taps=0,
                  n_samples=0, model_order=0, note='')
    if not 0.0 < frequency_min < frequency_max or frequency_max >= 0.5 / dt:
        result['note'] = (f'band {frequency_min:.4g} to {frequency_max:.4g} Hz must be increasing, positive and below '
                          f'the Nyquist frequency 1 / (2 dt) = {0.5 / dt:.4g} Hz')
        return result
    sample_interval = decimation_interval(dt, frequency_min, frequency_max, n_steps)
    taps = anti_alias_taps(dt, half_width, sample_interval)
    n_samples = min(MAX_SAMPLES, (n_steps - taps.size) // sample_interval + 1) if n_steps >= taps.size else 0
    result.update(sample_interval=sample_interval, n_taps=int(taps.size), n_samples=max(n_samples, 0))
    if n_samples < MIN_SAMPLES:
        result['note'] = (f'only {max(n_samples, 0)} decimated samples (every {sample_interval} steps after a '
                          f'{taps.size}-step filter) in the analysis window; at least {MIN_SAMPLES} are needed: run '
                          'longer or start the analysis earlier')
        return result
    used = channels[:, :taps.size + (n_samples - 1) * sample_interval]
    mixer = np.exp(-2j * math.pi * centre * dt * np.arange(used.shape[1]))
    samples = np.stack([_valid_convolution(row * mixer, taps)[::sample_interval][:n_samples] for row in used])
    peaks = np.abs(samples).max(axis=1)
    scaled = samples[peaks > 0.0] / peaks[peaks > 0.0, None]   # equal weight per probe in the joint pencil
    empty = np.zeros(0, dtype=np.complex128)
    poles = _pencil_poles(scaled, n_samples // 3) if scaled.shape[0] else empty
    check_poles = _pencil_poles(scaled, n_samples // 2) if scaled.shape[0] else empty
    result['model_order'] = int(poles.size)
    # drop strongly growing poles before the amplitude fit (their powers overflow and they are noise anyway)
    poles = poles[(np.abs(poles) > 0.0) & (np.log(np.abs(poles) + 1e-300) * (n_samples - 1) <= math.log(1e6))]
    if poles.size == 0:
        result['note'] = 'the matrix pencil found no poles (the signal is zero or pure noise)'
        return result
    vandermonde = poles[None, :] ** np.arange(n_samples)[:, None]
    fitted = np.linalg.lstsq(vandermonde, samples.T, rcond=None)[0]   # (n_poles, n_channels)
    step = sample_interval * dt
    exponents = np.log(poles) / step               # -gamma + j (omega - omega_centre) [1/s]
    angular = 2.0 * math.pi * centre + exponents.imag
    decay_rate = -exponents.real
    baseband_poles = np.exp(exponents * dt)        # per full-rate step
    # fitted = c H(zeta) zeta^(n_taps - 1) = c polyval(taps, zeta); real-signal amplitude 2 |c|
    complex_amplitudes = fitted / np.polyval(taps, baseband_poles)[:, None]
    error = np.full(poles.size, np.inf)
    if check_poles.size:
        with np.errstate(divide='ignore', invalid='ignore'):   # zero poles / out-of-band omega ~ 0: inf, dropped
            check_exponents = np.log(check_poles + 0j) / step
            error = np.min(np.abs(exponents[:, None] - check_exponents[None, :]), axis=1) / np.abs(angular)
        error = np.where(np.isnan(error), np.inf, error)
    frequency = angular / (2.0 * math.pi)
    amplitudes = 2.0 * np.abs(complex_amplitudes)
    largest = np.argmax(amplitudes, axis=1)
    pole_rows = np.arange(poles.size)
    amplitude = amplitudes[pole_rows, largest]
    window = (n_samples - 1) * step
    keep = ((frequency >= frequency_min) & (frequency <= frequency_max) & (error <= MATCH_TOLERANCE)
            & (decay_rate * window > -math.log(_MAX_GROWTH)))
    if keep.any():   # the floor per channel: a resonance strong at any probe stays
        channel_peak = amplitudes[keep].max(axis=0)
        relative = np.where(channel_peak > 0.0, amplitudes / np.where(channel_peak > 0.0, channel_peak, 1.0), 0.0)
        keep &= relative.max(axis=1) >= AMPLITUDE_FLOOR
    order = np.flatnonzero(keep)[np.argsort(-amplitude[keep], kind='stable')]
    quality = np.full(poles.size, np.inf)   # Q = omega / (2 gamma); inf for a non-decaying pole
    decaying = decay_rate > 0.0
    quality[decaying] = math.pi * frequency[decaying] / decay_rate[decaying]
    phases = np.angle(complex_amplitudes)
    result.update(frequency=frequency[order], decay_rate=decay_rate[order], Q=quality[order],
                  amplitude=amplitude[order], phase=phases[pole_rows, largest][order], error=error[order],
                  amplitudes=amplitudes[order], phases=phases[order])
    if not order.size:
        result['note'] = 'no resonance in the band passed the consistency and amplitude checks'
    return result


def pulse_end_time(waveforms):
    """Latest end [s] of the pulsed waveforms (objects with a `duration`: gaussian / ricker 2 t0, an acoustic heating
    pulse); None entries and cw waveforms are skipped; 0 when none is pulsed."""
    ends = (getattr(waveform, 'duration', None) for waveform in waveforms)
    return max((float(end) for end in ends if end is not None), default=0.0)


class ResonanceMonitor:
    """Harmonic inversion of the traces of one or more point probes of one component (get_data() -> times, values;
    field_component), poles shared, over a band of vacuum wavelengths [m] or frequencies [Hz], from start_time (None =
    the end of the latest source pulse); the analysis runs on request and is cached per sample count."""

    def __init__(self, probes, band_min, band_max, band='wavelength', start_time=None, sample_delay_steps=0.0,
                 point_names=()):
        self.probes = tuple(probes)
        if not self.probes:
            raise ValueError('a ResonanceMonitor needs at least one probe')
        components = {probe.field_component for probe in self.probes}
        if len(components) != 1:
            raise ValueError(f'the probes of a ResonanceMonitor record one component, got {sorted(components)}')
        self.field_component = self.probes[0].field_component
        self.point_names = tuple(point_names) or tuple(f'point {index + 1}' for index in range(len(self.probes)))
        self.band = band
        self.columns = resonance_columns(band)
        self.band_min = float(band_min)
        self.band_max = float(band_max)
        self.start_time = None if start_time is None else float(start_time)
        self.sample_delay_steps = float(sample_delay_steps)   # the sample lags its recorded time (EM H: 0.5 step)
        self._cached = None   # (key, result)

    def frequency_band(self):
        """(frequency_min, frequency_max) [Hz] of the band."""
        if self.band == 'wavelength':
            return C0 / self.band_max, C0 / self.band_min
        return self.band_min, self.band_max

    def result(self, waveforms, unit, dt):
        """'table' result: one row per resonance (descending amplitude) of `columns` plus the analysis extras
        (component, unit, dt, start_time, end_time, sample_interval, n_samples, model_order, <band>_min / _max, note,
        points, point_amplitudes / point_phases per row and point); waveforms = the sources' temporal objects (pulse
        end), dt [s] = the recorded step."""
        dt = float(dt)
        traces = [probe.get_data() for probe in self.probes]   # every probe records the same steps
        n_recorded = min(int(to_host(trace_times).size) for trace_times, _ in traces)
        times = np.asarray(to_host(traces[0][0]), dtype=np.float64)[:n_recorded]
        start = self.start_time if self.start_time is not None else pulse_end_time(waveforms)
        key = (n_recorded, start, dt)
        if self._cached is not None and self._cached[0] == key:
            return self._cached[1]
        first = int(np.searchsorted(times, start - 0.5 * dt))
        values = np.stack([np.asarray(to_host(trace_values), dtype=np.float64)[first:n_recorded]
                           for _, trace_values in traces])
        analysis = harmonic_inversion(values, dt, *self.frequency_band())
        if first >= times.size:
            analysis['note'] = (f'no samples after the analysis start {start:.4g} s (the run ended at '
                                f'{times[-1] if times.size else 0.0:.4g} s): run longer or set an earlier start_time')
        with_wavelength = self.band == 'wavelength'
        # frequency_imag = gamma / (2 pi) [Hz]: Im of the complex frequency, exp(+j omega t), > 0 decays
        rows = [[number] + ([float(C0 / frequency)] if with_wavelength else [])
                + [float(frequency), float(decay) / (2.0 * math.pi), float(quality), float(amplitude), float(phase),
                   float(error)]
                for number, (frequency, quality, decay, amplitude, phase, error) in enumerate(zip(
                    analysis['frequency'], analysis['Q'], analysis['decay_rate'], analysis['amplitude'],
                    analysis['phase'], analysis['error']), start=1)]
        sample_delay = self.sample_delay_steps * dt
        reference_time = float(times[first]) - sample_delay if first < times.size else math.nan
        end_index = first + analysis['n_taps'] - 1 + (analysis['n_samples'] - 1) * analysis['sample_interval']
        end_time = (float(times[end_index]) - sample_delay
                    if analysis['n_samples'] and end_index < times.size else math.nan)
        units = ['1'] + (['m'] if with_wavelength else []) + ['Hz', 'Hz', '1', unit, 'rad', '1']
        result = {'kind': 'table', 'columns': list(self.columns), 'units': units, 'rows': rows,
                  'component': self.field_component, 'unit': unit, 'dt': dt, 'start_time': reference_time,
                  'end_time': end_time, 'sample_interval': int(analysis['sample_interval']),
                  'n_samples': int(analysis['n_samples']), 'model_order': int(analysis['model_order']),
                  f'{self.band}_min': self.band_min, f'{self.band}_max': self.band_max, 'note': analysis['note'],
                  'points': list(self.point_names),
                  'point_amplitudes': [[float(value) for value in row] for row in analysis['amplitudes']],
                  'point_phases': [[float(value) for value in row] for row in analysis['phases']]}
        self._cached = (key, result)
        return result
