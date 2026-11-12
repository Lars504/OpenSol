"""Temporal envelopes (GaussianPulse, ContinuousWave, RickerWavelet), their sampled DFT spectra and the soft
PointSource / LineSource field injections of the FDTD steppers."""
import math

import numpy as np

from .backend import cp
from .constants import DTYPE
from .kernels import twin

_SPECTRUM_CHUNK_ELEMENTS = 1 << 20   # complex128 phase-matrix elements per chunk of sampled_waveform_spectrum (16 MiB)

# --- Temporal envelopes ---------------------------------------------------
# GaussianPulse / ContinuousWave / RickerWavelet take a float time (-> Python float) or a numpy array of times (-> float64 array);
# quadrature(t) is the carrier shifted by 90 degrees (cos for sin), the partner of the mode sources' complex phasors.
# A Python float / int time (the per-step call) takes the math fast path (0-d numpy calls cost microseconds each).

def _float_or_array(values, t):
    """A Python float when t is a scalar, else the float64 array `values`."""
    return float(values) if np.ndim(t) == 0 else values


class GaussianPulse:

    def __init__(self, f0: float, bandwidth: float = 0.5, t0_factor: float = 4.5):
        self.f0 = f0
        self.bandwidth = bandwidth
        self.tau = 1.0 / (bandwidth * f0 * np.pi)
        self.t0 = t0_factor * self.tau

    def _carried(self, t, scalar_carrier, array_carrier):
        """Envelope times carrier(phase) (math carrier for a Python number t, numpy one for arrays)."""
        if isinstance(t, (float, int)):
            delay = t - self.t0
            return math.exp(-(delay / self.tau) ** 2) * scalar_carrier(2.0 * math.pi * self.f0 * delay)
        delay = np.asarray(t, dtype=np.float64) - self.t0
        return _float_or_array(np.exp(-(delay / self.tau) ** 2) * array_carrier(2.0 * np.pi * self.f0 * delay), t)

    def __call__(self, t):
        return self._carried(t, math.sin, np.sin)

    def quadrature(self, t):
        """Envelope times cos of the carrier phase (the pulse is envelope times sin): no exact Hilbert partner, its
        spectrum errs by 2 exp(-4 f / (bandwidth^2 f0)) of the pulse's and holds DC 2 exp(-1 / bandwidth^2) of its peak."""
        return self._carried(t, math.cos, np.cos)

    @property
    def duration(self) -> float:
        return 2.0 * self.t0

    def spectral_limit(self) -> float:
        """Frequency [Hz] above which the spectrum exp(-((f - f0) / (bandwidth f0))^2) stays below 1e-6 of its peak."""
        return self.f0 * (1.0 + 3.72 * self.bandwidth)


class ContinuousWave:

    def __init__(self, f0: float, ramp_periods: float = 3.0):
        self.f0 = f0
        self.tau_ramp = ramp_periods / f0

    def _carried(self, t, scalar_carrier, array_carrier):
        """Ramp times carrier(2 pi f0 t) (math carrier for a Python number t, numpy one for arrays)."""
        if isinstance(t, (float, int)):
            return (1.0 - math.exp(-t / self.tau_ramp)) * scalar_carrier(2.0 * math.pi * self.f0 * t)
        times = np.asarray(t, dtype=np.float64)
        return _float_or_array((1.0 - np.exp(-times / self.tau_ramp)) * array_carrier(2.0 * np.pi * self.f0 * times), t)

    def __call__(self, t):
        return self._carried(t, math.sin, np.sin)

    def quadrature(self, t):
        """Ramp times cos(2 pi f0 t) (the wave is ramp times sin)."""
        return self._carried(t, math.cos, np.cos)

    def spectral_limit(self):
        """None: the switch-on ramp's spectrum decays only algebraically (DFT monitors sample every step)."""
        return None


class RickerWavelet:

    def __init__(self, f_peak: float, t0_factor: float = 4.0):
        self.f_peak = f_peak
        self.t0 = t0_factor / f_peak

    def __call__(self, t):
        if isinstance(t, (float, int)):
            arg = math.pi * self.f_peak * (t - self.t0)
            return (1.0 - 2.0 * arg * arg) * math.exp(-arg * arg)
        arg = np.pi * self.f_peak * (np.asarray(t, dtype=np.float64) - self.t0)
        return _float_or_array((1.0 - 2.0 * arg ** 2) * np.exp(-arg ** 2), t)

    @property
    def duration(self) -> float:
        return 2.0 * self.t0

    def spectral_limit(self) -> float:
        """Frequency [Hz] above which the spectrum (f / f_peak)^2 exp(-(f / f_peak)^2) stays below 1e-6 of its peak."""
        return 4.2 * self.f_peak


def sampled_waveform_spectrum(temporal, dt, frequencies, first_step, last_step, sample_interval=1):
    """S(omega) = sum of s(n dt) exp(-j omega n dt) m dt over the multiples n of m = sample_interval in [first_step,
    last_step] (the DFT monitors' sample grid), complex128 at `frequencies` [Hz]."""
    sample_interval = int(sample_interval)
    first_sampled_step = -(-int(first_step) // sample_interval) * sample_interval
    times = np.arange(first_sampled_step, int(last_step) + 1, sample_interval, dtype=np.float64) * dt
    weighted_signal = np.asarray(temporal(times), dtype=np.float64) * (dt * sample_interval)
    angular = -2.0 * np.pi * np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    spectrum = np.zeros(angular.size, dtype=np.complex128)
    chunk = max(1, _SPECTRUM_CHUNK_ELEMENTS // max(angular.size, 1))   # bounds the (n_f, chunk) phase matrix
    for start in range(0, times.size, chunk):
        phases = np.exp(1j * np.outer(angular, times[start:start + chunk]))
        spectrum += phases @ weighted_signal[start:start + chunk]
    return spectrum


class WaveformSpectrumCache:
    """sampled_waveform_spectrum over steps 1 .. n_step of one owner's waveform, cached per (temporal object, dt,
    frequency set, m): a later call adds only the steps run since (recomputed from step 1 if n_step shrank). An entry
    holds its temporal object (kept alive, compared by identity), so a new waveform reusing a freed id starts over."""

    def __init__(self):
        self._entries = {}

    def spectrum(self, temporal, dt, frequencies, n_step, sample_interval=1):
        """complex128 copy of S(omega) at `frequencies` [Hz] (sampled_waveform_spectrum from step 1 to n_step)."""
        frequencies = np.asarray(frequencies, dtype=np.float64)
        n_step, sample_interval = int(n_step), int(sample_interval)
        cache_key = (id(temporal), float(dt), frequencies.tobytes(), sample_interval)
        entry_temporal, summed_steps, spectrum = self._entries.get(cache_key, (None, 0, None))
        if spectrum is None or entry_temporal is not temporal or n_step < summed_steps:
            summed_steps, spectrum = 0, np.zeros(frequencies.size, dtype=np.complex128)
        if n_step > summed_steps:
            spectrum = spectrum + sampled_waveform_spectrum(temporal, dt, frequencies, summed_steps + 1, n_step,
                                                            sample_interval)
        self._entries[cache_key] = (temporal, n_step, spectrum)
        return spectrum.copy()


# --- Spatial sources ------------------------------------------------------

# In-place target += scale * profile over a (possibly strided) source view: no temporaries per step (a kernels twin);
# float32 profile / scale, the target float32 or float64 (2D FDTD double stepper).
def _add_scaled_profile_host(profile, scale, target):
    target += scale * profile


_add_scaled_profile = twin('core.source_add_scaled_profile', cp.ElementwiseKernel(
    'float32 profile, float32 scale',
    'T target',
    'target += (T)(scale * profile)',
    'source_add_scaled_profile'), _add_scaled_profile_host)


def _is_backend_array(value):
    """True for a numpy or cupy ndarray."""
    return isinstance(value, np.ndarray) or isinstance(value, cp.ndarray)


class PointSource:
    """Soft source at cell (ix, iy): adds amplitude * temporal(t) per step, times the optional run-backend `weight`
    (numpy / cupy float32 shape (1,), rewritten on the device by calibrations without host syncs)."""

    def __init__(self, ix, iy, temporal, amplitude=1.0, weight=None):
        if weight is not None and not (_is_backend_array(weight) and weight.dtype == DTYPE
                                       and weight.shape == (1,)):
            raise TypeError("PointSource weight must be a numpy / cupy float32 array of shape (1,)")
        self.ix = int(ix)
        self.iy = int(iy)
        self.temporal = temporal
        self.amplitude = DTYPE(amplitude)
        self.weight = weight

    def target_cell(self, target):
        """Shape-(1,) view of `target` at (ix, iy)."""
        return target[self.ix:self.ix + 1, self.iy]

    def inject(self, target, t: float):
        value = DTYPE(self.amplitude * self.temporal(t))
        if value == 0:   # a pulse past its float32 envelope: adding 0 is exact, skip the launch
            return
        if self.weight is None:
            target[self.ix, self.iy] += value
        else:
            _add_scaled_profile(self.weight, value, self.target_cell(target))


class LineSource:
    """Soft source on a grid line: axis 'x' = line along x at y-cell `position` (axis 'y': along y at x-cell).
    Adds amplitude * temporal(t) * profile over the half-open cell range `span` (None = whole grid line)."""

    def __init__(self, axis, position, temporal, amplitude=1.0, span=None, profile=None):
        if axis not in ('x', 'y'):
            raise ValueError(f"LineSource axis must be 'x' or 'y', got {axis!r}")
        if span is not None:
            span = (int(span[0]), int(span[1]))
            if not 0 <= span[0] < span[1]:
                raise ValueError(f"LineSource span must be a half-open cell range lo < hi, got {span}")
        if profile is not None:
            if not (_is_backend_array(profile) and profile.dtype == DTYPE and profile.ndim == 1):
                raise TypeError("LineSource profile must be a 1D numpy / cupy float32 array")
            if span is not None and profile.size != span[1] - span[0]:
                raise ValueError(f"LineSource profile has {profile.size} cells, span {span} has "
                                 f"{span[1] - span[0]}")
        self.axis = axis
        self.position = int(position)
        self.temporal = temporal
        self.amplitude = DTYPE(amplitude)
        self.span = span
        self.profile = profile
        self._along = slice(*span) if span is not None else slice(None)

    def target_line(self, target):
        """View of `target` covered by this source (cells along the line over the span)."""
        if self.axis == 'x':
            return target[self._along, self.position]
        return target[self.position, self._along]

    def inject(self, target, t: float):
        value = DTYPE(self.amplitude * self.temporal(t))
        if value == 0:   # exact no-op (a pulse past its envelope)
            return
        line = self.target_line(target)
        if self.profile is None:
            line += value
        else:
            _add_scaled_profile(self.profile, value, line)
