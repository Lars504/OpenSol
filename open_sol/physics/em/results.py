"""Result dicts of EM monitors (numpy arrays, builtin floats / ints / str) for plots, export and the GUI."""
import weakref

import numpy as np
from ...core.backend import to_host

from ...core.constants import C0, ETA0
from ...core.monitors import probe_result
from ...core.sources import ContinuousWave, WaveformSpectrumCache
from .mode_ports import ModeSource
from .placement import line_mirror_factor
from .sources import _carrier_angular_frequency

__all__ = ['probe_result', 'flux_result', 'dft_result', 'far_field_result', 'mode_source_result',
           'mode_power_series', 'mode_monitor_result']

# 'quantity' of the DFT 'intensity' and of the far-field 'power' per monitor polarisation.
_DFT_QUANTITY = {'ez': 'abs(E_z)^2', 'hz': 'abs(E_x)^2 + abs(E_y)^2'}
_FAR_FIELD_QUANTITY = {'ez': 'abs(E_z)^2', 'hz': 'abs(eta0 H_z)^2'}
_MODE_TABLE_COLUMNS = ('mode_number', 'n_eff', 'n_eff_imag', 'loss_db_per_cm', 'n_group', 'edge_field', 'selected')
_MODE_TABLE_UNITS = ('1', '1', '1', 'dB/cm', '1', '1', '')
_DIRECTIONS = ('positive', 'negative')
_VALID_SPECTRUM_FRACTION = 1e-6       # fraction_* / phase_* / *_per_cw are NaN where |S|^2 < this x |S(f0)|^2
_SOURCE_SPECTRA = weakref.WeakKeyDictionary()   # non-mode source -> its WaveformSpectrumCache (dies with the source)


def _host_spectrum(spectra, freq_index):
    """complex128 host copy of one frequency of a (n_freqs, mx, my) run-backend spectrum."""
    return to_host(spectra[freq_index]).astype(np.complex128)


def checked_freq_index(freq_index, n_freqs: int) -> int:
    """`freq_index` as an int in [0, n_freqs); TypeError for a non-int (bool included), IndexError out of range."""
    if isinstance(freq_index, bool) or not isinstance(freq_index, (int, np.integer)):
        raise TypeError(f"freq_index must be an int, got {freq_index!r}")
    if not 0 <= int(freq_index) < n_freqs:
        raise IndexError(f"freq_index {freq_index} out of range for {n_freqs} DFT frequencies")
    return int(freq_index)


def _spectral_axes(monitor):
    """(frequency [Hz], wavelength [m]) float64 numpy copies of a spectral monitor's frequencies."""
    frequency = np.array(monitor.frequencies, dtype=np.float64)
    return frequency, C0 / frequency


def _waveform_spectrum(source, physics, frequencies, sample_interval):
    """S(omega) of `source`'s waveform over the steps run, on a monitor's DFT grid (incrementally cached per
    source)."""
    n_step = physics.engine.n_step
    if isinstance(source, ModeSource):
        return source.waveform_spectrum(frequencies, n_step, sample_interval)
    cache = _SOURCE_SPECTRA.get(source)
    if cache is None:
        cache = _SOURCE_SPECTRA[source] = WaveformSpectrumCache()
    return cache.spectrum(source.temporal, float(physics.engine.dt), frequencies, n_step, sample_interval)


def _normalizing_spectrum(source, physics, frequency, sample_interval):
    """(S(omega) of `source`'s waveform at `frequency`, valid mask): valid where |S|^2 >= 1e-6 |S(carrier)|^2 and, cw,
    inside the main lobe |f - f0| < 1 / (n_step dt) of the truncated carrier."""
    spectrum = _waveform_spectrum(source, physics, frequency, sample_interval)
    # Reference = the source's carrier power, not the monitor band's maximum (a band off the pulse).
    carrier_frequency = _carrier_angular_frequency(source.temporal) / (2.0 * np.pi)
    carrier_power = float(np.abs(_waveform_spectrum(source, physics, [carrier_frequency], sample_interval)[0]) ** 2)
    valid = (np.abs(spectrum) ** 2 >= _VALID_SPECTRUM_FRACTION * carrier_power) & (carrier_power > 0.0)
    if isinstance(source.temporal, ContinuousWave):
        run_duration = physics.engine.n_step * float(physics.engine.dt)
        valid &= np.abs(frequency - source.temporal.f0) * run_duration < 1.0
    return spectrum, valid


def common_waveform(sources):
    """The temporal waveform every source shares (same type and parameters), None without sources or when they
    differ."""
    sources = list(sources)
    if not sources:
        return None
    first = sources[0].temporal
    if any(type(source.temporal) is not type(first) or vars(source.temporal) != vars(first) for source in sources):
        return None
    return first


def _shared_waveform_spectrum(physics, frequency, sample_interval):
    """_normalizing_spectrum of the one waveform (type and parameters) every source of `physics` shares, else None
    (no physics, no source or mixed waveforms; coupling slot drives are not counted)."""
    sources = [] if physics is None else list(physics.sources)
    if common_waveform(sources) is None:
        return None
    source = next((source for source in sources if isinstance(source, ModeSource)), sources[0])
    return _normalizing_spectrum(source, physics, frequency, sample_interval)


def _per_cw(values, spectrum, valid, factor=1.0):
    """values / (factor |S|^2) along the leading (frequency) axis, NaN where not valid."""
    scale = np.where(valid, 1.0 / (factor * np.where(valid, np.abs(spectrum) ** 2, 1.0)), np.nan)
    return np.asarray(values, dtype=np.float64) * scale.reshape(scale.shape + (1,) * (np.ndim(values) - 1))


def flux_result(monitor, physics=None) -> dict:
    """Flux spectrum of a FluxMonitor; power > 0 flows toward +x (axis 'x') or +y (axis 'y'), per unit z length in
    'unit' = 'a.u.' (dt-weighted spectra); with `physics` whose sources share one waveform also 'power_per_cw' [W/m].
    With `physics`, a line reaching a mirror face counts its image too: x 'mirror_factor' (2 per such face)."""
    frequency, wavelength = _spectral_axes(monitor)
    mirror_factor = 1.0 if physics is None else line_mirror_factor(physics.domain, physics.boundary_config,
                                                                   monitor.axis, monitor.span)
    result = {'kind': 'flux_spectrum', 'frequency': frequency, 'wavelength': wavelength,
              'power': mirror_factor * monitor.flux(), 'unit': 'a.u.', 'mirror_factor': mirror_factor}
    shared = _shared_waveform_spectrum(physics, frequency, monitor.sample_interval)
    if shared is not None:
        # flux = Re(E H*) of DFT spectra = 2 |S|^2 x the cw-phasor flux 1/2 Re(E H*).
        result['power_per_cw'] = _per_cw(result['power'], *shared, factor=2.0)
    return result


def dft_result(monitor, domain, freq_index=None, physics=None) -> dict:
    """Recorded cell centres [m], box extent, shape, 'downsample' and 'quantity'; an int freq_index adds 'intensity' =
    |E(omega)|^2 float64 (mx, my) ('hz' also 'hz_intensity' = |H_z(omega)|^2) and 'freq_index'; with `physics` whose
    sources share one waveform also 'intensity_per_cw_factor' = 1 / |S(omega)|^2 (n_f,)."""
    frequency, wavelength = _spectral_axes(monitor)
    x_centers, y_centers = domain.cell_centers()
    x_edges, y_edges = domain.cell_edges()
    x_start, x_stop = monitor.x_slice.start or 0, monitor.x_slice.stop
    y_start, y_stop = monitor.y_slice.start or 0, monitor.y_slice.stop
    result = {
        'kind': 'dft_fields', 'frequency': frequency, 'wavelength': wavelength,
        'x_centers': np.array(x_centers[monitor.x_samples], dtype=np.float64),
        'y_centers': np.array(y_centers[monitor.y_samples], dtype=np.float64),
        'extent': (float(x_edges[x_start]), float(x_edges[x_stop]),
                   float(y_edges[y_start]), float(y_edges[y_stop])),
        'shape': (int(monitor.mx), int(monitor.my)),
        'downsample': int(monitor.downsample),
        'quantity': _DFT_QUANTITY[monitor.polarization],
    }
    shared = _shared_waveform_spectrum(physics, frequency, monitor.sample_interval)
    if shared is not None:
        result['intensity_per_cw_factor'] = _per_cw(np.ones_like(frequency), *shared)
    if freq_index is not None:
        index = checked_freq_index(freq_index, monitor.n_freqs)
        result['freq_index'] = index
        # |E(omega)|^2 of the dt-weighted spectra (~1e-30 for 1 V/m fs pulses, |H_z|^2 ~1e-35) kept in float64:
        # float32 reaches denormals tens of dB below the peak.
        if monitor.polarization == 'ez':
            ez_spectrum = _host_spectrum(monitor.ez_dft, index)
            result['intensity'] = np.abs(ez_spectrum) ** 2
        else:
            ex_spectrum = _host_spectrum(monitor.ex_dft, index)
            ey_spectrum = _host_spectrum(monitor.ey_dft, index)
            hz_spectrum = _host_spectrum(monitor.hz_dft, index)
            result['intensity'] = np.abs(ex_spectrum) ** 2 + np.abs(ey_spectrum) ** 2
            result['hz_intensity'] = np.abs(hz_spectrum) ** 2
    return result


def far_field_result(monitor, n_angles: int = 360) -> dict:
    """Far-field 'power' per angle |pattern|^2 / eta0 of a NearToFarField ('unit' a.u.: dt-weighted, no 1/2, so its
    angle integral matches a closed flux box's flux_result), 'angles' [rad from +x] (n_angles,), 'power' (n_f, n_angles)
    = |amplitude|^2 (amplitude_re / _im: the complex pattern / sqrt(eta0), combine_sector_far_fields)."""
    n_angles = int(n_angles)
    if n_angles < 1:
        raise ValueError(f"n_angles must be >= 1, got {n_angles}")
    frequency, wavelength = _spectral_axes(monitor)
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    amplitude = np.asarray(monitor.far_field(angles), dtype=np.complex128) / np.sqrt(float(ETA0))
    return {'kind': 'far_field', 'frequency': frequency, 'wavelength': wavelength, 'angles': angles,
            'power': np.abs(amplitude) ** 2, 'amplitude_re': amplitude.real, 'amplitude_im': amplitude.imag,
            'unit': 'a.u.', 'quantity': _FAR_FIELD_QUANTITY[monitor.polarization]}


def mode_source_result(source) -> dict:
    """Mode table (kind 'table') of a ModeSource at its carrier wavelength: the first max(mode_number + 2, 4) modes;
    ValueError before the mode is solved."""
    return {'kind': 'table', 'columns': list(_MODE_TABLE_COLUMNS), 'units': list(_MODE_TABLE_UNITS),
            'rows': [list(row) for row in source.mode_table()],
            'wavelength': float(C0 / source.temporal.f0), 'power': float(source.power),
            'direction': source.direction, 'mode_number': int(source.mode_number),
            'tracked_mode_number': int(source.tracked_mode_number)}


def mode_power_series(a_by_direction, power_factor, unit, shared=None, incident=None, valid=True):
    """(series, series_units) of a mode monitor (FDTD and FDFD): power_* = power_factor |a|^2 in `unit`; `shared` (S,
    valid) adds power_per_cw_* [W/m]; `incident` adds fraction_* / phase_* of a / incident (NaN where not `valid`)."""
    series = {f'power_{direction}': power_factor * np.abs(amplitude) ** 2
              for direction, amplitude in a_by_direction.items()}
    series_units = dict.fromkeys(series, unit)
    if shared is not None:
        for direction in a_by_direction:
            series[f'power_per_cw_{direction}'] = _per_cw(series[f'power_{direction}'], *shared, factor=power_factor)
            series_units[f'power_per_cw_{direction}'] = 'W/m'
    if incident is not None:
        for direction, amplitude in a_by_direction.items():
            series[f'fraction_{direction}'] = np.where(valid, np.abs(amplitude / incident) ** 2, np.nan)
            series_units[f'fraction_{direction}'] = '1'
        for direction, amplitude in a_by_direction.items():
            series[f'phase_{direction}'] = np.where(valid, np.angle(amplitude / incident), np.nan)
            series_units[f'phase_{direction}'] = 'rad'
    return series, series_units


def mode_monitor_result(monitor, physics) -> dict:
    """Mode powers (kind 'xy') of a ModeMonitor vs wavelength: 2 |a|^2 per direction (a.u., flux-monitor convention);
    sources sharing one waveform add power_per_cw_* = |a|^2 / |S|^2 [W/m]; exactly one ModeSource in `physics` adds
    the fractions and phases relative to its injected mode."""
    frequency, wavelength = _spectral_axes(monitor)
    sample_interval = monitor.sample_interval
    mode_sources = [source for source in physics.sources if isinstance(source, ModeSource)]
    normalized = len(mode_sources) == 1
    incident, valid = None, True
    if normalized:
        source = mode_sources[0]
        # Incident mode amplitude sqrt(P_src) S(omega) of the source waveform over the steps run, on the monitor's DFT
        # sample grid (host float64).
        spectrum, valid = _normalizing_spectrum(source, physics, frequency, sample_interval)
        incident = np.where(valid, np.sqrt(source.power) * spectrum, 1.0)
    series, series_units = mode_power_series(dict(zip(_DIRECTIONS, monitor.amplitudes())), 2.0, 'a.u.',
                                             shared=_shared_waveform_spectrum(physics, frequency, sample_interval),
                                             incident=incident, valid=valid)
    n_eff = monitor.n_eff()
    return {'kind': 'xy', 'x': wavelength, 'x_label': 'Wavelength', 'x_unit': 'm', 'y_label': 'Mode power',
            'series': series, 'series_units': series_units, 'frequency': frequency, 'wavelength': wavelength.copy(),
            'n_eff': n_eff.real.copy(), 'n_eff_imag': n_eff.imag.copy(),
            'mode_number': int(monitor.mode_number), 'tracked_mode_number': int(monitor.tracked_mode_number),
            'normalized': normalized}
