"""Shared harmonic probe record: complex phasors per solved frequency [Hz] of named components (a repeated frequency
replaces by default) and its 'xy' result with series <component>_re / _im / _abs / _phase versus frequency."""
# Lifted from physics/em/fdfd_monitors.HarmonicProbe (which stays as it is: wavelength abscissa). The frequency-based
# solvers (frequency_response of the mechanics, membrane, pressure_acoustics) record through this one class, so the
# derived-value fits (model.analytic two_wave_fit pairs '<name>_re' with '<name>_im') read them all alike.
import math

import numpy as np

from .backend import array_module_of, to_host

SERIES_PARTS = ('re', 'im', 'abs', 'phase')   # series '<component>_<part>' of harmonic_probe_result, in this order
FREQUENCY_MATCH_RTOL = 1e-12                  # two frequencies closer than this (relative) are the same frequency


def series_name(component, part):
    """Series name '<component>_<part>' of a harmonic probe result (part one of SERIES_PARTS)."""
    if part not in SERIES_PARTS:
        raise ValueError(f"harmonic probe series part must be one of {', '.join(SERIES_PARTS)}, got {part!r}")
    return f'{component}_{part}'


def _stacked_complex(values):
    """complex128 host array (n,) of 0-d values (numbers, numpy or cupy 0-d arrays): one transfer per cupy group."""
    device = [index for index, value in enumerate(values) if array_module_of(value) is not np]
    on_device = set(device)
    host = np.empty(len(values), dtype=np.complex128)
    for index, value in enumerate(values):
        if index not in on_device:
            host[index] = complex(np.asarray(value).item())
    if device:
        xp = array_module_of(values[device[0]])
        stacked = to_host(xp.stack([xp.asarray(values[index]).astype(xp.complex128) for index in device]))
        host[device] = stacked
    return host


class HarmonicProbeRecord:
    """Complex phasors (exp(+j omega t)) of one probe's components per solved frequency [Hz]; values are kept as given
    (numbers or numpy / cupy 0-d arrays: no host sync per record) and moved to the host by recorded()."""

    def __init__(self, components, unit='', label=''):
        components = (components,) if isinstance(components, str) else tuple(components)
        if not components:
            raise ValueError('HarmonicProbeRecord needs at least one component name')
        for component in components:
            if not isinstance(component, str) or not component.isidentifier():
                raise ValueError(f'HarmonicProbeRecord component names must be identifiers (they name the result '
                                 f'series <component>_re ...), got {component!r}')
        if len(set(components)) != len(components):
            raise ValueError(f'HarmonicProbeRecord component names must be unique, got {components}')
        self.components = components
        self.unit = str(unit)
        self.label = str(label)
        self._frequencies = []   # [Hz] in record order
        self._rows = []          # per record: the values in component order

    def __len__(self):
        return len(self._frequencies)

    def clear(self):
        """Forget every record (a new frequency sweep)."""
        self._frequencies.clear()
        self._rows.clear()

    def record(self, frequency, values, replace_same_frequency=True):
        """Store the phasors {component: value} (every component; a bare value for a one-component record) at
        frequency [Hz] > 0; replace_same_frequency: overwrite an earlier record of that frequency instead of keeping
        both."""
        frequency = float(frequency)
        if not (math.isfinite(frequency) and frequency > 0.0):
            raise ValueError(f'harmonic probe frequency must be a finite number > 0 Hz, got {frequency!r}')
        if not isinstance(values, dict):
            if len(self.components) != 1:
                raise ValueError(f'harmonic probe of components {", ".join(self.components)}: record a dict '
                                 f'{{component: value}}, got one value')
            values = {self.components[0]: values}
        missing = [component for component in self.components if component not in values]
        unknown = [str(name) for name in values if name not in self.components]
        if missing or unknown:
            raise ValueError(f"harmonic probe record: missing components {missing or 'none'}, unknown "
                             f"{unknown or 'none'}; components: {', '.join(self.components)}")
        row = tuple(values[component] for component in self.components)
        for component, value in zip(self.components, row):
            if np.ndim(value) != 0:
                raise ValueError(f'harmonic probe component {component!r}: one value per frequency (0-d), got shape '
                                 f'{np.shape(value)}')
        index = self.index_of(frequency) if replace_same_frequency else None
        if index is None:
            self._frequencies.append(frequency)
            self._rows.append(row)
        else:
            self._rows[index] = row

    def index_of(self, frequency):
        """Record index of frequency [Hz] (within FREQUENCY_MATCH_RTOL; the latest of repeated ones), None if
        unrecorded."""
        frequency = float(frequency)
        for index in range(len(self._frequencies) - 1, -1, -1):
            recorded = self._frequencies[index]
            if abs(recorded - frequency) <= FREQUENCY_MATCH_RTOL * max(abs(recorded), abs(frequency)):
                return index
        return None

    def recorded(self):
        """(frequencies float64 ascending [Hz], {component: complex128 (n,)}) on the host (repeated frequencies in
        record order); ValueError before the first record."""
        if not self._frequencies:
            raise ValueError(f"not solved yet: harmonic probe {self.label or ', '.join(self.components)} has no "
                             f'recorded frequency (run a frequency_response study first)')
        order = np.argsort(np.asarray(self._frequencies, dtype=np.float64), kind='stable')
        frequencies = np.asarray(self._frequencies, dtype=np.float64)[order]
        values = {component: _stacked_complex([self._rows[index][position] for index in order])
                  for position, component in enumerate(self.components)}
        return frequencies, values


def phasor_series(component, phasor, unit, series, units) -> None:
    """series[<component>_re / _im / _abs / _phase] (own contiguous float arrays, SERIES_PARTS order) of a complex
    array and units[...] (unit, 'rad' for the phase), in place."""
    parts = {'re': phasor.real, 'im': phasor.imag, 'abs': np.abs(phasor), 'phase': np.angle(phasor)}
    for part in SERIES_PARTS:
        name = series_name(component, part)
        series[name] = np.array(parts[part])   # a copy: the real / imag views share the phasor's memory
        units[name] = 'rad' if part == 'phase' else unit


def harmonic_probe_result(record) -> dict:
    """'xy' result of a HarmonicProbeRecord vs ascending frequency [Hz]: per component series <component>_re, _im,
    _abs (record unit) and _phase [rad, in (-pi, pi]]; extras frequency [Hz], components, unit."""
    frequencies, values = record.recorded()
    series, units = {}, {}
    for component in record.components:
        phasor_series(component, values[component], record.unit, series, units)
    return {'kind': 'xy', 'x': frequencies, 'x_label': 'Frequency', 'x_unit': 'Hz',
            'y_label': record.label or ', '.join(record.components), 'series': series, 'series_units': units,
            'frequency': frequencies.copy(), 'components': list(record.components), 'unit': record.unit}
