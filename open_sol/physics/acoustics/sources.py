"""Acoustic sources (point force, point volume injection, traction sheet on a grid line, thermoelastic heating pulse),
the heating envelope and the field probe; injection weights are built once on the host, the kernels scale them; arrays
live on the physics' backend (xp)."""
import math

import numpy as np

from ...core.constants import DTYPE
from ...core.half_storage import half_bits
from ...core.monitors import ProbeRows
from ...core.sources import ContinuousWave, GaussianPulse, RickerWavelet
from . import kernels as _kernels

PROBE_COMPONENTS = ('pressure', 'vx', 'vy', 'sxx', 'syy', 'sxy', 'stt')   # stt: axisymmetric only
VELOCITY_COMPONENTS = frozenset({'vx', 'vy'})
# |waveform| / envelope values at or below this are not injected (a pulse's tail: far below the float32 resolution of
# any field it excited), so those steps replay the fixed step graph instead of launching scatters
NEGLIGIBLE_AMPLITUDE = 1e-30


def _waveform_frequency(temporal):
    """Centre / peak frequency [Hz] of a core.sources waveform (f0, or the Ricker f_peak)."""
    return float(getattr(temporal, 'f0', None) or getattr(temporal, 'f_peak'))


def make_waveform(waveform, frequency, bandwidth=0.5):
    """Temporal waveform s(t) (peak ~1) of 'gaussian' (sin carrier), 'ricker' or 'cw' at `frequency` [Hz]."""
    if waveform == 'gaussian':
        return GaussianPulse(frequency, bandwidth)
    if waveform == 'ricker':
        return RickerWavelet(frequency)
    if waveform == 'cw':
        return ContinuousWave(frequency)
    raise ValueError(f"waveform must be 'gaussian', 'ricker' or 'cw', got {waveform!r}")


class HeatingPulse:
    """Dimensionless heating envelope exp(-((t - delay) / width)^2); its time integral is sqrt(pi) width."""

    def __init__(self, width, delay=None):
        if not width > 0.0:
            raise ValueError(f'heating pulse width must be > 0 s, got {width!r}')
        self.width = float(width)
        self.delay = 3.0 * self.width if delay is None else float(delay)

    def __call__(self, t):
        return math.exp(-((float(t) - self.delay) / self.width) ** 2)

    @property
    def duration(self) -> float:
        """End of the pulse [s]: delay + 4.5 width (envelope exp(-20), like the 2 t0 of a gaussian waveform)."""
        return self.delay + 4.5 * self.width

    @property
    def integral(self) -> float:
        """Time integral of the envelope [s] (absorbed energy density = power density * integral)."""
        return math.sqrt(math.pi) * self.width


_GAUSSIAN_EXPONENT = -math.log(NEGLIGIBLE_AMPLITUDE)   # exp(-u) <= NEGLIGIBLE_AMPLITUDE for u >= this


def _ricker_exponent():
    """u* with (2u - 1) exp(-u) <= NEGLIGIBLE_AMPLITUDE for every u >= u* (fixed point of u = L + ln(2u - 1))."""
    exponent = _GAUSSIAN_EXPONENT
    for _ in range(30):
        exponent = _GAUSSIAN_EXPONENT + math.log(2.0 * exponent - 1.0)
    return exponent


_RICKER_EXPONENT = _ricker_exponent()
_WINDOW_MARGIN = 1.001   # on the half width: the bound holds with rounding to spare


def active_window(waveform):
    """(t_start, t_end) outside which |waveform(t)| <= NEGLIGIBLE_AMPLITUDE for sure (gaussian, ricker, heating
    pulse); (-inf, inf) for any other waveform (cw, a plain callable)."""
    if isinstance(waveform, GaussianPulse):
        centre, half = waveform.t0, waveform.tau * math.sqrt(_GAUSSIAN_EXPONENT)
    elif isinstance(waveform, RickerWavelet):
        centre, half = waveform.t0, math.sqrt(_RICKER_EXPONENT) / (math.pi * waveform.f_peak)
    elif isinstance(waveform, HeatingPulse):
        centre, half = waveform.delay, waveform.width * math.sqrt(_GAUSSIAN_EXPONENT)
    else:
        return -math.inf, math.inf
    half *= _WINDOW_MARGIN
    return centre - half, centre + half


def window_steps(window, n_step, t, dt, half_step, max_steps):
    """(first step index, start times t_n) of the steps n >= n_step whose source time (t_n, or t_n + 0.5 * dt with
    half_step: AcousticPhysics.advance's t_half) lies in the closed active window, t_n continuing t at n_step by t_n+1 =
    t_n + dt (the runner clock); (n_step, []) for an empty window, None when it is unbounded or ends more than max_steps
    steps ahead."""
    start, end = window
    if start > end:
        return n_step, []
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    first, times, time, n = None, [], t, n_step
    while True:
        source_time = time + 0.5 * dt if half_step else time
        if source_time > end:
            break
        if n - n_step >= max_steps:
            return None
        if source_time >= start:
            if first is None:
                first = n
            times.append(time)
        n += 1
        time = time + dt
    return (n_step if first is None else first), times


def _table_arguments(values, first, xp, step_counter):
    """(table, step_counter, table_start, table_length) tail of a VALUE_TABLE scatter launch."""
    table = xp.asarray(np.asarray(values, dtype=np.float32))
    return table, step_counter, np.int32(first), np.int32(len(values))


class _Scatter:
    """(index, weight) lists of one source on one field, arrays on xp (empty lists are skipped)."""

    def __init__(self, indices, weights, xp):
        self.n = len(indices)
        self.index = xp.asarray(np.asarray(indices, dtype=np.int32))
        self.weight = xp.asarray(np.asarray(weights, dtype=np.float32))
        self.n_i32 = np.int32(self.n)
        self.launch = _kernels.scatter_launch(self.n)

    def add(self, kernel, field, coefficient, value):
        """field[index] += coefficient[index] * weight * value (kernel: the physics' 'scatter' field kernel)."""
        if self.n:
            kernel(self.launch[0], self.launch[1],
                   (field, coefficient, self.index, self.weight, self.n_i32, DTYPE(value)))


class VelocitySource:
    """Body force injected into vx / vy after the velocity update at t_n: v += vb * weight * temporal(t_n); weights
    hold force density per unit waveform [N/m^3] (the vb coefficient carries dt / rho_face); peak = the stress it
    drives [Pa] (order of magnitude, half_storage scale)."""

    def __init__(self, temporal, x_entries, y_entries, peak, xp):
        self.temporal = temporal
        self.x = _Scatter(*x_entries, xp)
        self.y = _Scatter(*y_entries, xp)
        self.peak = float(peak)
        # no entries: an empty window (nothing ever injected)
        self._start, self._end = active_window(temporal) if self.x.n or self.y.n else (math.inf, -math.inf)

    def peak_stress(self, physics) -> float:
        return self.peak

    def amplitude(self, t) -> float:
        """Waveform value at t, 0.0 when negligible (outside the active window, or no entries): nothing to inject."""
        if not self._start <= t <= self._end:
            return 0.0
        value = float(self.temporal(t))
        return value if abs(value) > NEGLIGIBLE_AMPLITUDE else 0.0

    def inject_velocity(self, physics, value):
        """Scatter waveform value `value` (from amplitude(t_n)) into vx / vy."""
        fields, kernel, stored = physics.fields, physics.kernels['scatter'], value / physics.velocity_scale
        self.x.add(kernel, fields.vx, physics.vb_x, stored)
        self.y.add(kernel, fields.vy, physics.vb_y, stored)

    def table_launches(self, physics, n_step, t, dt, step_counter, max_steps):
        """Fixed 'scatter_table' launches (one per non-empty face list) injecting what inject_velocity would at every
        step n >= n_step of the active window (window_steps), () once it has passed; None when it is unbounded or
        ends more than max_steps steps ahead (eager injection)."""
        steps = window_steps((self._start, self._end), n_step, t, dt, False, max_steps)
        if steps is None:
            return None
        first, times = steps
        if not times:
            return ()
        scale = physics.velocity_scale
        tail = _table_arguments([self.amplitude(time) / scale for time in times], first, physics.xp, step_counter)
        kernel, fields = physics.kernels['scatter_table'], physics.fields
        return tuple((kernel, *scatter.launch, (field, coefficient, scatter.index, scatter.weight, scatter.n_i32, *tail))
                     for scatter, field, coefficient in ((self.x, fields.vx, physics.vb_x),
                                                         (self.y, fields.vy, physics.vb_y)) if scatter.n)


class VolumeSource:
    """Volume injection at cells after the stress update at t_n + dt/2: weight = volume rate per unit z (axisymmetric:
    ring) length / cell area [1/s per unit waveform]."""

    def __init__(self, temporal, indices, weights, xp):
        self.temporal = temporal
        self.cells = _Scatter(indices, weights, xp)
        self._rate = max((abs(float(weight)) for weight in weights), default=0.0)
        self._start, self._end = active_window(temporal) if self.cells.n else (math.inf, -math.inf)

    def peak_stress(self, physics) -> float:
        """Stiffness x the strain of a quarter period [Pa]: rate / cell area x the largest c11 / (2 pi f)."""
        return self._rate * (physics.largest_stiffness or 0.0) / (2.0 * math.pi * _waveform_frequency(self.temporal))

    def amplitude(self, t) -> float:
        """Waveform value at t, 0.0 when negligible (outside the active window, or no cell): nothing to inject."""
        if not self._start <= t <= self._end:
            return 0.0
        value = float(self.temporal(t))
        return value if abs(value) > NEGLIGIBLE_AMPLITUDE else 0.0

    def inject_stress(self, physics, value, dt):
        """Volume injection of waveform value `value` (from amplitude(t_n + dt/2)) over the step dt."""
        cells = self.cells
        fields = physics.fields
        tail = (physics.c11, physics.c12, physics.c22, cells.index, cells.weight, cells.n_i32,
                DTYPE(dt * value / physics.stress_scale))
        if physics.axisymmetric:
            physics.kernels['volume_scatter_axisymmetric'](cells.launch[0], cells.launch[1],
                                                           (fields.sxx, fields.syy, fields.stt, *tail))
        else:
            physics.kernels['volume_scatter'](cells.launch[0], cells.launch[1], (fields.sxx, fields.syy, *tail))

    def table_launches(self, physics, n_step, t, dt, step_counter, max_steps):
        """The fixed 'volume_scatter_table' launch injecting what inject_stress would (amplitude at t_n + dt / 2) at
        every step n >= n_step of the active window, () once it has passed; None as VelocitySource.table_launches."""
        steps = window_steps((self._start, self._end), n_step, t, dt, True, max_steps)
        if steps is None:
            return None
        first, times = steps
        if not times:
            return ()
        scale = physics.stress_scale
        table = _table_arguments([dt * self.amplitude(time + 0.5 * dt) / scale for time in times], first, physics.xp,
                                 step_counter)
        cells, fields = self.cells, physics.fields
        stresses = (fields.sxx, fields.syy, fields.stt) if physics.axisymmetric else (fields.sxx, fields.syy)
        kernel = physics.kernels['volume_scatter_table_axisymmetric' if physics.axisymmetric else 'volume_scatter_table']
        return ((kernel, *cells.launch, (*stresses, physics.c11, physics.c12, physics.c22, cells.index, cells.weight,
                                         cells.n_i32, *table)),)


class HeatPulseSource:
    """Thermoelastic stress of a heating power density [W/m^3] on its bounding box, times the envelope."""

    def __init__(self, box, heating_box, envelope):
        self.box = box
        self.heating = heating_box
        self.envelope = envelope
        self._start, self._end = active_window(envelope)

    def peak_stress(self, physics) -> float:
        """gruneisen x the absorbed energy density of the whole pulse [Pa], largest over the box (host sync)."""
        largest = abs(physics.materials.gruneisen[self.box] * self.heating).max()
        return float(largest) * self.envelope.integral

    def amplitude(self, t) -> float:
        """Envelope value at t, 0.0 when negligible (or outside the active window)."""
        if not self._start <= t <= self._end:
            return 0.0
        value = self.envelope(t)
        return value if value > NEGLIGIBLE_AMPLITUDE else 0.0

    def inject_stress(self, physics, value, dt):
        """Thermoelastic stress of envelope value `value` (from amplitude(t_n + dt/2)) over the step dt."""
        box = self.box
        gruneisen, scale = physics.materials.gruneisen[box], DTYPE(dt * value / physics.stress_scale)
        _kernels.thermoelastic_stress(gruneisen, self.heating, scale, half_bits(physics.fields.sxx[box]),
                                      half_bits(physics.fields.syy[box]))
        if physics.axisymmetric:
            _kernels.thermoelastic_hoop_stress(gruneisen, self.heating, scale, half_bits(physics.fields.stt[box]))


class ProbeRecorder(ProbeRows):
    """Every AcousticProbe of one physics, sampled by one 'probe_record' field kernel per step into the (rows, probes)
    float32 table of core.monitors.ProbeRows (stored units) with host times t_n+1 and dt per row: one launch per step
    instead of one or two cupy operations per probe (launch-bound small grids), fixed arguments (step graphs)."""

    def __init__(self, ny, xp, axisymmetric=False):
        super().__init__(xp)
        self.ny = int(ny)   # flat cell index i * ny + j
        self._axisymmetric = bool(axisymmetric)
        self.probes = []
        self._cells = self._codes = None   # int32 (probes,) on xp, rebuilt when a probe is added

    def add(self, ix, iy, component) -> 'AcousticProbe':
        """A probe of `component` in cell (ix, iy); added after steps were recorded it starts at the next row."""
        if component not in PROBE_COMPONENTS:
            raise ValueError(f"probe component must be one of {', '.join(PROBE_COMPONENTS)}, got {component!r}")
        if component == 'stt' and not self._axisymmetric:
            raise ValueError("probe component 'stt' (hoop stress) needs geometry axisymmetric")
        probe = AcousticProbe(self, self._add_column(), self.n_rows, int(ix), int(iy), component)
        if component == 'pressure' and self._axisymmetric:
            probe.code_name = 'pressure_axisymmetric'
        self.probes.append(probe)
        xp = self._xp
        self._cells = xp.asarray(np.array([item.ix * self.ny + item.iy for item in self.probes], dtype=np.int32))
        self._codes = xp.asarray(np.array([_kernels.PROBE_RECORD_CODES[item.code_name] for item in self.probes],
                                          dtype=np.int32))
        return probe

    def _build_launch(self, kernel, fields):
        """(kernel, grid, block, args) of 'probe_record' over every probe (next_launch(kernel, fields, rows=))."""
        stt = fields.sxx if fields.stt is None else fields.stt   # a planar dummy: no planar probe reads stt
        grid, block = _kernels.PROBE_RECORD_LAUNCH
        return (kernel, grid, block,
                (fields.vx, fields.vy, fields.sxx, fields.syy, fields.sxy, stt, self._cells, self._codes,
                 self._samples, self._row_counter, self._n_columns_i32))

    def column(self, probe) -> tuple:
        """(host times float64, host stored samples float32) of one probe's rows (one host transfer): stresses /
        pressure at the stamped t_n+1, velocities half a step earlier."""
        values = self.column_values(probe.column, probe.start)
        rows = slice(probe.start, self.n_rows)
        times = self._times[rows] - 0.5 * self._steps[rows] if probe.velocity else self._times[rows].copy()
        return times, values


class AcousticProbe:
    """Samples one component in a cell after every step (recorded by its physics' ProbeRecorder): stresses / pressure
    at t_n+1, velocities (on the +x / +y face of the cell) at t_n+1/2; samples in stored units, times value_scale
    (half_storage) in get_data."""

    def __init__(self, recorder, column, start, ix, iy, component):
        self.recorder = recorder
        self.column, self.start = int(column), int(start)
        self.ix, self.iy = int(ix), int(iy)
        self.field_component = component
        self.code_name = component              # PROBE_RECORD_CODES key (set by the recorder)
        self.value_scale = 1.0                  # physical value per stored unit (set by AcousticPhysics)
        self.velocity = component in VELOCITY_COMPONENTS

    def get_data(self):
        """(times float64, values float32 in m/s / Pa) numpy copies; single host transfer."""
        times, values = self.recorder.column(self)
        if self.value_scale != 1.0:
            values = values * np.float32(self.value_scale)
        return times, values
