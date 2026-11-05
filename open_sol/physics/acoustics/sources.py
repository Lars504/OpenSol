"""Acoustic sources (point force, point volume injection, traction sheet on a grid line, thermoelastic heating pulse),
the heating envelope and the field probe; injection weights are built once on the host, the kernels scale them; arrays
live on the physics' backend (xp)."""
import math

import numpy as np

from ...core.backend import to_host
from ...core.constants import DTYPE
from ...core.half_storage import half_bits
from ...core.sources import ContinuousWave, GaussianPulse, RickerWavelet
from . import kernels as _kernels

PROBE_COMPONENTS = ('pressure', 'vx', 'vy', 'sxx', 'syy', 'sxy', 'stt')   # stt: axisymmetric only
VELOCITY_COMPONENTS = frozenset({'vx', 'vy'})
_STORED_UNITS = DTYPE(1.0)   # pressure kernel scale of a probe sample (stored units; get_data scales)


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

    def peak_stress(self, physics) -> float:
        return self.peak

    def inject_velocity(self, physics, t):
        value = float(self.temporal(t))
        if value != 0.0:
            fields, kernel, stored = physics.fields, physics.kernels['scatter'], value / physics.velocity_scale
            self.x.add(kernel, fields.vx, physics.vb_x, stored)
            self.y.add(kernel, fields.vy, physics.vb_y, stored)


class VolumeSource:
    """Volume injection at cells after the stress update at t_n + dt/2: weight = volume rate per unit z (axisymmetric:
    ring) length / cell area [1/s per unit waveform]."""

    def __init__(self, temporal, indices, weights, xp):
        self.temporal = temporal
        self.cells = _Scatter(indices, weights, xp)
        self._rate = max((abs(float(weight)) for weight in weights), default=0.0)

    def peak_stress(self, physics) -> float:
        """Stiffness x the strain of a quarter period [Pa]: rate / cell area x the largest c11 / (2 pi f)."""
        return self._rate * (physics.largest_stiffness or 0.0) / (2.0 * math.pi * _waveform_frequency(self.temporal))

    def inject_stress(self, physics, t, dt):
        value = float(self.temporal(t))
        cells = self.cells
        if value == 0.0 or not cells.n:
            return
        fields = physics.fields
        tail = (physics.c11, physics.c12, physics.c22, cells.index, cells.weight, cells.n_i32,
                DTYPE(dt * value / physics.stress_scale))
        if physics.axisymmetric:
            physics.kernels['volume_scatter_axisymmetric'](cells.launch[0], cells.launch[1],
                                                           (fields.sxx, fields.syy, fields.stt, *tail))
        else:
            physics.kernels['volume_scatter'](cells.launch[0], cells.launch[1], (fields.sxx, fields.syy, *tail))


class HeatPulseSource:
    """Thermoelastic stress of a heating power density [W/m^3] on its bounding box, times the envelope."""

    def __init__(self, box, heating_box, envelope):
        self.box = box
        self.heating = heating_box
        self.envelope = envelope

    def peak_stress(self, physics) -> float:
        """gruneisen x the absorbed energy density of the whole pulse [Pa], largest over the box (host sync)."""
        largest = abs(physics.materials.gruneisen[self.box] * self.heating).max()
        return float(largest) * self.envelope.integral

    def inject_stress(self, physics, t, dt):
        value = self.envelope(t)
        if value > 1e-30:
            box = self.box
            gruneisen, scale = physics.materials.gruneisen[box], DTYPE(dt * value / physics.stress_scale)
            _kernels.thermoelastic_stress(gruneisen, self.heating, scale, half_bits(physics.fields.sxx[box]),
                                          half_bits(physics.fields.syy[box]))
            if physics.axisymmetric:
                _kernels.thermoelastic_hoop_stress(gruneisen, self.heating, scale, half_bits(physics.fields.stt[box]))


class AcousticProbe:
    """Samples one component in a cell after every step: stresses / pressure at t_n+1, velocities (on the +x / +y face
    of the cell) at t_n+1/2, each with its own time; one host transfer in get_data. Samples are in stored units, times
    value_scale (half_storage) in get_data; the sample buffer lives on xp."""

    def __init__(self, ix, iy, component, xp, axisymmetric=False):
        if component not in PROBE_COMPONENTS:
            raise ValueError(f"probe component must be one of {', '.join(PROBE_COMPONENTS)}, got {component!r}")
        if component == 'stt' and not axisymmetric:
            raise ValueError("probe component 'stt' (hoop stress) needs geometry axisymmetric")
        self.ix, self.iy = int(ix), int(iy)
        self.field_component = component
        self.value_scale = 1.0                    # physical value per stored unit (set by AcousticPhysics)
        self._axisymmetric = bool(axisymmetric)   # pressure = -(srr + szz + stt) / 3
        self._velocity = component in VELOCITY_COMPONENTS
        self._xp = xp
        self._buf = xp.empty(1024, dtype=DTYPE)
        self._times = np.empty(1024, dtype=np.float64)
        self._n = 0

    def record(self, fields, t_full, dt):
        if self._n == self._buf.size:
            grown = self._xp.empty(self._buf.size * 2, dtype=DTYPE)
            grown[:self._n] = self._buf
            self._buf = grown
            grown_times = np.empty(self._times.size * 2, dtype=np.float64)
            grown_times[:self._n] = self._times
            self._times = grown_times
        ix, iy, n = self.ix, self.iy, self._n
        if self.field_component == 'pressure' and self._axisymmetric:
            _kernels.pressure_from_stress_axisymmetric(fields.sxx[ix:ix + 1, iy], fields.syy[ix:ix + 1, iy],
                                                       fields.stt[ix:ix + 1, iy], _STORED_UNITS, self._buf[n:n + 1])
        elif self.field_component == 'pressure':
            _kernels.pressure_from_stress(fields.sxx[ix:ix + 1, iy], fields.syy[ix:ix + 1, iy], _STORED_UNITS,
                                          self._buf[n:n + 1])
        else:
            self._buf[n] = getattr(fields, self.field_component)[ix, iy]
        self._times[n] = t_full - 0.5 * dt if self._velocity else t_full
        self._n = n + 1

    def get_data(self):
        """(times float64, values float32 in m/s / Pa) numpy copies; single host transfer."""
        values = np.array(to_host(self._buf[:self._n]))
        if self.value_scale != 1.0:
            values = values * np.float32(self.value_scale)
        return self._times[:self._n].copy(), values
