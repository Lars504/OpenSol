"""FieldProbe: time trace of one field component at one grid cell, buffered on the field's backend, and its
probe_result dict."""
import numpy as np

from .backend import array_module_of, to_host


class FieldProbe:
    """Records `getattr(fields, field_component)[ix, iy]` (3D: `[ix, iy, iz]`) into a buffer on the field's backend
    (allocated at the first record; one host transfer in get_data)."""

    def __init__(self, ix: int, iy: int, field_component: str = 'ez', iz: int = None):
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self._cell = (ix, iy) if iz is None else (ix, iy, iz)
        self.field_component = field_component
        # Growable sample buffers: values on the field's device (no per-step device->host sync), times on the host.
        self._buf = None
        self._times = np.empty(1024, dtype=np.float64)
        self._n = 0
        self.value_scale = 1.0   # physical value per stored unit (em3d half_storage fields), applied in get_data

    def clear(self):
        """Forget the recorded samples (buffers kept for the next record)."""
        self._n = 0

    def record(self, fields, t: float, replace_same_time: bool = False):
        """Append the sample at time t; replace_same_time: overwrite the last sample instead when it has that time."""
        self.record_value(getattr(fields, self.field_component)[self._cell], t, replace_same_time)

    def record_value(self, value, t: float, replace_same_time: bool = False):
        """Append a 0-d sample of the field's backend (e.g. a sum of several fields at the cell) at time t; arguments
        as record."""
        if replace_same_time and self._n and abs(self._times[self._n - 1] - t) <= 1e-12 * abs(t):
            self._n -= 1
        if self._buf is None:
            self._buf = array_module_of(value).empty(self._times.size, dtype=np.float32)
        if self._n == self._buf.size:
            grown = array_module_of(self._buf).empty(self._buf.size * 2, dtype=self._buf.dtype)
            grown[:self._n] = self._buf
            self._buf = grown
            grown_times = np.empty(self._times.size * 2, dtype=np.float64)
            grown_times[:self._n] = self._times
            self._times = grown_times
        # Device-to-device scalar store: no host sync.
        self._buf[self._n] = value
        self._times[self._n] = t
        self._n += 1

    def get_data(self):
        """(times float64, values float32) numpy copies; single host transfer."""
        if self._buf is None:
            values = np.empty(0, dtype=np.float32)
        else:
            values = np.array(to_host(self._buf[:self._n]))
        if self.value_scale != 1.0:
            values = (values * self.value_scale).astype(np.float32)
        return self._times[:self._n].copy(), values


def probe_result(probe, unit: str = '') -> dict:
    """Timeseries result dict of a FieldProbe: numpy 't' [s] and 'values', plus component name and unit."""
    times, values = probe.get_data()
    return {'kind': 'timeseries', 't': times, 'values': values,
            'component': probe.field_component, 'unit': unit}
