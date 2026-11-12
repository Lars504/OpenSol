"""FieldProbe: time trace of one field component at one grid cell, buffered on the field's backend, and its
probe_result dict."""
import numpy as np

from .backend import array_module_of, cp, to_host
from .kernels import REAL_C_TYPES, real_raw_twin

# trace[slot] = field[field_offset] (element offset from the field view's first element, any strides): one launch
# per record instead of a 0-d view plus a cupy setitem (2D FDTD cavity: ~11 -> ~5 us of host time per sample)
_PROBE_STORE_SOURCE = r'''
extern "C" __global__ void probe_store(const real_t* field, float* trace, const long long field_offset,
                                       const long long slot)
{
    trace[slot] = (float)field[field_offset];
}
'''


def _probe_store_host(field, trace, field_offset, slot):
    """Host twin of probe_store for a C-contiguous field (FieldProbe.record sends numpy fields through record_value)."""
    trace[int(slot)] = field.reshape(-1)[int(field_offset)]


_probe_store = real_raw_twin('core.probe_store', _PROBE_STORE_SOURCE, 'probe_store', _probe_store_host)
_ONE_THREAD = (1,)


def _device_cell_offset(field, cell):
    """Element offset of `cell` in a float32 / float64 cupy field (any strides), or None (numpy, other dtype, wrong
    rank, index outside the shape: the indexed path handles those)."""
    if not isinstance(field, cp.ndarray) or field.dtype not in REAL_C_TYPES or field.ndim != len(cell):
        return None
    byte_offset = 0
    for index, size, stride in zip(cell, field.shape, field.strides):
        if not 0 <= index < size:
            return None
        byte_offset += index * stride
    return byte_offset // field.itemsize


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
        field = getattr(fields, self.field_component)
        field_offset = _device_cell_offset(field, self._cell)
        if field_offset is None:
            self.record_value(field[self._cell], t, replace_same_time)
            return
        slot = self._next_slot(field, t, replace_same_time)
        _probe_store(_ONE_THREAD, _ONE_THREAD, (field, self._buf, np.int64(field_offset), np.int64(slot)))

    def record_value(self, value, t: float, replace_same_time: bool = False):
        """Append a 0-d sample of the field's backend (e.g. a sum of several fields at the cell) at time t; arguments
        as record."""
        slot = self._next_slot(value, t, replace_same_time)
        # Device-to-device scalar store: no host sync.
        self._buf[slot] = value

    def _next_slot(self, value, t: float, replace_same_time: bool) -> int:
        """Index the sample at time t goes to (buffers allocated on value's backend / grown; its time stored)."""
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
        slot = self._n
        self._times[slot] = t
        self._n += 1
        return slot

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
