"""FieldProbe: time trace of one field component at one grid cell, buffered on the field's backend, its probe_result
dict, and the step-graph sample tables (ProbeRows; FieldProbeRows: every FieldProbe of an engine in one launch)."""
import numpy as np

from .backend import array_module_of, cp, to_host
from .half_storage import HALF_CONVERSIONS
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


def _device_cell_offset(field, cell, dtypes=REAL_C_TYPES):
    """Element offset of `cell` in a cupy field of one of `dtypes` (default float32 / float64; any strides), or None
    (numpy, other dtype, wrong rank, index outside the shape: the indexed path handles those)."""
    if not isinstance(field, cp.ndarray) or field.dtype not in dtypes or field.ndim != len(cell):
        return None
    byte_offset = 0
    for index, size, stride in zip(cell, field.shape, field.strides):
        if not 0 <= index < size:
            return None
        byte_offset += index * stride
    return byte_offset // field.itemsize


class ProbeRows:
    """Host side of a (rows, columns) float32 sample table on xp that one fixed-argument launch per step fills: the
    kernel writes the row of a device counter and advances it, so the launch can sit in a step graph; n_rows mirrors
    the counter, with host times and steps per row (stamp); the table doubles when full. Subclasses build the launch."""

    def __init__(self, xp):
        self._xp = xp
        self.n_columns = 0
        self._samples = None                              # float32 (capacity, columns) on xp
        self._times = np.empty(1024, dtype=np.float64)    # host time per row
        self._steps = np.empty(1024, dtype=np.float64)    # host step length per row (stamp)
        self.n_rows = 0
        self._row_counter = xp.zeros(1, dtype=np.int32)   # next row on xp (advanced by the kernel)
        self._n_columns_i32 = np.int32(0)
        self._launch = None   # persistent (kernel, grid, block, args) of the next record; None = rebuild

    def _add_column(self) -> int:
        """Index of a new column (zero before n_rows: its probe starts at the next row); the launch is rebuilt."""
        column = self.n_columns
        grown = self._xp.zeros((self._times.size, column + 1), dtype=np.float32)
        if self._samples is not None:
            grown[:, :column] = self._samples
        self._samples = grown
        self.n_columns = column + 1
        self._n_columns_i32 = np.int32(self.n_columns)
        self._launch = None
        return column

    def next_launch(self, *build_args, rows=1):
        """Persistent (kernel, grid, block, args) filling the next row, room for `rows` more rows made first (a
        multi-step block launches it once per step), None without columns; a new tuple after a column is added or the
        table grows (graph keys follow its id). The caller launches it after the step, then stamps each row."""
        if not self.n_columns:
            return None
        while self.n_rows + rows > self._times.size:
            self._grow()
        if self._launch is None:
            self._launch = self._build_launch(*build_args)
        return self._launch

    def _build_launch(self, *build_args):
        raise NotImplementedError

    def stamp(self, t, step=0.0) -> None:
        """Host time (and step length) of the row just recorded."""
        if not self.n_columns:
            return
        row = self.n_rows
        self._times[row] = t
        self._steps[row] = step
        self.n_rows = row + 1

    def _grow(self) -> None:
        capacity = 2 * self._times.size
        samples = self._xp.empty((capacity, self.n_columns), dtype=np.float32)
        samples[:self.n_rows] = self._samples[:self.n_rows]
        self._samples = samples
        self._launch = None
        for name in ('_times', '_steps'):
            grown = np.empty(capacity, dtype=np.float64)
            grown[:self.n_rows] = getattr(self, name)[:self.n_rows]
            setattr(self, name, grown)

    def column_values(self, column, start) -> np.ndarray:
        """Host float32 copy of one column's rows [start, n_rows) (one host transfer)."""
        rows = slice(start, self.n_rows)
        return np.array(to_host(self._xp.ascontiguousarray(self._samples[rows, column])), dtype=np.float32)


# Every FieldProbe of a FieldProbeRows in one launch per step: samples[row, k] = LOAD_REAL(field_k[offset_k]) (fields =
# a device table of the field pointers, one real_t; float16 fields: their bits, PTX-converted, stored units); row = the
# device counter, advanced by the launch (rewind 0) or the last row re-sampled (rewind 1: a field changed after the
# step graph's record). One block: every thread reads the row before thread 0 advances it.
_PROBE_ROWS_SOURCE = r'''
extern "C" __global__ void probe_rows(const unsigned long long* fields, const long long* offsets, float* samples,
                                      int* row_counter, const int n_probes, const int rewind)
{
    const long long row = (long long)row_counter[0] - rewind;
    for (int k = threadIdx.x; k < n_probes; k += blockDim.x) {
        const real_t* field = (const real_t*)fields[k];
        samples[row * n_probes + k] = LOAD_REAL(field[offsets[k]]);
    }
    __syncthreads();
    if (threadIdx.x == 0 && rewind == 0) row_counter[0] = (int)(row + 1);
}
'''
PROBE_ROWS_LAUNCH = ((1,), (128,))
# field dtype -> preamble of probe_rows (float16: the half_storage fields' uint16 bits, no cupy float16 kernel)
_PROBE_ROWS_PREAMBLES = {
    **{dtype: f'typedef {c_type} real_t;\n#define LOAD_REAL(value) ((float)(value))\n'
       for dtype, c_type in REAL_C_TYPES.items()},
    np.dtype(np.float16): HALF_CONVERSIONS + 'typedef unsigned short real_t;\n'
                          '#define LOAD_REAL(value) fp16_bits_to_float(value)\n',
}
_PROBE_ROWS_KERNELS = {}   # field dtype -> RawKernel (gpu only: FieldProbeRows serves cupy fields)


def _probe_rows_kernel(dtype):
    """probe_rows compiled for a float32 / float64 / float16 field dtype, built at the first use."""
    dtype = np.dtype(dtype)
    kernel = _PROBE_ROWS_KERNELS.get(dtype)
    if kernel is None:
        kernel = _PROBE_ROWS_KERNELS[dtype] = cp.RawKernel(_PROBE_ROWS_PREAMBLES[dtype] + _PROBE_ROWS_SOURCE,
                                                           'probe_rows')
    return kernel


class FieldProbeRows(ProbeRows):
    """FieldProbes of one engine's persistent cupy fields (one float32 / float64 / float16 dtype; float16 = stored
    units, the probe's value_scale applies in get_data) sampled by one 'probe_rows' launch per step (gpu only: cpu
    engines record per probe): add() takes a probe over, its get_data then reads its column; rewrite_last_row
    re-samples the step's row when a field changed after the record."""

    def __init__(self, fields, dtype):
        super().__init__(cp)
        self._fields = fields
        self.dtype = np.dtype(dtype)
        self.components = set()   # field components the probes read
        self._arrays, self._offsets_host = [], []   # the fields stay referenced (their pointers are in the table)
        self._pointers = self._offsets = None
        self._rewind = None

    def accepts(self, probe) -> bool:
        """Whether a FieldProbe's field is a cupy array of this dtype with its cell inside (else it records itself)."""
        field = getattr(self._fields, probe.field_component, None)
        return (field is not None and getattr(field, 'dtype', None) == self.dtype
                and _device_cell_offset(field, probe._cell, _PROBE_ROWS_PREAMBLES) is not None)

    def add(self, probe) -> None:
        """Take an accepted FieldProbe over from the next row on."""
        field = getattr(self._fields, probe.field_component)
        self._arrays.append(field)
        self._offsets_host.append(_device_cell_offset(field, probe._cell, _PROBE_ROWS_PREAMBLES))
        column = self._add_column()
        self._pointers = cp.asarray(np.array([array.data.ptr for array in self._arrays], dtype=np.uint64))
        self._offsets = cp.asarray(np.array(self._offsets_host, dtype=np.int64))
        self.components.add(probe.field_component)
        probe._attach(self, column, self.n_rows)

    def _build_launch(self):
        kernel = _probe_rows_kernel(self.dtype)
        grid, block = PROBE_ROWS_LAUNCH
        args = (self._pointers, self._offsets, self._samples, self._row_counter, self._n_columns_i32)
        self._rewind = (kernel, grid, block, args + (np.int32(1),))
        return kernel, grid, block, args + (np.int32(0),)

    def rewrite_last_row(self) -> None:
        """Re-sample every probe into the row recorded this step (eager; after next_launch of the step)."""
        kernel, grid, block, args = self._rewind
        kernel(grid, block, args)


class FieldProbe:
    """Records `getattr(fields, field_component)[ix, iy]` (3D: `[ix, iy, iz]`) into a buffer on the field's backend
    (allocated at the first record; one host transfer in get_data); a probe taken over by a FieldProbeRows is sampled
    by its engine instead (record refuses)."""

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
        self._rows = None        # FieldProbeRows that samples this probe (column, first row), None = own buffers
        self._column = self._start = 0

    def _attach(self, rows, column, start) -> None:
        """Sampled from now on by `rows` (column `column` from row `start`); own samples recorded before are dropped."""
        self._rows, self._column, self._start = rows, int(column), int(start)

    @property
    def recorded_by_rows(self) -> bool:
        """Whether a FieldProbeRows samples this probe (its engine records it; record would raise)."""
        return self._rows is not None

    def clear(self):
        """Forget the recorded samples (buffers kept for the next record)."""
        if self._rows is not None:
            self._start = self._rows.n_rows
        self._n = 0

    def record(self, fields, t: float, replace_same_time: bool = False):
        """Append the sample at time t; replace_same_time: overwrite the last sample instead when it has that time."""
        if self._rows is not None:
            raise RuntimeError('this FieldProbe is sampled by its engine (FieldProbeRows): record it there')
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
        if self._rows is not None:
            raise RuntimeError('this FieldProbe is sampled by its engine (FieldProbeRows): record it there')
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
        rows = self._rows
        if rows is not None:
            values = rows.column_values(self._column, self._start)
            if self.value_scale != 1.0:
                values = (values * self.value_scale).astype(np.float32)
            return rows._times[self._start:rows.n_rows].copy(), values
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
