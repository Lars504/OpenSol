"""Monitors of em_fdtd_3d: point probe, spectral Poynting flux through a cell-edge plane or a closed box, and a DFT
field map on a z layer; their result dicts (flux_spectrum / dft_fields / timeseries kinds of the 2D monitors). The
flux-plane spectra (flux planes and boxes, far fields, mode monitors) live in a SpectrumStore of the stepper's
spectrum_precision."""
import ctypes
import math

import numpy as np
from ...core.backend import array_module_or_default, to_host

from ...core.constants import C0
from ...core.mirror import touch_factor
from ...core.monitors import FieldProbe
from ...solvers.explicit_kernels import SPECTRUM_PRECISIONS
from ..em.monitors import (_IMAG_BLOCK_KERNELS, _IMAG_KERNELS, _IMAG_LINE_KERNELS, _REAL_BLOCK_KERNELS, _REAL_KERNELS,
                           _REAL_LINE_KERNELS, _centre_pairs, _check_frequencies, _dft_phase, _field_sets)
from ..em.results import _per_cw, _shared_waveform_spectrum
from .boundaries3d import AXES

DFT_PLANE_QUANTITY = 'abs(E_x)^2 + abs(E_y)^2 + abs(E_z)^2'

# Block length of 'mixed' spectra (SpectrumStore). A block sums N samples x_i = ph_i f_i in complex64 (the product
# rounded, then N - 1 rounded adds), then joins a complex128 total and restarts from zero. Recursive float32
# summation errs by at most gamma_N sum |x_i|, gamma_N = N u / (1 - N u), u = 2^-24 (Higham, Accuracy and Stability of
# Numerical Algorithms, 2nd ed., eq. 4.4, per real component; Minkowski's inequality carries it to the complex
# modulus). Rounding errors that behave as independent and uniform in [-u, u] relative (std u / sqrt 3; Higham &
# Mary 2019, probabilistic error analysis) with partial sums |S_k| <= k max|x| give an rms error of u sqrt(N) / 3 x
# sum |x_i| (worst for in-phase terms, i.e. a resonant frequency). The blocks are added in float64, so the errors of
# the blocks add up to at most the same fractions of the run's gross sum |x_i| (rms: less, they are independent): the
# floor does not grow with the run length, unlike one float32 running sum over all M samples (gamma_M, ~M u). N = the
# largest power of two whose SPECTRUM_BLOCK_SIGMAS rms errors stay below SPECTRUM_BLOCK_ERROR of the block's gross
# sum: 64 (rms 1.6e-7, worst case gamma_64 = 3.8e-6). For comparison: 'double' sums lose ~1e-16 per sample, and both
# carry the complex64 rounding of the DFT phase (u |x_i| per sample, ~u sqrt(sum |x_i|^2) in all).
SPECTRUM_BLOCK_ERROR = 1e-6
SPECTRUM_BLOCK_SIGMAS = 6.0
_FLOAT32_ROUNDOFF = 2.0 ** -24


def spectrum_block_samples(relative_error=SPECTRUM_BLOCK_ERROR, sigmas=SPECTRUM_BLOCK_SIGMAS) -> int:
    """Largest power of two N with sigmas x the rms complex64 block error u sqrt(N) / 3 <= relative_error (of the
    block's gross sum |x_i|; derivation at SPECTRUM_BLOCK_ERROR); ValueError when not even N = 1 meets it."""
    limit = (3.0 * float(relative_error) / (float(sigmas) * _FLOAT32_ROUNDOFF)) ** 2
    if not limit >= 1.0:
        raise ValueError(f'a complex64 spectrum block cannot reach a relative error of {relative_error:g} at '
                         f'{sigmas:g} sigma (float32 unit round-off {_FLOAT32_ROUNDOFF:.3g})')
    return 1 << int(math.floor(math.log2(limit)))


SPECTRUM_BLOCK_SAMPLES = spectrum_block_samples()   # 64
# worst-case relative error of one block (gamma_N of SPECTRUM_BLOCK_SAMPLES), ~3.8e-6: the self-tests' bound
SPECTRUM_BLOCK_WORST_ERROR = (SPECTRUM_BLOCK_SAMPLES * _FLOAT32_ROUNDOFF
                              / (1.0 - SPECTRUM_BLOCK_SAMPLES * _FLOAT32_ROUNDOFF))


class SpectrumStore:
    """Running-DFT spectra of one monitor (`shapes`, complex) by the stepper's spectrum_precision. 'double': one
    complex128 array per spectrum on xp, the line kernels add every sample into it. 'mixed': one contiguous complex64
    block on xp (`arrays` are views of it, so the launch views stay valid), the block kernels (em.monitors
    _REAL_BLOCK_KERNELS / _IMAG_BLOCK_KERNELS: complex64 product and sum) add SPECTRUM_BLOCK_SAMPLES samples, then
    sampled() folds the block into a complex128 host total and zeroes it in place: gpu an asynchronous copy into a
    pinned staging buffer on the current stream with the zero fill queued behind it (no host sync in the step loop,
    never inside a step graph: monitors update eagerly), the host adds the staging at the next fold or read after one
    event wait on that copy (issued a block of samples earlier: done unless the host ran that far ahead). Readers take
    host() (complex128 totals, the current partial block folded in)."""

    def __init__(self, shapes, precision, xp):
        if precision not in SPECTRUM_PRECISIONS:
            raise ValueError(f"spectrum_precision must be one of {', '.join(SPECTRUM_PRECISIONS)}, got {precision!r}")
        self.precision = precision
        self.shapes = tuple(tuple(int(count) for count in shape) for shape in shapes)
        self._xp = xp
        self.folds = 0          # blocks folded into the host total (work counter)
        self._samples = 0       # samples in the device block since its last fold
        if precision == 'double':
            self.arrays = tuple(xp.zeros(shape, dtype=np.complex128) for shape in self.shapes)
            self.kernels = (_REAL_LINE_KERNELS, _IMAG_LINE_KERNELS)
            return
        self.kernels = (_REAL_BLOCK_KERNELS, _IMAG_BLOCK_KERNELS)
        offsets = np.cumsum([0] + [math.prod(shape) for shape in self.shapes]).tolist()
        self._offsets = offsets
        self._block = xp.zeros(offsets[-1], dtype=np.complex64)
        self.arrays = tuple(self._block[start:stop].reshape(shape)
                            for start, stop, shape in zip(offsets[:-1], offsets[1:], self.shapes))
        self._total = np.zeros(offsets[-1], dtype=np.complex128)   # host
        self._staging = None    # gpu: pinned complex64 copy of the last folded block
        self._copied = None     # gpu: event of the copy in flight into _staging (None: nothing pending)
        if xp is not np and offsets[-1]:
            memory = xp.cuda.alloc_pinned_memory(offsets[-1] * np.dtype(np.complex64).itemsize)
            self._staging = np.frombuffer(memory, dtype=np.complex64, count=offsets[-1])

    def sampled(self) -> None:
        """Count one sample (after its kernels were queued); a full 'mixed' block folds (no host sync)."""
        if self.precision == 'double':
            return
        self._samples += 1
        if self._samples >= SPECTRUM_BLOCK_SAMPLES:
            self._fold()

    def _fold(self) -> None:
        """The device block into the host total, then zeroed in place (gpu: async copy + zero fill, collected at the
        next fold or read)."""
        self._collect()
        block = self._block
        if self._xp is np:
            np.add(self._total, block, out=self._total)
            block.fill(0)
        elif block.size:
            stream = self._xp.cuda.get_current_stream()
            block.data.copy_to_host_async(ctypes.c_void_p(self._staging.ctypes.data), block.nbytes, stream)
            self._copied = stream.record()
            block.fill(0)   # queued behind the copy on the same stream
        self._samples = 0
        self.folds += 1

    def _collect(self) -> None:
        """Add the staging of the copy in flight (if any) into the host total after its event (gpu)."""
        if self._copied is None:
            return
        self._copied.synchronize()
        self._copied = None
        np.add(self._total, self._staging, out=self._total)

    def host(self) -> tuple:
        """Per spectrum the host complex128 total ('double': one transfer each, as the arrays are; 'mixed': the host
        total plus the current device block, one transfer, the block left as it is). Arrays the caller must not
        write."""
        if self.precision == 'double':
            return tuple(to_host(array) for array in self.arrays)
        self._collect()
        total = self._total + to_host(self._block)
        return tuple(total[start:stop].reshape(shape)
                     for start, stop, shape in zip(self._offsets[:-1], self._offsets[1:], self.shapes))


def point_probe(cell, field_component):
    """FieldProbe of the raw Yee sample at cell (i, j, k) after every step (H samples dt/2 older, as in 2D)."""
    return FieldProbe(int(cell[0]), int(cell[1]), field_component, iz=int(cell[2]))


def _face_samples(widths, first_edge, last_edge, wrap=False):
    """(face index slice, trapezoid weights) of the E / H samples on cell edges first_edge..last_edge (face f = hi face
    of cell f = edge f + 1); edge 0 has no face and is dropped; wrap (a full periodic axis): every face, dual widths."""
    if wrap:
        widths = np.asarray(widths, dtype=np.float64)
        return slice(0, widths.size), 0.5 * (widths + np.roll(widths, -1))
    edges = np.arange(max(first_edge, 1), last_edge + 1)
    weights = np.empty(edges.size)
    for position, edge in enumerate(edges):
        below = widths[edge - 1] if edge > first_edge else 0.0
        above = widths[edge] if edge < last_edge else 0.0
        weights[position] = 0.5 * (below + above)
    return slice(int(edges[0]) - 1, int(edges[-1])), weights


def _cell_samples(widths, first_edge, last_edge):
    """(cell slice, widths) of the cells between edges first_edge and last_edge."""
    return slice(first_edge, last_edge), np.asarray(widths[first_edge:last_edge], dtype=np.float64)


class PlaneFlux3D:
    """Running DFT (complex128 totals) of the tangential E / H on a plane at cell edge `edge` along `normal_axis` (0 /
    1 / 2) over the edge ranges bounds[t] = (first, last) of the two tangential axes; flux() > 0 toward +normal. A
    periodic tangential axis (periodic[t]) spanning its whole grid wraps its face weights; spectra on array module xp
    in a SpectrumStore of spectrum_precision ('double': complex128, 'mixed': complex64 blocks + host totals)."""

    def __init__(self, normal_axis, edge, bounds, frequencies, dt, widths, shape, periodic=(False, False, False),
                 xp=None, spectrum_precision='double'):
        xp = array_module_or_default(xp)
        n_normal = shape[normal_axis]
        if not 1 <= edge <= n_normal - 1:
            raise ValueError(f'flux plane at {AXES[normal_axis]} edge {edge} needs a cell on both sides (edges 1 to '
                             f'{n_normal - 1})')
        self.normal_axis = normal_axis
        self.edge = int(edge)
        self.bounds = tuple(tuple(int(value) for value in axis_bounds) for axis_bounds in bounds)
        self.frequencies = _check_frequencies(frequencies)
        self.dt = float(dt)
        self.sample_interval = 1
        first_axis, second_axis = (normal_axis + 1) % 3, (normal_axis + 2) % 3
        # pair 0: (E_first, H_second) on (first: faces, second: cells); pair 1: (E_second, H_first) on (cells, faces)
        self._pairs = []
        spectrum_shapes = []
        for e_axis, h_axis, face_axis, cell_axis, sign in ((first_axis, second_axis, first_axis, second_axis, 1.0),
                                                           (second_axis, first_axis, second_axis, first_axis, -1.0)):
            wrap = bool(periodic[face_axis]) and tuple(bounds[face_axis]) == (0, shape[face_axis])
            face_slice, face_weights = _face_samples(widths[face_axis], *bounds[face_axis], wrap=wrap)
            cell_slice, cell_weights = _cell_samples(widths[cell_axis], *bounds[cell_axis])
            lower_index, upper_index, h_index = [None] * 3, [None] * 3, [None] * 3
            for index in (lower_index, upper_index, h_index):
                index[face_axis], index[cell_axis] = face_slice, cell_slice
            lower_index[normal_axis], upper_index[normal_axis], h_index[normal_axis] = edge - 1, edge, edge - 1
            by_axis = {face_axis: face_weights, cell_axis: cell_weights}
            low_axis, high_axis = sorted((face_axis, cell_axis))   # view dims follow the grid axis order
            area = np.outer(by_axis[low_axis], by_axis[high_axis])
            spectrum_shapes += [(self.frequencies.size,) + area.shape] * 2   # E, H
            self._pairs.append({'e': 'e' + 'xyz'[e_axis], 'h': 'h' + 'xyz'[h_axis], 'sign': sign, 'area': area,
                                'e_axis': e_axis, 'h_axis': h_axis, 'face_axis': face_axis, 'cell_axis': cell_axis,
                                'face_slice': face_slice, 'cell_slice': cell_slice,
                                'lower': tuple(lower_index), 'upper': tuple(upper_index), 'h_index': tuple(h_index)})
        # spectra 2 p = E, 2 p + 1 = H of pair p (the kernels add into store.arrays)
        self.store = SpectrumStore(spectrum_shapes, spectrum_precision, xp)
        for index, pair in enumerate(self._pairs):
            pair['e_dft'], pair['h_dft'] = self.store.arrays[2 * index], self.store.arrays[2 * index + 1]
        self._omega_dev = xp.asarray(2.0 * np.pi * self.frequencies)
        self._phases = xp.empty(self.frequencies.size, dtype=np.complex64)   # rewritten in place per sample
        self._plane_phases = self._phases[:, None, None]                     # the view the kernels broadcast
        self.field_scales = (1.0, 1.0)   # (E, H) per stored unit (engine3d field_scale; half_storage != 1)
        self._view_cache = {}            # id(field set) -> (field set, plane views): real and Bloch imaginary sets

    def set_field_scales(self, e_scale, h_scale):
        self.field_scales = (float(e_scale), float(h_scale))

    def update(self, fields, t, phases=None):
        """E (mean of the two layers across the plane) and H at the plane, both phased at the E time t (Bloch k != 0:
        plus j times the imaginary set's); phases: the (n_f, 1, 1) DFT phases of t when a FluxBox3D computed them once
        for its six planes. The plane views are built once per field set (the Fields3D arrays are persistent)."""
        if phases is None:
            _dft_phase(self._omega_dev, t, self.dt * self.sample_interval, self._phases)
            phases = self._plane_phases
        for field_set, (accumulate, accumulate_pair) in _field_sets(fields, *self.store.kernels):
            for lower, upper, h_view, pair in self._views(field_set):
                accumulate_pair(lower, upper, phases, pair['e_dft'])
                accumulate(h_view, phases, pair['h_dft'])
        self.store.sampled()

    @property
    def spectrum_stores(self) -> tuple:
        return (self.store,)

    def _views(self, field_set):
        """[(E lower, E upper, H) plane views with a leading frequency axis, pair)] of a field set, cached by identity."""
        cached = self._view_cache.get(id(field_set))
        if cached is None or cached[0] is not field_set:
            views = []
            for pair in self._pairs:
                e_field, h_field = getattr(field_set, pair['e']), getattr(field_set, pair['h'])
                views.append((e_field[pair['lower']][None], e_field[pair['upper']][None],
                              h_field[pair['h_index']][None], pair))
            cached = self._view_cache[id(field_set)] = (field_set, views)
        return cached[1]

    def host_pairs(self):
        """Per pair (pair dict, host E spectrum, host H spectrum re-timed to the E time), complex128 (n_f, ...) in grid
        axis order of the two tangential axes (SpectrumStore.host: one transfer each, mixed one for the block)."""
        e_scale, h_scale = self.field_scales
        retime = h_scale * np.exp(1j * np.pi * self.frequencies * self.dt)[:, None, None]
        spectra = self.store.host()
        return [(pair, spectra[2 * index] * e_scale, spectra[2 * index + 1] * retime)
                for index, pair in enumerate(self._pairs)]

    def flux(self):
        """Spectral power (n_f,) float64 a.u.: cos(omega dt/2) Re(E H*) over the plane, H re-timed to the E time."""
        total = np.zeros(self.frequencies.size)
        for pair, e_spectrum, h_spectrum in self.host_pairs():
            total += pair['sign'] * np.einsum('fij,ij->f', np.real(e_spectrum * np.conj(h_spectrum)), pair['area'])
        return np.cos(np.pi * self.frequencies * self.dt) * total


class FluxBox3D:
    """Net outward flux through the six cell-edge faces of a box (edge ranges per axis, inclusive)."""

    def __init__(self, edge_bounds, frequencies, dt, widths, shape, periodic=(False, False, False), xp=None,
                 spectrum_precision='double'):
        self.frequencies = _check_frequencies(frequencies)
        self.edge_bounds = tuple(tuple(int(edge) for edge in bounds) for bounds in edge_bounds)
        self.faces = [(sign, PlaneFlux3D(axis, self.edge_bounds[axis][side], self.edge_bounds, self.frequencies, dt,
                                         widths, shape, periodic, xp=xp, spectrum_precision=spectrum_precision))
                      for axis in range(3) for side, sign in ((0, -1.0), (1, 1.0))]

    @property
    def spectrum_stores(self) -> tuple:
        return tuple(plane.store for _, plane in self.faces)

    @property
    def sample_interval(self):
        return self.faces[0][1].sample_interval

    @sample_interval.setter
    def sample_interval(self, value):
        for _, plane in self.faces:
            plane.sample_interval = value

    def set_field_scales(self, e_scale, h_scale):
        for _, plane in self.faces:
            plane.set_field_scales(e_scale, h_scale)

    def update(self, fields, t):
        """One DFT phase evaluation for the six planes (same frequencies, dt and sample interval)."""
        first = self.faces[0][1]
        _dft_phase(first._omega_dev, t, first.dt * first.sample_interval, first._phases)
        for _, plane in self.faces:
            plane.update(fields, t, phases=first._plane_phases)

    def flux(self):
        return sum(sign * plane.flux() for sign, plane in self.faces)


class DFTPlane3D:
    """Running DFT (complex64, array module xp) of E_x, E_y, E_z averaged to the cell centres of window layer
    k_grid. On a Bloch-phased periodic x / y axis (wrap_angles = engine bloch_angles k L) cell 0's lo face is the
    wrapped last face times e^{+j k L}: its own and wrapped face accumulate apart and intensity() averages them."""

    def __init__(self, window, k_grid, frequencies, dt, periodic, nz, xp=None, wrap_angles=(0.0, 0.0, 0.0)):
        xp = array_module_or_default(xp)
        self.window = window[:2]
        self.k_grid = int(k_grid)
        self.frequencies = _check_frequencies(frequencies)
        self.n_freqs = self.frequencies.size
        self.dt = float(dt)
        self.sample_interval = 1
        x_cells, y_cells = self.window
        self.shape = (x_cells.stop - x_cells.start, y_cells.stop - y_cells.start)
        pairs = []
        # component -> (wrap factor e^{+j k L}, own face, wrapped face (n_f, 1, ny) / (n_f, nx, 1) spectra)
        self._wrapped = {}
        for axis, (name, cells) in enumerate((('ex', x_cells), ('ey', y_cells))):
            axis_pairs = _centre_pairs(cells.start, cells.stop, periodic[axis])
            if periodic[axis] and wrap_angles[axis] and cells.start == 0:
                axis_pairs = axis_pairs[1:]   # the first cell's pair (own face, wrapped last face) accumulates apart
                row_shape = (self.n_freqs, 1, self.shape[1]) if axis == 0 else (self.n_freqs, self.shape[0], 1)
                self._wrapped[name] = (complex(np.exp(1j * float(wrap_angles[axis]))),
                                       xp.zeros(row_shape, dtype=np.complex64), xp.zeros(row_shape, dtype=np.complex64))
            pairs.append(axis_pairs)
        self._x_pairs, self._y_pairs = pairs
        self._k_lower = self.k_grid - 1 if self.k_grid >= 1 else (nz - 1 if periodic[2] else self.k_grid)
        spectrum_shape = (self.n_freqs,) + self.shape
        self.spectra = {name: xp.zeros(spectrum_shape, dtype=np.complex64) for name in ('ex', 'ey', 'ez')}
        self._omega_dev = xp.asarray(2.0 * np.pi * self.frequencies)
        self._phases = xp.empty(self.n_freqs, dtype=np.complex64)
        self.e_scale = 1.0   # E per stored unit (engine3d field_scale)

    def set_field_scales(self, e_scale, h_scale):
        self.e_scale = float(e_scale)

    def update(self, fields, t):
        _dft_phase(self._omega_dev, t, self.dt * self.sample_interval, self._phases)
        phases = self._phases[:, None, None]
        xs, ys = self.window
        k = self.k_grid
        for field_set, (accumulate, accumulate_pair) in _field_sets(fields, _REAL_KERNELS, _IMAG_KERNELS):
            for faces, partners, box in self._x_pairs:
                accumulate_pair(field_set.ex[faces, ys, k][None], field_set.ex[partners, ys, k][None], phases,
                                self.spectra['ex'][:, box, :])
            for faces, partners, box in self._y_pairs:
                accumulate_pair(field_set.ey[xs, faces, k][None], field_set.ey[xs, partners, k][None], phases,
                                self.spectra['ey'][:, :, box])
            accumulate_pair(field_set.ez[xs, ys, k][None], field_set.ez[xs, ys, self._k_lower][None], phases,
                            self.spectra['ez'])
            if 'ex' in self._wrapped:
                _, own, wrapped = self._wrapped['ex']
                accumulate(field_set.ex[0:1, ys, k][None], phases, own)
                accumulate(field_set.ex[-1:, ys, k][None], phases, wrapped)
            if 'ey' in self._wrapped:
                _, own, wrapped = self._wrapped['ey']
                accumulate(field_set.ey[xs, 0:1, k][None], phases, own)
                accumulate(field_set.ey[xs, -1:, k][None], phases, wrapped)

    def spectrum(self, name, index):
        """complex128 host centre phasor of component `name` at frequency `index` (stored units; the Bloch cell-0
        mean = (own + e^{+j k L} wrapped) / 2)."""
        values = to_host(self.spectra[name][index]).astype(np.complex128)
        if name in self._wrapped:
            factor, own, wrapped = self._wrapped[name]
            first = (slice(0, 1), slice(None)) if name == 'ex' else (slice(None), slice(0, 1))
            values[first] = 0.5 * (to_host(own[index]).astype(np.complex128)
                                   + factor * to_host(wrapped[index]).astype(np.complex128))
        return values

    def intensity(self, index):
        """float64 host |E_x|^2 + |E_y|^2 + |E_z|^2 at frequency `index`."""
        return self.e_scale ** 2 * sum(np.abs(self.spectrum(name, index)) ** 2 for name in self.spectra)


# ---- results ----
def mirror_factor(monitor, physics):
    """Unfolded-total factor of a PlaneFlux3D (its tangential extent) or FluxBox3D (every axis): 2 per mirror face whose
    window edge the extent reaches (a box face on the plane carries no flux; the box joins its image)."""
    window = physics.domain.physical_window()
    plane_normal = getattr(monitor, 'normal_axis', None)
    bounds = monitor.bounds if plane_normal is not None else monitor.edge_bounds
    touched = set()
    for axis_index, axis in enumerate(AXES):
        if axis_index == plane_normal:
            continue
        first, last = bounds[axis_index]
        if first <= window[axis_index].start:
            touched.add(f'{axis}_lo')
        if last >= window[axis_index].stop:
            touched.add(f'{axis}_hi')
    return touch_factor(physics.boundary_config.ghost_faces, touched)


def flux_result(monitor, physics) -> dict:
    """'flux_spectrum' of a PlaneFlux3D / FluxBox3D: 'power' a.u. (dt-weighted), 'power_per_cw' [W] = power / (2 |S|^2)
    when every source shares one waveform; x 'mirror_factor' (mirror_factor) for the unfolded structure."""
    frequency = np.array(monitor.frequencies, dtype=np.float64)
    factor = mirror_factor(monitor, physics)
    result = {'kind': 'flux_spectrum', 'frequency': frequency, 'wavelength': C0 / frequency,
              'power': factor * monitor.flux(), 'unit': 'a.u.', 'mirror_factor': factor}
    shared = _shared_waveform_spectrum(physics, frequency, monitor.sample_interval)
    if shared is not None:
        result['power_per_cw'] = _per_cw(result['power'], *shared, factor=2.0)
    return result


def dft_plane_result(monitor, domain, freq_index=None, physics=None) -> dict:
    """'dft_fields' of a DFTPlane3D (window xy centres of its layer, extra 'z' [m]); freq_index adds 'intensity'."""
    frequency = np.array(monitor.frequencies, dtype=np.float64)
    x_centers, y_centers, z_centers = domain.cell_centers()
    x_edges, y_edges, _ = domain.cell_edges()
    xs, ys = monitor.window
    result = {'kind': 'dft_fields', 'frequency': frequency, 'wavelength': C0 / frequency,
              'x_centers': np.array(x_centers[xs], dtype=np.float64),
              'y_centers': np.array(y_centers[ys], dtype=np.float64),
              'extent': (float(x_edges[xs.start]), float(x_edges[xs.stop]), float(y_edges[ys.start]),
                         float(y_edges[ys.stop])),
              'shape': tuple(int(count) for count in monitor.shape), 'quantity': DFT_PLANE_QUANTITY,
              'z': float(z_centers[monitor.k_grid])}
    shared = _shared_waveform_spectrum(physics, frequency, monitor.sample_interval)
    if shared is not None:
        result['intensity_per_cw_factor'] = _per_cw(np.ones_like(frequency), *shared)
    if freq_index is not None:
        if isinstance(freq_index, bool) or not isinstance(freq_index, (int, np.integer)):
            raise TypeError(f'freq_index must be an int, got {freq_index!r}')
        if not 0 <= int(freq_index) < monitor.n_freqs:
            raise IndexError(f'freq_index {freq_index} out of range for {monitor.n_freqs} DFT frequencies')
        result['freq_index'] = int(freq_index)
        result['intensity'] = monitor.intensity(int(freq_index))
    return result
