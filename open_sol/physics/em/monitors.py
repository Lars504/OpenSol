"""Field probes, DFT monitors, flux monitors, and NF2FF for 2D FDTD; the near-to-far transform of the box face phasors
(near_to_far_pattern) is shared with the FDFD HarmonicNearToFar."""

from dataclasses import dataclass

import numpy as np
from ...core.backend import array_module_or_default, cp, to_host
from ...core.kernels import twin

from ...core.constants import C0, ETA0
from ...core.mirror import image_sign
from ...core.monitors import FieldProbe
from .yee_grid import lower_face

__all__ = ['FieldProbe', 'DFTMonitor', 'FluxMonitor', 'NearToFarField', 'NearToFarGeometry', 'near_to_far_geometry',
           'near_to_far_pattern']



# Every DFT kernel is a core.kernels twin (also em3d's): the numpy twins follow the cupy sources.
# Device-side DFT phase: ph_k = exp(-1j*omega_k*t) * dt, computed in float64 and stored complex64.
def _dft_phase_host(omega, t, dt, ph):
    phase = -omega * float(t)
    ph[...] = (np.cos(phase) * float(dt)).astype(np.float32) + 1j * (np.sin(phase) * float(dt)).astype(np.float32)


_dft_phase = twin('em.dft_phase', cp.ElementwiseKernel(
    'float64 omega, float64 t, float64 dt',
    'complex64 ph',
    '''
    double p = -omega * t;
    ph = complex<float>((float)(cos(p) * dt), (float)(sin(p) * dt));
    ''',
    'dft_phase'), _dft_phase_host)


def _dft_accum_host(f, ph, acc):
    acc += ph * f


def _dft_accum_pair_host(a, b, ph, acc):
    acc += ph * (np.float32(0.5) * (a + b))


def _dft_accum_line_host(f, ph, acc):
    acc += ph.astype(np.complex128) * f.astype(np.float64)


def _dft_accum_pair_line_host(a, b, ph, acc):
    acc += ph.astype(np.complex128) * (0.5 * (a.astype(np.float64) + b.astype(np.float64)))


def _dft_accum_imag_host(f, ph, acc):
    acc += (1j * ph).astype(np.complex64) * f


def _dft_accum_pair_imag_host(a, b, ph, acc):
    acc += (1j * ph).astype(np.complex64) * (np.float32(0.5) * (a + b))


def _dft_accum_line_imag_host(f, ph, acc):
    acc += 1j * ph.astype(np.complex128) * f.astype(np.float64)


def _dft_accum_pair_line_imag_host(a, b, ph, acc):
    acc += 1j * ph.astype(np.complex128) * (0.5 * (a.astype(np.float64) + b.astype(np.float64)))


# Fused accumulate: acc += field * phase (broadcast), no temporaries. F = the field dtype (float32, float64 for the double
# 2D stepper; em3d float32): box spectra stay complex64 (the sample rounded to float), line spectra complex128.
_dft_accum = twin('em.dft_accumulate', cp.ElementwiseKernel(
    'F f, complex64 ph',
    'complex64 acc',
    'acc += ph * (float)f',
    'dft_accumulate'), _dft_accum_host)

# Fused collocate + accumulate: acc += phase * mean of two face samples (an E face and its lo-side partner).
_dft_accum_pair = twin('em.dft_accumulate_pair', cp.ElementwiseKernel(
    'F a, F b, complex64 ph',
    'complex64 acc',
    'acc += ph * (float)((F)0.5 * (a + b))',
    'dft_accumulate_pair'), _dft_accum_pair_host)

# Line-monitor twins (flux, NF2FF, mode monitor) summing in complex128: (n_freqs, n_cells) is cheap, and float32 running
# sums over 1e5-1e6 steps lose the near-cancelling net fluxes and deep dips. The large DFTMonitor boxes stay complex64.
_dft_accum_line = twin('em.dft_accumulate_line', cp.ElementwiseKernel(
    'F f, complex64 ph',
    'complex128 acc',
    'acc += complex<double>(ph.real(), ph.imag()) * (double)f',
    'dft_accumulate_line'), _dft_accum_line_host)

_dft_accum_pair_line = twin('em.dft_accumulate_pair_line', cp.ElementwiseKernel(
    'F a, F b, complex64 ph',
    'complex128 acc',
    'acc += complex<double>(ph.real(), ph.imag()) * (0.5 * ((double)a + (double)b))',
    'dft_accumulate_pair_line'), _dft_accum_pair_line_host)

# Imaginary field set of a Bloch k != 0 engine (fields.imag): the DFT of F_re + j F_im adds j ph F_im, so the spectrum
# at +omega is the response at the Bloch k (the negative frequencies hold the -k response). Twins of the four above.
_dft_accum_imag = twin('em.dft_accumulate_imag', cp.ElementwiseKernel(
    'F f, complex64 ph',
    'complex64 acc',
    'acc += complex<float>(-ph.imag(), ph.real()) * (float)f',
    'dft_accumulate_imag'), _dft_accum_imag_host)

_dft_accum_pair_imag = twin('em.dft_accumulate_pair_imag', cp.ElementwiseKernel(
    'F a, F b, complex64 ph',
    'complex64 acc',
    'acc += complex<float>(-ph.imag(), ph.real()) * (float)((F)0.5 * (a + b))',
    'dft_accumulate_pair_imag'), _dft_accum_pair_imag_host)

_dft_accum_line_imag = twin('em.dft_accumulate_line_imag', cp.ElementwiseKernel(
    'F f, complex64 ph',
    'complex128 acc',
    'acc += complex<double>(-ph.imag(), ph.real()) * (double)f',
    'dft_accumulate_line_imag'), _dft_accum_line_imag_host)

_dft_accum_pair_line_imag = twin('em.dft_accumulate_pair_line_imag', cp.ElementwiseKernel(
    'F a, F b, complex64 ph',
    'complex128 acc',
    'acc += complex<double>(-ph.imag(), ph.real()) * (0.5 * ((double)a + (double)b))',
    'dft_accumulate_pair_line_imag'), _dft_accum_pair_line_imag_host)

# (accumulate, accumulate pair) kernels of the real and of the imaginary field set, box and line monitors.
_REAL_KERNELS = (_dft_accum, _dft_accum_pair)
_IMAG_KERNELS = (_dft_accum_imag, _dft_accum_pair_imag)
_REAL_LINE_KERNELS = (_dft_accum_line, _dft_accum_pair_line)
_IMAG_LINE_KERNELS = (_dft_accum_line_imag, _dft_accum_pair_line_imag)


def _field_sets(fields, real_kernels, imag_kernels):
    """((fields, kernels), ...) of the real set and, under Bloch k != 0, the imaginary set fields.imag."""
    imaginary = getattr(fields, 'imag', None)
    if imaginary is None:
        return ((fields, real_kernels),)
    return ((fields, real_kernels), (imaginary, imag_kernels))

_POLARIZATIONS = ('ez', 'hz')

# NF2FF (E, H) components on the left, right, bottom, top faces per polarisation.
_NF2FF_FACE_COMPONENTS = {'ez': (('ez', 'hy'), ('ez', 'hy'), ('ez', 'hx'), ('ez', 'hx')),
                          'hz': (('ey', 'hz'), ('ey', 'hz'), ('ex', 'hz'), ('ex', 'hz'))}


def _check_polarization(polarization, label):
    """`polarization` when it is 'ez' or 'hz'; ValueError naming the monitor otherwise."""
    if polarization not in _POLARIZATIONS:
        raise ValueError(f"{label} polarization must be one of {_POLARIZATIONS}, got {polarization!r}")
    return polarization


def dft_sample_slice(start: int, stop: int, stride: int) -> slice:
    """Cells a DFTMonitor samples along one axis of the box [start, stop): every stride-th cell, offset to the middle of
    the first stride-cell block (min(stride, box cells) // 2), so the samples centre on their blocks."""
    return slice(start + min(stride, stop - start) // 2, stop, stride)


def _centre_pairs(lo: int, hi: int, periodic: bool, stride: int = 1):
    """(face cells, partner face cells, box cells) slice triples averaging the hi faces of the sampled cells
    (dft_sample_slice of [lo, hi)) with their lo faces (cell - 1); cell 0 pairs with yee_grid.lower_face (wrapped last
    face or its own hi face)."""
    samples = dft_sample_slice(lo, hi, stride)
    if samples.start >= 1:
        return ((samples, slice(samples.start - 1, hi - 1, stride), slice(None)),)
    partner = lower_face(0, periodic)   # -1 (last face) or 0
    first_cell = (slice(0, 1), slice(partner, partner + 1 or None), slice(0, 1))
    if hi == 1:
        return (first_cell,)
    return (first_cell, (slice(1, hi), slice(0, hi - 1), slice(1, hi)))


# highest harmonic of the source band a nonlinear medium (chi2 SHG, Kerr / TPA THG) puts into the fields: the DFT
# decimation keeps its alias off the monitor bands (higher cascades are weak)
NONLINEAR_HARMONIC_ORDER = 3


def dft_sample_interval(dt, sources, monitor_frequencies, harmonic_order=1) -> int:
    """Steps m between DFT samples (each weighted m dt): the largest m with 1 / (m dt) >= max monitor frequency +
    harmonic_order x the sources' spectral_limit, so no alias of significant source content (nonlinear media:
    harmonic_order NONLINEAR_HARMONIC_ORDER, their SHG / THG) hits a monitor; 1 without one (cw, no source)."""
    if not sources or not monitor_frequencies:
        return 1
    source_limits = []
    for source in sources:
        spectral_limit = getattr(getattr(source, 'temporal', None), 'spectral_limit', None)
        limit = spectral_limit() if spectral_limit is not None else None
        if limit is None:
            return 1
        source_limits.append(float(limit))
    sample_rate = float(harmonic_order) * max(source_limits) + max(float(np.max(f)) for f in monitor_frequencies)
    return max(1, int(np.floor(1.0 / (sample_rate * float(dt)))))


def _check_frequencies(frequencies, label='frequencies'):
    """Non-empty 1D float64 copy of finite positive values (frequencies [Hz] or wavelengths [m])."""
    frequencies = np.atleast_1d(np.array(frequencies, dtype=np.float64))
    if frequencies.ndim != 1 or frequencies.size == 0 or not np.all(np.isfinite(frequencies) & (frequencies > 0)):
        raise ValueError(f"monitor {label} must be a non-empty 1D list of finite positive values, "
                         f"got {frequencies}")
    return frequencies


class _LineSpectra:
    """complex128 running DFTs (n_freqs, n_samples) of line samples phased at the E sample time t (flux, NF2FF, mode
    monitors); per spectrum (component, cells, lower_cells or None = no face mean onto the centres, n_samples); xp = the
    run backend's array module (None = the default backend)."""

    def __init__(self, samples, frequencies, dt, xp=None):
        xp = array_module_or_default(xp)
        self.frequencies = frequencies
        self.dt = float(dt)
        self._samples = tuple(samples)
        self.spectra = tuple(xp.zeros((frequencies.size, n_samples), dtype=np.complex128)
                             for _, _, _, n_samples in self._samples)
        self._omega_dev = xp.asarray(2.0 * np.pi * frequencies)   # float64, on the run backend
        self._phases = xp.empty(frequencies.size, dtype=np.complex64)
        self._launches = None   # (fields, [(kernel, args)]): basic-index views, valid while the arrays live in place

    def update(self, fields, t, sample_interval):
        _dft_phase(self._omega_dev, t, self.dt * sample_interval, self._phases)
        if self._launches is None or self._launches[0] is not fields:
            self._launches = (fields, self._launch_list(fields))
        for kernel, arguments in self._launches[1]:
            kernel(*arguments)

    def _launch_list(self, fields):
        """[(accumulate kernel, its arguments)] of every spectrum and field set (views built once per field set)."""
        phases = self._phases[:, None]
        launches = []
        for field_set, (accumulate, accumulate_pair) in _field_sets(fields, _REAL_LINE_KERNELS, _IMAG_LINE_KERNELS):
            for spectrum, (component, cells, lower_cells, _) in zip(self.spectra, self._samples):
                field = getattr(field_set, component)
                if lower_cells is None:
                    launches.append((accumulate, (field[cells][None, :], phases, spectrum)))
                else:
                    launches.append((accumulate_pair, (field[cells][None, :], field[lower_cells][None, :], phases,
                                                       spectrum)))
        return launches

    def host(self, index, retime_h=False):
        """Host copy of spectrum `index`; retime_h moves an H-type sample (taken at t - dt/2) to the E time t."""
        spectrum = np.array(to_host(self.spectra[index]))   # a copy also on the cpu (re-timed in place below)
        if retime_h:
            spectrum *= np.exp(1j * np.pi * self.frequencies * self.dt)[:, None]
        return spectrum


class DFTMonitor:
    """Running DFT over the cell box (x_slice, y_slice) at each monitor frequency: Ez ('ez'), or Hz plus Ex / Ey
    collocated to the cell centres ('hz'; periodic_x / periodic_y wrap the partner face of grid cell 0); downsample
    s > 1 records only every s-th cell along x and y (x_samples / y_samples, dft_sample_slice); xp = the run backend's
    array module (None = the default backend)."""

    def __init__(self, x_slice: slice, y_slice: slice, frequencies, dt: float, polarization: str = 'ez',
                 periodic_x: bool = False, periodic_y: bool = False, downsample: int = 1, xp=None):
        xp = array_module_or_default(xp)
        for label, cells in (('x_slice', x_slice), ('y_slice', y_slice)):
            if cells.step not in (None, 1) or cells.stop is None or not (cells.start or 0) < cells.stop:
                raise ValueError(f"DFTMonitor {label} must be a non-empty unit-step slice, got {cells}")
        if isinstance(downsample, bool) or int(downsample) != downsample or downsample < 1:
            raise ValueError(f"DFTMonitor downsample must be an int >= 1, got {downsample!r}")
        self.polarization = _check_polarization(polarization, 'DFTMonitor')
        self.x_slice = x_slice
        self.y_slice = y_slice
        self.downsample = int(downsample)
        self.x_samples = dft_sample_slice(x_slice.start or 0, x_slice.stop, self.downsample)
        self.y_samples = dft_sample_slice(y_slice.start or 0, y_slice.stop, self.downsample)
        self.frequencies = _check_frequencies(frequencies)
        self.dt = float(dt)
        self.n_freqs = len(self.frequencies)

        self.mx = len(range(self.x_samples.start, self.x_samples.stop, self.downsample))
        self.my = len(range(self.y_samples.start, self.y_samples.stop, self.downsample))

        spectrum_shape = (self.n_freqs, self.mx, self.my)
        if polarization == 'ez':
            self.ez_dft = xp.zeros(spectrum_shape, dtype=np.complex64)
        else:
            self.hz_dft = xp.zeros(spectrum_shape, dtype=np.complex64)
            self.ex_dft = xp.zeros(spectrum_shape, dtype=np.complex64)
            self.ey_dft = xp.zeros(spectrum_shape, dtype=np.complex64)
            self._phases_h = xp.empty(self.n_freqs, dtype=np.complex64)   # Hz sample time t - dt/2
            # Ex (hi-y faces) is averaged along y, Ey (hi-x faces) along x.
            self._ex_pairs = _centre_pairs(y_slice.start or 0, y_slice.stop, bool(periodic_y), self.downsample)
            self._ey_pairs = _centre_pairs(x_slice.start or 0, x_slice.stop, bool(periodic_x), self.downsample)

        self._omega_dev = xp.asarray(2.0 * np.pi * self.frequencies)   # float64, on the run backend
        self._phases = xp.empty(self.n_freqs, dtype=np.complex64)
        self.sample_interval = 1   # engine calls update every sample_interval steps (EMPhysics.finalize sets it)
        self._launches = None   # (fields, [(kernel, args)]): basic-index views, valid while the arrays live in place

    def update(self, fields, t: float):
        # Phase exp(-i omega t) * m dt from the actual sample time t, computed on the device.
        sample_weight = self.dt * self.sample_interval
        _dft_phase(self._omega_dev, t, sample_weight, self._phases)
        if self.polarization == 'hz':   # Hz holds t - dt/2 when the monitors run (Ex, Ey hold t)
            _dft_phase(self._omega_dev, t - 0.5 * self.dt, sample_weight, self._phases_h)
        if self._launches is None or self._launches[0] is not fields:
            self._launches = (fields, self._launch_list(fields))
        for kernel, arguments in self._launches[1]:
            kernel(*arguments)

    def _launch_list(self, fields):
        """[(accumulate kernel, its arguments)] of every spectrum and field set (views built once per field set)."""
        phases = self._phases[:, None, None]
        xs, ys = self.x_samples, self.y_samples
        launches = []
        for field_set, (accumulate, accumulate_pair) in _field_sets(fields, _REAL_KERNELS, _IMAG_KERNELS):
            if self.polarization == 'ez':
                launches.append((accumulate, (field_set.ez[xs, ys][None, :, :], phases, self.ez_dft)))
                continue
            launches.append((accumulate, (field_set.hz[xs, ys][None, :, :], self._phases_h[:, None, None],
                                          self.hz_dft)))
            for faces, partners, box in self._ex_pairs:
                launches.append((accumulate_pair, (field_set.ex[xs, faces][None, :, :],
                                                   field_set.ex[xs, partners][None, :, :], phases,
                                                   self.ex_dft[:, :, box])))
            for faces, partners, box in self._ey_pairs:
                launches.append((accumulate_pair, (field_set.ey[faces, ys][None, :, :],
                                                   field_set.ey[partners, ys][None, :, :], phases,
                                                   self.ey_dft[:, box, :])))
        return launches


class FluxMonitor:
    """Spectral Poynting flux through the hi face of a cell line (axis 'x' = line at x-cell `position` spanning y), > 0
    toward +x / +y. Ez at the line-cell centres, H on the hi face ('ez'); E on the hi face, Hz at the centres ('hz')."""

    def __init__(self, axis: str, position: int, span: tuple,
                 frequencies, dt: float, cell_widths, polarization: str = 'ez', xp=None):
        if axis not in ('x', 'y'):
            raise ValueError(f"FluxMonitor axis must be 'x' or 'y', got {axis!r}")
        span = (int(span[0]), int(span[1]))
        if not 0 <= span[0] < span[1]:
            raise ValueError(f"FluxMonitor span must be a half-open cell range lo < hi, got {span}")
        cell_widths = np.asarray(cell_widths, dtype=np.float64)
        if cell_widths.ndim != 1 or cell_widths.size < span[1]:
            raise ValueError(f"FluxMonitor cell_widths (shape {cell_widths.shape}) must cover span {span}")
        self.polarization = _check_polarization(polarization, 'FluxMonitor')
        self.axis = axis
        self.position = int(position)
        self.span = span
        self.frequencies = _check_frequencies(frequencies)
        self.dt = float(dt)
        self.n_freqs = len(self.frequencies)

        n_cells = span[1] - span[0]
        line = (self.position, slice(*span)) if axis == 'x' else (slice(*span), self.position)

        if polarization == 'ez':
            # Ez and the tangential H that carries the normal Poynting flux (Hy for axis 'x', Hx for axis 'y'):
            # S_x = -Ez Hy, S_y = +Ez Hx.
            e_component, h_component = 'ez', ('hy' if axis == 'x' else 'hx')
            self._flux_sign = -1.0 if axis == 'x' else 1.0
        else:
            # The tangential E (Ey on the hi-x face for axis 'x', Ex on the hi-y face for axis 'y') and Hz:
            # S_x = +Ey Hz, S_y = -Ex Hz.
            e_component, h_component = ('ey' if axis == 'x' else 'ex'), 'hz'
            self._flux_sign = 1.0 if axis == 'x' else -1.0
        # Spectra 0 = E, 1 = H, both phased at the E sample time t; H (sampled at t - dt/2) is re-timed in flux().
        self._spectra = _LineSpectra(((e_component, line, None, n_cells), (h_component, line, None, n_cells)),
                                     self.frequencies, self.dt, xp=xp)

        # Widths of the line cells along the line [m] (dy for axis 'x', dx for axis 'y'), host float64.
        self._cell_widths = cell_widths[span[0]:span[1]].copy()
        self.sample_interval = 1   # as DFTMonitor

    def update(self, fields, t: float):
        self._spectra.update(fields, t, self.sample_interval)

    def flux(self):
        """Spectral power per unit z length (n_freqs,) float64: the normal Re(E x H*) summed over the line, E averaged
        over steps n, n+1 to the H time n+1/2. Host float64: the products underflow float32 (~1e-40)."""
        e_spectra = self._spectra.host(0)
        h_spectra = self._spectra.host(1, retime_h=True)
        # The E average 0.5 (1 + e^{-j omega dt}) = cos(omega dt/2) e^{-j omega dt/2}; the phase is conj(H re-timing).
        e_time_average = np.cos(np.pi * self.frequencies * self.dt)
        integrand = self._flux_sign * e_time_average[:, None] * np.real(e_spectra * np.conj(h_spectra))
        return integrand @ self._cell_widths


@dataclass(frozen=True, eq=False)
class NearToFarGeometry:
    """Closed sample contour [m] through the cell centres of a near-to-far box: columns x0, x1 over rows [y0, y1] (left
    / right faces) and rows y0, y1 over columns [x0, x1] (bottom / top faces), trapezoid weights (half at corners)."""
    x_left: float
    x_right: float
    y_bottom: float
    y_top: float
    y_side: np.ndarray       # (y1 - y0 + 1,) row centres of the left / right faces
    dy_side: np.ndarray      # their trapezoid weights
    x_side: np.ndarray       # (x1 - x0 + 1,) column centres of the bottom / top faces
    dx_side: np.ndarray      # their trapezoid weights
    face_cells: tuple        # grid index of the left, right, bottom, top samples
    lower_face_cells: tuple  # the same one cell back along each face normal (x - 1 left / right, y - 1 bottom / top):
                             # the mean of a face-staggered field (hi faces) there and at face_cells sits on the centres
    images: tuple = ()       # mirror planes (face, coordinate [m], E parity sign, touching): the pattern adds the image
                             # contour; a touching box (its face in the first cell off the plane) drops that face


def _trapezoid_weights(centers):
    """Trapezoid weights [m] of samples at `centers` (float64, >= 2 points): half the gap to each neighbour."""
    half_gaps = 0.5 * np.diff(centers)
    weights = np.zeros(centers.size)
    weights[:-1] += half_gaps
    weights[1:] += half_gaps
    return weights


def near_to_far_geometry(x0, x1, y0, y1, cell_centers, label, mirrors=None):
    """NearToFarGeometry of the half-open cell box (Domain.cell_centers); ValueError naming `label` for an empty box, a
    sample column x1 / row y1 outside the grid or x0 / y0 = 0 (no face below them). mirrors = placement.mirror_planes
    {face: (coordinate, sign)}: the pattern unfolds over them (ValueError for mirrors on both faces of an axis)."""
    x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
    if not (0 <= x0 < x1 and 0 <= y0 < y1):
        raise ValueError(f"{label} box [{x0}, {x1}) x [{y0}, {y1}) must be non-empty")
    x_centers, y_centers = (np.asarray(c, dtype=np.float64) for c in cell_centers)
    if x1 >= x_centers.size or y1 >= y_centers.size:
        raise ValueError(f"{label} samples column {x1} / row {y1} outside the "
                         f"{x_centers.size} x {y_centers.size} grid")
    if x0 < 1 or y0 < 1:
        raise ValueError(f"{label} averages the faces below column {x0} / row {y0}: both must be >= 1")
    images = []
    for face, (coordinate, sign) in sorted((mirrors or {}).items()):
        axis = face[0]
        if f"{axis}_{'hi' if face.endswith('lo') else 'lo'}" in (mirrors or {}):
            raise ValueError(f"{label}: mirror faces on both {axis} sides describe an infinite array of images; the "
                             'far field needs a mirror on one side of an axis at most')
        centers, first, last = (x_centers, x0, x1) if axis == 'x' else (y_centers, y0, y1)
        # touching: no cell centre between the contour face and the plane (the face lies in the first cell off it)
        touching = (coordinate > centers[first - 1] if face.endswith('lo')
                    else last + 1 >= centers.size or coordinate < centers[last + 1])
        images.append((face, float(coordinate), int(sign), bool(touching)))
    side_rows, row_columns = slice(y0, y1 + 1), slice(x0, x1 + 1)
    return NearToFarGeometry(
        x_left=float(x_centers[x0]), x_right=float(x_centers[x1]),
        y_bottom=float(y_centers[y0]), y_top=float(y_centers[y1]),
        y_side=y_centers[side_rows].copy(), dy_side=_trapezoid_weights(y_centers[side_rows]),
        x_side=x_centers[row_columns].copy(), dx_side=_trapezoid_weights(x_centers[row_columns]),
        face_cells=((x0, side_rows), (x1, side_rows), (row_columns, y0), (row_columns, y1)),
        lower_face_cells=((x0 - 1, side_rows), (x1 - 1, side_rows), (row_columns, y0 - 1), (row_columns, y1 - 1)),
        images=tuple(images))


# (E, H) parity (core.mirror) of the contour samples per polarisation and face kind: left / right faces carry the
# y-face pair, bottom / top the x-face pair.
_CONTOUR_PARITIES = {'ez': {'side': ('polar_z', 'axial_y'), 'row': ('polar_z', 'axial_x')},
                     'hz': {'side': ('polar_y', 'axial_z'), 'row': ('polar_x', 'axial_z')}}
_SIDE_KINDS = ('left', 'right')
_MIRRORED_KIND = {'x': {'left': 'right', 'right': 'left'}, 'y': {'bottom': 'top', 'top': 'bottom'}}


def _contour_segments(polarization, geometry, faces):
    """[(kind, fixed coordinate, along coordinates, weights, E, H)] of the box contour plus its mirror images
    (geometry.images, in order): a touching plane drops the face beside it, extends the end weights of the faces
    crossing it by the gap to their image samples, and every segment gains its image (kind, coordinates, signs)."""
    (e_left, h_left), (e_right, h_right), (e_bottom, h_bottom), (e_top, h_top) = faces
    segments = [('left', geometry.x_left, geometry.y_side, geometry.dy_side, e_left, h_left),
                ('right', geometry.x_right, geometry.y_side, geometry.dy_side, e_right, h_right),
                ('bottom', geometry.y_bottom, geometry.x_side, geometry.dx_side, e_bottom, h_bottom),
                ('top', geometry.y_top, geometry.x_side, geometry.dx_side, e_top, h_top)]
    for face, plane, sign, touching in geometry.images:
        axis, lo = face[0], face.endswith('lo')
        normal_kinds = _SIDE_KINDS if axis == 'x' else ('bottom', 'top')
        if touching:
            dropped = normal_kinds[0] if lo else normal_kinds[1]
            kept = []
            for kind, fixed, along, weights, e, h in segments:
                if kind == dropped:
                    continue
                if kind not in normal_kinds:   # crosses the plane: its end sample meets its image half a gap away
                    weights = weights.copy()
                    end = int(np.argmin(along)) if lo else int(np.argmax(along))
                    weights[end] += abs(float(along[end]) - plane)
                kept.append((kind, fixed, along, weights, e, h))
            segments = kept
        images = []
        for kind, fixed, along, weights, e, h in segments:
            e_parity, h_parity = _CONTOUR_PARITIES[polarization]['side' if kind in _SIDE_KINDS else 'row']
            e_sign, h_sign = image_sign(e_parity, axis, sign), image_sign(h_parity, axis, sign)
            if kind in normal_kinds:   # parallel to the plane: the image face sits across it, opposite normal
                images.append((_MIRRORED_KIND[axis][kind], 2.0 * plane - fixed, along, weights, e_sign * e, h_sign * h))
            else:
                images.append((kind, fixed, 2.0 * plane - along, weights, e_sign * e, h_sign * h))
        segments = segments + images
    return segments


def _face_term(polarization, kind, e, h, cos_t, sin_t):
    """Far-field integrand of one face kind (outward normal n): equivalent currents J = n x H, M = -n x E; e, h = E and
    eta0 H. ez: Ez_ff ~ (r x M)_z - eta0 J_z; hz: eta0 Hz_ff ~ -eta0 (r x J)_z - M_z (e.g. the left face, n = -x, has
    J = Hz y and M = Ey z: -cos eta0 Hz - Ey)."""
    if polarization == 'ez':   # J_z and in-plane M
        if kind == 'left':
            return h - cos_t * e
        if kind == 'right':
            return cos_t * e - h
        return -sin_t * e - h if kind == 'bottom' else sin_t * e + h
    if kind == 'left':         # in-plane J and M_z (r x (r x z) = -z)
        return -cos_t * h - e
    if kind == 'right':
        return cos_t * h + e
    return -sin_t * h + e if kind == 'bottom' else sin_t * h - e


def near_to_far_pattern(polarization, frequencies, angles, faces, geometry):
    """Complex far-field pattern sqrt(rho) E_z ('ez') or sqrt(rho) eta0 H_z ('hz') at rho -> inf (phase referred to
    e^{-j k0 rho}), (n_freqs, n_angles) at `angles` [rad from +x], of `faces` = ((E, H) left, right, bottom, top)
    complex (n_freqs, n_side) host phasors at the contour samples, H at the E sample time; vacuum outside the box.
    With geometry.images the pattern is that of the unfolded structure (box contour plus its mirror images)."""
    _check_polarization(polarization, 'near_to_far_pattern')
    frequencies = np.atleast_1d(np.asarray(frequencies, dtype=np.float64))
    angles = np.asarray(angles, dtype=np.float64)
    cos_t = np.cos(angles)[:, None]
    sin_t = np.sin(angles)[:, None]
    eta0 = float(ETA0)
    segments = _contour_segments(polarization, geometry, faces)

    pattern = np.empty((frequencies.size, angles.size), dtype=np.complex128)
    for k, frequency in enumerate(frequencies):
        k0 = 2.0 * np.pi * frequency / C0
        # With N, L the integrals of J, M weighted by exp(j k0 r.r') dl and the asymptotic 2D Green function -j/4
        # H0^(2)(k0 rho), the far field is E = sqrt(k0 / (8 pi rho)) e^{j pi/4} e^{-j k0 rho} [r x L - eta0 N]_t and
        # eta0 H = r x E.
        total = np.zeros(angles.size, dtype=np.complex128)
        for kind, fixed, along, weights, e, h in segments:
            if kind in _SIDE_KINDS:
                phase = np.exp(1j * k0 * (fixed * cos_t + along[None, :] * sin_t))
            else:
                phase = np.exp(1j * k0 * (along[None, :] * cos_t + fixed * sin_t))
            term = _face_term(polarization, kind, e[k][None, :], eta0 * h[k][None, :], cos_t, sin_t)
            total += np.sum(term * phase * weights[None, :], axis=1)
        pattern[k] = np.sqrt(k0 / (8.0 * np.pi)) * np.exp(0.25j * np.pi) * total
    return pattern


class NearToFarField:
    """2D near-to-far-field transform over the closed centre contour of the cell box (NearToFarGeometry): columns x0, x1
    (Ez/Hy for 'ez', Hz/Ey for 'hz') and rows y0, y1 (Ez/Hx, Hz/Ex), the face field averaged onto the centres."""

    def __init__(self, x0, x1, y0, y1, frequencies, dt, cell_centers, polarization='ez', mirrors=None, xp=None):
        self.polarization = _check_polarization(polarization, 'NearToFarField')
        self.geometry = near_to_far_geometry(x0, x1, y0, y1, cell_centers, 'NearToFarField', mirrors)
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        self.x0, self.x1 = x0, x1
        self.y0, self.y1 = y0, y1
        self.frequencies = _check_frequencies(frequencies)
        self.dt = float(dt)
        self.n_freqs = len(self.frequencies)

        # Spectra 2k = E, 2k + 1 = H of face k (left, right, bottom, top), both phased at the E sample time t; H
        # (sampled at t - dt/2) is re-timed in far_field(). The hi-face field (ez H, hz E) is averaged onto the centres.
        staggered_e = polarization == 'hz'
        samples = []
        for (e_component, h_component), cells, lower_cells, n_samples in zip(
                _NF2FF_FACE_COMPONENTS[polarization], self.geometry.face_cells, self.geometry.lower_face_cells,
                (y1 - y0 + 1,) * 2 + (x1 - x0 + 1,) * 2):
            samples.append((e_component, cells, lower_cells if staggered_e else None, n_samples))
            samples.append((h_component, cells, None if staggered_e else lower_cells, n_samples))
        self._spectra = _LineSpectra(samples, self.frequencies, self.dt, xp=xp)
        self.sample_interval = 1   # as DFTMonitor

    def update(self, fields, t: float):
        self._spectra.update(fields, t, self.sample_interval)

    def far_field(self, angles, freq_index=None):
        """Complex far-field pattern sqrt(rho) E_z ('ez') or sqrt(rho) eta0 H_z ('hz') of the dt-weighted spectra (a.u.)
        at `angles` [rad from +x]: (n_angles,) for one frequency index, (n_freqs, n_angles) for None."""
        indices = np.arange(self.n_freqs) if freq_index is None else [int(freq_index)]
        frequencies = self.frequencies[indices]
        # Host (E, H at the E sample time) face spectra in the order left, right, bottom, top.
        faces = tuple((self._spectra.host(2 * face)[indices], self._spectra.host(2 * face + 1, retime_h=True)[indices])
                      for face in range(4))
        pattern = near_to_far_pattern(self.polarization, frequencies, angles, faces, self.geometry)
        return pattern if freq_index is None else pattern[0]
