"""Kottke / Farjadpour averaging (igor 1.5) of one 2D EM physics' current eps on the build's interface locations
(core.interface_geometry): 1 / eps_d = n_d^2 <1/eps> + (1 - n_d^2) / <eps> over the two materials of the location
(E_z: <eps>), reapplied in every E-coefficient rebuild (FDTD) / assembly (FDFD) / mode solve (em_modes) and given to
the mode-port lines."""
import numpy as np
from ...core.backend import array_module_of, cp, to_host
from ...core.kernels import twin

from ...core.constants import DTYPE, EPS0

# Field component -> (location kind, permittivity array) per polarisation (ez E_z at the cells; hz E_x on the hi-y and
# E_y on the hi-x faces; 'modes' = the em_modes cross-section: E_x on the hi-x, E_y on the hi-y faces, E_z (eps_r) at
# the cells).
_KINDS = {'ez': (('cells', 'eps_r'),), 'hz': (('faces_y', 'eps_xx'), ('faces_x', 'eps_yy')),
          'modes': (('faces_x', 'eps_xx'), ('faces_y', 'eps_yy'), ('cells', 'eps_r'))}
# Kinds whose component is the other in-plane one than the geometry's stored n_d^2 (the hz layout): 1 - n_d^2 there.
_COMPLEMENT_NORMAL = {'modes': frozenset({'faces_x', 'faces_y'})}

# core.kernels twins (FDTD, FDFD and em_modes on both backends): the numpy twins follow the cupy sources ('raw' arrays
# are flat views indexed by the location arrays).
def _smoothed_host(eps, cell_a, cell_b, fill, stored_normal_square, complement, smoothed):
    eps_a, eps_b = eps[cell_a], eps[cell_b]
    normal_square = np.float32(1.0) - stored_normal_square if complement else stored_normal_square
    mean = fill * eps_a + (np.float32(1.0) - fill) * eps_b
    inverse_mean = fill / eps_a + (np.float32(1.0) - fill) / eps_b
    smoothed[...] = np.float32(1.0) / (normal_square * inverse_mean + (np.float32(1.0) - normal_square) / mean)


def _face_mean_host(values, location, upper_cell, lower_share, upper_share, mean):
    mean[...] = values[location] * lower_share + values[upper_cell] * upper_share


def _scatter_coefficients_host(location, ca_value, cb_value, ca, cb):
    ca[location] = ca_value
    cb[location] = cb_value


def _scatter_scaled_host(location, value, scale, target):
    target[location] = value * scale


_SMOOTHED = twin('em.subpixel_smoothed', cp.ElementwiseKernel(
    'raw float32 eps, int32 cell_a, int32 cell_b, float32 fill, float32 stored_normal_square, int32 complement',
    'float32 smoothed',
    '''
    float eps_a = eps[cell_a];
    float eps_b = eps[cell_b];
    float normal_square = complement ? 1.0f - stored_normal_square : stored_normal_square;
    float mean = fill * eps_a + (1.0f - fill) * eps_b;
    float inverse_mean = fill / eps_a + (1.0f - fill) / eps_b;
    smoothed = 1.0f / (normal_square * inverse_mean + (1.0f - normal_square) / mean);
    ''',
    'subpixel_smoothed'), _smoothed_host)

_FACE_MEAN = twin('em.subpixel_face_mean', cp.ElementwiseKernel(
    'raw float32 values, int64 location, int32 upper_cell, float32 lower_share, float32 upper_share',
    'float32 mean',
    'mean = values[location] * lower_share + values[upper_cell] * upper_share',
    'subpixel_face_mean'), _face_mean_host)

# Targets in the engine's field dtype T (float32, float64 for the double 2D stepper); the averaged values are float32.
_SCATTER_COEFFICIENTS = twin('em.subpixel_scatter_coefficients', cp.ElementwiseKernel(
    'int64 location, float32 ca_value, float32 cb_value',
    'raw T ca, raw T cb',
    'ca[location] = ca_value; cb[location] = cb_value;',
    'subpixel_scatter_coefficients'), _scatter_coefficients_host)

_SCATTER_SCALED = twin('em.subpixel_scatter_scaled', cp.ElementwiseKernel(
    'int64 location, float32 value, T scale',
    'raw T target',
    'target[location] = (T)value * scale',
    'subpixel_scatter_scaled'), _scatter_scaled_host)


class SubpixelSmoothing:
    """Averaging of one physics ('ez' | 'hz' | 'modes') on an InterfaceGeometry: preallocated float32 buffers per
    location kind it uses (smoothed eps, face-mean eps / sigma, ca, cb), so a rebuild allocates nothing."""

    def __init__(self, geometry, polarization):
        if polarization not in _KINDS:
            raise ValueError(f"subpixel smoothing needs polarization {', '.join(map(repr, _KINDS))}, got "
                             f"{polarization!r}")
        self.geometry = geometry
        self.polarization = polarization
        self._array_kind = {array: kind for kind, array in _KINDS[polarization]}
        self._complement = _COMPLEMENT_NORMAL.get(polarization, frozenset())
        # buffers on the backend of the geometry's location arrays (model.subpixel_geometry builds them on the run's)
        self._buffers = {kind: tuple(array_module_of(geometry.kind(kind).location).empty(geometry.kind(kind).size,
                                                                                         dtype=DTYPE)
                                     for _ in range(4))
                         for kind, _ in _KINDS[polarization]}
        self._host_location_cache = {}   # kind -> host int64 locations (mode-port lines, em_modes point deltas)

    def smoothed(self, kind, eps):
        """float32 buffer of the averaged relative eps at the kind's locations from the cell array `eps` (overwritten by
        the next call of that kind)."""
        locations = self.geometry.kind(kind)
        out = self._buffers[kind][0]
        if out.size:
            _SMOOTHED(eps.ravel(), locations.cell_a, locations.cell_b, locations.fill, locations.normal_square,
                      np.int32(kind in self._complement), out)
        return out

    def face_mean(self, kind, values, out_index=1):
        """float32 buffer of the standard value at the kind's locations: the width-weighted face mean of `values`
        (yee_grid.face_average), the cell value for 'cells'."""
        locations = self.geometry.kind(kind)
        out = self._buffers[kind][out_index]
        if out.size:
            _FACE_MEAN(values.ravel(), locations.location, locations.upper_cell, locations.lower_share,
                       locations.upper_share, out)
        return out

    def correct_fdtd(self, engine):
        """After the engine's fused coefficient rebuild (before its PEC fold): ca / cb (and hz eps_x / eps_y [F/m]) at
        the interface locations from the averaged eps and the width-weighted face mean of sigma_e (the shared
        e_coefficients formula)."""
        from .solvers.fdtd_ez_2d_kernels import e_coefficients
        materials = engine.materials
        if self.polarization == 'ez':
            targets = (('cells', engine.ca, engine.cb, None),)
        else:
            targets = (('faces_y', engine.ca_x, engine.cb_x, engine.eps_x),
                       ('faces_x', engine.ca_y, engine.cb_y, engine.eps_y))
        for kind, ca, cb, eps_absolute in targets:
            locations = self.geometry.kind(kind)
            if not locations.size:
                continue
            eps = self.smoothed(kind, getattr(materials, _array_of(self.polarization, kind)))
            sigma = self.face_mean(kind, materials.sigma_e)
            _, _, ca_values, cb_values = self._buffers[kind]
            e_coefficients(sigma, eps, DTYPE(engine.dt), DTYPE(EPS0), ca_values, cb_values)
            _SCATTER_COEFFICIENTS(locations.location, ca_values, cb_values, ca.ravel(), cb.ravel())
            if eps_absolute is not None:
                _SCATTER_SCALED(locations.location, eps, eps_absolute.dtype.type(EPS0), eps_absolute.ravel())

    def add_deltas(self, kind, target, materials):
        """target (complex128 full grid: FDFD cell or face eps(omega)) += averaged - standard relative eps at the
        kind's locations (the background eps only; sigma_e, poles and extra permittivity keep their means)."""
        locations = self.geometry.kind(kind)
        if not locations.size:
            return
        flat = target.ravel()   # a view of the C-contiguous assembly array
        flat[locations.location] += self.delta(kind, materials)

    def delta(self, kind, materials):
        """float32 buffer of averaged - standard eps at the kind's locations of the current materials."""
        eps_cells = getattr(materials, _array_of(self.polarization, kind))
        smoothed = self.smoothed(kind, eps_cells)
        standard = self.face_mean(kind, eps_cells, out_index=2)
        standard -= smoothed
        standard *= DTYPE(-1.0)
        return standard

    def line_delta(self, array_name, materials, axis, line_cell):
        """Host float64 (n_along,) averaged - standard eps of `array_name`'s kind along the grid line through
        line_cell (axis 'x': the row j = line_cell, 'y': the column i = line_cell), 0 off the interfaces; one
        device-to-host transfer (mode_ports.PortLine)."""
        kind = self._array_kind[array_name]
        locations = self.geometry.kind(kind)
        ny = int(materials.shape[1])
        n_along = int(materials.shape[0] if axis == 'x' else ny)
        row = np.zeros(n_along, dtype=np.float64)
        if not locations.size:
            return row
        location = self._host_locations(kind)
        values = to_host(self.delta(kind, materials)).astype(np.float64)
        i_index, j_index = np.divmod(location, ny)
        on_line = (j_index if axis == 'x' else i_index) == line_cell
        row[(i_index if axis == 'x' else j_index)[on_line]] = values[on_line]
        return row

    def point_deltas(self, materials):
        """em_modes ('modes'): host float64 (nx, ny) averaged - standard eps at the E_x hi-x faces, E_y hi-y faces and
        E_z cells (0 off the interfaces): cross_section_modes.CrossSectionMaterials eps_point_deltas; one transfer per
        kind."""
        if self.polarization != 'modes':
            raise ValueError(f"point_deltas belong to the em_modes averaging ('modes'), not {self.polarization!r}")
        shape = tuple(int(count) for count in materials.shape[:2])
        deltas = []
        for kind in ('faces_x', 'faces_y', 'cells'):
            values = np.zeros(shape, dtype=np.float64)
            if self.geometry.kind(kind).size:
                values.ravel()[self._host_locations(kind)] = to_host(self.delta(kind, materials))
            deltas.append(values)
        return tuple(deltas)

    def _host_locations(self, kind):
        """Host copy of a kind's locations (static: cached at first use)."""
        if kind not in self._host_location_cache:
            self._host_location_cache[kind] = np.array(to_host(self.geometry.kind(kind).location))
        return self._host_location_cache[kind]


def _array_of(polarization, kind):
    """Permittivity array the kind averages for this polarisation."""
    return dict(_KINDS[polarization])[kind]
