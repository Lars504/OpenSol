"""Harmonic FDTD (A4a) of em_fdtd_3d: running DFT of E_x, E_y, E_z over the physical window at the frequency_domain
study wavelengths (complex64, 24 B per window cell and wavelength), normalised by the shared source spectrum."""
import math

import numpy as np
from ...core.backend import array_module_of, cp

from ...core.constants import C0, DTYPE, EPS0, MU0
from ...core.kernels import axis_first_reduce, twin
from ..em.harmonic_fdtd import _shared_pulse
from ..em.yee_grid import lower_face_partner
from ..em.monitors import (_IMAG_KERNELS, _REAL_KERNELS, _check_frequencies, _dft_phase, _field_sets,
                           dft_sample_interval)
from ..em.results import _shared_waveform_spectrum

_CONSTANTS = f'#define EPS0_SI {float(EPS0)!r}\n#define MU0_SI {float(MU0)!r}\n'


# 0.5 sum (eps0 eps E^2 + mu0 mu H^2) dV over the window [J]; raw Yee samples (a decay measure, not a balance).
def _field_energy_3d_host(eps, mu, ex, ey, ez, hx, hy, hz, wx, wy, wz):
    def squares(*components):
        return sum(component.astype(np.float64) ** 2 for component in components)
    density = (float(EPS0) * eps.astype(np.float64) * squares(ex, ey, ez)
               + float(MU0) * mu.astype(np.float64) * squares(hx, hy, hz))
    return np.asarray(0.5 * np.sum(wx * wy * wz * density))


_FIELD_ENERGY_3D = twin('em3d.harmonic_energy', cp.ReductionKernel(
    'float32 eps, float32 mu, float32 ex, float32 ey, float32 ez, float32 hx, float32 hy, float32 hz, '
    'float64 wx, float64 wy, float64 wz',
    'float64 energy',
    'wx * wy * wz * (EPS0_SI * (double)eps * ((double)ex * ex + (double)ey * ey + (double)ez * ez)'
    ' + MU0_SI * (double)mu * ((double)hx * hx + (double)hy * hy + (double)hz * hz))',
    'a + b', 'energy = 0.5 * a', '0', 'em3d_harmonic_energy', preamble=_CONSTANTS), _field_energy_3d_host)
COMPONENTS = ('ex', 'ey', 'ez')


class HarmonicAccumulator3D:
    """Window DFT of the three E components at their native faces; phasor(c) averages them to the cell centres."""

    def __init__(self, physics, frequencies):
        engine = physics.engine
        if engine.n_step != 0:
            raise ValueError(f'attach_harmonic needs an FDTD physics that has not stepped yet (n_step = '
                             f'{engine.n_step}): S(omega) is summed from the first step')
        self.source_end = float(_shared_pulse(physics).duration)
        self.window = physics.domain.physical_window()
        self.frequencies = _check_frequencies(frequencies)
        self.dt = float(engine.dt)
        self.sample_interval = dft_sample_interval(self.dt, physics.sources, [self.frequencies])
        self.xp = xp = physics.domain.xp
        # Per component its accumulated grid box: the window plus the lo face row of window cell 0 along the
        # component's own axis when the window starts inside the grid (padding / mirror ghost: the same lo face the
        # renderer and DFTPlane3D read); a periodic axis spanning the grid wraps to its last face x e^{+j k L} instead.
        self._boxes, self._lo_rows, self._wrap_factors = {}, {}, {}
        for axis, name in enumerate(COMPONENTS):
            cells = self.window[axis]
            box = list(self.window)
            if cells.start > 0:
                box[axis] = slice(cells.start - 1, cells.stop)
            elif engine.periodic[axis] and cells.stop == engine.shape[axis]:
                self._wrap_factors[name] = complex(np.exp(1j * float(engine.bloch_angles[axis])))
            self._boxes[name] = tuple(box)
            self._lo_rows[name] = int(cells.start > 0)
        self.spectra = {name: xp.zeros((self.frequencies.size,) + tuple(cells.stop - cells.start
                                                                        for cells in self._boxes[name]),
                                       dtype=np.complex64) for name in COMPONENTS}
        self._omega_dev = xp.asarray(2.0 * np.pi * self.frequencies)
        self._phases = xp.empty(self.frequencies.size, dtype=np.complex64)
        widths = list(physics.domain.cell_widths())
        if engine.polar:   # cylindrical sector: cell volumes r_c dr dphi dz (the r_c factor on the x widths)
            widths[0] = widths[0] * engine.radii.centre
        self._widths = tuple(xp.asarray(axis_widths[cells]).reshape(shape_3d)
                             for axis_widths, cells, shape_3d in zip(widths, self.window,
                                                                     ((-1, 1, 1), (1, -1, 1), (1, 1, -1))))
        self.normalized = False
        self.index = 0

    def update(self, fields, t):
        _dft_phase(self._omega_dev, t, self.dt * self.sample_interval, self._phases)
        phases = self._phases[:, None, None, None]
        for field_set, (accumulate, _) in _field_sets(fields, _REAL_KERNELS, _IMAG_KERNELS):   # Bloch: + j F_im
            for name in COMPONENTS:
                accumulate(getattr(field_set, name)[self._boxes[name]][None], phases, self.spectra[name])

    def field_energy(self, physics):
        """Device float64 0-d EM energy [J] in the physical window (Bloch k != 0: of both field sets, |F|^2; axis-first:
        core.kernels.axis_first_reduce)."""
        materials, window = physics.domain.materials, self.window
        energy = None
        for field_set in (physics.fields, physics.fields.imag):
            if field_set is None:
                continue
            term = axis_first_reduce(_FIELD_ENERGY_3D, materials.eps_r[window], materials.mu_r[window],
                                     *(getattr(field_set, name)[window] for name in ('ex', 'ey', 'ez', 'hx', 'hy', 'hz')),
                                     *self._widths)
            energy = term if energy is None else energy + term
        return energy

    def normalize(self, physics):
        """Divide by S(omega) of the shared waveform once -> warnings for wavelengths outside the pulse spectrum."""
        if self.normalized:
            return []
        shared = _shared_waveform_spectrum(physics, self.frequencies, self.sample_interval)
        if shared is None:
            _shared_pulse(physics)
            raise ValueError('harmonic FDTD: no shared source waveform to normalise by')
        spectrum, valid = shared
        inverse = np.zeros(spectrum.shape, dtype=np.complex128)
        np.divide(1.0, spectrum, out=inverse, where=spectrum != 0.0)
        scale = self.xp.asarray(inverse.astype(np.complex64))[:, None, None, None]
        for spectra in self.spectra.values():
            spectra *= scale
        self.normalized = True
        if bool(np.all(valid)):
            return []
        outside = C0 / self.frequencies[~valid]
        return [f"physics '{physics.name}': {outside.size} study wavelength(s) between {outside.min():.4g} and "
                f'{outside.max():.4g} m lie outside the source pulse spectrum, so their harmonic fields are noise']

    def select(self, index):
        index = int(index)
        if not 0 <= index < self.frequencies.size:
            raise IndexError(f'harmonic frequency index {index} out of range for {self.frequencies.size} wavelengths')
        self.index = index

    @property
    def omega(self) -> float:
        return 2.0 * math.pi * float(self.frequencies[self.index])

    def phasor(self, component, cut=None):
        """complex64 normalised phasor of 'ex' / 'ey' / 'ez' at the cell centres (mean of the cell's two faces;
        window cell 0 reads its lo face from the grid, the periodic wrap, or else its own face), whole window or the
        window plane cut = (axis 0 | 1 | 2, window index)."""
        if not self.normalized:
            raise ValueError('harmonic FDTD fields exist only after the pulse of a frequency_domain study')
        spectrum = self.spectra[component][self.index]
        axis = COMPONENTS.index(component)
        lo_row = self._lo_rows[component]
        wrap = self._wrap_factors.get(component)
        if cut is not None:
            cut_axis, index = cut
            if cut_axis == axis:   # the cut crosses the component's own faces: mean of the two
                own = _plane(spectrum, axis, index + lo_row)
                if index + lo_row > 0:
                    return 0.5 * (own + _plane(spectrum, axis, index + lo_row - 1))
                return 0.5 * (own + (own if wrap is None else wrap * _plane(spectrum, axis, -1)))
            spectrum = _plane(spectrum, cut_axis, index)   # other axes: the box is the window
            axis -= int(axis > cut_axis)
        upper = [slice(None)] * spectrum.ndim
        upper[axis] = slice(lo_row, None)
        if lo_row:
            lower = [slice(None)] * spectrum.ndim
            lower[axis] = slice(0, -1)
            return 0.5 * (spectrum[tuple(upper)] + spectrum[tuple(lower)])
        return 0.5 * (spectrum + lower_face_partner(spectrum, axis, wrap is not None, array_module_of(spectrum),
                                                    1.0 if wrap is None else wrap))

    def e_intensity(self, cut=None):
        """float32 |E|^2 of the centre phasors (whole window or one plane, as phasor)."""
        intensity = None
        for component in COMPONENTS:
            term = self.xp.abs(self.phasor(component, cut)) ** 2
            intensity = term if intensity is None else intensity + term
        return intensity.astype(DTYPE, copy=False)

    def absorbed_power(self, physics, cut=None):
        """float32 [W/m^3] time-averaged 1/2 (sigma_e + omega eps0 (-Im chi)) |E|^2 at the selected frequency: the
        engine's (re-quantised) sigma_e table, the continuous chi of the poles, the centre phasors; 0 on PEC cells."""
        return self._loss_conductivity(physics, cut) * DTYPE(0.5) * self.e_intensity(cut)

    def _loss_conductivity(self, physics, cut):
        """float32 sigma_e - omega eps0 Im chi [S/m] of the window cells (or one window plane, as phasor)."""
        engine = physics.engine
        selection = list(self.window)
        if cut is not None:
            axis, index = cut
            selection[axis] = self.window[axis].start + int(index)   # an integer drops the axis
        selection = tuple(selection)
        ids = engine.material_id[selection]
        pec = engine.table_eps_z[ids] < 0
        loss = self.xp.where(pec, DTYPE(0.0), engine.table_sigma[ids]).astype(DTYPE)
        omega = self.omega
        grid_shape = engine.material_id.shape
        for pole, box, mask in physics.dispersive_regions:
            weight = -omega * EPS0 * complex(pole.susceptibility(omega)).imag
            if weight == 0.0:
                continue
            target, source = [], []
            for axis, wanted in enumerate(selection):
                box_start = box[axis].start or 0
                box_stop = grid_shape[axis] if box[axis].stop is None else box[axis].stop
                if isinstance(wanted, int):
                    if not box_start <= wanted < box_stop:
                        break
                    source.append(wanted - box_start)
                    continue
                lower, upper = max(wanted.start, box_start), min(wanted.stop, box_stop)
                if lower >= upper:
                    break
                target.append(slice(lower - wanted.start, upper - wanted.start))
                source.append(slice(lower - box_start, upper - box_start))
            else:
                loss[tuple(target)] += DTYPE(weight) * mask[tuple(source)]
        loss[pec] = DTYPE(0.0)   # pole regions overlapping a pec_domain: no loss inside the metal
        return loss


def _plane(array, axis, index):
    """View of one plane of a 3D array: index along axis (the axis dropped)."""
    return array[(slice(None),) * axis + (int(index),)]
