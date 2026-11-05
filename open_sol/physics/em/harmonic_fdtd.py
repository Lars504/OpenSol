"""Harmonic FDTD (A4a): a window-sized running DFT of an EMPhysics at the frequency_domain study wavelengths, divided
by the shared source spectrum S(omega) into phasors of the FDFD convention, plus the window field energy that ends the pulse."""
# Normalisation: H(omega) = DFT(omega) / S(omega), S = results._shared_waveform_spectrum (the sampled waveform on the
# accumulator's DFT grid). Dividing by the spectrum of the real time signal removes the sin carrier, so H is the
# response to a unit real phasor at phase 0: the FDFD phasor of the same source up to numerical dispersion (FDTD
# omega_tilde vs FDFD omega, < 1 % at 25 cells per wavelength; harmonic.py header for the source sign).
import math

import numpy as np
from ...core.backend import cp
from ...core.kernels import axis_first_reduce, twin

from ...core.constants import C0, DTYPE, EPS0, MU0
from ...core.sources import GaussianPulse, RickerWavelet
from .harmonic import absorbed_power_ez, cell_permittivity
from .monitors import DFTMonitor, dft_sample_interval
from .results import _shared_waveform_spectrum, common_waveform

__all__ = ['HarmonicAccumulator']

_CONSTANTS = f'#define EPS0_SI {float(EPS0)!r}\n#define MU0_SI {float(MU0)!r}\n'


def _f64(values):
    return values.astype(np.float64)


# 0.5 sum (eps0 eps E^2 + mu0 mu H^2) w_x w_y over the window [J/m]; raw Yee samples (staggering ignored: a decay
# measure, not an energy balance). core.kernels twins: the numpy twins (0-d float64 arrays) follow the cupy sources.
def _field_energy_ez_host(eps, mu, ez, hx, hy, wx, wy):
    density = (float(EPS0) * _f64(eps) * _f64(ez) ** 2 + float(MU0) * _f64(mu) * (_f64(hx) ** 2 + _f64(hy) ** 2))
    return np.asarray(0.5 * np.sum(wx * wy * density))


def _field_energy_hz_host(eps_xx, eps_yy, mu, hz, ex, ey, wx, wy):
    density = (float(EPS0) * (_f64(eps_xx) * _f64(ex) ** 2 + _f64(eps_yy) * _f64(ey) ** 2)
               + float(MU0) * _f64(mu) * _f64(hz) ** 2)
    return np.asarray(0.5 * np.sum(wx * wy * density))


# F = the field dtype (float32, float64 for the double stepper); materials float32.
_FIELD_ENERGY_EZ = twin('em.harmonic_fdtd_energy_ez', cp.ReductionKernel(
    'float32 eps, float32 mu, F ez, F hx, F hy, float64 wx, float64 wy',
    'float64 energy',
    'wx * wy * (EPS0_SI * (double)eps * (double)ez * (double)ez'
    ' + MU0_SI * (double)mu * ((double)hx * (double)hx + (double)hy * (double)hy))',
    'a + b', 'energy = 0.5 * a', '0', 'harmonic_fdtd_energy_ez', preamble=_CONSTANTS), _field_energy_ez_host)
_FIELD_ENERGY_HZ = twin('em.harmonic_fdtd_energy_hz', cp.ReductionKernel(
    'float32 eps_xx, float32 eps_yy, float32 mu, F hz, F ex, F ey, float64 wx, float64 wy',
    'float64 energy',
    'wx * wy * (EPS0_SI * ((double)eps_xx * (double)ex * (double)ex + (double)eps_yy * (double)ey * (double)ey)'
    ' + MU0_SI * (double)mu * (double)hz * (double)hz)',
    'a + b', 'energy = 0.5 * a', '0', 'harmonic_fdtd_energy_hz', preamble=_CONSTANTS), _field_energy_hz_host)
_SPECTRA = {'ez': ('ez',), 'hz': ('hz', 'ex', 'ey')}   # DFTMonitor spectra per polarisation (E at the cell centres)


def _shared_pulse(physics):
    """The one gaussian / ricker waveform every source of `physics` shares; ValueError otherwise (cw never decays)."""
    label = f"physics '{getattr(physics, 'name', 'em')}'"
    sources = list(physics.sources)
    if not sources:
        raise ValueError(f'{label}: a frequency_domain study of an FDTD physics needs at least one enabled source')
    first = common_waveform(sources)
    if first is None:
        raise ValueError(f'{label}: every source of an FDTD physics in a frequency_domain study must share one '
                         'waveform (type, wavelength, bandwidth): the response is normalised by its spectrum')
    if not isinstance(first, (GaussianPulse, RickerWavelet)):
        raise ValueError(f'{label}: a frequency_domain study of an FDTD physics needs gaussian or ricker sources; '
                         f'{type(first).__name__} never decays, so the pulse cannot end')
    return first


class HarmonicAccumulator:
    """Running DFT (DFTMonitor) over the physical window of an EMPhysics at the study frequencies; normalize() turns it
    into unit-phasor responses once, select(index) picks the frequency the renderer shows."""

    def __init__(self, physics, frequencies):
        engine = physics.engine
        if engine.n_step != 0:
            raise ValueError(f'attach_harmonic needs an FDTD physics that has not stepped yet (n_step = '
                             f'{engine.n_step}): S(omega) is summed from the first step')
        self.polarization = physics.polarization
        self.source_end = float(_shared_pulse(physics).duration)   # [s] 2 t0 of the shared pulse
        window_x, window_y = physics.domain.physical_window()
        self.window = (window_x, window_y)
        config = physics.boundary_config
        dt = float(engine.dt)
        self._xp = xp = physics.domain.xp
        self.monitor = DFTMonitor(window_x, window_y, frequencies, dt, polarization=physics.polarization,
                                  periodic_x=config.periodic_x, periodic_y=config.periodic_y, xp=xp)
        self.monitor.sample_interval = dft_sample_interval(dt, physics.sources, [self.monitor.frequencies])
        self.frequencies = self.monitor.frequencies   # [Hz] float64, study order
        x_widths, y_widths = physics.domain.cell_widths()
        self._x_widths = xp.asarray(x_widths[window_x])[:, None]   # float64 (mx, 1)
        self._y_widths = xp.asarray(y_widths[window_y])[None, :]   # float64 (1, my)
        self.normalized = False
        self.index = 0

    def field_energy(self, physics):
        """Run-backend float64 0-d EM energy per unit z length [J/m] in the physical window (one reduction per field set,
        axis-first on large device windows: core.kernels.axis_first_reduce; under Bloch k != 0 the imaginary set
        fields.imag adds its share)."""
        window = self.window
        materials = physics.domain.materials
        energy = None
        field_sets = (physics.fields,) if physics.fields.imag is None else (physics.fields, physics.fields.imag)
        for fields in field_sets:
            if self.polarization == 'ez':
                set_energy = axis_first_reduce(_FIELD_ENERGY_EZ, materials.eps_r[window], materials.mu_r[window],
                                               fields.ez[window], fields.hx[window], fields.hy[window],
                                               self._x_widths, self._y_widths)
            else:
                set_energy = axis_first_reduce(_FIELD_ENERGY_HZ, materials.eps_xx[window], materials.eps_yy[window],
                                               materials.mu_r[window], fields.hz[window], fields.ex[window],
                                               fields.ey[window], self._x_widths, self._y_widths)
            energy = set_energy if energy is None else energy + set_energy
        return energy

    def normalize(self, physics):
        """Divide every spectrum by S(omega) of the shared waveform (once) -> warnings for frequencies where |S|^2 is
        below 1e-6 of its carrier value (outside the pulse spectrum: noise there)."""
        if self.normalized:
            return []
        shared = _shared_waveform_spectrum(physics, self.frequencies, self.monitor.sample_interval)
        if shared is None:
            _shared_pulse(physics)   # raises with the reason
            raise ValueError('harmonic FDTD: no shared source waveform to normalise by')
        spectrum, valid = shared
        inverse = np.zeros(spectrum.shape, dtype=np.complex128)
        np.divide(1.0, spectrum, out=inverse, where=spectrum != 0.0)
        scale = self._xp.asarray(inverse.astype(np.complex64))[:, None, None]
        for component in _SPECTRA[self.polarization]:
            spectra = getattr(self.monitor, f'{component}_dft')
            spectra *= scale   # in place: complex64 (n_freqs, mx, my)
        self.normalized = True
        if bool(np.all(valid)):
            return []
        outside = C0 / self.frequencies[~valid]
        return [f"physics '{getattr(physics, 'name', 'em')}': {outside.size} study wavelength(s) between "
                f'{outside.min():.4g} and {outside.max():.4g} m lie outside the source pulse spectrum (|S|^2 below '
                '1e-6 of its carrier value), so their harmonic fields are noise; move the source wavelength / '
                'bandwidth over the study range']

    def select(self, index):
        """Show study frequency `index` (0-based, study order)."""
        index = int(index)
        if not 0 <= index < self.frequencies.size:
            raise IndexError(f'harmonic frequency index {index} out of range for {self.frequencies.size} wavelengths')
        self.index = index

    @property
    def omega(self) -> float:
        """Angular frequency [rad/s] of the selected study wavelength."""
        return 2.0 * math.pi * float(self.frequencies[self.index])

    def phasor(self, component):
        """complex64 (mx, my) device view of the normalised phasor 'ez' | 'hz' | 'ex' | 'ey' (E at the cell centres)
        at the selected frequency; ValueError before normalize()."""
        if not self.normalized:
            raise ValueError('harmonic FDTD fields exist only after the pulse of a frequency_domain study')
        return getattr(self.monitor, f'{component}_dft')[self.index]

    def e_intensity(self):
        """float32 |E|^2 of the phasors at the cell centres: |E_z|^2 (ez), |E_x|^2 + |E_y|^2 (hz)."""
        xp = self._xp
        if self.polarization == 'ez':
            return (xp.abs(self.phasor('ez')) ** 2).astype(DTYPE, copy=False)
        intensity = xp.abs(self.phasor('ex')) ** 2
        intensity += xp.abs(self.phasor('ey')) ** 2
        return intensity.astype(DTYPE, copy=False)

    def e_abs(self):
        """float32 |E| of the phasors at the cell centres (sqrt of e_intensity)."""
        return self._xp.sqrt(self.e_intensity())

    def absorbed_power(self, physics):
        """float32 (mx, my) [W/m^3] 1/2 omega eps0 (-Im eps(omega)) |E|^2 at the selected frequency (continuous E7 chi
        of the poles, sigma_e); hz per component with the centre-collocated E phasor and the cell eps_xx / eps_yy."""
        omega = self.omega
        materials = physics.domain.materials
        regions = physics.dispersive_regions
        if self.polarization == 'ez':
            cell_eps = cell_permittivity(materials, 'eps_r', regions, omega, xp=self._xp)[self.window]
            return absorbed_power_ez(omega, cell_eps, self.phasor('ez'))
        power_density = None
        for component, array_name in (('ex', 'eps_xx'), ('ey', 'eps_yy')):
            modulus = self._xp.abs(self.phasor(component)).astype(np.float64)
            term = modulus * modulus * cell_permittivity(materials, array_name, regions, omega,
                                                         xp=self._xp)[self.window].imag
            power_density = term if power_density is None else power_density + term
        power_density *= -0.5 * omega * EPS0
        return power_density.astype(DTYPE)
