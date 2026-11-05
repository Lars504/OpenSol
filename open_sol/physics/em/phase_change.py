"""C5 phase-change medium (e.g. GST) of the in-plane EM types: a per-cell crystallinity X driven by the heat temperature
history (capped-Arrhenius JMAK crystallisation above an onset, melting above T_melt; whether a melt-quench stays
amorphous follows from the kinetics) and the mixed amorphous / crystalline permittivity it writes into the permittivity
arrays and sigma_e on its bounding box. Kinetics and mixing take the array module `xp` (numpy: CPU self-test; cupy: run
time); GST-225 defaults below."""
import math
from dataclasses import dataclass

import numpy as np
from ...core.backend import array_module_of, to_host

from ...core.constants import C0, DTYPE, ELEMENTARY_CHARGE, EPS0, K_B
from .solvers.fdtd_2d_base import _bounding_box

FEATURE_NAME = 'phase_change_medium'
BOLTZMANN_EV = K_B / ELEMENTARY_CHARGE   # eV/K (8.617333262e-5)
MIXING_RULES = ('lorentz_lorenz', 'linear')

# Ge2Sb2Te5 (GST-225) at 1.55 um: the phase_change_medium defaults and the gst_225_* library materials (materials/gst.py).
GST_225 = {
    'wavelength': 1.55e-6,
    'n_amorphous': 3.94, 'k_amorphous': 0.045,       # sputtered films, ellipsometry (GST_225_REFERENCES)
    'n_crystalline': 6.11, 'k_crystalline': 0.83,    # fcc phase
    'crystallization_temperature': 423.0,            # K: onset in slow anneals (~150 C)
    'fast_crystallization_temperature': 650.0,       # K: illustrative, see GST_225_REFERENCES
    'melting_temperature': 889.0,                    # K (616 C)
    'min_crystallization_time': 50e-9,               # s: illustrative
    'activation_energy': 2.24,                       # eV
    'avrami_exponent': 2.5,
}
GST_225_REFERENCES = (
    'GST-225 data: n, k at 1.55 um as used for sputtered films in integrated photonics (e.g. Rios et al., Nat. '
    'Photonics 9, 725 (2015); film dependent by ~10 %); crystallisation onset ~150 C and activation energy 2.24 eV from '
    'Friedrich et al., J. Appl. Phys. 87, 4130 (2000); melting point 616 C from Yamada et al., J. Appl. Phys. 69, 2849 '
    '(1991). The Avrami exponent 2.5 is typical of nucleation-dominated GST (2-4 reported); the cap of 50 ns above '
    '650 K is an illustrative calibration (100 s at the 150 C onset with 2.24 eV, tens of ns in laser / electrical '
    'SET pulses): fit both to the device.')


@dataclass(frozen=True)
class Kinetics:
    """Capped-Arrhenius JMAK kinetics: rate k(T) = exp(-Ea / kB (1/T - 1/T_fast)) / tau_min for onset <= T < T_fast,
    1 / tau_min for T_fast <= T < T_melt, 0 below the onset; X = 1 - exp(-theta^n), d theta = k(T) dt."""
    onset_temperature: float      # K
    fast_temperature: float       # K
    melting_temperature: float    # K
    min_time: float               # s
    activation_energy: float      # eV
    avrami_exponent: float

    def rate(self, temperature, xp=np):
        """k(T) [1/s] per cell (float64): 0 below the onset and at / above the melting temperature."""
        temperature = xp.maximum(xp.asarray(temperature, dtype=xp.float64), 1.0)
        exponent = xp.minimum(-self.activation_energy / BOLTZMANN_EV
                              * (1.0 / temperature - 1.0 / self.fast_temperature), 0.0)
        active = (temperature >= self.onset_temperature) & (temperature < self.melting_temperature)
        return xp.where(active, xp.exp(exponent) / self.min_time, 0.0)

    def advance(self, theta, temperature, elapsed, xp=np):
        """theta (float64, in place) after `elapsed` [s] at `temperature` (the end-of-interval value, as implicit-Euler
        heat): += k(T) elapsed; 0 where T >= T_melt (molten = amorphous; it recrystallises only while cooling slowly)."""
        temperature = xp.asarray(temperature, dtype=xp.float64)
        theta[...] = xp.where(temperature >= self.melting_temperature, 0.0,
                              theta + self.rate(temperature, xp) * float(elapsed))

    def crystallinity(self, theta, xp=np):
        """X = 1 - exp(-theta^n) (theta inf -> 1)."""
        return -xp.expm1(-(theta ** self.avrami_exponent))

    def theta_of(self, crystallinity):
        """Host theta of a crystallinity in [0, 1] (inf for 1)."""
        if crystallinity >= 1.0:
            return math.inf
        return (-math.log1p(-crystallinity)) ** (1.0 / self.avrami_exponent)


@dataclass(frozen=True)
class Optics:
    """Amorphous / crystalline complex index (n - j k) at `wavelength` and the mixing rule of the permittivity:
    'lorentz_lorenz' (Clausius-Mossotti: (eps - 1) / (eps + 2) linear in X) or 'linear' (eps linear in X)."""
    n_amorphous: float
    k_amorphous: float
    n_crystalline: float
    k_crystalline: float
    wavelength: float
    rule: str = 'lorentz_lorenz'

    def phase_permittivity(self, crystalline):
        """Complex relative permittivity (n - j k)^2 of one phase (Im <= 0 absorbs, exp(+j omega t))."""
        if crystalline:
            return complex(self.n_crystalline, -self.k_crystalline) ** 2
        return complex(self.n_amorphous, -self.k_amorphous) ** 2

    def permittivity(self, crystallinity, xp=np):
        """Complex permittivity per cell of crystallinity X (complex128)."""
        x = xp.asarray(crystallinity, dtype=xp.float64)
        eps_amorphous, eps_crystalline = self.phase_permittivity(False), self.phase_permittivity(True)
        if self.rule == 'linear':
            return (1.0 - x) * eps_amorphous + x * eps_crystalline
        polarizability = ((1.0 - x) * ((eps_amorphous - 1.0) / (eps_amorphous + 2.0))
                          + x * ((eps_crystalline - 1.0) / (eps_crystalline + 2.0)))
        return (1.0 + 2.0 * polarizability) / (1.0 - polarizability)

    def conductivity_factor(self):
        """sigma_e = factor * (-Im eps) [S/m]: omega eps0 at the reference wavelength."""
        return 2.0 * math.pi * C0 / self.wavelength * EPS0


class PhaseChangeMedium:
    """One phase_change_medium region on its cells' bounding box: theta (float64), crystallinity (float32), cropped mask;
    `update` advances them, rewrites eps / sigma_e there and records the mean crystallinity on the device."""

    def __init__(self, solver, mask, kinetics, optics, initial_crystallinity):
        box = _bounding_box(mask)
        if box is None:
            raise ValueError('phase-change medium: the selection covers no cell')
        self._solver = solver
        self.box = box
        self.mask = mask[box].copy()
        xp = self.xp = array_module_of(mask)   # the run backend of the solver's mask
        self.n_cells = int(xp.count_nonzero(self.mask))   # build-time host sync
        self._mean_weight = self.mask.astype(xp.float64) / self.n_cells
        self.kinetics = kinetics
        self.optics = optics
        self.theta = xp.full(self.mask.shape, kinetics.theta_of(float(initial_crystallinity)), dtype=xp.float64)
        self.crystallinity = xp.empty(self.mask.shape, dtype=DTYPE)
        self.time = 0.0   # heat time [s] of the last update
        self._times = []
        self._means = xp.zeros(64, dtype=xp.float64)
        self._refresh()

    def update(self, temperature, time):
        """Advance to heat time `time` [s] with the full-grid temperature [K] at its end; False (nothing done) for a
        repeated time (the end-of-run flush)."""
        elapsed = float(time) - self.time
        if elapsed < 0.0:
            raise ValueError(f'phase-change medium: heat time went back from {self.time:.6g} s to {time:.6g} s')
        if elapsed == 0.0:   # the constructor recorded the initial state at time 0
            return False
        self.kinetics.advance(self.theta, temperature[self.box], elapsed, self.xp)
        self.time = float(time)
        self._refresh()
        return True

    def _refresh(self):
        """crystallinity from theta, eps / sigma_e of the masked cells, one recorded mean."""
        xp = self.xp
        self.crystallinity[...] = self.kinetics.crystallinity(self.theta, xp)
        permittivity = self.optics.permittivity(self.crystallinity, xp)
        real_part = permittivity.real.astype(DTYPE)
        conductivity = (-permittivity.imag * self.optics.conductivity_factor()).astype(DTYPE)
        materials = self._solver.domain.materials
        for name in self._solver.permittivity_arrays:
            xp.copyto(getattr(materials, name)[self.box], real_part, where=self.mask)
        xp.copyto(materials.sigma_e[self.box], conductivity, where=self.mask)
        index = len(self._times)
        if index == self._means.size:
            self._means = xp.concatenate((self._means, xp.zeros_like(self._means)))
        self._means[index] = (self.crystallinity * self._mean_weight).sum()
        self._times.append(self.time)

    def paint_crystallinity(self, target):
        """Write the crystallinity into the full-grid float32 `target` on the medium's cells."""
        self.xp.copyto(target[self.box], self.crystallinity, where=self.mask)

    def result(self):
        """'timeseries' of the mean crystallinity over the selected cells at every update (t = heat time, t = 0 the
        initial state)."""
        return {'kind': 'timeseries', 't': np.asarray(self._times, dtype=np.float64),
                'values': to_host(self._means[:len(self._times)]), 'component': 'crystallinity', 'unit': '1'}


def add_phase_change_medium(solver, mask, kinetics, optics, initial_crystallinity):
    """Append a PhaseChangeMedium on the cupy bool (nx, ny) cell mask to solver.phase_change_media (EMPhysics or
    FDFDPhysics), write its initial permittivity and notify the solver; returns it."""
    medium = PhaseChangeMedium(solver, mask, kinetics, optics, initial_crystallinity)
    solver.phase_change_media.append(medium)
    solver.materials_changed(tuple(solver.permittivity_arrays) + ('sigma_e',))
    return medium


def update_phase_change(solver, temperature, time):
    """Advance every phase_change_media entry of `solver` to heat time `time` [s] with the full-grid (nx, ny) heat
    temperature [K], then one materials_changed of the arrays they rewrote (heat_em_phase_change transform)."""
    shape = solver.domain.materials.sigma_e.shape
    if tuple(temperature.shape) != tuple(shape):
        raise ValueError(f'phase-change medium: temperature of shape {temperature.shape}, expected the grid {shape}')
    changed = [medium.update(temperature, time) for medium in solver.phase_change_media]
    if any(changed):
        solver.materials_changed(tuple(solver.permittivity_arrays) + ('sigma_e',))
