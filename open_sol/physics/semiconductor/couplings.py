"""Transforms of the semiconductor <-> EM couplings (full grid, on the run backend): photogeneration G = eta q_abs /
(hbar omega) from the absorbed optical power, and the free-carrier plasma-dispersion index change (E13 'index')."""
import math

import numpy as np

from ...core.backend import cp
from ...core.kernels import twin

from ...core.constants import C0, DTYPE, HBAR

# Soref-Bennett fit of crystalline silicon at 1.55 um in SI: dn = -(a N + b P^c), N, P in 1/m^3 (the cm^-3 fit
# 8.8e-22 N + 8.5e-18 P^0.8 with N, P converted: a = 8.8e-22 * 1e-6, b = 8.5e-18 * (1e-6)^0.8).
PLASMA_ELECTRON_COEFFICIENT = 8.8e-28      # m^3
PLASMA_HOLE_COEFFICIENT = 8.5e-18 * 1e-6 ** 0.8   # m^2.4 (~1.347e-22)
PLASMA_HOLE_EXPONENT = 0.8

def _plasma_index_change_host(electrons, holes, electron_coefficient, hole_coefficient, hole_exponent, index_change):
    """float32 numpy twin of _plasma_index_change (writes index_change in place)."""
    zero = DTYPE(0.0)
    index_change[...] = -(electron_coefficient * np.maximum(electrons, zero)
                          + hole_coefficient * np.power(np.maximum(holes, zero), hole_exponent))
    return index_change


_plasma_index_change = twin('semiconductor.plasma_index_change', cp.ElementwiseKernel(
    'float32 electrons, float32 holes, float32 electron_coefficient, float32 hole_coefficient, float32 hole_exponent',
    'float32 index_change',
    'index_change = -(electron_coefficient * fmaxf(electrons, 0.0f)'
    ' + hole_coefficient * powf(fmaxf(holes, 0.0f), hole_exponent));',
    'semiconductor_plasma_index_change'), _plasma_index_change_host)


def make_generation_transform(photon_wavelength, quantum_efficiency, grid_shape, xp):
    """CouplingTerm transform (q_accum, em, semiconductor) -> G [1/(m^3 s)] = eta * take_absorbed_power() / (hbar
    omega) in a persistent float32 buffer of array module xp (the run backend's); omega from photon_wavelength [m],
    else the EM solver's omega (FDFD)."""
    buffer = xp.zeros(grid_shape, dtype=DTYPE)

    def transform(q_accum, em, semiconductor):
        absorbed = em.take_absorbed_power()
        if absorbed is None:
            return None
        if photon_wavelength is not None:
            omega = 2.0 * math.pi * C0 / photon_wavelength
        else:
            omega = getattr(em, 'omega', None)
            if omega is None or not (math.isfinite(omega) and omega > 0.0):
                raise ValueError(f"EM physics '{getattr(em, 'name', '?')}' has no single solve frequency: set the "
                                 'coupling photon_wavelength')
        xp.multiply(absorbed, DTYPE(quantum_efficiency / (HBAR * omega)), out=buffer)
        return buffer
    return transform


def make_plasma_transform(electron_coefficient, hole_coefficient, hole_exponent, grid_shape, xp):
    """CouplingTerm transform (N, semiconductor, em) -> dn = -(a N + b P^c) (P read from the source state) in a
    persistent float32 buffer of array module xp: the 'index' contribution of every permittivity component
    (isotropic), recomputed only when the semiconductor's n_solves changed (the per-slot terms of one transfer share it)."""
    buffer = xp.zeros(grid_shape, dtype=DTYPE)
    coefficients = (DTYPE(electron_coefficient), DTYPE(hole_coefficient), DTYPE(hole_exponent))
    computed_revision = [None]   # n_solves of the dn in buffer

    def transform(electrons, semiconductor, em):
        revision = getattr(semiconductor, 'n_solves', None)
        if revision is None or revision != computed_revision[0]:
            _plasma_index_change(electrons, semiconductor.state()['P'], *coefficients, buffer)
            computed_revision[0] = revision
        return buffer
    return transform
