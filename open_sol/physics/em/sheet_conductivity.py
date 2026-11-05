"""2D-material sheet conductivity sigma_s(omega) [S] of the FDFD conducting_sheet (host numpy, scipy inside a function):
'constant' or 'graphene_kubo' (Kubo intraband Drude + interband with temperature smearing), and the ConductingSheet
placement the FDFD assembly reads."""
# Convention exp(+j omega t), as every FDFD quantity: Im sigma < 0 is inductive (Drude 1 / (1 + j omega tau)), Im > 0
# capacitive, Re > 0 loss. The Kubo literature (Falkovsky 2008, Hanson 2008) writes exp(-i omega t): its sigma is the
# complex conjugate of the one here. Interband part (G(xi) = sinh(xi / kT) / (cosh(mu / kT) + cosh(xi / kT)), a =
# hbar omega): sigma_inter = sigma_0 [G(a / 2) - j (4 a / pi) PV int_0^inf (G(xi) - G(a / 2)) / (a^2 - 4 xi^2) dxi],
# sigma_0 = e^2 / (4 hbar); at T -> 0 it is sigma_0 [step(a - 2 |mu|) - (j / pi) ln|(a - 2 |mu|) / (a + 2 |mu|)|].
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ...core.constants import ELEMENTARY_CHARGE, EPS0, HBAR, K_B

UNIVERSAL_SHEET_CONDUCTIVITY = ELEMENTARY_CHARGE ** 2 / (4.0 * HBAR)   # S: sigma_0 = e^2 / (4 hbar) = 6.0853e-5
SHEET_MODELS = ('constant', 'graphene_kubo')
_INTEGRAL_TAIL_KT = 60.0   # interband integral cut at max(a, 2 |mu|) / kT + this; the rest of G - 1 is < e^-60


def _fermi_window(x, m):
    """G = n_F(-xi) - n_F(xi) in units of kT (x = xi / kT, m = mu / kT), overflow-free: expit(x + m) - expit(m - x)."""
    return 0.5 * (np.tanh(0.5 * (x + m)) - np.tanh(0.5 * (m - x)))


def graphene_intraband(omega, fermi_energy, scattering_time, temperature):
    """Intraband (Drude) Kubo sheet conductivity [S] at omega [rad/s], E_F [eV], tau [s], T [K]: e^2 tau (mu + 2 kT
    ln(1 + e^{-mu / kT})) / (pi hbar^2 (1 + j omega tau)) (-> e^2 |E_F| tau / (pi hbar^2 (1 + j omega tau)) at T -> 0)."""
    thermal = K_B * float(temperature)
    mu = float(fermi_energy) * ELEMENTARY_CHARGE
    drude_energy = mu + 2.0 * thermal * float(np.logaddexp(0.0, -mu / thermal))   # J, = |mu| at T -> 0
    tau = float(scattering_time)
    return ELEMENTARY_CHARGE ** 2 * drude_energy * tau / (math.pi * HBAR ** 2 * (1.0 + 1j * float(omega) * tau))


def graphene_interband(omega, fermi_energy, temperature):
    """Interband Kubo sheet conductivity [S] at T > 0 (module header formula; the principal value by scipy quad in units
    of kT with breakpoints at a / 2 and |mu|, the tail beyond the cut analytic)."""
    from scipy.integrate import quad
    thermal_ev = K_B * float(temperature) / ELEMENTARY_CHARGE
    photon = HBAR * float(omega) / ELEMENTARY_CHARGE / thermal_ev   # a / kT
    chemical = abs(float(fermi_energy)) / thermal_ev                  # |mu| / kT
    window_at_half = float(_fermi_window(0.5 * photon, chemical))
    cut = max(photon, 2.0 * chemical) + _INTEGRAL_TAIL_KT

    def integrand(x):
        return (float(_fermi_window(x, chemical)) - window_at_half) / (photon * photon - 4.0 * x * x)
    breakpoints = sorted(point for point in {0.5 * photon, chemical} if 0.0 < point < cut)
    principal, _ = quad(integrand, 0.0, cut, points=breakpoints or None, limit=500, epsabs=0.0, epsrel=1e-10)
    # beyond the cut G = 1: int_cut^inf dx / (a^2 - 4 x^2) = -ln((2 cut + a) / (2 cut - a)) / (4 a)
    principal -= (1.0 - window_at_half) * math.log((2.0 * cut + photon) / (2.0 * cut - photon)) / (4.0 * photon)
    return UNIVERSAL_SHEET_CONDUCTIVITY * (window_at_half - 1j * (4.0 * photon / math.pi) * principal)


@lru_cache(maxsize=256)
def graphene_kubo(omega, fermi_energy, scattering_time, temperature):
    """Graphene sheet conductivity [S] = graphene_intraband + graphene_interband (exp(+j omega t); cached per argument
    set: validation and every assembly at one omega ask for the same value)."""
    return (graphene_intraband(omega, fermi_energy, scattering_time, temperature)
            + graphene_interband(omega, fermi_energy, temperature))


@dataclass(frozen=True)
class SheetConductivity:
    """sigma_s model of a conducting sheet, called with omega [rad/s] -> complex [S]: 'constant' (`value` at every
    omega) or 'graphene_kubo' (fermi_energy [eV], scattering_time [s], temperature [K])."""
    model: str
    value: complex = 0j
    fermi_energy: float = 0.0
    scattering_time: float = 1e-13
    temperature: float = 300.0

    def __post_init__(self):
        if self.model not in SHEET_MODELS:
            raise ValueError(f"sheet conductivity model must be one of {', '.join(SHEET_MODELS)}, got {self.model!r}")
        if self.model == 'constant' and not (math.isfinite(complex(self.value).real)
                                             and math.isfinite(complex(self.value).imag)):
            raise ValueError(f"constant sheet conductivity must be finite [S], got {self.value!r}")
        if self.model == 'graphene_kubo':
            for name in ('scattering_time', 'temperature'):
                if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0.0):
                    raise ValueError(f"graphene_kubo {name} must be finite and > 0, got {getattr(self, name)!r}")
            if not math.isfinite(self.fermi_energy):
                raise ValueError(f"graphene_kubo fermi_energy must be finite [eV], got {self.fermi_energy!r}")

    def __call__(self, omega):
        if self.model == 'constant':
            return complex(self.value)
        return graphene_kubo(float(omega), float(self.fermi_energy), float(self.scattering_time),
                             float(self.temperature))


class ConductingSheet:
    """A conducting sheet on an FDFD grid line (axis 'x' = along x): `line` indexes the line cells (ez E_z) or their hi
    faces (hz tangential E); sigma(omega) is evaluated once per omega."""

    def __init__(self, axis, line_cell, along, normal_width, cell_width, conductivity):
        if axis not in ('x', 'y'):
            raise ValueError(f"conducting sheet axis (along the line) must be 'x' or 'y', got {axis!r}")
        self.axis = axis
        self.line = (along, int(line_cell)) if axis == 'x' else (int(line_cell), along)
        self.normal_width = float(normal_width)   # w_perp of the delta: ez the line-cell width, hz the face spacing
        self.cell_width = float(cell_width)       # line-cell width across the line: its loss density divides by it
        self.conductivity = conductivity          # callable omega -> complex sigma_s [S] (SheetConductivity)
        self._cached = None                       # (omega, sigma)

    def sigma(self, omega):
        """Complex sheet conductivity [S] at omega (ValueError when not finite)."""
        if self._cached is None or self._cached[0] != omega:
            value = complex(self.conductivity(omega))
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                raise ValueError(f"conducting sheet conductivity at omega {omega:.6g} rad/s is not finite: {value!r}")
            self._cached = (omega, value)
        return self._cached[1]

    def permittivity_change(self, omega):
        """Relative permittivity of the sheet spread over w_perp: -j sigma / (omega eps0 w_perp) (the 3-point delta)."""
        return -1j * self.sigma(omega) / (float(omega) * EPS0 * self.normal_width)
