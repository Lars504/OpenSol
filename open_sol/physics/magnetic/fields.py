"""Magnetic field sets: MagneticFields2D (A, B, H, w_mag, currents), MagneticFields3D (phi_m, B, H) and the AC phasor
MagneticFieldsAC2D."""
import numpy as np

from ...core.constants import DTYPE


class MagneticFields2D:
    """Full-grid float32 vector potential A [T m] (planar A_z, axisymmetric A_phi = psi / r), flux density B_x / B_y /
    B_abs [T], field H_x / H_y [A/m], energy density w_mag [J/m^3], coupled and applied current densities [A/m^2];
    xp: the Domain's array module."""

    def __init__(self, nx: int, ny: int, xp=np):
        self.nx = nx
        self.ny = ny
        shape = (nx, ny)
        self.A = xp.zeros(shape, dtype=DTYPE)
        self.B_x = xp.zeros(shape, dtype=DTYPE)
        self.B_y = xp.zeros(shape, dtype=DTYPE)
        self.B_abs = xp.zeros(shape, dtype=DTYPE)
        self.H_x = xp.zeros(shape, dtype=DTYPE)
        self.H_y = xp.zeros(shape, dtype=DTYPE)
        self.w_mag = xp.zeros(shape, dtype=DTYPE)
        self.J_coupled = xp.zeros(shape, dtype=DTYPE)   # coupled slot current_density (overwritten per inject_source)
        self.J_applied = xp.zeros(shape, dtype=DTYPE)   # current_source / coil features (they add)


# Full-grid float32 (nx, ny, nz) arrays of magnetostatics_3d: the reduced scalar potential and the B / H / energy fields.
MAGNETIC_3D_FIELD_NAMES = ('phi_m', 'B_x', 'B_y', 'B_z', 'B_abs', 'H_x', 'H_y', 'H_z', 'w_mag')
MAGNETIC_3D_CURRENT_NAMES = ('J_x', 'J_y', 'J_z')   # coil current density, allocated with the first coil
MAGNETIC_3D_PERMEABILITY_NAME = 'mu_eff'            # assembled (secant) permeability, allocated when a cell saturates
MAGNETIC_3D_PROBE_COMPONENTS = ('B_abs', 'phi_m')   # fields a magnetostatics_3d probe may record


class MagneticFields3D:
    """Full-grid float32 fields of magnetostatics_3d (MAGNETIC_3D_FIELD_NAMES): phi_m [A], B [T], H [A/m], w_mag
    [J/m^3]; J [A/m^2] and mu_eff [1] None until allocated; xp: the Domain's array module."""

    def __init__(self, shape, xp=np):
        self.shape = tuple(int(count) for count in shape)
        self.xp = xp
        for name in MAGNETIC_3D_FIELD_NAMES:
            setattr(self, name, xp.zeros(self.shape, dtype=DTYPE))
        for name in MAGNETIC_3D_CURRENT_NAMES:
            setattr(self, name, None)
        self.mu_eff = None

    @property
    def has_current_density(self) -> bool:
        return self.J_x is not None

    @property
    def has_effective_permeability(self) -> bool:
        return self.mu_eff is not None

    def allocate_current_density(self) -> None:
        """Allocate the zeroed J_x / J_y / J_z arrays (no-op when allocated)."""
        if not self.has_current_density:
            for name in MAGNETIC_3D_CURRENT_NAMES:
                setattr(self, name, self.xp.zeros(self.shape, dtype=DTYPE))

    def allocate_effective_permeability(self) -> None:
        """Allocate mu_eff, filled with 1 (no-op when allocated)."""
        if not self.has_effective_permeability:
            self.mu_eff = self.xp.ones(self.shape, dtype=DTYPE)


# Full-grid float32 arrays of the time-harmonic solve (exp(+j omega t) phasors, peak amplitudes): re / im parts,
# magnitudes, and the time-averaged loss and energy densities.
AC_FIELD_NAMES = ('A_re', 'A_im', 'A_abs', 'B_x_re', 'B_x_im', 'B_y_re', 'B_y_im', 'B_abs',
                  'J_conduction_re', 'J_conduction_im', 'J_total_re', 'J_total_im', 'J_total_abs',
                  'q_eddy', 'q_magnetic', 'w_mag', 'J_coupled', 'J_applied_re', 'J_applied_im')


class MagneticFieldsAC2D:
    """Full-grid float32 phasor fields of magnetic_fields_ac (AC_FIELD_NAMES): A [T m] (axisymmetric A_phi), B [T],
    conduction / total current density [A/m^2], q_eddy / q_magnetic [W/m^3], w_mag [J/m^3], J_coupled (real),
    J_applied (re / im); xp: the Domain's array module."""

    def __init__(self, nx: int, ny: int, xp=np):
        self.nx = nx
        self.ny = ny
        for name in AC_FIELD_NAMES:
            setattr(self, name, xp.zeros((nx, ny), dtype=DTYPE))
