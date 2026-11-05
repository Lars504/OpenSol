"""MechanicsFields2D: full-grid displacement, strain, stress and electric output arrays plus the coupled
temperature-rise and body-force slots of solid_mechanics."""
from ...core.constants import DTYPE

# Cell-centre outputs (written after each solve, edge-extended into the PML ring) and the coupled / applied inputs; u_z
# is nonzero only for a guided (2.5D) vibration mode (plane strain has none), the electric ones only with piezoelectric
# (written on every window cell, void included).
ELECTRIC_OUTPUT_ARRAYS = ('electric_potential', 'electric_field_x', 'electric_field_y')
OUTPUT_ARRAYS = ('u_x', 'u_y', 'u_z', 'u_abs', 'strain_xx', 'strain_yy', 'strain_xy',
                 'elastic_strain_xx', 'elastic_strain_yy', 'elastic_strain_zz',
                 'stress_xx', 'stress_yy', 'stress_xy', 'stress_zz', 'von_mises') + ELECTRIC_OUTPUT_ARRAYS
INPUT_ARRAYS = ('temperature_rise', 'temperature_rise_ext', 'body_force_x', 'body_force_y')
# Coupled inputs (inject_source slots, full grid): temperature rise above the stress-free state [K] and body force
# density [N/m^3] (the Maxwell-stress coupling's target); temperature_rise_ext is written by temperature_change features.
SOURCE_SLOTS = ('temperature_rise', 'body_force_x', 'body_force_y')
# frequency_response display fields (triangle meshes only, no state() arrays): Im of the complex displacement [m]
HARMONIC_IMAG_FIELDS = ('u_x_im', 'u_y_im')
# energy density display fields [J/m^3] (triangle meshes only, no state() arrays): a static solve 1/2 sigma : eps_el
# (no initial stress) and 0; a vibration mode or a frequency response the time averages 1/4 Re(sigma : eps_el*) and
# 1/4 rho omega^2 |u|^2 (azimuthal order m >= 1: twice the phi mean, so ring integrals with weight pi give the totals)
ENERGY_DENSITY_FIELDS = ('strain_energy_density', 'kinetic_energy_density')


class MechanicsFields2D:
    """Full-grid float32 displacement [m], total and elastic (total - alpha dT) tensor strain [1] and stress [Pa] at the
    cell centres (0 in void cells), the coupled temperature rise [K] and body force [N/m^3] slots and the applied
    temperature_rise_ext [K]; arrays of xp (the domain's numpy or cupy)."""

    def __init__(self, nx: int, ny: int, xp):
        self.nx = nx
        self.ny = ny
        self.shape = (nx, ny)
        for name in OUTPUT_ARRAYS + INPUT_ARRAYS:
            setattr(self, name, xp.zeros(self.shape, dtype=DTYPE))
