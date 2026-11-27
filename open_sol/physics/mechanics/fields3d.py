"""Full-grid float32 fields of the 3D solid (outputs in kernels3d order, coupled / applied inputs) on the domain's array
module; no cupy import, so definition3d can import the names."""
from ...core.constants import DTYPE
from .fields import PRESSURE_FIELDS
from .kernels3d import OUTPUT_ARRAYS_3D

# Piezoelectric outputs (tetrahedra only, 0 without the setting piezoelectric; a tensor mesh renders them as 0 and keeps
# no arrays): potential [V], E = -grad phi [V/m], D = e S + eps0 eps E [C/m^2].
ELECTRIC_OUTPUT_ARRAYS_3D = ('electric_potential', 'electric_field_x', 'electric_field_y', 'electric_field_z',
                             'electric_displacement_x', 'electric_displacement_y', 'electric_displacement_z')
# Energy densities [J/m^3] (tetrahedra only; a tensor mesh renders them as 0 and keeps no arrays):
# static 1/2 eps_el : C eps_el and 0; a shown mode / response the time averages 1/4 Re(eps* : C eps), 1/4 rho omega^2 |u|^2
ENERGY_OUTPUT_ARRAYS_3D = ('strain_energy_density', 'kinetic_energy_density')
# Coupled / applied inputs; inject_source slots = all but temperature_rise_ext (written by temperature_change features).
INPUT_ARRAYS_3D = ('temperature_rise', 'temperature_rise_ext', 'body_force_x', 'body_force_y', 'body_force_z')
SOURCE_SLOTS_3D = ('temperature_rise', 'body_force_x', 'body_force_y', 'body_force_z')
# frequency_response display fields (tetrahedra only, no state() arrays): Im of the complex displacement [m]
HARMONIC_IMAG_FIELDS_3D = ('u_x_im', 'u_y_im', 'u_z_im')
# frequency_response pressure display fields (tetrahedra only, nodal, no state() arrays): fields.PRESSURE_FIELDS
PRESSURE_FIELDS_3D = PRESSURE_FIELDS


class MechanicsFields3D:
    """(nx, ny, nz) float32 coupled temperature rise [K], body force [N/m^3] and applied temperature change [K]
    (INPUT_ARRAYS_3D) on xp (numpy or cupy). The outputs (displacement, strains, stresses, von Mises at the cell
    centres) are no arrays here: SolidMechanics3DPhysics.output computes one on read."""

    def __init__(self, nx: int, ny: int, nz: int, xp):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.shape = (nx, ny, nz)
        for name in INPUT_ARRAYS_3D:
            setattr(self, name, xp.zeros(self.shape, dtype=DTYPE))
