"""SolidRenderer3D: window-shaped (wx, wy, wz) displacement, strain and stress fields of a SolidMechanics3DPhysics."""
from ...core.renderer import FieldInfo, PhysicsRenderer
from .fields3d import ELECTRIC_OUTPUT_ARRAYS_3D, ENERGY_OUTPUT_ARRAYS_3D, HARMONIC_IMAG_FIELDS_3D

_FREQUENCY_RESPONSE_STUDY = 'frequency_response'   # = model.study_frequency_response.STUDY_NAME (requires_study)


def _signed(name, label, unit, parity, requires_study=''):
    return FieldInfo(name, label, unit=unit, signed=True, cmap='RdBu_r', parity=parity, requires_study=requires_study)


class SolidRenderer3D(PhysicsRenderer):
    """Window-shaped (wx, wy, wz) float32 fields of a SolidMechanics3DPhysics (views of the stored arrays; Session cuts
    the plane); parities of the roller (even) mirror (shear ij flips like the axial component of the third axis). The
    piezoelectric and energy-density fields are tetrahedra-only (fem.renderer3d over the same field_infos): zeros
    here."""

    name = 'solid'
    field_infos = {
        'u_abs': FieldInfo('u_abs', 'Displacement magnitude', unit='m', cmap='viridis', primary=True),
        'u_x': _signed('u_x', 'Displacement x', 'm', 'polar_x'),
        'u_y': _signed('u_y', 'Displacement y', 'm', 'polar_y'),
        'u_z': _signed('u_z', 'Displacement z', 'm', 'polar_z'),
        # frequency_response on tetrahedra: Im of the complex displacement (u_x .. u_z hold the real parts)
        'u_x_im': _signed('u_x_im', 'Displacement x, imaginary part', 'm', 'polar_x', _FREQUENCY_RESPONSE_STUDY),
        'u_y_im': _signed('u_y_im', 'Displacement y, imaginary part', 'm', 'polar_y', _FREQUENCY_RESPONSE_STUDY),
        'u_z_im': _signed('u_z_im', 'Displacement z, imaginary part', 'm', 'polar_z', _FREQUENCY_RESPONSE_STUDY),
        'von_mises': FieldInfo('von_mises', 'von Mises stress', unit='Pa', cmap='inferno'),
        'stress_xx': _signed('stress_xx', 'Stress xx', 'Pa', 'even'),
        'stress_yy': _signed('stress_yy', 'Stress yy', 'Pa', 'even'),
        'stress_zz': _signed('stress_zz', 'Stress zz', 'Pa', 'even'),
        'stress_xy': _signed('stress_xy', 'Shear stress xy', 'Pa', 'axial_z'),
        'stress_xz': _signed('stress_xz', 'Shear stress xz', 'Pa', 'axial_y'),
        'stress_yz': _signed('stress_yz', 'Shear stress yz', 'Pa', 'axial_x'),
        'strain_xx': _signed('strain_xx', 'Strain xx', '1', 'even'),
        'strain_yy': _signed('strain_yy', 'Strain yy', '1', 'even'),
        'strain_zz': _signed('strain_zz', 'Strain zz', '1', 'even'),
        'strain_xy': _signed('strain_xy', 'Shear strain xy', '1', 'axial_z'),
        'strain_xz': _signed('strain_xz', 'Shear strain xz', '1', 'axial_y'),
        'strain_yz': _signed('strain_yz', 'Shear strain yz', '1', 'axial_x'),
        'elastic_strain_xx': _signed('elastic_strain_xx', 'Elastic strain xx', '1', 'even'),
        'elastic_strain_yy': _signed('elastic_strain_yy', 'Elastic strain yy', '1', 'even'),
        'elastic_strain_zz': _signed('elastic_strain_zz', 'Elastic strain zz', '1', 'even'),
        'temperature_rise': _signed('temperature_rise', 'Coupled temperature rise', 'K', 'even'),
        'temperature_rise_ext': _signed('temperature_rise_ext', 'Applied temperature change', 'K', 'even'),
        'body_force_x': _signed('body_force_x', 'Coupled body force x', 'N/m^3', 'polar_x'),
        'body_force_y': _signed('body_force_y', 'Coupled body force y', 'N/m^3', 'polar_y'),
        'body_force_z': _signed('body_force_z', 'Coupled body force z', 'N/m^3', 'polar_z'),
        # piezoelectric only, tetrahedra (0 otherwise): nodal potential, E = -grad phi, D = e S + eps0 eps E
        'electric_potential': _signed('electric_potential', 'Electric potential (piezoelectric)', 'V', 'even'),
        'electric_field_x': _signed('electric_field_x', 'Electric field x (piezoelectric)', 'V/m', 'polar_x'),
        'electric_field_y': _signed('electric_field_y', 'Electric field y (piezoelectric)', 'V/m', 'polar_y'),
        'electric_field_z': _signed('electric_field_z', 'Electric field z (piezoelectric)', 'V/m', 'polar_z'),
        'electric_displacement_x': _signed('electric_displacement_x', 'Electric displacement x (piezoelectric)',
                                           'C/m^2', 'polar_x'),
        'electric_displacement_y': _signed('electric_displacement_y', 'Electric displacement y (piezoelectric)',
                                           'C/m^2', 'polar_y'),
        'electric_displacement_z': _signed('electric_displacement_z', 'Electric displacement z (piezoelectric)',
                                           'C/m^2', 'polar_z'),
        # tetrahedra only (0 on a tensor mesh): static 1/2 eps_el : C eps_el; a mode / response its time average
        'strain_energy_density': FieldInfo('strain_energy_density', 'Strain energy density', unit='J/m^3',
                                           cmap='inferno'),
        'kinetic_energy_density': FieldInfo('kinetic_energy_density', 'Kinetic energy density (time average)',
                                            unit='J/m^3', cmap='inferno'),
        'youngs_modulus': FieldInfo('youngs_modulus', "Young's modulus", unit='Pa', cmap='cividis'),
        'poisson_ratio': FieldInfo('poisson_ratio', 'Poisson ratio', unit='1', cmap='cividis'),
    }

    def __init__(self, physics):
        self.physics = physics

    def device_field(self, component: str):
        self.field_info(component)
        physics = self.physics
        window = physics.window
        if component in ('youngs_modulus', 'poisson_ratio'):
            return getattr(physics.domain.materials, component)[window]
        if component in ELECTRIC_OUTPUT_ARRAYS_3D or component in ENERGY_OUTPUT_ARRAYS_3D:
            return physics.domain.xp.zeros_like(physics.fields.u_x[window])
        if component in HARMONIC_IMAG_FIELDS_3D:
            raise ValueError(f"field '{component}' exists on a tetrahedron mesh only (the rectangular-cell solid has no "
                             'frequency response)')
        return getattr(physics.fields, component)[window]

    def time_label(self):
        return f"solve {self.physics.n_solves}"
