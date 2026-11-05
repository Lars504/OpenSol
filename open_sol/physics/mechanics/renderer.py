"""SolidRenderer: window-shaped displacement, strain, stress, piezoelectric, input and stiffness fields of a 2D
SolidMechanicsPhysics, plus the optomechanical moving-boundary density and (triangle meshes) the energy densities."""
from dataclasses import replace

from ...core.renderer import FieldInfo, PhysicsRenderer
from .fields import ENERGY_DENSITY_FIELDS, HARMONIC_IMAG_FIELDS
from .optomechanics import shown_integrand

_OPTOMECHANICS_STUDY = 'optomechanics'   # = model.study_optomechanics.STUDY_NAME (FieldInfo.requires_study)
_FREQUENCY_RESPONSE_STUDY = 'frequency_response'   # = model.study_frequency_response.STUDY_NAME


class SolidRenderer(PhysicsRenderer):
    """Window-shaped float32 fields of a SolidMechanicsPhysics: displacement, strain, stress, inputs, stiffness;
    parities of the roller (even) mirror of a model's EM mirror faces (core.mirror: shear xy flips like axial_z)."""

    name = 'solid'
    field_infos = {
        'u_abs': FieldInfo('u_abs', 'Displacement magnitude', unit='m', cmap='viridis', primary=True),
        'u_x': FieldInfo('u_x', 'Displacement x', unit='m', signed=True, cmap='RdBu_r', parity='polar_x'),
        'u_y': FieldInfo('u_y', 'Displacement y', unit='m', signed=True, cmap='RdBu_r', parity='polar_y'),
        # frequency_response on a triangle mesh: Im of the complex displacement (u_x / u_y hold the real parts)
        'u_x_im': FieldInfo('u_x_im', 'Displacement x, imaginary part', unit='m', signed=True, cmap='RdBu_r',
                            parity='polar_x', requires_study=_FREQUENCY_RESPONSE_STUDY),
        'u_y_im': FieldInfo('u_y_im', 'Displacement y, imaginary part', unit='m', signed=True, cmap='RdBu_r',
                            parity='polar_y', requires_study=_FREQUENCY_RESPONSE_STUDY),
        # guided (2.5D) vibration modes only: amplitude of u_z = u_z sin(k_z z - omega t); 0 in plane strain
        'u_z': FieldInfo('u_z', 'Displacement z (guided modes)', unit='m', signed=True, cmap='RdBu_r',
                         parity='polar_z'),
        'von_mises': FieldInfo('von_mises', 'von Mises stress', unit='Pa', cmap='inferno'),
        # triangle meshes only (fields.ENERGY_DENSITY_FIELDS): static 1/2 sigma : eps_el, modes / responses time averages
        'strain_energy_density': FieldInfo('strain_energy_density', 'Strain energy density', unit='J/m^3',
                                           cmap='inferno', parity='even'),
        'kinetic_energy_density': FieldInfo('kinetic_energy_density', 'Kinetic energy density (time average)',
                                            unit='J/m^3', cmap='inferno', parity='even'),
        'stress_xx': FieldInfo('stress_xx', 'Stress xx', unit='Pa', signed=True, cmap='RdBu_r', parity='even'),
        'stress_yy': FieldInfo('stress_yy', 'Stress yy', unit='Pa', signed=True, cmap='RdBu_r', parity='even'),
        'stress_xy': FieldInfo('stress_xy', 'Shear stress xy', unit='Pa', signed=True, cmap='RdBu_r',
                               parity='axial_z'),
        'stress_zz': FieldInfo('stress_zz', 'Stress zz (out of plane)', unit='Pa', signed=True, cmap='RdBu_r',
                               parity='even'),
        'strain_xx': FieldInfo('strain_xx', 'Strain xx', unit='1', signed=True, cmap='RdBu_r', parity='even'),
        'strain_yy': FieldInfo('strain_yy', 'Strain yy', unit='1', signed=True, cmap='RdBu_r', parity='even'),
        'strain_xy': FieldInfo('strain_xy', 'Shear strain xy', unit='1', signed=True, cmap='RdBu_r',
                               parity='axial_z'),
        'elastic_strain_xx': FieldInfo('elastic_strain_xx', 'Elastic strain xx', unit='1', signed=True, cmap='RdBu_r',
                                       parity='even'),
        'elastic_strain_yy': FieldInfo('elastic_strain_yy', 'Elastic strain yy', unit='1', signed=True, cmap='RdBu_r',
                                       parity='even'),
        'elastic_strain_zz': FieldInfo('elastic_strain_zz', 'Elastic strain zz (out of plane)', unit='1', signed=True,
                                       cmap='RdBu_r', parity='even'),
        'temperature_rise': FieldInfo('temperature_rise', 'Coupled temperature rise', unit='K', signed=True,
                                      cmap='RdBu_r', parity='even'),
        'temperature_rise_ext': FieldInfo('temperature_rise_ext', 'Applied temperature change', unit='K',
                                          signed=True, cmap='RdBu_r', parity='even'),
        'body_force_x': FieldInfo('body_force_x', 'Coupled body force x', unit='N/m^3', signed=True, cmap='RdBu_r',
                                  parity='polar_x'),
        'body_force_y': FieldInfo('body_force_y', 'Coupled body force y', unit='N/m^3', signed=True, cmap='RdBu_r',
                                  parity='polar_y'),
        # piezoelectric only (0 otherwise): nodal Q1 potential at the cell centres and E = -grad phi
        'electric_potential': FieldInfo('electric_potential', 'Electric potential (piezoelectric)', unit='V',
                                        signed=True, cmap='RdBu_r', parity='even'),
        'electric_field_x': FieldInfo('electric_field_x', 'Electric field x (piezoelectric)', unit='V/m',
                                      signed=True, cmap='RdBu_r', parity='polar_x'),
        'electric_field_y': FieldInfo('electric_field_y', 'Electric field y (piezoelectric)', unit='V/m',
                                      signed=True, cmap='RdBu_r', parity='polar_y'),
        'youngs_modulus': FieldInfo('youngs_modulus', "Young's modulus", unit='Pa', cmap='cividis'),
        'poisson_ratio': FieldInfo('poisson_ratio', 'Poisson ratio', unit='1', cmap='cividis'),
        # moving-boundary g_om density of the shown mechanical mode (first optomechanical_coupling): Q_n times the
        # permittivity-jump face term, half to each cell of the face, per cell area; mirror-even (Q_n, |E|^2 even)
        'om_integrand': FieldInfo('om_integrand', 'Optomechanical moving-boundary density', unit='rad/s/m^3',
                                  signed=True, cmap='RdBu_r', requires_study=_OPTOMECHANICS_STUDY, parity='even'),
    }

    def __init__(self, physics):
        self.physics = physics
        if getattr(physics, 'axisymmetric', False):   # (r, z): x = r, y = z, the out-of-plane zz is the hoop
            self.field_infos = {**type(self).field_infos, **_AXISYMMETRIC_LABELS}
            if getattr(physics, 'azimuthal_order', 0):   # order m >= 1: u_z holds the azimuthal amplitude U_phi
                self.field_infos['u_z'] = _AZIMUTHAL_U_PHI

    def device_field(self, component: str):
        self.field_info(component)
        physics = self.physics
        window = physics.window
        if component == 'om_integrand':
            return physics.domain.xp.asarray(shown_integrand(physics))
        if component in ENERGY_DENSITY_FIELDS:   # triangle meshes only: zeros here (as the 3D tensor renderer)
            return physics.domain.xp.zeros_like(physics.fields.u_x[window])
        if component in HARMONIC_IMAG_FIELDS:
            raise ValueError(f"field '{component}' exists on a triangle mesh only (the rectangular-cell solid shows the "
                             'real parts and u_abs of a frequency response)')
        if component in ('youngs_modulus', 'poisson_ratio'):
            return getattr(physics.domain.materials, component)[window]
        return getattr(physics.fields, component)[window]

    def time_label(self):
        return f"solve {self.physics.n_solves}"


# geometry axisymmetric: display labels of the hoop (theta theta) outputs (names unchanged)
_AXISYMMETRIC_LABELS = {
    'stress_zz': replace(SolidRenderer.field_infos['stress_zz'], label='Hoop stress (theta theta)'),
    'elastic_strain_zz': replace(SolidRenderer.field_infos['elastic_strain_zz'],
                                 label='Elastic hoop strain (theta theta)'),
}
# azimuthal order m >= 1: field u_z is the amplitude of u_phi = U_phi sin(m phi)
_AZIMUTHAL_U_PHI = replace(SolidRenderer.field_infos['u_z'], label='Displacement phi (amplitude of sin(m phi))')
