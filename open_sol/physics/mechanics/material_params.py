"""Material property Params shared verbatim by solid_mechanics and elastic_waves (one help text per name: the library
youngs_modulus / poisson_ratio / photoelastic data serve both); definition-layer module (imports model.schema)."""
from ...model.definition_helpers import optional_material_param
from ...model.schema import Param

YOUNGS_MODULUS_PARAM = optional_material_param(
    'youngs_modulus', "Young's modulus", 'Pa', ('youngs_modulus',), 0.0,
    help_text="Young's modulus E of an isotropic solid. solid_mechanics: unset = 0 = void (not solved, no stress: air, "
              'a liquid, a cladding left out); every solid region needs a support. elastic_waves: c11 = c22 = lambda + '
              '2 mu, c12 = lambda, c66 = mu of cells without a sound_speed (sound_speed / shear_speed win; set c_ij '
              'win over both). Typical: silica 7.3e10, silicon 1.65e11 (isotropic average; 1.3e11 along <100>, '
              '1.69e11 along <110>), gold 7.9e10, aluminium 7.0e10. A state field (e.g. heat_T) makes a stationary '
              'study iterate.',
    tip='Stiffness of the material. Empty takes the library value; a material with neither is empty space for the '
        'mechanics.\n\nTypical: silica 73 GPa, silicon 165 GPa, gold 79 GPa.')

POISSON_RATIO_PARAM = Param(
    'poisson_ratio', 'float', None, label='Poisson ratio', unit='1', min=-1.0, max=0.499, min_exclusive=True,
    help='Poisson ratio nu of an isotropic solid (unset = 0), used with youngs_modulus. Plane strain stiffens the '
         'in-plane response by 1 / (1 - nu^2), sigma_zz = nu (sigma_xx + sigma_yy) - E alpha_zz dT. solid_mechanics: '
         'low-order elements lock toward 0.5, too stiff above ~0.45 (validation warns). Typical: silica 0.17, silicon '
         '0.22, gold 0.42, aluminium 0.35.',
    tip='Sideways contraction per unit stretch, between -1 and 0.5.\n\nTypical: silica 0.17, silicon 0.22, gold '
        '0.42. Values close to 0.5 (rubber) make the result too stiff.',
    optional=True, field_dependent=True, state_dependent=True, expands_to=('poisson_ratio',))

PHOTOELASTIC_P11_PARAM = optional_material_param(
    'photoelastic_p11', 'Photoelastic coefficient p11', '1', ('photoelastic_p11',),
    help_text='Strain-optic coefficient p11 of an isotropic or cubic material (axes along x, y, z): Delta(1/n^2)_ii = '
              'p11 e_ii + p12 (e_jj + e_kk) of the elastic strain. Read only by solid_em_strain_optic, '
              'acoustics_em_acousto_optic (into em_fdfd_hz: shear term (p11 - p12) / 2) and optomechanical_coupling '
              '(shear: photoelastic_p44). Unset = 0. Typical: silica 0.121, silicon -0.094. With the elastic_waves '
              'tensor entries (photoelastic_p16 .. _p66; a crystal entry also sets p11 / p12) it is p_11 of the '
              'model-axis row xx; solid_em_strain_optic keeps the isotropic reading, optomechanical_coupling reads a '
              "crystal entry's full rotated p (unless the material sets a photoelastic_* itself).",
    tip='How strongly stretching along the light polarisation changes the refractive index.\n\nTypical: silica '
        '0.121, silicon -0.094.')

PHOTOELASTIC_P12_PARAM = optional_material_param(
    'photoelastic_p12', 'Photoelastic coefficient p12', '1', ('photoelastic_p12',),
    help_text='Strain-optic coefficient p12 (strain across the field; see photoelastic_p11). Unset = 0. Typical: '
              'silica 0.270, silicon 0.017. With the elastic_waves tensor entries: p_12 of row xx and the fallback of '
              'the unset p21 / p31 / p32.',
    tip='How strongly stretching across the light polarisation changes the refractive index.\n\nTypical: silica '
        '0.270, silicon 0.017.')
