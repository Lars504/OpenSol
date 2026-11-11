"""Model-layer pieces shared by the solid_mechanics (2D) and solid_mechanics_3d definitions: the build's linear
solver, the material vocabulary (one Param object per name), support helpers, the CPU material / state / modal-study /
prestress checks, the study a solid takes (StudySpec.solved_physics) and the resonance_table / frequency_probe
features."""
from dataclasses import replace

from ...model.definition_helpers import (conducting_fraction, eigen_search, evaluates_to, optional_material_param,
                                         reports_model_checks)
from ...model.expressions import TIME_NAMES, ExpressionError, evaluate_scalar, referenced_names
from ...model.schema import FeatureType, Issue, Param, feature_path, resolve_values, state_field_names
from ...solvers.pcg import resolve_max_iterations
from .material_params import (PHOTOELASTIC_P11_PARAM, PHOTOELASTIC_P12_PARAM, POISSON_RATIO_PARAM,
                              YOUNGS_MODULUS_PARAM)
from .materials import (MECHANICAL_PROPERTY_ARRAYS, PERMITTIVITY_TENSOR_ARRAYS_3D, PIEZO_TENSOR_ARRAYS_3D,
                        PRESTRESS_ARRAYS_3D, STIFFNESS_ARRAYS, STIFFNESS_TENSOR_ARRAYS_3D)
from .modal import MODAL_TABLE_COLUMNS, ModalTable, modal_table_result
from .tangent_update import SLOW_RESIDUAL_RATIO, TANGENT_UPDATES

LOCKING_POISSON_RATIO = 0.45   # bilinear / trilinear elements grow too stiff (volumetric locking) above about this
SUPPORT_FEATURES = frozenset({'fixed', 'roller', 'displacement', 'fixed_domain', 'prescribed_displacement_domain'})
LOAD_FEATURES = frozenset({'boundary_load', 'body_force', 'temperature_change'})
EIGENFREQUENCY_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME
OPTOMECHANICS_STUDY = 'optomechanics'     # = model.study_optomechanics.STUDY_NAME
FREQUENCY_RESPONSE_STUDY_NAME = 'frequency_response'   # = model.study_frequency_response.STUDY_NAME
BACKGROUND_STUDY = 'stationary'           # the background study of the modal / response studies
PRESTRESS_STUDIES = (EIGENFREQUENCY_STUDY, FREQUENCY_RESPONSE_STUDY_NAME, OPTOMECHANICS_STUDY)   # read stress_stiffening


# ---- the study a solid takes (StudySpec.solved_physics) ----
def solid_study(spec, physics_name) -> str:
    """The study type the solid physics_name runs in: the model's, or the background 'stationary' when
    StudySpec.solved_physics leaves it out."""
    return spec.study.type if spec.study.solves(physics_name) else BACKGROUND_STUDY


def background_context(context, physics_spec):
    """The ValidationContext / MemoryContext / TimeContext as the solid sees it: unchanged when the study solves it,
    else a copy whose study reads 'stationary' (StudySpec.solved_physics leaves it out: a background physics)."""
    study = context.spec.study
    if study.solves(physics_spec.name):
        return context
    changes = {'spec': replace(context.spec, study=replace(study, type=BACKGROUND_STUDY, solved_physics=[]))}
    if isinstance(getattr(context, 'study_type', None), str):   # ValidationContext
        changes['study_type'] = BACKGROUND_STUDY
    counts = getattr(context, 'study', None)
    if counts is not None and hasattr(counts, 'counts'):   # TimeContext.study (StudyCounts)
        changes['study'] = replace(counts, type=BACKGROUND_STUDY)
    return replace(context, **changes)

# ---- solver ----
def solid_linear_solver(ctx, physics_spec):
    """The build's 'linear' pcg solver of the physics; max_iterations 0 (auto) = twice the DiffusionSystem limit."""
    built = ctx.solver_records[physics_spec.name]['linear']
    if not built.resolved.values['max_iterations']:
        built.runtime.max_iterations = 2 * resolve_max_iterations(0, ctx.domain)
    return built.runtime


# ---- material vocabulary ----
def _material_property(name, label, unit, minimum=None, help_text='', tip=''):
    """Mechanical material property Param (default None: unset properties are never written)."""
    return optional_material_param(name, label, unit, MECHANICAL_PROPERTY_ARRAYS[name], minimum, help_text, tip)


_ALPHA_COMPONENT_TIP = "Thermal expansion along {axis}. Replaces 'Thermal expansion coefficient' for this direction."
MATERIAL_PROPERTIES = (   # isotropic entries before their components (components override)
    YOUNGS_MODULUS_PARAM,
    POISSON_RATIO_PARAM,
    _material_property('alpha', 'Thermal expansion coefficient', '1/K',
                       help_text='Linear thermal expansion coefficient; also read by heat_em_thermo_optic (see the heat '
                                 'type).',
                       tip='Growth in length per kelvin.'),
    _material_property('alpha_xx', 'Thermal expansion along x', '1/K',
                       help_text='Expansion coefficient along x (see alpha).',
                       tip=_ALPHA_COMPONENT_TIP.format(axis='x')),
    _material_property('alpha_yy', 'Thermal expansion along y', '1/K',
                       help_text='Expansion coefficient along y (see alpha).',
                       tip=_ALPHA_COMPONENT_TIP.format(axis='y')),
    _material_property('alpha_zz', 'Thermal expansion along z', '1/K',
                       help_text='Expansion coefficient along z, read only by solid_mechanics and solid_mechanics_3d; '
                                 'overrides alpha for this axis. 2D plane strain holds eps_zz = 0, so alpha_zz dT '
                                 'becomes out-of-plane stress and, through the Poisson effect, in-plane strain (plane '
                                 'stress: a free expansion along z; generalized plane strain: part of the uniform '
                                 'eps_zz balance; axisymmetric 2D: the hoop expansion); in 3D it is an ordinary '
                                 'expansion along z.',
                       tip=_ALPHA_COMPONENT_TIP.format(axis='z')),
    PHOTOELASTIC_P11_PARAM,
    PHOTOELASTIC_P12_PARAM,
)
# Both solid types (read by their eigenfrequency study only); the same name, unit and bounds as heat's / elastic_waves'.
DENSITY_PARAM = _material_property(
    'rho', 'Density', 'kg/m^3', 0.0,
    help_text='Mass density; solid_mechanics and solid_mechanics_3d read it only in an eigenfrequency (or '
              'optomechanics) study (consistent mass of the solid cells; every solid cell, youngs_modulus or c11 > 0, '
              'needs rho > 0 there). Shared with heat / elastic_waves. Typical: silica 2200, silicon 2329, gold 19300.',
    tip="Mass density. Solid mechanics uses it only in an 'Eigenfrequency' study.\n\nTypical: silica 2200, silicon "
        '2329, gold 19300.')
# 2D solid_mechanics only: read by the optomechanical_coupling overlap alone (the strain-optic couplings are diagonal).
PHOTOELASTIC_P44_PARAM = _material_property(
    'photoelastic_p44', 'Photoelastic coefficient p44', '1',
    help_text='Shear strain-optic coefficient p44 (Voigt: Delta(1/n^2)_xy = p44 x the engineering shear 2 strain_xy; '
              'cubic crystal axes along x, y, z), read only by the optomechanical_coupling g0 overlap '
              '(solid_em_strain_optic and acoustics_em_acousto_optic have no shear term). Unset = the isotropic '
              '(photoelastic_p11 - photoelastic_p12) / 2 per cell. Typical: silicon -0.051 (silica isotropic, -0.075).',
    tip='How strongly shearing the material changes its refractive index; only the optomechanical coupling uses '
        'it.\n\nEmpty takes the isotropic value from the other two coefficients. Typical: silicon -0.051.')
# 2D solid_mechanics with piezoelectric on (kind, unit, bounds and flags as elastic_waves' piezo_e* and electrostatics'
# eps_dc*: the registry rule; the arrays are shared).
PIEZO_MATERIAL_PROPERTIES = (
    *(_material_property(
        name, f'Piezoelectric e{name[-2:]}', 'C/m^2',
        help_text=f'Piezoelectric stress constant e_{name[-2]}{name[-1]} [C/m^2] of the in-plane block in model axes '
                  '(reduced Voigt: T_J = c_JK S_K - e_kJ E_k, D_k = e_kJ S_J + eps0 eps_dc E_k with k = 1 x / 2 y and J '
                  '= 1 xx, 2 yy, 6 xy, engineering shear S_6 = 2 strain_xy); default 0; the same arrays as '
                  'elastic_waves. solid_mechanics reads them only with its setting piezoelectric on, and only in solid '
                  "cells (youngs_modulus or c11 > 0). A {'model': 'crystal'} entry sets them together with the c_ij, "
                  'which solid_mechanics reads (2D).',
        tip=f'Piezoelectric constant e{name[-2:]}: how strongly an electric field along '
            f"{'x' if name[-2] == '1' else 'y'} stresses the material. Used only when Piezoelectric is on.")
      for name in ('piezo_e11', 'piezo_e12', 'piezo_e16', 'piezo_e21', 'piezo_e22', 'piezo_e26')),
    *(_material_property(
        name, f'Piezoelectric e{name[-2:]}', 'C/m^2',
        help_text=f'Out-of-plane piezoelectric constant e_{name[-2]}3 [C/m^2] (D_{name[-2]} versus the strain S_zz, '
                  'model axes); default 0. Plane strain holds S_zz = 0, so it enters only the out-of-plane stress: '
                  f'stress_zz -= e_{name[-2]}3 E_{"x" if name[-2] == "1" else "y"} (2D solid_mechanics with '
                  "piezoelectric on). A {'model': 'crystal'} entry sets it.",
        tip=f"How strongly an electric field along {'x' if name[-2] == '1' else 'y'} stresses the material along the "
            'depth of the drawing. Used only when Piezoelectric is on; usually filled in by a crystal entry.')
      for name in ('piezo_e13', 'piezo_e23')),
    _material_property('eps_dc', 'Static relative permittivity', '1', 1.0,
                       help_text='Isotropic static relative permittivity (>= 1): sets eps_dc_xx and eps_dc_yy (component '
                                 'entries override it); unset = 1 (vacuum). The same arrays as electrostatics / '
                                 'drift_diffusion; solid_mechanics reads them only with piezoelectric on, as the '
                                 'clamped (constant-strain) permittivity eps^S of every window cell, void ones '
                                 'included. NOT the optical eps_r: lithium niobate eps^S 44 / 29 (along the optic '
                                 'axis), silica 3.9.',
                       tip='Relative permittivity for static fields; empty means vacuum, 1. Used by solid mechanics '
                           'only when Piezoelectric is on.'),
    _material_property('eps_dc_xx', 'Static relative permittivity along x', '1', 1.0,
                       help_text='Static relative permittivity along x (>= 1); overrides eps_dc for this axis.',
                       tip="Static relative permittivity along x. Replaces 'Static relative permittivity' for this "
                           'direction.'),
    _material_property('eps_dc_yy', 'Static relative permittivity along y', '1', 1.0,
                       help_text='Static relative permittivity along y (>= 1); overrides eps_dc for this axis.',
                       tip="Static relative permittivity along y. Replaces 'Static relative permittivity' for this "
                           'direction.'),
    _material_property('eps_dc_xy', 'Static relative permittivity, cross term', '1',
                       help_text='Off-diagonal static permittivity of a rotated anisotropic material; keep eps_dc_xy^2 < '
                                 'eps_dc_xx * eps_dc_yy (else the solid_mechanics build fails). Default 0.',
                       tip='Cross term of a rotated material: the field also polarises it sideways. Usually empty.'),
)
# solid_mechanics_3d on tetrahedra with piezoelectric on: the full e_iJ and the six permittivities (the 2D Param objects
# for the shared names; its own eps_dc expands to xx, yy and zz).
_PIEZO_TEXT_3D = (
    'Piezoelectric stress constant e_{index} [C/m^2] in model axes (T_J = c_JK S_K - e_kJ E_k, D_k = e_kJ S_J + eps0 '
    'eps_dc_kl E_l with k = 1 x, 2 y, 3 z and J = 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy, engineering shear S_4 = 2 '
    'strain_yz); default 0. Read only by solid_mechanics_3d on a tetrahedron mesh with its setting piezoelectric on, in '
    'solid elements (youngs_modulus or c11 > 0); any piezo_e* entry allocates all 18 e_iJ there (the 2D names keep '
    "their meaning). A {{'model': 'crystal', 'plane': 'full'}} entry sets all 18.")
_PIEZO_2D_PARAMS = {param.name: param for param in PIEZO_MATERIAL_PROPERTIES}
_PIEZO_3D_ONLY = {
    name: _material_property(name, f'Piezoelectric e{name[-2:]}', 'C/m^2',
                             help_text=_PIEZO_TEXT_3D.format(index=name[-2:]),
                             tip=f"Piezoelectric constant e{name[-2:]}: how strongly an electric field along "
                                 f"{'xyz'[int(name[-2]) - 1]} stresses the material. Used only when Piezoelectric is on; "
                                 'usually filled in by a crystal entry.')
    for name in PIEZO_TENSOR_ARRAYS_3D if name not in _PIEZO_2D_PARAMS}
_PERMITTIVITY_3D_ONLY = {
    'eps_dc': optional_material_param(
        'eps_dc', 'Static relative permittivity', '1', ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_zz'), 1.0,
        help_text='Isotropic static relative permittivity (>= 1): in solid_mechanics_3d it sets eps_dc_xx, eps_dc_yy '
                  'and eps_dc_zz (component entries override it); unset = 1 (vacuum). Read only with piezoelectric on '
                  '(tetrahedra), as the clamped permittivity eps^S of every window element, void ones included.',
        tip='Relative permittivity for static fields; empty means vacuum, 1. Used by solid mechanics only when '
            'Piezoelectric is on.'),
    'eps_dc_zz': _material_property('eps_dc_zz', 'Static relative permittivity along z', '1', 1.0,
                                    help_text='Static relative permittivity along z (>= 1; solid_mechanics_3d with '
                                              'piezoelectric on); overrides eps_dc for this axis.',
                                    tip="Static relative permittivity along z. Replaces 'Static relative permittivity' "
                                        'for this direction.'),
    **{name: _material_property(name, f'Static relative permittivity, cross term {name[-2:]}', '1',
                                help_text=f'Off-diagonal static permittivity {name[-2:]} of a rotated anisotropic '
                                          'material (solid_mechanics_3d with piezoelectric on); the 3 x 3 eps_dc must '
                                          'be positive definite (else the build fails). Default 0.',
                                tip='Cross term of a rotated material: the field also polarises it sideways. Usually '
                                    'empty.')
       for name in ('eps_dc_yz', 'eps_dc_xz')},
}
# isotropic eps_dc before its components; e in PIEZO_TENSOR_ARRAYS_3D order
PIEZO_MATERIAL_PROPERTIES_3D = (
    *(_PIEZO_3D_ONLY.get(name) or _PIEZO_2D_PARAMS[name] for name in PIEZO_TENSOR_ARRAYS_3D),
    _PERMITTIVITY_3D_ONLY['eps_dc'],
    *(_PERMITTIVITY_3D_ONLY.get(name) or _PIEZO_2D_PARAMS[name] for name in PERMITTIVITY_TENSOR_ARRAYS_3D),
)
# 2D solid_mechanics only: the anisotropic stiffness (c11 / c12 / c22 / c66 / c16 / c26 with elastic_waves' kind, unit
# and bounds: the arrays are shared) and the frequency_response losses.
_TENSOR_ROLES = {'c11': 'xx-xx', 'c12': 'xx-yy', 'c13': 'xx-zz', 'c22': 'yy-yy', 'c23': 'yy-zz', 'c33': 'zz-zz',
                 'c16': 'xx-xy', 'c26': 'yy-xy', 'c36': 'zz-xy', 'c66': 'xy-xy'}
_TENSOR_FALLBACKS = {'c11': 'lambda + 2 mu', 'c22': 'lambda + 2 mu', 'c33': 'lambda + 2 mu', 'c12': 'lambda',
                     'c13': 'lambda', 'c23': 'lambda', 'c66': 'mu', 'c16': '0', 'c26': '0', 'c36': '0'}
_TENSOR_TEXT = (
    'Anisotropic stiffness entry {name} [Pa] ({role}) of the Voigt matrix in model axes (1 xx, 2 yy, 3 zz, 6 xy; '
    'engineering shear g = 2 strain_xy): T_I = c_IJ S_J - beta_I dT with beta_I = c_IJ alpha_J. solid_mechanics (2D) '
    'reads all ten (COMSOL plane strain: the full law with S_zz = S_xz = S_yz = 0, stress_zz = T_3); plane stress '
    'condenses T_3 = 0, generalized plane strain takes a uniform S_3, axisymmetric reads 3 as the hoop direction '
    '(orthotropic only: c16 = c26 = c36 = 0, else the build fails). Unset (NaN) = the isotropic value {fallback} of '
    'youngs_modulus / poisson_ratio per component (Lame constants), so a cell with only c_ij entries needs c11, c12, '
    'c22, c66 (plane stress / axisymmetric / generalized: also c13, c23, c33); any entry set allocates the eight '
    'arrays besides c16 / c26, which (as in elastic_waves) only c16 or c26 allocates. '
    'The set the formulation reads must be positive definite (build error). A cell is solid when its c11 > 0. Usually '
    "set by a {{'model': 'crystal'}} entry (rotated crystal data, with piezo_e* and eps_dc*). Not for guided (2.5D) "
    'modes (axial_wavenumber: isotropic only). solid_mechanics_3d on a tetrahedron mesh reads it too, as one of its 21 '
    'entries (see c44).')
# solid_mechanics_3d on tetrahedra only: the entries with a yz (4) or xz (5) index
_TENSOR_ROLES_3D = {'c14': 'xx-yz', 'c15': 'xx-xz', 'c24': 'yy-yz', 'c25': 'yy-xz', 'c34': 'zz-yz', 'c35': 'zz-xz',
                    'c44': 'yz-yz', 'c45': 'yz-xz', 'c46': 'yz-xy', 'c55': 'xz-xz', 'c56': 'xz-xy'}
_TENSOR_TEXT_3D = (
    'Anisotropic stiffness entry {name} [Pa] ({role}) of the Voigt matrix in model axes (1 xx, 2 yy, 3 zz, 4 yz, 5 xz, '
    '6 xy; engineering shear, S_4 = 2 strain_yz): T_I = c_IJ S_J - beta_I dT with beta_I = c_IJ alpha_J. Read only by '
    'solid_mechanics_3d on a tetrahedron mesh (a rectangular-cell mesh refuses c_ij entries), which takes all 21 '
    'entries c11 .. c66 (the 2D names keep their meaning); any entry set allocates all 21. Unset = {fallback} (c11, '
    'c22, c33: lambda + 2 mu, c12, c13, c23: lambda, c44, c55, c66: mu of youngs_modulus / poisson_ratio; every other '
    'entry 0), so a cell with only c_ij entries needs the nine of the orthotropic pattern. The 6 x 6 matrix must be '
    "positive definite (build error); a cell is solid when its c11 > 0. Usually set by a {{'model': 'crystal'}} entry "
    "with 'plane': 'full'.")
_STIFFNESS_TIP = ("Stiffness {name} of a crystal (an anisotropic material); empty takes the value from the Young's "
                  'modulus and Poisson ratio. Usually filled in by a crystal entry.')
STIFFNESS_TENSOR_PROPERTIES = tuple(
    _material_property(name, f'Stiffness {name}', 'Pa', 0.0 if name in ('c11', 'c22', 'c33', 'c66') else None,
                       help_text=_TENSOR_TEXT.format(name=name, role=_TENSOR_ROLES[name],
                                                     fallback=_TENSOR_FALLBACKS[name]),
                       tip=_STIFFNESS_TIP.format(name=name))
    for name in ('c11', 'c12', 'c13', 'c22', 'c23', 'c33', 'c16', 'c26', 'c36', 'c66'))
_STIFFNESS_3D_ONLY = {
    name: _material_property(name, f'Stiffness {name}', 'Pa', 0.0 if name in ('c44', 'c55') else None,
                             help_text=_TENSOR_TEXT_3D.format(name=name, role=_TENSOR_ROLES_3D[name],
                                                              fallback='mu' if name in ('c44', 'c55') else '0'),
                             tip=_STIFFNESS_TIP.format(name=name))
    for name in _TENSOR_ROLES_3D}
# solid_mechanics_3d: the same Param objects for the ten shared names, Voigt order c11, c12, .. c66
STIFFNESS_TENSOR_PROPERTIES_3D = tuple(
    _STIFFNESS_3D_ONLY.get(name) or next(param for param in STIFFNESS_TENSOR_PROPERTIES if param.name == name)
    for name in STIFFNESS_TENSOR_ARRAYS_3D)
LOSS_PROPERTIES = (
    _material_property('mechanical_loss_factor', 'Mechanical loss factor', '1', 0.0,
                       help_text='Isotropic structural loss factor eta (COMSOL: isotropic loss factor eta_s), read by '
                                 'the frequency_response study of solid_mechanics (2D) and solid_mechanics_3d '
                                 '(tetrahedron meshes) and by their eigenfrequency study on triangle / tetrahedron '
                                 'meshes (damped complex modes, Q = Re omega / (2 Im omega) in the resonance table; not '
                                 'with piezoelectric): every stiffness of the cell becomes c (1 + j eta) (exp(+j omega '
                                 't)), so a resonance has Q ~ 1 / eta; the geometric stiffness of stress_stiffening '
                                 'stays lossless (dissipation dilution: Q above 1 / eta). Default 0 (lossless); typical '
                                 '1e-4 (quartz, LN) to 1e-2 (metals, polymers higher).',
                       tip='Damping of the vibration as a fraction of the stiffness; the quality factor of a resonance '
                           'is about 1 divided by it. Used by the frequency response study and, on triangle or '
                           'tetrahedron meshes, by the eigenfrequency study.'),
    _material_property('dielectric_loss_tangent', 'Dielectric loss tangent', '1', 0.0,
                       help_text='Dielectric loss tangent tan delta (COMSOL: dielectric loss factor eta_epsilon), read '
                                 'only by the frequency_response study of a piezoelectric solid_mechanics (2D) or '
                                 'solid_mechanics_3d (tetrahedron meshes, every window element): the '
                                 'static permittivity becomes eps0 eps_dc (1 - j tan delta) (exp(+j omega t)), a '
                                 'conductance omega C tan delta in parallel with the electrode capacitance. Default 0; '
                                 'typical 1e-4 to 1e-3 for lithium niobate, ~1e-2 for PZT.',
                       tip='Electric loss of the material as the loss tangent. Used only by the frequency response '
                           'study with Piezoelectric on.'),
)


_INITIAL_STRESS_TEXT = (
    'Initial (residual) stress sigma0_{pair} [Pa] of the cell, e.g. the deposition stress of a film (tension > 0): '
    'the total stress is sigma0 + C eps - beta dT, so a stationary solve relaxes it (a released structure contracts '
    'until the total stress balances the supports); default 0. Triangle and tetrahedron meshes only (a rectangular-cell '
    'mesh refuses it). {where} With the stress_stiffening setting the eigenfrequency / frequency_response / '
    'optomechanics studies add the geometric stiffness K_sigma of the static state (a tensioned string or membrane '
    'rises in frequency). Constant per element (an x / y / z expression is evaluated at the centroids).')
_INITIAL_STRESS_WHERE = {
    'xx': 'In 3D a Voigt row (xx, yy, zz, yz, xz, xy); in 2D the in-plane row xx (axisymmetric: rr).',
    'yy': 'In 3D a Voigt row; in 2D the in-plane row yy (axisymmetric: zz, along the axis).',
    'xy': 'In 3D a Voigt row; in 2D the in-plane shear xy (axisymmetric: rz).',
    'zz': ('In 3D a Voigt row; in 2D plane strain the out-of-plane stress (stress_zz only, no in-plane effect), '
           'axisymmetric the hoop stress; plane stress (sigma_zz = 0) and generalized plane strain refuse it.'),
    'yz': 'solid_mechanics_3d only (a Voigt row).',
    'xz': 'solid_mechanics_3d only (a Voigt row).',
}
INITIAL_STRESS_PROPERTIES_3D = tuple(
    _material_property(name, f'Initial stress {name[-2:]}', 'Pa',
                       help_text=_INITIAL_STRESS_TEXT.format(pair=name[-2:], where=_INITIAL_STRESS_WHERE[name[-2:]]),
                       tip=f'Built-in stress {name[-2:]} of the material before any load, for example a tensile film '
                           'stress (positive). Triangle and tetrahedron meshes only.')
    for name in PRESTRESS_ARRAYS_3D)
INITIAL_STRESS_PROPERTIES = tuple(next(param for param in INITIAL_STRESS_PROPERTIES_3D if param.name == name)
                                  for name in ('initial_stress_xx', 'initial_stress_yy', 'initial_stress_xy',
                                               'initial_stress_zz'))
STRESS_STIFFENING_PARAM = Param(
    'stress_stiffening', 'bool', False, label='Stress stiffening',
    help='Triangle and tetrahedron meshes: the prestressed (linear perturbation) form of the eigenfrequency, '
         'frequency_response and optomechanics studies. On: the study first solves the static state (eigenfrequency: '
         'every load, prescribed displacement, initial_stress, thermal strain and coupled input, as a stationary solve; '
         'frequency_response: the initial_stress and thermal strain alone, held supports at 0, since the loads are its '
         'harmonic drive) and '
         'adds the geometric stiffness K_sigma = int grad N . sigma grad N dV of that total stress to K in every pencil '
         '(stress stiffening of strings, membranes, buckling-softening under compression); K_sigma carries no loss. A '
         'free body, a Bloch cell or an elastic PML cannot be solved statically: K_sigma then comes from the '
         'initial_stress alone (unrelaxed). Off (default): the modes and responses of the unstressed solid, loads and '
         'initial_stress play no part in the pencil. Not available with piezoelectric or on a rectangular-cell mesh.',
    tip='On: vibration frequencies include the stiffening by the static stress (from loads, built-in stress or '
        'heating), like a tensioned string. Used by the eigenfrequency and frequency response studies.')


_NONLINEAR_VISIBLE = (('geometric_nonlinearity', (True,)),)
GEOMETRIC_NONLINEARITY_PARAMS = (   # new_modules_plan K6 (solid_mechanics_3d on tetrahedra, solid_mechanics on triangles)
    Param('geometric_nonlinearity', 'bool', False, label='Geometric nonlinearity',
          help='Triangle and tetrahedron meshes: large deflections and rotations in every static solve (stationary and '
               'time_dependent studies, the static state of the eigenfrequency / frequency_response studies). '
               'Total-Lagrangian form on the reference geometry: Green-Lagrange strain E = (H + H^T + H^T H) / 2 (H the '
               'displacement gradient), second Piola-Kirchhoff stress S = C E + sigma0 - beta dT (Saint Venant-Kirchhoff '
               'material: valid for small strains with large rotations), solved by Newton iterations with the tangent K_T '
               '= int (B + B_NL)^T C (B + B_NL) dV + K_sigma(S), refactorised every iteration, damped (the step halves '
               'while the change grows) and over load_steps equal load increments. Boundary loads and body forces are '
               'dead loads (fixed direction and reference area: nominal tractions); load steps scale them, prescribed '
               'displacements and coupled body forces, while the initial stress and the thermal strain enter at full '
               'strength in every step. Outputs: strain_* are Green-Lagrange strains, stress_* the second Piola-Kirchhoff '
               'stress (equal to the Cauchy stress to first order in the strain); support reactions are the nominal '
               'forces of the internal force int (B + B_NL)^T S dV - f at the held dofs (not K u - f). The '
               'eigenfrequency study linearises about the nonlinear static state (its pencil stiffness is K_T, stress '
               'stiffening included, whether or not the stress_stiffening setting is on); frequency_response about the '
               'nonlinear state of the initial stress alone (holds at 0). A Bloch cell, an elastic PML or a free body '
               'has no static solve: their pencils take K + K_sigma of the initial stress. Not available with '
               'piezoelectric or on a rectangular-cell mesh, and on a triangle mesh not with azimuthal_order >= 1, an '
               'elastic PML or generalized plane strain. Cost: one tangent assembly and factorisation per Newton '
               'iteration while the residual falls slowly (typically 3 to 6 iterations per load step; tangent_update '
               'when_slow, the default, keeps the factor while the residual falls fast).',
          tip='On: large deflections and rotations are followed exactly, for example the stiffening of a string or a '
              'membrane under a large transverse load, or a released film that bends.\n\nThe static solve is repeated '
              'a few times (Newton iterations) and costs several times a linear solve. Triangle and tetrahedron meshes only.'),
    Param('load_steps', 'int', 1, label='Load steps', min=1, visible_if=_NONLINEAR_VISIBLE,
          help='Geometric nonlinearity: the applied loads and prescribed displacements are ramped in this many equal '
               'steps, each converged by Newton iterations from the previous one (raise it when a large load does not '
               'converge; the initial stress and thermal strain are not ramped). Default 1.',
          tip='Number of steps the loads are applied in; more steps help a strongly nonlinear solve converge.'),
    Param('nonlinear_tolerance', 'float', 1e-8, label='Nonlinear tolerance', unit='1', min=0.0, min_exclusive=True,
          visible_if=_NONLINEAR_VISIBLE,
          help='Geometric nonlinearity: a load step has converged when the largest Newton correction is at most this '
               'fraction of the largest displacement (quadratic convergence: the error after the last step is about its '
               'square). Default 1e-8.',
          tip='Accuracy of the nonlinear static solve: smaller is more accurate and takes an iteration more.'),
    Param('nonlinear_max_iterations', 'int', 25, label='Nonlinear iteration limit', min=1,
          visible_if=_NONLINEAR_VISIBLE,
          help='Geometric nonlinearity: Newton iterations per load step before it stops unconverged (a warning; the '
               'damped state is kept and the next step continues from it). Default 25.',
          tip='Most Newton iterations per load step.'),
    Param('tangent_update', 'choice', 'when_slow', label='Tangent update', choices=TANGENT_UPDATES,
          choice_labels=(('every_iteration', 'Every iteration (full Newton)'),
                         ('when_slow', 'Only when convergence slows (modified Newton)')),
          visible_if=_NONLINEAR_VISIBLE,
          help='Geometric nonlinearity: when the tangent K_T is reassembled and refactorised. every_iteration: full '
               'Newton, one factorisation per iteration, quadratic convergence. when_slow (default): modified '
               'Newton, the factorised tangent is kept (only the internal force is assembled, one back substitution) '
               f'while each iteration cuts the largest free residual below {SLOW_RESIDUAL_RATIO:g} x the previous one, '
               'and refactorised at the current state otherwise; the first iteration of a load step keeps the factor of '
               'the previous step when that step ended fast. Saves factorisations when they dominate an iteration (large '
               '3D meshes), at the price of linear convergence near the end (a few more, cheaper iterations; the '
               'remaining error is at most about a third of the nonlinear_tolerance correction). Work counters '
               'tangent_factorizations / tangent_reuses.',
          tip='How often the Newton stiffness matrix is rebuilt and factorised. Rebuilding only when convergence slows '
              'saves time on large models; every iteration is the most robust choice.'),
)


def displacement_param(name, axis):
    """Optional prescribed displacement component Param of the displacement feature."""
    return Param(name, 'float', None, label=f'Displacement {axis}', unit='m', optional=True,
                 help=f'Prescribed u_{axis} on the selected faces; empty (null) leaves u_{axis} free.',
                 tip=f'Displacement along {axis} the faces are moved to; empty leaves this direction free.')


# ---- runtime helpers (worker thread) ----
def union_masks(masks):
    """Elementwise OR of a non-empty list of masks."""
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined | mask
    return combined


# ---- validation helpers (CPU only) ----
def plain_value(value, parameters):
    """Float of a value that evaluates over the parameters; None when unset or field / state dependent."""
    if value is None:
        return None
    try:
        return evaluate_scalar(value, parameters)
    except ExpressionError:
        return None


def _stiff(properties, parameters, tensor):
    """True when a merged material evaluates to a youngs_modulus > 0 (tensor: or a c11 > 0, the 2D c_ij entries)."""
    return (evaluates_to(properties.get('youngs_modulus'), parameters, lambda value: value > 0.0)
            or (tensor and evaluates_to(properties.get('c11'), parameters, lambda value: value > 0.0)))


def merged_materials(vctx) -> list:
    """[(material, merged properties or None)] of every spec material as the builder merges it (library, crystal
    entry; None when merging fails: reported at the material), kept on the ValidationContext so the validate helpers
    merge once per validation."""
    kept = getattr(vctx, '_merged_mechanics_materials', None)
    if kept is None or kept[0] is not vctx.spec.materials:
        from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
        merged = []
        for material in vctx.spec.materials:
            try:
                properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
            except (ValueError, TypeError):
                properties = None
            merged.append((material, properties))
        kept = (vctx.spec.materials, merged)
        vctx._merged_mechanics_materials = kept   # a plain attribute of this context (not a dataclass field)
    return kept[1]


def materials_setting(vctx, names) -> list:
    """Names of the selected spec materials (merged) that set any of the property names (e.g. the c_ij entries)."""
    return [material.name for material, properties in merged_materials(vctx)
            if properties is not None and material.selection
            and any(properties.get(name) is not None for name in names)]


def material_issues(vctx, physics_spec, path, type_name, element_name, tensor=False):
    """Stiffness checks over the spec materials, reported by the first physics of type_name: no youngs_modulus (tensor:
    or c11) anywhere (error), poisson_ratio unset (info) or above the locking limit (warning naming element_name, e.g.
    'bilinear')."""
    if not reports_model_checks(vctx, physics_spec, type_name):
        return []
    parameters = vctx.parameters
    issues, stiff = [], []
    selected_stiff = False
    for material, properties in merged_materials(vctx):
        if properties is None:   # reported at the material: may be stiff
            stiff.append(True)
            selected_stiff = selected_stiff or bool(material.selection)
            continue
        has_stiffness = _stiff(properties, parameters, tensor)
        stiff.append(has_stiffness)
        if not (material.selection and has_stiffness):
            continue
        selected_stiff = True
        material_path = ('material', material.name)
        poisson_ratio = plain_value(properties.get('poisson_ratio'), parameters)
        if properties.get('youngs_modulus') is None:
            continue   # c_ij entries only (tensor)
        if properties.get('poisson_ratio') is None:
            issues.append(Issue('info', material_path,
                                f"material '{material.name}' sets youngs_modulus but no poisson_ratio: 0 is used"))
        elif poisson_ratio is not None and poisson_ratio > LOCKING_POISSON_RATIO:
            issues.append(Issue('warning', material_path,
                                f"material '{material.name}' has poisson_ratio {poisson_ratio:g} > "
                                f'{LOCKING_POISSON_RATIO:g}: {element_name} elements lock (grow too stiff) toward 0.5, '
                                'so displacements come out too small'))
    if conducting_fraction(vctx, stiff) == 0.0 and not selected_stiff:
        stiffness_text = 'a youngs_modulus or c11 > 0' if tensor else 'a youngs_modulus'
        issues.append(Issue('error', path, f'no physical cell has {stiffness_text}: assign a material with stiffness '
                                           '(e.g. a library solid such as silica) to the solid'))
    return issues


def modal_study_issues(vctx, physics_spec, path, modal_studies, thermal_strain_coupling, tensor=False,
                       prestress_info=True, stress_stiffening=False):
    """In modal_studies only: error per selected stiff material (tensor: c11 counts) without a density rho > 0 (plain
    numbers; the solve checks every cell), info (prestress_info, stress_stiffening off) when loads, initial stresses or
    a thermal strain coupling are present (no prestress)."""
    if vctx.study_type not in modal_studies:
        return []
    issues = []
    stiffness_names = 'youngs_modulus or c11' if tensor else 'youngs_modulus'
    for material, properties in merged_materials(vctx):
        if properties is None or not (material.selection and _stiff(properties, vctx.parameters, tensor)):
            continue
        density = properties.get('rho')
        plain_density = plain_value(density, vctx.parameters)   # None when unset or field / state dependent
        if density is None or (plain_density is not None and plain_density <= 0.0):
            issues.append(Issue('error', ('material', material.name),
                                f"material '{material.name}' has stiffness ({stiffness_names}) but no density rho > "
                                f'0: the {vctx.study_type} study of {physics_spec.name} needs the mass of every solid '
                                'cell'))
    if not prestress_info or stress_stiffening:
        return issues
    loads = sorted(feature.name for feature in physics_spec.features
                   if feature.enabled and feature.type in LOAD_FEATURES)
    thermal = sorted(coupling.name for coupling in vctx.spec.couplings
                     if coupling.enabled and coupling.type == thermal_strain_coupling
                     and coupling.target == physics_spec.name)
    stressed = [f'initial stress of {name}' for name in initial_stress_materials(vctx)]
    if loads or thermal or stressed:
        issues.append(Issue('info', path,
                            f"{', '.join(loads + thermal + stressed)} play(s) no part in the {vctx.study_type} study: "
                            'the modes are those of the unloaded, unstressed solid (no stress stiffening), with every '
                            'support held at 0; set stress_stiffening (triangle / tetrahedron meshes) to prestress '
                            'them'))
    return issues


def initial_stress_materials(vctx):
    """Names of the selected spec materials (library merged) that set an initial_stress_* entry."""
    return materials_setting(vctx, PRESTRESS_ARRAYS_3D)


def prestress_issues(vctx, physics_spec, settings_values, path, fem, refused_zz=None, static_refused=''):
    """initial_stress materials and the stress_stiffening setting: errors on a rectangular-cell mesh (fem False), with
    piezoelectric, for a nonzero initial_stress_zz under refused_zz (a 2D condition text); infos for the setting outside
    PRESTRESS_STUDIES and, in a modal study, when static_refused (why no static solve: K_sigma of sigma0 alone)."""
    stiffening = bool(settings_values.get('stress_stiffening'))
    stressed = initial_stress_materials(vctx)
    issues = []
    if not fem:
        issues.extend(Issue('error', ('material', name),
                            f"material '{name}' sets an initial_stress entry: {physics_spec.name} reads it only on a "
                            "triangle or tetrahedron mesh (MeshSpec kind 'triangles' / 'tetrahedra')")
                      for name in stressed)
        if stiffening:
            issues.append(Issue('error', path, 'stress_stiffening needs a triangle or tetrahedron mesh (the '
                                               'rectangular-cell solver has no geometric stiffness)'))
        return issues
    if stiffening and settings_values.get('piezoelectric'):
        issues.append(Issue('error', path, 'stress_stiffening is not available with piezoelectric on (the coupled '
                                           'pencil has no geometric stiffness yet)'))
    if refused_zz is not None:
        for material, properties in merged_materials(vctx):
            if properties is not None and material.selection and evaluates_to(
                    properties.get('initial_stress_zz'), vctx.parameters, lambda value: value != 0.0):
                issues.append(Issue('error', ('material', material.name),
                                    f"material '{material.name}' sets initial_stress_zz != 0, which {refused_zz} "
                                    'cannot hold: leave it out'))
    if stiffening and vctx.study_type not in PRESTRESS_STUDIES:
        issues.append(Issue('info', path, f"stress_stiffening is read by the {', '.join(PRESTRESS_STUDIES)} studies "
                                          f'only: the {vctx.study_type} study solves the static state anyway'))
    elif stiffening and static_refused and vctx.study_type == EIGENFREQUENCY_STUDY:
        issues.append(Issue('info', path, f'stress_stiffening: {static_refused} has no static solve, so K_sigma comes '
                                          'from the initial_stress alone (unrelaxed; loads and thermal strain play no '
                                          'part)'))
    elif stiffening and not stressed and not any(feature.enabled and feature.type in LOAD_FEATURES
                                                 for feature in physics_spec.features):
        issues.append(Issue('info', path, 'stress_stiffening is on but no load feature or initial_stress material '
                                          'stresses the solid (a coupled input may still do so)'))
    return issues


def _study_values(context) -> dict:
    """Resolved study settings of the context's model ({} when they do not resolve: validate reports)."""
    study = context.spec.study
    try:
        return resolve_values(context.registry.study_type(study.type).params,
                              study.settings if isinstance(study.settings, dict) else {}, context.parameters)
    except (KeyError, ValueError, TypeError):   # reported by validate
        return {}


def modal_resonance_count(context) -> int:
    """Modes the study asks of the solid per shift (n_resonances, or optomechanics' n_mechanical_modes; 1 when they do
    not resolve: validate reports), for the memory estimators."""
    study_values = _study_values(context)
    return int(study_values.get('n_resonances') or study_values.get('n_mechanical_modes') or 1)


def modal_kept_count(context) -> int:
    """Mode shapes the solve keeps (memory estimators): modal_resonance_count, x the targets of search 'targets',
    max_resonances of search 'band'."""
    study_values = _study_values(context)
    count = int(study_values.get('n_resonances') or study_values.get('n_mechanical_modes') or 1)
    search = study_values.get('search', 'nearest') if context.spec.study.type == EIGENFREQUENCY_STUDY else 'nearest'
    if search == 'targets':
        return count * max(len(study_values.get('target_frequencies') or ()), 1)
    if search == 'band':
        return int(study_values.get('max_resonances') or count)
    return count


def eigen_search_issues(vctx, path, piezoelectric, tensor_mesh_3d=False):
    """Errors for an eigenfrequency search 'targets' / 'band' the solid cannot serve: a 3D tensor mesh (lowest modes
    only), 'band' of a piezoelectric solid (indefinite coupled pencil: no inertia count)."""
    search = eigen_search(vctx.spec)
    if vctx.study_type != EIGENFREQUENCY_STUDY or search == 'nearest':
        return []
    if tensor_mesh_3d:
        return [Issue('error', path, f"eigenfrequency search '{search}' needs a tetrahedron mesh: on a tensor mesh "
                                     'solid_mechanics_3d finds the lowest modes only (matrix-free, no shift); use search '
                                     "'nearest' or a tetrahedra mesh")]
    if search == 'band' and piezoelectric:
        return [Issue('error', path, "eigenfrequency search 'band' is not available for a piezoelectric solid (its "
                                     'coupled pencil is indefinite, so the band cannot be counted): use search '
                                     "'targets'")]
    return []


# ---- resonance_table feature (both types; reads the solver's last modal.ModalSolve) ----
def _apply_resonance_table(ctx, solver, feature_spec, values, selection):
    return ModalTable()


def _resonance_table_result(session, solver, table, feature_spec, options):
    return modal_table_result(table, solver)


def _validate_resonance_table(vctx, physics_spec, feature_spec, values):
    """Warning outside the studies running solve_eigenfrequencies: the table has no solve to show."""
    if vctx.study_type in (EIGENFREQUENCY_STUDY, OPTOMECHANICS_STUDY):
        return []
    return [Issue('warning', feature_path(physics_spec, feature_spec),
                  f"resonance_table is filled only by an {EIGENFREQUENCY_STUDY} or {OPTOMECHANICS_STUDY} study (this "
                  f"model runs {vctx.study_type}): its result will be missing")]


def resonance_table_feature(order_text):
    """The resonance_table FeatureType; order_text says which modes come first (the type's search)."""
    return FeatureType(
        'resonance_table', 'Resonance table', 'monitor', None, result_kind='table',
        description=(f'Vibration modes of the last eigenfrequency study solve, {order_text}. Result \'table\', one row '
                     f"per mode: {', '.join(MODAL_TABLE_COLUMNS)} (1-based rank, Hz, rad/s); extras target_frequency "
                     '(Hz) and selected_resonance. Undamped: every frequency is real (Q infinite), except with a '
                     'resistive_load electrode (solid_mechanics_3d), which appends frequency_imag (Hz, > 0 decays), '
                     'q_factor and energy_decay_rate (1/s, = 2 Im omega, the external coupling rate kappa_e). For a '
                     'frequency '
                     "versus a parameter add a derived value with source '<physics>.<this feature>', expression "
                     'frequency, operation at and at = 0 (the first mode).'),
        summary="Lists the vibration frequencies found by an 'Eigenfrequency' study.",
        apply=_apply_resonance_table, result=_resonance_table_result, validate=_validate_resonance_table)


# ---- frequency_probe feature and complex reactions (new_modules_plan K4; triangle / tetrahedron meshes) ----
def _frequency_probe_result(session, solver, record, feature_spec, options):
    from ...core.harmonic_probe import harmonic_probe_result   # lazy: core.harmonic_probe imports core.backend (cupy)
    return harmonic_probe_result(record)


def _validate_frequency_probe(vctx, physics_spec, feature_spec, values):
    """The point inside the domain; a warning outside the frequency_response study (no frequency is recorded)."""
    from ...model.definition_helpers import validate_points_inside
    issues = list(validate_points_inside(vctx, physics_spec, feature_spec, values))
    if vctx.study_type != FREQUENCY_RESPONSE_STUDY_NAME:
        issues.append(Issue('warning', feature_path(physics_spec, feature_spec),
                            f'frequency_probe is filled only by a {FREQUENCY_RESPONSE_STUDY_NAME} study (this model '
                            f'runs {vctx.study_type}): its result will be missing'))
    return issues


def frequency_probe_feature(apply, components, grids):
    """The frequency_probe FeatureType of the displacement components (u_x, u_y[, u_z]) on the mesh kinds grids."""
    series = ', '.join(f'{component}_re / _im / _abs / _phase' for component in components)
    return FeatureType(
        'frequency_probe', 'Frequency probe', 'probe', 'point', object_kinds=('point',), max_entities=1,
        result_kind='xy',
        summary='Records the vibration amplitude and phase at a point for every frequency of a frequency response.',
        description=('frequency_response study only (triangle and tetrahedron meshes): the complex displacement '
                     '(exp(+j omega t), phase 0 = the drive) at the point per solved frequency, the nodal field '
                     'evaluated there (triangles: the mesher puts a node on the point; 0 off the solid). Result \'xy\' '
                     f'versus frequency [Hz] (core.harmonic_probe series): {series} (m; phase in rad, in (-pi, pi]); a '
                     'derived value two_wave_fit pairs a _re series with its _im sibling. The record restarts with each '
                     'frequency_response run.'),
        apply=apply, result=_frequency_probe_result, validate=_validate_frequency_probe, grids=grids)


def register_support(solver, region):
    """The support region, registered with a solver that records complex reactions per frequency (FEM meshes)."""
    register = getattr(solver, 'register_support', None)
    return region if register is None else register(region)


def harmonic_support_result(solver, region):
    """The complex reaction table of a support while a frequency response is shown (FEM meshes), else None."""
    if getattr(solver, '_shows_harmonic', False) and hasattr(solver, 'harmonic_reaction_table'):
        return solver.harmonic_reaction_table(region)
    return None


def stiffness_follows_state(context):
    """True when a spec material's youngs_modulus or poisson_ratio references a physics state field or time t (the
    operator is re-assembled during the run)."""
    state_names = frozenset(state_field_names(context.spec, context.registry)) | TIME_NAMES
    for material in context.spec.materials:
        properties = material.properties if isinstance(material.properties, dict) else {}
        for name in STIFFNESS_ARRAYS:
            value = properties.get(name)
            if not isinstance(value, str):
                continue
            try:
                if not referenced_names(value).isdisjoint(state_names):
                    return True
            except ExpressionError:   # reported by validate
                continue
    return False


# ---- reused sweep builds (SweepSpec.reuse_build; model.sweep_reuse) ----
FOLLOWING_LOAD_NAMES = frozenset({'f_x', 'f_y', 'f_z', 't_x', 't_y', 't_z', 'u_x', 'u_y', 'u_z'})


def following(params, names=FOLLOWING_LOAD_NAMES):
    """params with follows_parameters set on those named in names (the load / held values a kept sweep build re-sets:
    their apply wires follow_load / follow_held_values on every grid kind)."""
    return tuple(replace(param, follows_parameters=True) if param.name in names else param for param in params)


def follow_load(ctx, solver, feature_spec, values, names, add_change):
    """Sweep follower of the load values `names`: add_change(*changes) adds each value's change since the last set
    (the apply added the build values itself); nothing without a reused sweep build."""
    applied = [values[name] for name in names]

    def set_loads(*new):   # a value passed unchanged (the same object: an unfollowed x / y field) adds nothing
        add_change(*(0.0 if value is before else value - before for value, before in zip(new, applied)))
        applied[:] = new
    ctx.follow_parameters(feature_spec.params, values, names, set_loads, solver, f"feature '{feature_spec.name}'")


def follow_held_values(ctx, solver, feature_spec, values, names, dof_mask):
    """Sweep follower of the prescribed displacements `names` (None = free) held on the hold's dof_mask: the changed
    held values are written in place (solver.set_held_values), the held set and the assembly stay."""
    applied = [values[name] for name in names]

    def set_held(*new):   # unchanged components pass None (a later overlapping hold keeps its value)
        solver.set_held_values(dof_mask, *(None if value == before else value for value, before in zip(new, applied)))
        applied[:] = new
    ctx.follow_parameters(feature_spec.params, values, names, set_held, solver, f"feature '{feature_spec.name}'")
