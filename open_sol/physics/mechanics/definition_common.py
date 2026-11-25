"""Model-layer pieces shared by the solid_mechanics (2D) and solid_mechanics_3d definitions: the build's linear
solver, the material vocabulary (one Param object per name), support helpers, the CPU material / state / modal-study /
prestress checks, the study a solid takes (StudySpec.solved_physics), the symmetry (mirror plane) rules and the
resonance_table / frequency_probe / thermoelastic_damping features."""
from dataclasses import replace

from ...core.memory_estimate import MemoryEstimate, host_real_lu_bytes
from ...core.mirror import mirror_faces
from ...model.definition_helpers import (conducting_fraction, eigen_search, enabled_feature_entities, evaluates_to,
                                         material_follows_state, mirror_feature_issues, optional_material_param,
                                         reports_model_checks)
from ...model.expressions import ExpressionError, evaluate_scalar
from ...model.geometry import expand_entities
from ...model.schema import (EIGENFREQUENCY_STUDY, FREE_MESH_KINDS, FREQUENCY_RESPONSE_STUDY, OPTOMECHANICS_STUDY,
                             STATIONARY_STUDY, FeatureType, Issue, Param, feature_path, resolve_values)
from ...solvers.pcg import resolve_max_iterations
from .material_params import (PHOTOELASTIC_P11_PARAM, PHOTOELASTIC_P12_PARAM, POISSON_RATIO_PARAM,
                              YOUNGS_MODULUS_PARAM)
from .materials import (MECHANICAL_PROPERTY_ARRAYS, MODULUS_ARRAYS, PERMITTIVITY_TENSOR_ARRAYS_3D,
                        PIEZO_TENSOR_ARRAYS_3D, PRESTRESS_ARRAYS_3D, STIFFNESS_ARRAYS, STIFFNESS_TENSOR_ARRAYS_3D)
from .modal import DAMPED_TABLE_COLUMNS, MODAL_TABLE_COLUMNS, ModalTable, modal_table_result
from .tangent_update import SLOW_RESIDUAL_RATIO, TANGENT_UPDATES
from .viscoelastic import DISPLACEMENT, FORMULATIONS, MIXED

LOCKING_POISSON_RATIO = 0.45   # bilinear / trilinear elements grow too stiff (volumetric locking) above about this
# core.mirror.MIRROR_FEATURES of both solid types: a Roller (the even mirror of the displacement) whose outer faces are
# model mirror planes (unfolded fields, integrals x 2, max / min over the images); a support like the Roller
SYMMETRY_FEATURE = 'symmetry'
ROLLER_FEATURES = ('roller', SYMMETRY_FEATURE)   # hold the normal displacement of their outer faces (the Roller apply)
SUPPORT_FEATURES = frozenset({'fixed', 'roller', SYMMETRY_FEATURE, 'displacement', 'fixed_domain',
                              'prescribed_displacement_domain'})
LOAD_FEATURES = frozenset({'boundary_load', 'body_force', 'temperature_change'})
BACKGROUND_STUDY = STATIONARY_STUDY       # the background study of the modal / response studies
PRESTRESS_STUDIES = (EIGENFREQUENCY_STUDY, FREQUENCY_RESPONSE_STUDY, OPTOMECHANICS_STUDY)   # read stress_stiffening
MODAL_STUDIES = (EIGENFREQUENCY_STUDY, OPTOMECHANICS_STUDY)   # studies running solve_eigenfrequencies


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
    """The build's 'linear' solver of the physics (pcg: max_iterations 0 (auto) = twice the DiffusionSystem limit;
    sparse_direct: its SparseDirectSolver, wrapped by the 2D ElasticSystem)."""
    built = ctx.solver_records[physics_spec.name]['linear']
    if built.resolved.values.get('max_iterations', 1) == 0:
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
                       help_text='Overrides alpha for this axis.',
                       tip=_ALPHA_COMPONENT_TIP.format(axis='x')),
    _material_property('alpha_yy', 'Thermal expansion along y', '1/K',
                       help_text='Overrides alpha for this axis.',
                       tip=_ALPHA_COMPONENT_TIP.format(axis='y')),
    _material_property('alpha_zz', 'Thermal expansion along z', '1/K',
                       help_text='Overrides alpha for this axis (solid types only). 2D plane strain turns alpha_zz dT '
                                 'into out-of-plane stress (and, by Poisson, in-plane strain); plane stress: a free z '
                                 'expansion; generalized plane strain: part of the eps_zz balance; axisymmetric: the '
                                 'hoop expansion. 3D: an ordinary z expansion.',
                       tip=_ALPHA_COMPONENT_TIP.format(axis='z')),
    PHOTOELASTIC_P11_PARAM,
    PHOTOELASTIC_P12_PARAM,
)
# Both solid types (read by their eigenfrequency study only); the same name, unit and bounds as heat's / elastic_waves'.
DENSITY_PARAM = _material_property(
    'rho', 'Density', 'kg/m^3', 0.0,
    help_text='Mass density (consistent mass), read by the solid types only in the eigenfrequency, optomechanics and '
              'frequency_response studies, where every solid or fluid cell of the solid (youngs_modulus, c11, '
              'shear_modulus or bulk_modulus > 0; with formulation mixed also a liquid with a viscosity) needs rho > 0. '
              'Typical: silica 2200, silicon 2329, gold 19300.',
    tip="Mass density. Solid mechanics uses it only in an 'Eigenfrequency' study.\n\nTypical: silica 2200, silicon "
        '2329, gold 19300.')
# 2D solid_mechanics only: read by the optomechanical_coupling overlap alone (the strain-optic couplings are diagonal).
PHOTOELASTIC_P44_PARAM = _material_property(
    'photoelastic_p44', 'Photoelastic coefficient p44', '1',
    help_text='Shear strain-optic coefficient: Delta(1/n^2)_xy = p44 x 2 strain_xy (Voigt; cubic axes along x, y, '
              'z). Read only by optomechanical_coupling (the strain-optic couplings have no shear term). Unset = the '
              'isotropic (p11 - p12) / 2 per cell. Typical: silicon -0.051, silica -0.075.',
    tip='How strongly shearing the material changes its refractive index; only the optomechanical coupling uses '
        'it.\n\nEmpty takes the isotropic value from the other two coefficients. Typical: silicon -0.051.')
# 2D solid_mechanics with piezoelectric on (kind, unit, bounds and flags as elastic_waves' piezo_e* and electrostatics'
# eps_dc*: the registry rule; the arrays are shared).
_VOIGT_PAIRS = {'1': 'xx', '2': 'yy', '3': 'zz', '4': 'yz', '5': 'xz', '6': 'xy'}


def _piezo_help(name) -> str:
    """Help of piezo_e<kJ>: the law and the read rules once (e11, both types share it), the index meaning elsewhere."""
    if name == 'piezo_e11':
        return ('Piezoelectric stress constant e_11 in model axes: T_J = c_JK S_K - e_kJ E_k, D_k = e_kJ S_J + eps0 '
                'eps_dc_kl E_l (row k = 1 x, 2 y, 3 z; Voigt column J = 1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy, '
                'engineering shear S_6 = 2 strain_xy). Unset = 0. Read only with setting piezoelectric on, in solid '
                'cells (youngs_modulus or c11 > 0): solid_mechanics the in-plane block (k = 1, 2; J = 1, 2, 6; '
                "elastic_waves' arrays) and e13 / e23, solid_mechanics_3d (tetrahedra) all 18 (any entry allocates "
                "all). A {'model': 'crystal'} entry sets them (3D: 'plane': 'full').")
    row, column = name[-2], name[-1]
    text = (f"Piezoelectric stress constant e_{row}{column}: row {'xyz'[int(row) - 1]}, Voigt column "
            f'{_VOIGT_PAIRS[column]}; rules: piezo_e11.')
    if column == '3' and row in '12':
        text += f" 2D plane strain (S_zz = 0): it enters only stress_zz -= e_{row}3 E_{'xy'[int(row) - 1]}."
    return text


_EPS_DC_CROSS_HELP = ('Off-diagonal static permittivity of a rotated material; the eps_dc tensor must stay positive '
                      'definite (2D: eps_dc_xy^2 < eps_dc_xx eps_dc_yy), else a build error. Unset = 0.')
PIEZO_MATERIAL_PROPERTIES = (
    *(_material_property(
        name, f'Piezoelectric e{name[-2:]}', 'C/m^2',
        help_text=_piezo_help(name),
        tip=f'Piezoelectric constant e{name[-2:]}: how strongly an electric field along '
            f"{'x' if name[-2] == '1' else 'y'} stresses the material. Used only when Piezoelectric is on.")
      for name in ('piezo_e11', 'piezo_e12', 'piezo_e16', 'piezo_e21', 'piezo_e22', 'piezo_e26')),
    *(_material_property(
        name, f'Piezoelectric e{name[-2:]}', 'C/m^2',
        help_text=_piezo_help(name),
        tip=f"How strongly an electric field along {'x' if name[-2] == '1' else 'y'} stresses the material along the "
            'depth of the drawing. Used only when Piezoelectric is on; usually filled in by a crystal entry.')
      for name in ('piezo_e13', 'piezo_e23')),
    _material_property('eps_dc', 'Static relative permittivity', '1', 1.0,
                       help_text='Isotropic static relative permittivity: sets eps_dc_xx and eps_dc_yy (components '
                                 'override); unset = 1. Same arrays as electrostatics / drift_diffusion; read only with '
                                 'piezoelectric on, as the clamped permittivity eps^S of every window cell (void '
                                 'included). Not the optical eps_r: lithium niobate eps^S 44 / 29 (along the optic '
                                 'axis), silica 3.9.',
                       tip='Relative permittivity for static fields; empty means vacuum, 1. Used by solid mechanics '
                           'only when Piezoelectric is on.'),
    _material_property('eps_dc_xx', 'Static relative permittivity along x', '1', 1.0,
                       help_text='Overrides eps_dc for this axis.',
                       tip="Static relative permittivity along x. Replaces 'Static relative permittivity' for this "
                           'direction.'),
    _material_property('eps_dc_yy', 'Static relative permittivity along y', '1', 1.0,
                       help_text='Overrides eps_dc for this axis.',
                       tip="Static relative permittivity along y. Replaces 'Static relative permittivity' for this "
                           'direction.'),
    _material_property('eps_dc_xy', 'Static relative permittivity, cross term', '1',
                       help_text=_EPS_DC_CROSS_HELP,
                       tip='Cross term of a rotated material: the field also polarises it sideways. Usually empty.'),
)
# solid_mechanics_3d on tetrahedra with piezoelectric on: the full e_iJ and the six permittivities (the 2D Param objects
# for the shared names; its own eps_dc expands to xx, yy and zz).
_PIEZO_2D_PARAMS = {param.name: param for param in PIEZO_MATERIAL_PROPERTIES}
_PIEZO_3D_ONLY = {
    name: _material_property(name, f'Piezoelectric e{name[-2:]}', 'C/m^2',
                             help_text=_piezo_help(name),
                             tip=f"Piezoelectric constant e{name[-2:]}: how strongly an electric field along "
                                 f"{'xyz'[int(name[-2]) - 1]} stresses the material. Used only when Piezoelectric is on; "
                                 'usually filled in by a crystal entry.')
    for name in PIEZO_TENSOR_ARRAYS_3D if name not in _PIEZO_2D_PARAMS}
_PERMITTIVITY_3D_ONLY = {
    'eps_dc': optional_material_param(
        'eps_dc', 'Static relative permittivity', '1', ('eps_dc_xx', 'eps_dc_yy', 'eps_dc_zz'), 1.0,
        help_text='Isotropic static relative permittivity: sets eps_dc_xx, eps_dc_yy and eps_dc_zz (components '
                  'override); unset = 1. Read only with piezoelectric on (tetrahedra), as the clamped permittivity eps^S '
                  'of every window element (void included).',
        tip='Relative permittivity for static fields; empty means vacuum, 1. Used by solid mechanics only when '
            'Piezoelectric is on.'),
    'eps_dc_zz': _material_property('eps_dc_zz', 'Static relative permittivity along z', '1', 1.0,
                                    help_text='Overrides eps_dc for this axis.',
                                    tip="Static relative permittivity along z. Replaces 'Static relative permittivity' "
                                        'for this direction.'),
    **{name: _material_property(name, f'Static relative permittivity, cross term {name[-2:]}', '1',
                                help_text=_EPS_DC_CROSS_HELP,
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
_TENSOR_TEXT = 'Stiffness entry {name} ({role}) of the Voigt matrix in model axes; unset = {fallback}. Rules: c11.'
# c11 (shared by both types) carries the law and the read rules once
_TENSOR_TEXT_C11 = (
    'Anisotropic stiffness entry c11 (xx-xx) of the Voigt matrix in model axes (1 xx, 2 yy, 3 zz, 4 yz, 5 xz, 6 xy; '
    'engineering shear S_6 = 2 strain_xy): T_I = c_IJ S_J - beta_I dT, beta_I = c_IJ alpha_J. Unset (NaN) = the '
    'isotropic value of youngs_modulus / poisson_ratio (c11, c22, c33: lambda + 2 mu; c12, c13, c23: lambda; c44, c55, '
    'c66: mu; others 0). A cell is solid when c11 > 0; the entries its formulation reads must be positive definite '
    "(build error). Usually set by a {'model': 'crystal'} entry (3D: 'plane': 'full'), with piezo_e* and eps_dc*. "
    'solid_mechanics reads the ten 2D entries: plane strain the full law with S_zz = S_xz = S_yz = 0 (stress_zz = '
    'T_3), plane stress condenses T_3 = 0, generalized plane strain takes a uniform S_3, axisymmetric reads 3 as hoop '
    '(orthotropic only: c16 = c26 = c36 = 0, else a build error); a c_ij-only cell needs c11, c12, c22, c66 (plane '
    'stress / axisymmetric / generalized: also c13, c23, c33); any entry allocates the eight besides c16 / c26, which '
    'only c16 or c26 allocates (as in elastic_waves); guided modes (axial_wavenumber) refuse c_ij. solid_mechanics_3d '
    'reads all 21 on tetrahedra only (a tensor mesh refuses them); any entry allocates all 21; a c_ij-only cell needs '
    'the nine orthotropic ones.')
# solid_mechanics_3d on tetrahedra only: the entries with a yz (4) or xz (5) index
_TENSOR_ROLES_3D = {'c14': 'xx-yz', 'c15': 'xx-xz', 'c24': 'yy-yz', 'c25': 'yy-xz', 'c34': 'zz-yz', 'c35': 'zz-xz',
                    'c44': 'yz-yz', 'c45': 'yz-xz', 'c46': 'yz-xy', 'c55': 'xz-xz', 'c56': 'xz-xy'}
_STIFFNESS_TIP = ("Stiffness {name} of a crystal (an anisotropic material); empty takes the value from the Young's "
                  'modulus and Poisson ratio. Usually filled in by a crystal entry.')
STIFFNESS_TENSOR_PROPERTIES = tuple(
    _material_property(name, f'Stiffness {name}', 'Pa', 0.0 if name in ('c11', 'c22', 'c33', 'c66') else None,
                       help_text=_TENSOR_TEXT_C11 if name == 'c11' else _TENSOR_TEXT.format(
                           name=name, role=_TENSOR_ROLES[name], fallback=_TENSOR_FALLBACKS[name]),
                       tip=_STIFFNESS_TIP.format(name=name))
    for name in ('c11', 'c12', 'c13', 'c22', 'c23', 'c33', 'c16', 'c26', 'c36', 'c66'))
_STIFFNESS_3D_ONLY = {
    name: _material_property(name, f'Stiffness {name}', 'Pa', 0.0 if name in ('c44', 'c55') else None,
                             help_text=_TENSOR_TEXT.format(name=name, role=_TENSOR_ROLES_3D[name],
                                                           fallback='mu' if name in ('c44', 'c55') else '0'),
                             tip=_STIFFNESS_TIP.format(name=name))
    for name in _TENSOR_ROLES_3D}
# solid_mechanics_3d: the same Param objects for the ten shared names, Voigt order c11, c12, .. c66
STIFFNESS_TENSOR_PROPERTIES_3D = tuple(
    _STIFFNESS_3D_ONLY.get(name) or next(param for param in STIFFNESS_TENSOR_PROPERTIES if param.name == name)
    for name in STIFFNESS_TENSOR_ARRAYS_3D)
LOSS_PROPERTIES = (
    _material_property('mechanical_loss_factor', 'Mechanical loss factor', '1', 0.0,
                       help_text='Isotropic structural loss factor eta: every stiffness of the cell becomes c (1 + j '
                                 'eta) (exp(+j omega t)), so Q ~ 1 / eta. Read by the frequency_response study (3D: '
                                 'tetrahedra) and by the eigenfrequency study on triangle / tetrahedron meshes (damped '
                                 'complex modes, Q in the resonance table; not with piezoelectric). The '
                                 'stress_stiffening geometric stiffness stays lossless (dissipation dilution: Q above '
                                 '1 / eta). With viscoelastic moduli (viscosity, relaxation branches; see '
                                 'shear_relaxation_modulus_1) it multiplies only the relaxed G0 and K0. Triangle / '
                                 'tetrahedron meshes: the shown stresses of a response or damped mode are Re(C* eps), '
                                 'the loss part included. Unset = 0; typical 1e-4 (quartz, LN) to 1e-2 (metals; '
                                 'polymers higher).',
                       tip='Damping of the vibration as a fraction of the stiffness; the quality factor of a resonance '
                           'is about 1 divided by it. Used by the frequency response study and, on triangle or '
                           'tetrahedron meshes, by the eigenfrequency study.'),
    _material_property('dielectric_loss_tangent', 'Dielectric loss tangent', '1', 0.0,
                       help_text='Dielectric loss tangent tan delta: the static permittivity becomes eps0 eps_dc (1 - j '
                                 'tan delta) (exp(+j omega t)), a conductance omega C tan delta parallel to the '
                                 'electrode capacitance. Read only by the frequency_response study with piezoelectric '
                                 'on (3D: tetrahedra, every window element). Unset = 0; typical 1e-4 to 1e-3 lithium '
                                 'niobate, ~1e-2 PZT.',
                       tip='Electric loss of the material as the loss tangent. Used only by the frequency response '
                           'study with Piezoelectric on.'),
)
# Isotropic relaxed moduli and frequency-dependent (viscoelastic) moduli of the FEM solids (both types; physics/mechanics
# viscoelastic.py, rules definition_viscoelastic.py). viscosity is laminar_flow's property too (same unit, bounds and
# flags: the registry rule).
_VISCOELASTIC_LAW = ('Full law (exp(+j omega t)): G*(omega) = G0 (1 + j eta) + j omega mu_s + sum_i G_i j omega tau_i / '
                     '(1 + j omega tau_i), K* alike with bulk_viscosity and the bulk branches (eta = '
                     'mechanical_loss_factor).')
_SHEAR_BRANCH_TEXT = 'Branch 2 (3); rules: shear_relaxation_modulus_1.'
_SHEAR_TIME_TEXT = ('Relaxation time tau_i of shear branch i (> 0 when its modulus is set); rules: '
                    'shear_relaxation_modulus_1.')
_BULK_BRANCH_TEXT = 'Bulk branch 2 (3); rules: bulk_relaxation_modulus_1.'
_BULK_TIME_TEXT = ('Relaxation time tau_i of bulk branch i (> 0 when its modulus is set); rules: '
                   'bulk_relaxation_modulus_1.')
_BRANCH_MODULUS_TIP = 'Stiffness that a soft material gains at frequencies above one over its relaxation time.'
_BULK_BRANCH_TIP = 'Compression stiffness that a material gains at frequencies above one over its relaxation time.'
_BRANCH_TIME_TIP = 'Time over which that stiffness relaxes.'
VISCOELASTIC_PROPERTIES = (
    _material_property('shear_modulus', 'Shear modulus (relaxed)', 'Pa', 0.0,
                       help_text='Relaxed (long-time) shear modulus G0; with bulk_modulus the alternative to '
                                 'youngs_modulus / poisson_ratio (both routes on one material: error; not with c_ij '
                                 'entries). Triangle / tetrahedron meshes only. G0 = 0 (or unset) with a viscosity is a '
                                 'liquid (formulation mixed, frequency_response). Every study uses the relaxed G0, K0. '
                                 'Typical: PDMS 2e5-1e6, agarose gel 1e3-1e4, natural rubber 4e5.',
                       tip="Resistance to shearing at long times. An alternative to Young's modulus and Poisson "
                           'ratio.'),
    _material_property('bulk_modulus', 'Bulk modulus (relaxed)', 'Pa', 0.0,
                       help_text='Relaxed bulk modulus K0 (> 0). Unset with formulation mixed = incompressible (1 / K* '
                                 '= 0); the displacement formulation needs it (or youngs_modulus). Triangle / '
                                 'tetrahedron meshes only. Typical: water 2.2e9, silicone oil 1e9, rubber 2e9.',
                       tip='Resistance to compression. Leave empty for an incompressible material when the '
                           'Formulation is Displacement and pressure.'),
    _material_property('viscosity', 'Dynamic viscosity', 'Pa s', 0.0,
                       help_text='Shear viscosity mu_s: G* += j omega mu_s (Kelvin-Voigt dashpot, exp(+j omega t)); a '
                                 'Newtonian liquid has G0 = 0: G* = j omega mu, the linearised unsteady Stokes '
                                 'equations in displacement form. Read by the solid types on triangle / tetrahedron '
                                 'meshes in the frequency_response study only (other studies: ignored, relaxed moduli). '
                                 'A cell with viscosity and no modulus joins the solid only with formulation mixed '
                                 "(otherwise void). The same property as laminar_flow's, where viscosity > 0 makes a "
                                 'cell fluid. Typical: water 1.0e-3 (library), glycerol 1.4, silicone oil 1e-2 to 1e3.',
                       tip='Viscosity of a liquid, or the damping of a soft solid that grows with frequency. Solid '
                           'mechanics uses it only in a frequency response study.'),
    _material_property('bulk_viscosity', 'Bulk viscosity', 'Pa s', 0.0,
                       help_text='Bulk viscosity mu_b: K* += j omega mu_b; needs a finite bulk modulus (bulk_modulus or '
                                 'youngs_modulus). Triangle / tetrahedron meshes, frequency_response only. Typical: '
                                 'water 2.4e-3.',
                       tip='Damping of compression that grows with frequency. Usually empty.'),
    _material_property('shear_relaxation_modulus_1', 'Shear relaxation modulus 1', 'Pa', 0.0,
                       help_text='Generalized Maxwell (Prony) branch i of the shear modulus: G* += G_i j omega tau_i / '
                                 '(1 + j omega tau_i); G_i > 0 needs shear_relaxation_time_i > 0 (and a time needs its '
                                 'modulus); the unrelaxed (instantaneous) modulus is G0 + sum G_i; up to 3 branches. '
                                 'Triangle / tetrahedron meshes, frequency_response only (other studies: the relaxed '
                                 f'G0). {_VISCOELASTIC_LAW}',
                       tip=_BRANCH_MODULUS_TIP),
    *(_material_property(f'shear_relaxation_modulus_{index}', f'Shear relaxation modulus {index}', 'Pa', 0.0,
                         help_text=_SHEAR_BRANCH_TEXT, tip=_BRANCH_MODULUS_TIP) for index in (2, 3)),
    *(_material_property(f'shear_relaxation_time_{index}', f'Shear relaxation time {index}', 's', 0.0,
                         help_text=_SHEAR_TIME_TEXT, tip=_BRANCH_TIME_TIP) for index in (1, 2, 3)),
    _material_property('bulk_relaxation_modulus_1', 'Bulk relaxation modulus 1', 'Pa', 0.0,
                       help_text='Generalized Maxwell branch i of the bulk modulus: K* += K_i j omega tau_i / (1 + j '
                                 'omega tau_i); needs a finite bulk modulus (bulk_modulus or youngs_modulus); K_i > 0 '
                                 'needs bulk_relaxation_time_i > 0; up to 3 branches; rules otherwise as '
                                 'shear_relaxation_modulus_1.',
                       tip=_BULK_BRANCH_TIP),
    *(_material_property(f'bulk_relaxation_modulus_{index}', f'Bulk relaxation modulus {index}', 'Pa', 0.0,
                         help_text=_BULK_BRANCH_TEXT, tip=_BULK_BRANCH_TIP) for index in (2, 3)),
    *(_material_property(f'bulk_relaxation_time_{index}', f'Bulk relaxation time {index}', 's', 0.0,
                         help_text=_BULK_TIME_TEXT, tip=_BRANCH_TIME_TIP) for index in (1, 2, 3)),
)
# Setting formulation of both solid types (after discretization): the unknowns of the FEM solids
FORMULATION_PARAM = Param(
    'formulation', 'choice', DISPLACEMENT, label='Formulation', choices=FORMULATIONS,
    choice_labels=((DISPLACEMENT, 'Displacement'), (MIXED, 'Displacement and pressure (incompressible)')),
    help='Triangle / tetrahedron meshes. displacement: P1 / P2 displacement only (locks toward poisson_ratio 0.5; every '
         'member cell needs G0 > 0 and a finite K0; a viscosity-only cell is void); the frequency_response outputs '
         'pressure / pressure_im are -tr(sigma*) / 3, node-averaged like the stresses. mixed: Taylor-Hood u-p (P2 '
         'displacement, P1 pressure continuous inside each parameter group, duplicated where groups meet: the pressure '
         'jumps there; a group is one distinct set of material values, at most 32: keep the moduli piecewise constant '
         'per region), frequency_response only, discretization quadratic, planar plane strain, axisymmetric (any '
         'azimuthal_order) or 3D, isotropic materials only; unset bulk_modulus = incompressible; liquids (viscosity, no '
         'modulus) join the solid. Outputs pressure / pressure_im (the pressure unknown, p > 0 in compression); '
         'reactions include the pressure. Not with piezoelectric, plane_stress, stress_stiffening, '
         'geometric_nonlinearity, pml_*, periodic axes or sweep_method modal.',
    tip='How the deformation is computed. Displacement and pressure is needed for liquids such as water and for nearly '
        'incompressible materials such as rubber; it runs only in a frequency response study on a triangle or '
        'tetrahedron mesh with quadratic elements.')


# Thermoelastic damping (feature thermoelastic_damping): heat's conductivity and specific heat (the same names, units
# and bounds as the heat type's Params: one property per name in a model with heat; arrays OPTIONAL_TED_SCHEMA).
_TED_READ = 'the solid types read it only in the thermoelastic_damping feature'
TED_MATERIAL_PROPERTIES = (
    _material_property('k', 'Thermal conductivity', 'W/(m K)', 0.0,
                       help_text=f'Isotropic thermal conductivity (sets k_xx, k_yy, k_zz); {_TED_READ}. Typical: '
                                 'silicon 130 (library; pure bulk 148), silica 1.38, silicon nitride 10 (library; films '
                                 '2-15).',
                       tip='How well the material conducts heat. Solid mechanics uses it only for thermoelastic '
                           'damping.\n\nTypical: silicon 130, silica 1.38.'),
    _material_property('k_xx', 'Thermal conductivity along x', 'W/(m K)', 0.0,
                       help_text='Overrides k for this axis.',
                       tip="Thermal conductivity along x. Replaces 'Thermal conductivity' for this direction."),
    _material_property('k_yy', 'Thermal conductivity along y', 'W/(m K)', 0.0,
                       help_text='Overrides k for this axis.',
                       tip="Thermal conductivity along y. Replaces 'Thermal conductivity' for this direction."),
    _material_property('k_zz', 'Thermal conductivity along z', 'W/(m K)', 0.0,
                       help_text='Overrides k for this axis (3D only).',
                       tip="Thermal conductivity along z (3D models only). Replaces 'Thermal conductivity' for this "
                           'direction.'),
    _material_property('cp', 'Specific heat capacity', 'J/(kg K)', 0.0,
                       help_text=f'Specific heat capacity (heat capacity rho cp); {_TED_READ}, which needs rho > 0 and '
                                 'cp > 0 in every solid cell. Typical: silicon 700, silica 746, silicon nitride 700.',
                       tip='Heat stored per kilogram and kelvin. Solid mechanics uses it only for thermoelastic '
                           'damping.\n\nTypical: silicon 700, silica 746.'),
)


# initial_stress_xx (first in both types) carries the rules once
_INITIAL_STRESS_XX_TEXT = (
    'Initial (residual) stress sigma0_xx, e.g. the deposition stress of a film (tension > 0); unset = 0. The total '
    'stress is sigma0 + C eps - beta dT, so a static solve relaxes it (a released structure contracts until it '
    'balances the supports); with stress_stiffening it prestresses the modal / frequency_response pencils (a tensioned '
    'string rises in frequency). Triangle and tetrahedron meshes only (a tensor mesh refuses it). Constant per element '
    '(expressions evaluated at the centroids). 3D: a Voigt row (xx, yy, zz, yz, xz, xy); 2D: in-plane xx '
    '(axisymmetric: rr).')
_INITIAL_STRESS_WHERE = {
    'yy': '2D: in-plane yy (axisymmetric: zz, along the axis).',
    'xy': '2D: in-plane shear (axisymmetric: rz).',
    'zz': ('2D plane strain: out-of-plane (stress_zz only, no in-plane effect); axisymmetric: hoop; plane stress and '
           'generalized plane strain refuse it.'),
    'yz': 'solid_mechanics_3d only.',
    'xz': 'solid_mechanics_3d only.',
}
INITIAL_STRESS_PROPERTIES_3D = tuple(
    _material_property(name, f'Initial stress {name[-2:]}', 'Pa',
                       help_text=(_INITIAL_STRESS_XX_TEXT if name[-2:] == 'xx' else
                                  f'Initial stress sigma0_{name[-2:]}; {_INITIAL_STRESS_WHERE[name[-2:]]} Rules: '
                                  'initial_stress_xx.'),
                       tip=f'Built-in stress {name[-2:]} of the material before any load, for example a tensile film '
                           'stress (positive). Triangle and tetrahedron meshes only.')
    for name in PRESTRESS_ARRAYS_3D)
INITIAL_STRESS_PROPERTIES = tuple(next(param for param in INITIAL_STRESS_PROPERTIES_3D if param.name == name)
                                  for name in ('initial_stress_xx', 'initial_stress_yy', 'initial_stress_xy',
                                               'initial_stress_zz'))
STRESS_STIFFENING_PARAM = Param(
    'stress_stiffening', 'bool', False, label='Stress stiffening',
    help='Triangle and tetrahedron meshes; eigenfrequency, frequency_response and optomechanics studies: first solves '
         'the static state (eigenfrequency: every load, prescribed displacement, initial_stress, thermal strain and '
         'coupled input; frequency_response: initial_stress and thermal strain alone, holds at 0, since the loads are '
         'its drive) and adds the lossless geometric stiffness K_sigma = int grad N . sigma grad N dV of its total '
         'stress to K in every pencil (strings and membranes stiffen, compression softens toward buckling). A free '
         'body, Bloch cell or elastic PML has no static solve: K_sigma of the initial_stress alone (unrelaxed). Off: '
         'loads and initial_stress play no part in the pencil. Not with piezoelectric or on a tensor mesh.',
    tip='On: vibration frequencies include the stiffening by the static stress (from loads, built-in stress or '
        'heating), like a tensioned string. Used by the eigenfrequency and frequency response studies.')


_NONLINEAR_VISIBLE = (('geometric_nonlinearity', (True,)),)
GEOMETRIC_NONLINEARITY_PARAMS = (   # solid_mechanics_3d on tetrahedra, solid_mechanics on triangles
    Param('geometric_nonlinearity', 'bool', False, label='Geometric nonlinearity',
          help='Triangle and tetrahedron meshes: large deflections and rotations in every static solve (stationary, '
               'time_dependent, the static state of eigenfrequency / frequency_response). Total-Lagrangian Saint '
               'Venant-Kirchhoff (small strains, large rotations): Green-Lagrange strain E = (H + H^T + H^T H) / 2 (H '
               'the displacement gradient), second Piola-Kirchhoff stress S = C E + sigma0 - beta dT; damped Newton on '
               'K_T = int (B + B_NL)^T C (B + B_NL) dV + K_sigma(S) over load_steps equal increments (typically 3-6 '
               'iterations each). Boundary loads and body forces are dead loads (fixed direction, reference area); the '
               'steps scale them, prescribed displacements and coupled body forces; initial stress and thermal strain '
               'enter in full every step. Outputs: strain_* Green-Lagrange, stress_* second Piola-Kirchhoff (= Cauchy '
               'to first order); support reactions = int (B + B_NL)^T S dV - f at the held dofs (not K u - f). '
               'Eigenfrequency linearises about the nonlinear state (K_T, stress stiffening included whatever '
               'stress_stiffening says); frequency_response about the state of the initial stress alone (holds at 0). '
               'A Bloch cell, elastic PML or free body has no static solve: K + K_sigma of the initial stress. Not '
               'with piezoelectric or on a tensor mesh; on triangles not with azimuthal_order >= 1, an elastic PML or '
               'generalized plane strain.',
          tip='On: large deflections and rotations are followed exactly, for example the stiffening of a string or a '
              'membrane under a large transverse load, or a released film that bends.\n\nThe static solve is repeated '
              'a few times (Newton iterations) and costs several times a linear solve. Triangle and tetrahedron meshes only.'),
    Param('load_steps', 'int', 1, label='Load steps', min=1, visible_if=_NONLINEAR_VISIBLE,
          help='Geometric nonlinearity: loads and prescribed displacements ramp in this many equal steps, each '
               'Newton-converged from the previous (raise it when a large load does not converge; initial stress and '
               'thermal strain are not ramped).',
          tip='Number of steps the loads are applied in; more steps help a strongly nonlinear solve converge.'),
    Param('nonlinear_tolerance', 'float', 1e-8, label='Nonlinear tolerance', unit='1', min=0.0, min_exclusive=True,
          visible_if=_NONLINEAR_VISIBLE,
          help='Geometric nonlinearity: a load step converges when the largest Newton correction is at most this '
               'fraction of the largest displacement (quadratic convergence: the remaining error ~ its square).',
          tip='Accuracy of the nonlinear static solve: smaller is more accurate and takes an iteration more.'),
    Param('nonlinear_max_iterations', 'int', 25, label='Nonlinear iteration limit', min=1,
          visible_if=_NONLINEAR_VISIBLE,
          help='Geometric nonlinearity: Newton iterations per load step; then a warning, and the next step continues '
               'from the damped state.',
          tip='Most Newton iterations per load step.'),
    Param('tangent_update', 'choice', 'when_slow', label='Tangent update', choices=TANGENT_UPDATES,
          choice_labels=(('every_iteration', 'Every iteration (full Newton)'),
                         ('when_slow', 'Only when convergence slows (modified Newton)')),
          visible_if=_NONLINEAR_VISIBLE,
          help='Geometric nonlinearity: when K_T is refactorised. every_iteration: full Newton (quadratic convergence). '
               'when_slow: modified Newton, the factor is kept (one back substitution) while each iteration cuts the '
               f'largest free residual below {SLOW_RESIDUAL_RATIO:g} x the previous, else refactorised, and carried '
               'into the next load step when that ended fast; saves factorisations on large 3D meshes at the price of '
               'linear convergence near the end (remaining error at most ~1/3 of the nonlinear_tolerance correction). '
               'Counters tangent_factorizations / tangent_reuses.',
          tip='How often the Newton stiffness matrix is rebuilt and factorised. Rebuilding only when convergence slows '
              'saves time on large models; every iteration is the most robust choice.'),
)


DISPLACEMENT_COMPONENT_HELP = 'Prescribed displacement component; empty leaves it free.'


def displacement_param(name, axis):
    """Optional prescribed displacement component Param of the displacement feature."""
    return Param(name, 'float', None, label=f'Displacement {axis}', unit='m', optional=True,
                 help=DISPLACEMENT_COMPONENT_HELP,
                 tip=f'Displacement along {axis} the faces are moved to; empty leaves this direction free.')


# ---- runtime helpers (worker thread) ----
def union_masks(masks):
    """Elementwise OR of a non-empty list of masks."""
    combined = masks[0]
    for mask in masks[1:]:
        combined = combined | mask
    return combined


def apply_body_force(ctx, solver, feature_spec, values, selection, names):
    """Body force of the components `names` (f_x, f_y (, f_z)) on the selected cells; uniform values follow a reused
    sweep build (element meshes: on a host copy of the element mask)."""
    from ...core.backend import to_host   # lazy: definitions never load the backend at import
    mask = ctx.domain_mask(selection.entities)
    solver.add_body_force(mask, *(ctx.eval_field(values[name], mask=mask) for name in names))
    kept_mask = to_host(mask) if ctx.domain.grid_kind in FREE_MESH_KINDS else mask
    follow_load(ctx, solver, feature_spec, values, names, lambda *changes: solver.add_body_force(kept_mask, *changes))
    return None


def apply_temperature_change(ctx, solver, feature_spec, values, selection):
    mask = ctx.domain_mask(selection.entities)
    solver.add_temperature_change(mask, ctx.eval_field(values['delta_T'], mask=mask))
    return None


ELECTRODE_CONNECTION_HELP = (
    'voltage: phi = voltage on every node (0 = ground); the eigenfrequency study holds 0 (short circuit: resonances). '
    'floating: one shared unknown potential carrying the charge setting (0 = isolated); eigenfrequency: charge 0 (open '
    'circuit: antiresonances). At least one voltage electrode is required (the reference of phi).')
ELECTRODE_VOLTAGE_HELP = ('Potential of a voltage electrode in a static solve; ignored by floating electrodes and the '
                          'eigenfrequency study (0 there).')


def electrode_result(session, solver, electrode, feature_spec, options, charge_unit):
    """Table [feature, V, Q] of an electrode (Q in charge_unit); ValueError when piezoelectric is off."""
    if electrode is None:
        raise ValueError(f"electrode '{feature_spec.name}': piezoelectric is off, so it is not part of the solve")
    voltage, charge = solver.electrode_values(electrode)
    return {'kind': 'table', 'columns': ['feature', 'V', 'Q'], 'units': ['', 'V', charge_unit],
            'rows': [[feature_spec.name, voltage, charge]], 'V': voltage, 'Q': charge}


# ---- validation helpers (CPU only) ----
def plain_value(value, parameters):
    """Float of a value that evaluates over the parameters; None when unset or field / state dependent."""
    if value is None:
        return None
    try:
        return evaluate_scalar(value, parameters)
    except ExpressionError:
        return None


def outer_faces(vctx, feature, faces) -> list:
    """Outer faces (of `faces`) of a feature's selection; [] when it does not resolve (reported elsewhere)."""
    try:
        return [entity for entity in expand_entities(vctx.spec, feature.selection) if entity in faces]
    except (ValueError, TypeError):
        return []


def support_holders(vctx, physics_spec, held_components):
    """{(face or object, axis): [feature names]} of the enabled support features (held_components(vctx, feature))."""
    holders = {}
    for feature in physics_spec.features:
        if feature.enabled and feature.type in SUPPORT_FEATURES:
            for key in held_components(vctx, feature):
                holders.setdefault(key, []).append(feature.name)
    return holders


_SHEAR_PRESTRESS_NAMES = ('initial_stress_xy', 'initial_stress_xz', 'initial_stress_yz')
SYMMETRY_HELP = (   # the part of the symmetry help both solid types share (after their own first sentences)
    'Fields unfold across the plane as the even mirror of the displacement (the normal displacement and the shear '
    'components with one index along the normal flip, everything else keeps: Unfold mirror planes, '
    'Results.field(unfold=True)); field derived integrals reaching the plane count twice (the unfolded structure), '
    'max / min include the image, a mean is unchanged. It counts as a support for the rigid-body checks (it holds the '
    'normal direction). Every load, material, initial stress and coupled field of the drawn part has its mirror image '
    'in the unfolded model, so they need nothing on the plane. Results the solid reports itself are those of the '
    'drawn part: support reaction tables (this one: the normal force the mirrored part exerts across the plane on '
    'the drawn one; another support meeting the plane: the unfolded structure carries twice its components along '
    'the plane, its component normal to the plane cancels){own_results}; ratios (eigenfrequencies, thermoelastic Q) '
    'are those of the unfolded structure. Errors: another face feature of this solid on the same face, piezoelectric '
    'on, a material whose stiffness tensor changes sign under the reflection (its image would be another crystal cut, '
    'e.g. a monoclinic c16), a uniform shear initial stress with an index along the normal, an elastic PML on a face '
    'meeting the plane (its absorbing layer crosses the plane, where the hold covers the window face only), a '
    'periodic face, a polar domain, another physics with mirror bc settings (EM) that does not set the face '
    'symmetric / antisymmetric, parity_sectors (the solid has no odd mirror feature){own_errors}. Warning: a uniform '
    'body force normal to the plane (its image points the other way).')


def symmetry_issues(vctx, physics_spec, feature_spec, settings_params, faces_all, tensor_names) -> list:
    """Rules of the symmetry feature both solid types share: definition_helpers.mirror_feature_issues (Cartesian
    domain, not the axisymmetric axis, EM partners mirror the face), errors for piezoelectric on (the potential's
    mirror parity depends on the crystal: not modelled), per selected material whose stiffness entries (tensor_names)
    change sign under the reflection of a selected face (crystal_tensors.mirror_violation) or whose uniform shear
    initial stress has an index along its normal, and a warning per uniform body force normal to a selected face."""
    from ...materials.crystal_tensors import mirror_violation   # lazy: physics -> materials stays out of import time
    path = feature_path(physics_spec, feature_spec)
    issues = list(mirror_feature_issues(vctx, physics_spec, feature_spec, settings_params))
    settings = vctx.values(settings_params, physics_spec.settings) or {}
    if settings.get('piezoelectric'):
        issues.append(Issue('error', path, 'a symmetry face is not available with piezoelectric on (the mirror parity '
                                           'of the potential and the charges on the plane depend on the crystal and '
                                           'the electrodes): model the full structure, or turn piezoelectric off'))
    axes = sorted({face[0] for face in outer_faces(vctx, feature_spec, faces_all)})
    if not axes:
        return issues
    parameters = vctx.parameters
    for material, properties in merged_materials(vctx):
        if properties is None or not material.selection:   # a failed merge is reported at the material
            continue
        values = {name: plain_value(properties.get(name), parameters) for name in tensor_names}
        for axis in axes:
            relative, entry = mirror_violation(values, 'xyz'.index(axis))
            if entry:
                issues.append(Issue('error', path, f"material '{material.name}' is not symmetric under {axis} -> "
                                                   f'-{axis} ({entry} changes sign, {relative:.3g} of its largest '
                                                   'stiffness entry): the unfolded model would hold its mirror image '
                                                   '(another crystal cut) beyond the plane; model the full width or '
                                                   'use a Roller (a support, no mirror plane)'))
            for name in _SHEAR_PRESTRESS_NAMES:
                value = plain_value(properties.get(name), parameters)
                if axis in name[-2:] and value is not None and value != 0.0:
                    issues.append(Issue('error', path, f"material '{material.name}' sets a uniform {name} = "
                                                       f'{value:.4g} Pa, which flips sign under {axis} -> -{axis}: '
                                                       'the unfolded model would carry opposite shear prestresses '
                                                       'on both sides of the plane; model the full width'))
    body_force = vctx.registry.physics_type(physics_spec.type).feature_type('body_force')
    for feature in physics_spec.features:
        if not (feature.enabled and feature.type == 'body_force'):
            continue
        values = vctx.values(body_force.params, feature.params) or {}
        for axis in axes:
            value = values.get(f'f_{axis}')
            if isinstance(value, (int, float)) and value != 0.0:
                issues.append(Issue('warning', path, f"body force '{feature.name}' has a uniform f_{axis} = "
                                                     f'{value:.4g} N/m^3 normal to the symmetry plane: its mirror '
                                                     'image points the other way, so the unfolded model is pushed '
                                                     'apart (or together) symmetrically; for a force acting across '
                                                     'the plane (e.g. gravity) model the full width'))
    return issues


def model_mirror_issues(vctx, physics_spec, path, faces_all, axis_held=False) -> list:
    """Warning per model mirror plane (core.mirror.mirror_faces: an EM mirror face, another physics' symmetry face) on
    one of faces_all that no Roller, Symmetry or Fixed of the solid holds (axis_held: x_lo of geometry axisymmetric,
    held by the axis rule): the solid fields unfold there as the even mirror the face does not solve."""
    planes = [face for face in mirror_faces(vctx.spec) if face in faces_all]
    if not planes:
        return []
    held = {'x_lo'} if axis_held else set()
    for _, entities in enabled_feature_entities(vctx, physics_spec, (*ROLLER_FEATURES, 'fixed')):
        held.update(entity for entity in entities if entity in faces_all)
    return [Issue('warning', path, f"{face} is a mirror plane of the model (an EM mirror face or another physics' "
                                   f"symmetry face), where the fields of '{physics_spec.name}' unfold as the even "
                                   'mirror (normal displacement 0), but no Roller, Symmetry or Fixed of it holds that '
                                   'face: solid cells reaching it unfold as if cut along the plane; add a symmetry '
                                   f'feature on {face}')
            for face in planes if face not in held]


def pml_crossing_issues(path, faces, pml_faces) -> list:
    """Error per symmetry face (faces) that an elastic PML on a face of another axis meets: its ring / shell crosses
    the plane, where the hold covers the window face only, so the absorber is no mirror image there."""
    issues = []
    for face in faces:
        crossing = [pml for pml in pml_faces if pml[0] != face[0]]
        if crossing:
            issues.append(Issue('error', path, f"the elastic PML on {', '.join(crossing)} crosses the symmetry plane "
                                               f'{face} (its absorbing layer continues beyond the window face, which '
                                               'is all the symmetry feature holds): model the full width across '
                                               f'{face}, or use a Roller there (a support, no mirror plane)'))
    return issues


def free_body_allowed(study_type, settings_values) -> bool:
    """A free body may be solved (triangles, tetrahedra): the eigenfrequency / optomechanics study at target_frequency >
    0 (its 0 Hz modes dropped) or the frequency_response study (K - omega^2 M is regular at f > 0)."""
    return bool((study_type in MODAL_STUDIES and settings_values.get('target_frequency'))
                or study_type == FREQUENCY_RESPONSE_STUDY)


def nonlinear_state_infos(vctx, path, static_refused) -> list:
    """geometric_nonlinearity infos: no nonlinear static solve (static_refused), or the frequency_response study
    (linear about the state of the initial stress)."""
    if static_refused:
        return [Issue('info', path, f'geometric_nonlinearity: {static_refused} has no static solve, so the pencil takes '
                                    'K + K_sigma of the initial stress (the tangent at u = 0)')]
    if vctx.study_type == FREQUENCY_RESPONSE_STUDY:
        return [Issue('info', path, 'geometric_nonlinearity: the frequency response stays linear (small vibrations) '
                                    'about the nonlinear static state of the initial stress and thermal strain alone; '
                                    'the loads are its harmonic drive')]
    return []


def piezo_material_issues(vctx, physics_spec, path, piezo_arrays, void_text, merge_errors) -> list:
    """Warnings per selected material setting piezo_e* (any of piezo_arrays) without youngs_modulus or c11 > 0 (void:
    void_text says what remains) and when no selected material sets one (a material whose merge raises merge_errors
    counts as piezoelectric: reported at the material)."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    issues = []
    piezo_materials = []
    for material in vctx.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except merge_errors:   # reported at the material
            piezo_materials.append(material.name)
            continue
        if not (material.selection and any(properties.get(name) is not None for name in piezo_arrays)):
            continue
        piezo_materials.append(material.name)
        stiff = any(properties.get(name) is not None and (plain_value(properties.get(name), vctx.parameters) is None
                                                          or plain_value(properties.get(name), vctx.parameters) > 0.0)
                    for name in ('youngs_modulus', 'c11'))   # a non-plain expression counts as stiff
        if not stiff:
            issues.append(Issue('warning', ('material', material.name),
                                f"material '{material.name}' sets piezo_e* but no youngs_modulus or c11 > 0: "
                                f'{physics_spec.name} treats it as void {void_text}'))
    if not piezo_materials:
        issues.append(Issue('warning', path, 'piezoelectric is on but no selected material sets a piezo_e* entry: the '
                                             'potential only adds a dielectric, nothing couples'))
    return issues


def _stiff(properties, parameters, tensor, fluid=False):
    """True when a merged material evaluates to a youngs_modulus > 0 (tensor: or a c11 > 0, the c_ij entries, or a
    shear_modulus / bulk_modulus set: the FEM meshes' moduli route); fluid (the mixed formulation of a frequency
    response): a viscosity > 0 counts too (a liquid joins the solid)."""
    return (evaluates_to(properties.get('youngs_modulus'), parameters, lambda value: value > 0.0)
            or (tensor and (evaluates_to(properties.get('c11'), parameters, lambda value: value > 0.0)
                            or any(properties.get(name) is not None for name in MODULUS_ARRAYS)))
            or (fluid and evaluates_to(properties.get('viscosity'), parameters, lambda value: value > 0.0)))


def mixed_formulation(vctx, settings_values) -> bool:
    """The solid runs the mixed u-p formulation: setting formulation mixed in a frequency_response study (elsewhere
    validation refuses it)."""
    return (vctx.study_type == FREQUENCY_RESPONSE_STUDY and bool(settings_values)
            and settings_values.get('formulation', DISPLACEMENT) == MIXED)


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


def material_issues(vctx, physics_spec, path, type_name, element_name, tensor=False, fluid=False):
    """Stiffness checks over the spec materials, reported by the first physics of type_name: no youngs_modulus (tensor:
    or c11, shear_modulus / bulk_modulus; fluid, the mixed formulation: or a viscosity) anywhere (error), poisson_ratio
    unset (info) or above the locking limit (warning naming element_name, e.g. 'bilinear')."""
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
        has_stiffness = _stiff(properties, parameters, tensor, fluid)
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
        stiffness_text = ('a youngs_modulus, c11, shear_modulus or bulk_modulus > 0' if tensor else 'a youngs_modulus')
        if fluid:
            stiffness_text += ' or a viscosity > 0 (a liquid of the mixed formulation)'
        issues.append(Issue('error', path, f'no physical cell has {stiffness_text}: assign a material with stiffness '
                                           '(e.g. a library solid such as silica) to the solid'))
    return issues


def modal_study_issues(vctx, physics_spec, path, modal_studies, thermal_strain_coupling, tensor=False,
                       prestress_info=True, stress_stiffening=False, fluid=False):
    """In modal_studies only: error per selected stiff material (tensor: c11, shear_modulus / bulk_modulus count; fluid,
    the mixed formulation: a viscosity too) without a density rho > 0 (plain numbers; the solve checks every cell),
    info (prestress_info, stress_stiffening off) when loads, initial stresses or a thermal strain coupling are present
    (no prestress)."""
    if vctx.study_type not in modal_studies:
        return []
    issues = []
    stiffness_names = 'youngs_modulus, c11, shear_modulus or bulk_modulus' if tensor else 'youngs_modulus'
    if fluid:
        stiffness_names += ', or a viscosity'
    for material, properties in merged_materials(vctx):
        if properties is None or not (material.selection and _stiff(properties, vctx.parameters, tensor, fluid)):
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
        description=(f'Vibration modes of the last eigenfrequency solve, {order_text} (search targets / band: in '
                     f"frequency order). Result 'table', one row per mode: {', '.join(MODAL_TABLE_COLUMNS)} (1-based "
                     'rank, Hz, rad/s); damped modes (elastic PML, loss factor, resistive_load electrode) append '
                     f"{', '.join(DAMPED_TABLE_COLUMNS)} (Hz, > 0 decays; Re omega / (2 Im omega); 1/s, 2 Im omega = "
                     'kappa_e of a resistive load). Extras target_frequency (Hz; NaN for several targets or a band), '
                     'targets, band, expected_count, selected_resonance, axial_wavenumber (rad/m; None = plane strain). '
                     "Frequency versus a parameter: a derived value with source '<physics>.<this feature>', expression "
                     'frequency, operation at, at = 0 (the first mode).'),
        summary="Lists the vibration frequencies found by an 'Eigenfrequency' study.",
        apply=_apply_resonance_table, result=_resonance_table_result, validate=_validate_resonance_table)



# ---- thermoelastic_damping feature (both types; physics/mechanics/thermoelastic.py) ----
TED_FEATURE = 'thermoelastic_damping'
_TED_TEMPERATURE_PARAM = Param(
    'T0', 'float', 293.15, label='Reference temperature', unit='K', min=0.0, min_exclusive=True,
    help="Absolute temperature of the vibrating solid (the heat type's T0 default); 1 / Q_TED scales with it (Delta = E "
         'alpha^2 T0 / (rho cp)).',
    tip='Temperature of the resonator, in kelvin. Thermoelastic damping grows in proportion to it.\n\nRoom '
        'temperature: 293.15 K.')


def _apply_thermoelastic_damping(ctx, solver, feature_spec, values, selection):
    from .thermoelastic import ThermoelasticDamping
    damping = ThermoelasticDamping(feature_spec.name, values['T0'])
    if not hasattr(solver, 'thermoelastic_dampings'):   # name -> runtime (optomechanical_coupling quality_from)
        solver.thermoelastic_dampings = {}
    solver.thermoelastic_dampings[feature_spec.name] = damping
    return damping


def _thermoelastic_damping_result(session, solver, runtime, feature_spec, options):
    return runtime.table(solver)


def _validate_thermoelastic_damping(vctx, physics_spec, feature_spec, values):
    """Warning outside the modal studies; errors for the solid variants TED does not model and for solid materials
    with a thermal expansion but no specific heat cp / density rho."""
    path = feature_path(physics_spec, feature_spec)
    issues = []
    if vctx.study_type not in MODAL_STUDIES:
        issues.append(Issue('warning', path, f"thermoelastic_damping is evaluated only after an {EIGENFREQUENCY_STUDY} "
                                             f"or {OPTOMECHANICS_STUDY} study (this model runs {vctx.study_type}): its "
                                             'result will be missing'))
    try:
        settings = resolve_values(vctx.registry.physics_type(physics_spec.type).settings,
                                  physics_spec.settings if isinstance(physics_spec.settings, dict) else {},
                                  vctx.parameters)
    except (KeyError, ValueError, TypeError):   # reported at the physics
        settings = {}
    refused = [text for name, text in (('piezoelectric', 'a piezoelectric solid'),
                                       ('axial_wavenumber', 'guided (2.5D) modes (axial_wavenumber)'))
               if settings.get(name) not in (None, False)]
    if settings.get('azimuthal_order'):
        refused.append('an azimuthal_order other than 0')
    if refused:
        issues.append(Issue('error', path, f"thermoelastic_damping does not model {', '.join(refused)}"))
    for material, properties in merged_materials(vctx):
        if properties is None or not material.selection:
            continue
        solid = any(properties.get(name) not in (None, 0, 0.0) for name in ('youngs_modulus', 'c11'))
        expands = any(properties.get(name) not in (None, 0, 0.0)
                      for name in ('alpha', 'alpha_xx', 'alpha_yy', 'alpha_zz'))
        missing = [name for name in ('rho', 'cp') if properties.get(name) is None]
        if solid and expands and missing:
            issues.append(Issue('error', path, f"material '{material.name}' expands with temperature but sets no "
                                               f"{' / '.join(missing)}: thermoelastic damping needs the heat capacity "
                                               'rho cp of every solid cell'))
    return issues


def _thermoelastic_damping_memory(values, n_selected_cells, mesh):
    """Host items of one thermoelastic_damping evaluation (heuristic, uncalibrated): the thermal conductance, capacity
    and complex operator of the solid's scalar dofs (P2 on element meshes, cell corners on a tensor mesh) and the
    complex SuperLU factor of one mode at a time."""
    kind = getattr(mesh, 'kind', 'tensor')
    if kind in FREE_MESH_KINDS:
        unknowns = int(mesh.n_nodes) + int(mesh.n_edges)
        entries = (27 if kind == 'tetrahedra' else 15) * unknowns
    else:
        unknowns = int(mesh.n_cells)
        entries = 9 * unknowns
    dimensions = 3 if kind == 'tetrahedra' else 2
    return MemoryEstimate.from_items(host_items=(
        ('thermoelastic damping thermal operators (conductance, capacity, complex sum)', 4 * 12 * entries),
        ('thermoelastic damping complex SuperLU factor (heuristic)', 2 * host_real_lu_bytes(unknowns, dimensions))))


def thermoelastic_damping_feature(grids, where_text):
    """The thermoelastic_damping FeatureType on the mesh kinds grids; where_text names them for the description."""
    from .thermoelastic import TABLE_COLUMNS
    return FeatureType(
        TED_FEATURE, 'Thermoelastic damping', 'monitor', None, params=(_TED_TEMPERATURE_PARAM,), result_kind='table',
        summary='Estimates the quality factor each vibration mode loses to heat flow between compressed and stretched '
                'regions (thermoelastic damping), the main intrinsic loss of silicon and silicon nitride resonators.',
        description=(
            'Thermoelastic damping (TED) of every mode of the last eigenfrequency (or optomechanics) solve, '
            f'{where_text}; perturbative and one-way (the modes stay isothermal). Per mode (omega = Re omega, exp(+j '
            'omega t)) the strain-rate heat source q = -j omega T0 (beta : eps), beta = C : alpha the thermal stress per '
            'kelvin, drives j omega rho cp theta - div(k grad theta) = q on the solid, adiabatic on its whole boundary '
            '(theta on its own nodes: bilinear on a tensor mesh, P1 / P2 on element meshes); d = -int beta : eps* theta '
            'dV gives 1 / Q_TED = Im d / (u^H K u) (> 0) and ted_frequency_shift = Re d / (2 u^H K u) (> 0: toward the '
            'adiabatic stiffness). Thin-beam flexure reproduces Zener (Delta = E alpha^2 T0 / (rho cp), tau = h^2 rho '
            'cp / (pi^2 k)) and Lifshitz-Roukes within 2 %. Materials: k (k_xx, k_yy, k_zz), rho, cp (library silicon, '
            'silica, silicon_nitride have them), alpha, the stiffness; every solid cell needs rho > 0 and cp > 0. '
            f"Result 'table', one row per mode: {', '.join(TABLE_COLUMNS)} (Hz; Q inf without loss): q_other = the "
            "other losses (a damped pencil's Q: elastic PML and, on element meshes, mechanical_loss_factor; tensor "
            'mesh: the loss factor perturbatively, 1 / Q = u^H (eta K) u / u^H K u; tetrahedra: also q_gas), 1 / '
            'q_total = 1 / q_ted + 1 / q_other. Extras T0, selected_resonance. Complex modes use Re omega and Re(u^H M '
            'u). One complex host SuperLU per mode at the first read. Not modelled: piezoelectric, guided modes, '
            'azimuthal_order >= 1, Bloch cells; PML elements take no part. Q_TED versus a parameter: derived value '
            "source '<physics>.<this feature>', expression q_ted, operation at, at = 0."),
        apply=_apply_thermoelastic_damping, result=_thermoelastic_damping_result,
        validate=_validate_thermoelastic_damping, memory_estimate=_thermoelastic_damping_memory, grids=grids)

# ---- frequency_probe feature and complex reactions (triangle / tetrahedron meshes) ----
def _frequency_probe_result(session, solver, record, feature_spec, options):
    from ...core.harmonic_probe import harmonic_probe_result   # lazy: core.harmonic_probe imports core.backend (cupy)
    return harmonic_probe_result(record)


def _validate_frequency_probe(vctx, physics_spec, feature_spec, values):
    """The point inside the domain; a warning outside the frequency_response study (no frequency is recorded)."""
    from ...model.definition_helpers import validate_points_inside
    issues = list(validate_points_inside(vctx, physics_spec, feature_spec, values))
    if vctx.study_type != FREQUENCY_RESPONSE_STUDY:
        issues.append(Issue('warning', feature_path(physics_spec, feature_spec),
                            f'frequency_probe is filled only by a {FREQUENCY_RESPONSE_STUDY} study (this model '
                            f'runs {vctx.study_type}): its result will be missing'))
    return issues


def frequency_probe_feature(apply, components, grids):
    """The frequency_probe FeatureType of the displacement components (u_x, u_y[, u_z]) on the mesh kinds grids."""
    series = ', '.join(f'{component}_re / _im / _abs / _phase' for component in components)
    return FeatureType(
        'frequency_probe', 'Frequency probe', 'probe', 'point', object_kinds=('point',), max_entities=1,
        result_kind='xy',
        summary='Records the vibration amplitude and phase at a point for every frequency of a frequency response.',
        description=('frequency_response study only: the complex displacement (exp(+j omega t), phase 0 = the drive) '
                     'at the point per solved frequency (triangles: a node is put there; 0 off the solid). Result '
                     f"'xy' versus frequency [Hz]: {series} (m; phase rad in (-pi, pi]); derived value two_wave_fit "
                     'pairs a _re series with its _im. Restarts with each frequency_response run.'),
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


def support_result(solver, region, axes, unit):
    """Result of a support feature: the complex reaction table while a frequency response is shown (FEM meshes), else
    one row [feature, F_<axis>, ...] of solver.reaction(region) in `unit` with the F_<axis> extras."""
    table = harmonic_support_result(solver, region)
    if table is not None:
        return table
    forces = solver.reaction(region)
    names = [f'F_{axis}' for axis in axes]
    return {'kind': 'table', 'columns': ['feature', *names], 'units': ['', *(unit for _ in names)],
            'rows': [[region.name, *forces]], **dict(zip(names, forces))}


def stiffness_follows_state(context):
    """True when a spec material's stiffness (youngs_modulus, poisson_ratio, c_ij) references a physics state field or
    time t (the operator is re-assembled during the run)."""
    return material_follows_state(context, STIFFNESS_ARRAYS)


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
