"""Model-layer definitions of the 'solid_mechanics' physics type (static plane strain: features, validation, memory
estimate; settings and material vocabulary from definition_common), the solid couplings (heat_solid_thermal_strain also
targets solid_mechanics_3d of definition3d, registered from here), plus the template and selftests. Loaded lazily by the
registry (physics/mechanics/__init__.py must not import it)."""
from dataclasses import replace
from functools import partial

import numpy as np

from ...core import time_estimate as te
from ...core.constants import FACES
from ...core.memory_estimate import (BOOL_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, INT32_BYTES, MemoryEstimate, combine,
                                     host_real_lu_bytes)
from ...core.monitors import probe_result
from ...coupling.coupling_term import CouplingGroup, CouplingTerm
from ...coupling.index_terms import index_contributor, index_slots, target_type, writes_index_slots
from ..em.index_reference import INDEX_SLOTS
from ...model.definition_helpers import (AXISYMMETRIC_SPACE_SETTINGS, AZIMUTHAL_ORDER_PARAM, DIFFUSION_DIRECT_BACK_ENDS,
                                         axisymmetric_issues, azimuthal_order_issues, azimuthal_order_of,
                                         cg_solver_issues, coupling_target, cudss_external, eigen_factorization,
                                         eigen_role,
                                         eigen_search_counts, eigen_subspace_size,
                                         evaluates_to, frequency_sweep_counts, geometry_param, host_direct_back_end,
                                         is_axisymmetric,
                                         linear_solver_mixed, linear_solver_precision, on_triangles, point_cells,
                                         reduced_sweep_memory_items,
                                         reduced_sweep_time_items, validate_points_inside)
from ...model.geometry import expand_entities, line_orientation
from ...model.schema import (CouplingType, DomainRequirements, FeatureType, Issue, Param, PhysicsType, SolverRole,
                             feature_path, solver_backend)
from ...solvers.arpack_shift_invert import CUDSS_FACTORIZATIONS, LANCZOS_APPLICATIONS_PER_BASIS, arnoldi_basis_size
from ...solvers.host_direct import factor_label
from ...solvers.pcg import workspace_bytes
from . import definition3d
from .definition3d import eigen_host_direct, host_factor_items
from .coupling import make_temperature_rise_transform
from .definition_common import (DENSITY_PARAM, EIGENFREQUENCY_STUDY, GEOMETRIC_NONLINEARITY_PARAMS,
                                INITIAL_STRESS_PROPERTIES, LOSS_PROPERTIES, MATERIAL_PROPERTIES, PHOTOELASTIC_P44_PARAM, PIEZO_MATERIAL_PROPERTIES,
                                STIFFNESS_TENSOR_PROPERTIES, STRESS_STIFFENING_PARAM, SUPPORT_FEATURES,
                                background_context, displacement_param, eigen_search_issues, follow_held_values,
                                follow_load, following, frequency_probe_feature,
                                harmonic_support_result, initial_stress_materials, material_issues,
                                materials_setting, modal_kept_count, modal_resonance_count, modal_study_issues,
                                plain_value, prestress_issues, register_support, resonance_table_feature,
                                solid_linear_solver, solid_study, stiffness_follows_state, union_masks)
from .definition_frequency_response import (ADMITTANCE_FEATURE, FREQUENCY_RESPONSE_STUDY, anisotropy_issues,
                                            frequency_response_host_items, frequency_response_issues)
from .fields import ENERGY_DENSITY_FIELDS, HARMONIC_IMAG_FIELDS, INPUT_ARRAYS, OUTPUT_ARRAYS
from .generalized_plane_strain import BENDING_CHOICES, BENDING_TERMS
from .materials import (DENSITY_SCHEMA, LOSS_ARRAYS, MECHANICAL_SCHEMA, OPTIONAL_LOSS_SCHEMA, OPTIONAL_PIEZO_SCHEMA,
                        OPTIONAL_PRESTRESS_SCHEMA, OPTIONAL_STIFFNESS_SCHEMA, PIEZO_ARRAYS, PIEZO_DEFAULTS,
                        PRESTRESS_ARRAYS, SHEAR_PHOTOELASTIC_SCHEMA, SOLVER_PRECISIONS, STIFFNESS_TENSOR_ARRAYS)
from .piezo import ELECTRODE_CONNECTIONS, PIEZO_COMPONENTS
from .pml2d import DEFAULT_PML_CELLS, DEFAULT_PML_REFLECTION, pml_padding
from .definition_optomechanics import (OPTICAL_TYPES, OPTOMECHANICAL_COUPLING_FEATURE, OPTOMECHANICS_STUDY,
                                       backward_scattering, optomechanics_cases, register_optomechanics_templates)
from .modal import EXTRA_EIGENVALUES
from .optical_force import make_optical_force_transforms
from .renderer import SolidRenderer
from .selftests import HEATED_WAVEGUIDE_STRESS_DESCRIPTION, HEATED_WAVEGUIDE_STRESS_LABEL, heated_waveguide_stress_spec
from .solver import (GENERALIZED_PLANE_STRAIN, OUT_OF_PLANE_CONDITIONS, PLANE_STRAIN, PLANE_STRESS,
                     SolidMechanicsPhysics)
from .solver_common import SupportRegion
from .solver_fem import DISCRETIZATIONS, SolidMechanicsFEMPhysics
from .strain_optic import make_strain_optic_transform

SOLID_TYPE_NAME = 'solid_mechanics'                      # = selftests.SOLID_TYPE_NAME (selftests never import this)
SOLID_3D_TYPE_NAME = 'solid_mechanics_3d'                # = selftests_3d.SOLID_3D_TYPE_NAME
THERMAL_STRAIN_COUPLING = 'heat_solid_thermal_strain'    # = selftests.THERMAL_STRAIN_COUPLING
STRAIN_OPTIC_COUPLING = 'solid_em_strain_optic'          # = selftests.STRAIN_OPTIC_COUPLING
OPTICAL_FORCE_COUPLING = 'em_solid_optical_force'        # = selftests.OPTICAL_FORCE_COUPLING
HEAT_TYPE_NAME = 'heat'
_FDFD_EZ_TYPE_NAME = 'em_fdfd_ez'        # source of the optical force (E_z phasor)
_CROSS_SECTION_EM_TYPE = 'em_modes'      # the EM type whose xy plane is a cross-section like the solid's
_COMPONENT_NAMES = ('x', 'y')
_MODAL_STUDIES = (EIGENFREQUENCY_STUDY, OPTOMECHANICS_STUDY)   # studies running solve_eigenfrequencies
_TRIANGLES = 'triangles'
_PLANE_DISPLACEMENTS = ('u_x', 'u_y')   # load / held value names (follow_load / follow_held_values)
_PLANE_TRACTIONS = ('t_x', 't_y')
_PLANE_BODY_FORCES = ('f_x', 'f_y')
_BOTH_GRIDS = ('tensor', _TRIANGLES)   # features the triangle solid (solver_fem) runs too
_FEM_STUDIES = ('stationary', 'time_dependent', EIGENFREQUENCY_STUDY,   # studies of the solid on a triangle mesh
                FREQUENCY_RESPONSE_STUDY, OPTOMECHANICS_STUDY)

# The 'linear' SolverRole of solid_mechanics (local: no other type uses it): tensor
# mesh = block-Jacobi CSR pcg (no scalar multigrid: it does not fit the vector problem), 'mixed' = float32 inner CG on
# float32 copies of the CSR and the 2 x 2 inverse blocks (system.ElasticSystem._bind_linear_solver); Automatic on the
# gpu backend = pcg mixed (3D speed plan A1: float64 is slow on consumer GPUs; the CSR product is bound by bytes, so
# ~1.5x here, not 3D's 17x; a pcg node setting no precision takes it too); triangle mesh (mesh plan D11) = sparse_direct
# on both backends (back_end 'auto' = cudss on the gpu backend with nvmath, B1a: double, thin-film K is badly
# conditioned and its frequency response near resonance converges worst in float32)
SOLID_LINEAR_ROLE = SolverRole('spd', 'pcg', precisions=('single', 'double', 'mixed'),
                               setting_choices={'preconditioner': ('jacobi',), 'back_end': DIFFUSION_DIRECT_BACK_ENDS},
                               mixed_preconditioners=('jacobi',),
                               backend_defaults={'gpu': ('pcg', {'precision': 'mixed'})},
                               grid_defaults={_TRIANGLES: ('sparse_direct', {})},
                               auto_back_ends={_TRIANGLES: 'cudss'})

_DISCRETIZATION_PARAM = Param(
    'discretization', 'choice', 'quadratic', label='Discretization', choices=tuple(DISCRETIZATIONS),
    choice_labels=(('quadratic', 'Quadratic (P2)'), ('linear', 'Linear (P1)')),
    help="Triangle meshes only (mesh kind 'triangles'; a tensor mesh ignores it: bilinear cells): the displacement "
         "elements. quadratic (default, COMSOL's choice): P2 Lagrange, isoparametric on curved boundaries (circle / "
         'ellipse edges follow the drawn curve), strain and stress linear per element; linear: P1, constant strain '
         'per element (stiff in bending, locks toward poisson_ratio 0.5). Unknowns per element and component: ~2 '
         '(quadratic) or ~0.5 (linear); one host SuperLU per assembly (sparse_direct). Displayed strain / stress are '
         'averaged at the nodes over the solid elements around them (a material interface is smeared over one '
         'element; displacements and the state() element values are not).',
    tip='How the displacement varies inside each triangle: quadratic is far more accurate per element, linear is '
        'cheaper.\n\nOnly used on a triangle mesh.')

_TARGET_FREQUENCY_PARAM = Param(
    'target_frequency', 'float', 0.0, label='Eigenfrequency target', unit='Hz', min=0.0,
    help='Eigenfrequency study only: the vibration modes nearest this frequency are searched (shift-invert about '
         '(2 pi target_frequency)^2, moved 1e-4 above it so a target on a resonance still factors); 0 = the lowest '
         'modes (a body that moves without strain then fails: check the supports). On a triangle mesh a target > 0 '
         'also allows a free body (no supports; axisymmetric: no held u_z): its rigid-body modes (f below 1e-4 x the '
         'target) are dropped. Ignored by the other studies.',
    tip="Used only by an 'Eigenfrequency' study: the vibration modes closest to this frequency are found.\n\n"
        '0 finds the lowest modes.')
_AXIAL_WAVENUMBER_PARAM = Param(
    'axial_wavenumber', 'float', None, label='Axial wavenumber', unit='rad/m', min=0.0, optional=True,
    help='Eigenfrequency study only. Empty (default): the in-plane plane-strain vibration modes (u_x, u_y). A value '
         'k_z: the guided (2.5D) modes of the structure long along z, u ~ exp(i (k_z z - omega t)) with all three '
         'components coupled (Q1 elements, exact element integrals; the complex Hermitian operator is solved in the '
         'equivalent real symmetric form u_z -> -i u_z, exact for isotropic cells): the physical displacement is (u_x '
         'cos phi, u_y cos phi, u_z sin phi), phi = k_z z - omega t, so field u_z is the amplitude of the quadrature '
         'part and the strains / stresses are the phi = 0 values (elastic_strain_zz = k_z u_z enters stress_* and '
         'von_mises; the shear strains xz / yz peak at phi = pi / 2 and are not output). k_z > 0 leaves no rigid motion: '
         'supports are optional (a free suspended wire). k_z = 0 gives the plane-strain modes plus the antiplane (pure '
         'u_z) modes; these need u_z held on every solid region: Fixed and Fixed (domain) hold u_z, Roller and '
         'Displacement leave it free. Typical: backward Brillouin scattering needs k_z = 2 beta = 4 pi n_eff / lambda '
         '(~2.5e7 rad/m for a silicon wire at 1550 nm). For a dispersion relation f(k_z) sweep a parameter used here. '
         'Ignored by the other studies; the optomechanics study refuses it (an optomechanical_coupling with scattering '
         'backward makes that study set k_z = 2 Re beta itself).',
    tip="Used only by an 'Eigenfrequency' study. Empty finds the vibrations within the cross-section. A value finds "
        'the vibrations travelling along the structure with this wavenumber (2 pi divided by their wavelength), '
        'including motion along the structure.\n\nWith a value, the structure may be left without supports.')
_PIEZOELECTRIC_PARAM = Param(
    'piezoelectric', 'bool', False, label='Piezoelectric',
    help='On: piezoelectric plane strain. A nodal electric potential phi (the same bilinear elements, every window '
         'node, void cells included as dielectric) couples to the displacement through the piezo_e* material arrays '
         '(solid cells) with the clamped permittivity eps0 eps_dc_xx / _yy / _xy (every window cell, unset = 1): T = c S '
         '- e^T E, D = e S + eps E, E = -grad phi, div D = 0 (no free volume charge). Electrode / Electrode (domain) '
         'features fix phi (connection voltage; at least one, the ground, is required) or make an isolated '
         'equipotential carrying a net charge (floating); every other outer face is insulating (D . n = 0). The '
         'coupled operator [[K_uu, K_uphi], [K_phiu, -K_phiphi]] is symmetric but indefinite, so the static solve is '
         'one host SuperLU factorisation (float64, kept while the operator is unchanged; counters piezo_* in the '
         'summary) instead of the pcg linear solver; the eigenfrequency study solves the coupled pencil (no mass on '
         'phi) with the eigen solver: voltage electrodes held at 0 give the short-circuit (resonance) frequencies, '
         'floating ones with charge 0 the open-circuit (antiresonance) frequencies; k_eff^2 = (f_a^2 - f_r^2) / '
         'f_a^2 from two runs. The frequency_response study drives the electrodes harmonically (voltages and charges '
         'as amplitudes, losses mechanical_loss_factor / dielectric_loss_tangent) and its admittance feature gives Y(f) '
         'with a Butterworth-Van Dyke fit (f_s, f_p, k_eff^2, C0, R1, L1, C1). Outputs electric_potential / '
         'electric_field_x / _y and the stress with its -e^T E part (stress_zz with piezo_e13 / _e23). Stiffness: '
         "youngs_modulus / poisson_ratio or the c_ij entries (a {'model': 'crystal'} entry sets c_ij, piezo_e* and "
         'eps_dc*). Not with axial_wavenumber or the optomechanics study; the host factor limits it to ~10^6 dofs (3 '
         'per node).',
    tip='Turns on the piezoelectric effect: the electric potential is solved together with the deformation, and '
        'electrodes apply voltages or measure charges.\n\nNeeds at least one Electrode with a fixed voltage (the '
        'ground). In an eigenfrequency study, fixed-voltage electrodes are shorted and floating ones open.')
_GEOMETRY_PARAM = geometry_param(
    'the cross-section of a structure (out_of_plane decides eps_zz / sigma_zz), forces and reactions per m of depth '
    '(N/m)',
    '(r, z) deformation without torsion (u_phi = 0): unknowns u_r = u_x, u_z = u_y on the cell corners, hoop strain '
    'u_r / r, element integrals weighted by 2 pi r (2 x 2 Gauss), so the operator, loads and reactions are whole rings: '
    'tractions and body forces keep their units (Pa, N/m^3), nodal forces and support reactions are in N (F_y of a '
    'support = the axial force on its ring area, F_x = the integral of its radial traction, not a net vector force). '
    'The _zz outputs are the hoop components (stress_zz = sigma_theta theta, elastic_strain_zz = u_r / r - alpha_zz dT, '
    'alpha_zz = the hoop expansion), _yy the axial ones; u_r = 0 is held on the x_lo nodes automatically. Only u_z needs '
    'a support per solid region (a z translation is the one rigid motion). out_of_plane is not used; not with '
    'piezoelectric, axial_wavenumber or the optomechanics study; an eigenfrequency study finds the axisymmetric '
    '(torsion-free) modes with the whole-ring mass',
    'forces are per metre of depth.',
    'Forces and support reactions are those of the whole ring, in newtons.',
    axis_text='u_r = 0 is held there automatically; a support or load selecting x_lo acts on a line without area: '
              'validation warns')
_OUT_OF_PLANE_PARAM = Param(
    'out_of_plane', 'choice', PLANE_STRAIN, label='Out-of-plane condition', choices=OUT_OF_PLANE_CONDITIONS,
    choice_labels=((PLANE_STRAIN, 'Plane strain (long structure)'), (PLANE_STRESS, 'Plane stress (thin plate)'),
                   (GENERALIZED_PLANE_STRAIN, 'Generalized plane strain (long, free ends)')),
    visible_if=(('geometry', ('planar',)),),
    help='Planar geometry only (axisymmetric ignores it): what the drawing assumes along z. plane_strain (default): eps_zz '
         '= 0, a cross-section of a structure long along z with clamped ends; stress_zz = lambda (eps_xx + eps_yy) - '
         'beta_z dT is the stress that holds eps_zz at 0 (it overestimates the thermal stress of a waveguide whose ends '
         'are free). plane_stress: sigma_zz = 0, a thin plate in the drawing plane (top view of a film or membrane): the '
         'reduced stiffness lambda -> 2 mu lambda / (lambda + 2 mu) and thermal stress beta - lambda beta_z / (lambda + 2 '
         'mu); outputs elastic_strain_zz = eps_zz - alpha_zz dT with eps_zz = (beta_z dT - lambda (eps_xx + eps_yy)) / '
         '(lambda + 2 mu) per cell, stress_zz = 0; forces per m of thickness (N/m). generalized_plane_strain (COMSOL\'s '
         'form): eps_zz = e0 + e1 x + e2 y over the whole drawn cross-section (x, y of the drawing), e0 set by the net '
         'axial force (setting axial_force; 0 = free ends) and the bending terms e1 / e2 (setting bending, default both '
         'free) by the bending moments (bending_moment_x / _y; 0 = an unloaded, freely bending long body): the right '
         'model of thermal stress in a long waveguide or fibre; solved by superposition (one pcg solve of the eps_zz = '
         '0 problem plus, per assembly, one per free term; on a triangle mesh back substitutions on the K factor), '
         'the force and moment conditions hold exactly in the elements; outputs stress_zz and elastic_strain_zz = '
         'eps_zz - alpha_zz dT. Piezoelectric, '
         'axial_wavenumber and the optomechanics study need plane_strain; the eigenfrequency study needs plane_strain '
         'or plane_stress.',
    tip='What happens along the depth of the drawing.\n\n'
        '- Plane strain: a long structure held at its ends (a waveguide cross-section).\n'
        '- Plane stress: a thin plate lying in the drawing (seen from the top).\n'
        '- Generalized plane strain: a long structure whose ends are free to expand, for example a heated waveguide '
        'or fibre.')
_THICKNESS_PARAM = Param(
    'thickness', 'float', None, label='Thickness', unit='m', min=0.0, optional=True,
    visible_if=(('geometry', ('planar',)),),
    help='Planar geometry only: the out-of-plane thickness d of the structure (a plate, or the length of a long '
         'part). Empty '
         '(default): everything per m of depth, support reactions in N/m. A value: the support reaction tables are '
         'for that thickness in N (the per-m forces times d); the solution, tractions (Pa) and body forces (N/m^3) do '
         'not change, nor do axial_force (always the total on the cross-section), electrode charges (C/m) and the '
         'optomechanical length. Typical for plane stress: the film thickness.',
    tip='Thickness of the structure perpendicular to the drawing. Leave empty to get forces per metre of depth; with '
        'a value, support forces are given in newtons for that thickness.')
_AXIAL_FORCE_PARAM = Param(
    'axial_force', 'float', 0.0, label='Axial force', unit='N',
    visible_if=(('out_of_plane', (GENERALIZED_PLANE_STRAIN,)),),
    help='Generalized plane strain only: the net force along +z on the drawn cross-section (integral of stress_zz over '
         'the solid cells; a mirrored model: of the drawn part only). 0 = free ends (no net axial load); a positive '
         'value pulls. Ignored by the other conditions.',
    tip='Total pulling force along the length of the structure, on the drawn cross-section. 0 means the ends are '
        'free.')
_BENDING_PARAM = Param(
    'bending', 'choice', 'xy', label='Out-of-plane bending', choices=BENDING_CHOICES,
    choice_labels=(('xy', 'Free about both axes'), ('x', 'Only varying along x'), ('y', 'Only varying along y'),
                   ('none', 'None (uniform out-of-plane strain)')),
    visible_if=(('out_of_plane', (GENERALIZED_PLANE_STRAIN,)),),
    help="Generalized plane strain only: which linear terms of eps_zz = e0 + e1 x + e2 y are solved. 'xy' (default): "
         "e1 and e2, set by bending_moment_y and bending_moment_x; 'x': e1 only (e2 = 0), 'y': e2 only (e1 = 0); "
         "'none': uniform eps_zz = e0. A mirrored model draws half the section and the moments and force are those of "
         'the drawn part: hold the term across a mirror plane at 0 (a mirror on an x face: bending y; on a y face: '
         "bending x; on both: none), since the whole body's eps_zz is even there. Each free term adds one pcg solve per "
         'assembly.',
    tip='Whether the long structure may bend out of the drawing. Free bending suits a long body with free ends; with '
        'a mirror plane, keep the strain from varying across that plane.')
_BENDING_MOMENT_X_PARAM = Param(
    'bending_moment_x', 'float', 0.0, label='Bending moment about x', unit='N m',
    visible_if=(('out_of_plane', (GENERALIZED_PLANE_STRAIN,)),),
    help='Generalized plane strain with bending xy or y: M_x = integral of y stress_zz dA over the solid cells of the '
         'drawn cross-section, about the drawing origin (x = y = 0, the lower left domain corner); an axial force F '
         'through the point (x_F, y_F) gives M_x = y_F F. 0 (default) = no applied moment. Ignored otherwise.',
    tip='Moment about the x axis through the origin of the drawing applied at the ends of the long structure. 0 '
        'means no bending load.')
_BENDING_MOMENT_Y_PARAM = Param(
    'bending_moment_y', 'float', 0.0, label='Bending moment about y', unit='N m',
    visible_if=(('out_of_plane', (GENERALIZED_PLANE_STRAIN,)),),
    help='Generalized plane strain with bending xy or x: M_y = -(integral of x stress_zz dA) over the solid cells of '
         'the drawn cross-section, about the drawing origin (right-hand rule: an axial force F through (x_F, y_F) gives '
         'M_y = -x_F F). 0 (default) = no applied moment. Ignored otherwise.',
    tip='Moment about the y axis through the origin of the drawing applied at the ends of the long structure. 0 '
        'means no bending load.')


def _pml_params():
    """pml_<face> per outer face, pml_cells and pml_reflection (elastic PML ring: triangles, eigenfrequency)."""
    faces = tuple(Param(
        f'pml_{face}', 'bool', False, label=f'PML on {face}',
        help=(f'Triangle mesh, eigenfrequency study only: an elastic perfectly matched layer beyond {face}. The mesh grows '
              'a ring of pml_cells x the mesh dx there (the MeshSpec pml_layers element layers across), its material '
              'continued from the window edge, with the complex coordinate stretch s = 1 - j beta (d / w)^2 (fem.pml; d '
              'the depth, w the thickness; axisymmetric: the stretched radius r~ beyond x_hi), beta set by '
              'pml_reflection at the target frequency for the fastest wave sqrt(c_II / rho) of the ring; its outer side '
              'is clamped. Waves leaving the window through that face are absorbed, so the modes are complex: '
              'frequency_imag > 0 and the radiation Q = f / (2 frequency_imag) in the Resonance table (mode shapes '
              'shown as their real part and |u|); modes with more than half of their |u|^2 in the ring are PML modes '
              'and dropped. The pencil stays linear in omega^2 (beta is frequency independent); absorption weakens for '
              'modes far below the target. With stress_stiffening the static prestress state treats the ring as an '
              'ordinary (unstretched) elastic continuation clamped at its outer side, and its K_sigma is stretched '
              f'like K. Supports and loads must stay off {face} (the inner side of the ring); not on x_lo of an '
              'axisymmetric solid (the axis). Needs target_frequency > 0.'),
        tip=(f'Absorbs vibrations leaving through {face}, as if the solid continued forever there.\n\nGives each '
             'resonance a radiation quality factor. Only for an eigenfrequency study on a triangle mesh.'))
        for face in FACES)
    cells = Param('pml_cells', 'int', DEFAULT_PML_CELLS, label='PML thickness (elements)', unit='1', min=1, max=100,
                  help='Thickness of the elastic PML ring in multiples of the mesh dx (h_max) on every pml_<face> '
                       'face (the MeshSpec pml_layers sets the element layers across it); the profile needs about 6-10 '
                       'layers, and the ring should span a fair part of the longest wavelength (c / f at the target) '
                       'for a low reflection.',
                  tip='How thick the absorbing layer is, in element sizes.\n\n8 is typical.')
    reflection = Param('pml_reflection', 'float', DEFAULT_PML_REFLECTION, label='PML reflection', unit='1', min=0.0,
                       min_exclusive=True, max=1.0,
                       help='Theoretical round-trip reflection of the elastic PML for the fastest wave at the target '
                            'frequency (sets beta = 3 ln(1 / R) c / (2 omega_t w)); slower waves are absorbed more. A '
                            'very small value makes the profile steep, which the discretisation reflects instead: '
                            '1e-4 to 1e-8 typical.',
                       tip='How strongly the absorbing layer damps a wave crossing it and back.\n\n1e-6 is typical.')
    return faces + (cells, reflection)


def _pml_faces(settings_values) -> tuple:
    """Outer faces whose pml_<face> setting is on."""
    return tuple(face for face in FACES if settings_values and settings_values.get(f'pml_{face}'))


def _azimuthal_order(settings_values) -> int:
    """The azimuthal order m the solid solves (0 unless geometry axisymmetric)."""
    return azimuthal_order_of(settings_values) if is_axisymmetric(settings_values) else 0


def _domain_requirements(settings_values):
    """The PML ring padding (pml_cells x dx on the pml_<face> faces of a triangle mesh); none without a PML face."""
    faces = _pml_faces(settings_values)
    if not faces:
        return DomainRequirements()
    return DomainRequirements(boundary_config=pml_padding(faces, int(settings_values.get('pml_cells')
                                                                     or DEFAULT_PML_CELLS)))


# ---- feature texts ----
_SUPPORT_RESULT_TEXT = (
    "Result 'table' of the latest solve with one row [feature, F_x, F_y] (units '', N/m, N/m; with the thickness "
    'setting N, N for that thickness; geometry axisymmetric: N, N of the whole ring) and the extras F_x and F_y: the '
    'force per m of depth the support exerts on the solid (sum '
    'of K u - f over its held dofs, so components it does not hold read 0); a dof held by two supports counts in both '
    '(the later value holds it). After a frequency_response study on a triangle mesh: a table of the complex reaction '
    'per solved frequency instead, columns frequency, F_x_re, F_x_im, F_y_re, F_y_im (Hz and the force unit above).')
_FACE_ONLY_TEXT = ('Outer faces only (a support on an object boundary: Fixed (domain) on the object). Holds every '
                   'node of the face, corners included.')


# ---- feature apply / result callbacks (worker thread, GPU) ----
def _support(solver, feature_spec, mask):
    """SupportRegion of the held dof mask, registered with a triangle solver (complex reactions per frequency)."""
    return register_support(solver, SupportRegion(feature_spec.name, mask))


def _apply_fixed(ctx, solver, feature_spec, values, selection):
    """u = 0 on the selected outer faces (u_z too in a guided modal solve)."""
    return _support(solver, feature_spec, union_masks([solver.hold_face(face, 0.0, 0.0, u_z=0.0)
                                                       for face in selection.outer_faces]))


def _apply_roller(ctx, solver, feature_spec, values, selection):
    """Normal displacement 0 on the selected outer faces (u_x on x faces, u_y on y faces)."""
    masks = [solver.hold_face(face, u_x=0.0) if face in ('x_lo', 'x_hi') else solver.hold_face(face, u_y=0.0)
             for face in selection.outer_faces]
    return _support(solver, feature_spec, union_masks(masks))


def _apply_displacement(ctx, solver, feature_spec, values, selection):
    """Prescribed u_x and / or u_y (None = free) on the selected outer faces; the held values follow a reused sweep
    build (triangle meshes)."""
    held = union_masks([solver.hold_face(face, values['u_x'], values['u_y']) for face in selection.outer_faces])
    follow_held_values(ctx, solver, feature_spec, values, _PLANE_DISPLACEMENTS, held)
    return _support(solver, feature_spec, held)


def _apply_fixed_domain(ctx, solver, feature_spec, values, selection):
    """u = 0 on every corner node of the selected cells."""
    return _support(solver, feature_spec, solver.hold_cells(ctx.domain_mask(selection.entities)))


def _apply_prescribed_displacement_domain(ctx, solver, feature_spec, values, selection):
    """Prescribed u_x / u_y (None = free) on every dof of the selected elements (triangle meshes; u_phi of an azimuthal
    order stays free); the held values follow a reused sweep build."""
    held = solver.hold_cells(ctx.domain_mask(selection.entities), u_x=values['u_x'], u_y=values['u_y'], u_z=None)
    follow_held_values(ctx, solver, feature_spec, values, _PLANE_DISPLACEMENTS, held)
    return _support(solver, feature_spec, held)


def _apply_boundary_load(ctx, solver, feature_spec, values, selection):
    """Traction on the selected outer faces; follows a reused sweep build (triangle meshes: the change merges into the
    faces' traction entry)."""
    def add_traction(t_x, t_y):
        for face in selection.outer_faces:
            solver.add_face_traction(face, t_x, t_y)
    add_traction(values['t_x'], values['t_y'])
    follow_load(ctx, solver, feature_spec, values, _PLANE_TRACTIONS, add_traction)
    return None


def _apply_segment_load(ctx, solver, feature_spec, values, selection):
    """Traction on every selected (axis-parallel) line, on its nearest node line; a triangle mesh: on the line's mesh
    edges (any direction). Follows a reused sweep build."""
    if ctx.domain.grid_kind == _TRIANGLES:
        edges = ctx.boundary_edges(selection.objects)

        def add_traction(t_x, t_y):
            solver.add_edge_traction(edges, t_x, t_y)
    else:
        lines = [ctx.line(name) for name in selection.objects]

        def add_traction(t_x, t_y):
            for x0, y0, x1, y1 in lines:
                solver.add_segment_traction((x0, y0), (x1, y1), t_x, t_y)
    add_traction(values['t_x'], values['t_y'])
    follow_load(ctx, solver, feature_spec, values, _PLANE_TRACTIONS, add_traction)
    return None


def _validate_segment_load(vctx, physics_spec, feature_spec, values):
    """Errors for slanted lines; a warning for a line off every node line by more than a hundredth of a cell (it acts
    on the nearest one); none on a triangle mesh (the line's own mesh edges carry it)."""
    size = vctx.domain_size
    if size is None or on_triangles(vctx, physics_spec):
        return []
    issues = []
    path = feature_path(physics_spec, feature_spec)
    for name in vctx.objects(feature_spec.selection):
        shape = vctx.shapes.get(name)
        if shape is None or shape.kind != 'line':
            continue
        orientation = line_orientation(shape.values, *size)
        if orientation is None:
            issues.append(Issue('error', path, f"line '{name}' must be horizontal or vertical (a segment load acts on "
                                               'one node line)'))
            continue
        offset = _node_line_offset(vctx.mesh, shape.values, orientation)
        if offset is not None and offset > 0.01:
            issues.append(Issue('warning', path, f"line '{name}' lies {offset:.2g} cells off the nearest node line "
                                                 '(cell edge); the load acts on that node line'))
    return issues


def _node_line_offset(mesh, line, orientation):
    """Distance in cells from an axis-parallel line to the nearest window cell edge across it; None without a mesh."""
    if mesh is None:
        return None
    sides = mesh.npml_sides
    across, widths = (('y', mesh.dy_cells[sides['y_lo']:mesh.ny - sides['y_hi']]) if orientation == 'horizontal'
                      else ('x', mesh.dx_cells[sides['x_lo']:mesh.nx - sides['x_hi']]))
    edges = np.concatenate(([0.0], np.cumsum(np.asarray(widths, dtype=np.float64))))
    position = 0.5 * (float(line[f'{across}0']) + float(line[f'{across}1']))
    nearest = int(np.argmin(np.abs(edges - position)))
    cell = float(widths[min(nearest, len(widths) - 1)])
    return abs(edges[nearest] - position) / cell


def _apply_body_force(ctx, solver, feature_spec, values, selection):
    """f_x / f_y (finite under the selection only) on the selected cells; uniform values follow a reused sweep build
    (triangle meshes: on a host copy of the element mask)."""
    from ...core.backend import to_host   # lazy: definitions never load the backend at import
    mask = ctx.domain_mask(selection.entities)
    solver.add_body_force(mask, ctx.eval_field(values['f_x'], mask=mask), ctx.eval_field(values['f_y'], mask=mask))
    kept_mask = to_host(mask) if ctx.domain.grid_kind == _TRIANGLES else mask
    follow_load(ctx, solver, feature_spec, values, _PLANE_BODY_FORCES,
                lambda *changes: solver.add_body_force(kept_mask, *changes))
    return None


def _apply_temperature_change(ctx, solver, feature_spec, values, selection):
    mask = ctx.domain_mask(selection.entities)
    solver.add_temperature_change(mask, ctx.eval_field(values['delta_T'], mask=mask))
    return None


def _apply_probe(ctx, solver, feature_spec, values, selection):
    if ctx.domain.grid_kind == _TRIANGLES:   # the mesher puts a node on every point object
        if len(selection.objects) != 1:
            raise ValueError(f"solid probe '{feature_spec.name}' needs exactly one point object, got "
                             f'{len(selection.objects)}')
        return solver.add_node_probe(ctx.point_node(selection.objects[0]), values['component'])
    cells = point_cells(ctx, feature_spec, selection)
    if len(cells) != 1:
        raise ValueError(f"solid probe '{feature_spec.name}' needs exactly one point object, got {len(cells)}")
    ix, iy = cells[0]
    return solver.add_probe(ix, iy, values['component'])


def _probe_result(session, solver, probe, feature_spec, options):
    return probe_result(probe, unit=SolidRenderer.field_infos[probe.field_component].unit)


def _apply_electrode(ctx, solver, feature_spec, values, selection):
    """Electrode on the selected outer faces (None when piezoelectric is off: validation warns)."""
    if not solver.piezoelectric:
        return None
    return solver.add_face_electrode(feature_spec.name, selection.outer_faces, values['connection'], values['voltage'],
                                     values['charge'])


def _apply_electrode_domain(ctx, solver, feature_spec, values, selection):
    """Electrode on every corner node of the selected cells (None when piezoelectric is off)."""
    if not solver.piezoelectric:
        return None
    return solver.add_cell_electrode(feature_spec.name, ctx.domain_mask(selection.entities), values['connection'],
                                     values['voltage'], values['charge'])


def _electrode_result(session, solver, electrode, feature_spec, options):
    if electrode is None:
        raise ValueError(f"electrode '{feature_spec.name}': piezoelectric is off, so it is not part of the solve")
    voltage, charge = solver.electrode_values(electrode)
    return {'kind': 'table', 'columns': ['feature', 'V', 'Q'], 'units': ['', 'V', 'C/m'],
            'rows': [[feature_spec.name, voltage, charge]], 'V': voltage, 'Q': charge}


def _apply_frequency_probe(ctx, solver, feature_spec, values, selection):
    """The complex displacement record at the point object's mesh node (triangles; the grids keep tensor meshes
    out)."""
    if len(selection.objects) != 1:
        raise ValueError(f"frequency probe '{feature_spec.name}' needs exactly one point object, got "
                         f'{len(selection.objects)}')
    return solver.add_frequency_probe(ctx.point_node(selection.objects[0]), feature_spec.name)


def _support_result(session, solver, region, feature_spec, options):
    table = harmonic_support_result(solver, region)
    if table is not None:
        return table
    force_x, force_y = solver.reaction(region)
    unit = solver.force_unit
    return {'kind': 'table', 'columns': ['feature', 'F_x', 'F_y'], 'units': ['', unit, unit],
            'rows': [[region.name, force_x, force_y]], 'F_x': force_x, 'F_y': force_y}


# ---- validation helpers (CPU only) ----
def _validate_displacement(vctx, physics_spec, feature_spec, values):
    if values.get('u_x') is None and values.get('u_y') is None:
        return [Issue('error', feature_path(physics_spec, feature_spec),
                      'sets neither u_x nor u_y, so it holds nothing: set at least one (empty leaves a direction '
                      'free)')]
    return []


def _held_components(vctx, feature):
    """[(face, component)] an enabled support feature holds on outer faces ([('domain', 'x'), ('domain', 'y')] for
    fixed_domain, per set component for prescribed_displacement_domain); [] when its values or selection do not resolve
    (reported elsewhere)."""
    if feature.type == 'fixed_domain':
        return [('domain', 'x'), ('domain', 'y')]
    if feature.type == 'prescribed_displacement_domain':
        values = vctx.values(SOLID_MECHANICS.feature_type(feature.type).params, feature.params)
        return [] if values is None else [('domain', component) for component in _COMPONENT_NAMES
                                          if values[f'u_{component}'] is not None]
    try:
        faces = [entity for entity in expand_entities(vctx.spec, feature.selection) if entity in FACES]
    except (ValueError, TypeError):
        return []
    if feature.type == 'fixed':
        return [(face, component) for face in faces for component in _COMPONENT_NAMES]
    if feature.type == 'roller':
        return [(face, 'x' if face in ('x_lo', 'x_hi') else 'y') for face in faces]
    if feature.type != 'displacement':   # a SUPPORT_FEATURES type the 2D solid has not registered (3D only)
        return []
    values = vctx.values(SOLID_MECHANICS.feature_type('displacement').params, feature.params)
    if values is None:
        return []
    return [(face, component) for face in faces for component in _COMPONENT_NAMES
            if values[f'u_{component}'] is not None]


def _free_body_allowed(study_type, settings_values) -> bool:
    """A free body may be solved (triangle meshes): the eigenfrequency / optomechanics study at target_frequency > 0
    (its 0 Hz modes dropped) or the frequency_response study (regular at f > 0)."""
    return bool((study_type in _MODAL_STUDIES and settings_values.get('target_frequency'))
                or study_type == FREQUENCY_RESPONSE_STUDY)


def _support_issues(vctx, physics_spec, path, axial_wavenumber=None, guided_by_study=False, axisymmetric=False,
                    free_body=False, held_elsewhere=False):
    """Error without a held u_x or u_y anywhere (rigid translation; none for guided modes at k_z > 0, incl. the
    study-set backward k_z, at k_z = 0 also without a held u_z; axisymmetric: u_y only, the axis holds u_r; free_body
    (triangles, _free_body_allowed): an info instead; held_elsewhere: none, a clamped PML ring or an azimuthal order m
    >= 2 leaves no rigid motion); warning per face component held by several features and per boundary load on a fully
    held face."""
    holders = {}
    for feature in physics_spec.features:
        if feature.enabled and feature.type in SUPPORT_FEATURES:
            for key in _held_components(vctx, feature):
                holders.setdefault(key, []).append(feature.name)
    held_directions = {component for _, component in holders}
    if axisymmetric:
        held_directions.add('x')   # u_r = 0 on the axis; a radial translation strains the hoop, so it is no rigid mode
    issues = []
    if axial_wavenumber == 0.0 and not any(feature.enabled and feature.type in ('fixed', 'fixed_domain')
                                           for feature in physics_spec.features):
        issues.append(Issue('error', path,
                            'axial_wavenumber 0: the antiplane (u_z) modes need u_z held, else the solid translates '
                            'freely along z: add a Fixed face or a Fixed (domain) region (Roller and Displacement leave '
                            'u_z free), or set axial_wavenumber > 0'))
    # guided k_z > 0 (backward Brillouin: 2 Re beta > 0): no rigid motion
    supports_optional = (guided_by_study or (axial_wavenumber is not None and axial_wavenumber > 0.0)
                         or held_elsewhere)
    if free_body and held_directions != set(_COMPONENT_NAMES) and not held_elsewhere:
        free = [component for component in _COMPONENT_NAMES if component not in held_directions]
        consequence = ('regular at every frequency > 0 (its rigid motion follows the drive as a mass)'
                       if vctx.study_type == FREQUENCY_RESPONSE_STUDY else
                       'its rigid-body modes (0 Hz, below 1e-4 x target_frequency) are dropped')
        issues.append(Issue('info', path, f"no support holds u_{', u_'.join(free)}: a free body, {consequence}"))
        supports_optional = True
    if not held_directions and not supports_optional:
        issues.append(Issue('error', path,
                            'a static or modal solve needs supports: add Fixed, Roller or Displacement on outer faces or '
                            'a Fixed '
                            '(domain) region (otherwise the solid can move as a rigid body and the system is '
                            'singular)'))
    elif held_directions != set(_COMPONENT_NAMES) and not supports_optional:
        free = next(component for component in _COMPONENT_NAMES if component not in held_directions)
        issues.append(Issue('error', path,
                            f'no support holds u_{free}, so the solid can slide along {free}: add a Roller on a '
                            f"{free}_lo / {free}_hi face, a Fixed face or a Displacement with u_{free} set"))
    for (face, component), names in sorted(holders.items()):
        if face != 'domain' and len(names) > 1:
            issues.append(Issue('warning', path,
                                f"u_{component} on outer face {face} is held by features {', '.join(names)}; the last "
                                f"one ('{names[-1]}') sets its value"))
    for feature in physics_spec.features:
        if not feature.enabled or feature.type != 'boundary_load':
            continue
        try:
            faces = [entity for entity in expand_entities(vctx.spec, feature.selection) if entity in FACES]
        except (ValueError, TypeError):
            continue
        for face in faces:
            if (face, 'x') in holders and (face, 'y') in holders:
                issues.append(Issue('warning', path,
                                    f"boundary load '{feature.name}' acts on face {face}, whose u_x and u_y are both "
                                    'held: the load goes straight into the reaction and moves nothing'))
    return issues


_ELECTRODE_FEATURES = ('electrode', 'electrode_domain')


def _piezo_issues(vctx, physics_spec, settings_values, path):
    """Electrodes without piezoelectric (warning); with it: errors for the optomechanics study, axial_wavenumber and no
    voltage electrode; warnings for no material with piezo_e* and per piezo material without youngs_modulus."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    electrodes = [feature for feature in physics_spec.features
                  if feature.enabled and feature.type in _ELECTRODE_FEATURES]
    if not settings_values.get('piezoelectric'):
        if electrodes:
            return [Issue('warning', path, f"electrode feature(s) {', '.join(feature.name for feature in electrodes)} "
                                           'are ignored: the piezoelectric setting is off')]
        return []
    issues = []
    if vctx.study_type == OPTOMECHANICS_STUDY:
        issues.append(Issue('error', path, 'piezoelectric: the optomechanics study is not modelled for a piezoelectric '
                                           'solid (its modes carry the potential); turn piezoelectric off'))
    if vctx.study_type == EIGENFREQUENCY_STUDY and settings_values.get('axial_wavenumber') is not None:
        issues.append(Issue('error', path, 'piezoelectric has no guided (2.5D) modes: leave axial_wavenumber empty'))
    grounded = False
    for feature in electrodes:
        values = vctx.values(SOLID_MECHANICS.feature_type(feature.type).params, feature.params)
        grounded = grounded or values is None or values['connection'] == 'voltage'
    if not grounded:
        issues.append(Issue('error', path, 'piezoelectric needs at least one Electrode or Electrode (domain) with '
                                           'connection voltage (0 V = ground): otherwise the potential has no '
                                           'reference and the coupled system is singular'))
    piezo_materials = []
    for material in vctx.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:   # reported at the material
            piezo_materials.append(material.name)
            continue
        if not (material.selection and any(properties.get(name) is not None for name in PIEZO_ARRAYS)):
            continue
        piezo_materials.append(material.name)
        stiff = any(properties.get(name) is not None and (plain_value(properties.get(name), vctx.parameters) is None
                                                          or plain_value(properties.get(name), vctx.parameters) > 0.0)
                    for name in ('youngs_modulus', 'c11'))   # a non-plain expression counts as stiff
        if not stiff:
            issues.append(Issue('warning', ('material', material.name),
                                f"material '{material.name}' sets piezo_e* but no youngs_modulus or c11 > 0: "
                                f'{physics_spec.name} treats it as void (no stiffness, no coupling): set '
                                "youngs_modulus / poisson_ratio, the c_ij or a {'model': 'crystal'} entry"))
    if not piezo_materials:
        issues.append(Issue('warning', path, 'piezoelectric is on but no selected material sets a piezo_e* entry: the '
                                             'potential only adds a dielectric, nothing couples'))
    return issues


def _formulation_issues(vctx, physics_spec, settings_values, path, axial_wavenumber):
    """geometry / out_of_plane rules: piezoelectric, guided modes (axial_wavenumber in an eigenfrequency study) and the
    planar optomechanics study need plane strain, the eigenfrequency study refuses generalized plane strain (errors);
    axisymmetric: the shared (r, z) rules (optomechanics: the em_fdfd_bor pairing of the feature) and an info for a
    non-default out_of_plane (ignored)."""
    axisymmetric = is_axisymmetric(settings_values)
    condition = settings_values.get('out_of_plane', PLANE_STRAIN)
    if axisymmetric:
        text = 'geometry axisymmetric'
    elif condition != PLANE_STRAIN:
        text = f'out_of_plane {condition}'
    else:
        return []
    issues = []
    if settings_values.get('piezoelectric'):
        issues.append(Issue('error', path, f'piezoelectric needs planar plane strain, not {text}'))
    if axial_wavenumber is not None:
        issues.append(Issue('error', path, f'guided (2.5D) modes (axial_wavenumber) need planar plane strain, not '
                                           f'{text}: leave axial_wavenumber empty'))
    if vctx.study_type == OPTOMECHANICS_STUDY and not axisymmetric:   # axisymmetric: em_fdfd_bor pairing (feature)
        issues.append(Issue('error', path, f'the optomechanics study needs planar plane strain, not {text} (the overlap '
                                           'assumes eps_zz = 0)'))
    if vctx.study_type == EIGENFREQUENCY_STUDY and condition == GENERALIZED_PLANE_STRAIN and not axisymmetric:
        issues.append(Issue('error', path, 'generalized plane strain has no vibration modes (the uniform eps_zz carries '
                                           'no mass): use plane_strain or plane_stress in an eigenfrequency study'))
    if axisymmetric:
        issues.extend(axisymmetric_issues(vctx, physics_spec, path, 'solid mechanics',
                                          axis_consequence='u_r = 0 is held there already and the axis has no area, '
                                                           'so a support or load on it acts on a line'))
        if condition != PLANE_STRAIN:
            issues.append(Issue('info', path, f'out_of_plane {condition} is ignored in geometry axisymmetric (the hoop '
                                              'strain u_r / r takes its place)'))
    return issues


def _triangle_issues(vctx, physics_spec, settings_values, path):
    """A solid on a triangle mesh (solver_fem): errors for piezoelectric, guided modes, a study outside _FEM_STUDIES, a
    planar optomechanics study, a strain-optic coupling from it and a 'linear' solver other than sparse_direct (SOLID_LINEAR_ROLE keeps cusolver_qr out of
    its back_end choices); the host SuperLU size rule of its dofs (skipped with a cuDSS eigen factor in a modal study
    or a cuDSS 'linear' back_end in the other studies)."""
    from ...solvers.sparse_direct import CUDSS_BACK_ENDS, SPLU_LIMIT_UNKNOWNS, SPLU_WARN_UNKNOWNS   # numpy only
    issues = []
    refused = 'is not available on a triangle mesh yet (use a tensor mesh for it)'
    if settings_values.get('piezoelectric'):
        issues.append(Issue('error', path, f'piezoelectric {refused}'))
    if vctx.study_type == EIGENFREQUENCY_STUDY and settings_values.get('axial_wavenumber') is not None:
        issues.append(Issue('error', path, f'guided (2.5D) modes (axial_wavenumber) {refused}'))
    if vctx.study_type not in _FEM_STUDIES:
        issues.append(Issue('error', path, f'the {vctx.study_type} study of solid_mechanics {refused}; a triangle mesh '
                                           f"runs {', '.join(_FEM_STUDIES)}"))
    if vctx.study_type == OPTOMECHANICS_STUDY and not is_axisymmetric(settings_values):
        issues.append(Issue('error', path, 'the optomechanics study of a planar solid on a triangle mesh is not '
                                           'available (the triangle solid pairs with em_fdfd_bor optics in geometry '
                                           'axisymmetric): use a tensor mesh or geometry axisymmetric'))
    for coupling in vctx.spec.couplings:
        if coupling.enabled and coupling.type == STRAIN_OPTIC_COUPLING and coupling.source == physics_spec.name:
            issues.append(Issue('error', ('coupling', coupling.name),
                                f"{STRAIN_OPTIC_COUPLING} from '{physics_spec.name}' on a triangle mesh {refused}"))
    solvers = vctx.solvers(physics_spec) or {}
    resolved = solvers.get('linear')
    if resolved is not None and resolved.solver_type.name != 'sparse_direct':   # back_end: SOLID_LINEAR_ROLE's choices
        issues.append(Issue('error', path, f"linear solver '{resolved.node or resolved.solver_type.name}' is "
                                           f'{resolved.solver_type.name}: on a triangle mesh the solid solves with '
                                           'sparse_direct (host SuperLU, or cuDSS by its opt-in back ends): assign '
                                           'Automatic or a sparse_direct node'))
    mesh = vctx.mesh
    if vctx.study_type in _MODAL_STUDIES:
        cudss_factor = eigen_factorization(solvers) in CUDSS_FACTORIZATIONS   # cuDSS guards its device memory
    else:
        cudss_factor = resolved is not None and resolved.values.get('back_end') in CUDSS_BACK_ENDS
    if mesh is not None and getattr(mesh, 'kind', '') == _TRIANGLES and not cudss_factor:
        library = factor_label(host_direct_back_end(solvers, role='eigen' if vctx.study_type in _MODAL_STUDIES
                                                    else 'linear'))
        per_component = mesh.n_nodes + (mesh.n_edges if settings_values.get('discretization') == 'quadratic' else 0)
        unknowns = _fem_components(settings_values) * per_component
        if unknowns > SPLU_LIMIT_UNKNOWNS:
            remedy = ("eigen factorization 'cudss' on a GPU, " if vctx.study_type in _MODAL_STUDIES
                      else "linear solver back_end 'cudss' on a GPU, ")
            issues.append(Issue('error', path, f'~{unknowns:,} displacement unknowns exceed the {SPLU_LIMIT_UNKNOWNS:,} '
                                               f'limit of the host {library} factor: use {remedy}a coarser triangle '
                                               'mesh or discretization linear'))
        elif unknowns > SPLU_WARN_UNKNOWNS:
            issues.append(Issue('warning', path, f'~{unknowns:,} displacement unknowns exceed ~{SPLU_WARN_UNKNOWNS:,}, '
                                                 f'the practical size of the host {library} factor (memory and time '
                                                 'grow faster than the unknowns)'))
    return issues


def _fem_components(settings_values) -> int:
    """Displacement components per scalar dof on a triangle mesh: 3 for an azimuthal order m >= 1, else 2."""
    return 3 if _azimuthal_order(settings_values) else 2


def _tensor_probe_issues(physics_spec):
    """Rectangular-cell mesh: an error per solid_probe of an energy density (triangle meshes only)."""
    return [Issue('error', feature_path(physics_spec, feature),
                  f"probe component {feature.params.get('component')} exists only on a triangle mesh")
            for feature in physics_spec.features
            if feature.enabled and feature.type == 'solid_probe'
            and feature.params.get('component') in ENERGY_DENSITY_FIELDS]


def _outer_faces(vctx, feature) -> list:
    """Outer faces of a feature's selection ([] when it does not resolve: reported elsewhere)."""
    try:
        return [entity for entity in expand_entities(vctx.spec, feature.selection) if entity in FACES]
    except (ValueError, TypeError):
        return []


def _pml_issues(vctx, physics_spec, settings_values, path, triangles):
    """Elastic PML faces: errors on a rectangular-cell mesh, outside the eigenfrequency study, at target_frequency 0, on
    x_lo of an axisymmetric solid, and per support / load on a PML face."""
    faces = _pml_faces(settings_values)
    if not faces:
        return []
    names = ', '.join(faces)
    if not triangles:
        return [Issue('error', path, f"PML faces ({names}) need a triangle mesh (MeshSpec kind 'triangles'): the "
                                     'rectangular-cell solid has no elastic PML')]
    issues = []
    if vctx.study_type != EIGENFREQUENCY_STUDY:
        issues.append(Issue('error', path, f'an elastic PML ({names}) is solved by the {EIGENFREQUENCY_STUDY} study only; '
                                           f'the {vctx.study_type} study is not available with it: turn pml_* off'))
    elif not settings_values.get('target_frequency'):
        issues.append(Issue('error', path, 'an elastic PML needs target_frequency > 0: its strength is set for the '
                                           'waves at the target'))
    if is_axisymmetric(settings_values) and 'x_lo' in faces:
        issues.append(Issue('error', path, 'pml_x_lo: an axisymmetric solid has its axis (r = 0) on x_lo, which takes no '
                                           'PML: turn pml_x_lo off'))
    for feature in physics_spec.features:
        if not (feature.enabled and (feature.type in SUPPORT_FEATURES or feature.type == 'boundary_load')):
            continue
        held = [face for face in _outer_faces(vctx, feature) if face in faces]
        if held:
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f"{feature.type} on the PML face(s) {', '.join(held)}: that face is the inner side of "
                                'the absorbing ring, a hold or load there would reflect the waves; select other faces'))
    return issues


def _anisotropic_materials(vctx) -> list:
    """Names of the selected materials that set a c_ij entry (or a crystal entry expanding to them)."""
    return materials_setting(vctx, STIFFNESS_TENSOR_ARRAYS)


def _azimuthal_issues(vctx, physics_spec, settings_values, path, triangles):
    """azimuthal_order m >= 1: the shared rules (azimuthal_order_issues), then errors on a rectangular-cell mesh,
    outside the eigenfrequency / optomechanics studies and per material with c_ij entries (isotropic only)."""
    issues = azimuthal_order_issues(vctx, physics_spec, path, settings_values)
    order = _azimuthal_order(settings_values)
    if not order:
        return issues
    if not triangles:
        issues.append(Issue('error', path, f"azimuthal_order {order} needs a triangle mesh (MeshSpec kind 'triangles'): "
                                           'the rectangular-cell solid solves m = 0 only'))
        return issues
    if vctx.study_type not in _MODAL_STUDIES:
        issues.append(Issue('error', path, f"azimuthal_order {order} is solved by the {' / '.join(_MODAL_STUDIES)} "
                                           f'studies only (the {vctx.study_type} study would need cos(m phi) loads): '
                                           'set it to 0'))
    issues.extend(Issue('error', ('material', name),
                        f"material '{name}' sets c_ij entries (or a crystal entry): azimuthal_order {order} takes "
                        'isotropic youngs_modulus / poisson_ratio only (the r-phi and z-phi shear moduli are not part of '
                        'the 2D vocabulary)')
                  for name in _anisotropic_materials(vctx))
    return issues


def _nonlinear_issues(vctx, physics_spec, settings_values, path, triangles, static_refused):
    """geometric_nonlinearity: errors on a rectangular-cell mesh, with piezoelectric, an azimuthal order m >= 1, an
    elastic PML and generalized plane strain; infos where it has no nonlinear static solve (static_refused: a free body)
    and in the frequency_response study (linear about the state of the initial stress)."""
    if not settings_values.get('geometric_nonlinearity'):
        return []
    if not triangles:
        return [Issue('error', path, "geometric_nonlinearity needs a triangle mesh (MeshSpec kind 'triangles'): the "
                                     'rectangular-cell solid is linear (small strain) only')]
    issues = []
    if settings_values.get('piezoelectric'):
        issues.append(Issue('error', path, 'geometric_nonlinearity is not available with piezoelectric on (the coupled '
                                           'operator has no total-Lagrangian form): turn one of them off'))
    order = _azimuthal_order(settings_values)
    if order:
        issues.append(Issue('error', path, f'geometric_nonlinearity is not available with azimuthal_order {order} (its '
                                           'cos(m phi) amplitudes have no static solve to linearise about): set '
                                           'azimuthal_order 0 or turn geometric_nonlinearity off'))
    if _pml_faces(settings_values):
        issues.append(Issue('error', path, 'geometric_nonlinearity is not available with an elastic PML (pml_*): the '
                                           'absorbing ring has no nonlinear static state; turn pml_* or '
                                           'geometric_nonlinearity off'))
    if (not is_axisymmetric(settings_values)
            and settings_values.get('out_of_plane', PLANE_STRAIN) == GENERALIZED_PLANE_STRAIN):
        issues.append(Issue('error', path, 'geometric_nonlinearity is not available with generalized plane strain (the '
                                           'eps_zz superposition is linear): use plane_strain, plane_stress or '
                                           'geometry axisymmetric'))
    if static_refused:
        issues.append(Issue('info', path, f'geometric_nonlinearity: {static_refused} has no static solve, so the '
                                          'pencil takes K + K_sigma of the initial stress (the tangent at u = 0)'))
    elif vctx.study_type == FREQUENCY_RESPONSE_STUDY:
        issues.append(Issue('info', path, 'geometric_nonlinearity: the frequency response stays linear (small '
                                          'vibrations) about the nonlinear static state of the initial stress and '
                                          'thermal strain alone; the loads are its harmonic drive'))
    return issues


def _validate_solid(vctx, physics_spec, settings_values):
    """Physics-level rules (in the study the solid takes: a background physics of StudySpec.solved_physics sees
    'stationary'): supports, CG iteration limit (tensor) or the triangle-mesh rules, materials, eigenfrequency
    densities, the guided axial_wavenumber (eigenfrequency study only, refused by optomechanics, which sets it for
    backward scattering), prestress, the elastic PML and the azimuthal order (triangles, eigenfrequency), geometric
    nonlinearity (triangles)."""
    vctx = background_context(vctx, physics_spec)
    path = ('physics', physics_spec.name)
    triangles = on_triangles(vctx, physics_spec)
    axial_wavenumber = settings_values.get('axial_wavenumber')
    issues = []
    free_body = triangles and _free_body_allowed(vctx.study_type, settings_values)
    condition = settings_values.get('out_of_plane', PLANE_STRAIN)
    refused_zz = None
    if not is_axisymmetric(settings_values) and condition == PLANE_STRESS:
        refused_zz = f'out_of_plane {condition} (sigma_zz = 0)'
    held = any(feature.enabled and feature.type in SUPPORT_FEATURES for feature in physics_spec.features)
    order = _azimuthal_order(settings_values) if triangles else 0
    pml = bool(_pml_faces(settings_values)) and triangles
    static_refused = ''
    if order:
        static_refused = f'azimuthal order {order}'
    elif free_body and not held and not pml:
        static_refused = 'a free body'
    issues.extend(prestress_issues(vctx, physics_spec, settings_values, path, triangles, refused_zz=refused_zz,
                                   static_refused=static_refused))
    issues.extend(_pml_issues(vctx, physics_spec, settings_values, path, triangles))
    issues.extend(_azimuthal_issues(vctx, physics_spec, settings_values, path, triangles))
    issues.extend(_nonlinear_issues(vctx, physics_spec, settings_values, path, triangles, static_refused))
    if triangles and condition == GENERALIZED_PLANE_STRAIN and not is_axisymmetric(settings_values):
        issues.extend(Issue('error', ('material', name), f"material '{name}' sets an initial stress: generalized "
                                                         'plane strain does not take one (use plane_strain, '
                                                         'plane_stress or axisymmetric)')
                      for name in initial_stress_materials(vctx))
    if not triangles:
        issues.extend(Issue('error', path, f"stored field '{physics_spec.name}.{name}' exists on a triangle mesh only "
                                           '(the rectangular-cell solid stores the real parts and u_abs)')
                      for name in HARMONIC_IMAG_FIELDS
                      if f'{physics_spec.name}.{name}' in (vctx.spec.study.stored_fields or []))
    if axial_wavenumber is not None and vctx.study_type != EIGENFREQUENCY_STUDY:
        if vctx.study_type == OPTOMECHANICS_STUDY:
            issues.append(Issue('error', path, 'axial_wavenumber is set: the optomechanics study solves plane-strain '
                                               'modes, or with an optomechanical_coupling of scattering backward sets '
                                               'k_z = 2 Re beta itself; leave axial_wavenumber empty'))
        else:
            issues.append(Issue('info', path, f'axial_wavenumber is read only by an {EIGENFREQUENCY_STUDY} study: '
                                              f'the {vctx.study_type} study ignores it'))
        axial_wavenumber = None
    guided_by_study = vctx.study_type == OPTOMECHANICS_STUDY and backward_scattering(physics_spec)
    issues.extend(_formulation_issues(vctx, physics_spec, settings_values, path, axial_wavenumber))
    # azimuthal order 1: the sideways translation and tilt need the planar u_x / u_y rule; m >= 2 and a clamped PML ring
    # leave no rigid motion
    issues.extend(_support_issues(vctx, physics_spec, path, axial_wavenumber, guided_by_study,
                                  is_axisymmetric(settings_values) and order != 1, free_body=free_body,
                                  held_elsewhere=order >= 2 or pml))
    if triangles:
        issues.extend(_triangle_issues(vctx, physics_spec, settings_values, path))
        element_name = f"{settings_values.get('discretization', 'quadratic')} triangle"
    else:
        issues.extend(cg_solver_issues(vctx, physics_spec, path))
        issues.extend(_tensor_probe_issues(physics_spec))
        element_name = 'bilinear'
    issues.extend(material_issues(vctx, physics_spec, path, SOLID_TYPE_NAME, element_name, tensor=True))
    issues.extend(modal_study_issues(vctx, physics_spec, path, _MODAL_STUDIES, THERMAL_STRAIN_COUPLING, tensor=True,
                                     stress_stiffening=bool(settings_values.get('stress_stiffening')
                                                            or (triangles
                                                                and settings_values.get('geometric_nonlinearity')))))
    issues.extend(_piezo_issues(vctx, physics_spec, settings_values, path))
    issues.extend(eigen_search_issues(vctx, path, bool(settings_values.get('piezoelectric'))))
    issues.extend(anisotropy_issues(vctx, physics_spec, settings_values, path,
                                    guided=axial_wavenumber is not None or guided_by_study))
    issues.extend(frequency_response_issues(vctx, physics_spec, settings_values, path, THERMAL_STRAIN_COUPLING))
    return issues


# ---- physics factory callbacks ----
def _create_solid(ctx, physics_spec, values):
    """SolidMechanicsPhysics on ctx.domain with the build's linear (pcg) solver (solid_linear_solver), the 'eigen'
    (arpack_shift_invert) runtime in a modal study and, in an eigenfrequency study, the guided axial_wavenumber (an
    optomechanics study with backward scattering: guided, k_z set by the study). A triangle mesh:
    SolidMechanicsFEMPhysics with the 'linear' sparse_direct runtime (mesh plan D11), the azimuthal order, the elastic
    PML faces and the geometric nonlinearity settings."""
    if ctx.domain.grid_kind == _TRIANGLES:
        runtimes = ctx.solvers(physics_spec.name)
        return SolidMechanicsFEMPhysics(ctx.domain, linear_solver=runtimes.get('linear'),
                                        eigen_solver=runtimes.get('eigen'), target_frequency=values['target_frequency'],
                                        geometry=values['geometry'], out_of_plane=values['out_of_plane'],
                                        thickness=values['thickness'], discretization=values['discretization'],
                                        axial_force=values['axial_force'], bending=values['bending'],
                                        bending_moment_x=values['bending_moment_x'],
                                        bending_moment_y=values['bending_moment_y'],
                                        stress_stiffening=values['stress_stiffening'],
                                        free_body=_free_body_allowed(solid_study(ctx.spec, physics_spec.name), values),
                                        azimuthal_order=_azimuthal_order(values), pml_faces=_pml_faces(values),
                                        pml_reflection=values['pml_reflection'],
                                        geometric_nonlinearity=values['geometric_nonlinearity'],
                                        load_steps=values['load_steps'],
                                        nonlinear_tolerance=values['nonlinear_tolerance'],
                                        nonlinear_max_iterations=values['nonlinear_max_iterations'],
                                        tangent_update=values['tangent_update'])
    study = solid_study(ctx.spec, physics_spec.name)   # a background solid of solved_physics: 'stationary'
    guided = values['axial_wavenumber'] if study == EIGENFREQUENCY_STUDY else None
    return SolidMechanicsPhysics(ctx.domain, linear_solver=solid_linear_solver(ctx, physics_spec),
                                 target_frequency=values['target_frequency'],
                                 eigen_solver=ctx.solvers(physics_spec.name).get('eigen'), axial_wavenumber=guided,
                                 guided_by_study=study == OPTOMECHANICS_STUDY
                                 and backward_scattering(physics_spec), piezoelectric=values['piezoelectric'],
                                 geometry=values['geometry'], out_of_plane=values['out_of_plane'],
                                 axial_force=values['axial_force'], thickness=values['thickness'],
                                 bending=values['bending'], bending_moment_x=values['bending_moment_x'],
                                 bending_moment_y=values['bending_moment_y'])


_INT64_BYTES = 8
_STENCIL_ENTRIES_PER_DOF = 18   # 3 x 3 neighbour nodes x 2 components (system.ElasticPattern; boundary rows hold fewer)
_GUIDED_ENTRIES_PER_DOF = 27    # 3 x 3 neighbour nodes x 3 components (guided_modes)
# guided host assembly per CSR entry: pattern (layout bool, column int64, position int64, index / row int32), one live
# stencil (float64), the K and M data (float64)
_GUIDED_ASSEMBLY_BYTES_PER_ENTRY = 1 + 2 * _INT64_BYTES + 2 * INT32_BYTES + 3 * FLOAT64_BYTES


# triangle mesh (solver_fem): local dofs per element (P1 3, P2 6), COO entries summed per CSR entry (uncalibrated), the
# extra SuperLU fill of the vector P1 / P2 stencil over the 2D 5-point host_real_lu_bytes heuristic (uncalibrated:
# compare with summary.json solvers <physics>.roles.linear.counters.host_factor_bytes)
_FEM_LOCAL_DOFS = {'linear': 3, 'quadratic': 6}
_FEM_COO_PER_CSR_ENTRY = 2.0
_FEM_FILL_FACTOR = 3.0
_FEM_NODAL_OUTPUTS = 18   # host float64 nodal display arrays (solver_fem._nodal, energy densities and u_z included)


def _prestress_solved(settings_values, context, with_loads) -> bool:
    """Whether solve_static_state factorises K (or a Newton tangent) on the 'linear' role before the pencils."""
    prestressed = bool(settings_values.get('stress_stiffening') or settings_values.get('geometric_nonlinearity'))
    return _static_state_solved(dict(settings_values, stress_stiffening=prestressed), context, with_loads)


LINEAR_STATIC_STATE_PHASE = 'static state'   # the 'linear' factor solve_static_state releases before the pencils


def factor_fill(mesh, order, unknowns, components, dimensions):
    """The fill argument of cudss_factor.factor_device_bytes for a Lagrange operator of `unknowns` dofs (components per
    node) on an element mesh: a FactorFill counted on the built mesh (MeshEstimate.dissection: nested-dissection node
    pairs at this element order x components^2, scaled by the factored share of the mesh's dofs), else `dimensions`
    (the growth law: size-based counts have no mesh to count on; also used by definition3d)."""
    dissection = getattr(mesh, 'dissection', None)
    if dissection is None:
        return dimensions
    from ...solvers.cudss_factor import FactorFill
    share = min(float(unknowns) / max(components * dissection.n_nodes(order), 1), 1.0)
    return FactorFill(components ** 2 * dissection.node_pairs(order) * share)


def linear_factor_items(dofs, entries, back_end, on_gpu, dimensions, kept=True) -> tuple:
    """(run items, transient items, cuDSS factors) of system_fem.DirectElasticSystem's 'linear' K_ff / tangent factor;
    kept False: solve_static_state releases it before the pencils (phase 'static state'; also used by definition3d)."""
    if not on_gpu:
        return [], [], []
    from ...solvers.cudss_factor import (CUDSS_SOLVE_PHASE, index_map_bytes, refresh_peak_bytes, solve_buffer_bytes,
                                         solve_peak_bytes)
    from ...solvers.sparse_direct import CUDSS_BACK_ENDS
    phase = 'static factorisation' if kept else LINEAR_STATIC_STATE_PHASE
    # _solve_free uploads the reduced K_ff CSR for the factorize call only; the free-dof solution vector stays
    csr_index_bytes = INT32_BYTES * (entries + dofs + 1)
    csr = ('reduced K_ff CSR uploaded for the factor (float64)', FLOAT64_BYTES * entries + csr_index_bytes)
    solution = ('linear free-dof solution vector (float64)', FLOAT64_BYTES * dofs)
    if back_end not in CUDSS_BACK_ENDS:   # host SuperLU of a downloaded copy
        return ([solution], [(phase, *csr)], []) if kept else ([], [(phase, *csr), (phase, *solution)], [])
    # CudssFactorization kept by the DiffusionDirectSolver until the next assembly, planned from the device CSR
    # (factorize_device_csr -> from_device_values(check_symmetry=True)). In the order it allocates: the pattern's index
    # maps; the refresh on the uploaded values (equilibrated copy, int64 keys of the symmetry check:
    # refresh_peak_bytes counts the uploaded values, so only the CSR's index arrays come on top); then the kept arrays:
    # the transpose map, the equilibrated lower triangle + diagonal (their CSR shares the maps' index arrays; mixed: +
    # the float32 factor copy), rhs buffer and row scale, with the equilibrated full values alive through the probe
    mixed = back_end == 'cudss_mixed'
    lower = (entries + dofs) // 2
    pattern_maps = index_map_bytes(dofs, entries, symmetry_check=False)
    transpose_map = index_map_bytes(dofs, entries) - pattern_maps
    values = ('linear cuDSS equilibrated lower-triangle values, diagonal, rhs buffer and row scale',
              (FLOAT64_BYTES + (FLOAT32_BYTES if mixed else 0)) * lower + FLOAT64_BYTES * dofs
              + solve_buffer_bytes(dofs, FLOAT64_BYTES, mixed))
    solving = ('linear cuDSS solve vectors (mixed: the GMRES basis)', solve_peak_bytes(dofs, FLOAT64_BYTES, mixed))
    checking = [('reduced K_ff CSR index arrays (uploaded for the factor)', csr_index_bytes),
                ('linear cuDSS index maps of the planned pattern', pattern_maps),
                ('linear cuDSS first factorisation (uploaded and equilibrated values, symmetry check keys)',
                 refresh_peak_bytes(entries, FLOAT64_BYTES))]
    probing = [csr, ('linear cuDSS index maps of the planned pattern', pattern_maps + transpose_map),
               ('linear cuDSS equilibrated full values held through the probe', FLOAT64_BYTES * entries), values,
               solving]
    larger = max(checking, probing, key=lambda group: sum(size for _, size in group))
    factors = [('linear cuDSS factor of K_ff', dofs, entries, FLOAT64_BYTES, dimensions, mixed)]
    if kept:   # the first factorisation's peak beyond the kept arrays and a solve's vectors are phases of their own
        held = [solution, values, ('linear cuDSS index maps of the planned pattern', pattern_maps + transpose_map)]
        beyond = sum(size for _, size in larger) - sum(size for _, size in held[1:])
        return held, [(phase, 'linear cuDSS factorisation peak beyond the kept arrays (uploaded K_ff CSR, equilibrated '
                              'values, symmetry check keys)', beyond), (CUDSS_SOLVE_PHASE, *solving)], factors
    # released before the pencils: its whole life is one phase, the larger moment of its factorisation
    return [], [(phase, label, size) for label, size in larger + [solution]], factors


def _fem_memory_estimate(settings_values, mesh, context):
    """Triangle mesh: per-element float32 fields, host assembly / factor / eigen items per study, the cuDSS 'linear' and
    'eigen' pool items (kept, or transient per solve) and their factors outside the pool (notes in the body)."""
    discretization = settings_values.get('discretization', 'quadratic')
    local = _FEM_LOCAL_DOFS[discretization]
    elements = int(mesh.n_elements)
    scalar_dofs = int(mesh.n_nodes) + (int(mesh.n_edges) if discretization == 'quadratic' else 0)
    components = _fem_components(settings_values)
    vector_dofs = components * scalar_dofs
    coo_entries = elements * (components * local) ** 2
    entries = int(coo_entries / _FEM_COO_PER_CSR_ENTRY)
    n_fields = len(OUTPUT_ARRAYS) + len(INPUT_ARRAYS)
    csr_entry_bytes = FLOAT64_BYTES + INT32_BYTES
    study = context.spec.study.type
    static = study not in (EIGENFREQUENCY_STUDY, FREQUENCY_RESPONSE_STUDY)
    on_gpu = context.spec.solver_defaults.backend != 'cpu'   # the K_ff CSR goes to the device for the 'linear' factor
    from ...solvers.sparse_direct import CUDSS_BACK_ENDS   # numpy only at module level
    resolved = (context.solvers or {}).get('linear')
    linear_back_end = None if resolved is None else resolved.values.get('back_end')
    host_factor = linear_back_end not in CUDSS_BACK_ENDS   # cuDSS: device, outside the pool
    # finalize assembles on the host; the device sees K_ff only at the first solve of the run (linear_factor_items)
    run_items = [('output copies', 2 * FLOAT32_BYTES * elements)]
    transient_items, factors, one_at_a_time, factor_phases = [], [], False, {}
    # cuDSS fill of K_ff and of the pencil: counted on the built mesh, else the 2D growth law
    fill = factor_fill(mesh, 2 if discretization == 'quadratic' else 1, vector_dofs, components, 2)
    if static or _prestress_solved(settings_values, context, study == EIGENFREQUENCY_STUDY):
        held, passing, factors = linear_factor_items(vector_dofs, entries, linear_back_end, on_gpu, fill,
                                                     kept=static)
        run_items += held
        transient_items += passing
        one_at_a_time = not static   # released before the pencil's factor is planned
        if not static and study == EIGENFREQUENCY_STUDY:   # its phase, never live with the pencil phases
            factor_phases = {factor[0]: LINEAR_STATIC_STATE_PHASE for factor in factors}
    host_items = [('dof maps (full and solid)', 2 * (_INT64_BYTES * local * elements + 2 * _INT64_BYTES * scalar_dofs)),
                  ('element stiffness / body force / held arrays', 20 * FLOAT64_BYTES * elements
                   + 3 * FLOAT64_BYTES * vector_dofs),
                  ('COO assembly (rows, columns, values)', (2 * _INT64_BYTES + FLOAT64_BYTES) * coo_entries),
                  ('stiffness CSR and its free-dof copy', 2 * csr_entry_bytes * entries),
                  (f'nodal outputs ({_FEM_NODAL_OUTPUTS} x float64) and their element values',
                   _FEM_NODAL_OUTPUTS * FLOAT64_BYTES * (scalar_dofs + local * elements))]
    if study == EIGENFREQUENCY_STUDY:
        host_items.append(('consistent mass CSR', csr_entry_bytes * entries))
        pml = bool(_pml_faces(settings_values))
        complex_pencil = bool(context.material_sets(LOSS_ARRAYS[0]) or pml)   # complex pencil, factor and vectors
        rigid_motions = _estimated_rigid_motions(settings_values, context)
        _, n_basis = eigen_basis_counts(context, vector_dofs, rigid_motions, pml)
        # solve_eigenfrequencies releases the ModalPencil (DevicePencil, factor) after each solve; a complex pencil
        # runs eigs_shifted (the device Krylov-Schur on a DevicePencil), a real one eigsh_shifted
        eigen_transient, eigen_host_items, eigen_factors, eigen_phases, basis_on_device = eigen_device_items(
            context, vector_dofs, entries, 16 if complex_pencil else FLOAT64_BYTES, dimensions=fill,
            complex_eigs=complex_pencil, n_basis=n_basis)
        host_items.extend((name, (2 if complex_pencil else 1) * value) for name, value in _eigen_host_items(
            context, vector_dofs, entries, rigid_motions, pml, basis_on_device))
        transient_items += eigen_transient
        factors += eigen_factors
        factor_phases.update(eigen_phases)
        host_items.extend(eigen_host_items)
        if pml:
            host_items.append(('stretched PML stiffness and mass (complex CSR)', 2 * (16 + INT32_BYTES) * entries))
        if settings_values.get('stress_stiffening'):
            host_items.append(('geometric stiffness K_sigma and K + K_sigma (float64 CSR)',
                               2 * csr_entry_bytes * entries))
    elif study == FREQUENCY_RESPONSE_STUDY:   # K factor never built; the kept HarmonicSystem on the 'linear' runtime
        host_items.extend((('frequency response loss and mass CSR (float64)', 2 * csr_entry_bytes * entries),
                           ('frequency response complex operator copies', 3 * (16 + INT32_BYTES) * entries),
                           ('frequency response solution vectors (complex)', 4 * 16 * vector_dofs)))
        if host_factor:
            host_items.extend(host_factor_items(linear_back_end,
                                                'frequency response complex SuperLU L + U (heuristic)',
                                                2 * _FEM_FILL_FACTOR * host_real_lu_bytes(vector_dofs), vector_dofs,
                                                entries, complex_values=True))
        else:   # the kept HarmonicSystem: cuDSS LDL^T refreshed from the DevicePencil (K + j L, M)
            run_items += harmonic_pencil_items(vector_dofs, entries, linear_back_end, (16, FLOAT64_BYTES))
            factors.append(('frequency response cuDSS factor', vector_dofs, entries, 16, 2,
                            linear_back_end == 'cudss_mixed'))
        host_items.extend(reduced_sweep_memory_items(context, vector_dofs, entries))
    else:
        if host_factor:
            host_items.extend(host_factor_items(linear_back_end, 'SuperLU L + U factors (float64, heuristic)',
                                                _FEM_FILL_FACTOR * host_real_lu_bytes(vector_dofs), vector_dofs,
                                                entries))
        if settings_values.get('out_of_plane') == GENERALIZED_PLANE_STRAIN and not is_axisymmetric(settings_values):
            host_items.append(('generalized plane strain unit loads and solutions (3 + 3 x float64)',
                               6 * FLOAT64_BYTES * vector_dofs))
    if settings_values.get('geometric_nonlinearity'):   # the Newton tangent replaces K in the 'linear' factor
        # fem.elasticity.TangentAssembler: per element group the rule points / gradients / weights and C, the int32
        # scatter map of every element-block entry, element vector dofs, the pattern and its free take
        rule_points = 12 if is_axisymmetric(settings_values) else (6 if discretization == 'quadratic' else 3)
        assembler_bytes = (elements * (rule_points * (2 * local + 4) * FLOAT64_BYTES + 16 * FLOAT64_BYTES
                                       + components * local * _INT64_BYTES) + INT32_BYTES * coo_entries
                           + (2 * INT32_BYTES + _INT64_BYTES) * entries)
        host_items.extend([('Newton tangent K_T CSR and its free-dof copy', 2 * csr_entry_bytes * entries),
                           ('Newton vectors (state, internal force, residual, correction, proposal, load)',
                            6 * FLOAT64_BYTES * vector_dofs),
                           ('Newton tangent assembler: rule gradients, scatter map, pattern and free take',
                            assembler_bytes)])
        if settings_values.get('tangent_update') == 'when_slow':   # modified Newton keeps the factor's tangent values
            host_items.append(('modified Newton: values of the held tangent', FLOAT64_BYTES * entries))
        if not static and host_factor:   # the static state of the pencil is factorised too
            host_items.extend(host_factor_items(linear_back_end,
                                                'SuperLU L + U factors of the Newton tangent (float64, heuristic)',
                                                _FEM_FILL_FACTOR * host_real_lu_bytes(vector_dofs), vector_dofs,
                                                entries))
    return MemoryEstimate.from_items(((f'fields ({n_fields} x float32 per element)',
                                       n_fields * FLOAT32_BYTES * elements),
                                      ('window element mask', BOOL_BYTES * elements)),
                                     host_items=host_items, run_items=run_items, transient_items=transient_items,
                                     **cudss_external(factors, one_at_a_time=one_at_a_time, phases=factor_phases))


def _window_elements(mesh):
    """(elements along x, along y) of the physical window of a tensor MeshEstimate."""
    sides = mesh.npml_sides
    return int(mesh.nx) - sides['x_lo'] - sides['x_hi'], int(mesh.ny) - sides['y_lo'] - sides['y_hi']


def _memory_estimate(settings_values, mesh, context):
    """MechanicsFields2D plus the window ElasticSystem counted from its allocations, per-solve temporaries by phase (a
    triangle mesh: _fem_memory_estimate; a background solid of StudySpec.solved_physics: as stationary)."""
    context = background_context(context, context.physics_spec)
    if getattr(mesh, 'kind', 'tensor') == _TRIANGLES:
        return _fem_memory_estimate(settings_values, mesh, context)
    precision = linear_solver_precision(context)
    mixed = linear_solver_mixed(context)   # float64 system, float32 inner CG (system.ElasticSystem._bind_linear_solver)
    value_bytes = FLOAT64_BYTES if precision == 'float64' else FLOAT32_BYTES
    n_x, n_y = _window_elements(mesh)
    elements = n_x * n_y
    dofs = 2 * (n_x + 1) * (n_y + 1)
    entries = _STENCIL_ENTRIES_PER_DOF * dofs
    n_fields = len(OUTPUT_ARRAYS) + len(INPUT_ARRAYS)
    fields = MemoryEstimate.from_items(((f'fields ({n_fields} x float32)',
                                         n_fields * FLOAT32_BYTES * int(mesh.n_cells)),))
    # unknown, held values, applied load, rhs, CG unknown; held mask and its rigid-check copy; PCGSolver workspace
    # (residual, direction, CSR product, preconditioned; mixed: float32 + the float64 outer vectors and the float32 CSR
    # values); held-u_z node mask
    vectors = ((5 * value_bytes + 2 * BOOL_BYTES) * dofs + BOOL_BYTES * (dofs // 2)
               + workspace_bytes(dofs, 'mixed' if mixed else SOLVER_PRECISIONS[precision], jacobi=False,
                                 csr_entries=entries if mixed else 0))
    if mixed:   # float32 inverse blocks (3 per node), the two CSR product outputs (float64 + float32)
        vectors += 3 * FLOAT32_BYTES * (dofs // 2) + (FLOAT64_BYTES + FLOAT32_BYTES) * dofs
    # stencil positions (int64), indices / entry rows (int32) per entry; indptr per dof, diagonal-block positions
    pattern = (_INT64_BYTES + 2 * INT32_BYTES) * entries + (INT32_BYTES + 2 * _INT64_BYTES) * dofs
    operators = 2 * value_bytes * entries   # operator_full and A_solve data
    # fixed values, free, fixed coupling, kept load, 3 inverse-block entries per node; element area, applied body force
    # (2), the ten kept stiffness coefficients (stiffness.ElementStiffness); solid mask and its rigid-check copy
    assembly = int(5.5 * value_bytes * dofs) + (13 * value_bytes + 2 * BOOL_BYTES) * elements
    # + the element_stiffness / plane-stress condensation temporaries (E, nu, Lame pair, 6 condensed coefficients)
    build_extra = (BOOL_BYTES + 2 * _INT64_BYTES + value_bytes) * entries + 40 * dofs + 10 * value_bytes * elements
    # load copy, SpMV and output temporaries of one static solve
    transient_items = [('static solve', 'solve temporaries', (2 * dofs + 20 * elements) * value_bytes)]
    run_items, factors, factor_phases = [], (), {}
    if (settings_values.get('out_of_plane') == GENERALIZED_PLANE_STRAIN
            and not is_axisymmetric(settings_values)):
        # kept load and solution per free eps_zz term (_unit_axial_solve until the next _assemble), the summed load
        # copy, the eps_zz centre field
        n_terms = len(BENDING_TERMS[settings_values.get('bending', 'xy')])
        run_items.append(('generalized plane strain unit solves',
                          (2 * n_terms + 1) * value_bytes * dofs + value_bytes * elements))
    if stiffness_follows_state(context):
        # stencil, gathered data, product temporary and the new operator pair while the old pair is alive
        transient_items.append(('stiffness re-assembly', 'stiffness re-assembly (state-dependent stiffness)',
                                4 * value_bytes * entries))
    host_items = ()
    study = context.spec.study.type
    guided = ((study == EIGENFREQUENCY_STUDY and settings_values.get('axial_wavenumber') is not None)
              or (study == OPTOMECHANICS_STUDY and backward_scattering(context.physics_spec)))
    if guided:
        guided_dofs = 3 * (dofs // 2)
        guided_entries = _GUIDED_ENTRIES_PER_DOF * guided_dofs
        run_items.append(('guided shown u_z amplitude', value_bytes * (dofs // 2)))
        transient_items.append(('eigen assembly', 'guided density / Lame temporaries of the host copies',
                                4 * value_bytes * elements))
        eigen_transient, eigen_host_items, factors, factor_phases, _ = eigen_device_items(
            context, guided_dofs, guided_entries, FLOAT64_BYTES)
        transient_items += eigen_transient
        host_items = (('guided 2.5D host assembly (pattern, stencil, K and M)',
                       _GUIDED_ASSEMBLY_BYTES_PER_ENTRY * guided_entries),
                      *_eigen_host_items(context, guided_dofs, guided_entries), *eigen_host_items)
    elif settings_values.get('piezoelectric'):
        host_items = _piezo_host_items(context, dofs // 2, study in _MODAL_STUDIES)
        if study in _MODAL_STUDIES:   # the coupled (K, M) ShiftPencil
            coupled_dofs = PIEZO_COMPONENTS * (dofs // 2)
            eigen_transient, eigen_host_items, factors, factor_phases, _ = eigen_device_items(
                context, coupled_dofs, _GUIDED_ENTRIES_PER_DOF * coupled_dofs, FLOAT64_BYTES)
            transient_items += eigen_transient
            host_items += eigen_host_items
        if study == FREQUENCY_RESPONSE_STUDY:
            host_items += frequency_response_host_items(PIEZO_COMPONENTS * (dofs // 2))
    elif study in _MODAL_STUDIES:
        # mass stencil, its gathered data and CSR copy, density temporaries: freed once M is on the host
        transient_items.append(('eigen assembly', 'eigen mass assembly',
                                3 * value_bytes * entries + 4 * value_bytes * elements))
        eigen_transient, eigen_host_items, factors, factor_phases, _ = eigen_device_items(context, dofs, entries,
                                                                                          FLOAT64_BYTES)
        transient_items += eigen_transient
        host_items = (*_eigen_host_items(context, dofs, entries), *eigen_host_items)
    elif study == FREQUENCY_RESPONSE_STUDY:   # host stencils of K, loss, M and the complex factor per frequency
        host_items = (('frequency response host stencils', 3 * _GUIDED_ASSEMBLY_BYTES_PER_ENTRY * entries),
                      *frequency_response_host_items(dofs))
    if study == FREQUENCY_RESPONSE_STUDY:   # density, loss and amplitude window temporaries on the device
        transient_items.append(('frequency response', 'frequency response device temporaries',
                                6 * value_bytes * elements))
        host_items = (*host_items, *reduced_sweep_memory_items(context, dofs, entries))
    system_items = [('node vectors', vectors), ('sparse pattern', pattern), ('assembled operators', operators),
                    ('assembly arrays', assembly)]
    if settings_values.get('piezoelectric'):
        # shown nodal phi; window e (6) and eps (3) copies of the assembly; the material arrays the solver adds
        system_items.append(('piezoelectric device arrays', value_bytes * (dofs // 2) + 9 * value_bytes * elements))
        added = sum(1 for name in PIEZO_DEFAULTS if not context.material_sets(name))
        if added:
            system_items.append((f'piezoelectric material arrays ({added} x float32)',
                                 added * FLOAT32_BYTES * int(mesh.n_cells)))
    system = MemoryEstimate.from_items(system_items, build_extra=build_extra, host_items=host_items,
                                       run_items=run_items, transient_items=transient_items,
                                       **cudss_external(factors, phases=factor_phases))
    return combine((fields, system))


def _piezo_host_items(context, nodes, modal):
    """Host items of the piezoelectric coupled solves (3 dofs per node): the assembly (pattern, stencil, A), G and G^T A
    G with the free restriction, then the static SuperLU factor or, modal, the coupled mass and the eigen items."""
    coupled_dofs = PIEZO_COMPONENTS * nodes
    coupled_entries = _GUIDED_ENTRIES_PER_DOF * coupled_dofs
    csr_entry_bytes = FLOAT64_BYTES + INT32_BYTES
    items = [('piezoelectric host assembly (pattern, stencil, coupled operator)',
              _GUIDED_ASSEMBLY_BYTES_PER_ENTRY * coupled_entries),
             ('piezoelectric reduced operator copies (float64 CSR)', 3 * csr_entry_bytes * coupled_entries)]
    if modal:
        items.extend(_eigen_host_items(context, coupled_dofs, coupled_entries))
    else:
        items.append(('piezoelectric SuperLU L + U factors (float64, heuristic)', host_real_lu_bytes(coupled_dofs)))
    return tuple(items)


def eigen_basis_counts(context, dofs, rigid_motions=0, pml=False) -> tuple:
    """(n_wanted, n_basis) of one modal.ModalPencil shift solve: n_resonances + EXTRA_EIGENVALUES + rigid motions (+
    n_resonances again with an elastic PML shell, ModalPencil._n_wanted) and the eigs / eigsh basis (also definition3d)."""
    n_resonances = modal_resonance_count(context)
    n_wanted = max(min(n_resonances * (2 if pml else 1) + EXTRA_EIGENVALUES + int(rigid_motions), dofs - 1), 1)
    return n_wanted, arnoldi_basis_size(n_wanted, dofs, eigen_subspace_size(context.solvers))


def _eigen_host_items(context, dofs, entries, rigid_motions=0, pml=False, basis_on_device=False):
    """Host items of modal.solve_modes: K, M, K - sigma M copies, the SuperLU factor (cuDSS: eigen_device_items), the
    Arnoldi / Lanczos basis (the 'eigen' solver's ncv; basis_on_device: the device Krylov-Schur holds it, booked by
    eigen_device_items), the eigenvectors and the kept modes; rigid_motions asked on top."""
    n_wanted, n_basis = eigen_basis_counts(context, dofs, rigid_motions, pml)
    n_resonances = modal_kept_count(context)   # several targets / a band keep more mode shapes
    csr_entry_bytes = FLOAT64_BYTES + INT32_BYTES
    factor = (() if eigen_factorization(context.solvers) in CUDSS_FACTORIZATIONS
              else host_factor_items(eigen_host_direct(context.solvers, dofs),
                                     'eigen SuperLU L + U factors (float64, heuristic)', host_real_lu_bytes(dofs), dofs,
                                     entries))
    basis = (((f'eigenvectors ({n_wanted} vectors; the basis is on the device)', n_wanted * FLOAT64_BYTES * dofs),)
             if basis_on_device else
             ((f'Lanczos basis + eigenvectors ({n_basis} + {n_wanted} vectors)',
               (n_basis + n_wanted) * FLOAT64_BYTES * dofs),))
    return (('eigen host K, M and shifted copies (float64 CSR)', 5 * csr_entry_bytes * entries), *factor, *basis,
            (f'mode shapes ({n_resonances} x float64)', n_resonances * FLOAT64_BYTES * dofs))


# DevicePencil set-up transients per stored entry beside the term uploads and the int64 lower take
# (sparse_direct.DevicePencil.__init__ / full_values): the device symmetry check (int64 rows, columns, keys and
# argsort permutation); the mirror of a lower-stored M for the device mass product, live at its searchsorted peak
# (int64 rows, columns, lower rank, both key arrays, the bool mask; the later mirror / gather indices reuse freed blocks)
_SYMMETRY_CHECK_BYTES = 32
_MIRROR_BYTES = 48


# Transient phases of a modal pencil (ModalPencil.release() after each solve / search), in the order the code runs
# them (arpack_shift_invert eigsh_shifted / eigs_shifted): the DevicePencil upload before the factor is planned; eigs
# only: the M mirror of ShiftPencil.device_mass after the factor (eigsh mirrors before it, in the set-up phase); the
# Arnoldi solve with the factor, the mass product and (eigs) the device Krylov-Schur basis
EIGEN_SETUP_PHASE = 'eigen pencil set-up'
EIGEN_MIRROR_PHASE = 'eigen mass mirror'
EIGEN_PHASE = 'eigen solve'


def eigen_device_items(context, dofs, entries, value_bytes, terms=2, term_bytes=None, symmetric=True, checked=False,
                       dimensions=2, complex_eigs=False, n_basis=0) -> tuple:
    """(transient items by phase, host items, cuDSS factors, {factor label: its phases}, basis_on_device) of the eigen
    ShiftPencil (modal.ModalPencil) on a cuDSS factorization; ((), (), (), {}, False) on SuperLU. complex_eigs: the
    eigs_shifted path (a complex pencil: PML, loss, loaded electrodes; else eigsh_shifted), whose device matvec runs the
    device Krylov-Schur on n_basis vectors when device_arnoldi_fits (also used by definition3d)."""
    from ...solvers.arpack_shift_invert import (DEVICE_PENCIL_MIN_ENTRIES, MIXED_EIGEN_MIN_UNKNOWNS,
                                                arnoldi_basis_bytes)
    from ...solvers.cudss_factor import index_map_bytes, operator_copy_bytes, solve_buffer_bytes, solve_peak_bytes
    factorization = eigen_factorization(context.solvers)
    if factorization not in CUDSS_FACTORIZATIONS:
        return (), (), (), {}, False
    term_bytes = value_bytes if term_bytes is None else term_bytes
    # S5c (ArpackShiftInvertSolver.factorization_for): a small cudss_mixed pencil factors in double on the host route
    switched = factorization == 'cudss_mixed' and dofs < MIXED_EIGEN_MIN_UNKNOWNS
    mixed = factorization == 'cudss_mixed' and not switched
    buffers = ('eigen cuDSS rhs buffer and row scale', solve_buffer_bytes(dofs, value_bytes, mixed))
    solve_vectors = ('eigen cuDSS one solve\'s vectors', solve_peak_bytes(dofs, value_bytes, mixed))
    factor_label = 'eigen cuDSS factor of K - sigma M'
    factors = ((factor_label, dofs, entries, value_bytes, dimensions, mixed),)
    host_items = (('eigen shift pencil term values and diagonals', terms * term_bytes * (entries + dofs)),)
    if not (entries >= DEVICE_PENCIL_MIN_ENTRIES and not switched):
        # a factor of the host K - sigma M: its lower-triangle (LU: full) CSR copies in the pool; host ARPACK
        copies = ('eigen cuDSS lower-triangle CSR copies',
                  operator_copy_bytes(dofs, entries, value_bytes, symmetric=symmetric, mixed=mixed))
        items = tuple((EIGEN_PHASE, label, size) for label, size in (buffers, solve_vectors, copies))
        return items, host_items, factors, {factor_label: EIGEN_PHASE}, False
    # DevicePencil (terms of term_bytes, R4: a symmetric one keeps the lower triangle with its diagonal, (entries +
    # dofs) / 2 values per term), counted array by array; DevicePencil.__init__ holds the full terms, the int64 lower
    # take and the new lower terms together
    lower = (entries + dofs) // 2 if symmetric else entries
    stored = ('eigen DevicePencil stored terms', terms * term_bytes * lower)
    upload = ('eigen DevicePencil upload transients (full terms, lower take, symmetry check)',
              terms * term_bytes * entries + (8 * lower if symmetric else 0)
              + (int((_SYMMETRY_CHECK_BYTES + 2 * term_bytes) * entries) if checked else 0))
    # the CudssFactorization planned from the pencil's device values (refactorize_device_lower / _device_values): the
    # index maps of the pattern (the planned CSR shares their index arrays: no CSR copies on top), the equilibrated
    # values (symmetric: lower triangle + diagonal; mixed: + the half-size factor copy)
    maps = ('eigen cuDSS index maps of the pencil pattern', index_map_bytes(dofs, entries, symmetry_check=False))
    values = ('eigen cuDSS equilibrated values (symmetric: lower triangle and diagonal)',
              (value_bytes + (value_bytes // 2 if mixed else 0)) * lower + (value_bytes * dofs if symmetric else 0))
    # factorize_pencil keeps the combined pencil values until the new factor is probed (a full-pattern pencil: or the
    # float64 row-scale gathers and their product of its equilibration, if larger)
    combined = ('eigen combined pencil values of one factorisation',
                value_bytes * lower if symmetric else max(value_bytes, 3 * FLOAT64_BYTES) * entries)
    mirror = ('eigen M mirror of the device mass product (searchsorted peak)', int(_MIRROR_BYTES * entries))
    # ShiftPencil.device_mass: M on the full pattern (a lower-stored pencil: new values + the kept int64 full take),
    # int32 indices / indptr; the product vectors
    mass = ('eigen device mass product (M values, full take, int32 indices, vectors)',
            ((term_bytes + 8) if symmetric else 0) * entries + INT32_BYTES * (entries + dofs + 1)
            + 3 * value_bytes * dofs)
    basis_on_device, basis_bytes = (arnoldi_basis_bytes(dofs, n_basis, solver_backend(context.spec),
                                                        device_matvec=True) if complex_eigs else (False, 0))
    basis = (f'eigen device Krylov-Schur basis ({n_basis} + 3 x complex128)', basis_bytes if basis_on_device else 0)
    setup = [stored, upload]
    if symmetric and not complex_eigs:   # eigsh: mass_operator() mirrors M after the upload, before the factor
        setup = [stored, (upload[0] + ' / M mirror', max(upload[1], mirror[1]))]
    phases = {EIGEN_SETUP_PHASE: setup, EIGEN_PHASE: [stored, maps, values, buffers, solve_vectors, mass, basis]}
    factor_phases = [EIGEN_PHASE]
    if symmetric and complex_eigs:   # eigs: device_mass() after _factorize_shift, before the Arnoldi basis
        phases[EIGEN_MIRROR_PHASE] = [stored, maps, values, buffers, mirror]
        factor_phases.append(EIGEN_MIRROR_PHASE)
    else:   # the factorisation runs with the mass product already built (eigs with a mirror: below the mirror phase)
        phases[EIGEN_PHASE].append(combined)
    items = tuple((phase, label, size) for phase, group in phases.items() for label, size in group)
    return items, host_items, factors, {factor_label: tuple(factor_phases)}, basis_on_device


def harmonic_pencil_items(dofs, entries, back_end, term_value_bytes) -> list:
    """Pool run items of the kept frequency_response.HarmonicSystem on a cuDSS 'linear' runtime (DevicePencil, solve
    vectors, kept solution; no CSR copies: see eigen_device_items; also used by definition3d)."""
    from ...solvers.cudss_factor import solve_buffer_bytes, solve_peak_bytes
    from ...solvers.sparse_direct import device_pencil_bytes
    mixed = back_end == 'cudss_mixed'
    return [('frequency response cuDSS rhs buffer, row scale and one solve\'s vectors (complex)',
             solve_buffer_bytes(dofs, 16, mixed) + solve_peak_bytes(dofs, 16, mixed)),
            ('frequency response DevicePencil terms, index maps and per-frequency transients',
             device_pencil_bytes(entries, term_value_bytes, symmetric=True)),
            ('frequency response kept device solution (complex)', 16 * dofs)]


# run-time estimate: host passes over the complex CSR entries of one frequency-response solve (complex cast, K -
# omega^2 M, + j loss, free-row and free-column slicing, the CSC copy of factorize_host, the rhs and A x products;
# calibrate against the study seconds left after the factorisations)
_HARMONIC_HOST_PASSES = 8
_ASSEMBLY_LAUNCHES = 40   # kernels / cupy calls of one ElasticSystem.assemble (stencil, gathers, CSR pair, blocks)


def _assembly_ops(elements, dofs, entries, value_bytes):
    """One device ElasticSystem.assemble counted from its arrays: element coefficients -> stencil, the CSR data gather,
    the eliminated A_solve data (row / column free gathers), the fixed coupling product, the inverse node blocks."""
    reads = 10 * value_bytes * elements + (4 * value_bytes + 2 * _INT64_BYTES) * entries + 5 * value_bytes * dofs
    writes = 3 * value_bytes * entries + 3 * value_bytes * dofs
    return (te.Elementwise(reads, writes, launches=_ASSEMBLY_LAUNCHES), te.Spmv(dofs, entries, value_bytes),
            te.PythonCalls(2 * _ASSEMBLY_LAUNCHES))


def _solve_io_ops(elements, dofs, value_bytes):
    """Per solve: the nodal load (thermal, body force, tractions) and eliminated rhs, then the float32 outputs (strain,
    stress, von Mises at the cell centres)."""
    n_outputs = len(OUTPUT_ARRAYS)
    return (te.elementwise(elements, 12, 4, value_bytes, launches=20),
            te.elementwise(dofs, 4, 2, value_bytes, launches=4),
            te.elementwise(elements, 8, n_outputs, FLOAT32_BYTES, launches=2 * n_outputs),
            te.PythonCalls(40 + 3 * n_outputs))


def _modal_time_estimate(settings_values, context, nodes, dofs, entries, assembly_ops, value_bytes):
    """eigenfrequency / optomechanics: one shift-invert Lanczos of the 'eigen' role (real host SuperLU of K - sigma M,
    n_resonances + EXTRA_EIGENVALUES eigenpairs) after the device assembly and the mass copied to the host; guided: the
    3-component host pencil instead."""
    study = context.study.type
    eigenpairs = modal_resonance_count(context) + EXTRA_EIGENVALUES
    guided = ((study == EIGENFREQUENCY_STUDY and settings_values.get('axial_wavenumber') is not None)
              or (study == OPTOMECHANICS_STUDY and backward_scattering(context.physics_spec)))
    if guided:
        unknowns = 3 * nodes
        nnz = _GUIDED_ENTRIES_PER_DOF * unknowns
        items = (te.TimeItem('guided pencil assembly (host)', 'assembly', 1.0, 'exact',
                             (te.HostStream(_GUIDED_ASSEMBLY_BYTES_PER_ENTRY * nnz), te.PythonCalls(100))),)
    else:
        unknowns, nnz = dofs, entries
        csr_entry_bytes = FLOAT64_BYTES + INT32_BYTES
        # mass stencil + gather on the device, K and M to the host, their free-dof restrictions and K - sigma M
        copies = (te.Elementwise(3 * value_bytes * entries, 2 * value_bytes * entries, launches=10),
                  te.Transfer(2 * (value_bytes + INT32_BYTES) * entries),
                  te.HostStream(2 * 5 * csr_entry_bytes * entries), te.PythonCalls(60))
        items = (te.TimeItem('stiffness assembly', 'assembly', 1.0, 'exact', assembly_ops),
                 te.TimeItem('mass and host copies', 'assembly', 1.0, 'exact', copies))
    # real SuperLU priced on the scalar 5-point curve: the 2 x 2 (guided 3 x 3) node blocks fill b^1.5 more; one
    # shift-invert per shift of the eigenfrequency search (eigen_search_counts)
    shift_work, _ = eigen_search_counts(context)
    return te.combine((te.TimeEstimate(items),
                       context.solver_work('eigen', unknowns=float(unknowns), nnz=float(nnz), value_bytes=FLOAT64_BYTES,
                                           dimensions=2, eigenpairs=eigenpairs, plan_reused=True, **shift_work)))


def _frequency_response_time_estimate(context, dofs, entries, assembly_ops, io_ops):
    """frequency_response: the device assembly and the host K, loss, M stencils once, then per factorisation (the
    study's harmonic_factorizations: one per frequency, or a reduced sweep's) the complex host operator, one complex
    SuperLU + solve (frequency_response.HarmonicSystem, outside any solver role) and the load; the outputs per field
    frequency."""
    sweep = frequency_sweep_counts(context)
    factorizations, kind = sweep['factorizations'], sweep['factorizations_kind']
    complex_bytes = 2 * FLOAT64_BYTES
    per_factor = (te.HostStream(_HARMONIC_HOST_PASSES * (complex_bytes + INT32_BYTES) * entries),
                  te.HostSolve(dofs, 2, True), te.Transfer(2 * complex_bytes * dofs), te.PythonCalls(40))
    counter = ('solvers', context.name, 'counters', 'harmonic_factorizations')
    estimate = te.TimeEstimate((
        te.TimeItem('stiffness assembly', 'assembly', 1.0, 'exact', assembly_ops),
        te.TimeItem('harmonic host stencils', 'assembly', 1.0, 'exact',
                    (te.HostStream(3 * _GUIDED_ASSEMBLY_BYTES_PER_ENTRY * entries), te.PythonCalls(60))),
        te.TimeItem('complex SuperLU factorisations', 'factorisation', factorizations, kind,
                    (te.HostLU(dofs, 2, True),), counter=counter),
        te.TimeItem('harmonic operator, solve and load', 'factorisation', factorizations, kind, per_factor,
                    counter=counter),
        te.TimeItem('outputs', 'frequency', sweep['field_points'], 'exact', io_ops),
        *reduced_sweep_time_items(sweep, dofs)))
    return te.combine((estimate, *_reduced_sweep_extra_time(sweep, (te.HostSolve(dofs, 2, True),))))


def _reduced_sweep_extra_time(sweep, host_solve_ops) -> tuple:
    """(TimeEstimate,) of a reduced sweep's back substitutions beyond one per factorisation (+ the modal basis
    shift-invert, a private eigen runtime: not estimated); () for sweep_method 'direct'."""
    if sweep['method'] == 'direct':
        return ()
    extra = max(sweep['solves'] - sweep['factorizations'], 0.0)
    items = ((te.TimeItem('reduced model back substitutions', 'solve', extra, 'bounded', host_solve_ops),)
             if extra else ())
    missing = ('modal basis shift-invert (private eigen solver)',) if sweep['eigenpairs'] else ()
    return (te.TimeEstimate(items, not_estimated=missing),)


def _static_counts(settings_values, context):
    """(per_step, follows, outer solves, their kind, assemblies, their kind, linear solves) of a stationary /
    time_dependent solid: re-solved when a coupling writes into it or the stiffness follows a state (per step of a
    time_dependent study), plus the generalized plane-strain unit solves per assembly."""
    follows = stiffness_follows_state(context)
    driven = coupling_target(context) or follows
    per_step = context.time_dependent and driven
    terms = (len(BENDING_TERMS[settings_values.get('bending', 'xy')])
             if settings_values.get('out_of_plane') == GENERALIZED_PLANE_STRAIN
             and not is_axisymmetric(settings_values) else 0)
    if per_step:   # at most one re-solve per step (only after an input changed); unit solves of a first assembly that is
        # not repeated are left out (a few solves against one per step)
        outer_solves, kind = 1.0, 'bounded'
        assemblies, assemblies_kind = 1.0, 'bounded' if follows else 'exact'
        solves = 1.0 + (terms if follows else 0)
    else:
        outer_solves, kind, _ = context.static_solves() if driven else (1.0, 'exact', None)
        assemblies, assemblies_kind = (outer_solves, kind) if follows else (1.0, 'exact')
        solves = outer_solves + terms * assemblies
    return per_step, follows, outer_solves, kind, assemblies, assemblies_kind, solves


# triangle-mesh run-time recipe (solver_fem, host float64; uncalibrated: compare with the time split of the solid_fem_*
# selftests): host bytes per COO entry of one assembly (element B^T C B by quadrature, COO -> CSR, dof maps, the
# rigid-mode graph), host bytes per element and local dof of one solve's loads and outputs (stress / body / traction
# loads, strains at the local dof points and centroids, nodal averages, device copies), SuperLU time and solve scales
# over the 2D 5-point curves at the vector unknowns (= the memory fill heuristic _FEM_FILL_FACTOR)
_FEM_ASSEMBLY_BYTES_PER_COO = 300
_FEM_ASSEMBLY_CALLS = 200
_FEM_SOLVE_BYTES_PER_LOCAL = 1500
_FEM_SOLVE_CALLS = 150
# host CSR passes (read + write of the entry bytes each), counted from the code: scipy A[free][:, free] = the row take,
# the column map and the column take; fem.assembly.pattern_sum / pattern_combination = tocoo of each part, the
# concatenation, the COO -> CSR counting sort, its index sort and the duplicate sum
_RESTRICT_PASSES = 3
_CSR_REBUILD_PASSES = 5
# geometric nonlinearity: Newton iterations per load step (a priori: a linear predictor and quadratic convergence to the
# default 1e-8 take ~4-6) and the host numpy bytes per COO entry of one tangent + internal-force assembly (the
# total-Lagrangian rule: P1 degree 2, P2 degree 4, axisymmetric degree 6; operator, stress and K_sigma passes);
# uncalibrated (definition3d's iteration count)
_NEWTON_ITERATIONS_PER_STEP = 5
_FEM_TANGENT_BYTES_PER_COO = {'linear': 800.0, 'quadratic': 1500.0}


def _estimated_rigid_motions(settings_values, context) -> int:
    """Rigid motions an eigenfrequency solve on triangles asks on top of n + 4 (target_frequency > 0, no enabled
    support, no PML ring): 1 axisymmetric (u_z slide), 2 at azimuthal order 1 (sideways translation, tilt), 0 at m >=
    2, 3 planar; else 0 (partial supports: not counted)."""
    if not settings_values.get('target_frequency') or _pml_faces(settings_values):
        return 0
    if any(feature.enabled and feature.type in SUPPORT_FEATURES for feature in context.physics_spec.features):
        return 0
    if is_axisymmetric(settings_values):
        return {0: 1, 1: 2}.get(_azimuthal_order(settings_values), 0)
    return 3


def _fem_time_estimate(settings_values, mesh, context):
    """Triangle mesh: stationary / time_dependent = host assemblies, the 'linear' sparse_direct factor (one per
    assembly) and back substitutions, loads and outputs per outer solve; eigenfrequency = the stiffness and mass
    assemblies, the pencil set-up (_fem_pencil_items), the 'eigen' shift-invert and the outputs of each shown mode (a
    loss factor or an elastic PML: the complex factor and the stretched assembly); frequency_response = the study's
    harmonic_factorizations of the 'linear' role (complex, device refresh on cuDSS), the held-value rhs and outputs.
    stress_stiffening: the
    static state and K_sigma first (_fem_prestress_time). 3 components per dof for an azimuthal order m >= 1.
    Geometric nonlinearity: each static solve (once for the pencil of the modal / response studies) is
    _fem_newton_time."""
    discretization = settings_values.get('discretization', 'quadratic')
    local = _FEM_LOCAL_DOFS[discretization]
    elements = int(mesh.n_elements)
    components = _fem_components(settings_values)
    unknowns = float(components * (int(mesh.n_nodes) + (int(mesh.n_edges) if discretization == 'quadratic' else 0)))
    coo_entries = float(elements * (components * local) ** 2)
    entries = coo_entries / _FEM_COO_PER_CSR_ENTRY
    assembly_ops = (te.HostStream(_FEM_ASSEMBLY_BYTES_PER_COO * coo_entries), te.PythonCalls(_FEM_ASSEMBLY_CALLS))
    io_ops = (te.HostStream(_FEM_SOLVE_BYTES_PER_LOCAL * local * elements), te.PythonCalls(_FEM_SOLVE_CALLS))
    scales = dict(factor_scale=_FEM_FILL_FACTOR, solve_scale=_FEM_FILL_FACTOR)
    study = context.study.type
    newton = partial(_fem_newton_time, settings_values, context, unknowns, entries, coo_entries)
    nonlinear = bool(settings_values.get('geometric_nonlinearity'))
    if nonlinear and study in (*_MODAL_STUDIES, FREQUENCY_RESPONSE_STUDY):
        # the nonlinear static state of the pencils (once; its tangent replaces K + K_sigma), then the linear recipe
        linear = _fem_time_estimate(dict(settings_values, geometric_nonlinearity=False, stress_stiffening=False), mesh,
                                    context)
        return te.combine((linear, newton(1.0, 'exact', False)))
    prestress = partial(_fem_prestress_time, settings_values, context, unknowns, entries, assembly_ops, io_ops)
    if study in _MODAL_STUDIES:   # optomechanics: the same solve_eigenfrequencies (its overlap tables not estimated)
        n_resonances = modal_resonance_count(context)
        host = {} if eigen_factorization(context.solvers) in CUDSS_FACTORIZATIONS else {'backend': 'cpu'}
        pml = bool(_pml_faces(settings_values))
        loss = context.material_sets(LOSS_ARRAYS[0])
        complex_pencil = pml or loss   # eigs on the complex symmetric pencil, else eigsh (Lanczos)
        if not complex_pencil:
            scales = dict(scales, applications_per_basis=LANCZOS_APPLICATIONS_PER_BASIS)
        shift_work, resonances = eigen_search_counts(context)   # one shift-invert per shift of the search
        items = (te.TimeItem('stiffness assembly', 'assembly', 2.0 if pml else 1.0, 'exact', assembly_ops),
                 te.TimeItem('consistent mass assembly', 'assembly', 1.0, 'exact', assembly_ops),
                 *_fem_pencil_items(entries, assembly_ops, loss, complex_pencil),
                 te.TimeItem('mode outputs', 'resonance', resonances, 'bounded', io_ops))
        overlaps = ('optomechanical overlap tables',) if study == OPTOMECHANICS_STUDY else ()
        # ModalPencil._n_wanted: a PML shell asks n_resonances more (its PML modes are filtered out)
        eigenpairs = (n_resonances * (2 if pml else 1) + EXTRA_EIGENVALUES
                      + _estimated_rigid_motions(settings_values, context))
        # a cuDSS static state of the same pattern leaves its ordering in the run's store (R1a: the eigen plan reuses it)
        from ...solvers.sparse_direct import CUDSS_BACK_ENDS
        linear = (context.solvers or {}).get('linear')
        ordering_kept = getattr(context, 'ordering_kept', False) or bool(
            not host and _static_state_solved(settings_values, context, True) and linear is not None
            and linear.values.get('back_end') in CUDSS_BACK_ENDS)
        return te.combine((te.TimeEstimate(items, not_estimated=overlaps), *prestress(with_loads=True),
                           context.solver_work('eigen', unknowns=unknowns, nnz=entries,
                                               value_bytes=2 * FLOAT64_BYTES if complex_pencil else FLOAT64_BYTES,
                                               dimensions=2, eigenpairs=eigenpairs, plan_reused=True,
                                               ordering_kept=ordering_kept, **shift_work, **scales, **host)))
    if study == FREQUENCY_RESPONSE_STUDY:   # factorisations: the study's harmonic_factorizations (sweep_method)
        # the kept HarmonicSystem on the 'linear' runtime (solver_fem._harmonic_system): one complex symmetric factor
        # per factorisation there (host SuperLU, or cuDSS refreshed from its DevicePencil), plus per factorisation the
        # host rhs (K + j L - omega^2 M) applied to the held values and the reduction of the pencil terms once
        sweep = frequency_sweep_counts(context)
        extra_operators = 1.0 + float(context.material_sets(LOSS_ARRAYS[0]))   # mass (+ loss eta K)
        host_rhs = (te.HostStream(3 * (FLOAT64_BYTES + INT32_BYTES) * entries + 4 * 2 * FLOAT64_BYTES * unknowns),
                    te.PythonCalls(20))
        items = (te.TimeItem('stiffness assembly', 'assembly', 1.0, 'exact', assembly_ops),
                 te.TimeItem('mass and loss assembly', 'assembly', extra_operators, 'exact', assembly_ops),
                 te.TimeItem('harmonic pencil terms on one pattern (once)', 'assembly', 1.0, 'exact',
                             (_pattern_sum_stream(int(extra_operators) + 1, 2 * FLOAT64_BYTES, entries),)),
                 te.TimeItem('harmonic rhs of the held values', 'factorisation', sweep['factorizations'],
                             sweep['factorizations_kind'], host_rhs),
                 te.TimeItem('outputs (full solution products)', 'frequency', sweep['field_points'], 'exact',
                             io_ops + host_rhs),
                 *reduced_sweep_time_items(sweep, unknowns))
        work = context.solver_work('linear', unknowns=unknowns, nnz=entries, value_bytes=2 * FLOAT64_BYTES,
                                   dimensions=2, solves=sweep['solves'], solves_kind=sweep['solves_kind'],
                                   factorizations=sweep['factorizations'],
                                   factorizations_kind=sweep['factorizations_kind'], device_values=True, **scales)
        return te.combine((te.TimeEstimate(items), work, *prestress(with_loads=False), *_reduced_sweep_extra_time(
            sweep, (te.HostSolve(unknowns, 2, True, scale=_FEM_FILL_FACTOR),))))
    per_step, follows, outer_solves, kind, assemblies, assemblies_kind, solves = _static_counts(settings_values, context)
    items = (te.TimeItem('stiffness assembly', 'assembly', assemblies, assemblies_kind, assembly_ops,
                         per_step=per_step and follows),
             te.TimeItem('load and outputs', 'solve', outer_solves, kind, io_ops, per_step=per_step))
    if nonlinear:   # every static solve is a Newton loop: a tangent factor per iteration
        return te.combine((te.TimeEstimate(items), newton(outer_solves, kind, per_step)))
    return te.combine((te.TimeEstimate(items), context.solver_work(
        'linear', unknowns=unknowns, nnz=entries, value_bytes=FLOAT64_BYTES, dimensions=2, solves=solves,
        solves_kind=kind, factorizations=assemblies, factorizations_kind=assemblies_kind, per_step=per_step,
        factorizations_per_step=per_step and follows, device_values=True, **scales)))   # DiffusionDirectSolver


def _fem_pencil_items(entries, assembly_ops, loss, complex_pencil) -> tuple:
    """TimeItems of the host pencil set-up of solve_eigenfrequencies after the K / M assemblies: a loss factor's eta K
    assembly and complex K (1 + j eta) sum, then ModalPencil._reduced of K and M (free-dof restriction; a real pencil
    also its float64 copy)."""
    value_bytes = 2 * FLOAT64_BYTES if complex_pencil else FLOAT64_BYTES
    items = []
    if loss:   # complex cast of K and 1j eta K (read float64, write complex128 each), then their pattern_sum
        items += [te.TimeItem('loss stiffness assembly', 'assembly', 1.0, 'exact', assembly_ops),
                  te.TimeItem('complex K (1 + j eta) pattern sum', 'assembly', 1.0, 'exact',
                              (te.HostStream(2 * 3 * FLOAT64_BYTES * entries),
                               _pattern_sum_stream(2, value_bytes, entries)))]
    passes = _RESTRICT_PASSES + (0 if complex_pencil else 1)
    items.append(te.TimeItem('pencil free-dof restriction (K, M)', 'matrix', 2.0, 'exact',
                             (te.HostStream(2 * passes * (value_bytes + INT32_BYTES) * entries), te.PythonCalls(20))))
    return tuple(items)


def _pattern_sum_stream(parts, value_bytes, entries):
    """HostStream of a pattern_sum / pattern_combination of `parts` CSR operators of `entries` each (COO int32 rows and
    columns, _CSR_REBUILD_PASSES read + write passes)."""
    return te.HostStream(2 * _CSR_REBUILD_PASSES * parts * (value_bytes + 2 * INT32_BYTES) * entries)


def _static_state_solved(settings_values, context, with_loads) -> bool:
    """stress_stiffening: solve_static_state factorises K by the 'linear' role (loads, or an initial stress to relax),
    unless an azimuthal order or a free body (no enabled support, no PML ring) refuses it."""
    if not settings_values.get('stress_stiffening'):
        return False
    initial = any(context.material_sets(name) for name in PRESTRESS_ARRAYS)
    supported = any(feature.enabled and feature.type in SUPPORT_FEATURES for feature in context.physics_spec.features)
    refused = bool(_azimuthal_order(settings_values)) or not (supported or bool(_pml_faces(settings_values)))
    return (with_loads or initial) and not refused


def _fem_prestress_time(settings_values, context, unknowns, entries, assembly_ops, io_ops, with_loads) -> tuple:
    """(TimeEstimate, ...) of solve_static_state before the pencils (stress_stiffening, once; () without it):
    with_loads (modal studies) the static solve of every load, else (frequency_response) the relaxation of an initial
    stress: free-dof restriction and upload, one 'linear' factorisation and back substitution (+ generalized plane
    strain unit solves), loads and outputs (with_loads only); none when an azimuthal order or a free body (no enabled
    support, no PML ring) refuses it. Then the K_sigma assembly (+ the stress at its rule points, priced as a solve's
    loads and outputs) and the pattern_sum K + K_sigma, when a static state or an initial stress gives a stress."""
    if not settings_values.get('stress_stiffening'):
        return ()
    initial = any(context.material_sets(name) for name in PRESTRESS_ARRAYS)
    pml = bool(_pml_faces(settings_values))
    solved = _static_state_solved(settings_values, context, with_loads)
    if not (solved or initial):
        return ()
    value_bytes = 2 * FLOAT64_BYTES if pml else FLOAT64_BYTES   # an elastic PML stretches K and K_sigma (complex)
    estimates = [te.TimeEstimate((
        te.TimeItem('geometric stiffness K_sigma assembly', 'assembly', 1.0, 'exact', assembly_ops + io_ops),
        te.TimeItem('K + K_sigma pattern sum', 'assembly', 1.0, 'exact',
                    (_pattern_sum_stream(2, value_bytes, entries),))))]
    if solved:   # the unstretched real K: scipy K[free][:, free], then its CSR to the run backend (system_fem._solve_free)
        csr_entry_bytes = FLOAT64_BYTES + INT32_BYTES
        static_items = [te.TimeItem('static state: free-dof restriction and upload', 'solve', 1.0, 'exact',
                                    (te.HostStream(2 * _RESTRICT_PASSES * csr_entry_bytes * entries),
                                     te.Transfer(csr_entry_bytes * entries)))]
        if with_loads:
            static_items.append(te.TimeItem('static state: loads and outputs', 'solve', 1.0, 'exact', io_ops))
        terms = (len(BENDING_TERMS[settings_values.get('bending', 'xy')])
                 if with_loads and settings_values.get('out_of_plane') == GENERALIZED_PLANE_STRAIN
                 and not is_axisymmetric(settings_values) else 0)
        estimates += [te.TimeEstimate(tuple(static_items)), context.solver_work(
            'linear', unknowns=unknowns, nnz=entries, value_bytes=FLOAT64_BYTES, dimensions=2, solves=1.0 + terms,
            factorizations=1.0, factor_scale=_FEM_FILL_FACTOR, solve_scale=_FEM_FILL_FACTOR, device_values=True)]
    return tuple(estimates)


def _fem_newton_time(settings_values, context, unknowns, entries, coo_entries, solves, kind, per_step):
    """Geometric nonlinearity on triangles: per static solve load_steps x _NEWTON_ITERATIONS_PER_STEP host tangent +
    internal-force assemblies and 'linear' tangent factorisations with one back substitution each (count 'guess'; with
    tangent_update when_slow the full-Newton count is an upper bound of the factorisations: more, force-only iterations
    instead)."""
    discretization = settings_values.get('discretization', 'quadratic')
    iterations = float(solves) * int(settings_values.get('load_steps') or 1) * _NEWTON_ITERATIONS_PER_STEP
    count_kind = 'guess' if kind == 'exact' else kind
    assembly = te.TimeItem('tangent and internal force assembly', 'iteration', iterations, count_kind,
                           (te.HostStream(_FEM_TANGENT_BYTES_PER_COO[discretization] * coo_entries),
                            te.PythonCalls(_FEM_ASSEMBLY_CALLS)), per_step=per_step)
    return te.combine((te.TimeEstimate((assembly,)), context.solver_work(
        'linear', unknowns=unknowns, nnz=entries, value_bytes=FLOAT64_BYTES, dimensions=2, solves=iterations,
        solves_kind=count_kind, factorizations=iterations, factorizations_kind=count_kind, per_step=per_step,
        factorizations_per_step=per_step, factor_scale=_FEM_FILL_FACTOR, solve_scale=_FEM_FILL_FACTOR,
        device_values=True)))


def _time_estimate(settings_values, mesh, context):
    """Tensor mesh: stationary / time_dependent = the device assembly and block-Jacobi CSR PCG of the 'linear' role (+
    the generalized plane-strain unit solves per assembly), re-solved when a coupling writes into it or the stiffness
    follows a state (per step of a time_dependent study); modal and frequency_response studies by their own recipes.
    Triangle meshes: _fem_time_estimate; piezoelectric solids are not estimated. A background solid of
    StudySpec.solved_physics is priced as in a stationary study."""
    context = background_context(context, context.physics_spec)
    if mesh.kind == _TRIANGLES:
        return _fem_time_estimate(settings_values, mesh, context)
    if settings_values.get('piezoelectric'):
        return te.TimeEstimate(not_estimated=('piezoelectric coupled host solves',))
    precision = linear_solver_precision(context)
    value_bytes = FLOAT64_BYTES if precision == 'float64' else FLOAT32_BYTES
    n_x, n_y = _window_elements(mesh)
    elements, nodes = n_x * n_y, (n_x + 1) * (n_y + 1)
    dofs = 2 * nodes
    entries = _STENCIL_ENTRIES_PER_DOF * dofs
    assembly_ops = _assembly_ops(elements, dofs, entries, value_bytes)
    if context.study.type in _MODAL_STUDIES:
        return _modal_time_estimate(settings_values, context, nodes, dofs, entries, assembly_ops, value_bytes)
    io_ops = _solve_io_ops(elements, dofs, value_bytes)
    if context.study.type == FREQUENCY_RESPONSE_STUDY:
        return _frequency_response_time_estimate(context, dofs, entries, assembly_ops, io_ops)
    per_step, follows, outer_solves, kind, assemblies, assemblies_kind, solves = _static_counts(settings_values, context)
    items = (te.TimeItem('stiffness assembly', 'assembly', assemblies, assemblies_kind, assembly_ops,
                         per_step=per_step and follows),
             te.TimeItem('load and outputs', 'solve', outer_solves, kind, io_ops, per_step=per_step))
    # 2 x 2 node-block Jacobi: 2 residual + 3 inverse entries read, 2 written per node (mixed: in the float32 inner CG)
    precondition = (te.elementwise(nodes, 5, 2, FLOAT32_BYTES if linear_solver_mixed(context) else value_bytes),)
    return te.combine((te.TimeEstimate(items), context.solver_work(
        'linear', unknowns=float(dofs), nnz=float(entries), value_bytes=value_bytes, dimensions=2, solves=solves,
        solves_kind=kind, factorizations=assemblies, factorizations_kind=assemblies_kind, per_step=per_step,
        factorizations_per_step=per_step and follows, longest_axis=context.longest_axis(), warm_start=per_step,
        preconditioner_ops=precondition)))


# ---- electrode features (piezoelectric on) ----
_ELECTRODE_PARAMS = (
    Param('connection', 'choice', 'voltage', label='Connection', choices=ELECTRODE_CONNECTIONS,
          choice_labels=(('voltage', 'Fixed voltage'), ('floating', 'Floating (isolated)')),
          help='voltage: phi = voltage on every node of the electrode (0 = ground; the eigenfrequency study holds it at '
               '0: short circuit, the resonance frequencies). floating: one shared, unknown potential carrying the net '
               'free charge of the charge setting (0 = an isolated electrode; the eigenfrequency study uses 0: open circuit, the '
               'antiresonance frequencies). At least one voltage electrode is required (the reference of phi).',
          tip='Fixed voltage: the electrode is held at the voltage (or shorted to ground in an eigenfrequency study).'
              '\n\nFloating: the electrode is isolated, its voltage follows from the charge on it (open circuit in an '
              'eigenfrequency study).'),
    Param('voltage', 'float', 0.0, label='Voltage', unit='V',
          help='Potential of a voltage electrode in a static (stationary or time-dependent) solve; ignored by floating '
               'electrodes and by the eigenfrequency study (0 there).',
          tip='Voltage applied to the electrode; 0 is ground.'),
    Param('charge', 'float', 0.0, label='Charge', unit='C/m',
          help='Net free charge per m of depth on a floating electrode in a static solve (its voltage follows); '
               'ignored by voltage electrodes and by the eigenfrequency study (0 there).',
          tip='Electric charge put on an isolated electrode, per metre of depth.'),
)
_ELECTRODE_RESULT_TEXT = (
    "Result 'table' of the latest static solve with one row [feature, V, Q] (units '', V, C/m) and the extras V and Q: "
    'the electrode potential and the net free charge on it per m of depth (Q = -sum of the phi rows of A x over its '
    'nodes, the discrete Gauss law; a voltage electrode: the charge its source delivered, so Q / V of a single driven '
    'electrode against ground is the capacitance per m of depth). Not available after an eigenfrequency solve.')
_ELECTRODE_TEXT = (
    'piezoelectric only (ignored with a validation warning otherwise): an equipotential electrode, connection voltage '
    '(held phi) or floating (one shared phi with a net charge). Two electrodes must not share a node (adjacent outer '
    'faces share their corner node: a build error; select both faces in one feature instead). Its nodes keep their '
    'mechanical supports and loads (a thin metal film without stiffness).')


# ---- type ----
_FEATURES = (
    FeatureType('fixed', 'Fixed', 'boundary_condition', 'boundary', allow_interior=False, result_kind='table',
                summary='Clamps the selected domain edges: they cannot move in any direction.',
                description=f'Clamped support: u_x = u_y = 0 (and u_z = 0 for guided modes, axial_wavenumber). '
                            f'{_FACE_ONLY_TEXT} {_SUPPORT_RESULT_TEXT}',
                apply=_apply_fixed, result=_support_result, grids=_BOTH_GRIDS),
    FeatureType('roller', 'Roller', 'boundary_condition', 'boundary', allow_interior=False, result_kind='table',
                summary='Lets the selected domain edges slide along themselves but not move away from or into the '
                        'domain. Also the symmetry condition of a mirror-symmetric model.',
                description='Roller (symmetry) support: the normal displacement is 0 (u_x on x_lo / x_hi, u_y on '
                            'y_lo / y_hi), the tangential one free (zero shear traction; u_z of guided modes free '
                            'too). Rollers on one x face and one y face hold a solid completely. '
                            f'{_FACE_ONLY_TEXT} {_SUPPORT_RESULT_TEXT}',
                apply=_apply_roller, result=_support_result, grids=_BOTH_GRIDS),
    FeatureType('displacement', 'Displacement', 'boundary_condition', 'boundary', allow_interior=False,
                params=following((displacement_param('u_x', 'x'), displacement_param('u_y', 'y'))),
                result_kind='table',
                summary='Moves the selected domain edges by a given displacement, for example to stretch a sample.',
                description='Prescribed displacement: u_x and / or u_y [m] on every node of the selected faces; an '
                            'empty component stays free (validation error when both are empty; u_z of guided modes '
                            'stays free). A later support on '
                            f'the same face component sets its value (warning). {_FACE_ONLY_TEXT} '
                            f'{_SUPPORT_RESULT_TEXT}',
                apply=_apply_displacement, result=_support_result, validate=_validate_displacement,
                grids=_BOTH_GRIDS),
    FeatureType('fixed_domain', 'Fixed (domain)', 'domain_condition', 'domain', result_kind='table',
                summary='Holds the selected domains rigidly in place, like a substrate or a rigid clamp.',
                description='Holds u_x = u_y = 0 (and u_z = 0 for guided modes, axial_wavenumber) on every corner node '
                            'of the selected cells (a rigid region whose '
                            "boundary is clamped to the solid around it, whatever the cells' own stiffness). "
                            f'{_SUPPORT_RESULT_TEXT}',
                apply=_apply_fixed_domain, result=_support_result, grids=_BOTH_GRIDS),
    FeatureType('prescribed_displacement_domain', 'Prescribed displacement (domain)', 'domain_condition', 'domain',
                params=following(tuple(replace(displacement_param(f'u_{axis}', axis),
                                               help=f'Prescribed u_{axis} [m] on every node of the selected domains; '
                                                    f'empty (null) leaves u_{axis} free.',
                                               tip=f'Displacement along {axis} the selected domains are held at; empty '
                                                   'leaves this direction free.')
                                       for axis in _COMPONENT_NAMES)),
                result_kind='table',
                summary='Holds chosen displacement components of the selected domains, for example the axial motion '
                        'of a strip while it may still deflect.',
                description='Triangle meshes: prescribed displacement per component on every node of the selected '
                            'elements: u_x and / or u_y [m], an empty component stays free (validation error when '
                            'both are empty), e.g. u_x = 0 with u_y free on a thin strip that may deflect but not '
                            'slide (with geometric nonlinearity a no-slip string). Holding u_x on every node across '
                            'a member also locks the rotation of its cross-sections (bending becomes shear). A later '
                            f'support on the same node component sets its value. {_SUPPORT_RESULT_TEXT}',
                apply=_apply_prescribed_displacement_domain, result=_support_result, validate=_validate_displacement,
                grids=(_TRIANGLES,)),
    FeatureType('boundary_load', 'Boundary load', 'boundary_condition', 'boundary', allow_interior=False,
                params=following((Param('t_x', 'float', 0.0, label='Traction x', unit='Pa',
                              help='Force per area along +x on the selected outer faces (a pressure p pushing into the '
                                   'x_hi face is t_x = -p).',
                              tip='Force per area along x on the face; on the right edge a negative value pushes '
                                  'into the domain.'),
                        Param('t_y', 'float', -1e6, label='Traction y', unit='Pa',
                              help='Force per area along +y on the selected outer faces (a pressure p on y_hi is t_y = '
                                   '-p). Typical 1e5-1e8.',
                              tip='Force per area along y on the face; on the top edge a negative value presses down.'
                                  '\n\nTypical 0.1 to 100 MPa.'))),
                summary='Applies a force per area on the selected domain edges.',
                description='Surface traction (t_x, t_y) [Pa] on outer faces, in the model axes (not the face normal): '
                            'consistent nodal forces t * half of each adjacent face segment, per m of depth (geometry '
                            'axisymmetric: t times the tributary ring area, N; on x_hi the cylinder r = R, on a y face '
                            'the annulus). Several loads add; on a held component it only enters the reaction.',
                apply=_apply_boundary_load, grids=_BOTH_GRIDS),
    FeatureType('segment_load', 'Boundary load (segment)', 'boundary_condition', 'line', object_kinds=('line',),
                params=following((Param('t_x', 'float', 0.0, label='Traction x', unit='Pa',
                              help='Force per area along +x on the selected line segments (model axes).',
                              tip='Force per area along x on the segment.'),
                        Param('t_y', 'float', -1e6, label='Traction y', unit='Pa',
                              help='Force per area along +y on the selected line segments (a pressure p pressing down '
                                   'on a top face is t_y = -p). Typical 1e5-1e8.',
                              tip='Force per area along y on the segment; a negative value presses down.\n\n'
                                  'Typical 0.1 to 100 MPa.'))),
                summary='Applies a force per area on part of a domain edge, or along a line inside the solid, drawn '
                        'as a horizontal or vertical line.',
                description='Surface traction (t_x, t_y) [Pa] on a part of a boundary or along an interior line: '
                            'each selected line must be horizontal or vertical (validation error '
                            'otherwise) and acts on the node line (cell edge) nearest to it (validation warns beyond '
                            'a hundredth of a cell): a line on an outer face loads that part of the face, an interior '
                            'line a line force inside the solid (a traction jump across it). Consistent nodal forces: '
                            't times the integral of each node shape function over the segment, clipped to the window, '
                            'so the total force is t times the segment length per m of depth, also when its ends lie '
                            'inside cells (geometry axisymmetric: whole rings in N, a horizontal segment the annulus '
                            'pi (r_1^2 - r_0^2), a vertical one the cylinder 2 pi r h). Several loads add. On a '
                            'triangle mesh the line may run in any direction: the mesher puts element edges on it and '
                            'the traction acts on them exactly (int t N ds per node).',
                apply=_apply_segment_load, validate=_validate_segment_load, grids=_BOTH_GRIDS),
    FeatureType('body_force', 'Body force', 'domain_condition', 'domain',
                params=following((Param('f_x', 'float', 0.0, label='Body force x', unit='N/m^3', field_dependent=True,
                              help='Force per volume along +x; may depend on x and y.',
                              tip='Force per volume along x, for example from an acceleration.'),
                        Param('f_y', 'float', -2.16e4, label='Body force y', unit='N/m^3', field_dependent=True,
                              help='Force per volume along +y; gravity along -y is -rho g (silica -2.16e4). May depend '
                                   'on x and y.',
                              tip='Force per volume along y; gravity is minus density times 9.81.'))),
                summary='Applies a force per volume throughout the selected domains, such as gravity.',
                description='Applied body force density [N/m^3], consistent nodal forces f * cell area / 4 to the four '
                            'corners of each selected cell (geometry axisymmetric: f times the integral of the shape '
                            'function over the ring volume, with x = r; f_x is radial); void cells (no youngs_modulus '
                            'or c11 > 0) drop it. Several body forces '
                            'add; the coupled slots body_force_x / body_force_y are separate.',
                apply=_apply_body_force, grids=_BOTH_GRIDS),
    FeatureType('temperature_change', 'Temperature change', 'domain_condition', 'domain',
                params=(Param('delta_T', 'float', 10.0, label='Temperature change', unit='K', field_dependent=True,
                              help='Temperature above the stress-free state on the selected cells (added to '
                                   'temperature_rise_ext; several add); the thermal strain is alpha * delta_T. May '
                                   'depend on x and y.',
                              tip='How much warmer the selected domains are than when they were stress free.'),),
                summary='Heats the selected domains by a given amount without a heat physics, so that they expand.',
                description='Applied temperature change [K] for thermal strain, without a heat physics: added to '
                            'temperature_rise_ext, which adds to the coupled temperature_rise of '
                            f'{THERMAL_STRAIN_COUPLING}.',
                apply=_apply_temperature_change, grids=_BOTH_GRIDS),
    FeatureType('solid_probe', 'Solid probe', 'probe', 'point',
                params=(Param('component', 'choice', 'u_abs', label='Quantity',
                              choices=OUTPUT_ARRAYS + ENERGY_DENSITY_FIELDS,
                              choice_labels=tuple((name, SolidRenderer.field_infos[name].label)
                                                  for name in OUTPUT_ARRAYS + ENERGY_DENSITY_FIELDS),
                              help='Output field sampled at the point (unit of that field); the energy densities on '
                                   'a triangle mesh only.',
                              tip='Quantity recorded at the point.'),),
                object_kinds=('point',), max_entities=1, result_kind='timeseries',
                summary='Records a displacement, strain or stress at a point after every solve.',
                description='Samples one output field in the cell containing the point (one sample per re-solve and '
                            'one per stationary iteration or end-of-run flush, all at t = 0 in a stationary study); on '
                            'a triangle mesh the nodal value at the point (the mesher puts a node there; strain and '
                            'stress there are the node-averaged display values).',
                apply=_apply_probe, result=_probe_result, validate=validate_points_inside, grids=_BOTH_GRIDS),
    frequency_probe_feature(_apply_frequency_probe, ('u_x', 'u_y'), (_TRIANGLES,)),
    replace(resonance_table_feature('nearest the target (the target_frequency setting) first, ranked by |omega - '
                                    'omega_target|'), grids=_BOTH_GRIDS),
    OPTOMECHANICAL_COUPLING_FEATURE,
    FeatureType('electrode', 'Electrode', 'boundary_condition', 'boundary', allow_interior=False,
                params=_ELECTRODE_PARAMS, result_kind='table',
                summary='Puts an electrode on the selected domain edges of a piezoelectric solid: a fixed voltage, '
                        'ground, or an isolated electrode.',
                description=f'Electrode on outer faces (every node of the face, corners included). {_ELECTRODE_TEXT} '
                            f'{_ELECTRODE_RESULT_TEXT}',
                apply=_apply_electrode, result=_electrode_result),
    FeatureType('electrode_domain', 'Electrode (domain)', 'domain_condition', 'domain', params=_ELECTRODE_PARAMS,
                result_kind='table',
                summary='Makes the selected domains (for example a drawn metal film) one electrode of a piezoelectric '
                        'solid.',
                description='Electrode on every corner node of the selected cells (a drawn metal electrode; give it '
                            'its own stiffness and density, or none). '
                            f'{_ELECTRODE_TEXT} {_ELECTRODE_RESULT_TEXT}',
                apply=_apply_electrode_domain, result=_electrode_result),
    ADMITTANCE_FEATURE,
)

_FIELD_INFOS = tuple(SolidRenderer.field_infos.values())
# om_integrand, the imaginary parts and the energy densities: not in state()
_STATE_FIELD_INFOS = tuple(info for info in _FIELD_INFOS
                           if not info.requires_study and info.name not in ENERGY_DENSITY_FIELDS)
_SLOT_DOCS = {
    'temperature_rise': ('K', f'Coupled temperature above the stress-free state ({THERMAL_STRAIN_COUPLING}: heat T - '
                              'T0), added to the temperature_change features (field temperature_rise_ext); each '
                              'transfer replaces it.'),
    'body_force_x': ('N/m^3', 'Coupled body force density along x (an optical or Lorentz force; several such '
                              "couplings add), added to the body_force features; each transfer replaces it; void "
                              'cells drop it.'),
    'body_force_y': ('N/m^3', 'Coupled body force density along y, as body_force_x.'),
}

SOLID_MECHANICS = PhysicsType(
    name=SOLID_TYPE_NAME,
    label='Solid Mechanics (2D)',
    default_instance_name='solid',
    family='',
    group='structural',
    spaces_2d=('planar', 'cross_section', 'axisymmetric'),
    space_settings=AXISYMMETRIC_SPACE_SETTINGS,
    summary='Static deformation and stress of solids under supports, loads and thermal expansion: the cross-section '
            'of a long structure, a thin plate seen from the top, or a body of revolution.',
    description='2D static linear elasticity; setting out_of_plane: plane strain (default; eps_zz = 0: the '
                'cross-section of a structure long along z, e.g. a waveguide), plane stress (sigma_zz = 0: a thin '
                'plate) or generalized plane strain (uniform eps_zz set by the net axial force); setting geometry '
                'axisymmetric: an (r, z) body of revolution with the hoop strain (whole-ring forces in N). The rest '
                'is written for plane strain. Small strain, isotropic cells (youngs_modulus, poisson_ratio) or the '
                'anisotropic c_ij entries (c11 .. c66 in Voigt model axes, NaN = the isotropic value; a crystal entry '
                'sets them) with anisotropic '
                'thermal expansion alpha_xx / alpha_yy / alpha_zz: sigma = C (eps - alpha dT), dT = temperature_rise '
                f'(coupled, {THERMAL_STRAIN_COUPLING}) + temperature_rise_ext (temperature_change features). '
                'Discretisation: bilinear elements on the cells of the physical window, displacement on the cell '
                'corners, block-Jacobi PCG (Automatic on the gpu backend: precision mixed, a float32 inner CG with '
                'float64 residual refreshes to the same tolerance; GPUs with slow float64 arithmetic); outputs at the '
                'cell centres (u = mean of the 4 corners; strain, stress '
                'and von Mises from the centre gradients), 0 in void cells (no youngs_modulus or c11 > 0: not '
                'solved, carry nothing), copied edge-extended into an EM PML ring. Units per unit z length: forces and reactions '
                'N/m. strain_xy is the tensor shear strain (half the engineering strain); stress_zz = nu (sigma_xx + '
                'sigma_yy) - E alpha_zz dT is the out-of-plane stress plane strain implies; von_mises includes it. '
                'Faces without a support or load are traction free. Every solid region (cells with stiffness joined '
                'through edges or corners) needs supports that hold u_x and u_y and stop rotation, e.g. rollers on '
                'one x and one y face (validation checks the directions, the build each region). A stationary study '
                'solves it after the physics it reads; in a time-dependent study it re-solves only when a material, '
                'support, load or coupled input changed (quasi-static, no inertia). An eigenfrequency study finds the '
                'undamped in-plane vibration modes (plane strain) nearest target_frequency instead: K u = omega^2 M u '
                'with the consistent mass of rho (needed in every solid cell) on the host (eigen solver role '
                'arpack_shift_invert: symmetric shift-invert Lanczos on a real SuperLU factor), supports held at 0, '
                'loads / thermal strain ignored (no prestress: setting stress_stiffening on a triangle mesh adds '
                'it); with the axial_wavenumber setting k_z the guided '
                '(2.5D) modes u ~ exp(i (k_z z - omega t)) instead, u_z coupled (dispersion f(k_z) of a waveguide: '
                'flexural, torsional, longitudinal branches; supports optional for k_z > 0); one snapshot '
                'per mode labelled "resonance k: f = <Hz> Hz" with the mode shape scaled to a peak node |u| of 1 m '
                '(strain and stress per that shape, no thermal part; support reactions are not available), the '
                'resonance_table feature lists the frequencies; an optomechanics study solves the same modes after '
                f"an optical mode ({', '.join(OPTICAL_TYPES)}) and its optomechanical_coupling feature tabulates "
                'g0. Output u_x, u_y (u_z) are the displacement [m] and u_abs its magnitude (u is the flow velocity '
                'in laminar_flow, elastic_waves names its velocity vx / vy). elastic_strain_* = strain - alpha '
                'dT (elastic_strain_zz = -alpha_zz dT), the stress-producing part. Optics: '
                f'{STRAIN_OPTIC_COUPLING} (strain -> EM index) and {OPTICAL_FORCE_COUPLING} (em_fdfd_ez -> the '
                f'body_force_x / body_force_y slots). Setting piezoelectric: a nodal potential on the same elements '
                'with the piezo_e* / eps_dc_* material arrays and electrode features, static and modal (short / open '
                'circuit) coupled solves on the host. A frequency_response study solves the driven harmonic response '
                '(K (1 + j mechanical_loss_factor) - omega^2 M) u = f per frequency (host complex SuperLU; loads, '
                'prescribed displacements and electrode voltages are amplitudes at phase 0; a temperature change or '
                'coupled temperature is static, never a drive: with stress_stiffening its thermal stress joins the '
                'prestress), one snapshot per '
                'frequency with the real parts and u_abs = |u|; with piezoelectric the admittance feature gives Y(f) '
                'and a Butterworth-Van Dyke fit. A 3D model (size_z set) uses '
                f'{SOLID_3D_TYPE_NAME}. The instance is usually named solid. On a triangle mesh (mesh kind '
                "'triangles', mesh plan S7.5): Lagrange elements of the discretization setting (quadratic P2 "
                'default, isoparametric on curved boundaries) on the window elements with stiffness, one host '
                'SuperLU of K per assembly at its first static solve (linear solver sparse_direct, Automatic there), '
                'plane strain / plane stress / generalized plane strain (the same superposition, conditions exact by '
                'the same quadrature) / axisymmetric, c_ij, thermal strain, every support and load feature (Fixed '
                '(domain) holds every node of the selected elements; segment loads in any direction), probes at point '
                'nodes, the eigenfrequency study and the elastic frequency_response study (stationary, '
                'time_dependent, eigenfrequency and frequency_response; there also: initial_stress_* material entries '
                '(total stress sigma0 + C eps - beta dT, relaxed by static solves), setting stress_stiffening (K + '
                'K_sigma of the static state in the eigenfrequency / frequency_response pencils), damped modes with '
                'a mechanical_loss_factor, free bodies about a target_frequency > 0 and in frequency responses, the '
                'frequency_probe feature and complex support reactions per frequency; not yet: piezoelectric (so no '
                'admittance), '
                f'guided modes, {STRAIN_OPTIC_COUPLING}; the optomechanics study with geometry axisymmetric and '
                'em_fdfd_bor optics only); state() and coupled slots are one '
                'value per element (centroid), displayed fields nodal (displacement exact, strain / stress averaged '
                'over the solid elements at each node), the PML ring void unless an elastic PML. Triangle meshes, '
                'eigenfrequency study only: settings pml_<face> make the ring an elastic PML (fixed complex stretch '
                'set at target_frequency, outer side clamped, complex modes with a radiation Q, PML modes dropped; '
                'stress_stiffening sees the ring as a clamped elastic continuation in its static state), and '
                'azimuthal_order m >= 1 (geometry axisymmetric, isotropic materials) solves the modes u_r = U_r '
                'cos(m phi), u_z = U_z cos(m phi), u_phi = U_phi sin(m phi) with three amplitudes per node (fields '
                'u_x, u_y and u_z = U_phi; strain / stress outputs the cos parts at phi = 0; rings weighted by pi; the '
                'axis holds u_z = 0 and ties u_phi = -u_r at m = 1, holds all three at m >= 2; m = 1 has the sideways '
                'translation and tilt as rigid motions, m >= 2 none, so supports are optional there; stress_stiffening '
                'takes the initial stress alone). Triangle meshes, planar plane strain / plane stress or axisymmetric '
                '(azimuthal order 0), no PML: setting geometric_nonlinearity makes every static solve total-Lagrangian '
                '(Green-Lagrange strain, second Piola-Kirchhoff stress outputs, Newton over load_steps, support '
                'reactions the nominal forces of the internal force, not K u - f) and linearises the eigenfrequency '
                '/ frequency_response pencils about that state (tangent K_T); the Prescribed displacement (domain) '
                'feature holds chosen components of whole domains. Display fields strain_energy_density / '
                'kinetic_energy_density '
                '(triangle meshes; J/m^3; not in state()): a static solve 1/2 sigma : eps_el (no initial stress) and 0, '
                'a shown mode or response the time averages 1/4 Re(sigma : eps_el*) and 1/4 rho omega^2 |u|^2 at the '
                'mode / drive frequency (azimuthal order: twice the phi mean, so ring integrals give the totals); in a '
                'PML ring they use the unstretched gradients (not physical there).',
    settings=(_GEOMETRY_PARAM, AZIMUTHAL_ORDER_PARAM, _OUT_OF_PLANE_PARAM, _THICKNESS_PARAM, _AXIAL_FORCE_PARAM,
              _BENDING_PARAM, _BENDING_MOMENT_X_PARAM, _BENDING_MOMENT_Y_PARAM, _TARGET_FREQUENCY_PARAM,
              STRESS_STIFFENING_PARAM, *GEOMETRIC_NONLINEARITY_PARAMS, _AXIAL_WAVENUMBER_PARAM, _PIEZOELECTRIC_PARAM,
              _DISCRETIZATION_PARAM, *_pml_params()),
    features=_FEATURES,
    material_properties=(MATERIAL_PROPERTIES + (DENSITY_PARAM, PHOTOELASTIC_P44_PARAM) + PIEZO_MATERIAL_PROPERTIES
                         + STIFFNESS_TENSOR_PROPERTIES + LOSS_PROPERTIES + INITIAL_STRESS_PROPERTIES),
    material_schema={**MECHANICAL_SCHEMA, **DENSITY_SCHEMA, **SHEAR_PHOTOELASTIC_SCHEMA},
    optional_material_schema={**OPTIONAL_PIEZO_SCHEMA, **OPTIONAL_STIFFNESS_SCHEMA, **OPTIONAL_LOSS_SCHEMA,
                              **OPTIONAL_PRESTRESS_SCHEMA},
    state_fields=_STATE_FIELD_INFOS,
    output_fields=_FIELD_INFOS,
    source_slots=_SLOT_DOCS,
    supported_studies=('stationary', 'time_dependent', EIGENFREQUENCY_STUDY, OPTOMECHANICS_STUDY,
                       FREQUENCY_RESPONSE_STUDY),
    default_study={'type': 'stationary'},
    # eigen: the real symmetric eigsh_shifted pencil (piezo: indefinite), cudss_mixed refined to full accuracy per solve
    solver_roles={'linear': SOLID_LINEAR_ROLE, 'eigen': eigen_role(_MODAL_STUDIES, mixed_factorization=True,
                                                                auto_factorizations={'triangles': 'cudss'})},
    frequency_sweep_methods=('direct', 'adaptive', 'modal'),   # frequency_response.HarmonicSystem.plan_reduced
    backends=('gpu', 'cpu'),   # cpu: scipy CSR + numpy block-Jacobi twin; modal, guided, piezo solves are host anyway
    create=_create_solid,
    validate=_validate_solid,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    domain_requirements=_domain_requirements,   # the elastic PML ring (triangles)
    grids=_BOTH_GRIDS,   # triangles: solver_fem.SolidMechanicsFEMPhysics (create dispatches on the mesh kind)
    eigen_searches=('nearest', 'targets', 'band'),   # modal.MultiShiftModes (piezoelectric: band refused in validate)
)


# ---- coupling: heat -> solid thermal strain ----
def _writes_thermal_strain(vctx, coupling_spec):
    """One overwriting writer of temperature_rise (model.validate's writer rule rejects a second one)."""
    return {'temperature_rise': ''}


def _create_thermal_strain(ctx, coupling_spec, values, every):
    """CouplingTerm of heat T into temperature_rise (the transform reads the heat rise in solve precision; 2D or 3D
    buffer of the material grid's shape)."""
    return CouplingTerm(coupling_spec.source, 'T', coupling_spec.target, 'temperature_rise',
                        make_temperature_rise_transform(tuple(ctx.domain.materials.shape), ctx.xp), every=every,
                        name=coupling_spec.name,
                        after_source_step=ctx.sim.runner.strides[coupling_spec.source] == every)


def _validate_thermal_strain(vctx, coupling_spec, values):
    """Info when no selected material sets a thermal expansion (the writer rule is model.validate's)."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    path = ('coupling', coupling_spec.name)
    issues = []
    expands = False
    for material in vctx.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:
            expands = True
            continue
        if material.selection and any(evaluates_to(properties.get(name), vctx.parameters, lambda value: value != 0.0)
                                      for name in ('alpha', 'alpha_xx', 'alpha_yy', 'alpha_zz')):
            expands = True
    if not expands:
        issues.append(Issue('info', path, 'no material sets a thermal expansion coefficient (alpha), so the '
                                          'temperature rise causes no strain'))
    return issues


HEAT_SOLID_THERMAL_STRAIN = CouplingType(
    THERMAL_STRAIN_COUPLING, 'Thermal expansion (heat -> solid mechanics)', HEAT_TYPE_NAME,
    (SOLID_TYPE_NAME, SOLID_3D_TYPE_NAME),
    summary='Lets the solid expand with the temperature of the heat physics, causing thermal strain and stress. The '
            "heat physics' reference temperature is the stress-free temperature.",
    description='Thermal strain: each transfer writes dT = T - T0 of the heat physics (its rise in solve precision, '
                'not the float32 T; T0 = the stress-free temperature) into the temperature_rise slot of a '
                f'{SOLID_TYPE_NAME} (2D, with a 2D heat) or {SOLID_3D_TYPE_NAME} (3D, with a 3D heat) physics, which '
                're-solves with the thermal load C alpha dT (alpha_xx / alpha_yy / alpha_zz; 2D plane strain turns '
                'alpha_zz dT into out-of-plane stress, in 3D it is an ordinary expansion along z). Window only (the '
                'PML ring of the slot reads 0). Stationary study: applied after the heat solve; time-dependent study: '
                'right after each heat step. One writer of the slot (a second one, or a field_transfer to '
                'temperature_rise, is a validation error); temperature_change features add to it. For the optics (2D '
                f'only) add {STRAIN_OPTIC_COUPLING} (solid -> EM): with its deformation on, heat_em_thermo_optic on the '
                'same EM physics drops its (1 + alpha dT) factor, as the solid strain then supplies the deformation.',
    create=_create_thermal_strain,
    validate=_validate_thermal_strain,
    writes=_writes_thermal_strain,
    # across meshes: the transform reads heat.temperature_rise() on the heat window, so it runs there and the rise moves
    mesh_transfer='value', value_kind='interpolate')


# ---- coupling: solid -> EM strain-optic index (E13 'index' contributor) ----
def _create_strain_optic(ctx, coupling_spec, values, every):
    """One 'index' contributor per index slot of the target type, a CouplingGroup when several."""
    return index_contributor(ctx, coupling_spec, every, 'strain_xx',
                             lambda slot: make_strain_optic_transform(slot, values['deformation']))


def _validate_strain_optic(vctx, coupling_spec, values):
    """Error without an index slot on the target; info for an in-plane (top-view) EM target and when nothing changes."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    path = ('coupling', coupling_spec.name)
    physics_type = target_type(vctx, coupling_spec)
    if physics_type is None:
        return []
    if not index_slots(physics_type):
        return [Issue('error', path, f"target physics '{coupling_spec.target}' ({physics_type.name}) has no "
                                     f"permittivity slot ({', '.join(INDEX_SLOTS)})")]
    issues = []
    source = next((physics for physics in vctx.spec.physics if physics.name == coupling_spec.source), None)
    source_values = (vctx.values(SOLID_MECHANICS.settings, source.settings) if source is not None else None) or {}
    if is_axisymmetric(source_values):
        issues.append(Issue('info', path,
                            f"'{coupling_spec.source}' is axisymmetric: the hoop strain is the zz strain of the index "
                            'change (eps_phiphi of an (r, z) EM target), while the deformation term sqrt(1 + strain_xx '
                            '+ strain_yy) leaves the hoop stretch out'))
    elif physics_type.name != _CROSS_SECTION_EM_TYPE and source_values.get('out_of_plane') != PLANE_STRESS:
        issues.append(Issue('info', path,
                            f"'{coupling_spec.target}' is an in-plane EM physics: the solid's plane strain (eps_zz = 0) "
                            'describes a structure long along z; a top view of a thin film is closer to plane stress '
                            "(the solid's out_of_plane setting), else the out-of-plane strain and its index change are "
                            'approximate there'))
    photoelastic = False
    for material in vctx.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:
            photoelastic = True
            continue
        if material.selection and any(evaluates_to(properties.get(name), vctx.parameters, lambda value: value != 0.0)
                                      for name in ('photoelastic_p11', 'photoelastic_p12')):
            photoelastic = True
    if values['deformation']:
        issues.append(Issue('info', path, 'deformation on: the stretch image n sqrt(1 + strain_xx + strain_yy) is exact '
                                          'for isotropic in-plane strain only; under uniaxial strain (e_xx != e_yy) '
                                          'the anisotropic part of the transformation-optics image (mu or eps_xx / '
                                          'eps_yy) is left out, a first-order phase error along the stretched axis'))
    if not photoelastic and not values['deformation']:
        issues.append(Issue('warning', path, 'no selected material sets photoelastic_p11 / photoelastic_p12 and '
                                             'deformation is off: the coupling changes no index'))
    return issues


def _strain_optic_coupling_type(em_types):
    """solid_em_strain_optic from solid_mechanics to every registered family-'em' type (sorted names em_types)."""
    return CouplingType(
        STRAIN_OPTIC_COUPLING, 'Strain-optic effect (solid mechanics -> electromagnetic waves)', SOLID_TYPE_NAME,
        em_types,
        params=(Param('deformation', 'bool', True, label='Include deformation',
                      help='Also map the deformation of the structure onto the fixed grid: n times sqrt(1 + strain_xx '
                           '+ strain_yy) (total strain, the eps_zz det F image of the in-plane stretch, the phase-index '
                           'convention of heat_em_thermo_optic). On, heat_em_thermo_optic on the same EM physics drops '
                           'its (1 + alpha dT) factor (the solid strain replaces it); off, only the photoelastic part. '
                           'Exact for isotropic in-plane strain (thermal) only: the full image of a stretch e_xx != '
                           'e_yy is anisotropic (E_z: mu_xx, mu_yy times 1 -/+ (e_xx - e_yy); H_z: eps_xx, eps_yy '
                           'likewise), which no index slot holds, so the phase along a stretched axis (a strip under '
                           'uniaxial tension, the hoop strain of a ring) is off to first order in e_xx - e_yy.',
                      tip='Also account for the structure growing or shrinking, not only for the stress-induced '
                          'index change.'),),
        summary='Changes the refractive index of a wave physics from the strain of the solid mechanics: stress-induced '
                'birefringence and the stretching of the structure.',
        description='Strain-optic effect: each transfer writes, for every permittivity component ii of the EM type '
                    '(E_z types: zz; H_z types: xx, yy; em_modes: all three), the index change dn_ii = (n0_ii + '
                    'dn_pe) sqrt(1 + w (strain_xx + strain_yy)) - n0_ii with the photoelastic part dn_pe = -1/2 '
                    'n0_ii^3 (photoelastic_p11 e_ii + photoelastic_p12 (e_jj + e_kk)) of the elastic strain e = '
                    'elastic_strain_* (strain - alpha dT, e_zz = -alpha_zz dT: the measured dndT of a free material '
                    'already contains its free thermal expansion) and w = 1 with deformation (total strain, void cells '
                    'none: a first-order local image like the alpha factor of heat_em_thermo_optic; exact only for '
                    'isotropic in-plane strain, see the deformation setting). Isotropic (or '
                    'cubic, crystal axes along x, y, z) photoelastic coefficients p11 / p12; the shear term p44 '
                    'strain_xy (an off-diagonal eps) is not modelled. It is an E13 index contributor: with '
                    'heat_em_thermo_optic / electrostatics_em_pockels on the same physics the slot gets eps = (n0 + '
                    'the sum of dn)^2, all with one every. Plane strain is exact for a cross-section (em_modes); an '
                    'in-plane (top-view) EM target gets a validation info. Stationary study: after the solid solve; '
                    'frequency_domain / mode_analysis: once after the background solve. Source solid_mechanics only: '
                    f'from a {SOLID_3D_TYPE_NAME} physics it is not modelled yet (the em_fdtd_3d material index cannot '
                    'hold three continuous permittivity components, and the 3D deformation image is open).',
        create=_create_strain_optic,
        validate=_validate_strain_optic,
        writes=writes_index_slots)


# ---- coupling: em_fdfd_ez -> solid optical force ----
def _writes_optical_force(vctx, coupling_spec):
    return {'body_force_x': 'sum', 'body_force_y': 'sum'}


def _create_optical_force(ctx, coupling_spec, values, every):
    """CouplingGroup of the two E13 'sum' body-force terms sharing one OpticalForceDensity."""
    force_x, force_y = make_optical_force_transforms(ctx.domain)
    terms = [CouplingTerm(coupling_spec.source, 'ez_re', coupling_spec.target, slot, transform, every=1,
                          name=f'{coupling_spec.name}.{slot}', contribution='sum')
             for slot, transform in (('body_force_x', force_x), ('body_force_y', force_y))]
    return CouplingGroup(terms, every=every, name=coupling_spec.name,
                         after_source_step=ctx.sim.runner.strides[coupling_spec.source] == every)


OPTICAL_FORCE_TYPE = CouplingType(
    OPTICAL_FORCE_COUPLING, 'Optical force (electromagnetic waves -> solid mechanics)', _FDFD_EZ_TYPE_NAME,
    SOLID_TYPE_NAME,
    summary='Applies the force light exerts on material boundaries (radiation pressure, gradient force) to the solid '
            'mechanics. Needs a frequency-domain wave physics with the electric field out of the plane.',
    description='Optical (Maxwell-stress) force: the time-averaged force density f = -(eps0 / 4) |E_z|^2 grad eps_r '
                '(peak phasor E_z of the last em_fdfd_ez solve; non-magnetic, no electrostriction, the absorption '
                'force of lossy cells and the sigma_e / pole part of eps ignored) as a face jump: each cell face '
                'carries -(eps0 / 4) |E_z|^2 (eps_r jump across it) times its length, split between its solid cells '
                '(c11 > 0, else youngs_modulus > 0, as the solver) and divided by their area, so the force on a dielectric boundary lands in the '
                'solid next to it in full (E_z is tangential to every in-plane boundary, so |E_z|^2 is continuous '
                'there); into body_force_x / body_force_y as an additive contribution (it sums with '
                'magnetic_solid_lorentz_force, all with one every; a field_transfer into those slots cannot be '
                'combined with it). Only em_fdfd_ez: the in-plane '
                'E of em_fdfd_hz has a discontinuous normal component that this face form does not handle. Use a '
                'stationary study (a frequency_domain study does not feed harmonic solutions back).',
    create=_create_optical_force,
    writes=_writes_optical_force,
    # across meshes: computed on the EM mesh (the solid's stiffness painted there too), N/m^3 moved conservatively
    mesh_transfer='value', partner_materials=True)


def _selftest_cases(em_types):
    """The 2D solid SelfTestCases (optomechanics and CPU-vs-GPU included; the em_fdfd_ez strain-optic / optical-force
    chains only inside solid_optics_cpu_gpu; imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import SELFTEST_CASES
    from .selftests_anisotropy import ANISOTROPY_CASES
    from .selftests_backend import backend_cases
    from .selftests_eigen_search import EIGEN_SEARCH_CASES
    from .selftests_eigenfrequency import EIGENFREQUENCY_CASES
    from .selftests_fem import FEM_CASES
    from .selftests_formulations import FORMULATION_CASES
    from .selftests_frequency_response import FREQUENCY_RESPONSE_CASES
    from .selftests_guided_modes import GUIDED_MODES_CASES
    from .selftests_azimuthal import AZIMUTHAL_CASES
    from .selftests_nonlinear import NONLINEAR_CASES
    from .selftests_piezo import PIEZO_CASES
    from .selftests_pml2d import PML_2D_CASES
    from .selftests_prestress_modes import PRESTRESS_CASES
    from .selftests_prestress_string import PRESTRESS_STRING_CASES
    from .selftests_reduced_sweep import REDUCED_SWEEP_CASES
    from .selftests_segment_load import SEGMENT_LOAD_CASES
    return (SELFTEST_CASES + SEGMENT_LOAD_CASES + FORMULATION_CASES + EIGENFREQUENCY_CASES + EIGEN_SEARCH_CASES
            + GUIDED_MODES_CASES + PIEZO_CASES + ANISOTROPY_CASES + FREQUENCY_RESPONSE_CASES + FEM_CASES + PRESTRESS_CASES
            + PRESTRESS_STRING_CASES + PML_2D_CASES + AZIMUTHAL_CASES + NONLINEAR_CASES + REDUCED_SWEEP_CASES
            + optomechanics_cases(em_types) + backend_cases(em_types))


def register(registry):
    """solid_mechanics, solid_mechanics_3d (definition3d.register: type, template, 3D selftest loader), the
    heat_solid_thermal_strain coupling (both targets), solid_em_strain_optic (with family-'em' types) and
    em_solid_optical_force (with em_fdfd_ez), the 'heated_waveguide_stress' and the optomechanics templates
    (definition_optomechanics), then the 2D selftest loader (bound to the registered em types)."""
    registry.register_physics(SOLID_MECHANICS)
    definition3d.register(registry)
    registry.register_coupling(HEAT_SOLID_THERMAL_STRAIN)
    physics_types = registry.physics_types()
    em_types = tuple(sorted(physics_type.name for physics_type in physics_types if physics_type.family == 'em'))
    if em_types:
        registry.register_coupling(_strain_optic_coupling_type(em_types))
    if any(physics_type.name == _FDFD_EZ_TYPE_NAME for physics_type in physics_types):
        registry.register_coupling(OPTICAL_FORCE_TYPE)
    registry.register_template(
        'heated_waveguide_stress', HEATED_WAVEGUIDE_STRESS_LABEL, HEATED_WAVEGUIDE_STRESS_DESCRIPTION,
        heated_waveguide_stress_spec,
        summary='Thermal stress in the cross-section of a silicon waveguide under a gold heater: the heat spreads into '
                'the silica, everything expands, and the silicon core ends up compressed. Steady state.')
    register_optomechanics_templates(registry, em_types)
    registry.register_selftests(partial(_selftest_cases, em_types))
