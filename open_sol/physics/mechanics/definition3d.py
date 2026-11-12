"""Model-layer definition of the 'solid_mechanics_3d' physics type (static 3D elasticity on a Domain3D or a tetrahedron
mesh and its vibration modes: features, validation, memory / time estimates) plus its template and selftests;
definition.register calls register here (the heat_solid_thermal_strain coupling lives there). No cupy-importing sibling
module at module level: the solvers are imported in create."""
from dataclasses import replace
from functools import partial

import numpy as np

from ...core import time_estimate as te
from ...core.constants import FACES_3D
from ...core.memory_estimate import (BOOL_BYTES, COMPLEX128_CSR_ENTRY_BYTES, FLOAT32_BYTES, FLOAT64_BYTES,
                                     FLOAT64_CSR_ENTRY_BYTES, INT32_BYTES, MemoryEstimate, combine,
                                     host_counted_lu_bytes, host_real_lu_bytes)
from ...fem.diffusion import lagrange_entries_per_dof
from ...core.kernels import numba_available
from ...fem.elasticity3d import (TANGENT_DEVICE_GRADIENT_CACHE_BYTES, TANGENT_GRADIENT_CACHE_BYTES,
                                 device_tangent_chunk_elements, tangent_chunk_element_bytes)
from ...fem.tangent_kernel3d import element_flops
from ...fem.fixed_pattern import EXPAND_ENTRIES
from ...fem.squeeze_film import GAS_TABLE_COLUMNS, GAS_TABLE_UNITS, gas_damping_totals
from ...model.definition_helpers import (CG_ELASTIC_3D_LINEAR_ROLE, DISCRETIZATION_PARAM, SQUEEZE_FILM_GAS_PARAMS,
                                         cg_solver_issues, cudss_external, eigen_factorization, eigen_role,
                                         eigen_search_counts, eigen_subspace_size, host_direct_back_end,
                                         element_order, linear_solver_mixed, linear_solver_multigrid,
                                         linear_solver_precision, on_tetrahedra,
                                         point_cells, reduced_sweep_memory_items, validate_points_inside,
                                         validated_linear_precision)
from ...model.schema import DomainRequirements, FeatureType, Issue, Param, PhysicsType, feature_path
from ...solvers.arpack_shift_invert import APPLICATIONS_PER_BASIS_GUESS, CUDSS_FACTORIZATIONS, arnoldi_basis_size
from ...solvers.host_direct import (HOST_DIRECT_BACK_ENDS, double_of, factor_label, host_direct_precision,
                                    solver_host_bytes, superlu_bytes_scale)
from ...solvers.pcg import workspace_bytes
from ...solvers.sparse_direct import CUDSS_BACK_ENDS, SPLU_3D_LIMIT_UNKNOWNS, SPLU_3D_WARN_UNKNOWNS
from .definition_common import (DENSITY_PARAM, EIGENFREQUENCY_STUDY, GEOMETRIC_NONLINEARITY_PARAMS,
                                INITIAL_STRESS_PROPERTIES_3D, LOSS_PROPERTIES, MATERIAL_PROPERTIES, MODAL_STUDIES,
                                PIEZO_MATERIAL_PROPERTIES_3D, STIFFNESS_TENSOR_PROPERTIES_3D,
                                STRESS_STIFFENING_PARAM, SUPPORT_FEATURES, apply_body_force, apply_temperature_change,
                                background_context, displacement_param, eigen_search_issues, electrode_result,
                                follow_held_values, follow_load, following, free_body_allowed, frequency_probe_feature,
                                material_issues, materials_setting, modal_kept_count, modal_study_issues,
                                nonlinear_state_infos, outer_faces, piezo_material_issues, plain_value, prestress_issues,
                                register_support, resonance_table_feature, solid_linear_solver, solid_study,
                                stiffness_follows_state, support_holders, support_result, union_masks)
from .definition_frequency_response import ADMITTANCE_FEATURE, FREQUENCY_RESPONSE_STUDY, frequency_response_issues
from .definition_optomechanics import OPTOMECHANICAL_COUPLING_FEATURE, OPTOMECHANICS_STUDY
from .fields3d import (ELECTRIC_OUTPUT_ARRAYS_3D, ENERGY_OUTPUT_ARRAYS_3D, HARMONIC_IMAG_FIELDS_3D, INPUT_ARRAYS_3D,
                       OUTPUT_ARRAYS_3D)
from .materials import (DENSITY_SCHEMA, LOSS_ARRAYS, MECHANICAL_SCHEMA, OPTIONAL_LOSS_SCHEMA, OPTIONAL_PIEZO_SCHEMA_3D,
                        OPTIONAL_PRESTRESS_SCHEMA_3D, OPTIONAL_STIFFNESS_SCHEMA_3D, PIEZO_DEFAULTS_3D,
                        PIEZO_TENSOR_ARRAYS_3D, SOLVER_PRECISIONS, STIFFNESS_TENSOR_ARRAYS_3D)
from .modal import EXTRA_EIGENVALUES, LANCZOS_TOLERANCE_PER_CG, modal_table_result
from .multigrid3d import (CHEBYSHEV_DEGREE, POWER_ITERATIONS, coarsest_host_bytes, hierarchy_bytes,
                          level_element_shapes)
from .pml3d import DEFAULT_PML_CELLS, DEFAULT_PML_REFLECTION, PMLPadding
from .renderer3d import SolidRenderer3D
from .squeeze_film import GAP_SIDES, VENT_CHOICES
from .selftests_3d import HEATED_CHIP_STRESS_DESCRIPTION, HEATED_CHIP_STRESS_LABEL, heated_chip_stress_3d_spec
from .selftests_bloch3d import BAND_DIAGRAM_3D_DESCRIPTION, BAND_DIAGRAM_3D_LABEL, solid_bloch_band_diagram_3d_spec

SOLID_3D_TYPE_NAME = 'solid_mechanics_3d'               # = selftests_3d.SOLID_3D_TYPE_NAME (not imported)
THERMAL_STRAIN_COUPLING = 'heat_solid_thermal_strain'   # = selftests_3d.THERMAL_STRAIN_COUPLING
_AXES = ('x', 'y', 'z')
_DISPLACEMENTS = ('u_x', 'u_y', 'u_z')   # load / held value names (follow_load / follow_held_values)
_TRACTIONS = ('t_x', 't_y', 't_z')
_BODY_FORCES = ('f_x', 'f_y', 'f_z')
_TETRAHEDRA = 'tetrahedra'
_GRIDS = ('tensor', _TETRAHEDRA)   # tetrahedra: solver_fem3d.SolidMechanicsFEMPhysics3D (create dispatches)
_INTERFACE_FEATURES = ('fixed', 'displacement', 'boundary_load')   # also on object interface faces (tetrahedra)
# tetrahedra: SuperLU fill of the vector P1 / P2 stiffness (3 x 3 node blocks) over the 7-point 3D heuristic at equal
# unknowns: fem.diffusion3d.FEM_FILL_FACTOR_3D x ~2.1 (9 block entries vs 3^(4/3)); nodal display outputs kept on the
# host. Uncalibrated: compare summary.json solvers <physics>.roles.linear.counters.host_factor_bytes.
_TET_FILL_FACTOR = {1: 4.0, 2: 6.0}
# Host SuperLU seconds of the vector stiffness (and K - sigma M) over the 7-point 3D curves at equal unknowns. P2 from
# solid_fem3d_roller_box_modes (cpu profile 2026-10-22, 24k free dofs: factor 2.07 s vs 0.66 s, solve 0.018 s vs 0.006
# s); P1 a guess (smaller element blocks), uncalibrated. Pricing at stored entries / 7 instead overestimated ~60x.
_TET_LU_TIME_SCALE = {1: 1.5, 2: 3.1}
_TET_SOLVE_TIME_SCALE = {1: 1.5, 2: 3.0}
# Bloch cells (theory, George nested dissection on a cube: level l holds 2^-l of the flops ~ s^3 and 2^(-l/3) of the
# fill ~ s^2; the top levels cut x, y, z in turn): the first cut across a periodic axis must close its ring with two
# planes, so p periodic axes double the separators of the first p levels (flops x 8, fill x 4 there)
_PERIODIC_SEPARATOR_WIDTH = 2.0
_TET_NODAL_OUTPUTS = 22
_TET_ELECTRIC_NODAL_OUTPUTS = 7
# geometric nonlinearity: Newton iterations per load step (a priori: a linear predictor and quadratic convergence to
# the default 1e-8 take ~4-6) and the host numpy bytes per COO entry of one tangent + internal-force assembly on the
# batched formulas (no numba: the total-Lagrangian rule, P1 1 point, P2 the 14-point degree-4 rule, 6 Voigt rows,
# operator + stress passes); uncalibrated
_NEWTON_ITERATIONS_PER_STEP = 5
_TANGENT_BYTES_PER_COO = {1: 400.0, 2: 6000.0}
# the element kernel (fem.tangent_kernel3d; device, or numba on the host): its flops priced as dense flops, the block
# sum per COO entry (host bincount / device scatter_add: value + slot read, the data entry updated) and the launches
# (gather, kernel, force and block sums) per chunk; uncalibrated
_TANGENT_SCATTER_BYTES_PER_COO = 24.0
_ELEMENT_KERNEL_LAUNCHES = 6
_TANGENT_RULE_POINTS = {1: 1, 2: 14}   # tetrahedron_rule(4 (order - 1)): TangentAssembler3D's rule and gradient cache
# tensor bricks (kernels3d elastic_product / elastic_mass_product, a priori counts of the gpu Walsh-basis gather): each
# element is visited by its 8 nodes; a stiffness visit = 24 loads, Walsh transform (72 add / sub), gradients (36),
# stresses (~56), back-scaling (36), mode sums (15), the own-corner evaluation (21) and 3 accumulations; a mass visit =
# transform 72, scaling 24, evaluation 21 + 3; 6 divisions per node (inverse widths)
_BRICK_VISITS_PER_ELEMENT = 8
_BRICK_VISIT_FLOPS = 240
_BRICK_ELEMENT_FLOPS = _BRICK_VISITS_PER_ELEMENT * _BRICK_VISIT_FLOPS   # stiffness product per element
_BRICK_NODE_FLOPS = 6 * te.DIVISION_FLOPS                              # ... and per node
_BRICK_MASS_VISIT_FLOPS = 120
# fused CG step (kernels3d FusedBrickIteration): update + block precondition + r . z (per dof: p, q, x, r, 2 block
# entries read, x, r, z written, ~10 flops), direction + product + next . q (z, p read; next, q written; 3 flops per dof
# on top of the product)
_FUSED_UPDATE_FLOPS_PER_DOF = 10
_FUSED_DIRECTION_FLOPS_PER_DOF = 3
_PROBE_FIELDS = OUTPUT_ARRAYS_3D + ELECTRIC_OUTPUT_ARRAYS_3D + ENERGY_OUTPUT_ARRAYS_3D
_TETRAHEDRA_ONLY_FIELDS = ELECTRIC_OUTPUT_ARRAYS_3D + ENERGY_OUTPUT_ARRAYS_3D   # the tensor state() has none
# frequency response: a complex factor over the real one at equal unknowns (complex multiply-adds ~4 real flops, each
# value 2 x the bytes); uncalibrated
_COMPLEX_LU_FLOPS = 4.0
_COMPLEX_SOLVE_BYTES = 2.0
_COMPLEX_BYTES = 16
_INT64_BYTES = 8
# = piezo.ELECTRODE_CONNECTIONS + (piezo.RESISTIVE_LOAD,) (piezo imports cupy)
_ELECTRODE_CONNECTIONS = ('voltage', 'floating', 'resistive_load')
_ELECTRODE_FEATURES = ('electrode', 'electrode_domain')
_BAND_TEMPLATE = 'solid_bloch_band_diagram_3d'
_SQUEEZE_FILM = 'squeeze_film_damping'
# squeeze film estimates (uncalibrated): P2 footprint dofs ~ 4 x n_nodes^(2/3) (one face of a compact mesh: nodes +
# edges), stored entries per P2 triangle row, host SuperLU seconds of the P2 pressure operator over the 5-point curve
_FILM_DOFS_PER_FACE_NODE = 4.0
_FILM_ENTRIES_PER_DOF = 15
_FILM_LU_TIME_SCALE = 3.0
_FILM_PYTHON_CALLS = 40         # assembly, conditions and the solve per film and resonance
_FILM_KNUDSEN_LIMIT = 880.0     # Veijola fit range 0 <= Kn <= 880

_PIEZOELECTRIC_PARAM = Param(
    'piezoelectric', 'bool', False, label='Piezoelectric',
    help="On (tetrahedron mesh, MeshSpec kind 'tetrahedra', only): the full 3D piezoelectric solid. A potential phi "
         '(the Lagrange elements of the discretization setting) lives on every window element, void ones included as '
         'dielectric, and couples to the displacement through the 18 piezo_e* material entries (solid elements) with the '
         'clamped permittivity eps0 eps_dc (the six eps_dc_* entries, every window element, unset = 1): T = c S - e^T E, '
         'D = e S + eps E, E = -grad phi, div D = 0. Electrode / Electrode (domain) features fix phi (connection voltage; '
         'at least one, the ground, is required) or make an isolated equipotential carrying a net charge [C] (floating); '
         'every other boundary is insulating (D . n = 0). The coupled symmetric indefinite operator is factorised once '
         "per assembly by the 'linear' sparse_direct solver (host SuperLU, cuDSS LDL^T with back_end cudss on a GPU, or "
         'an opt-in MUMPS / PARDISO host LDL^T), with the potential scaled to balance the blocks. With every e = 0 it '
         'is a 3D anisotropic electrostatics solve (electrode capacitance with fringing in the air around the solid). Outputs electric_potential, '
         'electric_field_x / _y / _z, electric_displacement_x / _y / _z and the stress with its -e^T E part. Studies '
         'stationary / time_dependent (static solves) and eigenfrequency: the modes of the coupled pencil (massless '
         "potential) nearest target_frequency by the 'eigen' solver, voltage electrodes held at 0 V (short circuit: "
         'resonance frequencies f_r) and floating ones at zero charge (open circuit: antiresonance frequencies f_a); '
         'their voltage / charge values are ignored there. Run the model once with the drive electrode voltage and '
         'once floating: k_eff^2 = (f_a^2 - f_r^2) / f_a^2. An electrode with connection resistive_load damps the modes '
         '(complex frequencies: kappa_e = 2 Im omega of that load). Each mode shows its potential (the voltage map) '
         'and E / D '
         'with the shape. frequency_response: the driven coupled pencil with eps (1 - j dielectric_loss_tangent), '
         'electrode voltages / charges as phase-0 amplitudes, the Admittance feature Y = j omega Q / V in S. '
         "A {'model': 'crystal', 'plane': 'full'} entry sets c_IJ, piezo_e* and eps_dc*. A "
         'rectangular-cell mesh refuses it.',
    tip='Turns on the piezoelectric effect: the electric potential is solved together with the deformation, and '
        'electrodes apply voltages or measure charges. Tetrahedron meshes only.\n\nNeeds at least one Electrode with a '
        'fixed voltage (the ground). In an eigenfrequency study fixed-voltage electrodes are shorted and floating ones '
        'open.')

_ELECTRODE_EDGE_SIZE_PARAM = Param(
    'electrode_edge_size', 'float', 0.0, label='Electrode edge size', unit='m', min=0.0,
    visible_if=(('piezoelectric', (True,)),),
    help="Piezoelectric on a gmsh tetrahedron mesh (mesher 'gmsh'): the element size along the edges of every "
         'domain object an enabled Electrode / Electrode (domain) feature selects (box edges, cylinder rims), growing '
         'by the mesh max_ratio away from them; 0 = off. The electric field is singular at an electrode edge, so the '
         'capacitance C0 and k_eff^2 converge slowly without it; typical a fifth to a tenth of the film thickness. '
         'Outer-face electrodes, spheres and shapes gmsh does not mesh conformally get none; the builtin mesher '
         'ignores it.',
    tip='Finer mesh along the edges of the electrode objects, where the electric field is concentrated.\n\nOnly for '
        'the gmsh mesher. 0 means no extra refinement.')

_TARGET_FREQUENCY_PARAM = Param(
    'target_frequency', 'float', 0.0, label='Eigenfrequency target', unit='Hz', min=0.0,
    help="Eigenfrequency study on a tetrahedron mesh (MeshSpec kind 'tetrahedra') only: the vibration modes nearest "
         'this frequency are searched (shift-invert about (2 pi target_frequency)^2, moved 1e-4 above it so a target '
         'on a resonance still factors); 0 = the lowest modes. A free body (free-free: no supports, or supports that '
         'leave a rigid motion free) and a free Bloch cell (periodic axes with every bloch_k 0) need a target > 0: '
         'their rigid-body modes (f below 1e-4 x the target, six for an unsupported body) are dropped, and a target '
         'below the first elastic mode is fine. Static solves of a free body stay refused (a frequency response at f '
         '> 0 is regular). A '
         'tensor mesh finds the lowest modes only (matrix-free shift-invert at 0): a nonzero value is a validation '
         'error there. Ignored by the other studies.',
    tip="Used only by an 'Eigenfrequency' study on a tetrahedron mesh: the vibration modes closest to this frequency "
        'are found.\n\n0 finds the lowest modes. A body without supports (free-free) needs a value above 0: its '
        'rigid-body motions at 0 Hz are then left out.')


def _pml_params():
    """pml_<face> per outer face, pml_cells and pml_reflection (elastic PML shell: tetrahedra, eigenfrequency)."""
    faces = tuple(Param(
        f'pml_{face}', 'bool', False, label=f'PML on {face}',
        help=(f'Tetrahedron mesh, eigenfrequency study, piezoelectric off only: an elastic perfectly matched layer '
              f'beyond {face}. The mesh grows a shell of pml_cells element layers (x the mesh dx) there, its material '
              'continued from the window edge, with the complex coordinate stretch s = 1 - j beta (d / w)^2 (d the '
              'depth, w the thickness), beta set by pml_reflection at the target frequency for the fastest wave '
              '(sqrt(c_II / rho)); its outer side is clamped. Waves leaving the window through that face are absorbed, '
              'so the modes are complex: frequency_imag > 0 and the radiation Q = f / (2 frequency_imag) in the '
              'Resonance table (mode shapes shown as their real part and |u|); modes with more than half of their '
              '|u|^2 in the shell are PML modes and dropped. The pencil stays linear in omega^2 (beta is frequency '
              'independent), so no iteration on the target; absorption weakens for modes far below it. Supports and '
              f'loads must stay off {face} (it is the inner side of the shell). Needs target_frequency > 0.'),
        tip=(f'Absorbs vibrations leaving through {face}, as if the solid continued forever there.\n\nGives each '
             'resonance a radiation quality factor. Only for an eigenfrequency study on a tetrahedron mesh.'))
        for face in FACES_3D)
    cells = Param('pml_cells', 'int', DEFAULT_PML_CELLS, label='PML element layers', min=1, max=100,
                  help='Thickness of the elastic PML shell in element layers of the mesh dx (h_max) on every pml_<face> '
                       'face; the profile needs about 6-10 layers, and the shell should span a fair part of the '
                       'longest wavelength (c / f at the target) for a low reflection.',
                  tip='How many layers of elements the absorbing layer has.\n\n8 is typical.')
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
    return tuple(face for face in FACES_3D if settings_values.get(f'pml_{face}'))


def _periodic_params():
    """periodic_x / _y / _z and bloch_kx / _ky / _kz (Bloch / Floquet unit cells: tetrahedra, eigenfrequency)."""
    periodic = tuple(Param(
        f'periodic_{axis}', 'bool', False, label=f'Periodic {axis}',
        help=(f'Tetrahedron mesh, eigenfrequency study only: {axis}_lo and {axis}_hi are one Bloch-periodic pair, u '
              f'and phi on {axis}_hi equal those on {axis}_lo times exp(-j bloch_k{axis} L) (L = the mesh extent along '
              f'{axis}, exp(+j omega t)). The two faces must be meshed alike node for node (the builtin mesher does; '
              'mesher gmsh writes periodic surfaces for it); supports and electrodes must stay off them. The modes '
              'come from the reduced pencil P^H K P, P^H M P (complex Hermitian; real when every k L is a multiple '
              'of pi) and are complex: shown as their real part (phase 0) with u_abs = |u|, electrode tables not '
              'available. With every periodic bloch_k 0 a free cell keeps its rigid translations (and the rotation '
              'about a single periodic axis): set target_frequency > 0, which drops them, or hold the cell; a '
              'piezoelectric cell then also needs a ground electrode, while k != 0 needs none.'),
        tip=(f'Makes the {axis} faces periodic: the model is one cell of an infinite periodic structure, such as a '
             'phononic crystal.\n\nOnly for an eigenfrequency study on a tetrahedron mesh.')) for axis in _AXES)
    bloch = tuple(Param(
        f'bloch_k{axis}', 'float', 0.0, label=f'Bloch wave vector {axis}', unit='rad/m',
        visible_if=((f'periodic_{axis}', (True,)),),
        help=(f'Bloch wave vector along {axis} in rad/m, used only when periodic_{axis} is on: the phase step exp(-j '
              f'k{axis} L) per period L; k{axis} and k{axis} + 2 pi / L are the same state; 0 = plain periodic. Swept '
              'through a parameter, one eigenfrequency study per k gives the band diagram (template '
              f'{_BAND_TEMPLATE}). Typical: pi / L at the zone edge.'),
        tip=('Phase step of the vibration from one period to the next along this axis, in radians per metre.\n\n0 '
             'means no phase step. Swept, it gives the bands of a phononic crystal.')) for axis in _AXES)
    return periodic + bloch


# ---- feature texts ----
_SUPPORT_RESULT_TEXT = (
    "Result 'table' of the latest solve with one row [feature, F_x, F_y, F_z] (units '', N, N, N) and the extras F_x, "
    'F_y and F_z: the force the support exerts on the solid (sum of K u - f over its held dofs, so components it does '
    'not hold read 0); a dof held by two supports counts in both (the later value holds it). After a frequency_response '
    'study on a tetrahedron mesh: a table of the complex reaction per solved frequency instead, columns frequency, '
    'F_x_re, F_x_im, F_y_re, F_y_im, F_z_re, F_z_im (Hz, N; the sum of the complex A x - f).')
_FACE_ONLY_TEXT = ('Outer faces, z_lo / z_hi included; on a tetrahedron mesh also domain objects (their interface '
                   'faces, outer boundary faces of the object included; a tensor mesh refuses objects: use Fixed '
                   '(domain) there). Holds every node of the faces, edges and corners included.')
_OUTER_FACES_TEXT = ('Outer faces only, z_lo / z_hi included (a support on an object boundary: Fixed (domain) on the '
                     'object). Holds every node of the face, edges and corners included.')


# ---- feature apply / result callbacks (worker thread, GPU) ----
def _support(solver, feature_spec, masks):
    """SupportRegion of the masks, registered with a tetrahedra solver (complex reactions per frequency)."""
    from .solver_common import SupportRegion   # lazy: solver_common imports cupy
    return register_support(solver, SupportRegion(feature_spec.name, union_masks(masks)))


def _hold_selection(ctx, solver, selection, **components):
    """Held dof masks of the selected outer faces and (tetrahedra) the interface faces of the selected objects."""
    masks = [solver.hold_face(face, **components) for face in selection.outer_faces]
    if selection.objects:
        masks.append(solver.hold_faces(ctx.boundary_faces(selection.objects), **components))
    return masks


def _apply_fixed(ctx, solver, feature_spec, values, selection):
    """u = 0 on the selected outer faces (tetrahedra: and object interface faces)."""
    return _support(solver, feature_spec, _hold_selection(ctx, solver, selection, u_x=0.0, u_y=0.0, u_z=0.0))


def _apply_roller(ctx, solver, feature_spec, values, selection):
    """Normal displacement 0 on the selected outer faces (u_x on x faces, u_y on y faces, u_z on z faces)."""
    return _support(solver, feature_spec,
                    [solver.hold_face(face, **{f'u_{face[0]}': 0.0}) for face in selection.outer_faces])


def _apply_displacement(ctx, solver, feature_spec, values, selection):
    """Prescribed u_x / u_y / u_z (None = free) on the selected outer faces (tetrahedra: and object interface faces);
    the held values follow a reused sweep build."""
    held = union_masks(_hold_selection(ctx, solver, selection, u_x=values['u_x'], u_y=values['u_y'],
                                       u_z=values['u_z']))
    follow_held_values(ctx, solver, feature_spec, values, _DISPLACEMENTS, held)
    return _support(solver, feature_spec, [held])


def _apply_fixed_domain(ctx, solver, feature_spec, values, selection):
    """u = 0 on every corner node of the selected cells (tetrahedra: every dof of the selected elements)."""
    return _support(solver, feature_spec, [solver.hold_cells(ctx.domain_mask(selection.entities))])


def _apply_prescribed_displacement_domain(ctx, solver, feature_spec, values, selection):
    """Prescribed u_x / u_y / u_z (None = free) on every corner node of the selected cells (tetrahedra: every dof of the
    selected elements); the held values follow a reused sweep build."""
    held = solver.hold_cells(ctx.domain_mask(selection.entities), u_x=values['u_x'], u_y=values['u_y'],
                             u_z=values['u_z'])
    follow_held_values(ctx, solver, feature_spec, values, _DISPLACEMENTS, held)
    return _support(solver, feature_spec, [held])


def _apply_boundary_load(ctx, solver, feature_spec, values, selection):
    """Traction on the selected outer faces and (tetrahedra) object interface faces; follows a reused sweep build (a
    change is added on the same faces arrays: the FEM solver merges it into their entry)."""
    object_faces = ctx.boundary_faces(selection.objects) if selection.objects else None   # tetrahedra only

    def add_traction(t_x, t_y, t_z):
        for face in selection.outer_faces:
            solver.add_face_traction(face, t_x, t_y, t_z)
        if object_faces is not None:
            solver.add_surface_traction(object_faces, t_x, t_y, t_z)
    add_traction(*(values[name] for name in _TRACTIONS))
    follow_load(ctx, solver, feature_spec, values, _TRACTIONS, add_traction)
    return None


def _apply_probe(ctx, solver, feature_spec, values, selection):
    if ctx.domain.grid_kind == _TETRAHEDRA:   # the nodal display field evaluated at the point
        if len(selection.objects) != 1:
            raise ValueError(f"solid probe '{feature_spec.name}' needs exactly one point object, "
                             f'got {len(selection.objects)}')
        return solver.add_probe_at(ctx.point_position(selection.objects[0]), values['component'])
    cells = point_cells(ctx, feature_spec, selection)
    if len(cells) != 1:
        raise ValueError(f"solid probe '{feature_spec.name}' needs exactly one point object, got {len(cells)}")
    ix, iy, iz = cells[0]
    return solver.add_probe(ix, iy, iz, values['component'])


def _probe_result(session, solver, probe, feature_spec, options):
    from ...core.monitors import probe_result   # lazy: core.monitors imports cupy
    return probe_result(probe, unit=SolidRenderer3D.field_infos[probe.field_component].unit)


def _apply_frequency_probe(ctx, solver, feature_spec, values, selection):
    """The complex displacement record at the point object (tetrahedra; the grids keep tensor meshes out)."""
    if len(selection.objects) != 1:
        raise ValueError(f"frequency probe '{feature_spec.name}' needs exactly one point object, "
                         f'got {len(selection.objects)}')
    return solver.add_frequency_probe_at(ctx.point_position(selection.objects[0]), feature_spec.name)


def _support_result(session, solver, region, feature_spec, options):
    return support_result(solver, region, _AXES, 'N')


def _apply_electrode(ctx, solver, feature_spec, values, selection):
    """Electrode on the selected outer faces and object interface faces, one equipotential (None when piezoelectric is
    off: validation warns)."""
    if not solver.piezoelectric:
        return None
    faces = [solver.system.face_faces(face) for face in selection.outer_faces]
    if selection.objects:
        faces.append(ctx.boundary_faces(selection.objects))
    return solver.add_surface_electrode(feature_spec.name, np.concatenate(faces) if faces else np.zeros(0, np.int64),
                                        values['connection'], values['voltage'], values['charge'],
                                        values['load_resistance'])


def _apply_electrode_domain(ctx, solver, feature_spec, values, selection):
    """Electrode on every node of the selected elements (None when piezoelectric is off)."""
    if not solver.piezoelectric:
        return None
    return solver.add_cell_electrode(feature_spec.name, ctx.domain_mask(selection.entities), values['connection'],
                                     values['voltage'], values['charge'], values['load_resistance'])


def _apply_squeeze_film(ctx, solver, feature_spec, values, selection):
    """squeeze_film.SqueezeFilmDamping on the window faces of the selected outer face and the interface faces of the
    selected objects, registered with the tetrahedra solver (evaluated per resonance after each eigen solve)."""
    from .squeeze_film import SqueezeFilmDamping
    side = values['gap_side']
    if side == 'auto':
        if len(selection.outer_faces) != 1 or selection.objects:
            raise ValueError(f"squeeze film '{feature_spec.name}': gap_side auto needs exactly one outer face selected; "
                             'set gap_side for domain objects')
        side = selection.outer_faces[0]
    faces = [solver.system.face_faces(face) for face in selection.outer_faces]
    if selection.objects:
        faces.append(ctx.boundary_faces(selection.objects))
    ambient = values['ambient_pressure']
    film = SqueezeFilmDamping(feature_spec.name, np.concatenate(faces) if faces else np.zeros(0, np.int64), side,
                              values['gap'], values['viscosity'], ambient if ambient > 0.0 else None,
                              mean_free_path=values['mean_free_path'], slip=values['slip'], vents=values['vents'])
    return solver.add_squeeze_film(film)


def _squeeze_film_result(session, solver, film, feature_spec, options):
    return film.table()


def _resonance_table_result(session, solver, table, feature_spec, options):
    """modal.modal_table_result plus the GAS_TABLE_COLUMNS totals of the evaluated squeeze films (tetrahedra)."""
    result = modal_table_result(table, solver)
    totals = gas_damping_totals(getattr(solver, 'squeeze_films', ()))
    if totals is None or len(totals) != len(result['rows']):
        return result
    result['columns'] = result['columns'] + list(GAS_TABLE_COLUMNS)
    result['units'] = result['units'] + list(GAS_TABLE_UNITS)
    result['rows'] = [row + total for row, total in zip(result['rows'], totals)]
    return result


# ---- validation helpers (CPU only) ----
def _validate_displacement(vctx, physics_spec, feature_spec, values):
    if all(values.get(f'u_{axis}') is None for axis in _AXES):
        return [Issue('error', feature_path(physics_spec, feature_spec),
                      'sets none of u_x, u_y, u_z, so it holds nothing: set at least one (empty leaves a direction '
                      'free)')]
    return []


def _selected_objects(vctx, feature):
    """Domain objects of a feature's selection; [] when it does not resolve."""
    try:
        return list(vctx.objects(feature.selection))
    except (ValueError, TypeError, KeyError):
        return []


def _held_components(vctx, feature):
    """[(face or object, axis)] an enabled support feature holds on outer faces and (fixed / displacement) object
    interface faces ([('domain', axis)] x 3 for fixed_domain, per set component for prescribed_displacement_domain); []
    when its values or selection do not resolve (reported elsewhere)."""
    if feature.type == 'fixed_domain':
        return [('domain', axis) for axis in _AXES]
    if feature.type == 'prescribed_displacement_domain':
        values = vctx.values(SOLID_MECHANICS_3D.feature_type(feature.type).params, feature.params)
        return [] if values is None else [('domain', axis) for axis in _AXES if values[f'u_{axis}'] is not None]
    faces = outer_faces(vctx, feature, FACES_3D)
    if feature.type == 'roller':
        return [(face, face[0]) for face in faces]
    faces += _selected_objects(vctx, feature)
    if feature.type == 'fixed':
        return [(face, axis) for face in faces for axis in _AXES]
    values = vctx.values(SOLID_MECHANICS_3D.feature_type('displacement').params, feature.params)
    if values is None:
        return []
    return [(face, axis) for face in faces for axis in _AXES if values[f'u_{axis}'] is not None]


def _free_body_modes(vctx, settings_values, tetrahedra) -> bool:
    """A free body is allowed: a tetrahedron mesh and free_body_allowed (the rigid check counts the free motions)."""
    return bool(tetrahedra and free_body_allowed(vctx.study_type, settings_values))


def _support_issues(vctx, physics_spec, path, bloch_cell, free_body=False):
    """Error when no support holds u_x, u_y or u_z anywhere (rigid translation; bloch_cell: a Bloch cell, where
    _bloch_issues decides, or an elastic PML, whose clamped shell holds the solid; free_body (_free_body_modes): an
    info instead); warning per face component held by several features and per boundary load on a fully held face."""
    holders = support_holders(vctx, physics_spec, _held_components)
    held_directions = {axis for _, axis in holders}
    sliding = [axis for axis in _AXES if axis not in held_directions]
    issues = []
    remedy = ('' if vctx.study_type not in MODAL_STUDIES else
              '; for the vibration modes of a free body (free-free) set target_frequency > 0 on a tetrahedron mesh')
    if bloch_cell:
        pass
    elif free_body:
        if sliding and vctx.study_type == FREQUENCY_RESPONSE_STUDY:
            what = 'no support holds the solid' if not held_directions else f"no support holds u_{', u_'.join(sliding)}"
            issues.append(Issue('info', path, f'{what}: a free body, regular at every frequency > 0 (its rigid-body '
                                              'motion follows the drive as a mass, ~1 / omega^2)'))
        elif sliding:
            what = 'no support holds the solid' if not held_directions else f"no support holds u_{', u_'.join(sliding)}"
            issues.append(Issue('info', path,
                                f'{what}: a free body. Its rigid-body modes (0 Hz, below 1e-4 x target_frequency) are '
                                'dropped and the resonances are the elastic modes nearest the target (a rotation left '
                                'free by partial supports is dropped alike: the build counts every free motion)'))
    elif not held_directions:
        issues.append(Issue('error', path,
                            'a static solve needs supports: add Fixed, Roller or Displacement on outer faces or a Fixed '
                            '(domain) region (otherwise the solid can move as a rigid body and the system is '
                            f'singular){remedy}'))
    else:
        for free in sliding:
            issues.append(Issue('error', path,
                                f'no support holds u_{free}, so the solid can slide along {free}: add a Roller on a '
                                f"{free}_lo / {free}_hi face, a Fixed face or a Displacement with u_{free} set{remedy}"))
    for (face, axis), names in sorted(holders.items()):
        if face in FACES_3D and len(names) > 1:
            issues.append(Issue('warning', path,
                                f"u_{axis} on outer face {face} is held by features {', '.join(names)}; the last one "
                                f"('{names[-1]}') sets its value"))
    for feature in physics_spec.features:
        if not feature.enabled or feature.type != 'boundary_load':
            continue
        for face in outer_faces(vctx, feature, FACES_3D):
            if all((face, axis) in holders for axis in _AXES):
                issues.append(Issue('warning', path,
                                    f"boundary load '{feature.name}' acts on face {face}, whose u_x, u_y and u_z are "
                                    'all held: the load goes straight into the reaction and moves nothing'))
    return issues


def _anisotropic_materials(vctx):
    """Names of the selected spec materials (library and crystal entry merged) that set any c_IJ entry."""
    return materials_setting(vctx, STIFFNESS_TENSOR_ARRAYS_3D)


def _piezo_issues(vctx, physics_spec, settings_values, path, tetrahedra):
    """Electrodes without piezoelectric (warning); with it: errors on a rectangular-cell mesh, in the optomechanics
    study and without a voltage
    electrode (unless a periodic axis has bloch_k != 0: no constant potential then); warnings for no material with
    piezo_e* and per piezo material without youngs_modulus / c11."""
    electrodes = [feature for feature in physics_spec.features
                  if feature.enabled and feature.type in _ELECTRODE_FEATURES]
    if not settings_values.get('piezoelectric'):
        if electrodes:
            return [Issue('warning', path, f"electrode feature(s) {', '.join(feature.name for feature in electrodes)} "
                                           'are ignored: the piezoelectric setting is off')]
        return []
    if not tetrahedra:
        return [Issue('error', path, "piezoelectric needs a tetrahedron mesh (MeshSpec kind 'tetrahedra'): the "
                                     "rectangular-cell mesh's iterative solver cannot solve the indefinite coupled "
                                     'operator')]
    issues = []
    if vctx.study_type == OPTOMECHANICS_STUDY:
        issues.append(Issue('error', path, 'piezoelectric: the optomechanics study is not modelled for a piezoelectric '
                                           'solid (its modes carry the potential); turn piezoelectric off'))
    grounded = False
    for feature in electrodes:
        values = vctx.values(SOLID_MECHANICS_3D.feature_type(feature.type).params, feature.params)
        grounded = grounded or values is None or values['connection'] == 'voltage'
        if values is not None and values['connection'] == 'resistive_load':
            issues.extend(_load_issues(vctx, physics_spec, feature, values, settings_values))
    if not grounded and _trivial_bloch_phase(settings_values):
        issues.append(Issue('error', path, 'piezoelectric needs at least one Electrode or Electrode (domain) with '
                                           'connection voltage (0 V = ground): otherwise the potential has no '
                                           'reference and the coupled system is singular (a Bloch cell with some '
                                           'bloch_k != 0 needs none)'))
    issues.extend(piezo_material_issues(vctx, physics_spec, path, PIEZO_TENSOR_ARRAYS_3D,
                                        "(no stiffness, no coupling, only its permittivity): set youngs_modulus / "
                                        "poisson_ratio, the c_IJ or a {'model': 'crystal', 'plane': 'full'} entry",
                                        (ValueError, TypeError)))
    return issues


def _load_issues(vctx, physics_spec, feature, values, settings_values):
    """Errors of a resistive_load electrode: load_resistance <= 0, a static study, an eigenfrequency study at
    target_frequency 0 (the 1 / omega load term) and periodic axes."""
    path = feature_path(physics_spec, feature)
    issues = []
    resistance = plain_value(values['load_resistance'], vctx.parameters)
    if resistance is not None and not resistance > 0.0:
        issues.append(Issue('error', path, f'load_resistance = {resistance:.4g} Ohm: a resistive_load needs R > 0 (a '
                                           'short circuit is connection voltage at 0 V)'))
    if vctx.study_type not in (EIGENFREQUENCY_STUDY, FREQUENCY_RESPONSE_STUDY):
        issues.append(Issue('error', path, f'connection resistive_load is solved by the {EIGENFREQUENCY_STUDY} and '
                                           f'{FREQUENCY_RESPONSE_STUDY} studies only; in a {vctx.study_type} study the '
                                           'resistor grounds the electrode at DC: use connection voltage at 0 V'))
    if vctx.study_type == EIGENFREQUENCY_STUDY and not settings_values.get('target_frequency'):
        issues.append(Issue('error', path, 'connection resistive_load needs target_frequency > 0: the load term j / '
                                           '(omega R) is evaluated at the target, so set it near the wanted mode'))
    if _periodic_axes(settings_values):
        issues.append(Issue('error', path, 'connection resistive_load is not available in a Bloch cell (periodic axes)'))
    return issues


def _tensor_issues(vctx, physics_spec, settings_values, path):
    """Tensor mesh (trilinear bricks): CG iteration limit; errors for c_ij materials, supports / loads on object
    boundaries, electric / energy probes, geometric nonlinearity, the frequency_response / optomechanics studies and a
    nonzero target_frequency in an eigenfrequency study, which also needs a double (or mixed) 'linear' solver (the inner
    shift-invert solves); info for a cuDSS eigen factorization (no factor here)."""
    issues = list(cg_solver_issues(vctx, physics_spec, path))
    for feature in physics_spec.features:
        component = feature.params.get('component') if feature.type == 'solid_probe' else None
        if feature.enabled and component in ELECTRIC_OUTPUT_ARRAYS_3D:
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f'probe component {component} exists only on a tetrahedron mesh with piezoelectric on'))
        elif feature.enabled and component in ENERGY_OUTPUT_ARRAYS_3D:
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f'probe component {component} exists only on a tetrahedron mesh'))
    if settings_values.get('geometric_nonlinearity'):
        issues.append(Issue('error', path, "geometric_nonlinearity needs a tetrahedron mesh (MeshSpec kind 'tetrahedra'): "
                                           'the rectangular-cell solver is linear (small displacements)'))
    issues.extend(Issue('error', ('material', name),
                        f"material '{name}' sets anisotropic c_ij entries (or a crystal entry): solid_mechanics_3d reads "
                        "them only on a tetrahedron mesh (MeshSpec kind 'tetrahedra'); this rectangular-cell mesh "
                        'takes isotropic youngs_modulus / poisson_ratio')
                  for name in _anisotropic_materials(vctx))
    for feature in physics_spec.features:
        if feature.enabled and feature.type in _INTERFACE_FEATURES and _selected_objects(vctx, feature):
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f'{feature.type} on object boundaries needs a tetrahedra mesh; on this rectangular-cell '
                                'mesh select outer faces (a support inside the domain: Fixed (domain) on the object)'))
    if vctx.study_type == FREQUENCY_RESPONSE_STUDY:
        issues.append(Issue('error', path, f"the {FREQUENCY_RESPONSE_STUDY} study needs a tetrahedron mesh (MeshSpec kind "
                                           "'tetrahedra'): this rectangular-cell mesh solves by matrix-free CG, which "
                                           'cannot factor the complex K (1 + j eta) - omega^2 M'))
    if vctx.study_type == OPTOMECHANICS_STUDY:
        issues.append(Issue('error', path, f"the {OPTOMECHANICS_STUDY} study needs a tetrahedron mesh (MeshSpec kind "
                                           "'tetrahedra'): the overlap with em_fdfd_bor optics samples the "
                                           'finite-element modes; this rectangular-cell mesh has none'))
    if vctx.study_type != EIGENFREQUENCY_STUDY:
        return issues
    if validated_linear_precision(vctx, physics_spec) != 'float64':
        issues.append(Issue('error', path,
                            "the eigenfrequency study runs one 'linear' PCG solve per Lanczos step and needs it in "
                            "double precision: set the linear solver's precision to double or mixed (or use "
                            'Automatic)'))
    if settings_values.get('target_frequency'):
        issues.append(Issue('error', path,
                            'target_frequency is set: on a rectangular-cell mesh the eigenfrequency study finds the '
                            'lowest modes only (matrix-free shift-invert at 0); set it to 0 or use a tetrahedra mesh'))
    factorization = eigen_factorization(vctx.solvers(physics_spec))
    if factorization in CUDSS_FACTORIZATIONS:
        issues.append(Issue('info', path,
                            f"eigen factorization '{factorization}' is ignored on a rectangular-cell mesh: the modes "
                            "come from matrix-free shift-invert with 'linear' PCG solves, no factor"))
    return issues


def host_factor_items(name, superlu_label, superlu_bytes, n_unknowns, entries, symmetric=True,
                      complex_values=False, fill=None) -> tuple:
    """Host items of a factor booked by its SuperLU heuristic: ((superlu_label, bytes),) for any name but a MUMPS /
    PARDISO back end / factorization (host_direct.HOST_DIRECT_BACK_ENDS; eigen names through eigen_host_direct first),
    which gets the bytes x host_direct.superlu_bytes_scale with the library's factor kind in place of 'SuperLU L + U'
    in the label, plus its own copies beside the library factor (host_direct.solver_host_bytes of n_unknowns and the
    stored entries: equilibrated operator, row scale; mixed: GMRES vectors), as fem_system_estimate books them
    (uncalibrated; also used by definition). fill: a counted cudss_factor.FactorFill (factor_fill on a built mesh)
    replaces the SuperLU heuristic bytes for MUMPS / PARDISO (nested-dissection orderings, as the count)."""
    if name not in HOST_DIRECT_BACK_ENDS:
        return ((superlu_label, int(superlu_bytes)),)
    library = factor_label(name)
    kind = f"{library} {'LDL^T' if symmetric else 'LU'}"
    label = superlu_label.replace('SuperLU L + U', kind).replace('SuperLU', kind)
    counted_entries = getattr(fill, 'entries', None)
    if counted_entries is not None:
        superlu_bytes = host_counted_lu_bytes(counted_entries, n_unknowns)   # complex128 L + U of the counted triangle
        if not complex_values:
            superlu_bytes = superlu_bytes * FLOAT64_CSR_ENTRY_BYTES / COMPLEX128_CSR_ENTRY_BYTES
        label = label.replace('heuristic', 'counted fill')
    return ((f"{label} ({'' if counted_entries is not None else 'SuperLU heuristic '}scaled, uncalibrated)",
             int(superlu_bytes * superlu_bytes_scale(name, symmetric=symmetric, complex_values=complex_values))),
            (f'{library} solver copies (equilibrated operator, row scale; mixed: GMRES vectors)',
             solver_host_bytes(n_unknowns, entries, complex_values=complex_values, symmetric=symmetric,
                               mixed=host_direct_precision(name) == 'mixed')))


def eigen_host_direct(solvers, unknowns):
    """The MUMPS / PARDISO factorization the 'eigen' runtime factors a pencil of `unknowns` with (S5c,
    ArpackShiftInvertSolver.factorization_for: a *_mixed one below MIXED_EIGEN_MIN_UNKNOWNS factors in double), else
    None (also used by definition)."""
    from ...solvers.arpack_shift_invert import MIXED_EIGEN_MIN_UNKNOWNS
    name = host_direct_back_end(solvers, role='eigen')
    if name is not None and host_direct_precision(name) == 'mixed' and int(unknowns) < MIXED_EIGEN_MIN_UNKNOWNS:
        return double_of(name)
    return name


def _tetrahedra_issues(vctx, physics_spec, settings_values, path):
    """Tetrahedron mesh (FEM): the host SuperLU size rule of the factorised unknowns (displacement, + potential when
    piezoelectric: the 'linear' factor of a static solve or of the complex frequency-response pencil, the eigen factor
    with factorization superlu; a MUMPS / PARDISO factor keeps these limits, uncalibrated, named by its library; a cuDSS
    factor guards its own device memory)."""
    mesh = vctx.mesh
    if mesh is None or getattr(mesh, 'kind', '') != _TETRAHEDRA:
        return []
    unknowns = _tetrahedra_unknowns(settings_values, mesh, vctx.spec)
    piezoelectric = bool(settings_values.get('piezoelectric'))
    what = 'displacement + potential unknowns' if piezoelectric else 'displacement unknowns'
    if vctx.study_type in MODAL_STUDIES:
        if eigen_factorization(vctx.solvers(physics_spec)) in CUDSS_FACTORIZATIONS:
            return []
        library = factor_label(host_direct_back_end(vctx.solvers(physics_spec), role='eigen'))
        where = 'the coupled pencil A - sigma M (eigen)' if piezoelectric else 'K - sigma M (eigen)'
    else:
        resolved = (vctx.solvers(physics_spec) or {}).get('linear')
        if resolved is not None and resolved.values.get('back_end') in CUDSS_BACK_ENDS:
            return []   # a cuDSS LDL^T / Cholesky of the 'linear' role guards its own device memory
        library = factor_label(host_direct_back_end(vctx.solvers(physics_spec)))
        if vctx.study_type == FREQUENCY_RESPONSE_STUDY:
            where = 'the complex pencil per frequency (linear)'
        else:
            where = 'the coupled operator (linear)' if piezoelectric else 'K (linear)'
    if unknowns > SPLU_3D_LIMIT_UNKNOWNS:
        remedy = ("eigen factorization 'cudss' on a GPU, " if vctx.study_type in MODAL_STUDIES
                  else "linear solver back_end 'cudss' on a GPU, ")
        return [Issue('error', path, f'~{unknowns:,} {what} exceed the {SPLU_3D_LIMIT_UNKNOWNS:,} limit of the host '
                                     f'{library} factor of {where} (uncalibrated for 3D elasticity): use {remedy}a '
                                     'coarser mesh or discretization linear')]
    if unknowns > SPLU_3D_WARN_UNKNOWNS:
        return [Issue('warning', path, f'~{unknowns:,} {what} exceed ~{SPLU_3D_WARN_UNKNOWNS:,}, the practical size of '
                                       f'the host {library} factor of {where}: factor memory and time grow faster '
                                       'than the unknowns')]
    return []


def _periodic_axes(settings_values) -> tuple:
    """Axes ('x', 'y', 'z') whose periodic_<axis> setting is on."""
    return tuple(axis for axis in _AXES if settings_values.get(f'periodic_{axis}'))


def _bloch_wavenumber(settings_values, axis) -> float:
    value = settings_values.get(f'bloch_k{axis}', 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _trivial_bloch_phase(settings_values) -> bool:
    """Every periodic axis has bloch_k 0 (also without periodic axes): rigid translations and a constant potential
    stay admissible."""
    return all(_bloch_wavenumber(settings_values, axis) == 0.0 for axis in _periodic_axes(settings_values))


def _periodic_separator_scales(settings_values) -> tuple:
    """(factor time, fill) multipliers of a host / cuDSS factor of a cell with p periodic axes over the open cell
    (_PERIODIC_SEPARATOR_WIDTH on the first p dissection levels); (1, 1) without periodic axes."""
    count = len(_periodic_axes(settings_values))

    def multiplier(level_ratio, power):
        doubled, total = (1.0 - level_ratio ** count) / (1.0 - level_ratio), 1.0 / (1.0 - level_ratio)
        return (_PERIODIC_SEPARATOR_WIDTH ** power * doubled + total - doubled) / total
    return multiplier(0.5, 3), multiplier(2.0 ** (-1.0 / 3.0), 2)


def _bloch_issues(vctx, physics_spec, settings_values, path, tetrahedra):
    """Periodic axes (Bloch cells): warning per bloch_k on a non-periodic axis (ignored); errors on a rectangular-cell
    mesh, outside the eigenfrequency study, per support / electrode on a periodic face and for a cell free to slide
    with every periodic bloch_k 0 at target_frequency 0 (singular K - 0 M)."""
    periodic = _periodic_axes(settings_values)
    issues = [Issue('warning', path, f'bloch_k{axis} = {_bloch_wavenumber(settings_values, axis):.4g} rad/m is '
                                     f'ignored: periodic_{axis} is off')
              for axis in _AXES if axis not in periodic and _bloch_wavenumber(settings_values, axis) != 0.0]
    if not periodic:
        return issues
    names = ', '.join(periodic)
    if not tetrahedra:
        issues.append(Issue('error', path, f"periodic axes ({names}) need a tetrahedron mesh (MeshSpec kind "
                                           "'tetrahedra'): the rectangular-cell mesh has no Bloch cells"))
        return issues
    if vctx.study_type != EIGENFREQUENCY_STUDY:
        issues.append(Issue('error', path, f'periodic axes ({names}) are solved by the {EIGENFREQUENCY_STUDY} study '
                                           f'only (Bloch bands per k); the {vctx.study_type} study of a periodic cell '
                                           'is not available: turn periodic_* off'))
    for feature in physics_spec.features:
        if not (feature.enabled and (feature.type in SUPPORT_FEATURES or feature.type in _ELECTRODE_FEATURES)):
            continue
        faces = [face for face in outer_faces(vctx, feature, FACES_3D) if face[0] in periodic]
        if faces:
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f"{feature.type} on the periodic face(s) {', '.join(faces)}: a hold or electrode there "
                                'would contradict its periodic image; select non-periodic faces (or objects inside '
                                'the cell)'))
    if _trivial_bloch_phase(settings_values) and not settings_values.get('target_frequency'):
        held = {axis for _, axis in support_holders(vctx, physics_spec, _held_components)}
        sliding = [axis for axis in _AXES if axis not in held]
        if sliding:
            issues.append(Issue('error', path, f"with every periodic bloch_k 0 the cell can slide along "
                                               f"{', '.join(sliding)} (a rigid mode at 0 Hz): set target_frequency > 0 "
                                               '(the rigid modes are dropped), a bloch_k != 0 or supports'))
    return issues


def _pml_issues(vctx, physics_spec, settings_values, path, tetrahedra):
    """Elastic PML faces: errors on a rectangular-cell mesh, outside the eigenfrequency study, at target_frequency 0,
    with piezoelectric or periodic axes, and per support / load / electrode on a PML face."""
    faces = _pml_faces(settings_values)
    if not faces:
        return []
    names = ', '.join(faces)
    if not tetrahedra:
        return [Issue('error', path, f"PML faces ({names}) need a tetrahedron mesh (MeshSpec kind 'tetrahedra'): the "
                                     'rectangular-cell mesh has no elastic PML')]
    issues = []
    if vctx.study_type != EIGENFREQUENCY_STUDY:
        issues.append(Issue('error', path, f'an elastic PML ({names}) is solved by the {EIGENFREQUENCY_STUDY} study only; '
                                           f'the {vctx.study_type} study is not available with it: turn pml_* off'))
    elif not settings_values.get('target_frequency'):
        issues.append(Issue('error', path, 'an elastic PML needs target_frequency > 0: its strength is set for the '
                                           'waves at the target'))
    if settings_values.get('piezoelectric'):
        issues.append(Issue('error', path, 'an elastic PML is not available with piezoelectric on (the coupled '
                                           'potential has no stretched form yet): turn one of them off'))
    if _periodic_axes(settings_values):
        issues.append(Issue('error', path, 'an elastic PML is not available in a Bloch cell (periodic axes)'))
    for feature in physics_spec.features:
        if not (feature.enabled and (feature.type in SUPPORT_FEATURES or feature.type in _ELECTRODE_FEATURES
                                     or feature.type == 'boundary_load')):
            continue
        held = [face for face in outer_faces(vctx, feature, FACES_3D) if face in faces]
        if held:
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f"{feature.type} on the PML face(s) {', '.join(held)}: that face is the inner side of "
                                'the absorbing shell, a hold or load there would reflect the waves; select other faces'))
    return issues


def _mirror_issues(vctx, physics_spec):
    """Warning per Roller feature (a possible half-model symmetry plane) and selected material whose c_IJ / piezo_e* /
    eps_dc_* entries are not symmetric under the reflection normal to its faces (crystal_tensors.mirror_violation)."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    from ...materials.crystal_tensors import (FULL_PERMITTIVITY_PROPERTIES, FULL_PIEZO_PROPERTIES,
                                              FULL_STIFFNESS_PROPERTIES, mirror_violation)
    rollers = [feature for feature in physics_spec.features if feature.enabled and feature.type == 'roller']
    if not rollers:
        return []
    tensor_names = (*FULL_STIFFNESS_PROPERTIES, *FULL_PIEZO_PROPERTIES, *FULL_PERMITTIVITY_PROPERTIES)
    materials = []
    for material in vctx.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, TypeError):   # reported at the material
            continue
        values = {name: plain_value(properties.get(name), vctx.parameters) for name in tensor_names}
        if material.selection and any(value is not None for value in values.values()):
            materials.append((material.name, values))
    issues = []
    for feature in rollers:
        for axis in sorted({_AXES.index(face[0]) for face in feature.selection if face in FACES_3D}):
            for name, values in materials:
                relative, entry = mirror_violation(values, axis)
                if entry:
                    issues.append(Issue('warning', feature_path(physics_spec, feature), (
                        f"if roller '{feature.name}' stands for a mirror plane (half model), note that material "
                        f"'{name}' is not symmetric under {_AXES[axis]} -> -{_AXES[axis]} ({entry} changes sign, "
                        f'{relative:.3g} of its tensor\'s largest entry): the half model then differs from the full '
                        'one (a crystal cut such as X-cut LN generally has no such mirror); mesh the full width')))
    return issues


def _nonlinear_issues(vctx, physics_spec, settings_values, path, static_refused):
    """Geometric nonlinearity on tetrahedra: error with piezoelectric; infos where it has no nonlinear static solve
    (static_refused: a Bloch cell, an elastic PML, a free body) and in the frequency_response study (linear about the
    state of the initial stress)."""
    if not settings_values.get('geometric_nonlinearity'):
        return []
    issues = []
    if settings_values.get('piezoelectric'):
        issues.append(Issue('error', path, 'geometric_nonlinearity is not available with piezoelectric on (the coupled '
                                           'operator has no total-Lagrangian form): turn one of them off'))
    issues.extend(nonlinear_state_infos(vctx, path, static_refused))
    return issues


def _electrode_edge_issues(vctx, physics_spec, settings_values, path):
    """Info when electrode_edge_size is set but nothing uses it: piezoelectric off or the builtin mesher."""
    size = settings_values.get('electrode_edge_size') or 0.0
    if not size > 0.0:
        return []
    mesher = getattr(getattr(vctx.mesh, 'sizing', None), 'mesher', None)
    if settings_values.get('piezoelectric') and mesher == 'gmsh':
        return []
    return [Issue('info', path, f'electrode_edge_size {size:.4g} m is unused: it refines a gmsh tetrahedron mesh '
                                'along the edges of piezoelectric electrode objects only')]


def _squeeze_film_issues(vctx, physics_spec, settings_values, tetrahedra):
    """squeeze_film_damping features: warning outside the eigenfrequency / optomechanics studies; errors for several outer faces, an
    outer face with objects, a gap_side contradicting the face (or auto with objects), Bloch cells, a PML face, a closed
    incompressible film and a sealed axis normal to the film; warning beyond the Veijola Kn range (tetrahedra only)."""
    issues = []
    for feature in physics_spec.features:
        if not (feature.enabled and feature.type == _SQUEEZE_FILM):
            continue
        path = feature_path(physics_spec, feature)
        if vctx.study_type not in MODAL_STUDIES:
            issues.append(Issue('warning', path, f'squeeze_film_damping is evaluated per resonance by the '
                                                 f"{' / '.join(MODAL_STUDIES)} studies only; this {vctx.study_type} "
                                                 'study ignores it'))
        if not tetrahedra:
            continue   # the feature's grids refuse the tensor mesh
        faces, objects = outer_faces(vctx, feature, FACES_3D), _selected_objects(vctx, feature)
        values = vctx.values(SQUEEZE_FILM_FEATURE.params, feature.params)
        side = None if values is None else values['gap_side']
        if len(faces) > 1:
            issues.append(Issue('error', path, f"selects the outer faces {', '.join(faces)}: a squeeze film lies on one "
                                               'flat face (one gap); add one feature per face'))
        elif faces and objects:
            issues.append(Issue('error', path, f"selects the outer face {faces[0]} and domain objects: a squeeze film "
                                               'lies on one flat face (one gap); add one feature per surface'))
        elif faces and side not in (None, 'auto', faces[0]):
            issues.append(Issue('error', path, f'gap_side {side} contradicts the selected outer face {faces[0]} (the '
                                               'gap lies outside that face): set gap_side auto or to the face'))
        elif objects and side == 'auto':
            issues.append(Issue('error', path, 'gap_side auto needs an outer face; for domain objects set the side of '
                                               'the object the gap lies on (e.g. z_lo: a plate over a substrate)'))
        film_side = faces[0] if len(faces) == 1 and side in (None, 'auto') else side
        if _periodic_axes(settings_values):
            issues.append(Issue('error', path, 'squeeze_film_damping is not available in a Bloch cell (periodic axes): '
                                               'its footprint would need periodic vents'))
        if film_side in _pml_faces(settings_values):
            issues.append(Issue('error', path, f'the film face {film_side} carries an elastic PML (the inner side of '
                                               'the absorbing shell): select another face'))
        if values is None:
            continue
        ambient = plain_value(values['ambient_pressure'], vctx.parameters)
        if values['vents'] == 'none' and ambient is not None and not ambient > 0.0:
            issues.append(Issue('error', path, 'vents none with ambient_pressure 0 is a closed incompressible film: the '
                                               'gas cannot move, the pressure solve is singular; vent the rim or set '
                                               'the ambient pressure'))
        if film_side not in (None, 'auto') and values['vents'] == f'{film_side[0]}_sealed':
            issues.append(Issue('error', path, f"vents {values['vents']} names the film's normal axis "
                                               f'{film_side[0]}: seal an in-plane axis'))
        gap = plain_value(values['gap'], vctx.parameters)
        free_path = plain_value(values['mean_free_path'], vctx.parameters)
        if (values['slip'] == 'veijola' and gap and free_path is not None
                and free_path / gap > _FILM_KNUDSEN_LIMIT):
            issues.append(Issue('warning', path, f'Knudsen number {free_path / gap:.4g} lies beyond the Veijola slip '
                                                 f'fit range (Kn <= {_FILM_KNUDSEN_LIMIT:g})'))
    return issues


def _validate_solid_3d(vctx, physics_spec, settings_values):
    """Physics-level rules (in the study the solid takes: a background physics of StudySpec.solved_physics sees
    'stationary'): supports, Bloch cells, materials, the mesh kind's solver rules (_tensor_issues / _tetrahedra_issues);
    eigenfrequency / frequency_response: densities; admittance features, loss entries, prestress."""
    vctx = background_context(vctx, physics_spec)
    path = ('physics', physics_spec.name)
    tetrahedra = on_tetrahedra(vctx, physics_spec)
    issues = _validate_solid_3d_rules(vctx, physics_spec, settings_values, path, tetrahedra)
    static_refused = ''   # why the stress-stiffened modes take the initial stress alone
    if _periodic_axes(settings_values):
        static_refused = 'a Bloch cell'
    elif _pml_faces(settings_values):
        static_refused = 'an elastic PML'
    elif _free_body_modes(vctx, settings_values, tetrahedra) and not support_holders(vctx, physics_spec, _held_components):
        static_refused = 'a free body'
    issues.extend(prestress_issues(vctx, physics_spec, settings_values, path, tetrahedra,
                                   static_refused=static_refused))
    if tetrahedra:
        issues.extend(_nonlinear_issues(vctx, physics_spec, settings_values, path, static_refused))
    if not tetrahedra:
        issues.extend(Issue('error', path, f"stored field '{physics_spec.name}.{name}' exists on a tetrahedron mesh "
                                           'only (the rectangular-cell solid has no frequency response)')
                      for name in HARMONIC_IMAG_FIELDS_3D
                      if f'{physics_spec.name}.{name}' in (vctx.spec.study.stored_fields or []))
    return issues


def _validate_solid_3d_rules(vctx, physics_spec, settings_values, path, tetrahedra):
    """The rules of _validate_solid_3d before the prestress ones."""
    issues = _support_issues(vctx, physics_spec, path,
                             bool(_periodic_axes(settings_values) or _pml_faces(settings_values)),
                             free_body=_free_body_modes(vctx, settings_values, tetrahedra))
    issues.extend(_bloch_issues(vctx, physics_spec, settings_values, path, tetrahedra))
    issues.extend(_pml_issues(vctx, physics_spec, settings_values, path, tetrahedra))
    issues.extend(_piezo_issues(vctx, physics_spec, settings_values, path, tetrahedra))
    issues.extend(_mirror_issues(vctx, physics_spec))
    issues.extend(_electrode_edge_issues(vctx, physics_spec, settings_values, path))
    issues.extend(_squeeze_film_issues(vctx, physics_spec, settings_values, tetrahedra))
    if tetrahedra:
        issues.extend(_tetrahedra_issues(vctx, physics_spec, settings_values, path))
        element_name = f"{settings_values.get('discretization', 'quadratic')} tetrahedron"
    else:
        issues.extend(_tensor_issues(vctx, physics_spec, settings_values, path))
        element_name = 'trilinear'
    issues.extend(material_issues(vctx, physics_spec, path, SOLID_3D_TYPE_NAME, element_name, tensor=tetrahedra))
    issues.extend(modal_study_issues(vctx, physics_spec, path, MODAL_STUDIES, THERMAL_STRAIN_COUPLING,
                                     tensor=tetrahedra,
                                     stress_stiffening=bool(settings_values.get('stress_stiffening')
                                                            or (tetrahedra
                                                                and settings_values.get('geometric_nonlinearity')))))
    issues.extend(frequency_response_issues(vctx, physics_spec, settings_values, path, THERMAL_STRAIN_COUPLING))
    issues.extend(eigen_search_issues(vctx, path, bool(settings_values.get('piezoelectric')),
                                      tensor_mesh_3d=not tetrahedra))
    return issues


# ---- physics factory and memory estimate ----
def _create_solid_3d(ctx, physics_spec, values):
    """SolidMechanics3DPhysics on ctx.domain with the build's linear (pcg) solver (solid_linear_solver) and, in an
    eigenfrequency study, its 'eigen' (arpack_shift_invert) runtime; a tetrahedron mesh: SolidMechanicsFEMPhysics3D
    with the 'linear' sparse_direct runtime, the discretization order, target_frequency, piezoelectric, the periodic
    axes with their Bloch wavenumbers, stress_stiffening, the free-body allowance of the study it takes and the
    geometric nonlinearity settings."""
    if ctx.domain.grid_kind == _TETRAHEDRA:
        from .solver_fem3d import SolidMechanicsFEMPhysics3D   # lazy: the solver imports cupy
        runtimes = ctx.solvers(physics_spec.name)
        return SolidMechanicsFEMPhysics3D(ctx.domain, order=element_order(values), linear_solver=runtimes.get('linear'),
                                          eigen_solver=runtimes.get('eigen'),
                                          target_frequency=values['target_frequency'],
                                          piezoelectric=values['piezoelectric'],
                                          periodic=tuple(values[f'periodic_{axis}'] for axis in _AXES),
                                          bloch_k=tuple(values[f'bloch_k{axis}'] for axis in _AXES),
                                          pml_faces=_pml_faces(values), pml_reflection=values['pml_reflection'],
                                          stress_stiffening=values['stress_stiffening'],
                                          free_body=free_body_allowed(solid_study(ctx.spec, physics_spec.name),
                                                                       values),
                                          geometric_nonlinearity=values['geometric_nonlinearity'],
                                          load_steps=values['load_steps'],
                                          nonlinear_tolerance=values['nonlinear_tolerance'],
                                          nonlinear_max_iterations=values['nonlinear_max_iterations'],
                                          tangent_update=values['tangent_update'])
    from .solver3d import SolidMechanics3DPhysics   # lazy: the solver imports cupy
    return SolidMechanics3DPhysics(ctx.domain, linear_solver=solid_linear_solver(ctx, physics_spec),
                                   eigen_solver=ctx.solvers(physics_spec.name).get('eigen'))


def _domain_requirements(settings_values):
    """The periodic axes (a gmsh tetrahedron mesh writes periodic surfaces for them) and the PML padding (pml_cells
    layers on the pml_<face> faces: the mesh's shell)."""
    faces = _pml_faces(settings_values)
    padding = None
    if faces:
        cells = int(settings_values.get('pml_cells') or DEFAULT_PML_CELLS)
        padding = PMLPadding({face: cells if face in faces else 0 for face in FACES_3D})
    return DomainRequirements(boundary_config=padding, periodic_axes=_periodic_axes(settings_values))


def _mesh_edge_sizes(settings_values, physics_spec) -> tuple:
    """(entity, electrode_edge_size) of every object entity of the enabled electrode features (piezoelectric only)."""
    size = settings_values.get('electrode_edge_size') or 0.0
    if not settings_values.get('piezoelectric') or not size > 0.0:
        return ()
    return tuple((entity, float(size)) for feature in physics_spec.features
                 if feature.enabled and feature.type in _ELECTRODE_FEATURES
                 for entity in feature.selection if isinstance(entity, str) and entity not in FACES_3D)


def _complex_pencil(settings_values) -> bool:
    """A periodic axis with bloch_k != 0 (an upper figure: k L = pi is real) or an elastic PML: the eigen pencil and its
    factor are complex."""
    return not _trivial_bloch_phase(settings_values) or bool(_pml_faces(settings_values))


def _lossy_pencil(settings_values, context) -> bool:
    """A mechanical_loss_factor without piezoelectric: K (1 + j eta), a complex eigen pencil (PiezoSystem.modal_pencil
    takes no losses: they enter its frequency response only)."""
    return not settings_values.get('piezoelectric') and bool(context.material_sets(LOSS_ARRAYS[0]))


def _resistive_load(settings_values, physics_spec) -> bool:
    """Piezoelectric with an enabled electrode feature on connection resistive_load: ModalPencil's 'loaded' pencil
    K + (j / omega_t) B - sigma M (three real terms, a complex factor)."""
    return bool(settings_values.get('piezoelectric')) and physics_spec is not None and any(
        feature.enabled and feature.type in _ELECTRODE_FEATURES
        and (feature.params or {}).get('connection') == 'resistive_load' for feature in physics_spec.features)



def _tetrahedra_scalar_dofs(settings_values, mesh) -> int:
    """Scalar dofs of the full-mesh map: nodes (+ edges for quadratic) of a TetrahedronMeshEstimate."""
    return int(mesh.n_nodes) + (int(mesh.n_edges) if element_order(settings_values) == 2 else 0)


def _solid_fraction(spec, mesh) -> float:
    """Share of the estimated elements painted by a material with stiffness (youngs_modulus or c11 set): where the
    displacement dofs live (TetrahedronMeshEstimate.material_elements; 1 without them)."""
    from ...materials import effective_material_data   # materials imports physics.em
    painted = getattr(mesh, 'material_elements', None)
    if not painted or not int(mesh.n_elements):
        return 1.0
    stiff = 0.0
    for material in spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, TypeError):   # reported by validate
            continue
        if any(properties.get(name) is not None and not (isinstance(properties.get(name), (int, float))
                                                          and properties.get(name) == 0)
               for name in ('youngs_modulus', 'c11')):
            stiff += painted.get(material.name, 0.0)
    return min(stiff / int(mesh.n_elements), 1.0)


def _tetrahedra_vector_dofs(settings_values, mesh, spec) -> int:
    """Displacement unknowns: 3 x the scalar dofs of the solid share of the elements (_solid_fraction)."""
    return 3 * int(round(_tetrahedra_scalar_dofs(settings_values, mesh) * _solid_fraction(spec, mesh)))


def _tetrahedra_unknowns(settings_values, mesh, spec) -> int:
    """Unknowns of the factorised operator: the displacement ones, + the potential (every scalar dof of the window: void
    elements are dielectric) when piezoelectric."""
    vector_dofs = _tetrahedra_vector_dofs(settings_values, mesh, spec)
    return vector_dofs + (_tetrahedra_scalar_dofs(settings_values, mesh) if settings_values.get('piezoelectric') else 0)


def _tetrahedra_entries(settings_values, mesh, spec) -> int:
    """Stored entries of the factorised CSR: per unknown, components x the scalar Lagrange entries per dof of the mesh
    (fem.diffusion.lagrange_entries_per_dof; 3 for the vector stiffness, 4 for the coupled (u, phi) operator)."""
    components = 4 if settings_values.get('piezoelectric') else 3
    return int(round(components * lagrange_entries_per_dof(mesh, element_order(settings_values))
                     * _tetrahedra_unknowns(settings_values, mesh, spec)))


def _window_shape(mesh) -> tuple:
    """Window element counts (x, y, z) of a 3D MeshEstimate."""
    sides = mesh.npml_sides
    return tuple(int(count) - sides.get(f'{axis}_lo', 0) - sides.get(f'{axis}_hi', 0)
                 for axis, count in zip(_AXES, (mesh.nx, mesh.ny, mesh.nz)))


def _window_counts(mesh):
    """(window elements, window nodes) of a 3D MeshEstimate."""
    elements, nodes = 1, 1
    for count in _window_shape(mesh):
        elements *= count
        nodes *= count + 1
    return elements, nodes


def _multigrid_ops(element_shape, item_bytes, brick_product) -> tuple:
    """(one V-cycle, one hierarchy build) of multigrid3d.BrickMultigrid as op tuples: per non-coarsest level the
    smoother / correction products and Chebyshev passes and six separable transfer passes, the coarsest dense product;
    the build's Lame means, blocks, power iterations, the 81 coarsest probes and the host eigh."""
    shapes = level_element_shapes(element_shape)
    products = 2 * CHEBYSHEV_DEGREE   # pre: degree, correction: 1, post: degree - 1
    chebyshev_passes = 2 * CHEBYSHEV_DEGREE + 2
    cycle, setup = [], []
    for index, shape in enumerate(shapes):
        level_elements = int(np.prod(shape))
        level_nodes = int(np.prod([n + 1 for n in shape]))
        level_dofs = 3 * level_nodes
        product = brick_product(item_bytes, level_elements, level_nodes)
        if index + 1 < len(shapes):
            cycle += [(products, product),
                      (chebyshev_passes, te.Elementwise(18 * level_nodes * item_bytes, 9 * level_nodes * item_bytes,
                                                        flops=30 * level_nodes, flop_bytes=item_bytes)),
                      (6, te.elementwise(level_dofs, 3, 1, item_bytes))]
            setup += [(POWER_ITERATIONS, product), (POWER_ITERATIONS, te.elementwise(level_nodes, 9, 3, item_bytes)),
                      (2 * POWER_ITERATIONS, te.Reduction(level_dofs, item_bytes, read_back=True))]
        setup += [te.elementwise(level_elements, 2, 6, FLOAT64_BYTES), te.elementwise(level_nodes, 2, 6, item_bytes)]
    coarsest = 3 * int(np.prod([n + 1 for n in shapes[-1]]))
    cycle.append(te.Elementwise(coarsest * coarsest * item_bytes, coarsest * item_bytes, flops=2 * coarsest ** 2,
                                flop_bytes=item_bytes))
    coarsest_shape = shapes[-1]
    setup += [(81, brick_product(item_bytes, int(np.prod(coarsest_shape)), coarsest // 3)),
              (81, te.Transfer(2 * coarsest * item_bytes)), (81, te.PythonCalls(27)),
              te.DenseFlops(float(coarsest) ** 3 * 4.0, host=True), te.Transfer(coarsest * coarsest * item_bytes)]
    return tuple(cycle), tuple(setup)


def _time_estimate(settings_values, mesh, context):
    """Matrix-free block-Jacobi PCG of the 'linear' role: per application the brick product (element Lame values,
    gathered nodal vectors) and the 3 x 3 node-block preconditioner; stationary: the static solves (coupled: per step
    of a time_dependent study); eigenfrequency: one PCG solve per Lanczos application of the 'eigen' role (its own
    timer then holds the solves, so its items join the rest of the study). A tetrahedron mesh: _tetrahedra_time_estimate.
    A background solid of StudySpec.solved_physics is priced as in a stationary study."""
    context = background_context(context, context.physics_spec)
    if mesh.kind == _TETRAHEDRA:
        return _tetrahedra_time_estimate(settings_values, mesh, context)
    precision = linear_solver_precision(context)
    value_bytes = FLOAT64_BYTES if precision == 'float64' else FLOAT32_BYTES
    elements, nodes = _window_counts(mesh)
    dofs = 3 * nodes

    def brick_product(item_bytes, level_elements=elements, level_nodes=nodes):   # one matrix-free stiffness product
        return te.Elementwise((2 * level_elements + 24 * level_nodes) * item_bytes, 3 * level_nodes * item_bytes,
                              launches=2, numba=True,
                              flops=_BRICK_ELEMENT_FLOPS * level_elements + _BRICK_NODE_FLOPS * level_nodes,
                              flop_bytes=item_bytes)

    def fused_iteration(item_bytes):   # FusedBrickIteration.step: two launches (the cpu numba twins: four passes)
        product = brick_product(item_bytes)
        update = te.Elementwise(6 * dofs * item_bytes, 3 * dofs * item_bytes, numba=True,
                                flops=_FUSED_UPDATE_FLOPS_PER_DOF * dofs, flop_bytes=item_bytes)
        direction = replace(product, bytes_read=product.bytes_read + (dofs * item_bytes + BOOL_BYTES * dofs),
                            bytes_written=2 * dofs * item_bytes, launches=1,
                            flops=product.flops + _FUSED_DIRECTION_FLOPS_PER_DOF * dofs)
        return update, direction
    # mixed: the inner CG iterates on the float32 kernel copies, each outer refinement applies the float64 product
    inner_bytes = FLOAT32_BYTES if linear_solver_mixed(context) else value_bytes
    common = dict(unknowns=float(dofs), value_bytes=value_bytes, dimensions=3, longest_axis=context.longest_axis(),
                  product_ops=(brick_product(inner_bytes),),
                  preconditioner_ops=(te.elementwise(nodes, 9, 3, inner_bytes),),
                  refinement_ops=(brick_product(value_bytes),), iteration_ops=fused_iteration(inner_bytes))
    if linear_solver_multigrid(context):   # multigrid3d V-cycle (float32 at every precision), no fused iteration
        cycle, setup = _multigrid_ops(_window_shape(mesh), FLOAT32_BYTES, brick_product)
        if inner_bytes != FLOAT32_BYTES:   # double CG: residual cast in, correction cast back per cycle
            cycle += (te.Elementwise(dofs * (inner_bytes + FLOAT32_BYTES), dofs * (inner_bytes + FLOAT32_BYTES),
                                     launches=2),)
            setup += (te.elementwise(elements, 2, 2, FLOAT32_BYTES), te.elementwise(nodes, 6, 6, FLOAT32_BYTES))
        common.update(preconditioner_ops=cycle, iteration_ops=(), setup_ops=setup,   # built in assemble, timed there
                      setup_counters=('multigrid_builds', 'multigrid_build_seconds'))
    if context.study.type == EIGENFREQUENCY_STUDY:
        eigenpairs = int(context.count('eigenpairs', 1) or 1) + EXTRA_EIGENVALUES
        resolved = context.solvers.get('eigen')
        subspace = int(resolved.values.get('subspace_size', 0)) if resolved is not None else 0
        applications = APPLICATIONS_PER_BASIS_GUESS * arnoldi_basis_size(eigenpairs, dofs, subspace)
        mass = (te.Elementwise(3 * dofs * FLOAT64_BYTES, dofs * FLOAT64_BYTES, launches=2, numba=True,
                               flops=_BRICK_VISITS_PER_ELEMENT * _BRICK_MASS_VISIT_FLOPS * elements,
                               flop_bytes=FLOAT64_BYTES),)
        eigen = context.solver_work('eigen', unknowns=float(dofs), dimensions=3, solves=1.0, factorizations=0.0,
                                    eigenpairs=eigenpairs, product_ops=mass)
        eigen = replace(eigen, items=tuple(replace(item, seconds_counter=()) for item in eigen.items))
        return te.combine((eigen, context.solver_work('linear', solves=applications, solves_kind='guess',
                                                      factorizations=1.0, **common)))
    per_step = context.time_dependent and context.coupled
    solves, kind, _ = (1.0, 'exact', None) if per_step else context.static_solves()
    return context.solver_work('linear', solves=solves, solves_kind=kind, factorizations=solves,
                               factorizations_kind=kind, per_step=per_step, warm_start=per_step, **common)


def _estimated_rigid_motions(settings_values, context) -> int:
    """Rigid motions an eigenfrequency solve asks on top of n + 4 (tetrahedra, target_frequency > 0): 6 for a solid
    without any enabled support, 3 for a trivial-phase Bloch cell without supports, else 0 (partial supports: not
    counted)."""
    if not settings_values.get('target_frequency'):
        return 0
    if any(feature.enabled and feature.type in SUPPORT_FEATURES for feature in context.physics_spec.features):
        return 0
    if _periodic_axes(settings_values):
        return 3 if _trivial_bloch_phase(settings_values) else 0
    return 0 if _pml_faces(settings_values) else 6


def _static_state_solved(settings_values, context, with_loads) -> bool:
    """stress_stiffening on tetrahedra: solve_static_state factorises K by the 'linear' role before the pencils (every
    load with_loads, else an initial stress to relax), unless a Bloch cell, an elastic PML or a free body refuses it."""
    if not settings_values.get('stress_stiffening'):
        return False
    initial = any(context.material_sets(name) for name in OPTIONAL_PRESTRESS_SCHEMA_3D)
    supported = any(feature.enabled and feature.type in SUPPORT_FEATURES for feature in context.physics_spec.features)
    refused = bool(_periodic_axes(settings_values)) or bool(_pml_faces(settings_values)) or not supported
    return (with_loads or initial) and not refused


def _tetrahedra_time_estimate(settings_values, mesh, context):
    """Tetrahedra: the host factor recipes at the vector unknowns on the 7-point 3D curves times _TET_LU_TIME_SCALE /
    _TET_SOLVE_TIME_SCALE: static = 'linear' factorisations (one per assembly; per solve when the stiffness follows a
    state field) and back substitutions; eigenfrequency = one 'eigen' shift factorisation (SuperLU, or the cuDSS 3D
    recipe fitted to curl-curl systems, uncalibrated here) and its Lanczos applications; frequency_response = one
    complex 'linear' factorisation and solve per frequency (_COMPLEX_LU_FLOPS / _COMPLEX_SOLVE_BYTES over the real
    recipe; a Bloch cell with bloch_k != 0 or an elastic PML prices its eigen factor so too). Geometric nonlinearity:
    each static solve (once for the pencil of the modal / response studies) is _newton_time. Piezoelectric: every
    recipe at the coupled unknowns (the scales of the vector stiffness, uncalibrated for it); a resistive_load
    electrode: the complex 'loaded' pencil of three terms (SolverWork.pencil_terms); losses make only a non-piezo pencil
    complex. A cuDSS eigen factor is priced on the run backend (real LDL^T unless the pencil is complex), SuperLU on the
    host. Periodic axes (Bloch cells) scale the host factor and solves by _periodic_separator_scales."""
    order = element_order(settings_values)
    unknowns = float(_tetrahedra_unknowns(settings_values, mesh, context.spec))
    periodic_factor, periodic_fill = _periodic_separator_scales(settings_values)
    scales = dict(nnz=float(_tetrahedra_entries(settings_values, mesh, context.spec)), dimensions=3,
                  factor_scale=_TET_LU_TIME_SCALE[order] * periodic_factor,
                  solve_scale=_TET_SOLVE_TIME_SCALE[order] * periodic_fill)
    complex_scales = dict(scales, factor_scale=_COMPLEX_LU_FLOPS * scales['factor_scale'],
                          solve_scale=_COMPLEX_SOLVE_BYTES * scales['solve_scale'])
    nonlinear = bool(settings_values.get('geometric_nonlinearity'))
    if nonlinear and context.study.type in (*MODAL_STUDIES, FREQUENCY_RESPONSE_STUDY):
        # the nonlinear static state of the pencils (once, in place of the linear one), then the study's linear recipe
        linear = _tetrahedra_time_estimate(dict(settings_values, geometric_nonlinearity=False,
                                                stress_stiffening=False), mesh, context)
        return te.combine((linear, _newton_time(settings_values, mesh, context, unknowns, scales, 1.0, 'exact',
                                                False)))
    if context.study.type in MODAL_STUDIES:
        eigenpairs = (int(context.count('eigenpairs', 1) or 1) + EXTRA_EIGENVALUES
                      + _estimated_rigid_motions(settings_values, context))
        loaded = _resistive_load(settings_values, context.physics_spec)
        complex_pencil = _complex_pencil(settings_values) or loaded or _lossy_pencil(settings_values, context)
        if eigen_factorization(context.solvers) in CUDSS_FACTORIZATIONS:   # the device recipe reads value_bytes
            factor = dict(scales, value_bytes=_COMPLEX_BYTES if complex_pencil else FLOAT64_BYTES)
        else:   # host SuperLU: the complex cost is in complex_scales
            factor = dict(complex_scales if complex_pencil else scales, backend='cpu')
        shift_work, _ = eigen_search_counts(context)   # one shift-invert per shift of the eigenfrequency search
        parts = [context.solver_work('eigen', unknowns=unknowns, eigenpairs=eigenpairs, plan_reused=True,
                                     pencil_terms=3 if loaded else 2, **shift_work, **factor)]
        if _static_state_solved(settings_values, context, context.study.type == EIGENFREQUENCY_STUDY):
            # stress stiffening: solve_static_state factorises K_ff (planned from the host CSR) and solves once
            # before the pencils
            parts.append(context.solver_work('linear', unknowns=unknowns, solves=1.0, solves_kind='exact',
                                             factorizations=1.0, factorizations_kind='exact', **scales))
        films = _squeeze_film_time(mesh, context)
        return te.combine(parts + ([] if films is None else [films]))
    if context.study.type == FREQUENCY_RESPONSE_STUDY:   # the study's harmonic_factorizations (sweep_method)
        from ...model.definition_helpers import frequency_sweep_counts, reduced_sweep_time_items
        sweep = frequency_sweep_counts(context)
        work = context.solver_work('linear', unknowns=unknowns, solves=sweep['solves'], solves_kind=sweep['solves_kind'],
                                   factorizations=sweep['factorizations'],
                                   factorizations_kind=sweep['factorizations_kind'], **complex_scales)
        if sweep['method'] == 'direct':
            return work
        missing = ('modal basis shift-invert (private eigen solver)',) if sweep['eigenpairs'] else ()
        return te.combine((work, te.TimeEstimate(reduced_sweep_time_items(sweep, unknowns), not_estimated=missing)))
    per_step = context.time_dependent and context.coupled
    solves, kind, _ = (1.0, 'exact', None) if per_step else context.static_solves()
    if nonlinear:   # every static solve is a Newton loop: a tangent factor per iteration
        return _newton_time(settings_values, mesh, context, unknowns, scales, solves, kind, per_step)
    refactor = stiffness_follows_state(context)
    # system_fem3d's DiffusionDirectSolver factorises the host K_ff: planned from it, later ones upload its values
    # (piezoelectric: PiezoSystem's own host route)
    return context.solver_work('linear', unknowns=unknowns, solves=solves, **scales,
                               solves_kind=kind, factorizations=solves if refactor else 1.0,
                               factorizations_kind=kind if refactor else 'exact', per_step=per_step,
                               factorizations_per_step=per_step and refactor)


def _squeeze_film_unknowns(mesh) -> int:
    """P2 pressure dofs of one squeeze-film footprint: ~ _FILM_DOFS_PER_FACE_NODE x n_nodes^(2/3) (a face of a compact
    mesh; uncalibrated, the selection is not resolved on the estimate)."""
    return int(_FILM_DOFS_PER_FACE_NODE * max(int(mesh.n_nodes), 1) ** (2.0 / 3.0)) + 1


def _squeeze_film_time(mesh, context):
    """Eigenfrequency study with enabled squeeze_film_damping features: per film and resonance one host assembly and
    complex SuperLU factor + solve of the P2 footprint (2D curves x _FILM_LU_TIME_SCALE); None without films."""
    films = sum(1 for feature in context.physics_spec.features if feature.enabled and feature.type == _SQUEEZE_FILM)
    if not films:
        return None
    unknowns = float(_squeeze_film_unknowns(mesh))
    solves = float(films) * eigen_search_counts(context)[1]   # per film and kept resonance
    item = te.TimeItem('squeeze film pressure solves (per film and resonance)', 'solve', solves, 'bounded',
                       (te.HostLU(unknowns, dimensions=2, complex=True, scale=_FILM_LU_TIME_SCALE),
                        te.HostSolve(unknowns, dimensions=2, complex=True, scale=_FILM_LU_TIME_SCALE),
                        te.HostStream(_FILM_ENTRIES_PER_DOF * unknowns * (_COMPLEX_BYTES + INT32_BYTES)),
                        te.PythonCalls(_FILM_PYTHON_CALLS)))
    return te.TimeEstimate((item,))


def _squeeze_film_memory(values, n_selected_cells, mesh):
    """Host items of one squeeze film (tetrahedra): the footprint mesh and maps, the complex P2 pressure operator with
    its reduced copy and vectors, and its complex SuperLU factor (heuristic, uncalibrated)."""
    unknowns = _squeeze_film_unknowns(mesh)
    entries = _FILM_ENTRIES_PER_DOF * unknowns
    return MemoryEstimate.from_items(host_items=(
        ('squeeze film footprint mesh and dof maps', 12 * _INT64_BYTES * unknowns),
        ('squeeze film complex pressure operator, reduced copy and vectors',
         2 * (_COMPLEX_BYTES + INT32_BYTES) * entries + 6 * _COMPLEX_BYTES * unknowns),
        ('squeeze film complex SuperLU factor (heuristic)', 2 * host_real_lu_bytes(unknowns, dimensions=2))))


def _newton_time(settings_values, mesh, context, unknowns, scales, solves, kind, per_step):
    """Geometric nonlinearity: per static solve load_steps x _NEWTON_ITERATIONS_PER_STEP host tangent + internal-force
    assemblies and 'linear' tangent factorisations with one back substitution each (count 'guess'; with tangent_update
    when_slow the full-Newton count is an upper bound of the factorisations: more, force-only iterations instead); the
    gpu backend with a cuDSS 'linear' back end assembles on the device."""
    order = element_order(settings_values)
    local = 10 if order == 2 else 4
    iterations = float(solves) * int(settings_values.get('load_steps') or 1) * _NEWTON_ITERATIONS_PER_STEP
    count_kind = 'guess' if kind == 'exact' else kind
    coo_entries = float(int(mesh.n_elements) * (3 * local) ** 2)
    points, elements = _TANGENT_RULE_POINTS[order], float(mesh.n_elements)
    kernel_flops = element_flops(local, points) * elements
    scatter_bytes = _TANGENT_SCATTER_BYTES_PER_COO * coo_entries
    linear = (context.solvers or {}).get('linear')
    if context.backend == 'gpu' and linear is not None and linear.values.get('back_end') in CUDSS_BACK_ENDS:
        # device TangentAssembler3D (plan G5) on the element kernel: per chunk (sized by the chunk byte cap) one CUDA
        # block per element, the block scatter_add, the displacement up and the force down
        chunks = -(-int(mesh.n_elements) // device_tangent_chunk_elements(local, points, element_kernel=True))
        ops = (te.DenseFlops(kernel_flops),
               te.Elementwise(scatter_bytes, FLOAT64_BYTES * coo_entries, launches=_ELEMENT_KERNEL_LAUNCHES * chunks),
               te.Transfer(2.0 * FLOAT64_BYTES * 3 * unknowns), te.PythonCalls(_ELEMENT_KERNEL_LAUNCHES * chunks))
        label = 'tangent and internal force assembly on the GPU'
    elif numba_available():   # the numba twin of the element kernel, then the host bincount block sums
        ops = (te.DenseFlops(kernel_flops), te.HostStream(scatter_bytes), te.PythonCalls(60))
        label = 'tangent and internal force assembly'
    else:
        ops = (te.HostStream(_TANGENT_BYTES_PER_COO[order] * coo_entries), te.PythonCalls(60))
        label = 'tangent and internal force assembly'
    assembly = te.TimeItem(label, 'iteration', iterations, count_kind, ops, per_step=per_step)
    return te.combine((te.TimeEstimate((assembly,)), context.solver_work(
        'linear', unknowns=unknowns, solves=iterations, solves_kind=count_kind, factorizations=iterations,
        factorizations_kind=count_kind, per_step=per_step, factorizations_per_step=per_step, device_values=True,
        **scales)))


def _device_tangent_items(order, elements, scalar_dofs, entries, reduced_entries, factored):
    """(run items, transient items) of the device Newton tangents (plan G5, uncalibrated): the kept TangentAssembler3D
    caches, values, vectors and cuDSS index maps; one chunk's temporaries per tangent assembly."""
    local = 10 if order == 2 else 4
    points = _TANGENT_RULE_POINTS[order]
    columns = 3 * local
    gradient_bytes = FLOAT64_BYTES * elements * points * (columns + 1)
    caches = ((gradient_bytes if gradient_bytes <= TANGENT_DEVICE_GRADIENT_CACHE_BYTES else 0)
              + INT32_BYTES * elements * (columns ** 2 + 4 * local)
              + FLOAT64_BYTES * elements * (36 + 6 * points)
              + INT32_BYTES * (entries + 3 * scalar_dofs) + 2 * INT32_BYTES * reduced_entries + INT32_BYTES * factored)
    values = FLOAT64_BYTES * (entries + 3 * reduced_entries) + 4 * FLOAT64_BYTES * 3 * scalar_dofs
    # one chunk of TangentAssembler3D's device size (the byte cap; a tighter pool gives smaller chunks) on the element
    # kernel: forces, blocks and gathered displacements
    chunk = (min(elements, device_tangent_chunk_elements(local, points, element_kernel=True))
             * tangent_chunk_element_bytes(local, points, element_kernel=True))
    maps = INT32_BYTES * (2 * reduced_entries + (reduced_entries + factored) // 2) + (_INT64_BYTES + 1) * factored
    return ([('device Newton tangent assembler caches (gradients, scatter slots, dofs, C, S0, pattern)', caches),
             ('device Newton tangent values (full, reduced, equilibrated) and vectors', values),
             ('cuDSS device index maps of the tangent pattern (rows, columns, diagonal, lower take)', maps)],
            [('Newton tangent assembly', 'device Newton tangent chunk temporaries', chunk)])


def _tetrahedra_memory_estimate(settings_values, mesh, context):
    """Tetrahedra (SolidMechanicsFEMPhysics3D, uncalibrated): per-element fields, host assembly / factor / pencil items
    per study, the cuDSS 'linear' / 'eigen' pool items (kept or per solve) and factors outside the pool (body notes)."""
    # piezoelectric: the coupled operator (not K) is factored, Lanczos vectors span the coupled unknowns; a Bloch cell
    # with bloch_k != 0, an elastic PML, a loss or a resistive_load electrode: a complex pencil (x 2 host items)
    from .definition import (LINEAR_STATIC_STATE_PHASE, eigen_basis_counts, eigen_device_items,   # definition imports
                             factor_fill, harmonic_pencil_items, linear_factor_items)             # us
    order = element_order(settings_values)
    local = 10 if order == 2 else 4
    elements = int(mesh.n_elements)
    spec = context.spec
    scalar_dofs = _tetrahedra_scalar_dofs(settings_values, mesh)   # full-mesh maps: holds, loads, nodal outputs
    vector_dofs = _tetrahedra_vector_dofs(settings_values, mesh, spec)   # the solid's: K rows, the factor
    piezoelectric = bool(settings_values.get('piezoelectric'))
    scalar_entries = lagrange_entries_per_dof(mesh, order)
    entries = int(round(3 * scalar_entries * vector_dofs))   # the vector stiffness K
    csr_bytes = FLOAT64_CSR_ENTRY_BYTES * entries + INT32_BYTES * 3 * scalar_dofs
    n_fields = len(_PROBE_FIELDS) + len(INPUT_ARRAYS_3D)
    device_items = [(f'fields ({n_fields} x float32 per element)', n_fields * FLOAT32_BYTES * elements)]
    added = sum(1 for name in PIEZO_DEFAULTS_3D if not context.material_sets(name)) if piezoelectric else 0
    if added:
        device_items.append((f'piezoelectric material arrays ({added} x float32)', added * FLOAT32_BYTES * elements))
    eigenfrequency = context.spec.study.type in MODAL_STUDIES
    factored = _tetrahedra_unknowns(settings_values, mesh, spec)
    run_items = [('output copies', 2 * FLOAT32_BYTES * elements)]
    frequency_response = context.spec.study.type == FREQUENCY_RESPONSE_STUDY
    # static / Newton solves by the 'linear' runtime: every study but frequency_response, an eigenfrequency study only
    # for the static state of its pencils (geometric nonlinearity, or stress stiffening: solve_static_state)
    static_solve = not frequency_response and (
        not eigenfrequency or bool(settings_values.get('geometric_nonlinearity'))
        or _static_state_solved(settings_values, context, context.spec.study.type == EIGENFREQUENCY_STUDY))
    resolved = (context.solvers or {}).get('linear')
    linear_back_end = None if resolved is None else resolved.values.get('back_end')
    factorization = eigen_factorization(context.solvers) if eigenfrequency else None
    loaded = _resistive_load(settings_values, getattr(context, 'physics_spec', None))
    eigen_bytes = (_COMPLEX_BYTES if _complex_pencil(settings_values) or loaded or _lossy_pencil(settings_values, context)
                   else FLOAT64_BYTES)
    factored_entries = _tetrahedra_entries(settings_values, mesh, spec)
    on_gpu = context.spec.solver_defaults.backend != 'cpu'
    transient_items, factors = [], []
    # cuDSS fill of every factor of this operator: counted on the built mesh (a film of a few layers or a refined band
    # fills far less than a compact 3D mesh), else the 3D growth law
    fill = factor_fill(mesh, order, factored, 4 if piezoelectric else 3, 3)
    if static_solve:
        # system_fem3d plans the K_ff factor from the host CSR at the first solve (cuDSS: its lower triangle on the
        # device); in an eigenfrequency study solve_static_state releases that factor before the pencil's is planned
        held, passing, factors = linear_factor_items(factored, factored_entries, linear_back_end, on_gpu, fill,
                                                     kept=not eigenfrequency)
        run_items += held
        transient_items += passing
    if frequency_response and linear_back_end in CUDSS_BACK_ENDS:   # the kept HarmonicSystem: K + j L, M (+ load)
        run_items += harmonic_pencil_items(factored, factored_entries, linear_back_end,
                                           (_COMPLEX_BYTES, FLOAT64_BYTES, FLOAT64_BYTES))
        factors.append(('frequency response cuDSS factor', factored, factored_entries, _COMPLEX_BYTES, fill,
                        linear_back_end == 'cudss_mixed'))
    coupled_csr = (FLOAT64_CSR_ENTRY_BYTES * factored_entries + INT32_BYTES * factored)
    # Newton tangents on the GPU (plan G5): system_fem3d.device_tangent = gpu backend and a cuDSS 'linear' back end
    device_tangent = (bool(settings_values.get('geometric_nonlinearity')) and static_solve
                      and on_gpu and linear_back_end in CUDSS_BACK_ENDS)
    if device_tangent:
        held, passing = _device_tangent_items(order, elements, scalar_dofs, entries, factored_entries, factored)
        run_items += held
        transient_items += passing
    coo_entries = elements * (3 * local) ** 2
    factor_bytes = int(_TET_FILL_FACTOR[order] * host_real_lu_bytes(factored, dimensions=3))
    host_items = [('dof maps (P1 / P2, P0)', _INT64_BYTES * (local + 1) * elements),
                  ('element arrays, loads and holds',
                   12 * FLOAT64_BYTES * elements + 8 * FLOAT64_BYTES * 3 * scalar_dofs),
                  ('6 x 6 element stiffness (kept, its isotropic copy, a build temporary)',
                   3 * 36 * FLOAT64_BYTES * elements),
                  # fixed_pattern.kept_pattern (block 3): vector indptr / indices, scalar keys and row starts, kept
                  # per mesh (the Newton tangent shares it); its build: the scalar COO batch, then the int64 block
                  # expansion batch
                  ('stiffness block pattern kept per mesh (vector indices, scalar keys)',
                   INT32_BYTES * (entries + 3 * scalar_dofs) + _INT64_BYTES * (entries // 9 + scalar_dofs)),
                  ('stiffness pattern build (scalar COO batch or block expansion batch)',
                   max(min(elements * local ** 2, 20_000_000) * (2 * _INT64_BYTES + FLOAT32_BYTES),
                       3 * _INT64_BYTES * min(entries, EXPAND_ENTRIES))),
                  ('stiffness CSR and its free-dof copy', 2 * csr_bytes),
                  (f'nodal outputs ({_TET_NODAL_OUTPUTS} x float64) and their element point values',
                   _TET_NODAL_OUTPUTS * FLOAT64_BYTES * (scalar_dofs + local * elements)),
                  # solver_fem3d._point_values: (values (m, q, 3), Voigt strains (m, q, 6)) float64 of the solid
                  # elements at the dof points and the centroid, kept until the last point group is written; a complex
                  # solution (frequency response, a complex modal pencil) adds its imaginary part
                  ('shared output point values (displacement and strains at the dof points and centroid)',
                   (2 if frequency_response or eigen_bytes == _COMPLEX_BYTES else 1) * 9 * FLOAT64_BYTES * (local + 1)
                   * int(round(elements * _solid_fraction(spec, mesh))))]
    if piezoelectric:
        host_items.extend([
            ('piezoelectric coupled operator, G^T A G and its free restriction (float64 CSR)', 3 * coupled_csr),
            ('piezoelectric element tensors e, eps (kept, read copies) and the coupling COO batch',
             2 * 27 * FLOAT64_BYTES * elements + min(elements * 3 * local * local, 20_000_000)
             * (2 * INT32_BYTES + FLOAT64_BYTES)),
            (f'electric nodal outputs ({_TET_ELECTRIC_NODAL_OUTPUTS} x float64) and their element point values',
             _TET_ELECTRIC_NODAL_OUTPUTS * FLOAT64_BYTES * (scalar_dofs + local * elements))])
    if settings_values.get('geometric_nonlinearity'):   # the Newton tangent replaces K in the 'linear' factor
        # TangentAssembler3D: rule gradients + weights (kept up to its cache limit), the int32 scatter map of every
        # element-block entry, element vector dofs and the free-free restriction (indices, take) on the stiffness's
        # kept block pattern; building the restriction: int64 rows of every entry and the gathered renumbering
        rule_points = _TANGENT_RULE_POINTS[order]
        gradient_bytes = 0 if device_tangent else min(FLOAT64_BYTES * elements * rule_points * (3 * local + 1),
                                                      TANGENT_GRADIENT_CACHE_BYTES)   # device: kept on the GPU
        host_items.extend([('Newton tangent K_T CSR and its free-dof copy', 2 * csr_bytes),
                           ('Newton vectors (state, internal force, residual, correction, proposal, load)',
                            6 * FLOAT64_BYTES * 3 * scalar_dofs),
                           ('Newton tangent assembler: rule gradients, scatter map, free restriction',
                            gradient_bytes + INT32_BYTES * coo_entries + _INT64_BYTES * 3 * local * elements
                            + (INT32_BYTES + _INT64_BYTES) * entries),
                           ('Newton tangent free restriction build (a run item)',
                            (2 * _INT64_BYTES + 3) * entries)])
        if settings_values.get('tangent_update') == 'when_slow':   # modified Newton keeps the factor's tangent values
            host_items.append(('modified Newton: values of the held tangent', FLOAT64_BYTES * entries))
    if eigenfrequency:
        pencil = 'coupled A - sigma M' if piezoelectric else 'K - sigma M'
        complex_factor = 2 if eigen_bytes == _COMPLEX_BYTES else 1
        if settings_values.get('stress_stiffening'):
            host_items.append(('geometric stiffness K_sigma and K + K_sigma (float64 CSR)', 2 * csr_bytes))
        # mass, the restricted K and M; piezoelectric: + G^T M G, the restricted coupled A in place of K; Bloch: the
        # prolongation and the complex restricted pencil
        host_items.append(('consistent mass CSR and the restricted pencil',
                           (1 + 2 * complex_factor) * csr_bytes + (complex_factor * coupled_csr if piezoelectric else 0)))
        if _periodic_axes(settings_values):
            host_items.append(('Bloch masters, phases and prolongation', (2 * _INT64_BYTES + 2 * _COMPLEX_BYTES)
                               * factored))
        if factorization not in CUDSS_FACTORIZATIONS:
            kind = 'complex128' if complex_factor > 1 else 'float64'
            host_items.extend(host_factor_items(eigen_host_direct(context.solvers, factored),
                                                f'SuperLU factor of {pencil} ({kind}, heuristic)',
                                                complex_factor * factor_bytes, factored, factored_entries,
                                                complex_values=complex_factor > 1,
                                                symmetric=_trivial_bloch_phase(settings_values), fill=fill))
        # a PML, loss or resistive load: the complex symmetric pencil runs eigs_shifted (device Krylov-Schur on a
        # DevicePencil); a real or Hermitian (Bloch phase) one eigsh_shifted
        pml = bool(_pml_faces(settings_values))
        complex_eigs = pml or loaded or _lossy_pencil(settings_values, context)
        rigid_motions = _estimated_rigid_motions(settings_values, context)
        _, n_basis = eigen_basis_counts(context, factored, rigid_motions, pml)
        # the ShiftPencil (DevicePencil, mass product, factor), released after each solve
        eigen_transient, eigen_host_items, eigen_factors, eigen_phases, basis_on_device = eigen_device_items(
            context, factored, factored_entries, eigen_bytes, terms=3 if loaded else 2,
            term_bytes=FLOAT64_BYTES if loaded else eigen_bytes,
            # a Bloch phase: Hermitian or not vouched (the device check runs and fails), full terms
            symmetric=_trivial_bloch_phase(settings_values), checked=not _trivial_bloch_phase(settings_values),
            dimensions=fill, complex_eigs=complex_eigs, n_basis=n_basis,
            # the elastic mass couples one component per row (piezoelectric: the pencil's count, an upper bound)
            mass_entries=None if piezoelectric else factored_entries // 3)
        host_items.extend((name, complex_factor * value) for name, value in _eigen_host_items(
            context, factored, rigid_motions, pml, basis_on_device))
        host_items.extend(eigen_host_items)
        # the static factor's phase (solve_static_state releases it before the pencil's is planned)
        phases = {**{factor[0]: LINEAR_STATIC_STATE_PHASE for factor in factors}, **eigen_phases}
        return MemoryEstimate.from_items(device_items, host_items=host_items, run_items=run_items,
                                         transient_items=transient_items + list(eigen_transient),
                                         **cudss_external(factors + list(eigen_factors), phases=phases))
    host_factor = linear_back_end not in CUDSS_BACK_ENDS
    if frequency_response:
        # mass and loss CSR, the reduced K / loss / M of the free dofs, the complex sum and its factor copy
        complex_csr = (_COMPLEX_BYTES + INT32_BYTES) * _tetrahedra_entries(settings_values, mesh, spec)
        if settings_values.get('stress_stiffening'):
            host_items.append(('geometric stiffness K_sigma and K + K_sigma (float64 CSR)', 2 * csr_bytes))
        host_items.extend([('frequency response mass, loss and reduced K / loss / M (float64 CSR)',
                            2 * csr_bytes + 3 * coupled_csr),
                           ('frequency response complex operator and its factor copy', 2 * complex_csr),
                           ('frequency response summed pattern and its aligned complex / mass values', 2 * complex_csr),
                           ('frequency response solution, rhs and product vectors (complex)',
                            6 * _COMPLEX_BYTES * factored)])
        if host_factor:
            host_items.extend(host_factor_items(linear_back_end, 'complex SuperLU factor of the pencil (heuristic)',
                                                2 * factor_bytes, factored, factored_entries, complex_values=True,
                                                fill=fill))
        host_items.extend(reduced_sweep_memory_items(context, factored, factored_entries))   # adaptive / modal
    elif host_factor:
        what = 'the coupled operator' if piezoelectric else 'K'
        # piezoelectric: the coupled operator is real symmetric indefinite (LDL^T on MUMPS / PARDISO as well)
        host_items.extend(host_factor_items(linear_back_end, f'SuperLU factor of {what} (float64, heuristic)',
                                            factor_bytes, factored, factored_entries, fill=fill))
    return MemoryEstimate.from_items(device_items, host_items=host_items, run_items=run_items,
                                     transient_items=transient_items, **cudss_external(factors))


def _memory_estimate(settings_values, mesh, context):
    """MechanicsFields3D plus the window ElasticSystem3D counted from its allocations (a tetrahedron mesh:
    _tetrahedra_memory_estimate; a background solid of StudySpec.solved_physics: as stationary)."""
    context = background_context(context, context.physics_spec)
    if mesh.kind == _TETRAHEDRA:
        return _tetrahedra_memory_estimate(settings_values, mesh, context)
    precision = linear_solver_precision(context)
    mixed = linear_solver_mixed(context)   # float64 system, float32 inner CG
    value_bytes = FLOAT64_BYTES if precision == 'float64' else FLOAT32_BYTES
    elements, nodes = _window_counts(mesh)
    dofs = 3 * nodes
    n_fields = len(OUTPUT_ARRAYS_3D) + len(INPUT_ARRAYS_3D)
    fields = MemoryEstimate.from_items(((f'fields ({n_fields} x float32)',
                                         n_fields * FLOAT32_BYTES * int(mesh.n_cells)),))
    # unknown, held values, applied load, load, rhs, fixed values, fixed coupling, CG unknown; the matrix-free
    # PCGSolver workspace (residual, two directions of the fused iteration, product, preconditioned; mixed: float32 +
    # the float64 outer vectors; the fused iteration's block partial sums are negligible)
    multigrid = linear_solver_multigrid(context)   # multigrid3d: no fused iteration (one direction vector)
    vectors = (8 * value_bytes * dofs
               + workspace_bytes(dofs, 'mixed' if mixed else SOLVER_PRECISIONS[precision], jacobi=False,
                                 fused=not multigrid))
    masks = 3 * BOOL_BYTES * dofs + BOOL_BYTES * elements   # held, free, rigid-check held copy; rigid-check solid copy
    blocks = 6 * value_bytes * nodes                         # reduced inverse 3 x 3 blocks
    # lambda, mu, applied body force (3); the physics' solid mask
    element_arrays = 5 * value_bytes * elements + BOOL_BYTES * elements
    # Lame temporaries (E, nu, products) + solid / range masks; int32 labels, padded copy, node labels; dof masks
    build_extra = ((6 * value_bytes + 2 * BOOL_BYTES + 2 * INT32_BYTES) * elements + INT32_BYTES * nodes
                   + 2 * BOOL_BYTES * dofs)
    if context.spec.solver_defaults.backend == 'cpu':
        # numpy kernels3d twins of the assembly's product: 3 differences, 3 gathered sums, the result (node-sized x 3
        # components) and ~30 per-Gauss-point element temporaries (x 3 components for the stacked stresses)
        build_extra += 7 * value_bytes * dofs + 30 * value_bytes * 3 * elements
    items = [('node vectors', vectors), ('dof masks', masks), ('inverse node blocks', blocks),
             ('element arrays', element_arrays)]
    double_cycle = multigrid and not mixed and value_bytes != FLOAT32_BYTES   # float32 V-cycle of a double CG
    if mixed or double_cycle:   # float32 lambda, mu and reduced inverse blocks of the inner CG / cycle (widths small)
        items.append(('float32 operator copy (mixed / multigrid)',
                      2 * FLOAT32_BYTES * elements + 6 * FLOAT32_BYTES * nodes))
    if double_cycle:
        items.append(('float32 cycle output (double multigrid)', FLOAT32_BYTES * dofs))
    transient_items = []   # freed when their re-assembly / eigen solve ends
    if stiffness_follows_state(context):
        transient_items.append(('stiffness re-assembly', 'stiffness re-assembly (state-dependent stiffness)',
                                8 * value_bytes * elements))
    host_items = ()
    if multigrid:   # BrickMultigrid levels (float32 cycle at every precision) and its per-assembly build
        items.append(('multigrid levels (vectors, coarse operators, transfers, coarsest inverse)',
                      hierarchy_bytes(_window_shape(mesh), FLOAT32_BYTES)))
        # float64 lambda x V, mu x V, V and the solid mask of the fine level while the level means are formed
        transient_items.append(('multigrid build', 'multigrid build (float64 volume-weighted Lame values)',
                                (3 * FLOAT64_BYTES + BOOL_BYTES) * elements))
    if context.spec.study.type == EIGENFREQUENCY_STUDY:
        # density copy + masked copy, massless mask; free-dof index (int64), Lanczos vector, per-step gathers
        transient_items.append(('eigen solve', 'eigen solve vectors', (2 * FLOAT64_BYTES + BOOL_BYTES) * elements
                                + 4 * FLOAT64_BYTES * dofs))
        host_items = _eigen_host_items(context, dofs)
    if multigrid:
        host_items = tuple(host_items) + (('multigrid coarsest build (float64, transient)',
                                           coarsest_host_bytes(_window_shape(mesh))),)
    system = MemoryEstimate.from_items(items, build_extra=build_extra, host_items=host_items,
                                       transient_items=transient_items)
    return combine((fields, system))


def _eigen_host_items(context, dofs, rigid_motions=0, pml=False, basis_on_device=False):
    """Host items of ElasticSystem3D.lowest_modes / the tetrahedra ModalPencil: the Lanczos basis and eigenvectors
    (eigsh ncv; basis_on_device: the device Krylov-Schur holds the basis, the eigenvectors stay), ARPACK work vectors,
    the free-dof index copy and the kept mode shapes (float64 over every dof); rigid_motions asked on top (pml: the
    shell doubles the asked eigenvalues, ModalPencil._n_wanted)."""
    from .definition import eigen_basis_counts   # definition imports us
    n_wanted, n_basis = eigen_basis_counts(context, dofs, rigid_motions, pml)
    n_resonances = modal_kept_count(context)   # several targets / a band keep more mode shapes
    basis = ((f'eigenvectors ({n_wanted} vectors; the basis is on the device)', n_wanted * FLOAT64_BYTES * dofs)
             if basis_on_device else
             (f'Lanczos basis + eigenvectors ({n_basis} + {n_wanted} vectors)',
              (n_basis + n_wanted + 4) * FLOAT64_BYTES * dofs))
    return (basis, (f'mode shapes ({n_resonances} x float64)', n_resonances * FLOAT64_BYTES * dofs))


def _support_memory(values, n_selected_cells, mesh):
    """Held dof mask kept per support feature for its reaction; build extra: one face mask and the union."""
    if mesh.kind == _TETRAHEDRA:   # host (n_dofs, 3) masks on the full P2 map (an upper figure for P1)
        return MemoryEstimate.from_items(host_items=(('held dof mask', 3 * BOOL_BYTES * (int(mesh.n_nodes)
                                                                                     + int(mesh.n_edges))),))
    _, nodes = _window_counts(mesh)
    dof_mask = 3 * BOOL_BYTES * nodes
    return MemoryEstimate.from_items((('held dof mask', dof_mask),), build_extra=2 * dof_mask)


def _mesh_method(spec, physics_spec, settings_values, mesh_kind, parameters):
    """GUI text of the discretisation on mesh_kind (no extra problems: validation and the run cover the rest)."""
    if mesh_kind == _TETRAHEDRA:
        order = 'quadratic' if element_order(settings_values) == 2 else 'linear'
        return (f'Finite elements on tetrahedra ({order} elements), one direct factorisation per assembly. Vibration '
                'modes are searched near the eigenfrequency target. Supports, loads and electrodes may also act on the '
                'faces of drawn objects. Piezoelectric solids solve the deformation and the electric potential '
                'together, in static solves, vibration modes and frequency responses. Periodic faces make the model '
                'one cell of a periodic structure for band diagrams. Squeeze-film damping of a thin gas gap under a '
                'flat face is added to each vibration mode.', [])
    return ('Bricks on the rectangular cells, iterative solver. Vibration modes: the lowest ones only (no target). '
            'Supports and loads act on the outer faces only. No anisotropic or piezoelectric materials.', [])


# ---- electrode features (piezoelectric on, tetrahedra) ----
_ELECTRODE_PARAMS = (
    Param('connection', 'choice', 'voltage', label='Connection', choices=_ELECTRODE_CONNECTIONS,
          choice_labels=(('voltage', 'Fixed voltage'), ('floating', 'Floating (isolated)'),
                         ('resistive_load', 'Resistor to ground')),
          help='voltage: phi = voltage on every node of the electrode (0 = ground; the eigenfrequency study holds 0: '
               'short circuit, the resonance frequencies). floating: one shared, unknown potential carrying the net '
               'free charge of the charge setting (0 = an isolated electrode; the eigenfrequency study uses 0: open '
               'circuit, the antiresonance frequencies). resistive_load: one shared potential V tied to ground through '
               'load_resistance R, j omega Q = -V / R (exp(+j omega t)); eigenfrequency study (target_frequency > 0): '
               'damped complex modes, the Resonance table adds frequency_imag, q_factor and energy_decay_rate = 2 Im '
               'omega = kappa_e of a line of impedance R (the pencil is frozen at the target and each mode refined by '
               'Newton on its Rayleigh functional: exact for the mode on the target, second order for the others); '
               'frequency_response: the resistor loads the response of the driven electrode; static studies refuse it. '
               'At least one voltage electrode is required (the reference of phi).',
          tip='Fixed voltage: the electrode is held at the voltage (or shorted to ground in an eigenfrequency study).'
              '\n\nFloating: the electrode is isolated, its voltage follows from the charge on it (open circuit in an '
              'eigenfrequency study).\n\nResistor to ground: the electrode drains through a resistor, which damps the '
              'vibration modes (for example a 50 Ohm transmission line); eigenfrequency and frequency response '
              'studies only.'),
    Param('voltage', 'float', 0.0, label='Voltage', unit='V',
          help='Potential of a voltage electrode in a static (stationary or time-dependent) solve; ignored by floating '
               'electrodes and by the eigenfrequency study (0 V).',
          tip='Voltage applied to the electrode; 0 is ground.'),
    Param('charge', 'float', 0.0, label='Charge', unit='C',
          help='Net free charge on a floating electrode in a static solve (its voltage follows); ignored by voltage '
               'electrodes and by the eigenfrequency study (0 C).',
          tip='Electric charge put on an isolated electrode.'),
    Param('load_resistance', 'float', 50.0, label='Load resistance', unit='Ohm', min=0.0,
          visible_if=(('connection', ('resistive_load',)),),
          help='Resistance from a resistive_load electrode to ground, > 0 (typical: 50 Ohm, a matched transmission '
               'line); ignored by the other connections.',
          tip='Resistance between the electrode and ground, for example 50 Ohm for a transmission line.'),
)
_ELECTRODE_RESULT_TEXT = (
    "Result 'table' of the latest static solve (eigenfrequency study: of the shown mode, per its 1 m peak displacement) "
    "with one row [feature, V, Q] (units '', V, C) and the extras V and Q: the electrode potential and the net free "
    'charge on it (Q = -sum of the potential rows of A x over its nodes, the discrete Gauss law; a voltage electrode: '
    'the charge its source delivered, so Q / V of a single driven electrode against ground is its capacitance, '
    'fringing included; a mode: the short-circuit charge of a voltage electrode, the open-circuit voltage of a '
    'floating one).')
_ELECTRODE_TEXT = (
    "piezoelectric only, tetrahedron mesh (ignored with a validation warning when piezoelectric is off): an "
    'equipotential electrode, connection voltage (held phi), floating (one shared phi with a net charge) or '
    'resistive_load (one shared phi through a resistor to ground: damped modes, a loaded response). Two '
    'electrodes must not share a node (adjacent faces share their edge nodes: a build error; select both in one feature '
    'instead). Its nodes keep their mechanical supports and loads (a thin metal film without stiffness).')


# ---- squeeze_film_damping feature (tetrahedra, eigenfrequency study; physics/mechanics/squeeze_film.py runtime on
# fem/squeeze_film.py) ----
_GAS_COLUMNS_TEXT = (
    'per mode for the mode shape of the eigenfrequency study (peak |u| = 1 m): gas_damping c = -Re F [kg/s] and '
    'gas_stiffness k = omega Im F [N/m] of the modal gas force F = int p conj(w) dA per unit modal velocity (both scale '
    'with the square of the shape normalisation), gamma_gas = c / m_eff [1/s] (the energy decay rate, the linewidth in '
    'rad/s) and q_gas = omega m_eff / c, with m_eff = int rho |u|^2 dV of the same shape (independent of it)')
_SQUEEZE_FILM_PARAMS = (
    *SQUEEZE_FILM_GAS_PARAMS,   # gap, viscosity, ambient_pressure, slip, mean_free_path (model.definition_helpers)
    Param('gap_side', 'choice', 'auto', label='Gap side', choices=GAP_SIDES,
          help='Side of the solid the gap lies on: auto = the selected outer face (the gap lies outside the window '
               'there); for domain objects the direction of the face outward normal (z_lo: the bottom face of a plate '
               'over a substrate, z_hi: under a lid): of the selected objects\' interface faces those flat on that '
               'axis plane, beside a solid element and facing that side, on the outermost such plane, form the '
               'footprint.',
          tip='Where the gap is, seen from the solid: auto uses the selected domain face; for a drawn object choose '
              'the side of the object facing the wall (for example the bottom).'),
    Param('vents', 'choice', 'all', label='Vented rim', choices=VENT_CHOICES,
          choice_labels=(('all', 'Whole rim vented'), ('window_sealed', 'Sealed where it touches the domain boundary'),
                         ('x_sealed', 'Sealed on the x boundary planes'),
                         ('y_sealed', 'Sealed on the y boundary planes'),
                         ('z_sealed', 'Sealed on the z boundary planes'), ('none', 'Closed (no vent)')),
          help='Which rim edges of the footprint vent to the ambient (p = 0; the others are sealed, no gas flux): all = '
               'the whole rim; window_sealed = rim edges lying on the domain boundary planes sealed (mirror planes of '
               'a half / quarter model), the rest vented; x_sealed / y_sealed / z_sealed = only those on the x / y / z '
               'boundary planes sealed (an in-plane axis; e.g. a strip vented on two sides); none = a closed film '
               '(needs ambient_pressure > 0: a trapped gas spring).',
          tip='Where the gas can escape from the gap. Edges on a symmetry plane of a half or quarter model must be '
              'sealed.'),
)
SQUEEZE_FILM_FEATURE = FeatureType(
    _SQUEEZE_FILM, 'Squeeze-film damping', 'monitor', 'boundary', params=_SQUEEZE_FILM_PARAMS, result_kind='table',
    summary='Estimates how much a thin gas layer between a flat face of the solid and a fixed wall damps each vibration '
            'mode, as in MEMS resonators.',
    description=('Gas damping of a thin film (gap H) between one flat face of the solid and a fixed parallel wall, '
                 'evaluated per resonance after the eigenfrequency solve (tetrahedron mesh; it does not change the '
                 'modes). Selection: one outer face, or domain objects with gap_side (their interface faces flat on '
                 'that side form the footprint; a curved or stepped selection is a build error). On the footprint '
                 '(a P2 triangle mesh of the tetrahedron faces) the linearised compressible Reynolds equation div(H^3 / '
                 '(12 mu_eff) grad p) - j omega (H / p_a) p = j omega w is solved per mode, w = -u . n_out the '
                 'gap-opening normal displacement of the mode, p = 0 on the vented rim edges (vents), no flux on '
                 "sealed ones. Result 'table', one row per mode: resonance_number, frequency [Hz], "
                 f'effective_mass [kg], {_GAS_COLUMNS_TEXT}; extras area [m^2], gap [m], surface_elements, '
                 'vent_edges. The Resonance table appends gas_damping, gas_stiffness, gamma_gas and q_gas summed '
                 'over every enabled film (select each face once). Complex modes (PML, loss, resistive loads) use Re '
                 'omega and m_eff = Re(u^H M u); Bloch cells refuse it. Only the eigenfrequency study evaluates it.'),
    apply=_apply_squeeze_film, result=_squeeze_film_result, memory_estimate=_squeeze_film_memory,
    grids=(_TETRAHEDRA,))


def _resonance_table_feature():
    """The shared resonance_table with the squeeze-film columns of the tetrahedra solver."""
    feature = resonance_table_feature('the lowest first on a tensor mesh (target_frequency reads 0); on a tetrahedron '
                                      'mesh nearest the target (the target_frequency setting) first')
    return replace(feature, grids=_GRIDS, result=_resonance_table_result,
                   description=(f'{feature.description} With {_SQUEEZE_FILM} features (tetrahedron mesh) the table '
                                f"appends {', '.join(GAS_TABLE_COLUMNS)}: the totals of every enabled film "
                                f'({_GAS_COLUMNS_TEXT}; c, k and gamma add over the films).'))


# ---- type ----
_FEATURES = (
    FeatureType('fixed', 'Fixed', 'boundary_condition', 'boundary', result_kind='table',
                summary='Clamps the selected domain faces: they cannot move in any direction.',
                description=f'Clamped support: u_x = u_y = u_z = 0. {_FACE_ONLY_TEXT} {_SUPPORT_RESULT_TEXT}',
                apply=_apply_fixed, result=_support_result, memory_estimate=_support_memory, grids=_GRIDS),
    FeatureType('roller', 'Roller', 'boundary_condition', 'boundary', allow_interior=False, result_kind='table',
                summary='Lets the selected domain faces slide along themselves but not move away from or into the '
                        'domain. Also the symmetry condition of a mirror-symmetric model.',
                description='Roller (symmetry) support: the normal displacement is 0 (u_x on x_lo / x_hi, u_y on '
                            'y_lo / y_hi, u_z on z_lo / z_hi), the tangential ones free (zero shear traction). '
                            'Rollers on one x, one y and one z face hold a solid completely. '
                            f'{_OUTER_FACES_TEXT} {_SUPPORT_RESULT_TEXT}',
                apply=_apply_roller, result=_support_result, memory_estimate=_support_memory, grids=_GRIDS),
    FeatureType('displacement', 'Displacement', 'boundary_condition', 'boundary',
                params=following(tuple(displacement_param(f'u_{axis}', axis) for axis in _AXES)), result_kind='table',
                summary='Moves the selected domain faces by a given displacement, for example to stretch a sample.',
                description='Prescribed displacement: u_x, u_y and / or u_z [m] on every node of the selected faces; an '
                            'empty component stays free (validation error when all are empty). A later support on '
                            f'the same face component sets its value (warning). {_FACE_ONLY_TEXT} '
                            f'{_SUPPORT_RESULT_TEXT}',
                apply=_apply_displacement, result=_support_result, validate=_validate_displacement,
                memory_estimate=_support_memory, grids=_GRIDS),
    FeatureType('fixed_domain', 'Fixed (domain)', 'domain_condition', 'domain', result_kind='table',
                summary='Holds the selected domains rigidly in place, like a substrate or a rigid clamp.',
                description='Holds u_x = u_y = u_z = 0 on every corner node of the selected cells (a tetrahedron mesh: '
                            'every node of the selected elements; a rigid region whose boundary is clamped to the '
                            f"solid around it, whatever the cells' own stiffness). {_SUPPORT_RESULT_TEXT}",
                apply=_apply_fixed_domain, result=_support_result, memory_estimate=_support_memory, grids=_GRIDS),
    FeatureType('prescribed_displacement_domain', 'Prescribed displacement (domain)', 'domain_condition', 'domain',
                params=following(tuple(replace(displacement_param(f'u_{axis}', axis),
                                               help=f'Prescribed u_{axis} [m] on every node of the selected domains; '
                                                    f'empty (null) leaves u_{axis} free.',
                                               tip=f'Displacement along {axis} the selected domains are held at; empty '
                                                   'leaves this direction free.')
                                       for axis in _AXES)),
                result_kind='table',
                summary='Holds chosen displacement components of the selected domains, for example the in-plane '
                        'motion of a film while it may still deflect.',
                description='Prescribed displacement per component on every corner node of the selected cells (a '
                            'tetrahedron mesh: every node of the selected elements): u_x, u_y and / or u_z [m], an '
                            'empty component stays free (validation error when all are empty), e.g. u_x = u_y = 0 '
                            'with u_z free for a no-slip film. A 3D solid held so through its whole thickness cannot '
                            'rotate its cross-sections: its bending turns into transverse shear and, with geometric '
                            'nonlinearity, the membrane stretching is partly relieved by thickness change; a no-slip '
                            'mid-surface (a shell model) is not the same constraint. A later support on the same node '
                            f'component sets its value. {_SUPPORT_RESULT_TEXT}',
                apply=_apply_prescribed_displacement_domain, result=_support_result, validate=_validate_displacement,
                memory_estimate=_support_memory, grids=_GRIDS),
    FeatureType('boundary_load', 'Boundary load', 'boundary_condition', 'boundary',
                params=following((Param('t_x', 'float', 0.0, label='Traction x', unit='Pa',
                              help='Force per area along +x on the selected outer faces (a pressure p pushing into the '
                                   'x_hi face is t_x = -p).',
                              tip='Force per area along x on the face; on the x_hi face a negative value pushes into '
                                  'the domain.'),
                        Param('t_y', 'float', 0.0, label='Traction y', unit='Pa',
                              help='Force per area along +y on the selected outer faces (a pressure p on y_hi is t_y = '
                                   '-p).',
                              tip='Force per area along y on the face.'),
                        Param('t_z', 'float', -1e6, label='Traction z', unit='Pa',
                              help='Force per area along +z on the selected outer faces (a pressure p on z_hi is t_z = '
                                   '-p). Typical 1e5-1e8.',
                              tip='Force per area along z on the face; on the top face a negative value presses down.'
                                  '\n\nTypical 0.1 to 100 MPa.'))),
                summary='Applies a force per area on the selected domain faces.',
                description='Surface traction (t_x, t_y, t_z) [Pa] on outer faces, in the model axes (not the face '
                            'normal): consistent nodal forces t times the tributary area of each face node (a quarter '
                            'of each adjacent face cell) [N]. Several loads add; on a held component it only enters the '
                            'reaction. On a tetrahedron mesh the consistent face loads int t N dA of the element faces '
                            'beside a solid element, also on domain objects (their interface faces: between two solid '
                            'elements a sheet force on both; a tensor mesh refuses objects).',
                apply=_apply_boundary_load, grids=_GRIDS),
    FeatureType('body_force', 'Body force', 'domain_condition', 'domain',
                params=following((Param('f_x', 'float', 0.0, label='Body force x', unit='N/m^3', field_dependent=True,
                              help='Force per volume along +x; may depend on x, y and z.',
                              tip='Force per volume along x, for example from an acceleration.'),
                        Param('f_y', 'float', 0.0, label='Body force y', unit='N/m^3', field_dependent=True,
                              help='Force per volume along +y; may depend on x, y and z.',
                              tip='Force per volume along y.'),
                        Param('f_z', 'float', -2.16e4, label='Body force z', unit='N/m^3', field_dependent=True,
                              help='Force per volume along +z; gravity along -z is -rho g (silica -2.16e4). May depend '
                                   'on x, y and z.',
                              tip='Force per volume along z; gravity is minus density times 9.81.'))),
                summary='Applies a force per volume throughout the selected domains, such as gravity.',
                description='Applied body force density [N/m^3], consistent nodal forces f times cell volume / 8 to the '
                            'eight corners of each selected cell (a tetrahedron mesh: int f N dV with f per element at '
                            'its centroid); void cells (youngs_modulus 0) drop it. Several body forces add; the coupled '
                            'slots body_force_x / _y / _z are separate.',
                apply=partial(apply_body_force, names=_BODY_FORCES), grids=_GRIDS),
    FeatureType('temperature_change', 'Temperature change', 'domain_condition', 'domain',
                params=(Param('delta_T', 'float', 10.0, label='Temperature change', unit='K', field_dependent=True,
                              help='Temperature above the stress-free state on the selected cells (added to '
                                   'temperature_rise_ext; several add); the thermal strain is alpha * delta_T. May '
                                   'depend on x, y and z.',
                              tip='How much warmer the selected domains are than when they were stress free.'),),
                summary='Heats the selected domains by a given amount without a heat physics, so that they expand.',
                description='Applied temperature change [K] for thermal strain, without a heat physics: added to '
                            'temperature_rise_ext, which adds to the coupled temperature_rise of '
                            f'{THERMAL_STRAIN_COUPLING}.',
                apply=apply_temperature_change, grids=_GRIDS),
    FeatureType('solid_probe', 'Solid probe', 'probe', 'point',
                params=(Param('component', 'choice', 'u_abs', label='Quantity',
                              choices=_PROBE_FIELDS,
                              choice_labels=tuple((name, SolidRenderer3D.field_infos[name].label)
                                                  for name in _PROBE_FIELDS),
                              help='Output field sampled at the point (unit of that field); the electric_* ones (0 '
                                   'without piezoelectric) and the *_energy_density ones only on a tetrahedron mesh.',
                              tip='Quantity recorded at the point.'),),
                object_kinds=('point',), max_entities=1, result_kind='timeseries',
                summary='Records a displacement, strain or stress at a point after every solve.',
                description='Samples one output field in the cell containing the point (its z param, mid-height when '
                            'unset; one sample per re-solve and one per stationary iteration or end-of-run flush, all '
                            'at t = 0 in a stationary study); on a tetrahedron mesh the nodal display field evaluated '
                            'at the point (strain and stress: the node-averaged values).',
                apply=_apply_probe, result=_probe_result, validate=validate_points_inside, grids=_GRIDS),
    frequency_probe_feature(_apply_frequency_probe, ('u_x', 'u_y', 'u_z'), (_TETRAHEDRA,)),
    _resonance_table_feature(),
    SQUEEZE_FILM_FEATURE,
    FeatureType('electrode', 'Electrode', 'boundary_condition', 'boundary', params=_ELECTRODE_PARAMS,
                result_kind='table',
                summary='Puts an electrode on the selected faces of a piezoelectric solid: a fixed voltage, ground, or '
                        'an isolated electrode.',
                description='Electrode on outer faces and on domain objects (their interface faces, outer boundary '
                            'faces of the object included): every node of the faces, edges and corners included, one '
                            f'equipotential. {_ELECTRODE_TEXT} {_ELECTRODE_RESULT_TEXT}',
                apply=_apply_electrode, result=partial(electrode_result, charge_unit='C'), grids=(_TETRAHEDRA,)),
    FeatureType('electrode_domain', 'Electrode (domain)', 'domain_condition', 'domain', params=_ELECTRODE_PARAMS,
                result_kind='table',
                summary='Makes the selected domains (for example a drawn metal film) one electrode of a piezoelectric '
                        'solid.',
                description='Electrode on every node of the selected elements (a meshed metal electrode: give it its '
                            'own stiffness and density for mass loading, or none). '
                            f'{_ELECTRODE_TEXT} {_ELECTRODE_RESULT_TEXT}',
                apply=_apply_electrode_domain, result=partial(electrode_result, charge_unit='C'), grids=(_TETRAHEDRA,)),
    replace(ADMITTANCE_FEATURE, grids=(_TETRAHEDRA,)),
    replace(OPTOMECHANICAL_COUPLING_FEATURE, grids=(_TETRAHEDRA,)),   # em_fdfd_bor optics (optomechanics_fem)
)

_FIELD_INFOS = tuple(SolidRenderer3D.field_infos.values())
# state() of the tensor-mesh solver has no electric / energy arrays (rendered as 0): not usable in expressions
_STATE_FIELD_INFOS = tuple(info for info in _FIELD_INFOS
                           if info.name not in _TETRAHEDRA_ONLY_FIELDS and not info.requires_study)
_SLOT_DOCS = {
    'temperature_rise': ('K', f'Coupled temperature above the stress-free state ({THERMAL_STRAIN_COUPLING}: heat T - '
                              'T0), added to the temperature_change features (field temperature_rise_ext); each '
                              'transfer replaces it.'),
    'body_force_x': ('N/m^3', 'Coupled body force density along x, added to the body_force features; each transfer '
                              'replaces it; void cells drop it.'),
    'body_force_y': ('N/m^3', 'Coupled body force density along y, as body_force_x.'),
    'body_force_z': ('N/m^3', 'Coupled body force density along z, as body_force_x.'),
}

SOLID_MECHANICS_3D = PhysicsType(
    name=SOLID_3D_TYPE_NAME,
    label='Solid Mechanics (3D)',
    default_instance_name='solid',
    family='',
    group='structural',
    dimensions=(3,),
    summary='Static deformation and stress of solids in 3D under supports, loads and thermal expansion.',
    description='3D static linear elasticity, small strain, isotropic cells (youngs_modulus, poisson_ratio -> Lame '
                'lambda, mu) with anisotropic thermal expansion alpha_xx / alpha_yy / alpha_zz: sigma = lambda tr(eps) '
                'I + 2 mu eps - beta dT, beta_i = 2 mu alpha_ii + lambda (alpha_xx + alpha_yy + alpha_zz), dT = '
                f'temperature_rise (coupled, {THERMAL_STRAIN_COUPLING}) + temperature_rise_ext (temperature_change '
                'features). Discretisation: trilinear bricks on the cells of the physical window, displacement on the '
                'cell corners, matrix-free stiffness product, 3 x 3 block-Jacobi PCG (about 20 iterations per cell '
                "along the longest window axis; pcg precision mixed runs them in float32 on float32 copies of the "
                'operator, refined against the float64 residual to the double answer: much cheaper per iteration '
                'on GPUs with slow float64 arithmetic, so the Automatic linear solver on the gpu backend is pcg mixed; '
                'cpu: double); outputs at the cell centres (u = mean of the 8 corners; strain, '
                'stress and von Mises from the centre gradients), 0 in void cells (youngs_modulus 0: not solved, carry '
                'nothing), edge-extended into any PML shell. Forces and reactions in N. strain_xy / _xz / _yz are '
                'tensor shear strains (half the engineering strain); elastic_strain_* = strain - alpha dT. Faces '
                'without a support or load are traction free. Every solid region (cells with stiffness joined through '
                'faces, edges or corners) needs supports that stop all six rigid motions, e.g. rollers on one x, one y '
                'and one z face (validation checks the three directions, the build the rank per region; regions '
                'touching only along an edge or at a corner count as joined, so a hinge there shows as an unconverged '
                'solve). Bricks are stiff in bending of thin members: mesh a film with >= 4 cells through its '
                'thickness. A stationary study solves it after the physics it reads; in a time-dependent study it '
                're-solves only when a material, support, load or coupled input changed (quasi-static, no inertia). '
                'On a tensor mesh an eigenfrequency study finds the n_resonances LOWEST undamped vibration modes K u = '
                'omega^2 M u '
                '(consistent trilinear mass of rho, needed in every solid cell; no target frequency): matrix-free '
                'shift-invert Lanczos at 0 (scipy eigsh, the eigen solver role arpack_shift_invert without a factor), '
                "each Lanczos step one block-Jacobi PCG solve of the stiffness by the 'linear' solver, which "
                'must be double or mixed; the Lanczos tolerance is at least '
                f'{LANCZOS_TOLERANCE_PER_CG:g} x the CG tolerance (the inexact inner solves floor its residual), so '
                'the CG tolerance sets the frequency accuracy (1e-8: ~1e-8 relative; 1e-12 for exact checks). Cost: '
                '~(n_resonances + 4) x a few solves of ~20 CG iterations per cell along the longest axis, no host '
                'factor (memory: a few float64 vectors on the run backend, the Lanczos basis on the host; the cpu '
                'backend runs numpy versions of the matrix-free kernels, the stiffness product numba-compiled when '
                'numba is installed, much slower per CG iteration than the GPU). '
                'Supports held at '
                '0, loads / thermal strain ignored (no prestress); one snapshot per mode labelled "resonance k: f = '
                '<Hz> Hz" with the mode shape scaled to a peak node |u| of 1 m (strain and stress per that shape, no '
                'thermal part; support reactions are not available); the resonance_table feature lists the '
                'frequencies. Interior targets (an indefinite K - sigma M, which CG cannot solve) and free-free bodies '
                'are not available on that mesh. The strain-optic and optical-force couplings are 2D only '
                '(solid_mechanics); the optomechanics study runs on a tetrahedron mesh only, with em_fdfd_bor optics '
                '(optomechanical_coupling: the modes sampled on rings about the body-of-revolution axis; piezoelectric '
                'and periodic axes off). Output u_x, u_y, u_z are the displacement [m], u_abs its magnitude. The '
                "instance is usually named solid. On a tetrahedron mesh (MeshSpec kind 'tetrahedra', 3D Cartesian): "
                'Lagrange elements of the discretization setting (quadratic P2 default, linear P1 locks in bending and '
                'toward poisson_ratio 0.5; affine elements) on the window elements with stiffness (a PML shell is '
                'void); cells may also be anisotropic there: the 21 c_IJ material entries (Voigt 1 xx, 2 yy, 3 zz, 4 yz, '
                '5 xz, 6 xy, engineering shear; T = c S - beta dT with beta_I = c_IJ alpha_J, shear rows included; '
                'an unset entry takes the isotropic value of youngs_modulus / poisson_ratio, a cell is solid when c11 '
                "> 0, the 6 x 6 must be positive definite; a {'model': 'crystal', 'plane': 'full'} entry sets all 21; "
                'a rectangular-cell mesh refuses them), one host SuperLU of the stiffness per assembly (linear solver '
                'sparse_direct, Automatic there; '
                'pcg does not run on tetrahedra), every support and load feature (Fixed, Displacement and Boundary '
                'load also on domain objects: their interface faces; Fixed (domain) holds every node of the selected '
                'elements), probes evaluated at the point; the eigenfrequency study then finds the n_resonances modes '
                'nearest target_frequency (0 = the lowest) by shift-invert Lanczos on a factor of K - sigma M (eigen '
                "solver factorization superlu on the host, or cudss / cudss_mixed: cuDSS LDL^T of the real symmetric "
                'pencil on the GPU, the mixed one in float32 refined to full accuracy); with target_frequency > 0 a free '
                'body (free-free: no supports, or supports that leave rigid motions free) is allowed in that study, its '
                'rigid-body modes (0 Hz) dropped, and in the frequency_response study (regular at f > 0), while static '
                'solves still need supports; a mechanical_loss_factor damps the modes there (complex frequencies, Q '
                'in the resonance table); initial_stress_* material entries prestress every static solve (total '
                'stress sigma0 + C eps - beta dT) and setting stress_stiffening adds the geometric stiffness of the '
                'static state to the eigenfrequency / frequency-response pencils; the frequency_probe feature and '
                'the support results record complex values per frequency; setting geometric_nonlinearity solves '
                'every static state by total-Lagrangian Newton iterations (Green-Lagrange strain, second '
                'Piola-Kirchhoff stress, load_steps, reactions from the internal force) and linearises the pencils '
                'about it; outputs strain_energy_density / kinetic_energy_density (time averages for a mode or a '
                'response; 0 on the rectangular-cell mesh); the Prescribed displacement (domain) feature holds chosen '
                'components of every node of a domain. '
                'state() and coupled slots are '
                'one value per element (centroid; heat on the same tetrahedron mesh couples through '
                f'{THERMAL_STRAIN_COUPLING}), displayed fields nodal (displacement exact, strain / stress averaged over '
                'the solid elements at each node: a material interface is smeared over one element). Setting '
                'piezoelectric (tetrahedra only): a potential on every window element (void ones dielectric) with the '
                'piezo_e* / eps_dc* material entries, Electrode / Electrode (domain) features (voltage or floating, '
                'charges in C), the static coupled solve by the linear sparse_direct solver, the eigenfrequency study '
                'on the coupled pencil (voltage electrodes shorted, floating ones open; each mode with its potential), '
                'outputs electric_potential, electric_field_* and electric_displacement_* (0 without it; the '
                'rectangular-cell mesh shows them as 0). The frequency_response study (tetrahedra only; elastic or '
                'piezoelectric with mechanical_loss_factor / dielectric_loss_tangent) factorises the complex symmetric '
                "K (1 + j eta) - omega^2 M once per frequency by the 'linear' sparse_direct solver (host SuperLU, or "
                'cuDSS LDL^T with back_end cudss on a GPU, its plan reused across frequencies, or an opt-in MUMPS / '
                'PARDISO host LDL^T reusing its analysis); the Admittance feature '
                'gives Y(f) in S with the BVD fit and kappa_e. Settings periodic_x / _y / _z with bloch_kx / _ky / _kz '
                '(tetrahedra, eigenfrequency study only): Bloch / Floquet unit cells, u and phi on each hi face = the lo '
                'face values times exp(-j k L), the modes of the reduced pencil P^H K P, P^H M P (complex Hermitian, '
                'complex modes shown as their real part and |u|), a band diagram by sweeping bloch_k (template '
                f'{_BAND_TEMPLATE}). Settings pml_<face> (tetrahedra, eigenfrequency study, piezoelectric off): an '
                'elastic PML shell of pml_cells element layers beyond those faces (complex coordinate stretch, outer '
                'side clamped), complex modes with the radiation Q in the resonance table. Setting electrode_edge_size: '
                'a gmsh mesh refined along the edges of the electrode objects. The squeeze_film_damping feature '
                '(tetrahedra, eigenfrequency / optomechanics studies) adds the gas damping of a thin film under a flat face per mode '
                '(Reynolds equation on the face, vents, slip) to the resonance table. A Roller on a face of a '
                'crystal that is '
                'not mirror symmetric there gets a warning (a half model would be wrong). Sizes and estimates on '
                'tetrahedra are uncalibrated.',
    settings=(DISCRETIZATION_PARAM, _TARGET_FREQUENCY_PARAM, STRESS_STIFFENING_PARAM, *GEOMETRIC_NONLINEARITY_PARAMS,
              _PIEZOELECTRIC_PARAM, _ELECTRODE_EDGE_SIZE_PARAM, *_periodic_params(), *_pml_params()),
    features=_FEATURES,
    material_properties=(MATERIAL_PROPERTIES + (DENSITY_PARAM,) + STIFFNESS_TENSOR_PROPERTIES_3D
                         + PIEZO_MATERIAL_PROPERTIES_3D + LOSS_PROPERTIES + INITIAL_STRESS_PROPERTIES_3D),
    material_schema={**MECHANICAL_SCHEMA, **DENSITY_SCHEMA},
    optional_material_schema={**OPTIONAL_STIFFNESS_SCHEMA_3D, **OPTIONAL_PIEZO_SCHEMA_3D, **OPTIONAL_LOSS_SCHEMA,
                              **OPTIONAL_PRESTRESS_SCHEMA_3D},
    state_fields=_STATE_FIELD_INFOS,
    output_fields=_FIELD_INFOS,
    source_slots=_SLOT_DOCS,
    supported_studies=('stationary', 'time_dependent', EIGENFREQUENCY_STUDY, FREQUENCY_RESPONSE_STUDY,
                       OPTOMECHANICS_STUDY),
    default_study={'type': 'stationary'},
    # eigen: the factor-free eigsh_inverse on tensor meshes; on tetrahedra eigsh_shifted (SuperLU or cuDSS LDL^T of the
    # real SPD / symmetric pencil; cudss_mixed refines every solve to full accuracy). Automatic is the double factor:
    # ARPACK applies it 37-63 times, double wins from ~11 applications (2 ms per solve against 6-14 ms refined)
    solver_roles={'linear': CG_ELASTIC_3D_LINEAR_ROLE,
                  'eigen': eigen_role(MODAL_STUDIES, mixed_factorization=True,
                                      auto_factorizations={'tetrahedra': 'cudss'})},
    frequency_sweep_methods=('direct', 'adaptive', 'modal'),   # tetrahedra: frequency_response.HarmonicSystem
    backends=('gpu', 'cpu'),   # cpu: numpy twins of the kernels3d RawKernels (Gauss points), numba product
    create=_create_solid_3d,
    domain_requirements=_domain_requirements,
    mesh_edge_sizes=_mesh_edge_sizes,
    validate=_validate_solid_3d,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    grids=_GRIDS,
    mesh_method=_mesh_method,
    # modal.MultiShiftModes on tetrahedra; validate refuses targets / band on a tensor mesh and band for piezoelectric
    eigen_searches=('nearest', 'targets', 'band'),
)


def _selftest_cases():
    """The 3D solid SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_3d import SELFTEST_CASES_3D
    from .selftests_bloch3d import BLOCH_FEM_3D_CASES
    from .selftests_eigen_search import EIGEN_SEARCH_3D_CASES
    from .selftests_eigenfrequency import EIGENFREQUENCY_3D_CASES
    from .selftests_fem3d import FEM_3D_CASES
    from .selftests_free_modes3d import FREE_MODES_3D_CASES
    from .selftests_frequency_response_3d import FREQUENCY_RESPONSE_3D_CASES
    from .selftests_mesh_symmetry3d import MESH_SYMMETRY_3D_CASES
    from .selftests_nonlinear3d import NONLINEAR_3D_CASES
    from .selftests_piezo_fem3d import PIEZO_FEM_3D_CASES
    from .selftests_pml3d import PML_3D_CASES
    from .selftests_prestress_modes import PRESTRESS_3D_CASES
    from .selftests_squeeze_film import SQUEEZE_FILM_CASES
    from .selftests_squeeze_film_model import SQUEEZE_FILM_MODEL_CASES
    from .selftests_swept_plate import SWEPT_PLATE_CASES
    return (SELFTEST_CASES_3D + EIGENFREQUENCY_3D_CASES + EIGEN_SEARCH_3D_CASES + FEM_3D_CASES + PIEZO_FEM_3D_CASES
            + FREQUENCY_RESPONSE_3D_CASES + BLOCH_FEM_3D_CASES + MESH_SYMMETRY_3D_CASES + PML_3D_CASES
            + FREE_MODES_3D_CASES + PRESTRESS_3D_CASES + NONLINEAR_3D_CASES + SQUEEZE_FILM_CASES
            + SQUEEZE_FILM_MODEL_CASES + SWEPT_PLATE_CASES)


def register(registry):
    """solid_mechanics_3d, the 'heated_chip_stress_3d' and 'solid_bloch_band_diagram_3d' templates and the 3D selftest
    loader (called by definition.register after the 2D type, before heat_solid_thermal_strain names this type as a
    target)."""
    registry.register_physics(SOLID_MECHANICS_3D)
    registry.register_template(
        'heated_chip_stress_3d', HEATED_CHIP_STRESS_LABEL, HEATED_CHIP_STRESS_DESCRIPTION, heated_chip_stress_3d_spec,
        summary='Thermal stress in 3D: a silicon block clamped on a heat sink with a heated chip on top; the hot top '
                'expands and bends the block. Steady state.')
    registry.register_template(
        _BAND_TEMPLATE, BAND_DIAGRAM_3D_LABEL, BAND_DIAGRAM_3D_DESCRIPTION, solid_bloch_band_diagram_3d_spec,
        summary='Band diagram of a phononic crystal: one cell of a corrugated silicon nanobeam, its vibration '
                'frequencies for each wave number along the beam.')
    registry.register_selftests(_selftest_cases)
