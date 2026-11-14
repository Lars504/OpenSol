"""Validate / apply helpers shared by the physics definition.py modules (point features, outer-face and
held-object conflicts, parameter-evaluable predicates, the xy-tensor bound, the study duration, the wavelength-range
check, the solver roles and the CG iteration-limit warning (numbers from solvers.pcg), plus the material Params, flux /
probe callbacks and conductor checks of the DiffusionSystem types, and the shared 'geometry' setting (planar /
axisymmetric) with its (r, z) validation)."""
import math

import numpy as np

from ..core.constants import FACES_3D
from ..core.fv_diffusion import BCSide
from ..core.mirror import mirror_faces
from ..core.monitors import probe_result
from ..fem.squeeze_film import SLIP_MODELS, VEIJOLA_KNUDSEN_LIMIT
from ..solvers.host_direct import HOST_DIRECT_BACK_ENDS, HOST_DIRECT_DOUBLE_BACK_ENDS, HOST_DIRECT_MIXED_BACK_ENDS
from ..solvers.pcg import CG_ITERATIONS_PER_AXIS_CELL, SOLVER_DTYPE_NAMES, auto_max_iterations
from ..solvers.roles import SolverRole, resolved_precision
from .expressions import TIME_NAMES, Expression, ExpressionError, evaluate_scalar, referenced_names
from .geometry import (bounding_box, domain_size_z, expand_entities, perimeter_weights, point_z,
                       preview_pixel_centers)
from .schema import Issue, Param, feature_path, known_physics, state_field_names


def evaluates_to(value, parameters, predicate):
    """predicate(number) for a value that evaluates over the parameters; True for field/state-dependent values
    (unknown at validation); False when the value is unset."""
    if value is None:
        return False
    try:
        return predicate(evaluate_scalar(value, parameters))
    except ExpressionError:
        return True


def plain_number(value, parameters):
    """Float of a value that evaluates over the parameters; None when unset or field- / state-dependent."""
    if value is None:
        return None
    try:
        return float(evaluate_scalar(value, parameters))
    except ExpressionError:
        return None


def indefinite_cross_term_issue(material_name, properties, parameters, prefix, unset_value):
    """Error at the material when <prefix>_xy^2 >= xx * yy with all three plain numbers and xx, yy > 0 (an indefinite
    operator: CG stalls); xx / yy fall back to the isotropic <prefix>, else unset_value. None otherwise."""
    isotropic = properties.get(prefix, unset_value)
    try:
        cross = evaluate_scalar(properties.get(f'{prefix}_xy', 0.0), parameters)
        along_x = evaluate_scalar(properties.get(f'{prefix}_xx', isotropic), parameters)
        along_y = evaluate_scalar(properties.get(f'{prefix}_yy', isotropic), parameters)
    except ExpressionError:   # field/state-dependent (unknown at validation) or malformed (reported elsewhere)
        return None
    # cells with xx or yy <= 0 drop the cross term (the 2x2-block rule), so only conducting cells can go indefinite
    if cross == 0.0 or along_x <= 0.0 or along_y <= 0.0 or cross * cross < along_x * along_y:
        return None
    return Issue('error', ('material', material_name),
                 f"material '{material_name}': {prefix}_xy^2 >= {prefix}_xx * {prefix}_yy ({cross:.4g}^2 >= "
                 f"{along_x:.4g} * {along_y:.4g}) makes the {prefix} tensor indefinite, so the CG solve stalls; keep "
                 f"{prefix}_xy^2 < {prefix}_xx * {prefix}_yy")


def validate_points_inside(vctx, physics_spec, feature_spec, values):
    """Error for every selected point object outside the physical domain (a polar / spherical one: its sector; 3D:
    also z outside [0, size_z])."""
    window_map = vctx.domain_map
    if window_map is None:
        return []
    x_min, x_max, y_min, y_max = window_map.extent()
    edge_tolerance = 1e-9 * max(x_max - x_min, y_max - y_min)   # a sum of lengths can overshoot an edge by an ulp
    try:
        size_z = domain_size_z(vctx.spec, vctx.parameters)   # None in 2D
    except (ValueError, TypeError, ExpressionError):   # reported at the domain
        size_z = None
    where = (f"the {window_map.grid_text} domain ({'on y = 0, ' if window_map.line else ''}r "
             f'{window_map.radius_min:.10g} to {window_map.radius_max:.10g} m)'
             if window_map.mapped else f'the physical domain [0, {x_max:.10g}] x [0, {y_max:.10g}] m')
    issues = []
    for name in vctx.objects(feature_spec.selection):
        shape = vctx.shapes.get(name)
        if shape is None or shape.kind != 'point':
            continue
        x, y = shape.values['x'], shape.values['y']
        if not window_map.contains(x, y, edge_tolerance):
            issues.append(Issue('error', feature_path(physics_spec, feature_spec),
                                f"point '{name}' at ({x:.10g}, {y:.10g}) m lies outside {where}"))
        elif size_z is not None:
            z = point_z(shape.values, size_z)
            z_tolerance = 1e-9 * max(x_max - x_min, y_max - y_min, size_z)
            if not -z_tolerance <= z <= size_z + z_tolerance:
                issues.append(Issue('error', feature_path(physics_spec, feature_spec),
                                    f"point '{name}' at z = {z:.10g} m lies outside the domain height [0, "
                                    f"{size_z:.10g}] m"))
    return issues


def enabled_feature_entities(vctx, physics_spec, feature_types=None):
    """Yield (feature, expanded entities) of the physics' enabled features (of feature_types, None = all); a
    selection that does not expand is skipped (the selection checks report it)."""
    for feature in physics_spec.features:
        if not feature.enabled or (feature_types is not None and feature.type not in feature_types):
            continue
        try:
            entities = expand_entities(vctx.spec, feature.selection)
        except (ValueError, TypeError):   # geometry.expand_entities raises only these
            continue
        yield feature, entities


def face_conflicts(vctx, physics_spec, boundary_feature_types, path):
    """Warning per outer face set by more than one enabled feature of the given types (the last one wins)."""
    owners = {}
    for feature, entities in enabled_feature_entities(vctx, physics_spec, boundary_feature_types):
        for face in FACES_3D:   # z faces only pass validation in 3D models
            if face in entities:
                owners.setdefault(face, []).append(feature.name)
    return [Issue('warning', path, f"outer face {face} is set by features {', '.join(names)}; "
                                   f"the last one ('{names[-1]}') wins")
            for face, names in owners.items() if len(names) > 1]


def mirror_face_issues(vctx, physics_spec, boundary_feature_types, path, consequence, severity='warning'):
    """One issue per enabled boundary feature of physics_spec setting a face that is an EM mirror plane (core.mirror:
    a physics' bc_<face> symmetric / antisymmetric): only the default zero-flux face there is the even mirror image
    of the unfolded model; consequence = the type's sentence on what the feature models instead."""
    faces = mirror_faces(vctx.spec)
    issues = []
    for feature, entities in enabled_feature_entities(vctx, physics_spec, boundary_feature_types) if faces else ():
        on_plane = [face for face in faces if face in entities]
        if on_plane:
            issues.append(Issue(severity, feature_path(physics_spec, feature),
                                f"{', '.join(on_plane)} lies on an EM mirror plane, where the mirrored (unfolded) "
                                f'model needs this physics insulated (no boundary feature): {consequence}'))
    return issues


def _boxes_intersect(first_box, second_box):
    """True when two (x0, y0, x1, y1) boxes share interior area (touching edges do not count)."""
    return (first_box[0] < second_box[2] and second_box[0] < first_box[2]
            and first_box[1] < second_box[3] and second_box[1] < first_box[3])


def held_object_conflicts(vctx, physics_spec, held_types, perimeter_types, path):
    """Warning per pair of enabled features, at least one of held_types, whose objects may share cells: a shared
    object or 'all', or (two held features) intersecting bounding boxes. Held cells ignore perimeter_types."""
    regions = []   # (feature, object names, bounding boxes, selects 'all')
    for feature, entities in enabled_feature_entities(vctx, physics_spec, held_types | perimeter_types):
        object_names = [name for name in entities if name in vctx.shapes]
        boxes = []
        for name in object_names:
            try:
                boxes.append(bounding_box(name, vctx.shapes))
            except (KeyError, ValueError):   # a failed sub-object (reported by the geometry checks)
                continue
        selects_all = 'all' in entities
        if object_names or selects_all:
            regions.append((feature, set(object_names), boxes, selects_all))
    issues = []
    for index, (first, first_names, first_boxes, first_all) in enumerate(regions):
        for second, second_names, second_boxes, second_all in regions[index + 1:]:
            first_held, second_held = first.type in held_types, second.type in held_types
            if not (first_held or second_held):
                continue
            shared_names = first_names & second_names
            if shared_names:
                reason = f"both select {', '.join(sorted(shared_names))}"
            elif first_all or second_all:
                reason = "one selects 'all'"
            elif first_held and second_held and any(_boxes_intersect(first_box, second_box)
                                                    for first_box in first_boxes for second_box in second_boxes):
                reason = 'their bounding boxes intersect'
            else:
                continue
            if first_held and second_held:
                consequence = f"the later one ('{second.name}') holds the shared cells"
            else:
                held, other = (first, second) if first_held else (second, first)
                consequence = f"cells held by '{held.name}' leave the solve, so '{other.name}' does nothing there"
            issues.append(Issue('warning', path,
                                f"features '{first.name}' and '{second.name}' may overlap ({reason}); {consequence}"))
    return issues


def study_duration(vctx):
    """Resolved study 'duration' [s], or None when the study has none or it does not resolve."""
    try:
        study_type = vctx.registry.study_type(vctx.spec.study.type)
    except KeyError:
        return None
    if not any(param.name == 'duration' for param in study_type.params):
        return None
    values = vctx.values(study_type.params, vctx.spec.study.settings)
    return None if values is None else values.get('duration')


def wavelength_range_issues(values, path):
    """[] or the one error for a wavelength_start / wavelength_stop range that does not increase (equal ends only
    with n_wavelengths = 1)."""
    start, stop, count = values['wavelength_start'], values['wavelength_stop'], values['n_wavelengths']
    if stop > start or (count == 1 and stop == start):
        return []
    if count == 1:
        return [Issue('error', path, f'wavelength_stop ({stop:.6g} m) must be >= wavelength_start ({start:.6g} m)')]
    return [Issue('error', path, f'wavelength_stop ({stop:.6g} m) must be > wavelength_start ({start:.6g} m) for '
                                 f'n_wavelengths = {count}; swap them, or set n_wavelengths = 1 for one wavelength')]


def point_cells(ctx, feature_spec, selection):
    """Full-grid cells (ix, iy) of the selected point objects (3D models: (ix, iy, iz) at the point's height);
    ValueError for a point outside the physical domain."""
    cells = []
    for name in selection.objects:
        position = ctx.point(name) + ((ctx.height(name),) if ctx.dimensions == 3 else ())
        ctx.domain.validate_physical_pos(*position, f"feature '{feature_spec.name}' point '{name}'")
        cells.append(ctx.domain.pos_to_cell(*position))
    return cells


def cg_iteration_issues(vctx, solver_values, path, all_cell_sink=False):
    """[] or one warning for an explicit pcg max_iterations below the cold-start estimate of the meshed window
    (skipped with a volumetric sink on every cell, which bounds the condition number)."""
    max_iterations = (solver_values or {}).get('max_iterations')
    mesh = vctx.mesh
    if not max_iterations or mesh is None or all_cell_sink or mesh.kind != 'tensor':   # triangles: no cell axes
        return []
    longest_window_axis = max(mesh.window_axis_cells)
    cold_start_iterations = CG_ITERATIONS_PER_AXIS_CELL * longest_window_axis
    if max_iterations >= cold_start_iterations:
        return []
    return [Issue('warning', path,
                  f'max_iterations {max_iterations} is below the ~{cold_start_iterations} Jacobi-PCG iterations a '
                  f'cold-start solve needs on this window ({longest_window_axis} cells along its longest axis): solves '
                  'may stop unconverged and keep a biased result; use 0 (auto = '
                  f'{auto_max_iterations(longest_window_axis)}) or at least {cold_start_iterations}')]


# sparse_direct back_end values of the SPD 'linear' roles (below, and solid_mechanics' own
# physics.mechanics.definition.SOLID_LINEAR_ROLE), whose direct solves go through DiffusionDirectSolver (host
# SuperLU, the real cuDSS Cholesky of the opt-in cuDSS back ends or the host MUMPS / PARDISO Cholesky of the opt-in
# host direct ones, solvers.host_direct; never cusolver_qr)
DIFFUSION_DIRECT_BACK_ENDS = ('auto', 'scipy_splu', 'cudss', 'cudss_mixed') + HOST_DIRECT_BACK_ENDS
# The 'linear' SolverRole of solid_mechanics_3d: pcg single / double / 'mixed' (float32 inner CG on float32 copies of
# the matrix-free brick operator and its block inverses); Automatic on the gpu backend (tensor mesh)
# = pcg mixed (speed plan A1: the float64 brick product runs at the FP64 rate of consumer cards, slower than a 10-thread
# CPU; mixed reached the same 1e-12 answer 17x faster at 2e5 nodes; a pcg node setting no precision takes it too,
# schema.with_node_precision); on a tetrahedron mesh (FEM) Automatic =
# sparse_direct, one host SuperLU per assembly through DiffusionDirectSolver (a new sparse_direct back end belongs in
# DIFFUSION_DIRECT_BACK_ENDS); back_end 'auto' = cudss_mixed there on the gpu backend (B1a: the 3D factor dominates and
# double runs ~12x the mixed factor time; a failing float32 factor is rejected early and remembered per pattern, H4).
# preconditioner 'multigrid' (any precision, mixed included): the geometric V-cycle of physics.mechanics.multigrid3d on
# the brick operator (speed plan A2), opt-in until void / bilayer / graded cases pass with it
CG_ELASTIC_3D_LINEAR_ROLE = SolverRole('spd', 'pcg', precisions=('single', 'double', 'mixed'),
                                       setting_choices={'preconditioner': ('jacobi', 'multigrid'),
                                                        'back_end': DIFFUSION_DIRECT_BACK_ENDS},
                                       mixed_preconditioners=('jacobi', 'multigrid'),
                                       backend_defaults={'gpu': ('pcg', {'precision': 'mixed'})},
                                       grid_defaults={'tetrahedra': ('sparse_direct', {})},
                                       auto_back_ends={'tetrahedra': 'cudss_mixed'})
# ... of the DiffusionSystem types (heat, electric_currents, electrostatics, magnetostatics): an assembled scalar CSR,
# so a pcg node may also run 'mixed' or the multigrid preconditioner (2D solid mechanics: block Jacobi only); Automatic
# on the gpu backend: pcg with multigrid (mesh plan D16 / M0; with solver_defaults.precision 'mixed'
# schema.resolved_solvers falls back to jacobi); on the cpu backend: host SuperLU in 2D (solver_plan S3.4: a 2D direct
# solve beats Jacobi-PCG on a CPU up to ~1e6 cells; 700 x 700: 4.2 s for a factor + 20 steps vs 7.1 s multigrid PCG),
# but pcg with multigrid on a 3D tensor mesh on both backends (grid 'tensor_3d': SuperLU fill ~ n^(4/3); 60^3 cells:
# 38.6 s for one SuperLU solve vs 1.0 s multigrid PCG); in a 1D model (grid 'line', symmetry 2.20) on both backends:
# the banded host Cholesky; on a triangle / tetrahedron mesh (FEM, mesh plan D11) on both backends: sparse_direct (host
# SuperLU; a pcg node is opt-in there; back_end 'auto' = cudss on the gpu backend, B1a: double, since magnetostatics'
# mu_r makes cond(E) pass the float32 limit and one factor serves few solves). A time_dependent study on a 2D tensor
# mesh: sparse_direct on both backends (study_defaults; speed plan A3: heat_2d spent 5.0 of 5.6 s in 1000 multigrid
# PCG solves, while a fixed dt keeps one factor, DiffusionSystem.refresh_capacity), back_end 'auto' = cudss on the gpu;
# steady studies and 3D stay multigrid
CG_DIFFUSION_LINEAR_ROLE = SolverRole('spd', 'pcg', default_settings={'preconditioner': 'multigrid'},
                                      precisions=('single', 'double', 'mixed'),
                                      setting_choices={'back_end': DIFFUSION_DIRECT_BACK_ENDS},
                                      backend_defaults={'cpu': ('sparse_direct', {})},
                                      grid_defaults={'line': ('banded', {}),
                                                     'tensor_3d': ('pcg', {'preconditioner': 'multigrid'}),
                                                     'triangles': ('sparse_direct', {}),
                                                     'tetrahedra': ('sparse_direct', {})},
                                      study_defaults={('time_dependent', 'tensor'): ('sparse_direct', {})},
                                      auto_back_ends={'tensor': 'cudss', 'triangles': 'cudss', 'tetrahedra': 'cudss'})


def _linear_dtype_name(solvers) -> str:
    """dtype name ('float64' / 'float32') of the resolved 'linear' solver in {role: ResolvedSolver} (or None);
    'float64' (the pcg default) when it does not resolve."""
    resolved = (solvers or {}).get('linear')
    if resolved is None:
        return 'float64'
    return SOLVER_DTYPE_NAMES[resolved_precision(resolved.solver_type, resolved.values)]


def linear_solver_precision(context):
    """dtype name ('float64' / 'float32') of the physics' resolved 'linear' solver (memory estimators: MemoryContext
    solvers)."""
    return _linear_dtype_name(context.solvers)


def linear_solver_mixed(context) -> bool:
    """Whether the physics' resolved 'linear' solver runs mixed precision (its estimator adds the float32 copies)."""
    resolved = context.solvers.get('linear')
    return resolved is not None and resolved_precision(resolved.solver_type, resolved.values) == 'mixed'


def linear_solver_direct(context) -> str:
    """Back end name (truthy) when the physics' resolved 'linear' solver is sparse_direct with a host factor, else ''
    (the cpu Automatic of the diffusion types: 'scipy_splu', its estimator adds the host SuperLU factor; a host direct
    name, `host_direct_back_end`: the MUMPS / PARDISO factor; a cuDSS back end's device factor lies outside the cupy
    pool: an external item, cudss_factor.factor_device_bytes). Callers use it as a bool or pass it on as direct=."""
    resolved = context.solvers.get('linear')
    if resolved is None or resolved.solver_type.name != 'sparse_direct' or linear_solver_cudss(context.solvers):
        return ''
    return resolved_back_end(context.solvers) or 'scipy_splu'


def host_direct_back_end(solvers, role='linear'):
    """The host direct back end (solvers.host_direct.HOST_DIRECT_BACK_ENDS: MUMPS / PARDISO, + _mixed) the resolved
    `role` solver ({role: ResolvedSolver} of MemoryContext / ValidationContext, or None) uses: its sparse_direct
    back_end, for role 'eigen' its arpack_shift_invert factorization; None for any other solver or value."""
    resolved = (solvers or {}).get(role)
    if resolved is None:
        return None
    name = resolved.values.get('factorization' if role == 'eigen' else 'back_end')
    return name if name in HOST_DIRECT_BACK_ENDS else None


def linear_solver_cudss(solvers) -> bool:
    """Whether the resolved 'linear' solver ({role: ResolvedSolver} of MemoryContext / ValidationContext, or None) is
    sparse_direct on a cuDSS back end (cudss / cudss_mixed)."""
    from ..solvers.sparse_direct import CUDSS_BACK_ENDS   # numpy only at module level
    resolved = (solvers or {}).get('linear')
    return (resolved is not None and resolved.solver_type.name == 'sparse_direct'
            and resolved.values.get('back_end') in CUDSS_BACK_ENDS)


def linear_solver_multigrid(context) -> bool:
    """Whether the physics' resolved 'linear' solver uses the multigrid preconditioner (its estimator adds the
    hierarchy)."""
    resolved = context.solvers.get('linear')
    return resolved is not None and resolved.values.get('preconditioner') == 'multigrid'


def coupling_target(context) -> bool:
    """Whether an enabled coupling writes into the physics (run-time recipes of re-solve-on-change physics: a pure
    coupling source solves once)."""
    return any(coupling.enabled and coupling.target == context.name for coupling in context.spec.couplings)


def optimization_evaluations(context) -> tuple:
    """(count, kind, limit) of the design evaluations of an optimization study (run-time recipes): about one per
    L-BFGS-B iteration + the initial design ('guess'), at most the study's 'evaluations' count (maxfun per stage)."""
    limit = float(context.count('evaluations', 1))
    return min(float(context.count('iterations', 1)) + 1.0, limit), 'guess', limit


def material_follows_state(context, property_names) -> bool:
    """Whether a spec material sets one of property_names to an expression of a physics state field or time t (a
    DynamicMaterialUpdate re-writes it during the run)."""
    state_names = frozenset(state_field_names(context.spec, context.registry)) | TIME_NAMES
    for material in context.spec.materials:
        properties = material.properties if isinstance(material.properties, dict) else {}
        for name in property_names:
            value = properties.get(name)
            if not isinstance(value, str):
                continue
            try:
                if not referenced_names(value).isdisjoint(state_names):
                    return True
            except ExpressionError:   # reported by validate
                continue
    return False


def resolved_stepper_precision(solvers) -> str:
    """Precision of the resolved 'stepper' solver ({role: ResolvedSolver} of MemoryContext / ValidationContext, or
    None), 'single' without one."""
    resolved = (solvers or {}).get('stepper')
    return 'single' if resolved is None else resolved_precision(resolved.solver_type, resolved.values)


def validated_linear_precision(vctx, physics_spec):
    """linear_solver_precision at validation (ValidationContext.solvers; 'float64' when they do not resolve)."""
    return _linear_dtype_name(vctx.solvers(physics_spec))


def cg_solver_issues(vctx, physics_spec, path, all_cell_sink=False):
    """Iteration-limit warning of the physics' assigned pcg 'linear' node (Automatic = pcg defaults: none; the node's
    precision warning is at the node: pcg SolverType validate); the host SuperLU size rule of a sparse_direct one (also
    for a MUMPS / PARDISO back end, worded for its library; none on a cuDSS back end: guarded by cuDSS's device
    estimate at run time)."""
    solvers = vctx.solvers(physics_spec)
    resolved = (solvers or {}).get('linear')
    if resolved is not None and resolved.solver_type.name == 'sparse_direct':
        if linear_solver_cudss(solvers):
            return []
        role = vctx.registry.physics_type(physics_spec.type).solver_roles.get('linear')
        return _direct_diffusion_size_issues(vctx, path, host_direct_back_end(solvers) or 'scipy_splu',
                                             role.direct_components if role is not None else 1)
    if resolved is None or resolved.node is None or resolved.solver_type.name != 'pcg':
        return []
    return [Issue(issue.level, issue.path, f"linear solver '{resolved.node}': {issue.message}")
            for issue in cg_iteration_issues(vctx, resolved.values, path, all_cell_sink=all_cell_sink)]


# coefficient contrast from which a 'single' (float32) CG answer drifts: magnetostatics with iron at mu_dc 100 came out
# ~8 % off (measured); single_precision_contrast_issues warns from here
SINGLE_PRECISION_CONTRAST = 100.0


def single_precision_contrast_issues(vctx, physics_spec, path, property_names, label):
    """Warning when the physics' resolved 'linear' solver runs 'single' and the constant property_names of the selected
    materials span max / min >= SINGLE_PRECISION_CONTRAST, unset cells counting 1 (expression values are skipped)."""
    resolved = (vctx.solvers(physics_spec) or {}).get('linear')
    if resolved is None or resolved_precision(resolved.solver_type, resolved.values) != 'single':
        return []
    values = [1.0]
    for _, properties, _ in selected_material_data(vctx.spec):
        for name in property_names:
            value = plain_number(properties.get(name), vctx.parameters)
            if value is not None and value > 0.0:
                values.append(value)
    contrast = max(values) / min(values)
    if contrast < SINGLE_PRECISION_CONTRAST:
        return []
    solver = (f"linear solver '{resolved.node}'" if resolved.node is not None
              else "the Automatic linear solver (solver_defaults.precision 'single')")
    return [Issue('warning', path, f'{solver} runs single precision on a {label} contrast of {contrast:.3g} (>= '
                                   f'{SINGLE_PRECISION_CONTRAST:g}): float32 CG stops at a biased answer there (mu_dc '
                                   "100 iron: ~8 % error); use precision 'double' or 'mixed'")]


def direct_unknowns(mesh, components=1) -> int:
    """Unknowns of a direct 'linear' operator on a MeshEstimate (SolverRole.direct_components): element meshes
    components x (nodes + edges) (P2, an upper figure for P1); tensor meshes the window cells (components 1, cell-centred)
    or components x the window nodes (2D solid Q1)."""
    if mesh.kind in ('triangles', 'tetrahedra'):
        return components * (int(mesh.n_nodes) + int(mesh.n_edges))
    if components == 1:
        return math.prod(mesh.window_axis_cells)
    return components * math.prod(count + 1 for count in mesh.window_axis_cells)


def _direct_diffusion_size_issues(vctx, path, back_end='scipy_splu', components=1):
    """[] or the sparse_direct.diffusion_size_issue of direct_unknowns on the physics' mesh (3D: its voxels) for back_end
    ('scipy_splu' or a host direct one: the same limits) as an Issue at path."""
    from ..solvers.sparse_direct import diffusion_size_issue   # numpy only at module level
    mesh = vctx.mesh
    if mesh is None:
        return []
    label = None if components == 1 else f'unknowns ({components} per node)'
    issue = diffusion_size_issue(direct_unknowns(mesh, components), dimensions=3 if vctx.spec.is_3d else 2,
                                 back_end=back_end, unknowns_label=label)
    return [] if issue is None else [Issue(issue[0], path, issue[1])]


# ---- direct / eigen / stepper solver roles (solvers.sparse_direct, arpack_shift_invert, explicit_kernels) ----
# The host direct back ends (solvers.host_direct: MUMPS / PARDISO, opt-in, never 'auto') join every role whose
# consumers hold a generic factor (solve / solve_host, transposed solves, recycling, inertia: all supported); tuples
# derive from host_direct.HOST_DIRECT_* only, so dropping a value there needs no edit here.
# 'linear' of magnetic_fields_ac: host SuperLU, MUMPS / PARDISO or a cuDSS LDL^T of the complex symmetric bordered system
# (circuit.RecycledFactor, plan reused across operator changes); back_end 'auto' = cudss on triangles on the gpu
# backend (B1a: double, mu contrasts and circuit rows make complex64 doubtful, so no mixed back end; 1.7e5 P2 unknowns
# took 7 s per SuperLU factor)
MAGNETIC_AC_DIRECT_BACK_ENDS = ('auto', 'scipy_splu', 'cudss') + HOST_DIRECT_DOUBLE_BACK_ENDS
MAGNETIC_AC_LINEAR_ROLE = SolverRole('general', 'sparse_direct', precisions=('double',),
                                     setting_choices={'back_end': MAGNETIC_AC_DIRECT_BACK_ENDS},
                                     auto_back_ends={'triangles': 'cudss'})
# ... of em_fdfd_ez / _hz: also a line_block node (solver_plan Part II stage 2; Automatic stays sparse_direct) and the
# opt-in cuDSS back ends (tensor: LDL^T of the symmetrised D A, plan reused across wavelengths and material changes;
# triangles: LDL^T of the FEM pencil; physics/em sub file FDFD) and the host direct ones (the same LDL^T on the host);
# back_end 'auto' = cudss on the gpu backend (B1a: double, one factor serves many source / port / adjoint solves and
# mixed solves run ~2x slower)
FDFD_DIRECT_BACK_ENDS = ('auto', 'scipy_splu', 'cusolver_qr', 'cudss', 'cudss_mixed') + HOST_DIRECT_BACK_ENDS
FDFD_LINEAR_ROLE = SolverRole('general', 'sparse_direct', precisions=('double', 'mixed'), opt_in_types=('line_block',),
                              setting_choices={'back_end': FDFD_DIRECT_BACK_ENDS},
                              auto_back_ends={'tensor': 'cudss', 'triangles': 'cudss'})
# ... of drift_diffusion, em_fdfd_bor and em_fdfd_conical: the same back ends, double only (cuDSS: drift_diffusion a
# real LU of every Newton Jacobian, BOR an LDL^T of its complex symmetric operator, conical of D A; plan reused while
# the pattern holds; physics/semiconductor and physics/em sub files); back_end 'auto' = cudss on the gpu backend (B1a;
# BOR / conical drive on tensor meshes only, drift_diffusion also on triangles)
CUDSS_LINEAR_ROLE = SolverRole('general', 'sparse_direct', precisions=('double',),
                               setting_choices={'back_end': FDFD_DIRECT_BACK_ENDS},
                               auto_back_ends={'tensor': 'cudss', 'triangles': 'cudss'})
# ... of em_fdfd_nonlinear: the FDFD back ends, but no auto_back_ends: 'auto' stays host SuperLU (each channel keeps
# its factor, recycled by GMRES across picard Kerr / carrier sweeps; ~1e4-unknown strips: a SuperLU solve ~0.2 ms beats
# cuDSS's per-solve overhead and ~15 ms plan per channel). cudss / cudss_mixed are opt-in for large grids (a device
# factor pays from ~1e5 cells, unmeasured): no recycling, so picard Kerr channels refactorise per sweep
DIRECT_LINEAR_ROLE = SolverRole('general', 'sparse_direct', precisions=('double',),
                                setting_choices={'back_end': FDFD_DIRECT_BACK_ENDS})
# 'stepper' of the explicit time-stepping engines: em_fdtd_ez / _hz single or double (float64 fields); em_fdtd_3d,
# elastic_waves and laminar_flow single or fp16 field storage
STEPPER_ROLE = SolverRole('general', 'explicit_kernels', precisions=('single', 'double'))
HALF_STORAGE_STEPPER_ROLE = SolverRole('general', 'explicit_kernels', precisions=('single', 'half_storage'))


def eigen_role(studies=(), mixed_factorization=False, auto_factorizations=None, block_eigen=False):
    """'eigen' SolverRole (arpack_shift_invert by default) used in `studies` (() = every study); block_eigen: a real
    symmetric pencil whose physics also runs the factor-free 'block_eigen' solver (opt-in, precisions double / mixed:
    operator class 'spd'; solid_mechanics_3d on brick meshes); factorization
    cudss_mixed only with mixed_factorization (callers of solve_resonances with a symmetrizer: em_fdfd_3d, em_fdfd,
    em_fdfd_bor, em_fdfd_conical; the real symmetric eigsh_shifted pencils of solid_mechanics (2D) and
    solid_mechanics_3d on tetrahedra, refined to full accuracy; em_modes, whose triangle pencil is Rayleigh-polished:
    its validation refuses tensor meshes); the host direct factorizations (MUMPS / PARDISO) always, their mixed ones
    likewise only with mixed_factorization. auto_factorizations: {grid kind: cuDSS factorization} an 'auto'
    factorization becomes on the gpu with nvmath-python (the runtime falls back to superlu, B1b); other grids: superlu."""
    factorizations = (('auto', 'superlu', 'cudss') + HOST_DIRECT_DOUBLE_BACK_ENDS
                      + ((('cudss_mixed',) + HOST_DIRECT_MIXED_BACK_ENDS) if mixed_factorization else ()))
    return SolverRole('spd' if block_eigen else 'general', 'arpack_shift_invert', studies=tuple(studies),
                      precisions=('double', 'mixed') if block_eigen else ('double',),
                      setting_choices={'factorization': factorizations},
                      auto_back_ends=dict(auto_factorizations or {}),
                      opt_in_types=('block_eigen',) if block_eigen else (),
                      # block_eigen mixed runs the physics' own float32 multigrid (solver_misfit / resolved_solvers
                      # would else refuse it or drop it to jacobi); arpack has no preconditioner setting
                      mixed_preconditioners=('jacobi', 'multigrid') if block_eigen else ('jacobi',))


def resolved_back_end(solvers):
    """Back end (sparse_direct.BACK_ENDS) the resolved 'linear' sparse_direct solver uses on this machine, None
    without one (CPU: MemoryContext.solvers or ValidationContext.solvers)."""
    from ..solvers.sparse_direct import expected_back_end
    resolved = (solvers or {}).get('linear')
    if resolved is None or 'back_end' not in resolved.values:
        return None
    return expected_back_end(resolved.values['back_end'])


def eigen_subspace_size(solvers):
    """subspace_size of the resolved 'eigen' solver (0 = scipy's default; also without one)."""
    resolved = (solvers or {}).get('eigen')
    return 0 if resolved is None else int(resolved.values.get('subspace_size', 0))


def eigen_factorization(solvers):
    """factorization of the resolved 'eigen' solver (arpack_shift_invert.FACTORIZATIONS; 'superlu' also without one)."""
    resolved = (solvers or {}).get('eigen')
    return 'superlu' if resolved is None else resolved.values.get('factorization', 'superlu')


def eigen_search(spec) -> str:
    """The eigenfrequency study's search setting of a ModelSpec ('nearest' unset, or in any other study)."""
    study = spec.study
    if study.type != 'eigenfrequency':
        return 'nearest'
    search = (study.settings if isinstance(study.settings, dict) else {}).get('search', 'nearest')
    return search if isinstance(search, str) else 'nearest'


def eigen_search_counts(context) -> tuple:
    """(SolverWork kwargs solves / factorizations / factorizations_kind of the 'eigen' role, resonances) from the study
    time counts eigen_shifts / eigen_count_factorizations / eigen_band / resonances (absent: one shift, eigenpairs)."""
    shifts = float(context.count('eigen_shifts', 1) or 1)
    factorizations = shifts + float(context.count('eigen_count_factorizations', 0) or 0)
    kind = 'guess' if context.count('eigen_band') else 'exact'
    resonances = float(context.count('resonances') or context.count('eigenpairs', 1) or 1)
    return {'solves': shifts, 'factorizations': factorizations, 'factorizations_kind': kind}, resonances


# reduced frequency sweeps (fem.reduced_harmonic): complex128 bytes per unknown and basis vector of building the model
# (orthogonalisation twice, one product per pencil term, sketch) and of one V y of a published field; uncalibrated
_REDUCED_BUILD_PASSES = 8
_COMPLEX_BYTES = 16
_REDUCED_SKETCH_BYTES_PER_UNKNOWN = 40   # fem.reduced_harmonic sketch: 8 int32 targets + 8 int8 signs per unknown


def frequency_sweep_counts(context) -> dict:
    """Run-time counts of a frequency_response study for a physics' time recipe (StudyType.time_counts + settings):
    method, points, factorizations (+ _kind), back substitutions solves (+ _kind), field_points (frequencies whose fields
    are built), eigenpairs of the modal basis (0 otherwise) and basis (reduced basis bound, 0 direct)."""
    values = context.study.values or {}
    points = float(context.count('harmonic_points', 1) or 1)
    factorizations = float(context.count('harmonic_factorizations', points) or points)
    method = values.get('sweep_method', 'direct')
    if method == 'adaptive':   # moments + 1 vectors per expansion, the verification one solve
        basis = max(factorizations - 1.0, 1.0) * (int(values.get('rom_moments', 4)) + 1)
        solves, kind, eigenpairs = basis + 1.0, 'bounded', 0
    elif method == 'modal':    # the modes, the static correction per load (up to 3) and the verification
        eigenpairs = int(values.get('modal_count', 20))
        basis, solves, kind = eigenpairs + 3.0, 4.0, 'bounded'
    else:
        basis, solves, kind, eigenpairs = 0.0, points, 'exact', 0
    fields_all = method == 'direct' or values.get('snapshots', 'all') == 'all'
    return {'method': method, 'points': points, 'factorizations': factorizations,
            'factorizations_kind': 'bounded' if method == 'adaptive' else 'exact', 'solves': solves, 'solves_kind': kind,
            'field_points': points if fields_all else 1.0, 'eigenpairs': eigenpairs, 'basis': basis}


def reduced_sweep_memory_items(context, unknowns, csr_entries) -> list:
    """Host (label, bytes) items of a frequency_response reduced sweep (sweep_method adaptive / modal) on `unknowns`
    unknowns and `csr_entries` pencil entries (MemoryContext: spec, registry, parameters): the basis at its largest
    (x 2: orthogonalisation temporaries), the adjoint term copies, the residual sketch and a modal ARPACK basis."""
    from ..solvers.arpack_shift_invert import arnoldi_basis_size
    from .schema import resolve_values
    study = context.spec.study
    if study.type != 'frequency_response':
        return []
    try:
        values = resolve_values(context.registry.study_type(study.type).params,
                                study.settings if isinstance(study.settings, dict) else {}, context.parameters)
    except (KeyError, ValueError, TypeError):   # reported by validate
        return []
    method = values.get('sweep_method', 'direct')
    if method == 'adaptive':
        basis = int(values.get('rom_max_points', 40)) * (int(values.get('rom_moments', 4)) + 1)
    elif method == 'modal':
        basis = int(values.get('modal_count', 20)) + 4
    else:
        return []
    unknowns, csr_entry_bytes = int(unknowns), _COMPLEX_BYTES + 4
    items = [(f'reduced sweep basis (up to {basis} complex128 vectors + temporaries)',
              2 * basis * _COMPLEX_BYTES * unknowns),
             ('reduced sweep adjoint term copies (complex128)', 3 * csr_entry_bytes * int(csr_entries)),
             ('reduced sweep residual sketch', _REDUCED_SKETCH_BYTES_PER_UNKNOWN * unknowns)]
    if method == 'modal':
        wanted = 2 * int(values.get('modal_count', 20)) + 4
        items.append(('modal sweep ARPACK basis (complex128)',
                      arnoldi_basis_size(wanted, unknowns) * _COMPLEX_BYTES * unknowns))
    return items


def cudss_external(factors, one_at_a_time=False, phases=None) -> dict:
    """MemoryEstimate.from_items kwargs external_items / external_transient_items / external_extra of cuDSS factors
    outside the cupy pool, factors = ((label, n_unknowns, entries, value_bytes, dimensions, mixed[, symmetric]), ...)
    by cudss_factor.factor_device_bytes; phases {label: run phase or tuple of phases}: a factor alive only in those
    transient phases (paired with their pool items in device_peak); one_at_a_time: each of the others is released
    before the next is planned (largest booked)."""
    from ..solvers.cudss_factor import factor_device_bytes
    phases = phases or {}
    sized = [(factor[0], *factor_device_bytes(*factor[1:6], symmetric=factor[6] if len(factor) > 6 else True))
             for factor in factors]
    run_long = [entry for entry in sized if entry[0] not in phases]
    if one_at_a_time and run_long:
        run_long = [max(run_long, key=lambda entry: entry[1])]
    phased = tuple((phase, label, permanent) for label, permanent, _ in sized if label in phases
                   for phase in ((phases[label],) if isinstance(phases[label], str) else phases[label]))
    return {'external_items': tuple((label, permanent) for label, permanent, _ in run_long),
            'external_transient_items': phased,
            'external_extra': max((extra for _, _, extra in sized), default=0)}


def reduced_sweep_time_items(sweep, unknowns) -> tuple:
    """core.time_estimate TimeItems of a reduced sweep (frequency_sweep_counts) on `unknowns` free unknowns: building
    the basis products once and V y per field frequency; () for 'direct'."""
    from ..core.time_estimate import HostStream, PythonCalls, TimeItem
    if sweep['method'] == 'direct':
        return ()
    vector_bytes = _COMPLEX_BYTES * float(unknowns) * sweep['basis']
    return (TimeItem('reduced model basis and projections', 'solve', 1.0, 'bounded',
                     (HostStream(_REDUCED_BUILD_PASSES * vector_bytes), PythonCalls(200))),
            TimeItem('reduced model fields (V y)', 'frequency', sweep['field_points'], 'bounded',
                     (HostStream(vector_bytes), PythonCalls(20))))


# ---- scalar harmonic FEM on triangles (fem.harmonic ScalarHarmonicSystem / HarmonicSystemBase): op counts of the host
# build (lagrange_dofs, fem.assembly element blocks and COO -> CSR, pattern_sum, HarmonicPencil.of), the DevicePencil
# set-up and the per-frequency solve bookkeeping, counted from the code; nothing is fitted to seconds. Host bytes are
# read + written (host_stream passes); a numpy call on small arrays costs the host_stream latency (its smallest copy); a
# stacked small matmul makes one BLAS call per matrix, priced at python_call (a lower bound: the profile has no
# small-matrix gemm curve).
# element_geometry + weights per element and rule point, float64 values (elementwise since profiler S2): points sum_k
# L_k x_k (3 products + 2 adds over 2 components, 2 operands + 1 out each: 30), the per-element affine J / det /
# inverse copied to the point (4 + 1 + 4), folded check 1, weights 2 (axisymmetric x r: + 3)
_GEOMETRY_VALUES_PER_POINT = 42
# _tensor_blocks per rule point and local dof (profiler S2: no per-point matmul): physical gradients and tensor @ trial
# written per component, 2 columns x (product, product, add) ~ 7 values each, both passes; complex doubles the bytes
_COMPONENT_BYTES_PER_LOCAL = 2 * 2 * 7 * 8
_AXISYMMETRIC_WEIGHT_VALUES = 3
# fem.pml coefficient at the rule points, bytes per point: stretches, 1 / s, J = s_x s_y, the (2, 2) tensor factor and
# its product with the coefficient (stiffness); J and the product (mass); an axisymmetric layer's r~ / r on top
_PML_TENSOR_BYTES_PER_POINT = 600
_PML_VOLUME_BYTES_PER_POINT = 170
_PML_RADIAL_BYTES_PER_POINT = 230
# one sort pass per key: numpy's quicksort argsort (vectorised) streams key + permutation (a stable argsort would read
# the key through the permutation, a 64-byte cache line per random read)
SORT_PASS_BYTES = 24
_CHUNK_NUMPY_CALLS = 40      # one element-block chunk: geometry, basis, blocks, rows / columns
_PML_CHUNK_NUMPY_CALLS = 30  # the stretch layers, factors and products of a PML coefficient per chunk
_CSR_NUMPY_CALLS = 12        # _to_csr / pattern_sum: concatenations, coo checks, tocsr, sum_duplicates
_DOF_MAP_NUMPY_CALLS = 15
_PENCIL_NUMPY_CALLS = 40     # HarmonicPencil.of: canonical checks, aligned copies, the free / coupling blocks
_SOLVE_NUMPY_CALLS = 30      # HarmonicSystemBase.solve around the factor: loads, rhs, scatters, store
_SOLVE_BYTES_PER_DOF = 256   # load_vector (zeros, frequency load gather, boundary sum), rhs gather, solution and values scatters
_HOST_REDUCED_BYTES = 128    # HarmonicPencil._data per stored entry beside its term gathers: K + j omega C - omega^2 M
_INT32 = 4
# R6: a direct cuDSS solve of HarmonicSystemBase keeps its solution in the pool for _publish: complex128 full-numbering
# values + int64 free / fixed and system ids, bytes per full-numbering dof (a memory run item of the physics)
KEPT_SOLUTION_BYTES_PER_DOF = 32


def numpy_calls(count) -> tuple:
    """(multiplicity, op) of `count` numpy calls on small arrays, each at the host_stream latency."""
    from ..core.time_estimate import HostStream
    return float(count), HostStream(1.0)


def dof_map_bytes(local_ids, order) -> float:
    """Host bytes of one fem.dofs.lagrange_dofs over `local_ids` element-local ids: the P1 / P2 id gathers, np.unique
    with its inverse (a sort of log2(ids) passes, run mask, cumsum, inverse scatter) and the int64 element_dofs copy."""
    ids = max(float(local_ids), 2.0)
    return ids * ({0: 0.0, 1: 20.0, 2: 36.0}[order] + 132.0 + SORT_PASS_BYTES * math.log2(ids))


def csr_build_bytes(coo_entries, value_bytes, entries_per_row) -> float:
    """Host bytes of fem.assembly._to_csr of `coo_entries` COO entries (int64 rows / columns), entries_per_row of them
    per CSR row: concatenation, scipy's index check and int32 copy, coo_tocsr, the per-row index sort (2 + log2 passes
    over index + value) and the duplicate sum."""
    sort_and_sum_passes = 3.0 + math.log2(max(float(entries_per_row), 2.0))
    return float(coo_entries) * (2.0 * (value_bytes + 16.0) + 48.0 + 16.0 + 2.0 * value_bytes + 4.0
                                 + 2.0 * (_INT32 + value_bytes) * sort_and_sum_passes)


def _pattern_sum_bytes(stored_entries, value_bytes, parts) -> float:
    """Host bytes of HarmonicSystemBase.assemble's `summed` of `parts` canonical CSR terms of stored_entries each: one
    term is a CSR copy (values + indices; HarmonicPencil.of's canonical check reads the indices); several go through
    fem.assembly.pattern_sum: tocoo row expansion, concatenation, coo checks, coo_tocsr, the row sort and duplicate sum."""
    if parts <= 1:
        return float(stored_entries) * (2.0 * (value_bytes + _INT32) + _INT32)
    per_entry = 4.0 * value_bytes + 48.0 + 2.0 * (_INT32 + value_bytes) * (3.0 + math.log2(parts))
    return float(stored_entries) * parts * per_entry


def element_term_ops(elements, dofs, order, degree, kind, value_bytes, pml, axisymmetric) -> tuple:
    """Ops of one fem.assembly assemble_stiffness / assemble_mass (kind) on `elements` P1 / P2 triangles at a rule of
    `degree`: geometry, the PML coefficient, the element blocks, rows / columns and COO -> CSR (notes above)."""
    from ..core.time_estimate import DenseFlops, HostStream, PythonCalls
    from ..fem.assembly import CHUNK_ELEMENTS
    from ..fem.quadrature import TRIANGLE_DEGREES, triangle_rule
    elements = float(elements)
    local = 3.0 if order == 1 else 6.0
    points = elements * len(triangle_rule(min(max(int(degree), 2), TRIANGLE_DEGREES[-1])).weights)
    coo = elements * local * local
    complex_values = value_bytes >= 16
    stream = 8.0 * (_GEOMETRY_VALUES_PER_POINT + (_AXISYMMETRIC_WEIGHT_VALUES if axisymmetric else 0)) * points
    if pml:
        stream += points * ((_PML_TENSOR_BYTES_PER_POINT if kind == 'stiffness' else _PML_VOLUME_BYTES_PER_POINT)
                            + (_PML_RADIAL_BYTES_PER_POINT if axisymmetric else 0.0))
    complex_flops = 4.0 if complex_values else 1.0
    if kind == 'stiffness':   # _tensor_blocks: gradients, complex casts, tensor products, weighted test, block matmul
        stream += points * ((272.0 * local + 104.0) if complex_values else (112.0 * local + 72.0))
        stream += points * local * _COMPONENT_BYTES_PER_LOCAL * (2.0 if complex_values else 1.0)
        blas_calls = elements   # the block matmul per element (gradients / tensor products are elementwise passes)
        flops = complex_flops * 4.0 * local * local * points
    else:   # (m, q) weights @ (q, n n) basis products: one gemm per chunk
        stream += points * (8.0 + 3.0 * value_bytes)
        blas_calls = 0.0
        flops = complex_flops * 2.0 * local * local * points
    stream += coo * (value_bytes + 32.0) + elements * local * 32.0   # blocks out, rows / columns, element dof gathers
    stream += csr_build_bytes(coo, value_bytes, coo / max(float(dofs), 1.0))
    chunks = math.ceil(elements / CHUNK_ELEMENTS)
    calls = chunks * (_CHUNK_NUMPY_CALLS + (_PML_CHUNK_NUMPY_CALLS if pml else 0)) + _CSR_NUMPY_CALLS
    return HostStream(stream), PythonCalls(blas_calls), DenseFlops(flops, host=True), numpy_calls(calls)


def scalar_harmonic_build_items(mesh, order, *, elements, dofs, stored_entries, axisymmetric=False, pml=False,
                                complex_stiffness=False, complex_mass=False, mass_terms=1, extra_stiffness_parts=0,
                                counter=(), seconds_counter=()) -> tuple:
    """TimeItems of one fem.harmonic ScalarHarmonicSystem build on triangles (its __init__ dof maps + assemble) of
    `elements` active elements, `dofs` system dofs and stored_entries CSR entries: lagrange_dofs (full P1 / P2, P0, the
    active set), stiffness and mass_terms mass assemblies at the _assemble_terms rules (+ PML_EXTRA_DEGREE with a PML),
    the pattern sums (extra parts: edge / plate / port terms in K) and HarmonicPencil.of (notes above)."""
    from ..core.time_estimate import HostStream, TimeItem
    from ..fem.harmonic import PML_EXTRA_DEGREE
    axis, extra = int(axisymmetric), (PML_EXTRA_DEGREE if pml else 0)
    local = 3 if order == 1 else 6
    stiffness_bytes, mass_bytes = 16 if complex_stiffness or pml else 8, 16 if complex_mass or pml else 8
    maps = (dof_map_bytes(int(mesh.n_elements) * local, order) + dof_map_bytes(int(mesh.n_elements), 0)
            + dof_map_bytes(float(elements) * local, order))
    term = dict(elements=elements, dofs=dofs, order=order, pml=pml, axisymmetric=axisymmetric)
    pencil = (_pattern_sum_bytes(stored_entries, stiffness_bytes, 1 + int(extra_stiffness_parts))
              + mass_terms * _pattern_sum_bytes(stored_entries, mass_bytes, 1)
              + float(stored_entries) * (124.0 + 2.0 * (stiffness_bytes + mass_terms * mass_bytes) + 8.0 * mass_terms))
    items = [TimeItem('harmonic dof maps', 'setup', 1.0, 'exact',
                      (HostStream(maps), numpy_calls(3 * _DOF_MAP_NUMPY_CALLS)), seconds_counter=seconds_counter),
             TimeItem('stiffness element assembly', 'assembly', 1.0, 'exact',
                      element_term_ops(degree=max(2 * order - 2 + axis, 2) + extra, kind='stiffness',
                                       value_bytes=stiffness_bytes, **term),
                      counter=counter, seconds_counter=seconds_counter)]
    if mass_terms:
        items.append(TimeItem('mass element assembly', 'assembly', float(mass_terms), 'exact',
                              element_term_ops(degree=2 * order + axis + extra, kind='mass', value_bytes=mass_bytes,
                                               **term), seconds_counter=seconds_counter))
    items.append(TimeItem('pattern sums and harmonic pencil', 'assembly', 1.0, 'exact',
                          (HostStream(pencil), numpy_calls((2 + mass_terms) * _CSR_NUMPY_CALLS + _PENCIL_NUMPY_CALLS)),
                          seconds_counter=seconds_counter))
    return tuple(items)


# ---- scalar harmonic FEM on tetrahedra (fem.harmonic3d ScalarHarmonicSystem3D, S4): element-constant coefficients by
# fem.assembly3d's affine reference forms (geometry once per element, the 3 x 3 congruence |det J| J^-1 C J^-T, one
# (m, 9) @ (9, n n) gemm; mass c |det J| x the reference mass), the PML shell elements by quadrature at the symmetric
# rules (P2 stiffness 14, mass 24 points), COO -> CSR per term, the restriction of each term to the active dofs, then
# the 2D pattern sums / HarmonicPencil.of. Counted from the code like the triangle items; constants UNCALIBRATED.
_GEOMETRY_3D_BYTES_PER_ELEMENT = 900   # tetrahedron_geometry: corners, Jacobian, det / inv (stacked LAPACK), check
_CONGRUENCE_3D_FLOPS = 162             # |det J| J^-1 C J^-T: 81 multiply-adds per element
_PML_TENSOR_3D_BYTES_PER_POINT = 900   # stretches, 1 / s, J, the (3, 3) factor and its product with the coefficient
_PML_VOLUME_3D_BYTES_PER_POINT = 250   # J and its product with the coefficient (mass)
_RESTRICT_PASSES = 2                   # ScalarHarmonicSystem3D restrict: scipy row, then column fancy indexing


def shell_element_fraction(mesh, faces, thickness) -> float:
    """Share of a tetrahedra mesh estimate's elements in a PML shell of `thickness` [m] on `faces` (1 - the window's
    share of the meshed extent's volume, shell included in the extent)."""
    extent = getattr(mesh, 'extent', None)
    if not faces or extent is None or len(extent) != 6:
        return 0.0
    inner = 1.0
    for axis, name in enumerate('xyz'):
        length = float(extent[2 * axis + 1]) - float(extent[2 * axis])
        if length <= 0.0:
            return 0.0
        layers = sum(float(thickness) for side in ('lo', 'hi') if f'{name}_{side}' in faces)
        inner *= max(length - layers, 0.0) / length
    return min(max(1.0 - inner, 0.0), 1.0)


def tetrahedron_term_ops(elements, stretched, dofs, order, degree, kind, value_bytes) -> tuple:
    """Ops of one fem.assembly3d Lagrange stiffness / mass (kind) over `elements` P1 / P2 tetrahedra: `stretched` of
    them (the PML shell) by quadrature at tetrahedron_rule(degree), the rest by the affine reference forms; the blocks'
    concatenation (and reorder of a split), rows / columns and COO -> CSR (notes above)."""
    from ..core.time_estimate import DenseFlops, HostStream, PythonCalls
    from ..fem.assembly3d import CHUNK_ELEMENTS
    from ..fem.quadrature3d import TETRAHEDRON_MAX_DEGREE, tetrahedron_rule
    elements = float(elements)
    stretched = min(max(float(stretched), 0.0), elements)
    affine = elements - stretched
    local = 4.0 if order == 1 else 10.0
    square = local * local
    complex_values = value_bytes >= 16
    complex_flops = 4.0 if complex_values else 1.0
    points = stretched * len(tetrahedron_rule(min(max(int(degree), 1), TETRAHEDRON_MAX_DEGREE)).weights)
    stream = _GEOMETRY_3D_BYTES_PER_ELEMENT * elements + 24.0 * points   # + the shell's rule points
    if kind == 'stiffness':   # affine: tensor, congruence, (m, 9) values; shell: as element_term_ops with 3 components
        stream += affine * (72.0 + 3.0 * 9.0 * value_bytes)
        stream += points * (_PML_TENSOR_3D_BYTES_PER_POINT
                            + ((408.0 * local + 156.0) if complex_values else (168.0 * local + 108.0)))
        blas_calls = 2.0 * points + stretched
        flops = complex_flops * ((_CONGRUENCE_3D_FLOPS + 18.0 * square) * affine + 6.0 * square * points)
    else:   # affine: c |det J| x the reference mass; shell: (m, q) weights @ (q, n n)
        stream += affine * 16.0 + points * (_PML_VOLUME_3D_BYTES_PER_POINT + 8.0 + 3.0 * value_bytes)
        blas_calls = 0.0
        flops = complex_flops * (square * affine + 2.0 * square * points)
    coo = elements * square
    stream += coo * value_bytes * (3.0 if stretched else 2.0)   # blocks written and concatenated (+ the split reorder)
    stream += coo * 32.0 + csr_build_bytes(coo, value_bytes, coo / max(float(dofs), 1.0))   # rows / columns, CSR
    chunks = math.ceil(affine / CHUNK_ELEMENTS) + math.ceil(stretched / CHUNK_ELEMENTS)
    calls = chunks * _CHUNK_NUMPY_CALLS + math.ceil(stretched / CHUNK_ELEMENTS) * _PML_CHUNK_NUMPY_CALLS
    return HostStream(stream), PythonCalls(blas_calls), DenseFlops(flops, host=True), numpy_calls(calls + _CSR_NUMPY_CALLS)


def scalar_harmonic_build_items_3d(mesh, order, *, elements, dofs, stored_entries, pml_elements=0.0, mass_terms=1,
                                   extra_stiffness_parts=0, counter=(), seconds_counter=()) -> tuple:
    """TimeItems of one fem.harmonic3d ScalarHarmonicSystem3D build (the full-mesh dof map + assemble) of `elements`
    active tetrahedra, pml_elements of them in a PML shell, `dofs` system dofs and stored_entries CSR entries: the dof
    maps, stiffness and mass_terms mass assemblies (tetrahedron_term_ops at the _assemble_terms rules, + the PML extra
    degree with a shell), the restriction of every term, the pattern sums and HarmonicPencil.of (notes above)."""
    from ..core.time_estimate import HostStream, TimeItem
    from ..fem.harmonic3d import PML_EXTRA_DEGREE_3D
    pml = float(pml_elements) > 0.0
    extra = PML_EXTRA_DEGREE_3D if pml else 0
    local = 4 if order == 1 else 10
    value_bytes = 16 if pml else 8
    maps = dof_map_bytes(float(elements) * local, order) + 24.0 * float(elements) * local
    term = dict(elements=elements, stretched=pml_elements, dofs=dofs, order=order, value_bytes=value_bytes)
    terms = 1 + int(mass_terms) + int(extra_stiffness_parts)
    restricted = terms * _RESTRICT_PASSES * float(stored_entries) * (2.0 * (value_bytes + _INT32) + _INT32)
    pencil = (_pattern_sum_bytes(stored_entries, value_bytes, 1 + int(extra_stiffness_parts))
              + mass_terms * _pattern_sum_bytes(stored_entries, value_bytes, 1)
              + float(stored_entries) * (124.0 + 2.0 * (1 + mass_terms) * value_bytes + 8.0 * mass_terms))
    items = [TimeItem('harmonic dof maps (tetrahedra)', 'setup', 1.0, 'exact',
                      (HostStream(maps), numpy_calls(2 * _DOF_MAP_NUMPY_CALLS)), seconds_counter=seconds_counter),
             TimeItem('stiffness element assembly (tetrahedra)', 'assembly', 1.0, 'exact',
                      tetrahedron_term_ops(degree=max(2 * order - 2, 1) + extra, kind='stiffness', **term),
                      counter=counter, seconds_counter=seconds_counter)]
    if mass_terms:
        items.append(TimeItem('mass element assembly (tetrahedra)', 'assembly', float(mass_terms), 'exact',
                              tetrahedron_term_ops(degree=2 * order + extra, kind='mass', **term),
                              seconds_counter=seconds_counter))
    items.append(TimeItem('term restriction, pattern sums and harmonic pencil (tetrahedra)', 'assembly', 1.0, 'exact',
                          (HostStream(restricted + pencil),
                           numpy_calls((2 + mass_terms) * _CSR_NUMPY_CALLS + _PENCIL_NUMPY_CALLS)),
                          seconds_counter=seconds_counter))
    return tuple(items)


def harmonic_frequency_items(dofs, stored_entries, sweep, *, device_pencil, term_value_bytes,
                             frequency_terms=False) -> tuple:
    """Host work of fem.harmonic HarmonicSystemBase around the 'linear' role of a frequency_response study (sweep =
    frequency_sweep_counts): on a direct sweep per frequency the loads, rhs and solution scatters of solve; on a cuDSS
    back end (device_pencil) the DevicePencil set-up once (term gathers and nonzero checks, the upload, the device
    symmetry check: transposed-key argsort, gathers, max reductions), else HarmonicPencil.reduced per factorisation
    (frequency_terms: + the pattern_sum with the per-frequency block). term_value_bytes = value bytes of the stored
    pencil terms (K, mass; the zero damping is not uploaded)."""
    from ..core.time_estimate import Elementwise, HostStream, PythonCalls, Sync, TimeItem, Transfer
    stored = float(stored_entries)
    factorizations, kind = sweep['factorizations'], sweep['factorizations_kind']
    items = []
    if sweep['method'] == 'direct':
        items.append(TimeItem('harmonic solve bookkeeping (per frequency)', 'solve', sweep['solves'],
                              sweep['solves_kind'],
                              (HostStream(_SOLVE_BYTES_PER_DOF * float(dofs)), numpy_calls(_SOLVE_NUMPY_CALLS))))
    terms = len(term_value_bytes)
    if device_pencil:
        host = stored * (sum(8.0 + 2.0 * size for size in term_value_bytes) + 18.0 + 20.0)
        device_reads = stored * (160.0 + sum(3.0 * size + 8.0 for size in term_value_bytes))
        device_writes = stored * (136.0 + sum(term_value_bytes))
        items.append(TimeItem('device pencil set-up (term upload, symmetry check)', 'setup', 1.0, 'exact',
                              (HostStream(host), Transfer(stored * (sum(term_value_bytes) + 16.0)),
                               Elementwise(device_reads, device_writes, launches=20 + 6 * terms),
                               Sync(2 + 2 * terms), PythonCalls(30), numpy_calls(10))))
    else:   # damping gathered too (zeros)
        reduced = stored * (sum(8.0 + 2.0 * size for size in term_value_bytes) + 24.0 + _HOST_REDUCED_BYTES)
        if frequency_terms:
            reduced += _pattern_sum_bytes(stored, 16, 2) / 2.0   # the frequency block's own entries are few
        items.append(TimeItem('reduced operator K + j omega C - omega^2 M (per factorisation)', 'solve',
                              factorizations, kind, (HostStream(reduced), numpy_calls(12))))
    return tuple(items)


def optional_material_param(name, label, unit, expands_to, minimum=None, help_text='', tip=''):
    """Field- / state-dependent material property Param (default None: unset properties are never written)."""
    return Param(name, 'float', None, label=label, unit=unit, min=minimum, help=help_text, tip=tip, optional=True,
                 field_dependent=True, state_dependent=True, expands_to=expands_to)


def uses_time(value) -> bool:
    """True for a resolved feature value that is an expression of time t (a time_dependent param; BuildContext
    follow_time re-sets it before every step)."""
    return isinstance(value, Expression) and not value.names.isdisjoint(TIME_NAMES)


def time_following_values(physics_spec, feature_types=None) -> int:
    """Feature param values of an enabled feature (of feature_types, None = any) that are text expressions of time t:
    the values follow_time re-sets before every step (run-time estimates; bad expressions are not counted)."""
    count = 0
    for feature in getattr(physics_spec, 'features', ()):
        if not feature.enabled or (feature_types is not None and feature.type not in feature_types):
            continue
        for value in feature.params.values():
            if not isinstance(value, str):
                continue
            try:
                names = referenced_names(value)
            except ExpressionError:
                continue
            count += not names.isdisjoint(TIME_NAMES)
    return count


def make_apply_inward_flux(param_name):
    """apply of an inward-flux boundary feature: Neumann on outer faces; object perimeters get value * perimeter
    length / cell area (with solver_defaults.conformal_boundaries the drawn outline length of each cut cell). A
    time_dependent value of t is re-set per step (ctx.follow_time: face values in place, perimeters by the change).
    On a triangle mesh the object outlines carry the flux as an edge (line) source (solver.add_edge_flux), on a
    tetrahedron mesh the object interfaces as a face (sheet) source (solver.add_surface_flux)."""
    def apply_inward_flux(ctx, solver, feature_spec, values, selection):
        faces = list(selection.outer_faces)
        if ctx.domain.grid_kind in ('triangles', 'tetrahedra'):
            return _apply_inward_flux_elements(ctx, solver, feature_spec, values[param_name], selection)
        perimeter = None   # (weights, outline) of the selected object perimeters
        if selection.objects:
            outline = outline_pieces_of(ctx, solver, selection.objects)
            if outline is None:
                perimeter = (perimeter_flux_weights(ctx, solver, selection.objects), False)
            else:
                pieces, scale = outline
                perimeter = (outline_full_grid(solver.system, pieces, pieces.length * scale, per_volume=True), True)
        applied = [0.0]   # perimeter flux already in the applied source (a later value adds the difference)

        def set_inward_flux(inward_flux):
            for face in faces:
                solver.set_face_value(face, BCSide('neumann', value=inward_flux))
            if perimeter is not None and inward_flux != applied[0]:
                weights, outline = perimeter
                solver.add_boundary_flux(weights, inward_flux - applied[0], outline=outline)
                applied[0] = inward_flux

        ctx.follow_time(values[param_name], set_inward_flux, solver, f"feature '{feature_spec.name}' {param_name}")
        return None
    return apply_inward_flux


def _apply_inward_flux_elements(ctx, solver, feature_spec, value, selection):
    """Triangle / tetrahedron-mesh inward flux: Neumann on the window faces, an edge / face source on the object
    outlines / interfaces (both re-set per step for a value of t)."""
    handle = []   # the outline condition, made at the first value
    tetrahedra = ctx.domain.grid_kind == 'tetrahedra'

    def set_inward_flux(inward_flux):
        for face in selection.outer_faces:
            solver.set_face_value(face, BCSide('neumann', value=inward_flux))
        if not selection.objects:
            return
        if handle:
            (solver.set_surface_flux if tetrahedra else solver.set_edge_flux)(handle[0], inward_flux)
        elif tetrahedra:
            handle.append(solver.add_surface_flux(ctx.boundary_faces(selection.objects), inward_flux))
        else:
            handle.append(solver.add_edge_flux(ctx.boundary_edges(selection.objects), inward_flux))

    ctx.follow_time(value, set_inward_flux, solver, f"feature '{feature_spec.name}' inward flux")
    return None


def outline_pieces_of(ctx, solver, objects):
    """(core.cut_cells OutlinePieces of the objects' drawn outline, per-piece length factor: ring_angle x the piece's
    distance to the rotation axis on an axisymmetric (r, z) or spherical system, else 1) when
    solver_defaults.conformal_boundaries reached the solver (its conformal_locator, which takes the Domain's logical
    coordinates) on a 2D DiffusionSystem (Cartesian, polar or spherical); None otherwise (staircased perimeter weights)."""
    locator = getattr(solver, 'conformal_locator', None)
    system = solver.system
    if locator is None or system.metric is None:
        return None
    from ..core.backend import to_host
    from ..core.cut_cells import outline_pieces
    coordinates = system.metric.coordinates
    region = np.asarray(to_host(ctx.domain_mask(objects)[system.window]), dtype=bool)
    x_edges, y_edges = (np.asarray(edges, dtype=np.float64)[window.start:window.stop + 1]
                        for edges, window in zip(system.domain.cell_edges(), system.window))
    pieces = outline_pieces(region, x_edges, y_edges, locator.inside(objects),
                            radius_min=coordinates.radius_min if coordinates.polar else None)
    if not system.axisymmetric:
        factor = np.ones(pieces.length.size)
    elif coordinates.polar:   # spherical (r, theta): rho = r sin theta
        factor = system.ring_angle * ((coordinates.radius_min + pieces.point_x)
                                      * np.sin(coordinates.angle_min + pieces.point_y))
    else:   # (r, z): r from the x_lo window edge
        factor = system.ring_angle * (pieces.point_x - x_edges[0])
    return pieces, factor


def outline_full_grid(system, pieces, per_piece, per_volume=False):
    """Full-grid float32 run-backend array of per-piece values summed into their owner cells (0 elsewhere), divided by
    the geometric cell volumes (metric) with per_volume (outline flux weights [1/m])."""
    from ..core.cut_cells import outline_cell_sums
    sums = outline_cell_sums(pieces, per_piece, system.window_shape)
    if per_volume:
        sums = sums / system.metric.volume.full()
    full = system.xp.zeros(system.grid_shape, dtype=np.float32)
    full[system.window] = system.xp.asarray(sums, dtype=np.float32)
    return full


def perimeter_flux_weights(ctx, solver, objects):
    """Perimeter face area / cell volume [1/m] of the objects' union (ctx.perimeter_weights on the plain (x, y) grid);
    on another metric each exposed face is weighted by its area over the cell volume (the solver multiplies by the
    metric volume: (r, z) x faces r_face / r_c; polar / spherical (symmetry 2.16 item 3) also the y faces ~1 / r_c and
    the normal estimate on the physical widths r_c d_angle)."""
    system = solver.system
    metric = system.metric   # None in 3D
    if metric is None or metric.cartesian_planar:
        return ctx.perimeter_weights(objects)
    xp = system.xp   # the run backend's array module
    polar = metric.coordinates.polar
    weights = xp.zeros(system.grid_shape, dtype=np.float32)
    weights[system.window] = perimeter_weights(
        ctx.domain_mask(objects)[system.window], system.dx_window, system.dy_window, xp,
        x_face_scales=metric.x_face_volume_ratios(),
        y_face_scales=metric.y_face_volume_ratios(system.dy_window) if polar else None,
        y_width_scales=metric.y_length.along_x if polar else None)
    return weights


def make_point_probe(quantity, unit):
    """(apply, result) callbacks of a one-point DiffusionSystem probe feature reading `quantity` in `unit` (a triangle
    mesh: the point's mesh node; tetrahedra: the field evaluated at the point)."""
    def apply_probe(ctx, solver, feature_spec, values, selection):
        if ctx.domain.grid_kind in ('triangles', 'tetrahedra'):
            if len(selection.objects) != 1:
                raise ValueError(f"{quantity} probe '{feature_spec.name}' needs exactly one point object, "
                                 f"got {len(selection.objects)}")
            if ctx.domain.grid_kind == 'tetrahedra':
                return solver.add_probe_at(ctx.point_position(selection.objects[0]))
            return solver.add_probe_node(ctx.point_node(selection.objects[0]))
        cells = point_cells(ctx, feature_spec, selection)
        if len(cells) != 1:
            raise ValueError(f"{quantity} probe '{feature_spec.name}' needs exactly one point object, "
                             f"got {len(cells)}")
        return solver.add_probe(*cells[0])   # (ix, iy), 3D (ix, iy, iz)

    def probe_result_callback(session, solver, probe, feature_spec, options):
        return probe_result(probe, unit=unit)
    return apply_probe, probe_result_callback


def reports_model_checks(vctx, physics_spec, type_name):
    """True when physics_spec is the first physics of type_name: it reports the model-wide material checks."""
    same_type = [physics for physics in vctx.spec.physics if physics.type == type_name]
    return not same_type or same_type[0].name == physics_spec.name


def conducts(properties, parameters, prefix):
    """True when <prefix>_xx or <prefix>_yy (component, else isotropic <prefix>) may be > 0."""
    isotropic = properties.get(prefix)
    return any(evaluates_to(properties.get(component, isotropic), parameters, lambda value: value > 0.0)
               for component in (f'{prefix}_xx', f'{prefix}_yy'))


def conducting_fraction(vctx, conducting):
    """Area fraction of the preview pixels inside the window whose material conducts (conducting: one bool per spec
    material); polar / spherical: pixels inside the sector only. None without a preview or inside pixel."""
    preview = vctx.material_map()
    if preview is None or not preview[0].size:
        return None
    index_image, _, extent = preview
    conducts_by_index = np.array([False] + list(conducting), dtype=bool)   # index 0 = no material
    conducting_pixels = conducts_by_index[index_image]
    window_map = vctx.domain_map
    if window_map is not None and window_map.mapped:   # the image covers the sector's drawing-plane box
        x_centres, y_centres = preview_pixel_centers(extent, index_image.shape)
        conducting_pixels = conducting_pixels[window_map.contains(x_centres[:, None], y_centres[None, :])]
    return float(conducting_pixels.mean()) if conducting_pixels.size else None


def selected_material_data(spec, unresolved=False) -> list:
    """[(MaterialSpec, merged properties, poles)] of the spec materials with a selection, library / crystal data merged
    as the builder does; a material whose data does not resolve (reported at the material) is left out, or listed with
    properties and poles None when `unresolved`."""
    from ..materials import effective_material_data   # lazy: materials imports physics.em
    selected = []
    for material in spec.materials:
        if not material.selection:
            continue
        try:
            properties, poles = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:
            if unresolved:
                selected.append((material, None, None))
            continue
        selected.append((material, properties, poles))
    return selected


def selected_material_sets(vctx, names) -> bool:
    """True when a selected material may set a non-zero value of one of `names` (field-dependent values and materials
    whose data does not resolve count)."""
    return any(properties is None or any(evaluates_to(properties.get(name), vctx.parameters, lambda value: value != 0.0)
                                         for name in names)
               for _, properties, _ in selected_material_data(vctx.spec, unresolved=True))


def cross_term_materials(vctx, prefix):
    """Names of the selected spec materials whose <prefix>_xy may be != 0 (as the builder merges them)."""
    return [material.name for material, properties, _ in selected_material_data(vctx.spec)
            if evaluates_to(properties.get(f'{prefix}_xy'), vctx.parameters, lambda value: value != 0.0)]


# ---- 'geometry' setting: planar or axisymmetric (r, z) (heat, potential, magnetic) ----
GEOMETRY_SETTING = 'geometry'
AXISYMMETRIC = 'axisymmetric'
AXISYMMETRIC_SPACE_SETTINGS = {AXISYMMETRIC: {GEOMETRY_SETTING: AXISYMMETRIC}}   # PhysicsType.space_settings
_AXIS_CONSEQUENCE = 'it has no area, so this feature does nothing there'


def geometry_param(planar_help, axisymmetric_help, planar_tip, axisymmetric_tip,
                   axis_text='any feature there does nothing; validation warns'):
    """The 'geometry' choice setting (planar / axisymmetric) of a 2D type: the shared (r, z) rules around the type's own
    help / tip sentences (planar_help: what planar solves; axisymmetric_help: its weights, units, refused terms)."""
    return Param(
        GEOMETRY_SETTING, 'choice', 'planar', label='Geometry', choices=('planar', AXISYMMETRIC),
        choice_labels=(('planar', 'Planar (per metre of depth)'), (AXISYMMETRIC, 'Axisymmetric (r, z)')),
        tip=('How the drawing extends into 3D.\n\n'
             f'- Planar: the drawing is a cut through a long object; {planar_tip}\n'
             '- Axisymmetric: the drawing is a half cross-section of a round object, rotated about the left domain '
             f'edge. x is the distance from that axis, y the height. {axisymmetric_tip}\n\n'
             'Do not put a boundary condition on the left edge in axisymmetric mode: it is the axis.'),
        help=(f'planar: {planar_help}. axisymmetric: a body of revolution about the x_lo window edge (x = r >= 0, y = '
              f'z): {axisymmetric_help}. x_lo is the axis (r = 0: symmetry, {axis_text}). Field derived values of the '
              "whole model are weighted by the ring volume 2 pi r dx dy ('integral' a whole-body total, 'mean' a "
              'volume mean; also for fields of other physics). A coupled physics that is planar (its type has no '
              "geometry setting, e.g. EM, or it is set to planar) solves the drawing as a planar cut per unit z "
              'length, so its fields are applied per cell as rings (validation warns); a partner with geometry '
              'axisymmetric too is consistent.'))


def is_axisymmetric(settings_values):
    """True when resolved settings (or None) set geometry 'axisymmetric'."""
    return bool(settings_values) and settings_values.get(GEOMETRY_SETTING) == AXISYMMETRIC


# ---- 'discretization' setting of the types that run on 'triangles' meshes (FEM, mesh plan D5) ----
DISCRETIZATION_SETTING = 'discretization'
DISCRETIZATION_PARAM = Param(
    DISCRETIZATION_SETTING, 'choice', 'quadratic', label='Discretization', choices=('quadratic', 'linear'),
    choice_labels=(('quadratic', 'Quadratic elements'), ('linear', 'Linear elements')),
    tip=('Order of the finite elements on a triangle or tetrahedron mesh; ignored on a rectangular-cell mesh.\n\n'
         '- Quadratic: more accurate for the same mesh, the usual choice.\n'
         '- Linear: cheaper per element, needs a finer mesh.'),
    help=('Lagrange order of the unknown on a triangle or tetrahedron mesh (MeshSpec kind triangles / tetrahedra; '
          'ignored on tensor meshes): quadratic (P2, the industry-standard default) or linear (P1). On triangles '
          'elements with an edge on a circle / ellipse are mapped isoparametrically with quadratic elements (geometry '
          'shape order = the element order), straight with linear ones; tetrahedra are always straight (affine).'))


def element_order(settings_values) -> int:
    """Lagrange order of resolved settings' discretization: 1 linear, else 2 (quadratic, the default)."""
    return 1 if settings_values and settings_values.get(DISCRETIZATION_SETTING) == 'linear' else 2


def mesh_kind_of(vctx, physics_spec):
    """MeshSpec kind the physics runs on ('tensor', 'triangles', 'tetrahedra'), None for an unknown mesh name."""
    mesh_spec = vctx.spec.find_mesh(physics_spec)
    return None if mesh_spec is None else mesh_spec.kind


def on_triangles(vctx, physics_spec) -> bool:
    """True when the physics runs on a MeshSpec kind 'triangles' (False for an unknown mesh name)."""
    return mesh_kind_of(vctx, physics_spec) == 'triangles'


def on_tetrahedra(vctx, physics_spec) -> bool:
    """True when the physics runs on a MeshSpec kind 'tetrahedra' (False for an unknown mesh name)."""
    return mesh_kind_of(vctx, physics_spec) == 'tetrahedra'


# ---- 'azimuthal_order' setting of the axisymmetric DiffusionSystem types (symmetry 2.14; heat, potential) ----
AZIMUTHAL_ORDER_SETTING = 'azimuthal_order'
# one Param object shared by every type that has it (azimuthal_order_issues finds the partners by identity)
AZIMUTHAL_ORDER_PARAM = Param(
    AZIMUTHAL_ORDER_SETTING, 'int', 0, label='Azimuthal order', min=0,
    visible_if=((GEOMETRY_SETTING, (AXISYMMETRIC,)),),
    tip=('How the solution varies around the axis.\n\n'
         '- 0: the same at every angle (a rotationally symmetric load).\n'
         '- 1 or more: the solution is this part of a load that varies around the axis, varying as the cosine of '
         'this number times the angle, for example 1 for a uniform field across the axis.\n\n'
         'With an order above 0 every value you enter and every result is the size of that cosine part, and '
         'totals over the rings take half the weight of order 0.'),
    help=('Azimuthal order m >= 0 of an axisymmetric (r, z) solve (symmetry 2.14). 0: rotationally symmetric. m >= 1: '
          'the order-m part u_m(r, z) cos(m phi) of a load on the body of revolution (m = 1: a uniform transverse '
          'field, V = -E0 r cos(phi)); the operator gains c_xx m^2 / r^2 per cell volume (r_c the cell-centre radius; '
          'the radial coefficient _xx is also taken along phi), which makes u_m vanish on the axis and anchors every '
          'cell with c_xx > 0 (no other anchor needed). Every field, source density, boundary, held and initial value '
          'is then the amplitude of cos(m phi) (heat: T - T0 is the amplitude, T0 drops out), field components are '
          'the in-plane r / z amplitudes (the phi part is not output), and ring totals carry pi instead of 2 pi (the '
          "phi integral of cos^2): held currents / charges, scalar outputs and field 'integral' derived values are "
          'cos(m phi) moments (e.g. the integral of rho cos(m phi) dV), a point source P is the order-m part of a '
          'point source P at phi = 0 (its Fourier sum over m: the m = 0 ring of total P plus every order with the '
          'same P), and a dissipation density (q_joule) is twice its phi mean, so its integral is the total power. '
          "Needs geometry 'axisymmetric' on a Cartesian domain; every physics of the model with this setting shares "
          'one m (derived values weight rings by one angle); couplings of an order-m physics warn (products such as '
          'Joule heating are no single order). solid_mechanics: the mode amplitudes U_r, U_z ~ cos(m phi), U_phi ~ '
          'sin(m phi) (see its description).'))


def azimuthal_order_of(settings_values):
    """azimuthal_order of resolved settings (0 when absent or unresolved)."""
    return int(settings_values.get(AZIMUTHAL_ORDER_SETTING, 0)) if settings_values else 0


def azimuthal_order_issues(vctx, physics_spec, path, settings_values):
    """Rules of an azimuthal_order m != 0 (symmetry 2.14): geometry axisymmetric on a Cartesian domain (errors), one m
    shared by every physics with the setting (error), a warning naming the coupled partners."""
    order = azimuthal_order_of(settings_values)
    if not order:
        return []
    issues = []
    if not is_axisymmetric(settings_values):
        issues.append(Issue('error', path, f"azimuthal_order {order} needs geometry 'axisymmetric' (an order of the "
                                           'angle about the rotation axis); set it to 0 for a planar solve'))
    if vctx.domain_map is not None and vctx.domain_map.mapped:
        issues.append(Issue('error', path, f'azimuthal_order {order} needs a Cartesian (r, z) domain, not the '
                                           f'{vctx.domain_map.grid_text} grid: set it to 0'))
    for other, other_type in known_physics(vctx.spec, vctx.registry):
        if other.name == physics_spec.name:
            continue
        if not any(param is AZIMUTHAL_ORDER_PARAM for param in other_type.settings):
            continue
        other_values = vctx.values(other_type.settings, other.settings)
        if other_values is not None and is_axisymmetric(other_values) and azimuthal_order_of(other_values) != order:
            issues.append(Issue('error', path, f"azimuthal_order {order} differs from that of '{other.name}' "
                                               f'({azimuthal_order_of(other_values)}): field derived values weight the '
                                               'rings of a model by one angle (pi at m != 0, 2 pi at m = 0)'))
    partners = sorted(_coupled_partner_names(vctx, physics_spec))
    if partners:
        issues.append(Issue('warning', path, f"azimuthal_order {order}: the coupling(s) to "
                                             f"{', '.join(repr(name) for name in partners)} exchange order-{order} "
                                             'amplitudes as they are; a product such as Joule heating or a '
                                             'temperature-dependent coefficient is no single azimuthal order'))
    return issues


def _coupled_partner_names(vctx, physics_spec) -> set:
    """Names of the physics joined to physics_spec by an enabled coupling (either direction)."""
    return {coupling.source if coupling.target == physics_spec.name else coupling.target
            for coupling in vctx.spec.couplings
            if coupling.enabled and physics_spec.name in (coupling.source, coupling.target)}


def coupled_planar_partners(vctx, physics_spec):
    """Sorted names of the physics coupled to physics_spec by enabled couplings that are planar: their type has no
    'geometry' setting or it resolves to 'planar' (unknown partners and unresolvable settings are skipped)."""
    partner_names = _coupled_partner_names(vctx, physics_spec)
    physics_by_name = {physics.name: physics for physics in vctx.spec.physics}
    planar = []
    for name in sorted(partner_names):
        partner = physics_by_name.get(name)
        if partner is None:   # reported by the coupling checks
            continue
        try:
            partner_type = vctx.registry.physics_type(partner.type)
        except KeyError:
            continue
        if any(param.name == GEOMETRY_SETTING for param in partner_type.settings):
            partner_values = vctx.values(partner_type.settings, partner.settings)
            if partner_values is None or is_axisymmetric(partner_values):   # None: reported at that physics
                continue
        planar.append(name)
    return planar


def axisymmetric_issues(vctx, physics_spec, path, type_text, cross_prefix=None, axis_severity='warning',
                        axis_consequence=_AXIS_CONSEQUENCE):
    """(r, z) rules of a physics with geometry 'axisymmetric': features selecting x_lo (the axis), selected materials
    with <cross_prefix>_xy != 0 (error), and one warning naming the coupled planar partners. On a spherical domain
    only the partner warning (mapped_domain_issues has the axis faces and cross terms)."""
    issues = []
    mapped = vctx.domain_map is not None and vctx.domain_map.mapped
    for feature, entities in enabled_feature_entities(vctx, physics_spec) if not mapped else ():
        if 'x_lo' in entities:
            issues.append(Issue(axis_severity, feature_path(physics_spec, feature),
                                f'x_lo is the symmetry axis (r = 0) in axisymmetric geometry: {axis_consequence}'))
    if cross_prefix is not None and not mapped:
        for material_name in cross_term_materials(vctx, cross_prefix):
            issues.append(Issue('error', ('material', material_name),
                                f"material '{material_name}' sets {cross_prefix}_xy, which axisymmetric {type_text} "
                                f"'{physics_spec.name}' does not support (no cross term in (r, z)): remove it"))
    planar = coupled_planar_partners(vctx, physics_spec)
    if planar:
        issues.append(Issue('warning', path,
                            f"axisymmetric {type_text} is coupled to {', '.join(repr(name) for name in planar)}, which "
                            "solve(s) the drawing as a planar cut (no geometry 'axisymmetric'): their fields are applied "
                            'per cell as if they were rotationally symmetric too'))
    return issues


def antiperiodic_wrap(vctx) -> bool:
    """True on a polar domain with angle_wrap 'antiperiodic': the sign-flipping loop anchors a DiffusionSystem physics
    (its _has_anchor counts it; the build's anchor check confirms it per conducting region)."""
    window_map = vctx.domain_map
    return window_map is not None and window_map.angle_wrap == 'antiperiodic'


def mapped_domain_issues(vctx, physics_spec, perimeter_flux_types, cross_prefix):
    """Polar / spherical (and 1D) domain rules of a DiffusionSystem physics: no object-perimeter flux features
    (perimeter_flux_types) on 1D models and 3D sectors (2D polar / spherical grids weight them by the metric:
    perimeter_flux_weights, outline_pieces_of), no <cross_prefix>_xy materials, a warning per feature on a face
    without area (the origin, a spherical axis), an error per feature on a phi-wrapped face."""
    window_map = vctx.domain_map
    if window_map is None or not window_map.mapped:
        return []
    grid_text = (f'{window_map.grid_text} model' if window_map.line
                 else 'cylindrical (r, phi, z) sector' if vctx.spec.is_3d
                 else f"{window_map.kind} {'(r, theta)' if window_map.axisymmetric else '(r, phi)'} grid")
    issues = []
    zero_area_faces = window_map.zero_area_faces()
    wrapped_faces = window_map.wrapped_faces()
    for feature, selected in enabled_feature_entities(vctx, physics_spec):   # a bad selection: reported elsewhere
        if ((window_map.line or vctx.spec.is_3d) and feature.type in perimeter_flux_types
                and vctx.objects(feature.selection)):
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f'object-perimeter fluxes are not available on the {grid_text} yet: select outer '
                                'faces only (a Temperature / Voltage on an object or a domain feature still works)'))
        on_wrap = [face for face in wrapped_faces if face in selected]
        if on_wrap:
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f"{' / '.join(on_wrap)} {'is' if len(on_wrap) == 1 else 'are'} joined by the domain's "
                                f"angle_wrap {window_map.angle_wrap!r} (the y_hi rays continue at y_lo): no boundary "
                                "there, so remove the face from the selection or set angle_wrap 'none'"))
        on_faces = [face for face in zero_area_faces if face in selected]
        if on_faces:
            issues.append(Issue('warning', feature_path(physics_spec, feature),
                                f"{', '.join(on_faces)} {'has' if len(on_faces) == 1 else 'have'} no area on the "
                                f'{grid_text} (the origin r = 0 or the rotation axis): {_AXIS_CONSEQUENCE}'))
    for material_name in cross_term_materials(vctx, cross_prefix):
        issues.append(Issue('error', ('material', material_name),
                            f"material '{material_name}' sets {cross_prefix}_xy, which the {grid_text} does not "
                            f"support (no cross term; {cross_prefix}_xx / _yy are the radial / angular components "
                            'there): remove it'))
    return issues


# ---- gas Params of the squeeze_film_damping features (fem.squeeze_film; solid_mechanics_3d, membrane) ----
FILM_GAP_PARAM = Param(
    'gap', 'float', 2e-6, label='Gap height', unit='m', min=0.0, min_exclusive=True,
    help='Height H of the gas gap between the vibrating surface and the fixed wall facing it [m], uniform; the film '
         'equation assumes H much smaller than the footprint (typical MEMS 0.5-10 um). The damping scales as 1 / H^3.',
    tip='Thickness of the thin gas layer between the vibrating surface and the fixed wall below or above it.\n\n'
        'Typical 1 to 10 micrometres.')
GAS_VISCOSITY_PARAM = Param(
    'viscosity', 'float', 1.81e-5, label='Gas viscosity', unit='Pa s', min=0.0, min_exclusive=True,
    help='Dynamic viscosity mu of the gas [Pa s] (air at 20 C: 1.81e-5), independent of pressure; slip reduces it to '
         'mu_eff in the gap.',
    tip='How viscous the gas is. Air at room temperature: 1.81e-5 Pa s.')
AMBIENT_PRESSURE_PARAM = Param(
    'ambient_pressure', 'float', 101325.0, label='Ambient pressure', unit='Pa', min=0.0,
    help='Ambient gas pressure p_a [Pa] (1 atm = 101325): sets the gas compressibility H / p_a of the film; 0 = an '
         'incompressible film (pure damping, no gas stiffness; needs vents). Compressibility matters at squeeze numbers '
         'sigma = 12 mu omega L^2 / (p_a H^2) of order 1 and above (L the distance to a vent): the damping falls and a '
         'gas spring appears. The mean free path is not rescaled with it: set both.',
    tip='Pressure of the surrounding gas.\n\n0 treats the gas as incompressible (damping only, no spring).')
GAS_SLIP_PARAM = Param(
    'slip', 'choice', 'veijola', label='Gas slip', choices=SLIP_MODELS,
    choice_labels=(('veijola', 'Rarefied gas (Veijola slip)'), ('none', 'No slip')),
    help='veijola: the slip-flow viscosity mu_eff = mu / (1 + 9.638 Kn^1.159), Kn = mean_free_path / gap (Veijola et '
         'al. 1995, 0 <= Kn <= 880, about 5 %); none: mu_eff = mu (continuum, Kn << 0.01).',
    tip='Rarefied gas: the gas slips on the walls of a narrow gap or at low pressure, which lowers the damping.\n\n'
        'No slip: the continuum gas of a wide gap.')
MEAN_FREE_PATH_PARAM = Param(
    'mean_free_path', 'float', 68e-9, label='Mean free path', unit='m', min=0.0, visible_if=(('slip', ('veijola',)),),
    help='Molecular mean free path of the gas at the ambient pressure [m] (air: 68e-9 at 101325 Pa, scaling as 1 / p_a: '
         '6.8e-6 at 1013 Pa); used by slip veijola.',
    tip='Average distance a gas molecule travels between collisions; about 68 nanometres for air at 1 atm, longer at '
        'lower pressure.')
SQUEEZE_FILM_GAS_PARAMS = (FILM_GAP_PARAM, GAS_VISCOSITY_PARAM, AMBIENT_PRESSURE_PARAM, GAS_SLIP_PARAM,
                           MEAN_FREE_PATH_PARAM)


def squeeze_film_gas_issues(values, path, closed) -> list:
    """Squeeze-film gas checks of evaluated feature values: error for a closed (no vent) incompressible film, warning
    beyond the Veijola Knudsen range."""
    issues = []
    ambient, gap, free_path = (values.get(name) for name in ('ambient_pressure', 'gap', 'mean_free_path'))
    if closed and isinstance(ambient, (int, float)) and not ambient > 0.0:
        issues.append(Issue('error', path, 'vents none with ambient_pressure 0 is a closed incompressible film: the gas '
                                           'cannot move, the pressure solve is singular; vent the rim or set the '
                                           'ambient pressure'))
    if (values.get('slip') == 'veijola' and isinstance(gap, (int, float)) and gap > 0.0
            and isinstance(free_path, (int, float)) and free_path / gap > VEIJOLA_KNUDSEN_LIMIT):
        issues.append(Issue('warning', path, f'Knudsen number {free_path / gap:.4g} lies beyond the Veijola slip fit '
                                             f'(0 <= Kn <= {VEIJOLA_KNUDSEN_LIMIT:g}): the damping is extrapolated'))
    return issues
