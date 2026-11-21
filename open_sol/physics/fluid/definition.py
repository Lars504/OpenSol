"""Model-layer definitions of the 'laminar_flow' physics type (settings, features, material vocabulary, validation),
plus the template and selftests (fluid_heat_convection: coupling/definitions_fluid.py). Loaded lazily by the registry
(physics/fluid/__init__.py must not import it)."""
import math

from ...core.constants import FACES
from ...core.memory_estimate import BOOL_BYTES, FLOAT32_BYTES, MemoryEstimate, combine, window_cell_count
from ...core.mirror import feature_mirror_faces, mirror_faces
from ...core.monitors import probe_result
from ...core.time_estimate import TimeEstimate, explicit_step_estimate
from ...model.definition_helpers import (HALF_STORAGE_STEPPER_ROLE, evaluates_to, face_conflicts,
                                         mirror_feature_issues, optional_material_param, own_mirror_face_conflicts,
                                         plain_number, point_cells, reports_model_checks, resolved_stepper_precision,
                                         selected_material_data, uses_time, validate_points_inside)
from ...solvers.explicit_kernels import stepper_precision
from ...model.geometry import expand_entities
from ...model.schema import (STATIONARY_STUDY, TIME_DEPENDENT_STUDY, FeatureType, Issue, Param, PhysicsType,
                              coupling_writes, feature_path, solver_backend)
from .lattice import N_DIRECTIONS
from .renderer import FluidRenderer
from .selftests import HEATED_MICROCHANNEL_DESCRIPTION, HEATED_MICROCHANNEL_LABEL, heated_microchannel_spec
from .solver import (FLUID_PROPERTY_ARRAYS, FLUID_SCHEMA, MACH_WARNING, MIN_TAU_OFFSET, PROBE_QUANTITIES,
                     TAU_OFFSET_WARNING, FluidPhysics, march_steps_estimate, steady_march_steps)

FLUID_TYPE_NAME = 'laminar_flow'   # = selftests.FLUID_TYPE_NAME (selftests never import the definition)
# window float32 temporaries beside the lattice of one refresh / dt change, counted from the code (the largest of):
# solver._kinematic_viscosity (float64 window in place + float32 / float64 density: 5), _update_relaxation's per-cell
# rates (two float64 windows: 4) and _store_equilibrium (u^2, c.u, the previous direction, 3 expression temporaries: 6)
_REFRESH_TEMPORARY_FLOATS = 6
_BOUNDARY_FEATURES = frozenset({'wall', 'slip_wall', 'inlet_velocity', 'pressure', 'periodic'})
# core.mirror.MIRROR_FEATURES['laminar_flow']: the slip face kind (the exact even mirror of the lattice) whose outer
# faces are model mirror planes; another face feature on them is an error (own_mirror_face_conflicts), not the
# face_conflicts warning of _BOUNDARY_FEATURES
_SYMMETRY_FEATURE = 'symmetry'
# face kinds of the enabled face features (a face without one is a no-slip wall): the mirror-plane checks and the
# run-time estimate of a steady march (a priori, never fitted): the slowest shear mode the faces allow decays as
# exp(-t / tau), the march stops once the remaining change is below tolerance x peak (ln(1 / tolerance) decay times)
# on two checks in a row
_FACE_FEATURE_KINDS = {'wall': 'wall', 'slip_wall': 'slip', _SYMMETRY_FEATURE: 'slip', 'inlet_velocity': 'inlet',
                       'pressure': 'pressure', 'periodic': 'periodic'}
# what a non-slip flow face on a mirror plane of another physics mirrors to (warning text)
_MIRROR_PLANE_FACE_TEXT = {
    'wall': 'a no-slip wall (also every face without a flow condition), so the unfolded flow has a thin no-slip plate '
            'on the plane',
    'inlet': 'a velocity inlet, so the unfolded flow has a sheet injecting fluid to both sides of the plane',
    'pressure': 'a pressure face, so the unfolded flow has a plane held at that pressure feeding both sides',
}

# ---- settings ----
_SETTINGS = (
    Param('time_mode', 'choice', 'stationary', label='Time dependence', choices=('stationary', 'transient'),
          choice_labels=(('stationary', 'Steady state'), ('transient', 'Transient')),
          tip='How the flow is computed.\n\n'
              '- Steady state: the flow reached after a long time.\n'
              '- Transient: the flow evolves in time from rest.\n\n'
              "Only a 'Time dependent' study runs transient flow; every other study solves the steady state.",
          help='stationary: each solve marches the lattice to a steady state (re-solved only when a material, force '
               'or face changed; dt_limit inf); transient: one or more lattice steps per runner step (dt_limit = '
               'dt_lbm), only in a time_dependent study (other studies solve the steady state). Transient flow cannot '
               'share a time_dependent study with an FDTD wave physics (step far below dt_lbm: validation error), nor '
               'with elastic_waves when the acoustic step gives the least viscous fluid tau < 1/2 + 1e-3 (3 nu dt / '
               'dx^2, e.g. water on 10 um cells; validation error, a warning below 1/2 + 0.02).'),
    Param('relaxation_time', 'float', 1.0, label='Lattice relaxation time', unit='1', min=0.5, min_exclusive=True,
          max=5.0,
          tip='Numerical setting that fixes the time step of the flow solver from the viscosity and the cell size.\n\n'
              'Larger values take larger steps and reach a steady state sooner but allow less flow speed; 1 suits '
              'most models.',
          help='TRT tau+ of the most viscous fluid: dt_lbm = (tau - 1/2) dx^2 / (3 nu_max), nu = viscosity / rho; '
               'other fluids get tau = 1/2 + 3 nu dt_lbm / dx^2. The lattice Mach number sqrt(3) u dt_lbm / dx must '
               'stay small (validation warns above 0.2 for numeric inlet velocities; the run warns above 0.2, fails '
               'above 0.7): lower tau for fast flow, raise it (up to ~2) to reach a steady state in fewer steps. '
               'Steady results do not depend on tau (weakly with conformal_boundaries: interpolated walls).'),
    Param('tolerance', 'float', 1e-6, label='Steady-state tolerance', unit='1', min=0.0, min_exclusive=True,
          tip='How close to the final steady flow each steady solve gets, relative to the peak speed.\n\n'
              'Keep 1e-6. The single-precision solver cannot get much closer: smaller values stop at its rounding '
              'limit instead.',
          help='A steady march stops when the remaining velocity change, extrapolated from the geometric decay between '
               'checks (rate r: change * r / (1 - r)), is <= tolerance * peak lattice speed on two checks in a row, or '
               'when the change stops decreasing below 1e-5 of the peak (float32 noise floor ~1e-7..1e-6: a smaller '
               'tolerance stops there). Error ~ tolerance.'),
    Param('max_steps', 'int', 0, label='Lattice step limit', min=0,
          tip='Largest number of solver steps per steady solve; 0 picks a limit from the grid size.\n\n'
              'A solve that reaches the limit keeps its unfinished flow and logs a warning.',
          help='Per steady march; 0 = auto: 30 decay times L^2 / (pi^2 nu_lat) of the slowest diffusive mode (L = '
               'longest window axis in cells, doubled on an axis with exactly one slip / pressure face such as a '
               'free-surface film, halved on a periodic axis; nu_lat the smallest lattice viscosity), at least 10 of '
               'the largest check intervals (checks start every max(100, 3.5 L) steps, doubling while one sees < 5 % '
               'of a decay). A march hitting the limit keeps its result and logs a warning.'),
)


# ---- material vocabulary ----
_MATERIAL_PROPERTIES = (
    optional_material_param(
        'viscosity', 'Dynamic viscosity', 'Pa s', FLUID_PROPERTY_ARRAYS['viscosity'], 0.0,
        help_text='Dynamic viscosity mu. Cells with viscosity > 0 are fluid (they also need rho); unset or 0 = a solid '
                  'no-slip obstacle. May follow state fields (e.g. water 2.414e-5 * 10**(247.8 / (heat_T - 140)); a '
                  'stationary study iterates). Typical: water 1.0e-3 (20 C), air 1.81e-5, glycerol 1.4, silicone oil '
                  '1e-2.',
        tip='Viscosity of a liquid or gas; it makes the material a fluid in a flow physics. Empty means a solid that '
            'the flow goes around.\n\nTypical: water 1.0e-3, air 1.81e-5.'),
    optional_material_param(
        'rho', 'Density', 'kg/m^3', FLUID_PROPERTY_ARRAYS['rho'], 0.0,
        help_text='Mass density. Heat: transient heat capacity rho*cp (unset = 0: no heat capacity, those cells follow '
                  'their neighbours; validation warns when heat is transient). Laminar flow: required on fluid '
                  'cells (kinematic viscosity viscosity / rho, inertia, pressure scale). Typical: silica 2200, silicon '
                  '2329, water 998, air 1.2.',
        tip='Mass density. Transient heat uses it with the heat capacity; a flow physics needs it in every fluid.\n\n'
            'Typical: silica 2200, silicon 2329, water 998.'),
)


# ---- feature callbacks (worker thread, GPU) ----
def _outer_faces(selection):
    return [face for face in selection.outer_faces if face in FACES]


def _face_value(ctx, solver, feature_spec, faces, value, param_name, set_face):
    """set_face(face, value) on every face: scalars, or full-grid arrays (x, y expressions) whose window boundary row
    the solver takes; a value of t: set_face at t = 0, then solver.set_face_value before every step (follow_time)."""
    if not uses_time(value):
        evaluated = ctx.eval_field(value)
        for face in faces:
            set_face(face, evaluated)
        return
    for face in faces:
        set_face(face, 0.0)   # the face kind; follow_time sets the value of t at once

    def set_faces(number):
        for face in faces:
            solver.set_face_value(face, number)

    ctx.follow_time(value, set_faces, solver, f"feature '{feature_spec.name}' {param_name}")


def _apply_wall(ctx, solver, feature_spec, values, selection):
    _face_value(ctx, solver, feature_spec, _outer_faces(selection), values['tangential_velocity'],
                'tangential_velocity', solver.set_wall)
    return None


def _apply_slip(ctx, solver, feature_spec, values, selection):
    for face in _outer_faces(selection):
        solver.set_slip(face)
    return None


def _apply_inlet(ctx, solver, feature_spec, values, selection):
    _face_value(ctx, solver, feature_spec, _outer_faces(selection), values['velocity'], 'velocity',
                lambda face, velocity: solver.set_velocity_inlet(face, velocity, values['profile']))
    return None


def _apply_pressure(ctx, solver, feature_spec, values, selection):
    _face_value(ctx, solver, feature_spec, _outer_faces(selection), values['pressure'], 'pressure',
                solver.set_pressure)
    return None


def _apply_periodic(ctx, solver, feature_spec, values, selection):
    faces = set(_outer_faces(selection))
    for axis in ('x', 'y'):
        if {f'{axis}_lo', f'{axis}_hi'} <= faces:
            solver.set_periodic(axis)
    return None


def _apply_body_force(ctx, solver, feature_spec, values, selection):
    """f_x / f_y on the selected cells; a component of t changes before every step by the difference."""
    mask = ctx.domain_mask(selection.entities)
    constant = {}
    for component in ('x', 'y'):
        value = values[f'f_{component}']
        if not uses_time(value):
            constant[component] = ctx.eval_field(value, mask=mask)
            continue
        constant[component] = 0.0
        applied = [0.0]   # this component's force already in body_force_*_ext

        def set_force(number, component=component, applied=applied):
            if number != applied[0]:
                change = number - applied[0]
                solver.add_body_force(mask, change if component == 'x' else 0.0, change if component == 'y' else 0.0)
                applied[0] = number

        ctx.follow_time(value, set_force, solver, f"feature '{feature_spec.name}' f_{component}")
    solver.add_body_force(mask, constant['x'], constant['y'])
    return None


def _apply_initial_velocity(ctx, solver, feature_spec, values, selection):
    """Initial u_x, u_y and pressure on the selected cells (applied by the solver at finalize)."""
    mask = ctx.domain_mask(selection.entities)
    solver.set_initial_flow(mask, *(ctx.eval_field(values[name], mask=mask) for name in ('u_x', 'u_y', 'pressure')))
    return None


def _initial_velocity_memory(values, n_selected_cells, mesh):
    """Build peak: the three window input arrays (kept until finalize starts the flow, freed before its equilibrium)
    over the lattice refresh temporaries of finalize beside them (3 + _REFRESH_TEMPORARY_FLOATS x float32)."""
    return MemoryEstimate.from_items(
        (), build_extra=(3 + _REFRESH_TEMPORARY_FLOATS) * FLOAT32_BYTES * window_cell_count(mesh))


def _apply_flow_probe(ctx, solver, feature_spec, values, selection):
    cells = point_cells(ctx, feature_spec, selection)
    if len(cells) != 1:
        raise ValueError(f"flow probe '{feature_spec.name}' needs exactly one point object, got {len(cells)}")
    return solver.add_probe(*cells[0], values['component'])


def _flow_probe_result(session, solver, probe, feature_spec, options):
    return probe_result(probe, unit=PROBE_QUANTITIES[probe.field_component])


# ---- validation (CPU only) ----
def _validate_symmetry(vctx, physics_spec, feature_spec, values):
    """symmetry: the shared mirror-feature rules (definition_helpers.mirror_feature_issues)."""
    return mirror_feature_issues(vctx, physics_spec, feature_spec, _SETTINGS)


def _mirror_plane_issues(vctx, physics_spec, own_mirrors):
    """Flow faces on the model's mirror planes that this physics does not make itself (EM bc symmetric /
    antisymmetric, the symmetry feature of another physics): the even mirror of the flow is a free-slip face there
    (symmetry or slip_wall); a wall / inlet / pressure face mirrors another structure (warning), a periodic pair through
    the plane none (error)."""
    planes = [face for face in mirror_faces(vctx.spec) if face in FACES and face not in own_mirrors]
    if not planes:
        return []
    kinds = _face_kinds_of(vctx.spec, physics_spec)
    path = ('physics', physics_spec.name)
    name = physics_spec.name
    issues = []
    for face in planes:
        kind = kinds[face]
        if kind == 'slip':
            continue
        if kind == 'periodic':
            issues.append(Issue('error', path, f"{face} is a mirror plane of the model, but '{name}' wraps its flow "
                                               'through that face (periodic pair), and no mirror image has flow '
                                               'through the plane: remove the periodic pair or the mirror plane'))
            continue
        issues.append(Issue('warning', path, f"{face} is a mirror plane of the model, where the mirror image of the "
                                             'flow needs a free-slip face (no flow through it, no shear), but '
                                             f"'{name}' has {_MIRROR_PLANE_FACE_TEXT[kind]}. Add {face} to a "
                                             f"symmetry feature of '{name}' for the mirror image of the drawn half"))
    return issues


def _normal_force_issues(vctx, physics_spec):
    """Warning per body_force feature and per coupling into body_force_<axis> along the normal axis of a model mirror
    plane: the flow's even mirror images that force component as the opposite one, so a force along the normal that is
    the same on both sides (gravity, buoyancy from an even temperature) has no mirror symmetry there."""
    axes = sorted({face[0] for face in mirror_faces(vctx.spec) if face in FACES})
    if not axes:
        return []
    consequence = ('so the unfolded model pushes the mirror image the opposite way (the even mirror of the flow '
                   'needs a force odd about the plane): a normal force that is the same on both sides, such as '
                   'gravity or buoyancy, has no mirror symmetry there. Keep that component zero (odd about the plane), '
                   'or draw the full model across the plane')
    issues = []
    body_force = LAMINAR_FLOW.feature_type('body_force')
    for feature in physics_spec.features:
        if not feature.enabled or feature.type != 'body_force':
            continue
        values = vctx.values(body_force.params, feature.params)
        if values is None:   # reported by the param checks
            continue
        normal = [axis for axis in axes if plain_number(values[f'f_{axis}'], vctx.parameters) != 0.0]
        if normal:
            components = ', '.join(f'f_{axis}' for axis in normal)
            issues.append(Issue('warning', feature_path(physics_spec, feature),
                                f'{components} of this body force is normal to a mirror plane of the model '
                                f'({"/".join(normal)} faces), {consequence}'))
    for coupling in vctx.spec.couplings:
        if not coupling.enabled or coupling.target != physics_spec.name:
            continue
        slots = coupling_writes(vctx, coupling)
        normal = [axis for axis in axes if f'body_force_{axis}' in slots]
        if normal:
            components = ', '.join(f'body_force_{axis}' for axis in normal)
            issues.append(Issue('warning', ('coupling', coupling.name),
                                f"coupling '{coupling.name}' writes {components} of '{physics_spec.name}', normal to a "
                                f'mirror plane of the model ({"/".join(normal)} faces), {consequence}'))
    return issues


def _validate_periodic(vctx, physics_spec, feature_spec, values):
    """The selection must be both faces of one axis."""
    try:
        faces = {name for name in expand_entities(vctx.spec, feature_spec.selection) if name in FACES}
    except (ValueError, KeyError, TypeError):   # reported by the selection checks
        return []
    if faces in ({'x_lo', 'x_hi'}, {'y_lo', 'y_hi'}):
        return []
    return [Issue('error', feature_path(physics_spec, feature_spec),
                  f"a periodic pair selects both faces of one axis: ['x_lo', 'x_hi'] or ['y_lo', 'y_hi'], got "
                  f'{sorted(faces)}')]


def _time_mode(vctx, settings_values):
    """'transient' only in a time-dependent study with time_mode transient, else 'stationary'."""
    if vctx.study_type == TIME_DEPENDENT_STUDY and (settings_values or {}).get('time_mode') == 'transient':
        return 'transient'
    return 'stationary'


def fluid_materials(vctx):
    """[(material name, properties)] of the selected materials whose viscosity may be > 0 (library data merged)."""
    return [(material.name, properties) for material, properties, _ in selected_material_data(vctx.spec)
            if evaluates_to(properties.get('viscosity'), vctx.parameters, lambda value: value > 0.0)]


def _mesh_issues(vctx, path):
    """Error unless the window cells are uniform and square."""
    mesh = vctx.mesh
    if mesh is None:
        return []
    sides = mesh.npml_sides
    dx_window = mesh.dx_cells[sides['x_lo']:mesh.nx - sides['x_hi']]
    dy_window = mesh.dy_cells[sides['y_lo']:mesh.ny - sides['y_hi']]
    widths = list(dx_window) + list(dy_window)
    if mesh.uniform and max(widths) <= min(widths) * (1.0 + 1e-6):
        return []
    return [Issue('error', path, f'laminar flow needs a uniform mesh with square cells (the lattice-Boltzmann grid); '
                                 f'the window cells span {min(widths):.4g} .. {max(widths):.4g} m: set the mesh mode '
                                 'to uniform')]


def _kinematic_viscosities(vctx, fluids):
    """[nu = viscosity / rho] [m^2/s] of the fluids whose viscosity and density are plain numbers."""
    kinematic = []
    for _, properties in fluids:
        viscosity = plain_number(properties.get('viscosity'), vctx.parameters)
        density = plain_number(properties.get('rho'), vctx.parameters)
        if viscosity is not None and density:
            kinematic.append(viscosity / density)
    return kinematic


def _window_dx(mesh):
    return float(mesh.dx_cells[mesh.npml_sides['x_lo']])


def _wave_step_issues(vctx, fluids, path):
    """Transient flow beside a wave physics: an EM FDTD step always fails; an elastic_waves step fails when its
    estimated dt gives tau+ - 1/2 = 3 nu dt / dx^2 below MIN_TAU_OFFSET in the least viscous fluid (warns below
    TAU_OFFSET_WARNING); the build's acoustic dt is this estimate or smaller."""
    if getattr(vctx.mesh, 'dt', None):
        return [Issue('error', path, 'transient laminar flow cannot step on the femtosecond time step of a wave physics '
                                     "in the same time-dependent study: use time_mode 'stationary'")]
    kinematic = _kinematic_viscosities(vctx, fluids)
    if vctx.mesh is None or not kinematic:
        return []
    from ..acoustics.definition import estimated_time_step   # lazy: the acoustic dt estimate (never the reverse)
    issues = []
    for physics in vctx.spec.physics:
        dt = estimated_time_step(vctx, physics) if physics.type == 'elastic_waves' else None
        if not dt:
            continue
        offset = 3.0 * min(kinematic) * dt / _window_dx(vctx.mesh) ** 2
        if offset < TAU_OFFSET_WARNING:
            fails = offset < MIN_TAU_OFFSET
            issues.append(Issue('error' if fails else 'warning', path,
                                f"transient laminar flow steps on the acoustic time step of '{physics.name}' (about "
                                f'{dt:.3g} s): the least viscous fluid gets a lattice relaxation time 0.5 + '
                                f'{offset:.3g} ({"needs" if fails else "stable only above about"} 0.5 + '
                                f'{MIN_TAU_OFFSET if fails else TAU_OFFSET_WARNING:g}); use time_mode '
                                "'stationary'"))
    return issues


def _mach_issues(vctx, physics_spec, settings_values, fluids, path):
    """Warning when a numeric inlet velocity reaches a lattice Mach number above MACH_WARNING."""
    if vctx.mesh is None or not settings_values:
        return []
    kinematic = _kinematic_viscosities(vctx, fluids)
    if not kinematic:
        return []
    dx = _window_dx(vctx.mesh)
    dt_lbm = (settings_values['relaxation_time'] - 0.5) * dx * dx / (3.0 * max(kinematic))
    issues = []
    inlet_type = LAMINAR_FLOW.feature_type('inlet_velocity')
    for feature in physics_spec.features:
        if not feature.enabled or feature.type != 'inlet_velocity':
            continue
        values = vctx.values(inlet_type.params, feature.params)
        velocity = None if values is None else plain_number(values['velocity'], vctx.parameters)
        if velocity is None:
            continue
        peak = abs(velocity) * (1.5 if values['profile'] == 'parabolic' else 1.0)
        mach = math.sqrt(3.0) * peak * dt_lbm / dx
        if mach > MACH_WARNING:
            issues.append(Issue('warning', feature_path(physics_spec, feature),
                                f'inlet speed {peak:.4g} m/s is lattice Mach {mach:.3g} > {MACH_WARNING:g} (dt_lbm '
                                f'{dt_lbm:.4g} s, dx {dx:.4g} m): compressibility errors grow as Ma^2; lower '
                                'relaxation_time or refine the mesh'))
    return issues


def _validate_flow(vctx, physics_spec, settings_values):
    """Fluid materials (viscosity with rho), uniform square mesh, time mode, inflow balance and the lattice Mach number."""
    path = ('physics', physics_spec.name)
    issues = []
    time_mode = (settings_values or {}).get('time_mode')
    if time_mode == 'transient' and vctx.study_type != TIME_DEPENDENT_STUDY:
        issues.append(Issue('info', path, "time_mode 'transient' is ignored outside a time-dependent study: the flow is "
                                          'solved as a steady state'))
    issues.extend(_mesh_issues(vctx, path))
    issues.extend(face_conflicts(vctx, physics_spec, _BOUNDARY_FEATURES, path))
    own_mirrors = feature_mirror_faces(vctx.spec, physics_spec)
    issues.extend(own_mirror_face_conflicts(vctx, physics_spec, _BOUNDARY_FEATURES, own_mirrors))
    issues.extend(_mirror_plane_issues(vctx, physics_spec, own_mirrors))
    issues.extend(_normal_force_issues(vctx, physics_spec))
    fluids = fluid_materials(vctx)
    if _time_mode(vctx, settings_values) == 'transient':
        issues.extend(_wave_step_issues(vctx, fluids, path))
    elif resolved_stepper_precision(vctx.solvers(physics_spec)) == 'half_storage':
        issues.append(Issue('error', path, "half_storage (fp16 populations) runs transient flow only (time_mode "
                                           "'transient' in a time_dependent study): the steady march already sits at "
                                           'the float32 rounding floor; use a single-precision stepper solver'))
    if reports_model_checks(vctx, physics_spec, FLUID_TYPE_NAME):
        if not fluids:
            issues.append(Issue('error', path, 'no selected material has a viscosity > 0, so there is no fluid: set '
                                               "viscosity and rho on the flow region's material (library water has "
                                               'both)'))
        for name, properties in fluids:
            if not evaluates_to(properties.get('rho'), vctx.parameters, lambda value: value > 0.0):
                issues.append(Issue('error', ('material', name),
                                    f"material '{name}' has a viscosity but no density rho > 0: every fluid needs rho"))
    enabled = {feature.type for feature in physics_spec.features if feature.enabled}
    if 'inlet_velocity' in enabled and 'pressure' not in enabled:
        issues.append(Issue('warning', path, 'velocity inlets without any pressure face: the inflow must balance the '
                                             'outflow exactly, else mass accumulates (add a pressure outlet)'))
    issues.extend(_mach_issues(vctx, physics_spec, settings_values, fluids, path))
    return issues


# ---- physics factory callbacks ----
def _create_flow(ctx, physics_spec, values):
    return FluidPhysics(ctx.domain, time_mode=_time_mode(ctx, values), relaxation_time=values['relaxation_time'],
                        tolerance=values['tolerance'], max_steps=values['max_steps'],
                        precision=stepper_precision(ctx.solvers(physics_spec.name)))


def _steps_in_time(settings_values):
    return settings_values['time_mode'] == 'transient'


def _face_kinds_of(spec, physics_spec):
    """{outer face: 'wall' | 'slip' | 'inlet' | 'pressure' | 'periodic'} of the enabled face features (later ones win,
    as they apply in order; symmetry is a slip face); a face without one is a no-slip wall."""
    kinds = {face: 'wall' for face in FACES}
    for feature in physics_spec.features:
        if not feature.enabled or feature.type not in _FACE_FEATURE_KINDS:
            continue
        try:
            faces = [name for name in expand_entities(spec, feature.selection) if name in FACES]
        except (ValueError, KeyError, TypeError):   # reported by the selection checks
            continue
        kinds.update((face, _FACE_FEATURE_KINDS[feature.type]) for face in faces)
    return kinds


def _face_kinds(context):
    """_face_kinds_of the physics of a time-estimate context."""
    return _face_kinds_of(context.spec, context.physics_spec)


def _shear_free_counts(face_kinds):
    """Faces with a slip_wall / symmetry / pressure feature per axis (x, y), as FluidPhysics._shear_free_counts after
    the build."""
    return tuple(sum(face_kinds[f'{axis}_{side}'] in ('slip', 'pressure') for side in ('lo', 'hi')) for axis in 'xy')


def _slowest_shear_rate(window_shape, face_kinds) -> float:
    """Decay rate / lattice viscosity [1 / cells^2] of the slowest shear mode the faces allow: the flow-rate mode along
    an axis whose faces leave its flux free (periodic / both pressure), diffusing across the other (k = pi / L between
    no-slip / inlet faces, pi / 2L with one, solids inside taken as walls with none), or a circulation (stream function
    0 on every face but pressure ones: pi / L, pi / 2L, periodic 2 pi / L per axis)."""
    lengths = dict(zip('xy', window_shape))

    def faces(axis):
        return face_kinds[f'{axis}_lo'], face_kinds[f'{axis}_hi']
    rates = []
    for along, across in (('x', 'y'), ('y', 'x')):
        if set(faces(along)) not in ({'periodic'}, {'pressure'}):
            continue
        held = sum(kind in ('wall', 'inlet') for kind in faces(across))
        rates.append((math.pi / (lengths[across] * (2.0 if held == 1 else 1.0))) ** 2)
    circulation = 0.0
    for axis in 'xy':
        if set(faces(axis)) == {'periodic'}:
            circulation += (2.0 * math.pi / lengths[axis]) ** 2
            continue
        closed = sum(kind != 'pressure' for kind in faces(axis))
        if closed:
            circulation += (math.pi / (lengths[axis] * (1.0 if closed == 2 else 2.0))) ** 2
    if circulation > 0.0:
        rates.append(circulation)
    return min(rates) if rates else (math.pi / min(window_shape)) ** 2


def _time_estimate(settings_values, mesh, context):
    """Lattice steps streaming the populations: transient = one per runner step of dt_lbm (the most viscous fluid's
    relaxation time); steady = a march per solve: tau ln(1 / tolerance) of the slowest shear mode (_slowest_shear_rate,
    the mean lattice viscosity of the fluids) on the solver's growing check intervals (solver.march_steps_estimate), +
    one confirming check, at least 3 checks, below the resolved step limit. Counted by the physics' lattice_steps."""
    fluids = fluid_materials(context)
    kinematic = _kinematic_viscosities(context, fluids)
    if mesh.kind != 'tensor' or not kinematic:
        return TimeEstimate(not_estimated=('no fluid with numeric viscosity and density',))
    dx = _window_dx(mesh)
    offset = settings_values['relaxation_time'] - 0.5
    counter = ('solvers', context.name, 'counters', 'lattice_steps')
    transient = context.time_dependent and settings_values['time_mode'] == 'transient'
    if transient:
        dt_lbm = offset * dx * dx / (3.0 * max(kinematic))
        # per runner step: the lattice step (lattice_to_physical fused into it), ring copies, probe samples
        return explicit_step_estimate(context, 'lattice', dt_lbm, launches=3, counter=counter)
    window_shape = tuple(max(count, 1) for count in mesh.window_axis_cells[:2])
    face_kinds = _face_kinds(context)
    solves, kind, limit = context.static_solves()
    periodic_axes = tuple(face_kinds[f'{axis}_lo'] == 'periodic' for axis in 'xy')
    first_interval, largest_interval, max_steps = steady_march_steps(
        window_shape, _shear_free_counts(face_kinds), periodic_axes, settings_values['relaxation_time'],
        min(kinematic) / max(kinematic), settings_values['max_steps'])
    # lattice viscosity offset / 3 on the most viscous fluid (dt_lbm); the mean over the fluids: the Rayleigh quotient
    # of a mode spread evenly over them
    mean_lattice_viscosity = offset / 3.0 * (sum(kinematic) / len(kinematic)) / max(kinematic)
    decay_steps = 1.0 / (mean_lattice_viscosity * _slowest_shear_rate(window_shape, face_kinds))
    decay_times = max(math.log(1.0 / min(float(settings_values['tolerance']), 0.5)), 1.0)
    guess = float(march_steps_estimate(decay_steps, decay_times, first_interval, largest_interval, max_steps))
    # a march step is one kernel launch (on the gpu fewer: solver.GRAPH_STEPS steps per CUDA graph replay)
    return explicit_step_estimate(context, 'lattice', None, launches=1, counter=counter).with_steps(
        solves * guess, 'guess', steps_limit=solves * max_steps, measured_by_ticks=False)


def _memory_estimate(settings_values, mesh, context):
    """Full-grid SI fields (8 x float32) plus the window lattice: 2 x 9 populations, lattice macros and mass density
    (4 x float32), fluid mask, as FluidPhysics.prepare keeps them: the saved velocity of the steady march (stationary
    only), per-cell relaxation rates (cpu, or fluids of several / unknown kinematic viscosities: the gpu reads 1-element
    rates of one) and lattice forces (cpu, or a body_force feature / a coupling into the flow; a steady force uniform on
    the fluid frees them after its refresh): kept on the cpu (prepare), run items on the gpu (the first step's
    _prepare_lattice allocates them); the Bouzidi wall-row table (conformal boundaries; its (rows, 8) fractions, wall
    cells only, not counted); refresh temporaries (_REFRESH_TEMPORARY_FLOATS window float32: the float64 kinematic
    viscosity rebuilt in place, never kept, or the per-direction equilibrium of a dt change) in the build (finalize's
    refresh) and in the run (a materials change, a new lattice dt: the dt rescale); the vorticity buffer of the first
    request (run)."""
    window_cells = window_cell_count(mesh)
    half = resolved_stepper_precision(context.solvers) == 'half_storage'
    population_bytes = 2 if half else FLOAT32_BYTES
    conformal = getattr(context.spec.solver_defaults, 'conformal_boundaries', False) is True
    host = solver_backend(context.spec) == 'cpu'
    stationary = not (context.spec.study.type == TIME_DEPENDENT_STUDY and settings_values['time_mode'] == 'transient')
    fluids = fluid_materials(context)
    kinematic = _kinematic_viscosities(context, fluids)
    per_cell_rates = host or len(kinematic) < len(fluids) or len(set(kinematic)) > 1
    physics_spec = context.physics_spec
    forced = host or any(feature.enabled and feature.type == 'body_force' for feature in physics_spec.features) or any(
        coupling.enabled and coupling.target == physics_spec.name for coupling in context.spec.couplings)
    fields = MemoryEstimate.from_items((('fields u, u_abs, pressure, body forces (8 x float32)',
                                         8 * FLOAT32_BYTES * int(mesh.n_cells)),))
    pair_bytes = 2 * FLOAT32_BYTES * window_cells
    cell_arrays = (('per-cell relaxation rates (2 x float32)', pair_bytes if per_cell_rates else 0),
                   ('per-cell lattice forces (2 x float32)', pair_bytes if forced else 0))
    refresh_bytes = _REFRESH_TEMPORARY_FLOATS * FLOAT32_BYTES * window_cells
    lattice = MemoryEstimate.from_items(
        ((f"populations (2 x {N_DIRECTIONS} x {'float16' if half else 'float32'})",
          2 * N_DIRECTIONS * population_bytes * window_cells),
         ('lattice macros, mass density (4 x float32 + bool)', (4 * FLOAT32_BYTES + BOOL_BYTES) * window_cells),
         ('saved velocity of the steady march (2 x float32)', 2 * FLOAT32_BYTES * window_cells if stationary else 0),
         *(cell_arrays if host else ()),
         ('Bouzidi wall rows (int32)', 4 * window_cells if conformal else 0)),
        build_extra=refresh_bytes,
        run_items=(('vorticity (float32)', FLOAT32_BYTES * window_cells), *(() if host else cell_arrays)),
        transient_items=(('fluid lattice refresh',
                          f'lattice refresh temporaries ({_REFRESH_TEMPORARY_FLOATS} x float32)', refresh_bytes),))
    return combine((fields, lattice))


# ---- type ----
_TIME_VALUE_TEXT = ' Stationary flow re-marches to a steady state after each change in t.'


def _velocity_param(name, label, default, help_text, tip):
    return Param(name, 'float', default, label=label, unit='m/s', field_dependent=True, time_dependent=True,
                 help=help_text + _TIME_VALUE_TEXT, tip=tip)


_FACE_ONLY = dict(allow_interior=False, requires_uniform_mesh=True)
_FEATURES = (
    FeatureType('wall', 'Wall', 'boundary_condition', 'boundary', **_FACE_ONLY,
                params=(_velocity_param('tangential_velocity', 'Wall velocity', 0.0,
                                        'Along the wall: +x on y_lo / y_hi, +y on x_lo / x_hi (a lid-driven cavity); '
                                        'x, y at the centres of the face cells.',
                                        'Speed of the wall moving along itself; 0 for a fixed wall.'),),
                summary='No-slip wall on the selected domain edges, fixed or sliding along itself. Edges without a '
                        'flow condition are fixed walls.',
                description='No-slip halfway bounce-back (the wall sits on the face), moving along itself at '
                            'tangential_velocity. A later boundary feature on the same face wins (validation '
                            'warning).',
                apply=_apply_wall),
    FeatureType('slip_wall', 'Slip wall (symmetry)', 'boundary_condition', 'boundary', **_FACE_ONLY,
                summary='Wall without friction on the selected domain edges: no flow through it, free flow along it. '
                        'Use it on a symmetry line.',
                description='Free slip (specular reflection of the populations): zero normal velocity and shear '
                            'stress, a symmetry line of the flow, but no model mirror plane (results do not unfold, '
                            'integrals do not double): the symmetry feature is the same face as one.',
                apply=_apply_slip),
    FeatureType(_SYMMETRY_FEATURE, 'Symmetry', 'boundary_condition', 'boundary', **_FACE_ONLY,
                summary='Makes the selected domain edges mirror planes, so only half (or a quarter) of a flow that '
                        'is the same on both sides of them needs to be drawn.',
                description='Even mirror plane on outer faces (x_lo ... y_hi): the free-slip face of slip_wall '
                            '(specular reflection of the lattice populations, the exact mirror of the lattice: zero '
                            'normal velocity, zero shear stress) that also becomes a model mirror plane '
                            '(core.mirror): fields unfold across it (the velocity along the plane and the pressure '
                            'even, the velocity normal to it and the vorticity odd; Unfold mirror planes, '
                            'Results.field(unfold=True)), field derived integrals reaching it count twice (the '
                            'unfolded structure, e.g. the flow rate of a half channel), max / min include the image. '
                            "A parabolic velocity inlet reaching the plane takes the unfolded run's shape (the half "
                            'parabola, as beside a slip_wall). Solid objects may touch the plane (their mirror links '
                            'are halfway bounce-back, also with conformal_boundaries). Warnings: a body force '
                            'component normal to the plane (gravity, a coupled buoyancy: its image is the opposite '
                            'force). Errors: another flow face feature on the same face (also a periodic pair through '
                            'it), a polar / spherical domain, another physics with mirror bc settings (EM) that does '
                            'not set the face symmetric / antisymmetric, parity_sectors (laminar flow has no odd '
                            'mirror feature). A coupled heat physics must be insulated on the plane (heat validation '
                            'warns); fluid_heat_convection carries nothing through it.',
                apply=_apply_slip, validate=_validate_symmetry),
    FeatureType('inlet_velocity', 'Velocity inlet', 'boundary_condition', 'boundary', **_FACE_ONLY,
                params=(_velocity_param('velocity', 'Inflow velocity', 1e-3,
                                        "Normal velocity into the domain (< 0 outflow); with profile 'parabolic' the "
                                        'mean over each run of fluid cells along the face; x, y at the centres of the '
                                        'face cells. Typical microfluidics 1e-4 - 1e-2.',
                                        "Speed of the flow entering through the edge; with a parabolic profile its "
                                        'average.'),
                        Param('profile', 'choice', 'uniform', label='Profile', choices=('uniform', 'parabolic'),
                              choice_labels=(('uniform', 'Uniform'), ('parabolic', 'Parabolic (fully developed)')),
                              help='uniform: the value per cell; parabolic: the value times the discrete Poiseuille '
                                   'shape s (w - s) of each run of fluid cells along the face, mean 1 (peak 1.5 x the '
                                   'value); a run ending on a slip_wall / symmetry face takes the half parabola s (2 w '
                                   '- s) of its mirror image.',
                              tip='Uniform: the same speed across the edge. Parabolic: the fully developed profile of '
                                  'a channel, zero at its walls.')),
                summary='Flow entering through the selected domain edges at a given speed.',
                description='Moving-wall bounce-back with the face-normal velocity (solid cells along the face stay '
                            'solid). Combine with a pressure face elsewhere (validation warns: without one inflow and '
                            'outflow must balance exactly).',
                apply=_apply_inlet),
    FeatureType('pressure', 'Pressure', 'boundary_condition', 'boundary', **_FACE_ONLY,
                params=(Param('pressure', 'float', 0.0, label='Pressure', unit='Pa', field_dependent=True,
                              time_dependent=True,
                              help='Gauge pressure (0 at an open outlet); x, y at the centres of the face cells.'
                                   + _TIME_VALUE_TEXT,
                              tip='Pressure on the edge, relative to the surroundings; 0 for an open outlet.'),),
                summary='Inlet or outlet at a given pressure on the selected domain edges; flow follows the pressure '
                        'difference.',
                description='Anti-bounce-back with the boundary cell velocity: an open inlet or outlet. A model '
                            'without a pressure face keeps a zero mean pressure.',
                apply=_apply_pressure),
    FeatureType('periodic', 'Periodic pair', 'boundary_condition', 'boundary', min_entities=2, max_entities=2,
                **_FACE_ONLY,
                summary='Connects two opposite domain edges: flow leaving one re-enters through the other, as in an '
                        'infinitely long channel.',
                description="Both faces of one axis (['x_lo', 'x_hi'] or ['y_lo', 'y_hi']); the lattice wraps "
                            'around. Drive such a channel with a body_force. Only the flow is periodic (other physics '
                            'keep their own face conditions).',
                apply=_apply_periodic, validate=_validate_periodic),
    FeatureType('body_force', 'Body force', 'domain_condition', 'domain',
                params=(Param('f_x', 'float', 0.0, label='Body force x', unit='N/m^3', field_dependent=True,
                              time_dependent=True,
                              help='Guo forcing; a periodic channel pressure gradient dp/dx is f_x = -dp/dx (an '
                                   'oscillating one, e.g. F cos(omega t), gives Womersley flow).' + _TIME_VALUE_TEXT,
                              tip='Force per volume along x pushing the fluid, for example a pressure gradient.'),
                        Param('f_y', 'float', 0.0, label='Body force y', unit='N/m^3', field_dependent=True,
                              time_dependent=True,
                              help='E.g. gravity -rho g = -9.79e3 for water (a hydrostatic pressure gradient builds up '
                                   'against it).' + _TIME_VALUE_TEXT,
                              tip='Force per volume along y, for example gravity.')),
                summary='Pushes the fluid in the selected domains with a force per volume, such as gravity or a '
                        'pressure gradient.',
                description='Force density on the selected cells (several add; solid cells ignore it), held in the '
                            'solver state body_force_x_ext / _y_ext, separate from the coupled slots body_force_x / _y '
                            '(output fields). The same feature as the solid_mechanics body_force.',
                apply=_apply_body_force),
    FeatureType('initial_velocity', 'Initial values', 'domain_condition', 'domain',
                params=(Param('u_x', 'float', 0.0, label='Initial velocity x', unit='m/s', field_dependent=True,
                              help='At the cell centres, e.g. the Taylor-Green vortex U sin(k x) cos(k y).',
                              tip='Flow velocity along x at the start.'),
                        Param('u_y', 'float', 0.0, label='Initial velocity y', unit='m/s', field_dependent=True,
                              help='At the cell centres.',
                              tip='Flow velocity along y at the start.'),
                        Param('pressure', 'float', 0.0, label='Initial pressure', unit='Pa', field_dependent=True,
                              help='Gauge pressure; 0 is fine for slow flow (an inconsistent start only launches weak '
                                   'sound waves that the viscosity damps).',
                              tip='Pressure at the start, relative to the surroundings.')),
                summary='Starting flow (velocity and pressure) of the selected domains for transient flow.',
                description='Transient flow: the populations of the selected fluid cells start at the equilibrium of '
                            '(u_x, u_y, pressure) instead of rest (a later feature wins its cells; solid cells stay at '
                            'rest). Keep the velocity divergence free: a divergent start relaxes through sound waves. '
                            'Stationary flow only starts its first steady march from it.',
                apply=_apply_initial_velocity, memory_estimate=_initial_velocity_memory),
    FeatureType('flow_probe', 'Flow probe', 'probe', 'point', object_kinds=('point',), max_entities=1,
                result_kind='timeseries',
                params=(Param('component', 'choice', 'u_abs', label='Quantity', choices=tuple(PROBE_QUANTITIES),
                              choice_labels=(('u_abs', 'Flow speed'), ('u_x', 'Velocity x'), ('u_y', 'Velocity y'),
                                             ('pressure', 'Pressure')),
                              help='u_abs = |u|, u_x, u_y [m/s] or pressure [Pa].',
                              tip='Which flow quantity is recorded.'),),
                summary='Records the flow speed, a velocity component or the pressure at a point.',
                description='The cell containing the point: stationary flow once per steady solve and every '
                            'stationary iteration (t = 0); transient flow after every runner step, at its end time.',
                apply=_apply_flow_probe, result=_flow_probe_result, validate=validate_points_inside),
)

_FIELD_INFOS = tuple(FluidRenderer.field_infos.values())
_STATE_FIELDS = tuple(FluidRenderer.field_infos[name] for name in ('u_x', 'u_y', 'u_abs', 'pressure'))

LAMINAR_FLOW = PhysicsType(
    name=FLUID_TYPE_NAME,
    label='Laminar Flow (2D)',
    default_instance_name='fluid',
    family='',
    group='fluid',
    summary='Slow, smooth flow of liquids and gases through channels and around obstacles, driven by inlets, '
            'pressures or forces. Materials without viscosity are solid walls.',
    description='2D incompressible laminar flow per unit z length: D2Q9 lattice Boltzmann (TRT collision with the magic '
                'parameter 3/16, He-Luo incompressible equilibrium, Guo forcing) on the window, which needs a uniform '
                'mesh of square cells (no PML; outputs edge-extended into an EM PML ring). Solid cells (no viscosity) '
                'are no-slip halfway bounce-back walls (solver_defaults.conformal_boundaries: interpolated Bouzidi '
                'bounce-back at the drawn wall position); unreferenced outer faces are fixed no-slip walls, symmetry '
                'faces free-slip model mirror planes (a drawn half). On a mirror plane of another physics (EM bc '
                'symmetric / antisymmetric, a heat symmetry face) the flow face must be free slip (validation warns; '
                'a periodic pair there is an error). Steady flow is exact for Poiseuille channels, second-order '
                'otherwise; valid at lattice Mach < 0.2 and laminar Reynolds numbers. A steady solve needs ~ L^2 '
                'lattice steps (L = longest window axis in cells; a 500-cell axis ~1e6), each one GPU kernel (cpu '
                'backend: the numba twin when installed, else the far slower numpy twin; half_storage GPU only). '
                'pressure is gauge pressure. Heat: fluid_heat_convection; buoyancy is a field_transfer of heat T into '
                'body_force_y.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=dict(FLUID_SCHEMA),
    state_fields=_STATE_FIELDS,
    output_fields=_FIELD_INFOS,
    source_slots={'body_force_x': ('N/m^3', 'Coupled force density along x (e.g. buoyancy), added to the body_force '
                                            'features; replaced by each transfer.'),
                  'body_force_y': ('N/m^3', 'Coupled force density along y, e.g. Boussinesq buoyancy +rho g beta (T - '
                                            'T0) from heat T by field_transfer (g = 9.81, gravity along -y: warm '
                                            'fluid is pushed along +y); replaced by each transfer.')},
    supported_studies=(STATIONARY_STUDY, TIME_DEPENDENT_STUDY),
    default_study={'type': STATIONARY_STUDY},
    create=_create_flow,
    steps_in_time=_steps_in_time,
    validate=_validate_flow,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    solver_roles={'stepper': HALF_STORAGE_STEPPER_ROLE},
    conformal_boundaries=True,
    backends=('gpu', 'cpu'),   # core.kernels twins on the cpu (numba step); half_storage refused there (explicit_kernels)
    indicator_fields=('u_x', 'u_y'),   # velocity components (u_abs kinks where u passes through 0)
)


def _selftest_cases():
    """The laminar_flow SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import SELFTEST_CASES
    from .selftests_backend import FLUID_BACKEND_SELFTEST_CASES
    from .selftests_time import TIME_SELFTEST_CASES
    return (*SELFTEST_CASES, *TIME_SELFTEST_CASES, *FLUID_BACKEND_SELFTEST_CASES)


def register(registry):
    """laminar_flow, the 'heated_microchannel' template, then the selftest loader (fluid_heat_convection is registered
    by coupling/definitions_fluid.py)."""
    registry.register_physics(LAMINAR_FLOW)
    registry.register_template('heated_microchannel', HEATED_MICROCHANNEL_LABEL, HEATED_MICROCHANNEL_DESCRIPTION,
                               heated_microchannel_spec,
                               summary='Water flowing through a channel in a silica chip carries away the heat of a '
                                       'heater above it. Steady state.')
    registry.register_selftests(_selftest_cases)
