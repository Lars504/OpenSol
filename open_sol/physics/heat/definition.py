"""Model-layer definition of the 'heat' physics type (settings, features, material vocabulary, factory, validation)
plus its template and selftests. Loaded lazily by the registry (physics/heat/__init__.py must not import it)."""
import math

from ...core.backend import to_host
from ...core.constants import N_A
from ...core.memory_estimate import (COMPLEX128_BYTES, FLOAT32_BYTES, FLOAT64_BYTES, MemoryEstimate, combine,
                                     diffusion_system_estimate, host_lu_bytes, window_cell_count)
from ...core.renderer import FieldInfo
from ...core.time_estimate import (HeldValues, TimeEstimate, TimeItem, combine as combine_time,
                                   diffusion_time_estimate)
from ...fem.diffusion import (cudss_back_end, cudss_diffusion_estimate, fem_diffusion_estimate, fem_solve_host_ops,
                              fem_time_system, painted_fraction)
from ...fem.diffusion3d import fem_diffusion_estimate_3d
from ...model.definition_helpers import (
    AXISYMMETRIC_SPACE_SETTINGS, AZIMUTHAL_ORDER_PARAM, CG_DIFFUSION_LINEAR_ROLE, DISCRETIZATION_PARAM,
    antiperiodic_wrap,
    axisymmetric_issues, azimuthal_order_issues, azimuthal_order_of, cg_solver_issues, conducting_fraction, conducts, cross_term_materials, evaluates_to, face_conflicts,
    element_order, geometry_param, held_object_conflicts, indefinite_cross_term_issue, is_axisymmetric,
    linear_solver_direct,
    linear_solver_mixed, linear_solver_multigrid, linear_solver_precision, make_apply_inward_flux, make_point_probe, mapped_domain_issues,
    material_follows_state, time_following_values,
    mirror_face_issues, on_tetrahedra, on_triangles, optional_material_param, outline_full_grid, outline_pieces_of,
    perimeter_flux_weights, point_cells, reports_model_checks, uses_time, validate_points_inside,
)
from ...model.geometry import expand_entities
from ...model.schema import FeatureType, Issue, Param, PhysicsType, SolverRole, feature_path
from ...model.spec import RESERVED_ENTITY_NAMES
from ...solvers.host_direct import HOST_DIRECT_BACK_ENDS
from ...solvers.line_block import (MAX_CHUNK_SHIFTS, operator_bytes as line_block_operator_bytes,
                                   shift_bytes as line_block_shift_bytes)
from .boundaries import BCSide, HeatBoundaryConfig
from .materials import THERMAL_PROPERTY_ARRAYS, THERMAL_SCHEMA
from .renderer import HeatRenderer
from .selftests import HEATED_SLAB_TEMPLATE_DESCRIPTION, HEATED_SLAB_TEMPLATE_SUMMARY, heated_slab_spec
from .selftests_3d import HEATED_CHIP_TEMPLATE_DESCRIPTION, HEATED_CHIP_TEMPLATE_SUMMARY, heated_chip_3d_spec
from .selftests_thermal_noise import (WAVEGUIDE_TEMPLATE_DESCRIPTION, WAVEGUIDE_TEMPLATE_SUMMARY,
                                      waveguide_thermal_noise_spec)
from .solver import HeatPhysics
from .solver_fem import FEMHeatPhysics
from .solver_fem3d import FEMHeatPhysics3D
from .thermal_noise import (AXIAL_PROFILES, COLLECTED_SELECTIONS, DIRECTIONS, INCIDENT_SELECTIONS, WEIGHTINGS,
                            ThermalNoiseMonitor, thermal_noise_result)

PHYSICS_TYPE_NAME = 'heat'
THERMAL_NOISE_STUDY = 'thermal_noise'
_MODE_PHYSICS_TYPE = 'em_modes'
_ILLUMINATION_PHYSICS_TYPES = ('em_fdfd_ez', 'em_fdfd_hz')

# Features that pin the temperature level of a steady solve (convection-type ones only when h > 0).
_ANCHOR_FEATURES = frozenset({'temperature', 'convection', 'fixed_temperature_domain', 'out_of_plane_convection',
                              'radiation'})
_CONVECTIVE_FEATURES = frozenset({'convection', 'out_of_plane_convection'})
_BOUNDARY_FEATURES = frozenset({'temperature', 'heat_flux', 'convection'})
_HELD_FEATURES = frozenset({'temperature', 'fixed_temperature_domain'})   # pin cells (object perimeters / interiors)
_PERIMETER_FEATURES = frozenset({'heat_flux', 'convection', 'radiation'})   # act on object perimeter cells, ignored
# where held
_FACE_FEATURES = _BOUNDARY_FEATURES | {'radiation'}   # act on outer faces (radiation adds to the face condition)
MIN_EM_STEPS_PER_SOLVE = 100     # EM steps per heat solve below which the CG solves dominate the run time
# a heat boundary feature on an EM mirror plane (definition_helpers.mirror_face_issues; error in thermal_noise, whose
# x 2 per mirror assumes the even, insulated mirror)
_MIRROR_FACE_CONSEQUENCE = (
    'EM losses are mirror-even, so the image of the half model is insulated there. A temperature face is the odd '
    'mirror of T - T_face instead (exact: the Dirichlet half cell equals the antisymmetric image), i.e. a plane held '
    'at T_face inside the full structure; heat_flux is a sheet source of twice its flux on the plane; convection has '
    'no mirror image. Remove the feature for the mirror image of the drawn half')


# ---- settings ----
_SETTINGS = (
    Param('time_mode', 'choice', 'stationary', label='Time dependence', choices=('stationary', 'transient'),
          choice_labels=(('stationary', 'Steady state'), ('transient', 'Transient')),
          tip='How the temperature is computed.\n\n'
              '- Steady state: the temperature reached after a long time, without heat storage.\n'
              '- Transient: the temperature evolves in time, using density and heat capacity.\n\n'
              "Only a 'Time dependent' study runs transient heat; every other study solves the steady state.",
          help='stationary: a steady solve at every heat step (quasi-static, no heat capacity); transient: '
               'implicit Euler with rho*cp in steps <= time_step. Only a time_dependent study runs transient heat: '
               'every other study (stationary, frequency_domain) solves the steady state (transient is ignored '
               'there); with EM in a time-dependent study each heat step spans every * dt_EM.'),
    Param('T0', 'float', 293.15, label='Reference temperature', unit='K', min=0.0, min_exclusive=True,
          tip='Starting temperature of every cell, and the temperature at which the refractive indices of the '
              'materials apply.\n\nTypically 293.15 K.',
          help='Initial temperature of every cell and reference of the thermo-optic change n = n0 + dndT (T - T0). '
               'Cells that neither conduct, lose heat nor (transient, rho and cp set) store heat keep T0 (or their '
               'initial temperature) and ignore heat sources. Typical 293.15 K.'),
    Param('time_step', 'float', 1e-3, label='Time step', unit='s', min=0.0, min_exclusive=True,
          visible_if=(('time_mode', ('transient',)),),
          tip='Largest time step of transient heat. Any step is stable, but keep it well below the time heat needs to '
              'spread over the region you care about.\n\n'
              "With a wave physics in the model, each heat step is 'Step every' wave time steps, and must not exceed "
              'this value.',
          help='Largest implicit-Euler step (the heat dt_limit). Heat alone steps by time_step (every must be 1); '
               'with EM each heat step is every * dt_EM, which must not exceed time_step. Implicit Euler is stable '
               'for any step but first-order accurate: keep it well below the diffusion time rho*cp*L^2/k of '
               'interest. T is the temperature at the end of the step just taken, so the heat time label and '
               'probe times run up to (every - 1) * dt_EM ahead of the snapshot time (the last may exceed the '
               'duration).'),
    geometry_param(
        '2D conduction per unit z length (W/m)',
        "(r, z) conduction; every conductance, face and cell volume is weighted by 2 pi r (x-faces at their own r, "
        'y-faces and volumes at the cell-centre r; 2 pi r_c dr dz is the exact ring volume; object-perimeter heat_flux '
        '/ convection likewise), so densities (Q W/m^3, heat flux W/m^2, h W/(m^2 K)) keep their units, a '
        "point_heat_source P is the ring's total power in W (on the axis cell a point source) and the 'integral' of Q "
        "the body's total power in W; k_xy is not available (validation error); out_of_plane_convection stays a "
        'volumetric loss h * sides / thickness; azimuthal_order m != 0 weights by pi r instead',
        'powers are per metre of depth.',
        'Point sources are rings (on the axis: a point) with their total power in watts.'),
    AZIMUTHAL_ORDER_PARAM,
    DISCRETIZATION_PARAM,
)
_BOTH_GRIDS = ('tensor', 'triangles')   # FeatureType.grids of the features that run on triangle meshes too
_FEM_GRIDS = ('tensor', 'triangles', 'tetrahedra')   # ... and on tetrahedra (FEMHeatPhysics3D)


def _material_property(name, label, unit, minimum=None, help_text='', tip=''):
    """Thermal material property Param (default None: unset properties are never written)."""
    return optional_material_param(name, label, unit, THERMAL_PROPERTY_ARRAYS[name], minimum, help_text, tip)


def _component_tip(component_text, isotropic_label):
    """GUI tip of a component entry that replaces its isotropic property for one direction."""
    return f"{component_text}. Replaces '{isotropic_label}' for this direction."


_CONDUCTIVITY_UNIT = 'W/(m K)'
_CONDUCTIVITY_COMPONENT_TEXT = ('Thermal conductivity along {axis} of a material that conducts differently along x '
                                'and y')
_THERMO_OPTIC_COMPONENT_TEXT = 'Refractive index change per kelvin for light with its electric field along {axis}'
_MATERIAL_PROPERTIES = (   # isotropic entries before their components (components override)
    _material_property('k', 'Thermal conductivity', _CONDUCTIVITY_UNIT, 0.0,
                       'Isotropic conductivity: sets k_xx, k_yy and k_zz (component entries override it; k_zz acts only '
                       'in 3D models); unset (no '
                       'library value) means 0, so those cells do not conduct. May depend on x, y (z in 3D) and state '
                       'fields such as T (e.g. k_300*(300/T)**1.3; a stationary study then iterates). Typical: silica 1.38, '
                       'silicon 130, aluminium 237, gold 317, silver 429, water 0.6.',
                       tip='How well the material conducts heat. Empty takes the library value; a material with '
                           'neither does not conduct.\n\nTypical: silica 1.38, silicon 130, gold 317, water 0.6.'),
    _material_property('k_xx', 'Thermal conductivity along x', _CONDUCTIVITY_UNIT, 0.0,
                       'Conductivity along x; overrides k for this axis.',
                       tip=_component_tip(_CONDUCTIVITY_COMPONENT_TEXT.format(axis='x'), 'Thermal conductivity')),
    _material_property('k_yy', 'Thermal conductivity along y', _CONDUCTIVITY_UNIT, 0.0,
                       'Conductivity along y; overrides k for this axis.',
                       tip=_component_tip(_CONDUCTIVITY_COMPONENT_TEXT.format(axis='y'), 'Thermal conductivity')),
    _material_property('k_zz', 'Thermal conductivity along z', _CONDUCTIVITY_UNIT, 0.0,
                       'Conductivity along z (3D models only; a 2D model has no z conduction); overrides k for this '
                       'axis.',
                       tip="Thermal conductivity along z of a layered or anisotropic material, used only in 3D models. "
                           "Replaces 'Thermal conductivity' for this direction."),
    _material_property('k_xy', 'Thermal conductivity, cross term', _CONDUCTIVITY_UNIT,
                       tip='Cross term of a rotated crystal: heat also flows sideways to the temperature gradient. '
                           'Usually empty.\n\nKeep it smaller than both the x and y values. Not available in 3D '
                           'models.',
                       help_text='Off-diagonal conductivity (cross conduction, e.g. a rotated anisotropic crystal); '
                                 'keep k_xy^2 < k_xx*k_yy (validation error for plain numbers). Needs a 2D Cartesian '
                                 'planar model (validation error otherwise); uniform or graded meshes. '
                                 'Cross heat flows only inside 2x2 cell blocks that all conduct (area-weighted mean '
                                 'k_xy of the block), never through the outer faces; the heat_flux_abs field includes it '
                                 '(block values averaged over each cell\'s blocks). On strongly graded cells keep '
                                 'k_xy^2 clearly below k_xx*k_yy: the discrete operator is guaranteed positive for '
                                 'k_xy^2 < k_xx*k_yy * 4 r_x / (1 + r_x)^2 * 4 r_y / (1 + r_y)^2 (r = width ratio of '
                                 'neighbouring cells; each factor 0.99 at r = 1.2).'),
    _material_property('rho', 'Density', 'kg/m^3', 0.0,
                       tip="Mass density. Only transient heat uses it, together with 'Specific heat capacity', to "
                           'store heat.\n\nTypical: silica 2200, silicon 2329, water 998.',
                       help_text='Density; used only by transient heat (heat capacity rho*cp). Unset = 0: no heat '
                                 'capacity, so transient heat treats those cells quasi-statically (conducting ones '
                                 'follow their neighbours, non-conducting ones keep their temperature and ignore heat '
                                 'sources; validation warns when heat is transient). Typical: silica 2200, silicon '
                                 '2329, water 998.'),
    _material_property('cp', 'Specific heat capacity', 'J/(kg K)', 0.0,
                       tip='Heat stored per kilogram and kelvin; only transient heat uses it.\n\n'
                           'Typical: silica 746, silicon 700, water 4182.\n\n'
                           'Cells without density or heat capacity store no heat: they follow their surroundings at '
                           'once.',
                       help_text='Specific heat capacity; used only by transient heat (heat capacity rho*cp). Unset = '
                                 '0: no heat capacity (see rho). Typical: silica 746, silicon 700, water 4182.'),
    _material_property('dndT', 'Thermo-optic coefficient', '1/K',
                       tip='Change of the refractive index per kelvin, used by the thermo-optic coupling.\n\n'
                           'Use the measured value of the free material. Typical: silica 1e-5, silicon 1.8e-4, water '
                           '-9e-5.',
                       help_text='Isotropic thermo-optic coefficient dn/dT: sets dndT_xx, dndT_yy and dndT_zz '
                                 '(component entries override it), read by heat_em_thermo_optic (n = n0 + dndT '
                                 '(T - T0)). Use the measured dn/dT of the free material: it already contains the '
                                 'density change of expansion; alpha adds only the growth of the structure. On a '
                                 'dispersive (Drude / Lorentz) material it shifts only the background permittivity, '
                                 'not the pole part (see heat_em_thermo_optic). Typical: '
                                 'silica 1e-5, silicon 1.8e-4, water -9e-5.'),
    _material_property('dndT_xx', 'Thermo-optic coefficient along x', '1/K',
                       tip=_component_tip(_THERMO_OPTIC_COMPONENT_TEXT.format(axis='x, in the plane'),
                                          'Thermo-optic coefficient'),
                       help_text='dn/dT of n_xx: em_fdtd_hz, em_fdfd_hz and em_modes use it for eps_xx; overrides '
                                 'dndT for this component.'),
    _material_property('dndT_yy', 'Thermo-optic coefficient along y', '1/K',
                       tip=_component_tip(_THERMO_OPTIC_COMPONENT_TEXT.format(axis='y, in the plane'),
                                          'Thermo-optic coefficient'),
                       help_text='dn/dT of n_yy: em_fdtd_hz, em_fdfd_hz and em_modes use it for eps_yy; overrides '
                                 'dndT for this component.'),
    _material_property('dndT_zz', 'Thermo-optic coefficient along z', '1/K',
                       tip=_component_tip(_THERMO_OPTIC_COMPONENT_TEXT.format(axis='z, out of the plane'),
                                          'Thermo-optic coefficient')
                           + '\n\nSetting only the x and y values leaves light with an out-of-plane electric field '
                             'unchanged.',
                       help_text='dn/dT of n_zz, the out-of-plane index E_z sees (eps_zz of em_fdtd_ez, em_fdfd_ez '
                                 'and em_modes); set by dndT, which this entry overrides. Without dndT and dndT_zz it '
                                 'is 0, so the E_z types see no thermo-optic change even when dndT_xx / dndT_yy are '
                                 'set. Typical: z-cut lithium niobate (dn_e/dT) 3.3e-5.'),
    _material_property('alpha', 'Thermal expansion coefficient', '1/K',
                       tip='Growth in length per kelvin. The thermo-optic coupling uses it to lengthen the optical path '
                           'of the heated structure.\n\n'
                           "A thin film on a substrate grows with the substrate, so enter the substrate's value there. "
                           'Leave it empty for liquids.',
                       help_text='Linear thermal expansion coefficient: sets alpha_xx and alpha_yy (and alpha_zz when '
                                 'a solid_mechanics or solid_mechanics_3d physics is present, which turns alpha dT '
                                 'into thermal strain and '
                                 'stress through heat_solid_thermal_strain). '
                                 'heat_em_thermo_optic scales eps by (1 + alpha_xx dT)(1 + alpha_yy dT), i.e. n by '
                                 'about (1 + alpha dT): the in-plane growth of the structure on the fixed grid (dropped '
                                 'when solid_em_strain_optic maps the solid strain instead; a film clamped to a substrate grows with the '
                                 'substrate\'s alpha: library values are free-crystal, so override them, e.g. LN on Si '
                                 '2.6e-6 instead of 1.5e-5). '
                                 'Per cell this is first-order, exact only for an alpha uniform over the mode. Leave 0 '
                                 'for liquids: their region is the container\'s. Typical: silica 5.5e-7, silicon '
                                 '2.6e-6, gold 1.4e-5.'),
    _material_property('alpha_xx', 'Thermal expansion along x', '1/K',
                       help_text='Expansion coefficient along x (see alpha).',
                       tip=_component_tip('Thermal expansion along x', 'Thermal expansion coefficient')),
    _material_property('alpha_yy', 'Thermal expansion along y', '1/K',
                       help_text='Expansion coefficient along y (see alpha).',
                       tip=_component_tip('Thermal expansion along y', 'Thermal expansion coefficient')),
)

_RENDER_INFOS = HeatRenderer.field_infos
_STATE_FIELDS = (
    _RENDER_INFOS['T'], _RENDER_INFOS['Q'], _RENDER_INFOS['Q_ext'],
    FieldInfo('k_xx', 'Thermal conductivity xx', unit=_CONDUCTIVITY_UNIT),
    FieldInfo('k_yy', 'Thermal conductivity yy', unit=_CONDUCTIVITY_UNIT),
    FieldInfo('k_zz', 'Thermal conductivity zz', unit=_CONDUCTIVITY_UNIT),
)


# ---- feature params ----
def _temperature_param(help_text, tip, time_dependent=False, follows_parameters=False):
    return Param('temperature', 'float', 293.15, label='Temperature', unit='K', min=0.0, min_exclusive=True,
                 help=help_text, tip=tip, time_dependent=time_dependent, follows_parameters=follows_parameters)


_TIME_VALUE_TEXT = ('May depend on time t [s] (parameters and t only), re-set at the start of every heat step of a '
                    'time_dependent study (the implicit step from t_n to t_n + dt sees the value at t_n; t = 0 in '
                    'other studies).')
_TEMPERATURE_BOUNDS = _temperature_param('', '')   # min / max of a temperature evaluated at a time t


def _heat_transfer_coefficient_param():
    return Param('h', 'float', 10.0, label='Heat transfer coefficient', unit='W/(m^2 K)', min=0.0,
                 help='Heat lost per unit area and kelvin of T - T_ext; h = 0 removes no heat and does not anchor a '
                      'steady solve. Typical: still air 5-25, forced air 10-200, water 500-10000.',
                 tip='Heat lost per area and per kelvin above the ambient temperature; 0 removes nothing.\n\n'
                     'Typical: still air 5 to 25, forced air 10 to 200, water 500 to 10000.')


def _ambient_temperature_param():
    return Param('T_ext', 'float', 293.15, label='Ambient temperature', unit='K', min=0.0, min_exclusive=True,
                 help='Temperature of the surrounding fluid that the loss h (T - T_ext) drives toward; typically '
                      'room temperature, 293.15 K.',
                 tip='Temperature of the surrounding fluid that the surface cools or warms toward.\n\n'
                     'Usually room temperature, 293.15 K.')


# ---- feature apply / result callbacks (worker thread, GPU) ----
def _apply_temperature(ctx, solver, feature_spec, values, selection):
    """Dirichlet on outer faces (a value of t re-set per step in place); object perimeters pin their boundary cells
    (a constant value only: validation); a parameter value follows a reused sweep build (faces and objects)."""
    temperature = values['temperature']
    faces = list(selection.outer_faces)

    def set_face_temperature(value):
        for face in faces:
            solver.set_face_value(face, BCSide('dirichlet', value=value))

    label = f"feature '{feature_spec.name}'"
    ctx.follow_time(temperature, set_face_temperature, solver, f'{label} temperature',
                    bounds_param=_TEMPERATURE_BOUNDS)
    set_objects = None
    if selection.objects:
        if uses_time(temperature):
            raise ValueError(f"temperature '{feature_spec.name}': a temperature of t acts on outer faces only, not "
                             f'on the object perimeters {selection.objects}')
        if ctx.domain.grid_kind == 'triangles':   # Dirichlet on the outline edges
            handle = solver.add_edge_temperature(ctx.boundary_edges(selection.objects), temperature)

            def set_objects(value):
                solver.set_edge_condition(handle, BCSide('dirichlet', value=value))
        elif ctx.domain.grid_kind == 'tetrahedra':   # Dirichlet on the interface faces
            handle = solver.add_surface_temperature(ctx.boundary_faces(selection.objects), temperature)

            def set_objects(value):
                solver.set_surface_condition(handle, BCSide('dirichlet', value=value))
        else:
            mask = ctx.perimeter_mask(selection.objects)
            solver.add_fixed_temperature(mask, temperature, **_held_outline(solver, selection.objects))

            def set_objects(value):   # the held set stays: only its values change (cut-cell edges stay located)
                solver.add_fixed_temperature(mask, value)

    def set_temperature(value):
        set_face_temperature(value)
        if set_objects is not None:
            set_objects(value)
    ctx.follow_parameters(feature_spec.params, values, ('temperature',), set_temperature, solver, label)
    return None


def _validate_temperature(vctx, physics_spec, feature_spec, values):
    """Error for a temperature of t on object perimeters (held cells cannot follow a time-dependent value)."""
    if not uses_time(values.get('temperature')):
        return []
    try:
        objects = [name for name in expand_entities(vctx.spec, feature_spec.selection)
                   if name not in RESERVED_ENTITY_NAMES]
    except (ValueError, KeyError, TypeError):   # reported by the selection checks
        return []
    if not objects:
        return []
    return [Issue('error', feature_path(physics_spec, feature_spec),
                  f'a temperature of time t acts on outer faces only, but the selection holds the objects {objects}: '
                  'select outer faces, or use a constant temperature on the objects')]


def _held_outline(solver, entities):
    """{'inside': drawn outline test} with solver_defaults.conformal_boundaries (cut-cell held edges), else {}."""
    locator = getattr(solver, 'conformal_locator', None)
    return {} if locator is None else {'inside': locator.inside(entities)}


def _apply_convection(ctx, solver, feature_spec, values, selection):
    """Robin on outer faces; object perimeters lose h * perimeter length / cell area * (T - T_ext), with
    solver_defaults.conformal_boundaries per cut cell h L / (1 + h delta / k_n) on the drawn outline (L its chord,
    delta its distance from the owner centre, k_n the owner's normal k at build)."""
    transfer_coefficient, ambient_temperature = values['h'], values['T_ext']
    for face in selection.outer_faces:
        solver.set_face_bc(face, BCSide('robin', h=transfer_coefficient, external=ambient_temperature))
    if not selection.objects:
        return None
    if ctx.domain.grid_kind == 'triangles':   # Robin on the outline edges
        solver.add_edge_convection(ctx.boundary_edges(selection.objects), transfer_coefficient, ambient_temperature)
        return None
    if ctx.domain.grid_kind == 'tetrahedra':   # Robin on the interface faces
        solver.add_surface_convection(ctx.boundary_faces(selection.objects), transfer_coefficient,
                                      ambient_temperature)
        return None
    outline = outline_pieces_of(ctx, solver, selection.objects)
    if outline is None:
        weights = perimeter_flux_weights(ctx, solver, selection.objects)
        solver.add_volumetric_loss(ctx.perimeter_mask(selection.objects), transfer_coefficient * weights,
                                   ambient_temperature)
        return None
    import numpy as np
    pieces, factor = outline
    system = solver.system
    materials = solver.domain.materials
    k_xx, k_yy = (np.asarray(to_host(array[system.window]), dtype=np.float64)[pieces.owner_i, pieces.owner_j]
                  for array in (materials.k_xx, materials.k_yy))
    k_normal = k_xx * pieces.normal_x ** 2 + k_yy * pieces.normal_y ** 2
    surface = transfer_coefficient * pieces.length * factor   # h L (x 2 pi r axisymmetric)
    # in series with the conduction from the owner centre to the drawn edge (k = 0 at build: the bare h L)
    conductance = np.where(k_normal > 0.0, surface * k_normal / np.where(
        k_normal > 0.0, k_normal + transfer_coefficient * pieces.offset, 1.0), surface)
    conductance_full = outline_full_grid(system, pieces, conductance)
    solver.add_surface_loss(conductance_full > 0, conductance_full, ambient_temperature)
    return None


def _apply_radiation(ctx, solver, feature_spec, values, selection):
    """eps sigma (T^4 - T_amb^4) from the boundary cells of outer faces and (2D) object perimeters (staircase
    perimeter weights, also with conformal_boundaries); on triangles / tetrahedra from the boundary edges / faces;
    Newton-linearised per solve."""
    emissivity, ambient_temperature = values['emissivity'], values['T_amb']
    for face in selection.outer_faces:
        solver.add_face_radiation(face, emissivity, ambient_temperature)
    if not selection.objects:
        return None
    if ctx.domain.grid_kind == 'triangles':   # the outline edges
        solver.add_edge_radiation(ctx.boundary_edges(selection.objects), emissivity, ambient_temperature)
    elif ctx.domain.grid_kind == 'tetrahedra':   # the interface faces
        solver.add_surface_radiation(ctx.boundary_faces(selection.objects), emissivity, ambient_temperature)
    else:
        solver.add_radiation(perimeter_flux_weights(ctx, solver, selection.objects), emissivity, ambient_temperature)
    return None


def _apply_heat_source(ctx, solver, feature_spec, values, selection):
    """Q (finite under the selection only) added to Q_ext on the selected cells; a value of t changes it before every
    step by the difference to the last value."""
    source_mask = ctx.domain_mask(selection.entities)
    power_density = values['Q']
    applied = [0.0]   # Q already in Q_ext

    def set_power_density(value):
        if value != applied[0]:
            solver.add_heat_source(source_mask, value - applied[0])
            applied[0] = value

    if not uses_time(power_density):
        solver.add_heat_source(source_mask, ctx.eval_field(power_density, mask=source_mask))
        applied[0] = power_density   # a parameter-only number when it follows a reused sweep build
        ctx.follow_parameters(feature_spec.params, values, ('Q',), set_power_density, solver,
                              f"feature '{feature_spec.name}'")
        return None
    ctx.follow_time(power_density, set_power_density, solver, f"feature '{feature_spec.name}' Q")
    return None


def _apply_point_heat_source(ctx, solver, feature_spec, values, selection):
    """P in the cell of each point (a triangle mesh: at the point's node; tetrahedra: the delta load at the point); a
    value of t changes it before every step by the difference."""
    grid_kind = ctx.domain.grid_kind
    if grid_kind == 'triangles':
        cells = [ctx.point_node(name) for name in selection.objects]
    elif grid_kind == 'tetrahedra':
        cells = [ctx.point_position(name) for name in selection.objects]
    else:
        cells = list(point_cells(ctx, feature_spec, selection))
    applied = [0.0]   # P already in Q_ext

    def set_power(value):
        if value != applied[0]:
            for cell in cells:
                if grid_kind == 'tetrahedra':
                    solver.add_point_heat_source_at(cell, value - applied[0])
                elif grid_kind == 'triangles':
                    solver.add_point_heat_source_node(cell, value - applied[0])
                else:
                    ix, iy, *iz = cell   # iz: [k] in 3D models
                    solver.add_point_heat_source(ix, iy, value - applied[0], *iz)
            applied[0] = value

    ctx.follow_time(values['P'], set_power, solver, f"feature '{feature_spec.name}' P")
    ctx.follow_parameters(feature_spec.params, values, ('P',), set_power, solver, f"feature '{feature_spec.name}'")
    return None


def _apply_fixed_temperature_domain(ctx, solver, feature_spec, values, selection):
    solver.add_fixed_temperature(ctx.domain_mask(selection.entities), values['temperature'],
                                 **_held_outline(solver, selection.entities))
    return None


def _apply_out_of_plane_convection(ctx, solver, feature_spec, values, selection):
    """Loss through the top/bottom surfaces of a slab: h_volumetric = h * sides / thickness."""
    volumetric_coefficient = values['h'] * values['sides'] / values['thickness']
    solver.add_volumetric_loss(ctx.domain_mask(selection.entities), volumetric_coefficient, values['T_ext'])
    return None


def _apply_initial_temperature(ctx, solver, feature_spec, values, selection):
    solver.set_initial_temperature(ctx.domain_mask(selection.entities), values['temperature'])
    return None


_apply_temperature_probe, _temperature_probe_result = make_point_probe('temperature', 'K')


def _apply_thermorefractive_noise(ctx, solver, feature_spec, values, selection):
    """ThermalNoiseMonitor on the selected window cells: an expression weighting now, a mode pair at the solve;
    contributions = the materials under the selection when there are several."""
    import numpy as np
    run_mask = ctx.domain_mask(selection.entities)   # run backend; the monitor holds host arrays
    mask = to_host(run_mask)
    window = solver.system.window
    cell_mask = mask[window]
    contributions = []
    for region in ctx.material_regions:   # cell-disjoint (later materials win), still masked during feature apply
        region_cells = to_host(region.mask[window]) & cell_mask
        if region_cells.any():
            contributions.append((region.name, region_cells))
    common = dict(axial_points=values['axial_points'], contributions=contributions if len(contributions) > 1 else ())
    uniform = values['axial_profile'] == 'uniform'
    axial_length = values['interaction_length'] if uniform else values['waist']
    if values['weighting'] == 'expression':
        weight = np.broadcast_to(to_host(ctx.eval_field(values['weight'], mask=run_mask)), mask.shape)[window]
        monitor = ThermalNoiseMonitor(cell_mask, values['axial_profile'], axial_length,
                                      weight=np.where(cell_mask, weight, 0.0),
                                      axial_wavenumber=values['axial_wavenumber'], **common)
    else:
        monitor = ThermalNoiseMonitor(cell_mask, values['axial_profile'], axial_length,
                                      mode_pair=_mode_pair_request(values), **common)
    return solver.add_thermal_noise(monitor)


def _mode_pair_request(values):
    """ThermalNoiseMonitor mode_pair dict of a mode_pair / illumination weighting."""
    illuminated = values['weighting'] == 'illumination'
    return {'physics': values['mode_physics'],
            'incident': None if illuminated else (values['incident_selection'], values['incident_mode'],
                                                  values['incident_direction']),
            'collected': (values['collected_selection'], values['collected_mode'], values['collected_direction']),
            'illumination': values['illumination_physics'] if illuminated else None}


def _apply_particle_noise(ctx, solver, feature_spec, values, selection):
    """ThermalNoiseMonitor of dilute particles diffusing on the selected window cells (the liquid)."""
    mask = ctx.domain_mask(selection.entities)
    uniform = values['axial_profile'] == 'uniform'
    particle = {'index_increment': values['refractive_increment'] * values['molar_mass'] / N_A,
                'diffusivity': values['diffusivity'], 'number_density': values['concentration'] * N_A}
    monitor = ThermalNoiseMonitor(to_host(mask[solver.system.window]), values['axial_profile'],
                                  values['interaction_length'] if uniform else values['waist'],
                                  axial_points=values['axial_points'], mode_pair=_mode_pair_request(values),
                                  particle=particle)
    return solver.add_thermal_noise(monitor)


def _thermorefractive_noise_result(session, solver, monitor, feature_spec, options):
    return thermal_noise_result(monitor)


# ---- validation helpers (CPU only) ----
def _time_mode(vctx, physics_spec, settings_values=None):
    """Resolved time_mode; the raw setting when the settings do not resolve (reported by the param checks)."""
    values = settings_values if settings_values is not None else vctx.values(_SETTINGS, physics_spec.settings)
    if values:
        return values.get('time_mode')
    raw_settings = physics_spec.settings if isinstance(physics_spec.settings, dict) else {}
    return raw_settings.get('time_mode', 'stationary')


def _is_steady(study_type, time_mode):
    """True when heat is solved as a steady state (any study but time_dependent, or stationary time_mode)."""
    return study_type != 'time_dependent' or time_mode != 'transient'


def _study_text(vctx):
    """Lower-case label of the model's study type for messages (its name when the type is unknown)."""
    try:
        return vctx.registry.study_type(vctx.study_type).label.lower()
    except KeyError:
        return str(vctx.study_type)


def _validate_initial_temperature(vctx, physics_spec, feature_spec, values):
    if _is_steady(vctx.study_type, _time_mode(vctx, physics_spec)):
        return [Issue('info', feature_path(physics_spec, feature_spec),
                      'initial temperature only drives transient heat; a steady solve uses it as the starting '
                      'guess and as the temperature of non-conducting cells')]
    return []


def _has_anchor(vctx, physics_spec):
    """True when an enabled feature fixes the temperature level (temperature, h > 0 convection, fixed domain) or an
    antiperiodic phi wrap does (no constant mode)."""
    if antiperiodic_wrap(vctx):
        return True
    for feature in physics_spec.features:
        if not feature.enabled or feature.type not in _ANCHOR_FEATURES:
            continue
        if feature.type in _CONVECTIVE_FEATURES or feature.type == 'radiation':
            values = vctx.values(HEAT_PHYSICS.feature_type(feature.type).params, feature.params)
            strength = 'emissivity' if feature.type == 'radiation' else 'h'
            if values is not None and not values[strength] > 0.0:
                continue
        return True
    return False


def _has_all_cell_sink(vctx, physics_spec):
    """True when an enabled h > 0 out-of-plane convection selects 'all' (it bounds the CG condition number)."""
    for feature in physics_spec.features:
        selection = feature.selection
        if (not feature.enabled or feature.type != 'out_of_plane_convection'
                or not isinstance(selection, (list, tuple)) or 'all' not in selection):
            continue
        values = vctx.values(HEAT_PHYSICS.feature_type(feature.type).params, feature.params)
        if values is not None and values['h'] > 0.0:
            return True
    return False


def _heat_physics(spec):
    """PhysicsSpecs of every heat physics in the model, in spec order."""
    return [physics for physics in spec.physics if physics.type == PHYSICS_TYPE_NAME]


def _material_issues(vctx, physics_spec, path):
    """k / rho / cp / k_xy (xy^2 < xx*yy) checks over the spec materials, reported once (by the first heat physics);
    rho / cp are checked when any heat physics is transient."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    spec = vctx.spec
    if not reports_model_checks(vctx, physics_spec, PHYSICS_TYPE_NAME):
        return []
    transient = any(not _is_steady(vctx.study_type, _time_mode(vctx, heat_spec)) for heat_spec in _heat_physics(spec))
    parameters = vctx.parameters
    issues, conducting = [], []
    selected_conductor = False
    for material in spec.materials:
        try:   # as the builder merges it
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:   # unknown library / malformed properties (reported at the material): may conduct
            conducting.append(True)
            selected_conductor = selected_conductor or bool(material.selection)
            continue
        has_conductivity = conducts(properties, parameters, 'k')
        conducting.append(has_conductivity)
        if not material.selection:
            continue
        selected_conductor = selected_conductor or has_conductivity
        material_path = ('material', material.name)
        if not has_conductivity:
            issues.append(Issue('warning', material_path,
                                f"material '{material.name}' has no thermal conductivity (k unset or 0): heat does "
                                "not conduct there, and its cells keep their temperature and drop any heat deposited "
                                "there (unless transient heat with rho and cp set stores it)"))
        if transient:
            missing = [name for name in ('rho', 'cp') if name not in properties]
            if missing:
                issues.append(Issue('warning', material_path,
                                    f"material '{material.name}' sets no {' / '.join(missing)}: it has no heat "
                                    "capacity in transient heat (quasi-static there; a conducting region without "
                                    "any capacity needs a temperature anchor)"))
        if evaluates_to(properties.get('k_xy'), parameters, lambda k_xy: k_xy != 0.0):
            indefinite = indefinite_cross_term_issue(material.name, properties, parameters, 'k', 0.0)
            if indefinite is not None:
                issues.append(indefinite)

    fraction = conducting_fraction(vctx, conducting)
    # a conductor smaller than a preview cell can be missed by the raster: only a warning then
    if fraction == 0.0 and not selected_conductor:
        issues.append(Issue('error', path, 'no physical cell has a thermal conductivity: assign a material '
                                           'with k to the heat domain'))
    elif fraction is not None and fraction < 1.0:
        issues.append(Issue('warning', path,
                            f'{100.0 * (1.0 - fraction):.3g}% of the domain has no thermal '
                            'conductivity (no material with k): heat does not conduct there'))
    return issues


def _sync_step(vctx):
    """Smallest runner step [s] known at validation: the EM Courant step and the time_step of every transient heat
    physics (inf when there is none)."""
    steps = [vctx.mesh.dt] if getattr(vctx.mesh, 'dt', None) else []
    for heat_spec in _heat_physics(vctx.spec):
        values = vctx.values(_SETTINGS, heat_spec.settings)
        if values and not _is_steady(vctx.study_type, values['time_mode']) and values['time_step'] > 0.0:
            steps.append(values['time_step'])
    return min(steps, default=math.inf)


def _cadence_issues(vctx, physics_spec, settings, steady, path):
    """Stride checks: a transient heat step must not exceed time_step; a heat solve at (nearly) every EM step."""
    every = physics_spec.every
    if not isinstance(every, int):   # malformed every: reported by the generic checks
        return []
    time_step = settings.get('time_step')
    if not steady and time_step is not None and every > 1:
        dt_sync = min(time_step, _sync_step(vctx))
        if every * dt_sync > time_step * (1.0 + 1e-9):
            remedy = ('lower every or raise time_step' if dt_sync < time_step
                      else 'use every = 1 (time_step itself sets the heat step)')
            return [Issue('error', path, f'transient heat with every = {every} advances {every} * {dt_sync:.4g} s '
                                         f'per step, more than its time_step {time_step:.4g} s; {remedy}')]
    em_dt = getattr(vctx.mesh, 'dt', None)
    if vctx.study_type == 'time_dependent' and em_dt and every < MIN_EM_STEPS_PER_SOLVE:
        return [Issue('warning', path,
                      f'heat is solved every {every} EM step(s) ({every * em_dt:.3g} s), and one CG solve costs far '
                      'more than an EM step: set every to a few hundred (one heat solve per Joule averaging window)')]
    return []


def _three_d_issues(vctx, physics_spec, settings_values, path):
    """3D model rules: planar geometry only, boundary features on outer faces only (a tetrahedra mesh: also on object
    interfaces), no out-of-plane convection, no k_xy, no thermal_noise study on tetrahedra."""
    issues = []
    tetrahedra = on_tetrahedra(vctx, physics_spec)
    if is_axisymmetric(settings_values):
        issues.append(Issue('error', path, "geometry 'axisymmetric' needs a 2D model (this model is 3D: size_z is "
                                           "set); use geometry 'planar'"))
    if tetrahedra and vctx.study_type == THERMAL_NOISE_STUDY:
        issues.append(Issue('error', path, 'the thermal_noise study needs a planar 2D heat physics on a tensor mesh'))
    for feature in physics_spec.features:
        if not feature.enabled:
            continue
        if feature.type in _FACE_FEATURES and not tetrahedra and vctx.objects(feature.selection):
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                f"{feature.type} on object boundaries needs a 2D model or a tetrahedra mesh; on this "
                                '3D tensor mesh select outer faces (x_lo ... z_hi) only, and hold an object at a '
                                'temperature with a Fixed temperature (domain) feature'))
        elif feature.type == 'out_of_plane_convection':
            issues.append(Issue('error', feature_path(physics_spec, feature),
                                'out-of-plane convection stands in for the z faces of a 2D model; a 3D model has '
                                'them: use Convective cooling on z_lo / z_hi instead'))
    for material_name in cross_term_materials(vctx, 'k'):
        issues.append(Issue('error', ('material', material_name),
                            f"material '{material_name}' sets k_xy, which 3D heat '{physics_spec.name}' does not "
                            'support (no cross conduction in 3D): remove it'))
    return issues


def _selections_overlap(first, second):
    """True when two entity selections may share cells: either holds 'all', or they name a common entity."""
    first = set(first) if isinstance(first, (list, tuple)) else set()
    second = set(second) if isinstance(second, (list, tuple)) else set()
    return 'all' in first or 'all' in second or bool(first & second)


def _validate_thermorefractive_noise(vctx, physics_spec, feature_spec, values):
    """Study, planar 2D, the em_modes mode pair (type, mode numbers, no bend) and heat capacity under the selection."""
    from ...materials import effective_material_data   # lazy, as in _material_issues
    path = feature_path(physics_spec, feature_spec)
    issues = _noise_setup_issues(vctx, physics_spec, feature_spec, path, values)
    for material in vctx.spec.materials:
        if not material.selection or not _selections_overlap(material.selection, feature_spec.selection):
            continue
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except ValueError:   # reported at the material
            continue
        missing = [name for name in ('rho', 'cp')
                   if not evaluates_to(properties.get(name), vctx.parameters, lambda value: value > 0.0)]
        if missing:
            issues.append(Issue('error', path, f"material '{material.name}' may lie under the selection but sets no "
                                               f"positive {' / '.join(missing)}: a fluctuating cell without heat "
                                               'capacity has infinite temperature-noise variance; set rho and cp, or '
                                               'narrow the selection'))
    return issues


def _noise_setup_issues(vctx, physics_spec, feature_spec, path, values):
    """Shared noise-feature checks: thermal_noise study (info), planar 2D, the mode pair and the illumination, a
    warning for a mode number its selection ignores."""
    issues = []
    raw_params = feature_spec.params if isinstance(feature_spec.params, dict) else {}
    for role in ('incident', 'collected'):
        if f'{role}_mode' in raw_params and values.get(f'{role}_selection') not in (None, 'number'):
            issues.append(Issue('warning', path, f"{role}_mode is ignored: {role}_selection is "
                                                 f"'{values[f'{role}_selection']}' (set it to 'number' to use the "
                                                 'mode number, or remove the mode number)'))
    if vctx.study_type != THERMAL_NOISE_STUDY:
        issues.append(Issue('info', path, 'only a thermal_noise study computes the noise spectrum; this study leaves '
                                          'the result empty'))
    settings_values = vctx.values(_SETTINGS, physics_spec.settings)
    if vctx.spec.dimensions != 2 or is_axisymmetric(settings_values):
        issues.append(Issue('error', path, 'noise spectra need a planar 2D heat physics: the xy plane is the '
                                           'z-invariant cross-section'))
    if values['weighting'] in ('mode_pair', 'illumination'):
        issues.extend(_mode_pair_issues(vctx, path, values))
    if values['weighting'] == 'illumination':
        issues.extend(_illumination_issues(vctx, path, values))
    return issues


def _validate_particle_noise(vctx, physics_spec, feature_spec, values):
    """Study, planar 2D, the mode pair and the illumination (as thermorefractive_noise; no heat capacity needed)."""
    return _noise_setup_issues(vctx, physics_spec, feature_spec, feature_path(physics_spec, feature_spec), values)


def _mode_pair_issues(vctx, path, values):
    """The named physics is an em_modes physics without a bend whose n_modes covers the mode numbers given; a warning
    for both families under a mirror."""
    name = values['mode_physics']
    if name is None:
        return [Issue('error', path, "weighting 'mode_pair' needs mode_physics: the name of an em_modes physics")]
    mode_spec = next((physics for physics in vctx.spec.physics if physics.name == name), None)
    if mode_spec is None:   # reported by the generic ref check
        return []
    if mode_spec.type != _MODE_PHYSICS_TYPE:
        return [Issue('error', path, f"mode_physics '{name}' is a {mode_spec.type} physics; it must be an em_modes "
                                     'physics')]
    mode_values = vctx.values(vctx.registry.physics_type(_MODE_PHYSICS_TYPE).settings, mode_spec.settings)
    if not mode_values:   # reported at that physics
        return []
    issues = []
    for role in ('incident', 'collected') if values['weighting'] == 'mode_pair' else ('collected',):
        if values[f'{role}_selection'] == 'number' and values[f'{role}_mode'] > mode_values['n_modes']:
            issues.append(Issue('error', path, f"{role}_mode {values[f'{role}_mode']} exceeds n_modes "
                                               f"{mode_values['n_modes']} of '{name}'"))
    mirrors = [face for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')
               if mode_values.get(f'bc_{face}') in ('symmetric', 'antisymmetric')]
    if mirrors and values['collected_selection'] == 'te_and_tm':
        issues.append(Issue('warning', path, f"collected_selection te_and_tm with the mirror face(s) "
                                             f"{', '.join(mirrors)} of '{name}': a mirror keeps one mode family "
                                             'only, so one of the two finds no mode (run error); use a full domain '
                                             'or one family'))
    if mode_values['bend_radius'] != 0.0:
        issues.append(Issue('error', path, f"'{name}' solves a bent waveguide (bend_radius != 0), which is not "
                                           'z-invariant: thermal noise needs a straight one'))
    return issues


def _illumination_issues(vctx, path, values):
    """The illumination physics is an em_fdfd_ez / _hz physics at the mode physics' wavelength; an info on mirrors."""
    name = values['illumination_physics']
    if name is None:
        return [Issue('error', path, "weighting 'illumination' needs illumination_physics: the name of an em_fdfd_ez "
                                     'or em_fdfd_hz physics on the same cross-section')]
    illumination_spec = next((physics for physics in vctx.spec.physics if physics.name == name), None)
    if illumination_spec is None:   # reported by the generic ref check
        return []
    if illumination_spec.type not in _ILLUMINATION_PHYSICS_TYPES:
        return [Issue('error', path, f"illumination_physics '{name}' is a {illumination_spec.type} physics; it must be "
                                     f"one of {', '.join(_ILLUMINATION_PHYSICS_TYPES)}")]
    illumination_values = vctx.values(vctx.registry.physics_type(illumination_spec.type).settings,
                                      illumination_spec.settings)
    mode_spec = next((physics for physics in vctx.spec.physics if physics.name == values['mode_physics']), None)
    if not illumination_values or mode_spec is None or mode_spec.type != _MODE_PHYSICS_TYPE:
        return []   # reported at that physics / by the mode-pair checks
    mode_values = vctx.values(vctx.registry.physics_type(_MODE_PHYSICS_TYPE).settings, mode_spec.settings)
    if not mode_values:
        return []
    issues = []
    wavelength, mode_wavelength = illumination_values['wavelength'], mode_values['wavelength']
    if abs(wavelength - mode_wavelength) > 1e-9 * mode_wavelength:
        issues.append(Issue('error', path, f"illumination_physics '{name}' solves at {wavelength:.6g} m, mode_physics "
                                           f"'{mode_spec.name}' at {mode_wavelength:.6g} m: set equal wavelengths"))
    mirrors = [face for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')
               if mode_values.get(f'bc_{face}') in ('symmetric', 'antisymmetric')]
    if mirrors:
        issues.append(Issue('info', path, f"mirror face(s) {', '.join(mirrors)}: the illumination is mirrored with the "
                                          'parity of the collected mode, so only illumination of that parity is '
                                          'modelled (e.g. an x-cut quasi-TE mode with an antisymmetric x mirror '
                                          'pairs with an E_x-polarised em_fdfd_hz beam; an E_z-polarised em_fdfd_ez '
                                          'beam into it needs a full domain)'))
    return issues


def _thermorefractive_noise_memory(values, n_selected_cells, mesh):
    """Host: the window conduction operator copy (5 entries per cell), one complex SuperLU factor at a time, and the
    weighting / response vectors."""
    if mesh.kind != 'tensor':   # refused on element meshes (FeatureType.grids): nothing allocated
        return MemoryEstimate()
    cells = window_cell_count(mesh)
    return MemoryEstimate.from_items(host_items=(
        ('thermal noise operator (host CSR)', 5 * cells * (FLOAT64_BYTES + 4)),
        ('thermal noise SuperLU factor (host)', host_lu_bytes(cells)),
        ('thermal noise vectors (host)', 6 * cells * COMPLEX128_BYTES)))


def _triangle_issues(vctx, settings_values, path):
    """Rules of heat on a triangle mesh (FEMHeatPhysics, mesh plan T4): no azimuthal order m != 0, no thermal_noise
    study (its operator is the finite-volume one)."""
    issues = []
    if azimuthal_order_of(settings_values):
        issues.append(Issue('error', path, 'azimuthal_order != 0 needs a tensor mesh (rectangular cells): the '
                                           'finite-element heat on triangles solves order 0 only'))
    if vctx.study_type == THERMAL_NOISE_STUDY:
        issues.append(Issue('error', path, 'the thermal_noise study needs heat on a tensor mesh (rectangular cells): '
                                           'assign the heat physics a mesh of kind tensor'))
    return issues


def _validate_heat(vctx, physics_spec, settings_values):
    """Physics-level heat rules: anchors, materials, k_xy mesh, CG iteration limit, time mode, solve cadence."""
    path = ('physics', physics_spec.name)
    issues = []
    time_mode = _time_mode(vctx, physics_spec, settings_values)
    steady = _is_steady(vctx.study_type, time_mode)
    if time_mode == 'transient' and vctx.study_type != 'time_dependent':
        issues.append(Issue('info', path, f"time_mode 'transient' is ignored in a {_study_text(vctx)} study: heat "
                                          'is solved as a steady state (only a time-dependent study runs it '
                                          'transient)'))
    if (steady and vctx.study_type != THERMAL_NOISE_STUDY and not _has_anchor(vctx, physics_spec)
            and not azimuthal_order_of(settings_values)):   # an order m != 0 anchors by its m^2 / r^2 term
        issues.append(Issue('error', path,
                            'a steady heat solve needs a temperature anchor: add a Temperature or Convection '
                            'boundary condition, a Fixed temperature (domain) or an Out-of-plane convection '
                            'feature (otherwise the temperature level is undetermined)'))
    issues.extend(cg_solver_issues(vctx, physics_spec, path, all_cell_sink=_has_all_cell_sink(vctx, physics_spec)))
    issues.extend(_cadence_issues(vctx, physics_spec, settings_values, steady, path))
    issues.extend(face_conflicts(vctx, physics_spec, _BOUNDARY_FEATURES, path))
    issues.extend(mirror_face_issues(
        vctx, physics_spec, _FACE_FEATURES, path, _MIRROR_FACE_CONSEQUENCE,
        severity='error' if vctx.study_type == THERMAL_NOISE_STUDY else 'warning'))
    issues.extend(held_object_conflicts(vctx, physics_spec, _HELD_FEATURES, _PERIMETER_FEATURES, path))
    issues.extend(_material_issues(vctx, physics_spec, path))
    if vctx.spec.is_3d:
        issues.extend(_three_d_issues(vctx, physics_spec, settings_values, path))
    elif is_axisymmetric(settings_values):
        issues.extend(axisymmetric_issues(vctx, physics_spec, path, 'heat', cross_prefix='k'))
    issues.extend(azimuthal_order_issues(vctx, physics_spec, path, settings_values))
    issues.extend(mapped_domain_issues(vctx, physics_spec, _PERIMETER_FEATURES, 'k'))
    if on_triangles(vctx, physics_spec):
        issues.extend(_triangle_issues(vctx, settings_values, path))
    if vctx.spec.dimensions == 1:
        issues.extend(Issue('error', feature_path(physics_spec, feature),
                            'out-of-plane convection stands in for the z faces of a 2D model; a 1D model has no '
                            'such faces (a slab is infinite across, a cylinder along its axis): remove it')
                      for feature in physics_spec.features
                      if feature.enabled and feature.type == 'out_of_plane_convection')
    return issues


# ---- physics factory callbacks ----
def _create_heat(ctx, physics_spec, values):
    """HeatPhysics on ctx.domain (insulated faces until features apply) with the build's linear (pcg) and, in a
    thermal_noise study, shifted solvers; every study but time_dependent forces steady mode; thermal_noise needs no
    anchor (never CG-solved). Triangles: FEMHeatPhysics; tetrahedra: FEMHeatPhysics3D."""
    time_mode = values['time_mode'] if ctx.study_type == 'time_dependent' else 'stationary'
    solvers = ctx.solvers(physics_spec.name)
    time_step = values['time_step'] if time_mode == 'transient' else math.inf
    if ctx.domain.grid_kind == 'triangles':   # FEM (mesh plan T4): discretization order, no azimuthal order / noise
        return FEMHeatPhysics(ctx.domain, T0=values['T0'], order=element_order(values), time_mode=time_mode,
                              time_step=time_step, axisymmetric=is_axisymmetric(values),
                              linear_solver=solvers['linear'])
    if ctx.domain.grid_kind == 'tetrahedra':   # FEM 3D (fem.diffusion3d): discretization order, diagonal k
        return FEMHeatPhysics3D(ctx.domain, T0=values['T0'], order=element_order(values), time_mode=time_mode,
                                time_step=time_step, linear_solver=solvers['linear'])
    return HeatPhysics(ctx.domain, bcs=HeatBoundaryConfig(), T0=values['T0'], time_mode=time_mode,
                       time_step=time_step,
                       axisymmetric=is_axisymmetric(values), azimuthal_order=azimuthal_order_of(values),
                       require_anchor=ctx.study_type != THERMAL_NOISE_STUDY,
                       linear_solver=solvers['linear'], shifted_solver=solvers.get('shifted'))


def _steps_in_time(settings_values):
    return settings_values['time_mode'] == 'transient'


def _operator_follows_state(context, transient) -> bool:
    """Whether the run re-assembles and refactorises the heat operator: k (transient: rho, cp) follows a state or t."""
    return material_follows_state(context, ('k', 'k_xx', 'k_yy', 'k_zz', 'k_xy') + (('rho', 'cp') if transient else ()))


def _active_fraction(context, mesh, transient) -> float:
    """Element share of the materials joining the FEM heat system (k > 0; transient: rho and cp set; an out-of-plane
    convection loss: all), fem.diffusion.painted_fraction (1 without material_elements)."""
    from ...materials import effective_material_data   # lazy: physics -> materials stays out of module import time
    if any(feature.enabled and feature.type == 'out_of_plane_convection' for feature in context.physics_spec.features):
        return 1.0
    names = []
    for material in context.spec.materials:
        try:
            properties, _ = effective_material_data(material.library, material.properties, material.dispersion)
        except (ValueError, TypeError):   # reported by validate: may conduct
            names.append(material.name)
            continue
        if conducts(properties, context.parameters, 'k') or (transient and 'rho' in properties and 'cp' in properties):
            names.append(material.name)
    return painted_fraction(mesh, names)


def _memory_estimate(settings_values, mesh, context):
    """T, Q, Q_ext (float32 full grid) plus the DiffusionSystem on the window (k_xy: 9-point pattern; 3D: 7-point on
    the window voxels, 9 slots with a cylindrical sector's phi wrap) at the linear solver's precision with its flux
    post-processing (transient phase), a cuDSS 'linear' factor (fem.diffusion.cudss_diffusion_estimate) and the
    line_block noise solver's run items; element meshes: the FEM system on the active elements."""
    transient = context.spec.study.type == 'time_dependent' and settings_values['time_mode'] == 'transient'
    cudss = cudss_back_end(context.solvers)
    refactorizes = _operator_follows_state(context, transient)
    if mesh.kind in ('triangles', 'tetrahedra'):   # FEMHeatPhysics[3D]: T, Q, Q_ext, |q| per element + the FEM system
        estimator = fem_diffusion_estimate if mesh.kind == 'triangles' else fem_diffusion_estimate_3d
        return estimator(mesh, element_order(settings_values), element_fields=4, direct=linear_solver_direct(context),
                         multigrid=linear_solver_multigrid(context), cudss=cudss, refactorizes=refactorizes,
                         active_fraction=_active_fraction(context, mesh, transient))
    fields = MemoryEstimate.from_items((('T, Q, Q_ext (3 x float32)', 3 * FLOAT32_BYTES * int(mesh.n_cells)),))
    precision = linear_solver_precision(context)
    # flux: heat_flux_magnitude (renderer reads, snapshots, error_indicator) after a solve
    solver_options = dict(mixed=linear_solver_mixed(context), multigrid=linear_solver_multigrid(context),
                          direct=linear_solver_direct(context), flux=True)
    wrap = mesh.domain_map is not None and mesh.domain_map.angle_wrap != 'none'   # phi wrap: 2 more build slots
    cross_term = not context.spec.is_3d and context.material_sets('k_xy')
    if context.spec.is_3d:
        sides = mesh.npml_sides
        window_cells = window_cell_count(mesh) * (int(mesh.nz) - sides.get('z_lo', 0) - sides.get('z_hi', 0))
        system = diffusion_system_estimate(window_cells, precision, transient=transient, dimensions=3, wrap=wrap,
                                           **solver_options)
    else:
        window_cells = window_cell_count(mesh)
        system = diffusion_system_estimate(window_cells, precision, cross_term=cross_term, transient=transient,
                                           azimuthal=azimuthal_order_of(settings_values) != 0, wrap=wrap,
                                           **solver_options)
    parts = [fields, system, _line_block_noise_estimate(mesh, context)]
    if cudss:   # DiffusionDirectSolver factorises the kept A_solve
        row_entries = 7 if context.spec.is_3d else 9 if cross_term else 5
        parts.append(cudss_diffusion_estimate(window_cells, row_entries * window_cells,
                                              3 if context.spec.is_3d else 2, cudss, input_copy=False,
                                              refactorizes=refactorizes))
    return combine(parts)


# where heat holds T, for the Jacobi-PCG condition estimate: held temperatures, the convective losses counted as held
_HELD_TEMPERATURE = HeldValues(feature_types=frozenset({'temperature', 'fixed_temperature_domain', 'convection',
                                                        'out_of_plane_convection'}))


def _time_estimate(settings_values, mesh, context):
    """The DiffusionSystem recipe on the window (transient: time_step per step); triangles / tetrahedra: the 2D / 3D
    recipe at the Lagrange unknowns with their CSR entries per row (fem.diffusion.fem_time_system, uncalibrated) plus
    the host work per solve (fem_solve_host_ops); the thermal noise spectra are not estimated yet."""
    if context.study.type == THERMAL_NOISE_STUDY:
        return TimeEstimate(not_estimated=('thermal noise spectra',))
    transient = context.time_dependent and settings_values['time_mode'] == 'transient'
    held = None if transient else _HELD_TEMPERATURE   # implicit steps: the capacity term is not in the estimate
    # the operator is re-assembled only when k changes (a coupling writing Q leaves the build's factorisation)
    operator_changes = material_follows_state(context, ('k', 'k_xx', 'k_yy', 'k_zz', 'k_xy'))
    if mesh.kind in ('triangles', 'tetrahedra'):
        order = element_order(settings_values)
        unknowns, entries = fem_time_system(mesh, order)
        estimate = diffusion_time_estimate(context, unknowns, 3 if mesh.kind == 'tetrahedra' else 2,
                                           transient=transient, time_step=settings_values['time_step'], held=held,
                                           operator_changes=operator_changes, entries_per_row=entries)
        # the FEM system's host work per solve (loads, T at the centroids, transfers), counted like its rhs
        rhs = next(item for item in estimate.items if item.label == 'right-hand side')
        host = TimeItem('FEM host work per solve (loads, T, transfers)', 'solve', rhs.count, rhs.count_kind,
                        fem_solve_host_ops(mesh, order, transient), limit=rhs.limit, per_step=rhs.per_step,
                        counter=rhs.counter)
        return combine_time((estimate, TimeEstimate((host,))))
    return diffusion_time_estimate(context, context.window_cells(), 3 if context.spec.is_3d else 2,
                                   cross_term=context.material_sets('k_xy'), transient=transient,
                                   time_step=settings_values['time_step'], held=held,
                                   operator_changes=operator_changes,
                                   face_value_updates=time_following_values(context.physics_spec))


_NOISE_FEATURE_TYPES = ('thermorefractive_noise', 'particle_noise')


def _count_or(value, default):
    """A literal count setting as int, else (unset, parameter name, invalid: validate reports it) the default."""
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return default


def _line_block_noise_estimate(mesh, context):
    """Run items of a line_block shifted solver (thermal_noise study): operator caches over the whole window (lines
    along the longer axis) and one chunk of the largest channel (n_frequencies x axial samples, capped)."""
    study = context.spec.study
    noise_features = [feature for feature in context.physics_spec.features
                      if feature.enabled and feature.type in _NOISE_FEATURE_TYPES]
    shifted = context.solvers.get('shifted')   # present in a thermal_noise study only
    if shifted is None or shifted.solver_type.name != 'line_block' or not noise_features:
        return MemoryEstimate()
    sides = mesh.npml_sides
    window = (int(mesh.nx) - sides['x_lo'] - sides['x_hi'], int(mesh.ny) - sides['y_lo'] - sides['y_hi'])
    n_blocks, block_size = max(window), min(window)
    cells = window[0] * window[1]
    study_settings = study.settings if isinstance(study.settings, dict) else {}
    frequency_list = study_settings.get('frequencies') or []
    n_frequencies = (len(frequency_list) if isinstance(frequency_list, list) and frequency_list
                     else _count_or(study_settings.get('n_frequencies'), 31))
    feature_params = [feature.params if isinstance(feature.params, dict) else {} for feature in noise_features]
    samples = max(_count_or(params.get('axial_points'), 8) if params.get('axial_profile') == 'gaussian' else 1
                  for params in feature_params)
    n_rhs = len(context.spec.materials) + 1   # contribution parts + 'other' at most
    precision = shifted.values['precision']
    entries =(9 if context.material_sets('k_xy') else 5) * cells
    chunk = min(MAX_CHUNK_SHIFTS, max(n_frequencies * samples, 1))
    return MemoryEstimate.from_items(
        run_items=(('thermal noise line_block operator', line_block_operator_bytes(n_blocks, block_size, cells, entries,
                                                                                   2, precision)),),
        transient_items=(('thermal noise chunk', f'thermal noise line_block chunk ({chunk} shifts)',
                          chunk * line_block_shift_bytes(n_blocks, block_size, cells, n_rhs, precision)),))


_MODE_PAIR = (('weighting', ('mode_pair',)),)
_ILLUMINATION = (('weighting', ('illumination',)),)
_COLLECTED = (('weighting', ('mode_pair', 'illumination')),)
_EXPRESSION = (('weighting', ('expression',)),)
_NOISE_PARAMS = (
    Param('weighting', 'choice', 'mode_pair', label='Weighting', choices=WEIGHTINGS,
          choice_labels=(('mode_pair', 'Scattering between two waveguide modes'),
                         ('illumination', 'Illumination scattered into a waveguide mode'),
                         ('expression', 'Expression')),
          help="mode_pair: X = amplitude scattered into collected_mode per incident amplitude (1 W modes of "
               "mode_physics, Lorentz reciprocity): w = omega eps0 / 4 sum_i 2 n_i dndT_i E_inc,i E_partner,i with the "
               "partner mode travelling against the collected direction (backward collection of a forward mode: "
               "E_t . E_t + E_z E_z of the same forward mode, i.e. |E_t|^2 - |E_z|^2; forward collection of the same "
               "mode: |E|^2, then X is the thermorefractive phase noise in rad), q0 = |beta_inc +- beta_coll|; n_i = "
               "sqrt(Re eps_ii) (eps_xx, eps_yy, eps_r = eps_zz), dndT_xx / _yy / _zz of the heat materials. "
               "illumination: the same with E_inc = the field [V/m] of illumination_physics (an em_fdfd_ez / _hz "
               "physics solving the same cross-section at k_z = 0, e.g. a gaussian_beam from the top): X = amplitude "
               "[W^0.5] scattered into collected_mode for that illumination, q0 = beta_coll, S in W/Hz (scales with "
               "the source amplitude squared). expression: w = weight (x, y), q0 = axial_wavenumber.",
          tip='What the noise is measured on.\n\n'
              '- Scattering between two waveguide modes: light of one mode scattered by temperature fluctuations '
              'into another (or the same) mode, using the modes of a mode-analysis physics.\n'
              '- Illumination scattered into a waveguide mode: light of a frequency-domain wave physics (for example '
              'a beam from the top) scattered into a waveguide mode.\n'
              '- Expression: a weighting function you give.'),
    Param('illumination_physics', 'ref', None, label='Illumination physics', ref_kind='physics', optional=True,
          visible_if=_ILLUMINATION,
          help='Name of the em_fdfd_ez or em_fdfd_hz physics whose solution is the incident field: it solves the '
               'cross-section (xy, light along z uniform: normal incidence in the cross-section plane, k_z = 0) as a '
               'background physics of the thermal_noise study, at the wavelength of mode_physics and with its '
               'boundary settings. em_fdfd_ez = E along the waveguide (E_z), em_fdfd_hz = E in the cross-section '
               "(E_x, E_y). With a mirror face the illumination has the collected mode's parity (validation info).",
          tip='The frequency-domain wave physics that computes the illuminating light on the same cross-section.'),
    Param('mode_physics', 'ref', None, label='Mode physics', ref_kind='physics', optional=True,
          visible_if=_COLLECTED,
          help='Name of the em_modes physics (same model, straight waveguide) whose modes give the weighting; the '
               'thermal_noise study solves it first. A mirror face lies on the window edge (its em ghost cell is '
               'outside the window), where an insulated heat face is the matching even mirror (the whole mirrored '
               'cross-section counts). The outermost cells of its pec / pmc faces are ghosts outside the wall plane '
               '(weighting 0) that keep the heat condition of that face (e.g. a temperature sink), on the window '
               'edge one cell outside the em wall plane.',
          tip='The waveguide mode physics whose modes scatter or collect the light.'),
    Param('incident_selection', 'choice', 'collected', label='Incident mode', choices=INCIDENT_SELECTIONS,
          choice_labels=(('collected', 'Same as each collected mode'), ('te', 'Fundamental quasi-TE'),
                         ('tm', 'Fundamental quasi-TM'), ('number', 'By mode number')),
          visible_if=_MODE_PAIR,
          help="Incident mode of a mode_pair weighting: collected = the collected mode itself (backscatter / phase "
               "noise of each collected mode), te / tm = the fundamental quasi-TE / quasi-TM mode (the highest-ranked "
               "mode with te_fraction > 0.5 / <= 0.5), number = incident_mode.",
          tip='Which waveguide mode carries the incoming light: the collected mode itself, the fundamental mode of '
              'one polarisation, or a mode number.'),
    Param('incident_mode', 'int', 1, label='Incident mode number', min=1,
          visible_if=(('incident_selection', ('number',)),),
          help='Mode number (em_modes rank, 1 = highest Re n_eff) of the incident light when incident_selection is '
               'number; <= n_modes of mode_physics.',
          tip='Number of the waveguide mode that carries the incoming light (1 = highest effective index).'),
    Param('incident_direction', 'choice', 'forward', label='Incident direction', choices=DIRECTIONS,
          choice_labels=(('forward', 'Forward (+z)'), ('backward', 'Backward (-z)')),
          visible_if=_MODE_PAIR,
          help='Travel direction of the incident mode along z (backward: E_z sign flipped, beta -> -beta).',
          tip='Direction along the waveguide in which the incoming light travels.'),
    Param('collected_selection', 'choice', 'te', label='Collected mode', choices=COLLECTED_SELECTIONS,
          choice_labels=(('te', 'Fundamental quasi-TE'), ('tm', 'Fundamental quasi-TM'),
                         ('te_and_tm', 'Both fundamental modes'), ('number', 'By mode number')),
          visible_if=_COLLECTED,
          help='Collected (detected) mode(s): te / tm = the fundamental quasi-TE (E_x dominant) / quasi-TM mode, i.e. '
               'the highest-ranked mode (largest Re n_eff) with te_fraction > 0.5 / <= 0.5 (run error when the solve '
               'found none: raise n_modes; a mirror face keeps one family only); te_and_tm = both in one run, each '
               'its own spectrum (series psd_te*, psd_tm*, extras with suffix _te / _tm; one set of factorisations '
               'per mode, as their beta differ); number = collected_mode. The resolved mode number, n_eff and '
               'te_fraction are in the result (collected_mode, collected_n_eff, collected_te_fraction).',
          tip='Which waveguide mode collects the scattered light: the fundamental mode of one polarisation, both of '
              'them (one spectrum each, about twice the time), or a mode number.'),
    Param('collected_mode', 'int', 1, label='Collected mode number', min=1,
          visible_if=(('collected_selection', ('number',)),),
          help='Mode number (em_modes rank, 1 = highest Re n_eff) of the collected light when collected_selection is '
               'number; <= n_modes of mode_physics.',
          tip='Number of the waveguide mode that collects the scattered light (1 = highest effective index).'),
    Param('collected_direction', 'choice', 'backward', label='Collected direction', choices=DIRECTIONS,
          choice_labels=(('forward', 'Forward scattering'), ('backward', 'Backscattering')),
          visible_if=_COLLECTED,
          help='Travel direction of the collected light: backward = backscattering (q0 = beta_inc + beta_coll), '
               'forward = forward scattering (q0 = |beta_inc - beta_coll|, 0 for the same mode); illumination: q0 = '
               'beta_coll either way (only the sign of the E_z cross term differs).',
          tip='Direction in which the collected light travels: backward for light scattered back toward the source.'),
    Param('weight', 'float', 1.0, label='Weighting function', unit='1/(K m^3)', field_dependent=True,
          visible_if=_EXPRESSION,
          help='w(x, y): X = integral w dT dV (per unit z length times the axial profile); X has the units of w '
               'times K m^3, the PSD those squared per Hz. Evaluated at the cell centres; may depend on x and y.',
          tip='Weighting of the temperature change at each point; the noise is computed for the weighted '
              'integral of the temperature.'),
    Param('axial_wavenumber', 'float', 0.0, label='Axial wavenumber', unit='1/m', min=0.0, visible_if=_EXPRESSION,
          help='q0 of the weighting phase exp(-j q0 z) along the waveguide axis; 0 = uniform along z.',
          tip='How fast the weighting oscillates along the waveguide (radians per metre); 0 for none.'),
    Param('axial_profile', 'choice', 'uniform', label='Axial profile', choices=AXIAL_PROFILES,
          choice_labels=(('uniform', 'Uniform over a length'), ('gaussian', 'Gaussian spot')),
          help='Envelope g(z) of the weighting along z. uniform: g = 1 over interaction_length L, S = L S_2D(q0) '
               '(asymptotic in L: exact once L >> 1/q0 and the thermal diffusion length sqrt(D / omega); at q0 = 0 '
               'and low f the finite-L integral is the right one, use gaussian). gaussian: g = exp(-z^2 / waist^2) '
               '(a Gaussian illumination spot of 1/e^2 intensity radius waist), S = integral dq/2pi |g~(q - q0)|^2 '
               'S_2D(q) by axial_points-point Gauss-Hermite (exact for any waist).',
          tip='How the illuminated part of the waveguide is shaped along its length.'),
    Param('interaction_length', 'float', 1e-4, label='Interaction length', unit='m', min=0.0, min_exclusive=True,
          visible_if=(('axial_profile', ('uniform',)),),
          help='Length L along z over which the weighting acts (uniform profile); the PSD scales with L.',
          tip='Length of waveguide over which the light interacts with the fluctuations.'),
    Param('waist', 'float', 1e-5, label='Spot radius', unit='m', min=0.0, min_exclusive=True,
          visible_if=(('axial_profile', ('gaussian',)),),
          help='w0 of g(z) = exp(-z^2 / w0^2) (gaussian profile); integral |g|^2 dz = w0 sqrt(pi / 2).',
          tip='Radius along the waveguide of a Gaussian illumination spot (where the intensity falls to 13.5 %).'),
    Param('axial_points', 'int', 8, label='Axial samples', min=1, max=64,
          visible_if=(('axial_profile', ('gaussian',)),),
          help='Gauss-Hermite nodes of the q integral (gaussian profile); each costs one factorisation per frequency '
               '(nodes with equal |q| are merged, so q0 = 0 halves the cost). 6-12 typical.',
          tip='Number of samples of the spot spectrum; each costs one solve per frequency.'),
)
_NOISE_DESCRIPTION = (
    'One-sided power spectral density of X = integral w(x, y) g(z) exp(-j q0 z) dT dV over the selected cells '
    '(thermal_noise study only; elsewhere the result stays empty): S_X(f) = 4 k_B T0^2 integral dq/2pi |g~(q - q0)|^2 '
    'Re sum conj(w) u dA, (j 2 pi f rho cp + k_zz q^2 - div k grad) u = w on the heat window. Assumptions: uniform '
    'temperature T0, z-invariant cross-section, dT of the isobaric (entropy) fluctuations only (pressure / Brillouin '
    'fluctuations and the sound speed are not modelled), dndT the measured isobaric coefficient. Held cells (temperature '
    'on objects, fixed_temperature_domain) and Dirichlet faces do not fluctuate; insulated faces act as mirrors. '
    'mode_pair / illumination: an em mirror face lies on the window edge (keep the heat face there insulated: the even '
    'mirror) and doubles the result (the whole mirrored cross-section counts); w = 0 on the ghost cells outside em '
    'pec / pmc wall planes, which keep the heat boundary condition of that face. Every '
    'selected cell with w != 0 needs rho and cp > 0 (infinite variance otherwise: validation and run error). '
    'mode_pair: X = complex amplitude scattered into the collected mode per incident amplitude, both modes normalised '
    'to 1 W, so S is the PSD of the scattered field in (power fraction) / Hz and |X|^2 the scattered power per incident '
    'power; a heterodyne beat with a local oscillator of power P_LO carries 4 P_LO P_in S per Hz of optical power '
    'squared. illumination: X = amplitude [W^0.5] scattered into the collected mode by the FDFD illumination as '
    'solved (its source amplitude sets the scale; g(z) multiplies it along the waveguide), S in W/Hz: the PSD of the '
    'collected scattered field; the static (mean) scattering of the structure is not part of it. Units of S: 1/Hz '
    '(mode_pair), W/Hz (illumination) or (w K m^3)^2 / Hz. Result: xy over the study frequencies, series psd '
    '(all selected cells combined) and psd_db (10 log10 psd). When the selection covers several materials, also '
    'psd_<material> (self term: w on that material\'s cells alone; psd_other for weighted cells in no material), '
    'psd_cross = psd - sum of the self terms (interference between materials, may be negative; its integral over '
    'all f is 0) and psd_<material>_db; a material name that makes two series collide is a run error. Extra keys '
    'variance (exact integral of psd over all f: k_B T0^2 integral |w g|^2 / (rho cp) dV, the equal-time variance of '
    'X), contribution_variances (the same per material, summing to variance), static_variance (the part of variance '
    'that never relaxes: at q = 0 the mean dT of a connected region without temperature anchor (no Dirichlet / h > 0 '
    'face, held cell or volumetric loss) is conserved, a delta at f = 0 outside every band; 0 otherwise), '
    'band_fraction (share of the relaxing part variance - static_variance inside the solved band, trapezoid in ln '
    'f), axial_wavenumber q0 [1/m], mirror_factor, axial_norm (integral '
    '|g|^2 dz [m]) and, for mode_pair / illumination, wavelength [m] and the resolved collected_mode (number), '
    'collected_n_eff, collected_te_fraction (and incident_mode). collected_selection te_and_tm solves both fundamental '
    'modes in one run: every series gets the prefix psd_te / psd_tm instead of psd (psd_te_<material>, psd_tm_cross, '
    '...) and every extra key the suffix _te / _tm. Cost: one host SuperLU of the window per frequency, axial sample '
    'and collected mode (each material adds one back substitution).')

_SHARED_NOISE_PARAMS = {param.name: param for param in _NOISE_PARAMS}
_PARTICLE_PARAMS = (
    Param('weighting', 'choice', 'illumination', label='Weighting', choices=('illumination', 'mode_pair'),
          choice_labels=(('illumination', 'Illumination scattered into a waveguide mode'),
                         ('mode_pair', 'Scattering between two waveguide modes')),
          help='As thermorefractive_noise, with dndT_i replaced by the isotropic index increment per particle dn/dc * '
               'M / N_A: illumination: X = amplitude [W^0.5] scattered into collected_mode, S in W/Hz; mode_pair: X = '
               'amplitude per incident amplitude (1 W modes), S in 1/Hz.',
          tip='Which light the particles scatter: a frequency-domain illumination (for example a beam from the top) '
              'into a waveguide mode, or one waveguide mode into another (or itself, backward).'),
    *(_SHARED_NOISE_PARAMS[name] for name in ('illumination_physics', 'mode_physics', 'incident_selection',
                                              'incident_mode', 'incident_direction', 'collected_selection',
                                              'collected_mode', 'collected_direction')),
    Param('refractive_increment', 'float', 1.85e-4, label='Refractive index increment', unit='m^3/kg', min=0.0,
          min_exclusive=True,
          help='dn/dc of the particle in the liquid [m^3/kg] (1 mL/g = 1e-3 m^3/kg); proteins ~1.85e-4 (0.185 mL/g). '
               'With molar_mass it sets the index increment per particle dn/dc * M / N_A [m^3] (the excess '
               'polarisability alpha / eps0 = 2 n_liquid times it).',
          tip='Change of refractive index per mass concentration of the particle (dn/dc).\n\nProteins: about '
              '1.85e-4 m^3/kg (0.185 mL/g).'),
    Param('molar_mass', 'float', 66.5, label='Molar mass', unit='kg/mol', min=0.0, min_exclusive=True,
          help='Molar mass M of the particle [kg/mol] (= kDa); BSA 66.5.',
          tip='Molar mass of the particle in kg/mol (equal to kDa). BSA: 66.5.'),
    Param('diffusivity', 'float', 5.9e-11, label='Diffusion coefficient', unit='m^2/s', min=0.0, min_exclusive=True,
          help='Translational diffusion coefficient D of the particle in the liquid; Stokes-Einstein k_B T / (6 pi '
               'eta R_h). BSA in water at 20-25 C: 5.9e-11 to 6.1e-11 m^2/s.',
          tip='How fast the particle diffuses. BSA in water: about 6e-11 m^2/s.'),
    Param('concentration', 'float', 1e-6, label='Concentration', unit='mol/m^3', min=0.0, min_exclusive=True,
          help='Mean molar concentration c of the particles [mol/m^3] (1 nM = 1e-6 mol/m^3); n0 = c N_A per m^3. The '
               'PSD scales with c; psd_single does not depend on it.',
          tip='Mean particle concentration in mol/m^3 (1 nM = 1e-6 mol/m^3). The spectrum scales with it.'),
    *(_SHARED_NOISE_PARAMS[name] for name in ('axial_profile', 'interaction_length', 'waist', 'axial_points')),
)
_PARTICLE_DESCRIPTION = (
    'Scattering spectrum of dilute particles (e.g. proteins) diffusing in the liquid on the selected cells '
    '(thermal_noise study only; hosted by heat for its noise solver, independent of the heat materials): one-sided PSD '
    'of X = integral w(x, y) g(z) exp(-j q0 z) dc dV, dc the number-density fluctuation of independent particles '
    '(Poisson, <dc dc\'> = n0 delta, n0 = c N_A). S_X(f) = 4 n0 integral dq/2pi |g~(q - q0)|^2 Re sum conj(w) u dA with '
    '(j 2 pi f + D q^2 - div D grad) u = w on the selected cells, which the particles cannot leave (no flux through '
    'the selection edge or the window faces, which an em mirror plane shares). w is the mode-pair / '
    'illumination weighting of thermorefractive_noise with dndT_i -> dn/dc * M / N_A, i.e. the scattered amplitude of '
    'one particle at (x, y) (first Born / dipole approximation, isotropic polarisability, local field of the liquid). '
    'Lineshape: about D (q0^2 + 1/d^2) / (2 pi), d the evanescent decay length (BSA, q0 = beta: ~1 kHz). The '
    'coherent mean signal n0 integral w g exp(-j q0 z) dV (zero for a spot much longer than 1/q0) is not part of S. '
    'Not modelled: hindered diffusion near walls, adsorption / binding, particle interactions. Result: xy as '
    'thermorefractive_noise (psd, psd_db; prefix psd_te / psd_tm and key suffix _te / _tm with collected_selection '
    'te_and_tm) plus psd_single = psd / (n0 V_s): the time-averaged PSD of one particle '
    'diffusing within the sensing volume V_s = (integral |w g|^2 dV)^2 / integral |w g|^4 dV. Extra keys as '
    'thermorefractive_noise plus bound_signal_max (|w|^2 of one particle held at the strongest cell at the spot '
    'centre: the static signal of a bound particle, W or power fraction) at (bound_signal_x, bound_signal_y) [m], '
    'sensing_volume [m^3], particles_in_sensing_volume (n0 V_s), index_increment [m^3] and number_density [1/m^3]. '
    'Cost: one host SuperLU of the selected cells per frequency and axial sample.')


# ---- type ----
_FEATURES = (
    FeatureType('temperature', 'Temperature', 'boundary_condition', 'boundary',
                params=(_temperature_param('Held at the selected outer faces (on the face itself) or in the '
                                           'boundary cells inside the selected objects. ' + _TIME_VALUE_TEXT
                                           + ' A temperature of t acts on outer faces only (validation error for '
                                             'objects in the selection).',
                                           'Temperature held on the selected domain edges, or in the boundary cells '
                                           'just inside the selected objects.\n\n'
                                           'On domain edges it may change in time: write it as an expression of t.',
                                           time_dependent=True, follows_parameters=True),),
                summary='Holds the selected domain edges or object boundaries at a fixed temperature. A steady heat '
                        'solve needs at least one fixed temperature or cooling feature.',
                description='Fixed temperature (Dirichlet). On outer faces T is imposed at the face (half-cell '
                            'conductance 2 k / dx); on object perimeters the boundary cells inside the object are held '
                            'at T, imposed on their faces toward the free cells as on an outer face (the window edge '
                            'is no perimeter: select the outer face there). Unreferenced outer '
                            'faces are insulated, and a later boundary feature on the same face wins (validation '
                            'warning); on cells held by two features the later one wins, and held cells ignore heat '
                            'flux / convection (validation warns when their objects may overlap). Anchors a steady '
                            'solve. With solver_defaults.conformal_boundaries (2D) a perimeter holds T on the drawn '
                            'outline instead (conductance k / (t d) along each link from outside, t >= 0.02).',
                apply=_apply_temperature, validate=_validate_temperature, grids=_FEM_GRIDS),
    FeatureType('heat_flux', 'Heat flux', 'boundary_condition', 'boundary',
                params=(Param('flux', 'float', 1e4, label='Inward heat flux', unit='W/m^2', time_dependent=True,
                              help='> 0 heats the domain (outer faces) or the object (perimeters), < 0 extracts heat. '
                                   'Typical 1e3-1e6 (sunlight ~1e3). ' + _TIME_VALUE_TEXT,
                              tip='Heat flow per area into the domain or object; negative values extract heat.\n\n'
                                  'Typical 1e3 to 1e6; sunlight is about 1e3.'),),
                summary='Feeds a given heat flow per area in through the selected domain edges or object boundaries.',
                description='Prescribed heat flux, > 0 into the domain or the object. On outer faces a Neumann '
                            'condition; on object perimeters it heats the boundary cells inside the object by flux * '
                            'perimeter length / cell area (each exposed cell face weighted by its normal component, so '
                            'slanted and curved perimeters get their true length to ~2 %). Does not anchor a steady '
                            'solve. Polar / spherical domains weight each exposed face by its metric area. With '
                            'solver_defaults.conformal_boundaries (2D: Cartesian, (r, z), polar, spherical) the drawn '
                            'outline is used instead: each cut cell feeds flux * its chord of the outline into the '
                            'object cell owning it (exact for straight edges; arcs about a polar origin too), '
                            'whatever the cut-cell volumes.',
                apply=make_apply_inward_flux('flux'), grids=_FEM_GRIDS),
    FeatureType('convection', 'Convective cooling', 'boundary_condition', 'boundary',
                params=(_heat_transfer_coefficient_param(), _ambient_temperature_param()),
                summary="Cools the selected domain edges or object boundaries toward a surrounding fluid at 'Ambient "
                        "temperature'.",
                description='Convective loss h (T - T_ext) per unit area. On outer faces a Robin condition in series '
                            'with the half cell; on object perimeters the boundary cells inside the object lose h * '
                            'perimeter length / cell area * (T - T_ext) (each exposed cell face weighted by its normal '
                            'component, so slanted and curved perimeters get their true length to ~2 %), on top of '
                            'conduction to neighbouring cells (polar / spherical domains: the metric face areas). '
                            'Anchors a steady solve when h > 0. With solver_defaults.conformal_boundaries (2D: '
                            'Cartesian, (r, z), polar, spherical) on the drawn outline '
                            'instead: each cut cell loses h L / (1 + h delta / k_n) * (T - T_ext) from the object cell '
                            'owning it, L its chord of the outline, delta the distance from that cell centre to the '
                            'chord, k_n the normal conductivity at build time.',
                apply=_apply_convection, grids=_FEM_GRIDS),
    FeatureType('radiation', 'Surface-to-ambient radiation', 'boundary_condition', 'boundary',
                params=(Param('emissivity', 'float', 0.9, label='Surface emissivity', unit='1', min=0.0, max=1.0,
                              help='Hemispherical emissivity eps of the surface (grey body): oxidised metals and '
                                   'most dielectrics 0.8-0.95, silicon ~0.6, polished metals 0.02-0.1.',
                              tip='How strongly the surface radiates compared with a black body: 1 = black, polished '
                                  'metal about 0.05.'),
                        Param('T_amb', 'float', 293.15, label='Ambient temperature', unit='K', min=0.0,
                              min_exclusive=True,
                              help='Temperature of the surroundings the surface exchanges radiation with (a large '
                                   'enclosure, e.g. the vacuum chamber walls).',
                              tip='Temperature of the surroundings that the surface radiates to.')),
                summary='Lets the selected domain edges or object boundaries radiate heat to surroundings at a given '
                        'temperature, as a hot chip or a MEMS device in vacuum does.',
                description='Grey-body radiation to an ambient enclosure: a loss eps sigma (T^4 - T_amb^4) per unit '
                            'area (sigma = 5.67e-8 W/(m^2 K^4)) from the boundary cells of the selected outer faces '
                            '(their face area) and object perimeters (staircase perimeter weights, also under '
                            'conformal_boundaries). It adds to the face condition (Convective cooling on the same '
                            'face: both act; on a Temperature face it is pointless) and uses the boundary cell '
                            'temperature (error ~ h_r dx / (2 k), h_r = 4 eps sigma T^3 ~ 5 W/(m^2 K) at 300 K). '
                            'Newton-linearised at the current temperature: a steady solve re-linearises and re-solves '
                            'until the radiating temperatures change by < 1e-7 (relative; at most 50 solves, then a '
                            'run warning), a transient step uses the temperature at its start. Anchors a steady '
                            'solve when the emissivity is > 0. On triangles / tetrahedra: a Robin term per boundary '
                            'edge / face linearised at its midpoint / centroid temperature (each re-linearisation '
                            're-assembles and refactorises). No view factors or surface-to-surface exchange.',
                apply=_apply_radiation, grids=_FEM_GRIDS),
    FeatureType('heat_source', 'Heat source', 'domain_condition', 'domain',
                params=(Param('Q', 'float', 1e12, label='Power density', unit='W/m^3', field_dependent=True,
                              time_dependent=True, follows_parameters=True,
                              help='Heat per unit volume; in 2D, Q times the area is the power per unit z length '
                                   '(W/m). May depend on x and y (3D models: also z); must be finite on the selected '
                                   'cells. Or (not both) on time t: ' + _TIME_VALUE_TEXT,
                              tip='Heat generated per volume. The model is per metre of depth: power density times '
                                  'area gives watts per metre.\n\n'
                                  'A steady solve ignores it in cells that neither conduct nor lose heat.'),),
                summary='Adds heat generated throughout the selected domains.',
                description='Applied volumetric heat source: Q is added to Q_ext on the selected cells (several '
                            'sources add; the coupled source Q of em_heat_joule is separate). A steady solve ignores '
                            'it on cells that neither conduct nor lose heat.',
                apply=_apply_heat_source, grids=_FEM_GRIDS),
    FeatureType('point_heat_source', 'Point heat source', 'source', 'point',
                params=(Param('P', 'float', 1.0, label='Power', unit='W/m', time_dependent=True,
                              follows_parameters=True,
                              space_units=(('axisymmetric', 'W'), ('3d', 'W')),
                              help='Planar 2D: power per unit z length in W/m (a 2D point is a line source along z), '
                                   'deposited in the '
                                   'cell containing the point (Q_ext += P / cell area; a point exactly on a cell edge '
                                   'goes to the cell on its +x / +y side). The temperature of that cell rises as the '
                                   'cell shrinks (logarithmic singularity of a line source). With geometry '
                                   "'axisymmetric' P is the ring's total power in W instead (a point on the axis "
                                   'cell: a point source); in a 3D model P is the total power in W of a point source '
                                   "at the point's height (Q_ext += P / cell volume). " + _TIME_VALUE_TEXT,
                              tip='Power per metre of depth, deposited in the cell that contains the point. In an '
                                  'axisymmetric or 3D model it is the total power in watts instead.\n\n'
                                  'The temperature at the point keeps rising as the mesh is refined, as for any ideal '
                                  'line source.'),),
                object_kinds=('point',),
                summary='Heat source at a point, which in 2D is a line running through the depth of the model.',
                description='Line heat source along z (a point in 2D): P deposited in the cell containing each point; '
                            'a coordinate exactly on a cell edge goes to the cell on its +x / +y side. Points outside '
                            'the physical domain are an error.',
                apply=_apply_point_heat_source, validate=validate_points_inside, grids=_FEM_GRIDS),
    FeatureType('fixed_temperature_domain', 'Fixed temperature (domain)', 'domain_condition', 'domain',
                params=(_temperature_param('Temperature every selected cell is held at for the whole run.',
                                           'Temperature the selected domains are held at for the whole run.'),),
                summary='Holds the whole selected domains at a fixed temperature, like an ideal heat sink or '
                        'reservoir.',
                description='Holds every selected cell at a fixed temperature for the whole run (the cells leave the '
                            'solve, heat flows in or out freely): an ideal reservoir whose temperature sits on its '
                            'faces toward the free cells (half-cell conductance 2 k / dx of the free neighbour, '
                            "whatever the held cells' own k, so a k = 0 sink still conducts); the heat_flux_abs field reads 0 "
                            'inside it. Anchors a steady solve. With solver_defaults.conformal_boundaries (2D) the '
                            'temperature sits on the drawn edge instead: conductance k / (t d) of the free neighbour, '
                            't d the distance from its centre to the edge along the link (t >= 0.02).',
                apply=_apply_fixed_temperature_domain, grids=_FEM_GRIDS),
    FeatureType('out_of_plane_convection', 'Out-of-plane convection', 'domain_condition', 'domain',
                params=(_heat_transfer_coefficient_param(), _ambient_temperature_param(),
                        Param('thickness', 'float', 1e-6, label='Layer thickness', unit='m', min=0.0,
                              min_exclusive=True,
                              help='Out-of-plane (z) thickness of the slab the 2D model represents; the loss per '
                                   'unit volume scales as 1 / thickness, so thin slabs cool strongly.',
                              tip='Thickness of the layer the 2D model stands for, measured out of the plane.\n\n'
                                  'Thinner layers cool more strongly.'),
                        Param('sides', 'int', 2, label='Cooled sides', min=1, max=2,
                              help='Number of cooled slab faces: 1 = one face (the other insulated), 2 = top and '
                                   'bottom.',
                              tip='Number of cooled surfaces: 2 for top and bottom, 1 when the other surface is '
                                  'insulated.')),
                summary='Cools the selected domains through their top and bottom surfaces. Use it when the 2D model '
                        'stands for a thin membrane or chip layer.',
                description='Convective loss through the top / bottom faces of a thin slab (2D model of a membrane '
                            'or chip layer): volumetric loss h * sides / thickness * (T - T_ext) on the selected '
                            'cells. Anchors a steady solve when h > 0.',
                apply=_apply_out_of_plane_convection, grids=_BOTH_GRIDS),
    FeatureType('initial_temperature', 'Initial temperature', 'domain_condition', 'domain',
                params=(_temperature_param('Temperature of the selected cells at t = 0; other cells start at T0.',
                                           'Temperature of the selected domains at time zero; other cells start at '
                                           "'Reference temperature'.\n\n"
                                           'A steady solve uses it only as a starting guess.'),),
                summary='Starting temperature of the selected domains for transient heat.',
                description='Temperature of the selected cells at t = 0 for transient heat. A steady solve uses it '
                            'only as the starting guess and as the temperature of cells that neither conduct nor lose '
                            'heat (validation info).',
                apply=_apply_initial_temperature, validate=_validate_initial_temperature, grids=_FEM_GRIDS),
    FeatureType('temperature_probe', 'Temperature probe', 'probe', 'point',
                object_kinds=('point',), max_entities=1, result_kind='timeseries',
                summary='Records the temperature at a point after every heat solve.',
                description='Temperature (K) of the cell containing the point, one sample after every heat solve '
                            '(time-dependent study: every heat step; stationary study and the background stage of a '
                            'frequency-domain study: one per fixed-point iteration, all at t = 0).',
                apply=_apply_temperature_probe, result=_temperature_probe_result, validate=validate_points_inside,
                grids=_FEM_GRIDS),
    FeatureType('thermorefractive_noise', 'Thermorefractive noise', 'monitor', 'domain',
                params=_NOISE_PARAMS, result_kind='xy',
                summary='Spectrum of the noise that temperature fluctuations cause in light, for example light '
                        'scattered from one waveguide mode into another, or the phase of a guided mode. Computed by a '
                        "'Thermal noise' study.",
                description=_NOISE_DESCRIPTION,
                apply=_apply_thermorefractive_noise, result=_thermorefractive_noise_result,
                validate=_validate_thermorefractive_noise, memory_estimate=_thermorefractive_noise_memory),
    FeatureType('particle_noise', 'Diffusing particle scattering', 'monitor', 'domain',
                params=_PARTICLE_PARAMS, result_kind='xy',
                summary='Spectrum of the light scattered into a waveguide mode by dilute particles, such as proteins, '
                        "diffusing in the selected liquid. Computed by a 'Thermal noise' study.",
                description=_PARTICLE_DESCRIPTION,
                apply=_apply_particle_noise, result=_thermorefractive_noise_result,
                validate=_validate_particle_noise, memory_estimate=_thermorefractive_noise_memory),
)

HEAT_PHYSICS = PhysicsType(
    name=PHYSICS_TYPE_NAME,
    label='Heat Transfer in Solids',
    default_instance_name='heat',
    family='heat',
    dimensions=(1, 2, 3),   # 1D (symmetry 2.20): a 2D Domain one cell across, Automatic linear solver 'banded'
    coordinate_grids=('cartesian', 'polar'),
    group='heat',
    spaces_2d=('planar', 'cross_section', 'axisymmetric'),
    space_settings=AXISYMMETRIC_SPACE_SETTINGS,
    summary='Heat conduction in solids, either the steady state or changing in time. Domain edges without a boundary '
            'condition are insulated.',
    description='Anisotropic heat conduction (finite volumes on the rise T - T0; Automatic linear solver: multigrid PCG '
                'on the gpu backend and on 3D meshes, host SuperLU on the cpu in 2D, banded Cholesky in 1D, the sparse '
                'direct solver on triangle / tetrahedron meshes: a double cuDSS Cholesky on the gpu backend when '
                'nvmath-python is installed, host SuperLU otherwise or after a cuDSS refusal), 2D per unit z length or '
                '3D (7-point stencil with k_zz; boundary features on outer faces x_lo ... z_hi only, no k_xy, no '
                'out_of_plane_convection; fields are shown as z cuts): '
                'steady state (quasi-static) or implicit-Euler transient (time_mode). Solves on the physical window '
                'only (no PML; T is copied edge-extended into an EM PML ring). Unreferenced outer faces are '
                'insulated (a later boundary feature on the same face wins), material properties are unset unless a '
                'material or its library entry sets them (cells without k do not conduct), and a steady solve needs '
                'an anchor: a temperature, h > 0 convection, a fixed-temperature domain or out-of-plane convection. '
                'With EM in a time-dependent study set every to a few hundred EM steps (one CG solve costs far more '
                f'than an EM step; validation warns below {MIN_EM_STEPS_PER_SOLVE}). '
                'On a triangle mesh (MeshSpec kind triangles; 2D, planar or axisymmetric) finite elements of the '
                'discretization order (quadratic default) on the conducting window elements: object perimeters are '
                'their outline edges (temperature = Dirichlet there, convection = Robin, heat_flux = an edge source '
                'heating both sides), a point is its mesh node, field T is nodal (other fields and every state slot '
                'one value per element, T at the element centroid); no azimuthal_order, noise features or '
                'thermal_noise study. On a tetrahedron mesh (MeshSpec kind tetrahedra; 3D Cartesian) the same with '
                'Lagrange elements on tetrahedra (affine, diagonal k_xx / k_yy / k_zz): outer faces act on the window '
                'faces, object boundaries are their interface faces (temperature = Dirichlet, convection = Robin, '
                'heat_flux = a sheet source heating both sides), a point_heat_source is the exact point load at its '
                '(x, y, z), probes evaluate T at the point; fields are sampled on the uniform lattice of the mesh '
                '(field derived values are not available there yet); Automatic linear solver: sparse_direct (cuDSS on '
                'the gpu backend with nvmath-python; host SuperLU otherwise, ~1e5 unknowns P2 before its 3D size '
                'warning).',
    settings=_SETTINGS,
    features=_FEATURES,
    grids=('tensor', 'triangles', 'tetrahedra'),   # FEMHeatPhysics (mesh plan T4) / FEMHeatPhysics3D
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=THERMAL_SCHEMA,
    state_fields=_STATE_FIELDS,
    output_fields=tuple(HeatRenderer.field_infos.values()),
    source_slots={'volumetric_power': ('W/m^3', 'Heat source Q: each transfer replaces Q (copied; the PML ring is '
                                                'ignored) with the sum of the additive heating couplings '
                                                '(em_heat_joule, currents_heat_joule, fluid_heat_convection).')},
    supported_studies=('stationary', 'time_dependent', THERMAL_NOISE_STUDY),
    default_study={'type': 'stationary'},
    # linear: the CG solves; shifted: (j omega C + q^2 k_zz A + K) u = w of the noise spectra (sparse_direct =
    # host SuperLU per shift, the reference, or an opt-in MUMPS / PARDISO host factor through the same factorize_host /
    # solve_host; line_block = batched GPU solves, mixed or double)
    solver_roles={'linear': CG_DIFFUSION_LINEAR_ROLE,
                  'shifted': SolverRole('complex_symmetric', 'sparse_direct', studies=(THERMAL_NOISE_STUDY,),
                                        precisions=('mixed', 'double'),
                                        setting_choices={'back_end': ('auto', 'scipy_splu', 'cusolver_qr')
                                                         + HOST_DIRECT_BACK_ENDS})},
    backends=('gpu', 'cpu'),   # numpy twins of the system / field kernels (solver_plan S6 stage 2)
    conformal_boundaries=True,   # cut cells (held edges, insulated walls, perimeter heat_flux / convection; 2D)
    conformal_coordinates=('cartesian', 'polar', 'spherical'),   # cut cells on the logical (r, angle) cells too
    create=_create_heat,
    steps_in_time=_steps_in_time,
    validate=_validate_heat,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
)


def _selftest_cases():
    """The heat SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests import SELFTEST_CASES
    from .selftests_3d import SELFTEST_CASES_3D
    from .selftests_axisymmetric import AXISYMMETRIC_SELFTEST_CASES
    from .selftests_backend import BACKEND_SELFTEST_CASES
    from .selftests_coordinates import COORDINATE_SELFTEST_CASES
    from .selftests_cylindrical_3d import CYLINDRICAL_3D_SELFTEST_CASES
    from .selftests_fem import FEM_SELFTEST_CASES
    from .selftests_fem3d import FEM_3D_SELFTEST_CASES
    from .selftests_multigrid import MULTIGRID_SELFTEST_CASES
    from .selftests_radiation import RADIATION_SELFTEST_CASES
    from .selftests_thermal_noise import THERMAL_NOISE_SELFTEST_CASES
    from .selftests_time import TIME_SELFTEST_CASES
    from .selftests_unbounded import UNBOUNDED_SELFTEST_CASES
    return (SELFTEST_CASES + AXISYMMETRIC_SELFTEST_CASES + COORDINATE_SELFTEST_CASES + SELFTEST_CASES_3D
            + CYLINDRICAL_3D_SELFTEST_CASES + UNBOUNDED_SELFTEST_CASES + THERMAL_NOISE_SELFTEST_CASES
            + MULTIGRID_SELFTEST_CASES + BACKEND_SELFTEST_CASES + TIME_SELFTEST_CASES + FEM_SELFTEST_CASES
            + FEM_3D_SELFTEST_CASES + RADIATION_SELFTEST_CASES)


def register(registry):
    """Heat physics type, then its 'heated_slab', 'heated_chip_3d' and 'waveguide_thermal_noise' templates, then its
    selftest loader."""
    registry.register_physics(HEAT_PHYSICS)
    registry.register_template('heated_slab', 'Heated slab', HEATED_SLAB_TEMPLATE_DESCRIPTION, heated_slab_spec,
                               summary=HEATED_SLAB_TEMPLATE_SUMMARY)
    registry.register_template('heated_chip_3d', 'Heated chip (3D)', HEATED_CHIP_TEMPLATE_DESCRIPTION,
                               heated_chip_3d_spec, summary=HEATED_CHIP_TEMPLATE_SUMMARY)
    registry.register_template('waveguide_thermal_noise', 'Waveguide thermal noise', WAVEGUIDE_TEMPLATE_DESCRIPTION,
                               waveguide_thermal_noise_spec, summary=WAVEGUIDE_TEMPLATE_SUMMARY)
    registry.register_selftests(_selftest_cases)
