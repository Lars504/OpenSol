"""Model-layer definition of the 'pressure_acoustics' physics type (frequency-domain pressure acoustics on triangle and
tetrahedron meshes: settings, PML, features, the elastic_waves rho / sound_speed Params, validation, estimates) plus its
'rigid_disk_scattering' template and selftest loader; registered by physics/acoustics/definition.register."""
import cmath
import math

import numpy as np

from ...core import time_estimate as te
from ...core.backend import to_host
from ...core.constants import FACE_LABELS, FACES, FACES_3D
from ...core.harmonic_probe import harmonic_probe_result
from ...core.memory_estimate import (COMPLEX128_BYTES, COMPLEX128_CSR_ENTRY_BYTES, FLOAT32_BYTES, FLOAT64_BYTES,
                                     MemoryEstimate, host_lu_bytes)
from ...fem.diffusion import FEM_FILL_FACTOR, lagrange_entries_per_dof
from ...fem.diffusion3d import FEM_FILL_FACTOR_3D
from ...fem.pml import DEFAULT_PML_REFLECTION
from ...model.definition_helpers import (AXISYMMETRIC, AXISYMMETRIC_SPACE_SETTINGS, DIFFUSION_DIRECT_BACK_ENDS,
                                         DISCRETIZATION_PARAM, KEPT_SOLUTION_BYTES_PER_DOF, axisymmetric_issues,
                                         cudss_external, dof_map_bytes, element_order, element_term_ops, frequency_sweep_counts,
                                         geometry_param, harmonic_frequency_items, host_direct_back_end,
                                         is_axisymmetric, numpy_calls, plain_number, reduced_sweep_memory_items,
                                         reduced_sweep_time_items,
                                         reports_model_checks, resolved_back_end, scalar_harmonic_build_items,
                                         scalar_harmonic_build_items_3d, selected_material_data,
                                         shell_element_fraction, validate_points_inside)
from ...model.expressions import Expression, ExpressionError
from ...model.geometry import expand_entities
from ...model.schema import (DomainRequirements, FeatureType, Issue, Param, PhysicsType, SolverRole, feature_path,
                             resolve_values)
from ...solvers.host_direct import factor_label, host_direct_precision, solver_host_bytes, superlu_bytes_scale
from ...solvers.sparse_direct import CUDSS_BACK_ENDS, device_pencil_bytes, scipy_available
from ..mechanics.pml3d import DEFAULT_PML_CELLS, PMLPadding
from .definition import ELASTIC_WAVES
from .selftests_harmonic import RIGID_DISK_DESCRIPTION, RIGID_DISK_LABEL, RIGID_DISK_SUMMARY, rigid_disk_spec
from .solver_harmonic_fem import (ACOUSTIC_HARMONIC_SCHEMA, BACKGROUND_KINDS, FIELD_INFOS, SIDES, STATE_NAMES,
                                  BackgroundPressure)

TYPE_NAME = 'pressure_acoustics'       # = selftests_harmonic.TYPE_NAME (selftests never import this)
INSTANCE_NAME = 'pressure_acoustics'
FREQUENCY_RESPONSE_STUDY = 'frequency_response'   # = model.study_frequency_response.STUDY_NAME
_TETRAHEDRA = 'tetrahedra'
_BOUNDARY_FEATURES = ('sound_hard', 'sound_soft', 'prescribed_pressure', 'impedance', 'vibrating_boundary')
_MIN_ELEMENTS_PER_WAVELENGTH = {1: 10.0, 2: 5.0}   # resolution warning (P1 / P2, the usual rule of thumb)
_DIRECTIONS = ('+x', '-x', '+y', '-y', '+z', '-z', 'custom')
_MOTION_POWERS = {'acceleration': 0, 'velocity': 1, 'displacement': 2}   # (j omega)^power x amplitude = acceleration
# 'linear' role: sparse_direct, the cuDSS back ends (gpu backend: fem.harmonic keeps the reduced K, C, M on the device
# and refactorises the LDL^T per frequency on one plan, factorize_pencil); back_end 'auto' = cudss_mixed on the gpu
# backend with nvmath-python (basiri2019 E2, 76,740 unknowns: 15 ms per frequency vs 3.2 s SuperLU), else host
# SuperLU; the opt-in MUMPS / PARDISO back ends of DIFFUSION_DIRECT_BACK_ENDS factorise the host A(omega) by
# factorize_host (LDL^T, analysis reused per frequency through previous=); cusolver_qr refused (factorize_host would
# silently run SuperLU)
PRESSURE_LINEAR_ROLE = SolverRole('general', 'sparse_direct', precisions=('double',),
                                  setting_choices={'back_end': DIFFUSION_DIRECT_BACK_ENDS},
                                  auto_back_ends={'triangles': 'cudss_mixed', 'tetrahedra': 'cudss_mixed'})
# one Param object per shared material property name: the elastic_waves rho / sound_speed (registered first)
_ELASTIC_PROPERTIES = {param.name: param for param in ELASTIC_WAVES.material_properties}
_MATERIAL_PROPERTIES = (_ELASTIC_PROPERTIES['rho'], _ELASTIC_PROPERTIES['sound_speed'])
# uncalibrated estimate constants: host complex CSR copies (K, C, M data, the union and reduced operators), complex
# full-numbering vectors (values, loads, background, total, scattered, extra) per dof, local dofs per element
_HOST_CSR_COPIES = 5
_HOST_VECTORS_PER_DOF = 10
_LOCAL_DOFS = {(2, 1): 3, (2, 2): 6, (3, 1): 4, (3, 2): 10}
_INT64_BYTES = 8

# ---- settings ----
_GEOMETRY_PARAM = geometry_param(
    'the drawing is a cross-section of a fluid uniform along z; point sources are line sources (flow rate per metre of '
    'depth, m^2/s), surface results are per metre of depth',
    'the fluid is the body of revolution of the drawing and every field is independent of the angle (azimuthal order '
    '0); a point source on the axis is a point source, off the axis a ring of that total flow rate; surface means are '
    'area-weighted and weighted integrals cover the whole ring; no PML on x_lo. 2D models only (a 3D model is solved '
    'in full: validation errors on axisymmetric there)',
    'point sources are line sources along the depth.',
    'A point source on the axis is a point source, elsewhere a ring around the axis.',
    axis_text='a boundary condition there does nothing; validation warns')


def _pml_params():
    """pml_<face> per outer face, pml_cells, pml_reflection and pml_frequency."""
    faces = tuple(Param(
        f'pml_{face}', 'bool', False, label=f'Absorbing layer beyond the {FACE_LABELS[face].lower()}',
        help=(f'Perfectly matched layer beyond {face}: the mesh grows a layer of pml_cells x the mesh dx (h_max) there '
              '(triangles: the absorbing ring of pml_layers structured element layers; tetrahedra: the shell), its '
              'material continued from the window edge, with the fixed complex stretch s = 1 - j beta (d / w)^2 '
              '(fem.pml; d the depth, w the thickness), beta set by pml_reflection for the fastest sound speed in the '
              'layer at pml_frequency (default the lowest study frequency); its outer side is held at p = 0. Waves '
              'leave the window through it; higher frequencies and slower waves are absorbed more. No boundary '
              f'feature may select {face} (it is the inner side of the layer); z faces in 3D models only; geometry '
              'axisymmetric: not on x_lo (the axis). Another physics on the same mesh must not request layers on '
              'other faces (the run fails).'),
        tip=(f'Lets sound leave through the {FACE_LABELS[face].lower()} as if the fluid continued forever.\n\nAdds an '
             'absorbing layer outside the drawn region; no boundary condition can be placed on this edge then.'))
        for face in FACES_3D)
    cells = Param('pml_cells', 'int', DEFAULT_PML_CELLS, label='Absorbing layer thickness', unit='1', min=1, max=100,
                  help='Thickness of every PML in element sizes (x the mesh dx, h_max); about half a wavelength or '
                       'more at the lowest frequency, 6-12 typical. Triangles: MeshSpec pml_layers sets the structured '
                       'element layers across it.',
                  tip='Thickness of the absorbing layers, in element sizes.\n\nAbout half a wavelength or more at the '
                      'lowest frequency; 8 is typical.')
    reflection = Param('pml_reflection', 'float', DEFAULT_PML_REFLECTION, label='Absorbing layer reflection', unit='1',
                       min=0.0, min_exclusive=True, max=0.5,
                       help='Theoretical round-trip reflection R of a normally incident wave in the PML at '
                            'pml_frequency for the fastest sound speed c of the layer: beta = 3 ln(1 / R) c / (2 omega '
                            'w). A very small R makes the profile steep, which the elements reflect instead: 1e-3 to '
                            '1e-8 typical.',
                       tip='How strongly the absorbing layer damps a wave crossing it and back.\n\n1e-6 is typical.')
    frequency = Param('pml_frequency', 'float', None, label='Absorbing layer frequency', unit='Hz', min=0.0,
                      min_exclusive=True, optional=True,
                      help='Frequency [Hz] at which the PML strength is set (pml_reflection holds there; the stretch '
                           'does not change with frequency, so K and M are assembled once); empty = the lowest '
                           'frequency of the frequency_response study (absorption only grows above it).',
                      tip='Frequency at which the absorbing layers are tuned.\n\nLeave empty to use the lowest '
                          'frequency of the study.')
    return faces + (cells, reflection, frequency)


_SETTINGS = (_GEOMETRY_PARAM, DISCRETIZATION_PARAM, *_pml_params())

# ---- feature params ----
_PHASE_TIP = 'Phase shift of this drive in degrees.'


def _phase_param(what):
    return Param('phase', 'float', 0.0, label='Phase', unit='deg',
                 help=f'Phase of the complex amplitude of {what} in degrees (phasors exp(+j omega t)).', tip=_PHASE_TIP)


_BACKGROUND_WAVE_VISIBLE = (('wave', ('plane_wave', 'standing_wave')),)
_BACKGROUND_PARAMS = (
    Param('wave', 'choice', 'plane_wave', label='Background wave', choices=BACKGROUND_KINDS,
          choice_labels=(('plane_wave', 'Plane wave'), ('standing_wave', 'Plane wave on a rigid plane'),
                         ('expression', 'Expression')),
          help='plane_wave: p_b = p0 exp(-j k d . x) travelling along the unit direction d, k = 2 pi f / '
               'background_sound_speed; standing_wave: the plane wave reflected by a rigid plane normal to an axis '
               'direction, p_b = 2 p0 cos(k (u - plane_position)), u the coordinate along that axis (normal incidence '
               'onto a floor: direction -y, plane_position 0 gives 2 p0 cos(k y)); expression: p_b = pressure_re + j '
               'pressure_im as expressions of x, y (z in 3D) and parameters, the same at every frequency (sweep a '
               'parameter for a frequency-dependent one; its gradient on outer faces by central differences).',
          tip='The sound wave arriving at the structure.\n\n- Plane wave: a wave travelling in one direction.\n- Plane '
              'wave on a rigid plane: the wave plus its reflection from a hard floor or wall.\n- Expression: any '
              'pressure field you enter.'),
    Param('amplitude', 'float', 1.0, label='Amplitude', unit='Pa', visible_if=_BACKGROUND_WAVE_VISIBLE,
          help='Peak pressure p0 of the incident plane wave (a standing wave peaks at 2 p0).',
          tip='Peak pressure of the arriving wave.'),
    Param('phase', 'float', 0.0, label='Phase', unit='deg', visible_if=_BACKGROUND_WAVE_VISIBLE,
          help='Phase of p0 in degrees: of the plane wave at the origin, of the standing wave everywhere.',
          tip='Phase of the arriving wave in degrees.'),
    Param('direction', 'choice', '-y', label='Direction of travel', choices=_DIRECTIONS,
          choice_labels=(('+x', 'Along +x'), ('-x', 'Along -x'), ('+y', 'Along +y'), ('-y', 'Along -y'),
                         ('+z', 'Along +z (3D)'), ('-z', 'Along -z (3D)'), ('custom', 'Custom direction')),
          visible_if=_BACKGROUND_WAVE_VISIBLE,
          help='Propagation direction d of the incident wave: an axis, or custom = (direction_x, direction_y, '
               'direction_z) normalised (plane_wave only). +z / -z need a 3D model; geometry axisymmetric: +y or -y '
               '(along the axis) only.',
          tip='Direction in which the arriving wave travels.\n\nIn a rotationally symmetric model it must travel '
              'along the axis.'),
    *(Param(f'direction_{axis}', 'float', -1.0 if axis == 'y' else 0.0, label=f'Direction {axis}', unit='1',
            visible_if=(('direction', ('custom',)),),
            help=f'{axis} component of a custom propagation direction (normalised; the z component is ignored in 2D '
                 'and must be 0 there).',
            tip=f'{axis} part of the travel direction; the three parts are scaled to length 1.')
      for axis in 'xyz'),
    Param('plane_position', 'float', 0.0, label='Rigid plane position', unit='m',
          visible_if=(('wave', ('standing_wave',)),),
          help='Coordinate of the rigid plane along the axis of direction (y = plane_position for +y / -y).',
          tip='Position of the hard floor or wall that reflects the wave, along the travel direction.'),
    Param('pressure_re', 'float', 0.0, label='Pressure, real part', unit='Pa', field_dependent=True,
          visible_if=(('wave', ('expression',)),),
          help='Real part of p_b [Pa]: an expression of x, y (z in 3D) and parameters; it should solve the Helmholtz '
               'equation of the background medium (else the mismatch radiates).',
          tip='Real part of the background pressure, as a formula of the coordinates.'),
    Param('pressure_im', 'float', 0.0, label='Pressure, imaginary part', unit='Pa', field_dependent=True,
          visible_if=(('wave', ('expression',)),),
          help='Imaginary part of p_b [Pa] (phasors exp(+j omega t)), as pressure_re.',
          tip='Imaginary part of the background pressure, as a formula of the coordinates.'),
    Param('background_density', 'float', 1.2, label='Background density', unit='kg/m^3', min=0.0, min_exclusive=True,
          help='Density rho_b of the medium p_b travels in; elements whose 1 / rho and 1 / (rho c^2) differ from the '
               'background ones by more than 1e-6 (relative) are scatterers (the contrast source).',
          tip='Density of the fluid the arriving wave travels in; 1.2 for air.'),
    Param('background_sound_speed', 'float', 343.0, label='Background sound speed', unit='m/s', min=0.0,
          min_exclusive=True,
          help='Sound speed c_b of the background medium: the wavenumber k = 2 pi f / c_b of p_b and, with '
               'background_density, the medium the contrast is taken against. Air 343, water 1480.',
          tip='Speed of sound in the fluid the arriving wave travels in; 343 for air, 1480 for water.'),
)
_PRESSURE_PARAMS = (
    Param('pressure', 'float', 1.0, label='Pressure amplitude', unit='Pa', field_dependent=True,
          help='Total pressure amplitude held on the boundary (with a background: p_s = this - p_b there); may depend '
               'on x, y (z in 3D).',
          tip='Pressure held on the selected boundaries.'),
    _phase_param('the held pressure'),
)
_IMPEDANCE_PARAMS = (
    Param('impedance', 'float', None, label='Acoustic impedance', unit='Pa s/m', min=0.0, min_exclusive=True,
          optional=True,
          help='Specific acoustic impedance Z = p / v_n (v_n the normal velocity out of the fluid), a real number; '
               'empty = rho c of the adjacent fluid element, the plane-wave absorber (no reflection at normal '
               'incidence, a reflection growing with the angle). Air 413, water 1.48e6.',
          tip='Pressure over the normal velocity at the boundary.\n\nLeave empty for the impedance of the fluid itself, '
              'which absorbs a wave arriving straight on.'),
)
_MOTION_VISIBLE = {quantity: (('quantity', (quantity,)),) for quantity in _MOTION_POWERS}
_MOTION_PARAMS = (
    Param('quantity', 'choice', 'velocity', label='Given quantity', choices=tuple(_MOTION_POWERS),
          choice_labels=(('velocity', 'Normal velocity'), ('acceleration', 'Normal acceleration'),
                         ('displacement', 'Normal displacement')),
          help='Which normal motion of the wall is given (positive into the fluid): velocity v, acceleration a = j '
               'omega v or displacement d = v / (j omega); the flux n_out . grad p / rho = j omega v_in.',
          tip='Whether the speed, acceleration or displacement of the moving wall is given.'),
    Param('velocity', 'float', 1e-3, label='Normal velocity amplitude', unit='m/s', field_dependent=True,
          visible_if=_MOTION_VISIBLE['velocity'],
          help='Normal velocity amplitude into the fluid [m/s]; may depend on x, y (z in 3D).',
          tip='Peak speed of the wall into the fluid.'),
    Param('acceleration', 'float', 1.0, label='Normal acceleration amplitude', unit='m/s^2', field_dependent=True,
          visible_if=_MOTION_VISIBLE['acceleration'],
          help='Normal acceleration amplitude into the fluid [m/s^2]; may depend on x, y (z in 3D).',
          tip='Peak acceleration of the wall into the fluid.'),
    Param('displacement', 'float', 1e-9, label='Normal displacement amplitude', unit='m', field_dependent=True,
          visible_if=_MOTION_VISIBLE['displacement'],
          help='Normal displacement amplitude into the fluid [m]; may depend on x, y (z in 3D).',
          tip='Peak displacement of the wall into the fluid.'),
    _phase_param('the wall motion'),
)
_POINT_SOURCE_PARAMS = (
    Param('flow_rate', 'float', 1e-6, label='Volume flow rate', unit='m^3/s', space_units=(('planar', 'm^2/s'),),
          help='Peak volume flow rate Q of a monopole: the load j omega Q at the point; 3D free field p = j omega rho '
               'Q exp(-j k r) / (4 pi r). Planar 2D: per metre of depth (a line source, m^2/s); axisymmetric: a point '
               'on the axis, else a ring of total rate Q.',
          tip='Volume of fluid pushed out per second at the point, like a tiny pulsating sphere.'),
    _phase_param('the flow rate'),
)
_SIDE_LABELS = (('all', 'Every selected boundary'), ('+x', 'Facing +x'), ('-x', 'Facing -x'), ('+y', 'Facing +y'),
                ('-y', 'Facing -y'), ('+z', 'Facing +z (3D)'), ('-z', 'Facing -z (3D)'))
_SURFACE_PARAMS = (
    Param('side', 'choice', 'all', label='Side', choices=SIDES, choice_labels=_SIDE_LABELS,
          help="Part of the selection: 'all', or the boundary pieces whose outward normal (out of the selected object; "
               'an outer face: out of the domain) has a positive component along +x ... -z, e.g. +y = the top face of '
               'a rectangle (2D: +y = up; axisymmetric: +y = +z). +z / -z need a 3D model.',
          tip='Which faces of the selected objects are used, by the direction they face.\n\nFor example the top face '
              'of a disk faces up.'),
    Param('weight', 'float', 1.0, label='Weight', unit='1', field_dependent=True,
          help='Weight psi of the weighted integral int p psi dA (e.g. a mode shape expression of x, y, z); its unit '
               'multiplies the result (Pa m^2; planar 2D Pa m per metre of depth).',
          tip='Weighting function of the second result, e.g. the shape of a vibration mode.'),
)


# ---- expression helpers ----
def point_function(value, parameters):
    """Number, or a function of points (..., d) [m] -> float64 (...) of a resolved field-dependent value (an
    Expression of x, y, z and parameters)."""
    if not isinstance(value, Expression):
        return float(value)
    base = dict(parameters)

    def evaluate(points):
        points = np.asarray(points, dtype=np.float64)
        namespace = dict(base, x=points[..., 0], y=points[..., 1])
        if points.shape[-1] == 3:
            namespace['z'] = points[..., 2]
        values = np.broadcast_to(np.asarray(value.evaluate(namespace, xp=np), dtype=np.float64),
                                 points.shape[:-1]).copy()
        if not np.isfinite(values).all():
            raise ExpressionError(f'{value.text!r} gives non-finite values')
        return values
    return evaluate


def complex_function(value, phase_deg, parameters):
    """Complex number, or a function of points (..., d) -> complex (...), of an amplitude value and a phase [deg]."""
    rotation = cmath.exp(1j * math.radians(float(phase_deg)))
    function = point_function(value, parameters)
    if not callable(function):
        return function * rotation
    return lambda points: function(points) * rotation


def _direction_vector(values, dimensions):
    """Unit direction tuple (dimensions components) of the background direction settings."""
    choice = values['direction']
    if choice == 'custom':
        vector = np.array([values['direction_x'], values['direction_y'], values['direction_z']], dtype=np.float64)
    else:
        vector = np.zeros(3)
        vector['xyz'.index(choice[1])] = 1.0 if choice[0] == '+' else -1.0
    vector = vector[:dimensions]
    norm = float(np.linalg.norm(vector))
    if not norm > 0.0:
        raise ValueError(f'background direction {tuple(vector.tolist())} has no length in this {dimensions}D model')
    return tuple(float(item) for item in vector / norm)


# ---- apply / result callbacks (worker thread) ----
def _three_d(ctx):
    return ctx.domain.grid_kind == _TETRAHEDRA


def _boundary_entities(ctx, selection):
    """Mesh entities (2D edges, 3D faces) of the selected domain objects' outlines / interfaces; None without one."""
    if not selection.objects:
        return None
    return ctx.boundary_faces(selection.objects) if _three_d(ctx) else ctx.boundary_edges(selection.objects)


def _make_boundary_apply(kind):
    """apply of a boundary feature of BOUNDARY_KINDS `kind` on its outer faces and object outlines."""
    def apply(ctx, solver, feature_spec, values, selection):
        options = {}
        if kind == 'pressure':
            options['value'] = complex_function(values['pressure'], values['phase'], ctx.parameters)
        elif kind == 'impedance':
            options['impedance'] = values['impedance']
        elif kind == 'motion':
            quantity = values['quantity']
            options['value'] = complex_function(values[quantity], values['phase'], ctx.parameters)
            options['power'] = _MOTION_POWERS[quantity]
        for face in selection.outer_faces:
            solver.add_boundary(face, kind, **options)
        entities = _boundary_entities(ctx, selection)
        if entities is not None and len(entities):
            solver.add_boundary(entities, kind, **options)
        return None
    return apply


def _apply_background(ctx, solver, feature_spec, values, selection):
    dimensions = 3 if _three_d(ctx) else 2
    common = dict(density=values['background_density'], sound_speed=values['background_sound_speed'])
    if values['wave'] == 'expression':
        background = BackgroundPressure('expression', real_part=_as_points_callable(values['pressure_re'], ctx),
                                        imaginary_part=_as_points_callable(values['pressure_im'], ctx), **common)
    else:
        amplitude = values['amplitude'] * cmath.exp(1j * math.radians(values['phase']))
        background = BackgroundPressure(values['wave'], amplitude=amplitude,
                                        direction=_direction_vector(values, dimensions),
                                        plane_position=values['plane_position'], **common)
    solver.set_background(background)
    return None


def _as_points_callable(value, ctx):
    """Function of points (..., d) of a resolved field-dependent value (a number becomes a constant one)."""
    function = point_function(value, ctx.parameters)
    if callable(function):
        return function
    return lambda points: np.full(np.shape(points)[:-1], function)


def _apply_obstacle(ctx, solver, feature_spec, values, selection):
    solver.add_obstacle(to_host(ctx.domain_mask(selection.entities)))
    return None


def _point_position(ctx, name):
    return ctx.point_position(name) if _three_d(ctx) else ctx.point(name)


def _single_point(feature_spec, selection):
    if len(selection.objects) != 1:
        raise ValueError(f"feature '{feature_spec.name}' needs exactly one point object, got {len(selection.objects)}")
    return selection.objects[0]


def _apply_point_source(ctx, solver, feature_spec, values, selection):
    point = _point_position(ctx, _single_point(feature_spec, selection))
    solver.add_point_source(point, values['flow_rate'] * cmath.exp(1j * math.radians(values['phase'])))
    return None


def _apply_probe(ctx, solver, feature_spec, values, selection):
    point = _point_position(ctx, _single_point(feature_spec, selection))
    return solver.add_frequency_probe(point, label=f'Pressure at {feature_spec.name}')


def _probe_result(session, solver, record, feature_spec, options):
    return harmonic_probe_result(record)


def _apply_surface(ctx, solver, feature_spec, values, selection):
    entities = _boundary_entities(ctx, selection)
    member = to_host(ctx.domain_mask(selection.objects)) if selection.objects else None
    weight = point_function(values['weight'], ctx.parameters)
    return solver.add_surface_monitor(feature_spec.name, faces=tuple(selection.outer_faces), entities=entities,
                                      member=member, side=values['side'], weight=weight)


def _surface_result(session, solver, monitor, feature_spec, options):
    return solver.surface_result(monitor)


# ---- study frequencies (CPU) ----
def study_frequency_band(spec, registry, parameters):
    """(lowest, highest) frequency [Hz] of a frequency_response study (as model.study._frequency_list spaces them),
    None for another study or values that do not resolve."""
    study = spec.study
    if study.type != FREQUENCY_RESPONSE_STUDY:
        return None
    try:
        values = resolve_values(registry.study_type(study.type).params, study.settings or {}, parameters)
    except (KeyError, ValueError, TypeError):
        return None
    if values.get('frequencies'):
        frequencies = [float(value) for value in values['frequencies']]
        return min(frequencies), max(frequencies)
    start, stop = float(values['frequency_start']), float(values['frequency_stop'])
    if int(values['n_frequencies']) == 1:
        centre = math.sqrt(start * stop) if values['log_spacing'] else 0.5 * (start + stop)
        return centre, centre
    return min(start, stop), max(start, stop)


# ---- physics factory ----
def _pml_faces(settings_values, three_d=True) -> tuple:
    """Outer faces whose pml_<face> setting is on (z faces only in 3D)."""
    faces = FACES_3D if three_d else FACES
    return tuple(face for face in faces if settings_values.get(f'pml_{face}'))


def _create(ctx, physics_spec, values):
    """PressureAcousticsFEMPhysics (triangles) / PressureAcousticsFEMPhysics3D (tetrahedra) with the 'linear'
    sparse_direct runtime, the element order and the PML (reference frequency: pml_frequency or the lowest study
    frequency)."""
    from ...model.schema import get_registry
    three_d = _three_d(ctx)
    reference = values['pml_frequency']
    if reference is None:
        band = study_frequency_band(ctx.spec, get_registry(), ctx.parameters)
        reference = None if band is None else band[0]
    options = dict(linear_solver=ctx.solvers(physics_spec.name).get('linear'),
                   pml_faces=_pml_faces(values, three_d), pml_reflection=values['pml_reflection'],
                   pml_frequency=reference)
    if three_d:
        from .solver_harmonic_fem3d import PressureAcousticsFEMPhysics3D   # lazy: solver code at create only
        return PressureAcousticsFEMPhysics3D(ctx.domain, element_order(values), **options)
    from .solver_harmonic_fem import PressureAcousticsFEMPhysics
    return PressureAcousticsFEMPhysics(ctx.domain, element_order(values), axisymmetric=is_axisymmetric(values),
                                       **options)


def _domain_requirements(settings_values):
    """The PML padding (pml_cells element sizes on the pml_<face> faces: the triangle ring / tetrahedron shell)."""
    faces = _pml_faces(settings_values)
    if not faces:
        return DomainRequirements()
    cells = int(settings_values.get('pml_cells') or DEFAULT_PML_CELLS)
    return DomainRequirements(boundary_config=PMLPadding({face: cells if face in faces else 0 for face in FACES_3D}))


# ---- validation (CPU only) ----
def _outer_faces(vctx, feature):
    """Outer faces in a feature's selection (bad selections: none, reported by the selection checks)."""
    try:
        return [name for name in expand_entities(vctx.spec, feature.selection) if name in FACES_3D]
    except (ValueError, KeyError, TypeError):
        return []


def _pml_issues(vctx, physics_spec, settings_values, path, band):
    """PML faces: z faces in 2D, x_lo axisymmetric, a missing reference frequency, boundary features on PML faces."""
    faces = _pml_faces(settings_values)
    if not faces:
        return []
    issues = []
    if not vctx.spec.is_3d:
        three_d_faces = [face for face in faces if face not in FACES]
        if three_d_faces:
            issues.append(Issue('error', path, f"PML on {', '.join(three_d_faces)}: z faces exist in 3D models only"))
    if is_axisymmetric(settings_values) and 'x_lo' in faces:
        issues.append(Issue('error', path, 'an axisymmetric PML cannot lie on x_lo: that face is the axis (r = 0)'))
    if not settings_values.get('pml_frequency') and not (band and band[0] > 0.0):
        issues.append(Issue('error', path, 'a PML needs a reference frequency: set pml_frequency, or run a '
                                           'frequency_response study (its lowest frequency is the default)'))
    for feature in physics_spec.features:
        if feature.enabled and feature.type in _BOUNDARY_FEATURES:
            held = [face for face in _outer_faces(vctx, feature) if face in faces]
            if held:
                issues.append(Issue('error', feature_path(physics_spec, feature),
                                    f"{feature.type} on the PML face(s) {', '.join(held)}: that face is the inner side "
                                    'of the absorbing layer and takes no condition; select other faces or turn the '
                                    'PML off there'))
    return issues


def _material_issues(vctx, physics_spec, path):
    """Once per model: no material with rho and sound_speed (error); a selected material missing one of them and not
    covered by rigid_obstacle features (warning: the run fails on its fluid elements)."""
    if not reports_model_checks(vctx, physics_spec, TYPE_NAME):
        return []
    obstacles = set()
    for feature in physics_spec.features:
        if feature.enabled and feature.type == 'rigid_obstacle':
            obstacles.update(vctx.objects(feature.selection))
    issues, carriers = [], 0
    for material, properties, _ in selected_material_data(vctx.spec, unresolved=True):
        if properties is None:   # reported at the material
            carriers += 1
            continue
        rho, speed = (properties.get(name) for name in ('rho', 'sound_speed'))
        plain_rho, plain_speed = plain_number(rho, vctx.parameters), plain_number(speed, vctx.parameters)
        sets_both = (rho is not None and speed is not None and (plain_rho is None or plain_rho > 0.0)
                     and (plain_speed is None or plain_speed > 0.0))
        if sets_both:
            carriers += 1
            continue
        objects = vctx.objects(material.selection)
        if 'all' not in material.selection and objects and set(objects) <= obstacles:
            continue
        issues.append(Issue('warning', ('material', material.name),
                            f"material '{material.name}' lacks rho or sound_speed (> 0), which pressure_acoustics "
                            f"'{physics_spec.name}' needs in every fluid element: the run fails on the elements it "
                            'paints unless a rigid_obstacle feature selects them'))
    if carriers == 0:
        issues.append(Issue('error', path, 'no selected material sets both a density (rho) and a sound speed '
                                           '(sound_speed): nothing carries sound (air: 1.2 kg/m^3, 343 m/s)'))
    return issues


def _resolution_issues(vctx, physics_spec, settings_values, path, band):
    """Warning when the largest element spans more than 1 / _MIN_ELEMENTS_PER_WAVELENGTH of the shortest wavelength
    (slowest plain material sound speed at the highest study frequency)."""
    mesh = vctx.mesh
    if mesh is None or band is None or not band[1] > 0.0:
        return []
    slowest = None
    for _, properties, _ in selected_material_data(vctx.spec):
        speed = plain_number(properties.get('sound_speed'), vctx.parameters)
        if speed and speed > 0.0:
            slowest = speed if slowest is None else min(slowest, speed)
    if slowest is None:
        return []
    order = element_order(settings_values)
    per_wavelength = slowest / band[1] / float(mesh.dx_max)
    minimum = _MIN_ELEMENTS_PER_WAVELENGTH[order]
    if per_wavelength >= minimum:
        return []
    return [Issue('warning', path, f'only {per_wavelength:.3g} elements per shortest wavelength ({slowest:g} m/s at '
                                   f'{band[1]:.6g} Hz over elements up to {float(mesh.dx_max):.4g} m); use >= '
                                   f"{minimum:g} with {'quadratic' if order == 2 else 'linear'} elements or the "
                                   'waves disperse')]


def _validate(vctx, physics_spec, settings_values):
    """Physics rules: geometry / dimension, scipy, PML, background features, materials, resolution, the (r, z) axis."""
    path = ('physics', physics_spec.name)
    if not settings_values:
        return []
    issues = []
    if vctx.spec.is_3d and is_axisymmetric(settings_values):
        issues.append(Issue('error', path, "geometry 'axisymmetric' is a 2D setting: a 3D model is solved in full; set "
                                           'it to planar'))
    if not scipy_available():
        issues.append(Issue('error', path, 'pressure_acoustics factorises with scipy (host SuperLU) or cuDSS on host '
                                           'operators, and scipy is not installed: pip install scipy'))
    band = study_frequency_band(vctx.spec, vctx.registry, vctx.parameters)
    issues.extend(_pml_issues(vctx, physics_spec, settings_values, path, band))
    backgrounds = [feature for feature in physics_spec.features
                   if feature.enabled and feature.type == 'background_pressure_field']
    if len(backgrounds) > 1:
        issues.append(Issue('warning', feature_path(physics_spec, backgrounds[-1]),
                            f"{len(backgrounds)} background_pressure_field features: the last one ('"
                            f"{backgrounds[-1].name}') is the background, the others are ignored"))
    issues.extend(_material_issues(vctx, physics_spec, path))
    issues.extend(_resolution_issues(vctx, physics_spec, settings_values, path, band))
    if is_axisymmetric(settings_values) and not vctx.spec.is_3d:
        issues.extend(axisymmetric_issues(vctx, physics_spec, path, 'pressure_acoustics'))
    return issues


def _validate_background(vctx, physics_spec, feature_spec, values):
    """Direction rules: z directions need 3D, axisymmetric needs +y / -y, standing waves need an axis direction, a
    custom direction needs length (and no z component in 2D)."""
    if not values or values.get('wave') == 'expression':
        return []
    path = feature_path(physics_spec, feature_spec)
    direction, three_d = values['direction'], vctx.spec.is_3d
    issues = []
    if direction in ('+z', '-z') and not three_d:
        issues.append(Issue('error', path, f'direction {direction} needs a 3D model; use +y / -y in 2D'))
    if direction == 'custom':
        if values['wave'] == 'standing_wave':
            issues.append(Issue('error', path, 'a standing_wave needs an axis direction (its rigid plane is normal to '
                                               'an axis), not custom'))
        components = [values['direction_x'], values['direction_y']] + ([values['direction_z']] if three_d else [])
        if not any(components):
            issues.append(Issue('error', path, 'the custom direction has no length'))
        if not three_d and values['direction_z']:
            issues.append(Issue('error', path, 'direction_z must be 0 in a 2D model'))
    if is_axisymmetric(vctx.values(_SETTINGS, physics_spec.settings)) and not three_d and direction not in ('+y', '-y'):
        issues.append(Issue('error', path, 'geometry axisymmetric: the background wave must travel along the axis '
                                           '(direction +y or -y); an oblique wave is not rotationally symmetric'))
    return issues


def _validate_surface(vctx, physics_spec, feature_spec, values):
    if values and values.get('side') in ('+z', '-z') and not vctx.spec.is_3d:
        return [Issue('error', feature_path(physics_spec, feature_spec),
                      f"side {values['side']} needs a 3D model; use +y / -y in 2D")]
    return []


# ---- estimates (CPU) ----
def _counts(settings_values, mesh) -> tuple:
    """(order, three_d, scalar dofs of the full-mesh map, stored CSR entries of the operator) of a mesh estimate."""
    order = element_order(settings_values)
    three_d = mesh.kind == _TETRAHEDRA
    dofs = int(mesh.n_nodes) + (int(mesh.n_edges) if order == 2 else 0)
    entries = int(round(lagrange_entries_per_dof(mesh, order) * dofs))
    return order, three_d, dofs, entries


def _memory_estimate(settings_values, mesh, context):
    """Per-element state; host SuperLU factor (none on cuDSS), pencil copies, vectors and assembly COO (uncalibrated);
    on cuDSS the kept DevicePencil, solution, publish maps and p_b (run items) and the factor outside the pool."""
    if getattr(mesh, 'kind', 'tensor') not in _GRIDS:   # refused by validation
        return MemoryEstimate()
    order, three_d, dofs, entries = _counts(settings_values, mesh)
    elements = int(mesh.n_elements)
    fill = FEM_FILL_FACTOR_3D[order] if three_d else FEM_FILL_FACTOR
    back_end = resolved_back_end(context.solvers)
    cudss = back_end in CUDSS_BACK_ENDS
    # the HarmonicSystem's factor, refreshed per frequency on one plan and kept for the session (no CSR copies in the
    # pool beside the DevicePencil: pressure_acoustics_rigid_disk measured within 4 %)
    factors = ((('cuDSS factor of K + j omega C - omega^2 M', dofs, entries, COMPLEX128_BYTES, 3 if three_d else 2,
                 back_end == 'cudss_mixed'),) if cudss else ())
    local = _LOCAL_DOFS[(3 if three_d else 2, order)]
    factor = 0 if cudss else int(fill * host_lu_bytes(dofs, dimensions=3 if three_d else 2))
    # a MUMPS / PARDISO back end: the complex symmetric LDL^T at the SuperLU heuristic x superlu_bytes_scale + its own
    # copies (uncalibrated; refactorised per frequency on its analysis, fem.harmonic previous=)
    host_direct = host_direct_back_end(context.solvers)
    factor_items = [('complex SuperLU factor (FEM heuristic, uncalibrated)', factor)]
    if host_direct is not None:
        label = factor_label(host_direct)
        factor_items = [(f'complex {label} LDL^T factor (SuperLU FEM heuristic scaled, uncalibrated)',
                         int(factor * superlu_bytes_scale(host_direct, symmetric=True, complex_values=True))),
                        (f'{label} solver copies (equilibrated operator, row scale; mixed: GMRES vectors)',
                         solver_host_bytes(dofs, entries, complex_values=True, symmetric=True,
                                           mixed=host_direct_precision(host_direct) == 'mixed'))]
    # fem.harmonic on a cuDSS back end keeps the reduced K, C, M values on the device (sparse_direct.DevicePencil)
    # and keeps a direct solve's solution there (R6): complex128 full-numbering values, int64 free / fixed / system ids
    run_items = []
    if cudss:
        physics_spec = getattr(context, 'physics_spec', None)
        background = physics_spec is not None and any(
            feature.enabled and feature.type == 'background_pressure_field' for feature in physics_spec.features)
        run_items = [('device pencil terms, index maps and per-frequency transients',
                      device_pencil_bytes(entries, (COMPLEX128_BYTES,) * 3, symmetric=True)),
                     ('kept device solution (complex128 + int64 ids per full dof)', KEPT_SOLUTION_BYTES_PER_DOF * dofs),
                     # the device publish (R6): int64 element dofs, active dof / element masks, the centroid basis
                     ('device publish maps (element dofs, active masks, centroid basis)',
                      _INT64_BYTES * local * elements + dofs + elements + FLOAT64_BYTES * local)]
        if background:   # p_b at the dof points, uploaded per shown frequency
            run_items.append(('device background nodal values (complex128)', COMPLEX128_BYTES * dofs))
    return MemoryEstimate.from_items(
        ((f'element state ({len(STATE_NAMES)} x float32)', len(STATE_NAMES) * FLOAT32_BYTES * elements),),
        host_items=(*factor_items,
                    ('complex pencil copies (CSR)', _HOST_CSR_COPIES * COMPLEX128_CSR_ENTRY_BYTES * entries),
                    ('full-numbering complex vectors', _HOST_VECTORS_PER_DOF * COMPLEX128_BYTES * dofs),
                    ('assembly COO (transient, complex128)', 3 * COMPLEX128_BYTES * local * local * elements),
                    *reduced_sweep_memory_items(context, dofs, entries)), run_items=run_items,
        **cudss_external(factors))


# run-time recipe of the triangle host work beside definition_helpers' shared harmonic counts, counted from
# solver_harmonic_fem (bytes read + written): per full-numbering dof and frequency of each (j omega)^p load (scale, sum),
# of the background wave (values at the dof points, window mask, load sum, held offsets), of a shown solution (copy,
# background, total); per element of the shown state (centroid gather + basis dot per local dof, mask, |p| / Re / Im,
# float32 casts); per stored entry of the background source operators' two host products; dof points per dof (build)
_LOAD_BYTES_PER_DOF = 80
_BACKGROUND_BYTES_PER_DOF = 290
_BACKGROUND_OPERATOR_BYTES = 56
_SHOW_BYTES_PER_DOF = 130
_SHOW_BYTES_PER_LOCAL = 56
_SHOW_BYTES_PER_ELEMENT = 160
_DOF_POINT_BYTES = 80
_EDGE_LOAD_NUMPY_CALLS = 30   # one assemble_edge_load / precomputed edge quadrature (background faces, impedances)
_PROBE_NUMPY_CALLS = 15
_SURFACE_NUMPY_CALLS = 40


def _triangle_host_items(settings_values, mesh, context, order, dofs, entries, sweep, cudss) -> tuple:
    """Host items on triangles: the build at finalize (scalar_harmonic_build_items + the background source operators,
    bounded by every element; timed by the assembly_seconds counter), the DevicePencil set-up / host reduced operators
    and per frequency the solve bookkeeping (harmonic_frequency_items), loads and background, shown fields, records."""
    physics_spec = getattr(context, 'physics_spec', None)

    def enabled(*types) -> int:
        return 0 if physics_spec is None else sum(feature.enabled and feature.type in types
                                                  for feature in physics_spec.features)
    seconds = ('solvers', context.name, 'counters', 'assembly_seconds')
    axisymmetric = is_axisymmetric(settings_values)
    pml_faces = [face for face in FACES if settings_values.get(f'pml_{face}')]
    pml, elements = bool(pml_faces), int(mesh.n_elements)
    items = list(scalar_harmonic_build_items(
        mesh, order, elements=elements, dofs=dofs, stored_entries=entries, axisymmetric=axisymmetric, pml=pml,
        counter=('solvers', context.name, 'counters', 'assemblies'), seconds_counter=seconds))
    background = enabled('background_pressure_field') > 0
    if background:   # _source_operators (no PML stretch, real coefficients) and the dof points
        local, axis = (3 if order == 1 else 6), int(axisymmetric)
        term = dict(elements=elements, dofs=dofs, order=order, pml=False, axisymmetric=axisymmetric, value_bytes=8)
        items.append(te.TimeItem(
            'background source operators (bounded: every element)', 'setup', 1.0, 'bounded',
            (te.HostStream(dof_map_bytes(elements * local, order) + _DOF_POINT_BYTES * dofs),
             *element_term_ops(degree=max(2 * order - 2 + axis, 2), kind='stiffness', **term),
             *element_term_ops(degree=2 * order + axis, kind='mass', **term)), seconds_counter=seconds))
    items += harmonic_frequency_items(dofs, entries, sweep, device_pencil=cudss,
                                      term_value_bytes=(16, 16) if pml else (8, 8))
    if sweep['method'] != 'direct':
        return tuple(items)
    loads = enabled('point_source', 'vibrating_boundary')
    stream = (_LOAD_BYTES_PER_DOF * loads + (_BACKGROUND_BYTES_PER_DOF if background else 0)) * float(dofs)
    edge_loads = 0
    if background:   # source operator products (bounded by every entry), a flux load per open face and impedance
        stream += _BACKGROUND_OPERATOR_BYTES * float(entries)
        open_faces = [face for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')
                      if face not in pml_faces and not (axisymmetric and face == 'x_lo')]
        edge_loads = len(open_faces) + enabled('impedance')
    items.append(te.TimeItem('frequency loads and background (per frequency)', 'solve', sweep['solves'],
                             sweep['solves_kind'], (te.HostStream(stream),
                                                    numpy_calls(10 + 5 * loads + _EDGE_LOAD_NUMPY_CALLS * edge_loads))))
    local = 3 if order == 1 else 6
    items.append(te.TimeItem('shown fields (per field frequency)', 'frequency', sweep['field_points'], 'bounded', (
        te.HostStream(_SHOW_BYTES_PER_DOF * float(dofs)
                      + (_SHOW_BYTES_PER_LOCAL * local + _SHOW_BYTES_PER_ELEMENT) * float(elements)),
        te.Transfer(3.0 * FLOAT32_BYTES * elements), numpy_calls(25))))
    probes, surfaces = enabled('frequency_probe'), enabled('surface_pressure')
    if probes or surfaces:
        items.append(te.TimeItem('probe and surface records (per frequency)', 'solve', sweep['solves'],
                                 sweep['solves_kind'],
                                 (numpy_calls(_PROBE_NUMPY_CALLS * probes + _SURFACE_NUMPY_CALLS * surfaces),)))
    return tuple(items)


def _tetrahedron_host_items(settings_values, mesh, context, order, dofs, entries, sweep, cudss) -> tuple:
    """Host items on tetrahedra: the build at finalize (scalar_harmonic_build_items_3d: affine reference forms, the PML
    shell of pml_cells x h_max by quadrature; timed by the assembly_seconds counter), the DevicePencil set-up / per
    frequency solve bookkeeping (harmonic_frequency_items); background terms and post-processing are not estimated."""
    faces = _pml_faces(settings_values)
    thickness = int(settings_values.get('pml_cells') or DEFAULT_PML_CELLS) * float(mesh.dx_max)
    elements = int(mesh.n_elements)
    items = list(scalar_harmonic_build_items_3d(
        mesh, order, elements=elements, dofs=dofs, stored_entries=entries,
        pml_elements=shell_element_fraction(mesh, faces, thickness) * elements,
        counter=('solvers', context.name, 'counters', 'assemblies'),
        seconds_counter=('solvers', context.name, 'counters', 'assembly_seconds')))
    items += harmonic_frequency_items(dofs, entries, sweep, device_pencil=cudss,
                                      term_value_bytes=(16, 16) if faces else (8, 8))
    return tuple(items)


def _time_estimate(settings_values, mesh, context):
    """The study's harmonic_factorizations (one per frequency_response frequency, or a reduced sweep's) of the complex
    'linear' role (host factorize_host; cuDSS: run backend, device refactorisations; factor / solve scale the FEM fill
    FEM_FILL_FACTOR(_3D), as the memory estimate) at the full-map dofs, a reduced
    sweep's back substitutions and basis items; the host work on triangles (_triangle_host_items: build, DevicePencil,
    per-frequency loads, fields, records) and tetrahedra (_tetrahedron_host_items: build, DevicePencil; background
    terms and post-processing not estimated); a modal basis' shift-invert is not estimated."""
    if getattr(mesh, 'kind', 'tensor') not in _GRIDS:   # refused by validation
        return te.TimeEstimate(not_estimated=('pressure_acoustics on a rectangular-cell mesh',))
    order, three_d, dofs, entries = _counts(settings_values, mesh)
    sweep = frequency_sweep_counts(context)
    cudss = resolved_back_end(context.solvers) in CUDSS_BACK_ENDS
    fill = FEM_FILL_FACTOR_3D[order] if three_d else FEM_FILL_FACTOR   # the FEM operator over the 5- / 7-point curves
    linear = context.solver_work('linear', unknowns=float(dofs), nnz=float(entries), value_bytes=COMPLEX128_BYTES,
                                 dimensions=3 if three_d else 2, solves=sweep['solves'],
                                 solves_kind=sweep['solves_kind'], factorizations=sweep['factorizations'],
                                 factorizations_kind=sweep['factorizations_kind'], factor_scale=fill, solve_scale=fill,
                                 **({'device_values': True} if cudss else {'backend': 'cpu'}))
    missing = ('pressure_acoustics 3D background terms and post-processing',) if three_d else ()
    host = (_tetrahedron_host_items if three_d else _triangle_host_items)(settings_values, mesh, context, order, dofs,
                                                                          entries, sweep, cudss)
    if sweep['eigenpairs']:
        missing += ('modal basis shift-invert (private eigen solver)',)
    return te.combine((te.TimeEstimate((*host, *reduced_sweep_time_items(sweep, dofs)), not_estimated=missing),
                       linear))


def _mesh_method(spec, physics_spec, settings_values, mesh_kind, parameters):
    """GUI text of the discretisation on mesh_kind (validation covers the rest)."""
    order = 'quadratic' if element_order(settings_values) == 2 else 'linear'
    if mesh_kind in ('triangles', _TETRAHEDRA):
        return (f'Finite elements ({order}) for the complex sound pressure, one direct factorisation per frequency. '
                'Absorbing layers grow outside the drawn region; rigid obstacles are left out of the mesh solve.', [])
    return ('Needs a triangle mesh (2D) or a tetrahedron mesh (3D).', ['rectangular-cell meshes are not supported'])


# ---- type ----
_GRIDS = ('triangles', _TETRAHEDRA)
_FEATURES = (
    FeatureType('background_pressure_field', 'Background pressure field', 'domain_condition', None,
                params=_BACKGROUND_PARAMS, validate=_validate_background, apply=_apply_background, grids=_GRIDS,
                summary='An incoming sound wave, such as a plane wave or a wave reflected by a floor. The solution is '
                        'then the field scattered by the structure, and the total pressure is the two together.',
                description='Scattered-field formulation: the unknown is p_s, the total pressure p = p_s + p_b in the '
                            'window (p_s in a PML, which absorbs p_s only). p_b is assumed to solve the background '
                            'medium (background_density, background_sound_speed) exactly, so the source is only where '
                            'the model differs from it: elements whose material differs (the contrast), rigid_obstacle '
                            'elements (their sound-hard outline: n . grad p_s = -n . grad p_b), the outer faces (every '
                            'non-PML face is a total-field condition: 0 for a symmetry plane of p_b, e.g. the floor of '
                            'a standing wave) and the total-field impedance / held values (p_s = given - p_b). One per '
                            'physics (the last one wins). Probes and surface results add p_b exactly at their points; '
                            'the p_* fields hold the nodal p_s + p_b, the p_s_* fields the scattered part.'),
    FeatureType('rigid_obstacle', 'Rigid obstacle', 'domain_condition', 'domain', apply=_apply_obstacle, grids=_GRIDS,
                summary='Makes the selected domains immovable and impenetrable for sound; sound reflects from their '
                        'surface.',
                description='The selected elements are left out of the solve (no pressure inside, fields 0 there); '
                            'their outline is sound hard (n . grad p = 0), with a background the total field is. '
                            'Select solids and thin plates here (structural back-action neglected).'),
    FeatureType('sound_hard', 'Sound hard wall', 'boundary_condition', 'boundary', allow_interior=False,
                apply=_make_boundary_apply('hard'), grids=_GRIDS,
                summary='A rigid wall: no air moves through it. Every outer edge is a sound-hard wall unless something '
                        'else is set there.',
                description='n . grad p = 0 on outer faces (the default of every face without a PML or other '
                            'condition; also a mirror-symmetry plane), for the total field with a background. A later '
                            'feature on the same face wins, so this resets an earlier one.'),
    FeatureType('sound_soft', 'Sound soft boundary', 'boundary_condition', 'boundary',
                apply=_make_boundary_apply('soft'), grids=_GRIDS,
                summary='A pressure-release surface where the sound pressure is zero, like the surface of water seen '
                        'from below.',
                description='p = 0 (total field; with a background p_s = -p_b) on outer faces and object outlines. '
                            'Held values: the later feature wins shared nodes.'),
    FeatureType('prescribed_pressure', 'Pressure', 'boundary_condition', 'boundary', params=_PRESSURE_PARAMS,
                apply=_make_boundary_apply('pressure'), grids=_GRIDS,
                summary='Holds the sound pressure on the selected boundaries at a given amplitude and phase.',
                description='p = pressure exp(j phase) (total field) on outer faces and object outlines (the window '
                            'edge of an outer face; along a PML side the wall stays sound hard). Held values: the '
                            'later feature wins shared nodes.'),
    FeatureType('impedance', 'Impedance', 'boundary_condition', 'boundary', params=_IMPEDANCE_PARAMS,
                apply=_make_boundary_apply('impedance'), grids=_GRIDS,
                summary='A boundary that absorbs part of the sound, given by its acoustic impedance; empty absorbs a '
                        'wave arriving straight on completely.',
                description='Robin condition n_out . grad p / rho = -j omega p / Z (Z the specific impedance; empty: '
                            'rho c of the adjacent fluid, the first-order absorbing boundary) on outer faces and '
                            'object outlines; with a background the total field obeys it.'),
    FeatureType('vibrating_boundary', 'Vibrating wall', 'boundary_condition', 'boundary', params=_MOTION_PARAMS,
                apply=_make_boundary_apply('motion'), grids=_GRIDS,
                summary='A wall moving back and forth with a given velocity, acceleration or displacement, like a '
                        'loudspeaker membrane or a piston.',
                description='Inward normal motion: n_out . grad p / rho = j omega v_in (velocity), = a_in '
                            '(acceleration), = -omega^2 d_in (displacement), on outer faces and object outlines (a '
                            'load; the wall is sound hard otherwise).'),
    FeatureType('point_source', 'Point source', 'source', 'point', params=_POINT_SOURCE_PARAMS,
                object_kinds=('point',), max_entities=1, apply=_apply_point_source, validate=validate_points_inside,
                grids=_GRIDS,
                summary='A small pulsating source at a point, sending sound equally in all directions.',
                description='Monopole of volume flow rate Q exp(j phase): the load j omega Q phi_i(x0) (axisymmetric: '
                            'divided by 2 pi, the per-radian solve). Must lie in a fluid element.'),
    FeatureType('frequency_probe', 'Pressure probe', 'probe', 'point', object_kinds=('point',), max_entities=1,
                result_kind='xy', apply=_apply_probe, result=_probe_result, validate=validate_points_inside,
                grids=_GRIDS,
                summary='Records the complex sound pressure at a point at every frequency of the study.',
                description="Complex pressure at the point per solved frequency (core.harmonic_probe 'xy' result vs "
                            'frequency [Hz]): series p_re, p_im, p_abs, p_phase (total, p_b exact at the point) and '
                            'p_s_re ... (scattered; = p without a background), in Pa; 0 in a rigid obstacle.'),
    FeatureType('surface_pressure', 'Surface pressure', 'monitor', 'boundary', params=_SURFACE_PARAMS,
                result_kind='xy', apply=_apply_surface, result=_surface_result, validate=_validate_surface,
                grids=_GRIDS,
                summary='The mean sound pressure over selected surfaces, such as the top or bottom face of an object, '
                        'and a weighted integral of it, at every frequency.',
                description="'xy' result vs frequency [Hz]: series p_mean_re / _im / _abs / _phase = int p dA / int dA "
                            '[Pa] and p_weighted_* = int p weight dA (Pa m^2; planar 2D Pa m per metre of depth; '
                            'axisymmetric whole rings) over the selected outer faces (their window side) and object '
                            'outlines / interfaces on side (outward normal out of the object); extras area [m^2, '
                            'planar m], weight_integral (int weight dA), side. p is continuous, so a side picks faces, '
                            'not a field side; p_b enters exactly (quadrature). E.g. r = |<p_top> - <p_bottom>| / p_max '
                            'by a derived curve of two features (side +y, -y).'),
)

PRESSURE_ACOUSTICS = PhysicsType(
    name=TYPE_NAME,
    label='Pressure Acoustics, frequency domain',
    default_instance_name=INSTANCE_NAME,
    family='',
    group='structural',
    spaces_2d=('planar', AXISYMMETRIC),
    space_settings=AXISYMMETRIC_SPACE_SETTINGS,
    summary='Sound in air or water at given frequencies: the steady sound pressure around obstacles, with absorbing '
            'boundaries, incoming waves and pressure probes. Planar, rotationally symmetric or 3D.',
    description='Frequency-domain pressure acoustics for the frequency_'
                'response study: -div(grad p / rho) - omega^2 p / (rho c^2) = j omega q (exp(+j omega t), peak '
                'amplitudes), p the complex pressure, rho / c the material density and sound_speed (every fluid '
                'element needs both > 0; elastic_waves shares the Params; no losses but impedance and PML), on a '
                'triangle mesh (planar, or axisymmetric (r, z) with geometry, azimuthal order 0) or a tetrahedron '
                'mesh (3D), complex Lagrange P2 (default) or P1 elements (fem.harmonic / harmonic3d). K and M are '
                'assembled once; each frequency is one complex symmetric factorisation by the linear role '
                "(sparse_direct: back_end 'cudss' / 'cudss_mixed' on the gpu backend: cuDSS LDL^T reusing its plan, the "
                "pencil kept on the device; 'auto' = cudss_mixed there when nvmath-python is installed, else host "
                'SuperLU; opt-in MUMPS / PARDISO host LDL^T reusing its analysis per frequency). Outer faces default '
                'to sound hard '
                '(also mirror planes); features: sound_hard, sound_soft, prescribed_pressure, impedance, '
                'vibrating_boundary (outer faces act on their window edge, object outlines in 2D / interfaces in 3D), '
                'rigid_obstacle (excluded voids), point_source, background_pressure_field (scattered-field '
                'formulation), frequency_probe and surface_pressure (complex records per frequency, harmonic probe '
                'series <c>_re / _im / _abs / _phase). PML: settings pml_<face> grow absorbing layers (fixed complex '
                'stretch, fem.pml) of pml_cells element sizes beyond those faces, tuned at pml_frequency (default the '
                'lowest study frequency). Fields: p_abs / p_re / p_im / spl (total), p_s_* (scattered), v_* particle '
                'velocity j grad p / (omega rho) per element (v_z in 3D), rho, sound_speed; state p_abs / p_re / p_im '
                'per element. Snapshots per frequency labelled "f = <Hz> Hz". The instance is usually named '
                'pressure_acoustics.',
    settings=_SETTINGS,
    features=_FEATURES,
    material_properties=_MATERIAL_PROPERTIES,
    material_schema=dict(ACOUSTIC_HARMONIC_SCHEMA),
    state_fields=tuple(FIELD_INFOS[name] for name in STATE_NAMES),
    output_fields=tuple(FIELD_INFOS.values()),
    supported_studies=(FREQUENCY_RESPONSE_STUDY,),
    default_study={'type': FREQUENCY_RESPONSE_STUDY,
                   'settings': {'frequency_start': 100.0, 'frequency_stop': 1000.0, 'n_frequencies': 10}},
    create=_create,
    domain_requirements=_domain_requirements,
    validate=_validate,
    memory_estimate=_memory_estimate,
    time_estimate=_time_estimate,
    dimensions=(2, 3),
    solver_roles={'linear': PRESSURE_LINEAR_ROLE},
    frequency_sweep_methods=('direct', 'adaptive', 'modal'),   # fem.harmonic plan_reduced (modal: no damping terms)
    backends=('gpu', 'cpu'),   # host assembly and factorize_host either way; the state arrays on the Domain's xp
    grids=_GRIDS,
    mesh_method=_mesh_method,
)


def _selftest_cases():
    """The pressure_acoustics SelfTestCases (imported on the first registry.selftests(), not by get_registry)."""
    from .selftests_harmonic import HARMONIC_SELFTEST_CASES
    from .selftests_reduced_sweep import REDUCED_SWEEP_CASES
    return HARMONIC_SELFTEST_CASES + REDUCED_SWEEP_CASES


def register(registry):
    """The pressure_acoustics type, its 'rigid_disk_scattering' template, then its selftest loader."""
    registry.register_physics(PRESSURE_ACOUSTICS)
    registry.register_template('rigid_disk_scattering', RIGID_DISK_LABEL, RIGID_DISK_DESCRIPTION, rigid_disk_spec,
                               summary=RIGID_DISK_SUMMARY)
    registry.register_selftests(_selftest_cases)
