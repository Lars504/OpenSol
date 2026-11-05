"""Analytic validation of 3D heat (B3, 7-point stencil): a Robin slab along z on a graded z axis (exact discrete
identity), z-dependent k_zz and Q (igor 1.10), the exact product-of-sines eigenvector read through cuts / segments /
the volume (igor 1.3), plus the 'heated_chip_3d' template; registered by definition.register."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import EXACT_CG_NAME, with_exact_cg
from ...model.spec import (AnalyticCurveSpec, CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec,
                           MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec, RefinementSpec, StudySpec)

HEAT_TYPE_NAME = 'heat'            # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'heat'
TEMPERATURE_FIELD = f'{PHYSICS_NAME}.T'

CELL_SIZE = 1e-5                   # m, uniform cubic cells
BOX_SIZE = (2e-4, 1.2e-4, 1.6e-4)  # m, (x, y, z): 20 x 12 x 16 cells, even along every axis
CONDUCTIVITY = (1.0, 2.0, 4.0)     # W/(m*K), k_xx, k_yy, k_zz: a swapped component or axis changes the eigenvalue
WALL_TEMPERATURE = 300.0           # K, the Dirichlet faces; the held face of the Robin slab
ROBIN_H = 2e4                      # W/(m^2*K): 1 / h = 5e-5 m^2 K/W, comparable to L_z / k_zz = 4e-5
ROBIN_AMBIENT = 400.0              # K
EXACT_IDENTITY_TOLERANCE = 1e-5    # float32 T (~3e-5 K steps near 400 K) over rises of 16-100 K: ~2e-6
_AXES = 'xyz'


def _robin_flux():
    """Heat flux [W/m^2] through the Robin slab: (T_ext - T_wall) / (L_z / k_zz + 1 / h)."""
    return (ROBIN_AMBIENT - WALL_TEMPERATURE) / (BOX_SIZE[2] / CONDUCTIVITY[2] + 1.0 / ROBIN_H)


def _box_spec(title, features):
    """Stationary 3D heat-only box, one anisotropic conductor everywhere, uniform CELL_SIZE cells."""
    k_xx, k_yy, k_zz = CONDUCTIVITY
    return with_exact_cg(ModelSpec(
        title=title,
        domain=DomainSpec(size_x=BOX_SIZE[0], size_y=BOX_SIZE[1], size_z=BOX_SIZE[2]),
        materials=[MaterialSpec('conductor', properties={'k_xx': k_xx, 'k_yy': k_yy, 'k_zz': k_zz},
                                selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=list(features))],
        mesh=MeshSpec(mode='uniform', dx=CELL_SIZE),
        study=StudySpec(type='stationary')))


def _cut_plane(results):
    """(float64 T of the mid-height z cut, x centres, y centres, z of the cut [m])."""
    field_data = results.field(TEMPERATURE_FIELD)
    plane = getattr(field_data, 'plane', None)
    # mid-height lies on a cell edge (even z count): the cut takes the upper cell
    z_cut = float(plane[1]) if plane else 0.5 * BOX_SIZE[2] + 0.5 * CELL_SIZE
    return (np.asarray(field_data.values, dtype=np.float64), np.asarray(field_data.x_centers, dtype=np.float64),
            np.asarray(field_data.y_centers, dtype=np.float64), z_cut)


# ---- heat_3d_z_expressions (igor 1.10): z in a material property and in a feature value ----
Z_HIGH_TEMPERATURE = 400.0         # K on z_hi of the z_material run
Z_SOURCE_PEAK = 1.2e11             # W/m^3, Q = Z_SOURCE_PEAK z / L_z of the z_source run (rise ~ 50 K)
_Z_MOMENT_POWERS = (0, 1, 2)       # derived values mean of (T - T_w) (z / L_z)^p


def _z_moment_name(power):
    return f'rise_moment_{power}'


def _z_expression_runs():
    """'z_material': k_zz linear in z between two held z faces; 'z_source': Q linear in z, both z faces held."""
    height = BOX_SIZE[2]
    walls = [FeatureSpec('low', 'temperature', ['z_lo'], {'temperature': WALL_TEMPERATURE})]
    material = _box_spec('3D slab with k_zz linear in z', walls + [
        FeatureSpec('high', 'temperature', ['z_hi'], {'temperature': Z_HIGH_TEMPERATURE})])
    material.materials[0].properties['k_zz'] = f'{CONDUCTIVITY[2]!r}*(1 + z/{height!r})'
    source = _box_spec('3D slab heated by Q linear in z', [
        FeatureSpec('walls', 'temperature', ['z_lo', 'z_hi'], {'temperature': WALL_TEMPERATURE}),
        FeatureSpec('source', 'heat_source', ['all'], {'Q': f'{Z_SOURCE_PEAK!r}*z/{height!r}'})])
    for spec in (material, source):
        spec.derived_values = [DerivedValueSpec(_z_moment_name(power),
                                                f'(T - {WALL_TEMPERATURE!r})*(z/{height!r})**{power}', [], 'mean')
                               for power in _Z_MOMENT_POWERS]
    return {'z_material': material, 'z_source': source}


def _discrete_z_profile(conductivity, power_density, low_temperature, high_temperature):
    """Cell-centre T [K] of the 1D finite-volume z problem on CELL_SIZE cells: series half-cell conductances, the
    two z faces held (half-cell conductance 2 k / h), source Q h per unit area."""
    count = conductivity.size
    interior = 2.0 * conductivity[:-1] * conductivity[1:] / (CELL_SIZE * (conductivity[:-1] + conductivity[1:]))
    low_face, high_face = 2.0 * conductivity[0] / CELL_SIZE, 2.0 * conductivity[-1] / CELL_SIZE
    lower = np.arange(count - 1)
    matrix = np.zeros((count, count))
    np.add.at(matrix, (lower, lower), interior)
    np.add.at(matrix, (lower + 1, lower + 1), interior)
    matrix[lower, lower + 1] -= interior
    matrix[lower + 1, lower] -= interior
    matrix[0, 0] += low_face
    matrix[-1, -1] += high_face
    rhs = power_density * CELL_SIZE
    rhs[0] += low_face * low_temperature
    rhs[-1] += high_face * high_temperature
    return np.linalg.solve(matrix, rhs)


def _z_expected_profiles():
    """{run: (z centres, discrete T, closed-form T, rise)} of the two heat_3d_z_expressions runs."""
    height, conductivity = BOX_SIZE[2], CONDUCTIVITY[2]
    z_centres = (np.arange(round(height / CELL_SIZE)) + 0.5) * CELL_SIZE
    material_rise = Z_HIGH_TEMPERATURE - WALL_TEMPERATURE
    material_discrete = _discrete_z_profile(conductivity * (1.0 + z_centres / height), np.zeros_like(z_centres),
                                            WALL_TEMPERATURE, Z_HIGH_TEMPERATURE)
    material_closed = WALL_TEMPERATURE + material_rise * np.log1p(z_centres / height) / math.log(2.0)
    source_discrete = _discrete_z_profile(np.full_like(z_centres, conductivity), Z_SOURCE_PEAK * z_centres / height,
                                          WALL_TEMPERATURE, WALL_TEMPERATURE)
    source_closed = (WALL_TEMPERATURE + Z_SOURCE_PEAK * z_centres * (height ** 2 - z_centres ** 2)
                     / (6.0 * conductivity * height))
    source_rise = float(np.max(source_closed)) - WALL_TEMPERATURE
    return {'z_material': (z_centres, material_discrete, material_closed, material_rise),
            'z_source': (z_centres, source_discrete, source_closed, source_rise)}


def _check_z_expressions(results_by_name):
    """Largest deviation of each run's mid-height cut and z moments from the discrete 1D profile, / its rise."""
    errors, details = {}, {}
    for name, (z_centres, discrete, closed, rise) in _z_expected_profiles().items():
        results = results_by_name[name]
        values, _, _, z_cut = _cut_plane(results)
        cut_index = int(np.argmin(np.abs(z_centres - z_cut)))
        cut_deviation = float(np.max(np.abs(values - discrete[cut_index])))   # NaN propagates
        derived = results.summary()['derived']
        moment_deviations = {}
        for power in _Z_MOMENT_POWERS:
            expected = float(np.mean((discrete - WALL_TEMPERATURE) * (z_centres / BOX_SIZE[2]) ** power))
            moment_deviations[power] = abs(float(derived[_z_moment_name(power)]) - expected)   # float() reads 'nan'
        deviations = [cut_deviation, *moment_deviations.values()]
        finite = all(math.isfinite(value) for value in deviations)   # max() would drop a later NaN
        errors[name] = max(deviations) / rise if finite else math.inf
        details[name] = {'cut_deviation_K': cut_deviation, 'z_cut_m': z_cut,
                         'moment_deviations_K': {str(power): value for power, value in moment_deviations.items()},
                         'rise_K': rise, 'closed_form_gap_relative': float(np.max(np.abs(discrete - closed))) / rise}
    error = max(errors.values())
    return {'measured': error, 'expected': 0.0, 'error': error,
            'tolerance': EXACT_IDENTITY_TOLERANCE, 'details': {'relative_errors': errors, **details}}


# ---- heat_3d_z_couplings (igor 1.10): z in a field_transfer expression and in a state-dependent property ----
REFERENCE_PHYSICS = 'reference'    # the field_transfer source: linear T between z_lo (T_w) and z_hi (Z_HIGH_TEMPERATURE)
Z_DYNAMIC_BETA = 2e-3              # 1/K: k_zz = k0 (1 + z / L_z) (1 + beta (T - T_w)) of the z_dynamic run
Z_COUPLED_TOLERANCE = 1e-4         # float32 T and the stationary fixed point's float32 floor (z_dynamic)
_Z_DYNAMIC_STUDY = {'tolerance': 1e-7, 'max_iterations': 60}


def _z_coupling_runs():
    """'z_transfer': physics reference (linear T) -> field_transfer Q0 (T - T_w) / dT z / L_z = Q0 (z / L_z)^2 into
    the volumetric_power of 'heat' (z faces at T_w); 'z_dynamic': k_zz of z and T (a DynamicMaterialUpdate)."""
    height = BOX_SIZE[2]
    rise = Z_HIGH_TEMPERATURE - WALL_TEMPERATURE
    transfer = _box_spec('3D heat source from a field_transfer of z', [
        FeatureSpec('walls', 'temperature', ['z_lo', 'z_hi'], {'temperature': WALL_TEMPERATURE})])
    transfer.physics.insert(0, PhysicsSpec(REFERENCE_PHYSICS, HEAT_TYPE_NAME, features=[
        FeatureSpec('low', 'temperature', ['z_lo'], {'temperature': WALL_TEMPERATURE}),
        FeatureSpec('high', 'temperature', ['z_hi'], {'temperature': Z_HIGH_TEMPERATURE})],
        solvers={'linear': EXACT_CG_NAME}))
    transfer.couplings = [CouplingSpec('z_heating', 'field_transfer', REFERENCE_PHYSICS, PHYSICS_NAME, params={
        'source_field': 'T', 'target_slot': 'volumetric_power',
        'expression': f'{Z_SOURCE_PEAK!r}*(value - {WALL_TEMPERATURE!r})/{rise!r}*z/{height!r}'})]
    dynamic = _box_spec('3D slab with k_zz of z and T', [
        FeatureSpec('low', 'temperature', ['z_lo'], {'temperature': WALL_TEMPERATURE}),
        FeatureSpec('high', 'temperature', ['z_hi'], {'temperature': Z_HIGH_TEMPERATURE})])
    dynamic.materials[0].properties['k_zz'] = (f'{CONDUCTIVITY[2]!r}*(1 + z/{height!r})'
                                               f'*(1 + {Z_DYNAMIC_BETA!r}*(T - {WALL_TEMPERATURE!r}))')
    dynamic.study.settings = dict(_Z_DYNAMIC_STUDY)
    for spec in (transfer, dynamic):
        spec.derived_values = [DerivedValueSpec(_z_moment_name(power),
                                                f'({PHYSICS_NAME}_T - {WALL_TEMPERATURE!r})*(z/{height!r})**{power}',
                                                [], 'mean')
                               for power in _Z_MOMENT_POWERS]
    return {'z_transfer': transfer, 'z_dynamic': dynamic}


def _z_coupling_profiles():
    """{run: (z centres, discrete T, closed-form T, rise)} of the heat_3d_z_couplings runs (z_dynamic: Picard on the
    1D finite volumes; closed form by the Kirchhoff transform u + beta u^2 / 2 = C ln(1 + z / L_z))."""
    height, conductivity = BOX_SIZE[2], CONDUCTIVITY[2]
    z_centres = (np.arange(round(height / CELL_SIZE)) + 0.5) * CELL_SIZE
    transfer_discrete = _discrete_z_profile(np.full_like(z_centres, conductivity),
                                            Z_SOURCE_PEAK * (z_centres / height) ** 2, WALL_TEMPERATURE,
                                            WALL_TEMPERATURE)
    transfer_closed = (WALL_TEMPERATURE + Z_SOURCE_PEAK * z_centres * (height ** 3 - z_centres ** 3)
                       / (12.0 * conductivity * height ** 2))
    rise = Z_HIGH_TEMPERATURE - WALL_TEMPERATURE
    dynamic = WALL_TEMPERATURE + rise * z_centres / height
    for _ in range(500):
        conductivities = conductivity * (1.0 + z_centres / height) * (1.0 + Z_DYNAMIC_BETA * (dynamic
                                                                                             - WALL_TEMPERATURE))
        updated = _discrete_z_profile(conductivities, np.zeros_like(z_centres), WALL_TEMPERATURE, Z_HIGH_TEMPERATURE)
        converged = float(np.max(np.abs(updated - dynamic))) <= 1e-12 * rise
        dynamic = updated
        if converged:
            break
    transform_constant = (rise + 0.5 * Z_DYNAMIC_BETA * rise ** 2) / math.log(2.0)
    dynamic_closed = WALL_TEMPERATURE + (np.sqrt(1.0 + 2.0 * Z_DYNAMIC_BETA * transform_constant
                                                 * np.log1p(z_centres / height)) - 1.0) / Z_DYNAMIC_BETA
    return {'z_transfer': (z_centres, transfer_discrete, transfer_closed,
                           float(np.max(transfer_closed)) - WALL_TEMPERATURE),
            'z_dynamic': (z_centres, dynamic, dynamic_closed, rise)}


def _check_z_couplings(results_by_name):
    """Largest deviation of each run's mid-height cut and z moments from its 1D finite-volume profile, / its rise."""
    errors, details = {}, {}
    for name, (z_centres, discrete, closed, rise) in _z_coupling_profiles().items():
        results = results_by_name[name]
        values, _, _, z_cut = _cut_plane(results)
        cut_index = int(np.argmin(np.abs(z_centres - z_cut)))
        cut_deviation = float(np.max(np.abs(values - discrete[cut_index])))   # NaN propagates
        summary = results.summary()
        derived = summary['derived']
        moment_deviations = {}
        for power in _Z_MOMENT_POWERS:
            expected = float(np.mean((discrete - WALL_TEMPERATURE) * (z_centres / BOX_SIZE[2]) ** power))
            moment_deviations[power] = abs(float(derived[_z_moment_name(power)]) - expected)   # float() reads 'nan'
        deviations = [cut_deviation, *moment_deviations.values()]
        finite = all(math.isfinite(value) for value in deviations)
        errors[name] = max(deviations) / rise if finite else math.inf
        details[name] = {'cut_deviation_K': cut_deviation, 'z_cut_m': z_cut,
                         'moment_deviations_K': {str(power): value for power, value in moment_deviations.items()},
                         'rise_K': rise, 'closed_form_gap_relative': float(np.max(np.abs(discrete - closed))) / rise,
                         'iterations': summary.get('ticks'), 'warnings': list(summary.get('warnings') or [])}
    error = max(errors.values())
    return {'measured': error, 'expected': 0.0, 'error': error,
            'tolerance': Z_COUPLED_TOLERANCE, 'details': {'relative_errors': errors, **details}}


# ---- heat_3d_graded_z_robin (igor 1.4): the Robin slab on a graded z axis ----
GRADED_FINE_CELL = 5e-6            # m, refinement of the 'layer' box (x, y and z bands; margin 0; contrast 2)
GRADED_LAYER_Z = (0.25, 0.5)       # z0 / z1 of the layer box as fractions of L_z: z snap nodes (40 um: 8 fine cells)
# x / y spans of the box as fractions of L_x / L_y: whole fine cells (40 / 30 um), else the snapped segment's rescale
# shrinks its middle cells below GRADED_FINE_CELL and dx_min misses the 5 % check
GRADED_LAYER_X, GRADED_LAYER_Y = (0.4, 0.6), (0.375, 0.625)


def _graded_robin_runs():
    """'graded_robin_z': T_w on z_lo, convection on z_hi, no source, a centred box refined to GRADED_FINE_CELL (graded
    x, y and z, z faces snapped); derived value linear_gap = max |T - (T_w + q z / k_zz)| over every cell."""
    length_x, length_y, length_z = BOX_SIZE
    spec = _box_spec('3D Robin slab along z, graded z', [
        FeatureSpec('held', 'temperature', ['z_lo'], {'temperature': WALL_TEMPERATURE}),
        FeatureSpec('cooling', 'convection', ['z_hi'], {'h': ROBIN_H, 'T_ext': ROBIN_AMBIENT})])
    spec.geometry = [GeometrySpec('layer', 'box', {
        'x0': GRADED_LAYER_X[0] * length_x, 'y0': GRADED_LAYER_Y[0] * length_y, 'z0': GRADED_LAYER_Z[0] * length_z,
        'x1': GRADED_LAYER_X[1] * length_x, 'y1': GRADED_LAYER_Y[1] * length_y, 'z1': GRADED_LAYER_Z[1] * length_z})]
    spec.mesh = MeshSpec(mode='nonuniform', dx=CELL_SIZE, refinements=[RefinementSpec('layer', GRADED_FINE_CELL, 0.0)])
    slope = _robin_flux() / CONDUCTIVITY[2]
    spec.derived_values = [DerivedValueSpec('linear_gap', f'abs(T - ({WALL_TEMPERATURE!r} + {slope!r}*z))', [], 'max')]
    return {'graded_robin_z': spec}


def _check_graded_robin(results_by_name):
    """Mid-height cut and every-cell max of |T - linear| / the rise; raises unless the build's z axis is graded."""
    results = results_by_name['graded_robin_z']
    summary = results.summary()
    build = summary.get('build') or {}
    uniform_nz = round(BOX_SIZE[2] / CELL_SIZE)
    if not (int(build.get('nz', 0)) > uniform_nz
            and abs(float(build.get('dx_min', 0.0)) / GRADED_FINE_CELL - 1.0) < 0.05):
        raise ValueError(f'expected a graded z axis (nz > {uniform_nz}, dx_min {GRADED_FINE_CELL:g} m), got build '
                         f"nz {build.get('nz')}, dx_min {build.get('dx_min')}")
    values, _, _, z_cut = _cut_plane(results)
    rise = ROBIN_AMBIENT - WALL_TEMPERATURE
    cut_deviation = float(np.max(np.abs(values - (WALL_TEMPERATURE + _robin_flux() * z_cut / CONDUCTIVITY[2]))))
    volume_deviation = float(summary['derived']['linear_gap'])   # float() reads 'nan'
    deviations = (cut_deviation, volume_deviation)
    error = max(deviations) / rise if all(math.isfinite(value) for value in deviations) else math.inf
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': EXACT_IDENTITY_TOLERANCE,
            'details': {'cut_deviation_K': cut_deviation, 'max_deviation_K': volume_deviation, 'z_cut_m': z_cut,
                        'nz': int(build['nz']), 'dx_min_m': float(build['dx_min']), 'rise_K': rise}}


# ---- heat_3d_field_access (igor 1.3): cuts along x / y / z, stored cuts, 3D segments, the volume ----
ACCESS_AMPLITUDE = 50.0            # K, peak rise of the product-of-sines field
ACCESS_CUT_CELLS = (6, 4, 10)      # cell index of the x / y / z cut (positions at those cell centres)
ACCESS_OBLIQUE = ((1.3, 2.2, 0.7), (18.6, 9.4, 15.1))   # oblique segment ends in cells
_ALL_FACES = ('x_lo', 'x_hi', 'y_lo', 'y_hi', 'z_lo', 'z_hi')


def _access_eigenvalue():
    """Discrete 7-point eigenvalue [W/(m^3 K)] of prod sin(pi (i + 1/2) / N_a) with the half-cell Dirichlet faces:
    sum_a k_a (4 / h^2) sin^2(pi / (2 N_a)) (the face ghost value is the odd image, so the sines are exact)."""
    return sum(conductivity * 4.0 / CELL_SIZE ** 2 * math.sin(math.pi * CELL_SIZE / (2.0 * length)) ** 2
               for conductivity, length in zip(CONDUCTIVITY, BOX_SIZE))


def _access_expression():
    """Analytic T [K] of x, y, z [m]: T_w + A prod sin(pi s_a / L_a), exact at the cell centres."""
    return (f'{WALL_TEMPERATURE!r} + {ACCESS_AMPLITUDE!r}*sin(pi*x/{BOX_SIZE[0]!r})*sin(pi*y/{BOX_SIZE[1]!r})'
            f'*sin(pi*z/{BOX_SIZE[2]!r})')


def _access_exact(x, y, z):
    return WALL_TEMPERATURE + ACCESS_AMPLITUDE * (np.sin(np.pi * x / BOX_SIZE[0]) * np.sin(np.pi * y / BOX_SIZE[1])
                                                  * np.sin(np.pi * z / BOX_SIZE[2]))


def _access_cut_position(axis):
    return (ACCESS_CUT_CELLS[axis] + 0.5) * CELL_SIZE


def _access_segments():
    """{curve name: [x0, y0, z0, x1, y1, z1]} of the 3D-segment line curves: one along each axis through the cut
    cells, one oblique."""
    centre = [_access_cut_position(axis) for axis in range(3)]
    segments = {}
    for axis, axis_name in enumerate(_AXES):
        start, end = list(centre), list(centre)
        start[axis], end[axis] = 0.0, BOX_SIZE[axis]
        segments[f'along_{axis_name}'] = start + end
    segments['oblique'] = [cells * CELL_SIZE for point in ACCESS_OBLIQUE for cells in point]
    return segments


def _field_access_runs():
    """'field_access': all six faces at T_w, Q = lambda A prod sin (the discrete eigenvector), stored cuts normal to
    x and y, 3D-segment line curves of the analytic T."""
    source = (f'{_access_eigenvalue() * ACCESS_AMPLITUDE!r}*sin(pi*x/{BOX_SIZE[0]!r})*sin(pi*y/{BOX_SIZE[1]!r})'
              f'*sin(pi*z/{BOX_SIZE[2]!r})')
    spec = _box_spec('3D field access: product-of-sines temperature', [
        FeatureSpec('walls', 'temperature', list(_ALL_FACES), {'temperature': WALL_TEMPERATURE}),
        FeatureSpec('source', 'heat_source', ['all'], {'Q': source})])
    spec.study.stored_cuts = [['x', _access_cut_position(0)], ['y', _access_cut_position(1)]]
    spec.analytic_curves = [AnalyticCurveSpec(name, 'line', _access_expression(), line=line, field=TEMPERATURE_FIELD)
                            for name, line in _access_segments().items()]
    spec.analytic_curves.append(AnalyticCurveSpec(TRILINEAR_CURVE, 'line', _access_expression(),
                                                  line=_access_segments()['oblique'], field=TEMPERATURE_FIELD,
                                                  sampling='trilinear'))
    spec.analytic_curves.append(AnalyticCurveSpec(SNAPSHOT_CURVE, 'line', _access_expression(),
                                                  line=_access_segments()['oblique'], field=TEMPERATURE_FIELD,
                                                  snapshot=-1))
    return {'field_access': spec}


TRILINEAR_CURVE = 'oblique_trilinear'   # the oblique segment with sampling 'trilinear'
SNAPSHOT_CURVE = 'oblique_snapshot'     # the oblique segment read from the final snapshot's stored profile


def _trilinear_expected_error():
    """{max, rms} curve_error of the trilinear interpolant of the exact cell-centre field along the oblique segment
    (the samples of model.analytic.segment_cells: two per cell, all inside) against the exact field there."""
    start, end = (np.asarray(point, dtype=np.float64) * CELL_SIZE for point in ACCESS_OBLIQUE)
    length = float(np.linalg.norm(end - start))
    fraction = np.linspace(0.0, 1.0, int(np.clip(2.0 * length / CELL_SIZE, 2, 4000)))
    points = [low + fraction * (high - low) for low, high in zip(start, end)]
    corners = []   # per axis: (lower centre, upper centre, upper weight); every sample lies between two centres
    for coordinates, size in zip(points, BOX_SIZE):
        position = coordinates / CELL_SIZE - 0.5
        lower = np.clip(np.floor(position), 0, round(size / CELL_SIZE) - 2)
        corners.append(((lower + 0.5) * CELL_SIZE, (lower + 1.5) * CELL_SIZE, np.clip(position - lower, 0.0, 1.0)))
    interpolated = np.zeros(fraction.size)
    for x_centre, x_part in ((corners[0][0], 1.0 - corners[0][2]), (corners[0][1], corners[0][2])):
        for y_centre, y_part in ((corners[1][0], 1.0 - corners[1][2]), (corners[1][1], corners[1][2])):
            for z_centre, z_part in ((corners[2][0], 1.0 - corners[2][2]), (corners[2][1], corners[2][2])):
                interpolated += x_part * y_part * z_part * _access_exact(x_centre, y_centre, z_centre)
    exact = _access_exact(*points)
    relative = np.abs(interpolated - exact) / float(np.ptp(exact))
    return {'max': float(relative.max()), 'rms': float(np.sqrt(np.mean(relative ** 2)))}


def _cut_deviation(field_data, axis):
    """(max |T - exact| [K] of a cut normal to axis (0, 1, 2), its axes as expected and its position at the cell)."""
    in_plane = [index for index in range(3) if index != axis]
    u, v = np.meshgrid(np.asarray(field_data.x_centers, dtype=np.float64),
                       np.asarray(field_data.y_centers, dtype=np.float64), indexing='ij')
    coordinates = [None, None, None]
    coordinates[in_plane[0]], coordinates[in_plane[1]] = u, v
    coordinates[axis] = float(field_data.plane[1])
    deviation = float(np.max(np.abs(np.asarray(field_data.values, dtype=np.float64) - _access_exact(*coordinates))))
    geometry_ok = (tuple(field_data.axes) == tuple(_AXES[index] for index in in_plane)
                   and field_data.plane[0] == _AXES[axis]
                   and abs(float(field_data.plane[1]) - _access_cut_position(axis)) < 1e-3 * CELL_SIZE)
    return deviation, geometry_ok


def _check_field_access(results_by_name):
    """Cuts along x / y / z, the stored cuts of the final snapshot, the volume and the 3D-segment curve errors against
    the exact discrete field, / the amplitude; the cuts must equal the volume slices exactly."""
    results = results_by_name['field_access']
    deviations, details, problems = {}, {}, []
    volume = results.volume(TEMPERATURE_FIELD)
    grids = np.meshgrid(*(np.asarray(centres, dtype=np.float64) for centres in volume.centres), indexing='ij')
    volume_values = np.asarray(volume.values, dtype=np.float64)
    deviations['volume'] = float(np.max(np.abs(volume_values - _access_exact(*grids))))
    for axis, axis_name in enumerate(_AXES):
        plane = (axis_name, _access_cut_position(axis))
        field_data = results.field(TEMPERATURE_FIELD, plane=plane)
        deviations[f'cut_{axis_name}'], geometry_ok = _cut_deviation(field_data, axis)
        if not geometry_ok:
            problems.append(f'cut {axis_name}: axes {field_data.axes}, plane {field_data.plane}')
        volume_slice = np.take(volume.values, ACCESS_CUT_CELLS[axis], axis=axis)
        if not np.array_equal(np.asarray(field_data.values), volume_slice):
            problems.append(f'cut {axis_name} differs from the volume slice')
        if axis_name in ('x', 'y'):   # stored cuts of the final snapshot: the same values
            stored = results.snapshot_field(-1, TEMPERATURE_FIELD, plane=plane)
            if not (np.array_equal(np.asarray(stored.values), np.asarray(field_data.values))
                    and tuple(stored.plane) == tuple(field_data.plane)):
                problems.append(f'stored cut {axis_name} differs from the live cut')
    relative = {name: value / ACCESS_AMPLITUDE for name, value in deviations.items()}
    curve_errors = results.summary()['curve_error']
    for name in _access_segments():
        record = curve_errors.get(name) or {}
        relative[f'segment_{name}'] = float(record.get('max', math.nan))   # / the curve's range; float() reads 'nan'
        details[f'segment_{name}'] = {'curve_error': record}
        if int(record.get('n_points', 0)) < 2:
            problems.append(f"segment {name}: {record.get('note', 'no compared points')}")
    # trilinear: its curve_error is the interpolation error itself (~1e-2), so it must equal the numpy interpolant's
    record = curve_errors.get(TRILINEAR_CURVE) or {}
    expected = _trilinear_expected_error()
    gaps = [abs(float(record.get(key, math.nan)) - expected[key]) for key in ('max', 'rms')]   # float() reads 'nan'
    relative[f'segment_{TRILINEAR_CURVE}'] = max(gaps) if all(math.isfinite(gap) for gap in gaps) else math.inf
    details[f'segment_{TRILINEAR_CURVE}'] = {'curve_error': record, 'expected_curve_error': expected}
    if int(record.get('n_points', 0)) < 2:
        problems.append(f"segment {TRILINEAR_CURVE}: {record.get('note', 'no compared points')}")
    # the final snapshot's stored segment profile: the same curve_error as the live oblique segment
    snapshot_record, live_record = curve_errors.get(SNAPSHOT_CURVE) or {}, curve_errors.get('oblique') or {}
    gaps = [abs(float(snapshot_record.get(key, math.nan)) - float(live_record.get(key, math.nan)))
            for key in ('max', 'rms')]
    relative[f'segment_{SNAPSHOT_CURVE}'] = max(gaps) if all(math.isfinite(gap) for gap in gaps) else math.inf
    details[f'segment_{SNAPSHOT_CURVE}'] = {'curve_error': snapshot_record}
    if int(snapshot_record.get('n_points', 0)) != int(live_record.get('n_points', -1)):
        problems.append(f"segment {SNAPSHOT_CURVE}: {snapshot_record.get('note', 'n_points differ from oblique')}")
    finite = all(math.isfinite(value) for value in relative.values())
    error = max(relative.values()) if finite and not problems else math.inf
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': EXACT_IDENTITY_TOLERANCE,
            'details': {'relative_errors': relative, 'max_abs_deviation_K': deviations, 'problems': problems,
                        'eigenvalue': _access_eigenvalue(), 'volume_shape': list(volume.values.shape), **details}}


SELFTEST_CASES_3D = (
    SelfTestCase(
        name='heat_3d_field_access', label='Heat 3D: one analytic field through x / y / z cuts, segments, the volume',
        description=(f'igor 1.3: stationary 3D heat in a {BOX_SIZE[0] * 1e3:g} x {BOX_SIZE[1] * 1e3:g} x '
                     f'{BOX_SIZE[2] * 1e3:g} mm box of uniform {CELL_SIZE * 1e6:g} um cubic cells, anisotropic k_xx, '
                     f'k_yy, k_zz = {CONDUCTIVITY[0]:g}, {CONDUCTIVITY[1]:g}, {CONDUCTIVITY[2]:g} W/(m*K) (a swapped '
                     'component or axis changes lambda; this also pins the z face conductances, the z_lo / z_hi '
                     f'Dirichlet terms and the cell-volume source), float64 CG to 1e-12, all six faces at '
                     f'{WALL_TEMPERATURE:g} K and Q = lambda A prod_a sin(pi s_a / L_a), lambda = sum_a k_a (4 / '
                     'h^2) sin^2(pi / (2 N_a)): the 7-point operator with half-cell Dirichlet faces (odd ghost image) '
                     f'has the product of sines as an exact eigenvector, so T = T_w + {ACCESS_AMPLITUDE:g} K prod sin '
                     'at every cell centre. Read through Results.field cuts normal to x, y and z (values against '
                     'the exact field, FieldData axes and plane, each equal to the slice of Results.volume), the '
                     'StudySpec.stored_cuts planes x and y of the final snapshot (equal to the live cuts), the '
                     'volume itself and four 3D-segment line curves (along x, y, z through the cut cells and one '
                     'oblique; summary curve_error at the sampled cell centres, relative to the curve range), plus '
                     f'the oblique one with sampling trilinear ({TRILINEAR_CURVE}: its curve_error max / rms must '
                     'equal those of the numpy trilinear interpolant of the exact cell-centre field, ~1e-2) and '
                     f'with snapshot -1 ({SNAPSHOT_CURVE}: the profile stored with the final snapshot, its '
                     'curve_error equal to the live oblique one). '
                     'measured = the largest of the cut / volume deviations over the amplitude and the curve errors, '
                     f'tolerance {EXACT_IDENTITY_TOLERANCE:g} (float32 T); a mismatch of axes, planes or '
                     'slices fails the case (measured inf, listed in details problems).'),
        specs=_field_access_runs, check=_check_field_access),
    SelfTestCase(
        name='heat_3d_graded_z_robin', label='Heat 3D: Robin slab along z on a graded z axis',
        description=('igor 1.4: the heat_3d_field_access box (k_zz = '
                     f'{CONDUCTIVITY[2]:g} W/(m*K)) with {WALL_TEMPERATURE:g} K on z_lo, convection h = {ROBIN_H:g} '
                     f'W/(m^2*K), T_ext = {ROBIN_AMBIENT:g} K on z_hi and no source (q = (T_ext - T_w) / (L_z / k_zz + '
                     '1 / h); also the only 3D Robin face case), on mesh nonuniform: a centred box (z '
                     f'{GRADED_LAYER_Z[0]:g}..{GRADED_LAYER_Z[1]:g} L_z, no material of its own) refined to '
                     f'{GRADED_FINE_CELL * 1e6:g} um with margin 0, base {CELL_SIZE * 1e6:g} um, so x, y and z are '
                     'graded (uniform base cells at the z faces) and the box z faces are cell edges. Source-free 1D '
                     'conduction is exact on any widths '
                     '(series half-cell conductances, half-cell Dirichlet and Robin terms): T = T_w + q z / k_zz at '
                     'every cell centre. Checked on the mid-height cut and by the derived value max |T - linear| over '
                     f"every cell; measured = the larger / the rise, tolerance {EXACT_IDENTITY_TOLERANCE:g}. The check "
                     'raises unless the build has more z cells than the uniform grid and dx_min is the refinement '
                     f'size within 5 %. Not yet run at {GRADED_FINE_CELL * 1e6:g} um.'),
        specs=_graded_robin_runs, check=_check_graded_robin),
    SelfTestCase(
        name='heat_3d_z_expressions', label='Heat 3D: z-dependent conductivity and heat source',
        description=('igor 1.10: the heat_3d_field_access box with expressions of the height z. Run z_material: k_zz = '
                     f'{CONDUCTIVITY[2]:g} (1 + z / L_z) (a material property written one z plane at a time), z_lo at '
                     f'{WALL_TEMPERATURE:g} K, z_hi at {Z_HIGH_TEMPERATURE:g} K, no source (closed form T_lo + (T_hi - '
                     'T_lo) ln(1 + z / L_z) / ln 2). Run z_source: k_zz = '
                     f'{CONDUCTIVITY[2]:g}, Q = {Z_SOURCE_PEAK:g} z / L_z W/m^3 (a z-dependent feature value) between '
                     f'z faces at {WALL_TEMPERATURE:g} K (closed form T_w + Q0 z (L_z^2 - z^2) / (6 k L_z)). Both are '
                     'compared with the 1D finite-volume solution of the same cells (series half-cell conductances, '
                     'numpy solve, exact for the 7-point operator): the mid-height cut and the derived values mean of '
                     '(T - T_w) (z / L_z)^p, p = 0, 1, 2 (z in a field expression); measured = largest deviation / '
                     f'the run\'s rise, tolerance {EXACT_IDENTITY_TOLERANCE:g}. The closed-form gaps are reported.'),
        specs=_z_expression_runs, check=_check_z_expressions),
    SelfTestCase(
        name='heat_3d_z_couplings', label='Heat 3D: z in a field transfer and in a state-dependent conductivity',
        description=('igor 1.10: the heat_3d_field_access box (k_zz = '
                     f'{CONDUCTIVITY[2]:g} W/(m*K)). Run z_transfer: a second heat physics {REFERENCE_PHYSICS!r} '
                     f'between z faces at {WALL_TEMPERATURE:g} / {Z_HIGH_TEMPERATURE:g} K (exact linear T at the cell '
                     'centres) feeds a field_transfer into the volumetric_power slot of heat, expression Q0 (value - '
                     f'T_w) / dT z / L_z = Q0 (z / L_z)^2 (z = cell centres), Q0 = {Z_SOURCE_PEAK:g} W/m^3; heat has '
                     f'both z faces at {WALL_TEMPERATURE:g} K. Run z_dynamic: k_zz = k0 (1 + z / L_z) (1 + '
                     f'{Z_DYNAMIC_BETA:g} (T - T_w)) (a DynamicMaterialUpdate of z and T; stationary fixed point to '
                     f'{_Z_DYNAMIC_STUDY["tolerance"]:g}) between z faces at {WALL_TEMPERATURE:g} / '
                     f'{Z_HIGH_TEMPERATURE:g} K. Both against the 1D finite-volume solution of the same cells (z_dynamic: '
                     'its Picard fixed point): the mid-height cut and the derived values mean of (heat_T - T_w) (z / '
                     f"L_z)^p, p = 0, 1, 2; measured = largest deviation / the run's rise, tolerance "
                     f'{Z_COUPLED_TOLERANCE:g} (float32 T, the fixed point stalls at the float32 floor). The closed-form '
                     'gaps (z_dynamic: Kirchhoff transform) and the iteration counts are reported.'),
        specs=_z_coupling_runs, check=_check_z_couplings),
)


# ---- 'heated_chip_3d' template ----
CHIP_SUBSTRATE = (1e-3, 1e-3, 5e-4)   # m, silicon block (x, y, z)
CHIP_FOOTPRINT = (3e-4, 7e-4)         # m, x and y extent of the heated chip on top
CHIP_THICKNESS = 5e-5                 # m, heated layer under z_hi
CHIP_POWER_DENSITY = 1e11             # W/m^3: 0.8 W in the chip
CHIP_POWER = (CHIP_POWER_DENSITY * (CHIP_FOOTPRINT[1] - CHIP_FOOTPRINT[0]) ** 2 * CHIP_THICKNESS)
SILICON_CONDUCTIVITY = 130.0          # W/(m*K)
SINK_TEMPERATURE = 300.0              # K on z_lo

HEATED_CHIP_TEMPLATE_DESCRIPTION = (
    f'Heat (stationary, 3D): a {CHIP_SUBSTRATE[0] * 1e3:g} x {CHIP_SUBSTRATE[1] * 1e3:g} x {CHIP_SUBSTRATE[2] * 1e3:g} '
    f'mm silicon block (k = {SILICON_CONDUCTIVITY:g} W/(m*K)) on a {SINK_TEMPERATURE:g} K heat sink (z_lo) with a '
    f'{(CHIP_FOOTPRINT[1] - CHIP_FOOTPRINT[0]) * 1e3:g} mm square chip dissipating {CHIP_POWER:g} W in its top '
    f'{CHIP_THICKNESS * 1e6:g} um, convection on z_hi; derived value T_max.')
HEATED_CHIP_TEMPLATE_SUMMARY = (
    'Steady 3D heat: a silicon block on a heat sink with a heated chip on top, air cooling above; shows the hot spot '
    'in z cuts.')


def heated_chip_3d_spec():
    """3D 'heated_chip_3d' template: silicon block, heated box under z_hi, z_lo held, z_hi convection."""
    low, high = CHIP_FOOTPRINT
    size_x, size_y, size_z = CHIP_SUBSTRATE
    return ModelSpec(
        title='Heated chip on a silicon block (3D)',
        domain=DomainSpec(size_x=size_x, size_y=size_y, size_z=size_z),
        geometry=[GeometrySpec('chip', 'box', {'x0': low, 'y0': low, 'z0': size_z - CHIP_THICKNESS,
                                               'x1': high, 'y1': high, 'z1': size_z})],
        materials=[MaterialSpec('silicon', properties={'k': SILICON_CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=[
            FeatureSpec('sink', 'temperature', ['z_lo'], {'temperature': SINK_TEMPERATURE}),
            FeatureSpec('air', 'convection', ['z_hi'], {'h': 10.0, 'T_ext': 293.15}),
            FeatureSpec('chip_power', 'heat_source', ['chip'], {'Q': CHIP_POWER_DENSITY})])],
        mesh=MeshSpec(mode='uniform', dx=2.5e-5),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T_max', 'T', [], 'max')])
