"""Analytic validation of axisymmetric (r, z) heat (B4): a tube heated through its inner perimeter (exact discrete
identity) and the azimuthal order 2 harmonic (symmetry 2.14); registered by definition.register."""
import math

import numpy as np

from ...model.schema import SelfTestCase, SelfTestVariant
from ...model.selftest_helpers import with_exact_cg
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, StudySpec)

HEAT_TYPE_NAME = 'heat'            # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'heat'
TEMPERATURE_FIELD = f'{PHYSICS_NAME}.T'
RUN_NAME = 'run'

CYLINDER_RADIUS = 2e-4             # m, x extent (x = r, the axis on x_lo)
CYLINDER_HEIGHT = 1e-4             # m, y extent (10 cells), insulated z faces
CELL_SIZE = 1e-5                   # m, uniform: 20 cells along r
THERMAL_CONDUCTIVITY = 2.0         # W/(m*K)
WALL_TEMPERATURE = 300.0           # K, held on x_hi (r = R)
AXISYMMETRIC_SETTINGS = {'geometry': 'axisymmetric'}
EXACT_IDENTITY_TOLERANCE = 1e-5    # float32 T (~3e-5 K near 320 K) over rises of ~20 K, ~1e-6

# heat_axisymmetric_inner_flux: a long tube (insulated z faces) of conductor r_i <= r <= R, no material inside, heated
# by heat_flux q on the tube object's perimeter (only its inner face r = r_i is exposed), T(R) held. Every x-face
# carries the ring power q 2 pi r_i dz, so exactly T_j = T_w + (q r_i / k) [dr / (2 R) + sum of dr / r_f over the faces
# between cell j and R] (a midpoint rule of (q r_i / k) ln(R / r)), and the ring-weighted Q_ext integrates to
# q 2 pi r_i L. A perimeter face weighted at the cell-centre radius (r_c = 2.5 dr, r_i = 2 dr) injects 25 % more.
TUBE_INNER_CELLS = 2               # r_i in cells
TUBE_INNER_RADIUS = TUBE_INNER_CELLS * CELL_SIZE   # m
TUBE_FLUX = 1e6                    # W/m^2 into the tube: rise ~ q r_i ln(R / r_i) / k ~ 20 K

# heat_azimuthal_order_2 (symmetry 2.14): the order-2 harmonic T - T0 = b r^2 cos(2 phi) (= b (x^2 - y^2)) in the
# cylinder, fed only by the inward heat_flux q = 2 k b R on x_hi (insulated z faces, no temperature anchor: the
# k m^2 / r^2 term anchors). On uniform dr (r_c = (i + 1/2) dr) the rise b r_c^2 is exact: the net radial outflow of
# cell i, k b pi dz dr^2 (4 i + 2) (faces pi r_f dz at m != 0), equals 4 k / r_c^2 x its volume pi r_c dr dz x b r_c^2,
# and the Neumann face matches the interior pattern flux 2 k b R. The pi-ring integral of T is sum (T0 + b r_c^2) pi r_c
# dr H (the reference T0 only shifts T: the solve is for the amplitude T - T0).
ORDER_2_RISE = 20.0                # K, b R^2
ORDER_2_B = ORDER_2_RISE / CYLINDER_RADIUS ** 2   # K/m^2
ORDER_2_FLUX = 2.0 * THERMAL_CONDUCTIVITY * ORDER_2_B * CYLINDER_RADIUS   # W/m^2 inward on x_hi
ORDER_2_T0 = 300.0                 # K
ORDER_2_SETTINGS = {'geometry': 'axisymmetric', 'azimuthal_order': 2, 'T0': ORDER_2_T0}


def _tube_runs():
    """Axisymmetric tube r_i..R (conductor), heat_flux on its perimeter, T_w on x_hi, empty core."""
    return {RUN_NAME: with_exact_cg(ModelSpec(
        title='Axisymmetric tube heated through its inner surface',
        domain=DomainSpec(size_x=CYLINDER_RADIUS, size_y=CYLINDER_HEIGHT),
        geometry=[GeometrySpec('tube', 'rectangle', {'x0': TUBE_INNER_RADIUS, 'y0': 0.0, 'x1': CYLINDER_RADIUS,
                                                     'y1': CYLINDER_HEIGHT})],
        materials=[MaterialSpec('conductor', properties={'k': THERMAL_CONDUCTIVITY}, selection=['tube'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, settings=dict(AXISYMMETRIC_SETTINGS), features=[
            FeatureSpec('wall', 'temperature', ['x_hi'], {'temperature': WALL_TEMPERATURE}),
            FeatureSpec('inner_flux', 'heat_flux', ['tube'], {'flux': TUBE_FLUX})])],
        mesh=MeshSpec(mode='uniform', dx=CELL_SIZE),
        study=StudySpec(type='stationary')))}


def _tube_expected_rise(n_radial):
    """Discrete rise [K] of the conductor cells j = r_i / dr .. n_radial - 1 (face r_f = m dr between cells m-1, m)."""
    scale = TUBE_FLUX * TUBE_INNER_RADIUS / THERMAL_CONDUCTIVITY
    rises = []
    for cell in range(TUBE_INNER_CELLS, n_radial):
        rises.append(scale * (CELL_SIZE / (2.0 * CYLINDER_RADIUS)
                              + sum(1.0 / face for face in range(cell + 1, n_radial))))
    return np.asarray(rises)


def _check_tube(results_by_name):
    """Largest deviation of the conductor cells from the discrete solution relative to its peak rise, and the
    relative error of the ring-weighted Q_ext integral against q 2 pi r_i L."""
    results = results_by_name[RUN_NAME]
    values, r_centers, _ = _field_arrays(results)
    expected_rise = _tube_expected_rise(r_centers.size)
    peak_rise = float(expected_rise[0])
    deviation = np.abs(values[TUBE_INNER_CELLS:, :] - (WALL_TEMPERATURE + expected_rise[:, None]))
    temperature_error = float(np.max(deviation)) / peak_rise   # NaN propagates
    source = results.field(f'{PHYSICS_NAME}.Q_ext')
    q_ext = np.asarray(source.values, dtype=np.float64)
    ring_volumes = (2.0 * math.pi * np.asarray(source.x_centers, dtype=np.float64) * CELL_SIZE * CELL_SIZE)
    injected = float(np.sum(q_ext * ring_volumes[:, None]))
    expected_power = TUBE_FLUX * 2.0 * math.pi * TUBE_INNER_RADIUS * CYLINDER_HEIGHT
    power_error = abs(injected - expected_power) / expected_power
    return {'measured': temperature_error, 'expected': 0.0, 'error': max(temperature_error, power_error),
            'tolerance': EXACT_IDENTITY_TOLERANCE,
            'details': {'errors': {'temperature': temperature_error, 'power': power_error},
                        'peak_rise_K': peak_rise, 'measured_peak_rise_K': float(np.max(values) - WALL_TEMPERATURE),
                        'injected_power_W': injected, 'expected_power_W': expected_power,
                        'n_cells': list(values.shape)}}


def _order_2_runs():
    """Axisymmetric order-2 cylinder: conductor everywhere, heat_flux 2 k b R on x_hi, no temperature anchor, the
    pi-ring integral of T."""
    return {RUN_NAME: with_exact_cg(ModelSpec(
        title='Axisymmetric heat at azimuthal order 2',
        domain=DomainSpec(size_x=CYLINDER_RADIUS, size_y=CYLINDER_HEIGHT),
        materials=[MaterialSpec('conductor', properties={'k': THERMAL_CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, settings=dict(ORDER_2_SETTINGS), features=[
            FeatureSpec('edge_flux', 'heat_flux', ['x_hi'], {'flux': ORDER_2_FLUX})])],
        mesh=MeshSpec(mode='uniform', dx=CELL_SIZE),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('T_integral', f'{PHYSICS_NAME}_T', [], 'integral')]))}


def _check_order_2(results_by_name):
    """Largest |T - T0 - b r_c^2| relative to b R^2 and the relative error of the pi-ring integral of T."""
    results = results_by_name[RUN_NAME]
    values, _, z_centers = _field_arrays(results)
    n_radial = int(round(CYLINDER_RADIUS / CELL_SIZE))
    if values.shape != (n_radial, z_centers.size):
        raise ValueError(f'expected {n_radial} radial cells, got the window shape {values.shape}')
    r_centres = (np.arange(n_radial) + 0.5) * CELL_SIZE
    expected = ORDER_2_T0 + ORDER_2_B * r_centres ** 2
    temperature_error = float(np.max(np.abs(values - expected[:, None]))) / ORDER_2_RISE   # NaN propagates
    expected_integral = float(np.sum(expected * math.pi * r_centres * CELL_SIZE)) * CYLINDER_HEIGHT
    measured_integral = float(results.summary()['derived']['T_integral'])
    integral_error = abs(measured_integral / expected_integral - 1.0)
    return {'measured': temperature_error, 'expected': 0.0, 'error': max(temperature_error, integral_error),
            'tolerance': EXACT_IDENTITY_TOLERANCE,
            'details': {'errors': {'temperature': temperature_error, 'T_integral': integral_error},
                        'axis_cell_rise_K': float(values[0, 0] - ORDER_2_T0),
                        'edge_cell_rise_K': float(values[-1, 0] - ORDER_2_T0),
                        'T_integral': measured_integral, 'T_integral_expected': expected_integral,
                        'n_cells': list(values.shape)}}


def _field_arrays(results):
    """(float64 T values, x centres, y centres) of one run's temperature field."""
    field_data = results.field(TEMPERATURE_FIELD)
    return (np.asarray(field_data.values, dtype=np.float64), np.asarray(field_data.x_centers, dtype=np.float64),
            np.asarray(field_data.y_centers, dtype=np.float64))


AXISYMMETRIC_SELFTEST_CASES = (
    SelfTestCase(
        name='heat_axisymmetric_inner_flux', label='Heat: axisymmetric tube heated through its inner surface',
        description=(f'Stationary axisymmetric heat in a tube r_i = {TUBE_INNER_RADIUS * 1e6:g} um ({TUBE_INNER_CELLS} '
                     f'cells) .. R = {CYLINDER_RADIUS * 1e3:g} mm (rectangle object, k = {THERMAL_CONDUCTIVITY:g} '
                     f'W/(m*K), no material inside r_i, uniform {CELL_SIZE * 1e6:g} um cells, insulated z faces), '
                     f'heat_flux q = {TUBE_FLUX:g} W/m^2 on the tube perimeter (only its inner face is exposed) and '
                     f'T = {WALL_TEMPERATURE:g} K on x_hi. Every x-face carries the ring power q 2 pi r_i dz, so the '
                     'discrete solution is exactly T_j = T_w + (q r_i / k) [dr / (2 R) + sum of dr / r_f over the faces '
                     'between cell j and R] (a midpoint rule of (q r_i / k) ln(R / r)), and the ring-weighted Q_ext '
                     'integrates to q 2 pi r_i L. error = max(max |T - T_j| over the conductor cells / peak rise, '
                     f'relative power error), tolerance {EXACT_IDENTITY_TOLERANCE:g} (float32 T and Q_ext). This pins '
                     'the radius of object-perimeter x-faces: weighted at the cell-centre radius r_c = 2.5 dr instead '
                     'of r_i = 2 dr the tube takes 25 % more power.'),
        specs=_tube_runs, check=_check_tube),
    SelfTestCase(
        name='heat_azimuthal_order_2', label='Heat: azimuthal order 2 harmonic in a cylinder',
        description=(f'Stationary axisymmetric heat at azimuthal_order 2 (symmetry 2.14) in a cylinder R x H = '
                     f'{CYLINDER_RADIUS * 1e3:g} x {CYLINDER_HEIGHT * 1e3:g} mm (k = {THERMAL_CONDUCTIVITY:g} '
                     f'W/(m*K), uniform {CELL_SIZE * 1e6:g} um cells, insulated z faces, T0 = {ORDER_2_T0:g} K) fed '
                     f'only by heat_flux q = 2 k b R = {ORDER_2_FLUX:g} W/m^2 on x_hi, no temperature anchor (the k m^2 '
                     '/ r^2 term anchors; validation and the build anchor check must accept it): the cos(2 phi) part '
                     f'of the harmonic T - T0 = b (x^2 - y^2) = b r^2 cos(2 phi), b R^2 = {ORDER_2_RISE:g} K. Exact '
                     'on uniform dr: T - T0 = b r_c^2 (the net radial outflow k b pi dz dr^2 (4 i + 2) of cell i '
                     'equals 4 k / r_c^2 x its volume pi r_c dr dz x b r_c^2; the Neumann face matches the interior '
                     'flux), and the pi-ring derived integral of T is sum (T0 + b r_c^2) pi r_c dr H. error = max(max '
                     f'|T - expected| / (b R^2), relative integral error), tolerance {EXACT_IDENTITY_TOLERANCE:g} '
                     '(float32 T). This pins m^2 in the diagonal, the pi ring weights of Neumann faces against volumes '
                     'and of Session derived values, and the amplitude reading of T - T0. Variant cpu: the same run on '
                     'the cpu backend (numpy, the exact pcg node), same tolerance.'),
        specs=_order_2_runs, check=_check_order_2,
        variants=(SelfTestVariant('cpu', tolerance=EXACT_IDENTITY_TOLERANCE, backend='cpu'),)),
)
