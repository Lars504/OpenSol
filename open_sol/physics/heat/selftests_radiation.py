"""Self-tests heat_radiation_slab / heat_radiation_slab_triangles: steady slab heated by a uniform source, insulated at
x_lo and radiating at x_hi, against the exact finite-volume solution (energy balance Q L = eps sigma (T_N^4 - T_amb^4)
at the last cell) and, on P2 triangles, the exact quadratic profile."""
import numpy as np

from ...core.constants import STEFAN_BOLTZMANN
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import with_exact_cg
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)

HEAT_TYPE_NAME = 'heat'        # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'heat'
CELL = 1e-5                    # m, uniform mesh
CELLS = 20                     # along x (2 cells across, insulated y faces)
CONDUCTIVITY = 0.02            # W/(m K): ~14 K across the slab
EMISSIVITY = 0.8
AMBIENT_T = 293.15             # K
POWER_DENSITY = 2.77e7         # W/m^3: Q L = 5540 W/m^2, the radiating cell near 600 K
TOLERANCE = 1e-5               # of max |T - T_exact| / (T_N - T_amb): float32 T, Newton to 1e-7, CG to 1e-12
TRIANGLE_ELEMENT = 4.0 * CELL  # m, uniform P2 triangles: the 1D quadratic profile is exact on them
TRIANGLE_TOLERANCE = 1e-4      # of |T - T_exact| / (T_N - T_amb) at the probes (float32 state, Newton to 1e-7)


def _radiation_spec():
    """Strip of CELLS x 2 cells, heat_source on all, radiation on x_hi (the only anchor)."""
    return with_exact_cg(ModelSpec(
        title='Self-test: slab cooled by surface radiation',
        domain=DomainSpec(size_x=CELLS * CELL, size_y=2.0 * CELL),
        materials=[MaterialSpec('conductor', properties={'k': CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=[
            FeatureSpec('source', 'heat_source', ['all'], {'Q': POWER_DENSITY}),
            FeatureSpec('glow', 'radiation', ['x_hi'], {'emissivity': EMISSIVITY, 'T_amb': AMBIENT_T})])],
        mesh=MeshSpec(mode='uniform', dx=CELL),
        study=StudySpec(type='stationary')))


def _exact_profile():
    """Cell-centre T of the finite-volume solution: the last cell carries the whole Q L by radiation (cell-centre T),
    face j -> j + 1 carries Q (j + 1) h, a drop of Q (j + 1) h^2 / k."""
    last = (AMBIENT_T ** 4 + POWER_DENSITY * CELLS * CELL / (EMISSIVITY * STEFAN_BOLTZMANN)) ** 0.25
    drops = POWER_DENSITY * np.arange(1, CELLS) * CELL * CELL / CONDUCTIVITY   # face j: cell j - cell j + 1
    rises = np.concatenate([np.cumsum(drops[::-1])[::-1], [0.0]])
    return last + rises


def _check_radiation(results):
    field = np.asarray(results.field(f'{PHYSICS_NAME}.T').values, dtype=np.float64)
    expected = _exact_profile()
    scale = float(expected[-1] - AMBIENT_T)
    if field.shape[0] != CELLS:
        return {'measured': float('nan'), 'expected': float(expected[-1]), 'error': float('inf'),
                'tolerance': TOLERANCE, 'details': {'shape': list(field.shape)}}
    error = float(np.max(np.abs(field - expected[:, None]))) / scale
    return {'measured': float(field[-1, 0]), 'expected': float(expected[-1]), 'error': error, 'tolerance': TOLERANCE,
            'details': {'hot_end_measured_K': float(field[0, 0]), 'hot_end_expected_K': float(expected[0]),
                        'radiated_W_per_m2': POWER_DENSITY * CELLS * CELL}}


def _triangle_spec():
    """The same slab on uniform P2 triangles (cpu, host SuperLU), probes at x = CELL / 2 and x = L / 2 (mid-height)."""
    length, width = CELLS * CELL, 2.0 * CELL
    return ModelSpec(
        title='Self-test: slab cooled by surface radiation on triangles',
        domain=DomainSpec(size_x=length, size_y=width),
        geometry=[GeometrySpec('hot_point', 'point', {'x': 0.5 * CELL, 'y': 0.5 * width}),
                  GeometrySpec('middle_point', 'point', {'x': 0.5 * length, 'y': 0.5 * width})],
        materials=[MaterialSpec('conductor', properties={'k': CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=[
            FeatureSpec('source', 'heat_source', ['all'], {'Q': POWER_DENSITY}),
            FeatureSpec('glow', 'radiation', ['x_hi'], {'emissivity': EMISSIVITY, 'T_amb': AMBIENT_T}),
            FeatureSpec('hot_end', 'temperature_probe', ['hot_point']),
            FeatureSpec('middle', 'temperature_probe', ['middle_point'])])],
        meshes=[MeshSpec(name='triangles', kind='triangles', mode='uniform', dx=TRIANGLE_ELEMENT)],
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'))


def _continuous_profile(x):
    """T(x) = T_L + Q (L^2 - x^2) / (2 k), T_L = (T_amb^4 + Q L / (eps sigma))^(1/4) at the radiating face x = L."""
    length = CELLS * CELL
    face = (AMBIENT_T ** 4 + POWER_DENSITY * length / (EMISSIVITY * STEFAN_BOLTZMANN)) ** 0.25
    return face + POWER_DENSITY * (length ** 2 - x ** 2) / (2.0 * CONDUCTIVITY)


def _check_triangles(results):
    length = CELLS * CELL
    scale = float(_continuous_profile(length) - AMBIENT_T)
    measured, errors = {}, {}
    for feature, x in (('hot_end', 0.5 * CELL), ('middle', 0.5 * length)):
        values = np.asarray(results.result(PHYSICS_NAME, feature)['values'], dtype=np.float64)
        measured[feature] = float(values[-1]) if values.size else float('nan')
        errors[feature] = abs(measured[feature] - float(_continuous_profile(x))) / scale
    error = max(errors.values())
    return {'measured': measured['hot_end'], 'expected': float(_continuous_profile(0.5 * CELL)),
            'error': error if np.isfinite(error) else float('inf'), 'tolerance': TRIANGLE_TOLERANCE,
            'details': {'measured_K': measured, 'errors': errors}}


RADIATION_SELFTEST_CASES = (
    SelfTestCase(
        name='heat_radiation_slab', label='Heat: steady slab cooled by surface radiation',
        description=(f'Steady heat in a {CELLS} x 2 cell strip ({CELL * 1e6:g} um cells, k = {CONDUCTIVITY:g} '
                     f'W/(m K)) with Q = {POWER_DENSITY:g} W/m^3 everywhere, x_lo insulated and x_hi radiating '
                     f'(emissivity {EMISSIVITY:g}, T_amb {AMBIENT_T:g} K; the only anchor). Exact finite-volume '
                     'solution: the radiating boundary cell at T_N = (T_amb^4 + Q L / (eps sigma))^(1/4) (the '
                     'feature uses the boundary-cell temperature), each interior face carrying Q x its distance to '
                     'x_lo. error = max |T - T_exact| / (T_N - T_amb) over every cell; tolerance '
                     f'{TOLERANCE:g}. Checks the Newton re-linearisation loop of a steady solve. Not yet run.'),
        specs=lambda: {'run': _radiation_spec()}, check=lambda results_by_name: _check_radiation(
            results_by_name['run'])),
    SelfTestCase(
        name='heat_radiation_slab_triangles', label='Heat: steady slab cooled by surface radiation on triangles',
        description=('The heat_radiation_slab strip on uniform P2 triangles '
                     f'({TRIANGLE_ELEMENT * 1e6:g} um, cpu): radiation on x_hi is a Robin condition per boundary '
                     'edge linearised at the edge-midpoint temperature. The 1D profile T_L + Q (L^2 - x^2) / (2 k) '
                     'with T_L = (T_amb^4 + Q L / (eps sigma))^(1/4) is quadratic, so P2 represents it exactly. '
                     'error = max |T - T_exact| / (T_L - T_amb) at probes x = 5 um and L / 2; tolerance '
                     f'{TRIANGLE_TOLERANCE:g}. Not yet run.'),
        specs=lambda: {'run': _triangle_spec()}, check=lambda results_by_name: _check_triangles(
            results_by_name['run'])),
)
