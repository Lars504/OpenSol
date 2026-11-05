"""Unbounded decaying physics by outer grading (symmetry plan 2.15): an axisymmetric disc heater on a half-space whose
held far faces sit 1000 radii away on a few dozen geometrically growing cells, against the infinite-medium closed
forms; one CPU case (cpu backend, Automatic sparse_direct), registered by definition.register."""
import numpy as np

from ...model.schema import SelfTestCase
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, RefinementSpec, SolverDefaultsSpec, StudySpec)

HEAT_TYPE_NAME = 'heat'            # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'heat'
RUN_NAME = 'unbounded'

HEAT_FLUX = 1e4                    # W/m^2 into the disc
CONDUCTIVITY = 0.2                 # W/(m*K)
FAR_TEMPERATURE = 293.15           # K, held on the far faces x_hi (r = L) and y_lo (depth L)
DISC_RADIUS = 1e-3                 # m
FINE_CELL = DISC_RADIUS / 20.0     # m, uniform cells of the refined core (20 per radius, as the bounded example)
CORE_SIZE = 2.0 * DISC_RADIUS      # m, refined core 0..2a along r and the top 2a along z
DOMAIN_SIZE = 1000.0 * DISC_RADIUS   # m: L, a zero face effectively at infinity (truncation ~ a / (2 L) of the rise)
OUTER_CELL = 100.0 * DISC_RADIUS   # m, outer_dx
OUTER_RATIO = 1.2                  # outer_ratio
MAX_CELLS_PER_AXIS = 150           # vs 20000 uniform cells across L
DEPTH = DISC_RADIUS + 0.5 * FINE_CELL   # m, second axis probe (a cell centre)
# Closed forms (Carslaw and Jaeger, uniform flux disc on a half-space) carry the disc-edge singularity at 20 cells per
# radius (~1 %, as examples/heat/disc_heater_half_space); truncation at L adds q a^2 / (2 k L) ~ 5e-4 of the rise.
TRUNCATION_RISE = HEAT_FLUX * DISC_RADIUS ** 2 / (2.0 * CONDUCTIVITY * DOMAIN_SIZE)   # K
TOLERANCES = {'theta_centre': 3e-2, 'theta_depth': 3e-2, 'heater_mean_rise': 1.5e-2, 'P_total': 1e-5}


def _axis_rise(depth):
    """Infinite half-space axis rise q / k (sqrt(a^2 + z^2) - z) [K] minus the truncation shift."""
    return HEAT_FLUX / CONDUCTIVITY * (np.hypot(DISC_RADIUS, depth) - depth) - TRUNCATION_RISE


def _expected():
    """Closed-form value of every derived value (rises in K, P in W)."""
    return {'theta_centre': _axis_rise(0.5 * FINE_CELL), 'theta_depth': _axis_rise(DEPTH),
            'heater_mean_rise': (8.0 * HEAT_FLUX * DISC_RADIUS / (3.0 * np.pi * CONDUCTIVITY)
                                 - HEAT_FLUX * FINE_CELL / (2.0 * CONDUCTIVITY) - TRUNCATION_RISE),
            'P_total': HEAT_FLUX * np.pi * DISC_RADIUS ** 2}


def _unbounded_runs():
    """Axisymmetric heat on L x L (x = r, axis x_lo; surface y_hi insulated, q / dx in the top cell layer r < a), T held
    on x_hi and y_lo; nonuniform mesh: the core at FINE_CELL, beyond it outer grading to OUTER_CELL."""
    size = DOMAIN_SIZE
    geometry = [
        GeometrySpec('heater', 'rectangle', {'x0': 0.0, 'y0': size - FINE_CELL, 'x1': DISC_RADIUS, 'y1': size}),
        GeometrySpec('near_core', 'rectangle', {'x0': 0.0, 'y0': size - CORE_SIZE, 'x1': CORE_SIZE, 'y1': size}),
        GeometrySpec('centre_point', 'point', {'x': 0.5 * FINE_CELL, 'y': size - 0.5 * FINE_CELL}),
        GeometrySpec('depth_point', 'point', {'x': 0.5 * FINE_CELL, 'y': size - DEPTH})]
    features = [
        FeatureSpec('far_field', 'temperature', ['x_hi', 'y_lo'], {'temperature': FAR_TEMPERATURE}),
        FeatureSpec('disc_flux', 'heat_source', ['heater'], {'Q': HEAT_FLUX / FINE_CELL}),
        FeatureSpec('centre', 'temperature_probe', ['centre_point']),
        FeatureSpec('depth', 'temperature_probe', ['depth_point'])]
    rise = f'{PHYSICS_NAME}_T - {FAR_TEMPERATURE!r}'
    derived = [
        DerivedValueSpec('theta_centre', f'values - {FAR_TEMPERATURE!r}', operation='last',
                         source=f'{PHYSICS_NAME}.centre'),
        DerivedValueSpec('theta_depth', f'values - {FAR_TEMPERATURE!r}', operation='last',
                         source=f'{PHYSICS_NAME}.depth'),
        DerivedValueSpec('heater_mean_rise', rise, ['heater'], 'mean'),
        DerivedValueSpec('P_total', f'{PHYSICS_NAME}_Q_ext', [], 'integral')]
    return {RUN_NAME: ModelSpec(
        title='Disc heater on an unbounded half-space (outer grading)',
        domain=DomainSpec(size_x=size, size_y=size),
        geometry=geometry,
        materials=[MaterialSpec('solid', properties={'k': CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME,
                             settings={'geometry': 'axisymmetric', 'T0': FAR_TEMPERATURE}, features=features)],
        mesh=MeshSpec(mode='nonuniform', dx=2.0 * FINE_CELL, max_ratio=1.15, outer_dx=OUTER_CELL,
                      outer_ratio=OUTER_RATIO, refinements=[RefinementSpec('near_core', FINE_CELL, 0.0)]),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'),
        derived_values=derived)}


def _check_unbounded(results_by_name):
    """error = largest relative deviation over its own tolerance (tolerance 1), + 1 when an axis has more than
    MAX_CELLS_PER_AXIS cells."""
    results = results_by_name[RUN_NAME]
    measured = results.summary()['derived']
    expected = _expected()
    relative = {name: abs(float(measured[name]) / value - 1.0) for name, value in expected.items()}   # NaN propagates
    error = max(relative[name] / TOLERANCES[name] for name in relative)
    cells = list(np.asarray(results.field(f'{PHYSICS_NAME}.T').values).shape)
    if max(cells) > MAX_CELLS_PER_AXIS:
        error += 1.0
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': 1.0,
            'details': {'relative_errors': relative, 'tolerances': TOLERANCES,
                        'measured': {name: float(measured[name]) for name in expected},
                        'expected': expected, 'truncation_rise_K': TRUNCATION_RISE, 'n_cells': cells,
                        'uniform_cells_per_axis': int(round(DOMAIN_SIZE / FINE_CELL))}}


UNBOUNDED_SELFTEST_CASES = (
    SelfTestCase(
        name='heat_unbounded_disc_heater', label='Heat: disc heater on an unbounded half-space by outer grading',
        description=('Symmetry plan 2.15 (far-field grading of decaying quasi-static physics): stationary axisymmetric '
                     f'heat, a uniform flux q = {HEAT_FLUX:g} W/m^2 into a disc of radius a = {DISC_RADIUS * 1e3:g} mm '
                     f'(q / dx in the top cell layer) on the insulated surface of a half-space (k = {CONDUCTIVITY:g} '
                     f'W/(m*K)) whose far faces x_hi and y_lo, L = {DOMAIN_SIZE / DISC_RADIUS:g} a away, are held at '
                     f'{FAR_TEMPERATURE:g} K. Mesh: {FINE_CELL * 1e6:g} um cells on the 2a core around the disc, '
                     f'beyond it outer grading at ratio {OUTER_RATIO:g} to {OUTER_CELL * 1e3:g} mm cells (fewer than '
                     f'{MAX_CELLS_PER_AXIS} cells per axis instead of {round(DOMAIN_SIZE / FINE_CELL)}); cpu backend, '
                     'Automatic linear solver (SuperLU). Expected: the infinite-medium closed forms (Carslaw and '
                     'Jaeger) of the axis rise q / k (sqrt(a^2 + z^2) - z) at z = dx/2 and a + dx/2, the heater mean '
                     '8 q a / (3 pi k) - q dx / (2 k), each minus the truncation q a^2 / (2 k L), and the ring-weighted '
                     'source integral q pi a^2. A box truncated at 10 a misses 6 % of the mean rise. error = largest '
                     'relative error over its tolerance (centre / depth 3e-2, mean 1.5e-2: the disc-edge singularity '
                     'at 20 cells per radius; power 1e-5), + 1 above the cell bound; tolerance 1 (estimated, not yet '
                     'confirmed by a run).'),
        specs=_unbounded_runs, check=_check_unbounded, max_wall_time=120.0),
)
