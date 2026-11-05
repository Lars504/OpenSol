"""Analytic heat validation cases (the heated slab by mixed-precision CG, a uniform k_xy box on a graded mesh, the
graded k_xy convergence order and flux / dissipation post-processing (CPU), Robin strips along x and y), the point
source under convection spec / check (run by selftests_backend) and the 'heated_slab' template spec, registered by
definition.register; the spec factories return fresh objects on every call."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import EXACT_CG_TOLERANCE, solver_role, with_exact_cg
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, RefinementSpec, SolverSpec, StudySpec)

HEAT_TYPE_NAME = 'heat'            # = definition.PHYSICS_TYPE_NAME (definition imports this module)
PHYSICS_NAME = 'heat'
TEMPERATURE_FIELD = f'{PHYSICS_NAME}.T'
RUN_NAME = 'run'

SLAB_LENGTH = 1e-3                 # m, x extent between the two fixed-temperature faces
SLAB_HEIGHT = 1e-4                 # m, y extent; the insulated y faces keep the solution 1D
CELL_SIZE = 1e-5                   # m, uniform mesh (an even cell count: two cells sit symmetrically about the centre)
THERMAL_CONDUCTIVITY = 2.0         # W/(m*K)
COLD_TEMPERATURE = 300.0           # K: both walls of the heated slab, the held end of the Robin strips
HOT_TEMPERATURE = 400.0            # K: the Robin strips' T_ext, the cross-conduction box faces
TEMPERATURE_SPAN = HOT_TEMPERATURE - COLD_TEMPERATURE   # K
POWER_DENSITY = 1e9                # W/m^3, uniform source of the heated slab
PEAK_RISE = POWER_DENSITY * SLAB_LENGTH ** 2 / (8.0 * THERMAL_CONDUCTIVITY)        # K, Q L^2 / (8 k) = 62.5
UNIFORM_OFFSET = POWER_DENSITY * CELL_SIZE ** 2 / (8.0 * THERMAL_CONDUCTIVITY)     # K, discrete offset Q h^2 / (8 k)
# Error budget of the heated slab: CG ~1e-8, float32 T half a step (1.5e-5 K) ~2.4e-7 of the rise. Its uncancelled
# offset UNIFORM_OFFSET is 1e-4 of the rise, so the tolerance resolves it.
HEATED_SLAB_TOLERANCE = 1e-6       # on |rise - PEAK_RISE| / PEAK_RISE

# heat_cross_conduction_box_graded: every outer face at HOT_TEMPERATURE around a k_xy background with a k_xy block on a
# graded mesh, so T must stay uniform (any row of the cross stencil that does not sum to 0 exchanges heat with T0 at
# the window corners and the block corners). Uniform and graded widths share one assembly (core
# fv_diffusion.cross_quads), and the mesh keeps uniform zones (block, far field), so no separate uniform-mesh case.
BOX_SIZE = 2e-4                    # m, square window (20 x 20 base cells)
CROSS_BLOCK = (6e-5, 1.4e-4)       # m, x and y extent of the centred block (cell edges)
BACKGROUND_K_XY = 0.5              # W/(m*K), with k = THERMAL_CONDUCTIVITY: k_xy^2 < k_xx k_yy
BLOCK_K_XY = 1.0                   # W/(m*K)
DEFAULT_T0 = 293.15                # K, heat 'T0' default (the reference the spurious exchange pulls toward)
CROSS_BOX_TOLERANCE = 1e-5         # on max |T - HOT_TEMPERATURE| / (HOT_TEMPERATURE - DEFAULT_T0); float32 T ~3e-7
CROSS_GRADED_CELL = 2.5e-6         # m, block refinement (cells grade around it)
# heat_cross_conduction_graded_convergence (CPU, the core operator): -div(K grad u) = f with constant anisotropic K and
# u = sin(pi x / L) sin(pi y / L) (zero on every face, so no cross flux through them) on smoothly graded grids.
CROSS_CONVERGENCE_K = (2.0, 1.0, 0.8)      # W/(m*K) k_xx, k_yy, k_xy (k_xy^2 = 0.64 < k_xx k_yy = 2)
CROSS_CONVERGENCE_SIZE = 1e-3              # m, square side L
CROSS_CONVERGENCE_CELLS = (16, 32, 64)     # cells per axis
CROSS_X_WAVE = 0.5                         # x widths L (1 + a cos(2 pi s)) / n: largest / smallest 3
CROSS_Y_STRETCH = 1.5                      # y widths ~ e^(b s): largest / smallest e^1.5 ~ 4.5
CROSS_ORDER_TOLERANCE = 0.4                # on 2 - observed order (supraconvergent cell-centred FV: ~2)
# heat_cross_flux_dissipation (CPU, the DiffusionSystem post-processing shared with the potential sigma_xy): the
# 16-cell convergence grid with K = CROSS_CONVERGENCE_K, T = 0 on every face (T0 = 0) and a uniform source
CROSS_FLUX_CELLS = 16
CROSS_FLUX_SOURCE = 1e9                    # W/m^3
CROSS_FLUX_GRADIENT = (3e4, -2e4)          # K/m, the prescribed linear T of the flux check
CROSS_FLUX_TOLERANCE = 1e-6                # float64 solve and post-processing (~1e-12)
EXACT_IDENTITY_TOLERANCE = 1e-5    # cases below: float32 T (~3e-5 K near 400 K) over rises of 50-100 K, ~1e-6
# heat_convection_strips: held T at one end, convection (h, T_ext) at the other, insulated sides, anisotropic k (the
# across-strip component never enters a 1D profile, so a k_xx / k_yy swap shows). The 3-point scheme is exact for the
# linear profile: flux q = (T_ext - T_held) / (L_free / k_along + 1 / h), T = T_held + q s / k_along at distance s
# from the held face. Run 'along_x': uniform cells, a fixed_temperature_domain reservoir (cells 0..9) on x_lo, so
# L_free = 0.9 mm; run 'along_y': the strip rotated (temperature face y_lo, convection y_hi) on a graded mesh whose
# refinement band spans the full width, so every cell has dx = GRADED_FINE_CELL and dy grades from it up to CELL_SIZE.
K_ALONG = 2.0                      # W/(m*K), conductivity along the strip
K_ACROSS = 50.0                    # W/(m*K), conductivity across it
ROBIN_H = 2000.0                   # W/(m^2*K): 1 / h = 5e-4 m^2 K/W, comparable to L / k_along
RESERVOIR_LENGTH = 1e-4            # m, held block of run 'along_x' (a cell edge)
GRADED_FINE_CELL = 5e-6            # m, refinement cell of run 'along_y' (20 cells across)
GRADED_BAND = (4.5e-4, 5.5e-4)     # m, y extent of the refinement band


def _robin_strip_flux(free_length):
    """Heat flux [W/m^2] through a strip of free_length from the held COLD_TEMPERATURE face to convection at
    HOT_TEMPERATURE."""
    return TEMPERATURE_SPAN / (free_length / K_ALONG + 1.0 / ROBIN_H)


# Point source under convection (checked in diffusion_cpu_gpu, selftests_backend.py, on both backends): a point source
# P and an inward heat_flux on x_lo in the BOX_SIZE square (k_xx / k_yy = THERMAL_CONDUCTIVITY / POINT_K_YY),
# convection on x_hi, y_lo, y_hi. Steady state is exactly conservative: every Robin face of length dx carries
# (T_cell - T_ext) dx / (1 / h + dx / (2 k_normal)), and their sum is P + flux * BOX_SIZE.
POINT_K_YY = 8.0                   # W/(m*K)
POINT_POWER = 1.0                  # W/m
POINT_POSITION = (6.5e-5, 1.25e-4)   # m, centre of window cell (6, 12)
POINT_INWARD_FLUX = 1e4            # W/m^2 on x_lo
POINT_H = 1e5                      # W/(m^2*K): h dx / (2 k) = 0.25, so the half-cell series term matters
POINT_TOTAL_POWER = POINT_POWER + POINT_INWARD_FLUX * BOX_SIZE   # W/m, 3.0


HEATED_SLAB_TEMPLATE_DESCRIPTION = (
    f'Heat (stationary): a {SLAB_LENGTH * 1e3:g} mm slab with a uniform {POWER_DENSITY:g} W/m^3 source between two '
    f'{COLD_TEMPERATURE:g} K walls; peak rise Q L^2 / (8 k) = {PEAK_RISE:g} K.')
HEATED_SLAB_TEMPLATE_SUMMARY = (
    f'Steady heat: a {SLAB_LENGTH * 1e3:g} mm slab with a uniform heat source between two {COLD_TEMPERATURE:g} K '
    f'walls; peak rise {PEAK_RISE:g} K.')


# ---- specs ----
def _slab_spec(title, features, derived_values=()):
    """Stationary heat-only slab with the conductor everywhere on a uniform CELL_SIZE mesh."""
    return ModelSpec(
        title=title,
        domain=DomainSpec(size_x=SLAB_LENGTH, size_y=SLAB_HEIGHT),
        materials=[MaterialSpec('conductor', properties={'k': THERMAL_CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=list(features))],
        mesh=MeshSpec(mode='uniform', dx=CELL_SIZE),
        study=StudySpec(type='stationary'),
        derived_values=list(derived_values))


def heated_slab_spec():
    """Uniform source between two walls at COLD_TEMPERATURE, with derived value T_max (the 'heated_slab' template)."""
    return _slab_spec('Uniformly heated slab', [
        FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi'], {'temperature': COLD_TEMPERATURE}),
        FeatureSpec('source', 'heat_source', ['all'], {'Q': POWER_DENSITY})],
        [DerivedValueSpec('T_max', 'T', [], 'max')])


def _heated_slab_mixed_runs():
    """The template solved by a pcg node at precision 'mixed' (float32 CG passes, float64 residual) to 1e-12."""
    spec = heated_slab_spec()
    spec.solvers = [SolverSpec('cg_mixed', 'pcg',
                               {'precision': 'mixed', 'tolerance': EXACT_CG_TOLERANCE})]
    spec.physics[0].solvers = {'linear': 'cg_mixed'}
    return {RUN_NAME: spec}


def _check_heated_slab_mixed(results_by_name):
    """_check_heated_slab plus a raise unless summary 'solvers' records the mixed pcg with refinements."""
    linear = solver_role(results_by_name[RUN_NAME], PHYSICS_NAME)
    if linear.get('precision') != 'mixed' or not (linear.get('counters') or {}).get('refinements'):
        raise ValueError(f"expected a mixed pcg with refinements, summary 'solvers' records {linear!r}")
    record = _check_heated_slab(results_by_name)
    record['details'].update(counters=linear['counters'])
    return record


def _cross_conduction_box_runs():
    """All four faces at HOT_TEMPERATURE; k_xy BACKGROUND_K_XY everywhere but BLOCK_K_XY on the centred block, refined
    to CROSS_GRADED_CELL (cells grade around it)."""
    block_lo, block_hi = CROSS_BLOCK
    mesh = MeshSpec(mode='nonuniform', dx=CELL_SIZE, refinements=[RefinementSpec('block', CROSS_GRADED_CELL)])
    return {RUN_NAME: with_exact_cg(ModelSpec(
        title='Cross-conduction box at uniform temperature',
        domain=DomainSpec(size_x=BOX_SIZE, size_y=BOX_SIZE),
        geometry=[GeometrySpec('block', 'rectangle', {'x0': block_lo, 'y0': block_lo, 'x1': block_hi, 'y1': block_hi})],
        materials=[MaterialSpec('background', properties={'k': THERMAL_CONDUCTIVITY, 'k_xy': BACKGROUND_K_XY},
                                selection=['all']),
                   MaterialSpec('crystal', properties={'k': THERMAL_CONDUCTIVITY, 'k_xy': BLOCK_K_XY},
                                selection=['block'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=[
            FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi', 'y_lo', 'y_hi'], {'temperature': HOT_TEMPERATURE})])],
        mesh=mesh,
        study=StudySpec(type='stationary')))}


def _rectangle(name, x0, y0, x1, y1):
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})


def _convection_strip_runs():
    """'along_x': reservoir block held cold, convection on x_hi, uniform cells; 'along_y': temperature on y_lo,
    convection on y_hi, graded mesh with dx != dy."""
    convection = {'h': ROBIN_H, 'T_ext': HOT_TEMPERATURE}
    along_x = ModelSpec(
        title='Robin strip along x with a held reservoir',
        domain=DomainSpec(size_x=SLAB_LENGTH, size_y=SLAB_HEIGHT),
        geometry=[_rectangle('reservoir', 0.0, 0.0, RESERVOIR_LENGTH, SLAB_HEIGHT)],
        materials=[MaterialSpec('conductor', properties={'k_xx': K_ALONG, 'k_yy': K_ACROSS}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=[
            FeatureSpec('reservoir', 'fixed_temperature_domain', ['reservoir'], {'temperature': COLD_TEMPERATURE}),
            FeatureSpec('cooling', 'convection', ['x_hi'], dict(convection))])],
        mesh=MeshSpec(mode='uniform', dx=CELL_SIZE),
        study=StudySpec(type='stationary'))
    along_y = ModelSpec(
        title='Robin strip along y on a graded mesh',
        domain=DomainSpec(size_x=SLAB_HEIGHT, size_y=SLAB_LENGTH),
        geometry=[_rectangle('band', 0.0, GRADED_BAND[0], SLAB_HEIGHT, GRADED_BAND[1])],
        materials=[MaterialSpec('conductor', properties={'k_xx': K_ACROSS, 'k_yy': K_ALONG}, selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=[
            FeatureSpec('held', 'temperature', ['y_lo'], {'temperature': COLD_TEMPERATURE}),
            FeatureSpec('cooling', 'convection', ['y_hi'], dict(convection))])],
        mesh=MeshSpec(mode='nonuniform', dx=CELL_SIZE, refinements=[RefinementSpec('band', GRADED_FINE_CELL)]),
        study=StudySpec(type='stationary'))
    return {'along_x': with_exact_cg(along_x), 'along_y': with_exact_cg(along_y)}


def _point_source_convection_runs():
    """Point source and x_lo inward flux in the BOX_SIZE square, convection on the other three faces."""
    point_x, point_y = POINT_POSITION
    return {RUN_NAME: with_exact_cg(ModelSpec(
        title='Point heat source under convection',
        domain=DomainSpec(size_x=BOX_SIZE, size_y=BOX_SIZE),
        geometry=[GeometrySpec('source_point', 'point', {'x': point_x, 'y': point_y})],
        materials=[MaterialSpec('conductor', properties={'k_xx': THERMAL_CONDUCTIVITY, 'k_yy': POINT_K_YY},
                                selection=['all'])],
        physics=[PhysicsSpec(PHYSICS_NAME, HEAT_TYPE_NAME, features=[
            FeatureSpec('line_source', 'point_heat_source', ['source_point'], {'P': POINT_POWER}),
            FeatureSpec('inflow', 'heat_flux', ['x_lo'], {'flux': POINT_INWARD_FLUX}),
            FeatureSpec('cooling', 'convection', ['x_hi', 'y_lo', 'y_hi'],
                        {'h': POINT_H, 'T_ext': COLD_TEMPERATURE})])],
        mesh=MeshSpec(mode='uniform', dx=CELL_SIZE),
        study=StudySpec(type='stationary')))}


# ---- checks (CPU, on Results-like objects) ----
def _check_heated_slab(results_by_name):
    """Peak rise T_max - COLD_TEMPERATURE against Q L^2 / (8 k), relative error."""
    peak_temperature = float(results_by_name[RUN_NAME].summary()['derived']['T_max'])   # float() also reads 'nan'
    measured = peak_temperature - COLD_TEMPERATURE
    return {'measured': measured, 'expected': PEAK_RISE, 'error': abs(measured - PEAK_RISE) / PEAK_RISE,
            'tolerance': HEATED_SLAB_TOLERANCE,
            'details': {'T_max_K': peak_temperature, 'uniform_offset_K': UNIFORM_OFFSET}}


def _check_cross_conduction_box(results_by_name):
    """Largest deviation of T from HOT_TEMPERATURE over all window cells, relative to HOT_TEMPERATURE - T0; inf unless
    the x centre spacings vary by more than 1.5x."""
    values, x_centers, _ = _field_arrays(results_by_name[RUN_NAME])
    max_deviation = float(np.max(np.abs(values - HOT_TEMPERATURE)))   # NaN propagates
    measured = max_deviation / (HOT_TEMPERATURE - DEFAULT_T0)
    spacings = np.diff(x_centers)
    spacing_ratio = float(spacings.max() / spacings.min()) if spacings.size else 1.0
    error = measured if spacing_ratio > 1.5 else math.inf
    return {'measured': measured, 'expected': 0.0, 'error': error, 'tolerance': CROSS_BOX_TOLERANCE,
            'details': {'max_abs_deviation_K': max_deviation, 'n_cells': list(values.shape),
                        'x_spacing_ratio': spacing_ratio}}


def _graded_cross_edges(n_cells):
    """(x, y) edges [m] of the convergence grids: x L (s + a sin(2 pi s) / (2 pi)), y L (e^(b s) - 1) / (e^b - 1)."""
    s = np.linspace(0.0, 1.0, n_cells + 1)
    x_edges = CROSS_CONVERGENCE_SIZE * (s + CROSS_X_WAVE * np.sin(2.0 * math.pi * s) / (2.0 * math.pi))
    y_edges = CROSS_CONVERGENCE_SIZE * np.expm1(CROSS_Y_STRETCH * s) / math.expm1(CROSS_Y_STRETCH)
    return x_edges, y_edges


def _cross_conduction_error(n_cells):
    """Relative max error at the centres of the FV solution of -div(K grad u) = f, u = sin(pi x / L) sin(pi y / L)
    (u = 0 on every face: no cross flux through them), on the graded n_cells x n_cells grid (core operator, float64)."""
    import scipy.sparse.linalg
    from ...core.fv_diffusion import BCSide, ScalarBoundaryConfig, build_anisotropic_laplacian
    x_edges, y_edges = _graded_cross_edges(n_cells)
    dx, dy = np.diff(x_edges), np.diff(y_edges)
    wave = math.pi / CROSS_CONVERGENCE_SIZE
    x_phase, y_phase = wave * 0.5 * (x_edges[:-1] + x_edges[1:]), wave * 0.5 * (y_edges[:-1] + y_edges[1:])
    k_xx, k_yy, k_xy = CROSS_CONVERGENCE_K
    exact = np.outer(np.sin(x_phase), np.sin(y_phase))
    source = wave ** 2 * ((k_xx + k_yy) * exact - 2.0 * k_xy * np.outer(np.cos(x_phase), np.cos(y_phase)))
    held_zero = BCSide('dirichlet', value=0.0)
    bcs = ScalarBoundaryConfig(x_lo=held_zero, x_hi=held_zero, y_lo=held_zero, y_hi=held_zero)
    shape = (n_cells, n_cells)
    operator, rhs_bc, *_ = build_anisotropic_laplacian(np.full(shape, k_xx), np.full(shape, k_yy), np.full(shape, k_xy),
                                                       dx, dy, bcs, dtype=np.float64)
    rhs = (source * np.outer(dx, dy) + rhs_bc).ravel()
    solution = scipy.sparse.linalg.spsolve(operator.tocsc(), rhs).reshape(shape)
    return float(np.max(np.abs(solution - exact)) / np.max(np.abs(exact)))


def _check_cross_conduction_convergence(_results):
    """Observed order of the max error between the two finest graded grids against 2."""
    errors = [_cross_conduction_error(n_cells) for n_cells in CROSS_CONVERGENCE_CELLS]
    orders = [math.log2(coarse / fine) if coarse > 0.0 and fine > 0.0 else math.nan
              for coarse, fine in zip(errors[:-1], errors[1:])]
    order = orders[-1]
    error = max(0.0, 2.0 - order) if math.isfinite(order) else math.inf
    x_edges, y_edges = _graded_cross_edges(CROSS_CONVERGENCE_CELLS[-1])
    width_ratios = [float(np.max(np.diff(edges)) / np.min(np.diff(edges))) for edges in (x_edges, y_edges)]
    return {'measured': order, 'expected': 2.0, 'error': error, 'tolerance': CROSS_ORDER_TOLERANCE,
            'details': {'cells_per_axis': list(CROSS_CONVERGENCE_CELLS), 'relative_max_errors': errors,
                        'observed_orders': orders, 'width_ratio_x_y': width_ratios}}


def _check_cross_flux_dissipation(_results):
    """Solved k_xy square: dissipation integral = sum T Q V; prescribed linear T: flux = -K grad T (cpu backend)."""
    from ...core.constants import FACES
    from ...core.domain import Domain
    from ...core.fv_diffusion import BCSide
    from ...core.material_grid import MaterialGrid
    from ...core.mesh import NonUniformGrid2D
    from .solver import HeatPhysics
    x_edges, y_edges = _graded_cross_edges(CROSS_FLUX_CELLS)
    grid = NonUniformGrid2D(np.diff(x_edges), np.diff(y_edges), verbose=False)
    domain = Domain(grid, MaterialGrid(grid.nx, grid.ny), size_x=grid.size_x, size_y=grid.size_y, non_uniform=True,
                    npml_sides={face: 0 for face in FACES}, backend='cpu')
    heat = HeatPhysics(domain, T0=0.0, tol=EXACT_CG_TOLERANCE)
    k_xx, k_yy, k_xy = CROSS_CONVERGENCE_K
    for name, value in (('k_xx', k_xx), ('k_yy', k_yy), ('k_xy', k_xy)):
        domain.materials.assign(name, None, value)
    heat.materials_changed(('k_xx', 'k_yy', 'k_xy'))
    for face in FACES:
        heat.set_face_bc(face, BCSide('dirichlet', value=0.0))
    heat.add_heat_source(np.ones((grid.nx, grid.ny), dtype=bool), CROSS_FLUX_SOURCE)
    heat.step(0.0, math.inf)
    system = heat.system

    # (1) the operator's energy u^T A u = u^T b (T = 0 faces: b = Q V) must all be in the dissipation, cross part too
    cell_area = np.asarray(system.geometric_cell_area, dtype=np.float64)
    work = float(np.sum(np.asarray(system.unknown2d, dtype=np.float64) * CROSS_FLUX_SOURCE * cell_area))
    dissipated = float(np.sum(np.asarray(system.dissipation_density(), dtype=np.float64) * cell_area))
    dissipation_error = abs(dissipated - work) / work

    # (2) post-processing of a prescribed linear T: cross flux -k_xy (dT/dy, dT/dx) in every cell, the total -K grad T
    # in the cells off the outer faces (whose Dirichlet half cells see the linear T as a jump)
    gradient_x, gradient_y = CROSS_FLUX_GRADIENT
    x_centres, y_centres = 0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:])
    system.unknown2d[...] = gradient_x * x_centres[:, None] + gradient_y * y_centres[None, :]
    expected = (-(k_xx * gradient_x + k_xy * gradient_y), -(k_xy * gradient_x + k_yy * gradient_y))
    expected_cross = (-k_xy * gradient_y, -k_xy * gradient_x)
    scale = max(abs(value) for value in expected)
    interior = (slice(1, -1), slice(1, -1))
    flux_error = max(float(np.max(np.abs(np.asarray(flux, dtype=np.float64)[interior] - value)))
                     for flux, value in zip(system.flux_at_centres(), expected)) / scale
    cross_error = max(float(np.max(np.abs(np.asarray(flux, dtype=np.float64) - value)))
                      for flux, value in zip(system.cross_flux_at_centres(), expected_cross)) / scale
    error = max(dissipation_error, flux_error, cross_error)
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': CROSS_FLUX_TOLERANCE,
            'details': {'dissipation_vs_source_work': dissipation_error, 'interior_flux': flux_error,
                        'cross_flux_all_cells': cross_error, 'source_work_W_per_m': work}}


def _field_arrays(results):
    """(float64 T values, x centres, y centres) of one run's temperature field."""
    field_data = results.field(TEMPERATURE_FIELD)
    return (np.asarray(field_data.values, dtype=np.float64), np.asarray(field_data.x_centers, dtype=np.float64),
            np.asarray(field_data.y_centers, dtype=np.float64))


def _check_convection_strips(results_by_name):
    """Largest deviation from the linear Robin profile over both runs, relative to TEMPERATURE_SPAN; inf when run
    'along_y' did not get cells with dx != dy."""
    values_x, x_centers, _ = _field_arrays(results_by_name['along_x'])
    flux_x = _robin_strip_flux(SLAB_LENGTH - RESERVOIR_LENGTH)
    profile_x = COLD_TEMPERATURE + flux_x * np.maximum(x_centers - RESERVOIR_LENGTH, 0.0) / K_ALONG
    deviation_x = float(np.max(np.abs(values_x - profile_x[:, None])))   # NaN propagates

    values_y, x_centers_y, y_centers = _field_arrays(results_by_name['along_y'])
    flux_y = _robin_strip_flux(SLAB_LENGTH)
    profile_y = COLD_TEMPERATURE + flux_y * y_centers / K_ALONG
    deviation_y = float(np.max(np.abs(values_y - profile_y[None, :])))
    largest_dx = float(np.max(np.diff(x_centers_y))) if x_centers_y.size > 1 else GRADED_FINE_CELL
    largest_dy = float(np.max(np.diff(y_centers)))
    graded = largest_dy > 1.5 * largest_dx   # coarse rows outside the band: dy ~ CELL_SIZE, dx = GRADED_FINE_CELL

    measured = max(deviation_x, deviation_y) / TEMPERATURE_SPAN
    return {'measured': measured, 'expected': 0.0, 'error': measured if graded else math.inf,
            'tolerance': EXACT_IDENTITY_TOLERANCE,
            'details': {'max_abs_deviation_K': {'along_x': deviation_x, 'along_y': deviation_y},
                        'flux_W_per_m2': {'along_x': flux_x, 'along_y': flux_y},
                        'convection_face_T_K': {'along_y': COLD_TEMPERATURE + flux_y * SLAB_LENGTH / K_ALONG},
                        'along_y_cells': list(values_y.shape), 'along_y_largest_dx_m': largest_dx,
                        'along_y_largest_dy_m': largest_dy, 'along_y_graded': graded}}


def _robin_face_power(boundary_temperatures, face_length, normal_conductivity):
    """Heat [W/m] leaving through Robin faces of one length: (T - T_ext) L / (1 / h + L / (2 k)) summed."""
    series_resistance = 1.0 / POINT_H + face_length / (2.0 * normal_conductivity)
    return float(np.sum(boundary_temperatures - COLD_TEMPERATURE)) * face_length / series_resistance


def _check_point_source_convection(results_by_name):
    """Heat leaving through the three Robin faces against P + inward flux * BOX_SIZE, relative error."""
    values, x_centers, y_centers = _field_arrays(results_by_name[RUN_NAME])
    cell_x, cell_y = float(x_centers[1] - x_centers[0]), float(y_centers[1] - y_centers[0])
    outflow = {'x_hi': _robin_face_power(values[-1, :], cell_y, THERMAL_CONDUCTIVITY),
               'y_lo': _robin_face_power(values[:, 0], cell_x, POINT_K_YY),
               'y_hi': _robin_face_power(values[:, -1], cell_x, POINT_K_YY)}
    measured = sum(outflow.values())
    return {'measured': measured, 'expected': POINT_TOTAL_POWER,
            'error': abs(measured - POINT_TOTAL_POWER) / POINT_TOTAL_POWER, 'tolerance': EXACT_IDENTITY_TOLERANCE,
            'details': {'outflow_W_per_m': outflow, 'T_max_K': float(np.max(values)),
                        'T_min_K': float(np.min(values)), 'n_cells': list(values.shape)}}


SELFTEST_CASES = (
    SelfTestCase(
        name='heat_uniform_source_slab_mixed', label='Heat: uniformly heated slab, mixed-precision CG',
        description=(f'The heated_slab template ({SLAB_LENGTH * 1e3:g} mm slab, k = {THERMAL_CONDUCTIVITY:g} W/(m*K), '
                     f'uniform Q = {POWER_DENSITY:g} W/m^3, both x faces at {COLD_TEMPERATURE:g} K, an even count of '
                     f'{CELL_SIZE * 1e6:g} um cells) solved by a pcg solver node at precision mixed (float64 system, '
                     'float32 CG passes each reducing the float64 residual by up to 1e-4, then a float64 residual '
                     f"refresh) to tolerance {EXACT_CG_TOLERANCE:g}, below the float32 CG floor. Exact: derived "
                     f'T_max - T_wall = Q L^2 / (8 k) = {PEAK_RISE:g} K (the discrete offset Q h^2 / (8 k) cancels '
                     f'at the two centre cells); tolerance {HEATED_SLAB_TOLERANCE:g} (single precision CG would stall '
                     'near 1e-6 and miss it). The check raises unless summary solvers records the linear role at '
                     'precision mixed with at least one refinement.'),
        specs=_heated_slab_mixed_runs, check=_check_heated_slab_mixed),
    SelfTestCase(
        name='heat_cross_conduction_box_graded', label='Heat: cross conduction keeps a uniform box uniform (graded)',
        description=(f'Stationary conduction in a {BOX_SIZE * 1e3:g} mm square (k = {THERMAL_CONDUCTIVITY:g} W/(m*K), '
                     f'k_xy = {BACKGROUND_K_XY:g}, a centred {(CROSS_BLOCK[1] - CROSS_BLOCK[0]) * 1e3:g} mm block with '
                     f'k_xy = {BLOCK_K_XY:g}) with all four faces at {HOT_TEMPERATURE:g} K and T0 = {DEFAULT_T0:g} K, '
                     f'on mesh nonuniform: base {CELL_SIZE * 1e6:g} um, the block refined to '
                     f'{CROSS_GRADED_CELL * 1e6:g} um, so the cells grade around it (solver_plan S7.1: the vertex '
                     'rectangles take area-weighted k_xy; uniform and graded widths share that one assembly, and the '
                     f'block and far field are uniform zones). Exact: T = {HOT_TEMPERATURE:g} K everywhere; measured '
                     f'= max |T - {HOT_TEMPERATURE:g} K| / (T - T0), expected 0. The vertex-based k_xy stencil is '
                     'conservative (every row sums to 0 on any widths), so only the CG stop and float32 storage of T '
                     'remain; a cross stencil without its diagonal exchanges heat with T0 at the window and block '
                     f'corners (errors ~1e-2 and more); tolerance {CROSS_BOX_TOLERANCE:g}; error inf unless the x '
                     'centre spacings vary by more than 1.5x.'),
        specs=_cross_conduction_box_runs, check=_check_cross_conduction_box),
    SelfTestCase(
        name='heat_cross_conduction_graded_convergence',
        label='Heat: cross conduction on graded grids converges at second order',
        description=('CPU check of core.fv_diffusion.build_anisotropic_laplacian (float64, scipy spsolve): -div(K '
                     f'grad u) = f on a {CROSS_CONVERGENCE_SIZE * 1e3:g} mm square, K = (k_xx, k_yy, k_xy) = '
                     f'{CROSS_CONVERGENCE_K} W/(m*K), u = sin(pi x / L) sin(pi y / L) with Dirichlet u = 0 on all '
                     'four faces (so the omitted cross flux through outer faces is exactly 0), f = (pi / L)^2 ((k_xx + '
                     'k_yy) u - 2 k_xy cos cos) sampled at the centres times the cell area, on smoothly graded grids '
                     f'of {", ".join(str(n) for n in CROSS_CONVERGENCE_CELLS)} cells per axis (x widths 1 + '
                     f'{CROSS_X_WAVE:g} cos(2 pi s): ratio 3; y widths ~ e^({CROSS_Y_STRETCH:g} s): ratio ~4.5). The '
                     'vertex energy 2 c_q (mean x difference)(mean y difference) is exact for linear u over the '
                     'rectangle between the four centres on any widths, so the cell-centred scheme should stay '
                     'supraconvergent: measured = observed order of the relative max error between the two finest '
                     f'grids, expected 2, error = max(0, 2 - order), tolerance {CROSS_ORDER_TOLERANCE:g} (a cross term '
                     'with the wrong weight or sign converges to another solution: order -> 0). Not yet confirmed by '
                     'a run.'),
        specs=lambda: {}, check=_check_cross_conduction_convergence, max_wall_time=30.0),
    SelfTestCase(
        name='heat_cross_flux_dissipation',
        label='Heat: k_xy cross flux and dissipation of the post-processing',
        description=('CPU check (cpu backend, no model run) of DiffusionSystem.flux_at_centres / '
                     'cross_flux_at_centres / dissipation_density with a cross term (shared by heat k_xy and the '
                     f'potential sigma_dc_xy / eps_dc_xy): the {CROSS_FLUX_CELLS}-cell graded grid of '
                     f'heat_cross_conduction_graded_convergence, K = {CROSS_CONVERGENCE_K} W/(m*K), T = 0 on every '
                     f'face (T0 = 0), uniform Q = {CROSS_FLUX_SOURCE:g} W/m^3, float64 exact CG. (1) The dissipation '
                     'integral must equal the source work sum T Q V (= the operator energy, its cross part included). '
                     f'(2) T then overwritten by the linear {CROSS_FLUX_GRADIENT} K/m field: the cross flux must be '
                     '-k_xy (dT/dy, dT/dx) in every cell and the total flux -K grad T in the cells off the outer '
                     'faces. error = the worst relative error (flux over max |-K grad T|), tolerance '
                     f'{CROSS_FLUX_TOLERANCE:g}; post-processing without the cross part misses (2) by ~55 % and (1) '
                     'by the cross share of the energy. Not yet confirmed by a run.'),
        specs=lambda: {}, check=_check_cross_flux_dissipation, max_wall_time=30.0),
    SelfTestCase(
        name='heat_convection_strips', label='Heat: Robin strips along x and y (graded mesh)',
        description=(f'Two stationary strips of {SLAB_LENGTH * 1e3:g} x {SLAB_HEIGHT * 1e3:g} mm with k = '
                     f'{K_ALONG:g} W/(m*K) along and {K_ACROSS:g} across them (insulated sides), held at '
                     f'{COLD_TEMPERATURE:g} K at one end and cooled by convection h = {ROBIN_H:g} W/(m^2*K), T_ext = '
                     f'{HOT_TEMPERATURE:g} K at the other. Run along_x: uniform {CELL_SIZE * 1e6:g} um cells, a '
                     f'fixed_temperature_domain block on the first {RESERVOIR_LENGTH * 1e3:g} mm; run along_y: the '
                     'strip rotated (temperature on y_lo, convection on y_hi, k_xx and k_yy swapped) on a nonuniform '
                     f'mesh refined to {GRADED_FINE_CELL * 1e6:g} um in a full-width band y = {GRADED_BAND[0] * 1e3:g}'
                     f'..{GRADED_BAND[1] * 1e3:g} mm, so dx = {GRADED_FINE_CELL * 1e6:g} um everywhere while dy grades '
                     f'up to {CELL_SIZE * 1e6:g} um. Exact: flux q = (T_ext - T_held) / (L_free / k_along + 1 / h), T '
                     '= T_held + q s / k_along at distance s from the held face (held cells at T_held). The '
                     'finite-volume scheme (face conductance 2 k / (w_l + w_r), held and Robin faces with the half '
                     'cell 2 k / w in series) is exact for this linear profile on any grading; measured = max |T - '
                     f'T_exact| over both runs / {TEMPERATURE_SPAN:g} K, expected 0, tolerance '
                     f'{EXACT_IDENTITY_TOLERANCE:g} (float64 CG to 1e-12, float32 T). This pins the Robin series '
                     'term, the held-domain faces, k_yy and the y-face conductances on cells with dx != dy (a k_xx / '
                     'k_yy or dx / dy swap fails); error inf when along_y gets no cells with dx != dy. Not yet run '
                     f'at the {GRADED_FINE_CELL * 1e6:g} um band.'),
        specs=_convection_strip_runs, check=_check_convection_strips),
)
