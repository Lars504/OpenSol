"""Exact checks of the coordinate metric (symmetry plan 2.16, core.metric) and of heat on polar (r, phi) and spherical
(r, theta) grids through the DiffusionSystem (scripted HeatPhysics on padding-free polar Domains, incl. the phi
wrap); one CPU geometry case and three GPU cases, registered by definition.register."""
import math

import numpy as np

from ...core.constants import FACES
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import max_relative_error

CONDUCTIVITY = 2.0                 # W/(m*K)
POWER_DENSITY = 1e10               # W/m^3 (exact in float32)
INNER_RISE = 10.0                  # K, Dirichlet rise of the inner face of the annulus / shell runs
CG_TOLERANCE = 1e-13               # float64 Jacobi-PCG, relative to ||b||
CG_MAX_ITERATIONS = 20000
# float32 source densities (Q_ext) carry 6e-8 relative rounding; the solves are float64
EXACT_TOLERANCE = 1e-6
GEOMETRY_TOLERANCE = 1e-12         # float64 sums of exact areas / volumes

OUTER_RADIUS = 1e-4                # m
SECTOR_ANGLE = math.pi / 3.0       # rad, polar runs
DISK_CELLS = 20                    # uniform dr: the parabola identity needs equal widths about every face
ANNULUS_INNER = 4e-5               # m, polar radius_min of the annulus run
SHELL_INNER = 3e-5                 # m, spherical radius_min of the shell run
ANGULAR_INNER = 2e-5               # m, radius_min of the pure-angle runs
ANGULAR_SOURCE = 64.0              # W/m: polar Q = q0 / r^2 (rise ~ q0 Phi^2 / (8 k))
THETA_SOURCE = 60.0                # W/m: spherical Q = q0 cos(theta) / s^2 (amplitude ~ q0 / (3 k))
THETA_LOSS = 1.0                   # beta of the loss h = beta k / s^2
WRAP_CELLS = 24                    # uniform angle cells of the phi-wrapped runs
WRAP_SOURCE = 64.0                 # W/m: Q = q0 / r_c^2, so every ring is one angular chain with S / g = q0 dphi^2 / k
WRAP_SECTORS = 4                   # antiperiodic run: one C_4 sector (span pi / 2)


def _geometric_widths(length, n_cells, ratio):
    """n_cells widths growing by ratio, summing to length."""
    widths = ratio ** np.arange(n_cells, dtype=np.float64)
    return length * widths / widths.sum()


def _heat_on_widths(dx_cells, dy_cells, coordinates, axisymmetric):
    """(stationary HeatPhysics with T0 = 0 and exact float64 CG, its padding-free Domain) of the given cell widths,
    k = CONDUCTIVITY everywhere."""
    from ...core.domain import Domain
    from ...core.material_grid import MaterialGrid
    from ...core.mesh import NonUniformGrid2D
    from .solver import HeatPhysics
    grid = NonUniformGrid2D(dx_cells, dy_cells, verbose=False)
    domain = Domain(grid, MaterialGrid(grid.nx, grid.ny), size_x=grid.size_x, size_y=grid.size_y, non_uniform=True,
                    npml_sides={face: 0 for face in FACES}, coordinates=coordinates)
    heat = HeatPhysics(domain, T0=0.0, tol=CG_TOLERANCE, max_iter=CG_MAX_ITERATIONS, axisymmetric=axisymmetric)
    for name in ('k_xx', 'k_yy'):
        domain.materials.assign(name, None, CONDUCTIVITY)
    heat.materials_changed(('k_xx', 'k_yy'))
    return heat, domain


def _solved_rise(heat):
    """Steady solve; float64 host rise (window = grid: no padding)."""
    import cupy as cp
    heat.step(0.0, math.inf)
    return cp.asnumpy(heat.system.unknown2d).astype(np.float64)


def _set_dirichlet(heat, faces_values):
    from ...core.fv_diffusion import BCSide
    for face, value in faces_values.items():
        heat.set_face_bc(face, BCSide('dirichlet', value=value))


def _full_grid(domain, values, dtype=np.float32):
    """Device full-grid array of host values broadcastable to (nx, ny) (the window: no padding)."""
    import cupy as cp
    return cp.asarray(np.ascontiguousarray(np.broadcast_to(values, (domain.grid.nx, domain.grid.ny)), dtype=dtype))


def _everywhere(domain):
    import cupy as cp
    return cp.ones((domain.grid.nx, domain.grid.ny), dtype=cp.bool_)


def _axis_geometry(domain, axis, origin=0.0):
    """(float64 widths, edges, centres) of one axis, edges / centres shifted by origin (radius_min)."""
    widths = np.asarray(domain.cell_widths()[axis], dtype=np.float64)
    edges = origin + np.asarray(domain.cell_edges()[axis], dtype=np.float64)
    return widths, edges, 0.5 * (edges[:-1] + edges[1:])


def _radial_chain(widths, faces, area_power, inner_value):
    """Exact discrete centre values and flow (per unit angular measure) of a source-free radial chain between
    Dirichlet faces faces[0] (inner_value) and faces[-1] (0): half cells 2 k A / w, interior 2 k A / (w_l + w_r),
    A = r_face ** area_power."""
    areas = faces ** area_power
    resistances = np.concatenate(([widths[0] / (2.0 * CONDUCTIVITY * areas[0])],
                                  (widths[:-1] + widths[1:]) / (2.0 * CONDUCTIVITY * areas[1:-1]),
                                  [widths[-1] / (2.0 * CONDUCTIVITY * areas[-1])]))
    flow = inner_value / float(np.sum(resistances))
    return inner_value - flow * np.cumsum(resistances)[:-1], flow


# ---- metric_coordinate_geometry (CPU) ----

def _check_geometry(_results):
    """Exact areas / volumes of the four 2D metrics on graded widths, against closed forms written independently."""
    from ...core.metric import Coordinates, window_metric
    r_min, length = 3e-5, 7e-5
    dr = _geometric_widths(length, 9, 1.17)
    r_faces = r_min + np.concatenate(([0.0], np.cumsum(dr)))
    theta_min, theta_span = 0.2, 2.5
    d_angle = _geometric_widths(theta_span, 7, 0.8)
    angle_faces = theta_min + np.concatenate(([0.0], np.cumsum(d_angle)))
    height = 4e-5
    dz = _geometric_widths(height, 5, 1.3)
    origin_faces = np.concatenate(([0.0], np.cumsum(dr)))   # (r, z) and the r_min = 0 polar case: r from 0
    errors = {}

    def relative(measured, expected):
        return float(np.max(np.abs(np.asarray(measured) - expected))) / float(np.max(np.abs(expected)))

    planar = window_metric(dr, dz)
    errors['xy_volume'] = relative(planar.volume.full().sum(), length * height)
    errors['xy_faces'] = relative(planar.x_face_area.full(), np.broadcast_to(dz, (dr.size + 1, dz.size)))

    cylinder = window_metric(dr, dz, axisymmetric=True)
    errors['rz_volume'] = relative(cylinder.volume.full().sum(), math.pi * origin_faces[-1] ** 2 * height)
    # div(r e_r) = 2 per ring: r_hi A_hi - r_lo A_lo = 2 V on every cell
    flux = origin_faces[:, None] * cylinder.x_face_area.full()
    errors['rz_divergence'] = relative(flux[1:] - flux[:-1], 2.0 * cylinder.volume.full())
    errors['rz_axis_face'] = float(np.max(np.abs(cylinder.x_face_area.full()[0])))

    sector = window_metric(dr, d_angle, Coordinates('polar', radius_min=r_min, angle_min=theta_min))
    errors['polar_volume'] = relative(sector.volume.full().sum(), 0.5 * (r_faces[-1] ** 2 - r_min ** 2) * theta_span)
    flux = r_faces[:, None] * sector.x_face_area.full()
    errors['polar_divergence'] = relative(flux[1:] - flux[:-1], 2.0 * sector.volume.full())
    errors['polar_angle_faces'] = relative(sector.y_face_area.full()[:, 3].sum(), length)
    errors['polar_arc'] = relative(sector.x_face_area.full()[-1].sum(), r_faces[-1] * theta_span)
    disk = window_metric(dr, d_angle, Coordinates('polar'))
    errors['polar_origin_face'] = float(np.max(np.abs(disk.x_face_area.full()[0])))

    sphere = window_metric(dr, d_angle, Coordinates('polar', radius_min=r_min, angle_min=theta_min), axisymmetric=True)
    cosine_span = math.cos(angle_faces[0]) - math.cos(angle_faces[-1])
    errors['spherical_volume'] = relative(sphere.volume.full().sum(),
                                          2.0 * math.pi / 3.0 * (r_faces[-1] ** 3 - r_min ** 3) * cosine_span)
    flux = r_faces[:, None] * sphere.x_face_area.full()   # div(r e_r) = 3
    errors['spherical_divergence'] = relative(flux[1:] - flux[:-1], 3.0 * sphere.volume.full())
    errors['spherical_sphere_face'] = relative(sphere.x_face_area.full()[-1].sum(),
                                               2.0 * math.pi * r_faces[-1] ** 2 * cosine_span)
    errors['spherical_cone_faces'] = relative(sphere.y_face_area.full()[:, 4].sum(),
                                              math.pi * (r_faces[-1] ** 2 - r_min ** 2) * math.sin(angle_faces[4]))
    full_meridian = window_metric(dr, np.full(6, math.pi / 6.0), Coordinates('polar'), axisymmetric=True)
    errors['spherical_axis_faces'] = float(np.max(np.abs(full_meridian.y_face_area.full()[:, [0, -1]])))
    errors['spherical_full_band'] = relative(full_meridian.volume.along_y.sum(), 2.0)

    refused = 0
    for make in (lambda: Coordinates('polar', radius_min=-1e-6), lambda: Coordinates('cartesian', radius_min=1e-6),
                 lambda: Coordinates('cylindrical'),
                 lambda: window_metric(dr, np.full(8, math.pi / 6.0), Coordinates('polar'), axisymmetric=True),
                 lambda: window_metric(dr, np.full(8, math.pi), Coordinates('polar'))):
        try:
            make()
        except ValueError:
            refused += 1
    errors['refusals'] = float(5 - refused)
    error = float(np.max(list(errors.values())))
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': GEOMETRY_TOLERANCE,
            'details': {'errors': errors}}


# ---- heat_polar_metric (GPU) ----

def _polar_disk():
    """Uniformly heated sector r <= R, T(R) = 0: T = Q (R^2 - r^2) / (4 k) + Q dr^2 / (16 k) exactly."""
    from ...core.metric import Coordinates
    heat, domain = _heat_on_widths(np.full(DISK_CELLS, OUTER_RADIUS / DISK_CELLS),
                                   _geometric_widths(SECTOR_ANGLE, 12, 1.15), Coordinates('polar'), False)
    _set_dirichlet(heat, {'x_hi': 0.0})
    heat.add_heat_source(_everywhere(domain), POWER_DENSITY)
    rise = _solved_rise(heat)
    widths, edges, centres = _axis_geometry(domain, 0)
    expected = POWER_DENSITY * ((edges[-1] ** 2 - centres ** 2) / (4.0 * CONDUCTIVITY)
                                + widths[0] ** 2 / (16.0 * CONDUCTIVITY))
    return max_relative_error(rise, expected[:, None], float(expected[0]))


def _annulus(coordinates, axisymmetric, n_angle, area_power, angular_measure):
    """Source-free annulus / shell between Dirichlet r faces (graded dr): exact discrete chain, and the inward flow
    through x_lo = chain flow x angular measure; (rise error, flow error)."""
    heat, domain = _heat_on_widths(_geometric_widths(OUTER_RADIUS - coordinates.radius_min, 14, 1.12),
                                   np.full(n_angle, (SECTOR_ANGLE if not axisymmetric else math.pi) / n_angle),
                                   coordinates, axisymmetric)
    _set_dirichlet(heat, {'x_lo': INNER_RISE, 'x_hi': 0.0})
    rise = _solved_rise(heat)
    widths, edges, _ = _axis_geometry(domain, 0, coordinates.radius_min)
    expected, flow = _radial_chain(widths, edges, area_power, INNER_RISE)
    expected_inflow = flow * angular_measure(domain)
    inflow_error = abs(heat.system.face_inward_flux('x_lo') - expected_inflow) / abs(expected_inflow)
    return max_relative_error(rise, expected[:, None], INNER_RISE), inflow_error


def _polar_angular():
    """Q = q0 / r_c^2 with T = 0 on both phi faces, insulated r faces: every ring solves -k T'' = q0 alike, so exactly
    T = q0 phi (Phi - phi) / (2 k) + q0 dphi^2 / (8 k) and no radial flow (pins the phi-face weights to the volumes)."""
    from ...core.metric import Coordinates
    coordinates = Coordinates('polar', radius_min=ANGULAR_INNER, angle_min=0.25)
    heat, domain = _heat_on_widths(_geometric_widths(OUTER_RADIUS - ANGULAR_INNER, 10, 1.2),
                                   np.full(16, 0.5 * math.pi / 16), coordinates, False)
    _set_dirichlet(heat, {'y_lo': 0.0, 'y_hi': 0.0})
    _, _, r_centres = _axis_geometry(domain, 0, ANGULAR_INNER)
    heat.add_heat_source(_everywhere(domain), _full_grid(domain, (ANGULAR_SOURCE / r_centres ** 2)[:, None]))
    rise = _solved_rise(heat)
    angle_widths, angle_edges, angle_centres = _axis_geometry(domain, 1)
    strength = ANGULAR_SOURCE / CONDUCTIVITY
    expected = strength * (0.5 * angle_centres * (angle_edges[-1] - angle_centres) + angle_widths[0] ** 2 / 8.0)
    return max_relative_error(rise, expected[None, :], float(np.max(expected)))


def _check_polar(_results):
    from ...core.metric import Coordinates
    errors = {'disk_parabola': _polar_disk()}
    errors['annulus_rise'], errors['annulus_inflow'] = _annulus(
        Coordinates('polar', radius_min=ANNULUS_INNER), False, 6, 1.0,
        lambda domain: float(np.sum(domain.cell_widths()[1])))
    errors['angular_parabola'] = _polar_angular()
    error = float(np.max(list(errors.values())))
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': EXACT_TOLERANCE,
            'details': {'errors': errors}}


# ---- heat_spherical_metric (GPU) ----

def _spherical_ball():
    """Uniformly heated ball (theta 0..pi graded), T(R) = 0: T = Q (R^2 - r^2) / (6 k) + Q dr^2 / (24 k) exactly."""
    from ...core.metric import Coordinates
    heat, domain = _heat_on_widths(np.full(DISK_CELLS, OUTER_RADIUS / DISK_CELLS),
                                   _geometric_widths(math.pi, 12, 1.2), Coordinates('polar'), True)
    _set_dirichlet(heat, {'x_hi': 0.0})
    heat.add_heat_source(_everywhere(domain), POWER_DENSITY)
    rise = _solved_rise(heat)
    widths, edges, centres = _axis_geometry(domain, 0)
    expected = POWER_DENSITY * ((edges[-1] ** 2 - centres ** 2) / (6.0 * CONDUCTIVITY)
                                + widths[0] ** 2 / (24.0 * CONDUCTIVITY))
    return max_relative_error(rise, expected[:, None], float(expected[0]))


def _sphere_band_measure(domain):
    """2 pi (cos theta_min - cos theta_max) of a theta range starting at 0 (whole-sphere solid angle 4 pi)."""
    return 2.0 * math.pi * (1.0 - math.cos(float(np.sum(domain.cell_widths()[1], dtype=np.float64))))


def _theta_eigenmode():
    """Q = q0 cos(theta) / s^2 and loss h = beta k / s^2 (s^2 = (r_hi^3 - r_lo^3) / (3 dr)), every face insulated:
    the cell-centred cos(theta) is an exact eigenvector of the uniform-dtheta operator with eigenvalue lambda = 2
    sin(dtheta) / dtheta, so T = q0 cos(theta) / (k (lambda + beta)) exactly (pins the theta faces sin(theta_f) and
    the volumes)."""
    from ...core.metric import Coordinates
    n_theta = 24
    heat, domain = _heat_on_widths(_geometric_widths(OUTER_RADIUS - ANGULAR_INNER, 8, 1.25),
                                   np.full(n_theta, math.pi / n_theta), Coordinates('polar', radius_min=ANGULAR_INNER),
                                   True)
    _, r_edges, _ = _axis_geometry(domain, 0, ANGULAR_INNER)
    shell_square = (r_edges[:-1] ** 2 + r_edges[:-1] * r_edges[1:] + r_edges[1:] ** 2) / 3.0
    angle_widths, _, angle_centres = _axis_geometry(domain, 1)
    everywhere = _everywhere(domain)
    heat.add_heat_source(everywhere, _full_grid(domain, THETA_SOURCE * np.cos(angle_centres)[None, :]
                                                / shell_square[:, None]))
    heat.add_volumetric_loss(everywhere, _full_grid(domain, (THETA_LOSS * CONDUCTIVITY / shell_square)[:, None],
                                                    np.float64), 0.0)
    rise = _solved_rise(heat)
    d_theta = float(angle_widths[0])
    amplitude = THETA_SOURCE / (CONDUCTIVITY * (2.0 * math.sin(d_theta) / d_theta + THETA_LOSS))
    return max_relative_error(rise, amplitude * np.cos(angle_centres)[None, :], amplitude)


def _check_spherical(_results):
    from ...core.metric import Coordinates
    errors = {'ball_parabola': _spherical_ball()}
    errors['shell_rise'], errors['shell_inflow'] = _annulus(
        Coordinates('polar', radius_min=SHELL_INNER), True, 8, 2.0, _sphere_band_measure)
    errors['theta_eigenmode'] = _theta_eigenmode()
    error = float(np.max(list(errors.values())))
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': EXACT_TOLERANCE,
            'details': {'errors': errors}}


# ---- heat_polar_wrap (GPU) ----

def _wrapped_heat(angle_wrap, span, n_angle=WRAP_CELLS):
    """(heat, domain, r centres) on a phi-wrapped polar window: radius_min ANGULAR_INNER, graded dr, n_angle uniform
    angle cells over span, insulated r faces."""
    from ...core.metric import Coordinates
    coordinates = Coordinates('polar', radius_min=ANGULAR_INNER, angle_min=0.25, angle_wrap=angle_wrap)
    heat, domain = _heat_on_widths(_geometric_widths(OUTER_RADIUS - ANGULAR_INNER, 10, 1.2),
                                   np.full(n_angle, span / n_angle), coordinates, False)
    _, _, r_centres = _axis_geometry(domain, 0, ANGULAR_INNER)
    return heat, domain, r_centres


def _ring_source(heat, domain, r_centres, columns):
    """Q = WRAP_SOURCE / r_c^2 on the angle columns (bool (ny,)); float64 host W/m per cell (Q x metric volume)."""
    density = (WRAP_SOURCE / r_centres ** 2)[:, None] * np.asarray(columns, dtype=np.float64)[None, :]
    heat.add_heat_source(_everywhere(domain), _full_grid(domain, density))
    return density * heat.system.metric.volume.full()


def _wrap_flux_errors(system, expected_flux_y, prefix):
    """{prefix_flux: angular flux at the centres vs expected, prefix_radial_flux: max |radial flux| (0 expected)}."""
    import cupy as cp
    flux_x, flux_y = (cp.asnumpy(component).astype(np.float64) for component in system.flux_at_centres())
    scale = float(np.max(np.abs(expected_flux_y)))
    return {f'{prefix}_flux': max_relative_error(flux_y, expected_flux_y, scale),
            f'{prefix}_radial_flux': float(np.max(np.abs(flux_x))) / scale}


def _dissipation_error(system, rise, cell_power):
    """|sum of dissipation_density x cell area - sum T S| / sum T S (every face, the wrap face included)."""
    import cupy as cp
    dissipated = float(cp.asnumpy(system.dissipation_density() * system.cell_area).sum())
    work = float(np.sum(rise * cell_power))
    return abs(dissipated - work) / abs(work)


def _periodic_ring():
    """Full 2 pi ring, periodic wrap, Q = q0 / r_c^2 everywhere, T = 0 held on the j = 0 cells (ideal: T = 0 on their
    faces): the free chain closes through the wrap face, T_j = q0 dphi^2 (j (N - j) - (N - 1) / 2) / (2 k), angular
    flux q0 dphi (2 j - N) / (2 r_c), no radial flow, held outflow -sum Q V, dissipation = sum T Q V."""
    import cupy as cp
    heat, domain, r_centres = _wrapped_heat('periodic', 2.0 * math.pi)
    n_angle = domain.grid.ny
    cell_power = _ring_source(heat, domain, r_centres, np.ones(n_angle, dtype=bool))
    held = np.zeros((domain.grid.nx, n_angle), dtype=bool)
    held[:, 0] = True
    heat.add_fixed_temperature(cp.asarray(held), 0.0)
    rise = _solved_rise(heat)
    d_angle = 2.0 * math.pi / n_angle
    index = np.arange(n_angle, dtype=np.float64)
    expected = np.where(index > 0, WRAP_SOURCE * d_angle ** 2 / (2.0 * CONDUCTIVITY)
                        * (index * (n_angle - index) - 0.5 * (n_angle - 1)), 0.0)
    errors = {'periodic_rise': max_relative_error(rise, expected[None, :], float(np.max(expected)))}
    system = heat.system
    total_power = float(cell_power.sum())
    errors['periodic_held_outflow'] = abs(system.fixed_region_outflow(cp.asarray(held)) + total_power) / total_power
    expected_flux = np.where(held, 0.0, WRAP_SOURCE * d_angle * (2.0 * index[None, :] - n_angle)
                             / (2.0 * r_centres[:, None]))
    errors.update(_wrap_flux_errors(system, expected_flux, 'periodic'))
    errors['periodic_dissipation'] = _dissipation_error(system, rise, cell_power)
    return errors


def _antiperiodic_sector():
    """One C_4 sector, antiperiodic wrap, Q = q0 / r_c^2 on the j = 0 cells only, no held cell (the sign-flipping loop
    anchors): T_j = q0 dphi^2 (N - 2 j) / (4 k), angular flux q0 dphi / (2 r_c) off the source column (0 on it), the
    outflow of the source column 0 (pins the +2 g wrap row sums), dissipation = sum T Q V."""
    import cupy as cp
    heat, domain, r_centres = _wrapped_heat('antiperiodic', 2.0 * math.pi / WRAP_SECTORS)
    n_angle = domain.grid.ny
    column = np.arange(n_angle) == 0
    cell_power = _ring_source(heat, domain, r_centres, column)
    rise = _solved_rise(heat)
    d_angle = 2.0 * math.pi / WRAP_SECTORS / n_angle
    index = np.arange(n_angle, dtype=np.float64)
    expected = WRAP_SOURCE * d_angle ** 2 / (4.0 * CONDUCTIVITY) * (n_angle - 2.0 * index)
    errors = {'antiperiodic_rise': max_relative_error(rise, expected[None, :], float(np.max(np.abs(expected))))}
    system = heat.system
    total_power = float(cell_power.sum())
    column_mask = cp.asarray(np.broadcast_to(column[None, :], rise.shape).copy())
    errors['antiperiodic_outflow'] = abs(system.fixed_region_outflow(column_mask)) / total_power
    expected_flux = np.where(column[None, :], 0.0, WRAP_SOURCE * d_angle / (2.0 * r_centres[:, None]))
    errors.update(_wrap_flux_errors(system, expected_flux, 'antiperiodic'))
    errors['antiperiodic_dissipation'] = _dissipation_error(system, rise, cell_power)
    return errors


def _wrap_refusals():
    """Number of the expected ValueErrors that did not occur: a periodic wrap without anchor (constant null mode), a
    boundary condition on a wrapped face, 2 angle cells, a wrap on a Cartesian / spherical grid, an unknown wrap."""
    from ...core.metric import Coordinates, window_metric

    def unanchored_periodic():
        heat, domain, r_centres = _wrapped_heat('periodic', 2.0 * math.pi / WRAP_SECTORS)
        _ring_source(heat, domain, r_centres, np.arange(domain.grid.ny) == 0)
        _solved_rise(heat)

    def wrapped_face_condition():
        heat, _, _ = _wrapped_heat('periodic', 2.0 * math.pi)
        _set_dirichlet(heat, {'x_hi': 0.0, 'y_lo': 0.0})
        _solved_rise(heat)

    def two_angle_cells():
        heat, _, _ = _wrapped_heat('periodic', 2.0 * math.pi, n_angle=2)
        _set_dirichlet(heat, {'x_hi': 0.0})
        _solved_rise(heat)

    makers = (unanchored_periodic, wrapped_face_condition, two_angle_cells,
              lambda: Coordinates('cartesian', angle_wrap='periodic'),
              lambda: Coordinates('polar', angle_wrap='bloch'),
              lambda: window_metric(np.full(4, 1e-5), np.full(6, math.pi / 6.0),
                                    Coordinates('polar', angle_wrap='periodic'), axisymmetric=True))
    refused = 0
    for make in makers:
        try:
            make()
        except ValueError:
            refused += 1
    return float(len(makers) - refused)


def _check_wrap(_results):
    errors = _periodic_ring()
    errors.update(_antiperiodic_sector())
    errors['refusals'] = _wrap_refusals()
    error = float(np.max(list(errors.values())))
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': EXACT_TOLERANCE,
            'details': {'errors': errors}}


COORDINATE_SELFTEST_CASES = (
    SelfTestCase(
        name='metric_coordinate_geometry', label='Metric: exact areas and volumes of the 2D coordinate grids',
        description=('CPU check of core.metric.window_metric on graded widths: (x, y), (r, z), polar (r, phi) and '
                     'spherical (r, theta) cell volumes summed against the closed-form window volumes, '
                     'r_hi A_hi - r_lo A_lo = 2 V ((r, z), (r, phi)) / 3 V ((r, theta)) on every cell (div of r e_r), '
                     'outer arc / sphere / cone face sums, zero area on the axis faces (r = 0, theta = 0 / pi), and the '
                     'refused inputs (negative radius_min, an offset on a Cartesian grid, an unknown grid, theta beyond '
                     f'pi, phi beyond 2 pi). error = largest relative deviation + refusals missed, tolerance '
                     f'{GEOMETRY_TOLERANCE:g}.'),
        specs=lambda: {}, check=_check_geometry, max_wall_time=30.0),
    SelfTestCase(
        name='heat_polar_metric', label='Heat: polar (r, phi) grid, exact discrete solutions',
        description=('Scripted stationary HeatPhysics on padding-free polar Domains (k = '
                     f'{CONDUCTIVITY:g} W/(m*K), float64 CG to {CG_TOLERANCE:g}). (1) Uniform Q = {POWER_DENSITY:g} '
                     f'W/m^3 on the sector r <= {OUTER_RADIUS * 1e6:g} um ({DISK_CELLS} uniform dr, graded phi), T(R) '
                     '= 0: the face areas r_f dphi and volumes r_c dr dphi reproduce T = Q (R^2 - r^2) / (4 k) + Q '
                     'dr^2 / (16 k) in every cell. (2) Source-free annulus (radius_min '
                     f'{ANNULUS_INNER * 1e6:g} um, graded dr), rise {INNER_RISE:g} K on x_lo, 0 on x_hi: the exact '
                     'series chain of half-cell conductances 2 k r_f / w and its inward flow. (3) Q = q0 / r_c^2 with '
                     'T = 0 on both phi faces: T = q0 phi (Phi - phi) / (2 k) + q0 dphi^2 / (8 k) on every ring (no '
                     'radial flow; pins the phi-face conductances dr / (r_c dphi) against the volumes). error = '
                     f'largest relative deviation, tolerance {EXACT_TOLERANCE:g} (float32 source densities).'),
        specs=lambda: {}, check=_check_polar, max_wall_time=60.0, backends=('gpu',)),
    SelfTestCase(
        name='heat_spherical_metric', label='Heat: spherical (r, theta) grid, exact discrete solutions',
        description=('Scripted stationary axisymmetric HeatPhysics on padding-free polar Domains (spherical (r, '
                     f'theta), k = {CONDUCTIVITY:g} W/(m*K)). (1) Uniformly heated ball r <= '
                     f'{OUTER_RADIUS * 1e6:g} um (uniform dr, graded theta 0..pi), T(R) = 0: T = Q (R^2 - r^2) / (6 '
                     'k) + Q dr^2 / (24 k) exactly (r faces 2 pi r_f^2 (cos theta_lo - cos theta_hi), exact shell '
                     f'volumes). (2) Source-free shell (radius_min {SHELL_INNER * 1e6:g} um): the exact series chain '
                     'of conductances 2 k r_f^2 / w and its inward flow over the solid angle 4 pi. (3) Insulated '
                     'shell with Q = q0 cos(theta) / s^2 and loss beta k / s^2 (s^2 = the shell mean square radius): '
                     'cos(theta_c) is an exact eigenvector of the uniform-dtheta operator (theta faces 2 pi r_c dr '
                     'sin(theta_f), zero on the axis), T = q0 cos(theta) / (k (2 sin(dtheta) / dtheta + beta)). '
                     f'error = largest relative deviation, tolerance {EXACT_TOLERANCE:g}.'),
        specs=lambda: {}, check=_check_spherical, max_wall_time=60.0, backends=('gpu',)),
    SelfTestCase(
        name='heat_polar_wrap', label='Heat: phi wrap of polar grids (periodic / antiperiodic), exact chains',
        description=('Scripted stationary HeatPhysics on phi-wrapped polar Domains (Coordinates angle_wrap; radius_min '
                     f'{ANGULAR_INNER * 1e6:g} um, graded dr, {WRAP_CELLS} uniform angle cells, insulated r faces, '
                     f'Q = q0 / r_c^2 with q0 = {WRAP_SOURCE:g} W/m: every ring solves the same angular chain, no radial '
                     'flow). (1) Full ring, periodic, Q everywhere, T = 0 held on the j = 0 cells: T_j = q0 dphi^2 '
                     '(j (N - j) - (N - 1) / 2) / (2 k) (the last cell reaches the held cells only through the wrap '
                     'face), angular flux q0 dphi (2 j - N) / (2 r_c), held outflow = -sum Q V, sum of '
                     'dissipation_density x area = sum T Q V. (2) One C_4 sector, antiperiodic, Q on the j = 0 '
                     'cells only and no held cell (the sign-flipping loop is the only anchor): T_j = q0 dphi^2 (N - 2 '
                     'j) / (4 k), flux q0 dphi / (2 r_c) off the source column, the source column outflow 0 (wrap row '
                     'sums), dissipation. (3) Refusals: periodic without anchor, a Dirichlet on a wrapped face, 2 angle '
                     'cells, a wrap on Cartesian / spherical grids, an unknown wrap. error = largest relative '
                     f'deviation + refusals missed, tolerance {EXACT_TOLERANCE:g}.'),
        specs=lambda: {}, check=_check_wrap, max_wall_time=60.0, backends=('gpu',)),
)
