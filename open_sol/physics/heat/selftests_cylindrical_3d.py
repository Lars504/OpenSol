"""Exact checks of 3D cylindrical (r, phi, z) sectors (symmetry plan 2.21: core.metric.window_metric_3d and the 3D
DiffusionSystem on it): metric identities and scripted HeatPhysics on padding-free polar Domain3D windows of the cpu
backend, incl. the phi wrap; the model path (DomainSpec polar + size_z: mesh, memory estimate, validation, mapped
rasters, build and Session on the cpu backend); registered by definition.register."""
import copy
import math

import numpy as np

from ...core.constants import FACES_3D
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_edges, max_relative_error, with_exact_cg
from ...model.spec import (AnalyticCurveSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                           MeshSpec, ModelSpec, PhysicsSpec, SolverDefaultsSpec, StudySpec)

CONDUCTIVITY = 2.0                 # W/(m*K)
POWER_DENSITY = 1e10               # W/m^3 (exact in float32)
INNER_RISE = 10.0                  # K, Dirichlet rise of the annulus' inner face
CG_TOLERANCE = 1e-13               # float64 Jacobi-PCG, relative to ||b||
CG_MAX_ITERATIONS = 20000
EXACT_TOLERANCE = 1e-6             # float32 source densities carry 6e-8 relative rounding; the solves are float64
GEOMETRY_TOLERANCE = 1e-12         # float64 sums of exact areas / volumes

OUTER_RADIUS = 1e-4                # m
HEIGHT = 4e-5                      # m, z extent
SECTOR_ANGLE = math.pi / 3.0       # rad
DISK_CELLS = 12                    # uniform dr: the parabola identity needs equal widths about every face
ANNULUS_INNER = 4e-5               # m, radius_min of the annulus run
ANGULAR_INNER = 2e-5               # m, radius_min of the angle / axial / wrap runs
ANGULAR_SOURCE = 64.0              # W/m: Q = q0 / r_c^2 (every ring one angular chain)
AXIAL_CELLS = 10                   # uniform dz of the axial run
WRAP_CELLS = 24                    # uniform angle cells of the wrapped runs
WRAP_SECTORS = 4                   # antiperiodic run: one C_4 sector


def _geometric_widths(length, n_cells, ratio):
    """n_cells widths growing by ratio, summing to length."""
    widths = ratio ** np.arange(n_cells, dtype=np.float64)
    return length * widths / widths.sum()


def _sector_heat(dr, d_angle, dz, radius_min, angle_wrap='none', angle_min=0.25):
    """(stationary HeatPhysics with T0 = 0 and exact float64 CG, its padding-free cpu Domain3D) on a cylindrical
    sector of the given widths, k = CONDUCTIVITY along r, phi and z."""
    from ...core.domain3d import Domain3D
    from ...core.metric import Coordinates
    from .solver import HeatPhysics
    coordinates = Coordinates('polar', radius_min=radius_min, angle_min=angle_min, angle_wrap=angle_wrap)
    domain = Domain3D.from_widths(dr, d_angle, dz, {face: 0 for face in FACES_3D}, size_x=float(np.sum(dr)),
                                  size_y=float(np.sum(d_angle)), size_z=float(np.sum(dz)), coordinates=coordinates,
                                  backend='cpu')
    heat = HeatPhysics(domain, T0=0.0, tol=CG_TOLERANCE, max_iter=CG_MAX_ITERATIONS)
    for name in ('k_xx', 'k_yy', 'k_zz'):
        domain.materials.assign(name, None, CONDUCTIVITY)
    heat.materials_changed(('k_xx', 'k_yy', 'k_zz'))
    return heat, domain


def _axis_geometry(domain, axis, origin=0.0):
    """(float64 widths, edges, centres) of one axis of the domain (float32 cells), shifted by origin (radius_min)."""
    widths = np.asarray(domain.cell_widths()[axis], dtype=np.float64)
    edges = origin + np.asarray(domain.cell_edges()[axis], dtype=np.float64)
    return widths, edges, 0.5 * (edges[:-1] + edges[1:])


def _full(domain, values, dtype=np.float32):
    """Full-grid host array of values broadcastable to the (nx, ny, nz) grid (the window: no padding)."""
    return np.ascontiguousarray(np.broadcast_to(values, domain.grid.shape), dtype=dtype)


def _everywhere(domain):
    return np.ones(domain.grid.shape, dtype=bool)


def _set_dirichlet(heat, faces_values):
    from ...core.fv_diffusion import BCSide
    for face, value in faces_values.items():
        heat.set_face_bc(face, BCSide('dirichlet', value=value))


def _solved_rise(heat):
    heat.step(0.0, math.inf)
    return np.asarray(heat.system.unknown2d, dtype=np.float64)


# ---- metric_cylindrical_geometry_3d (CPU) ----

def _check_geometry(_results):
    """Exact areas / volumes of window_metric_3d on graded widths against closed forms written independently."""
    from ...core.metric import Coordinates, window_metric_3d
    r_min, depth = 3e-5, 7e-5
    dr = _geometric_widths(depth, 9, 1.17)
    r_faces = r_min + np.concatenate(([0.0], np.cumsum(dr)))
    angle_span = 2.0
    d_angle = _geometric_widths(angle_span, 7, 0.8)
    dz = _geometric_widths(HEIGHT, 5, 1.3)
    errors = {}

    def relative(measured, expected):
        expected = np.asarray(expected, dtype=np.float64)
        return float(np.max(np.abs(np.asarray(measured) - expected))) / float(np.max(np.abs(expected)))

    box = window_metric_3d(dr, d_angle, dz)
    errors['xyz_volume'] = relative(box.volume.full(), dr[:, None, None] * d_angle[None, :, None] * dz[None, None, :])
    errors['xyz_x_faces'] = relative(box.face_area[0].full(), np.broadcast_to(d_angle[:, None] * dz[None, :],
                                                                              (dr.size + 1, d_angle.size, dz.size)))
    sector = window_metric_3d(dr, d_angle, dz, Coordinates('polar', radius_min=r_min))
    errors['sector_volume'] = relative(sector.volume.full().sum(), 0.5 * (r_faces[-1] ** 2 - r_min ** 2) * angle_span
                                       * HEIGHT)
    flux = r_faces[:, None, None] * sector.face_area[0].full()   # div(r e_r) = 2 per cell
    errors['sector_divergence'] = relative(flux[1:] - flux[:-1], 2.0 * sector.volume.full())
    errors['sector_arc'] = relative(sector.face_area[0].full()[-1].sum(), r_faces[-1] * angle_span * HEIGHT)
    errors['sector_phi_face'] = relative(sector.face_area[1].full()[:, 3, :].sum(), depth * HEIGHT)
    errors['sector_z_face'] = relative(sector.face_area[2].full()[:, :, 2].sum(),
                                       0.5 * (r_faces[-1] ** 2 - r_min ** 2) * angle_span)
    errors['sector_phi_length'] = relative(sector.length[1].full()[:, :, 0],
                                           0.5 * (r_faces[:-1] + r_faces[1:])[:, None] * d_angle[None, :])
    disk = window_metric_3d(dr, d_angle, dz, Coordinates('polar'))
    errors['axis_face'] = float(np.max(np.abs(disk.face_area[0].full()[0])))

    refused = 0
    for make in (lambda: window_metric_3d(dr, np.full(8, math.pi), dz, Coordinates('polar')),
                 lambda: window_metric_3d(dr, np.ones(1), dz, Coordinates('polar', line=True)),
                 lambda: window_metric_3d(dr, d_angle, np.zeros(3))):
        try:
            make()
        except ValueError:
            refused += 1
    errors['refusals'] = float(3 - refused)
    error = float(np.max(list(errors.values())))
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': GEOMETRY_TOLERANCE,
            'details': {'errors': errors}}


# ---- heat_cylindrical_sector_3d (CPU) ----

def _sector_disk():
    """Uniformly heated sector r <= R (uniform dr, graded phi and z), T(R) = 0: T = Q (R^2 - r^2) / (4 k) + Q dr^2 /
    (16 k) in every cell, and cell_volume = r_c dr dphi dz (pins the r faces against the volumes, the axis face)."""
    heat, domain = _sector_heat(np.full(DISK_CELLS, OUTER_RADIUS / DISK_CELLS),
                                _geometric_widths(SECTOR_ANGLE, 6, 1.15), _geometric_widths(HEIGHT, 4, 1.4), 0.0)
    _set_dirichlet(heat, {'x_hi': 0.0})
    heat.add_heat_source(_everywhere(domain), POWER_DENSITY)
    rise = _solved_rise(heat)
    widths, edges, centres = _axis_geometry(domain, 0)
    expected = POWER_DENSITY * ((edges[-1] ** 2 - centres ** 2) / (4.0 * CONDUCTIVITY)
                                + widths[0] ** 2 / (16.0 * CONDUCTIVITY))
    angle_widths = _axis_geometry(domain, 1)[0]
    z_widths = _axis_geometry(domain, 2)[0]
    cell = (DISK_CELLS // 2, 2, 1)
    volume = centres[cell[0]] * widths[cell[0]] * angle_widths[cell[1]] * z_widths[cell[2]]
    return {'disk_parabola': max_relative_error(rise, expected[:, None, None], float(expected[0])),
            'disk_cell_volume': abs(heat.system.cell_volume(*cell) / volume - 1.0)}


def _radial_chain(widths, faces, inner_value):
    """Exact discrete centre values and flow per radian and metre of height of a source-free radial chain between
    Dirichlet faces faces[0] (inner_value) and faces[-1] (0): half cells 2 k r_f / w, interior 2 k r_f / (w_l + w_r)."""
    resistances = np.concatenate(([widths[0] / (2.0 * CONDUCTIVITY * faces[0])],
                                  (widths[:-1] + widths[1:]) / (2.0 * CONDUCTIVITY * faces[1:-1]),
                                  [widths[-1] / (2.0 * CONDUCTIVITY * faces[-1])]))
    flow = inner_value / float(np.sum(resistances))
    return inner_value - flow * np.cumsum(resistances)[:-1], flow


def _sector_annulus():
    """Source-free annulus between Dirichlet r faces (graded dr, phi and z): the exact series chain in every cell and
    the inflow through x_lo = chain flow x phi span x height."""
    heat, domain = _sector_heat(_geometric_widths(OUTER_RADIUS - ANNULUS_INNER, 10, 1.12),
                                _geometric_widths(SECTOR_ANGLE, 5, 0.9), _geometric_widths(HEIGHT, 4, 1.3),
                                ANNULUS_INNER)
    _set_dirichlet(heat, {'x_lo': INNER_RISE, 'x_hi': 0.0})
    rise = _solved_rise(heat)
    widths, edges, _ = _axis_geometry(domain, 0, ANNULUS_INNER)
    expected, flow = _radial_chain(widths, edges, INNER_RISE)
    expected_inflow = flow * float(np.sum(domain.cell_widths()[1])) * float(np.sum(domain.cell_widths()[2]))
    return {'annulus_rise': max_relative_error(rise, expected[:, None, None], INNER_RISE),
            'annulus_inflow': abs(heat.system.face_inward_flux('x_lo') / expected_inflow - 1.0)}


def _sector_angular():
    """Q = q0 / r_c^2 with T = 0 on both phi faces, insulated r and z faces (graded dr and z): every ring solves the
    same chain, T = q0 phi (Phi - phi) / (2 k) + q0 dphi^2 / (8 k) (pins the phi faces dr dz / (r_c dphi))."""
    heat, domain = _sector_heat(_geometric_widths(OUTER_RADIUS - ANGULAR_INNER, 8, 1.2),
                                np.full(12, 0.5 * math.pi / 12), _geometric_widths(HEIGHT, 3, 1.5), ANGULAR_INNER)
    _set_dirichlet(heat, {'y_lo': 0.0, 'y_hi': 0.0})
    r_centres = _axis_geometry(domain, 0, ANGULAR_INNER)[2]
    heat.add_heat_source(_everywhere(domain), _full(domain, (ANGULAR_SOURCE / r_centres ** 2)[:, None, None]))
    rise = _solved_rise(heat)
    angle_widths, angle_edges, angle_centres = _axis_geometry(domain, 1)
    strength = ANGULAR_SOURCE / CONDUCTIVITY
    expected = strength * (0.5 * angle_centres * (angle_edges[-1] - angle_centres) + angle_widths[0] ** 2 / 8.0)
    return {'angular_parabola': max_relative_error(rise, expected[None, :, None], float(np.max(expected)))}


def _sector_axial():
    """Uniform Q with T = 0 on z_lo / z_hi (uniform dz; graded dr and phi): T = Q z (H - z) / (2 k) + Q dz^2 / (8 k)
    in every cell (pins the z faces r_c dr dphi against the volumes)."""
    heat, domain = _sector_heat(_geometric_widths(OUTER_RADIUS - ANGULAR_INNER, 8, 1.2),
                                _geometric_widths(SECTOR_ANGLE, 5, 1.2), np.full(AXIAL_CELLS, HEIGHT / AXIAL_CELLS),
                                ANGULAR_INNER)
    _set_dirichlet(heat, {'z_lo': 0.0, 'z_hi': 0.0})
    heat.add_heat_source(_everywhere(domain), POWER_DENSITY)
    rise = _solved_rise(heat)
    z_widths, z_edges, z_centres = _axis_geometry(domain, 2)
    expected = POWER_DENSITY / CONDUCTIVITY * (0.5 * z_centres * (z_edges[-1] - z_centres) + z_widths[0] ** 2 / 8.0)
    return {'axial_parabola': max_relative_error(rise, expected[None, None, :], float(np.max(expected)))}


def _wrapped_heat(angle_wrap, span, n_angle=WRAP_CELLS):
    """(heat, domain, r centres) of a phi-wrapped sector: radius_min ANGULAR_INNER, graded dr and z, n_angle uniform
    angle cells over span, insulated r and z faces."""
    heat, domain = _sector_heat(_geometric_widths(OUTER_RADIUS - ANGULAR_INNER, 6, 1.2), np.full(n_angle, span / n_angle),
                                _geometric_widths(HEIGHT, 3, 1.4), ANGULAR_INNER, angle_wrap=angle_wrap)
    return heat, domain, _axis_geometry(domain, 0, ANGULAR_INNER)[2]


def _ring_source(heat, domain, r_centres, columns):
    """Q = ANGULAR_SOURCE / r_c^2 on the angle columns (bool (ny,)); float64 host W per cell (Q x metric volume)."""
    density = ((ANGULAR_SOURCE / r_centres ** 2)[:, None, None]
               * np.asarray(columns, dtype=np.float64)[None, :, None] * np.ones(domain.grid.shape))
    heat.add_heat_source(_everywhere(domain), _full(domain, density))
    return density * heat.system.metric_3d.volume.full()


def _flux_errors(system, expected_flux_phi, prefix):
    """{prefix_flux: angular flux at the centres vs expected, prefix_cross_flux: max |r or z flux| (0 expected)}."""
    flux_r, flux_phi, flux_z = (np.asarray(component, dtype=np.float64) for component in system.flux_at_centres())
    scale = float(np.max(np.abs(expected_flux_phi)))
    return {f'{prefix}_flux': max_relative_error(flux_phi, expected_flux_phi, scale),
            f'{prefix}_cross_flux': max(float(np.max(np.abs(flux_r))), float(np.max(np.abs(flux_z)))) / scale}


def _periodic_ring():
    """Full 2 pi ring, periodic wrap, Q = q0 / r_c^2 everywhere, T = 0 held on the j = 0 cells: T_j = q0 dphi^2 (j (N -
    j) - (N - 1) / 2) / (2 k) on every ring and height, angular flux q0 dphi (2 j - N) / (2 r_c), held outflow -sum Q V."""
    heat, domain, r_centres = _wrapped_heat('periodic', 2.0 * math.pi)
    n_angle = domain.grid.ny
    cell_power = _ring_source(heat, domain, r_centres, np.ones(n_angle, dtype=bool))
    held = np.zeros(domain.grid.shape, dtype=bool)
    held[:, 0, :] = True
    heat.add_fixed_temperature(held, 0.0)
    rise = _solved_rise(heat)
    d_angle = 2.0 * math.pi / n_angle
    index = np.arange(n_angle, dtype=np.float64)
    expected = np.where(index > 0, ANGULAR_SOURCE * d_angle ** 2 / (2.0 * CONDUCTIVITY)
                        * (index * (n_angle - index) - 0.5 * (n_angle - 1)), 0.0)
    errors = {'periodic_rise': max_relative_error(rise, expected[None, :, None], float(np.max(expected)))}
    system = heat.system
    total_power = float(cell_power.sum())
    errors['periodic_held_outflow'] = abs(system.fixed_region_outflow(held) + total_power) / total_power
    expected_flux = np.where(held, 0.0, ANGULAR_SOURCE * d_angle * (2.0 * index[None, :, None] - n_angle)
                             / (2.0 * r_centres[:, None, None]))
    errors.update(_flux_errors(system, expected_flux, 'periodic'))
    return errors


def _antiperiodic_sector():
    """One C_4 sector, antiperiodic wrap, Q = q0 / r_c^2 on the j = 0 cells only, no held cell (the sign-flipping loop
    anchors): T_j = q0 dphi^2 (N - 2 j) / (4 k), angular flux q0 dphi / (2 r_c) off the source column, its outflow 0."""
    heat, domain, r_centres = _wrapped_heat('antiperiodic', 2.0 * math.pi / WRAP_SECTORS)
    n_angle = domain.grid.ny
    column = np.arange(n_angle) == 0
    cell_power = _ring_source(heat, domain, r_centres, column)
    rise = _solved_rise(heat)
    d_angle = 2.0 * math.pi / WRAP_SECTORS / n_angle
    index = np.arange(n_angle, dtype=np.float64)
    expected = ANGULAR_SOURCE * d_angle ** 2 / (4.0 * CONDUCTIVITY) * (n_angle - 2.0 * index)
    errors = {'antiperiodic_rise': max_relative_error(rise, expected[None, :, None], float(np.max(np.abs(expected))))}
    system = heat.system
    column_mask = np.broadcast_to(column[None, :, None], rise.shape).copy()
    errors['antiperiodic_outflow'] = abs(system.fixed_region_outflow(column_mask)) / float(cell_power.sum())
    expected_flux = np.broadcast_to(np.where(column[None, :, None], 0.0, ANGULAR_SOURCE * d_angle
                                             / (2.0 * r_centres[:, None, None])), rise.shape)
    errors.update(_flux_errors(system, expected_flux, 'antiperiodic'))
    return errors


def _wrap_refusals():
    """Number of the expected ValueErrors that did not occur: a periodic wrap without anchor (constant null mode), a
    Dirichlet on a wrapped face, 2 angle cells under a wrap, a sector beyond 2 pi."""

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

    def beyond_full_turn():
        _sector_heat(np.full(4, 1e-5), np.full(8, math.pi / 3.0), np.full(2, 1e-5), ANGULAR_INNER)

    makers = (unanchored_periodic, wrapped_face_condition, two_angle_cells, beyond_full_turn)
    refused = 0
    for make in makers:
        try:
            make()
        except ValueError:
            refused += 1
    return float(len(makers) - refused)


def _check_sector(_results):
    errors = _sector_disk()
    errors.update(_sector_annulus())
    errors.update(_sector_angular())
    errors.update(_sector_axial())
    errors.update(_periodic_ring())
    errors.update(_antiperiodic_sector())
    errors['refusals'] = _wrap_refusals()
    error = float(np.max(list(errors.values())))
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': EXACT_TOLERANCE,
            'details': {'errors': errors}}


# ---- model path: DomainSpec coordinates polar + size_z (cpu backend) ----
HEAT_TYPE_NAME = 'heat'                # = definition.PHYSICS_TYPE_NAME (definition imports this module)
TEMPERATURE_FIELD = 'heat.T'
WALL_TEMPERATURE = 1.0                 # K, held faces and T0: rises of ~1-10 K keep float32 T near 1e-7 relative
MODEL_POWER = 1e10                     # W/m^3
MODEL_CG_TOLERANCE = 1e-13
MODEL_TOLERANCE = 2e-6                 # float32 T / Q storage
# 'ring': full 2 pi periodic ring r < R (uniform 10 um cells: 12 x 76 x 3), Q on the core r < a, T held on x_hi
RING_RADIUS = 1.2e-4                   # m
RING_HEIGHT = 3e-5                     # m
RING_CELL = 1e-5                       # m
CORE_RADIUS = 6e-5                     # m, a cell edge: the mapped centres sit 5 um off it
RING_PROBE = (3.3e-5, 1.0, 1.4e-5)     # (r [m], phi [rad], z [m]) of the 'at' point
# 'layers': a pi / 3 annular sector, Q on z < 16 um, k_top on 26 um < z, T held on z_lo / z_hi (5 um cells)
LAYER_INNER = 2e-5                     # m, radius_min
LAYER_DEPTH = 8e-5                     # m
LAYER_START = 0.25                     # rad, angle_min
LAYER_SPAN = math.pi / 3.0             # rad
LAYER_HEIGHT = 4e-5                    # m
LAYER_CELL = 5e-6                      # m: 16 x 21 x 8 cells
LOWER_TOP = 1.6e-5                     # m, z1 of the heated layer (3 z cells)
CAP_BOTTOM = 2.6e-5                    # m, z0 of the k_top layer (3 z cells)
K_BASE, K_TOP = 2.0, 5.0               # W/(m*K)
LAYER_PROBE = (5e-5, 0.85, 2.2e-5)     # (r, phi, z): z cell 4
LAYER_CUT = 2.8e-5                     # m, a z cut inside the k_top layer (cell 5)


def _polar_point(name, r, phi, z):
    return GeometrySpec(name, 'point', {'x': r * math.cos(phi), 'y': r * math.sin(phi), 'z': z})


def _ring_spec():
    """'ring': stationary heat on a full periodic (r, phi, z) ring, Q on the core circle, T held on x_hi, cpu."""
    return with_exact_cg(ModelSpec(
        title='Self-test: periodic cylindrical ring (model path)',
        domain=DomainSpec(size_x=RING_RADIUS, size_y=2.0 * math.pi, size_z=RING_HEIGHT, coordinates='polar',
                          angle_wrap='periodic'),
        geometry=[GeometrySpec('core', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': CORE_RADIUS}),
                  _polar_point('probe', *RING_PROBE)],
        materials=[MaterialSpec('medium', properties={'k': CONDUCTIVITY}, selection=['all'])],
        physics=[PhysicsSpec('heat', HEAT_TYPE_NAME, settings={'T0': WALL_TEMPERATURE}, features=[
            FeatureSpec('rim', 'temperature', ['x_hi'], {'temperature': WALL_TEMPERATURE}),
            FeatureSpec('source', 'heat_source', ['core'], {'Q': MODEL_POWER})])],
        mesh=MeshSpec(mode='uniform', dx=RING_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('V_total', '1', [], 'integral'),
                        DerivedValueSpec('V_core', '1', ['core'], 'integral'),
                        DerivedValueSpec('r2_integral', 'x**2 + y**2', [], 'integral'),
                        DerivedValueSpec('T_probe', 'heat_T', [], 'at', point='probe')]),
        tolerance=MODEL_CG_TOLERANCE)


def _layer_spec():
    """'layers': stationary heat on an annular sector, Q on the lower layer, k_top on the cap layer, T held on z_lo /
    z_hi, cpu."""
    return with_exact_cg(ModelSpec(
        title='Self-test: layered cylindrical sector (model path)',
        domain=DomainSpec(size_x=LAYER_DEPTH, size_y=LAYER_SPAN, size_z=LAYER_HEIGHT, coordinates='polar',
                          radius_min=LAYER_INNER, angle_min=LAYER_START),
        geometry=[GeometrySpec('lower', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': 1e-3, 'z0': 0.0, 'z1': LOWER_TOP}),
                  GeometrySpec('cap', 'circle', {'cx': 0.0, 'cy': 0.0, 'r': 1e-3, 'z0': CAP_BOTTOM,
                                                 'z1': LAYER_HEIGHT}),
                  _polar_point('probe', *LAYER_PROBE)],
        materials=[MaterialSpec('base', properties={'k': K_BASE}, selection=['all']),
                   MaterialSpec('top', properties={'k': K_TOP}, selection=['cap'])],
        physics=[PhysicsSpec('heat', HEAT_TYPE_NAME, settings={'T0': WALL_TEMPERATURE}, features=[
            FeatureSpec('ends', 'temperature', ['z_lo', 'z_hi'], {'temperature': WALL_TEMPERATURE}),
            FeatureSpec('source', 'heat_source', ['lower'], {'Q': MODEL_POWER})])],
        mesh=MeshSpec(mode='uniform', dx=LAYER_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('V_lower', '1', ['lower'], 'integral'),
                        DerivedValueSpec('T_probe', 'heat_T', [], 'at', point='probe')]),
        tolerance=MODEL_CG_TOLERANCE)


def _model_specs():
    return {'ring': _ring_spec(), 'layers': _layer_spec()}


def _volume_axes(volume_data, radius_min, angle_min):
    """float64 (values, (r, phi, z) edges, (r, phi, z) centres) of a Session.volume of a sector."""
    centres = tuple(np.asarray(axis, dtype=np.float64) for axis in volume_data.centres)
    edges = tuple(cell_edges(axis, low) for axis, low in zip(centres, (radius_min, angle_min, 0.0)))
    return np.asarray(volume_data.values, dtype=np.float64), edges, centres


def _ring_errors(results):
    """Ring: the exact radial chain of the core source in every cell, metric volume integrals, x / y integral, 'at'."""
    values, (r_edges, phi_edges, z_edges), (r_centres, _, _) = _volume_axes(results.volume(TEMPERATURE_FIELD), 0.0,
                                                                           0.0)
    widths = np.diff(r_edges)
    source = np.where(r_centres < CORE_RADIUS, MODEL_POWER, 0.0)
    outflow = np.cumsum(source * r_centres * widths)   # W per radian and m of height through face i + 1
    expected = np.empty(r_centres.size)
    expected[-1] = WALL_TEMPERATURE + outflow[-1] * widths[-1] / (2.0 * CONDUCTIVITY * r_edges[-1])
    for i in range(r_centres.size - 2, -1, -1):
        expected[i] = expected[i + 1] + outflow[i] * (widths[i] + widths[i + 1]) / (2.0 * CONDUCTIVITY * r_edges[i + 1])
    rise = expected - WALL_TEMPERATURE
    derived = results.summary()['derived']
    height, span = float(z_edges[-1]), float(phi_edges[-1] - phi_edges[0])
    core_edge = r_edges[np.searchsorted(r_centres, CORE_RADIUS)]
    probe_cell = int(np.searchsorted(r_edges, RING_PROBE[0], side='right')) - 1
    return {'ring_rise': max_relative_error(values - WALL_TEMPERATURE, rise[:, None, None], float(np.max(rise))),
            'ring_span': abs(span / (2.0 * math.pi) - 1.0),
            'ring_volume': abs(derived['V_total'] / (math.pi * r_edges[-1] ** 2 * height) - 1.0),
            'ring_core_volume': abs(derived['V_core'] / (math.pi * core_edge ** 2 * height) - 1.0),
            'ring_r2_integral': abs(derived['r2_integral'] / (span * height * float(np.sum(r_centres ** 3 * widths)))
                                    - 1.0),
            'ring_probe': abs(derived['T_probe'] - expected[probe_cell]) / float(np.max(rise))}


def _layer_chain(z_edges, z_centres):
    """Exact axial chain per unit area: conductances 1 / (dz_l / 2 k_l + dz_r / 2 k_r), held half cells 2 k / dz, Q dz
    on the lower layer; T at the z centres."""
    widths = np.diff(z_edges)
    conductivity = np.where(z_centres > CAP_BOTTOM, K_TOP, K_BASE)
    source = np.where(z_centres < LOWER_TOP, MODEL_POWER, 0.0) * widths
    links = 1.0 / (widths[:-1] / (2.0 * conductivity[:-1]) + widths[1:] / (2.0 * conductivity[1:]))
    ends = 2.0 * conductivity[[0, -1]] / widths[[0, -1]]
    diagonal = np.concatenate(([0.0], links)) + np.concatenate((links, [0.0]))
    diagonal[0] += ends[0]
    diagonal[-1] += ends[1]
    matrix = np.diag(diagonal) - np.diag(links, 1) - np.diag(links, -1)
    return WALL_TEMPERATURE + np.linalg.solve(matrix, source)


def _layer_errors(results):
    """Layers: the exact axial chain in every cell (mapped z cross-sections of materials and source), the selected
    volume, 'at', the native and drawing-plane z cut, the refused x cut."""
    values, (r_edges, phi_edges, z_edges), (_, _, z_centres) = _volume_axes(
        results.volume(TEMPERATURE_FIELD), LAYER_INNER, LAYER_START)
    expected = _layer_chain(z_edges, z_centres)
    rise_scale = float(np.max(expected - WALL_TEMPERATURE))
    derived = results.summary()['derived']
    area = 0.5 * (r_edges[-1] ** 2 - r_edges[0] ** 2) * float(phi_edges[-1] - phi_edges[0])
    lower_height = float(np.sum(np.diff(z_edges)[z_centres < LOWER_TOP]))
    probe_layer = int(np.searchsorted(z_edges, LAYER_PROBE[2], side='right')) - 1
    cut_layer = int(np.searchsorted(z_edges, LAYER_CUT, side='right')) - 1
    native = results.field(TEMPERATURE_FIELD, plane=('z', LAYER_CUT), native=True)
    image = results.field(TEMPERATURE_FIELD, plane=('z', LAYER_CUT))
    image_values = np.asarray(image.values, dtype=np.float64)
    finite = image_values[np.isfinite(image_values)]
    try:
        results.field(TEMPERATURE_FIELD, plane=('x', 5e-5))
        x_cut_refused = 1.0
    except ValueError:
        x_cut_refused = 0.0
    return {'layer_rise': max_relative_error(values, expected[None, None, :], rise_scale),
            'layer_lower_volume': abs(derived['V_lower'] / (area * lower_height) - 1.0),
            'layer_probe': abs(derived['T_probe'] - expected[probe_layer]) / rise_scale,
            'layer_native_cut': max_relative_error(np.asarray(native.values, dtype=np.float64), expected[cut_layer],
                                                   rise_scale),
            'layer_native_axes': 0.0 if (tuple(native.axes) == ('r', 'phi')
                                         and abs(native.plane[1] - z_centres[cut_layer]) <= 1e-12) else 1.0,
            'layer_image': (max_relative_error(finite, expected[cut_layer], rise_scale)
                            + (0.0 if finite.size < image_values.size else 1.0)),   # NaN outside the sector
            'layer_x_cut_refused': x_cut_refused}


def _check_model_runs(results_by_name):
    errors = _ring_errors(results_by_name['ring'])
    errors.update(_layer_errors(results_by_name['layers']))
    error = float(np.max(np.asarray(list(errors.values()), dtype=np.float64)))   # NaN propagates
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': MODEL_TOLERANCE,
            'details': {'errors': errors}}


def _refusal_specs():
    """{label: (spec, path head, message part)} of model errors validate must report on a sector."""
    spherical = _ring_spec()
    spherical.domain.coordinates = 'spherical'
    spherical.domain.angle_wrap = 'none'
    spherical.domain.size_y = math.pi
    x_cut = _ring_spec()
    x_cut.study.stored_cuts = [['x', 5e-5]]
    segment = _ring_spec()
    segment.analytic_curves = [AnalyticCurveSpec('radial', 'line', '0', line=[0.0, 0.0, 1e-5, 1e-4, 0.0, 1e-5],
                                                 field=TEMPERATURE_FIELD)]
    em = _ring_spec()
    em.physics.append(PhysicsSpec('em', 'em_fdtd_3d'))
    return {'spherical_3d': (spherical, 'geometry', 'cylindrical'),
            'stored_x_cut': (x_cut, 'study', 'constant z'),
            'segment_curve': (segment, 'curve', 'Cartesian 3D'),
            'cartesian_type': (em, 'physics', 'Cartesian grids only')}


def _check_model_paths(_results):
    """CPU: mesh estimate and memory estimate (9-slot wrap) of the ring, mapped box rasters vs closed forms, a clean
    validation and the refusals."""
    from ...model.geometry import domain_map, rasterize_mapped_box, resolve_geometry
    from ...model.meshing import estimate_memory, estimate_mesh
    from ...model.schema import get_registry
    from ...model.validate import validate
    registry = get_registry()
    errors = {}
    ring = _ring_spec()
    mesh = estimate_mesh(ring, {}, registry)
    expected_counts = (round(RING_RADIUS / RING_CELL), math.ceil(2.0 * math.pi * RING_RADIUS / RING_CELL - 1e-9),
                       round(RING_HEIGHT / RING_CELL))
    errors['mesh_counts'] = float((mesh.nx, mesh.ny, mesh.nz) != expected_counts
                                  or mesh.n_cells != math.prod(expected_counts) or mesh.domain_map is None)
    float32_dz = float(np.float32(RING_HEIGHT / expected_counts[2]))   # mapped widths are the Domain's float32 cells
    errors['mesh_height'] = abs(float(mesh.z_edges[-1]) / (expected_counts[2] * float32_dz) - 1.0)
    unwrapped = copy.deepcopy(ring)
    unwrapped.domain.angle_wrap = 'none'
    errors['wrap_build_bytes'] = float(not estimate_memory(ring, {}, registry).gpu_build_peak
                                       > estimate_memory(unwrapped, {}, registry).gpu_build_peak)

    def raster_error(spec, name, inside):
        estimate = estimate_mesh(spec, {}, registry)
        centres = [0.5 * (edges[:-1] + edges[1:]) for edges in (estimate.x_edges, estimate.y_edges, estimate.z_edges)]
        x_mapped, y_mapped = (np.asarray(values, dtype=np.float32) for values in domain_map(spec, {}).to_physical(
            centres[0][:, None], centres[1][None, :]))
        z_centres = centres[2].astype(np.float32)
        window, cropped = rasterize_mapped_box(name, resolve_geometry(spec, {}), x_mapped, y_mapped, z_centres, np)
        mask = np.zeros((x_mapped.shape[0], x_mapped.shape[1], z_centres.size), dtype=bool)
        mask[window] = cropped
        radius = np.hypot(x_mapped.astype(np.float64), y_mapped.astype(np.float64))
        return float(np.count_nonzero(mask != inside(radius[:, :, None], centres[2][None, None, :])))
    errors['raster_core'] = raster_error(ring, 'core', lambda radius, z: (radius < CORE_RADIUS) & (z > -1.0))
    layers = _layer_spec()
    errors['raster_lower'] = raster_error(layers, 'lower', lambda radius, z: (z < LOWER_TOP) & (radius >= 0.0))
    errors['raster_cap'] = raster_error(layers, 'cap', lambda radius, z: (z > CAP_BOTTOM) & (radius >= 0.0))
    for label, spec in (('ring', ring), ('layers', layers)):
        errors[f'clean_{label}'] = float(sum(issue.level == 'error' for issue in validate(spec, registry)))
    for label, (spec, head, text) in _refusal_specs().items():
        errors[f'refused_{label}'] = float(not any(issue.level == 'error' and issue.path[:1] == (head,)
                                                   and text in issue.message for issue in validate(spec, registry)))
    error = float(np.max(list(errors.values())))
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': GEOMETRY_TOLERANCE,
            'details': {'errors': errors}}


CYLINDRICAL_3D_SELFTEST_CASES = (
    SelfTestCase(
        name='metric_cylindrical_geometry_3d', label='Metric: exact areas and volumes of 3D cylindrical sectors',
        description=('CPU check of core.metric.window_metric_3d (symmetry plan 2.21) on graded widths: Cartesian '
                     'volumes / x-face areas as width products; a cylindrical (r, phi, z) sector (radius_min 30 um, '
                     'span 2 rad): summed volumes = (R^2 - r_min^2) Phi H / 2, r_hi A_hi - r_lo A_lo = 2 V on every '
                     'cell (div of r e_r), the outer arc R Phi H, a phi face (R - r_min) H, a z face the sector area, '
                     'phi lengths r_c dphi, zero area on the r = 0 axis face, and the refused inputs (phi beyond 2 '
                     f'pi, a 1D line grid, zero widths). error = largest relative deviation + refusals missed, '
                     f'tolerance {GEOMETRY_TOLERANCE:g}.'),
        specs=lambda: {}, check=_check_geometry, max_wall_time=30.0),
    SelfTestCase(
        name='heat_cylindrical_sector_3d', label='Heat: 3D cylindrical (r, phi, z) sectors, exact discrete solutions',
        description=('Scripted stationary HeatPhysics on padding-free polar Domain3D windows of the cpu backend '
                     f'(symmetry plan 2.21; k = {CONDUCTIVITY:g} W/(m*K) along r, phi and z, float64 CG to '
                     f'{CG_TOLERANCE:g}). (1) Uniform Q = {POWER_DENSITY:g} W/m^3 on the sector r <= '
                     f'{OUTER_RADIUS * 1e6:g} um ({DISK_CELLS} uniform dr, graded phi and z), T(R) = 0: T = Q (R^2 - '
                     'r^2) / (4 k) + Q dr^2 / (16 k) in every cell, cell_volume = r_c dr dphi dz. (2) Source-free '
                     f'annulus (radius_min {ANNULUS_INNER * 1e6:g} um), rise {INNER_RISE:g} K on x_lo, 0 on x_hi: the '
                     'exact chain of half cells 2 k r_f / w and the inflow chain flow x Phi x H. (3) Q = q0 / r_c^2 '
                     'with T = 0 on both phi faces: T = q0 phi (Phi - phi) / (2 k) + q0 dphi^2 / (8 k). (4) Uniform Q '
                     'between held z faces (uniform dz): T = Q z (H - z) / (2 k) + Q dz^2 / (8 k). (5) Full ring, '
                     f'periodic phi wrap ({WRAP_CELLS} cells), Q = q0 / r_c^2, T = 0 held on the j = 0 cells: T_j = '
                     'q0 dphi^2 (j (N - j) - (N - 1) / 2) / (2 k), angular flux q0 dphi (2 j - N) / (2 r_c), no r / z '
                     'flux, held outflow -sum Q V. (6) One C_4 sector, antiperiodic, Q on the j = 0 cells only, no '
                     'held cell: T_j = q0 dphi^2 (N - 2 j) / (4 k), its source column outflow 0 (wrap row sums), flux '
                     'q0 dphi / (2 r_c). (7) Refusals: periodic without anchor, a Dirichlet on a wrapped face, 2 angle '
                     'cells under a wrap, a sector beyond 2 pi. error = largest relative deviation + refusals missed, '
                     f'tolerance {EXACT_TOLERANCE:g} (float32 source densities).'),
        specs=lambda: {}, check=_check_sector, max_wall_time=60.0),
    SelfTestCase(
        name='model_cylindrical_sector_paths', label='Model: cylindrical (r, phi, z) sector mesh, rasters, validation',
        description=('CPU check of the model path of DomainSpec coordinates polar + size_z (symmetry plan 2.21): the '
                     f'mesh estimate of a full periodic ring (R = {RING_RADIUS * 1e6:g} um, H = '
                     f'{RING_HEIGHT * 1e6:g} um, dx = {RING_CELL * 1e6:g} um: radial cells R / dx, angle cells ceil(2 '
                     'pi R / dx), z cells H / dx, the domain map), the heat memory estimate (the phi wrap\'s 9-slot '
                     'build array: more build bytes than the unwrapped ring), geometry.rasterize_mapped_box of a '
                     'whole-height core circle and two z-limited layer prisms against the drawing-plane / z closed '
                     'forms cell by cell, no validation error on both self-test models, and the refusals: spherical '
                     'with size_z, a stored x cut, a 3D segment line curve and a Cartesian-only 3D type (em_fdtd_3d) '
                     'on the sector. error = deviations + misses, tolerance '
                     f'{GEOMETRY_TOLERANCE:g}.'),
        specs=lambda: {}, check=_check_model_paths, max_wall_time=60.0),
    SelfTestCase(
        name='heat_cylindrical_sector_3d_model', label='Heat: cylindrical (r, phi, z) sector models, exact discrete',
        description=('Two stationary heat models on DomainSpec coordinates polar + size_z (symmetry plan 2.21), cpu '
                     f'backend, float64 CG to {MODEL_CG_TOLERANCE:g}, read by Results.volume ((r, phi, z) centres). '
                     f'(1) ring: a full 2 pi periodic ring r < {RING_RADIUS * 1e6:g} um, k = {CONDUCTIVITY:g} '
                     f'W/(m*K), Q = {MODEL_POWER:g} W/m^3 on the core circle r < {CORE_RADIUS * 1e6:g} um (mapped '
                     f'build masks), T = {WALL_TEMPERATURE:g} K on x_hi: the exact radial chain T_i - T_i+1 = F_i (w_i '
                     '+ w_i+1) / (2 k r_f) with F_i = sum Q r_c dr in every cell; derived integrals of 1 (pi R^2 H, '
                     'the core pi a^2 H by the session selection mask) and of x^2 + y^2 (2 pi H sum r_c^3 dr: the '
                     "mapped x, y in 3D), 'at' a drawn point. (2) layers: a pi / 3 annular sector, k_base = "
                     f'{K_BASE:g}, k_top = {K_TOP:g} W/(m*K) on the cap prism z > {CAP_BOTTOM * 1e6:g} um, Q on the '
                     f'lower prism z < {LOWER_TOP * 1e6:g} um, T held on z_lo / z_hi: the exact axial chain '
                     '(conductances 1 / (dz / 2 k_l + dz / 2 k_r), held half cells 2 k / dz) in every cell, the '
                     "selected volume, 'at', the native z cut (axes r, phi, plane centre), the drawing-plane image "
                     '(NaN outside the sector) and the refused x cut. error = largest relative deviation / misses, '
                     f'tolerance {MODEL_TOLERANCE:g}.'),
        specs=_model_specs, check=_check_model_runs, max_wall_time=120.0),
)
