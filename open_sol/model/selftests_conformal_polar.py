"""Self-tests of solver_defaults.conformal_boundaries on polar / spherical domains (solver_plan S7.6, symmetry 2.16 item
3): outline pieces and staircased perimeter weights on the logical (r, phi) cells (CPU), held cut-cell edges and a
perimeter heat flux of circles off the polar origin (heat), concentric sphere terminals off the spherical origin
(electric_currents); cpu model runs."""
import math

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import failed_count_result, with_exact_cg
from .selftests_conformal_cuts import _FAR_CELLS, _RING_TOLERANCE_CELLS, _WALL_RING_CELLS, profile_band_errors
from .spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                   SolverDefaultsSpec, StudySpec)

# ---- polar window: r from 1 to 1.6 um, phi from 0 to 0.5 rad; circles about a point off the origin ----
_CELL = 10e-9                          # m: dr; d phi from the cell at r_max (1.6 um x 0.5 / 80 = 10 nm)
_RADIUS_MIN = 1.0e-6
_RADIAL_DEPTH = 0.6e-6
_ANGLE_SPAN = 0.5
_POLAR_CENTRE = (1.3e-6 * math.cos(0.25), 1.3e-6 * math.sin(0.25))   # drawing plane
_INNER_RADIUS, _OUTER_RADIUS = 83.7e-9, 206.1e-9
_T_INNER, _T_OUTER = 350.0, 300.0
_FLUX = 1e8                            # W/m^2 into the conductor at rho = a (rise q a ln(b / a) ~ 7.5 K at k = 1)
_HELD_TOLERANCE = 5e-3                 # Shortley-Weller cut cells (second order everywhere), as conformal_heat_annulus
_FLUX_TOLERANCE = 1e-2                 # far band beyond _FAR_CELLS (the exact total of the local (dr, r dphi) chords);
#                                        the ring is first order as in conformal_heat_outline; estimated, not measured
# ---- outline geometry (CPU) on the same logical cells (unit-free: lengths in m) ----
_N_RADIAL, _N_ANGLE = 60, 80
_ARC_RADIUS = 1.2337e-6                # an arc about the origin: exact (a coordinate line in the local frame)
_ARC_TOLERANCE = 1e-8
_CIRCLE_LENGTH_TOLERANCE = 1e-3        # 2 pi a of the off-origin circle from the local-frame chords (estimated)
_STAIRCASE_LENGTH_TOLERANCE = 3e-2     # staircased metric perimeter weights (~2 % like the Cartesian ones; estimated)
# ---- spherical window: r from 0.5 to 1.5 um, theta from 0 to 0.6 rad; spheres about the axis point (0, 1 um) ----
_SPHERE_RADIUS_MIN = 0.5e-6
_SPHERE_DEPTH = 1.0e-6
_SPHERE_ANGLE_SPAN = 0.6
_SPHERE_CENTRE_HEIGHT = 1.0e-6
_V_INNER = 1.0
_SPHERE_TOLERANCE = 1e-2               # 1 / rho profile, held cut-cell edges; estimated, not measured


def _polar_map(x_logical, y_logical, radius_min=_RADIUS_MIN):
    """Drawing-plane (x, y) of logical polar (r - radius_min, phi)."""
    radius = radius_min + np.asarray(x_logical, dtype=np.float64)
    return radius * np.cos(y_logical), radius * np.sin(y_logical)


# ---- outline geometry ----
def _check_polar_outline_geometry(results_by_name):
    """Outline pieces on the logical polar cells: an arc about the origin (length, normals, offsets exact), an
    off-origin circle (2 pi a); the staircased metric perimeter weights of that circle (2 pi a to ~2 %)."""
    from ..core.cut_cells import outline_pieces
    from ..core.metric import Coordinates, window_metric
    from .geometry import perimeter_weights
    dr, dphi = _RADIAL_DEPTH / _N_RADIAL, _ANGLE_SPAN / _N_ANGLE
    x_edges, y_edges = np.arange(_N_RADIAL + 1) * dr, np.arange(_N_ANGLE + 1) * dphi
    x_centres, y_centres = 0.5 * (x_edges[:-1] + x_edges[1:]), 0.5 * (y_edges[:-1] + y_edges[1:])

    def arc(x, y):
        return _RADIUS_MIN + np.asarray(x) + 0.0 * np.asarray(y) < _ARC_RADIUS
    arc_region = arc(x_centres[:, None], y_centres[None, :])
    arc_pieces = outline_pieces(arc_region, x_edges, y_edges, arc, radius_min=_RADIUS_MIN)
    arc_offsets = _ARC_RADIUS - (_RADIUS_MIN + x_centres[arc_pieces.owner_i])

    cx, cy = _POLAR_CENTRE

    def circle(x, y):
        mapped_x, mapped_y = _polar_map(x, y)
        return (mapped_x - cx) ** 2 + (mapped_y - cy) ** 2 <= _INNER_RADIUS ** 2
    circle_region = circle(x_centres[:, None], y_centres[None, :])
    circle_pieces = outline_pieces(circle_region, x_edges, y_edges, circle, radius_min=_RADIUS_MIN)
    exact_length = 2.0 * math.pi * _INNER_RADIUS

    metric = window_metric(np.full(_N_RADIAL, dr), np.full(_N_ANGLE, dphi),
                           Coordinates('polar', radius_min=_RADIUS_MIN))
    weights = perimeter_weights(circle_region, np.full(_N_RADIAL, dr), np.full(_N_ANGLE, dphi), np,
                                x_face_scales=metric.x_face_volume_ratios(),
                                y_face_scales=metric.y_face_volume_ratios(np.full(_N_ANGLE, dphi)),
                                y_width_scales=metric.y_length.along_x)
    staircase_length = float(np.sum(np.asarray(weights, dtype=np.float64) * metric.volume.full()))
    return failed_count_result({
        'arc_length': (abs(float(arc_pieces.length.sum()) / (_ARC_RADIUS * _ANGLE_SPAN) - 1.0), _ARC_TOLERANCE),
        'arc_normals': (max(float(np.max(np.abs(arc_pieces.normal_x - 1.0))),
                            float(np.max(np.abs(arc_pieces.normal_y)))), _ARC_TOLERANCE),
        'arc_offsets': (float(np.max(np.abs(arc_pieces.offset - arc_offsets))) / dr, _ARC_TOLERANCE),
        'circle_length': (abs(float(circle_pieces.length.sum()) / exact_length - 1.0), _CIRCLE_LENGTH_TOLERANCE),
        'staircase_length': (abs(staircase_length / exact_length - 1.0), _STAIRCASE_LENGTH_TOLERANCE)},
        details={'arc_pieces': int(arc_pieces.length.size), 'circle_pieces': int(circle_pieces.length.size),
                 'staircase_length_ratio': staircase_length / exact_length})


# ---- heat on the polar window ----
def _polar_geometry(kind):
    """core (rho < a), ring (rho < b), shell = box minus ring (box: the whole sector); kind 'flux' adds outer = a second
    box minus core (composite inputs feed one composite only and cannot be selected)."""
    cx, cy = _POLAR_CENTRE
    sector_box = {'x0': 0.8e-6, 'y0': -0.1e-6, 'x1': 1.7e-6, 'y1': 0.9e-6}
    geometry = [GeometrySpec('core', 'circle', {'cx': cx, 'cy': cy, 'r': _INNER_RADIUS}),
                GeometrySpec('box', 'rectangle', sector_box),
                GeometrySpec('ring', 'circle', {'cx': cx, 'cy': cy, 'r': _OUTER_RADIUS}),
                GeometrySpec('shell', 'difference', {'base': 'box', 'subtract': ['ring']})]
    if kind == 'flux':
        geometry += [GeometrySpec('outer_box', 'rectangle', dict(sector_box)),
                     GeometrySpec('outer', 'difference', {'base': 'outer_box', 'subtract': ['core']})]
    return geometry


def _polar_heat_spec(kind):
    """Steady heat (k = 1, cpu, exact CG, conformal_boundaries on) on the polar sector, held at T_b beyond rho = b;
    kind 'held': the core held at T_a (conductor everywhere); 'flux': no material in the core (an insulated hole) and
    heat_flux q on the perimeter of the conductor around it."""
    if kind == 'held':
        materials = [MaterialSpec('conductor', properties={'k': 1.0}, selection=['all'])]
        edge = FeatureSpec('hot', 'fixed_temperature_domain', ['core'], {'temperature': _T_INNER})
    else:
        materials = [MaterialSpec('conductor', properties={'k': 1.0}, selection=['outer'])]
        edge = FeatureSpec('inner_edge', 'heat_flux', ['outer'], {'flux': _FLUX})
    return with_exact_cg(ModelSpec(
        title=f'Self-test: polar heat, {kind} circle off the origin (conformal)',
        domain=DomainSpec(size_x=_RADIAL_DEPTH, size_y=_ANGLE_SPAN, coordinates='polar', radius_min=_RADIUS_MIN),
        geometry=_polar_geometry(kind),
        materials=materials,
        physics=[PhysicsSpec('heat', 'heat', features=[
            FeatureSpec('cold', 'fixed_temperature_domain', ['shell'], {'temperature': _T_OUTER}), edge])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu', conformal_boundaries=True),
        study=StudySpec(type='stationary')))


def _polar_heat_specs():
    return {f'{kind}_on': _polar_heat_spec(kind) for kind in ('held', 'flux')}


def _native_distance(field, mapping, centre):
    """(values, distance of each native cell centre from `centre` in the drawing plane) of a native (r, angle) field;
    mapping(r, angle) -> drawing-plane (x, y)."""
    values = np.asarray(field.values, dtype=np.float64)
    radius = np.asarray(field.x_centers, dtype=np.float64)[:, None]
    angle = np.asarray(field.y_centers, dtype=np.float64)[None, :]
    x, y = mapping(radius, angle)
    return values, np.hypot(x - centre[0], y - centre[1])


def _polar_heat_errors(results, kind):
    """profile_band_errors of T relative to the profile's drop (held: T_a + (T_b - T_a) ln(rho / a) / ln(b / a); flux:
    T_b + q a ln(b / rho))."""
    temperature, distance = _native_distance(results.field('heat.T', native=True),
                                             lambda r, phi: (r * np.cos(phi), r * np.sin(phi)), _POLAR_CENTRE)
    rho = np.where(distance > _INNER_RADIUS, distance, _OUTER_RADIUS)
    log_ratio = math.log(_OUTER_RADIUS / _INNER_RADIUS)
    rise_above_outer = np.log(_OUTER_RADIUS / rho)   # per unit slope
    if kind == 'held':
        slope = (_T_INNER - _T_OUTER) / log_ratio
        drop = _T_INNER - _T_OUTER
    else:
        slope = _FLUX * _INNER_RADIUS
        drop = slope * log_ratio
    return profile_band_errors(temperature, distance, _T_OUTER + slope * rise_above_outer, drop,
                               slope * rise_above_outer, _CELL, _INNER_RADIUS, _OUTER_RADIUS)


def _check_polar_heat(results_by_name):
    """Held circle: all cells at second order; perimeter flux: the far band against the exact total, the wall ring
    within 1.5 cells of the wall gradient (first order); near band in the details."""
    ring_tolerance = _RING_TOLERANCE_CELLS * _CELL / (_INNER_RADIUS * math.log(_OUTER_RADIUS / _INNER_RADIUS))
    checks, details = {}, {}
    for kind in ('held', 'flux'):
        conformal = _polar_heat_errors(results_by_name[f'{kind}_on'], kind)
        if kind == 'held':
            checks[kind] = (max(conformal['wall_ring'], conformal['near'], conformal['far']), _HELD_TOLERANCE)
        else:
            checks[f'{kind}_far'] = (conformal['far'], _FLUX_TOLERANCE)
            checks[f'{kind}_wall_ring'] = (conformal['wall_ring'], ring_tolerance)
        details[kind] = {'conformal': conformal}
    return failed_count_result(checks, {**details, 'ring_tolerance': ring_tolerance,
                                        'radii_cells': [_INNER_RADIUS / _CELL, _OUTER_RADIUS / _CELL]})


# ---- electric_currents on the spherical window ----
def _sphere_spec():
    """Steady electric_currents (sigma 1, cpu, exact CG, conformal_boundaries on) on the spherical (r, theta) window: a
    sphere rho < a about the axis point (0, 1 um) held at V_a, everything beyond rho = b held at 0 (terminals)."""
    centre = {'cx': 0.0, 'cy': _SPHERE_CENTRE_HEIGHT}
    return with_exact_cg(ModelSpec(
        title='Self-test: spherical currents between drawn sphere terminals (conformal)',
        domain=DomainSpec(size_x=_SPHERE_DEPTH, size_y=_SPHERE_ANGLE_SPAN, coordinates='spherical',
                          radius_min=_SPHERE_RADIUS_MIN),
        geometry=[GeometrySpec('core', 'circle', {**centre, 'r': _INNER_RADIUS}),
                  GeometrySpec('box', 'rectangle', {'x0': -0.1e-6, 'y0': 0.3e-6, 'x1': 1.0e-6, 'y1': 1.6e-6}),
                  GeometrySpec('ring', 'circle', {**centre, 'r': _OUTER_RADIUS}),
                  GeometrySpec('shell', 'difference', {'base': 'box', 'subtract': ['ring']})],
        materials=[MaterialSpec('conductor', properties={'sigma_dc': 1.0}, selection=['all'])],
        physics=[PhysicsSpec('currents', 'electric_currents', settings={'geometry': 'axisymmetric'}, features=[
            FeatureSpec('inner', 'terminal', ['core'], {'voltage': _V_INNER}),
            FeatureSpec('ground', 'terminal', ['shell'], {'voltage': 0.0})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu', conformal_boundaries=True),
        study=StudySpec(type='stationary')))


def _sphere_error(results):
    """max |V - V_exact| over the cells a < rho < b / V_a, V_exact = V_a (1 / rho - 1 / b) / (1 / a - 1 / b)."""
    potential, distance = _native_distance(results.field('currents.V', native=True),
                                           lambda r, theta: (r * np.sin(theta), r * np.cos(theta)),
                                           (0.0, _SPHERE_CENTRE_HEIGHT))
    free = (distance > _INNER_RADIUS) & (distance < _OUTER_RADIUS)
    rho = np.where(free, distance, _OUTER_RADIUS)
    exact = _V_INNER * (1.0 / rho - 1.0 / _OUTER_RADIUS) / (1.0 / _INNER_RADIUS - 1.0 / _OUTER_RADIUS)
    return float(np.max(np.abs(potential - exact)[free])) / _V_INNER


def _check_sphere(results_by_name):
    """Cut-cell sphere terminals within tolerance of the 1 / rho profile."""
    error = _sphere_error(results_by_name['on'])
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _SPHERE_TOLERANCE,
            'details': {'radii_cells': [_INNER_RADIUS / _CELL, _OUTER_RADIUS / _CELL]}}


_POLAR_TEXT = ('a polar sector r = 1 .. 1.6 um, phi = 0 .. 0.5 rad (10 nm dr, 80 angle cells), circles of radius 8.37 '
               'and 20.61 cells about the point r = 1.3 um, phi = 0.25 (off the origin, so no coordinate line)')
CONFORMAL_POLAR_SELFTEST_CASES = (
    SelfTestCase(
        name='conformal_polar_outline_geometry',
        label='Cut cells on polar grids: outline geometry',
        description='CPU (core.cut_cells outline_pieces with radius_min, model.geometry perimeter_weights with the '
                    'polar metric) on the logical cells of ' + _POLAR_TEXT + ': an arc about the origin (r = 1.2337 '
                    f'um) gives its length r x span, radial normals and offsets r - r_c exactly ({_ARC_TOLERANCE:g}); '
                    'the inner circle gives 2 pi a from the local (dr, r dphi) chords '
                    f'({_CIRCLE_LENGTH_TOLERANCE:g}) and from the staircased metric perimeter weights summed over '
                    f'the cell volumes ({_STAIRCASE_LENGTH_TOLERANCE:g}; symmetry 2.16 item 3). Tolerances estimated, '
                    'not measured. error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_polar_outline_geometry),
    SelfTestCase(
        name='conformal_heat_polar',
        label='Cut cells on polar grids: heat circles off the origin',
        description='heat (k = 1, steady, exact CG, cpu backend) on ' + _POLAR_TEXT + ', everything beyond the outer '
                    'circle held at 300 K; held: the inner disc held at 350 K, against T_a + (T_b - T_a) ln(rho / a) / '
                    f'ln(b / a) ({_HELD_TOLERANCE:g} of the drop over all cells: Shortley-Weller edges, second order); '
                    'flux: no material in the inner disc and heat_flux q = 1e8 W/m^2 on the perimeter of the conductor '
                    f'around it, against T_b + q a ln(b / rho) in bands as conformal_heat_outline: the wall ring '
                    f'({_WALL_RING_CELLS:g} cells) within {_RING_TOLERANCE_CELLS:g} cells of the wall gradient '
                    '1 / (a ln(b / a)) (first order there; measured 0.083 vs 0.020 staircased), the near band to '
                    f'{_FAR_CELLS:g} cells reported (measured 0.021), the far band within {_FLUX_TOLERANCE:g} (the '
                    'exact total flux sets it; fitted flux ratio in the details). conformal_boundaries on (held edges at '
                    'their drawn position, outline chords in the local frame; the locator maps the logical cells to '
                    'the drawing plane; the staircase gave held 3.9e-2, flux far band 1.8e-2). Tolerances '
                    'estimated, not measured. error = number of failed sub-checks, tolerance 0.',
        specs=_polar_heat_specs,
        check=_check_polar_heat),
    SelfTestCase(
        name='conformal_currents_spherical',
        label='Cut cells on spherical grids: sphere terminals off the origin',
        description='electric_currents (sigma 1, steady, exact CG, cpu backend, geometry axisymmetric) on a spherical '
                    'window r = 0.5 .. 1.5 um, theta = 0 .. 0.6 rad (10 nm cells): a sphere of radius 8.37 cells '
                    'about the axis point 1 um above the origin held at 1 V and everything beyond a concentric sphere '
                    'of radius 20.61 cells held at 0 V (terminals), conformal_boundaries on: the free cells '
                    'against V_a (1 / rho - 1 / b) / (1 / a - 1 / b), relative to V_a; tolerance '
                    f'{_SPHERE_TOLERANCE:g} for the cut cells (estimated, not measured; staircase 5.2e-2).',
        specs=lambda: {'on': _sphere_spec()},
        check=_check_sphere),
)
