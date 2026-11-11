"""Self-tests of solver_defaults.conformal_boundaries (solver_plan S7.6): exact cut geometry and the conformal PEC
operators of a disk cavity (CPU) and the heat cut cells at a held curved edge (GPU model run); the insulated-ring spec
is graded in selftests_conformal_cuts (k_switch)."""
import math

import numpy as np

from ..core.constants import C0
from .schema import SelfTestCase
from .selftest_helpers import failed_count_result, with_exact_cg
from .spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                   SolverDefaultsSpec, StudySpec)

# ---- cut geometry (CPU): unit cells on [0, 10]^2 ----
_GEOMETRY_CELLS = 10
_HALF_PLANE_ANGLE = math.radians(23.0)
_HALF_PLANE_OFFSET = 4.37              # inside: cos(a) x + sin(a) y > offset
_DISC = (5.13, 4.71, 3.29)             # cx, cy, r
_FRACTION_TOLERANCE = 1e-9             # bisection to 2^-40 of a 1/16 sub-interval
_AREA_TOLERANCE = 5e-4                 # cell area: 32 midpoint rows, O(1 / 32^2) where the free length kinks
_DISC_AREA_TOLERANCE = 1e-3            # disc area from the free areas of all cells (curvature inside the rows)

# ---- disk cavity: vacuum disk of radius 15.37 cells in a PEC block, centre off the cell centres ----
_PHYSICS = 'em'
_CAVITY_CELL = 50e-9
_CAVITY_CELLS = 36
_CAVITY_SIZE = _CAVITY_CELLS * _CAVITY_CELL
_CAVITY_RADIUS = 15.37 * _CAVITY_CELL
_CAVITY_CENTRE = ((18.0 + 0.31) * _CAVITY_CELL, (18.0 + 0.17) * _CAVITY_CELL)
_COURANT = 0.99
_BESSEL_ZERO = {'ez': 2.404825557695773,   # TM01: first zero of J0
                'hz': 1.841183781340659}   # TE11: first zero of J1'
_MODE_NAMES = {'ez': 'TM01', 'hz': 'TE11'}
# conformal omega vs Bessel: measured 5.7e-3 / 6.0e-3 (2026-10-19); at courant 0.99 the vacuum row fills the
# stability budget, so the floor clamps E_z wall fractions to t >= ~0.5 (58 / 124 faces) and small H_z areas. The
# staircase operators (no longer computed) gave 2.4e-2 TM01 / 4.0e-3 TE11, largest eigenvalue dt^2 / 4 0.977
_OPERATOR_TOLERANCE = {'ez': 8e-3, 'hz': 8e-3}
# cavity spec (the geometry the operators are built from): band around the one mode, gaussian pulse, probe / source
# inside the disk (TE11: off the centre); the FDTD ring-down runs in example analytic_conformal_pec_disk_cavity
_BANDS = {'ez': (1.6e-6, 2.6e-6), 'hz': (2.1e-6, 3.4e-6)}
_PULSE_WAVELENGTH = {'ez': 2.0e-6, 'hz': 2.6e-6}
_SOURCE_OFFSET = (0.41, 0.23)          # x radius from the centre
_PROBE_OFFSET = (-0.29, 0.47)
_CAVITY_DURATION = 0.25e-12            # s
_RESONANCE_FEATURE = 'resonances'

# ---- annulus: steady heat between a held disc and a held outside, both drawn off the cell centres ----
_HEAT = 'heat'
_HEAT_CELL = 10e-9
_HEAT_SIZE = 46 * _HEAT_CELL
_HEAT_CENTRE = ((23.0 + 0.27) * _HEAT_CELL, (23.0 - 0.38) * _HEAT_CELL)
_INNER_RADIUS, _OUTER_RADIUS = 8.37 * _HEAT_CELL, 20.61 * _HEAT_CELL
_T_INNER, _T_OUTER = 350.0, 300.0
_ANNULUS_TOLERANCE = 5e-3              # cut cells: second order (Shortley-Weller); staircase 3.7e-2 (no longer run)
# ---- insulated ring: the same held disc, uniformly heated conductor up to the outer circle, k = 0 beyond ----
_SOURCE_DENSITY = 1e15                 # W/m^3: rise Q b^2 ln(b / a) / (2 k) - Q (b^2 - a^2) / (4 k) ~ 10 K at k = 1
_INSULATED_TOLERANCE = 5e-3            # apertures + merged volumes: measured 4.2e-3; staircase 5.7e-2 (no longer run)
_MERGED_AREA_TOLERANCE = 2e-3          # CPU: free area kept by the merged volume fractions around the disc (estimated)


# ---- cut geometry ----
def _half_plane_side(x, y):
    """Signed distance cos(a) x + sin(a) y - offset (> 0 inside)."""
    return math.cos(_HALF_PLANE_ANGLE) * x + math.sin(_HALF_PLANE_ANGLE) * y - _HALF_PLANE_OFFSET


def _half_plane(x, y):
    return _half_plane_side(np.asarray(x), np.asarray(y)) > 0.0


def _exact_segment_free(start_x, start_y, end_x, end_y):
    """Free (outside) fraction of segments against the half-plane (linear distance along the segment)."""
    first, last = _half_plane_side(start_x, start_y), _half_plane_side(end_x, end_y)
    crossing = np.where(first != last, first / np.where(first != last, first - last, 1.0), 0.0)
    return np.where((first <= 0.0) & (last <= 0.0), 1.0,
                    np.where((first > 0.0) & (last > 0.0), 0.0, np.where(first <= 0.0, crossing, 1.0 - crossing)))


def _exact_cell_free_area(x0, x1, y0, y1):
    """Free area of a cell: the rectangle clipped to the half-plane side <= 0 (Sutherland-Hodgman), shoelace."""
    polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    clipped = []
    for index, point in enumerate(polygon):
        previous = polygon[index - 1]
        side, previous_side = _half_plane_side(*point), _half_plane_side(*previous)
        if (side <= 0.0) != (previous_side <= 0.0):
            t = previous_side / (previous_side - side)
            clipped.append((previous[0] + t * (point[0] - previous[0]), previous[1] + t * (point[1] - previous[1])))
        if side <= 0.0:
            clipped.append(point)
    if len(clipped) < 3:
        return 0.0
    return 0.5 * abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(clipped, clipped[1:] + clipped[:1])))


def _check_cut_geometry(results_by_name):
    """Half-plane face fractions and cell areas vs the exact clip, disc area from every cell's free area, the free area
    the insulated-wall volume fractions keep around the disc (cut cells plus merged slivers)."""
    from ..core.cut_cells import free_areas, free_fractions, insulated_wall_geometry
    n = _GEOMETRY_CELLS
    i, j = (index.ravel().astype(np.float64) for index in np.meshgrid(np.arange(n), np.arange(n + 1), indexing='ij'))
    horizontal = free_fractions(_half_plane, i, j, i + 1.0, j)
    vertical = free_fractions(_half_plane, j, i, j, i + 1.0)
    fraction_error = max(float(np.max(np.abs(horizontal - _exact_segment_free(i, j, i + 1.0, j)))),
                         float(np.max(np.abs(vertical - _exact_segment_free(j, i, j, i + 1.0)))))
    x0, y0 = (index.ravel().astype(np.float64) for index in np.meshgrid(np.arange(n), np.arange(n), indexing='ij'))

    def cell_free_areas(test):
        """free_areas of the unit cells, rows chosen as physics/em/conformal_pec does (from the face fractions)."""
        hi_y, lo_y = free_fractions(test, x0, y0 + 1.0, x0 + 1.0, y0 + 1.0), free_fractions(test, x0, y0, x0 + 1.0, y0)
        hi_x, lo_x = free_fractions(test, x0 + 1.0, y0, x0 + 1.0, y0 + 1.0), free_fractions(test, x0, y0, x0, y0 + 1.0)
        return free_areas(test, x0, x0 + 1.0, y0, y0 + 1.0, np.abs(lo_x - hi_x) >= np.abs(lo_y - hi_y))
    areas = cell_free_areas(_half_plane)
    exact = np.array([_exact_cell_free_area(a, a + 1.0, b, b + 1.0) for a, b in zip(x0, y0)])
    cx, cy, radius = _DISC

    def disc(x, y):
        return (np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2 <= radius ** 2
    disc_area = float(n * n - np.sum(cell_free_areas(disc)))
    centres = np.arange(n, dtype=np.float64) + 0.5
    outside = ~disc(centres[:, None], centres[None, :])   # conducting cells: the disc is the insulator
    edges = np.arange(n + 1, dtype=np.float64)
    _, _, volume_fraction = insulated_wall_geometry(outside, outside, edges, edges, np.ones((n, n)), disc)
    merged_area = float(np.sum(volume_fraction[outside]))
    return failed_count_result({
        'face_fractions': (fraction_error, _FRACTION_TOLERANCE),
        'cell_areas': (float(np.max(np.abs(areas - exact))), _AREA_TOLERANCE),
        'disc_area': (abs(disc_area / (math.pi * radius ** 2) - 1.0), _DISC_AREA_TOLERANCE),
        'merged_area': (abs(merged_area / (n * n - math.pi * radius ** 2) - 1.0), _MERGED_AREA_TOLERANCE)})


# ---- disk cavity: geometry, host operators ----
def _cavity_geometry():
    """GeometrySpecs of the cavity: the PEC 'metal' = box minus disk, source and probe points."""
    cx, cy = _CAVITY_CENTRE
    points = {name: {'x': cx + offset[0] * _CAVITY_RADIUS, 'y': cy + offset[1] * _CAVITY_RADIUS}
              for name, offset in (('source_point', _SOURCE_OFFSET), ('probe_point', _PROBE_OFFSET))}
    return [GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _CAVITY_SIZE, 'y1': _CAVITY_SIZE}),
            GeometrySpec('disk', 'circle', {'cx': cx, 'cy': cy, 'r': _CAVITY_RADIUS}),
            GeometrySpec('metal', 'difference', {'base': 'box', 'subtract': ['disk']}),
            *(GeometrySpec(name, 'point', values) for name, values in points.items())]


def _cavity_spec(polarization):
    """em_fdtd_<polarization> ring-down of the PEC disk cavity (pec walls, no PML), conformal_boundaries on."""
    wavelength_min, wavelength_max = _BANDS[polarization]
    pulse = {'waveform': 'gaussian', 'wavelength': _PULSE_WAVELENGTH[polarization], 'bandwidth': 0.3}
    pulse.update({'current': 1e-3} if polarization == 'ez' else
                 {'excitation': 'magnetic_current', 'magnetic_current': 1.0})
    return ModelSpec(
        title=f'Self-test: conformal PEC disk cavity ({polarization})',
        domain=DomainSpec(size_x=_CAVITY_SIZE, size_y=_CAVITY_SIZE),
        geometry=_cavity_geometry(),
        physics=[PhysicsSpec(_PHYSICS, f'em_fdtd_{polarization}',
                             settings={f'bc_{face}': 'pec' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')},
                             features=[FeatureSpec('metal', 'pec_domain', ['metal']),
                                       FeatureSpec('source', 'point_source', ['source_point'], pulse),
                                       FeatureSpec(_RESONANCE_FEATURE, 'resonance_monitor', ['probe_point'],
                                                   {'component': polarization, 'wavelength_min': wavelength_min,
                                                    'wavelength_max': wavelength_max})])],
        mesh=MeshSpec(mode='uniform', dx=_CAVITY_CELL),
        solver_defaults=SolverDefaultsSpec(conformal_boundaries=True),
        study=StudySpec(type='time_dependent', settings={'duration': _CAVITY_DURATION}))


def _cavity_grid():
    """(x edges, y edges float64, Yee widths (float32 dx), float32 cell-centre grids X, Y) of the cavity mesh."""
    edges = np.arange(_CAVITY_CELLS + 1) * _CAVITY_CELL
    widths = np.full(_CAVITY_CELLS, float(np.float32(_CAVITY_CELL)))
    centres = ((np.arange(_CAVITY_CELLS) + 0.5) * _CAVITY_CELL).astype(np.float32)
    shape = (_CAVITY_CELLS, _CAVITY_CELLS)
    return edges, widths, np.broadcast_to(centres[:, None], shape), np.broadcast_to(centres[None, :], shape)


def _cavity_coefficients(polarization, dt):
    """(pec mask, ConformalPEC) as the build makes them: metal cells at the float32 centres, the outline test of
    geometry_locator, physics.em.conformal_pec on the whole grid."""
    from ..physics.em.conformal_pec import ez_wall_faces, hz_cut_cells
    from .boundary_locator import GeometryLocator
    from .geometry import rasterize, resolve_geometry
    spec = _cavity_spec(polarization)
    shapes = resolve_geometry(spec, {})
    edges, widths, centre_x, centre_y = _cavity_grid()
    pec = np.asarray(rasterize('metal', shapes, centre_x, centre_y, np), dtype=bool)
    inside = GeometryLocator(shapes, []).inside(['metal'])
    window = (slice(0, _CAVITY_CELLS), slice(0, _CAVITY_CELLS))
    if polarization == 'ez':
        return pec, ez_wall_faces(edges, edges, widths, widths, pec, pec, inside, dt, window)
    return pec, hz_cut_cells(edges, edges, widths, widths, pec, inside, dt, window, (False, False))


def _tm_operator(pec, conformal, widths):
    """Dense K (omega^2 = eigenvalues) of the E_z update on the free cells: H links with factor 1 + extra."""
    n = pec.shape[0]
    number = -np.ones(pec.shape, dtype=int)
    number[~pec] = np.arange(int(np.count_nonzero(~pec)))
    extra = {(int(cell), int(axis)): float(value) for cell, axis, value in
             zip(conformal.face_cell, conformal.face_axis, conformal.face_extra)}
    operator = np.zeros((int(np.count_nonzero(~pec)),) * 2)
    for axis, (di, dj) in ((0, (1, 0)), (1, (0, 1))):
        for i in range(n - di):
            for j in range(n - dj):
                a, b = (i, j), (i + di, j + dj)
                if pec[a] and pec[b]:
                    continue
                spacing = 0.5 * (widths[a[axis]] + widths[b[axis]])
                factor = 1.0 + extra.get((i * n + j, axis), 0.0)
                for cell, other in ((a, b), (b, a)):
                    if pec[cell]:
                        continue
                    weight = C0 ** 2 * factor / (widths[cell[axis]] * spacing)
                    operator[number[cell], number[cell]] += weight
                    if not pec[other]:
                        operator[number[cell], number[other]] -= weight
    return operator


def _te_operator(keep_x, keep_y, conformal, widths):
    """Dense K of the H_z update on the cells with a live E face: weights (E_x hi, lo, E_y hi, lo) of the corrected
    cells, 1 elsewhere."""
    n = keep_x.shape[0]
    weights = np.ones((n, n, 4))
    weights.reshape(n * n, 4)[conformal.cell] += conformal.cell_extra
    live = np.zeros((n, n), dtype=bool)
    live[:, :-1] |= keep_x[:, :-1] > 0
    live[:, 1:] |= keep_x[:, :-1] > 0
    live[:-1, :] |= keep_y[:-1, :] > 0
    live[1:, :] |= keep_y[:-1, :] > 0
    number = -np.ones((n, n), dtype=int)
    number[live] = np.arange(int(np.count_nonzero(live)))
    operator = np.zeros((int(np.count_nonzero(live)),) * 2)
    for keep, axis, (di, dj), (lower_slot, upper_slot) in ((keep_x, 1, (0, 1), (0, 1)), (keep_y, 0, (1, 0), (2, 3))):
        for i in range(n - di):
            for j in range(n - dj):
                if keep[i, j] <= 0:
                    continue
                a, b = (i, j), (i + di, j + dj)
                spacing = 0.5 * (widths[a[axis]] + widths[b[axis]])
                for cell, other, slot in ((a, b, lower_slot), (b, a, upper_slot)):
                    weight = C0 ** 2 * weights[cell][slot] / (widths[cell[axis]] * spacing)
                    operator[number[cell], number[cell]] += weight
                    operator[number[cell], number[other]] -= weight
    return operator


def _operator_eigenvalues(polarization, dt):
    """Sorted real parts of every eigenvalue [1/s^2] of the conformal host operator of the cavity (eigvalsh when the
    operator is symmetric: the E_z one on uniform cells)."""
    pec, coefficients = _cavity_coefficients(polarization, dt)
    widths = _cavity_grid()[1]
    if polarization == 'ez':
        operator = _tm_operator(pec, coefficients, widths)
    else:
        operator = _te_operator(coefficients.keep_x, coefficients.keep_y, coefficients, widths)
    if np.array_equal(operator, operator.T):
        return np.linalg.eigvalsh(operator), coefficients
    return np.sort(np.real(np.linalg.eigvals(operator))), coefficients


def _cavity_dt():
    """Engine dt of the uniform cavity mesh at courant 0.99 (core.grid rule: fraction of the 2D CFL limit)."""
    cell = float(np.float32(_CAVITY_CELL))
    return _COURANT / (C0 * math.sqrt(2.0) / cell)


def _nearest_omega(eigenvalues, target):
    """sqrt of the eigenvalue nearest target^2."""
    return math.sqrt(float(eigenvalues[np.argmin(np.abs(eigenvalues - target ** 2))]))


def _check_operator(results_by_name):
    """Per polarisation: conformal omega of the lowest mode vs Bessel and the largest eigenvalue against the leap-frog
    limit 4 / dt^2."""
    dt = _cavity_dt()
    checks, details = {}, {}
    for polarization in ('ez', 'hz'):
        exact = C0 * _BESSEL_ZERO[polarization] / _CAVITY_RADIUS
        conformal_values, coefficients = _operator_eigenvalues(polarization, dt)
        conformal_error = abs(_nearest_omega(conformal_values, exact) / exact - 1.0)
        mode = _MODE_NAMES[polarization]
        checks[f'{mode}_conformal'] = (conformal_error, _OPERATOR_TOLERANCE[polarization])
        checks[f'{polarization}_stability'] = (float(conformal_values[-1]) * dt ** 2 / 4.0, 1.0)
        details[polarization] = {'conformal_error': conformal_error,
                                 'largest_eigenvalue_dt2_over_4': float(conformal_values[-1]) * dt ** 2 / 4.0,
                                 'coefficients': coefficients.summary()}
    return failed_count_result(checks, details)


# ---- annulus ----
def _annulus_spec():
    """Steady heat (k = 1) between a disc held at T_inner and everything beyond the outer circle held at T_outer,
    conformal_boundaries on."""
    cx, cy = _HEAT_CENTRE
    return with_exact_cg(ModelSpec(
        title='Self-test: heat annulus between drawn held edges (conformal)',
        domain=DomainSpec(size_x=_HEAT_SIZE, size_y=_HEAT_SIZE),
        geometry=[GeometrySpec('core', 'circle', {'cx': cx, 'cy': cy, 'r': _INNER_RADIUS}),
                  GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _HEAT_SIZE, 'y1': _HEAT_SIZE}),
                  GeometrySpec('ring', 'circle', {'cx': cx, 'cy': cy, 'r': _OUTER_RADIUS}),
                  GeometrySpec('shell', 'difference', {'base': 'box', 'subtract': ['ring']})],
        materials=[MaterialSpec('conductor', properties={'k': 1.0}, selection=['all'])],
        physics=[PhysicsSpec(_HEAT, 'heat', features=[
            FeatureSpec('hot', 'fixed_temperature_domain', ['core'], {'temperature': _T_INNER}),
            FeatureSpec('cold', 'fixed_temperature_domain', ['shell'], {'temperature': _T_OUTER})])],
        mesh=MeshSpec(mode='uniform', dx=_HEAT_CELL),
        solver_defaults=SolverDefaultsSpec(conformal_boundaries=True),
        study=StudySpec(type='stationary')))


def _annulus_error(results):
    """max |T - T_exact| over the free cells (a < r < b) / (T_inner - T_outer), T_exact the logarithmic profile."""
    field = results.field(f'{_HEAT}.T')
    temperature = np.asarray(field.values, dtype=np.float64)
    x = np.asarray(field.x_centers, dtype=np.float64)[:, None] - _HEAT_CENTRE[0]
    y = np.asarray(field.y_centers, dtype=np.float64)[None, :] - _HEAT_CENTRE[1]
    radius = np.sqrt(x ** 2 + y ** 2)
    free = (radius > _INNER_RADIUS) & (radius < _OUTER_RADIUS)
    shape = np.log(np.where(free, radius, _INNER_RADIUS) / _INNER_RADIUS) / math.log(_OUTER_RADIUS / _INNER_RADIUS)
    exact = _T_INNER + (_T_OUTER - _T_INNER) * shape
    return float(np.max(np.abs(temperature - exact)[free])) / (_T_INNER - _T_OUTER)


def _check_annulus(results_by_name):
    """Cut-cell annulus within tolerance of T(r) = T_a + (T_b - T_a) ln(r / a) / ln(b / a)."""
    error = _annulus_error(results_by_name['on'])
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _ANNULUS_TOLERANCE,
            'details': {'radii_cells': [_INNER_RADIUS / _HEAT_CELL, _OUTER_RADIUS / _HEAT_CELL]}}


# ---- insulated ring ----
def _insulated_spec():
    """Steady heat: conductor (k = 1) and Q only inside the outer circle (no material beyond: k = 0, an insulated
    curved wall), the inner disc held at T_inner, conformal_boundaries on (graded in conformal_heat_k_switch)."""
    cx, cy = _HEAT_CENTRE
    return with_exact_cg(ModelSpec(
        title='Self-test: heated ring inside a drawn insulated wall (conformal)',
        domain=DomainSpec(size_x=_HEAT_SIZE, size_y=_HEAT_SIZE),
        geometry=[GeometrySpec('core', 'circle', {'cx': cx, 'cy': cy, 'r': _INNER_RADIUS}),
                  GeometrySpec('ring', 'circle', {'cx': cx, 'cy': cy, 'r': _OUTER_RADIUS})],
        materials=[MaterialSpec('conductor', properties={'k': 1.0}, selection=['ring'])],
        physics=[PhysicsSpec(_HEAT, 'heat', features=[
            FeatureSpec('hot', 'fixed_temperature_domain', ['core'], {'temperature': _T_INNER}),
            FeatureSpec('heating', 'heat_source', ['ring'], {'Q': _SOURCE_DENSITY})])],
        mesh=MeshSpec(mode='uniform', dx=_HEAT_CELL),
        solver_defaults=SolverDefaultsSpec(conformal_boundaries=True),
        study=StudySpec(type='stationary')))


def _insulated_error(results):
    """max |T - T_exact| over the cells a < r < b / the rise T(b) - T_a, T_exact = T_a + Q (a^2 - r^2) / (4 k) + Q b^2
    ln(r / a) / (2 k) (dT/dr = 0 on r = b)."""
    field = results.field(f'{_HEAT}.T')
    temperature = np.asarray(field.values, dtype=np.float64)
    x = np.asarray(field.x_centers, dtype=np.float64)[:, None] - _HEAT_CENTRE[0]
    y = np.asarray(field.y_centers, dtype=np.float64)[None, :] - _HEAT_CENTRE[1]
    radius = np.sqrt(x ** 2 + y ** 2)
    free = (radius > _INNER_RADIUS) & (radius < _OUTER_RADIUS)

    def exact(r):
        return (_T_INNER + _SOURCE_DENSITY * (_INNER_RADIUS ** 2 - r ** 2) / 4.0
                + _SOURCE_DENSITY * _OUTER_RADIUS ** 2 * np.log(r / _INNER_RADIUS) / 2.0)
    rise = float(exact(_OUTER_RADIUS)) - _T_INNER
    return float(np.max(np.abs(temperature - exact(np.where(free, radius, _INNER_RADIUS)))[free])) / rise


_CAVITY_TEXT = (f'vacuum disk of radius {_CAVITY_RADIUS * 1e9:g} nm (15.37 cells of {_CAVITY_CELL * 1e9:g} nm, centre '
                '(0.31, 0.17) cells off a cell centre) cut out of a PEC block')
CONFORMAL_SELFTEST_CASES = (
    SelfTestCase(
        name='conformal_cut_geometry',
        label='Conformal boundaries: exact cut geometry',
        description='CPU (core.cut_cells): free fractions of every face of a 10 x 10 grid against a half-plane at 23 '
                    f'degrees ({_FRACTION_TOLERANCE:g} of a face), free cell areas against the exact clipped polygon '
                    f'({_AREA_TOLERANCE:g} of a cell: 32 midpoint rows across the boundary), and a disc area from the '
                    f'free areas of all cells ({_DISC_AREA_TOLERANCE:g} relative), and the free area around the disc '
                    'kept by the insulated-wall volume fractions (cut cells + merged slivers, '
                    f'{_MERGED_AREA_TOLERANCE:g} relative). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_cut_geometry),
    SelfTestCase(
        name='conformal_pec_operator',
        label='Conformal PEC: disk cavity operators',
        description=f'CPU (physics/em/conformal_pec, dense eigenvalues): the E_z and H_z FDTD operators of a '
                    f'{_CAVITY_TEXT}, with the conformal coefficients the engines get. TM01 / TE11 omega within '
                    f'{_OPERATOR_TOLERANCE["ez"]:g} / {_OPERATOR_TOLERANCE["hz"]:g} of c j / R (j = 2.4048, 1.8412; '
                    'the margin holds the 5-point dispersion and the stability floor of the cut cells), and the '
                    'largest eigenvalue times dt^2 / 4 below 1 at courant 0.99 (the cut cells do not set the time '
                    'step). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_operator),
    SelfTestCase(
        name='conformal_heat_annulus',
        label='Cut cells: heat annulus between drawn held edges',
        description='heat (k = 1, steady, exact CG) on 10 nm cells: a disc of radius 8.37 cells held at 350 K and '
                    'everything beyond a circle of radius 20.61 cells held at 300 K (fixed_temperature_domain, centre '
                    'off the cell centres), conformal_boundaries on (held edges at their drawn position, '
                    'core.cut_cells): the free cells against T(r) = T_a + (T_b - T_a) ln(r / a) / ln(b / a), '
                    f'relative to T_a - T_b; tolerance {_ANNULUS_TOLERANCE:g} (staircase: 3.7e-2).',
        specs=lambda: {'on': _annulus_spec()},
        check=_check_annulus),
)
