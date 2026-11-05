"""Self-tests of the later cut-cell parts of solver_defaults.conformal_boundaries (solver_plan S7.6): outline pieces and
dual-cell cuts (CPU), perimeter heat_flux / convection on a drawn curved edge and insulated walls, static and following
k(T) (GPU heat runs), the em_modes conformal PEC of a circular metal pipe (cpu model runs)."""
import math

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import with_exact_cg
from .selftests_conformal import (
    _AREA_TOLERANCE, _BESSEL_ZERO, _CAVITY_CELL, _CAVITY_CENTRE, _CAVITY_RADIUS, _CAVITY_SIZE, _FRACTION_TOLERANCE,
    _GEOMETRY_CELLS, _HALF_PLANE_ANGLE, _HALF_PLANE_OFFSET, _HEAT, _HEAT_CELL, _HEAT_CENTRE, _HEAT_SIZE, _INNER_RADIUS,
    _INSULATED_TOLERANCE, _OUTER_RADIUS, _SOURCE_DENSITY, _T_INNER, _T_OUTER, _exact_cell_free_area,
    _exact_segment_free, _failed_count_result, _half_plane, _half_plane_side, _insulated_error, _insulated_spec)
from .spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                   SolverDefaultsSpec, StudySpec)

# ---- outline geometry (CPU): the half-plane of selftests_conformal on its 10 x 10 unit cells, plus ----
_OUTLINE_TOLERANCE_EXACT = 1e-8          # straight / face-aligned outlines: exact up to the 1e-9 face inset
_RECTANGLE = (2.0, 3.0, 7.0, 8.0)        # face-aligned rectangle x0, y0, x1, y1 [cells] (half-open, as rasterize)
_OUTLINE_DISC_CELLS = 40
_OUTLINE_DISC = (20.13, 19.71, 15.37)    # cx, cy, r [cells]
_DISC_OUTLINE_TOLERANCE = 1e-4           # 2 pi R from the refined chords (estimated: (theta / 4)^2 / 24 per piece)

# ---- outline heat: a hole (r < a, no material) inside a conductor held beyond r = b ----
_FLUX = 1e8                              # W/m^2 into the conductor at r = a: rise q a ln(b / a) / k ~ 7.5 K at k = 1
_TRANSFER = 1.0 / _INNER_RADIUS          # W/(m^2 K): Biot number h a / k = 1, so the h delta / k_n term matters
_T_FLUID = 400.0                         # K
_OUTLINE_HEAT_TOLERANCE = 1e-2           # far band: the total flux (exact) sets the profile; estimated, not measured
_WALL_RING_CELLS = 2.0                   # the two cell rings next to the wall: first order (cut-cell unknowns at centres)
_FAR_CELLS = 5.0                         # beyond: the dipole tails of flux entering a cell off the wall (~2.6 % of the
#                                          rise at 2 cells, measured) have decayed
_RING_TOLERANCE_CELLS = 1.5              # wall ring within 1.5 cells of the wall gradient 1 / (a ln(b / a)) of the rise

# ---- k(T) walls: the static insulated ring, and that ring plus a skin b < r < c that stops conducting when hot ----
_SKIN_RADIUS = 22.3 * _HEAT_CELL
_SWITCH_TEMPERATURE = 330.0              # K: above the start T0 = 293.15 K, below the held 350 K

# ---- em_modes pipe: vacuum disk of the conformal cavity in a PEC block, k0 R = 3.5 (TE11, TM01, TE21 guided) ----
_PIPE = 'modes'
_PIPE_TABLE = 'table'
_PIPE_K0_RADIUS = 3.5
_PIPE_WAVELENGTH = 2.0 * math.pi * _CAVITY_RADIUS / _PIPE_K0_RADIUS
_PIPE_MODES = 3                          # TE11 pair, TM01 (the TE21 pair, n_eff ~0.53, lies below the shift's three)
# staircase (no longer run): outline far band 1.8e-2 flux / 2.6e-2 convection (wall ring 4.4e-2 / 5.6e-2); pipe k_c R
# 4.4e-2 TE11 / 2.4e-2 TM01; the k(T) switch ends at the insulated ring's 5.7e-2
_PIPE_GUESS = 0.95                       # n_eff shift above TE11 (sqrt(1 - (1.8412 / 3.5)^2) = 0.851)
_PIPE_N_EFF_COLUMN = 1                   # mode_table rows: mode_number, n_eff, ...
_PIPE_TOLERANCE = {'TE11': 5e-3, 'TM01': 3e-3}   # k_c R vs the Bessel zero; estimated, not measured


# ---- outline geometry ----
def _window_chord(size):
    """Length of the half-plane line cos(a) x + sin(a) y = offset inside [0, size]^2."""
    cosine, sine = math.cos(_HALF_PLANE_ANGLE), math.sin(_HALF_PLANE_ANGLE)
    points = [(x, (_HALF_PLANE_OFFSET - cosine * x) / sine) for x in (0.0, size)]
    points += [((_HALF_PLANE_OFFSET - sine * y) / cosine, y) for y in (0.0, size)]
    points = [(x, y) for x, y in points if -1e-12 <= x <= size + 1e-12 and -1e-12 <= y <= size + 1e-12]
    return max(math.hypot(a[0] - b[0], a[1] - b[1]) for a in points for b in points)


def _check_outline_geometry(results_by_name):
    """Outline pieces of the half-plane (window chord, owner offsets, normals), of a face-aligned rectangle (perimeter,
    half-cell offsets) and of a disc (2 pi R); dual-cell cuts of the half-plane (link fractions, square areas)."""
    from ..core.cut_cells import DUAL_CUT_FLOOR, dual_cell_cuts, outline_pieces
    n = _GEOMETRY_CELLS
    edges = np.arange(n + 1, dtype=np.float64)
    centres = edges[:-1] + 0.5
    region = _half_plane(centres[:, None], centres[None, :])
    line = outline_pieces(region, edges, edges, _half_plane)
    owner_side = _half_plane_side(centres[line.owner_i], centres[line.owner_j])
    normal_error = max(float(np.max(np.abs(line.normal_x + math.cos(_HALF_PLANE_ANGLE)))),
                       float(np.max(np.abs(line.normal_y + math.sin(_HALF_PLANE_ANGLE)))))

    x0, y0, x1, y1 = _RECTANGLE

    def rectangle(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    rectangle_region = rectangle(centres[:, None], centres[None, :])
    box = outline_pieces(rectangle_region, edges, edges, rectangle)

    m = _OUTLINE_DISC_CELLS
    disc_edges = np.arange(m + 1, dtype=np.float64)
    disc_centres = disc_edges[:-1] + 0.5
    cx, cy, radius = _OUTLINE_DISC

    def disc(x, y):
        return (np.asarray(x) - cx) ** 2 + (np.asarray(y) - cy) ** 2 <= radius ** 2
    circle = outline_pieces(disc(disc_centres[:, None], disc_centres[None, :]), disc_edges, disc_edges, disc)

    link_x, link_y, area = dual_cell_cuts(centres, centres, region, _half_plane)
    i, j = np.nonzero(region[:-1, :] != region[1:, :])
    exact_x = np.maximum(_exact_segment_free(centres[i], centres[j], centres[i + 1], centres[j]), DUAL_CUT_FLOOR)
    column_i, column_j = np.nonzero(region[:, :-1] != region[:, 1:])
    exact_y = np.maximum(_exact_segment_free(centres[column_i], centres[column_j], centres[column_i],
                                             centres[column_j + 1]), DUAL_CUT_FLOOR)
    corners = (region[:-1, :-1].astype(np.int8) + region[1:, :-1] + region[:-1, 1:] + region[1:, 1:])
    p, q = np.nonzero((corners > 0) & (corners < 4))
    exact_area = np.clip([_exact_cell_free_area(centres[a], centres[a + 1], centres[b], centres[b + 1])
                          for a, b in zip(p, q)], DUAL_CUT_FLOOR, 1.0)
    return _failed_count_result({
        'line_length': (abs(float(line.length.sum()) / _window_chord(float(n)) - 1.0), _OUTLINE_TOLERANCE_EXACT),
        'line_offsets': (float(np.max(np.abs(line.offset - owner_side))), _OUTLINE_TOLERANCE_EXACT),
        'line_normals': (normal_error, _OUTLINE_TOLERANCE_EXACT),
        'line_owners_outside': (float(np.count_nonzero(~region[line.owner_i, line.owner_j])), 0.0),
        'rectangle_length': (abs(float(box.length.sum()) / (2.0 * (x1 - x0 + y1 - y0)) - 1.0),
                             _OUTLINE_TOLERANCE_EXACT),
        'rectangle_offsets': (float(np.max(np.abs(box.offset - 0.5))), _OUTLINE_TOLERANCE_EXACT),
        'rectangle_owners_outside': (float(np.count_nonzero(~rectangle_region[box.owner_i, box.owner_j])), 0.0),
        'disc_length': (abs(float(circle.length.sum()) / (2.0 * math.pi * radius) - 1.0), _DISC_OUTLINE_TOLERANCE),
        'dual_links': (max(float(np.max(np.abs(link_x[i, j] - exact_x))),
                           float(np.max(np.abs(link_y[column_i, column_j] - exact_y)))), _FRACTION_TOLERANCE),
        'dual_areas': (float(np.max(np.abs(area[p, q] - exact_area))), _AREA_TOLERANCE)},
        details={'line_pieces': int(line.length.size), 'rectangle_pieces': int(box.length.size),
                 'disc_pieces': int(circle.length.size)})


# ---- outline heat ----
def _outline_spec(kind):
    """Steady heat (k = 1) outside a hole r < a (no material: an insulated curved wall), held at T_b beyond r = b;
    heat_flux q (kind 'flux') or convection h, T_fluid (kind 'convection') on the perimeter of the object outside
    the hole (its outline is the circle a); conformal_boundaries on."""
    cx, cy = _HEAT_CENTRE
    edge = (FeatureSpec('inner_edge', 'heat_flux', ['outer'], {'flux': _FLUX}) if kind == 'flux' else
            FeatureSpec('inner_edge', 'convection', ['outer'], {'h': _TRANSFER, 'T_ext': _T_FLUID}))
    box = {'x0': 0.0, 'y0': 0.0, 'x1': _HEAT_SIZE, 'y1': _HEAT_SIZE}
    return with_exact_cg(ModelSpec(
        title=f'Self-test: {kind} on a drawn curved edge (conformal)',
        domain=DomainSpec(size_x=_HEAT_SIZE, size_y=_HEAT_SIZE),
        geometry=[GeometrySpec('core', 'circle', {'cx': cx, 'cy': cy, 'r': _INNER_RADIUS}),
                  GeometrySpec('box', 'rectangle', box),
                  GeometrySpec('ring', 'circle', {'cx': cx, 'cy': cy, 'r': _OUTER_RADIUS}),
                  GeometrySpec('shell', 'difference', {'base': 'box', 'subtract': ['ring']}),
                  # composite inputs feed one composite only: a second box for outer
                  GeometrySpec('outer_box', 'rectangle', dict(box)),
                  GeometrySpec('outer', 'difference', {'base': 'outer_box', 'subtract': ['core']})],
        materials=[MaterialSpec('conductor', properties={'k': 1.0}, selection=['outer'])],
        physics=[PhysicsSpec(_HEAT, 'heat', features=[
            FeatureSpec('cold', 'fixed_temperature_domain', ['shell'], {'temperature': _T_OUTER}), edge])],
        mesh=MeshSpec(mode='uniform', dx=_HEAT_CELL),
        solver_defaults=SolverDefaultsSpec(conformal_boundaries=True),
        study=StudySpec(type='stationary')))


def _outline_specs():
    return {f'{kind}_on': _outline_spec(kind) for kind in ('flux', 'convection')}


def _outline_slope(kind):
    """C of T = T_b + C ln(b / r) (k = 1): q a for the flux, h a (T_fluid - T_b) / (1 + h a ln(b / a)) for convection."""
    if kind == 'flux':
        return _FLUX * _INNER_RADIUS
    biot = _TRANSFER * _INNER_RADIUS
    return biot * (_T_FLUID - _T_OUTER) / (1.0 + biot * math.log(_OUTER_RADIUS / _INNER_RADIUS))


def profile_band_errors(temperature, distance, exact, drop, exact_rise, cell, inner_radius, outer_radius):
    """{wall_ring, near, far: max |T - exact| / drop over the cells inner_radius < distance < + _WALL_RING_CELLS cells,
    up to + _FAR_CELLS cells and beyond (< outer_radius); flux_ratio: fitted / exact profile slope over the far band
    = 1 + mean (T - exact) / exact_rise, exact_rise the exact rise above the outer held edge per cell}."""
    deviation = (temperature - exact) / drop
    ring_edge, far_edge = inner_radius + _WALL_RING_CELLS * cell, inner_radius + _FAR_CELLS * cell
    wall_ring = (distance > inner_radius) & (distance < ring_edge)
    near = (distance >= ring_edge) & (distance < far_edge)
    far = (distance >= far_edge) & (distance < outer_radius)
    return {'wall_ring': float(np.max(np.abs(deviation[wall_ring]))), 'near': float(np.max(np.abs(deviation[near]))),
            'far': float(np.max(np.abs(deviation[far]))),
            'flux_ratio': 1.0 + float(np.mean((temperature - exact)[far] / exact_rise[far]))}


def _outline_errors(results, kind):
    """profile_band_errors of T against T_b + C ln(b / r) relative to the rise C ln(b / a)."""
    field = results.field(f'{_HEAT}.T')
    temperature = np.asarray(field.values, dtype=np.float64)
    x = np.asarray(field.x_centers, dtype=np.float64)[:, None] - _HEAT_CENTRE[0]
    y = np.asarray(field.y_centers, dtype=np.float64)[None, :] - _HEAT_CENTRE[1]
    radius = np.sqrt(x ** 2 + y ** 2)
    slope = _outline_slope(kind)
    log_ratio = np.log(_OUTER_RADIUS / np.where(radius > _INNER_RADIUS, radius, _OUTER_RADIUS))
    return profile_band_errors(temperature, radius, _T_OUTER + slope * log_ratio,
                               slope * math.log(_OUTER_RADIUS / _INNER_RADIUS), slope * log_ratio, _HEAT_CELL,
                               _INNER_RADIUS, _OUTER_RADIUS)


def _check_outline_heat(results_by_name):
    """Cut-cell flux / convection on the drawn circle a: the far band against the log profile (the exact total), the
    wall ring within 1.5 cells of the wall gradient (first order); near band in the details."""
    ring_tolerance = _RING_TOLERANCE_CELLS * _HEAT_CELL / (_INNER_RADIUS * math.log(_OUTER_RADIUS / _INNER_RADIUS))
    checks, details = {}, {}
    for kind in ('flux', 'convection'):
        conformal = _outline_errors(results_by_name[f'{kind}_on'], kind)
        checks[f'{kind}_far'] = (conformal['far'], _OUTLINE_HEAT_TOLERANCE)
        checks[f'{kind}_wall_ring'] = (conformal['wall_ring'], ring_tolerance)
        details[kind] = {'conformal': conformal}
    return _failed_count_result(checks, {**details, 'ring_tolerance': ring_tolerance,
                                         'radii_cells': [_INNER_RADIUS / _HEAT_CELL, _OUTER_RADIUS / _HEAT_CELL],
                                         'biot': _TRANSFER * _INNER_RADIUS})


# ---- k(T) walls ----
def _switch_spec():
    """The heated insulated ring (selftests_conformal._insulated_spec) plus a skin b < r < c with k = 1 below 330 K, 0
    above: the first stationary iteration conducts through the skin (walls at c), the next ones not (walls re-located
    to b); conformal_boundaries on."""
    cx, cy = _HEAT_CENTRE
    return with_exact_cg(ModelSpec(
        title='Self-test: insulated walls following k(T) (conformal)',
        domain=DomainSpec(size_x=_HEAT_SIZE, size_y=_HEAT_SIZE),
        geometry=[GeometrySpec('core', 'circle', {'cx': cx, 'cy': cy, 'r': _INNER_RADIUS}),
                  GeometrySpec('ring', 'circle', {'cx': cx, 'cy': cy, 'r': _OUTER_RADIUS}),
                  GeometrySpec('skin_disc', 'circle', {'cx': cx, 'cy': cy, 'r': _SKIN_RADIUS}),
                  # ring stays selectable: the skin subtracts a copy of it (composite inputs are consumed)
                  GeometrySpec('skin_hole', 'circle', {'cx': cx, 'cy': cy, 'r': _OUTER_RADIUS}),
                  GeometrySpec('skin', 'difference', {'base': 'skin_disc', 'subtract': ['skin_hole']})],
        materials=[MaterialSpec('conductor', properties={'k': 1.0}, selection=['ring']),
                   MaterialSpec('skin', properties={'k': f'where(T < {_SWITCH_TEMPERATURE:g}, 1.0, 0.0)'},
                                selection=['skin'])],
        physics=[PhysicsSpec(_HEAT, 'heat', features=[
            FeatureSpec('hot', 'fixed_temperature_domain', ['core'], {'temperature': _T_INNER}),
            FeatureSpec('heating', 'heat_source', ['ring'], {'Q': _SOURCE_DENSITY})])],
        mesh=MeshSpec(mode='uniform', dx=_HEAT_CELL),
        solver_defaults=SolverDefaultsSpec(conformal_boundaries=True),
        study=StudySpec(type='stationary')))


def _switch_specs():
    return {'switch': _switch_spec(), 'insulated': _insulated_spec()}


def _check_switch(results_by_name):
    """The static insulated ring (walls drawn at b) and, after the skin switched off, the k(T) ring (walls re-located
    from c to b) against the insulated-ring profile; error = number of failed sub-checks, tolerance 0."""
    return _failed_count_result(
        {name: (_insulated_error(results_by_name[name]), _INSULATED_TOLERANCE) for name in ('insulated', 'switch')},
        {'radii_cells': [_INNER_RADIUS / _HEAT_CELL, _OUTER_RADIUS / _HEAT_CELL, _SKIN_RADIUS / _HEAT_CELL]})


# ---- em_modes pipe ----
def _pipe_spec():
    """em_modes (cpu backend, conformal_boundaries on) of the vacuum disk of the conformal cavity in a PEC block, pec
    faces, mode table."""
    cx, cy = _CAVITY_CENTRE
    settings = {f'bc_{face}': 'pec' for face in ('x_lo', 'x_hi', 'y_lo', 'y_hi')}
    settings.update(wavelength=_PIPE_WAVELENGTH, n_modes=_PIPE_MODES, n_eff_guess=_PIPE_GUESS)
    return ModelSpec(
        title='Self-test: em_modes circular PEC pipe (conformal)',
        domain=DomainSpec(size_x=_CAVITY_SIZE, size_y=_CAVITY_SIZE),
        geometry=[GeometrySpec('box', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _CAVITY_SIZE, 'y1': _CAVITY_SIZE}),
                  GeometrySpec('disk', 'circle', {'cx': cx, 'cy': cy, 'r': _CAVITY_RADIUS}),
                  GeometrySpec('metal', 'difference', {'base': 'box', 'subtract': ['disk']})],
        physics=[PhysicsSpec(_PIPE, 'em_modes', settings=settings, features=[
            FeatureSpec('metal', 'pec_domain', ['metal']), FeatureSpec(_PIPE_TABLE, 'mode_table')])],
        mesh=MeshSpec(mode='uniform', dx=_CAVITY_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu', conformal_boundaries=True),
        study=StudySpec(type='mode_analysis'))


def _pipe_errors(results):
    """{TE11, TM01: relative error of the nearest k_c R = k0 R sqrt(1 - n_eff^2) to the Bessel zero} and the k_c R."""
    rows = results.result(_PIPE, _PIPE_TABLE)['rows']
    cutoffs = [_PIPE_K0_RADIUS * math.sqrt(max(0.0, 1.0 - float(row[_PIPE_N_EFF_COLUMN]) ** 2)) for row in rows]
    errors = {mode: min((abs(value / _BESSEL_ZERO[component] - 1.0) for value in cutoffs), default=math.inf)
              for mode, component in (('TE11', 'hz'), ('TM01', 'ez'))}
    return errors, cutoffs


def _check_pipe(results_by_name):
    """Conformal TE11 / TM01 cutoffs k_c R within tolerance of 1.8412 / 2.4048."""
    errors, cutoffs = _pipe_errors(results_by_name['on'])
    return _failed_count_result({mode: (errors[mode], _PIPE_TOLERANCE[mode]) for mode in _PIPE_TOLERANCE},
                                {'conformal_errors': errors, 'conformal_cutoffs_kc_r': cutoffs})


CONFORMAL_CUT_SELFTEST_CASES = (
    SelfTestCase(
        name='conformal_outline_geometry',
        label='Cut cells: outline pieces and dual-cell cuts',
        description='CPU (core.cut_cells): outline_pieces of the 23-degree half-plane on 10 x 10 unit cells (window '
                    'chord length, owner-centre offsets and normals exact to 1e-8, owners in the region), of a '
                    'face-aligned rectangle (perimeter and half-cell offsets exact: face pieces) and of a disc of '
                    f'radius 15.37 cells on 40 x 40 cells (2 pi R within {_DISC_OUTLINE_TOLERANCE:g}: refined '
                    'chords); dual_cell_cuts of the half-plane (link fractions between centres within '
                    f'{_FRACTION_TOLERANCE:g}, dual-square free areas within {_AREA_TOLERANCE:g}). error = number of '
                    'failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_outline_geometry),
    SelfTestCase(
        name='conformal_heat_outline',
        label='Cut cells: heat flux and convection on a drawn curved edge',
        description='heat (k = 1, steady, exact CG) on 10 nm cells: a hole of radius 8.37 cells (no material: an '
                    'insulated cut-cell wall) inside a conductor held at 300 K beyond radius 20.61 cells, centre off '
                    f'the cell centres; heat_flux {_FLUX:g} W/m^2 or convection (h a / k = 1, T_ext {_T_FLUID:g} K) '
                    'on the perimeter of the object outside the hole, conformal_boundaries on (outline pieces, '
                    'h L / (1 + h delta / k)): the cells a < r < b against T_b + C ln(b / r), relative to '
                    f'the rise, in bands: the wall ring ({_WALL_RING_CELLS:g} cells) within {_RING_TOLERANCE_CELLS:g} '
                    'cells of the wall gradient 1 / (a ln(b / a)) (first order there: cut-cell unknowns at the cell '
                    'centres, orphaned outline pieces feeding their open neighbour; measured 0.11 vs 0.04-0.06 '
                    f'staircased), the near band to {_FAR_CELLS:g} cells reported (the dipole tails of flux entering '
                    f'a cell off the wall: measured 0.026), the far band within {_OUTLINE_HEAT_TOLERANCE:g} '
                    '(estimated, not measured: the exact total flux sets it; its fitted flux ratio in the details; '
                    'staircase 1.8e-2 / 2.6e-2). error = number of failed sub-checks, tolerance 0.',
        specs=_outline_specs,
        check=_check_outline_heat),
    SelfTestCase(
        name='conformal_heat_k_switch',
        label='Cut cells: heated ring inside a drawn insulated wall, static and following k(T)',
        description='heat (k = 1, steady, exact CG) on 10 nm cells, conformal_boundaries on. insulated: conductor and '
                    'Q = 1e15 W/m^3 only inside a circle of radius 20.61 cells (no material beyond: k = 0, an '
                    'insulated curved wall), a disc of radius 8.37 cells held at 350 K, centre off the cell centres '
                    '(held edge at its drawn position, fractional face apertures and merged cut-cell volumes at the '
                    'insulated wall, core.cut_cells). switch: that ring plus a skin out to radius 22.3 cells with k = '
                    f'where(T < {_SWITCH_TEMPERATURE:g}, 1, 0): the first stationary iteration conducts through the '
                    'cold skin (walls located at its outer circle), the hot skin then stops conducting and the walls '
                    'are re-located to the ring circle. Both: the cells a < r < b against T = T_a + Q (a^2 - r^2) / '
                    f'(4 k) + Q b^2 ln(r / a) / (2 k), relative to the rise, within {_INSULATED_TOLERANCE:g} (measured '
                    '4.2e-3 for both; staircase 5.7e-2). error = number of failed sub-checks, tolerance 0. Not yet '
                    'confirmed by a run as one case.',
        specs=_switch_specs,
        check=_check_switch),
    SelfTestCase(
        name='conformal_pec_modes',
        label='Conformal PEC: em_modes circular metal pipe',
        description='em_modes (cpu backend, mode_analysis) of a vacuum disk of radius 15.37 cells of 50 nm (centre '
                    '(0.31, 0.17) cells off a cell centre) cut out of a pec_domain block, k0 R = 3.5, '
                    f'{_PIPE_MODES} modes, conformal_boundaries on (Dey-Mittra dual-cell cuts): k_c R = k0 R sqrt(1 - '
                    f'n_eff^2) of TE11 / TM01 within {_PIPE_TOLERANCE["TE11"]:g} / {_PIPE_TOLERANCE["TM01"]:g} of '
                    '1.8412 / 2.4048 (estimated, not measured; staircase 4.4e-2 / 2.4e-2). error = number of failed '
                    'sub-checks, tolerance 0. 3 modes not yet confirmed by a run.',
        specs=lambda: {'on': _pipe_spec()},
        check=_check_pipe),
)
