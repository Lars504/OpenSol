"""Self-tests and templates of the NLFD-chi nonlinear sheets (nonlinear_optics_plan stage 4, self-test 9): the CPU site
check (tensors, sampling / deposit adjoint, power identity, smoothed disk, sheet transpose and absorption), the GPU
surface dipole layers (normal incidence exact, grazing on two meshes), the smoothed disk SHG vs the continuum integral,
the 2D-material sheet SHG / THG against exact lattice references, and the gold_ridge / graphene templates."""
import cmath
import math

import numpy as np

from ...core.constants import C0, EPS0, ETA0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import PERIODIC_Y, probe_phasor, relative_error
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, StudySpec, SweepSpec)
from .selftests import (_BACKWARD_CELL, _FORWARD_GAP, _PHYSICS, _PUMP_AMPLITUDE, _PUMP_OMEGA, _PUMP_WAVELENGTH,
                        _SOURCE_CELL, _STRIP_CELL, _STRIP_ROWS, _cell_wavenumber, _channel_probe, _channel_strip_spec,
                        _normalized_result, _pair, _random_complex, _result, _strip_centre, _strip_point)

_ROUND_OFF = 1e-12
_LENGTH_TOLERANCE = 1e-2        # smoothed disk perimeter vs 2 pi R (estimate: O(h^1.5) end terms, R = 20.3 cells)
_NORMAL_TOLERANCE = 1e-9        # rad, smoothed normal vs the exact one (bisections to 2^-40, exact for a circle)
_STAIRCASE_EXCESS = (1.2, 1.35)   # staircase perimeter / 2 pi R (the mean excess is 4 / pi)
_EXACT_TOLERANCE = 1e-9         # relative: the dipole layer needs no propagation (solver round-off only)
_LATTICE_TOLERANCE = 2e-3       # relative complex phasor, as the bulk strip cases (PML floor not measured yet)
_SURFACE_START = 200
_SURFACE_SEGMENTS = {'inside': (200, 220), 'outside': (260, 280)}   # selvedge -> surface_chi2 strip cells [a, b)
_SURFACE_CHI = {'chi_s_nnn': 3e-18, 'chi_s_ntt': 1e-18, 'chi_s_tnt': 2e-18}   # m^2/V (nnn / tnt need E_n: 0 here)
_YEE_CELL = float(np.float32(_STRIP_CELL))   # the solver's float32 grid width (deposits divide by it)
_SHEET_CELL = 200               # D3h line sheet column of the 2D-material strip
_GRAPHENE_CELL = 240            # graphene sigma3 line sheet column (run ez_pump, beside the D3h sheet)
_TMD_CHI = 2e-19                # m^2/V
_GRAPHENE_THG_SIGMA = complex(1e-21, -4e-22)   # S m^2/V^2
_GRAPHENE_KERR_SIGMA = complex(3e-20, 1e-20)   # must not act undepleted (SPM / XPM only on the pump)


# ---- CPU: sites, tensors, adjoint, power identity, smoothed disk ----
def _surface_closed_form(a, b, d, angles):
    """(n, 3, 3, 3) a nnn + b (ntt + nzz) + d (tnt + ttn + znz + zzn) at in-plane normals of the given angles."""
    zeros = np.zeros_like(angles)
    normal = np.stack((np.cos(angles), np.sin(angles), zeros), axis=1)
    tangent = np.stack((-np.sin(angles), np.cos(angles), zeros), axis=1)
    out = np.zeros_like(normal)
    out[:, 2] = 1.0

    def outer(i, j, k):
        return np.einsum('si,sj,sk->sijk', i, j, k)
    return (a * outer(normal, normal, normal) + b * (outer(normal, tangent, tangent) + outer(normal, out, out))
            + d * (outer(tangent, normal, tangent) + outer(tangent, tangent, normal) + outer(out, normal, out)
                   + outer(out, out, normal)))


def _disk(widths, centre, radius):
    """(inside by the cell-centre test, point test (x, y) -> bool) of a closed disk on a grid of Yee widths from 0."""
    def test(x, y):
        return (np.asarray(x) - centre[0]) ** 2 + (np.asarray(y) - centre[1]) ** 2 <= radius ** 2
    edges = [np.concatenate(([0.0], np.cumsum(width))) for width in widths]
    centres = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
    return test(centres[0][:, None], centres[1][None, :]), test


def _adjoint_error(sheet, widths, rng):
    """|sum_k A_k conj(E_k) P_k - sum_s L_s conj(E_s) . P_s| / scale for random grid fields and site P_s."""
    from .sheets import sheet_grid_density, site_fields
    nx, ny = widths[0].size, widths[1].size
    dual = [0.5 * (width + np.roll(width, -1)) for width in widths]
    areas = (np.outer(widths[0], dual[1]), np.outer(dual[0], widths[1]), np.outer(widths[0], widths[1]))
    grid = tuple(_random_complex(rng, (nx, ny)) for _ in range(3))
    sites = _random_complex(rng, (sheet.n_sites, 3))
    density = sheet_grid_density({'sheet': sites}, {'sheet': sheet}, (0, 1, 2), (nx, ny), np)
    grid_terms = [areas[component] * np.conj(grid[component]) * density[component] for component in range(3)]
    site_terms = sheet.site_length[:, None] * np.conj(site_fields(sheet, grid, np)) * sites
    scale = sum(np.sum(np.abs(term)) for term in grid_terms) + np.sum(np.abs(site_terms))
    return float(abs(sum(np.sum(term) for term in grid_terms) - np.sum(site_terms)) / scale)


def _check_sheet_sites(results):
    from .mixing import ChannelSpec, build_mixing_table, resolve_channels
    from .sheets import (SELVEDGES, line_sheet, rotate_sites, sheet_chi2_local, sheet_cotangents, sheet_grid_density,
                         sheet_sigma3_local, sheet_site_absorption, sheet_site_polarizations, sheet_site_work,
                         surface_frames, surface_local_tensor, surface_sheet)
    violations, details = [], {}
    rng = np.random.default_rng(29)

    # 1. surface tensor in model axes = the closed form; D3h sheet tensor 120 deg periodic
    a, b, d = 1.3, -0.7, 0.4
    angles = rng.uniform(0.0, 2.0 * math.pi, 7)
    expected = _surface_closed_form(a, b, d, angles)
    rotated = rotate_sites(surface_local_tensor(a, b, d), surface_frames(np.cos(angles), np.sin(angles)))
    tensor_error = float(np.max(np.abs(rotated - expected)) / np.max(np.abs(expected)))
    d3h_error = float(np.max(np.abs(sheet_chi2_local('D3h', {'chi_s': 1.0}, 10.0)
                                    - sheet_chi2_local('D3h', {'chi_s': 1.0}, 130.0))))
    details.update(surface_tensor_error=tensor_error, d3h_period_error=d3h_error)
    if max(tensor_error, d3h_error) > _ROUND_OFF:
        violations.append(f'tensors: surface {tensor_error:.3g}, D3h 120 deg period {d3h_error:.3g}')

    # 2. deposit = adjoint of the sampling (graded grid, a disk, both selvedges, staircase / smoothed, periodic or not)
    nx, ny = 12, 10
    widths = (rng.uniform(0.6, 1.4, nx) * 1e-8, rng.uniform(0.6, 1.4, ny) * 1e-8)
    disk_inside, disk_test = _disk(widths, (0.55 * np.sum(widths[0]), 0.48 * np.sum(widths[1])),
                                   0.3 * np.sum(widths[1]))
    local = surface_local_tensor(a, b, d)
    adjoint_errors = {}
    for periodic in ((False, False), (True, True)):
        for selvedge in SELVEDGES:
            for label, test in (('staircase', None), ('smoothed', disk_test)):
                sheet = surface_sheet('surface', local, disk_inside, None, widths, periodic, selvedge, test)
                adjoint_errors[f'surface {label} {selvedge} periodic {periodic[0]}'] = _adjoint_error(sheet, widths,
                                                                                                    rng)
    d3h = sheet_chi2_local('D3h', {'chi_s': 0.8}, 25.0)
    for axis, line_cell in (('x', 4), ('y', 6)):
        sheet = line_sheet('line', 2, d3h, axis, line_cell, slice(1, 9), (nx, ny), widths)
        adjoint_errors[f'line {axis}'] = _adjoint_error(sheet, widths, rng)
    worst_adjoint = max(adjoint_errors.values())
    details['adjoint_errors'] = adjoint_errors
    if worst_adjoint > _ROUND_OFF:
        violations.append(f'deposit is not the adjoint of the sampling: {adjoint_errors}')

    # 3. power exchange identity on random fields: Kleinman surface (ntt = tnt) and a D3h line sheet, SHG both
    channels, _ = resolve_channels((ChannelSpec('p', 'both', wavelength=1.55e-6),
                                    ChannelSpec('sh', 'both', combination='2 p')))
    sheets = {'surface': surface_sheet('surface', surface_local_tensor(1.1, 0.6, 0.6), disk_inside, None, widths,
                                       (False, False), 'outside', disk_test),
              'tmd': line_sheet('tmd', 2, d3h, 'y', 5, slice(0, ny), (nx, ny), widths)}
    table = build_mixing_table(channels, [sheet.mixing_tensor() for sheet in sheets.values()])
    grids = {index: ((None, None, _random_complex(rng, (nx, ny))) if channel.polarization == 'ez'
                     else (_random_complex(rng, (nx, ny)), _random_complex(rng, (nx, ny)), None))
             for index, channel in enumerate(channels)}
    exchange = []
    for target, channel in enumerate(channels):
        terms = table.terms_for(target)
        if terms:
            sites = sheet_site_polarizations(terms, grids, sheets, channel.components, channel.omega, np)
            exchange.append(sheet_site_work(sites, sheets, grids[target], channel.omega, np))
    total = sum(abs(power) for power in exchange)
    imbalance = abs(sum(exchange)) / total if total > 0.0 else math.inf
    details.update(power_imbalance=imbalance, n_terms=len(table.terms))
    if imbalance > _ROUND_OFF:
        violations.append(f'power exchange imbalance {imbalance:.3g} (terms {len(table.terms)})')

    # 5. sheet_cotangents = the transpose of the deposited density along random tangents, Re(c . dD[V]) = Re(g . V)
    # (the sheets of 3 plus a complex sigma3 Kerr line sheet: conj slots, current response); 6. sheet_site_absorption
    # sums to -site work
    adjoint_sheets = {**sheets, 'kerr': line_sheet('kerr', 3, sheet_sigma3_local(2e-20 + 7e-21j), 'x', 4, slice(1, 9),
                                                   (nx, ny), widths, response='current')}
    adjoint_table = build_mixing_table(channels, [sheet.mixing_tensor() for sheet in adjoint_sheets.values()])
    tangents = {index: tuple(None if part is None else _random_complex(rng, (nx, ny)) for part in grid)
                for index, grid in grids.items()}
    areas = np.outer(widths[0], widths[1])
    transpose_errors, absorption_errors = {}, {}
    for target, channel in enumerate(channels):
        terms = adjoint_table.terms_for(target)
        if not terms:
            continue
        tangent_sites = sheet_site_polarizations(terms, grids, adjoint_sheets, channel.components, channel.omega, np,
                                                 tangents)
        d_density = sheet_grid_density(tangent_sites, adjoint_sheets, channel.components, (nx, ny), np)
        cotangent = {component: _random_complex(rng, (nx, ny)) for component in channel.components}
        transposed = sheet_cotangents(terms, grids, adjoint_sheets, channel.components, channel.omega, cotangent,
                                      (nx, ny), np)
        products = [cotangent[component] * d_density[component] for component in channel.components]
        forward = sum(np.sum(product) for product in products).real
        backward = sum(np.sum(parts[component] * tangents[index][component]) for index, parts in transposed.items()
                       for component in range(3) if parts[component] is not None).real
        transpose_errors[channel.name] = abs(forward - backward) / sum(float(np.sum(np.abs(product)))
                                                                       for product in products)
        sites = sheet_site_polarizations(terms, grids, adjoint_sheets, channel.components, channel.omega, np)
        absorbed = float(np.sum(areas * sheet_site_absorption(sites, adjoint_sheets, grids[target], channel.omega,
                                                              areas, np)))
        work = sheet_site_work(sites, adjoint_sheets, grids[target], channel.omega, np)
        absorption_errors[channel.name] = abs(absorbed + work) / abs(work)
    worst_transpose = max(transpose_errors.values(), default=math.inf)
    worst_absorption = max(absorption_errors.values(), default=math.inf)
    details.update(sheet_transpose_errors=transpose_errors, sheet_absorption_errors=absorption_errors)
    if worst_transpose > _ROUND_OFF:
        violations.append(f'sheet_cotangents is not the transpose of the sheet density: {transpose_errors}')
    if worst_absorption > _ROUND_OFF:
        violations.append(f'sheet_site_absorption does not sum to -site work: {absorption_errors}')

    # 4. a disk of radius 20.3 cells: staircase length ~4 / pi too long; smoothed normals = the exact radius at each
    # face's crossing (the circle on the line through the face midpoint along its axis), smoothed length
    cells = 64
    uniform = (np.full(cells, 1e-8), np.full(cells, 1e-8))
    centre, radius = (32.1e-8, 31.7e-8), 20.3e-8
    inside, test = _disk(uniform, centre, radius)
    perimeter = 2.0 * math.pi * radius
    staircase = surface_sheet('disk', local, inside, None, uniform, (False, False))
    smoothed = surface_sheet('disk', local, inside, None, uniform, (False, False), boundary_test=test)
    x_faces = np.abs(staircase.normals[:, 0]) > 0.5   # same site order in both sheets
    offset = smoothed.positions - np.asarray(centre)
    across, along = np.where(x_faces, offset[:, 0], offset[:, 1]), np.where(x_faces, offset[:, 1], offset[:, 0])
    crossing = np.sign(across) * np.sqrt(radius ** 2 - along ** 2)
    exact = np.stack((np.where(x_faces, crossing, along), np.where(x_faces, along, crossing)), axis=1) / radius
    angle_error = np.abs(np.arctan2(exact[:, 0] * smoothed.normals[:, 1] - exact[:, 1] * smoothed.normals[:, 0],
                                    np.sum(exact * smoothed.normals, axis=1)))
    staircase_ratio = staircase.total_length / perimeter
    smoothed_error = abs(smoothed.total_length / perimeter - 1.0)
    details.update(staircase_length_ratio=staircase_ratio, smoothed_length_error=smoothed_error,
                   normal_angle_max=float(angle_error.max()), n_sites=smoothed.n_sites)
    if not _STAIRCASE_EXCESS[0] <= staircase_ratio <= _STAIRCASE_EXCESS[1]:
        violations.append(f'staircase perimeter ratio {staircase_ratio:.4f} outside {_STAIRCASE_EXCESS}')
    if smoothed_error > _LENGTH_TOLERANCE:
        violations.append(f'smoothed perimeter off by {smoothed_error:.3g}')
    if angle_error.max() > _NORMAL_TOLERANCE:
        violations.append(f'smoothed normals off the exact radius by up to {angle_error.max():.3g} rad')
    return _result(violations, details, measured=max(worst_adjoint, imbalance, worst_transpose, worst_absorption),
                   tolerance=_ROUND_OFF)


# ---- GPU 9 (a / b): the surface dipole layer ----
# y-periodic vacuum strip (as the bulk cases), pump p 'both' (E_z and E_y line sources in cell 100), SH sh = 2 p as
# H_z channel, surface_chi2 selvedge inside on the cells [200, 220) and selvedge outside on [260, 280) over the full
# height (one run): the sites of a segment [a, b) are the x-normal faces a-1|a (n = -x) and b-1|b (n = +x). At normal incidence E_n = E_x = 0, so only chi_s_ntt acts: P_s,n = (1/2) eps0
# chi_s_ntt (E_z^2 + E_t^2) with E_z = the mean of the two cells and E_t = E_y on the face; the hz deposit is P_x =
# n_x P_s,n / w on the E_x faces of the selvedge cell, which radiates nothing (y-uniform: the H_z rhs is 0), so
# exactly E_x = -P_x / eps0 there and 0 elsewhere. E_y on a face = (p_c + p_(c+1)) / (1 + cos K h) of the centre probes
# (each the mean of its two faces), exact with the CPML-reflected pump too (the harmonic mean 2 p_c p_(c+1) / (p_c +
# p_(c+1)) of a pure forward wave is off by ~r (K h)^2: 1.2e-8 > the tolerance at r ~ 1e-5).
def _surface_spec():
    """One strip with both selvedges: surface_chi2 'inside' on _SURFACE_SEGMENTS['inside'], 'outside' on its own
    segment (the dipole layers do not radiate, so the segments do not interact)."""
    surface_end = max(end for _, end in _SURFACE_SEGMENTS.values())
    medium = MaterialSpec('medium', properties={'eps_r': 1.0}, selection=['all'])
    channels = [('p', 'both', {'definition': 'wavelength', 'wavelength': _PUMP_WAVELENGTH}),
                ('sh', 'hz', {'definition': 'combination', 'combination': '2 p'})]
    face_cells = [cell for start, end in _SURFACE_SEGMENTS.values() for cell in (start - 1, start, end - 1, end)]
    probes = [(f'ez_{cell}', 'p_ez', 'ez', cell) for cell in face_cells]
    probes += [(f'ey_{cell}', 'p_hz', 'ey', cell) for cell in face_cells]
    probes += [(f'sh_ex_{cell}', 'sh', 'ex', cell) for cell in face_cells]
    probes.append(('sh_ey_forward', 'sh', 'ey', surface_end + _FORWARD_GAP))
    spec = _channel_strip_spec('surface chi2 dipole layers (inside / outside)', medium, surface_end - _SURFACE_START,
                               ('surface_chi2', {**_SURFACE_CHI, 'selvedge': 'inside'}), channels,
                               [('pump_ez', 'p_ez', _PUMP_AMPLITUDE), ('pump_hz', 'p_hz', _PUMP_AMPLITUDE)], probes)
    size_y = _STRIP_ROWS * _STRIP_CELL
    spec.geometry = [geometry for geometry in spec.geometry if geometry.name != 'nl_slab'] + [
        GeometrySpec(f'surface_{selvedge}', 'rectangle', {'x0': start * _STRIP_CELL, 'y0': 0.0,
                                                          'x1': end * _STRIP_CELL, 'y1': size_y})
        for selvedge, (start, end) in _SURFACE_SEGMENTS.items()]
    physics = spec.physics[0]
    physics.features = [feature for feature in physics.features if feature.name != 'nonlinear'] + [
        FeatureSpec(f'surface_{selvedge}', 'surface_chi2', [f'surface_{selvedge}'],
                    {**_SURFACE_CHI, 'selvedge': selvedge}) for selvedge in _SURFACE_SEGMENTS]
    return spec


def _surface_specs():
    return {'run': _surface_spec()}


def _face_value(lower, upper):
    """E on the face between two cells from their centre probes (face means) of a pure geometric wave."""
    return 2.0 * lower * upper / (lower + upper)


def _lattice_face_value(lower, upper, omega):
    """Face E between two vacuum cells from their centre probes (face means), exact for any forward + backward wave of
    the 3-point lattice: F_c = (a + b) / (1 + cos K h), 1 + cos K h = 2 - (k h)^2 / 2."""
    return (lower + upper) / (2.0 - 0.5 * (omega * _YEE_CELL / C0) ** 2)


def _check_surface(results_by_name):
    results = results_by_name['run']
    errors, details, tolerances = {}, {}, {}
    scale = 0.0

    def probe(name):
        return probe_phasor(results, _PHYSICS, name)
    for selvedge, (start, end) in _SURFACE_SEGMENTS.items():
        faces = {'left': (start - 1, start, -1.0), 'right': (end - 1, end, 1.0)}   # (lower, upper cell, n_x)
        for side, (lower, upper, normal_x) in faces.items():
            ez = 0.5 * (probe(f'ez_{lower}') + probe(f'ez_{upper}'))
            ey = _lattice_face_value(probe(f'ey_{lower}'), probe(f'ey_{upper}'), _PUMP_OMEGA)
            expected = -normal_x * 0.5 * _SURFACE_CHI['chi_s_ntt'] * (ez ** 2 + ey ** 2) / _YEE_CELL
            inner, outer = (upper, lower) if normal_x < 0 else (lower, upper)
            carrying, empty = (inner, outer) if selvedge == 'inside' else (outer, inner)
            measured = probe(f'sh_ex_{carrying}')
            scale = max(scale, abs(expected))
            errors[f'{selvedge} {side} E_x'] = relative_error(measured, expected)
            errors[f'{selvedge} {side} E_x beside'] = abs(probe(f'sh_ex_{empty}')) / abs(expected)
            details.update({f'{selvedge}_{side}_measured': _pair(measured), f'{selvedge}_{side}_expected': _pair(expected)})
    errors['SH E_y forward'] = abs(probe('sh_ey_forward')) / scale
    tolerances.update({item: _EXACT_TOLERANCE for item in errors})
    return _normalized_result(errors, tolerances, details)


# ---- GPU 9 (b): the normal dipole layer at grazing incidence on two meshes ----
# y-periodic vacuum strip (period 200 nm, CPML along x), H_z pump p (line source, E_y along the full height) and SH
# sh = 2 p, surface_chi2 chi_s_nnn only on the band y 60-120 nm, x 1-2.5 um: the pump E_y is the NORMAL field of the
# band's y-normal faces, so P_s,n = (1/2) eps0 chi_s_nnn E_n^2 (selvedge inside: E_n = the centre E_y of the cell below
# the interface y0 = 120 nm, n = +y) forms a dipole layer P_y = P_s,n / h in that row. The band's two layers are
# opposite, so no k_y = 0 wave is driven (the period is below the SH wavelength: the rest is evanescent), and the
# tangential E_x jumps across the row by -D_x P_s,n / eps0 = j (sin(2 K_w h) / h) P_s,n / eps0 (the central difference
# of the face-mean deposit; K_w h = 2 asin(k h / 2)) plus at most O(h) from the smooth fields. Measured: E_x(y0 + h/2)
# - E_x(y0 - 3h/2) of the cell-centre probes (the C means of the faces), symmetric about the deposit row; meshes 10 and
# 5 nm (probe column x = 1.7525 um inside one cell of both).
_JUMP_PERIOD = 200e-9
_JUMP_BAND = (60e-9, 120e-9)        # y range of the selection (upper interface y0, n = +y)
_JUMP_X = (1.0e-6, 2.5e-6)          # x range of the selection
_JUMP_SIZE_X = 3e-6
_JUMP_SOURCE_X = 0.5025e-6
_JUMP_PROBE_X = 1.7525e-6
_JUMP_MESHES = {'coarse': 10e-9, 'fine': 5e-9}
_JUMP_CHI = 1e-18                   # m^2/V, chi_s_nnn
_JUMP_AMPLITUDE = 1e8               # V/m
_RICHARDSON_TOLERANCE = 2e-2        # relative, 2 r(h/2) - r(h) vs 1 (estimate: O(h^2) of the evanescent near field)
_ORDER_RANGE = (0.6, 1.6)           # observed convergence order log2(e(h) / e(h/2)) of the first-order scheme
# Below this |r(h) - 1| on both meshes the order is not judged: the probes sit symmetric about the deposit row and the
# grazing SH is nearly phase matched (evanescent decay ~0), so the O(h) terms cancel; first run: 2.3e-5 / 1.6e-5
_ORDER_FLOOR = 1e-3


def _jump_spec(label, cell):
    y_interface = _JUMP_BAND[1]
    points = {'pump_point': y_interface - 0.5 * cell, 'above_point': y_interface + 0.5 * cell,
              'below_point': y_interface - 1.5 * cell}
    geometry = [GeometrySpec('source_line', 'line', {'x0': _JUMP_SOURCE_X, 'y0': 0.0, 'x1': _JUMP_SOURCE_X,
                                                     'y1': _JUMP_PERIOD}),
                GeometrySpec('band', 'rectangle', {'x0': _JUMP_X[0], 'y0': _JUMP_BAND[0], 'x1': _JUMP_X[1],
                                                   'y1': y_interface}),
                *(GeometrySpec(name, 'point', {'x': _JUMP_PROBE_X, 'y': y}) for name, y in points.items())]
    features = [
        FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': _PUMP_WAVELENGTH,
                                                  'polarization': 'hz'}),
        FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                   'polarization': 'hz'}),
        FeatureSpec('surface', 'surface_chi2', ['band'], {'chi_s_nnn': _JUMP_CHI, 'chi_s_ntt': 0.0, 'chi_s_tnt': 0.0,
                                                          'selvedge': 'inside', 'interface': 'staircase'}),
        FeatureSpec('pump', 'line_source', ['source_line'], {'channel': 'p', 'amplitude': _JUMP_AMPLITUDE,
                                                             'phase': 0.0}),
        _channel_probe('pump_normal', 'p', 'ey', 'pump_point'),
        _channel_probe('sh_above', 'sh', 'ex', 'above_point'),
        _channel_probe('sh_below', 'sh', 'ex', 'below_point')]
    return ModelSpec(
        title=f'Self-test: em_fdfd_nonlinear surface chi2 normal dipole layer ({label}, {cell * 1e9:g} nm)',
        domain=DomainSpec(size_x=_JUMP_SIZE_X, size_y=_JUMP_PERIOD), geometry=geometry,
        materials=[MaterialSpec('vacuum', properties={'eps_r': 1.0}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings=dict(PERIODIC_Y), features=features)],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='nonlinear_frequency_domain'))


def _jump_specs():
    return {label: _jump_spec(label, cell) for label, cell in _JUMP_MESHES.items()}


def _check_jump(results_by_name):
    ratios, details = {}, {}
    for label, cell in _JUMP_MESHES.items():
        results = results_by_name[label]
        normal = probe_phasor(results, _PHYSICS, 'pump_normal')
        pump_k = 2.0 * math.asin(0.5 * _PUMP_OMEGA * cell / C0)
        expected = 1j * (math.sin(2.0 * pump_k) / cell) * 0.5 * _JUMP_CHI * normal ** 2
        measured = probe_phasor(results, _PHYSICS, 'sh_above') - probe_phasor(results, _PHYSICS, 'sh_below')
        ratios[label] = measured / expected
        details.update({f'{label}_jump_measured': _pair(measured), f'{label}_jump_expected': _pair(expected),
                        f'{label}_ratio': _pair(ratios[label]), f'{label}_pump_normal': _pair(normal)})
    deviations = {label: abs(ratio - 1.0) for label, ratio in ratios.items()}
    extrapolated = 2.0 * ratios['fine'] - ratios['coarse']
    order = (math.log2(deviations['coarse'] / deviations['fine'])
             if deviations['coarse'] > 0.0 and deviations['fine'] > 0.0 else math.nan)
    in_range = (_ORDER_RANGE[0] <= order <= _ORDER_RANGE[1]
                or all(deviation < _ORDER_FLOOR for deviation in deviations.values()))   # NaN fails both
    details.update(deviations=deviations, richardson=_pair(extrapolated), observed_order=order)
    errors = {'Richardson-extrapolated jump': abs(extrapolated - 1.0),
              'convergence order outside range': 0.0 if in_range else math.inf}
    return _normalized_result(errors, {'Richardson-extrapolated jump': _RICHARDSON_TOLERANCE,
                                       'convergence order outside range': 1.0}, details)


# ---- GPU: smoothed-surface disk SHG against the continuum surface integral ----
# A vacuum disk (radius 200 nm, off-grid centre) with surface_chi2 (nnn, ntt, tnt) in an all-CPML vacuum box, H_z
# plane wave (TFSF, E_y = E0 e^{-j k (x - x_c)}) at 1550 nm; the disk has eps 1, so the pump is unperturbed and the
# continuum reference is exact: P_s,n = (1/2) eps0 (chi_nnn E_n^2 + chi_ntt E_t^2), P_s,t = eps0 chi_tnt E_n E_t on the
# circle (E_n = E0 sin phi, E_t = E0 cos phi times the pump phase), radiating in vacuum at 2 w: dP / dtheta = eta0 k2
# w2^2 / (16 pi) |F(theta)|^2, F = a int (r_hat x P_s)_z e^{j k2 a cos(theta - phi)} dphi (2D far field of an in-plane
# polarisation), P_sh = int dP / dtheta (trapezoid quadrature, spectrally exact). Measured: power_exchange_sh (the
# work of the sites = the radiated SH power, lossless). Runs: smoothed at 10 and 5 nm (R = 20 / 40 cells); the
# staircase perimeter excess (4 / pi) is pinned on the CPU by nlfd_sheet_sites (4). Box 1.2 um (disk edge 0.4 um from
# the CPML): only the CPML back-reflection onto the disk enters the site work.
_DISK_SIZE = 1.2e-6                 # disk edge ~0.4 um (~0.5 SH wavelength) from the box edge (CPML back-reflection)
_DISK_CENTRE = (0.6031e-6, 0.5987e-6)
_DISK_RADIUS = 0.2e-6
_DISK_TFSF = (0.25e-6, 0.95e-6)
_DISK_CHI = {'chi_s_nnn': 1e-18, 'chi_s_ntt': 2e-19, 'chi_s_tnt': 3e-19}   # m^2/V
_DISK_AMPLITUDE = 1e8               # V/m
_DISK_RUNS = {'smoothed_coarse': ('smoothed', 10e-9), 'smoothed_fine': ('smoothed', 5e-9)}
_DISK_TOLERANCE = 0.1               # relative SH power, smoothed runs (estimate: first order by the half-cell selvedge)
_DISK_POWER = 'P_sh'
_DISK_QUADRATURE = (2048, 1024)     # (circle points, far-field angles)


def _disk_spec(interface, cell):
    x_centre, y_centre = _DISK_CENTRE
    low, high = _DISK_TFSF
    features = [
        FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': _PUMP_WAVELENGTH,
                                                  'polarization': 'hz'}),
        FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                   'polarization': 'hz'}),
        FeatureSpec('surface', 'surface_chi2', ['disk'], {**_DISK_CHI, 'selvedge': 'inside', 'interface': interface}),
        FeatureSpec('pump', 'plane_wave_tfsf', ['tfsf_box'], {'channel': 'p', 'amplitude': _DISK_AMPLITUDE,
                                                              'phase': 0.0}),
        _channel_probe('pump_centre', 'p', 'ey', 'centre_point')]
    return ModelSpec(
        title=f'Self-test: em_fdfd_nonlinear surface chi2 disk SHG ({interface}, {cell * 1e9:g} nm)',
        domain=DomainSpec(size_x=_DISK_SIZE, size_y=_DISK_SIZE),
        geometry=[GeometrySpec('disk', 'circle', {'cx': x_centre, 'cy': y_centre, 'r': _DISK_RADIUS}),
                  GeometrySpec('tfsf_box', 'rectangle', {'x0': low, 'y0': low, 'x1': high, 'y1': high}),
                  GeometrySpec('centre_point', 'point', {'x': x_centre, 'y': y_centre})],
        materials=[MaterialSpec('vacuum', properties={'eps_r': 1.0}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', features=features)],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='nonlinear_frequency_domain'),
        derived_values=[DerivedValueSpec(_DISK_POWER, 'nl_power_exchange_sh', [], 'mean')])


def _disk_specs():
    return {run: _disk_spec(interface, cell) for run, (interface, cell) in _DISK_RUNS.items()}


def _disk_reference_power(amplitude):
    """Continuum SH power [W/m] radiated by the surface sheet of the vacuum disk at pump amplitude |E0| [V/m]."""
    k_pump = _PUMP_OMEGA / C0
    k_sh, omega_sh = 2.0 * k_pump, 2.0 * _PUMP_OMEGA
    n_phi, n_theta = _DISK_QUADRATURE
    phi = (np.arange(n_phi) + 0.5) * (2.0 * math.pi / n_phi)
    phase = amplitude * np.exp(-1j * k_pump * _DISK_RADIUS * np.cos(phi))
    e_normal, e_tangent = np.sin(phi) * phase, np.cos(phi) * phase
    p_normal = 0.5 * EPS0 * (_DISK_CHI['chi_s_nnn'] * e_normal ** 2 + _DISK_CHI['chi_s_ntt'] * e_tangent ** 2)
    p_tangent = EPS0 * _DISK_CHI['chi_s_tnt'] * e_normal * e_tangent
    p_x = p_normal * np.cos(phi) - p_tangent * np.sin(phi)
    p_y = p_normal * np.sin(phi) + p_tangent * np.cos(phi)
    theta = (np.arange(n_theta) + 0.5) * (2.0 * math.pi / n_theta)
    cross = np.cos(theta)[:, None] * p_y[None, :] - np.sin(theta)[:, None] * p_x[None, :]
    far = (_DISK_RADIUS * 2.0 * math.pi / n_phi) * np.sum(
        cross * np.exp(1j * k_sh * _DISK_RADIUS * np.cos(theta[:, None] - phi[None, :])), axis=1)
    return float(np.sum(np.abs(far) ** 2) * (2.0 * math.pi / n_theta) * ETA0 * k_sh * omega_sh ** 2 / (16.0 * math.pi))


def _check_disk(results_by_name):
    deviations, details = {}, {}
    for run in _DISK_RUNS:
        results = results_by_name[run]
        amplitude = abs(probe_phasor(results, _PHYSICS, 'pump_centre'))
        reference = _disk_reference_power(amplitude)
        measured = float(results.summary()['derived'][_DISK_POWER])
        deviations[run] = abs(measured / reference - 1.0)
        details.update({f'{run}_P_sh': measured, f'{run}_P_reference': reference, f'{run}_pump_amplitude': amplitude})
    coarse, fine = deviations['smoothed_coarse'], deviations['smoothed_fine']
    converging = fine <= coarse or coarse < 0.2 * _DISK_TOLERANCE
    details.update(relative_deviations=deviations)
    errors = {'smoothed 10 nm': coarse, 'smoothed 5 nm': fine,
              'smoothed not converging': 0.0 if converging else math.inf}
    return _normalized_result(errors, {'smoothed 10 nm': _DISK_TOLERANCE, 'smoothed 5 nm': _DISK_TOLERANCE,
                                       'smoothed not converging': 1.0}, details)


# ---- GPU 9 (c): 2D-material sheets on a line ----
# The same strip with vertical sheet lines (vacuum; D3h in cell 200, graphene in cell 240 of run ez_pump): each
# harmonic's sheet current or polarisation is a single lattice source, so the harmonic is the 3-point lattice Green's
# function G(i - s) = j h^2 / (2 sin(K h)) e^{-j K h |i - s|} (s = its sheet cell) times -(n w)^2 mu0 P / h (P_s
# over the cell width / face spacing h): E_z channels at the centres, H_z channels on the faces (E_y obeys the same
# 3-point equation on the face lattice; the hz probe is the mean of the cell's two faces). ez_pump D3h: armchair along
# the line, E_z pump -> P_t = -(1/2) eps0 chi_s E_z^2 into an H_z SH; tmd_hz: armchair out of the plane (90 deg), E_y
# pump -> P_z = -(1/2) eps0 chi_s E_t^2 into an E_z SH; ez_pump graphene: sigma3 THG K_z = sigma3 E_z^3 / 4, P = K /
# (j 3 w), with a Kerr sigma3 that must not act (SPM / XPM undepleted).
def _sheet_strip_spec(title, sheets, channels, sources, probes):
    """Vacuum strip with sheet_nonlinearity lines sheets [(feature name, cell, values)]."""
    size_y = _STRIP_ROWS * _STRIP_CELL
    source_x = _strip_centre(_SOURCE_CELL)
    cells = sorted({cell for _, _, _, cell in probes})
    geometry = [GeometrySpec('source_line', 'line', {'x0': source_x, 'y0': 0.0, 'x1': source_x, 'y1': size_y}),
                *(GeometrySpec(f'{name}_line', 'line', {'x0': _strip_centre(cell), 'y0': 0.0,
                                                        'x1': _strip_centre(cell), 'y1': size_y})
                  for name, cell, _ in sheets),
                *(_strip_point(f'point_{cell}', cell) for cell in cells)]
    features = [FeatureSpec(name, 'harmonic_channel', [], {'polarization': polarization, **definition})
                for name, polarization, definition in channels]
    features += [FeatureSpec(name, 'sheet_nonlinearity', [f'{name}_line'], dict(values)) for name, _, values in sheets]
    features += [FeatureSpec(name, 'line_source', ['source_line'], {'channel': channel, 'amplitude': amplitude,
                                                                    'phase': 0.0})
                 for name, channel, amplitude in sources]
    features += [_channel_probe(name, channel, component, f'point_{cell}') for name, channel, component, cell in probes]
    return ModelSpec(
        title=f'Self-test: em_fdfd_nonlinear {title}',
        domain=DomainSpec(size_x=(_SHEET_CELL + 2 * _FORWARD_GAP) * _STRIP_CELL, size_y=size_y),
        geometry=geometry, materials=[MaterialSpec('vacuum', properties={'eps_r': 1.0}, selection=['all'])],
        physics=[PhysicsSpec(_PHYSICS, 'em_fdfd_nonlinear', settings=dict(PERIODIC_Y), features=features)],
        mesh=MeshSpec(mode='uniform', dx=_STRIP_CELL),
        study=StudySpec(type='nonlinear_frequency_domain'))


_FORWARD_PROBE = _SHEET_CELL + _FORWARD_GAP
_HARMONIC_PROBES = {'forward': _FORWARD_PROBE, 'backward': _BACKWARD_CELL}
# run -> [(harmonic channel, harmonic n, its polarisation, sheet cell)]; 'ez_pump' = the tmd_ez D3h sheet (SH h2) and
# the graphene sheet (TH h3) on two lines of one strip: undepleted, h2 and h3 both rank 1, so no p + h2 -> h3 term acts
# and each harmonic has one source sheet
_SHEET_HARMONICS = {'ez_pump': [('h2', 2, 'hz', _SHEET_CELL), ('h3', 3, 'ez', _GRAPHENE_CELL)],
                    'tmd_hz': [('h', 2, 'ez', _SHEET_CELL)]}


def _sheet_harmonic_probes(run):
    """[(probe name, channel, component, cell)] of the forward / backward probes of every harmonic of `run`."""
    return [(f'{channel}_{label}', channel, 'ey' if polarization == 'hz' else 'ez', cell)
            for channel, _, polarization, _ in _SHEET_HARMONICS[run] for label, cell in _HARMONIC_PROBES.items()]


def _sheet_specs():
    pump = {'definition': 'wavelength', 'wavelength': _PUMP_WAVELENGTH}
    second, third = {'definition': 'combination', 'combination': '2 p'}, {'definition': 'combination',
                                                                          'combination': '3 p'}
    tmd = {'sheet_model': 'D3h', 'chi_s': _TMD_CHI}
    graphene = {'sheet_model': 'sigma3', 'thg_sigma3_real': _GRAPHENE_THG_SIGMA.real,
                'thg_sigma3_imag': _GRAPHENE_THG_SIGMA.imag, 'sigma3_real': _GRAPHENE_KERR_SIGMA.real,
                'sigma3_imag': _GRAPHENE_KERR_SIGMA.imag}
    sources = [('pump', 'p', _PUMP_AMPLITUDE)]
    return {
        'ez_pump': _sheet_strip_spec(
            'D3h sheet SHG (E_z pump, H_z SH) and graphene sigma3 sheet THG',
            [('tmd', _SHEET_CELL, {**tmd, 'armchair_angle': 0.0}), ('graphene', _GRAPHENE_CELL, graphene)],
            [('p', 'ez', pump), ('h2', 'hz', second), ('h3', 'ez', third)], sources,
            [('pump_tmd', 'p', 'ez', _SHEET_CELL), ('pump_graphene', 'p', 'ez', _GRAPHENE_CELL)]
            + _sheet_harmonic_probes('ez_pump')),
        'tmd_hz': _sheet_strip_spec(
            'D3h sheet SHG (H_z pump, E_z SH)', [('tmd', _SHEET_CELL, {**tmd, 'armchair_angle': 90.0})],
            [('p', 'hz', pump), ('h', 'ez', second)], sources,
            [('pump_lower', 'p', 'ey', _SHEET_CELL), ('pump_upper', 'p', 'ey', _SHEET_CELL + 1)]
            + _sheet_harmonic_probes('tmd_hz')),
    }


def _green(probe_cell, source_cell, wavenumber):
    return (1j * _STRIP_CELL ** 2 / (2.0 * math.sin(wavenumber))) * cmath.exp(
        -1j * wavenumber * abs(probe_cell - source_cell))


def _check_sheets(results_by_name):
    errors, details = {}, {}
    surface_sources = {   # harmonic channel of a run -> P_s / eps0 of the probed pump [V]
        ('ez_pump', 'h2'): lambda probe: -0.5 * _TMD_CHI * probe('pump_tmd') ** 2,
        ('ez_pump', 'h3'): lambda probe: (0.25 * _GRAPHENE_THG_SIGMA * probe('pump_graphene') ** 3
                                          / (1j * 3.0 * _PUMP_OMEGA * EPS0)),
        ('tmd_hz', 'h'): lambda probe: -0.5 * _TMD_CHI * _face_value(probe('pump_lower'), probe('pump_upper')) ** 2,
    }
    for run, harmonics in _SHEET_HARMONICS.items():
        results = results_by_name[run]

        def probe(name):
            return probe_phasor(results, _PHYSICS, name)
        for channel, harmonic, polarization, sheet_cell in harmonics:
            surface_over_eps0 = surface_sources[(run, channel)](probe)
            omega = harmonic * _PUMP_OMEGA
            wavenumber = _cell_wavenumber(omega, 1.0)
            source = -(omega ** 2) * MU0 * EPS0 * surface_over_eps0 / _YEE_CELL
            details[f'{run}_{channel}_surface_polarization_over_eps0'] = _pair(surface_over_eps0)
            for label, cell in _HARMONIC_PROBES.items():
                if polarization == 'ez':
                    expected = _green(cell, sheet_cell, wavenumber) * source
                else:   # centre probe = mean of the hi face `cell` and the lo face `cell - 1`
                    expected = 0.5 * (_green(cell, sheet_cell, wavenumber)
                                      + _green(cell - 1, sheet_cell, wavenumber)) * source
                measured = probe(f'{channel}_{label}')
                errors[f'{run} {channel} {label}'] = relative_error(measured, expected)
                details.update({f'{run}_{channel}_{label}_measured': _pair(measured),
                                f'{run}_{channel}_{label}_expected': _pair(expected)})
    return _normalized_result(errors, {item: _LATTICE_TOLERANCE for item in errors}, details)


SHEET_CPU_SELFTESTS = (
    SelfTestCase(
        name='nlfd_sheet_sites',
        label='Nonlinear FDFD: sheet sites and power identity',
        description='CPU check of physics/nlfd/sheets.py: (1) the surface chi_s tensor rotated to model axes equals '
                    'a nnn + b (ntt + nzz) + d (tnt + ttn + znz + zzn) at random normals, and the D3h sheet tensor is '
                    '120 deg periodic in its armchair angle; (2) on a graded 12 x 10 grid the deposit of random site '
                    'polarisations is the exact adjoint of the site sampling (sum_k A_k conj(E_k) P_k = sum_s L_s '
                    'conj(E_s) . P_s) for a disk surface (staircase and smoothed, inside / outside selvedge, periodic '
                    'or not) and line sheets along x and y; (3) on random fields the site work of every mixing-table '
                    'term (p and sh = 2 p both polarisations, a Kleinman surface ntt = tnt and a D3h line sheet) sums '
                    'to 0; (4) a disk of radius 20.3 cells: staircase perimeter / 2 pi R in '
                    f'{_STAIRCASE_EXCESS} (mean 4 / pi), smoothed perimeter within {_LENGTH_TOLERANCE:g} (estimate) '
                    f'and normals within {_NORMAL_TOLERANCE:g} rad of the exact radius where each face line crosses '
                    'the circle (the crossing / chord bisections are exact for a circle); (5) the adjoint transpose '
                    'sheets.sheet_cotangents: Re(c . '
                    'dD[V]) = Re(g . V) of the deposited density along random tangents for the sheets of (3) plus a '
                    'complex sigma3 Kerr line sheet (conjugated slots, current response); (6) the per-cell sheet '
                    'absorption (sheet_site_absorption) times the cell areas sums to -site work. measured = the worst '
                    f'of (2), (3), (5), (6); error = measured + failed checks; tolerance {_ROUND_OFF:g}.',
        specs=lambda: {},
        check=_check_sheet_sites,
    ),
)
SHEET_GPU_SELFTESTS = (
    SelfTestCase(
        name='nlfd_surface_sheet',
        label='Nonlinear FDFD: surface chi2 dipole layer',
        description='y-periodic vacuum strip (4 rows, uniform 10 nm mesh, CPML along x), pump channel p at 1.55 um '
                    'polarization both (E_z and E_y line sources in cell 100, 1e7 V/m), H_z SH channel sh = 2 p, '
                    f'surface_chi2 (chi_s_nnn {_SURFACE_CHI["chi_s_nnn"]:g}, chi_s_ntt {_SURFACE_CHI["chi_s_ntt"]:g}, '
                    f'chi_s_tnt {_SURFACE_CHI["chi_s_tnt"]:g} m^2/V), one run with two features: selvedge inside on '
                    f'the cells {list(_SURFACE_SEGMENTS["inside"])}, selvedge outside on '
                    f'{list(_SURFACE_SEGMENTS["outside"])} (the layers do not radiate, so the segments do not '
                    'interact; merged run not yet confirmed by a run). At normal incidence only chi_s_ntt acts: P_s,n = (1/2) eps0 chi_s_ntt (E_z^2 + '
                    'E_t^2) with E_z the mean of the two cells at each interface face and E_t the face E_y (from the '
                    'two centre probes by the vacuum lattice recurrence), a one-cell dipole layer that does not radiate, so the '
                    'SH E_x in the selvedge cell must be exactly -n_x P_s,n / (eps0 h) at both ends of each segment '
                    '(outward normal -x / +x), 0 in the other cell, and the forward SH E_y 0. measured = error = worst relative '
                    f'deviation / {_EXACT_TOLERANCE:g}, tolerance 1. Pins the face sites, the outward normal sign, '
                    'the selvedge choice, the E_z interface mean, the tangential face sample and the hz face deposit.',
        specs=_surface_specs,
        check=_check_surface),
    SelfTestCase(
        name='nlfd_surface_normal_jump',
        label='Nonlinear FDFD: surface chi2 normal dipole layer, two meshes',
        description='Self-test 9 (b). y-periodic vacuum strip (period 200 nm, CPML along x), H_z pump p at 1.55 um '
                    f'(line source, {_JUMP_AMPLITUDE:g} V/m: E_y along the full height), H_z SH sh = 2 p, surface_chi2 '
                    f'with chi_s_nnn {_JUMP_CHI:g} m^2/V only on the band y 60-120 nm, x 1-2.5 um: at grazing '
                    'incidence the pump E_y is the normal field of the y-normal faces, so P_s,n = (1/2) eps0 chi_s_nnn '
                    'E_n^2 is a dipole layer whose tangential E_x jump is -D_x P_s,n / eps0 = j (sin(2 K_w h) / h) '
                    'P_s,n / eps0 (E_n = the probed centre E_y of the selvedge cell). Measured: E_x(y0 + h/2) - '
                    'E_x(y0 - 3h/2) of the SH (cell-centre probes, x = 1.7525 um) on uniform 10 and 5 nm meshes, ratio '
                    'r(h) to the closed form: at most first order in h (the probes sit symmetric about the deposit '
                    'row, so the O(h) terms nearly cancel). Checks: the Richardson value 2 r(h/2) - r(h) within '
                    f'{_RICHARDSON_TOLERANCE:g} of 1 and the observed order log2(|r(h) - 1| / |r(h/2) - 1|) in '
                    f'{_ORDER_RANGE} unless both |r - 1| < {_ORDER_FLOOR:g} (all estimates). measured = error = the worst normalised item, tolerance 1. Pins '
                    'chi_s_nnn, the normal-field sample and the dipole-layer deposit.',
        specs=_jump_specs,
        check=_check_jump),
    SelfTestCase(
        name='nlfd_surface_disk_shg',
        label='Nonlinear FDFD: smoothed surface chi2 disk SHG vs the continuum integral',
        description='A vacuum disk (radius 200 nm, centre off the grid) with surface_chi2 (chi_s_nnn '
                    f'{_DISK_CHI["chi_s_nnn"]:g}, chi_s_ntt {_DISK_CHI["chi_s_ntt"]:g}, chi_s_tnt '
                    f'{_DISK_CHI["chi_s_tnt"]:g} m^2/V, selvedge inside) in a {_DISK_SIZE * 1e6:g} um all-CPML '
                    f'vacuum box (disk edge 0.4 um from the box edge; was 1.6 um, not yet confirmed by a run), H_z '
                    f'plane wave (TFSF, {_DISK_AMPLITUDE:g} V/m) at 1550 nm. With eps 1 the pump is unperturbed, so the '
                    'reference is the continuum sheet on the circle radiating in vacuum: P_sh = int eta0 k2 w2^2 / (16 '
                    'pi) |a int (r_hat x P_s)_z e^{j k2 a cos(theta - phi)} dphi|^2 dtheta (quadrature, pump amplitude '
                    'from the probe at the centre). Measured: power_exchange_sh (= the radiated SH power) of interface '
                    "'smoothed' on uniform 10 and 5 nm meshes (R = 20 / 40 cells), each within "
                    f'{_DISK_TOLERANCE:g} relative (estimate), the 5 nm one no worse than the 10 nm one (the staircase '
                    'perimeter is pinned on the CPU by nlfd_sheet_sites). measured = error = the worst normalised item, '
                    'tolerance 1. Pins the smoothed normals and lengths in a radiating model run.',
        specs=_disk_specs,
        check=_check_disk,
        heavy=True),
    SelfTestCase(
        name='nlfd_sheet_nonlinearity',
        label='Nonlinear FDFD: 2D-material sheets vs the lattice Green function',
        description='y-periodic vacuum strip (as nlfd_surface_sheet) with sheet_nonlinearity features on vertical '
                    'lines, pump p at 1.55 um (line source in cell 100), two runs: ez_pump (E_z pump, channels h2 = 2 '
                    f'p H_z and h3 = 3 p E_z) with a D3h sheet in cell {_SHEET_CELL} (chi_s {_TMD_CHI:g} m^2/V, '
                    'armchair along the line: P_t = -(1/2) eps0 chi_s E_z^2 on the line cell\'s hi face, H_z SH) and '
                    f'a graphene sheet in cell {_GRAPHENE_CELL} (sigma3 THG K_z = sigma3 E_z^3 / 4 entered as P = K / '
                    '(j 3 w), a Kerr sigma3 that must stay inactive undepleted, E_z TH; h2 and h3 both rank 1, so no '
                    'p + h2 -> h3 term acts and each harmonic has one source sheet), and tmd_hz (D3h in cell '
                    f'{_SHEET_CELL}, armchair 90 deg: E_y pump -> P_z = -(1/2) eps0 chi_s E_t^2, E_z SH). Merged '
                    'ez_pump run not yet confirmed by a run. Each harmonic is the single-site lattice '
                    "Green's function source -(n w)^2 mu0 P_s / h from the probed pump (hz: on the face lattice, "
                    'probed as the mean of two faces): forward (cell 300) and backward (cell 50) phasors, '
                    f'measured = error = worst relative deviation / {_LATTICE_TOLERANCE:g} (CPML floor not '
                    'measured yet), tolerance 1. Pins the line sites (ez centres, hz hi faces), the sheet frame and '
                    'armchair rotation, the current response and the sigma3 process split.',
        specs=_sheet_specs,
        check=_check_sheets),
)


# ---- templates ----
# gold_ridge_surface_shg: a gold ridge (library Drude gold) in vacuum lit by an in-plane polarised (H_z) TFSF plane
# wave at 1550 nm along x, uniform 5 nm mesh (~5 cells per skin depth; every swept edge on a cell edge, so the
# staircase sites are exact), surface_chi2 on the ridge with the normal field inside the metal; the H_z SH = 2 p
# radiates from the corners and the y-normal faces (E_n = E_y there); sweep of the ridge width, generated SH power and
# the SH far field. Coefficients illustrative.
GOLD_RIDGE_LABEL = 'Gold ridge surface SHG'
GOLD_RIDGE_DESCRIPTION = ('Nonlinear FDFD (undepleted, surface chi2): an in-plane polarised 1550 nm plane wave on a '
                          'gold ridge generates second harmonic at the metal surface; sweep of the ridge width, SH '
                          'power and far field.')
_GOLD_SIZE = 2.4e-6
_GOLD_WIDTHS = (100e-9, 150e-9, 200e-9, 250e-9, 300e-9, 400e-9)


def gold_ridge_surface_shg_spec():
    """Template 'gold_ridge_surface_shg': H_z-pumped gold ridge, surface chi2, ridge width sweep, SH far field."""
    centre = 0.5 * _GOLD_SIZE
    return ModelSpec(
        title=GOLD_RIDGE_LABEL,
        description=('Surface second-harmonic generation of a gold ridge (2D, in vacuum): an in-plane polarised (H_z) '
                     'plane wave at 1550 nm drives the isotropic surface susceptibility of the gold boundary '
                     '(surface_chi2: chi_s_nnn, chi_s_ntt, chi_s_tnt, normal field taken inside the metal, sites on '
                     'the cell faces, exact for this axis-aligned ridge whose swept edges lie on cell edges). At '
                     'normal incidence a flat surface radiates no SH, so the harmonic comes from the '
                     'corners and the faces normal to the incident field; P_sh (power_exchange_sh, W/m) follows the '
                     'ridge width w_ridge (swept). Gold is the library Drude entry (fit 0.6-2 um); the surface '
                     'coefficients are illustrative orders of magnitude, not fitted values.'),
        parameters=[ParameterSpec('lambda_p', '1550*nm', 'pump vacuum wavelength'),
                    ParameterSpec('E0', 1e8, 'incident in-plane field [V/m]'),
                    ParameterSpec('w_ridge', '200*nm', 'ridge width (swept)'),
                    ParameterSpec('h_ridge', '100*nm', 'ridge height'),
                    ParameterSpec('chi_nnn', 1e-18, 'chi_s_nnn [m^2/V]'),
                    ParameterSpec('chi_ntt', 1e-19, 'chi_s_ntt [m^2/V]'),
                    ParameterSpec('chi_tnt', 5e-20, 'chi_s_tnt [m^2/V]')],
        domain=DomainSpec(size_x=_GOLD_SIZE, size_y=_GOLD_SIZE),
        geometry=[GeometrySpec('ridge', 'rectangle', {'x0': f'{centre!r} - w_ridge/2', 'y0': f'{centre!r} - h_ridge/2',
                                                      'x1': f'{centre!r} + w_ridge/2',
                                                      'y1': f'{centre!r} + h_ridge/2'}),
                  GeometrySpec('tfsf_box', 'rectangle', {'x0': 0.6e-6, 'y0': 0.6e-6, 'x1': 1.8e-6, 'y1': 1.8e-6}),
                  GeometrySpec('far_field_box', 'rectangle', {'x0': 0.4e-6, 'y0': 0.4e-6, 'x1': 2.0e-6,
                                                              'y1': 2.0e-6})],
        materials=[MaterialSpec('gold', library='gold', selection=['ridge'])],
        physics=[PhysicsSpec('nl', 'em_fdfd_nonlinear', features=[
            FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': 'lambda_p',
                                                      'polarization': 'hz'}),
            FeatureSpec('sh', 'harmonic_channel', [], {'definition': 'combination', 'combination': '2 p',
                                                       'polarization': 'hz'}),
            FeatureSpec('gold_surface', 'surface_chi2', ['ridge'], {
                'chi_s_nnn': 'chi_nnn', 'chi_s_ntt': 'chi_ntt', 'chi_s_tnt': 'chi_tnt', 'selvedge': 'inside',
                'interface': 'staircase'}),
            FeatureSpec('pump', 'plane_wave_tfsf', ['tfsf_box'], {'channel': 'p', 'amplitude': 'E0', 'phase': 0.0}),
            FeatureSpec('sh_far_field', 'far_field', ['far_field_box'], {'channel': 'sh', 'n_angles': 360})])],
        mesh=MeshSpec(mode='uniform', dx=5e-9),
        study=StudySpec(type='nonlinear_frequency_domain', sweep=SweepSpec('w_ridge', list(_GOLD_WIDTHS))),
        derived_values=[DerivedValueSpec('P_sh', 'nl_power_exchange_sh', [], 'mean')])


# graphene_thg_grating: x-periodic cell (period 0.9 um) of a 200 nm silicon nitride slab on silica with a nitride grating
# (half the period, 150 nm high) on top, graphene across the period just above the teeth (conducting_sheet Kubo E_F
# 0.4 eV for the linear response + sheet_nonlinearity sigma3 THG), an E_z line-source pump from the air side, uniform
# 10 nm mesh; sweep of the pump wavelength across the guided-mode resonance of the slab, TH flux up / down.
GRAPHENE_LABEL = 'Graphene on a nitride grating: THG'
GRAPHENE_DESCRIPTION = ('Nonlinear FDFD (undepleted, sigma3 sheet): third harmonic of graphene on a silicon nitride '
                        'grating; pump wavelength sweep across the guided-mode resonance, TH power up and down.')
_GRAPHENE_PERIOD = 0.9e-6
_GRAPHENE_HEIGHT = 3e-6
_GRAPHENE_WAVELENGTHS = tuple(1.40e-6 + 0.025e-6 * index for index in range(13))   # 1.40-1.70 um


def graphene_thg_grating_spec():
    """Template 'graphene_thg_grating': graphene sigma3 THG on a nitride grating, pump wavelength sweep."""
    substrate_top, slab_top, teeth_top = 1.2e-6, 1.4e-6, 1.55e-6
    sheet_y = teeth_top + 5e-9
    return ModelSpec(
        title=GRAPHENE_LABEL,
        description=('Third-harmonic generation of a graphene sheet on a silicon nitride grating (2D, E_z, '
                     'x-periodic): the grating couples the pump into the guided mode of the nitride slab, so the field '
                     'at the graphene and its third harmonic (sheet_nonlinearity sigma3, K = sigma3 E^3 / 4) peak at '
                     'the guided-mode resonance. conducting_sheet (graphene Kubo, E_F 0.4 eV) adds the linear sheet '
                     'response on the same line. The study sweeps the pump wavelength; P_th_up / P_th_down are the TH '
                     'fluxes above and below (W/m per unit length in z, per period), T_pump the pump transmission. '
                     'sigma3 is an illustrative magnitude (graphene THG values depend strongly on E_F and the photon '
                     'energy).'),
        parameters=[ParameterSpec('lambda_p', '1550*nm', 'pump vacuum wavelength (swept)'),
                    ParameterSpec('E0', 1e8, 'pump amplitude [V/m]'),
                    ParameterSpec('sigma3_thg', 1e-21, 'graphene THG sigma3 [S m^2/V^2]'),
                    ParameterSpec('P_inc', f'E0**2 / (2 * {float(ETA0)!r}) * {_GRAPHENE_PERIOD!r}',
                                  'incident pump power per period [W/m]')],
        domain=DomainSpec(size_x=_GRAPHENE_PERIOD, size_y=_GRAPHENE_HEIGHT),
        geometry=[GeometrySpec('substrate', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': _GRAPHENE_PERIOD,
                                                          'y1': substrate_top}),
                  GeometrySpec('slab', 'rectangle', {'x0': 0.0, 'y0': substrate_top, 'x1': _GRAPHENE_PERIOD,
                                                     'y1': slab_top}),
                  GeometrySpec('tooth', 'rectangle', {'x0': 0.0, 'y0': slab_top, 'x1': 0.5 * _GRAPHENE_PERIOD,
                                                      'y1': teeth_top}),
                  GeometrySpec('graphene_line', 'line', {'x0': 0.0, 'y0': sheet_y, 'x1': _GRAPHENE_PERIOD,
                                                         'y1': sheet_y}),
                  GeometrySpec('source_line', 'line', {'x0': 0.0, 'y0': 2.6e-6, 'x1': _GRAPHENE_PERIOD,
                                                       'y1': 2.6e-6}),
                  GeometrySpec('up_line', 'line', {'x0': 0.0, 'y0': 2.2e-6, 'x1': _GRAPHENE_PERIOD, 'y1': 2.2e-6}),
                  GeometrySpec('down_line', 'line', {'x0': 0.0, 'y0': 0.6e-6, 'x1': _GRAPHENE_PERIOD,
                                                     'y1': 0.6e-6})],
        materials=[MaterialSpec('silica', properties={'eps_r': 2.085}, selection=['substrate']),
                   MaterialSpec('nitride', properties={'eps_r': 3.98}, selection=['slab', 'tooth'])],
        physics=[PhysicsSpec('nl', 'em_fdfd_nonlinear', settings={'bc_x_lo': 'periodic', 'bc_x_hi': 'periodic'},
                             features=[
            FeatureSpec('p', 'harmonic_channel', [], {'definition': 'wavelength', 'wavelength': 'lambda_p',
                                                      'polarization': 'ez'}),
            FeatureSpec('th', 'harmonic_channel', [], {'definition': 'combination', 'combination': '3 p',
                                                       'polarization': 'ez'}),
            FeatureSpec('graphene', 'conducting_sheet', ['graphene_line'], {
                'model': 'graphene_kubo', 'fermi_energy': 0.4, 'scattering_time': 1e-13, 'temperature': 300.0}),
            FeatureSpec('graphene_nl', 'sheet_nonlinearity', ['graphene_line'], {
                'sheet_model': 'sigma3', 'thg_sigma3_real': 'sigma3_thg'}),
            FeatureSpec('pump', 'line_source', ['source_line'], {'channel': 'p', 'amplitude': 'E0', 'phase': 0.0}),
            FeatureSpec('th_up', 'flux_monitor', ['up_line'], {'channel': 'th'}),
            FeatureSpec('th_down', 'flux_monitor', ['down_line'], {'channel': 'th'}),
            FeatureSpec('p_down', 'flux_monitor', ['down_line'], {'channel': 'p'})])],
        mesh=MeshSpec(mode='uniform', dx=10e-9),
        study=StudySpec(type='nonlinear_frequency_domain', sweep=SweepSpec('lambda_p', list(_GRAPHENE_WAVELENGTHS))),
        derived_values=[DerivedValueSpec('P_th_up', 'power', [], 'mean', source='nl.th_up'),
                        DerivedValueSpec('P_th_down', '-power', [], 'mean', source='nl.th_down'),
                        DerivedValueSpec('T_pump', '-power/P_inc', [], 'mean', source='nl.p_down')])


SHEET_TEMPLATES = (   # (key, label, description, factory, summary) for registry.register_template
    ('gold_ridge_surface_shg', GOLD_RIDGE_LABEL, GOLD_RIDGE_DESCRIPTION, gold_ridge_surface_shg_spec,
     'Second harmonic from the surface of a gold ridge lit by light polarised in the plane: harmonic power against '
     'the ridge width, and where the harmonic goes.'),
    ('graphene_thg_grating', GRAPHENE_LABEL, GRAPHENE_DESCRIPTION, graphene_thg_grating_spec,
     'Third harmonic of graphene on a silicon nitride grating: the harmonic peaks where the grating couples the pump '
     'into the waveguide below.'),
)
