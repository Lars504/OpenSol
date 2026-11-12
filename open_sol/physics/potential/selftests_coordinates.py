"""Exact self-tests of electric_currents on polar (r, phi) model domains (symmetry plan 2.16, DomainSpec.coordinates):
a sector resistor between two held arcs and a periodic (phi-wrapped) annulus with a sin(phi) source, read on the
native (r, angle) cells; registered by definition."""
import math

import numpy as np

from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_edges, with_exact_cg
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)

_RUN = 'run'
_CURRENTS = 'currents'
# Sector a < r < b on uniform dr = 20 um (40 radial cells); the inner arc x_lo held at V, the outer x_hi at 0, the angle
# faces insulated, so V depends on r only. Half-cell conductances 2 sigma A_f / dr on the held arcs, 2 sigma A_f / (w_l
# + w_r) inside, A_f = Phi r_f (per unit depth): in series the exact discrete flux F = V / sum 1 / G, the trapezoid rule
# of the analytic sigma Phi V / ln(b / a) (~4.8e-4 at 20 um, 1.2e-4 measured at 10 um, ~dr^2).
_SECTOR_CELL = 2e-5                # m, the sector's dr and outer-arc angle cell (40 x 79 cells)
_INNER_RADIUS = 2e-4               # a = radius_min
_OUTER_RADIUS = 1e-3               # b
_VOLTAGE = 1.0                     # V on the inner arc
_SECTOR_START = 0.3                # rad, polar angle_min
_SECTOR_ANGLE = 0.5 * math.pi      # rad, polar size_y
_SECTOR_SIGMA = 10.0               # S/m
_CG_TOLERANCE = 1e-13
_TOLERANCE = 1e-6                  # float32 V / J storage ~1e-7
_ANALYTIC_BOUND = 1e-3             # allowed |F / F_analytic - 1|
_GEOMETRY_TOLERANCE = 1e-9         # mapped centres vs the native r / angle centres (float64)


def _analytic_current():
    """Sector resistor sigma Phi V / ln(b / a) [A/m]."""
    return _SECTOR_SIGMA * _SECTOR_ANGLE * _VOLTAGE / math.log(_OUTER_RADIUS / _INNER_RADIUS)


def _sector_specs():
    """'run': the two held arcs on the polar sector, stationary, float64 CG."""
    return {_RUN: with_exact_cg(ModelSpec(
        title='Self-test: polar two-arc electric_currents',
        domain=DomainSpec(size_x=_OUTER_RADIUS - _INNER_RADIUS, size_y=_SECTOR_ANGLE, coordinates='polar',
                          radius_min=_INNER_RADIUS, angle_min=_SECTOR_START),
        materials=[MaterialSpec('medium', properties={'sigma_dc': _SECTOR_SIGMA}, selection=['all'])],
        physics=[PhysicsSpec(_CURRENTS, 'electric_currents', features=[
            FeatureSpec('inner', 'voltage', ['x_lo'], {'voltage': _VOLTAGE}),
            FeatureSpec('outer', 'voltage', ['x_hi'], {'voltage': 0.0})])],
        mesh=MeshSpec(mode='uniform', dx=_SECTOR_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('F_inner', f'{_CURRENTS}_current_inner', [], 'mean'),
                        DerivedValueSpec('P_joule', 'q_joule', [], 'integral')]), tolerance=_CG_TOLERANCE)}


def _discrete_chain(radius_edges, angle_edges):
    """(flux F per V, V at the centres per V, radial current density at the centres per V) of the series face
    conductances on the actual cells."""
    areas = float(angle_edges[-1] - angle_edges[0]) * radius_edges
    widths = np.diff(radius_edges)
    resistances = np.concatenate(([widths[0] / (2.0 * _SECTOR_SIGMA * areas[0])],
                                  (widths[:-1] + widths[1:]) / (2.0 * _SECTOR_SIGMA * areas[1:-1]),
                                  [widths[-1] / (2.0 * _SECTOR_SIGMA * areas[-1])]))
    flux = 1.0 / float(np.sum(resistances))
    potential = 1.0 - flux * np.cumsum(resistances)[:-1]
    density = 0.5 * (flux / areas[:-1] + flux / areas[1:])
    return flux, potential, density


def _check_sector(results_by_name):
    """Native V and radial current density against the discrete chain, both held currents, their unit, the scalar
    output, P_joule, the native axes / mapped centres / radial label, and the analytic current within the bound."""
    results = results_by_name[_RUN]
    potential_field = results.field(f'{_CURRENTS}.V', native=True)
    density_field = results.field(f'{_CURRENTS}.J_x', native=True)
    potential = np.asarray(potential_field.values, dtype=np.float64)
    density = np.asarray(density_field.values, dtype=np.float64)
    radius_centres = np.asarray(potential_field.x_centers, dtype=np.float64)
    angle_centres = np.asarray(potential_field.y_centers, dtype=np.float64)
    radius_edges = cell_edges(radius_centres, _INNER_RADIUS)
    angle_edges = cell_edges(angle_centres, _SECTOR_START)
    flux, expected_potential, expected_density = (value * _VOLTAGE for value in _discrete_chain(
        radius_edges, angle_edges))
    inner = results.result(_CURRENTS, 'inner')
    inner_flux = float(inner['current'])
    outer_flux = float(results.result(_CURRENTS, 'outer')['current'])
    scalar_flux = float(results.summary()['derived']['F_inner'])
    power_joule = float(results.summary()['derived']['P_joule'])
    mapped_x, mapped_y = (np.asarray(centres, dtype=np.float64) for centres in potential_field.mapped_centres)
    radius_scale = float(radius_centres[-1])
    analytic = _analytic_current()
    analytic_deviation = abs(inner_flux / analytic - 1.0)
    errors = {'profile': float(np.max(np.abs(potential - expected_potential[:, None]))) / _VOLTAGE,
              'flux_density': float(np.max(np.abs(density - expected_density[:, None])))
              / float(np.max(expected_density)),
              'inner': abs(inner_flux - flux) / flux,
              'outer': abs(outer_flux + flux) / flux,
              'scalar_output': abs(scalar_flux - flux) / flux,
              'power': abs(power_joule - _VOLTAGE * flux) / (_VOLTAGE * flux),
              'unit': 0.0 if inner['units'][2] == 'A/m' else 1.0,
              'axes': 0.0 if tuple(potential_field.axes) == ('r', 'phi') else 1.0,
              'radial_label': 0.0 if density_field.info.label.endswith('radial (r)') else 1.0,
              'mapped_radius': max(0.0, float(np.max(np.abs(np.hypot(mapped_x, mapped_y)
                                                              - radius_centres[:, None]))) / radius_scale
                                   - _GEOMETRY_TOLERANCE),
              'analytic_excess': max(0.0, analytic_deviation - _ANALYTIC_BOUND)}
    error = float(np.max(np.asarray(list(errors.values()), dtype=np.float64)))   # NaN propagates
    return {'measured': inner_flux, 'expected': flux, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'current_inner': inner_flux, 'current_discrete': flux,
                        'current_analytic': analytic, 'analytic_deviation': analytic_deviation,
                        'unit': inner['units'][2], 'axes': list(potential_field.axes),
                        'n_cells': list(potential.shape), 'P_joule_W_per_m': power_joule,
                        'P_discrete_W_per_m': _VOLTAGE * flux}}


# ---- currents_polar_wrap: full annulus a < r < b with a periodic phi wrap (DomainSpec.angle_wrap) ----
# Both arcs at 0 V, current source s = s0 sin(phi) / r^2 (drawn as s0 y / hypot(x, y)^3): the cell-centred sin(phi_j)
# is an exact eigenvector of the periodic uniform-dphi chain (it is odd about the angle_min ray, so insulated rays
# would not reproduce it), V = a_i sin(phi_j) with a_i the exact discrete radial chain.
_WRAP_INNER = 4e-4                 # m, radius_min
_WRAP_CELL = 4e-5                  # m: 15 radial x 158 angle cells
_WRAP_START = 0.3                  # rad, angle_min: the wrap ray off phi = 0
_WRAP_SOURCE = 10.0                # A/m, s0 (V ~ s0 / sigma)
_WRAP_SIGMA = 10.0                 # S/m
_WRAP_SPAN_TOLERANCE = 1e-9        # relative: the native angle cells span 2 pi


def _wrap_specs():
    """'run': the periodic annulus, both arcs at 0 V, the sin(phi) / r^2 source, stationary, float64 CG."""
    return {_RUN: with_exact_cg(ModelSpec(
        title='Self-test: periodic polar annulus',
        domain=DomainSpec(size_x=_OUTER_RADIUS - _WRAP_INNER, size_y=2.0 * math.pi, coordinates='polar',
                          radius_min=_WRAP_INNER, angle_min=_WRAP_START, angle_wrap='periodic'),
        materials=[MaterialSpec('medium', properties={'sigma_dc': _WRAP_SIGMA}, selection=['all'])],
        physics=[PhysicsSpec(_CURRENTS, 'electric_currents', features=[
            FeatureSpec('inner', 'voltage', ['x_lo'], {'voltage': 0.0}),
            FeatureSpec('outer', 'voltage', ['x_hi'], {'voltage': 0.0}),
            FeatureSpec('drive', 'current_source', ['all'], {'density': f'{_WRAP_SOURCE!r} * y / hypot(x, y)**3'})])],
        mesh=MeshSpec(mode='uniform', dx=_WRAP_CELL),
        study=StudySpec(type='stationary'),
        derived_values=[DerivedValueSpec('P_joule', 'q_joule', [], 'integral')]), tolerance=_CG_TOLERANCE)}


def _wrap_chain(radius_edges, d_angle):
    """(a_i [V], radial face conductances, widths, centres) per unit dphi: 2 sigma r_f / (w_l + w_r) inside, 2 sigma r_f
    / w on the 0 V arcs, the angular term 2 sigma w (1 - cos dphi) / (r_c dphi^2) of the periodic sin eigenvector,
    driven by s0 w / r_c."""
    widths = np.diff(radius_edges)
    centres = 0.5 * (radius_edges[:-1] + radius_edges[1:])
    faces = np.empty(radius_edges.size)
    faces[1:-1] = 2.0 * _WRAP_SIGMA * radius_edges[1:-1] / (widths[:-1] + widths[1:])
    faces[0] = 2.0 * _WRAP_SIGMA * radius_edges[0] / widths[0]
    faces[-1] = 2.0 * _WRAP_SIGMA * radius_edges[-1] / widths[-1]
    angular = 2.0 * _WRAP_SIGMA * widths * (1.0 - math.cos(d_angle)) / (centres * d_angle ** 2)
    matrix = np.diag(faces[:-1] + faces[1:] + angular) - np.diag(faces[1:-1], 1) - np.diag(faces[1:-1], -1)
    return np.linalg.solve(matrix, _WRAP_SOURCE * widths / centres), faces, widths, centres


def _check_wrap(results_by_name):
    """Native V, radial and angular current densities (the angular one crosses the wrap face) and P_joule against
    the exact discrete chain."""
    results = results_by_name[_RUN]
    potential_field = results.field(f'{_CURRENTS}.V', native=True)
    potential = np.asarray(potential_field.values, dtype=np.float64)
    radial = np.asarray(results.field(f'{_CURRENTS}.J_x', native=True).values, dtype=np.float64)
    angular = np.asarray(results.field(f'{_CURRENTS}.J_y', native=True).values, dtype=np.float64)
    radius_edges = cell_edges(np.asarray(potential_field.x_centers, dtype=np.float64), _WRAP_INNER)
    angle_edges = cell_edges(np.asarray(potential_field.y_centers, dtype=np.float64), _WRAP_START)
    angle_centres = 0.5 * (angle_edges[:-1] + angle_edges[1:])
    n_angle = angle_centres.size
    d_angle = float(angle_edges[-1] - angle_edges[0]) / n_angle
    amplitude, faces, widths, centres = _wrap_chain(radius_edges, d_angle)
    sine, cosine = np.sin(angle_centres), np.cos(angle_centres)
    # radial density at the faces (toward +r) per sin(phi): flow per dphi / r_f, the arcs at 0 V
    face_density = faces * np.concatenate(([-amplitude[0]], amplitude[:-1] - amplitude[1:], [amplitude[-1]])) \
        / radius_edges
    expected_radial = 0.5 * (face_density[:-1] + face_density[1:])[:, None] * sine[None, :]
    expected_angular = (-_WRAP_SIGMA * amplitude * math.sin(d_angle) / (centres * d_angle))[:, None] * cosine[None, :]
    expected_power = float(np.sum(amplitude * _WRAP_SOURCE * widths / centres)) * d_angle * 0.5 * n_angle
    power = float(results.summary()['derived']['P_joule'])
    span_deviation = abs(float(angle_edges[-1] - angle_edges[0]) / (2.0 * math.pi) - 1.0)
    errors = {'profile': float(np.max(np.abs(potential - amplitude[:, None] * sine[None, :])))
              / float(np.max(np.abs(amplitude))),
              'radial_density': float(np.max(np.abs(radial - expected_radial))) / float(np.max(np.abs(expected_radial))),
              'angular_density': float(np.max(np.abs(angular - expected_angular)))
              / float(np.max(np.abs(expected_angular))),
              'power': abs(power - expected_power) / expected_power,
              'span_excess': max(0.0, span_deviation - _WRAP_SPAN_TOLERANCE)}
    error = float(np.max(np.asarray(list(errors.values()), dtype=np.float64)))   # NaN propagates
    return {'measured': power, 'expected': expected_power, 'error': error, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'n_cells': list(potential.shape), 'amplitude_max_V': float(np.max(amplitude)),
                        'P_joule_W_per_m': power, 'P_discrete_W_per_m': expected_power}}


COORDINATE_SELFTEST_CASES = (
    SelfTestCase(
        name='currents_polar_sector', label='Electric currents: polar (r, phi) sector resistor',
        description=(f'Electric currents on a polar domain (coordinates polar, a quarter sector from phi = '
                     f'{_SECTOR_START:g} rad), sigma_dc = {_SECTOR_SIGMA:g} S/m everywhere, derived P_joule = the '
                     f'integral of q_joule (sector areas). Inner arc x_lo (r = a = {_INNER_RADIUS * 1e3:g} mm) held '
                     f'at {_VOLTAGE:g} V by a voltage feature, outer arc x_hi (r = b = {_OUTER_RADIUS * 1e3:g} mm) at '
                     f'0 V, insulated angle faces, uniform {_SECTOR_CELL * 1e6:g} um cells (dr and along the outer '
                     f'arc), stationary, float64 CG to {_CG_TOLERANCE:g}. The metric face areas A_f = Phi r_f give '
                     'the exact discrete radial current F = V / sum 1 / G (half-cell 2 sigma A_f / dr on the held '
                     'arcs, 2 sigma A_f / (w_l + w_r) inside, cells rebuilt from the native centres), V at each '
                     'centre the series drop, J_x (radial) the mean of F / A_f over its two faces. Read with '
                     "Results.field(native=True) on the (r, phi) cells. error = max(max |V - V_j| / V, J_x / its "
                     'max, |F_inner - F| / F, |F_outer + F| / F, the scalar output, |P_joule - V F| / (V F), the '
                     'flux unit A/m, the native axes (r, phi), the radial label of J_x and the mapped drawing-plane '
                     f'radius (0 / 1 or excess), the excess of |F / F_analytic - 1| over {_ANALYTIC_BOUND:g}), '
                     f'tolerance {_TOLERANCE:g}. Analytic current sigma Phi V / ln(b / a) = '
                     f'{_analytic_current():.6g} A/m (trapezoid deviation ~4.8e-4 at this dr, estimated from 1.2e-4 '
                     'at 10 um). The angular-face (ray-driven) sector and the spherical shell are examples '
                     '(polar_sector_angular_resistor, spherical_two_layer_capacitor). Not yet run at '
                     f'{_SECTOR_CELL * 1e6:g} um.'),
        specs=_sector_specs, check=_check_sector),
    SelfTestCase(
        name='currents_polar_wrap', label='Electric currents: periodic phi wrap of a full polar annulus',
        description=(f'Electric currents on a full 2 pi polar annulus (coordinates polar, angle_wrap periodic, angle_min '
                     f'{_WRAP_START:g} rad, a = {_WRAP_INNER * 1e3:g} mm, b = {_OUTER_RADIUS * 1e3:g} mm, uniform '
                     f'{_WRAP_CELL * 1e6:g} um cells), sigma_dc = {_WRAP_SIGMA:g} S/m, both arcs held at 0 V, a '
                     f'current source s0 y / hypot(x, y)^3 = s0 sin(phi) / r^2 (s0 = {_WRAP_SOURCE:g} A/m). The '
                     'cell-centred sin(phi_j) is an exact eigenvector of the periodic uniform-dphi chain (odd about the '
                     'wrap ray, so insulated rays would differ), so V = a_i sin(phi_j) with a_i the exact discrete '
                     'radial chain (2 sigma r_f / (w_l + w_r), half cells on the arcs, angular term 2 sigma w (1 - cos '
                     'dphi) / (r_c dphi^2)). Checked on the native cells: V, J_x (centre mean of the face densities), '
                     'J_y = -sigma a_i cos(phi_j) sin(dphi) / (r_c dphi) (the faces of the first and last angle cells '
                     'cross the wrap) and P_joule = sum V s = sum_i a_i s0 w_i dphi / r_c x N / 2, plus the native span '
                     f'2 pi. error = largest relative deviation, tolerance {_TOLERANCE:g}; stationary, float64 CG to '
                     f'{_CG_TOLERANCE:g}. Not yet run at {_WRAP_CELL * 1e6:g} um.'),
        specs=_wrap_specs, check=_check_wrap),
)
