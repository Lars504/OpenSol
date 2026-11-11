"""Self-tests of the FEM / axisymmetric optomechanical overlaps (optomechanics_fem): a dielectric sphere in a PEC shell
(exact TE / TM fields) vs the derivative of its characteristic roots, the photoelastic dilation coefficient and hoop /
shear contraction on a curved half-disk triangle mesh, and the 3D ring average; all CPU, physics imports inside checks."""
import math

import numpy as np

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import no_runs, scaled_worst_result

# ---- sphere (index 1.444, radius a) in a PEC shell (radius b), m = 0 modes of degree l: TE E = F(rho) dP_l/dtheta
# e_phi, TM H_phi = F(rho) dP_l/dtheta (E_rho = -l (l + 1) F P_l / (eps rho), E_theta = -(rho F)' dP_l/dtheta / (eps
# rho)); F = j_l(n k rho) inside, A j_l(k rho) + B y_l(k rho) outside. Closed and lossless, so the moving-boundary
# formula with q_n = 1 is the exact d omega / d a of the characteristic root (b fixed): Richardson central differences
# of brentq roots; plain-array Gauss samples in (rho, theta) mapped to (r, z)
_SPHERE_INDEX = 1.444
_SPHERE_RADIUS = 1.0e-6                # m
_SHELL_RADIUS = 1.6e-6                 # m
_SPHERE_DEGREE = 3                     # l
_SPHERE_SCAN = (1.0, 14.0, 4000)       # k a range and samples of the root scan
_SPHERE_POINTS = 64                    # Gauss points per radial interval and over theta
_SPHERE_STEP = 2e-3                    # relative radius step of the root differences
_SPHERE_BRACKET = 1e-2                 # relative k bracket of the shifted roots
_SPHERE_TOLERANCE = 1e-8               # relative d omega / d a

# ---- half-disk (r, z) triangle mesh: silica (n 1.444, p11 0.121, p12 0.270) inside radius a, air out to b, curved
# P2 rings; uniform dilation u = s (r, z) by P2 interpolation must give g_pe / (omega s) = n^2 / 2 (p11 + 2 p12) x the
# silica energy fraction exactly (0.689 for silica, the B-E6 number); u = s (z, r) (hoop s z / r and S_rz = s) with
# diagonal eps and explicit p44 vs an independent contraction; interface / volume 2 pi r sums vs the sphere area / volume
_MESH_RADII = (1.0e-6, 1.5e-6)         # m, interface a, outer b
_MESH_RINGS = (6, 4)                   # rings inside a, between a and b
_MESH_ANGLES = 24                      # segments over theta in [0, pi]
_SILICA = (1.444, 0.121, 0.270)        # n, p11, p12
_DILATION_COEFFICIENT = 0.689          # n^2 / 2 (p11 + 2 p12) of silica, rounded (basiri-esfahani2019 plan)
_DILATION_ROUNDING = 5e-4
_STRAIN = 1e-4                         # s
_SHEAR_P44 = -0.07
_SHEAR_EPS = (2.0, 2.1, 2.2)           # diagonal (rr, phiphi, zz) of the contraction check
_GEOMETRY_TOLERANCE = 1e-4             # relative: P2 arcs of 24 segments vs the exact sphere area / volume
_IDENTITY_TOLERANCE = 1e-12
_MESH_WAVELENGTH = 1.55e-6             # m, omega of the rates (cancels in the ratios)

# ---- ring average: linear 3D field = breathing s1 (X, Y, 0) + crown s2 (X, -Y, 0) + s3 (0, 0, Z) + translation t e_x +
# rotation w_z (-Y, X, 0) + xz shear s5 (Z, 0, X) about the axis point; <q . n>_phi = s1 r n_r + s3 z n_z exactly
_RING_AXIS = (1.0e-6, -2.0e-6, 3.0e-7)                       # m
_RING_RADII = (0.2e-6, 0.7e-6, 1.3e-6, 2.1e-6)               # m
_RING_HEIGHTS = (-0.4e-6, 0.1e-6, 0.5e-6, 0.9e-6)            # m
_RING_ANGLES = (0.3, 1.2, 2.5, -2.0)                         # normal angles [rad] from +r
_RING_FIELD = (2e-4, -3e-4, 1.5e-4, 2e-11, 4e-4, -1e-4)      # s1, s2, s3, t [m], w_z, s5
_RING_POINTS = 16
_RING_TOLERANCE = 1e-12


def _report(errors, details):
    """Case result of {name: (error, tolerance)}: scaled_worst_result (the worst error / tolerance, tolerance 1)."""
    return scaled_worst_result({name: error for name, (error, _) in errors.items()},
                               {name: tolerance for name, (_, tolerance) in errors.items()}, **details)


# ---- sphere in a PEC shell ----
def _radial_values(function, degree, kappa, rho):
    """(F, (rho F)') of a spherical Bessel function F(kappa rho) [(rho F)' = F(x) + x F'(x), x = kappa rho]."""
    argument = kappa * rho
    value = function(degree, argument)
    return value, value + argument * function(degree, argument, derivative=True)


def _shell_rows(kind, wavenumber, radius):
    """3 x 3 matching matrix of the interior amplitude 1 and (A, B) at the interface radius and the PEC shell."""
    from scipy.special import spherical_jn, spherical_yn
    index, degree = _SPHERE_INDEX, _SPHERE_DEGREE
    inner, inner_slope = _radial_values(spherical_jn, degree, index * wavenumber, radius)
    j_a, j_a_slope = _radial_values(spherical_jn, degree, wavenumber, radius)
    y_a, y_a_slope = _radial_values(spherical_yn, degree, wavenumber, radius)
    j_b, j_b_slope = _radial_values(spherical_jn, degree, wavenumber, _SHELL_RADIUS)
    y_b, y_b_slope = _radial_values(spherical_yn, degree, wavenumber, _SHELL_RADIUS)
    if kind == 'te':   # F, (rho F)' continuous; F(b) = 0
        weight, wall = 1.0, (j_b, y_b)
    else:              # F, (rho F)' / eps continuous; (rho F)'(b) = 0
        weight, wall = 1.0 / index ** 2, (j_b_slope, y_b_slope)
    return np.array([[inner, -j_a, -y_a], [weight * inner_slope, -j_a_slope, -y_a_slope], [0.0, wall[0], wall[1]]])


def _shell_root(kind, radius, guess=None):
    """Wavenumber k [1/m] of the first matching root (guess None: scan k a over _SPHERE_SCAN), else the root within
    _SPHERE_BRACKET of guess; ValueError without a sign change."""
    from scipy.optimize import brentq

    def determinant(wavenumber):
        return float(np.linalg.det(_shell_rows(kind, wavenumber, radius)))

    if guess is None:
        low, high, count = _SPHERE_SCAN
        grid = np.linspace(low, high, count) / radius
        values = np.array([determinant(value) for value in grid])
        changes = np.flatnonzero(np.sign(values[:-1]) * np.sign(values[1:]) < 0.0)
        if not len(changes):
            raise ValueError(f'sphere shell {kind}: no root for k a in {low} .. {high}')
        bracket = (grid[changes[0]], grid[changes[0] + 1])
    else:
        bracket = (guess * (1.0 - _SPHERE_BRACKET), guess * (1.0 + _SPHERE_BRACKET))
        if determinant(bracket[0]) * determinant(bracket[1]) >= 0.0:
            raise ValueError(f'sphere shell {kind}: the shifted root left the bracket around k = {guess:g}')
    return brentq(determinant, *bracket, xtol=1e-16 * bracket[0], rtol=4.0 * np.finfo(float).eps, maxiter=200)


def _shell_field(kind, wavenumber, rho, theta, inside, amplitudes):
    """(cylindrical (E_r, E_phi, E_z) (...,3), eps (...)) of the shell mode at (rho, theta) arrays in one region."""
    from scipy.special import eval_legendre, lpmv, spherical_jn, spherical_yn
    degree = _SPHERE_DEGREE
    if inside:
        eps = _SPHERE_INDEX ** 2
        value, slope = _radial_values(spherical_jn, degree, _SPHERE_INDEX * wavenumber, rho)
    else:
        eps = 1.0
        j_value, j_slope = _radial_values(spherical_jn, degree, wavenumber, rho)
        y_value, y_slope = _radial_values(spherical_yn, degree, wavenumber, rho)
        value = amplitudes[0] * j_value + amplitudes[1] * y_value
        slope = amplitudes[0] * j_slope + amplitudes[1] * y_slope
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    legendre, angular = eval_legendre(degree, cos_theta), lpmv(1, degree, cos_theta)   # P_l, dP_l / dtheta
    field = np.zeros(np.broadcast(rho, theta).shape + (3,), dtype=np.complex128)
    if kind == 'te':
        field[..., 1] = value * angular
    else:
        radial = -degree * (degree + 1) * value * legendre / (eps * rho)
        polar = -slope * angular / (eps * rho)
        field[..., 0] = radial * sin_theta + polar * cos_theta   # e_rho = (sin, cos), e_theta = (cos, -sin) in (r, z)
        field[..., 2] = radial * cos_theta - polar * sin_theta
    return field, np.full(field.shape[:-1], eps)


def _sphere_rate(kind, wavenumber):
    """g_om [rad/s/m] of q_n = 1 on the sphere surface by optomechanics_fem on Gauss samples of the exact mode."""
    from . import optomechanics_fem
    rows = _shell_rows(kind, wavenumber, _SPHERE_RADIUS)
    amplitudes = np.linalg.solve(rows[[0, 2]][:, 1:], np.array([-rows[0, 0], 0.0]))   # (A, B): rows 0 and 2 = 0
    nodes, gauss_weights = np.polynomial.legendre.leggauss(_SPHERE_POINTS)
    theta, theta_weights = 0.5 * math.pi * (nodes + 1.0), 0.5 * math.pi * gauss_weights
    energy = 0.0j
    for low, high, inside in ((0.0, _SPHERE_RADIUS, True), (_SPHERE_RADIUS, _SHELL_RADIUS, False)):
        rho = low + 0.5 * (high - low) * (nodes + 1.0)
        rho_weights = 0.5 * (high - low) * gauss_weights
        rho_grid, theta_grid = np.meshgrid(rho, theta, indexing='ij')
        area = np.outer(rho_weights, theta_weights) * rho_grid   # dA = rho d rho d theta in the (r, z) plane
        field, eps = _shell_field(kind, wavenumber, rho_grid, theta_grid, inside, amplitudes)
        energy += optomechanics_fem.electric_energy(rho_grid * np.sin(theta_grid), area, field, eps)
    surface_rho = np.full_like(theta, _SPHERE_RADIUS)
    field_in, eps_in = _shell_field(kind, wavenumber, surface_rho, theta, True, amplitudes)
    field_out, eps_out = _shell_field(kind, wavenumber, surface_rho, theta, False, amplitudes)
    normals = np.stack([np.sin(theta), np.cos(theta)], axis=-1)
    weights = optomechanics_fem.moving_boundary_weights(_SPHERE_RADIUS * np.sin(theta), _SPHERE_RADIUS * theta_weights,
                                                        normals, field_in, field_out, eps_in, eps_out)
    overlap = optomechanics_fem.AxisymmetricOverlap(omega=C0 * wavenumber, energy=energy, boundary_weights=weights)
    uniform = optomechanics_fem.prescribed_normal_displacement(1.0, _SPHERE_RADIUS * np.sin(theta),
                                                               _SPHERE_RADIUS * np.cos(theta))
    return overlap.rate(overlap.moving_boundary(uniform))


def _check_sphere(results_by_name):
    """TE / TM moving-boundary rate of the shelled sphere vs C0 dk/da of the characteristic roots (Richardson)."""
    errors, details = {}, {}
    for kind in ('te', 'tm'):
        wavenumber = _shell_root(kind, _SPHERE_RADIUS)

        def central(step):
            upper = _shell_root(kind, _SPHERE_RADIUS + step, wavenumber)
            lower = _shell_root(kind, _SPHERE_RADIUS - step, wavenumber)
            return (upper - lower) / (2.0 * step)

        step = _SPHERE_STEP * _SPHERE_RADIUS
        expected = C0 * (4.0 * central(0.5 * step) - central(step)) / 3.0
        measured = _sphere_rate(kind, wavenumber)
        errors[kind] = (abs(measured / expected - 1.0), _SPHERE_TOLERANCE)
        details[kind] = {'k_a': wavenumber * _SPHERE_RADIUS, 'measured_d_omega_d_a': measured,
                         'expected_d_omega_d_a': expected,
                         'a_d_omega_d_a_over_omega': measured * _SPHERE_RADIUS / (C0 * wavenumber)}
    return _report(errors, details)


# ---- half-disk mesh ----
def _half_disk_mesh():
    """(TriangleMesh of the (r, z) half disk of radius b, curved P2 rings at a and b; bool silica elements)."""
    from ...fem.mesh import Curve, build_triangle_mesh
    inner_radius, outer_radius = _MESH_RADII
    inner_rings, outer_rings = _MESH_RINGS
    radii = np.concatenate([np.linspace(0.0, inner_radius, inner_rings + 1)[1:],
                            np.linspace(inner_radius, outer_radius, outer_rings + 1)[1:]])
    angles = np.linspace(0.0, math.pi, _MESH_ANGLES + 1)
    sines = np.sin(angles)
    sines[[0, -1]] = 0.0   # the axis nodes exactly on r = 0
    ring_nodes = np.stack([np.outer(radii, sines), np.outer(radii, np.cos(angles))], axis=-1).reshape(-1, 2)
    points = np.concatenate([np.zeros((1, 2)), ring_nodes])
    columns = _MESH_ANGLES + 1

    def node(ring, j):
        return 1 + ring * columns + j

    triangles = [(0, node(0, j), node(0, j + 1)) for j in range(_MESH_ANGLES)]
    for ring in range(len(radii) - 1):
        for j in range(_MESH_ANGLES):
            triangles.append((node(ring, j), node(ring + 1, j), node(ring + 1, j + 1)))
            triangles.append((node(ring, j), node(ring + 1, j + 1), node(ring, j + 1)))
    curved_pairs, curved_ids = [], []
    for curve_id, ring in enumerate((inner_rings - 1, len(radii) - 1)):
        curved_pairs += [(node(ring, j), node(ring, j + 1)) for j in range(_MESH_ANGLES)]
        curved_ids += [curve_id] * _MESH_ANGLES
    curves = tuple(Curve(0.0, 0.0, radius, radius) for radius in _MESH_RADII)
    mesh = build_triangle_mesh(points, np.array(triangles), curves=curves, curved_pairs=np.array(curved_pairs),
                               curved_ids=np.array(curved_ids), window=(0.0, -outer_radius, outer_radius, outer_radius))
    silica = np.linalg.norm(mesh.centroids(), axis=1) < inner_radius
    return mesh, silica


def _test_field(points):
    """Smooth complex cylindrical (E_r, E_phi, E_z) (..., 3) [V/m] at (r, z) points (..., 2)."""
    scale = _MESH_RADII[1]
    r, z = points[..., 0] / scale, points[..., 1] / scale
    return np.stack([(0.7 + 0.2j) * np.cos(3.0 * z) + 0.1 * r, (0.4 - 0.5j) * (1.0 + r * z),
                     (0.3 + 0.1j) * np.sin(2.0 * r) + 0.2 - 0.3j * z], axis=-1)


def _interpolated_displacement(mesh, silica, elements, reference, components):
    """((u_r, u_z) (k, q, 2), gradients (k, q, 2, 2)) of P2 interpolants of callables components(points) -> (u_r, u_z)
    on the silica elements at the reference points."""
    from ...fem.dofs import lagrange_dofs
    from ...fem.indicators import lagrange_at
    dofs = lagrange_dofs(mesh, 2, silica)
    nodal = components(dofs.dof_points())
    sampled = [lagrange_at(dofs, values, elements, reference) for values in nodal]
    return (np.stack([values for values, _ in sampled], axis=-1),
            np.stack([gradients for _, gradients in sampled], axis=-2))


def _independent_contraction(field, eps, strains, p11, p12, p44):
    """E* . Delta eps E (...) of diagonal eps (3,), cylindrical tensor strains (..., 6) written out per component."""
    s_rr, s_pp, s_zz, _, s_rz, _ = (strains[..., index] for index in range(6))
    eps_rr, eps_pp, eps_zz = eps
    inverse_rr = p11 * s_rr + p12 * (s_pp + s_zz)
    inverse_pp = p11 * s_pp + p12 * (s_rr + s_zz)
    inverse_zz = p11 * s_zz + p12 * (s_rr + s_pp)
    inverse_rz = p44 * 2.0 * s_rz
    return (-eps_rr ** 2 * inverse_rr * np.abs(field[..., 0]) ** 2 - eps_pp ** 2 * inverse_pp * np.abs(field[..., 1]) ** 2
            - eps_zz ** 2 * inverse_zz * np.abs(field[..., 2]) ** 2
            - eps_rr * eps_zz * inverse_rz * 2.0 * (np.conj(field[..., 0]) * field[..., 2]).real)


def _check_dilation(results_by_name):
    """Mesh geometry (interface area, silica volume, moving boundary of a radial D field), the dilation coefficient and
    the hoop / shear contraction on the half-disk mesh."""
    from . import optomechanics_fem
    mesh, silica = _half_disk_mesh()
    index, p11, p12 = _SILICA
    inner_radius = _MESH_RADII[0]
    element_eps = np.where(silica, index ** 2, 1.0).astype(np.complex128)
    errors, details = {}, {'elements': int(mesh.n_elements)}

    # interface: sum 2 pi r dl = 4 pi a^2; D = e_rho (|D_n| = 1 both sides), E_phi = 1, q = e_rho
    sides = optomechanics_fem.interface_sides(mesh, element_eps)
    radius = sides.points[..., 0]
    area = float(np.sum(2.0 * math.pi * radius * sides.weights))
    errors['interface_area'] = (abs(area / (4.0 * math.pi * inner_radius ** 2) - 1.0), _GEOMETRY_TOLERANCE)
    radial = sides.points / np.linalg.norm(sides.points, axis=-1, keepdims=True)
    side_fields, side_eps = [], []
    for side in (0, 1):
        eps = element_eps[sides.elements[:, side]][:, None] * np.ones(radial.shape[:-1])
        field = np.stack([radial[..., 0] / eps, np.ones_like(eps), radial[..., 1] / eps], axis=-1)
        side_fields.append(field)
        side_eps.append(eps)
    weights = optomechanics_fem.moving_boundary_weights(radius, sides.weights, sides.normals, *side_fields, *side_eps)
    moving = complex(np.sum(optomechanics_fem.normal_displacement(radial, sides.normals) * weights))
    expected_moving = EPS0 * ((index ** 2 - 1.0) - (1.0 / index ** 2 - 1.0)) * 4.0 * math.pi * inner_radius ** 2
    errors['moving_boundary_radial_d'] = (abs(moving / expected_moving - 1.0), _GEOMETRY_TOLERANCE)
    details.update(interface_edges=int(len(sides.edges)), interface_area=area)

    # volume: silica 2 pi r dA = 4/3 pi a^3; energy over every element, photoelastic samples on silica
    _, all_points, all_weights = optomechanics_fem.element_rule(mesh, np.arange(mesh.n_elements))
    all_eps = element_eps[:, None] * np.ones(all_weights.shape)
    all_field = _test_field(all_points)
    energy = optomechanics_fem.electric_energy(all_points[..., 0], all_weights, all_field, all_eps)
    elements = np.flatnonzero(silica)
    reference, points, volume_weights = optomechanics_fem.element_rule(mesh, elements)
    volume = float(np.sum(2.0 * math.pi * points[..., 0] * volume_weights))
    errors['silica_volume'] = (abs(volume / (4.0 / 3.0 * math.pi * inner_radius ** 3) - 1.0), _GEOMETRY_TOLERANCE)
    field = _test_field(points)
    omega = 2.0 * math.pi * C0 / _MESH_WAVELENGTH
    overlap = optomechanics_fem.AxisymmetricOverlap(omega=omega, energy=energy,
                                                    volume_samples=(points[..., 0], volume_weights, field, index ** 2))

    # uniform dilation u = s (r, z): S_rr = S_phiphi (hoop) = S_zz = s
    displacement, gradient = _interpolated_displacement(
        mesh, silica, elements, reference, lambda nodes: (_STRAIN * nodes[:, 0], _STRAIN * nodes[:, 1]))
    strains = optomechanics_fem.axisymmetric_strains(points[..., 0], displacement, gradient)
    coefficient = overlap.rate(overlap.photoelastic(strains, p11, p12)) / (omega * _STRAIN)
    silica_energy = complex(np.sum(2.0 * math.pi * points[..., 0] * volume_weights * EPS0 * index ** 2
                                   * np.sum(np.abs(field) ** 2, axis=-1)))
    fraction = (silica_energy / energy).real
    closed_form = 0.5 * index ** 2 * (p11 + 2.0 * p12)
    errors['dilation_identity'] = (abs(coefficient / (closed_form * fraction) - 1.0), _IDENTITY_TOLERANCE)
    errors['dilation_coefficient_0.689'] = (abs(closed_form - _DILATION_COEFFICIENT), _DILATION_ROUNDING)
    details.update(dilation_rate_over_omega_s=coefficient, silica_energy_fraction=fraction,
                   dilation_coefficient=closed_form, silica_volume=volume)

    # u = s (z, r): S_rz = s, hoop s z / r; diagonal eps, explicit p44
    displacement, gradient = _interpolated_displacement(
        mesh, silica, elements, reference, lambda nodes: (_STRAIN * nodes[:, 1], _STRAIN * nodes[:, 0]))
    strains = optomechanics_fem.axisymmetric_strains(points[..., 0], displacement, gradient)
    measured = optomechanics_fem.photoelastic_integral(points[..., 0], volume_weights, field,
                                                       np.broadcast_to(np.array(_SHEAR_EPS), field.shape), strains,
                                                       p11, p12, _SHEAR_P44)
    analytic = np.zeros(points.shape[:-1] + (6,))
    analytic[..., 1] = _STRAIN * points[..., 1] / points[..., 0]
    analytic[..., 4] = _STRAIN
    expected = complex(np.sum(2.0 * math.pi * points[..., 0] * volume_weights * EPS0 * _independent_contraction(
        field, _SHEAR_EPS, analytic, p11, p12, _SHEAR_P44)))
    errors['shear_hoop_contraction'] = (abs(measured - expected) / abs(expected), _IDENTITY_TOLERANCE)
    details.update(shear_measured=[measured.real, measured.imag], shear_expected=[expected.real, expected.imag])
    return _report(errors, details)


# ---- 3D ring average ----
def _ring_displacement(points):
    """Cartesian linear test displacement (..., 3) [m] at 3D points (..., 3) (_RING_FIELD about _RING_AXIS)."""
    s1, s2, s3, translation, rotation, shear = _RING_FIELD
    x, y, z = (points[..., axis] - _RING_AXIS[axis] for axis in range(3))
    return np.stack([s1 * x + s2 * x + translation - rotation * y + shear * z, s1 * y - s2 * y + rotation * x,
                     s3 * z + shear * x], axis=-1)


def _check_ring(results_by_name):
    """<q . n>_phi, cylindrical vectors / strains per phi and their average of the linear 3D field vs closed forms."""
    from . import optomechanics_fem
    s1, s2, s3, translation, rotation, shear = _RING_FIELD
    radius, height = np.array(_RING_RADII), np.array(_RING_HEIGHTS)
    normals = np.stack([np.cos(_RING_ANGLES), np.sin(_RING_ANGLES)], axis=-1)
    points, phi = optomechanics_fem.ring_points(radius, height, _RING_POINTS, _RING_AXIS)
    displacement = _ring_displacement(points)
    scale = max(abs(value) for value in (s1, s2, s3, rotation, shear)) * float(np.max(radius))
    errors = {}

    averaged = optomechanics_fem.ring_average_normal_displacement(displacement, normals, phi)
    expected = s1 * radius * normals[:, 0] + s3 * height * normals[:, 1]
    errors['average_normal_displacement'] = (float(np.max(np.abs(averaged - expected))) / scale, _RING_TOLERANCE)

    r, h, cos_phi, sin_phi = radius[:, None], height[:, None], np.cos(phi), np.sin(phi)
    cylindrical = optomechanics_fem.cylindrical_vectors(displacement, phi)
    expected_vectors = np.stack(np.broadcast_arrays(
        s1 * r + s2 * r * np.cos(2.0 * phi) + (translation + shear * h) * cos_phi,
        -s2 * r * np.sin(2.0 * phi) - (translation + shear * h) * sin_phi + rotation * r,
        s3 * h + shear * r * cos_phi), axis=-1)
    errors['cylindrical_vectors'] = (float(np.max(np.abs(cylindrical - expected_vectors))) / scale, _RING_TOLERANCE)

    cartesian = np.zeros(points.shape[:-1] + (6,))   # (xx, yy, zz, yz, xz, xy) tensor of the linear field
    cartesian[..., 0], cartesian[..., 1], cartesian[..., 2], cartesian[..., 4] = s1 + s2, s1 - s2, s3, shear
    strains = optomechanics_fem.cylindrical_strains(cartesian, phi)
    zero = np.zeros_like(phi)
    expected_strains = np.stack(np.broadcast_arrays(
        s1 + s2 * np.cos(2.0 * phi), s1 - s2 * np.cos(2.0 * phi), s3 + zero, -shear * sin_phi, shear * cos_phi,
        -s2 * np.sin(2.0 * phi)), axis=-1)
    strain_scale = max(abs(value) for value in (s1, s2, s3, shear))
    errors['cylindrical_strains'] = (float(np.max(np.abs(strains - expected_strains))) / strain_scale, _RING_TOLERANCE)
    average = optomechanics_fem.ring_average_strains(cartesian, phi)
    expected_average = np.array([s1, s1, s3, 0.0, 0.0, 0.0])
    errors['average_strains'] = (float(np.max(np.abs(average - expected_average))) / strain_scale, _RING_TOLERANCE)
    return _report(errors, {'ring_points': _RING_POINTS, 'samples': len(radius)})


OPTOMECHANICS_FEM_CASES = (
    SelfTestCase(
        name='om_fem_sphere_moving_boundary',
        label='Optomechanics FEM: moving boundary of a shelled sphere vs its characteristic roots',
        description=f'CPU: dielectric sphere (n {_SPHERE_INDEX:g}, radius {_SPHERE_RADIUS * 1e6:g} um) in a PEC shell '
                    f'(radius {_SHELL_RADIUS * 1e6:g} um), m = 0 TE and TM modes of degree l = {_SPHERE_DEGREE} '
                    '(exact spherical Bessel fields, first root of the matching determinant). The 2 pi r weighted '
                    'moving-boundary rate (optomechanics_fem.moving_boundary_weights, electric_energy, '
                    'AxisymmetricOverlap) of a uniform q . n = 1 on plain Gauss samples in (rho, theta) must equal '
                    'd omega / d a = C0 dk/da of the characteristic roots (Richardson central differences, b fixed; '
                    f'relative {_SPHERE_TOLERANCE:g}). TE pins Delta eps |E_par|^2 with E_phi, TM the '
                    '-Delta(1/eps) |D_n|^2 term and the normal / tangential split of (E_r, E_z).',
        specs=no_runs, check=_check_sphere),
    SelfTestCase(
        name='om_fem_photoelastic_dilation',
        label='Optomechanics FEM: photoelastic dilation coefficient and hoop strain on a curved mesh',
        description=f'CPU: hand-built (r, z) half-disk triangle mesh ({_MESH_ANGLES} segments over theta, curved P2 '
                    f'rings at {_MESH_RADII[0] * 1e6:g} / {_MESH_RADII[1] * 1e6:g} um), silica (n {_SILICA[0]:g}, '
                    f'p11 {_SILICA[1]:g}, p12 {_SILICA[2]:g}) inside the inner ring. interface_sides and element_rule '
                    'give 2 pi r sums = the sphere area / volume and the moving boundary of a radial D field '
                    f'(relative {_GEOMETRY_TOLERANCE:g}); a uniform dilation u = s (r, z) (P2 interpolant, hoop u_r / '
                    'r) gives g_pe / (omega s) = n^2 / 2 (p11 + 2 p12) x the silica energy fraction of an arbitrary '
                    f'field (relative {_IDENTITY_TOLERANCE:g}; the coefficient {_DILATION_COEFFICIENT:g} within '
                    f'{_DILATION_ROUNDING:g}); u = s (z, r) with diagonal eps {_SHEAR_EPS} and p44 {_SHEAR_P44:g} '
                    'matches an independent hoop / rz contraction.',
        specs=no_runs, check=_check_dilation),
    SelfTestCase(
        name='om_fem_ring_average',
        label='Optomechanics FEM: 3D ring average of q . n and cylindrical strains',
        description=f'CPU: a linear 3D displacement (breathing, m = 2 crown, axial stretch, translation, rotation '
                    f'about z, xz shear) about an off-origin axis, sampled on ring_points ({_RING_POINTS} angles): '
                    'ring_average_normal_displacement must equal s1 r n_r + s3 z n_z (the crown, translation, '
                    'rotation and shear parts average out), cylindrical_vectors / cylindrical_strains the polar closed '
                    f'forms per phi and ring_average_strains (s1, s1, s3, 0, 0, 0) (relative {_RING_TOLERANCE:g}).',
        specs=no_runs, check=_check_ring),
)
