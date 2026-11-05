"""Squeeze-film self-tests (new_modules_plan S6; CPU, specs() empty, check({}), builtin Kuhn meshes, no gmsh): the
fem.surface_mesh footprint and P1 / P2 displacement transfer with the compressible strip closed form, and the vented
annulus of Basiri E4 (rigid piston c_sq, mode-weighted gamma_gas of the Kirchhoff bracket shapes). error = number of
failed sub-checks (tolerance 0), values in details. definition3d registers them."""
import math

import numpy as np

from ...fem.selftests_harmonic import _checker
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import checks_result

UM = 1e-6
# air and the Basiri microdisk (plan section 2): mu [Pa s], p_a [Pa], mean free path [m]; annulus radii a, b and
# thickness t [m]; SiO2 E [Pa], nu, rho [kg/m^3]
_VISCOSITY = 1.81e-5
_AMBIENT_PRESSURE = 101325.0
_MEAN_FREE_PATH = 68e-9
_OUTER = 148.0 * UM
_INNER = 82.0 * UM
_THICKNESS = 1.8 * UM
_YOUNGS_MODULUS = 73e9
_POISSON_RATIO = 0.17
_DENSITY = 2200.0
# footprint: one 2 pi / 16 sector of the annulus (sealed radial sides = symmetry planes; totals x 16), 8 radial x 12
# angular P2 cells: a 192-gon, whose chords move c_sq by ~ -4e-4 against the circles (estimate)
_SECTORS = 16
_RADIAL_CELLS = 8
_ANGULAR_CELLS = 12
_PISTON_TOLERANCE = 5e-3   # plan S6 / Basiri E4: rigid annular piston within 0.5 %
_PISTON_TABLE = {10.0 * UM: 3.782e-6, 20.0 * UM: 4.727e-7}   # Basiri section 6 c_sq [kg/s]
_TABLE_TOLERANCE = 1e-3    # closed form vs the 4-digit table
# Basiri section 6 brackets, lowest axisymmetric mode at H = 10 um (max |psi| = 1): f [Hz], m_eff [kg], zeta,
# gamma_gas [1/s]
_BRACKETS = {'free_free': (225.0e3, 49.9e-12, 0.0, 4235.0),
             'clamped': (356.0e3, 58.0e-12, 0.457, 12239.0)}
_MODE_GAP = 10.0 * UM
_FREQUENCY_TOLERANCE = 1e-3   # Kirchhoff roots vs the 4-digit table
_MASS_TOLERANCE = 1e-2        # m_eff / zeta vs the 3-digit table
_ZETA_FREE_TOLERANCE = 1e-6   # free-free: int psi dA = 0 exactly (momentum)
_RADIAL_TOLERANCE = 5e-3      # FE c_psi vs the 1D radial integral of the same shape (plan 0.5 %)
_GAMMA_TOLERANCE = 1e-2       # FE gamma_gas vs the 4-digit table (first estimate)
_RADIAL_SAMPLES = 20001       # trapezoid samples of the 1D radial references over [b, a]
_ROOT_GRID = np.linspace(0.5, 8.0, 1501)   # lambda = k a scanned for the Kirchhoff roots
_SHEAR_SIGNS = np.array([-1.0, -1.0, 1.0, 1.0])   # m = 0 shear V_r ~ -k^2 R' (J0, Y0), +k^2 R' (I0, K0)
# strip: L (vented x sides) x W footprint of an L x W x T plate, 16 x 4 Kuhn cells
_STRIP = (100.0 * UM, 20.0 * UM, 2.0 * UM)
_STRIP_CELLS = (16, 4)
_STRIP_GAP = 5.0 * UM
_STRIP_SQUEEZE = 10.0           # sigma of the compressible run
_STRIP_TOLERANCE = 1e-3         # complex force / c / k vs the closed form (first estimate)
_LOW_SQUEEZE = 1e-4             # sigma of the near-incompressible run
_STRIP_KNUDSEN = 0.0136
_EXACT_RTOL = 1e-10             # identities and fields the element space holds


# ---- 1D references ----
def _cumulative(values, radii):
    """Cumulative trapezoid integral of values over radii (0 at the first)."""
    return np.concatenate([[0.0], np.cumsum(0.5 * (values[1:] + values[:-1]) * np.diff(radii))])


def _integral(values, radii) -> float:
    return float(_cumulative(values, radii)[-1])


def _radial_damping(shape, radii, gap) -> float:
    """Incompressible c_psi [kg/s] of psi(r) vented at both rims: -(12 mu / H^3) int P psi 2 pi r dr with
    (1 / r)(r P')' = psi, P(b) = P(a) = 0 (Basiri section 4)."""
    enclosed = _cumulative(shape * radii, radii)
    particular = _cumulative(enclosed / radii, radii)
    pressure = particular - particular[-1] * np.log(radii / radii[0]) / math.log(radii[-1] / radii[0])
    return -12.0 * _VISCOSITY / gap ** 3 * 2.0 * math.pi * _integral(pressure * shape * radii, radii)


def _annulus_moment() -> float:
    """G = (a^4 - b^4) - (a^2 - b^2)^2 / ln(a / b) [m^4]."""
    return (_OUTER ** 4 - _INNER ** 4) - (_OUTER ** 2 - _INNER ** 2) ** 2 / math.log(_OUTER / _INNER)


# ---- Kirchhoff annular plate, m = 0 (Basiri section 4) ----
def _plate_wave_scale() -> float:
    """sqrt(D / (rho t)) [m^2/s]: omega = k^2 x this."""
    rigidity = _YOUNGS_MODULUS * _THICKNESS ** 3 / (12.0 * (1.0 - _POISSON_RATIO ** 2))
    return math.sqrt(rigidity / (_DENSITY * _THICKNESS))


def _bessel_columns(x):
    """(values, d / dx, d^2 / dx^2), each (4, ...), of J0, Y0, I0, K0 at x."""
    from scipy.special import iv, jv, kv, yv
    x = np.asarray(x, dtype=np.float64)
    j0, j1, y0, y1 = jv(0, x), jv(1, x), yv(0, x), yv(1, x)
    i0, i1, k0, k1 = iv(0, x), iv(1, x), kv(0, x), kv(1, x)
    return (np.stack([j0, y0, i0, k0]), np.stack([-j1, -y1, i1, -k1]),
            np.stack([-j0 + j1 / x, -y0 + y1 / x, i0 - i1 / x, k0 + k1 / x]))


def _edge_rows(x, clamped) -> list:
    """Clamped (R, R') or free (M_r ~ R'' + nu R' / r, V_r) rows of the four columns at x = k r."""
    values, first, second = _bessel_columns(x)
    if clamped:
        return [values, first]
    return [second + _POISSON_RATIO * first / x, _SHEAR_SIGNS * first]


def _plate_matrix(wavenumber_radius, clamped_inner) -> tuple:
    """(column-scaled 4 x 4 edge matrix, column scales) at lambda = k a: free at a, free or clamped at b."""
    rows = _edge_rows(wavenumber_radius, False) + _edge_rows(wavenumber_radius * _INNER / _OUTER, clamped_inner)
    matrix = np.array(rows, dtype=np.float64)
    scales = np.abs(matrix).max(axis=0)
    return matrix / scales, scales


def _kirchhoff_mode(clamped_inner, expected_frequency) -> tuple:
    """(frequency [Hz], k [1/m], column coefficients with max |psi| = 1 on [b, a]) of the m = 0 root nearest the
    expected frequency."""
    from scipy.optimize import brentq

    def determinant(wavenumber_radius):
        return float(np.linalg.det(_plate_matrix(wavenumber_radius, clamped_inner)[0]))

    wave_scale = _plate_wave_scale()
    expected = _OUTER * math.sqrt(2.0 * math.pi * expected_frequency / wave_scale)
    values = np.array([determinant(value) for value in _ROOT_GRID])
    changes = np.flatnonzero(np.sign(values[1:]) * np.sign(values[:-1]) < 0.0)
    if not len(changes):
        raise RuntimeError('Kirchhoff annulus: no m = 0 root in the scanned lambda range')
    roots = [brentq(determinant, _ROOT_GRID[index], _ROOT_GRID[index + 1], xtol=1e-14, rtol=1e-14)
             for index in changes]
    root = min(roots, key=lambda value: abs(value - expected))
    matrix, scales = _plate_matrix(root, clamped_inner)
    coefficients = np.linalg.svd(matrix)[2][-1] / scales
    wavenumber = root / _OUTER
    shape = coefficients @ _bessel_columns(wavenumber * np.linspace(_INNER, _OUTER, _RADIAL_SAMPLES))[0]
    coefficients = coefficients / shape[np.argmax(np.abs(shape))]
    return wavenumber * wavenumber * wave_scale / (2.0 * math.pi), wavenumber, coefficients


# ---- meshes ----
def _annulus_sector_mesh():
    """Kuhn TetrahedronMesh of one 2 pi / _SECTORS sector of the b < r < a, 0 < z < t annulus (a box in (r, a theta, z)
    mapped onto the plane: nodes on the rim circles, straight chords between)."""
    from ...fem.mesh3d import build_tetrahedron_mesh, structured_box_mesh
    sector = 2.0 * math.pi / _SECTORS
    logical = structured_box_mesh(np.linspace(_INNER, _OUTER, _RADIAL_CELLS + 1),
                                  np.linspace(0.0, sector * _OUTER, _ANGULAR_CELLS + 1), np.array([0.0, _THICKNESS]))
    radius, arc, height = logical.points.T
    angle = arc / _OUTER
    points = np.column_stack([radius * np.cos(angle), radius * np.sin(angle), height])
    return build_tetrahedron_mesh(points, logical.tets)


def _surface_dof_points_3d(surface) -> np.ndarray:
    """(n_points + n_edges, 3) points of the surface P2 dofs on its plane (normal z)."""
    planar = np.concatenate([surface.mesh.points, surface.mesh.midside_points])
    return np.column_stack([planar, np.full(len(planar), surface.plane)])


def _refused(build) -> float:
    """0 when build() raises ValueError, else 1."""
    try:
        build()
    except ValueError:
        return 0.0
    return 1.0


# ---- strip: surface mesh, transfer, compressible closed form ----
def _check_strip(_):
    from ...fem.dofs3d import lagrange_dofs_3d
    from ...fem.mesh3d import structured_box_mesh
    from ...fem.surface_mesh import (boundary_edges_where, lagrange_field_function, normal_displacement,
                                     surface_triangle_mesh)
    from ...fem.squeeze_film import SqueezeFilm, squeeze_number

    checks, add = _checker()
    length, width, thickness = _STRIP
    n_x, n_y = _STRIP_CELLS
    tet_mesh = structured_box_mesh(np.linspace(0.0, length, n_x + 1), np.linspace(0.0, width, n_y + 1),
                                   np.array([0.0, thickness]))
    bottom = surface_triangle_mesh(tet_mesh, tet_mesh.face_tags['z_lo'])
    top = surface_triangle_mesh(tet_mesh, tet_mesh.face_tags['z_hi'])
    mesh = bottom.mesh
    add('bottom_axis_and_normal', abs(bottom.axis - 2) + abs(bottom.outward_sign + 1.0), 0.0)
    add('top_axis_and_normal', abs(top.axis - 2) + abs(top.outward_sign - 1.0), 0.0)
    add('footprint_area', abs(float(np.abs(mesh.areas()).sum()) / (length * width) - 1.0), _EXACT_RTOL)
    node_error = float(np.abs(mesh.points - tet_mesh.points[bottom.nodes][:, :2]).max())
    edge_error = float(np.abs(bottom.nodes[mesh.edges] - tet_mesh.edges[bottom.edges]).max())
    face_error = float(np.abs(np.sort(bottom.nodes[mesh.triangles], axis=1) - tet_mesh.faces[bottom.faces]).max())
    add('surface_maps', node_error + edge_error + face_error, 0.0)
    add('refuses_two_planes', _refused(lambda: surface_triangle_mesh(
        tet_mesh, np.concatenate([tet_mesh.face_tags['z_lo'], tet_mesh.face_tags['z_hi']]), axis=2)), 0.0)
    add('refuses_bent_selection', _refused(lambda: surface_triangle_mesh(
        tet_mesh, np.concatenate([tet_mesh.face_tags['z_lo'], tet_mesh.face_tags['x_lo']]))), 0.0)

    def quadratic(points):
        x, y, z = points[:, 0] / length, points[:, 1] / width, points[:, 2] / thickness
        return 1.0 + 2.0 * x - y + 3.0 * x * y + x * x - 0.5 * y * y + 0.7 * z + 0.4 * x * z

    dof_points = lagrange_dofs_3d(tet_mesh, 2).dof_points()
    displacement = np.column_stack([np.sin(dof_points[:, 0] / length), np.cos(dof_points[:, 1] / width),
                                    quadratic(dof_points)])
    w_bottom = normal_displacement(bottom, displacement.ravel())   # interleaved 3 d + component
    scale = float(np.abs(quadratic(dof_points)).max())
    add('transfer_p2_bottom', np.abs(w_bottom - quadratic(_surface_dof_points_3d(bottom))).max() / scale, _EXACT_RTOL)
    add('transfer_p2_top', np.abs(normal_displacement(top, displacement)
                                  + quadratic(_surface_dof_points_3d(top))).max() / scale, _EXACT_RTOL)
    node_points = np.column_stack([mesh.points, np.zeros(mesh.n_points)])
    nodal = np.column_stack([np.zeros(tet_mesh.n_points), np.zeros(tet_mesh.n_points), quadratic(tet_mesh.points)])
    add('transfer_p1', np.abs(normal_displacement(bottom, nodal) - quadratic(node_points)).max() / scale, _EXACT_RTOL)
    add('transfer_p2_to_p1', np.abs(normal_displacement(bottom, displacement, order=1)
                                    - quadratic(node_points)).max() / scale, _EXACT_RTOL)
    corners = mesh.points[mesh.triangles]
    reference = np.array([0.2, 0.3])
    inner = corners[:, 0] + (corners[:, 1] - corners[:, 0]) * reference[0] + (corners[:, 2] - corners[:, 0]) * reference[1]
    evaluated = lagrange_field_function(mesh, w_bottom)(inner[:, None, :], np.arange(mesh.n_elements))[:, 0]
    add('field_at_points', np.abs(evaluated - quadratic(np.column_stack([inner, np.zeros(len(inner))]))).max() / scale,
        _EXACT_RTOL)

    side = 1e-9 * length
    vents = boundary_edges_where(mesh, lambda points: (points[:, 0] <= side) | (points[:, 0] >= length - side))
    add('vent_count', abs(len(vents) - 2 * n_y), 0.0)
    lift = np.zeros_like(displacement)
    lift[:, 2] = 1.0
    w_rigid = normal_displacement(bottom, lift)
    add('rigid_transfer', np.abs(w_rigid - 1.0).max(), 0.0)
    gap = _STRIP_GAP
    incompressible = SqueezeFilm(mesh, gap, _VISCOSITY, None, slip='none', vent_edges=vents)
    rigid = incompressible.solve(w_rigid, 0.0)
    strip_damping = _VISCOSITY * width * length ** 3 / gap ** 3   # mu W L^3 / H^3: the quadratic p is in P2
    add('strip_incompressible', abs(rigid.damping / strip_damping - 1.0), 1e-9)
    add('constant_shape_number', abs(incompressible.solve(1.0, 0.0).force / rigid.force - 1.0), _EXACT_RTOL)
    per_element = SqueezeFilm(mesh, np.full(mesh.n_elements, gap), _VISCOSITY, None, slip='none', vent_edges=vents)
    add('per_element_gap', abs(per_element.solve(w_rigid, 0.0).damping / rigid.damping - 1.0), _EXACT_RTOL)
    slip = SqueezeFilm(mesh, gap, _VISCOSITY, None, mean_free_path=_STRIP_KNUDSEN * gap, slip='veijola',
                       vent_edges=vents)
    add('veijola_slip', abs(slip.solve(w_rigid, 0.0).damping / rigid.damping
                            * (1.0 + 9.638 * _STRIP_KNUDSEN ** 1.159) - 1.0), 1e-9)

    conductance, compressibility = gap ** 3 / (12.0 * _VISCOSITY), gap / _AMBIENT_PRESSURE
    omega = _STRIP_SQUEEZE * _AMBIENT_PRESSURE * gap ** 2 / (12.0 * _VISCOSITY * length ** 2)
    add('squeeze_number', abs(squeeze_number(_VISCOSITY, omega, length, _AMBIENT_PRESSURE, gap) / _STRIP_SQUEEZE - 1.0),
        1e-12)
    film = SqueezeFilm(mesh, gap, _VISCOSITY, _AMBIENT_PRESSURE, slip='none', vent_edges=vents)
    response = film.solve(w_rigid, omega)
    # a p'' - j omega c p = 1 (unit velocity), p(0) = p(L) = 0: F = -W / (j omega c) (L - (2 / kappa) tanh(kappa L / 2))
    kappa = np.sqrt(1j * omega * compressibility / conductance)
    exact_force = -width / (1j * omega * compressibility) * (length - 2.0 / kappa * np.tanh(kappa * length / 2.0))
    exact_damping, exact_stiffness = -exact_force.real, omega * exact_force.imag
    add('strip_compressible_force', abs(response.force - exact_force) / abs(exact_force), _STRIP_TOLERANCE)
    add('strip_compressible_damping', abs(response.damping / exact_damping - 1.0), _STRIP_TOLERANCE)
    add('strip_compressible_stiffness', abs(response.stiffness / exact_stiffness - 1.0), _STRIP_TOLERANCE)
    low = film.solve(w_rigid, omega * _LOW_SQUEEZE / _STRIP_SQUEEZE)
    add('strip_low_squeeze_limit', abs(low.damping / rigid.damping - 1.0), 1e-6)
    compressible = {'force_re': float(response.force.real), 'force_im': float(response.force.imag),
                    'exact_re': float(exact_force.real), 'exact_im': float(exact_force.imag)}
    return checks_result(checks, compressible=compressible,
                         incompressible_damping=float(rigid.damping), strip_damping=float(strip_damping))


# ---- vented annulus: rigid piston and the Kirchhoff bracket shapes (Basiri E4) ----
def _check_annulus(_):
    from ...fem.surface_mesh import boundary_edges_where, normal_displacement, surface_triangle_mesh
    from ...fem.squeeze_film import SqueezeFilm, mode_gas_damping, rigid_annulus_damping

    checks, add = _checker()
    tet_mesh = _annulus_sector_mesh()
    surface = surface_triangle_mesh(tet_mesh, tet_mesh.face_tags['z_lo'])
    mesh = surface.mesh
    sector = 2.0 * math.pi / _SECTORS
    polygon_area = 0.5 * (_OUTER ** 2 - _INNER ** 2) * math.sin(sector / _ANGULAR_CELLS) * _ANGULAR_CELLS
    add('footprint_area', abs(float(np.abs(mesh.areas()).sum()) / polygon_area - 1.0), _EXACT_RTOL)

    def on_rims(points):
        radius = np.hypot(points[:, 0], points[:, 1])
        return (np.abs(radius - _OUTER) <= 1e-6 * _OUTER) | (np.abs(radius - _INNER) <= 1e-6 * _OUTER)

    vents = boundary_edges_where(mesh, on_rims)
    add('rim_vents', abs(len(vents) - 2 * _ANGULAR_CELLS), 0.0)
    radii = np.linspace(_INNER, _OUTER, _RADIAL_SAMPLES)
    lift = np.zeros((tet_mesh.n_points + tet_mesh.n_edges, 3))
    lift[:, 2] = 1.0
    w_rigid = normal_displacement(surface, lift)
    pistons = {}
    for gap, table in _PISTON_TABLE.items():
        tag = f'{gap / UM:g}um'
        closed = 3.0 * math.pi * _VISCOSITY * _annulus_moment() / (2.0 * gap ** 3)
        film = SqueezeFilm(mesh, gap, _VISCOSITY, None, slip='none', vent_edges=vents)
        damping = _SECTORS * film.solve(w_rigid, 0.0).damping
        add(f'piston_{tag}', abs(damping / closed - 1.0), _PISTON_TOLERANCE)
        add(f'piston_{tag}_table', abs(closed / table - 1.0), _TABLE_TOLERANCE)
        add(f'piston_{tag}_function', abs(rigid_annulus_damping(_VISCOSITY, gap, _OUTER, _INNER) / closed - 1.0),
            1e-12)
        add(f'piston_{tag}_radial', abs(_radial_damping(np.ones_like(radii), radii, gap) / closed - 1.0), 1e-6)
        pistons[tag] = {'fem': float(damping), 'closed_form': float(closed), 'table': table}

    film = SqueezeFilm(mesh, _MODE_GAP, _VISCOSITY, None, slip='none', vent_edges=vents)
    gas = SqueezeFilm(mesh, _MODE_GAP, _VISCOSITY, _AMBIENT_PRESSURE, mean_free_path=_MEAN_FREE_PATH,
                      slip='veijola', vent_edges=vents)
    annulus_area = math.pi * (_OUTER ** 2 - _INNER ** 2)
    shapes, omegas, sector_masses, modes = [], [], [], {}
    for name, (frequency_table, mass_table, zeta_table, _) in _BRACKETS.items():
        frequency, wavenumber, coefficients = _kirchhoff_mode(name == 'clamped', frequency_table)
        shape = coefficients @ _bessel_columns(wavenumber * radii)[0]
        mass = _DENSITY * _THICKNESS * 2.0 * math.pi * _integral(shape * shape * radii, radii)
        zeta = abs(2.0 * math.pi * _integral(shape * radii, radii)) / annulus_area
        add(f'{name}_frequency', abs(frequency / frequency_table - 1.0), _FREQUENCY_TOLERANCE)
        add(f'{name}_effective_mass', abs(mass / mass_table - 1.0), _MASS_TOLERANCE)
        if zeta_table == 0.0:
            add(f'{name}_zeta', zeta, _ZETA_FREE_TOLERANCE)
        else:
            add(f'{name}_zeta', abs(zeta / zeta_table - 1.0), _MASS_TOLERANCE)

        def psi(points, elements, wavenumber=wavenumber, coefficients=coefficients):
            radius = np.hypot(points[..., 0], points[..., 1])
            return np.tensordot(coefficients, _bessel_columns(wavenumber * radius)[0], axes=1)

        shapes.append(psi)
        omegas.append(2.0 * math.pi * frequency)
        sector_masses.append(mass / _SECTORS)
        modes[name] = {'frequency': float(frequency), 'effective_mass': float(mass), 'zeta': float(zeta),
                       'psi_outer': float(shape[-1]), 'psi_inner': float(shape[0]),
                       'radial_damping': _radial_damping(shape, radii, _MODE_GAP)}
    rows = mode_gas_damping(film, shapes, omegas, sector_masses)
    gas_rows = mode_gas_damping(gas, shapes, omegas, sector_masses)
    for (name, (_, _, _, gamma_table)), row, gas_row in zip(_BRACKETS.items(), rows, gas_rows):
        mode = modes[name]
        damping = _SECTORS * row['gas_damping']
        add(f'{name}_damping_vs_radial', abs(damping / mode['radial_damping'] - 1.0), _RADIAL_TOLERANCE)
        add(f'{name}_gamma_gas', abs(row['gamma_gas'] / gamma_table - 1.0), _GAMMA_TOLERANCE)
        mode.update(fem_damping=float(damping), gamma_gas=float(row['gamma_gas']), q_gas=float(row['q_gas']),
                    gamma_table=gamma_table, gamma_gas_compressible_slip=float(gas_row['gamma_gas']),
                    gas_stiffness_compressible_slip=float(_SECTORS * gas_row['gas_stiffness']))
    return checks_result(checks, pistons=pistons, modes=modes, n_surface_elements=int(mesh.n_elements))


SQUEEZE_FILM_CASES = (
    SelfTestCase(
        name='squeeze_film_strip',
        label='Squeeze film: surface mesh, transfer and compressible strip',
        description='CPU checks of fem.surface_mesh and fem.squeeze_film (no model) on a 100 x 20 x 2 um '
                    'Kuhn plate: the z_lo / z_hi footprints (normal -z / +z, area, node / edge / face maps, refusals of '
                    'two planes and of a bent selection), the P2 / P1 trace of a quadratic u_z as w = -u . n_out '
                    '(exact, 1e-10), the field at element points; a rigid lift over a H = 5 um gap vented at x = 0 / L: '
                    'incompressible c = mu W L^3 / H^3 (1e-9), Veijola slip 1 / (1 + 9.638 Kn^1.159), and at squeeze '
                    'number 10 the complex force -W / (j omega c) (L - (2 / kappa) tanh(kappa L / 2)), kappa^2 = j '
                    'omega 12 mu / (H^2 p_a), within 1e-3 (first estimate). error = number of failed sub-checks, '
                    'tolerance 0.',
        specs=lambda: {},
        check=_check_strip,
        max_wall_time=60.0),
    SelfTestCase(
        name='squeeze_film_vented_annulus',
        label='Squeeze film: vented annulus damping (Basiri E4)',
        description='CPU checks of fem.squeeze_film on the bottom footprint of one 2 pi / 16 sector of the '
                    'Basiri annulus (a = 148, b = 82, t = 1.8 um; 8 x 12 P2 cells, rims vented, radial sides sealed): '
                    'the rigid piston c_sq = 3 pi mu G / (2 H^3), G = (a^4 - b^4) - (a^2 - b^2)^2 / ln(a / b), at H = '
                    '10 / 20 um within 0.5 % (plan S6) and the table values; the m = 0 Kirchhoff bracket shapes (free-'
                    'free 225.0 kHz, clamped at b 356.0 kHz; m_eff 49.9 / 58.0 ng, zeta 0 / 0.457) give c_psi within '
                    '0.5 % of the 1D radial integral and gamma_gas = c_psi / m_eff = 4235 / 12239 1/s at H = 10 um '
                    'within 1e-2 (first estimate). error = number of failed sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_annulus,
        max_wall_time=60.0),
)
