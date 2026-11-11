"""CPU self-tests of the 2D FDFD on polar (r, phi) Domains (symmetry plan 2.16 item 5 / 2.18; scripted FDFDPhysics on
the cpu backend): PEC disk cavity and dielectric whispering-gallery resonances vs Bessel / Hankel closed forms, a
ring of phased line currents in a C_4 sector with an angular Bloch phase and a cylindrical PML."""
# Closed forms are semi-discrete: a phasor e^{-j m phi} on uniform angle cells dphi is an exact eigenvector of the
# discrete angular operator with order nu = (2 / dphi) sin(m dphi / 2) (m_eff), so the Bessel functions take that
# non-integer order and only the radial discretisation (second order) and the PML remain as errors.
import cmath
import math

import numpy as np

from ...core.constants import C0, EPS0, MU0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, max_relative_error
from .selftests import tolerance_units_result

# ---- PEC disk cavity: full disk from the origin, 2 pi periodic, pec outer wall (ez: E_z = 0 in the outermost cells,
# so the wall is the centre of the last cell; hz: no flux through the inner face of the last cell, so the wall is
# that face, as the Cartesian FDFD walls)
_DISK_RADIUS = 1e-6                 # m, window radius
_DISK_RADIAL_CELLS = 50
_DISK_ANGLE_CELLS = 32
_DISK_ORDERS = (0, 1, 3)
_TARGET_OFFSET = 1.005              # target omega = closed form x this (off the resonance: a regular shift)
_DISK_TOLERANCE = 3e-3              # relative omega; estimated (k dr)^2 / 24 ~ 7e-4 at m = 3

# ---- whispering-gallery disk (n = 2, radius R) in vacuum, one C_16 sector with the complex Bloch phase of order m
_WGM_INDEX = 2.0
_WGM_RADIUS = 1e-6                  # m, dielectric radius (a cell face)
_WGM_DISK_CELLS = 50                # uniform dr inside the disk
_WGM_WINDOW_CELLS = 110             # window to 2.2 R, then the cylindrical PML
_WGM_PML_CELLS = 24
_WGM_SECTORS = 16
_WGM_SECTOR_CELLS = 8
_WGM_ORDER = 6                      # phase 6 x 2 pi / 16 = 3 pi / 4 across the sector
_WGM_OMEGA_TOLERANCE = 3e-3         # relative Re omega; estimated
_WGM_Q_TOLERANCE = 5e-2             # relative Q; estimated (Q follows k exponentially)

# ---- ring of N phased line currents I e^{-j m phi_j} (one per angle cell of column i0) in a C_4 sector, vacuum
_RING_WAVELENGTH = 1e-6             # m
_RING_RADIAL_CELLS = 90             # dr = lambda / 60 to 1.5 lambda
_RING_PML_CELLS = 20
_RING_SECTORS = 4
_RING_SECTOR_CELLS = 12
_RING_ORDER = 3                     # Bloch phase 3 pi / 2
_RING_SOURCE_COLUMN = 30
_RING_FLUX_COLUMNS = (40, 70)       # hi faces of these columns: lossless, before the PML
_RING_CURRENT = 1e-3                # A (ez) / V (hz) per line current
_RING_EXCLUDED = 3                  # cells beside the source column left out of the field comparison
_RING_FIELD_TOLERANCE = 2e-2        # estimated: radial phase error k L (k dr)^2 / 24 ~ 3e-3, PML reflection
_RING_FLUX_TOLERANCE = 2e-2         # closed-form power; estimated
_EXACT_TOLERANCE = 1e-8             # discrete power balance (solver precision)


def _discrete_order(order, d_angle):
    """Semi-discrete angular order nu = (2 / dphi) sin(m dphi / 2) of e^{-j m phi} on uniform cells dphi."""
    return 2.0 / d_angle * math.sin(0.5 * order * d_angle)


def _polar_fdfd(polarization, dr_cells, pml_cells, angle_cells, d_angle, faces, angle_wrap, bloch_order, wavelength):
    """(FDFDPhysics, Domain) on a cpu polar Domain from the origin: window radial widths dr_cells, pml_cells cpml cells
    on x_hi (of the last width), uniform angle cells; bloch_order = azimuthal order of bloch phi faces."""
    from ...core.domain import Domain
    from ...core.material_grid import MaterialGrid
    from ...core.mesh import NonUniformGrid2D
    from ...core.metric import Coordinates
    from .boundaries import BoundaryConfig
    from .solvers.fdfd_2d import FDFDPhysics
    bc = BoundaryConfig(*faces)
    if pml_cells:
        bc.cpml_npml = pml_cells
    if bc.bloch_y:
        bc.bloch_ky = float(bloch_order)
    sides = bc.npml_sides
    dr_cells = np.asarray(dr_cells, dtype=np.float64)
    dx_cells = np.concatenate((dr_cells, np.full(sides['x_hi'], dr_cells[-1])))
    grid = NonUniformGrid2D(dx_cells, np.full(angle_cells, d_angle), verbose=False)
    window_x = float(np.sum(grid.dx_arr[:dr_cells.size], dtype=np.float64))
    domain = Domain(grid, MaterialGrid(grid.nx, grid.ny, backend='cpu'), boundary_config=bc, ox=0, oy=0,
                    size_x=window_x, size_y=grid.size_y, non_uniform=True, npml_sides=sides,
                    coordinates=Coordinates('polar', angle_wrap=angle_wrap))
    physics = FDFDPhysics(domain, polarization, boundary_config=bc, wavelength=wavelength)
    return physics, domain


def _bessel_root(order, integer_order, derivative):
    """First root x > 0 of J_nu (derivative False) or J_nu' (True) of a non-integer nu just below integer_order."""
    from scipy.optimize import brentq
    from scipy.special import jn_zeros, jnp_zeros, jv, jvp
    guess = float((jnp_zeros if derivative else jn_zeros)(integer_order, 1)[0])
    function = (lambda x: jvp(order, x)) if derivative else (lambda x: jv(order, x))
    return brentq(function, guess - 0.6, guess + 0.1, xtol=1e-14)


def _nearest_omega(physics, omega_target, n_resonances=1):
    """Complex omega and Q of the resonance nearest omega_target (the physics re-targeted there)."""
    physics.omega = omega_target
    solve = physics.solve_eigenfrequencies(n_resonances)
    if not solve.resonances:
        raise ValueError(f'no resonance found near lambda = {2.0 * math.pi * C0 / omega_target * 1e9:.6g} nm')
    resonance = solve.resonances[0]
    return complex(resonance.omega), float(resonance.q_factor)


def _check_disk_cavity(_results):
    """Resonances of the PEC disk (TM_m1 of ez, TE_m1 of hz, m in _DISK_ORDERS) vs J_nu / J_nu' roots."""
    errors, details = {}, {}
    d_angle = 2.0 * math.pi / _DISK_ANGLE_CELLS
    for polarization in ('ez', 'hz'):
        physics, domain = _polar_fdfd(polarization, np.full(_DISK_RADIAL_CELLS, _DISK_RADIUS / _DISK_RADIAL_CELLS),
                                      0, _DISK_ANGLE_CELLS, d_angle, ('pmc', 'pec', 'periodic', 'periodic'),
                                      'periodic', 0.0, 1.55e-6)
        physics.finalize()
        r_faces = physics.metric.x_face_area.along_x
        r_centres = physics.metric.y_length.along_x
        wall = float(r_centres[-1] if polarization == 'ez' else r_faces[-2])
        d_angle_grid = float(domain.cell_widths()[1][0])
        for order in _DISK_ORDERS:
            nu = _discrete_order(order, d_angle_grid)
            omega_expected = C0 * _bessel_root(nu, order, polarization == 'hz') / wall
            omega, _ = _nearest_omega(physics, _TARGET_OFFSET * omega_expected)
            item = f'{polarization} m = {order}'
            errors[item] = abs(omega.real / omega_expected - 1.0)
            details[item] = {'omega_expected': omega_expected, 'omega_measured': [omega.real, omega.imag],
                             'order_nu': nu, 'wall_radius': wall}
    error = max(errors.values())
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _DISK_TOLERANCE,
            'details': {'errors': errors, **details}}


def _wgm_mismatch(order, index, polarization, u):
    """Field-matching determinant of a dielectric disk at u = k R (outgoing H^(2) outside): ez n J' H - J H', hz J' H
    / n - J H'."""
    from scipy.special import h2vp, hankel2, jv, jvp
    inner = index * u
    ratio = index if polarization == 'ez' else 1.0 / index
    return ratio * jvp(order, inner) * hankel2(order, u) - jv(order, inner) * h2vp(order, u)


def _wgm_root(order, index, polarization):
    """Complex u = k R of the first radial whispering-gallery resonance (secant from the Airy asymptote n k R = nu +
    2^(-1/3) a_1 nu^(1/3) - P / sqrt(n^2 - 1), P = n for E_z, 1 / n for H_z)."""
    pole = index if polarization == 'ez' else 1.0 / index
    x_guess = order + 1.8557571 * order ** (1.0 / 3.0) - pole / math.sqrt(index ** 2 - 1.0)
    previous, current = complex(x_guess / index), complex(x_guess / index) * (1.0 + 1e-3) + 1e-3j
    f_previous, f_current = (_wgm_mismatch(order, index, polarization, value) for value in (previous, current))
    for _ in range(200):
        step = f_current * (current - previous) / (f_current - f_previous)
        previous, f_previous = current, f_current
        current = current - step
        f_current = _wgm_mismatch(order, index, polarization, current)
        if abs(step) <= 1e-13 * abs(current):
            return current
    raise ValueError(f'whispering-gallery root of order {order} ({polarization}) did not converge')


def _check_whispering_gallery(_results):
    """Complex resonance of the dielectric disk in one C_16 sector (Bloch order 6, cylindrical PML) vs the
    semi-discrete Hankel matching root, both polarisations."""
    errors, tolerances, details = {}, {}, {}
    span = 2.0 * math.pi / _WGM_SECTORS
    dr = _WGM_RADIUS / _WGM_DISK_CELLS
    for polarization in ('ez', 'hz'):
        physics, domain = _polar_fdfd(polarization, np.full(_WGM_WINDOW_CELLS, dr), _WGM_PML_CELLS, _WGM_SECTOR_CELLS,
                                      span / _WGM_SECTOR_CELLS, ('pmc', 'cpml', 'bloch', 'bloch'), 'periodic',
                                      _WGM_ORDER, 1.55e-6)
        radius = float(physics.metric.x_face_area.along_x[_WGM_DISK_CELLS])
        disk = np.zeros((domain.grid.nx, domain.grid.ny), dtype=bool)
        disk[:_WGM_DISK_CELLS, :] = True
        names = ('eps_r',) if polarization == 'ez' else ('eps_xx', 'eps_yy')
        for name in names:
            domain.materials.assign(name, disk, _WGM_INDEX ** 2)
        physics.materials_changed(names)
        physics.finalize()
        nu = _discrete_order(_WGM_ORDER, float(domain.cell_widths()[1][0]))
        omega_expected = C0 * _wgm_root(nu, _WGM_INDEX, polarization) / radius
        q_expected = omega_expected.real / (2.0 * omega_expected.imag)
        omega, q_factor = _nearest_omega(physics, _TARGET_OFFSET * omega_expected.real)
        errors[f'{polarization} omega'] = abs(omega.real / omega_expected.real - 1.0)
        tolerances[f'{polarization} omega'] = _WGM_OMEGA_TOLERANCE
        errors[f'{polarization} Q'] = abs(q_factor / q_expected - 1.0)
        tolerances[f'{polarization} Q'] = _WGM_Q_TOLERANCE
        details[polarization] = {'omega_expected': [omega_expected.real, omega_expected.imag],
                                 'omega_measured': [omega.real, omega.imag], 'q_expected': q_expected,
                                 'q_measured': q_factor, 'order_nu': nu,
                                 'wavelength_expected': 2.0 * math.pi * C0 / omega_expected.real}
    return tolerance_units_result(errors, tolerances, details)


def _ring_run(polarization):
    """(physics, domain, source phasor currents (ny,), recorded fluxes) of the C_4 ring-source run."""
    span = 2.0 * math.pi / _RING_SECTORS
    dr = _RING_WAVELENGTH / 60.0
    physics, domain = _polar_fdfd(polarization, np.full(_RING_RADIAL_CELLS, dr), _RING_PML_CELLS, _RING_SECTOR_CELLS,
                                  span / _RING_SECTOR_CELLS, ('pmc', 'cpml', 'bloch', 'bloch'), 'periodic', _RING_ORDER,
                                  _RING_WAVELENGTH)
    x_centres, y_centres = domain.cell_centers()
    currents = np.empty(domain.grid.ny, dtype=np.complex128)
    add = physics.add_point_source if polarization == 'ez' else physics.add_magnetic_line_current
    for j, angle in enumerate(y_centres):
        phase = -math.degrees(_RING_ORDER * angle)
        add(float(x_centres[_RING_SOURCE_COLUMN]), float(angle), _RING_CURRENT, phase)
        currents[j] = _RING_CURRENT * cmath.exp(1j * math.radians(phase))
    monitors = [physics.add_flux_monitor('y', float(x_centres[column]), (0.0, domain.size_y))
                for column in _RING_FLUX_COLUMNS]
    physics.finalize()
    physics.solve_harmonic(2.0 * math.pi * C0 / _RING_WAVELENGTH)
    fluxes = [float(monitor.recorded()[1][0]) for monitor in monitors]
    return physics, domain, currents, fluxes


def _ring_errors(polarization):
    """Field vs C J_nu(k r<) H_nu(k r>) e^{-j m phi}, the two arc fluxes vs the source power (exact) and vs the closed
    form 2 |C J_nu(k r0)|^2 / (omega mu0 | eps0) x span / 2 pi."""
    from scipy.special import hankel2, jv
    physics, domain, currents, fluxes = _ring_run(polarization)
    omega = physics.solved_omega
    wavenumber = omega / C0
    r_centres = physics.metric.y_length.along_x
    d_angle = float(domain.cell_widths()[1][0])
    nu = _discrete_order(_RING_ORDER, d_angle)
    coupling = MU0 if polarization == 'ez' else EPS0
    amplitude = omega * coupling * _RING_CURRENT * (2.0 * math.pi / d_angle) / 4.0
    r_source = float(r_centres[_RING_SOURCE_COLUMN])
    window = domain.physical_window()
    columns = [i for i in range(window[0].start, window[0].stop) if abs(i - _RING_SOURCE_COLUMN) >= _RING_EXCLUDED]
    radii = r_centres[columns]
    inner, outer = np.minimum(radii, r_source), np.maximum(radii, r_source)
    radial = amplitude * jv(nu, wavenumber * inner) * hankel2(nu, wavenumber * outer)
    _, y_centres = domain.cell_centers()
    expected = radial[:, None] * np.exp(-1j * _RING_ORDER * y_centres)[None, :]
    solution = np.asarray(physics.solution)
    errors = {f'{polarization} field': max_relative_error(solution[columns, :], expected)}
    source_power = 0.5 * float(np.real(np.sum(np.conj(solution[_RING_SOURCE_COLUMN, :]) * currents)))
    errors[f'{polarization} power balance'] = max(abs(flux / source_power - 1.0) for flux in fluxes)
    span = float(np.sum(domain.cell_widths()[1]))
    closed_power = (2.0 * abs(amplitude * jv(nu, wavenumber * r_source)) ** 2 / (omega * coupling)
                    * span / (2.0 * math.pi))
    errors[f'{polarization} power'] = abs(fluxes[0] / closed_power - 1.0)
    details = {'fluxes': float_list(fluxes), 'source_power': source_power, 'closed_power': closed_power,
               'order_nu': nu}
    return errors, details


def _polar_refusals():
    """Number of refused setups that went through (the origin needs pmc, joined faces need periodic phi faces and the
    reverse, Cartesian-only sources)."""
    def expect_refusal(make):
        try:
            make()
        except (ValueError, NotImplementedError):
            return 0
        return 1

    cells, d_angle = np.full(20, 5e-8), 2.0 * math.pi / 16
    missed = expect_refusal(lambda: _polar_fdfd('ez', cells, 0, 16, d_angle, ('pec', 'pec', 'periodic', 'periodic'),
                                                'periodic', 0.0, 1e-6))
    missed += expect_refusal(lambda: _polar_fdfd('ez', cells, 0, 16, d_angle, ('pmc', 'pec', 'periodic', 'periodic'),
                                                 'none', 0.0, 1e-6))
    missed += expect_refusal(lambda: _polar_fdfd('ez', cells, 0, 16, d_angle, ('pmc', 'pec', 'pec', 'pec'),
                                                 'periodic', 0.0, 1e-6))
    physics, domain = _polar_fdfd('ez', cells, 0, 16, d_angle, ('pmc', 'pec', 'periodic', 'periodic'), 'periodic',
                                  0.0, 1e-6)
    missed += expect_refusal(lambda: physics.add_tfsf(1e-7, 0.1, 5e-7, 0.5))
    missed += expect_refusal(lambda: physics.add_line_source('y', 5e-7))
    hz_physics, _ = _polar_fdfd('hz', cells, 0, 16, d_angle, ('pmc', 'pec', 'periodic', 'periodic'), 'periodic',
                                0.0, 1e-6)
    missed += expect_refusal(lambda: hz_physics.add_electric_dipole(5e-7, 0.5))
    return missed


def _check_ring_source(_results):
    errors, tolerances, details = {}, {}, {}
    for polarization in ('ez', 'hz'):
        polarization_errors, details[polarization] = _ring_errors(polarization)
        errors.update(polarization_errors)
        tolerances.update({f'{polarization} field': _RING_FIELD_TOLERANCE,
                           f'{polarization} power balance': _EXACT_TOLERANCE,
                           f'{polarization} power': _RING_FLUX_TOLERANCE})
    errors['refusals'] = _polar_refusals()
    tolerances['refusals'] = 0.5
    return tolerance_units_result(errors, tolerances, details)


POLAR_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdfd_polar_disk_cavity', label='EM FDFD polar grid: PEC disk cavity resonances (Bessel zeros)',
        description=('Scripted FDFDPhysics (cpu backend) on a polar (r, phi) Domain from the origin (x_lo pmc = the '
                     f'origin, zero face length), 2 pi periodic phi with {_DISK_ANGLE_CELLS} cells, '
                     f'{_DISK_RADIAL_CELLS} uniform dr to {_DISK_RADIUS * 1e6:g} um, pec on x_hi (ez: wall at the last '
                     'cell centre; hz: at the inner face of the last cell). Eigenfrequency nearest the closed form '
                     f'for m = {", ".join(str(order) for order in _DISK_ORDERS)}: TM_m1 J_nu(k a) = 0 (em_fdfd_ez), '
                     "TE_m1 J_nu'(k a) = 0 (em_fdfd_hz) with the semi-discrete order nu = (2 / dphi) sin(m dphi / 2) "
                     '(exact for the angular differences; the radial error is second order). error = the largest '
                     f'relative Re omega deviation, tolerance {_DISK_TOLERANCE:g} (estimated, not measured).'),
        specs=lambda: {}, check=_check_disk_cavity, max_wall_time=120.0),
    SelfTestCase(
        name='em_fdfd_polar_whispering_gallery',
        label='EM FDFD polar grid: whispering-gallery disk in a C_N sector with cylindrical PML',
        description=(f'Scripted FDFDPhysics (cpu backend): dielectric disk n = {_WGM_INDEX:g}, radius '
                     f'{_WGM_RADIUS * 1e6:g} um ({_WGM_DISK_CELLS} uniform dr, window to 2.2 R, '
                     f'{_WGM_PML_CELLS} cpml cells on x_hi = cylindrical PML with complex r), one 2 pi / '
                     f'{_WGM_SECTORS} sector of {_WGM_SECTOR_CELLS} angle cells with bloch phi faces at azimuthal order '
                     f'm = {_WGM_ORDER} (phase 3 pi / 4). The resonance nearest the closed form must match the '
                     "outgoing-wave matching root (ez: n J_nu'(n u) H2_nu(u) = J_nu(n u) H2_nu'(u); hz: J_nu' H2 / n "
                     '= J_nu H2\', u = k R, nu = the semi-discrete order): relative Re omega to '
                     f'{_WGM_OMEGA_TOLERANCE:g} and Q to {_WGM_Q_TOLERANCE:g} (estimated). measured = error = the '
                     'worst deviation in tolerance units, tolerance 1.'),
        specs=lambda: {}, check=_check_whispering_gallery, max_wall_time=120.0),
    SelfTestCase(
        name='em_fdfd_polar_ring_source',
        label='EM FDFD polar grid: phased ring of line currents, angular Bloch sector, power balance',
        description=(f'Scripted FDFDPhysics (cpu backend), vacuum, lambda = {_RING_WAVELENGTH * 1e6:g} um, polar '
                     f'Domain from the origin with dr = lambda / 60 to 1.5 lambda and {_RING_PML_CELLS} cpml cells '
                     f'(cylindrical PML), one 2 pi / {_RING_SECTORS} sector of {_RING_SECTOR_CELLS} cells with bloch '
                     f'phi faces at order m = {_RING_ORDER} (phase 3 pi / 2). One line current I e^{{-j m phi_j}} per '
                     f'angle cell of column {_RING_SOURCE_COLUMN} (ez point sources, hz magnetic line currents). '
                     'Checks per polarisation: the field vs C J_nu(k r<) H2_nu(k r>) e^{-j m phi}, C = omega mu0 '
                     '(eps0) I (2 pi / dphi) / 4 (the FDFD sign of a line current) outside 3 cells around the source '
                     f'column ({_RING_FIELD_TOLERANCE:g} of the peak); the outward flux through two arcs (r_f dphi '
                     f'weights) = the source power 1/2 Re sum conj(u) I exactly ({_EXACT_TOLERANCE:g}); the flux vs '
                     f'2 |C J_nu(k r0)|^2 / (omega mu0) x span / 2 pi ({_RING_FLUX_TOLERANCE:g}); refusals (origin '
                     'without pmc, periodic phi faces without a joined domain and the reverse, TFSF, line source, '
                     'electric dipole). measured = error = the worst deviation in tolerance units, tolerance 1; the '
                     'approximate tolerances are estimated, not measured.'),
        specs=lambda: {}, check=_check_ring_source, max_wall_time=120.0),
)
