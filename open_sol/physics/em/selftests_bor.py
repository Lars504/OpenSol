"""Self-tests of the body-of-revolution FDFD (symmetry 2.3): CPU checks of body_of_revolution / bor_ports (cavities vs
Bessel zeros, dipole field, flux and far field, coax ports, order polynomial and factor recycling), the m = 1 cavity
model spec / check of em_bor_cpu_gpu, and the 'microdisk_bor_eigenmodes' template; physics imports inside the checks
only."""
import math

import numpy as np

from ...core.constants import C0, ETA0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import float_list, skip_without_cudss
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           ParameterSpec, PhysicsSpec, StudySpec)
_TYPE = 'em_fdfd_bor'
_PHYSICS = 'em'
_TABLE = 'resonances'
_EIGEN_STUDY = 'eigenfrequency'   # = model.study_eigenfrequency.STUDY_NAME

# ---- PEC cylinder cavity: radius a, height L, walls on the grid edges (r_hi, z_lo, z_hi pec). Exact: TM_mnp k^2 =
# (x_mn / a)^2 + (p pi / L)^2 (p >= 0), TE_mnp k^2 = (x'_mn / a)^2 + (p pi / L)^2 (p >= 1), x_mn / x'_mn the zeros of
# J_m / J_m'. Second-order scheme: at h = a / 50 the relative k error is a few 1e-4.
_CAVITY_CELL = 20e-9
_CAVITY_CELLS = (50, 50)              # a = L = 1 um
_CAVITY_TARGET = 1.5e-6               # m
_CAVITY_ORDERS = (0, 1, 2)            # the three axis classes (m = 0, |m| = 1, |m| >= 2)
_CAVITY_RESONANCES = 8                # solved per order
_CAVITY_CHECKED = 4                   # analytic resonances nearest the target matched per order
_CAVITY_TOLERANCE = 3e-3              # relative k
_ANALYTIC_MAX_INDEX = 8               # radial zeros and axial orders of the analytic list
# re-solve at this order with the target on its discrete resonance nearest _CAVITY_TARGET (lossless: A singular at
# k_t^2; the own factor's offset shift must still return it and _CAVITY_RESONANCES resonances)
_ON_RESONANCE_ORDER = 1
_ON_RESONANCE_TOLERANCE = 1e-8        # relative omega of the targeted resonance (same discrete operator)

# ---- radiation grid: vacuum, cpml on r_hi and both z faces. The on-axis z current element (m = 0, cell 0 = a disc of
# current I, moment I h) of em_bor_radiation also gives |E_z| in its equatorial plane vs the exact dipole field eta0 k I
# h / (4 pi R) |1 + 1 / (j k R) - 1 / (k R)^2|. The sources reach the PML only through reflections, so the window keeps
# 12 cells between the closed box and the PML (2 x 3 um with 20 before: ~3x the LU time).
_DIPOLE_WAVELENGTH = 1.0e-6
_DIPOLE_CELL = 25e-9                  # 40 cells per wavelength
_DIPOLE_PML = 15
_DIPOLE_WINDOW = (48, 72)             # 1.2 x 1.8 um
_DIPOLE_CURRENT = 1e-3                # A
_DIPOLE_PROBE_CELLS = (12, 20, 28)    # r cells of the probes (R ~ 0.31, 0.51, 0.71 wavelengths), inside the box
_DIPOLE_TOLERANCE = 0.03              # relative |E_z|: discrete Green's function O((k h)^2, (h / R)^2), PML reflection

# ---- the cavity filled with a z-uniaxial medium: eps_rr = eps_phiphi = eps_t, eps_zz = eps_z (TE sees eps_t only)
_UNIAXIAL_EPS_TRANSVERSE = 2.0
_UNIAXIAL_EPS_AXIAL = 3.0
_UNIAXIAL_ORDERS = (0, 2)

# ---- radiation: the dipole grid (vacuum, cpml), sources inside a closed cell box away from the PML. Exact discrete
# identity box_flux = source_power; vs the analytic radiated power and pattern of an on-axis z dipole (m = 0, moment
# I h: U = eta0 k^2 (I h)^2 sin^2 theta / (32 pi^2)) and of an azimuthal ring current I exp(-j m phi) of radius rho0
# (U = eta0 k^2 I^2 rho0^2 / 8 [cos^2 theta ((J_(m-1) + J_(m+1)) / 2)^2 + ((J_(m-1) - J_(m+1)) / 2)^2], a = k rho0
# sin theta); power = 2 pi int U sin theta dtheta.
_BOX_R_CELLS = (0, 36)                # 0.9 um
_BOX_Z_ROWS = (12, 60)                # window rows (+ the PML offset): 1.2 um tall around the source row 36
_RING_CELL = 12                       # rho0 = 0.3125 um
_RING_ORDER = 2
_RADIATION_THETA = 721
_IDENTITY_TOLERANCE = 1e-6            # box flux vs source power (exact discrete Poynting theorem)
_POWER_TOLERANCE = 0.04               # vs the analytic power (discrete source, PML reflection)
_PATTERN_TOLERANCE = 0.05             # max |U - U_analytic| / max U_analytic
_FAR_POWER_TOLERANCE = 0.03           # far-field solid-angle integral vs the box flux

# ---- port guide grid (radius 1 um = 40 cells of 25 nm, pec r_hi, cpml on both z faces): a 1 W launch at row 30 toward
# +z, monitors 15 rows upstream and 45 rows downstream, flux downstream (the pipe TE11 ports at m = 1 are the example
# em/analytic_bor_pipe_te11_section)
_GUIDE_CELL = 25e-9
_GUIDE_CELLS = (40, 120)
_GUIDE_PML = 15
_GUIDE_SOURCE_ROW = 30                # window rows
_GUIDE_UPSTREAM_ROW = 15
_GUIDE_DOWNSTREAM_ROW = 75
_GUIDE_PORT_TOLERANCE = 5e-3          # |T psi^-45 - 1|, |R|^2, |P - 1|: exact launch, z PML reflection only

# ---- assembly reuse on a small open grid (25 nm cells, window 24 x 40, 8-cell cpml on r_hi and both z faces: the
# identities hold on any grid; nz > nr keeps the line-block layout choice) with an anisotropic dielectric block: the
# order polynomial K0 + m K1 + m^2 K2 vs C(m)^T W C(m) per order, and a driven solve after a small eps change by GMRES on
# the old factor vs a fresh one (em_bor_line_block_family solves the same assembly)
_IDENTITY_WINDOW = (24, 40)
_IDENTITY_PML = 8
_IDENTITY_SOURCE_ROW = _IDENTITY_PML + _IDENTITY_WINDOW[1] // 2   # grid row of the ring current
_SWEEP_ORDERS = (-3, -2, -1, 0, 1, 2, 3)
_POLYNOMIAL_TOLERANCE = 1e-12         # relative Frobenius norm (roundoff)
_RECYCLE_ORDER = 2
_RECYCLE_CELL = 6                     # ring current r cell
_RECYCLE_BLOCK = (slice(8, 16), slice(18, 30))   # dielectric block cells (r, z grid indices)
_RECYCLE_EPS = 2.0
_RECYCLE_DELTA = 1e-3                 # thermo-optic-like eps change in the block
_RECYCLE_TOLERANCE = 1e-7             # recycled vs direct solution, relative (RECYCLE_RTOL residual x conditioning)

# ---- coaxial guide (inner pec radius 0.25 um = 10 cells of 25 nm as pec cells, outer pec r_hi at 1 um), vacuum,
# lambda 2 um, annular port lines from the inner conductor: m = 0 TEM n_eff = the exact discrete 2 asin(k h / 2) / (k
# h) (the transverse field 1 / r is exact on the ring metric), m = 1 TE11 n_eff vs the root of J1'(k a) Y1'(k b) -
# J1'(k b) Y1'(k a), the annular line vs a line from the axis (the same modes: the cells below are pec), and the
# one-sided TE11 launch on the port guide grid.
_COAX_INNER_CELLS = 10
_COAX_WAVELENGTH = 2.0e-6
_COAX_TEM_TOLERANCE = 1e-7            # exact discrete (ARPACK precision)
_COAX_TE11_TOLERANCE = 2e-3           # second order in r and z
_COAX_SAME_LINE_TOLERANCE = 1e-9      # annular vs axis line n_eff

# ---- model run of em_bor_cpu_gpu: the cavity at m = 1 (axis E_r / H_phi path) through the model layer (25 nm cells)
_MODEL_CELL = 25e-9
_MODEL_ORDER = 1
_MODEL_RESONANCES = 3
_MODEL_TOLERANCE = 5e-3               # relative omega

# ---- template: silicon microdisk (r 1.5 um, 240 nm thick = 2 x 6 cells of 20 nm) in air, equatorial pmc mirror
# (quasi-TE), m = 15 (m ~ 2 pi R n_eff / lambda with the mode ~0.15 um inside the rim, n_eff ~ 2.9)
MICRODISK_LABEL = 'Microdisk whispering-gallery modes (axisymmetric)'
MICRODISK_DESCRIPTION = (
    'EM body-of-revolution FDFD, eigenfrequency study: the whispering-gallery resonances of azimuthal order m = 15 '
    'of a silicon microdisk (eps_r 12.11, radius r_disk = 1.5 um, 240 nm thick) in air nearest 1.55 um, solved on '
    'the (r, z) half plane above its equatorial plane (pmc mirror on z_lo: quasi-TE modes, E in the disk plane), '
    'cpml on r_hi and z_hi, 20 nm mesh. Wavelength, Q (radiation into the PML) and field per resonance; derived '
    'lambda_1 and Q_1. A full 3D answer at 2D cost; list several orders in azimuthal_orders (e.g. 14, 15, 16) for '
    'the free spectral range.')
MICRODISK_SUMMARY = ('Finds the whispering-gallery resonances of a thin silicon disk near 1550 nm from a cut through '
                     'its axis: wavelength, quality factor and field shape, at the cost of a 2D model.')
_DISK_RADIUS = 1.5e-6
_DISK_THICKNESS = 0.24e-6
_DISK_DOMAIN = (3.0e-6, 1.2e-6)


def microdisk_bor_eigenmodes_spec():
    """Template 'microdisk_bor_eigenmodes': silicon disk on the axis, mirror on z = 0, em_fdfd_bor m = 15
    eigenfrequency study near 1.55 um with a resonance table and lambda_1 / Q_1."""
    return ModelSpec(
        title=MICRODISK_LABEL,
        parameters=[ParameterSpec('r_disk', _DISK_RADIUS, 'disk radius [m]'),
                    ParameterSpec('t_disk', _DISK_THICKNESS, 'disk thickness [m]')],
        domain=DomainSpec(size_x=_DISK_DOMAIN[0], size_y=_DISK_DOMAIN[1]),
        geometry=[GeometrySpec('disk', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': 'r_disk', 'y1': 't_disk / 2'})],
        materials=[MaterialSpec('silicon_disk', properties={'eps_r': 12.11}, selection=['disk'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings={'wavelength': 1.55e-6, 'azimuthal_order': 15,
                                                        'bc_x_hi': 'cpml', 'bc_y_lo': 'pmc', 'bc_y_hi': 'cpml'},
                             features=[FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=StudySpec(type=_EIGEN_STUDY, settings={'n_resonances': 4}),
        derived_values=[DerivedValueSpec('lambda_1', 'wavelength', operation='at', source=f'{_PHYSICS}.{_TABLE}', at=0),
                        DerivedValueSpec('Q_1', 'q_factor', operation='at', source=f'{_PHYSICS}.{_TABLE}', at=0)])


def _analytic_cavity(order, radius, length, eps_transverse=1.0, eps_axial=1.0):
    """Sorted analytic vacuum k [1/m] of the PEC cylinder (TM p >= 0, TE p >= 1) at azimuthal order `order`, filled
    with a z-uniaxial medium: TE k^2 eps_t = (x'/a)^2 + (p pi / L)^2, TM k^2 eps_t = (p pi / L)^2 + eps_t / eps_z
    (x / a)^2."""
    import scipy.special
    values = []
    for zero in scipy.special.jn_zeros(order, _ANALYTIC_MAX_INDEX):
        values.extend(math.sqrt(((p * math.pi / length) ** 2 + eps_transverse / eps_axial * (zero / radius) ** 2)
                                / eps_transverse) for p in range(_ANALYTIC_MAX_INDEX))
    for zero in scipy.special.jnp_zeros(order, _ANALYTIC_MAX_INDEX):
        values.extend(math.hypot(zero / radius, p * math.pi / length) / math.sqrt(eps_transverse)
                      for p in range(1, _ANALYTIC_MAX_INDEX))
    return sorted(values)


def _match(computed, analytic_all, target, n_checked):
    """(worst relative error of the n_checked analytic k nearest target, each greedily matched to the nearest unused
    computed k; number of computed k inside that band matching no analytic k within the tolerance)."""
    checked = sorted(analytic_all, key=lambda value: abs(value - target))[:n_checked]
    unused, worst = list(computed), 0.0
    for value in sorted(checked):
        if not unused:
            return math.inf, 0
        best = min(range(len(unused)), key=lambda index: abs(unused[index] - value))
        worst = max(worst, abs(unused.pop(best) / value - 1.0))
    band = max(abs(value - target) for value in checked) * (1.0 - _CAVITY_TOLERANCE)
    spurious = sum(1 for value in computed if abs(value - target) < band
                   and min(abs(value / reference - 1.0) for reference in analytic_all) > _CAVITY_TOLERANCE)
    return worst, spurious


def _check_cavity(results_by_name):
    """Resonances of the vacuum PEC cylinder per azimuthal order vs the Bessel closed form, + the on-resonance target."""
    return _cavity_check(_CAVITY_ORDERS, 1.0, 1.0, on_resonance_order=_ON_RESONANCE_ORDER)


def _check_uniaxial_cavity(results_by_name):
    """Resonances of the PEC cylinder filled with a z-uniaxial medium (eps_rr = eps_phiphi != eps_zz)."""
    return _cavity_check(_UNIAXIAL_ORDERS, _UNIAXIAL_EPS_TRANSVERSE, _UNIAXIAL_EPS_AXIAL)


def _on_resonance_solve(widths, config, materials, order, solve):
    """(details, violation or None) of the cavity re-solved at `order` with its target on solve's nearest resonance."""
    from .body_of_revolution import BORAssembly, solve_bor_resonances
    if not solve.resonances:
        return {}, f'm = {order}: no resonance to target'
    omega_resonance = float(solve.resonances[0].omega.real)
    system = BORAssembly(widths, config, _CAVITY_CELL, materials, omega_resonance).system(order)
    found = [resonance.omega for resonance in solve_bor_resonances(system, _CAVITY_RESONANCES).resonances]
    error = abs(found[0] / omega_resonance - 1.0) if found else math.inf
    details = {'order': order, 'target_k': omega_resonance / C0, 'relative_error': error,
               'found_k': float_list([omega.real / C0 for omega in found])}
    if len(found) < _CAVITY_RESONANCES or not error <= _ON_RESONANCE_TOLERANCE:
        return details, (f'm = {order} targeted on its resonance k = {omega_resonance / C0:.8g} 1/m: {len(found)} of '
                         f'{_CAVITY_RESONANCES} resonances, the targeted one off by {error:.3g} relative')
    return details, None


def _cavity_check(orders, eps_transverse, eps_axial, on_resonance_order=None):
    """Self-test record of the PEC cylinder cavity (eps_rr = eps_phiphi = eps_transverse, eps_zz = eps_axial) at the
    azimuthal orders vs _analytic_cavity; on_resonance_order also gets _on_resonance_solve."""
    from .body_of_revolution import BORAssembly, BORMaterials, solve_bor_resonances
    from .boundaries import BoundaryConfig
    config = BoundaryConfig(x_lo='pmc', x_hi='pec', y_lo='pec', y_hi='pec')
    shape = _CAVITY_CELLS
    widths = tuple(np.full(count, _CAVITY_CELL) for count in shape)
    radius, length = shape[0] * _CAVITY_CELL, shape[1] * _CAVITY_CELL
    transverse = np.full(shape, eps_transverse, dtype=np.complex128)
    materials = BORMaterials(eps=transverse, mu=np.ones(shape, dtype=np.complex128), pec=np.zeros(shape, dtype=bool),
                             eps_rr=transverse, eps_zz=np.full(shape, eps_axial, dtype=np.complex128))
    omega_target = 2.0 * math.pi * C0 / _CAVITY_TARGET
    k_target = omega_target / C0
    errors, details, violations = {}, {}, []
    assembly = BORAssembly(widths, config, _CAVITY_CELL, materials, omega_target)   # one assembly: the m sweep
    for order in orders:
        solve = solve_bor_resonances(assembly.system(order), _CAVITY_RESONANCES)
        computed = [resonance.omega.real / C0 for resonance in solve.resonances]
        analytic = _analytic_cavity(order, radius, length, eps_transverse, eps_axial)
        worst, spurious = _match(computed, analytic, k_target, _CAVITY_CHECKED)
        errors[f'm = {order}'] = worst
        if spurious:
            violations.append(f'm = {order}: {spurious} computed resonance(s) near the target match no analytic one')
        if order == on_resonance_order:
            details['on_resonance'], violation = _on_resonance_solve(widths, config, materials, order, solve)
            if violation:
                violations.append(violation)
        details[f'm = {order}'] = {'computed_k': float_list(sorted(computed)),
                                   'analytic_k': float_list(sorted(analytic, key=lambda v: abs(v - k_target))
                                                            [:_CAVITY_CHECKED]),
                                   'max_abs_imag_omega': float(max((abs(resonance.omega.imag) for resonance
                                                                    in solve.resonances), default=0.0)),
                                   'unknowns': solve.info['n_unknowns']}
    worst_item = max(errors, key=errors.get)
    error = errors[worst_item] if not violations else math.inf
    details.update(errors=errors, worst_item=worst_item, violations=violations)
    return {'measured': errors[worst_item], 'expected': 0.0, 'error': error, 'tolerance': _CAVITY_TOLERANCE,
            'details': details}


def _open_grid(cells, pml):
    """(BoundaryConfig, cell widths, window, full shape) of a _DIPOLE_CELL grid of `cells` window cells (r, z): pmc
    axis, `pml` cpml cells on r_hi and both z faces."""
    from .boundaries import BoundaryConfig
    config = BoundaryConfig(x_lo='pmc', x_hi='cpml', y_lo='cpml', y_hi='cpml')
    config.cpml_npml = pml
    shape = (cells[0] + pml, cells[1] + 2 * pml)
    widths = tuple(np.full(count, _DIPOLE_CELL) for count in shape)
    return config, widths, (slice(0, cells[0]), slice(pml, pml + cells[1])), shape


def _open_vacuum_grid():
    """(BoundaryConfig, cell widths, window, vacuum BORMaterials, omega) of the em_bor_radiation grid: cpml on r_hi and
    both z faces, at its wavelength."""
    from .body_of_revolution import BORMaterials
    config, widths, window, shape = _open_grid(_DIPOLE_WINDOW, _DIPOLE_PML)
    materials = BORMaterials(eps=np.ones(shape, dtype=np.complex128), mu=np.ones(shape, dtype=np.complex128),
                             pec=np.zeros(shape, dtype=bool))
    return config, widths, window, materials, 2.0 * math.pi * C0 / _DIPOLE_WAVELENGTH


def _open_vacuum_solve(order, source_cell, component):
    """(system, source row, solution, rhs) of a _DIPOLE_CURRENT element `component` in r cell source_cell of the
    window's middle row: vacuum, cpml on r_hi and both z faces, azimuthal order `order`."""
    from .body_of_revolution import assemble_bor, point_current_rhs
    from ...solvers.sparse_direct import factorize_host
    config, widths, window, materials, omega = _open_vacuum_grid()
    system = assemble_bor(widths, config, _DIPOLE_CELL, materials, omega, order, window=window)
    source_row = _DIPOLE_PML + _DIPOLE_WINDOW[1] // 2
    rhs = point_current_rhs(system, source_cell, source_row, component, _DIPOLE_CURRENT)
    return system, source_row, factorize_host(system.operator).solve_host(rhs), rhs


def _trapezoid(values, abscissa):
    """Trapezoid integral of samples over a 1D abscissa (numpy-version independent)."""
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(abscissa)))


def _radiation_case(order, source_cell, component, analytic_pattern):
    """(normalised errors, details, (system, source row, solution)) of one radiating source: box flux vs source power,
    vs the analytic power, the far-field pattern and its solid-angle integral."""
    from .body_of_revolution import box_flux, far_field_power, far_field_samples, source_power, yee_fields
    system, source_row, solution, rhs = _open_vacuum_solve(order, source_cell, component)
    fields = yee_fields(system, solution)
    z_rows = (_DIPOLE_PML + _BOX_Z_ROWS[0], _DIPOLE_PML + _BOX_Z_ROWS[1])
    flux = box_flux(system, fields, _BOX_R_CELLS, z_rows)
    delivered = source_power(system, solution, rhs)
    theta = np.linspace(0.0, math.pi, _RADIATION_THETA)
    pattern = far_field_power(order, system.omega, far_field_samples(system, fields, _BOX_R_CELLS, z_rows), theta)[0]
    expected_pattern = analytic_pattern(system.omega / C0, theta)
    expected_power = 2.0 * math.pi * _trapezoid(expected_pattern * np.sin(theta), theta)
    far_power = 2.0 * math.pi * _trapezoid(pattern * np.sin(theta), theta)
    errors = {'identity': abs(flux / delivered - 1.0) / _IDENTITY_TOLERANCE,
              'power': abs(flux / expected_power - 1.0) / _POWER_TOLERANCE,
              'pattern': float(np.max(np.abs(pattern - expected_pattern)) / np.max(expected_pattern))
              / _PATTERN_TOLERANCE,
              'far_power': abs(far_power / flux - 1.0) / _FAR_POWER_TOLERANCE}
    details = {'box_flux_W': flux, 'source_power_W': delivered, 'analytic_power_W': expected_power,
               'far_field_power_W': far_power, 'pattern_peak_W_per_sr': float(np.max(pattern)),
               'analytic_peak_W_per_sr': float(np.max(expected_pattern)), 'unknowns': system.n_unknowns}
    return errors, details, (system, source_row, solution)


def _dipole_probe_errors(system, source_row, solution):
    """(|E_z| error / _DIPOLE_TOLERANCE, details) of the on-axis dipole solve at _DIPOLE_PROBE_CELLS in its equatorial
    plane vs the exact dipole field (worst probe)."""
    from .body_of_revolution import centre_fields
    fields, _ = centre_fields(system, solution)
    k = system.omega / C0
    measured, expected, relative = [], [], []
    for cell in _DIPOLE_PROBE_CELLS:
        distance = (cell + 0.5) * _DIPOLE_CELL
        kr = k * distance
        reference = (ETA0 * k * _DIPOLE_CURRENT * _DIPOLE_CELL / (4.0 * math.pi * distance)
                     * abs(1.0 + 1.0 / (1j * kr) - 1.0 / kr ** 2))
        value = abs(complex(fields['ez'][cell, source_row]))
        measured.append(value)
        expected.append(reference)
        relative.append(abs(value / reference - 1.0))
    return max(relative) / _DIPOLE_TOLERANCE, {
        'probe_r': float_list([(cell + 0.5) * _DIPOLE_CELL for cell in _DIPOLE_PROBE_CELLS]),
        'ez_abs': float_list(measured), 'ez_abs_expected': float_list(expected),
        'relative_errors': float_list(relative)}


def _check_radiation(results_by_name):
    """Exact flux identity, radiated power and far field of an on-axis dipole (m = 0) and a ring current (m = 2); the
    dipole's |E_z| probes in its equatorial plane vs the exact dipole field."""
    import scipy.special
    moment = _DIPOLE_CURRENT * _DIPOLE_CELL
    ring_radius = (_RING_CELL + 0.5) * _DIPOLE_CELL

    def dipole_pattern(wavenumber, theta):
        return ETA0 * wavenumber ** 2 * moment ** 2 * np.sin(theta) ** 2 / (32.0 * math.pi ** 2)

    def ring_pattern(wavenumber, theta):
        argument = wavenumber * ring_radius * np.sin(theta)
        lower, upper = (scipy.special.jv(_RING_ORDER + offset, argument) for offset in (-1, 1))
        return (ETA0 * wavenumber ** 2 * _DIPOLE_CURRENT ** 2 * ring_radius ** 2 / 8.0
                * (np.cos(theta) ** 2 * (0.5 * (lower + upper)) ** 2 + (0.5 * (lower - upper)) ** 2))

    errors, details = {}, {}
    for name, order, cell, component, pattern in (('dipole m = 0', 0, 0, 'z', dipole_pattern),
                                                  (f'ring m = {_RING_ORDER}', _RING_ORDER, _RING_CELL, 'phi',
                                                   ring_pattern)):
        case_errors, details[name], solve = _radiation_case(order, cell, component, pattern)
        errors.update({f'{name}: {key}': value for key, value in case_errors.items()})
        if order == 0:   # the axial dipole: its near / intermediate field at the probes
            errors[f'{name}: probe |E_z|'], details[f'{name} probes'] = _dipole_probe_errors(*solve)
    worst = max(errors, key=errors.get)
    details.update(normalised_errors=errors, worst_item=worst,
                   tolerances={'identity': _IDENTITY_TOLERANCE, 'power': _POWER_TOLERANCE,
                               'pattern': _PATTERN_TOLERANCE, 'far_power': _FAR_POWER_TOLERANCE,
                               'probe': _DIPOLE_TOLERANCE})
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0, 'details': details}


def _open_dielectric_assembly(eps_block):
    """BORAssembly of the small identity grid (_IDENTITY_WINDOW, _IDENTITY_PML) with an anisotropic block (eps_rr =
    eps_block, eps_phiphi = eps_block + 0.5, eps_zz = eps_block + 1) in _RECYCLE_BLOCK."""
    from .body_of_revolution import BORAssembly, BORMaterials
    config, widths, window, shape = _open_grid(_IDENTITY_WINDOW, _IDENTITY_PML)
    components = []
    for offset in (0.0, 0.5, 1.0):
        values = np.ones(shape, dtype=np.complex128)
        values[_RECYCLE_BLOCK] = eps_block + offset
        components.append(values)
    materials = BORMaterials(eps=components[1], mu=np.ones(shape, dtype=np.complex128),
                             pec=np.zeros(shape, dtype=bool), eps_rr=components[0], eps_zz=components[2])
    omega = 2.0 * math.pi * C0 / _DIPOLE_WAVELENGTH
    return BORAssembly(widths, config, _DIPOLE_CELL, materials, omega, window=window)


def _check_order_sweep(results_by_name):
    """One BORAssembly for every order: K0 + m K1 + m^2 K2 vs C(m)^T W C(m) - k0^2 M; a driven solve after a small
    eps change by GMRES on the previous factor (SparseDirectSolver.solve_recycled_host) vs a fresh factorisation."""
    import scipy.sparse
    import scipy.sparse.linalg
    from .body_of_revolution import point_current_rhs
    from ...solvers.sparse_direct import SparseDirectSolver
    assembly = _open_dielectric_assembly(_RECYCLE_EPS)
    k0_squared = (assembly.omega / C0) ** 2
    polynomial_errors = {}
    for order in _SWEEP_ORDERS:
        system = assembly.system(order)
        direct = (system.curl.T @ scipy.sparse.diags(system.a_weights) @ system.curl
                  - k0_squared * scipy.sparse.diags(system.mass))
        polynomial_errors[f'm = {order}'] = (scipy.sparse.linalg.norm(system.operator - direct)
                                             / scipy.sparse.linalg.norm(direct))
    solver = SparseDirectSolver(back_end='scipy_splu')
    system = assembly.system(_RECYCLE_ORDER)
    rhs = point_current_rhs(system, _RECYCLE_CELL, _IDENTITY_SOURCE_ROW, 'phi', _DIPOLE_CURRENT)
    old_factor = solver.factorize_host(system.operator)
    old_solution = old_factor.solve_host(rhs)
    changed = _open_dielectric_assembly(_RECYCLE_EPS + _RECYCLE_DELTA).system(_RECYCLE_ORDER)
    recycled, iterations = solver.solve_recycled_host(old_factor, changed.operator, rhs, old_solution)
    direct_solution = solver.factorize_host(changed.operator).solve_host(rhs)
    if recycled is None:
        recycle_error = math.inf
    else:
        recycle_error = float(np.linalg.norm(recycled - direct_solution) / np.linalg.norm(direct_solution))
    worst_polynomial = max(polynomial_errors, key=polynomial_errors.get)
    errors = {'polynomial': polynomial_errors[worst_polynomial] / _POLYNOMIAL_TOLERANCE,
              'recycled solve': recycle_error / _RECYCLE_TOLERANCE}   # None (inf) = GMRES missed RECYCLE_RTOL
    worst = max(errors, key=errors.get)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0,
            'details': {'polynomial_errors': {key: float(value) for key, value in polynomial_errors.items()},
                        'recycled_relative_difference': recycle_error, 'gmres_iterations': int(iterations),
                        'solution_change': float(np.linalg.norm(direct_solution - old_solution)
                                                 / np.linalg.norm(direct_solution)),
                        'unknowns': system.n_unknowns, 'normalised_errors': errors, 'worst_item': worst,
                        'tolerances': {'polynomial': _POLYNOMIAL_TOLERANCE, 'recycled': _RECYCLE_TOLERANCE}}}


def _coax_te11_root(inner_radius, outer_radius):
    """k_c [1/m] of the coaxial TE11 mode: first root of J1'(k a) Y1'(k b) - J1'(k b) Y1'(k a)."""
    import scipy.optimize
    import scipy.special

    def cross_product(k):
        return (scipy.special.jvp(1, k * outer_radius) * scipy.special.yvp(1, k * inner_radius)
                - scipy.special.jvp(1, k * inner_radius) * scipy.special.yvp(1, k * outer_radius))

    samples = np.linspace(0.05, 6.0, 1200) / outer_radius
    values = np.array([cross_product(k) for k in samples])
    crossing = int(np.flatnonzero(np.sign(values[:-1]) != np.sign(values[1:]))[0])
    return float(scipy.optimize.brentq(cross_product, samples[crossing], samples[crossing + 1]))


def _check_coax_ports(results_by_name):
    """Annular port lines on a coaxial guide: TEM (m = 0, exact discrete) and TE11 (m = 1, Bessel cross product)
    n_eff, annular vs axis line, the one-sided TE11 launch (transmission phase, reflection, flux)."""
    from .body_of_revolution import BORAssembly, BORMaterials, line_flux, yee_fields
    from .bor_ports import mode_amplitudes, mode_source_rhs, solve_port_modes
    from .boundaries import BoundaryConfig
    from ...solvers.sparse_direct import factorize_host
    config = BoundaryConfig(x_lo='pmc', x_hi='pec', y_lo='cpml', y_hi='cpml')
    config.cpml_npml = _GUIDE_PML
    shape = (_GUIDE_CELLS[0], _GUIDE_CELLS[1] + 2 * _GUIDE_PML)
    widths = tuple(np.full(count, _GUIDE_CELL) for count in shape)
    window = (slice(0, shape[0]), slice(_GUIDE_PML, _GUIDE_PML + _GUIDE_CELLS[1]))
    pec = np.zeros(shape, dtype=bool)
    pec[:_COAX_INNER_CELLS] = True
    materials = BORMaterials(eps=np.ones(shape, dtype=np.complex128), mu=np.ones(shape, dtype=np.complex128), pec=pec)
    line_pec = np.zeros((shape[0], 1), dtype=bool)
    line_pec[:_COAX_INNER_CELLS] = True
    line = BORMaterials(eps=np.ones((shape[0], 1), dtype=np.complex128), mu=np.ones((shape[0], 1), dtype=np.complex128),
                        pec=line_pec)
    omega = 2.0 * math.pi * C0 / _COAX_WAVELENGTH
    k0 = omega / C0

    def modes(order, start):
        return solve_port_modes(widths[0], _GUIDE_CELL, 'pec', config, _GUIDE_CELL, line, omega, order, 4, shape[0],
                                start=start)

    tem, te11, te11_axis_line = modes(0, _COAX_INNER_CELLS), modes(1, _COAX_INNER_CELLS), modes(1, 0)
    if not (tem.modes and te11.modes and te11_axis_line.modes):
        return {'measured': 0.0, 'expected': 1.0, 'error': math.inf, 'tolerance': 1.0,
                'details': {'violations': ['no guided mode found on a coax port line']}}
    expected_tem = 2.0 * math.asin(0.5 * k0 * _GUIDE_CELL) / (k0 * _GUIDE_CELL)
    # PEC cells drop their centre E_phi / E_z, so tangential E vanishes on the last PEC cell centre (not its hi edge)
    outer_radius, inner_radius = shape[0] * _GUIDE_CELL, (_COAX_INNER_CELLS - 0.5) * _GUIDE_CELL
    expected_te11 = math.sqrt(1.0 - (_coax_te11_root(inner_radius, outer_radius) / k0) ** 2)
    mode = te11.modes[0]
    system = BORAssembly(widths, config, _GUIDE_CELL, materials, omega, window=window).system(1)
    source_row, upstream_row, downstream_row = (_GUIDE_PML + row for row in (_GUIDE_SOURCE_ROW, _GUIDE_UPSTREAM_ROW,
                                                                             _GUIDE_DOWNSTREAM_ROW))
    solution = factorize_host(system.operator).solve_host(mode_source_rhs(system, te11, mode, source_row, 'positive',
                                                                          1.0))
    transmitted, _ = mode_amplitudes(system, te11, mode, solution, downstream_row)
    _, reflected = mode_amplitudes(system, te11, mode, solution, upstream_row)
    # flux line from the axis: the conductor-surface E_r pairs with its whole dual length (half inside the pec cells)
    flux = line_flux(system, yee_fields(system, solution), 'z', downstream_row, (0, shape[0]))
    phase_reference = mode.psi ** (downstream_row - source_row)
    errors = {'TEM n_eff': abs(complex(tem.modes[0].n_eff) - expected_tem) / _COAX_TEM_TOLERANCE,
              'TE11 n_eff': abs(complex(mode.n_eff).real - expected_te11) / _COAX_TE11_TOLERANCE,
              'annular vs axis line': abs(complex(mode.n_eff) - complex(te11_axis_line.modes[0].n_eff))
              / _COAX_SAME_LINE_TOLERANCE,
              'transmission': abs(transmitted / phase_reference - 1.0) / _GUIDE_PORT_TOLERANCE,
              'reflection': abs(reflected) ** 2 / _GUIDE_PORT_TOLERANCE,
              'flux': abs(flux - 1.0) / _GUIDE_PORT_TOLERANCE}
    worst = max(errors, key=errors.get)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0,
            'details': {'tem_n_eff': [float(tem.modes[0].n_eff.real), float(tem.modes[0].n_eff.imag)],
                        'tem_n_eff_expected': expected_tem,
                        'te11_n_eff': [float(mode.n_eff.real), float(mode.n_eff.imag)],
                        'te11_n_eff_expected': expected_te11,
                        'te11_n_eff_axis_line': float(te11_axis_line.modes[0].n_eff.real),
                        'transmission': [transmitted.real, transmitted.imag],
                        'transmission_expected': [phase_reference.real, phase_reference.imag],
                        'reflection_power': abs(reflected) ** 2, 'flux_W': flux, 'normalised_errors': errors,
                        'worst_item': worst}}


def _model_cavity_specs():
    """{'cavity': em_fdfd_bor PEC cylinder (1 x 1 um, 25 nm), m = 1, eigenfrequency near the cavity target}."""
    settings = {'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec', 'azimuthal_order': _MODEL_ORDER,
                'wavelength': _CAVITY_TARGET}
    size = _CAVITY_CELLS[0] * _CAVITY_CELL
    return {'cavity': ModelSpec(
        title='em_fdfd_bor PEC cylinder eigenfrequencies (m = 1)',
        domain=DomainSpec(size_x=size, size_y=size),
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=[FeatureSpec(_TABLE, 'resonance_table')])],
        mesh=MeshSpec(mode='uniform', dx=_MODEL_CELL),
        study=StudySpec(type=_EIGEN_STUDY, settings={'n_resonances': _MODEL_RESONANCES}))}


# ---- cuDSS on the Yee operator: the PEC cylinder filled with a lossy dielectric (complex symmetric, not Hermitian)
_CUDSS_EPS = 2.0 - 0.02j
_CUDSS_ORDERS = (0, 1)
_CUDSS_FACTORIZATIONS = ('cudss', 'cudss_mixed')
_CUDSS_TOLERANCE = 1e-9               # relative omega to the SuperLU solve (first estimate)


def _check_bor_cudss(results_by_name):
    """Every SuperLU resonance of the lossy cylinder per order matched by the nearest cudss / cudss_mixed one; both
    factors LDL^T (the FIT operator is complex symmetric, D = I), cudss_mixed Rayleigh-polished."""
    from ...solvers.arpack_shift_invert import ArpackShiftInvertSolver
    from .body_of_revolution import BORAssembly, BORMaterials, solve_bor_resonances
    from .boundaries import BoundaryConfig
    skipped = skip_without_cudss()
    if skipped is not None:
        return skipped
    config = BoundaryConfig(x_lo='pmc', x_hi='pec', y_lo='pec', y_hi='pec')
    shape = _CAVITY_CELLS
    widths = tuple(np.full(count, _CAVITY_CELL) for count in shape)
    materials = BORMaterials(eps=np.full(shape, _CUDSS_EPS, dtype=np.complex128),
                             mu=np.ones(shape, dtype=np.complex128), pec=np.zeros(shape, dtype=bool))
    assembly = BORAssembly(widths, config, _CAVITY_CELL, materials, 2.0 * math.pi * C0 / _CAVITY_TARGET)

    def omegas(eigen_solver):
        solves = {order: solve_bor_resonances(assembly.system(order), _CAVITY_RESONANCES, eigen_solver=eigen_solver)
                  for order in _CUDSS_ORDERS}
        values = {order: [complex(resonance.omega) for resonance in solve.resonances]
                  for order, solve in solves.items()}
        return values, solves

    reference, _ = omegas(None)
    errors, details = {}, {'omega_superlu': {str(order): [[value.real, value.imag] for value in values]
                                             for order, values in reference.items()}}
    violations = []
    for factorization in _CUDSS_FACTORIZATIONS:
        eigen_solver = ArpackShiftInvertSolver(factorization=factorization)
        found, solves = omegas(eigen_solver)
        error = max((min(abs(value - expected) for value in found[order]) / abs(expected) if found[order]
                     else math.inf) for order in _CUDSS_ORDERS for expected in reference[order])
        errors[factorization] = error
        if not eigen_solver.symmetric_factor:
            violations.append(f'{factorization}: the factor is no LDL^T (symmetry check failed)')
        polished = all('rayleigh_change' in solve.info for solve in solves.values())
        if factorization == 'cudss_mixed' and not polished:
            violations.append('cudss_mixed: no Rayleigh polish')
        details[factorization] = {'relative_error': error, 'symmetric_factor': bool(eigen_solver.symmetric_factor),
                                  'device_bytes': int(eigen_solver.device_bytes),
                                  'refinement_steps': int(eigen_solver.refinement_steps),
                                  'factor_texts': [solve.info.get('factor_text') for solve in solves.values()],
                                  'rayleigh_changes': [solve.info.get('rayleigh_change') for solve in solves.values()]}
    worst = max(errors, key=errors.get)
    details.update(violations=violations, worst_item=worst)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst] if not violations else math.inf,
            'tolerance': _CUDSS_TOLERANCE, 'details': details}


def _check_model_cavity(results_by_name):
    """Table omega of the model run vs the nearest analytic resonances, and one snapshot per resonance."""
    results = results_by_name['cavity']
    rows = results.result(_PHYSICS, _TABLE)['rows']
    size = _CAVITY_CELLS[0] * _CAVITY_CELL
    k_target = 2.0 * math.pi / _CAVITY_TARGET
    computed = [2.0 * math.pi * float(row[2]) / C0 for row in rows]
    worst, spurious = _match(computed, _analytic_cavity(_MODEL_ORDER, size, size), k_target,
                             min(_MODEL_RESONANCES - 1, len(computed)) or 1)
    labels = results.summary().get('snapshot_labels') or []
    snapshot_error = abs(len(labels) - _MODEL_RESONANCES)
    error = worst if not (spurious or snapshot_error) else math.inf
    return {'measured': worst, 'expected': 0.0, 'error': error, 'tolerance': _MODEL_TOLERANCE,
            'details': {'computed_k': float_list(computed), 'spurious': spurious, 'snapshot_labels': list(labels)}}


BOR_SELFTEST_CASES = (
    SelfTestCase(
        name='em_bor_pec_cylinder',
        label='EM body of revolution: PEC cylinder cavity',
        description=(
            'CPU check of physics/em/body_of_revolution (no model run): a vacuum cylinder of radius and height 1 um '
            f'with pec on r_hi, z_lo and z_hi (all walls on the grid edges), {_CAVITY_CELL * 1e9:g} nm cells, '
            f'shift-invert near {_CAVITY_TARGET * 1e6:g} um for azimuthal orders {", ".join(map(str, _CAVITY_ORDERS))} '
            f'({_CAVITY_RESONANCES} resonances each). The {_CAVITY_CHECKED} analytic resonances nearest the target, '
            'TM_mnp k^2 = (x_mn / a)^2 + (p pi / L)^2 (p >= 0) and TE_mnp with the zeros x\'_mn of J_m\' (p >= 1), '
            f'must each have a computed one within {_CAVITY_TOLERANCE:g} relative k (second-order scheme), and no '
            'computed resonance inside that band may match none (spurious modes). One BORAssembly serves every order '
            '(K0 + m K1 + m^2 K2 per axis class: the m sweep). Pins the ring metric, the axis rules per |m| (m = 0 '
            'H_z disc, |m| >= 1 no axis unknown), the pec walls and the static null-space filter. Orders 0..2 cover '
            'the three axis classes (m = 3 dropped; not yet confirmed by a run without it). Then m = '
            f'{_ON_RESONANCE_ORDER} is re-solved with its target on the discrete resonance nearest the first target '
            '(lossless: A = K - k_t^2 M singular there): the own factor at the offset shift k_t^2 (1 + '
            f'SHIFT_RELATIVE_OFFSET) must return it within {_ON_RESONANCE_TOLERANCE:g} relative omega and '
            f'{_CAVITY_RESONANCES} resonances.'),
        specs=lambda: {},
        check=_check_cavity,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_bor_uniaxial_cavity',
        label='EM body of revolution: uniaxial PEC cylinder cavity',
        description=(
            'CPU check of the diagonal anisotropy of physics/em/body_of_revolution (no model run): the cavity of '
            f'em_bor_pec_cylinder filled with eps_rr = eps_phiphi = {_UNIAXIAL_EPS_TRANSVERSE:g}, eps_zz = '
            f'{_UNIAXIAL_EPS_AXIAL:g} (optic axis along the rotation axis), orders '
            f'{", ".join(map(str, _UNIAXIAL_ORDERS))}. Exact: TE k^2 eps_t = (x\'_mn / a)^2 + (p pi / L)^2, TM k^2 '
            'eps_t = (p pi / L)^2 + eps_t / eps_z (x_mn / a)^2, matched as em_bor_pec_cylinder (same tolerance, no '
            'spurious resonance). Pins eps_rr on E_r, eps_phiphi on E_phi and eps_zz on E_z.'),
        specs=lambda: {},
        check=_check_uniaxial_cavity,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_bor_radiation',
        label='EM body of revolution: flux and far field',
        description=(
            'CPU check of the flux and far-field numerics of physics/em/body_of_revolution (no model run) on an open '
            f'vacuum grid ({_DIPOLE_CELL * 1e9:g} nm cells, {_DIPOLE_WINDOW[0] * _DIPOLE_CELL * 1e6:g} x '
            f'{_DIPOLE_WINDOW[1] * _DIPOLE_CELL * 1e6:g} um window, {_DIPOLE_PML}-cell cpml on r_hi and both z faces: '
            'cylindrical PML, the stretched r in every metric factor) at '
            f'{_DIPOLE_WAVELENGTH * 1e6:g} um: an on-axis z dipole (m = 0, I = 1 mA in the axis cell, moment I h) '
            f'and an azimuthal ring current I exp(-j m phi) (m = {_RING_ORDER}, radius '
            f'{(_RING_CELL + 0.5) * _DIPOLE_CELL * 1e6:g} um) inside a closed cell box (axis to '
            f'{_BOX_R_CELLS[1] * _DIPOLE_CELL * 1e6:g} um, {(_BOX_Z_ROWS[1] - _BOX_Z_ROWS[0]) * _DIPOLE_CELL * 1e6:g} um '
            'tall, 12 cells from the PML; the former 2 x 3 um window kept 20; not yet confirmed by a run on this '
            'grid). Normalised errors (error / tolerance, the case passes at <= 1): box_flux vs '
            f'source_power (the exact discrete Poynting theorem, {_IDENTITY_TOLERANCE:g}), vs the analytic radiated '
            f'power ({_POWER_TOLERANCE:g}: eta0 k^2 (I h)^2 / (12 pi) and the ring power integral), the far-field '
            'pattern of far_field_samples / far_field_power vs the closed forms U = eta0 k^2 (I h)^2 sin^2 theta / '
            '(32 pi^2) and U = eta0 k^2 I^2 rho0^2 / 8 [cos^2 theta ((J_(m-1) + J_(m+1)) / 2)^2 + ((J_(m-1) - '
            f'J_(m+1)) / 2)^2] (max deviation / peak, {_PATTERN_TOLERANCE:g}) and its solid-angle integral vs the '
            f'box flux ({_FAR_POWER_TOLERANCE:g}); the dipole solve also gives |E_z| at r = '
            f'{", ".join(f"{(cell + 0.5) * _DIPOLE_CELL * 1e6:g}" for cell in _DIPOLE_PROBE_CELLS)} um in its '
            'equatorial plane vs the exact dipole field eta0 k I h / (4 pi R) |1 + 1 / (j k R) - 1 / (k R)^2| '
            f'({_DIPOLE_TOLERANCE:g} relative; merged from the former em_bor_axial_dipole, which solved the same '
            'system). Pins the flux pairing and ring weights, the Bessel phi integral, the 2 pi factors, the source '
            'normalisation (current through the ring cross-section), the m = 0 axis and the radial PML.'),
        specs=lambda: {},
        check=_check_radiation,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_bor_order_sweep',
        label='EM body of revolution: order polynomial and factor recycling',
        description=(
            'CPU check of body_of_revolution.BORAssembly and the recycled host solve of solvers/sparse_direct (no model '
            f'run) on a small open grid ({_DIPOLE_CELL * 1e9:g} nm cells, window {_IDENTITY_WINDOW[0]} x '
            f'{_IDENTITY_WINDOW[1]} cells, {_IDENTITY_PML}-cell cylindrical PML: both identities hold on any grid; '
            'not yet confirmed by a run on it) with an anisotropic dielectric block (eps_rr / eps_phiphi / eps_zz = '
            f'{_RECYCLE_EPS:g} / {_RECYCLE_EPS + 0.5:g} / {_RECYCLE_EPS + 1:g}). Normalised errors '
            '(passes at <= 1): for m = -3..3 from one assembly the operator K0 + m K1 + m^2 K2 - k0^2 M vs the product '
            f'C(m)^T diag(V / mu) C(m) - k0^2 M (relative Frobenius norm, {_POLYNOMIAL_TOLERANCE:g}); a ring current '
            f'at m = {_RECYCLE_ORDER} solved after a {_RECYCLE_DELTA:g} eps change of the block by GMRES on the factor '
            'of the unchanged operator (SparseDirectSolver.solve_recycled_host, the thermo-optic Picard path of '
            'BORPhysics) vs a fresh factorisation (relative difference, '
            f'{_RECYCLE_TOLERANCE:g}; a GMRES miss of RECYCLE_RTOL fails). Pins the m sweep on one assembly and the '
            'recycling.'),
        specs=lambda: {},
        check=_check_order_sweep,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_bor_coax_ports',
        label='EM body of revolution: annular ports of a coaxial guide',
        description=(
            'CPU check of annular port lines of physics/em/bor_ports (no model run): a vacuum coaxial guide (inner '
            f'conductor = pec cells up to {_COAX_INNER_CELLS * _GUIDE_CELL * 1e6:g} um, outer pec r_hi at '
            f'{_GUIDE_CELLS[0] * _GUIDE_CELL * 1e6:g} um, {_GUIDE_CELL * 1e9:g} nm cells, cpml on both z faces) at '
            f'{_COAX_WAVELENGTH * 1e6:g} um, port lines from the inner conductor surface. Normalised errors (passes at '
            '<= 1): m = 0 TEM n_eff vs the exact discrete 2 asin(k h / 2) / (k h) (the 1 / r transverse field is '
            f'exact on the ring metric; {_COAX_TEM_TOLERANCE:g}), m = 1 TE11 n_eff vs sqrt(1 - (k_c / k)^2) with '
            "k_c the first root of J1'(k a) Y1'(k b) - J1'(k b) Y1'(k a), a = the centre of the last inner pec cell "
            '(where tangential E vanishes) '
            f'({_COAX_TE11_TOLERANCE:g}), the annular line vs a line from the axis (identical modes: the cells below '
            f'are pec; {_COAX_SAME_LINE_TOLERANCE:g}) and a 1 W TE11 launch toward +z (the downstream amplitude vs '
            'psi^rows: complex transmission, exact discrete launch; the upstream reflected power |a_-|^2: the '
            f'one-sided TFSF rhs; the downstream line_flux vs 1 W; {_GUIDE_PORT_TOLERANCE:g}). Pins the annular strip '
            '(pec inner end), the PEC-cell point materials, the |m| = 0 / 1 port paths, the 1 W normalisation and the '
            'reciprocity decomposition (the pipe TE11 ports with the |m| = 1 axis are the example '
            'em/analytic_bor_pipe_te11_section).'),
        specs=lambda: {},
        check=_check_coax_ports,
        max_wall_time=120.0),
    SelfTestCase(
        name='em_bor_cudss',
        label='EM body of revolution: cuDSS LDL^T factor',
        description=(
            'GPU (body_of_revolution directly, no model run): the em_bor_pec_cylinder grid filled with a lossy '
            f'dielectric eps = {_CUDSS_EPS.real:g} - {-_CUDSS_EPS.imag:g}j (complex symmetric, not Hermitian), orders '
            f'{", ".join(map(str, _CUDSS_ORDERS))}, solved with the eigen factorization superlu, cudss and '
            'cudss_mixed. Expected: every SuperLU resonance omega matched by the nearest one of each cuDSS solve to '
            f'{_CUDSS_TOLERANCE:g} relative (first estimate), both cuDSS factors the LDL^T (the finite-integration '
            'operator C^T W C - k0^2 M is complex symmetric: D = I, no symmetrizer needed) and every cudss_mixed solve '
            'Rayleigh-polished. Recorded: factor texts (unrefined complex64 residual), device bytes, Rayleigh '
            'changes. Skipped without nvmath-python. Written, not run.'),
        specs=lambda: {},
        check=_check_bor_cudss,
        backends=('gpu',)),
)
