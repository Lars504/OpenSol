"""Self-tests of the body-of-revolution FDTD driven order sum (symmetry 2.3, fdtd_bor.BOROrderSum): a CPU check of an
off-axis z + y (phi) dipole pair against the em_fdfd_bor order sum at omega_tilde and the closed forms, and a cpu-backend
model run of the same sources; physics imports inside the checks only."""
import math

import numpy as np

from ...core.constants import C0, ETA0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_centre, float_list
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)
from .selftests_fdtd_bor import (_BOX_R_CELLS, _BOX_Z_ROWS, _DIPOLE_CELL, _DIPOLE_CURRENT, _DIPOLE_PML, _DIPOLE_TIME,
                                 _DIPOLE_WAVELENGTH, _DIPOLE_WINDOW, open_vacuum_grid)

_TYPE = 'em_fdtd_bor'
_PHYSICS = 'em'

# ---- a z and a y (= phi at phi = 0) point current of moment I d (d = 25 nm) in one cell rho_s = 2.5 cells off the axis
# (62.5 nm, k rho_s = 0.39 at 1 um: J_4(k rho_s)^2 ~ 1e-9, so orders 0..3 hold the power), both with the gaussian pulse,
# on the em_fdtd_bor_radiated_power grid (vacuum, cpml). Per order m > 0 two steppers (r / z and phi families), m = 0
# one. Exact: each dipole radiates P = eta0 k^2 (I d)^2 / (12 pi) (orthogonal dipoles: powers add); in the meridional
# plane through them (phi = 0 and pi) the z dipole gives E_theta with U_max sin^2 theta, the y dipole E_phi with U_max
# (U_max = 3 P / (8 pi)); phi averages U_max sin^2 theta and U_max (1 - sin^2 theta / 2).
_ORDERS = (0, 1, 2, 3)
_SOURCE_CELL = 2
# Analysed frequencies / f0: the carrier only. The order-sum logic (families, term phases, phi = 0 weights) does not
# depend on frequency and em_fdtd_bor_radiated_power covers the DFT timing at 0.9 / 1.0 / 1.1 f0; each extra frequency
# costs a full-layout DFT on the 7 members and 4 em_fdfd_bor factorisations.
_ORDER_SUM_FACTORS = (1.0,)
_FDFD_ORDER_TOLERANCE = 1e-2      # per-order power vs em_fdfd_bor at omega_tilde, / total (time-domain CPML vs stretch)
_FDFD_PATTERN_TOLERANCE = 1e-2    # summed far field vs em_fdfd_bor, / U_max
_FDFD_FIELD_TOLERANCE = 2e-2      # phi = 0 phasor E in the box vs em_fdfd_bor, relative L2
_POWER_TOLERANCE = 0.04           # vs the closed form (discrete source, PML reflection; as em_bor_offaxis_dipole)
_PATTERN_TOLERANCE = 0.05         # max |U - U_closed form| / U_max
_SUM_TOLERANCE = 1e-9             # power_orders summed vs power (model results)
_THETA = 361


def _dipole_reference(frequency):
    """(wavenumber [1/m], power P [W], U_max [W/sr]) of one dipole of moment I d at frequency [Hz]."""
    wavenumber = 2.0 * math.pi * frequency / C0
    power = float(ETA0) * wavenumber ** 2 * (_DIPOLE_CURRENT * _DIPOLE_CELL) ** 2 / (12.0 * math.pi)
    return wavenumber, power, 3.0 * power / (8.0 * math.pi)


def _source_row():
    return _DIPOLE_PML + _DIPOLE_WINDOW[1] // 2


def _fdfd_terms(widths, config, materials, window, omega):
    """[(signed order, BORSystem, yee_fields)] of the em_fdfd_bor order sum of the dipole pair at omega (term rhs:
    ring current I (z), I d / (2 pi rho_s) (phi))."""
    from ...solvers.sparse_direct import factorize_host
    from .body_of_revolution import BORAssembly, order_pair_fields, point_current_rhs
    assembly = BORAssembly(widths, config, _DIPOLE_CELL, materials, omega, window=window)
    radius = assembly.r_axis.real_centres[_SOURCE_CELL]
    phi_current = _DIPOLE_CURRENT * assembly.dr_cells[_SOURCE_CELL] / (2.0 * math.pi * radius)
    terms = []
    for order in _ORDERS:
        system = assembly.system(order)
        rhs = (point_current_rhs(system, _SOURCE_CELL, _source_row(), 'z', _DIPOLE_CURRENT)
               + point_current_rhs(system, _SOURCE_CELL, _source_row(), 'phi', phi_current))
        factor = factorize_host(system.operator)
        terms.extend((signed, system, fields) for signed, fields in order_pair_fields(system, rhs, factor.solve_host))
    return terms


def _check_order_sum(results_by_name):
    """BOROrderSum of the dipole pair vs the em_fdfd_bor order sum at omega_tilde (per-order box flux, summed far field,
    phi = 0 phasor field) and vs the closed forms (power, meridional pattern)."""
    from ...core.sources import GaussianPulse
    from .body_of_revolution import box_flux, far_field_samples, order_sum_far_field
    from .fdtd_bor import BOROrderSum, source_spectrum
    config, widths, window, materials = open_vacuum_grid()
    engine = BOROrderSum(widths, config, _DIPOLE_CELL, materials, _ORDERS, ('z', 'phi'), window=window)
    f0 = C0 / _DIPOLE_WAVELENGTH
    pulse = GaussianPulse(f0, 0.5)
    for component in ('z', 'phi'):
        engine.add_current(_SOURCE_CELL, _source_row(), component, _DIPOLE_CURRENT, pulse)
    frequencies = f0 * np.array(_ORDER_SUM_FACTORS)
    spectra = engine.add_spectra(frequencies)
    for _ in range(int(math.ceil(_DIPOLE_TIME / engine.dt))):
        engine.step()
    spectra.scale(1.0 / source_spectrum(pulse, engine.dt, frequencies, engine.n_step))
    z_rows = (_DIPOLE_PML + _BOX_Z_ROWS[0], _DIPOLE_PML + _BOX_Z_ROWS[1])
    theta = np.linspace(0.0, math.pi, _THETA)
    sin_squared = np.sin(theta) ** 2
    errors, rows = {}, []
    for index, frequency in enumerate(frequencies):
        label = f'{frequency / f0:g} f0'
        omega = 2.0 * math.pi * frequency
        _, expected_power, peak = _dipole_reference(frequency)
        terms = engine.spectra_terms(spectra, index)
        reference = _fdfd_terms(widths, config, materials, window,
                                2.0 / engine.dt * math.sin(0.5 * omega * engine.dt))
        fdtd_orders = np.zeros(len(_ORDERS))
        fdfd_orders = np.zeros(len(_ORDERS))
        for signed, fields in terms:
            fdtd_orders[_ORDERS.index(abs(signed))] += box_flux(engine.assembly, fields, _BOX_R_CELLS, z_rows)
        for signed, system, fields in reference:
            fdfd_orders[_ORDERS.index(abs(signed))] += box_flux(system, fields, _BOX_R_CELLS, z_rows)
        total = float(np.sum(fdtd_orders))
        errors[f'{label}: order powers vs fdfd'] = (float(np.max(np.abs(fdtd_orders - fdfd_orders)))
                                                    / float(np.sum(fdfd_orders)) / _FDFD_ORDER_TOLERANCE)
        errors[f'{label}: power'] = abs(total / (2.0 * expected_power) - 1.0) / _POWER_TOLERANCE
        fdtd_samples = [(signed, far_field_samples(engine.assembly, fields, _BOX_R_CELLS, z_rows))
                        for signed, fields in terms]
        fdfd_samples = [(signed, far_field_samples(system, fields, _BOX_R_CELLS, z_rows))
                        for signed, system, fields in reference]
        for side, phi in (('phi = 0', 0.0), ('phi = pi', math.pi)):
            _, power_theta, power_phi, _ = order_sum_far_field(fdtd_samples, omega, theta, phi)
            reference_total = order_sum_far_field(fdfd_samples, omega, theta, phi)[0]
            errors[f'{label}: pattern vs fdfd {side}'] = (float(np.max(np.abs(power_theta + power_phi
                                                                             - reference_total)))
                                                          / peak / _FDFD_PATTERN_TOLERANCE)
            errors[f'{label}: E_theta pattern {side}'] = (float(np.max(np.abs(power_theta - peak * sin_squared)))
                                                          / peak / _PATTERN_TOLERANCE)
            errors[f'{label}: E_phi pattern {side}'] = (float(np.max(np.abs(power_phi - peak))) / peak
                                                        / _PATTERN_TOLERANCE)
        shown = engine.harmonic_yee(spectra, index)   # physical phi = 0 phasors
        (i0, i1), (j0, j1) = _BOX_R_CELLS, z_rows
        boxes = {'er': (slice(i0, i1 + 1), slice(j0, j1)), 'ephi': (slice(i0, i1), slice(j0, j1)),
                 'ez': (slice(i0, i1), slice(j0, j1 + 1))}
        difference = reference_norm = 0.0
        for name, cells in boxes.items():
            fdfd_value = sum(fields[name] for _, _, fields in reference)[cells]
            difference += float(np.sum(np.abs(np.asarray(shown[name])[cells] - fdfd_value) ** 2))
            reference_norm += float(np.sum(np.abs(fdfd_value) ** 2))
        errors[f'{label}: phi = 0 field vs fdfd'] = math.sqrt(difference / reference_norm) / _FDFD_FIELD_TOLERANCE
        rows.append({'frequency_hz': float(frequency), 'fdtd_order_powers_W': float_list(fdtd_orders),
                     'fdfd_order_powers_W': float_list(fdfd_orders), 'analytic_power_W': 2.0 * expected_power})
    worst = max(errors, key=errors.get)
    details = {'per_frequency': rows, 'dt_s': engine.dt, 'steps': engine.n_step, 'members': len(engine.members),
               'normalised_errors': errors, 'worst_item': worst}
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0, 'details': details}


# ---- model run (cpu backend): the same pair through the model layer, flux lines closing the box of window rows 16..65
# and r edges 0..40, a far-field box, one monitor wavelength (1 um)
_MODEL_WAVELENGTH = 1.0e-6
_MODEL_BOX = (40, (16, 65))       # r edge of the side face; window z edges of the bottom / top faces
# orders of the model run: the phi = 0 pattern adds the order amplitudes coherently, so a cut at |m| <= M errs by
# amplitude terms, not power shares (x = k rho_s = 0.39): M = 1 gives |J0 + 2j J1|^2 - 1 = +7 % (E_theta) and
# |J0 - J2 + j J1|^2 - 1 = -7 % (E_phi, the phi current pairs J_{m -+ 1}), M = 2 still +4 % on E_phi
# (|J0 - J2 + j (2 J1 - J3)|^2), M = 3 < 0.1 %; as the CPU case
_MODEL_ORDERS = (0, 1, 2, 3)


def _model_specs():
    """{'run': em_fdtd_bor time_dependent order sum (azimuthal_orders _MODEL_ORDERS) of the z + phi point sources, flux
    lines and far field, cpu backend}."""
    cell = _DIPOLE_CELL
    size_x, size_y = (count * cell for count in _DIPOLE_WINDOW)
    source = {'x': cell_centre(_SOURCE_CELL, cell), 'y': cell_centre(_DIPOLE_WINDOW[1] // 2, cell)}
    side_edge, (bottom_edge, top_edge) = _MODEL_BOX
    top_y, bottom_y = cell_centre(top_edge - 1, cell), cell_centre(bottom_edge - 1, cell)   # line rows below each face
    side_x = cell_centre(side_edge - 1, cell)
    line_end = side_edge * cell - cell / 4.0
    geometry = [GeometrySpec('z_point', 'point', dict(source)), GeometrySpec('phi_point', 'point', dict(source)),
                GeometrySpec('top_line', 'line', {'x0': 0.0, 'y0': top_y, 'x1': line_end, 'y1': top_y}),
                GeometrySpec('bottom_line', 'line', {'x0': 0.0, 'y0': bottom_y, 'x1': line_end, 'y1': bottom_y}),
                GeometrySpec('side_line', 'line', {'x0': side_x, 'y0': bottom_edge * cell + cell / 4.0,
                                                   'x1': side_x, 'y1': top_edge * cell - cell / 4.0}),
                GeometrySpec('far_box', 'rectangle', {'x0': 0.0, 'y0': source['y'] - 0.7e-6, 'x1': 1.1e-6,
                                                      'y1': source['y'] + 0.7e-6})]
    pulse = {'waveform': 'gaussian', 'wavelength': _DIPOLE_WAVELENGTH, 'bandwidth': 0.5, 'current': _DIPOLE_CURRENT}
    band = {'wavelength_start': _MODEL_WAVELENGTH, 'wavelength_stop': _MODEL_WAVELENGTH, 'n_wavelengths': 1}
    features = [FeatureSpec('z_dipole', 'point_source', ['z_point'], {**pulse, 'component': 'z'}),
                FeatureSpec('y_dipole', 'point_source', ['phi_point'], {**pulse, 'component': 'phi'}),
                *(FeatureSpec(name, 'flux_monitor', [f'{name}_line'], dict(band)) for name in ('top', 'bottom', 'side')),
                FeatureSpec('far', 'far_field', ['far_box'], {**band, 'n_angles': 360})]
    settings = {'bc_x_hi': 'cpml', 'bc_y_lo': 'cpml', 'bc_y_hi': 'cpml', 'azimuthal_orders': list(_MODEL_ORDERS)}
    return {'run': ModelSpec(
        title='em_fdtd_bor off-axis dipole pair, driven order sum',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=cell),
        study=StudySpec(type='time_dependent', settings={'duration': _DIPOLE_TIME}),
        solver_defaults=SolverDefaultsSpec(backend='cpu'))}


def _check_model(results_by_name):
    """Box power and per-order sums, meridional E_theta / E_phi patterns and the phi average vs the closed forms, the
    order-sum extras."""
    results = results_by_name['run']
    _, expected_power, peak = _dipole_reference(C0 / _MODEL_WAVELENGTH)
    signs = {'top': 1.0, 'bottom': -1.0, 'side': 1.0}
    faces = {name: results.result(_PHYSICS, name) for name in signs}
    violations = [f'{name}: azimuthal_orders {face.get("azimuthal_orders")}' for name, face in faces.items()
                  if list(face.get('azimuthal_orders') or []) != list(_MODEL_ORDERS)
                  or np.shape(face.get('power_orders_per_cw')) != (1, len(_MODEL_ORDERS))]
    if violations:
        return {'measured': math.inf, 'expected': 0.0, 'error': math.inf, 'tolerance': 1.0,
                'details': {'violations': violations}}
    total = sum(sign * float(faces[name]['power_per_cw'][0]) for name, sign in signs.items())
    sums = max(abs(float(np.sum(face['power_orders_per_cw'][0])) / float(face['power_per_cw'][0]) - 1.0)
               for face in faces.values())
    far = results.result(_PHYSICS, 'far')
    angles = np.asarray(far['angles'], dtype=np.float64)
    sin_squared = np.cos(angles) ** 2   # sin(theta) = |cos(angle)| in the meridional plane
    power_theta = np.asarray(far['power_theta'], dtype=np.float64)[0]
    power_phi = np.asarray(far['power_phi'], dtype=np.float64)[0]
    mean = np.asarray(far['power_azimuthal_mean'], dtype=np.float64)[0]
    errors = {'power': abs(total / (2.0 * expected_power) - 1.0) / _POWER_TOLERANCE,
              'power_orders sum': sums / _SUM_TOLERANCE,
              'E_theta pattern': float(np.max(np.abs(power_theta - peak * sin_squared))) / peak / _PATTERN_TOLERANCE,
              'E_phi pattern': float(np.max(np.abs(power_phi - peak))) / peak / _PATTERN_TOLERANCE,
              'azimuthal mean': (float(np.max(np.abs(mean - peak * (1.0 + 0.5 * sin_squared)))) / peak
                                 / _PATTERN_TOLERANCE)}
    if (np.shape(far['power_orders']) != (1, len(_MODEL_ORDERS), angles.size)
            or list(far['azimuthal_orders']) != list(_MODEL_ORDERS)):
        errors['far-field extras'] = math.inf
    worst = max(errors, key=errors.get)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0,
            'details': {'power_W': total, 'analytic_power_W': 2.0 * expected_power, 'analytic_peak_W_per_sr': peak,
                        'pattern_peak_W_per_sr': float(np.max(power_theta + power_phi)), 'normalised_errors': errors,
                        'worst_item': worst}}


FDTD_BOR_ORDER_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdtd_bor_offaxis_dipole',
        label='EM body-of-revolution FDTD: off-axis dipoles as an order sum',
        description=(
            'CPU check of fdtd_bor.BOROrderSum (no model run) on the em_fdtd_bor_radiated_power grid (vacuum, cpml): '
            f'a z and a y (phi at phi = 0) point current of moment I d in the cell {_SOURCE_CELL + 0.5:g} cells off '
            f'the axis, orders {", ".join(map(str, _ORDERS))} (r / z and phi families per m > 0, all in lockstep), one '
            f'gaussian pulse; at {", ".join(map(str, _ORDER_SUM_FACTORS))} f0 the DFT phasors / S(omega) as travelling '
            'terms (order_term_fields). Normalised errors (passes at <= 1) vs the em_fdfd_bor order sum of the same '
            'elements at omega_tilde = (2 / dt) sin(omega dt / 2) on the same grid: box flux per order pair '
            f'({_FDFD_ORDER_TOLERANCE:g} of the total), the summed far field at phi = 0 and pi '
            f'({_FDFD_PATTERN_TOLERANCE:g} of U_max), the phi = 0 phasor E inside the box (harmonic_yee, '
            f'{_FDFD_FIELD_TOLERANCE:g} relative); vs the closed forms: power 2 eta0 k^2 (I d)^2 / (12 pi) '
            f'({_POWER_TOLERANCE:g}) and the meridional patterns U_max sin^2 theta (E_theta) and U_max (E_phi) '
            f'({_PATTERN_TOLERANCE:g} of U_max). Pins the member currents and families, the travelling-term phases '
            '(the quarter-period j of the phi family), the phi = 0 weights and the lockstep time step. Run length '
            f'{_DIPOLE_TIME * 1e15:g} fs, not yet confirmed by a run.'),
        specs=lambda: {},
        check=_check_order_sum,
        heavy=True,
        max_wall_time=600.0),
    SelfTestCase(
        name='em_fdtd_bor_offaxis_dipole_model',
        label='EM body-of-revolution FDTD: off-axis dipole order-sum model run',
        description=(
            'Model run of em_fdtd_bor on the cpu backend (time_dependent, '
            f'{_DIPOLE_TIME * 1e15:g} fs, azimuthal_orders {list(_MODEL_ORDERS)}: the driven order sum; the phi = 0 '
            'pattern adds the order amplitudes coherently, so a cut at |m| <= 1 errs ~7 % and |m| <= 2 ~4 % (E_phi) of '
            'U_max) of a z and a phi point_source in one cell '
            f'{_SOURCE_CELL + 0.5:g} cells off the axis in vacuum, flux_monitor lines closing a box and a far_field '
            f'box, one monitor wavelength ({_MODEL_WAVELENGTH * 1e6:g} um). Normalised errors (passes at <= 1): top '
            f'- bottom + side power_per_cw vs 2 eta0 k^2 (I d)^2 / (12 pi) ({_POWER_TOLERANCE:g}), power_orders_per_cw '
            f'summing to power_per_cw ({_SUM_TOLERANCE:g}), power_theta vs U_max cos^2(angle), power_phi vs U_max and '
            f'power_azimuthal_mean vs U_max (1 + cos^2(angle) / 2) over the full 360 degrees ({_PATTERN_TOLERANCE:g} '
            'of U_max), the azimuthal_orders / power_orders extras. Pins the model layer of the order sum (create, '
            'members from the source components at finalize, monitors and results).'),
        specs=_model_specs,
        check=_check_model,
        heavy=True,
        max_wall_time=900.0),
)
