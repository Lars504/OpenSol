"""Self-tests of the body-of-revolution FDTD (symmetry 2.3, fdtd_bor): CPU checks against em_fdfd_bor on the same grid
(cavity ring-down vs the leap-frog-corrected eigenfrequencies, radiated power of a dipole and a ring current) and a
cpu-backend model run of the ring-down; physics imports inside the checks only."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_centre, float_list
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)

_TYPE = 'em_fdtd_bor'
_PHYSICS = 'em'

# ---- PEC cylinder cavity (a = 0.96 um, L = 1.2 um: no accidental TE / TM degeneracy), vacuum, walls on the grid
# edges. A gaussian 'z' + 'phi' current pulse at 1.5 um rings the modes of each order; harmonic inversion of an E_z and
# an E_phi trace vs the em_fdfd_bor resonances of the same operator, mapped by the leap-frog relation
# omega_FDTD = (2 / dt) asin(omega_FDFD dt / 2) (exact up to float32 rounding and the inversion's resolution; the
# uncorrected mismatch (omega dt)^2 / 24 ~ 5e-4 is far above the tolerance).
_RING_CELL = 40e-9
_RING_CELLS = (24, 30)
_RING_TARGET = 1.5e-6                 # m: pulse carrier and FDFD shift
_RING_ORDERS = (0, 2)                 # m = 1: em_fdtd_bor_model_ringdown (same cavity, operator and tolerance)
_RING_BAND = (0.7, 1.3)               # analysed band / f0 (the pulse spectrum stays above 0.7 of its peak)
_RING_BAND_MARGIN = 0.02              # FDFD resonances this close to a band edge are not required
_RING_BANDWIDTH = 0.5
_RING_STEPS = 12000
_RING_SOURCE_CELL = (7, 11)
_RING_PROBE_CELL = (15, 20)
_RING_FDFD_RESONANCES = 16
_RING_TOLERANCE = 1e-5                # relative frequency
_RING_SPURIOUS_AMPLITUDE = 1e-2       # FDTD lines above this share of the strongest must match an FDFD resonance

# ---- radiation in vacuum: cpml on r_hi and both z faces, an on-axis z dipole (m = 0) and an azimuthal ring current
# (m = 2), gaussian pulse at 1 um; per frequency the DFT phasors / S(omega) give the closed-box flux, which must equal
# the source power of the same phasors (exact discrete Poynting theorem at omega_tilde) and the em_fdfd_bor source
# power at omega_tilde on the same grid (the PMLs differ: the complex stretch vs the time-domain CPML convolutions).
_DIPOLE_CELL = 25e-9
_DIPOLE_WAVELENGTH = 1.0e-6
_DIPOLE_PML = 12
_DIPOLE_WINDOW = (48, 80)
_DIPOLE_CURRENT = 1e-3                # A
# s simulated: the pulse ends at 2 t0 ~ 19 fs and the fields leave the 1.2 x 2 um window within ~5 fs (sin carrier:
# no static field); the CPU radiation case reports the final / peak field energy (sampled every _ENERGY_SAMPLE_STEPS)
_DIPOLE_TIME = 50e-15
_ENERGY_SAMPLE_STEPS = 20
_DIPOLE_FACTORS = (0.9, 1.0, 1.1)     # analysed frequencies / f0
_DIPOLE_CASES = (('dipole m = 0', 0, 0, 'z'), ('ring m = 2', 2, 8, 'phi'))   # (name, order, r cell, component)
_BOX_R_CELLS = (0, 40)
_BOX_Z_ROWS = (10, 70)                # window rows (+ the PML offset)
_IDENTITY_TOLERANCE = 1e-4            # box flux vs the phasors' source power (float32 fields)
_FDFD_POWER_TOLERANCE = 1e-2          # vs em_fdfd_bor (PML reflections ~1e-3)

# ---- model run (cpu backend): the cavity at m = 1 through the model layer, z + phi sources, resonance_monitors on
# E_z (TM110 has no E_phi) and E_phi (TE111 has no E_z)
_MODEL_ORDER = 1
_MODEL_MONITORS = {'ring_ez': 'ez', 'ring_ephi': 'ephi'}   # feature name -> recorded component
_MODEL_DURATION = 1.0e-12             # s
_MODEL_TOLERANCE = 1e-5


def _cavity_setup():
    """(BoundaryConfig, cell widths, vacuum BORMaterials) of the PEC cylinder."""
    from .body_of_revolution import BORMaterials
    from .boundaries import BoundaryConfig
    config = BoundaryConfig(x_lo='pmc', x_hi='pec', y_lo='pec', y_hi='pec')
    widths = tuple(np.full(count, _RING_CELL) for count in _RING_CELLS)
    ones = np.ones(_RING_CELLS, dtype=np.complex128)
    return config, widths, BORMaterials(eps=ones, mu=ones, pec=np.zeros(_RING_CELLS, dtype=bool))


def _corrected_resonances(order, dt):
    """(leap-frog-corrected, raw) em_fdfd_bor resonance frequencies [Hz] of the cavity at `order` (time step dt)."""
    from .body_of_revolution import BORAssembly, solve_bor_resonances
    config, widths, materials = _cavity_setup()
    omega_target = 2.0 * math.pi * C0 / _RING_TARGET
    solve = solve_bor_resonances(BORAssembly(widths, config, _RING_CELL, materials, omega_target).system(order),
                                 _RING_FDFD_RESONANCES)
    raw = np.array([resonance.omega.real for resonance in solve.resonances]) / (2.0 * math.pi)
    corrected = 2.0 / dt * np.arcsin(np.pi * raw * dt) / (2.0 * math.pi)   # asin(omega dt / 2)
    return corrected, raw


def _match_lines(found, corrected, raw, band):
    """(worst relative error of the corrected FDFD lines inside the band vs the nearest FDTD line, matched count,
    details, violations); found = [(frequency, relative amplitude)]."""
    lo, hi = band[0] * (1.0 + _RING_BAND_MARGIN), band[1] * (1.0 - _RING_BAND_MARGIN)
    required = [(value, raw_value) for value, raw_value in zip(corrected, raw) if lo <= value <= hi]
    frequencies = np.array([frequency for frequency, _ in found])
    violations, errors, uncorrected = [], [], []
    for value, raw_value in required:
        if not frequencies.size:
            errors.append(math.inf)
            continue
        nearest = float(frequencies[np.argmin(np.abs(frequencies - value))])
        errors.append(abs(nearest / value - 1.0))
        uncorrected.append(abs(nearest / raw_value - 1.0))
    if not required:
        violations.append('no em_fdfd_bor resonance inside the band')
    for frequency, amplitude in found:
        if amplitude >= _RING_SPURIOUS_AMPLITUDE and not np.any(np.abs(corrected / frequency - 1.0)
                                                                <= 10.0 * _RING_TOLERANCE):
            violations.append(f'FDTD line {frequency:.6g} Hz (amplitude {amplitude:.2g}) matches no FDFD resonance')
    details = {'fdfd_corrected_hz': float_list([value for value, _ in required]),
               'fdtd_lines_hz': float_list(sorted(frequencies)), 'relative_errors': float_list(errors),
               'uncorrected_relative_errors': float_list(uncorrected)}
    return max(errors, default=math.inf), len(errors), details, violations


def _check_ringdown(results_by_name):
    """Ring-down lines of the PEC cylinder per order vs the leap-frog-corrected em_fdfd_bor resonances."""
    from ...core.resonance import harmonic_inversion
    from ...core.sources import GaussianPulse
    from .fdtd_bor import BORTimeStepper
    config, widths, materials = _cavity_setup()
    f0 = C0 / _RING_TARGET
    band = (_RING_BAND[0] * f0, _RING_BAND[1] * f0)
    errors, details, violations = {}, {}, []
    for order in _RING_ORDERS:
        stepper = BORTimeStepper(widths, config, _RING_CELL, materials, order)
        pulse = GaussianPulse(f0, _RING_BANDWIDTH)
        for component in ('z', 'phi'):
            stepper.add_current(*_RING_SOURCE_CELL, component, 1e-3, pulse)
        ix, iz = _RING_PROBE_CELL
        traces = {name: np.empty(_RING_STEPS) for name in ('ez', 'ephi')}
        for step in range(_RING_STEPS):
            stepper.step()
            traces['ez'][step] = stepper.fields.ez[ix, iz + 1]
            traces['ephi'][step] = stepper.fields.ephi[ix, iz]
        first = int(math.ceil(pulse.duration / stepper.dt))   # trace sample k is at (k + 1) dt: after the pulse
        found = []
        for trace in traces.values():
            analysis = harmonic_inversion(trace[first:].astype(np.float32), stepper.dt, *band)
            peak = float(np.max(analysis['amplitude'], initial=0.0))
            found.extend((float(frequency), float(amplitude) / peak)
                         for frequency, amplitude in zip(analysis['frequency'], analysis['amplitude']))
        corrected, raw = _corrected_resonances(order, stepper.dt)
        error, matched, order_details, order_violations = _match_lines(found, corrected, raw, band)
        errors[f'm = {order}'] = error
        violations.extend(f'm = {order}: {text}' for text in order_violations)
        details[f'm = {order}'] = {**order_details, 'dt_s': stepper.dt, 'matched': matched}
    worst = max(errors, key=errors.get)
    error = errors[worst] if not violations else math.inf
    details.update(errors=errors, worst_item=worst, violations=violations)
    return {'measured': errors[worst], 'expected': 0.0, 'error': error, 'tolerance': _RING_TOLERANCE,
            'details': details}


def open_vacuum_grid():
    """(BoundaryConfig, cell widths, window, vacuum BORMaterials) of the radiation grid: cpml on r_hi and both z faces."""
    from .body_of_revolution import BORMaterials
    from .boundaries import BoundaryConfig
    config = BoundaryConfig(x_lo='pmc', x_hi='cpml', y_lo='cpml', y_hi='cpml')
    config.cpml_npml = _DIPOLE_PML
    shape = (_DIPOLE_WINDOW[0] + _DIPOLE_PML, _DIPOLE_WINDOW[1] + 2 * _DIPOLE_PML)
    widths = tuple(np.full(count, _DIPOLE_CELL) for count in shape)
    window = (slice(0, _DIPOLE_WINDOW[0]), slice(_DIPOLE_PML, _DIPOLE_PML + _DIPOLE_WINDOW[1]))
    materials = BORMaterials(eps=np.ones(shape, dtype=np.complex128), mu=np.ones(shape, dtype=np.complex128),
                             pec=np.zeros(shape, dtype=bool))
    return config, widths, window, materials


def _radiation_case(order, source_cell, component):
    """(normalised errors, details) of one radiating source: box flux of the FDTD phasors vs their source power and vs
    the em_fdfd_bor source power at omega_tilde, per analysed frequency."""
    from ...core.sources import GaussianPulse
    from ...solvers.sparse_direct import factorize_host
    from .body_of_revolution import assemble_bor, box_flux, point_current_rhs, source_power
    from .fdtd_bor import BORTimeStepper, phasor_yee, source_spectrum
    config, widths, window, materials = open_vacuum_grid()
    stepper = BORTimeStepper(widths, config, _DIPOLE_CELL, materials, order, window=window)
    f0 = C0 / _DIPOLE_WAVELENGTH
    pulse = GaussianPulse(f0, 0.5)
    source_row = _DIPOLE_PML + _DIPOLE_WINDOW[1] // 2
    injection = stepper.add_current(source_cell, source_row, component, _DIPOLE_CURRENT, pulse)
    frequencies = f0 * np.array(_DIPOLE_FACTORS)
    spectra = stepper.add_spectra(frequencies)
    peak_energy = 0.0
    for step in range(int(math.ceil(_DIPOLE_TIME / stepper.dt))):
        stepper.step()
        if step % _ENERGY_SAMPLE_STEPS == 0:
            peak_energy = max(peak_energy, stepper.field_energy())
    final_energy_ratio = stepper.field_energy() / peak_energy if peak_energy > 0.0 else math.nan
    spectrum = source_spectrum(pulse, stepper.dt, frequencies, stepper.n_step)
    z_rows = (_DIPOLE_PML + _BOX_Z_ROWS[0], _DIPOLE_PML + _BOX_Z_ROWS[1])
    errors, rows = {}, []
    for index, frequency in enumerate(frequencies):
        e_values, a_values = spectra.full_phasors(index, 1.0 / spectrum[index])
        flux = box_flux(stepper.assembly, phasor_yee(e_values, a_values, stepper.shape), _BOX_R_CELLS, z_rows)
        delivered = math.pi * float(np.real(np.vdot(e_values[injection.indices], injection.charges)))
        omega_tilde = 2.0 / stepper.dt * math.sin(math.pi * frequency * stepper.dt)
        system = assemble_bor(widths, config, _DIPOLE_CELL, materials, omega_tilde, order, window=window)
        rhs = point_current_rhs(system, source_cell, source_row, component, _DIPOLE_CURRENT)
        reference = source_power(system, factorize_host(system.operator).solve_host(rhs), rhs)
        label = f'{frequency / f0:g} f0'
        errors[f'{label}: identity'] = abs(flux / delivered - 1.0) / _IDENTITY_TOLERANCE
        errors[f'{label}: fdfd power'] = abs(flux / reference - 1.0) / _FDFD_POWER_TOLERANCE
        rows.append({'frequency_hz': float(frequency), 'box_flux_W': flux, 'source_power_W': delivered,
                     'fdfd_power_W': reference})
    return errors, {'per_frequency': rows, 'dt_s': stepper.dt, 'steps': stepper.n_step,
                    'final_over_peak_field_energy': final_energy_ratio}


def _check_radiation(results_by_name):
    """Radiated power of an on-axis dipole (m = 0) and a ring current (m = 2): flux identity and vs em_fdfd_bor."""
    errors, details = {}, {}
    for name, order, cell, component in _DIPOLE_CASES:
        case_errors, details[name] = _radiation_case(order, cell, component)
        errors.update({f'{name}: {key}': value for key, value in case_errors.items()})
    worst = max(errors, key=errors.get)
    details.update(normalised_errors=errors, worst_item=worst,
                   tolerances={'identity': _IDENTITY_TOLERANCE, 'fdfd power': _FDFD_POWER_TOLERANCE})
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0, 'details': details}


def _model_ringdown_specs():
    """{'run': em_fdtd_bor PEC cylinder (the ring-down grid) at m = 1 on the cpu backend, z + phi current pulses,
    resonance_monitors on E_z and E_phi (TE modes have no E_z, TM_mn0 no E_phi)}."""
    size_x, size_y = (count * _RING_CELL for count in _RING_CELLS)
    source_x, source_y = (cell_centre(index, _RING_CELL) for index in _RING_SOURCE_CELL)
    probe_x, probe_y = (cell_centre(index, _RING_CELL) for index in _RING_PROBE_CELL)
    settings = {'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec', 'azimuthal_order': _MODEL_ORDER}
    band = {'wavelength_min': _RING_TARGET / _RING_BAND[1], 'wavelength_max': _RING_TARGET / _RING_BAND[0]}
    sources = [FeatureSpec(f'source_{component}', 'point_source', ['source_point'],
                           {'waveform': 'gaussian', 'wavelength': _RING_TARGET, 'bandwidth': _RING_BANDWIDTH,
                            'component': component, 'current': 1e-3}) for component in ('z', 'phi')]
    monitors = [FeatureSpec(name, 'resonance_monitor', ['probe_point'], {'component': component, **band})
                for name, component in _MODEL_MONITORS.items()]
    return {'run': ModelSpec(
        title='em_fdtd_bor PEC cylinder ring-down (m = 1)',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('source_point', 'point', {'x': source_x, 'y': source_y}),
                  GeometrySpec('probe_point', 'point', {'x': probe_x, 'y': probe_y})],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=sources + monitors)],
        mesh=MeshSpec(mode='uniform', dx=_RING_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _MODEL_DURATION}),
        solver_defaults=SolverDefaultsSpec(backend='cpu'))}


def _check_model_ringdown(results_by_name):
    """resonance_monitor rows (E_z and E_phi, amplitudes per table peak) of the model run vs the leap-frog-corrected
    em_fdfd_bor resonances (its dt)."""
    found = []
    for name in _MODEL_MONITORS:
        table = results_by_name['run'].result(_PHYSICS, name)
        columns = list(table['columns'])
        rows = [list(row) for row in table['rows']]
        amplitudes = [row[columns.index('amplitude')] for row in rows]
        peak = max(amplitudes, default=1.0) or 1.0
        found.extend((float(row[columns.index('frequency')]), float(amplitude) / peak)
                     for row, amplitude in zip(rows, amplitudes))
    corrected, raw = _corrected_resonances(_MODEL_ORDER, float(table['dt']))
    f0 = C0 / _RING_TARGET
    error, matched, details, violations = _match_lines(found, corrected, raw, (_RING_BAND[0] * f0, _RING_BAND[1] * f0))
    details.update(matched=matched, violations=violations, dt_s=float(table['dt']))
    return {'measured': error, 'expected': 0.0, 'error': error if not violations else math.inf,
            'tolerance': _MODEL_TOLERANCE, 'details': details}


FDTD_BOR_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdtd_bor_cavity_ringdown',
        label='EM body-of-revolution FDTD: cavity ring-down vs FDFD',
        description=(
            'CPU check of physics/em/fdtd_bor (no model run): a vacuum PEC cylinder (radius 0.96 um, height 1.2 um, '
            f'{_RING_CELL * 1e9:g} nm cells, walls on the grid edges) rung by a gaussian z + phi current pulse at '
            f'{_RING_TARGET * 1e6:g} um for azimuthal orders {", ".join(map(str, _RING_ORDERS))}; harmonic inversion '
            f'of an E_z and an E_phi trace over {_RING_BAND[0]:g}-{_RING_BAND[1]:g} f0 vs the em_fdfd_bor '
            'resonances of the same operator (body_of_revolution.solve_bor_resonances) mapped by the leap-frog '
            'relation omega_FDTD = (2 / dt) asin(omega_FDFD dt / 2): every FDFD resonance inside the band within '
            f'{_RING_TOLERANCE:g} relative (the uncorrected mismatch (omega dt)^2 / 24 is ~5e-4), and every FDTD line '
            f'above {_RING_SPURIOUS_AMPLITUDE:g} of the strongest matches an FDFD resonance. Pins the shared Yee '
            'operator, the axis rules per |m|, the time-step bound and the leap-frog timing; m = 1 (its own axis '
            'rule) is checked the same way by em_fdtd_bor_model_ringdown.'),
        specs=lambda: {},
        check=_check_ringdown,
        max_wall_time=300.0),
    SelfTestCase(
        name='em_fdtd_bor_radiated_power',
        label='EM body-of-revolution FDTD: radiated power vs FDFD',
        description=(
            'CPU check of physics/em/fdtd_bor (no model run): an on-axis z dipole (m = 0) and an azimuthal ring current '
            f'(m = 2, r cell 8) in vacuum, {_DIPOLE_CELL * 1e9:g} nm cells, {_DIPOLE_PML}-cell CPML on r_hi and both z '
            f'faces, gaussian pulse at {_DIPOLE_WAVELENGTH * 1e6:g} um. At {", ".join(map(str, _DIPOLE_FACTORS))} f0 '
            'the DFT phasors divided by S(omega) give the closed cell-box flux (body_of_revolution.box_flux), which '
            f'must equal the source power of the same phasors (exact discrete Poynting theorem at omega_tilde, '
            f'{_IDENTITY_TOLERANCE:g}) and the em_fdfd_bor source power at omega_tilde = (2 / dt) sin(omega dt / 2) on '
            f'the same grid ({_FDFD_POWER_TOLERANCE:g}: the FDFD complex stretch vs the time-domain CPML). '
            'Normalised errors, the case passes at <= 1. Pins the source normalisation, the DFT timing (E at n dt, H '
            f'and sources at (n + 1/2) dt), the phasor convention and the CPML. Run {_DIPOLE_TIME * 1e15:g} fs (final / '
            'peak field energy in the details), not yet confirmed by a run.'),
        specs=lambda: {},
        check=_check_radiation,
        max_wall_time=300.0, heavy=True),
    SelfTestCase(
        name='em_fdtd_bor_model_ringdown',
        label='EM body-of-revolution FDTD: model ring-down (CPU backend)',
        description=(
            'Model run of em_fdtd_bor on the cpu backend: the em_fdtd_bor_cavity_ringdown cylinder at m = 1 with pec '
            'r_hi / z faces, gaussian z and phi point sources in one cell and resonance_monitors on E_z and E_phi '
            f'({_MODEL_DURATION * 1e12:g} ps; TE modes have no E_z, TM_mn0 modes no E_phi). The table frequencies vs the leap-frog-corrected em_fdfd_bor resonances at the run\'s dt (table extra dt), '
            f'matched as em_fdtd_bor_cavity_ringdown within {_MODEL_TOLERANCE:g}. Pins the model layer: create, pec '
            'walls, point source, probe and the shared resonance_monitor row.'),
        specs=_model_ringdown_specs,
        check=_check_model_ringdown,
        max_wall_time=600.0),
)
