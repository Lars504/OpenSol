"""Self-tests of the em_fdtd_bor media and ports (symmetry 2.3, fdtd_bor / fdtd_bor_ports): Drude / Lorentz poles (CPU:
a lossy dispersive PEC cavity vs em_fdfd_bor at omega_tilde with the stepped susceptibility; a cpu-backend model run vs
the direct stepper), the cylindrical r_hi CPML (CPU: a radial parallel-plate line vs a far-PML em_fdfd_bor) and the
mode ports (a cpu-backend pipe TE11 model run vs the closed form); physics imports inside the checks only."""
import math

import numpy as np

from ...core.constants import C0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import cell_centre, float_list
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           SolverDefaultsSpec, StudySpec)

_TYPE = 'em_fdtd_bor'
_PHYSICS = 'em'

# ---- dispersive PEC cavity (the em_fdtd_bor_cavity_ringdown cylinder, m = 1): a Drude layer over the bottom rows
# (eps_inf 4, Re eps ~1.5 at 1.5 um, lossy) and a Lorentz band over the outer columns (background 2.25), overlapping in
# the lower outer corner; r / z / phi currents in vacuum, Lorentz and Drude + Lorentz cells, gaussian pulse at 1.5 um.
# The DFT phasors / S(omega) after the ring-down must equal the em_fdfd_bor solution at omega_tilde with eps +
# fdtd_bor.leapfrog_susceptibility per pole (exact up to float32 rounding); the continuous chi at omega misses it by
# O((omega dt)^2) x Q (reported, not checked).
_CAVITY_CELL = 40e-9
_CAVITY_CELLS = (24, 30)
_CAVITY_ORDER = 1
_CAVITY_WAVELENGTH = 1.5e-6
_DRUDE = {'omega_p': 2.0e15, 'gamma': 1.0e14, 'eps_inf': 4.0}
_LORENTZ = {'delta_eps': 1.5, 'omega_0': 2.5e15, 'gamma': 2.0e14}
_LORENTZ_BACKGROUND = 2.25
_DRUDE_ROWS = 15                      # z cells below this carry the Drude pole
_LORENTZ_COLUMN = 10                  # r cells from this one carry the Lorentz pole
_CAVITY_SOURCES = (((5, 22), 'z'), ((16, 22), 'phi'), ((14, 6), 'r'))
_CAVITY_CURRENT = 1e-3                # A
_CAVITY_FACTORS = (0.9, 1.0, 1.1)     # analysed frequencies / f0
_DECAY = 1e-9                         # stop once the field energy fell below this share of its peak (field ~3e-5)
_CHECK_INTERVAL = 500                 # steps between energy checks
_MAX_STEPS = 80000
_FIELD_TOLERANCE = 1e-3               # relative L2 of the E phasors vs em_fdfd_bor

# ---- model run (cpu backend): the cavity with a Drude and a (non-overlapping) Lorentz material, z + phi sources, an
# E_z probe; its trace must equal the direct BORTimeStepper of the same cells (the model path: the dispersion list,
# write_dispersive_materials' eps_inf background, add_dispersive, the cell shares)
_MODEL_DURATION = 0.2e-12             # s (~2100 steps)
_MODEL_SOURCES = (((5, 22), 'z'), ((16, 22), 'phi'))
_MODEL_PROBE_CELL = (8, 18)
_MODEL_TOLERANCE = 1e-5               # max |trace difference| / max |trace|

# ---- radial parallel-plate line (pec z faces 0.2 um apart: only the z-uniform E_z / H_phi wave propagates at 1 um),
# cylindrical CPML on r_hi only, a z current element on the axis (m = 0) or a ring at m = 3 off it. Source power of the
# FDTD phasors vs em_fdfd_bor at omega_tilde on the same grid (same complex radius stretch, the convolutions in time)
# and on a far-PML grid (the reference: kR = 31 at 1 um); the m = 3 window ends at kR = 6.3, where a radius-blind
# (Cartesian) layer reflects (~1 / (2 kR) in amplitude). A layer reflection r shifts the source power by up to 2 |r| at
# any window radius (the Hankel coefficient of the returning wave is r), so the two FDFD grids alone differ by ~2 % here.
_PLATE_CELL = 25e-9
_PLATE_ROWS = 8
_PLATE_PML = 12
_PLATE_WINDOW = 40                    # r cells of the FDTD / same-grid FDFD window (1 um)
_PLATE_REFERENCE_WINDOW = 200         # r cells of the far-PML reference (5 um)
_PLATE_WAVELENGTH = 1.0e-6
_PLATE_SOURCE_ROW = 3
_PLATE_CURRENT = 1e-3                 # A
_PLATE_CASES = (('axis dipole m = 0', 0, 0), ('ring m = 3', 3, 16))   # (name, order, source r cell)
_PLATE_FACTORS = (0.9, 1.0, 1.1)
_PLATE_DECAY = 1e-8                   # energy share: field ~1e-4 of its peak left out of the DFT
_PLATE_MAX_TIME = 400e-15             # s
_PLATE_REFERENCE_TOLERANCE = 3e-2     # vs the far-PML em_fdfd_bor (PML reflection of both grids)
_PLATE_SAME_GRID_TOLERANCE = 5e-3     # vs em_fdfd_bor on the same grid

# ---- pipe ports (the em_bor_cpu_gpu pipe-port model in the time domain, cpu backend): PEC pipe a = 1 um along the
# axis, m = 1 (TE11), cpml z ends, a filling dielectric section eps 2.25 of 1 um; a gaussian TE11 mode source (1 W,
# carrier 2.25 um, bandwidth 0.3) and mode monitors below / above the section. Exact: T = 1 / (1 + F sin^2(2 pi n2 L /
# lambda)), F = (1 - eps)^2 / (4 n1^2 n2^2), n1,2 = sqrt(1 | eps - (lambda / lambda_c)^2), lambda_c = 2 pi a / x'11.
_PIPE_CELL = 25e-9
_PIPE_RADIUS = 1.0e-6
_PIPE_LENGTH = 3.0e-6
_PIPE_EPS = 2.25
_PIPE_SECTION = (1.0e-6, 2.0e-6)
_PIPE_LINES = {'source': 0.4e-6, 'incident': 0.8e-6, 'transmitted': 2.4e-6}   # + half a cell
_PIPE_BAND = (1.9e-6, 2.6e-6, 5)    # monitor wavelength start, stop [m], count
_PIPE_CARRIER = 2.25e-6
_PIPE_BANDWIDTH = 0.2               # little spectrum left at the TE11 cutoff (3.41 um), whose ringing lingers
_PIPE_DURATION = 0.6e-12            # s
_PIPE_X11 = 1.8411837813            # first zero of J1'
_PIPE_T_TOLERANCE = 0.01            # T and R vs the closed form (mesh, z PML)
_PIPE_LAUNCH_TOLERANCE = 5e-3       # launched power per cw vs 1 W at the carrier


def _cavity_cells():
    """(float64 background eps, bool Drude cells, bool Lorentz cells) of the dispersive cavity."""
    drude = np.zeros(_CAVITY_CELLS, dtype=bool)
    drude[:, :_DRUDE_ROWS] = True
    lorentz = np.zeros(_CAVITY_CELLS, dtype=bool)
    lorentz[_LORENTZ_COLUMN:, :] = True
    eps = np.ones(_CAVITY_CELLS)
    eps[lorentz] = _LORENTZ_BACKGROUND
    eps[drude] = _DRUDE['eps_inf']
    return eps, drude, lorentz


def _cavity_grid():
    """(BoundaryConfig, cell widths) of the PEC cylinder (walls on the grid edges)."""
    from .boundaries import BoundaryConfig
    config = BoundaryConfig(x_lo='pmc', x_hi='pec', y_lo='pec', y_hi='pec')
    return config, tuple(np.full(count, _CAVITY_CELL) for count in _CAVITY_CELLS)


def _run_until_decayed(stepper, pulse, decay, max_steps):
    """Step until the field energy is below decay x its peak after the pulse (or max_steps) -> final energy share."""
    peak, energy = 0.0, 0.0
    while stepper.n_step < max_steps:
        for _ in range(_CHECK_INTERVAL):
            stepper.step()
        energy = stepper.field_energy()
        peak = max(peak, energy)
        if stepper.time > pulse.duration and energy <= decay * peak:
            break
    return energy / peak if peak > 0.0 else 0.0


def _relative_e_error(fdtd, fdfd):
    """Relative L2 difference of the E phasors (er, ephi, ez yee dicts)."""
    names = ('er', 'ephi', 'ez')
    difference = sum(float(np.sum(np.abs(fdtd[name] - fdfd[name]) ** 2)) for name in names)
    reference = sum(float(np.sum(np.abs(fdfd[name]) ** 2)) for name in names)
    return math.sqrt(difference / reference) if reference > 0.0 else math.inf


def _check_dispersive_cavity(results_by_name):
    """DFT phasors of the dispersive cavity vs em_fdfd_bor at omega_tilde with the stepped susceptibility."""
    from ...core.sources import GaussianPulse
    from ...solvers.sparse_direct import factorize_host
    from .body_of_revolution import BORMaterials, assemble_bor, point_current_rhs, yee_fields
    from .fdtd_bor import BORTimeStepper, leapfrog_susceptibility, phasor_yee, source_spectrum
    from .materials import DrudePole, LorentzPole
    config, widths = _cavity_grid()
    eps, drude_cells, lorentz_cells = _cavity_cells()
    ones = np.ones(_CAVITY_CELLS, dtype=np.complex128)
    no_pec = np.zeros(_CAVITY_CELLS, dtype=bool)
    drude, lorentz = DrudePole(**_DRUDE), LorentzPole(**_LORENTZ)
    poles = ((drude, drude_cells), (lorentz, lorentz_cells))
    stepper = BORTimeStepper(widths, config, _CAVITY_CELL, BORMaterials(eps=eps.astype(np.complex128), mu=ones,
                                                                        pec=no_pec), _CAVITY_ORDER,
                             poles=[(pole, cells.astype(np.float64)) for pole, cells in poles])
    f0 = C0 / _CAVITY_WAVELENGTH
    pulse = GaussianPulse(f0, 0.5)
    for (ix, iz), component in _CAVITY_SOURCES:
        stepper.add_current(ix, iz, component, _CAVITY_CURRENT, pulse)
    frequencies = f0 * np.array(_CAVITY_FACTORS)
    spectra = stepper.add_spectra(frequencies)
    remaining = _run_until_decayed(stepper, pulse, _DECAY, _MAX_STEPS)
    spectrum = source_spectrum(pulse, stepper.dt, frequencies, stepper.n_step)

    def fdfd_fields(omega, susceptibility):
        cell_eps = eps.astype(np.complex128)
        for pole, cells in poles:
            cell_eps = cell_eps + susceptibility(pole) * cells
        system = assemble_bor(widths, config, _CAVITY_CELL, BORMaterials(eps=cell_eps, mu=ones, pec=no_pec), omega,
                              _CAVITY_ORDER)
        rhs = sum(point_current_rhs(system, ix, iz, component,
                                    1j * _CAVITY_CURRENT if component == 'phi' else _CAVITY_CURRENT)
                  for (ix, iz), component in _CAVITY_SOURCES)
        return yee_fields(system, factorize_host(system.operator).solve_host(rhs))

    errors, rows = {}, []
    for index, frequency in enumerate(frequencies):
        omega = 2.0 * math.pi * frequency
        omega_tilde = 2.0 / stepper.dt * math.sin(0.5 * omega * stepper.dt)
        fdtd = phasor_yee(*spectra.full_phasors(index, 1.0 / spectrum[index]), stepper.shape)
        stepped = fdfd_fields(omega_tilde, lambda pole: leapfrog_susceptibility(pole, omega, stepper.dt))
        continuous = fdfd_fields(omega, lambda pole: pole.susceptibility(omega))
        label = f'{frequency / f0:g} f0'
        errors[label] = _relative_e_error(fdtd, stepped)
        rows.append({'frequency_hz': float(frequency), 'relative_error': errors[label],
                     'continuous_chi_relative_error': _relative_e_error(fdtd, continuous),
                     'drude_chi_stepped': str(leapfrog_susceptibility(drude, omega, stepper.dt)),
                     'drude_chi_continuous': str(drude.susceptibility(omega))})
    worst = max(errors, key=errors.get)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': _FIELD_TOLERANCE,
            'details': {'per_frequency': rows, 'worst_item': worst, 'dt_s': stepper.dt, 'steps': stepper.n_step,
                        'final_energy_share': remaining}}


def _dispersive_model_specs():
    """{'run': em_fdtd_bor dispersive cavity (Drude bottom rows, Lorentz outer upper band) on the cpu backend, z + phi
    point sources, an E_z field_probe}."""
    size_x, size_y = (count * _CAVITY_CELL for count in _CAVITY_CELLS)
    rows_top, column_start = _DRUDE_ROWS * _CAVITY_CELL, _LORENTZ_COLUMN * _CAVITY_CELL
    geometry = [GeometrySpec('drude_layer', 'rectangle', {'x0': 0.0, 'y0': 0.0, 'x1': size_x, 'y1': rows_top}),
                GeometrySpec('lorentz_band', 'rectangle', {'x0': column_start, 'y0': rows_top, 'x1': size_x,
                                                           'y1': size_y}),
                GeometrySpec('probe_point', 'point', {'x': cell_centre(_MODEL_PROBE_CELL[0], _CAVITY_CELL),
                                                      'y': cell_centre(_MODEL_PROBE_CELL[1], _CAVITY_CELL)})]
    features = []
    for number, ((ix, iz), component) in enumerate(_MODEL_SOURCES):
        name = f'source_{number}'
        geometry.append(GeometrySpec(f'{name}_point', 'point', {'x': cell_centre(ix, _CAVITY_CELL),
                                                                'y': cell_centre(iz, _CAVITY_CELL)}))
        features.append(FeatureSpec(name, 'point_source', [f'{name}_point'],
                                    {'waveform': 'gaussian', 'wavelength': _CAVITY_WAVELENGTH, 'bandwidth': 0.5,
                                     'component': component, 'current': _CAVITY_CURRENT}))
    features.append(FeatureSpec('probe', 'field_probe', ['probe_point'], {'component': 'ez'}))
    settings = {'bc_x_hi': 'pec', 'bc_y_lo': 'pec', 'bc_y_hi': 'pec', 'azimuthal_order': _CAVITY_ORDER}
    return {'run': ModelSpec(
        title='em_fdtd_bor dispersive cavity (Drude + Lorentz, m = 1)',
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=geometry,
        materials=[MaterialSpec('drude_metal', properties={}, selection=['drude_layer'],
                                dispersion=[{'model': 'drude', **_DRUDE}]),
                   MaterialSpec('lorentz_dielectric', properties={'eps_r': _LORENTZ_BACKGROUND},
                                selection=['lorentz_band'], dispersion=[{'model': 'lorentz', **_LORENTZ}])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CAVITY_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _MODEL_DURATION}),
        solver_defaults=SolverDefaultsSpec(backend='cpu'))}


def _check_dispersive_model(results_by_name):
    """Model probe trace vs the direct BORTimeStepper of the same cells, poles and sources."""
    from ...core.sources import GaussianPulse
    from .body_of_revolution import BORMaterials
    from .fdtd_bor import BORTimeStepper
    from .materials import DrudePole, LorentzPole
    trace = results_by_name['run'].result(_PHYSICS, 'probe')
    values = np.asarray(trace['values'], dtype=np.float64)
    config, widths = _cavity_grid()
    _, drude_cells, lorentz_cells = _cavity_cells()
    lorentz_cells &= ~drude_cells   # the model's band starts above the Drude layer
    eps = np.ones(_CAVITY_CELLS)
    eps[lorentz_cells] = _LORENTZ_BACKGROUND
    eps[drude_cells] = _DRUDE['eps_inf']
    ones = np.ones(_CAVITY_CELLS, dtype=np.complex128)
    stepper = BORTimeStepper(widths, config, _CAVITY_CELL,
                             BORMaterials(eps=eps.astype(np.complex128), mu=ones,
                                          pec=np.zeros(_CAVITY_CELLS, dtype=bool)), _CAVITY_ORDER,
                             poles=[(DrudePole(**_DRUDE), drude_cells.astype(np.float64)),
                                    (LorentzPole(**_LORENTZ), lorentz_cells.astype(np.float64))])
    pulse = GaussianPulse(C0 / _CAVITY_WAVELENGTH, 0.5)
    for (ix, iz), component in _MODEL_SOURCES:
        stepper.add_current(ix, iz, component, _CAVITY_CURRENT, pulse)
    ix, iz = _MODEL_PROBE_CELL
    direct = np.empty(values.size)
    for step in range(values.size):
        stepper.step()
        direct[step] = stepper.fields.ez[ix, iz + 1]
    scale = float(np.max(np.abs(direct))) if direct.size else 0.0
    error = float(np.max(np.abs(values - direct))) / scale if scale > 0.0 else math.inf
    return {'measured': error, 'expected': 0.0, 'error': error, 'tolerance': _MODEL_TOLERANCE,
            'details': {'steps': int(values.size), 'trace_max_v_per_m': scale, 'dt_s': stepper.dt}}


def _plate_grid(window_cells):
    """(BoundaryConfig, cell widths, window, vacuum BORMaterials) of the parallel-plate line with window_cells r cells
    and the r_hi cpml."""
    from .body_of_revolution import BORMaterials
    from .boundaries import BoundaryConfig
    config = BoundaryConfig(x_lo='pmc', x_hi='cpml', y_lo='pec', y_hi='pec')
    config.cpml_npml = _PLATE_PML
    shape = (window_cells + _PLATE_PML, _PLATE_ROWS)
    widths = tuple(np.full(count, _PLATE_CELL) for count in shape)
    materials = BORMaterials(eps=np.ones(shape, dtype=np.complex128), mu=np.ones(shape, dtype=np.complex128),
                             pec=np.zeros(shape, dtype=bool))
    return config, widths, (slice(0, window_cells), slice(0, _PLATE_ROWS)), materials


def _fdfd_power(window_cells, order, source_cell, omega):
    """em_fdfd_bor source power [W] of the plate source at omega on a grid of window_cells r cells."""
    from ...solvers.sparse_direct import factorize_host
    from .body_of_revolution import assemble_bor, point_current_rhs, source_power
    config, widths, window, materials = _plate_grid(window_cells)
    system = assemble_bor(widths, config, _PLATE_CELL, materials, omega, order, window=window)
    rhs = point_current_rhs(system, source_cell, _PLATE_SOURCE_ROW, 'z', _PLATE_CURRENT)
    return source_power(system, factorize_host(system.operator).solve_host(rhs), rhs)


def _plate_case(order, source_cell):
    """(normalised errors, details) of one plate source: FDTD source power vs em_fdfd_bor at omega_tilde on the same
    and on the far-PML grid."""
    from ...core.sources import GaussianPulse
    from .fdtd_bor import BORTimeStepper, source_spectrum
    config, widths, window, materials = _plate_grid(_PLATE_WINDOW)
    stepper = BORTimeStepper(widths, config, _PLATE_CELL, materials, order, window=window)
    f0 = C0 / _PLATE_WAVELENGTH
    pulse = GaussianPulse(f0, 0.5)
    injection = stepper.add_current(source_cell, _PLATE_SOURCE_ROW, 'z', _PLATE_CURRENT, pulse)
    frequencies = f0 * np.array(_PLATE_FACTORS)
    spectra = stepper.add_spectra(frequencies, e_index=injection.indices, a_index=np.zeros(1, dtype=np.int64))
    remaining = _run_until_decayed(stepper, pulse, _PLATE_DECAY, int(math.ceil(_PLATE_MAX_TIME / stepper.dt)))
    spectrum = source_spectrum(pulse, stepper.dt, frequencies, stepper.n_step)
    errors, rows = {}, []
    for index, frequency in enumerate(frequencies):
        e_values, _ = spectra.full_phasors(index, 1.0 / spectrum[index])
        delivered = math.pi * float(np.real(np.vdot(e_values[injection.indices], injection.charges)))
        omega_tilde = 2.0 / stepper.dt * math.sin(math.pi * frequency * stepper.dt)
        same_grid = _fdfd_power(_PLATE_WINDOW, order, source_cell, omega_tilde)
        reference = _fdfd_power(_PLATE_REFERENCE_WINDOW, order, source_cell, omega_tilde)
        label = f'{frequency / f0:g} f0'
        errors[f'{label}: reference'] = abs(delivered / reference - 1.0) / _PLATE_REFERENCE_TOLERANCE
        errors[f'{label}: same grid'] = abs(delivered / same_grid - 1.0) / _PLATE_SAME_GRID_TOLERANCE
        rows.append({'frequency_hz': float(frequency), 'fdtd_power_W': delivered, 'fdfd_same_grid_W': same_grid,
                     'fdfd_reference_W': reference, 'fdfd_same_grid_vs_reference': same_grid / reference - 1.0})
    return errors, {'per_frequency': rows, 'dt_s': stepper.dt, 'steps': stepper.n_step,
                    'final_energy_share': remaining}


def _check_cylindrical_pml(results_by_name):
    """Source power of the plate line sources with the cylindrical CPML vs em_fdfd_bor (same grid, far PML)."""
    errors, details = {}, {}
    for name, order, source_cell in _PLATE_CASES:
        case_errors, details[name] = _plate_case(order, source_cell)
        errors.update({f'{name}: {key}': value for key, value in case_errors.items()})
    worst = max(errors, key=errors.get)
    details.update(normalised_errors=errors, worst_item=worst,
                   tolerances={'reference': _PLATE_REFERENCE_TOLERANCE, 'same grid': _PLATE_SAME_GRID_TOLERANCE},
                   factors=float_list(_PLATE_FACTORS))
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0, 'details': details}


def _pipe_ports_specs():
    """{'run': em_fdtd_bor pipe TE11 ports through the dielectric section, time_dependent on the cpu backend}."""
    half = _PIPE_CELL / 2.0
    settings = {'azimuthal_order': 1, 'bc_x_hi': 'pec', 'bc_y_lo': 'cpml', 'bc_y_hi': 'cpml', 'pml_cells': 20}
    geometry = [GeometrySpec('section', 'rectangle', {'x0': 0.0, 'y0': _PIPE_SECTION[0], 'x1': _PIPE_RADIUS,
                                                      'y1': _PIPE_SECTION[1]})]
    geometry += [GeometrySpec(f'{name}_line', 'line', {'x0': 0.0, 'y0': z + half, 'x1': _PIPE_RADIUS, 'y1': z + half})
                 for name, z in _PIPE_LINES.items()]
    band = {'wavelength_start': _PIPE_BAND[0], 'wavelength_stop': _PIPE_BAND[1], 'n_wavelengths': _PIPE_BAND[2],
            'mode_number': 1}
    features = [FeatureSpec('launch', 'mode_source', selection=['source_line'],
                            params={'waveform': 'gaussian', 'wavelength': _PIPE_CARRIER, 'bandwidth': _PIPE_BANDWIDTH,
                                    'power': 1.0, 'mode_number': 1, 'direction': 'positive', 'phase': 0.0}),
                FeatureSpec('incident', 'mode_monitor', selection=['incident_line'], params=dict(band)),
                FeatureSpec('transmitted', 'mode_monitor', selection=['transmitted_line'], params=dict(band))]
    return {'run': ModelSpec(
        title='em_fdtd_bor pipe TE11 ports through a dielectric section',
        domain=DomainSpec(size_x=_PIPE_RADIUS, size_y=_PIPE_LENGTH),
        geometry=geometry,
        materials=[MaterialSpec('section_material', properties={'eps_r': _PIPE_EPS}, selection=['section'])],
        physics=[PhysicsSpec(_PHYSICS, _TYPE, settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_PIPE_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _PIPE_DURATION}),
        solver_defaults=SolverDefaultsSpec(backend='cpu'))}


def _pipe_transmission(wavelength):
    """Closed-form TE11 power transmission through the filling section at `wavelength` [m] (array)."""
    cutoff = 2.0 * math.pi * _PIPE_RADIUS / _PIPE_X11
    ratio = (np.asarray(wavelength, dtype=np.float64) / cutoff) ** 2
    n_empty, n_section = np.sqrt(1.0 - ratio), np.sqrt(_PIPE_EPS - ratio)
    contrast = (1.0 - _PIPE_EPS) ** 2 / (4.0 * n_empty ** 2 * n_section ** 2)
    length = _PIPE_SECTION[1] - _PIPE_SECTION[0]
    return 1.0 / (1.0 + contrast * np.sin(2.0 * math.pi * n_section * length / wavelength) ** 2)


def _check_pipe_ports(results_by_name):
    """T = transmitted / launched forward power and R = incident backward / forward vs the closed form per monitor
    wavelength; the launched power per cw at the carrier vs 1 W."""
    results = results_by_name['run']
    transmitted = results.result(_PHYSICS, 'transmitted')
    incident = results.result(_PHYSICS, 'incident')
    wavelength = np.asarray(incident['x'], dtype=np.float64)
    forward = np.asarray(incident['series']['power_positive'], dtype=np.float64)
    measured_t = np.asarray(transmitted['series']['power_positive'], dtype=np.float64) / forward
    measured_r = np.asarray(incident['series']['power_negative'], dtype=np.float64) / forward
    expected = _pipe_transmission(wavelength)
    carrier = int(np.argmin(np.abs(wavelength - _PIPE_CARRIER)))
    launched = float(np.asarray(incident['series']['power_per_cw_positive'])[carrier])
    errors = {'T': float(np.max(np.abs(measured_t - expected))) / _PIPE_T_TOLERANCE,
              'R': float(np.max(np.abs(measured_r - (1.0 - expected)))) / (2.0 * _PIPE_T_TOLERANCE),
              'launch at the carrier': abs(launched - 1.0) / _PIPE_LAUNCH_TOLERANCE}
    worst = max(errors, key=errors.get)
    return {'measured': errors[worst], 'expected': 0.0, 'error': errors[worst], 'tolerance': 1.0,
            'details': {'normalised_errors': errors, 'worst_item': worst, 'wavelength_m': float_list(wavelength),
                        'transmission': float_list(measured_t), 'reflection': float_list(measured_r),
                        'expected_transmission': float_list(expected), 'launched_power_W': launched,
                        'fraction_positive_transmitted': float_list(transmitted['series'].get('fraction_positive',
                                                                                              []))}}


FDTD_BOR_MEDIA_SELFTEST_CASES = (
    SelfTestCase(
        name='em_fdtd_bor_dispersive_cavity',
        label='EM body-of-revolution FDTD: Drude / Lorentz cavity vs FDFD',
        description=(
            'CPU check of physics/em/fdtd_bor poles (no model run): the em_fdtd_bor_cavity_ringdown PEC cylinder at m = '
            f'{_CAVITY_ORDER} with a Drude layer over the bottom {_DRUDE_ROWS} rows (omega_p {_DRUDE["omega_p"]:g} '
            f'rad/s, gamma {_DRUDE["gamma"]:g} 1/s, eps_inf {_DRUDE["eps_inf"]:g}) and a Lorentz band over the r cells '
            f'from {_LORENTZ_COLUMN} on (delta_eps {_LORENTZ["delta_eps"]:g}, omega_0 {_LORENTZ["omega_0"]:g} rad/s, '
            f'gamma {_LORENTZ["gamma"]:g} 1/s, background {_LORENTZ_BACKGROUND:g}), overlapping in one corner; r, z and '
            f'phi currents with a gaussian pulse at {_CAVITY_WAVELENGTH * 1e6:g} um, stepped until the field energy '
            f'fell below {_DECAY:g} of its peak. At {", ".join(map(str, _CAVITY_FACTORS))} f0 the DFT E phasors / '
            'S(omega) vs em_fdfd_bor at omega_tilde = (2 / dt) sin(omega dt / 2) with eps + '
            'fdtd_bor.leapfrog_susceptibility per pole (the stepped chi: the continuous one at omega_tilde with gamma '
            f'cos(omega dt / 2)), relative L2 within {_FIELD_TOLERANCE:g}; the error with the continuous chi at omega '
            'is reported. Pins the velocity-form pole update, its timing (E^n drive, correction after the E update), '
            'the point shares of E_r / E_z and the gain-weighted correction. The truncated DFT tail (field ~3e-5 of '
            f'its peak at {_DECAY:g} of the energy) is far below the tolerance; {_DECAY:g} not yet confirmed by a '
            'run.'),
        specs=lambda: {},
        check=_check_dispersive_cavity,
        max_wall_time=300.0),
    SelfTestCase(
        name='em_fdtd_bor_dispersive_model',
        label='EM body-of-revolution FDTD: dispersive model run (CPU backend)',
        description=(
            'Model run of em_fdtd_bor on the cpu backend: the em_fdtd_bor_dispersive_cavity cylinder with a Drude '
            'material (dispersion list, eps_inf background) on the bottom rows and a Lorentz material (eps_r '
            f'{_LORENTZ_BACKGROUND:g} + pole) above them on the outer columns, z and phi point sources, an E_z '
            f'field_probe ({_MODEL_DURATION * 1e12:g} ps). The trace vs the direct fdtd_bor.BORTimeStepper of the same '
            f'cells, poles and sources: max difference within {_MODEL_TOLERANCE:g} of the trace maximum. Pins the model '
            'path: the dispersion list, write_dispersive_materials, add_dispersive and the cell shares.'),
        specs=_dispersive_model_specs,
        check=_check_dispersive_model,
        max_wall_time=300.0),
    SelfTestCase(
        name='em_fdtd_bor_cylindrical_pml',
        label='EM body-of-revolution FDTD: cylindrical CPML vs FDFD',
        description=(
            'CPU check of the fdtd_bor r_hi CPML (no model run): a radial parallel-plate line (pec z faces '
            f'{_PLATE_ROWS * _PLATE_CELL * 1e6:g} um apart, {_PLATE_CELL * 1e9:g} nm cells, {_PLATE_PML}-cell CPML on '
            f'r_hi only, {_PLATE_WINDOW * _PLATE_CELL * 1e6:g} um window) driven by a z current element on the axis at '
            f'm = 0 and a ring at m = 3 ({cell_centre(16, _PLATE_CELL) * 1e6:g} um off it), gaussian pulse at '
            f'{_PLATE_WAVELENGTH * 1e6:g} um. At {", ".join(map(str, _PLATE_FACTORS))} f0 the source power of the DFT '
            'phasors / S(omega) vs em_fdfd_bor at omega_tilde on the same grid (the same complex radius stretch: '
            f'{_PLATE_SAME_GRID_TOLERANCE:g}) and on a {_PLATE_REFERENCE_WINDOW * _PLATE_CELL * 1e6:g} um far-PML grid '
            f'({_PLATE_REFERENCE_TOLERANCE:g}). Stepped until the field energy fell below {_PLATE_DECAY:g} of its '
            'peak (field ~1e-4 left out of the DFT; not yet confirmed by a run). Normalised errors, the case passes '
            'at <= 1. Pins the cylindrical layer: the kappa-stretched metric, the 1 / w, 1 / L and 1 / r poles and '
            'the radius integral.'),
        specs=lambda: {},
        check=_check_cylindrical_pml,
        max_wall_time=300.0),
    SelfTestCase(
        name='em_fdtd_bor_pipe_ports',
        label='EM body-of-revolution FDTD: pipe TE11 ports (CPU backend)',
        description=(
            'Model run of em_fdtd_bor on the cpu backend (time_dependent, '
            f'{_PIPE_DURATION * 1e12:g} ps): a PEC pipe of radius {_PIPE_RADIUS * 1e6:g} um at m = 1 with a filling '
            f'dielectric section (eps {_PIPE_EPS:g}, {_PIPE_SECTION[0] * 1e6:g}-{_PIPE_SECTION[1] * 1e6:g} um), cpml z '
            f'ends, a 1 W gaussian TE11 mode_source (carrier {_PIPE_CARRIER * 1e6:g} um, bandwidth {_PIPE_BANDWIDTH:g}) '
            f'and mode monitors below and above the section at {_PIPE_BAND[2]} wavelengths of '
            f'{_PIPE_BAND[0] * 1e6:g}-{_PIPE_BAND[1] * 1e6:g} um. T = transmitted / launched forward power and R = '
            'backward / forward power at the incident monitor vs the closed form (single-mode TE11 Fabry-Perot of the '
            f'section) within {_PIPE_T_TOLERANCE:g} / {2.0 * _PIPE_T_TOLERANCE:g}; the launched power per cw at the '
            f'carrier vs 1 W within {_PIPE_LAUNCH_TOLERANCE:g} (exact launch there). Pins fdtd_bor_ports: the stepped '
            'mode solve, the time-domain TFSF injections, the DFT reciprocity decomposition and the model features.'),
        specs=_pipe_ports_specs,
        check=_check_pipe_ports,
        max_wall_time=900.0),
)
