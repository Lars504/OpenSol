"""Coupling self-tests (E12) on one y-periodic glass strip: stationary em_heat_joule + heat_em_thermo_optic chains
(E_z; H_z with thermal expansion and a field_transfer twin), the currents_heat_joule chain, and an FDTD H_z +
transient heat Joule energy identity, against closed forms; no physics imports, specs built fresh."""
import math

import numpy as np

from ..core.constants import C0
from ..model.schema import SelfTestCase
from ..model.selftest_helpers import PERIODIC_Y, cell_centre, float_list, single_value, vertical_line, with_exact_cg
from ..model.spec import (CouplingSpec, DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec,
                          MeshSpec, ModelSpec, PhysicsSpec, StudySpec)

_EM = 'em'
_HEAT = 'heat'
_CURRENTS = 'currents'
_REFERENCE = 'reference'
_HEATED = 'heated'
_TRANSFER = 'transfer'
_EM_ONLY = 'em_only'
_COUPLED = 'coupled'
_SOURCE = 'source'
_FLUX_UP = 'flux_up'
_FLUX_DOWN = 'flux_down'
_MODE_UP = 'mode_up'
_MODE_DOWN = 'mode_down'
_SLAB = 'slab'
_RESISTOR = 'resistor'
_TERMINAL_A = 'terminal_a'
_TERMINAL_B = 'terminal_b'

# Strip shared by both cases: y-periodic 8.0 x 0.2 um, CPML along x, uniform 20 nm mesh (400 x 10 window cells, 430 x
# 10 FDFD unknowns), vacuum wavelength 1.55 um (52 cells per wavelength in the glass); window cell k spans [20 k,
# 20 (k + 1)] nm, vertical lines sit at cell centres (k + 0.5) 20 nm and span the whole period. Glass n = 1.5 (eps_r
# 2.25), k = 0.2 W/(m K) everywhere incl. the PML. Heat: T0 walls on x_lo / x_hi, insulated y faces (1D in x).
# em_heat_thermo_optic_strip: the slab x in [2, 6) um (cells 100..299) adds sigma_e = 2.0e3 S/m ~ 2 n / (eta0 L): Im
# eps = sigma / (omega eps0) = 0.186, Im eps / eps_r = 0.083 (weak loss), field attenuation alpha = sigma eta0 / (2 n)
# = 2.51e5 /m, alpha L = 1.00, so the slab absorbs 1 - exp(-2 alpha L) = 86.5 % of the 1 W/m mode. Source cell 50
# (interface on the hi face of cell 49), upstream flux line cell 65, mode monitors cells 75 / 325 (5.0 um apart),
# downstream flux line cell 335: no measuring face on the source interface, none inside the slab. The 1 W/m plane wave
# over the 0.2 um period has E0^2 = 2 (P / w) eta0 / n = 2.5e9 V^2/m^2, so q0 = 1/2 sigma E0^2 = 2.5e12 W/m^3 at the
# slab front; the Dirichlet solution of k theta'' = -q0 exp(-2 alpha (x - 2 um)) peaks at ~33 K near x = 3.4 um (25 K
# at the slab front, 18 K at its back) and integrates to ~1.33e-4 K m between the mode monitors, so dndT = 1e-3 /K
# gives dphi ~ -k0 dndT int(theta dx) = -0.54 rad with dn / n at most 2.2 %. Stationary tolerance 2e-6: the fixed
# point monitors the float32 rise T - T0 (one ulp at 33 K = 3.8e-6 K = 1.2e-7 of the rise), well above its floor.
# currents_heat_thermo_optic_strip: the same lines without the flux monitors; the slab region x in [2, 6) um is a
# resistor with sigma_dc = 15 S/m (no sigma_e: the glass stays lossless, the EM mode is the plane wave of uniform
# glass) held by the terminals x in [2.0, 2.1) um (cells 100..104, 1 V) and [5.9, 6.0) um (cells 295..299, 0 V), so
# the free chain runs between the inner held faces x = 2.1 and 5.9 um: L = 3.8 um, I = sigma w V / L = 15 * 0.2e-6 /
# 3.8e-6 = 0.789474 A/m, P = V I = 0.789474 W/m, q0 = sigma (V / L)^2 = 1.03878e12 W/m^3 in cells 105..294 (0
# elsewhere, also in the held cells). Continuum: theta = 9.9 K/um x outside the source, peak q0 * 5.795e-12 m^2 / k
# ~ 30.1 K at x = 4 um (~14.9 K at the upstream monitor, 14.7 K at the downstream one), int(theta dx) ~ 1.24e-4 K m
# between the monitors, dphi ~ -0.50 rad with dndT = 1e-3 /K. The chain is feed-forward (no sigma_dc(T), no EM loss).
_WAVELENGTH = 1.55e-6                     # m, vacuum
_CELL = 20e-9                             # m, uniform mesh
_SIZE = (8e-6, 0.2e-6)                    # m
_N_CELLS = (round(_SIZE[0] / _CELL), round(_SIZE[1] / _CELL))   # (400, 10) window cells
_GLASS_EPS_R = 2.25
_GLASS_INDEX = math.sqrt(_GLASS_EPS_R)    # 1.5 = n0 recorded by the heat physics
_CONDUCTIVITY = 0.2                       # W/(m K)
_SLAB_X_RANGE = (2e-6, 6e-6)              # m, edges on cell faces (the absorbing slab and the resistor)
_SLAB_SIGMA = 2.0e3                       # S/m, optical (sigma_e)
_DNDT = 1e-3                              # 1/K, heated run
_T0 = 293.15                              # K, walls and reference temperature
_LINE_CELLS = {_SOURCE: 50, _FLUX_UP: 65, _MODE_UP: 75, _MODE_DOWN: 325, _FLUX_DOWN: 335}   # window cell along x
_CURRENTS_LINE_CELLS = {name: _LINE_CELLS[name] for name in (_SOURCE, _MODE_UP, _MODE_DOWN)}
_MODE_SOURCE = {'mode_number': 1, 'direction': 'positive', 'power': 1.0, 'phase': 0.0}
_STUDY = {'tolerance': 2e-6, 'max_iterations': 30}
_K0 = 2.0 * math.pi / _WAVELENGTH         # 1/m
_BALANCE_TOLERANCE = 0.01                 # relative to P_up
_SOURCE_POWER_TOLERANCE = 0.01            # relative, P_up vs the mode source power
_PROFILE_TOLERANCE = 1e-4                 # relative to the peak rise: an exact discrete identity (see the description)
_PHASE_TOLERANCE = 0.02                   # relative to the predicted shift
_EPS_TOLERANCE = 1e-3                     # relative to eps_r, a failed check above it
_MAX_TICKS = {_REFERENCE: 3, _HEATED: 10}
_PROFILE_SAMPLES = 40

# em_hz_heat_thermo_expansion_strip: the em_heat_thermo_optic_strip layout with em_fdfd_hz (E_y tangential, so eps_yy
# drives the same discrete k), alpha = 1e-4 /K in the heated run (n (1 + alpha theta): ~5e-3 of the ~0.038 index rise,
# 13 % of the shift, so a missing or doubled alpha fails the 2 % phase check) and a 'transfer' run writing the same
# eps through two field_transfer couplings instead of heat_em_thermo_optic.
_HZ_ALPHA = 1e-4                          # 1/K, heated run
_HZ_EPS_FIELDS = (f'{_EM}.eps_xx', f'{_EM}.eps_yy')
_HZ_MAX_TICKS = {_REFERENCE: 3, _HEATED: 10, _TRANSFER: 10}
_TRANSFER_TOLERANCE = 1e-3                # relative to the heated dphi: the same eps up to float32 rounding

# fdtd_heat_joule_energy: em_fdtd_hz strip (absorbing slab, no monitors) with a gaussian mode pulse (bandwidth 0.3: tau
# = 5.5 fs, over by ~50 fs, through the slab by ~85 fs) and transient heat (insulated box, rho cp = 1e-3 J/(m^3 K) so
# the ~6e-15 J/m absorbed raise it ~4 K) stepped every 200 EM steps (9.3 fs) over 140 fs: the last two windows (never
# stepped / flushed, from ~121 fs) carry only CPML ripples (light leaving the slab reaches the CPML within ~10 fs), so
# sum rho cp theta dA must equal the absorbed energy of an em_only run. The heated run checks only eps against its own
# final T, which holds at any time, so it stops at 100 fs (most of the energy absorbed: theta ~3.5-4 K).
_FDTD_MODE_SOURCE = {'mode_number': 1, 'direction': 'positive', 'waveform': 'gaussian', 'wavelength': _WAVELENGTH,
                     'bandwidth': 0.3, 'power': 1.0}
_FDTD_DURATION = 140e-15                  # s, em_only and coupled runs
_FDTD_HEATED_DURATION = 100e-15           # s, heated run (eps check only)
_FDTD_HEAT_EVERY = 200                    # EM steps per heat step (9.3e-15 s at courant 0.99)
_FDTD_HEAT_TIME_STEP = 1e-13              # s, > every * dt_EM
_FDTD_DENSITY = 1.0                       # kg/m^3 (unphysical: a visible rise from a femtosecond pulse)
_FDTD_SPECIFIC_HEAT = 1e-3                # J/(kg K)
_FDTD_DNDT = 1e-3                         # 1/K, heated run
_FDTD_ALPHA = 1e-3                        # 1/K, heated run: 2 alpha theta ~ 7-8e-3 > _EPS_TOLERANCE
_ENERGY_TOLERANCE = 1e-3                  # relative to the absorbed energy
_TAIL_LIMIT = 1e-4                        # flushed (never stepped) energy bound relative to the absorbed energy

_RESISTOR_SIGMA = 15.0                    # S/m, DC (sigma_dc)
_TERMINAL_VOLTAGE = 1.0                   # V on terminal_a (terminal_b at 0 V)
_TERMINAL_CELLS = 5                       # cells along x of each terminal: 100..104 and 295..299
_TERMINAL_LENGTH = _TERMINAL_CELLS * _CELL
_CURRENT_TOLERANCE = 1e-6                 # relative, terminal currents and the Joule power identity (exact CG node)
_CURRENTS_MAX_TICKS = {_REFERENCE: 1, _HEATED: 1}   # feed-forward: every physics is a sink, one pass
_RESISTOR_LENGTH = _SLAB_X_RANGE[1] - _SLAB_X_RANGE[0] - 2 * _TERMINAL_LENGTH          # 3.8 um between the held faces
_RESISTOR_CURRENT = _RESISTOR_SIGMA * _SIZE[1] * _TERMINAL_VOLTAGE / _RESISTOR_LENGTH   # 0.789474 A/m
_RESISTOR_JOULE = _RESISTOR_SIGMA * (_TERMINAL_VOLTAGE / _RESISTOR_LENGTH) ** 2          # 1.03878e12 W/m^3


# ---- specs ----
def _cell_line(name, cell_index):
    """Line geometry at the centre of window cell `cell_index` spanning the whole period (y 0..size_y)."""
    return vertical_line(name, cell_centre(cell_index, _CELL), 0.0, _SIZE[1])


def _line_name(feature):
    return f'{feature}_line'


def _full_period_rectangle(name, x0, x1):
    """Rectangle between x0 and x1 (edges on cell faces) over the whole period."""
    return GeometrySpec(name, 'rectangle', {'x0': x0, 'y0': 0.0, 'x1': x1, 'y1': _SIZE[1]})


def _line_geometry(line_cells):
    return [_cell_line(_line_name(feature), cell) for feature, cell in line_cells.items()]


def _em_feature(feature):
    """mode_source for _SOURCE, flux_monitor for the flux lines, mode_monitor (mode 1) for the mode lines."""
    line = [_line_name(feature)]
    if feature == _SOURCE:
        return FeatureSpec(feature, 'mode_source', line, dict(_MODE_SOURCE))
    if feature in (_FLUX_UP, _FLUX_DOWN):
        return FeatureSpec(feature, 'flux_monitor', line)
    return FeatureSpec(feature, 'mode_monitor', line, {'mode_number': 1})


def _em_physics(line_cells, em_type='em_fdfd_ez'):
    """FDFD `em_type` on the y-periodic strip at _WAVELENGTH with one feature per line, in line_cells order."""
    return PhysicsSpec(_EM, em_type, settings={**PERIODIC_Y, 'wavelength': _WAVELENGTH},
                       features=[_em_feature(feature) for feature in line_cells])


def _heat_physics():
    """Heat with T0 walls on x_lo / x_hi (the y faces stay insulated)."""
    return PhysicsSpec(_HEAT, 'heat', settings={'T0': _T0}, features=[
        FeatureSpec('walls', 'temperature', ['x_lo', 'x_hi'], {'temperature': _T0})])


def _glass_properties(dndT, alpha=0.0):
    properties = {'eps_r': _GLASS_EPS_R, 'k': _CONDUCTIVITY, 'dndT': dndT}
    if alpha:
        properties['alpha'] = alpha
    return properties


def _thermo_optic_expression(dndT, alpha):
    """field_transfer expression of value = T giving eps = ((n0 + dndT dT)(1 + alpha dT))^2 of the glass."""
    rise = f'(value - {_T0!r})'
    return f'(({_GLASS_INDEX!r} + {dndT!r} * {rise}) * (1.0 + {alpha!r} * {rise})) ** 2'


def _thermo_optic_couplings(transfer_slots):
    """heat_em_thermo_optic, or with transfer_slots one field_transfer per EM permittivity slot (_DNDT, _HZ_ALPHA)."""
    if not transfer_slots:
        return [CouplingSpec('thermo_optic', 'heat_em_thermo_optic', _HEAT, _EM)]
    expression = _thermo_optic_expression(_DNDT, _HZ_ALPHA)
    return [CouplingSpec(f'transfer_{slot}', 'field_transfer', _HEAT, _EM,
                         params={'source_field': 'T', 'target_slot': slot, 'expression': expression})
            for slot in transfer_slots]


def _strip_spec(run_name, dndT, em_type='em_fdfd_ez', alpha=0.0, transfer_slots=()):
    """FDFD `em_type` + heat strip with the absorbing slab, the mode source, two flux and two mode monitors,
    em_heat_joule, heat_em_thermo_optic (or field_transfer into transfer_slots) and a stationary study; dndT / alpha
    of every cell."""
    slab_x0, slab_x1 = _SLAB_X_RANGE
    glass = _glass_properties(dndT, alpha)
    return ModelSpec(
        title=f'Self-test: EM + heat thermo-optic strip ({em_type}, {run_name}, dndT = {dndT:g} /K, alpha = '
              f'{alpha:g} /K)',
        domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]),
        geometry=[_full_period_rectangle(_SLAB, slab_x0, slab_x1), *_line_geometry(_LINE_CELLS)],
        materials=[MaterialSpec('glass', properties=glass, selection=['all']),
                   MaterialSpec('absorber', properties={**glass, 'sigma_e': _SLAB_SIGMA}, selection=[_SLAB])],
        physics=[_em_physics(_LINE_CELLS, em_type), _heat_physics()],
        couplings=[CouplingSpec('joule', 'em_heat_joule', _EM, _HEAT), *_thermo_optic_couplings(transfer_slots)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary', settings=dict(_STUDY)),
        derived_values=[DerivedValueSpec('P_absorbed', 'absorbed_power', [_SLAB], 'integral'),
                        DerivedValueSpec('T_max', 'T', [], 'max')])


def _currents_strip_spec(run_name, dndT):
    """em_fdfd_ez + heat + electric_currents strip: the resistor between two terminals, the mode source and two mode
    monitors, currents_heat_joule + heat_em_thermo_optic and a stationary study; dndT of every cell."""
    resistor_x0, resistor_x1 = _SLAB_X_RANGE
    glass = _glass_properties(dndT)
    return with_exact_cg(ModelSpec(
        title=f'Self-test: currents + heat thermo-optic strip ({run_name}, dndT = {dndT:g} /K)',
        domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]),
        geometry=[_full_period_rectangle(_RESISTOR, resistor_x0, resistor_x1),
                  _full_period_rectangle(_TERMINAL_A, resistor_x0, resistor_x0 + _TERMINAL_LENGTH),
                  _full_period_rectangle(_TERMINAL_B, resistor_x1 - _TERMINAL_LENGTH, resistor_x1),
                  *_line_geometry(_CURRENTS_LINE_CELLS)],
        materials=[MaterialSpec('glass', properties=glass, selection=['all']),
                   MaterialSpec(_RESISTOR, properties={**glass, 'sigma_dc': _RESISTOR_SIGMA}, selection=[_RESISTOR])],
        physics=[_em_physics(_CURRENTS_LINE_CELLS), _heat_physics(),
                 PhysicsSpec(_CURRENTS, 'electric_currents', features=[
                     FeatureSpec(_TERMINAL_A, 'terminal', [_TERMINAL_A], {'voltage': _TERMINAL_VOLTAGE}),
                     FeatureSpec(_TERMINAL_B, 'terminal', [_TERMINAL_B], {'voltage': 0.0})])],
        couplings=[CouplingSpec('joule', 'currents_heat_joule', _CURRENTS, _HEAT),
                   CouplingSpec('thermo_optic', 'heat_em_thermo_optic', _HEAT, _EM)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='stationary', settings=dict(_STUDY)),
        derived_values=[DerivedValueSpec('P_joule', f'{_CURRENTS}_q_joule', ['all'], 'integral'),
                        DerivedValueSpec('T_max', 'T', [], 'max')]), [_CURRENTS])


def _strip_specs():
    """'reference' (dndT 0) then 'heated' (dndT _DNDT): the same layout, sources and study."""
    return {_REFERENCE: _strip_spec(_REFERENCE, 0.0), _HEATED: _strip_spec(_HEATED, _DNDT)}


def _currents_strip_specs():
    """'reference' (dndT 0) then 'heated' (dndT _DNDT) of the currents strip."""
    return {_REFERENCE: _currents_strip_spec(_REFERENCE, 0.0), _HEATED: _currents_strip_spec(_HEATED, _DNDT)}


def _hz_strip_specs():
    """em_fdfd_hz strip: 'reference' (dndT 0), 'heated' (dndT _DNDT, alpha _HZ_ALPHA), 'transfer' (the heated eps
    through field_transfer into eps_xx_update / eps_yy_update; material dndT and alpha 0)."""
    hz = 'em_fdfd_hz'
    return {_REFERENCE: _strip_spec(_REFERENCE, 0.0, hz),
            _HEATED: _strip_spec(_HEATED, _DNDT, hz, _HZ_ALPHA),
            _TRANSFER: _strip_spec(_TRANSFER, 0.0, hz, transfer_slots=('eps_xx_update', 'eps_yy_update'))}


def _fdtd_strip_spec(run_name, with_heat, thermo_optic, duration=_FDTD_DURATION):
    """em_fdtd_hz strip with the absorbing slab and the gaussian mode source, time_dependent over duration;
    with_heat adds insulated transient heat + em_heat_joule, thermo_optic heat_em_thermo_optic (dndT, alpha)."""
    slab_x0, slab_x1 = _SLAB_X_RANGE
    glass = {'eps_r': _GLASS_EPS_R}
    physics = [PhysicsSpec(_EM, 'em_fdtd_hz', settings={**PERIODIC_Y, 'track_absorption': True}, features=[
        FeatureSpec(_SOURCE, 'mode_source', [_line_name(_SOURCE)], dict(_FDTD_MODE_SOURCE))])]
    couplings = []
    if with_heat:
        glass.update(k=_CONDUCTIVITY, rho=_FDTD_DENSITY, cp=_FDTD_SPECIFIC_HEAT)
        physics.append(PhysicsSpec(_HEAT, 'heat', settings={'T0': _T0, 'time_mode': 'transient',
                                                            'time_step': _FDTD_HEAT_TIME_STEP},
                                   every=_FDTD_HEAT_EVERY))
        couplings.append(CouplingSpec('joule', 'em_heat_joule', _EM, _HEAT))
    if thermo_optic:
        glass.update(dndT=_FDTD_DNDT, alpha=_FDTD_ALPHA)
        couplings.append(CouplingSpec('thermo_optic', 'heat_em_thermo_optic', _HEAT, _EM))
    return ModelSpec(
        title=f'Self-test: FDTD H_z + transient heat Joule energy ({run_name})',
        domain=DomainSpec(size_x=_SIZE[0], size_y=_SIZE[1]),
        geometry=[_full_period_rectangle(_SLAB, slab_x0, slab_x1), _cell_line(_line_name(_SOURCE),
                                                                              _LINE_CELLS[_SOURCE])],
        materials=[MaterialSpec('glass', properties=glass, selection=['all']),
                   MaterialSpec('absorber', properties={**glass, 'sigma_e': _SLAB_SIGMA}, selection=[_SLAB])],
        physics=physics,
        couplings=couplings,
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': duration}))


def _fdtd_specs():
    """'em_only' (whole-run absorbed energy), 'coupled' (+ transient heat), 'heated' (+ thermo-optic, shorter)."""
    return {_EM_ONLY: _fdtd_strip_spec(_EM_ONLY, False, False), _COUPLED: _fdtd_strip_spec(_COUPLED, True, False),
            _HEATED: _fdtd_strip_spec(_HEATED, True, True, _FDTD_HEATED_DURATION)}


# ---- checks (CPU, on Results-like objects) ----
def _window_field(results, field_name):
    """float64 (nx, ny) window values of a field; ValueError unless it has the strip's 400 x 10 cells."""
    values = np.asarray(results.field(field_name).values, dtype=np.float64)
    if values.shape != _N_CELLS:
        raise ValueError(f'{field_name}: expected the window shape {_N_CELLS}, got {values.shape}')
    return values


def _mode_phase(results, feature):
    """(phase_positive [rad], n_eff) of a mode monitor at the single solved wavelength."""
    result = results.result(_EM, feature)
    if not result.get('normalized'):
        raise ValueError(f'{feature}: mode monitor phases need exactly one mode source')
    return (single_value(result['series']['phase_positive'], f'{feature} phase_positive'),
            single_value(result['n_eff'], f'{feature} n_eff'))


def _monitor_phases(results_by_name, run_name):
    """(phases, n_eff, dphi) of the mode monitors of the reference run and run_name ({'<run>_<feature>': value}):
    dphi = wrap([phi_down - phi_up] of run_name - [phi_down - phi_up] of the reference)."""
    phases, n_eff = {}, {}
    for name in (_REFERENCE, run_name):
        for feature in (_MODE_UP, _MODE_DOWN):
            phases[f'{name}_{feature}'], n_eff[f'{name}_{feature}'] = _mode_phase(results_by_name[name], feature)
    dphi = _wrap((phases[f'{run_name}_{_MODE_DOWN}'] - phases[f'{run_name}_{_MODE_UP}'])
                 - (phases[f'{_REFERENCE}_{_MODE_DOWN}'] - phases[f'{_REFERENCE}_{_MODE_UP}']))
    return phases, n_eff, dphi


def _dirichlet_profile(x_centres, power_density):
    """theta(x_i) of k theta'' = -q with theta = 0 at both window faces: sum_j G(x_i, s_j) q_j dx / k with G(x, s) =
    s (L - x) / L for s <= x, x (L - s) / L for s > x (the self-cell uses the same formula)."""
    window_length = _SIZE[0]
    x = x_centres[:, None]
    s = x_centres[None, :]
    green = np.where(s <= x, s * (window_length - x), x * (window_length - s)) / window_length
    return green @ power_density * _CELL / _CONDUCTIVITY


def _discrete_wavenumber(index):
    """k along x of the FDFD plane wave in a medium of index `index`: (2 / dx) arcsin(n k0 dx / 2)."""
    return 2.0 / _CELL * np.arcsin(index * _K0 * _CELL / 2.0)


def _monitor_phase_shift(index, reference_index=_GLASS_INDEX):
    """Discrete-k phase shift [rad] between the mode monitor line cells of an (nx,) index profile against
    reference_index: -sum [k_num(index) - k_num(reference_index)] dx, trapezoid between the line-cell centres."""
    wavenumber_shift = _discrete_wavenumber(index) - _discrete_wavenumber(reference_index)
    path = slice(_LINE_CELLS[_MODE_UP], _LINE_CELLS[_MODE_DOWN] + 1)
    trapezoid = np.ones(_LINE_CELLS[_MODE_DOWN] - _LINE_CELLS[_MODE_UP] + 1)
    trapezoid[0] = trapezoid[-1] = 0.5
    return -float(np.sum(trapezoid * wavenumber_shift[path])) * _CELL


def _wrap(angle):
    """angle [rad] wrapped into (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _summaries(results_by_name):
    return {run_name: results.summary() for run_name, results in results_by_name.items()}


def _profile_sample():
    return slice(0, _N_CELLS[0], max(1, _N_CELLS[0] // _PROFILE_SAMPLES))


def _profile_check(reference, power_density_field):
    """(err_profile, details) of the reference run: theta = T - T0 (y-mean per cell) against the Dirichlet Green sum
    of its own q(x) = the y-mean of `power_density_field` [W/m^3]."""
    x_centres = cell_centre(np.arange(_N_CELLS[0], dtype=np.float64), _CELL)
    power_density = _window_field(reference, power_density_field).mean(axis=1)
    temperature = _window_field(reference, f'{_HEAT}.T')
    theta_measured = temperature.mean(axis=1) - _T0
    theta_closed = _dirichlet_profile(x_centres, power_density)
    theta_peak_closed = float(np.max(theta_closed))
    err_profile = float(np.max(np.abs(theta_measured - theta_closed))) / theta_peak_closed
    sample = _profile_sample()
    details = {'theta_max_measured_K': float(np.max(theta_measured)), 'theta_max_closed_K': theta_peak_closed,
               'T_y_variation_relative': float(np.max(np.ptp(temperature, axis=1))) / theta_peak_closed,
               'x_profile_m': float_list(x_centres[sample]), 'theta_measured_K': float_list(theta_measured[sample]),
               'theta_closed_K': float_list(theta_closed[sample])}
    return err_profile, details


def _heated_index(theta, dndT, alpha):
    """sqrt(eps) of the heated glass: (n0 + dndT theta)(1 + alpha theta), the alpha factor on eps, never on n."""
    return (_GLASS_INDEX + dndT * theta) * (1.0 + alpha * theta)


def _eps_error(results, theta_field, eps_fields, dndT, alpha, rise=False):
    """max over eps_fields of max |eps - ((n0 + dndT theta)(1 + alpha theta))^2| / eps_r, theta from theta_field
    (minus T0 unless rise)."""
    theta = _window_field(results, theta_field) - (0.0 if rise else _T0)
    eps_expected = _heated_index(theta, dndT, alpha) ** 2
    return max(float(np.max(np.abs(_window_field(results, eps_field) - eps_expected))) / _GLASS_EPS_R
               for eps_field in eps_fields)


def _phase_check(results_by_name, heated_name=_HEATED, alpha=0.0, eps_fields=(f'{_EM}.eps_r',)):
    """(dphi_measured, dphi_predicted, err_phase, err_eps, details): the thermo-optic phase shift between the mode
    monitors, heated_name minus reference, against the discrete-k prediction from its theta (index (n0 + _DNDT theta)
    (1 + alpha theta)); err_eps = its eps_fields against that index squared."""
    heated = results_by_name[heated_name]
    phases, n_eff, dphi_measured = _monitor_phases(results_by_name, heated_name)
    theta_heated = _window_field(heated, f'{_HEAT}.T').mean(axis=1) - _T0
    dphi_predicted = _monitor_phase_shift(_heated_index(theta_heated, _DNDT, alpha))
    err_phase = abs(dphi_measured - dphi_predicted) / abs(dphi_predicted)
    err_eps = _eps_error(heated, f'{_HEAT}.T', eps_fields, _DNDT, alpha)
    details = {'theta_max_heated_K': float(np.max(theta_heated)),
               'dphi_measured_rad': dphi_measured, 'dphi_predicted_rad': dphi_predicted,
               'phase_positive_rad': phases, 'n_eff': n_eff,
               'theta_heated_K': float_list(theta_heated[_profile_sample()])}
    return dphi_measured, dphi_predicted, err_phase, err_eps, details


def _unconverged_failures(summaries):
    """Failed-check text per run with a 'did not converge' warning (stationary fixed point or a CG solve)."""
    failed = []
    for run_name, summary in summaries.items():
        unconverged = [text for text in summary['warnings'] if 'did not converge' in text]
        if unconverged:
            failed.append(f'{run_name}: {unconverged[0]}')
    return failed


def _eps_failures(run_name, err_eps):
    """Failed-check text when err_eps exceeds _EPS_TOLERANCE (NaN counts as failed)."""
    if err_eps <= _EPS_TOLERANCE:
        return []
    return [f'{run_name}: eps deviates from ((n0 + dndT theta)(1 + alpha theta))^2 by {err_eps:.3g} > '
            f'{_EPS_TOLERANCE:g}']


def _fixed_point_failures(summaries, max_ticks, err_eps):
    """Failed-check texts: iteration counts above max_ticks, 'did not converge' warnings, err_eps above its limit."""
    failed = []
    for run_name, summary in summaries.items():
        ticks = int(summary['ticks'])
        if ticks > max_ticks[run_name]:
            failed.append(f"{run_name}: {ticks} iterations > {max_ticks[run_name]}")
    return failed + _unconverged_failures(summaries) + _eps_failures(_HEATED, err_eps)


def _strip_outcome(normalized_errors, failed, summaries, dphi_measured, dphi_predicted, details):
    """Outcome dict: error = the largest normalised error + the number of failed checks (tolerance 1), measured /
    expected = dphi; details gain the errors, failed checks, iteration counts and run warnings."""
    error = float(np.max(np.asarray(list(normalized_errors.values())))) + len(failed)   # NaN propagates
    details.update(normalized_errors=normalized_errors, failed_checks=failed,
                   ticks={run_name: int(summary['ticks']) for run_name, summary in summaries.items()},
                   warnings={run_name: list(summary['warnings']) for run_name, summary in summaries.items()})
    return {'measured': dphi_measured, 'expected': dphi_predicted, 'error': error, 'tolerance': 1.0,
            'details': details}


def _check_strip(results_by_name):
    """Energy balance and conduction profile of the reference run, thermo-optic phase shift heated vs reference and
    the fixed-point iteration counts; error = max relative error in tolerance units + number of failed checks."""
    return _strip_outcome(*_strip_errors(results_by_name, _MAX_TICKS))


def _check_hz_strip(results_by_name):
    """_check_strip on em_fdfd_hz with alpha (eps_xx and eps_yy) plus the transfer run: its dphi against the heated
    one and its eps against its own theta."""
    normalized_errors, failed, summaries, dphi_measured, dphi_predicted, details = _strip_errors(
        results_by_name, _HZ_MAX_TICKS, _HZ_ALPHA, _HZ_EPS_FIELDS)
    dphi_transfer, _, _, err_eps_transfer, _ = _phase_check(results_by_name, _TRANSFER, _HZ_ALPHA, _HZ_EPS_FIELDS)
    err_transfer = abs(dphi_transfer - dphi_measured) / abs(dphi_measured)
    normalized_errors['transfer'] = err_transfer / _TRANSFER_TOLERANCE
    failed += _eps_failures(_TRANSFER, err_eps_transfer)
    details.update(err_transfer=err_transfer, err_eps_transfer=err_eps_transfer, dphi_transfer_rad=dphi_transfer)
    return _strip_outcome(normalized_errors, failed, summaries, dphi_measured, dphi_predicted, details)


def _strip_errors(results_by_name, max_ticks, alpha=0.0, eps_fields=(f'{_EM}.eps_r',)):
    """(normalized_errors, failed, summaries, dphi_measured, dphi_predicted, details) of the absorbing strip: balance,
    source power, profile, phase (heated vs reference, alpha) and fixed-point checks."""
    reference = results_by_name[_REFERENCE]
    summaries = _summaries(results_by_name)

    # 1. energy balance of the reference run: net flux in - net flux out = absorbed power
    power_up = single_value(reference.result(_EM, _FLUX_UP)['power'], f'{_FLUX_UP} power')
    power_down = single_value(reference.result(_EM, _FLUX_DOWN)['power'], f'{_FLUX_DOWN} power')
    power_absorbed = float(summaries[_REFERENCE]['derived']['P_absorbed'])   # float() also reads 'nan'
    err_balance = abs((power_up - power_down) - power_absorbed) / power_up
    err_source = abs(power_up / _MODE_SOURCE['power'] - 1.0)   # absolute mode-source power on the flux monitor

    # 2. temperature profile of the reference run against the Dirichlet Green's function of its own q(x)
    err_profile, profile_details = _profile_check(reference, f'{_EM}.absorbed_power')

    # 3. thermo-optic phase shift between the mode monitors, heated minus reference
    dphi_measured, dphi_predicted, err_phase, err_eps, phase_details = _phase_check(results_by_name, _HEATED, alpha,
                                                                                    eps_fields)

    # 4. fixed point: iteration counts, convergence warnings, the thermo-optic transform itself
    failed = _fixed_point_failures(summaries, max_ticks, err_eps)

    normalized_errors = {'balance': err_balance / _BALANCE_TOLERANCE, 'source': err_source / _SOURCE_POWER_TOLERANCE,
                         'profile': err_profile / _PROFILE_TOLERANCE, 'phase': err_phase / _PHASE_TOLERANCE}
    details = {
        'err_balance': err_balance, 'err_source': err_source, 'err_profile': err_profile, 'err_phase': err_phase,
        'err_eps': err_eps,
        'unit_power': 'W/m', 'P_up': power_up, 'P_down': power_down, 'P_absorbed': power_absorbed,
        'absorbed_fraction': power_absorbed / power_up if power_up != 0.0 else math.nan,
        **profile_details, **phase_details,
    }
    return normalized_errors, failed, summaries, dphi_measured, dphi_predicted, details


def _check_fdtd_energy(results_by_name):
    """Transient heat energy sum rho cp theta dA of the coupled run against the whole-run absorbed energy of em_only
    (tail of the flushed Q bounded), plus the heated run's eps_xx / eps_yy transform with alpha."""
    summaries = _summaries(results_by_name)
    cell_area = _CELL * _CELL
    heat_capacity = _FDTD_DENSITY * _FDTD_SPECIFIC_HEAT   # J/(m^3 K)

    # 1. em_only: no em_heat_joule, so absorbed_power is the mean over the whole run (q_accum_time = t_end)
    t_end = float(summaries[_EM_ONLY]['t_end'])
    absorbed_power = _window_field(results_by_name[_EM_ONLY], f'{_EM}.absorbed_power')
    energy_absorbed = float(np.sum(absorbed_power)) * cell_area * t_end

    # 2. coupled: insulated implicit Euler conserves sum rho cp theta dA = the energy of every stepped window
    coupled = results_by_name[_COUPLED]
    energy_heat = heat_capacity * float(np.sum(_window_field(coupled, f'{_HEAT}.T_rise'))) * cell_area
    err_energy = abs(energy_heat - energy_absorbed) / energy_absorbed

    # 3. the flushed Q (last partial window merged with the previous full one, never stepped) must be negligible
    window_seconds = _FDTD_HEAT_EVERY * float(summaries[_COUPLED]['t_end']) / int(summaries[_COUPLED]['ticks'])
    tail_energy = float(np.sum(_window_field(coupled, f'{_HEAT}.Q'))) * cell_area * 2.0 * window_seconds
    tail_fraction = tail_energy / energy_absorbed

    # 4. heated: eps_xx / eps_yy = ((n0 + dndT theta)(1 + alpha theta))^2 of the final temperature
    heated = results_by_name[_HEATED]
    err_eps = _eps_error(heated, f'{_HEAT}.T_rise', _HZ_EPS_FIELDS, _FDTD_DNDT, _FDTD_ALPHA, rise=True)
    energy_heated = heat_capacity * float(np.sum(_window_field(heated, f'{_HEAT}.T_rise'))) * cell_area

    failed = _unconverged_failures(summaries) + _eps_failures(_HEATED, err_eps)
    if not tail_fraction <= _TAIL_LIMIT:   # NaN counts as failed
        failed.append(f'{_COUPLED}: flushed tail {tail_fraction:.3g} of the absorbed energy > {_TAIL_LIMIT:g}')
    pulse_duration = 1.0 / (math.pi * _FDTD_MODE_SOURCE['bandwidth'] * C0 / _WAVELENGTH)   # tau of the envelope
    energy_pulse = _FDTD_MODE_SOURCE['power'] * pulse_duration * math.sqrt(math.pi / 2.0)
    error = err_energy / _ENERGY_TOLERANCE + len(failed)   # NaN propagates
    details = {'err_energy': err_energy, 'tail_fraction': tail_fraction, 'err_eps': err_eps, 'unit_energy': 'J/m',
               'E_absorbed': energy_absorbed, 'E_heat': energy_heat, 'E_heat_heated': energy_heated,
               'E_pulse_closed': energy_pulse, 'absorbed_fraction': energy_absorbed / energy_pulse,
               'theta_max_coupled_K': float(np.max(_window_field(coupled, f'{_HEAT}.T_rise'))),
               'failed_checks': failed,
               'ticks': {run_name: int(summary['ticks']) for run_name, summary in summaries.items()},
               'warnings': {run_name: list(summary['warnings']) for run_name, summary in summaries.items()}}
    return {'measured': energy_heat, 'expected': energy_absorbed, 'error': error, 'tolerance': 1.0,
            'details': details}


def _check_currents_strip(results_by_name):
    """Terminal currents and the Joule power identity of the reference run, its conduction profile from q_joule, the
    thermo-optic phase shift heated vs reference and the fixed-point counts; error as _check_strip."""
    reference = results_by_name[_REFERENCE]
    summaries = _summaries(results_by_name)

    # 1. terminal currents against sigma w V / L and the integral of q_joule against V I_a (reference run)
    current_a = float(reference.result(_CURRENTS, _TERMINAL_A)['current'])
    current_b = float(reference.result(_CURRENTS, _TERMINAL_B)['current'])
    power_joule = float(summaries[_REFERENCE]['derived']['P_joule'])   # float() also reads 'nan'
    err_current = float(np.max(np.abs([current_a - _RESISTOR_CURRENT, current_b + _RESISTOR_CURRENT])))
    err_current /= _RESISTOR_CURRENT
    power_terminal = _TERMINAL_VOLTAGE * current_a
    err_power = abs(power_joule - power_terminal) / abs(power_terminal)

    # 2. temperature profile of the reference run against the Dirichlet Green's function of its own q_joule(x)
    err_profile, profile_details = _profile_check(reference, f'{_CURRENTS}.q_joule')

    # 3. thermo-optic phase shift between the mode monitors, heated minus reference
    dphi_measured, dphi_predicted, err_phase, err_eps, phase_details = _phase_check(results_by_name)

    # 4. fixed point: iteration counts, convergence warnings (stationary, heat CG, currents CG), the transform
    failed = _fixed_point_failures(summaries, _CURRENTS_MAX_TICKS, err_eps)

    normalized_errors = {'current': err_current / _CURRENT_TOLERANCE, 'power': err_power / _CURRENT_TOLERANCE,
                         'profile': err_profile / _PROFILE_TOLERANCE, 'phase': err_phase / _PHASE_TOLERANCE}
    details = {
        'err_current': err_current, 'err_power': err_power, 'err_profile': err_profile, 'err_phase': err_phase,
        'err_eps': err_eps, 'unit_current': 'A/m', 'I_a': current_a, 'I_b': current_b, 'I_closed': _RESISTOR_CURRENT,
        'P_joule_W_per_m': power_joule, 'P_terminal_W_per_m': power_terminal,
        'q_joule_closed_W_per_m3': _RESISTOR_JOULE,
        **profile_details, **phase_details,
    }
    return _strip_outcome(normalized_errors, failed, summaries, dphi_measured, dphi_predicted, details)


CASE = SelfTestCase(
    name='em_heat_thermo_optic_strip',
    label='EM + heat: absorption, conduction and thermo-optic phase (stationary chain)',
    description='A mode source (mode 1, direction positive, 1 W/m, phase 0) launches the plane-wave mode of a '
                'y-periodic glass strip (n = 1.5, 8.0 x 0.2 um, CPML along x, 20 nm mesh, 4,300 em_fdfd_ez unknowns) '
                'at the vacuum wavelength 1.55 um from x = 1.01 um; a slab x = 2..6 um adds sigma_e = 2.0e3 S/m ~ '
                '2 n / (eta0 L) (Im eps / eps_r = 0.083, field attenuation alpha = sigma eta0 / (2 n) = 2.5e5 /m, '
                'alpha L = 1, so it absorbs 1 - exp(-2 alpha L) = 86 % of the mode); flux monitors at 1.31 and 6.71 um '
                'and mode monitors at 1.51 and 6.51 um (5 um apart) straddle it. The heat physics (k = 0.2 W/(m K) '
                'everywhere, walls at T0 = 293.15 K on x_lo / x_hi, insulated y faces: 1D in x) takes the absorbed '
                'power through em_heat_joule and returns eps = (n0 + dndT theta)^2 through heat_em_thermo_optic in a '
                'stationary study (tolerance 2e-6, max 30 iterations). Two runs: reference with dndT = 0 and heated '
                'with dndT = 1e-3 /K. Checks: (1) energy balance of the reference run, err_balance = |(P_up - P_down) '
                '- P_absorbed| / P_up with P_absorbed = the integral of absorbed_power over the slab, tolerance 1 %: '
                'the discrete Poynting balance of the Yee FDFD with q_abs = 1/2 sigma |E_z|^2 at the centres is exact '
                'up to float32 solution rounding, and the net upstream flux already holds the |r|^2 ~ (Im eps / (4 '
                'eps_r))^2 ~ 4e-4 slab reflection. Also err_source = |P_up / 1 W/m - 1|, tolerance 1 %: the port '
                'mode is normalised to 1 W/m of discrete Yee flux, so P_up differs from the source power only by '
                'that |r|^2 (this pins the absolute mode-source power and with it the W/m^3 scale of q). (2) '
                'Temperature profile of the reference run: theta = T - T0 '
                '(y-mean per cell) against the Dirichlet Green function of k theta\'\' = -q(x) on the window [0, 8 um] '
                'with q = the y-mean absorbed_power per cell, G(x, s) = s (L - x) / L for s <= x and x (L - s) / L for '
                's > x, summed over the cells as G(x_i, s_j) q_j dx / k (the self-cell included with the same '
                'formula); with E0^2 = 2 (P / w) eta0 / n = 2.5e9 V^2/m^2 the front of the slab has q0 = 1/2 sigma '
                'E0^2 = 2.5e12 W/m^3 and the profile peaks at ~33 K near x = 3.4 um; err_profile = max |theta - '
                'theta_closed| / max theta_closed, tolerance 1e-4: the Green sum sampled at the cell centres is the '
                'exact solution of the 3-point finite-volume scheme (the second difference of G(., s_j) is -1 / dx at '
                'i = j and 0 elsewhere, and the half-cell Dirichlet conductance 2 k / dx is the linear extrapolation '
                'of G to the face), so the check is an identity up to the float32 T (1e-6 of the rise) and the CG '
                'residual 1e-8: it pins the W/m^3 scaling of em_heat_joule, the Dirichlet conductance and the dx and '
                'k factors. (3) '
                'Thermo-optic phase: dphi = wrap([phi_down - phi_up]_heated - [phi_down - phi_up]_reference) from the '
                'phase_positive series of the mode monitors against -sum [k_num(n0 + dndT theta_i) - k_num(n0)] dx '
                '(trapezoid over the cell centres between the two monitor line cells, theta_i from the heated run, '
                'k_num(n) = (2 / dx) arcsin(n k0 dx / 2) the discrete plane-wave k of the 5-point stencil); the '
                'profile integrates to ~1.3e-4 K m between the monitors, so the prediction is ~ -k0 dndT int(theta '
                'dx) = -0.54 rad with dn / n at most 2.2 %; err_phase = |dphi - dphi_predicted| / |dphi_predicted|, '
                'tolerance 2 %: the sum is first order in dn (the index gradient reflects ~(dn / n)^2 ~ 5e-4; the '
                'source and monitor modes re-solve in their local n, moving their half-cell measuring offsets by '
                '~k dx / 2 dn / n ~ 1e-3 rad; the loss correction of Re k, s^2 / (8 n^3) with s = Im eps = 0.19, '
                'changes between the runs by 3 s^2 / (8 n^4) ~ 0.3 % of the shift). Also err_eps = max |eps_r - (n0 + '
                'dndT theta)^2| / eps_r over the heated window (the transform itself, ~1e-7 in float32; a failed check '
                'above 1e-3). (4) Fixed point (Gauss-Seidel, em then heat): the reference run converges in <= 3 '
                'iterations (1: EM from zero, then T from its absorbed power; 2: nothing changes), the heated run in '
                '<= 10 (the loop gain is ~1 %: a 2 % index rise lowers '
                'alpha by 2 % and the absorbed power by ~0.5 %), and no warning contains \'did not converge\' (the '
                'stationary fixed point or the heat CG); each violation is a failed check. error = max(err_balance / '
                '0.01, err_source / 0.01, err_profile / 1e-4, err_phase / 0.02) + the number of failed checks, '
                'tolerance 1; measured / '
                'expected = the measured / predicted dphi in rad; details list every error, both iteration counts, '
                'P_up, P_down, P_absorbed, the peak rises, the four phases and n_eff values, a 40-sample x '
                'profile of theta measured / closed / heated and the run warnings. The stationary tolerance is 2e-6 '
                'because T is stored as float32: one ulp at 326 K (3.05e-5 K) is 9.3e-7 of the 33 K rise.',
    specs=_strip_specs,
    check=_check_strip)

CURRENTS_CASE = SelfTestCase(
    name='currents_heat_thermo_optic_strip',
    label='Currents + heat: terminal currents, Joule power, conduction and thermo-optic phase (stationary chain)',
    description='The glass strip of em_heat_thermo_optic_strip (n = 1.5, k = 0.2 W/(m K), 8.0 x 0.2 um, y-periodic, '
                'CPML along x, 20 nm mesh) carries a resistor x = 2..6 um with sigma_dc = 15 S/m (no sigma_e: the '
                'glass stays lossless and the EM mode is the plane wave of uniform glass) whose ends are terminals of '
                'the electric_currents physics (float64 CG to 1e-12): terminal_a = cells 100..104 (x = 2.0..2.1 um) '
                'at 1 V, terminal_b = cells 295..299 (x = 5.9..6.0 um) at 0 V. currents_heat_joule hands q_joule to '
                'the heat physics (walls at T0 = 293.15 K on x_lo / x_hi, insulated y faces: 1D in x) and '
                'heat_em_thermo_optic returns eps = (n0 + dndT theta)^2 to the em_fdfd_ez mode (source at 1.01 um, '
                '1 W/m, mode monitors at 1.51 and 6.51 um) in a stationary study (tolerance 2e-6, max 30 '
                'iterations); the chain is feed-forward, currents -> heat -> EM. Two runs: reference with dndT = 0 '
                'and heated with dndT = 1e-3 /K. Closed forms: the terminals are ideal conductors whose voltage sits '
                'on their inner faces x = 2.1 and 5.9 um (half-cell conductance 2 sigma / dx toward the free cells), '
                'L = 3.8 um, so I = sigma w V / L = 15 * 0.2e-6 * 1 / 3.8e-6 = 0.789474 A/m '
                '(terminal_a +I, terminal_b -I: positive = out of the held cells into the domain), P = V I = 0.789474 '
                'W/m and q0 = sigma (V / L)^2 = 1.03878e12 W/m^3 in cells 105..294 (0 elsewhere: a held-free face '
                'dissipates wholly in its free half cell): the 3-point '
                'finite-volume scheme is exact for the linear profile and the face-split q_joule integrates to V I '
                'exactly. Checks: (1) err_current = max(|I_a - I|, |I_b + I|) / I from the terminal table results and '
                'err_power = |P_joule - V I_a| / (V I_a) with P_joule = the integral of currents_q_joule over the '
                'window, both of the reference run, tolerance 1e-6 (the CG residual 1e-12 times a condition number '
                '~1e4, and q_joule stored as float32, ~1e-7). (2) Temperature profile of the reference run: theta = '
                'T - T0 (y-mean per cell) against the Dirichlet Green function of k theta\'\' = -q(x) on [0, 8 um] '
                'with q = the y-mean of currents_q_joule per cell, G(x, s) = s (L - x) / L for s <= x and x (L - s) '
                '/ L for s > x, summed as G(x_i, s_j) q_j dx / k; err_profile = max |theta - theta_closed| / max '
                'theta_closed, tolerance 1e-4: the same exact identity of the finite-volume scheme as in '
                'em_heat_thermo_optic_strip, here pinning the W/m^3 scaling of currents_heat_joule; the continuum '
                'profile is theta = 9.9 K/um x outside the source and peaks at q0 * 5.795e-12 m^2 / k ~ 30.1 K at x = '
                '4 um (~14.9 K at the upstream monitor, 14.7 K at the downstream one). (3) Thermo-optic phase: dphi = '
                'wrap([phi_down - phi_up]_heated - [phi_down - phi_up]_reference) from the phase_positive series of '
                'the mode monitors against -sum [k_num(n0 + dndT theta_i) - k_num(n0)] dx (trapezoid over the cell '
                'centres between the monitor line cells, theta_i from the heated run, k_num(n) = (2 / dx) arcsin(n k0 '
                'dx / 2)); theta integrates to ~1.24e-4 K m between the monitors, so the prediction is ~ -k0 dndT '
                'int(theta dx) = -0.50 rad with dn / n at most 2 %; err_phase = |dphi - dphi_predicted| / '
                '|dphi_predicted|, tolerance 2 % (first order in dn: the index gradient reflects ~(dn / n)^2 ~ 4e-4 '
                'and the re-solved source and monitor modes move their half-cell measuring offsets by ~1e-3 rad; no '
                'loss correction, the glass is lossless). Also err_eps = max |eps_r - (n0 + dndT theta)^2| / eps_r '
                'over the heated window (a failed check above 1e-3). (4) Fixed point: the chain is feed-forward, so '
                'every physics is a sink and both runs take exactly 1 iteration (currents, joule, heat, '
                'thermo_optic, EM, each solved once), and no warning contains \'did not converge\' (the stationary '
                'fixed point, the heat CG or the currents CG); each violation is a failed check. error = '
                'max(err_current / 1e-6, err_power / 1e-6, err_profile / 1e-4, err_phase / 0.02) + the number of '
                'failed checks, tolerance 1; measured / expected = the measured / predicted dphi in rad; details list '
                'every error, I_a, I_b, P_joule, the closed-form I and q0, both iteration counts, the peak rises, the '
                'four phases and n_eff values, a 40-sample x profile of theta measured / closed / heated and '
                'the run warnings.',
    specs=_currents_strip_specs,
    check=_check_currents_strip)

HZ_CASE = SelfTestCase(
    name='em_hz_heat_thermo_expansion_strip',
    label='EM (H_z) + heat: absorption, conduction, thermo-optic + expansion phase, field_transfer (stationary)',
    description='The em_heat_thermo_optic_strip layout and checks with em_fdfd_hz: at normal incidence E_y is '
                'tangential, so eps_yy drives the same discrete k_num(n) = (2 / dx) arcsin(n k0 dx / 2), and q_abs '
                '(each cell\'s -Im eps times the mean |E|^2 of its two faces) sums to the face-averaged Poynting loss, '
                'so the balance (1 %), source power (1 %) and profile (1e-4) checks hold as for E_z. Three runs: '
                'reference (dndT = 0), heated (dndT = 1e-3 /K and alpha = 1e-4 /K: heat_em_thermo_optic writes eps_xx '
                'and eps_yy = (n0 + dndT dT)^2 (1 + alpha dT)^2 through eps_xx_update / eps_yy_update) and transfer '
                '(material dndT = alpha = 0; two field_transfer couplings write the same expression of T into '
                'eps_xx_update and eps_yy_update). Phase: dphi heated - reference against -sum [k_num((n0 + dndT '
                'theta)(1 + alpha theta)) - k_num(n0)] dx (trapezoid between the monitor line cells), tolerance 2 %; '
                'the expansion adds ~5e-3 of the ~0.038 index rise (13 % of the shift), so a missing alpha or one '
                'applied on n instead of eps fails. err_eps over eps_xx and eps_yy of the heated and transfer runs '
                '(failed check above 1e-3); err_transfer = |dphi_transfer - dphi_heated| / |dphi_heated|, tolerance '
                '1e-3 (the same eps up to float32 rounding and the 2e-6 fixed point). Iterations: reference <= 3, '
                'heated and transfer <= 10, no \'did not converge\' warning. error = max(err_balance / 0.01, '
                'err_source / 0.01, err_profile / 1e-4, err_phase / 0.02, err_transfer / 1e-3) + the number of '
                'failed checks, tolerance 1; measured / expected = the heated dphi in rad.',
    specs=_hz_strip_specs,
    check=_check_hz_strip)

FDTD_CASE = SelfTestCase(
    name='fdtd_heat_joule_energy',
    label='FDTD (H_z) + transient heat: windowed Joule energy, thermo-optic slots with expansion',
    description='The glass strip (n = 1.5, 8.0 x 0.2 um, y-periodic, CPML along x, 20 nm mesh) with the absorbing '
                'slab x = 2..6 um (sigma_e = 2.0e3 S/m) in em_fdtd_hz: a gaussian mode source (mode 1, positive, '
                '1.55 um, bandwidth 0.3, 1 W/m at the envelope peak, energy P tau sqrt(pi / 2) ~ 6.9e-15 J/m) at x = '
                '1.01 um, duration 140 fs (the pulse has crossed the slab by ~85 fs; ~6 heat windows of margin before '
                'the two never-stepped ones). Runs: em_only (EM alone, '
                'track_absorption: absorbed_power is the mean over the whole run, so E_abs = integral of '
                'absorbed_power x t_end), coupled (+ transient heat, rho cp = 1e-3 J/(m^3 K), k = 0.2 W/(m K), all '
                'faces insulated, every = 200 EM steps, em_heat_joule) and heated (coupled + heat_em_thermo_optic with '
                'dndT = 1e-3 /K, alpha = 1e-3 /K, stopped at 100 fs: its eps check holds at any time). Checks: err_energy = |E_heat - E_abs| / E_abs with E_heat = sum rho '
                'cp T_rise dA of the coupled run, tolerance 1e-3: implicit Euler with insulated faces conserves the '
                'heat exactly and the EM of both runs is identical, so this pins the window means (q_accum / '
                'q_accum_time), every = the heat every, the W/m^3 scaling and the transient step; the last window and '
                'the flushed Q are never stepped, so tail_fraction = sum Q dA x 2 W / E_abs of the coupled run '
                'must stay below 1e-4 (else a failed check: the pulse did not finish). err_eps = max |eps - ((n0 + dndT theta)(1 '
                '+ alpha theta))^2| / eps_r over eps_xx and eps_yy of the heated run (H_z FDTD slots, failed check '
                'above 1e-3; 2 alpha theta ~ 7-8e-3 at theta ~ 3.5-4 K). No \'did not converge\' warning (heat CG). '
                'error = err_energy / 1e-3 + the number of failed checks, tolerance 1; measured / expected = E_heat / '
                'E_abs in J/m. Not covered: the effect of the refreshed H_z coefficients on the fields, and the '
                'flushed Q value itself. Not yet run at 140 / 100 fs.',
    specs=_fdtd_specs,
    check=_check_fdtd_energy)

# CASE and CURRENTS_CASE run only inside coupling_cpu_gpu (selftests_backend): examples/multiphysics
# analytic_lossy_slab_heating (em_heat_joule balance and profile) and analytic_thermo_optic_phase_shifter
# (currents_heat_joule current / power / profile identities, the em_fdfd_ez thermo-optic phase) check the same chains.
SELFTEST_CASES = (HZ_CASE, FDTD_CASE)
