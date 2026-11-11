"""C5 self-tests of the nonlinear_medium / gain_medium features of em_fdtd_ez / _hz: Kerr self-phase modulation,
two-photon absorption (+ the absorbed = incident - transmitted balance), TPA free carriers (absorption + dispersion)
and chi2 second-harmonic generation (unpoled and with one inverted domain) in an n0 = 2 background and four-level
small-signal gain in vacuum, of a cw plane wave crossing an index-matched slab, against the cycle-averaged z-ODE / the
Lorentz gain line / the space-time lattice Green's function; plus the 'nonlinear_silicon_slab' template. CPU only,
every spec is built fresh on each call."""
import math

import numpy as np

from ...core.constants import C0, EPS0, HBAR
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import PERIODIC_Y, relative_error, vertical_line
from ...model.spec import (DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                           PhysicsSpec, StudySpec)

_PHYSICS_NAME = 'em'
_PROBE_FEATURE = 'probe'
_MEDIUM_FEATURE = 'medium'
_ABSORBED_VALUE = 'absorbed_line_power'   # derived value [W/m]: absorbed_power integrated over the window (tpa)
_PHYSICS_TYPES = {'ez': 'em_fdtd_ez', 'hz': 'em_fdtd_hz'}
_PROBED = {'ez': 'ez', 'hz': 'ey'}   # the line source's tangential E

# y-periodic strip (20 nm cells = 50 per vacuum wavelength, CPML along x) filled with a background of index n0 (the
# nonlinear cases n0 = 2, so the n0^2 / n0 factors of chi3, TPA and FCA / FCD are exercised; gain n0 = 1): a cw line
# source at x = 1 um radiates _AMPLITUDE / sqrt(n0) V/m of tangential E toward +x (the same I0 for every n0), the
# nonlinear slab covers x in [2, 4) um (100 whole cells, eps = the background: no linear reflection), the probe sits at
# x = 5.01 um. The field is fitted at the probe over the last _FIT_PERIODS periods (cos, sin at omega and 3 omega (and
# 2 omega for chi2), constant), nonlinear / linear run = sqrt(T) exp(-j phi).
_WAVELENGTH = 1e-6            # m, vacuum
_AMPLITUDE = 1e9              # V/m at n0 = 1: I0 = n0 C0 EPS0 A^2 / 2 = 1.33e15 W/m^2
_GAIN_AMPLITUDE = 1.0         # V/m: the gain case stays small-signal
_BACKGROUND_INDEX = {'kerr': 2.0, 'tpa': 2.0, 'carriers': 2.0, 'gain': 1.0, 'shg': 2.0,
                     'shg_poled': 2.0}   # n0 of the strip per case
_D_EFF = 5e-12                # m/V: E_2w / E_w ~ omega d E_w L / (n0 c) ~ 0.02 (depletion ~4e-4)
_POLING_WALL = 3.5e-6         # m, a cell edge: the shg_poled slab is inverted beyond it (the last quarter)
_SIZE = (6e-6, 0.1e-6)        # m
_CELL = 20e-9                 # m
_SLAB = (2e-6, 4e-6)          # m along x
_SOURCE_X = 1e-6              # m
_PROBE_POINT = (5.01e-6, 0.05e-6)
_FIT_PERIODS = 10
_ODE_STEPS = 4000             # RK4 steps of the expected z-ODE over the slab

# case -> (feature type, its values, run length [s], tolerance, what the error measures). The window starts 10 periods
# before the end, after the ~8 period transit to the probe (n0 = 2): kerr / tpa (3%) run 36 periods (ramp residual
# exp(-6) = 2.5e-3 in amplitude), shg (1%) and gain keep 40 (shg: exp(-22/3) = 6.5e-4).
_CASES = {
    'kerr': ('nonlinear_medium', {'n2': 3e-18}, 36 * _WAVELENGTH / C0, 0.03, 'phase'),
    'tpa': ('nonlinear_medium', {'tpa_beta': 7.5e-11}, 36 * _WAVELENGTH / C0, 0.03, 'loss and power balance'),
    'carriers': ('nonlinear_medium', {'tpa_beta': 7.5e-11, 'carrier_lifetime': 50e-15, 'fca_cross_section': 2.9e-21,
                                      'fcd_coefficient': 6e-28, 'pump_wavelength': _WAVELENGTH},
                 350e-15, 0.05, 'loss and phase'),
    # line centre at the probe wavelength, initial inversion N2 - N1 = 2e22 1/m^3 (no pump, tau_21 >> the run)
    'gain': ('gain_medium', {'wavelength': _WAVELENGTH, 'linewidth': 1e14, 'density': 4e22, 'tau_21': 1e-9,
                             'initial_upper_fraction': 0.5}, 40 * _WAVELENGTH / C0, 0.03, 'gain exponent'),
    'shg': ('nonlinear_medium', {'d_eff': _D_EFF}, 40 * _WAVELENGTH / C0, 0.01, 'second-harmonic amplitude'),
    'shg_poled': ('nonlinear_medium', {'d_eff': _D_EFF, 'poling': f'sign({_POLING_WALL!r} - x)'},
                  40 * _WAVELENGTH / C0, 0.01, 'second-harmonic amplitude'),
}
_CHI2_CASES = ('shg', 'shg_poled')
_CASE_TEXT = {
    'kerr': ('kerr_self_phase', 'Kerr self-phase modulation',
             'n2 = 3e-18 m^2/W: phi = k0 n2 I L ~ 0.05 rad (exactly k0 L (sqrt(n0^2 + 2 n0 n2 I) - n0), the cycle mean '
             'of the instantaneous chi3 = 4/3 eps0 c n0^2 n2; a missing n0^2 would give 4x); error = relative phase '
             'error. Tolerance 3%: the discrete dk / dn = k0 (1 + (k dx)^2 / 8) at 25 cells per medium wavelength '
             '(0.8%), the phase-matched third harmonic (~5% amplitude at the slab end, fitted out at 3 omega but '
             'draining the fundamental by ~0.1%), the ramp residual of the 36-period run (intensity ~5e-3 low at '
             'the window start). 36 periods not yet confirmed by a run.'),
    'tpa': ('tpa_transmission', 'two-photon absorption transmission',
            'beta = 7.5e-11 m/W: T = 1 / (1 + beta I L) = 0.833 from dI/dz = -beta I^2; error = the larger of the '
            'relative error of the absorbed fraction 1 - T and of the power balance: with track_absorption, the '
            'absorbed power (absorbed_power integrated over the window, its energy difference between this run and '
            'one ending 10 periods earlier per window time and strip height) vs incident - transmitted intensity '
            '(fundamental + third harmonic of the probe fits). Tolerance 3%: third-harmonic currents, the discrete '
            'flux / amplitude relation (< 1%), the ramp residual of the 36-period run (intensity ~5e-3 low at the '
            'window start). 36 periods not yet confirmed by a run.'),
    'carriers': ('free_carrier_absorption', 'TPA free-carrier absorption and dispersion',
                 'beta as tpa, carrier lifetime 50 fs, FCA cross-section 2.9e-21 m^2, dn / dN = -6e-28 m^3: steady N = '
                 'tau beta I^2 / (2 hbar omega) ~ 1.7e25 m^-3 adds alpha = sigma N ~ 5e4 1/m and dn ~ -0.01; expected '
                 'from dI/dz = -beta I^2 - sigma N I and phi = k0 int(sqrt(n0^2 + 2 n0 (-k N)) - n0) dz; error = the '
                 'larger relative error of the absorbed fraction and of the phase. Tolerance 5%: the carrier ripple at '
                 '2 omega (~0.7% of N at 2 omega tau = 190) beating with the field, the first-order index dn = -k N; '
                 '350 fs leave ~6.5 lifetimes after the transit and ramp before the window (N residual exp(-5.5) ~ '
                 '4e-3). 350 fs not yet confirmed by a run.'),
    'gain': ('gain_small_signal', 'four-level small-signal gain',
             'gain_medium at line centre (lambda_a = 1 um, gamma = 1e14 rad/s, tau_21 = 1 ns, density 4e22 m^-3, half '
             'of it initially in level 2, no pump): chi = kappa (N1 - N2) / (eps0 (omega_a^2 - omega^2 + j gamma '
             'omega)), kappa = 6 pi eps0 c^3 / (omega_a^2 tau_21), gives T = exp(2 k0 L Im sqrt(1 + chi)) ~ 1.21; error '
             '= relative error of ln T. Tolerance 3%: the leap-frog Lorentz line (O((omega dt)^2), < 1% of chi), the '
             'discrete dk / dn (0.2%), the polarisation transient (2 / gamma = 20 fs, decayed long before the fit '
             'window); the inversion drifts by < 2e-4 over the run (tau_21, stimulated emission at 1 V/m).'),
}
_SHG_TEXT = ('d_eff = 5e-12 m/V (chi2 = 2 d_eff): the P = eps0 chi2 E^2 of the integer-time E samples drives P(2 '
             'omega) = eps0 d_eff E1^2 exactly, so the second harmonic at the probe is the 3-point lattice Green\'s '
             'function sum |E_2w| = (W_2w / c)^2 d_eff h^2 |E1|^2 |sum_m w_m exp(j (K_2w - 2 k_w) m h)| / (2 sin(K_2w '
             'h)) '
             'over the chi2 sites m (E_z cells; H_z: E_y faces, end and domain-wall faces at the face mean), with the '
             'space-time discrete wavenumbers sin(K h / 2) = n0 W h / (2 c), W = 2 sin(omega dt / 2) / dt (the FDTD '
             'numerical dispersion, so the slab is not exactly phase matched) and |E1| of the reference run; continuum '
             'limit omega d_eff |E1|^2 L / (n0 c) ~ 0.02 |E1|. error = relative error of |E_2w| (fitted at 2 omega). '
             'Tolerance 1% (provisional, not yet run): the undepleted reference misses the pump depletion and the '
             'cascaded third harmonic (both ~(E_2w / E_w)^2 ~ 4e-4) and the CPML reflection at 2 omega.')
_CASE_TEXT['shg'] = ('chi2_second_harmonic', 'chi2 second-harmonic generation', _SHG_TEXT)
_CASE_TEXT['shg_poled'] = ('chi2_poled_second_harmonic', 'chi2 second harmonic of a poled slab',
                           f'poling sign({_POLING_WALL:g} - x): the last quarter of the slab is inverted (a domain wall '
                           'on a cell edge), so its sites enter the sum with -1 (about half the unpoled amplitude). '
                           + _SHG_TEXT)

TEMPLATE_LABEL = 'Nonlinear silicon slab'
TEMPLATE_DESCRIPTION = ('EM (E_z): a 1.55 um gaussian pulse of 1 GV/m peak crosses a 2 um silicon slab with Kerr (n2 '
                        '6e-18 m^2/W), two-photon absorption (beta 8e-12 m/W) and TPA free carriers (lifetime 1 ns, '
                        'FCA 1.45e-21 m^2, dn/dN -1.35e-27 m^3) in a y-periodic strip; transmitted flux spectrum, '
                        'probe trace and the carrier_density map.')


def _strip_spec(polarization, title, case, with_medium, early=False):
    """Strip of background index n0 with the cw line source and the probe; with_medium adds the slab with the case's
    feature (tpa: plus track_absorption and the absorbed-power integral); early ends _FIT_PERIODS periods sooner."""
    feature_type, medium_values, duration, _, _ = _CASES[case]
    if early:
        duration -= _FIT_PERIODS * _WAVELENGTH / C0
    size_x, size_y = _SIZE
    index = _BACKGROUND_INDEX[case]
    amplitude = _GAIN_AMPLITUDE if case == 'gain' else _AMPLITUDE / math.sqrt(index)
    geometry = [vertical_line('source_line', _SOURCE_X, 0.0, size_y),
                GeometrySpec('probe_point', 'point', {'x': _PROBE_POINT[0], 'y': _PROBE_POINT[1]})]
    features = [FeatureSpec('source', 'line_source', ['source_line'],
                            {'waveform': 'cw', 'wavelength': _WAVELENGTH, 'amplitude': amplitude}),
                FeatureSpec(_PROBE_FEATURE, 'field_probe', ['probe_point'], {'component': _PROBED[polarization]})]
    settings = {**PERIODIC_Y, 'pml_sigma_factor': 1.0 / index}   # the CPML optimum ~1 / n of the cladding
    derived_values = []
    if with_medium:
        geometry.append(GeometrySpec('slab', 'rectangle', {'x0': _SLAB[0], 'y0': 0.0, 'x1': _SLAB[1], 'y1': size_y}))
        features.append(FeatureSpec(_MEDIUM_FEATURE, feature_type, ['slab'], dict(medium_values)))
        if case == 'tpa':
            settings['track_absorption'] = True
            # the only absorber: the window integral is the slab's (hz face losses split to both cells)
            derived_values.append(DerivedValueSpec(_ABSORBED_VALUE, f'{_PHYSICS_NAME}_absorbed_power', [],
                                                   'integral'))
    materials = ([MaterialSpec('background', properties={'eps_r': index ** 2}, selection=['all'])]
                 if index != 1.0 else [])
    return ModelSpec(
        title=title, domain=DomainSpec(size_x=size_x, size_y=size_y), geometry=geometry, materials=materials,
        physics=[PhysicsSpec(_PHYSICS_NAME, _PHYSICS_TYPES[polarization], settings=settings, features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': duration}),
        derived_values=derived_values)


def _probe_phasors(results, harmonics=(1, 3)):
    """Complex amplitudes c_h of the probe trace at h omega for h in harmonics, value = Re(c exp(+j h omega t)) + ...,
    least squares over the last _FIT_PERIODS periods with a constant column; (*c_h, samples)."""
    probe = results.result(_PHYSICS_NAME, _PROBE_FEATURE)
    times = np.asarray(probe['t'], dtype=np.float64)
    values = np.asarray(probe['values'], dtype=np.float64)
    if times.size == 0:
        raise ValueError('the field probe recorded no samples')
    in_window = times >= times[-1] - _FIT_PERIODS * _WAVELENGTH / C0
    phase = 2.0 * math.pi * C0 / _WAVELENGTH * times[in_window]
    columns = np.stack([function(harmonic * phase) for harmonic in harmonics for function in (np.cos, np.sin)]
                       + [np.ones_like(phase)], axis=1)
    coefficients = np.linalg.lstsq(columns, values[in_window], rcond=None)[0]
    return (*(complex(coefficients[2 * k], -coefficients[2 * k + 1]) for k in range(len(harmonics))),
            int(np.count_nonzero(in_window)))


def _probe_time_step(results):
    """Time step [s] of the probe trace (one sample per FDTD step)."""
    times = np.asarray(results.result(_PHYSICS_NAME, _PROBE_FEATURE)['t'], dtype=np.float64)
    return float(np.median(np.diff(times)))


def chi2_site_weights(medium_values, polarization):
    """chi2 sign weight of every E site along x from the first slab site on (E_z: the slab cells; H_z: the E_y faces
    from the slab's lo edge to its hi edge, face mean of the cell signs), poling = sign(_POLING_WALL - x) or none."""
    n_cells = int(round((_SLAB[1] - _SLAB[0]) / _CELL))
    centres = _SLAB[0] + (np.arange(n_cells) + 0.5) * _CELL
    signs = np.where(centres < _POLING_WALL, 1.0, -1.0) if 'poling' in medium_values else np.ones(n_cells)
    if polarization == 'ez':
        return signs
    padded = np.concatenate(([0.0], signs, [0.0]))
    return 0.5 * (padded[:-1] + padded[1:])


def expected_second_harmonic(amplitude, d_eff, index, site_weights, dt):
    """|E_2w| [V/m] behind the slab: 3-point lattice Green's function sum over the chi2 sites (spacing _CELL, weights
    site_weights) of -W^2 mu0 eps0 d_eff E1^2, E1 the discrete pump plane wave of amplitude `amplitude` (module notes)."""
    omega = 2.0 * math.pi * C0 / _WAVELENGTH

    def lattice(angular_frequency):
        """(K, W): space-time discrete wavenumber [1/m] and time-difference frequency [rad/s] in the background."""
        discrete_frequency = 2.0 * math.sin(0.5 * angular_frequency * dt) / dt
        return 2.0 * math.asin(0.5 * discrete_frequency * index * _CELL / C0) / _CELL, discrete_frequency

    pump_wavenumber, _ = lattice(omega)
    harmonic_wavenumber, harmonic_frequency = lattice(2.0 * omega)
    sites = np.arange(np.size(site_weights)) * _CELL
    phase_sum = abs(np.sum(site_weights * np.exp(1j * (harmonic_wavenumber - 2.0 * pump_wavenumber) * sites)))
    return ((harmonic_frequency / C0) ** 2 * d_eff * _CELL ** 2 / (2.0 * math.sin(harmonic_wavenumber * _CELL))
            * amplitude ** 2 * phase_sum)


def _absorbed_energy(results):
    """Energy [J/m] absorbed per unit z up to the end of the run: the window integral of the run-mean absorbed_power
    times the accumulation window (the whole run, t_end)."""
    summary = results.summary()
    return float(summary['derived'][_ABSORBED_VALUE]) * float(summary['t_end'])   # float() also reads 'nan'


def expected_response(intensity, medium_values, length, wavelength, index=1.0):
    """(power transmission, phase delay [rad]) of the cycle-averaged plane-wave z-ODE through a slab of background
    index n0 = index: dI/dz = -beta I^2 - sigma N I, dphi/dz = k0 (sqrt(n0^2 + 2 n0 (n2 I - k N)) - n0), N = tau beta
    I^2 / (2 hbar omega)."""
    n2 = medium_values.get('n2', 0.0)
    beta = medium_values.get('tpa_beta', 0.0)
    lifetime = medium_values.get('carrier_lifetime', 0.0)
    cross_section = medium_values.get('fca_cross_section', 0.0)
    index_coefficient = medium_values.get('fcd_coefficient', 0.0)
    pair = medium_values.get('pump_wavelength', wavelength) / (4.0 * math.pi * HBAR * C0)   # 1 / (2 hbar omega)
    k0 = 2.0 * math.pi / wavelength

    def slope(level):
        density = lifetime * beta * level ** 2 * pair
        return (-beta * level ** 2 - cross_section * density * level,
                k0 * (math.sqrt(index ** 2 + 2.0 * index * (n2 * level - index_coefficient * density)) - index))

    step = length / _ODE_STEPS
    level, phase = intensity, 0.0
    for _ in range(_ODE_STEPS):
        k1 = slope(level)
        k2 = slope(level + 0.5 * step * k1[0])
        k3 = slope(level + 0.5 * step * k2[0])
        k4 = slope(level + step * k3[0])
        level += step / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        phase += step / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    return level / intensity, phase


def expected_gain_response(gain_values, length, wavelength):
    """(power transmission, phase delay [rad]) of an n0 = 1 slab with the small-signal four-level gain line: chi =
    kappa (N1 - N2) / (eps0 (omega_a^2 - omega^2 + j gamma omega)), N1 = 0, N2 = initial_upper_fraction x density."""
    omega_a = 2.0 * math.pi * C0 / gain_values['wavelength']
    omega = 2.0 * math.pi * C0 / wavelength
    kappa = 6.0 * math.pi * EPS0 * C0 ** 3 / (omega_a ** 2 * gain_values['tau_21'])
    difference = -gain_values['initial_upper_fraction'] * gain_values['density']   # N1 - N2
    chi = kappa * difference / (EPS0 * (omega_a ** 2 - omega ** 2 + 1j * gain_values['linewidth'] * omega))
    index_change = complex(np.sqrt(np.complex128(1.0 + chi))) - 1.0
    k0_length = 2.0 * math.pi / wavelength * length
    return math.exp(2.0 * k0_length * index_change.imag), k0_length * index_change.real


def _balance(results_by_name, linear_phasors, medium_phasors, index):
    """(absorbed power [W/m^2] of the last _FIT_PERIODS periods per unit strip height, incident - transmitted
    intensity [W/m^2] of the probe fits at omega and 3 omega, its relative error)."""
    medium, early = results_by_name['medium'], results_by_name['medium_early']
    window = float(medium.summary()['t_end']) - float(early.summary()['t_end'])
    absorbed = (_absorbed_energy(medium) - _absorbed_energy(early)) / window / _SIZE[1]
    intensity_scale = 0.5 * C0 * EPS0 * index
    incident = intensity_scale * sum(abs(amplitude) ** 2 for amplitude in linear_phasors)
    transmitted = intensity_scale * sum(abs(amplitude) ** 2 for amplitude in medium_phasors)
    return absorbed, incident - transmitted, relative_error(absorbed, incident - transmitted)


def _check_second_harmonic(results_by_name, case, polarization):
    """|E_2w| of the medium run's probe fit vs expected_second_harmonic at the reference run's pump amplitude."""
    _, medium_values, _, tolerance, measure = _CASES[case]
    index = _BACKGROUND_INDEX[case]
    linear, _, linear_samples = _probe_phasors(results_by_name['linear'])
    pump, harmonic, third, medium_samples = _probe_phasors(results_by_name['medium'], (1, 2, 3))
    dt = _probe_time_step(results_by_name['medium'])
    weights = chi2_site_weights(medium_values, polarization)
    expected = expected_second_harmonic(abs(linear), medium_values['d_eff'], index, weights, dt)
    measured = abs(harmonic)
    error = relative_error(measured, expected)
    return {'measured': float(measured), 'expected': float(expected),
            'error': float(math.inf if math.isnan(error) else error), 'tolerance': tolerance,
            'details': {'error_measures': measure, 'incident_amplitude': float(abs(linear)),
                        'pump_amplitude_with_medium': float(abs(pump)), 'third_harmonic_amplitude': float(abs(third)),
                        'harmonic_to_pump': float(measured / abs(linear)), 'time_step': dt,
                        'site_weight_sum': float(np.sum(weights)), 'background_index': index,
                        'fit_samples': [linear_samples, medium_samples]}}


def _check(results_by_name, case):
    """Medium / linear probe phasor vs expected_response (nonlinear cases, at the linear run's measured intensity) or
    expected_gain_response (gain); tpa adds the absorbed = incident - transmitted balance."""
    _, medium_values, _, tolerance, measure = _CASES[case]
    index = _BACKGROUND_INDEX[case]
    linear, linear_harmonic, linear_samples = _probe_phasors(results_by_name['linear'])
    with_medium, medium_harmonic, medium_samples = _probe_phasors(results_by_name['medium'])
    ratio = with_medium / linear
    transmission, phase = abs(ratio) ** 2, -math.atan2(ratio.imag, ratio.real)
    intensity = 0.5 * C0 * EPS0 * index * abs(linear) ** 2   # the incident intensity in the slab
    length = _SLAB[1] - _SLAB[0]
    if case == 'gain':
        expected_transmission, expected_phase = expected_gain_response(medium_values, length, _WAVELENGTH)
    else:
        expected_transmission, expected_phase = expected_response(intensity, medium_values, length, _WAVELENGTH,
                                                                  index)
    loss_error = relative_error(1.0 - transmission, 1.0 - expected_transmission)
    phase_error = relative_error(phase, expected_phase)
    balance = {}
    if case == 'kerr':
        measured, expected, error = phase, expected_phase, phase_error
    elif case == 'tpa':
        absorbed, intensity_drop, balance_error = _balance(results_by_name, (linear, linear_harmonic),
                                                           (with_medium, medium_harmonic), index)
        measured, expected, error = transmission, expected_transmission, max(loss_error, balance_error)
        balance = {'absorbed_power': float(absorbed), 'incident_minus_transmitted': float(intensity_drop),
                   'balance_error': float(balance_error)}
    elif case == 'gain':
        measured = math.log(transmission) if transmission > 0 else -math.inf
        expected = math.log(expected_transmission)
        error = relative_error(measured, expected)
    else:
        measured, expected, error = transmission, expected_transmission, max(loss_error, phase_error)
    if math.isnan(error):
        error = math.inf
    return {'measured': float(measured), 'expected': float(expected), 'error': float(error),
            'tolerance': tolerance,
            'details': {'error_measures': measure, 'transmission': float(transmission),
                        'expected_transmission': float(expected_transmission), 'phase_rad': float(phase),
                        'expected_phase_rad': float(expected_phase), 'loss_error': float(loss_error),
                        'phase_error': float(phase_error), 'incident_amplitude': float(abs(linear)),
                        'incident_intensity': float(intensity), 'background_index': index,
                        'third_harmonic_amplitude': float(abs(medium_harmonic)),
                        'fit_samples': [linear_samples, medium_samples], **balance}}


def nonlinear_case(case, polarization):
    """SelfTestCase of `case` ('kerr' | 'tpa' | 'carriers' | 'gain' | 'shg' | 'shg_poled') in polarisation 'ez' |
    'hz'."""
    feature_type = _CASES[case][0]
    name, label, text = _CASE_TEXT[case]
    prefix, field = ('em_', 'E_z') if polarization == 'ez' else ('em_hz_', 'E_y (faces, per component)')
    index = _BACKGROUND_INDEX[case]
    amplitude = (f'{_GAIN_AMPLITUDE:g} V/m' if case == 'gain'
                 else f'{_AMPLITUDE / math.sqrt(index):.3g} V/m, I0 ~ 1.33e15 W/m^2')
    medium = ('vacuum' if index == 1.0
              else f'n0 = {index:g} (eps_r {index ** 2:g}, CPML pml_sigma_factor 1 / n0)')
    compared = ('the second harmonic fitted at 2 omega is compared with the lattice sum' if case in _CHI2_CASES
                else 'the ratio to a reference run without the slab gives sqrt(T) exp(-j phi)')
    check = ((lambda results_by_name: _check_second_harmonic(results_by_name, case, polarization))
             if case in _CHI2_CASES else (lambda results_by_name: _check(results_by_name, case)))

    def specs():
        runs = {'linear': _strip_spec(polarization, f'Self-test: {label} (reference)', case, False),
                'medium': _strip_spec(polarization, f'Self-test: {label}', case, True)}
        if case == 'tpa':
            runs['medium_early'] = _strip_spec(polarization, f'Self-test: {label} (10 periods shorter)', case, True,
                                               early=True)
        return runs
    return SelfTestCase(
        name=f'{prefix}{name}',
        label=f"EM {'' if polarization == 'ez' else 'H_z '}{label}",
        description=f'A cw plane wave ({field}, 1 um, {amplitude}) from a line source spanning a y-periodic strip of '
                    f'{medium} (20 nm cells, CPML along x) crosses a 2 um {feature_type} slab in the same background '
                    f'(no linear reflection); the probe 1 um behind it is fitted over the last 10 periods, and '
                    f'{compared} (pump amplitude from a reference run without the slab). {text}',
        specs=specs,
        check=check,
        heavy=case == 'carriers')   # the carrier runs took >= 5 s at 450 fs


def nonlinear_silicon_slab_spec():
    """Template 'nonlinear_silicon_slab': E_z pulse through a 2 um silicon slab with Kerr, TPA and free carriers."""
    size_x, size_y = 8e-6, 0.1e-6
    return ModelSpec(
        title=TEMPLATE_LABEL, domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[GeometrySpec('source_line', 'line', {'x0': 1.5e-6, 'y0': 0.0, 'x1': 1.5e-6, 'y1': size_y}),
                  GeometrySpec('slab', 'rectangle', {'x0': 3e-6, 'y0': 0.0, 'x1': 5e-6, 'y1': size_y}),
                  GeometrySpec('flux_line', 'line', {'x0': 6.5e-6, 'y0': 0.0, 'x1': 6.5e-6, 'y1': size_y}),
                  GeometrySpec('probe_point', 'point', {'x': 6.01e-6, 'y': 0.05e-6})],
        materials=[MaterialSpec('silicon', properties={'eps_r': 12.1}, selection=['slab'])],
        physics=[PhysicsSpec(_PHYSICS_NAME, 'em_fdtd_ez', settings=dict(PERIODIC_Y), features=[
            FeatureSpec('source', 'line_source', ['source_line'],
                        {'waveform': 'gaussian', 'wavelength': 1.55e-6, 'bandwidth': 0.1, 'amplitude': 1e9}),
            FeatureSpec(_MEDIUM_FEATURE, 'nonlinear_medium', ['slab'],
                        {'n2': 6e-18, 'tpa_beta': 8e-12, 'carrier_lifetime': 1e-9, 'fca_cross_section': 1.45e-21,
                         'fcd_coefficient': 1.35e-27, 'pump_wavelength': 1.55e-6}),
            FeatureSpec('transmitted', 'flux_monitor', ['flux_line'],
                        {'wavelength_start': 1.45e-6, 'wavelength_stop': 1.65e-6, 'n_wavelengths': 41}),
            FeatureSpec(_PROBE_FEATURE, 'field_probe', ['probe_point'], {'component': 'ez'})])],
        mesh=MeshSpec(mode='uniform', dx=20e-9),
        study=StudySpec(type='time_dependent', settings={'duration': 400e-15}))


# (case, polarisation) pairs not registered on their own: selftests_backend em_fdtd_cpu_gpu runs them with their own
# check on both backends (ez gain: also the example em/analytic_gain_threshold_cavity)
_BACKEND_GRADED = frozenset({('gain', 'ez'), ('tpa', 'hz')})
NONLINEAR_SELFTEST_CASES = tuple(nonlinear_case(case, polarization) for polarization in ('ez', 'hz')
                                 for case in _CASES if (case, polarization) not in _BACKEND_GRADED)
