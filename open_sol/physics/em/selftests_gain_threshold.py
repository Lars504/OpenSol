"""C5 gain-threshold self-test of the gain_medium feature (em_fdtd_ez): a lossy PEC-walled cavity filled with the
four-level gain medium rings down after a pulse; the resonance_monitor decay rates of a cold and a half-inverted run
give the lasing threshold inversion by extrapolation, compared with the root of the complex mode frequency. CPU check,
GPU runs; specs built fresh on each call."""
import cmath
import math

from ...core.constants import C0, EPS0
from ...model.schema import SelfTestCase
from ...model.selftest_helpers import vertical_line
from ...model.spec import (DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec, PhysicsSpec,
                           StudySpec)

_EM, _MONITOR = 'em', 'ring'
# Cavity: PEC walls on x (E_z = 0 at the outermost centres 25 nm / 2975 nm, L = 59 cells), y-periodic, eps 1 + sigma_e
# everywhere, so the uniformly filled cavity's modes obey W^2 eps(omega) = c^2 K^2 whatever their shape (W, eps the
# leap-frog forms of _dispersion: 0.22 rad per step and 2.8e13 rad/s detuning shift the stepped line ~7 %); mode 6
# sits near 0.99 um, its antinodes at (2 m + 1) L / 12 = 4.92 (2 m + 1) cells from the wall centre.
_SIZE = (3.0e-6, 0.1e-6)       # m
_CELL = 50e-9                  # m (20 cells per wavelength)
_SIGMA = 50.0                  # S/m: cold amplitude decay sigma / (2 eps0) = 2.8e12 1/s (Q ~ 340)
_LINE_WAVELENGTH = 1e-6        # m, gain line centre
_LINEWIDTH = 1e14              # rad/s
_TAU_21 = 1e-9                 # s
_THRESHOLD_FRACTION = 0.5      # inversion of the gain run / the line-centre threshold sigma gamma / kappa
_SOURCE_X = 1.275e-6           # m: centre of cell 25, 25 cells from the wall centre (antinode m = 2 at 24.6)
_PROBE = (0.775e-6, 0.05e-6)   # m: centre of cell 15, 15 cells from the wall centre (antinode m = 1 at 14.75)
_BAND = (0.93e-6, 1.08e-6)     # m: mode 6 only (modes 5 / 7 near 1.19 / 0.85 um)
_DURATION = 0.8e-12            # s: ~240 periods; cold field down to exp(-2.2)
_TOLERANCE = 0.03
_COURANT = 0.99


def _time_step():
    """The engine's dt [s] on the square cells."""
    from ...core.grid import courant_time_step   # inside: selftests import core.constants only at module level
    return float(courant_time_step(_CELL, _CELL, _COURANT))


def _coupling():
    """kappa = 6 pi eps0 c^3 / (omega_a^2 tau_21) of the gain polarisation [C^2 / kg]."""
    omega_a = 2.0 * math.pi * C0 / _LINE_WAVELENGTH
    return 6.0 * math.pi * EPS0 * C0 ** 3 / (omega_a ** 2 * _TAU_21)


def _run_inversion():
    """N2 - N1 of the gain run [1/m^3]: _THRESHOLD_FRACTION of the line-centre threshold."""
    return _THRESHOLD_FRACTION * _SIGMA * _LINEWIDTH / _coupling()


def _cavity_spec(with_gain):
    """Lossy PEC cavity with a gaussian line-source pulse and the resonance monitor; with_gain fills it with the gain
    medium at _run_inversion (half the density in level 2, no pump)."""
    size_x, size_y = _SIZE
    features = [FeatureSpec('source', 'line_source', ['source_line'],
                            {'waveform': 'gaussian', 'wavelength': _LINE_WAVELENGTH, 'bandwidth': 0.2,
                             'amplitude': 1.0}),
                FeatureSpec(_MONITOR, 'resonance_monitor', ['probe_point'],
                            {'component': 'ez', 'wavelength_min': _BAND[0], 'wavelength_max': _BAND[1]})]
    if with_gain:
        features.append(FeatureSpec('gain', 'gain_medium', ['all'], {
            'wavelength': _LINE_WAVELENGTH, 'linewidth': _LINEWIDTH, 'density': 2.0 * _run_inversion(),
            'tau_21': _TAU_21, 'tau_32': 1e-13, 'tau_10': 1e-13, 'initial_upper_fraction': 0.5}))
    return ModelSpec(
        title=f"Self-test: gain threshold ({'half inverted' if with_gain else 'cold'} cavity)",
        domain=DomainSpec(size_x=size_x, size_y=size_y),
        geometry=[vertical_line('source_line', _SOURCE_X, 0.0, size_y),
                  GeometrySpec('probe_point', 'point', {'x': _PROBE[0], 'y': _PROBE[1]})],
        materials=[MaterialSpec('lossy', properties={'eps_r': 1.0, 'sigma_e': _SIGMA}, selection=['all'])],
        physics=[PhysicsSpec(_EM, 'em_fdtd_ez', settings={'bc_x_lo': 'pec', 'bc_x_hi': 'pec', 'bc_y_lo': 'periodic',
                                                          'bc_y_hi': 'periodic', 'courant_factor': _COURANT},
                             features=features)],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        study=StudySpec(type='time_dependent', settings={'duration': _DURATION}))


def _dispersion(omega, inversion):
    """Leap-frog W^2 eps, W = (2 / dt) sin(omega dt / 2), eps = 1 + chi_gain - j sigma cos(omega dt / 2) / (W eps0),
    chi_gain = kappa (N1 - N2) / (eps0 (omega_a^2 - W^2 + j gamma sin(omega dt) / dt)) (exp(+j omega t))."""
    omega_a = 2.0 * math.pi * C0 / _LINE_WAVELENGTH
    time_step = _time_step()
    half_phase = 0.5 * omega * time_step
    stepped = 2.0 / time_step * cmath.sin(half_phase)
    stepped_damping = _LINEWIDTH * cmath.sin(omega * time_step) / time_step
    chi = _coupling() * (-inversion) / (EPS0 * (omega_a ** 2 - stepped ** 2 + 1j * stepped_damping))
    return stepped ** 2 * (1.0 + chi) - 1j * stepped * cmath.cos(half_phase) * _SIGMA / EPS0


def _mode_frequency(squared_wavenumber, inversion, guess):
    """Complex omega (Im > 0 decays) with _dispersion(omega) = c^2 K^2, Newton from guess."""
    omega = complex(guess)
    for _ in range(60):
        step = 1e-7 * abs(omega)
        derivative = (_dispersion(omega + step, inversion) - _dispersion(omega - step, inversion)) / (2.0 * step)
        correction = (_dispersion(omega, inversion) - squared_wavenumber) / derivative
        omega -= correction
        if abs(correction) < 1e-14 * abs(omega):
            break
    return omega


def _threshold(squared_wavenumber, guess):
    """Inversion [1/m^3] where the mode stops decaying (Im omega = 0), secant from 0 and the line-centre value."""
    low, high = 0.0, _SIGMA * _LINEWIDTH / _coupling()
    decay_low = _mode_frequency(squared_wavenumber, low, guess).imag
    decay_high = _mode_frequency(squared_wavenumber, high, guess).imag
    for _ in range(40):
        if decay_high == decay_low:
            break
        low, high, decay_low = high, high - decay_high * (high - low) / (decay_high - decay_low), decay_high
        decay_high = _mode_frequency(squared_wavenumber, high, guess).imag
        if abs(high - low) < 1e-12 * abs(high):
            break
    return high


def _strongest_pole(results):
    """Complex omega = 2 pi (frequency + j frequency_imag) of the strongest resonance in the band, None without one."""
    table = results.result(_EM, _MONITOR)
    if not table['rows']:
        return None, table.get('note', '')
    columns = table['columns']
    row = table['rows'][0]
    return 2.0 * math.pi * complex(row[columns.index('frequency')], row[columns.index('frequency_imag')]), ''


def _check(results_by_name):
    """Cold pole -> c^2 K^2; predicted gain-run decay and threshold vs the measured decay and its linear extrapolation."""
    cold, cold_note = _strongest_pole(results_by_name['cold'])
    gain, gain_note = _strongest_pole(results_by_name['gain'])
    inversion = _run_inversion()
    if cold is None or gain is None:
        return {'measured': math.nan, 'expected': math.nan, 'error': math.inf, 'tolerance': _TOLERANCE,
                'details': {'cold_note': cold_note, 'gain_note': gain_note}}
    squared_wavenumber = _dispersion(cold, 0.0)
    predicted = _mode_frequency(squared_wavenumber, inversion, cold)
    expected_threshold = _threshold(squared_wavenumber, cold)
    measured_threshold = (inversion * cold.imag / (cold.imag - gain.imag) if cold.imag != gain.imag else math.inf)
    errors = {'gain_decay_change': abs(gain.imag - predicted.imag) / abs(cold.imag - predicted.imag),
              'threshold': abs(measured_threshold / expected_threshold - 1.0)}
    error = max(errors.values())
    return {'measured': measured_threshold, 'expected': expected_threshold,
            'error': error if math.isfinite(error) else math.inf, 'tolerance': _TOLERANCE,
            'details': {'errors': errors, 'run_inversion': inversion,
                        'line_centre_threshold': _SIGMA * _LINEWIDTH / _coupling(),
                        'cold_decay_rate': cold.imag, 'gain_decay_rate': gain.imag,
                        'predicted_gain_decay_rate': predicted.imag, 'cold_wavelength': 2.0 * math.pi * C0 / cold.real,
                        'gain_wavelength': 2.0 * math.pi * C0 / gain.real,
                        'predicted_gain_wavelength': 2.0 * math.pi * C0 / predicted.real,
                        'sigma_decay_rate': _SIGMA / (2.0 * EPS0), 'dt_s': _time_step(),
                        'cold_wavenumber_imag_ratio': squared_wavenumber.imag / squared_wavenumber.real}}


GAIN_THRESHOLD_SELFTEST_CASES = (
    SelfTestCase(
        name='em_gain_threshold', label='EM four-level gain: cavity lasing threshold',
        description='em_fdtd_ez cavity between PEC walls on x (3 um, y-periodic, 50 nm cells) filled with eps 1, '
                    f'sigma_e {_SIGMA:g} S/m: a gaussian line-source pulse at 1 um excites mode 6 (~0.99 um), whose '
                    'ring-down the resonance_monitor (0.93-1.08 um) fits. Runs: cold, and with a gain_medium on every '
                    'cell (line centre 1 um, gamma 1e14 rad/s, tau_21 1 ns, half the density in level 2, no pump) at '
                    'half the line-centre threshold inversion sigma gamma / kappa. A uniformly filled cavity keeps '
                    'W^2 eps(omega) = c^2 K^2 for any mode shape, in the leap-frog forms W = (2 / dt) sin(omega dt / '
                    '2), conductivity sigma cos(omega dt / 2), chi_gain = kappa (N1 - N2) / (eps0 (omega_a^2 - W^2 + j '
                    'gamma sin(omega dt) / dt)): near the line the stepping error of chi is ~(omega dt)^2 omega / (12 '
                    'gamma) ~ 7 %, not (omega dt)^2, so the continuous Lorentzian is no reference. The cold pole gives '
                    'c^2 K^2 (absorbing the discrete wavenumber), and the gain run is predicted by the root. Measured '
                    'threshold = run inversion x cold decay / (cold - gain decay) (Im omega is linear in N2 - N1 to '
                    'O(chi^2) ~ 1e-5), expected = the inversion where the root stops decaying. error = the larger of '
                    'the relative threshold error and the error of the gain-induced decay change; tolerance 3 %: the '
                    'pencil fit (~1e-4; mode 6 alone in the band), the population drift over the 0.8 ps run (N2 '
                    'decays by ~5e-4 through tau_21).',
        specs=lambda: {'cold': _cavity_spec(False), 'gain': _cavity_spec(True)},
        check=_check,
        heavy=True),
)
