"""Butterworth-Van Dyke fit of a resonator admittance Y(f) (host numpy, scipy inside the fit): C0 with a dielectric loss
tangent in parallel with the motional R1 - L1 - C1 branch, by least squares from a closed-form start."""
# Y(omega) = j omega C0 (1 - j tan_delta_0) + 1 / (R1 + j omega L1 + 1 / (j omega C1)) (exp(+j omega t)). Fitted in the
# well-conditioned set (ln C0, ln r, ln omega_s, ln Q_s, tan_delta_0) with r = C1 / C0, omega_s = 1 / sqrt(L1 C1), Q_s =
# omega_s L1 / R1, on the relative residual (Y_model - Y) / |Y| (real and imaginary parts), so the antiresonance dip
# weighs like the peak. Start: f_s0 = the sample of max |Y|, f_p0 = the sample of min |Y| above it, r0 = (f_p0 / f_s0)^2
# - 1, C0 from the lossless model at the band end farthest from f_s0, R1 from Re Y at f_s0. The lossless parallel
# resonance f_p = f_s sqrt(1 + r) and k_eff^2 = (f_p^2 - f_s^2) / f_p^2 = C1 / (C0 + C1) follow. A resistive load Z in
# series with the motional branch adds Z / L1 to its energy decay rate R1 / L1 (= omega_s / Q_s): kappa_e = Z / L1
# (piezo_3d_plan D8), valid while omega C0 Z << 1.
import math

import numpy as np

FIT_KEYS = ('f_s', 'f_p', 'k_eff2', 'Q_s', 'C0', 'C1', 'L1', 'R1', 'tan_delta_0', 'f_peak', 'f_dip', 'fit_error',
            'fit_points')
_MIN_POINTS = 6
_TOLERANCE = 1e-15      # least_squares ftol / xtol / gtol: the round-trip self-test recovers the parameters exactly
_MAX_EVALUATIONS = 4000
_DEFAULT_Q = 1e3        # start when Re Y at the peak is not positive


def bvd_admittance(frequencies, fit):
    """Complex Y [S] of the BVD model `fit` (bvd_fit keys C0, C1, L1, R1, tan_delta_0) at frequencies [Hz]."""
    omega = 2.0 * math.pi * np.asarray(frequencies, dtype=np.float64)
    motional = fit['R1'] + 1j * omega * fit['L1'] + 1.0 / (1j * omega * fit['C1'])
    return 1j * omega * fit['C0'] * (1.0 - 1j * fit['tan_delta_0']) + 1.0 / motional


def external_coupling(fit, load_impedance):
    """{'kappa_e': Z / L1 [rad/s], 'omega_C0_Z': omega_s C0 Z} of a bvd_fit (C0 / L1 in F / H) loaded by a resistive
    load_impedance Z [Ohm]: the energy decay rate the load adds to the motional branch, valid while omega_C0_Z << 1
    (the load then sees the series RLC, not the C0 shunt); NaN with the fit's NaN."""
    load_impedance = float(load_impedance)
    return {'kappa_e': load_impedance / fit['L1'],
            'omega_C0_Z': 2.0 * math.pi * fit['f_s'] * fit['C0'] * load_impedance}


def _model(parameters, omega):
    log_c0, log_ratio, log_omega_s, log_q, tan_delta = parameters
    c0, omega_s, quality = math.exp(log_c0), math.exp(log_omega_s), math.exp(log_q)
    c1 = math.exp(log_ratio) * c0
    inductance = 1.0 / (omega_s * omega_s * c1)
    resistance = omega_s * inductance / quality
    return 1j * omega * c0 * (1.0 - 1j * tan_delta) + 1.0 / (resistance + 1j * omega * inductance
                                                               + 1.0 / (1j * omega * c1))


def _failed(reason, n_points, f_peak=math.nan, f_dip=math.nan):
    fit = dict.fromkeys(FIT_KEYS, math.nan)
    fit.update(f_peak=f_peak, f_dip=f_dip, fit_points=n_points, fit_warning=reason)
    return fit


def bvd_fit(frequencies, admittance, fit_min=None, fit_max=None):
    """{FIT_KEYS: float} of the BVD model fitted to complex Y [S] at ascending frequencies [Hz] inside [fit_min,
    fit_max] (None = open); NaN values and a 'fit_warning' text when the band holds < 6 points or no peak with a dip
    above it."""
    from scipy.optimize import least_squares
    frequencies = np.asarray(frequencies, dtype=np.float64)
    admittance = np.asarray(admittance, dtype=np.complex128)
    inside = np.isfinite(frequencies) & np.isfinite(admittance) & (frequencies > 0.0)
    if fit_min is not None:
        inside &= frequencies >= fit_min
    if fit_max is not None:
        inside &= frequencies <= fit_max
    frequencies, admittance = frequencies[inside], admittance[inside]
    n_points = int(frequencies.size)
    if n_points < _MIN_POINTS:
        return _failed(f'BVD fit needs >= {_MIN_POINTS} frequencies in the fit band, got {n_points}', n_points)
    magnitude = np.abs(admittance)
    peak = int(np.argmax(magnitude))
    above = np.flatnonzero(frequencies > frequencies[peak])
    if above.size == 0:
        return _failed('BVD fit: the largest |Y| is at the top of the fit band (no antiresonance above it): widen the '
                       'band upward', n_points, float(frequencies[peak]))
    dip = int(above[np.argmin(magnitude[above])])
    f_s0, f_p0 = float(frequencies[peak]), float(frequencies[dip])
    omega = 2.0 * math.pi * frequencies
    ratio = (f_p0 / f_s0) ** 2 - 1.0
    far = int(np.argmax(np.abs(1.0 - (frequencies / f_s0) ** 2)))
    detuning = 1.0 - (frequencies[far] / f_s0) ** 2
    lossless_factor = 1.0 + ratio / detuning   # 0 when the far sample is f_p0 itself
    c0 = admittance[far].imag / omega[far] / lossless_factor if lossless_factor != 0.0 else math.nan
    if not (math.isfinite(c0) and c0 > 0.0):
        c0 = abs(admittance[far].imag / omega[far]) or 1e-12
    omega_s = 2.0 * math.pi * f_s0
    inductance = 1.0 / (omega_s * omega_s * ratio * c0)
    conductance = admittance[peak].real
    quality = omega_s * inductance * conductance if conductance > 0.0 else _DEFAULT_Q
    start = np.array([math.log(c0), math.log(ratio), math.log(omega_s), math.log(max(quality, 1e-3)), 0.0])
    weights = 1.0 / magnitude

    def residual(parameters):
        difference = (_model(parameters, omega) - admittance) * weights
        return np.concatenate((difference.real, difference.imag))

    try:
        solution = least_squares(residual, start, method='lm', ftol=_TOLERANCE, xtol=_TOLERANCE, gtol=_TOLERANCE,
                                 max_nfev=_MAX_EVALUATIONS)
        log_c0, log_ratio, log_omega_s, log_q, tan_delta = (float(value) for value in solution.x)
        c0, ratio, omega_s, quality = math.exp(log_c0), math.exp(log_ratio), math.exp(log_omega_s), math.exp(log_q)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:   # non-finite start / residuals, runaway step
        return _failed(f'BVD fit: least squares failed ({exc})', n_points, f_s0, f_p0)
    c1 = ratio * c0
    inductance = 1.0 / (omega_s * omega_s * c1)
    f_s = omega_s / (2.0 * math.pi)
    fit = {'f_s': f_s, 'f_p': f_s * math.sqrt(1.0 + ratio), 'k_eff2': ratio / (1.0 + ratio), 'Q_s': quality, 'C0': c0,
           'C1': c1, 'L1': inductance, 'R1': omega_s * inductance / quality, 'tan_delta_0': tan_delta,
           'f_peak': f_s0, 'f_dip': f_p0,
           'fit_error': float(np.sqrt(np.mean(residual(solution.x) ** 2) * 2.0)), 'fit_points': n_points}
    if not solution.success:
        fit['fit_warning'] = f'BVD fit: least squares did not converge ({solution.message})'
    return fit
