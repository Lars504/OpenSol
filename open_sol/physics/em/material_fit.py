"""Multi-pole Drude-Lorentz fit of tabulated n, k (A7; host numpy + scipy least squares, validation / build time) in
the exp(+j omega t) convention of dispersion.py: eps = (n - j k)^2, poles with the DrudePole / LorentzPole chi."""
import math
from dataclasses import dataclass

import numpy as np

from ...core.constants import C0

NK_FIT_MODEL = 'nk_fit'
NK_FIT_DEFAULTS = {'n_lorentz': 2, 'wavelength_start': 400e-9, 'wavelength_stop': 800e-9}
MAX_LORENTZ_POLES = 6
MIN_FIT_SAMPLES = 24          # the table rows inside the band are densified (linear n, k) up to this count
LOSS_SCALE_FLOOR = 0.03       # loss residual / loss_error scale: |Im eps| + 0.03 |eps|
ERROR_WARNING_LEVEL = 0.02    # relative_error above this (or loss_error above LOSS_WARNING_LEVEL) -> warning
LOSS_WARNING_LEVEL = 0.1
_PRUNE_FRACTION = 1e-6        # a pole whose largest |chi| in the band is below this fraction of max |eps| is dropped
# normalised bounds (frequencies / omega_ref): eps_inf >= 1 keeps the FDTD background stable, all values >= 0 passive
_EPS_INF_BOUNDS = (1.0, 1e3)
_DRUDE_BOUNDS = ((0.0, 1e3), (0.0, 1e2))                      # omega_p, gamma
_LORENTZ_BOUNDS = ((0.0, 1e4), (1e-3, None), (0.0, 1e2))      # delta_eps, omega_0, gamma
_MAX_RESONANCE = 5.0   # omega_0 <= 5 x the highest fitted frequency: above it a pole is a constant, and a far-UV
                       # omega_0 would tighten the FDTD Lorentz Courant term delta_eps (w0 dt)^2 / (4 - (w0 dt)^2)


@dataclass(frozen=True, eq=False)
class PoleFit:
    """Fitted eps_inf + optional Drude (omega_p, gamma) + Lorentz (delta_eps, omega_0, gamma) poles [rad/s] and the
    fit errors over its samples (relative_error = max |d eps| / |eps|, loss_error = max |d Im eps| / loss scale)."""
    eps_inf: float
    drude: tuple | None
    lorentz: tuple
    sample_wavelengths: np.ndarray
    relative_error: float
    loss_error: float

    def pole_dicts(self):
        """Model dispersion dicts (builtin floats): the Drude pole first (carrying eps_inf), then the Lorentz poles."""
        poles = []
        if self.drude is not None:
            poles.append({'model': 'drude', 'omega_p': self.drude[0], 'gamma': self.drude[1],
                          'eps_inf': self.eps_inf})
        poles += [{'model': 'lorentz', 'delta_eps': delta_eps, 'omega_0': omega_0, 'gamma': gamma}
                  for delta_eps, omega_0, gamma in self.lorentz]
        return poles

    def summary_text(self):
        """'Drude + 2 Lorentz poles, eps_inf 1.23: max |eps error| 0.81 %, max loss error 3.1 % (41 samples)'."""
        terms = (['Drude'] if self.drude is not None else []) + (
            [f"{len(self.lorentz)} Lorentz"] if self.lorentz else [])
        return (f"{' + '.join(terms) or 'no'} pole{'s' if len(self.lorentz) + (self.drude is not None) != 1 else ''}, "
                f"eps_inf {self.eps_inf:.4g}: max |eps error| {100.0 * self.relative_error:.3g} %, max loss error "
                f"{100.0 * self.loss_error:.3g} % ({self.sample_wavelengths.size} samples)")


def nk_permittivity(n, k):
    """Relative permittivity (n - j k)^2 of index n and extinction k >= 0 (lossy Im < 0), complex128."""
    return (np.asarray(n, dtype=np.float64) - 1j * np.asarray(k, dtype=np.float64)) ** 2


def drude_lorentz_permittivity(omega, eps_inf, drude, lorentz):
    """eps_inf - omega_p^2 / (omega^2 - j gamma omega) + sum delta_eps omega_0^2 / (omega_0^2 - omega^2 + j gamma
    omega) (the DrudePole / LorentzPole chi); omega [rad/s] array, drude None = no Drude term."""
    omega = np.asarray(omega, dtype=np.float64)
    permittivity = np.full(omega.shape, eps_inf, dtype=np.complex128)
    if drude is not None:
        omega_p, gamma = drude
        permittivity -= omega_p ** 2 / (omega ** 2 - 1j * gamma * omega)
    for delta_eps, omega_0, gamma in lorentz:
        permittivity += delta_eps * omega_0 ** 2 / (omega_0 ** 2 - omega ** 2 + 1j * gamma * omega)
    return permittivity


def nk_fit_settings(entry):
    """(n_lorentz, wavelength_start, wavelength_stop) of an nk_fit dispersion dict (defaults filled); ValueError for
    unknown keys, values that are not plain numbers, n_lorentz outside 0..MAX_LORENTZ_POLES or start >= stop."""
    unknown = sorted(str(key) for key in entry if key != 'model' and key not in NK_FIT_DEFAULTS)
    if unknown:
        raise ValueError(f"nk_fit takes {', '.join(NK_FIT_DEFAULTS)}; unknown {unknown}")
    values = {}
    for key, default in NK_FIT_DEFAULTS.items():
        raw = entry.get(key, default)
        try:
            values[key] = float(raw if raw is not None else default)
        except (TypeError, ValueError):
            raise ValueError(f'nk_fit {key} must be a plain number (no parameter expressions: the fit runs '
                             f'once per material), got {raw!r}') from None
    n_lorentz = values['n_lorentz']
    if not (n_lorentz == int(n_lorentz) and 0 <= n_lorentz <= MAX_LORENTZ_POLES):
        raise ValueError(f'nk_fit n_lorentz must be an integer 0..{MAX_LORENTZ_POLES}, got {n_lorentz:g}')
    start, stop = values['wavelength_start'], values['wavelength_stop']
    if not (math.isfinite(start) and math.isfinite(stop) and 0.0 < start < stop):
        raise ValueError(f'nk_fit needs 0 < wavelength_start < wavelength_stop [m], got {start:g} and {stop:g}')
    return int(n_lorentz), start, stop


def table_samples(table_wavelengths, table_n, table_k, wavelength_start, wavelength_stop):
    """(wavelengths [m], eps) inside [start, stop]: the table rows plus linearly interpolated n, k at both ends,
    densified to MIN_FIT_SAMPLES; ValueError when the range leaves the tabulated one."""
    wavelengths = np.asarray(table_wavelengths, dtype=np.float64)
    lowest, highest = float(wavelengths[0]), float(wavelengths[-1])
    if wavelength_start < lowest * (1.0 - 1e-9) or wavelength_stop > highest * (1.0 + 1e-9):
        raise ValueError(f'fit range {wavelength_start * 1e9:.4g}-{wavelength_stop * 1e9:.4g} nm leaves the '
                         f'tabulated {lowest * 1e9:.4g}-{highest * 1e9:.4g} nm')
    start, stop = max(wavelength_start, lowest), min(wavelength_stop, highest)
    inside = wavelengths[(wavelengths > start) & (wavelengths < stop)]
    samples = np.unique(np.concatenate(([start], inside, [stop])))
    if samples.size < MIN_FIT_SAMPLES:
        samples = np.unique(np.concatenate((samples, np.linspace(start, stop, MIN_FIT_SAMPLES))))
    return samples, nk_permittivity(np.interp(samples, wavelengths, table_n), np.interp(samples, wavelengths, table_k))


def fit_drude_lorentz(wavelengths, permittivity, n_lorentz, drude=None):
    """PoleFit of eps_inf (>= 1) + optional Drude + n_lorentz Lorentz poles (all values >= 0: passive) to complex eps
    samples at vacuum wavelengths [m]: bounded least squares from several starts; drude None = when min Re eps < 0."""
    try:
        import scipy.optimize
    except ImportError:
        raise ValueError('the nk_fit dispersion model needs scipy (pip install scipy)') from None
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    target = np.asarray(permittivity, dtype=np.complex128)
    if drude is None:
        drude = bool(np.min(target.real) < 0.0)
    omega = 2.0 * math.pi * C0 / wavelengths
    omega_ref = math.sqrt(float(omega.min()) * float(omega.max()))
    x = omega / omega_ref
    scale = np.abs(target)
    loss_scale = np.abs(target.imag) + LOSS_SCALE_FLOOR * scale

    def unpack(params):
        eps_inf, position = params[0], 1
        drude_terms = None
        if drude:
            drude_terms, position = (params[1], params[2]), 3
        lorentz_terms = tuple(tuple(params[position + 3 * index: position + 3 * index + 3])
                              for index in range(n_lorentz))
        return eps_inf, drude_terms, lorentz_terms

    def residuals(params):
        difference = drude_lorentz_permittivity(x, *unpack(params)) - target
        return np.concatenate((difference.real / scale, difference.imag / loss_scale))

    def jacobian(params):
        """Closed-form d residuals / d params (columns in the params order), the real / imaginary rows as residuals."""
        _, drude_terms, lorentz_terms = unpack(params)
        columns = [np.ones(x.shape, dtype=np.complex128)]             # d eps / d eps_inf
        if drude_terms is not None:
            omega_p, gamma = drude_terms
            denominator = x ** 2 - 1j * gamma * x                       # eps -= omega_p^2 / denominator
            columns += [-2.0 * omega_p / denominator, -1j * x * omega_p ** 2 / denominator ** 2]
        for delta_eps, omega_0, gamma in lorentz_terms:
            denominator = omega_0 ** 2 - x ** 2 + 1j * gamma * x        # eps += delta_eps omega_0^2 / denominator
            columns += [omega_0 ** 2 / denominator,
                        2.0 * delta_eps * omega_0 * (1j * gamma * x - x ** 2) / denominator ** 2,
                        -1j * x * delta_eps * omega_0 ** 2 / denominator ** 2]
        derivative = np.stack(columns, axis=1)
        return np.concatenate((derivative.real / scale[:, None], derivative.imag / loss_scale[:, None]))

    lorentz_bounds = [_LORENTZ_BOUNDS[0], (_LORENTZ_BOUNDS[1][0], _MAX_RESONANCE * float(x.max())), _LORENTZ_BOUNDS[2]]
    bounds = [_EPS_INF_BOUNDS] + (list(_DRUDE_BOUNDS) if drude else []) + lorentz_bounds * n_lorentz
    lower, upper = np.array([bound[0] for bound in bounds]), np.array([bound[1] for bound in bounds])
    best = None
    for start in _initial_guesses(x, target, n_lorentz, drude):
        start = np.clip(start, lower + 1e-9 * (upper - lower), upper - 1e-9 * (upper - lower))
        result = scipy.optimize.least_squares(residuals, start, jac=jacobian, bounds=(lower, upper), method='trf',
                                              x_scale='jac',
                                              ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=60 * len(start))
        if best is None or result.cost < best.cost:
            best = result
    eps_inf, drude_terms, lorentz_terms = unpack(best.x)
    drude_terms = None if drude_terms is None else (drude_terms[0] * omega_ref, drude_terms[1] * omega_ref)
    lorentz_terms = tuple((delta_eps, omega_0 * omega_ref, gamma * omega_ref)
                          for delta_eps, omega_0, gamma in lorentz_terms)
    return scored_fit(eps_inf, *_pruned(drude_terms, lorentz_terms, omega, float(scale.max())), wavelengths, target)


def scored_fit(eps_inf, drude, lorentz, wavelengths, target):
    """PoleFit of float32-rounded pole values (as DrudePole / LorentzPole store them) with its errors against target."""
    eps_inf = _float32(eps_inf)
    drude = None if drude is None else tuple(_float32(value) for value in drude)
    lorentz = tuple(tuple(_float32(value) for value in pole) for pole in lorentz)
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    target = np.asarray(target, dtype=np.complex128)
    fitted = drude_lorentz_permittivity(2.0 * math.pi * C0 / wavelengths, eps_inf, drude, lorentz)
    scale = np.abs(target)
    relative_error = float(np.max(np.abs(fitted - target) / scale))
    loss_error = float(np.max(np.abs(fitted.imag - target.imag) / (np.abs(target.imag) + LOSS_SCALE_FLOOR * scale)))
    return PoleFit(eps_inf, drude, lorentz, wavelengths, relative_error, loss_error)


def _float32(value):
    """Builtin float of value rounded to float32."""
    return float(np.float32(value))


def _pruned(drude, lorentz, omega, largest_permittivity):
    """(drude, lorentz) without poles whose largest |chi| over omega is below _PRUNE_FRACTION of max |eps| (no
    FDTD pole arrays for them)."""
    threshold = _PRUNE_FRACTION * largest_permittivity
    if drude is not None and np.max(np.abs(drude_lorentz_permittivity(omega, 0.0, drude, ()))) < threshold:
        drude = None
    lorentz = tuple(pole for pole in lorentz
                    if np.max(np.abs(drude_lorentz_permittivity(omega, 0.0, None, (pole,)))) >= threshold)
    return drude, lorentz


def _initial_guesses(x, target, n_lorentz, drude):
    """Normalised start vectors: Drude from the lowest-frequency sample, Lorentz centres in / above the band with
    narrow and wide damping."""
    eps_inf = 1.5
    starts_drude = []
    if drude:
        lowest = int(np.argmin(x))
        x_low, eps_low = float(x[lowest]), complex(target[lowest])
        omega_p = x_low * math.sqrt(max(eps_inf - eps_low.real, 0.1))
        gamma = min(max(-eps_low.imag * x_low ** 3 / omega_p ** 2, 1e-3), 1.0)
        starts_drude = [omega_p, gamma]
    if n_lorentz == 0:
        return [np.array([eps_inf] + starts_drude)]
    x_min, x_max = float(x.min()), float(x.max())
    strength = 1.0 if drude else max(0.1, (float(np.mean(target.real)) - eps_inf) / n_lorentz)
    layouts = (np.geomspace(0.8 * x_min, 1.25 * x_max, n_lorentz),          # across the band
               np.geomspace(1.2 * x_max, 3.0 * x_max, n_lorentz),           # above it (UV background)
               np.concatenate((np.geomspace(0.8 * x_min, 1.25 * x_max, (n_lorentz + 1) // 2),
                               np.geomspace(1.5 * x_max, 3.0 * x_max, n_lorentz // 2))))
    starts = []
    for centres in layouts:
        for width in (0.05, 0.3):
            lorentz = [value for centre in centres for value in (strength, centre, width * centre)]
            starts.append(np.array([eps_inf] + starts_drude + lorentz))
    return starts
