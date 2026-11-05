"""Analytic comparison and post-processing on the CPU (numpy only, GUI-thread safe): derived-value references, derived
values of 1D results (windows, fits, far field), analytic and derived curves (calculus, references), curve errors."""
import math
import numbers
import re

import numpy as np

from .expressions import ExpressionError, compile_expression, error_message, evaluate_parameters, evaluate_scalar

DERIVED_RESULT_KINDS = ('timeseries', 'flux_spectrum', 'far_field', 'xy', 'table')   # DerivedValueSpec.source kinds
CURVE_RESULT_KINDS = ('timeseries', 'flux_spectrum', 'far_field', 'xy', 'table')   # kinds of AnalyticCurveSpec targets
CURVE_TARGETS = ('sweep', 'line')   # AnalyticCurveSpec targets besides 'physics.feature'
RATIO_SERIES = 'ratio'              # series of a spectral curve drawn on the Spectra ratio-to-reference plot
FFT_SERIES = 'fft'                  # series of a timeseries curve drawn on the probe FFT plot (variable frequency)
# result kind -> (abscissa name, sample names known before the run or None: taken from the result)
RESULT_SAMPLE_NAMES = {'timeseries': ('t', ('t', 'values')),
                       'flux_spectrum': ('wavelength', ('wavelength', 'frequency', 'power', 'power_per_cw')),
                       'far_field': ('angle', ('angle', 'power', 'power_norm', 'power_per_cw', 'wavelength',
                                               'frequency')),
                       'xy': ('x', None), 'table': ('row', None)}
FAR_FIELD_PERIOD = 2.0 * math.pi    # far_field angles are periodic: 'at', windows and peaks wrap around
# fit operation -> its fitted quantities (DerivedValueSpec.fit); polyfit: 'c0' .. 'c<degree>' (coefficient of x**k)
FIT_QUANTITIES = {'polyfit': None, 'exp_fit': ('amplitude', 'rate', 'decay'),
                  'lorentz_fit': ('center', 'fwhm', 'amplitude', 'offset', 'q'),
                  'two_exp_fit': ('kappa', 'decay_length', 'amplitude_start', 'amplitude_end', 'r_squared'),
                  'two_wave_fit': ('beta', 'wavelength', 'forward_abs', 'forward_phase', 'backward_abs',
                                   'backward_phase', 'reflection_abs', 'r_squared'),
                  'richardson': ('extrapolated', 'order', 'coefficient', 'error_estimate', 'relative_error_estimate'),
                  'multi_lorentz_fit': None, 'pole_fit': None}
MAX_FIT_DEGREE = 10
# peak fits (degree = number of peaks / poles): (per-peak quantities '<name>_<k>', k = 1 .. degree ordered by centre,
# shared quantities after them; 'spacing' only for degree >= 2)
PEAK_FIT_QUANTITIES = {'multi_lorentz_fit': (('center', 'fwhm', 'amplitude', 'q'), ('offset', 'spacing', 'r_squared')),
                       'pole_fit': (('frequency', 'q', 'fwhm', 'residue_abs'), ('spacing', 'r_squared'))}
DEGREE_FIT_OPERATIONS = ('polyfit',) + tuple(PEAK_FIT_QUANTITIES)   # fit operations that read DerivedValueSpec.degree
# complex fits: the expression names a '<name>_re' series, paired with '<name>_im' of the same result (the series
# convention of core.harmonic_probe; analytic imports no core module)
COMPLEX_FIT_OPERATIONS = ('two_wave_fit', 'pole_fit')
REAL_PART_SUFFIX, IMAGINARY_PART_SUFFIX = '_re', '_im'
# curve target (result kind, 'line' or 'sweep') -> variables of its expression besides the parameters
CURVE_VARIABLES = {'timeseries': ('t', 'iteration'), 'flux_spectrum': ('wavelength',),
                   'far_field': ('angle', 'wavelength'), 'xy': ('x', 'wavelength'), 'table': ('x',),
                   'line': ('x', 'y', 'z', 's'), 'sweep': ()}   # line z: 3D models only (the cut height)
FFT_VARIABLES = ('frequency',)      # variables of a timeseries curve with series FFT_SERIES
_PARITY_TAG = re.compile(r'\s*\[[^\]]*\]\s*$')   # trailing ' [x_lo symmetric]' of a mirrored eigen snapshot label
CURVE_SAMPLES = 501
MAX_CURVE_SAMPLES = 20000           # timeseries curves: the result's own samples up to this count
SWEEP_CURVE_SAMPLES = 201
SWEEP_FAR_FIELD_DB_FLOOR = -60.0    # dB floor of a far field drawn per sweep value (as the GUI far-field view)
_PER_POINT_FLOOR = 1e-12            # curve_error per_point: |analytic| floored at this fraction of its largest value
_ABSCISSA_KEYS = {'timeseries': 't', 'flux_spectrum': 'wavelength', 'far_field': 'angles', 'xy': 'x'}
_LORENTZ_ITERATIONS = 200           # Levenberg-Marquardt iterations of lorentz_fit (scaled data, 4 parameters)
_MULTI_LORENTZ_ITERATIONS = 500     # Levenberg-Marquardt iterations of multi_lorentz_fit (3 K + 1 parameters)
_POLE_ITERATIONS = 30               # Sanathanan-Koerner pole relocations of pole_fit at most (stops when poles settle)
_POLE_SEED_DAMPING = 0.02           # pole_fit: Im of a seed pole without a resolved peak (scaled abscissa, span 2)
_GOLDEN_ITERATIONS = 200            # golden-section steps of the one-parameter fits (bracket shrinks 0.618 per step)
_TWO_EXP_SCAN = np.geomspace(1e-3, 700.0, 600)   # kappa x (fitted span) scanned by two_exp_fit before refining
_TWO_WAVE_SCAN_STEP = 0.2           # beta scan step of two_wave_fit x (fitted span): well below the pi / span lobe
_TWO_WAVE_MAX_SCAN = 20000          # beta scan points of two_wave_fit at most
_RICHARDSON_ORDER_SCAN = np.linspace(0.05, 12.0, 598)   # |order| scanned per sign by richardson before refining


def split_key(key, what='physics.feature'):
    """(physics, name) of a 'physics.name' key (result key or field key); ValueError naming `what` otherwise."""
    parts = key.split('.') if isinstance(key, str) else ()
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"expected '{what}', got {key!r}")
    return parts[0], parts[1]


def built_parameters(spec, summary=None):
    """Parameter values of the last session built from spec: the study summary's 'parameters' when given (the session
    the results come from, also after a cancelled sweep), else a sweep's parameter at its last value; parameters that
    fail are left out."""
    recorded = None if summary is None else summary.get('parameters')
    if isinstance(recorded, dict):
        return {name: float(value) for name, value in recorded.items()}
    parameters = evaluate_parameters(spec.parameters, errors={})
    sweep = spec.study.sweep
    sweep_points = [] if sweep is None else sweep.point_values()
    if not sweep_points:
        return parameters
    try:
        last_value = evaluate_scalar(sweep_points[-1], parameters)
    except ExpressionError:
        return parameters
    return parameters_with(spec, sweep.parameter, last_value)


def relative_error(value, reference):
    """|value - reference| / |reference| (absolute difference for a zero reference); NaN unless both are finite."""
    value, reference = float(value), float(reference)
    if not (math.isfinite(value) and math.isfinite(reference)):
        return math.nan
    difference = abs(value - reference)
    return difference / abs(reference) if reference != 0.0 else difference


def derived_references(derived_specs, parameters, values):
    """({name: reference}, {name: relative error}, [failure messages]) of the derived values with a reference; NaN
    for a failing reference or a missing value."""
    references, errors, messages = {}, {}, []
    for derived in derived_specs:
        if derived.reference is None:
            continue
        try:
            reference = evaluate_scalar(derived.reference, parameters)
        except ExpressionError as exc:
            reference = math.nan
            messages.append(f"derived value '{derived.name}' reference: {exc}")
        references[derived.name] = reference
        errors[derived.name] = relative_error(values.get(derived.name, math.nan), reference)
    return references, errors, messages


# ---- derived values of 1D feature results ----
def result_samples(result, wavelength=None):
    """(abscissa name, abscissa float64 (n,), {name: float64 (n,) or float}) of a 1D feature result: its numeric
    scalars and n-sample arrays by key, plus xy series / numeric table columns by name and the table 'row' index;
    far_field: the pattern at the recorded wavelength nearest to `wavelength` [m] (None: the first)."""
    kind = result.get('kind') if isinstance(result, dict) else None
    if kind not in RESULT_SAMPLE_NAMES:
        raise ValueError(f"a {kind} result has no 1D samples; valid result kinds: {', '.join(RESULT_SAMPLE_NAMES)}")
    abscissa_name = RESULT_SAMPLE_NAMES[kind][0]
    if kind == 'far_field':
        return abscissa_name, np.asarray(result['angles'], dtype=np.float64), _far_field_samples(result, wavelength)
    rows = list(result['rows']) if kind == 'table' else None
    abscissa = (np.arange(len(rows), dtype=np.float64) if kind == 'table'
                else np.asarray(result[_ABSCISSA_KEYS[kind]], dtype=np.float64))
    samples = {}
    for key, value in result.items():
        if not isinstance(key, str) or isinstance(value, (bool, np.bool_, str)):
            continue
        if isinstance(value, numbers.Real):
            samples[key] = float(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            array = _float_array(value)
            if array is not None and array.ndim == 1 and array.size == abscissa.size:
                samples[key] = array
    if kind == 'xy':
        samples.update({str(name): np.asarray(values, dtype=np.float64) for name, values in result['series'].items()})
    if kind == 'table':
        samples.update(numeric_table_columns(result))
    samples[abscissa_name] = abscissa
    return abscissa_name, abscissa, samples


def _far_field_samples(result, wavelength=None):
    """Sample names of a far_field result at one recorded wavelength: angle, power, power_norm (over its peak),
    power_per_cw when present, and that wavelength / frequency as numbers."""
    wavelengths = np.atleast_1d(np.asarray(result['wavelength'], dtype=np.float64))
    index = 0 if wavelength is None else int(np.argmin(np.abs(wavelengths - float(wavelength))))
    power = np.atleast_2d(np.asarray(result['power'], dtype=np.float64))[index]
    samples = {'angle': np.asarray(result['angles'], dtype=np.float64), 'power': power,
               'power_norm': _peak_normalised(power), 'wavelength': float(wavelengths[index]),
               'frequency': float(np.atleast_1d(np.asarray(result['frequency'], dtype=np.float64))[index])}
    if result.get('power_per_cw') is not None:
        samples['power_per_cw'] = np.atleast_2d(np.asarray(result['power_per_cw'], dtype=np.float64))[index]
    return samples


def numeric_table_columns(result):
    """{column: float64 (n_rows,)} of the columns of a 'table' result whose cells are all numbers or None (NaN)."""
    rows = list(result['rows'])
    columns = {}
    for index, column in enumerate(result['columns']):
        cells = [row[index] if index < len(row) else None for row in rows]
        if all(cell is None or isinstance(cell, numbers.Real) for cell in cells):
            columns[str(column)] = np.array([math.nan if cell is None else float(cell) for cell in cells],
                                            dtype=np.float64)
    return columns


def table_xy(result, x_column=''):
    """'xy' result of a 'table': x = numeric column x_column ('' = the 0-based row), series = the other numeric
    columns, rows sorted by x."""
    columns = numeric_table_columns(result)
    names = [str(column) for column in result['columns']]
    units = dict(zip(names, (str(unit) for unit in (result.get('units') or [''] * len(names)))))
    if x_column:
        if x_column not in columns:
            raise ValueError(f"the table has no numeric column {x_column!r}; numeric columns: "
                             f"{', '.join(columns) or 'none'}")
        x = columns[x_column]
    else:
        x = np.arange(len(result['rows']), dtype=np.float64)
    order = np.argsort(x, kind='stable')
    series = {name: values[order] for name, values in columns.items() if name != x_column}
    return {'kind': 'xy', 'x': x[order], 'x_label': x_column or 'row', 'x_unit': units.get(x_column, ''),
            'y_label': '', 'series': series, 'series_units': {name: units.get(name, '') for name in series}}


def is_curve_source(source):
    """True when a DerivedValueSpec.source names a derived curve (no '.'), not a 'physics.feature' result."""
    return isinstance(source, str) and '.' not in source


def result_value(result, derived, parameters):
    """Derived value (DerivedValueSpec with a source) of a 1D result: its expression of the sample names (they shadow
    parameters) per sample, reduced by its operation over the finite samples inside its window (reduce_samples), last
    (result order) or at (linear interpolation; far_field angles wrap around)."""
    wavelength = None if derived.wavelength is None else evaluate_scalar(derived.wavelength, parameters)
    abscissa_name, abscissa, samples = result_samples(result, wavelength)
    if abscissa.size == 0:
        raise ValueError('the result has no samples')
    if derived.operation in COMPLEX_FIT_OPERATIONS:
        values = _complex_series(result, derived.expression, samples, abscissa, derived.operation)
    else:
        values = np.broadcast_to(np.asarray(compile_expression(derived.expression).evaluate(
            {**parameters, **samples}), dtype=np.float64), abscissa.shape)
    if derived.operation == 'last':
        return float(values[-1])
    period = FAR_FIELD_PERIOD if result.get('kind') == 'far_field' else None
    order = np.argsort(abscissa, kind='stable')
    x, y = abscissa[order], values[order]
    if derived.operation == 'at':
        position = evaluate_scalar(derived.at, parameters)
        if period is not None:
            return float(np.interp(position, x, y, period=period))
        if not x[0] <= position <= x[-1]:
            raise ValueError(f'at = {position:g} lies outside the sampled {abscissa_name} range '
                             f'[{x[0]:g}, {x[-1]:g}]')
        return float(np.interp(position, x, y))
    window = window_bounds(derived.window, parameters) if derived.window else None
    x, y = _reduction_samples(x, y, window, period, derived.operation)
    if x.size == 0:
        if window is not None:
            raise ValueError(f'no finite sample inside the window {abscissa_name} = [{window[0]:g}, {window[1]:g}]')
        return math.nan
    value = reduce_samples(x, y, derived.operation, derived.fit, derived.degree)
    if period is not None and window is None and derived.operation == 'argmax':
        value %= period   # reported in [0, 2 pi) like the result's angles
    return value


def complex_fit_series(expression, operation='two_wave_fit'):
    """(real-part name, imaginary-part name) of a complex fit's expression (COMPLEX_FIT_OPERATIONS): one bare
    '<name>_re' sample name, paired with '<name>_im'; ValueError otherwise."""
    name = expression.strip() if isinstance(expression, str) else ''
    if not (name.isidentifier() and name.endswith(REAL_PART_SUFFIX) and len(name) > len(REAL_PART_SUFFIX)):
        raise ValueError(f"{operation} needs the expression to be one series name ending in '{REAL_PART_SUFFIX}' "
                         f"(its '{IMAGINARY_PART_SUFFIX}' sibling is paired by name), got {expression!r}")
    return name, name[:-len(REAL_PART_SUFFIX)] + IMAGINARY_PART_SUFFIX


def _complex_series(result, expression, samples, abscissa, operation):
    """complex128 samples re + j im of a complex fit's source: its '<name>_re' series and the '<name>_im' sibling on
    the result's abscissa; ValueError otherwise."""
    names = complex_fit_series(expression, operation)
    for name in names:
        if name not in samples or np.ndim(samples[name]) == 0:
            sampled = ', '.join(key for key, value in samples.items() if np.ndim(value) == 1) or 'none'
            raise ValueError(f'{operation}: the result has no sampled series {name!r} (pairs {names[0]!r} with '
                             f'{names[1]!r}); sampled names: {sampled}')
    own_abscissae = result.get('series_x') or {}
    for name in names:
        if name in own_abscissae and not np.array_equal(np.asarray(own_abscissae[name], dtype=np.float64), abscissa):
            raise ValueError(f'{operation}: series {name!r} has its own abscissa; {names[0]!r} and {names[1]!r} '
                             f'must share the x samples of the result')
    return np.asarray(samples[names[0]], dtype=np.float64) + 1j * np.asarray(samples[names[1]], dtype=np.float64)


def window_bounds(window, parameters):
    """(lo, hi) of a DerivedValueSpec.window [lo, hi] (numbers or parameter expressions, sorted); ValueError
    otherwise."""
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError(f'window must be [lo, hi] in the abscissa unit, got {window!r}')
    lo, hi = sorted(evaluate_scalar(value, parameters) for value in window)
    if lo == hi:
        raise ValueError(f'the window [{lo:g}, {hi:g}] is empty')
    return lo, hi


def _reduction_samples(x, y, window, period, operation):
    """Sorted finite (x, y) inside the window (None: every sample); periodic abscissae (period) are tiled so a window
    or a peak may wrap (no window: one period, centred on the peak for argmax / fwhm); integral adds the window edges
    by linear interpolation."""
    if period is None and window is None:
        finite = np.isfinite(y)
        return x[finite], y[finite]
    closed = True
    if window is not None:
        lo, hi = window
        if period is not None and hi - lo > period * (1.0 + 1e-12):
            raise ValueError(f'the window [{lo:g}, {hi:g}] spans more than one period ({period:g})')
    elif operation in ('argmax', 'fwhm') and np.isfinite(y).any():
        peak = float(x[np.nanargmax(y)])
        lo, hi, closed = peak - 0.5 * period, peak + 0.5 * period, False
    else:
        lo, hi, closed = float(x[0]), float(x[0]) + period, operation == 'integral'
    if period is not None:
        x, y = np.concatenate([x - period, x, x + period]), np.tile(y, 3)
        order = np.argsort(x, kind='stable')
        x, y = x[order], y[order]
    tolerance = 1e-9 * (hi - lo)
    if operation == 'integral':
        edges = [edge for edge in (lo, hi) if x[0] < edge < x[-1] and not np.any(np.abs(x - edge) <= tolerance)]
        if edges:
            x, y = np.concatenate([x, edges]), np.concatenate([y, np.interp(edges, x, y)])
            order = np.argsort(x, kind='stable')
            x, y = x[order], y[order]
    inside = (x >= lo - tolerance) & ((x <= hi + tolerance) if closed else (x < hi - tolerance)) & np.isfinite(y)
    return x[inside], y[inside]


def reduce_samples(x, y, operation, fit='', degree=1):
    """One number of finite samples y(x), x sorted: mean / max / min, integral (trapezoid), argmax (abscissa of the
    maximum, 3-point parabola), fwhm (full width at half the maximum), or the fitted quantity fit of polyfit (degree),
    exp_fit (y = amplitude exp(rate x)), lorentz_fit (y = offset + amplitude / (1 + (2 (x - center) / fwhm)^2)),
    two_exp_fit, two_wave_fit (y complex), richardson, multi_lorentz_fit or pole_fit (y complex; degree = number of
    peaks / poles) (see their functions)."""
    if operation == 'max':
        return float(y.max())
    if operation == 'min':
        return float(y.min())
    if operation == 'mean':
        return float(y.mean())
    if operation == 'integral':
        return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))
    if operation == 'argmax':
        return _argmax(x, y)
    if operation == 'fwhm':
        return _fwhm(x, y)
    if operation in FIT_QUANTITIES:
        problem = fit_problem(operation, fit, degree)
        if problem is not None:
            raise ValueError(problem)
        if operation == 'polyfit':
            return _polyfit(x, y, degree)[int(fit[1:])]
        if operation == 'exp_fit':
            return _exp_fit(x, y)[fit]
        if operation == 'two_exp_fit':
            return two_exp_fit(x, y)[fit]
        if operation == 'two_wave_fit':
            return two_wave_fit(x, y)[fit]
        if operation == 'richardson':
            return richardson_fit(x, y)[fit]
        if operation == 'multi_lorentz_fit':
            return multi_lorentz_fit(x, y, int(degree))[fit]
        if operation == 'pole_fit':
            return pole_fit(x, y, int(degree))[fit]
        return _lorentz_fit(x, y)[fit]
    raise ValueError(f"unknown operation {operation!r} for a 1D result")


def _lowest_fit_degree(operation):
    """Smallest valid degree of a fit operation: 1 for the peak fits (one peak / pole), else 0."""
    return 1 if operation in PEAK_FIT_QUANTITIES else 0


def fit_quantities(operation, degree):
    """Fitted quantities (DerivedValueSpec.fit choices) of a fit operation at degree; () for a non-fit operation or an
    invalid degree. The one rule for analytic, validate and the GUI."""
    if operation not in FIT_QUANTITIES:
        return ()
    if FIT_QUANTITIES[operation] is not None:
        return tuple(FIT_QUANTITIES[operation])
    if (isinstance(degree, bool) or not isinstance(degree, numbers.Integral)
            or not _lowest_fit_degree(operation) <= degree <= MAX_FIT_DEGREE):
        return ()
    if operation == 'polyfit':
        return tuple(f'c{k}' for k in range(int(degree) + 1))
    per_peak, shared = PEAK_FIT_QUANTITIES[operation]
    return (tuple(f'{name}_{k}' for k in range(1, int(degree) + 1) for name in per_peak)
            + tuple(name for name in shared if name != 'spacing' or degree >= 2))


def fit_problem(operation, fit, degree):
    """Message when fit / degree do not fit a fit operation (fit_quantities; degree 0 .. MAX_FIT_DEGREE, peak fits
    from 1), else None."""
    lowest = _lowest_fit_degree(operation)
    if isinstance(degree, bool) or not isinstance(degree, numbers.Integral) or not lowest <= degree <= MAX_FIT_DEGREE:
        return f'degree must be a whole number {lowest} .. {MAX_FIT_DEGREE}, got {degree!r}'
    valid = fit_quantities(operation, degree)
    if fit not in valid:
        return f"fit {fit!r} is no quantity of {operation} at degree {degree}; valid: {', '.join(valid)}"
    return None


def _argmax(x, y):
    """Abscissa of the largest sample, refined by the parabola through it and its neighbours (kept within them)."""
    k = int(np.argmax(y))
    if not 0 < k < x.size - 1:
        return float(x[k])
    left, right = x[k - 1] - x[k], x[k + 1] - x[k]   # relative to the peak sample: no cancellation in x**2
    rise_left, rise_right = y[k - 1] - y[k], y[k + 1] - y[k]
    denominator = left * right * (left - right)
    if denominator == 0.0:
        return float(x[k])
    curvature = (rise_left * right - rise_right * left) / denominator
    slope = (rise_left * right ** 2 - rise_right * left ** 2) / (-denominator)
    if not curvature < 0.0:
        return float(x[k])
    return float(x[k] + min(max(-slope / (2.0 * curvature), left), right))


def _fwhm(x, y):
    """Full width of the largest peak where it falls to half its maximum on both sides (linear crossings)."""
    k = int(np.argmax(y))
    peak = float(y[k])
    if not peak > 0.0:
        raise ValueError(f'fwhm needs a positive maximum, got {peak:g} (subtract a baseline in the expression)')
    half = 0.5 * peak
    left_below, right_below = np.nonzero(y[:k] <= half)[0], np.nonzero(y[k + 1:] <= half)[0]
    if left_below.size == 0 or right_below.size == 0:
        side = 'below' if left_below.size == 0 else 'above'
        raise ValueError(f'the peak at {x[k]:g} does not fall to half its maximum {side} it within the samples '
                         f'(widen the window or the sampled range)')
    i, j = int(left_below[-1]), int(k + 1 + right_below[0])
    x_left = x[i] + (half - y[i]) * (x[i + 1] - x[i]) / (y[i + 1] - y[i])
    x_right = x[j - 1] + (half - y[j - 1]) * (x[j] - x[j - 1]) / (y[j] - y[j - 1])
    return float(x_right - x_left)


def _polyfit(x, y, degree):
    """Least-squares coefficients c0 .. c<degree> (of x**k) of the samples (scaled internally, numpy Polynomial)."""
    if x.size <= degree or (degree > 0 and float(np.ptp(x)) == 0.0):
        raise ValueError(f'polyfit of degree {degree} needs at least {degree + 1} distinct abscissae, got {x.size} '
                         f'samples')
    coefficients = np.polynomial.Polynomial.fit(x, y, degree).convert().coef
    return np.pad(coefficients, (0, degree + 1 - coefficients.size))


def _exp_fit(x, y):
    """{amplitude, rate, decay = -1 / rate} of y = amplitude exp(rate x): log-linear least squares weighted by |y|
    (about the residuals of y); every sample must have one sign."""
    if not (np.all(y > 0.0) or np.all(y < 0.0)):
        raise ValueError('exp_fit needs samples of one sign and none zero (shift or window the curve)')
    if x.size < 2 or float(np.ptp(x)) == 0.0:
        raise ValueError(f'exp_fit needs at least 2 distinct abscissae, got {x.size} samples')
    magnitude = np.abs(y)
    coefficients = np.polynomial.Polynomial.fit(x, np.log(magnitude), 1, w=magnitude).convert().coef
    log_amplitude, rate = np.pad(coefficients, (0, 2 - coefficients.size))
    amplitude = math.copysign(math.exp(log_amplitude), float(y[0]))
    return {'amplitude': amplitude, 'rate': float(rate), 'decay': -1.0 / float(rate) if rate != 0.0 else math.inf}


def _lorentz_fit(x, y):
    """{center, fwhm, amplitude, offset, q = |center| / fwhm} of y = offset + amplitude / (1 + (2 (x - center) /
    fwhm)^2): Levenberg-Marquardt on scaled data from the extremum farthest from the median (peak or dip)."""
    if x.size < 5:
        raise ValueError(f'lorentz_fit needs at least 5 samples, got {x.size}')
    x_mid, x_scale = 0.5 * (x[0] + x[-1]), 0.5 * (x[-1] - x[0])
    y_mid = float(np.median(y))
    y_scale = float(np.max(np.abs(y - y_mid)))
    if not (x_scale > 0.0 and y_scale > 0.0):
        raise ValueError('lorentz_fit needs a varying curve over distinct abscissae')
    u, v = (x - x_mid) / x_scale, (y - y_mid) / y_scale
    k = int(np.argmax(np.abs(v)))
    beyond_half = np.nonzero(v / v[k] <= 0.5)[0]   # start width from the half-height crossings around the extremum
    below, above = beyond_half[beyond_half < k], beyond_half[beyond_half > k]
    half_width = 0.5 * (u[above[0]] - u[below[-1]]) if below.size and above.size else 0.1
    p = np.array([0.0, v[k], u[k], max(half_width, 1e-6)])   # offset, amplitude, center, half width (scaled)

    def residual(params):
        return v - (params[0] + params[1] / (1.0 + ((u - params[2]) / params[3]) ** 2))

    cost, damping = float(np.sum(residual(p) ** 2)), 1e-3
    for _ in range(_LORENTZ_ITERATIONS):
        s = (u - p[2]) / p[3]
        lorentz = 1.0 / (1.0 + s * s)
        jacobian = np.column_stack([np.ones_like(u), lorentz, 2.0 * p[1] * s * lorentz ** 2 / p[3],
                                    2.0 * p[1] * s * s * lorentz ** 2 / p[3]])
        normal, gradient = jacobian.T @ jacobian, jacobian.T @ residual(p)
        while damping < 1e12:
            try:
                step = np.linalg.solve(normal + damping * (np.diag(np.diag(normal)) + 1e-12 * np.eye(4)), gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            trial = p + step
            trial_cost = float(np.sum(residual(trial) ** 2)) if trial[3] != 0.0 else math.inf
            if math.isfinite(trial_cost) and trial_cost < cost:
                break
            damping *= 10.0
        else:
            break   # no downhill step left: converged (or stuck)
        improvement = cost - trial_cost
        p, cost, damping = trial, trial_cost, max(damping / 10.0, 1e-12)
        if improvement <= 1e-15 * max(cost, 1e-300) or np.max(np.abs(step)) <= 1e-13:
            break
    center, fwhm = x_mid + p[2] * x_scale, 2.0 * abs(p[3]) * x_scale
    if not (math.isfinite(center) and math.isfinite(fwhm) and fwhm > 0.0) or not x[0] <= center <= x[-1]:
        raise ValueError('lorentz_fit found no resolved line inside the samples (window the curve around one line)')
    return {'center': float(center), 'fwhm': float(fwhm), 'amplitude': float(p[1] * y_scale),
            'offset': float(y_mid + p[0] * y_scale), 'q': abs(float(center)) / float(fwhm)}


def _golden_minimum(cost, lo, hi):
    """Abscissa in [lo, hi] of the minimum of a cost unimodal there (golden-section search)."""
    ratio = 0.5 * (math.sqrt(5.0) - 1.0)
    a, b = min(lo, hi), max(lo, hi)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    cost_c, cost_d = cost(c), cost(d)
    for _ in range(_GOLDEN_ITERATIONS):
        if b - a <= 1e-15 * max(abs(a), abs(b)):
            break
        if cost_c <= cost_d:
            b, d, cost_d = d, c, cost_c
            c = b - ratio * (b - a)
            cost_c = cost(c)
        else:
            a, c, cost_c = c, d, cost_d
            d = a + ratio * (b - a)
            cost_d = cost(d)
    return c if cost_c <= cost_d else d


def _least_squares(basis, y):
    """(squared residual norm, coefficients) of the linear least squares basis @ coefficients ~ y (real or complex)."""
    coefficients = np.linalg.lstsq(basis, y, rcond=None)[0]
    residual = y - basis @ coefficients
    return float(np.vdot(residual, residual).real), coefficients


def _fit_samples(x, y, name, minimum, dtype=np.float64):
    """(x float64, y of dtype) sorted by x for a fit: distinct abscissae, at least `minimum` samples, else ValueError."""
    x, y = np.asarray(x, dtype=np.float64).ravel(), np.asarray(y, dtype=dtype).ravel()
    if x.size != y.size:
        raise ValueError(f'{name}: {x.size} abscissae for {y.size} values')
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    if x.size < minimum:
        raise ValueError(f'{name} needs at least {minimum} samples, got {x.size}')
    if np.any(np.diff(x) <= 0.0):
        raise ValueError(f'{name} needs distinct abscissae (repeated x in the samples)')
    return x, y


def _r_squared(cost, y, name):
    """Coefficient of determination 1 - cost / sum |y - mean y|^2; ValueError for a constant curve."""
    spread = y - y.mean()
    total = float(np.vdot(spread, spread).real)
    if not total > 0.0:
        raise ValueError(f'{name} needs a varying curve (every sample equal)')
    return 1.0 - cost / total


def two_exp_fit(x, y):
    """{kappa, decay_length, amplitude_start, amplitude_end, r_squared} of y = amplitude_start exp(-kappa (x - x_first))
    + amplitude_end exp(kappa (x - x_last)), kappa > 0 (outermost samples x_first / x_last; least squares)."""
    x, y = _fit_samples(x, y, 'two_exp_fit', 4)
    span = float(x[-1] - x[0])
    position = (x - x[0]) / span   # 0 at the first sample, 1 at the last

    def solve(scaled_kappa):
        return _least_squares(np.column_stack([np.exp(-scaled_kappa * position),
                                               np.exp(scaled_kappa * (position - 1.0))]), y)

    costs = np.array([solve(scaled_kappa)[0] for scaled_kappa in _TWO_EXP_SCAN])
    k = int(np.argmin(costs))
    if not 0 < k < costs.size - 1:
        raise ValueError(f'two_exp_fit found no decay rate inside kappa * span = [{_TWO_EXP_SCAN[0]:g}, '
                         f'{_TWO_EXP_SCAN[-1]:g}] (span {span:g}): window the curve over the decay')
    log_scan = np.log(_TWO_EXP_SCAN)
    scaled_kappa = math.exp(_golden_minimum(lambda value: solve(math.exp(value))[0], log_scan[k - 1],
                                            log_scan[k + 1]))
    cost, (amplitude_start, amplitude_end) = solve(scaled_kappa)
    kappa = scaled_kappa / span
    return {'kappa': kappa, 'decay_length': 1.0 / kappa, 'amplitude_start': float(amplitude_start),
            'amplitude_end': float(amplitude_end), 'r_squared': _r_squared(cost, y, 'two_exp_fit')}


def _two_wave_scan_costs(x, y, betas, chunk=256):
    """Squared residual of the two-wave least squares per beta (2 x 2 normal equations; inf where the waves are nearly
    parallel on the samples)."""
    n = float(x.size)
    total = float(np.vdot(y, y).real)
    costs = np.empty(betas.size)
    for start in range(0, betas.size, chunk):
        beta = betas[start:start + chunk, None]
        phase = np.exp(1j * beta * x[None, :])      # exp(+j beta x): conj of the forward wave
        forward_projection = phase @ y               # forward^H y
        backward_projection = phase.conj() @ y       # backward^H y
        overlap = (phase * phase).sum(axis=1)        # forward^H backward = sum exp(2 j beta x)
        determinant = n * n - np.abs(overlap) ** 2
        quadratic = (n * (np.abs(forward_projection) ** 2 + np.abs(backward_projection) ** 2)
                     - 2.0 * (forward_projection.conj() * overlap * backward_projection).real)
        with np.errstate(divide='ignore', invalid='ignore'):   # |overlap| > 0.995 n: no resolvable pair of waves
            part = np.where(determinant > 1e-2 * n * n, np.maximum(total - quadratic / determinant, 0.0), math.inf)
        costs[start:start + chunk] = part
    return costs


def two_wave_fit(x, y):
    """{beta, wavelength, forward_abs / _phase, backward_abs / _phase, reflection_abs, r_squared} of complex y = A_f
    exp(-j beta x) + A_b exp(+j beta x), beta > 0 (A_f toward +x in exp(+j omega t), phases at x = 0; least squares)."""
    x, y = _fit_samples(x, y, 'two_wave_fit', 4, dtype=np.complex128)
    span = float(x[-1] - x[0])
    middle = 0.5 * (x[0] + x[-1])
    centred = x - middle   # solved about the middle (conditioning), phases moved back to x = 0
    beta_low, beta_high = 0.25 * math.pi / span, math.pi / float(np.median(np.diff(x)))
    if not beta_high > 2.0 * beta_low:
        raise ValueError(f'two_wave_fit needs more samples per span (got {x.size} over {span:g})')
    n_scan = int(min(_TWO_WAVE_MAX_SCAN, max(64, math.ceil((beta_high - beta_low) * span / _TWO_WAVE_SCAN_STEP))))
    betas = np.linspace(beta_low, beta_high, n_scan)
    costs = _two_wave_scan_costs(centred, y, betas)
    k = int(np.argmin(costs))
    if not 0 < k < costs.size - 1 or not math.isfinite(costs[k]):
        raise ValueError(f'two_wave_fit found no wavenumber inside beta = [{beta_low:g}, {beta_high:g}] (a quarter '
                         f'wave over the span up to the sample Nyquist limit): window the curve or sample it finer')

    def solve(beta):
        return _least_squares(np.column_stack([np.exp(-1j * beta * centred), np.exp(1j * beta * centred)]), y)

    beta = _golden_minimum(lambda value: solve(value)[0], betas[k - 1], betas[k + 1])
    cost, (forward, backward) = solve(beta)
    forward, backward = forward * np.exp(1j * beta * middle), backward * np.exp(-1j * beta * middle)
    forward_abs, backward_abs = float(abs(forward)), float(abs(backward))
    return {'beta': float(beta), 'wavelength': 2.0 * math.pi / float(beta), 'forward_abs': forward_abs,
            'forward_phase': float(np.angle(forward)), 'backward_abs': backward_abs,
            'backward_phase': float(np.angle(backward)),
            'reflection_abs': backward_abs / forward_abs if forward_abs > 0.0 else math.inf,
            'r_squared': _r_squared(cost, y, 'two_wave_fit')}


def richardson_fit(x, y):
    """{extrapolated, order, coefficient, error_estimate, relative_error_estimate} of y = extrapolated + coefficient
    x^order, x > 0 (least squares; error_estimate = |y - extrapolated| of the sample nearest the limit)."""
    x, y = _fit_samples(x, y, 'richardson', 3)
    if np.any(x <= 0.0):
        raise ValueError('richardson needs abscissae > 0 (a step size, or a cell / point count)')
    changes = np.diff(y)
    if np.all(changes == 0.0):
        return {'extrapolated': float(y[-1]), 'order': math.nan, 'coefficient': 0.0, 'error_estimate': 0.0,
                'relative_error_estimate': 0.0}
    if not (np.all(changes > 0.0) or np.all(changes < 0.0)):
        raise ValueError('richardson needs values changing monotonically with x (the asymptotic range): the changes '
                         'between neighbouring samples alternate in sign or vanish')
    reference = math.exp(float(np.mean(np.log(x))))
    scaled = x / reference   # about 1: the order scan does not overflow

    def solve(order):
        return _least_squares(np.column_stack([np.ones_like(scaled), scaled ** order]), y)

    best = None
    for sign in (1.0, -1.0):
        orders = sign * _RICHARDSON_ORDER_SCAN
        costs = np.array([solve(order)[0] for order in orders])
        k = int(np.argmin(costs))
        if best is None or costs[k] < best[0]:
            best = (float(costs[k]), orders, k)
    _cost, orders, k = best
    if k == orders.size - 1:
        raise ValueError(f'richardson: the fitted order exceeds {_RICHARDSON_ORDER_SCAN[-1]:g} in magnitude')
    if k == 0:
        raise ValueError(f'richardson: the fitted order is below {_RICHARDSON_ORDER_SCAN[0]:g} in magnitude (the '
                         f'values do not converge like a power of x)')
    order = _golden_minimum(lambda value: solve(value)[0], orders[k - 1], orders[k + 1])
    _cost, (extrapolated, scaled_coefficient) = solve(order)
    extrapolated, scaled_coefficient = float(extrapolated), float(scaled_coefficient)
    if scaled_coefficient == 0.0:
        coefficient = 0.0
    else:
        log_coefficient = math.log(abs(scaled_coefficient)) - order * math.log(reference)
        coefficient = math.copysign(math.exp(log_coefficient) if log_coefficient < 700.0 else math.inf,
                                    scaled_coefficient)
    error = abs(float(y[0] if order > 0.0 else y[-1]) - extrapolated)
    return {'extrapolated': extrapolated, 'order': float(order), 'coefficient': coefficient, 'error_estimate': error,
            'relative_error_estimate': error / abs(extrapolated) if extrapolated != 0.0 else math.inf}


# ---- peak fits (degree = number of peaks / poles) ----
def _prominent_maxima(values, count):
    """Ascending indices of the `count` most prominent interior local maxima of values (fewer when fewer exist);
    prominence = height above the higher of the lowest points toward the nearest higher sample on either side."""
    n = values.size
    is_peak = (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
    candidates = np.arange(1, n - 1)[is_peak]
    prominences = []
    for index in candidates:
        higher = np.nonzero(values > values[index])[0]
        left_higher, right_higher = higher[higher < index], higher[higher > index]
        start = int(left_higher[-1]) if left_higher.size else 0
        stop = int(right_higher[0]) if right_higher.size else n - 1
        base = max(float(values[start:index + 1].min()), float(values[index:stop + 1].min()))
        prominences.append(float(values[index]) - base)
    chosen = np.argsort(np.asarray(prominences, dtype=np.float64), kind='stable')[::-1][:count]
    return np.sort(candidates[chosen])


def _seed_half_widths(u, values, peaks):
    """Half widths (abscissa u) where each seeded peak of values falls to half its height (baseline 0; the nearer
    side), at most half the distance to a neighbouring seed; 2 sample spacings where no crossing exists."""
    spacing = float(np.median(np.diff(u)))
    widths = []
    for index in peaks:
        below = np.nonzero(values <= 0.5 * values[index])[0]
        left, right = below[below < index], below[below > index]
        sides = ([float(u[index] - u[left[-1]])] if left.size else []) + \
                ([float(u[right[0]] - u[index])] if right.size else [])
        width = min(sides) if sides else 2.0 * spacing
        neighbours = [abs(float(u[other] - u[index])) for other in peaks if other != index]
        if neighbours:
            width = min(width, 0.5 * min(neighbours))
        widths.append(max(width, spacing))
    return np.asarray(widths, dtype=np.float64)


def _line_spacing(centers):
    """Least-squares slope of ascending line centres against their number k (the mean line spacing)."""
    numbers_k = np.arange(centers.size, dtype=np.float64)
    numbers_k -= numbers_k.mean()
    return float(np.sum(numbers_k * (centers - centers.mean())) / np.sum(numbers_k * numbers_k))


def _levenberg_marquardt(model_and_jacobian, target, start, iterations):
    """(parameters, squared residual) minimising |target - model(parameters)|^2 from start (Gauss-Newton damped with
    Marquardt's diagonal scaling); model_and_jacobian(parameters) -> (model, d model / d parameters)."""
    parameters = np.asarray(start, dtype=np.float64).copy()
    with np.errstate(all='ignore'):
        model, jacobian = model_and_jacobian(parameters)
        residual = target - model
        cost, damping = float(residual @ residual), 1e-3
        identity = np.eye(parameters.size)
        for _ in range(iterations):
            normal, gradient = jacobian.T @ jacobian, jacobian.T @ residual
            scaling = np.diag(np.diag(normal)) + 1e-12 * identity
            while damping < 1e12:
                try:
                    step = np.linalg.solve(normal + damping * scaling, gradient)
                except np.linalg.LinAlgError:
                    damping *= 10.0
                    continue
                trial = parameters + step
                trial_model, trial_jacobian = model_and_jacobian(trial)
                trial_residual = target - trial_model
                trial_cost = float(trial_residual @ trial_residual)
                if math.isfinite(trial_cost) and trial_cost < cost:
                    break
                damping *= 10.0
            else:
                break   # no downhill step left: converged (or stuck)
            improvement = cost - trial_cost
            parameters, cost, residual, jacobian = trial, trial_cost, trial_residual, trial_jacobian
            damping = max(damping / 10.0, 1e-12)
            if improvement <= 1e-15 * max(cost, 1e-300) or np.max(np.abs(step)) <= 1e-13:
                break
    return parameters, cost


def multi_lorentz_fit(x, y, peak_count):
    """{center_<k>, fwhm_<k>, amplitude_<k>, q_<k>, offset, spacing (peak_count >= 2), r_squared} of y = offset +
    sum_k amplitude_k / (1 + (2 (x - center_k) / fwhm_k)^2), k by centre: joint Levenberg-Marquardt from the most
    prominent maxima (or minima: dips)."""
    name = 'multi_lorentz_fit'
    x, y = _fit_samples(x, y, name, 3 * peak_count + 2)
    x_mid, x_scale = 0.5 * (x[0] + x[-1]), 0.5 * (x[-1] - x[0])
    y_mid = float(np.median(y))
    y_scale = float(np.max(np.abs(y - y_mid)))
    if not y_scale > 0.0:
        raise ValueError(f'{name} needs a varying curve (every sample equal)')
    u, v = (x - x_mid) / x_scale, (y - y_mid) / y_scale
    sign = 1.0 if v[int(np.argmax(np.abs(v)))] > 0.0 else -1.0   # peaks, or dips below the background
    peaks = _prominent_maxima(sign * v, peak_count)
    if peaks.size < peak_count:
        extrema = 'maxima' if sign > 0.0 else 'minima'
        raise ValueError(f'{name} of degree {peak_count} found only {peaks.size} local {extrema} (lower the degree '
                         f'or widen the window)')
    start = np.empty(1 + 3 * peak_count)
    start[0] = 0.0
    start[1::3], start[2::3], start[3::3] = v[peaks], u[peaks], _seed_half_widths(u, sign * v, peaks)

    def model_and_jacobian(parameters):
        amplitude, center, half_width = parameters[1::3], parameters[2::3], parameters[3::3]
        s = (u[:, None] - center[None, :]) / half_width[None, :]
        lorentz = 1.0 / (1.0 + s * s)
        jacobian = np.empty((u.size, parameters.size))
        jacobian[:, 0] = 1.0
        jacobian[:, 1::3] = lorentz
        jacobian[:, 2::3] = 2.0 * amplitude * s * lorentz ** 2 / half_width
        jacobian[:, 3::3] = 2.0 * amplitude * s * s * lorentz ** 2 / half_width
        return parameters[0] + lorentz @ amplitude, jacobian

    parameters, cost = _levenberg_marquardt(model_and_jacobian, v, start, _MULTI_LORENTZ_ITERATIONS)
    centers = x_mid + parameters[2::3] * x_scale
    fwhms = 2.0 * np.abs(parameters[3::3]) * x_scale
    amplitudes = parameters[1::3] * y_scale
    order = np.argsort(centers, kind='stable')
    record = {}
    for k, index in enumerate(order, start=1):
        center, fwhm = float(centers[index]), float(fwhms[index])
        if not (math.isfinite(center) and math.isfinite(fwhm) and fwhm > 0.0) or not x[0] <= center <= x[-1]:
            raise ValueError(f'{name}: line {k} found no resolved peak inside the samples (center {center:g}): lower '
                             f'the degree or window the curve around the lines')
        record.update({f'center_{k}': center, f'fwhm_{k}': fwhm, f'amplitude_{k}': float(amplitudes[index]),
                       f'q_{k}': abs(center) / fwhm})
    record['offset'] = float(y_mid + parameters[0] * y_scale)
    if peak_count >= 2:
        record['spacing'] = _line_spacing(centers[order])
    record['r_squared'] = _r_squared(cost * y_scale ** 2, y, name)
    return record


def _pole_basis(u, poles):
    """Columns 1 / (u - pole_k), 1, u of the pole_fit model on the scaled abscissa u."""
    return np.column_stack([1.0 / (u[:, None] - poles[None, :]), np.ones_like(u), u])


def _seed_poles(u, v, pole_count):
    """Starting poles (scaled abscissa) of pole_fit: the most prominent maxima of |v - line through the end samples|^2
    with their half-power half widths as Im, the rest spread evenly over the samples."""
    line = v[0] + (v[-1] - v[0]) * (u - u[0]) / (u[-1] - u[0])
    power = np.abs(v - line) ** 2
    peaks = _prominent_maxima(power, pole_count)
    poles = [complex(position, width) for position, width in zip(u[peaks], _seed_half_widths(u, power, peaks))]
    for position in np.linspace(-1.0, 1.0, pole_count + 2)[1:-1]:
        if len(poles) >= pole_count:
            break
        poles.append(complex(position, _POLE_SEED_DAMPING))
    return np.asarray(poles, dtype=np.complex128)


def _relocated_poles(u, v, poles):
    """Zeros of the weight sigma(u) = 1 + sum_k c_k / (u - pole_k) of the linearised fit sigma v ~ sum_k r_k / (u -
    pole_k) + d0 + d1 u (vector fitting step): the eigenvalues of diag(poles) - 1 c^T."""
    partial = 1.0 / (u[:, None] - poles[None, :])
    basis = np.column_stack([partial, np.ones_like(u), u, -v[:, None] * partial])
    norms = np.linalg.norm(basis, axis=0)
    norms[norms == 0.0] = 1.0
    weights = (np.linalg.lstsq(basis / norms, v, rcond=None)[0] / norms)[-poles.size:]
    relocated = np.linalg.eigvals(np.diag(poles) - np.outer(np.ones(poles.size), weights))
    floor = 1e-12   # keep a pole off the real sample axis
    flat = np.abs(relocated.imag) < floor
    return np.where(flat, relocated.real + 1j * np.where(relocated.imag < 0.0, -floor, floor), relocated)


def pole_fit(x, y, pole_count):
    """{frequency_<k> = Re nu_k, q_<k> = |Re nu_k| / (2 |Im nu_k|), fwhm_<k> = 2 |Im nu_k|, residue_abs_<k>, spacing
    (pole_count >= 2), r_squared} of complex y = sum_k residue_k / (x - nu_k) + d0 + d1 x, k by Re nu (vector
    fitting: Sanathanan-Koerner pole relocation)."""
    name = 'pole_fit'
    x, y = _fit_samples(x, y, name, 2 * pole_count + 3, dtype=np.complex128)
    x_mid, x_scale = 0.5 * (x[0] + x[-1]), 0.5 * (x[-1] - x[0])
    y_scale = float(np.max(np.abs(y)))
    if not y_scale > 0.0:
        raise ValueError(f'{name} needs a non-zero curve')
    u, v = (x - x_mid) / x_scale, y / y_scale
    poles = _seed_poles(u, v, pole_count)
    with np.errstate(all='ignore'):
        for _ in range(_POLE_ITERATIONS):
            relocated = _relocated_poles(u, v, poles)
            if not np.all(np.isfinite(relocated)):
                raise ValueError(f'{name}: the pole relocation diverged (lower the degree or window the curve)')
            change = float(np.max(np.abs(np.sort_complex(relocated) - np.sort_complex(poles))))
            poles = relocated
            if change <= 1e-12:
                break
        cost, coefficients = _least_squares(_pole_basis(u, poles), v)
    residues = coefficients[:pole_count] * x_scale * y_scale
    frequencies = x_mid + poles * x_scale
    order = np.argsort(frequencies.real, kind='stable')
    record = {}
    for k, index in enumerate(order, start=1):
        pole = complex(frequencies[index])
        if not x[0] <= pole.real <= x[-1]:
            raise ValueError(f'{name}: pole {k} at Re nu = {pole.real:g} lies outside the samples [{x[0]:g}, '
                             f'{x[-1]:g}] (lower the degree or window the curve around the lines)')
        half_width = abs(pole.imag)
        record.update({f'frequency_{k}': pole.real, f'q_{k}': abs(pole.real) / (2.0 * half_width),
                       f'fwhm_{k}': 2.0 * half_width, f'residue_abs_{k}': float(abs(residues[index]))})
    if pole_count >= 2:
        record['spacing'] = _line_spacing(frequencies.real[order])
    record['r_squared'] = _r_squared(cost * y_scale ** 2, y, name)
    return record


def sweep_table_xy(parameter, sweep_values, results, column, sort=True):
    """'xy' result of one numeric 'table' column per sweep value (results: per value a table dict or None): x = the
    swept values, series '<column>_<k>' = the k-th row's value (sort: the k-th smallest, e.g. band k of
    resonance_monitor frequencies vs Bloch k), NaN where a value has fewer rows or failed."""
    if len(results) != len(sweep_values):
        raise ValueError(f'{len(results)} sweep results for {len(sweep_values)} sweep values')
    unit, per_value = '', []
    for result in results:
        if result is None:
            per_value.append(np.empty(0))
            continue
        if result.get('kind') != 'table':
            raise ValueError(f"sweep_table_xy needs 'table' results, got a {result.get('kind')} result")
        columns = [str(name) for name in result['columns']]
        if column not in columns:
            raise ValueError(f"table has no column {column!r}; columns: {', '.join(columns)}")
        index = columns.index(column)
        unit = (result.get('units') or [''] * len(columns))[index]
        cells = [row[index] if index < len(row) else None for row in result['rows']]
        values = np.array([math.nan if not isinstance(cell, numbers.Real) or isinstance(cell, bool) else float(cell)
                           for cell in cells])
        per_value.append(np.sort(values) if sort else values)   # NaN sort last
    n_series = max((values.size for values in per_value), default=0)
    series = {f'{column}_{k + 1}': np.array([values[k] if k < values.size else math.nan for values in per_value])
              for k in range(n_series)}
    return {'kind': 'xy', 'x': np.asarray(sweep_values, dtype=np.float64), 'x_label': str(parameter), 'x_unit': '',
            'y_label': column, 'series': series, 'series_units': dict.fromkeys(series, unit), 'sorted': bool(sort)}


def sweep_derived_xy(parameter, sweep_values, derived):
    """'xy' result of a sweep's derived values ({name: value per sweep value}: summary sweep 'derived'; None / 'nan' =
    NaN) vs the swept values, sorted: e.g. the source of a richardson reduction over a refinement sweep."""
    x = np.asarray(sweep_values, dtype=np.float64)
    order = np.argsort(x, kind='stable')
    series = {}
    for name, values in derived.items():
        values = np.array([math.nan if value is None else float(value) for value in values], dtype=np.float64)
        if values.size != x.size:
            raise ValueError(f'derived value {name!r}: {values.size} values for {x.size} sweep values')
        series[str(name)] = values[order]
    return {'kind': 'xy', 'x': x[order], 'x_label': str(parameter), 'x_unit': '', 'y_label': '', 'series': series,
            'series_units': dict.fromkeys(series, '')}


def sweep_richardson(parameter, sweep_values, derived, name):
    """richardson_fit record of one derived value's finite values over a refinement sweep (arguments as
    sweep_derived_xy; the swept parameter a step size or a cell / point count)."""
    xy = sweep_derived_xy(parameter, sweep_values, derived)
    if name not in xy['series']:
        raise ValueError(f"no derived value {name!r} in the sweep; derived values: {', '.join(xy['series']) or 'none'}")
    values = xy['series'][name]
    finite = np.isfinite(values) & np.isfinite(xy['x'])
    return richardson_fit(xy['x'][finite], values[finite])


def sweep_value_curve(result, series=''):
    """{x, y, x_label, x_unit, y_label, unit} of one sweep value's 1D result as the Sweep tab draws it: flux spectrum
    series '' / 'power' (or 'power_per_cw') vs wavelength, timeseries vs t (the 1-based iteration when every t is
    equal), far field in dB below its peak vs degrees at the first wavelength, xy series ('' = the first; derived
    series too); None for other kinds or a series the result lacks."""
    kind = result.get('kind') if isinstance(result, dict) else None
    if kind == 'flux_spectrum':
        name = series or 'power'
        if name not in ('power', 'power_per_cw') or name not in result:
            return None
        return {'x': result['wavelength'], 'y': result[name], 'x_label': 'Wavelength', 'x_unit': 'm',
                'y_label': 'Power', 'unit': result.get('unit', 'a.u.') if name == 'power' else 'W/m'}
    if kind == 'timeseries' and not series:
        component, unit = str(result.get('component', 'value')), result.get('unit', '')
        if is_iteration_series(result['t']):
            return {'x': np.arange(1.0, np.size(result['t']) + 1.0), 'y': result['values'], 'x_label': 'Iteration',
                    'x_unit': '', 'y_label': component, 'unit': unit}
        return {'x': result['t'], 'y': result['values'], 'x_label': 'Time', 'x_unit': 's', 'y_label': component,
                'unit': unit}
    if kind == 'far_field' and not series:
        angles = np.asarray(result['angles'], dtype=np.float64)
        power = np.asarray(result['power'], dtype=np.float64).reshape(-1, angles.size)[0]
        finite = power[np.isfinite(power)]
        peak = float(finite.max()) if finite.size else 0.0
        normalised = power / peak if peak > 0.0 else power
        db = 10.0 * np.log10(np.maximum(normalised, 10.0 ** (SWEEP_FAR_FIELD_DB_FLOOR / 10.0)))
        return {'x': np.degrees(angles), 'y': db, 'x_label': 'Angle from +x', 'x_unit': 'deg',
                'y_label': 'Normalised power, 1st frequency', 'unit': 'dB'}
    if kind == 'xy' and result['series']:
        name = series or next(iter(result['series']))
        if name not in result['series']:
            return None
        return {'x': (result.get('series_x') or {}).get(name, result['x']), 'y': result['series'][name],
                'x_label': result.get('x_label') or 'x', 'x_unit': result.get('x_unit', ''), 'y_label': name,
                'unit': (result.get('series_units') or {}).get(name, '')}
    return None


def sweep_overlay_xy(parameter, sweep_values, results, series=''):
    """'xy' result of one sweep_value_curve per swept value (results: per value a result dict or None): series
    '<parameter> = <value .4g>' (a repeated value ' (k)'), x = the first curve's abscissa, series_x the others' own
    where it differs; None when no value has a curve."""
    curves, series_x, units, axes, x_first = {}, {}, {}, None, None
    for value, result in zip(sweep_values, results):
        curve = sweep_value_curve(result, series)
        if curve is None:
            continue
        name = f'{parameter} = {float(value):.4g}'
        if name in curves:
            name = f'{name} ({len(curves) + 1})'
        x = np.asarray(curve['x'], dtype=np.float64)
        if x_first is None:
            x_first, axes = x, curve
        elif x.shape != x_first.shape or not np.array_equal(x, x_first):
            series_x[name] = x
        curves[name] = np.asarray(curve['y'], dtype=np.float64)
        units[name] = curve['unit']
    if x_first is None:
        return None
    xy = {'kind': 'xy', 'x': x_first, 'x_label': axes['x_label'], 'x_unit': axes['x_unit'],
          'y_label': axes['y_label'], 'series': curves, 'series_units': units}
    if series_x:
        xy['series_x'] = series_x
    return xy


def _float_array(value):
    """float64 array of a numeric list / array, else None."""
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None


# ---- analytic curves ----
def is_wavelength_xy(result):
    """True for an 'xy' result versus vacuum wavelength (x_label 'Wavelength', x_unit 'm')."""
    return (isinstance(result, dict) and result.get('kind') == 'xy' and result.get('x_label') == 'Wavelength'
            and result.get('x_unit') == 'm')


def curve_values(expression, parameters, variables, functions=None):
    """float64 values of a curve expression over the parameters and `variables` (which shadow parameters of the same
    name), broadcast to the variables' shape; functions: {name: callable} of extra calls (derived-curve calculus)."""
    shape = np.broadcast_shapes(*(np.shape(value) for value in variables.values()))
    arity = None if functions is None else {name: CURVE_FUNCTION_ARITY[name] for name in functions}
    result = compile_expression(expression, arity).evaluate({**parameters, **variables}, functions=functions)
    return np.array(np.broadcast_to(np.asarray(result, dtype=np.float64), shape))


def dense_axis(samples, n=CURVE_SAMPLES):
    """n points spanning the finite range of samples (empty when none is finite)."""
    values = np.asarray(samples, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return np.linspace(finite.min(), finite.max(), n) if finite.size else np.empty(0)


def sample_axis(samples, n_min=CURVE_SAMPLES, n_max=MAX_CURVE_SAMPLES):
    """The sorted distinct finite samples when n_min <= their count <= n_max, else n_min (fewer) or n_max (more)
    points spanning their range: a ring-down sampled like the result does not alias."""
    values = np.asarray(samples, dtype=np.float64)
    finite = np.unique(values[np.isfinite(values)])
    if n_min <= finite.size <= n_max:
        return finite
    return dense_axis(finite, n_min if finite.size < n_min else n_max)


def is_iteration_series(times):
    """True when every sample of a timeseries shares one t (stationary iterations: plotted vs the 1-based index)."""
    times = np.asarray(times, dtype=np.float64)
    return times.size > 0 and (times.size == 1 or float(np.ptp(times)) == 0.0)


def curve_variables(target_kind, series=''):
    """Variables of a curve expression on target_kind (result kind, 'line' or 'sweep') and series."""
    if target_kind == 'timeseries' and series == FFT_SERIES:
        return FFT_VARIABLES
    return CURVE_VARIABLES[target_kind]


def feature_curve(curve, result, parameters, wavelength=None, frequency=None):
    """(abscissa in the result's units, values) of an AnalyticCurveSpec over the sampled range of a fetched 1D result.
    Far field: angle [rad] at `wavelength` [m] (None: the first); timeseries series 'fft': on `frequency` [Hz]."""
    kind = result.get('kind') if isinstance(result, dict) else None
    if kind == 'timeseries' and curve.series == FFT_SERIES:
        if frequency is None:
            raise ValueError("a curve with series 'fft' needs the probe FFT frequencies (drawn on the FFT plot only)")
        axis = np.asarray(frequency, dtype=np.float64)
        return axis, curve_values(curve.expression, parameters, {'frequency': axis})
    if kind == 'timeseries' and is_iteration_series(result['t']):
        axis = sample_axis(np.arange(1, np.size(result['t']) + 1))
    elif kind == 'timeseries':
        axis = sample_axis(result['t'])
    elif kind == 'table':
        axis = dense_axis(table_xy(result, curve.x_column)['x'])
    elif kind == 'flux_spectrum':
        axis = dense_axis(result['wavelength'])
    elif kind == 'far_field':
        axis = dense_axis(result['angles'])
    elif kind == 'xy':
        axis = dense_axis(result['x'])
    else:
        raise ValueError(f"no analytic curve on a {kind} result; valid result kinds: {', '.join(CURVE_RESULT_KINDS)}")
    return axis, curve_values(curve.expression, parameters, _axis_variables(kind, result, axis, wavelength))


def _axis_variables(kind, result, axis, wavelength=None):
    """Curve variables on a fetched 1D result at abscissa values axis (far field: at wavelength [m], None = the
    first)."""
    if kind == 'timeseries' and is_iteration_series(result['t']):
        return {'iteration': axis, 't': float(np.asarray(result['t'], dtype=np.float64)[0])}
    if kind == 'timeseries':
        return {'t': axis}
    if kind == 'table':
        return {'x': axis}
    if kind == 'flux_spectrum':
        return {'wavelength': axis}
    if kind == 'far_field':
        if wavelength is None:
            wavelength = float(np.atleast_1d(result['wavelength'])[0])
        return {'angle': axis, 'wavelength': float(wavelength)}
    return {'x': axis, 'wavelength': axis} if is_wavelength_xy(result) else {'x': axis}


def sweep_curve(expression, spec, n=SWEEP_CURVE_SAMPLES):
    """(swept values, curve values) on the swept values plus n linear and, for a positive range, n geometric points
    (smooth on linear and log axes, exact at the simulated points); every parameter is re-evaluated with the swept one
    at each point (NaN where that fails)."""
    sweep = spec.study.sweep
    if sweep is None:
        raise ValueError('the study has no sweep')
    base = evaluate_parameters(spec.parameters, errors={})
    swept = np.asarray([evaluate_scalar(value, base) for value in sweep.point_values()], dtype=np.float64)
    finite = swept[np.isfinite(swept)]
    parts = [finite, dense_axis(finite, n)]
    if finite.size and finite.min() > 0.0:
        parts.append(np.geomspace(finite.min(), finite.max(), n))
    axis = np.unique(np.concatenate(parts))
    return axis, sweep_curve_values(expression, spec, axis)


def sweep_curve_values(expression, spec, swept_values):
    """float64 curve values at each swept value (every parameter re-evaluated with the swept one; NaN where that
    fails)."""
    compiled = compile_expression(expression)
    compiled.check_names(evaluate_parameters(spec.parameters, errors={}))
    values = np.full(len(swept_values), np.nan)
    for index, value in enumerate(swept_values):
        try:
            values[index] = evaluate_scalar(compiled, parameters_with(spec, spec.study.sweep.parameter, value))
        except ExpressionError:
            continue
    return values


SEGMENT_3D_LENGTH = 6               # AnalyticCurveSpec.line [x0, y0, z0, x1, y1, z1]: the 3D-segment form
SEGMENT_SAMPLINGS = ('nearest', 'trilinear')   # AnalyticCurveSpec.sampling (trilinear: 3D segments only)
_MAX_SEGMENT_SAMPLES = 4000         # as viz.common line_profile


def is_segment_3d(line):
    """True for the 3D-segment form of an AnalyticCurveSpec.line (six coordinates, 3D models only)."""
    return isinstance(line, (list, tuple)) and len(line) == SEGMENT_3D_LENGTH


def line_endpoints(line, parameters):
    """((x0, y0), (x1, y1)) in m of an AnalyticCurveSpec.line, ((x0, y0, z0), (x1, y1, z1)) for the 3D-segment form;
    ValueError (ExpressionError included) otherwise."""
    if not isinstance(line, (list, tuple)) or len(line) not in (4, SEGMENT_3D_LENGTH):
        raise ValueError(f'line must be [x0, y0, x1, y1] or (3D) [x0, y0, z0, x1, y1, z1] in m, got {line!r}')
    coordinates = [evaluate_scalar(value, parameters) for value in line]
    half = len(coordinates) // 2
    start, end = tuple(coordinates[:half]), tuple(coordinates[half:])
    if start == end:
        raise ValueError('the line starts and ends at the same point')
    return start, end


def line_curve(expression, parameters, start, end, n=CURVE_SAMPLES, z=None):
    """(distance s from start [m], values) of a curve of x, y (3D segment: z) [m] and s along start -> end (z: the
    cut height [m] of a 2D line in a 3D model, None otherwise)."""
    distance = np.linspace(0.0, 1.0, n) * math.dist(start, end)
    return distance, line_curve_values(expression, parameters, start, end, distance, z)


def line_curve_values(expression, parameters, start, end, distance, z=None):
    """float64 values of a line curve at the distances [m] from start along start -> end (variables x, y, s; z of a
    3D segment, else z when given)."""
    distance = np.asarray(distance, dtype=np.float64)
    fraction = distance / math.dist(start, end)
    variables = {name: low + fraction * (high - low) for name, low, high in zip('xyz', start, end)}
    variables['s'] = distance
    return curve_values(expression, parameters, _with_height(variables, z))


def line_cell_curve_values(expression, parameters, start, end, centres, z=None):
    """float64 values of a line curve at the sampled cell centres [m] (centres: x, y (3D segment: z) arrays; NaN
    stays NaN): the centre's coordinates, s = its projection onto start -> end, so a containing-cell profile is
    compared without a sub-cell offset; z when given."""
    centres = [np.asarray(values, dtype=np.float64) for values in centres]
    length = math.dist(start, end)
    projection = sum((values - low) * (high - low) for values, low, high in zip(centres, start, end)) / length
    variables = dict(zip('xyz', centres))
    variables['s'] = projection
    return curve_values(expression, parameters, _with_height(variables, z))


def _with_height(variables, z):
    """Line-curve variables plus z [m] (the 3D cut height of a 2D line) when it is not None."""
    return variables if z is None else {**variables, 'z': float(z)}


def segment_cells(start, end, edges):
    """(distance from start [m], inside mask, per-axis cell indices) of samples along start -> end [m] (one
    coordinate per axis of edges = the window cell edges [m] per axis), two per finest cell (at most 4000); each
    sample's containing cell as pos_to_cell (1e-9-cell snap, an edge goes to the upper cell), clamped."""
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    edges = [np.asarray(axis_edges, dtype=np.float64) for axis_edges in edges]
    length = math.dist(start, end)
    finest_cell = min(float(np.diff(axis_edges).min()) for axis_edges in edges)
    n_samples = int(np.clip(2.0 * length / finest_cell, 2, _MAX_SEGMENT_SAMPLES)) if finest_cell > 0 else 2
    fraction = np.linspace(0.0, 1.0, n_samples)
    inside = np.ones(n_samples, dtype=bool)
    cells = []
    for low, high, axis_edges in zip(start, end, edges):
        coordinates = low + fraction * (high - low)
        inside &= (coordinates >= axis_edges[0]) & (coordinates <= axis_edges[-1])
        snap = 1e-9 * float(np.diff(axis_edges).min())
        cells.append(np.clip(np.searchsorted(axis_edges, coordinates + snap, side='right') - 1, 0,
                             axis_edges.size - 2))
    return fraction * length, inside, tuple(cells)


def segment_trilinear_weights(start, end, distance, centres):
    """Per axis (lower cell, upper cell, upper weight in [0, 1]) of samples at `distance` [m] along start -> end
    between the window cell centres [m] of that axis; constant beyond the outermost centres (half cells at the edge)."""
    length = math.dist(start, end)
    weights = []
    for low, high, axis_centres in zip(start, end, centres):
        axis_centres = np.asarray(axis_centres, dtype=np.float64)
        coordinates = low + np.asarray(distance, dtype=np.float64) / length * (high - low)
        if axis_centres.size < 2:
            zeros = np.zeros(coordinates.size, dtype=np.intp)
            weights.append((zeros, zeros, np.zeros(coordinates.size)))
            continue
        lower = np.clip(np.searchsorted(axis_centres, coordinates, side='right') - 1, 0, axis_centres.size - 2)
        upper_weight = np.clip((coordinates - axis_centres[lower]) / (axis_centres[lower + 1] - axis_centres[lower]),
                               0.0, 1.0)
        weights.append((lower, lower + 1, upper_weight))
    return tuple(weights)


def field_plane_height(field_data):
    """z [m] of a FieldData's 3D cut plane (its line-curve variable z), None in 2D."""
    plane = getattr(field_data, 'plane', None)
    return None if plane is None else float(plane[1])


def line_plane(curve, parameters):
    """Session.field plane ('z', z [m]) of a line curve's `plane`, None when unset (the caller's default cut)."""
    return None if curve.plane is None else ('z', evaluate_scalar(curve.plane, parameters))


def snapshot_index(snapshot, labels):
    """Index into the snapshots (labels: one per snapshot, None for time snapshots) of a line curve's `snapshot`: an
    int (negative from the end) or a label, matched whole or by its text before ':' (e.g. 'resonance 2'), else by
    that text without its trailing mirror parity tag ('resonance 2 [x_lo symmetric]'); ValueError when no label or
    several match."""
    if isinstance(snapshot, numbers.Integral) and not isinstance(snapshot, bool):
        if not -len(labels) <= snapshot < len(labels):
            raise ValueError(f'snapshot {snapshot} out of range for {len(labels)} snapshots')
        return int(snapshot) % len(labels)
    wanted = str(snapshot).strip()
    matches = [index for index, label in enumerate(labels)
               if label is not None and wanted in (label.strip(), label.split(':')[0].strip())]
    if not matches:
        matches = [index for index, label in enumerate(labels)
                   if label is not None and wanted == _PARITY_TAG.sub('', label.split(':')[0]).strip()]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"snapshot {wanted!r} matches {len(matches)} snapshots (indices "
                         f"{', '.join(str(index) for index in matches)}): give the index instead")
    named = ', '.join(label.split(':')[0] for label in labels if label is not None)
    raise ValueError(f"no snapshot labelled {wanted!r}; labels: {named or 'none (time snapshots: use an index)'}")


# ---- derived curves ----
DERIVED_CURVE_VARIABLES = ('x',)   # besides the input aliases and the parameters
# derived-curve-only calls (not FUNCTION_NAMES: a field expression has no curve abscissa); (min args, max args)
CURVE_FUNCTION_ARITY = {'gradient': (1, 1), 'cumtrapz': (1, 1)}
CURVE_FUNCTION_DESCRIPTIONS = {   # describe('spec') DerivedCurveSpec.expression
    'gradient': 'gradient(y): dy/dx along the curve abscissa x (second-order differences, also at the ends), over '
                'the finite samples of y (NaN ones stay NaN)',
    'cumtrapz': 'cumtrapz(y): running trapezoid integral of y over x, 0 at the first finite sample; NaN samples '
                "(outside an interpolated input's range) are skipped and stay NaN, so it integrates over the overlap",
}
_ABSCISSA_TEXT = {'timeseries': ('Time', 's'), 'flux_spectrum': ('Wavelength', 'm'), 'far_field': ('Angle', 'rad'),
                  'table': ('row', '')}


def curve_functions(axis):
    """{name: callable} of CURVE_FUNCTION_ARITY on the sorted curve abscissa axis (float64 (n,)); arguments broadcast
    to it."""
    axis = np.asarray(axis, dtype=np.float64)

    def finite_part(values, name):
        values = np.array(np.broadcast_to(np.asarray(values, dtype=np.float64), axis.shape))
        finite = np.isfinite(values)
        if np.any(np.diff(axis[finite]) <= 0.0):
            raise ValueError(f'{name}() needs distinct abscissa values (repeated x on the curve abscissa)')
        return values, finite

    def gradient(values):
        values, finite = finite_part(values, 'gradient')
        result = np.full(axis.shape, math.nan)
        n_finite = int(finite.sum())
        if n_finite >= 2:
            result[finite] = np.gradient(values[finite], axis[finite], edge_order=2 if n_finite >= 3 else 1)
        return result

    def cumtrapz(values):
        values, finite = finite_part(values, 'cumtrapz')
        result = np.full(axis.shape, math.nan)
        x, y = axis[finite], values[finite]
        if x.size:
            result[finite] = np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])
        return result

    return {'gradient': gradient, 'cumtrapz': cumtrapz}


def split_curve_input(reference):
    """('physics.feature', sample name) of a derived curve input 'physics.feature:sample'; ValueError otherwise."""
    key, separator, sample = reference.partition(':') if isinstance(reference, str) else ('', '', '')
    if not separator or not sample:
        raise ValueError(f"expected 'physics.feature:sample', got {reference!r}")
    split_key(key)
    return key, sample


def derived_curve(curve, fetch, parameters):
    """(x, values, x label, x unit) of a DerivedCurveSpec: its expression over the input samples, x and parameters on
    the abscissa of the first array input (sorted), with the curve_functions calls; later inputs are interpolated
    linearly onto it (NaN outside their range), scalar samples enter as numbers. fetch(key) returns the 1D result dict
    of 'physics.feature' (far_field: its first wavelength)."""
    if not curve.inputs:
        raise ValueError('the curve has no inputs')
    variables, axis, axis_text = {}, None, None
    for alias, reference in curve.inputs.items():
        key, sample = split_curve_input(reference)
        result = fetch(key)
        abscissa_name, abscissa, samples = result_samples(result)
        if sample not in samples or sample == abscissa_name:
            raise ValueError(f"input {alias}: {key} has no sample {sample!r}; samples: "
                             f"{', '.join(name for name in samples if name != abscissa_name)}")
        values = samples[sample]
        if np.ndim(values) == 0:
            variables[alias] = float(values)
            continue
        iterations = result['kind'] == 'timeseries' and is_iteration_series(abscissa)
        if iterations:   # stationary iterations share one t: the 1-based iteration, as feature_curve
            abscissa = np.arange(1.0, abscissa.size + 1.0)
        order = np.argsort(abscissa, kind='stable')
        sorted_abscissa, sorted_values = abscissa[order], np.asarray(values, dtype=np.float64)[order]
        text = ((result.get('x_label') or 'x', result.get('x_unit', '')) if result['kind'] == 'xy'
                else ('Iteration', '') if iterations else _ABSCISSA_TEXT[result['kind']])
        if axis is None:
            axis, axis_text = sorted_abscissa, text
            variables[alias] = sorted_values
        elif text != axis_text:
            raise ValueError(f'input {alias}: abscissa {text[0]} [{text[1]}] differs from the first input\'s '
                             f'{axis_text[0]} [{axis_text[1]}]')
        else:
            variables[alias] = np.interp(axis, sorted_abscissa, sorted_values, left=math.nan, right=math.nan)
    if axis is None:
        raise ValueError('every input is a scalar sample: the curve has no abscissa')
    values = curve_values(curve.expression, parameters, {**variables, 'x': axis}, functions=curve_functions(axis))
    return axis, values, *axis_text


def derived_curve_result(curve, fetch, parameters):
    """'xy' result of one DerivedCurveSpec (series = its name), the source of a derived value naming the curve."""
    x, values, x_label, x_unit = derived_curve(curve, fetch, parameters)
    return {'kind': 'xy', 'x': x, 'x_label': x_label, 'x_unit': x_unit, 'y_label': curve.label or curve.name,
            'series': {curve.name: values}, 'series_units': {curve.name: curve.unit}}


def derived_curve_reference(curve, x, values, parameters):
    """(reference values on x, curve_error record) of a DerivedCurveSpec with a reference (expression of x and
    parameters); a failing reference: (None, uncompared record with a note)."""
    expression = curve.reference if isinstance(curve.reference, str) else repr(float(curve.reference))
    try:
        reference = curve_values(expression, parameters, {'x': np.asarray(x, dtype=np.float64)})
    except (ValueError, KeyError, TypeError) as exc:   # ExpressionError included
        return None, {'max': math.nan, 'rms': math.nan, 'n_points': 0, 'note': f'reference: {error_message(exc)}'}
    return reference, curve_error(values, reference)


def derived_curve_plots(curves, fetch, parameters):
    """[(plot name, xy result or None, {curve name: error text})] of DerivedCurveSpecs grouped by plot (first-seen
    order; plot '' = a plot of its own named after the curve): series = labels (else names) with their unit, on the
    first drawn curve's abscissa (others interpolated) and axes (keys log_x / log_y), curve_series = {curve name:
    series name}, a curve's reference as series '<series> reference' (reference_series {curve name: series name},
    curve_error {curve name: record} on the curve's own abscissa), parameters = the values used (written with an
    export); failing curves are left out."""
    groups = {}
    for curve in curves:
        groups.setdefault(curve.plot or curve.name, []).append(curve)
    plots = []
    for plot, members in groups.items():
        axis, series, units, errors, first, curve_series = None, {}, {}, {}, None, {}
        reference_series, curve_errors = {}, {}
        for curve in members:
            try:
                abscissa, values, label, unit = derived_curve(curve, fetch, parameters)
                reference = None
                if curve.reference is not None:
                    reference, curve_errors[curve.name] = derived_curve_reference(curve, abscissa, values, parameters)
                if axis is None:
                    axis, first = (abscissa, label, unit), curve
                elif (label, unit) != axis[1:]:
                    curve_errors.pop(curve.name, None)
                    raise ValueError(f"abscissa {label} [{unit}] differs from the plot's {axis[1]} [{axis[2]}]")
                else:
                    values = np.interp(axis[0], abscissa, values, left=math.nan, right=math.nan)
                    if reference is not None:
                        reference = np.interp(axis[0], abscissa, reference, left=math.nan, right=math.nan)
            except (ValueError, KeyError, TypeError) as exc:   # text only: one curve must not cost the plot
                errors[curve.name] = error_message(exc)
                continue
            name = _unique_series_name(curve, series)
            series[name], units[name], curve_series[curve.name] = values, curve.unit, name
            if reference is not None:
                reference_name = _free_series_name(f'{name} reference', series)
                series[reference_name], units[reference_name] = reference, curve.unit
                reference_series[curve.name] = reference_name
        result = None if axis is None else {
            'kind': 'xy', 'x': axis[0], 'x_label': axis[1], 'x_unit': axis[2], 'y_label': '', 'series': series,
            'series_units': units, 'log_x': bool(first.log_x), 'log_y': bool(first.log_y),
            'curve_series': curve_series, 'reference_series': reference_series, 'curve_error': curve_errors,
            'parameters': dict(parameters)}
        plots.append((plot, result, errors))
    return plots


def _unique_series_name(curve, series):
    """Series name of a derived curve in its plot: the label, else (or when taken) the curve name, else
    '<label or name> (<curve name>)' and then a counter, so no curve replaces another's series."""
    candidates = [name for name in (curve.label, curve.name) if name]
    for name in candidates:
        if name not in series:
            return name
    base = f'{candidates[0]} ({curve.name})' if candidates[0] != curve.name else curve.name
    return _free_series_name(base, series)


def _free_series_name(base, series):
    """base, else '<base> 2', '<base> 3', ... : the first name not in series."""
    name, count = base, 2
    while name in series:
        name, count = f'{base} {count}', count + 1
    return name


# ---- curve errors (summary.json curve_error) ----
def curve_error(simulated, analytic, per_point=False):
    """{max, rms, n_points} of |simulated - analytic| on the points where both are finite, relative to the analytic
    range ptp (an absolute T against its rise, not ~300 K; a flat curve: max |analytic|, 0: absolute); per_point (sweep
    curves spanning decades): each point relative to |analytic| there. NaN with n_points 0 when no point is finite."""
    simulated = np.asarray(simulated, dtype=np.float64).ravel()
    analytic = np.asarray(analytic, dtype=np.float64).ravel()
    finite = np.isfinite(simulated) & np.isfinite(analytic)
    if not finite.any():
        return {'max': math.nan, 'rms': math.nan, 'n_points': 0}
    reference = analytic[finite]
    magnitude = np.abs(reference)
    if per_point:   # floored so a point where the analytic crosses 0 gives a large finite error, not inf
        scale = np.maximum(magnitude, _PER_POINT_FLOOR * float(magnitude.max())) if magnitude.max() > 0.0 else 1.0
    else:
        scale = float(np.ptp(reference)) or float(magnitude.max()) or 1.0
    relative = np.abs(simulated[finite] - reference) / scale
    return {'max': float(relative.max()), 'rms': float(np.sqrt(np.mean(relative ** 2))), 'n_points': int(finite.sum())}


def feature_curve_samples(curve, result, parameters):
    """(simulated, analytic) float64 of a curve at the samples of its fetched 1D result: timeseries values, flux
    power (or the sample named by series), far field power per wavelength over its peak (the curve likewise), the xy
    series / table column named by series ('' = the only one); ValueError when no simulated quantity matches."""
    kind = result.get('kind') if isinstance(result, dict) else None
    series = curve.series
    if series in (FFT_SERIES, RATIO_SERIES):
        raise ValueError(f"series '{series}' is drawn on the FFT / ratio-to-reference plot only: not compared")
    if kind == 'timeseries':
        times = np.asarray(result['t'], dtype=np.float64)
        axis = np.arange(1.0, times.size + 1.0) if is_iteration_series(times) else times
        simulated = result['values']
    elif kind == 'flux_spectrum':
        name = series or 'power'
        if name not in RESULT_SAMPLE_NAMES['flux_spectrum'][1] or name == 'wavelength':
            raise ValueError(f"series {name!r} is no flux_spectrum quantity (power, power_per_cw, frequency)")
        axis, simulated = np.asarray(result['wavelength'], dtype=np.float64), result[name]
    elif kind == 'far_field':
        angles = np.asarray(result['angles'], dtype=np.float64)
        powers = np.atleast_2d(np.asarray(result['power'], dtype=np.float64))
        rows = [(_peak_normalised(power), _peak_normalised(curve_values(
            curve.expression, parameters, _axis_variables(kind, result, angles, wavelength))))
            for power, wavelength in zip(powers, np.atleast_1d(result['wavelength']))]
        return np.concatenate([row[0] for row in rows]), np.concatenate([row[1] for row in rows])
    elif kind in ('xy', 'table'):
        xy = result if kind == 'xy' else table_xy(result, curve.x_column)
        names = [str(name) for name in xy['series']]
        if not series and len(names) != 1:
            raise ValueError(f"series '' is drawn with every series ({', '.join(names) or 'none'}): name one to "
                             f'compare')
        name = series or names[0]
        if name not in names:
            raise ValueError(f"the result has no series {name!r}; series: {', '.join(names) or 'none'}")
        axis, simulated = np.asarray(xy['x'], dtype=np.float64), xy['series'][name]
    else:
        raise ValueError(f"no analytic curve on a {kind} result; valid result kinds: {', '.join(CURVE_RESULT_KINDS)}")
    analytic = curve_values(curve.expression, parameters, _axis_variables(kind, result, axis))
    return np.asarray(simulated, dtype=np.float64), analytic


def _peak_normalised(values):
    """values / their largest finite value (unchanged when that is not > 0), as the far-field plots normalise."""
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    peak = float(finite.max()) if finite.size else 0.0
    return values / peak if peak > 0.0 else values


def sweep_simulated(series, sweep):
    """float64 per sweep value of the quantity a 'sweep' curve of series belongs to (sweep: the study summary table):
    a derived value, the final sample of a 'physics.feature' timeseries or a '<column>_<k>' table band (sorted, as
    sweep_table_xy); series '' only with exactly one derived value; ValueError otherwise."""
    derived = sweep.get('derived') or {}
    if series in derived or (not series and len(derived) == 1):
        return np.asarray(derived[series or next(iter(derived))], dtype=np.float64)
    if not series:
        raise ValueError(f"series '' is drawn with every sweep quantity ({len(derived)} derived values): name one to "
                         f'compare')
    n_values = len(sweep['values'])
    per_value = [results or {} for results in list(sweep.get('results') or [])[:n_values]]
    per_value += [{}] * (n_values - len(per_value))
    finals = [result.get(series) for result in per_value]
    if any(isinstance(result, dict) and result.get('kind') == 'timeseries' for result in finals):
        return np.array([float(np.asarray(result['values'], dtype=np.float64)[-1])
                         if isinstance(result, dict) and np.size(result.get('values', [])) else math.nan
                         for result in finals])
    column, _, band = series.rpartition('_')
    if column and band.isdigit():
        for key in dict.fromkeys(name for results in per_value for name in results):
            tables = [results.get(key) if (results.get(key) or {}).get('kind') == 'table' else None
                      for results in per_value]
            if any(table is not None and column in [str(name) for name in table['columns']] for table in tables):
                bands = sweep_table_xy(sweep['parameter'], sweep['values'], tables, column)['series']
                if series in bands:
                    return np.asarray(bands[series], dtype=np.float64)
    raise ValueError(f"series {series!r} is no derived value ({', '.join(derived) or 'none'}), timeseries result or "
                     f"'<column>_<k>' table band of the sweep")


def parameters_with(spec, name, value):
    """Parameters of spec with parameter `name` set to the number value (dependants re-evaluated; failures left
    out)."""
    return evaluate_parameters([(item.name, repr(float(value)) if item.name == name else item.expression)
                                for item in spec.parameters], errors={})
