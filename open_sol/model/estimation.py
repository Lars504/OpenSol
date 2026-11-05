"""Parameter estimation parts that need no model run (StudySpec.estimation): the resolved setup and its checks (shared
by validate and model.study_estimation), the weighted residuals of targets and measured curves, and fit statistics."""
import math
from dataclasses import dataclass

import numpy as np

from .analytic import derived_curve, is_curve_source, split_curve_input
from .expressions import error_message, evaluate_scalar
from .spec import DerivedCurveSpec, EstimatedParameterSpec, EstimationCurveSpec, EstimationTargetSpec

ESTIMATION_METHODS = ('least_squares', 'nelder_mead')
# summary estimation stop_reason values ('converged' alone means the optimizer met its tolerance)
STOP_REASONS = ('converged', 'max_evaluations', 'not_finite', 'failed', 'cancelled')
_MODEL_ALIAS = 'model_samples'   # derived-curve alias that reads a curve's 'physics.feature:sample' source
_EDGE_SLACK = 1e-9               # measured x may lie this fraction of the model's span beyond its ends


@dataclass
class MeasuredTarget:
    """One EstimationTargetSpec evaluated: derived value name, measured value and sigma."""
    derived: str
    value: float
    sigma: float


@dataclass
class MeasuredCurve:
    """One EstimationCurveSpec evaluated: source, measured abscissa / values and per-point sigma (float64 (n,))."""
    source: str
    x: np.ndarray
    y: np.ndarray
    sigma: np.ndarray


@dataclass
class ResolvedEstimation:
    """StudySpec.estimation with every value evaluated over the base parameters: names, start, bounds (+-inf when
    unset) and scale per parameter (float64 (n,)), targets, curves, settings; absolute_sigma = every sigma was given."""
    names: list
    initial: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    scale: np.ndarray
    targets: list
    curves: list
    method: str
    max_evaluations: int
    diff_step: float
    tolerance: float
    absolute_sigma: bool

    @property
    def n_residuals(self) -> int:
        return len(self.targets) + sum(curve.x.size for curve in self.curves)

    def values_of(self, scaled) -> np.ndarray:
        """Parameter values of the search vector z (z = 1 at the start: value = initial + (z - 1) scale)."""
        return self.initial + (np.asarray(scaled, dtype=np.float64) - 1.0) * self.scale

    def scaled_bounds(self) -> tuple:
        """(lower, upper) bounds of the search vector (+-inf where unbounded)."""
        return 1.0 + (self.lower - self.initial) / self.scale, 1.0 + (self.upper - self.initial) / self.scale


def resolve_estimation(spec, parameters) -> tuple:
    """(ResolvedEstimation or None, [error text], [warning text]) of spec.study.estimation over the base parameters
    {name: value}; None whenever an error is listed."""
    estimation = spec.study.estimation
    errors, warnings = [], []

    def number(raw, what, positive=False):
        """float of a number / parameter expression; None after recording an error."""
        try:
            value = evaluate_scalar(raw, parameters)
        except (ValueError, TypeError, KeyError) as exc:   # ExpressionError included
            errors.append(f'{what}: {error_message(exc)}')
            return None
        if positive and not value > 0.0:
            errors.append(f'{what} must be > 0, got {value:g}')
            return None
        return value

    names, initial, lower, upper, scale = _resolved_parameters(spec, estimation.parameters, parameters, number, errors)
    targets = _resolved_targets(spec, estimation.targets, number, errors)
    curves = _resolved_curves(spec, estimation.curves, number, errors)
    if estimation.method not in ESTIMATION_METHODS:
        errors.append(f"method {estimation.method!r} is not one of {', '.join(ESTIMATION_METHODS)}")
    max_evaluations = estimation.max_evaluations
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or max_evaluations < 1:
        errors.append(f'max_evaluations must be a whole number >= 1, got {max_evaluations!r}')
    for name in ('diff_step', 'tolerance'):
        value = getattr(estimation, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not value > 0.0:
            errors.append(f'{name} must be a number > 0, got {value!r}')
    if isinstance(estimation.targets, list) and isinstance(estimation.curves, list) \
            and not estimation.targets and not estimation.curves:
        errors.append('add a target (a measured derived value) or a curve (measured samples) to fit')
    if errors:
        return None, errors, warnings
    absolute_sigma = (all(target.sigma is not None for target in estimation.targets)
                      and all(curve.sigma is not None for curve in estimation.curves))
    resolved = ResolvedEstimation(names, np.array(initial), np.array(lower), np.array(upper), np.array(scale), targets,
                                  curves, estimation.method, int(max_evaluations), float(estimation.diff_step),
                                  float(estimation.tolerance), absolute_sigma)
    if resolved.n_residuals < len(names):
        warnings.append(f'{resolved.n_residuals} measured values for {len(names)} parameters: the parameters are not '
                        'all determined by the data')
    if resolved.method == 'least_squares' and resolved.max_evaluations < len(names) + 2:
        warnings.append(f'max_evaluations {resolved.max_evaluations} leaves no step after the first finite-difference '
                        f'Jacobian ({len(names) + 1} runs)')
    return resolved, errors, warnings


def _resolved_parameters(spec, items, parameters, number, errors) -> tuple:
    """(names, initial, lower, upper, scale) lists of the estimated parameters; problems go to errors."""
    resolved = ([], [], [], [], [])
    if not isinstance(items, list) or not items:
        errors.append('list at least one model parameter to estimate under parameters')
        return resolved
    known = [parameter.name for parameter in spec.parameters]
    for index, item in enumerate(items):
        if not isinstance(item, EstimatedParameterSpec):
            errors.append(f'parameter {index + 1} is not an object with name, initial, lower, upper')
            continue
        if item.name not in known:
            errors.append(f"parameter {index + 1}: {item.name!r} is not a model parameter; parameters: "
                          f"{', '.join(known) or 'none'}")
            continue
        if item.name in resolved[0]:
            errors.append(f"parameter '{item.name}' is listed twice")
            continue
        where = f"parameter '{item.name}'"
        if item.initial is None:
            start = parameters.get(item.name)
            if start is None:
                errors.append(f'{where}: its model value does not evaluate; give initial')
        else:
            start = number(item.initial, f'{where} initial')
        low = -math.inf if item.lower is None else number(item.lower, f'{where} lower')
        high = math.inf if item.upper is None else number(item.upper, f'{where} upper')
        if start is None or low is None or high is None:
            continue
        if not low < high:
            errors.append(f'{where}: lower {low:g} must be below upper {high:g}')
            continue
        if not low < start < high:
            errors.append(f'{where}: initial {start:g} must lie strictly between lower {low:g} and upper {high:g}')
            continue
        width = high - low
        for column, value in zip(resolved, (item.name, start, low, high,
                                            abs(start) if start != 0.0 else width if math.isfinite(width) else 1.0)):
            column.append(value)
    return resolved


def _resolved_targets(spec, items, number, errors) -> list:
    """[MeasuredTarget] of the targets (sigma None -> |value|, 1 when value is 0); problems go to errors."""
    if not isinstance(items, list):
        errors.append('targets must be a list')
        return []
    derived_names = [derived.name for derived in spec.derived_values]
    targets = []
    for index, item in enumerate(items):
        if not isinstance(item, EstimationTargetSpec):
            errors.append(f'target {index + 1} is not an object with derived, value, sigma')
            continue
        where = f'target {index + 1}'
        if item.derived not in derived_names:
            errors.append(f"{where}: {item.derived!r} is not a derived value; derived values: "
                          f"{', '.join(derived_names) or 'none'}")
            continue
        value = number(item.value, f'{where} value')
        sigma = None if item.sigma is None else number(item.sigma, f'{where} sigma', positive=True)
        if value is None or (item.sigma is not None and sigma is None):
            continue
        if sigma is None:
            sigma = abs(value) if value != 0.0 else 1.0
        targets.append(MeasuredTarget(item.derived, value, sigma))
    return targets


def _resolved_curves(spec, items, number, errors) -> list:
    """[MeasuredCurve] of the curves (sigma None -> RMS of y, 1 when y is all 0); problems go to errors."""
    if not isinstance(items, list):
        errors.append('curves must be a list')
        return []
    curves = []
    for index, item in enumerate(items):
        if not isinstance(item, EstimationCurveSpec):
            errors.append(f'curve {index + 1} is not an object with source, x, y, sigma')
            continue
        where = f'curve {index + 1}'
        problem = curve_source_problem(spec, item.source)
        if problem:
            errors.append(f'{where}: {problem}')
        x, y = _number_array(item.x, f'{where} x', errors), _number_array(item.y, f'{where} y', errors)
        if x is None or y is None or problem:
            continue
        if x.size == 0 or x.size != y.size:
            errors.append(f'{where}: x and y need the same number of samples (>= 1), got {x.size} and {y.size}')
            continue
        if item.sigma is None:
            rms = float(np.sqrt(np.mean(y ** 2)))
            sigma = np.full(y.size, rms if rms > 0.0 else 1.0)
        elif isinstance(item.sigma, list):
            sigma = _number_array(item.sigma, f'{where} sigma', errors)
            if sigma is None:
                continue
            if sigma.size != y.size or not np.all(sigma > 0.0):
                errors.append(f'{where}: sigma needs one value > 0 per sample ({y.size})')
                continue
        else:
            value = number(item.sigma, f'{where} sigma', positive=True)
            if value is None:
                continue
            sigma = np.full(y.size, value)
        curves.append(MeasuredCurve(item.source, x, y, sigma))
    return curves


def curve_source_problem(spec, source) -> str:
    """Why a curve source names neither a derived curve of spec nor a 'physics.feature:sample' of a feature in it ('' =
    fine; whether the feature gives that sample shows at run time)."""
    if not isinstance(source, str) or not source:
        return "source must be 'physics.feature:sample' or a derived curve name"
    if is_curve_source(source):
        curve_names = [curve.name for curve in spec.derived_curves]
        if source not in curve_names:
            return (f"source {source!r} is not a derived curve (derived curves: {', '.join(curve_names) or 'none'}); "
                    "a feature result is 'physics.feature:sample'")
        return ''
    try:
        key, _ = split_curve_input(source)
    except ValueError as exc:
        return f'source: {exc}, or a derived curve name'
    physics_name, _, feature_name = key.partition('.')
    try:
        spec.physics_by_name(physics_name).feature(feature_name)
    except KeyError as exc:
        return f'source {source!r}: {error_message(exc)}'
    return ''


def _number_array(values, what, errors):
    """float64 array of a list of finite numbers; None after recording an error."""
    if not isinstance(values, list) or not all(isinstance(value, (int, float)) and not isinstance(value, bool)
                                               for value in values):
        errors.append(f'{what} must be a list of numbers')
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        errors.append(f'{what} holds a value that is not finite')
        return None
    return array


# ---- residuals and statistics (pure) ----
def curve_model(spec, source, fetch, parameters) -> tuple:
    """(abscissa, values) float64 of a curve source on a run: a derived curve of spec, or the 'physics.feature:sample'
    samples (sorted as a derived-curve input); fetch('physics.feature') -> result dict."""
    if is_curve_source(source):
        curve = spec.derived_curve(source)
    else:
        curve = DerivedCurveSpec(name=_MODEL_ALIAS, expression=_MODEL_ALIAS, inputs={_MODEL_ALIAS: source})
    abscissa, values, _, _ = derived_curve(curve, fetch, parameters)
    return np.asarray(abscissa, dtype=np.float64), np.asarray(values, dtype=np.float64)


def model_on_data(model_x, model_y, data_x) -> np.ndarray:
    """Model samples linearly interpolated onto the measured abscissa data_x (finite model samples only); ValueError
    when a measured x lies outside the model's abscissa range."""
    model_x, model_y = np.asarray(model_x, dtype=np.float64), np.asarray(model_y, dtype=np.float64)
    finite = np.isfinite(model_x) & np.isfinite(model_y)
    x, y = model_x[finite], model_y[finite]
    if x.size == 0:
        raise ValueError('the model curve has no finite samples')
    order = np.argsort(x, kind='stable')
    x, y = x[order], y[order]
    data_x = np.asarray(data_x, dtype=np.float64)
    slack = _EDGE_SLACK * max(float(x[-1] - x[0]), abs(float(x[-1])), 1e-300)
    outside = (data_x < x[0] - slack) | (data_x > x[-1] + slack)
    if np.any(outside):
        raise ValueError(f'{int(outside.sum())} measured x outside the model abscissa [{x[0]:g}, {x[-1]:g}]')
    if x.size == 1:
        return np.full(data_x.shape, float(y[0]))
    return np.interp(data_x, x, y)


def weighted_residuals(resolved, derived, curve_models) -> np.ndarray:
    """(model - measured) / sigma: the targets in order (derived {name: value}; NaN when missing), then the points of
    each curve (curve_models: [(abscissa, values)] per resolved curve)."""
    parts = [np.array([(float(derived.get(target.derived, math.nan)) - target.value) / target.sigma
                       for target in resolved.targets], dtype=np.float64)]
    for curve, (model_x, model_y) in zip(resolved.curves, curve_models):
        parts.append((model_on_data(model_x, model_y, curve.x) - curve.y) / curve.sigma)
    return np.concatenate(parts)


def fit_statistics(jacobian, residuals, scale, absolute_sigma) -> dict:
    """{rank, reduced_chi_square, standard_errors [per parameter], correlation [[...]], note} of the weighted residuals'
    Jacobian d r / d z (z = search vector, value = initial + (z - 1) scale): covariance (J^T J)^-1, times the reduced
    chi-square unless absolute_sigma; standard errors None (with a note) when rank-deficient or undetermined."""
    jacobian = np.atleast_2d(np.asarray(jacobian, dtype=np.float64))
    residuals = np.asarray(residuals, dtype=np.float64)
    n_residuals, n_parameters = jacobian.shape
    dof = n_residuals - n_parameters
    chi_square = float(residuals @ residuals)
    record = {'rank': None, 'reduced_chi_square': chi_square / dof if dof > 0 else None, 'standard_errors': None,
              'correlation': None, 'note': ''}
    if not np.all(np.isfinite(jacobian)):
        record['note'] = 'the Jacobian is not finite'
        return record
    _, singular, right = np.linalg.svd(jacobian, full_matrices=False)
    threshold = np.finfo(np.float64).eps * max(jacobian.shape) * (float(singular[0]) if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > threshold))
    record['rank'] = rank
    if rank < n_parameters:
        record['note'] = (f'the Jacobian has rank {rank} < {n_parameters} parameters: the data do not determine them '
                          'separately')
        return record
    covariance = (right.T / singular ** 2) @ right
    if not absolute_sigma:
        if dof <= 0:
            record['note'] = ('as many parameters as measured values and no sigma given: the residual variance (and '
                              'with it the standard errors) is unknown')
            return record
        covariance = covariance * record['reduced_chi_square']
    scale = np.asarray(scale, dtype=np.float64)
    covariance = covariance * np.outer(scale, scale)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation = covariance / np.outer(standard_errors, standard_errors)
    record['standard_errors'] = standard_errors.tolist()
    record['correlation'] = correlation.tolist()
    return record
