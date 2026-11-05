"""Self-tests of the parameter estimation (StudySpec.estimation): CPU checks of its parts (curve interpolation,
residuals, fit statistics, validation refusals, JSON round trip, renames) and 1D heat slab runs recovering the
conductivity (least_squares, Nelder-Mead) and a bounded two-parameter fit."""
import math

import numpy as np

from .schema import SelfTestCase
from .selftest_helpers import SubChecks
from .spec import (DerivedValueSpec, DomainSpec, EstimatedParameterSpec, EstimationCurveSpec, EstimationSpec,
                   EstimationTargetSpec, FeatureSpec, MaterialSpec, MeshSpec, ModelSpec, ParameterSpec, PhysicsSpec,
                   SolverDefaultsSpec, StudySpec, SweepSpec)

_LENGTH = 1e-3               # m, the 1D slab: x_lo insulated, x_hi held at _T_FACE
_CELL = 1e-5                 # m, uniform cells (100)
_T_FACE = 300.0              # K
_K_TRUE = 2.3                # W/(m K), the conductivity the fits recover
_Q_TRUE = 5e7                # W/m^3, the uniform source of the two-parameter fit
# the cell-centred chain gives max T_rise = Q L^2 / (2 k) exactly (x_lo cell: the half-cell shift cancels the
# parabola's offset there), so the synthetic data hold no discretisation error; float32 T ~1e-6 of the rise
_FIT_TOLERANCE = 1e-4        # relative error of the least_squares fits
_SIMPLEX_TOLERANCE = 1e-3    # relative error of the Nelder-Mead fit (its stop on the scaled step, 1e-6)


def _rise(source, conductivity):
    """Analytic max temperature rise Q L^2 / (2 k) [K] of the slab."""
    return source * _LENGTH ** 2 / (2.0 * conductivity)


def _slab_spec(estimation=None):
    """Stationary 1D heat slab (cpu, Automatic banded solver): uniform Q = Q_heat, k = k_fit, x_hi held; derived
    rise_max (max T_rise) and source_power (integral of Q: Q L)."""
    return ModelSpec(
        title='Self-test: parameter estimation on a 1D heat slab',
        parameters=[ParameterSpec('k_fit', 1.0), ParameterSpec('Q_heat', 1e8)],
        domain=DomainSpec(size_x=_LENGTH, size_y=None),
        materials=[MaterialSpec('solid', properties={'k': 'k_fit'}, selection=['all'])],
        physics=[PhysicsSpec('heat', 'heat', settings={'geometry': 'planar', 'T0': _T_FACE}, features=[
            FeatureSpec('source', 'heat_source', ['all'], {'Q': 'Q_heat'}),
            FeatureSpec('sink', 'temperature', ['x_hi'], {'temperature': _T_FACE})])],
        mesh=MeshSpec(mode='uniform', dx=_CELL),
        solver_defaults=SolverDefaultsSpec(backend='cpu'),
        study=StudySpec(type='stationary', estimation=estimation),
        derived_values=[DerivedValueSpec('rise_max', 'T_rise', [], 'max'),
                        DerivedValueSpec('source_power', 'heat_Q_ext', [], 'integral')])


def _conductivity_estimation(method, max_evaluations, tolerance):
    """k_fit from 1 in [0.1, 10] to the measured rise of k = _K_TRUE at Q_heat = 1e8 (relative sigma)."""
    return EstimationSpec(parameters=[EstimatedParameterSpec('k_fit', initial=1.0, lower=0.1, upper=10.0)],
                          targets=[EstimationTargetSpec('rise_max', _rise(1e8, _K_TRUE))], method=method,
                          max_evaluations=max_evaluations, tolerance=tolerance)


def _two_parameter_estimation():
    """k_fit and Q_heat (bounded) from the rise and the source power, sigma 1e-3 of each (absolute)."""
    rise, power = _rise(_Q_TRUE, _K_TRUE), _Q_TRUE * _LENGTH
    return EstimationSpec(
        parameters=[EstimatedParameterSpec('k_fit', initial=1.5, lower=1.0, upper=4.0),
                    EstimatedParameterSpec('Q_heat', initial=2e7, lower=1e7, upper=1e8)],
        targets=[EstimationTargetSpec('rise_max', rise, sigma=1e-3 * rise),
                 EstimationTargetSpec('source_power', power, sigma=1e-3 * power)],
        max_evaluations=40, tolerance=1e-6)


def _fit_check(expected, tolerance, need_errors=False):
    """check(results): the worst relative error of the fitted values {name: value} of the single run; error inf when
    the run did not complete, has no estimation record or (need_errors) no finite standard errors."""
    def check(results):
        (run,) = results.values()
        summary = run.summary()
        report = summary.get('estimation') or {}
        fitted = report.get('fitted') or {}
        errors = {name: abs(fitted[name] / value - 1.0) if name in fitted else math.inf
                  for name, value in expected.items()}
        worst = max(errors.values())
        standard_errors = report.get('standard_errors') or {}
        ok = summary.get('status') == 'completed' and report.get('stop_reason') != 'failed'
        if need_errors:
            ok = ok and all(isinstance(standard_errors.get(name), float) and math.isfinite(standard_errors[name])
                            and standard_errors[name] > 0.0 for name in expected)
        details = {'errors': errors, 'fitted': fitted, 'standard_errors': standard_errors,
                   'stop_reason': report.get('stop_reason'), 'evaluations': report.get('evaluations'),
                   'message': report.get('message'), 'status': summary.get('status')}
        return {'measured': worst, 'expected': 0.0, 'error': worst if ok else math.inf, 'tolerance': tolerance,
                'details': details}
    return check


def _check_parts(_results):
    """Every sub-check pass / fail; error = number of failed sub-checks, tolerance 0."""
    from .estimation import (curve_model, fit_statistics, model_on_data, resolve_estimation,
                             weighted_residuals)
    from .schema import get_registry
    from .validate import validate
    checks = SubChecks()
    add = checks.add

    # curve interpolation and its range check
    add('interpolate', np.allclose(model_on_data([2.0, 0.0, 1.0], [20.0, 0.0, 10.0], [0.5, 1.5]), [5.0, 15.0]))
    try:
        model_on_data([0.0, 1.0], [0.0, 1.0], [1.5])
        add('outside_range_raises', False)
    except ValueError:
        add('outside_range_raises', True)
    # a 'physics.feature:sample' source read like a derived-curve input
    result = {'kind': 'xy', 'x': [0.0, 1.0, 2.0], 'x_label': 'x', 'x_unit': 'm', 'series': {'s': [0.0, 10.0, 20.0]}}
    x, values = curve_model(ModelSpec(), 'p.f:s', {'p.f': result}.__getitem__, {})
    add('curve_source_samples', np.allclose(x, [0.0, 1.0, 2.0]) and np.allclose(values, [0.0, 10.0, 20.0]))
    # residuals: targets then curve points, relative / RMS default sigmas
    spec = _slab_spec(EstimationSpec(
        parameters=[EstimatedParameterSpec('k_fit', initial=2.0, lower=1.0, upper=4.0)],
        targets=[EstimationTargetSpec('rise_max', 10.0)],
        curves=[EstimationCurveSpec('heat.source:s', x=[0.5, 1.5], y=[3.0, 4.0])]))
    resolved, errors, _ = resolve_estimation(spec, {'k_fit': 1.0, 'Q_heat': 1e8})
    if resolved is None:
        add('resolve', False, errors)
    else:
        rms = math.sqrt(12.5)
        residuals = weighted_residuals(resolved, {'rise_max': 12.0}, [(x, values)])
        add('residuals', np.allclose(residuals, [0.2, (5.0 - 3.0) / rms, (15.0 - 4.0) / rms]), residuals.tolist())
        add('scaling', np.allclose(resolved.values_of([1.0]), [2.0]) and np.allclose(resolved.values_of([1.5]), [3.0])
            and np.allclose(resolved.scaled_bounds()[0], [0.5]) and not resolved.absolute_sigma)
    # fit statistics: J^T J = [[2, 1], [1, 5]], scale (1, 10), absolute sigma
    jacobian = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    statistics = fit_statistics(jacobian, np.zeros(3), [1.0, 10.0], True)
    expected_errors = [math.sqrt(5.0 / 9.0), 10.0 * math.sqrt(2.0 / 9.0)]
    add('standard_errors', statistics['standard_errors'] is not None
        and np.allclose(statistics['standard_errors'], expected_errors, rtol=1e-12)
        and math.isclose(statistics['correlation'][0][1], -1.0 / math.sqrt(10.0), rel_tol=1e-12), statistics)
    relative = fit_statistics(jacobian, np.array([1.0, 0.0, 0.0]), [1.0, 10.0], False)
    add('standard_errors_relative', relative['standard_errors'] is not None
        and np.allclose(relative['standard_errors'], expected_errors, rtol=1e-12), relative)   # chi^2 / dof = 1
    deficient = fit_statistics([[1.0, 2.0], [2.0, 4.0]], np.zeros(2), [1.0, 1.0], True)
    add('rank_deficient', deficient['rank'] == 1 and deficient['standard_errors'] is None, deficient)

    # validation refusals ('estimation' errors at the study node)
    registry = get_registry()

    def estimation_errors(model):
        return [issue.message for issue in validate(model, registry)
                if issue.level == 'error' and 'estimation' in issue.message]
    good = _slab_spec(_conductivity_estimation('least_squares', 10, 1e-6))
    add('valid_no_errors', not estimation_errors(good), estimation_errors(good))
    refusals = {
        'with_sweep': (lambda model: setattr(model.study, 'sweep', SweepSpec('Q_heat', [1e8, 2e8])),
                       'parameter sweep'),
        'unknown_parameter': (lambda model: setattr(model.study.estimation.parameters[0], 'name', 'nope'),
                              'is not a model parameter'),
        'initial_outside': (lambda model: setattr(model.study.estimation.parameters[0], 'initial', 20.0),
                            'strictly between'),
        'unknown_derived': (lambda model: setattr(model.study.estimation.targets[0], 'derived', 'nope'),
                            'is not a derived value'),
        'unknown_method': (lambda model: setattr(model.study.estimation, 'method', 'newton'), 'is not one of'),
        'nothing_to_fit': (lambda model: setattr(model.study.estimation, 'targets', []), 'add a target'),
        'curve_lengths': (lambda model: model.study.estimation.curves.append(
            EstimationCurveSpec('heat.source:s', x=[0.0, 1.0], y=[1.0])), 'same number of samples'),
        'curve_source': (lambda model: model.study.estimation.curves.append(
            EstimationCurveSpec('no_such_curve', x=[0.0], y=[1.0])), 'is not a derived curve'),
    }
    for name, (change, text) in refusals.items():
        model = good.copy()
        change(model)
        messages = estimation_errors(model)
        add(f'refuses_{name}', any(text in message for message in messages), messages)

    # JSON round trip and renames
    loaded = ModelSpec.from_dict(good.to_dict())
    add('round_trip', loaded.study.estimation == good.study.estimation)
    renamed = good.copy()
    renamed.rename_parameter('k_fit', 'k_solid', registry)
    renamed.rename_derived_value('rise_max', 'peak_rise')
    add('renames', renamed.study.estimation.parameters[0].name == 'k_solid'
        and renamed.study.estimation.targets[0].derived == 'peak_rise')

    return checks.result()


ESTIMATION_SELFTEST_CASES = (
    SelfTestCase(
        name='estimation_parts',
        label='Parameter estimation: curve interpolation, residuals, statistics and validation',
        description='CPU checks of the parameter estimation parts (no model run): model curves interpolated onto the '
                    'measured x (outside the range raises), a physics.feature:sample source read like a derived-curve '
                    'input, weighted residuals (targets relative to |value|, curve points to the RMS of y), the '
                    'search scaling, standard errors and correlation from J^T J (absolute sigma, and scaled by '
                    'chi^2 / dof = 1) to 1e-12, a rank-deficient Jacobian giving none; validation refusals (with a '
                    'sweep, unknown parameter, initial outside the bounds, unknown derived value, unknown method, '
                    'nothing to fit, x / y length mismatch, unknown derived curve) and none for a valid estimation; '
                    'the JSON round trip and parameter / derived value renames. error = number of failed '
                    'sub-checks, tolerance 0.',
        specs=lambda: {},
        check=_check_parts),
    SelfTestCase(
        name='estimation_heat_conductivity',
        label='Parameter estimation: conductivity of a 1D heat slab (least squares)',
        description='Stationary 1D heat slab (1 mm, 100 cells, uniform Q = 1e8 W/m^3, x_lo insulated, x_hi held, cpu '
                    'banded solve): least_squares recovers k = 2.3 W/(m K) from k = 1 (bounds 0.1..10) and the '
                    'measured max T_rise Q L^2 / (2 k), which the cell-centred chain reproduces exactly (float32 T: '
                    '~1e-6). measured = |k_fit / 2.3 - 1|; tolerance 1e-4.',
        specs=lambda: {'fit': _slab_spec(_conductivity_estimation('least_squares', 20, 1e-6))},
        check=_fit_check({'k_fit': _K_TRUE}, _FIT_TOLERANCE)),
    SelfTestCase(
        name='estimation_heat_two_parameters',
        label='Parameter estimation: bounded two-parameter fit of a 1D heat slab',
        description='The estimation_heat_conductivity slab with k_fit (1.5 in 1..4) and Q_heat (2e7 in 1e7..1e8 '
                    'W/m^3) fitted to the max T_rise and the source power Q L of k = 2.3, Q = 5e7 with sigma 1e-3 of '
                    'each (absolute: finite standard errors required). measured = the worse relative error of the '
                    'two fitted values; tolerance 1e-4.',
        specs=lambda: {'fit': _slab_spec(_two_parameter_estimation())},
        check=_fit_check({'k_fit': _K_TRUE, 'Q_heat': _Q_TRUE}, _FIT_TOLERANCE, need_errors=True)),
    SelfTestCase(
        name='estimation_heat_nelder_mead',
        label='Parameter estimation: conductivity of a 1D heat slab (Nelder-Mead)',
        description='The estimation_heat_conductivity fit with the bounded Nelder-Mead simplex (tolerance 1e-6, at '
                    'most 60 runs). measured = |k_fit / 2.3 - 1|; tolerance 1e-3. Run count and wall time not yet '
                    'confirmed by a run.',
        specs=lambda: {'fit': _slab_spec(_conductivity_estimation('nelder_mead', 60, 1e-6))},
        check=_fit_check({'k_fit': _K_TRUE}, _SIMPLEX_TOLERANCE), heavy=True),
)
