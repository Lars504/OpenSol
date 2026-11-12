"""Parameter estimation of one study (StudySpec.estimation): the study is built and run
per evaluation, scipy least_squares (TRF) or bounded Nelder-Mead searches the estimated parameters, and a run at the
best point stays live (summary 'estimation')."""
import math
import time
from dataclasses import replace

import numpy as np

from .analytic import split_key
from .estimation import curve_model, fit_statistics, resolve_estimation, weighted_residuals
from .expressions import evaluate_parameters
from .meshing import remember_element_meshes, remembers_element_meshes
from .study import (Cancelled, LogEvent, Progress, StudyFailed, StudyStarted, _clamped_fraction, _fail_sweep_setup,
                    _parity_sectors_requested, _record_failure, _run_built, build_session, release_session,
                    summed_runs)

_STAGE = 'Parameter estimation'


class _Stop(Exception):
    """Ends the search early: reason (estimation.STOP_REASONS) and message."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class _FirstRunFailed(Exception):
    """The first evaluation (the initial point) could not be computed: (message, traceback text, out_of_memory)."""

    def __init__(self, record):
        super().__init__(record[0])
        self.record = record


def run_estimation(spec, registry, build, emit, control, service, on_session):
    """Search the estimated parameters that fit the targets and curves (one build + study run per evaluation, element
    meshes reused while their key matches), then keep a run at the best point live -> its summary + 'estimation', or
    None after a failure of the first evaluation (StudyFailed emitted)."""
    try:
        resolved, errors, _ = resolve_estimation(spec, evaluate_parameters(spec.parameters))
        if errors:
            raise ValueError('; '.join(errors))
        sectors = _parity_sectors_requested(spec, registry)
    except Exception as exc:   # text only; reported through the build path with the node paths
        _fail_sweep_setup(spec, registry, build, emit, on_session, _record_failure(exc), what='parameter estimation')
        return None
    remembered = remembers_element_meshes()
    if not remembered:   # a geometry-free parameter then rebuilds without remeshing
        remember_element_meshes()
    try:
        return _Estimation(spec, resolved, sectors, registry, build, emit, control, service, on_session).run()
    finally:
        if not remembered:
            remember_element_meshes(False)


class _Estimation:
    """One estimation run: evaluations (scaled search vector z, value = initial + (z - 1) scale), one live session at a
    time, the cost history, the best point and the final best-point run."""

    def __init__(self, spec, resolved, sectors, registry, build, emit, control, service, on_session):
        self.spec, self.resolved, self.sectors, self.registry = spec, resolved, sectors, registry
        self.build, self.emit, self.control, self.service, self.on_session = build, emit, control, service, on_session
        self.session, self.session_key, self.session_summary = None, None, None
        self.started = False
        self.evaluations = 0
        self.runs = []            # [(label, summary)] of every computed run (summed_runs)
        self.cache = {}           # z tuple -> weighted residuals of an evaluated point
        self.best = None          # {'key', 'cost', 'residuals', 'derived', 'evaluation'} of the lowest cost
        self.history = {'evaluation': [], 'cost': [], 'parameters': {name: [] for name in resolved.names}}
        self.wall_start = time.perf_counter()

    # ---- driver ----
    def run(self):
        resolved = self.resolved
        stop_reason, message, jacobian_record = 'converged', '', None
        try:
            stop_reason, message, jacobian_record = self._search()
        except _FirstRunFailed as failure:
            text, failure_traceback, out_of_memory = failure.record
            self.emit(StudyFailed(f'parameter estimation, evaluation 1 (the initial point): {text}', failure_traceback,
                                  out_of_memory=out_of_memory, session_live=self.session is not None))
            return None
        except _Stop as stop:
            stop_reason, message = stop.reason, str(stop)
        if self.best is None:
            self.emit(StudyFailed(f'parameter estimation: no evaluation gave a finite misfit ({message})', '',
                                  session_live=self.session is not None))
            return None
        if stop_reason != 'cancelled' and self.session_key != self.best['key']:
            self._best_point_run()
        summary = dict(self.session_summary)
        statistics = (fit_statistics(jacobian_record[0], jacobian_record[1], resolved.scale, resolved.absolute_sigma)
                      if jacobian_record is not None else
                      {'rank': None, 'reduced_chi_square': None, 'standard_errors': None, 'correlation': None,
                       'note': 'no Jacobian at the best point (nelder_mead, or the search stopped early)'})
        summary['estimation'] = self._report(stop_reason, message, statistics)
        summary.update(summed_runs(self.runs))
        summary['ticks'], summary['tick_unit'] = self.evaluations, 'evaluation'
        if stop_reason == 'cancelled':
            summary['status'] = 'cancelled'
        level = 'info' if stop_reason == 'converged' else 'warning'
        self.emit(LogEvent(level, f"parameter estimation: {summary['estimation']['message']}"))
        summary['sweep'], summary['sweep_index'] = None, None
        return summary

    def _search(self):
        """(stop reason, message, (jacobian, residuals) at the solution or None) of the scipy search."""
        from scipy.optimize import Bounds, least_squares, minimize
        resolved = self.resolved
        start = np.ones(len(resolved.names))
        lower, upper = resolved.scaled_bounds()
        if resolved.method == 'least_squares':
            result = least_squares(self.residuals, start, bounds=(lower, upper), method='trf',
                                   diff_step=resolved.diff_step, x_scale=1.0, ftol=resolved.tolerance,
                                   xtol=resolved.tolerance, gtol=resolved.tolerance, max_nfev=resolved.max_evaluations)
            reason = 'converged' if result.status > 0 else 'max_evaluations'
            return reason, str(result.message), (result.jac, result.fun)
        bounds = Bounds(lower, upper) if np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)) else None
        result = minimize(self.cost, start, method='Nelder-Mead', bounds=bounds,
                          options={'maxfev': resolved.max_evaluations, 'xatol': resolved.tolerance,
                                   'fatol': resolved.tolerance})
        return ('converged' if result.success else 'max_evaluations'), str(result.message), None

    # ---- evaluations ----
    def cost(self, scaled):
        """0.5 sum r^2 of a point (inf when not finite: Nelder-Mead steps away from it)."""
        residuals = self.residuals(scaled)
        value = 0.5 * float(residuals @ residuals)
        return value if math.isfinite(value) else math.inf

    def residuals(self, scaled):
        """Weighted residuals of a point (cached per point); raises _Stop at the evaluation limit, on cancel, on a failed
        run and (least_squares) on a non-finite residual."""
        key = tuple(float(value) for value in np.asarray(scaled, dtype=np.float64))
        if key in self.cache:
            return self.cache[key]
        resolved = self.resolved
        if self.evaluations >= resolved.max_evaluations:
            raise _Stop('max_evaluations', f'max_evaluations ({resolved.max_evaluations}) reached')
        if self.evaluations > 0:
            try:
                self.control.checkpoint(self.service)
            except Cancelled:
                raise _Stop('cancelled', f'cancelled after {self.evaluations} evaluations') from None
        values = resolved.values_of(key)
        self.evaluations += 1
        index = self.evaluations
        self.emit(LogEvent('info', f'parameter estimation, evaluation {index}: {_values_text(resolved.names, values)}'))
        summary, failure = self._run_point(values, key, self._evaluation_emit(index - 1))
        if failure is None and summary['status'] == 'cancelled':
            self.runs.append((f'evaluation {index}', summary))
            raise _Stop('cancelled', f'cancelled during evaluation {index}')
        residuals = None
        if failure is None:
            self.runs.append((f'evaluation {index}', summary))
            try:
                residuals = self._weighted_residuals(summary)
            except Exception as exc:   # text only: a curve source or abscissa problem of this run
                failure = _record_failure(exc)
        if failure is not None:
            if index == 1:
                raise _FirstRunFailed(failure)
            raise _Stop('failed', f'evaluation {index} failed: {failure[0]}')
        cost = 0.5 * float(residuals @ residuals)
        self._record(index, key, values, residuals, cost, summary)
        if not math.isfinite(cost):
            message = f'evaluation {index} gave a non-finite misfit ({_values_text(resolved.names, values)})'
            if resolved.method == 'least_squares':
                raise _Stop('not_finite', message)
            self.emit(LogEvent('warning', f'parameter estimation: {message}; the search steps away from it'))
        self.cache[key] = residuals
        return residuals

    def _record(self, index, key, values, residuals, cost, summary):
        """History row, log line and the best point of one evaluation."""
        self.history['evaluation'].append(index)
        self.history['cost'].append(cost)
        for name, value in zip(self.resolved.names, values):
            self.history['parameters'][name].append(float(value))
        self.emit(LogEvent('info', f'parameter estimation, evaluation {index}: cost {cost:.6g}'))
        if math.isfinite(cost) and (self.best is None or cost < self.best['cost']):
            self.best = {'key': key, 'cost': cost, 'residuals': residuals, 'evaluation': index,
                         'derived': {name: float(value) for name, value in summary['derived'].items()}}

    def _weighted_residuals(self, summary):
        """Residuals of the live run: targets from the summary's derived values (parity sectors: the combined ones),
        curves from the live session's results (parity sectors: the combined results where they exist)."""
        curve_models = []
        if self.resolved.curves:
            combined = (summary.get('parity_sectors') or {}).get('combined_results') or {}
            fetched = {}

            def fetch(key):
                if key in combined:
                    return combined[key]
                if key not in fetched:
                    fetched[key] = self.session.feature_result(*split_key(key))
                return fetched[key]
            curve_models = [curve_model(self.session.spec, curve.source, fetch, self.session.parameters)
                            for curve in self.resolved.curves]
        return weighted_residuals(self.resolved, summary['derived'], curve_models)

    def _run_point(self, values, key, point_emit):
        """Build and run the study at parameter values, closing the previous session first -> (summary, None) with the
        new session live, or (None, (message, traceback text, out_of_memory)) with no session left."""
        point_spec = self.spec.copy()
        for name, value in zip(self.resolved.names, values):
            point_spec.parameter(name).expression = repr(float(value))
        point_spec.study.estimation = None
        previous, self.session, self.session_key, self.session_summary = self.session, None, None, None
        if self.sectors:
            from .study_sectors import _merge_runs, _run_sectors   # it imports study's helpers, as this module
            outcome = _run_sectors(point_spec, self.registry, self.build, point_emit, self.control, self.service,
                                   self.on_session, previous, self.started)
            self.started = outcome.started
            if outcome.failure is not None:
                message, failure_traceback, out_of_memory, _ = outcome.failure
                if outcome.session is not None:
                    release_session(self.on_session, outcome.session, self.emit)
                return None, (message, failure_traceback, out_of_memory)
            summary, session = _merge_runs(point_spec, outcome, point_emit), outcome.session
        else:
            release_session(self.on_session, previous, self.emit if previous is not None else None)
            session, failure = build_session(point_spec, point_emit, self.on_session, build=self.build,
                                             registry=self.registry)
            if failure is not None:
                return None, (f'build failed: {failure.message}', failure.traceback, failure.out_of_memory)
            if not self.started:
                self.emit(StudyStarted())
                self.started = True
            run_failure = None
            try:
                summary = _run_built(session, point_spec, self.registry, point_emit, self.control, self.service)
            except Exception as exc:   # text only: the traceback holds the run's arrays
                run_failure = _record_failure(exc)
            if run_failure is not None:
                release_session(self.on_session, session, self.emit)
                return None, run_failure
        self.session, self.session_key, self.session_summary = session, key, summary
        return summary, None

    def _best_point_run(self):
        """Rebuild and run the best point so its session stays live (the last evaluation was elsewhere); a failure
        raises RuntimeError (run_study reports it as StudyFailed)."""
        key = self.best['key']
        self.emit(LogEvent('info', 'parameter estimation: run at the best point '
                                   f'({_values_text(self.resolved.names, self.resolved.values_of(key))})'))
        summary, failure = self._run_point(self.resolved.values_of(key), key,
                                           self._evaluation_emit(self.resolved.max_evaluations))
        if failure is not None:
            raise RuntimeError(f'parameter estimation: the run at the best point failed: {failure[0]}')
        self.runs.append(('best point', summary))

    def _evaluation_emit(self, index):
        """emit for evaluation `index` (0-based; max_evaluations = the best-point run): Progress over the whole
        estimation, the run's own fraction as the stage fraction."""
        planned = self.resolved.max_evaluations + 1

        def emit(event):
            if isinstance(event, Progress):
                inner = _clamped_fraction(event)
                label = 'best point' if index >= self.resolved.max_evaluations else f'evaluation {index + 1}'
                event = replace(event, fraction=min((index + inner) / planned, 1.0),
                                wall_elapsed=time.perf_counter() - self.wall_start,
                                message=f'{label}: {event.message}' if event.message else label, stages=(_STAGE,),
                                stage_index=0, stage_fraction=inner, stage_elapsed=event.wall_elapsed)
            self.emit(event)
        return emit

    # ---- report ----
    def _report(self, stop_reason, message, statistics):
        """summary 'estimation': settings, initial / fitted values, standard errors, residuals, history, stop."""
        resolved, best = self.resolved, self.best
        names = resolved.names
        fitted = resolved.values_of(best['key'])
        errors = statistics['standard_errors']
        initial_cost = self.history['cost'][0] if self.history['cost'] else math.nan
        residuals = best['residuals']
        targets = [{'derived': target.derived, 'measured': target.value, 'sigma': target.sigma,
                    'model': best['derived'].get(target.derived, math.nan), 'residual': float(residuals[index])}
                   for index, target in enumerate(resolved.targets)]
        curves, offset = [], len(resolved.targets)
        for curve in resolved.curves:
            part = residuals[offset:offset + curve.x.size]
            offset += curve.x.size
            curves.append({'source': curve.source, 'n_points': int(curve.x.size),
                           'rms_residual': float(np.sqrt(np.mean(part ** 2))),
                           'max_abs_residual': float(np.max(np.abs(part)))})
        text = (f"{_values_text(names, fitted)}; cost {best['cost']:.6g} (initial {initial_cost:.6g}) after "
                f'{self.evaluations} evaluations; {stop_reason}' + (f': {message}' if message else ''))
        return {'method': resolved.method, 'parameters': list(names),
                'initial': dict(zip(names, map(float, resolved.initial))),
                'lower': dict(zip(names, map(_bound, resolved.lower))),
                'upper': dict(zip(names, map(_bound, resolved.upper))),
                'fitted': dict(zip(names, map(float, fitted))),
                'standard_errors': None if errors is None else dict(zip(names, errors)),
                'correlation': statistics['correlation'], 'rank': statistics['rank'],
                'reduced_chi_square': statistics['reduced_chi_square'], 'statistics_note': statistics['note'],
                'absolute_sigma': resolved.absolute_sigma, 'cost': best['cost'], 'cost_initial': initial_cost,
                'best_evaluation': best['evaluation'], 'evaluations': self.evaluations,
                'max_evaluations': resolved.max_evaluations, 'targets': targets, 'curves': curves,
                'history': self.history, 'live_is_best': self.session_key == best['key'],
                'converged': stop_reason == 'converged', 'stop_reason': stop_reason, 'message': text}


def _values_text(names, values):
    return ', '.join(f'{name} = {float(value):.8g}' for name, value in zip(names, values))


def _bound(value):
    """A bound for JSON: None when unbounded."""
    return float(value) if math.isfinite(value) else None
