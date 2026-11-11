"""Results panel 'Estimation' tab (summary 'estimation' of a parameter estimation): fitted values with standard errors,
measured versus model values, and plots of the misfit and the parameters over the runs (GUI thread)."""
import math

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..model.expressions import error_message
from .estimation_form import ESTIMATION_METHOD_LABELS
from .forms import fill_table, read_only_table, rich_tooltip, table_number, wrapped_label
from .helpers import USER_ERRORS
from .plot_view import POINT_OPTIONS, curve_xy

NO_ESTIMATION = 'No parameter estimation in the last study (Study settings: Parameter estimation).'
_STOP_REASON_TEXTS = {'converged': 'converged', 'max_evaluations': 'stopped at the maximum number of runs',
                      'not_finite': 'stopped: a run gave a misfit that is not finite',
                      'failed': 'stopped: a run failed', 'cancelled': 'stopped by the user'}
_RUN_AXIS = 'Run of the search'
_LINE_SERIES = ('lowest so far', 'fitted value')   # drawn as lines; every other history series as points
_PARAMETER_COLUMNS = ('Parameter', 'Fitted', 'Standard error', 'Relative error', 'Start', 'Lower bound', 'Upper bound')
_DATA_COLUMNS = ('Measured data', 'Measured', 'Model', 'Sigma', 'Residual / sigma')
_PARAMETERS_TABLE_TIP = ('Fitted value of each estimated parameter (the best run) with its standard error: one '
                         'standard deviation from the curvature of the misfit at the fit.\n\n'
                         'With an uncertainty given for every measured value and curve, the standard errors follow '
                         'from them; otherwise they are scaled by the scatter of the residuals.')
_DATA_TABLE_TIP = ('Measured values: measured, computed at the fitted parameters, the uncertainty and (model - '
                   'measured) / sigma.\n\n'
                   'Measured curves: the number of samples and the RMS and largest |residual / sigma| over them.')
_USE_FITTED_TIP = ('Write the fitted values into the Parameters node (one undoable edit), so the model runs at them '
                   'from now on and a new estimation starts there.')


def estimation_report(summary):
    """summary 'estimation' of a study, None without one."""
    return (summary or {}).get('estimation') or None


def estimation_plots(report) -> list:
    """[(name, xy)]: the misfit (and its lowest value so far, log y) and each parameter (and its fitted value) versus
    the run of the search; [] without a history."""
    history = (report or {}).get('history') or {}
    runs = np.asarray(history.get('evaluation') or [], dtype=np.float64)
    if runs.size == 0:
        return []
    cost = np.asarray(history.get('cost') or [], dtype=np.float64)
    if cost.size != runs.size:
        return []
    with np.errstate(invalid='ignore'):
        finite_cost = np.where(np.isfinite(cost), cost, np.nan)
        lowest = np.fmin.accumulate(finite_cost)
        positive = bool(np.all(finite_cost[np.isfinite(finite_cost)] > 0.0))
    plots = [('Misfit', curve_xy([('misfit', runs, finite_cost, '1'), ('lowest so far', runs, lowest, '1')],
                                 _RUN_AXIS, '', 'misfit (half the sum of squared residual / sigma)', log_y=positive))]
    fitted = report.get('fitted') or {}
    for name, values in (history.get('parameters') or {}).items():
        values = np.asarray(values, dtype=np.float64)
        if values.size != runs.size:
            continue
        curves = [(name, runs, values, '')]
        if fitted.get(name) is not None:
            curves.append(('fitted value', runs, np.full(runs.size, float(fitted[name])), ''))
        plots.append((f'Parameter {name}', curve_xy(curves, _RUN_AXIS, '', name)))
    return plots


def estimation_curve_options(name, xy, series):
    """Points for the runs, lines for the running minimum and the fitted value."""
    return {} if series in _LINE_SERIES else dict(POINT_OPTIONS)


def _state_text(report) -> str:
    """'Least squares converged: k = 2.3, ...; cost ... The results shown are those of the best run.'"""
    method = ESTIMATION_METHOD_LABELS.get(report.get('method'), str(report.get('method'))).split(' (')[0]
    stop_reason = str(report.get('stop_reason', ''))
    shown = ('The results shown are those of the best run.' if report.get('live_is_best')
             else 'The results shown are those of the last run, not of the best one.')
    return f"{method}, {_STOP_REASON_TEXTS.get(stop_reason, stop_reason)}: {report.get('message', '')}. {shown}"


def _statistics_text(report) -> str:
    """Misfit, best run, reduced chi-square, how the standard errors were scaled, the statistics note and the
    correlations of the parameter pairs."""
    parts = [f"misfit {table_number(report.get('cost'))} (start {table_number(report.get('cost_initial'))}), best run "
             f"{report.get('best_evaluation')} of {report.get('evaluations')} (at most {report.get('max_evaluations')})"]
    if report.get('reduced_chi_square') is not None:
        parts.append(f"reduced chi-square {table_number(report['reduced_chi_square'], 4)}")
    if report.get('standard_errors') is not None:
        parts.append('standard errors from the given uncertainties' if report.get('absolute_sigma')
                     else 'standard errors scaled by the residual scatter (not every uncertainty is given)')
    if report.get('statistics_note'):
        parts.append(f"no standard errors: {report['statistics_note']}")
    names = list(report.get('parameters') or [])
    correlation = report.get('correlation')
    if correlation and len(names) > 1:
        pairs = [f'{names[row]}–{names[column]} {table_number(correlation[row][column], 3)}'
                 for row in range(len(names)) for column in range(row + 1, len(names))]
        parts.append('correlation ' + ', '.join(pairs))
    return '; '.join(parts) + '.'


class EstimationTab(QtWidgets.QWidget):
    """The Estimation tab: state line, fitted-parameter and measured-data tables, the history plot view (owned by the
    ResultsPanel) and 'Use fitted values'."""
    message = QtCore.Signal(str, bool)   # status text, error flag (the panel's message line)

    def __init__(self, document, plot_view, parent=None):
        super().__init__(parent)
        self._document = document
        self._report = None
        self.plot_view = plot_view
        self._info = wrapped_label(NO_ESTIMATION, QtCore.Qt.TextFormat.PlainText)
        self._parameters_table = read_only_table(_PARAMETER_COLUMNS, _PARAMETERS_TABLE_TIP)
        self._data_table = read_only_table(_DATA_COLUMNS, _DATA_TABLE_TIP)
        self._statistics = QtWidgets.QLabel()
        self._statistics.setWordWrap(True)
        self._statistics.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._statistics.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._use_button = QtWidgets.QPushButton('Use fitted values')
        self._use_button.setToolTip(rich_tooltip(_USE_FITTED_TIP))
        self._use_button.setEnabled(False)
        self._use_button.clicked.connect(lambda _checked=False: self._use_fitted_values())
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        for widget in (self._parameters_table, self._data_table, plot_view):
            splitter.addWidget(widget)
        splitter.setStretchFactor(2, 2)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self._use_button)
        button_row.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._info)
        layout.addWidget(splitter, 1)
        layout.addWidget(self._statistics)
        layout.addLayout(button_row)

    def show_report(self, report):
        """Fill the tab from a summary 'estimation' (None: empty) and redraw the history plots."""
        self._report = report
        self.plot_view.request_redraw()
        self._use_button.setEnabled(bool(report and report.get('fitted')))
        if not report:
            self._info.setText(NO_ESTIMATION)
            self._info.setStyleSheet('')
            self._statistics.setText('')
            fill_table(self._parameters_table, [])
            fill_table(self._data_table, [])
            return
        self._info.setText(_state_text(report))
        self._info.setStyleSheet(f"color:{'#205020' if report.get('converged') else '#9a5b00'};")
        self._statistics.setText(_statistics_text(report))
        fitted, errors = report.get('fitted') or {}, report.get('standard_errors') or {}
        rows = []
        for name in report.get('parameters') or []:
            value, error = fitted.get(name), errors.get(name)
            relative = (abs(float(error) / float(value)) if error is not None and value not in (None, 0.0)
                        and math.isfinite(float(value)) else None)
            rows.append([name, table_number(value, 8), table_number(error, 3), table_number(relative, 3),
                         table_number((report.get('initial') or {}).get(name)),
                         table_number((report.get('lower') or {}).get(name)) or 'none',
                         table_number((report.get('upper') or {}).get(name)) or 'none'])
        fill_table(self._parameters_table, rows)
        rows = [[target.get('derived', ''), table_number(target.get('measured')), table_number(target.get('model')),
                 table_number(target.get('sigma'), 3), table_number(target.get('residual'), 3)]
                for target in report.get('targets') or []]
        rows += [[f"curve {curve.get('source', '')}", f"{curve.get('n_points', 0)} samples", '', '',
                  f"RMS {table_number(curve.get('rms_residual'), 3)}, largest {table_number(curve.get('max_abs_residual'), 3)}"]
                 for curve in report.get('curves') or []]
        fill_table(self._data_table, rows)

    def _use_fitted_values(self):
        """Set the model parameters to the fitted values (one undoable 'parameters' edit)."""
        fitted = (self._report or {}).get('fitted') or {}
        if not fitted:
            return

        def apply(spec):
            for name, value in fitted.items():
                spec.parameter(name).expression = float(value)   # KeyError: the parameter is gone
        label = 'Use fitted parameter values'
        try:
            changed = self._document.mutate('parameters', apply, label=label)
        except USER_ERRORS as exc:
            self.message.emit(f'{label}: {error_message(exc)}', True)
            return
        values = ', '.join(f'{name} = {float(value):.8g}' for name, value in fitted.items())
        self.message.emit(f'Parameters set to the fitted values: {values}' if changed
                          else f'The parameters already hold the fitted values ({values}).', False)
