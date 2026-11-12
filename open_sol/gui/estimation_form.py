"""Parameter estimation node (StudySpec.estimation): the study form's group, the node's form (a SettingsPanel mixin),
its editors and texts (GUI thread, CPU only)."""
import itertools
import math
import os
import re
from functools import partial

from PySide6 import QtCore, QtWidgets

from ..model.analytic import CURVE_RESULT_KINDS, RESULT_SAMPLE_NAMES
from ..model.estimation import ESTIMATION_METHODS, curve_source_problem
from ..model.schema import Param
from ..model.spec import EstimatedParameterSpec, EstimationCurveSpec, EstimationSpec, EstimationTargetSpec
from .document import ESTIMATION_NODE
from .forms import (CommitLineEdit, CommitTextEdit, ExpressionEdit, IntSpinBox, NumberEdit, ValueComboBox,
                    display_number, format_number, growing_form, hint_label, mark_invalid, param_row_label,
                    rich_tooltip, tip_label)
from .helpers import remember_directory, start_directory

ESTIMATION_METHOD_LABELS = {'least_squares': 'Least squares (trust region, finite-difference derivatives)',
                            'nelder_mead': 'Nelder-Mead (no derivatives, for noisy results)'}
ESTIMATION_TIP = ('Adjusts model parameters until the results match measured values and curves (industry-standard '
                  'parameter estimation). Each try is a full build and run of the study; the search ends with a run at '
                  'the best parameters, whose results stay shown.\n\n'
                  'The fitted values, their standard errors and the misfit of every run are in the Estimation tab of '
                  'Results.')
_INTRO = 'Fit model parameters to measured values or curves.'
_DATA_FILE_FILTER = 'Data files (*.csv *.txt *.dat *.tsv);;All files (*)'
_ITEM_WORDS = {'parameters': 'parameter', 'targets': 'measured value', 'curves': 'measured curve'}
_SETTING_ROWS = {   # EstimationSpec attribute -> (row label, tooltip)
    'method': ('Method', 'How the search picks the next parameters to try.\n\n'
                         '- Least squares: estimates how each result changes with each parameter (one extra run per '
                         'parameter) and steps toward the best fit; fast for smooth results and gives standard '
                         'errors.\n'
                         '- Nelder-Mead: compares runs only, with no derivatives; slower, but robust when the results '
                         'jump a little between nearby parameters (for example a geometry parameter that remeshes).'),
    'max_evaluations': ('Maximum number of runs', 'Study runs the search may use; the final run at the best '
                                                  'parameters comes on top. Reaching it gives a warning.'),
    'diff_step': ('Derivative step (relative)', 'Least squares: relative change of each parameter used to estimate '
                                                'how the results depend on it, for example 0.01 for 1 %. Larger steps '
                                                'average over mesh noise; smaller ones are more exact for smooth '
                                                'results.'),
    'tolerance': ('Tolerance (relative)', 'The search stops when the misfit or the parameters change by less than '
                                          'this, relative to their size.'),
}
_PARAMETERS_TIP = ('Model parameters the search adjusts. Each starts at its start value (empty: its value in the '
                   'Parameters node) and stays within its bounds (empty: no bound).')
_TARGETS_TIP = ('Measured values of derived values (Result settings > Derived values). The misfit of each is (model - '
                'measured) / uncertainty.')
_CURVES_TIP = ('Measured curves compared with a result curve or a derived curve. The model curve is interpolated '
               'linearly onto the measured x, which must lie within the computed range.')
_SOURCE_TIP = ('Model curve compared with the data: the name of a derived curve, or "physics.feature:sample" of a '
               'result, for example "heat.probe1:values" (a probe trace versus time) or "em.flux1:power" (a spectrum '
               'versus wavelength).\n\n'
               'The measured x must be in the unit of its abscissa: t in s, wavelength in m, x in the unit of an xy '
               'result, the x of a derived curve.')
_DATA_TIP = ('Measured samples, one per line: "x, y", or "x, y, sigma" with an uncertainty per sample. Commas, tabs '
             'or spaces separate the numbers; lines starting with # are ignored. Load… reads such a text or CSV file '
             '(header lines are skipped).')
_PARAMETER_NAME_TIP = 'Model parameter (Parameters node) that the search adjusts.'
_TARGET_NAME_TIP = 'Derived value whose computed value is compared with the measured one.'
_INITIAL = Param('initial', 'float', None, label='Start value', optional=True,
                 help='Value of the first run (number or expression of parameters); empty = the parameter value',
                 tip='Value of the parameter in the first run: a number or an expression of the other parameters. '
                     'Empty: its value in the Parameters node.')
_LOWER = Param('lower', 'float', None, label='Lower bound', optional=True,
               help='Smallest value the search may try; empty = no bound',
               tip='Smallest value the search may try (a number or an expression of parameters). Empty: no bound. '
                   'The start value must lie strictly inside the bounds.')
_UPPER = Param('upper', 'float', None, label='Upper bound', optional=True,
               help='Largest value the search may try; empty = no bound',
               tip='Largest value the search may try (a number or an expression of parameters). Empty: no bound. '
                   'The start value must lie strictly inside the bounds.')
_TARGET_VALUE = Param('value', 'float', None, label='Measured value',
                      help='Measured value of the derived value (number or expression of parameters)',
                      tip='Measured value of the derived value, in its unit: a number or an expression of parameters.')
_TARGET_SIGMA = Param('sigma', 'float', None, label='Uncertainty (sigma)', optional=True, min=0.0, min_exclusive=True,
                      help='Standard deviation of the measured value; empty = |measured value| (a relative misfit)',
                      tip='Uncertainty (one standard deviation) of the measured value, in its unit. It weights this '
                          'value against the others and sets the standard errors of the fit.\n\n'
                          'Empty: the size of the measured value, so the misfit is relative.')
_CURVE_SIGMA = Param('sigma', 'float', None, label='Uncertainty (sigma)', optional=True, min=0.0, min_exclusive=True,
                     help='Standard deviation of every sample; empty = the RMS of the measured y',
                     tip='Uncertainty (one standard deviation) of every measured y, in its unit. A third data column '
                         'gives one per sample instead.\n\n'
                         'Empty: the RMS of the measured y, so the misfit is relative to the size of the curve.')


# ---- pure helpers ----
def estimation_blocker(spec) -> str:
    """Why a parameter estimation cannot be switched on for spec (validation's rules), '' when it can."""
    if spec.study.sweep is not None:
        return 'Not available together with a parameter sweep.'
    if spec.study.adaptation is not None:
        return 'Not available together with mesh refinement.'
    if not spec.parameters:
        return 'Add a parameter first: the estimation adjusts model parameters.'
    return ''


def estimation_toggle(enabled):
    """apply(spec) of the study edit switching StudySpec.estimation on (a new_estimation; ValueError with the
    estimation_blocker reason) or off: the tree menu and the estimation form share it."""
    def apply(spec):
        if not enabled:
            spec.study.estimation = None
        elif spec.study.estimation is None:
            blocker = estimation_blocker(spec)
            if blocker:
                raise ValueError(blocker)
            spec.study.estimation = new_estimation(spec)
    return apply


def _items(estimation, kind) -> list:
    """The list estimation.<kind> ([] for a malformed value)."""
    items = getattr(estimation, kind, None)
    return items if isinstance(items, list) else []


def estimated_parameter_names(estimation) -> list:
    """Names of the estimated parameters (malformed entries skipped)."""
    return [item.name for item in _items(estimation, 'parameters') if isinstance(item, EstimatedParameterSpec)]


def _count_text(count, noun):
    return f"{count} {noun}{'' if count == 1 else 's'}"


def estimation_overview_text(estimation) -> str:
    """'Estimates k, h from 1 measured value and 1 measured curve; least squares, at most 50 runs.'"""
    names = estimated_parameter_names(estimation)
    method = ESTIMATION_METHOD_LABELS.get(estimation.method, str(estimation.method)).split(' (')[0].lower()
    return (f"Estimates {', '.join(names) or 'no parameter yet'} from "
            f"{_count_text(len(_items(estimation, 'targets')), 'measured value')} and "
            f"{_count_text(len(_items(estimation, 'curves')), 'measured curve')}; {method}, at most "
            f"{estimation.max_evaluations} runs.")


def estimation_tree_label(estimation) -> str:
    """Model tree label of ESTIMATION_NODE: 'Parameter estimation (k, h)'."""
    names = estimated_parameter_names(estimation)
    return f"Parameter estimation ({', '.join(names)})" if names else 'Parameter estimation'


def curve_source_entries(document) -> list:
    """[(source, description)] a measured curve can be compared with: the derived curves, then the samples of the
    feature results ('physics.feature:' alone for xy / table results, whose sample names come from the result)."""
    entries = [(curve.name, f'derived curve {curve.name}') for curve in document.spec.derived_curves]
    for key, kind in document.result_features(CURVE_RESULT_KINDS):
        abscissa, sample_names = RESULT_SAMPLE_NAMES[kind]
        if sample_names is None:
            entries.append((f'{key}:', f"{key}: followed by a {'series' if kind == 'xy' else 'column'} name"))
        else:
            entries.extend((f'{key}:{sample}', f'{key}:{sample} versus {abscissa}')
                           for sample in sample_names if sample != abscissa)
    return entries


def new_estimation(spec) -> EstimationSpec:
    """EstimationSpec of a newly switched-on estimation: the first parameter, and the first derived value as a measured
    value (its reference, else 0) when there is one."""
    targets = [_new_target(spec)] if spec.derived_values else []
    return EstimationSpec(parameters=[EstimatedParameterSpec(name=spec.parameters[0].name)], targets=targets)


def _new_target(spec, taken=()):
    """Measured value of the first derived value not in taken (else the first), value = its reference or 0."""
    derived = next((item for item in spec.derived_values if item.name not in taken), spec.derived_values[0])
    return EstimationTargetSpec(derived=derived.name, value=derived.reference if derived.reference is not None else 0.0)


def new_estimation_item(spec, kind, sources=()):
    """New entry of estimation.<kind> ('parameters' | 'targets' | 'curves'); ValueError when nothing fits.
    sources: curve_source_entries values (the first complete one becomes a new curve's source)."""
    estimation = spec.study.estimation
    if kind == 'parameters':
        listed = set(estimated_parameter_names(estimation))
        free = [parameter.name for parameter in spec.parameters if parameter.name not in listed]
        if not free:
            raise ValueError('every model parameter is estimated already; add a parameter first')
        return EstimatedParameterSpec(name=free[0])
    if kind == 'targets':
        if not spec.derived_values:
            raise ValueError('add a derived value first (Result settings > Derived values)')
        return _new_target(spec, {item.derived for item in _items(estimation, 'targets')
                                  if isinstance(item, EstimationTargetSpec)})
    source = next((value for value in sources if not value.endswith(':')), next(iter(sources), ''))
    return EstimationCurveSpec(source=source)


def estimation_item(spec, kind, index):
    """estimation.<kind>[index]; KeyError when the study has no estimation or no such entry (forms then rebuild)."""
    estimation = spec.study.estimation
    items = _items(estimation, kind) if estimation is not None else []
    if not 0 <= index < len(items):
        raise KeyError(f'the parameter estimation has no {_ITEM_WORDS[kind]} {index + 1}')
    return items[index]


def study_estimation(spec):
    """spec.study.estimation; KeyError when the study has none."""
    if spec.study.estimation is None:
        raise KeyError('the study has no parameter estimation')
    return spec.study.estimation


def measured_data_text(x, y, sigma=None) -> str:
    """Editor text of measured samples: 'x, y' per line, 'x, y, sigma' with a per-sample sigma list."""
    columns = [x if isinstance(x, list) else [], y if isinstance(y, list) else []]
    if isinstance(sigma, list):
        columns.append(sigma)
    return '\n'.join(', '.join('' if value is None else format_number(value) if isinstance(value, (int, float))
                               else str(value) for value in row)
                     for row in itertools.zip_longest(*columns))


def parse_measured_data(text) -> tuple:
    """((x, y, sigma list or None), None) of measured-sample text (2 or 3 plain numbers per line; '#' lines and blank
    lines skipped), or (None, problem)."""
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        parts = [part for part in re.split(r'[,;\s]+', stripped) if part]
        try:
            values = [float(part) for part in parts]
        except ValueError:
            return None, f'line {number}: expected plain numbers "x, y" or "x, y, sigma"'
        if not all(math.isfinite(value) for value in values):
            return None, f'line {number}: a value is not finite'
        if len(values) not in (2, 3):
            return None, f'line {number}: expected 2 or 3 numbers, got {len(values)}'
        rows.append(values)
    if len({len(row) for row in rows}) > 1:
        return None, 'every line needs the same number of columns (x, y or x, y, sigma)'
    sigma = [row[2] for row in rows] if rows and len(rows[0]) == 3 else None
    if sigma is not None and not all(value > 0.0 for value in sigma):
        return None, 'every sigma must be > 0'
    return ([row[0] for row in rows], [row[1] for row in rows], sigma), None


def read_measured_data(path) -> tuple:
    """(x, y, sigma list or None) of a text / CSV file: its lines of 2 or 3 numbers (headers and other lines are
    skipped); ValueError when none or when the column counts differ."""
    with open(path, encoding='utf-8-sig', errors='replace') as handle:
        lines = handle.read().splitlines()
    kept = []
    for line in lines:
        value, problem = parse_measured_data(line)
        if problem is None and value[0]:   # a line of numbers (blank, comment and header lines skipped)
            kept.append(line)
    name = os.path.basename(path)
    if not kept:
        raise ValueError(f'{name}: no line of 2 or 3 numbers (x, y or x, y, sigma)')
    value, problem = parse_measured_data('\n'.join(kept))
    if problem is not None:
        raise ValueError(f'{name}: {problem}')
    return value


# ---- editors ----
class MeasuredDataEdit(CommitTextEdit):
    """Measured samples of a curve, one 'x, y' or 'x, y, sigma' line each; commits (x, y, sigma list or None)."""

    def __init__(self, value, help_text='', parent=None):
        super().__init__(measured_data_text(*value), 'x, y  or  x, y, sigma  (one sample per line)', visible_lines=6,
                         parent=parent)
        self._help = help_text
        self.textChanged.connect(self.refresh_check)
        self.refresh_check()

    def refresh_check(self):
        value, problem = self._parse(self.toPlainText())
        info = ''
        if value is not None:
            info = _count_text(len(value[0]), 'sample') + (', each with its own sigma' if value[2] is not None else '')
        mark_invalid(self, problem)
        self.setToolTip(rich_tooltip(self._help, problem or info))

    def _text_from_value(self, value):
        return measured_data_text(*value)

    def _parse(self, text):
        return parse_measured_data(text)


class CurveSourceEdit(CommitLineEdit):
    """Source of a measured curve (a derived curve name or 'physics.feature:sample') with completion over entries;
    an unknown source is outlined in red but still committed (validation names it)."""

    def __init__(self, source, document, entries, help_text='', parent=None):
        super().__init__(source if isinstance(source, str) else '', parent)
        self._document = document
        self._help = help_text
        self.setPlaceholderText('derived curve or physics.feature:sample')
        completer = QtWidgets.QCompleter([value for value, _ in entries], self)
        completer.setCaseSensitivity(QtCore.Qt.CaseSensitivity.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchFlag.MatchContains)
        self.setCompleter(completer)
        self.refresh_check()

    def _parse(self, text):
        stripped = text.strip()
        return stripped, (None if stripped else 'a source is required')

    def _check(self, text):
        value, problem = self._parse(text)
        if problem is None:
            problem = curve_source_problem(self._document.spec, value) or None
        return problem, ''


def _with_button(editor, text, callback, tip=''):
    """Row widget: editor, then a tool button calling callback()."""
    row = QtWidgets.QWidget()
    row_layout = QtWidgets.QHBoxLayout(row)
    row_layout.setContentsMargins(0, 0, 0, 0)
    row_layout.addWidget(editor, 1)
    button = QtWidgets.QToolButton()
    button.setText(text)
    if tip:
        button.setToolTip(rich_tooltip(tip))
    button.clicked.connect(lambda _checked=False: callback())
    row_layout.addWidget(button)
    return row


def _separator():
    line = QtWidgets.QFrame()
    line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
    line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
    return line


# ---- SettingsPanel part ----
class EstimationFormMixin:
    """SettingsPanel part of StudySpec.estimation: the study form's 'Parameter estimation' group and the form of
    ESTIMATION_NODE (uses the panel's _document, _commit, _refreshers, _context and _report)."""

    def _add_estimation_group(self, layout, spec):
        """Study form group: checkable like Parameter sweep; on, an overview and a button to the estimation node."""
        estimation = spec.study.estimation
        box = QtWidgets.QGroupBox('Parameter estimation')
        box.setCheckable(True)
        box.setChecked(estimation is not None)
        box.setToolTip(rich_tooltip(ESTIMATION_TIP))
        box.clicked.connect(self._set_estimation_enabled)
        form = growing_form(box)
        layout.addWidget(box)
        if estimation is None:
            hint = hint_label('')

            def refresh_blocker():
                if self._document.spec.study.estimation is not None:   # switched on elsewhere (tree): rebuild
                    raise KeyError('the study has a parameter estimation now')
                blocker = estimation_blocker(self._document.spec)
                hint.setText(_INTRO + (f' {blocker}' if blocker else ''))
                box.setEnabled(not blocker)
            refresh_blocker()
            self._refreshers.append(refresh_blocker)
            form.addRow(hint)
            return
        overview = hint_label(estimation_overview_text(estimation))
        self._refreshers.append(lambda: overview.setText(estimation_overview_text(
            study_estimation(self._document.spec))))
        edit_button = QtWidgets.QPushButton('Edit parameter estimation')
        edit_button.setToolTip('Measured values and curves, estimated parameters and search settings (the Parameter '
                               'estimation node under Study)')
        edit_button.clicked.connect(lambda _checked=False: self._document.select(ESTIMATION_NODE))
        form.addRow(overview)
        form.addRow(edit_button)

    def _build_estimation(self, path, layout):
        """Form of ESTIMATION_NODE: estimated parameters, measured values, measured curves, search settings."""
        spec = self._document.spec
        estimation = study_estimation(spec)
        layout.addWidget(hint_label(f'{_INTRO} Each try builds and runs the study; Results > Estimation shows the '
                                    'fitted values, their standard errors and the misfit of every run.'))
        ctx = self._context()
        self._add_estimated_parameters(layout, spec, estimation, ctx)
        self._add_estimation_targets(layout, spec, estimation, ctx)
        self._add_estimation_curves(layout, estimation, ctx)
        self._add_estimation_settings(layout, estimation)
        remove_button = QtWidgets.QPushButton('Remove parameter estimation')
        remove_button.setToolTip('Run the study once again, without fitting (Undo brings the estimation back)')
        remove_button.clicked.connect(lambda _checked=False: self._set_estimation_enabled(False))
        row = QtWidgets.QHBoxLayout()
        row.addWidget(remove_button)
        row.addStretch(1)
        layout.addLayout(row)
        return False

    def _add_estimated_parameters(self, layout, spec, estimation, ctx):
        box = QtWidgets.QGroupBox('Estimated parameters')
        box.setToolTip(rich_tooltip(_PARAMETERS_TIP))
        form = growing_form(box)
        names = [parameter.name for parameter in spec.parameters]
        for index, item in enumerate(_items(estimation, 'parameters')):
            if index:
                form.addRow(_separator())
            name_combo = ValueComboBox([(name, name) for name in names], item.name, _PARAMETER_NAME_TIP)
            name_combo.value_committed.connect(partial(self._set_estimation_value, 'parameters', index, 'name'))
            form.addRow(tip_label(f'Parameter {index + 1}', _PARAMETER_NAME_TIP),
                        _with_button(name_combo, 'Remove', partial(self._remove_estimation_item, 'parameters', index),
                                     'Stop estimating this parameter'))
            editors = []
            for param in (_INITIAL, _LOWER, _UPPER):
                editor = ExpressionEdit(param, getattr(item, param.name), ctx)
                editor.value_committed.connect(partial(self._set_estimation_value, 'parameters', index, param.name))
                form.addRow(param_row_label(param), editor)
                editors.append((param.name, editor))
            initial_edit = editors[0][1]
            for name, editor in editors[1:]:
                editor.setPlaceholderText('no bound')

            def refresh(index=index, name_combo=name_combo, editors=editors, initial_edit=initial_edit):
                current = estimation_item(self._document.spec, 'parameters', index)
                name_combo.set_value(current.name)
                name_combo.refresh_check()
                for attribute, editor in editors:
                    editor.set_value(getattr(current, attribute))
                    editor.refresh_check()
                model_value = ctx.parameters.get(current.name)
                initial_edit.setPlaceholderText('model value' if model_value is None
                                                else f'model value ({display_number(model_value)})')
            refresh()
            self._refreshers.append(refresh)
        add_button = QtWidgets.QPushButton('Add parameter')
        add_button.setToolTip('Estimate one more model parameter')
        add_button.setEnabled(len(_items(estimation, 'parameters')) < len(names))
        add_button.clicked.connect(lambda _checked=False: self._add_estimation_item('parameters'))
        form.addRow(add_button)
        layout.addWidget(box)

    def _add_estimation_targets(self, layout, spec, estimation, ctx):
        box = QtWidgets.QGroupBox('Measured values')
        box.setToolTip(rich_tooltip(_TARGETS_TIP))
        form = growing_form(box)
        entries = [(derived.name, derived.name) for derived in spec.derived_values]
        for index, item in enumerate(_items(estimation, 'targets')):
            if index:
                form.addRow(_separator())
            derived_combo = ValueComboBox(entries, item.derived, _TARGET_NAME_TIP)
            derived_combo.value_committed.connect(partial(self._set_estimation_value, 'targets', index, 'derived'))
            form.addRow(tip_label(f'Derived value {index + 1}', _TARGET_NAME_TIP),
                        _with_button(derived_combo, 'Remove', partial(self._remove_estimation_item, 'targets', index),
                                     'Remove this measured value'))
            editors = []
            for param in (_TARGET_VALUE, _TARGET_SIGMA):
                editor = ExpressionEdit(param, getattr(item, param.name), ctx)
                editor.value_committed.connect(partial(self._set_estimation_value, 'targets', index, param.name))
                form.addRow(param_row_label(param), editor)
                editors.append((param.name, editor))
            editors[1][1].setPlaceholderText('|measured value| (relative misfit)')

            def refresh(index=index, derived_combo=derived_combo, editors=editors):
                current = estimation_item(self._document.spec, 'targets', index)
                derived_combo.set_value(current.derived)
                derived_combo.refresh_check()
                for attribute, editor in editors:
                    editor.set_value(getattr(current, attribute))
                    editor.refresh_check()
            self._refreshers.append(refresh)
        if not entries:
            form.addRow(hint_label('Add a derived value first (Result settings > Derived values): a measured value '
                                   'is compared with one.'))
        add_button = QtWidgets.QPushButton('Add measured value')
        add_button.setToolTip('Compare one more derived value with a measured value')
        add_button.setEnabled(bool(entries))
        add_button.clicked.connect(lambda _checked=False: self._add_estimation_item('targets'))
        form.addRow(add_button)
        layout.addWidget(box)

    def _add_estimation_curves(self, layout, estimation, ctx):
        box = QtWidgets.QGroupBox('Measured curves')
        box.setToolTip(rich_tooltip(_CURVES_TIP))
        form = growing_form(box)
        entries = curve_source_entries(self._document)
        for index, item in enumerate(_items(estimation, 'curves')):
            if index:
                form.addRow(_separator())
            source_edit = CurveSourceEdit(item.source, self._document, entries, _SOURCE_TIP)
            source_edit.value_committed.connect(partial(self._set_estimation_value, 'curves', index, 'source'))
            form.addRow(tip_label(f'Model curve {index + 1}', _SOURCE_TIP),
                        _with_button(source_edit, 'Remove', partial(self._remove_estimation_item, 'curves', index),
                                     'Remove this measured curve'))
            data_edit = MeasuredDataEdit((item.x, item.y, item.sigma), _DATA_TIP)
            data_edit.value_committed.connect(partial(self._set_estimation_curve_data, index))
            load_button = QtWidgets.QPushButton('Load…')
            load_button.setToolTip(rich_tooltip('Read the measured samples from a text or CSV file: the lines of 2 '
                                                '(x, y) or 3 (x, y, sigma) numbers; header lines are skipped.'))
            load_button.clicked.connect(lambda _checked=False, index=index: self._load_estimation_curve_data(index))
            data_widget = QtWidgets.QWidget()
            data_layout = QtWidgets.QVBoxLayout(data_widget)
            data_layout.setContentsMargins(0, 0, 0, 0)
            data_layout.addWidget(data_edit)
            data_layout.addWidget(load_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
            form.addRow(tip_label('Measured data', _DATA_TIP), data_widget)
            sigma_edit = ExpressionEdit(_CURVE_SIGMA, None if isinstance(item.sigma, list) else item.sigma, ctx)
            sigma_edit.value_committed.connect(partial(self._set_estimation_value, 'curves', index, 'sigma'))
            form.addRow(param_row_label(_CURVE_SIGMA), sigma_edit)

            def refresh(index=index, source_edit=source_edit, data_edit=data_edit, sigma_edit=sigma_edit):
                current = estimation_item(self._document.spec, 'curves', index)
                source_edit.set_value(current.source)
                source_edit.refresh_check()
                data_edit.set_value((current.x, current.y, current.sigma))
                per_sample = isinstance(current.sigma, list)
                sigma_edit.set_value(None if per_sample else current.sigma)
                sigma_edit.setEnabled(not per_sample)
                sigma_edit.setPlaceholderText('per sample: third data column' if per_sample
                                              else 'RMS of the measured y')
                sigma_edit.refresh_check()
            refresh()
            self._refreshers.append(refresh)
        if not entries:
            form.addRow(hint_label('No result curve in the model yet: add a probe, flux monitor or other curve '
                                   'result, or a derived curve.'))
        add_button = QtWidgets.QPushButton('Add measured curve')
        add_button.setToolTip('Compare one more result or derived curve with measured samples')
        add_button.clicked.connect(lambda _checked=False: self._add_estimation_item('curves'))
        form.addRow(add_button)
        layout.addWidget(box)

    def _add_estimation_settings(self, layout, estimation):
        box = QtWidgets.QGroupBox('Search')
        form = growing_form(box)

        def tip(attribute):
            return _SETTING_ROWS[attribute][1]

        method_combo = ValueComboBox([(method, ESTIMATION_METHOD_LABELS.get(method, method))
                                      for method in ESTIMATION_METHODS], estimation.method, tip('method'))
        runs_spin = IntSpinBox(1, 100_000, estimation.max_evaluations, help_text=tip('max_evaluations'))
        # Nelder-Mead takes no derivatives: no step row (a method edit rebuilds)
        step_edit = NumberEdit(estimation.diff_step, tip('diff_step')) if estimation.method == 'least_squares' else None
        tolerance_edit = NumberEdit(estimation.tolerance, tip('tolerance'))
        for attribute, editor in (('method', method_combo), ('max_evaluations', runs_spin), ('diff_step', step_edit),
                                  ('tolerance', tolerance_edit)):
            if editor is None:
                continue
            editor.value_committed.connect(partial(self._set_estimation_setting, attribute))
            form.addRow(tip_label(_SETTING_ROWS[attribute][0], tip(attribute)), editor)

        def refresh():
            current = study_estimation(self._document.spec)
            method_combo.set_value(current.method)
            method_combo.refresh_check()
            runs_spin.refresh_value(current.max_evaluations)
            for editor, value in ((step_edit, current.diff_step), (tolerance_edit, current.tolerance)):
                if editor is not None:
                    editor.set_value(value)
                    editor.refresh_check()
        self._refreshers.append(refresh)
        layout.addWidget(box)

    # ---- edits ----
    def _set_estimation_enabled(self, checked):
        if self._commit('study', estimation_toggle(checked), 'Parameter estimation', rebuild=True) and checked:
            self._document.select(ESTIMATION_NODE)

    def _set_estimation_setting(self, attribute, value):
        def apply(spec):
            setattr(study_estimation(spec), attribute, value)
        self._commit('study', apply, f'Set estimation {_SETTING_ROWS[attribute][0].lower()}',
                     rebuild=attribute == 'method')   # the method shows / hides the derivative step

    def _set_estimation_value(self, kind, index, attribute, value):
        def apply(spec):
            setattr(estimation_item(spec, kind, index), attribute, value)
        self._commit('study', apply, f'Set estimation {_ITEM_WORDS[kind]} {index + 1} {attribute}')

    def _set_estimation_curve_data(self, index, value):
        """Store measured samples (x, y, sigma list or None); a list sigma is replaced, or dropped without one."""
        x, y, sigma = value

        def apply(spec):
            curve = estimation_item(spec, 'curves', index)
            curve.x, curve.y = list(x), list(y)
            if sigma is not None:
                curve.sigma = list(sigma)
            elif isinstance(curve.sigma, list):
                curve.sigma = None
        self._commit('study', apply, f'Set estimation measured curve {index + 1} data')

    def _load_estimation_curve_data(self, index):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Load measured data', start_directory(self._document),
                                                        _DATA_FILE_FILTER)
        if not path:
            return
        remember_directory(path)
        try:
            value = read_measured_data(path)
        except (OSError, ValueError) as exc:
            self._report(f'Load measured data: {exc}')
            return
        self._set_estimation_curve_data(index, value)

    def _add_estimation_item(self, kind):
        sources = [value for value, _ in curve_source_entries(self._document)] if kind == 'curves' else ()

        def apply(spec):
            study_estimation(spec)
            getattr(spec.study.estimation, kind).append(new_estimation_item(spec, kind, sources))
        self._commit('study', apply, f'Add estimation {_ITEM_WORDS[kind]}', rebuild=True)

    def _remove_estimation_item(self, kind, index):
        def apply(spec):
            estimation_item(spec, kind, index)
            del getattr(spec.study.estimation, kind)[index]
        self._commit('study', apply, f'Remove estimation {_ITEM_WORDS[kind]} {index + 1}', rebuild=True)
