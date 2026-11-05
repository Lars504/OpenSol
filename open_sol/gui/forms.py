"""Qt editors for model values (GUI thread, CPU only): ExpressionEdit, per-Param editors, ParamForm, SelectionPicker.
Every editor has value_committed(object), set_value(value) and refresh_check(); owners apply commits to the document."""
import html
import math
import numbers

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.constants import FACE_LABELS, FACES, FACES_3D
from ..model.analytic import split_curve_input
from ..model.expressions import ExpressionError, evaluate_scalar, is_identifier, referenced_names
from ..model.geometry import (GEOMETRY_KINDS, boundary_entities, domain_entities, interface_entities, line_entities,
                              point_entities)
from ..model.schema import check_value, fill_defaults, param_visible

ERROR_COLOR = '#c62828'
HINT_COLOR = '#616161'
NO_VALUE = '–'                   # text of a missing / non-finite number
# derived-value operation -> GUI text (settings drop-downs, model tree, Derived values table)
DERIVED_OPERATION_LABELS = (('mean', 'Mean'), ('max', 'Maximum'), ('min', 'Minimum'), ('integral', 'Integral'),
                            ('last', 'Last sample'), ('at', 'Value at a position'),
                            ('argmax', 'Position of the maximum'), ('fwhm', 'Width at half maximum'),
                            ('polyfit', 'Polynomial fit'), ('exp_fit', 'Exponential fit'),
                            ('lorentz_fit', 'Lorentzian fit'), ('two_exp_fit', 'Two-exponential fit'),
                            ('two_wave_fit', 'Two-wave fit (complex)'), ('richardson', 'Richardson extrapolation'),
                            ('multi_lorentz_fit', 'Several Lorentzian peaks'),
                            ('pole_fit', 'Resonance poles (complex)'),
                            ('mode_volume', 'Mode volume'), ('expression', 'Expression of other values'))
SELECTION_KIND_LABELS = (('domain', 'Domain'), ('boundary', 'Boundary'), ('point', 'Point'), ('line', 'Line'),
                         ('interface', 'Interface'))   # selection kind -> GUI text (settings Kind, model tree)
_SI_PREFIXES = ((1e9, 'G'), (1e6, 'M'), (1e3, 'k'), (1.0, ''), (1e-3, 'm'), (1e-6, 'µ'), (1e-9, 'n'),
                (1e-12, 'p'), (1e-15, 'f'), (1e-18, 'a'))
_PREFIXED_UNITS = frozenset({'m', 's', 'Hz', 'W', 'V', 'A'})   # editor readback with an SI prefix (K: no 'kK')
_ENTITY_LISTERS = {'domain': domain_entities, 'boundary': boundary_entities, 'point': point_entities,
                   'line': line_entities, 'interface': interface_entities}
_EMPTY_SELECTION_HINTS = {
    'domain': 'No objects to select yet: draw one on the canvas, or right-click Geometry in the model tree.',
    'boundary': 'No edges to select yet: add a geometry object first.',
    'point': 'No point objects: add one with the canvas Point tool or the model tree.',
    'line': 'No line objects: add a line segment with the canvas Line tool or the model tree.',
    'interface': 'No objects whose outline could be read yet: draw one on the canvas first.',
}
_ALL_ENTRY_TIP = ('The whole simulated region.\n\n'
                  'A material on it is the background: it fills only the cells no other material covers.')
_STYLE_SELECTORS = ((QtWidgets.QComboBox, 'QComboBox'), (QtWidgets.QAbstractButton, 'QAbstractButton'),
                    (QtWidgets.QLabel, 'QLabel'), (QtWidgets.QAbstractSpinBox, 'QAbstractSpinBox'))


# ---- text helpers ----
def format_number(value):
    """Shortest round-trip editing text of a number ('2e-06' -> '2e-6', '300.0' -> '300')."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return str(int(value))
    mantissa, _, exponent = repr(float(value)).partition('e')
    if mantissa.endswith('.0'):
        mantissa = mantissa[:-2]
    if not exponent:
        return mantissa
    sign = '-' if exponent.startswith('-') else ''
    return f"{mantissa}e{sign}{exponent.lstrip('+-').lstrip('0') or '0'}"


def display_label(text):
    """GUI text of a snapshot / field label: the ASCII 'lambda = <nm> nm' of core.wavelengths as 'λ = <nm> nm' (the
    results combos' wording); None / '' unchanged. Labels stay ASCII in the model (snapshot matching, files)."""
    return text.replace('lambda = ', 'λ = ') if text else text


def derived_operation_text(operation):
    """GUI text of a derived-value operation (DERIVED_OPERATION_LABELS; an unknown one as stored)."""
    return dict(DERIVED_OPERATION_LABELS).get(operation, str(operation))


def display_number(value):
    """Compact (6 significant digits) text of an evaluated number."""
    return f'{value:.6g}'


def format_quantity(value, unit, digits=4):
    """'12.5 ns'-style text: SI prefix (G to a, the smallest below that) and `digits` significant digits (editor
    readback: 6); NO_VALUE for None, bools and non-finite values."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool) or not math.isfinite(value):
        return NO_VALUE
    if value == 0:
        return f'0 {unit}'
    magnitude = abs(value)
    scale, prefix = next(((scale, prefix) for scale, prefix in _SI_PREFIXES if magnitude >= scale),
                         _SI_PREFIXES[-1])
    return f'{value / scale:.{digits}g} {prefix}{unit}'


def value_text(value):
    """Editor text of a raw spec value: numbers formatted, expressions as written, None empty."""
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, numbers.Real):
        return format_number(value)
    return str(value)


def parse_number_or_text(text, integer=False):
    """Finite numeric literal -> float (int when `integer` and integral); anything else -> the stripped text."""
    stripped = text.strip()
    try:
        number = float(stripped)
    except ValueError:
        return stripped
    if not math.isfinite(number):
        return stripped
    if integer and number.is_integer():
        return int(number)
    return number


def face_cell_counts_text(cells_per_face):
    """'left 15, right 15, bottom 15, top 15' of a {face: cell count} dict (absorbing layer wording); a dict with
    z faces (3D) adds 'lower z 15, upper z 15'."""
    faces = FACES_3D if any(face in cells_per_face for face in FACES_3D[len(FACES):]) else FACES
    return ', '.join(f"{FACE_LABELS[face].lower().removesuffix(' edge').removesuffix(' face')} "
                     f"{cells_per_face.get(face, 0)}" for face in faces)


def _shown_unit(unit):
    """Unit text shown next to a value: '' for the dimensionless unit '1'."""
    return '' if unit == '1' else unit


def param_label(param):
    """Form label: label (or name) plus [unit], without a unit when dimensionless ('' or '1')."""
    text = param.label or param.name
    unit = _shown_unit(param.unit)
    return f'{text} [{unit}]' if unit else text


def param_tooltip(param):
    """Tooltip text of a Param (tip, else help) plus a paragraph on what a float value may reference."""
    paragraphs = [param.tip or param.help]
    if param.kind == 'float' and param.state_dependent:
        paragraphs.append("Can be an expression of position x, y, of other physics' fields, such as the "
                          'temperature, and of time t.')
    elif param.kind == 'float' and param.field_dependent and param.time_dependent:
        paragraphs.append('Can be an expression of position x and y, or of time t (not both).')
    elif param.kind == 'float' and param.field_dependent:
        paragraphs.append('Can be an expression of position x and y.')
    elif param.kind == 'float' and param.time_dependent:
        paragraphs.append('Can be an expression of time t, for example a value that switches on or oscillates.')
    return '\n\n'.join(paragraph for paragraph in paragraphs if paragraph)


def rich_tooltip(*texts):
    """Rich-text tooltip (Qt word-wraps rich text, not plain text) of texts whose blank-line separated paragraphs
    become <p> and whose '- ' lines become list items; '' when every text is empty."""
    blocks = []
    for text in texts:
        for paragraph in (text or '').split('\n\n'):
            lines = [line.strip() for line in paragraph.strip().splitlines() if line.strip()]
            items = [html.escape(line[2:]) for line in lines if line.startswith('- ')]
            plain = [html.escape(line) for line in lines if not line.startswith('- ')]
            if plain:
                blocks.append('<p>' + '<br>'.join(plain) + '</p>')
            if items:
                blocks.append('<ul style="margin-left: 12px; -qt-list-indent: 0;">'
                              + ''.join(f'<li>{item}</li>' for item in items) + '</ul>')
    return ''.join(blocks)


def table_number(value, digits=6):
    """Table text of a number ('' for None, 'nan' / 'inf' as such, non-numbers as they are)."""
    if value is None:
        return ''
    try:
        return f'{float(value):.{digits}g}'
    except (TypeError, ValueError):
        return str(value)


def read_only_table(columns, tip):
    """QTableWidget with these header labels (none: set when filled), no row numbers, the last column stretched,
    read-only, with a rich tooltip."""
    table = QtWidgets.QTableWidget(0, len(columns))
    if columns:
        table.setHorizontalHeaderLabels(list(columns))
    table.verticalHeader().hide()
    table.horizontalHeader().setStretchLastSection(True)
    table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setToolTip(rich_tooltip(tip))
    return table


def fill_table(table, rows):
    """Replace the rows of a table by rows of texts, then fit the columns."""
    table.setRowCount(len(rows))
    for row, texts in enumerate(rows):
        for column, text in enumerate(texts):
            table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
    table.resizeColumnsToContents()


def growing_form(widget=None):
    """QFormLayout (installed on `widget` when given) whose fields grow to the available width."""
    form = QtWidgets.QFormLayout(widget)
    form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


def tip_label(text, tip):
    """Form row label with a rich tooltip."""
    label = QtWidgets.QLabel(text)
    label.setToolTip(rich_tooltip(tip))
    return label


def param_row_label(param):
    """Label of a Param (label + unit) carrying its tooltip and What's This help."""
    label = QtWidgets.QLabel(param_label(param))
    set_param_help(label, param)
    return label


def set_param_help(widget, param):
    """Rich tooltip of the Param's tip and its full help text as What's This text (Shift+F1)."""
    widget.setToolTip(rich_tooltip(param_tooltip(param)))
    widget.setWhatsThis(rich_tooltip(param.help))


def mark_invalid(widget, message):
    """Red outline (line edits) or red text (other widgets) while `message` is set; plain style otherwise."""
    if not message:
        style = ''
    elif isinstance(widget, QtWidgets.QLineEdit):
        style = f'QLineEdit {{ border: 1px solid {ERROR_COLOR}; }}'
    else:
        selector = next((name for widget_type, name in _STYLE_SELECTORS if isinstance(widget, widget_type)), '')
        style = f'{selector} {{ color: {ERROR_COLOR}; }}' if selector else f'color: {ERROR_COLOR};'
    if widget.styleSheet() != style:
        widget.setStyleSheet(style)


def _show_problem(widget, help_text, problem, info=''):
    """mark_invalid(widget, problem) plus a rich tooltip of the help text and the problem (else the info line)."""
    mark_invalid(widget, problem)
    widget.setToolTip(rich_tooltip(help_text, problem or info))


class _FocusWheelMixin:
    """The mouse wheel acts only while the widget is focused (scrolling a form never changes a value)."""

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


def hint_label(text):
    """Grey, word-wrapped plain-text label."""
    label = QtWidgets.QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
    label.setStyleSheet(f'QLabel {{ color: {HINT_COLOR}; }}')
    return label


def error_label(text):
    """Red, word-wrapped plain-text label."""
    label = hint_label(text)
    label.setStyleSheet(f'QLabel {{ color: {ERROR_COLOR}; }}')
    return label


def _same_value(first, second):
    """Equality of JSON scalars that keeps True distinct from 1."""
    if isinstance(first, bool) or isinstance(second, bool):
        return first is second
    try:
        return bool(first == second)
    except (TypeError, ValueError):
        return False


def _split_top_level(text, separator):
    """Split on `separator` outside parentheses (so 'min(a, b), 2' keeps the call); parts stripped."""
    parts, current, depth = [], [], 0
    for char in text:
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        if char == separator and depth == 0:
            parts.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    parts.append(''.join(current).strip())
    return parts


def list_text(kind, value):
    """Editor text of a float_list ('a, b, c') or points ('x, y; x, y') value."""
    if value is None:
        return ''
    if not isinstance(value, (list, tuple)):
        return value_text(value)
    if kind == 'points':
        return '; '.join(', '.join(value_text(coordinate) for coordinate in point)
                         if isinstance(point, (list, tuple)) else value_text(point) for point in value)
    return ', '.join(value_text(item) for item in value)


def table_text(value):
    """Editor text of a 'table' value: one row per line, entries separated by ', '."""
    if not isinstance(value, (list, tuple)):
        return value_text(value)
    return '\n'.join(', '.join(value_text(item) for item in row) if isinstance(row, (list, tuple))
                     else value_text(row) for row in value)


def parse_table(text):
    """Rows of table text: one per non-empty line, entries split at top-level commas or tabs (a line with neither at
    whitespace, for pasted number columns); entries are numbers or expression text."""
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ',' in stripped or '\t' in stripped:
            parts = _split_top_level(stripped.replace('\t', ','), ',')
        else:
            parts = stripped.split()
        rows.append([parse_number_or_text(part) for part in parts])
    return rows


def _cuts_text(value):
    """Editor text 'axis, position; ...' of a StudySpec.stored_cuts value."""
    if not isinstance(value, (list, tuple)):
        return value_text(value)
    return '; '.join(', '.join(value_text(item) for item in cut) if isinstance(cut, (list, tuple))
                     else value_text(cut) for cut in value)


def parse_list(kind, text):
    """(list value, problem) of float_list / points text; items are numbers or expression text."""
    stripped = text.strip()
    if not stripped:
        return [], None
    if kind == 'float_list':
        return [parse_number_or_text(item) for item in _split_top_level(stripped, ',') if item], None
    points = []
    for index, pair_text in enumerate(part for part in _split_top_level(stripped, ';') if part):
        coordinates = _split_top_level(pair_text, ',')
        if len(coordinates) != 2 or not all(coordinates):
            return None, f'point {index + 1}: expected "x, y" (points separated by ";")'
        points.append([parse_number_or_text(coordinates[0]), parse_number_or_text(coordinates[1])])
    return points, None


def _checked_keys(checked):
    """Text keys of a stored name list, duplicates removed (malformed values give [])."""
    if not isinstance(checked, (list, tuple)):
        return []
    return list(dict.fromkeys(key for key in checked if isinstance(key, str)))


# ---- context ----
class FormContext:
    """Editor environment: document, evaluated parameters (+ errors of failed ones), state field names, names allowed
    in kind-'expression' params (None = the state names), their extra call names (functions: {name: (min, max)
    args}) and node names hidden from ref / names choices."""

    def __init__(self, document, expression_names=None, exclude_names=(), functions=None):
        self.document = document
        self.expression_names = None if expression_names is None else frozenset(expression_names)
        self.exclude_names = frozenset(exclude_names)
        self.functions = dict(functions) if functions else None
        self.parameters, self.parameter_errors, self.state_names = {}, {}, frozenset()
        self.update_from_document()

    def update_from_document(self):
        """Re-read the evaluated parameters, their errors and the state names after the spec changed."""
        self.parameters = self.document.parameters()
        self.parameter_errors = self.document.evaluation_errors()
        self.state_names = frozenset(self.document.state_names())

    def failed_parameter_problem(self, raw):
        """Message naming a failed parameter that raw (text or nested lists of texts) references, else None."""
        failed, pending = set(), [raw]
        while pending:
            item = pending.pop()
            if isinstance(item, str):
                try:
                    failed.update(referenced_names(item).intersection(self.parameter_errors))
                except ExpressionError:
                    pass
            elif isinstance(item, (list, tuple)):
                pending.extend(item)
        return f"depends on parameter '{min(failed)}', which has an error" if failed else None

    def allowed_names(self, param):
        """Names beyond parameters (and x, y) that check_value accepts for `param`."""
        if param.kind == 'expression':
            return self.state_names if self.expression_names is None else self.expression_names
        return self.state_names if param.state_dependent else frozenset()

    def reference_names(self, ref_kind):
        """Node names a 'ref' / 'names' param of `ref_kind` may point to (spec order, exclusions removed)."""
        spec = self.document.spec
        nodes = {'material': spec.materials, 'geometry': spec.geometry, 'physics': spec.physics,
                 'parameter': spec.parameters}.get(ref_kind, ())
        return [node.name for node in nodes if node.name not in self.exclude_names]


# ---- line edits ----
class CommitLineEdit(QtWidgets.QLineEdit):
    """Line edit committing on editingFinished only when its text changed; a parse problem blocks the commit
    (the text stays, outlined in red)."""
    value_committed = QtCore.Signal(object)

    def __init__(self, text='', parent=None):
        super().__init__(parent)
        self._committed_text = text
        self._help = ''
        self.setText(text)
        self.editingFinished.connect(self._on_editing_finished)
        self.textEdited.connect(self._on_text_edited)

    def set_value(self, value):
        """Show `value` when it differs from the last committed value (an unfinished edit is kept otherwise)."""
        text = self._text_from_value(value)
        if text != self._committed_text:
            self._committed_text = text
            self.setText(text)

    def refresh_check(self):
        """Re-check the current text: red outline + problem, or the info line, in the tooltip."""
        self._show_check(*self._check(self.text()))

    def _show_check(self, problem, info):
        _show_problem(self, self._help, problem, info)

    def _text_from_value(self, value):
        return value_text(value)

    def _parse(self, text):
        """(value to commit, problem that blocks the commit or None)."""
        return text, None

    def _check(self, text):
        """(problem shown in red or None, info line)."""
        return self._parse(text)[1], ''

    def _on_text_edited(self, _text):
        self.refresh_check()

    def _on_editing_finished(self):
        text = self.text()
        if text == self._committed_text:
            return
        value, problem = self._parse(text)
        if problem is not None:
            self.refresh_check()
            return
        self._committed_text = text
        self.value_committed.emit(value)


class CommitTextEdit(QtWidgets.QPlainTextEdit):
    """Multi-line editor committing on focus out when its text changed (Tab moves the focus); a _parse problem blocks
    the commit (subclasses show it in refresh_check)."""
    value_committed = QtCore.Signal(object)

    def __init__(self, text='', placeholder='', visible_lines=5, parent=None):
        super().__init__(parent)
        self._committed_text = text
        self.setPlainText(text)
        self.setPlaceholderText(placeholder)
        self.setTabChangesFocus(True)
        self.setMinimumHeight(self.fontMetrics().lineSpacing() * visible_lines + 2 * self.frameWidth() + 8)

    def set_value(self, value):
        """Show `value` when it differs from the last committed text (an unfinished edit is kept otherwise)."""
        text = self._text_from_value(value)
        if text != self._committed_text:
            self._committed_text = text
            self.setPlainText(text)
            self.refresh_check()

    def refresh_check(self):
        """Free text: nothing to check."""

    def _text_from_value(self, value):
        return value_text(value)

    def _parse(self, text):
        """(value to commit, problem that blocks the commit or None)."""
        return text, None

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if event.reason() == QtCore.Qt.FocusReason.PopupFocusReason:   # own context menu / completer: still editing
            return
        text = self.toPlainText()
        if text == self._committed_text:
            return
        value, problem = self._parse(text)
        self.refresh_check()
        if problem is None:
            self._committed_text = text
            self.value_committed.emit(value)


class CurveInputsEdit(CommitTextEdit):
    """Editor of a derived curve's inputs: one 'alias = physics.feature:sample' line each, committed as a dict."""

    def __init__(self, inputs, help_text='', parent=None):
        super().__init__(_inputs_text(inputs), 'power = em.flux1:power', visible_lines=3, parent=parent)
        self._help = help_text
        self.textChanged.connect(self.refresh_check)
        self.refresh_check()

    def refresh_check(self):
        _show_problem(self, self._help, self._parse(self.toPlainText())[1])

    def _text_from_value(self, value):
        return _inputs_text(value)

    def _parse(self, text):
        inputs = {}
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            alias, separator, reference = (part.strip() for part in line.partition('='))
            if not separator or not is_identifier(alias):
                return None, f"line {number}: expected 'alias = physics.feature:sample'"
            if alias in inputs:
                return None, f'line {number}: alias {alias!r} is used twice'
            try:
                split_curve_input(reference)
            except ValueError as exc:
                return None, f'line {number}: {exc}'
            inputs[alias] = reference
        return inputs, None


class TableEdit(CommitTextEdit):
    """Editor of a 'table' Param: one row per line (columns in the placeholder); empty text commits None when optional,
    else []. The check_value problem, else the row count, goes to the tooltip."""

    def __init__(self, param, value, ctx, parent=None):
        columns = ', '.join(f'{name} [{unit}]' if unit else name for name, unit in param.columns)
        super().__init__(table_text(value), f'{columns}  (one row per line)', visible_lines=6, parent=parent)
        self._param = param
        self._ctx = ctx
        self._help = param_tooltip(param)
        self.textChanged.connect(self.refresh_check)
        self.refresh_check()

    def refresh_check(self):
        rows, problem = self._parse(self.toPlainText())
        info = ''
        if rows is not None:
            problem = check_value(self._param, rows, self._ctx.parameters)
            if problem is not None:
                problem = self._ctx.failed_parameter_problem(rows) or problem
            else:
                info = f"{len(rows)} row{'' if len(rows) == 1 else 's'}"
        _show_problem(self, self._help, problem, info)

    def _text_from_value(self, value):
        return table_text(value)

    def _parse(self, text):
        rows = parse_table(text)
        return (None if self._param.optional and not rows else rows), None


def _inputs_text(inputs):
    """Editor text of a derived curve's inputs dict: one 'alias = reference' line each."""
    if not isinstance(inputs, dict):
        return value_text(inputs)
    return '\n'.join(f'{alias} = {value_text(reference)}' for alias, reference in inputs.items())


class NameEdit(CommitLineEdit):
    """Node-name editor: commits the stripped text when it is a valid identifier."""

    def __init__(self, name, parent=None):
        super().__init__(str(name), parent)
        self._help = 'Letters, digits and _, not starting with a digit'
        self.refresh_check()

    def _parse(self, text):
        name = text.strip()
        return name, (None if is_identifier(name)
                      else f'{name!r} is not a valid name (letters, digits and _, not starting with a digit)')


class NumberEdit(CommitLineEdit):
    """Editor of one plain finite number (no parameters or expressions); commits a float."""

    def __init__(self, value, help_text='', parent=None):
        super().__init__(value_text(value), parent)
        self._help = help_text
        self.refresh_check()

    def _parse(self, text):
        number = parse_number_or_text(text)
        if isinstance(number, float):
            return number, None
        return None, 'a plain number is required (no parameters or expressions)'


class _ParamLineEdit(CommitLineEdit):
    """CommitLineEdit of one Param: the check_value problem (a failed referenced parameter named first), else
    _describe(value) as the info line."""

    def __init__(self, text, param, ctx, parent=None):
        super().__init__(text, parent)
        self._param = param
        self._ctx = ctx
        self._help = param_tooltip(param)

    def _allowed_names(self):
        """Names beyond parameters (and x, y) that check_value accepts."""
        return frozenset()

    def _describe(self, value):
        return ''

    def _check(self, text):
        value, problem = self._parse(text)
        if problem is None:
            problem = check_value(self._param, value, self._ctx.parameters, self._allowed_names(),
                                  functions=self._ctx.functions if self._param.kind == 'expression' else None)
            if problem is not None:
                problem = self._ctx.failed_parameter_problem(value) or problem
        if problem is not None:
            return problem, ''
        return None, self._describe(value)


class ExpressionEdit(_ParamLineEdit):
    """Number-or-expression editor of one Param: red outline + tooltip from check_value, else the evaluated value +
    unit (also grey at the right edge) or the non-parameter names it depends on. Commits a float (int for kind 'int')
    or the text."""

    def __init__(self, param, value, ctx, parent=None):
        super().__init__(value_text(value), param, ctx, parent)
        self._readback = QtWidgets.QLabel(self)
        self._readback.setStyleSheet(f'QLabel {{ color: {HINT_COLOR}; background: transparent; }}')
        self._readback.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._readback.hide()
        if param.optional:
            self.setPlaceholderText('optional' if param.default is None else value_text(param.default))
        self.refresh_check()

    def _show_check(self, problem, info):
        super()._show_check(problem, info)
        readback = info if problem is None and info.startswith('= ') else ''
        self._readback.setText(readback)
        self._readback.adjustSize()
        self._readback.setVisible(bool(readback))
        self.setTextMargins(0, 0, self._readback.width() + 6 if readback else 0, 0)
        self._place_readback()

    def _place_readback(self):
        self._readback.move(self.width() - self._readback.width() - 4,
                            (self.height() - self._readback.height()) // 2)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place_readback()

    def _parse(self, text):
        stripped = text.strip()
        if not stripped:
            return None, (None if self._param.optional else 'a value is required')
        return parse_number_or_text(stripped, integer=self._param.kind == 'int'), None

    def _allowed_names(self):
        return self._ctx.allowed_names(self._param)

    def _describe(self, value):
        """'= 220 nm' (SI prefix in a _PREFIXED_UNITS unit) or '= value unit' of every number and parameter expression,
        6 significant digits; 'depends on ...' for field expressions, '' otherwise."""
        unit = self._param.unit
        if isinstance(value, str):
            try:
                depends_on = sorted(referenced_names(value).difference(self._ctx.parameters))
                if depends_on:
                    return 'depends on ' + ', '.join(depends_on)
                number = evaluate_scalar(value, self._ctx.parameters)
            except ExpressionError:
                return ''
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            number = value
        else:
            return ''
        if unit in _PREFIXED_UNITS:
            return f'= {format_quantity(number, unit, digits=6)}'
        return f'= {display_number(number)} {_shown_unit(unit)}'.rstrip()


class _TextEditor(CommitLineEdit):
    """Editor of a 'str' Param (an empty optional value commits None)."""

    def __init__(self, param, value, parent=None):
        super().__init__(value_text(value), parent)
        self._optional = param.optional
        self._help = param_tooltip(param)
        self.refresh_check()

    def _parse(self, text):
        return (None if self._optional and not text.strip() else text), None


class _ListEditor(_ParamLineEdit):
    """Editor of a 'float_list' ('a, b, c') or 'points' ('x, y; x, y') Param; items may be expressions."""

    def __init__(self, param, value, ctx, parent=None):
        super().__init__(list_text(param.kind, value), param, ctx, parent)
        self.setPlaceholderText('x, y; x, y; ...' if param.kind == 'points' else 'value, value, ...')
        self.refresh_check()

    def _text_from_value(self, value):
        return list_text(self._param.kind, value)

    def _parse(self, text):
        return parse_list(self._param.kind, text)

    def _describe(self, value):
        noun = 'point' if self._param.kind == 'points' else 'value'
        return f"{len(value)} {noun}{'' if len(value) == 1 else 's'}"


class StoredCutsEdit(_ParamLineEdit):
    """Editor of StudySpec.stored_cuts as 'axis, position; ...' text (axis x / y / z, position a number or an
    expression of parameters in m); commits [[axis, position], ...]."""

    def __init__(self, param, value, ctx, parent=None):
        super().__init__(_cuts_text(value), param, ctx, parent)
        self.setPlaceholderText('z, 1e-6; x, width/2; ...')
        self.refresh_check()

    def _text_from_value(self, value):
        return _cuts_text(value)

    def _parse(self, text):
        cuts = []
        for index, cut_text in enumerate(part for part in _split_top_level(text.strip(), ';') if part):
            parts = _split_top_level(cut_text, ',')
            if len(parts) != 2 or parts[0].lower() not in ('x', 'y', 'z') or not parts[1]:
                return None, f'cut {index + 1}: expected "axis, position" with axis x, y or z (cuts separated by ";")'
            cuts.append([parts[0].lower(), parse_number_or_text(parts[1])])
        return cuts, None

    def _check(self, text):
        cuts, problem = self._parse(text)
        for axis, position in cuts or ():
            if problem is None:
                problem = check_value(self._param, position, self._ctx.parameters, frozenset())
                if problem is not None:
                    problem = f'cut {axis}: ' + (self._ctx.failed_parameter_problem(position) or problem)
        if problem is not None:
            return problem, ''
        return None, f"{len(cuts)} cut plane{'' if len(cuts) == 1 else 's'}"


# ---- other editors ----
class _BoolEditor(QtWidgets.QCheckBox):
    """Editor of a 'bool' Param (commits on user clicks only)."""
    value_committed = QtCore.Signal(object)

    def __init__(self, value, help_text='', parent=None):
        super().__init__(parent)
        self._help = help_text
        self._value = None
        self.set_value(value)
        self.refresh_check()
        self.clicked.connect(self._on_clicked)

    def set_value(self, value):
        self._value = value
        self.setChecked(value is True)

    def refresh_check(self):
        problem = None if isinstance(self._value, bool) else f'expected true or false, got {self._value!r}'
        _show_problem(self, self._help, problem)

    def _on_clicked(self, checked):
        self._value = bool(checked)
        self.refresh_check()
        self.value_committed.emit(self._value)


class ValueComboBox(_FocusWheelMixin, QtWidgets.QComboBox):
    """Combo over (value, label) entries (item_tips: value -> item tooltip; disabled: value -> reason, listed greyed
    out); a user pick emits value_committed(value). A current value that is no entry is listed first and shown in red,
    like a current disabled one. The mouse wheel only acts while focused."""
    value_committed = QtCore.Signal(object)

    def __init__(self, entries, value, help_text='', item_tips=None, disabled=None, parent=None):
        super().__init__(parent)
        self._entries = [(entry_value, str(label)) for entry_value, label in entries]
        self._item_tips = dict(item_tips or {})
        self._disabled = dict(disabled or {})
        self._help = help_text
        self._item_values = []
        self._value = None
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.setMinimumContentsLength(8)
        self._populate(value)
        self.activated.connect(self._on_activated)

    def set_value(self, value):
        if not _same_value(value, self._value):
            self._populate(value)

    def refresh_check(self):
        problem = None if self._is_entry(self._value) else f"{value_text(self._value) or '(none)'} is not available"
        if problem is None:
            problem = self._disabled_reason(self._value)
        _show_problem(self, self._help, problem)

    def _disabled_reason(self, value):
        """Reason of a disabled entry value, else None (stored values may be unhashable)."""
        return next((reason for key, reason in self._disabled.items() if _same_value(key, value)), None)

    def _is_entry(self, value):
        return any(_same_value(entry_value, value) for entry_value, _label in self._entries)

    def _populate(self, value):
        self._value = value
        self.blockSignals(True)
        try:
            self.clear()
            self._item_values = []
            if not self._is_entry(value):
                self.addItem(f"{value_text(value) or '(none)'}  (not available)")
                self._item_values.append(value)
            for entry_value, label in self._entries:
                self.addItem(label)
                self._item_values.append(entry_value)
                reason = self._disabled_reason(entry_value)
                tip = reason or self._item_tips.get(entry_value)
                if tip:
                    self.setItemData(self.count() - 1, rich_tooltip(tip), QtCore.Qt.ItemDataRole.ToolTipRole)
                if reason:
                    self.model().item(self.count() - 1).setEnabled(False)
            self.setCurrentIndex(next((index for index, item_value in enumerate(self._item_values)
                                       if _same_value(item_value, value)), 0))
        finally:
            self.blockSignals(False)
        self.refresh_check()

    def _on_activated(self, index):
        value = self._item_values[index]
        if _same_value(value, self._value):
            return
        self._value = value
        self.refresh_check()
        self.value_committed.emit(value)


class _RefComboBox(ValueComboBox):
    """ValueComboBox of a 'ref' Param; '' means no reference (None)."""

    def set_value(self, value):
        super().set_value(None if value == '' else value)


class IntSpinBox(_FocusWheelMixin, QtWidgets.QSpinBox):
    """Integer spin box committing on Enter, focus loss or 400 ms after the last arrow / wheel step (one undo entry
    per burst); with special_text its minimum means None. A stored value it cannot show exactly is flagged in red."""
    value_committed = QtCore.Signal(object)
    STEP_COMMIT_DELAY_MS = 400

    def __init__(self, minimum, maximum, value, special_text='', help_text='', parent=None):
        super().__init__(parent)
        self._help = help_text
        self._stored_value = None
        self._step_commit_timer = QtCore.QTimer(self)
        self._step_commit_timer.setSingleShot(True)
        self._step_commit_timer.setInterval(self.STEP_COMMIT_DELAY_MS)
        self._step_commit_timer.timeout.connect(self._commit_pending)
        self.setRange(minimum, maximum)
        self.setKeyboardTracking(False)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        if special_text:
            self.setSpecialValueText(special_text)
        self.set_value(value)
        self.valueChanged.connect(self._on_value_changed)
        self.editingFinished.connect(self._commit_pending)

    def set_value(self, value):
        self._step_commit_timer.stop()
        self._has_pending_commit = False
        self._stored_value = value
        try:
            number = self.minimum() if value is None else int(value)
        except (TypeError, ValueError, OverflowError):
            number = self.minimum()
        self.blockSignals(True)
        try:
            self.setValue(min(max(number, self.minimum()), self.maximum()))
        finally:
            self.blockSignals(False)
        self.refresh_check()

    def refresh_value(self, value):
        """set_value for a change made elsewhere; a pending arrow / wheel burst is kept."""
        if not self._has_pending_commit and not _same_value(value, self._stored_value):
            self.set_value(value)

    def refresh_check(self):
        problem = (None if self._shows_stored_value()
                   else f'stored value {self._stored_value!r} cannot be shown here; set a value to replace it')
        _show_problem(self, self._help, problem)

    def _shows_stored_value(self):
        """True when the displayed number (or the special text) is exactly the stored value."""
        value, special = self._stored_value, bool(self.specialValueText())
        if value is None:
            return special
        return (isinstance(value, int) and not isinstance(value, bool) and self.minimum() <= value <= self.maximum()
                and not (special and value == self.minimum()))

    def _on_value_changed(self, value):
        """Show the new value at once; the commit waits for editingFinished or the step timer."""
        self._stored_value = None if self.specialValueText() and value == self.minimum() else value
        self.refresh_check()
        self._has_pending_commit = True
        self._step_commit_timer.start()

    def _commit_pending(self):
        """Emit the value of a pending change (no-op when nothing changed since the last commit)."""
        self._step_commit_timer.stop()
        if not self._has_pending_commit:
            return
        self._has_pending_commit = False
        self.value_committed.emit(self._stored_value)


class CheckList(QtWidgets.QListWidget):
    """Checkable list over (key, label, tooltip) entries; a user toggle emits value_committed(checked keys: earlier
    order kept, new keys appended). Checked keys that are no entry are listed in red; single=True keeps one."""
    value_committed = QtCore.Signal(object)
    key_hovered = QtCore.Signal(object)   # key of the entry under the mouse, None when it leaves the entries

    def __init__(self, entries, checked, help_text='', single=False, max_visible_rows=8, parent=None):
        super().__init__(parent)
        self._entries = list(entries)
        self._single = single
        self._max_visible_rows = max_visible_rows
        self._checked = []
        self._hovered_key = None
        self.setMouseTracking(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        if help_text:
            self.setToolTip(rich_tooltip(help_text))
        self._populate(checked)
        self.itemChanged.connect(self._on_item_changed)

    def set_value(self, checked):
        if _checked_keys(checked) != self._checked:
            self._populate(checked)

    def refresh_check(self):
        """Nothing to re-check: unavailable keys are flagged whenever the list is populated."""

    def _populate(self, checked):
        self._checked = _checked_keys(checked)
        checked_set = set(self._checked)
        listed = set()
        self.blockSignals(True)
        try:
            self.clear()
            for key, label, tooltip in self._entries:
                listed.add(key)
                self._add_item(key, label, tooltip, key in checked_set, unavailable=False)
            for key in self._checked:
                if key not in listed:
                    self._add_item(key, f'{key}  (not available here)', 'Missing or not eligible; uncheck to remove',
                                   True, unavailable=True)
            if self.count() == 0:
                placeholder = QtWidgets.QListWidgetItem('(nothing to select)')
                placeholder.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                self.addItem(placeholder)
        finally:
            self.blockSignals(False)
        self._fit_height()

    def _add_item(self, key, label, tooltip, is_checked, unavailable):
        item = QtWidgets.QListWidgetItem(label)
        item.setData(QtCore.Qt.ItemDataRole.UserRole, key)
        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(QtCore.Qt.CheckState.Checked if is_checked else QtCore.Qt.CheckState.Unchecked)
        if tooltip:
            item.setToolTip(rich_tooltip(tooltip))
        if unavailable:
            item.setForeground(QtGui.QBrush(QtGui.QColor(ERROR_COLOR)))
        self.addItem(item)

    def mouseMoveEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        self._set_hovered_key(item.data(QtCore.Qt.ItemDataRole.UserRole) if item is not None else None)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._set_hovered_key(None)
        super().leaveEvent(event)

    def _set_hovered_key(self, key):
        if key != self._hovered_key:
            self._hovered_key = key
            self.key_hovered.emit(key)

    def _fit_height(self):
        rows = min(max(self.count(), 1), self._max_visible_rows)
        row_height = max(self.sizeHintForRow(0), self.fontMetrics().height() + 4)
        self.setFixedHeight(rows * row_height + 2 * self.frameWidth() + 2)

    def _on_item_changed(self, item):
        key = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if key is None:
            return
        is_checked = item.checkState() == QtCore.Qt.CheckState.Checked
        if is_checked == (key in self._checked):
            return
        if is_checked and self._single:
            self._checked = [key]
            toggled_row = self.row(item)
            self.blockSignals(True)
            try:
                for row in range(self.count()):
                    other = self.item(row)
                    if row != toggled_row and other.data(QtCore.Qt.ItemDataRole.UserRole) is not None:
                        other.setCheckState(QtCore.Qt.CheckState.Unchecked)
            finally:
                self.blockSignals(False)
        elif is_checked:
            self._checked.append(key)
        else:
            self._checked.remove(key)
        self.value_committed.emit(list(self._checked))


# ---- Param editors and forms ----
def make_param_editor(param, value, ctx, on_commit):
    """Editor widget for one Param of any PARAM_KINDS kind; on_commit(param.name, value) after each user edit."""
    kind = param.kind
    help_text = param_tooltip(param)
    if kind in ('float', 'int', 'expression'):
        editor = ExpressionEdit(param, value, ctx)
    elif kind == 'bool':
        editor = _BoolEditor(value, help_text)
    elif kind == 'choice':
        choice_texts = dict(param.choice_labels)
        entries = [(choice, choice_texts.get(choice, str(choice))) for choice in param.choices]
        if param.optional:   # None = the type's own default (e.g. em_fdfd_nonlinear: the channel's field)
            entries.insert(0, (None, '(default)'))
        editor = ValueComboBox(entries, value, help_text)
    elif kind == 'str':
        editor = _TextEditor(param, value)
    elif kind == 'ref':
        entries = [(name, name) for name in ctx.reference_names(param.ref_kind)]
        if param.optional:
            entries.insert(0, (None, '(none)'))
        editor = _RefComboBox(entries, None if value == '' else value, help_text)
    elif kind in ('float_list', 'points'):
        editor = _ListEditor(param, value, ctx)
    elif kind == 'table':
        editor = TableEdit(param, value, ctx)
    elif kind == 'names':
        names = ctx.reference_names(param.ref_kind) if param.ref_kind else _checked_keys(value)
        editor = CheckList([(name, name, '') for name in names], value, help_text)
    else:
        raise ValueError(f"Param '{param.name}': no editor for kind {kind!r}")
    name = param.name
    editor.value_committed.connect(lambda committed: on_commit(name, committed))
    return editor


class ParamForm(QtWidgets.QWidget):
    """Form of Param editors over a raw values dict (defaults shown when absent); rows hidden per visible_if;
    on_commit(name, value) after each user edit."""

    def __init__(self, params, values, ctx, on_commit, parent=None):
        super().__init__(parent)
        self._params = tuple(params)
        self._on_commit = on_commit
        self._values = dict(values) if isinstance(values, dict) else {}
        filled = fill_defaults(self._params, self._values)
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self._rows = []
        self._shown = {}            # param name -> value shown read only instead of the form's value
        self._disabled = set()      # param names whose rows are greyed out (not used in the current state)
        for param in self._params:
            label = QtWidgets.QLabel(param_label(param))
            set_param_help(label, param)
            editor = make_param_editor(param, filled.get(param.name), ctx, self._commit)
            layout.addRow(label, editor)
            self._rows.append((param, label, editor))
        self._apply_visibility(filled)

    def refresh(self, values):
        """Show `values` (changed editors only), re-run every check and update row visibility."""
        self._values = dict(values) if isinstance(values, dict) else {}
        filled = fill_defaults(self._params, self._values)
        for param, _label, editor in self._rows:
            editor.set_value(self._shown[param.name] if param.name in self._shown else filled.get(param.name))
            editor.refresh_check()
        self._apply_visibility(filled)

    def set_shown_value(self, name, value):
        """Show `value` read only in the editor of param `name` instead of the form's value (a value the model derives,
        e.g. an automatic size); None = the form's value again, editable. Takes effect at the next refresh."""
        if value is None:
            self._shown.pop(name, None)
        else:
            self._shown[name] = value

    def set_row_enabled(self, name, enabled):
        """Grey out (False) or enable the row of param `name`; takes effect at the next refresh."""
        if enabled:
            self._disabled.discard(name)
        else:
            self._disabled.add(name)

    def set_placeholder(self, name, text):
        """Placeholder shown by the line editor of param `name` while it is empty (an optional value left unset)."""
        for param, _label, editor in self._rows:
            if param.name == name:
                editor.setPlaceholderText(text)

    def _commit(self, name, value):
        self._on_commit(name, value)
        self._values[name] = value
        self._apply_visibility(fill_defaults(self._params, self._values))

    def _apply_visibility(self, filled_values):
        for param, label, editor in self._rows:
            visible = param_visible(param, filled_values)
            label.setVisible(visible)
            editor.setVisible(visible)
            enabled = param.name not in self._shown and param.name not in self._disabled
            label.setEnabled(enabled)
            editor.setEnabled(enabled)


# ---- selections ----
def selection_entries(spec, kind, allow_outer_faces=True, allow_interior=True, object_kinds=(), exclude=()):
    """[(entity, label, tooltip)] a selection of `kind` may hold: eligible outer faces / 'all' / unconsumed
    objects (filtered by object_kinds and the boundary flags), then named selections of the same kind."""
    if kind not in _ENTITY_LISTERS:
        raise ValueError(f"unknown selection kind {kind!r}; valid: {', '.join(_ENTITY_LISTERS)}")
    geometry_kind_of = {obj.name: obj.kind for obj in spec.geometry}
    entries = []
    for name in _ENTITY_LISTERS[kind](spec):
        if name == 'all':
            if not object_kinds:
                entries.append((name, 'all  (entire domain)', _ALL_ENTRY_TIP))
        elif name in FACES_3D:   # z_lo / z_hi are listed only for 3D models
            if allow_outer_faces:
                entries.append((name, FACE_LABELS[name], 'Outer edge of the simulated region.'))
        else:
            geometry_kind = geometry_kind_of.get(name)
            if (kind == 'boundary' and not allow_interior) or (object_kinds and geometry_kind not in object_kinds):
                continue
            kind_label = (GEOMETRY_KINDS[geometry_kind].label.lower() if geometry_kind in GEOMETRY_KINDS
                          else str(geometry_kind))
            label = (f'{name}  (perimeter of {kind_label})' if kind == 'boundary' else
                     f'{name}  (outline of {kind_label})' if kind == 'interface' else f'{name}  ({kind_label})')
            entries.append((name, label, ''))
    excluded = set(exclude)
    for selection in spec.selections:
        if selection.kind == kind and selection.name not in excluded:
            entries.append((selection.name, f'{selection.name}  (named selection)',
                            ', '.join(map(str, selection.entities))))
    return entries


class SelectionPicker(QtWidgets.QWidget):
    """Checkable list of the entities a `kind` selection may hold plus named selections of that kind; outer faces
    are labelled by side. on_commit(entities) after each user toggle (max_entities=1 keeps a single entity)."""
    # (entities, fill domains) of the hovered entry ([] when the mouse leaves): the canvas hover highlight
    entities_hovered = QtCore.Signal(object, bool)

    def __init__(self, spec, kind, selected, on_commit, allow_outer_faces=True, allow_interior=True,
                 object_kinds=(), exclude=(), max_entities=None, parent=None):
        super().__init__(parent)
        entries = selection_entries(spec, kind, allow_outer_faces, allow_interior, object_kinds, exclude)
        self._list = CheckList(entries, selected, single=max_entities == 1)
        self._list.value_committed.connect(on_commit)
        self._list.key_hovered.connect(
            lambda entity: self.entities_hovered.emit([entity] if entity is not None else [], kind == 'domain'))
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._list)
        if not entries:
            layout.addWidget(hint_label(_EMPTY_SELECTION_HINTS[kind]))

    def set_selected(self, selected):
        """Show `selected` (after a change made elsewhere)."""
        self._list.set_value(selected)
