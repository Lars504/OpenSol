"""Result plot box: a pyqtgraph PlotWidget over the tabs 'Axes' (log x / y, fixed limits, tick steps; axis widths,
ticks, grid), 'Curves' (per curve show, legend text, colour, line, line width, marker, marker size; legend below),
'Labels' (fonts, texts, colours) and 'Exports' (Export plot, export file, resolution); the shared settings are a
viz.curve_figure.PlotStyle. The screen shows the style; the export redraws the plot as shown (size, fonts, ticks,
labels) with matplotlib (pdf / svg / eps / png)."""
import json
import math
import os
import weakref
from dataclasses import replace

import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from PySide6 import QtCore, QtGui, QtWidgets

from ..viz.curve_figure import (LEGEND_LOCATIONS, TICK_DIRECTIONS, AxesData, AxisTicks, ColorbarData, CurveData,
                                ImageLayer, PlotStyle, SceneData, TextLabel, save_curve_figure)
from .forms import ERROR_COLOR, rich_tooltip

_AXES = ('x', 'y')
_LIMIT_SIDES = (0, 1)   # min, max
_MAX_LOG10 = 300.0      # clamp of a log axis view coordinate before 10 ** (10 ** 309 overflows)
_MPL_LINE_STYLES = {QtCore.Qt.PenStyle.SolidLine: '-', QtCore.Qt.PenStyle.DashLine: '--',
                    QtCore.Qt.PenStyle.DotLine: ':', QtCore.Qt.PenStyle.DashDotLine: '-.',
                    QtCore.Qt.PenStyle.DashDotDotLine: (0, (3, 1, 1, 1, 1, 1))}
_MPL_MARKERS = {'o': 'o', 's': 's', 't': 'v', 't1': '^', 't2': '>', 't3': '<', 'd': 'd', '+': 'P', 'x': 'X',
                'p': 'p', 'h': 'h', 'star': '*'}
_LEGEND_GAP = 10        # px between the legend and the plot edge
# legend corner -> pyqtgraph (itemPos, parentPos, offset) inside the view box
_LEGEND_ANCHORS = {'upper right': ((1, 0), (1, 0), (-_LEGEND_GAP, _LEGEND_GAP)),
                   'upper left': ((0, 0), (0, 0), (_LEGEND_GAP, _LEGEND_GAP)),
                   'lower left': ((0, 1), (0, 1), (_LEGEND_GAP, -_LEGEND_GAP)),
                   'lower right': ((1, 1), (1, 1), (-_LEGEND_GAP, -_LEGEND_GAP))}
_LEGEND_CORNERS = tuple(_LEGEND_ANCHORS)   # 'best' order: the first corner with the fewest covered points wins
_BEST_SAMPLES = 400     # points per curve sampled along it for 'best' (long straight segments count too)
_PLOT_LAYOUT_MARGIN = 1  # pyqtgraph's PlotItem layout margin (px), restored when the legend moves back inside
_CURVES_TIP = ('Untick a curve to hide it on screen and in the export (clicking its legend sample does the same). '
               'Edit the legend text in the second column; empty restores the original name. Colour, line, line '
               'width, marker and marker size apply to that curve only (Auto = as plotted; the arrow next to a '
               'colour returns it to the plotted colour). A curve of a single point always keeps a marker unless '
               'its marker is None.')
_CHOICE_LABELS = {
    'tick_direction': {'in': 'Inside', 'out': 'Outside'},
    'legend_location': {'best': 'Best place', 'upper right': 'Top right', 'upper left': 'Top left',
                        'lower left': 'Bottom left', 'lower right': 'Bottom right', 'outside right': 'Outside right',
                        'none': 'Hidden'}}
_CHOICES = {'tick_direction': TICK_DIRECTIONS, 'legend_location': LEGEND_LOCATIONS}
_MAX_FIXED_TICKS = 50   # a fixed tick step giving more major ticks than this falls back to automatic
_STEP_TIP = ('Distance between major ticks of this plot in the unit shown on the axis (with its prefix, e.g. 0.5 for '
             '0.5 MHz). Empty = automatic. Linear axes only; a step giving more than '
             f'{_MAX_FIXED_TICKS} ticks (e.g. after switching to another plot) is ignored.')
# (group title, [(PlotStyle field, label, editor kind, editor options, tooltip)]) per tab: Labels, Axes, Exports
_LABEL_STYLE_GROUPS = (
    ('Text', (
        ('font_family', 'Font', 'font', {}, 'Font of every text. The export needs a font matplotlib finds.'),
        ('title', 'Title', 'text', {}, 'Title above the plot; empty = none.'),
        ('x_label', 'x label', 'text', {}, 'Replaces the horizontal axis label; the unit is still appended.'),
        ('y_label', 'y label', 'text', {}, 'Replaces the vertical axis label; the unit is still appended.'),
        ('label_size', 'Axis label size (pt)', 'float', {'range': (4.0, 40.0), 'step': 0.5}, ''),
        ('tick_label_size', 'Tick label size (pt)', 'float', {'range': (4.0, 40.0), 'step': 0.5}, ''),
        ('title_size', 'Title size (pt)', 'float', {'range': (4.0, 40.0), 'step': 0.5}, ''),
        ('si_prefix', 'Prefixed units', 'bool', {},
         'Linear axes show values in a prefixed unit (MHz) instead of powers of ten (1e6 Hz).'))),
    ('Colours', (
        ('foreground', 'Axes and text', 'color', {}, ''),
        ('background', 'Background', 'color', {}, ''))),
)
_AXIS_STYLE_GROUPS = (
    ('Axes and ticks', (
        ('axis_width', 'Axis line width (pt)', 'float', {'range': (0.0, 5.0), 'step': 0.1}, ''),
        ('tick_width', 'Tick width (pt)', 'float', {'range': (0.0, 5.0), 'step': 0.1}, ''),
        ('tick_length', 'Tick length (pt)', 'float', {'range': (0.0, 20.0), 'step': 0.5},
         'Major ticks; minor ticks are two thirds as long.'),
        ('tick_direction', 'Ticks', 'choice', {}, ''),
        ('minor_ticks', 'Minor ticks', 'bool', {}, 'Unlabelled ticks between the major ticks.'),
        ('boxed', 'Frame on all sides', 'bool', {}, 'Axis lines and ticks also at the top and right.'),
        ('grid', 'Grid', 'bool', {}, ''),
        ('grid_alpha', 'Grid opacity', 'float', {'range': (0.0, 1.0), 'step': 0.05}, ''))),
)
_EXPORT_STYLE_GROUPS = (
    ('Export settings', (
        ('dpi', 'Resolution (dpi)', 'int', {'range': (72, 2400)}, 'Pixels per inch of PNG exports. The exported figure has the size of '
         'the plot on screen: resize the window to change it.'),
        ('transparent', 'Transparent background', 'bool', {}, ''))),
)
_EXPORT_PLOT_TIP = ('Save the plot as shown (size on screen, fonts, curves, ticks, limits): PDF, SVG or EPS (vector, '
                    'fonts kept as text) or PNG.')
_EXPORT_PATH_TIP = ('File the plot is written to, for this plot only (not saved). Empty: Export plot asks for a file '
                    'every time. Set: Export plot overwrites this file without asking, its ending (.pdf, .svg, .eps, '
                    '.png) choosing the format.')
# the PlotStyle part of the Curves tab, below its per-curve table
_LEGEND_STYLE_GROUPS = (
    ('Legend', (
        ('legend_location', 'Position', 'choice', {},
         'Best place: the corner covering the fewest curve points, chosen on screen and used by the export.'),
        ('legend_size', 'Font size (pt)', 'float', {'range': (4.0, 40.0), 'step': 0.5}, ''),
        ('legend_frame', 'Frame', 'bool', {}, ''),
        ('legend_columns', 'Columns', 'int', {'range': (1, 8)}, ''))),
)
# per-curve choices of the Curves table: (key, label); '' = automatic (as the owner plotted it)
_CURVE_LINE_STYLES = {'solid': QtCore.Qt.PenStyle.SolidLine, 'dash': QtCore.Qt.PenStyle.DashLine,
                      'dot': QtCore.Qt.PenStyle.DotLine, 'dashdot': QtCore.Qt.PenStyle.DashDotLine,
                      'dashdotdot': QtCore.Qt.PenStyle.DashDotDotLine, 'none': QtCore.Qt.PenStyle.NoPen}
_LINE_CHOICES = (('', 'Auto'), ('solid', 'Solid'), ('dash', 'Dashed'), ('dot', 'Dotted'), ('dashdot', 'Dash-dot'),
                 ('dashdotdot', 'Dash-dot-dot'), ('none', 'No line'))
_MARKER_CHOICES = (('', 'Auto'), ('none', 'None'), ('o', 'Circle'), ('s', 'Square'), ('t1', 'Triangle'),
                   ('d', 'Diamond'), ('x', 'Cross'), ('+', 'Plus'), ('star', 'Star'))
_CURVE_COLUMNS = ('Curve', 'Legend text', 'Colour', 'Line', 'Width (pt)', 'Marker', 'Size (pt)')
_DEFAULT_CURVE_SETTINGS = {'visible': True, 'label': '', 'color': '', 'line': '', 'line_width': 1.5, 'marker': '',
                           'marker_size': 4.0}
_SIZE_EDITORS = {'line_width': ((0.1, 10.0), 0.25), 'marker_size': ((1.0, 20.0), 0.5)}   # (range, step) in pt


def _short_edit(tip, on_finished, placeholder=''):
    """Narrow QLineEdit (a limit or tick step field) calling on_finished() on editingFinished."""
    edit = QtWidgets.QLineEdit()
    edit.setMaximumWidth(90)
    if placeholder:
        edit.setPlaceholderText(placeholder)
    edit.setToolTip(tip)
    edit.editingFinished.connect(on_finished)
    return edit


class PlotAxisControls(QtWidgets.QWidget):
    """Grid 'x / y: Log, lowest, highest, tick step' + Auto acting on `plot` (this plot only, not saved); empty limit
    fields show the current view range (mouse zoom included) as grey placeholders. The owner calls apply() after
    every redraw of the plot and styles the ticks from tick_step(). log_axes False hides the Log boxes; color_axis adds a
    'colour' row (colour-bar limits and tick step: the owner reads color_limits() / tick_step('colour'), shows its
    levels with show_color_range and redraws on color_changed)."""
    log_changed = QtCore.Signal()
    ticks_changed = QtCore.Signal()
    color_changed = QtCore.Signal()

    def __init__(self, plot, parent=None, log_axes=True, color_axis=False):
        super().__init__(parent)
        self._plot = plot
        self._log_checks = {}
        self._limit_edits = {}   # (axis, side) -> QLineEdit
        self._step_edits = {}    # axis -> QLineEdit of the major tick step
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for row, axis in enumerate(_AXES):
            check = QtWidgets.QCheckBox(f'Log {axis}', self)
            check.setToolTip(f'Logarithmic {axis} axis. Values at or below zero are left out.')
            check.toggled.connect(lambda _checked, axis=axis: self._on_log_toggled(axis))
            self._log_checks[axis] = check
            if log_axes:
                layout.addWidget(check, row, 0)
            else:
                check.hide()
                layout.addWidget(QtWidgets.QLabel(axis), row, 0)
            for side, side_text in zip(_LIMIT_SIDES, ('lowest', 'highest')):
                edit = _short_edit(f'{side_text.capitalize()} {axis} value shown, in the plotted unit without prefix '
                                   '(for example Hz rather than MHz; wavelengths in nm). Leave empty for automatic; '
                                   'the grey text is the current view.',
                                   lambda axis=axis: self._apply_axis(axis, fit_unset=True))
                self._limit_edits[(axis, side)] = edit
                layout.addWidget(edit, row, 1 + side)
            step_edit = _short_edit(_STEP_TIP, lambda axis=axis: self._on_step_edited(axis), 'tick step')
            self._step_edits[axis] = step_edit
            layout.addWidget(step_edit, row, 3)
        self._color_log_check = None
        if color_axis:
            row = len(_AXES)
            self._color_log_check = QtWidgets.QCheckBox('Log colour')
            self._color_log_check.setToolTip(rich_tooltip(
                'Logarithmic colour scale of the positive field values.\n\nZero and negative values take the lowest '
                'colour; with a lowest limit at or below zero the scale starts at the smallest positive value.'))
            self._color_log_check.toggled.connect(lambda _checked: self.color_changed.emit())
            layout.addWidget(self._color_log_check, row, 0)
            for side, side_text in zip(_LIMIT_SIDES, ('lowest', 'highest')):
                edit = _short_edit(f'{side_text.capitalize()} value of the colour bar, in the unit of the field. Leave '
                                   'empty for automatic (the field\'s own range, or the replay range); the grey text '
                                   'is the current range.', self.color_changed.emit)
                self._limit_edits[('colour', side)] = edit
                layout.addWidget(edit, row, 1 + side)
            step_edit = _short_edit(f'Distance between the colour-bar ticks in the unit of the field. Empty = '
                                    f'automatic; a step giving more than {_MAX_FIXED_TICKS} ticks is ignored.',
                                    lambda: self._on_step_edited('colour'), 'tick step')
            self._step_edits['colour'] = step_edit
            layout.addWidget(step_edit, row, 3)
        auto_button = QtWidgets.QPushButton('Auto')
        auto_button.setToolTip('Clear the axis limits and tick steps and fit both axes to the data')
        auto_button.clicked.connect(lambda _checked=False: self._clear_limits())
        layout.addWidget(auto_button, 0, 4)
        layout.setColumnStretch(5, 1)
        plot.getViewBox().sigRangeChanged.connect(lambda *_args: self._show_view_range())

    def apply(self):
        """Log modes and the fixed limits of the fields (after a redraw); axes without limits keep their range."""
        self._set_log_modes()
        for axis in _AXES:
            self._apply_axis(axis, fit_unset=False)

    def set_log_modes(self, log_x, log_y):
        """Tick the Log boxes as a user would (a changed axis is refitted), e.g. from a model's plot settings."""
        for axis, flag in zip(_AXES, (log_x, log_y)):
            self._log_checks[axis].setChecked(bool(flag))

    def tick_step(self, axis):
        """Major tick step of axis in the shown (prefixed) unit; 0 = automatic (empty or invalid field)."""
        text = self._step_edits[axis].text().strip()
        try:
            step = float(text) if text else 0.0
        except ValueError:
            return 0.0
        return step if math.isfinite(step) and step > 0.0 else 0.0

    def color_limits(self):
        """(lowest, highest) of the colour row, None where empty or invalid (outlined red); (None, None) without it."""
        if ('colour', 0) not in self._limit_edits:
            return None, None
        return tuple(self._limit('colour', side, log=False) for side in _LIMIT_SIDES)

    def color_log(self):
        """True while the colour row's Log box is ticked."""
        return self._color_log_check is not None and self._color_log_check.isChecked()

    def mark_color_limits_invalid(self):
        """Outline both colour limits red (the owner found lowest >= highest and ignores them)."""
        for side in _LIMIT_SIDES:
            self._mark('colour', side, valid=False)

    def show_color_range(self, low, high):
        """Placeholders of the colour row = the levels drawn."""
        if ('colour', 0) in self._limit_edits:
            for side, value in zip(_LIMIT_SIDES, (low, high)):
                self._limit_edits[('colour', side)].setPlaceholderText(f'{value:.4g}')

    # ---- internal ----
    def _on_step_edited(self, axis):
        """Outline an invalid step red (it counts as automatic), then restyle the ticks."""
        text = self._step_edits[axis].text().strip()
        valid = not text or self.tick_step(axis) > 0.0
        self._step_edits[axis].setStyleSheet('' if valid else f'border: 1px solid {ERROR_COLOR};')
        self.ticks_changed.emit()

    def _set_log_modes(self):
        self._plot.setLogMode(x=self._log_checks['x'].isChecked(), y=self._log_checks['y'].isChecked())

    def _on_log_toggled(self, axis):
        self._set_log_modes()
        self._apply_axis(axis, fit_unset=True)   # the view coordinates of this axis changed
        self.log_changed.emit()

    def _clear_limits(self):
        for edit in (*self._limit_edits.values(), *self._step_edits.values()):
            edit.clear()
            edit.setStyleSheet('')
        for axis in _AXES:
            self._apply_axis(axis, fit_unset=True)
        self.ticks_changed.emit()
        self.color_changed.emit()

    def _apply_axis(self, axis, fit_unset):
        """Fixed range from the axis's fields (a missing side from the data bounds); no field set: fit the data when
        fit_unset, else leave the range alone."""
        view_box = self._plot.getViewBox()
        log = self._log_checks[axis].isChecked()
        limits = [self._limit(axis, side, log) for side in _LIMIT_SIDES]
        if limits == [None, None]:
            if fit_unset:
                view_box.enableAutoRange(axis=axis)
            return
        axis_index = _AXES.index(axis)
        bounds = view_box.childrenBounds()[axis_index] or view_box.viewRange()[axis_index]
        low, high = (bound if limit is None else limit for limit, bound in zip(limits, bounds))
        if not low < high:
            for side in _LIMIT_SIDES:
                self._mark(axis, side, valid=False)
            return
        if axis == 'x':
            view_box.setXRange(low, high, padding=0.0)
        else:
            view_box.setYRange(low, high, padding=0.0)

    def _limit(self, axis, side, log):
        """View coordinate of one limit field (log10 of the value on a log axis); None when empty or invalid
        (an invalid field is outlined red)."""
        text = self._limit_edits[(axis, side)].text().strip()
        if not text:
            self._mark(axis, side, valid=True)
            return None
        try:
            value = float(text)
        except ValueError:
            value = math.nan
        valid = math.isfinite(value) and (value > 0.0 or not log)
        self._mark(axis, side, valid)
        if not valid:
            return None
        return math.log10(value) if log else value

    def _mark(self, axis, side, valid):
        self._limit_edits[(axis, side)].setStyleSheet('' if valid else f'border: 1px solid {ERROR_COLOR};')

    def _show_view_range(self):
        """Placeholders = current view range in data units."""
        for axis_index, axis in enumerate(_AXES):
            values = _data_range(self._plot.getViewBox().viewRange()[axis_index], self._log_checks[axis].isChecked())
            for side in _LIMIT_SIDES:
                self._limit_edits[(axis, side)].setPlaceholderText(f'{values[side]:.4g}')


class PlotStyleForm(QtWidgets.QWidget):
    """Editors of the PlotStyle fields of `groups` (two columns of group boxes), written into `style` in place;
    changed() after each edit; its Defaults button resets only these fields."""
    changed = QtCore.Signal()

    def __init__(self, style, groups, parent=None):
        super().__init__(parent)
        self._style = style
        self._setters = {}   # PlotStyle field -> setter(value) of its editor (used by Defaults)
        grid = QtWidgets.QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        for index, (title, rows) in enumerate(groups):
            group = QtWidgets.QGroupBox(title)
            form = QtWidgets.QFormLayout(group)
            for name, label, kind, options, tip in rows:
                editor = self._editor(name, kind, options)
                if tip:
                    editor.setToolTip(tip)
                form.addRow(label, editor)
            grid.addWidget(group, index // 2, index % 2, QtCore.Qt.AlignmentFlag.AlignTop)
        defaults_button = QtWidgets.QPushButton('Defaults')
        defaults_button.setToolTip('Reset the settings above')
        defaults_button.clicked.connect(lambda _checked=False: self._reset())
        grid.addWidget(defaults_button, (len(groups) + 1) // 2, 0, QtCore.Qt.AlignmentFlag.AlignLeft)

    def _editor(self, name, kind, options):
        """Editor widget of one field; its edits set the field and emit changed."""
        value = getattr(self._style, name)
        commit = lambda new_value, name=name: self._commit(name, new_value)
        if kind == 'bool':
            editor = QtWidgets.QCheckBox()
            editor.setChecked(value)
            editor.toggled.connect(commit)
            self._setters[name] = editor.setChecked
        elif kind in ('float', 'int'):
            editor = QtWidgets.QDoubleSpinBox() if kind == 'float' else QtWidgets.QSpinBox()
            editor.setRange(*options['range'])
            editor.setKeyboardTracking(False)   # commit after typing, not per keystroke
            if kind == 'float':
                editor.setDecimals(2)
                editor.setSingleStep(options['step'])
            editor.setValue(value)
            editor.valueChanged.connect(commit)
            self._setters[name] = editor.setValue
        elif kind in ('choice', 'font'):
            editor = QtWidgets.QComboBox()
            if kind == 'choice':
                keys = list(_CHOICES[name])
                editor.addItems([_CHOICE_LABELS[name][key] for key in keys])
            else:
                keys = [''] + list(QtGui.QFontDatabase.families())
                editor.addItems(['(default)'] + keys[1:])
            editor.setCurrentIndex(keys.index(value) if value in keys else 0)
            editor.currentIndexChanged.connect(lambda index, keys=keys: commit(keys[index]))
            self._setters[name] = lambda new_value, editor=editor, keys=keys: editor.setCurrentIndex(
                keys.index(new_value) if new_value in keys else 0)
        elif kind == 'color':
            editor = _ColorButton(value)
            editor.color_changed.connect(commit)
            self._setters[name] = editor.set_color
        else:   # 'text': committed on editingFinished
            editor = QtWidgets.QLineEdit()
            editor.setPlaceholderText('automatic' if name.endswith('label') else 'none')
            editor.editingFinished.connect(lambda editor=editor: commit(editor.text().strip()))
            self._setters[name] = editor.setText
            self._setters[name](value)
        return editor

    def _commit(self, name, value):
        if getattr(self._style, name) != value:
            setattr(self._style, name, value)
            self.changed.emit()

    def refresh(self):
        """Editors from the style (edited elsewhere); each editor's own commit is a no-op as the value is unchanged."""
        for name, setter in self._setters.items():
            setter(getattr(self._style, name))

    def _reset(self):
        """Every field to its default; the field is set before its editor, so the editor's own commit is a no-op."""
        defaults = PlotStyle()
        for name, setter in self._setters.items():
            setattr(self._style, name, getattr(defaults, name))
            setter(getattr(defaults, name))
        self.changed.emit()


class _SharedPlotStyle(QtCore.QObject):
    """The one PlotStyle of a QSettings key, edited in place by every PlotPanel using that key; changed() after an
    edit (saved first)."""
    changed = QtCore.Signal()

    def __init__(self, settings_key):
        super().__init__()
        self.settings_key = settings_key
        self.style = _load_style(settings_key)

    def commit(self):
        _save_style(self.settings_key, self.style)
        self.changed.emit()


_SHARED_STYLES = {}   # settings key -> _SharedPlotStyle (created on first use, after the QApplication)


def _shared_style(settings_key):
    if settings_key not in _SHARED_STYLES:
        _SHARED_STYLES[settings_key] = _SharedPlotStyle(settings_key)
    return _SHARED_STYLES[settings_key]


class _ColorButton(QtWidgets.QPushButton):
    """Swatch button choosing a colour '#rrggbb' through QColorDialog; an automatic colour reads 'auto'."""
    color_changed = QtCore.Signal(str)

    def __init__(self, color, automatic=False, parent=None):
        super().__init__(parent)
        self.setFixedWidth(60)
        self._color = ''
        self.set_color(color, automatic)
        self.clicked.connect(lambda _checked=False: self._choose())

    def set_color(self, color, automatic=False):
        """Show color; automatic: it comes from a scheme, not from this button."""
        self._color = color
        self.setText('auto' if automatic else '')
        text_color = '#ffffff' if QtGui.QColor(color).lightness() < 128 else '#000000'
        self.setStyleSheet(f'background: {color}; color: {text_color};')

    def _choose(self):
        chosen = QtWidgets.QColorDialog.getColor(QtGui.QColor(self._color), self)
        if chosen.isValid() and chosen.name() != self._color:
            self.set_color(chosen.name())
            self.color_changed.emit(self._color)


_SUPERSCRIPT_DIGITS = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')


class _ArticleAxis(pg.AxisItem):
    """AxisItem whose minor tick levels can be switched off (show_minor_ticks); log labels always read 10ⁿ; keeps the
    ticks of its last paint (drawn_ticks) for the export."""

    def __init__(self, orientation):
        super().__init__(orientation)
        self.show_minor_ticks = True
        self.drawn_ticks = None   # [(view values, label texts '' = unlabelled)] per tick level of the last paint
        self._painted_levels = []   # tick levels of the paint in progress

    def tickSpacing(self, minVal, maxVal, size):
        """pyqtgraph's spacing, automatic steps raised until labels do not overlap; a fixed tick step giving more than
        _MAX_FIXED_TICKS major ticks on this range (e.g. kept across a plot switch) is ignored (automatic)."""
        fixed = self._tickSpacing
        if fixed is not None and abs(maxVal - minVal) <= _MAX_FIXED_TICKS * fixed[0][0]:
            return super().tickSpacing(minVal, maxVal, size)
        self._tickSpacing = None
        try:
            levels = super().tickSpacing(minVal, maxVal, size)
        finally:
            self._tickSpacing = fixed
        return self._uncrowded_levels(levels, minVal, maxVal, size)

    def _uncrowded_levels(self, levels, minVal, maxVal, size):
        """Automatic levels with the major step raised along 1-2-5 until its labels fit `size` pixels apart (pyqtgraph
        always labels the major level, so a short axis of a thin geometry stacked its labels)."""
        low, high = min(minVal, maxVal), max(minVal, maxVal)
        if not levels or self.logMode or size <= 0 or high <= low:
            return levels
        metrics = QtGui.QFontMetricsF(self.style['tickFont'] or self.font())
        gap_px = 0.6 * metrics.height()
        along_text = self.orientation in ('bottom', 'top')
        step = levels[0][0]
        raised = False
        for _attempt in range(64):
            first, last = math.ceil(low / step), math.floor(high / step)
            if last - first < 1:   # one tick or none: nothing can overlap
                break
            values = [index * step for index in range(first, last + 1)]
            if along_text:
                texts = self.tickStrings(values, self.autoSIPrefixScale * self.scale, step)
                label_px = max(metrics.horizontalAdvance(text) for text in texts)
            else:
                label_px = metrics.height()
            if step * size / (high - low) >= label_px + gap_px:
                break
            leading = round(step / 10.0 ** math.floor(math.log10(step)))
            step *= 2.5 if leading == 2 else 2.0   # 1 -> 2 -> 5 -> 10
            raised = True
        if not raised:
            return levels
        leading = round(step / 10.0 ** math.floor(math.log10(step)))
        raised_levels = [(step, 0)]
        if self.style['maxTickLevel'] >= 1:
            raised_levels.append((step / (4 if leading == 2 else 5), 0))
        return raised_levels

    def tickValues(self, minVal, maxVal, size):
        levels = super().tickValues(minVal, maxVal, size)
        self._painted_levels = levels if self.show_minor_ticks else levels[:1]
        return self._painted_levels

    def generateDrawSpecs(self, p):
        """pyqtgraph's draw specs; records each level's tick values with the label texts it actually drew."""
        specs = super().generateDrawSpecs(p)
        if specs is not None:
            drawn_texts = {text for _rect, _flags, text in specs[2]}
            self.drawn_ticks = []
            for spacing, values in self._painted_levels:
                texts = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
                self.drawn_ticks.append((list(values), [text if text in drawn_texts else '' for text in texts]))
        return specs

    def logTickStrings(self, values, scale, spacing):
        return log_tick_strings(values, scale)


def log_tick_strings(values, scale):
    """Log-axis labels (values = log10 view coordinates) always as 10ⁿ or m·10ⁿ; pyqtgraph's own mixes 0.01 with
    10⁻⁶."""
    labels = []
    for exponent in values:
        mantissa, power = (f'{10.0 ** float(exponent) * scale:.0e}').split('e')
        power = int(power)
        superscript = str(abs(power)).translate(_SUPERSCRIPT_DIGITS)
        decade = '10' + ('⁻' if power < 0 else '') + superscript
        labels.append(decade if mantissa == '1' else f'{mantissa}·{decade}')
    return labels


class _PanelSample(ItemSample):
    """Legend sample whose click hides its curve through the owning PlotPanel (on_click), so the Curves tab
    follows; pyqtgraph's own sample would hide the curve behind the panel's back."""
    on_click = None   # callable(PlotDataItem), set per sample by PlotPanel

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self.on_click is not None:
            self.on_click(self.item)
        event.accept()


class PlotPanel(QtWidgets.QGroupBox):
    """Box around a result plot: the plot over the tabs 'Axes' (PlotAxisControls + axis style), 'Curves' (per curve
    looks per apply() context, legend style), 'Labels' (text, colours) and 'Exports' (Export plot button ->
    export_requested, export_path() / Choose… -> export_path_requested, export style); the style forms edit one
    PlotStyle kept in QSettings under settings_key, shared by every panel of that key. apply() after every redraw;
    export(path) writes the plot as shown with matplotlib.
    The GUI canvas: plot = its GraphicsLayoutWidget, plot_item = its PlotItem, embed_plot False (the tabs only),
    curves_tab / log_axes False, colorbar = its ColorBarItem (styled; exported while opaque), scene_export True (images,
    PlotCurveItem / ScatterPlotItem / TextItem / InfiniteLine overlays go to the export too)."""
    export_requested = QtCore.Signal()
    export_path_requested = QtCore.Signal()

    def __init__(self, plot, title='Plot', settings_key='curve_plot_style', parent=None, plot_item=None,
                 embed_plot=True, curves_tab=True, log_axes=True, colorbar=None, scene_export=False):
        super().__init__(title, parent)
        self._plot = plot
        self._plot_item = plot_item or plot.getPlotItem()
        self._colorbar = colorbar
        self._scene_export = scene_export
        self._owner_title = None   # title text the owner last set (None = hidden)
        self._shown_title = None   # title text this panel last set
        self._shared_style = _shared_style(settings_key)
        self._style = self._shared_style.style
        self._table_rows_key = None   # (context, curve rows) the Curves table was last filled for
        self._base_looks = weakref.WeakKeyDictionary()   # PlotDataItem -> (pen, symbol) as the owner drew it
        self._owner_labels = {}   # 'bottom' / 'left' -> (text, units) the owner last set
        self._shown_labels = {}   # 'bottom' / 'left' -> (text, units) this panel last set (text: override or owner's)
        # apply() context -> curve key -> _DEFAULT_CURVE_SETTINGS keys (label / color / line / marker '' = automatic)
        self._curve_settings = {}
        self._context = None
        self._curve_rows = []       # [(curve key, original name)] of the Curves table rows
        self._effective_colors = {}   # curve key -> colour drawn (own colour or the owner's)
        self._legend_corner = 'upper right'   # where the legend sits inside the plot ('best' resolved)
        plot_item = self._plot_item
        plot_item.setAxisItems({name: _ArticleAxis(name) for name in ('left', 'bottom', 'top', 'right')})
        if plot_item.legend is not None:
            plot_item.legend.setSampleType(_PanelSample)
        plot.setAntialiasing(True)
        view_box = plot_item.getViewBox()
        view_box.sigRangeChanged.connect(lambda *_args: self._update_best_legend())
        view_box.sigResized.connect(lambda *_args: self._update_best_legend())

        self.axis_controls = PlotAxisControls(plot_item, log_axes=log_axes, color_axis=colorbar is not None)
        self.axis_controls.log_changed.connect(self._apply_style)   # SI prefixes and tick steps follow log axes
        self.axis_controls.ticks_changed.connect(self._apply_style)
        label_style_form = PlotStyleForm(self._style, _LABEL_STYLE_GROUPS)
        axis_style_form = PlotStyleForm(self._style, _AXIS_STYLE_GROUPS)
        export_style_form = PlotStyleForm(self._style, _EXPORT_STYLE_GROUPS)
        legend_style_form = PlotStyleForm(self._style, _LEGEND_STYLE_GROUPS)
        self._style_forms = (label_style_form, axis_style_form, export_style_form, legend_style_form)
        for form in self._style_forms:
            form.changed.connect(self._shared_style.commit)
        self._shared_style.changed.connect(self._on_style_changed)   # this panel's or another panel's edit
        self._curve_table = QtWidgets.QTableWidget(0, len(_CURVE_COLUMNS))
        self._curve_table.setHorizontalHeaderLabels(list(_CURVE_COLUMNS))
        self._curve_table.horizontalHeader().setSectionResizeMode(
            _CURVE_COLUMNS.index('Legend text'), QtWidgets.QHeaderView.ResizeMode.Stretch)
        self._curve_table.verticalHeader().hide()
        self._curve_table.setToolTip(_CURVES_TIP)
        self._curve_table.setMinimumHeight(120)
        self._curve_table.itemChanged.connect(self._on_curve_item_changed)
        reset_curves_button = QtWidgets.QPushButton('Show all, original names and looks')
        reset_curves_button.setToolTip('Clear every curve setting of this plot (visibility, legend text, colour, '
                                       'line, line width, marker, marker size)')
        reset_curves_button.clicked.connect(lambda _checked=False: self._reset_curves())
        curves_page = QtWidgets.QWidget()
        curves_layout = QtWidgets.QVBoxLayout(curves_page)
        curves_layout.addWidget(self._curve_table, 1)
        curves_layout.addWidget(reset_curves_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        curves_layout.addWidget(legend_style_form)
        axes_page = QtWidgets.QWidget()
        axes_layout = QtWidgets.QVBoxLayout(axes_page)
        axes_layout.addWidget(self.axis_controls)
        axes_layout.addWidget(axis_style_form)
        axes_layout.addStretch(1)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(_scroll(axes_page), 'Axes')
        if curves_tab:
            tabs.addTab(_scroll(curves_page), 'Curves')
        else:
            self._hidden_curves_page = curves_page   # kept alive: apply() still fills its (empty) table
        tabs.addTab(_scroll(label_style_form), 'Labels')
        tabs.addTab(_scroll(self._exports_page(export_style_form)), 'Exports')
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        if not embed_plot:
            layout.addWidget(tabs)
            return
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(plot)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

    def _exports_page(self, export_style_form):
        """Exports tab: Export plot button, the export file row (path, Choose…, clear) and the export style."""
        export_button = QtWidgets.QPushButton('Export plot…')
        export_button.setToolTip(_EXPORT_PLOT_TIP)
        export_button.clicked.connect(lambda _checked=False: self.export_requested.emit())
        self._export_path_edit = QtWidgets.QLineEdit()
        self._export_path_edit.setPlaceholderText('ask for a file on every export')
        self._export_path_edit.setToolTip(_EXPORT_PATH_TIP)
        self._export_path_edit.setClearButtonEnabled(True)
        choose_button = QtWidgets.QPushButton('Choose…')
        choose_button.setToolTip('Pick the export file in a dialog')
        choose_button.clicked.connect(lambda _checked=False: self.export_path_requested.emit())
        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(QtWidgets.QLabel('Export file'))
        path_row.addWidget(self._export_path_edit, 1)
        path_row.addWidget(choose_button)
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.addWidget(export_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        page_layout.addLayout(path_row)
        page_layout.addWidget(export_style_form)
        page_layout.addStretch(1)
        return page

    def export_path(self):
        """Export file of this plot ('' = ask on every export), user folder expanded."""
        text = self._export_path_edit.text().strip()
        return os.path.expanduser(text) if text else ''

    def set_export_path(self, path):
        self._export_path_edit.setText(path)

    def apply(self, context=None):
        """Log modes, limits, the Curves tab settings of `context` (e.g. the plotted result; curves are matched by
        name within it) and the shared style on the plot as drawn (call after every redraw)."""
        self._context = context
        self.axis_controls.apply()
        self._apply_style()
        self._fill_curve_table(only_if_changed=True)   # after the style: the swatches show the colours drawn

    def export(self, path):
        """Write the plot as shown (on-screen size and font, visible curves with their legend texts, log axes, view
        range, drawn ticks and axis texts) with matplotlib in the shared style; a 'best' legend goes where the screen
        put it. scene_export: the images, overlays, texts and an opaque colour bar too; the owner's title is kept."""
        plot_item = self._plot_item
        curves = [_curve_data(item, self._legend_text(key, name), self._settings_of(key))
                  for item, key, name in self._curves()
                  if item.isVisible() and item.xData is not None and len(item.xData)]
        style = replace(self._style, font_family=self._style.font_family or QtWidgets.QApplication.font().family())
        if not style.title and plot_item.titleLabel.isVisible():
            style = replace(style, title=plot_item.titleLabel.item.toPlainText().strip())
        scene = None
        if self._scene_export:
            scene_curves, images, texts = _scene_layers(plot_item, _px_per_pt(self._plot))
            curves = scene_curves + curves
            colorbar = self._colorbar
            scene = SceneData(images=tuple(images), texts=tuple(texts), equal_aspect=bool(
                plot_item.getViewBox().state['aspectLocked']),
                colorbar=(_colorbar_data(colorbar, self._color_tick_step())
                          if colorbar is not None and colorbar.opacity() > 0.0 else None))
        if style.legend_location == 'best':
            style = replace(style, legend_location=self._legend_corner)
        texts, logs, ranges, scales, ticks = [], [], [], [], []
        for name, view_range in zip(('bottom', 'left'), plot_item.getViewBox().viewRange()):
            axis = plot_item.getAxis(name)
            texts.append(axis.label.toPlainText().strip())
            logs.append(bool(axis.logMode))
            ranges.append(_data_range(view_range, axis.logMode))
            scales.append(axis.autoSIPrefixScale if axis.autoSIPrefix else 1.0)
            ticks.append(_drawn_axis_ticks(axis))
        size_inches = (self._plot.width() / self._plot.logicalDpiX(), self._plot.height() / self._plot.logicalDpiY())
        save_curve_figure(path, curves, AxesData(x_label=texts[0], y_label=texts[1], x_log=logs[0], y_log=logs[1],
                                                 x_range=ranges[0], y_range=ranges[1], x_scale=scales[0],
                                                 y_scale=scales[1], x_ticks=ticks[0], y_ticks=ticks[1]),
                          style, size_inches, scene=scene)

    # ---- curves tab ----
    def _curves(self):
        """[(PlotDataItem, curve key, original name)] in drawing order; unnamed curves are keyed 'curve <n>'. Bare
        PlotCurveItem / ScatterPlotItem overlays (also data items to pyqtgraph: the canvas) are not curves here."""
        data_items = [item for item in self._plot_item.listDataItems() if isinstance(item, pg.PlotDataItem)]
        return [(item, item.name() or f'curve {index + 1}', item.name() or '')
                for index, item in enumerate(data_items)]

    def _settings_of(self, key):
        return self._curve_settings.setdefault(self._context, {}).setdefault(key, dict(_DEFAULT_CURVE_SETTINGS))

    def _legend_text(self, key, name):
        """Legend text of a curve: the edited text, else its name ('' = not in the legend)."""
        return self._settings_of(key)['label'] or name

    def _fill_curve_table(self, only_if_changed=False):
        """One row per plotted curve: show box + key, editable legend text, colour (swatch + back-to-plotted
        button), line, line width, marker and marker size (signals blocked while filling; never called from a cell's
        own signal).
        only_if_changed: skipped while context and curve keys are those of the last fill (a live redraw)."""
        curve_rows = [(key, name) for _item, key, name in self._curves()]
        rows_key = (self._context, tuple(curve_rows))
        if only_if_changed and rows_key == self._table_rows_key:
            return
        self._table_rows_key = rows_key
        table = self._curve_table
        table.blockSignals(True)
        try:
            self._curve_rows = curve_rows
            table.setRowCount(len(self._curve_rows))
            for row, (key, name) in enumerate(self._curve_rows):
                settings = self._settings_of(key)
                show_item = QtWidgets.QTableWidgetItem(key)
                show_item.setFlags(QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                show_item.setCheckState(QtCore.Qt.CheckState.Checked if settings['visible']
                                        else QtCore.Qt.CheckState.Unchecked)
                table.setItem(row, 0, show_item)
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(self._legend_text(key, name)))
                table.setCellWidget(row, 2, self._color_cell(key))
                table.setCellWidget(row, 3, self._choice_cell(key, 'line', _LINE_CHOICES))
                table.setCellWidget(row, 4, self._size_cell(key, 'line_width'))
                table.setCellWidget(row, 5, self._choice_cell(key, 'marker', _MARKER_CHOICES))
                table.setCellWidget(row, 6, self._size_cell(key, 'marker_size'))
            table.resizeColumnToContents(0)
        finally:
            table.blockSignals(False)

    def _color_cell(self, key):
        """Colour swatch of one curve ('auto' while it keeps the plotted colour) and a button back to that colour."""
        own_color = self._settings_of(key)['color']
        swatch = _ColorButton(own_color or self._effective_colors.get(key, '#000000'), automatic=not own_color)
        swatch.color_changed.connect(lambda color, key=key, swatch=swatch: self._set_curve_color(key, color, swatch))
        automatic_button = QtWidgets.QToolButton()
        automatic_button.setText('↺')
        automatic_button.setToolTip('Back to the plotted colour')
        automatic_button.clicked.connect(lambda _checked=False, key=key, swatch=swatch:
                                         self._set_curve_color(key, '', swatch))
        cell = QtWidgets.QWidget()
        cell_layout = QtWidgets.QHBoxLayout(cell)
        cell_layout.setContentsMargins(2, 0, 2, 0)
        cell_layout.addWidget(swatch)
        cell_layout.addWidget(automatic_button)
        return cell

    def _choice_cell(self, key, option, choices):
        """Combo of one curve's 'line' / 'marker' setting ('' = Auto)."""
        combo = QtWidgets.QComboBox()
        values = [value for value, _label in choices]
        combo.addItems([label for _value, label in choices])
        current = self._settings_of(key)[option]
        combo.setCurrentIndex(values.index(current) if current in values else 0)
        combo.currentIndexChanged.connect(lambda index, key=key, option=option, values=values:
                                          self._set_curve_option(key, option, values[index]))
        return combo

    def _size_cell(self, key, option):
        """Spin box of one curve's 'line_width' / 'marker_size' in pt."""
        value_range, step = _SIZE_EDITORS[option]
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(*value_range)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setKeyboardTracking(False)   # commit after typing, not per keystroke
        spin.setValue(self._settings_of(key)[option])
        spin.valueChanged.connect(lambda value, key=key, option=option: self._set_curve_option(key, option, value))
        return spin

    def _set_curve_color(self, key, color, swatch):
        """Own colour of a curve ('' = the plotted one), then the swatch shows the colour drawn."""
        self._set_curve_option(key, 'color', color)
        swatch.set_color(color or self._effective_colors.get(key, '#000000'), automatic=not color)

    def _set_curve_option(self, key, option, value):
        self._settings_of(key)[option] = value
        self._style_curves()
        self._update_curves_and_legend()

    def _on_curve_item_changed(self, table_item):
        row = table_item.row()
        if not 0 <= row < len(self._curve_rows):
            return
        key, name = self._curve_rows[row]
        settings = self._settings_of(key)
        if table_item.column() == 0:
            settings['visible'] = table_item.checkState() == QtCore.Qt.CheckState.Checked
        else:
            text = table_item.text().strip()
            settings['label'] = text if text != name else ''
            if not text:   # empty restores the original name
                self._curve_table.blockSignals(True)
                table_item.setText(name)
                self._curve_table.blockSignals(False)
        self._update_curves_and_legend()

    def _on_sample_clicked(self, clicked_curve):
        """A legend sample click hides its curve (shown again from the Curves tab)."""
        for item, key, _name in self._curves():
            if item is clicked_curve:
                self._settings_of(key)['visible'] = False
        self._fill_curve_table()
        self._update_curves_and_legend()

    def _reset_curves(self):
        self._curve_settings.pop(self._context, None)
        self._style_curves()
        self._update_curves_and_legend()
        self._fill_curve_table()

    # ---- style preview ----
    def _on_style_changed(self):
        """The shared style changed (saved by its commit): editors and preview follow."""
        for form in self._style_forms:
            form.refresh()
        self._apply_style()

    def _apply_style(self):
        style = self._style
        plot_item = self._plot_item
        self._plot.setBackground(style.background)
        self._style_axes(plot_item, style)
        plot_item.showGrid(x=style.grid, y=style.grid, alpha=style.grid_alpha)
        title_label = plot_item.titleLabel
        current_title = title_label.text if title_label.isVisible() else None
        if current_title != self._shown_title:
            self._owner_title = current_title   # the owner set a title since ('' keeps an empty title row)
        shown_title = style.title or self._owner_title
        plot_item.setTitle(shown_title, size=f'{style.title_size}pt', color=style.foreground)
        self._shown_title = shown_title
        self._style_curves()
        self._update_curves_and_legend()

    def _update_curves_and_legend(self):
        """Curve visibility from the Curves tab, then the legend rebuilt from the visible curves' texts, styled and
        placed."""
        plot_item = self._plot_item
        for item, key, _name in self._curves():
            item.setVisible(self._settings_of(key)['visible'])
        legend = plot_item.legend
        if legend is None:
            return
        style = self._style
        legend.setLabelTextSize(f'{style.legend_size}pt')
        legend.setLabelTextColor(style.foreground)
        legend.setPen(style.foreground if style.legend_frame else None)
        legend.setBrush(style.background if style.legend_frame else None)
        legend.clear()
        for item, key, name in self._curves():
            text = self._legend_text(key, name)
            if item.isVisible() and text:
                legend.addItem(item, text)
                legend.items[-1][0].on_click = self._on_sample_clicked
        legend.setColumnCount(max(1, style.legend_columns))
        legend.setVisible(style.legend_location != 'none' and bool(legend.items))
        self._place_legend(plot_item, legend)

    def _place_legend(self, plot_item, legend):
        """'outside right': beside the view box in a right margin of the PlotItem; else a corner inside the view box
        ('best' = the corner covering the fewest curve points)."""
        layout_margin = _PLOT_LAYOUT_MARGIN
        if self._style.legend_location == 'outside right':
            if legend.parentItem() is not plot_item:
                legend.setParentItem(plot_item)
            legend_width = legend.boundingRect().width() if legend.isVisible() else 0.0
            plot_item.layout.setContentsMargins(layout_margin, layout_margin,
                                                layout_margin + legend_width + 2 * _LEGEND_GAP, layout_margin)
            legend.anchor(itemPos=(1, 0), parentPos=(1, 0),
                          offset=(-_LEGEND_GAP, plot_item.getViewBox().geometry().top()))
            return
        view_box = plot_item.getViewBox()
        if legend.parentItem() is not view_box:
            legend.setParentItem(view_box)
        plot_item.layout.setContentsMargins(layout_margin, layout_margin, layout_margin, layout_margin)
        location = self._style.legend_location
        self._legend_corner = (self._best_corner(plot_item, legend) if location == 'best'
                               else location if location in _LEGEND_ANCHORS else 'upper right')
        item_pos, parent_pos, offset = _LEGEND_ANCHORS[self._legend_corner]
        legend.anchor(itemPos=item_pos, parentPos=parent_pos, offset=offset)

    def _update_best_legend(self):
        """Re-place a 'best' legend after a zoom / resize (other positions do not depend on the view)."""
        plot_item = self._plot_item
        if self._style.legend_location == 'best' and plot_item.legend is not None and plot_item.legend.isVisible():
            self._place_legend(plot_item, plot_item.legend)

    @staticmethod
    def _best_corner(plot_item, legend):
        """Corner whose legend-sized box covers the fewest points sampled along the visible curves (view
        coordinates, so log axes count as shown); ties keep the _LEGEND_CORNERS order."""
        view_box = plot_item.getViewBox()
        (x_low, x_high), (y_low, y_high) = view_box.viewRange()
        width_px, height_px = view_box.width(), view_box.height()
        if width_px <= 0 or height_px <= 0 or x_high == x_low or y_high == y_low:
            return _LEGEND_CORNERS[0]
        legend_rect = legend.boundingRect()
        box_width = (legend_rect.width() + _LEGEND_GAP) / width_px     # fractions of the view box
        box_height = (legend_rect.height() + _LEGEND_GAP) / height_px
        points = []
        for item in plot_item.listDataItems():
            x, y = item.getData() if item.isVisible() else (None, None)
            if x is None or len(x) == 0:
                continue
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = np.asarray(x, dtype=np.float64)[finite], np.asarray(y, dtype=np.float64)[finite]
            if x.size > 1:   # resample along the curve: a long segment through a corner counts too
                along = np.linspace(0.0, x.size - 1.0, max(x.size, _BEST_SAMPLES))
                x, y = np.interp(along, np.arange(x.size), x), np.interp(along, np.arange(x.size), y)
            points.append(np.column_stack(((x - x_low) / (x_high - x_low), (y - y_low) / (y_high - y_low))))
        if not points:
            return _LEGEND_CORNERS[0]
        points = np.concatenate(points)

        def covered(corner):
            x_range = (1.0 - box_width, 1.0) if 'right' in corner else (0.0, box_width)
            y_range = (1.0 - box_height, 1.0) if 'upper' in corner else (0.0, box_height)
            return int(np.count_nonzero((points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
                                        & (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])))
        return min(_LEGEND_CORNERS, key=covered)

    def _style_axes(self, plot_item, style):
        """Axis pens, tick fonts / lengths / steps, the frame axes and the (overridden) labels with SI prefixes."""
        px_per_pt = _px_per_pt(self._plot)
        tick_length = int(round(style.tick_length * px_per_pt)) * (-1 if style.tick_direction == 'in' else 1)
        plot_item.showAxis('top', style.boxed)
        plot_item.showAxis('right', style.boxed)
        label_style = {'font-size': f'{style.label_size}pt', 'color': style.foreground}
        if style.font_family:
            label_style['font-family'] = f"'{style.font_family}'"
        for name, override in (('bottom', style.x_label), ('left', style.y_label)):
            axis = plot_item.getAxis(name)
            if (axis.labelText, axis.labelUnits) != self._shown_labels.get(name):
                self._owner_labels[name] = (axis.labelText, axis.labelUnits)   # the owner set a new label since
            owner_text, owner_units = self._owner_labels[name]
            axis.autoSIPrefix = style.si_prefix and not axis.logMode
            if not axis.autoSIPrefix:
                axis.autoSIPrefixScale = 1.0   # enableAutoSIPrefix(False) would compute a scale anyway
            axis.labelStyle = label_style
            axis.setLabel(override or owner_text, units=owner_units)
            self._shown_labels[name] = (axis.labelText, axis.labelUnits)
        x_step, y_step = (self.axis_controls.tick_step(axis) for axis in _AXES)
        for name, step in (('bottom', x_step), ('left', y_step), ('top', x_step), ('right', y_step)):
            axis = plot_item.getAxis(name)
            axis.setPen(pg.mkPen(style.foreground, width=style.axis_width * px_per_pt))
            axis.setTickPen(pg.mkPen(style.foreground, width=style.tick_width * px_per_pt))
            axis.setTextPen(style.foreground)
            # opaque ticks of every level as in the export; with the grid on pyqtgraph draws the ticks as grid lines
            axis.setStyle(tickFont=_font(style, style.tick_label_size), tickLength=tick_length,
                          showValues=name in ('bottom', 'left'), tickAlpha=None if style.grid else 1.0)
            axis.show_minor_ticks = style.minor_ticks
            labelled = plot_item.getAxis('bottom' if name in ('bottom', 'top') else 'left')
            prefix_scale = labelled.autoSIPrefixScale if labelled.autoSIPrefix else 1.0
            if step > 0.0 and not axis.logMode:   # the step is in the shown (prefixed) unit
                axis.setTickSpacing(step / prefix_scale, step / prefix_scale / 5.0)
            else:
                axis.setTickSpacing()
        if self._colorbar is not None:
            axis = self._colorbar.axis
            axis.setPen(pg.mkPen(style.foreground, width=style.axis_width * px_per_pt))
            axis.setTickPen(pg.mkPen(style.foreground, width=style.tick_width * px_per_pt))
            axis.setTextPen(style.foreground)
            axis.setStyle(tickFont=_font(style, style.tick_label_size), tickLength=tick_length, tickAlpha=1.0)
            step = self._color_tick_step()
            if step > 0.0:
                axis.setTickSpacing(step, step / 5.0)
            else:
                axis.setTickSpacing()

    def _color_tick_step(self):
        """The colour row's tick step, 0 = automatic (also when it gives more than _MAX_FIXED_TICKS ticks on the
        colour bar's levels; a log colour bar keeps pyqtgraph's decade ticks)."""
        if self._colorbar.axis.logMode:
            return 0.0
        step = self.axis_controls.tick_step('colour')
        low, high = self._colorbar.levels()
        return step if step > 0.0 and abs(high - low) <= _MAX_FIXED_TICKS * step else 0.0

    def _style_curves(self):
        """Every data item from the look its owner drew (colour, line style, marker), with the Curves tab's line width
        and marker size and its own colour / line / marker where set. Records the colour drawn per curve key."""
        px_per_pt = _px_per_pt(self._plot)
        self._effective_colors = {}
        for item, key, _name in self._curves():
            base_look = self._base_looks.get(item)
            if base_look is None:
                base_look = (pg.mkPen(item.opts['pen']), item.opts['symbol'])
                self._base_looks[item] = base_look
            base_pen, base_symbol = base_look
            settings = self._settings_of(key)
            color = settings['color'] or base_pen.color().name().lower()
            self._effective_colors[key] = color
            pen = QtGui.QPen(base_pen)
            pen.setColor(QtGui.QColor(color))
            pen.setWidthF(settings['line_width'] * px_per_pt)
            pen.setCosmetic(True)
            if settings['line']:
                pen.setStyle(_CURVE_LINE_STYLES[settings['line']])
            item.setPen(pen)
            if settings['marker']:
                symbol = None if settings['marker'] == 'none' else settings['marker']
            else:
                symbol = base_symbol
                if symbol is None and _single_sample(item):   # e.g. a one-row table: no line, so keep it visible
                    symbol = 'o'
            item.setSymbol(symbol)
            item.setSymbolSize(settings['marker_size'] * px_per_pt)
            item.setSymbolBrush(color)
            item.setSymbolPen(color)


def _single_sample(item):
    """True when a PlotDataItem holds at most one finite y sample (nothing for a line to join)."""
    y = item.yData
    return y is not None and np.count_nonzero(np.isfinite(y)) <= 1


def _scroll(widget):
    """widget in a resizable QScrollArea (a tab page taller than the tab area scrolls)."""
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(widget)
    return scroll


def _px_per_pt(widget):
    """Screen pixels per point at the widget's logical dpi (the one Qt renders pt font sizes with)."""
    return widget.logicalDpiX() / 72.0


def _drawn_axis_ticks(axis):
    """AxisTicks (data units) of the ticks an _ArticleAxis drew last: its first level major, finer levels minor
    without the major positions (pyqtgraph's log minor level repeats the decades; matplotlib would drop them and shift
    the minor labels); None before its first paint."""
    if not axis.drawn_ticks:
        return None
    to_data = ((lambda values: 10.0 ** np.asarray(values, dtype=np.float64)) if axis.logMode
               else (lambda values: np.asarray(values, dtype=np.float64)))
    (major, major_labels), finer_levels = axis.drawn_ticks[0], axis.drawn_ticks[1:]
    tolerance = 1e-4 * abs(axis.range[1] - axis.range[0])   # view coordinates
    minor_pairs = [(value, label) for values, labels in finer_levels for value, label in zip(values, labels)
                   if not np.any(np.abs(np.asarray(major, dtype=np.float64) - value) <= tolerance)]
    minor = [value for value, _label in minor_pairs]
    return AxisTicks(major=tuple(to_data(major)), major_labels=tuple(major_labels), minor=tuple(to_data(minor)),
                     minor_labels=tuple(label for _value, label in minor_pairs))


def _font(style, size_pt):
    font = QtGui.QFont(style.font_family) if style.font_family else QtGui.QFont()
    font.setPointSizeF(size_pt)
    return font


def _data_range(view_range, log):
    """[min, max] data values of a view range (10 ** of the log10 view coordinates on a log axis)."""
    if not log:
        return tuple(float(value) for value in view_range)
    return tuple(10.0 ** max(-_MAX_LOG10, min(_MAX_LOG10, float(value))) for value in view_range)


def _curve_data(item, legend_text, settings):
    """CurveData of a PlotDataItem with data (its original, not log-transformed, values) and the line width / marker
    size of its Curves tab settings; legend_text '' = none."""
    pen = pg.mkPen(item.opts['pen'])
    line_style = '' if pen.style() == QtCore.Qt.PenStyle.NoPen else _MPL_LINE_STYLES.get(pen.style(), '-')
    symbol = item.opts['symbol']
    marker = _MPL_MARKERS.get(symbol, 'o') if isinstance(symbol, str) and symbol else ''
    marker_color = pg.mkBrush(item.opts['symbolBrush']).color().name() if marker else ''
    return CurveData(x=np.array(item.xData, dtype=np.float64), y=np.array(item.yData, dtype=np.float64),
                     label=legend_text or None, color=pen.color().name(), line_style=line_style, marker=marker,
                     marker_color=marker_color, line_width=settings['line_width'],
                     marker_size=settings['marker_size'])


def _scene_layers(plot_item, px_per_pt):
    """(curves, images, texts) of the visible non-data items of the view box in z order: ImageItems as drawn (RGBA),
    PlotCurveItem lines, ScatterPlotItem markers, InfiniteLines across the view, TextItems; ROIs are left out."""
    view_box = plot_item.getViewBox()
    items = sorted((item for item in view_box.addedItems
                    if item.isVisible() and not isinstance(item, pg.PlotDataItem)), key=lambda item: item.zValue())
    (x_low, x_high), (y_low, y_high) = view_box.viewRange()
    curves, images, texts = [], [], []
    for rank, item in enumerate(items):
        zorder = 1.0 + rank
        if isinstance(item, pg.ImageItem):
            rgba = _image_rgba(item)
            if rgba is not None:
                rect = item.mapRectToView(item.boundingRect())
                images.append(ImageLayer(rgba=rgba, extent=(rect.left(), rect.right(), rect.top(), rect.bottom()),
                                         zorder=zorder))
        elif isinstance(item, pg.PlotCurveItem):
            pen = pg.mkPen(item.opts['pen'])
            if item.xData is None or len(item.xData) == 0 or pen.style() == QtCore.Qt.PenStyle.NoPen:
                continue
            x, y = np.asarray(item.xData, dtype=np.float64), np.asarray(item.yData, dtype=np.float64)
            if isinstance(item.opts.get('connect'), str) and item.opts['connect'] == 'pairs':   # NaN-split segments
                pairs = x.size // 2
                x = np.column_stack((x[:2 * pairs].reshape(pairs, 2), np.full(pairs, np.nan))).ravel()
                y = np.column_stack((y[:2 * pairs].reshape(pairs, 2), np.full(pairs, np.nan))).ravel()
            curves.append(CurveData(x=x, y=y, label=None, color=_mpl_color(pen.color()),
                                    line_style=_MPL_LINE_STYLES.get(pen.style(), '-'),
                                    line_width=_pen_width_pt(pen, px_per_pt), zorder=zorder))
        elif isinstance(item, pg.ScatterPlotItem):
            x, y = item.getData()
            if x is None or len(x) == 0:
                continue
            symbol = item.opts['symbol']
            pen, brush = pg.mkPen(item.opts['pen']), pg.mkBrush(item.opts['brush'])
            face = 'none' if brush.style() == QtCore.Qt.BrushStyle.NoBrush else _mpl_color(brush.color())
            edge = 'none' if pen.style() == QtCore.Qt.PenStyle.NoPen else _mpl_color(pen.color())
            curves.append(CurveData(x=np.asarray(x, dtype=np.float64), y=np.asarray(y, dtype=np.float64), label=None,
                                    color=edge if edge != 'none' else face, line_style='',
                                    marker=_MPL_MARKERS.get(symbol, 'o') if isinstance(symbol, str) else 'o',
                                    marker_color=face, marker_edge_color=edge,
                                    marker_size=float(item.opts['size']) / px_per_pt, zorder=zorder))
        elif isinstance(item, pg.InfiniteLine):
            pen = pg.mkPen(item.pen)
            value = float(item.value())
            vertical = abs(item.angle % 180.0 - 90.0) < 1e-6
            x, y = ([value, value], [y_low, y_high]) if vertical else ([x_low, x_high], [value, value])
            curves.append(CurveData(x=np.asarray(x), y=np.asarray(y), label=None, color=_mpl_color(pen.color()),
                                    line_style=_MPL_LINE_STYLES.get(pen.style(), '-'),
                                    line_width=_pen_width_pt(pen, px_per_pt), zorder=zorder))
        elif isinstance(item, pg.TextItem):
            anchor_x, anchor_y = float(item.anchor[0]), float(item.anchor[1])
            fill = item.fill if isinstance(item.fill, QtGui.QBrush) else pg.mkBrush(item.fill)
            point_size = item.textItem.font().pointSizeF()
            texts.append(TextLabel(x=float(item.pos().x()), y=float(item.pos().y()),
                                   text=item.textItem.toPlainText(), color=_mpl_color(QtGui.QColor(item.color)),
                                   ha='left' if anchor_x < 0.25 else 'right' if anchor_x > 0.75 else 'center',
                                   va='top' if anchor_y < 0.25 else 'bottom' if anchor_y > 0.75 else 'center',
                                   angle=float(item.angle), size=point_size if point_size > 0 else 0.0,
                                   background='' if fill.style() == QtCore.Qt.BrushStyle.NoBrush
                                   else _mpl_color(fill.color()), zorder=zorder))
    return curves, images, texts


def _image_rgba(item):
    """(rows, columns, 4) floats 0..1 of an ImageItem as drawn (levels, lookup table, opacity), row 0 at its lowest
    y; None without an image."""
    if item.image is None:
        return None
    if item.qimage is None:
        item.render()
    if item.qimage is None:
        return None
    image = item.qimage.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
    height, width, stride = image.height(), image.width(), image.bytesPerLine()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8)[:height * stride]
    rgba = raw.reshape(height, stride)[:, :4 * width].reshape(height, width, 4).astype(np.float64) / 255.0
    rgba[..., 3] *= item.opacity()
    return rgba


def _colorbar_data(colorbar, tick_step):
    """ColorbarData of a pyqtgraph ColorBarItem: its colour map as a 256-entry RGBA table, its levels (a log axis: the
    levels are log10 values) and the tick step (0 = automatic)."""
    lut = np.asarray(colorbar.colorMap().getLookupTable(nPts=256, mode='byte'), dtype=np.float64) / 255.0
    if lut.shape[1] == 3:
        lut = np.column_stack((lut, np.ones(len(lut))))
    low, high = (float(level) for level in colorbar.levels())
    log = bool(colorbar.axis.logMode)
    if log:
        low, high = 10.0 ** low, 10.0 ** high
    return ColorbarData(lut=lut, low=low, high=high, tick_step=tick_step, log=log)


def _mpl_color(color):
    """'#rrggbbaa' of a QColor (matplotlib order; Qt's HexArgb puts alpha first)."""
    return f'#{color.red():02x}{color.green():02x}{color.blue():02x}{color.alpha():02x}'


def _pen_width_pt(pen, px_per_pt):
    """Width of a cosmetic pen in pt (width 0 = one pixel)."""
    return (pen.widthF() or 1.0) / px_per_pt


def _load_style(settings_key):
    """PlotStyle stored as JSON under settings_key (defaults when missing or unreadable)."""
    text = QtCore.QSettings().value(settings_key, '')
    try:
        data = json.loads(text) if text else {}
    except (TypeError, ValueError):
        data = {}
    return PlotStyle.from_dict(data if isinstance(data, dict) else {})


def _save_style(settings_key, style):
    QtCore.QSettings().setValue(settings_key, json.dumps(style.to_dict()))
