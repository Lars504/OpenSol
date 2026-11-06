"""ResultPlotView: one result curve plot with its toolbar (owner widgets, plot / unit / series combos, Export curve
data, Compare data) over a PlotPanel (its Exports tab: Export plot) and note / error lines; it draws the xy dicts its
owner supplies."""
import os
import re
from dataclasses import dataclass

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..io.model_io import export_curve_plots, read_curve_data
from ..model.analytic import is_wavelength_xy
from ..model.expressions import error_message
from ..viz.result_plots import xy_series_groups
from .forms import ERROR_COLOR, rich_tooltip, selectable
from .helpers import ask_save_path, remember_directory, start_directory
from .plot_controls import PlotPanel

SERIES_COLORS = ('#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#7f7f7f',
                 '#bcbd22')
METRES_PER_NM = 1e-9   # every wavelength axis and label of the results panel is in nm
WAVELENGTH_AXIS_LABEL = 'Wavelength (nm)'
POINT_OPTIONS = {'symbol': 'o', 'symbolSize': 5}   # curve_options of point data (tables, spectra, sweep values)
TABLE_FILE_TYPES = (('NumPy archive', '.npz'), ('CSV table', '.csv'))   # (dialog label, suffix); first = default
EXPORT_ERRORS = (OSError, ValueError, RuntimeError, TypeError, KeyError)   # RuntimeError: e.g. a font matplotlib lacks
PLOT_FILE_TYPES = (('PDF document', '.pdf'), ('SVG drawing', '.svg'), ('EPS figure', '.eps'), ('PNG image', '.png'))
_NO_AXIS_UNITS = frozenset({'', '1', 'a.u.', 'dB', 'deg'})   # shown in the label instead of SI-prefixed axis units
_COMPARED_LINE_STYLES = (QtCore.Qt.PenStyle.DashLine, QtCore.Qt.PenStyle.DotLine, QtCore.Qt.PenStyle.DashDotLine,
                         QtCore.Qt.PenStyle.DashDotDotLine)   # one per compared set, cycled
_ALL_COMPARED_PLOTS = 'All plots (each over the plot of the same name)'   # Compare data choice of a multi-plot file
_LIVE_INTERVAL_MS = 200   # shortest interval between two live redraws
_COMPARE_TIP = ('Draw other curves over this plot, to compare runs with other parameters or other plots of this '
                'run.\n\n'
                '- Add curve data file: a file saved with Export curve data (pick one of its plots, or all of them: '
                'then each plot is drawn over the plot of the same name here), a raw result saved with Export curve '
                'data (probe, spectrum, far field or table), or any CSV file with the horizontal axis values in its '
                'first column\n'
                '- Add plot of this run: another plot of this view, updated with every run\n'
                '- Same colour = same curve position, one line style per compared set\n'
                '- Only curves of the shown unit are drawn; a set with another horizontal axis is skipped\n'
                '- Not saved with the model')
_EXPORT_DATA_TIP = ('Save the curves with the model title and the parameter values of the run, for Compare data.\n\n'
                    '- Shown plot: every curve of the chosen plot (all units), NPZ or CSV (x and one column per '
                    'curve)\n'
                    '- All plots: every plot of this view in one NPZ file')


# ---- drawing helpers shared with gui/results ----
def series_pen(index, width=1.5):
    return pg.mkPen(SERIES_COLORS[index % len(SERIES_COLORS)], width=width)


def dashed_pen(color, width=1.5):
    return pg.mkPen(color, width=width, style=QtCore.Qt.PenStyle.DashLine)


def axis_label(label, unit):
    """(label text, axis units or None): units without SI prefixes ('1', 'a.u.', dB, deg) go into the text."""
    if unit in _NO_AXIS_UNITS:
        return (label if unit in ('', '1') else f'{label} ({unit})'), None
    return label, unit


def make_plot(bottom_label='x', bottom_units=None, left_label='', left_units=None, legend=True, downsample=False):
    """PlotWidget; downsample (min / max per pixel, clip to view) only for sorted, uniformly spaced x."""
    plot = pg.PlotWidget()
    if downsample:
        plot.setDownsampling(auto=True, mode='peak')
        plot.setClipToView(True)
    plot.showGrid(x=True, y=True, alpha=0.2)
    plot.setLabel('bottom', bottom_label, units=bottom_units)
    plot.setLabel('left', left_label, units=left_units)
    if legend:
        plot.addLegend(offset=(10, 10))
    return plot


def xy_axis(xy):
    """(displayed x, label, axis units) of an xy dict: wavelengths in nm, other axes through axis_label."""
    x = np.asarray(xy['x'], dtype=np.float64)
    if is_wavelength_xy(xy):
        return x / METRES_PER_NM, WAVELENGTH_AXIS_LABEL, None
    x_label, x_units = axis_label(xy.get('x_label') or 'x', xy.get('x_unit', ''))
    return x, x_label, x_units


def curve_xy(curves, x_label, x_unit, y_label='', **extra):
    """xy dict of curves [(series name, x, y, unit)], each sorted by x: x of the first curve; a curve on other abscissae
    keeps its own in series_x {name: x}. extra: further keys (log_x, log_y, preview, ...)."""
    xy = {'kind': 'xy', 'x': np.empty(0), 'x_label': x_label, 'x_unit': x_unit, 'y_label': y_label, 'series': {},
          'series_units': {}, **extra}
    series_x = {}
    for index, (name, x, y, unit) in enumerate(curves):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        order = np.argsort(x, kind='stable')
        x, y = x[order], y[order]
        if index == 0:
            xy['x'] = x
        elif not np.array_equal(x, xy['x']):
            series_x[name] = x
        xy['series'][name] = y
        xy['series_units'][name] = unit
    if series_x:
        xy['series_x'] = series_x
    return xy


def exportable_xy(xy):
    """xy without series_x: a series on other abscissae goes onto the sorted union of every x, NaN off its own
    samples (exact values in one table); an x repeated within a series is repeated in the union, so no sample is lost."""
    series_x = xy.get('series_x')
    if not series_x:
        return xy
    own_x = {name: np.asarray(series_x.get(name, xy['x']), dtype=np.float64) for name in xy['series']}
    distinct = np.unique(np.concatenate(list(own_x.values())))
    repeats = np.zeros(distinct.size, dtype=np.int64)   # most samples any series has at each distinct x
    for x in own_x.values():
        repeats = np.maximum(repeats, np.bincount(np.searchsorted(distinct, x), minlength=distinct.size))
    union = np.repeat(distinct, repeats)
    series = {}
    for name, values in xy['series'].items():
        order = np.argsort(own_x[name], kind='stable')
        sorted_x = own_x[name][order]
        rank = np.arange(sorted_x.size) - np.searchsorted(sorted_x, sorted_x, side='left')   # k-th sample at that x
        column = np.full(union.size, np.nan)
        column[np.searchsorted(union, sorted_x, side='left') + rank] = np.asarray(values)[order]
        series[name] = column
    return dict({key: value for key, value in xy.items() if key != 'series_x'}, x=union, series=series)


def write_export(write, path, report, written='') -> bool:
    """write(path); then remember its folder and report('<written> written to <path>', False) (none for an empty
    written: the caller reports); report('Export failed: ...', True) on EXPORT_ERRORS. True when written."""
    try:
        write(path)
    except EXPORT_ERRORS as exc:
        report(f'Export failed: {error_message(exc)}', True)
        return False
    remember_directory(path)
    if written:
        report(f'{written} written to {path}', False)
    return True


def file_stem(text, default='plot'):
    """File-name stem of a plot / result name: runs of characters other than letters, digits, '-' become '_'."""
    return re.sub(r'[^\w\-]+', '_', str(text or '')).strip('_') or default


def fill_combo(combo, labels, select=0):
    """Refill a combo without firing signals, showing item `select` (the first when out of range)."""
    combo.blockSignals(True)
    combo.clear()
    combo.addItems(labels)
    if 0 <= select < len(labels):
        combo.setCurrentIndex(select)
    combo.blockSignals(False)


def _unit_group_text(unit, names):
    """Unit-combo text of one xy unit group: 'unit: series, ...'."""
    unit_text = {'': 'no unit', '1': 'dimensionless'}.get(unit, unit)
    return f"{unit_text}: {', '.join(names)}"


def _series_axis(xy, name, x_scale):
    """Displayed x of one series: its own series_x entry (divided by x_scale), else the plot's x."""
    own_x = (xy.get('series_x') or {}).get(name)
    return xy_axis(xy)[0] if own_x is None else np.asarray(own_x, dtype=np.float64) / x_scale


@dataclass(frozen=True)
class DrawnPlot:
    """One redraw for the overlays hook: plot name, xy, unit of the shown group, first_group (curves of no particular
    series go with it), x_scale (SI x per displayed x unit) and drawn ((series, colour), ...)."""
    name: str
    xy: dict
    unit: str
    first_group: bool
    x_scale: float
    drawn: tuple


class ResultPlotView(QtWidgets.QWidget):
    """Result curve plot over GUI-thread data: plots() -> [(name, xy or None)] (xy: x, x_label, x_unit, series,
    series_units, y_label; optional series_x, log_x / log_y, parameters, preview) and optional hooks overlays(plot,
    DrawnPlot), context(name), curve_options(name, xy, series), note(name, xy), metadata() -> (model title,
    parameters) or None, export_stem(name; None = all plots), extra_export_actions(menu). Never calls the bridge."""
    message = QtCore.Signal(str, bool)   # status text (export written / failed, compare problems), error flag

    def __init__(self, document, plots, *, overlays=None, context=None, curve_options=None, note=None,
                 metadata=None, export_stem=None, extra_export_actions=None, leading_widgets=(),
                 empty_text='compute first', downsample=False, parent=None):
        super().__init__(parent)
        self._document = document
        self._plots_hook = plots
        self._overlays_hook = overlays
        self._context_hook = context or (lambda name: name)
        self._curve_options_hook = curve_options or (lambda name, xy, series: {})
        self._note_hook = note
        self._metadata_hook = metadata
        self._export_stem_hook = export_stem or (lambda name: name or 'curves')
        self._extra_export_hook = extra_export_actions
        self._empty_text = empty_text
        self._plot_list = []        # [(name, xy or None)] of the last redraw
        self._group_units = []      # units parallel to the unit combo
        self._series_names = []     # series of the shown unit group, parallel to the series combo after 'All series'
        self._compared = []         # (kind 'plot' / 'match' / 'run', legend tag, payload) drawn over the shown plot
        self._log_modes = None      # (plot name, log_x, log_y) last put on the axis controls
        self._owner_error = ''      # set_error text (kept until replaced)
        self._draw_errors = []      # add_error texts of the last draw
        self._dirty = False         # plots changed since the last redraw
        self._pending = False       # a coalesced redraw is queued
        self._live_timer = QtCore.QTimer(self)
        self._live_timer.setSingleShot(True)
        self._live_timer.setInterval(_LIVE_INTERVAL_MS)
        self._live_timer.timeout.connect(self._redraw_if_dirty)

        self.plot = make_plot(downsample=downsample)
        self.panel = PlotPanel(self.plot)
        self.plot_combo = QtWidgets.QComboBox()
        self.plot_combo.setToolTip('Plot shown')
        self.plot_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.plot_combo.currentIndexChanged.connect(lambda index: self._on_plot_chosen())
        self._unit_combo = QtWidgets.QComboBox()
        self._unit_combo.setToolTip('Curves of one unit are plotted together')
        self._unit_combo.currentIndexChanged.connect(lambda index: self._on_unit_chosen())
        self.series_combo = QtWidgets.QComboBox()
        self.series_combo.setToolTip('Plot every curve of the chosen unit, or one of them alone')
        self.series_combo.currentIndexChanged.connect(lambda index: self._draw())
        self.panel.export_requested.connect(self.export_plot)
        self.panel.export_path_requested.connect(self._choose_export_path)
        export_data_button = QtWidgets.QPushButton('Export curve data')
        export_data_button.setToolTip(rich_tooltip(_EXPORT_DATA_TIP))
        self._export_menu = QtWidgets.QMenu(export_data_button)
        self._export_menu.aboutToShow.connect(self._fill_export_menu)
        export_data_button.setMenu(self._export_menu)
        compare_button = QtWidgets.QPushButton('Compare data')
        compare_button.setToolTip(rich_tooltip(_COMPARE_TIP))
        compare_menu = QtWidgets.QMenu(compare_button)
        compare_menu.addAction('Add curve data file').triggered.connect(lambda checked=False: self._add_compared_file())
        self._compare_run_menu = compare_menu.addMenu('Add plot of this run')   # refilled by _fill_compare_menu
        self._compare_clear_action = compare_menu.addAction('Clear compared data')
        self._compare_clear_action.triggered.connect(lambda checked=False: self._clear_compared())
        compare_menu.aboutToShow.connect(self._fill_compare_menu)
        compare_button.setMenu(compare_menu)
        self._note_label = selectable(QtWidgets.QLabel())
        self._note_label.setWordWrap(True)
        self._note_label.hide()
        self._error_label = QtWidgets.QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)   # error texts may contain '<...>'
        self._error_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self._error_label.setStyleSheet(f'color:{ERROR_COLOR};')
        self._error_label.hide()

        toolbar = QtWidgets.QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        for widget in (*leading_widgets, self.plot_combo, self._unit_combo, self.series_combo):
            toolbar.addWidget(widget)
        toolbar.addStretch(1)
        for widget in (export_data_button, compare_button):
            toolbar.addWidget(widget)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(toolbar)
        layout.addWidget(self.panel, 1)
        layout.addWidget(self._note_label)
        layout.addWidget(self._error_label)
        for combo in (self.plot_combo, self._unit_combo, self.series_combo):
            combo.hide()

    # ---- public API ----
    def request_redraw(self, live=False):
        """Redraw from fresh plots() soon (coalesced); live: at most every _LIVE_INTERVAL_MS; a hidden view (e.g. on a
        tab not shown) redraws when it is shown."""
        self._dirty = True
        if not self.isVisible():
            return
        if live:
            if not self._live_timer.isActive():
                self._live_timer.start()
        elif not self._pending:
            self._pending = True
            QtCore.QTimer.singleShot(0, self._on_pending_redraw)

    def redraw(self):
        """plots() now: refill the combos (choices kept by name / unit), then draw the shown plot."""
        self._dirty = False
        self._plot_list = list(self._plots_hook())
        names = [name for name, _xy in self._plot_list]
        current = self.plot_combo.currentText()
        fill_combo(self.plot_combo, names, names.index(current) if current in names else 0)
        self.plot_combo.setVisible(len(names) > 1)
        self._on_plot_chosen()

    def shown_plot(self):
        """(name, xy or None) of the plot shown; (None, None) without plots."""
        index = self.plot_combo.currentIndex()
        return self._plot_list[index] if 0 <= index < len(self._plot_list) else (None, None)

    def set_error(self, text):
        """Owner error text (e.g. curves that failed to evaluate), kept until replaced; '' clears it."""
        self._owner_error = text
        self._show_errors()

    def add_error(self, text):
        """Error of the current draw (an overlay that failed); cleared by the next draw."""
        if text not in self._draw_errors:
            self._draw_errors.append(text)
        self._show_errors()

    def export_plot(self):
        """Write the shown plot (PlotPanel.export, matplotlib) to the panel's export file, or, without one, to a
        .pdf / .svg / .eps / .png path asked for."""
        self._redraw_if_stale()
        name, xy = self.shown_plot()
        if xy is None:
            self.message.emit(f'No plot to export yet: {self._empty_text}.', True)
            return
        path = self.panel.export_path()
        if path:
            suffixes = [suffix for _label, suffix in PLOT_FILE_TYPES]
            if os.path.splitext(path)[1].lower() not in suffixes:
                self.message.emit(f"Export file {path}: the name must end in {', '.join(suffixes)}.", True)
                return
        else:
            path = self._ask_plot_path(name)
            if not path:
                return
        write_export(self.panel.export, path, self.message.emit, 'Plot')

    def _ask_plot_path(self, name):
        """Save dialog of a plot export file ('' when cancelled)."""
        return ask_save_path(self, self._document, 'Export plot', f'{file_stem(self._export_stem_hook(name))}.pdf',
                             PLOT_FILE_TYPES)

    def _choose_export_path(self):
        """The Exports tab's Choose…: the picked file becomes the panel's export file."""
        name, _xy = self.shown_plot()
        path = self._ask_plot_path(name)
        if path:
            remember_directory(path)
            self.panel.set_export_path(path)

    def export_curve_data(self, all_plots):
        """export_curve_plots of the shown plot (NPZ or CSV) or of every plot (NPZ) with the metadata() title and
        parameters (an xy's own parameters win); refused for live previews and without metadata."""
        self._redraw_if_stale()
        if all_plots:
            plots = {name: xy for name, xy in self._plot_list if xy is not None}
            stem, file_types = self._export_stem_hook(None), TABLE_FILE_TYPES[:1]
        else:
            name, xy = self.shown_plot()
            plots = {} if xy is None else {name or 'plot': xy}
            stem, file_types = self._export_stem_hook(name), TABLE_FILE_TYPES
        if not plots:
            self.message.emit(f'No curve data to export yet: {self._empty_text}.', True)
            return
        if any(xy.get('preview') for xy in plots.values()):
            self.message.emit('A live preview (min / max per display bin) is not the result: export after the run.',
                              True)
            return
        metadata = self._metadata_hook() if self._metadata_hook is not None else ('', {})
        if metadata is None:
            self.message.emit('The session of these results is gone: build or compute again to export them with '
                              'their parameters.', True)
            return
        title, parameters = metadata
        exported = {name: dict(exportable_xy(xy), parameters=xy.get('parameters') or dict(parameters))
                    for name, xy in plots.items()}
        path = ask_save_path(self, self._document, 'Export curve data', f'{file_stem(stem, "curves")}.npz',
                             file_types)
        if not path:
            return
        write_export(lambda target: export_curve_plots(exported, target, model=title), path, self.message.emit,
                     'Curve data')

    # ---- redraw ----
    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty:
            self.request_redraw()

    def _on_pending_redraw(self):
        self._pending = False
        self._redraw_if_dirty()

    def _redraw_if_dirty(self):
        if self._dirty and self.isVisible():
            self.redraw()

    def _redraw_if_stale(self):
        if self._dirty:
            self.redraw()

    def _shown_groups(self):
        _name, xy = self.shown_plot()
        return xy_series_groups(xy) if xy is not None else []

    def _on_plot_chosen(self):
        """Unit groups of the shown plot (the unit kept when listed; hidden for one), then the series choice."""
        groups = self._shown_groups()
        index = self._unit_combo.currentIndex()
        current = self._group_units[index] if 0 <= index < len(self._group_units) else None
        self._group_units = [unit for unit, _names, _label in groups]
        fill_combo(self._unit_combo, [_unit_group_text(unit, names) for unit, names, _label in groups],
                   self._group_units.index(current) if current in self._group_units else 0)
        self._unit_combo.setVisible(len(groups) > 1)
        self._on_unit_chosen()

    def _on_unit_chosen(self):
        """Series choices of the shown unit group ('All series' first; a series still listed stays chosen), then
        draw."""
        groups = self._shown_groups()
        index = self._unit_combo.currentIndex()
        names = list(groups[index][1]) if 0 <= index < len(groups) else []
        chosen = self._chosen_series()
        self._series_names = names
        fill_combo(self.series_combo, ['All series'] + names, 1 + names.index(chosen) if chosen in names else 0)
        self.series_combo.setVisible(len(names) > 1)
        self._draw()

    def _chosen_series(self):
        """Series drawn alone; None = every series of the unit group."""
        index = self.series_combo.currentIndex() - 1
        return self._series_names[index] if 0 <= index < len(self._series_names) else None

    def _draw(self):
        """Series of the shown unit group (colour by position in the whole plot, all-NaN ones skipped), overlays,
        compared sets, axis labels, the plot's log_x / log_y when they changed, then the PlotPanel settings."""
        self.plot.clear()
        self._draw_errors = []
        name, xy = self.shown_plot()
        note = self._note_hook(name, xy) if self._note_hook is not None else ''
        self._note_label.setText(note)
        self._note_label.setVisible(bool(note))
        groups = self._shown_groups()
        group_index = self._unit_combo.currentIndex()
        if 0 <= group_index < len(groups):
            self._draw_group(name, xy, groups[group_index], group_index == 0)
        self.panel.apply(context=self._context_hook(name))
        self._show_errors()

    def _draw_group(self, name, xy, group, first_group):
        unit, names, label = group
        chosen = self._chosen_series()
        _x, x_label, x_units = xy_axis(xy)
        x_scale = METRES_PER_NM if is_wavelength_xy(xy) else 1.0
        drawn = []
        for position, series in enumerate(xy['series']):
            if series not in names or (chosen is not None and series != chosen):
                continue
            values = np.asarray(xy['series'][series], dtype=np.float64)
            if not np.any(np.isfinite(values)):   # e.g. a mode table's n_group without group_index: pyqtgraph warns
                continue
            color = SERIES_COLORS[position % len(SERIES_COLORS)]
            options = dict(self._curve_options_hook(name, xy, series))   # may set its own pen (e.g. grey references)
            if options.get('symbol'):
                options.setdefault('symbolBrush', color)
            options.setdefault('pen', series_pen(position))
            self.plot.plot(_series_axis(xy, series, x_scale), values, name=series, connect='finite', **options)
            drawn.append((series, color))
        if self._overlays_hook is not None:
            self._overlays_hook(self.plot, DrawnPlot(name, xy, unit, first_group, x_scale, tuple(drawn)))
        self._draw_compared(name, unit, x_label, x_units)
        y_label, y_units = axis_label(label, unit)
        self.plot.setLabel('bottom', x_label, units=x_units)
        self.plot.setLabel('left', y_label, units=y_units)
        if 'log_x' in xy or 'log_y' in xy:
            log_modes = (name, bool(xy.get('log_x')), bool(xy.get('log_y')))
            if log_modes != self._log_modes:   # the plot's choice; a user toggle stays until it changes
                self._log_modes = log_modes
                self.panel.axis_controls.set_log_modes(*log_modes[1:])

    def _show_errors(self):
        text = '; '.join(part for part in (self._owner_error, *self._draw_errors) if part)
        self._error_label.setText(text)
        self._error_label.setVisible(bool(text))

    # ---- export menu ----
    def _fill_export_menu(self):
        """Shown plot, All plots (with several plots), then the owner's extra actions."""
        menu = self._export_menu
        menu.clear()
        menu.addAction('Shown plot').triggered.connect(lambda checked=False: self.export_curve_data(all_plots=False))
        all_action = menu.addAction('All plots')
        all_action.triggered.connect(lambda checked=False: self.export_curve_data(all_plots=True))
        all_action.setEnabled(sum(xy is not None for _name, xy in self._plot_list) > 1)
        if self._extra_export_hook is not None:
            menu.addSeparator()
            self._extra_export_hook(menu)

    # ---- compared data ----
    def _fill_compare_menu(self):
        """Compare data menu before it shows: the other plots of this view, Clear only with compared data."""
        self._compare_run_menu.clear()
        shown, _xy = self.shown_plot()
        run_plots = [name for name, xy in self._plot_list if xy is not None and name != shown]
        for name in run_plots:
            self._compare_run_menu.addAction(name).triggered.connect(
                lambda checked=False, name=name: self._add_compared(('run', name, name)))
        self._compare_run_menu.setEnabled(bool(run_plots))
        self._compare_clear_action.setEnabled(bool(self._compared))

    def _add_compared_file(self):
        """Read chosen curve files (read_curve_data); a file of several plots asks for one plot or all (matched to
        the shown plot by name). Failures go to the message signal."""
        paths, _selected_filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, 'Compare curve data', start_directory(self._document), 'Curve data (*.npz *.csv);;All files (*)')
        failures = []
        for path in paths:
            try:
                plots = read_curve_data(path)
            except (OSError, ValueError) as exc:
                failures.append(error_message(exc))
                continue
            remember_directory(path)
            file_name = os.path.splitext(os.path.basename(path))[0]
            if len(plots) == 1:
                self._add_compared(('plot', file_name, next(iter(plots.values()))))
                continue
            choice, accepted = QtWidgets.QInputDialog.getItem(
                self, 'Compare data', f'{file_name} holds {len(plots)} plots. Compare:',
                [_ALL_COMPARED_PLOTS] + list(plots), 0, False)
            if not accepted:
                continue
            if choice != _ALL_COMPARED_PLOTS:
                self._add_compared(('plot', f'{file_name}: {choice}', plots[choice]))
                continue
            self._add_compared(('match', file_name, plots))
            if self._plot_list and not {name for name, _xy in self._plot_list} & set(plots):
                failures.append(f'{file_name}: none of its plots is named like a plot here')
        if failures:
            self.message.emit('Compare data: ' + '; '.join(failures), True)
        elif paths and self.shown_plot()[1] is None:
            self.message.emit(f'Compared data is drawn over a plot: {self._empty_text}.', False)

    def _add_compared(self, entry):
        """Append a compared set (kind, legend tag, payload) with its tag made unique, then redraw."""
        kind, tag, payload = entry
        tags = {existing_tag for _kind, existing_tag, _payload in self._compared}
        unique_tag, copy_number = tag, 1
        while unique_tag in tags:   # legend names stay unique (the Curves tab keys curves by name)
            copy_number += 1
            unique_tag = f'{tag} {copy_number}'
        self._compared.append((kind, unique_tag, payload))
        self._draw()

    def _clear_compared(self):
        self._compared.clear()
        self._draw()

    def _compared_sets(self, shown):
        """[(set index, legend tag, xy)] drawn over plot `shown`: a fixed plot, a file's plot of the shown plot's name
        ('match') or another plot of this view ('run', never over itself)."""
        run_plots = dict(self._plot_list)
        drawn = []
        for index, (kind, tag, payload) in enumerate(self._compared):
            if kind == 'plot':
                xy = payload
            elif kind == 'match':
                xy = payload.get(shown)
            else:
                xy = None if payload == shown else run_plots.get(payload)
            if xy is not None:
                drawn.append((index, tag, xy))
        return drawn

    def _draw_compared(self, shown, unit, x_label, x_units):
        """Compared curves of the shown unit (or of no recorded unit): colour by series position, one line style per
        compared set; a set with recorded units on another abscissa is named in the error line."""
        skipped = []
        for set_index, tag, xy in self._compared_sets(shown):
            _x, set_x_label, set_x_units = xy_axis(xy)
            recorded_units = xy.get('series_units') or {}
            if recorded_units and (set_x_label, set_x_units) != (x_label, x_units):
                skipped.append(f'{tag} ({set_x_label})')
                continue
            line_style = _COMPARED_LINE_STYLES[set_index % len(_COMPARED_LINE_STYLES)]
            x_scale = METRES_PER_NM if is_wavelength_xy(xy) else 1.0
            for position, (name, values) in enumerate(xy['series'].items()):
                if name in recorded_units and recorded_units[name] != unit:
                    continue
                pen = pg.mkPen(SERIES_COLORS[position % len(SERIES_COLORS)], width=1.5, style=line_style)
                self.plot.plot(_series_axis(xy, name, x_scale), np.asarray(values, dtype=np.float64), pen=pen,
                               name=f'{name} ({tag})', connect='finite')
        if skipped:
            self.add_error(f"Compared data on another horizontal axis than {x_label}, not drawn: {', '.join(skipped)}")
