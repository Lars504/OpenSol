"""Solve flow window (Solvers > Show solve flow): the model.solve_flow chart of the document drawn as boxes and arrows,
and for the clicked step its solver choice, memory items and run-time items with the ops they are priced from."""
import html
import math
import sys
import traceback

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.gpu_memory import format_bytes
from ..core.time_estimate import format_seconds
from ..model.solve_flow import MEMORY_GROUPS, TIME_SOURCES, op_text, solve_flow
from .document import results_inputs
from .forms import hint_label, rich_tooltip
from .helpers import ask_save_path, mesh_color

BOX_WIDTH = 236
BOX_HEIGHT = 70
WIDE_BOX_WIDTH = 520      # largest width of a box spanning several lanes (model, couplings, study, results)
LANE_WIDTH = 256
ROW_HEIGHT = 94
LOOP_SPACING = 30
_PADDING = 9.0            # box text / bar inset
_BAR_HEIGHT = 3.0
_REFRESH_DELAY_MS = 400
_PICTURE_SCALE = 2.0
_TAG_POINTS = 7.0         # box texts at 100 % zoom
_TITLE_POINTS = 10.0
_TEXT_POINTS = 8.5
_MEASURED_LINE = 16.0     # extra box height for the measured line once a run of this model exists
_WHEEL_ZOOM_BASE = 1.0015 # zoom factor per wheel angle unit (120 units = one notch: about 20 %)
_CHART_SHARE = 0.6        # opening width share of the chart pane
# box fills per step kind (light: the box text is always dark, whatever the application theme)
_KIND_COLORS = {'model': '#dbe7f6', 'mesh': '#d5eee8', 'materials': '#e4efd0', 'solver': '#fbe1c4',
                'physics': '#d3ebcf', 'couplings': '#e5dcf3', 'study': '#e1d8f4', 'output': '#e6e6e6',
                'unplaced': '#f6cccc'}
_BORDER_COLOR = '#5a6470'
_SELECTED_COLOR = '#1d5fbf'
_TEXT_COLOR = '#1b1f24'
_SUBTEXT_COLOR = '#4a525c'
_MEMORY_BAR_COLOR = '#3f7fbf'
_TIME_BAR_COLOR = '#d9822b'
_BAR_TRACK_COLOR = QtGui.QColor(0, 0, 0, 22)
_TAG_COLOR = '#6b7480'
_MEASURED_COLOR = '#22663a'
# row kind shown in the corner of each box
_KIND_TAGS = {'model': 'MODEL', 'mesh': 'MESH', 'materials': 'MATERIALS', 'solver': 'SOLVER JOB',
              'physics': 'PHYSICS', 'couplings': 'COUPLINGS', 'study': 'STUDY', 'output': 'RESULTS',
              'unplaced': 'UNPLACED'}
_LINK_COLOR = '#8a4fbf'
_LOOP_COLOR = '#7a5c2e'
_INTRO = ('Every step of a run of this model, top to bottom: the build (mesh, materials, solver jobs, physics, '
          'couplings), the study and the results. Each box shows the memory and run time the estimates book at it '
          '(blue bar: share of the memory items, orange bar: share of the run time); click a box for the items and '
          'what they are computed from. Brackets on the right are repetitions (sweeps, sectors, refinement levels). '
          'After a Build / Compute of this model the measured values appear next to the estimates (green). The chart '
          'follows the model as you edit it; nothing is run. Mouse wheel zooms, drag pans.')


def show_solve_flow(document, parent):
    """Show (raise) the solve flow window of the document, made on first use as a child of the main window: the first
    ancestor of parent holding the WorkerBridge (`bridge`: measured values of the last run), else parent's window."""
    window = parent
    while window is not None and not hasattr(window, 'bridge'):
        window = window.parentWidget()
    bridge = getattr(window, 'bridge', None)
    window = window if window is not None else parent.window()
    for dialog in window.findChildren(SolveFlowDialog):
        if dialog.document is document:
            break
    else:
        dialog = SolveFlowDialog(document, window, bridge)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog


class SolveFlowDialog(QtWidgets.QDialog):
    """Non-modal window: flow chart (left) and the selected step's details (right); redrawn after document edits
    (debounced, only while visible), estimate changes (issues_invalidated: budgets, built element counts) and the end
    of every build / run of the bridge (measured values, shown only while the bridge's results are of this model)."""

    def __init__(self, document, parent=None, bridge=None):
        super().__init__(parent)
        self.document = document
        self.bridge = bridge
        self._results_key = (None, None)   # (results_spec, its results_inputs) cached by identity
        self.flow = None
        self._boxes = {}
        self._measured_note = ''
        self._selected_key = None
        self._dirty = True
        self._fitted = False
        self.setWindowTitle('Solve flow')
        self.setModal(False)
        self.resize(1320, 840)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(hint_label(_INTRO))
        buttons = QtWidgets.QHBoxLayout()
        fit_button = QtWidgets.QPushButton('Fit to window')
        fit_button.setToolTip(rich_tooltip('Show the whole chart (the mouse wheel zooms, drag to pan).'))
        fit_button.clicked.connect(lambda _checked=False: self._fit())
        actual_button = QtWidgets.QPushButton('100 %')
        actual_button.clicked.connect(lambda _checked=False: self._zoom_to(1.0))
        save_button = QtWidgets.QPushButton('Save picture…')
        save_button.clicked.connect(lambda _checked=False: self._save_picture())
        overview_button = QtWidgets.QPushButton('Totals')
        overview_button.setToolTip(rich_tooltip('Show the estimate totals and notes instead of a step.'))
        overview_button.clicked.connect(lambda _checked=False: self._select(None))
        for button in (fit_button, actual_button, save_button, overview_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.selectionChanged.connect(self._on_selection_changed)
        self.view = _FlowView(self.scene)
        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenLinks(False)
        self.details.setMinimumWidth(320)
        self.details.document().setDefaultStyleSheet(_DETAILS_CSS)
        self.view.setMinimumWidth(360)
        splitter.addWidget(self.view)
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, 1)
        self._splitter = splitter
        self._split_set = False
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_REFRESH_DELAY_MS)
        self._timer.timeout.connect(self.refresh)
        document.changed.connect(self._on_document_changed)
        document.issues_invalidated.connect(self._schedule)
        if bridge is not None:
            for signal in (bridge.built, bridge.build_failed, bridge.study_finished):
                signal.connect(lambda _event: self._schedule())
            bridge.study_failed.connect(lambda _message, _traceback: self._schedule())

    # refresh

    def _on_document_changed(self, _section, _origin):
        self._schedule()

    def _schedule(self):
        self._dirty = True
        if self.isVisible():
            self._timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._split_set:   # the text browser's size hint would otherwise take most of the width
            self._split_set = True
            total = max(self._splitter.width(), 2)
            self._splitter.setSizes([int(total * _CHART_SHARE), total - int(total * _CHART_SHARE)])
        if self._dirty:
            self.refresh()

    def refresh(self):
        """Rebuild the flow from the document's cached estimates and redraw it, keeping the selected step."""
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        document = self.document
        keep = self._selected_key
        self.scene.blockSignals(True)
        try:
            self.scene.clear()
            self._boxes = {}
            self.flow = None
            estimates, _errors = document.used_mesh_estimates()
            run_time, message = document.run_time_estimate()
            study_time, _ = document.time_estimate()
            build_record, study_summary, self._measured_note = self._measured()
            self.flow = solve_flow(document.spec, document.registry, document.parameters(), estimates,
                                   document.memory_estimate(), document.plain_roles(), run_time,
                                   document.time_overheads(), study_time, message, build_record, study_summary)
            self._boxes = _draw_flow(self.scene, self.flow, document.spec)
            self.scene.setSceneRect(self.scene.itemsBoundingRect().adjusted(-24, -24, 24, 24))
        except Exception:   # a solve_flow / drawing bug: shown in the window and logged, never a blank window
            self.scene.clear()
            self._boxes, self.flow = {}, None
            self.details.setHtml('<h3 style="color:#b00020">The solve flow could not be drawn</h3>'
                                 '<p>This is a bug in the solve flow; the traceback is also in the Messages tab.</p>'
                                 f'<pre>{html.escape(traceback.format_exc())}</pre>')
            sys.excepthook(*sys.exc_info())
            return
        finally:
            self.scene.blockSignals(False)
        self._select(keep if keep in self._boxes else None)
        if not self._fitted:
            self._fitted = True
            QtCore.QTimer.singleShot(0, self._fit)

    def _measured(self):
        """(build record, study summary, note) of the bridge's last build / run when its results are of the current
        model (results_inputs equal), else (None, None, why there are no measured values)."""
        bridge = self.bridge
        if bridge is None:
            return None, None, 'Measured values: not available in this window.'
        spec = bridge.results_spec
        if spec is None or (bridge.last_build_summary is None and bridge.last_study_summary is None):
            return None, None, 'Measured values: Build or Compute this model to compare them with the estimates.'
        if self._results_key[0] is not spec:
            self._results_key = (spec, results_inputs(spec.to_dict()))
        if self._results_key[1] != results_inputs(self.document.spec_dict()):
            return None, None, 'Measured values hidden: the model changed after the last build / run.'
        note = ('Measured values: from the last build of this model (Compute to measure the study too).'
                if bridge.last_study_summary is None else 'Measured values: from the last run of this model.')
        return bridge.last_build_summary, bridge.last_study_summary, note

    # selection and details

    def _on_selection_changed(self):
        selected = [item for item in self.scene.selectedItems() if isinstance(item, _StepBox)]
        self._selected_key = selected[0].key if selected else None
        self._show_details()

    def _select(self, key):
        self.scene.blockSignals(True)
        try:
            self.scene.clearSelection()
            if key is not None:
                self._boxes[key].setSelected(True)
        finally:
            self.scene.blockSignals(False)
        self._selected_key = key
        self._show_details()

    def _show_details(self):
        if self.flow is None:
            self.details.setHtml('')
            return
        if self._selected_key is None:
            self.details.setHtml(_overview_html(self.flow, self._measured_note))
        else:
            self.details.setHtml(_step_html(self.flow, self.flow.step(self._selected_key)))

    # view actions

    def _fit(self):
        """Scale to show the whole chart (also the opening view)."""
        rect = self.scene.sceneRect()
        viewport = self.view.viewport().rect()
        if not rect.isValid() or viewport.width() < 10 or viewport.height() < 10:
            return
        self._zoom_to(min(viewport.width() / rect.width(), viewport.height() / rect.height()))

    def _zoom_to(self, scale):
        """Set the zoom to scale, centred across and showing the top of the chart."""
        rect = self.scene.sceneRect()
        self.view.resetTransform()
        self.view.scale(scale, scale)
        visible_height = self.view.viewport().height() / scale
        top = rect.top() + 0.5 * min(visible_height, rect.height())
        self.view.centerOn(QtCore.QPointF(rect.center().x(), top))

    def _save_picture(self):
        path = ask_save_path(self, self.document, 'Save solve flow picture', 'solve_flow.png', (('PNG image', '.png'),))
        if not path:
            return
        selected = list(self.scene.selectedItems())
        self.scene.clearSelection()   # no selection outline in the file
        rect = self.scene.itemsBoundingRect().adjusted(-16, -16, 16, 16)
        image = QtGui.QImage(int(rect.width() * _PICTURE_SCALE), int(rect.height() * _PICTURE_SCALE),
                             QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtGui.QColor('white'))
        painter = QtGui.QPainter(image)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
            self.scene.render(painter, QtCore.QRectF(image.rect()), rect)
        finally:
            painter.end()
        for item in selected:
            item.setSelected(True)
        if not image.save(path):
            QtWidgets.QMessageBox.warning(self, 'Save picture', f'Could not write {path}')


class _FlowView(QtWidgets.QGraphicsView):
    """Graphics view: drag to pan, the mouse wheel zooms about the cursor."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing
                            | QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignHCenter | QtCore.Qt.AlignmentFlag.AlignTop)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        delta = event.angleDelta().y() or event.angleDelta().x()
        if delta:
            factor = _WHEEL_ZOOM_BASE ** delta
            self.scale(factor, factor)
        event.accept()


class _StepBox(QtWidgets.QGraphicsPathItem):
    """Rounded, selectable box of one FlowStep that paints its own texts (kind tag, title, subtitle, estimate line,
    each one elided line) and its memory / time bars; no Qt selection outline: a thicker blue border instead."""

    def __init__(self, step, rect, fill, border, shares, show_measured=False):
        super().__init__()
        self.key = step.key
        self._rect = rect
        self._shares = shares   # (memory share, time share) of the largest step, 0 = no bar
        self._border = QtGui.QColor(border)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 9.0, 9.0)
        self.setPath(path)
        self.setBrush(QtGui.QColor(fill))
        self._set_border(False)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        estimate = []
        if step.memory:
            estimate.append(format_bytes(step.memory_bytes))
        if step.time:
            estimate.append(format_seconds(step.time_seconds))
        if step.not_estimated:
            estimate.append(f'{len(step.not_estimated)} not estimated')
        self._tag = _KIND_TAGS.get(step.kind, step.kind.upper())
        self._title, self._subtitle = step.title, step.subtitle
        self._estimate = ('estimated ' if show_measured and estimate else '') + '  ·  '.join(estimate)
        measured = []
        if step.measured_bytes is not None:
            measured.append(format_bytes(step.measured_bytes))
        if step.measured_seconds is not None:
            measured.append(format_seconds(step.measured_seconds))
        self._measured = ('measured ' + '  ·  '.join(measured)) if show_measured and measured else ''

    def _set_border(self, selected):
        pen = QtGui.QPen(QtGui.QColor(_SELECTED_COLOR) if selected else self._border, 3.0 if selected else 1.3)
        self.setPen(pen)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._set_border(bool(value))
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        option = QtWidgets.QStyleOptionGraphicsItem(option)
        option.state = option.state & ~QtWidgets.QStyle.StateFlag.State_Selected
        super().paint(painter, option, widget)
        rect = self._rect
        left, width = rect.left() + _PADDING, rect.width() - 2.0 * _PADDING
        painter.save()
        try:
            painter.setClipRect(rect.adjusted(1.0, 1.0, -1.0, -1.0))
            tag_font, title_font, text_font = _box_fonts()
            tag_metrics = QtGui.QFontMetricsF(tag_font)
            title_metrics = QtGui.QFontMetricsF(title_font)
            baseline = rect.top() + _PADDING - 2.0 + title_metrics.ascent()
            tag_width = tag_metrics.horizontalAdvance(self._tag)
            painter.setFont(tag_font)
            painter.setPen(QtGui.QColor(_TAG_COLOR))
            painter.drawText(QtCore.QPointF(rect.right() - _PADDING - tag_width, baseline), self._tag)
            painter.setFont(title_font)
            painter.setPen(QtGui.QColor(_TEXT_COLOR))
            painter.drawText(QtCore.QPointF(left, baseline),
                             _elided(self._title, title_font, width - tag_width - 8.0))
            text_metrics = QtGui.QFontMetricsF(text_font)
            painter.setFont(text_font)
            painter.setPen(QtGui.QColor(_SUBTEXT_COLOR))
            for text, color in ((self._subtitle, _SUBTEXT_COLOR), (self._estimate, _SUBTEXT_COLOR),
                                (self._measured, _MEASURED_COLOR)):
                baseline += text_metrics.lineSpacing() + 1.0
                if text:
                    painter.setPen(QtGui.QColor(color))
                    painter.drawText(QtCore.QPointF(left, baseline), _elided(text, text_font, width))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            for offset, (share, color) in enumerate(zip(self._shares, (_MEMORY_BAR_COLOR, _TIME_BAR_COLOR))):
                if share <= 0.0:
                    continue
                top = rect.bottom() - _PADDING - 1.0 - (1 - offset) * (_BAR_HEIGHT + 2.0)
                painter.setBrush(_BAR_TRACK_COLOR)
                painter.drawRect(QtCore.QRectF(left, top, width, _BAR_HEIGHT))
                painter.setBrush(QtGui.QColor(color))
                painter.drawRect(QtCore.QRectF(left, top, max(width * share, 3.0), _BAR_HEIGHT))
        finally:
            painter.restore()


_FONTS = []


def _box_fonts():
    """(tag, title, text) QFonts of the box texts, made once (after the QApplication exists)."""
    if not _FONTS:
        tag, title, text = QtGui.QFont(), QtGui.QFont(), QtGui.QFont()
        tag.setPointSizeF(_TAG_POINTS)
        tag.setWeight(QtGui.QFont.Weight.DemiBold)
        tag.setLetterSpacing(QtGui.QFont.SpacingType.PercentageSpacing, 108.0)
        title.setPointSizeF(_TITLE_POINTS)
        title.setWeight(QtGui.QFont.Weight.DemiBold)
        text.setPointSizeF(_TEXT_POINTS)
        _FONTS.extend((tag, title, text))
    return tuple(_FONTS)


def _share(value, largest) -> float:
    """value / largest in [0, 1]; 0 for a zero or non-finite pair (NaN geometry can crash Qt's painter)."""
    if not largest or not math.isfinite(largest) or not math.isfinite(value):
        return 0.0
    return min(max(float(value) / float(largest), 0.0), 1.0)


def _elided(text, font, width) -> str:
    return QtGui.QFontMetricsF(font).elidedText(text, QtCore.Qt.TextElideMode.ElideRight, max(width, 0.0))


# --- drawing --------------------------------------------------------------------------------------------------------

def _box_rect(step, extra=0.0) -> QtCore.QRectF:
    """Scene rectangle of a step: centred over its lanes, one lane's box width per lane (capped for wide steps);
    extra: added box (and row) height (the measured line)."""
    span = step.lane_end - step.lane_start
    width = BOX_WIDTH if span <= 1.0 else min(span * LANE_WIDTH - (LANE_WIDTH - BOX_WIDTH), WIDE_BOX_WIDTH)
    centre = 0.5 * (step.lane_start + step.lane_end) * LANE_WIDTH
    return QtCore.QRectF(centre - 0.5 * width, step.row * (ROW_HEIGHT + extra), width, BOX_HEIGHT + extra)


def _draw_flow(scene, flow, spec) -> dict:
    """Draw arrows, coupling links, loops and step boxes (the row kind is each box's tag); returns {step key:
    _StepBox}."""
    extra = _MEASURED_LINE if flow.measured else 0.0
    rects = {step.key: _box_rect(step, extra) for step in flow.steps}
    largest_bytes = max((step.memory_bytes for step in flow.steps), default=0)
    largest_seconds = max((step.time_seconds for step in flow.steps), default=0.0)
    right = max((rect.right() for rect in rects.values()), default=flow.lanes * LANE_WIDTH)
    arrow_pen = QtGui.QPen(QtGui.QColor(_BORDER_COLOR), 1.2)
    for step in flow.steps:
        for before in step.after:
            if before in rects:
                _draw_arrow(scene, rects[before], rects[step.key], arrow_pen)
    for index, link in enumerate(flow.links):
        if link.source in rects and link.target in rects:
            _draw_link(scene, rects[link.source], rects[link.target], link.label, index)
    for index, loop in enumerate(flow.loops):
        _draw_loop(scene, loop, right + 18 + index * LOOP_SPACING, extra)
    boxes = {}
    for step in flow.steps:
        border = (mesh_color(spec, step.key.split(':', 1)[1]) if step.kind in ('mesh', 'materials')
                  else _BORDER_COLOR)
        shares = tuple(_share(value, largest) for value, largest in ((step.memory_bytes, largest_bytes),
                                                                     (step.time_seconds, largest_seconds)))
        box = _StepBox(step, rects[step.key], _KIND_COLORS.get(step.kind, '#eeeeee'), border, shares,
                       flow.measured)
        box.setZValue(2)
        box.setToolTip(rich_tooltip(step.title, step.subtitle, step.explanation))
        scene.addItem(box)
        boxes[step.key] = box
    return boxes


def _draw_arrow(scene, source, target, pen):
    """Cubic arrow from the bottom of source to the top of target (ends kept over the other box where possible)."""
    start_x = min(max(target.center().x(), source.left() + 20.0), source.right() - 20.0)
    end_x = min(max(source.center().x(), target.left() + 20.0), target.right() - 20.0)
    start, end = QtCore.QPointF(start_x, source.bottom()), QtCore.QPointF(end_x, target.top())
    middle = 0.5 * (start.y() + end.y())
    path = QtGui.QPainterPath(start)
    path.cubicTo(QtCore.QPointF(start.x(), middle), QtCore.QPointF(end.x(), middle), end)
    scene.addPath(path, pen).setZValue(1)
    _arrow_head(scene, end, QtCore.QPointF(0.0, 1.0), pen.color())


def _draw_link(scene, source, target, label, index):
    """Dashed coupling arc between two physics boxes of one row, below them, with its name."""
    pen = QtGui.QPen(QtGui.QColor(_LINK_COLOR), 1.3, QtCore.Qt.PenStyle.DashLine)
    if source == target:   # a self-coupling: a small loop under the box
        start = QtCore.QPointF(source.center().x() - 24.0, source.bottom())
        end = QtCore.QPointF(source.center().x() + 24.0, source.bottom())
    else:
        start = QtCore.QPointF(source.center().x() + 12.0, source.bottom())
        end = QtCore.QPointF(target.center().x() - 12.0, target.bottom())
    depth = 10.0 + 4.0 * (index % 3)   # inside the gap to the next row
    path = QtGui.QPainterPath(start)
    path.cubicTo(start + QtCore.QPointF(0.0, depth), end + QtCore.QPointF(0.0, depth), end)
    scene.addPath(path, pen).setZValue(3)
    _arrow_head(scene, end, QtCore.QPointF(0.0, -1.0), pen.color())
    font = QtGui.QFont()
    font.setPointSizeF(_TEXT_POINTS)
    text = scene.addSimpleText(label, font)
    text.setBrush(QtGui.QColor(_LINK_COLOR))
    width = text.boundingRect().width()
    text.setPos(0.5 * (start.x() + end.x()) - 0.5 * width, start.y() + 0.75 * depth)
    text.setZValue(3)


def _draw_loop(scene, loop, x, extra=0.0):
    """Bracket at x from the top of first_row to the bottom of last_row, an arrow back up and its title."""
    pen = QtGui.QPen(QtGui.QColor(_LOOP_COLOR), 1.6)
    row_height, box_height = ROW_HEIGHT + extra, BOX_HEIGHT + extra
    top, bottom = loop.first_row * row_height + 4.0, loop.last_row * row_height + box_height - 4.0
    path = QtGui.QPainterPath(QtCore.QPointF(x - 10.0, bottom))
    path.lineTo(x, bottom)
    path.lineTo(x, top)
    path.lineTo(x - 10.0, top)
    item = scene.addPath(path, pen)
    item.setToolTip(rich_tooltip(f'{loop.title}: {loop.text}'))
    _arrow_head(scene, QtCore.QPointF(x - 10.0, top), QtCore.QPointF(-1.0, 0.0), pen.color())
    font = QtGui.QFont()
    font.setPointSizeF(_TEXT_POINTS)
    text = scene.addSimpleText(loop.title, font)
    text.setBrush(QtGui.QColor(_LOOP_COLOR))
    text.setRotation(90.0)
    text.setPos(x + 4.0 + text.boundingRect().height(), top + 6.0)
    text.setToolTip(rich_tooltip(f'{loop.title}: {loop.text}'))


def _arrow_head(scene, tip, direction, color):
    """Filled triangle at tip pointing along the unit vector direction."""
    size = 7.0
    normal = QtCore.QPointF(-direction.y(), direction.x())
    base = tip - direction * size
    polygon = QtGui.QPolygonF([tip, base + normal * (0.5 * size), base - normal * (0.5 * size)])
    head = scene.addPolygon(polygon, QtGui.QPen(QtCore.Qt.PenStyle.NoPen), QtGui.QBrush(color))
    head.setZValue(3)


# --- details --------------------------------------------------------------------------------------------------------

_DETAILS_CSS = '''
h3 { margin-top: 2px; margin-bottom: 2px; }
h4 { margin-top: 14px; margin-bottom: 4px; }
p.sub { color: gray; margin-top: 0px; }
p.note { color: gray; }
table.facts td.name { color: gray; padding-right: 14px; }
table.items { margin-top: 2px; }
table.items th { background-color: #e9edf2; color: #1b1f24; padding: 4px 8px; text-align: left; }
table.items td { padding: 4px 8px; border-bottom: 1px solid #d8dde3; vertical-align: top; }
td.number { text-align: right; white-space: nowrap; }
span.small { color: gray; font-size: small; }
code { font-family: monospace; font-size: small; }
'''


def _units(count, unit) -> str:
    """'1 upload', '8 passes', '2.5e+04 iterations'."""
    if count == 1:
        return f'1 {unit}'
    plural = unit + ('es' if unit.endswith(('s', 'sh', 'ch', 'x')) else 's')
    return f'{count:.4g} {plural}'


def _overview_html(flow, measured_note='') -> str:
    parts = ['<h3>Solve flow totals</h3>', _totals_table(flow) if flow.totals else '<p>No estimates.</p>']
    if measured_note:
        parts.append(f'<p style="color:{_MEASURED_COLOR}">{html.escape(measured_note)}</p>')
    if flow.notes:
        parts.append('<h4>Notes</h4><ul>' + ''.join(f'<li>{html.escape(note)}</li>' for note in flow.notes) + '</ul>')
    parts.append('<p class="note">Memory items are the arrays each step\'s estimator counts from the code '
                 '(meshing.estimate_memory); time items are one study run priced on this computer\'s profile '
                 '(model.time_estimate, python -m open_sol profile). The whole-study time also counts every sweep '
                 'value, sector and estimation run. Measured: the build\'s memory ledger (pool change and wall time '
                 'per build stage) and the run summary\'s time check. Click a box for its items.</p>')
    return ''.join(parts)


def _totals_table(flow) -> str:
    rows = []
    for name, estimate, measured in flow.totals:
        cells = (f'<td>{html.escape(str(name))}</td><td class="number">{html.escape(str(estimate))}</td>'
                 + (f'<td class="number" style="color:{_MEASURED_COLOR}">{html.escape(measured or "—")}</td>'
                    if flow.measured else ''))
        rows.append(f'<tr>{cells}</tr>')
    header = '<tr><th></th><th>Estimated</th>' + ('<th>Measured</th>' if flow.measured else '') + '</tr>'
    return '<table class="items" cellspacing="0">' + header + ''.join(rows) + '</table>'


def _step_html(flow, step) -> str:
    parts = [f'<h3>{html.escape(step.title)}</h3>']
    if step.subtitle:
        parts.append(f'<p class="sub">{html.escape(step.subtitle)}</p>')
    if step.explanation:
        parts.append(f'<p>{html.escape(step.explanation)}</p>')
    if step.facts:
        parts.append(_table(step.facts))
    parts.append(_memory_html(flow, step))
    parts.append(_stages_html(flow, step))
    parts.append(_time_html(flow, step))
    if step.not_estimated:
        parts.append('<h4>Not estimated</h4><ul>'
                     + ''.join(f'<li>{html.escape(label)}</li>' for label in step.not_estimated) + '</ul>')
    return ''.join(parts)


def _measured_cell(text, number=True) -> str:
    align = ' class="number"' if number else ''
    return f'<td{align} style="color:{_MEASURED_COLOR}">{text}</td>'


def _memory_html(flow, step) -> str:
    if not step.memory:
        return ''
    rows = []
    for entry in sorted(step.memory, key=lambda entry: -entry.bytes):
        group = MEMORY_GROUPS.get(entry.group, entry.group)
        if entry.phase:
            group += f' (phase: {entry.phase})'
        rows.append(f'<tr><td>{html.escape(entry.label)}</td><td class="number">{format_bytes(entry.bytes)}</td>'
                    + (_measured_cell('—') if flow.measured else '')
                    + f'<td><span class="small">{html.escape(group)}</span></td></tr>')
    if flow.measured:   # the ledger measures each build stage's pool change: comparable with the kept items only
        measured = step.measured_bytes
        rows.append(f'<tr><td><b>Kept from the build</b><br><span class="small">the items the build ledger measures '
                    f'(pool change of this step\'s build stages)</span></td>'
                    f'<td class="number"><b>{format_bytes(step.kept_bytes)}</b></td>'
                    + _measured_cell(f'<b>{format_bytes(measured)}</b>' if measured is not None else '—')
                    + '<td></td></tr>')
    note = ('<p class="note">On the cpu backend the device arrays live in processor memory.</p>'
            if flow.backend == 'cpu' else '')
    header = ('<tr><th>Item</th><th>Estimated</th>' + ('<th>Measured</th>' if flow.measured else '')
              + '<th>Kind</th></tr>')
    return (f'<h4>Memory items ({format_bytes(step.memory_bytes)} listed)</h4>'
            '<table class="items" width="100%" cellspacing="0">' + header + ''.join(rows) + '</table>' + note
            + ('<p class="note">Items are not measured one by one: the measured column compares the step as a whole.'
               '</p>' if flow.measured else ''))


def _stages_html(flow, step) -> str:
    """Build stages of the memory ledger booked at the step (after a build of this model)."""
    if not flow.measured or not (step.measured_stages or step.measured_study_seconds is not None):
        return ''
    rows = ''.join(f'<tr><td>{html.escape(stage.owner)}</td>'
                   + _measured_cell(format_bytes(stage.bytes) if stage.bytes is not None else '—')
                   + _measured_cell(format_seconds(stage.seconds) if stage.seconds is not None else '—') + '</tr>'
                   for stage in step.measured_stages)
    if step.measured_study_seconds is not None:
        rows += ('<tr><td>study time of its timers<br><span class="small">time check seconds groups of this step'
                 + (' (the study time no solver timer covers)' if step.kind == 'study' else '') + '</span></td>'
                 + _measured_cell('—') + _measured_cell(format_seconds(step.measured_study_seconds)) + '</tr>')
    return ('<h4>Measured</h4><table class="items" width="100%" cellspacing="0">'
            '<tr><th>Build stage / timer</th><th>Pool change</th><th>Wall time</th></tr>' + rows + '</table>')


def _time_html(flow, step) -> str:
    if not step.time:
        return ''
    rows = []
    for entry in sorted(step.time, key=lambda entry: -entry.seconds):
        item = entry.item
        low, guess, high = item.counts()
        count = _units(guess, item.unit)
        spread = f'{item.count_kind}; {low:.3g} - {high:.3g}' if item.count_kind != 'exact' else item.count_kind
        if item.limit is not None:
            spread += f'; at most {item.limit:.4g}'
        ops = '<br>'.join(f'<code>{html.escape(op_text(op))}</code>' for op in item.ops) or '—'
        counter = '.'.join(str(part) for part in item.counter) if item.counter else ''
        cells = (f'<tr><td>{html.escape(entry.label)}<br><span class="small">'
                 f'{html.escape(TIME_SOURCES.get(entry.source, entry.source))}'
                 + (f'<br>checked against summary {html.escape(counter)}' if counter else '') + '</span></td>'
                 f'<td class="number"><b>{format_seconds(entry.seconds)}</b></td>'
                 f'<td>{html.escape(count)} x {format_seconds(item.seconds_per_unit or 0.0)}<br>'
                 f'<span class="small">{html.escape(spread)}</span></td>')
        if flow.measured:
            cells += _measured_cell(_measured_time_text(entry), number=False)
        rows.append(cells + f'<td>{ops}</td></tr>')
    header = ('<tr><th>Item</th><th>Estimated</th><th>Count x per unit</th>'
              + ('<th>Measured</th>' if flow.measured else '') + '<th>Operations per unit</th></tr>')
    note = ('<p class="note">time = count x time per unit; each unit costs the listed operations, priced on this '
            'computer\'s profile.'
            + (' Measured: the counted units of the run and its time, which is timed per solver timer (or the rest of '
               'the study), so each item gets its group\'s measured / estimated ratio.' if flow.measured else '')
            + '</p>')
    return (f'<h4>Run-time items ({format_seconds(step.time_seconds)})</h4>' + note
            + '<table class="items" width="100%" cellspacing="0">' + header + ''.join(rows) + '</table>')


def _measured_time_text(entry) -> str:
    """Measured column of a time item: counted units (x their prediction) and the group-scaled seconds."""
    record = entry.measured
    if record is None:
        return '—'
    parts = []
    count = entry.measured_count
    if count is not None:
        ratio = record.get('count_ratio')
        parts.append(html.escape(_units(count, entry.item.unit))
                     + (f' <span class="small">(x{ratio:.2g})</span>' if ratio is not None else ''))
    else:
        parts.append('<span class="small">count not recorded</span>')
    seconds = entry.measured_seconds
    if seconds is not None:
        ratio = record.get('unit_ratio')
        parts.append(f'<b>{format_seconds(seconds)}</b>'
                     + (f' <span class="small">(group x{ratio:.2g})</span>' if ratio is not None else ''))
    return '<br>'.join(parts)


def _table(rows) -> str:
    return ('<table class="facts" cellspacing="0" cellpadding="2">'
            + ''.join(f'<tr><td class="name">{html.escape(str(name))}</td>'
                      f'<td>{html.escape(str(value))}</td></tr>' for name, value in rows) + '</table>')
