"""Solve flow window (Solvers > Show solve flow): the model.solve_flow chart of the document drawn as boxes and arrows,
and for the clicked step its solver choice, memory items and run-time items with the ops they are priced from."""
import html

from PySide6 import QtCore, QtGui, QtWidgets

from ..core.gpu_memory import format_bytes
from ..core.time_estimate import format_seconds
from ..model.solve_flow import MEMORY_GROUPS, TIME_SOURCES, op_text, solve_flow
from .forms import hint_label, rich_tooltip
from .helpers import ask_save_path, mesh_color

BOX_WIDTH = 214
BOX_HEIGHT = 76
WIDE_BOX_WIDTH = 520      # largest width of a box spanning several lanes (model, couplings, study, results)
LANE_WIDTH = 240
ROW_HEIGHT = 116
ROW_LABEL_WIDTH = 104
LOOP_SPACING = 30
_BAR_HEIGHT = 3
_REFRESH_DELAY_MS = 400
_PICTURE_SCALE = 2.0
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
_LINK_COLOR = '#8a4fbf'
_LOOP_COLOR = '#7a5c2e'
_INTRO = ('Every step of a run of this model, top to bottom: the build (mesh, materials, solver jobs, physics, '
          'couplings), the study and the results. Each box shows the memory and run time the estimates book at it '
          '(blue bar: share of the memory items, orange bar: share of the run time); click a box for the items and '
          'what they are computed from. Brackets on the right are repetitions (sweeps, sectors, refinement levels). '
          'The chart follows the model as you edit it; nothing is run.')


def show_solve_flow(document, parent):
    """Show (raise) the solve flow window of the document, made on first use as a child of parent's window."""
    window = parent.window()
    for dialog in window.findChildren(SolveFlowDialog):
        if dialog.document is document:
            break
    else:
        dialog = SolveFlowDialog(document, window)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog


class SolveFlowDialog(QtWidgets.QDialog):
    """Non-modal window: flow chart (left) and the selected step's details (right); redrawn after document edits
    (debounced, only while visible) and estimate changes (issues_invalidated: budgets, built element counts)."""

    def __init__(self, document, parent=None):
        super().__init__(parent)
        self.document = document
        self.flow = None
        self._boxes = {}
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
        fit_button.clicked.connect(lambda _checked=False: self._fit())
        save_button = QtWidgets.QPushButton('Save picture…')
        save_button.clicked.connect(lambda _checked=False: self._save_picture())
        overview_button = QtWidgets.QPushButton('Totals')
        overview_button.setToolTip(rich_tooltip('Show the estimate totals and notes instead of a step.'))
        overview_button.clicked.connect(lambda _checked=False: self._select(None))
        for button in (fit_button, save_button, overview_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.selectionChanged.connect(self._on_selection_changed)
        self.view = _FlowView(self.scene)
        self.details = QtWidgets.QTextBrowser()
        self.details.setOpenLinks(False)
        splitter.addWidget(self.view)
        splitter.addWidget(self.details)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(_REFRESH_DELAY_MS)
        self._timer.timeout.connect(self.refresh)
        document.changed.connect(self._on_document_changed)
        document.issues_invalidated.connect(self._schedule)

    # refresh

    def _on_document_changed(self, _section, _origin):
        self._schedule()

    def _schedule(self):
        self._dirty = True
        if self.isVisible():
            self._timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty:
            self.refresh()

    def refresh(self):
        """Rebuild the flow from the document's cached estimates and redraw it, keeping the selected step."""
        if not self.isVisible():
            self._dirty = True
            return
        self._dirty = False
        document = self.document
        estimates, _errors = document.used_mesh_estimates()
        run_time, message = document.run_time_estimate()
        study_time, _ = document.time_estimate()
        self.flow = solve_flow(document.spec, document.registry, document.parameters(), estimates,
                               document.memory_estimate(), document.plain_roles(), run_time,
                               document.time_overheads(), study_time, message)
        keep = self._selected_key
        self.scene.blockSignals(True)
        try:
            self.scene.clear()
            self._boxes = _draw_flow(self.scene, self.flow, document.spec)
        finally:
            self.scene.blockSignals(False)
        self._select(keep if keep in self._boxes else None)
        if not self._fitted:
            self._fitted = True
            QtCore.QTimer.singleShot(0, self._fit)

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
            self.details.setHtml(_overview_html(self.flow))
        else:
            self.details.setHtml(_step_html(self.flow, self.flow.step(self._selected_key)))

    # view actions

    def _fit(self):
        rect = self.scene.itemsBoundingRect().adjusted(-20, -20, 20, 20)
        if rect.isValid():
            self.view.fitInView(rect, QtCore.Qt.AspectRatioMode.KeepAspectRatio)

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
    """Graphics view: drag to pan, Ctrl + wheel to zoom about the cursor (plain wheel scrolls)."""

    def __init__(self, scene):
        super().__init__(scene)
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | QtGui.QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            factor = 1.0015 ** event.angleDelta().y()
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)


class _StepBox(QtWidgets.QGraphicsPathItem):
    """Rounded, selectable box of one FlowStep (no Qt selection outline: a thicker blue border instead)."""

    def __init__(self, step, rect, fill, border):
        super().__init__()
        self.key = step.key
        self._border = QtGui.QColor(border)
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, 8.0, 8.0)
        self.setPath(path)
        self.setBrush(QtGui.QColor(fill))
        self._set_border(False)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemClipsChildrenToShape, True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

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


# --- drawing --------------------------------------------------------------------------------------------------------

def _box_rect(step) -> QtCore.QRectF:
    """Scene rectangle of a step: centred over its lanes, one lane's box width per lane (capped for wide steps)."""
    span = step.lane_end - step.lane_start
    width = BOX_WIDTH if span <= 1.0 else min(span * LANE_WIDTH - (LANE_WIDTH - BOX_WIDTH), WIDE_BOX_WIDTH)
    centre = ROW_LABEL_WIDTH + 0.5 * (step.lane_start + step.lane_end) * LANE_WIDTH
    return QtCore.QRectF(centre - 0.5 * width, step.row * ROW_HEIGHT, width, BOX_HEIGHT)


def _draw_flow(scene, flow, spec) -> dict:
    """Draw row labels, arrows, coupling links, loops and step boxes; returns {step key: _StepBox}."""
    rects = {step.key: _box_rect(step) for step in flow.steps}
    largest_bytes = max((step.memory_bytes for step in flow.steps), default=0)
    largest_seconds = max((step.time_seconds for step in flow.steps), default=0.0)
    for row, label in enumerate(flow.rows):
        if label:
            text = scene.addSimpleText(label)
            text.setBrush(QtGui.QColor(_SUBTEXT_COLOR))
            text.setPos(0.0, row * ROW_HEIGHT + 0.5 * BOX_HEIGHT - 0.5 * text.boundingRect().height())
    arrow_pen = QtGui.QPen(QtGui.QColor(_BORDER_COLOR), 1.2)
    for step in flow.steps:
        for before in step.after:
            if before in rects:
                _draw_arrow(scene, rects[before], rects[step.key], arrow_pen)
    for index, link in enumerate(flow.links):
        if link.source in rects and link.target in rects:
            _draw_link(scene, rects[link.source], rects[link.target], link.label, index)
    right = ROW_LABEL_WIDTH + flow.lanes * LANE_WIDTH
    for index, loop in enumerate(flow.loops):
        _draw_loop(scene, loop, right + 16 + index * LOOP_SPACING)
    boxes = {}
    for step in flow.steps:
        border = (mesh_color(spec, step.key.split(':', 1)[1]) if step.kind in ('mesh', 'materials')
                  else _BORDER_COLOR)
        box = _StepBox(step, rects[step.key], _KIND_COLORS.get(step.kind, '#eeeeee'), border)
        box.setZValue(2)
        box.setToolTip(rich_tooltip(step.explanation))
        scene.addItem(box)
        _add_box_text(box, step, rects[step.key])
        _add_bars(box, step, rects[step.key], largest_bytes, largest_seconds)
        boxes[step.key] = box
    return boxes


def _add_box_text(box, step, rect):
    estimate = []
    if step.memory:
        estimate.append(format_bytes(step.memory_bytes))
    if step.time:
        estimate.append(format_seconds(step.time_seconds))
    if step.not_estimated:
        estimate.append(f'{len(step.not_estimated)} not estimated')
    text = QtWidgets.QGraphicsTextItem(box)
    text.setDefaultTextColor(QtGui.QColor(_TEXT_COLOR))
    text.setHtml(f'<b>{html.escape(step.title)}</b><br>'
                 f'<span style="color:{_SUBTEXT_COLOR}">{html.escape(step.subtitle)}</span>'
                 + (f'<br><span style="color:{_SUBTEXT_COLOR}; font-size:small">{html.escape(" · ".join(estimate))}'
                    '</span>' if estimate else ''))
    text.setTextWidth(rect.width() - 10.0)
    text.setPos(rect.left() + 5.0, rect.top() + 2.0)


def _add_bars(box, step, rect, largest_bytes, largest_seconds):
    """Thin bars along the box bottom: memory (blue) and time (orange) relative to the largest step."""
    width = rect.width() - 16.0
    for offset, (value, largest, color) in enumerate(((step.memory_bytes, largest_bytes, _MEMORY_BAR_COLOR),
                                                      (step.time_seconds, largest_seconds, _TIME_BAR_COLOR))):
        if not largest or not value:
            continue
        bar = QtWidgets.QGraphicsRectItem(rect.left() + 8.0, rect.bottom() - 10.0 + offset * (_BAR_HEIGHT + 2),
                                          max(width * value / largest, 2.0), _BAR_HEIGHT, box)
        bar.setBrush(QtGui.QColor(color))
        bar.setPen(QtGui.QPen(QtCore.Qt.PenStyle.NoPen))


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
    depth = 18.0 + 8.0 * (index % 3)
    path = QtGui.QPainterPath(start)
    path.cubicTo(start + QtCore.QPointF(0.0, depth), end + QtCore.QPointF(0.0, depth), end)
    scene.addPath(path, pen).setZValue(3)
    _arrow_head(scene, end, QtCore.QPointF(0.0, -1.0), pen.color())
    text = scene.addSimpleText(label)
    text.setBrush(QtGui.QColor(_LINK_COLOR))
    width = text.boundingRect().width()
    text.setPos(0.5 * (start.x() + end.x()) - 0.5 * width, start.y() + 0.75 * depth)
    text.setZValue(3)


def _draw_loop(scene, loop, x):
    """Bracket at x from the top of first_row to the bottom of last_row, an arrow back up and its title."""
    pen = QtGui.QPen(QtGui.QColor(_LOOP_COLOR), 1.6)
    top, bottom = loop.first_row * ROW_HEIGHT + 4.0, loop.last_row * ROW_HEIGHT + BOX_HEIGHT - 4.0
    path = QtGui.QPainterPath(QtCore.QPointF(x - 10.0, bottom))
    path.lineTo(x, bottom)
    path.lineTo(x, top)
    path.lineTo(x - 10.0, top)
    item = scene.addPath(path, pen)
    item.setToolTip(rich_tooltip(f'{loop.title}: {loop.text}'))
    _arrow_head(scene, QtCore.QPointF(x - 10.0, top), QtCore.QPointF(-1.0, 0.0), pen.color())
    text = scene.addSimpleText(loop.title)
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

def _overview_html(flow) -> str:
    parts = ['<h3>Solve flow totals</h3>', _table(flow.totals) if flow.totals else '<p>No estimates.</p>']
    if flow.notes:
        parts.append('<h4>Notes</h4><ul>' + ''.join(f'<li>{html.escape(note)}</li>' for note in flow.notes) + '</ul>')
    parts.append('<p style="color:gray">Memory items are the arrays each step\'s estimator counts from the code '
                 '(meshing.estimate_memory); time items are one study run priced on this computer\'s profile '
                 '(model.time_estimate, python -m open_sol profile). The whole-study time also counts every sweep '
                 'value, sector and estimation run. Click a box for its items.</p>')
    return ''.join(parts)


def _step_html(flow, step) -> str:
    parts = [f'<h3>{html.escape(step.title)}</h3>']
    if step.subtitle:
        parts.append(f'<p style="color:gray">{html.escape(step.subtitle)}</p>')
    if step.explanation:
        parts.append(f'<p>{html.escape(step.explanation)}</p>')
    if step.facts:
        parts.append(_table(step.facts))
    parts.append(_memory_html(flow, step))
    parts.append(_time_html(step))
    if step.not_estimated:
        parts.append('<h4>Not estimated</h4><ul>'
                     + ''.join(f'<li>{html.escape(label)}</li>' for label in step.not_estimated) + '</ul>')
    return ''.join(parts)


def _memory_html(flow, step) -> str:
    if not step.memory:
        return ''
    rows = []
    for entry in sorted(step.memory, key=lambda entry: -entry.bytes):
        group = MEMORY_GROUPS.get(entry.group, entry.group)
        if entry.phase:
            group += f' (phase: {entry.phase})'
        rows.append(f'<tr><td>{html.escape(entry.label)}</td><td align="right">{format_bytes(entry.bytes)}</td>'
                    f'<td style="color:gray">{html.escape(group)}</td></tr>')
    note = ('<p style="color:gray">On the cpu backend the device arrays live in processor memory.</p>'
            if flow.backend == 'cpu' else '')
    return (f'<h4>Memory items ({format_bytes(step.memory_bytes)} listed)</h4>'
            '<table cellspacing="0" cellpadding="3" border="0">'
            '<tr><th align="left">Item</th><th align="right">Bytes</th><th align="left">Kind</th></tr>'
            + ''.join(rows) + '</table>' + note)


def _time_html(step) -> str:
    if not step.time:
        return ''
    rows = []
    for entry in sorted(step.time, key=lambda entry: -entry.seconds):
        item = entry.item
        low, guess, high = item.counts()
        count = f'{guess:.4g} {item.unit}s ({item.count_kind}'
        count += f'; {low:.3g} - {high:.3g})' if item.count_kind != 'exact' else ')'
        if item.limit is not None:
            count += f', at most {item.limit:.4g}'
        ops = '<br>'.join(html.escape(op_text(op)) for op in item.ops) or '—'
        measured = '.'.join(str(part) for part in item.counter) if item.counter else ''
        rows.append(f'<tr><td>{html.escape(entry.label)}<br><span style="color:gray; font-size:small">'
                    f'{html.escape(TIME_SOURCES.get(entry.source, entry.source))}'
                    + (f'; measured by summary {html.escape(measured)}' if measured else '') + '</span></td>'
                    f'<td align="right">{format_seconds(entry.seconds)}</td>'
                    f'<td>{html.escape(count)}<br>x {format_seconds(item.seconds_per_unit or 0.0)} each</td>'
                    f'<td style="font-family:monospace; font-size:small">{ops}</td></tr>')
    return (f'<h4>Run-time items ({format_seconds(step.time_seconds)})</h4>'
            '<p style="color:gray">seconds = count x seconds per unit; each unit costs the listed operations, priced '
            'on this computer\'s profile.</p>'
            '<table cellspacing="0" cellpadding="3" border="0">'
            '<tr><th align="left">Item</th><th align="right">Time</th><th align="left">Count</th>'
            '<th align="left">Operations per unit</th></tr>' + ''.join(rows) + '</table>')


def _table(rows) -> str:
    return ('<table cellspacing="0" cellpadding="2" border="0">'
            + ''.join(f'<tr><td style="color:gray; padding-right:12px">{html.escape(str(name))}</td>'
                      f'<td>{html.escape(str(value))}</td></tr>' for name, value in rows) + '</table>')
