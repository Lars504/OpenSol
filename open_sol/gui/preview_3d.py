"""3D preview dock (view_3d C): perspective mplot3d picture of the document's 3D model (model.scene_3d drawn by
viz.scene_figure), redrawn 300 ms after an edit and only while visible; drag to turn, Reset view, Export image;
optionally the Results field on x, y and z cut planes (fetched through the ResultsPanel)."""
from PySide6 import QtCore, QtWidgets

from ..model.expressions import error_message
from ..model.scene_3d import build_scene
from ..viz.scene_figure import DEFAULT_VIEW, PLANE_CELLS, draw_scene_3d, new_scene_axes
from .canvas import _CATEGORY_COLORS, _HIGHLIGHT_COLOR, _MATERIAL_COLORS
from .forms import rich_tooltip
from .helpers import ask_save_path, remember_directory

_REDRAW_DELAY_MS = 300
_SCENE_ERRORS = (ValueError, KeyError, TypeError, ArithmeticError)   # ExpressionError is a ValueError
_EXPORT_ERRORS = (OSError, ValueError, RuntimeError)
_EXPORT_TYPES = (('PNG image', '.png'), ('PDF document', '.pdf'), ('SVG image', '.svg'))
_STRETCH_HEIGHT_FRACTION = 0.5   # 'Stretch z': the box height becomes at least this fraction of its width
_PROBLEMS_SHOWN = 3


class Preview3DPanel(QtWidgets.QWidget):
    """Perspective view of a 3D model's objects in their material colours; the current node's selection is drawn
    opaque in the highlight colour. Matplotlib's Qt canvas is created on the first draw (nothing costs while closed)."""

    message = QtCore.Signal(str)   # status-bar text (export results)

    def __init__(self, document, registry, highlight_request, field_source=None, parent=None):
        super().__init__(parent)
        self._document = document
        self._registry = registry
        self._highlight_request = highlight_request   # () -> (entities, fill): ModelCanvas._highlight_request
        self._field_source = field_source   # ResultsPanel (fetch_cut_planes, cut_plane_colors, field_view_changed)
        self._planes = []                   # FieldData cut planes drawn while 'Field cut planes' is on
        self._planes_stale = True           # the Results field / cut / session changed since the last fetch
        self._planes_pending = False        # a fetch_cut_planes call is on its way
        self._planes_note = ''              # why no planes are drawn ('' when they are or the box is off)
        self._dirty = True
        self._last_highlight = None
        self._figure = None
        self._canvas = None
        self._axes = None
        self._view = DEFAULT_VIEW                      # (elevation, azimuth) kept across redraws
        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._redraw_if_visible)

        self._stretch_check = QtWidgets.QCheckBox('Stretch z')
        self._stretch_check.setToolTip(rich_tooltip('Draw the height larger, so thin layers become visible. The '
                                                    'height is then not to scale with the width.'))
        self._stretch_check.toggled.connect(lambda _checked: self._schedule(0))
        self._planes_check = QtWidgets.QCheckBox('Field cut planes')
        self._planes_check.setToolTip(rich_tooltip(
            'Show the field chosen in Results on three cuts through the model in one colour scale: the cut plane '
            'chosen there, and the middle of the model across the other two directions.\n\n'
            '- The objects are then drawn as outlines only.\n'
            '- Colours follow the colormap and symmetric-limits choice of Results.\n'
            '- The planes show the current results (after a build or a computation), not the edited model.'))
        self._planes_check.toggled.connect(self._on_planes_toggled)
        self._planes_check.setVisible(field_source is not None)
        reset_button = QtWidgets.QPushButton('Reset view')
        reset_button.setToolTip('Back to the starting viewing direction and zoom.')
        reset_button.clicked.connect(self._reset_view)
        export_button = QtWidgets.QPushButton('Export image…')
        export_button.setToolTip('Save the picture as shown (PNG, PDF or SVG).')
        export_button.clicked.connect(self._export_image)
        controls = QtWidgets.QHBoxLayout()
        controls.setContentsMargins(4, 2, 4, 0)
        for widget in (self._stretch_check, self._planes_check, reset_button, export_button):
            controls.addWidget(widget)
        controls.addStretch()
        self._note = QtWidgets.QLabel()
        self._note.setWordWrap(True)
        self._note.setStyleSheet('color:#555; padding:2px 4px;')
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(2)
        self._layout.addLayout(controls)
        self._layout.addWidget(self._note)
        self.setToolTip(rich_tooltip('Perspective picture of the model: drag with the left mouse button to turn it, '
                                     'with the right one to zoom.\n\nView only: objects are drawn and edited on the '
                                     'canvas. Dashed outlines are objects cut out of (or intersected with) another '
                                     'in a combination.'))

        document.changed.connect(self._on_document_changed)
        document.replaced.connect(self._on_document_replaced)
        document.node_selected.connect(self._on_node_selected)
        document.geometry_group_changed.connect(self._on_node_selected)
        if field_source is not None:
            field_source.field_view_changed.connect(self._on_field_view_changed)

    # ---- scheduling ----
    def _schedule(self, delay_ms=_REDRAW_DELAY_MS):
        self._dirty = True
        if self.isVisible():
            self._timer.start(delay_ms)

    def showEvent(self, event):
        super().showEvent(event)
        if self._dirty:
            self._timer.start(0)

    def _on_document_changed(self, section, origin):
        if section in ('model', 'results'):   # title / description / result settings: only a highlight may change
            self._on_node_selected(None)
            return
        self._schedule()

    def _on_document_replaced(self):
        self._view = DEFAULT_VIEW
        self._axes = None
        self._planes, self._planes_stale = [], True
        self._planes_pending = False   # ResultsPanel._call drops the answer of a call made before the replacement
        self._schedule()

    def _on_node_selected(self, node_path):
        if self._current_highlight() != self._last_highlight:
            self._schedule(0)

    def _current_highlight(self) -> list:
        entities, _fill = self._highlight_request()
        return list(entities)

    def _redraw_if_visible(self):
        if self.isVisible():
            self._dirty = False
            self._redraw()

    # ---- drawing ----
    def _redraw(self):
        spec = self._document.spec
        if spec.domain.size_z is None:
            self._show_note("The 3D preview shows 3D models: set 'Depth (z, 3D only)' in the domain settings.",
                            canvas=False)
            return
        self._last_highlight = self._current_highlight()
        try:
            scene = build_scene(spec, self._document.parameters(), self._registry, highlight=self._last_highlight)
        except _SCENE_ERRORS as exc:
            self._show_note(f'Cannot draw the model: {error_message(exc)}', canvas=False)
            return
        self._ensure_canvas()
        if self._axes is not None:
            self._view = (self._axes.elev, self._axes.azim)
        field_planes = self._field_planes()
        cmap, symmetric = self._field_source.cut_plane_colors() if field_planes else (None, None)
        self._figure.clear()
        self._axes = new_scene_axes(self._figure)
        draw_scene_3d(scene, self._axes, material_colors=_MATERIAL_COLORS, category_colors=_CATEGORY_COLORS,
                      highlight_color=_HIGHLIGHT_COLOR, z_stretch=self._z_stretch(scene.size), view=self._view,
                      field_planes=field_planes, cmap=cmap, symmetric=symmetric)
        self._canvas.draw_idle()
        problems = scene.problems[:_PROBLEMS_SHOWN] + (['…'] if len(scene.problems) > _PROBLEMS_SHOWN else [])
        notes = (['Not drawn: ' + '; '.join(problems)] if problems else []) + (
            [self._planes_note] if self._planes_check.isChecked() and self._planes_note else [])
        self._show_note('\n'.join(notes), canvas=True)

    # ---- field cut planes ----
    def _on_planes_toggled(self, checked):
        self._planes_stale = True
        self._schedule(0)

    def _on_field_view_changed(self, refetch):
        """ResultsPanel.field_view_changed: refetch (field, cut or session changed) or redraw (colours only)."""
        if refetch:
            self._planes_stale = True
        if self._planes_check.isChecked():
            self._schedule(0)

    def _field_planes(self) -> list:
        """Cut planes to draw: [] unless 'Field cut planes' is on; a stale set is refetched (redrawn on arrival)."""
        if self._field_source is None or not self._planes_check.isChecked():
            return []
        if self._planes_stale and not self._planes_pending:
            self._planes_stale = False
            reason = self._field_source.fetch_cut_planes(self._on_planes_fetched, self._on_planes_failed,
                                                         PLANE_CELLS)
            self._planes_pending = not reason
            if reason:
                self._planes, self._planes_note = [], reason
        return self._planes

    def _on_planes_fetched(self, field_planes):
        self._planes_pending = False
        self._planes, self._planes_note = list(field_planes), ''
        self._schedule(0)

    def _on_planes_failed(self, message):
        self._planes_pending = False
        first_line = str(message).strip().splitlines()[0] if str(message).strip() else 'unknown error'
        self._planes, self._planes_note = [], f'Field cut planes not drawn: {first_line}'
        self._schedule(0)

    def _z_stretch(self, size) -> float:
        """Height scale of the box: 1, or with 'Stretch z' enough to make it _STRETCH_HEIGHT_FRACTION of its width."""
        if not self._stretch_check.isChecked():
            return 1.0
        size_x, size_y, size_z = size
        return max(1.0, _STRETCH_HEIGHT_FRACTION * max(size_x, size_y) / size_z)

    def _ensure_canvas(self):
        """Create matplotlib's Qt canvas on first use (matplotlib picks PySide6, imported first by gui/__init__)."""
        if self._canvas is not None:
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self._figure = Figure(figsize=(5.0, 4.0))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self._layout.addWidget(self._canvas, 1)

    def _show_note(self, text, canvas):
        self._note.setText(text)
        self._note.setVisible(bool(text))
        if self._canvas is not None:
            self._canvas.setVisible(canvas)

    # ---- buttons ----
    def _reset_view(self):
        self._view = DEFAULT_VIEW
        self._axes = None          # a redraw also drops mplot3d's mouse zoom
        self._schedule(0)

    def _export_image(self):
        if self._figure is None or self._canvas is None or not self._canvas.isVisible():
            self.message.emit('Nothing to export: the 3D preview shows no model')
            return
        path = ask_save_path(self, self._document, 'Export 3D preview', '3d_preview.png', _EXPORT_TYPES)
        if not path:
            return
        remember_directory(path)
        try:
            self._figure.savefig(path, dpi=200)
        except _EXPORT_ERRORS as exc:
            self.message.emit(f'Could not export the 3D preview: {exc}')
            return
        self.message.emit(f'Saved {path}')
