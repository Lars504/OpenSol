"""Main window: model canvas in the centre, model tree / settings / results / 3D preview docks, Messages / Validation /
Progress tabs, menus and a toolbar with progress + ETA; every GPU job goes through the WorkerBridge."""
import collections
import math
import numbers
import os
import time
from functools import partial

from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__
from ..core.gpu_memory import format_bytes
from ..model.expressions import error_message
from ..model.geometry import GEOMETRY_KINDS
from ..model.schema import ISSUE_LEVELS, Issue
from ..model.templates import template_entries
from .canvas import ModelCanvas
from .document import COMPUTED_RESULTS_NODE, issue_node_path
from .examples import MULTIPHYSICS_FOLDER, ExampleCatalog
from .forms import NO_VALUE, format_quantity, rich_tooltip
from .helpers import (USER_ERRORS, ask_run_directory, ask_save_path, format_duration, remember_directory,
                      run_user_operation, start_directory, study_summary_text)
from .mesh_preview import ElementMeshStore
from .model_tree import ISSUE_LEVEL_PIXMAPS, ModelTree, decode_node_path, encode_node_path
from .preview_3d import Preview3DPanel
from .results import ResultsPanel
from .settings_panel import SettingsPanel

SETTINGS_ORGANIZATION = 'open_sol'
SETTINGS_APPLICATION = 'open_sol'
_MAX_RECENT_FILES = 10
_WORKER_POLL_MS = 250
_STATUS_TIMEOUT_MS = 6000
_JOB_LOG_MAX_LINES = 100_000      # Messages lines kept for a saved run's log.txt (the view itself keeps 5000)
_PROGRESS_STEPS = 1000            # progress bar resolution
_MEMORY_STEPS = 1000              # GPU memory gauge resolution
_MEMORY_WARNING_FRACTION = 0.9    # gauge turns red above this fraction of the budget
_MEMORY_WARNING_STYLE = 'QProgressBar::chunk { background-color: #e53935; }'
_MEMORY_LEGEND = ('In use: simulation data of this program. Reserved: memory it keeps for reuse.\n\n'
                  'Limit: the most it may reserve, set before each job. Other: graphics driver, libraries and other '
                  'programs.\n\nAvailable processor memory: free main memory (RAM), the limit of the processor-side '
                  'solvers.')
_CANVAS_TOOLS = (('select', 'Select'), ('rectangle', 'Draw rectangle'), ('circle', 'Draw circle'),
                 ('ellipse', 'Draw ellipse'), ('polygon', 'Draw polygon'), ('point', 'Add point'),
                 ('line', 'Draw line'), ('profile', 'Line profile'))
_TOOLBAR_TOOLS = ('select', 'rectangle', 'circle', 'polygon', 'line', 'point')
_LEVEL_COLORS = {'error': '#c62828', 'warning': '#a15c00', 'info': '#202020'}
_MESSAGES_MIN_WRAP_COLUMNS = 80   # Messages lines wrap at the view width, never narrower (then scroll sideways)
_MESSAGES_HANGING_INDENT_COLUMNS = 9   # wrapped / continuation lines start below the text after 'HH:MM:SS '
_MESSAGES_GAP_PX = 4                   # space between two messages
_COMPUTE_ICON_COLOR = '#2e9d44'
_STOP_ICON_COLOR = '#d32f2f'
_PROGRESS_FIELDS = (('status', 'Status'), ('stage', 'Current stage'), ('stages', 'Stages'),
                    ('time', 'Simulated time'), ('tick', 'Step / iteration / wavelength'),
                    ('speed', 'Speed'), ('elapsed', 'Elapsed time'), ('eta', 'Time remaining'), ('sweep', 'Sweep'),
                    ('build', 'Last build'), ('gpu', 'Graphics card memory'))
_BUILD_MEMORY_TIP = ('Graphics card memory each step of the last build added. "Build temporaries freed" is the '
                     'memory the build needed only while it ran (masks, coordinate grids), returned at its end.\n\n'
                     'The peak is the most the build held at once. The estimate is computed before building; a '
                     'large difference is reported as a warning.')
_WHATS_THIS_TIP = "Click a setting's label to read its full technical description (Shift+F1)."


class _MessagesView(QtWidgets.QTextEdit):
    """Read-only log wrapping at the view width, but at no fewer than _MESSAGES_MIN_WRAP_COLUMNS characters."""

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_wrap()

    def _update_wrap(self):
        min_wrap_width = int(self.fontMetrics().horizontalAdvance('0') * _MESSAGES_MIN_WRAP_COLUMNS
                             + 2 * self.document().documentMargin())
        if self.viewport().width() >= min_wrap_width:
            self.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
            return
        if self.lineWrapColumnOrWidth() != min_wrap_width:   # each set relayouts the whole document
            self.setLineWrapColumnOrWidth(min_wrap_width)
        self.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.FixedPixelWidth)


def _finite(value) -> bool:
    """True for a finite real number (numpy scalars included, bools excluded)."""
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _unit_interval(value) -> float:
    return min(1.0, max(0.0, float(value))) if _finite(value) else 0.0


def _eta_text(fraction, elapsed) -> str:
    """Remaining wall time extrapolated linearly from the completed fraction."""
    if not _finite(elapsed) or fraction <= 0.0:
        return NO_VALUE
    return format_duration(elapsed * (1.0 - fraction) / fraction)


def _stages_text(stages, stage_index, stage_fraction) -> str:
    """'Background solve ✓ → Wavelength solves (45 %) → ...': done, current and waiting stages of a run."""
    parts = []
    for position, name in enumerate(stages):
        if position < stage_index:
            parts.append(f'{name} ✓')
        elif position == stage_index:
            parts.append(f'{name} ({100.0 * stage_fraction:.0f} %)')
        else:
            parts.append(name)
    return ' → '.join(parts)


def _media_icon(shape, color) -> QtGui.QIcon:
    """Filled 'play' triangle or 'stop' square in color; Qt greys it out while the action is disabled."""
    pixmap = QtGui.QPixmap(64, 64)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    painter.setBrush(QtGui.QColor(color))
    if shape == 'play':
        painter.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(14, 6), QtCore.QPointF(58, 32), QtCore.QPointF(14, 58)]))
    else:
        painter.drawRoundedRect(QtCore.QRectF(10, 10, 44, 44), 5, 5)
    painter.end()
    return QtGui.QIcon(pixmap)


def _node_text(path) -> str:
    return ' / '.join(str(part) for part in path)


def _issue_key(issue) -> tuple:
    return issue.level, tuple(issue.path), issue.message


class _WorkerWaitDialog(QtWidgets.QDialog):
    """Modal 'Waiting for the running solve…' box: closes once worker_stopped() is True; Force quit ends the process."""

    def __init__(self, parent, worker_stopped, force_quit):
        super().__init__(parent)
        self.setWindowTitle('Closing open_sol')
        self.setModal(True)
        self.setWindowFlag(QtCore.Qt.WindowType.WindowCloseButtonHint, False)
        self._worker_stopped = worker_stopped
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel('Waiting for the running solve…\nThe worker (GPU kernel or CPU solve, '
                                          'e.g. an eigen-solve) has to return before open_sol can exit cleanly.\n'
                                          'Force quit ends the process now; unsaved changes are lost.'))
        buttons = QtWidgets.QDialogButtonBox()
        force_button = buttons.addButton('Force quit', QtWidgets.QDialogButtonBox.ButtonRole.DestructiveRole)
        force_button.clicked.connect(lambda _checked=False: force_quit())
        layout.addWidget(buttons)
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(_WORKER_POLL_MS)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start()

    def _poll(self):
        if self._worker_stopped():
            self._poll_timer.stop()
            self.accept()

    def reject(self):
        """Esc / window close do nothing: only the worker stopping or Force quit ends the wait."""


class MainWindow(QtWidgets.QMainWindow):
    """Top-level window: owns the views and wires the document, the worker bridge and the canvas together."""

    def __init__(self, document, bridge, registry, parent=None):
        super().__init__(parent)
        self.document = document
        self.bridge = bridge
        self._settings = QtCore.QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)
        self._validation_refresh_pending = False
        self._shown_issue_keys = None         # (level, path, message) of the issues the Validation tab shows
        self._build_issues = []               # Issues of the last build (failure issues or summary warnings)
        self._build_issue_revision = None     # document revision those issues belong to
        self._requested_revision = None       # document revision of the last Build / Compute request
        self._build_failed_in_job = False
        self._job_kind = None                 # 'build' | 'run' of the last request
        self._run_started_at = None           # time.monotonic() of the last Build / Compute request
        # (clock, level, message) since that request: log.txt of Save run (the last _JOB_LOG_MAX_LINES)
        self._job_log = collections.deque(maxlen=_JOB_LOG_MAX_LINES)
        self._job_model_file = None           # document path at that request
        self._sweep_position = None           # (index, count) of the running sweep step
        self._sweep_text = ''
        self._worker_stopped = False
        self._gpu_device_text = ''            # memory gauge tooltip header from the worker's DeviceInfo
        self._example_catalog = ExampleCatalog(registry)

        self.canvas = ModelCanvas(document, registry)
        self.model_tree = ModelTree(document, registry)
        self.settings_panel = SettingsPanel(document, registry)
        self.results_panel = ResultsPanel(document, bridge, self.canvas)
        self.mesh_store = ElementMeshStore(document, bridge, self)   # after ResultsPanel: its slots run later
        self.canvas.set_mesh_store(self.mesh_store)
        self.settings_panel.set_mesh_store(self.mesh_store)
        self.preview_3d_panel = Preview3DPanel(document, registry, self.canvas._highlight_request,
                                               field_source=self.results_panel)
        self.setCentralWidget(self.canvas)

        self._create_docks()
        self._create_actions()
        self._create_toolbar()
        self._create_menus(registry)
        self._create_status_bar()
        self._connect_signals()
        self.resize(1500, 950)
        self._restore_settings()
        self._update_title()
        self._update_undo_actions()
        self._update_run_actions()
        self._refresh_validation()

    # ---- public ----
    @QtCore.Slot(str, str)
    def log_message(self, level, text):
        """Append a timestamped line to the Messages tab, coloured by level ('error' | 'warning' | 'info')."""
        char_format = QtGui.QTextCharFormat()
        char_format.setForeground(QtGui.QBrush(QtGui.QColor(_LEVEL_COLORS.get(level, _LEVEL_COLORS['info']))))
        hanging_indent = self.messages_view.fontMetrics().horizontalAdvance('0') * _MESSAGES_HANGING_INDENT_COLUMNS
        block_format = QtGui.QTextBlockFormat()
        block_format.setLeftMargin(hanging_indent)
        block_format.setTextIndent(-hanging_indent)
        block_format.setBottomMargin(_MESSAGES_GAP_PX)
        text_document = self.messages_view.document()
        cursor = QtGui.QTextCursor(text_document)
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        if text_document.isEmpty():
            cursor.setBlockFormat(block_format)
        else:
            cursor.insertBlock(block_format)
        clock, message = time.strftime('%H:%M:%S'), str(text).rstrip()
        self._job_log.append((clock, level, message))
        shown_message = message.replace('\n', ' ')   # line separator: a multi-line message stays one indented block
        cursor.insertText(f'{clock} [{level}] {shown_message}', char_format)
        scroll_bar = self.messages_view.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    @QtCore.Slot(str)
    def report_uncaught_exception(self, text):
        """Show an uncaught exception (sys/threading excepthook) in the Messages tab."""
        self.log_message('error', text)
        self._show_bottom_tab(self.messages_view)
        self.show_status('Unexpected error: see the Messages tab')

    def show_status(self, message, timeout_ms=_STATUS_TIMEOUT_MS):
        self.statusBar().showMessage(message, timeout_ms)

    # ---- layout ----
    def _make_dock(self, title, object_name, widget, area):
        dock = QtWidgets.QDockWidget(title, self)
        dock.setObjectName(object_name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def _create_docks(self):
        left = QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
        right = QtCore.Qt.DockWidgetArea.RightDockWidgetArea
        bottom = QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        self.setCorner(QtCore.Qt.Corner.BottomLeftCorner, left)
        self.setCorner(QtCore.Qt.Corner.BottomRightCorner, right)
        self.tree_dock = self._make_dock('Model', 'model_tree_dock', self.model_tree, left)
        self.settings_dock = self._make_dock('Settings', 'settings_dock', self.settings_panel, right)
        self.results_dock = self._make_dock('Results', 'results_dock', self.results_panel, right)
        self.tabifyDockWidget(self.settings_dock, self.results_dock)
        self.preview_3d_dock = self._make_dock('3D preview', 'preview_3d_dock', self.preview_3d_panel, right)
        self.tabifyDockWidget(self.results_dock, self.preview_3d_dock)
        self.preview_3d_dock.hide()           # opened from the View menu (draws only while visible)
        self.settings_dock.raise_()

        self.messages_view = _MessagesView()
        self.messages_view.setReadOnly(True)
        self.messages_view.document().setMaximumBlockCount(5000)
        self.messages_view.setWordWrapMode(QtGui.QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.messages_view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.messages_view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.messages_view.customContextMenuRequested.connect(self._show_messages_menu)

        self.validation_view = QtWidgets.QTreeWidget()
        self.validation_view.setColumnCount(3)
        self.validation_view.setHeaderLabels(['Level', 'Node', 'Message'])
        self.validation_view.setRootIsDecorated(False)
        self.validation_view.setUniformRowHeights(True)
        self.validation_view.setAlternatingRowColors(True)
        self.validation_view.setColumnWidth(0, 90)
        self.validation_view.setColumnWidth(1, 220)
        self.validation_view.itemDoubleClicked.connect(self._on_validation_double_clicked)

        self.progress_view = QtWidgets.QWidget()
        progress_columns = QtWidgets.QHBoxLayout(self.progress_view)
        progress_form = QtWidgets.QWidget()
        progress_layout = QtWidgets.QFormLayout(progress_form)
        self._progress_fields = {}
        self.stage_bar = QtWidgets.QProgressBar()   # fraction of the current stage of the run
        self.stage_bar.setRange(0, _PROGRESS_STEPS)
        self.stage_bar.setFixedWidth(120)
        for key, label in _PROGRESS_FIELDS:
            value_label = QtWidgets.QLabel(NO_VALUE)
            value_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            if key == 'stage':
                stage_row = QtWidgets.QWidget()
                stage_layout = QtWidgets.QHBoxLayout(stage_row)
                stage_layout.setContentsMargins(0, 0, 0, 0)
                stage_layout.addWidget(self.stage_bar)
                stage_layout.addWidget(value_label, 1)
                progress_layout.addRow(f'{label}:', stage_row)
            else:
                progress_layout.addRow(f'{label}:', value_label)
            self._progress_fields[key] = value_label
        self._progress_fields['stages'].setWordWrap(True)
        self.build_memory_view = QtWidgets.QTreeWidget()   # graphics card memory per build stage (build ledger)
        self.build_memory_view.setColumnCount(2)
        self.build_memory_view.setHeaderLabels(['Build stage', 'Graphics card memory'])
        self.build_memory_view.setRootIsDecorated(False)
        self.build_memory_view.setUniformRowHeights(True)
        self.build_memory_view.setColumnWidth(0, 260)
        self.build_memory_view.setToolTip(rich_tooltip(_BUILD_MEMORY_TIP))
        progress_columns.addWidget(progress_form, 1)
        progress_columns.addWidget(self.build_memory_view, 1)

        self.bottom_tabs = QtWidgets.QTabWidget()
        self.bottom_tabs.addTab(self.messages_view, 'Messages')
        self.bottom_tabs.addTab(self.validation_view, 'Validation')
        self.bottom_tabs.addTab(self.progress_view, 'Progress')
        self.bottom_dock = self._make_dock('Output', 'output_dock', self.bottom_tabs, bottom)

        self.resizeDocks([self.tree_dock, self.settings_dock], [280, 420], QtCore.Qt.Orientation.Horizontal)
        self.resizeDocks([self.bottom_dock], [190], QtCore.Qt.Orientation.Vertical)

    def _make_action(self, text, slot, shortcut=None, tip='', icon=None):
        action = QtGui.QAction(text, self)
        if isinstance(shortcut, QtGui.QKeySequence.StandardKey):
            action.setShortcuts(shortcut)             # every platform binding (e.g. Ctrl+Y and Ctrl+Shift+Z)
        elif shortcut is not None:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        if tip:
            action.setStatusTip(tip)
            action.setToolTip(tip)
        if isinstance(icon, QtGui.QIcon):
            action.setIcon(icon)
        elif icon is not None:
            action.setIcon(self.style().standardIcon(icon))
        action.triggered.connect(lambda _checked=False: slot())
        return action

    def _create_actions(self):
        keys = QtGui.QKeySequence.StandardKey
        pixmaps = QtWidgets.QStyle.StandardPixmap
        self.new_action = self._make_action('&New', partial(self._new_model, None), keys.New,
                                            'Start a blank model', pixmaps.SP_FileIcon)
        self.open_action = self._make_action('&Open…', self._open_dialog, keys.Open, 'Open a model file',
                                             pixmaps.SP_DialogOpenButton)
        self.save_action = self._make_action('&Save', self._save, keys.Save, 'Save the model',
                                             pixmaps.SP_DialogSaveButton)
        self.save_as_action = self._make_action('Save &As…', self._save_as, keys.SaveAs,
                                                'Save the model under a new name')
        self.save_run_action = self._make_action('Save &run…', self._save_run,
                                                 tip='Save the last computed run (summary.json, log, model, result '
                                                     'arrays) into a folder, as the command-line run does')
        self.export_field_action = self._make_action('Export field &data…', self._export_field_data,
                                                     tip='Save the displayed field as a NumPy file (.npz) or a text '
                                                         'table (.csv)')
        self.export_image_action = self._make_action('Export &figure…', self._export_image,
                                                     tip='Save the canvas figure as shown (PDF, SVG, EPS or PNG; style '
                                                         'in the Results Field tab, Figure box)')
        self.quit_action = self._make_action('&Quit', self.close, 'Ctrl+Q', 'Close open_sol')
        self.undo_action = self._make_action('&Undo', self.document.undo, keys.Undo, 'Undo the last model change',
                                             pixmaps.SP_ArrowBack)
        self.redo_action = self._make_action('&Redo', self.document.redo, keys.Redo, 'Redo the undone change',
                                             pixmaps.SP_ArrowForward)
        # Delete only while the tree or canvas has focus, so Delete in tables and line edits keeps its meaning
        self.delete_action = self._make_action('&Delete node', self.model_tree.delete_current, keys.Delete,
                                               'Delete the selected model node', pixmaps.SP_TrashIcon)
        self.delete_action.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.model_tree.addAction(self.delete_action)
        self.canvas.addAction(self.delete_action)

        self.tool_group = QtGui.QActionGroup(self)
        self.tool_group.setExclusive(True)
        self._tool_actions = {}
        for tool, label in _CANVAS_TOOLS:
            action = QtGui.QAction(label, self)
            action.setCheckable(True)
            action.setStatusTip(f'Canvas tool: {label.lower()}')
            action.triggered.connect(lambda _checked=False, tool=tool: self.canvas.set_tool(tool))
            self.tool_group.addAction(action)
            self._tool_actions[tool] = action
        self._tool_actions['select'].setChecked(True)

        self.build_action = self._make_action('&Build', self._build, 'F7',
                                              'Set up the model on the graphics card without running the study, and '
                                              'report any setup errors',
                                              pixmaps.SP_BrowserReload)
        self.compute_action = self._make_action('&Compute', self._compute, 'F8',
                                                'Validate, build and run the study',
                                                _media_icon('play', _COMPUTE_ICON_COLOR))
        self.pause_action = self._make_action('&Pause', self._toggle_pause, tip='Pause or resume the running study',
                                              icon=pixmaps.SP_MediaPause)
        self.stop_action = self._make_action('S&top', self.bridge.stop, tip='Stop the running study',
                                             icon=_media_icon('stop', _STOP_ICON_COLOR))
        self.show_geometry_action = self._make_action('Show &geometry', self.canvas.show_geometry,
                                                      tip='Show the geometry on the canvas')
        self.show_mesh_action = self._make_action('Show &mesh', self.canvas.show_mesh,
                                                  tip='Show the mesh cells on the canvas')
        self.show_results_action = self._make_action('Show &results', self._show_results,
                                                     tip='Show the last result field on the canvas')
        self.about_action = self._make_action('&About open_sol', self._show_about)
        self.whats_this_action = QtWidgets.QWhatsThis.createAction(self)
        self.whats_this_action.setStatusTip(_WHATS_THIS_TIP)
        self.whats_this_action.setToolTip(_WHATS_THIS_TIP)

    def _create_toolbar(self):
        toolbar = self.addToolBar('Main toolbar')
        toolbar.setObjectName('main_toolbar')
        toolbar.setIconSize(QtCore.QSize(18, 18))
        toolbar.addActions([self.new_action, self.open_action, self.save_action])
        toolbar.addSeparator()
        toolbar.addActions([self.undo_action, self.redo_action])
        toolbar.addSeparator()
        toolbar.addActions([self._tool_actions[tool] for tool in _TOOLBAR_TOOLS])
        toolbar.addSeparator()
        toolbar.addActions([self.build_action, self.compute_action, self.pause_action, self.stop_action])
        toolbar.addSeparator()
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, _PROGRESS_STEPS)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedWidth(200)
        self.progress_bar.setToolTip(rich_tooltip('Progress of the whole run: every stage (background solve, '
                                                  'wavelength solves, ...) and every sweep value.\n\n'
                                                  'The Progress tab shows the current stage on its own.'))
        self.progress_label = QtWidgets.QLabel('Idle')
        self.progress_label.setMinimumWidth(240)
        progress_widget = QtWidgets.QWidget()
        progress_layout = QtWidgets.QHBoxLayout(progress_widget)
        progress_layout.setContentsMargins(6, 0, 6, 0)
        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.progress_label)
        toolbar.addWidget(progress_widget)
        self.main_toolbar = toolbar

    def _create_menus(self, registry):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(self.new_action)
        template_menu = file_menu.addMenu('New from &template')
        template_menu.setToolTipsVisible(True)
        entries = template_entries(registry)
        builtin_entries = [entry for entry in entries if entry.builtin]
        registered_entries = [entry for entry in entries if not entry.builtin]
        for group in (builtin_entries, registered_entries):
            if group and not template_menu.isEmpty():
                template_menu.addSeparator()      # registered (physics-package) templates after the built-ins
            for entry in group:
                action = template_menu.addAction(entry.label)
                template_text = entry.summary or entry.description
                action.setToolTip(rich_tooltip(template_text))
                action.setStatusTip(template_text)
                action.triggered.connect(lambda _checked=False, key=entry.key: self._new_model(key))
        file_menu.addAction(self.open_action)
        self.recent_menu = file_menu.addMenu('Open &recent')
        self.recent_menu.aboutToShow.connect(self._populate_recent_menu)
        file_menu.addSeparator()
        file_menu.addActions([self.save_action, self.save_as_action, self.save_run_action])
        file_menu.addSeparator()
        file_menu.addActions([self.export_field_action, self.export_image_action])
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        edit_menu = menu_bar.addMenu('&Edit')
        edit_menu.addActions([self.undo_action, self.redo_action])
        edit_menu.addSeparator()
        edit_menu.addAction(self.delete_action)

        geometry_menu = menu_bar.addMenu('&Geometry')
        geometry_menu.addActions([self._tool_actions[tool] for tool, _label in _CANVAS_TOOLS if tool != 'profile'])
        geometry_menu.addSeparator()
        self.add_object_menu = geometry_menu.addMenu('Add &object')
        self.add_object_menu.setToolTipsVisible(True)
        self.add_object_menu.aboutToShow.connect(self._populate_add_object_menu)

        study_menu = menu_bar.addMenu('&Study')
        study_menu.addActions([self.build_action, self.compute_action, self.pause_action, self.stop_action])
        study_menu.addSeparator()
        study_menu.addActions([self.show_geometry_action, self.show_mesh_action, self.show_results_action])
        study_menu.addSeparator()
        study_menu.addAction(self._tool_actions['profile'])

        view_menu = menu_bar.addMenu('&View')
        for dock in (self.tree_dock, self.settings_dock, self.results_dock, self.bottom_dock):
            view_menu.addAction(dock.toggleViewAction())
        preview_action = self.preview_3d_dock.toggleViewAction()
        preview_action.setStatusTip('Perspective picture of a 3D model (drag to turn it)')
        preview_action.triggered.connect(lambda checked: checked and self.preview_3d_dock.raise_())
        view_menu.addAction(preview_action)
        view_menu.addAction(self.main_toolbar.toggleViewAction())

        self.examples_menu = menu_bar.addMenu('E&xamples')
        self.examples_menu.setToolTipsVisible(True)
        self.examples_menu.aboutToShow.connect(self._populate_examples_menu)

        help_menu = menu_bar.addMenu('&Help')
        help_menu.setToolTipsVisible(True)
        help_menu.addAction(self.whats_this_action)
        help_menu.addSeparator()
        help_menu.addAction(self.about_action)

    def _create_status_bar(self):
        """Permanent widgets, left to right: GPU memory gauge + its text, canvas cursor position."""
        self.memory_bar = QtWidgets.QProgressBar()
        self.memory_bar.setRange(0, _MEMORY_STEPS)
        self.memory_bar.setValue(0)
        self.memory_bar.setFixedWidth(110)
        self.memory_bar.setTextVisible(True)
        self.memory_bar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.memory_bar.setFormat(NO_VALUE)
        self.memory_label = QtWidgets.QLabel('GPU: waiting for the worker')
        self.cursor_label = QtWidgets.QLabel('')
        self.cursor_label.setMinimumWidth(280)
        status_bar = self.statusBar()
        status_bar.addPermanentWidget(self.memory_bar)
        status_bar.addPermanentWidget(self.memory_label)
        status_bar.addPermanentWidget(self.cursor_label)

    def _connect_signals(self):
        document = self.document
        document.replaced.connect(self._on_document_replaced)
        document.changed.connect(self._on_document_changed)
        document.issues_invalidated.connect(self._schedule_validation_refresh)
        document.dirty_changed.connect(lambda _dirty: self._update_title())
        document.path_changed.connect(lambda _path: self._update_title())
        document.undo_state_changed.connect(self._update_undo_actions)
        document.notice.connect(self._report_document_notice)

        self.canvas.cursor_info.connect(self.cursor_label.setText)
        self.canvas.tool_changed.connect(self._on_tool_changed)
        self.canvas.edit_refused.connect(self._report_refused_edit)
        self.preview_3d_panel.message.connect(self.show_status)
        self.mesh_store.message.connect(self.show_status)
        self.mesh_store.log.connect(self.log_message)
        self.model_tree.status_message.connect(self._report_refused_edit)
        self.settings_panel.status_message.connect(self._report_refused_edit)
        self.model_tree.itemClicked.connect(self._on_tree_item_clicked)
        self.settings_panel.show_results_requested.connect(self._show_results)
        self.settings_panel.hover_entities.connect(self.canvas.set_hover_entities)
        self.model_tree.node_hovered.connect(self.canvas.set_hover_node)
        self.results_panel.availability_changed.connect(self.model_tree.set_results_available)

        bridge = self.bridge
        bridge.log.connect(self.log_message)
        bridge.built.connect(self._on_built)
        bridge.build_failed.connect(self._on_build_failed)
        bridge.progress.connect(self._on_progress)
        bridge.study_started.connect(self._on_study_started)
        bridge.study_finished.connect(self._on_study_finished)
        bridge.study_failed.connect(self._on_study_failed)
        bridge.sweep_step.connect(self._on_sweep_step)
        bridge.busy_changed.connect(lambda _busy: self._update_run_actions())
        bridge.paused_changed.connect(self._on_paused_changed)
        bridge.task_failed.connect(lambda message: self.log_message('warning', f'Task failed: {message}'))
        bridge.device_info.connect(self._on_device_info)
        bridge.memory_status.connect(self._on_memory_status)

    # ---- small view helpers ----
    def _raise_dock(self, dock):
        dock.show()
        dock.raise_()

    def _on_tree_item_clicked(self, item, _column):
        """The Results node shows the results (also when clicked again); any other node its settings."""
        if decode_node_path(item.data(0, QtCore.Qt.ItemDataRole.UserRole)) == COMPUTED_RESULTS_NODE:
            self._show_results()
        else:
            self._raise_dock(self.settings_dock)

    def _show_bottom_tab(self, widget):
        self._raise_dock(self.bottom_dock)
        self.bottom_tabs.setCurrentWidget(widget)

    def _show_messages_menu(self, position):
        menu = self.messages_view.createStandardContextMenu()
        menu.addSeparator()
        clear_action = menu.addAction('Clear messages')
        clear_action.triggered.connect(lambda _checked=False: self.messages_view.clear())
        menu.exec(self.messages_view.mapToGlobal(position))
        menu.deleteLater()

    def _run_user_operation(self, operation):
        """Run one user-triggered document operation; user errors are reported as refused edits."""
        run_user_operation(operation, self._report_refused_edit)

    def _report_refused_edit(self, message):
        """A refused edit: status bar message plus a lasting warning line in the Messages tab."""
        self.show_status(message)
        self.log_message('warning', message)

    def _report_document_notice(self, message):
        """A side effect of an edit (ModelDocument.notice): status bar plus an info line in the Messages tab."""
        self.show_status(message)
        self.log_message('info', message)

    def _commit_pending_edit(self):
        """Commit a half-typed settings edit (its editor commits on focus-out) before the spec is read."""
        focus_widget = QtWidgets.QApplication.focusWidget()
        if focus_widget is not None and self.settings_panel.isAncestorOf(focus_widget):
            focus_widget.clearFocus()

    def _update_title(self):
        path = self.document.path
        name = os.path.basename(path) if path else 'Untitled'
        self.setWindowTitle(f'{name}[*] - open_sol')
        self.setWindowModified(bool(self.document.dirty))

    def _update_undo_actions(self):
        """Enable undo / redo and name the edit they act on in their text and tooltip."""
        for action, verb, enabled, label, idle_tip in (
                (self.undo_action, 'Undo', self.document.can_undo(), self.document.undo_text(),
                 'Undo the last model change'),
                (self.redo_action, 'Redo', self.document.can_redo(), self.document.redo_text(),
                 'Redo the undone change')):
            action.setEnabled(enabled)
            action.setText(f"&{verb} {label.replace('&', '&&')}" if label else f'&{verb}')   # && = literal &
            tip = f'{verb} {label}' if label else idle_tip
            action.setToolTip(tip)
            action.setStatusTip(tip)

    def _on_tool_changed(self, tool):
        action = self._tool_actions.get(tool)
        if action is not None:
            action.setChecked(True)

    # ---- validation tab ----
    def _on_document_changed(self, section, origin):
        self._schedule_validation_refresh()

    def _schedule_validation_refresh(self):
        if not self._validation_refresh_pending:
            self._validation_refresh_pending = True
            QtCore.QTimer.singleShot(0, self._refresh_validation)

    def _refresh_validation(self):
        """Fill the Validation tab: document.issues() plus the last build's issues while the revision matches."""
        self._validation_refresh_pending = False
        issues = list(self.document.issues())
        if self._build_issue_revision is not None and self._build_issue_revision == self.document.revision:
            known = {_issue_key(issue) for issue in issues}
            issues.extend(issue for issue in self._build_issues if _issue_key(issue) not in known)
        issues.sort(key=lambda issue: ISSUE_LEVELS.index(issue.level))
        shown = tuple((issue.level, tuple(issue.path), issue.message) for issue in issues)
        if shown == self._shown_issue_keys:   # unchanged: keep the items (and the selection)
            return
        self._shown_issue_keys = shown
        self.validation_view.clear()
        icons = {level: self.style().standardIcon(pixmap) for level, pixmap in ISSUE_LEVEL_PIXMAPS.items()}
        for issue in issues:
            item = QtWidgets.QTreeWidgetItem([issue.level, _node_text(issue.path), issue.message])
            item.setIcon(0, icons[issue.level])
            item.setToolTip(2, issue.message)
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, encode_node_path(issue_node_path(issue)))
            self.validation_view.addTopLevelItem(item)
        counts = {level: sum(1 for issue in issues if issue.level == level) for level in ISSUE_LEVELS}
        summary = ', '.join(f'{counts[level]} {level}{"s" if counts[level] != 1 else ""}'
                            for level in ('error', 'warning') if counts[level])
        tab_index = self.bottom_tabs.indexOf(self.validation_view)
        self.bottom_tabs.setTabText(tab_index, f'Validation ({summary})' if summary else 'Validation')

    def _on_validation_double_clicked(self, item, _column):
        path = decode_node_path(item.data(0, QtCore.Qt.ItemDataRole.UserRole))
        if path is not None and self.document.node_exists(path):
            self.document.select(path)
            self._raise_dock(self.settings_dock)
        else:
            self.show_status(f'Node {_node_text(path or ())} no longer exists')

    # ---- build / compute ----
    def _start_job(self, job_kind, label):
        self._job_kind = job_kind
        self._requested_revision = self.document.revision
        self._run_started_at = time.monotonic()
        self._job_log = collections.deque(maxlen=_JOB_LOG_MAX_LINES)
        self._job_model_file = self.document.path   # the file the job's model came from (Save run summary)
        self._build_failed_in_job = False
        self._sweep_position = None
        self._sweep_text = ''
        self._reset_progress(label)

    def _reset_progress(self, label):
        """Empty bars and blank Progress rows (the device memory row is kept); label becomes the status text."""
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat('%p%')
        self.stage_bar.setValue(0)
        for key, value_label in self._progress_fields.items():
            if key != 'gpu':                  # device memory row: kept until the next MemoryStatus
                value_label.setText(NO_VALUE)
        self.build_memory_view.clear()
        self._set_progress_text(label)

    def _show_build_memory(self, record):
        """Fill the build-memory table from a build ledger record (Session.build_memory layout); None clears it."""
        view = self.build_memory_view
        view.clear()
        if not record:
            return
        failed_owner = record.get('failed_owner')
        rows = [(f'{owner} (ran out / failed here)' if owner == failed_owner else owner, format_bytes(size))
                for owner, size in record.get('breakdown') or ()]
        rows.append(('Kept by the model' if failed_owner is None else 'Held at the failure',
                     format_bytes(record.get('total_bytes'))))
        rows.append(('Build peak', format_bytes(record.get('peak_bytes'))))
        rows.append(('Limit', format_bytes(record.get('budget'))))
        estimate = record.get('estimate')
        if estimate:
            for label, key, deviation_key in (('Estimate: kept', 'gpu_steady', 'steady_deviation'),
                                              ('Estimate: build peak', 'gpu_build_peak', 'peak_deviation')):
                deviation = estimate.get(deviation_key)
                deviation_text = '' if deviation is None else f' (measured {deviation:+.0%})'
                rows.append((label, format_bytes(estimate.get(key)) + deviation_text))
            if estimate.get('host'):
                rows.append(('Estimate: processor memory', format_bytes(estimate.get('host'))))
        for stage_text, size_text in rows:
            view.addTopLevelItem(QtWidgets.QTreeWidgetItem([stage_text, size_text]))

    def _on_document_replaced(self):
        """New / opened model: the finished job's progress belongs to the old one (a running job keeps updating)."""
        if not self.bridge.busy:
            self._reset_progress('Idle')

    def _build(self):
        self._commit_pending_edit()
        self._start_job('build', 'Building…')
        self.bridge.build(self.document.spec)

    def _compute(self):
        """Validate first (errors -> Validation tab, no run), then run the study on the worker."""
        self._commit_pending_edit()
        errors = [issue for issue in self.document.issues() if issue.level == 'error']
        if errors:
            self._refresh_validation()
            self._show_bottom_tab(self.validation_view)
            self.show_status(f'{len(errors)} validation error{"s" if len(errors) != 1 else ""}: '
                             'fix them before computing')
            return
        self._start_job('run', 'Starting…')
        self.bridge.run(self.document.spec)

    def _toggle_pause(self):
        if self.bridge.paused:
            self.bridge.resume()
        else:
            self.bridge.pause()

    def _update_run_actions(self):
        busy = bool(self.bridge.busy)
        paused = bool(self.bridge.paused)
        controllable = busy and self._job_kind != 'build'   # a build can be neither paused nor cancelled
        self.build_action.setEnabled(not busy)
        self.compute_action.setEnabled(not busy)
        self.save_run_action.setEnabled(not busy and self.bridge.results_spec is not None
                                        and self.bridge.last_study_summary is not None)
        self.pause_action.setEnabled(controllable)
        self.stop_action.setEnabled(controllable)
        self.pause_action.setText('&Resume' if paused else '&Pause')
        pixmaps = QtWidgets.QStyle.StandardPixmap
        self.pause_action.setIcon(self.style().standardIcon(pixmaps.SP_MediaPlay if paused else pixmaps.SP_MediaPause))

    def _on_paused_changed(self, paused):
        self._update_run_actions()
        if paused:
            self._set_progress_text('Paused')
        elif self.bridge.busy:
            self._set_progress_text('Running')

    def _set_progress_text(self, text):
        self.progress_label.setText(text)
        self.progress_label.setToolTip(text)
        self._progress_fields['status'].setText(text)

    def _on_progress(self, progress):
        """Toolbar bar = the whole run (stages and sweep values); its text names the stage and what is solved now.
        Time remaining: whole sweep, else the current stage of a run with several (later stages are unknown)."""
        fraction = _unit_interval(progress.fraction)
        stages = tuple(progress.stages or ())
        stage_index = progress.stage_index if 0 <= progress.stage_index < len(stages) else None
        stage_name = stages[stage_index] if stage_index is not None else ''
        stage_fraction = fraction if progress.stage_fraction is None else _unit_interval(progress.stage_fraction)
        stage_elapsed = progress.wall_elapsed if progress.stage_elapsed is None else progress.stage_elapsed
        several_stages = len(stages) > 1 and stage_index is not None   # 'Stage i/n' texts need a valid index
        if self._sweep_position is not None:
            index, count = self._sweep_position
            overall = _unit_interval((index + fraction) / max(count, 1))
            elapsed = time.monotonic() - self._run_started_at if self._run_started_at is not None else None
            eta = _eta_text(overall, elapsed)
            eta_text = f'Time remaining {eta}'
        elif several_stages:
            overall = fraction
            eta = _eta_text(stage_fraction, stage_elapsed)
            eta_text = f'{eta} left in this stage'
        else:
            overall = fraction
            eta = _eta_text(fraction, progress.wall_elapsed)
            eta_text = f'Time remaining {eta}'
        message = 'Paused' if self.bridge.paused else (progress.message or stage_name or 'Running')
        stage_text = f'Stage {stage_index + 1}/{len(stages)} {stage_name}' if several_stages else stage_name
        self.progress_bar.setValue(int(round(_PROGRESS_STEPS * overall)))
        self.progress_bar.setFormat(f'Stage {stage_index + 1}/{len(stages)}: %p%' if several_stages else '%p%')
        self.stage_bar.setValue(int(round(_PROGRESS_STEPS * stage_fraction)))
        headline = f'{stage_text}: {message}' if stage_text and message != stage_name else message
        label_parts = [headline, eta_text] + ([self._sweep_text] if self._sweep_text else [])
        self.progress_label.setText(' | '.join(label_parts))
        self.progress_label.setToolTip('\n'.join(label_parts))
        fields = self._progress_fields
        fields['status'].setText(message)
        fields['stage'].setText(f'{stage_text}, {format_duration(stage_elapsed)} so far, '
                                f'{_eta_text(stage_fraction, stage_elapsed)} left' if stage_text else NO_VALUE)
        fields['stages'].setText(NO_VALUE if stage_index is None
                                 else _stages_text(stages, stage_index, stage_fraction))
        if _finite(progress.t_end) and progress.t_end > 0:
            fields['time'].setText(f'{format_quantity(progress.t, "s")} / {format_quantity(progress.t_end, "s")}')
        else:
            fields['time'].setText(format_quantity(progress.t, 's'))
        fields['tick'].setText(NO_VALUE if progress.tick is None else f'{progress.tick:,}')
        fields['speed'].setText(f'{progress.ticks_per_second:.4g} {progress.tick_unit}s/s'
                                if _finite(progress.ticks_per_second) else NO_VALUE)
        fields['elapsed'].setText(format_duration(progress.wall_elapsed))
        fields['eta'].setText(f'{eta} (this stage)' if several_stages and self._sweep_position is None else eta)

    def _on_sweep_step(self, step):
        """SweepStep.index is 0-based; the bar and ETA then cover the whole sweep."""
        self._sweep_position = (int(step.index), int(step.count))
        value_text = f'{step.value:g}' if _finite(step.value) else str(step.value)
        self._sweep_text = f'sweep {step.index + 1}/{step.count}: {step.parameter} = {value_text}'
        self._progress_fields['sweep'].setText(self._sweep_text)
        self.log_message('info', self._sweep_text)

    def _on_study_started(self):
        """Select the tree's Results node (field view on the canvas) and raise the Results panel."""
        self._set_progress_text('Running')
        if self.results_panel.has_results:   # False for a job of a replaced model
            self.document.select(COMPUTED_RESULTS_NODE)
        self._raise_dock(self.results_dock)

    def _on_study_finished(self, event):
        summary = event.summary or {}
        status = summary.get('status', 'completed')
        duration = format_duration(summary.get('wall_time'))
        self.progress_bar.setFormat('%p%')
        if status == 'completed':
            self.progress_bar.setValue(_PROGRESS_STEPS)
            self.stage_bar.setValue(_PROGRESS_STEPS)
            text = f'Completed in {duration}'
        else:
            text = f'{str(status).capitalize()} after {duration}'
        self._sweep_position = None
        self._set_progress_text(text)
        self.log_message('info', study_summary_text(summary, duration))
        derived = summary.get('derived') or {}
        references, errors = summary.get('derived_reference') or {}, summary.get('derived_error') or {}
        if derived:
            self.log_message('info', 'Derived values: ' + ', '.join(
                f'{name} = {value:.6g}' + (f' (reference {references[name]:.6g}, relative error {errors[name]:.6g})'
                                           if name in references else '')
                for name, value in derived.items()))
        self.show_status(f'Study {text.lower()}')

    def _on_study_failed(self, message, traceback_text):
        in_sweep = self._sweep_position is not None
        self._sweep_position = None
        self.progress_bar.setFormat('%p%')   # drop the 'Stage i/n' of the failed run
        if self._build_failed_in_job and not in_sweep:
            return                        # the single build failure was already reported by _on_build_failed
        self._set_progress_text('Failed')
        self.log_message('error', f'Study failed: {message}')
        if traceback_text:
            self.log_message('info', traceback_text)
        self._show_bottom_tab(self.messages_view)
        self.show_status('Study failed: see the Messages tab')

    def _on_built(self, summary):
        self._build_failed_in_job = False     # a sweep value built after a failed one: later failures are new
        z_count = f' × {summary.nz}' if summary.nz > 1 else ''
        parts = [f'{summary.nx} × {summary.ny}{z_count} cells ({int(summary.n_cells):,})']
        if summary.dx_min == summary.dx_max:
            parts.append(f'dx {format_quantity(summary.dx_min, "m")}')
        else:
            parts.append(f'dx {format_quantity(summary.dx_min, "m")} - {format_quantity(summary.dx_max, "m")}')
        if summary.dt is not None:
            parts.append(f'dt {format_quantity(summary.dt, "s")}')
        session_gpu_bytes = summary.session_gpu_bytes   # this build's share of the pool, None without a baseline
        parts.append(f'GPU {format_bytes(summary.gpu_bytes if session_gpu_bytes is None else session_gpu_bytes)}')
        if summary.gpu_peak_bytes is not None:
            parts.append(f'build peak {format_bytes(summary.gpu_peak_bytes)}')
        parts.append(f'built in {summary.build_seconds:.2f} s')
        text = ', '.join(parts)
        self._progress_fields['build'].setText(text)
        self._show_build_memory(None if summary.gpu_peak_bytes is None else {
            'breakdown': summary.gpu_breakdown, 'total_bytes': sum(size for _, size in summary.gpu_breakdown),
            'peak_bytes': summary.gpu_peak_bytes, 'budget': summary.gpu_budget, 'failed_owner': None,
            'estimate': summary.gpu_estimate})
        self.log_message('info', f'Built {summary.title!r}: {text}')
        self._build_issues = [Issue('warning', ('model',), f'build: {warning}') for warning in summary.warnings]
        self._build_issue_revision = self._requested_revision
        self._schedule_validation_refresh()
        if self._job_kind == 'build':
            self.progress_bar.setValue(_PROGRESS_STEPS)
            self._set_progress_text('Built')
        self.show_status('Build complete')

    def _on_build_failed(self, event):
        self._build_failed_in_job = True
        self._set_progress_text('Build failed')
        self._show_build_memory(event.build_memory)
        self.log_message('error', f'Build failed: {event.message}')
        issues = list(event.issues or [])
        if issues:
            self._build_issues = issues
            self._build_issue_revision = self._requested_revision
            self._schedule_validation_refresh()
            self._show_bottom_tab(self.validation_view)
        else:
            if event.traceback:
                self.log_message('info', event.traceback)
            self._show_bottom_tab(self.messages_view)
        self.show_status('Build failed')

    # ---- GPU memory ----
    def _on_device_info(self, device):
        """Gauge tooltip header from the worker's start-up DeviceInfo (the worker itself logs device and budget); kind
        'cpu': no usable graphics card, the gauge shows the computer's memory."""
        if getattr(device, 'kind', 'gpu') == 'cpu':
            self._gpu_device_text = (f'{device.name}: no usable graphics card, models run on the processor; '
                                     f'{format_bytes(device.total_bytes)} memory, {format_bytes(device.free_bytes)} '
                                     'available at start-up')
        else:
            self._gpu_device_text = (f'{device.name} (graphics card {device.device_id}): '
                                     f'{format_bytes(device.total_bytes)} total, {format_bytes(device.free_bytes)} '
                                     'free at start-up')
        self._update_memory_tooltip(self.bridge.last_memory_status)

    def _on_memory_status(self, status):
        """Status-bar gauge and Progress row: pool used against the budget (the device total before one is set); the
        budget and the available host RAM also go to the document for the validation memory checks."""
        self.document.set_memory_budget(status.budget, status.host_available)
        on_cpu = getattr(status, 'kind', 'gpu') == 'cpu'   # host RAM in use (all programs) against the installed RAM
        used = status.device_total - status.device_free if on_cpu else status.pool_used
        limit = status.device_total if on_cpu or not status.budget else status.budget
        fraction = (used / limit if _finite(used) and _finite(limit) and limit > 0
                    else 0.0)
        self.memory_bar.setValue(int(round(_MEMORY_STEPS * _unit_interval(fraction))))
        self.memory_bar.setFormat(f'{100 * fraction:.0f}%')
        style = _MEMORY_WARNING_STYLE if fraction > _MEMORY_WARNING_FRACTION else ''
        if self.memory_bar.styleSheet() != style:
            self.memory_bar.setStyleSheet(style)
        if on_cpu:
            text = (f'CPU memory {format_bytes(used)} / {format_bytes(limit)} '
                    f'(available {format_bytes(status.device_free)})')
        else:
            text = (f'GPU {format_bytes(status.pool_used)} / {format_bytes(limit)} '
                    f'(device {format_bytes(status.device_total)}; other {format_bytes(status.other)})')
        self.memory_label.setText(text)
        self._progress_fields['gpu'].setText(text)
        self._update_memory_tooltip(status)

    def _update_memory_tooltip(self, status):
        """Gauge tooltip: the device header, then the counters of `status` (None before the first MemoryStatus)."""
        lines = [self._gpu_device_text] if self._gpu_device_text else []
        if status is not None and getattr(status, 'kind', 'gpu') == 'cpu':
            lines += [f'In use by all programs: {format_bytes(status.device_total - status.device_free)}',
                      f'Available: {format_bytes(status.device_free)} of {format_bytes(status.device_total)}']
        elif status is not None:
            budget_text = 'not set' if status.budget is None else format_bytes(status.budget)
            lines += [f'In use: {format_bytes(status.pool_used)}',
                      f'Reserved: {format_bytes(status.pool_reserved)}',
                      f'Free on the card: {format_bytes(status.device_free)} of {format_bytes(status.device_total)}',
                      f'Limit: {budget_text}',
                      f'Other: {format_bytes(status.other)}']
            if status.host_available is not None:
                lines.append(f'Available processor memory: {format_bytes(status.host_available)}')
        tooltip = rich_tooltip('\n'.join(lines), _MEMORY_LEGEND if status is not None else '')
        self.memory_bar.setToolTip(tooltip)
        self.memory_label.setToolTip(tooltip)

    def _show_results(self):
        """Select the tree's Results node and re-show the displayed field; before the first one, fetch the Results
        panel's selected field."""
        if self.results_panel.has_results:
            self.document.select(COMPUTED_RESULTS_NODE)
        if self.canvas.current_field is None and self.bridge.results_spec is not None:
            self.results_panel.show_selected_field()
        else:
            self.canvas.show_field()              # canvas banner when no field was ever shown
        self._raise_dock(self.results_dock)

    def _show_about(self):
        QtWidgets.QMessageBox.about(
            self, 'About open_sol',
            f'<b>open_sol {__version__}</b><p>GPU-accelerated 2D and 3D multiphysics (electromagnetics, heat '
            'transfer, electric currents, mechanics, acoustics, fluid flow and more) built from a declarative model; '
            'one worker thread owns every GPU call, and models also run on the processor.</p>'
            '<p>Results are not saved with the model file.</p>')

    # ---- files ----
    def _confirm_discard_changes(self) -> bool:
        """True when the model may be replaced (clean, saved, or the user discarded the changes)."""
        self._commit_pending_edit()
        if not self.document.dirty:
            return True
        buttons = QtWidgets.QMessageBox.StandardButton
        answer = QtWidgets.QMessageBox.question(
            self, 'Unsaved changes', 'The model has unsaved changes. Save them first?',
            buttons.Save | buttons.Discard | buttons.Cancel, buttons.Save)
        if answer == buttons.Save:
            return self._save()
        return answer == buttons.Discard

    def _new_model(self, template_key):
        if not self._confirm_discard_changes():
            return
        try:
            self.document.new_model(template_key)
        except USER_ERRORS as exc:
            self._report_file_failure('New model failed', error_message(exc))

    def _report_file_failure(self, title, text):
        """A failed new / open / save: error line in the Messages tab, then the dialog."""
        self.log_message('error', f"{title}: {text.replace(chr(10), ' ')}")
        QtWidgets.QMessageBox.critical(self, title, text)

    def _open_dialog(self):
        if not self._confirm_discard_changes():
            return
        path, _selected_filter = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open model', start_directory(self.document), 'Model files (*.json);;All files (*)')
        if path:
            self._open_path(path)

    def _open_path(self, path):
        try:
            self.document.open(path)
        except (OSError, ValueError, KeyError) as exc:
            self._report_file_failure('Open failed', f'Could not open {path}:\n{error_message(exc)}')
            if not os.path.exists(path):
                self._set_recent_files([recent for recent in self._recent_files() if recent != path])
            return
        self._add_recent_file(path)
        remember_directory(path)
        self.show_status(f'Opened {path}')

    def _open_recent(self, path):
        if self._confirm_discard_changes():
            self._open_path(path)

    def _save(self) -> bool:
        if not self.document.path:
            return self._save_as()
        return self._save_to(self.document.path)

    def _save_as(self) -> bool:
        path = ask_save_path(self, self.document, 'Save model', os.path.basename(self.document.path or 'model.json'),
                             (('Model files', '.json'),))
        return self._save_to(path) if path else False

    def _save_to(self, path) -> bool:
        self._commit_pending_edit()
        try:
            self.document.save(path)
        except (OSError, ValueError) as exc:
            self._report_file_failure('Save failed', f'Could not save {path}:\n{error_message(exc)}')
            return False
        self._add_recent_file(path)
        remember_directory(path)
        self.show_status(f'Saved {path} (results are not saved with the model: File > Save run)')
        return True

    def _save_run(self):
        """Write the last computed run into a chosen folder (api.save_session_run on the worker: the files of the
        command-line `run --out`); the session stays open."""
        bridge = self.bridge
        spec, build_summary, study_summary = bridge.results_spec, bridge.last_build_summary, bridge.last_study_summary
        if bridge.busy or spec is None or study_summary is None:
            self.show_status('No finished run to save: compute first')
            return
        model_file = self._job_model_file   # the run's model file, not the document opened since
        stem = os.path.splitext(os.path.basename(model_file or 'model.json'))[0]
        directory = ask_run_directory(self, self.document, f'{stem}_out')
        if not directory:
            return
        log_entries = list(self._job_log)

        def save(session):
            from ..api import save_session_run   # worker thread: fetches the fields and results from the session
            return save_session_run(directory, spec, session, build_summary, study_summary, log_entries, model_file)

        def saved(path):
            remember_directory(path)
            self.show_status(f'Saved the run to {path}')
            self.log_message('info', f'Run saved to {path} (summary.json, log.txt, model.json, results.npz)')

        def failed(message):
            self._report_file_failure('Save run failed', f'Could not save the run to {directory}:\n{message}')

        self.show_status('Saving the run…')
        bridge.call(save, on_result=saved, on_error=failed)

    def _export_field_data(self):
        """Raise the results dock (where the outcome is reported) and export the displayed field from there."""
        self._raise_dock(self.results_dock)
        self.results_panel.export_field_data()

    def _export_image(self):
        """Raise the results dock (where the outcome is reported) and export the canvas figure from there."""
        self._raise_dock(self.results_dock)
        self.results_panel.export_image()

    def _populate_add_object_menu(self):
        """Geometry > Add object: the kinds of the model's dimension (box / cylinder_z / sphere only in 3D), as the
        model tree (composite shapes after a separator)."""
        self.add_object_menu.clear()
        model_dimensions = 3 if self.document.spec.is_3d else 2
        kinds = [kind for kind in GEOMETRY_KINDS.values() if model_dimensions in kind.dimensions]
        ordered_kinds = [kind for kind in kinds if not kind.composite] + [kind for kind in kinds if kind.composite]
        for index, kind in enumerate(ordered_kinds):
            if kind.composite and index > 0 and not ordered_kinds[index - 1].composite:
                self.add_object_menu.addSeparator()
            action = self.add_object_menu.addAction(kind.label)
            if kind.summary:
                action.setToolTip(rich_tooltip(kind.summary))
            action.triggered.connect(lambda _checked=False, kind_name=kind.name: self._run_user_operation(
                partial(self.document.add_geometry, kind_name)))

    # ---- examples ----
    def _populate_examples_menu(self):
        """List the example folders, one submenu each (multiphysics last, after a separator); a submenu scans only
        its own folder, when shown."""
        for submenu in self.examples_menu.findChildren(QtWidgets.QMenu):   # clear() keeps addMenu's QMenu objects
            submenu.deleteLater()
        self.examples_menu.clear()
        folders = self._example_catalog.folders()
        for folder_name, label in folders:
            if folder_name == MULTIPHYSICS_FOLDER and self.examples_menu.actions():
                self.examples_menu.addSeparator()
            folder_menu = self.examples_menu.addMenu(label.replace('&', '&&'))
            folder_menu.setToolTipsVisible(True)
            folder_menu.aboutToShow.connect(partial(self._populate_example_folder_menu, folder_menu, folder_name))
        if not folders:
            placeholder = self.examples_menu.addAction('(no example folders)')
            placeholder.setToolTip(f'Subfolders of {self._example_catalog.directory} with model files (*.json) '
                                   'are listed here.')
            placeholder.setEnabled(False)

    def _populate_example_folder_menu(self, folder_menu, folder_name):
        """Scan one example folder into its submenu, a folded submenu per physics combination when there are several
        (single physics first, a separator before the combinations); unreadable files are logged, none is validated."""
        examples, newly_unreadable = self._example_catalog.scan(folder_name)
        for relative_path, reason in newly_unreadable:
            self.log_message('info', f'Example {relative_path} is not listed in the Examples menu: {reason}')
        for submenu in folder_menu.findChildren(QtWidgets.QMenu):   # clear() keeps addMenu's QMenu objects
            submenu.deleteLater()
        folder_menu.clear()
        several_groups = len({example.group for example in examples}) > 1
        group, group_menu, previous_n_types = None, folder_menu, 0
        for example in examples:
            if several_groups and example.group != group:
                if previous_n_types == 1 and example.n_physics_types > 1:
                    folder_menu.addSeparator()
                group, previous_n_types = example.group, example.n_physics_types
                group_menu = folder_menu.addMenu(group.replace('&', '&&'))
                group_menu.setToolTipsVisible(True)
            action = group_menu.addAction(example.title.replace('&', '&&'))
            action.setToolTip(example.tooltip())
            action.setStatusTip(f'Open the example {example.relative_path} as an untitled model')
            action.triggered.connect(lambda _checked=False, path=example.path: self._open_example(path))
        if not examples:
            placeholder = folder_menu.addAction('(no examples)')
            placeholder.setToolTip(f'Model files (*.json) in {self._example_catalog.directory / folder_name} are '
                                   'listed here; they are checked when opened.')
            placeholder.setEnabled(False)

    def _open_example(self, path):
        if not self._confirm_discard_changes():
            return
        try:
            self.document.new_from_file(path)
        except (OSError, ValueError, KeyError) as exc:
            self._report_file_failure('Open example failed', f'Could not open {path}:\n{error_message(exc)}')
            return
        self.show_status(f'Opened example {path} as an untitled model (Save As keeps your changes)')

    # ---- recent files ----
    def _recent_files(self) -> list:
        value = self._settings.value('recent_files', [])
        if not value:
            return []
        if isinstance(value, str):
            return [value]
        return [str(path) for path in value]

    def _set_recent_files(self, paths):
        self._settings.setValue('recent_files', list(paths[:_MAX_RECENT_FILES]))

    def _add_recent_file(self, path):
        path = os.path.abspath(path)
        others = [recent for recent in self._recent_files() if os.path.normcase(recent) != os.path.normcase(path)]
        self._set_recent_files([path] + others)

    def _populate_recent_menu(self):
        self.recent_menu.clear()
        recent_files = self._recent_files()
        for index, path in enumerate(recent_files, start=1):
            shown_path = path.replace('&', '&&')       # a literal '&', not a mnemonic marker
            action = self.recent_menu.addAction(f'&{index} {shown_path}' if index < 10 else shown_path)
            action.triggered.connect(lambda _checked=False, path=path: self._open_recent(path))
        if not recent_files:
            self.recent_menu.addAction('(no recent files)').setEnabled(False)
            return
        self.recent_menu.addSeparator()
        clear_action = self.recent_menu.addAction('Clear list')
        clear_action.triggered.connect(lambda _checked=False: self._set_recent_files([]))

    # ---- settings / shutdown ----
    def _restore_settings(self):
        geometry = self._settings.value('main_window/geometry')
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = self._settings.value('main_window/state')
        if state is not None:
            self.restoreState(state)

    def _save_settings(self):
        self._settings.setValue('main_window/geometry', self.saveGeometry())
        self._settings.setValue('main_window/state', self.saveState())
        self._settings.sync()

    def _force_quit(self):
        self._save_settings()
        os._exit(0)

    def closeEvent(self, event):
        """Offer to save, stop the study and shut the worker down (waiting dialog with Force quit if it hangs)."""
        if not self._worker_stopped:
            if not self._confirm_discard_changes():
                event.ignore()
                return
            self.bridge.stop()
            if not self.bridge.shutdown(timeout=0.1):   # short join: a busy worker gets the dialog, not a frozen GUI
                _WorkerWaitDialog(self, partial(self.bridge.shutdown, timeout=0.05), self._force_quit).exec()
            self._worker_stopped = True
        self._save_settings()
        event.accept()
