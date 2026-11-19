"""GUI entry point: Qt application set-up, uncaught exceptions routed to the Messages tab, and the
document / worker bridge / main window wiring."""
import argparse
import os
import sys
import threading
import traceback

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from ..model.expressions import error_message
from ..model.schema import get_registry
from ..model.templates import make_template
from .document import ModelDocument
from .main_window import SETTINGS_APPLICATION, SETTINGS_ORGANIZATION, MainWindow, window_copy_directory
from .worker_bridge import WorkerBridge

DEFAULT_TEMPLATE = 'dielectric_cylinder'
# Light Fusion colours per palette role: (active / inactive, disabled)
_LIGHT_PALETTE = {
    'Window': ('#efefef', '#efefef'), 'WindowText': ('#000000', '#9e9e9e'),
    'Base': ('#ffffff', '#efefef'), 'AlternateBase': ('#f7f7f7', '#f7f7f7'),
    'ToolTipBase': ('#ffffdc', '#ffffdc'), 'ToolTipText': ('#000000', '#000000'),
    'PlaceholderText': ('#7f7f7f', '#bebebe'), 'Text': ('#000000', '#bebebe'),
    'Button': ('#efefef', '#efefef'), 'ButtonText': ('#000000', '#9e9e9e'),
    'BrightText': ('#ffffff', '#ffffff'), 'Light': ('#ffffff', '#ffffff'),
    'Midlight': ('#f7f7f7', '#f7f7f7'), 'Mid': ('#b8b8b8', '#b8b8b8'),
    'Dark': ('#9f9f9f', '#bebebe'), 'Shadow': ('#767676', '#b1b1b1'),
    'Highlight': ('#308cc6', '#918d7e'), 'HighlightedText': ('#ffffff', '#ffffff'),
    'Link': ('#0000ff', '#0000ff'), 'LinkVisited': ('#ff00ff', '#ff00ff'),
}


def _force_light_theme(app):
    """Light palette whatever the OS colour scheme: pyqtgraph's white background and hard-coded colours assume it."""
    style_hints = app.styleHints()
    if hasattr(style_hints, 'setColorScheme'):          # Qt >= 6.8; also keeps native title bars light
        style_hints.setColorScheme(QtCore.Qt.ColorScheme.Light)
    palette = QtGui.QPalette()
    groups = QtGui.QPalette.ColorGroup
    for role_name, (normal_color, disabled_color) in _LIGHT_PALETTE.items():
        role = getattr(QtGui.QPalette.ColorRole, role_name)
        for group, color in ((groups.Active, normal_color), (groups.Inactive, normal_color),
                             (groups.Disabled, disabled_color)):
            palette.setColor(group, role, QtGui.QColor(color))
    app.setPalette(palette)


class _ExceptionRelay(QtCore.QObject):
    """Carries uncaught-exception text from any thread into the GUI thread (queued, also from the GUI thread, so
    widgets are never touched inside the event that raised) and hands it to `report` once that is set."""
    exception_raised = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.report = None              # GUI-thread callable(text); the event loop delivers only after it is set
        self.exception_raised.connect(self._deliver, QtCore.Qt.ConnectionType.QueuedConnection)

    @QtCore.Slot(str)
    def _deliver(self, text):
        if self.report is not None:
            self.report(text)


def _install_exception_hooks(relay):
    """Route sys/threading uncaught exceptions to stderr and relay; returns a function restoring the old hooks."""
    previous_sys_hook = sys.excepthook
    previous_thread_hook = threading.excepthook

    def report(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            previous_sys_hook(exc_type, exc_value, exc_traceback)
            return
        text = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        if sys.__stderr__ is not None:
            sys.__stderr__.write(text)
        try:
            relay.exception_raised.emit(f'Uncaught exception:\n{text}')
        except RuntimeError:
            pass                        # relay already deleted at interpreter shutdown; stderr has the text

    def report_thread(hook_args):
        if hook_args.exc_type is SystemExit:
            return
        report(hook_args.exc_type, hook_args.exc_value, hook_args.exc_traceback)

    sys.excepthook = report
    threading.excepthook = report_thread

    def restore():
        sys.excepthook = previous_sys_hook
        threading.excepthook = previous_thread_hook

    return restore


def main(argv=None) -> int:
    """Run the GUI; argv excludes the program name (default sys.argv[1:]). Returns the Qt exit code."""
    sys.setswitchinterval(5e-4)          # the GUI thread gets the GIL back quickly from the worker thread
    parser = argparse.ArgumentParser(prog='python -m open_sol.gui',
                                     description='open_sol model editor and GPU solver')
    parser.add_argument('model', nargs='?', default=None,
                        help=f'model file (.json) to open; default: the {DEFAULT_TEMPLATE} template')
    # internal, from MainWindow.launch_window: Duplicate in new window's copy (read as untitled, then deleted) and its
    # display name; --cascade = started by another window
    parser.add_argument('--copy', default=None, help=argparse.SUPPRESS)
    parser.add_argument('--copy-name', default='another window', help=argparse.SUPPRESS)
    parser.add_argument('--cascade', action='store_true', help=argparse.SUPPRESS)
    arguments = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    if arguments.copy is not None:
        if arguments.model:
            parser.error('give a model file or --copy, not both')
        if os.path.dirname(os.path.abspath(arguments.copy)) != os.path.abspath(window_copy_directory()):
            parser.error(f'--copy reads (and deletes) only files in {window_copy_directory()}')

    registry = get_registry()
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([sys.argv[0] if sys.argv and sys.argv[0] else 'open_sol'])
    app.setStyle('Fusion')
    _force_light_theme(app)                  # after setStyle, which resets the palette
    app.setOrganizationName(SETTINGS_ORGANIZATION)
    app.setApplicationName(SETTINGS_APPLICATION)
    pg.setConfigOptions(imageAxisOrder='col-major', antialias=False, background='w', foreground='k')

    relay = _ExceptionRelay()
    restore_hooks = _install_exception_hooks(relay)
    try:
        document = ModelDocument(registry, make_template(DEFAULT_TEMPLATE))
        open_error = None
        copy_status = None
        if arguments.copy is not None:
            try:
                document.new_from_file(arguments.copy)
            except (OSError, ValueError, KeyError) as exc:   # the copy is kept for a look at what failed
                open_error = (f'Could not read the copy of {arguments.copy_name} ({arguments.copy}): '
                              f'{error_message(exc)}\nShowing the {DEFAULT_TEMPLATE} template.')
            else:
                os.remove(arguments.copy)
                copy_status = f'Copy of {arguments.copy_name}, untitled: Save As keeps it'
        elif arguments.model:
            try:
                document.open(arguments.model)
            except (OSError, ValueError, KeyError) as exc:
                open_error = (f'Could not open {arguments.model}: {error_message(exc)}\n'
                              f'Showing the {DEFAULT_TEMPLATE} template.')
        bridge = WorkerBridge()
        try:
            window = MainWindow(document, bridge, registry)
            relay.report = window.report_uncaught_exception   # also gets exceptions raised while it was built
            if arguments.cascade:
                window.cascade()
            window.show()
            if copy_status is not None:
                window.show_status(copy_status)
            if open_error is not None:
                window.log_message('error', open_error)
                QtWidgets.QMessageBox.warning(window, 'Open failed', open_error)
            return app.exec()
        finally:
            bridge.shutdown()
    finally:
        restore_hooks()
