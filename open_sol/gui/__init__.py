"""Qt GUI (PySide6 + pyqtgraph) for open_sol models: ``python -m open_sol.gui [model.json]``.
PySide6 is imported here first so pyqtgraph always binds to it; GPU work never runs on the GUI thread."""
from PySide6 import QtCore  # noqa: F401  (must be imported before any pyqtgraph import)
