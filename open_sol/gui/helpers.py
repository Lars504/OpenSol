"""Small GUI helpers shared by main_window, model_tree, settings_panel, canvas and results: user-error handling,
file-dialog folders, the study-finished text and the mesh colours / texts (S7.2: which mesh a physics or field uses)."""
import html
import math
import numbers
import os
import re

from PySide6 import QtCore, QtGui, QtWidgets

from ..model.expressions import ExpressionError, error_message, referenced_names
from ..model.schema import MESH_PARAMS, output_field_names
from ..model.study import tick_count_text
from .forms import NO_VALUE

USER_ERRORS = (ValueError, KeyError, ExpressionError)
_LAST_DIRECTORY_KEY = 'last_directory'   # QSettings key (app.main sets the organisation / application names)
_INVALID_NAME_CHARACTERS = '<>:"/\\|?*'   # not allowed in a Windows file name (also no path separators)
# one fixed colour per model mesh (its index in ModelSpec.meshes): tree swatches, canvas badge, results labels
MESH_COLORS = ('#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e', '#17becf', '#8c564b', '#e377c2')
_UNKNOWN_MESH_COLOR = '#808080'
_MESH_MODE_TEXTS = {mode: label.lower() for mode, label in
                    next(param.choice_labels for param in MESH_PARAMS if param.name == 'mode')}
_MESH_KIND_TEXTS = {'tensor': 'rectangular cells', 'triangles': 'triangles', 'tetrahedra': 'tetrahedra'}


def run_user_operation(operation, report, label=''):
    """Run one user-triggered document operation; a user error goes to report('<label>: <message>', as the settings
    panel's refused edits; the bare message without a label) instead of raising."""
    try:
        operation()
    except USER_ERRORS as exc:
        report(f'{label}: {error_message(exc)}' if label else error_message(exc))


def start_directory(document) -> str:
    """Folder of the model file, else the last folder used for files (QSettings)."""
    if document.path:
        return os.path.dirname(os.path.abspath(document.path))
    return str(QtCore.QSettings().value(_LAST_DIRECTORY_KEY, '') or '')


def remember_directory(path):
    QtCore.QSettings().setValue(_LAST_DIRECTORY_KEY, os.path.dirname(os.path.abspath(path)))


def ask_save_path(parent, document, title, default_name, file_types):
    """Save-dialog path ('' when cancelled) for file_types ((label, suffix), ...); the selected type's suffix is added
    when none was typed, by the dialog (default suffix), so its overwrite question sees the final name."""
    filters = [f'{label} (*{suffix})' for label, suffix in file_types]
    suffix_by_filter = dict(zip(filters, (suffix for _, suffix in file_types)))
    dialog = QtWidgets.QFileDialog(parent, title, os.path.join(start_directory(document), default_name),
                                   ';;'.join(filters))
    dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)

    def set_default_suffix(selected_filter):
        dialog.setDefaultSuffix(suffix_by_filter.get(selected_filter, file_types[0][1]).lstrip('.'))
    dialog.filterSelected.connect(set_default_suffix)
    set_default_suffix(filters[0])
    accepted = dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted
    path = dialog.selectedFiles()[0] if accepted and dialog.selectedFiles() else ''
    selected_filter = dialog.selectedNameFilter()
    dialog.deleteLater()
    if path and not os.path.splitext(path)[1]:   # a dialog that ignored the default suffix
        path += suffix_by_filter.get(selected_filter, file_types[0][1])
    return path


def next_free_name(folder, name) -> str:
    """name, else the first of name_2, name_3, ... (a trailing _<n> counts on from n) not present in folder."""
    match = re.fullmatch(r'(.+)_(\d+)', name)
    base, number = (match.group(1), int(match.group(2))) if match else (name, 1)
    candidate = name
    while os.path.exists(os.path.join(folder, candidate)):
        number += 1
        candidate = f'{base}_{number}'
    return candidate


def _folder_name_problem(name) -> str:
    """Why name cannot be one new folder name ('' when it can)."""
    if not name:
        return 'Enter a folder name.'
    if name in ('.', '..') or any(character in _INVALID_NAME_CHARACTERS for character in name):
        return f"'{name}' is not a folder name: no {' '.join(_INVALID_NAME_CHARACTERS)} or '.' / '..'."
    if name != name.rstrip(' .'):
        return f"'{name}' ends in a space or dot, which Windows drops."
    return ''


def _ask_existing_folder(parent, target, free_name):
    """Overwrite / next-free-name question for a non-empty existing folder: the folder to save into, or None (back)."""
    box = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Icon.Question, 'Folder exists',
                                f'{target} already exists and is not empty.', parent=parent)
    box.setInformativeText('Overwrite writes the run files (summary.json, log.txt, model.json, results.npz, ...) over '
                           'the ones in it; other files stay.')
    overwrite_button = box.addButton('Overwrite', QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
    free_button = box.addButton(f"Save as '{free_name}'", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.setDefaultButton(free_button)
    box.exec()
    clicked = box.clickedButton()
    box.deleteLater()
    if clicked is overwrite_button:
        return target
    if clicked is free_button:
        return os.path.join(os.path.dirname(target), free_name)
    return None


def ask_run_directory(parent, document, default_name) -> str:
    """Run-folder path ('' when cancelled) from a 'Save in' folder + new folder name dialog; a non-empty existing folder
    asks to overwrite or take the next free name (next_free_name)."""
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle('Save run')
    dialog.setMinimumWidth(560)
    parent_edit = QtWidgets.QLineEdit(os.path.normpath(start_directory(document) or os.getcwd()))
    browse_button = QtWidgets.QPushButton('Browse…')
    name_edit = QtWidgets.QLineEdit(default_name)
    target_label = QtWidgets.QLabel()
    target_label.setWordWrap(True)
    buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Save
                                         | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
    folder_row = QtWidgets.QHBoxLayout()
    folder_row.addWidget(parent_edit, 1)
    folder_row.addWidget(browse_button)
    form = QtWidgets.QFormLayout()
    form.addRow('Save in', folder_row)
    form.addRow('Folder name', name_edit)
    layout = QtWidgets.QVBoxLayout(dialog)
    layout.addLayout(form)
    layout.addWidget(target_label)
    layout.addWidget(buttons)
    chosen = []

    def update_target():
        target = os.path.join(parent_edit.text().strip(), name_edit.text().strip())
        note = ' (exists: you will be asked before writing into it)' if os.path.isdir(target) else ''
        target_label.setText(f'Run files go to {target}{note}')

    def browse():
        folder = QtWidgets.QFileDialog.getExistingDirectory(dialog, 'Save the run folder in', parent_edit.text())
        if folder:
            parent_edit.setText(os.path.normpath(folder))

    def try_accept():
        folder, name = os.path.abspath(parent_edit.text().strip() or '.'), name_edit.text().strip()
        problem = _folder_name_problem(name) or ('' if os.path.isdir(folder) else f'{folder} is not a folder.')
        target = os.path.join(folder, name)
        if not problem and os.path.isfile(target):
            problem = f'{target} is a file, not a folder.'
        if not problem and os.path.isdir(target):
            try:
                occupied = bool(os.listdir(target))
            except OSError as exc:
                problem = f'Cannot read {target}: {exc}'
            else:
                target = _ask_existing_folder(dialog, target, next_free_name(folder, name)) if occupied else target
                if target is None:   # back to the dialog
                    return
        if problem:
            QtWidgets.QMessageBox.warning(dialog, 'Save run', problem)
            return
        chosen.append(target)
        dialog.accept()

    parent_edit.textChanged.connect(update_target)
    name_edit.textChanged.connect(update_target)
    browse_button.clicked.connect(browse)
    buttons.accepted.connect(try_accept)
    buttons.rejected.connect(dialog.reject)
    update_target()
    name_edit.setFocus()
    name_edit.selectAll()
    dialog.exec()
    dialog.deleteLater()
    return chosen[0] if chosen else ''


def format_duration(seconds) -> str:
    """Wall-clock duration as '12.3 s', '4 min 05 s' or '2 h 07 min' (log, toolbar, Results line); NO_VALUE when not
    a finite number."""
    if not isinstance(seconds, numbers.Real) or isinstance(seconds, bool) or not math.isfinite(seconds):
        return NO_VALUE
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f'{seconds:.1f} s'
    whole_seconds = int(round(seconds))
    if whole_seconds < 3600:
        return f'{whole_seconds // 60} min {whole_seconds % 60:02d} s'
    return f'{whole_seconds // 3600} h {(whole_seconds % 3600) // 60:02d} min'


def study_summary_text(summary, duration_text) -> str:
    """'Study completed: 120 steps in 4.2 s', counted in the summary's tick_unit (as the run log)."""
    return f"Study {summary.get('status', 'completed')}: {tick_count_text(summary)} in {duration_text}"


# ---- meshes (S7.2): colours and texts naming the mesh of a physics, coupling or field ----
def several_meshes(spec) -> bool:
    """True when the model has more than one mesh (labels then name the mesh everywhere)."""
    return len(spec.meshes) > 1


def mesh_color(spec, mesh_name) -> str:
    """Fixed colour of a model mesh: MESH_COLORS by its index in spec.meshes; grey for an unknown name."""
    names = [mesh_spec.name for mesh_spec in spec.meshes]
    if mesh_name not in names:
        return _UNKNOWN_MESH_COLOR
    return MESH_COLORS[names.index(mesh_name) % len(MESH_COLORS)]


def mesh_kind_text(mesh_spec) -> str:
    """'rectangular cells, automatic' / 'triangles, uniform' ... of a MeshSpec (kind, mode)."""
    kind = _MESH_KIND_TEXTS.get(mesh_spec.kind, str(mesh_spec.kind))
    return f'{kind}, {_MESH_MODE_TEXTS.get(mesh_spec.mode, str(mesh_spec.mode))}'


def mesh_kind_shown(spec, kind) -> bool:
    """Whether the GUI offers mesh `kind` in this model at all: triangles only in 2D, tetrahedra only in 3D."""
    if kind == 'triangles':
        return spec.dimensions == 2
    if kind == 'tetrahedra':
        return spec.dimensions == 3
    return True


def mesh_kind_problem(spec, kind) -> str:
    """'' when a mesh of `kind` fits the model, else why not (the disabled Add mesh entries' tooltip)."""
    cartesian = spec.domain.coordinates == 'cartesian'
    if kind == 'triangles' and not (spec.dimensions == 2 and cartesian):
        return 'Triangle meshes work in 2D models with a rectangular (x, y) grid only.'
    if kind == 'tetrahedra' and not (spec.dimensions == 3 and cartesian):
        return 'Tetrahedron meshes work in 3D models with a rectangular (x, y, z) grid only.'
    return ''


def mesh_size_text(estimate) -> str:
    """'412 × 210 cells' (3D: × nz) of a tensor estimate, '~18,340 elements' of a triangle / tetrahedron one; '' for
    None."""
    if estimate is None:
        return ''
    if estimate.kind != 'tensor':
        return f'~{int(estimate.n_elements):,} elements'
    z_count = f' × {estimate.nz}' if estimate.dz_cells is not None else ''
    return f'{estimate.nx} × {estimate.ny}{z_count} cells'


def mesh_summary_text(mesh_spec, estimate) -> str:
    """'rectangular cells, automatic, 412 × 210 cells' (no size without an estimate)."""
    return ', '.join(text for text in (mesh_kind_text(mesh_spec), mesh_size_text(estimate)) if text)


def mesh_swatch_html(color) -> str:
    """Rich-text colour swatch of a mesh (legends, labels)."""
    return f'<span style="background-color:{color}">&nbsp;&nbsp;&nbsp;</span>'


def mesh_badge_html(spec, mesh_name, detail='') -> str:
    """Rich-text badge: swatch + "Mesh 'name'" in the mesh colour, then the escaped detail text."""
    color = mesh_color(spec, mesh_name)
    tail = f' – {html.escape(detail)}' if detail else ''
    return (f'{mesh_swatch_html(color)} <b style="color:{color}">Mesh &#39;{html.escape(str(mesh_name))}&#39;</b>'
            f'{tail}')


def mesh_swatch_icon(color, overlay=None, size=16) -> QtGui.QIcon:
    """Square icon in a mesh colour; overlay (a QIcon, e.g. an issue level) drawn in its lower-right corner."""
    pixmap = QtGui.QPixmap(size, size)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    try:
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor(color).darker(140)))
        painter.setBrush(QtGui.QColor(color))
        painter.drawRoundedRect(QtCore.QRectF(1.5, 1.5, size - 3, size - 3), 3, 3)
        if overlay is not None:
            corner = size * 5 // 8
            painter.drawPixmap(size - corner, size - corner, overlay.pixmap(corner, corner))
    finally:
        painter.end()
    return QtGui.QIcon(pixmap)


def physics_mesh_name(spec, physics) -> str:
    """Name of the mesh a physics runs on; '' when the physics or its mesh is unknown (validation reports it)."""
    try:
        return spec.mesh_of(physics).name
    except KeyError:
        return ''


def field_mesh_name(field_data, spec) -> str:
    """Mesh a FieldData lives on: its own `mesh` (stage P3), else its physics' mesh in spec (the built spec)."""
    return getattr(field_data, 'mesh', '') or physics_mesh_name(spec, field_data.physics)


def coupling_mesh_text(spec, registry, coupling) -> str:
    """Text like 'em_mesh' → 'heat_mesh' (conservative transfer) of a coupling across meshes, both on mesh 'mesh'
    when it stays on one; '' when a physics or mesh is unknown."""
    source, target = physics_mesh_name(spec, coupling.source), physics_mesh_name(spec, coupling.target)
    if not source or not target:
        return ''
    if source == target:
        return f"both on mesh '{source}'"
    try:
        coupling_type = registry.coupling_type(coupling.type)
    except KeyError:
        return f"'{source}' → '{target}'"
    transfer = coupling_type.mesh_transfer
    kind = ('interpolate transfer' if transfer == 'field' else f'{coupling_type.value_kind} transfer'
            if transfer == 'value' else 'needs both physics on one mesh')
    return f"'{source}' → '{target}' ({kind})"


def evaluation_mesh(spec, registry, expression) -> str:
    """Mesh a derived-value expression is evaluated on (S7.2 D3): that of the first physics in model order whose
    output fields it names; the default mesh when it names none; '' when unknown."""
    try:
        names = referenced_names(expression) if isinstance(expression, str) else set()
    except ExpressionError:
        names = set()
    field_physics = output_field_names(spec, registry)
    named = {field_physics[name][0] for name in names if name in field_physics}
    first = next((physics.name for physics in spec.physics if physics.name in named), None)
    if first is None:
        return spec.meshes[0].name if spec.meshes else ''
    return physics_mesh_name(spec, first)
