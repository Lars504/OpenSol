"""Model tree: the spec's nodes (8.1 paths) with issue icons, context menus that call document node operations and
in-place rename; a deferred rebuild runs only when the structural signature changed."""
from dataclasses import dataclass
from functools import partial

from PySide6 import QtCore, QtGui, QtWidgets

from ..materials import REGISTRY as MATERIAL_LIBRARY
from ..model.geometry import GEOMETRY_KINDS, consumed_objects
from ..model.schema import (FEATURE_CATEGORIES, ISSUE_LEVELS, MESH_KINDS, MESH_PARAMS, PHYSICS_GROUPS, SOLVER_ROLES,
                            coupling_type_accepts, feature_precision_note, fill_defaults, physics_type_misfit,
                            solver_backend, with_default_precision)
from ..solvers.roles import BACKENDS
from .document import (ANALYTIC_CURVES_NODE, COMPUTED_RESULTS_NODE, DERIVED_CURVES_NODE, DERIVED_VALUES_NODE,
                       ESTIMATION_NODE, MESHES_NODE, derived_plot_names, issue_node_path, parent_node, sibling_names,
                       solver_fits, solver_users)
from .estimation_form import (ESTIMATION_TIP, estimation_blocker, estimation_overview_text, estimation_toggle,
                              estimation_tree_label)
from .forms import SELECTION_KIND_LABELS, derived_operation_text, rich_tooltip
from .helpers import (coupling_mesh_text, mesh_color, mesh_kind_problem, mesh_kind_shown, mesh_summary_text,
                      mesh_swatch_icon, physics_mesh_name, run_user_operation, several_meshes)

ISSUE_LEVEL_PIXMAPS = {
    'error': QtWidgets.QStyle.StandardPixmap.SP_MessageBoxCritical,
    'warning': QtWidgets.QStyle.StandardPixmap.SP_MessageBoxWarning,
    'info': QtWidgets.QStyle.StandardPixmap.SP_MessageBoxInformation,
}
_PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole
_PATH_SEPARATOR = '\x1f'          # never part of an identifier, so node paths round-trip through item data
_CATEGORY_LABELS = {'source': 'Sources', 'probe': 'Probes', 'monitor': 'Monitors',
                    'boundary_condition': 'Boundary conditions', 'domain_condition': 'Domain conditions'}
_SELECTION_KIND_LABELS = (('domain', 'Domain selection'), ('boundary', 'Boundary selection'),
                          ('point', 'Point selection'), ('line', 'Line selection'),
                          ('interface', 'Interface selection'))
_SELECTION_KIND_TEXTS = dict(SELECTION_KIND_LABELS)   # the settings form's Kind texts
# tooltips of the result-settings Add entries (tree menus and the settings panel's buttons)
RESULT_SETTINGS_TIPS = {
    'derived_value': 'One number computed from the results after the run, e.g. the mean temperature over a region or '
                     'the value of a spectrum at one wavelength, shown in the Derived values tab',
    'analytic_curve': 'Analytic formula drawn dashed over a result curve, the sweep plot or a line profile',
    'derived_curve': 'Curve computed from result curves after the run, e.g. a ratio of two spectra, drawn in a plot of '
                     'its own in the Derived curves tab',
    'derived_curve_to_plot': 'Curve computed from result curves after the run, drawn together with the curves of this '
                             'plot in the Derived curves tab',
}
# kind -> (Add button / menu text, ModelDocument method name) of the three result-setting adders (tree menus, panel)
RESULT_SETTING_ADDERS = {'derived_value': ('Add derived value', 'add_derived_value'),
                         'analytic_curve': ('Add analytic curve', 'add_analytic_curve'),
                         'derived_curve': ('Add derived curve', 'add_derived_curve')}
MESH_KIND_LABELS = dict(next(param.choice_labels for param in MESH_PARAMS if param.name == 'kind'))
_MESHES_TIP = ('The meshes of the model: each physics runs on one of them (its Mesh setting). One mesh is enough for '
               'most models; a second one lets a slowly varying physics such as heat use coarser cells than a wave '
               'physics. Fields and heat sources are carried between meshes automatically.')


def encode_node_path(path) -> str:
    """Node path tuple -> text stored in Qt item data."""
    return _PATH_SEPARATOR.join(str(part) for part in path)


def decode_node_path(text):
    """Inverse of encode_node_path; None for missing item data."""
    return None if text is None else tuple(str(text).split(_PATH_SEPARATOR))


def worst_issue_level(issues):
    """Most severe level among issues ('error' > 'warning' > 'info'), or None."""
    levels = {issue.level for issue in issues}
    return next((level for level in ISSUE_LEVELS if level in levels), None)


@dataclass(frozen=True)
class _TreeNode:
    path: tuple
    parent: tuple | None
    label: str
    enabled: bool = True
    description: str = ''
    color: str = ''          # mesh colour swatch (mesh nodes; physics nodes of a model with several meshes)


def _type_or_none(lookup, name):
    try:
        return lookup(name)
    except KeyError:
        return None


def _gui_text(type_info):
    """GUI tooltip text of a registered type: its summary, else its description ('' for None)."""
    return (type_info.summary or type_info.description) if type_info is not None else ''


def _feature_gui_text(feature_type):
    """_gui_text of a feature type plus, as its own paragraph, the time stepping precisions it is limited to."""
    if feature_type is None:
        return ''
    return '\n\n'.join(text for text in (_gui_text(feature_type), feature_precision_note(feature_type)) if text)


def _mesh_description(spec, mesh, estimate):
    """Mesh node tooltip: its kind and size, the physics using it, then the refinements, per-axis options and outer
    grading of a graded / automatic mesh."""
    users = ', '.join(physics.name for physics in spec.physics_on(mesh.name))
    default = ' It is the default mesh: new physics run on it.' if spec.meshes and spec.meshes[0] is mesh else ''
    summary = mesh_summary_text(mesh, estimate)
    text = (f'{summary[:1].upper()}{summary[1:]}.\n\nUsed by: {users or "no physics (not built)"}.{default}\n\n'
            'Cell sizes of the grid.')
    if mesh.size_points:   # every mode (adaptive refinement of a triangle mesh writes them)
        text += f' {len(mesh.size_points):,} size points from mesh refinement.'
    if mesh.mode == 'uniform':
        return text
    parts = []
    if mesh.refinements:
        parts.append(f"{len(mesh.refinements)} refinement{'s' if len(mesh.refinements) != 1 else ''}")
    for axis_spec in mesh.axes:
        cells = ('cells listed' if axis_spec.edges else 'segments' if axis_spec.segments
                 else f"{len(axis_spec.bands)} band{'s' if len(axis_spec.bands) != 1 else ''}")
        parts.append(f'axis {axis_spec.axis}: {cells}')
    if mesh.outer_dx is not None:
        parts.append('outer cells growing to the outer cell size')
    joined = '; '.join(parts)
    return f'{text} {joined[:1].upper()}{joined[1:]}.' if parts else text


def coupling_candidates(spec, registry) -> list:
    """[(CouplingType, [(source, target), ...])] of the registered coupling types that fit two distinct physics of
    the spec (shared by the Add coupling menu and the Couplings settings panel)."""
    present = [(physics.name, physics.type) for physics in spec.physics]
    candidates = []
    for coupling_type in registry.coupling_types():
        pairs = [(source, target) for source, source_type in present for target, target_type in present
                 if source != target and coupling_type_accepts(coupling_type, 'source', source_type)
                 and coupling_type_accepts(coupling_type, 'target', target_type)]
        if pairs:
            candidates.append((coupling_type, pairs))
    return candidates


def solver_candidates(spec, registry) -> list:
    """[(role, SolverType, reason)] per registered type and role it fills: reason '' when some physics of the spec can
    use it for that role in this study (Add solver menu and Solvers settings panel), else why not."""
    candidates = []
    for solver_type in registry.solver_types():
        values = fill_defaults(solver_type.params, with_default_precision(spec, solver_type, {}))   # as add_solver
        for role in solver_type.roles:
            reasons = [reason for _physics, _role, reason in solver_fits(spec, registry, solver_type, values, (role,))]
            if not reasons:
                reason = f'no physics in this model needs {SOLVER_ROLES[role].lower()}'
            else:
                reason = '' if '' in reasons else reasons[0]
            candidates.append((role, solver_type, reason))
    return candidates


def _tree_nodes(spec, registry, results_available=False, mesh_estimates=None) -> list:
    """Every node of the spec in display order (parents before children), then 'Results' when results_available;
    mesh_estimates: {mesh name: estimate or None} for the mesh labels."""
    mesh_estimates = mesh_estimates or {}
    multi_mesh = several_meshes(spec)
    root = ('model',)
    nodes = [_TreeNode(root, None, spec.title or 'Untitled'),
             _TreeNode(('parameters',), root, f'Parameters ({len(spec.parameters)})'),
             _TreeNode(('geometry',), root, 'Geometry')]
    consumed = consumed_objects(spec)
    for obj in spec.geometry:
        kind = GEOMETRY_KINDS.get(obj.kind)
        kind_label = kind.label if kind is not None else f'unknown kind {obj.kind!r}'
        used = ', used by a composite' if obj.name in consumed else ''
        nodes.append(_TreeNode(('geometry', obj.name), ('geometry',), f'{obj.name} ({kind_label}{used})'))

    nodes.append(_TreeNode(('selections',), root, 'Selections'))
    for selection in spec.selections:
        nodes.append(_TreeNode(('selection', selection.name), ('selections',),
                               f'{selection.name} ({_SELECTION_KIND_TEXTS.get(selection.kind, selection.kind)})'))

    nodes.append(_TreeNode(('materials',), root, 'Materials'))
    for material in spec.materials:
        label = f'{material.name} ({material.library})' if material.library else material.name
        nodes.append(_TreeNode(('material', material.name), ('materials',), label))

    for physics in spec.physics:
        physics_type = _type_or_none(registry.physics_type, physics.type)
        type_label = physics_type.label if physics_type is not None else f'unknown type {physics.type!r}'
        physics_path = ('physics', physics.name)
        mesh_name = physics_mesh_name(spec, physics)
        mesh_text = f"Mesh: '{mesh_name}'" if mesh_name else f'Mesh: unknown ({physics.mesh!r})'
        nodes.append(_TreeNode(physics_path, root,
                               f'{physics.name} ({type_label})' + (f' [{mesh_name or physics.mesh}]' if multi_mesh
                                                                    else ''),
                               description='\n\n'.join(text for text in (_gui_text(physics_type), mesh_text) if text),
                               color=mesh_color(spec, mesh_name) if multi_mesh else ''))
        for feature in physics.features:
            feature_type = (_type_or_none(physics_type.feature_type, feature.type)
                            if physics_type is not None else None)
            feature_label = feature_type.label if feature_type is not None else f'unknown type {feature.type!r}'
            nodes.append(_TreeNode(('feature', physics.name, feature.name), physics_path,
                                   f'{feature.name} ({feature_label})', enabled=bool(feature.enabled),
                                   description=_feature_gui_text(feature_type)))

    nodes.append(_TreeNode(('couplings',), root, 'Couplings'))
    for coupling in spec.couplings:
        coupling_type = _type_or_none(registry.coupling_type, coupling.type)
        type_label = coupling_type.label if coupling_type is not None else f'unknown type {coupling.type!r}'
        meshes_text = coupling_mesh_text(spec, registry, coupling) if multi_mesh else ''
        nodes.append(_TreeNode(('coupling', coupling.name), ('couplings',),
                               f'{coupling.name} ({type_label}: {coupling.source} → {coupling.target})',
                               enabled=bool(coupling.enabled),
                               description='\n\n'.join(text for text in (_gui_text(coupling_type),
                                                                         f'Meshes: {meshes_text}' if meshes_text
                                                                         else '') if text)))

    nodes.append(_TreeNode(MESHES_NODE, root, f'Meshes ({len(spec.meshes)})', description=_MESHES_TIP))
    for mesh in spec.meshes:
        estimate = mesh_estimates.get(mesh.name)
        used = bool(spec.physics_on(mesh.name)) or (not spec.physics and spec.meshes[0] is mesh)
        nodes.append(_TreeNode(('mesh', mesh.name), MESHES_NODE,
                               f'{mesh.name} – {mesh_summary_text(mesh, estimate)}' + ('' if used else ' (unused)'),
                               enabled=used, description=_mesh_description(spec, mesh, estimate),
                               color=mesh_color(spec, mesh.name)))
    backend = solver_backend(spec)
    nodes.append(_TreeNode(('solvers',), root, f'Solvers ({BACKENDS.get(backend, backend)})',
                           description='How the equations of each physics are solved. Without solver nodes every '
                                       'physics uses its automatic choice; add one to change the method or its '
                                       'precision.'))
    for solver in spec.solvers:
        solver_type = _type_or_none(registry.solver_type, solver.type)
        type_label = solver_type.label if solver_type is not None else f'unknown type {solver.type!r}'
        users = ', '.join(f'{physics} ({SOLVER_ROLES.get(role, role).lower()})'
                          for physics, role in solver_users(spec, solver.name))
        nodes.append(_TreeNode(('solver', solver.name), ('solvers',), f'{solver.name} ({type_label})',
                               description='\n\n'.join(text for text in (_gui_text(solver_type),
                                                                         f'Used by: {users or "nothing yet"}')
                                                       if text)))
    study_type = _type_or_none(registry.study_type, spec.study.type)
    study_label = study_type.label if study_type is not None else f'unknown type {spec.study.type!r}'
    sweep = ' + sweep' if spec.study.sweep is not None else ''
    refinement = ' + mesh refinement' if spec.study.adaptation is not None else ''
    nodes.append(_TreeNode(('study',), root, f'Study ({study_label}{sweep}{refinement})',
                           description=_gui_text(study_type)))
    if spec.study.estimation is not None:
        nodes.append(_TreeNode(ESTIMATION_NODE, ('study',), estimation_tree_label(spec.study.estimation),
                               description=f'{estimation_overview_text(spec.study.estimation)}\n\n{ESTIMATION_TIP}'))
    nodes.append(_TreeNode(('results',), root, 'Result settings',
                           description='Derived values, analytic and derived curves: what is computed from and drawn '
                                       'over the results of every run.'))
    nodes.append(_TreeNode(DERIVED_VALUES_NODE, ('results',), f'Derived values ({len(spec.derived_values)})',
                           description='Numbers computed from the fields after every run, e.g. a mean temperature.'))
    for derived in spec.derived_values:
        over = '' if derived.source is None else f' of {derived.source}'
        reference = ', with reference' if derived.reference is not None else ''
        nodes.append(_TreeNode(('derived', derived.name), DERIVED_VALUES_NODE,
                               f'{derived.name} ({derived_operation_text(derived.operation)}{over}{reference})'))
    nodes.append(_TreeNode(ANALYTIC_CURVES_NODE, ('results',), f'Analytic curves ({len(spec.analytic_curves)})',
                           description='Formulas drawn dashed over a result curve, the sweep plot or a line profile.'))
    for curve in spec.analytic_curves:
        nodes.append(_TreeNode(('curve', curve.name), ANALYTIC_CURVES_NODE, f'{curve.name} (on {curve.target})',
                               description=f'Analytic curve: {curve.expression}'))
    nodes.append(_TreeNode(DERIVED_CURVES_NODE, ('results',), f'Derived curves ({len(spec.derived_curves)})',
                           description='Curves computed from result curves after every run, one sub-node per plot '
                                       '(the tab of the Derived curves results view).'))
    for plot in derived_plot_names(spec):
        plot_path = ('derived_plot', plot)
        nodes.append(_TreeNode(plot_path, DERIVED_CURVES_NODE, plot,
                               description=f'Plot {plot}: the derived curves drawn together in it. Rename it to '
                                           'rename the plot of all of them.'))
        for curve in spec.derived_curves:
            if (curve.plot or curve.name) == plot:
                legend = f' ({curve.label})' if curve.label and curve.label != curve.name else ''
                nodes.append(_TreeNode(('derived_curve', curve.name), plot_path, f'{curve.name}{legend}',
                                       description=f'Derived curve: {curve.expression}'))
    if results_available:
        nodes.append(_TreeNode(COMPUTED_RESULTS_NODE, root, 'Results',
                               description='The fields, probes and spectra of the last build or run: opens the '
                                           'Results panel and shows the field on the canvas.'))
    return nodes


def _issues_by_node(issues, node_paths) -> dict:
    """Node path -> issues; an issue whose path has no tree item attaches to its nearest ancestor that has one
    (document.parent_node; ('model',) always has one)."""
    by_node = {}
    for issue in issues:
        path = issue_node_path(issue)
        while path not in node_paths:
            path = parent_node(path)
        by_node.setdefault(path, []).append(issue)
    return by_node


def _tooltip(node, issues) -> str:
    """Rich tooltip: the node's GUI text, then one paragraph per issue."""
    return rich_tooltip(node.description, *(f'{issue.level.capitalize()}: {issue.message}' for issue in issues))


def _enabled_flag(spec, path):
    """enabled of a feature / coupling node; None for nodes without the flag (or unresolved ones)."""
    try:
        if path[0] == 'feature':
            return bool(spec.physics_by_name(path[1]).feature(path[2]).enabled)
        if path[0] == 'coupling':
            return bool(spec.coupling(path[1]).enabled)
    except KeyError:
        return None
    return None


class _RenameDelegate(QtWidgets.QStyledItemDelegate):
    """In-place rename: the editor starts from the bare node name; a changed name goes to on_rename(path, name)."""

    def __init__(self, on_rename, parent):
        super().__init__(parent)
        self._on_rename = on_rename

    def setEditorData(self, editor, index):
        path = decode_node_path(index.data(_PATH_ROLE))
        editor.setText(path[-1] if path else '')
        editor.selectAll()

    def setModelData(self, editor, model, index):
        path = decode_node_path(index.data(_PATH_ROLE))
        new_name = editor.text().strip()
        if path is not None and len(path) > 1 and new_name and new_name != path[-1]:
            # deferred: the rename rebuilds the tree, which must not happen while the editor commits
            QtCore.QTimer.singleShot(0, partial(self._on_rename, path, new_name))


class ModelTree(QtWidgets.QTreeWidget):
    """Tree of the model's nodes; selecting an item selects the node in the document."""
    status_message = QtCore.Signal(str)
    node_hovered = QtCore.Signal(object)   # node path under the mouse (the canvas hover highlight), None = none

    def __init__(self, document, registry, parent=None):
        super().__init__(parent)
        self.document = document
        self.registry = registry
        self._items = {}                 # node path -> QTreeWidgetItem of the current build
        self._signature = None
        self._collapsed_paths = set()
        self._refresh_pending = False
        self._results_available = False  # show the 'Results' node of the last build / run
        self._hovered_path = None
        self._range_anchor = None        # object a Shift+click range starts from (last plain / Ctrl-clicked object)

        self.setMouseTracking(True)
        self.setHeaderHidden(True)
        self.setColumnCount(1)
        self.setUniformRowHeights(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed)
        self.setItemDelegate(_RenameDelegate(self._rename_node, self))
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)
        self.currentItemChanged.connect(self._on_current_item_changed)
        self.itemExpanded.connect(self._on_item_expanded)
        self.itemCollapsed.connect(self._on_item_collapsed)
        document.changed.connect(self._on_document_changed)
        document.issues_invalidated.connect(self._schedule_refresh)
        document.node_selected.connect(self._on_node_selected)
        document.geometry_group_changed.connect(lambda _names: self._on_node_selected(self.document.current_node))
        self._refresh()

    # ---- public ----
    def delete_current(self):
        """Delete the document's current node (named nodes only), or every shift-click selected object."""
        group = self.document.geometry_group
        if group:
            self._run_operation(partial(self.document.delete_geometry, group), f'Delete {", ".join(group)}')
            return
        path = tuple(self.document.current_node or ())
        if path == ESTIMATION_NODE:
            self._set_estimation(False)
            return
        if len(path) < 2:
            self.status_message.emit('Select a named model node to delete')
            return
        if path[0] == 'derived_plot':   # not a spec item: its curves are deleted one by one
            self.status_message.emit(f"Plot '{path[1]}' is deleted with its last derived curve: delete the curves")
            return
        self._run_operation(partial(self.document.delete_node, path), f'Delete {path[-1]}')

    def set_results_available(self, available):
        """Show / hide the 'Results' node (results of the last build / run exist)."""
        if bool(available) != self._results_available:
            self._results_available = bool(available)
            self._schedule_refresh()

    # ---- deferred refresh ----
    def _on_document_changed(self, section, origin):
        self._schedule_refresh()

    def _schedule_refresh(self):
        if not self._refresh_pending:
            self._refresh_pending = True
            QtCore.QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        self._refresh_pending = False
        spec = self.document.spec
        mesh_estimates = {mesh.name: self.document.mesh_estimate(mesh.name)[0] for mesh in spec.meshes}
        nodes = _tree_nodes(spec, self.registry, self._results_available, mesh_estimates)
        issues_by_node = _issues_by_node(self.document.issues(), {node.path for node in nodes})
        signature = tuple((node.path, node.label, node.enabled, node.color,
                           worst_issue_level(issues_by_node.get(node.path, ()))) for node in nodes)
        if signature != self._signature:
            self._rebuild(nodes, issues_by_node)
            self._signature = signature
        else:
            for node in nodes:
                item = self._items.get(node.path)
                tooltip = _tooltip(node, issues_by_node.get(node.path, ()))
                if item is not None and item.toolTip(0) != tooltip:
                    item.setToolTip(0, tooltip)   # only on change: dataChanged would reset an open rename editor
        self._sync_current_item()

    def _rebuild(self, nodes, issues_by_node):
        self._set_hovered_path(None)     # its item is gone; the next mouse move finds the new one
        scroll_position = self.verticalScrollBar().value()
        item_flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        disabled_brush = self.palette().brush(QtGui.QPalette.ColorGroup.Disabled, QtGui.QPalette.ColorRole.Text)
        self.blockSignals(True)
        try:
            self.clear()
            self._items = {}
            for node in nodes:
                item = QtWidgets.QTreeWidgetItem([node.label])
                parent_item = self._items.get(node.parent) if node.parent is not None else None
                if parent_item is None:
                    self.addTopLevelItem(item)
                else:
                    parent_item.addChild(item)
                named = len(node.path) > 1
                item.setFlags(item_flags | QtCore.Qt.ItemFlag.ItemIsEditable if named else item_flags)
                item.setData(0, _PATH_ROLE, encode_node_path(node.path))
                issues = issues_by_node.get(node.path, ())
                level = worst_issue_level(issues)
                issue_icon = self.style().standardIcon(ISSUE_LEVEL_PIXMAPS[level]) if level is not None else None
                if node.color:   # mesh swatch, the issue level drawn into its corner
                    item.setIcon(0, mesh_swatch_icon(node.color, overlay=issue_icon))
                elif issue_icon is not None:
                    item.setIcon(0, issue_icon)
                item.setToolTip(0, _tooltip(node, issues))
                if node.path == ('model',) or not node.enabled:
                    font = item.font(0)
                    font.setBold(node.path == ('model',))
                    font.setItalic(not node.enabled)
                    item.setFont(0, font)
                if not node.enabled:
                    item.setForeground(0, disabled_brush)
                self._items.setdefault(node.path, item)
            for path, item in self._items.items():
                item.setExpanded(path not in self._collapsed_paths)
        finally:
            self.blockSignals(False)
        self.verticalScrollBar().setValue(scroll_position)

    def _sync_current_item(self):
        """Make the item of document.current_node current and select the shift-click selected objects' items, without
        re-selecting in the document (the view is SingleSelection: only the group is selected programmatically)."""
        item = self._items.get(tuple(self.document.current_node or ()))
        group_items = [self._items[('geometry', name)] for name in self.document.geometry_group
                       if ('geometry', name) in self._items]
        self.blockSignals(True)
        try:
            if item is None:
                self.clearSelection()
                return
            if item is not self.currentItem():
                self.setCurrentItem(item)
                self.scrollToItem(item)
            wanted = {id(other): other for other in [item, *group_items]}
            if {id(other) for other in self.selectedItems()} != set(wanted):
                self.clearSelection()
                for other in wanted.values():
                    other.setSelected(True)
        finally:
            self.blockSignals(False)

    # ---- selection / expansion ----
    def mousePressEvent(self, event):
        """Ctrl+click on an object toggles it in the selected objects, Shift+click selects the objects from the anchor
        (last plain / Ctrl-clicked object) to it (Ctrl+Shift: adds them); a right click on a selected object keeps
        the selection (its context menu acts on all); a plain click selects one node alone."""
        item = self.itemAt(event.position().toPoint())
        path = decode_node_path(item.data(0, _PATH_ROLE)) if item is not None else None
        is_object = path is not None and len(path) == 2 and path[0] == 'geometry'
        shift = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        control = bool(event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        left = event.button() == QtCore.Qt.MouseButton.LeftButton
        if is_object and left and shift:
            self._select_object_range(path[1], add=control)
            return
        if is_object and left and control:
            self._range_anchor = path[1]
            self.document.toggle_geometry_group(path[1])
            return
        if is_object and event.button() == QtCore.Qt.MouseButton.RightButton and path[1] in self.document.geometry_group:
            return                       # customContextMenuRequested still follows
        if is_object and left:
            self._range_anchor = path[1]
        super().mousePressEvent(event)
        if left and path == tuple(self.document.current_node or ()) and self.document.geometry_group:
            self.document.select(path)   # a click on the current item changes no current item: drop the group here

    def _select_object_range(self, name, add=False):
        """Select the objects between the range anchor (else the current object) and `name` in tree order, `name`
        current; add=True keeps the objects already selected."""
        selected = self.document.selected_geometry()
        anchor = self._range_anchor if self._range_anchor in selected else (selected[-1] if selected else None)
        order = [obj.name for obj in self.document.spec.geometry]
        if anchor not in order:
            self._range_anchor = name
            self.document.select(('geometry', name))
            return
        self._range_anchor = anchor
        start, end = sorted((order.index(anchor), order.index(name)))
        names = [*selected, *order[start:end + 1]] if add else order[start:end + 1]
        self.document.select_geometry(names, name)

    def _on_current_item_changed(self, current, previous):
        if current is None:
            return
        path = decode_node_path(current.data(0, _PATH_ROLE))
        if path is not None and path != tuple(self.document.current_node or ()):
            self.document.select(path)

    def _on_node_selected(self, path):
        if self._refresh_pending:
            return                       # the pending refresh syncs the current item
        if tuple(path) in self._items:
            self._sync_current_item()
        else:
            self._schedule_refresh()

    # ---- hover ----
    def mouseMoveEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        self._set_hovered_path(decode_node_path(item.data(0, _PATH_ROLE)) if item is not None else None)
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._set_hovered_path(None)
        super().leaveEvent(event)

    def _set_hovered_path(self, path):
        if path != self._hovered_path:
            self._hovered_path = path
            self.node_hovered.emit(path)

    def _on_item_expanded(self, item):
        self._collapsed_paths.discard(decode_node_path(item.data(0, _PATH_ROLE)))

    def _on_item_collapsed(self, item):
        self._collapsed_paths.add(decode_node_path(item.data(0, _PATH_ROLE)))

    # ---- operations ----
    def _run_operation(self, operation, label=''):
        """Run one document node operation; user errors go to status_message ('<label>: <message>' with a label)."""
        run_user_operation(operation, self.status_message.emit, label)

    def _rename_node(self, path, new_name):
        self._run_operation(partial(self.document.rename_node, path, new_name), f'Rename {path[-1]}')

    def _start_rename(self, path):
        item = self._items.get(path)
        if item is not None:
            self.setCurrentItem(item)
            self.editItem(item, 0)

    # ---- context menus ----
    def _show_context_menu(self, position):
        item = self.itemAt(position)
        path = decode_node_path(item.data(0, _PATH_ROLE)) if item is not None else ('model',)
        menu = self._context_menu(path)
        if not menu.isEmpty():
            menu.exec(self.viewport().mapToGlobal(position))
        menu.deleteLater()

    def _context_menu(self, path):
        menu = QtWidgets.QMenu(self)
        menu.setToolTipsVisible(True)
        head = path[0]
        if head == 'model':
            self._add_action(menu, 'Add parameter', self.document.add_parameter)
            self._add_geometry_menu(menu)
            self._add_selection_menu(menu)
            self._add_material_menu(menu)
            self._add_physics_menu(menu)
            self._add_coupling_menu(menu)
            self._add_result_settings_actions(menu)
        elif head == 'parameters':
            self._add_action(menu, 'Add parameter', self.document.add_parameter)
        elif head == 'geometry' and len(path) == 2 and path[1] in self.document.geometry_group:
            self._add_geometry_group_actions(menu, self.document.geometry_group)
            return menu
        elif head == 'geometry':
            self._add_geometry_menu(menu)
        elif head in ('selections', 'selection'):
            self._add_selection_menu(menu)
        elif head in ('materials', 'material'):
            self._add_material_menu(menu)
        elif head == 'physics':
            self._add_feature_menu(menu, path[1])
            self._add_physics_mesh_menu(menu, path[1])
            self._add_physics_menu(menu)
        elif head == 'feature':
            self._add_feature_menu(menu, path[1])
        elif head in ('couplings', 'coupling'):
            self._add_coupling_menu(menu)
        elif head == 'meshes':
            self._add_mesh_menu(menu)
        elif head == 'mesh':
            self._add_mesh_menu(menu)
            self._add_node_actions(menu, path, delete=False)
            self._add_mesh_delete_actions(menu, path[1])
            return menu
        elif head in ('solvers', 'solver'):
            self._add_solver_menu(menu)
        elif head in ('study', 'estimation'):
            self._add_estimation_action(menu)
        elif head == 'derived_curve':   # a curve: new curves join its plot
            plot = next((curve.plot or curve.name for curve in self.document.spec.derived_curves
                         if curve.name == path[1]), '')
            self._add_action(menu, 'Add derived curve to this plot', partial(self.document.add_derived_curve, plot),
                             tooltip=RESULT_SETTINGS_TIPS['derived_curve_to_plot'])
        elif head in ('derived_values', 'derived'):
            self._add_result_settings_actions(menu, ('derived_value',))
        elif head in ('analytic_curves', 'curve'):
            self._add_result_settings_actions(menu, ('analytic_curve',))
        elif head in ('results', 'derived_curves'):
            self._add_result_settings_actions(menu, None if head == 'results' else ('derived_curve',))
        elif head == 'derived_plot':
            self._add_action(menu, 'Add derived curve to this plot', partial(self.document.add_derived_curve, path[1]),
                             tooltip=RESULT_SETTINGS_TIPS['derived_curve_to_plot'])
            rename_action = menu.addAction('Rename plot')
            rename_action.triggered.connect(
                lambda _checked=False, path=path: QtCore.QTimer.singleShot(0, partial(self._start_rename, path)))
            return menu
        if len(path) > 1:
            self._add_node_actions(menu, path)
        return menu

    def _add_estimation_action(self, menu):
        """'Add parameter estimation' (disabled with validation's blocker as tooltip) or 'Remove parameter estimation'."""
        if self.document.spec.study.estimation is not None:
            self._add_action(menu, 'Remove parameter estimation', partial(self._set_estimation, False),
                             tooltip='Run the study once, without fitting (Undo brings the estimation back)')
            return
        blocker = estimation_blocker(self.document.spec)
        self._add_action(menu, 'Add parameter estimation', partial(self._set_estimation, True),
                         tooltip=blocker or ESTIMATION_TIP, enabled=not blocker)

    def _set_estimation(self, enabled):
        """Switch StudySpec.estimation on (a new one, its node selected) or off; refusals go to status_message."""
        def operation():
            if self.document.mutate('study', estimation_toggle(enabled), label='Parameter estimation') and enabled:
                self.document.select(ESTIMATION_NODE)
        self._run_operation(operation, 'Parameter estimation')

    def _add_result_settings_actions(self, menu, kinds=None):
        """'Add derived value' / 'Add analytic curve' / 'Add derived curve' (kinds: a subset), with their tooltips."""
        for kind in kinds or tuple(RESULT_SETTING_ADDERS):
            text, method = RESULT_SETTING_ADDERS[kind]
            self._add_action(menu, text, getattr(self.document, method), tooltip=RESULT_SETTINGS_TIPS[kind])

    def _add_action(self, menu, text, operation, tooltip='', enabled=True, label=''):
        """Menu action running operation; label prefixes a refused edit's message (_run_operation)."""
        action = menu.addAction(text)
        if tooltip:
            action.setToolTip(rich_tooltip(tooltip))
        action.setEnabled(enabled)
        action.triggered.connect(
            lambda _checked=False, operation=operation: self._run_operation(operation, label))
        return action

    @staticmethod
    def _add_submenu(menu, title):
        submenu = menu.addMenu(title)
        submenu.setToolTipsVisible(True)
        return submenu

    def _add_geometry_menu(self, menu):
        geometry_menu = self._add_submenu(menu, 'Add geometry')
        model_dimensions = 3 if self.document.spec.is_3d else 2   # box / cylinder_z / sphere only in 3D models
        kinds = [kind for kind in GEOMETRY_KINDS.values() if model_dimensions in kind.dimensions]
        for kind in (kind for kind in kinds if not kind.composite):
            self._add_action(geometry_menu, kind.label, partial(self.document.add_geometry, kind.name),
                             tooltip=kind.summary)
        geometry_menu.addSeparator()
        for kind in (kind for kind in kinds if kind.composite):
            self._add_action(geometry_menu, kind.label, partial(self.document.add_geometry, kind.name),
                             tooltip=kind.summary)

    def _add_geometry_group_actions(self, menu, names):
        """Actions on the shift-click selected objects: combine them (domain objects not yet in a composite) or delete
        them."""
        spec = self.document.spec
        consumed = consumed_objects(spec)
        objects = {obj.name: obj for obj in spec.geometry}
        problem = next((f'{name} is not a domain object' for name in names
                        if name in objects and GEOMETRY_KINDS[objects[name].kind].entity_kind != 'domain'),
                       next((f'{name} is already part of a combination' for name in names if name in consumed), ''))
        label = ', '.join(names)
        for kind, text in (('union', 'Union'), ('intersection', 'Intersection'),
                           ('difference', f'Difference ({names[0]} minus the others)')):
            self._add_action(menu, text, partial(self.document.combine_geometry, kind, names),
                             tooltip=problem or GEOMETRY_KINDS[kind].summary, enabled=not problem,
                             label=f'{GEOMETRY_KINDS[kind].label} of {label}')
        menu.addSeparator()
        self._add_action(menu, f'Delete {len(names)} objects', partial(self.document.delete_geometry, names),
                         label=f'Delete {label}')

    def _add_selection_menu(self, menu):
        selection_menu = self._add_submenu(menu, 'Add selection')
        for kind, label in _SELECTION_KIND_LABELS:
            self._add_action(selection_menu, label, partial(self.document.add_selection, kind))

    def _add_material_menu(self, menu):
        material_menu = self._add_submenu(menu, 'Add material')
        self._add_action(material_menu, 'Blank material', partial(self.document.add_material, None))
        library_menu = self._add_submenu(material_menu, 'From library')
        for name in sorted(MATERIAL_LIBRARY):
            self._add_action(library_menu, name, partial(self.document.add_material, name),
                             tooltip=MATERIAL_LIBRARY[name].summary)

    def _add_physics_menu(self, menu):
        """One submenu per PHYSICS_GROUPS entry holding the types that fit the model (dimensions, 2D space); the
        others are disabled under 'Not available in this model' with the reason as tooltip."""
        physics_menu = self._add_submenu(menu, 'Add physics')
        spec = self.document.spec
        fitting, misfits = {}, []
        for physics_type in self.registry.physics_types():
            reason = physics_type_misfit(physics_type, spec, self.registry)
            if reason:
                misfits.append((physics_type, reason))
            else:
                fitting.setdefault(physics_type.group, []).append(physics_type)
        for group, group_label in (*PHYSICS_GROUPS, ('', 'Other')):
            if group not in fitting:
                continue
            group_menu = self._add_submenu(physics_menu, group_label)
            for physics_type in fitting[group]:
                self._add_action(group_menu, physics_type.label,
                                 partial(self.document.add_physics, physics_type.name), tooltip=_gui_text(physics_type))
        if not fitting:
            self._add_action(physics_menu, '(no physics type fits this model)', lambda: None, enabled=False)
        self._add_misfit_menu(physics_menu, [(physics_type.label, reason) for physics_type, reason in misfits])

    def _add_feature_menu(self, menu, physics_name):
        physics_spec = _type_or_none(self.document.spec.physics_by_name, physics_name)
        physics_type = (_type_or_none(self.registry.physics_type, physics_spec.type)
                        if physics_spec is not None else None)
        if physics_type is None:
            return
        feature_menu = self._add_submenu(menu, 'Add feature')
        by_category = {}
        for feature_type in physics_type.features:
            by_category.setdefault(feature_type.category, []).append(feature_type)
        categories = [category for category in FEATURE_CATEGORIES if category in by_category]
        for category in categories:
            target_menu = (self._add_submenu(feature_menu, _CATEGORY_LABELS.get(category, category))
                           if len(categories) > 1 else feature_menu)
            for feature_type in by_category[category]:
                self._add_action(target_menu, feature_type.label,
                                 partial(self.document.add_feature, physics_name, feature_type.name),
                                 tooltip=_feature_gui_text(feature_type))
        if not categories:
            self._add_action(feature_menu, '(this physics has no features)', lambda: None, enabled=False)

    def _add_coupling_menu(self, menu):
        """Coupling types valid for the physics present; one candidate pair -> direct action, several -> submenu."""
        coupling_menu = self._add_submenu(menu, 'Add coupling')
        candidates = coupling_candidates(self.document.spec, self.registry)
        for coupling_type, pairs in candidates:
            if len(pairs) == 1:
                source, target = pairs[0]
                self._add_action(coupling_menu, f'{coupling_type.label} ({source} → {target})',
                                 partial(self.document.add_coupling, coupling_type.name, source, target),
                                 tooltip=_gui_text(coupling_type))
                continue
            pair_menu = self._add_submenu(coupling_menu, coupling_type.label)
            for source, target in pairs:
                self._add_action(pair_menu, f'{source} → {target}',
                                 partial(self.document.add_coupling, coupling_type.name, source, target),
                                 tooltip=_gui_text(coupling_type))
        if not candidates:
            self._add_action(coupling_menu, '(no coupling fits the physics present)', lambda: None, enabled=False)

    def _add_solver_menu(self, menu):
        """One submenu per solver role holding the types some physics of the model can use for it in this study; the
        others are disabled under 'Not available in this model' with the reason as tooltip."""
        solver_menu = self._add_submenu(menu, 'Add solver')
        candidates = solver_candidates(self.document.spec, self.registry)
        for role, role_label in SOLVER_ROLES.items():
            fitting = [solver_type for candidate_role, solver_type, reason in candidates
                       if candidate_role == role and not reason]
            if not fitting:
                continue
            role_menu = self._add_submenu(solver_menu, role_label)
            for solver_type in fitting:
                self._add_action(role_menu, solver_type.label, partial(self.document.add_solver, solver_type.name),
                                 tooltip=_gui_text(solver_type))
        misfits = [(role, solver_type, reason) for role, solver_type, reason in candidates if reason]
        if len(misfits) == len(candidates):
            self._add_action(solver_menu, '(no solver fits the physics present)', lambda: None, enabled=False)
        self._add_misfit_menu(solver_menu, [(f'{solver_type.label} ({SOLVER_ROLES[role].lower()})', reason)
                                            for role, solver_type, reason in misfits])

    def _add_misfit_menu(self, menu, entries):
        """Disabled (text, reason tooltip) entries under 'Not available in this model' after a separator; nothing without
        entries."""
        if not entries:
            return
        menu.addSeparator()
        misfit_menu = self._add_submenu(menu, 'Not available in this model')
        for text, reason in entries:
            self._add_action(misfit_menu, text, lambda: None, tooltip=reason, enabled=False)

    def _add_mesh_menu(self, menu):
        """'Add mesh' submenu: one entry per mesh kind of the model's dimension (triangles 2D, tetrahedra 3D; disabled
        on non-Cartesian grids)."""
        mesh_menu = self._add_submenu(menu, 'Add mesh')
        spec = self.document.spec
        for kind in MESH_KINDS:
            if not mesh_kind_shown(spec, kind):
                continue
            problem = mesh_kind_problem(spec, kind)
            enabled = not problem
            tip = problem or 'A new mesh, automatic sizes. Choose it as the Mesh of a physics to use it.'
            self._add_action(mesh_menu, MESH_KIND_LABELS.get(kind, kind), partial(self.document.add_mesh, kind),
                             tooltip=tip, enabled=enabled, label='Add mesh')

    def _add_physics_mesh_menu(self, menu, physics_name):
        """'Mesh' submenu of a physics: the model's meshes (the current one checked) and 'New mesh for this physics'."""
        spec = self.document.spec
        current = physics_mesh_name(spec, physics_name)
        mesh_menu = self._add_submenu(menu, f"Mesh ({current or 'unknown'})")
        for mesh in spec.meshes:
            action = self._add_action(mesh_menu, mesh.name,
                                      partial(self.document.set_physics_mesh, physics_name, mesh.name),
                                      label=f'Set {physics_name} mesh')
            action.setCheckable(True)
            action.setChecked(mesh.name == current)
            action.setIcon(mesh_swatch_icon(mesh_color(spec, mesh.name)))
        mesh_menu.addSeparator()
        self._add_action(mesh_menu, 'New mesh for this physics',
                         partial(self.document.add_mesh, 'tensor', physics_name),
                         tooltip='Add an automatic mesh sized for this physics alone and move the physics onto it.',
                         label=f'New mesh for {physics_name}')

    def _add_mesh_delete_actions(self, menu, mesh_name):
        """Delete of a mesh: plain when no physics uses it, else 'Delete and move its physics to <default>'."""
        spec = self.document.spec
        users = [physics.name for physics in spec.physics_on(mesh_name)]
        menu.addSeparator()
        if not users:
            self._add_action(menu, 'Delete', partial(self.document.delete_mesh, mesh_name),
                             enabled=len(spec.meshes) > 1, label=f'Delete {mesh_name}',
                             tooltip='' if len(spec.meshes) > 1 else 'The model needs at least one mesh.')
            return
        remaining = [mesh.name for mesh in spec.meshes if mesh.name != mesh_name]
        if not remaining:
            self._add_action(menu, 'Delete', lambda: None, enabled=False,
                             tooltip='The model needs at least one mesh.')
            return
        self._add_action(menu, f"Delete and move {', '.join(users)} to '{remaining[0]}'",
                         partial(self.document.delete_mesh, mesh_name, True), label=f'Delete {mesh_name}',
                         tooltip=f"Used by {', '.join(users)}: they move to the default mesh '{remaining[0]}'.")

    def _add_node_actions(self, menu, path, delete=True):
        """Enable/disable, rename, duplicate, move up/down and (delete) delete for a named node."""
        spec = self.document.spec
        menu.addSeparator()
        enabled = _enabled_flag(spec, path)
        if enabled is not None:
            enabled_action = menu.addAction('Enabled')
            enabled_action.setCheckable(True)
            enabled_action.setChecked(enabled)
            enabled_action.triggered.connect(
                lambda checked=False, path=path: self._run_operation(
                    partial(self.document.set_enabled, path, bool(checked)),
                    f"{'Enable' if checked else 'Disable'} {path[-1]}"))
        rename_action = menu.addAction('Rename')
        rename_action.triggered.connect(
            lambda _checked=False, path=path: QtCore.QTimer.singleShot(0, partial(self._start_rename, path)))
        name = path[-1]
        self._add_action(menu, 'Duplicate', partial(self.document.duplicate_node, path), label=f'Duplicate {name}')
        siblings = sibling_names(spec, path)
        index = siblings.index(name) if name in siblings else -1
        self._add_action(menu, 'Move up', partial(self.document.move_node, path, -1), enabled=index > 0,
                         label=f'Move {name}')
        self._add_action(menu, 'Move down', partial(self.document.move_node, path, 1),
                         enabled=0 <= index < len(siblings) - 1, label=f'Move {name}')
        if delete:
            menu.addSeparator()
            self._add_action(menu, 'Delete', partial(self.document.delete_node, path), label=f'Delete {name}')
