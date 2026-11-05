"""Qt model document: the edited ModelSpec with undo/redo, file state, the current tree node and node operations.
CPU-side checks (parameters, mesh estimate, validation) are cached per revision; nothing here touches the GPU."""
import copy
import os
import re
import sys
from dataclasses import fields

from PySide6 import QtCore

from ..core.renderer import primary_field_info
from ..io.model_io import load_model, save_model
from ..materials import REGISTRY as MATERIAL_LIBRARY
from ..model.analytic import CURVE_RESULT_KINDS
from ..model.expressions import RESERVED_NAMES, error_message, evaluate_parameters, is_identifier
from ..model.geometry import GEOMETRY_KINDS, Z_PARAM_NAMES, domain_extent, domain_size_z, z_param_set
from ..model.meshing import MESH_ESTIMATE_ERRORS, estimate_memory, estimate_mesh, physics_settings
from ..model.schema import (GLOBAL_DERIVED_OPERATION, INTERFACE_SIDES, MESH_KINDS, SELECTION_KINDS, SOLVER_ROLES,
                            coupling_role_text,
                            coupling_type_accepts, fill_defaults, new_physics_space, output_field_names,
                            scalar_output_names, solver_backend, solver_misfit, state_field_names,
                            with_default_precision)
from ..model.spec import (RESERVED_ENTITY_NAMES, AnalyticCurveSpec, CouplingSpec, DerivedCurveSpec,
                          DerivedValueSpec, DomainSpec, FeatureSpec, GeometrySpec, MaterialSpec, MeshSpec, ModelSpec,
                          ParameterSpec, PhysicsSpec, SelectionSpec, SolverSpec, StudySpec, SweepSpec)
from ..model.templates import make_template
from ..model.time_estimate import estimate_time
from ..model.validate import internal_error_text, validate
from ..solvers.roles import BACKENDS

SECTIONS = ('all', 'model', 'parameters', 'geometry', 'selections', 'materials', 'physics', 'couplings', 'mesh',
            'solvers', 'study', 'results')
MAX_UNDO_STEPS = 200
MEMORY_BUDGET_TOLERANCE = 0.05   # relative GPU budget / available host RAM change that re-runs validation

# the tree's 'Results' node of the last build / run (not part of the spec; ('results',) is 'Result settings')
COMPUTED_RESULTS_NODE = ('computed_results',)
# group nodes under Result settings (not spec items); derived curves hold one ('derived_plot', plot) node per plot
DERIVED_VALUES_NODE = ('derived_values',)
ANALYTIC_CURVES_NODE = ('analytic_curves',)
DERIVED_CURVES_NODE = ('derived_curves',)
RESULT_GROUP_NODES = (DERIVED_VALUES_NODE, ANALYTIC_CURVES_NODE, DERIVED_CURVES_NODE)
MESHES_NODE = ('meshes',)        # the Meshes group; one ('mesh', name) child per ModelSpec.meshes entry
# StudySpec.estimation under Study: exists (node_exists) only while the study has one; edits are section 'study'
ESTIMATION_NODE = ('estimation',)
_ROOT_NODES =frozenset({('model',), ('parameters',), ('geometry',), ('selections',), ('materials',),
                         ('couplings',), MESHES_NODE, ('solvers',), ('study',), ('results',), COMPUTED_RESULTS_NODE,
                         *RESULT_GROUP_NODES})
# named node kind -> (ModelSpec list attribute, ModelSpec lookup method); features live in their physics
_NAMED_NODES = {'parameter': ('parameters', 'parameter'), 'geometry': ('geometry', 'geometry_object'),
                'selection': ('selections', 'selection'), 'material': ('materials', 'material'),
                'physics': ('physics', 'physics_by_name'), 'coupling': ('couplings', 'coupling'),
                'mesh': ('meshes', 'mesh_by_name'), 'solver': ('solvers', 'solver'),
                'derived': ('derived_values', 'derived_value'), 'curve': ('analytic_curves', 'analytic_curve'),
                'derived_curve': ('derived_curves', 'derived_curve')}
_TREE_NODE_KINDS = frozenset({'geometry', 'selection', 'material', 'physics', 'feature', 'coupling', 'mesh',
                              'solver', 'derived', 'curve', 'derived_curve'})
_PARENT_NODES = {'parameter': ('parameters',), 'geometry': ('geometry',), 'selection': ('selections',),
                 'material': ('materials',), 'physics': ('model',), 'coupling': ('couplings',),
                 'mesh': MESHES_NODE, 'solver': ('solvers',),
                 'derived': DERIVED_VALUES_NODE, 'curve': ANALYTIC_CURVES_NODE, 'derived_curve': DERIVED_CURVES_NODE,
                 'derived_plot': DERIVED_CURVES_NODE}
_SECTION_OF_KIND = {'parameter': 'parameters', 'geometry': 'geometry', 'selection': 'selections',
                    'material': 'materials', 'physics': 'physics', 'feature': 'physics', 'coupling': 'couplings',
                    'mesh': 'mesh', 'solver': 'solvers', 'derived': 'results', 'curve': 'results',
                    'derived_curve': 'results'}
# named elsewhere (renames / deletes rewrite other sections: an 'all' edit); a solver is named by PhysicsSpec.solvers,
# a mesh by PhysicsSpec.mesh
_REFERENCED_KINDS = frozenset({'parameter', 'geometry', 'selection', 'material', 'physics', 'mesh', 'solver'})
_SELECTION_ATTRIBUTES = {'selection': 'entities', 'material': 'selection', 'feature': 'selection',
                         'derived': 'selection'}
_STUDY_FIELDS = frozenset(study_field.name for study_field in fields(StudySpec))
# every of a new physics without a Courant factor (e.g. heat) beside an EM Courant step: one solve per Joule window
_STEPS_PER_SOLVE_BESIDE_COURANT_DT = 500


class ModelDocument(QtCore.QObject):
    """The edited ModelSpec: every edit is one undoable `mutate`; views re-read `spec` on `changed`."""

    changed = QtCore.Signal(str, object)     # section (SECTIONS), origin (the editing view or None)
    replaced = QtCore.Signal()               # new / open (not undo / redo), just before changed('all')
    node_selected = QtCore.Signal(object)    # node path tuple (8.1)
    geometry_group_changed = QtCore.Signal(object)   # names of the shift-click selected objects (() = single selection)
    dirty_changed = QtCore.Signal(bool)
    path_changed = QtCore.Signal(str)        # '' when the model has no file
    undo_state_changed = QtCore.Signal()
    notice = QtCore.Signal(str)              # side effect of an edit worth telling the user (status bar)
    issues_invalidated = QtCore.Signal()     # issues() changes without an edit (a new GPU budget)

    def __init__(self, registry, spec=None):
        super().__init__()
        self.registry = registry
        self.spec = None
        self.path = None
        self.dirty = False
        self.current_node = ('model',)
        self.geometry_group = ()             # >= 2 object names selected together, current_node's among them; else ()
        self.revision = 0
        self._undo_stack = []                # [(label, spec dict before the edit)]
        self._redo_stack = []
        self._saved_snapshot = None          # spec dict at the last open/save/new (dirty = differs from it)
        self._spec_dict = None               # spec.to_dict() of the current revision (spec_dict)
        self._cache = {}
        self._cache_revision = None
        self._in_mutation = False
        self.memory_budget = None            # last GPU budget [bytes] the worker reported (set_memory_budget)
        self.host_memory_budget = None       # last available host RAM [bytes] the worker reported (likewise)
        self.set_spec(spec if spec is not None else ModelSpec())

    # ---- whole-model state ----
    def set_spec(self, spec, path=None):
        """Replace the model (new/open): clears undo/redo, marks it clean, emits replaced then changed('all') and
        selects the root."""
        self._check_not_reentrant()
        if not isinstance(spec, ModelSpec):
            raise TypeError(f'expected a ModelSpec, got {type(spec).__name__}')
        snapshot = spec.to_dict()            # rejects non-JSON content before anything changes
        self._in_mutation = True
        try:
            self.spec = spec
            self._spec_dict = snapshot
            self._saved_snapshot = snapshot
            self._undo_stack.clear()
            self._redo_stack.clear()
            self.revision += 1
            self.current_node = ('model',)   # already valid when changed fires, as after every edit
            self.geometry_group = ()
            self._set_path(path)
            self._set_dirty(False)
            self.replaced.emit()
            self.changed.emit('all', None)
            self.node_selected.emit(self.current_node)
            self.undo_state_changed.emit()
        finally:
            self._in_mutation = False

    def mutate(self, section, fn, label='', origin=None) -> bool:
        """Apply fn(spec) as one undoable edit; False (no signal) when nothing changed. fn raising restores the
        spec and re-raises; calling this from a changed handler raises RuntimeError."""
        return self._apply(section, fn, label, origin, select=None)

    def spec_dict(self) -> dict:
        """spec.to_dict() of the current revision, serialised once per edit: read only, never modify it."""
        return self._spec_dict

    def undo(self):
        """Restore the model before the last edit (keeps the current node if it still exists, else its parent)."""
        self._step_history(self._undo_stack, self._redo_stack)

    def redo(self):
        """Re-apply the last undone edit."""
        self._step_history(self._redo_stack, self._undo_stack)

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo_text(self) -> str:
        """Label of the edit undo would revert ('' when none or unlabelled)."""
        return self._undo_stack[-1][0] if self._undo_stack else ''

    def redo_text(self) -> str:
        """Label of the edit redo would re-apply ('' when none or unlabelled)."""
        return self._redo_stack[-1][0] if self._redo_stack else ''

    def new_model(self, template_key=None):
        """Start an untitled model: empty, or the model.templates entry `template_key`."""
        self.set_spec(make_template(template_key) if template_key is not None else ModelSpec(), path=None)

    def open(self, path):
        """Load a model file (io.model_io.load_model) and make it the document."""
        self.set_spec(load_model(path), path=path)

    def new_from_file(self, path):
        """Start an untitled model from a model file (Examples menu): Save asks for a name, never overwrites it."""
        self.set_spec(load_model(path), path=None)

    def save(self, path=None):
        """Write the model to path (default: the current file) and mark it clean; ValueError when there is no file."""
        target = path if path is not None else self.path
        if target is None:
            raise ValueError('the model has no file yet; choose a file name (Save As)')
        save_model(self.spec, target)
        self._saved_snapshot = self._spec_dict
        self._set_path(target)
        self._set_dirty(False)

    # ---- current node ----
    def select(self, node_path):
        """Make node_path (or its nearest existing ancestor) the current node, alone; node_selected only on a change."""
        self._set_selection(tuple(node_path), ())

    def toggle_geometry_group(self, name):
        """Shift-click: add geometry object `name` to the selected objects (it becomes current) or remove it (the last
        one added of the rest becomes current); a plain select when no geometry object is current."""
        current = tuple(self.current_node or ())
        if len(current) != 2 or current[0] != 'geometry':
            self.select(('geometry', name))
            return
        names = list(self.selected_geometry())
        if name in names:
            names.remove(name)
            if not names:
                return               # the only selected object stays selected
            current = ('geometry', names[-1]) if current[1] == name else current
        else:
            names.append(name)
            current = ('geometry', name)
        self._set_selection(current, tuple(names) if len(names) > 1 else ())

    def select_geometry(self, names, current):
        """Select the geometry objects `names` together (tree Shift+click range), `current` (one of them) current."""
        names = tuple(dict.fromkeys(names))
        if current not in names:
            raise ValueError(f'{current!r} is not among the selected objects')
        self._set_selection(('geometry', current), names if len(names) > 1 else ())

    def selected_geometry(self) -> tuple:
        """Names of the selected geometry objects: the shift-click group, else the current object alone, else ()."""
        if self.geometry_group:
            return self.geometry_group
        current = tuple(self.current_node or ())
        return (current[1],) if len(current) == 2 and current[0] == 'geometry' else ()

    def node_exists(self, node_path) -> bool:
        """True for a root node or a tree node (8.1) whose named items exist in the spec."""
        path = tuple(node_path)
        if path in _ROOT_NODES:
            return True
        if path == ESTIMATION_NODE:
            return self.spec.study.estimation is not None
        if len(path) == 2 and path[0] == 'derived_plot':
            return path[1] in derived_plot_names(self.spec)
        if not path or path[0] not in _TREE_NODE_KINDS:
            return False
        try:
            _locate(self.spec, path)
        except KeyError:
            return False
        return True

    # ---- cached CPU-side checks (per revision) ----
    def parameters(self) -> dict:
        """Parameter values that evaluate (partial mode; failures are in evaluation_errors())."""
        return dict(self._evaluated_parameters()[0])

    def evaluation_errors(self) -> dict:
        """{parameter name: message} of parameters that fail to evaluate."""
        return dict(self._evaluated_parameters()[1])

    def default_mesh_name(self):
        """Name of the default mesh (meshes[0], where physics without a mesh of their own run); None without meshes."""
        return self.spec.meshes[0].name if self.spec.meshes else None

    def mesh_estimate(self, name=None):
        """(MeshEstimate or TriangleMeshEstimate, None) of one mesh (its name; None = the default mesh), or (None,
        message) when it cannot be estimated (CPU only)."""
        name = self.default_mesh_name() if name is None else name

        def estimate():
            if name is None:
                return None, 'the model has no mesh'
            try:
                return estimate_mesh(self.spec, self.parameters(), self.registry, name), None
            except MESH_ESTIMATE_ERRORS as exc:
                return None, error_message(exc)
            except Exception as exc:   # crashing definition callback: traceback to the log, views get a message
                sys.excepthook(*sys.exc_info())
                return None, internal_error_text('the mesh estimate', exc)
        return self._cached(('mesh_estimate', name), estimate)

    def used_mesh_estimates(self):
        """({used mesh name: estimate or None}, {name: failure message or None}) in spec.used_meshes() order."""
        estimates, errors = {}, {}
        for mesh_spec in self.spec.used_meshes():
            estimates[mesh_spec.name], errors[mesh_spec.name] = self.mesh_estimate(mesh_spec.name)
        return estimates, errors

    def memory_estimate(self):
        """meshing.estimate_memory of the cached used-mesh estimates (CPU only), None when a mesh or the estimate
        fails (validation reports why)."""
        return self._memory_estimates()[0]

    def mesh_memory_estimate(self, name):
        """The part of memory_estimate() of one used mesh (its arrays, physics and features; transfers excluded);
        None for an unused mesh or when the estimate fails."""
        return self._memory_estimates()[1].get(name)

    def _memory_estimates(self):
        """(MemoryEstimate or None, {used mesh name: its part}), cached per revision."""
        def estimate():
            estimates, _errors = self.used_mesh_estimates()
            if not estimates or any(mesh is None for mesh in estimates.values()):
                return None, {}
            by_mesh = {}
            try:
                return estimate_memory(self.spec, self.parameters(), self.registry, estimates, by_mesh=by_mesh), by_mesh
            except MESH_ESTIMATE_ERRORS:
                return None, {}
            except Exception:   # crashing estimator: traceback to the log, validation reports it
                sys.excepthook(*sys.exc_info())
                return None, {}
        return self._cached('memory_estimate', estimate)

    def time_estimate(self):
        """(model.time_estimate TimeEstimate or None, message) of the whole study on the cached used-mesh estimates (a
        sweep: each value on its own mesh) and the stored machine profile (CPU), cached per revision; (None, '') when a
        mesh estimate fails."""
        def estimate():
            estimates, _errors = self.used_mesh_estimates()
            if not estimates or any(mesh is None for mesh in estimates.values()):
                return None, ''
            try:
                return estimate_time(self.spec, self.parameters(), self.registry, estimates)
            except MESH_ESTIMATE_ERRORS:
                return None, ''
            except Exception:   # crashing recipe: traceback to the log
                sys.excepthook(*sys.exc_info())
                return None, ''
        return self._cached('time_estimate', estimate)

    def set_memory_budget(self, budget, host_budget=None) -> bool:
        """Keep the worker's GPU budget and available host RAM [bytes] for validation; True (and issues_invalidated)
        when either moved by more than MEMORY_BUDGET_TOLERANCE (or appeared), so the cached issues were dropped."""
        gpu_moved = _budget_moved(self.memory_budget, budget)
        host_moved = _budget_moved(self.host_memory_budget, host_budget)
        if not (gpu_moved or host_moved):
            return False
        if gpu_moved:
            self.memory_budget = int(budget)
        if host_moved:
            self.host_memory_budget = int(host_budget)
        self._cache.pop('issues', None)
        self.issues_invalidated.emit()
        return True

    def issues(self) -> list:
        """validate(spec) findings (Issue list), reusing the cached mesh and memory estimates (or the mesh failure
        message) and checking against the last known GPU budget and available host RAM."""
        def check():
            meshes, mesh_errors = self.used_mesh_estimates()
            return validate(self.spec, self.registry, mesh_estimate=meshes, mesh_error=mesh_errors,
                            memory_budget=self.memory_budget, memory_estimate=self.memory_estimate(),
                            host_budget=self.host_memory_budget)
        return list(self._cached('issues', check))

    def state_names(self) -> frozenset:
        """Physics state-field names usable in material / coupling expressions (keys of state_field_names)."""
        return self._cached('state_names', lambda: frozenset(state_field_names(self.spec, self.registry)))

    # ---- selections ----
    def selection_kind(self, node_path):
        """Entity kind a node selects: its FeatureType.selection_kind, 'domain' for materials and derived values of
        fields, a named selection's kind; None otherwise."""
        path = tuple(node_path)
        if not self.node_exists(path):
            return None
        if path[0] == 'derived' and self.spec.derived_value(path[1]).operation == GLOBAL_DERIVED_OPERATION:
            return None   # reads no fields: nothing to select
        if path[0] == 'material' or (path[0] == 'derived' and self.spec.derived_value(path[1]).source is None):
            return 'domain'
        if path[0] == 'selection':
            return self.spec.selection(path[1]).kind
        if path[0] == 'feature':
            try:
                return self._feature_type(path).selection_kind
            except KeyError:
                return None
        return None

    def selection_of(self, node_path) -> list:
        """Copy of the node's entity list ([] for nodes without a selection or missing nodes)."""
        try:
            owner, attribute = _selection_owner(self.spec, tuple(node_path))
        except KeyError:
            return []
        return list(getattr(owner, attribute))

    def toggle_selection_entity(self, node_path, entity):
        """Add entity to the node's selection or remove it (single-entity features: replace); one undoable edit."""
        path = tuple(node_path)
        _selection_owner(self.spec, path)    # KeyError for nodes without a selection
        if path[0] == 'selection' and entity == path[1]:
            raise ValueError(f'selection {entity!r} cannot contain itself')
        try:
            single_entity = path[0] == 'feature' and self._feature_type(path).max_entities == 1
        except KeyError:
            single_entity = False

        def toggle(spec):
            owner, attribute = _selection_owner(spec, path)
            entities = list(getattr(owner, attribute))
            if entity in entities:
                entities = [other for other in entities if other != entity]
            elif single_entity:
                entities = [entity]
            else:
                entities.append(entity)
            setattr(owner, attribute, entities)
        self.mutate(_SECTION_OF_KIND[path[0]], toggle, label=f'Toggle {entity}')

    # ---- node operations (one edit each; return the new name) ----
    def add_parameter(self) -> str:
        """Append parameter p<N> = 0 and select the parameters table."""
        name = self._new_name('p')
        self._apply('parameters', lambda spec: spec.parameters.append(ParameterSpec(name, 0.0)),
                    'Add parameter', None, select=('parameters',))
        return name

    def add_geometry(self, kind, params=None) -> str:
        """Add a geometry object (params=None -> a shape centred in the domain) and select it."""
        geometry_kind = GEOMETRY_KINDS.get(kind)
        if geometry_kind is None:
            raise ValueError(f'unknown geometry kind {kind!r}; valid: {", ".join(GEOMETRY_KINDS)}')
        unknown = sorted(set(params or {}) - {param.name for param in geometry_kind.params})
        if unknown:
            raise ValueError(f'{kind}: unknown parameters {unknown}; valid: '
                             f'{", ".join(param.name for param in geometry_kind.params)}')
        values = fill_defaults(geometry_kind.params,
                               self._centred_shape_params(kind) if params is None else dict(params))
        values = {key: value for key, value in values.items()   # unset 3D params stay out of the file
                  if not (key in Z_PARAM_NAMES and not z_param_set(key, value))}
        name = self._new_name(kind)
        self._apply('geometry', lambda spec: spec.geometry.append(GeometrySpec(name, kind, values)),
                    f'Add {geometry_kind.label.lower()}', None, select=('geometry', name))
        return name

    def combine_geometry(self, kind, names) -> str:
        """Add a union / intersection of the domain objects `names` (difference: the first minus the rest) and select
        it; one edit."""
        names = list(names)
        if kind not in ('union', 'intersection', 'difference'):
            raise ValueError(f'{kind!r} does not combine objects; valid: union, intersection, difference')
        for name in names:
            if GEOMETRY_KINDS[self.spec.geometry_object(name).kind].entity_kind != 'domain':
                raise ValueError(f'{name} is not a domain object: only domain objects can be combined')
        params = ({'base': names[0], 'subtract': names[1:]} if kind == 'difference' else {'inputs': names})
        values = fill_defaults(GEOMETRY_KINDS[kind].params, params)
        name = self._new_name(kind)
        self._apply('geometry', lambda spec: spec.geometry.append(GeometrySpec(name, kind, values)),
                    f'{GEOMETRY_KINDS[kind].label} of {", ".join(names)}', None, select=('geometry', name))
        return name

    def delete_geometry(self, names):
        """Delete several geometry objects and the references that would dangle; one edit."""
        names = list(names)
        for name in names:
            self.spec.geometry_object(name)   # KeyError for unknown objects

        def delete(spec):
            for name in names:
                spec.remove_geometry(name, self.registry)
        self.mutate('all', delete, label=f'Delete {", ".join(names)}')

    def add_selection(self, kind) -> str:
        """Add an empty named selection of entity kind `kind` and select it."""
        if kind not in SELECTION_KINDS:
            raise ValueError(f'unknown selection kind {kind!r}; valid: {", ".join(SELECTION_KINDS)}')
        name = self._new_name('selection')
        side = INTERFACE_SIDES[0] if kind == 'interface' else ''
        self._apply('selections', lambda spec: spec.selections.append(SelectionSpec(name, kind, [], side)),
                    'Add selection', None, select=('selection', name))
        return name

    def add_material(self, library=None) -> str:
        """Add a material (blank, or inheriting library material `library`) with an empty selection; select it."""
        if library is not None and library not in MATERIAL_LIBRARY:
            raise ValueError(f'unknown library material {library!r}; available: {", ".join(MATERIAL_LIBRARY)}')
        name = self._new_name(library or 'material', preferred=library)
        self._apply('materials', lambda spec: spec.materials.append(MaterialSpec(name, library=library)),
                    'Add material', None, select=('material', name))
        return name

    def add_physics(self, type_name) -> str:
        """Add a physics with default settings in the space `schema.new_physics_space` picks (its space_settings
        applied; a space of the other dimension switches the domain: size_z = size_x, or cleared with the objects' z
        params and the stored cuts); the first physics also sets the study to its default_study. Whichever is added second, a physics
        without a Courant factor beside one stepping on the Courant dt solves every 500 steps (existing ones only while
        still at every 1). One edit; dimension switches and slowed physics are reported through `notice`."""
        physics_type = self.registry.physics_type(type_name)
        instance_name = physics_type.default_instance_name
        name = self._new_name(instance_name, preferred=instance_name)
        space, settings = new_physics_space(physics_type, fill_defaults(physics_type.settings, {}), self.spec,
                                            self.registry)
        switch_dimension = space is not None and (space == '3d') != self.spec.is_3d
        study = (self._default_study(physics_type)
                 if not self.spec.physics and physics_type.default_study else None)
        # MeshEstimate.dt: set when a physics on that mesh steps on the Courant dt (the runner's dt spans the meshes)
        meshes = list(self.used_mesh_estimates()[0].values())
        meshes_known = bool(meshes) and all(mesh is not None for mesh in meshes)
        courant_dt = meshes_known and any(mesh.dt is not None for mesh in meshes)
        new_declares_courant = _declares_courant_factor(physics_type, settings)
        every = _STEPS_PER_SOLVE_BESIDE_COURANT_DT if courant_dt and not new_declares_courant else 1
        slowed_names = (self._non_courant_physics_at_every_one()
                        if new_declares_courant and meshes_known and not courant_dt else [])

        def add(spec):
            if switch_dimension and space == '3d':
                spec.domain.size_z = spec.domain.size_x
            elif switch_dimension:
                spec.domain.size_z = None
                spec.study.stored_cuts = []   # 3D only
                for obj in spec.geometry:
                    for z_name in Z_PARAM_NAMES:
                        obj.params.pop(z_name, None)
            for slowed_name in slowed_names:
                spec.physics_by_name(slowed_name).every = _STEPS_PER_SOLVE_BESIDE_COURANT_DT
            spec.physics.append(PhysicsSpec(name, type_name, settings, every=every))
            if study is not None:
                spec.study = study
        self._apply('all' if study is not None or switch_dimension else 'physics', add, f'Add {physics_type.label}',
                    None, select=('physics', name))
        notices = []
        if switch_dimension:
            notices.append("model switched to 3D ('Depth (z, 3D only)' = 'Width (x)')" if space == '3d'
                           else "model switched to 2D ('Depth (z, 3D only)', object z values and stored cut planes "
                                'cleared)')
        if slowed_names:
            notices.append(f'{", ".join(slowed_names)}: solve interval (every) set to '
                           f'{_STEPS_PER_SOLVE_BESIDE_COURANT_DT} Courant steps beside {name}')
        if notices:
            self.notice.emit(f"{'; '.join(notices)} (Undo reverts the whole addition)")
        return name

    def add_feature(self, physics, type_name, create_object=True) -> str:
        """Add a feature with default params (a 'channel' param: the physics' first harmonic channel);
        point/line/single-rectangle features get a new centred object."""
        physics_spec = self.spec.physics_by_name(physics)
        feature_type = self.registry.physics_type(physics_spec.type).feature_type(type_name)
        object_kind = _default_object_kind(feature_type) if create_object else None
        object_name = self._new_name(object_kind) if object_kind is not None else None
        object_params = (fill_defaults(GEOMETRY_KINDS[object_kind].params, self._centred_shape_params(object_kind))
                         if object_kind is not None else None)
        name = self._new_name(type_name, also_taken=(object_name,))
        params = fill_defaults(feature_type.params, {})
        channels = harmonic_channel_names(physics_spec)
        if channels and params.get('channel') == '':   # em_fdfd_nonlinear source / monitor: the first channel
            params['channel'] = channels[0]

        def add(spec):
            if object_kind is not None:
                spec.geometry.append(GeometrySpec(object_name, object_kind, object_params))
            selection = [object_name] if object_name is not None else []
            spec.physics_by_name(physics).features.append(FeatureSpec(name, type_name, selection, params))
        self._apply('all' if object_kind is not None else 'physics', add, f'Add {feature_type.label}', None,
                    select=('feature', physics, name))
        return name

    def add_coupling(self, type_name, source, target) -> str:
        """Add a coupling between two physics (types checked against the CouplingType) and select it."""
        coupling_type = self.registry.coupling_type(type_name)
        for role, physics_name in (('source', source), ('target', target)):
            physics_type_name = self.spec.physics_by_name(physics_name).type
            if not coupling_type_accepts(coupling_type, role, physics_type_name):
                raise ValueError(f'{coupling_type.label}: the {role} must be a '
                                 f'{coupling_role_text(coupling_type, role)} physics; '
                                 f'{physics_name!r} is {physics_type_name}')
        name = self._new_name(type_name)
        params = fill_defaults(coupling_type.params, {})
        self._apply('couplings',
                    lambda spec: spec.couplings.append(CouplingSpec(name, type_name, source, target, params=params)),
                    f'Add {coupling_type.label}', None, select=('coupling', name))
        return name

    def add_solver(self, type_name) -> str:
        """Add a solver node of SolverType type_name (default settings, the Solvers precision when it offers it), assign
        it in the same edit to every (physics, role) still on Automatic that it fits in this study and select it."""
        solver_type = self.registry.solver_type(type_name)
        name = self._new_name(type_name)
        settings = fill_defaults(solver_type.params, with_default_precision(self.spec, solver_type, {}))
        assigned = {physics.name: physics.solvers for physics in self.spec.physics}
        assignments = [(physics_name, role) for physics_name, role, reason
                       in solver_fits(self.spec, self.registry, solver_type, settings)
                       if not reason and isinstance(assigned[physics_name], dict)
                       and not assigned[physics_name].get(role)]

        def add(spec):
            spec.solvers.append(SolverSpec(name, type_name, settings))
            for physics_name, role in assignments:
                spec.physics_by_name(physics_name).solvers[role] = name
        self._apply('all' if assignments else 'solvers', add, f'Add {solver_type.label}', None,
                    select=('solver', name))
        if assignments:
            used_by = ', '.join(f'{physics_name} ({SOLVER_ROLES[role].lower()})' for physics_name, role in assignments)
            self.notice.emit(f'{name} now solves {used_by} (Undo reverts the whole addition)')
        return name

    def set_physics_solver(self, physics_name, role, solver_name):
        """Assign solver node solver_name to the role of a physics; None returns the role to Automatic."""
        if solver_name is not None:
            self.spec.solver(solver_name)    # KeyError for an unknown node

        def assign(spec):
            solvers = spec.physics_by_name(physics_name).solvers
            if not isinstance(solvers, dict):
                raise ValueError(f"physics '{physics_name}': solvers must be a mapping; fix the model file")
            if solver_name is None:
                solvers.pop(role, None)
            else:
                solvers[role] = solver_name
        self.mutate('physics', assign, label=f'Set {physics_name} {role} solver')

    def add_mesh(self, kind='tensor', physics=None) -> str:
        """Add a mesh of MeshSpec kind `kind` in automatic mode (COMSOL's physics-controlled default; tetrahedra:
        mesher gmsh when it is installed) and select it; with `physics` it is named '<physics>_mesh' and that physics
        moves onto it in the same edit ('New mesh for this physics')."""
        from ..fem.gmsh_io import find_gmsh

        if kind not in MESH_KINDS:
            raise ValueError(f"unknown mesh kind {kind!r}; valid: {', '.join(MESH_KINDS)}")
        if physics is not None:
            self.spec.physics_by_name(physics)   # KeyError for an unknown physics
        name = self._new_mesh_name('mesh', preferred=f'{physics}_mesh' if physics is not None else None)
        mesher = 'gmsh' if kind == 'tetrahedra' and find_gmsh() is not None else 'builtin'

        def add(spec):
            spec.meshes.append(MeshSpec(name=name, kind=kind, mode='auto', mesher=mesher))
            if physics is not None:
                spec.physics_by_name(physics).mesh = name
        label = f'New mesh for {physics}' if physics is not None else 'Add mesh'
        self._apply('all' if physics is not None else 'mesh', add, label, None, select=('mesh', name))
        return name

    def set_physics_mesh(self, physics_name, mesh_name):
        """Run a physics on the mesh mesh_name (stored as '' when that is the default mesh, meshes[0])."""
        self.spec.mesh_by_name(mesh_name)   # KeyError for an unknown mesh

        def assign(spec):
            spec.physics_by_name(physics_name).mesh = '' if mesh_name == spec.meshes[0].name else mesh_name
        self.mutate('physics', assign, label=f'Set {physics_name} mesh')

    def delete_mesh(self, name, move_physics=False):
        """Delete a mesh; move_physics first moves the physics on it to the default mesh that remains (one edit),
        else a used mesh is refused (ModelSpec.remove_mesh)."""
        self.spec.mesh_by_name(name)   # KeyError for an unknown mesh

        def delete(spec):
            if move_physics:
                remaining = [mesh_spec.name for mesh_spec in spec.meshes if mesh_spec.name != name]
                if not remaining:
                    raise ValueError('the model needs at least one mesh')
                target = '' if spec.meshes[0].name != name else remaining[0]   # '' = the default mesh
                for physics in spec.physics_on(name):
                    physics.mesh = target
            spec.remove_mesh(name)
        self.mutate('all', delete, label=f'Delete {name}')

    def keep_adapted_meshes(self, mesh_dicts) -> list:
        """'Keep adapted mesh' (S7.7 D15): add each MeshSpec dict of a convergence report's adapted_meshes as
        '<name>_adapted', move the physics on mesh <name> onto it and rename it in the study's adaptation meshes; one
        edit, the first new mesh selected. Returns the new names; KeyError when a refined mesh is gone."""
        adapted = ModelSpec.from_dict({'meshes': copy.deepcopy(list(mesh_dicts or []))}).meshes
        if not adapted:
            raise ValueError('the last run has no adapted mesh')
        taken = {mesh_spec.name for mesh_spec in self.spec.meshes}
        renames = {}
        for mesh_spec in adapted:
            self.spec.mesh_by_name(mesh_spec.name)   # KeyError: renamed or deleted since the run
            base = new_name = f'{mesh_spec.name}_adapted'
            index = 1
            while new_name in taken:
                new_name, index = f'{base}{index}', index + 1
            taken.add(new_name)
            renames[mesh_spec.name] = new_name

        def keep(spec):
            for mesh_spec in adapted:
                for physics in spec.physics_on(mesh_spec.name):
                    physics.mesh = renames[mesh_spec.name]
                new_mesh = copy.deepcopy(mesh_spec)
                new_mesh.name = renames[mesh_spec.name]
                spec.meshes.append(new_mesh)
            adaptation = spec.study.adaptation
            if adaptation is not None and isinstance(adaptation.meshes, list):
                adaptation.meshes = [renames.get(name, name) for name in adaptation.meshes]
        new_names = list(renames.values())
        self._apply('all', keep, 'Keep adapted mesh', None, select=('mesh', new_names[0]))
        refinement = (' The study still refines further from it; switch Mesh refinement off in the Study settings to '
                      'run on it as it is.' if self.spec.study.adaptation is not None else '')
        self.notice.emit(f"Added {', '.join(new_names)}: the physics of the refined meshes now run on "
                         f'{"it" if len(new_names) == 1 else "them"} (Undo reverts it).{refinement}')
        return new_names

    def add_derived_value(self) -> str:
        """Add a derived value (mean of the first physics' primary field, else of 1) and select it."""
        name = self._new_name('derived')
        expression = self._default_derived_expression()
        self._apply('results', lambda spec: spec.derived_values.append(DerivedValueSpec(name, expression)),
                    'Add derived value', None, select=('derived', name))
        return name

    def add_analytic_curve(self) -> str:
        """Add an analytic curve (on the first 1D feature result, else the sweep, else a line) and select it."""
        name = self._new_name('curve')
        curve = self._default_analytic_curve(name)
        self._apply('results', lambda spec: spec.analytic_curves.append(curve), 'Add analytic curve', None,
                    select=('curve', name))
        return name

    def add_derived_curve(self, plot='') -> str:
        """Add a derived curve (expression '0', no inputs yet) to plot ('' = a plot of its own) and select it."""
        name = self._new_name('dcurve')
        self._apply('results', lambda spec: spec.derived_curves.append(DerivedCurveSpec(name, '0', plot=plot)),
                    'Add derived curve', None, select=('derived_curve', name))
        return name

    def rename_derived_plot(self, old, new):
        """Move every derived curve of plot `old` (plot or, when unset, name) to plot `new` (an identifier)."""
        if not is_identifier(new):
            raise ValueError(f'{new!r} is not a valid plot name (letters, digits and _, not starting with a digit)')
        if new == old:
            return

        def rename(spec):
            for curve in spec.derived_curves:
                if (curve.plot or curve.name) == old:
                    curve.plot = new
        self._apply('results', rename, f'Rename plot {old}', None, select=('derived_plot', new))

    def delete_node(self, path):
        """Delete a named node and the references that would dangle (entity lists, couplings, stored fields)."""
        path = tuple(path)
        _locate(self.spec, path)             # KeyError for unknown / root nodes
        kind, name = path[0], path[-1]

        def delete(spec):
            if kind == 'geometry':
                spec.remove_geometry(name, self.registry)
            elif kind == 'selection':
                spec.remove_selection(name)
            elif kind == 'physics':
                spec.remove_physics(name, self.registry)
            elif kind == 'solver':
                spec.remove_solver(name)
            elif kind == 'mesh':
                spec.remove_mesh(name)
            else:
                items, item = _locate(spec, path)
                del items[_index_of(items, item)]
        section = 'all' if kind in ('geometry', 'selection', 'physics', 'solver', 'mesh') else _SECTION_OF_KIND[kind]
        self.mutate(section, delete, label=f'Delete {name}')

    def duplicate_node(self, path) -> str:
        """Insert a copy of a named node right after it (name numbered from the original) and select it; a feature's
        own object (_owned_object) is copied with it, a derived curve stays in the original's plot."""
        path = tuple(path)
        _, original = _locate(self.spec, path)
        name = (self._new_mesh_name(_name_prefix(original.name)) if path[0] == 'mesh'
                else self._new_name(_name_prefix(original.name)))
        owned_object = self._owned_object(path) if path[0] == 'feature' else None
        object_name = (self._new_name(_name_prefix(owned_object), also_taken=(name,))
                       if owned_object is not None else None)

        def duplicate(spec):
            items, item = _locate(spec, path)
            clone = copy.deepcopy(item)
            clone.name = name
            if object_name is not None:   # the copy gets its own object instead of sharing (moving with) the original's
                obj = spec.geometry_object(owned_object)
                object_clone = copy.deepcopy(obj)
                object_clone.name = object_name
                spec.geometry.insert(_index_of(spec.geometry, obj) + 1, object_clone)
                clone.selection = [object_name]
            if path[0] == 'derived_curve' and not item.plot:   # its plot is its name: name it, as Add does
                item.plot = clone.plot = item.name
            items.insert(_index_of(items, item) + 1, clone)
        new_path = ('parameters',) if path[0] == 'parameter' else path[:-1] + (name,)
        section = 'all' if object_name is not None else _SECTION_OF_KIND[path[0]]
        self._apply(section, duplicate, f'Duplicate {original.name}', None, select=new_path)
        return name

    def _owned_object(self, path):
        """Object name a feature alone uses: its only entity, of the kind add_feature creates for its type, that no other
        entity list, composite, refinement or reference names; else None."""
        physics_spec = self.spec.physics_by_name(path[1])
        feature = physics_spec.feature(path[2])
        if len(feature.selection) != 1:
            return None
        object_name = feature.selection[0]
        try:
            obj = self.spec.geometry_object(object_name)
            feature_type = self.registry.physics_type(physics_spec.type).feature_type(feature.type)
        except KeyError:
            return None
        if obj.kind != _default_object_kind(feature_type):
            return None
        without_feature = self.spec.copy()
        features = without_feature.physics_by_name(path[1]).features
        features[:] = [other for other in features if other.name != feature.name]
        stripped = without_feature.copy()
        stripped.remove_geometry(object_name, self.registry)   # strips every other reference too
        without_feature.geometry = [other for other in without_feature.geometry if other.name != object_name]
        return object_name if stripped.to_dict() == without_feature.to_dict() else None

    def rename_node(self, path, new_name):
        """Rename a node and every reference to it (ValueError for an invalid or taken name); ('model',) sets
        the title. ('parameter', name) renames a parameter."""
        path = tuple(path)
        if path == ('model',):
            self.mutate('model', lambda spec: setattr(spec, 'title', str(new_name)), label='Rename model')
            return
        if path[0] == 'derived_plot':
            self.rename_derived_plot(path[1], new_name)
            return
        _locate(self.spec, path)
        kind, old_name = path[0], path[-1]
        if new_name == old_name:
            return
        spec_renames = {
            'parameter': lambda spec: spec.rename_parameter(old_name, new_name, self.registry),
            'geometry': lambda spec: spec.rename_geometry(old_name, new_name, self.registry),
            'selection': lambda spec: spec.rename_selection(old_name, new_name),
            'material': lambda spec: spec.rename_material(old_name, new_name, self.registry),
            'physics': lambda spec: spec.rename_physics(old_name, new_name, self.registry),
            'mesh': lambda spec: spec.rename_mesh(old_name, new_name),
            'solver': lambda spec: spec.rename_solver(old_name, new_name),
        }
        rename = spec_renames.get(kind) or (lambda spec: _rename_unreferenced(spec, path, new_name))
        section = 'all' if kind in _REFERENCED_KINDS or kind == 'feature' else _SECTION_OF_KIND[kind]
        self._apply(section, rename, f'Rename {old_name}', None,
                    select=_renamed_path(self.current_node, path, new_name))

    def set_enabled(self, path, flag):
        """Enable or disable a feature or coupling."""
        path = tuple(path)
        if not path or path[0] not in ('feature', 'coupling'):
            raise ValueError(f'only features and couplings can be enabled or disabled, not {path!r}')
        _locate(self.spec, path)
        enabled = bool(flag)

        def set_flag(spec):
            _locate(spec, path)[1].enabled = enabled
        self.mutate(_SECTION_OF_KIND[path[0]], set_flag, label=f"{'Enable' if enabled else 'Disable'} {path[-1]}")

    def move_node(self, path, delta):
        """Move a named node `delta` places within its list (clamped; order matters for materials and physics); a
        derived curve moves among the curves of its plot (the tree's siblings), which keep their list positions; a mesh
        move first names the default mesh in the physics on it (the first mesh is the default)."""
        path = tuple(path)
        current_items, current_item = _locate(self.spec, path)
        steps = int(delta)
        current_positions = _sibling_positions(current_items, current_item, path[0])
        current_index = _index_of([current_items[position] for position in current_positions], current_item)
        if min(max(current_index + steps, 0), len(current_positions) - 1) == current_index:
            return   # a clamped no-op: no edit (a mesh move would still rename the default mesh in its physics)

        def move(spec):
            if path[0] == 'mesh':   # physics on the default mesh stay on it when another mesh becomes the first
                for physics in spec.physics:
                    if not physics.mesh:
                        physics.mesh = spec.meshes[0].name
            items, item = _locate(spec, path)
            positions = _sibling_positions(items, item, path[0])
            members = [items[position] for position in positions]
            index = _index_of(members, item)
            members.insert(min(max(index + steps, 0), len(members) - 1), members.pop(index))
            for position, member in zip(positions, members):
                items[position] = member
        self.mutate('all' if path[0] == 'mesh' else _SECTION_OF_KIND[path[0]], move, label=f'Move {path[-1]}')

    # ---- internals ----
    def _apply(self, section, fn, label, origin, select):
        """Body of mutate; `select` = node made current after a successful edit (None: keep / repair current)."""
        self._check_not_reentrant()
        if section not in SECTIONS:
            raise ValueError(f'unknown section {section!r}; valid: {", ".join(SECTIONS)}')
        self._in_mutation = True
        try:
            before = self._spec_dict         # every edit goes through here, so it is the spec's dict
            try:
                fn(self.spec)
                after = self.spec.to_dict()
            except BaseException:
                self.spec = ModelSpec.from_dict(before)
                raise
            if after == before:
                return False
            self._undo_stack.append((label, before))
            del self._undo_stack[:-MAX_UNDO_STEPS]
            self._redo_stack.clear()
            self._spec_dict = after
            self._next_revision(before, after)
            self._set_dirty(after != self._saved_snapshot)
            self._publish_change(section, origin, select if select is not None else self.current_node)
            return True
        finally:
            self._in_mutation = False

    def _step_history(self, source_stack, target_stack):
        """Pop a snapshot from source_stack (undo or redo), push the current state onto target_stack, restore it."""
        self._check_not_reentrant()
        if not source_stack:
            return
        label, snapshot = source_stack[-1]
        restored = ModelSpec.from_dict(snapshot)
        source_stack.pop()
        target_stack.append((label, self._spec_dict))
        self._in_mutation = True
        try:
            self.spec = restored
            before, self._spec_dict = self._spec_dict, snapshot
            self._next_revision(before, snapshot)
            self._set_dirty(snapshot != self._saved_snapshot)
            self._publish_change('all', None, self.current_node)
        finally:
            self._in_mutation = False

    def _publish_change(self, section, origin, wanted_node):
        """Repair the current node, emit changed, then node_selected (if the node moved) and undo_state_changed."""
        previous_node = self.current_node
        self.current_node = self._existing_node(wanted_node)
        previous_group = self.geometry_group
        group = tuple(name for name in previous_group if self.node_exists(('geometry', name)))
        current = tuple(self.current_node)
        keep_group = len(group) > 1 and len(current) == 2 and current[0] == 'geometry' and current[1] in group
        self.geometry_group = group if keep_group else ()
        self.changed.emit(section, origin)
        if self.current_node != previous_node:
            self.node_selected.emit(self.current_node)
        if self.geometry_group != previous_group:
            self.geometry_group_changed.emit(self.geometry_group)
        self.undo_state_changed.emit()

    def _check_not_reentrant(self):
        if self._in_mutation:
            raise RuntimeError('ModelDocument edits are not re-entrant: defer edits made from a changed / '
                               'node_selected handler with QTimer.singleShot(0, ...)')

    def _set_current(self, path):
        node = self._existing_node(path)
        if node != self.current_node:
            self.current_node = node
            self.node_selected.emit(node)

    def _set_selection(self, path, group):
        """Current node + geometry group; the group is set first, so node_selected handlers already see it."""
        previous_group = self.geometry_group
        self.geometry_group = group
        self._set_current(path)
        if group != previous_group:
            self.geometry_group_changed.emit(group)

    def _existing_node(self, path):
        """path, or its nearest ancestor that exists (('model',) always does)."""
        while not self.node_exists(path):
            path = parent_node(path)
        return path

    def _set_path(self, path):
        new_path = os.fspath(path) if path is not None else None
        if new_path != self.path:
            self.path = new_path
            self.path_changed.emit(new_path or '')

    def _set_dirty(self, flag):
        if flag != self.dirty:
            self.dirty = flag
            self.dirty_changed.emit(flag)

    def _next_revision(self, before, after):
        """Bump the revision; a post-processing edit (equal results_inputs) carries every cached estimate and the
        parameters over, only the validation issues are recomputed."""
        keep = self._cache_revision == self.revision and results_inputs(before) == results_inputs(after)
        self.revision += 1
        if keep:
            self._cache.pop('issues', None)
            self._cache_revision = self.revision

    def _cached(self, key, compute):
        """compute() memoised until the next revision (_next_revision keeps the estimates of post-processing edits)."""
        if self._cache_revision != self.revision:
            self._cache.clear()
            self._cache_revision = self.revision
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def _evaluated_parameters(self):
        def evaluate():
            errors = {}
            values = evaluate_parameters(self.spec.parameters, errors)
            return values, errors
        return self._cached('parameters', evaluate)

    def _feature_type(self, path):
        """FeatureType of a ('feature', physics, name) node; KeyError when the node or its type is unknown."""
        if len(path) != 3 or path[0] != 'feature':
            raise KeyError(f'{path!r} is not a feature node')
        physics = self.spec.physics_by_name(path[1])
        return self.registry.physics_type(physics.type).feature_type(physics.feature(path[2]).type)

    def _new_name(self, prefix, preferred=None, also_taken=()):
        """`preferred` when free, else prefix + the smallest free integer suffix; avoids every node name, reserved
        expression / entity names and physics field names."""
        taken = (self.spec.all_names() | RESERVED_NAMES | set(RESERVED_ENTITY_NAMES)
                 | set(state_field_names(self.spec, self.registry))
                 | set(output_field_names(self.spec, self.registry))
                 | set(scalar_output_names(self.spec, self.registry)) | set(also_taken))
        if preferred is not None and preferred not in taken and is_identifier(preferred):
            return preferred
        index = 1
        while f'{prefix}{index}' in taken:
            index += 1
        return f'{prefix}{index}'

    def _new_mesh_name(self, prefix, preferred=None):
        """`preferred` when free, else prefix + the smallest free integer suffix (mesh names are their own namespace)."""
        taken = {mesh_spec.name for mesh_spec in self.spec.meshes}
        if preferred is not None and preferred not in taken and is_identifier(preferred):
            return preferred
        index = 1
        while f'{prefix}{index}' in taken:
            index += 1
        return f'{prefix}{index}'

    def _domain_box(self):
        """(x0, y0, width, height) [m] of the domain in the drawing plane (geometry.domain_extent: a polar sector's
        box); the DomainSpec default size at the origin when it fails."""
        try:
            x0, x1, y0, y1 = domain_extent(self.spec, self.parameters())
            return x0, y0, x1 - x0, y1 - y0
        except ValueError:
            defaults = DomainSpec()
            return 0.0, 0.0, defaults.size_x, defaults.size_y

    def _centred_shape_params(self, kind):
        """Params of a `kind` object centred in the domain: the middle half of each side, lines horizontal over
        the middle 80%; {} for composites."""
        x0, y0, size_x, size_y = self._domain_box()
        center_x, center_y = x0 + size_x / 2, y0 + size_y / 2
        half_width_x, half_width_y = size_x / 4, size_y / 4
        radius = min(size_x, size_y) / 4
        shapes = {
            'rectangle': {'x0': center_x - half_width_x, 'y0': center_y - half_width_y,
                          'x1': center_x + half_width_x, 'y1': center_y + half_width_y},
            'circle': {'cx': center_x, 'cy': center_y, 'r': radius},
            'ellipse': {'cx': center_x, 'cy': center_y, 'a': radius, 'b': radius / 2, 'angle': 0.0},
            'polygon': {'points': [[center_x - half_width_x, center_y - half_width_y],
                                   [center_x + half_width_x, center_y - half_width_y],
                                   [center_x, center_y + half_width_y]]},
            'point': {'x': center_x, 'y': center_y},
            'line': {'x0': x0 + 0.1 * size_x, 'y0': center_y, 'x1': x0 + 0.9 * size_x, 'y1': center_y},
        }
        shapes['rounded_rectangle'] = {**shapes['rectangle'], 'r': min(half_width_x, half_width_y) / 2}
        size_z = self._domain_size_z()
        if size_z is not None:   # 3D-only kinds: the middle half of the height
            center_z, half_height = size_z / 2, size_z / 4
            radius = min(radius, half_height)
            shapes.update({
                'box': {**shapes['rectangle'], 'z0': center_z - half_height, 'z1': center_z + half_height},
                'cylinder_z': {**shapes['circle'], 'z0': center_z - half_height, 'z1': center_z + half_height},
                'sphere': {'cx': center_x, 'cy': center_y, 'cz': center_z, 'r': radius}})
        return {name: _rounded(value) for name, value in shapes.get(kind, {}).items()}

    def _domain_size_z(self):
        """geometry.domain_size_z [m] of a 3D model; None for 2D or when it fails."""
        try:
            return domain_size_z(self.spec, self.parameters())
        except ValueError:
            return None

    def _non_courant_physics_at_every_one(self) -> list:
        """Names of physics still at every 1 whose resolved settings declare no Courant factor (unresolvable skipped)."""
        names = []
        for physics_spec in self.spec.physics:
            if physics_spec.every != 1:
                continue
            try:
                physics_type = self.registry.physics_type(physics_spec.type)
                settings = physics_settings(physics_type, physics_spec, self.parameters())
                if not _declares_courant_factor(physics_type, settings):
                    names.append(physics_spec.name)
            except MESH_ESTIMATE_ERRORS:    # unknown type / bad settings: validation reports them
                continue
        return names

    def _default_study(self, physics_type):
        """StudySpec from physics_type.default_study, settings completed with the study type's defaults."""
        study_values = copy.deepcopy(physics_type.default_study)
        unknown = sorted(set(study_values) - _STUDY_FIELDS)
        if unknown:
            raise ValueError(f"physics type '{physics_type.name}': default_study has unknown keys {unknown}")
        study_type_name = study_values.get('type', StudySpec().type)
        try:
            study_params = self.registry.study_type(study_type_name).params
        except KeyError:                     # reported by validation
            study_params = ()
        sweep = study_values.get('sweep')
        return StudySpec(type=study_type_name,
                         settings=fill_defaults(study_params, study_values.get('settings', {})),
                         stored_fields=list(study_values.get('stored_fields', [])),
                         sweep=SweepSpec(**sweep) if sweep is not None else None)

    def result_features(self, kinds) -> list:
        """[('physics.feature', result kind)] of the features (disabled ones too) whose result kind is in kinds, in
        model order."""
        entries = []
        for physics_spec in self.spec.physics:
            try:
                physics_type = self.registry.physics_type(physics_spec.type)
            except KeyError:
                continue
            for feature in physics_spec.features:
                try:
                    result_kind = physics_type.feature_type(feature.type).result_kind
                except KeyError:
                    continue
                if result_kind in kinds:
                    entries.append((f'{physics_spec.name}.{feature.name}', result_kind))
        return entries

    def _default_analytic_curve(self, name):
        """Curve '0' on the first enabled feature with a curve result, else the sweep, else a horizontal line through
        the domain middle profiling the first primary field."""
        enabled_keys = {f'{physics.name}.{feature.name}' for physics in self.spec.physics
                        for feature in physics.features if feature.enabled}
        for key, _kind in self.result_features(CURVE_RESULT_KINDS):
            if key in enabled_keys:
                return AnalyticCurveSpec(name, key, '0')
        if self.spec.study.sweep is not None:
            return AnalyticCurveSpec(name, 'sweep', '0')
        line, field_key = self.default_line_target()
        return AnalyticCurveSpec(name, 'line', '0', line=line, field=field_key)

    def default_line_target(self):
        """(line [x0, y0, x1, y1] across the middle 80% of the domain at mid height [m], 'physics.component' of the
        first primary output field or '') of a new line curve."""
        x0, y0, size_x, size_y = self._domain_box()
        line = [_rounded(value) for value in (x0 + 0.1 * size_x, y0 + 0.5 * size_y, x0 + 0.9 * size_x,
                                              y0 + 0.5 * size_y)]
        for physics_spec in self.spec.physics:
            try:
                primary = primary_field_info(self.registry.physics_type(physics_spec.type).output_fields)
            except KeyError:
                continue
            if primary is not None:
                return line, f'{physics_spec.name}.{primary.name}'
        return line, ''

    def _default_derived_expression(self):
        """Primary output field of the first registered physics (plain name when unique), else '1'."""
        field_names = output_field_names(self.spec, self.registry)
        for physics_spec in self.spec.physics:
            try:
                primary = primary_field_info(self.registry.physics_type(physics_spec.type).output_fields)
            except KeyError:
                continue
            if primary is None:
                continue
            if field_names.get(primary.name) == (physics_spec.name, primary.name):
                return primary.name
            return f'{physics_spec.name}_{primary.name}'
        return '1'


# ---- module helpers ----
def results_inputs(spec_dict):
    """Spec dict content a build / run (and the mesh / memory / time estimates) depends on: everything but title,
    description (free text), the derived values, analytic and derived curves (post-processing of a session); a new
    top-level dict, spec_dict unchanged."""
    return {key: value for key, value in spec_dict.items()
            if key not in ('title', 'description', 'derived_values', 'analytic_curves', 'derived_curves')}


def _budget_moved(old, new):
    """True when a reported budget appeared or moved by more than MEMORY_BUDGET_TOLERANCE (None never moves)."""
    return new is not None and (old is None or abs(new - old) > MEMORY_BUDGET_TOLERANCE * old)


def _locate(spec, path):
    """(owning list, item) of a named node path (also ('parameter', name)); KeyError when it does not exist."""
    if len(path) == 3 and path[0] == 'feature':
        physics = spec.physics_by_name(path[1])
        return physics.features, physics.feature(path[2])
    if len(path) != 2 or path[0] not in _NAMED_NODES:
        raise KeyError(f'{path!r} is not a named model node')
    list_attribute, lookup = _NAMED_NODES[path[0]]
    return getattr(spec, list_attribute), getattr(spec, lookup)(path[1])


def _selection_owner(spec, path):
    """(spec object, attribute name) holding the entity list of a selection/material/feature/derived node."""
    attribute = _SELECTION_ATTRIBUTES.get(path[0]) if path else None
    if attribute is None:
        raise KeyError(f'node {path!r} has no selection')
    return _locate(spec, path)[1], attribute


def _index_of(items, item):
    """Position of `item` in `items` by identity."""
    return next(index for index, other in enumerate(items) if other is item)


def solver_fits(spec, registry, solver_type, values, roles=None):
    """[(physics name, role, misfit reason or '')] of every physics of a registered type declaring a role (of `roles`,
    default solver_type.roles) that solver_type fills, judged by schema.solver_misfit in the spec's study."""
    backend, study = solver_backend(spec), spec.study.type
    if backend not in BACKENDS:   # an invalid backend value: reported by validation
        backend = 'gpu'
    fits = []
    for physics_spec in spec.physics:
        try:
            physics_type = registry.physics_type(physics_spec.type)
        except KeyError:
            continue
        for role in (solver_type.roles if roles is None else roles):
            if role in physics_type.solver_roles and role in solver_type.roles:
                fits.append((physics_spec.name, role,
                             solver_misfit(solver_type, values, physics_type, role, study, backend)))
    return fits


def solver_users(spec, solver_name):
    """[(physics name, role)] whose PhysicsSpec.solvers name the solver node solver_name."""
    return [(physics.name, role) for physics in spec.physics if isinstance(physics.solvers, dict)
            for role, node in physics.solvers.items() if node == solver_name]


def harmonic_channel_names(physics_spec):
    """Channel names of the 'harmonic_channel' features of a physics spec (em_fdfd_nonlinear; polarization 'both'
    splits into <name>_ez and <name>_hz), in model order."""
    return list(harmonic_channel_polarizations(physics_spec))


def harmonic_channel_polarizations(physics_spec):
    """{channel name: 'ez' | 'hz'} of the 'harmonic_channel' features in model order (default 'ez'; 'both' gives
    <name>_ez and <name>_hz)."""
    polarizations = {}
    for feature in physics_spec.features:
        if feature.type != 'harmonic_channel':
            continue
        params = feature.params if isinstance(feature.params, dict) else {}
        polarization = params.get('polarization') or 'ez'
        if polarization == 'both':
            polarizations.update({f'{feature.name}_ez': 'ez', f'{feature.name}_hz': 'hz'})
        else:
            polarizations[feature.name] = polarization
    return polarizations


def derived_plot_names(spec):
    """Plot names of the derived curves (plot, else the curve name) in first-use order."""
    return list(dict.fromkeys(curve.plot or curve.name for curve in spec.derived_curves))


def parent_node(path):
    """Tree parent of a node path (8.1); ('model',) for roots and unknown paths."""
    if path in RESULT_GROUP_NODES:
        return ('results',)
    if path == ESTIMATION_NODE:
        return ('study',)
    if len(path) == 3 and path[0] == 'feature':
        return ('physics', path[1])
    if len(path) == 2:
        return _PARENT_NODES.get(path[0], ('model',))
    return ('model',)


def issue_node_path(issue):
    """Node whose tree item and form show a validation issue: ESTIMATION_NODE for the study's 'estimation…' issues,
    else issue.path."""
    path = tuple(issue.path)
    if path == ('study',) and str(issue.message).startswith('estimation'):
        return ESTIMATION_NODE
    return path


def mesh_of_node(spec, path):
    """Name of the mesh a tree node is about: a mesh node's own, a physics / feature node's physics' mesh, a
    coupling's target's mesh; None for other nodes and unresolvable names."""
    path = tuple(path or ())
    try:
        if len(path) == 2 and path[0] == 'mesh':
            return spec.mesh_by_name(path[1]).name
        if len(path) >= 2 and path[0] in ('physics', 'feature'):
            return spec.mesh_of(path[1]).name
        if len(path) == 2 and path[0] == 'coupling':
            return spec.mesh_of(spec.coupling(path[1]).target).name
    except KeyError:
        return None
    return None


def sibling_names(spec, path):
    """Names of the node's siblings in spec order (move up/down; a derived curve's: its plot's curves); [] for a missing
    node."""
    try:
        items, item = _locate(spec, tuple(path))
    except KeyError:
        return []
    return [items[position].name for position in _sibling_positions(items, item, path[0])]


def _sibling_positions(items, item, kind):
    """List positions of a node's tree siblings: every position, for a derived curve those of its plot's curves."""
    if kind != 'derived_curve':
        return list(range(len(items)))
    plot = item.plot or item.name
    return [position for position, other in enumerate(items) if (other.plot or other.name) == plot]


def _renamed_path(path, old_path, new_name):
    """`path` after renaming node old_path to new_name (features follow a renamed physics)."""
    if path == old_path:
        return old_path[:-1] + (new_name,)
    if old_path[0] == 'physics' and len(path) == 3 and path[:2] == ('feature', old_path[1]):
        return ('feature', new_name, path[2])
    return path


def _rename_unreferenced(spec, path, new_name):
    """Rename a feature / coupling / derived value / analytic or derived curve (no other node refers to them by name,
    except the result keys 'physics.feature' of sources, targets and inputs, renamed with a feature, and derived
    values named in later derived-value expressions)."""
    if path[0] == 'derived':
        spec.rename_derived_value(path[1], new_name)
        return
    items, item = _locate(spec, path)
    category = {'feature': f'feature in physics {path[1]!r}', 'coupling': 'coupling',
                'derived': 'derived value', 'curve': 'analytic curve', 'derived_curve': 'derived curve'}[path[0]]
    if not is_identifier(new_name):
        raise ValueError(f'{new_name!r} is not a valid {category} name (letters, digits and _, not starting with a '
                         f'digit)')
    if any(other.name == new_name for other in items if other is not item):
        raise ValueError(f'a {category} named {new_name!r} already exists')
    item.name = new_name
    if path[0] == 'feature':
        spec.rename_result_key(f'{path[1]}.{path[2]}', f'{path[1]}.{new_name}')


def _declares_courant_factor(physics_type, settings_values):
    """True when the type's domain requirements for settings_values set a Courant factor."""
    if physics_type.domain_requirements is None:
        return False
    return physics_type.domain_requirements(settings_values).courant_factor is not None


def _default_object_kind(feature_type):
    """Geometry kind created with a new feature: 'point' / 'line' selections, single-rectangle boxes; else None."""
    if feature_type.selection_kind in ('point', 'line'):
        return feature_type.selection_kind
    if (feature_type.selection_kind == 'domain' and tuple(feature_type.object_kinds) == ('rectangle',)
            and feature_type.max_entities == 1):
        return 'rectangle'
    return None


def _name_prefix(name):
    """Name without trailing digits ('rectangle2' -> 'rectangle'), for numbering copies."""
    return re.sub(r'\d+$', '', name) or name


def _rounded(value):
    """Lengths rounded to 6 significant digits (nested lists too) so default shapes read cleanly in forms."""
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return float(f'{value:.6g}')
