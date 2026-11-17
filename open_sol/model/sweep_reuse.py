"""Reused sweep builds (SweepSpec.reuse_build): the spec values a swept parameter reaches (what refuses the reuse) and
the ParameterFollower that re-sets a following feature value, physics setting or material value on the kept build
per sweep value."""
import dataclasses

from .expressions import COORDINATE_NAMES, ExpressionError, evaluate_scalar, referenced_names
from .geometry import GEOMETRY_KINDS
from .schema import known_physics, state_field_names
from .solver_hooks import RerunSolver

REUSE_REFUSED_STUDIES = ('time_dependent', 'optimization')   # their run state is no rerun of a kept build
# Param kinds whose values are numbers / expressions (the others hold names or choices)
_EXPRESSION_KINDS = frozenset({'float', 'int', 'float_list', 'points', 'table', 'expression'})
# keys of free-form spec dicts (meshes, domain, material entries) that hold names or choices, never expressions
_TEXT_KEYS = frozenset({'name', 'kind', 'mode', 'mesher', 'object', 'entity', 'face', 'axis', 'profile', 'where',
                        'model', 'plane', 'library', 'coordinates', 'angle_wrap', 'type', 'inputs', 'selection'})


def value_names(value) -> frozenset:
    """Free names of every expression inside a spec value (lists, dicts minus their text keys); unparsable text and
    non-numeric leaves add none (validation reports them)."""
    if value is None or isinstance(value, bool):
        return frozenset()
    if isinstance(value, (list, tuple)):
        return frozenset().union(*(value_names(item) for item in value))
    if isinstance(value, dict):
        return frozenset().union(*(value_names(item) for key, item in value.items() if key not in _TEXT_KEYS))
    try:
        return referenced_names(value)
    except ExpressionError:
        return frozenset()


def live_parameter_names(spec, parameter) -> frozenset:
    """The swept parameter and every parameter depending on it, directly or through others."""
    dependencies = {item.name: value_names(item.expression) for item in spec.parameters}
    live = {parameter}
    grown = True
    while grown:
        grown = False
        for name, used in dependencies.items():
            if name not in live and not used.isdisjoint(live):
                live.add(name)
                grown = True
    return frozenset(live)


def reuse_issues(spec, registry) -> list:
    """[(node path, message)] that refuse SweepSpec.reuse_build: a refused study, parity sectors, mesh refinement levels
    and every spec value the swept parameter reaches besides following values of parameters only (feature values and
    physics settings flagged Param.follows_parameters, repaintable material values: _material_follows), derived
    values, curves, references and study settings."""
    sweep = spec.study.sweep
    issues = []
    if spec.study.type in REUSE_REFUSED_STUDIES:
        issues.append((('study',), f'sweep reuse_build: the {spec.study.type} study cannot rerun a kept build (its '
                                   'state is the run itself); switch reuse_build off'))
    settings = spec.study.settings
    if isinstance(settings, dict) and settings.get('parity_sectors') is True:
        issues.append((('study',), 'sweep reuse_build: parity sectors build one model per sector; switch reuse_build '
                                   'or parity_sectors off'))
    if spec.study.adaptation is not None:
        issues.append((('study',), 'sweep reuse_build: mesh refinement levels rebuild the model; switch reuse_build '
                                   'off or clear adaptation'))
    parameter = sweep.parameter
    if not any(item.name == parameter for item in spec.parameters):
        return issues   # validation names the unknown parameter
    live = live_parameter_names(spec, parameter)
    for path, what, names, following in _value_uses(spec, registry):
        reached = sorted(names & live)
        if not reached:
            continue
        uses = ', '.join(repr(name) for name in reached)
        if following and names.isdisjoint(COORDINATE_NAMES):
            continue
        if following:
            issues.append((path, f'{what} uses {uses} (changed by the sweep over {parameter!r}) together with x / y '
                                 '/ z / t: only values of parameters follow a reused build; switch reuse_build off or '
                                 'make the value uniform'))
        else:
            issues.append((path, f'{what} uses {uses}, which the sweep over {parameter!r} changes: a reused build '
                                 '(reuse_build) keeps it at the first value; switch reuse_build off'))
    return issues


def _value_uses(spec, registry):
    """Yield (node path, text, names, following) of every spec value the build reads (derived values, curves and the
    study settings excluded): following = a feature value or physics setting flagged Param.follows_parameters (a
    setting the auto mesh sizes by excluded: _auto_wavelength_mesh), or a material value _material_follows."""
    domain = spec.domain
    for item in dataclasses.fields(domain):
        if item.name not in _TEXT_KEYS:
            yield ('geometry',), f'domain {item.name}', value_names(getattr(domain, item.name)), False
    for obj in spec.geometry:
        kind = GEOMETRY_KINDS.get(obj.kind)
        for name, names in _param_uses(None if kind is None else kind.params, obj.params):
            yield ('geometry', obj.name), f'geometry object {obj.name!r} {name}', names, False
    state_names = state_field_names(spec, registry)
    for material in spec.materials:
        properties = material.properties if isinstance(material.properties, dict) else {}
        for name, raw in properties.items():
            names = value_names(raw)
            yield (('material', material.name), f'material {material.name!r} {name}', names,
                   _material_follows(spec, registry, name, names, state_names))
        yield ('material', material.name), f'material {material.name!r} dispersion', value_names(
            material.dispersion), False
    for mesh_spec in spec.meshes:
        for key, raw in dataclasses.asdict(mesh_spec).items():
            if key not in _TEXT_KEYS:
                yield ('mesh', mesh_spec.name), f'mesh {mesh_spec.name!r} {key}', value_names(raw), False
    for physics_spec, physics_type in known_physics(spec, registry):
        following = set()
        if not _auto_wavelength_mesh(spec, physics_spec, physics_type):
            following = {param.name for param in physics_type.settings if param.follows_parameters}
        for name, names in _param_uses(physics_type.settings, physics_spec.settings):
            yield ('physics', physics_spec.name), f'physics {physics_spec.name!r} setting {name}', names, \
                name in following
        for feature_spec in physics_spec.features:
            if not feature_spec.enabled:
                continue
            try:
                params = physics_type.feature_type(feature_spec.type).params
            except KeyError:
                continue
            following = {param.name for param in params if param.follows_parameters}
            path = ('feature', physics_spec.name, feature_spec.name)
            for name, names in _param_uses(params, feature_spec.params):
                yield path, f'feature {physics_spec.name}/{feature_spec.name} {name}', names, name in following
    for coupling_spec in spec.couplings:
        if not coupling_spec.enabled:
            continue
        try:
            params = registry.coupling_type(coupling_spec.type).params
        except KeyError:
            continue
        for name, names in _param_uses(params, coupling_spec.params):
            yield ('coupling', coupling_spec.name), f'coupling {coupling_spec.name!r} {name}', names, False
    for node in spec.solvers:
        try:
            params = registry.solver_type(node.type).params
        except KeyError:
            params = None
        for name, names in _param_uses(params, node.settings):
            yield ('solver', node.name), f'solver {node.name!r} {name}', names, False
    yield ('study',), 'stored cut', value_names(spec.study.stored_cuts), False


def _material_follows(spec, registry, name, names, state_names) -> bool:
    """True when a reused build repaints material property `name` (builder._follow_material; x / y / z / t in it are
    reported by reuse_issues): no state field in it, no 3D tensor mesh, and every type reading it declares it
    state_dependent (materials_changed re-reads the array) and is no EM type (index references, dispersive
    backgrounds and subpixel averages are taken from the build's permittivity)."""
    if not names.isdisjoint(state_names):
        return False   # a DynamicMaterialUpdate writes it
    if spec.is_3d and any(mesh_spec.kind != 'tetrahedra' for mesh_spec in spec.meshes):
        return False   # 3D tensor materials are painted per z-slab run: no region mask is kept
    for _, physics_type in known_physics(spec, registry):
        for param in physics_type.material_properties:
            if param.name == name and (physics_type.family == 'em' or not param.state_dependent):
                return False
    return True


def _auto_wavelength_mesh(spec, physics_spec, physics_type) -> bool:
    """True when the physics' mesh is an auto mesh sized by the physics' shortest wavelength (its settings shape the
    mesh, so none of them follows a reused build)."""
    try:
        mesh_spec = spec.mesh_of(physics_spec)
    except KeyError:
        return False   # validation reports the unknown mesh
    return (physics_type.shortest_wavelength is not None and mesh_spec.mode == 'auto'
            and mesh_spec.wavelength_min is None)


def _param_uses(params, raw_values):
    """Yield (name, names) of the raw values of a Param-typed dict: expression kinds by their expressions, a 'ref' to a
    parameter by its name; params None (unknown type): every value as an expression."""
    if not isinstance(raw_values, dict):
        return
    if params is None:
        for name, raw in raw_values.items():
            yield name, value_names(raw)
        return
    for param in params:
        if param.name not in raw_values:
            continue
        raw = raw_values[param.name]
        if param.kind == 'ref' and param.ref_kind == 'parameter':
            yield param.name, frozenset({raw}) if isinstance(raw, str) else frozenset()
        elif param.kind in _EXPRESSION_KINDS:
            yield param.name, value_names(raw)


def non_rerun_physics(session) -> list:
    """Names of the session's physics whose solver cannot rerun a kept build (no RerunSolver hook)."""
    return [name for name, solver in session.runner.physics.items() if not isinstance(solver, RerunSolver)]


class ParameterFollower:
    """Feature values of a reused sweep build: update(parameters) calls setter(*values) when one changed (setters are
    incremental, like BuildContext.follow_time's). followed: per value its raw expression of parameters, or None to
    pass fixed[i] (the build's own value object: a free None, an x / y field) unchanged every time."""

    def __init__(self, label, followed, fixed, setter, parameters):
        self.label = label
        self._followed = tuple(followed)
        self._fixed = tuple(fixed)
        self._setter = setter
        self._applied = self._values(parameters)   # the values the build applied

    def _values(self, parameters):
        values = []
        for raw, fixed in zip(self._followed, self._fixed):
            if raw is None:
                values.append(fixed)
                continue
            try:
                values.append(evaluate_scalar(raw, parameters))
            except ExpressionError as exc:
                raise ValueError(f'{self.label}: {exc}') from None
        return tuple(values)

    def update(self, parameters):
        """Re-set the values of the new parameters when any changed."""
        values = self._values(parameters)
        if values != self._applied:
            self._setter(*values)
            self._applied = values
