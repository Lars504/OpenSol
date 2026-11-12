"""Self-describing parameter / feature / physics / coupling / study types, templates, selftest cases and the
discovered type registry."""
import copy
import importlib
import importlib.util
import math
import pkgutil
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property

import numpy as np

from ..core.mesh import has_uniform_spacing
from ..core.metric import GRIDS
from ..core.renderer import FieldInfo
from ..core.backend import cupy_installed
from ..solvers.roles import (BACKENDS, PRECISIONS, SOLVER_ROLES, ResolvedSolver,   # pure (no model imports):
                             SolverRole, SolverType, resolved_precision, solver_misfit)   # re-exported
from .expressions import (FIELD_COORDINATE_NAMES, TIME_NAMES, compile_expression, evaluate_parameters,
                          evaluate_scalar, is_identifier)

PARAM_KINDS = ('float', 'int', 'bool', 'choice', 'str', 'expression', 'ref', 'float_list', 'points', 'names',
               'table')
REF_KINDS = ('material', 'geometry', 'physics', 'parameter')
FEATURE_CATEGORIES = ('source', 'probe', 'monitor', 'boundary_condition', 'domain_condition')
SELECTION_KINDS = ('domain', 'boundary', 'point', 'line', 'interface')
INTERFACE_SIDES = ('inside', 'outside')   # SelectionSpec.side of an 'interface' selection (igor 1.8)
# per-sample variables of a derived value over an interface selection: the outward unit normal of the object and the
# tangent (normal turned +90 deg); they shadow parameters
INTERFACE_VARIABLES = ('normal_x', 'normal_y', 'tangent_x', 'tangent_y')
INTERFACE_OPERATIONS = ('mean', 'max', 'min', 'integral')   # field operations over an interface selection
INTERFACE_PROJECTIONS = ('normal', 'tangential')   # AnalyticCurveSpec.projection of an interface line curve
RESULT_KINDS = ('timeseries', 'flux_spectrum', 'dft_fields', 'far_field', 'xy', 'table')
ISSUE_LEVELS = ('error', 'warning', 'info')
FIELD_DERIVED_OPERATIONS = ('mean', 'max', 'min', 'integral', 'mode_volume', 'at')   # source None (fields)
SOURCE_DERIVED_OPERATIONS = ('mean', 'max', 'min', 'integral', 'last', 'at', 'argmax', 'fwhm', 'polyfit', 'exp_fit',
                             'lorentz_fit', 'two_exp_fit', 'two_wave_fit', 'richardson', 'multi_lorentz_fit',
                             'pole_fit')   # a DerivedValueSpec.source (1D result or derived curve)
# source None, no fields: one number from parameters, scalar outputs, t and earlier derived values (a global
# evaluation of other results, e.g. sqrt(N_int / rho_int))
GLOBAL_DERIVED_OPERATION = 'expression'
SOURCE_FREE_DERIVED_OPERATIONS = FIELD_DERIVED_OPERATIONS + (GLOBAL_DERIVED_OPERATION,)   # source None
DERIVED_OPERATIONS = SOURCE_DERIVED_OPERATIONS + ('mode_volume', GLOBAL_DERIVED_OPERATION)
COUPLING_ROLES = ('source', 'target')
PERMITTIVITY_PROPERTY_NAMES = ('eps_r', 'eps_xx', 'eps_yy', 'eps_zz')   # material properties writing permittivity
# PhysicsType.group -> GUI Add physics submenu (in this order; '' -> 'Other')
PHYSICS_GROUPS = (('em', 'Electromagnetic waves'), ('heat', 'Heat transfer'),
                  ('electrical', 'Electric fields, currents and semiconductors'),
                  ('structural', 'Structural mechanics and acoustics'), ('fluid', 'Fluid flow'))
# what the drawn xy plane of a 2D model is (PhysicsType.spaces_2d) -> GUI text; SPACES adds the 3D model
SPACES_2D = {'planar': 'a planar model (the drawing is the plane of travel, uniform in depth)',
             'cross_section': 'a waveguide cross-section (light travels out of the drawing)',
             'axisymmetric': 'an axisymmetric model (the drawing is rotated about its left edge)'}
SPACES = {**SPACES_2D, '3d': 'a 3D model'}
# DomainSpec.coordinates of a polar grid -> the SPACES its physics take (validate's _check_coordinate_grid: the
# coordinates fix how the drawing extends into 3D); a Cartesian domain restricts nothing
COORDINATE_SPACES = {'polar': ('planar', 'cross_section'), 'spherical': ('axisymmetric',)}
MESH_KINDS = ('tensor', 'triangles', 'tetrahedra')   # MeshSpec.kind; also the grid kinds PhysicsType.grids name
FREE_MESH_KINDS = ('triangles', 'tetrahedra')   # element meshes sized by h_max / h_min / growth (no tensor options)
# MeshSpec values whose null means a default of the mesh kind (free triangle / tetrahedron meshes: the industry-standard
# growth 1.3; five quadratic elements per wavelength)
MESH_KIND_DEFAULTS = {'tensor': {'max_ratio': 1.15, 'points_per_wavelength': 20.0},
                      'triangles': {'max_ratio': 1.3, 'points_per_wavelength': 5.0},
                      'tetrahedra': {'max_ratio': 1.3, 'points_per_wavelength': 5.0}}
MESH_TRANSFERS = ('field', 'value')                  # CouplingType.mesh_transfer (None = one mesh required)
TRANSFER_KINDS = ('interpolate', 'conservative')     # how a value crosses meshes (CouplingType.value_kind)
ADAPTATION_METHODS = ('adaptive', 'uniform')         # AdaptationSpec.method


def _require_unique_names(items, what, owner):
    """Raise ValueError when two items in `items` share a name."""
    seen = set()
    for item in items:
        if item.name in seen:
            raise ValueError(f"{owner}: duplicate {what} name '{item.name}'")
        seen.add(item.name)


def _check_params(params, owner):
    """Raise ValueError for a definition error: a visible_if naming an unknown sibling or a value its choice / bool
    controller never takes, or a non-None default that its own kind / bounds reject."""
    params_by_name = {param.name: param for param in params}
    for param in params:
        for controller_name, accepted in param.visible_if:
            controller = params_by_name.get(controller_name)
            if controller is None:
                raise ValueError(f"{owner}: param '{param.name}' visible_if names unknown param '{controller_name}'")
            valid = {'choice': controller.choices, 'bool': (True, False)}.get(controller.kind)
            if valid is not None and any(value not in valid for value in accepted):
                raise ValueError(f"{owner}: param '{param.name}' visible_if accepts {accepted!r}, but "
                                 f"'{controller_name}' takes only {valid!r}")
        if param.default is None or (param.kind == 'ref' and param.default == ''):
            continue
        try:
            _convert(param, param.default, {}, None)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            raise ValueError(f"{owner}: param '{param.name}' default {short_repr(param.default)} is invalid: "
                             f'{exc}') from exc


_SHARED_MATERIAL_PROPERTY_FIELDS = ('kind', 'unit', 'min', 'max', 'min_exclusive', 'optional', 'field_dependent',
                                    'state_dependent', 'time_dependent', 'columns', 'increasing')


def _require_matching_material_properties(registered_type, new_type):
    """Raise ValueError when a material property of new_type differs from the same-named one of registered_type in
    kind, unit, bounds or flags (label, help, default and expands_to may differ)."""
    registered_props = {prop.name: prop for prop in registered_type.material_properties}
    for prop in new_type.material_properties:
        other = registered_props.get(prop.name)
        differing = [name for name in _SHARED_MATERIAL_PROPERTY_FIELDS
                     if other is not None and getattr(prop, name) != getattr(other, name)]
        if differing:
            raise ValueError(f"PhysicsType '{new_type.name}': material property '{prop.name}' differs from "
                             f"'{registered_type.name}' in {', '.join(differing)}")


def short_repr(value, limit=80) -> str:
    """repr(value) for messages, cut to `limit` characters (ending in '...')."""
    text = repr(value)
    return text if len(text) <= limit else text[:limit - 3] + '...'


@dataclass(frozen=True)
class Param:
    """Descriptor of one user-editable value (setting, feature/coupling/study/geometry/mesh parameter)."""
    name: str
    kind: str
    default: object = None
    label: str = ''
    unit: str = ''
    min: float | None = None
    max: float | None = None
    choices: tuple = ()
    help: str = ''                   # full text for agents (describe -> api_reference.md)
    tip: str = ''                    # GUI tooltip: '\n\n' between paragraphs, '- ' list lines; '' -> help
    choice_labels: tuple = ()        # ((choice, GUI drop-down text), ...); a missing choice shows str(choice)
    field_dependent: bool = False    # float may reference x, y (z in 3D models)
    state_dependent: bool = False    # float may also reference physics state fields and time t
    # float may reference time t [s] (with parameters only, no x / y): the feature re-evaluates it at the start of
    # every runner step of a time_dependent study (t = 0 elsewhere); state_dependent material properties always may
    time_dependent: bool = False
    # feature float re-set in place per value of a reused sweep build (SweepSpec.reuse_build): its apply wires a
    # setter through BuildContext.follow_parameters (model.sweep_reuse); unflagged values naming the swept parameter
    # block the reuse
    follows_parameters: bool = False
    visible_if: tuple = ()           # ((param, (values...)), ...): visible when ANY pair matches; () = always
    expands_to: tuple = ()           # material properties: MaterialGrid arrays written
    ref_kind: str = ''               # REF_KINDS; required for kind 'ref', optional for 'names'
    optional: bool = False           # None allowed
    min_exclusive: bool = False      # True -> value must be > min (else >= min)
    columns: tuple = ()              # kind 'table' (required): ((column name, unit), ...); min / max bound every entry
    increasing: bool = False         # kind 'table': >= 2 rows, every column strictly increasing down the rows
    # ((space, unit), ...): the unit in those SPACES of the physics (e.g. a point source W/m planar, W axisymmetric /
    # 3d); unit holds in every other space. `param_in_space` resolves it (GUI forms).
    space_units: tuple = ()

    def __post_init__(self):
        if not is_identifier(self.name):
            raise ValueError(f'Param name {self.name!r} is not an identifier')
        if self.kind not in PARAM_KINDS:
            raise ValueError(f"Param '{self.name}': unknown kind {self.kind!r}; valid: {', '.join(PARAM_KINDS)}")
        if self.kind == 'choice' and (not self.choices or
                                      (self.default is not None and self.default not in self.choices)):
            raise ValueError(f"Param '{self.name}': choice needs choices containing the default")
        if (self.kind == 'ref' and not self.ref_kind) or (self.ref_kind and self.ref_kind not in REF_KINDS):
            raise ValueError(f"Param '{self.name}': ref_kind must be one of {', '.join(REF_KINDS)}")
        if (self.field_dependent or self.state_dependent or self.time_dependent) and self.kind != 'float':
            raise ValueError(f"Param '{self.name}': field_dependent / state_dependent / time_dependent apply to kind "
                             "'float' only")
        if self.follows_parameters and self.kind != 'float':
            raise ValueError(f"Param '{self.name}': follows_parameters applies to kind 'float' only")
        if ((self.kind == 'table') != bool(self.columns)
                or not all(isinstance(entry, tuple) and len(entry) == 2 and all(isinstance(text, str) for text in entry)
                           for entry in self.columns)):
            raise ValueError(f"Param '{self.name}': kind 'table' needs columns ((name, unit), ...), other kinds none")
        if self.increasing and self.kind != 'table':
            raise ValueError(f"Param '{self.name}': increasing applies to kind 'table' only")
        for entry in self.choice_labels:
            if not (isinstance(entry, tuple) and len(entry) == 2 and entry[0] in self.choices
                    and isinstance(entry[1], str)):
                raise ValueError(f"Param '{self.name}': choice_labels entries must be (choice, text) of its choices")
        for entry in self.visible_if:
            if not (isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str)
                    and isinstance(entry[1], tuple)):
                raise ValueError(f"Param '{self.name}': visible_if entries must be (param_name, (values, ...))")
        for entry in self.space_units:
            if not (isinstance(entry, tuple) and len(entry) == 2 and entry[0] in SPACES and isinstance(entry[1], str)):
                raise ValueError(f"Param '{self.name}': space_units entries must be (space, unit) with a space of "
                                 f"{', '.join(SPACES)}")


@dataclass(frozen=True)
class FeatureType:
    """A feature a physics can attach to a selection (source, probe, monitor, boundary or domain condition)."""
    name: str
    label: str
    category: str                    # FEATURE_CATEGORIES
    selection_kind: str | None       # SELECTION_KINDS or None
    params: tuple = ()
    object_kinds: tuple = ()         # allowed geometry kinds (empty = any of the entity kind)
    min_entities: int = 1
    max_entities: int | None = None
    allow_outer_faces: bool = True   # boundary kind
    allow_interior: bool = True      # boundary kind: object perimeters
    requires_uniform_mesh: bool = False
    acts_on_hi_face: bool = False    # line feature acting on the hi face of its line cells (GUI glyph position)
    result_kind: str | None = None   # RESULT_KINDS or None
    description: str = ''
    summary: str = ''                # GUI text (tree / menu tooltip, settings panel); '' -> description
    apply: Callable | None = None    # apply(ctx, solver, feature_spec, values, selection) -> runtime obj | None
    result: Callable | None = None   # result(session, solver, runtime_obj, feature_spec, options: dict) -> dict
    validate: Callable | None = None  # validate(vctx, physics_spec, feature_spec, values) -> list[Issue]
    # memory_estimate(values, n_selected_cells, mesh_estimate) -> core.memory_estimate.MemoryEstimate (CPU only);
    # set on features that allocate per-region or per-frequency arrays (meshing.estimate_memory)
    memory_estimate: Callable | None = None
    # stepper precisions the feature works at (solver_plan S3.3; validate checks the resolved 'stepper' solver)
    precisions: tuple = tuple(PRECISIONS)
    # MESH_KINDS of the meshes the feature works on (validate refuses an enabled one on another kind)
    grids: tuple = ('tensor',)

    def __post_init__(self):
        if not (isinstance(self.grids, tuple) and self.grids and set(self.grids) <= set(MESH_KINDS)):
            raise ValueError(f"FeatureType '{self.name}': grids must be a non-empty tuple of {', '.join(MESH_KINDS)}, "
                             f'got {self.grids!r}')
        if self.category not in FEATURE_CATEGORIES:
            raise ValueError(f"FeatureType '{self.name}': unknown category {self.category!r}")
        if self.selection_kind is not None and self.selection_kind not in SELECTION_KINDS:
            raise ValueError(f"FeatureType '{self.name}': unknown selection_kind {self.selection_kind!r}")
        if self.result_kind is not None and self.result_kind not in RESULT_KINDS:
            raise ValueError(f"FeatureType '{self.name}': unknown result_kind {self.result_kind!r}")
        if (not isinstance(self.precisions, tuple) or not self.precisions
                or not set(self.precisions) <= set(PRECISIONS)):
            raise ValueError(f"FeatureType '{self.name}': precisions must be a non-empty tuple of "
                             f"{', '.join(PRECISIONS)}, got {self.precisions!r}")
        _require_unique_names(self.params, 'param', f"FeatureType '{self.name}'")
        _check_params(self.params, f"FeatureType '{self.name}'")


@dataclass(frozen=True)
class DomainRequirements:
    """What a physics needs from the Domain: EM BoundaryConfig (PML padding), Courant factor, periodic axes and edge
    sizes."""
    boundary_config: object | None = None   # EM BoundaryConfig -> PML padding
    courant_factor: float | None = None     # fraction of the 2D CFL limit
    periodic_axes: tuple = ()               # 'x' / 'y' / 'z' meshed periodic (gmsh tetrahedra: periodic surfaces)
    edge_sizes: tuple = ()                  # (entity, size [m]): gmsh tetrahedra refine along the object's edges


@dataclass(frozen=True)
class PhysicsType:
    """Self-describing physics: settings, features, material vocabulary, fields and factory callbacks."""
    name: str
    label: str
    default_instance_name: str
    description: str = ''
    summary: str = ''                  # GUI text (tree / menu tooltip, settings panel); '' -> description
    family: str = ''                   # physics family shared by interchangeable types ('em'); '' = none
    settings: tuple = ()
    features: tuple = ()               # FeatureType, ordered
    material_properties: tuple = ()    # Param with expands_to; isotropic before component entries
    material_schema: dict = field(default_factory=dict)
    state_fields: tuple = ()           # FieldInfo of arrays in solver.state() usable in material/coupling exprs
    output_fields: tuple = ()          # FieldInfo, identical to the renderer's field_infos (same order)
    # inject_source slot name -> (unit, meaning) shown by describe; iterates / joins / `in` as the names, in order
    source_slots: dict = field(default_factory=dict)
    supported_studies: tuple = ()      # 'time_dependent', 'stationary', 'frequency_domain', 'mode_analysis'
    default_study: dict = field(default_factory=dict)   # StudySpec fields applied when it is the first physics
    uses_dispersion: bool = False
    create: Callable | None = None               # create(ctx, physics_spec, settings_values) -> solver
    domain_requirements: Callable | None = None  # domain_requirements(settings_values) -> DomainRequirements
    # mesh_edge_sizes(settings_values, physics_spec) -> ((entity or named selection, size [m]), ...): element size along
    # the edges of those geometry objects on a gmsh tetrahedra mesh (field singularities at electrode edges); CPU, never
    # raises
    mesh_edge_sizes: Callable | None = None
    steps_in_time: Callable | None = None        # steps_in_time(settings_values) -> bool (finite dt_limit)
    # shortest_wavelength(spec, physics_spec, settings_values, parameters, registry) -> vacuum m | None (inputs that
    # do not resolve are skipped): the auto mesh's default wavelength_min; never raises for model errors
    shortest_wavelength: Callable | None = None
    validate: Callable | None = None             # validate(vctx, physics_spec, settings_values) -> list[Issue]
    # memory_estimate(settings_values, mesh_estimate, context) -> core.memory_estimate.MemoryEstimate (CPU only;
    # context = meshing.MemoryContext: spec, physics_spec, parameters, registry, material_regions); the solver's own
    # arrays, not the MaterialGrid arrays of material_schema (meshing.estimate_memory counts those once)
    memory_estimate: Callable | None = None
    # time_estimate(settings_values, mesh_estimate, context) -> core.time_estimate.TimeEstimate (CPU, unpriced; context
    # = model.time_estimate.TimeContext): the work per step / solve / wavelength as ops and solver roles' work, never
    # seconds; None = 'not estimated' (time_estimation_plan)
    time_estimate: Callable | None = None
    # scalar_outputs(settings_values, physics_spec) -> tuple of names the solver's scalars() reports (derived values:
    # <physics>_<name>; physics_spec for names that follow its features, CPU only); scalar_outputs_help (units,
    # meaning; describe) is required with it
    scalar_outputs: Callable | None = None
    scalar_outputs_help: str = ''
    # model dimensions the type runs in: (2,) 2D only, (3,) 3D only (spec v2 size_z set), (2, 3) both, 1 = 1D models
    # (size_y null: a 2D Domain one cell across, symmetry 2.20); validate rejects a mismatch. A 3D-capable type builds
    # on core.domain3d.Domain3D (see the core sub file, 3D contracts)
    dimensions: tuple = (2,)
    # MaterialGrid array -> (default, trigger property names): allocated (besides material_schema) only when a spec
    # material (library merged) sets one of its trigger properties; the builder skips writes into unallocated ones
    # (meshing.optional_material_arrays; em_fdtd_3d's eps_xx / eps_yy / sigma_m)
    optional_material_schema: dict = field(default_factory=dict)
    group: str = ''                    # PHYSICS_GROUPS key (GUI Add physics submenu); '' = 'Other'
    # SPACES_2D the type runs in (used when 2 in dimensions), and space -> {setting: value} putting an instance into
    # that space (heat 'axisymmetric'); GUI only: Add physics offers the types sharing a space with the physics present
    # (physics_type_misfit; '3d' from dimensions)
    spaces_2d: tuple = ('planar',)
    space_settings: dict = field(default_factory=dict)
    # SOLVER_ROLES key -> SolverRole (solvers.roles): the numerical jobs the type hands out (solver_plan S3.3);
    # register_physics checks every default_type is a registered SolverType
    solver_roles: dict = field(default_factory=dict)
    # honours solver_defaults.subpixel_smoothing (igor 1.5): the builder hands its solver the interface geometry by
    # set_subpixel_smoothing(geometry) right after create (subpixel_smoothing_physics decides where it applies)
    subpixel_smoothing: bool = False
    # honours solver_defaults.conformal_boundaries (solver_plan S7.6): the builder hands its solver a
    # model.boundary_locator.GeometryLocator by set_conformal_boundaries(locator) right after create, in models of
    # the conformal_dimensions (em_fdtd_3d: (3,), a 3D locator) whose 2D DomainSpec.coordinates are in
    # conformal_coordinates (heat / potential also 'polar' / 'spherical': the locator then takes logical coordinates)
    conformal_boundaries: bool = False
    conformal_dimensions: tuple = (2,)
    conformal_coordinates: tuple = ('cartesian',)
    # run backends (solvers.roles BACKENDS) the type's own code runs on: engine, renderer, features, the couplings it
    # owns (solver_plan S6); a type adds 'cpu' only once all of it runs on numpy (validate errors on the cpu otherwise)
    backends: tuple = ('gpu',)
    # core.metric GRIDS of the 2D in-plane grid the type solves on (DomainSpec.coordinates 'polar' / 'spherical' map to
    # 'polar'); validate refuses the others; its solver's PhysicsSolver.coordinate_grids must agree
    coordinate_grids: tuple = ('cartesian',)
    # MESH_KINDS of the meshes the type runs on (S7.5: it dispatches on the kind of its mesh; validate refuses others;
    # solver_misfit reads it as the physics' grid kinds)
    grids: tuple = ('tensor',)
    # mesh_method(spec, physics_spec, settings_values, mesh_kind, parameters) -> (text, problems): GUI texts (mesh
    # form) naming how the type solves on that mesh kind and, as problems, what that method cannot handle in this
    # model (mirrors validation); CPU, GUI-thread safe, never raises for model errors; None = no text
    mesh_method: Callable | None = None
    # state / output field names the S7.7 interpolation-error indicator reads when the solver has no error_indicator()
    # hook (() = the primary output field)
    indicator_fields: tuple = ()
    # frequency_response sweep_method values the type's solver serves ('adaptive' / 'modal': solver_hooks
    # ReducedSweepSolver); the study's validate refuses the others
    frequency_sweep_methods: tuple = ('direct',)
    # eigenfrequency search values the type's solver serves ('targets' / 'band': solver_hooks MultiShiftEigenSolver);
    # the study's validate refuses the others
    eigen_searches: tuple = ('nearest',)

    def __post_init__(self):
        owner = f"PhysicsType '{self.name}'"
        if not (isinstance(self.grids, tuple) and self.grids and set(self.grids) <= set(MESH_KINDS)):
            raise ValueError(f"{owner}: grids must be a non-empty tuple of {', '.join(MESH_KINDS)}, got {self.grids!r}")
        field_names = {info.name for info in (*self.state_fields, *self.output_fields)
                       if isinstance(info, FieldInfo)}
        if not (isinstance(self.indicator_fields, tuple) and set(self.indicator_fields) <= field_names):
            raise ValueError(f'{owner}: indicator_fields must be a tuple of its state / output field names, got '
                             f'{self.indicator_fields!r}')
        if not (isinstance(self.backends, tuple) and self.backends and set(self.backends) <= set(BACKENDS)):
            raise ValueError(f"{owner}: backends must be a non-empty tuple of {', '.join(BACKENDS)}, got "
                             f'{self.backends!r}')
        if not (isinstance(self.coordinate_grids, tuple) and 'cartesian' in self.coordinate_grids
                and set(self.coordinate_grids) <= set(GRIDS)):
            raise ValueError(f"{owner}: coordinate_grids must be a tuple of {', '.join(GRIDS)} holding 'cartesian', "
                             f'got {self.coordinate_grids!r}')
        for role, solver_role in self.solver_roles.items():
            if role not in SOLVER_ROLES or not isinstance(solver_role, SolverRole):
                raise ValueError(f"{owner}: solver_roles maps {', '.join(SOLVER_ROLES)} to SolverRole, got "
                                 f"{role!r}: {solver_role!r}")
        if not isinstance(self.family, str):
            raise TypeError(f'{owner}: family must be text, got {type(self.family).__name__}')
        if not (isinstance(self.dimensions, tuple) and self.dimensions
                and set(self.dimensions) <= {1, 2, 3}):
            raise ValueError(f'{owner}: dimensions must be a non-empty tuple of 1, 2, 3, got {self.dimensions!r}')
        if self.group and self.group not in dict(PHYSICS_GROUPS):
            raise ValueError(f"{owner}: unknown group {self.group!r}; valid: {', '.join(dict(PHYSICS_GROUPS))}")
        if not (isinstance(self.spaces_2d, tuple) and self.spaces_2d and set(self.spaces_2d) <= set(SPACES_2D)):
            raise ValueError(f"{owner}: spaces_2d must be a non-empty tuple of {', '.join(SPACES_2D)}")
        setting_names = {param.name for param in self.settings}
        for space, required in self.space_settings.items():
            if space not in self.spaces_2d or not set(required) <= setting_names:
                raise ValueError(f"{owner}: space_settings '{space}' must be one of spaces_2d and name settings")
        if self.scalar_outputs is not None and not self.scalar_outputs_help:
            raise ValueError(f'{owner}: scalar_outputs needs a scalar_outputs_help text (units and meaning)')
        _require_unique_names(self.settings, 'setting', owner)
        _require_unique_names(self.features, 'feature', owner)
        _require_unique_names(self.material_properties, 'material property', owner)
        _check_params(self.settings, owner)
        _check_params(self.material_properties, owner)
        for prop in self.material_properties:   # a 'table' property writes no array: create reads the regions
            if (prop.kind == 'table') == bool(prop.expands_to):
                raise ValueError(f"{owner}: material property '{prop.name}' needs expands_to, except a 'table' "
                                 'property, which has none')
        if not (isinstance(self.state_fields, tuple) and isinstance(self.output_fields, tuple)):
            raise TypeError(f'{owner}: state_fields and output_fields must be tuples (consumers concatenate them)')
        for info in (*self.state_fields, *self.output_fields):
            if not isinstance(info, FieldInfo):
                raise TypeError(f'{owner}: state_fields / output_fields must hold FieldInfo, got {info!r}')
        if not isinstance(self.source_slots, dict):
            raise TypeError(f'{owner}: source_slots must be a dict slot -> (unit, meaning)')
        for slot, entry in self.source_slots.items():
            if not (isinstance(entry, tuple) and len(entry) == 2 and all(isinstance(text, str) for text in entry)):
                raise TypeError(f"{owner}: source slot '{slot}' needs a (unit, meaning) tuple of text, got {entry!r}")

    def feature_type(self, name):
        """FeatureType called `name`; KeyError listing the valid names."""
        for feature in self.features:
            if feature.name == name:
                return feature
        valid = ', '.join(feature.name for feature in self.features) or 'none'
        raise KeyError(f"physics type '{self.name}' has no feature type '{name}'; valid: {valid}")


def type_spaces(physics_type) -> tuple:
    """SPACES the type can run in: its spaces_2d when it runs in 2D, plus '3d' when it runs in 3D."""
    return ((physics_type.spaces_2d if 2 in physics_type.dimensions else ())
            + (('3d',) if 3 in physics_type.dimensions else ()))


def instance_spaces(physics_type, settings_values) -> tuple:
    """SPACES an instance with these settings runs in: the spaces whose space_settings it matches, else the type's
    spaces without space_settings."""
    matched = tuple(space for space, required in physics_type.space_settings.items()
                    if all(settings_values.get(name) == value for name, value in required.items()))
    return matched or tuple(space for space in type_spaces(physics_type) if space not in physics_type.space_settings)


def instance_space(physics_type, settings_values, is_3d) -> str:
    """The one SPACE an instance runs in: '3d' in a 3D model, else its first 2D instance space ('' when none)."""
    if is_3d:
        return '3d'
    return next((space for space in instance_spaces(physics_type, settings_values) if space != '3d'), '')


def param_in_space(param, space):
    """param with the unit of its space_units entry for `space` (param itself without one)."""
    unit = dict(param.space_units).get(space)
    return param if unit is None or unit == param.unit else replace(param, unit=unit)


def known_physics(spec, registry):
    """Yield (PhysicsSpec, PhysicsType) of the physics of spec in order, skipping unregistered types (validate reports
    them)."""
    for physics_spec in spec.physics:
        try:
            physics_type = registry.physics_type(physics_spec.type)
        except KeyError:
            continue
        yield physics_spec, physics_type


def model_spaces(spec, registry) -> tuple:
    """SPACES every physics of spec runs in (SPACES order; the domain's size_z does not restrict them, its polar /
    spherical coordinates do: COORDINATE_SPACES); () when they share none. Unknown types are skipped."""
    spaces = set(COORDINATE_SPACES.get(spec.domain.coordinates, SPACES))
    for physics_spec, physics_type in known_physics(spec, registry):
        settings_values = {param.name: param.default for param in physics_type.settings} | physics_spec.settings
        spaces &= set(instance_spaces(physics_type, settings_values))
    return tuple(space for space in SPACES if space in spaces)


def physics_type_misfit(physics_type, spec, registry) -> str:
    """'' when a new physics of physics_type shares a space with every physics of spec (and solves on the domain's
    grid and runs on the model's backend), else the reason as GUI text."""
    if solver_backend(spec) not in physics_type.backends:
        return ('Runs only on the graphics card; this model runs on the processor (Solvers node, Run on).'
                if solver_backend(spec) == 'cpu' else 'Runs only on the processor; this model runs on the graphics '
                                                      'card (Solvers node, Run on).')
    coordinates = spec.domain.coordinates
    if coordinates in COORDINATE_SPACES and 'polar' not in physics_type.coordinate_grids:
        return (f'Works only on a Cartesian domain grid; this model uses a {coordinates} grid (domain setting '
                "'Grid').")
    spaces = model_spaces(spec, registry)
    if set(type_spaces(physics_type)) & set(spaces):
        return ''
    if not spaces:
        return 'The physics already in this model do not fit together.'
    return (f"Works only in {' or '.join(SPACES[space] for space in type_spaces(physics_type))}; the physics in this "
            f"model need {' or '.join(SPACES[space] for space in spaces)}.")


def new_physics_space(physics_type, settings_values, spec, registry) -> tuple:
    """(space, settings) of a new physics: the first space shared with spec, preferring the domain's current
    dimension, and settings_values plus that space's space_settings; (None, settings_values) when none is shared."""
    shared = [space for space in model_spaces(spec, registry) if space in type_spaces(physics_type)]
    if not shared:
        return None, settings_values
    candidates = [space for space in shared if (space == '3d') == spec.is_3d] or shared
    own = instance_spaces(physics_type, settings_values)
    space = next((space for space in candidates if space in own), candidates[0])
    if space in own:
        return space, settings_values
    return space, {**settings_values, **physics_type.space_settings.get(space, {})}


def permittivity_arrays(material_property_params) -> tuple:
    """Ordered union of the MaterialGrid arrays the permittivity properties (PERMITTIVITY_PROPERTY_NAMES) write."""
    arrays = {}
    for param in material_property_params:
        if param.name in PERMITTIVITY_PROPERTY_NAMES:
            arrays.update(dict.fromkeys(param.expands_to))
    return tuple(arrays)


def array_owner(array_name, material_property_params, set_names):
    """Property (name in set_names) that writes array_name: the builder ownership rule (a component listed after
    eps_r wins); None when no set property writes it."""
    owner = None
    for param in material_property_params:
        if param.name in set_names and array_name in param.expands_to:
            owner = param.name
    return owner


@dataclass(frozen=True)
class CouplingType:
    """A coupling between two physics instances; source/target type: a physics type name, a tuple of them, or None
    (any type). Test endpoints with coupling_type_accepts."""
    name: str
    label: str
    source_type: str | tuple | None
    target_type: str | tuple | None
    params: tuple = ()
    description: str = ''
    summary: str = ''                  # GUI text (tree / menu tooltip, settings panel); '' -> description
    create: Callable | None = None     # create(ctx, coupling_spec, values, every) -> coupling object
    validate: Callable | None = None   # validate(vctx, coupling_spec, values) -> list[Issue]
    # writes(vctx, coupling_spec) -> {target slot: '' (overwrites) | 'sum' | 'index' (E13 contributor)} for the
    # validation writer rule (CPU, {} when unknown); must match the terms create returns
    writes: Callable | None = None
    # S7.2 source and target on different meshes: 'field' = the transform runs on the target mesh and every source
    # state field it reads is interpolated there; 'value' = the transform runs on the source mesh (reading only the
    # source's state) and its result moves by value_kind (TRANSFER_KINDS); None = both physics share one mesh (validate)
    mesh_transfer: str | None = None
    value_kind: str = 'conservative'
    # across meshes the transform reads the material arrays of the partner physics (the 'field' source / 'value'
    # target) through its view's domain: the builder paints that type's materials on the transform mesh too
    partner_materials: bool = False
    # the transform also reads the target's own state (fluid_heat_convection: heat T): a loop-back the stationary study
    # iterates, so the time estimate's fixed point counts it as a self-coupling of the target
    reads_target: bool = False

    def __post_init__(self):
        owner = f"CouplingType '{self.name}'"
        if self.mesh_transfer is not None and self.mesh_transfer not in MESH_TRANSFERS:
            raise ValueError(f"{owner}: mesh_transfer must be None or one of {', '.join(MESH_TRANSFERS)}, got "
                             f'{self.mesh_transfer!r}')
        if self.value_kind not in TRANSFER_KINDS:
            raise ValueError(f"{owner}: value_kind must be one of {', '.join(TRANSFER_KINDS)}, got "
                             f'{self.value_kind!r}')
        for role in COUPLING_ROLES:
            accepted = getattr(self, f'{role}_type')
            if not (accepted is None or isinstance(accepted, str)
                    or (isinstance(accepted, tuple) and accepted and all(isinstance(name, str) for name in accepted))):
                raise TypeError(f'{owner}: {role}_type must be None, a physics type name or a non-empty tuple of '
                                f'names, got {accepted!r}')
        _require_unique_names(self.params, 'param', owner)
        _check_params(self.params, owner)


def _coupling_role_types(coupling_type, role):
    """Physics type names a coupling role accepts as a tuple, or None (any type); ValueError for an unknown role."""
    if role not in COUPLING_ROLES:
        raise ValueError(f"coupling role must be one of {', '.join(COUPLING_ROLES)}, got {role!r}")
    accepted = getattr(coupling_type, f'{role}_type')
    return (accepted,) if isinstance(accepted, str) else accepted


def coupling_writes(vctx, coupling_spec) -> dict:
    """{target slot: contribution ('' overwrite, 'sum', 'index')} a coupling spec writes; {} for an unknown type or a
    type without a writes callback."""
    try:
        coupling_type = vctx.registry.coupling_type(coupling_spec.type)
    except KeyError:
        return {}
    return dict(coupling_type.writes(vctx, coupling_spec)) if coupling_type.writes is not None else {}


def coupling_type_accepts(coupling_type, role, physics_type_name) -> bool:
    """True when a physics of type physics_type_name may be the role ('source' | 'target') of the coupling type."""
    accepted = _coupling_role_types(coupling_type, role)
    return accepted is None or physics_type_name in accepted


def coupling_role_text(coupling_type, role) -> str:
    """Accepted physics types of a coupling role for messages: "'a'", "'a' or 'b'", or 'any'."""
    accepted = _coupling_role_types(coupling_type, role)
    return 'any' if accepted is None else ' or '.join(f"'{name}'" for name in accepted)


@dataclass(frozen=True)
class StudyType:
    """A study (time dependent, stationary, ...); its param values live in StudySpec.settings."""
    name: str
    label: str
    description: str = ''
    summary: str = ''                      # GUI text (tree tooltip, settings panel); '' -> description
    params: tuple = ()
    requires_time_stepping: bool = False   # needs >= 1 physics whose steps_in_time is True
    # a physics that supports this study type instead of the study itself runs as a background physics ('' = none)
    background_study: str = ''
    # validation remedy appended when a coupling / material reads a physics this study solves itself ('' = none)
    feedback_alternative: str = ''
    # True: a coupling may read a physics this study solves itself when it writes a background physics; the run then
    # alternates background and study solves to a fixed point (model.study._feedback_rounds; one operating point only)
    solution_feedback: bool = False
    # True: couplings / material properties may read a physics this study solves itself; the run then solves each
    # point of its own axis (frequency_domain: each wavelength) as a stationary fixed point of every physics
    coupled_points: bool = False
    run: Callable | None = None            # run(session, values, emit, control, service) -> dict
    validate: Callable | None = None       # validate(vctx, values) -> list[Issue], after the generic param checks
    # default_stored_fields(spec) -> 'physics.component' keys stored right after the primary fields when
    # StudySpec.stored_fields is [] (every available field follows them; None = no extra keys)
    default_stored_fields: Callable | None = None
    # StudySpec.adaptation allowed (S7.7: mesh refinement levels, model.study_adaptation); validate refuses it else
    supports_adaptation: bool = False
    # time_counts(values) -> {name: number} the run-time recipes read (model.time_estimate.StudyCounts: duration,
    # harmonic_points, max_duration, eigenpairs, iterations, fixed_point_limit); None = no counts (recipes assume one
    # solve)
    time_counts: Callable | None = None
    # model.solver_hooks Protocols: the solver of a physics whose type lists this study satisfies one of them (the
    # builder checks right after create); () = no study hook (time_dependent, stationary)
    solver_protocols: tuple = ()

    def __post_init__(self):
        _require_unique_names(self.params, 'param', f"StudyType '{self.name}'")
        _check_params(self.params, f"StudyType '{self.name}'")
        if not (isinstance(self.solver_protocols, tuple) and all(
                getattr(protocol, '_is_runtime_protocol', False) for protocol in self.solver_protocols)):
            raise TypeError(f"StudyType '{self.name}': solver_protocols must be a tuple of runtime-checkable "
                            f'Protocols (model.solver_hooks), got {self.solver_protocols!r}')


def _require_text(owner, **texts):
    """Raise TypeError naming the first keyword whose value is not a str."""
    for field_name, text in texts.items():
        if not isinstance(text, str):
            raise TypeError(f'{owner}: {field_name} must be text, got {type(text).__name__}')


@dataclass(frozen=True)
class TemplateEntry:
    """A ready-made example model; factory() returns a fresh ModelSpec (builtin: listed in model/templates.py)."""
    key: str
    label: str
    description: str          # one-line purpose for agents (describe('templates'))
    factory: Callable         # factory() -> fresh ModelSpec
    builtin: bool = False
    summary: str = ''         # GUI template menu tooltip; '' -> description

    def __post_init__(self):
        if not is_identifier(self.key):
            raise ValueError(f'template key {self.key!r} is not an identifier')
        _require_text(f"template '{self.key}'", label=self.label, description=self.description, summary=self.summary)
        if not callable(self.factory):
            raise TypeError(f"template '{self.key}': factory must be callable, got {type(self.factory).__name__}")


@dataclass(frozen=True)
class SelfTestVariant:
    """A repeat of a SelfTestCase's runs on another backend and / or solver precision, passed against its own
    tolerance (solver_plan S3.6): api.selftest re-runs specs() moved by api._variant_spec and re-uses check()."""
    name: str                       # identifier, unique within the case (report key, run directory variant_<name>)
    tolerance: float                # replaces the tolerance check() returns
    backend: str | None = None      # 'cpu' | 'gpu': solver_defaults.backend of every run; None = the specs' own
    precision: str | None = None    # PRECISIONS key set on the solver of every role in `roles`; None = unchanged
    # roles the precision applies to; () = every role whose resolved solver type has a 'precision' setting
    roles: tuple = ()

    def __post_init__(self):
        owner = f"selftest variant '{self.name}'"
        if not is_identifier(self.name):
            raise ValueError(f'{owner}: name is not an identifier')
        if isinstance(self.tolerance, bool) or not isinstance(self.tolerance, (int, float)) or not self.tolerance >= 0:
            raise ValueError(f'{owner}: tolerance must be a number >= 0, got {self.tolerance!r}')
        if self.backend not in (None, *SOLVER_BACKENDS[1:]):
            raise ValueError(f"{owner}: backend must be None, 'gpu' or 'cpu', got {self.backend!r}")
        if self.precision is not None and self.precision not in PRECISIONS:
            raise ValueError(f"{owner}: precision must be None or one of {', '.join(PRECISIONS)}, got "
                             f'{self.precision!r}')
        if not isinstance(self.roles, tuple) or not set(self.roles) <= set(SOLVER_ROLES):
            raise ValueError(f"{owner}: roles must be a tuple of {', '.join(SOLVER_ROLES)}, got {self.roles!r}")
        if self.roles and self.precision is None:
            raise ValueError(f'{owner}: roles need a precision')
        if self.backend is None and self.precision is None:
            raise ValueError(f'{owner}: set a backend, a precision or both')


@dataclass(frozen=True)
class SelfTestCase:
    """An analytic validation case: api.selftest runs specs(), then check() compares the results to theory."""
    name: str                 # identifier, unique
    label: str
    description: str          # what is compared, the analytic formula, why the tolerance
    specs: Callable           # specs() -> dict[str, ModelSpec] (run name -> spec; insertion order = run order)
    check: Callable           # check(results_by_name) -> {'measured', 'expected', 'error', 'tolerance', 'details'}
    max_wall_time: float = 600.0   # s per run
    heavy: bool = False       # True when the case took >= 5 s: run only by selftest --heavy / --all (api heavy=)
    # backends its runs / check need ('gpu': cupy + a CUDA device); () = from its runs (solver_defaults.backend 'cpu'
    # -> 'cpu', else 'gpu'; no runs -> ('cpu',)). Without a GPU api.selftest skips the gpu runs, and the case when none is left
    backends: tuple = ()
    # SelfTestVariants: the same runs again per backend / precision, each with its own tolerance (needs runs; a gpu
    # variant skipped without a usable GPU, a cpu one runs and judges the case also when its gpu runs are skipped);
    # the case passes only when every variant does
    variants: tuple = ()

    def __post_init__(self):
        if not is_identifier(self.name):
            raise ValueError(f'selftest name {self.name!r} is not an identifier')
        owner = f"selftest '{self.name}'"
        _require_text(owner, label=self.label, description=self.description)
        unknown_backends = [name for name in self.backends if name not in SOLVER_BACKENDS[1:]]
        if not isinstance(self.backends, tuple) or unknown_backends:
            raise ValueError(f"{owner}: backends must be a tuple of 'gpu' / 'cpu', got {self.backends!r}")
        if not (callable(self.specs) and callable(self.check)):
            raise TypeError(f'{owner}: specs and check must be callable')
        if (isinstance(self.max_wall_time, bool) or not isinstance(self.max_wall_time, (int, float))
                or not self.max_wall_time > 0):
            raise ValueError(f'{owner}: max_wall_time must be a number of seconds > 0, got {self.max_wall_time!r}')
        if not isinstance(self.heavy, bool):
            raise TypeError(f'{owner}: heavy must be a bool, got {self.heavy!r}')
        if not isinstance(self.variants, tuple) or not all(isinstance(item, SelfTestVariant) for item in self.variants):
            raise TypeError(f'{owner}: variants must be a tuple of SelfTestVariant, got {self.variants!r}')
        _require_unique_names(self.variants, 'variant', owner)


@dataclass(frozen=True)
class Issue:
    """One validation finding attached to a GUI node path."""
    level: str      # ISSUE_LEVELS
    path: tuple     # GUI node path
    message: str

    def __post_init__(self):
        if self.level not in ISSUE_LEVELS:
            raise ValueError(f"Issue level must be one of {', '.join(ISSUE_LEVELS)}, got {self.level!r}")
        if not isinstance(self.path, tuple):
            raise TypeError(f'Issue path must be a tuple, got {type(self.path).__name__}')


@dataclass
class ValidationContext:
    """Partially evaluated model handed to every validate callback (CPU only)."""
    spec: object
    registry: object
    parameters: dict
    parameter_errors: dict
    shapes: dict                      # geometry name -> ResolvedShape (failed objects absent)
    geometry_errors: dict
    mesh: object | None               # MeshEstimate of mesh_name (on_mesh), None if estimation failed
    mesh_error: str | None
    study_type: str
    state_names: dict                 # state_field_names(spec, registry)
    meshes: dict = field(default_factory=dict)        # used mesh name -> MeshEstimate | None (S7.2)
    mesh_errors: dict = field(default_factory=dict)   # used mesh name -> estimate failure message | None
    mesh_name: str = ''               # the mesh `mesh` / `mesh_error` describe: the default (first used) mesh, or
                                      # inside on_mesh the current physics' own
    _material_maps: dict = field(default_factory=dict, init=False, repr=False, compare=False)
    plain_roles: frozenset = frozenset()   # meshing.fitted_memory_estimate: roles resolved without study_defaults
    _solvers: dict = field(default_factory=dict, init=False, repr=False, compare=False)   # id -> (spec, roles)

    @contextmanager
    def on_mesh(self, name):
        """Within the block mesh / mesh_error / mesh_name describe mesh `name` (None: no estimate, e.g. a mesh kind
        the physics does not run on); the previous ones return afterwards."""
        previous = (self.mesh, self.mesh_error, self.mesh_name)
        self.mesh, self.mesh_error = self.meshes.get(name), self.mesh_errors.get(name)
        self.mesh_name = name or ''
        try:
            yield self
        finally:
            self.mesh, self.mesh_error, self.mesh_name = previous

    @cached_property
    def domain_size(self):
        """geometry.domain_size (size_x, size_y) [m], or None when it fails (reported at ('geometry',))."""
        from .geometry import domain_size as evaluate_domain_size
        try:
            return evaluate_domain_size(self.spec, self.parameters)
        except ValueError:
            return None

    @cached_property
    def domain_map(self):
        """geometry.domain_map (the 2D in-plane grid), or None when it fails (reported at ('geometry',))."""
        from .geometry import domain_map
        try:
            return domain_map(self.spec, self.parameters)
        except ValueError:
            return None

    @cached_property
    def size_z(self):
        """geometry.domain_size_z [m] of a 3D model; None for a 2D model or when it fails (reported at ('geometry',))."""
        from .geometry import domain_size_z
        try:
            return domain_size_z(self.spec, self.parameters)
        except ValueError:
            return None

    @property
    def mesh_uniform(self):
        """True when a tensor mesh estimate exists and its dx, dy (3D: dz) cell widths are each constant (rel 1e-6)."""
        return (self.mesh is not None and self.mesh.kind == 'tensor' and has_uniform_spacing(self.mesh.dx_cells)
                and has_uniform_spacing(self.mesh.dy_cells)
                and (self.mesh.dz_cells is None or has_uniform_spacing(self.mesh.dz_cells)))

    def values(self, params, raw):
        """resolve_values over the partial parameters; None on error (reported by the param checks)."""
        try:
            return resolve_values(params, raw, self.parameters)
        except ValueError:
            return None

    def solvers(self, physics_spec):
        """resolved_solvers of a physics in this study ({role: ResolvedSolver}); None when they do not resolve (unknown
        node or type, bad values: check_solvers reports those). Cached per physics spec (read-only for callers)."""
        cached = self._solvers.get(id(physics_spec))
        if cached is None or cached[0] is not physics_spec:   # the entry holds its spec: a reused id starts over
            try:
                roles = resolved_solvers(self.spec, self.registry, physics_spec, self.study_type, self.parameters,
                                         self.plain_roles)
            except (KeyError, ValueError, TypeError, AttributeError):
                roles = None
            cached = self._solvers[id(physics_spec)] = (physics_spec, roles)
        return cached[1]

    def objects(self, entities):
        """Geometry names of an entity list (selections expanded, outer faces and 'all' removed); [] on a bad
        selection (reported by the selection checks)."""
        from .geometry import expand_entities
        from .spec import RESERVED_ENTITY_NAMES
        try:
            expanded = expand_entities(self.spec, entities)
        except (ValueError, KeyError, TypeError):   # cycle, or a malformed (non-list / unhashable) entity list
            return []
        return [name for name in expanded if name not in RESERVED_ENTITY_NAMES]

    def material_map(self, max_cells=200):
        """Cached geometry.material_preview (index_image, material_names, extent), or None when it fails."""
        if max_cells not in self._material_maps:
            from .geometry import material_preview
            try:
                preview = material_preview(self.spec, self.parameters, max_cells=max_cells, shapes=self.shapes)
            except (ValueError, KeyError, TypeError, ArithmeticError):
                preview = None
            self._material_maps[max_cells] = preview
        return self._material_maps[max_cells]


class Registry:
    """Name -> type tables of physics, coupling, study and solver types, plus the templates and (lazily loaded)
    selftest cases that definition modules register; duplicate names raise."""

    def __init__(self):
        self._physics = {}
        self._couplings = {}
        self._studies = {}
        self._solver_types = {}
        self._templates = {}
        self._selftests = {}
        self._selftest_loaders = []
        self._selftest_lock = threading.Lock()

    @staticmethod
    def _add(table, item, expected_type):
        if not isinstance(item, expected_type):
            raise TypeError(f'expected a {expected_type.__name__}, got {type(item).__name__}')
        if item.name in table:
            raise ValueError(f"duplicate {expected_type.__name__} name '{item.name}'")
        table[item.name] = item

    @staticmethod
    def _lookup(table, name, what):
        try:
            return table[name]
        except KeyError:
            raise KeyError(f"unknown {what} '{name}'; registered: {', '.join(table) or 'none'}") from None

    def register_physics(self, physics_type):
        """Add a PhysicsType; ValueError when a material property disagrees with a same-named registered one or a
        SolverRole names an unregistered (or unfitting) default solver type."""
        if isinstance(physics_type, PhysicsType):
            for registered_type in self._physics.values():
                _require_matching_material_properties(registered_type, physics_type)
            for role, solver_role in physics_type.solver_roles.items():
                for type_name in {solver_role.default_type, *(name for name, _ in solver_role.backend_defaults.values()),
                                  *(name for name, _ in solver_role.grid_defaults.values())}:
                    solver_type = self.solver_type(type_name)
                    if role not in solver_type.roles or solver_role.operator not in solver_type.operators:
                        raise ValueError(f"PhysicsType '{physics_type.name}': default solver '{solver_type.name}' of "
                                         f"role '{role}' does not fill that role for {solver_role.operator} operators")
        self._add(self._physics, physics_type, PhysicsType)

    def register_coupling(self, coupling_type):
        self._add(self._couplings, coupling_type, CouplingType)

    def register_study(self, study_type):
        self._add(self._studies, study_type, StudyType)

    def register_solver_type(self, solver_type):
        """Add a SolverType (solvers.definition registers them before the physics definitions)."""
        self._add(self._solver_types, solver_type, SolverType)

    def solver_type(self, name):
        return self._lookup(self._solver_types, name, 'solver type')

    def solver_types(self):
        return list(self._solver_types.values())

    def physics_type(self, name):
        return self._lookup(self._physics, name, 'physics type')

    def coupling_type(self, name):
        return self._lookup(self._couplings, name, 'coupling type')

    def study_type(self, name):
        return self._lookup(self._studies, name, 'study type')

    def physics_types(self):
        return list(self._physics.values())

    def coupling_types(self):
        return list(self._couplings.values())

    def study_types(self):
        return list(self._studies.values())

    def register_template(self, key, label, description, factory, summary=''):
        """Add an example model (factory() -> fresh ModelSpec); model.templates.template_entries lists it after the
        built-in ones."""
        entry = TemplateEntry(key, label, description, factory, summary=summary)
        if key in self._templates:
            raise ValueError(f"duplicate template key '{key}'")
        self._templates[key] = entry

    def registered_templates(self):
        """Registered TemplateEntry list in registration order (built-ins excluded)."""
        return list(self._templates.values())

    def register_selftests(self, loader):
        """Add a selftest loader (loader() -> SelfTestCases, importing its selftests modules inside), run on the first
        selftests() call so get_registry imports no selftests module."""
        if not callable(loader):
            raise TypeError(f'selftest loader must be callable, got {type(loader).__name__}')
        self._selftest_loaders.append(loader)

    def selftests(self):
        """Every SelfTestCase in loader order; the first call runs the loaders (duplicate names raise)."""
        with self._selftest_lock:
            if self._selftest_loaders:
                cases = dict(self._selftests)
                for loader in self._selftest_loaders:
                    for case in loader():
                        self._add(cases, case, SelfTestCase)
                self._selftests, self._selftest_loaders = cases, []
        return list(self._selftests.values())

    def material_property_params(self):
        """[(physics type name, Param)] over every physics type's material_properties, first Param per name
        (register_physics keeps same-named ones equal in kind, unit, bounds and flags)."""
        seen, pairs = set(), []
        for physics_type in self._physics.values():
            for prop in physics_type.material_properties:
                if prop.name not in seen:
                    seen.add(prop.name)
                    pairs.append((physics_type.name, prop))
        return pairs


_ROOT_PACKAGE = __package__.rpartition('.')[0]
_registry = None
_registry_lock = threading.Lock()
_discovering_thread = None


def get_registry():
    """Process-wide Registry; the first call imports every definition module and calls its register(registry)."""
    global _registry, _discovering_thread
    if _discovering_thread == threading.get_ident():
        raise RuntimeError('get_registry() called from a register() function during registry discovery')
    with _registry_lock:
        if _registry is None:
            registry = Registry()
            _discovering_thread = threading.get_ident()
            try:
                for module_name in _definition_module_names():
                    module = importlib.import_module(module_name)
                    register = getattr(module, 'register', None)
                    if register is None:
                        raise AttributeError(f'definition module {module_name} has no register(registry)')
                    register(registry)
            finally:
                _discovering_thread = None
            _registry = registry
        return _registry


def _definition_module_names():
    """Built-in studies, the solver types, every physics/<pkg>/definition.py that exists, then the coupling
    definitions."""
    names = [f'{_ROOT_PACKAGE}.model.study', f'{_ROOT_PACKAGE}.solvers.definition']
    physics_package = importlib.import_module(f'{_ROOT_PACKAGE}.physics')
    for module_info in sorted(pkgutil.iter_modules(physics_package.__path__), key=lambda info: info.name):
        candidate = f'{physics_package.__name__}.{module_info.name}.definition'
        if module_info.ispkg and importlib.util.find_spec(candidate) is not None:
            names.append(candidate)
    names.append(f'{_ROOT_PACKAGE}.coupling.definitions')
    return names


def state_field_names(spec, registry):
    """Expression name -> (physics instance, field) over the state_fields of every physics in the spec:
    '<physics>_<field>' always, the plain field name when one physics alone provides it."""
    return _field_names(spec, registry, include_outputs=False)


def output_field_names(spec, registry):
    """Like state_field_names over state_fields + output_fields (derived-value expressions)."""
    return _field_names(spec, registry, include_outputs=True)


def scalar_output_names(spec, registry, parameters=None):
    """Expression name '<physics>_<scalar>' -> (physics instance, scalar) of every PhysicsType.scalar_outputs, settings
    resolved over `parameters` (None: the spec's own); physics whose type or settings do not resolve are skipped."""
    if parameters is None:
        parameters = evaluate_parameters(spec.parameters, errors={})
    names = {}
    for physics_spec, physics_type in known_physics(spec, registry):
        if physics_type.scalar_outputs is None:
            continue
        try:
            settings_values = resolve_values(physics_type.settings, physics_spec.settings, parameters)
        except (ValueError, TypeError, AttributeError):   # reported by validation
            continue
        for scalar in physics_type.scalar_outputs(settings_values, physics_spec):
            names[f'{physics_spec.name}_{scalar}'] = (physics_spec.name, scalar)
    return names


SOLVER_BACKENDS = ('auto', 'gpu', 'cpu')   # SolverDefaultsSpec.backend
# SolverDefaultsSpec.memory_check: the issue level of a memory estimate over the GPU / host budget ('warning': the
# runtime guards decide, i.e. the cupy pool limit and the cuDSS plan check)
MEMORY_CHECKS = ('error', 'warning')


def memory_check_level(spec) -> str:
    """Issue level of an over-budget memory estimate: solver_defaults.memory_check ('error' for an unknown value,
    which validate reports)."""
    level = spec.solver_defaults.memory_check
    return level if level in MEMORY_CHECKS else 'error'


def solver_backend(spec) -> str:
    """Backend of a run: solver_defaults.backend, 'auto' = 'gpu' when cupy imports, else 'cpu' (no driver query:
    GUI-thread safe; a machine with cupy but no CUDA device still resolves 'gpu', solver_plan S6 stage 0)."""
    backend = spec.solver_defaults.backend
    if backend == 'auto':
        return 'gpu' if cupy_installed() else 'cpu'
    return backend


def solver_grid(spec, physics_spec=None) -> str:
    """solvers.roles GRID_KINDS key of a physics' mesh (None: the default mesh): 'line' for a 1D model (size_y
    null), 'triangles' / 'tetrahedra' on a MeshSpec of that kind, else 'tensor_3d' in a 3D model and 'tensor' (also for
    an unknown mesh name)."""
    if spec.dimensions == 1:
        return 'line'
    tensor = 'tensor_3d' if spec.dimensions == 3 else 'tensor'
    try:
        mesh_spec = spec.mesh_of(physics_spec) if physics_spec is not None else spec.meshes[0]
    except (KeyError, IndexError):   # reported by validate
        return tensor
    return mesh_spec.kind if mesh_spec.kind in FREE_MESH_KINDS else tensor


SOLVER_DEFAULT_PRECISIONS = ('auto', *PRECISIONS)   # SolverDefaultsSpec.precision ('auto' = each solver's own default)


def with_default_precision(spec, solver_type, settings, solver_role=None) -> dict:
    """settings (raw) plus 'precision' = solver_defaults.precision (solver_plan S9) when settings set none and
    solver_type (and solver_role, when given) offer it; else settings unchanged. An explicit node precision wins."""
    precision = getattr(spec.solver_defaults, 'precision', 'auto')
    if (precision == 'auto' or 'precision' in settings or precision not in solver_type.precisions
            or not any(param.name == 'precision' for param in solver_type.params)
            or (solver_role is not None and precision not in solver_role.precisions)):
        return settings
    return {**settings, 'precision': precision}


def with_node_precision(spec, solver_type, settings, solver_role, backend, grid, study=None) -> dict:
    """with_default_precision of a node's settings; when that sets no precision and the node's type is the role's
    Automatic type on backend / grid (in the study), the Automatic precision (solid_mechanics_3d gpu tensor: mixed)."""
    filled = with_default_precision(spec, solver_type, settings, solver_role)
    if 'precision' in filled:
        return filled
    automatic_type, automatic_settings = solver_role.automatic(backend, grid, study)
    if automatic_type != solver_type.name or 'precision' not in automatic_settings:
        return filled
    return {**filled, 'precision': automatic_settings['precision']}


def default_precision_ignored(spec, registry, study) -> list:
    """[(physics name, role, reason)] of the roles in `study` whose resolved solver does not take
    solver_defaults.precision (type or role without it, or a node setting another one); [] when it is 'auto'."""
    precision = getattr(spec.solver_defaults, 'precision', 'auto')
    if precision == 'auto' or precision not in PRECISIONS:
        return []
    nodes = {node.name: node for node in spec.solvers}
    backend = solver_backend(spec)
    ignored = []
    for physics_spec, physics_type in known_physics(spec, registry):
        assigned = physics_spec.solvers if isinstance(physics_spec.solvers, dict) else {}
        grid = solver_grid(spec, physics_spec)
        for role, solver_role in physics_type.solver_roles.items():
            if not solver_role.used_in(study):
                continue
            node = nodes.get(assigned.get(role) or '')
            type_name = node.type if node is not None else solver_role.automatic(backend, grid, study)[0]
            try:
                solver_type = registry.solver_type(type_name)
            except KeyError:   # reported at the node
                continue
            if node is not None and isinstance(node.settings, dict) and 'precision' in node.settings:
                if node.settings['precision'] == precision:
                    continue
                reason = f"solver '{node.name}' sets its own precision"
            elif precision not in solver_type.precisions:
                reason = f'{solver_type.label} has no {PRECISIONS[precision]}'
            elif precision not in solver_role.precisions:
                reason = f'{physics_type.label} cannot run it in {PRECISIONS[precision]}'
            else:
                continue
            ignored.append((physics_spec.name, role, reason))
    return ignored


_SUBPIXEL_IGNORED_STUDY = 'optimization'   # its adjoint gradient (physics/em/adjoint) assumes the plain face means


def _on_element_mesh(spec, physics_spec) -> bool:
    """Whether the physics runs on a triangle / tetrahedron mesh (unknown mesh names count as not; validate reports
    them)."""
    mesh_spec = spec.find_mesh(physics_spec)
    return mesh_spec is not None and mesh_spec.kind in FREE_MESH_KINDS


def _honouring_physics(spec, registry, enabled, flag, extra_reason) -> tuple:
    """((honouring physics names), {ignoring name: reason}) of a model-wide geometry option: () and {} when off; a
    type without `flag` or on an element mesh ignores it, else extra_reason(physics_spec, physics_type) (None:
    honours)."""
    if enabled is not True:
        return (), {}
    honouring, ignoring = [], {}
    for physics_spec, physics_type in known_physics(spec, registry):
        if not getattr(physics_type, flag):
            reason = 'its type does not implement it'
        elif _on_element_mesh(spec, physics_spec):
            reason = 'element meshes follow the drawn outlines already'
        else:
            reason = extra_reason(physics_spec, physics_type)
        if reason is None:
            honouring.append(physics_spec.name)
        else:
            ignoring[physics_spec.name] = reason
    return tuple(honouring), ignoring


def subpixel_smoothing_physics(spec, registry) -> tuple:
    """((honouring physics names), {ignoring physics name: reason}) of solver_defaults.subpixel_smoothing: honoured
    by types declaring PhysicsType.subpixel_smoothing (2D or 3D) outside the optimization study, on either backend;
    ((), {}) when off. CPU, GUI-thread safe (validate, builder, memory estimate, GUI)."""
    def study_reason(physics_spec, physics_type):
        return ('the optimization study needs the plain face means for its gradient'
                if spec.study.type == _SUBPIXEL_IGNORED_STUDY else None)
    return _honouring_physics(spec, registry, spec.solver_defaults.subpixel_smoothing, 'subpixel_smoothing',
                              study_reason)


def conformal_boundaries_physics(spec, registry) -> tuple:
    """((honouring physics names), {ignoring physics name: reason}) of solver_defaults.conformal_boundaries: honoured
    by types declaring PhysicsType.conformal_boundaries in a model of their conformal_dimensions (2D: on their
    conformal_coordinates); ((), {}) when off. CPU, GUI-thread safe."""
    dimensions = spec.dimensions

    def domain_reason(physics_spec, physics_type):
        if dimensions not in physics_type.conformal_dimensions:
            return f'{dimensions}D models keep its staircased boundaries'
        if dimensions == 2 and spec.domain.coordinates not in physics_type.conformal_coordinates:
            return f'{spec.domain.coordinates} domains keep its staircased boundaries'
        return None
    return _honouring_physics(spec, registry, spec.solver_defaults.conformal_boundaries, 'conformal_boundaries',
                              domain_reason)


def resolved_solvers(spec, registry, physics_spec, study, parameters, plain_roles=frozenset()) -> dict:
    """{role: ResolvedSolver} of the roles the physics' type uses in `study`: its assigned solver node, else Automatic
    (SolverRole.automatic of the study, run backend and solver_grid: default_type with default_settings unless
    study_defaults / grid_defaults / backend_defaults differ; a (physics name, role) in plain_roles skips its
    study_defaults: meshing.fitted_memory_estimate); solver_defaults.precision fills a precision no node sets (with_default_precision; it
    also replaces an Automatic one), else a node of the Automatic type takes the Automatic one (with_node_precision); a
    preconditioner the node does not set resolves 'jacobi' at precision 'mixed'; back_end 'auto' resolves by
    auto_back_end_values (the role's cuDSS back end on the gpu backend). CPU, GUI-thread safe; ValueError
    naming the physics for an unknown node / type or bad values (validate reports them at their nodes first)."""
    physics_type = registry.physics_type(physics_spec.type)
    assigned = physics_spec.solvers if isinstance(physics_spec.solvers, dict) else {}
    nodes = {node.name: node for node in spec.solvers}
    backend, grid = solver_backend(spec), solver_grid(spec, physics_spec)
    resolved = {}
    for role, solver_role in physics_type.solver_roles.items():
        if not solver_role.used_in(study):
            continue
        node_name = assigned.get(role) or None
        try:
            if node_name is not None:
                node = nodes.get(node_name)
                if node is None or not isinstance(node.settings, dict):
                    raise ValueError(f"unknown solver '{node_name}'" if node is None
                                     else f"solver '{node_name}' settings must be a mapping")
                solver_type = registry.solver_type(node.type)
                raw = with_node_precision(spec, solver_type, node.settings, solver_role, backend, grid, study)
            else:
                automatic_study = None if (physics_spec.name, role) in plain_roles else study
                type_name, raw = solver_role.automatic(backend, grid, automatic_study)
                solver_type = registry.solver_type(type_name)
                # a role's Automatic precision (solid_mechanics_3d gpu: mixed) yields to solver_defaults.precision
                filled = with_default_precision(spec, solver_type,
                                                {key: value for key, value in raw.items() if key != 'precision'},
                                                solver_role)
                raw = filled if 'precision' in filled else raw
            values = resolve_values(solver_type.params, with_default_precision(spec, solver_type, raw, solver_role),
                                    parameters)
        except (KeyError, ValueError) as exc:
            text = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
            raise ValueError(f"physics '{physics_spec.name}' {role} solver: {text}") from exc
        node_sets_preconditioner = node_name is not None and 'preconditioner' in raw
        if (not node_sets_preconditioner and values.get('precision') == 'mixed'
                and values.get('preconditioner', 'jacobi') not in solver_role.mixed_preconditioners):
            values = {**values, 'preconditioner': 'jacobi'}   # mixed runs Jacobi only (Automatic multigrid, mesh M0)
        resolved[role] = ResolvedSolver(solver_type, auto_back_end_values(values, solver_role, backend, grid),
                                        node_name)
    return resolved


def study_default_roles(spec, registry, parameters) -> frozenset:
    """{(physics name, role)} of the Automatic roles whose solver comes from SolverRole.study_defaults on a cuDSS back
    end (a factor outside the pool), the roles meshing.fitted_memory_estimate may resolve without the study."""
    from ..solvers.sparse_direct import AUTO_BACK_END_KEY   # numpy only at module level
    study = spec.study.type
    roles = set()
    for physics_spec, physics_type in known_physics(spec, registry):
        grid = solver_grid(spec, physics_spec)
        assigned = physics_spec.solvers if isinstance(physics_spec.solvers, dict) else {}
        candidates = [role for role, solver_role in physics_type.solver_roles.items()
                      if not assigned.get(role) and (study, grid) in solver_role.study_defaults]
        if not candidates:
            continue
        try:
            resolved = resolved_solvers(spec, registry, physics_spec, study, parameters)
        except ValueError:   # reported by validate
            continue
        roles.update((physics_spec.name, role) for role in candidates
                     if role in resolved and resolved[role].values.get(AUTO_BACK_END_KEY))
    return frozenset(roles)


def auto_back_end_values(values, solver_role, backend, grid) -> dict:
    """values with back_end 'auto' replaced by the role's auto_back_ends[grid] (a cuDSS back end) on the gpu backend
    when nvmath-python is installed (find_spec: GUI-thread safe), marked by sparse_direct.AUTO_BACK_END_KEY True (the
    runtime falls back to host SuperLU when cuDSS refuses an operator, B1b); else values unchanged."""
    key = next((name for name in ('back_end', 'factorization') if values.get(name) == 'auto'), None)
    back_end = solver_role.auto_back_ends.get(grid)
    if key is None or back_end is None or backend != 'gpu':
        return values
    from ..solvers.cudss_factor import cudss_available   # numpy only at module level
    from ..solvers.sparse_direct import AUTO_BACK_END_KEY
    return {**values, key: back_end, AUTO_BACK_END_KEY: True} if cudss_available() else values


def restricted_feature_precisions(feature_type) -> tuple:
    """The stepper precisions feature_type is limited to, () when it works at every precision (solver_plan S3.3)."""
    return () if set(feature_type.precisions) == set(PRECISIONS) else feature_type.precisions


def feature_precision_note(feature_type) -> str:
    """GUI sentence naming the time stepping precisions feature_type needs, '' when it works at every one."""
    restricted = restricted_feature_precisions(feature_type)
    if not restricted:
        return ''
    return f"Works only when the time stepping solver runs in {' or '.join(PRECISIONS[name] for name in restricted)}."


def feature_precision_misfit(feature_type, resolved) -> str:
    """'' when feature_type works at the precision of the resolved 'stepper' solver ({role: ResolvedSolver}; no
    stepper = no restriction), else the reason naming that precision (solver_plan S3.3)."""
    stepper = resolved.get('stepper')
    if stepper is None:
        return ''
    precision = resolved_precision(stepper.solver_type, stepper.values)
    if precision in feature_type.precisions:
        return ''
    solver = f"stepper solver '{stepper.node}'" if stepper.node else 'the Automatic stepper solver'
    needed = ' or '.join(PRECISIONS[name] for name in feature_type.precisions)
    return f'{feature_type.label} does not work with {PRECISIONS[precision]} ({solver}); it needs {needed}'


def _field_names(spec, registry, include_outputs):
    """Shared body of state_field_names / output_field_names; physics of unknown type are skipped."""
    qualified, providers = {}, {}
    for physics_spec, physics_type in known_physics(spec, registry):
        infos = physics_type.state_fields + (physics_type.output_fields if include_outputs else ())
        for info in infos:
            qualified[f'{physics_spec.name}_{info.name}'] = (physics_spec.name, info.name)
            providers.setdefault(info.name, {})[physics_spec.name] = None
    names = {field_name: (next(iter(physics_names)), field_name)
             for field_name, physics_names in providers.items() if len(physics_names) == 1}
    names.update(qualified)
    return names


def feature_path(physics_spec, feature_spec) -> tuple:
    """GUI node path ('feature', physics name, feature name) of a feature."""
    return ('feature', physics_spec.name, feature_spec.name)


def node_label(path) -> str:
    """Report text of a node path: () -> 'model', ('mesh',) -> 'mesh', ('feature', 'em', 'src') -> 'feature em/src'."""
    if len(path) <= 1:
        return str(path[0]) if path else 'model'
    return f"{path[0]} {'/'.join(str(part) for part in path[1:])}"


def valid_every(every) -> bool:
    """True for a usable runner stride: an int >= 1 (bools excluded)."""
    return isinstance(every, int) and not isinstance(every, bool) and every >= 1


def effective_coupling_every(coupling_spec, source_spec, target_spec):
    """Transfer stride of a coupling: its every, else (None) the larger every of its source and target physics;
    None when a value it needs is not a valid every."""
    if coupling_spec.every is not None:
        return coupling_spec.every if valid_every(coupling_spec.every) else None
    if valid_every(source_spec.every) and valid_every(target_spec.every):
        return max(source_spec.every, target_spec.every)
    return None


_SPECTRAL_AXIS_DOCS = (('frequency', 'monitored frequencies, float64 (n_f,), Hz'),
                       ('wavelength', 'vacuum wavelengths c / frequency, float64 (n_f,), m'))
# result kind -> ((key, 'meaning, dtype (shape), unit'), ...) of the dicts Session.feature_result returns
RESULT_KIND_DOCS = {
    'timeseries': (
        ('kind', "'timeseries', str"),
        ('t', 'sample times, float64 (n,), s'),
        ('values', "sampled field values, float32 (n,), in 'unit'"),
        ('component', 'name of the sampled field component, str'),
        ('unit', "unit of 'values', str"),
    ),
    'flux_spectrum': (
        ('kind', "'flux_spectrum', str"),
        *_SPECTRAL_AXIS_DOCS,
        ('power', "power through the line per unit z length, float64 (n_f,), in 'unit'; > 0 toward +x (vertical "
                  'line) or +y (horizontal line); em_fdtd_3d: through the plane (> 0 toward +normal) or out of the '
                  'box, and power_per_cw in W'),
        ('unit', "unit of power, str: 'a.u.' (em_fdtd_*: dt-weighted spectra, compare runs by ratios) or 'W/m' "
                 '(em_fdfd_*: time-averaged power per unit z length)'),
        ('power_per_cw', 'em_fdtd_* only, when every source shares one waveform: power / (2 abs(S(omega))^2), the '
                         'cw power comparable with em_fdfd_* power, float64 (n_f,), W/m; NaN where the source '
                         'spectrum S is too weak (see the feature description)'),
    ),
    'dft_fields': (
        ('kind', "'dft_fields', str"),
        *_SPECTRAL_AXIS_DOCS,
        ('x_centers', 'x of the recorded cell centres (every box cell, or every downsample-th), float64 (mx,), m'),
        ('y_centers', 'y of the recorded cell centres, float64 (my,), m'),
        ('extent', 'box edges (x0, x1, y0, y1), 4 floats, m'),
        ('shape', 'recorded cell counts (mx, my), 2 ints, cells'),
        ('downsample', 'recording stride in cells along x and y (1 = every box cell), int'),
        ('quantity', "monitor quantity that intensity holds, str: 'abs(E_z)^2' (em_fdtd_ez) or 'abs(E_x)^2 + "
                     "abs(E_y)^2' (em_fdtd_hz, E_x and E_y averaged to the cell centres)"),
        ('intensity_per_cw_factor', 'only when every source shares one waveform: 1 / abs(S(omega))^2 per '
                                    'frequency, float64 (n_f,), 1/s^2; intensity times its freq_index entry is the cw '
                                    'intensity in (V/m)^2; NaN where the source spectrum S is too weak'),
        ('freq_index', 'only with the option freq_index: the chosen frequency index (0-based), int'),
        ('intensity', 'only with the option freq_index: abs(E(omega))^2 of the monitor quantity (see quantity) at '
                      'that frequency, float64 (mx, my), a.u.'),
        ('hz_intensity', 'em_fdtd_hz only, with the option freq_index on a live Results (closed and loaded Results '
                         'lack it): abs(H_z)^2 at that frequency, float64 (mx, my), a.u.'),
    ),
    'far_field': (
        ('kind', "'far_field', str"),
        *_SPECTRAL_AXIS_DOCS,
        ('quantity', "far-field quantity that power is proportional to, str: 'abs(E_z)^2' (em_fdtd_ez, em_fdfd_ez) "
                     "or 'abs(eta0 H_z)^2' (em_fdtd_hz, em_fdfd_hz)"),
        ('angles', 'observation angles counter-clockwise from +x, float64 (n_angles,), rad'),
        ('power', "radiated power per unit z length and angle, rho quantity / (2 eta0) at rho -> inf, float64 "
                  "(n_f, n_angles), in 'unit'; its integral over the angle is the power through a closed flux box"),
        ('unit', "unit of power, str: 'a.u.' (em_fdtd_*: dt-weighted spectra without the 1/2, as their "
                 "flux_spectrum) or 'W/(m rad)' (em_fdfd_*)"),
    ),
    'xy': (
        ('kind', "'xy', str; the producing feature may add further keys, listed in its description"),
        ('x', 'abscissa shared by every series, float64 (n,), in x_unit'),
        ('x_label', 'name of the abscissa, str'),
        ('x_unit', "unit of x, str ('1' = dimensionless, '' = none)"),
        ('y_label', 'name of the plotted quantity (y axis), str'),
        ('series', 'named curves {name: float64 (n,)} in plot order, each in its series_units unit; NaN where a '
                   'value cannot be computed'),
        ('series_units', 'unit of each series {name: str}, the same keys as series'),
    ),
    'table': (
        ('kind', "'table', str; the producing feature may add further keys, listed in its description"),
        ('columns', 'column names, list of str'),
        ('units', "unit of each column, list of str ('1' = dimensionless, '' = none)"),
        ('rows', 'one list per row with one value per column: number, str, bool or null; non-finite numbers are '
                 'floats, also in a loaded run (results.npz metadata stores them as "nan", "inf", "-inf")'),
    ),
}


MESH_PARAMS = (
    Param('kind', 'choice', 'tensor', label='Mesh kind', choices=MESH_KINDS,
          choice_labels=(('tensor', 'Rectangular cells'), ('triangles', 'Triangles (finite elements)'),
                         ('tetrahedra', 'Tetrahedra (finite elements, 3D)')),
          tip='Shape of the mesh elements.\n\n'
              '- Rectangular cells: rows and columns of cells, uniform or graded. Every physics runs on them.\n'
              '- Triangles: free triangles that follow curved boundaries, for the physics that support them (2D '
              'Cartesian models only).\n'
              '- Tetrahedra: free tetrahedra that follow spheres, boxes and cylinders, for the physics that support '
              'them (3D Cartesian models only; best with the Gmsh mesher).',
          help="tensor: rectilinear cells (uniform or graded rows and columns; every physics type); triangles: a "
               "conforming triangle mesh for finite elements (2D Cartesian models only); tetrahedra: a tetrahedron "
               "mesh for finite elements (3D Cartesian models only; mesher 'gmsh' conforms to boxes, spheres, z "
               "cylinders and z-prism rectangles / circles, composite inputs included, other objects are tagged "
               "per element; 'builtin' = a Kuhn split of the bricks of a rectilinear grid graded like a tensor mesh, "
               "see mesher). Physics, features and solvers declare the kinds they run on, validation refuses the "
               "others. On triangles / gmsh tetrahedra dx is the maximum element size, max_ratio the maximum element "
               "growth rate, refinements the maximum sizes in or along their objects (margin and dx_x / dx_y / dx_z "
               "unused; gmsh tetrahedra also along the edges or at the surface only: RefinementSpec.where), axes "
               "and outer_dx / outer_ratio do not apply; builtin tetrahedra take the tensor rules "
               "(refinements with margin and dx_x / dx_y / dx_z, axes, outer_dx / outer_ratio; no size_points); "
               "h_min, curvature_factor and mesher apply to triangles / tetrahedra only, pml_layers and mirror_x / "
               "mirror_y to triangles only (a tetrahedron PML shell is meshed like the window), face_sizes to "
               "tetrahedra only; size_points are [x, y, h] on triangles, [x, y, z, h] on gmsh tetrahedra."),
    Param('mode', 'choice', 'uniform', label='Mesh mode', choices=('uniform', 'nonuniform', 'auto'),
          choice_labels=(('uniform', 'Uniform'), ('nonuniform', 'Graded'), ('auto', 'Automatic')),
          tip='How the cell sizes are chosen.\n\n'
              '- Uniform: one cell size everywhere.\n'
              "- Graded: 'Cell size' plus finer cells in the refinements below, growing smoothly in between.\n"
              "- Automatic: with a wave physics on this mesh, a cell size per material from 'Shortest wavelength' "
              "and 'Points per wavelength'; without one, a fiftieth of the longest domain side, refined to a quarter "
              'of the smallest side of every object its physics use; plus the refinements.\n\n'
              'Automatic does not work with material values that depend on position or on other fields: use Graded '
              'there.\n\n'
              "A 'Cell size' in Automatic mode is the smallest cell size: the automatic sizes are scaled to it "
              "instead of following 'Points per wavelength'; empty, the rule above alone sizes the cells.\n\n"
              'Automatic shows the sizes it chose. Switching from Automatic to Graded fills them into the cell size '
              'and the refinements (marked automatic) so you can change them; choosing Automatic again recalculates '
              'them. Switching from Automatic to Uniform keeps the cell size (empty: fills in the smallest automatic '
              'size).',
          help='uniform: one cell size dx; nonuniform: base size dx plus the mesh refinements, graded by max_ratio; '
               'auto with an EM physics (family em) on this mesh or wavelength_min set: cell size wavelength_min / '
               '(points_per_wavelength * n) in each material, n = |sqrt(eps * mu_r)| '
               'at wavelength_min, largest over the permittivity arrays the mesh\'s physics types use, eps = the '
               'array background (a component such as eps_zz over eps_r, else the Drude eps_inf) - j sigma_e / '
               '(omega eps0) + the dispersion poles at that wavelength (a metal is meshed by its skin depth: gold '
               '|n| ~11 at 1.55 um), plus the refinements; auto without either (physics-controlled, S7.8): base '
               'size = the longest domain side / 50, every domain object the mesh\'s physics use (material '
               'selections setting one of their properties, their feature selections) refined to its smallest '
               'bounding-box side / 4, plus the refinements; with laminar_flow on the mesh a uniform grid at the '
               'smallest such side / 16 (lattice Boltzmann); a set dx scales every auto size so the finest equals '
               'it (see dx; null = the rule alone). auto raises for a permittivity, sigma_e, mu_r or pole '
               'value it uses that depends on x, y, z or a field (use nonuniform there). '
               'Graded meshes are rectilinear (one width per grid column and per row): a refinement refines every '
               'column and row its object plus margin spans, across the whole domain, so a waveguide crossing the '
               'domain refines that whole axis unless the refinement sets per-axis sizes (dx_x / dx_y / dx_z; a '
               'value >= the base size leaves that axis unrefined); the grading ramps lie outside the refined bands. '
               'MeshSpec.axes adds per-axis options: max_ratio, coordinate bands, and custom edges or segments '
               '(uniform / geometric / tanh) that replace the graded cells of that axis. '
               'In 3D models z is graded the same way: a refinement refines the z layers of its object\'s z extent '
               'plus margin (points / lines at their z, unset z = the whole height; an open prism end reaches the '
               'domain edge), set z0 / z1 '
               'faces of prisms are exact cell edges, and z cells away from refinements take the base size. '
               'nonuniform and auto fall back to a uniform grid when no refinement is finer than the base size, or '
               '(at the finest size) when a feature needs a uniform mesh (FDFD plane wave, laminar flow faces).'),
    Param('dx', 'float', 20e-9, label='Cell size', unit='m', min=0.0, min_exclusive=True, optional=True,
          tip='Size of every cell of a uniform mesh, or of the largest cells of a graded mesh (on triangles: the '
              'maximum element size).\n\n'
              'For waves use about a twentieth of the wavelength inside the material with the highest refractive '
              "index.\n\n"
              "In Automatic mode it is the smallest cell size: the automatic sizes are scaled so the finest cells "
              "get it, which replaces 'Points per wavelength'. Empty: the automatic rule alone sizes the cells.",
          help='Cell size of a uniform mesh, or the base (coarsest) size of a nonuniform one (required in both); '
               'typically lambda / (20 n) for EM, n the largest refractive index. In auto mode the smallest cell '
               'size: every auto size (base and object refinements, of the wavelength or the physics-controlled '
               'rule) is scaled by one factor so the finest equals dx, replacing points_per_wavelength (the '
               'wavelength rule still sets the size ratios between materials); the lattice Boltzmann rule takes it '
               'as its uniform size; null = the auto rule alone. Explicit refinements are not scaled. The grid is '
               'limited to 400e6 cells and '
               '50,000 cells per axis, counting the PML cells that are added outside the domain; a uniform mesh '
               'rounds size_x / size_y up to whole cells, so the meshed window can be one cell larger. Validation '
               'previews materials on float64 centres while the build tests float32 clamped cell centres, so a '
               'material edge exactly on a cell centre can differ: avoid placing edges exactly on cell centres.'),
    Param('max_ratio', 'float', None, label='Maximum growth ratio', unit='1', min=1.0, min_exclusive=True,
          visible_if=(('mode', ('nonuniform', 'auto')), ('kind', FREE_MESH_KINDS)), optional=True,
          tip='Largest size ratio between neighbouring cells of a graded or automatic mesh, or between '
              'neighbouring triangles or tetrahedra.\n\n'
              '1.1 to 1.3 is typical. Smaller values grade more smoothly, with fewer reflections, but need more '
              'cells. Empty: 1.15 for rectangular cells, 1.3 for triangles.',
          help='Largest size ratio of neighbouring cells on a graded mesh, or the maximum element growth rate of a '
               'triangle mesh (> 1); typically 1.1-1.3. Smaller ratios grade more smoothly (less numerical '
               'reflection) but need more cells. null = the kind default: 1.15 (tensor), 1.3 (triangles, the '
               'industry-standard default).'),
    Param('wavelength_min', 'float', None, label='Shortest wavelength', unit='m', min=0.0, min_exclusive=True,
          visible_if=(('mode', ('nonuniform', 'auto')),), optional=True,
          tip='Shortest vacuum wavelength the automatic mesh must resolve, in metres: 1.55e-6 for 1550 nm. Used in '
              'Automatic mode only.\n\n'
              'Empty: the shortest source or solved wavelength of the physics. Set it to keep one mesh during a '
              'sweep.',
          help='Shortest vacuum wavelength to resolve in auto mode. The auto cell size in a material of index n is '
               'wavelength_min / (points_per_wavelength * n). Empty (null): the shortest wavelength over the '
               'physics: FDFD and em_modes the shortest solved one (a frequency_domain study: its shortest '
               'wavelength, else the wavelength setting), FDTD the shortest significant source wavelength (gaussian lambda / (1 + '
               'bandwidth), ricker lambda / 2) of the physics on this mesh; a model error when an EM physics is on '
               'the mesh but none gives one (without EM physics auto mode takes the physics-controlled rule). A '
               'parameter sweep that changes that wavelength re-meshes per value. Set it to keep one mesh, to mesh '
               'finer, or for a wavelength the physics do not know (e.g. one generated inside the model).'),
    Param('points_per_wavelength', 'float', None, label='Points per wavelength', unit='1', min=1.0,
          visible_if=(('mode', ('nonuniform', 'auto')),), optional=True,
          tip='Cells per wavelength inside each material, for the automatic mesh (used in Automatic mode only).\n\n'
              '15 to 30 is typical for rectangular cells, about 5 for triangles. Too few cause phase errors that grow '
              "with the distance travelled. Empty: 20 for rectangular cells, 5 for triangles. Not used while 'Cell "
              "size' is set: it fixes the smallest cells instead.",
          help='Cells (triangles: elements) per wavelength inside each material in auto mode; typically 15-30 on '
               'tensor meshes. Too few cells give numerical dispersion (phase errors that grow with the propagation '
               "distance). null = the kind default: 20 (tensor), 5 (triangles: the industry-standard five quadratic "
               'elements per wavelength). Unused while dx is set (the finest cells take dx).'),
    Param('outer_dx', 'float', None, label='Outer cell size', unit='m', min=0.0, min_exclusive=True,
          visible_if=(('mode', ('nonuniform', 'auto')),), optional=True,
          tip='Largest cell size far from the structure. Leave empty to keep the main cell size everywhere.\n\n'
              'Outside the refined region the cells keep growing up to this size, so a large domain (for example '
              'a thousand times the structure) costs only a few dozen extra cells per side. Meant for heat, '
              'electrostatics and other slowly decaying fields, not for wave physics.',
          help='Outer grading (graded meshes; null = off): outside the refined core of each axis (the span of every '
               'refinement and band on that axis, margins included) cells grow geometrically from the base size '
               'by outer_ratio per cell up to outer_dx, so a domain ~1e3 x the structure costs a few dozen cells '
               'per side (quasi-static far fields: steady heat, electrostatics, low-frequency thermal noise). An '
               'axis without refinements is not outer graded; outer_dx <= the base size does nothing. Independent '
               'of the PML ramp (PML cells still grade toward the base size). Jacobi-PCG iterations grow with the '
               'grading: sparse direct solvers suit it better.'),
    Param('outer_ratio', 'float', None, label='Outer growth ratio', unit='1', min=1.0, min_exclusive=True,
          visible_if=(('mode', ('nonuniform', 'auto')),), optional=True,
          tip='Size ratio between neighbouring cells of the outer grading. Empty: the maximum growth ratio.',
          help="Cell growth per cell of the outer grading (> 1; null = the axis' max_ratio); 1.2-1.5 is typical "
               'for quasi-static fields.'),
    Param('h_min', 'float', None, label='Minimum element size', unit='m', min=0.0, min_exclusive=True,
          visible_if=(('kind', FREE_MESH_KINDS),), optional=True,
          tip='Smallest element size the mesher may make, for example at sharp corners or tight curves. Empty: a '
              'hundredth of the maximum element size.',
          help='Triangles / tetrahedra: smallest element size (> 0, <= the maximum size); null = the maximum size '
               '(dx, or the auto size) / 100.'),
    Param('curvature_factor', 'float', 0.3, label='Curvature factor', unit='1', min=0.0, min_exclusive=True,
          visible_if=(('kind', FREE_MESH_KINDS),),
          tip='Element edges along a curved boundary are at most this fraction of its radius of curvature: '
              'smaller values follow circles, ellipses, spheres and cylinders more closely.',
          help="Triangles / tetrahedra: element size along circle / ellipse / sphere / cylinder boundaries <= "
               "curvature_factor x the local radius of curvature (0.3 = the "
               "industry-standard default)."),
    Param('mesher', 'choice', 'builtin', label='Mesher', choices=('builtin', 'gmsh'),
          choice_labels=(('builtin', 'Built-in'), ('gmsh', 'Gmsh (external program)')),
          visible_if=(('kind', FREE_MESH_KINDS),),
          tip='Program that makes the elements. Gmsh must be installed separately; set the environment variable '
              'OPEN_SOL_GMSH to its path when it is not on the PATH. Tetrahedra: only Gmsh follows curved '
              'surfaces; Built-in splits the boxes of a rectangular grid into tetrahedra. That grid is graded '
              'like rectangular cells (refinements with sizes per axis, face sizes and the per-axis options), '
              'so it suits thin or long flat structures; curved objects get a staircase.',
          help="Triangles: 'builtin' = open_sol's conforming Delaunay refinement (numpy / scipy, deterministic); "
               "tetrahedra: 'builtin' = the bricks of a rectilinear grid split into 6 tetrahedra each (Kuhn "
               "orthoschemes: every dihedral angle <= 90 deg, no slivers at any brick aspect ratio; periodic sides "
               'match). Uniform mode, or nothing finer than dx: equal cells <= dx per axis. Else the grid is graded '
               'like a nonuniform tensor mesh at base dx: refinements (and auto ones) with margin and dx_x / dx_y / '
               'dx_z, axes (bands, edges, segments, max_ratio), outer_dx / outer_ratio, face_sizes as bands '
               '[c - distance, c + distance] across their face plane, and every flat face of a domain object (box, '
               'rectangle prism sides, set prism ends) snapped as an element face. The PML shell takes cells of dx. '
               'Curved objects are tagged per element (a staircase); size_points are refused (mesher gmsh), h_min '
               "and a physics' edge sizes are unused (a warning). Example, a slender clamped string: refinement "
               '{string, dx 0.5 um, dx_y = dx_z = 50 nm} plus face_sizes x_lo / x_hi 50 nm. '
               "'gmsh' = the gmsh program run as an external process (OPEN_SOL_GMSH or the PATH; never linked), a "
               'validation error when it is not found.'),
    Param('sweep_axis', 'choice', 'none', label='Swept layers', choices=('none', 'z'),
          choice_labels=(('none', 'Off'), ('z', 'Along z (prism layers)')),
          visible_if=(('kind', ('tetrahedra',)),),
          tip='Along z: triangles are made in the x-y plane (by the chosen mesher) and stacked in z layers; each '
              'prism between two layers is cut into 3 tetrahedra.\n\n'
              'For thin plates, films and other flat structures: a few thin layers across the thickness instead of '
              "many small tetrahedra. The in-plane size is 'Cell size' and the refinement sizes; the layer "
              'thickness comes from the z grading (refinement z sizes, z per-axis options, face sizes on the top '
              'and bottom faces). Every flat top or bottom face of an object is a layer boundary.\n\n'
              'Boxes, z cylinders and straight z prisms follow exactly; spheres and tapered or x / y extruded shapes '
              '(except rectangles) get a staircase.',
          help="Tetrahedra only (experiment plan D5). 'none' = the 3D mesher of mesher. 'z' = swept prism layers: "
               'the xy footprint meshed by the triangle mesher of mesher (builtin / gmsh 2D; dx = maximum element '
               'size, refinement dx in / along their objects\' footprints over the whole height, h_min, '
               "curvature_factor, max_ratio growth; a physics' edge sizes refine along the objects' outlines), "
               'extruded through the z layers of the tensor planner (as builtin tetrahedra along z: refinements '
               '(dx_z, else dx) with margins, a z MeshAxisSpec, face_sizes on z_lo / z_hi faces as bands, outer_dx; '
               'every set z face of a domain object snapped as a layer plane; the PML shell in cells of dx) and '
               'each prism split into 3 tetrahedra by the minimum-vertex rule (conforming; periodic sides matched; '
               'about 3 x triangles x layers elements). Conforming: boxes, z cylinders, untapered z prisms and '
               'untapered x / y rectangles; spheres, other x / y prisms and tapers are tagged per element (a '
               'warning). The x / y PML ring has pml_layers structured layers. Refused: size_points, x / y axes '
               'options, dx_x / dx_y unlike dx, face sizes on x / y faces, where edges / surface.'),
    Param('pml_layers', 'int', 8, label='Absorbing layers', unit='1', min=1.0,
          visible_if=(('kind', ('triangles',)), ('sweep_axis', ('z',))),
          tip='Number of structured element layers across each absorbing (PML) boundary layer of a triangle mesh '
              '(swept layers: of the x and y sides).',
          help='Triangles (and swept tetrahedra, sweep_axis z: the x / y sides): structured element layers across '
               'each PML ring side (thickness = the PML cells of the physics x the maximum element size, the tensor '
               'CPML stretch profile).'),
    Param('mirror_x', 'bool', False, label='Mirror-symmetric in x', visible_if=(('kind', ('triangles',)),),
          tip='Make the triangles the mirror image of themselves across the vertical centre line of the domain: '
              'the left half is meshed and copied to the right.\n\n'
              'For structures that are symmetric left to right, so the results keep the symmetry exactly (no '
              'mixing of even and odd modes by an uneven mesh). The mesh follows the drawn objects and their '
              'mirror images; the absorbing layers on the left and right must be equal.',
          help='Triangles: mesh the half window x <= size_x / 2 (it conforms to the geometry and its mirror image), '
               'then mirror it about x = size_x / 2 (nodes on the line shared, the PML ring mirrored too), so the '
               'mesh and every operator on it are exactly mirror symmetric: an even drive gives no odd response '
               'beyond rounding (parity selection rules). The size field is the smaller of the sizes at a point and '
               'at its image. Needs equal absorbing rings on x_lo and x_hi (else the mesh estimate fails); a '
               'warning names material objects that are not mirror symmetric (the mesh still conforms to them; '
               'sources and probes are not checked). Default false.'),
    Param('mirror_y', 'bool', False, label='Mirror-symmetric in y', visible_if=(('kind', ('triangles',)),),
          tip='Like Mirror-symmetric in x, across the horizontal centre line: the bottom half is meshed and copied '
              'to the top. Both together mesh one quarter.',
          help='Triangles: as mirror_x about y = size_y / 2 (equal absorbing rings on y_lo and y_hi); with mirror_x '
               'the quarter window is meshed and mirrored twice. Default false.'),
)


def fill_defaults(params, values):
    """Raw values with every missing param set to (a copy of) its default; unknown keys are kept."""
    filled = {param.name: copy.deepcopy(param.default) for param in params}
    filled.update(values)
    return filled


def param_visible(param, values):
    """visible_if semantics: no conditions, or ANY (param, accepted values) pair matches `values`.
    `values` must include defaults (fill_defaults) — a missing controlling key reads as None."""
    return not param.visible_if or any(values.get(controller) in accepted
                                       for controller, accepted in param.visible_if)


def resolve_values(params, values, parameters):
    """Raw spec values -> runtime values (defaults filled); ValueError naming the param on bad input.
    Invisible params whose value fails resolve to their default (None if that fails too)."""
    known = [param.name for param in params]
    unknown = sorted(set(values).difference(known))
    if unknown:
        raise ValueError(f"unknown parameter{'s' if len(unknown) > 1 else ''} {', '.join(unknown)}; "
                         f"valid: {', '.join(known) or 'none'}")
    raw_values = fill_defaults(params, values)
    resolved = {}
    for param in params:
        try:
            resolved[param.name] = _convert(param, raw_values[param.name], parameters, None)
        except ValueError as exc:
            if param_visible(param, raw_values):
                raise ValueError(f'{param.name}: {exc}') from exc
            try:
                resolved[param.name] = _convert(param, copy.deepcopy(param.default), parameters, None)
            except ValueError:
                resolved[param.name] = None
    return resolved


def check_value(param, raw_value, parameters, allowed_names=frozenset(), functions=None):
    """Problem message for a raw value, or None; never raises for user input. Expressions may use parameters,
    builtins, x/y (field_dependent or kind 'expression') and allowed_names (state_dependent or 'expression');
    functions: extra call names {name: (min, max) args} of kind 'expression' (derived-curve calculus)."""
    try:
        _convert(param, raw_value, parameters, frozenset(allowed_names), functions)
    except (ValueError, TypeError, KeyError, ArithmeticError, RecursionError, MemoryError) as exc:
        return str(exc)
    return None


def _convert(param, raw, parameters, state_names, functions=None):
    """Raw value -> runtime value; state_names None = resolve mode (state-dependent names unchecked); functions: extra
    call names of kind 'expression' (compile_expression)."""
    if raw is None or (param.kind == 'ref' and raw == ''):
        if param.optional:
            return None
        raise ValueError('a value is required')
    kind = param.kind
    if kind == 'float':
        return _number(param, raw, parameters, _expression_names(param, state_names))
    if kind == 'int':
        value = _number(param, raw, parameters, frozenset())
        if value != math.floor(value):
            raise ValueError(f'must be an integer, got {value:g}')
        return int(value)
    if kind == 'bool':
        if not isinstance(raw, (bool, np.bool_)):
            raise ValueError(f'expected true or false, got {raw!r}')
        return bool(raw)
    if kind == 'choice':
        if raw not in param.choices:
            raise ValueError(f"{raw!r} is not one of {', '.join(map(str, param.choices))}")
        return raw
    if kind in ('str', 'ref'):
        if not isinstance(raw, str):
            raise ValueError(f'expected text, got {short_repr(raw)}')
        return raw
    if kind == 'names':
        if not isinstance(raw, (list, tuple)) or not all(isinstance(name, str) for name in raw):
            raise ValueError(f'expected a list of names, got {short_repr(raw)}')
        return list(raw)
    if kind == 'expression':
        if isinstance(raw, str):
            expression = compile_expression(raw, functions)
        else:
            expression = compile_expression(repr(evaluate_scalar(raw, {})))
        if state_names is not None:
            expression.check_names(FIELD_COORDINATE_NAMES.union(parameters, state_names))
        return expression
    if kind == 'float_list':
        if not isinstance(raw, (list, tuple)):
            raise ValueError(f'expected a list of numbers, got {short_repr(raw)}')
        return [_list_item(f'item {index + 1}', param, item, parameters) for index, item in enumerate(raw)]
    if kind == 'table':
        return _table_rows(param, raw, parameters)
    if not isinstance(raw, (list, tuple)):   # 'points'
        raise ValueError(f'expected a list of (x, y) points, got {short_repr(raw)}')
    points = []
    for index, point in enumerate(raw):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f'point {index + 1}: expected (x, y), got {short_repr(point)}')
        points.append((_list_item(f'point {index + 1} x', param, point[0], parameters),
                       _list_item(f'point {index + 1} y', param, point[1], parameters)))
    return points


def _expression_names(param, state_names):
    """Names besides parameters a float param may reference; None = unchecked (resolve mode, state-dependent)."""
    if param.state_dependent and state_names is None:
        return None
    names = FIELD_COORDINATE_NAMES if param.field_dependent else frozenset()
    if param.state_dependent:
        return names | state_names | TIME_NAMES
    return names | TIME_NAMES if param.time_dependent else names


_ALLOWED_NAME_TEXTS = {   # field_names of _number -> what the value may use (unknown-name messages)
    frozenset(): 'this value may use parameters only',
    FIELD_COORDINATE_NAMES: 'this value may use parameters and x, y (z in 3D models) only',
    TIME_NAMES: 'this value may use parameters and time t only',
    FIELD_COORDINATE_NAMES | TIME_NAMES: 'this value may use parameters with either x, y (z in 3D models) or time t',
}


def _number(param, raw, parameters, field_names):
    """Number/expression -> bounds-checked float, or an Expression when it references names beyond the
    parameters (allowed only when they are in `field_names`; None = any). A time_dependent value of t takes no x, y,
    z (unless the param is state_dependent)."""
    if isinstance(raw, str):
        expression = compile_expression(raw)
        if not expression.names.issubset(parameters):
            if field_names is not None:
                expression.check_names(field_names.union(parameters), _ALLOWED_NAME_TEXTS.get(field_names))
                if (not param.state_dependent and not expression.names.isdisjoint(TIME_NAMES)
                        and not expression.names.isdisjoint(FIELD_COORDINATE_NAMES)):
                    raise ValueError(f'{expression.text!r} uses time t together with x / y / z: a feature value that '
                                     'changes in time is uniform in space (parameters and t only)')
            return expression
        raw = expression
    value = evaluate_scalar(raw, parameters)
    if param.min is not None:
        if param.min_exclusive and not value > param.min:
            raise ValueError(f'must be > {param.min:g}, got {value:g}')
        if not param.min_exclusive and not value >= param.min:
            raise ValueError(f'must be >= {param.min:g}, got {value:g}')
    if param.max is not None and not value <= param.max:
        raise ValueError(f'must be <= {param.max:g}, got {value:g}')
    return value


def _list_item(position, param, item, parameters):
    """One float of a float_list / points / table value; errors are prefixed with `position`."""
    try:
        return _number(param, item, parameters, frozenset())
    except ValueError as exc:
        raise ValueError(f'{position}: {exc}') from exc


def _table_rows(param, raw, parameters):
    """'table' value -> [row tuple of bounds-checked floats, ...] (>= 1 row; increasing: >= 2 rows, every column
    strictly increasing down the rows)."""
    names = [name for name, _ in param.columns]
    row_text = f"[{', '.join(names)}]"
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f'expected a list of {row_text} rows, got {short_repr(raw)}')
    rows = []
    for index, row in enumerate(raw):
        if not isinstance(row, (list, tuple)) or len(row) != len(names):
            raise ValueError(f'row {index + 1}: expected {row_text}, got {short_repr(row)}')
        rows.append(tuple(_list_item(f'row {index + 1} {name}', param, item, parameters)
                          for name, item in zip(names, row)))
    minimum_rows = 2 if param.increasing else 1
    if len(rows) < minimum_rows:
        raise ValueError(f'needs at least {minimum_rows} rows, got {len(rows)}')
    if param.increasing:
        for column, name in enumerate(names):
            for index in range(1, len(rows)):
                if not rows[index][column] > rows[index - 1][column]:
                    raise ValueError(f'column {name} must increase strictly down the rows: row {index + 1} '
                                     f'({rows[index][column]:g}) <= row {index} ({rows[index - 1][column]:g})')
    return rows


_check_params(MESH_PARAMS, 'MESH_PARAMS')   # here: _check_params needs _convert
