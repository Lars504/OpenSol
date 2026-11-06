"""Builds a validated ModelSpec into solvers (Domain, materials, physics, features, couplings) -> Session, on the
run backend (schema.solver_backend: cupy arrays on the gpu, numpy on the cpu; `ctx.xp` / `domain.xp`)."""
import math
import time
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, replace

import numpy as np

from ..core import gpu_memory, host_memory
from ..core.backend import array_module_of, to_host
from ..core.box_values import BoxValues
from ..core.constants import DTYPE
from ..coupling.material_update import DynamicMaterialUpdate, TimeDependentUpdate
from ..materials import effective_material_data
from ..simulation import Simulation
from ..solvers.roles import SolverRequest
from .expressions import (FIELD_COORDINATE_NAMES, THREE_D_COORDINATE_NAMES, TIME_NAMES, Expression, ExpressionError,
                          compile_expression, error_message, evaluate_parameters, evaluate_scalar)
from .build_masks import mesh_masks
from .geometry import (GEOMETRY_KINDS, cross_section_runs, domain_map, domain_object_rasterizer, expand_entities,
                       paint_material_index, point_z, resolve_geometry)
from .meshing import make_domain, mesh_material_types, optional_material_arrays, physics_settings
from .mesh_transfer import make_transfer, transfer_kind, transform_side
from .boundary_locator import geometry_locator
from .schema import (Issue, ResolvedSolver, conformal_boundaries_physics, effective_coupling_every, get_registry,
                     node_label, resolve_values, resolved_solvers, solver_backend, state_field_names,
                     subpixel_smoothing_physics)
from .session import Session
from .solver_hooks import RerunSolver, check_study_hooks
from .subpixel_geometry import interface_geometry
from .sweep_reuse import ParameterFollower, value_names
from .spec import OUTER_FACES_3D, RESERVED_ENTITY_NAMES

_DEFERRED_RANGE_CHECK_CALLS = 100   # host read of a time-stepped dynamic material update's running range
_OBJECT_ENTITY_KIND = {'domain': 'domain', 'boundary': 'domain', 'point': 'point', 'line': 'line'}
# memory estimate = the solver's big co-resident arrays, so it sits just below the measured bytes: above them it books
# what the run did not hold (the estimator misreads the solver), far below them memory is unaccounted (a missing item,
# or memory the code could free)
ESTIMATE_SMALL_ITEMS = 0.25         # measured may exceed the estimate by this: the small arrays it leaves out
ESTIMATE_FLOOR_BYTES = 1 << 20      # differences below this are not judged (pool block rounding)


class ModelError(Exception):
    """The model cannot be built; `issues` holds the errors (Issue with GUI node paths), str() summarises them."""

    def __init__(self, issues):
        self.issues = list(issues)
        lines = [f'{node_label(issue.path)}: {issue.message}' for issue in self.issues]
        count = len(self.issues)
        super().__init__(f"{count} model error{'' if count == 1 else 's'}: " + '; '.join(lines))


@dataclass
class MaterialRegion:
    """A material's cells after the winning materials took their overlap (geometry.material_paint_order);
    dispersion values evaluated to floats."""
    name: str
    properties: dict            # library model properties overridden by the spec's (raw numbers / expressions)
    dispersion: list            # [{'model': ..., <float params>}]; [] when no physics uses dispersion
    # cupy bool (nx, ny) effective mask; None once the build released its caches. 3D: None (no full-grid mask per
    # region) unless a state-dependent property needs one: use BuildContext.region_window_mask / region_mask
    mask: object


@dataclass
class BuiltSolver:
    """The solver runtime of one (physics, role) and what it was resolved from (study summary 'solvers')."""
    resolved: ResolvedSolver
    backend: str
    runtime: object


@dataclass
class ResolvedSelection:
    kind: str | None
    entities: list              # expanded, ordered
    outer_faces: list           # OUTER_FACES members (3D models: OUTER_FACES_3D)
    objects: list               # geometry object names
    is_all: bool


class _SharedBuild:
    """Model-wide part of a build that every per-mesh BuildContext shares (S7.2): log, warnings, the Simulation, solver
    records, feature time updates, conformal locators, the grid transfers between meshes and the contexts by mesh name
    (weak references: the first mesh's context holds the others, so a closed Session frees them without gc)."""

    def __init__(self, log):
        self.log_callback = log
        self.warnings = []
        self.sim = None
        self.solver_records = {}         # physics name -> {role: BuiltSolver} (solver stage, before physics create)
        self.time_updates = []           # TimeDependentUpdate of feature values of t (follow_time), attached after them
        self.live_parameters = frozenset()   # reused sweep build: the swept parameter and those depending on it
        self.parameter_followers = []    # sweep_reuse.ParameterFollower of follow_parameters (Session.set_sweep_value)
        self.conformal_locators = {}     # honouring physics names -> the build's GeometryLocator
        self.transfers = {}              # (source mesh, target mesh, kind, axisymmetric) -> GridTransfer
        self.contexts = weakref.WeakValueDictionary()   # mesh name -> BuildContext


class BuildContext:
    """Evaluated model + the Domain of one mesh + Simulation and the mask helpers definition callbacks use while
    building. Several meshes (S7.2): one context per mesh sharing the model-wide state (_SharedBuild);
    for_physics(name) is the context of that physics' mesh."""

    def __init__(self, spec, registry, parameters, resolved_geometry, domain, log, shared=None):
        self.spec = spec
        self.registry = registry
        self.parameters = parameters
        self.resolved_geometry = resolved_geometry
        self.domain = domain
        self.xp = domain.xp              # array module of the run backend (numpy or cupy)
        self._shared = shared if shared is not None else _SharedBuild(log)
        self.mesh_name = domain.mesh_name
        self._shared.contexts[self.mesh_name] = self
        self.other_meshes = {}           # first mesh's context: the other meshes' contexts (strong references)
        self.study_type = spec.study.type
        self.material_regions = []
        self._state_names = state_field_names(spec, registry)
        self.domain_map = domain_map(spec, parameters)   # geometry.DomainMap: Cartesian, polar or spherical window
        self.masks = mesh_masks(self)    # build_masks.MeshMasks of this mesh's grid kind (masks, coordinates, lookups)
        # 3D (Domain3D): int16 (nx, ny, nz) 1 + material region index (0 = none) painted per z-slab; build only
        self.material_index = None
        self._region_objects = {}        # 3D: region name -> (selection objects, is_all) for region_window_mask

    # ---- model-wide state (shared by the contexts of every mesh) ----
    @property
    def sim(self):
        return self._shared.sim

    @sim.setter
    def sim(self, value):
        self._shared.sim = value

    @property
    def warnings(self) -> list:
        return self._shared.warnings

    @property
    def solver_records(self) -> dict:
        return self._shared.solver_records

    @solver_records.setter
    def solver_records(self, value):
        self._shared.solver_records = value

    @property
    def time_updates(self) -> list:
        return self._shared.time_updates

    @property
    def parameter_followers(self) -> list:
        return self._shared.parameter_followers

    @property
    def conformal_locators(self) -> dict:
        """solver_defaults.conformal_boundaries (S7.6): honouring physics names -> the build's GeometryLocator (set
        before the creates, so a create can use the drawn outlines, e.g. the 2D FDTD conformal pole weights)."""
        return self._shared.conformal_locators

    @conformal_locators.setter
    def conformal_locators(self, value):
        self._shared.conformal_locators = value

    def release_shared(self):
        """Drop the model-wide build state that holds solvers or device arrays (solver records, grid transfers, time
        updates, parameter followers, conformal locators): Session.close, without relying on gc."""
        shared = self._shared
        shared.solver_records, shared.transfers, shared.conformal_locators = {}, {}, {}
        shared.time_updates.clear()
        shared.parameter_followers.clear()

    # ---- meshes (S7.2) ----
    def mesh_of_physics(self, physics_name) -> str:
        """Mesh name a physics runs on (this context's in a one-mesh build)."""
        if len(self._shared.contexts) <= 1:
            return self.mesh_name
        return self.spec.mesh_of(physics_name).name

    def context_of_mesh(self, mesh_name):
        """The BuildContext of a mesh of this build."""
        return self._shared.contexts[mesh_name]

    def for_physics(self, physics_name):
        """The BuildContext of the mesh physics_name runs on: its domain, masks and eval_field act on that mesh, the
        model-wide state is shared; this context itself in a one-mesh build."""
        return self.context_of_mesh(self.mesh_of_physics(physics_name))

    def transfer(self, kind, source_mesh, target_mesh, axisymmetric=False):
        """Cached GridTransfer ('interpolate' | 'conservative') between two meshes of this build
        (model.mesh_transfer.make_transfer; axisymmetric: r-weighted conservative measures)."""
        key = (source_mesh, target_mesh, kind, bool(axisymmetric) and kind == 'conservative')
        cached = self._shared.transfers.get(key)
        if cached is None:
            cached = self._shared.transfers[key] = make_transfer(
                kind, self.context_of_mesh(source_mesh).domain, self.context_of_mesh(target_mesh).domain, key[3])
        return cached

    def conformal_locator_of(self, physics_name):
        """The GeometryLocator when solver_defaults.conformal_boundaries reaches this physics, else None (the same one
        set_conformal_boundaries hands its solver right after create)."""
        return self.conformal_locators.get(physics_name)

    @property
    def dimensions(self) -> int:
        """2, or 3 for a Domain3D build (spec v2 size_z)."""
        return self.domain.dimensions

    def solvers(self, physics_name) -> dict:
        """{role: solver runtime} of the roles the physics uses in this study (schema.resolved_solvers), created by the
        build's solver stage; {} for a type without solver_roles."""
        return {role: built.runtime for role, built in self.solver_records.get(physics_name, {}).items()}

    def log(self, level, message, keep=True):
        """Forward to the build log; build warnings (keep) are also kept for BuildSummary.warnings, run-time ones
        (keep False, e.g. Session.derived_values) are not, so repeated calls never grow the list."""
        if level == 'warning' and keep:
            self.warnings.append(message)
        self._shared.log_callback(level, message)

    # ---- evaluation ----
    def eval_scalar(self, value) -> float:
        """Number or expression of parameters -> finite float."""
        return evaluate_scalar(value, self.parameters)

    def eval_field(self, value, mask=None, bounds_param=None):
        """Float, or a fresh cupy float32 array when the expression uses x, y (clamped cell centres): full-grid in 2D,
        (nx, ny, 1) in 3D (broadcast along z, no full-grid float temporary); 3D with z: a BoxValues on the mask's
        bounding box. ExpressionError on unknown names, non-finite values or values outside bounds_param's min / max
        (under mask; 3D x, y values: cells it covers in any z)."""
        if value is None:
            raise ValueError('eval_field: missing value (None)')
        if not isinstance(value, (str, Expression)):
            scalar = evaluate_scalar(value, self.parameters)
            _check_range(repr(value), bounds_param, scalar, scalar)
            return scalar
        expression = compile_expression(value)
        text = repr(expression.text)
        if expression.names.issubset(self.parameters):
            scalar = evaluate_scalar(expression, self.parameters)
            _check_range(text, bounds_param, scalar, scalar)
            return scalar
        namespace = dict(self.parameters)
        three_d = self.masks.paints_by_slabs   # 3D tensor grids (tetrahedra: per element, as triangles)
        xp = self.xp
        if three_d and not expression.names.isdisjoint(THREE_D_COORDINATE_NAMES):
            return self._eval_box_planes(expression, mask, bounds_param)
        if not expression.names.isdisjoint(FIELD_COORDINATE_NAMES):
            namespace.update(self.masks.field_coordinates())
        result = expression.evaluate(namespace, xp=xp)
        if not isinstance(result, xp.ndarray):
            scalar = evaluate_scalar(result, {})
            _check_range(text, bounds_param, scalar, scalar)
            return scalar
        grid_shape = self.domain.materials.shape
        grid = xp.empty(grid_shape[:2] + (1,) if three_d else grid_shape, dtype=DTYPE)
        grid[...] = result
        if three_d and mask is not None and mask.ndim == 3:
            mask = mask.any(axis=2, keepdims=True)
        lowest, highest = to_host(xp.stack(_masked_extremes(grid, mask)))   # the one host sync
        _check_range(text, bounds_param, float(lowest), float(highest), '' if mask is None else ' inside its region')
        return grid

    def _eval_box_planes(self, expression, mask, bounds_param):
        """3D BoxValues of a z-dependent expression on the bounding box of the full-grid mask (None: the grid), one z
        plane at a time; range-checked under mask (one host sync after the box's three)."""
        box = _mask_box(mask, self.domain.materials.shape)
        values = self.xp.empty(tuple(axis.stop - axis.start for axis in box), dtype=DTYPE)
        running_extremes = _new_running_extremes(self.xp)
        for offset, plane in _z_planes(self, expression, box):
            values[:, :, offset] = plane
            plane_mask = None if mask is None else mask[box[0], box[1], box[2].start + offset]
            _accumulate_extremes(running_extremes, plane, plane_mask)
        _check_running_range(repr(expression.text), bounds_param, running_extremes,
                             '' if mask is None else ' inside its region')
        return BoxValues(box, values)

    def follow_time(self, value, setter, solver, label, bounds_param=None):
        """setter(float value) now; a value of time t (time_dependent param: an Expression of parameters and t) is
        set at the runner time (0 at the build) and again before every step of `solver` (its runner stride) by a
        TimeDependentUpdate attached after the features; setter runs only when the value changed (setters are
        incremental, so a repeat is a no-op: ramps that level off, steps). ExpressionError for a value outside
        bounds_param's range."""
        if not (isinstance(value, Expression) and not value.names.isdisjoint(TIME_NAMES)):
            setter(self.eval_scalar(value) if isinstance(value, (str, Expression)) else value)
            return
        runner = self.sim.runner
        namespace = dict(self.parameters)
        text = repr(value.text)

        def value_at(time):
            namespace['t'] = time
            number = evaluate_scalar(value, namespace)   # finite or ExpressionError
            if bounds_param is not None:
                _check_range(f'{label}: {text} at t = {time:g} s', bounds_param, number, number)
            return number

        last_set = [None]   # the value setter last received

        def set_changed(number):
            if number != last_set[0]:
                last_set[0] = number
                setter(number)

        set_changed(value_at(float(runner.t)))
        self.time_updates.append(TimeDependentUpdate(label, solver.name, value_at, set_changed, runner,
                                                     every=runner.strides[solver.name]))

    def follow_parameters(self, raw_params, values, names, setter, solver, label) -> bool:
        """Feature values `names` (Param.follows_parameters) on a reused sweep build: when the raw spec value of one
        (raw_params: the feature spec's params; a missing name = its default) names a live sweep parameter and solver
        is a RerunSolver, registers a sweep_reuse.ParameterFollower whose setter(*values of names) Session.set_sweep_value
        calls when one changed (the others pass their resolved `values` object unchanged; incremental: the apply set
        the build values itself, keeping host data only, never build caches). True when registered."""
        live = self._shared.live_parameters
        if not live or not isinstance(solver, RerunSolver):
            return False
        raw = raw_params if isinstance(raw_params, dict) else {}
        followed = [raw[name] if name in raw and not value_names(raw[name]).isdisjoint(live) else None
                    for name in names]
        if all(entry is None for entry in followed):
            return False
        self.parameter_followers.append(ParameterFollower(label, followed, [values[name] for name in names], setter,
                                                          self.parameters))
        return True

    # ---- selections and masks ----
    def selection(self, entities, kind=None) -> ResolvedSelection:
        """Expand and check an entity list against a selection kind ('domain'|'boundary'|'point'|'line'|None)."""
        expanded = expand_entities(self.spec, entities)
        outer_faces = [name for name in expanded if name in OUTER_FACES_3D]   # z faces only pass validation in 3D
        objects = [name for name in expanded if name not in RESERVED_ENTITY_NAMES]
        is_all = 'all' in expanded
        for name in objects:
            self._shape(name)
        if kind is not None:
            if kind not in _OBJECT_ENTITY_KIND:
                raise ValueError(f'unknown selection kind {kind!r}; valid: {list(_OBJECT_ENTITY_KIND)}')
            if outer_faces and kind != 'boundary':
                raise ValueError(f'a {kind} selection cannot contain the outer faces {outer_faces}')
            if is_all and kind != 'domain':
                raise ValueError(f"a {kind} selection cannot contain 'all'")
            wanted = _OBJECT_ENTITY_KIND[kind]
            for name in objects:
                shape = self._shape(name)
                if GEOMETRY_KINDS[shape.kind].entity_kind != wanted:
                    raise ValueError(f'{kind} selection: {name!r} is a {shape.kind}, not a {wanted} object')
        return ResolvedSelection(kind, expanded, outer_faces, objects, is_all)

    def domain_mask(self, entities):
        """Fresh cupy bool full-grid union of domain entities ('all' = every cell, PML included); 3D: (nx, ny, nz)."""
        selection = self.selection(entities, 'domain')
        grid_shape = self.domain.materials.shape
        if selection.is_all:
            return self.xp.ones(grid_shape, dtype=np.bool_)
        mask = self.xp.zeros(grid_shape, dtype=np.bool_)
        for name in selection.objects:
            window, cropped = self.masks.object_mask(name)
            mask[window] |= cropped
        return mask

    def object_box_mask(self, name):
        """3D: cached ((slice x, slice y, slice z), cupy bool mask cropped to the object's bounding-box cells;
        read-only by convention) on Domain3D.host_centres (geometry.rasterize_box, one 2D raster per cross-section; a
        cylindrical sector: on the drawing-plane centres)."""
        return self.masks.object_box_mask(name)

    def region_mask(self, region):
        """3D: fresh cupy bool full-grid mask of a MaterialRegion from material_index (1 B/cell: prefer
        region_window_mask)."""
        return self.material_index == self._region_number(region)

    def region_window_mask(self, region):
        """3D: (window slices, fresh cupy bool mask cropped to it) of a MaterialRegion: the union of its objects'
        bounding boxes (the full grid for an 'all' material) intersected with material_index."""
        window = self.masks.region_window(*self._region_objects[region.name])   # a sector: the whole grid
        return window, self.material_index[window] == self._region_number(region)

    def _region_number(self, region):
        """1 + index of region in material_regions (identity: regions hold device arrays, never compare them)."""
        return next(index for index, other in enumerate(self.material_regions) if other is region) + 1

    def height(self, name):
        """3D: z [m] of a point or line object (geometry.point_z: its z param, else mid-height)."""
        shape = self._shape(name)
        if shape.kind not in ('point', 'line'):
            raise ValueError(f'{name!r} is a {shape.kind}, expected a point or line')
        return point_z(shape.values, self.domain.size_z)

    def object_window_mask(self, name):
        """Cached ((slice x, slice y), cupy bool mask cropped to the object's bounding-box cells; read-only by
        convention) of a domain object on the float32 clamped cell centres (geometry.rasterize_window); on a triangle /
        tetrahedron mesh ((slice of all elements,), element mask) by the float64 clamped centroids (the mesh's region
        tag rule; tetrahedra: tetrahedral_meshing.element_membership); a polar / spherical window on the drawing-plane
        centres of its cells. 3D tensor builds use object_box_mask."""
        return self.masks.object_window_mask(name)

    def coordinate_grids(self):
        """2D float32 full-grid (x, y) cell centres [m] in the drawing plane (field expressions, masks): the Domain's
        clamped zero-stride grids, or the cached mapped centres of a polar / spherical window (DomainMap)."""
        return self.masks.coordinate_grids()

    def plane_coordinates(self, window_x=slice(None), window_y=slice(None)):
        """3D: device float32 drawing-plane (x, y) [m] of the cells of a full-grid (window x, window y) box,
        broadcastable to (bx, by): the clamped 1D centres, or the cropped mapped centres of a cylindrical sector."""
        return self.masks.plane_coordinates(window_x, window_y)

    def boundary_edges(self, entities) -> np.ndarray:
        """Triangle mesh: sorted int64 mesh edges of a boundary / line selection (outer faces, domain object outlines,
        line entities; the mesh's edge tags); the tensor counterpart is perimeter_mask / perimeter_weights."""
        return self.masks.boundary_edges(self.selection(entities))

    def boundary_faces(self, entities) -> np.ndarray:
        """Tetrahedron mesh: sorted int64 mesh faces of a boundary selection: outer faces = the window faces on that
        side (fem.mesh3d.window_faces), domain objects = their interface faces (mesh.object_faces)."""
        return self.masks.boundary_faces(self.selection(entities))

    def point_position(self, name) -> tuple:
        """(x, y, z) [m] of a point object in a 3D Cartesian model (z: its z param, else mid-height)."""
        return self.point(name) + (self.height(name),)

    def point_node(self, name) -> int:
        """Triangle mesh: the mesh node of a point object (the mesher puts one there); ValueError for a point outside
        the window (it has no node)."""
        return self.masks.point_node(name)

    def perimeter_mask(self, objects):
        """2D tensor meshes: cached cupy bool full-grid mask of the union's perimeter cells, window cells only."""
        return self.masks.perimeter_mask(objects)

    def perimeter_weights(self, objects):
        """2D tensor meshes: cached cupy float32 full-grid perimeter length / cell area [1/m] of the union
        (geometry.perimeter_weights), window cells only."""
        return self.masks.perimeter_weights(objects)

    def point(self, name):
        """(x, y) of a point object in Domain coordinates (pos_to_cell): the drawn position, on a polar / spherical
        window its logical (r - radius_min, angle - angle_min)."""
        values = self._shape(name, 'point').values
        x, y = self.domain_map.to_logical(values['x'], values['y'])
        return float(x), float(y)

    def line(self, name):
        """(x0, y0, x1, y1) of a line object (end points as given)."""
        values = self._shape(name, 'line').values
        return float(values['x0']), float(values['y0']), float(values['x1']), float(values['y1'])

    def rect(self, name):
        """(x0, y0, x1, y1) of a rectangle object (sorted corners)."""
        values = self._shape(name, 'rectangle').values
        return float(values['x0']), float(values['y0']), float(values['x1']), float(values['y1'])

    def release_caches(self):
        """Drop the domain coordinate grids, the cached object / perimeter masks and the material-region masks
        (dynamic material updates keep their own reference to the masks they write)."""
        self.domain.release_coordinate_grids()
        self.masks.release()
        self.material_index = None
        for region in self.material_regions:
            region.mask = None

    # ---- internal ----
    def _shape(self, name, kind=None):
        shape = self.resolved_geometry.get(name)
        if shape is None:
            raise ValueError(f'unknown geometry object {name!r}; resolved: {list(self.resolved_geometry)}')
        if kind is not None and shape.kind != kind:
            raise ValueError(f'{name!r} is a {shape.kind}, expected a {kind}')
        return shape

    def _dynamic_evaluator(self, expression, state_references, mask, bounds_param, label, clock=None):
        """evaluate(physics_by_name) closure of a state- or time-dependent expression (x, y and in 3D z captured as
        zero-stride grids; t = clock.t, the runner time; state of a physics on another mesh interpolated onto this one
        into a persistent buffer, S7.2); see _check_dynamic_range for when its running range under mask is checked
        against bounds_param (host sync)."""
        names = expression.names
        readers = {}   # state name of a physics on another mesh -> [GridTransfer, its output buffer]
        for name, physics, _ in state_references:
            physics_mesh = self.mesh_of_physics(physics)
            if physics_mesh != self.mesh_name:
                readers[name] = [self.transfer('interpolate', physics_mesh, self.mesh_name), None]
        base = {name: self.parameters[name] for name in names if name in self.parameters}
        if not names.isdisjoint(FIELD_COORDINATE_NAMES):   # 2D (X, Y), 3D (X, Y, Z)
            base.update(self.masks.dynamic_coordinates())
        source_physics = tuple(sorted({physics for _, physics, _ in state_references}))
        xp = self.xp
        running_extremes = _new_running_extremes(xp)
        n_evaluations = [0]
        checked_solve_counts = [None]   # source n_solves at the last host check

        def evaluate(physics_by_name):
            namespace = dict(base)
            if clock is not None:
                namespace['t'] = float(clock.t)
            for name, physics, field_name in state_references:
                value = physics_by_name[physics].state()[field_name]
                reader = readers.get(name)
                if reader is not None:
                    reader[1] = value = reader[0].apply(value, out=reader[1])
                namespace[name] = value
            value = expression.evaluate(namespace, xp=xp)
            if not isinstance(value, xp.ndarray):   # t and parameters only: one number for every cell
                value = xp.asarray(value, dtype=DTYPE)
            _accumulate_extremes(running_extremes, value, mask)
            n_evaluations[0] += 1
            _check_dynamic_range(running_extremes, n_evaluations[0], checked_solve_counts, physics_by_name,
                                 source_physics, f'{label} = {expression.text!r}', bounds_param)
            return value

        return evaluate


def build(spec, registry=None, log=None, live_parameters=()) -> Session:
    """Validate and build spec into a Session (GPU); ModelError lists the problems at their node paths.
    live_parameters: a reused sweep build's swept parameter and its dependents (BuildContext.follow_parameters)."""
    from .meshing import element_meshes_kept
    with element_meshes_kept():   # an element mesh made for the memory judgement is the one the domain stage uses
        return _build(spec, registry, log, live_parameters)


def _build(spec, registry, log, live_parameters):
    """build() inside its element mesh scope."""
    from .meshing import has_estimated_counts
    from .validate import validate
    start = time.perf_counter()
    registry = registry if registry is not None else get_registry()
    log = log if log is not None else _print_log

    budget = gpu_memory.budget_bytes()
    if budget is not None:   # arrays of other live sessions (an earlier api Results) already hold part of the pool
        budget = max(budget - gpu_memory.pool_used_bytes(), 0)
    host_budget = host_memory.available_bytes()
    # refuse a model that does not fit; the available RAM already excludes what live sessions hold on the host.
    # One mesh / memory estimate serves validate and the ledger check (None: validate estimates and reports).
    mesh_estimate, memory_estimate, cached_shapes = _cached_estimates(spec, registry)
    meshing_seconds = 0.0
    if (memory_estimate is not None and mesh_estimate
            and all(estimate is not None for estimate in mesh_estimate.values())
            and any(has_estimated_counts(estimate) for estimate in mesh_estimate.values())):
        # size-based element counts (a size-field integral: too high or too low, e.g. 0.36x on a graded 5 mm mesh):
        # every other error first (no mesher run for a model that fails anyway), then the meshes are built (kept for
        # the domain stage, so no second mesher run) and the memory judgement and ledger check are on their counts
        _raise_errors(validate(spec, registry, mesh_estimate=mesh_estimate, memory_estimate=memory_estimate))
        meshing_start = time.perf_counter()
        mesh_estimate, memory_estimate = _built_estimates(spec, registry, mesh_estimate, cached_shapes, log)
        meshing_seconds = time.perf_counter() - meshing_start
    issues = validate(spec, registry, mesh_estimate=mesh_estimate, memory_budget=budget,
                      memory_estimate=memory_estimate, host_budget=host_budget)
    _raise_errors(issues)
    for issue in issues:
        log(issue.level, f'{node_label(issue.path)}: {issue.message}')
    validated = time.perf_counter()

    with _issue_at(('parameters',)):
        parameters = evaluate_parameters(spec.parameters)
    geometry_errors = {}
    shapes = (cached_shapes if cached_shapes is not None   # the same parameters, resolved without errors
              else resolve_geometry(spec, parameters, errors=geometry_errors))
    if geometry_errors:
        raise ModelError([Issue('error', ('geometry', name), message) for name, message in geometry_errors.items()])

    estimate = _memory_estimate(spec, parameters, registry, log, memory_estimate)
    from .meshing import kept_element_mesh_keys
    kept_mesh_keys = kept_element_mesh_keys()   # before meshing: the build estimate skips a mesher run reusing one
    ledger = gpu_memory.MemoryLedger(backend=solver_backend(spec))   # cpu: traced numpy bytes (tracemalloc)
    ledger.add_time('validation + mesh / memory estimates', validated - start - meshing_seconds)
    if meshing_seconds:
        ledger.add_time('element meshes for the memory check', meshing_seconds)
    ledger.add_time('parameters, geometry + memory record', time.perf_counter() - validated)
    try:
        session = _build_on_gpu(spec, registry, parameters, shapes, log, ledger, start, live_parameters)
        ledger.finish()
    except Exception as exc:   # the ledger so far travels with the exception (CPU data only, no frames)
        exc.build_memory = _memory_record(ledger, estimate, compare=False)
        raise
    finally:
        ledger.stop()
    session.build_memory = _memory_record(ledger, estimate, compare=True)
    log('info', gpu_memory.ledger_time_text(session.build_memory))
    _log_build_estimate(spec, parameters, registry, mesh_estimate, session.build_memory, kept_mesh_keys, log, shapes)
    check_message = _estimate_check_message(session.build_memory)
    if check_message:
        log('warning', check_message)
        session.warnings.append(check_message)
    session.time_estimate = _time_estimate(spec, parameters, registry, mesh_estimate, log, shapes)
    return session


def _time_estimate(spec, parameters, registry, mesh_estimate, log, shapes=None):
    """model.time_estimate of one study run (CPU, priced on the stored machine profile), logged; None without a
    profile; a failing estimate is logged as a warning (a recipe bug), never stops the build."""
    from .time_estimate import estimate_time
    try:
        estimate, message = estimate_time(spec, parameters, registry, mesh_estimate, per_run=True, shapes=shapes)
    except Exception as exc:
        log('warning', f'run-time estimate failed (no time check for this run): {_exception_message(exc)}')
        return None
    log('info', estimate.text(count=3) if estimate is not None else message)
    return estimate


def _log_build_estimate(spec, parameters, registry, mesh_estimate, build_memory, kept_mesh_keys, log, shapes=None):
    """Log the build-time estimate of this run (time_estimate.estimate_overheads per_run, the mesher left out for a
    kept mesh) against the ledger's stage seconds; nothing without a machine profile, a failure as a warning."""
    from .time_estimate import build_time_text, estimate_overheads
    try:
        build_part = estimate_overheads(spec, parameters, registry, mesh_estimate, per_run=True,
                                        kept_mesh_keys=kept_mesh_keys, shapes=shapes).get('build')
        if build_part is None:
            return
        names = [mesh_spec.name for mesh_spec in spec.used_meshes()]
        owners = {name: _mesh_owner('domain', name, len(names) > 1) for name in names}
        log('info', build_time_text(build_part, build_memory.get('seconds') or (), owners))
    except Exception as exc:   # an estimator bug never stops the build
        log('warning', f'build-time estimate failed: {_exception_message(exc)}')


def _build_on_gpu(spec, registry, parameters, shapes, log, ledger, start, live_parameters=()):
    """GPU part of build (per mesh: domain, materials, subpixel geometry; then solvers, physics, features, updates,
    couplings), each stage tracked by ledger; per-mesh stages are labelled '<stage> <mesh>' when there are several."""
    mesh_names = [mesh_spec.name for mesh_spec in spec.used_meshes()]
    several = len(mesh_names) > 1
    backend = solver_backend(spec)
    shared = _SharedBuild(log)
    shared.live_parameters = frozenset(live_parameters)
    contexts = {}   # mesh name -> BuildContext (spec order of the used meshes)
    for mesh_name in mesh_names:
        with ledger.track(_mesh_owner('domain', mesh_name, several)), _issue_at(('mesh', mesh_name)):
            domain = make_domain(spec, parameters, registry, resolved_geometry=shapes, mesh=mesh_name, log=log)
            domain.use_backend(backend)   # still array-free: every later array lands on that backend
        domain.mesh_name = mesh_name
        contexts[mesh_name] = BuildContext(spec, registry, parameters, shapes, domain, log, shared=shared)
    ctx = contexts[mesh_names[0]]   # the Session's context: it holds the others
    ctx.other_meshes = {name: context for name, context in contexts.items() if context is not ctx}

    present_types = {}   # physics type name -> PhysicsType, in spec order
    for physics_spec in spec.physics:
        if physics_spec.type not in present_types:
            with _issue_at(('physics', physics_spec.name)):
                present_types[physics_spec.type] = registry.physics_type(physics_spec.type)
    deferred = {}   # mesh name -> state-dependent material properties of _apply_static_materials
    for mesh_name, mesh_ctx in contexts.items():
        mesh_types = mesh_material_types(spec, registry, mesh_name)   # its physics' types (+ coupling partners)
        for physics_type in mesh_types:
            first = next(physics_spec for physics_spec in spec.physics if physics_spec.type == physics_type.name)
            with ledger.track(_mesh_owner(f"material arrays '{physics_type.name}'", mesh_name, several)), \
                    _issue_at(('physics', first.name)):
                mesh_ctx.domain.materials.add_properties(physics_type.material_schema)
                mesh_ctx.domain.materials.add_properties(optional_material_arrays(spec, (physics_type,)))
        with ledger.track(_mesh_owner('materials', mesh_name, several)):
            deferred[mesh_name] = _apply_static_materials(mesh_ctx, mesh_types)
    # igor 1.5: interface geometry once per mesh, owned by the honouring solvers only (each averages its current eps)
    smoothed_physics, _ = subpixel_smoothing_physics(spec, registry)
    smoothing_geometries = {}   # mesh name -> its interface geometry
    for mesh_name in dict.fromkeys(ctx.mesh_of_physics(name) for name in smoothed_physics):
        with ledger.track(_mesh_owner('subpixel geometry', mesh_name, several)), _issue_at(('solvers',)):
            smoothing_geometries[mesh_name] = interface_geometry(contexts[mesh_name])
    # solver_plan S7.6: one host geometry locator (mesh-independent), handed to the honouring solvers before their
    # features apply
    conformal_physics, _ = conformal_boundaries_physics(spec, registry)
    locator = geometry_locator(ctx) if conformal_physics else None
    ctx.conformal_locators = {name: locator for name in conformal_physics}

    for physics_spec in spec.physics:   # own stage: a solver is never created inside a physics create (ledger nesting)
        physics_type = present_types[physics_spec.type]
        with _issue_at(('physics', physics_spec.name)):
            resolved = resolved_solvers(spec, registry, physics_spec, ctx.study_type, parameters)
        for role, resolved_solver in resolved.items():
            path = ('solver', resolved_solver.node) if resolved_solver.node else ('physics', physics_spec.name)
            with ledger.track(f"solver '{physics_spec.name}.{role}'"), _issue_at(path):
                runtime = _create_solver(ctx.for_physics(physics_spec.name), physics_spec.name, physics_type, role,
                                         resolved_solver, backend)
            ctx.solver_records.setdefault(physics_spec.name, {})[role] = BuiltSolver(resolved_solver, backend, runtime)

    sim = Simulation.multiphysics([context.domain for context in contexts.values()])
    ctx.sim = sim
    features = {}
    for physics_spec in spec.physics:
        physics_type = present_types[physics_spec.type]
        physics_ctx = ctx.for_physics(physics_spec.name)
        with ledger.track(f"physics '{physics_spec.name}'"), _issue_at(('physics', physics_spec.name)):
            if physics_type.create is None:
                raise ValueError(f"physics type '{physics_type.name}' has no create callback")
            settings = physics_settings(physics_type, physics_spec, parameters)
            solver = physics_type.create(physics_ctx, physics_spec, settings)
            if ctx.study_type in physics_type.supported_studies:
                check_study_hooks(physics_spec.name, solver, registry.study_type(ctx.study_type))
            if physics_spec.name in smoothed_physics:
                solver.set_subpixel_smoothing(smoothing_geometries[physics_ctx.mesh_name])
            if physics_spec.name in conformal_physics:
                solver.set_conformal_boundaries(locator)
            sim.add_physics(physics_spec.name, solver, every=int(physics_spec.every), domain=physics_ctx.domain)
        for feature_spec in physics_spec.features:
            if feature_spec.enabled:
                feature_owner = f"feature '{physics_spec.name}/{feature_spec.name}'"
                with ledger.track(feature_owner), _issue_at(('feature', physics_spec.name, feature_spec.name)):
                    features[(physics_spec.name, feature_spec.name)] = _apply_feature(
                        physics_ctx, physics_type, solver, feature_spec)

    updates = []
    for mesh_name, entries in deferred.items():
        for region, property_param, expression, grid_properties in entries:
            update_owner = _mesh_owner(f"material update '{region.name}.{property_param.name}'", mesh_name, several)
            with ledger.track(update_owner), _issue_at(('material', region.name)):
                update = _dynamic_update(contexts[mesh_name], region, property_param, expression, grid_properties)
                update.apply_now(sim.runner.physics)
                updates.append(update)
    for name in sim.physics:
        with ledger.track(f"physics '{name}' finalize"), _issue_at(('physics', name)):
            finalize_messages = sim.finalize([name])[name]   # marked finalized: a later sim.run() skips it
        for message in finalize_messages:
            ctx.log('warning', f"physics '{name}': {message}")
        for note in getattr(sim.runner.physics[name], 'finalize_notes', ()):   # optional: e.g. assembly timings
            ctx.log('info', f"physics '{name}': {note}")
    for update in updates:
        sim.add_coupling(update)
    for update in ctx.time_updates:   # feature values of t: re-set before the steps of their physics
        sim.add_coupling(update)

    for coupling_spec in spec.couplings:
        if coupling_spec.enabled:
            with ledger.track(f"coupling '{coupling_spec.name}'"), _issue_at(('coupling', coupling_spec.name)):
                sim.add_coupling(_create_coupling(ctx, coupling_spec))
    with ledger.track('runner prepare'), _issue_at(('model',)):
        sim.runner.prepare()

    with ledger.track('build temporaries freed'):
        for context in contexts.values():
            context.release_caches()
    return Session(spec, registry, parameters, ctx.domain, sim, ctx, features,
                   time.perf_counter() - start, warnings=ctx.warnings)


def _mesh_owner(owner, mesh_name, several):
    """Ledger owner of a per-mesh build stage: '<owner> <mesh>' when the model has several meshes (S7.2)."""
    return f'{owner} {mesh_name}' if several else owner


# ---- build steps ----
def _create_solver(ctx, physics_name, physics_type, role, resolved, backend):
    """Runtime object of one (physics, role) by its SolverType's create callback."""
    solver_type = resolved.solver_type
    if solver_type.create is None:
        raise ValueError(f"solver type '{solver_type.name}' has no create callback")
    return solver_type.create(ctx, SolverRequest(physics_name, role, physics_type.solver_roles[role], resolved.values,
                                                 backend))


def _apply_static_materials(ctx, present_types):
    """Write constant / field-dependent material properties under effective masks into ctx.material_regions;
    returns the state-dependent ones as [(region, property Param, Expression, grid properties)]."""
    spec = ctx.spec
    property_params = merged_property_params(present_types)
    uses_dispersion = any(physics_type.uses_dispersion for physics_type in present_types)

    expanded_selections, material_entries = [], []
    for material in spec.materials:
        with _issue_at(('material', material.name)):
            properties, dispersion = effective_material_data(material.library, material.properties,
                                                             material.dispersion)
            if uses_dispersion:
                dispersion = [_evaluate_pole(ctx, pole) for pole in dispersion]
            else:
                dispersion = []
            selection = ctx.selection(material.selection, 'domain')
            if not ctx.masks.paints_by_slabs:
                for name in selection.objects:   # rasterise here so a failure is reported at this material
                    ctx.object_window_mask(name)
            expanded_selections.append(selection.entities)
            material_entries.append((material.name, properties, dispersion))
    if ctx.masks.paints_by_slabs:
        return _apply_static_materials_3d(ctx, property_params, expanded_selections, material_entries)

    index_image = paint_material_index(expanded_selections, ctx.object_window_mask, ctx.domain.materials.shape,
                                       ctx.xp)   # (nx, ny); a triangle / tetrahedron mesh: (n_elements,)
    regions = [MaterialRegion(name, properties, dispersion, index_image == index + 1)
               for index, (name, properties, dispersion) in enumerate(material_entries)]
    del index_image
    ctx.material_regions = regions

    materials = ctx.domain.materials
    state_names = ctx._state_names
    deferred = []
    for region in regions:
        with _issue_at(('material', region.name)):
            for name, param, owned_arrays in owned_property_writes(region.properties, property_params,
                                                                    materials.property_names()):
                raw_value = region.properties[name]
                if isinstance(raw_value, (str, Expression)):
                    expression = _named_evaluation(name, lambda: compile_expression(raw_value))
                    if _is_dynamic(expression, state_names):
                        deferred.append((region, param, expression, owned_arrays))
                        continue
                value = _named_evaluation(name, lambda: ctx.eval_field(raw_value, mask=region.mask,
                                                                       bounds_param=param))
                for array in owned_arrays:
                    materials.assign(array, region.mask, value)
    return deferred


def _is_dynamic(expression, state_names):
    """True for a material expression of physics state fields or time t (a DynamicMaterialUpdate writes it)."""
    return not (expression.names.isdisjoint(state_names) and expression.names.isdisjoint(TIME_NAMES))


def merged_property_params(physics_types) -> dict:
    """property name -> its first Param over physics_types (material_properties order) with expands_to the ordered
    union over them: eps_r -> (eps_r, eps_xx, eps_yy) with E_z and H_z present (also Simulation.add_material)."""
    property_params = {}
    for physics_type in physics_types:
        for param in physics_type.material_properties:
            first = property_params.get(param.name)
            property_params[param.name] = param if first is None else replace(
                first, expands_to=tuple(dict.fromkeys(first.expands_to + param.expands_to)))
    return property_params


def owned_property_writes(properties, property_params, allocated):
    """(property name, Param, owned arrays) of the set properties (a {name: value} dict): isotropic before component,
    so an array a later entry writes is owned by that entry; only arrays in `allocated` (an unallocated
    optional_material_schema array is skipped; entries owning no array are skipped)."""
    entries = [(name, param) for name, param in property_params.items() if name in properties]
    for position, (name, param) in enumerate(entries):
        later_arrays = {array for _, later in entries[position + 1:] for array in later.expands_to}
        owned_arrays = tuple(array for array in param.expands_to if array not in later_arrays and array in allocated)
        if owned_arrays:
            yield name, param, owned_arrays


def _apply_static_materials_3d(ctx, property_params, expanded_selections, material_entries):
    """3D part of _apply_static_materials: the int16 ctx.material_index painted per z-slab run (one 2D raster per
    distinct cross-section), then per region its properties under a bool mask on the region's window; x, y
    dependent values are 2D (wx, wy) arrays broadcast along z, z-dependent ones are written per z plane: no
    full-grid float temporaries."""
    domain = ctx.domain
    shapes = ctx.resolved_geometry
    x_centres, y_centres, z_centres = domain.host_centres()
    painted_objects = sorted({name for entities in expanded_selections for name in entities
                              if name in shapes and GEOMETRY_KINDS[shapes[name].kind].entity_kind == 'domain'})
    xp = ctx.xp
    index_volume = xp.zeros(domain.materials.shape, dtype=np.int16)
    reused_masks = {}   # name -> (cross_section_key, mask): an object unchanged since its last run is not re-rasterised
    mapped = ctx.coordinate_grids() if ctx.domain_map.mapped else None   # a cylindrical sector's drawing plane
    with _issue_at(('geometry',)):
        for k_start, k_stop in cross_section_runs(painted_objects, shapes, z_centres):
            object_mask = domain_object_rasterizer(shapes, x_centres, y_centres, xp, z=z_centres[k_start],
                                                   reuse=reused_masks, mapped=mapped)
            slab_image = paint_material_index(expanded_selections, object_mask, domain.materials.shape[:2], xp)
            index_volume[:, :, k_start:k_stop] = slab_image[:, :, None]
            del slab_image
    reused_masks.clear()
    ctx.material_index = index_volume
    regions = [MaterialRegion(name, properties, dispersion, None)
               for name, properties, dispersion in material_entries]
    ctx.material_regions = regions
    for region, entities in zip(regions, expanded_selections):
        ctx._region_objects[region.name] = ([name for name in entities if name not in RESERVED_ENTITY_NAMES],
                                            'all' in entities)

    materials = domain.materials
    state_names = ctx._state_names
    deferred = []
    for region in regions:
        with _issue_at(('material', region.name)):
            window, region_mask = ctx.region_window_mask(region)
            for name, param, owned_arrays in owned_property_writes(region.properties, property_params,
                                                                    materials.property_names()):
                raw_value = region.properties[name]
                expression = (_named_evaluation(name, lambda: compile_expression(raw_value))
                              if isinstance(raw_value, (str, Expression)) else None)
                if expression is not None and _is_dynamic(expression, state_names):
                    if region.mask is None:   # the dynamic update writes full-grid values under a full-grid mask
                        region.mask = ctx.region_mask(region)
                    deferred.append((region, param, expression, owned_arrays))
                    continue
                if expression is not None and not expression.names.isdisjoint(THREE_D_COORDINATE_NAMES):
                    _named_evaluation(name, lambda: _write_z_planes_3d(ctx, expression, window, region_mask, param,
                                                                       owned_arrays))
                    continue
                if expression is not None and not expression.names.issubset(ctx.parameters):
                    value = _named_evaluation(name, lambda: _xy_values_3d(ctx, expression, window, region_mask,
                                                                          param))
                else:
                    value = _named_evaluation(name, lambda: ctx.eval_field(raw_value, bounds_param=param))
                for array in owned_arrays:
                    materials.assign_window(array, window, region_mask, value)
            del region_mask
    return deferred


def _xy_values_3d(ctx, expression, window, region_mask, bounds_param):
    """float32 (wx, wy, 1) values of an x, y dependent property on the 3D window's clamped centres (broadcast along
    z); range-checked under the cells region_mask covers in any z (one host sync)."""
    xp = ctx.xp
    namespace = dict(ctx.parameters)
    namespace['x'], namespace['y'] = ctx.plane_coordinates(window[0], window[1])
    result = expression.evaluate(namespace, xp=xp)
    values = xp.empty(region_mask.shape[:2], dtype=DTYPE)
    values[...] = result
    lowest, highest = to_host(xp.stack(_masked_extremes(values, region_mask.any(axis=2))))
    _check_range(repr(expression.text), bounds_param, float(lowest), float(highest), ' inside its region')
    return values[:, :, None]


def _write_z_planes_3d(ctx, expression, window, region_mask, bounds_param, arrays):
    """Write a z-dependent property into `arrays` one z plane of the region's window at a time ((wx, wy) values on
    the clamped centres: no window-sized float temporary); range-checked under region_mask (one host sync)."""
    materials = ctx.domain.materials
    running_extremes = _new_running_extremes(ctx.xp)
    for offset, plane in _z_planes(ctx, expression, window):
        k = window[2].start + offset
        plane_mask = region_mask[:, :, offset:offset + 1]
        _accumulate_extremes(running_extremes, plane, plane_mask[:, :, 0])
        for array in arrays:
            materials.assign_window(array, (window[0], window[1], slice(k, k + 1)), plane_mask, plane[:, :, None])
    _check_running_range(repr(expression.text), bounds_param, running_extremes, ' inside its region')


def _z_planes(ctx, expression, box):
    """Yield (z offset in box, float32 (bx, by) values) of expression per z cell of a full-grid box of the Domain3D, on
    the clamped (a sector: mapped) centres of every build mask; one buffer reused per plane: copy before keeping it."""
    z_centres = ctx.domain.host_centres()[2]
    xp = ctx.xp
    namespace = dict(ctx.parameters)
    namespace['x'], namespace['y'] = ctx.plane_coordinates(box[0], box[1])
    plane = xp.empty((box[0].stop - box[0].start, box[1].stop - box[1].start), dtype=DTYPE)
    for offset, z in enumerate(z_centres[box[2]]):
        namespace['z'] = float(z)
        plane[...] = expression.evaluate(namespace, xp=xp)
        yield offset, plane


def _mask_box(mask, grid_shape):
    """Full-grid slices of the bounding box of a bool backend mask (None: the whole grid; empty slices when no cell is
    set); one host transfer per axis."""
    if mask is None:
        return tuple(slice(0, count) for count in grid_shape)
    box = []
    for axis in range(mask.ndim):
        other_axes = tuple(other for other in range(mask.ndim) if other != axis)
        occupied = np.flatnonzero(to_host(mask.any(axis=other_axes)))
        box.append(slice(int(occupied[0]), int(occupied[-1]) + 1) if occupied.size else slice(0, 0))
    return tuple(box)


def _apply_feature(ctx, physics_type, solver, feature_spec):
    feature_type = physics_type.feature_type(feature_spec.type)
    if not isinstance(feature_spec.params, dict):
        raise ValueError(f'params must be a dict, got {type(feature_spec.params).__name__}')
    values = resolve_values(feature_type.params, feature_spec.params, ctx.parameters)
    selection = (ctx.selection(feature_spec.selection, feature_type.selection_kind)
                 if feature_type.selection_kind is not None else None)
    if feature_type.apply is None:
        return None
    return feature_type.apply(ctx, solver, feature_spec, values, selection)


def _dynamic_update(ctx, region, property_param, expression, grid_properties):
    """DynamicMaterialUpdate of a state- or time-dependent property on ctx's mesh (every = largest stride of the
    physics it reads; of t only: the smallest stride, rewritten before those steps; several meshes: only the physics
    on this mesh are told materials_changed)."""
    state_names = ctx._state_names
    references = [(name, *state_names[name]) for name in sorted(expression.names) if name in state_names]
    runner = ctx.sim.runner
    strides = runner.strides
    every = max((strides[physics] for _, physics, _ in references), default=min(strides.values()))
    source_fields = frozenset((physics, field_name) for _, physics, field_name in references)
    name = f'{region.name}.{property_param.name}'
    time_dependent = not expression.names.isdisjoint(TIME_NAMES)
    evaluate = ctx._dynamic_evaluator(expression, references, region.mask, property_param,
                                      f"material property '{name}'", clock=runner if time_dependent else None)
    notify = None
    if len(ctx.spec.used_meshes()) > 1:
        notify = [physics_spec.name for physics_spec in ctx.spec.physics_on(ctx.mesh_name)]
    return DynamicMaterialUpdate(ctx.domain.materials, grid_properties, region.mask, evaluate, every=every,
                                 name=name, source_fields=source_fields, time_dependent=time_dependent, notify=notify)


def _masked_extremes(values, mask):
    """Device 0-d (lowest, highest) of values under the bool mask (all cells when None); NaN propagates."""
    if mask is None:
        return values.min(), values.max()
    xp = array_module_of(values)
    return xp.where(mask, values, math.inf).min(), xp.where(mask, values, -math.inf).max()


def _new_running_extremes(xp):
    """Device (lowest, -highest) accumulator of _accumulate_extremes (inf, inf = no cell yet); xp numpy or cupy."""
    return xp.array([math.inf, math.inf])


def _accumulate_extremes(running_extremes, values, mask):
    """Fold the masked extremes of values into the device (lowest, -highest) in place (no host sync)."""
    lowest, highest = _masked_extremes(values, mask)
    xp = array_module_of(running_extremes)
    xp.minimum(running_extremes, xp.stack((lowest, -highest)), out=running_extremes)


def _check_running_range(text, bounds_param, running_extremes, where):
    """_check_range of a device (lowest, -highest) accumulator (one host sync)."""
    lowest, negative_highest = to_host(running_extremes)
    _check_range(text, bounds_param, float(lowest), -float(negative_highest), where)


def _check_range(text, bounds_param, lowest, highest, where=''):
    """ExpressionError unless [lowest, highest] is finite and inside bounds_param's min / max (None: finiteness
    only); lowest > highest means no cells."""
    if lowest > highest:
        return
    requirement = ''
    if not (math.isfinite(lowest) and math.isfinite(highest)):
        found = 'non-finite values'
    elif bounds_param is None:
        return
    elif bounds_param.min is not None and not (lowest > bounds_param.min if bounds_param.min_exclusive
                                               else lowest >= bounds_param.min):
        found = f'{lowest:g}' if lowest == highest else f'values down to {lowest:g}'
        requirement = f" (must be {'>' if bounds_param.min_exclusive else '>='} {bounds_param.min:g})"
    elif bounds_param.max is not None and not highest <= bounds_param.max:
        found = f'{highest:g}' if lowest == highest else f'values up to {highest:g}'
        requirement = f' (must be <= {bounds_param.max:g})'
    else:
        return
    raise ExpressionError(f'{text} gives {found}{where}{requirement}')


def _check_dynamic_range(running_extremes, n_evaluations, checked_solve_counts, physics_by_name, source_physics,
                         text, bounds_param):
    """Host check of a dynamic update's running (lowest, -highest): when every source physics is a non-transient solver
    (n_solves) and one re-solved since the last check, else at power-of-two evaluations (short runs) and every
    _DEFERRED_RANGE_CHECK_CALLS."""
    solve_counts = tuple(None if getattr(physics_by_name[physics], 'time_mode', None) == 'transient'
                         else getattr(physics_by_name[physics], 'n_solves', None) for physics in source_physics)
    solved = None not in solve_counts and solve_counts != checked_solve_counts[0]
    power_of_two = (n_evaluations & (n_evaluations - 1)) == 0   # 1, 2, 4, ...: few updates still get checked
    if power_of_two or solved or n_evaluations % _DEFERRED_RANGE_CHECK_CALLS == 0:
        checked_solve_counts[0] = solve_counts
        _check_running_range(text, bounds_param, running_extremes, ' inside its region')


def _create_coupling(ctx, coupling_spec):
    coupling_type = ctx.registry.coupling_type(coupling_spec.type)
    if coupling_type.create is None:
        raise ValueError(f"coupling type '{coupling_type.name}' has no create callback")
    if not isinstance(coupling_spec.params, dict):
        raise ValueError(f'params must be a dict, got {type(coupling_spec.params).__name__}')
    values = resolve_values(coupling_type.params, coupling_spec.params, ctx.parameters)
    strides = ctx.sim.runner.strides
    for role, physics in (('source', coupling_spec.source), ('target', coupling_spec.target)):
        if physics not in strides:
            raise ValueError(f"{role} physics '{physics}' does not exist; physics: {list(strides)}")
    every = effective_coupling_every(coupling_spec, ctx.spec.physics_by_name(coupling_spec.source),
                                     ctx.spec.physics_by_name(coupling_spec.target))
    if every is None:
        raise ValueError(f'every must be an integer >= 1, got {coupling_spec.every!r}')
    source_mesh, target_mesh = (ctx.mesh_of_physics(coupling_spec.source), ctx.mesh_of_physics(coupling_spec.target))
    if source_mesh == target_mesh:
        return coupling_type.create(ctx.context_of_mesh(source_mesh), coupling_spec, values, every)
    # S7.2: the create runs on the mesh of the transform ('field': target, 'value': source), its terms get the transfer
    side = transform_side(coupling_type)
    coupling = coupling_type.create(ctx.context_of_mesh(target_mesh if side == 'target' else source_mesh),
                                    coupling_spec, values, every)
    axisymmetric = bool(getattr(ctx.sim.physics[coupling_spec.target], 'axisymmetric', False))
    transfer = ctx.transfer(transfer_kind(coupling_type), source_mesh, target_mesh, axisymmetric)
    shared_views = {}   # one TransferredPhysics per physics for every term: the source interpolated once per revision
    for term in getattr(coupling, 'terms', (coupling,)):
        if not hasattr(term, 'set_mesh_transfers') or (term.source, term.target) != (coupling_spec.source,
                                                                                      coupling_spec.target):
            raise ValueError(f"coupling type '{coupling_type.name}' builds a part that is not a CouplingTerm from "
                             f"'{coupling_spec.source}' to '{coupling_spec.target}', so it cannot cross meshes; put "
                             'both physics on one mesh')
        if side == 'target':
            term.set_mesh_transfers(source_transfer=transfer, views=shared_views)
        else:
            term.set_mesh_transfers(value_transfer=transfer, views=shared_views)
    return coupling


def _evaluate_pole(ctx, pole):
    """Dispersion dict with every value except 'model' evaluated to a float."""
    if not isinstance(pole, dict) or not isinstance(pole.get('model'), str):
        raise ValueError(f"dispersion entries must be dicts with a 'model' name, got {pole!r}")
    return {key: value if key == 'model' else _named_evaluation(f"{pole['model']} {key}",
                                                                 lambda value=value: ctx.eval_scalar(value))
            for key, value in pole.items()}


def _named_evaluation(label, evaluate):
    """evaluate() with ExpressionError messages prefixed by label."""
    try:
        return evaluate()
    except ExpressionError as exc:
        raise ExpressionError(f'{label}: {exc}') from exc


# ---- GPU memory: estimate and ledger ----
def _cached_estimates(spec, registry):
    """({used mesh name: estimate}, MemoryEstimate, resolved geometry) of the spec for validate, the ledger check and
    the build (geometry resolved once), None where one fails (validate then estimates again and reports the failure at
    its node; the geometry is None unless it resolved without errors)."""
    from . import meshing
    try:
        parameters = evaluate_parameters(spec.parameters)
        geometry_errors = {}
        shapes = resolve_geometry(spec, parameters, errors=geometry_errors)
        shapes = None if geometry_errors else shapes
        mesh_estimate = meshing.estimate_meshes(spec, parameters, registry, shapes)
    except Exception:
        return None, None, None
    try:
        return mesh_estimate, meshing.estimate_memory(spec, parameters, registry, mesh_estimate, shapes=shapes), shapes
    except Exception:
        return mesh_estimate, None, shapes


def _raise_errors(issues):
    """ModelError of the error-level issues, if any."""
    errors = [issue for issue in issues if issue.level == 'error']
    if errors:
        raise ModelError(errors)


def _built_estimates(spec, registry, mesh_estimate, shapes, log):
    """({mesh name: estimate with its element mesh built and counted}, their MemoryEstimate or None when it fails:
    validate reports why): the numbers the memory judgement and the build ledger check are on; the meshes stay kept
    for the domain stage."""
    from . import meshing
    with _issue_at(('parameters',)):
        parameters = evaluate_parameters(spec.parameters)
    with _issue_at(('geometry',)):
        shapes = resolve_geometry(spec, parameters) if shapes is None else shapes
    built = {}
    for name, estimate in mesh_estimate.items():
        with _issue_at(('mesh', name)):
            built.update(meshing.built_mesh_estimates(spec, parameters, registry, {name: estimate}, shapes, log))
        if built[name] is not estimate:
            log('info', f"mesh '{name}': {built[name].n_elements:,} {estimate.kind} built before the memory check (the "
                        f'sizes gave an estimate of {estimate.n_elements:,})')
    try:
        return built, meshing.estimate_memory(spec, parameters, registry, built, shapes=shapes)
    except Exception:
        return built, None


def _memory_estimate(spec, parameters, registry, log, estimate=None):
    """{gpu_steady, gpu_build_peak, gpu_run, gpu_external, external_extra, external_peak, host, items / run_items /
    external_items [[label, bytes]], transient_items / external_transient_items [[phase, label, bytes]]} of
    meshing.estimate_memory (CPU; estimate: an already
    computed one); a failing estimate is logged as a warning (an estimator bug), never stops the build (None then)."""
    from . import meshing
    try:
        if estimate is None:
            estimate = meshing.estimate_memory(spec, parameters, registry,
                                               meshing.estimate_meshes(spec, parameters, registry))
        return {'gpu_steady': _optional_bytes(estimate.gpu_steady),
                'gpu_build_peak': _optional_bytes(estimate.gpu_build_peak),
                'gpu_run': _optional_bytes(estimate.gpu_run),
                'gpu_external': _optional_bytes(estimate.gpu_external),
                'external_extra': _optional_bytes(estimate.external_extra),
                'host': _optional_bytes(estimate.host),
                'items': [[str(label), _optional_bytes(size)] for label, size in (estimate.items or ())],
                'run_items': [[str(label), _optional_bytes(size)] for label, size in (estimate.run_items or ())],
                'transient_items': [[str(phase), str(label), _optional_bytes(size)]
                                    for phase, label, size in (estimate.transient_items or ())],
                'external_items': [[str(label), _optional_bytes(size)]
                                   for label, size in (estimate.external_items or ())],
                'external_transient_items': [[str(phase), str(label), _optional_bytes(size)]
                                             for phase, label, size in (estimate.external_transient_items or ())],
                'external_peak': _optional_bytes(estimate.external_peak)}
    except Exception as exc:
        log('warning', f'GPU memory estimate failed (no estimate check for this build): {_exception_message(exc)}')
        return None


def _memory_record(ledger, estimate, compare):
    """MemoryLedger.as_record() + 'estimate' (the _memory_estimate dict + steady_deviation, peak_deviation; None
    without an estimate); compare=False (failed build) leaves the deviations None."""
    record = ledger.as_record()
    if estimate is None:
        record['estimate'] = None
        return record
    unseen_bytes = int(record.get('start_free_bytes') or 0)   # the pool peak cannot see temporaries placed there
    record['estimate'] = dict(
        estimate, peak_unseen_bytes=unseen_bytes,
        steady_deviation=_deviation(record['total_bytes'], estimate['gpu_steady']) if compare else None,
        peak_deviation=(_deviation(record['peak_bytes'], estimate['gpu_build_peak'], unseen_bytes) if compare
                        else None))
    return record


def estimate_gap(measured_bytes, estimated_bytes, unseen_bytes=0) -> int:
    """Held minus estimated bytes, the held bytes taken within [measured, measured + unseen_bytes] nearest the
    estimate (0 inside): a pool high-water mark misses up to the free blocks stranded before it started."""
    measured, estimated = int(measured_bytes), int(estimated_bytes)
    if estimated <= measured:
        return measured - estimated
    return min(measured + max(int(unseen_bytes), 0) - estimated, 0)


def estimate_verdict(measured_bytes, estimated_bytes, unseen_bytes=0) -> str:
    """'over' (the estimate books more than was held), 'unaccounted' (more held than ESTIMATE_SMALL_ITEMS beyond it)
    or '' (just below the held bytes, or within ESTIMATE_FLOOR_BYTES); estimate_gap, module constants."""
    difference = estimate_gap(measured_bytes, estimated_bytes, unseen_bytes)
    if abs(difference) <= ESTIMATE_FLOOR_BYTES:
        return ''
    if difference < 0:
        return 'over'
    return 'unaccounted' if difference > ESTIMATE_SMALL_ITEMS * max(int(estimated_bytes), 1) else ''


def _verdict_text(label, measured_bytes, estimated_bytes, deviation, verdict):
    """One mismatch phrase of an estimate check ('' for no verdict)."""
    measured, estimated = gpu_memory.format_bytes(measured_bytes), gpu_memory.format_bytes(estimated_bytes)
    if verdict == 'over':
        return (f'{label}: {estimated} estimated but {measured} held ({deviation:+.0%}): an item booked that the run '
                'did not hold (not there, not held at the same time, or smaller)')
    if verdict == 'unaccounted':
        return (f'{label}: {measured} held vs {estimated} estimated ({deviation:+.0%}): beyond the '
                f'{ESTIMATE_SMALL_ITEMS:.0%} small-item margin, an array no item lists (a missing item, or memory '
                'the code could free)')
    return ''


def _estimate_check_message(record):
    """Warning text when the ledger's kept or peak bytes are not just above the estimate (estimate_verdict), else ''."""
    estimate = record.get('estimate')
    if not estimate:
        return ''
    mismatches = []
    for label, measured_bytes, estimated_bytes, deviation, unseen_bytes in (
            ('kept', record['total_bytes'], estimate['gpu_steady'], estimate['steady_deviation'], 0),
            ('build peak', record['peak_bytes'], estimate['gpu_build_peak'], estimate['peak_deviation'],
             estimate.get('peak_unseen_bytes') or 0)):
        if deviation is not None:
            text = _verdict_text(label, measured_bytes, estimated_bytes, deviation,
                                 estimate_verdict(measured_bytes, estimated_bytes, unseen_bytes))
            if text:
                mismatches.append(text)
    if not mismatches:
        return ''
    memory = 'Host (cpu backend, traced numpy)' if record.get('kind') == 'cpu' else 'GPU'
    return (f"{memory} memory estimate: {'; '.join(mismatches)}. The memory_estimate of a physics or feature type "
            "misreads its solver (summary.json build.gpu_breakdown and build.gpu_estimate list both sides)")


def run_memory_record(build_memory, measured_bytes, unseen_bytes=0):
    """({measured_bytes, estimated_bytes, unseen_bytes, deviation, verdict}, warning text or '') of a study run's pool
    high-water mark since its build against the estimate's gpu_steady + gpu_run (estimate_verdict; unseen_bytes: the
    free pool blocks stranded before the build, which the mark misses when the run reuses them); (None, '') without
    an estimate."""
    estimate = (build_memory or {}).get('estimate')
    if not estimate or estimate.get('gpu_steady') is None:
        return None, ''
    estimated_bytes = estimate['gpu_steady'] + (estimate.get('gpu_run') or 0)
    deviation = _deviation(measured_bytes, estimated_bytes, unseen_bytes)
    verdict = estimate_verdict(measured_bytes, estimated_bytes, unseen_bytes)
    record = {'measured_bytes': int(measured_bytes), 'estimated_bytes': int(estimated_bytes),
              'unseen_bytes': int(unseen_bytes), 'deviation': deviation, 'verdict': verdict}
    if not verdict:
        return record, ''
    return record, ('GPU memory of the run (pool high-water mark since the build vs kept + run items): '
                    + _verdict_text('run', measured_bytes, estimated_bytes, deviation, verdict)
                    + '. The memory_estimate of a physics or feature type (its run_items / transient_items: memory '
                    'allocated after the build, and the phases in which it lives) misreads its solver')


def external_memory_record(build_memory, measured_bytes):
    """({measured_bytes, estimated_bytes, deviation, verdict}, warning text or '') of the cuDSS high-water mark outside
    the pool since the build started (cudss_factor.external_high_water_bytes minus the bytes live then) against the
    estimate's external_peak (run-long factors + the largest phase of phase-bound ones + external_extra); (None, '')
    without an estimate or when neither side has any."""
    estimate = (build_memory or {}).get('estimate')
    if not estimate or estimate.get('gpu_external') is None:
        return None, ''
    estimated_bytes = int(estimate['external_peak'])
    if not estimated_bytes and not measured_bytes:
        return None, ''
    deviation = _deviation(measured_bytes, estimated_bytes)
    verdict = estimate_verdict(measured_bytes, estimated_bytes)
    record = {'measured_bytes': int(measured_bytes), 'estimated_bytes': int(estimated_bytes), 'deviation': deviation,
              'verdict': verdict}
    if not verdict:
        return record, ''
    return record, ('GPU memory outside the pool (cuDSS factors): '
                    + _verdict_text('external', measured_bytes, estimated_bytes, deviation, verdict)
                    + '. A physics estimator books the wrong factors (external_items through '
                    'cudss_factor.factor_device_bytes), or the fill rule misreads the ordering (summary.json solvers '
                    'counters cudss_lu_entries / cudss_permanent_bytes)')


def _optional_bytes(value):
    """int(round(value)) bytes, None stays None."""
    return None if value is None else int(round(float(value)))


def _deviation(measured_bytes, estimated_bytes, unseen_bytes=0):
    """estimate_gap / estimated: measured / estimated - 1 when nothing is unseen (estimate floored at 1 byte), None
    without an estimate."""
    if estimated_bytes is None:
        return None
    return estimate_gap(measured_bytes, estimated_bytes, unseen_bytes) / max(estimated_bytes, 1)


# ---- error reporting ----
@contextmanager
def _issue_at(path):
    """Wrap any exception (except ModelError) into a ModelError with one Issue at path; a GPU out-of-memory message
    names the budget and the remedies."""
    try:
        yield
    except ModelError:
        raise
    except Exception as exc:
        message = (gpu_memory.out_of_memory_message(exc) if gpu_memory.is_out_of_memory(exc)
                   else _exception_message(exc))
        raise ModelError([Issue('error', path, message)]) from exc


def _exception_message(exc):
    text = error_message(exc)
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return text or type(exc).__name__
    return f'{type(exc).__name__}: {text}' if text else type(exc).__name__


def _print_log(level, message):
    print(f'[{level}] {message}')
