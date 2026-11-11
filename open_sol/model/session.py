"""A built model on the GPU (worker thread only): field sampling, feature results and derived-value evaluation, each
on the mesh of the physics concerned (S7.2); close() releases its GPU data."""
import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field

from ..core.gpu_memory import pool_used_bytes
from .analytic import SEGMENT_SAMPLINGS, derived_curve_result, is_curve_source, result_value, split_key
from .expressions import ExpressionError, compile_expression, evaluate_parameters, evaluate_scalar, referenced_names
from .field_access import DISPLAY_MAX_PIXELS
from .field_access_tensor import LineAccess, MappedAccess, TensorAccess
from .field_access_tetrahedra import TetrahedraAccess
from .field_access_triangles import TriangleAccess
from .field_data import PLANE_AXES, FieldData, LineProfile, VolumeData   # re-exported
from .geometry import domain_map
from .schema import FIELD_DERIVED_OPERATIONS, GLOBAL_DERIVED_OPERATION, output_field_names, scalar_output_names

VOLUME_MAX_BYTES = 2 ** 30   # Session.volume default size guard of the host float32 values
CANVAS_NATIVE_CELLS = 4_000_000   # canvas_field(): graded Cartesian planes up to this many cells come native
ELEMENT_ACCESS = {'triangles': TriangleAccess, 'tetrahedra': TetrahedraAccess}   # grid kind -> FieldAccess class


@dataclass
class Snapshot:
    time: float
    fields: dict                # 'physics.component' -> FieldData (3D: the mid-height z cut)
    label: str | None = None    # e.g. 'lambda = 1550 nm' (frequency_domain); None for time snapshots
    cuts: list = dataclass_field(default_factory=list)   # 3D StudySpec.stored_cuts: {'physics.component': FieldData}
                                                          # per stored cut, in stored_cuts order
    segments: dict = dataclass_field(default_factory=dict)   # 3D: {line curve name: LineProfile} of every 3D-segment
                                                              # analytic line curve at this state
    live: bool = False          # holds the state the session keeps after the study when that is not the last
                                # snapshot (eigenfrequency resonance 1, mode_analysis mode_index)


def live_snapshot_index(snapshots):
    """Index of the last snapshot flagged live (summary live_snapshot), None = the last snapshot is the live state."""
    return next((index for index in range(len(snapshots) - 1, -1, -1) if snapshots[index].live), None)


@dataclass
class BuildSummary:
    """CPU-only description of a built session (sent to the GUI)."""
    title: str
    nx: int
    ny: int
    n_cells: int
    dt: float | None            # runner sync step; None when nothing steps in time
    dx_min: float
    dx_max: float
    npml_sides: dict
    physics: list               # [(name, physics type label)]
    fields: list                # [(physics, FieldInfo)]
    feature_results: list       # [(physics, feature, result_kind)]
    gpu_bytes: int              # cupy pool used bytes of the whole process (other live sessions included)
    build_seconds: float
    warnings: list
    session_gpu_bytes: int | None = None   # pool used minus Session.pool_baseline_bytes; None without a baseline
    gpu_breakdown: list = dataclass_field(default_factory=list)   # [(build stage, pool used-bytes change)] (ledger)
    stage_seconds: list = dataclass_field(default_factory=list)   # [(build stage, wall s)] (ledger, + validation)
    gpu_peak_bytes: int | None = None      # build high-water mark of the pool (MemoryLedger.peak_bytes)
    gpu_budget: int | None = None          # pool limit during the build, None when none was set
    gpu_estimate: dict | None = None       # builder memory-estimate record (gpu_steady, gpu_build_peak, host, items,
                                           # steady_deviation, peak_deviation), None when no estimate was made
    nz: int = 1                            # 3D models: z cells (PML included); n_cells = nx * ny * nz
    z_centers: tuple = ()                  # 3D models: window cell-centre heights [m] (results cut-plane selector)
    x_centers: tuple = ()                  # 3D models: window cell centres along x / y [m] (side cut planes)
    y_centers: tuple = ()
    meshes: list = dataclass_field(default_factory=list)   # Session.mesh_records(): one record per built mesh; the
                                                            # nx / ny / dx fields above describe the first mesh


class Session:
    """Solvers, runner and renderers of a built ModelSpec plus stored snapshots; use from the worker thread only.
    close() releases the GPU data; snapshots, spec, registry and parameters stay readable. Several meshes (S7.2):
    fields, feature results and 3D reads of a physics act on its own mesh (a mesh view: a shallow copy whose domain /
    context are that mesh's), derived values on the mesh of the first physics their expression names (D3), the other
    fields interpolated onto it. A physics on a triangle mesh exposes renderer.fem_field(component) -> (Lagrange
    DofMap, coefficients) instead of device_field (rasterised fields, quadrature-sample derived values)."""

    def __init__(self, spec, registry, parameters, domain, sim, context, features, build_seconds, warnings=()):
        self.spec = spec
        self.registry = registry
        self.parameters = parameters
        self.domain = domain                # the first mesh's (sim.domain); a mesh view: its own mesh's
        self.domains = {mesh_domain.mesh_name: mesh_domain for mesh_domain in sim.domains}   # mesh name -> domain
        self.sim = sim
        self.runner = sim.runner
        self.physics = dict(sim.physics)
        self.context = context
        self.features = features            # (physics, feature) -> runtime object (None when apply returned None)
        self.snapshots = []
        self.build_seconds = float(build_seconds)
        self.warnings = list(warnings)
        self._renderers = {}
        self._access = None                 # this view's FieldAccess (the `access` property; display caches inside)
        self.domain_map = domain_map(spec, parameters)   # geometry.DomainMap: Cartesian, polar or spherical window
        self._field_names = output_field_names(spec, registry)
        self._scalar_names = scalar_output_names(spec, registry, parameters)   # '<physics>_<scalar>' (E5)
        self.pool_baseline_bytes = None     # pool used bytes before this build (set by study.build_session)
        self.pool_baseline_free_bytes = 0   # free pool bytes then (chunk remainders of open Results' arrays)
        self.external_baseline_bytes = None   # live cuDSS bytes outside the pool before this build (build_session)
        self.build_memory = None            # builder memory record: MemoryLedger.as_record() + 'estimate' (or None)
        self.time_estimate = None           # core.time_estimate.TimeEstimate of one study run (None: no profile)
        self.closed = False
        self._parent = None                 # a mesh view: the Session it acts for (None: this is the Session)
        self._mesh_views = {}               # mesh name -> mesh view (other meshes than the first)
        self._window_transfers = {}         # (source mesh, target mesh) -> window-to-window interpolating GridTransfer
        # {(mesh, physics, field): window array} while shared_field_reads is active (None: every read renders afresh)
        self._field_reads = None
        # {(mesh, renderer id, component): (renderer, plane source)} while shared_plane_reads is active
        self._plane_reads = None

    @property
    def time(self) -> float:
        self._require_open()
        return float(self.runner.t)

    def close(self):
        """Release the GPU data (solvers, runner, couplings, feature objects, renderers, build caches) without
        relying on gc; idempotent. Afterwards only snapshots, spec, registry and parameters are usable."""
        if self.closed:
            return
        context = self.context
        if context is not None:
            for mesh_context in [context, *context.other_meshes.values()]:
                if mesh_context.domain is not None:
                    mesh_context.release_caches()
                mesh_context.material_regions = []
                mesh_context.sim = mesh_context.domain = None
            context.release_shared()   # solver runtimes, grid transfers, time updates hold device data
            context.other_meshes = {}
        if self.sim is not None:
            self.sim.clear()
        self.physics.clear()
        self.features.clear()
        self._renderers.clear()
        self._access = None
        self._mesh_views.clear()
        self._window_transfers.clear()
        self.domains = {}
        self.domain = self.sim = self.runner = self.context = None
        self.closed = True

    # ---- reused sweep builds (SweepSpec.reuse_build, model.sweep_reuse) ----
    def set_sweep_value(self, parameter, value):
        """Set the swept parameter to value on this kept build: the spec and parameters change in place (the build
        contexts and mesh views share them), then every parameter follower whose values changed re-sets them."""
        self._require_open()
        root = self._parent or self
        root.spec.parameter(parameter).expression = repr(float(value))
        parameters = evaluate_parameters(root.spec.parameters)
        root.parameters.clear()
        root.parameters.update(parameters)
        for follower in root.context.parameter_followers:
            follower.update(root.parameters)

    def reset_run_records(self):
        """Before a rerun of this kept build: drop the snapshots, set the runner clock to 0 and call every physics'
        reset_records() (solver_hooks.RerunSolver; take the kept snapshots first)."""
        self._require_open()
        root = self._parent or self
        root.snapshots.clear()   # in place: mesh views share the list
        root.runner.t, root.runner.tick = 0.0, 0
        for solver in root.runner.physics.values():
            solver.reset_records()

    # ---- meshes (S7.2) ----
    def domain_of(self, physics):
        """Domain (Domain3D, TriangleDomain) of the mesh a physics runs on."""
        self._require_open()
        self._solver(physics)
        return self.sim.domain_of(physics)

    def mesh_of(self, physics) -> str:
        """Name of the mesh a physics runs on."""
        return self.domain_of(physics).mesh_name

    def window_shape(self, physics=None) -> tuple:
        """Window cells (wx, wy(, wz)) of the physics' mesh (None: the first mesh); a triangle mesh: (window
        elements,); a tetrahedron mesh: its sample lattice (the cells of its 3D reads)."""
        self._require_open()
        return (self if physics is None else self._physics_view(physics)).access.window_shape()

    @property
    def access(self):
        """FieldAccess of this view's mesh (cached): by grid kind, a tensor window also by its DomainMap (1D line,
        polar / spherical, Cartesian)."""
        if self._access is None:
            if self.domain.grid_kind != 'tensor':
                access_class = ELEMENT_ACCESS[self.domain.grid_kind]
            else:
                access_class = (LineAccess if self.domain_map.line else MappedAccess if self.domain_map.mapped
                                else TensorAccess)
            self._access = access_class(self)
        return self._access

    def mesh_records(self) -> list:
        """[{name, kind, shape (cells incl. padding; null on triangles), elements (triangles, else null), cells,
        window_cells, dx_min, dx_max (physical cell lengths / element edges, m), physics, ledger [[build stage, bytes]
        of this mesh: owners naming it or one of its physics; every line with one mesh]}] in mesh order."""
        self._require_open()
        breakdown = (self.build_memory or {}).get('breakdown', ())
        records = []
        for name in self.domains:
            view = self._mesh_view(name)
            mesh_physics = [physics for physics in self.physics if self.mesh_of(physics) == name]
            prefixes = tuple(prefix for physics in mesh_physics   # the builder's ledger owners of its physics
                             for prefix in (f"physics '{physics}'", f"solver '{physics}.", f"feature '{physics}/"))
            ledger = [[owner, int(size)] for owner, size in breakdown
                      if len(self.domains) == 1 or owner.endswith(f' {name}') or owner.startswith(prefixes)]
            dx_min, dx_max = view.access.size_range()
            records.append({'name': name, **view.access.mesh_record(), 'dx_min': dx_min, 'dx_max': dx_max,
                            'physics': mesh_physics, 'ledger': ledger})
        return records

    def mesh_arrays(self) -> dict:
        """{mesh name: host arrays} for meshes.npz: tensor {kind, x_edges, y_edges(, z_edges)} = the window cell edges
        in Domain coordinates (logical r - radius_min / angle on polar windows), triangles {kind, points, triangles,
        pml_faces}, tetrahedra {kind, points, tets}."""
        self._require_open()
        return {name: self._mesh_view(name).access.mesh_arrays() for name in self.domains}

    def _mesh_view(self, mesh_name):
        """This Session on one of its meshes: itself for the first mesh, else a cached shallow copy whose domain,
        context and display caches are that mesh's (everything else shared)."""
        root = self._parent or self
        if mesh_name == root.domain.mesh_name:
            return root
        view = root._mesh_views.get(mesh_name)
        if view is None:
            if mesh_name not in root.domains:
                raise KeyError(f"no mesh '{mesh_name}' in this session; built: {', '.join(root.domains)}")
            view = copy.copy(root)
            view.domain = root.domains[mesh_name]
            view.context = root.context.context_of_mesh(mesh_name)
            view._access = None
            view._mesh_views = {}
            view._parent = root
            root._mesh_views[mesh_name] = view
        return view

    def _physics_view(self, physics):
        """The mesh view of the mesh physics runs on."""
        return self._mesh_view(self.sim.domain_of(physics).mesh_name)

    def _evaluation_view(self, names):
        """S7.2 D3: the mesh view of the first physics (model order) whose output fields the names read (as validate's
        info); this view when they read none."""
        order = [physics_spec.name for physics_spec in self.spec.physics]
        named = {self._field_names[name][0] for name in names if name in self._field_names}
        named = [physics for physics in order if physics in named]
        return self._physics_view(named[0]) if named else self

    # ---- description ----
    def summary(self) -> BuildSummary:
        """BuildSummary of the first mesh (a triangle mesh: nx = elements, ny = 1) plus every mesh's record."""
        self._require_open()
        grid = getattr(self.domain, 'grid', None)   # a TriangleDomain has none
        dx_min, dx_max = self.access.size_range()
        shape = self.domain.materials.shape
        dt = self.runner.dt_sync
        feature_results = []
        for physics_name, feature_name in self.features:
            result_kind = self._feature_type(physics_name, feature_name).result_kind
            if result_kind is not None:
                feature_results.append((physics_name, feature_name, result_kind))
        used_bytes = pool_used_bytes()
        baseline_bytes = self.pool_baseline_bytes
        memory = self.build_memory or {}
        return BuildSummary(
            title=self.spec.title, nx=shape[0] if grid is None else grid.nx, ny=1 if grid is None else grid.ny,
            n_cells=math.prod(shape), dt=float(dt) if math.isfinite(dt) else None, dx_min=dx_min, dx_max=dx_max,
            npml_sides=dict(getattr(self.domain, 'npml_sides', None) or {}),
            physics=[(name, self._physics_type(name).label) for name in self.physics],
            fields=self.available_fields(), feature_results=feature_results,
            gpu_bytes=used_bytes, build_seconds=self.build_seconds, warnings=list(self.warnings),
            session_gpu_bytes=None if baseline_bytes is None else used_bytes - baseline_bytes,
            gpu_breakdown=[(owner, size) for owner, size in memory.get('breakdown', ())],
            stage_seconds=[(owner, seconds) for owner, seconds in memory.get('seconds', ())],
            gpu_peak_bytes=memory.get('peak_bytes'), gpu_budget=memory.get('budget'),
            gpu_estimate=memory.get('estimate'),
            nz=shape[2] if len(shape) == 3 else 1, z_centers=self.cut_centers('z'), x_centers=self.cut_centers('x'),
            y_centers=self.cut_centers('y'), meshes=self.mesh_records())

    def renderer(self, physics):
        """Cached PhysicsRenderer of a physics."""
        self._require_open()
        renderer = self._renderers.get(physics)
        if renderer is None:
            renderer = self._solver(physics).renderer()
            self._renderers[physics] = renderer
        return renderer

    def available_fields(self):
        """[(physics, FieldInfo)] of every renderable component of this study type (FieldInfo.requires_study), physics in
        build order."""
        self._require_open()
        study_type = self.spec.study.type
        return [(name, info) for name in self.physics for info in self.renderer(name).field_infos.values()
                if not info.requires_study or info.requires_study == study_type]

    # ---- fields ----
    def display_field(self, component, physics=None, max_pixels=DISPLAY_MAX_PIXELS, plane=None) -> FieldData:
        """Field on a uniform display grid over physical_extent(); nearest-cell GPU resampling (native_shape set) when
        the mesh is non-uniform or the window has more than max_pixels cells. physics=None -> unique owner; plane as
        in field(). The physics' own mesh; a triangle mesh: a raster of max_pixels over the window (+ triangulation)."""
        self._require_open()
        physics = self._owner(component, physics)
        view = self._physics_view(physics)
        if view is not self:
            return view.display_field(component, physics, max_pixels, plane)
        return self.access.display_field(physics, component, max_pixels, plane)

    def canvas_field(self, component, physics=None, plane=None, max_cells=CANVAS_NATIVE_CELLS) -> FieldData:
        """Field for the GUI canvas: on a graded Cartesian tensor mesh the native window cells with their centres
        (field(); the canvas resamples them to the zoomed range) when the plane has at most max_cells cells, else
        display_field()."""
        self._require_open()
        physics = self._owner(component, physics)
        view = self._physics_view(physics)
        if view is not self:
            return view.canvas_field(component, physics, plane, max_cells)
        return self.access.canvas_field(physics, component, plane, max_cells)

    def field(self, component, physics=None, plane=None, native=False) -> FieldData:
        """Field on the native window cells, with the centres of those cells (as Results.field). 3D models: plane
        (axis 'x' | 'y' | 'z', position [m]) cuts normal to axis at the window cell containing the position (None =
        mid-height z); 2D models take None only. Polar / spherical windows: the containing-cell image of
        _mapped_field_data at 4 pixels per window cell (within MAPPED_FIELD_PIXELS), or with native the window cells
        themselves (_logical_field_data); native changes nothing on a Cartesian grid. A cylindrical (r, phi, z) sector
        takes z cuts only, each shown as the polar window. 1D models: the (wx, 1) cells along the drawing's x axis
        (_line_field_data), with or without native. The physics' own mesh (FieldData.mesh); a triangle mesh: a raster
        of 4 pixels per element (within TRIANGLE_FIELD_PIXELS) plus the triangulation."""
        self._require_open()
        physics = self._owner(component, physics)
        view = self._physics_view(physics)
        if view is not self:
            return view.field(component, physics, plane, native)
        return self.access.field(physics, component, plane, native)

    def segment_profile(self, component, start, end, physics=None, sampling='nearest') -> LineProfile:
        """Field along the 3D segment start -> end [m] (x, y, z) of a 3D model: sampling 'nearest' = the containing
        window cell of each sample (segment_cells), 'trilinear' = interpolated between the 8 surrounding cell centres
        (centres = the sample points); NaN outside the window, no mirror unfolding; z planes via device_plane when the
        renderer has it (no volume temporaries); on the physics' own mesh."""
        self._require_open()
        physics = self._owner(component, physics)
        view = self._physics_view(physics)
        if view is not self:
            return view.segment_profile(component, start, end, physics, sampling)
        if self.domain.dimensions != 3:
            raise ValueError('segment_profile needs a 3D model (domain size_z); 2D line profiles sample field()')
        if sampling not in SEGMENT_SAMPLINGS:
            raise ValueError(f'sampling must be one of {SEGMENT_SAMPLINGS}, got {sampling!r}')
        start, end = (tuple(float(value) for value in point) for point in (start, end))
        if len(start) != 3 or len(end) != 3:
            raise ValueError(f'segment start and end must be (x, y, z) in m, got {start!r}, {end!r}')
        if start == end:
            raise ValueError(f'the segment has zero length (start = end = {start!r}): give distinct points')
        return self.access.segment_profile(physics, component, start, end, sampling)

    def volume(self, component, physics=None, max_bytes=VOLUME_MAX_BYTES) -> VolumeData:
        """Host float32 (wx, wy, wz) copy of a 3D field on the window cells with their centres (a cylindrical sector:
        r [m], phi [rad], z [m]), fetched one z plane at a time (device_plane when the renderer has it); ValueError
        above max_bytes of host values; on the physics' own mesh."""
        self._require_open()
        if self.domain.dimensions != 3:
            raise ValueError('volume needs a 3D model (domain size_z); 2D models: field()')
        physics = self._owner(component, physics)
        view = self._physics_view(physics)
        if view is not self:
            return view.volume(component, physics, max_bytes)
        window_shape = self.access.window_shape()
        n_bytes = 4 * math.prod(window_shape)
        if n_bytes > max_bytes:
            raise ValueError(f'the {" x ".join(str(count) for count in window_shape)} window field needs '
                             f'{n_bytes / 2**20:.0f} MiB, above max_bytes {max_bytes / 2**20:.0f} MiB: raise '
                             f'max_bytes, or export cut planes (field(plane=)) instead')
        return self.access.volume(physics, component)

    def cut_centers(self, axis, physics=None) -> tuple:
        """Window cell centres [m] along axis of a 3D model (the cut positions field() can show) on the physics' mesh
        (None: the first mesh); () in 2D and for x / y of a cylindrical sector (z cuts only)."""
        self._require_open()
        if physics is not None and self._physics_view(physics) is not self:
            return self._physics_view(physics).cut_centers(axis)
        if self.domain.dimensions != 3:
            return ()
        return self.access.cut_centers(axis)

    def feature_result(self, physics, feature, **options) -> dict:
        """FeatureType.result of an enabled, built feature (options passed through as a dict); the callback gets the
        mesh view of the physics (session.domain = its mesh's)."""
        self._require_open()
        key = (physics, feature)
        if key not in self.features:
            built = ', '.join(f'{p}.{f}' for p, f in self.features) or 'none'
            raise KeyError(f"no built feature '{feature}' in physics '{physics}'; built features: {built}")
        feature_type = self._feature_type(physics, feature)
        if feature_type.result is None:
            raise ValueError(f"feature '{physics}.{feature}' ({feature_type.name}) produces no result")
        feature_spec = self.spec.physics_by_name(physics).feature(feature)
        return feature_type.result(self._physics_view(physics), self.physics[physics], self.features[key], feature_spec,
                                   dict(options))

    # ---- derived values ----
    def evaluate(self, expression, selection=(), operation='mean', point='', earlier=None) -> float:
        """Reduce an expression of parameters, x, y (3D: z), t, scalar outputs, output fields and `earlier` {derived
        value name: number} over the window (or a domain selection): 'mean' (area-weighted), 'max', 'min', 'integral'
        (sum of value * cell area), 'mode_volume' (integral / max, or / the value at point) or 'at' (the cell holding
        point object `point`); 3D models: over the window volume; a model with an axisymmetric solver: over ring
        volumes 2 pi r_c dx dy; polar / spherical: over the core.metric sector areas / shell volumes. An integral is
        doubled per mirror face the selection reaches (the unfolded total of a mirror-even integrand); max / min (and
        the mode_volume maximum) include the mirror images there (field parities). An 'interface' selection reduces
        one-sided outline samples instead (no images). Several meshes (D3): on the mesh of the first physics (model
        order) whose fields the expression names, the other fields interpolated onto it; a triangle mesh reduces
        degree-4 quadrature samples, a tetrahedron mesh renderer samples at quadrature points (FieldAccess.evaluate of
        the mesh's kind)."""
        self._require_open()
        if operation not in FIELD_DERIVED_OPERATIONS:
            raise ValueError(f"unknown operation {operation!r} for fields; valid: "
                             f"{', '.join(FIELD_DERIVED_OPERATIONS)}")
        compiled = compile_expression(expression)
        view = self._evaluation_view(compiled.names)
        if view is not self:
            return view.evaluate(expression, selection, operation, point, earlier)
        return self.access.evaluate(compiled, selection, operation, point, earlier)

    def derived_value(self, derived, fetch=None, earlier=None) -> float:
        """Value of a DerivedValueSpec: evaluate() over its selection, or its source (a feature result, or a derived
        curve of the spec built from them) reduced by analytic.result_value; fetch('physics.feature') -> result dict
        (default feature_result); earlier: {name: value} of the derived values listed before it (expression names).
        Operation 'expression' (source None): one float64 number of parameters, scalar outputs, t and earlier."""
        earlier = earlier or {}
        if derived.source is None and derived.operation == GLOBAL_DERIVED_OPERATION:
            self._require_open()
            names = compile_expression(derived.expression).names
            return evaluate_scalar(derived.expression, self._plain_namespace(names, earlier))
        if derived.source is None:
            return self.evaluate(derived.expression, derived.selection, derived.operation, derived.point, earlier)
        if fetch is None:
            fetch = self._feature_result_of_key
        if is_curve_source(derived.source):
            result = derived_curve_result(self.spec.derived_curve(derived.source), fetch, self.parameters)
        else:
            result = fetch(derived.source)
        return result_value(result, derived, {**earlier, **self.parameters})

    def _feature_result_of_key(self, key):
        return self.feature_result(*split_key(key))

    def derived_values(self) -> dict:
        """{name: value} of every DerivedValueSpec (each feature result fetched once, each window field read once:
        shared_field_reads); NaN (and a logged warning) for entries that fail."""
        self._require_open()
        results, fetched = {}, {}

        def fetch(key):
            if key not in fetched:
                fetched[key] = self._feature_result_of_key(key)
            return fetched[key]
        with self.shared_field_reads():
            for derived in self.spec.derived_values:
                try:
                    results[derived.name] = self.derived_value(derived, fetch, results)
                except Exception as exc:   # any failure of user input -> NaN, by contract
                    results[derived.name] = math.nan
                    self.context.log('warning', f"derived value '{derived.name}': {exc}", keep=False)
        return results

    def shared_field_reads(self):
        """Context: within the block every 2D tensor window field (state entry or renderer field) is read once per
        (mesh, physics, field) and shared by the evaluations (a pass over several derived values of one state); the
        state must not change inside. Nested blocks share the outer cache."""
        return self._shared_reads('_field_reads')

    def shared_plane_reads(self):
        """Context: within the block a 3D tensor field without a renderer device_plane is rendered once per (mesh,
        physics, component) and every cut / profile slices that volume (one snapshot record of one field: keep the
        block to one component, each held volume is device memory); the state must not change inside."""
        return self._shared_reads('_plane_reads')

    @contextmanager
    def _shared_reads(self, attribute):
        root = self._parent or self
        if getattr(root, attribute) is not None:
            yield
            return
        setattr(root, attribute, {})
        try:
            yield
        finally:
            setattr(root, attribute, None)

    # ---- helpers ----
    def _require_open(self):
        if self.closed or (self._parent is not None and self._parent.closed):
            raise RuntimeError('session closed: its GPU data was released (rebuild or rerun)')

    def _solver(self, physics):
        try:
            return self.physics[physics]
        except KeyError:
            raise KeyError(f"no physics '{physics}'; built: {', '.join(self.physics)}") from None

    def _physics_type(self, physics):
        return self.registry.physics_type(self.spec.physics_by_name(physics).type)

    def _feature_type(self, physics, feature):
        return self._physics_type(physics).feature_type(self.spec.physics_by_name(physics).feature(feature).type)

    def _owner(self, component, physics):
        """Physics providing component (checked when given); ValueError unless exactly one provides it."""
        if physics is not None:
            self.renderer(physics).field_info(component)
            return physics
        owners = [name for name in self.physics if component in self.renderer(name).field_infos]
        if len(owners) != 1:
            raise ValueError(f"field '{component}' is provided by {owners or 'no physics'}; "
                             f"pass physics= (available: {[(p, i.name) for p, i in self.available_fields()]})")
        return owners[0]

    def _axisymmetric_solvers(self, names=()):
        """Solvers with `axisymmetric` set among the physics whose output fields `names` read (none read: the physics
        on this view's mesh), so a planar physics of a mixed model keeps plain areas."""
        read = {self._field_names[name][0] for name in names if name in self._field_names}
        if not read:
            read = {physics for physics in self.physics
                    if self.sim.domain_of(physics).mesh_name == self.domain.mesh_name}
        return [solver for physics, solver in self.physics.items()
                if physics in read and getattr(solver, 'axisymmetric', False)]

    def _has_axisymmetric_solver(self, names=()):
        """True when an evaluation of `names` weighs by ring volumes 2 pi r_c (_axisymmetric_solvers)."""
        return bool(self._axisymmetric_solvers(names))

    def _ring_angle(self, names=()):
        """Ring angle of those axisymmetric solvers' `ring_angle` (pi when one solves an azimuthal order m != 0,
        symmetry 2.14; validate keeps one m per model), else 2 pi."""
        return min((solver.ring_angle for solver in self._axisymmetric_solvers(names) if hasattr(solver, 'ring_angle')),
                   default=2.0 * math.pi)

    def _plain_namespace(self, names, earlier=None):
        """Mesh-free part of an evaluation namespace: parameters, earlier derived values {name: float}, solver scalars
        (floats) and t."""
        namespace = {name: self.parameters[name] for name in names if name in self.parameters}
        namespace.update({name: float(value) for name, value in (earlier or {}).items()
                          if name in names and name not in namespace})
        solver_scalars = {}
        for name in names:
            if name in namespace or name not in self._scalar_names:
                continue
            physics, scalar = self._scalar_names[name]
            if physics not in solver_scalars:
                solver_scalars[physics] = self._solver(physics).scalars()
            if scalar not in solver_scalars[physics]:
                reported = ', '.join(sorted(solver_scalars[physics])) or 'none'
                raise ValueError(f"scalar output '{name}' has no value after this run (physics '{physics}' reports: "
                                 f"{reported})")
            namespace[name] = float(solver_scalars[physics][scalar])
        if 't' in names and 't' not in namespace:
            namespace['t'] = self.time
        return namespace


def earlier_derived_names(spec, derived):
    """Names of the derived values listed before `derived` that its expression references (empty on bad syntax)."""
    index = next(index for index, other in enumerate(spec.derived_values) if other is derived)
    earlier = {other.name for other in spec.derived_values[:index]}
    try:
        names = referenced_names(derived.expression)
    except ExpressionError:
        return frozenset()
    return frozenset(names & earlier)
