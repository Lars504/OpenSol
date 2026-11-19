# open_sol/model — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first): spec, validation, build, session, studies, worker,
templates, describe. Current state, no changelog.

## Model layer
- Expressions: `evaluate` is silent on invalid / divide on both backends, so consumers of its values check finiteness
  themselves (evaluate_scalar, builder `_check_range`, curves drop non-finite samples). An uncalled function name is a
  free name that only a namespace entry supplies (an FDFD probe xy series 'abs').
- Names: `<physics>_<field>` always, plain `<field>` when one physics alone provides it (parameters may not use it).
  Settings / feature params / material properties see parameters + x, y (field_dependent) + state names and t
  (state_dependent) / t alone (time_dependent: no x, y with t, `schema._number`); coupling 'expression': value + x, y +
  parameters; derived values: + t + `output_field_names` + `scalar_output_names` (settings resolved per call: names
  follow e.g. n_modes; with a `source`: the result's sample names instead, which shadow parameters); derived `at` /
  `reference`: parameters only; analytic curves: parameters + `analytic.curve_variables(target kind, series)`
  (variables shadow; the one rule of validate, the settings form and `feature_curve`). z (3D only, root file) is also
  refused in 2D derived values and line curves (cut height: `analytic.field_plane_height`).
- Analytic comparison (`analytic.py`, numpy only, GUI-thread safe): analytic curves never touch a run: drawn over
  `built_parameters(spec, summary)` (also after a cancelled sweep), Sweep tab overlays over `parameters_with(spec,
  parameter, value)`. Snapshots hold the mid-height cut, so validate refuses a line curve's plane + snapshot; a 3D
  segment's profile is stored per snapshot (`Snapshot.segments`, results.npz 'snapshot/<i>/segment/<name>/'; parity
  sectors keep half profiles and mean them for linear signed fields in the combined snapshot). `sweep_value_curve` /
  `sweep_overlay_xy` are the one rule of GUI Sweep overlays and api sweep figures. Result keys 'physics.feature' follow
  `rename_physics` / `ModelSpec.rename_result_key`; a deleted feature leaves a dangling key (validation error).
- Derived curves (`DerivedCurveSpec`, `analytic.derived_curve` / `derived_curve_plots`): expression names = parameters +
  input aliases (shadowing) + x; `rename_parameter` skips every name that shadows parameters
  (`ModelSpec._parameter_shadows`) and refuses a new name one of them would shadow; `rename_derived_curve` follows the
  derived-value and estimation sources. Consumers pass `fetch(key)`: api `Results.derived_curves()` (host cache, else
  live; summary.json derived_curves with `curve_error` / `y_reference`, read by selftest_examples as 'derived_curve'
  entries; figures/derived_<plot>.png) and the GUI tab (worker). Final results only: no per-sweep-value curves.
- Curve calculus: `gradient` / `cumtrapz` (`analytic.CURVE_FUNCTION_ARITY`) are not FUNCTION_NAMES (not reserved, not
  callable in field expressions): every compile of a derived-curve expression passes them as `functions=` (validate,
  `curve_values`, GUI `FormContext(functions=)`, spec `_expression_names` for renames).
- Derived values over a source: `fit_quantities` is the one rule for `fit` choices (validate, run, GUI): a new fit adds
  its names to FIT_QUANTITIES / PEAK_FIT_QUANTITIES only. A source without '.' is a derived curve (`is_curve_source`),
  built in the ordinary derived-value step (`Session.derived_value`; the GUI passes its document's curves as
  `curves=`), so sweeps get it per value.
- Derived values read the values of the derived values listed BEFORE them (`earlier=`: Session.derived_values, the GUI
  and the derived-series recorder pass them; a series keeps only values whose references are in the series too,
  `session.earlier_derived_names`); validate errors for a later name or one shadowed by a parameter / field / scalar /
  reserved name. Operation 'expression' (source None) is branched in `Session.derived_value` before `evaluate`, so
  every consumer (summaries, sweeps, series, references, parity-sector recomputes) takes it unchanged. Figures of merit
  (stored energy, Purcell, n_eff, perturbative shift) are recipes over the operations (describe
  DerivedValueSpec.expression), no own code.
- Interface selections (`interface_sampling.py`; SelectionSpec kind 'interface' + side, 2D only): a derived value whose
  selection is exactly one interface selection reduces one-sided outline samples (INTERFACE_VARIABLES shadow
  parameters; no mirror images). Line curves with `AnalyticCurveSpec.interface` sample host FieldData through
  `interface_line_profile` (api and GUI alike; polar: live / GUI read `field(native=True)`, saved results and snapshots
  the image); `projection` reads the x component and its `partner_component` (validate checks it is an output field).
  A side thinner than ~4 cells has no usable samples (error).
- Curve errors (summary.json curve_error, api `_curve_errors`) all go through analytic: `feature_curve_samples` uses
  the `_axis_variables` of `feature_curve` (a new curve target kind / series rule updates both); line curves compare at
  the centres of the sampled cells (`line_cell_curve_values`: a half-cell offset would cost ~pi dx / lambda of a pulse
  peak), trilinear 3D segments at the sample points. A line ending on a material interface or void edge still samples
  one cell of either side, so example lines start a fraction of a cell off it.
- `selftests_analytic.py` (CPU, registered by study.register) checks calculus, fits, windows and far-field wrap on
  synthetic results: keep it in step with analytic.
- `selftest_examples.py` (CPU, no api import: api passes summary dicts): `tolerance` on DerivedValueSpec /
  AnalyticCurveSpec is read only there (validate: finite >= 0); sweep_not_converging sees a refinement sweep only when
  the swept parameter changes a used mesh's dx (auto: points_per_wavelength), not MeshSpec.refinements.
- `selftest_cache.py` (no api import; root file "Selftest reuse"; unit rules: module docstring,
  `DependencyTrace.units`). Gap: a plain class used only through attributes of an instance built outside the trace.
  The case profilers call `code_units` on cProfile's code objects with a path filter and without the MODULE_UNIT
  references and the second hop (tools_profilers/time_profilers sub file "Reuse"): keep its keyword defaults the
  selftest's. run_all_profilers loads the file alone (no open_sol import): no package import at its module level. The
  parsed-units store is keyed by the file's sha1, so an edit to the unit rules lives in this file or changes
  `_units_store_key`. `wall_times` feed the progress estimate: api.selftest / selftest_examples decide reuse for every
  phase up front (`api._SelftestPlan`; --all plans the examples before the first case), so a new run path goes through
  the plan's start / finish or the numbering and estimates drift.
- Selftest subsets and reports: api `selftest(heavy=False | True | None)`; `--all` = both, then the examples once
  (each on its own backend; a 'gpu' one is `skipped_backend` without a usable GPU); named cases run whatever their
  flag; built-ins run in every unnamed subset, in a named run only when named. Files: `api.SELFTEST_REPORT_FILES`
  (`--all` writes light and heavy, split by case) and `selftest_report_analytic.json`; a run rewrites only its own,
  through `api._write_report_keeping_timings` (the old status / wall_time per case go to `<stem>_prev.json`, one run
  back), so run times compare across branches; a reused case keeps its old wall_time, so compare `--reset` runs.
- CLI time split ('build / study / output / other'): 'other' = the check's Session reads plus per-run plumbing
  (release_session's `gc.collect()`, validation, estimates, run directory). A large 'other' on few runs is the check's
  reads, usually tetrahedron point location: check code sampling its own lattice per component re-pays it; read
  through the Session or reuse one `locate` result.
- `SelfTestCase.backends`: without a usable GPU the check gets the cpu runs only (a cpu-vs-gpu check must cope); with
  none left the variants still judge the case (record `variants_only`, report warning). A check({}) case using cupy
  declares ('gpu',); one comparing both backends itself declares ('cpu', 'gpu') and skips its gpu half by
  `gpu_available()`.
- Timing: a study of several runs (sweep values, refinement levels, parity sectors, estimation points) sums ticks /
  study_seconds and merges time_check through `study.summed_runs` (a new multi-run loop uses it too; `solvers` stays
  the last run's), and `run_study` sums builds into `build_seconds_total`, else the selftest split counts only the
  last run. Record fields (timing, vram, counts) and the CLI deviation line: api `_run_timing` / `_run_vram` /
  `_run_counts` / `_set_worst_deviations`, `__main__._deviations_text`.
- Schema: Param min/max apply only to values evaluating over the parameters (field- / state-dependent floats become
  Expressions, `schema._number`). The builder range-checks material properties instead (`eval_field(bounds_param=)`
  under the region, one sync); a DynamicMaterialUpdate keeps device running extremes, read at the moments of
  `builder._check_dynamic_range` and once more at the study end (`BuildContext.check_dynamic_ranges`); other
  field-dependent params are checked only for finite values.
- Param kind 'table' as a material property writes no array: the type's `create` reads `ctx.material_regions` (2D
  `region.mask` until `release_caches`) and resolves the rows with `resolve_values` (magnetostatics `bh_curve`).
- 3D prisms (spec v2; root file "3D models"; optional z params of the 2D kinds: `geometry.Z_PARAM_NAMES`, axis /
  taper in the kind help): every per-z consumer goes through `geometry._section(shape, z)` -> (key, plain parts) and
  the footprint through `_footprint` (`is_plain` = neither), so a new 3D kind or option extends `_section` (and
  `surface_faces`, the 3D preview surface), never a consumer; vertical cuts (GUI side views) likewise through
  `vertical_section` / `section_mask` / `material_preview(cut=)`. A taper makes every z cell its own run
  (`cross_section_runs`). Mesh snap faces come only from `axis_parallel_faces`: curved and tilted walls stay
  staircased.
- Polar / spherical domains (`DomainSpec.coordinates`; core sub file "Coordinate metric"): `geometry.DomainMap`
  (`domain_map(spec, parameters)`; also on ValidationContext, MeshEstimate (None on Cartesian), BuildContext, Session)
  is the one map between the drawing plane and the logical cells. Masks test the mapped centres over the whole window
  (`rasterize_mapped` in the mapped build_masks / field_access classes; validate: `_mapped_window_centres`), never
  `rasterize_window` on logical centres. `material_preview` is index 0 outside the sector, so preview shares /
  statistics go by `DomainMap.contains` (as `definition_helpers.conducting_fraction`). `BuildContext.point` maps;
  `line` / `rect` raise there. `_mapped_plan` widths are float32-rounded like the Domain's cells (edges[-1] off size
  by ~1e-8 relative): compare to the float32 widths, not size.
- Cylindrical (r, phi, z) sectors ('polar' + size_z, heat only): the DomainMap is the (r, phi) plane, z unmapped, on a
  padding-free polar Domain3D. 3D xy masks are full (nx, ny) mapped rasters per cross_section_key
  (`build_masks.SectorMasks`), never `rasterize_box` / `box_window` on the logical host centres; x, y values of 3D
  builds come from `BuildContext.plane_coordinates` (a new 3D x, y consumer uses it too). Session cuts at constant z
  only (validate refuses x / y stored_cuts and 3D segment line curves). Not yet: GUI (canvas / scene_3d / side views
  draw the logical box), meridional cuts at fixed phi.
- 1D models (root file "1D models"): `DomainMap(line=True)` on every kind (`_line_domain_map`), so the polar path
  carries them. Objects select the cells whose centres lie on the drawing's x axis: circles about the origin = shells /
  layers, rectangles across y = 0. Not yet: GUI (forms, canvas, Add physics: `schema.type_spaces` ignores 1, GUI
  solver_misfit calls pass no grid), 1D EM (slab_modes), point-source units in 1D (heat P is per the 1D total: W/m^2,
  W/m or W; its help says W/m).
- Arrays (kind 'array'; `geometry._array_copies`; root file "Arrays, coordinates"): copy values = base + i pitch
  (exact float sums, so written-out copies mesh identically); the resolved array's values['inputs'] = the copies. So
  code walking composites treats an array as a union (`geometry.UNION_KINDS`; `composite_inputs` of a resolved array =
  its copies, of spec params = [base]), and code listing spec objects from the resolved dict skips copies (canvas
  picking / consumed, scene_3d tops, `_section_preview`). Estimate rasters group distance sources by size
  (`meshing.SizedUnions`), so curved copies share one transform. Not yet: `smallest_side` takes the array's whole box
  (auto sizes); the fem mesher's PSLG crossing search over 1e4 curves is untimed.
- Rasterisation precision: `material_preview` and the validate coverage warnings use float64 centres on the CPU,
  build masks float32 clamped centres, so cells on an exact boundary can differ (likewise a prism end exactly on a
  cell centre: float32 in `_section`, float64 in `contains_points_3d`; tests comparing them keep ends off centres).
  Grid masks go through `geometry.rasterize_window` (bounding-box pruning: a new composite kind keeps it exact), never
  `rasterize` on full meshgrids; `rasterize` stays for point probes.
- Per-physics meshes (root file "Meshes per physics"): meshing entries take a mesh name (None = `default_mesh_name`,
  the first used mesh) and the evaluated {name: float} `parameters` (`evaluate_parameters(spec.parameters)`), never
  the spec's: self-tests pass `{}` only for parameter-free specs. Requirements, auto sizes, uniform-feature fallback
  and padding come from the physics on that mesh only. Auto mode: a set dx is the smallest cell size (MESH_PARAMS
  help); the dataclass default 20e-9 counts as set, so an auto MeshSpec / JSON meant to follow the rule alone sets dx
  None / null explicitly.
- Per-mesh build (`builder._build_on_gpu`, `_SharedBuild`): the first mesh's context holds the others
  (`other_meshes`; the shared dict is weak), so a closed Session frees them without gc: a new holder of contexts keeps
  that direction. A mesh's material arrays are `meshing.mesh_material_types` (the build and `_mesh_memory` alike).
- Per-mesh Session (`Session` docstring; per kind: `field_access*.py`): a mesh view (`_mesh_view`) is a cached
  shallow copy sharing everything but domain / context / access; a new Session read of a physics' fields enters
  through `_physics_view(physics)` (expressions: `_evaluation_view`), so it acts on that physics' mesh. Caches that
  span meshes live on the parent (`_window_transfers`, freed by close). Warm starts compare every mesh
  (`study._mesh_key`).
- Triangle meshes (MeshSpec kind 'triangles'): `meshing.triangle_sizing` is the one size rule of estimate and mesher;
  every size is floored at h_min (default h_max / 100), so a finer refinement needs an explicit h_min. The raster
  estimate (`_triangle_estimate`) sets memory, run time and MAX_MESH_CELLS: recalibrate TRIANGLE_FILL /
  TRIANGLE_EDGE_FILL against selftest mesh_triangle_estimate when the mesher's refinement rule changes. Element meshes
  (triangles and tetrahedra) go through `meshing.element_mesh`, reused per mesh name while `element_mesh_key` matches.
  Reuse is on in the GUI's BackendWorker, in `run_study` around sweeps and parity sectors and in `run_estimation`; off
  elsewhere (single api / CLI runs, between selftest cases: a mesh made by another case would hide the mesher from the
  selftest cache's tracer).
- Meshing (tensor): equal boundary configs of the physics on a mesh merge, different ones raise (`_requirements`).
  `MeshEstimate.uniform` judges the window widths, `ValidationContext.mesh_uniform` the full grid. nonuniform / auto
  fall back to `Domain.uniform` (`note` says why) when no refinement, band or custom axis is finer than the base size,
  and to the finest size when an enabled feature `requires_uniform_mesh`. Tensor options (core sub file for
  MeshConfig): `MeshSpec.axes` are checked only by `_axis_plans`, so validate reports their errors as 'mesh estimate
  failed'; an axis without band ends or custom edges is one segment whose whole-axis rescale moves a band by up to half
  an outer cell. Auto wavelength rule: the index is taken at wavelength_min only (`_material_index`), so metals mesh by
  skin depth (gold ~11 at 1.55 um).
- Triangle meshes (triangulate): `triangle_meshing.triangulate` (module header; mesh data and mesher rules: fem sub
  file). `mesher_size_field` is the mesher's form of `meshing.triangle_size_field` (the estimate's raster): a size
  rule changes in both. Element tags use the build-mask rule (`element_masks` -> `contained`, bit-identical to
  rasterize), never a second point test. `mirror_x` / `mirror_y` mesh the half window and reflect it
  (`_mirrored_parts`); the estimate takes the denser of each pixel and its image; asymmetric material objects only
  warn (`mirror_mismatch`).
- Tetrahedron meshes (MeshSpec kind 'tetrahedra'): sizes `meshing.tetrahedron_sizing` (the triangle rules through
  `_element_sizing`), mesher `tetrahedral_meshing.tetrahedralize` (gmsh solids, sized regions / surfaces, Kuhn split:
  its module header), estimate `_tetrahedron_estimate`. Every gmsh size field has its twin in
  `tetrahedron_element_density` (estimate and mesh must agree: a new field adds both); gmsh counts are
  TETRAHEDRON_FILL (uncalibrated) x a size integral, only builtin counts are exact. `material_elements` (elements
  each material paints, the builder's rule) lets a physics whose dofs live on part of the mesh scale its estimate
  (solid_mechanics_3d: stiff materials). Builder object masks and the mesh's `object_faces` both come from
  `element_membership`: a tag rule changes there only. A non-EM physics pads the mesh with any boundary_config
  carrying npml_sides (`pml3d.PMLPadding`). Session reads: `field_access_tetrahedra` (its refusals raise).
- Builtin tetrahedra: `TetrahedronSizing.axis_edges` (`builtin_axis_edges` = `graded_window_edges` + the PML shell)
  is the one grid of mesher and estimate (`sizing_axis_edges`), so the estimate is exact; its repr is
  `axis_edges_digest` (swept: `layer_edges_digest`), so the element-mesh key follows every grading input. Snap merge
  (core `snap_positions`): an unrefined film thinner than h_max / 4 keeps one face only (refine it to resolve both;
  a swept mesh then tags it per element). A remainder-rounded segment may hold cells up to ~1.5x its target.
- Swept prism layers (MeshSpec.sweep_axis 'z'): `model/swept_meshing.py` (rules in its header, user rules in the
  sweep_axis help, refusals `swept_mesh_problem` / `swept_refinement_problem`), fem `prism_layers`. The footprint
  goes through `triangle_meshing._plain_parts(..., size_field=)`: a change to the 2D mesher path reaches swept meshes.
- Memory estimate (`meshing.estimate_memory`; parts per mesh: `_mesh_memory`): MESH_ESTIMATE_ERRORS are model errors
  reported at their node (check_memory then stays silent), any other exception an internal-error Issue at
  ('meshes',). Three budget checks: validate `check_memory` (`memory_verdict`: issues at ('mesh', the largest part);
  80 % warnings, the 8 GiB host warning without a host budget), `_sweep_value_problem` (each sweep value, at
  ('study',)) and study_adaptation `_memory_problem` (each next level, stops the refinement). All judge
  `fitted_memory_estimate` (an over-budget study default resolves without its study: solvers sub file "Model path")
  through `backend_memory` at `schema.memory_check_level`: a new budget check does the same. A refusal is final only
  on built element meshes (`meshing.awaits_built_mesh`): build and api.check (`refused_on_estimated_counts`) validate
  without budgets first, then mesh (`built_mesh_estimates`, kept for the build's domain stage) and judge the built
  counts and `estimate.dissection` (the counted factor fill); per sweep value and refinement level the check stays on
  the size-based estimate. Budgets: builder = `gpu_memory.budget_bytes()` minus the pool bytes of other live sessions
  and `host_memory.available_bytes()`; GUI = the worker's last MemoryStatus; CLI check / show = host RAM; api.check
  only its arguments.
- Memory estimate parts: `core.memory_estimate.MemoryEstimate` (its docstring and `device_peak` / `external_peak`);
  estimator rules: root "Physics definitions" ("Estimates mirror the solver"). Session reads cache no pixel-sized
  device arrays (a polar display image is gathered on the host), so they book nothing; post-solve field updates do
  (DiffusionSystem flux: `diffusion_system_estimate(flux=, dissipation=)`).
- Build: every step runs under `builder._issue_at(path)`, so an exception becomes ModelError with one Issue at that
  step's node (a solver: ('solver', node), on Automatic its physics; `runner.prepare`: ('model',)), and under
  `ledger.track(owner)` (nesting raises; untracked bytes land in 'other (untracked)'): a new step gets both.
- Build memory: the ledger's `peak_bytes` (pool reserved minus used at the start) relies on `build_session` emptying
  the pool first and no `free_all_blocks` during the build. The record (`Session.build_memory`, BuildFailed /
  StudyFailed `.build_memory`, summary.json) is CPU data only, so it travels with an exception without its frames.
  Build, run and outside-pool records are judged by `builder.estimate_verdict` ('over' / 'unaccounted'). Free pool
  blocks stranded beside earlier open Results hide temporaries from the high-water marks (`estimate_gap` takes the
  nearest point of [measured, measured + unseen]): a run beside open Results reads low (adaptation_tensor_strip_heater
  second run: build peak -61 %, alone -12 %), so judge estimators on a fresh process.
- Session lifetime: `Session.close()` / `study.release_session` (docstrings); the Simulation it clears is dead. The
  leak warning compares process-wide pool counters, so a later live session reads as a leak (api `Results.close`
  skips the check then).
- GPU OOM = `gpu_memory.is_out_of_memory` (cupy pool or a CUDA library allocation failure; host MemoryError is
  ordinary), never a type or text match of its own. Build: Issue at the failing step (outside one ('model',)), memory
  released before BuildFailed / StudyFailed (`out_of_memory=True`); run_study closes the session that ran out. A
  handler that turns failures into NaN / warnings re-raises it (study series, live fields), except
  `Session.derived_values` (NaN and a warning: the run's results stay).
- Conformal boundaries (root "Physics definitions"; honouring types, cut geometry: core sub file "Cut geometry"):
  one host `boundary_locator.geometry_locator(ctx)` per build, also during create (`BuildContext.conformal_locator_of`:
  2D FDTD conformal pole weights). It tests float64 points by the build-mask
  rules (`geometry.rasterize`; 3D `geometry.contains_points_3d`, where a new 3D kind or prism option goes), so a
  centre within float32 rounding of an edge may disagree with the build mask: callers fall back to the staircase
  link. Conformal self-test cases run with conformal_boundaries on only and tolerances below the staircase errors, so
  a silent staircase fallback fails: keep it so in new cases.
- Subpixel smoothing (averaging and ownership: em sub file "Subpixel smoothing"):
  `subpixel_geometry.interface_geometry` runs once per mesh in a 'subpixel geometry' ledger stage after 'materials'
  (issues at ('solvers',)). `subpixel_memory_estimate` is rough (bounding-box perimeters), within the ledger
  tolerance only because the arrays are small. The build is backend neutral (`array_module_of` the coarse index / `domain.xp`, host work by `to_host`):
  a new array op there must exist in numpy and cupy alike (no `cp.` names, no cupy-only kernels).
- Static materials: `geometry.material_paint_order` / `paint_material_index` is the one painting order (later
  materials win whole cells, a selection containing 'all' only fills uncovered cells) for builder, preview,
  validation, meshing estimates and the symmetry check: never a local order. Only properties the material sets are
  written (unset arrays keep the MaterialGrid defaults); which arrays a set property writes:
  `builder.merged_property_params` / `owned_property_writes`, shared with `Simulation.add_material` (its script
  differences are in its docstring; script names and defaults follow the model: `waveform`, EM point `current`
  1e-3 A). Code asking which property writes an array (eps_xx / _yy / _zz win over eps_r) uses `schema.array_owner`.
- An `optional_material_schema` array exists only when triggered (schema comment): a solver reading one checks
  `materials.property_names()` first.
- A material value of state names or t becomes a DynamicMaterialUpdate '<material>.<property>'
  (`builder._dynamic_update`), ahead of the time updates and user couplings. CPU code asking "is this property
  dynamic" (estimates, refactorisation choices) uses `definition_helpers.dynamic_material_values` /
  `material_follows_state`, never its own name test.
- Feature values of t (Param `time_dependent`): the apply hands them to `BuildContext.follow_time` (docstring: when
  the setter runs; `definition_helpers.uses_time` tells the apply). A setter is cheap and incremental (a source adds
  the change to its applied array; a face value updates in place: `DiffusionPhysicsBase.set_face_value`,
  `FluidPhysics.set_face_value`), never re-allocates per step and is not re-called for an equal value.
- resonance_monitor: a new time-stepped physics with a point probe declares a `ResonanceProbe`
  (`definition_resonance`) instead of copying the feature (numerics: core sub file "Resonance finder").
- BuildContext masks: the cached ones (`object_window_mask`, `object_box_mask`, `perimeter_mask` / `_weights`) are
  shared across features: never write into them (cupy has no read-only flag); `domain_mask` is fresh. Object-perimeter
  fluxes (heat_flux, convection, radiation, current_density, ...) take
  `definition_helpers.perimeter_flux_weights(ctx, solver, objects)` (adds the (r, z) / polar face scales), never
  `ctx.perimeter_weights` alone; `geometry.perimeter_weights` weights each exposed face by |normal| (curved edges
  within ~2 %, rectangles exact; the raw staircase is +27 % on circles).
- Session reads: Session checks arguments, picks the mesh view and delegates to `session.access` (root "Meshes per
  physics"); a cross-mesh read asks the SOURCE view's access (`values_on_window(target, ...)`, `values_at(points)`).
- 3D field access: every 3D read (cuts, `segment_profile`, `volume`, snapshot cuts) slices planes from
  `TensorAccess._plane_source` (the renderer's `device_plane`, else ONE `device_field` volume per read, shared inside
  `Session.shared_plane_reads`), never a device_field per plane; host volumes come one z plane at a time. A closed /
  loaded Results has no `field(plane=)`, only `snapshot_field(..., plane=)` (nearest `StudySpec.stored_cuts` plane).
- `FieldInfo.requires_study` fields stay in `output_fields`: code listing readable fields filters through
  `Session.available_fields`; reading one outside its study raises ValueError (derived values: NaN + warning).
- Validation never raises (a callback exception becomes an 'internal error in <owner>' Issue). A type's validate runs
  after the generic checks (unknown keys, visible param values, 'ref' / 'names' targets, selection kind / object_kinds
  / counts, `requires_uniform_mesh`): do not repeat them; ValidationContext helpers return None / [] for problems
  reported elsewhere. Model-wide material checks run once (`definition_helpers.reports_model_checks`; EM: the first
  physics of family 'em'); domain size is reported at ('geometry',). Messages name the offending value (`short_repr`)
  and the valid alternatives. Checks see the base parameter values only; `_check_sweep` re-runs parameters, geometry,
  domain size, `estimate_mesh` and (with a budget) `estimate_memory` per sweep value.
- Worker (`worker.py`; GUI side: gui sub file): the only thread touching cupy or a Session in the GUI process; events
  carry CPU data; Progress / LiveFrame / LiveResults / MemoryStatus are `take_latest` slots, the rest queued. Each job
  ends with exactly one final event (Built / BuildFailed, StudyFinished / StudyFailed; a run may first emit Built /
  BuildFailed per sweep value). LiveResults timeseries above 4000 samples are min / max per bin
  (`decimation_stride` > 1): never FFT them. A `call(fn)` made for an older session (`Built.session_serial`) fails.
  A new study sets its summary `tick_unit` (default 'step').
- Progress stages: every study run creates one `study._StageProgress(emit, names)` (only the stages this run has) and
  hands `stage_emit(name)` to each stage as its `emit`; a new study type or stage does the same, else the bar restarts
  at 0 per stage. A stage emits its own fraction 0..1 (solve counts: `_count_progress`; fixed points:
  `_fixed_point_fraction`, never decreasing); messages name the physics (`_physics_text`).
- `run_study` (docstring: never raises, which session it returns): a sweep closes the previous value's session
  first, a failed last value its own (`session_live` False); a non-OOM failure of a single run keeps its session.
  Failed sweep values are listed in the sweep table `failed` (status stays 'completed'; api 'partial', CLI exit 1);
  a feature result a value cannot produce is left out with one warning per key (`_feature_results`), and a stationary
  loop that does not converge stays 'completed' (`fixed_point`, sweep table `converged`). Loops that stop on a cancel
  between items test `control.checkpoint_cancelled(service)`.
- Stationary fixed point (`solve_stationary`, also the background step of frequency_domain / mode / eigen studies;
  rules in its docstring and the fixed-point param helps): with a loop-back coupling the loop never converges in
  sweep 1 (`_has_loop_back`: that sweep solved with slots from before the solve, so a stale fixed point showed a
  change of ~0: slug_flow_graetz returned the previous value's T every second sweep value); a non-finite change raises
  (NaN compares False and would run all max_iterations). Warm starts: `solver_hooks.WarmStartSolver`.
- Reused sweep builds (`SweepSpec.reuse_build`, `sweep_reuse.py`): `reuse_issues(spec, registry)` = what refuses it
  (time_dependent / optimization studies, parity_sectors, adaptation, and every spec value reaching a live parameter =
  the swept one + its dependents (`live_parameter_names`), except feature Params and physics settings flagged
  `follows_parameters` with parameter-only values (no x / y / z / t; a setting of a type with `shortest_wavelength` on
  an auto mesh without wavelength_min never follows: the mesh is sized by it), material values `_material_follows`
  accepts, derived values / curves / references and study settings); validate
  `_check_sweep` reports them at their node paths, `_run_sweep` sends a refused spec through `_fail_sweep_setup`. The
  first build gets `live_parameters` (build_session -> builder.build keyword; a custom build callback must accept and
  pass `**options`, as api's `_BuildRecorder`); a flagged param's apply (a setting's: the create) sets its build value,
  then calls `BuildContext.follow_parameters` (mechanics `follow_load` / `follow_held_values`), which registers a
  `ParameterFollower` only for a live raw value on a `solver_hooks.RerunSolver`. Material values: a uniform value of
  parameters, no state field, no 3D tensor mesh, and every type reading the property non-EM with the property Param
  `state_dependent` (the "materials_changed of an array it reads re-solves" contract, coupling sub file); EM types
  are refused because index references, dispersive backgrounds and subpixel averages come from the build's eps.
  `builder._follow_material` then repaints the owned arrays under the region mask (index-painted regions never
  overlap; the follower keeps its own mask reference) and calls `materials_changed` of every solver on that mesh.
  When every physics is a RerunSolver
  (`non_rerun_physics`, else an info line and one build per value) the session is kept: per later value
  `Session.set_sweep_value` (spec parameter and parameters dict changed IN PLACE: contexts and mesh views share them;
  followers re-set changed values incrementally) + `reset_run_records` (snapshots cleared in place, runner t / tick 0,
  every `reset_records()`), then `_run_built`; kept snapshots are taken before that reset; a run failure drops the
  kept session; table `builds` counts builds. Followers keep host data only (a material follower: also its region's
  bool mask, as a dynamic material update). Not range-checked per value: flag only unbounded params, or check the
  bounds in the setter (EME wavelength, AC frequency > 0, doping >= 0: a ValueError fails that value, the next one
  builds anew); material values are never range-checked per value. `estimate_overheads` prices the one build, `estimate_time` still every value's run.
- Study catalogue: `time_dependent` (em_fdfd_* too: stroboscopic re-solves every k steps beside a time-stepped physics,
  never beside em_fdtd_*), `stationary`, `frequency_domain`, `mode_analysis`, `eigenfrequency` (em_fdfd_* incl.
  em_fdfd_3d, solid_mechanics, solid_mechanics_3d, membrane), `frequency_response` (solid_mechanics,
  solid_mechanics_3d (tetrahedra), membrane, pressure_acoustics: frequencies in Hz; one snapshot per frequency, real
  parts, u_abs = |u|), `nonlinear_frequency_domain` (the last five with background study 'stationary'; it alone has
  `StudyType.solution_feedback`), `optimization` (em_fdfd_* and 2D em_eme only, no background, no couplings),
  `thermal_noise` (heat + em_modes, background 'stationary'; heat is never CG-solved there and needs no anchor),
  `optomechanics` (background 'stationary': solid_mechanics with an optomechanical_coupling feature + the em_modes /
  em_fdfd_ez / _hz physics it names (solid_mechanics_3d on tetrahedra: em_fdfd_bor optics); validation refuses
  couplings / material readers of every FDFD physics there, also one no feature names (runs as background);
  `em.definition_common.eigen_solved` tells the FDFD estimator / resonance_table which one is eigen-solved; solid field
  om_integrand exists in this study only).
- StudySpec.solved_physics ([] = the Protocol decides): the physics the study solves itself; the others, also of a
  type listing the study, run as stationary background physics (allowed as coupling sources into the listed ones).
  Only in validate._SOLVED_PHYSICS_STUDIES (frequency_domain, mode_analysis, eigenfrequency, frequency_response,
  optomechanics; elsewhere an error). Run side: `study._selected_physics(session)` into `_solver_roles(selected=)` (a
  listed physics without the Protocol raises; frequency_domain filters its pulsed physics too) and optomechanics
  `_roles(selected)`; validate `_check_solved_physics` and `_check_background_study` count only the listed ones as
  solved (`_solved_by_study`). A type definition whose rules depend on the study asks `spec.study.solves(name)`
  (mechanics `definition_common.background_context`). `ModelSpec.rename_physics` / `remove_physics` update the list
  (removing the last listed one empties it: every physics is solved again).
- Optional PrestressSolver hook (`solver_hooks`, stress stiffening): eigenfrequency / optomechanics call
  `study._solve_static_states(runner, solved, True, emit)` right after their background stage, frequency_response with
  False (its loads are the drive).
- frequency_domain and mode_analysis (`StudyType.background_study` 'stationary'): validate accepts a physics
  supporting the study or its background_study, needs >= 1 supporting the study itself, and rejects an enabled
  coupling or material property reading such a physics (`_check_background_study`, remedy
  `StudyType.feedback_alternative`; not with `coupled_points`, below). Runs share `_solver_roles(physics, protocol,
  ...)` (solved = solvers satisfying the Protocol; the rest must have dt_limit inf) and `_background_stage` (refuse
  solution readers, prepare, recorder, live feed, stages, background `solve_stationary`,
  `_apply_background_couplings`; eigenfrequency and thermal_noise too: a new background study uses it). mode_analysis
  calls `solve_modes()` per mode physics through `_numbered_snapshots` (shared with eigenfrequency; cancel only between
  solves): one snapshot per mode number (a physics with fewer modes stores NaN fields there), re-selects
  `mode_index`, summary `n_modes`. `StudyType.validate(vctx, values)` runs after the generic param checks
  (`definition_helpers.wavelength_range_issues`, shared with EM spectral monitors). The run raises for a background
  physics that steps in time unless it has `attach_harmonic` (em_fdtd_*: `stepped` in `_solver_roles`, run by
  `_run_harmonic_pulse`: its own tick loop over those solvers only, advancing `runner.t / tick`, never
  `runner.advance`, which would step the background too); wavelengths from `core.wavelengths.resolved_wavelengths`
  (list, else linspace, n = 1 the centre); no `runner.flush`. `_solve_wavelengths` hands the omegas (the same floats
  solve_harmonic gets) to `plan_harmonic_frequencies` before the loop and `()` after, also on cancel; the coupled scan
  never plans. Default stored fields = every primary field plus the `<field>_im` partner of a `<field>_re` primary
  (`_stored_field_keys`; frequency_domain takes a renderer's `harmonic_primary`), so wavelength snapshots also store
  the background fields.
- Coupled wavelength scan (`StudyType.coupled_points`, frequency_domain only): validate turns the reader errors into one
  info when every physics lists the background study (else errors name the physics that cannot join: em_modes, FDTD);
  the run (`_run_coupled_wavelengths`, chosen when a runner coupling or DMU reads a harmonic / pulsed physics) skips
  `_background_stage`: per wavelength `omega` (settable property: FDFD, BOR, EME) of every harmonic physics, then
  `solve_stationary` over every physics (monitors keep one value per omega), one snapshot; nothing is reset between
  wavelengths (list order = scan direction). Summary `fixed_point` over all + `wavelength_fixed_points`;
  `log_solves=False` after the first wavelength.
- Derived series: `_SnapshotRecorder(series_axis='time' | 'wavelength')` (time_dependent, frequency_domain) evaluates
  every source-free derived value on the live state at each `record` (also once full; a source result would be
  partial mid-run): summary `derived_series` = an 'xy' dict of plain lists (summary.json, not results.npz), None below
  2 points or with parity sectors. Sweeps: table `derived_series` per value, `snapshots` per value with
  `SweepSpec.keep_fields` (`_KeptSweepFields`: host Snapshots taken before the session closes, 1 GB total, then finals
  only), and `SweepValueDone(index, count, parameter, value, sweep_table_copy(table))` after every value (also failed;
  study_sectors likewise). `selftests_sweep` builds on coupling/selftests' absorbing strip (imported inside its
  specs): keep both in step.
- Study modules `study_eigenfrequency.py`, `study_optimization.py`, `study_thermal_noise.py`, `study_nonlinear.py`,
  `study_optomechanics.py`, `study_frequency_response.py` (and study_sectors / study_adaptation / study_estimation)
  import study's private helpers: never import them from study at module level (registered at the end of
  `study.register`).
- eigenfrequency (`study_eigenfrequency.py`) follows mode_analysis: `solve_eigenfrequencies(n_resonances)` ->
  EigenSolve (`summary()` and `info['warnings']` logged), `select_resonance(k)`, one snapshot per resonance, resonance
  1 re-selected, summary `n_resonances`, tick_unit 'resonance' (Progress 'eigen solve'); validate errors without
  scipy. The target is each physics' wavelength setting (no study wavelength); solid_mechanics uses target_frequency
  and returns a duck-typed `modal.ModalSolve` (`text()` 'f = ... Hz'): a new eigen-solved type only needs that
  protocol. `search` 'targets' / 'band' calls the `MultiShiftEigenSolver` hooks (`solve_eigenfrequencies_targets(n,
  targets_hz)` / `_band(f_lo, f_hi, n_per_slice, max_resonances)`, both in frequency order); validate refuses a
  type whose `PhysicsType.eigen_searches` lacks the search (solid definitions also refuse tensor 3D meshes and a
  piezoelectric band). time_counts: eigenpairs per shift, eigen_shifts, eigen_count_factorizations / eigen_band,
  resonances; recipes read them through `definition_helpers.eigen_search_counts` (absent = one shift).
- optimization (`study_optimization.py`; no background: every physics em_fdfd_*, `runner.couplings` empty; validate
  also errors without scipy, without a design_region or adjoint_objective, and for an enabled coupling). Each physics
  solves at its own wavelength setting. `_DesignProblem.evaluate(densities)` writes every region, calls
  `materials_changed` on every solver, `solve_harmonic(own omega)` + `eps_sensitivity()` per physics (a solver with
  `solve_design()`, em_fdfd_nonlinear, is solved by it instead), sums J and the chained gradient; a repeated point
  returns cached values, so the L-BFGS-B callback normally costs nothing. L-BFGS-B minimises -J / |J0| (bounds [0, 1],
  maxfun 5 x max_iterations, gtol 0); Cancelled from `control.checkpoint` unwinds through scipy. Snapshots: iteration
  0 + each accepted iteration, plus 'final: J = ...' when the held state differs from the last snapshot (or the
  recorder is full). Summary `optimization` {iterations, evaluations, design_cells, objective_initial / _final,
  continuation_stages, message}; ticks = iterations. Beta continuation (n > 1 stages): stage s restarts L-BFGS-B from
  the previous result with beta = feature value x beta_growth^(s - n); a beta change invalidates the cached point
  (`problem.last = None`). Default stored fields add `<physics>.design_density` (`StudyType.default_stored_fields(spec)`,
  used only when StudySpec.stored_fields is []).
- time_dependent snapshots: output_interval > 0 wins, else n_snapshots (default 20, GUI and `api.run`) evenly from
  t = 0 to the final state (interval duration / (n - 1)). A study type whose snapshots are an extra, not its result,
  lists the settings that store only the final state in `StudyType.final_state_settings` (keys checked against its
  params): `api.run(snapshots=False)` and the self-tests apply them (root sub file "api.run").
- StudySpec.stored_fields [] (the default) stores every `session.available_fields()` entry in each snapshot
  (`_stored_field_keys`: primaries first): a new renderer field must be readable after every study solve its
  `requires_study` allows, else every default run of that type fails at its first snapshot.
- GUI run features (`FeatureType.gui_run_studies`, `schema.with_gui_run_features`): a GUI Compute adds such a
  selection-free result feature (default params, name = the type, numbered when taken) to every physics the study
  solves without one of its type (a disabled one counts), on the feature's grids; headless runs never do. The tuple
  must mirror the feature's validate rule, else a GUI Compute fails at a feature the tree does not show. Today:
  mechanics mode_shapes (eigenfrequency / optomechanics, tetrahedra).
- Snapshot solutions (`RunControl.keep_solutions`: GUI worker on, `api.run` / CLI / self-tests off ->
  `session.keep_solutions` in `_run_built`): besides the stored fields each Snapshot keeps per physics in
  `Snapshot.solutions` what every field of it renders from, chosen by `_SnapshotRecorder._solutions`: the study's
  `selections` {physics: (select, key)} first (a SelectedSolution: `_numbered_snapshots` modes / resonances,
  pulsed FDTD harmonic frequencies, nonlinear channels; the study notes each select in `session.live_selections` via
  `study._select`, which the read restores afterwards), else LIVE_SOLUTION on the final record (the state the session
  keeps) and for `recorder.fixed_physics` (background physics, optomechanics optics), else a StoredSolution of a
  `solver_hooks.SolutionStateSolver` (host copies; `materials=True` for `recorder.changing_materials`: coupling /
  DMU targets, optimization all) while they fit a quarter of the free host RAM (one info line, later snapshots none);
  blank physics none. `Session.snapshot_field(index, component, physics, plane)` = the stored field (exact plane),
  else `restored_snapshot` (backup = the solver's own solution_state / the live selection, runner.t = the snapshot's
  time) + `canvas_field`: any field, any 3D cut. Solutions never leave the session: `Session.close`, kept sweep
  snapshots (`_KeptSweepFields.take`), parity-sector merges and adaptation `keep_levels` copies drop them; summary
  `snapshot_solutions` lists the physics per snapshot. A new study that selects a kept solution passes `selections`
  and calls `_select`; one that re-solves a physics after a snapshot must not put it in `fixed_physics`.
- thermal_noise (`study_thermal_noise.py`, background 'stationary'): roles by Protocol, not `_solver_roles` (noise =
  ThermalNoiseSolver = every heat; modes = ModeSolver; the rest background, dt_limit inf), background solve +
  `_apply_background_couplings`, `study._solve_mode_snapshots(..., final_last=False)`, then
  `solve_thermal_noise(runner.physics, frequencies, progress)` per heat (progress per factorisation runs
  `control.checkpoint`); frequencies `study._frequency_list(values, ascending=True)` (list sorted ascending, else linspace / geomspace); one final
  snapshot 'thermal noise'; ticks = factorisations, summary `noise_frequencies`; a band holding < 50 % of a feature's
  relaxing variance logs a warning. An em_fdfd_ez / _hz illumination physics (beside em_modes only in this study) is a
  background physics solved before the mode and noise solves.
- frequency_response (`study_frequency_response.py`, background 'stationary'): `_solver_roles` by
  FrequencyResponseSolver, `reset_frequency_response()` of each, then per frequency of `study._frequency_list(values)`
  (list as given, else linspace / geomspace) every solved physics' `solve_frequency_response(f)` and one snapshot 'f =
  <Hz> Hz' (cancel between frequencies). Setting `snapshots` ('all' / 'last' / 'none') gates only the recorder:
  solver-side per-frequency records (harmonic probes, reaction tables, ports, mode projections) never depend on it, so
  a new per-frequency output belongs in the solver's records, never in snapshots. ticks = frequencies, summary
  `response_frequencies`. `sweep_method` 'adaptive' / 'modal': after the resets `plan_frequency_response(frequencies,
  options)` of every solved physics (ReducedSweepSolver; missing raises) -> summary `frequency_sweep`; validate refuses
  a method missing from `PhysicsType.frequency_sweep_methods` (default ('direct',)). time_counts adds
  `harmonic_factorizations` (direct: frequencies, adaptive: min(frequencies, rom_max_points) + 1, modal: 2); recipes
  read them through `definition_helpers.frequency_sweep_counts` / `reduced_sweep_time_items` /
  `reduced_sweep_memory_items` (never re-derive). options['fields'] = 'all' with snapshots 'all', else 'last'; 'direct'
  passes it too (only stores the policy). After every sweep with a solved frequency (also a cancel) each
  ReducedSweepSolver's `publish_frequency_response()` runs before the 'last' snapshot. Frequencies in Hz, unlike the
  wavelength-based frequency_domain.
- Solution feedback (`StudyType.solution_feedback`, nonlinear_frequency_domain only): a study whose own solve ends in
  ONE operating point may let couplings read its solved physics when they write background physics (validate info;
  solved -> solved and material readers stay errors). A study with many solved states (wavelengths, modes,
  resonances) must not set the flag. Run side (generic in study.py): `_feedback_couplings` picks them (never a DMU),
  `_background_stage(feedback=)` solves the background with `solve_stationary(inputs=solved)` (input physics not
  stepped; their couplings land in schedule[0], applied before every sweep, unrelaxed); after the study's first solve
  `_feedback_rounds(..., solve(emit) -> status, ...)` repeats background solve + `_apply_background_couplings` + solve
  until `_FixedPointMonitor` over background + solved primaries changes by <= tolerance (setting `feedback_rounds`;
  growth stop as the fixed point; one round only when nothing writes a solved physics; no relaxation) -> summary
  `feedback`. The monitor's copies are the run item `meshing._feedback_copy_count` (a change of its members updates
  it). A SlotSum mixing solved and background sources is a feedback coupling: its background contributor then also
  only refreshes before each background sweep.
- nonlinear_frequency_domain (`study_nonlinear.py`, background 'stationary', solution_feedback): `_solver_roles` on
  NonlinearSolver, `_background_stage`, the channel solves (physics/nlfd sub file), feedback rounds, one snapshot per
  channel 'channel <name>: lambda = <nm> nm', summary `n_channels`, `nonlinear_solve`, `feedback`.
- optomechanics (`study_optomechanics.py`, background 'stationary'): roles by feature, not Protocol (mechanical =
  OptomechanicalSolvers with couplings, optical = the physics they name, the rest background with dt_limit inf),
  `_background_stage`, optical stage (`solve_modes()` or `solve_eigenfrequencies(n_optical_resonances)`, the first
  coupling's mode selected for display), then `_numbered_snapshots` over the solids ('mechanical mode k: f = ..., g0 /
  2 pi = ...'; each solve also evaluates every coupling table), summary `n_resonances`. The generic
  `_check_background_study` refuses couplings / material readers of the listed types; the study's validate adds only
  scipy and the feature checks.
- `Snapshot.label` (time snapshots None); summaries carry `snapshot_labels`, frequency_domain adds `wavelengths`
  (check.json: `n_wavelengths`); results_io saves the label (read with .get). `Snapshot.live` flags the snapshot
  holding the state the session keeps when that is not the last (`_numbered_snapshots`: the number `reselect`
  returns, only with final_last); summaries carry its index as `live_snapshot` (None = the last; code rebuilding a
  snapshot list (sectors, refinement levels, failed last sweep value) recomputes or clears it); the GUI slider starts
  there. `_numbered_snapshots` inserts the mirror parity before the colon ('mode k [x_lo symmetric]: ...',
  `core.mirror.parity_tag`); `analytic.snapshot_index` matches the head without the tag only when nothing matches
  whole or by head ('resonance 2' finds a single mirrored run's snapshot, ambiguous in a sector run).
- Parity sectors (`study_sectors.py`; setting `parity_sectors` of time_dependent, stationary, frequency_domain,
  frequency_response, mode_analysis, eigenfrequency, `study._PARITY_SECTORS_PARAM`; validation `_with_parity_sectors`:
  needs a mirror face (EM bc or membrane symmetry face; error), warns for enabled couplings (linear model assumed)):
  `run_study` dispatches before the sweep test to `run_parity_sectors`; `_run_sectors` builds and runs every
  `ordered_sector_specs` copy (setting off; the model's own sign combination last so the live session matches the
  user spec's mirrors; api `Results._state_spec` unfolds with the last sector's), one session at a time (a failed
  sector stops the loop: the study fails, a sweep value is NaN). With a sweep (`_run_sector_sweep`, cold starts,
  `study.sweep_value_spec` / `sweep_values_of` shared with `_run_sweep`): sweep derived = combined values, results =
  the last sector's, `sector_derived` per sector, parity_sectors the last value's. Merge: every sector snapshot
  unfolded (`FieldData.unfolded`); driven studies (time_dependent included: equal snapshot times per sector) first one
  combined snapshot per index (combine_sector_fields of linear signed fields), sector snapshots relabelled '<label>
  [<sector>]'; eigen studies sector by sector (labels carry the parity). Summary: derived (driven, all sectors ran;
  `_combined_derived`) = feature-sourced values from combined_results, field values by `_field_sector_rule` (the
  expression probed on random field values with the FieldInfo parities: linear in signed fields -> sector mean for
  integral / mean / at, quadratic (or linear in even fields) -> combine_sector_totals for integral / mean, else NaN +
  warning), field-free values re-evaluated from the combined ones; else NaN, references recomputed,
  ticks / study_seconds / n_modes / n_resonances summed, `parity_sectors` {faces, labels, status, combined, derived and
  results per sector, combined_results (driven, every sector ran: `core.mirror.combine_sector_result` per key all
  sectors have, then `combine_sector_port_totals`), snapshot_sectors}. Live state, summary fields and plain feature
  results = the last sector. The GUI unfold box uses the user spec: wrong only when physics disagree on a face sign.
- Mesh refinement levels (`study_adaptation.py`): `run_study` dispatches `StudySpec.adaptation` first (validation
  refuses it with a sweep, parity sectors or a study without `StudyType.supports_adaptation`: stationary,
  frequency_domain, mode_analysis, eigenfrequency). Per level one build + `_run_built`, one `_StageProgress` stage each,
  the previous level's session closed first; level 0 = the user's spec (its failure fails the study); a later level's
  failure stops with the previous summary and no live session. Stop rule (`level_stop` / `next_level_stop`, pure):
  converged |v_k - v_k-1| <= tolerance |v_k|, not_finite, max_levels, then for the planned next level no_refinement,
  max_cells (incl. padding), memory (CPU `estimate_memory` vs budget / host RAM). Next meshes (`mesh_refinement.py`,
  CPU): `frozen_mesh_spec` = numbers only, auto replaced by the grid it gives (lattice Boltzmann: uniform), segments ->
  edges; 'uniform' = every size x growth^(-1/dimensions) (custom edges resampled); 'adaptive' (tensor 2D / 1D / polar
  radius; refused for 3D; a mesh that must stay uniform stops the loop as 'failed') = top-eta^2 cells by bisection on
  their count until `estimate_mesh` ~ growth x cells, per axis runs of marked indices -> MeshBandSpec at base / 2^k;
  a uniform mesh turns nonuniform and gets
  `snap_refinements` (dx = base x (1 - 1e-6), margin 0: object faces stay cell edges, else grading moves
  lattice-aligned interfaces off them). Triangles (`adaptive_triangle_spec`): per-element target sizes from the
  indicator; nodes of marked elements get `MeshSpec.size_points` [x, y, h] (appended, only where finer than the old
  size field) and the whole mesh is rebuilt. Size points apply in every mode as distance sources graded by max_ratio
  (`triangle_sizing` rounds h down to SIZE_POINT_STEPS per octave so KD-tree levels stay few; three forms to keep
  equal: `size_point_function` (raster / prediction), `mesher_size_field` (fem SizeField), `_triangle_estimate`
  sub-pixel sampling; `TriangleSizing` repr shows their sha1: GUI mesh keys are reprs); validate refuses them on
  tensor meshes. The level record counts built triangle elements (`_built_counts`). Indicator (`mesh_indicator`): a
  solver's optional `error_indicator()` -> (2, *window) eta^2 parts (triangles: (n_elements,), required: no
  interpolation fallback), else `core.error_indicators` on `PhysicsType.indicator_fields` (() = primary; only
  laminar_flow sets it), scaled per field by max |u| and per physics to a unit total. The hook wins over
  indicator_fields; it raises (warning, physics skipped, no fallback) where it does not apply. Known gap: an FDFD TFSF
  plane wave's outline jump is flagged as error; 3D hooks raise. Hooks: DiffusionPhysicsBase Kelly flux jumps
  (`DiffusionSystem.flux_jump_indicator`, energy norm 1 / c, only faces coupled between free cells), SolidMechanicsBase
  traction jumps (2D, 1 / c11), em_modes (the mode_index mode) / em_fdfd_ez / _hz (shown solution or resonance)
  energy-weighted interpolation of every E / H component; triangles in `fem.indicators` (fem sub file: residual + flux,
  mode tangential H / normal D, traction, tangential-H jumps). Window shapes come from the level's `estimate_mesh`, not the Session.
  `convergence_report` -> summary 'convergence' (api `_STUDY_SUMMARY_KEYS`, `Results.convergence()`, convergence.json):
  Richardson over the last three levels (`richardson`, v = v_inf + C N^-q in the refined window cells; None + note
  when changes alternate or do not shrink); ticks / study_seconds summed; keep_levels prefixes earlier snapshots 'level
  k: '.
- Parameter estimation (`StudySpec.estimation`, `study_estimation.py`; pure parts in `estimation.py`): `run_study`
  dispatches it before adaptation (validation refuses it with a sweep or adaptation; parity sectors run
  `study_sectors._run_sectors` + `_merge_runs` per evaluation, targets read the combined values). `resolve_estimation(
  spec, parameters)` is the one rule set: validate calls it (imported inside `_check_estimation`, so only estimation
  models depend on it in the selftest cache), the run raises its errors through `_fail_sweep_setup`. Each evaluation =
  a spec copy with the estimated parameters as numbers (`study.estimation` None), built and run like a sweep value;
  search vector z = 1 + (value - initial) / scale (scale |initial|, else upper - lower, else 1), so least_squares'
  relative diff_step never falls back to 1e-8 at 0. Own evaluation counter (scipy's max_nfev does not count Jacobian
  runs) raises `_Stop` through scipy, also on cancel, a failed later run and a non-finite residual; a failure of
  evaluation 1 is StudyFailed. The best point is rerun unless it is the live session; standard errors only from a
  least_squares Jacobian (`fit_statistics`, scaled by chi^2 / dof unless every sigma was given).
  `remember_element_meshes` is on for the run, so a geometry-free parameter never remeshes. ticks = evaluations;
  summary 'estimation'; time / build estimates x `estimation_runs`. Not yet: re-evaluating derived-value-only
  parameters on one session.
- Run-time estimate rules (CPU, GUI-thread safe): `estimate_time(spec, parameters)` -> (`core.time_estimate.
  TimeEstimate`, message) priced on the stored machine profile of the run backend (`python -m open_sol profile
  [--backend] [--quick]` / `api.profile_machine` -> `time_profiles/time_profile_<backend>.json` next to the package,
  git-ignored, folder `OPEN_SOL_PROFILE_DIR`; a profile of other benchmark code or library versions is ignored: (None,
  reason)). `PhysicsType.time_estimate(settings, mesh, TimeContext)` counts the physics' own work and asks its roles
  through `context.solver_work(role, ...)` -> `SolverType.time_estimate(values, SolverWork)`: each op priced once,
  each module counts only its own work. Per-step items (`per_step`, the part's `dt_limit`) are multiplied by the study
  (`StudyType.time_counts(values)`); explicit engines take per-step bytes from their memory estimate
  (`explicit_step_estimate`), so a new array updates both. Counts are 'exact', 'bounded' or 'guess' (a named module
  constant to calibrate, x/÷ 3 span). Work without a recipe is listed as not estimated, never 0 s. Host matrix set-up
  between an assembly and its factor (free-dof restrictions, pattern_sum, complex casts, a static state before a
  pencil) is an item of the module doing it, as passes over the entry bytes (solid_mechanics `_RESTRICT_PASSES` /
  `_CSR_REBUILD_PASSES`); no factor curve holds it. fem.harmonic scalar systems on triangles (pressure_acoustics,
  membrane) price build and per-frequency host work through `definition_helpers.scalar_harmonic_build_items` /
  `harmonic_frequency_items` (counted from fem.dofs / elements / assembly / harmonic: a change there updates them). A
  complex operator goes to a role as value_bytes 16 (`SolverWork.complex`), never as a hand-made complex factor_scale.
  MUMPS / PARDISO back ends / factorizations are priced as the SuperLU items (HostLU / HostSolve on the host profile)
  scaled by the uncalibrated `sparse_direct.HOST_DIRECT_*` time constants: no profile curve times those libraries.
  After a run summary.json `time_check` compares items with their `counter` paths and seconds groups
  (`seconds_counter`: timers at existing syncs only; the rest of study_seconds = 'study'); warnings: an exact count
  off, a group outside UNIT_RATIO_BAND with >= MIN_CHECK_SECONDS; guesses never warn. Physics work at build (port mode
  solves, first-use kernel compiles) has no recipe yet. Several runs: `study_runs` lists sweep values and, with parity_sectors, every sector run (labels
  '<sector>' / '<parameter> = <value> [<sector>]'), each estimated on its own without counters; a parameter
  estimation repeats the whole (and `estimate_overheads`' build items) `estimation_runs(spec)` = max_evaluations + 1
  times (an upper bound: exact counts become 'bounded'). per_run (the post-run check) stays one run.
- Run-time estimate (`time_estimate.py`; ops in the core sub file): `estimate_time` runs
  `meshing.estimate_memory(by_physics=)` first, so a recipe's `context.memory` is its own part (per_run: the builder
  hands its own estimate's by_physics in as `memory_parts` and its estimate to `estimate_overheads(memory=)`, so a
  build runs estimate_memory once; with fitted plain_roles the parts are estimated again). Per-step totals
  (`_with_study_steps`): time_dependent ticks = round(duration / dt_sync), dt_sync = min dt_limit, a physics with
  `every` n runs ticks // n steps; frequency_domain with a finite dt_limit (harmonic FDTD) a guess of
  HARMONIC_PULSE_FRACTION_GUESS x max_duration / dt; other studies 0 steps. A recipe raising ValueError / KeyError /
  TypeError / ZeroDivisionError lands in not_estimated ('recipe failed'), never stops validation (selftest
  time_estimate_recipes fails on it). A sweep estimates every value's spec (its own mesh estimate; the caller's
  mesh_estimate is not used), labels '<parameter> = <value>: ...', clears the counters; the builder estimates
  per_run=True for the post-run `_time_check` (None without a profile or for a non-completed run). A run whose mesh
  signature (`_mesh_signature`) and parity sector an earlier run of the same estimate had gets
  `TimeContext.ordering_kept` -> `SolverWork.ordering_kept` (kept cuDSS orderings: `cudss_plan_fraction`); a later
  run of a reuse_build sweep (same sector) or of a warm_start study with ordering_kept gets
  `TimeContext.continues_previous` (the solver state continues: drift_diffusion prices one continuation step) and,
  with reuse_build, `TimeContext.live_parameters` (a recipe whose followed setter cold-starts checks it); the
  per_run check sets none of them. Stored-field snapshots on triangle meshes are study work (`snapshot_read_items`, whose
  locate counts follow fem.sampling's PointLocator: a change there updates them). A quasi-static physics on or downstream of a cycle of enabled couplings
  (`fixed_point_physics`, TimeContext.fixed_point; a self-coupling is a cycle, so is a coupling whose type reads its
  target, `CouplingType.reads_target`) solves FIXED_POINT_ITERATIONS_GUESS times (`static_solves`), every other
  physics once (TimeContext.coupled = "a coupling reads or writes it", the time_dependent re-solve rule). Jacobi-PCG
  condition `TimeContext.jacobi_conditioning(HeldValues)`: (0, 0) on polar / spherical; contrast, axisymmetric
  weighting and the transient capacity term not counted (heat passes no HeldValues for implicit steps). Re-solve-on-change recipes
  (drift_diffusion, solid_mechanics, magnetic_fields_ac) use `definition_helpers.coupling_target(context)` /
  `material_follows_state(context, names)` and count per-step re-solves 'bounded'. `estimate_overheads` (build per
  sweep value (first only with `reuse_build`) x `estimation_runs`, output once per run; *_GUESS constants) is separate
  from estimate_time, so time_check and study totals never include it; calibrate its guesses from the selftest time
  split.
  Counter-timed study work a physics does at build (a factorisation in finalize): the counters at study start
  (`solvers_at_study_start`) give time_check `seconds_at_build`, and `api._study_work_at_build` moves it out of build,
  else the 'study' rest group is short by that work.
- Build estimate of element meshes: `mesh_build_items` adds per 'triangles' mesh "mesh '<name>': triangle mesher"
  (guess, `mesher_rounds`), "mesh tagging" and "TriangleDomain set-up"; per swept tetrahedra mesh only "mesh '<name>': footprint triangle mesher"; mesher items only when the
  run's `element_mesh_key` differs from the kept one. Other tetrahedra meshing is not priced. Labels must keep the
  "mesh '<name>': " prefix: `build_time_text` sums those into the mesh's 'domain' stage. The builder logs "Build
  estimate: domain estimated X, took Y; ..." after "Build time by stage"; check.json `time_estimate.build` is the
  whole study's build. Constants are UNCALIBRATED; `triangulate(log=)` logs the split to calibrate them. The 'domain'
  stage is host-only and independent of earlier sweep values: every value that remeshes pays it in full.
- Solve flow (`solve_flow.py`, the GUI's Show solve flow chart; CPU, GUI-thread safe, nothing built): steps per row
  (model, one mesh + materials step per used mesh, one solver step per resolved (physics, role), one physics step,
  couplings, the study's steps, results) over lanes (one per resolved role of a physics). It reads the estimate
  item labels, so they are an interface: memory "[mesh '<m>': ]physics '<p>': ", "feature '<p>.<f>': ", "model: ",
  "subpixel smoothing: ", "conformal boundaries: ", "grid transfer "; time "physics '<p>': " (a solver item goes to its
  role by its `counter` / `seconds_counter` path ('solvers', p, 'roles', role, ...), so a role recipe keeps
  `work.counter(...)` paths), "mesh '<m>': " build items, "snapshots on mesh '<m>': ", not-estimated "physics '<p>'" /
  "coupling '<c>'". An item no rule matches lands in the red 'Unplaced estimate items' step (never dropped): a new
  label prefix gets a rule in `_Builder._memory_step` / `_time_step`. Study steps are per StudyType name in
  `_study_steps` (a new study type falls back to its summary with a note: add its steps there); the study's
  `time_counts` are listed generically. Time items come from one run (per_run); the whole-study total from estimate_time.
  Measured values (`build_record`, `study_summary` of a run of the same model): build ledger owners go to steps by
  `_Builder._ledger_step` (model.builder's `ledger.track` owner names, per-mesh ones suffixed ' <mesh>' with several
  meshes: a new or renamed owner needs a rule there, else it lands in 'unplaced'); time_check items by label (a
  multi-run check: the last '<run>: <label>' record), seconds groups by counter path (`_group_step`; 'study' = the
  first study step). Items are timed per group only: an item's measured seconds = its count x the group ratio.

## Registry, templates, describe
- A loop over `spec.physics` that skips unregistered types (validate reports them) takes
  `schema.known_physics(spec, registry)` -> (PhysicsSpec, PhysicsType).
- Templates: list them by `templates.template_entries(registry)`, build one by `template_model(entry)` /
  `make_template(key)` (they check the factory's return), never `entry.factory()`.
- `describe()` is deterministic (sorted sets, declared order, floats '%.6g', ASCII, no addresses / timestamps). Its
  hand-written tables are checked (`_table_problems` names the one a new key misses); a new nested spec dataclass
  also goes in `spec._NESTED_FIELDS` (unchecked; load reads it too). Help texts: each rule lives in one topic, others
  carry a one-line pointer; lead-ins shared by every physics / feature / coupling / study entry live once in the spec
  notes 'Physics, couplings and study'; keep repeated help texts identical word for word, so consecutive params share
  one entry and the full reference prints a long text once under '## Shared help texts'.

## Import contracts
- `import_contracts.CONTRACTS`: pattern and scope rules in its comments. A stale exception or importer pattern fails
  the check: delete it with the code it excused. Imports by `importlib.import_module` are invisible to it.
