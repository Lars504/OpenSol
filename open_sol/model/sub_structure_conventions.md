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
  field-dependent params are checked only for finite values. The masked extremes are fused NaN-keeping reductions on
  the gpu (`builder._masked_extremes`: no where() temporaries per evaluation); a NaN must keep failing the check.
- Param kind 'table' as a material property writes no array: the type's `create` reads `ctx.material_regions` (2D
  `region.mask` until `release_caches`) and resolves the rows with `resolve_values` (magnetostatics `bh_curve`).
- 3D prisms (root file "3D models"; optional z params of the 2D kinds: `geometry.Z_PARAM_NAMES`, axis /
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
  solver_misfit calls pass no grid), 1D EM other than em_fdfd_sphere (slab_modes), point-source units in 1D (the 1D
  slab's P unit reads W/m; it is W/m^2, the help says so: `SPACES` has no 1D entry).
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
- Triangle mesher (`triangle_meshing.triangulate`; module header; mesh data and mesher rules: fem sub file):
  `mesher_size_field` is the mesher's form of `meshing.triangle_size_field` (the estimate's raster; size points also
  `_size_point_density`'s sub-pixel samples): a size rule changes in all of them. Element tags use the build-mask
  rule (`element_masks` -> `contained`, bit-identical to rasterize), never a second point test. `mirror_x` /
  `mirror_y` mesh the half window and reflect it (`_mirrored_parts`); the estimate takes the denser of each pixel and
  its image; asymmetric material objects only warn (`mirror_mismatch`).
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
  tolerance only because the arrays are small. The build is backend neutral (`array_module_of` the coarse index /
  `domain.xp`, host work by `to_host`): a new array op there must exist in numpy and cupy alike (no `cp.` names, no
  cupy-only kernels).
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
- Reused sweep builds (`SweepSpec.reuse_build`, `sweep_reuse.py`): `reuse_issues` lists what refuses it (validate
  `_check_sweep` reports them at their node paths, `_run_sweep` fails a refused spec through `_fail_sweep_setup`);
  what follows the sweep is a `Param.follows_parameters` feature value / physics setting of parameters only, or a
  material value `_material_follows` accepts (every type reading it non-EM and the property `state_dependent`: the
  "materials_changed of an array it reads re-solves" contract, coupling sub file). The first build gets
  `live_parameters` (builder.build keyword: a custom build callback must accept and pass `**options`, as api's
  `_BuildRecorder`); a flagged value's apply (a setting's: the create) sets its build value, then calls
  `BuildContext.follow_parameters` (mechanics `follow_load` / `follow_held_values`), which registers a
  `ParameterFollower` only for a live raw value on a `solver_hooks.RerunSolver`; a material value is repainted by
  `builder._follow_material`. Followers keep host data only. The session is kept only when every physics is a
  RerunSolver (`non_rerun_physics`, else an info line and one build per value) and every following value a live
  parameter reaches got a follower (`unfollowed_values`, matched by its raw spec dict: an apply skipping
  follow_parameters on some grid kind / study gets a warning and one build per value): per later value
  `Session.set_sweep_value` + `reset_run_records` (take kept snapshots before), then `_run_built`; a failed run drops
  it (the next value builds anew); table `builds` counts builds. Feature values / settings are not range-checked per
  value (material values are, by their Param): flag only unbounded params, or check the bounds in the setter (EME
  wavelength, AC frequency > 0, doping >= 0: a ValueError fails that value). `estimate_overheads` prices the one
  build, `estimate_time` still every value's run.
- Study catalogue: `study.register` registers time_dependent, stationary, frequency_domain and mode_analysis, then the
  `study_*.py` modules in registry order; which physics types run a study is `PhysicsType.supported_studies`
  (api_reference per type), what a study does its `StudyType.description` and the module docstrings. frequency_domain
  scans wavelengths, frequency_response frequencies in Hz. Optomechanics: `em.definition_common.eigen_solved` tells
  the FDFD estimator / resonance_table which FDFD physics is eigen-solved (one no optomechanical_coupling names runs as
  background, yet `_check_background_study` still refuses readers of it: its type lists the study); the solid field
  om_integrand exists in this study only.
- StudySpec.solved_physics ([] = the Protocol decides; only in a study with `StudyType.supports_solved_physics`): the
  physics the study solves itself; the others, also of a type listing the study, run as stationary background
  physics (allowed as coupling sources into the listed ones). Run side `study._selected_physics` into
  `_solver_roles(selected=)` / optomechanics `_roles(selected)`, validate side `_solved_by_study`
  (`_check_solved_physics`, `_check_background_study`): both must read the list alike. A type definition whose rules
  depend on the study asks `spec.study.solves(name)` (mechanics `definition_common.background_context`).
- Stress stiffening (`solver_hooks.PrestressSolver`): a study eigen- or harmonic-solving solids calls
  `study._solve_static_states(runner, solved, with_loads, emit)` right after its background stage (with_loads False
  only in frequency_response: its loads are the drive).
- Background studies (`StudyType.background_study` 'stationary'): validate accepts a physics supporting the study or
  its background_study, needs >= 1 supporting the study itself and refuses an enabled coupling / material property
  reading a solved physics (`_check_background_study`, remedy `StudyType.feedback_alternative`; exceptions
  `coupled_points` and `solution_feedback`, below). A new one starts from `_solver_roles` (solved = solvers satisfying
  its Protocol; the rest must have dt_limit inf) and `_background_stage`. frequency_domain: a time-stepped background
  physics must have `attach_harmonic` (em_fdtd_*: `_run_harmonic_pulse` steps only those, never `runner.advance`,
  which would step the background too).
- Coupled wavelength scan (`StudyType.coupled_points`, frequency_domain only): `_run_coupled_wavelengths` is chosen when
  a runner coupling or DMU reads a harmonic / pulsed physics; validate turns the reader errors into one info only when
  every physics lists the background study (else the errors name the physics that cannot join: em_modes, FDTD).
  Nothing is reset between wavelengths (list order = scan direction); a harmonic solver needs a settable `omega`
  property (FDFD, BOR, EME).
- Derived series (`_SnapshotRecorder(series_axis=)`: time_dependent, frequency_domain): summary `derived_series` lives
  in summary.json, not results.npz; a sweep table keeps one per value. Every sweep loop (`_run_sweep`, study_sectors)
  emits `SweepValueDone(..., sweep_table_copy(table))` after every value, also a failed one.
- StudySpec.stored_fields [] (the default) stores every `session.available_fields()` entry in each snapshot
  (`_stored_field_keys`): a new renderer field must be readable after every study solve its `requires_study` allows,
  else every default run of that type fails at its first snapshot.
- GUI run features (`FeatureType.gui_run_studies`, `schema.with_gui_run_features`): only a GUI Compute adds them, so
  a headless run of the same spec has no such result; the tuple must mirror the feature's validate rule, else a GUI
  Compute fails at a feature the tree does not show.
- Snapshot solutions (`RunControl.keep_solutions`: GUI worker on, `api.run` / CLI / self-tests off): which solution a
  snapshot keeps per physics is `_SnapshotRecorder._solutions`, how it is read back `Session.snapshot_field` /
  `restored_snapshot`. A new study that selects a kept solution (modes, resonances, harmonic frequencies, channels)
  passes `selections` to `record` and selects through `study._select`; one that re-solves a physics after a snapshot
  must not put it in `recorder.fixed_physics`. Solutions never leave the session: code copying snapshots out of it
  (kept sweep snapshots, parity-sector merges, adaptation `keep_levels`) drops them.
- frequency_response: setting `snapshots` ('all' / 'last' / 'none') gates only the recorder; solver-side per-frequency
  records (harmonic probes, reaction tables, ports, mode projections) never depend on it, so a new per-frequency
  output belongs in the solver's records. A ReducedSweepSolver's `publish_frequency_response()` runs after every sweep
  with a solved frequency (also a cancel), before the 'last' snapshot.
- Solution feedback (`StudyType.solution_feedback`): only a study whose own solve ends in ONE operating point may set
  it (never one with many solved states: wavelengths, modes, resonances); couplings reading its solved physics may then
  write background physics (validate info; solved -> solved and material readers stay errors). Run side generic in
  study.py: `_feedback_couplings` (never a DMU), `_background_stage(feedback=)` (`solve_stationary(inputs=solved)`:
  the solved physics are not stepped, their couplings land in schedule[0], applied before every sweep, unrelaxed),
  then `_feedback_rounds` after the study's first solve. The monitor's copies are the run item
  `meshing._feedback_copy_count` (a change of its members updates it). A SlotSum mixing solved and background sources
  is a feedback coupling: its background contributor then also only refreshes before each background sweep.
- Snapshot list rebuilds: summaries carry `live_snapshot` (the `Snapshot.live` index, None = the last); code rebuilding
  a snapshot list (sectors, refinement levels, a failed last sweep value) recomputes or clears it.
- Parity sectors (`study_sectors.py`, study setting `study._PARITY_SECTORS_PARAM`, validate `_with_parity_sectors`):
  `run_study` dispatches to `run_parity_sectors` before the sweep test; the merge assumes a linear model (enabled
  couplings: a warning). The model's own sign combination runs last, so the live session, summary fields and plain
  feature results are the last sector's (api `Results._state_spec` unfolds with it); combined values are in summary
  `parity_sectors` (`_merged_summary`, `_field_sector_rule`). The GUI unfold box uses the user spec: wrong only when
  physics disagree on a face sign.
- Mesh refinement levels (`study_adaptation.py`, next meshes `mesh_refinement.py`; loop, stop rule and mesh rules in
  their docstrings): level 0 is the user's spec, so its failure fails the study; a later level's failure ends with
  the previous level's summary and no live session (that session was closed before the failed build). 'adaptive'
  turns a tensor mesh nonuniform and adds `snap_refinements`, so object faces stay cell edges (else grading moves
  lattice-aligned interfaces off them); a mesh that must stay uniform raises, which stops the loop as 'failed'.
  Indicator window shapes come from the level's `estimate_mesh`, never the Session; level records count the built
  elements of element meshes (`_built_counts`: their estimate is a size integral). Indicators: the solver's
  `solver_hooks.ErrorIndicatorSolver` hook (grep `def error_indicator`; triangles: fem sub file "Error indicators"),
  else `core.error_indicators` on `PhysicsType.indicator_fields`. Known gap: an FDFD TFSF plane wave's outline jump
  is flagged as error.
- Parameter estimation (`study_estimation.py`; run-free parts `estimation.py`): `resolve_estimation` is the one rule
  set; validate imports it inside `_check_estimation`, so only estimation models depend on it in the selftest cache.
  Each evaluation is a spec copy with the estimated parameters as numbers, built and run like a sweep value (parity
  sectors: `_run_sectors` + `_merge_runs`; targets read the combined values). Not yet: re-evaluating parameters that
  only derived values read on one session. Study records of these studies (summary 'convergence', 'estimation')
  reach summary.json through api `_STUDY_SUMMARY_KEYS`, as every study-specific record: a new one is added there.
- Run-time estimate rules (`time_estimate.py`, ops and pricing: core sub file "Run-time estimate"; recipe rules: root
  "Physics definitions"): `PhysicsType.time_estimate(settings, mesh, TimeContext)` counts the physics' own work and
  asks its roles through `context.solver_work(role, ...)`: each op is priced once, by the module doing the work (host
  matrix set-up between an assembly and its factor, such as free-dof restrictions or complex casts, is an item of
  that module: no factor curve holds it). A complex operator goes to a role as value_bytes 16 (`SolverWork.complex`),
  never as a hand-made factor_scale. Work without a recipe is listed as not estimated, never 0 s; a 'guess' count is
  a named `*_GUESS` module constant to calibrate. Shared counts come from their helpers, never re-derived:
  `definition_helpers.eigen_search_counts` / `frequency_sweep_counts` / `reduced_sweep_time_items` /
  `reduced_sweep_memory_items` (eigenfrequency / frequency_response), `scalar_harmonic_build_items` /
  `harmonic_frequency_items` (fem.harmonic systems on triangles), `coupling_target` / `material_follows_state`
  (re-solves on change, counted 'bounded'). Items counted from another module's code (those helpers from fem.dofs /
  elements / assembly / harmonic, `snapshot_read_items` from fem.sampling's PointLocator) change with that code.
  `estimate_overheads` (build and output) stays outside estimate_time, so time_check and study totals never include
  it; calibrate its guesses from the selftest time split. Its build part is one generic item per owner of the memory
  items ('build of <owner>', `meshing.memory_item_owner`: a stage per kept / host item, passes over their bytes), so a
  host item a physics allocates only in the study is still priced as build work there (a lead when the ledger's stage
  seconds of that step disagree). Counter-timed study work done at build (a factorisation in
  finalize) reaches time_check as `seconds_at_build` (counters at `solvers_at_study_start`); api
  `_study_work_at_build` moves it out of the build seconds, else the 'study' rest group is short by that work.
- Build estimate of meshes (`mesh_build_items`, `build_time_text`): its labels keep the "mesh '<name>': " prefix,
  which `build_time_text` sums into that mesh's 'domain' stage (the "Build estimate: ..." log line) and the solve
  flow places; no other build item may start with it (the owner items start with 'build of '). A tensor mesh is one
  generic stage (no arrays); the element-mesh constants are UNCALIBRATED; `triangulate(log=)` logs the mesher split
  to calibrate them. Unswept tetrahedra have no item yet.
- Solve flow (`solve_flow.py`; layout and placing rules in its module docstring): estimate item labels, role
  recipes' `work.counter(...)` paths and the builder's `ledger.track` owner names are its interface. A new label
  prefix needs a rule in `_Builder._memory_step` / `_time_step` and in `meshing.memory_item_owner`, a new or renamed
  owner one in `_ledger_step`, a new study type its steps in `_study_steps` (else a generic step with a note):
  unmatched items land in the red 'Unplaced estimate items' step, never dropped. Memory goes to a solver job's box
  only for items its physics estimator tags (`core.memory_estimate.solver_job_items(role, items)`; so far the EM
  resonance searches: em_fdfd_3d and the `em.memory_estimates` eigen branches); untagged factors show at the physics.
  A step's memory figure is its device items alive at the estimate's peak (`MemoryEntry.at_peak`, the rule of
  `MemoryEstimate.peak_items`) and, apart, its host items: the steps add up to `device_peak` and `host`. Never sum a
  step's listed items (card + processor memory and phases that never overlap: a figure no total matches). The
  Measured column holds measurements only, with one marked exception: the outside-pool row shows cuDSS's own plan
  estimate ("cuDSS's figure"; nothing measures memory outside the pool during a run). Sums of a measurement and an
  estimate (`_card_bound_note`: pool high-water + cuDSS's figure, an upper bound of the card's peak) go in the
  notes, never in that column.
- Memory issues (`validate.check_memory`, the sweep value check, `_host_memory_text`) name the same steps:
  `meshing.memory_by_owner` of `estimate.peak_items()` / `host_items` (owner = `memory_item_owner`, the label part
  the solve flow places by). Judge and list from that one estimate; a second count of the model's memory in an issue
  drifts from the chart.

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
