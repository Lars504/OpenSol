# open_sol/model — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first): spec, validation, build, session, studies, worker,
templates, describe. Current state, no changelog.

## Model layer
- Expressions: ExpressionError is a ValueError; integer literals become floats; overflow fails at evaluate; any cupy
  argument switches to cupy. `evaluate` ignores invalid / divide (np.errstate, like cupy): consumers catch NaN / inf
  (evaluate_scalar, builder `_check_range`, curves drop non-finite samples). Syntax fix hints: `_SYNTAX_HINTS`; unknown
  names list close matches (qualified `<physics>_<name>` first). The parser NFKC-maps the micro sign to Greek mu
  (`_NAME_ALIASES` rewrites um, us and reserves them). An uncalled function name is a free name (`names`): only a
  namespace entry supplies it (FDFD probe xy series 'abs'); one name both called and uncalled in one expression
  raises at compile (one eval scope slot).
- Special functions (j0, j1, y0, y1, i0, i1, k0, k1, erf, erfc): scipy.special on numpy (imported on first call),
  cupyx.scipy.special on cupy; reserved like every function name (no parameter named y0 / k0).
- Names: `<physics>_<field>` always, plain `<field>` when one physics alone provides it (parameters may not use it).
  Settings / feature params / material properties see parameters + x, y (field_dependent) + state names and t
  (state_dependent) / t alone (time_dependent: no x, y with t, `schema._number`); coupling 'expression': value + x, y +
  parameters; derived values: + t + `output_field_names` + `scalar_output_names` (settings resolved per call: names
  follow e.g. n_modes; with a `source`: the result's sample names instead, which shadow parameters); derived `at` /
  `reference`: parameters only; analytic curves: parameters + `analytic.curve_variables(target kind, series)`
  (variables shadow). z joins x, y only in 3D (`expressions.FIELD_COORDINATE_NAMES`); validate
  `_height_in_2d_message` rejects it in 2D values of kinds float / int / expression (a choice such as prism axis 'z'
  is literal text), derived values and line curves (cut height: `analytic.field_plane_height`).
- Analytic comparison (`analytic.py`, numpy only, GUI-thread safe): references are evaluated on the CPU from the
  session's parameters (`_run_built` -> derived_reference / derived_error; sweep table also for failed values).
  `result_value` reduces a source result; xy series / table column names exist only at run time (validation checks
  names only for timeseries / flux_spectrum). Analytic curves never touch a run: drawn over `built_parameters(spec,
  summary)` (the summary's `parameters`, also after a cancelled sweep; no summary: the sweep's last value), Sweep tab
  overlays over `parameters_with(spec, parameter, value)`; 'sweep' curves / references by `sweep_curve` on the swept
  values plus linear and (positive range) geometric points, so a log axis still passes every simulated point.
  `curve_variables(kind, series)` is the one variable rule (validate, settings form, `feature_curve`): timeseries on
  `sample_axis` (own t at 501..20000 samples, so ring-downs do not alias), `iteration` (1-based) when every t is equal
  (`is_iteration_series`); series 'fft' only with the caller's FFT bins (`frequency=`: analytic never computes the
  FFT); tables via `table_xy(result, x_column)`. Line curves: `plane` (3D, `line_plane`) and `snapshot`
  (`snapshot_index`: int, or label / its text before ':'; several matches raise; wavelength labels '.6g' nm;
  snapshots are the mid-height cut, so validate refuses plane + snapshot). A six-coordinate `line` is a 3D segment
  (`is_segment_3d`; 3D only, no plane; final state by `Session.segment_profile`: live Results only; with `snapshot`,
  `_SnapshotRecorder` stores every 3D-segment curve's profile per snapshot in `Snapshot.segments[curve name]`,
  results.npz 'snapshot/<i>/segment/<name>/'; parity sectors keep sector profiles (half, not unfolded) and mean them
  for linear signed fields in the combined snapshot). `sweep_table_xy` (api `Results.sweep_xy`, GUI Sweep table
  columns) turns per-value 'table's into one 'xy' vs the swept parameter on request (never stored);
  `sweep_value_curve` / `sweep_overlay_xy` are the one rule of GUI Sweep overlays and api sweep figures. Result keys
  'physics.feature' follow `rename_physics` / `ModelSpec.rename_result_key`; a deleted feature leaves a dangling key
  (validation error).
- Derived curves (`DerivedCurveSpec`, `analytic.derived_curve` / `derived_curve_plots`): inputs
  'physics.feature:sample' through `result_samples` (any derived-value sample, scalars too; far_field: its first
  wavelength); abscissa = the first array input's, sorted; later inputs `np.interp`-ed onto it (NaN outside).
  Expression names = parameters + aliases (shadowing) + x. `rename_parameter` skips every name that shadows
  parameters (`ModelSpec._parameter_shadows`: aliases / x, curve variables, result sample names; xy / table sourced
  derived-value expressions untouched) and refuses a new name one of them would shadow. Plots group by `plot or
  name`; failing curves drop out with a text (never raise); series names unique per plot (`_unique_series_name`),
  `curve_series` maps each curve to its own. Consumers pass `fetch(key)`: api `Results.derived_curves()` (host cache,
  else live; summary.json derived_curves, figures/derived_<plot>.png honour log_x / log_y) and the GUI tab (worker).
  Final results only: no per-sweep-value curves.
- Curve calculus: `gradient` / `cumtrapz` are `analytic.CURVE_FUNCTION_ARITY`, not FUNCTION_NAMES (not reserved, not
  callable in field expressions): every compile of a derived-curve expression passes `functions=` (validate,
  `curve_values(functions=)`, GUI `FormContext(functions=)` -> `schema.check_value(functions=)`, kind 'expression'
  only). `curve_functions(axis)` closes over the abscissa and works on finite samples only (NaN of an interpolated
  input stays NaN, cumtrapz starts at the overlap).
- Derived-curve `reference` (x + parameters) / `tolerance`: series '<series> reference', xy keys `reference_series`
  / `curve_error` (on the curve's own abscissa, ptp scale) -> summary.json derived_curves (`curve_error`,
  `y_reference`); selftest_examples reads them as 'derived_curve' entries of 'curves'.
- Derived values over a source (`result_value(result, derived, parameters)`): `window` [lo, hi] filters the sorted
  finite samples (integral adds interpolated edges), then `reduce_samples` (mean / max / min / integral, argmax,
  fwhm, polyfit / exp_fit / lorentz_fit / two_exp_fit / two_wave_fit / richardson / multi_lorentz_fit / pole_fit with
  `fit` / `degree`). `COMPLEX_FIT_OPERATIONS` (two_wave_fit, pole_fit) pair `<name>_re` + `_im` in result_value
  (validate cannot see xy series names). `fit_quantities(operation, degree)` is the one rule for `fit` choices
  (validate `fit_problem`, run, GUI): polyfit 'c<k>'; peak fits (`PEAK_FIT_QUANTITIES`, degree = number of lines)
  '<name>_<k>' per line ordered by position, then the shared ones ('spacing' only for degree >= 2);
  `DEGREE_FIT_OPERATIONS` read `degree`. A new fit adds its names there, not in validate or the GUI. far_field sources
  (`wavelength`: nearest recorded) are periodic (`FAR_FIELD_PERIOD`): samples tiled over three periods, 'at' wraps, no
  window = one period (peak-centred for argmax / fwhm). A source without '.' is a derived curve (`is_curve_source`):
  Session.derived_value builds it (`derived_curve_result`, sample names curve name + x) from the session's feature
  results in the ordinary derived-value step (no second pass), so sweeps get it per value; renaming the curve leaves
  the source dangling (validation error).
- Derived values read the values of the derived values listed BEFORE them (`Session.derived_values` /
  `derived_value(earlier=)`; the GUI `_evaluate_derived_values` and the derived-series recorder pass them; a series
  keeps only values whose references are in the series too, `session.earlier_derived_names`); validate errors for a
  later name or one shadowed by a parameter / field / scalar / reserved name; `ModelSpec.rename_derived_value` renames
  the references (later expressions, adaptation target, 'sweep' analytic-curve series). Operation 'expression'
  (`schema.GLOBAL_DERIVED_OPERATION`, source None; in `SOURCE_FREE_DERIVED_OPERATIONS`) = a standard global evaluation:
  one float64 `evaluate_scalar` over `Session._plain_namespace` (parameters, scalar outputs, t, earlier values; no
  mesh, selection or point), branched in `Session.derived_value` before `evaluate`, so every consumer (summaries,
  sweeps, series, references, parity-sector recomputes) takes it unchanged; validate `_check_global_expression`
  refuses fields / coordinates / itself. A field-free expression under a field operation still works (broadcast
  constant, float32 rounding). Field operations add 'mode_volume' (integral incl. mirror touch factor / max, or / the
  value at `DerivedValueSpec.point`) and 'at' (containing cell of `point`; `at` stays the source abscissa): schema
  `FIELD_DERIVED_OPERATIONS` / `SOURCE_DERIVED_OPERATIONS`. Figures of merit (stored energy, Purcell, n_eff,
  perturbative shift) are recipes over these (describe DerivedValueSpec.expression), no own code. Parity-sector runs
  re-evaluate field-free values from the combined ones (study_sectors `_recompute_value_expressions`). Mirror planes:
  max / min (and mode_volume's maximum) of `Session.evaluate` include the images of a selection reaching a mirror face
  (`_image_namespaces`: field x its parity's image sign, coordinates reflected; a field without parity, polar windows
  with x / y, or a coordinate mirrored on both faces raise -> NaN + warning); interface selections take no images.
- Interface selections (`interface_sampling.py`, host numpy; SelectionSpec kind 'interface' + side, 2D only,
  Cartesian or polar / spherical): a derived value whose selection is exactly one interface selection reduces outline
  samples extrapolated from one side (`interface_stencil`) with INTERFACE_VARIABLES normal_x / _y, tangent_x / _y
  (shadow parameters) and x, y. The membership side test at +-0.1 spacing along the normal must exceed the chord sag
  of the polyline midpoints, else small circles lose every sample. Device fields are gathered on the stencil cells only
  (one transfer per name). Cells come through a `CellFrame` (`window_frame`): blocks picked in index space, planes
  fitted on drawing-plane centres, `grid_normals` / `grid_tangents` along the field components' axes (radial / angular
  on polar). Line curves with `AnalyticCurveSpec.interface` sample host FieldData (`interface_line_profile`, api and
  GUI alike; non-finite cells never enter a fit; polar: live / GUI read `field(native=True)`, saved results and
  snapshots the image).
  `AnalyticCurveSpec.projection` ('normal' | 'tangential'): field is an x component, its y partner by
  `partner_component` (validate checks it is an output field), keys `line_curve_field_keys`, label `projected_info`. A
  side thinner than ~4 cells has no usable samples (error). Polar: blocks across an angle_wrap seam are clamped to one
  side.
- Curve errors (summary.json curve_error, api `_curve_errors`): every comparison goes through analytic
  (`feature_curve_samples` = the curve at the result's own samples via the same `_axis_variables` as `feature_curve`;
  `line_cell_curve_values` at the centres of the cells `line_profile` / `segment_cells` sampled: the containing-cell
  staircase, not the sample positions (a half-cell offset costs ~pi dx / lambda of a pulse peak), except a 3D segment
  with `AnalyticCurveSpec.sampling` 'trilinear' (`segment_trilinear_weights`: compares at the sample points); a line
  ending on a material interface or void edge still samples one cell of either side, so example lines start a
  fraction of a cell off it; `sweep_curve_values` + `sweep_simulated` for 'sweep' targets). Metric `curve_error`:
  |difference| / ptp(analytic) (flat curve: max |analytic|); 'sweep' targets `per_point` (relative per value). A new
  curve target kind / series rule must update both the drawing and the sample path. Series '' compares only with one
  candidate (one xy series / table column / sweep derived value); 'fft' / 'ratio' are never compared (note, n_points
  0).
- `selftests_analytic.py` (CPU, registered by study.register; harmonic_probe_record's check lives in
  core/selftests_harmonic_probe: core may not import model.schema) checks calculus, fits, windows and far-field wrap
  on synthetic results: keep it in step with analytic.
- `selftest_examples.py` (CPU, no api import: api runs the examples and passes summary dicts): example records, flags
  and the ranked `selftest_report_analytic.json`. `tolerance` on DerivedValueSpec / AnalyticCurveSpec is read only
  there (validate: finite >= 0); sweep_not_converging needs the swept parameter to change the evaluated mesh dx (auto:
  points_per_wavelength); refinement dx sweeps are not detected.
- Reused (cached) selftest records come back through read_json: non-finite numbers are the strings 'nan' / 'inf', so
  report code reading them must `float()` first.
- Case records come from `selftest_helpers`, not hand-built dicts: `SubChecks.result` / `failed_count_result` ({name:
  (value, tolerance)}) count failed sub-checks; `worst_result` (one tolerance), `scaled_worst_result` /
  `tolerance_units_result` / `normalized_result` (worst error in tolerance units, tolerance 1) rank NaN as the worst.
- `selftest_cache.py` (no api import; root file "Selftest reuse"): `UNTRACKED_PATHS` files add no units and no
  followed names, and their units are dropped from every set. Dependencies are code units: per package file each
  top-level function, each undecorated class with methods, and MODULE_UNIT (the rest: imports, constants, kernel
  source strings, decorated / method-less classes such as dataclasses, whose generated methods run as '<string>' and
  are never traced), hashed as `ast.dump` without docstrings (comments, blank lines, moves do not count). A run
  depends on the units whose code ran (`co_qualname` -> top-level name), MODULE_UNIT and its references of every
  touched file (twin(..., numba=factory), FeatureType callbacks), the units the run code names (`_unit_refs`, imports
  followed to the defining module) and one level more from named units that did not run (memoised getters, numba
  factories). Gap: a plain class used only through attributes of an instance built outside the trace.
  `package_units` is one snapshot per process (an edit mid-run makes a case stale next time, never falsely fresh);
  unit ids 'normcase'd path::name'; its own file is in no dependency set. Overhead: one Python callback per Python call
  (3.11 has no sys.monitoring). `wall_times` keep every key's last run time (any status) for the progress estimate:
  api.selftest / selftest_examples decide reuse for every phase up front (`api._SelftestPlan`; --all plans the
  examples run before the first case), so a new run path must go through the plan's start / finish or the numbering
  and estimates drift.
- Selftest subsets and reports: api `selftest(heavy=False | True | None)`; `--all` = both, then the examples run
  once (each on its own backend; one setting backend 'gpu' is `skipped_backend` without a usable GPU); named cases run
  whatever their flag; built-ins run in every unnamed subset, in a named run only when named; the report records
  `subset` / `n_skipped`. Files (`api.SELFTEST_REPORT_FILES`): `selftest_report_light.json`, `_heavy.json` (`--all`
  writes both, split by case), `_named.json`; examples `selftest_report_analytic.json` (`--examples` and `--all`); a
  run rewrites only its own files. Every report write goes through
  `api._write_report_keeping_timings`: the old file's per-case status / wall_time first lands in `<stem>_prev.json`
  (one run back only), so run times can be compared across branches; a reused case keeps its old wall_time, so
  compare `--reset` runs.
- Reading the CLI time split ('build / study / output / other'): 'other' = everything outside the timed phases: the
  check's Session reads plus per-run plumbing (release_session's `gc.collect()`, validation, estimates, run
  directory). `SelftestCache` freezes the parsed cache (`gc.freeze`), else many small runs paid ~0.4 s each in
  collection. A large 'other' on few runs means the check's reads cost: usually tetrahedron point location
  (`TetrahedronDomain.locate` remembers point sets; check code sampling its own lattice per component re-pays it:
  read through the Session or reuse one `locate` result).
- `SelfTestCase.backends` (default () = from its runs: solver_defaults.backend 'cpu' = cpu run, else gpu; no runs =
  cpu): without a usable GPU api.selftest skips the gpu runs (check gets the rest: a cpu-vs-gpu check must cope); with
  none left its variants still judge the case (record `variants_only`, report warning); with no runnable variant the
  case is skipped (`skipped_backend`, counted in n_skipped). A check({}) case using cupy declares ('gpu',); one
  comparing both backends itself declares ('cpu', 'gpu') and skips its gpu half by `gpu_available()`.
- Timing: `_run_built` adds `study_seconds` and `solvers` (`study._solver_records`: per physics the role solvers
  {type, solver node, backend, precision, counters} and the physics' optional `work_counters()`) to the summary. A
  study of several runs (sweep values, refinement levels, parity sectors) sums ticks / study_seconds and merges
  time_check through `study.summed_runs` (a new multi-run loop uses it too; `solvers` stays the last run's), and
  `run_study` sums builds into `build_seconds_total`, else the selftest split counts only the last run. Selftest run
  records carry `timing` (api `_run_timing`), `vram` (`_run_vram`: ledger / run_memory deviations from the estimate,
  gpu runs only) and `counts` (`_run_counts`: the time_check item most off its prediction; needs a machine profile);
  case / example records keep the worst as `vram_deviation` / `count_deviation` + `count_item`. The vram worst is
  picked over every part (verdict first, then beyond the 1 MiB floor), so a gpu run with an estimate always prints
  `vram ±xx% max N MB` (sign = direction; max = `vram_max_bytes`, the largest measured part over the runs, MiB; the
  verdict / part stay in the record, not the CLI line). Reused cache records from before the field print no max.
- Schema: Param min/max checked only for values evaluating over the parameters (field- / state-dependent floats
  become Expressions). The builder range-checks material properties instead (`eval_field(bounds_param=)` under the
  region, one sync; DynamicMaterialUpdate evaluators keep device running extremes, read on the host when every source
  physics has `n_solves`, is not transient and one re-solved since the last read, else at power-of-two evaluations
  and every `_DEFERRED_RANGE_CHECK_CALLS`; no read at the run end, so a range exit after the last read goes
  unreported); other field-dependent params only for finite values. `param_visible` expects `fill_defaults`;
  `resolve_values` resolves an invisible failing param to its default; a 'ref' of '' is None. Type definitions are
  checked at construction (`_check_params`: visible_if names a sibling param with valid choice / bool values;
  non-None defaults pass their own kind / bounds; field_ / state_dependent only on 'float').
- Param kind 'table' (`columns` ((name, unit), ...) required, `increasing`: >= 2 rows, every column strictly
  increasing): rows of parameter-only numbers, min / max per entry, resolved to [row tuple, ...]
  (`schema._table_rows`); renames treat it as a parameter-expression kind. As a material property it has no expands_to
  (required on every other property): the builder writes nothing; the type's `create` reads `ctx.material_regions`
  (2D `region.mask` until `release_caches`) and resolves the rows with `resolve_values` (magnetostatics `bh_curve`);
  `Simulation.add_material` refuses table properties.
- Spec v2 (3D): `from_dict` accepts versions 1 and 2 and sets 2; `ModelSpec.is_3d` = size_z set. 2D domain kinds carry
  optional `z0 / z1` (points / lines `z`; `geometry.Z_PARAM_NAMES`): validate rejects them in 2D, as well as 3D-only
  kinds (`GeometryKind.dimensions`), z faces and types whose `dimensions` miss the model's; resolve sorts z0 / z1. In
  3D, `material_preview(z=)` / coverage checks judge the xy footprint unless a layer height is given; vertical cuts
  (GUI side views): `vertical_section` / `section_mask` / `material_preview(cut=(axis, position))` (u = x for a 'y'
  cut, y for an 'x' cut; one evaluation per `cross_section_runs` run); `MeshEstimate.nz / dz_cells / z_edges` and
  `n_cells` include z.
- 3D prisms: rectangle / rounded_rectangle / circle / ellipse / polygon carry `axis` ('z' | 'x' | 'y'; x / y: the
  shape's (x, y) are the model's (y, z) / (x, z), `geometry.plane_axes`; z0 / z1 = ends along the axis, both
  required) and `sidewall_angle` (both ends, inward from min(z0, z1) along the axis; polygons mitred, simple only;
  collapse / self-crossing rejected at resolve). An x / y taper's cross-section is sampled into polygons
  (`_swept_polygons`, `_TAPER_LEVELS`; key = the height), as are taper side views. Neutral values ('z', 0) count as
  unset (`z_param_set`). Every per-z consumer goes through `geometry._section(shape, z)` -> (key, plain parts) and the
  footprint through `_footprint` (`is_plain` = neither), so a new 3D kind or option extends `_section` (and
  `surface_faces`, the 3D preview surface), never a consumer. `scene_3d.build_scene` colours an object by the last
  material of `material_paint_order` naming it ('all': legend only); a composite draws its primitive inputs. A taper
  makes every z cell its own run; `domain_object_rasterizer(reuse=)` keeps each object's last (key, mask). Mesh snap
  faces come only from `geometry.axis_parallel_faces` (meshing `_add_refinement` / `_add_z_refinement`); curved and
  tilted walls stay staircased. Selftest `model_3d_core` calls builder internals (`BuildContext`,
  `_apply_static_materials`) on a bare Domain3D and needs a 2D-only type (magnetostatics): keep it in step.
- Polar / spherical domains (`DomainSpec.coordinates`, 2D only): cells are logical (r - radius_min, angle - angle_min;
  core sub file "Coordinate metric"), geometry stays in the drawing plane about the polar origin. `geometry.DomainMap`
  (`domain_map(spec, parameters)`; also on ValidationContext, MeshEstimate (None on Cartesian), BuildContext, Session)
  is the one map (`to_physical` / `to_logical` / `contains` / `extent` = the sector's box: material_preview is index 0
  outside the sector, so preview shares / statistics go by `contains`, as `definition_helpers.conducting_fraction`).
  Rules: a type runs there only when its
  `coordinate_grids` list 'polar', and has geometry 'axisymmetric' exactly on a spherical domain (validate
  `_check_coordinate_grid`; GUI Add physics via `schema.COORDINATE_SPACES`). Meshing (`_mapped_plan`): no PML, uniform
  dr = size_x / ceil(size_x / dx), angle cells ceil(size_y r_max / dx); nonuniform only through MeshAxisSpec x (in r -
  radius_min) and y (in rad); object refinements and auto refused; `mapped_widths` are float32-rounded like the
  Domain's cells (edges[-1] off size by ~1e-8 relative: compare to the float32 widths, not size). Masks test the
  mapped centres over the whole window (`geometry.rasterize_mapped`: builder masks / `coordinate_grids` (also field
  expression x, y), Session selections, validate coverage); `_selection_bounds` (memory estimates) takes the logical
  cells of `DomainMap.logical_bounds` of each object's box (an upper bound); `BuildContext.point` returns the logical
  position (pos_to_cell / validate_physical_pos unchanged; lines / rects are not mapped: no polar type reads them).
  Session `evaluate` weights by the metric volumes (no `_ring_weights`); `field()` / `display_field()` return a
  containing-cell image over the sector's box (`_mapped_field_data`, NaN outside; saved FieldRecords carry that image
  with uniform pixel edges); `field(native=True)` (live only) gives the (r, angle) cells (axes ('r', 'phi' | 'theta'),
  y in rad, `FieldData.mapped_centres` = drawing-plane centres); the symmetry check is skipped. `definition_helpers.mapped_domain_issues` (heat,
  potential) refuses object-perimeter flux features and *_xy materials, warns for features on faces without area and
  replaces the x_lo-axis / cross-term part of `axisymmetric_issues`. `DomainSpec.angle_wrap` (polar only, core.metric
  ANGLE_WRAPS) reaches the Domain via `DomainMap.angle_wrap` -> `coordinates()`; validate `_check_angle_wrap` (warning
  when 2 pi / size_y is not whole, error for antiperiodic unless even); `mapped_domain_issues` errors for features on
  `DomainMap.wrapped_faces()`; `definition_helpers.antiperiodic_wrap` counts as an anchor in heat / potential
  `_has_anchor`. The canvas drops the two wrapped rays (`_EvaluatedModel.face_segments`).
- Cylindrical (r, phi, z) sectors ('polar' + size_z, heat only; spherical + size_z raises in `domain_map`): the
  DomainMap is the (r, phi) plane, z unmapped; a padding-free polar Domain3D (`MeshEstimate.domain_map` set,
  `_mapped_z_axis`). Every 3D xy mask goes through the mapped rasters (`rasterize_mapped(..., z=)`,
  `rasterize_mapped_box`: full (nx, ny) rasters per cross_section_key, no xy cropping before the raster), never
  `rasterize_box` / `box_window` on the logical host centres; x, y values of 3D builds come from
  `BuildContext.plane_coordinates` (a new 3D x, y consumer uses it too); region windows are the whole grid. Session
  cuts at constant z only (x / y planes raise; validate refuses x / y stored_cuts and 3D segment line curves);
  `evaluate` runs the whole-volume path (`by_planes` False) over `window_metric_3d`; `volume()` centres are (r, phi,
  z). Not yet: GUI (canvas / scene_3d / side views draw the logical box), meridional cuts at fixed phi. The CPU case
  of physics/heat/selftests_cylindrical_3d.py imports model.meshing / validate / geometry inside its check.
- 1D models (`DomainSpec.size_y` None; root file "1D models"): `geometry.domain_size` returns
  `LINE_SPANS[coordinates]` as size_y; `domain_map` -> `_line_domain_map` (refuses size_z, angle_wrap, angle_min != 0,
  a slab radius_min): a `DomainMap(line=True)` on every kind, so the polar path carries it: meshing `_mapped_plan` (one
  y cell, no mesh axis 'y', x graded by MeshAxisSpec x); masks on the mapped centres (radius_min + logical x, 0):
  objects select cells whose centres lie on the drawing's x axis (circles about the origin = shells / layers,
  rectangles across y = 0); `to_logical` gives the cell's y centre, `contains` needs |y| <= tolerance, `extent()` is
  the strip |y| <= size_x / 2 (previews: one pixel row at y = 0), `logical_bounds` the x range, `zero_area_faces` the
  centre x_lo of a cylinder / sphere with radius_min 0. Session `field()` / `display_field()` return
  `_line_field_data` ((wx, 1) cells, x_centers = slab x or r, y_centers [0]; native changes nothing); evaluate weighs
  by metric volumes (per m^2 / per m / whole sphere). validate: `check_physics` compares `spec.dimensions`,
  `_check_coordinate_grid` asks planar on a slab, interface selections and conformal boundaries refuse / ignore 1D,
  heat refuses out_of_plane_convection and the noise features. Not yet: GUI (forms, canvas, Add physics:
  `schema.type_spaces` ignores 1, GUI solver_misfit calls pass no grid), 1D EM (slab_modes), point-source units in 1D
  (heat P is per the 1D total: W/m^2, W/m or W; its help says W/m).
- Spec: pass the registry to renames (typed 'ref' / 'names' params, field names). Geometry objects and selections
  share one namespace minus `RESERVED_ENTITY_NAMES` (FACES + 'all'). Material `dispersion` None inherits the library
  poles, [] none. Composite geometry consumes its inputs (the industry-standard default): an object feeds one composite
  only and is no longer selectable on its own; a second use needs a duplicate.
- Arrays (kind 'array': base, nx / ny / nz, pitch_x / _y / _z; nz / pitch_z in `Z_PARAM_NAMES`): `resolve_geometry`
  adds every copy as a resolved shape ('<array>[i,j]', '[i,j,k]' when nz > 1; a composite base is copied whole, its
  inputs '<input>@<copy>'; `geometry.is_array_copy`) right after the array, values = base + i pitch (exact float
  sums, so written-out copies mesh identically), and sets the array's values['inputs'] = the copies (base first, i
  fastest, then j, k). So code iterating the resolved dict (model_pslg, size fields, model_solids, canvas outlines)
  sees every copy, code walking composites treats an array as a union (`geometry.UNION_KINDS`; `composite_inputs` of a
  resolved array = its copies, of spec params = [base]), and code listing spec objects from the resolved dict must
  skip copies (canvas picking / consumed, scene_3d tops, `_section_preview`). rasterize / contains_points_3d of >=
  `_BUCKETED_COPIES` copies bucket points by the pitch arithmetic (`_array_points`); bounding_box / z_extent read the
  first and last copy; cross_section_key the z column of copies. validate warns on overlapping neighbour boxes
  (`array_overlap`). A prism end exactly on a cell centre is decided in float32 by `_section` (the build) and float64
  by contains_points_3d: tests comparing them keep ends off centres. Estimate rasters
  (`triangle_size_field`, `tetrahedron_element_density`, swept `_with_edge_sizes`) group distance sources by size
  (`meshing.SizedUnions`: one distance transform per size of the OR of region masks / outlines, each rasterised on its
  own window), so curved copies share one transform. Not yet: `smallest_side` takes the array's whole box (auto
  sizes); the fem mesher's PSLG crossing search over 1e4 curves is untimed.
- `material_preview` rasterises float64 centres on the CPU; build masks use float32 clamped centres: cells on an exact
  boundary can differ. The empty-object warning (objects selected by materials, features, derived values) and the
  material-override warning (a material not on the preview re-judged on its bounding box) rasterise the MeshEstimate
  float64 window centres (above 1e6 cells the first is skipped, the second falls back to the preview with a
  'probably' wording). Grid masks go through `geometry.rasterize_window` (1D non-decreasing host centres -> (window
  slices, cropped mask); primitives tested only on their bounding-box cells widened by a float32 margin; union inputs
  and difference subtracts whose box misses the centres, and array copies outside the pitch range of the centres
  (`_array_inputs_near`), are never recursed into: a new composite kind keeps that pruning exact), never `rasterize`
  on full meshgrids; `rasterize` stays for point probes.
- Per-physics meshes (root file "Meshes per physics"): `ModelSpec.mesh` is NOT a field: a property = meshes[0] plus
  the constructor keyword `ModelSpec(mesh=m)` (an __init__ wrapper, not an InitVar: `dataclasses.replace` would pass
  the old meshes[0] back in). Lookups are ModelSpec methods (`mesh_of`, `physics_on(name)`, `used_meshes()`), not
  meshing (definitions may not import meshing). A legacy JSON 'mesh' object loads as meshes = [it] (`from_dict`, the
  one rule; both keys raise). Every meshing entry takes a mesh name, None = `default_mesh_name` (first used mesh):
  `estimate_mesh(..., mesh=)`, `estimate_meshes` ({used name: estimate}), `make_domain(..., mesh=)`; requirements,
  auto sizes, uniform-feature fallback and padding come from the physics on that mesh only. Their `parameters` are
  the evaluated {name: float} (`evaluate_parameters(spec.parameters)`), never the spec's: self-tests pass `{}` only for
  parameter-free specs. `estimate_memory(mesh_estimate=)` takes {name: estimate} or one estimate, one part per used
  mesh (labels "mesh '<name>': ..." only when several) plus `mesh_transfers` of cross-mesh couplings. Auto mode per
  mesh: with an EM physics (family 'em') or wavelength_min the wavelength rule, else physics-controlled
  (`physics_objects` / `smallest_side`; laminar_flow uniform) (`_auto_rule_sizes`); a set dx is the smallest cell
  size: `_auto_sizes` scales every rule size by one factor so the finest applied one equals it (points_per_wavelength
  then unused), dx None = the rule alone (MeshSpec.dx is optional only in auto: `mesh_values` and validate refuse
  None elsewhere; an auto MeshSpec / JSON that should follow the rule sets dx None / null explicitly, the dataclass
  default 20e-9 counts as set); `MeshEstimate.auto_rule` says which, `auto_dx` /
  `auto_refinements` what it chose, scaled (the GUI's fill-in); `RefinementSpec.automatic` refinements are skipped by
  `_explicit_refinements` in auto mode only. Validation: `ValidationContext.meshes` / `mesh_errors`; physics / feature / coupling (source
  physics) callbacks run inside `vctx.on_mesh(name)`; mesh issues at ('mesh', name), memory at the ('mesh', name) of the largest part. One-mesh
  rules (`_check_one_mesh_rules`, coupling / feature-ref checks): family 'em', `CouplingType.mesh_transfer` None,
  `_ONE_MESH_STUDIES`, Param ref_kind 'physics'.
- Per-mesh build: `builder._build_on_gpu` makes one Domain (`domain.mesh_name` set) and BuildContext per used mesh;
  they share the model-wide state (`_SharedBuild`: log, warnings, sim, solver_records, time_updates,
  conformal_locators, grid transfers; contexts by mesh name held weakly: the first mesh's context (the Session's)
  keeps the others in `other_meshes`, so nothing cycles). A new build step touching a physics' cells takes
  `ctx.for_physics(name)`, never the first context. Per mesh: the MaterialGrid of `meshing.mesh_material_types` (its
  physics' types plus partner types of cross-mesh couplings with `partner_materials`; `_mesh_memory` counts the same
  set), static materials, state-dependent updates (`notify` = that mesh's physics; a state of a physics on another
  mesh enters through `ctx.transfer`), the subpixel geometry; the GeometryLocator is one. Ledger owners of per-mesh
  stages get ' <mesh>' only with several meshes. Cross-mesh couplings: `create` runs with the context of the
  transform mesh (`mesh_transfer.transform_side`: 'field' target, 'value' source); every term must be a CouplingTerm
  from source to target (else build error) and gets `set_mesh_transfers` with the cached transfer (`transfer_kind`;
  conservative r weights from the target solver's `axisymmetric`).
- Per-mesh Session: `Session.domain` = the first mesh's, `domains` {mesh name: domain}, `domain_of(physics)` /
  `mesh_of` / `window_shape(physics)`. Field reads (`field`, `display_field`, `segment_profile`, `volume`,
  `cut_centers(axis, physics)`) and every `FeatureType.result` callback run on a mesh view (`_physics_view`: a cached
  shallow copy whose `domain` / `context` / display caches are that mesh's; `_require_open` follows the parent).
  `evaluate` picks `_evaluation_view` (first physics in model order whose output fields the expression names; none:
  the first mesh); fields of other meshes enter interpolated (`_from_mesh`: 'interpolate' GridTransfer cached per mesh
  pair in the parent, freed by close; non-window state values pass through); selections / metric / point cells are the
  evaluation mesh's. FieldData.mesh names the field's mesh (snapshots too); `_SnapshotRecorder` decides native vs
  display per field on its own mesh. Warm starts compare every mesh (`_mesh_key`). Triangle meshes: a renderer
  provides `fem_field(component)` -> (Lagrange DofMap, coefficients) (no device_field / state reads there); `field` /
  `display_field` give a raster over the window (NaN outside the field's elements) plus `FieldData.triangulation`;
  derived values reduce degree-4 quadrature samples of window elements (selections by centroid: `triangle_meshing.
  element_masks`; tensor fields bilinear at the samples; the stationary `_FixedPointMonitor` reads fem.renderer's
  `device_field` = the coefficient vector). Not yet on triangles: interface selections, max / min over mirror images.
- Triangle meshes (MeshSpec kind 'triangles'): `PhysicsType.grids` / `FeatureType.grids` (MESH_KINDS) gate them in
  validation; `schema.solver_grid(spec, physics)` returns 'triangles' / 'tetrahedra' (solvers.roles GRID_KINDS).
  `meshing.triangle_sizing` is the one size rule (h_max = dx or the auto rule, refinements as object sizes, h_min null
  = h_max / 100, max_ratio null = 1.3 (`MESH_KIND_DEFAULTS`), PML ring = cpml_npml x h_max on cpml faces only,
  `MeshSpec.size_points` in every mode ("Mesh refinement levels"); every size is floored at h_min, so a finer local
  size needs an explicit h_min). `triangle_size_field` + `_triangle_estimate` = the raster element-count estimate
  (`TriangleMeshEstimate`, n_cells = elements; memory, run time and MAX_MESH_CELLS depend on it; material_elements =
  the raster's window elements per spec material in the builder's paint order, as for tetrahedra). Points,
  axis-parallel lines and objects thinner than a raster pixel use the exact pixel mean across their band
  (`_thin_sources`), because the band within one pixel of a refined film holds most of its elements and a centre
  sample misses it. h is a size-field value: the mesher keeps every edge <= h, so built elements average ~h / 1.5:
  recalibrate TRIANGLE_FILL / TRIANGLE_EDGE_FILL against selftest mesh_triangle_estimate when the mesher's refinement
  rule changes. `make_domain` returns a `fem.domain.TriangleDomain`. Element meshes (triangles and tetrahedra) go through
  `meshing.element_mesh`: with `remember_element_meshes` on, the last mesh per mesh name is reused while
  `element_mesh_key(sizing, window size, resolved geometry)` matches (the build logs the reuse; its 'domain' stage then
  holds no meshing). A new input that changes the mesh but none of those three must enter the key, else a stale mesh
  is reused. On: the GUI's BackendWorker; `run_study` (`element_meshes_kept()`, state restored after, forgetting the
  meshes) around a sweep and parity sectors; `run_estimation` around its evaluations. Off elsewhere (single api / CLI
  runs, between selftest cases: a mesh made by another case would hide the mesher from the selftest cache's tracer).
  `run_study` likewise wraps every study in `cudss_factor.cudss_orderings_kept()`, the worker sets
  `remember_cudss_orderings` for its lifetime (solvers sub file "Ordering reuse"). `time_estimate.mesh_build_items`
  follows the same key (`kept_element_mesh_keys()`). `mesh_transfer.make_transfer` with a triangle mesh on either side
  returns a `TriangleTransfer` (per source space: per-element / full-mesh P1 / P2 by array length, or `apply(...,
  dofs=)`; triangle targets per element; 2D Cartesian tensor partner only); `meshing._TRIANGLE_TRANSFER_ENTRIES` (6
  per target cell / element) matches; conservative transfers hold more (size ratio squared) plus a host float64 copy.
- Meshing: no boundary_config -> no PML. Equal configs from several physics merge into the first provider's;
  different ones raise naming both (`boundary_config_difference_text`; validate reports it at the later physics).
  Merged courant_factor = smallest declared. `MeshEstimate.uniform` tests window widths,
  `ValidationContext.mesh_uniform` the full grid (both incl. z in 3D). Graded 3D: `_plan` sets `_MeshPlan.size_z` and
  `MeshConfig(size_z=)`; `_add_z_refinement` adds each refinement's z band; the cell-count guard bounds z like x / y
  (`_graded_axis_cell_bound`); estimate and make_domain both take z from `_z_cells`.
- Tensor options (nonuniform / auto; core sub file for MeshConfig): `RefinementSpec.dx_x / dx_y / dx_z`
  (`_Refinement.along / refines`: an axis refines only when its size beats the base, margin default 3 x the axis size;
  a refinement refining no axis is dropped); `MeshSpec.axes` (MeshAxisSpec per axis, checked only by `_axis_plans`, so
  validate reports its errors as 'mesh estimate failed'): max_ratio, MeshBandSpec bands (ends are snap nodes; without
  one the axis is a single segment whose whole-axis rescale moves a band by up to half an outer cell; bounds
  `_snapped_axis_cell_bound`), custom `edges` or MeshSegmentSpec `segments` (replace the axis;
  `MeshEstimate.custom_axes`, validate warns above max_ratio); `outer_dx` / `outer_ratio`. Any refinement, band or
  custom axis keeps the grid graded (no uniform fallback). `MeshEstimate.line_blocks` / `line_block_hint` (2D: cost
  n_blocks b^3 per block axis) go into check.json mesh and the GUI estimate (only when a line_block solver fits a role).
- nonuniform / auto fall back to `Domain.uniform` when no refinement is finer than base*(1-1e-9) or an enabled feature
  `requires_uniform_mesh` (`note` says why). Auto index = max over permittivity arrays of |sqrt(eps(omega) mu_r)| at
  wavelength_min (`background_permittivity` (owner, else Drude eps_inf) through `dispersion.complex_permittivity`), so
  metals mesh by skin depth (gold ~11 at 1.55 um); only wavelength_min is evaluated; a non-constant eps / sigma_e /
  mu_r / pole raises. The cell-count guard (400e6, 50,000 per axis) runs before meshing. `MESH_ESTIMATE_ERRORS` are
  model errors; anything else is a bug.
- Triangle meshes (triangulate): `triangle_meshing.triangulate(spec, parameters, registry, mesh)` = sizes,
  `mesher_size_field` (fem SizeField form of `meshing.triangle_size_field`: keep the two size rules equal),
  `model_pslg` (every 2D primitive outline incl. composite inputs, lines, points), the fem mesher (gmsh when `mesher`
  'gmsh'), the PML ring of `TriangleSizing.pml_widths`, tags by `element_masks` (rasterize at centroids, ring
  centroids clamped into the window). Point tests go through `contained`: a union of >= 4 inputs tests each input
  only at points in its bounding box (KD tree), bit-identical to rasterize. It is the only reader of shape internals
  for the mesher. `mirror_x` / `mirror_y` (mirror line = window centre): `_mirrored_parts` meshes the half window
  (geometry plus mirror images, symmetric size field), rings its outer faces, then `fem.mesher.reflect_mesh`; line /
  point tags found geometrically; the estimate takes the denser of each pixel and its image; asymmetric material
  objects only warn. `TriangleSizing.periodic_axes` (from `DomainRequirements.periodic_axes`): opposite sides meshed
  with matching nodes; no PML ring there (refused in `_element_sizing`). Mesh data and mesher rules: fem sub file.
- Tetrahedron meshes (MeshSpec kind 'tetrahedra', 3D Cartesian only): sizes by `meshing.tetrahedron_sizing` (the
  triangle rules through `_element_sizing`; `TetrahedronSizing`: pml_widths over six faces = cpml_npml x h_max;
  size_points [x, y, z, h] and face_sizes (SizedFace: window / object box faces) act as gmsh Ball / MathEval fields
  with twins in `tetrahedron_element_density` (estimate and mesh must agree; builtin: face_sizes are bands,
  size_points refused); periodic_axes = union of the physics' `DomainRequirements.periodic_axes` (gmsh periodic
  surfaces); edge_sizes = object -> smallest size below h_max of `PhysicsType.mesh_edge_sizes` (`_edge_sizes`; gmsh
  refines along primitive edges, the estimate adds `Solid.edge_distance` sources; builtin ignores them, validate
  warns)). A non-EM physics may pad with any boundary_config carrying npml_sides and differences()
  (`pml3d.PMLPadding`). Estimate `_tetrahedron_estimate` (`TetrahedronMeshEstimate`, extent (x0, x1, y0, y1, z0,
  z1)): sweep_axis 'z' -> `swept_estimate` (footprint with the 2D thin-source pixel means, `_thin_sources`); builtin
  exact (Kuhn split of `axis_edges`); gmsh TETRAHEDRON_FILL (uncalibrated) x integral of `tetrahedron_element_density` on a voxel raster (each voxel averages h^-3 of its
  winning source over +- half a voxel: centre values counted thin films ~2x). `material_elements` = elements each material paints (builder's rule): a physics whose dofs
  live on part of the mesh scales by it (solid_mechanics_3d: stiff materials). `tetrahedral_meshing.tetrahedralize` =
  gmsh of `model_solids` (spheres, boxes, z cylinders, untapered z circles, rectangle prisms of any axis, untapered z
  rounded rectangles; composite inputs included; others tagged per element: a staircase) or the Kuhn split. Plain
  objects carry their refinement size on their solids; a refined composite goes in as a `sized_regions` field tree,
  so a difference's hole is not refined and the mesh matches the estimate's composite mask. RefinementSpec.where:
  'edges' joins `TetrahedronSizing.edge_sizes`, 'surface' fills `surface_sizes` (`sized_surfaces`, estimate twin
  `SizedRegion.surface_distance`); both gmsh only: `_explicit_refinements(places=)` refuses them elsewhere (default
  'inside'; `frozen_mesh_spec` passes every place through), validate errors / warns (`primitives_without_solids`).
  Regions by contains_points_3d at clamped centroids with per-gmsh-volume majority (`by_entity_majority`); builder
  object masks and the mesh's `object_faces` both come from `tetrahedral_meshing.element_membership`, so masks and
  face tags agree. Build: the 2D element paths (x, y, z of clamped centroids; `ctx.dimensions` is 3, so 3D code tests
  `grid_kind == 'tensor'` first); boundaries `ctx.boundary_faces(entities)`, points `ctx.point_position(name)`
  (located by the physics, no nodes). Session (`TetrahedraAccess`): field / segment_profile / volume sample the
  renderer's `sample(component, points)` on the uniform lattice `TetrahedronDomain.sample_centres`; `evaluate` at
  degree-4 quadrature points in element chunks, integrals x touch_factor; refused: interface selections, max / min (and mode_volume without a
  point) touching a mirror face, fields of physics on another mesh; a NaN sample (missed location) raises. Adaptation
  only 'uniform', no cross-mesh transfers. On-disk gmsh cache: fem sub file.
- Builtin tetrahedra: `tetrahedron_sizing` fills `TetrahedronSizing.axis_edges` from `builtin_axis_edges` (repr =
  `axis_edges_digest`, so the element-mesh key follows every grading input); mesher and estimate read them through
  `sizing_axis_edges` (a hand-made sizing without them: uniform `kuhn_axis_edges`), so the estimate stays exact. The
  window part is `graded_window_edges(..., axes='xyz', keep_uniform=True)` (per axis; swept prisms take its z layers
  with keep_uniform=False to always snap object faces), then `shell_axis_edges`: nothing finer than h_max (uniform mode
  ignores refinements / axes; face_sizes act in every mode) = equal cells <= h_max, no snaps; else a MeshConfig at
  base h_max as `_plan`'s nonuniform branch (refinements + auto refinements, MeshAxisSpec plans, max_ratio = growth,
  outer grading), each SizedFace a band across its plane, every primitive's `axis_parallel_faces` snapped at h_max
  (snap merge rule: an unrefined film thinner than h_max / 4 keeps one face only). A remainder-rounded segment may hold
  cells up to ~1.5x its target. Allowed: axes / dx_x.. / outer grading (validate checks refinements as tensor ones);
  size_points refused; physics edge sizes ignored with a warning; `frozen_mesh_spec` keeps axes.
- Swept prism layers (MeshSpec.sweep_axis 'z', tetrahedra with either mesher; `model/swept_meshing.py`, fem
  `prism_layers`): `_element_sizing` sets `sizing.sweep_axis` / `layer_edges` (repr = `layer_edges_digest`) and is
  NOT graded builtin (axis_edges stay ()), so code branching on `sizing.mesher == 'builtin'` checks `sweep_axis` first.
  In-plane = the triangle rules on `footprint_shapes(shapes)` (box -> rectangle, z cylinder / sphere -> circle, x / y
  prisms and tapers -> `geometry._footprint`; vertical-walled primitives go into the PSLG, the rest are tagged per
  element) via `triangle_meshing._plain_parts(..., size_field=)`. A refinement's dx sizes in-plane over the object's
  whole column and also grades the z layers unless dx_z is given: give dx_z = dx (or more) for in-plane only. Entities
  = (PSLG face, z band, membership of region objects holding an unswept primitive or a z face that is no layer plane):
  the majority squares faceted curved walls but never wipes out per-element tags. Estimate `swept_estimate`: footprint
  triangles (`meshing.triangle_counts`) x 3 x layers. Refused (`swept_mesh_problem` / `swept_refinement_problem`):
  size_points, x / y axes, dx_x / dx_y unlike dx, x / y face sizes, where edges / surface, mirrors; x / y PML ring =
  pml_layers layers.
- Auto `wavelength_min` None = min over the physics' `PhysicsType.shortest_wavelength(...)` hooks (FDFD / em_modes:
  shortest solved wavelength; FDTD `make_source_shortest_wavelength`: gaussian f0 (1 + bandwidth), ricker 2 f0), the
  same rule as the FDTD resolution warning; hooks skip unresolvable inputs and return None; none at all is a model
  error. `MeshEstimate.wavelength_min / wavelength_source` feed the GUI estimate, check.json and a validation info. A
  sweep changing that wavelength re-meshes per value.
- Memory estimate (`meshing.estimate_memory`): MESH_ESTIMATE_ERRORS on bad input (validate's `check_memory` then stays
  silent: the input is reported at its node; another exception is an internal-error Issue at ('meshes',)). With a
  memory_budget or host_budget, `_sweep_value_problem` also estimates each sweep value (error at ('study',) above
  either budget; the 80 % warnings and the fixed 8 GiB host warning only for the base values). Every budget check folds
  the estimate by `meshing.backend_memory` (cpu backend: device bytes count as host, no GPU budget): check_memory,
  `_sweep_value_problem` and study_adaptation `_memory_problem` alike. `MemoryContext.material_regions` box cells are
  the selection's bounding box (an upper bound). `builder.build` estimates mesh and memory once (`_cached_estimates`)
  and passes them to validate and the ledger check; only `make_domain` plans the grid again. Against a GPU budget the
  judged estimate is `meshing.fitted_memory_estimate` (an over-budget study default resolves without its study,
  `plain_roles`: solvers sub file "Model path"); `_sweep_value_problem` fits each sweep value the same way.
- Memory estimate parts: `core.memory_estimate.MemoryEstimate` (gpu_steady, gpu_build_peak, host, gpu_run, items /
  host_items / run_items, transient_items, gpu_external / external_extra / external_items; gpu_run = device bytes the
  study allocates after the build: run_items stay once allocated (harmonic FDTD accumulators, growing probe traces,
  the FDFD system / solution, adjoint caches, fixed-point field copies) + the largest phase of transient_items
  ((phase, label, bytes): arrays freed when their phase ends, e.g. assembly temporaries, one solve's vectors, one
  feature's adjoint solve, a refactorisation that runs only at a second frequency; phases of the whole model never
  overlap, so `combine` takes the largest one, never their sum: an item alive across phases is a run item; a phase
  that runs once before some run items exist (a single solve's assembly, a released cold-start factor) books
  `phase_before_items(phase, later run items)`, so the sum stays the true peak); post-solve
  field updates (DiffusionSystem flux: `diffusion_system_estimate(flux=, dissipation=)`); Session reads cache no
  pixel-sized device arrays (a polar display image is gathered on the host); gpu_external = cuDSS factors outside the
  cupy pool alive through the run, external_transient_items ((phase, label, bytes)) = factors alive only in a transient
  phase, listed once per phase they live in (`cudss_factor.factor_device_bytes`, never a local formula; physics
  estimators go through `definition_helpers.cudss_external(phases={label: phase(s)})`, one_at_a_time for unphased
  factors released before the next is planned; external_extra = the largest factorisation's peak on top); a MUMPS /
  PARDISO factor (solvers.host_direct, host memory outside numpy) is a host item, never an external one:
  `host_direct.factor_host_bytes` (or a booked SuperLU item x `superlu_bytes_scale(name)`) + `solver_host_bytes`, its
  fill / workspace constants uncalibrated (an estimator left at the plain SuperLU size says so in its sub file). The budget
  check uses `device_peak` = max(build peak, steady + run items + the largest phase of (its pool transients + the
  factors alive in it)) + gpu_external + external_extra, `external_peak` (the external_memory record) = gpu_external +
  the largest phase of phase-bound factors + external_extra: split a solve into phases in the order the code allocates
  (mechanics `eigen_device_items`: pencil set-up before the factor, the eigs M mirror and the Arnoldi solve with it),
  never one phase holding arrays that are never live together) = model arrays
  (MaterialGrid union of `material_schema`, dynamic material updates, couplings) + every `PhysicsType.memory_estimate(
  settings, mesh, MemoryContext)` + every enabled `FeatureType.memory_estimate(values, n_selected_cells, mesh)`
  (bounding-box cells), combined as parts built in sequence plus the build caches. `validate(..., memory_budget=,
  host_budget=)` errors when device_peak / host exceeds its budget and warns above `MEMORY_WARN_FRACTION` (80 %); without
  a host budget the fixed 8 GiB host warning applies. `solver_defaults.memory_check` 'warning' turns every over-budget
  error into a warning (check_memory, `_sweep_value_problem` returning (text, level), study_adaptation `_memory_problem`
  never stopping a refinement for memory): a new budget check takes its level from `schema.memory_check_level(spec)`.
  The budget issues sit at ('mesh', name) of `validate.memory_verdict` (the used mesh with the largest part:
  `validate(memory_parts=)` = estimate_memory's by_mesh, else computed when an issue is reported). An over-budget error
  is final only on a built element mesh (`meshing.awaits_built_mesh`: any triangle / tetrahedron estimate not yet
  counts_built): a triangle / gmsh or swept tetrahedron estimate is a size-field integral
  (`meshing.has_estimated_counts`; builtin tetrahedra counts are exact, but their factor fill still needs the built
  mesh; the size integral is too low as well as too high, e.g. 0.36x on the graded 5 mm hirsch2025 mesh), so build,
  whenever a used element mesh is not built, and api.check, when
  `refused_on_estimated_counts`, first validate without budgets (other errors come before a mesher run), mesh
  (`meshing.built_mesh_estimates` -> `with_built_counts`, `counts_built`, + `estimate.dissection`: the built mesh's
  fem.dissection.MeshDissection, from which a physics estimator counts its cuDSS factor entries instead of using a
  growth law; build keeps the meshes for its domain stage
  through `element_meshes_kept`, so no second mesher run) and judge those counts (build: also its ledger check);
  plain validate reports the error with a note naming the meshes still to build. Per sweep value and per refinement level the check is still on the size-based estimate.
  The runtime guards stay either way (pool limit, the cuDSS plan check of its own estimate against free memory). Budgets: builder = `gpu_memory.budget_bytes()` minus the pool
  bytes other live sessions hold and `core.host_memory.available_bytes()` (None when unknown); GUI = the worker's last
  MemoryStatus `budget` / `host_available` (5 % tolerance); CLI check / show = host RAM (api.check only its arguments).
- Build: an exception in a step becomes ModelError with one Issue at that step's path ('parameters', 'geometry',
  'mesh', 'material', 'physics', 'feature', 'coupling', 'model' for `runner.prepare`; a solver: ('solver', node), on
  Automatic its physics). Validation warnings / infos only go to `log`; `BuildSummary.warnings` = `BuildContext.log`
  warnings (+ the estimate-check warning).
- Build memory: `build` = CPU part (validate, parameters, geometry, memory estimate) + `_build_on_gpu` (ledger stages:
  domain, material arrays per type, materials, one per (physics, role) solver, physics create, feature apply,
  material updates, physics finalize, couplings, runner prepare, build temporaries freed (negative), `finish()` books
  the rest as 'other (untracked)'). Any exception leaving `_build_on_gpu` gets `exc.build_memory` (failed_owner = the
  raising stage); build_session copies it into BuildFailed and logs `gpu_memory.ledger_text` on OOM. `peak_bytes`
  (pool reserved minus used at the start) relies on the pool being emptied before the build (`build_session` frees
  the cached blocks first: an api Results kept open leaves its blocks cached) and no `free_all_blocks` during it. A
  new build stage gets its own `MemoryLedger.track(owner)` (nesting raises RuntimeError), else its bytes land in
  'other (untracked)'. The record (`Session.build_memory`, `BuildSummary.gpu_*`, BuildFailed / OOM StudyFailed
  `.build_memory`, summary.json) is CPU data only. The builder compares it with `meshing.estimate_memory`
  (`builder.estimate_verdict`, differences within 1 MiB not judged): the estimate should sit just below, a warning
  when it is above ('over': it books what was not held) or more than ESTIMATE_SMALL_ITEMS (25 %) below
  ('unaccounted': an unlisted array). After each run `study._run_built` compares the pool's reserved
  high-water mark minus `Session.pool_baseline_bytes` with gpu_steady + gpu_run the same way
  (`builder.run_memory_record`, summary `run_memory`), and the outside-pool high-water mark of the live cuDSS solvers
  since `build_session` (`cudss_factor.external_high_water_bytes` minus `Session.external_baseline_bytes`) with
  gpu_external + external_extra (`builder.external_memory_record`, summary `external_memory`). Free bytes `free_all_blocks` cannot return (chunk remainders
  around arrays of earlier open Results) are left out of build peak and run memory (`start_free_bytes`,
  `pool_baseline_free_bytes`), floored at the used-bytes growth. Temporaries placed in those blocks do not raise
  the reserved mark, so the held bytes lie in [measured, measured + those free bytes] (`peak_unseen_bytes`,
  run_memory `unseen_bytes`): verdict and deviation use the point of that interval nearest the estimate
  (`builder.estimate_gap`; an estimate inside it is not judged). adaptation_tensor_strip_heater: its second run
  starts beside the first run's open Results and read build peak -61 % where the same model alone reads -12 %.
- Session lifetime: `Session.close()` (idempotent) clears solvers, couplings, features, renderers and caches
  (`Simulation.clear()` / `MultiphysicsRunner.clear()`); a cleared Simulation is dead. Snapshots, spec, registry,
  parameters stay; other accessors raise RuntimeError. `study.release_session(on_session, session, emit,
  baseline_bytes)`: on_session(None), `close()`, `release_all` (gc, cuFFT cache, `free_all_blocks`); with emit, pool
  use above `pool_baseline_bytes` + 1 MiB logs a 'GPU memory leak' warning (process-wide counters: a later live
  session gives a false leak; api checks only when no later live Results exists). The run-start release in
  `_run_single` skips the gc.collect when neither the pool nor cuDSS holds live bytes (the last close collected).
- GPU OOM = cupy OutOfMemoryError or a CUDA / library error naming an allocation failure (host MemoryError is
  ordinary): Issue at the failing step's node (outside any step ('model',)), memory released before BuildFailed /
  StudyFailed (`out_of_memory=True`), run_study closes the run's session, a derived value gives NaN.
- Conformal boundaries: when `schema.conformal_boundaries_physics` names any physics, the builder makes one
  `boundary_locator.geometry_locator(ctx)` (host only: resolved shapes + the materials' expanded selections) and hands
  it to each named solver by `set_conformal_boundaries` right after create, before its features (the solvers keep it;
  Session.close frees them); `BuildContext.conformal_locator_of(name)` gives it during create already (2D FDTD
  conformal pole weights). Honoured 2D grids: `PhysicsType.conformal_coordinates` (default cartesian; heat / potential
  also polar, spherical); on a mapped window the locator maps logical coordinates by `DomainMap.to_physical` before
  testing, so core.cut_cells works on logical cells unchanged; `material_test(index)` = the effective region of one
  spec material. Tests use `geometry.rasterize` on float64 points (a centre within float32 rounding of an edge may
  disagree with the build mask; callers fall back to the staircase link). 3D (`GeometryLocator(dimensions=3)`,
  em_fdtd_3d only): `geometry.contains_points_3d` (the tetra tags' test) = the per-height `_section` rules per point
  (prisms half-open along their axis with the taper offset per point, boxes / cylinder_z half-open in z, spheres
  closed, composites recursive); a new 3D kind or prism option goes there (the build's x / y tapers sample 256 levels,
  the point tests are exact).
  `conformal_memory_estimate` is added by `meshing.estimate_memory`. Perimeter flux features reach the outline through
  `definition_helpers.outline_pieces_of` / `outline_full_grid` (None without a locator or in 3D: the staircased
  `perimeter_flux_weights`, metric-weighted on polar). Conformal model self-test cases run conformal_boundaries on only,
  with tolerances below the staircase errors, so a silent staircase fallback still fails: keep it so in new cases.
- Subpixel smoothing: when `schema.subpixel_smoothing_physics` names any physics, a 'subpixel geometry' ledger stage
  after 'materials' runs `subpixel_geometry.interface_geometry(ctx)` (raster strips of STRIP_SAMPLES freed at once;
  issues at ('solvers',)), then each named solver gets `set_subpixel_smoothing` right after create (buffers book under
  the physics stage). The geometry is a local of `_build_on_gpu`, never on ctx / Session (solvers own it).
  `subpixel_memory_estimate` is rough (bounding-box perimeters over the finest cell), inside the ledger tolerance only
  because the arrays are small. 3D (`build_interface_geometry_3d`): the 2D builder runs once per `cross_section_runs`
  run on a `_PlaneView` at the run's height and its sets are lifted to the run's layers (E_x / E_y faces take 1 -
  stored n^2: the 2D sets store the hz component's); run boundaries sample cell-centre columns at Z_SAMPLES heights per
  half cell (E_z faces z-normal; faces kept with fill 0 / 1 when every sample shows one material, since a dropped face
  takes the arithmetic two-cell mean; per location the highest `_SCORES` candidate wins). Walls are vertical within a
  run: tapers / spheres smoothed per layer, staircased along z (documented limit). The build is backend-neutral (2D:
  `array_module_of` the coarse index; 3D: `domain.xp`; host index work by `to_host`): a new array op there must exist
  in numpy and cupy alike (no `cp.` names, no cupy-only kernels).
- Static materials: later materials win whole cells, but a selection containing 'all' only fills uncovered cells
  (`geometry.material_paint_order` is the one rule; `paint_material_index` paints it for preview, validation fallback
  and builder, whose region masks are `index == k + 1`). Only `material_properties` of present types that the
  material sets are written (defaults never filled); a name shared by several present types uses its first Param with
  expands_to the ordered union; an isotropic property writes only arrays no later property of the same material
  writes; non-finite values are checked only under the mask. Same-named properties across types need equal kind,
  unit, min / max and flags (`register_physics` raises; `Registry.material_property_params()` keeps the first).
- `Simulation.add_material(mask, material, dispersion, **properties)` is the script copy of those rules (keep both in
  step): present types = registered types whose `material_schema` arrays all exist; Drude background + poles on every
  solver with `add_dispersive`; numbers / full-grid arrays only; later calls win cells, but a mask overlapping
  existing pole cells raises (poles are never removed). Script names and defaults follow the model (`waveform`, EM
  point `current` 1e-3 A); heat / potential take coordinates via `add_probe_at` / `add_point_heat_source_at`.
- Permittivity: eps_xx / eps_yy / eps_zz win over eps_r per array (`schema.array_owner`, `permittivity_arrays`). The
  E_z types (em_fdtd_ez, em_fdfd_ez) map eps_r and eps_zz to eps_r, the H_z types eps_r to eps_xx + eps_yy.
- `PhysicsType.optional_material_schema` {array: (default, trigger properties)}: `meshing.optional_material_arrays`
  (builder allocation and model-array estimate alike) allocates an array only when a spec material (library merged)
  sets a trigger property to anything but None; `builder.owned_property_writes` (also Simulation.add_material) skips unallocated arrays, so an expands_to
  entry may name one (em_fdtd_3d eps_xx / eps_yy / sigma_m). Consumers check `materials.property_names()`.
- Memory estimate selections (`_selection_cells`, region box_cells, feature n_selected_cells) are 3D box cell counts
  in 3D (z cells of the union `z_extent`, open prism ends clamped); `MaterialRegionInfo.box_bounds` holds that box's
  per-axis (first, last) full-grid cells (em_fdtd_3d sizes its pole boxes from them).
- A state-dependent property becomes DynamicMaterialUpdate '<material>.<property>' (every = largest stride of the
  physics it reads), ahead of user couplings; one using t too (`builder._is_dynamic`; t only: every = the smallest
  stride, a 0-d value under the mask) reads `runner.t` at each cadenced tick (the clock moves first). Code asking "is
  this property dynamic" tests state names | `expressions.TIME_NAMES` (meshing count, mechanics
  `stiffness_follows_state`, em `_permittivity_writer_issues`).
- Feature values of t (Param `time_dependent`, resolved to an Expression of parameters and t): the apply hands them to
  `BuildContext.follow_time(value, setter, solver, label, bounds_param)` (plain numbers: setter once), which sets the
  value at t = 0 and registers a `TimeDependentUpdate` (after the material updates, before user couplings) calling
  setter(value(t)) before every step of that physics whose value changed (an equal value skips the setter, so a setter
  must not rely on being re-called). A setter must be cheap and incremental (a source adds the change to its applied
  array; a face value updates in place: `DiffusionPhysicsBase.set_face_value`, `FluidPhysics.set_face_value`), never
  re-allocate per step; `definition_helpers.uses_time` tells the apply.
- resonance_monitor is one shared row (`definition_resonance.resonance_monitor_feature`) over a physics'
  `ResonanceProbe` declaration (probe of 1 to MAX_POINTS points with one joint pole fit, dt read at result time, source
  waveforms, unit, per-component sample delay in steps (a declaration, never a name rule; self-stamping probes declare
  none), band 'wavelength' for EM / 'frequency' else). A new time-stepped physics with a point probe declares one
  instead of copying the feature (numerics: core sub file "Resonance finder").
- BuildContext: `object_window_mask` (window slices + cropped mask) / `perimeter_mask / perimeter_weights` are cached:
  never modify them; `domain_mask` is fresh and 'all' includes the PML; perimeters cover window cells only;
  `selection(entities, kind=None)` skips the kind check. On triangles: masks per element ((slice of all elements,),
  mask by `geometry.rasterize` at float64 clamped centroids = the mesh's region rule), `eval_field` / painting give
  (n_elements,) arrays, boundaries `boundary_edges(entities)` (edge tags of outer faces, object outlines, lines),
  points `point_node(name)`; perimeter_mask / perimeter_weights raise there.
- `ctx.masks` (build_masks.py, `mesh_masks(ctx)`): CartesianMasks / MappedMasks (2D windows; polar, spherical, 1D),
  BoxMasks / SectorMasks (3D tensor: `paints_by_slabs` True = material_index per z-slab), TriangleMasks,
  TetrahedraMasks. BuildContext keeps the public methods and delegates; a lookup the kind lacks raises in MeshMasks;
  mask caches live on the masks object (`release_caches` -> `release`). A new grid kind = a MeshMasks subclass + its
  mesh_masks entry, next to its FieldAccess.
- `geometry.perimeter_weights` (object-perimeter heat_flux, convection, current_density, surface_charge) weights each
  exposed face by |normal component| (Sobel gradient of the mask over 3x3 perimeter cells): slanted / curved edges
  within ~2 % at any dx (fixed-stencil bias, no convergence), never the raw staircase (+27 % on circles). Thin cells
  and axis corners keep the staircase, so rectangles stay exact; a concave corner loses ~0.02 dx. Feature applies call
  `definition_helpers.perimeter_flux_weights(ctx, solver, objects)` (adds `x_face_scales` r_face / r_c on an
  axisymmetric DiffusionSystem, uncached).
- Session reads go through `session.access` (field_access*.py: TensorAccess / MappedAccess / LineAccess by DomainMap,
  TriangleAccess, TetrahedraAccess; ElementAccess holds element selections). Session checks arguments, picks the mesh
  view and delegates; the access calls back through `access.view`. A read a kind lacks raises ValueError in the base
  class; a cross-mesh read asks the SOURCE view's access (`values_on_window(target, ...)`, `values_at(points)`). A
  new grid kind = a FieldAccess subclass + its entry in session.ELEMENT_ACCESS (or the tensor choice in
  `Session.access`); display caches live on the access, so `_mesh_view` / close reset `_access`. FieldData /
  LineProfile / VolumeData live in field_data (session re-exports them).
- Session (GPU-owning thread only): `field()` native cells with centres; `display_field()` native on uniform meshes
  up to max_pixels, else nearest-cell resampling on square pixels (graded always); `canvas_field()` = `field()` on
  graded Cartesian tensor planes up to CANVAS_NATIVE_CELLS, else `display_field()`. Snapshots take `field` when the
  field's mesh window has <= 250k cells, else `display_field`; results_io saves snapshot centres when present.
  `evaluate` on window arrays ('mean' area-weighted, 'integral' per unit z length, parameters shadow fields; a solver
  with `axisymmetric` True weights by 2 pi r_c, `_ring_weights`); 3D `_evaluate_planes` evaluates one z plane at a
  time (`_plane_namespace`): field expressions are elementwise, so a new non-local function would break this. Failing
  derived values give NaN plus a warning. dft results without freq_index hold metadata only.
- `FieldInfo.requires_study` (e.g. harmonic FDTD phasors, 'frequency_domain'): `Session.available_fields` and GUI lists
  drop it in other studies, validate rejects it as a stored field there, its `device_field` raises ValueError outside
  the study (derived values: NaN + warning). It stays in `output_fields`.
- 3D field access: every 3D read goes through `TensorAccess._plane_source(renderer, component)` -> plane(axis, window
  index) (the renderer's `device_plane`, else slices of ONE `device_field` volume per read, never per plane).
  `segment_profile(component, start, end, sampling)` -> host `LineProfile` (`analytic.segment_cells`: two samples per
  finest cell, NaN outside; 'nearest' = containing cells, 'trilinear' = 8 surrounding centres, constant in the outer
  half cells; one gather per touched z layer; no mirror unfolding); `volume(component, max_bytes=VOLUME_MAX_BYTES)` ->
  host `VolumeData` one z plane at a time (ValueError above max_bytes before any fetch). `StudySpec.stored_cuts`
  [[axis, position]] (3D only, `_check_stored_cuts`): `_SnapshotRecorder` stores the stored fields per cut in
  `Snapshot.cuts[i]` (same 250k-cell rule per plane, counted in the 1 GB limit; `study.stored_cut_planes`), the summary
  lists the recorded planes, the estimate adds one 'stored-cut snapshot planes' transient item (phase 'snapshot cut fetch'). A closed / loaded Results
  has no `field(plane=)`, only `snapshot_field(..., plane=)` (nearest stored cut, or an index).
- Shared reads of one state: `Session.shared_field_reads()` (Session.derived_values, each derived-series point) keeps
  every 2D tensor window field per (mesh, physics, field) for the block (`TensorAccess.window_values`; 3D volumes are
  not kept); `shared_plane_reads()` keeps one `_plane_source` per (mesh, renderer, component), so `_SnapshotRecorder`
  takes all cuts and 3D-segment profiles of one stored field from one volume (one block per field: one volume held at
  a time). Code inside a block must not step or re-solve; a blank physics' snapshot field is a NaN copy of its last
  stored FieldData (no read).
- Validation never raises (a callback exception becomes an 'internal error in <owner>' Issue). A type's validate runs
  after the generic checks (unknown keys, visible param values, 'ref' / 'names' targets, selection kind / object_kinds
  / counts, `requires_uniform_mesh`): do not repeat them; ValidationContext helpers return None / [] for problems
  reported elsewhere. Model-wide material checks run once, in the first physics of each type (EM: first of family
  'em'); domain size is reported at ('geometry',). Messages name the offending value (`short_repr`) and the valid
  alternatives. Other checks see only the base parameter values; `_check_sweep` re-runs parameters, geometry, domain
  size, `estimate_mesh` and (with a budget) `estimate_memory` per sweep value (when the base passes them).
- Sweep values: read `SweepSpec.point_values()` (custom `values` for spacing 'list', else count linear / log points
  lower..upper with exact end points), never `sweep.values` directly (kept but unused with a range spacing).
  `study.sweep_values_of` evaluates them over the base parameters.
- Symmetry check (`symmetry_check.symmetry_issues`, `check_symmetry` at the physics' path): per mirror face (bc
  symmetric / antisymmetric, else the x_lo axis of an axisymmetric physics) material objects drawn beyond the plane
  are sampled on a grid mirrored about it (160 per side and axis in 2D, 40 in 3D) and compared with the mirrored inside
  (one-sample tolerance; an object reaching beyond by <= 25 % of its inside reach or to the far window edge is an edge
  extension, compared only within its reach); a warning names the objects. Not checked: x / y material expressions,
  sources.
- Worker: the only thread touching cupy or a Session in the GUI process; events are CPU dataclasses; Progress,
  LiveFrame, LiveResults, MemoryStatus are overwrite slots (`take_latest`), DeviceInfo queued. Without a usable GPU
  DeviceInfo / MemoryStatus are kind 'cpu' (host RAM), no budget; a job whose `solver_backend(spec)` is 'cpu' skips
  the GPU budget refresh. LiveResults timeseries above 4000 samples are min / max per bin (`decimation_stride` > 1):
  never FFT them (the GUI waits for StudyFinished). Every job first `release_session`s the previous session;
  `call(fn)` runs between jobs and at study checkpoints (each new session bumps `Built.session_serial`; WorkerBridge
  passes the last delivered one and the worker refuses calls for an older session); `call(fn, needs_session=False)`
  runs host work without a session (GUI triangle mesh preview); cancel / pause / resume act only on a running study.
  Each job ends with exactly one final event (Built/BuildFailed, StudyFinished/StudyFailed); a run job may first emit
  Built/BuildFailed per sweep value. A new study sets the summary `tick_unit` (names its `ticks`; `_run_built`
  defaults 'step'; printed by `study.tick_count_text`); `Progress.tick_unit` names a progress tick.
- Progress stages: every built-in study run creates one `study._StageProgress(emit, names)` (only the stages this run
  has) and hands `stage_emit(name)` to each stage function as its `emit`; a stage emits its own fraction 0..1 and wall
  time, the wrapper sets `stage_fraction` / `stage_elapsed` and the run `fraction` (equal share per stage) /
  `wall_elapsed`. A new study type or stage does the same, else the bar restarts at 0 per stage. Messages name the
  physics (`_physics_text`); `solve_stationary`'s loop fraction is `_fixed_point_fraction` (decades of relative change
  toward the tolerance, never decreasing). Solve-counting stages use `_count_progress`.
- `run_study` never raises and returns the last session it built that is still alive; a sweep first closes the
  previous value's session; a completed sweep whose last value failed closes that session (`session_live` False); a
  non-OOM failure of a single run keeps its session. The sweep table lists failed values in `failed` [{index, value,
  message}] (study status stays 'completed'; api reports status 'partial', CLI exit 1). A feature result a value cannot
  produce (`_feature_results`: an FDFD probe in an eigenfrequency sweep) is left out with one warning per key, never
  fails the value. The summary carries `parameters` (of the session its values come from) and `sweep_index`. Loops
  that stop on a cancel between items test `control.checkpoint_cancelled(service)` (RunControl.checkpoint as a flag).
- `solve_stationary` returns `fixed_point` {converged, iterations, relative_change, tolerance} (None without a loop);
  stationary / frequency_domain summaries carry it and the sweep table a `converged` per value. Non-convergence keeps
  status 'completed'; `api._final_status` only notes it in the message.
- Time-dependent loop per ~0.1 s chunk: checkpoint, due snapshot, `runner.advance(n, min(t_final, next_output))`, one
  `Device().synchronize()` + 1 ms sleep, Progress, live output; `runner.flush()` at the end (also when cancelled)
  before the final snapshot and derived values. Intermediate snapshots stop above 1 GB. Stationary: `runner.prepare`,
  then `solve_stationary(session, physics_names, ...)` (also the frequency_domain background): `_stationary_roles`
  splits the physics into sinks (no coupling reading them writes a remaining physics; a DynamicMaterialUpdate counts
  as writing all) and the loop; Gauss-Seidel sweeps of the loop in runner order (`step(0.0, inf)`, each coupling
  `apply_now`'d right after its last source steps) until the loop's primary fields reach a fixed point (change /
  max|f|, `_FixedPointMonitor`: one window copy per loop field (a run item); heat T compared as `T_rise`: float32
  absolute T quantises at ~3e-5 K); a non-finite change raises (NaN compares False, so it would run all
  max_iterations); `_GROWTH_STOP_ITERATIONS` (3) non-decreasing changes in a row from iteration 4 stop with a warning
  (converged False; also a noise-floor stall); then one sweep of the sinks upstream first (no loop: that sweep is the
  single pass; a cancel skips it); never calls `Simulation.run`. Picard with study `relaxation` w (default 1) on the
  loop-back couplings only (`_wrap_loop_back`: target stepped at or before the coupling in the sweep, every DMU): each
  term injects previous + w (new - previous) via `transfer_value` + `write` (the first transfer is unrelaxed, so the
  change is divided by w from iteration 3). Study setting `anderson_depth` m > 0 (opt-in) replaces that by
  `_MixedTransfer` + `_LoopBackMixer`: each loop-back part writes its new value at once (plain Gauss-Seidel inside the
  sweep) and records a host float64 copy; `mix()` at the next sweep's start re-injects every part with the
  `core.damped_iteration.AndersonHistory` step (each part scaled by its largest magnitude, complex as real pairs), so
  the last sweep's own values stay injected when the loop stops. A changed set of parts restarts the history; the
  change is divided by min(w, 1). Opt-in because a secant method can converge where Picard cannot: an unstable branch
  or an unphysical fixed point past a runaway (analytic_electrothermal_runaway at J_ratio 1.05: negative
  resistivity). With a loop-back coupling the loop never converges in iteration 1 (`_has_loop_back`): that sweep
  solved with slots from before the solve (warm-start seed, previous wavelength), so a stale fixed point would show a
  change of ~0 (slug_flow_graetz returned the previous value's T every second sweep value). Without acceleration
  oscillating loops with |g| > 1 need w ~ 1/(1 - g). Each sweep value starts cold unless study setting `warm_start`
  (stationary, frequency_domain): `_run_sweep` keeps host copies of `warm_start_fields()` and, on an unchanged mesh,
  `seed_warm_start` writes them into the next build (free cells only via `DiffusionSystem.seed_unknown`), then
  `apply_now`s every coupling whose source_fields are all seeded slots. A solver with `warm_start_extrapolate = True`
  gets float array fields linearly extrapolated in the swept value from the last two states on the same meshes
  (factor clamped to [-1, 1]; non-array / integer fields pass unchanged). Never keep device copies across sessions
  (release_session reports them as a leak). Each sweep value builds a spec copy with `sweep` None and without the
  target-'sweep' analytic curves (api / GUI evaluate those on the full spec).
- Reused sweep builds (`SweepSpec.reuse_build`, `sweep_reuse.py`): `reuse_issues(spec, registry)` = what refuses it
  (time_dependent / optimization studies, parity_sectors, adaptation, and every spec value reaching a live parameter =
  the swept one + its dependents (`live_parameter_names`), except feature Params flagged `follows_parameters` with
  parameter-only values (no x / y / z / t), derived values / curves / references and study settings); validate
  `_check_sweep` reports them at their node paths, `_run_sweep` sends a refused spec through `_fail_sweep_setup`. The
  first build gets `live_parameters` (build_session -> builder.build keyword; a custom build callback must accept and
  pass `**options`, as api's `_BuildRecorder`); a flagged param's apply sets its build value, then calls
  `BuildContext.follow_parameters` (mechanics `follow_load` / `follow_held_values`), which registers a
  `ParameterFollower` only for a live raw value on a `solver_hooks.RerunSolver`. When every physics is a RerunSolver
  (`non_rerun_physics`, else an info line and one build per value) the session is kept: per later value
  `Session.set_sweep_value` (spec parameter and parameters dict changed IN PLACE: contexts and mesh views share them;
  followers re-set changed values incrementally) + `reset_run_records` (snapshots cleared in place, runner t / tick 0,
  every `reset_records()`), then `_run_built`; kept snapshots are taken before that reset; a run failure drops the
  kept session; table `builds` counts builds. Followers keep host data only. Not range-checked per value: flag only
  unbounded params. `estimate_overheads` prices the one build, `estimate_time` still every value's run.
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
- StudySpec.stored_fields [] (the default) stores every `session.available_fields()` entry in each snapshot
  (`_stored_field_keys`: primaries first): a new renderer field must be readable after every study solve its
  `requires_study` allows, else every default run of that type fails at its first snapshot.
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
  `meshing.estimate_memory(by_physics=)` first, so a recipe's `context.memory` is its own part. Per-step totals
  (`_with_study_steps`): time_dependent ticks = round(duration / dt_sync), dt_sync = min dt_limit, a physics with
  `every` n runs ticks // n steps; frequency_domain with a finite dt_limit (harmonic FDTD) a guess of
  HARMONIC_PULSE_FRACTION_GUESS x max_duration / dt; other studies 0 steps. A recipe raising ValueError / KeyError /
  TypeError / ZeroDivisionError lands in not_estimated ('recipe failed'), never stops validation (selftest
  time_estimate_recipes fails on it). A sweep estimates every value's spec (its own mesh estimate; the caller's
  mesh_estimate is not used), labels '<parameter> = <value>: ...', clears the counters; the builder estimates
  per_run=True for the post-run `_time_check` (None without a profile or for a non-completed run). A run whose mesh
  signature (`_mesh_signature`) and parity sector an earlier run of the same estimate had gets
  `TimeContext.ordering_kept` -> `SolverWork.ordering_kept` (kept cuDSS orderings: `cudss_plan_fraction`); the
  per_run check never sets it. Stored-field snapshots on triangle meshes are study work (`snapshot_read_items`, whose
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

## Registry, templates, describe
- CouplingType source_type / target_type are None, a name or a tuple: test only with `schema.coupling_type_accepts` /
  `coupling_role_text`, never ==.
- A loop over `spec.physics` that skips unregistered types (validate reports them) takes
  `schema.known_physics(spec, registry)` -> (PhysicsSpec, PhysicsType).
- Templates: read only through `templates.template_entries(registry)` (built-ins first; duplicate key raises; a fresh
  ModelSpec by `template_model(entry)` / `make_template(key)`, both refusing a factory that returns no ModelSpec); every
  template must validate cleanly (selftest templates_validate), likewise every examples/**/*.json (examples_validate,
  api.EXAMPLES_DIR; mirrors the GUI catalog's file rule). `app.main` starts with dielectric_cylinder.
- `describe()` is deterministic (sorted sets, declared order, floats '%.6g', ASCII, no addresses / timestamps) and
  checks its tables first (`_checked_reference`): a new spec field / expression function / builtin constant / result
  kind / Param kind needs its entry in `_SPEC_FIELD_HELP` (MeshSpec fields of MESH_PARAMS: derived from the Param) /
  `FUNCTION_DESCRIPTIONS` / `CONSTANT_DESCRIPTIONS` / `RESULT_KIND_DOCS` / `_PARAM_KIND_HELP`; a new nested spec
  dataclass goes in `spec._NESTED_FIELDS`. Each rule lives in one topic; others carry a one-line pointer.
  `reference_files()` returns both texts ({file name: text}); api_reference.md's Templates section is an index
  (`_template_uses`); the JSON is in api_templates.md (header + own Contents + describe('templates'), compact
  examples/*.json layout `_json_lines`). Markdown size: selftest cases appear only in the 'selftests' topic;
  consecutive params differing only in name / label share one entry; in the full reference only, a physics / coupling
  / study text of >= `_SHARED_TEXT_MIN_CHARS` occurring twice is printed once under '## Shared help texts' with "Help:
  shared `<key>`" at each occurrence: keep repeated help texts identical word for word so they merge. Lead-ins
  repeated per entry live once in the spec notes 'Physics, couplings and study'; the header Topics line comes from
  `_topic_groups` (a new topic group goes there only).

## Import contracts, structure index
- `import_contracts.CONTRACTS` patterns are fnmatch on module names relative to the package ('*' crosses dots);
  packages are '<package>.__init__' (the root '__init__'), so `tree('core')` = ('core', 'core.*') covers a package and
  everything below, while 'fem.__init__' alone is the package file. `from x import y` resolves to x.y when y is a
  module, else to x. Scope 'module' = imports run at import time (also inside a module-level try / if); scope
  'registration' adds those inside a top-level function named `register` (nested defs count as their outermost
  function); an `allowed` entry with 'function' excuses only imports inside functions. An exception no import uses,
  or an importer pattern matching no module, fails import_contracts as stale: delete it with the code it excused.
  Imports by `importlib.import_module` are invisible to the check.
- Both modules use the standard library only and read the package's .py files with ast (no package module imported,
  so a module that fails to import still gets checked).
- `Registry.selftests()` runs the `register_selftests` loaders once, under a lock (a loader never calls selftests());
  a loader that raises registers none of its cases and the next call retries every loader, so a broken selftests
  module fails `selftest` / `describe('selftests')`, not get_registry.
- `structure_index` takes the first sentence of each module docstring: the first paragraph up to '. ' + a capital /
  bracket / quote (not after e.g. / i.e. / vs / cf / etc), cut at MAX_SUMMARY_CHARS; a missing docstring shows as
  NO_DOCSTRING. The file lives beside structure_conventions.md, so outside a repository checkout
  structure_index_current passes with a note.
