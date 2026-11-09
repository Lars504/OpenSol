# open_sol/gui — conventions & gotchas

Sub file of the root `structure_conventions.md`. GUI gotchas.

- Thread rule: the GUI thread calls only model.spec / schema / expressions, geometry with numpy, validate,
  `meshing.estimate_mesh`, `time_estimate.estimate_time`, templates, io, the registry and `gpu_memory.format_bytes`; GPU
  work goes through `WorkerBridge.call(fn)` (a return value holding cupy arrays within 4 references fails the task).
- Bridge state: `WorkerBridge.results_spec` (copy of the built spec) is cleared by BuildFailed and by StudyFailed /
  StudyFinished with `session_live` False (BuildFailed always leaves no session: build_session closes a session whose
  tail failed); `last_build_summary` / `last_study_summary` (None after Built, a failure or a dead session) feed File >
  Save run (`MainWindow._save_run`: `api.save_session_run` in a `bridge.call`, the `run --out` files; log.txt = the
  Messages lines since the last Build / Compute, `_job_log` capped; summary model.file = the document path at the
  request). The results panel then names that cause in its banner and skips session fetches after a sweep whose last
  value failed.
- "Results out of date" compares the content minus title, description, derived_values, analytic_curves and
  derived_curves with the document (never revisions). The Derived values tab evaluates the document's current
  definitions via `Session.derived_value` (no rebuild; a derived-curve source is built from the document's curve, not
  the built spec's) and shows a value only while its `_derived_definition` (expression, selection, operation, source, at, window, wavelength, fit, degree, a curve
  source's expression + ordered inputs) is unchanged;
  'Export table…' writes `_derived_rows` as a 'table' result. The Derived curves tab runs
  `analytic.derived_curve_plots` of a deep copy of the document's derived_curves on the worker after Built /
  StudyFinished / a 'results' or 'all' edit, skipped while a job runs (its end re-evaluates); inputs editor
  `forms.CurveInputsEdit` (a `_parse` problem blocks the commit).
- References and analytic curves are evaluated on the GUI thread from the document's definitions over
  `ResultsPanel._curve_parameters` (the worker session's parameters, fetched after Built and StudyFinished: a swept
  model's Build is at the base value; until then `built_parameters(results_spec or document spec, study summary)`), so
  a 'results' / 'all' change redraws every result view and refetches a chosen line-curve profile (`session.field` at
  the curve's plane, else the Field tab's z cut (mid-height while a side cut is shown), whose change refetches it too;
  a `snapshot` curve reads `session.snapshots[i].fields`, its index resolved on the GUI thread from the summary labels).
- Result curve plots (every tab but Field, DFT, Derived values, Parity sectors and the far-field polar plot) are
  `plot_view.ResultPlotView`s made by `ResultsPanel._plot_view` (exports stamped by `_export_metadata`; None without
  results_spec: curve-data export refused). Contract: `plots()` -> [(name, xy or None)], xy = x (SI), x_label, x_unit,
  series, series_units, y_label, optional series_x {name: own x} (`plot_view.curve_xy`), log_x / log_y (applied only
  when (plot, flags) change), parameters, preview (live min / max: curve-data export refused); x_label 'Wavelength'
  with x_unit 'm' is shown in nm. The view owns the plot / unit-group / series combos (hidden with one entry, choices
  kept by name / unit), Export plot (`panel.export_path()` without asking when set, else a dialog), Export curve data
  (`io.export_curve_plots` after `exportable_xy`: series_x on the union x, NaN off each series' samples; then the
  owner's `extra_export_actions`, e.g. `_export_key`, which refuses a decimated preview) and Compare data (per view,
  never saved: 'plot' = one plot of an `io.read_curve_data` file, 'match' = its plots over the shown plot of the same
  name, 'run' = another plot of the same view; a set with recorded units on another abscissa is skipped in the error
  line).
- Drawing: colour = position of the series in the whole xy (not its unit group), all-NaN series skipped,
  `connect='finite'`, markers or an own `pen` from `curve_options`, then `overlays(plot, DrawnPlot)` (the owner's
  analytic / reference curves via `ResultsPanel._draw_feature_curves`, reporting to `view.add_error`), compared sets,
  labels via `plot_view.axis_label` ('', '1', a.u., dB, deg go into the label text), `panel.apply(context(name))`. Note
  line = `note(name, xy)`; error line = `set_error` (kept) + this draw's `add_error`s. `request_redraw(live=)`
  coalesces (singleShot 0; live: at most every 200 ms); a hidden view only marks itself dirty and redraws on
  showEvent: owners call it on every data change, never draw themselves. Overlay pens use `SERIES_COLORS` or a colour
  off the palette; PlotPanel keeps the owner's colour unless a curve has its own.
- Per view: Probes = time view (stationary iterations get a plot 'Probe iterations' of their own) + FFT view, live
  from `live_results`; Spectra: 'Ratio to reference' swaps `_spectrum_plots` for `_ratio_plots` under the same plot
  names; Curves & tables: `analytic.table_xy` (panel hidden for a table without number columns); Sweep: computed once
  per study summary or `SweepValueDone` (`_current_sweep`: the finished table, else the running sweep's partial one,
  dropped by a failure, a new sweep's first SweepStep or a replaced document; `_sweep_xy`); Derived values: table +
  plot of summary `derived_series`; Parity sectors: a QTableWidget from the summary's parity_sectors (a sweep: the last
  value's sectors); Line profile: a changed line-curve list while no line is drawn selects the first curve; a profile
  carries start / end / plane / snapshot / resampled (True on a display grid, False on native cells) / mid_run into its
  xy and 'Raw profile…'.
- `plot_controls.PlotPanel` (tabs Axes / Curves / Labels / Exports; style groups `_LABEL_` / `_AXIS_` / `_EXPORT_` /
  `_LEGEND_STYLE_GROUPS`) ends every redraw with `apply(context)`. Axis limits (view coordinates are log10 on a log
  axis) and tick steps (`tick_step`, read by `_style_axes`) belong to that view only, not saved; an axis is fitted only
  when its log toggle or limits change, so a redraw keeps a mouse zoom. Curves tab: per-curve settings kept per
  context, curves matched by name (unnamed 'curve <n>'), not saved; apply refills the table only when (context, curve
  keys) changed, never from a cell widget's own signal; the export takes widths / sizes from these settings; a
  one-sample curve on Auto marker gets 'o'. The `viz.curve_figure.PlotStyle` is ONE object per QSettings key
  ('curve_plot_style') shared by every panel (`_SharedPlotStyle.changed` refreshes all), so a style value must suit
  every plot: data-dependent settings go to the Axes / Curves tables.
- PlotPanel screen preview: `_ArticleAxis` items (a fixed tick step giving more than `_MAX_FIXED_TICKS` falls back to
  automatic in `tickSpacing`, else pyqtgraph allocates range / step ticks; automatic steps are raised 1-2-5 until the
  major labels fit, since pyqtgraph always labels the major level; tickAlpha 1 unless the grid is on, as
  pyqtgraph draws grid lines with the ticks); legend in a view-box corner ('best' re-placed on zoom / resize and handed
  to the export as that corner) or 'outside right' in a widened layout margin; curves restyled from the owner's look
  (kept per item in a WeakKeyDictionary), axis labels overridden while the owner's (text, units) are remembered; SI
  prefix off = `autoSIPrefixScale` 1 before `setLabel` (enableAutoSIPrefix(False) still scales). Export plot =
  `PlotPanel.export`, WYSIWYG: visible PlotDataItems, view range, log flags, shown axis texts, SI scales and the ticks
  `_ArticleAxis` drew last (`drawn_ticks`) go to `save_curve_figure` (matplotlib) at the widget's size / logical dpi;
  pt -> px through `_px_per_pt` so screen and file agree; never the pyqtgraph exporters (cosmetic pens stay thin).
- Canvas display grid (`canvas._display_grid`): float64 stays float64 (DFT maps are sent float64); uniform centres
  span first / last centre +- half a spacing. Non-uniform centres (graded cells, native snapshots) keep `_NativeCells`:
  edges from `viz.common.edges_from_centers` (never midpoints of centres: wrong on graded cells) and the
  `viz.common.containing_cells` rule of line_profile for image, detail, hover and drawn profiles ('resampled' False;
  also for a uniform field shown cell for cell); colour limits span every cell; `_detail_image` (above the field
  image, ignoreBounds so the A button fits the whole field) redraws the visible range +- `_DETAIL_MARGIN` after each
  zoom / pan settles (`_detail_timer`), hidden where the whole-window image already resolves the cells.
- Progress (`MainWindow._on_progress`): the toolbar bar is the whole run (Progress.fraction over its stages, times the
  sweep position); the Progress tab's `stage_bar` shows the stage itself. Time remaining = the whole sweep's, else the
  current stage's for a multi-stage run (later stages have no estimate).
- WorkerBridge keeps `device` / `last_memory_status` and re-emits them; the gauge shows pool_used / budget, on kind
  'cpu' (no usable GPU) host RAM. The build-memory table is filled from Built (`BuildSummary.gpu_*`) and BuildFailed
  (`build_memory`). Examples menu: one submenu per `<repo>/examples` subfolder (`ExampleCatalog.folders()`, no file
  read; json directly in examples/ is not listed); aboutToShow runs `ExampleCatalog.scan(folder)` (skips `*_out`,
  hidden folders; raw JSON title / description / physics types only, cached per file version), listing every readable
  file: never load_model / validate there (on a CPU-only machine validation probes cuDSS / gmsh per file and stalls the
  hover); a model is checked only when opened, like any file; menus are rebuilt
  on every aboutToShow and old submenus deleteLater'd (`clear()` keeps them). An example opens as an untitled copy.
  Entries show `ModelSpec.title`, never the file stem (examples/README.md indexes stems: its "Naming" paragraph has the
  menu path folder label (`examples._FOLDER_LABELS`) -> physics type label(s) -> title, and the title convention
  'Analytic: ...' for benchmarks); the tooltip ends with the relative path. A folder without a `_FOLDER_LABELS` entry shows its name capitalised.
- PySide6 is imported before pyqtgraph (`gui/__init__`). `app.main` sets `imageAxisOrder='col-major'` (images are
  `[i, j]`) and the QSettings names ('open_sol', 'open_sol') that results.py relies on.
- Views: defer rebuilds with `QTimer.singleShot(0)` and a pending flag; block an editor's signals before deleting it;
  mutate fns look nodes up by name; catch `helpers.USER_ERRORS` around user edits and show
  `expressions.error_message(exc)` (`helpers.run_user_operation`); a refused edit reads '<undo label>: <message>' and
  reaches `MainWindow._report_refused_edit`. File dialogs via `helpers.start_directory` / `remember_directory` /
  `ask_save_path`; Save run uses `helpers.ask_run_directory` (never a QFileDialog in save mode: it enters an existing
  folder typed as the name). `MainWindow._commit_pending_edit` clears focus before Build, Compute, saves and the
  discard prompt (an external rebuild drops uncommitted editor text). SettingsPanel rebuilds only on a node change,
  section 'all' or (external edits) the `_LIST_BACKED_NODES` forms, else refreshes in place: an edit outside 'all' that
  changes a form's structure must select another node, and every editor a non-'all' edit can change needs a refresher.
- ModelDocument: the estimate methods live in `SpecEstimates` (shared with `SweepValueEstimates`): a new cached
  estimate goes there. `spec` is replaced on undo / redo / set_spec and after a failed mutate (never cache it); edits from
  inside a document signal handler raise; `spec_dict()` = the current revision's to_dict, kept by every edit (next
  undo snapshot, out-of-date input): never modify it, never change `spec` outside mutate; `node_selected` carries a
  tuple. `mesh_estimate(name)` (None = default mesh) / `used_mesh_estimates()` / `memory_estimate()` /
  `mesh_memory_estimate(name)` / `time_estimate()` (no benchmark on the GUI thread) / `issues()` are cached per
  revision, carried over (all but issues) by an edit leaving `results_inputs` unchanged (title, description, result
  settings: `_next_revision`); a new estimate reading those parts of the spec must not be cached there. Canvas and 3D
  preview likewise skip re-evaluating on 'model' / 'results' edits (selection overlays and the title only);
  `set_memory_budget` (fed every MemoryStatus budget and host_available) drops cached issues and emits
  `issues_invalidated` on a > 5 % change. `replaced` fires only on new / open, just before `changed('all')` (undo / redo
  emit only `changed('all')`): canvas and ResultsPanel drop the old field and results (the worker session lives until
  the next build), and ResultsPanel calls `bridge.advance_generation()`: an earlier job keeps running but ResultsPanel
  ignores its signals (`bridge.event_is_current`) and drops its pending `bridge.call` results (`ResultsPanel._call`: a
  new view call must use it).
- Node paths: physics are children of ('model',); ('parameter', name) belongs to ('parameters',) and has no tree item;
  ('results',) holds the always-present `RESULT_GROUP_NODES` (not spec items, no rename / delete / move):
  `DERIVED_VALUES_NODE`, `ANALYTIC_CURVES_NODE`, `DERIVED_CURVES_NODE` (one ('derived_plot', plot) node per
  `derived_plot_names` entry: renaming sets the plot of its curves, no delete / move; 'derived_curve' nodes sit under
  it, their siblings are their plot's curves, a duplicate stays in its plot); `COMPUTED_RESULTS_NODE` = the tree's
  'Results' node (not in the spec, always `node_exists`, shown while `ResultsPanel.has_results`). `parent_node` /
  `sibling_names` live in gui/document. Same-edit rules: `add_physics` sets every = 500 when a used mesh's estimate has
  a dt and the type declares no courant_factor (and, when the first courant_factor type arrives, on existing
  non-Courant physics still at every 1; `notice` -> status bar); `add_feature` creates its geometry; `duplicate_node`
  copies a feature's own object (`_owned_object`).
- Add physics menu (`ModelTree._add_physics_menu`, rebuilt on every context menu): one submenu per `PHYSICS_GROUPS`
  entry, misfits disabled under 'Not available in this model' (tooltip = `physics_type_misfit` reason, which also
  greys out types without the model's `solver_backend`). Only the physics present restrict it: `add_physics` switches
  2D <-> 3D in the same edit when the picked space needs it (3D: size_z = size_x; 2D: size_z and every object's z params
  cleared). The Solvers form's 'Run on' Automatic label follows `cupy_installed()`; mesh memory lines on the cpu are one
  processor total (gpu_peak + host). `solver_defaults.memory_check` is edited only by the mesh form's tick box 'Override
  memory warning' (`_MEMORY_OVERRIDE`: ticked = 'warning') in its Estimate group, shown on the mesh the budget issue is
  reported at (`document.memory_verdict()`) while the estimate is over a budget or the override is on. Mesh form order:
  Build mesh group, Estimate (+ tick box), the node's issues. Built element counts: the store hands
  `built_mesh_counts` of every preview / built mesh, and after BuildFailed of the meshes the worker kept
  (`meshing.kept_element_mesh_counts`: a build that meshed, then refused the model), to
  `document.set_element_counts(name, key, counts)`; while `document.element_mesh_key(name)` is that key,
  `mesh_estimate` carries them (`counts_built`), so the memory / time estimates and issues follow
  (`issues_invalidated`, which SettingsPanel refreshes on, as for a budget change). Compute stays refused on an
  over-budget error until the override is ticked or the built counts fit.
- Meshes (S7.2; it must always be clear which mesh a physics uses): `MESHES_NODE` ('meshes',) between Couplings and
  Solvers, one ('mesh', name) child per ModelSpec.meshes entry (section 'mesh'; rename / delete / move are 'all' edits:
  `ModelSpec.rename_mesh` / `remove_mesh`; `delete_mesh(move_physics=True)` first moves its physics to the remaining
  default; a move names the default mesh in the physics on it, since meshes[0] is the default). Mesh names are their
  own namespace (`_new_mesh_name`). `add_mesh(kind, physics=None)` adds an automatic mesh (with a physics: named
  '<physics>_mesh' and assigned in the same edit); `set_physics_mesh` stores '' for meshes[0]. Every view names meshes
  through `helpers.mesh_color` (by index in spec.meshes), `mesh_summary_text`, `mesh_badge_html`, `field_mesh_name`
  (FieldData.mesh, else the physics' mesh in the built spec), `coupling_mesh_text`, `evaluation_mesh`. The mesh form
  addresses its MeshSpec by the form's node name (`_form_mesh`, inside every mutate); its 'Used by' untick = back to the
  default mesh (refused on the default mesh). Canvas `_active_mesh_name` = the shown field's mesh in the field view,
  else `document.mesh_of_node(current node)`, else the default; `_view_is_stale` re-evaluates on a change. 'Show mesh'
  (`canvas.set_show_field_mesh`) draws the document's current estimate (triangles: the built ones; after mesh
  refinement the last level's).
- Mesh refinement (StudySpec.adaptation): the off group is disabled by `settings_panel._adaptation_blocker` (=
  validation's rules, re-checked by a refresher since a parity-sector edit only refreshes); a method edit rebuilds. A
  run emits Built per level (StudyStarted after level 0's): ResultsPanel counts them in `_refinement_level`. The
  results_spec stays the submitted spec (level 0 meshes): the last level's meshes exist only as summary 'convergence'
  adapted_meshes, so the Convergence tab hands their estimates to `canvas.set_results_meshes` and `_level_label` names
  the level the results come from. 'Keep adapted mesh' = `document.keep_adapted_meshes` (one 'all' edit; the
  adaptation stays on).
- Parameter estimation (`estimation_form.py` / `estimation_results.py`): `ESTIMATION_NODE` ('estimation',) is a child
  of ('study',) that exists only while the study has one (not in `_ROOT_NODES`, no rename / move; Delete switches it
  off, a 'study' edit); switched on via `estimation_blocker` (= validation's rules) / `new_estimation`.
  `EstimationFormMixin` is a SettingsPanel base (uses `_commit` / `_refreshers` / `_context` / `_report`), list-backed:
  rows address items by index through `estimation_item`, which raises KeyError for a gone item (refresh -> rebuild);
  the off-state group's refresher raises KeyError once an estimation exists so the study form rebuilds. Validation
  reports estimation issues on ('study',) with messages starting 'estimation': `document.issue_node_path` moves them to
  the node; use it wherever issues are matched to nodes. The Estimation tab is cleared by every evaluation's Built
  until StudyFinished; 'Use fitted values' is one undoable 'parameters' edit.
- Element meshes (`mesh_preview.ElementMeshStore`, made by MainWindow after ResultsPanel): built meshes come from a
  `bridge.call` after every current Built / StudyFinished, previews from 'Build mesh' through
  `bridge.call_without_session` (no session; serviced between jobs or at study checkpoints) via `meshing.element_mesh`;
  the worker keeps the last element mesh per name (`meshing.remember_element_meshes`, on in `BackendWorker._serve`), so
  a later Build with the same key reuses it, and a preview after a build reuses the built one. Automatic previews
  (`store.request_auto_preview` from the mesh view's draw, up to `AUTO_PREVIEW_MAX_ELEMENTS` / `_TETRAHEDRA`, not
  after a failure of the same key) start after a 400 ms quiet timer that every redraw restarts, never while a preview of
  the mesh runs. Drawings are keyed by `element_mesh_key(sizing, window size, resolved geometry)`: `store.drawing(mesh)`
  returns one only while the document's key matches (never stale triangles); 'Show mesh' takes `store.built_drawing`
  (the mesh the field was computed on). The canvas field view stays a raster (tripcolor is for exported figures).
  `settings_panel._TRIANGLE_ONLY_SETTINGS` show only when the physics' mesh is in `schema.FREE_MESH_KINDS` (or while
  stored); a Mesh combo change rebuilds the form.
- Tetrahedron meshes: same store and 'Build mesh' group; the canvas draws the tetrahedra cut by the shown plane
  (`viz.common.tetrahedron_mesh_cut`, cached per data / plane) and the PML shell from the estimate. Mesh kind choices
  list only the model dimension's kinds (`helpers.mesh_kind_shown`; a stored other kind is kept so it can be changed),
  enabled by `helpers.mesh_kind_problem`; `add_mesh('tetrahedra')` picks gmsh when installed. Fields come from the
  Session's sample lattice.
- Solvers (solver_plan S4): ('solvers',) between Meshes and Study, ('solver', name) children (section 'solvers'; rename
  / delete are 'all' edits via `ModelSpec.rename_solver` / `remove_solver`, which follow PhysicsSpec.solvers). Every
  fit judgement is `document.solver_fits` (= `schema.solver_misfit` per (physics, role) in the spec's study), used by
  `model_tree.solver_candidates`, `add_solver` (assigns the new node to every fitting role still on Automatic) and the
  physics form's 'Solver' group (misfits disabled via `ValueComboBox(disabled=)`). A new node kind needs entries in
  every document.py table (SECTIONS, _ROOT_NODES, _NAMED_NODES, _TREE_NODE_KINDS, _PARENT_NODES, _SECTION_OF_KIND,
  _REFERENCED_KINDS, spec_renames, delete_node).
- Build (F7) skips the validation pre-check (ModelError -> Validation tab); Compute (F8) refuses on validation errors.
  `pause()` is ignored before study_started; `stop()` also resumes a paused study, cancels a run still queued
  (`BackendWorker._cancel_next_run`, applied after `control.reset()` when popped) and logs that a build cannot be
  cancelled.
- Canvas views: selecting Meshes / a mesh / Results switches to the mesh / field view (`_NODE_VIEWS`), leaving it shows
  geometry; any other `_set_mode` switch cancels that. View fit (`_reset_view_if_needed`): a mode switch keeps the zoom;
  it refits on a change of view plane, plane extent, scale, stretch, or the geometry / mesh view's PML cells / triangle
  ring; a new field (live frames, each sweep value's build) and another sweep value (`_view_sweep_value`: sliders,
  running sweep, results' value; a new length unit rescales the range) never refit.
- Canvas sweep sliders (`_sweep_box` in the info box, below the legend): one slider per swept parameter
  (`_sweep_axes(spec)`; one position per sweep value, the model value is no position: its handle sits at the nearest
  value, 'Model values' returns to it), shown while the study has a sweep. `_shown_sweep_value()` = the running
  value (`set_sweep_value` from SweepStep, sliders disabled; cleared by StudyFinished / StudyFailed / a replaced
  document), else in the field view the shown results' value (`set_results_sweep_value` from
  `ResultsPanel._show_results_sweep_value`: the 'Sweep value' combo's kept value, else the session's value = the last
  computed; it wins over the sliders there), else the slider choices (`_sweep_value`); geometry, material preview,
  mesh estimate, element meshes (store calls take `estimates=`) and the status line (size, memory, over budget) then
  come from `document.at_sweep_values(pairs)` (`SweepValueEstimates`: the `SpecEstimates` of
  `study.sweep_value_spec`, cached per revision and `estimate_inputs_version`). Its Build mesh previews that value
  (mesh store keeps `_KEPT_PREVIEWS` per mesh, the counts go to the document's shared `_element_counts`). View only:
  validation, Compute, forms and the 3D preview stay the document's.
- Canvas layout: the plot never changes size with the mode or the text (colour bar stays in the layout, opacity 0
  without a field; fixed tick-axis widths; the title sets no minimum width; `_place_plot` puts the widget at the top of
  the layout-free `_PlotArea` at the last fit's aspect, each view side at least `_MIN_FRAME_FRACTION` of its largest).
  The aspect is never locked: equal x / y scales only where the fitted frame allows (a thin geometry gets unequal
  ones), a wheel over an axis zooms that axis alone (pyqtgraph AxisItem), so code needing a pixel size takes both
  `viewPixelSize()` components. Banner / mesh badge / legend live in the scroll box below: a
  new canvas text line goes there, never above or below the plot.
- Canvas figure style: `canvas.plot_panel` = a PlotPanel on the canvas PlotItem (own QSettings key
  'canvas_plot_style'; placed by ResultsPanel at the end of the Field tab); fixed colour limits override the field's /
  replay range in `_update_field_image` (lowest >= highest: ignored); 'Log colour' images hold `_color_values` = log10
  of values clipped at `_field_log_floor`, levels and colour-bar axis in log10; hover readout and data exports keep the
  values. `_redraw` and `_update_field_image` end with its `apply()`. Its Export plot (and File > Export figure) is
  `scene_export`: visible ImageItems as drawn (RGBA), overlays in z order and the colour bar to
  `save_curve_figure(scene=)`; ROIs are left out. A new overlay item type needs a case in `plot_controls._scene_layers`.
- Canvas interaction: hit tests respect object_kinds, allow_outer_faces, allow_interior; select clicks are ignored in
  the field view; a draw tool switches to the geometry view (leaving it returns to select); with a selecting node
  current a plain click only toggles an eligible entity, Ctrl+click toggles 'all' (empty domain space) or selects the
  object under the cursor. Expression-positioned objects and prisms extruded along x / y get no ROI (a taper's ROI edits
  its base shape). Every new ScatterPlotItem overlay must ignore clicks (`_PassiveScatter`). Glyphs of monitors and
  `FeatureType.acts_on_hi_face` features sit on the hi face (em_fdfd_nonlinear channel features take the flag of their
  channel polarisation's type, `_channel_glyph_type`). An ROI commit keeps every param within half a pixel of its stored
  value (never re-round what a drag did not move); moved / drawn coordinates round via `_to_si` (finer of 1e-3 scene
  units and one significant digit of the pixel size); a rectangle / polygon coordinate on a uniform mesh's cell centre
  moves up one step (no float32 build tie).
- Object multi-selection: `ModelDocument.geometry_group` (>= 2 object names incl. current_node's, else ()) set by
  `toggle_geometry_group` (canvas Shift+click without a selecting node; tree Ctrl+click) or `select_geometry` (tree
  Shift+click: spec.geometry order from `ModelTree._range_anchor`), cleared by every `select` and by an edit leaving
  current_node outside it (`_publish_change`); set before node_selected fires, then `geometry_group_changed`. Form and
  ROI stay the current node's; `canvas._highlight_request()` highlights the whole group. The tree stays
  SingleSelection: `_sync_current_item` selects the group items programmatically; a right click on a group item keeps
  the group; Delete deletes them all.
- Hover highlight (one way: panels -> canvas, never canvas hover): a hovered `SelectionPicker` entry (connected by
  `SettingsPanel.rebuild` via findChildren -> `hover_entities`) and a hovered tree node (`canvas.set_hover_node`)
  draw dashed overlays through the same `_draw_entities` as the selection. Discarded forms are signal-blocked, so
  `rebuild` and the tree's `_rebuild` send the clearing hover themselves.
- Interface selections (`_draw_interface_highlight`, 2D / top view only): outlines as
  `interface_sampling.object_outline` reads them (composites: the result's boundary; window-edge parts dropped) plus a
  `_side_band` on the side it reads. Projected and polar interface curves fetch every `line_curve_field_keys` field
  (polar: `native=True`) in one `bridge.call`.
- `Session.display_field` (canvas on uniform / triangle / polar meshes, snapshots, live frames) is a display grid: a
  nearest-cell resampling sets `FieldData.native_shape`, which `export_field` writes. 'Export field data' refetches
  `Session.field` only for a field from `show_selected_field` with no build / study since (`_session_state_field`) and
  the worker idle; snapshots and live frames export resampled.
- Results field tab: the Physics combo (`_field_physics`) filters the Field combo (`_field_choices` that physics',
  `_field_infos` every physics'); a rebuild keeps the previous (physics, component) when still built, else the first
  primary field's physics.
- Parameter sweep group (`_add_sweep`): the refresher greys out rows the spacing does not use and writes the 'n runs'
  preview from `point_values()`; switching spacing seeds an empty custom list from the range, or empty bounds from the
  list's ends.
- Kept sweep snapshots (SweepSpec.keep_fields): the 'Sweep value' combo (entry 0 = the session) points the slider at
  `summary['sweep']['snapshots'][i]` (`_kept_snapshots`): those reads use the host Snapshots on the GUI thread, never
  `bridge.call`; each Built / StudyFinished resets it. With SweepSpec.reuse_build a sweep emits one Built only: views
  must not assume one per value.
- Snapshot slider (`_setup_snapshot_slider`): starts at the summary's `live_snapshot` (kept values: their last
  `Snapshot.live`), else the last snapshot, so its label names the state the canvas shows (eigenfrequency keeps
  resonance 1 live).
- Snapshot replay (Play): one `bridge.call` fetches every snapshot's FieldData of the field (`_replay_frames`, dropped
  by `_setup_snapshot_slider` via `_replay_token`); cached slider moves show frames with `color_range` over all frames
  ('Fixed colour limits' off: None). Any other `canvas.show_field(field_data)` resets the range;
  `show_selected_field`, study start and a snapshot reset stop it.
- GUI texts: shared wordings live in forms (`DERIVED_OPERATION_LABELS`, `SELECTION_KIND_LABELS`, `display_label`
  (ASCII 'lambda = <nm> nm' shown as 'λ = <nm> nm'; raw labels kept for matching and files), `format_quantity`, editor
  readback 6 significant digits), helpers.format_duration and model_tree.RESULT_SETTINGS_TIPS: use them, never a local
  copy. Tooltips show `Param.tip` (else help) and `summary` (else description), always through `forms.rich_tooltip` (Qt
  wraps only rich-text tooltips); feature types add `schema.feature_precision_note` (`model_tree._feature_gui_text`).
  Param labels get `forms.set_param_help` (full help as What's This); the full type description goes in a collapsed
  'Technical details' group. Coupling pair search: `model_tree.coupling_candidates`. field_transfer 'source_field' /
  'target_slot' become choice Params from the endpoints' state_fields / source_slots (`_coupling_params`), so a source
  / target change rebuilds the coupling form; likewise a feature's text 'channel' param becomes a choice of
  `document.harmonic_channel_names(physics)` (`_feature_params`; `add_feature` defaults it to the first channel). An
  optional 'choice' Param lists None as '(default)' first (`make_param_editor`). Text a user may want to copy (hints,
  errors, issues, estimates, banners, notes) is a `forms.hint_label` / `error_label` or a QLabel wrapped in
  `forms.selectable` (it takes click focus, so a click on it commits a pending edit like any other click); form row
  labels and the status bar stay plain.
- Polar / spherical domains (an 'all' edit): the canvas keeps the drawing plane; `_EvaluatedModel.domain_map` (None on
  Cartesian) drives `plane_extent` / `face_segments` (rays joined by `angle_wrap` dropped: not drawn, labelled, picked
  or highlighted) / `inside_domain`, outline and mesh view (`_draw_mapped_mesh`); `model.cells` is None there (no cell
  snapping, no mesh readout). Fields arrive as containing-cell images over the sector's box (NaN outside); 'Export
  field data' refetches with `Session.field(native=True)`. New objects centre on the sector's box
  (`document._domain_box`). Add physics offers only polar-capable types and gives them the domain's space
  (`schema.COORDINATE_SPACES`).
- 3D models (B1): 'Depth (z)' switches 2D / 3D (an 'all' edit); z params shown only in 3D (or while set in 2D, to
  clear them); `add_geometry` drops unset z params. A prism along x / y relabels params by model axis
  (`settings_panel._axis_params`), so an axis edit rebuilds the form (`_add_param_group(rebuild_keys=)`). Canvas Top
  view: a layer selector (`_layer_bounds`); absent objects get no outline and no hits (`_EvaluatedModel.hidden`);
  objects drawn on a layer get its exact z0 / z1. Front / Side: scene coordinates (u, v) = (x, z) / (y, z)
  (`_EvaluatedModel.scene_axes` etc.), outlines `geometry.vertical_section` at the cut, preview `material_preview(cut=)`,
  highlights `section_mask`; outlines / hits / preview belong to one view (re-evaluated on a view or cut change);
  composites have no outline; side views select and highlight only (draw tools / ROIs refused); mesh / PML use the
  graded z edges. The field view takes the shown field's plane (`canvas._field_view`; `_view_is_stale` compares the
  cut); there View combo / cut spin emit `field_cut_requested` -> ResultsPanel `_on_canvas_cut_requested`, and the row
  keeps the shown cut until the fetched field arrives; outlines, hover and axis labels follow `FieldData.axes`.
- 3D results: 'Cut plane' axis + position (from `BuildSummary.x / y / z_centers`) -> `plane` of
  `session.display_field`; live frames are the mid-height z cut; a snapshot shows the stored cut nearest the chosen
  position (`_snapshot_field` matches by `FieldData.plane`); a vertical axis without a summary `stored_cuts` entry is
  refused and the combos set back (`_set_plane_choice`), never a fallback to the z cut (it would switch the canvas view
  and refit). `_replay_key` = (field key, `_snapshot_plane()`); while a snapshot is shown (`_snapshot_shown`) a
  cut-plane change refetches that snapshot. A line curve without its own plane uses the chosen cut only when it is a z
  cut (`_line_curve_plane`), else mid-height; a 3D-segment line curve fetches `session.segment_profile` (a snapshot's
  `segments[curve name]`). Drawn profiles on a side cut carry `axes` and `plane_<axis>` into exports. 'Export volume…'
  = `session.volume` -> `io.export_volume`. 'Stored cut planes' is shown in 3D, in 2D only while set; `add_physics`
  switching to 2D clears them.
- 3D preview (`preview_3d.Preview3DPanel`, dock tabified with Results, hidden until View > 3D preview): matplotlib
  imported only on the first draw; redraws debounced 300 ms after a document change, at once on a highlight change
  (the canvas's `_highlight_request`, passed in), only while visible (hidden: marks dirty, showEvent redraws); keeps the
  mouse-turned view across redraws. Colours are the canvas's `_MATERIAL_COLORS` / `_CATEGORY_COLORS` /
  `_HIGHLIGHT_COLOR` (imported, never copied). 'Field cut planes' fetch only from its own redraw, through
  `ResultsPanel.fetch_cut_planes` (refused while a job runs or without a session). `ResultsPanel.field_view_changed(
  refetch)` is emitted by `_show_session_field`, a study end without Follow live, build failure and document
  replacement (True) and by colormap / symmetric controls (False); a new way the shown session field changes emits it
  too.
- Material form: one property group per physics type in the spec, then other registered types, then 'Unknown
  properties'; placeholders = library value, else the crystal entry's (`_inherited_placeholders`, refreshed in place).
  A 'table' property gets `forms.TableEdit` (commits on focus out). 'Crystal data' group (the {'model': 'crystal'} item
  of MaterialSpec.dispersion): edits refresh in place, add / remove rebuild. The pole group skips the crystal entry;
  'Use the library poles' is disabled while an entry exists (null cannot hold it). canvas and results import viz's
  `DARK_CMAPS` and the `viz.common` helpers: never copy them here.
- Mesh form (one per ('mesh', name) node): a kind edit rebuilds. Refinements: 'Per axis' dx_z in 3D or while set,
  'Refine' (RefinementSpec.where) on gmsh tetrahedra or while not 'inside'; meshes not graded by the tensor rules
  (`_tensor_graded`: triangles, gmsh tetrahedra; swept tetrahedra count as graded, only their z options apply) show the
  per-axis rows / Per-axis options only while set, so a kind, mesher or sweep_axis edit rebuilds. Per-axis options are
  located by (index, axis) via `_mesh_axis_at`, bands / segments by index via `_mesh_axis_item`; the Cells combo is
  derived from the data, so a switch or an emptied edge list rebuilds (switching copies the current cells as edges, <=
  `_EDGE_LIST_MAX_CELLS`, or one even segment). 'Methods on this mesh' (`_mesh_methods_html`): per physics with
  `PhysicsType.mesh_method`. Mode edits go through `_set_mesh_mode` (rebuilds): entering auto keeps dx (there the
  smallest cell size, meshing `_auto_sizes`) and drops the automatic refinements; leaving auto to graded writes the
  estimate's `auto_dx` + `auto_refinements` (automatic=True RefinementSpecs) at 4 digits, to uniform keeps a set dx
  (e.g. an example's swept cell-size parameter), else writes the smallest auto size. `_show_mesh_numbers` (every
  refresh, before `ParamForm.refresh`) reads an empty auto dx back as the rule's finest size, greys points per
  wavelength while dx is set, greys the wavelength rows outside auto and puts resolved numbers in the placeholders; an empty 'Minimum
  element size' reads back '= maximum size / 100' like a typed value (`ParamForm.set_empty_value`; the maximum from the
  estimate, else dx outside auto, so it shows when the estimate fails). Estimate adds
  `meshing.line_block_hint` only when a line_block solver fits a role (`_line_block_fits`).
