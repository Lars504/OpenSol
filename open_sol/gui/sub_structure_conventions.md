# open_sol/gui — conventions & gotchas

Sub file of the root `structure_conventions.md`. GUI gotchas.

- Thread rule: the GUI thread calls only model.spec / schema / expressions, geometry with numpy, validate,
  `meshing.estimate_mesh`, `time_estimate.estimate_time`, templates, io, the registry and `gpu_memory.format_bytes`; GPU
  work goes through `WorkerBridge.call(fn)` (worker side: model sub file "Worker"; cupy in the returned value fails the
  task). Its callbacks are dropped once the document is replaced (`advance_generation`, called by ResultsPanel on
  `replaced`) unless `keep_if_replaced` (Save run, View in browser: they act on the run they captured).
- Bridge state (`WorkerBridge.results_spec`, `last_build_summary`, `last_study_summary`; when each is cleared: its
  handler) says whether a session / finished run exists: File > Save run, View in browser and the results panel read
  it (a sweep whose last value failed leaves no session: the banner names the cause, session fetches are skipped).
- View in browser has one handler: the canvas button and the Results Field tab's button both emit
  `canvas.view_in_browser_requested` -> `MainWindow._view_in_browser` (target: `_web_page_target`). Compute runs
  `schema.with_gui_run_features` of the document spec (model sub file "GUI run features"), so the mode animation needs
  no feature in the model: the run spec (results_spec, Save run's model.json) holds it, the document and tree do not.
- Post-processing edits (title, description, derived values, analytic and derived curves: `document.results_inputs`)
  neither make results out of date nor drop cached estimates. Post-processing is evaluated from the document's current
  definitions, never the built spec's: derived values by `Session.derived_value` on the worker (shown only while
  `results._derived_definition` is unchanged), derived curves likewise (`analytic.derived_curve_plots`), references
  and analytic curves on the GUI thread over `ResultsPanel._curve_parameters` (the session's parameters: a swept
  model's Build is at the base value). A 'results' / 'all' edit redraws every result view.
- Result curve plots (all but the Field / DFT maps, tables and the far-field polar plot) are
  `plot_view.ResultPlotView`s made by `ResultsPanel._plot_view` (contract: the class docstring). Owners call
  `request_redraw(live=)` on every data change and never draw themselves (a hidden view redraws when shown). Series
  colour = position in the whole xy (not its unit group): overlays take the colour `DrawnPlot.drawn` hands them or
  one off `SERIES_COLORS`.
- `plot_controls.PlotPanel` ends every redraw (`apply(context)`); axis limits, tick steps and per-curve settings
  belong to that view / context and are not saved (a redraw keeps a mouse zoom). The `viz.curve_figure.PlotStyle` is
  ONE object per QSettings key shared by every panel (`_SharedPlotStyle`), so a style value must suit every plot:
  data-dependent settings go to the Axes / Curves tables. Export plot is WYSIWYG: `PlotPanel.export` hands what the
  screen drew (visible items, view range, shown axis texts, SI scales, the ticks `_ArticleAxis` drew last) to
  `save_curve_figure` (pt -> px through `_px_per_pt`), never the pyqtgraph exporters (cosmetic pens stay thin).
- Canvas display grid (`canvas._display_grid`): non-uniform cells (graded, native snapshots) keep `_NativeCells`, edges
  from `viz.common.edges_from_centers` (never midpoints of centres: wrong on graded cells) and the `containing_cells`
  rule of line_profile for image, detail, hover and drawn profiles.
- Examples menu (`gui/examples.py`, `MainWindow._populate_examples_menu`): menu path and title convention are written
  up in examples/README.md "Naming": change both together. The catalog reads raw JSON only, never load_model /
  validate (why: `ExampleCatalog.scan`). Menus are rebuilt on every aboutToShow and old submenus deleteLater'd
  (`QMenu.clear()` keeps addMenu's submenus).
- More windows = more processes, never more documents in one MainWindow: one window owns one document, one
  WorkerBridge and one worker session (`main_window.launch_window`; Duplicate: `window_copy_directory`). The GPU budget
  and the element-mesh and cuDSS ordering caches are per process, so two windows computing at once can run the later
  one out of memory. Windows share QSettings.
- PySide6 is imported before pyqtgraph (`gui/__init__`). `app.main` sets `imageAxisOrder='col-major'` (images are
  `[i, j]`) and the QSettings organisation / application names every bare `QSettings()` relies on.
- Views: defer rebuilds with `QTimer.singleShot(0)` and a pending flag; block an editor's signals before deleting it;
  mutate fns look nodes up by name; catch `helpers.USER_ERRORS` around user edits (`helpers.run_user_operation`; every
  panel's refused edit reaches `MainWindow._report_refused_edit`). File dialogs via `helpers.start_directory` /
  `remember_directory` / `ask_save_path`; Save run uses `helpers.ask_run_directory` (never a QFileDialog in save mode:
  it enters an existing folder typed as the name). `MainWindow._commit_pending_edit` clears focus before Build,
  Compute, saves and the discard prompt (an external rebuild drops uncommitted editor text). SettingsPanel rebuilds
  only on a node change, section 'all' or the cases of `_on_document_changed`, else refreshes in place: an edit outside
  'all' that changes a form's structure must select another node, and every editor a non-'all' edit can change needs a
  refresher.
- ModelDocument: never cache `spec` (replaced on undo / redo / set_spec and after a failed mutate) or change it outside
  `mutate` (from a document signal handler it raises: defer); never modify `spec_dict()`. A new cached estimate goes
  into `SpecEstimates` (shared with `SweepValueEstimates`) and must depend only on `results_inputs` (an edit leaving
  them unchanged carries the cache over); canvas and 3D preview likewise skip re-evaluating on 'model' / 'results'
  edits. `replaced` fires only on new / open, just before `changed('all')` (undo / redo emit only `changed('all')`):
  views drop run data there (canvas, ResultsPanel, mesh previews), and ResultsPanel advances the bridge generation.
- Node paths: the node constants, `parent_node`, `sibling_names`, `mesh_of_node` and `issue_node_path` live in
  gui/document (tree, forms and issue placement share them). Group nodes (`RESULT_GROUP_NODES`, ('derived_plot', plot),
  `COMPUTED_RESULTS_NODE` = the tree's 'Results', `MESHES_NODE`, `ESTIMATION_NODE`) are not spec items: no rename /
  delete / move (renaming a derived_plot sets the plot of its curves). Same-edit side effects of `add_physics`,
  `add_feature`, `duplicate_node`, `add_mesh`: their docstrings; the document's `notice` signal reports them.
- Add physics menu (`ModelTree._add_physics_menu`, rebuilt on every context menu): one submenu per
  `schema.PHYSICS_GROUPS` entry; misfits (`physics_type_misfit`, also a missing `solver_backend`) disabled under 'Not
  available in this model' with the reason as tooltip. Only the physics present restrict it: `add_physics` switches
  2D <-> 3D in the same edit when the picked space needs it.
- Memory budget: `solver_defaults.memory_check` is edited only by the mesh form's 'Override memory warning' tick box
  (`settings_panel._MEMORY_OVERRIDE`), shown on the mesh the budget issue is reported at (`document.memory_verdict()`)
  while over a budget or overridden. Built element counts (previews, built meshes, the meshes a failed build kept)
  reach the estimates through `document.set_element_counts` (mesh_preview); its `issues_invalidated` (also a > 5 %
  budget change) is what SettingsPanel refreshes on. Compute stays refused on an over-budget error until the override
  is ticked or the built counts fit.
- Meshes (S7.2; it must always be clear which mesh a physics uses): `MESHES_NODE` between Couplings and Solvers, one
  ('mesh', name) child per ModelSpec.meshes entry (rename / delete / move are 'all' edits; meshes[0] is the default, so
  `set_physics_mesh` stores '' for it and a move names the default mesh in the physics on it). Every view names and
  colours meshes through the mesh helpers of gui/helpers (`mesh_color`: by index in spec.meshes), never its own text.
  The mesh form addresses its MeshSpec by the form's node name inside every mutate (`_form_mesh`). Canvas
  `_active_mesh_name` = the shown field's mesh in the field view, else `document.mesh_of_node(current node)`, else the
  default; 'Show mesh' draws the document's current estimate (triangles: the built ones; after mesh refinement the
  last level's).
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
  without a field, its bar aligned to the view box by `_align_colorbar` margins on either view box's sigResized; fixed
  tick-axis widths (`_ArticleAxis` moves the left label beside the widest drawn tick label each paint); the title sets
  no minimum width; `_place_plot` puts the widget at the top of
  the layout-free `_PlotArea` at the last fit's aspect, each view side at least `_MIN_FRAME_FRACTION` of its largest).
  The aspect is never locked: equal x / y scales only where the fitted frame allows (a thin geometry gets unequal
  ones), a wheel over an axis zooms that axis alone (pyqtgraph AxisItem), so code needing a pixel size takes both
  `viewPixelSize()` components. Banner / mesh badge / legend live in the scroll box below: a
  new canvas text line goes there, never above or below the plot.
- Canvas figure style: `canvas.plot_panel` = a PlotPanel on the canvas PlotItem (own QSettings key
  'canvas_plot_style'; placed by ResultsPanel at the end of the Field tab); fixed colour limits override the field's /
  replay range in `_update_field_image` (lowest >= highest: ignored); 'Log colour' images hold `_color_values` = log10
  of values clipped at `_field_log_floor`, levels and colour-bar axis in log10; hover readout and data exports keep the
  values. `_redraw` and `_update_field_image` end with its `apply()`. Its Export plot is
  `scene_export`: visible ImageItems as drawn (RGBA), overlays in z order and the colour bar to
  `save_curve_figure(scene=)`; ROIs are left out. A new overlay item type needs a case in `plot_controls._scene_layers`,
  which reads `PlotItem.items` (pyqtgraph's `ViewBox.addedItems` holds only auto-range items: nothing added with
  ignoreBounds=True, i.e. no overlay). The colour bar's ticks and texts are the ones its axis drew last
  (`_keep_drawn_ticks` records them on pyqtgraph's own AxisItem), so matplotlib never picks its own format / offset
  text; a canvas text or tick format that must match the file goes through such drawn values, never a second formatter.
- Canvas scale bar (`PlotStyle.scale_bar`, default on; Axes tab group 'Scale bar', scene panels only): every canvas
  mode then has no axes (all four hidden in `_style_axes`; the status bar's cursor position stays), so nothing on
  the canvas may rely on an axis being painted (`drawn_ticks` of a hidden axis are stale). `plot_controls._ScaleBar`
  = plain Qt items under the view box (not in `PlotItem.items`: not a scene layer, no bounds, no mouse input), laid
  out by `viz.curve_figure.scale_bar_layout` on every range / size / style change; its length text needs
  `plot_panel.set_unit_length(model.scale)` (`_draw_view`). x and y pixel scales further apart than
  `_SCALE_TOLERANCE` (thin geometry, Stretch z) get a second, vertical bar; the export takes `_ScaleBar.drawn`. The
  bar sits in a bottom corner of the view box's `childrenBounds` cut to the view (the field / material image, not
  the margin a fit leaves around it), so an item meant as content must count for the bounds and an overlay must be
  added with ignoreBounds, as they already are for the auto range.
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
- Snapshot solutions (model sub file): a session snapshot whose summary `snapshot_solutions` entry lists the field's
  physics shows any field and any cut (`_snapshot_solution`; replay: every snapshot, `every_snapshot=True`); every
  session snapshot read goes through `_session_snapshot_field` (worker: `Session.snapshot_field`, else the nearest
  held cut), never `session.snapshots[i].fields` directly. Kept sweep values have none.
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
  errors, issues, estimates, banners, notes) is a `forms.hint_label` / `error_label` / `wrapped_label` or a QLabel
  wrapped in `forms.selectable` (it takes click focus, so a click on it commits a pending edit like any other click);
  form row labels and the status bar stay plain.
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
  keeps the shown cut until the fetched field arrives; outlines, hover and axis labels follow `FieldData.axes`. A
  field's z cut in the Top view sets `_EvaluatedModel.cut` = its z (`_spin_axis` 'z': the cut spin replaces the Layer
  combo) and shows the layer holding it; the edited `_layer` is kept, and leaving the field view re-evaluates.
- 3D results: 'Cut plane' axis + position (from `BuildSummary.x / y / z_centers`) -> `plane` of
  `session.display_field`; live frames are the mid-height z cut; a snapshot shows the stored cut nearest the chosen
  position (`_snapshot_field` matches by `FieldData.plane`; a kept solution: exactly that cut); a vertical axis
  without a summary `stored_cuts` entry or kept solution is refused and the combos set back (`_set_plane_choice`),
  never a fallback to the z cut (it would switch the canvas view and refit). `_replay_key` = (field key, `_snapshot_plane()`); while a snapshot is shown (`_snapshot_shown`) a
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
- Solve flow window (`solve_flow_view.py`, Solvers form button 'Show solve flow…' -> `show_solve_flow`: one
  non-modal `SolveFlowDialog` per document, a child of the first ancestor holding `bridge` (MainWindow), found again by
  `findChildren`): draws `model.solve_flow.solve_flow` of the document's cached estimates only (`used_mesh_estimates`,
  fitted `memory_estimate` + `plain_roles()`, `run_time_estimate()` = estimate_time per_run (counters kept: solver
  items go to their role box), `time_overheads()`, `time_estimate()` for the whole-study total): GUI thread, nothing
  run, no `bridge.call`. Measured values = `bridge.last_build_summary` (ledger gpu_breakdown / stage_seconds) and
  `last_study_summary` (time_check, run_memory, external_memory), passed only while `results_inputs` of
  `bridge.results_spec` equals the document's (the Results panel's out-of-date rule), else a note says why; boxes then
  grow a green measured line. Redraw debounced 400 ms after `changed` / `issues_invalidated` / every bridge build or
  run end, only while visible (showEvent redraws a dirty chart), keeping the selected step by key; the first draw fits
  the whole chart, the mouse wheel zooms (no scrolling), and showEvent sets the splitter share once (the text browser's
  size hint would take most of the width). Boxes paint their own texts (kind tag, title, subtitle, estimate and
  measured lines: one elided line each, full texts in the tooltip) and bars in `_StepBox.paint`; the row kind is the
  box tag (no label column). Box fills are light with dark text in every theme; mesh / materials borders take
  `helpers.mesh_color`. A new step kind needs `_KIND_COLORS` and `_KIND_TAGS` entries.
