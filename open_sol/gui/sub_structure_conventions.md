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
- Meshes (it must always be clear which mesh a physics uses): `MESHES_NODE` between Couplings and Solvers, one
  ('mesh', name) child per ModelSpec.meshes entry (rename / delete / move are 'all' edits; meshes[0] is the default, so
  `set_physics_mesh` stores '' for it and a move names the default mesh in the physics on it). Every view names and
  colours meshes through the mesh helpers of gui/helpers (`mesh_color`: by index in spec.meshes), never its own text.
  The mesh form addresses its MeshSpec by the form's node name inside every mutate (`_form_mesh`). Canvas
  `_active_mesh_name` = the shown field's mesh in the field view, else `document.mesh_of_node(current node)`, else the
  default; 'Show mesh' draws the document's current estimate (triangles: the built ones; after mesh refinement the
  last level's).
- Mesh refinement (StudySpec.adaptation): the results_spec stays the submitted spec (level 0 meshes); the last level's
  meshes exist only as summary 'convergence' adapted_meshes, so code showing the results' meshes takes them from there
  (`ResultsPanel._adapted_mesh_estimates` -> `canvas.set_results_meshes`). A run emits Built per level
  (`_refinement_level` counts them).
- Switch-on blockers (`settings_panel._adaptation_blocker`, `estimation_form.estimation_blocker`) repeat validation's
  rules (`validate._check_adaptation` / `_check_estimation`): a new refusal there goes into the blocker too. The
  off-state groups re-check them in a refresher (a sweep or parity-sector edit only refreshes).
- Parameter estimation: `ESTIMATION_NODE` exists only while the study has one (no rename / move; Delete switches it
  off, a 'study' edit). Validation reports its issues on ('study',) with messages starting 'estimation': match issues
  to nodes only through `document.issue_node_path`, which moves them to the node.
- Element meshes (`mesh_preview.ElementMeshStore`, triangles and tetrahedra): views draw `store.drawing(mesh)` (None
  once the document's `element_mesh_key` no longer matches: never stale elements), 'Show mesh' `store.built_drawing`
  (the mesh the fields live on). The worker keeps the last element mesh per name (`meshing.remember_element_meshes`
  in `BackendWorker._serve`), so a Build after a preview of the same key reuses it. The canvas field view stays a
  raster on every mesh kind (tripcolor only in exported figures).
- Solver fits: a solver type over the whole model goes through `document.solver_fits` (Add solver menu, `add_solver`'s
  automatic assignment, `_line_block_fits`); the physics form's Solver combo judges its one (physics, role) by
  `schema.solver_misfit` on the node values resolved for that role (`with_node_precision`). No third local rule.
  Machine availability (a package not installed, GPU-only on the cpu) is the SolverType's `unavailable_choices`: the
  node form greys those choices, the physics' Solver combo greys nodes set to one (`schema.unavailable_reason`), and
  Automatic labels name what 'auto' runs as here (`schema.expected_auto_choice`), never a local find_spec.
- New node kind: an entry in every node table at the top of gui/document.py (SECTIONS, _ROOT_NODES, _NAMED_NODES,
  _TREE_NODE_KINDS, _PARENT_NODES, _SECTION_OF_KIND, _REFERENCED_KINDS) and in `spec_renames` / `delete_node`.
- Canvas views: selecting Meshes / a mesh / Results switches to the mesh / field view (`_NODE_VIEWS`); when the view
  refits: `_reset_view_if_needed` (a mode switch, a new field or another sweep value never refits).
- Canvas sweep sliders: the canvas shows the model at `_shown_sweep_value()`; geometry, material preview, mesh
  estimate, element meshes (store calls take `estimates=`) and the status line then come from
  `document.at_sweep_values(pairs)` (`SweepValueEstimates`). View only: validation, Compute, forms and the 3D preview
  stay the document's.
- Canvas layout: the plot never changes size with the mode or its texts: a new canvas text line goes into the scroll
  box below the plot (banner, mesh badge, legend), never above or below it. The aspect is never locked (fits allow
  unequal scales, a wheel over an axis zooms that axis alone): code needing a pixel size takes both `viewPixelSize()`
  components.
- Canvas Export plot (`canvas.plot_panel`, `scene_export`) writes what was drawn: a new overlay item type needs a case
  in `plot_controls._scene_layers`, and a tick text / format that must match the file comes from the drawn ticks
  (`drawn_ticks`), never a second formatter. With the scale bar (default on) all axes are hidden, so nothing may rely
  on an axis being painted (its drawn_ticks are stale). The bar sits on the view box's `childrenBounds`: content items
  count for the bounds, overlays are added with ignoreBounds (as for the auto range).
- Canvas interaction: every new ScatterPlotItem overlay is a `_PassiveScatter` (clicks pass through). An ROI commit
  keeps every param within half a pixel of its stored value (a drag never re-rounds what it did not move); new
  coordinates round via `_to_si`.
- Object multi-selection: `ModelDocument.geometry_group` (`toggle_geometry_group` / `select_geometry`, cleared by every
  `select`) is the one record; form and ROI stay the current node's, the tree stays SingleSelection
  (`_sync_current_item` marks the group). Hover highlight flows one way, panels -> canvas (`hover_entities`,
  `set_hover_node`), never from canvas hover.
- Canvas fields are display grids: `Session.display_field` resamples (nearest cell, `FieldData.native_shape` set;
  polar: containing-cell images over the sector's box, NaN outside), so 'Export field data' refetches
  `Session.field(native=True)` while it still can (`_session_state_field`, worker idle); snapshots and live frames
  export resampled.
- Snapshots: every session snapshot read goes through `results._session_snapshot_field` (worker; kept solutions: model
  sub file "Snapshot solutions"), never `session.snapshots[i].fields`. Kept sweep values (SweepSpec.keep_fields: the
  Field tab's 'Sweep value' combo, `_kept_snapshots`) are host Snapshots read on the GUI thread, never through
  `bridge.call`, and keep no solutions. With SweepSpec.reuse_build a sweep emits one Built only: views must not
  assume one per value.
- GUI texts: shared wordings and formatters live in gui/forms (labels, `display_label`, `format_quantity`),
  `helpers.format_duration` and `model_tree.RESULT_SETTINGS_TIPS`: grep there before writing a local one.
  `display_label` turns ASCII 'lambda = <nm> nm' into 'λ = <nm> nm' for display only: match and write raw labels.
  Tooltips show `Param.tip` (else help) / `summary` (else description) through `forms.rich_tooltip` (Qt wraps only
  rich-text tooltips). Text a user may want to copy (hints, errors, issues, estimates, banners, notes) is a
  `forms.hint_label` / `error_label` / `wrapped_label` or wrapped in `forms.selectable` (it takes click focus, so a
  click on it commits a pending edit); form row labels and the status bar stay plain.
- Red marks go through `forms.mark_invalid`; combos are marked by palette, never a style sheet colour (Qt applies a
  style sheet `color` to the popup list in every colour group, so greyed-out entries turned red too).
- Colours, colour maps and drawing helpers are imported where they live (canvas `MATERIAL_COLORS` /
  `CATEGORY_COLORS` / `HIGHLIGHT_COLOR`, viz `DARK_CMAPS`, `viz.common`), never copied into another GUI module.
- Polar / spherical domains: the canvas keeps the drawing plane; everything mapping it to cells goes through
  `_EvaluatedModel.domain_map` (None on Cartesian): plane extent, faces (rays joined by `angle_wrap` are no faces: not
  drawn, labelled, picked or highlighted), inside test, outline, mesh view. `model.cells` is None there (no cell
  snapping or mesh readout).
- 3D models: z params show only in 3D or while set (so they can be cleared); a prism along x / y relabels params by
  model axis (`settings_panel._axis_params`), so an axis edit rebuilds the form. Front / Side views use scene
  coordinates (u, v) = (x, z) / (y, z) (`_EvaluatedModel.scene_axes`); outlines, hits and preview belong to one view
  (re-evaluated on a view or cut change); side views select and highlight only (no draw tools or ROIs). In the field
  view the shown field's plane rules (`canvas._field_view`): its View combo / cut spin ask ResultsPanel
  (`field_cut_requested`) and keep the shown cut until the fetched field arrives.
- 3D preview (`preview_3d`): field cut planes come only through `ResultsPanel.fetch_cut_planes` from its own redraw,
  refetched on `ResultsPanel.field_view_changed` (every way the shown session field changes emits it).
- Mesh form: its rows depend on kind, mesher, sweep_axis (`_tensor_graded`) and mode (`_set_mesh_mode` rewrites the
  sizes), so those edits rebuild; axis options are addressed by (index, axis), bands / segments by index
  (`_mesh_axis_at` / `_mesh_axis_item`: KeyError -> rebuild).
- Solve flow window (`solve_flow_view.py`): GUI thread, nothing run: it draws `model.solve_flow` from the document's
  cached estimates; measured values (`bridge.last_build_summary` / `last_study_summary`) only while
  `document.results_current(bridge.results_spec)` (the Results panel's out-of-date rule). A box shows
  `FlowStep.device_peak_bytes` (GPU) and `host_bytes` (CPU) apart (`_held_text`), which add up to the totals and to
  the mesh node's memory issue; the item table marks device items outside the peak phase.
