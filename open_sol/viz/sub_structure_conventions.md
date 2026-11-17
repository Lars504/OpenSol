# open_sol/viz — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Visualizer and result_plots gotchas.

- Visualizer holds its Simulation by `weakref` (`sim` raises once collected): never give it a strong back-reference
  (the cycle kept dropped script Simulations' GPU arrays until a full gc). Every figure comes from `_new_figure`
  (`matplotlib.figure.Figure`, never pyplot: no backend switch, nothing in pyplot's registry, nothing to close).
- Drawing positions: flux and mode monitors on the hi face of their line cell, line and mode sources through the
  centre (H_z sheet currents on the hi face), NF2FF boxes through the sampled centres (`center_box`), TFSF / DFT boxes
  on cell edges; each FDFD class like its FDTD counterpart; an unknown source / monitor class raises TypeError (em_modes
  table / dispersion monitors are skipped).
- Several physics: sources, monitors, probes and the dashboard cover every `_em_solvers()` entry (EMPhysics,
  FDFDPhysics, ModePhysics); the material map is eps_r with any physics whose `permittivity_arrays` hold it, eps_xx with
  only H_z, else heat k_xx, else sigma_dc_xx / eps_dc_xx of the first potential physics; boundary bars likewise (first
  EM, else heat, else potential). Geometry panels overlay the union of every EM `pec_cells` (grey) and a dashed sigma_e >
  0 outline (`_draw_conductors`: PEC never touches eps). Component None = the primary field of the given, else the
  first physics.
- FDFD probes only from `_harmonic_probes()` (plot_probe refuses a HarmonicProbe); `_flux_spectrum` /
  `_far_field_pattern` dispatch FDTD / FDFD for plot_flux, plot_transmission and plot_radiation_pattern (only
  `freq_idx`, range-checked: FDTD monitor order, FDFD ascending wavelength); the dashboard flux panel stays FDTD only.
- `common.py` (numpy only, no matplotlib / Qt) holds the helpers shared with the GUI: add a shared helper there, never
  a GUI copy. `safe_ratio` (1e-30 relative threshold); `probe_spectrum` is the one probe FFT (resampled onto the median
  step: flush adds off-stride samples; >= 4 samples, DC bin dropped); `line_profile` uses the `core.domain.containing_cell`
  rule (an edge sample reads the upper cell), so it agrees with probes on the same point. `normalised_db` and the probe
  spectra normalise by the largest finite value (one NaN sample keeps the scale).
- result_plots draws analytic curves only as given (`save_result_figure(curves=)`: abscissa in the result's units, far
  field normalised, timeseries vs the 1-based iteration; a 'table' draws none: api plots `analytic.table_xy` of it as an
  xy figure); evaluating them is api's job (viz never imports model). `_draw_xy` honours an xy `series_x` {series: own
  abscissa} (sweep overlays, drawn by api `Results.sweep_figure`: no separate sweep-figure function).
- result_plots reuses visualization's private helpers (`_STYLE`, `_draw_cells`, ...), ASCII labels (micro -> 'u'),
  never prints. Panels drawn by both live once in visualization (`_draw_map`, `_draw_timeseries`, `_draw_flux_panel`,
  `_draw_far_field_panel`; `ascii_text=True` from result_plots): the Visualizer builds the result dict and draws the
  same panel, never a copy.
- Legends list (line, label) pairs explicitly (`_legend`, curve_figure): matplotlib's automatic legend drops labels
  starting with '_' (a valid curve name).
- curve_figure (no Qt; the GUI's 'Export plot…'): `PlotStyle` sizes in pt, figure size (inches) from the caller, exact
  through the constrained layout (never bbox_inches='tight'), fonts kept as text (pdf / ps fonttype 42, svg fonttype
  none); `AxesData.x_ticks` / `y_ticks` (`AxisTicks`, data units) fix ticks and labels, else automatic ticks
  (style.minor_ticks then does not apply; tick steps are per-plot GUI controls, never PlotStyle fields);
  `PlotStyle.from_dict` drops wrong-typed / unknown stored values; curves come coloured with their own line width /
  marker size (`CurveData`; PlotStyle has no curve fields).
- Units as in the GUI: every wavelength axis / label in nm (`visualization.WAVELENGTH_UNIT`;
  `core.wavelengths.wavelength_label`, '.6g'); result_plots xy axes (x and each y panel) get an SI prefix
  (`_axis_scale`: m / s by length_unit / time_unit, the units of `_PREFIXABLE_UNITS` 10^(3k); any other unit (kg, rad,
  ppm, K, dB, deg, compound) keeps scale 1), analytic curves the same scales. Line profiles: the field label plus 'name
  (analytic)'; dashboard probe spectra are labelled like the probe signals.
- scene_figure (view_3d C): draws a `model.scene_3d.Scene3D` duck-typed (no model import) on an Axes3D from
  `new_scene_axes`; every solid face goes into ONE Poly3DCollection so mplot3d depth-sorts across objects (painter's
  algorithm: interpenetrating faces can still sort wrongly; ~1e4 faces is the practical ceiling, build_scene's automatic
  n_segments aims below it); `z_stretch` changes only the box aspect (the z label says so). `field_planes` (FieldData
  duck-typed) join that same collection, one quad per cell (above `PLANE_CELLS` every k-th cell), opaque, one
  `color_limits` scale over all planes plus a colour bar; the objects are then edges only and the material legend is
  dropped.
- Every cell map goes through `_draw_cells` = `pcolorfast` over 1D cell edges (cost follows output pixels; never
  pcolormesh); Visualizer panels pass `frame.window_edges()` (DFT boxes their slice), None edges = uniform cells over the
  extent. FieldData plots (`save_field_figure`, `plot_field_array(field_data)`) and saved dft_fields take edges from the
  centres (`field_data_edges` / `common.edges_from_centers`, outer edges pinned to the extent): never uniform cells for
  native graded cells. Map axes are labelled from `FieldData.axes` (('x', 'z') of a 3D y cut); native polar / spherical
  FieldData is refused (ValueError: plot the drawing-plane image). `save_field_figure(show_mesh=True)` adds 'mesh
  <name>' to the default title.
- A FieldData with `triangulation` (triangle meshes) draws by `_draw_triangles`: Gouraud tripcolor over the display
  triangulation (P2: 1 -> 4 split), triangles with a non-finite node value masked, clipped to the field extent (the
  absorbing ring lies outside), colour limits from the raster values, optional `contours`; P0 fields have no
  triangulation and draw their raster. `common.triangle_mesh_lines` gives the GUI's host line sets (outlines only above
  MESH_LINES_MAX_EDGES); `common.tetrahedron_mesh_data` / `tetrahedron_mesh_cut` give a tetrahedron mesh's host nodes /
  elements and the outlines of axis-normal cross-sections (nodes on the plane nudged above it, shared edges once, PML
  shell segments apart).
- mode_player (HTML mode animation; `Results.export_html` of a mode_shapes result, CLI `show <run dir> --html`, GUI
  Results dock Export > Export mode animation): `mode_player.py` packs numpy data into `mode_player.html` (package
  data) at `DATA_MARKER`; the page is self-contained on purpose (hand-written WebGL2, no CDN, works offline in a talk): keep it free of external scripts
  and fonts. Data: positions float32 / triangles uint32 / per mode re + im int16 (base64), each mode phase-aligned (largest
  component real) and scaled to max |u| = 1 (`_aligned`), lengths in the unit `length_scale` picks; '</' escaped in
  the JSON. Animation u = Re[U exp(j (cell_phase n + theta))], theta = omega t, tile n along the lattice axis: open_sol
  Bloch modes satisfy u(x + L) = u(x) exp(-j k L) with time exp(+j omega t), so the adapter sets cell_phase = -k L.
  `mode_player_from_results` duck-types api.Results (result_names / result / sweep_result / summary / spec; no api or
  model import) and reads only 'mode_shapes' results; sweep values on another surface mesh than the last are dropped;
  band = rank of the frequency among the computed resonances at that k (a search window, not an absolute band index),
  the page labels it 'mode j of n'. Colour maps come in two kinds: sequential for the magnitudes, diverging (middle stop =
  0) for the signed u_x / u_y / u_z; the menu lists the shown quantity's kind and remembers a choice per kind. Band gaps on the page: intervals no band segment crosses (bands continuous between
  samples, matched across k with an index shift for a moving search window) and inside the window at every k.
- field_viewer (HTML field page; `Results.export_html` of a field or dft_fields result, CLI `show <run dir> --html
  --what NAME`, GUI Results dock Export > Export field / DFT web page): `field_viewer.py` packs `ViewerFrame`s of
  `ViewerLayer`s (grid of cell centres, or a FieldData display triangulation; plane = a 3D cut placed in the scene,
  None = a 2D map at z = 0) into `field_viewer.html` at its `DATA_MARKER`, self-contained like mode_player (its
  camera, colour maps, theme, PNG and recording code are copies of mode_player's: change both). Every frame holds the
  same surfaces (`_check_layers` raises otherwise); values are int16 per frame over that frame's [lo, hi] (-32768 = no
  value: the page discards triangles and edges touching one). `plan_frames` fits MAX_BYTES / MAX_FRAMES: the smallest
  grid stride (never below MIN_LAYER_CELLS on the largest layer), then evenly thinned frames, both named in the
  subtitle; a caller loading frames lazily (api's DFT path, one frequency at a time) plans first and passes `plan=`.
  frame_kind 'time' plays (Play, Speed, Record = one pass), 'labelled' (wavelengths, modes, DFT frequencies) only steps
  (slider, arrows; Record = one turn of the view); maps are drawn unlit so colours match the legend, lit only with
  relief. Signed fields: diverging maps, symmetric limits; log scale only for non-negative fields.
