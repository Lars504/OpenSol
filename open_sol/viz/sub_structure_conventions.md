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
  marker size (`CurveData`; PlotStyle has no curve fields). `SceneData.scale_bar` (`ScaleBarData`, lengths in data
  units; the `PlotStyle.scale_bar*` fields mean something only for scene plots) replaces ticks, labels, frame and
  grid: `scale_bar_layout` / `scale_bar_origin` (pt; the GUI preview calls the same two) placed in axes fractions
  after a `draw_without_rendering` pass, its artists out of the layout; `ScaleBarData.content` (data units) is the
  region whose corner the bar sits in. A single bar (y_length 0) promises equal x /
  y scales, so the export sets that aspect with adjustable 'datalim' (a limit widens by the few percent the two
  layouts differ; the axes box and so the colour bar height stay); two bars are each sized from their own axis.
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
  View in browser on a canvas field of that physics): `mode_player.py` packs numpy data into `mode_player.html` (package
  data) at `DATA_MARKER`; the page is self-contained on purpose (hand-written WebGL2, no CDN, works offline in a talk): keep it free of external scripts
  and fonts. Data: positions float32 / triangles uint32 / per mode re + im int16 (base64), each mode phase-aligned (largest
  component real) and scaled to max |u| = 1 (`_aligned`), lengths in the unit `length_scale` picks; '</' escaped in
  the JSON. Animation u = Re[U exp(j (cell_phase n + theta))], theta = omega t, tile n along the lattice axis: open_sol
  Bloch modes satisfy u(x + L) = u(x) exp(-j k L) with time exp(+j omega t), so the adapter sets cell_phase = -k L.
  `mode_player_from_results` duck-types api.Results (result_names / result / sweep_result / summary / spec; no api or
  model import) and reads only 'mode_shapes' results; sweep values on another surface mesh than the last are dropped;
  band = rank of the frequency among the computed resonances at that k (a search window, not an absolute band index),
  the page labels it 'mode j of n'. Colour by (`QUANTITIES` on the page, code = the shaders' uColor): the
  displacement always; with the mode_shapes result's kept groups (`DATA.fields`: stress (+ potential), strain;
  `fields=` / `max_bytes` choose them, `ModePageTooLarge` carries the sizes for the GUI's dialog) the page derives the
  rest itself: tensor components, von Mises, principal / maximum shear stress, pressure, volumetric strain, the time
  averaged strain (1/4 Re sigma : eps*, needs both groups) and kinetic (rho |U|^2, `DATA.density`) energy densities.
  Every group is complex, turned by the displacement's alignment rotation (`_aligned` returns it: a group without
  that turn would animate out of phase) and scaled to its own max |component| = 1 (no scales on the page: each
  quantity is drawn normalised by its largest magnitude over nodes and cycle, `quantityLimit`, so constant factors
  cancel). Linear quantities animate as Re[X exp(j psi)] in the vertex shader (`TENSOR_GLSL`); `quantityLimit`
  mirrors those formulas in JS (nonlinear ones sampled over LIMIT_PHASES): change both. Node data are decoded per mode
  on first use (a few modes cached) and only the groups of the shown quantity are expanded to the unindexed vertex
  buffers (`uploadMode`, also on a quantity change). Rest shape (G) is a hidden-line drawing, never the mesh: page-side candidate edges
  (creases above FEATURE_COS, open / non-manifold edges, plus every non-flat edge that the line shader keeps only
  while it is a silhouette, so curved bodies get an outline), screen-space quads drawn after the body in two passes
  (depth body + rest surface: full strength; depth rest surface alone: faint where the body covers it), one fragment
  per pixel through the stencil (cores before fringes; the context asks for `stencil: true`), surfaces under a polygon
  offset so lines on them pass. Edges in the lattice end planes show only at the ends of the tiled row (seams between
  cells otherwise), and only when the body spans the period. Colour maps come in two kinds: sequential for the magnitudes, diverging (middle stop =
  0) for the signed u_x / u_y / u_z; the menu lists the shown quantity's kind and remembers a choice per kind. Band gaps on the page: intervals no band segment crosses (bands continuous between
  samples, matched across k with an index shift for a moving search window) and inside the window at every k.
- field_viewer (HTML field page; `Results.export_html` of a field or dft_fields result, CLI `show <run dir> --html
  --what NAME`, GUI View in browser of the canvas field or DFT map): `field_viewer.py` packs `ViewerFrame`s of
  `ViewerLayer`s (grid of cell centres, or a FieldData display triangulation; plane = a 3D cut placed in the scene,
  None = a 2D map at z = 0) into `field_viewer.html` at its `DATA_MARKER`, self-contained like mode_player (its
  camera, colour maps and recording code are copies of mode_player's: change both; theme and PNG differ by the Plot
  settings background / PNG size). Every frame holds the
  same surfaces (`_check_layers` raises otherwise); values are int16 per frame over that frame's [lo, hi], linear, or
  log10-spaced over LOG_DECADES below hi when the frame is non-negative and reaches ~0 (`_log_floor`; frame `log_lo`
  set; linear int16 would leave the log colour scale a handful of codes per decade); -32768 = no value (the page
  discards triangles and edges touching one). A display triangulation is clipped to the field extent (the solver's
  absorbing ring is not shown or counted in the range). `plan_frames` fits MAX_BYTES / MAX_FRAMES: the smallest grid
  stride (never below MIN_LAYER_CELLS on the largest layer), applied as stride x stride block means (decimation
  aliases oscillating fields), then evenly thinned frames, both named in the subtitle; a caller loading frames lazily
  (api's DFT path, one frequency at a time) plans first and passes `plan=`. Maps need >= 2 x 2 samples (a 1D field
  raises). The hidden `.empty` overlay needs `.empty[hidden] { display: none }` in both pages: its author display
  would otherwise beat [hidden] and the transparent overlay would take the canvas's pointer input.
  frame_kind 'time' plays (Play, Speed, Record = one pass), 'labelled' (wavelengths, modes, DFT frequencies) only steps
  (slider, arrows; Record = one turn of the view); maps are drawn unlit so colours match the legend, lit only with
  relief. Colour limits default to all frames for 'time', to the shown frame for 'labelled' (independent states).
  Signed fields: diverging maps, symmetric limits; log scale only for non-negative fields.
  Quasi-2D (page-side, no Python input): a 2D map or a set of 3D layers in one plane (a single cut) is 'planar': it
  opens face-on as the canvas shows it ((u, v) of `FieldData.axes` right / up), tightly fitted inside the readout /
  legend / axis-label insets, with relief. An axis spanning below 1 / MAX_ASPECT (4) of the largest is stretched by a
  1-2-5 factor (Stretch button, on by default; the page's counterpart of the canvas's unequal scales and Stretch z),
  applied as `uStretch` in the vertex shader: the box, relief height, fit and axis projection use the stretched
  spans, while the tick labels stay true coordinates and the axis title names the factor. Axes are a 2D overlay canvas
  (`#axes`); PNG and recordings take the composite of both canvases, unlike mode_player's GL canvas only. Everything
  a picture must hold is drawn on that overlay (`drawOverlay`: heading, colour bar, material legend), never as DOM
  over the view: DOM parts (hint, REC, the panel) are left out of PNG and recordings on purpose. The overlay takes
  no pointer input, so a part moved by hand is hit-tested in the GL canvas' pointer handlers against the box it was
  last drawn in: `MOVABLE` (the colour bar (home = its Plot settings place; a new place homes it), the material legend
  and the scale bar; `placePart` = home place plus the part's offset in CSS px, kept inside the canvas, remembered per
  file; drag = move, double-click on it = home instead of the view reset). A new movable part is an entry there and a
  `placePart` call in its draw function.
  Plot frame (field_viewer only; `mapNavigable`: planar, relief 0, face-on): wheel / pinch zoom and right- or
  Shift-drag pan change the in-plane Plot settings axis limits (`zoomMap` / `panMap` -> `setPlaneWindow` ->
  `applySettings(true)`), never the camera, so the frame (`mapFrame`: the limits' rectangle on screen) keeps its size
  and place while the shaders clip the map to it; zoom and pan are therefore saved, shown in the panel and exported
  like typed limits, and double-click on the map clears them (automatic limits, full size). Its size is `plotScale`
  (fraction of the free area inside `viewInsets`, frame centred there, remembered per file, Reset = 1), set by the
  DOM corner grip (`placeGrip` from draw, outside PNG / recordings). Other views keep the camera zoom / pan.
  Scale bar (Plot settings `scale_bar`, default 'on'; on = the axes start off on a planar view, the Axes button brings
  them back; a non-planar 3D view ignores it and keeps its axes: `scaleBarOffered`):
  drawn only on a planar view seen face-on (one scale on screen); its px per length come from projecting a step along
  the camera's right axis divided by the stretch, so it is right on a stretched axis; length a 1-2-5 value near a
  fifth of the view (follows the zoom) or `scale_bar_length` (m, remembered per file with the limits); home: the lower
  right corner inside the plot frame.
  Plot settings (field_viewer only; mode_player has none): one object `fig` on the page = the keys of
  `field_viewer.figure_settings` (`FIGURE_CHOICES` / `FIGURE_RANGES` / `FIGURE_LIMITS` / `FIGURE_LENGTHS`, repeated in the page: change
  both, plus the `fs_<key>` input of the panel and describe's export_html entry). Lengths are stored in m and colour
  limits in the field unit (the panel shows axis limits in the page's length unit, `DATA.length_scale`); null =
  automatic. Colour bar ticks are round values only (`barTicks`: 1-2-5 steps, decades on a log scale; a limit is
  labelled only when it is round), its name right after the widest label. Start values: defaults, then localStorage
  (style under 'style' for every page, only the values that differ from the defaults so a changed default reaches
  every browser; limits under
  'limits:<file path>' since the GUI rewrites one file per model and field, colour limits only for the same
  quantity; remembered axis limits that miss the data are dropped), then `DATA.figure` (api `export_html(figure=)`,
  checked in Python, raises). Axis limits are plot limits: `setBounds` puts them in bmin / bmax (box, ticks, fit,
  Stretch follow; they may reach past the data) and the three programs discard on `vPlace` (mirrored true
  coordinates) outside `uClipLo` / `uClipHi`; the box is drawn uncut; none along a planar view's normal. Colour
  limits go through `limits()` only (`autoLimits()` = data and Range slider). Overlay sizes derive from
  `fig.font_size` / `title_size` x `pixelRatio` (`barMetrics`, `headingHeight`), and `viewInsets` reserves exactly
  those, so a new overlay part or place updates both. PNG size: `renderScale` multiplies `pixelRatio` for one draw
  (canvas pixels, texts and screen-space lines scale together; the fit is scale-invariant). A settings change refits
  the view only when it was not panned / zoomed by hand (`cam.moved`) or the bounds changed. Panel inputs: ranged
  numbers (font sizes, bar length) apply while typed, limits on Enter / leaving the field (a half-typed number would
  crop and refit); panel headings name units, so no CSS uppercase there.
  Model objects (`viewer_data(geometry=ViewerGeometry)`, api `export_html(geometry=True)` from
  `model.scene_3d.plane_outlines` per layer plane + `build_scene` in 3D; duck-typed items): outlines are one list per
  layer of frame 0 (a layer count mismatch raises), packed as float32 segment rows (start, end, arc length at the
  start, unit direction into the object: 10 floats, only the first 7 in length units) per (layer, material, solid)
  group and clipped to the layer's `extent` (FieldData extent, so objects reaching
  into the absorbing ring stop at the window); the inward direction (`_ring_sides`: closed rings of a solid object by
  their turning sense, a ring inside another ring of the same object is a hole; zeros for open polylines, dashed
  inputs, edges and glyphs) lets the page draw material-coloured outlines wholly on their own object's side
  (`uInset`: the screen side is found per segment from a point inside, so mirror images and Stretch need no
  case), so two objects meeting on a line show both colours; one-colour outlines stay centred on the boundary; 3D solids add rim edges and their faces fan / ear-clip triangulated per
  material; point objects are crosses, lines segments. Colours: tab10 by material index (scene_figure's default).
  The page draws them in its own programs after the layers (lines = instanced screen-space quads, 1 px GL lines being
  too thin; dashes for not-solid composite inputs; glass = blended faces without depth writes), through the same
  centre and `uStretch` as the field; planar views draw lines without depth test (on top of relief). Objects select
  (G): outlines / edges / glass / off, default edges for non-planar 3D; outline colour Contrast (dark over a light
  colour-map middle) / white / black / material, remembered; the material legend shows whenever material colours do.
  Mirror planes (`viewer_data(mirrors=[ViewerMirror])`; api `_viewer_mirrors`: `core.mirror.mirror_faces` of the
  state spec with the field's `field_plane_signs`, value sign `image_sign(field_parity)`, plane positions by
  `mirror_plane` from the layers' extents (the window edges; a DFT map from a field of its physics, its box may stop
  short); none for unfolded fields (parity-sector snapshots) and polar / spherical domains, a RuntimeWarning for a
  field without parity or a face no layer spans): the page draws the images itself (no data doubling) as reflected
  copies of every layer, mesh line and object (uniforms `uMirA` / `uMirB`: a x + b per axis, `uValSign`; a phase
  gains pi), one image each side per axis like `unfold_field`, products across axes; a planar view drops faces
  along its normal. Bounds, centre, box, axes and the Stretch factors follow the shown images (`setBounds`); Mirror
  button (M), on by default. Grid layers ending on a plane get a sample row on it (`_plane_pads`: nearest row even,
  0 odd), else the map and its image would leave a one-cell gap at the plane. A layer normal to a mirror axis within
  the nearest grid sample row of that plane (api's default live cuts are those cells) is moved onto it by
  `snap_to_mirrors` (api calls it before computing the outlines, so they are the plane's sections); the page then
  skips the image that maps a layer onto itself (`selfImage`: field, mesh lines, outlines), else the coincident,
  for odd fields sign-flipped, copy z-fights.
  Flow particles (field_viewer only; 2D maps): a layer's `ViewerVectors` (its in-plane vector on its own raster, a
  triangle layer passes its FieldData raster) with a page `ViewerFlow` label; `viewer_data` resamples them onto a
  uniform grid over the layer extent (`_uniform_flow`: block means of the finite samples, NaN for cells holding only
  NaN, nearest source sample where the grid is finer; size by `_flow_shape` within FLOW_MAX_BYTES), int16 per frame
  over +-its largest magnitude, rows along v (texture rows). `_strided` / `_plane_padded` rebuild layers: they carry
  `vectors` (a new ViewerLayer construction must too). api `_viewer_flow` picks the vector (`_FLOW_TYPES`: steady
  flows only, never waves (em, acoustics) or displacements (mechanics); `_FLOW_BASES` /
  `<base>_abs|_x|_y`, components `<base>_x` or `<base>x`; heat: -grad T scaled to heat_flux_abs when stored) from the
  same snapshot / state as each frame, else no flow (no error: optional). Mirror images: `ViewerMirror.flow_sign` =
  image sign of the component along the plane (api from the components' parities): -1 runs that image's particles
  backwards. Page (`// ---- vectors ----`): stateless pathlines, each particle's latest FLOW_TAIL + 1 path points
  recomputed from its hashed seed every draw by a transform-feedback pass (so PNG, recordings, frame changes, zoom
  and mirrors keep no particle state), drawn as one-pass quads (core + halo in the fragment shader: two passes
  brighten every joint). Speeds normalise by `speed_reference` (a percentile, not the max: a 1/r peak would freeze
  the rest), clamped to 1; below STILL a particle dies (holes, solids and insulators are zero vectors, not NaN: they
  would collect crawling dots). Seeds fill the shown window per image (count by area, so zoom and mirrors keep the
  screen density). Hidden on a raised map (relief); `uploadFrame` resets `flowState.frame` so the textures follow.
