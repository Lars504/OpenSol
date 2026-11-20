# open_sol/viz — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). What each module, class and page block does is in
its docstring / block comment; this file keeps the rules that span files. GUI-side rules (PlotStyle sharing, Export
plot, imports of `viz.common` / `DARK_CMAPS`, canvas cell edges) live in the gui sub file.

## Matplotlib figures (visualization, result_plots, curve_figure, scene_figure)

- Every figure is a pyplot-free `matplotlib.figure.Figure` (`_new_figure` in visualization / result_plots): no backend
  switch, nothing in pyplot's registry, nothing to close. The Visualizer holds its Simulation weakly: never give it a
  strong back-reference (the cycle kept dropped script Simulations' GPU arrays until a full gc).
- A new EM physics class joins `Visualizer._em_solvers` (annotations, material map, boundary bars and the dashboard
  read it); a new EM source / monitor class needs its branch in `_annotate_sources` / `_annotate_monitors`, which
  raise TypeError on an unknown class.
- A panel both modules draw lives once in visualization (`_draw_map`, `_draw_timeseries`, `_draw_flux_panel`,
  `_draw_far_field_panel`; result_plots passes `ascii_text=True`) and result_plots reuses its private helpers; never a
  copy. result_plots takes analytic curves already evaluated (api's job: viz never imports model) and draws them over
  the panel's own abscissa scale (`_draw_timeseries` returns it), never with a second iteration / time test.
- Legends of user-named lines list (line, label) pairs explicitly (result_plots `_legend`, curve_figure): matplotlib's
  automatic legend drops labels starting with '_' (a valid curve / series name).
- Every cell map goes through `_draw_cells` (pcolorfast: cost follows output pixels; never pcolormesh); native graded
  cells pass their edges (`field_data_edges` / `common.edges_from_centers`), never uniform cells over the extent.
- Every wavelength axis / label in nm, as in the GUI (`visualization.WAVELENGTH_UNIT`,
  `core.wavelengths.wavelength_label`).
- curve_figure is the GUI's Export plot: the figure size is exact through the constrained layout (never
  bbox_inches='tight'); the GUI preview calls the same `scale_bar_layout` / `scale_bar_origin`; PlotStyle holds only
  data-independent settings (gui sub file).
- scene_figure puts every solid face (and the field planes) into ONE Poly3DCollection so mplot3d depth-sorts across
  objects; painter's sorting can still fail on interpenetrating faces and ~1e4 faces is the practical ceiling (what
  `model.scene_3d.build_scene`'s automatic n_segments aims below).

## HTML pages (mode_player, field_viewer)

- Both pages are self-contained on purpose (hand-written WebGL2, data embedded at `DATA_MARKER`, work offline in a
  talk): no external scripts, fonts or CDNs. field_viewer.html copies mode_player.html's camera, colour maps
  (`CMAPS` / `CMAP_CHOICES`), recording (`// ---------- video` block) and `.empty[hidden]` CSS: change both (the head
  comment of each page says so).
- Plot settings (field_viewer only): the page takes the keys and checks from `DATA.figure_rules` (`FIGURE_CHOICES` /
  `_RANGES` / `_LIMITS` / `_LENGTHS` of field_viewer.py); a new setting also needs its default in the page's
  `FIGURE_STYLE`, an `fs_<key>` panel input and a line in describe's export_html entry (choices and ranges there are
  read from `FIGURE_*`; defaults are prose).
- Per-sample arrays of a `ViewerLayer` follow every layer rebuild (the class docstring lists them): a new one joins all.
- One page shows one animation clock per thing: api drops the phasor partner when the deformed shape oscillates
  (`ViewerMotion.oscillate`) or a volume is attached, `viewer_data` refuses a phasor page with volumes, and the page's
  terrain does not open on phasor, motion or volume pages.
- Page start-up runs top to bottom and a block's top-level code may call shared functions (limits, syncControls,
  updateReadout) that read another block's state: such state is declared with the page constants near the top (as
  the waves `PHASOR` / `SMOOTH_OK` / `wave`), never inside a later block (a `const` read before its line throws and
  the page stays blank).
- Everything a picture must hold is drawn on the overlay canvas (`drawOverlay`), never as DOM over the view (PNG and
  recordings take the canvases only); a new overlay part also reserves its room in `viewInsets`, a new movable part
  is a `MOVABLE` entry plus a `placePart` call in its draw function.
- Key bindings live in each block's own keydown listener; taken: V flow, Shift+V arrows / particles (2D flow pages)
  and view cycling (3D volume pages; never on the same page), C contours, R terrain, D deform animation, G objects,
  M mirror, O revolve.
- Pages open flat (no automatic terrain). Flow opens as arrows; its browser memory is `FLOW_STORE` (renamed when the
  defaults change, so stored untouched defaults do not pin the old ones).
- Revolution (field_viewer `// ---- revolution ----`, api `_viewer_revolution`): the page's mirror faces are
  `unfold_faces` (api `_page_mirror_faces`, also the `_hide_void` crop), so an axisymmetric map's far half is the x_lo
  image of `AXIS_UNFOLD_TYPES` (core sub file "Mirror planes"); other types and azimuthal_order != 0 show the drawn half
  in the glass. viewer_data refuses x mirrors off the axis on a revolution page (no part of the body). While revolved
  the camera orbits in another frame (`setOrbitFrame`: UP / E1 / E2 are `let`): code reading UP assumes nothing about
  z being up; the bodies skip the axis image (whole revolutions) but take the other mirror images.
