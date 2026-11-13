# open_sol/io — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). File formats and results IO. Current state, no changelog.

- model_io: `save_model` / `write_json` / `_write_npz` write '<file>.partial' then replace (`atomic=False`: in place,
  only inside `save_results`, which deletes the old summary.json first and replaces only summary.json, written last:
  a summary.json present vouches for every file it lists); `load_model` reads
  'utf-8-sig' and re-raises ValueError as `ValueError(f'{path}: {exc}')`; OSError propagates. Model files are written
  with 'meshes'; a legacy 'mesh' object is read as meshes = [it] only by `ModelSpec.from_dict` (load_model, api.load and
  GUI undo snapshots share it). npz 'metadata' is a 0-d unicode JSON string. CSV starts with '# key: value' lines
  (strings as is only when they read back unchanged, `_plain_metadata_text`; line breaks, edge spaces, a leading quote
  or, outside `_CSV_TEXT_KEYS`, JSON-like text such as '1' go as a JSON string), then the csv-quoted column-name row
  (curve labels are free text, line breaks included). A dft_fields CSV needs a result fetched with `freq_index`;
  far-field columns `power_<nm>nm` get more digits until unique; table numeric extras go to the metadata lines.
- Field exports carry `plane` (3D cut), the mesh name (metadata 'mesh') and, .npz, x / y centres plus edges
  (`viz.common.edges_from_centers`, as results.npz); a native polar / spherical FieldData (`mapped_centres` set) adds
  'x_mapped' / 'y_mapped' (.npz) or x / y columns after the axes (.csv); a triangle-mesh FieldData adds 'nodes' /
  'triangles' / 'node_values' (.npz). `export_volume` writes VolumeData as .npz only (a .csv path raises).
- `export_curve_plots` writes {plot name: xy} (.npz: index keys 'plot<i>/x' / 'plot<i>/series<j>', since plot and
  series names are free text; names / axes / units in the metadata 'plots' list, kind 'curve_plots'; .csv: one plot
  via export_result). `read_curve_data` is the inverse -> {plot name: xy}, also for a plain xy export, an
  export_result of timeseries / flux_spectrum / far_field (one plot per wavelength, dB below the peak vs degrees, as
  the GUI) / table (`analytic.table_xy` over its first number column when it has two or more), or any numeric CSV
  (first column x, named after the file). It reads `_CSV_TEXT_KEYS` back as text (a quoted one JSON-decoded), other
  metadata as JSON, else text. Only the leading '#' block is metadata (later '#' rows skipped); column names of an
  open_sol export (metadata 'kind') are kept exactly, foreign ones stripped; repeated curve column names raise
  ValueError. np.savetxt writes '\r\n' on Windows, so the reader folds '\r\n' to '\n'.
- results_io: `json_safe` writes non-finite floats as 'nan' / 'inf' / '-inf' (`read_json` keeps the strings) and
  refuses arrays over 10,000 elements (use results.npz + a digest). `save_results` writes summary.json last, its
  'files' naming what was written (a summary 'convergence' report also goes to convergence.json); `load_results` reads
  only those and raises ValueError for another format / version. results.npz holds >= 1-D numeric arrays, also one
  level inside dict values (`<key>/<name>`, e.g. xy `series/<name>`; '/' in a result key or name raises); other
  entries go to the metadata JSON (tuples come back as lists).
- Loading maps exactly 'nan' / 'inf' / '-inf' back to floats only in the numeric slots (`load_summary`: derived,
  derived_reference, derived_error, sweep derived / reference / error, curve_error max / rms (top level and sweep),
  fixed_point and wavelength_fixed_points relative_change, derived_series values (top level and sweep: lists in
  summary.json, not arrays), convergence level value / relative_change / derived; `_restore_nonfinite`: top-level
  scalar result entries and table rows), so those match live; digests keep the strings. A new non-finite numeric
  summary slot must be added there.
- Saved fields and snapshots keep `plane`; loading rebuilds saved fields' `FieldData.axes` from it (`_plane_axes`);
  snapshot fields store 'axes' and, native polar / spherical cells, `mapped_centres` as '<key>/mapped_x' / 'mapped_y'.
  Snapshot stored cuts go to 'snapshot/<i>/cut<c>/<key>/...' (metadata 'cuts'); kept sweep snapshots to
  'sweep_snapshot/<value>/<i>/...' through the same `_snapshot_entries` / `_read_snapshot` (metadata
  'sweep_snapshots'). Saved fields and snapshot fields keep their mesh name (metadata 'mesh'); `ResultsData.meshes`
  goes to meshes.npz ('<mesh>/x_edges' ..., triangles '<mesh>/points' / 'triangles' / 'pml_faces', tetrahedra
  '<mesh>/points' / 'tets'; written with the arrays only; `read_meshes`). Tetrahedron-mesh fields are plain uniform
  rasters of the sample lattice. Triangle-mesh fields are saved as their raster plus the triangulation: node values
  under '<field prefix>node_values', the node table once per distinct table as 'triangulation/<sha256[:16]>/nodes' /
  'triangles' (metadata 'triangulation' = that digest), so snapshots of one mesh share it. Parity sectors: per sector
  'sector/<i>/<key>/' (metadata 'sector_results'), combined 'sector_combined/<key>/' (metadata
  'sector_combined_results'). Flux / far-field / xy digests take max and argmax over finite values
  (`_finite_argmax`). CLI `show` reads only summary.json (`load_summary`) unless `--figures`.
- Result kinds 'xy' / 'table': formats in `RESULT_KIND_DOCS`, producer extras in the feature description. xy .csv = x
  + series columns; table .csv via the csv module, table .npz metadata only. result_plots draws xy as one panel per
  series unit and tables as their first 30 rows; `result_plots.xy_series_groups` / `table_cell_text` are shared with
  the GUI 'Curves & tables' tab. A timeseries whose samples all share one t (stationary iterations) plots against the
  1-based iteration index (`analytic.is_iteration_series`; no FFT) in `_draw_timeseries` and the GUI Probes / Sweep
  tabs alike.
