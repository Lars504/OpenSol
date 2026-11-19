# open_sol/io — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). File formats and results IO. Current state, no changelog.

- Every file is written through '<file>.partial' + replace (`_write_text_atomic`, `_write_npz`, `write_json`,
  `save_model`), except inside `save_results` (`atomic=False`), which deletes the old summary.json first and writes it
  last: a summary.json present vouches for every file its 'files' lists, and `load_results` reads only those.
- Writers and readers are pairs: `export_result` / `export_curve_plots` with `read_curve_data` (`_result_plots` draws
  a read result as the GUI result views do), `_results_arrays` with `_read_arrays`, `_snapshot_field_entries` with
  `_snapshot_fields`; each lists the saved entries (FieldData attributes, result kinds, CSV metadata keys) explicitly,
  so a new one is added on both sides. Text metadata that must not be JSON-decoded is a `_CSV_TEXT_KEYS` key.
- Non-finite floats: `json_safe` writes 'nan' / 'inf' / '-inf'; loading maps them back to floats only in the numeric
  slots `load_summary` names (and result scalars / table cells, `_restore_nonfinite`), so those match live while
  digests keep the strings. A new non-finite numeric summary slot is added to `load_summary`, else loaded runs differ
  from live ones.
