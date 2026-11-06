# open_sol (package root: api, CLI) — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Gotchas of api.py / __main__.py. Current state, no changelog.

- `api.run` blocks and makes the calling thread the GPU thread: never mix it with a GUI BackendWorker in one process.
  It runs a private spec copy (ModelSpec copied, dict or file path) with `live_update_hz` = 0 and
  `RunControl(live_results=False)`; `parameters=` overrides go through `_apply_parameters` on that copy (all names
  checked first; unknown -> Issue at ('parameters',), without out_dir KeyError), so model.json records them.
  `max_wall_time` cancels at the next study checkpoint (status 'timeout', partial results; the final flush still
  runs). With out_dir, model / build / run / post-run failures never raise and summary.json + log.txt are always
  written; only argument errors and KeyboardInterrupt escape. A post-run failure (summary, save) makes the status
  'failed' (CLI exit 1); `_write_fallback_report` never raises (unwritable out_dir: files None, reason in errors,
  session still returned). Without out_dir a failed build re-raises (ModelError with `.issues`), a failed run raises
  RunError, 'timeout' returns. Default out_dir `<model dir>/<model stem>_out`.
- A Results keeps its session (GPU memory) until `close()` / `with`, which copies fields and feature results (DFT:
  stacked 'intensity_all') to the host; its leak check runs only when no live Results has a larger `build_sequence`.
  `run` warns (log + RuntimeWarning) while earlier Results are live (`r = run(...)` in a loop builds next to the old
  session); `_selftest_run` silences the Python warning. Loaded or closed Results are CPU-only; `snapshots()` /
  `snapshot_field` / `sweep_result` / `sweep_field` / `sweep()` / `sweep_figure` read only the host cache and summary
  (`_run_summary` fills it; snapshots fall back to the live session's list), so they work in every state. Non-finite
  derived values are floats live and loaded; digest statistics are 'nan' / 'inf' strings in both. Snapshots are saved
  only in results.npz; the field digest 'mean' is area-weighted.
- summary.json copies the study summary's `parameters`, `sweep_index` and `run_memory`; `Results._built_parameters()`
  = `built_parameters(spec, summary)`, so curves of a cancelled sweep or a loaded run use the parameters the results
  came from.
- Figures draw the spec's analytic curves over `_built_parameters()`: feature-result figures get their curves (series
  'ratio' / 'fft': no figure; flux spectra only '' / 'power', as the GUI), each (table key, x_column) gets
  `figures/<key>_vs_<x_column or row>.png`, line curves `figures/curve_<name>.png` while the session is live (or, with a
  `snapshot`, from the host snapshots); a failing curve fails only that figure (logged warning). Sweep runs add
  `_sweep_figure_jobs`, a derived series `figures/derived_series.png`; `figure()` stays the run's own (last value)
  result. The job list is ONE `Results._figure_jobs(series)` for run --figures and CLI `show <run dir> --figures`
  (`--series` applied only where the result has it): a new figure kind goes there.
- summary.json `curve_error` is computed in `_run_summary` while the session is live (line curves need `Results.field`
  / snapshots; `_line_profile` is shared with the line figure); per sweep value (feature targets only) over
  `parameters_with(spec, parameter, value)` like the GUI Sweep tab. A failing curve becomes a NaN record with a 'note',
  never a run failure.
- 3D scene figures: `api.scene_figure` is CPU only (tab10 material colours: api never imports the GUI palette);
  `Results.scene_figure` uses `_built_parameters()` and mixes no times (closed / loaded: stored cuts only from a last
  snapshot at the saved field's time). `_scene_figure_jobs` is part of `_figure_jobs`.
- selftest progress is a dict stream: `_SelftestPlan` events ('begin' per phase, 'start' per test, 'skipped') and the
  finished records; the CLI dispatches on 'event' (`_print_selftest_record`); result lines carry no name (the start
  line has it). `selftest_examples` = `_plan_examples` (load, skips, reuse; KeyError for unknown names before anything
  runs) + `_run_examples`.
- Selftest run-time estimate: the 'start' event carries `_specs_time_estimate` of the test's runs (case: own and
  variant specs on available backends; example: its spec on the forced backend; plus `estimate_overheads` build /
  output), printed as a low - high span; the finished record gets `time_estimate` (`_time_estimate_after`: `split`
  {total, build / study / output / other: {seconds, estimate}}). Measured: build / study from the run summaries,
  output = the run() call minus summary wall_time, other = case wall time minus those (checks, set-up, spec making).
  The study estimate uses the runs' measured counts, build / output the guesses before. Estimates never stop a
  selftest (failures become its message). A sweep run's time_check covers one value, the estimate every value.
- `selftest_examples` runs each example with `save_arrays=False` and closes its Results before the next; run
  directories `<out_dir>/examples/<path without .json>/`, each example once on its own backend (records carry the
  run's `backend`; one setting 'gpu' is `skipped_backend` without a usable GPU).
- A new argument in `api.py` / `__main__.py` is added to `model/describe.py` by hand (`api_reference_current` only
  compares the .md with describe's text), then `describe --update-reference`.
