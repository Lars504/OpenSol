# open_sol (package root: api, CLI) — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Gotchas of api.py / __main__.py. Current state, no changelog.

- `api.run` blocks and makes the calling thread the GPU thread: never mix it with a GUI BackendWorker in one process.
  It runs a private copy (live output off; `parameters=` / `backend=` set on it, so model.json records them;
  `snapshots=False` writes `StudyType.final_state_settings` into the copy only). Selftest examples pass
  `snapshots=False` unless an analytic curve names a snapshot; `_selftest_run` fills only the final-state settings a
  case leaves unset. With out_dir only argument errors (an unreadable model file's OSError included) and
  KeyboardInterrupt escape (every other failure is status 'failed' with summary.json + log.txt); without, a failed
  build re-raises its exception (model errors: ModelError with `.issues`), a failed run raises RunError, and
  'timeout' (max_wall_time, checked at study checkpoints only) returns partial results.
- A live Results holds its session (GPU memory) until `close()` / `with`. `r = run(...)` in a loop builds next to the
  old session: `run` warns (log + RuntimeWarning) while earlier Results are live, and a close skips its leak check
  while a later-built Results is live.
- Whatever is drawn or compared over a run's results (curves, figures, scene figures, summary `curve_error`) takes the
  parameters the results came from, `Results._built_parameters()` (summary 'parameters'; per sweep value
  `analytic.parameters_with`), never `evaluate_parameters(spec)`: that is wrong after a cancelled sweep and for a
  loaded run. `curve_error` is computed in `_run_summary` while the session is live (line curves read the field);
  a failing curve is a NaN record with a 'note', never a run failure.
- Every figure of a run comes from ONE `Results._figure_jobs` (run figures=True and CLI `show <run dir> --figures`):
  a new figure kind goes there.
- A new argument in `api.py` / `__main__.py` is added to `model/describe.py` by hand (`api_reference_current` only
  compares the .md with describe's text), then `describe --update-reference`.
