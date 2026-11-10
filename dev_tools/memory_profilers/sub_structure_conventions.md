# dev_tools/memory_profilers — conventions & gotchas

Root rules: structure_conventions.md.

## vram_profile.py
- Run from any folder (gpu, cupy and nvmath imported at module level): `python dev_tools/memory_profilers/vram_profile.py
  <case> [more cases] [--depth 3] [--top 25] [--tag _x]`; each case runs through `api.selftest` (named, so never
  reused from the selftest cache) into `out/runs/` (run directories overwritten per case), the report goes to
  `out/<case><tag>.md` + `.json` (git-ignored; same tag = overwritten) and a verdict table per run to the console.
- Measures per estimate part, judged with `builder.estimate_verdict` beside open_sol's own `vram` verdicts:
  kept / build_peak / run from a cupy `MemoryHook` (pool used bytes, not reserved: open_sol's run part is
  reserved-based, so the two differ by pool fragmentation) marked around every `build_session` / `_run_built` call
  (sweeps: the max over builds); external against `external_peak` twice: open_sol's cuDSS counter (cuDSS's own plan
  estimates) and the device bytes of every nvmath `DirectSolver` call (`__init__`, plan, factorize, solve, free).
- Device bytes outside the pool come from memGetInfo, which on Windows / WDDM moves in 2 MiB steps and is device-wide
  (other GPU processes move it): only deltas around calls that do no cupy pool traffic are clean (the DirectSolver
  calls are), never a delta around open_sol code that allocates or frees pool blocks (`_plan_solver` frees the pool's
  cached blocks). A background memGetInfo poller was tried: it slowed runs 6 - 30 x and added nothing beyond the
  per-call deltas.
- Peak site tables: one per stage (build, study, other), snapshotted at the first free after a new stage peak (exact
  within SNAPSHOT_SLACK_BYTES).
- A case without model runs (`specs` {}: its check builds the solver directly) is profiled as one 'direct check' run
  through `api._registered_case` (pool sites and cuDSS calls, no estimate to judge against).
- It patches at import: `api._selftest_run`, `api._registered_case`, `model.study.build_session` / `_run_built` (and every loaded open_sol
  module holding them by name, found by identity; later imports read the patched study attributes), nvmath
  `DirectSolver` `__init__` / `plan` / `factorize` / `solve` / `free` (phase records keyed by id(solver), never on the
  solver: its `free()` clears its attributes; a failure inside a wrapper is listed in the report, never raised), and
  `CudssFactorization` `_plan_solver`, `_factorize_and_probe`, `refactorize*`, `solve`, `_plan`, `_device_plan`,
  `_device_index_maps`, `_device_asymmetry` (a missing one raises at import) plus its attributes `permanent_bytes`,
  `peak_bytes`, `lu_entries`, `last_ordering_reused`, `_solver`, `_factor_matrix`, `_factor_dtype`, `_pattern`.
