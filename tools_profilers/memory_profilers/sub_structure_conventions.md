# tools_profilers/memory_profilers — conventions & gotchas

Usage, output and report layout: the `vram_profile.py` docstring, `--help` and the report's own notes. Root rules:
structure_conventions.md.

## vram_profile.py
- It patches open_sol's private names at import (api's selftest run hooks, `model.study.build_session` /
  `_run_built` in every loaded open_sol module holding them, nvmath `DirectSolver` phases, `CudssFactorization`
  methods; their attributes are read at call time): a missing method raises at import, a renamed attribute inside
  the run. After a study / api / cudss_factor refactor, run it once.
- Pool bytes are the hook's used bytes; open_sol's run part is reserved-based (`gpu_memory.pool_high_water_bytes`),
  so the two differ by pool fragmentation.
- Device bytes outside the pool come from memGetInfo, which on Windows / WDDM moves in 2 MiB steps and is device-wide
  (other GPU processes move it): only deltas around calls without cupy pool traffic are clean (the nvmath
  DirectSolver calls are), never a delta around open_sol code that allocates or frees pool blocks (`_plan_solver`
  releases the pool's cached blocks). A background memGetInfo poller was tried: 6 - 30 x slower runs, nothing beyond
  the per-call deltas.

## gpu_selftests.py
- Imports open_sol from the checkout it is given (sys.path first; raises if another copy wins), so one script runs
  both sides of a comparison. It patches `api._available_backends` to ('gpu',) and names its cases / examples (named
  runs never reuse cached records): cpu runs of mixed cases are skipped, so their checks may fail or error; only the
  vram records matter. Uses api private names (`_case_estimate_specs`, `_selected_example_paths`): a rename raises.
- Light (default) runs `out/gpu_light_selection.json`, one file for both checkouts so both run the same names;
  `--pick-from` rebuilds it from a `--heavy` run's measured peaks (cases grouped by the module of their `check`).
  Cases new since that run are not in it: rebuild after adding heavy GPU cases. compare_vram_reports imports no
  open_sol at module level, since this script imports it after loading the other checkout's open_sol.

## compare_vram_reports.py
- Reads only selftest reports (no GPU): runs are matched by case / variant / run name or example path + backend, so a
  case whose spec changed between the checkouts compares different models under one name. A reused record (selftest
  cache) carries the vram of the run that stored it; edits of `selftest_cache.UNTRACKED_PATHS` files need `--reset`.
- By solver / physics type: a run's labels are the roles with a work counter above 0 (`timing.solvers`) and the
  physics types of its run directory's model.json; a run counts in every label it has, so the group sums overlap.
- The base checkout is a second worktree run as `python -m open_sol` from its own folder (the editable install points
  at the main clone; `-m` puts the current folder first). 'external' is cuDSS's plan estimate: device bytes of a moved
  case come from vram_profile.py in both checkouts.

## dissection_vs_cudss.py
- Compares `fem.dissection` counts (both `CUT_AXES`) with cuDSS's own ordering on the same clique graphs: graph
  Laplacian + I values, so the fill depends on the pattern only. Read a ratio on the site graphs first (cheap); the
  `--dofs` row is the run's Nedelec 2 pattern and needs its float32 factor on the GPU.
