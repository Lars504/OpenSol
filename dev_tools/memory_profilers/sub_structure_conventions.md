# dev_tools/memory_profilers — conventions & gotchas

Root rules: structure_conventions.md.

## vram_profile.py
- Run from any folder (gpu, cupy imported at module level): `python dev_tools/memory_profilers/vram_profile.py
  <case> [more cases]`; each case runs through `api.selftest` into `out/runs/`, the report goes to `out/<case>.md` +
  `.json` (git-ignored).
- Measures: a cupy `MemoryHook` (live pool bytes by open_sol allocation site, stage build / study, snapshots near
  the build and run peaks), cuDSS plans (memGetInfo minus the pool around `_plan_solver` / `_factorize_and_probe`,
  fill per n log2 n and n^(4/3)), and reconciles them with the build's `gpu_estimate` through
  `builder.estimate_verdict` (estimate just below measured = ok; 'over' / 'unaccounted' are leads on the solver).
- It patches private names at import: `api._selftest_run`, `build_session` / `_run_built` in model.study and every
  module importing them by name (study_adaptation, study_estimation, study_sectors), and `CudssFactorization`
  `_plan_solver`, `_factorize_and_probe`, `refactorize*`, `solve`, `_plan`, `_device_plan`, `_device_index_maps`,
  `_device_asymmetry` plus its attributes `permanent_bytes`, `peak_bytes`, `lu_entries`, `_factor_matrix`,
  `_pattern`. A rename there breaks it at import (AttributeError); a new module importing build_session /
  _run_built by name needs adding to the patch loop, else its runs are not staged.
