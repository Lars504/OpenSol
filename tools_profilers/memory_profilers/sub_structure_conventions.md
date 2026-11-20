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

## dissection_vs_cudss.py
- Compares `fem.dissection` counts (both `CUT_AXES`) with cuDSS's own ordering on the same clique graphs: graph
  Laplacian + I values, so the fill depends on the pattern only. Read a ratio on the site graphs first (cheap); the
  `--dofs` row is the run's Nedelec 2 pattern and needs its float32 factor on the GPU.
