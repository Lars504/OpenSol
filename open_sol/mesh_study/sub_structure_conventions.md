# open_sol/mesh_study — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). The solver_plan S7.8 mesh study; what it measures and
the gates: `mesh_study_plan.md` (repository root). Current state, no changelog.

- Not product code: imports `open_sol.api` (and through it everything but gui), so only `__main__._mesh_study` imports
  it, inside the handler; nothing else in open_sol does. The audit part skips its own files.
- Parts (`PARTS`, run order) each expose `run(options: StudyOptions)`; a part that raises is printed and the others
  still run. A new part adds a module + a PARTS entry and, when it gates something, a row in `analysis.gates` and in
  the plan's gate table.
- Records: one JSON line per run in `<out>/records/<part>_<case>.jsonl`, appended also for check_failed / failed runs;
  `read_records` keeps the last line per label, so --force / --retry-failed re-runs supersede older lines. Label =
  family + h + the variant's sweep values: two variants of one case must differ in it, or the second is skipped as
  already recorded. The prototypes append to `accuracy_coax.jsonl` beside the tensor runs.
- One-mesh models: a variant replaces `spec.meshes` by its one MeshSpec (physics mesh ''); per_physics takes
  `physics_objects` / `smallest_side` from model.meshing (the auto rule's own helpers).
- Variants change a model only through `apply_variant`: parameters, the mesh, `solver_defaults` flags and one solver
  node `mesh_study_solver` (replaced nodes nobody references are dropped, so no unused-node warning). Models whose
  solver settings matter carry a pcg node themselves (`models/*.json`: precision double, tolerance 1e-10).
- Coax, the polar annulus and the FDFD line source are read from `examples/` (`load_example`, sweep removed): the study
  runs from a repository checkout only. `models/*.json` ship as package data.
- Counters come from summary.json `solvers` (`analysis.pcg_counters`: the role whose counters hold cg_iterations); a
  renamed pcg counter breaks the PCG table silently (empty iterations).
