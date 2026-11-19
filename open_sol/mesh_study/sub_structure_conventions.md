# open_sol/mesh_study — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). The solver_plan S7.8 mesh study (CLI mesh-study);
the gates and their rules are `analysis.gates`. Current state, no changelog.

- Parts (`PARTS`, run order) each expose `run(options: StudyOptions)`; a part that raises is printed and the others
  still run. A new part adds a module + a PARTS entry and, when it gates something, a record in `analysis.gates`.
- Records: one JSON line per run in `<out>/records/<part>_<case>.jsonl`, appended also for check_failed / failed runs;
  `read_records` keeps the last line per label (family + h + the variant's sweep values), so --force /
  --retry-failed re-runs supersede older lines. `run_variants` refuses repeated labels within a case; the prototypes
  part appends to `accuracy_coax.jsonl` too, so its families must not reuse the tensor runs' labels.
- Variants change a model only through `apply_variant` (parameters, the one mesh, `solver_defaults` flags, one solver
  node `mesh_study_solver`). The study models whose solver settings matter carry their own pcg node (`models/*.json`:
  precision double, tolerance 1e-10).
