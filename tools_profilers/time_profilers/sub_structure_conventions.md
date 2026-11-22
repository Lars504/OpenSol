# tools_profilers/time_profilers — conventions & gotchas

Usage and what is measured: `README.md` here. Root rules: structure_conventions.md. Module docstrings and header
comments hold the mechanics; this file keeps the cross-file rules.

## Layout and imports
- `solver_profiler.py` (CLI, parent: enumerates, never factorises, makes no CUDA context) -> `solver_catalog.py`,
  `report.py`, `model_cases.py`; each case is a child `python -m tools_profilers.time_profilers.case_runner`
  (`problems.py` + `runners.py`, or `model_cases.prepare` / `run_model`), started with cwd = the repository root
  (`REPO_ROOT`) so `-m` finds both packages.
- Nothing here is checked: open_sol's import_contracts, structure_index, selftests and selftest cache scan only
  `open_sol/`. The runners, problems and profiler parts mirror open_sol's consumers by name and signature (module
  docstrings name them): after changing such a consumer, a private method or a summary / info key, grep
  `tools_profilers/` and rerun the profiler that uses it.
- open_sol is imported inside functions in solver_profiler's modules (the parent stays light until it enumerates) and
  in `profiler_harness.py` (run_all loads it on its own).
- open_sol's gmsh disk cache is turned off (`OPEN_SOL_MESH_CACHE` '') by `tools_profilers/__init__.py` (every `-m` run
  and its children) and `case_profilers/profiler_harness.py` (script runs): a new entry point outside both sets it
  too, or its profile times a disk read instead of gmsh.

## solver_profiler
- Solver choices come from the registry (`solver_catalog.solver_candidates`), never a hand list; `runners.RUNNERS`
  needs an entry for every SolverType that can fill a profiled role (a missing one: KeyError, status 'error').
- The child prints exactly one `RESULT_MARKER` line; before it one `PARTIAL_MARKER` line for the assembly and one per
  finished repeat, from which the parent rebuilds a killed / timed-out case: a new case-level field the tables need
  goes into the assembly partial too.
- Timers cover the solver call only: inputs of a timed call (the scaled refactor copy, dtype casts) are made before
  its timer starts, and every reading goes through `runners._clock` (device synchronise first).
- `ProblemClass.ignored_settings` pins a setting the consumer provably ignores (the claim is checked against
  open_sol's code: see the ProblemClass docstring); never use it to drop a choice that only seems redundant.
- Matrix-free 'modes' problems build their 'linear' solver through the pcg SolverType's create and
  `definition_common.solid_linear_solver` on stand-ins of the build context (`problems.elastic_modes_system`): a
  change of either context's attributes shows up there as an AttributeError.

## case_profilers
- One `profiler_solver_<type>[_<variant>].py` per solver engine on `profiler_harness.py`; the shared part lists
  (`profiler_*_parts.py`, `_common.py`) and the harness's `case_specs` / `refined` / `without_exact_cg` are imported,
  never copied into a script. Scripts import their siblings by bare name and run as scripts (not `-m`) from any folder;
  output in `out/` (git-ignored).
- Parts are wrapped before the registry exists (`wrapped_timers` in `main`): StudyType / FeatureType callbacks keep
  the function objects of registration time, so time what a callback calls, never the callback itself.
- A target that does not resolve is listed in the report ("parts that did not resolve"), never raised. The harness's
  `SOLVER_ROLE_PARTS` names private phases (cuDSS, `_PardisoFactor` / `_MumpsFactor`, multigrid): a new solver
  module or library adapter adds its phases there, and after renaming one check that report section.
- Every timed call synchronises the GPU at entry and exit: time per-phase functions, never per-step kernels (compare
  with `--no-sync`).
- Selftest specs put the double, tolerance 1e-12 `exact_cg` node on every pcg physics; a profile on it times the
  selftest's accuracy, not production (~70 % of the tensor 3D profile once). Drop it (`without_exact_cg`) or keep
  only the setting under study on it (tensor 3D `_production`).
- `--compare` / `--solver-type` leave a role as written only when its physics' validation error names the compared
  node in quotes (`_refused_roles`): a physics validate refusing a solver node value names the node (`"... solver
  '<node>': ..."`, membrane, em_modes), else the variant runs and fails its build.
- A command's plotted point sums its runs' walls: a solver twin of a model gets its own script
  (`*_double_factor.py`, `*_3d_block_eigen.py`), not a run added to an existing one.
- A result-changing option goes into the folder name (`output_name`) or into `case_input` in `main` (the reuse hash),
  else profiles of different settings share a report or a changed profile is reused.
- `run_all_profilers.py` reads the harness's stdout markers, variant names (`variant_run` / `base_run`) and
  `PROFILED_CODE` from the harness module it loads on its own (`_standalone`), so the harness module level stays
  stdlib only, as `open_sol/model/selftest_cache.py`'s does; run_all must not import open_sol.
- `run_all_profilers.main` is shared: another suite replaces module globals (`COMMANDS`, `OUT`, `SOLVERS`, ...:
  `example_profilers/run_all_examples.py`) before calling it, so its helpers read them at call time, never bind them
  at import or as defaults.
- `profile_points.json` is the only record of first / previous / current: `load_points` never drops what it cannot
  read; a layout change or a renamed variant needs a mapping there (`RENAMED_VARIANTS`), a renamed COMMANDS entry
  starts a new series. A new library back end needs a harness `LIBRARY_GROUPS` entry (else it is grouped by array
  library and precision); a new opt-in solver type a harness `TYPE_GROUPS` entry and a run_all `_typed` SOLVERS entry
  (one `SOLVER_COLORS` slot each: more SOLVERS than slots raises).
- Keys read by name on both sides (silently empty when renamed on one): runs.json fields run_all reads
  (`wall_seconds`, `gpu_peak_bytes`, `host_peak_bytes`, `status`, `groups`, `automatic_groups`), the page data keys
  `profile_points_page.html` reads, and open_sol's `factorizations` / `eigen_factorizations` role counters (no counter
  = the run counts, so a renamed counter counts runs that never factorised).
- Reuse (`--if-changed`): the units of a profile come from the timed cProfile's code objects through
  `selftest_cache.code_units(keep=_profiled_path, module_refs=False, second_hop=False)` (no settrace: a tracer would
  slow the timed runs); code of the warm-up and of other threads is not seen. A new per-run quantity goes into the
  `RunPieces` piece, else a resumed profile loses it. `check_profiles.py` runpy-runs every script in one process under
  `CHECK_ONLY`: a script's `__main__` block reaches harness `main()` without other work, reads its options from
  sys.argv at call time and keeps module state harmless for the next script. A source run_all must see goes under
  `FINGERPRINT_FOLDERS`; a file a run writes must not (it would defeat the memory).
- Solution check: `case_specs` tags each spec (`ORIGIN_ATTRIBUTE`); deepcopy, `refined` and in-place edits keep it,
  a spec rebuilt from `to_dict()` or loaded from an example loses it ('no selftest'). The check runs on the scaled
  spec (`scaled_spec`: dx / refine, time_step / steps), so a check() whose reference depends on dx or dt reads it from
  the results (heat `_decay_rate` takes dt from the probe spacing), never from its case constants.

## example_profilers
- The performance set (`example_profiler.py` `GROUPS`, one group = one command = one plotted point): a module's
  examples by selftest wall, one or two of a few seconds, several short ones only when each is under ~2 s; span over
  engines, studies and mesh kinds before adding seconds. `run_all_examples.py` takes `GROUPS`.
- Built on case_profilers, never a copy: the harness (`OUT_ROOT` set before `main`), the shared part lists and the
  PARTS / loaders of the case scripts that profile an example; parts deduplicated by target (`unique_parts`).
- Engine parts come from the code (`engine_hooks`: lifecycle calls + the methods of `model/solver_hooks.py`'s
  Protocols, minus `UNTIMED_HOOKS`); explicit steppers leave out `step` (one GPU sync per time step).
- The examples run on `--backend` (default gpu), not on their own `solver_defaults.backend`, so a profile can differ
  from the example selftest's run (em_fdfd's dbr cavity: cpu there).
