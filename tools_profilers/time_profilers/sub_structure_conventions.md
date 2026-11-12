# tools_profilers/time_profilers — conventions & gotchas

Usage and what is measured: `README.md` here. Root rules: structure_conventions.md.

## Layout and imports
- `solver_profiler.py` (CLI, parent; cpu and gpu backends, synthetic problems and whole models) -> `solver_catalog.py` (enumeration per
  backend), `report.py` (machine record, summaries, markdown), `model_cases.py` (model targets); each case runs
  `python -m tools_profilers.time_profilers.case_runner` (child: `problems.py` assembly + `runners.py` per SolverType, or
  `model_cases.prepare` / `run_model` for a whole api.run). The parent never factorises and never makes a CUDA context
  (its GPU check is a device count; the device name comes from the children).
- tools_profilers imports open_sol, never the reverse; open_sol's import_contracts, structure_index and selftest cache scan
  only `open_sol/`, and `pyproject.toml` ships only `open_sol*`, so nothing here is checked or pip-installed (the repository is public: a clone has it). open_sol is
  imported inside functions (problems / runners / catalog), so the parent stays light until it enumerates.
- Run from the repository root: the child is started with cwd = the repository root (`REPO_ROOT`) so `-m` finds both
  packages; the script form inserts the root into sys.path.

## Rules
- Solver choices come from the code, never a hand list: a new SolverType, back end or choice value shows up through
  `solver_catalog.solver_candidates` (registry + `solver_misfit` per backend + SolverType validate). A new problem
  class names the physics type / role / study / grid kind it stands for; `runners.RUNNERS` needs an entry for every
  SolverType that can fill a profiled role (a missing one raises KeyError in the child: status 'error').
- Runners call open_sol the way its consumers do (SparseDirectSolver.factorize with symmetric / symmetrizer for host
  direct FDFD factors as fdfd_2d, DiffusionDirectSolver for SPD, solve_resonances with the model runtime's
  mixed_min_unknowns): check those call sites when their signatures change. `_run_diffusion_direct` reads the
  private `DiffusionDirectSolver._factor` for details only.
- The child prints exactly one `RESULT_MARKER` line (libraries may print to stdout too); every exception becomes a
  status there, never a traceback without a result. Before it, one `PARTIAL_MARKER` line for the assembly and one
  per finished repeat: the parent rebuilds a killed / timed-out case from them, so a new case-level field the
  tables need goes into the assembly partial too.
- Backends: a Candidate's `label` is backend-free (what --solvers patterns match), its `row` ('<label> @gpu') keys the
  records and report rows; the baseline is always a cpu candidate (host SuperLU, else the cpu Automatic), so the
  catalogue always enumerates the cpu backend. Runners take `BackendArrays` (`runners.backend_arrays`: the device
  copies are made once per case, before the memory baseline) and hand each consumer the arrays it gets in a gpu model
  (check the consumer when it changes: line_block and solve_resonances take host operators); residuals are computed on
  the host. Every timer reading goes through `runners._clock` (device synchronise first).
- Model cases assign the choice as a new solver node (`model_cases.assign_solver`), never by editing the node the
  model has (it may be shared with other physics). Their phases come from summary.json (`build_seconds_total`,
  `study_seconds`): a renamed summary key silently empties the b / st / o cells.
- Timers cover the solver call only: inputs of a timed call (the scaled refactor copy, dtype casts) are made before
  its timer starts. Every runner hands `_repeat` a tolerance, from which `_repeat` sets `inaccurate` (the report
  reads `ACCURACY_MARGIN` from runners for its "> tol" mark).
- The catalogue varies every 'choice' param of a SolverType; a choice its short label does not name is appended as
  '/name=value', so a new choice param multiplies the rows instead of running silently at its default.
- Thread variables are set in the child environment before it starts (pools read them at library load), never by
  the child itself.

## case_profilers
- One `profiler_solver_<type>[_<variant>].py` per solver engine (tensor / triangles / tetrahedra), each a list of
  'module:Class.method' parts on `profiler_harness.py`; `profiler_*_parts.py` / `_common.py` are shared part lists
  (their `methods` / `functions` part builders and the harness's `case_specs` / `refined` are imported, never copied
  into a script); `run_all_profilers.py` runs every script in a subprocess and keeps each report and its `runs.json`
  in `out/history/<timestamp>/` (`<folder>.txt`, `<folder>.runs.json`); `profile_delta.py <folder>` prints the parts
  that changed between the two newest kept `runs.json` (it reads `wall_seconds` and `parts[].exclusive_seconds` /
  `calls`: a renamed key empties it). Scripts import their siblings by bare name (own folder on sys.path) and find the
  repository from `__file__`, so they run from any folder as scripts, not with `-m`. Output: `out/` next to the
  scripts (git-ignored).
- Parts are wrapped before the registry exists (`wrapped_timers` in `main`): StudyType / FeatureType callbacks keep
  the function objects of registration time, so time what they call instead of the callback itself.
- A target that does not resolve is listed in the report ("parts that did not resolve"), never raised: after a
  refactor check that section, or the part silently drops out of the table.
- Every timed call synchronises the GPU at entry and exit: time per-phase functions, not per-step kernels
  (or compare with `--no-sync`). The exit synchronize is timed alone as the part's device wait (record field 4,
  `EMPTY_RECORD` layout): cpu/wall cannot tell device-bound parts (the blocking sync spins a core), the wait column
  only partly can: it counts the work still queued at exit, so a device-bound part ending in a host read (a PCG solve
  reading its residual) shows ~0 wait.
- Selftest specs assign the double, tolerance 1e-12 Jacobi `exact_cg` node (`with_exact_cg`) to every pcg physics; a
  profiler run on it times the selftest's accuracy, not production (it was ~70 % of the tensor 3D profile). Profilers
  drop the node (Automatic) or keep only the setting under study on it (`scaled_spec`, tensor 3D `_production`).
- `SOLVER_ROLE_PARTS` includes private cuDSS / DevicePencil / host direct (`_PardisoFactor` / `_MumpsFactor`) /
  multigrid phases by name: a new solver module or library adapter needs its phases added there. A script that also lists one of its
  targets gets "already timed" in the unresolved section (drop it from the script). A renamed private method drops
  out the same way, so check that section after a cuDSS refactor.
- The harness output folder is `output_name(...)`: the profile name + backend + `--compare` (+ `only`) + `--set`, so
  profiles of different solvers never share a report / runs.json; a new result-changing option belongs in that name.
- `run_all_profilers.py` reads each command's points from the `runs.json` next to the report it printed, and only
  when that file is newer than the command's start. `profile_points.json` is the only record of first / previous /
  current: `load_points` never drops what it cannot read (it raises; the old one-solver layout maps to 'auto @gpu'),
  `save_points` replaces the file whole, and every run first copies it to its history folder. A layout change needs
  a mapping in `load_points`, never a fresh start. The Automatic ring reads `runs.json` `automatic` (`_automatic_choices`:
  per role used, `SolverRole.automatic('gpu', grid)` over the type's param defaults, back_end / factorization 'auto'
  through `auto_back_end_values`, then `expected_back_end`), from any variant's point of the command; it ignores the
  spec's nodes and a runtime cuDSS fallback. A back end name that is not a SOLVERS key (or a renamed
  `back_end` / `factorization` / `preconditioner` param) silently drops the rings. A changed `runs.json` key (`wall_seconds`, `gpu_peak_bytes`,
  `host_peak_bytes`) silently empties that panel; the log line of an auto command names the solvers its runs used
  (`used_solvers`: `solvers[]` back_end, else preconditioner, else type), and falls back to 'auto' without them; its
  bracketed time is the `command_seconds` of the variant's current point before the run replaces it (the whole command,
  not the plotted `run_seconds`). On the cpu backend summary.json `build.gpu_peak_bytes` is the
  MemoryLedger's tracemalloc peak (host bytes), so `_run_memory` writes null there instead of reading it, and the plot skips `@cpu` variants in the GPU panel
  (old points included). The points are keyed by '<solver> @<backend>', then the COMMANDS
  entry: renaming an entry starts a new series and leaves the old one plotted at the end of the axis. A command's
  point is recorded only when every run whose name holds the solver's `NAME=value` token (the harness's variant
  label; all runs for auto) completed (`status` 'completed' of summary.json, copied into runs.json): one failed run
  records nothing (log 'FAIL'; 'pass' = exit 0 and every run completed) and the earlier point stays. A change of that label drops every compare point; and only those in which a role holding the value factorised (`runs.json`
  `solvers[].factorizations`, from the role's `factorizations` / `eigen_factorizations` counter; None = no counter,
  counted): a matrix-free eigen solve or a line_block run under `back_end=pardiso` measures another solver. A command
  whose every run of the solver is such a run loses that solver's points of the command (the one case where points
  are dropped; `profile_points_before.json` of the run keeps them). A renamed counter makes every run count again. A timed-out command counts as profiling unless its partial stdout holds the
  harness's 'nothing profiled' line (`NOTHING_PROFILED`): keep the two texts the same. `--compare` leaves out the roles
  that refuse the value (setting_choices, opt-in values, validation: below), so no run fails for the value itself. The plot is saved to a timestamped sibling
  and then replaces `profile_points.png`; when a viewer holds the png open the new one stays beside it. A new SOLVERS entry takes the next colour slot (10 in all: 8 hues, 2 neutrals) and names its fixed
  `backends` (host libraries cpu, cuDSS gpu) or None to follow `--backend`. On the cpu backend a case's own gpu-only
  nodes (cuDSS, device line_block) fail at build and those runs are not counted.
- Time limit: `run_all_profilers` passes `--timeout` to the harness as `--time-limit` (seconds since the harness's
  `PROCESS_START`) and kills only `REPORT_GRACE` s later. `profile_runs` arms a `_TimeLimit` timer
  (`_thread.interrupt_main`) and catches the KeyboardInterrupt (also Ctrl+C) per run, so the timed parts unwind
  through `_call`'s finally and count up to the interrupt; it then writes the report with a first line starting with
  `PARTIAL_REPORT_MARKER`, which run_all reads (`PARTIAL_REPORT`: keep the two texts the same) to log the command as
  timed out with its report kept. The limit is armed in profile_runs only: spec loading and `--compare` validation
  before it are covered by the kill alone. api.run turns the interrupt into a 'cancelled' summary; the harness
  overwrites the status as 'timed_out' when the limit fired.
- `LIBRARY_CLOCKS` reads a seconds attribute the instance sets inside the wrapped call (cuDSS `factorize_seconds`).
  It counts only a new value, so keep that attribute assigned fresh on every call.
- `--compare` adds a solver node per (physics, role) to a deep copy of the spec, never editing the spec's nodes. Values
  stay strings, so it suits choice settings (`back_end`, `precision`). A node whose type lacks the setting (an
  example's exact pcg node under `back_end`) is replaced by the role's Automatic type and settings on that backend /
  grid / study when that type has it (2D tensor diffusion on the cpu: sparse_direct); else the role is skipped. `SETTING_ALIASES` pairs `back_end` with the
  shift-invert `factorization`; a value outside the role's `setting_choices` is skipped, never an invalid node, and so
  is an opt-in value (cuDSS, MUMPS / PARDISO: `SolverType.opt_in_settings`) the role does not list (em_modes / em_fdfd_bor
  'shifted' does real work only as line_block); a compared node an error of the variant's validation names (a physics
  refusing the value itself: the 1D membrane) is dropped again (`_refused_roles`), so the run keeps its other roles. A run
  that differs from an earlier one only by a node the value replaces (a mixed / double twin of one model) is the same
  model under that value and is dropped (`_run_key`: the specs' JSON without the displaced nodes).
- A command's point sums its runs' walls, so a run added to a script doubles the plotted point: a solver twin of a
  model gets its own script (`*_double_factor.py` reuses its sibling's PARTS / example loader), not a run in the
  existing one.
- `run_all_profilers.main` is shared: another suite replaces its module globals `COMMANDS`, `OUT`, `POINTS_PATH`,
  `PLOT_PATH`, `PLOT_TITLE` and `command_line(entry)` before calling it (example_profilers/run_all_examples.py), so
  main and its helpers read those globals at call time, never bind them at import.
- Report caller lists are the harness's own (`_caller_lines`: calls, own s, cumulative s per caller, largest first):
  callers of the outside functions below `CALLER_MIN_SECONDS` fold into one line, the sync / transfer section keeps
  every caller (a sync site matters at 0 s; callers of `syncdetect._declare_synchronize` are the implicit syncs).

## example_profilers
- The performance set: analytic examples (`examples/<folder>/<stem>.json`, loaded as the example selftest loads them,
  never refined or edited) grouped per physics module in `example_profiler.py` `GROUPS` (one group = one command =
  one plotted point, its examples the runs). Choice rule: a module's examples by selftest wall, one or two of a few
  seconds, several short ones only when each is under ~2 s; span over engines, studies and mesh kinds before
  adding seconds. A new example group needs no other edit (`run_all_examples.py` takes `GROUPS`).
- It is built on case_profilers, never a copy: `profiler_harness` (its `OUT_ROOT` set to `example_profilers/out`
  before `main`), the shared part lists (`profiler_em2d_parts`, `profiler_diffusion_parts`,
  `profiler_em_bor_fdtd_common`) and the PARTS / loaders of the case scripts that profile an example
  (`profiler_solver_em_fdfd_3d*.py`, `profiler_example_sin_square_membrane.py`): deleting one of those breaks the
  import here. Parts are deduplicated by target (`unique_parts`).
- Engine parts come from the code: `engine_parts('module:Class', explicit)` times the `ENGINE_HOOKS` the class or an
  open_sol base defines; `explicit` (FDTD, elastic_waves, laminar_flow, heat beside a flow) leaves out `step`, which
  would sync the GPU once per time step (the runner's `advance` times the stepping). A new study hook in
  `model/solver_hooks.py` belongs in `ENGINE_HOOKS`.
- The examples run on `--backend` (default gpu, as the case profilers), not on their own `solver_defaults.backend`
  (em_fdfd's dbr cavity runs on cpu in the selftest; its SuperLU node keeps the factor on the host either way).
