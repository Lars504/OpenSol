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
- open_sol keeps gmsh tetrahedron meshes in a disk cache by default: `tools_profilers/__init__.py` (every `-m` run and
  its children) and `case_profilers/profiler_harness.py` (script runs, example_profilers too) set
  `OPEN_SOL_MESH_CACHE` '' so a profile times gmsh; a new entry point outside both sets it too.

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
  '/name=value', so a new choice param multiplies the rows instead of running silently at its default. The one
  exception is `ProblemClass.ignored_settings` {type: (setting, ...)}: a setting the problem's consumer provably
  ignores is pinned to what its default resolves to (`_pinned_values`; 'auto' libraries through `_resolved_value`, so
  the row keeps the Automatic / baseline label: solid_modes_3d's `arpack_shift_invert:superlu`). It is a claim about
  open_sol's code (arpack's factorization on brick meshes: `eigsh_inverse`, no factor): recheck it when that consumer
  changes, and never use it to drop a choice that only seems redundant.
- Matrix-free problems (`Problem.kind` 'modes', operator None: `n_unknowns` = free dofs, `nnz` None) assemble the
  physics' own system on the run backend in `runners.backend_arrays` (`problems.elastic_modes_system`, before the
  memory baseline; its time is the case's `transfer_seconds`), one system for every repeat. Its 'linear' solver is
  resolved from the role's Automatic in the problem's study and grid (`automatic_linear_values`), never spelled out,
  and built as the pcg create + `solid_linear_solver` do (max_iterations 0 = twice the auto limit): check both when
  they change. The mode runners call `ElasticSystem3D.lowest_modes` (its dispatch on the eigen runtime's type picks
  eigsh_inverse or the block path) and read the ModalSolve `info` keys of both paths ('eigen_seconds',
  'operator_applications', 'cg_iterations', 'lanczos_tolerance', 'preconditioner', 'block_residual', ...) with
  `.get`: a renamed key silently empties a detail, not the run. The closed-form reference (`_roller_box_reference`)
  parametrises selftests_eigenfrequency's 3D roller box: it is exact only for uniform cells, rollers on all six faces
  and the trilinear consistent mass; a change of geometry or supports there needs a new reference, not a looser one.
- Frequency sweeps (`Problem.kind` 'sweep', `acoustic_sweep_2d`): the problem holds the mesh and omegas, its
  `operator` / `rhs` are only the middle frequency's reduced A and load (counts); `runners._run_sweep` builds a fresh
  `problems.sweep_system` on the runtime per repeat (assembled before the timer) and calls `solve(omega)` in order,
  after `plan_batched(omegas)` for batched_direct, as the pressure_acoustics hook does: check `sweep_system` against
  the physics' system build (coefficients, impedance as Robin damping) and the hook when they change. Its phases are
  not factor / solve / refactor of one system (factor = the first frequency, solve = the mean later one): its
  rows compare by `total`. batched_direct has no library setting, so `solver_catalog._unavailable` names it by type
  (gpu without nvmath-python); its `batch_size` is an int and stays at its default.
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
- The harness output folder is `output_name(...)`: the profile name + backend (+ `automatic` for
  `--automatic-solvers`) + `--compare` (+ `only`) + `--set`, so
  profiles of different solvers never share a report / runs.json; a new result-changing option belongs in that name.
- `run_all_profilers.py` reads each command's points from the `runs.json` next to the report it printed, and only
  when that file is newer than the command's start. `profile_points.json` is the only record of first / previous /
  current: `load_points` never drops what it cannot read (it raises; the old one-solver layout maps to 'auto @gpu'),
  `save_points` replaces the file whole, and every run first copies it to its history folder. A layout change needs
  a mapping in `load_points`, never a fresh start. The Automatic ring reads `runs.json` `automatic` (`_automatic_choices`:
  per role used, `SolverRole.automatic('gpu', grid, study)` over the type's param defaults, back_end / factorization
  'auto' through `auto_back_end_values`, then `expected_back_end`), from any variant's point of the command; it ignores
  the spec's nodes and a runtime cuDSS fallback. Points hold solver groups (harness `solver_group` / `run_groups`:
  runs.json `groups` / `automatic_groups`), and a group name is its SOLVERS key, so the ring needs no mapping. The
  plot draws only `plotted_variants` (auto per backend + the five groups on their backend) and only COMMANDS entries:
  other variants / commands stay in profile_points.json, undrawn; a renamed flag needs RENAMED_VARIANTS to stay drawn. A new library back end needs a `LIBRARY_GROUPS` entry, else it is grouped
  by array library and precision; a renamed variant needs a RENAMED_VARIANTS entry so `load_points` keeps its points.
  One ring per column (`auto_picks`): the column's runs' `run_automatic` groups; exactly one -> that group's line,
  several or the line not run -> auto @gpu when its `chosen` holds them. Ringing every picked group gave two rings.
  SOLVERS `backends` is a flag's default backend, `runs_on` what an explicit `--backend` may pick (and the lines
  `plotted_variants` draws); a single-group ring looks for the gpu line first. The groups come from ONE point, auto @gpu first, the --precision groups last: `automatic` is worked out on the
  variant's spec, so a forced precision changes it, and a union over variants (or old back end names next to group
  names, hence LEGACY_GROUPS on both sides) failed the subset test and left most columns without a ring. In a split
  column the source is the first point whose `run_automatic` covers the column's runs (then one that measured them),
  and the ringed line must hold `run_walls` for them: an auto @gpu point from before per-run fields read first left
  split commands (heat, mechanics_3d) ringless.
  Split columns (`plot_columns`) match runs across variants by `base_run` (the run name without the harness's
  ' SETTING=value' compare suffix): a new suffix format in `setting_variants` must keep that parse, else every compared
  run becomes its own column. The 'skipped' log line keys on the harness's NOTHING_PROFILED text. The example
  suite adds `--automatic-solvers` to every variant (nodes dropped); the case suite's auto runs as written.
  A renamed `back_end` / `factorization` / `precision` param silently regroups. A changed `runs.json` key (`wall_seconds`, `gpu_peak_bytes`,
  `host_peak_bytes`) silently empties that panel; the log line of an auto command names the groups its runs used
  (`used_solvers`), and falls back to 'auto' without a runs.json; its
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
  harness's 'nothing profiled' line (`NOTHING_PROFILED`): keep the two texts the same. Likewise `PLANNED_GROUPS`:
  profile_runs prints the groups the runs' roles take (spec nodes, else Automatic) before the build, and run_all
  (streaming stdout, `run_command`) prints the auto variant's '>>>' line from it. `--compare` leaves out the roles
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
  main and its helpers read those globals at call time, never bind them at import. `SOLVER_EXCLUDED_COMMANDS`
  {group: entries} skips a variant whose answer is wrong and drops its points (examples: cudss_mixed's single
  precision on 'magnetic', ~8 % off with iron).
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
