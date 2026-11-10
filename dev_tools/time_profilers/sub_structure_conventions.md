# dev_tools/time_profilers — conventions & gotchas

Usage and what is measured: `README.md` here. Root rules: structure_conventions.md.

## Layout and imports
- `solver_profiler.py` (CLI, parent; cpu and gpu backends, synthetic problems and whole models) -> `solver_catalog.py` (enumeration per
  backend), `report.py` (machine record, summaries, markdown), `model_cases.py` (model targets); each case runs
  `python -m dev_tools.time_profilers.case_runner` (child: `problems.py` assembly + `runners.py` per SolverType, or
  `model_cases.prepare` / `run_model` for a whole api.run). The parent never factorises and never makes a CUDA context
  (its GPU check is a device count; the device name comes from the children).
- dev_tools imports open_sol, never the reverse; open_sol's import_contracts, structure_index and selftest cache scan
  only `open_sol/`, and `pyproject.toml` ships only `open_sol*`, so nothing here is checked or shipped. open_sol is
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
  'module:Class.method' parts on `profiler_harness.py`; `profiler_*_parts.py` / `_common.py` are shared part lists;
  `run_all_profilers.py` runs every script in a subprocess. Scripts import their siblings by bare name (own folder
  on sys.path) and find the repository from `__file__`, so they run from any folder as scripts, not with `-m`.
  Output: `out/` next to the scripts (git-ignored).
- Parts are wrapped before the registry exists (`wrapped_timers` in `main`): StudyType / FeatureType callbacks keep
  the function objects of registration time, so time what they call instead of the callback itself.
- A target that does not resolve is listed in the report ("parts that did not resolve"), never raised: after a
  refactor check that section, or the part silently drops out of the table.
- Every timed call synchronises the GPU at entry and exit: time per-phase functions, not per-step kernels
  (or compare with `--no-sync`).
