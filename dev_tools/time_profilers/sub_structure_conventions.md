# dev_tools/time_profilers — conventions & gotchas

Usage and what is measured: `README.md` here. Root rules: structure_conventions.md.

## Layout and imports
- `cpu_solvers.py` (CLI, parent) -> `solver_catalog.py` (enumeration), `report.py` (machine record, summaries,
  markdown); each case runs `python -m dev_tools.time_profilers.case_runner` (child: `problems.py` assembly,
  `runners.py` per SolverType). The parent never factorises.
- dev_tools imports open_sol, never the reverse; open_sol's import_contracts, structure_index and selftest cache scan
  only `open_sol/`, and `pyproject.toml` ships only `open_sol*`, so nothing here is checked or shipped. open_sol is
  imported inside functions (problems / runners / catalog), so the parent stays light until it enumerates.
- Run from the repository root: the child is started with cwd = the repository root (`REPO_ROOT`) so `-m` finds both
  packages; the script form inserts the root into sys.path.

## Rules
- Solver choices come from the code, never a hand list: a new SolverType, back end or choice value shows up through
  `solver_catalog.cpu_candidates` (registry + `solver_misfit` on the cpu backend + SolverType validate). A new problem
  class names the physics type / role / study / grid kind it stands for; `runners.RUNNERS` needs an entry for every
  SolverType that can fill a profiled role (a missing one raises KeyError in the child: status 'error').
- Runners call open_sol the way its consumers do (SparseDirectSolver.factorize with symmetric / symmetrizer for host
  direct FDFD factors as fdfd_2d, DiffusionDirectSolver for SPD, solve_resonances with the model runtime's
  mixed_min_unknowns): check those call sites when their signatures change. `_run_diffusion_direct` reads the
  private `DiffusionDirectSolver._factor` for details only.
- The child prints exactly one `RESULT_MARKER` line (libraries may print to stdout too); every exception becomes a
  status there, never a traceback without a result.
- Thread variables are set in the child environment before it starts (pools read them at library load), never by
  the child itself.
