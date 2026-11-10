# Solver profiler (`solver_profiler`)

Developer benchmark of every solver choice open_sol offers on the cpu backend and, with `--backends cpu gpu`, on the
gpu backend too, at two levels: synthetic problems (one system assembled by open_sol's own operator code: the solver
call alone) and, with `--models`, whole model files run end to end by `api.run` (the solver inside the run it serves).
Not part of the package (never imported by `open_sol`, not shipped by `pyproject.toml`).

## Usage

Run from the repository root (the child processes import `dev_tools` and `open_sol` from there):

```
python -m dev_tools.time_profilers.solver_profiler --list             # problems, solver choices, availability
python -m dev_tools.time_profilers.solver_profiler --quick            # small sizes, 1 repeat, 120 s per case
python -m dev_tools.time_profilers.solver_profiler --quick --backends cpu gpu   # + cuDSS, gpu pcg / line_block / banded
python -m dev_tools.time_profilers.solver_profiler --problems heat_3d fdfd_driven --sizes heat_3d=16,24,32 \
    --solvers 'sparse_direct:*' 'pcg:double/*' --repeat 3 --threads 8 --timeout 900 --out profiles
python -m dev_tools.time_profilers.solver_profiler --models examples/em/nanobeam_cavity_fdfd.json --backends cpu gpu \
    --roles linear --repeat 1 --timeout 1800                          # whole runs, every linear solver choice
python -m dev_tools.time_profilers.solver_profiler --models MODEL.json --physics heat --scan mesh_n=20,40,80
python dev_tools/time_profilers/solver_profiler.py --quick            # the same as a script
```

Options: `--problems` (default all; none with `--models` unless named), `--sizes` (`64,128` for every selected problem
or `problem=a,b` per problem), `--backends` (default `cpu`; the cpu baseline always runs), `--solvers` (fnmatch
patterns of the labels `--list` prints, backend-free: `'*cudss*'` matches the gpu rows; the baseline always runs),
`--repeat` (warm repeats at least, default 3; a case repeats until its warm repeats sum to 0.5 s, at most 50),
`--no-warmup` (skip the cold repeat that runs first), `--threads` (sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS` in every child; default: the environment as it is), `--timeout` (seconds per case, all
repeats), `--quick`, `--out` (default `./solver_profiles_out`; git-ignored like every `*_out/` folder). Model options: `--models`
(JSON files), `--physics` / `--roles` (default every physics and every role its study uses), `--scan NAME=v1,v2`
(one column per value of a model parameter, e.g. a resolution; default one column, the model as written), `--set
NAME=EXPR` (an override for every run). Output: `solver_profile_<date>.json` (machine record, settings, the solver
catalogue, every raw record with per-repeat phases and details) and `solver_profile_<date>.md` (also printed): per
problem a time / speed-up table, a memory table and a phase / accuracy table, rows solvers (gpu ones `<label> @gpu`),
columns sizes / scan values; model runs keep their run folders under `<out>/model_runs/<date>/` (summary.json, log).

## Synthetic problem or whole model

A synthetic case times the solver call alone (factor, solve, value-only refactor) on one assembled system: no mesh,
materials, assembly, kept orderings across a sweep, coupling or output, and simplified geometry (see Caveats). It
ranks the solvers' kernels and shows how they scale; it does not say how much a whole run speeds up. A model case
answers that: each repeat is a full `api.run` with the choice assigned to one physics role (a new solver node; the
settings of the node the physics already used for that role are kept when it is of the same type), split into build
(every build of the study), study and output seconds, with the physics' `solver_work` counters and the derived
values compared against the baseline run (`dd`; `err` against the model's analytic references). Amdahl applies: a
10x faster factor in a run that spends 20 % of its time factorising saves 18 %.

## Problems

| name | stands for (physics / role / study / grid) | operator |
|---|---|---|
| `fdfd_driven` | em_fdfd_ez / linear / stationary / tensor | complex Helmholtz, CPML; MUMPS / PARDISO factor the LDL^T of D A |
| `fdfd_eigen` | em_fdfd_ez / eigen / eigenfrequency / tensor | `eigenfrequency.solve_resonances`, 3 disc resonances |
| `heat_2d` | heat / linear / stationary / tensor | finite-volume 5-point, real SPD, contrast 100 inclusion |
| `heat_3d` | heat / linear / stationary / tensor_3d | finite-volume 7-point (SuperLU from grid nested dissection) |
| `heat_fem_3d` | heat / linear / stationary / tetrahedra | P2 FEM on Kuhn tetrahedra, Dirichlet dofs eliminated |
| `heat_line` | heat / linear / stationary / line | 1D model (tridiagonal): where `banded` applies |

The solver choices are not listed here: `solver_catalog` enumerates them from the registry (every SolverType filling
the role, every combination of its back_end / factorization / precision / preconditioner choices that
`solvers.roles.solver_misfit` admits on the cpu backend and the SolverType's validate does not refuse). With all
optional packages: `sparse_direct:{scipy_splu, mumps, mumps_mixed, pardiso, pardiso_mixed}`,
`pcg:{double, single, mixed}/jacobi`, `pcg:{double, single}/multigrid`, `banded` (1D), `line_block:{mixed, double}`
(FDFD), `arpack_shift_invert:{superlu, mumps, mumps_mixed, pardiso, pardiso_mixed}` (eigen). With `--backends gpu`
the same enumeration runs on the gpu backend: the cuDSS back ends / factorizations (where the role offers them),
cuSOLVER QR (FDFD), gpu pcg, line_block and banded, and the host libraries again (their gpu-backend transfers
included). A missing python-mumps / pydiso (or PARDISO off x86-64), nvmath-python for cuDSS, or a usable GPU is a
`skipped` record with the reason.

## What is measured

- Phases per repeat (seconds; min / median / max over the warm repeats in the JSON summary, the tables show the
  median and +- half the min-max range): `analysis` (MUMPS / PARDISO ordering + symbolic analysis, part of
  `factor`), `factor` (first factorisation of the repeat or pcg set-up incl. multigrid hierarchy and the host direct
  accuracy probe), `solve` (one solve; pcg: the CG iterations), `refactor` (value-only refactorisation of the same
  pattern, A x 1.001, the scaled copy made before the timer: kept MUMPS / PARDISO analysis, kept SuperLU ordering,
  kept multigrid aggregates), `total` = factor + solve, `sweep` = refactor + solve (the cost of one more point of a
  sweep on the same pattern; its own speed-up column). Eigen: `factor` the shifted factor, `solve` the ARPACK
  iterations, `total` the whole `solve_resonances`.
- Cold repeat: the first repeat (library loading, first touch) is reported apart (`cold`) and left out of the
  medians. `cpu/wall`: process CPU time over wall time of the warm repeats, roughly the threads that worked.
- Peak RSS: each case runs in its own child process. On Linux the kernel high-water mark (VmHWM) is reset after the
  assembly (exact). Elsewhere (Windows: the peak working set) the peak cannot be reset: exact when the solver phase
  raised the process peak, otherwise "<= +X" (the assembly set the peak; the solver added at most X). RSS includes
  MUMPS / PARDISO / SuperLU allocations (tracemalloc would miss them). The memory table also shows the factor size the
  library reports (`factor_bytes`).
- Accuracy: relative residual ||b - A x|| / ||b|| against the float64 operator (single-precision pcg also against
  its own float32 operator, `own`: a gap between the two is operator rounding, not CG); eigen: the eigen-pair
  residual of each resonance and the largest |omega - omega_SuperLU| / |omega| against the baseline. Tolerances: pcg
  its stop tolerance, line_block its accept tolerance, direct factors and eigen pairs the probe bound
  `cudss_factor.ACCEPT_RESIDUAL`; above 10x the tolerance the case is `inaccurate`.
- Status: `ok`, `inaccurate` (residual above 10x its tolerance), `unconverged` (pcg), `rejected` (line_block above
  its accept tolerance: the physics would redo the solve on SuperLU), `no_resonances`, `refused` (size rules,
  bandwidth, not SPD), `refused_memory` (the MUMPS / PARDISO memory guard), `out_of_memory`, `failed` (singular /
  accuracy), `timeout`, `killed` (signal; SIGKILL is usually the OOM killer), `crashed` (these three keep the repeats
  the child finished), `skipped`, `error`.

## Caveats

- Threads: SuperLU and ARPACK are single-threaded (BLAS aside); MUMPS uses OpenMP + BLAS threads, PARDISO MKL threads.
  Compare at a fixed `--threads`; without it the libraries take every core, so MUMPS / PARDISO speed-ups mix
  parallelism with algorithm. The JSON records the environment of each child, the library's thread count and, with
  threadpoolctl installed, the BLAS / OpenMP pools.
- OpenBLAS beside MKL: numpy / scipy on OpenBLAS and PARDISO on MKL keep two thread pools that slow each other when
  their calls alternate. open_sol holds the OpenBLAS pools at one thread while a PARDISO factor works when
  threadpoolctl is installed (`openblas_pools_held` in the details: 0 = not held, expect PARDISO solves several times
  slower); `--threads` sets `OPENBLAS_NUM_THREADS` to the same count as MKL's, it does not avoid the conflict.
- Noise: a spread of tens of % is common for cases below ~50 ms and on hybrid (performance + efficiency core) CPUs;
  read speed-ups within the spread as ties. The machine line lists the core classes on Windows.
- RSS granularity: pages, not bytes; the peak covers all repeats of a case (factors are released between them); the
  residual check allocates a few vectors.
- mixed (`*_mixed`): a float32 / complex64 factor refined by float64 GMRES; a rejected mixed factor is refactorised in
  double (`factor_kind` in the details says so). In the eigen problem a `*_mixed` factorization below
  `MIXED_EIGEN_MIN_UNKNOWNS` (20 000) factors in double, as model runtimes do (`factorization_used`).
- pcg runs a cold start to tolerance 1e-8 (single: 1e-6, its float32 floor), so its residuals are not those of a
  direct solve; real time-stepped runs warm-start and are cheaper per solve.
- Synthetic problems are simple geometries on open_sol's real assembly code, without PEC / mirror pinning, Bloch
  phases or conformal cuts, and the thermal-noise `shifted` role has none: use `--models` for those.
- gpu rows: the operator and rhs are moved to the device once per case before the timers (a gpu model assembles them
  there); line_block and the eigen solve take host operators on both backends, as fdfd_2d hands them. Peak RSS is
  host memory only: the cuDSS factor size is the `device factor` of the memory table. The cold repeat carries the
  kernel compiles and cuDSS library load; a consumer GPU runs float64 at 1/32-1/64 rate, so compare `cudss` with
  `cudss_mixed`.
- Model cases: a warm repeat is a second `api.run` in the same process (libraries loaded, kernels compiled); a model
  whose run takes minutes wants `--repeat 1` and a larger `--timeout`. Peak RSS covers the whole run (mesh, build,
  study, output). A choice that changes what the run computes (a loose pcg tolerance, single precision) shows in `dd`.

# Case profilers (`case_profilers/`)

Whole selftest runs per solver engine (gpu by default), to find host work (numpy / scipy / Python, device-host
round trips) around the solves. Run from any folder:

```
python dev_tools/time_profilers/case_profilers/profiler_solver_pressure_acoustics.py [--repeat N] [--warmup]
python dev_tools/time_profilers/case_profilers/run_all_profilers.py [--only SUBSTRING ...]
```

Common options (`profiler_harness.main`): `--case`, `--runs`, `--variant`, `--backend gpu|cpu`, `--repeat`,
`--warmup`, `--no-cprofile`, `--no-sync`, `--no-arrays`, `--set NAME=EXPR`; some scripts add `--kind` / `--refine`.
Output `out/<name>/report.txt` + `profile.prof`: per part calls, inclusive / exclusive wall and cpu/wall (~1
host-bound, ~0 waiting on the GPU, > 1 threaded BLAS), the untimed rest, cProfile own / cumulative time of open_sol
functions, callers of the largest outside functions and of every sync / transfer site. `run_all_profilers.py` logs
to `out/run_all.log` and keeps each report in `out/history/<timestamp>/`. `measure_cudss_reordering.py` times cuDSS
reordering choices on synthetic matrices; `profiler_example_sin_square_membrane.py` profiles an example file.
