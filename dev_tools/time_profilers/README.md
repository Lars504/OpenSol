# CPU solver profiler

Developer benchmark of every CPU solver choice open_sol offers, on problems assembled by open_sol's own operator code.
Not part of the package (never imported by `open_sol`, not shipped by `pyproject.toml`).

## Usage

Run from the repository root (the child processes import `dev_tools` and `open_sol` from there):

```
python -m dev_tools.time_profilers.cpu_solvers --list                 # problems, solver choices, availability
python -m dev_tools.time_profilers.cpu_solvers --quick                # small sizes, 1 repeat, 120 s per case
python -m dev_tools.time_profilers.cpu_solvers --problems heat_3d fdfd_driven --sizes heat_3d=16,24,32 \
    --solvers 'sparse_direct:*' 'pcg:double/*' --repeat 3 --threads 8 --timeout 900 --out profiles
python dev_tools/time_profilers/cpu_solvers.py --quick                # the same as a script
```

Options: `--problems` (default all), `--sizes` (`64,128` for every selected problem or `problem=a,b` per problem),
`--solvers` (fnmatch patterns of the labels `--list` prints; the SuperLU baseline always runs), `--repeat` (default
3), `--threads` (sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` in every child; default: the
environment as it is), `--timeout` (seconds per case, all repeats), `--quick`, `--out` (default
`./cpu_solver_profiles`; do not commit the results). Output: `cpu_solvers_<date>.json` (machine record, settings,
the solver catalogue, every raw record with per-repeat phases and details) and `cpu_solvers_<date>.md` (also
printed): per problem a time / peak RSS / speed-up table and a phase / accuracy table, rows solvers, columns sizes.

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
(FDFD), `arpack_shift_invert:{superlu, mumps, mumps_mixed, pardiso, pardiso_mixed}` (eigen). GPU back ends (cuDSS,
cuSOLVER QR) are not offered on the cpu backend and so never appear. A missing python-mumps / pydiso (or PARDISO off
x86-64) is a `skipped` record with `host_direct.missing_text`.

## What is measured

- Phases per repeat (seconds; min / median over the repeats in the JSON summary, the tables show medians):
  `analysis` (MUMPS / PARDISO ordering + symbolic analysis, part of `factor`), `factor` (first, cold factorisation or
  pcg set-up incl. multigrid hierarchy and the host direct accuracy probe), `solve` (one solve; pcg: the CG
  iterations), `refactor` (value-only refactorisation of the same pattern, A x 1.001: kept MUMPS / PARDISO analysis,
  kept SuperLU ordering, kept multigrid aggregates), `total` = factor + solve. Eigen: `factor` the shifted factor,
  `solve` the ARPACK iterations, `total` the whole `solve_resonances`.
- Peak RSS: each case runs in its own child process. On Linux the kernel high-water mark (VmHWM) is reset after the
  assembly, so the tables show what the solver phase added (MUMPS / PARDISO / SuperLU allocations included; tracemalloc
  would miss them). Elsewhere `ru_maxrss` of the whole process, marked "(process)".
- Accuracy: relative residual ||b - A x|| / ||b|| in double; eigen: the eigen-pair residual of each resonance and the
  largest |omega - omega_SuperLU| / |omega| against the baseline.
- Status: `ok`, `unconverged` (pcg), `rejected` (line_block above its accept tolerance: the physics would redo the
  solve on SuperLU), `no_resonances`, `refused` (size rules, bandwidth, not SPD), `refused_memory` (the MUMPS /
  PARDISO memory guard), `out_of_memory`, `failed` (singular / accuracy), `timeout`, `killed` (signal; SIGKILL is
  usually the OOM killer), `crashed`, `skipped`, `error`.

## Caveats

- Threads: SuperLU and ARPACK are single-threaded (BLAS aside); MUMPS uses OpenMP + BLAS threads, PARDISO MKL threads.
  Compare at a fixed `--threads`; without it the libraries take every core, so MUMPS / PARDISO speed-ups mix
  parallelism with algorithm. The JSON records the environment of each child, the library's thread count and, with
  threadpoolctl installed, the BLAS / OpenMP pools.
- OpenBLAS beside MKL: numpy / scipy on OpenBLAS and PARDISO on MKL keep two thread pools that slow each other when
  their calls alternate. open_sol holds the OpenBLAS pools at one thread while a PARDISO factor works when
  threadpoolctl is installed (`openblas_pools_held` in the details: 0 = not held, expect PARDISO solves several times
  slower); `--threads` sets `OPENBLAS_NUM_THREADS` to the same count as MKL's, it does not avoid the conflict.
- First-call overhead: library loading and first-touch allocations land in repeat 0 (min vs median shows it); use
  `--repeat 3` or more for timings you quote.
- RSS granularity: pages, not bytes; the peak covers all repeats of a case (factors are released between them); the
  residual check allocates a few vectors. Off Linux the whole-process peak includes imports and the assembly.
- mixed (`*_mixed`): a float32 / complex64 factor refined by float64 GMRES; a rejected mixed factor is refactorised in
  double (`factor_kind` in the details says so). In the eigen problem a `*_mixed` factorization below
  `MIXED_EIGEN_MIN_UNKNOWNS` (20 000) factors in double, as model runtimes do (`factorization_used`).
- pcg runs a cold start to tolerance 1e-8 (single: 1e-6, its float32 floor), so its residuals are not those of a
  direct solve; real time-stepped runs warm-start and are cheaper per solve.
- Problems are synthetic geometries on open_sol's real assembly code, without PEC / mirror pinning, Bloch phases or
  conformal cuts; GPU back ends and the thermal-noise `shifted` role are out of scope.
