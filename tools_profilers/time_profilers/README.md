# Solver profiler (`solver_profiler`)

Developer benchmark of every solver choice open_sol offers on the cpu backend and, with `--backends cpu gpu`, on the
gpu backend too, at two levels: synthetic problems (one system assembled by open_sol's own operator code: the solver
call alone) and, with `--models`, whole model files run end to end by `api.run` (the solver inside the run it serves).
Not part of the package (never imported by `open_sol`, not shipped by `pyproject.toml`).

## Usage

Run from the repository root (the child processes import `tools_profilers` and `open_sol` from there):

```
python -m tools_profilers.time_profilers.solver_profiler --list             # problems, solver choices, availability
python -m tools_profilers.time_profilers.solver_profiler --quick            # small sizes, 1 repeat, 120 s per case
python -m tools_profilers.time_profilers.solver_profiler --quick --backends cpu gpu   # + cuDSS, gpu pcg / line_block / banded
python -m tools_profilers.time_profilers.solver_profiler --problems heat_3d fdfd_driven --sizes heat_3d=16,24,32 \
    --solvers 'sparse_direct:*' 'pcg:double/*' --repeat 3 --threads 8 --timeout 900 --out profiles
python -m tools_profilers.time_profilers.solver_profiler --models examples/em/nanobeam_cavity_fdfd.json --backends cpu gpu \
    --roles linear --repeat 1 --timeout 1800                          # whole runs, every linear solver choice
python -m tools_profilers.time_profilers.solver_profiler --models MODEL.json --physics heat --scan mesh_n=20,40,80
python -m tools_profilers.time_profilers.solver_profiler --problems solid_modes_3d --backends cpu gpu \
    --sizes solid_modes_3d=16,32,64 --repeat 1 --timeout 3600          # block_eigen vs arpack's matrix-free Lanczos
python tools_profilers/time_profilers/solver_profiler.py --quick            # the same as a script
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
| `solid_modes_3d` | solid_mechanics_3d / eigen / eigenfrequency / tensor_3d | `ElasticSystem3D.lowest_modes`, 6 lowest modes of a steel roller box, matrix-free (no operator) |

`solid_modes_3d` is the point where the factor-free `block_eigen` is weighed against what it replaces: arpack's
`eigsh_inverse` (shift-invert Lanczos at 0, one inner PCG solve per step) on brick meshes. The box has (n + 1) x n x
(n - 1) cubic cells (n even: pairwise coprime counts, no exact degeneracy), rollers on all six faces, so the exact
discrete frequencies have a closed form (the 3D roller box self-test's, `ex` in the tables). Sizes 16 / 32 / 48 / 64
(64: ~0.82 M free dofs). Its system (and the PCG workspace, multigrid hierarchy when Automatic has one) is assembled
on the run backend before the memory baseline and kept for every repeat (`transfer` seconds); its 'linear' solver is
the role's Automatic in the eigenfrequency study on that backend (gpu: pcg mixed, cpu: pcg double, both block Jacobi:
what arpack's inner solves use in a model without a solver node; the linear role is not varied here, so arpack with a
multigrid inner PCG needs `--models`).

The solver choices are not listed here: `solver_catalog` enumerates them from the registry (every SolverType filling
the role, every combination of its back_end / factorization / precision / preconditioner choices that
`solvers.roles.solver_misfit` admits on the cpu backend and the SolverType's validate does not refuse; a setting the
problem class lists in `ignored_settings` is pinned to its resolved default instead: one row). With all
optional packages: `sparse_direct:{scipy_splu, mumps, mumps_mixed, pardiso, pardiso_mixed}`,
`pcg:{double, single, mixed}/jacobi`, `pcg:{double, single}/multigrid`, `banded` (1D), `line_block:{mixed, double}`
(FDFD), `arpack_shift_invert:{superlu, mumps, mumps_mixed, pardiso, pardiso_mixed}` (eigen; `solid_modes_3d`: only
`arpack_shift_invert:superlu`, its factorization being ignored on brick meshes), `block_eigen:{mixed, double}/{multigrid,
jacobi}` (opt-in eigen of solid_mechanics_3d: `solid_modes_3d` only). With `--backends gpu`
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
  iterations, `total` the whole `solve_resonances`. Vibration modes (`solid_modes_3d`): `total` the whole
  `lowest_modes` call, `solve` its eigen iterations (arpack: Lanczos with the inner PCG solves), `factor` block_eigen's
  set-up around them (float32 operator copies, the private multigrid hierarchy of preconditioner multigrid when the
  linear solver has none: built in every call, as in every model eigen solve); arpack has no factor there.
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
  `cudss_factor.ACCEPT_RESIDUAL`; above 10x the tolerance the case is `inaccurate`. Vibration modes: the residual is
  max ||K u - omega^2 M u|| / (omega^2 ||M u||) of the returned modes (float64 brick products: block_eigen's own
  measure) against at least the solver's tolerance (arpack: the Lanczos tolerance it used, 1e3 x the CG tolerance with
  the default 0), `dw` against the cpu arpack baseline and `ex` = the largest relative frequency error against the
  exact discrete modes (rank order); details also hold iterations (arpack: inner CG; block_eigen: block iterations,
  float32 / float64 split), operator / mass / preconditioner applications, the block size and the method text.
- GPU memory of the vibration modes: the cupy pool's high water during the `lowest_modes` call over its live bytes at
  the start (the pool's cached blocks are freed before the timer), the memory table's `gpu peak`.
- Status: `ok`, `inaccurate` (residual above 10x its tolerance), `unconverged` (pcg; vibration modes: fewer modes
  than asked, ARPACK short of its eigenvalues, an inner PCG solve or block_eigen stopped short), `rejected` (line_block above
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
- `solid_modes_3d` on the cpu backend runs the numpy twins of the brick kernels: the cpu arpack baseline (thousands
  of block-Jacobi CG iterations per inner solve, tens of solves) takes minutes from n = 32 and can exceed the default
  timeout at 48 / 64; the baseline always runs, so pass `--sizes solid_modes_3d=...`, `--repeat 1` and a larger
  `--timeout` there (a timed-out case keeps its finished repeats). The Lame values are float64 constants (a model
  derives them per cell from float32 E / nu), so `ex` measures the solver, not the material rounding.
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
round trips) around the solves. They cover the engines and mesh kinds that no analytic example covers (pressure
acoustics, em_eme_3d, magnetostatics_3d, 3D tensor / tetrahedra / nonlinear solid mechanics, the triangle and
tetrahedra FEM engines, the membrane plate), plus the example-based em_fdfd_3d and sin_square scripts; the per-module
performance set is `example_profilers/` below. Run from any folder:

```
python tools_profilers/time_profilers/case_profilers/profiler_solver_pressure_acoustics.py [--repeat N] [--warmup]
python tools_profilers/time_profilers/case_profilers/run_all_profilers.py [--only SUBSTRING ...] [--cudss --cudss-mixed --scipy --pardiso --mumps | --all] [--backend gpu cpu]
    [--timeout S] [--new-first] [--plot-only]
```

Common options (`profiler_harness.main`): `--case`, `--runs`, `--variant`, `--backend gpu|cpu`, `--repeat`,
`--warmup`, `--no-cprofile`, `--no-sync`, `--no-arrays`, `--set NAME=EXPR`, `--compare NAME=v1,v2`,
`--automatic-solvers` (every role on Automatic), `--precision single|double|mixed` (solver_defaults); some scripts
add `--kind` / `--refine`. Output `out/<name>__<backend>[__<compare>[__only]][__<set>]/report.txt` + `profile.prof`
(the settings are in the folder name, e.g. `membrane_film__gpu__back_end-pardiso__only`, so a pardiso profile never
overwrites a cudss one): per part calls, inclusive / exclusive wall, cpu/wall (> 1 threaded BLAS; a blocking device sync spins a core, so a part waiting on the GPU shows ~1 too) and
wait s (the part's own exit synchronize = device work it left queued: ~0 host-bound, ~incl device-bound; json
`device_wait_seconds`), the untimed rest, cProfile own / cumulative
time of open_sol functions, callers of the largest outside functions and of every sync / transfer site.

- Every profiler times the solver-role calls and the cuDSS phases inside them (`_equilibrated`, `_plan` /
  `_plan_solver` / `_device_plan`, `_upload_values`, `_factorize_and_probe`, `_gmres_refine`, `relative_asymmetry`,
  `_lower_triangle`, `DevicePencil.__init__` / `combined`). The `library:` row is cuDSS's own numeric factorisation
  (`factorize_seconds`), split out of `_factorize_and_probe`. The difference is our probe and refinement.
- Host direct (PARDISO / MUMPS, double and mixed): `HostDirectFactorization` refactorize / `_equilibrated` /
  `_factorize` / `_probe_factor` / solves, and the library calls as `_PardisoFactor` / `_MumpsFactor` rows (`__init__`
  = analysis, `factor` / `refactor` = numeric factorisation, `solve`). Multigrid PCG: `setup` with
  `aggregate_strong_links` (host), `_build_level`, `_coarsest_solver`, and `precondition` (one V-cycle per iteration).
  The cases run these only when a role asks for them: `--compare back_end=cudss,pardiso_mixed` or
  `--compare preconditioner=jacobi,multigrid`.
- `runs.md` / `runs.json`: one column per run (each repeat its own), cells `inclusive ms per call x calls`; the
  report repeats the table when there is more than one run. `runs.json` also holds per run `gpu_peak_bytes` (the
  larger of the build peak and the pool high-water + cuDSS factors outside the pool; null on the cpu backend, where the
  build peak is the traced host numpy instead; the plot leaves `@cpu` lines out of the GPU panel) and `host_peak_bytes` (the
  process's peak working set so far, so it only grows over the runs of one script), `status`, and `solvers`: per
  role the back end it used (the runtime one of sparse_direct, an `auto` factorization resolved as the model does) and
  its preconditioner.
- `--compare back_end=cudss,cudss_mixed` runs every spec once per value. Each solver role the study uses whose type
  has that setting gets it, Automatic roles included (a new node of their automatic type). `back_end` also sets the
  shift-invert `factorization`, and the other way round. A role skips a value its `setting_choices` refuse (a
  double-only eigen role takes `cudss`, not `cudss_mixed`). A run no value applies to runs once as written
  (`--compare-only`: dropped, and a script with no run left profiles nothing); both cases are printed. With
  `--repeat 2 --warmup` this is the double / mixed phase comparison that found the cuDSS speed-ups.

`run_all_profilers.py` logs to `out/run_all.log` and keeps each report and its `runs.json` in
`out/history/<timestamp>/`, together with `profile_points_before.json` (the points file as it was when that run
started). A log line gives the command wall and, in brackets, the wall of the same command and solver the last time
it was recorded (`pass, 15.2 s (14.9 s)`; `(timed out)` when that one stopped at the timeout; nothing on the first
run). After a change, `python profile_delta.py <report folder name>` (e.g. `solid_mechanics_3d_nonlinear__gpu`)
lists per run the parts whose exclusive time moved by more than `--min-ms` (20) and `--min-percent` (30) between the
two newest kept profiles of that folder (`--before` / `--after`: any two `runs.json`).

**Solver groups and backends.** Every solver is in one of five groups (harness `solver_group`): a library back end by
its package, every other solver (open_sol's own pcg with jacobi / multigrid, explicit kernels, line_block, ...) by
array library and precision. One flag per group, `--all` runs the five in turn:
- `--cudss`: cuDSS + cupy double, gpu backend: harness `--precision double --compare back_end=cudss`.
- `--cudss-mixed`: cuDSS mixed + cupy single, gpu backend: `--precision single --compare back_end=cudss_mixed`.
- `--scipy`: host SuperLU + numpy, cpu backend: `--compare back_end=scipy_splu`.
- `--pardiso`, `--mumps`: their host library, cpu backend: `--compare back_end=<value> --compare-only`.

The backends above are each flag's default; `--backend gpu` (or `gpu cpu`) runs the host groups on those instead (a
gpu model factorising on the host: `pardiso @gpu`, its own line). cudss / cudss_mixed run on the gpu only. No flag
runs auto (default `gpu`): the cases' solvers as written, Automatic in
`run_all_examples.py` (which adds `--automatic-solvers` to every variant, so a group flag changes only what the group
sets). Its log line names the groups its runs used (`fluid [cudss_mixed+scipy @gpu]`). `--precision` sets
`solver_defaults.precision`: a solver without that precision, or whose role refuses it, keeps its own, and its group
says so. Older `scipy_splu @cpu` / `numpy @cpu` points load as `scipy @cpu`.

A command (example group) holds several runs (tasks). The cudss, cudss_mixed and scipy groups profile every run.
pardiso and mumps profile only the runs that have a role taking the value; the harness finds that from the specs
before any build, and a command with none logs `skipped: no run takes this solver` (its earlier points of that
command are removed). A point counts the runs that completed (`status` in `runs.json`) and in which the compared
solver factorised (a matrix-free eigen solve under `--pardiso` is a PCG profile, not a PARDISO one); the others are
named in the log. Two runs that differ only by the node the value replaces run once.

The points keep each counted run's wall and peaks (`run_walls`, `run_gpu_peaks`, `run_host_peaks`). When the solvers
of a command measured different runs, the plot splits it into one column per set of runs the same solvers measured
(`command: runA+runB`), so every column compares solvers on the same tasks; a solver that did not do a column's runs
has no point there.

The plot draws the auto pick and the five groups only (older variants and commands stay in `profile_points.json`,
undrawn).

**Timeout.** Each command stops after `--timeout` seconds (default 60; `0`: no limit), counted from its process
start. The harness gets it as `--time-limit`: it interrupts the run going on (status `timed_out` in `runs.json`), skips
the rest and writes a partial report: the first lines say where it stopped and which runs did not run, and the parts
are those timed until then (a part running at the interrupt counts up to it). The report is kept in the history
folder and the log line says `partial`. The interrupt waits for a running library call (a PARDISO factorisation)
to return: a command still running 30 s (`REPORT_GRACE`) after the timeout is killed and has no report. Ctrl+C on a
single profiler script also writes a partial report. A stopped command that was profiling becomes a point at the `>60`
tick of the run wall panel, above a thin line at the timeout, and has no memory point. A command stopped before it
found a run for the solver gives none.

**The plot.** Each command that finishes is a point in `out/profile_points.json`, plotted in
`out/profile_points.png`. There is one panel each for run wall (the counted runs only, log scale), GPU peak and RAM
peak, with the commands on the x axis. Each solver @backend has its own line: a colour per solver, a marker per
backend. Line style marks the profile:
- `first` (dotted): from the first complete profile of that solver @backend, a run without `--only`. A command
  added later gets its first point at the next complete run; `--new-first` makes this complete run the first one.
- `previous` (dashed) and `current` (solid): the command's last two points for that solver @backend. A run with
  `--only` or other solver flags moves only those points; a failed command keeps its old ones.

The plot shows the auto pick and the five groups in the file, not only the ones of this run. One black ring per
column marks the current point of Automatic's gpu pick, whichever solvers were run. The harness works out the groups
Automatic picks for every run from the case's roles (`runs.json` `automatic_groups`: the role's Automatic type and
settings, back_end 'auto' resolved to cuDSS where the role lists it, else SuperLU; the point keeps them per run,
`run_automatic`). When the column's runs need one group, the ring sits on that group's line; when their roles fall in
several groups (a gpu model with a SuperLU role, pcg next to cuDSS) or that line was not run, it sits on the
`auto @gpu` point, which ran that mix. A ring needs a line that measured the column's runs: an older point without
per-run walls cannot carry one in a split column. A runtime fallback (cuDSS refusing an operator, then SuperLU) is not seen;
`solvers` holds what the runs actually used.

**Saving.** Each point is written as soon as its command finishes, through a temporary file and a replace, and a
complete run sets each command's `first` point at the same moment. A failure never drops points: an unreadable file stops
the run (restore it from a `profile_points_before.json`), and a file in the first one-solver layout is read as
`auto @gpu`.

`--plot-only` redraws the plot without running anything. `measure_cudss_reordering.py` times cuDSS
reordering choices on synthetic matrices; `profiler_example_sin_square_membrane.py` profiles an example file.
`../cudss_phase_profile.py` (gpu, script form) times cudss_factor's measure-first switches against the nvmath path on
real CudssFactorization objects: REFACTORIZATION_PHASE refreshes, RAW_SOLVE solves, REORDERING_ALGORITHM 'AMD' on 2D
operators, a parked-plan adoption (KEPT_PLANS) against a kept-ordering plan, and the library-load stamps; it writes
`time_profilers/out/cudss_phase_profile/results.json`. Turn a switch on in cudss_factor only when it wins there.

# Example profilers (`example_profilers/`)

The performance set: analytic examples at their own sizes, grouped per physics module (one or two examples of a
few seconds each, several short ones where each is under ~2 s), so the time is the physics, the geometry and the
solve. `python example_profiler.py --list` prints the groups with each example's selftest wall (about 110 s of runs in
all). Run from any folder:

```
python tools_profilers/time_profilers/example_profilers/example_profiler.py GROUP [--runs STEM ...] [--warmup] [--compare back_end=cudss,cudss_mixed]
python tools_profilers/time_profilers/example_profilers/run_all_examples.py [--only SUBSTRING ...] [--cudss --cudss-mixed --scipy --pardiso --mumps | --all] [--backend gpu cpu]
python tools_profilers/time_profilers/case_profilers/profile_delta.py em_fdtd__gpu --history tools_profilers/time_profilers/example_profilers/out/history
```

The harness options, reports, `runs.json`, solver flags, timeout, history and plot are those of the case profilers
above, written to `example_profilers/out/`. Each group times the study / runner / output parts, the solver roles, the
study hooks of its engine classes and the shared part lists of its engines.
