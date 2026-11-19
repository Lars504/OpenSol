# open_sol/solvers — conventions & gotchas

Design: `solver_plan.md` Part I. Root rules: structure_conventions.md "Solver roles".

## Layout and imports
- `roles.py` (SolverRole, SolverType, SolverRequest, ResolvedSolver, the one fit rule `solver_misfit`) is pure Python
  and `__init__` imports only it (contract solver_roles_pure): model.schema imports and re-exports it.
- Runtimes import numpy and sibling solvers modules only at module level (contracts solver_runtimes*, cupy / scipy /
  nvmath inside functions; kernels built on first use), so core.fv_diffusion, model.definition_helpers and GUI-thread
  checks (`scipy_available`, `size_issue`, `expected_back_end`) may import them. `cudss_factor.py` is the one cuDSS
  wrapper.
- Shared helpers, never a local copy (grep these modules first): cudss_factor (CSR canonicalisation `canonical_csr` /
  `csr_rows`, the GMRES-IR loop `gmres_refine` on any array module, the probed-factor base `ProbedFactor` of cuDSS and
  host_direct), sparse_direct (`ExactSolveRecord`, `require_backend_operator`, `recycled_gmres`, `cudss_precision`),
  pcg (`DeviceCsrProduct`, the capturable gpu CSR product; the CPU CG numbers of definition_helpers' warning).
- Naming: `PhysicsSolver` (core.solver_base) is the physics runtime; objects here are named by role (PCGSolver is a
  LinearSolver).

## Model path (spec -> resolution -> build -> summary)
- Spec: `ModelSpec.solvers` [SolverSpec(name, type, settings)], `ModelSpec.solver_defaults`, `PhysicsSpec.solvers`
  {role: node name}; a missing role = Automatic. `schema.resolved_solvers` is the one resolution (validate through
  `ValidationContext.solvers`, estimators through `MemoryContext.solvers`, builder, api.check, time estimate,
  solve_flow); other code needing a role's type or values composes the same pieces (`SolverRole.automatic`,
  `schema.with_node_precision` / `with_default_precision`, `auto_back_end_values`), never a node's raw settings.
- Automatic = `SolverRole.automatic(backend, grid, study)` (all required; study None = without study_defaults). The
  grid is `schema.solver_grid(spec, physics_spec)`, also required by `solver_misfit`, so no caller judges a 1D line or
  an FEM mesh as 'tensor'. 'line' (1D) and 'tensor_3d' are tensor meshes a declared 'tensor' fits (test
  `roles.TENSOR_GRID_KINDS`, never == 'tensor'); 'tensor_3d' exists only so grid_defaults can differ from 2D.
  grid_defaults win over backend_defaults, so a backend default (the solids' gpu pcg 'mixed') holds on tensor meshes
  only. Role data and the measurements behind each choice sit at the roles (shared ones in model.definition_helpers,
  solid_mechanics' own in physics.mechanics.definition), not here.
- Precision (S9): `solver_defaults.precision` also replaces a precision in a role's Automatic settings (e.g.
  solid_mechanics_3d gpu 'mixed'), and a node of the Automatic type that sets no precision takes the Automatic one
  (`with_node_precision`: a node added only for a tolerance keeps mixed). GUI-added nodes get the default written in at
  creation (a later solver_defaults change does not reach them); JSON nodes without one follow it.
- Study defaults (`SolverRole.study_defaults`, e.g. a kept sparse_direct factor in 2D diffusion / solid transients):
  `meshing.fitted_memory_estimate` resolves them without the study when the factor does not fit (`plain_roles`). The
  builder (`_fitted_estimate`), api.check and the GUI document fit once and hand the same plain_roles to
  `resolved_solvers`, `estimate_memory`, `validate` and `estimate_time`: all but validate default to frozenset() and
  would silently resolve the study default. A self-test counting CG iterations in a 2D transient pins a pcg node.
- Runtimes take `backend=` from `SolverRequest.backend` and work on that backend's arrays only
  (`core.backend.array_module` / `sparse_module`, `sparse_direct.require_backend_operator`): an array of the other
  backend is a TypeError, never a silent transfer.
- A back end not every physics can hold goes into `SolverType.opt_in_settings` (offered where a role's
  `setting_choices` list it), a role of a shared type into `opt_in_roles` (offered where the physics' role lists the
  type in `opt_in_types`), so neither reaches other physics by default.
- Size rules: `sparse_direct.diffusion_size_issue` / `vector_3d_size_issue` at validation (the diffusion and 2D solid
  linear roles through `definition_helpers.cg_solver_issues`, counting `SolverRole.direct_components` per cell / node);
  DiffusionDirectSolver refuses above the limit and the fit guard then resolves a study default without the study.
- Estimators and physics validate callbacks read the resolution through the definition_helpers readers
  (`linear_solver_precision`, `resolved_back_end`, `resolved_stepper_precision`, ...; `vctx.solvers` is None when it
  does not resolve), a create reads its runtimes (`explicit_kernels.stepper_precision(ctx.solvers(name))`).
- Feature precisions (S3.3): `solver_misfit` cannot see features, so a feature refusing a stepper precision declares
  `FeatureType.precisions` (validate errors at the feature; describe and GUI show it through
  `schema.restricted_feature_precisions`, no edit there); a refusal depending on physics settings stays in the physics
  validate.
- Self-tests: nodes through `model.selftest_helpers` (`with_exact_cg`, `with_direct_node`, `skip_without_host_direct`);
  `SelfTestCase.variants` get nodes `<variant>_<physics>_<role>` (api `_variant_spec` raises on a clash). A change of a
  role's precisions or Automatic type updates its row in the `solver_default_precision` tables.
- Counters: a build-injected runtime reports in its role record (summary.json 'solvers', `study._solver_records`); a
  runtime a script constructor builds from a choice string or None is private and reports in the physics'
  `work_counters`.
- Run-time recipes: every SolverType sets `time_estimate(values, SolverWork)` (explicit_kernels none: the physics counts
  its steps); its items' `counter` / `seconds_counter` name its own work_counters, timers only at existing syncs; a new
  counter or timer joins work_counters and the recipe in the same edit (iteration counts: pcg.py header).

## pcg (`PCGSolver`)
- `factorize` options after (operator, preconditioner, n_unknowns) exist only on PCGSolver (sparse_direct, banded and
  DiffusionDirectSolver factorize do not take them): a caller passes them only to a PCGSolver.
- Fused iteration (`FUSED_ITERATION_PROTOCOL`; ElasticSystem3D: kernels3d.FusedBrickIteration): the caller's object
  runs whole CG iterations in its own launches and keeps its own partial sums; its estimator books the second direction
  (`workspace_bytes(fused=True)`) and those partial sums.
- CUDA graphs (`capture_blocks`, gpu): a captured check block bakes every pointer and scalar the callbacks read, so a
  caller replacing those arrays calls `release_block_graphs()` (ElasticSystem3D.assemble, its multigrid guard);
  factorize drops them. Callbacks and fused objects launch plain kernels inside a block (no sync, no allocation, no
  graph of their own); the own multigrid hierarchy launches into the capture through its `inside_capture` flag.
- Memory: `workspace_bytes` / `mixed_extra_bytes` are what the physics' estimators count (the pcg SolverType estimate
  is empty, the solution vector is physics-owned, a matrix-free mixed caller counts its own float32 operator):
  `core.memory_estimate.diffusion_system_estimate` and the mechanics estimators; a workspace change updates them.
  `fem.diffusion.fem_system_estimate` books its CG vectors by hand (no mixed copies).
- Residual reads (`check_interval`) follow the BOUND preconditioner: a callback bound under the setting 'multigrid'
  reads every 2 (gpu) / 1 (cpu) iterations, everything else every 16 / 4 (a Jacobi bind of 'multigrid' too);
  cg_iterations and the time estimate round up to that block. magnetic `kept_preconditioner._bind_callback` relies on
  it (binds its callback under 'jacobi', so the own hierarchy stays out, then restores 'multigrid').
- Stop reference per solve (`STOP_REFERENCES`): 'rhs' for steady solves, 'initial_residual' for implicit steps
  (DiffusionSystem: core sub file); `last_residual` is relative to it.
- 'mixed' (`_iterate_mixed`: one float32 CG with residual replacement against the float64 residual): `last_stalled`
  means it stopped at the system's round-off floor, not at the iteration limit, so a caller with a fallback
  preconditioner must not take it for a weak preconditioner (ElasticSystem3D._guarded_solve).
- Users: the 'linear' role of the DiffusionSystem types (`DiffusionSystem(linear_solver=)`) and solid_mechanics(_3d)
  (`ElasticSystemBase(linear_solver=)`: block Jacobi, 'rhs'; `definition_common.solid_linear_solver` doubles the auto
  iteration limit). Both take the injected solver's precision and set its label and compute_residual.

## multigrid (`MultigridPreconditioner`, pcg setting preconditioner 'multigrid')
- Design (smoothed aggregation, l1-Jacobi V(1,1), filtered prolongator smoothing, float32 hierarchy, power-of-two
  residual scaling, dense coarsest inverse): multigrid.py constants and `_build_level` / `_filtered_rows` /
  `_coarsest_solver`. Which roles run it, at which precisions, and the jacobi default of a pcg node: the roles
  (`CG_DIFFUSION_LINEAR_ROLE`, `SolverRole.mixed_preconditioners`) and the preconditioner help text.
- Never multiply the full fine operator on the backend: Galerkin products run in row blocks (`_row_blocks`) and on
  the gpu in column chunks of P (`_column_ranges`), both against cuSPARSE workspace limits. A cupyx / scipy CSR row
  slice copies even over the full range: block code takes `_rows`, never `matrix[start:stop]`.
- PCGSolver builds its own hierarchy only from an assembled scalar CSR at 'single' / 'double' (else it raises); a
  matrix-free operator brings its own V-cycle callback (ElasticSystem3D: `physics.mechanics.multigrid3d`, any
  precision), reporting through `PCGSolver.preconditioner_counters` and `SolverWork.setup_ops`.
- Jacobi binds and lagged setups (pcg module header; the own CSR hierarchy only, magnetic `KeptPreconditionerSolver`
  keeps its own outside them): the summary still names 'multigrid' (counters multigrid_jacobi_binds / _fallbacks /
  _lagged_*); the time recipe prices only the small-model bind as Jacobi (`bound_preconditioner`), the memory estimate
  always books the hierarchy; a mid-solve setup counts as factorize seconds.
- Memory constants (`hierarchy_bytes`, `setup_extra_bytes`, `aggregation_host_bytes`; booked by
  `diffusion_system_estimate(multigrid=)` and `fem_system_estimate`) follow the hierarchy layout: a layout change
  updates them; the work counters of `_record_size` measure them. Setup seconds are machine-dependent: never pin them
  in a selftest.
- Measured: 3D iterations to 1e-8 (20-35 at 40^3-60^3) exceed the ~12 of MULTIGRID_RESIDUAL_REDUCTION (3D run-time
  estimates run low: level-1 aggregates hold ~3 rows, most level-1 P rows unsmoothed). Rejected: threshold 0 below the
  finest level, |a_ij| strength, smoother weight, V(2,2), float64 hierarchy, a larger coarsest level.

## line_block (`LineBlockSolver`, Part II; SolverType 'line_block', roles 'shifted' and opt-in 'linear')
- One runtime serves several consumers: each passes `owner=` to set_operator / factorize_operators and re-binds only
  when `solver.owner` is another; any set_operator / factorize_operators / release invalidates the kept LineBlockFactors
  (`valid` False), so a holder tests `valid` before solving.
- Operators must be block-tridiagonal in the line order (`_checked_matrix` raises): a periodic / Bloch wrap goes
  inside the blocks (block-axis rule: physics/em/fdfd_line_block).
- Memory: consumers estimate `operator_bytes` (run item) + ONE `shift_bytes`: chunks grow only into free memory at run
  time, so min(n_shifts, MAX_CHUNK_SHIFTS) x shift_bytes over-books and refuses runs that fit. Kept factors
  (`factorize`, `factorize_operators`) hold three block stacks per shift: every estimate and sizing of one passes
  `kept=True` (batches: `batch_size`).
- Rejected shifts (above ACCEPT_TOLERANCE or non-finite) are the consumer's to re-solve with its reference solver:
  solve_shifted flags them, `LineBlockFactorization.solve_host` raises `RejectedLineBlockSolve`, which every
  shift-invert consumer (em_fdfd_ez / _hz eigenfrequency, em_modes, em_fdfd_bor) catches to redo that eigen-solve on
  host SuperLU.
- Role 'linear' (em_fdfd_ez / _hz only; em_fdfd_bor and em_modes reach line_block through 'shifted'): one kept factor
  per assembled system, or a wavelength batch of whole operators (`factorize_operators`: Helmholtz is not affine in
  omega) of at most `fdfd_line_block.MAX_BATCH_WAVELENGTHS`, kept while the material serial holds. Adjoint solves use
  the factor through the symmetrizer; an unsymmetric D A goes to the reference sparse_direct factor.
- cpu: measured 7-9x slower per solve than host SuperLU; the cpu path is for parity, not speed.
- Time recipe: launch-bound on small blocks (Python loops over blocks, ~80 us gemm floor per call on an RTX 3050), so
  it prices each block call (the `*_PER_BLOCK` constants, counted from `_factor` / `_sweep` / `_kept_sweep`: a loop
  change updates them); a solve_shifted caller passes `SolverWork.batch_shifts` (mean shifts per chunk; 1 for kept
  factors). Block inverses are priced on the profile's own inverse curves, since a chunk does not scale as b^3 (RTX
  3050, thermal noise, per block and shift: 44 us at b <= 67, 59-74 us at b 111, ~0.55 ms at b ~200 looped).
- Not yet: transposed solves of an unsymmetric operator, per-shift rhs, the BATCHED_INVERSE_MAX_BLOCK crossover, a
  device matvec for shift-invert (LineBlockFactorization solves host in / host out), a one-launch sweep of
  solve_shifted chunks or several rhs (only the single-rhs kept sweep is one kernel).

## sparse_direct (`SparseDirectSolver`; roles 'linear' and 'shifted')
Back ends and entry points: sparse_direct.py header and docstrings; cuDSS internals (probes, deferred checks, plan /
ordering reuse, parked plans, ordering store, device-value refreshes, raw phases): cudss_factor.py header; MUMPS /
PARDISO: section "host_direct". Rules across files:
- The runtime keeps no factor: consumers hold their own (one runtime serves the NLFD channels or the thermal noise
  operators) and pass the factor to replace as `previous=` (cuDSS plan / MUMPS-PARDISO analysis reused).
- A host SuperLU factor made on an ordering (`factorize_host(ordering=)`, the runtime's SuperLUOrdering reuse) factors
  A[p][:, p]: solve through `Factorization._lu_solve` / solve_host, never `_host_lu.solve`.
- Back-end groups are module constants (CUDSS_BACK_ENDS, SYMMETRIC_OPTION_BACK_ENDS, MIXED_BACK_ENDS,
  HOST_FACTOR_BACK_ENDS, host_direct.HOST_DIRECT_*), never a tuple spelled out; `host_size_back_end(name)` for
  size_issue.
- Types that factorise host matrices (factorize_host) refuse cusolver_qr at validation
  (`definition_common.host_matrix_back_end_issues`; magnetic_fields_ac its own). cpu backend: scipy_splu or a host
  direct back end (`host_arrays`: numpy in / out).
- cuDSS opt-in: a role lists cudss / cudss_mixed in `setting_choices['back_end']` only when every consumer on its
  grids holds a CudssFactorization (no transposed LU solve, not recyclable, no SuperLU inertia); the 'shifted' roles
  do not. Estimators book the factor as an external item (`linear_solver_cudss(solvers)`,
  `cudss_factor.factor_device_bytes`); the host size rules skip it (`size_issue` None, `cg_solver_issues`).
- 'auto' -> cuDSS (B1b) only through `SolverRole.auto_back_ends` (grep `auto_back_ends=` for the roles), resolved by
  `schema.auto_back_end_values` (gpu with nvmath; values carry AUTO_BACK_END_KEY); role-less `expected_back_end` /
  `resolve_back_end` map 'auto' to SuperLU. The runtime then has `fallback_back_end='scipy_splu'`: a cuDSS refusal
  (CUDSS_REFUSALS) within SPLU_LIMIT_UNKNOWNS switches it to SuperLU for good (`fall_back`, RuntimeWarning, counter
  auto_fallbacks) and the entry point redoes the factorisation there; an explicit cuDSS node raises. Self-tests that
  need a SuperLU reference pin a scipy_splu node.
- Host direct back ends: opt-in only, never 'auto'; on both backends (a cupyx operator is pulled to the host once,
  solutions return as cupy); refusals raise, no fallback. `fall_back`, `device_pencil` and `factorize_device_csr` are
  cuDSS only: ask `pencil` (a HostPencil on MUMPS / PARDISO).
- Entry points on fixed patterns: a DiffusionSystem CSR -> DiffusionDirectSolver (cuDSS: `factorize_device_csr`);
  FEM Newton tangents -> `factorize_host_operator(symmetric_known=, defer_check=)` / `factorize_device_values`;
  unsymmetric Newton Jacobians -> `factorize_values`; frequency sweeps of a pencil -> `pencil` + `factorize_pencil`
  (never a host A(omega) per frequency into factorize_host: root "Performance"); 2D FDFD wavelengths -> `factorize`
  with a valid previous (refreshed from the device values). symmetric_known, pencils and factorize_values match the
  pattern by identity of indptr / indices: callers never modify those arrays in place, and pass symmetric_known only
  for a canonical symmetric CSR. Pencil users book `device_pencil_bytes` as a run item (fem.harmonic, mechanics
  HarmonicSystem; a HostPencil's copied terms as a host item).
- `DiffusionDirectSolver.release_factor()` before another large factorisation, or `detach_factor()` (the caller owns
  the factor: released, or passed as previous= to a factorisation of the same kind); `positive_definite = False`
  first makes factorize_host_operator ask for the LDL^T of a shifted pencil instead of the SPD kind (mechanics stress
  stiffening).
- defer_check (default True in factorize_pencil / factorize_values, False elsewhere): a refresh skips its probe and
  its first solve is the check, so the caller catches CudssAccuracyError (HostDirectAccuracyError) at the solve and,
  when `fall_back(error, n)` is True, drops the factor and factorises again; a first plan is always probed;
  inertia-only factors pass defer_check=False (a count never solves). Callers: fem.harmonic.solve, the FDFD driven
  solves (physics/em sub file).
- solid_mechanics_3d's 'linear' runtime on tetrahedra also factorises the piezo coupled operator (real symmetric
  indefinite) and the complex symmetric frequency-response pencil, although the role's operator class is 'spd'.
- Ordering reuse (cuDSS): across factor objects only while `remember_cudss_orderings` is on (`model.study.run_study`
  per study through `cudss_orderings_kept()`, the GUI worker); off for selftest checks and scripts, so a case never
  plans on another case's ordering. Parked plans live in the same scope (freed at each build start); the on-disk
  store (OPEN_SOL_CUDSS_ORDERING_CACHE) is the GUI worker's only.
- Measure-first switches (REFACTORIZATION_PHASE, RAW_SOLVE, REORDERING_ALGORITHM; off) are turned on only after
  `tools_profilers/time_profilers/cudss_phase_profile.py` shows a gain.
- Time recipe: a caller sets `SolverWork.factor_scale` / `solve_scale` to its own fill over the profile curve (never
  the SuperLU fill unless nothing better is known), `device_values` for device refreshes, `ordering_kept` for a plan
  on a kept ordering. Stored profiles hash sparse_direct.py and cudss_factor.py (`core.time_profile._TIMED_SOURCES`):
  an edit there outdates them.
- Self-tests: `selftests.py` (synthetic operators vs scipy; cuDSS cases gpu only, host direct cases skipped without
  the wrappers). `cudss_factor.gmres_refine` is checked on the GPU by solver_arpack_cudss, solver_sparse_direct_cudss
  and fem_harmonic_cudss_device_pencil, on the host by solver_host_direct_cpu_gpu.
- Validation: the SolverType validate reports scipy / nvmath availability at a node; for Automatic the physics checks
  report it (`definition_fdfd.direct_solver_issues`, also nlfd's) plus the size rule.

## banded (`BandedSolver`; role 'linear' of 1D models, symmetry 2.20)
- The runtime is the PCGSolver protocol: `factorize` copies the CSR to the host and raises ValueError above
  `max_bandwidth` (8, before allocating bands), for an unsymmetric operator (`SYMMETRY_TOLERANCE`) or a non-SPD one.
  Double only; host O(n) memory, not estimated. Not yet: per-order (m / l / q) batching, complex operators.

## arpack_shift_invert (`ArpackShiftInvertSolver`; role 'eigen')
- `eigs(matvec, n, k, v0, tolerance_floor, max_restarts, default_basis, device_matvec, real)` -> (theta, vectors,
  applications, converged): tolerance = max(node, floor) (0 = machine precision); subspace_size 0 = the caller's
  default_basis (`arnoldi_basis_size` for estimators). `real=True` (a real map and v0: cross_section_modes' float64
  own factor) runs host ARPACK as dnaupd (one real solve per application, `last_arnoldi` 'ARPACK (real)'; ncv >= k + 2
  holds); the Krylov-Schur paths stay complex and hand the matvec complex vectors. Time recipe: `SolverWork.rhs` =
  triangular solves per application, `applications_per_basis` (0: APPLICATIONS_PER_BASIS_GUESS 2.5).
- `eigs_host(...)` = `eigs_device`'s 5-tuple on the host (`_HostKrylovSchur`, a subclass sharing run / Schur ordering /
  restart test; numpy rows, CGS2 by gemv on the C-contiguous rows, never a conjugated basis copy): solve_resonances
  around a host-only factor, so a widening resumes (`resume`, `reserve_basis`); below DEVICE_ARNOLDI_MIN_RATIO host
  ARPACK, state None. Runs inside quiet_openblas like the scipy calls, and inside `host_blas_limit(n)`
  (HOST_ARPACK_BLAS_THREADS, 0 = off: every BLAS pool at that thread count up to HOST_ARPACK_BLAS_MAX_UNKNOWNS, through
  `host_direct.limited_blas_threads`, since threadpoolctl lives there; unmeasured, E6) like host ARPACK and the BOR
  port eigs.
- Device Krylov-Schur on a device matvec (E4, `_expand_device`): the H column and |w|^2 of every step stay in device
  rows (`_columns`), read once per expansion (m - keep steps) into a pinned twin; the breakdown test runs on that read
  and a breakdown redoes the steps after its column on a fresh direction. Around a host matvec the packed read per step
  stays (the host solve needs the row anyway). The time recipe prices Sync(1 / (m - keep)) per application there.
- Warm starts (E5): `guarded_start_vector(warm, seed, real)` adds WARM_START_RANDOM_SHARE (by norm) of a seeded random
  vector to earlier modes used as v0 (em_modes frequency scans, the em3d port source re-solve, BOR ports), so a
  wanted eigenvector the warm start lacks keeps a component; group-index and monitor-tracking starts take no share.
- One uninterruptible host call (a GUI stop waits): eigs / eigsh_shifted / eigsh_inverse cap restarts at MAX_RESTARTS
  (300; scipy's 10 n let a shift on a resonance restart for minutes) and return the converged part with converged False
  (callers warn); bor_ports / slab_modes' direct scipy calls take the same cap.
- Tolerance floors only where nothing needs round-off eigenvalues: fem_modes, FDFD / BOR eigenfrequency and
  mechanics keep machine precision, except cross-section solves dropping PML modes on a CPML cross-section (em_modes,
  mode ports, bends; not EME): 1e-12 (`cross_section_modes.PML_ARPACK_TOLERANCE_FLOOR`; PML-mode clusters hit
  MAX_RESTARTS at machine precision); em_modes' own solves at the study wavelengths 1e-10
  (`cross_section_modes.MODE_ARPACK_TOLERANCE_FLOOR`, E3; its group-index, detached and port solves keep machine
  precision); em_fdfd_3d with a cpml face 1e-10 (+ a 5 k basis, ARNOLDI_BASIS_PER_WANTED);
  physics/em/conical 1e-10. Callers without a runtime get a default one inside `solve_cross_section_modes` /
  `solve_resonances` / `solve_bor_resonances`; BOR port mode solves call scipy eigs directly (no role).
- `eigsh_inverse(...)`: factor-free symmetric shift-invert at 0 for a possibly inexact inverse (solid_mechanics_3d: one
  PCG solve each); counted like eigs.
- nvmath / cuDSS API notes, pitfalls and measurements: `nvmath_guide.md` (repo root).
- Host direct factorizations mumps / mumps_mixed / pardiso / pardiso_mixed (FACTORIZATIONS order: auto, superlu,
  cudss, cudss_mixed, then HOST_DIRECT_FACTORIZATIONS): factorize_host returns a HostDirectFactorization (host
  solves, the caller's ordering unused, symmetric / row_scale / previous honoured: LDLT_FACTORIZATIONS = cuDSS + host,
  for which _factorize_shift defaults row_scale to ShiftPencil.row_scale); a host *_mixed request below
  mixed_min_unknowns factors in double (S5c, `host_direct.double_of`; MIXED_FACTORIZATIONS = cudss_mixed + the host
  mixed names); inertia from the library pivots (real LDL^T / Cholesky: MUMPS INFOG(12), PARDISO iparm[21] / [22]);
  None (LU, complex, symmetrizer, perturbed or null pivots) -> one counting SuperLU in _shift_inertia (exact); a
  Hermitian pencil is always counted by `_superlu_count`. No deferred check, no fallback (refusals raise). Counters:
  eigen_host_direct_analyses, eigen_lu_entries, eigen_factor_symmetric, eigen_double_fallbacks,
  eigen_refinement_steps, eigen_host_direct_threads; time recipe the SuperLU items x sparse_direct.HOST_DIRECT_*.
- Setting `factorization` 'superlu' / 'cudss' / 'cudss_mixed' / 'auto' (default). cudss is GPU only (`backend_misfit`;
  validate errors without nvmath, `cudss_factor.cudss_available` = find_spec); `factorize_host` then returns a
  `CudssFactorization` for every caller (the eigen callers' Factorization stand-in; no transposed solve, not
  recyclable), the caller's `ordering` unused. Host planning single-threaded: never pass the wheel's threading layer
  (`cudss_mtlayer_vcomp*` crashed the process on Windows); only an env CUDSS_THREADING_LIB, read by nvmath, loads one.
- cuDSS memory: the factor lives outside the cupy pool; after planning a peak estimate above DEVICE_FRACTION (0.8) of
  free device memory is a ValueError. Estimators count the pool part through the cudss_factor helpers only: kept per
  plan `operator_copy_bytes` (CSR copies, diagonal, the mixed factor copy) + `solve_buffer_bytes` (rhs buffer, row
  scale); one solve or probe `solve_peak_bytes` (4 vectors; mixed 2 GMRES_RESTART + 10: the GMRES basis lives only in
  the solve) as a transient (CUDSS_SOLVE_PHASE, or inside the phase of an own factor), never kept; a caller
  refactorising on the planned pattern adds `index_map_bytes` (kept maps: a run item) and `refresh_peak_bytes(...,
  n_unknowns=, mixed=)` (a refresh, or the probe on the still-held uploaded / equilibrated values: a transient item;
  `refresh_bytes` = maps + refresh) and the factor itself as MemoryEstimate external_items / external_extra from
  `cudss_factor.factor_device_bytes(n, entries, value_bytes, dimensions, mixed, symmetric)` (permanent, peak extra;
  `definition_helpers.cudss_external` wraps several, `phases=` binds a factor released after its solve to the
  transient phases it lives in): fill 2D n log2 n, 3D n^(4/3), scaled by sqrt(rows per reference
  row), a general LU x CUDSS_LU_OVER_LDLT (L and U), a mixed factor halves only the values (CUDSS_INDEX_BYTES per
  entry stay), peak = permanent (CUDSS_PEAK_RATIO 1: cuDSS reports them equal); a `FactorFill(entries)` given as
  `dimensions` replaces the growth law by counted factor entries (fem.dissection on a built element mesh; mechanics
  triangles / tetrahedra: `definition.factor_fill`; em_fdfd_3d tetrahedra: `definition_fdfd3d.nedelec_factor_entries`,
  also for its host SuperLU via `core.memory_estimate.host_counted_lu_bytes`; 3D tensor grids:
  `sparse_direct.box_dissection_entries(window shape, unknowns per cell)`, the box nested-dissection count of A5, for
  which CUDSS_FILL_3D read 48 % / 76 % low; `yee_faces=True` for the Yee curl-curl: tangential-component separators) x the factor value bytes + the kept matrix copy +
  CUDSS_ROW_BYTES per row; the growth-law fill fitted to the selftest
  counters (constants, data, the known double count and open points in the cudss_factor notes above it). The pool items are counted from the
  arrays the code makes (an estimate above the measured bytes misreads the code); a wrong entries-per-row count is the
  usual miss, as every per-entry item scales with it; refit the fill
  from summary.json counters cudss_permanent_bytes / cudss_peak_bytes / cudss_lu_entries (sparse_direct),
  eigen_permanent_bytes / eigen_lu_entries (arpack_shift_invert) and the run's external_memory record, never from
  cudss_factor_device_bytes / eigen_factor_device_bytes (`CudssFactorization.device_bytes` = permanent + the in-pool
  copies, rhs, scales and index maps the estimators already book as pool items: fitting to it counts them twice).
  Estimates are counts of what the code and cuDSS should hold, never constants scaled to measured bytes: a scaled
  constant absorbs an extra copy that the estimate-vs-measured verdict should expose. `CudssFactorization._plan_solver` / `_free_solver` book every planned solver in the
  process-wide `external_live_bytes` / `external_high_water_bytes` (reset per build by study.build_session): free a
  DirectSolver only through `_free_solver`. em_fdfd_3d's host size rule is skipped; 2D callers keep their SuperLU size
  rules and items.
- cuDSS time: 3D by the fitted `CUDSS_*` constants (`_cudss_ops`; fitted on complex curl-curl on one RTX 3050:
  recalibrate on another GPU from the log's plan / factorise / device-solve seconds; real operators by CUDSS_REAL_*,
  uncalibrated); 2D by sparse_direct.cudss_2d_ops. Callers pass the run backend for a cuDSS factor, never 'cpu' (that
  prices device work on the host profile). A fresh plan per factorisation unless SolverWork.plan_reused (callers
  keeping the plan via previous=, e.g. modal.ModalPencil).
- Mode infos take `factor_kind` from the factor's description ('SuperLU' only when it has none; em_modes'
  solve_summary branches on 'cuDSS'). `eigenfrequency.solve_resonances` with its own cuDSS factor applies B and both
  Cayley solves on the device and logs plan / factorise / device-solve seconds.
- `factorize_host(symmetric=True)` (cudss only): `relative_asymmetry` <= SYMMETRY_TOLERANCE (1e-12, transpose not
  conjugate), then the lower triangle as SYMMETRIC / LOWER (LDL^T: about half the factor); no full copy kept: residuals
  apply E y = L y + L^T y - diag(L) y (`_product`; the cached `L.T` is a non-conjugate CSC view); else general LU,
  keeping the full E. Users: eigsh_shifted and `solve_resonances(symmetrizer=D)` (factors D A, applies D B): em_fdfd_3d
  D = stretched face dual volumes (half on antisymmetric faces; None with a Bloch phase), 2D FDFD D =
  `operators.stretched_cell_areas` (None with Bloch; hz eps_xy couplings fall back to LU via the asymmetry check); BOR
  none yet.
- Both cuDSS kinds factor E = R A R, R = |diag A|^-1/2 (probe residuals are of E): unscaled, pinned identity rows of a
  D-scaled operator sat below cuDSS's pivot epsilon. A caller's `row_scale` replaces R where diag A may nearly cancel
  (em3d `FEMPhysics3D`: (|diag K| + k_t^2 |diag M|)^-1/2). `solve_resonances(symmetric=True)` = A itself complex
  symmetric (FEM pencil): the same LDL^T.
- 'cudss_mixed': complex64 / float32 factor of E (keeping complex symmetry), float64 E on the device for residuals.
  `solve(rhs, refinement_steps=None)` refines by restarted flexible GMRES preconditioned by the low-precision solve
  (module `gmres_refine` with xp = cupy, shared with host_direct; GMRES_RESTART 8, vectors in
  `solve_peak_bytes(mixed=True)`) to MIXED_SOLVE_TOLERANCE (1e-12) or
  backward error MIXED_SOLVE_BACKWARD_ERROR (1e-15), at most MIXED_MAX_REFINEMENTS 16; a cycle whose FGMRES estimate met
  the target while the true residual did not is the float64 floor and ends the solve. Consumers checking true
  residuals (reduced sweep rom_tolerance) need this (a 1e-12 backward error on resonance left 1e-7 - 1e-5 residuals).
  Target per factor `solve_backward_error` or per call `solve(..., backward_error=)`; probes / deferred checks capped at
  BACKWARD_ERROR_TOLERANCE. An int fixes plain refinement steps (eigen polish; contracts ~eps32 kappa(E) per step, poor
  near high-Q drives, which GMRES fixes). Each single solve scales its rhs to unit norm (float32 underflow).
- Mixed probes / deferred checks stop GMRES after MIXED_STALL_ITERATIONS (4) when the extrapolated rate misses the gate
  by MIXED_STALL_MARGIN; never a threshold on the unrefined residual alone (high-Q leaves it ~0.4 and GMRES passes). A
  mixed probe that misses re-plans double (`_next_attempt`; positive_definite LDL^T -> double Cholesky):
  `last_double_fallback`, `mixed_rejection`, and the object stays double (sticky). While `remember_cudss_orderings` is
  on, the rejection is kept per (kind, real, pattern digest) (<= KEPT_ORDERINGS): a new factor of that key goes double at
  once (mixed_rejection ends in '(remembered ...)', last_double_fallback False, so cudss_double_fallbacks counts real
  rejections only); values are not in the key. `precision` stays the request (can_refactorize); `factor_precision` /
  `mixed_factor` say what is factored: callers choosing loose solves or log texts branch on factor_precision.
  Errors `CudssSingularError` / `CudssAccuracyError` (both RuntimeError; fem.harmonic words them apart).
- cudss_mixed only for roles whose `setting_choices['factorization']` accept it (`eigen_role(mixed_factorization=True)`:
  em_fdfd_3d, em_fdfd_ez / _hz), because only `eigenfrequency.solve_resonances` asks for loose solves: with an own mixed
  LDL^T of D A it uses MIXED_EIGS_REFINEMENTS (1), floors ARPACK at MIXED_ARPACK_TOLERANCE (1e-6) and replaces every k^2
  by the float64 Rayleigh quotient (stationary for the complex symmetric pencil; `info['rayleigh_change']`); a double
  own LDL^T (cuDSS / MUMPS / PARDISO) of a complex A whose caller floored ARPACK (cpml / PML / lossy) takes the same
  floor and polish, without the refinement steps; without a symmetric D A (Bloch) it refines every solve to 1e-12. Measured on the Mie eighth sphere: factorise 12x faster, half
  the memory, lambda / Q identical to double.
- Probe gates: a probe after factorize must reach ACCEPT_RESIDUAL (1e-10) or cuDSS refinement (REFINEMENT_STEPS) is
  switched on; still missing = RuntimeError (cuDSS pivots only within supernodes). Every gate and the mixed stop also
  accept a normwise backward error <= BACKWARD_ERROR_TOLERANCE (1e-12): a random probe's residual floors near eps
  kappa(E) on stiff thin structures, out of reach of 1e-10. `probe_backward_error` beside probe_residual. Counters
  eigen_factor_device_bytes / eigen_refinement_steps. Selftest solver_arpack_cudss (gpu; skipped without nvmath; fails
  when a mixed factor fell back).
- Not yet: a device ARPACK loop, a device-memory estimate of the factor. Multi-rhs solves were measured and do not pay
  (factor read bandwidth-bound per rhs): no block Krylov.
- `eigsh_shifted(stiffness, mass, shift, n, start, row_scale=None)`: cuDSS equilibration defaults to (|diag K| + |sigma|
  |diag M|)^-1/2 (never cancels); SuperLU of K - shift M (real: float64; complex Hermitian stays complex128 via eigs mode
  3); returns the factor; singular raises RuntimeError (caller words it). Factor and eigsh use sigma = shift (1 +
  EIGSH_SHIFT_RELATIVE_OFFSET 1e-4), so a target on a resonance is not singular; a zero shift gets no offset (rigid-mode
  checks keep K SPD; only an unchecked mechanism is singular, modal.py words it).
- `eigs_shifted(...)`: complex symmetric K with real semi-definite M (solid_mechanics_3d resistive_load): same sigma and
  equilibration, complex `factorize_host(symmetric=True)`, eigs on (K - sigma M)^-1 M, lambda = sigma + 1 / theta (theta
  = 0 -> NaN). Not for a Hermitian pencil (eigsh_shifted keeps M-orthogonality). A caller's `pencil=` keeps its own
  `symmetric` flag: a ShiftPencil(symmetric=False) factors by LU (modal.ModalPencil does so for a damped pencil reduced
  by a complex Bloch P: P^H A P of a complex symmetric A is neither symmetric nor Hermitian).
- 'auto': the role's `auto_back_ends` (`eigen_role(auto_factorizations=)`) on the gpu with nvmath, else superlu (the
  runtime maps an unresolved auto to superlu). A runtime resolved from auto falls back to SuperLU for good on
  CUDSS_REFUSALS (factorize_host, the ShiftPencil device path, the deferred check inside eigsh_shifted / eigs_shifted;
  counter eigen_auto_fallbacks) and counts a Hermitian pencil by a counting SuperLU (`_superlu_count`). A caller's
  `fallback_refusal` hook (n_unknowns -> '' or a reason, asked only when a fallback would happen) turns that fallback
  into a RuntimeError with the remedy (A4: em_fdfd_3d tetrahedra above `NEDELEC_SUPERLU_SLOW_BYTES` of counted fill,
  where SuperLU takes tens of seconds to minutes). Opted in:
  solid_mechanics triangles (cudss), solid_mechanics_3d tetrahedra (cudss), em_fdfd_conical (cudss), em_fdfd_3d
  (tensor, tetrahedra: cudss_mixed, the polished path of solve_resonances; Bloch: a general LU refined to 1e-12); other em /
  membrane / eme eigen roles stay superlu (own-factor and SuperLU-inertia consumers, unchecked).
- Several shifts: eigsh_shifted / eigs_shifted / factorize_host take `previous=` (cuDSS refactorised in place,
  `last_plan_reused`; SuperLU ignores it). Only because the shifted operator is built by `pattern_combination((K, M), (1,
  -sigma))` (root "Solver contract"): any new caller combining a pencil for a reused factor goes through it.
- `ShiftPencil` (`shift_pencil`): terms on one pattern once (M and the diagonals placed by `merge_positions`, a
  row-parallel numba merge, host twin 'solvers.pattern_merge', self-test solver_pattern_merge_numba; a global key
  searchsorted took 1.7 s on a 3D P2 pencil); host route M products for ARPACK by `mass_product` (numba prange CSR
  product, host twin 'solvers.csr_product', from MASS_PRODUCT_MIN_ENTRIES; scipy's is single-threaded) through
  `mass_operator`; SuperLU one-pass host combination per shift; cuDSS a
  DevicePencil + `factorize_pencil` on a private SparseDirectSolver (eigen_cudss_plans / eigen_double_fallbacks /
  eigen_cudss_reused_orderings / eigen_cudss_plan_seconds folded in; `eigen_reused_orderings` is SuperLU's order count,
  not cuDSS's). count_below / eigsh_shifted / eigs_shifted take `pencil=` to keep it across shifts. `device_mass()` uploads
  the host M on its own pattern (a vector FEM mass: a third of the pencil pattern), not the pencil's mirrored M term.
- Inertia (Sylvester; real symmetric / Hermitian with M positive definite, or semi-definite with a fixed massless block
  such as piezo phi: then only count differences are meaningful): `eigsh_shifted(..., inertia=True)` appends
  `ShiftInertia(sigma, n_below, exact)` at the offset sigma; `count_below(K, M, sigma, previous)` at sigma itself
  (singular: retried once at sigma (1 + COUNT_SHIFT_NUDGE); a refreshed DevicePencil factor is probed at once,
  `factorize_pencil(defer_check=False)`: a count never solves, so a deferred check would never judge the pivots it
  reads). Sources: `CudssFactorization.inertia()` (real LDL^T / Cholesky only; None for LU, complex symmetric and
  symmetrizer factors; exact False when `factor_precision` is mixed, so a double fallback counts as exact) and
  `Factorization.inertia()` (U diagonal signs, valid only when perm_r == perm_c, else an extra counting SuperLU with
  diag_pivot_thresh 0). A Hermitian pencil on cuDSS has no count. Counting factorisations count in
  eigen_factorizations: a time recipe pricing a band search counts them there.

## block_eigen (`BlockEigenSolver`; role 'eigen', opt-in)
- Factor-free LOBPCG of the lowest eigenpairs of a symmetric pencil on caller block callbacks (module notes: layout,
  the five steps of an iteration, mixed phases). Opt-in only: SolverType `opt_in_roles=('eigen',)` and the physics'
  role from `definition_helpers.eigen_role(block_eigen=True)` (operator 'spd', precisions double / mixed,
  opt_in_types ('block_eigen',)); grids ('tensor',), so a tetrahedra / triangle physics with that role misfits it.
  Today: solid_mechanics_3d on brick meshes (`ElasticSystem3D._lowest_modes_block`). A new consumer: an 'spd' eigen
  role with block_eigen=True, a dispatch on `isinstance(eigen_solver, BlockEigenSolver)` where it calls the eigen
  runtime, its validation of what LOBPCG cannot do (interior targets, bands, zero eigenvalues: the relative residual
  divides by |lambda|), and memory / time items (below).
- Callback contract: `(rows_in, rows_out)` on C-contiguous (m, n) row blocks of the backend array module, float32 or
  float64 (mixed calls both: keep an operator copy per dtype), rows_out written, never aliased with rows_in; rows_in
  may be the runtime's scratch (the preconditioner's input rows are overwritten afterwards). With a `mask` every row is
  0 outside it and K / M must keep that (masked products); the runtime masks the preconditioner output itself. The
  preconditioner must be SPD (a symmetric V-cycle, block Jacobi), not exact: no inner solves, no tolerance coupling.
  A callback gets every row of a step in one call and must run them in ONE launch (a per-row loop of single-vector
  launches is latency-bound and leaves the GPU idle: the reason the Lanczos path does not load it): solid_mechanics_3d
  uses the `*_rows` kernel entries (one thread per (row, node)) and `multigrid3d.BlockCycle`.
- Only the wanted rows must converge; the guard rows widen the search space. A consumer asks exactly the eigenpairs
  it keeps (no ARPACK-style margin such as EXTRA_EIGENVALUES on top: that only adds pairs to converge).
- Memory: 9 b n values of the phase dtype (S, KS, MS of (3 b, n); b = wanted + guards) + CHUNK_BYTES scratch, nothing
  else of that size: every block product runs by column chunks in place (`_rotate`, the W projection), compaction is
  forward row copies, the start rows are drawn row by row on the gpu. Mixed frees the float32 stores before the
  float64 ones (the float32 X is the only copy held across): `block_eigen.block_bytes` is the estimate item (a
  transient of the eigen solve; the consumer adds its own copies, e.g. float32 operator copies and a private
  multigrid hierarchy). Host: the k eigenvectors only.
- Syncs per iteration: two host reads (residual norms, Gram pair) and one small upload (the rotation coefficients);
  the block callbacks stay asynchronous. The Rayleigh-Ritz is a host eigh of
  (3 b)^2 (diag(B)^-1/2 scaled, B eigenvalues below DROP dropped: dependent W / P directions near convergence).
- Mixed: float32 to max(tolerance, SINGLE_PHASE_TOLERANCE) or a STALL_ITERATIONS stall (the float32 floor of the
  products is ~eps32 lambda_max / lambda: ~1e-3 on fine 3D elasticity, so the float64 phase is always needed for tight
  tolerances; it starts from the float32 X with a fresh P). Double stalls over DOUBLE_STALL_ITERATIONS stop
  unconverged (tolerance below the float64 floor), with a warning. A stall counts only once the worst wanted residual
  has been below STALL_ARMED_RESIDUAL: the O(1) start-up plateau of a weak preconditioner is no rounding floor (it
  runs to the iteration cap instead). The cap is the node's max_iterations, else DEFAULT_MAX_ITERATIONS (never a
  linear CG limit).
- Fewer free unknowns than DENSE_MAX_RATIO x b: a dense scipy eigh of the applied K and M (`eigen_dense_solves`).
- Counters: eigen_solves, eigen_iterations / _single_ / _double_iterations, operator_applications (K rows),
  mass_applications, preconditioner_applications, eigen_seconds, eigen_unconverged, eigen_dense_solves,
  eigen_block_size, eigen_last_residual. Time recipe `block_eigen.time_estimate`: per iteration and row the caller's
  product_ops (K and M) + preconditioner_ops, block passes, DenseFlops (single for the mixed share), 2 syncs;
  iterations a priori (multigrid: MULTIGRID_ITERATIONS_PER_DECADE x decades, flagged by work.setup_ops; Jacobi:
  sqrt(kappa) or CG_ITERATIONS_PER_AXIS_CELL x longest axis): uncalibrated guesses.
- Not yet: warm starts across sweep values / assemblies, hard locking (converged rows leave X), a fused residual +
  norm kernel, dropping unconverged pairs from the result (today returned with a warning), FEM pencils (tetrahedra /
  triangles: SA-AMG of K + tau M as preconditioner), Maxwell (gradient-space projection + auxiliary-space
  preconditioner).
- Self-tests: solver_block_eigen_cpu_gpu (synthetic 3D Laplacian pencils, both backends and precisions, mask, Jacobi
  callback, dense path, counters); solver_default_precision rows eigen_* (role resolution, mixed keeps multigrid);
  model runs: mechanics sub file "Self-tests".
- Open estimator gaps (profiled 2026-11-17): mixed under-books the device peak by ~3·3b·n·4 B on large meshes (the
  float32 stores stay cached in the pool when `_to_double` allocates the float64 ones: reuse one flat float64 buffer
  per store instead); the chunk scratch is booked as a flat 2·CHUNK_BYTES whatever n (over-books small meshes); the
  time recipe never prices a private multigrid build (`work.setup_ops` is only a flag), only the shared one through pcg.

## Tried and dropped
Three opt-in solvers, profiled on 2026-11-17 and dropped. Code: tag `archive/four-new-solvers`; the latest version
(merged onto later main, with their sub-file sections): tag `archive/batched-direct-time-filter-mumps-blr`. Do not
rebuild one without a reason that answers these findings.
- batched_direct (same-pattern frequency sweeps in cuDSS uniform / explicit batches): the uniform batch factorised at
  ~2x (up to 3x) the per-system numeric cost and solved at the same per-system cost (0.35M-1.8M stored entries), so
  sweeps ran 1.1-2.2x slower than per-system cudss_mixed (one analysis, values refreshed per frequency).
- time_filter (factor-free FDFD resonances from an FDTD run + Rayleigh-Ritz): found 0-3 of 6 resonances on open (CPML)
  disks, residuals 5e-5-1e-3 even in double, and the stepping alone cost about the factor's wall at 4e5 cells. Both
  tags predate gyration_z: a rebuild must refuse it in validation and unsupported_reason (em_fdtd_hz never reads it).
- mumps_blr (block low-rank MUMPS + GMRES): compressed 2-10 % of the entries on the profiled 3D cases, factor 1.4-2.1x
  slower, no memory saved (only ICNTL(35) / CNTL(7) set: MUMPS allocated for the full-rank factor). A retry needs the
  memory flags, a looser drop threshold and a case much larger than solid_mechanics_3d_tetrahedra_large.

## host_direct (`HostDirectFactorization`; MUMPS / MKL PARDISO back ends of sparse_direct and arpack_shift_invert)
- Factor of E = R S R (S = D A with a symmetrizer, R = |diag S|^-1/2 with zero / non-finite / missing diagonals 1, or
  the caller's row_scale), entry by entry on an own copy of the canonical pattern, as CudssFactorization: LDL^T when
  symmetry was requested (symmetric, positive_definite or a symmetrizer) and relative_asymmetry <= SYMMETRY_TOLERANCE
  (complex symmetric, not Hermitian), else LU; PARDISO real positive_definite = Cholesky (matrix type 2; a zero-pivot
  error -> LDL^T, sticky); MUMPS always LDL^T for symmetric kinds (python-mumps sets SYM=2 and exposes no SYM=1).
  `back_end` = the library ('mumps' / 'pardiso'), `precision` the request, `factor_precision` what is factored;
  `description` starts with the library label ('MUMPS LDL^T', 'PARDISO LU float32 + float64 GMRES refinement').
  The scaling itself is `scaled_values` (a new array; a core.kernels.host_twin: numba prange over rows from
  HOST_SCALE_MIN_ENTRIES entries when numba is already in sys.modules, else the numpy form `_scale_host`: it never
  imports numba, ~1 s per process against ~30 ms of gathers at 3e6 entries; next-gains N6, self-test
  solver_host_scale_numba).
- Solves: A x = b as x = R E^-1 R (D) b; A^T x = b as R E^-T R b (no symmetrizer) or D R E^-T R b (symmetrizer;
  E^-T = E^-1 for a symmetric kind); a real factor takes a complex rhs as two real solves; solve() of a cupy rhs moves it
  to the host and the solution back (host_arrays / numpy rhs: numpy). `_lu_solve(rhs, trans)` serves
  DiffusionDirectSolver and sparse_direct.recycled_gmres (solve_preconditioned*: recyclable).
- Probe after every factorisation (cudss_factor gates: ACCEPT_RESIDUAL or BACKWARD_ERROR_TOLERANCE); a double miss
  switches the factor to float64 GMRES-IR (gmres_refine with the double factor as preconditioner, every later solve;
  refinement_steps = the probe's iterations), a second miss releases and raises HostDirectAccuracyError. The libraries'
  own refinement (MUMPS ICNTL(10), PARDISO iparm[7]) is not used. PARDISO perturbs tiny pivots silently (1e-13
  unsymmetric, 1e-8 symmetric): a singular operator mostly shows as an accuracy error, not HostDirectSingularError.
- Mixed (*_mixed): float32 / complex64 factor, float64 E kept; full-accuracy solves by gmres_refine to
  solve_backward_error, an int refinement_steps = plain steps; each low-precision solve scales its rhs to unit norm;
  the probe's stall test and mixed gates as cuDSS; a miss refactors in double (sticky: last_double_fallback,
  mixed_rejection). No low-precision Cholesky (as cuDSS), no remembered rejections per pattern (cuDSS H4).
- Memory: after the analysis the library's estimate (MUMPS INFOG(17) MB read as MiB, PARDISO max(iparm[14],
  iparm[15] + iparm[16]) KiB) above HOST_FRACTION (0.8) of `core.host_memory.available_bytes()` is
  HostDirectMemoryError (no guard when unknown); library allocation errors map to it too. The factor lives in library
  memory outside numpy: holders release() it before dropping or replacing it (host_direct_factorization releases a
  `previous` it cannot refactorise). Estimators: `factor_host_bytes` (SuperLU heuristics x `superlu_bytes_scale`:
  LU_FILL_OVER_SUPERLU / LDLT_FILL_OVER_SUPERLU x WORKSPACE_OVER_FACTOR x the mixed value share, uncalibrated: refit
  from host_direct_lu_entries / host_factor_bytes) + `solver_host_bytes` (E, row scale, the wrapper's matrix copy,
  mixed: the factor-dtype CSR and one solve's GMRES vectors).
- Refactorize: the same canonical pattern and kind = numeric factorisation on the kept analysis (last_plan_reused);
  pydiso's refactor checks only the nnz count, so the pattern is compared here first. The per-entry rows and diagonal
  positions are kept per pattern and ||E|| (`operator_norm`) is lazy: a double probe whose relative residual passes
  ACCEPT_RESIDUAL leaves probe_backward_error None (cuDSS R4).
- Kept analyses (`host_analyses_kept()`, entered by run_study beside `cudss_orderings_kept`; off for scripts and
  selftest checks): release() parks the library factor (analysis and factor memory) and a new analysis of another
  factor object adopts a parked one of the same library, kind, real, factor dtype and canonical pattern (compared
  entry by entry; pydiso checks only the nnz), so a sweep value or rebuilt session with a known pattern runs the
  numeric factorisation only (`last_analysis_kept`, counted in host_direct_reused_analyses; no memory guard, the bytes
  were already held). A miss releases every parked factor before it analyses (a changed mesh never holds a stale
  factor beside the new one; a pattern released mid-build loses its entry to the next other pattern), at most
  KEPT_ANALYSES (4) stay parked, a failed factorisation never parks, the block's end releases the rest. The parked
  bytes are no estimate item: the adopting build's own factor item is that memory; until then
  `core.host_memory.available_bytes()` (the host budget, the guard) sees them as used (`kept_analysis_bytes()`). MUMPS
  takes the same path (`_MumpsFactor.refactor` reuses its Context's analysis).
- PARDISO thread rule (gotcha): phase 22 on a handle whose phase 33 ran on one thread (the serial solve below
  `_PARDISO_SERIAL_SOLVE_LU_ENTRIES`) gave garbage factors (probe residuals 1e20 - 1e125, "0 perturbed pivots") and
  access violations (em_fdfd / em_modes / em_nlfd / semiconductor / magnetic example profilers, 2026-11-12); handles that
  solved on all threads (>= 1e6 entries: potential cylinder, membrane) refactor fine. So `_PardisoFactor.solved_serially`
  makes the handle not `refactorable` (`refactor` raises): it is neither reused in place nor parked. A factor object
  that then refactorises in place analyses anew once with `keep_threads` (every phase on the library thread count, no
  serial solves) and reuses that analysis afterwards; small-factor sweeps of new objects (em wavelengths) analyse per
  value as before the kept analyses. Checked by solver_host_direct_cpu_gpu (value sweeps at the rule and with every
  solve on all threads).
- `refactorize_values(values, indptr, indices, row_scale, symmetric_known, defer_check)` (HostPencil route; no
  symmetrizer): the pattern matched by identity with the last refactorize_values arrays (a plain refactorize clears
  that), so its caller never modifies them in place. defer_check: a double factor on the kept analysis whose last
  probe passed unrefined skips the probe and `_checked_solve` judges the next solve (miss: GMRES-IR and solve again;
  second miss: released, HostDirectAccuracyError from solve / solve_host, which consumers translate as at factorize);
  probe_residual lags one factor. Mixed factors always probe (no in-place double re-factor from a solve).
- PARDISO values: `_PardisoFactor` checks once per analysis that pydiso's private `_data` / `_indptr` / `_indices`
  hold the pattern as kept here (the upper take of a symmetric type) and then writes the values straight into the
  handle and calls `_factor` (no sp.triu / sort / sum_duplicates per refactor); any other pydiso layout uses the
  public refactor. A pydiso update that renames those: the check fails quietly and the gain is lost, never results.
- Library notes (PARDISO: run on pydiso 0.3.1 / oneMKL 2026.2; MUMPS: verified against the python-mumps 0.0.6
  sources, not run): python-mumps
  `Context.set_matrix(a, symmetric=True)` keeps scipy.sparse.triu itself and creates the s/d/c/z instance only when
  the dtype changes (SYM stays that of the first call: one Context per analysis); icntl / infog 1-based; ICNTL(9) = 2
  solves A^T (believed unconjugated for complex: checked by solver_host_direct_cpu_gpu only); INFOG(12) negative
  pivots of SYM=2; MUMPSError.error = INFOG(1) (-10 / -6 singular; -5, -7, -8, -9, -13, -17, -19, -20 memory);
  `factor()` retries -8 / -9 itself. pydiso: symmetric types keep sp.triu of the matrix given; MKL needs every diagonal
  entry of a symmetric type stored, so the factor adds explicit zero diagonal slots once per pattern (the same padded
  pattern every refactorisation); `solve(transpose=True)` = iparm[11] = 2 (A^T, not A^H); float32 data sets iparm[27]
  (single precision); PardisoError carries only a message (mapped back by `_err_messages`); release = dropping the
  handle (dealloc phase -1). Every library call sits in `_MumpsFactor` / `_PardisoFactor` at the end of the module.
- PARDISO ordering: the analysis asks for the OpenMP nested dissection (iparm[1] = 3, `_PARDISO_ORDERING`; an older
  pydiso without the keyword, 0.2.0 in the conda env, runs `_ordered_pardiso_class`, which sets iparm[1] on the
  handle before the analysis its constructor runs: without it the analysis was serial METIS, 2.4 s of a 6 s heat
  profile): the same fill as METIS at half the analysis time on
  2 - 4 threads (with METIS the analysis is 50 - 85 % of a cold PARDISO factorisation up to ~600 000 unknowns), equal
  on one thread, every matrix type and both precisions. Measured and not adopted: minimum degree (iparm[1] = 0: the
  fastest cold factor on 2D grids, 2 - 6 x slower solves and 1.7 x the fill in 3D), the two-level factorisation
  (iparm[23]) and iparm[24] (nothing on 4 threads; untested on more).
- Where a PARDISO refactorisation spends its time (2D FDFD, 160 000 unknowns, 4 threads: ~185 ms): the numeric
  factorisation ~50 %, the equilibration + asymmetry check ~30 % (single-threaded numpy / scipy; `symmetric_known`
  skips the check), pydiso's own `scipy.sparse.triu` of a symmetric type ~6 %, the probe solve ~12 % (before the
  HostPencil route; FEM sweeps on factorize_host measured 2026-11: pressure_acoustics_tetrahedra ~250 ms per frequency
  of which MKL ~100 ms, host A(omega) 20, equilibration 26, asymmetry 17, probe 17). *_mixed halves
  the factor memory but costs time in 2D (every full-accuracy solve = 3 library solves + GMRES; the float32
  factorisation is only ~1.4 x faster): it pays for memory, or where the factorisation dominates (large 3D).
- Availability (GUI-thread safe, nothing imported): MUMPS = find_spec('mumps') and the python-mumps distribution
  metadata (PyMUMPS installs another `mumps`); PARDISO = find_spec('pydiso') on x86-64. `missing_text(name)` is neutral
  ("'<name>' needs ...": back_end and factorization settings alike), `install_hint`, `factor_label` (texts).
- Threads: library defaults (MUMPS OpenMP / BLAS: OMP_NUM_THREADS, recorded; PARDISO MKL_NUM_THREADS,
  get_mkl_pardiso_max_threads recorded); no open_sol setting, oversubscription with numba / scipy BLAS threads is the
  user's to set. Exception: a PARDISO solve (phase 33) of a factor below `_PARDISO_SERIAL_SOLVE_LU_ENTRIES` (1e6)
  runs on one MKL thread, and the count is restored after (fluid heat transient, 57 000 entries: 0.88 ms per solve on
  16 threads, mostly barriers). Both wrappers release the GIL in their calls; the calls are uninterruptible like SuperLU.
  `_PardisoFactor.solve` sums MKL's own refinement steps (iparm[6]; iparm[7] = 0 still refines twice per solve when the
  factor has perturbed pivots, iparm[13]) into `solve_text` (solve_resonances' factor text via `library_solve_text`);
  `OPEN_SOL_PARDISO_SOLVE_PARALLEL` (temporary measurement override) writes iparm[24] before a handle's first solve.
- PARDISO beside OpenBLAS (pip wheels and conda-forge defaults: numpy / scipy on OpenBLAS, pydiso on MKL): both pools
  spin after a call, so an MKL call right after a threaded BLAS call (a norm, a Gram-Schmidt product, ARPACK's own
  BLAS) shares the cores with spinning OpenBLAS workers (4 cores: a solve 15 -> 70 ms, a mixed solve 75 -> 450 ms, an
  ARPACK shift-invert run 0.16 -> 4 s). `host_direct.quiet_openblas(name)` (context; a back end / factorization /
  library name) holds every loaded OpenBLAS pool at one thread through threadpoolctl and restores it on exit: nothing
  for a name that is not PARDISO (MUMPS's BLAS may be that OpenBLAS), without threadpoolctl (optional, in the
  `pardiso` extra) or with numpy on MKL (one pool, no conflict). HostDirectFactorization wraps refactorize, solve_host
  and solve_preconditioned* itself (`openblas_pools` = pools held, 0 none); a consumer whose own loop does BLAS work
  between PARDISO solves wraps the loop (arpack_shift_invert: the scipy eigs / eigsh calls, by `self.factorization`).
  `limited_blas_threads(threads)` (every BLAS pool, MKL included, at `threads`; nothing for 0) is the other
  threadpoolctl user, for arpack_shift_invert.host_blas_limit (E6; contract direct_wrappers_lazy keeps threadpoolctl here).
  It is process-wide while held: only vector-sized BLAS work belongs inside, never a dense factorisation. BLAS calls
  of a physics between two PARDISO calls stay outside it: OPENBLAS_NUM_THREADS=1 (INSTALL.md) covers those. The pools
  are looked up once; `refresh=True` (refactorize / refactorize_values) looks again only when a scipy module that loads
  scipy's OpenBLAS (`_SCIPY_BLAS_MODULES`) appeared in sys.modules since (a ThreadpoolController walks every loaded
  library: 4.5 ms per Newton factorisation). A new lazily loaded BLAS user of scipy joins that list.
- Not yet: out-of-core (MUMPS ooc / ICNTL(22), PARDISO iparm[59]), MUMPS BLR (ICNTL(35)), MPI, user orderings,
  a HostPencil route for the em FDFD sweeps (host_fdfd, fdfd_2d_fem: A(omega) rebuilt for factorize_host), deferred
  checks of mixed factors, pydiso >= 0.3 in the conda env (needs Python >= 3.12, the env is 3.11; 0.2.0 reaches the
  same parallel ordering through `_ordered_pardiso_class`: analysis cpu/wall ~5 on 16 threads, measured), PARDISO
  Hermitian types (4 / -4), remembered mixed rejections, a thread-count setting,
  calibrated memory / time
  constants and profile curves.

## explicit_kernels (`StepperConfig`; role 'stepper')
- Frozen settings bundle: precision 'single', 'double' or 'half_storage' (fp16 storage, float32 arithmetic;
  `storage_dtype` / `compute_dtype`). Readers: em_fdtd_ez / _hz (STEPPER_ROLE; physics/em sub file "Stepper
  precision"), em_fdtd_3d, elastic_waves, laminar_flow (HALF_STORAGE_STEPPER_ROLE). fp16 kernels use
  `core.half_storage.storage_preamble`; each engine keeps values O(1) by its own power-of-two storage scale.
- Backends ('gpu', 'cpu'): on the cpu the engines run the `core.kernels` twins; `backend_misfit` refuses half_storage
  there (no twin of PTX fp16 storage); double runs there.
