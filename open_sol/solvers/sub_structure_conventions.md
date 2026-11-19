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
Algorithms, constants and their measurements (several shifts and ShiftPencil, inertia counts, device / host
Krylov-Schur, warm starts, the restart cap, shift offsets): arpack_shift_invert.py header and docstrings; cuDSS
internals: cudss_factor.py header; MUMPS / PARDISO: section "host_direct"; nvmath / cuDSS API notes and measurements:
`nvmath_guide.md` (repo root). Rules across files:
- Tolerance floors (`tolerance_floor=`; grep `ARPACK_TOLERANCE_FLOOR`, each with its reason at the constant) only
  where nothing needs round-off eigenvalues: PML / CPML problems whose PML-mode clusters run into MAX_RESTARTS at
  machine precision, Rayleigh-polished solves. Everything else passes 0 (machine precision).
- 'auto' -> cuDSS only where a role sets `eigen_role(auto_factorizations=)` (grep it): a CudssFactorization has no
  transposed LU solve, no recycling and no inertia of an LU / complex factor, so only roles whose consumers all cope
  get it; the rest stay superlu (unchecked). A runtime resolved from 'auto' falls back to SuperLU for good on
  CUDSS_REFUSALS (`_fall_back`), unless the caller's `fallback_refusal` hook gives a reason (em_fdfd_3d tetrahedra:
  SuperLU would take minutes at that fill), which then raises with the remedy.
- cuDSS memory items (this role and sparse_direct): only through the cudss_factor helpers: per plan
  `operator_copy_bytes` + `solve_buffer_bytes` (kept), `solve_peak_bytes` as a transient (CUDSS_SOLVE_PHASE, or inside
  the phase of an own factor), refreshing callers `index_map_bytes` (run item) + `refresh_peak_bytes` (transient), the
  factor an external item from `factor_device_bytes` (a counted `FactorFill` wherever the mesh gives one). Refit the
  factor rule from cudss_permanent_bytes / eigen_permanent_bytes and *_lu_entries, never from *_factor_device_bytes
  (those include the pool copies the estimators book as items: counted twice).

## block_eigen (`BlockEigenSolver`; role 'eigen', opt-in)
Layout, iteration, mixed phases, stall rules, the dense path and the callback contract: block_eigen.py header,
`lowest`, `block_bytes`, `time_estimate`.
- Opt-in only: SolverType `opt_in_roles=('eigen',)`, offered by `definition_helpers.eigen_role(block_eigen=True)`
  (operator 'spd', precisions double / mixed); grids ('tensor',), so a triangle / tetrahedra physics misfits it. A new
  consumer (today: grep `block_eigen=True`): such a role, a dispatch on `isinstance(eigen_solver, BlockEigenSolver)`
  where it calls the eigen runtime (`ElasticSystem3D.lowest_modes`), block callbacks of one launch each, exactly the
  eigenpairs it keeps (no EXTRA_EIGENVALUES margin), validation of what LOBPCG cannot do (interior targets, bands, zero
  eigenvalues: the relative residual divides by |lambda|), memory (`block_bytes` + its own operator copies and
  preconditioner) and time items (product_ops, preconditioner_ops; setup_ops flags a caller-built multigrid).
- Not yet: warm starts across sweep values / assemblies, hard locking, a fused residual + norm kernel, dropping
  unconverged pairs (returned with a warning), FEM pencils (tetrahedra / triangles: SA-AMG of K + tau M), Maxwell
  (gradient-space projection + auxiliary-space preconditioner); the time recipe never prices a private multigrid build
  (`work.setup_ops` is only a flag).

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
