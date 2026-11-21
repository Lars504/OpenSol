# open_sol/solvers — conventions & gotchas

Root rules: structure_conventions.md "Solver roles".

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
  pcg (`DeviceCsrProduct`, the capturable gpu CSR product; the CPU CG numbers of definition_helpers' warning),
  device_matrices (`device_matrix`: the device copy of a host matrix that products read beside a factor; estimators
  `device_matrix_bytes`, `device_product_bytes`).
- Real stays real on the device: a cupyx CSR of a real matrix times a complex cupy operand casts the MATRIX to
  complex128 on every product (cusparse `_cast_common_type`), so a stored float64 copy silently costs a complex copy
  per call. Real matrices meeting complex vectors go through `device_matrix` (float64 view of the operand), never a
  bare `cupyx...csr_matrix(real) @ complex`; a complex128 upload of a real matrix (`astype(np.complex128)` before the
  upload) is the same waste kept for good.
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
- Precision: `solver_defaults.precision` also replaces a precision in a role's Automatic settings (e.g.
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
- Feature precisions: `solver_misfit` cannot see features, so a feature refusing a stepper precision declares
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
  `core.memory_estimate.diffusion_system_estimate`, `fem.diffusion.fem_system_estimate` and the mechanics
  estimators; a workspace change updates them.
- A caller re-assembling a CSR calls `release_operator()` BEFORE it builds or uploads the successor (DiffusionSystem
  `_release_assembly`, FEM `_factorize`, magnetic system_fem / KeptPreconditionerSolver), else the old operator, its
  Jacobi diagonal or mixed float32 copy stay bound beside the new one through the setup (estimates book one). It
  keeps the workspace and the own hierarchy (the lagged setup compares against the operator that hierarchy holds);
  solve raises until the next factorize.
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
- Memory constants (`hierarchy_bytes`, `setup_extra_bytes`, `tentative_bytes`, `kept_host_bytes`,
  `aggregation_host_bytes`; booked by `diffusion_system_estimate(multigrid=)` and `fem_system_estimate`) follow the
  hierarchy layout: a layout change updates them; the work counters of `_record_size` measure them (device arrays
  only). Setup seconds are machine-dependent: never pin them in a selftest.
- Device between setups: P, R, coarse operators, level vectors, the coarsest solver and the nonzero mask only. The
  tentative prolongators T stay on the host (`_host_tentatives`, rebuilt with the aggregates) and are uploaded per
  level build, freed before the Galerkin products; a setup drops the old levels and coarsest solver before building.
  The cached pattern takes the current operator's index arrays (a kept first-operator copy would pin its indices).
  Kept as is (speed / determinism): R = P^T stored explicitly and full (not lower-triangle) coarse operators, since
  the capturable CSR row kernels gather without atomics.
- Measured: 3D iterations to 1e-8 (20-35 at 40^3-60^3) exceed the ~12 of MULTIGRID_RESIDUAL_REDUCTION (3D run-time
  estimates run low: level-1 aggregates hold ~3 rows, most level-1 P rows unsmoothed). Rejected: threshold 0 below the
  finest level, |a_ij| strength, smoother weight, V(2,2), float64 hierarchy, a larger coarsest level.

## line_block (`LineBlockSolver`; SolverType 'line_block', roles 'shifted' and opt-in 'linear')
- One runtime serves several consumers: each passes `owner=` to set_operator / factorize_operators and re-binds only
  when `solver.owner` is another; any set_operator / factorize_operators / release invalidates the kept LineBlockFactors
  (`valid` False), so a holder tests `valid` before solving.
- Operators must be block-tridiagonal in the line order (`_checked_matrix` raises): a periodic / Bloch wrap goes
  inside the blocks (block-axis rule: physics/em/fdfd_line_block).
- Memory: consumers estimate `operator_bytes` (run item) + ONE `shift_bytes`: chunks grow only into free memory at run
  time, so min(n_shifts, MAX_CHUNK_SHIFTS) x shift_bytes over-books and refuses runs that fit. Kept factors
  (`factorize`, `factorize_operators`) hold three block stacks per shift: every estimate and sizing of one passes
  `kept=True` (batches: `batch_size`). A real K keeps its residual CSR float64 (`_times_complex` on the float64 view
  of the complex columns; `operator_bytes(complex_operator=False)` books it so); matrix terms stay complex128.
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

## banded (`BandedSolver`; role 'linear' of 1D models)
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
- `eigs_device(single_basis=True)` / `eigs(single_basis=True)` (complex64 basis on the device Krylov-Schur; host
  ARPACK ignores it) only on a tolerance >= SINGLE_BASIS_MIN_TOLERANCE (ValueError below, on every path, so a misuse
  fails at any size): its Arnoldi relation holds to fp32 x |theta_max| (SINGLE_BASIS_RELATION), while the Ritz test
  asks tolerance x |theta_i|, so the wanted pairs far from the shift are not certified by it. Every caller therefore
  has a float64 polish on a symmetric pencil: solve_resonances / fem_modes their own Rayleigh quotients (`refine`;
  estimators book it through `memory_estimates.polished_mixed_search`); `eigs_shifted(single_basis=)` (mechanics
  `ModalPencil._shift_solve`, node tolerance >= 1e-6; ValueError on a general pencil) through `polished_theta` (one
  float64 solve per far pair, counters eigen_single_basis_polished / _unpolished); fem.harmonic_resonances (complex
  symmetric LDL^T only) its own v^T K v / v^T M v. A new caller without such a polish keeps complex128 (em
  cross_section_modes says why); estimators read the same rule (`eigen_device_items`, the membrane estimate) through
  `arnoldi_basis_bytes(single_basis=True)`. One search keeps one basis dtype (a resume checks it).
- Device B of the shift-invert maps: `device_factor_matvec(factorization, B, symmetric=)` and
  `ShiftPencil.device_mass` store B through `device_matrix` (float64 when real, also a complex-typed B with zero
  imaginary parts; a diagonal B as its vector; symmetric: lower triangle + diagonal, probed). Estimators book
  `device_matrix_bytes(n, B entries, symmetric, real, diagonal)`, never a complex128 CSR of a real B. The symmetric
  storage applies L^T by cuSPARSE's transposed product (slower than an untransposed SpMV): fine beside a factor solve
  per application, not for a product-bound loop.
- ShiftPencil keeps a real term (K_i or M) float64 inside a complex pencil (DevicePencil term bytes 8, not 16, for
  `device_pencil_bytes(term_value_bytes=)`); a complex pencil whose terms and coefficients are all real factors real
  on the DevicePencil route (factorize_pencil judges the combined values), complex on the host route.
- 'auto' -> cuDSS only where a role sets `eigen_role(auto_factorizations=)` (grep it): a CudssFactorization has no
  transposed LU solve, no recycling and no inertia of an LU / complex factor, so only roles whose consumers all cope
  get it; the rest stay superlu (unchecked). A runtime resolved from 'auto' falls back to SuperLU for good on
  CUDSS_REFUSALS (`_fall_back`), unless the caller's `fallback_refusal` hook gives a reason (em_fdfd_3d tetrahedra:
  SuperLU would take minutes at that fill), which then raises with the remedy.
- cuDSS memory items (this role and sparse_direct): only through the cudss_factor helpers: per plan
  `operator_copy_bytes` + `solve_buffer_bytes` (kept), `solve_peak_bytes` as a transient (CUDSS_SOLVE_PHASE, or inside
  the phase of an own factor), refreshing callers `index_map_bytes` (run item) + `refresh_peak_bytes` (transient), the
  factor an external item from `factor_device_bytes` (a counted `FactorFill` wherever the mesh gives one). A mixed
  plan's factor-dtype copy shares the value-dtype copy's index arrays (`operator_copy_bytes` counts one index set; an
  estimator subtracting index arrays shared with its own maps subtracts one set, `fem.diffusion`). Refit the
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
Factor and solve maps, probes and mixed refinement, the memory guard, refactorisation and kept analyses, the PARDISO
thread rule, value refreshes, threads and the OpenBLAS clash: host_direct.py header; library API notes, PARDISO matrix
types and ordering: its adapter section (`_MumpsFactor`, `_PardisoFactor`). Rules across files:
- Kept analyses are scoped like the cuDSS orderings (`host_analyses_kept()`, entered by run_study beside
  `cudss_orderings_kept`; off for scripts and selftest checks), so a case never adopts another case's analysis.
- `quiet_openblas` covers only a factor's own calls and loops wrapped in it; BLAS work of a physics between two PARDISO
  calls stays outside it (OPENBLAS_NUM_THREADS=1, INSTALL.md). `limited_blas_threads` (every BLAS pool) is
  arpack_shift_invert's host_blas_limit; both are process-wide while held: vector-sized BLAS work only.
- Measured (2D FDFD, 160 000 unknowns, 4 threads, ~185 ms per PARDISO refactorisation): numeric factorisation ~50 %,
  equilibration + asymmetry check ~30 % (single-threaded; `symmetric_known` skips the check), pydiso's triu ~6 %, probe
  ~12 %. *_mixed halves the factor memory but costs time in 2D (a full-accuracy solve = 3 library solves + GMRES, the
  float32 factorisation only ~1.4 x faster): it pays for memory, or where the factorisation dominates (large 3D).
- Not yet: out-of-core (MUMPS ICNTL(22), PARDISO iparm[59]), MUMPS BLR, MPI, user orderings, a HostPencil route for
  the em FDFD sweeps (host_fdfd, fdfd_2d_fem rebuild A(omega) for factorize_host), PARDISO Hermitian types (4 / -4), a
  thread-count setting, calibrated memory / time constants and profile curves; pydiso >= 0.3 in the conda env (needs
  Python >= 3.12, the env is 3.11; 0.2.0 reaches the parallel ordering through `_ordered_pardiso_class`).

## explicit_kernels (`StepperConfig`; role 'stepper')
Precisions and backends: explicit_kernels.py header; the roles: `STEPPER_ROLE` / `HALF_STORAGE_STEPPER_ROLE`
(model.definition_helpers); fp16 storage and scales: core sub file "Half storage"; em_fdtd_ez / _hz double: physics/em
sub file "Stepper precision".
