# open_sol/solvers — conventions & gotchas

Design: `solver_plan.md` Part I. Root rules: structure_conventions.md "Solver roles".

## Layout and imports
- `roles.py` (SolverRole, SolverType, SolverRequest, ResolvedSolver, `solver_misfit` = the one fit rule) is pure Python
  and imports nothing from open_sol, so model.schema may import it; `__init__` imports only `roles`; model.schema
  re-exports SOLVER_ROLES, ResolvedSolver, SolverRole, SolverType, solver_misfit.
- Runtimes (`pcg.py` (+ `multigrid.py`, imported inside factorize), `line_block.py`, `sparse_direct.py`, `banded.py`,
  `arpack_shift_invert.py`, `explicit_kernels.py`) import numpy only at module level; cupy / scipy / nvmath /
  core.gpu_memory / core.backend / core.host_memory inside functions, kernels built on first use, so core.fv_diffusion,
  model.definition_helpers and GUI-thread checks (`scipy_available`, `size_issue`, `expected_back_end`) may import them.
  `cudss_factor.py` is the one cuDSS wrapper (sparse_direct imports it; arpack_shift_invert reaches it inside
  factorize_host; sparse_direct's cuDSS time recipe uses arpack's CUDSS_* fit inside functions). Shared helpers, never
  a local copy: `cudss_factor.csr_rows` / `canonical_csr` (numpy only, module level in sparse_direct, arpack,
  multigrid), `sparse_direct.cudss_precision`, `require_backend_operator`, `ExactSolveRecord` (the PCGSolver-protocol
  record of BandedSolver and DiffusionDirectSolver), `cudss_factor.gmres_refine` (the GMRES-IR loop of cuDSS mixed
  factors and of host_direct, array module xp), `sparse_direct.recycled_gmres` (GMRES on an earlier factor's solve),
  `cudss_factor.ProbedFactor` (base of CudssFactorization and HostDirectFactorization: symmetric / mixed_factor /
  factor_precision / can_refactorize, the probe vector and gate; host_direct also takes its `_FactorRejected`).
- `host_direct.py` (MUMPS via python-mumps, MKL PARDISO via pydiso) imports numpy and the cudss_factor helpers at module
  level (sparse_direct, arpack_shift_invert, definition_helpers and solvers.definition import it there), the wrappers
  (`mumps`, `pydiso.mkl_solver`), scipy, core and `sparse_direct.recycled_gmres` inside functions only (import
  contracts direct_wrappers_lazy / direct_wrappers_inside_functions). `HOST_DIRECT_BACK_ENDS` is the one list of the
  host names: every other tuple, choice label and text derives from it.
- `definition.py` (SolverTypes + `register`) <- model.schema, never cupy at module level; `get_registry()` imports it
  right after model.study, so `register_physics` can check each `PhysicsType.solver_roles` default_type.
- pcg.py holds the CPU numbers the definitions share (FLOAT32_TOLERANCE_FLOOR, CG_ITERATIONS_PER_AXIS_CELL,
  `auto_max_iterations`, `resolve_max_iterations`, `DTYPE_PRECISIONS` / `SOLVER_DTYPE_NAMES`); definition_helpers builds
  the CG iteration-limit warning from them. `resolve_max_iterations` on element meshes takes 2 x the longest window
  side / the mean element size (triangles sqrt(area / n), tetrahedra cbrt(V / n)) as the axis.
- Naming: `PhysicsSolver` (core.solver_base) is the physics runtime; objects here are named by role (PCGSolver is a
  LinearSolver).

## Model path (spec -> resolution -> build -> summary)
- Spec: `ModelSpec.solvers` [SolverSpec(name, type, settings)], `ModelSpec.solver_defaults` (backend 'auto' = 'gpu'
  when cupy is installed, else 'cpu': `schema.solver_backend`, api / CLI `backend` override), `PhysicsSpec.solvers`
  {role: node name}; a missing role = Automatic.
- Default precision (S9): `solver_defaults.precision` 'auto' (each solver's own) or a PRECISIONS key, applied by
  `schema.with_default_precision(spec, solver_type, settings, solver_role)` to settings without a 'precision' key when
  both the type and the role offer it (else unchanged, no error), per (physics, role) in resolved_solvers, validate and
  the GUI role combos. An explicit node precision overrides; GUI-added nodes start with the default written in, JSON
  nodes without one follow it. `schema.default_precision_ignored` = the validate info. Code resolving solver values
  outside resolved_solvers must go through the helper.
- Backends (S6): the run backend must be in `PhysicsType.backends` (default ('gpu',); see each definition; misfits are
  validate errors per physics, greyed out in Add physics: `schema.physics_type_misfit`) and fit every role. Automatic
  resolves per backend: `SolverRole.backend_defaults` / `SolverRole.automatic(backend, grid)` (register_physics checks
  those types too): CG_DIFFUSION_LINEAR_ROLE on the cpu = sparse_direct (host SuperLU via `DiffusionDirectSolver`;
  cusolver_qr refused) in 2D, multigrid pcg on a 3D tensor mesh. Every role must fit (`solver_misfit`:
  `SolverType.backends` + optional `backend_misfit(values, backend)`, Automatic roles too). Runtimes take `backend=` from
  `SolverRequest.backend` and work on that backend's arrays only (`core.backend.array_module` / `sparse_module`): an
  array of the other backend is a TypeError, never a silent transfer. On the cpu `check_memory` counts the device
  estimate against host RAM.
- Size rules: a sparse_direct diffusion 'linear' solver has `sparse_direct.diffusion_size_issue` (2D SPLU_*, 3D
  SPLU_3D_* on window voxels; remedy: a pcg node) at the physics via `definition_helpers.cg_solver_issues`;
  DiffusionDirectSolver refuses above SPLU_LIMIT_UNKNOWNS; its host bytes are host items of
  `diffusion_system_estimate(direct=)`. em_fdfd_3d's curl-curl eigen factor: `sparse_direct.vector_3d_size_issue`
  (SPLU_VECTOR_3D_*, first values to lower from measured factors; no iterative remedy) + `host_lu_bytes(n,
  dimensions=3)`.
- `schema.resolved_solvers(spec, registry, physics_spec, study, parameters)` is the one resolution rule (validate,
  `meshing.estimate_memory` -> `MemoryContext.solvers`, builder): only roles `used_in(study)`. A pcg preconditioner the
  node does not set resolves 'jacobi' at precision 'mixed' (so the Automatic multigrid never meets mixed); a
  sparse_direct back_end 'auto' becomes the role's `auto_back_ends[grid]` on the gpu with nvmath (sparse_direct
  section). Solver knobs never come from physics settings (CG keys on a physics fail validation as unknown settings).
  Estimators read `context.solvers` only (`definition_helpers.linear_solver_precision(context)`, `resolved_back_end`,
  `eigen_subspace_size`; heat's line_block items from `context.solvers['shifted']`); physics validate callbacks get
  `ValidationContext.solvers(physics_spec)` (None when unresolved: check_solvers reports; `validated_linear_precision`
  then assumes float64).
- Self-test specs needing a tight CG: `model.selftest_helpers.with_exact_cg(spec[, physics_names])` (node 'exact_cg',
  double, tolerance 1e-12, as 'linear'). Host direct nodes: `with_direct_node(spec, back_end, role='linear',
  physics_names=(), node_name=None)` (sparse_direct back_end, or for role 'eigen' an arpack_shift_invert factorization;
  node 'host_direct_<back_end>' / 'host_direct_eigen_<back_end>'), `available_host_direct()` (installed libraries) and
  `skip_without_host_direct()` (skip_result when none is installed).
- Shared roles (model.definition_helpers): `CG_LINEAR_ROLE` (pcg single / double, jacobi only: solid_mechanics),
  `CG_ELASTIC_3D_LINEAR_ROLE` (+ 'mixed', jacobi only: solid_mechanics_3d), `CG_DIFFUSION_LINEAR_ROLE` (+ 'mixed': heat,
  electric_currents, electrostatics, magnetostatics; Automatic on the gpu = pcg multigrid), `DIRECT_LINEAR_ROLE`
  (sparse_direct double, setting_choices back_end auto / scipy_splu / cusolver_qr + the host direct names:
  em_fdfd_nonlinear), `MAGNETIC_AC_LINEAR_ROLE` (auto / scipy_splu / cudss + the double host names), `FDFD_LINEAR_ROLE`
  (sparse_direct double / mixed, opt_in_types ('line_block',), `FDFD_DIRECT_BACK_ENDS` incl. cudss / cudss_mixed and the
  four host names: em_fdfd_ez / _hz), `CUDSS_LINEAR_ROLE` (sparse_direct double, same back ends: drift_diffusion,
  em_fdfd_bor, em_fdfd_conical), `DIFFUSION_DIRECT_BACK_ENDS` (+ the four host names: the CG / diffusion / pressure /
  membrane linear roles), `eigen_role(studies)` (arpack_shift_invert; factorization auto / superlu / cudss + the double
  host names, with mixed_factorization also cudss_mixed + the host *_mixed names), `STEPPER_ROLE` (explicit_kernels
  single / double: em_fdtd_ez / _hz),
  `HALF_STORAGE_STEPPER_ROLE` (+ 'half_storage': em_fdtd_3d, elastic_waves, laminar_flow). Stepper precision: estimators
  / validate by `resolved_stepper_precision(solvers)`, a create by `explicit_kernels.stepper_precision(ctx.solvers(name))`.
  `cg_solver_issues` gives the iteration-limit warning of an assigned pcg node (Automatic: none); the precision warning
  is the pcg SolverType validate at the node; type-specific CG advice lives in the pcg Param help texts.
- Opt-in roles: a SolverType fills its `opt_in_roles` only for a SolverRole whose `opt_in_types` names it, so adding a
  role to a shared type does not offer it to every physics with that role (line_block 'linear': em_fdfd_ez / _hz only).
  Likewise `SolverType.opt_in_settings` {setting: values} fit only a role whose `setting_choices[setting]` lists them
  (sparse_direct back_end cudss / cudss_mixed and the host direct names), so a new back end never reaches a physics
  that cannot hold its factor (em_modes / em_fdfd_bor 'shifted' misfit the host names: their eigen node's
  factorization is the knob).
  `SolverRole.setting_choices` is the generic per-role restriction of setting values (checked by `solver_misfit`).
- Private runtimes: a script constructor given a choice string or None builds its own runtime and reports its
  counters in the physics' `work_counters` (no role record); a build-injected one never does.
- Validate (`check_solvers`): backend, node type / settings / SolverType.validate at ('solver', name), unused node =
  warning; per physics: unknown role / node = error, a role this study does not use = warning, `solver_misfit` = error
  naming the fitting choices.
- Feature precisions (S3.3): `FeatureType.precisions` (default every PRECISIONS key); `solver_misfit` cannot see
  features, so `_check_feature_precisions` (only when the physics' role checks passed) errors at the feature on
  `schema.feature_precision_misfit` (the resolved 'stepper' precision): em_fdtd_3d plane wave / mode source / mode
  monitor and resonance_monitor refuse half_storage, 2D nonlinear_medium / gain_medium refuse double. Physics-level refusals (em3d poles and
  frequency_domain, laminar_flow steady march, elastic_waves piezo) stay in the physics validate. Shown via
  `schema.restricted_feature_precisions` / `feature_precision_note`: a new restriction needs no describe or GUI edit.
- Self-test variants (S3.6): `SelfTestCase.variants` repeat a case on another backend / precision with its own
  tolerance; api `_variant_spec` gives every affected (physics, role) its own node `<variant>_<physics>_<role>`, so a
  case's node names must not collide with that pattern. A 'cpu' variant of a gpu case is its CPU coverage: without a
  usable GPU it still runs and judges the case (`_judge_by_variants`).
- `solver_default_precision` (selftests.py, CPU) pins the S9 resolution table and the multigrid resolution: a change of
  a role's precisions or Automatic type updates its `_PRECISION_CASES` / `_MULTIGRID_CASES` row.
- Build: its own stage after 'materials', one `MemoryLedger.track("solver '<physics>.<role>'")` per solver (never inside
  a physics create: the ledger does not nest); `BuildContext.solver_records`, `ctx.solvers(name)` -> {role: runtime}.
  `Session.close` clears the records.
- Summary (`study._solver_records`, summary.json 'solvers'): {physics: {roles: {role: {type, solver, backend,
  precision, settings (resolved), counters}}, counters (the physics' own work_counters)}}; check.json the same without
  counters. It records a machine-dependent Automatic (backend auto, cuDSS back end). A physics never repeats its role
  counters.
- Run-time recipes: every SolverType sets `time_estimate(values, SolverWork)` (explicit_kernels empty: the physics
  counts its steps); items' `counter` / `seconds_counter` are its own work_counters; timers only at existing syncs.
  Iteration counts are a priori, never fitted (a time_check deviation is a lead on the solver or model): multigrid ln(1
  / tol) / ln(1 / MULTIGRID_RESIDUAL_REDUCTION), Jacobi sqrt(kappa) ln(2 / tol) / 2 of `SolverWork.jacobi_condition`
  (0: CG_ITERATIONS_PER_AXIS_CELL x longest_axis), both capped by `distinct_modes` and rounded up to the check block;
  warm solves WARM_START_ITERATION_FRACTION of a cold one (a guess). A new counter or timer is added to work_counters
  and named by the recipe in the same edit.

## pcg (`PCGSolver`)
- `factorize(operator, preconditioner=None, n_unknowns=None, single_operator=None, single_preconditioner=None)` (single_*
  only for a mixed pcg; sparse_direct / banded factorize do not take them): cupyx / scipy CSR or matrix-free
  `apply(vector, out)` (the CSR product writes one owned vector that the next product overwrites); preconditioner None
  = Jacobi from the diagonal, a flat inverse diagonal (fused, z never stored on the gpu) or `precondition(residual,
  out)` (block Jacobi). gpu dots are multi-block RawKernels (deterministic per size and device), so gpu inputs must be
  C-contiguous (solve raises).
- cpu: numpy twins of the fused kernels (`_HOST_KERNELS`, same signatures: a kernel change changes its twin),
  allocation-free (BLAS axpy, scipy's private `_sparsetools.csr_matvec` into the workspace product, falling back to
  `dot` when missing or dtypes differ; Jacobi z in the product vector). Workspace reallocated only when size, dtype or
  kinds change; `workspace_bytes` is what the physics' estimator counts (SolverType estimate empty). The solution
  vector stays physics-owned.
- `solve(rhs, solution, reference)`: warm start, in place; flat arrays of the solver dtype (ValueError otherwise); b = 0
  returns 0. The residual is read every `check_interval(preconditioner, backend)` iterations (gpu 16, multigrid 2; cpu
  4, multigrid 1), so cg_iterations round up to that block (the estimate rounds too, by `SolverWork.backend`). A
  NaN / inf residual (start or block read; mixed: the outer residual) stops at once, unconverged, with a 'non-finite
  residual' last_warning (`_non_finite`), never iterating to max_iterations.
- Stop reference: 'rhs' (tol ||b||) for steady solves (warm re-solves cheap); 'initial_residual' (tol ||b - A x0||,
  floored at ROUNDOFF_FLOOR_EPSILONS eps ||b||) for implicit steps: against ||b|| the C/dt u_old term dominates and CG
  stopped before the step moved. The caller picks it per solve; `last_residual` is relative to it.
- Precision names are solver ones ('single' / 'double' / 'mixed'); `SOLVER_DTYPE_NAMES` maps them to the assembled
  dtype ('mixed' -> 'float64').
- 'mixed' (`_iterate_mixed`): CSR + Jacobi (float32 copy of the values sharing the float64 indices + float32 inverse
  diagonal; a block preconditioner with a CSR: TypeError) or matrix-free (float64 apply + the caller's float32
  `single_operator` / `single_preconditioner` on its own float32 copy; TypeError without both). Each outer step scales
  r = b - A x to unit norm before the float32 CG (physical residuals underflow float32 dots at 1e-12), x += ||r|| d;
  stops at the float64 reference, the iteration limit (inner summed), MIXED_MAX_REFINEMENTS or a step shrinking ||r|| by
  less than MIXED_STALL_RATIO. `workspace_bytes(..., 'mixed')` / `mixed_extra_bytes` count the solver's copies; a
  matrix-free caller counts its own float32 operator.
- Users: the 'linear' role of heat, electric_currents, electrostatics, magnetostatics (`DiffusionSystem(linear_solver=)`)
  and solid_mechanics(_3d) (`ElasticSystemBase(linear_solver=)`: block-Jacobi, 'rhs'; Automatic and max_iterations 0
  get twice the auto limit, `definition_common.solid_linear_solver`; 3D 'mixed' via ElasticSystem3D's float32
  callbacks). Both take the injected solver's precision and set its label and compute_residual. A workspace change here
  updates `memory_estimate.diffusion_system_estimate` and the mechanics estimators.

## multigrid (`MultigridPreconditioner`, pcg setting preconditioner 'multigrid')
- Smoothed-aggregation AMG, one symmetric V(1,1)-cycle per application (l1-Jacobi, 1 / a_ii on rows without
  off-diagonals: SPD for any SPD A); P = (I - W_F A_F) T with the filtered A_F (`_filtered_rows`: weak entries lumped
  into the diagonal; built by array masks, since cupyx `eliminate_zeros` failed with INSUFFICIENT_RESOURCES); Galerkin
  R A P with the full A; coarsest <= COARSEST_DENSE_LIMIT by a host eigh pseudo-inverse (raises on a non-finite
  inverse). Rounding only weakens the preconditioner (SPD holds for any symmetric A_c).
- Products run in row blocks of ~PRODUCT_BLOCK_PRODUCTS (`_row_blocks`): one SpGEMM over a 5e6-row 3D operator needs a
  workspace larger than an 8 GB card, so never multiply the full fine operator on the backend. The block sum merges
  only equal block counts (binary counter): a running sum recopies A_c per block. A cupyx / scipy CSR row slice copies
  the matrix even over the full range: block code takes `_rows`, never `matrix[start:stop]`.
- Hierarchy (T, P, R, coarse operators / vectors) is HIERARCHY_DTYPE float32 at every precision; a float64 fine level
  casts its residual before R. `precondition` scales the residual by a power of two to max |r| in [0.5, 1) and back
  (exact, no sync): unscaled ~1e-40 residuals underflowed the float32 levels and CG ran to its limit.
- PCGSolver builds it in `factorize`: assembled scalar CSR only (TypeError for matrix-free / callback preconditioners;
  the passed inverse diagonal is ignored), not with 'mixed' (ValueError; validate errors at the node). Automatic for
  `CG_DIFFUSION_LINEAR_ROLE` (gpu, and 3D tensor meshes on the cpu; jacobi at mixed); a pcg node without the setting is
  jacobi (`with_exact_cg` too).
- Aggregation (host numpy): strength -a_ij >= STRENGTH_THRESHOLD sqrt(a_ii a_jj) (negative off-diagonals only) plus each
  row's largest negative coupling (floor; without it a 3D level-1 operator left ~98 % of rows out of P), symmetrised;
  distance-2 independent-set roots (Luby, fixed seed: deterministic); rows without negative couplings (held / pinned
  identity rows) stay out of P. The prolongator filtering keeps the plain threshold rule.
- Reuse: one host transfer of each level's CSR only when the fine pattern changed (`_pattern_matches`: identity, then
  array_equal); a value-only factorize (re-assembly, Newton / B-H iterations, refresh_diagonal) rebuilds P, R and coarse
  operators on the backend from the cached aggregates; rows whose strong links turned weak get prolongator weight 0
  (P = T there), which keeps value-only setups converging like fresh aggregation (no re-aggregation needed).
- Counters (merged into pcg's): multigrid_setups / aggregations / levels, the last setup's size
  (`_record_size`: calibrate HIERARCHY_ENTRIES_PER_UNKNOWN / SETUP_ENTRIES_PER_UNKNOWN from them) and seconds
  (machine-dependent: never pin them in a selftest). Memory: `hierarchy_bytes` / `setup_extra_bytes` /
  `aggregation_host_bytes`, counted by `diffusion_system_estimate(multigrid=)`: a hierarchy layout change updates those
  constants. The product workspace is one block's (PRODUCT_WORKSPACE_BYTES) capped by a small operator's whole A @ P
  (a fixed block booked 256 MiB on 1e3-cell models). `setup_extra_bytes` books the largest of the fine level's
  three steps on a row block, never their sum: `_filtered_rows` frees its temporaries on return (only the filtered
  block lives into A_F T), and the Galerkin product starts after the blocks are stacked (the sum read the 2D
  adaptation_tensor_strip_heater build peak 12 % high at 4e4 .. 6.4e5 cells and 22 % at 2.56e6). Open:
  HIERARCHY_ENTRIES_PER_UNKNOWN = 12 is the 3D 7-point figure (11.96); the 2D 5-point strip heater holds 8.47
  (complexity 1.355: P rows ~2.8 in both, the coarse operators differ), kept ~7 % high there; a 3D setup's A @ P
  result (more than SETUP_ENTRIES_PER_UNKNOWN entries per row) is not counted apart.
- Level vectors are preallocated; on the gpu each V-cycle allocates its SpMV products, on the cpu `_product` writes into
  level vectors by `pcg.host_csr_matvec` (falls back to `dot` when dtypes differ).
- Measured: 3D iterations to 1e-8 (20-35 at 40^3-60^3) exceed the ~12 of MULTIGRID_RESIDUAL_REDUCTION (3D run-time
  estimates run low: level-1 aggregates hold ~3 rows, most level-1 P rows unsmoothed). Rejected: threshold 0 below the
  finest level, |a_ij| strength, smoother weight, V(2,2), float64 hierarchy, a larger coarsest level.

## line_block (`LineBlockSolver`, Part II; SolverType 'line_block', roles 'shifted' and opt-in 'linear')
- Family K + sum_i s_i M_i + sum_j t_j N_j: diagonal M_i (thermal noise: s = (j omega, q^2)) and sparse N_j with K's line
  coupling (em_fdfd_bor: t = (m, m^2)); coefficient columns = the M_i, then the N_j. One rhs set shared by all shifts.
  `set_operator(host K, LineBlockLayout, terms, owner=None, matrix_terms=())` keeps device state and `owner`, so
  consumers sharing one role solver re-bind only when another operator holds it; `release()` drops both, counters stay.
- `factorize(coefficients)` -> `LineBlockFactor` (size the batch with `chunk_size(n, 1)`): repeated `solve(rhs,
  index)`; `valid` turns False after the solver's next set_operator / release; `release()` frees it.
- `LineBlockLayout.from_grid_cells`: blocks = occupied grid lines (empty lines drop out), block axis = the cheaper
  n_blocks b^3; a coupling more than one line apart raises (no wraps along the line axis).
- `solve_shifted` yields chunks (start, host solutions, residuals, accepted): rejected shifts (above ACCEPT_TOLERANCE or
  non-finite) are the consumer's to re-solve with its reference solver. Chunk = min(n_shifts, MAX_CHUNK_SHIFTS,
  MEMORY_FRACTION of budget - pool used); consumers estimate `operator_bytes` + chunk x `shift_bytes` as run items.
- cpu: the same code on numpy / scipy, inputs and solutions host numpy on both backends; measured 7-9x slower per solve
  than host SuperLU, so the cpu path is for parity, not speed.
- Refinement keeps the better iterate per shift and stops when no open shift improves by STALL_RATIO; `debug=True`
  syncs and records `batch_log`.
- Role 'linear': em_fdfd_ez / _hz only (em_fdfd_bor and em_modes reach line_block through 'shifted'): one kept factor
  (`factorize(zeros((1, 0)))`) per assembled system, layout rules in physics/em/fdfd_line_block.
- `factorize_operators(operators, layout, owner)`: a kept factor of whole operators, one per shift (Helmholtz is not
  affine in omega); per-shift caches live in the factor's chunk (`_ShiftOperators`: `_coupling` / `_residual` branch on
  `chunk.operators`); the call releases the bound operator (solve_shifted needs a new set_operator). Size by
  `batch_size(n, layout, n_entries)`; consumers cap at `fdfd_line_block.MAX_BATCH_WAVELENGTHS` and keep factors only
  while a material serial is unchanged.
- `LineBlockFactorization(factor, index, label)`: one shift as a sparse_direct Factorization stand-in (release = the
  whole batch); solve_host raises `RejectedLineBlockSolve` above the accept tolerance, and every shift-invert consumer
  (em_fdfd_ez / _hz eigenfrequency, em_modes, em_fdfd_bor) catches it and redoes that eigen-solve on host SuperLU.
- Launch-bound on small blocks (Python loops over blocks, ~80 us gemm floor per call on an RTX 3050). The time recipe
  prices each block product per call (FACTOR_DENSE_CALLS_PER_BLOCK, SWEEP_PRODUCTS_PER_BLOCK, *_LAUNCHES_PER_BLOCK,
  counted from the code: a loop change updates them).
- Not yet: transposed solves (FDFD adjoint uses host SuperLU), per-shift rhs, BATCHED_INVERSE_MAX_BLOCK crossover, a
  device ARPACK loop, a fused one-launch sweep kernel.

## sparse_direct (`SparseDirectSolver`; roles 'linear' and 'shifted')
- Module: Factorization (solve / solve_host / solve_preconditioned GMRES on an earlier factor), `factorize` (device CSR),
  `factorize_host`, size rules, `resolve_back_end` (raises) and `expected_back_end` (CPU, never raises). The cusolver_qr
  workspace lives outside the cupy pool: the size rules are the only guard.
- Orderings: SuperLU MMD_AT_PLUS_A by default; `factorize_host(ordering=)` factors A[p][:, p] with NATURAL and maps every
  solve back through `Factorization._lu_solve` (never `_host_lu.solve` directly). `grid_nested_dissection(shape)` (low
  fill only for +-1 cell stencils): em_fdfd_3d and DiffusionDirectSolver on a 3D tensor window (`grid_shape`). SuperLU
  ordering reuse: `SuperLUOrdering` (one per runtime) keeps the last pattern's elimination order; a later factorisation
  of the same indptr / indices pre-permutes and skips MMD (counters reused_orderings / eigen_reused_orderings).
- The runtime keeps no factor: consumers hold their own (one runtime serves the NLFD channels or thermal noise).
  `back_end` (auto / scipy_splu / cusolver_qr / cudss / cudss_mixed / mumps / mumps_mixed / pardiso / pardiso_mixed:
  LINEAR_SOLVER_CHOICES, the order solvers.definition's labels follow) resolved on first use; factorize / factorize_host /
  `solve_recycled` (device) / `solve_recycled_host` (em_fdfd_bor, magnetic AC circuit rows) are counted. The shifted
  role, magnetic_fields_ac and em_fdfd_bor always factorise on the host (validation refuses cusolver_qr). cpu backend:
  scipy_splu or a host direct back end, `host_arrays` factors numpy in / out.
- cuDSS back ends ('cudss' double, 'cudss_mixed' float32 / complex64 factor + float64 refinement to 1e-12): opt-in per
  role (Model path "Opt-in roles"): the SPD roles of `DIFFUSION_DIRECT_BACK_ENDS` (via DiffusionDirectSolver),
  FDFD_LINEAR_ROLE (LDL^T, counter cudss_symmetric_factors) and CUDSS_LINEAR_ROLE (drift_diffusion real LU,
  BOR / conical LDL^T of A / D A); DIRECT_LINEAR_ROLE and every 'shifted' role misfit them until their physics adopts
  them. A role opting in needs consumers holding a CudssFactorization on that grid (no transposed LU solve, not
  recyclable, no SuperLU inertia). GPU only (`backend_misfit`); without nvmath validate errors and `resolve_back_end`
  raises. `size_issue` returns None for them; cuDSS's own peak estimate after planning is the guard and memory
  estimators count the factor as an external item (`linear_solver_direct` '', `linear_solver_cudss(solvers)`;
  `cg_solver_issues` skips the SuperLU size rule). `definition_helpers.linear_solver_direct(context)` returns '' or the
  resolved non-cuDSS back end name (scipy_splu, a host direct name; truthy as before), which estimators pass on as
  `direct=` (`diffusion_system_estimate` books a host direct name by host_direct.factor_host_bytes).
- Back-end groups (module constants, never local copies): SYMMETRIC_OPTION_BACK_ENDS (cuDSS + host direct: factorize
  honours symmetric / symmetrizer / row_scale / previous), MIXED_BACK_ENDS (cudss_mixed + the host *_mixed: float32 /
  complex64 factor; validation rules about float32 accuracy use it), HOST_FACTOR_BACK_ENDS (scipy_splu + host direct:
  host memory, the SuperLU size rules); `host_size_back_end(name)` maps superlu / auto / None to scipy_splu for
  size_issue.
- Host direct back ends (mumps / mumps_mixed / pardiso / pardiso_mixed; section "host_direct"): opt-in per role (never
  'auto'), no backend misfit (host factors on both backends: factorize pulls a cupyx operator to the host once and
  returns cupy solutions, a scipy one numpy: host_arrays), a missing wrapper is a validate error and `resolve_back_end`
  raises `host_direct.missing_text`. factorize / factorize_host (+ symmetric_known) / DiffusionDirectSolver
  (factorize, factorize_host_operator, factorize_device_values through a host copy; real SPD request: PARDISO
  Cholesky, MUMPS LDL^T; solves on the host) return a HostDirectFactorization: transposed solves, recyclable
  (solve_recycled*), inertia, previous= refactorises on the kept analysis. Size rules: the SuperLU ones (size_issue /
  diffusion_size_issue(back_end=) with library-named texts; uncalibrated, LDL^T / nested dissection usually allow
  more) plus the library's own estimate after the analysis (HOST_FRACTION of the available RAM); refusals
  (HOST_DIRECT_REFUSALS) raise, no fallback. `fall_back`, `device_pencil`, `factorize_pencil` and
  `factorize_device_csr` stay cuDSS / auto only. Counters (host direct runtimes): host_direct_analyses (fallback
  re-analyses included), host_direct_reused_analyses, host_direct_peak_bytes (the library estimate), host_direct_lu_entries,
  host_direct_symmetric_factors, host_direct_refinement_steps, host_direct_single_residual, host_direct_double_fallbacks,
  host_direct_threads; host_factor_bytes = the library's factor bytes. Time recipe: the SuperLU items (labels named by
  the library) x HOST_DIRECT_FACTOR_TIME / HOST_DIRECT_SOLVE_TIME, mixed solves x HOST_DIRECT_MIXED_FULL_SOLVES (all
  uncalibrated, no profile curves yet).
- 'auto' -> cuDSS only through `SolverRole.auto_back_ends` {grid: back end}: `schema.auto_back_end_values` (inside
  resolved_solvers: validate, estimators, build and summary see the concrete back end) on the gpu with nvmath
  (find_spec); role-less `expected_back_end` / `resolve_back_end` map 'auto' to SuperLU. Opted in: pressure_acoustics
  (triangles, tetrahedra) and membrane (triangles) 'linear' cudss_mixed; CG_ELASTIC_3D_LINEAR_ROLE (tetrahedra)
  cudss_mixed; cudss for FDFD_LINEAR_ROLE / CUDSS_LINEAR_ROLE (tensor, triangles), CG_LINEAR_ROLE (triangles),
  CG_DIFFUSION_LINEAR_ROLE (triangles, tetrahedra).
  Self-tests that used Automatic as a SuperLU reference pin a scipy_splu node. Values carry `AUTO_BACK_END_KEY`
  ('back_end_from_auto'); the runtime gets `fallback_back_end='scipy_splu'`, and a cuDSS refusal
  (`cudss_factor.CUDSS_REFUSALS`: zero pivot, accuracy, `CudssMemoryError`) within SPLU_LIMIT_UNKNOWNS switches it to
  SuperLU for good (`fall_back`: RuntimeWarning, counter auto_fallbacks) and redoes the factorisation there (every entry
  point, incl. DiffusionDirectSolver's and fem.harmonic's deferred check at solve). An explicit cuDSS node raises. The em
  FDFD / BOR / conical consumers free their plan-kept cuDSS factor once on SuperLU.
- Entry points `factorize(..., previous=, symmetric=, symmetrizer=)` (device CSR, pulled to the host once), 
  `factorize_host(matrix, real, symmetric, positive_definite, symmetrizer, row_scale, previous)` (SuperLU ignores the
  cuDSS options), module `cudss_factorization(...)`: all return a `cudss_factor.CudssFactorization`; `transpose=True`
  only with a symmetric factor (x = D (D A)^-1 b for a symmetrizer D), else NotImplementedError; a real factor (also a
  real SuperLU `solve_host`) takes a complex rhs as two real solves, never a cast; `recyclable` False (decided: a
  value-only refactorisation skips the plan and is exact).
- Plan reuse: pass the factor to replace as `previous=`; same options = refactorised in place (values written into the
  planned device CSR, never nvmath `reset_operands(a=)`, which drops the plan for device operands); a new pattern
  re-plans; other options release it first. A Cholesky is planned only for a double factor (`_requested_kind`): a mixed
  positive_definite request plans float32 LDL^T (float32 Cholesky hung > 175 s), its double fallback is the double
  Cholesky.
- DiffusionDirectSolver on cuDSS: `factorize_device_csr(operator, real=True, positive_definite=True, previous=)`
  (Cholesky, LDL^T after a failed one), device solves; `release_factor()` before another large factorisation.
  `factorize_host_operator(host_csr, symmetric_known=)` (FEM Newton tangents): symmetric_known = a canonical symmetric
  CSR (values copied only, index arrays shared and never modified, asymmetry check skipped); never pass it for an
  operator that may be unsorted, duplicated or unsymmetric. `factorize_device_values(values, indptr, indices)` (cupy
  values on the HOST canonical pattern) -> `refactorize_device_values` / `from_device_values`: equilibration, lower take
  and refresh or plan on the device by index maps built once per plan (`_device_maps`), no host value copy. Either
  symmetry vouched (pencils, device tangents) or `check_symmetry=True` (device transpose map), which also serves
  `refactorize` of a host operator on the planned pattern without symmetrizer (values uploaded only) and
  `SparseDirectSolver.factorize_device_csr` (non-canonical cupyx CSR canonicalised in a copy). A symmetrizer factor
  takes the device path too (`symmetrizer=`: rows x D before R, exactly when the factor was built with one): module
  `factorize` of a cupyx operator with a valid `previous` of the same options refreshes it there
  (`_refreshed_on_device`; the 2D FDFD per-wavelength route, Sp3), only a first factor goes through the host copy;
  `row_scale=` replaces R = |diag|^-1/2.
- Device pencils (sweeps): `SparseDirectSolver.device_pencil(indptr, indices, term_values, symmetric,
  symmetric_known=False)` -> `DevicePencil` (None off cuDSS / on the cpu: the caller keeps factorize_host); terms
  uploaded once, symmetry checked once unless vouched (max |X - X^T| <= SYMMETRY_TOLERANCE max |X| per term). A
  symmetric pencil stores lower triangles only (`lower_stored`; `full_term(k)` returns a new array per call, its take
  built once; `combined(c)` in the stored layout). `factorize_pencil(pencil, coefficients, row_scale=, previous=,
  extra=)` combines on the device and refreshes `previous` (refactorize_device_lower / refactorize_device_values with
  defer_check); the pattern is matched by identity of the pencil's indptr / indices first: callers never modify those
  arrays in place. `extra=(pencil.slots(rows, columns), values)` adds a small per-frequency block (port DtN) at
  stored-layout slots (lower entries only for a lower-stored pencil, symmetry vouched by the caller; a None slot =
  outside the pattern: keep factorize_host). `device_pencil_bytes(entries, term_value_bytes, symmetric)`; users
  (fem.harmonic, mechanics frequency_response.HarmonicSystem) book it + factor maps + transients as a run item. The
  operator norm of a device refresh is lazy.
- defer_check: a factor on a reused plan whose earlier probe passed skips its probe; its first (full-accuracy) solve is
  the check, so probe_residual / single_residual / refinement_steps counters lag one factor. A double miss turns on
  cuDSS refinement and solves again; a second miss releases the factor and raises CudssAccuracyError from the solve
  (inertia-only factors are never judged: count_below passes defer_check=False). A mixed miss re-plans double in place from the device E (`fallback_listener` ->
  cudss_plans / cudss_reused_orderings / cudss_double_fallbacks): callers translate CudssAccuracyError at solve /
  solve_host too (fem.harmonic.solve). Fixed-step solves (refinement_steps=) leave a pending check pending.
- solid_mechanics_3d's 'linear' runtime on tetrahedra also factorises the piezo coupled operator (real symmetric
  indefinite) and the complex symmetric frequency-response pencil, although the role's operator class is 'spd'.
- Counters (cuDSS only): cudss_plans (incl. fallback re-plans of one call: `_counted` adds `last_plans` /
  `last_reused_orderings` / `last_plans_seconds`), cudss_reused_orderings, cudss_plan_seconds,
  cudss_factor_device_bytes, cudss_refinement_steps, cudss_single_residual; factor_seconds includes plans and probes.
- Ordering reuse (`cudss_factor` header): a plan keeps its row permutation + nested-dissection partition tree per
  (ordering family, n, nnz, pattern digest) and a later plan of that key reuses both (USER_PERM +
  USER_ND_PARTITION_TREE; ~10-20x faster planning, same fill). `ORDERING_FAMILIES`: Cholesky and LDL^T share one,
  real and complex alike; LU its own. Never the permutation alone (same fill, 10x slower factorisation). Within a factor
  object always; across objects only while `remember_cudss_orderings` is on: `model.study.run_study`
  (`cudss_orderings_kept()` per study) and the GUI worker; off for selftest checks and scripts, so a case never plans on
  another case's ordering. nvmath has no public setter (DirectSolver.handle / data_ptr): any error plans from scratch
  (logged at INFO).
- Time recipe `_cudss_time_items`: one plan (a work marked `ordering_kept`, read by getattr, at
  CUDSS_KEPT_ORDERING_PLAN_FRACTION) + factorisations + solves (+ the deferred check of device-value refreshes); real
  operators scaled by CUDSS_REAL_BYTES / CUDSS_REAL_FLOPS (uncalibrated). 2D: `cudss_2d_ops` (also arpack's) on the
  profile's cudss curves (core sub file) at work.factor_scale / solve_scale = the caller's fill over that curve, never the
  SuperLU fill unless nothing better is known; SolverWork.device_values selects the device refresh curve. The profile's
  code hash does not cover solvers/: re-run `python -m open_sol profile` after changes to cudss_factor / sparse_direct (a
  profile without the '_device' curves prices device refreshes 4.3x high). 3D: arpack's CUDSS_* fit, mixed solves x
  CUDSS_MIXED_FULL_SOLVES.
- Self-tests (`selftests.py`): solver_pcg_cpu_gpu, solver_sparse_direct_cpu_gpu, solver_line_block_cpu_gpu (both
  backends vs scipy on synthetic operators; gpu half skipped by `_backends()` without a GPU); solver_sparse_direct_cudss
  (gpu; its role / misfit part is CPU and judged without nvmath); solver_cudss_device_pencil (gpu);
  fem_harmonic_cudss_device_pencil (heavy); solver_host_direct_cpu_gpu and solver_arpack_host_direct (each installed
  library, both precisions; skipped without python-mumps / pydiso, the role / misfit / resolution part of the first
  always judged). A change to `cudss_factor.gmres_refine` is checked on the GPU by solver_arpack_cudss,
  solver_sparse_direct_cudss and fem_harmonic_cudss_device_pencil, on the host by solver_host_direct_cpu_gpu.
- Validation: the SolverType validate reports scipy / nvmath availability at the node; for Automatic the physics checks
  report it (`definition_fdfd.direct_solver_issues`, also nlfd's) plus the size rule.

## banded (`BandedSolver`; role 'linear' of 1D models, symmetry 2.20)
- Grid kinds `roles.GRID_KINDS`: 'tensor', 'line' (1D model), 'tensor_3d', 'triangles', 'tetrahedra';
  `schema.solver_grid(spec, physics_spec)` per physics. 'line' and 'tensor_3d' are tensor meshes
  (`roles.TENSOR_GRID_KINDS`: code testing "tensor" tests that set); 'tensor_3d' exists only so grid_defaults can differ
  from 2D. Callers of `automatic` / `solver_misfit` pass the grid (validate, resolved_solvers, api `_variant_spec`, GUI
  combos and 'In use' list).
- `SolverRole.grid_defaults` {grid: (type, settings)} wins over backend_defaults: CG_DIFFUSION_LINEAR_ROLE 'line' ->
  banded, 'tensor_3d' -> pcg multigrid on both backends (cpu: 7-130x faster than 3D SuperLU); CG_DIFFUSION_LINEAR_ROLE /
  CG_LINEAR_ROLE 'triangles' and CG_DIFFUSION_LINEAR_ROLE / CG_ELASTIC_3D_LINEAR_ROLE 'tetrahedra' -> sparse_direct.
  pcg, sparse_direct and arpack_shift_invert declare grids ('tensor', 'triangles', 'tetrahedra'); banded ('line',) only.
  pcg on tetrahedra is an opt-in node of the scalar FEM systems; solid_mechanics_3d on tetrahedra refuses it at create
  (sparse_direct only; its validation does not catch it yet).
- The runtime is the PCGSolver protocol itself: `factorize` pulls the CSR to the host, refuses a bandwidth above
  `max_bandwidth` (8, before the band array exists), an unsymmetric operator (`banded.SYMMETRY_TOLERANCE`: only the
  upper triangle is read) and a non-positive-definite one (ValueError), keeps the `cholesky_banded` factor. Precision
  'double'; host O(n) memory, not estimated. Not yet: per-order (m / l / q) batching, complex operators.

## arpack_shift_invert (`ArpackShiftInvertSolver`; role 'eigen')
- `eigs(matvec, n, k, v0, tolerance_floor, max_restarts, default_basis)` -> (theta, vectors, applications, converged):
  tolerance = max(node, floor) (0 = machine precision); subspace_size 0 = the caller's default_basis
  (`arnoldi_basis_size` for estimators). Time recipe: `SolverWork.rhs` = triangular solves per application,
  `applications_per_basis` (0: APPLICATIONS_PER_BASIS_GUESS 2.5).
- One uninterruptible host call (a GUI stop waits): eigs / eigsh_shifted / eigsh_inverse cap restarts at MAX_RESTARTS
  (300; scipy's 10 n let a shift on a resonance restart for minutes) and return the converged part with converged False
  (callers warn); bor_ports / slab_modes' direct scipy calls take the same cap.
- Tolerance floors only where nothing needs round-off eigenvalues: em_modes / fem_modes, FDFD / BOR eigenfrequency and
  mechanics keep machine precision, except cross-section solves dropping PML modes on a CPML cross-section (em_modes,
  mode ports, bends; not EME): 1e-12 (`cross_section_modes.PML_ARPACK_TOLERANCE_FLOOR`; PML-mode clusters hit
  MAX_RESTARTS at machine precision); em_fdfd_3d with a cpml face 1e-10 (+ a 5 k basis, ARNOLDI_BASIS_PER_WANTED);
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
  also for its host SuperLU via `core.memory_estimate.host_counted_lu_bytes`) x the factor value bytes + the kept matrix copy +
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
  by the float64 Rayleigh quotient (stationary for the complex symmetric pencil; `info['rayleigh_change']`); without a
  symmetric D A (Bloch) it refines every solve to 1e-12. Measured on the Mie eighth sphere: factorise 12x faster, half
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
  = 0 -> NaN). Not for a Hermitian pencil (eigsh_shifted keeps M-orthogonality).
- 'auto': the role's `auto_back_ends` (`eigen_role(auto_factorizations=)`) on the gpu with nvmath, else superlu (the
  runtime maps an unresolved auto to superlu). A runtime resolved from auto falls back to SuperLU for good on
  CUDSS_REFUSALS (factorize_host, the ShiftPencil device path, the deferred check inside eigsh_shifted / eigs_shifted;
  counter eigen_auto_fallbacks) and counts a Hermitian pencil by a counting SuperLU (`_superlu_count`). Opted in:
  solid_mechanics triangles (cudss), solid_mechanics_3d tetrahedra (cudss_mixed), em_fdfd_conical (cudss); other em /
  membrane / eme eigen roles stay superlu (own-factor and SuperLU-inertia consumers, unchecked).
- Several shifts: eigsh_shifted / eigs_shifted / factorize_host take `previous=` (cuDSS refactorised in place,
  `last_plan_reused`; SuperLU ignores it). Only because the shifted operator is built by `pattern_combination((K, M), (1,
  -sigma))` (root "Solver contract"): any new caller combining a pencil for a reused factor goes through it.
- `ShiftPencil` (`shift_pencil`): terms on one pattern once; SuperLU one-pass host combination per shift; cuDSS a
  DevicePencil + `factorize_pencil` on a private SparseDirectSolver (eigen_cudss_plans / eigen_double_fallbacks folded
  in). count_below / eigsh_shifted / eigs_shifted take `pencil=` to keep it across shifts.
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

## host_direct (`HostDirectFactorization`; MUMPS / MKL PARDISO back ends of sparse_direct and arpack_shift_invert)
- Factor of E = R S R (S = D A with a symmetrizer, R = |diag S|^-1/2 with zero / non-finite / missing diagonals 1, or
  the caller's row_scale), entry by entry on an own copy of the canonical pattern, as CudssFactorization: LDL^T when
  symmetry was requested (symmetric, positive_definite or a symmetrizer) and relative_asymmetry <= SYMMETRY_TOLERANCE
  (complex symmetric, not Hermitian), else LU; PARDISO real positive_definite = Cholesky (matrix type 2; a zero-pivot
  error -> LDL^T, sticky); MUMPS always LDL^T for symmetric kinds (python-mumps sets SYM=2 and exposes no SYM=1).
  `back_end` = the library ('mumps' / 'pardiso'), `precision` the request, `factor_precision` what is factored;
  `description` starts with the library label ('MUMPS LDL^T', 'PARDISO LU float32 + float64 GMRES refinement').
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
  pydiso's refactor checks only the nnz count, so the pattern is compared here first.
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
  pydiso without the keyword gets the library default METIS): the same fill as METIS at half the analysis time on
  2 - 4 threads (with METIS the analysis is 50 - 85 % of a cold PARDISO factorisation up to ~600 000 unknowns), equal
  on one thread, every matrix type and both precisions. Measured and not adopted: minimum degree (iparm[1] = 0: the
  fastest cold factor on 2D grids, 2 - 6 x slower solves and 1.7 x the fill in 3D), the two-level factorisation
  (iparm[23]) and iparm[24] (nothing on 4 threads; untested on more).
- Where a PARDISO refactorisation spends its time (2D FDFD, 160 000 unknowns, 4 threads: ~185 ms): the numeric
  factorisation ~50 %, the equilibration + asymmetry check ~30 % (single-threaded numpy / scipy; `symmetric_known`
  skips the check), pydiso's own `scipy.sparse.triu` of a symmetric type ~6 %, the probe solve ~12 %. *_mixed halves
  the factor memory but costs time in 2D (every full-accuracy solve = 3 library solves + GMRES; the float32
  factorisation is only ~1.4 x faster): it pays for memory, or where the factorisation dominates (large 3D).
- Availability (GUI-thread safe, nothing imported): MUMPS = find_spec('mumps') and the python-mumps distribution
  metadata (PyMUMPS installs another `mumps`); PARDISO = find_spec('pydiso') on x86-64. `missing_text(name)` is neutral
  ("'<name>' needs ...": back_end and factorization settings alike), `install_hint`, `factor_label` (texts).
- Threads: library defaults (MUMPS OpenMP / BLAS: OMP_NUM_THREADS, recorded; PARDISO MKL_NUM_THREADS,
  get_mkl_pardiso_max_threads recorded); no open_sol setting, oversubscription with numba / scipy BLAS threads is the
  user's to set. Both wrappers release the GIL in their calls; the calls are uninterruptible like SuperLU.
- PARDISO beside OpenBLAS (pip wheels and conda-forge defaults: numpy / scipy on OpenBLAS, pydiso on MKL): both pools
  spin after a call, so an MKL call right after a threaded BLAS call (a norm, a Gram-Schmidt product, ARPACK's own
  BLAS) shares the cores with spinning OpenBLAS workers (4 cores: a solve 15 -> 70 ms, a mixed solve 75 -> 450 ms, an
  ARPACK shift-invert run 0.16 -> 4 s). `host_direct.quiet_openblas(name)` (context; a back end / factorization /
  library name) holds every loaded OpenBLAS pool at one thread through threadpoolctl and restores it on exit: nothing
  for a name that is not PARDISO (MUMPS's BLAS may be that OpenBLAS), without threadpoolctl (optional, in the
  `pardiso` extra) or with numpy on MKL (one pool, no conflict). HostDirectFactorization wraps refactorize, solve_host
  and solve_preconditioned* itself (`openblas_pools` = pools held, 0 none); a consumer whose own loop does BLAS work
  between PARDISO solves wraps the loop (arpack_shift_invert: the scipy eigs / eigsh calls, by `self.factorization`).
  It is process-wide while held: only vector-sized BLAS work belongs inside, never a dense factorisation. BLAS calls
  of a physics between two PARDISO calls stay outside it: OPENBLAS_NUM_THREADS=1 (INSTALL.md) covers those.
- Not yet: out-of-core (MUMPS ooc / ICNTL(22), PARDISO iparm[59]), MUMPS BLR (ICNTL(35)), MPI, user orderings,
  per-pattern caches of the equilibration / asymmetry index work, a factorisation through the pydiso handle (no
  triu per refactorisation), PARDISO Hermitian types (4 / -4), remembered mixed rejections, a thread-count setting,
  calibrated memory / time
  constants and profile curves.

## explicit_kernels (`StepperConfig`; role 'stepper')
- Frozen settings bundle: precision 'single', 'double' or 'half_storage' (fp16 storage, float32 arithmetic;
  `storage_dtype` / `compute_dtype`). Readers: em_fdtd_ez / _hz (STEPPER_ROLE; physics/em sub file "Stepper
  precision"), em_fdtd_3d, elastic_waves, laminar_flow (HALF_STORAGE_STEPPER_ROLE). fp16 kernels use
  `core.half_storage.storage_preamble`; each engine keeps values O(1) by its own power-of-two storage scale.
- Backends ('gpu', 'cpu'): on the cpu the engines run the `core.kernels` twins; `backend_misfit` refuses half_storage
  there (no twin of PTX fp16 storage); double runs there.
