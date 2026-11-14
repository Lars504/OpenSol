# open_sol/solvers — conventions & gotchas

Design: `solver_plan.md` Part I. Root rules: structure_conventions.md "Solver roles".

## Layout and imports
- `roles.py` (SolverRole, SolverType, SolverRequest, ResolvedSolver, `solver_misfit` = the one fit rule) is pure Python
  and imports nothing from open_sol, so model.schema may import it; `__init__` imports only `roles`; model.schema
  re-exports SOLVER_ROLES, ResolvedSolver, SolverRole, SolverType, solver_misfit.
- Runtimes (`pcg.py` (+ `multigrid.py`, imported inside factorize), `line_block.py`, `sparse_direct.py`,
  `batched_direct.py`, `banded.py`, `arpack_shift_invert.py`, `block_eigen.py`, `time_filter.py`, `explicit_kernels.py`) import numpy only at module level; cupy / scipy / nvmath /
  core.gpu_memory / core.backend / core.host_memory inside functions, kernels built on first use, so core.fv_diffusion,
  model.definition_helpers and GUI-thread checks (`scipy_available`, `size_issue`, `expected_back_end`) may import them.
  `cudss_factor.py` is the one cuDSS wrapper (sparse_direct imports it; arpack_shift_invert reaches it inside
  factorize_host; sparse_direct's cuDSS time recipe uses arpack's CUDSS_* fit inside functions). Shared helpers, never
  a local copy: `cudss_factor.csr_rows` / `canonical_csr` (numpy only, module level in sparse_direct, arpack,
  multigrid), `sparse_direct.cudss_precision`, `require_backend_operator`, `ExactSolveRecord` (the PCGSolver-protocol
  record of BandedSolver and DiffusionDirectSolver), `cudss_factor.gmres_refine` (the GMRES-IR loop of cuDSS mixed
  factors and of host_direct, array module xp), `sparse_direct.recycled_gmres` (GMRES on an earlier factor's solve),
  `cudss_factor.ProbedFactor` (base of CudssFactorization and HostDirectFactorization: symmetric / mixed_factor /
  factor_precision / can_refactorize, the probe vector and gate; host_direct also takes its `_FactorRejected`),
  `pcg.DeviceCsrProduct` (the capturable gpu CSR product: call -> its own vector, `apply(vector, out)`; mechanics
  `_CsrApply`, time_profile).
- `cudss_factor.warm_cudss_library()` (builder `_warm_cudss`, a gpu build resolving a cuDSS back end): a daemon host
  thread imports nvmath.sparse.advanced and loads the library (`cudss.get_property`), nothing on the device; handles and
  plans stay on the GPU-owning thread, and `_direct_solver` joins the thread before the first DirectSolver (cuDSS is
  not thread-safe). nvmath's `cudss_data_ifc` / `cudss_config_ifc` set their thread-local `size_written` buffer only on
  the importing thread, so `_direct_solver` gives the GPU thread its own (`_thread_size_buffers`; a renamed nvmath
  module or attribute is an ImportError / AttributeError there). The handle creation is still in the first plan; measure what is left of speedup B3's 0.65 s.
  `cudss_library_timing()` gives the thread's perf_counter stamps (started / loaded / joined) and the first join's
  wait: an earlier start (api / CLI entry) waits on them (the load is GIL-bound: it overlaps only GIL-releasing
  main-thread work, and an earlier start would cost every gpu process without cuDSS the nvmath import).
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
  nodes without one follow it. A precision in a role's Automatic settings (`backend_defaults` / `grid_defaults`, e.g.
  solid_mechanics_3d gpu 'mixed') is the role's own default: resolved_solvers replaces it by a non-'auto'
  solver_defaults.precision the role takes, and a node of the Automatic type that sets no precision (and gets none from
  solver_defaults) takes it too (`schema.with_node_precision`: resolved_solvers, validate, GUI role combos; so a node
  added only for a tolerance keeps mixed). `schema.default_precision_ignored` = the validate info. Code resolving
  solver values outside resolved_solvers must go through the helpers.
- Backends (S6): the run backend must be in `PhysicsType.backends` (default ('gpu',); see each definition; misfits are
  validate errors per physics, greyed out in Add physics: `schema.physics_type_misfit`) and fit every role. Automatic
  resolves per backend: `SolverRole.backend_defaults` / `SolverRole.automatic(backend, grid, study)` (register_physics
  checks those types too): CG_DIFFUSION_LINEAR_ROLE on the cpu = sparse_direct (host SuperLU via `DiffusionDirectSolver`;
  cusolver_qr refused) in 2D, multigrid pcg on a 3D tensor mesh. `SolverRole.study_defaults` {(study, grid): (type,
  settings)} is read first, on every backend unless `study_default_backends` names some: CG_DIFFUSION_LINEAR_ROLE
  ('time_dependent', 'tensor') -> sparse_direct (a fixed dt keeps one factor; back_end 'auto' -> cudss on the gpu
  through auto_back_ends['tensor']); the 2D SOLID_LINEAR_ROLE stationary / time_dependent on 'tensor', both backends
  -> sparse_direct (cpu / no nvmath: host SuperLU); a fallback-table rule autopick will replace. `schema.with_node_precision` reads the plain Automatic
  when the study default is another type (a 2D solid's pcg node keeps the gpu 'mixed'). Fit guard (no size constant): `meshing.fitted_memory_estimate(spec, parameters,
  registry, meshes, memory_budget, estimate=None, by_mesh=None, shapes=None, host_budget=None)` -> (estimate,
  plain_roles); `schema.study_default_roles` (Automatic roles from study_defaults; a node never) resolve with study None
  if that estimate is lower: cuDSS candidates when the gpu estimate exceeds memory_budget; host candidates
  (`host=True`: a host factor whose Automatic without the study is another type, so cpu heat transients are none)
  when the `backend_memory` host estimate exceeds host_budget (None: the available RAM), and regardless of the budget
  when their `definition_helpers.direct_unknowns` (`SolverRole.direct_components` per cell / node) exceed the host
  SuperLU limit (SPLU_LIMIT_UNKNOWNS / SPLU_3D_LIMIT_UNKNOWNS: DiffusionDirectSolver refuses it). `plain_roles` {(physics, role)} goes to
  `resolved_solvers(..., plain_roles)`, `estimate_memory(plain_roles=)`, `ValidationContext.plain_roles`,
  `validate(plain_roles=)` (None: fitted there against memory_budget / host_budget; one info line per physics), `estimate_time(
  per_run, plain_roles=)`; the builder fits once against its budgets (`_fitted_estimate`) and hands it to all of them,
  api.check (its 'solvers' block) and the GUI document likewise (its memory cache drops on a budget change). Self-test
  `solver_study_default_fit`. Every caller of `automatic` passes the study (resolved_solvers,
  with_node_precision, default_precision_ignored, validate, api `_variant_spec`, GUI combos); a self-test counting CG
  iterations in a 2D transient pins a pcg node. Every role must fit (`solver_misfit`:
  `SolverType.backends` + optional `backend_misfit(values, backend)`, Automatic roles too). Runtimes take `backend=` from
  `SolverRequest.backend` and work on that backend's arrays only (`core.backend.array_module` / `sparse_module`): an
  array of the other backend is a TypeError, never a silent transfer. On the cpu `check_memory` counts the device
  estimate against host RAM.
- Size rules: a sparse_direct diffusion 'linear' solver has `sparse_direct.diffusion_size_issue` (2D SPLU_*, 3D
  SPLU_3D_* on window voxels; the 2D solid on 2 unknowns per node, `SolverRole.direct_components`; remedy: a pcg
  node) at the physics via `definition_helpers.cg_solver_issues`;
  DiffusionDirectSolver refuses above SPLU_LIMIT_UNKNOWNS; its host bytes are host items of
  `diffusion_system_estimate(direct=)`. em_fdfd_3d's curl-curl eigen factor: `sparse_direct.vector_3d_size_issue`
  (SPLU_VECTOR_3D_*, first values to lower from measured factors; no iterative remedy) + `host_lu_bytes(n,
  dimensions=3)`.
- `schema.resolved_solvers(spec, registry, physics_spec, study, parameters, plain_roles=frozenset())` is the one
  resolution rule (validate,
  `meshing.estimate_memory` -> `MemoryContext.solvers`, builder): only roles `used_in(study)`. A pcg preconditioner the
  node does not set resolves 'jacobi' at precision 'mixed' unless the role lists it in `mixed_preconditioners` (so
  the Automatic CSR multigrid never meets mixed); a
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
- Shared roles (model.definition_helpers; solid_mechanics keeps its own `physics.mechanics.definition.SOLID_LINEAR_ROLE`:
  pcg single / double / 'mixed', jacobi only, Automatic on the gpu backend pcg 'mixed'):
  `CG_ELASTIC_3D_LINEAR_ROLE` (pcg single / double / 'mixed', Automatic on the gpu backend pcg 'mixed': solid_mechanics_3d),
  `CG_DIFFUSION_LINEAR_ROLE` (+ 'mixed': heat,
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
  An opt-in type with precisions the role's Automatic lacks (batched_direct 'mixed', time_filter 'single',
  block_eigen 'mixed') needs them in the role's `precisions` too (solver_misfit checks both); with_default_precision
  then still leaves the Automatic alone because its own type does not offer them.
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
  a role's precisions or Automatic type updates its `_PRECISION_CASES` / `_MULTIGRID_CASES` / `_EIGEN_ROLE_CASES`
  ('eigen' roles, opt-in block_eigen) row.
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
- `factorize(operator, preconditioner=None, n_unknowns=None, single_operator=None, single_preconditioner=None,
  fused_iteration=None, single_fused_iteration=None, capture_blocks=False)` (all but the first three only for pcg;
  sparse_direct / banded factorize do not take them, so a caller passes them only to a PCGSolver): cupyx / scipy CSR or
  matrix-free `apply(vector, out)` (the CSR product writes one owned vector that the next product overwrites);
  preconditioner None = Jacobi from the diagonal, a flat inverse diagonal (fused, z never stored on the gpu) or
  `precondition(residual, out)` (block Jacobi). gpu dots are multi-block RawKernels (deterministic per size and device),
  so gpu inputs must be C-contiguous (solve raises).
- Fused iteration (speed plan A3, `FUSED_ITERATION_PROTOCOL`; matrix-free + callback preconditioner only, else
  TypeError; mixed: the float32 `single_fused_iteration`): the caller's object runs `start` (z0, rho, p0, q0) and
  `step` (x / r update, z, rho_next, the next direction into the second direction buffer, its product and p . q) in its
  own launches, keeping its own partial sums, and `rescale(direction, product, factor)` (mixed residual replacement:
  the p . q partials x factor^2); scalars and directions ping-pong, so CG blocks of an even length start in
  the same order. User: ElasticSystem3D (kernels3d.FusedBrickIteration, 2 launches per step instead of 8).
  `workspace_bytes(fused=True)` books the second direction.
- CUDA graphs (`capture_blocks`, gpu, `CG_BLOCK_GRAPHS`): each full even-length check block runs once eagerly per
  (operator, fused object, solution pointer, length), then is captured once and replayed as one launch (`_BlockGraphs`,
  at most MAX_BLOCK_GRAPHS kept); a failed capture warns and stays eager. Graphs bake every pointer the callbacks read:
  an opting-in caller calls `release_block_graphs()` whenever it replaces those arrays (ElasticSystem3D.assemble);
  factorize drops them. Nothing inside a block may sync or allocate (the residual read sits between blocks). The own
  multigrid hierarchy (`_multigrid_bound`: the bound callback is `self._multigrid.precondition`) is a `_BlockGraphs`
  participant: its `inside_capture` flag is set during a capture, so `_DeviceCycle.apply` launches its plan's kernels
  into the block's graph instead of launching / capturing its own (a graph launch inside a capture is not relied on);
  the hierarchy is rebuilt only by factorize, which makes new _BlockGraphs. Caller-built V-cycles (multigrid3d) are no
  participants.
- cpu: numpy twins of the fused kernels (`_HOST_KERNELS`, same signatures: a kernel change changes its twin),
  allocation-free (BLAS axpy, scipy's private `_sparsetools.csr_matvec` into the workspace product, falling back to
  `dot` when missing or dtypes differ; Jacobi z in the product vector); dots / axpys of vectors shorter than
  HOST_SMALL_VECTOR_LIMIT run as serial numba loops when numba imports (threaded BLAS start-up was ~20 us per call:
  speed plan A6). Workspace reallocated only when size, dtype or kinds change; `workspace_bytes` is what the physics'
  estimator counts (SolverType estimate empty). The solution vector stays physics-owned.
- `solve(rhs, solution, reference)`: warm start, in place; flat arrays of the solver dtype (ValueError otherwise); b = 0
  returns 0 (||b||^2 and ||r0||^2 come back in one read: `start_norms`). The residual is read every
  `_check_interval()` iterations of the BOUND preconditioner (gpu 16, a callback bound under the 'multigrid' setting 2;
  cpu 4 / 1): a Jacobi bind of 'multigrid' reads every 16, and magnetic `kept_preconditioner._bind_callback` (binds
  under 'jacobi', restores 'multigrid') relies on the callback rule. cg_iterations round up to that block (the
  estimate rounds too, by `SolverWork.backend` and `bound_preconditioner`). A
  NaN / inf residual (start or block read; mixed: the outer residual) stops at once, unconverged, with a 'non-finite
  residual' last_warning (`_non_finite`), never iterating to max_iterations.
- Stop reference: 'rhs' (tol ||b||) for steady solves (warm re-solves cheap); 'initial_residual' (tol ||b - A x0||,
  floored at ROUNDOFF_FLOOR_EPSILONS eps ||b||) for implicit steps: against ||b|| the C/dt u_old term dominates and CG
  stopped before the step moved. The caller picks it per solve; `last_residual` is relative to it.
- Precision names are solver ones ('single' / 'double' / 'mixed'); `SOLVER_DTYPE_NAMES` maps them to the assembled
  dtype ('mixed' -> 'float64').
- 'mixed' (`_iterate_mixed`): CSR + Jacobi (float32 copy of the values sharing the float64 indices + float32 inverse
  diagonal; a block preconditioner with a CSR: TypeError) or matrix-free (float64 apply + the caller's float32
  `single_operator` / `single_preconditioner` on its own float32 copy; TypeError without both). One float32 CG on A d = r / s,
  s = ||r|| (unit scale: physical residuals underflow float32 dots at 1e-12), with residual replacement: when the
  recurrence residual fell by MIXED_REPLACEMENT_RATIO (or met the stop) at a check-block boundary (never inside a
  captured block), x += s d and the float64 r = b - A x replaces it at the new unit. Within MIXED_DRIFT_LIMIT of the
  recurrence norm the direction is kept and the recurrence state (direction, rho x factor^2; fused: product and p . q
  via `rescale`) rescaled, so the CG coefficients stay consistent (van der Vorst-Ye); beyond it CG restarts from r
  (recomputing rho against a kept direction diverged on stiff 2D / 3D models). Stops at the float64 reference, the
  iteration limit, or a replacement shrinking ||r|| by less than MIXED_STALL_RATIO while the recurrence did;
  `refinements` counts the float64 refreshes. `workspace_bytes(..., 'mixed')` / `mixed_extra_bytes` count the solver's copies; a
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
  R A P with the full A; coarsest <= COARSEST_DENSE_LIMIT by a host dense inverse (densified on the host,
  `_host_dense`: cupyx toarray always copies and coosorts; raises on a non-finite inverse):
  LAPACK potrf / potri (`_cholesky_inverse`, ~4x cheaper than eigh) when min pivot^2 > PSEUDO_INVERSE_CUTOFF x max
  pivot^2, else the eigh pseudo-inverse (floating regions, pure Neumann; counter multigrid_coarsest_eigh). Rounding
  only weakens the preconditioner (SPD holds for any symmetric A_c).
- Products run in row blocks of ~PRODUCT_BLOCK_PRODUCTS (`_row_blocks`): one SpGEMM over a 5e6-row 3D operator needs a
  workspace larger than an 8 GB card, so never multiply the full fine operator on the backend. The block sum merges
  only equal block counts (binary counter): a running sum recopies A_c per block. A cupyx / scipy CSR row slice copies
  the matrix even over the full range: block code takes `_rows`, never `matrix[start:stop]`.
- gpu Galerkin products also run in column chunks of P (`_column_ranges`, <= PRODUCT_MAX_COLUMNS): cuSPARSE's default
  SpGEMM raises INSUFFICIENT_RESOURCES when a segment of intermediate products overflows its 32-bit coding, which grows
  with the right factor's columns, so smaller row blocks do not cure it (carey2026 E2 mesh_fine, 5.6e6 cells). A chunk
  gives the column block R A P_c of A_c; its transpose (A_c symmetric) is the row block, stacked by `_stack_rows`.
  `_column_block` slices by array masks; its P_c copy is not in `setup_extra_bytes` (a few % of the setup peak). The
  prolongation smoothing A_F,b T (all coarse columns, one product per entry) is not chunked.
- Hierarchy (T, P, R, coarse operators / vectors) is HIERARCHY_DTYPE float32 at every precision; a float64 fine level
  casts its residual before R. `precondition` scales the residual by a power of two to max |r| in [0.5, 1) and back
  (exact, no sync): unscaled ~1e-40 residuals underflowed the float32 levels and CG ran to its limit.
- PCGSolver builds it in `factorize`: assembled scalar CSR only (TypeError for a CSR with a callback preconditioner;
  the passed inverse diagonal is ignored), not with 'mixed' (ValueError in `_factorize_mixed`). A matrix-free operator
  with a callback brings its own multigrid: pcg uses the callback as given (ElasticSystem3D:
  `physics.mechanics.multigrid3d`, any precision; its counters through `PCGSolver.preconditioner_counters`, its
  setup ops through `SolverWork.setup_ops`). Whether a role runs multigrid at 'mixed' is the role's
  `SolverRole.mixed_preconditioners` (default ('jacobi',); `solver_misfit` reports it at the physics; the node-level
  pcg validate no longer checks it), and `resolved_solvers` falls back to jacobi only for a preconditioner the node
  does not set and the role does not list there. Automatic for
  `CG_DIFFUSION_LINEAR_ROLE` (gpu, and 3D tensor meshes on the cpu; jacobi at mixed); a pcg node without the setting is
  jacobi (`with_exact_cg` too).
- Aggregation (host numpy): strength -a_ij >= STRENGTH_THRESHOLD sqrt(a_ii a_jj) (negative off-diagonals only) plus each
  row's largest negative coupling (floor; without it a 3D level-1 operator left ~98 % of rows out of P), symmetrised
  by scipy's row-local CSR merge S + S^T (int8 ones: no sum cancels; it replaced a np.unique of 2 E int64 keys, 70 %
  of a 3D setup); distance-2 independent-set roots (Luby, fixed seed: deterministic), each round on the edges of rows
  still near an undecided row (the list only shrinks; the same roots as full rounds); rows without negative couplings
  (held / pinned identity rows) stay out of P. The prolongator filtering keeps the plain threshold rule. gpu setups
  run the device twin `device_aggregate_strong_links` (DEVICE_AGGREGATION; RawKernels per row + one cuSPARSE csrgeam
  union, one sync per Luby round): the same aggregates bit for bit (float64 strength sums per row in CSR order, the
  same keys / rounds, max-only joins, so union order is free), no host copy of the level CSR. A change of the host rule
  changes the kernels in the same edit; self-test multigrid_device_aggregation (heat selftests_multigrid) compares
  both on synthetic operators and every level of a gpu setup. Its transients (~60 B per unknown + ~25 B per entry) stay
  below the booked setup_extra_bytes; the host 'multigrid aggregation' item is booked only off the gpu
  (`diffusion_system_estimate(backend=)`: the heat / potential / magnetic estimators pass `solver_backend(spec)`) and
  `setup_ops(backend=)` prices device passes + LUBY_ROUNDS_GUESS syncs there.
- Reuse: one host transfer of each level's CSR only when the fine pattern changed (`_pattern_matches`: identity, then
  array_equal); a value-only factorize (re-assembly, Newton / B-H iterations, refresh_diagonal) rebuilds P, R and coarse
  operators on the backend from the cached aggregates; rows whose strong links turned weak get prolongator weight 0
  (P = T there), which keeps value-only setups converging like fresh aggregation (no re-aggregation needed). An empty
  fine-level aggregation is never reused: every setup re-tests it (one sync on the gpu when still empty), else a larger
  dt or a steady solve on the same pattern stayed smoother-only on a Laplacian.
- Jacobi binds and lagged setups (PCGSolver, own CSR hierarchy only; pcg module header): 'multigrid' binds the passed
  inverse diagonal when the setup found no strong link on a fine level above COARSEST_DENSE_LIMIT
  (`MultigridPreconditioner.smoother_only`) or below MULTIGRID_MIN_UNKNOWNS on the solver's first factorize (no setup;
  the binding's second solve or the next factorize sets it up, later ones rebind); a Jacobi solve over CG_ITERATIONS_PER_AXIS_CELL sqrt(n) iterations continues on the hierarchy and turns the
  binds off for that solver. A same-pattern factorize keeps the hierarchy (`can_rebind` / `rebind`: level 0 takes the
  new operator and its l1 weights, the device cycle re-plans when the operator object changed; DiffusionSystem's
  refresh_diagonal changes A_solve in place) while the last solve took <= LAGGED_SETUP_ITERATION_RATIO x the fresh
  one; a kept solve over its cap continues on a fresh setup. Mid-solve setups count as factorize seconds and make new
  _BlockGraphs. Counters multigrid_jacobi_binds / _jacobi_fallbacks / _lagged_binds / _lagged_fallbacks: the summary
  still says 'multigrid'. The time recipe prices case (b) as Jacobi (`bound_preconditioner`: one factorization
  and one solve, not stepping); case (a) and the lag are not known a priori (priced as multigrid, every factorize a
  setup). The memory estimate keeps booking the hierarchy (case (b) builds it on a second solve).
  MULTIGRID_MIN_UNKNOWNS = 4000 sits below the measured crossover (~6.5e3, solver_profiler heat_2d gpu) so the
  outer-graded heated-dot selftest keeps the hierarchy. magnetic `KeptPreconditionerSolver` keeps its own hierarchy
  and binds under 'jacobi', outside all of this.
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
  MEMORY_FRACTION of `gpu_memory.pool_room_bytes`: budget - pool used, at most the free device memory, since cuDSS
  factors of other physics lie outside the pool); consumers estimate `operator_bytes` (run item) + ONE `shift_bytes` (the
  chunk grows only into free budget, so min(n_shifts, MAX_CHUNK_SHIFTS) x shift_bytes over-books and refuses runs
  that fit).
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
  counted from the code: a loop change updates them). A solve_shifted chunk shares those calls between its shifts:
  the caller passes `SolverWork.batch_shifts` (mean shifts per chunk; 1 = a call per shift, kept factors) and the
  recipe prices one chunk over its shifts (`_factor_ops`, `_sweep_ops`, `_per_shift`); above BATCHED_INVERSE_MAX_BLOCK
  the block inverses are one call per matrix, as `_invert`. Per block: two complex products (DenseFlops, 8 b^3 real
  flops per shift each; mixed: `single`, the complex64 curve) and the inverses as `BlockInverse` on the profile's own
  inverse curves: a chunk does not scale as b^3 (measured on an RTX 3050, thermal noise, whole chunk time per block
  and shift: 44 us at b <= 67, 59-74 us at b 111, ~0.55 ms at b ~200 with looped inverses), so pricing all three
  calls as float64 product flops was 3.4x high at b 111 and 1.4x at b 67. solve_shifted times its chunks: work counter `shift_seconds`, the
  seconds group of the caller's items (heat thermal noise).
- Kept factors (`factorize`, `factorize_operators`: `_factor(kept=True)`) also store G_k = D_k^-1 L_k and H_k =
  D_k^-1 U_k of the scaled system (`_Chunk.lower / upper`; H_k is the elimination's coupling anyway, G_k one more
  product per block): `_kept_sweep` = y_k = D_k^-1 y_k - G_k y_(k-1), y_k -= H_k y_(k+1), scales only at both ends. A
  single rhs on the gpu whose three stacks fit KEPT_SWEEP_KERNEL_MAX_BYTES is one launch of the raw twin
  'solvers.line_block_kept_sweep' (one CUDA block per shift, a warp per row, blocks in sequence; numpy twin = the
  block loop `_kept_sweep_loop`, which also runs larger gpu stacks); solve_shifted chunks keep `_sweep`. Memory: 3 block
  stacks per kept shift (`shift_bytes(kept=True)`, `chunk_size(kept=True)`, batch_size): every kept-factor estimate
  and sizing passes it (fdfd / bor / cross-section line_block bytes); the recipe of a kept solve is
  `factor_solve_ops` (`_kept_sweep_ops`). A change of the sweep formula changes the CUDA source and the loop
  together.
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
  of the same indptr / indices pre-permutes and skips MMD (counters reused_orderings / eigen_reused_orderings). It
  also keeps the permuted CSC pattern and the CSR entry of every CSC slot (`_permuted_pattern`, made once in
  `remember`: H2), so a reused factor permutes by one gather (`permuted`, canonical CSC: splu's sum_duplicates is
  skipped) instead of `symmetric_permutation`'s COO -> CSC sort; a pattern with duplicate entries keeps the COO route.
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
  (HOST_DIRECT_REFUSALS) raise, no fallback. `fall_back`, `device_pencil` and `factorize_device_csr` stay cuDSS /
  auto only; `pencil` / `factorize_pencil` serve them by a HostPencil (Device pencils below). Counters (host direct runtimes): host_direct_analyses (fallback
  re-analyses included), host_direct_reused_analyses, host_direct_peak_bytes (the library estimate), host_direct_lu_entries,
  host_direct_symmetric_factors, host_direct_refinement_steps, host_direct_single_residual, host_direct_double_fallbacks,
  host_direct_threads; host_factor_bytes = the library's factor bytes. Time recipe: the SuperLU items (labels named by
  the library) x HOST_DIRECT_FACTOR_TIME / HOST_DIRECT_SOLVE_TIME, mixed solves x HOST_DIRECT_MIXED_FULL_SOLVES (all
  uncalibrated, no profile curves yet).
- 'auto' -> cuDSS only through `SolverRole.auto_back_ends` {grid: back end}: `schema.auto_back_end_values` (inside
  resolved_solvers: validate, estimators, build and summary see the concrete back end) on the gpu with nvmath
  (find_spec); role-less `expected_back_end` / `resolve_back_end` map 'auto' to SuperLU. Opted in: pressure_acoustics
  (triangles, tetrahedra) and membrane (triangles) 'linear' cudss_mixed; CG_ELASTIC_3D_LINEAR_ROLE (tetrahedra)
  cudss_mixed; cudss for FDFD_LINEAR_ROLE / CUDSS_LINEAR_ROLE (tensor, triangles), SOLID_LINEAR_ROLE (triangles),
  CG_DIFFUSION_LINEAR_ROLE (triangles, tetrahedra).
  Self-tests that used Automatic as a SuperLU reference pin a scipy_splu node. Values carry `AUTO_BACK_END_KEY`
  ('back_end_from_auto'); the runtime gets `fallback_back_end='scipy_splu'`, and a cuDSS refusal
  (`cudss_factor.CUDSS_REFUSALS`: zero pivot, accuracy, `CudssMemoryError`) within SPLU_LIMIT_UNKNOWNS switches it to
  SuperLU for good (`fall_back`: RuntimeWarning, counter auto_fallbacks) and redoes the factorisation there (every entry
  point, incl. DiffusionDirectSolver's and fem.harmonic's deferred check at solve). An explicit cuDSS node raises. The em
  FDFD / BOR / conical consumers free their plan-kept cuDSS factor once on SuperLU.
- Entry points `factorize(..., previous=, symmetric=, symmetrizer=, defer_check=)` (device CSR, pulled to the host
  once), `factorize_host(matrix, real, symmetric, positive_definite, symmetrizer, row_scale, previous, symmetric_known,
  defer_check)` (SuperLU ignores the cuDSS options; symmetric_known reaches cuDSS and the host direct back ends alike:
  a canonical symmetric CSR whose index arrays are never modified, no canonical copy, no asymmetry check, the planned
  pattern matched by identity), module `cudss_factorization(...)`: all return a `cudss_factor.CudssFactorization`;
  `transpose=True` only with a symmetric factor (x = D (D A)^-1 b for a symmetrizer D), else NotImplementedError; a real
  factor takes a complex rhs without a cast (cuDSS: two real solves; a real SuperLU `solve_host`: one solve of the
  (n, 2) real / imaginary columns, H1); `recyclable` False (decided: a value-only refactorisation skips the plan and is
  exact).
- Plan reuse: pass the factor to replace as `previous=`; same options = refactorised in place (values written into the
  planned device CSR, never nvmath `reset_operands(a=)`, which drops the plan for device operands); a new pattern
  re-plans; other options release it first. A Cholesky is planned only for a double factor (`_requested_kind`): a mixed
  positive_definite request plans float32 LDL^T (float32 Cholesky hung > 175 s), its double fallback is the double
  Cholesky.
- DiffusionDirectSolver on cuDSS: `factorize_device_csr(operator, real=True, positive_definite=True, previous=)`
  (Cholesky, LDL^T after a failed one), device solves; `release_factor()` before another large factorisation, or
  `detach_factor()` (the caller owns the factor: released or passed as previous= to a factorisation of the same kind;
  `positive_definite = False` first makes factorize_host_operator ask cuDSS / MUMPS / PARDISO for the LDL^T of a
  shifted pencil instead of the SPD kind, which can_refactorize would refuse: mechanics stress stiffening).
  `factorize_host_operator(host_csr, symmetric_known=, defer_check=)` (FEM Newton tangents): symmetric_known = a
  canonical symmetric CSR (values copied only, index arrays shared and never modified, asymmetry check skipped; MUMPS /
  PARDISO then refresh by `host_direct_values_factorization`, pattern matched by identity); never pass it for an
  operator that may be unsorted, duplicated or unsymmetric. defer_check (both tangent routes; Newton solves right after
  factorising): a reused analysis / plan skips its probe, the next solve is the check, so a CudssAccuracyError of an
  'auto' runtime then comes from that solve (no SuperLU fallback). `SparseDirectSolver.factorize_values(values, indptr,
  indices, real=, previous=, defer_check=)` is the general role's twin for unsymmetric operators on a fixed canonical
  pattern (drift_diffusion Newton Jacobians): cuDSS refresh / plan from device values, MUMPS / PARDISO
  `host_direct_values_factorization`, else SuperLU of the CSR; no host canonicalising or asymmetry check per call.
  `factorize_device_values(values, indptr, indices)` (cupy
  values on the HOST canonical pattern) -> `refactorize_device_values` / `from_device_values`: equilibration, lower take
  and refresh or plan on the device by index maps (`_DeviceIndexMaps`, one per plan: each array uploaded at its first
  read, so `refactorize_device_lower` holds the lower triangle's rows / indices alone, ~4 B per pattern entry against
  ~18 B on the full-values route; read a new array through `maps[key]`, never assume it exists; its host passes over
  the pattern are freed by `_finish_refactorize`; a lower-stored DevicePencil hands its rows and lower take over once,
  `host_passes=` -> `seed_host`, R5), no host value copy. A refresh on the reused plan without symmetrizer or symmetry
  check runs two kernels (`_fused_refresh`, K2: R and a symmetric kind's diagonal, then E and its factor-dtype copy into
  the planned buffers); a change of the step-by-step route's scaling rule changes `_row_scale_kernel` /
  `_scaled_take_kernel` in the same edit. The host route (`refactorize` -> `_equilibrated`) makes the
  int32 rows once for the scaling and `_lower_triangle`, and scales by `equilibrate_values` (host twin
  'solvers.cudss_equilibrate': one numba prange pass from EQUILIBRATE_NUMBA_MIN_ENTRIES once numba is imported, K4;
  case solver_cudss_equilibrate_numba); symmetric_known takes the norm from the row sums alone. Either
  symmetry vouched (pencils, device tangents) or `check_symmetry=True` (device transpose map), which also serves
  `refactorize` of a host operator on the planned pattern without symmetrizer (values uploaded only) and
  `SparseDirectSolver.factorize_device_csr` (non-canonical cupyx CSR canonicalised in a copy). A symmetrizer factor
  takes the device path too (`symmetrizer=`: rows x D before R, exactly when the factor was built with one): module
  `factorize` of a cupyx operator with a valid `previous` of the same options refreshes it there
  (`_refreshed_on_device`; the 2D FDFD per-wavelength route, Sp3), only a first factor goes through the host copy;
  `row_scale=` replaces R = |diag|^-1/2.
- Device pencils (sweeps): `SparseDirectSolver.device_pencil(indptr, indices, term_values, symmetric,
  symmetric_known=False)` -> `DevicePencil` (None off cuDSS / on the cpu: the caller keeps factorize_host); terms
  uploaded once, symmetry checked once unless vouched (max |X - X^T| <= SYMMETRY_TOLERANCE max |X| per term; a vouched
  pencil of host values takes each lower triangle on the host, so the full terms never reach the device). A
  symmetric pencil stores lower triangles only (`lower_stored`; `full_term(k)` returns a new array per call, its take
  built once at ~48 B per pattern entry of int64 transients: the SuperLU fallback and self-tests only, never a
  product; `combined(c)` in the stored layout). `factorize_pencil(pencil, coefficients, row_scale=, previous=,
  extra=)` combines on the device and refreshes `previous` (refactorize_device_lower / refactorize_device_values with
  defer_check); the pattern is matched by identity of the pencil's indptr / indices first: callers never modify those
  arrays in place. `extra=(pencil.slots(rows, columns), values)` adds a small per-frequency block (port DtN) at
  stored-layout slots (lower entries only for a lower-stored pencil, symmetry vouched by the caller; a None slot =
  outside the pattern: keep factorize_host). `device_pencil_bytes(entries, term_value_bytes, symmetric)`; users
  (fem.harmonic, mechanics frequency_response.HarmonicSystem) book it + factor maps + transients as a run item. The
  operator norm of a device refresh is lazy and summed by one atomic-add kernel (`_magnitude_sums_kernel`: no |E|
  copy; cupyx.scatter_add of cp.abs(E) and its masked copy held 17 B per stored entry at the probe, which is after
  the plan returned the pool's cached blocks, so it set the run's pool mark).
- Host pencils: consumers ask `SparseDirectSolver.pencil(...)` (same arguments): a `HostPencil` on a host direct back
  end (either backend), else device_pencil's answer. HostPencil keeps the terms on the host on the full pattern
  (`lower_stored` False: an extra block gives both triangles; consumers test `pencil.symmetric`, never lower_stored, before
  adding an unsymmetric block), symmetry checked once per term; `factorize_pencil` combines on the host and calls
  `host_direct_values_factorization` (refactorize_values: identity-matched pattern, no canonical copy, no asymmetry check,
  defer_check; host_direct section). Users: fem.harmonic (acoustics / membrane estimators book its terms as a host
  item) and mechanics HarmonicSystem (references its own term arrays: nothing new to book).
- defer_check: a factor on a reused plan whose earlier probe passed skips its probe; its first (full-accuracy) solve is
  the check, so probe_residual / single_residual / refinement_steps counters lag one factor. A double miss turns on
  cuDSS refinement and solves again; a second miss releases the factor and raises CudssAccuracyError from the solve
  (inertia-only factors are never judged: count_below passes defer_check=False). A mixed miss re-plans double in place from the device E (`fallback_listener` ->
  cudss_plans / cudss_reused_orderings / cudss_double_fallbacks): callers translate CudssAccuracyError at solve /
  solve_host too (fem.harmonic.solve; FDFD driven solves: fdfd_2d `_solve_linear`, host_fdfd `_checked_solve_host`,
  physics/em sub file). Fixed-step solves (refinement_steps=) leave a pending check pending. Entry points with
  defer_check (K1): factorize_pencil / factorize_values / factorize_host_operator, `SparseDirectSolver.factorize`
  (`_refreshed_on_device`, the FDFD wavelengths) and `factorize_host` (`CudssFactorization.refactorize`, BOR omegas;
  default False on both): a caller passing it catches CudssAccuracyError at the solve and, when `fall_back(error, n)`
  is True, drops the factor and factorises again (SuperLU now); a first plan is always probed.
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
- Parked plans (K5, `cudss_factor` header): in the same scope `release()` of an accepted factor parks its planned
  DirectSolver (KEPT_PLANS, key = ordering key + kind + factor dtype, operand references dropped: the pool arrays go,
  the outside-pool bytes stay in external_live_bytes), and the next plan of that key by any factor object adopts it
  (`_adopted_plan`: nvmath's pointer updates, no analysis, no memory check; counted as a plan on a kept ordering,
  `last_plan_adopted`); a rejected adopted plan re-plans from scratch once. A plan of another key, each build start
  (`reset_external_high_water`, study.build_session), the scope end and every `cudss_orderings_kept` exit (a GUI study)
  free the parked plans, so a parked factor outlives its phase only until the next plan of the same build. Self-test:
  `_ordering_reuse_checks`.
- On-disk ordering store (K6): `remember_cudss_orderings(store=True)` (default; the GUI worker's call) with a folder
  in OPEN_SOL_CUDSS_ORDERING_CACHE (unset = off, like OPEN_SOL_MESH_CACHE) also writes each fresh plan's (permutation,
  tree) there per (key, cuDSS version), pruned by mtime to ORDERING_STORE_MAX_BYTES, and reads a key the in-process
  store lacks (a bad file plans from scratch). run_study's `cudss_orderings_kept()` turns the scope on without the
  store, so api / CLI runs and selftests never read it (a GUI run inside the worker's scope does).
- Measure-first switches (off; `tools_profilers/time_profilers/cudss_phase_profile.py` times each against the nvmath
  path and writes its results file): REFACTORIZATION_PHASE (K3: refreshes of a plan factorised before run cuDSS's
  REFACTORIZATION phase by `cudss.execute`), RAW_SOLVE (K7: `cudss.execute(SOLVE)` into one kept solution buffer, no
  nvmath sync; solve_buffer_bytes books the buffer), REORDERING_ALGORITHM (K10: e.g. 'AMD' for fresh plans; no
  ordering kept or reused while set). Any raw-phase error uses the nvmath call from then on (`_raw_failed`).
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
  SOLID_LINEAR_ROLE 'triangles' and CG_DIFFUSION_LINEAR_ROLE / CG_ELASTIC_3D_LINEAR_ROLE 'tetrahedra' -> sparse_direct
  (so the gpu Automatic 'mixed' of both solid roles applies on tensor meshes only).
  pcg, sparse_direct and arpack_shift_invert declare grids ('tensor', 'triangles', 'tetrahedra'); banded ('line',) only.
  pcg on tetrahedra is an opt-in node of the scalar FEM systems; solid_mechanics_3d on tetrahedra refuses it at create
  (sparse_direct only; its validation does not catch it yet).
- The runtime is the PCGSolver protocol itself: `factorize` pulls the CSR to the host, refuses a bandwidth above
  `max_bandwidth` (8, before the band array exists), an unsymmetric operator (`banded.SYMMETRY_TOLERANCE`: only the
  upper triangle is read) and a non-positive-definite one (ValueError), keeps the `cholesky_banded` factor. Precision
  'double'; host O(n) memory, not estimated. Not yet: per-order (m / l / q) batching, complex operators.

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
  which CUDSS_FILL_3D read 48 % / 76 % low) x the factor value bytes + the kept matrix copy +
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
  = 0 -> NaN). Not for a Hermitian pencil (eigsh_shifted keeps M-orthogonality).
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
- Memory: 9 b n values of the phase dtype (S, KS, MS of (3 b, n); b = wanted + guards) + CHUNK_BYTES scratch, nothing
  else of that size: every block product runs by column chunks in place (`_rotate`, the W projection), compaction is
  forward row copies, the start rows are drawn row by row on the gpu. Mixed frees the float32 stores before the
  float64 ones (the float32 X is the only copy held across): `block_eigen.block_bytes` is the estimate item (a
  transient of the eigen solve; the consumer adds its own copies, e.g. float32 operator copies and a private
  multigrid hierarchy). Host: the k eigenvectors only.
- Syncs per iteration: two host reads (residual norms, Gram pair) and one small upload (the rotation coefficients);
  the per-row callbacks stay asynchronous. The Rayleigh-Ritz is a host eigh of
  (3 b)^2 (diag(B)^-1/2 scaled, B eigenvalues below DROP dropped: dependent W / P directions near convergence).
- Mixed: float32 to max(tolerance, SINGLE_PHASE_TOLERANCE) or a STALL_ITERATIONS stall (the float32 floor of the
  products is ~eps32 lambda_max / lambda: ~1e-3 on fine 3D elasticity, so the float64 phase is always needed for tight
  tolerances; it starts from the float32 X with a fresh P). Double stalls over DOUBLE_STALL_ITERATIONS stop
  unconverged (tolerance below the float64 floor), with a warning.
- Fewer free unknowns than DENSE_MAX_RATIO x b: a dense scipy eigh of the applied K and M (`eigen_dense_solves`).
- Counters: eigen_solves, eigen_iterations / _single_ / _double_iterations, operator_applications (K rows),
  mass_applications, preconditioner_applications, eigen_seconds, eigen_unconverged, eigen_dense_solves,
  eigen_block_size, eigen_last_residual. Time recipe `block_eigen.time_estimate`: per iteration and row the caller's
  product_ops (K and M) + preconditioner_ops, block passes, DenseFlops (single for the mixed share), 2 syncs;
  iterations a priori (multigrid: MULTIGRID_ITERATIONS_PER_DECADE x decades, flagged by work.setup_ops; Jacobi:
  sqrt(kappa) or CG_ITERATIONS_PER_AXIS_CELL x longest axis): uncalibrated guesses.
- Not yet: warm starts across sweep values / assemblies, hard locking (converged rows leave X), a fused residual +
  norm kernel, block (multi-vector) brick kernels (today one launch per row), FEM pencils (tetrahedra / triangles:
  SA-AMG of K + tau M as preconditioner), Maxwell (gradient-space projection + auxiliary-space preconditioner).
- Self-tests: solver_block_eigen_cpu_gpu (synthetic 3D Laplacian pencils, both backends and precisions, mask, Jacobi
  callback, dense path, counters); solver_default_precision rows eigen_* (role resolution, mixed keeps multigrid);
  model runs: mechanics sub file "Self-tests".

## batched_direct (`BatchedDirectSolver`; role 'linear', opt-in)
- A SparseDirectSolver subclass (per-system solves on cudss_mixed / cudss by precision, SuperLU after a cuDSS refusal;
  cpu: scipy_splu), so every consumer of the role works unchanged and only consumers that test
  `isinstance(linear_solver, BatchedDirectSolver)` batch. Opt-in by `opt_in_types=('batched_direct',)`:
  PRESSURE_LINEAR_ROLE and MEMBRANE_LINEAR_ROLE, whose precisions list 'mixed' for it alone (sparse_direct offers
  double only, so with_default_precision leaves the Automatic untouched). Grids triangles / tetrahedra.
- `pattern(indptr, indices, terms, symmetric)` -> BatchPattern (kept per consumer pattern: never modify its arrays);
  `batch_capacity(pattern, n)`; `solve_batch(pattern, coefficients (N, T), rhs (N, n), extras)` -> host solutions,
  relative residuals, accepted: values = coefficients @ terms on the device (one GEMM; extras = (pattern slots, values)
  per system, the port DtN blocks), R = |diag|^-1/2 equilibration, the lower triangle of a symmetric pencil, one
  explicitly batched nvmath DirectSolver (lists of operators and rhs buffers) planned once per (pattern, batch size,
  dtype) and refreshed in place (`data[k]` rows are the CSRs' data arrays: never rebind them), refinement against each
  system's complex128 operator by one SpMM of a row selector. A shorter batch pads with copies of its last system.
- Rejected systems (above ACCEPT_TOLERANCE, non-finite) are the consumer's to re-solve on the inherited per-system
  path. Any error of the batched API or a plan over DEVICE_FRACTION of the free memory raises BatchUnavailable and
  sets `unavailable` for good (the consumer warns once and goes per system): explicit batching is assumed from nvmath
  1.0 (unmeasured here: check `batch_unavailable` in the counters after a first gpu run).
- The kept plan lives outside the cupy pool (book_external_bytes / release_external_bytes in cudss_factor):
  consumers call `release()` when their plan ends (fem.harmonic: after the last planned omega and in clear_reduced).
- Consumer fem.harmonic `plan_batched(omegas, extra_loads, held_offsets_of)` (the plan_reduced interface; module
  header): the physics' plan_frequency_response calls it for sweep_method 'direct' with a batched node, and their
  solve_frequency_response must then take the planned branch (no per-frequency set_frequency_load: it changes the
  load object the plan keys on, and solve raises). frequency_terms must be a pure function of omega (they are
  evaluated for a whole batch ahead of the solve).
- Counters: the SparseDirectSolver ones + batches, batched_systems, batch_plans, batch_rejected,
  batch_refinement_steps, batch_seconds, batch_plan_seconds, batch_permanent_bytes / batch_peak_bytes (nvmath's plan
  estimates, summed over systems), worst_batch_residual, batch_unavailable. Time recipe: per-system cuDSS
  (an upper bound: the batching gain shows as a time_check deviation, not a lead on the recipe).
- Not yet: mechanics HarmonicSystem (2D / 3D frequency response), FDFD wavelength scans on triangles / cuDSS,
  thermal noise off line_block, BOR order sums, lockstep ARPACK on batched factors (several targets, EME sections).

## time_filter (`TimeFilterEigenSolver`; role 'eigen', opt-in)
- Settings holder + dense Rayleigh-Ritz (`ritz_pairs`, any array module) + counters; the time stepping is the
  physics' (physics.em.time_filter_eigen: solvers never import physics). Opt-in through `eigen_role(time_filter=True)`
  (opt_in_types, precisions + 'single' = the FDTD stepper precision): em_fdfd_ez / em_fdfd_hz, grids tensor; the FDFD
  physics dispatches on `isinstance(eigen_solver, TimeFilterEigenSolver)` before any factor is made.
- ritz_pairs: pinned rows (mass 0) zeroed, QR + SVD rank cut (RANK_TOLERANCE), bilinear projection (D Q)^T A Q with a
  symmetrizer (the stationary quotient of a complex symmetric pencil), else Galerkin; polish steps append the
  mass-scaled residuals of the wanted pairs (Gram-Schmidt twice, columns already in the span dropped). Non-finite mu
  (directions with a singular projected mass) are dropped. Eigenvalue accuracy is the projection's: residuals are
  reported per resonance and warned above RESIDUAL_WARNING (no machine-precision guarantee as ARPACK's).
- Counters: eigen_solves, time_steps, sampled_frequencies, basis_columns, operator_applications, step_seconds,
  projection_seconds, eigen_last_residual. Time recipe: work.factorizations = time steps of one run (no factor),
  work.rhs = sampled frequencies; guesses.
- Not yet: em_fdfd_3d (the 3D FDTD engine on the FDFD domain), elastic pencils, a Bloch phase (complex FDTD fields),
  subpixel / conformal / sheet models (the FDTD would differ from the FDFD operator), several targets / bands.

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
- Compressed (mumps_blr, `HOST_DIRECT_COMPRESSED_BACK_ENDS`; module header): a double MUMPS factor with ICNTL(35) =
  BLR_ICNTL (set between set_matrix and analyze: the analysis needs it) and CNTL(7) = BLR_TOLERANCE (re-set before
  every factor: python-mumps keeps one instance per dtype). Its probe misses ACCEPT_RESIDUAL by design, so it always
  runs the double GMRES-IR path; a GMRES miss refactors full rank once (`_next_attempt`, sticky
  `_compression_rejected`, last_compression_fallback / compression_rejection) before raising. Name helpers:
  `library_of` strips `_NAME_SUFFIXES` ('_mixed', '_blr'), `host_direct_precision` is 'double', `host_direct_compressed`;
  a new suffix goes into `_NAME_SUFFIXES` or every helper misreads it. Kept analyses and `previous=` refactors key on
  the compression (a full-rank analysis is never adopted by a compressed request or the reverse). Estimators:
  `superlu_bytes_scale` x BLR_FACTOR_FRACTION; time recipes HOST_DIRECT_COMPRESSED_FACTOR_TIME /
  HOST_DIRECT_COMPRESSED_FULL_SOLVES (sparse_direct; arpack reads them): all uncalibrated. Unverified: that
  python-mumps' instance exposes `cntl` (a RuntimeError naming the wrapper otherwise) and the INFOG fields of a BLR
  factor (INFOG(29) may count full-rank entries).
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
- Not yet: out-of-core (MUMPS ooc / ICNTL(22), PARDISO iparm[59]), BLR memory estimates (ICNTL(38), INFOG of the
  compressed factor), a mixed BLR factor, a two-level factor (em_fdfd_3d order 2 on an order-1 coarse factor: ND1 sits
  inside ND2, so the prolongation is sparse with +-1 entries on an edge's two dofs and the face dofs around it, not an
  injection; scalar P2 on P1: corners 1, edge midpoints 1/2), MPI, user orderings,
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
