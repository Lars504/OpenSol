# nvmath-python / cuDSS: notes for GPU sparse direct solves

What was learned putting cuDSS (NVIDIA's GPU sparse direct solver, via nvmath-python) under the shift-invert
eigensolver and the direct linear solves (`open_sol/solvers/cudss_factor.py`, the one wrapper: used by
`arpack_shift_invert` factorization `cudss` / `cudss_mixed` and by `sparse_direct` back_end `cudss` / `cudss_mixed`).
Environment: nvmath-python 1.0.0, cuDSS 0.8 (`nvidia-cudss-cu13`), CuPy 13.3 on CUDA 12.x, Windows, consumer GPU.

## Why cuDSS at all
CuPy has no non-Hermitian `eigs`, no shift-invert mode and no reusable device sparse LU (`cupyx...splu` factors on the
host with SciPy; `spsolve` is cuSOLVER QR, refactorised every call). cuDSS gives a device factor reused for every solve.
ARPACK still runs on the host (scipy `eigs`): one vector to the device and back per application (~1 ms at 2e5 unknowns).

## API essentials (read the installed source: `site-packages/nvmath/sparse/advanced/direct_solver.py`)
- `DirectSolver(a, b, options={...})`: `a` a cupyx CSR with `b` a cupy (n,) or column-major (n, k) array, or a scipy
  CSR with a numpy `b` (only the pairs scipy-numpy, cupyx-cupy, torch-torch are accepted; host operands are copied to
  the device and force blocking). Indices int32 here (nvmath also takes int64 when unbatched). Then `plan()`
  (reordering + symbolic, on the host), `factorize()`, `solve()` -> a new array of b's package. `free()` (or a `with`
  block) releases the factor; it has no `__del__`, so wrap it in an object that frees.
- New right-hand side: write into the original `b` buffer in place (`b[...] = rhs`) and call `solve()` again; cuDSS keeps
  the pointers of `a` and `b`, so keep both arrays alive. `reset_operands(b=new)` is also cheap for device operands
  (pointer swap, plan kept); only `a=` drops the plan (below).
- `options={'blocking': True}` makes every phase synchronous (simple timing, no stream bookkeeping). The default
  `'auto'` (non-blocking for device operands) also installs an async workspace allocator on the device's stream-ordered
  memory pool (not CuPy's pool either).
- Symmetric: `sparse_system_type=DirectSolverMatrixType.SYMMETRIC` (complex symmetric A = A^T; HERMITIAN is separate)
  with `sparse_system_view=DirectSolverMatrixViewType.LOWER` and pass the lower triangle (diagonal included) as a cupyx
  CSR (`cudss_factor._lower_triangle` keeps explicit zeros, so the pattern stays value-independent).
- After `plan()`: `solver.plan_info.memory_estimates['peak_device_memory' | 'permanent_device_memory']` (a numpy
  recarray; also `*_host_memory` and `hybrid_min/max_device_memory`). Use the peak to refuse oversized factors before
  `factorize()`. With `blocking=True` cuDSS memory is cudaMalloc'd **outside the CuPy pool** (invisible to pool budgets
  and pool-based memory checks).
- `factorize()` returns `solver.factorization_info`: `.info` (non-zero = zero pivot / failure), `.lu_nnz`, `.inertia`
  (int array [n_positive, n_negative] pivots of an LDL^T: `CudssFactorization.inertia()`, Sylvester counts in
  `arpack_shift_invert.count_below`), also `.npivots`, `.diag`.
- Tuning: `solver.plan_config` (`reordering_algorithm`, `matching_algorithm`, `pivot_type`, ...),
  `solver.factorization_config.pivot_eps`, `solver.solution_config.ir_num_steps` (cuDSS's own iterative refinement).
  `PivotType.PIVOT_BUNCH_KAUFMAN` with SYMMETRIC raised NOT_SUPPORTED. Enum names: `nvmath/bindings/cudss.pyi`.
- Pass a quiet `logger` in options (`cudss_factor` sets it to ERROR): nvmath logs every phase at INFO, a WARNING at
  every construction without a threading layer, and a WARNING on `reset_operands(a=)`; without a logger it uses the
  root logger.
- Matrix types (`DirectSolverMatrixType`): GENERAL, SYMMETRIC, HERMITIAN, SPD, HPD; views
  (`DirectSolverMatrixViewType`): FULL, LOWER, UPPER. `CudssFactorization(positive_definite=True)` (real only) uses SPD
  with LOWER (Cholesky); a failed status or probe re-plans once as SYMMETRIC (LDL^T) and stays there.
- Plan reuse: write the new values into the planned device CSR's `data` in place, then `factorize()` / `solve()`
  (nvmath's own advice). NOT `solver.reset_operands(a=new_csr)`: with device operands nvmath 1.0 sets
  `solver_planned = False` there, and the next `factorize()` raises "Factorization cannot be performed before plan()
  has been called" (selftest solver_sparse_direct_cudss). `CudssFactorization.refactorize(new_operator)` does this when the canonical CSR pattern (duplicates summed, explicit zeros kept) and
  the matrix type are unchanged, else re-plans. Keep patterns value-independent: scale CSR values entry by entry, never
  through scipy products (`diags @ A` drops entries that cancel to exactly 0, changing the pattern). Unmeasured:
  the factorise-only time vs plan + factorise at the 2e5 size (expected: the ~2.5 s host plan saved).
- Device values (plan G5): `CudssFactorization.refactorize_device_values(values, indptr, indices)` takes cupy values on
  the planned host pattern and does the equilibration, lower-triangle take and value refresh on the device (index maps
  built once per plan); a new pattern / kind, or the LDL^T re-plan after a rejected Cholesky, goes through one host copy.
- Unused here: hybrid memory (`execution=ExecutionCUDA(hybrid_memory_mode_options=HybridMemoryModeOptions(
  hybrid_memory_mode=True))`, factors kept in host memory) and hybrid execution (`execution='hybrid'`, single rhs, no
  batching). A possible alternative to refusing factors over the device budget; untested.

## Pitfalls hit
- **Threading layer crash**: passing the wheel's `nvidia/cu13/bin/cudss_mtlayer_vcomp14064_0.dll` as
  `multithreading_lib` solved correctly, then killed the process with access violations (no Python traceback; find it
  with `python -X faulthandler`). Planning single-threaded costs only ~2.5 s at 2e5 unknowns, so leave it out. With no
  `multithreading_lib` nvmath falls back to the environment variable `CUDSS_THREADING_LIB` (used when it names an
  existing file), so keep that variable unset too.
- CUDA 12 CuPy + CUDA 13 cuDSS in one process works (driver 13.0); the probe residuals were ~1e-16.
- cuDSS pivots only within supernodes and perturbs tiny pivots: check every new factor with one probe solve's
  residual and enable `ir_num_steps` if it misses. Equilibrate (R A R, R = |diag A|^-1/2) even in double: rows of very
  different scale (identity rows of pinned cells next to D-scaled rows ~1e16 larger) gave a residual 0.13.

## Performance (220,500-unknown complex 3D curl-curl, Mie eighth sphere)
| | factorise | solve (1 rhs) | factor memory |
|---|---|---|---|
| complex128 LU | 24.6 s | 31 ms | 4.3 GiB |
| complex128 LDL^T | 13.9 s | 31.6 ms | 2.7 GiB |
| complex64 LDL^T (equilibrated) | 1.1 s | 18 ms | 1.4 GiB |

- LDL^T halves factor time and memory but **not** solve time: the solve reads L twice (= L + U bytes). One-rhs solves
  are bandwidth-bound (~175 GB/s here).
- Multi-rhs solves barely help (16 rhs: 22.8 ms per vector): float64 throughput on consumer GPUs. Block Krylov is not
  worth it here.
- Single precision is ~12x faster to factorise on consumer GPUs (slow float64).

## Making single precision usable
- A complex64 factor of the raw CPML operator was useless (error 0.38, refinement diverging): entries spanned 4e5.
- **Symmetric diagonal equilibration** E = R A R, R = |diag A|^-1/2 (keeps complex symmetry) fixed it: unrefined error
  9e-5, each float64 refinement step (r = c - E y in complex128, y += solve32(r)) gains ~3e3 (3e-8, 2e-11, 2e-13).
- Scale each rhs to unit norm before casting to complex64 (physical magnitudes underflow float32).
- Mixed refinement does not make solves cheaper (k+1 single solves per accurate solve); the gain is factor time and
  memory, plus loose solves where the caller can tolerate them. Measured on the Mie run (subpixel eps: unrefined
  residual 4.3e-4, worse than the staircase benchmark): run 37.6 s -> 22.4 s, factorise 13.9 s -> 1.1 s, device 2.8 ->
  1.5 GiB, ARPACK 249 -> 195 applications at the 1e-6 floor, Rayleigh quotients moved k^2 by <= 6e-7, lambda / Q
  unchanged to 6 digits. For eigenproblems: loose solves (1 step), ARPACK at
  ~1e-6, then the float64 Rayleigh quotient u^T D A u / u^T D M u per eigenvector restores eigenvalues to ~1e-12 when
  D A is complex symmetric (stationary bilinear form).

## Symmetrising FDFD / Yee operators
The Yee curl-curl with SC-PML is complex symmetric after a diagonal row scaling D = the stretched dual volume of each
unknown (dual length x stretch along its own axis, cell width x stretch across, halved on antisymmetric mirror-plane
normals): `operators3d.face_stretches` x `face_dual_volumes`. Factor D A and apply D B; eigenpairs of (D A)^-1 D B =
A^-1 B are unchanged. Bloch phases break it (Hermitian-like). Always verify: `relative_asymmetry(D A) <= 1e-12`.
2D (E_z / H_z Helmholtz): each axis term is -W_e G^T W W_h G, so D = the stretched cell area dx s_x dy s_y (polar: times
the stretched centre radius), `operators.stretched_cell_areas`. A shift exactly on a lossless resonance makes A singular:
the eigen solves shift sigma 1e-4 above the target (`eigenfrequency.SHIFT_RELATIVE_OFFSET`).
Transposed (adjoint) solves: cuDSS exposes no LU^T solve here, so `CudssFactorization.solve(transpose=True)` works only
with a symmetric factor: A = A^T (same solve) or `symmetrizer=D` (factor S = D A; A x = b is S x = D b, A^T x = b is
x = D S^-1 b). An LU factor raises NotImplementedError.

## sparse_direct back ends (linear / shifted roles)
- back_end `cudss` / `cudss_mixed` of `sparse_direct`, opt-in per role (`SolverType.opt_in_settings`): only roles whose
  `setting_choices['back_end']` list them (today `definition_helpers.DIFFUSION_DIRECT_BACK_ENDS`: heat,
  electric_currents, electrostatics, magnetostatics, solid_mechanics(_3d) through `DiffusionDirectSolver` (real
  Cholesky, plan reused across re-assemblies, solves on the device), and membrane). The same linear runtimes also
  factorise non-SPD systems through `factorize_host(symmetric=True, previous=)`: complex symmetric LDL^T per frequency
  (`fem/harmonic.py`, `mechanics/frequency_response.py`, squeeze film) and the real indefinite piezo operator. Details
  in `open_sol/solvers/sub_structure_conventions.md`. Not recyclable (no GMRES on an earlier factor): pass the old
  factor as `previous=` to refactorise instead.
- Real `cudss_mixed`: float32 factor of R A R + float64 refinement to 1e-12; contraction ~ kappa(E) eps32 per step, so
  high-contrast diffusion (mu or conductivity ratios 1e3-1e5) may be refused at the probe: measure single_residual.

## Benchmark scripts
`.temp_scratchpad/cudss_rhs_benchmark.py`, `cudss_precision_benchmark.py`, `cudss_single_variants.py`, `cudss_probe.py`
(staged crash isolation), `cudss_2d_check.py` (2D eigen models, superlu vs cudss). Self-tests: `solver_arpack_cudss`
(gpu), `solver_sparse_direct_cudss` (gpu: Cholesky, plan reuse / re-plan, symmetrizer transposed solves, real LU),
`eigen_inertia_count` (LDL^T inertia, one plan over eight shifts), `em3d_fdfd_operator_identities` / `em_fdfd_cpml_complex_symmetry` (symmetrizers, cpu);
every em_fdfd_3d model-run self-test, the Mie example and the Igor nanobeam experiment run cudss_mixed on the gpu.
