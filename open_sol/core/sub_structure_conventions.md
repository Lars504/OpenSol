# open_sol/core — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Gotchas of core modules only; general array /
coordinate rules stay in the root file.

## Backends (`backend`)
- Without cupy, `cp` / `cupyx_sparse` / `cupyx_ndimage` are stand-ins (module docstring): module-level kernel
  definitions still import, `isinstance(x, cp.ndarray)` is False, any other call raises `BackendUnavailable`.
  `gpu_available()` is a driver query (worker thread only); `cupy_installed()` is GUI-thread safe.
- The run backend lives on the Domain (`domain.backend`, `domain.xp`, from its MaterialGrid); the builder calls
  `domain.use_backend(schema.solver_backend(spec))` right after `make_domain`, before any array exists. Host copies
  come from `to_host` / `host_copy`, device syncs from `domain.backend.synchronize()` (nothing on the cpu).
- Never `xp.copyto(..., where=None)`: cupy reads it as no mask, numpy as False (nothing written); branch on a None
  mask (MaterialGrid.assign / assign_window).
- Every physics type declares 'cpu' (`PhysicsType.backends`, enforced by validate); a coupling runs on the cpu once
  both partner types declare it. gpu only: the half_storage steppers and the cusolver_qr / cuDSS back ends (their
  SolverType `backend_misfit`). Engines on the cpu record probes per FieldProbe (`FieldProbeRows` is gpu only), and
  `replace_same_time` stays on that per-probe path (FDTD never uses it).

## GPU memory (`gpu_memory`, `host_memory`, `memory_estimate`)
- `set_budget()` runs before every job (worker: after `release_session`; `api.run`: at start); when it raises the job
  fails (Issue at ('model',)), never runs without a pool limit (the Windows sysmem fallback is 10-100x slower).
  `budget_device_info()` reuses its memory query: one per run. Pool counters are process-wide
  (`BuildSummary.session_gpu_bytes` is one build's share); `memory_status()` is a driver query: <= ~1 Hz.
- Sizing to the free memory at run time (line_block chunks, eigen batches) asks `pool_room_bytes(release_cached=True)`,
  not budget - pool used: the budget never sees outside-pool memory (cuDSS factors of other physics). A mid-run release
  of the pool's cached blocks goes through `release_cached_blocks()` (also inside pool_room_bytes), never a bare
  `free_all_blocks()`: it keeps the reserved high water that the run memory measurement (`pool_high_water_bytes`)
  reads. Between jobs: `release_all` (gc + cuFFT plan cache, cleared only when non-empty: cupy's `clear()` of an
  empty cache is a full `gc.collect`).
- Outside the cupy pool: the cusolver_qr workspace (`sparse_direct.check_solver_size`), every cuDSS factor (its own
  plan estimate in `cudss_factor`; estimated as MemoryEstimate gpu_external) and MUMPS / PARDISO factors (host memory
  outside numpy, guarded against HOST_FRACTION of `host_memory.available_bytes()`, no guard when None): numpy-based
  checks (`owned_bytes(backend='cpu')`, the cpu MemoryLedger's tracemalloc) do not see them. `Simulation.memory_report`
  counts a shared array for its first owner and lists CPU factorisations (`host_factor_bytes`) separately; the model
  path reports those only in the study log.
- cpu backend: `MemoryLedger(backend='cpu')` traces numpy bytes with tracemalloc (the builder `stop()`s it in a
  finally) against the same gpu_steady / gpu_build_peak estimate; there is no run-memory record on the cpu (tracing
  would slow the run). `memory_estimate.diffusion_system_estimate` books the host factors of its `direct` options (all
  heuristics uncalibrated: refit from summary.json host_direct_lu_entries / host_direct_peak_bytes); a cuDSS 'linear'
  back end is the caller's `fem.diffusion.cudss_diffusion_estimate(input_copy=False)` (core may not import
  solvers.cudss_factor).
- `release_all`'s full `gc.collect` walks every tracked object (~25 ms): a sweep (`study._run_sweep`) freezes the gc
  after its first value (`freeze_collect_baseline` / `thaw_collect_baseline`), never undoing a freeze of its caller
  (selftest_cache freezes for the whole selftest).

## Accepted host syncs (the root rule: none per step in hot loops)
A new accepted sync joins this list.
- Study: the build's Courant check; one `domain.backend.synchronize()` per study chunk and one at the end of each run
  (`study_seconds`); the builder's material range checks (model sub file, Schema).
- DiffusionSystem (heat, potential): CG reads one at the start (||b|| and ||r0|| together) and one per check
  interval (`solvers.pcg` CG_CHECK_INTERVAL / MULTIGRID_CHECK_INTERVAL; the gpu anchor-code comparison of a
  re-assembly is read right before it); one per `refresh_diagonal` / `refresh_capacity` (pinned / anchor cells
  unchanged) and `refresh_held_values` (held count); one per radiation term and per steady radiation
  re-linearisation (its convergence test).
- EM: the FDFD solve (scipy_splu: one rhs / solution transfer each); one HarmonicModeMonitor transfer per solve; one
  grid-row transfer per FDFD port mode re-solve (per reassembly) and per FDTD mode-port recalibration after the first
  step (material drift check, re-solve above `mode_ports._RESOLVE_DRIFT`); the E_z ohmic-box refresh per
  `materials_changed` of sigma_e.
- laminar_flow: the velocity check once per lattice check interval (>= 100 steps).
- Mechanics: per Newton iteration of a device tangent (fem.elasticity3d.TangentAssembler3D(device=True)) the
  displacement up / force down and the operator norm of `CudssFactorization.refactorize_device_values` (its scatter
  sums are cupyx.scatter_add, never cupy.bincount, which syncs on max(x) per call); per assembly with the
  solid_mechanics_3d multigrid preconditioner (`multigrid3d.BrickMultigrid.build`) the power-iteration dots of every
  level and the coarsest probe reads (never inside a V-cycle).

## Kernel twins (`kernels`)
- A cupy kernel that a cpu-capable path launches is a `twin` / `raw_twin` / `real_raw_twin` (dispatch on the first
  array argument: cupy -> the kernel, numpy -> the host twin), never a hand-rolled `X_host if on_cpu else X` branch.
  Names are '<package>.<kernel>' (a duplicate raises); the twin keeps the old kernel name so call sites never change.
- Host twin contract: the same positional arguments as the cupy call (Elementwise: inputs then outputs; Raw: the args
  tuple, grid / block ignored), outputs written IN PLACE (`out[...] =`, `+=`: outputs are often box / line views,
  never rebind them), the same dtypes (float64 where the cuda body uses double; numpy keeps float32 for float32 arrays
  with numpy / Python scalars), the same branch rules (periodic wraps, own faces) and broadcasting (1D spacings per row
  / column). Reductions return a 0-d numpy array (callers take float()). Twins may allocate per call; numpy scalars of
  the GPU path work unchanged (`int(...)` them before slicing). A wrong twin shows as an analytic-case or parity
  failure of the CPU-vs-GPU self-tests (`selftests_backend`).
- numba (optional, never required): only for per-step / per-iteration kernels (grep `numba=`), built lazily at the
  first host call; a build failure or a numba typing / dispatch error falls back to numpy for good with one warning,
  any other exception re-raises (in-place outputs may be partly written). Host-only numba loops without a cupy kernel
  are `host_twin(name, numpy_form, numba_factory)`, never a hand-rolled try / except around njit; modules that may not
  import core.kernels at module level (pcg, host_direct, fixed_pattern) call it inside the function.
- numba traps: `@njit(parallel=True)` types every branch, so every array a numba twin indexes has the loop's
  dimensionality even when unread (placeholders (1, 1, 1)). The prange index is uint64: a conditional mixing it with a
  signed int (`j if c else i`, `0 if c else i + 1`) unifies to float64 and fails as an index (the twin then falls back
  to numpy), so every numba twin rebinds it at the loop top: `for prange_i in numba.prange(n): i = np.int64(prange_i)`.
  Variant flags are closure constants of the factory; a Python wrapper may unpack / reshape the args (absent arrays ->
  same-type dummies). Each numba twin gets a CPU case `<...>_numba` (`selftest_helpers.numba_twin_result`).
- Precision variants (2D FDTD 'double', mechanics kernels3d / multigrid3d): `real_raw_twin` compiles a RealRawKernel
  per dtype of args[dtype_argument] (intrinsics such as __fmul_rn need an `#if REAL_IS_DOUBLE` branch);
  ElementwiseKernels take the field dtype as a type placeholder (T / F) and cast float32 material inputs explicitly.
  Their scalars are numpy scalars of the field dtype (cupy casts Elementwise scalars to T, so a float32 dt silently
  rounds the step; a RawKernel reads the wrong bytes). Numpy twins lift float32 material arrays to the output dtype
  first (`astype(out.dtype, copy=False)`: under numpy 1.x value-based casting float32 array x float64 scalar stays
  float32).
- A new explicit engine on the cpu: every kernel a twin, every array module from the Domain / an input, `xp=` into the
  objects that allocate (fields, CPML, sources, monitors), `PhysicsType.backends = ('gpu', 'cpu')`, refuse what stays
  gpu-only (half_storage: explicit_kernels backend_misfit), a `selftests_backend.backend_parity_case` wrapping its
  analytic cases.

## Half storage (`half_storage`)
- float16 arrays (em_fdtd_3d / elastic_waves fields, laminar_flow populations) are touched only by own kernels:
  RawKernels through `storage_preamble` (FLOAD / FSTORE), ElementwiseKernels with a T placeholder through
  `STORED_ACCESS` called with `half_bits(array)` (the uint16 view). cupy's own float16 kernels (astype / copyto /
  arithmetic / fills INTO float16) fail at ptxas on some CUDA / cupy pairs ("Unresolved extern function
  '_ZN6__halfC1Ef'"); bit-level ops on the uint16 view are fine (zero fill, `*= bool mask`, sign flip by xor 0x8000).
  Reading float16 into float32 by cupy (astype, `(float)` in a kernel) still works there (convert those readers too if
  one fails).

## Graded meshes (`mesh`)
- Graded grids are rectilinear: a refinement fills whole columns / rows (3D: also z layers, same march). Refinements
  keep their target and the ramps (growth 2 - 1/r per cell) lie outside them; `meshing._graded_axis_cell_bound`
  bounds the same profile piecewise (overlapping refinements count once): change both together.
- Rectangle edges and band snap faces are snap nodes (exact cell edges, so masks see no staircase); each segment between
  them is rounded to whole cells and rescaled to its length (never a short last cell: it would set dx_min, hence dt),
  the correction tapered toward the snap nodes: adjacent ratios there can exceed max_ratio by ~3/n^2 for an n-cell
  segment, and an object spanning no whole number of target cells gets dx_min up to ~0.7 / n (relative) below target,
  so a self-test pinning dx_min sizes its box in whole target cells (e.g. heat_3d_graded_z_robin).
- Outer grading (`outer_dx`, `outer_ratio`; model side: model sub file, Meshing): outside `outer_core(axis)` cells grow
  from the refinement edge straight to outer_dx, so dx_base matters only inside the core, while PML cells still grade
  toward dx_base (they shrink beside outer cells). `window_edges(axis)` = float64 edges with snaps and ends exact (the
  float32 widths drift ~1e-7 relative): code needing exact edges reads it (the builtin tetrahedron mesher).
- Padding: PML cells grow outward from the window edge cell by the axis ratio up to pml_dx (default dx_base), but the
  CPML sigma grading assumes dx_base cells, so a long ramp (finely meshed edge) absorbs somewhat less. `ghost_faces`
  (an EM mirror's ghost cells, passed by Domain.from_mesh and model.meshing) copy the edge cell's width exactly: a
  mirror image needs equal widths across the plane.

## Grid transfers (`grid_transfer`)
- Built through `model.mesh_transfer.make_transfer` (a triangle mesh on either side: `TriangleTransfer`, fem sub
  file); edges are the Domains' full-grid edges in one frame (window lower corner = 0) and only source WINDOW cells are
  read. 'conservative' keeps the window integral exactly in the float64 `host_matrices` (the applied float32 ones to
  ~1e-7 relative) and leaves target padding 0; 'interpolate' extends the edge values into padding and has no
  angle-wrap seam handling.
- Real arrays only (complex raises). `apply` allocates its products per call (fine at the coupling cadence; not
  timed: a time needs a device sync). Estimate: `memory_estimate.grid_transfer_estimate`.
- `MaterialGrid(n)` (ny None) is the 1D per-element grid of a triangle mesh: code taking `materials.nx / ny` as a cell
  grid checks the domain's `grid_kind`.

## Mirror planes (`mirror`, numpy only)
- Model mirror planes = the EM bc_<face> 'symmetric' / 'antisymmetric' faces plus the faces of the enabled
  `MIRROR_FEATURES` (a magnetic-only model has none, a membrane half model has its symmetry face). Per physics
  (`mirror_faces(spec, physics)`) a face the physics does not set itself unfolds even, so its own face there must be
  insulated (`definition_helpers.mirror_face_issues` warns), except the `FACE_FEATURE_SIGNS` types, signed by their own
  face feature on the plane.
- Totals take `mirror_faces`, displays (`unfold_field` callers) `unfold_faces`: it adds the x_lo rotation axis of an
  axisymmetric `AXIS_UNFOLD_TYPES` physics, which ring weights already count, so it never doubles a total. A new axis
  type needs every field even under the axis image (E_r / A_phi flip).
- `FieldInfo.parity` names a renderer field's image rule (`image_sign`; '' = even when unsigned, else unknown:
  `unfold_field` raises); `staggered` = axes of hi-face samples (plane value 0 when odd, else the nearest sample: exact
  only for odd staggered components); `plane_signs` (`plane_signs_text`) replaces the face signs for one field (the
  nlfd channels, nlfd sub file). Fields are never mirrored twice: `FieldData.unfolded` (parity-sector snapshots are
  stored unfolded; results_io saves it).
- Every total over a selection, line, plane or box touching a plane goes through `touch_factor` (2 per touched face;
  mirror-even integrands; callers: grep `touch_factor`). heat reports no feature totals.
- Parity sectors: `sector_specs` = 2^k spec copies (MIRROR_FEATURES swapped and split per face by
  `_swap_mirror_features`; a type without features of both signs, heat, raises, reported by study validation
  `_parity_sector_issues`). The drawn source alone = the mean of the unfolded sector fields (linear signed components
  only) and mean / 2^k of quadratic totals over regions / surfaces symmetric about every plane (never a line off a
  plane); `combine_sector_result` holds the rule per feature result kind (None = not combined). Study loop:
  `model.study_sectors` (model sub file "Parity sectors").

## Cut geometry (`cut_cells`, host numpy)
- Everything works on a membership test inside(x, y) -> bool of float64 host arrays (2D names wrap the
  any-dimension `segment_*` / `rectangle_free_areas`: em_fdtd_3d's conformal PEC passes 3D points). Callers locate only
  links between cells whose centre tests differ: a cut no centre sees stays staircased; features thinner than
  1 / EDGE_SAMPLES of a segment vanish. `free_areas` takes the row axis from the caller: pick it by the face fractions
  (rows across the steeper boundary), as `rectangle_inside_fractions` and physics/em/conformal_pec do.
- Polar / spherical windows: callers pass logical coordinates (x = r - radius_min, y an angle) and map them in
  inside(x, y) (boundary_locator with a DomainMap); fractions are shares of the logical link / face / cell (areas
  O(dr / r) off the metric weighting inside a cut cell); `outline_pieces(..., radius_min=)` returns the local physical
  frame. Links / faces across an angle_wrap seam are never cut (staircased).
- Honouring types of `solver_defaults.conformal_boundaries`: grep `conformal_boundaries=True` in the definitions
  (`schema.conformal_boundaries_physics` adds `conformal_dimensions` / `conformal_coordinates`); per type: em3d sub file
  (Dey-Mittra H faces), em sub file "Conformal poles" and FDFD / em_modes pec_domain (the FDFD weights keep the operator
  symmetric; `HarmonicSystem.pec_cells` = the fixed cells only), fluid sub file (Bouzidi), DiffusionSystem below. The
  locator: model sub file "Conformal boundaries".

## Coordinate metric (`metric`, numpy only)
- On a polar Domain (`Coordinates`) widths, edges, centres, masks and `pos_to_cell` stay logical (x = r - radius_min
  [m], y = angle - angle_min [rad]); `Domain.dx` / `grid.dt` mix units and mean nothing in 2D. Polar + the physics'
  axisymmetric flag = spherical (r, theta).
- `angle_wrap` (planar (r, phi) only): DiffusionSystem (real) reads `wrap_sign` (u beyond y_hi = wrap_sign x u at
  y_lo); EM physics (polar FDFD, em_fdtd_3d sectors) read it only as "faces joined": their phase is bloch_ky = the
  azimuthal order m [1/rad], F(phi + span) = F(phi) e^{-j m span} (antiperiodic C_N sector: m = N / 2).
- `Coordinates.line` (1D models) keeps the ordinary metric on a one-cell-wide window, so operators need nothing;
  consumers read `line` only for units / labels (`label()`, potential `flux_unit` per m^2 on a slab).
- `window_metric` / `window_metric_3d` is the one FV geometry: curvature enters only through areas and volumes
  (uniform sources exact, source-free radial profiles second order). Kernels gain 1D `Separable` factors, never
  per-cell metrics.
- `PhysicsSolver.coordinate_grids` (checked by `Simulation.add_physics`) agrees with `PhysicsType.coordinate_grids`
  except FDFDPhysics and EMPhysics3D, polar in scripts only (em sub file "Polar (r, phi) grids", em3d sub file
  "Cylindrical sectors"). `to_cartesian` / `from_cartesian` map logical positions to the drawing plane and back.
  Object-perimeter fluxes on 2D polar windows weight faces by `x_face_volume_ratios` / `y_face_volume_ratios`
  (`definition_helpers.perimeter_flux_weights`).
- 3D: a polar Domain3D is a cylindrical (r, phi, z) sector (`WindowMetric3D`; the 3D DiffusionSystem, em_fdtd_3d
  through `yee_radii`). Unlike 2D its `grid.dt` is meaningful (innermost arc as the phi width); a first cell face at
  r < 0 (radius_min below the x_lo padding) raises.
- Components stay in the grid frame: `renderer.polar_field_infos` relabels _x / _y / _xx / _yy as radial / angular;
  a polar-ready renderer with such components sets it as its instance field_infos (potential does).

## 3D contracts (`grid3d`, `domain3d`, MaterialGrid)
- `Domain3D` has the 2D Domain's accessors per axis (3-tuples), `plane_extent(axis)` for a cut, no rect_mask /
  circle_mask (scripting is 2D). Flat index `(i * ny + j) * nz + k`: kernels get k-stride 1, j-stride nz, i-stride
  ny nz.
- z cells (`meshing._z_cells`): graded meshes march z like x / y (`add_refinement_band('z', ...)` per object
  z_extent + margin; a band refines whole z layers, never the xy profiles). z PML counts come from
  `boundary_config.npml_sides['z_lo' / 'z_hi']` (missing = 0: the 2D EM BoundaryConfig gives none, so a 3D boundary
  config adds them and keeps `differences()`).
- Materials: `MaterialGrid.assign_window` (prefer it over full-grid `assign` in 3D); x, y dependent values are
  (wx, wy, 1) arrays, z-dependent ones are written one z plane at a time. A `BoxValues` is undefined outside its box:
  accumulating consumers add it on the box only (`add_masked_to` / `copy_masked_to`; `DiffusionSystem.add_masked` /
  `copy_masked`), consumers handing window arrays to a system take `DiffusionSystem.window_values`.
  `MaterialRegion.mask` is None in 3D unless a state-dependent property needs it (its DynamicMaterialUpdate).
- BuildContext 3D: `object_box_mask` (cached, read-only), never `object_window_mask` (raises); `region_window_mask`
  over `region_mask`; `perimeter_mask / perimeter_weights` are 2D only. Prism z intervals are half-open with float32
  compares, spheres closed.
- Renderers of 3D physics: `device_field(component)` = the (wx, wy, wz) window array (a view where possible); an
  optional `device_plane(component, axis, index_window)` avoids materialising derived 3D fields (reads: model sub file
  "3D field access"). `Session.evaluate` 'integral' is a true volume integral in 3D.

## Harmonic probe record (`harmonic_probe`, numpy + core.backend)
- Frequency-domain FEM solvers record point phasors through `HarmonicProbeRecord` / `harmonic_probe_result` (series
  `<c>_re/_im/_abs/_phase` vs Frequency [Hz]), so `two_wave_fit` / `pole_fit` read them all alike; series built
  elsewhere use `phasor_series`. The FDFD HarmonicProbe (wavelength abscissa) stays separate.

## Resonance finder (`resonance`, host numpy; the shared row is `model.definition_resonance`)
- The analysis runs only in `ResonanceMonitor.result(waveforms, unit, dt)` (cached per (samples, start, dt)): dt is
  passed then because a physics may fix its step only at finalize (elastic_waves). `sample_delay_steps` per component
  (EM H: 0.5; probes stamping their own times, AcousticProbe velocities: 0).
- Its decimation is NOT the DFT `sample_interval`: the trace is mixed to the band centre, anti-alias filtered and
  decimated on the host (module docstring); the first n_taps samples after the start go to the filter, so low-Q modes
  decay there.
- `result` never raises for a short run (the study end computes every table result, `_feature_results`): failures go
  to the 'note' extra with empty rows.
- Memory: `analysis_host_bytes` (host temporaries of one analysis); the growable probe traces are counted by the
  physics estimator through `model.definition_resonance.point_trace_items` (a FeatureType estimator does not see the
  study duration). The row's apply hands each point to the physics' `add_probe` as a one-object selection, so
  add_probe keeps its single-object rule.

## Run-time estimate (`time_estimate` pure Python, `time_profile`; time_estimation_plan)
- Ops carry sizes only (bytes, rows / nnz, unknowns, flops, calls); `price` maps each to one profile curve:
  Elementwise -> 'fused' ('fused_numba' on the cpu when measured) at max(launches x the curve at bytes / launches, the
  curve at all bytes): the curve repeats one call on arrays of its size (cache-hot below the cpu L2 / L3 cliffs), while
  an op's launches stream distinct arrays, so an op's bytes are the distinct bytes all its launches move (a loop over
  the same small arrays is a count of single-launch ops, not launches). Spmv 'spmv', Reduction 'dot' (+ sync on
  read_back), Transfer 'transfer' and Sync only on the gpu, HostLU / HostSolve 'host_{lu,solve}_{2,3}d[_complex]' (no
  complex curve: COMPLEX_LU_FACTOR x the real one), DenseFlops 'gemm' / 'host_gemm' (float64 products; `single`:
  'gemm_single', complex64 products at 8 n^3 real flops: consumer GPUs run them far above the float64 rate),
  BlockInverse = dense complex inverses, which do not follow the product rate (cuBLAS batched LU): one batched call on
  'block_inverse[_single]_<b>' (seconds over the batch at BLOCK_INVERSE_SIZES, log-log interpolated in the block size)
  or `looped` one matrix per call on 'matrix_inverse[_single]' (over MATRIX_INVERSE_SIZES), HostStream 'host_stream',
  PythonCalls 'python_call', DevicePlan / DeviceFactor / DeviceSolve 'cudss_{plan,factor,solve}_<d>d[_mixed]' (x
  scale; only 2D curves exist; DeviceFactor.device_values: 'cudss_factor_<d>d_device[_mixed]', the host curve on an
  older profile without it). A new op needs its curve in run_profile (or prices from an existing one); a profile
  lacking a curve moves the item to not_estimated.
- cuDSS curves (gpu, nvmath installed; `_cudss_curves`, {} on any failure): complex symmetric 5-point Laplacians
  (+0.1j, CUDSS_AXES cells per axis) through `sparse_direct.cudss_factorization(symmetric=True)` as the engines call
  it: plan = a fresh factor's plan_seconds (best of 2, double only: one curve for both precisions), factor =
  refactorisation on the kept plan (previous=: host equilibration, symmetry check, value upload, factorise, probe),
  factor_device = the same plan refreshed by `SparseDirectSolver.factorize_pencil` (DevicePencil; mixed check deferred
  to the solve; E7: ~4x below the host path in double, ~13x in mixed), solve = solve_host of one vector (mixed: refined
  to full accuracy). Real operators and other fills are the caller's scale (sparse_direct.cudss_2d_ops), never extra
  curves. `library_versions('gpu')` records nvmath and the nvidia-cudss wheel, so a cuDSS update invalidates the gpu
  profile.
- `MachineProfile.seconds`: log-log interpolation; below the first point its seconds (latency floor), above the last
  at least linear growth. run_profile makes curves non-decreasing (`_monotone`: each point clamped down to the next
  larger one, never up, so a slow first call cannot lift a curve); every benchmark has a warm-up call
  (`_best_seconds`; host factorisations best of 2). `_best_seconds` sizes its batches from a short doubling probe so
  each lasts ~minimum_seconds: never grow batches in x4 steps (a call just under the threshold quadrupled the task's
  wall time without changing the measurement). The progress lines are task wall seconds, not measured values. Each benchmark times the kind of call the engines make (gpu 'dot' =
  pcg's multi-block `device_vector_dot`; the study max-abs checks and harmonic FDTD energies reduce axis-first, ~1.5x
  that curve at 1e6 values; 'fused' = an ElementwiseKernel / numpy expression twins): an engine switching to another
  primitive needs its own curve.
- Stored profiles are invalidated by any edit of time_profile.py or of the solver code its curves time (`code_hash` of
  time_profile.py + `_TIMED_SOURCES`: solvers/sparse_direct.py, cudss_factor.py) or other library versions: editing the
  benchmarks or the cuDSS routes forces every user to re-profile. `library_versions` lists only what the backend's
  benchmarks run (numba on the cpu only: installing numba invalidates just the cpu profile, whose re-run then measures
  'fused_numba'); the load reason names the changed libraries. `run_profile(save=False)` for tests. Estimates still
  price on an out-of-date stored profile (`model.time_estimate.stored_profile`: `outdated=True`, text "(outdated gpu
  profile)", `TimeEstimate.outdated_text`); only the `time_profile_current` selftest judges it (kept out of this file so
  the fallback itself does not invalidate profiles).
- Profile tasks are the `report(text)` calls: each task's wall seconds go to `task_seconds` keyed by that text, and the
  progress line compares them with the stored file's (read raw, ignoring code_hash / versions). Renaming a progress
  text drops its previous time.
- Shared recipes: `explicit_step_estimate` (per step the memory estimate's gpu_steady read once + write_fraction
  written; frequency_domain also gpu_run; numba_items / numba_launches split the kernels with numba twins from the
  numpy-twin rest, priced 'fused_numba' / 'fused' on the cpu: an engine giving another kernel a numba twin widens its
  predicate); `diffusion_time_estimate` (the 'linear' role, assembly, right-hand side, solution update and probe
  samples, counted by the role's factorizations / solves counters (a refresh_diagonal / refresh_capacity counts as an
  assembly); transient vs quasi-static vs coupled stepping; a fixed-point / nonlinear series = one cold solve + warm
  ones; `held` = the type's `HeldValues`, turned into SolverWork.jacobi_condition / distinct_modes by
  TimeContext.jacobi_conditioning): a new DiffusionSystem type passes its HeldValues.
- `time_check` groups items by `seconds_counter`; an item without one joins 'study' = study_seconds minus every counted
  group, so a role timer must not also sit inside another group's timer (mechanics eigenfrequency clears its eigen
  items' seconds counter: the PCG timer already holds them).

## Damped iteration (`damped_iteration`; plans_overview decision 7)
- The one loop driver of nonlinear DiffusionSystem physics (electrostatics Poisson-Boltzmann Newton, magnetic B-H fixed
  point); no solver role, its knobs are physics settings. Array-module agnostic (numpy + core.backend.array_module_of).
- `DampingSettings(tolerance, max_iterations, relaxation, max_step, min_relaxation, anderson_depth=0)` (frozen,
  validated); `DampedIteration(settings, label).run(state, propose, scale) -> converged`: `proposal = propose(state)`
  must be a new array, never a view of state (the caller re-assembles / re-solves inside it); `change = max |proposal -
  state| / scale` (float or broadcastable array, e.g. V_T or |nu|; the one host sync); stop when change <= tolerance
  with `state[...] = proposal`, else `state += factor (proposal - state)`, factor = relaxation x min(1, max_step /
  change); relaxation halves (to min_relaxation) while the change grows, doubles back once it shrinks. Convergence uses
  the undamped change (a small relaxation never fakes convergence). Non-finite change raises FloatingPointError.
- `anderson_depth` m > 0 (magnetic B-H default): Walker-Ni Anderson mixing instead of the halving, step = beta f - (dX +
  beta dF) gamma, gamma the least squares of f / scale on the last m differences, beta = relaxation fixed, max_step caps
  the mixed step (one extra sync when finite); history (3m + 3 state-sized arrays) lives for one `run`; one transfer of
  the (m + 1) x m Gram products per iteration; the step is two GEMVs of gamma over the ring rows (no per-row loop), and
  only the newest weighted df row is rebuilt while `scale` is the same array object (or an equal number): a scale
  array must not change in place during one run. The state must stay meaningful under linear combinations (magnetics
  mixes ln nu, not nu). `AndersonHistory` is also the stationary study's loop-back acceleration (model sub file).
- Record: `n_runs`, `total_iterations`, `last_iterations`, `last_change`, `last_converged`, `n_unconverged`,
  `last_warning` (never cleared; an unconverged run returns False and keeps the damped state: the caller appends the
  warning to its `runtime_warnings`); `work_counters()` -> nonlinear_runs / _iterations / _unconverged (merge into the
  physics' `work_counters` only when the loop is in use).
- A Picard on a coefficient (magnetostatics B-H nu) keeps the state in the physics, not in a material array (that holds
  the curve's parameters): the assembly reads the state array, `propose` assembles, solves and returns nu(|B|), and
  `scale` may be the state itself.

## DiffusionSystem (`fv_diffusion`, used by heat and potential)
- `DiffusionPhysicsBase(PhysicsSolver)` is the shared wrapper (delegated tol / max_iter / compute_residual and
  convergence record, `set_face_bc`, `add_boundary_flux` into `_applied_source_window`, probes, PML edge extension via
  `_bind_window`); subclasses call `_init_diffusion` in `__init__` and set `_probe_field` / `_probe_label`. Plain
  attributes on a physics never reach the system: add new settings as properties here.
- Snapshot solutions: `DiffusionSystem.solution_state` / `restore_solution` (unknown, face rhs terms, dirty marks)
  are tied to `n_assemblies` (bumped per full assembly in `_keep_assembly`): a state of another assembly dirties the
  operator instead of restoring the terms, so a read re-assembles from the owner's restored inputs (heat sub file).
- `assemble` caches face conductances, outer face terms, owned c_xx / c_yy copies (`assembled_c_xx / _yy`) and, with a
  2D cross term, `cross` (`CrossQuads`); `flux_at_centres` (= `axis_flux_at_centres` + `cross_flux_at_centres`) /
  `dissipation_density` read only that cache and raise while `operator_dirty` (`heat_flux_magnitude` re-assembles
  first; potential calls them right after a solve, E = -grad V from the axis part alone, J / D from both).
  `flux_jump_indicator(flux)` (S7.7 A2, `DiffusionPhysicsBase.error_indicator`) reads the same cache: a physics whose
  shown flux is not the system's (potential driven terminals, magnetic H with remanence, the AC phasor) passes it via
  `_indicator_flux`.
- Fused re-assembly (`fv_diffusion_fused`; `FUSED_REASSEMBLY`, per system `fused_reassembly`): an assembly on the kept
  plain 5 / 7-point pattern (no phi wrap, c_xy term, insulated cut-cell walls or held edges; window-shaped c and C/dt)
  runs two grid-stride `real_raw_twin` kernels instead of build_laplacian + the elimination glue: entries (CSR values in
  the `_SlotLayout` order, interior / outer-face conductances and rhs into one buffer each, held / pinned masks, fixed
  values, rhs drive) and elimination (A_solve, Jacobi, rhs offset, fixed cells of the unknown); the anchor check sits
  between them. Arithmetic order = build_laplacian's (geometry as packed 1D factors, unit factors exact), so it differs
  only by FMA / summation rounding (selftest diffusion_fused_reassembly, cpu twins and gpu kernels vs build_laplacian).
  The first assembly (no pattern) and every other case take the build_laplacian path. A new operator term either goes
  into both paths or makes `_fused_reassembly_applies` refuse it; `_keep_assembly` stores what both paths keep.
  Run-time estimate: `diffusion_time_estimate` prices re-assemblies at the fused launch count.
- c_xy cross term (`build_anisotropic_laplacian`): per interior vertex, energy c_q/2 [(u_NE - u_SW)^2 - (u_NW -
  u_SE)^2], exact for linear u over the rectangle between the four centres on any widths; c_q = area-weighted mean c_xy
  of that rectangle, 0 unless all four conduct (c_xx, c_yy > 0) or are held; it adds +-c_q/2 to the diagonals, so every
  row sums to 0 (selftests heat_cross_conduction_box_graded, heat_cross_conduction_graded_convergence). No cross flux
  through outer faces; never drop the diagonal part (row sums then act as a spurious exchange with u = 0 at window /
  region corners). Definiteness on graded cells only for c_xy^2 < c_xx c_yy x 4 r / (1 + r)^2 per axis (r = neighbour
  width ratio), stricter than the continuum bound. Whether c_xy is nonzero is decided (one sync) at the first assembly
  and again only after `cross_term_dirty` is set (heat / potential `materials_changed` of the xy array): a script
  writing the xy array in place must call `materials_changed` too. Post-processing uses the same quads (`cross_quads`):
  cross flux averaged over the cell's quads, cross energy split by area share (held corners' shares to the free ones),
  so the dissipation integral is the operator energy (selftest heat_cross_flux_dissipation).
- Linear solve by `system.linear_solver` (a `solvers.pcg.PCGSolver`, factorized at every `assemble`; it holds the
  convergence record and work_counters, the system's tol / max_iter / compute_residual / last_* are properties on it;
  precision via `pcg.DTYPE_PRECISIONS`). The model path injects the build's 'linear' solver (`linear_solver=`: its
  precision wins, its counters are the summary 'solvers' role record; neither the system nor DiffusionPhysicsBase has
  work_counters). Stop reference: ||b|| in a steady solve, the step's initial residual when `capacity_over_dt` is set
  (PHY-1, floored at ~16 eps ||b||); float32 stalls near 1e-6 (default float64); unconverged solves set `last_warning`
  (never cleared) and `n_unconverged`. CG solves for u - u_fix (0 on fixed cells, `_solution`), so b holds only [c u]
  drives: never put held values [u] into b (against eps0-scaled rows the tolerance then stops CG after a few
  iterations). An injected sparse_direct runtime (a node, or the cpu Automatic) is wrapped in
  `solvers.sparse_direct.DiffusionDirectSolver` (PCGSolver protocol over one real float64 host SuperLU per assembly;
  tol / max_iter unused, compute_residual honoured). PCG details: solvers sub file. On the gpu a PCGSolver whose
  preconditioner is in `CAPTURED_PRECONDITIONERS` (Jacobi and multigrid: the V-cycle kernels launch into the block's
  capture, solvers sub file "pcg") is factorized with capture_blocks (`_factorize_solver`): every array a CG block reads
  (A_solve data, inverse_diagonal, the solution vector, the multigrid hierarchy) may change only together with a new
  factorize. A 2D tensor time_dependent study runs sparse_direct by Automatic (solvers sub file "Model path").
- `BCSide(conductance_factor=f)` (dirichlet / robin, default 1) scales that face's half-cell conductance 2 c / w in
  `fv_diffusion_3d.boundary_face_terms`, so assembly, anchors, flux, dissipation and `face_inward_flux` stay consistent
  (magnetic axis: 2, the exact half cell of psi with c = nu / r).
- Anchors: `require_anchors` (default True) gates `_check_anchors` in `assemble`; only heat in a thermal_noise study
  clears it. `assemble(check_anchors=False)` skips it for one assembly: only for a caller that changed coefficient
  values but not which c are > 0, nor held cells or faces (the magnetic B-H iterations after a loop's first).
  `mark_coefficients_changed()` (heat / potential `materials_changed` of the coefficient arrays: k(T), sigma(T)) dirties
  the operator like `operator_dirty = True` but marks a coefficient-only change (cleared by every other dirtying, as
  the diagonal-only mark): the next assembly then runs `positive_pattern_moved` (c > 0 per component and C/dt > 0
  against the last assembly, one reduction) and the full check only when a cell moved (gpu + PCG: read at the next
  solve, as `_pending_anchor_check`; not while an older check is still pending; `n_anchor_shortcuts`). The check
  computes a per-cell uint8 anchor code (`_anchor_code` twin: c_xx / c_yy / c_zz > 0, held, fixed, loss, C/dt > 0,
  outer face conductance > 0) and labels the link graph (`_label_anchors`, 2x resolution) only when it differs from
  `anchor_signature` (the first assembly always labels). On the gpu with a PCGSolver the comparison is
  `_pending_anchor_check`, read at the start of the next `solve()` (no sync of its own), so a floating region raises
  from that solve; a direct solver or the cpu checks in `assemble`. A new input of the labelling needs its bit in the
  code.
- Diagonal losses live in `extra_diag` [c] / `extra_rhs` [c u] (x cell_area; a cell with extra_diag > 0 anchors;
  `fixed_region_outflow` counts them): `add_diagonal_loss` accumulates (heat convection), `set_diagonal_loss` replaces
  on its mask (the replaceable diagonal of an iterated term: potential writes its whole ionic term with it at every
  assembly, then clears held cells). Both are wiped by `prepare`.
- `refresh_diagonal()` (Newton tangent update of an iterated loss): re-adds extra_diag (+ C/dt) to the cached
  conduction diagonal and redoes the diagonal-dependent parts and `linear_solver.factorize`, bit-identical to a full
  `assemble`. It returns False (the caller assembles) unless only the loss setters changed the system since the last
  assembly (any `operator_dirty = True`, `set_fixed` or `set_face_bc` forces a full assembly) and the pinned and
  loss-anchored cells are unchanged (one host sync). Writing `extra_diag` directly bypasses the mark: such callers must
  always assemble. `refresh_capacity(capacity_over_dt)` (heat rho / cp or step length changes) does the same with a new
  C/dt when the operator is otherwise current and the last assembly had a C/dt; it refuses when the C/dt > 0 cells
  change (anchors).
- `update_face_value(face, old_bc, new_bc)` (time-dependent feature values, `DiffusionPhysicsBase.set_face_value`)
  replaces one outer face's rhs term of the last 2D assembly in place (neumann: `FaceGeometry.outer_face_rhs`, the
  assembly's own formula) when kind, h and conductance_factor are unchanged
  (a dirichlet value, neumann flux, robin external); else, stale or 3D, False and the caller dirties the operator. It
  relies on outer faces carrying no cross (c_xy) flux.
- `refresh_held_values()` (potential driven-terminal bases): after `set_fixed` alone on cells already held (the held
  mask enters the face conductances, so a newly held cell needs a full assembly; detected by the held count, one sync)
  redoes fixed_values / coupling / rhs_offset on the last assembly, no factorisation; else False and the caller
  assembles. Combine with `update_face_value` before the set_fixed calls (it refuses once the operator is dirty).
- Held cells (`fixed_user`: terminals, voltage / temperature perimeters, fixed_temperature_domain; not the pinned
  empty-row cells) are ideal (c -> inf): a held-free face gets the free half cell 2 c_f L / w_f (`held=` of
  `fv_diffusion_3d.face_conductances` / `face_lower_shares` (`_series_link` / `_lower_share`: one formula per face,
  interior and phi wrap, 2D and 3D), passed by assembly, flux and dissipation alike), so the value sits on the held region's faces like an outer Dirichlet and a k = 0 / sigma_dc = 0
  held cell still anchors its conducting neighbours. `flux_at_centres` is 0 in held cells (their flux is
  `fixed_region_outflow`); a held-free face dissipates wholly in the free cell.
- Cut-cell held edges (2D, solver_defaults.conformal_boundaries; S7.6): `set_held_wall(t_x, t_y, located_x,
  located_y)` stores per interior face the drawn held edge at t of the centre spacing d from the free centre
  (`held_wall`, NaN = none, reset by `prepare`; later calls win per face); `face_conductances(held_wall=)` then gives
  c_f A / (max(t, HELD_WALL_FLOOR) d) (Shortley-Weller; shares unchanged, so flux / dissipation / outflow follow).
  Callers: `core.cut_cells.locate_held_edges(system, region, inside)` right after `set_fixed` of a region with a drawn
  outline (heat add_fixed_temperature, potential add_fixed_potential `inside=`). Only links leaving the region are
  located.
- Cut-cell insulated walls (2D: Cartesian, (r, z), polar, spherical; same option):
  `DiffusionPhysicsBase._set_insulated_wall_locator` hands `locator.material_index` to `set_insulated_wall_test`; the
  NEXT assembly (after features and dynamic updates) calls `core.cut_cells.locate_insulated_walls` with its conducting
  cells: closed = neither conducting nor held; a material is insulating when most cells of its centres are closed.
  `insulated_wall_geometry`: faces between open cells near a cut keep their free fraction (>= APERTURE_FLOOR, so the
  link graph / anchors never change), free cut cells their free area (>= VOLUME_FLOOR) plus the free slivers of adjacent
  closed cells, merged across the most open face (no new unknown). `set_insulated_walls(geometry or None)` stores
  `apertures` (`build_anisotropic_laplacian(apertures=)`) and `volume_fraction` (host): `cell_area` =
  `geometric_cell_area` x it; volumetric losses added so far (and that assembly's capacity) are rescaled by new / old
  fractions; `dissipation_density` divides by `geometric_cell_area`, `flux_at_centres` by the full face areas,
  `cell_volume` includes the fraction once located (point sources added before the first assembly keep the geometric
  area). k(T) / sigma(T): every assembly with check_anchors compares its conducting cells with the located ones (one
  sync) and re-locates on a change (`n_insulated_wall_locations`; walls follow drawn material outlines only, a material
  switching as a whole). Selftests conformal_cut_geometry, conformal_heat_k_switch.
- Cut-cell perimeter terms (same option; definition_helpers `outline_pieces_of`): flux features call
  `add_boundary_flux(weights, value, outline=True)` with weights = owner sums of piece length (x 2 pi r_piece on (r, z);
  ring_angle r sin theta on spherical) / geometric volume; the physics keeps that part (`_outline_source_window`) and
  rescales it in its `volume_fraction_listener` (`_volume_fractions_changed`), so injected totals stay q L.
  Convection: `add_surface_loss(mask, conductance, external, reference)` adds totals [c] to extra_diag / extra_rhs and to
  `surface_diag` / `surface_rhs`, which `set_insulated_walls` keeps out of the rescale (`set_diagonal_loss` on the same
  cells would drop them: no caller combines the two). Polar / spherical: on the logical cells. Selftests
  conformal_heat_outline, conformal_polar_outline_geometry, conformal_heat_polar, conformal_currents_spherical. Accuracy:
  perimeter terms and apertures are first order in the cell rings next to a curved insulated wall (errors ~ wall
  gradient x cell, dipole tails a few % of the rise two cells out), exact totals beyond ~5 cells; held cut-cell edges
  are second order. A second-order Neumann wall needs unknowns in the slivers (a true cut-cell FV); Johansen-Colella
  face-centroid interpolation would make the operator nonsymmetric (PCG).
- Jacobi-PCG needs ~10 iterations per cell along the longest window axis (kappa ~ n^2). pcg `max_iterations` 0
  (default, Automatic too) = auto, resolved by `solvers.pcg.resolve_max_iterations` (max(5000, 20 x longest window
  axis)); `cg_iteration_issues` warns for an explicit value below 10 x. Script constructors keep max_iter = 5000.
  Preconditioner 'multigrid' (Automatic on the gpu; solvers sub file) removes that growth; DiffusionSystem needs nothing
  for it (`linear_solver.factorize(A_solve, inverse_diagonal)` builds the hierarchy, held / pinned identity rows stay
  out).
- Geometry = `self.metric` (2D, `core.metric`), built in `prepare` from `domain.coordinates` and `axisymmetric`:
  `build_anisotropic_laplacian(metric=)` takes face areas and physical widths from it (conductance A / (w_l / 2 c_l +
  w_r / 2 c_r), Dirichlet / Robin half cells 2 c A / w), `cell_area` = its volumes, so an axis face (r = 0, theta = 0 /
  pi) has zero conductance and rhs (any BC there is void; anchors come from face conductance > 0, never from the BC
  kind). (r, z): totals (`fixed_region_outflow`, `face_inward_flux`) are whole-ring [c u m] (W for heat); densities keep
  their units. c_xy raises NotImplementedError on any metric but (x, y). `cell_volume(ix, iy)` = host value of one cell
  (point sources). `DiffusionPhysicsBase.axisymmetric` makes Session field derived values ring-weighted on a Cartesian
  model domain (polar / spherical weight by the metric volumes; the angle is the solvers' `ring_angle`). Pinned by
  examples mechanics/axisymmetric_heated_cylinder_thermal_stress, heat/cylinder_cooling_bessel and selftests
  heat_polar_metric / heat_spherical_metric.
- Azimuthal order (symmetry 2.14; `DiffusionSystem(azimuthal_order=m)`, axisymmetric (r, z) on a Cartesian domain
  only): u is the amplitude of u_m(r, z) cos(m phi). Rings carry pi (`ring_angle`), and assemble adds
  `azimuthal_diagonal` = c_xx m^2 / r_c^2 x cell_area (0 on held cells) to the operator diagonal BEFORE the
  conduction-diagonal copy, so refresh_diagonal keeps it; the radial c_xx is taken along phi. It is the axis rule (u_m ->
  0 at r = 0, no BC) and anchors every c_xx > 0 cell; `fixed_region_outflow` adds it to the row sums,
  `dissipation_density` adds azimuthal_diagonal u^2 (so the pi-ring integral stays the total and P = sum V I holds). u =
  r_c (m = 1) is exact on any r widths, r_c^2 (m = 2) on uniform dr; the axis cell's flux_at_centres averages in the
  zero axis face (half the value). Selftests potential_azimuthal_transverse_field, heat_azimuthal_order_2.
- Phi wrap (`metric.wrap_sign` s, `system.wrap_sign`): one extra face per ring between the last (j = ny-1) and first
  angle cells, `wrap_face_conductance` (the y_hi face areas; held as interior faces), energy g (u_last - s u_first)^2:
  +g on both diagonals and -s g in the two wrap slots of the CSR pattern (`LaplacianPattern.wrap`; ny >= 3). `build_anisotropic_laplacian` returns it as a 6th value (`system.wrap_conductance`;
  `face_conductances` stays the in-window 2-tuple) and raises for any y_lo / y_hi condition but the insulated default.
  Consumers: `flux_at_centres`, `dissipation_density` (the wrap share of `face_lower_shares`), `fixed_region_outflow` (an
  antiperiodic pair's rows sum to 2 g), the anchor check (an extra link-image column, then `_join_wrapped_regions`:
  host union-find with sign parity; an antiperiodic loop anchors its region, periodic keeps the constant null mode).
  Multigrid needs nothing (antiperiodic +g entries are never strong links). Selftests heat_polar_wrap,
  currents_polar_wrap.
- One operator for 2D and 3D (`fv_diffusion_3d`: `FaceGeometry` of `metric` / `metric_3d` built once in `prepare`,
  separable 1D face-area factors and physical lengths, never full-window area arrays; `LaplacianPattern` slot layout
  (`_SlotLayout`: sorted neighbour displacements, + 4 c_xy corners in 2D, + 2 wrap slots), `build_laplacian`,
  `anchor_link_graph`, `flux_at_centres`); `build_anisotropic_laplacian` is its 2D wrapper (checks, cross quads).
  Cut-cell apertures and held-wall fractions enter `face_conductances` as one more area factor / free-half rule.
- 3D (B3, chosen in `prepare` by `domain.dimensions`): 7-point operator,
  window (wx, wy, wz) flat C-order, `cell_area` = cell volume, `system.faces` = FACES_3D (ScalarBoundaryConfig carries
  z_lo / z_hi, insulated by default, ignored in 2D). `assemble(..., c_zz=)` is required; c_xy must be 0
  (NotImplementedError), axisymmetric raises. The pattern's column order is -x, -y, -z, diag, +z, +y, +x (int32 limit
  7 n < 2^31). Anchor check: the same
  2x-resolution link graph (8 voxels per cell). `flux_at_centres` returns (flux_x, flux_y, flux_z);
  `dissipation_density` raises (2D only: potential runs 3D only on tetrahedra). Probes / point cells take iz
  (`FieldProbe(iz=)`, `require_window_cell(..., iz)`), PML edge extension via `edge_extension_views_3d`. Geometry =
  `system.metric_3d` (`system.metric` stays None in 3D), on the device a `FaceGeometry` built once in prepare
  (separable factors, physical lengths at most (wx, wy, 1): no full-size metric arrays), so a cylindrical sector needs
  no other branch. Its phi wrap is the 2D one: two extra slots (`_SlotLayout`: 9, wrap columns between -x / -y and +y /
  +x; wy >= 3), `wrap_conductance` (wx, wz), a last y voxel layer joined by `_join_wrapped_regions`. Face drives (a cell
  vector w whose physical flux is -c (grad u - w): magnetostatics_3d remanence and coils): `face_drive_integrals_3d`
  (held halves 0, an outer face its boundary half), `face_drive_terms_3d` (x the last assembly's conductance; Neumann
  none), `face_drive_divergence_3d` (rhs), `face_terms_at_centres_3d`; geometry `system.geometry_3d`. `face_flows_3d`
  (2D windows too) = operator face flows (n + 1 along each axis), `face_flow_energy_3d` = sum of flow^2 / conductance over faces touching
  a free cell (twice the discrete energy), `centre_gradient_3d` (Cartesian; one-sided in the outer layer). Memory:
  `diffusion_system_estimate(wrap=True)` counts the dense 9-slot build array; the CSR stays within 7 per row. Selftests
  heat_3d_field_access, heat_cylindrical_sector_3d.
