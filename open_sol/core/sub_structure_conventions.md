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
- `MaterialGrid.park(names)` (em3d / acoustics sub files "Parked material arrays"): parked arrays are host numpy;
  `assign` / `assign_window` write them there and `device_array(name, index)` uploads (a view when not parked). Code
  reading a MaterialGrid array as a device array (`getattr(materials, name)` into a kernel or xp expression) breaks on
  a parked one: a parking solver's rule excludes the meshes where such readers run. The rule is shared:
  `model.definition_helpers.parkable_material_arrays` (a new parking type calls it, never a copy).
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
  `materials_changed` of sigma_e; the 2D FDTD coefficient layout checks (sigma_e == 0 at the first E rebuild after a
  sigma_e change, mu_r / sigma_m uniform per H rebuild, none once per cell); em_fdtd_3d: the chunk uploads of parked
  material arrays per index rebuild, one face kind's geometry upload per subpixel refresh; elastic_waves with parked
  materials: the host coefficient rebuild (downloads, uploads) per rebuild (build, a new runner dt).
- laminar_flow: the velocity check once per lattice check interval (>= 100 steps).
- Probe tables: `monitors.park_probe_rows` right after the time loops' per-chunk sync (study `_run_time_dependent`,
  `_run_harmonic_pulse`; `Simulation.run` per chunk), one transfer per table holding >= PROBE_PARK_ROWS rows; a HarmonicProbeRecord's stacked
  transfer every HARMONIC_PARK_VALUES device values (between frequency solves).
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
  same-type dummies). `cache=True` keys a closure by its pickled cell values: never another numba dispatcher in a cell
  (it pickles with a per-process uuid, so every process recompiles; symptom: `<name>-<line>.py311.N.nbc` in
  `__pycache__` with N rising run after run): publish the helper as a module global
  (`fluid.lattice._numba_step_helpers`). Each numba twin gets a CPU case `<...>_numba`
  (`selftest_helpers.numba_twin_result`).
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
- Model mirror planes = the EM bc_<face> 'symmetric' / 'antisymmetric' faces, elastic_waves bc 'symmetric'
  (`EVEN_ONLY_BC_TYPES`: no odd face, so sector_specs refuses it) plus the faces of the enabled `MIRROR_FEATURES`: the
  `symmetry` feature of every non-EM type (its natural even face; magnetostatics_3d -1), membrane / pressure_acoustics
  also `antisymmetry` (-1). magnetic_insulation never is (it is also the far boundary). A new mirror feature validates
  through `definition_helpers.mirror_feature_issues` (Cartesian domain, not the axisymmetric axis, EM partners mirror
  the face) and `own_mirror_face_conflicts` (no other face feature of the type there). Per physics
  (`mirror_faces(spec, physics)`) a face the physics does not set itself unfolds even, so its own face there must be
  insulated (`definition_helpers.mirror_face_issues` warns), except the `FACE_FEATURE_SIGNS` types, signed by their own
  face feature on the plane.
- Totals take `mirror_faces`, displays (`unfold_field` callers) `unfold_faces`: it adds the x_lo rotation axis of an
  axisymmetric `AXIS_UNFOLD_TYPES` physics with that type's sign, which ring weights already count, so it never
  doubles a total. The true image at phi = pi is a rotation by pi about the axis, which one mirror sign reproduces only
  when no field mixes in-plane and azimuthal vector parts: +1 for scalar / (r, z) vector types, -1 for the A_phi
  magnetic types; never the BOR EM types (E_phi beside E_r / E_z). A new axis type needs every renderer field's
  parity right under its sign.
- `FieldInfo.parity` names a renderer field's image rule (`image_sign`; '' = even when unsigned, else unknown:
  `unfold_field` raises); `staggered` = axes of hi-face samples (plane value 0 when odd, else the nearest sample: exact
  only for odd staggered components); `plane_signs` (`plane_signs_text`) replaces the face signs for one field (the
  nlfd channels, nlfd sub file). Fields are never mirrored twice: `FieldData.unfolded` (parity-sector snapshots are
  stored unfolded; results_io saves it).
- Every total over a selection, line, plane or box touching a plane goes through `touch_factor` (2 per touched face;
  mirror-even integrands; callers: grep `touch_factor`). heat reports no feature totals.
- Parity sectors: `sector_specs` = 2^k spec copies (MIRROR_FEATURES swapped and split per face by
  `_swap_mirror_features`; a type without features of both signs (heat and the other symmetry-only types) raises,
  reported by study validation
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
- Device 0-d values go to the host (complex128 numbers) every HARMONIC_PARK_VALUES: a recorder may pass cupy 0-d
  values without a sync, but never relies on them staying device arrays in `_rows`.

## Probe tables (`monitors`)
- gpu `ProbeRows` (FieldProbeRows, acoustics ProbeRecorder) park their device rows on a host table at study chunk
  boundaries (`park_probe_rows`, a weak registry of live tables): `n_rows` counts every row, the device counter and
  `device_rows` only those since the last park, the same device table and launch stay (step graphs keep their
  pointers). Code reading samples goes through `column_values`, never `_samples` rows by absolute index; a park
  between a record and its `stamp` / `rewrite_last_row` would lose that row (never inside a step). Scripts park in
  `Simulation.run` (chunks of at most PROBE_PARK_ROWS ticks; a script stepping solvers itself does not). Per-probe FieldProbe buffers (cpu
  engines, probes no table took over) never park. Estimate: `definition_resonance.point_trace_items`.

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

## Run-time estimate (`time_estimate` pure Python, `time_profile`)
- Ops carry sizes only; `price` maps each to its profile curve (the op docstrings name them). An Elementwise op's bytes
  are the distinct bytes all its launches move (launches stream distinct arrays, priced at the rate of their total:
  the curve repeats one cache-hot call); a loop re-reading the same small arrays is a multiplicity of single-launch
  ops, not launches. A kernel bound by arithmetic also counts `flops` (gpu 'arith_float<bits>'); complex64
  DenseFlops count 8 n^3 real flops. A new op needs its curve in run_profile (or prices from an existing one); a
  profile lacking a curve moves the item to not_estimated.
- Each benchmark times the kind of call the engines make (gpu 'dot' = pcg's multi-block `device_vector_dot`; the study
  max-abs checks and harmonic FDTD energies reduce axis-first, ~1.5x that curve at 1e6 values; 'fused' = an
  ElementwiseKernel / numpy expression twins; cuDSS = `sparse_direct.cudss_factorization` as the engines call it): an
  engine switching to another primitive needs its own curve. Real operators and other fills are the caller's `scale`
  (`sparse_direct.cudss_2d_ops`), never extra curves.
- `code_hash` covers time_profile.py and `_TIMED_SOURCES` (solvers/sparse_direct.py, cudss_factor.py): any edit there
  forces every user to re-profile, so code that is not a benchmark lives elsewhere (the outdated-profile fallback is
  `model.time_estimate.stored_profile`). Progress texts key `task_seconds`: renaming one drops its previous time.
- Shared recipes: `explicit_step_estimate` (an engine giving another kernel a numba twin widens its `numba_items`
  predicate) and `diffusion_time_estimate` (a new DiffusionSystem type passes its `held` HeldValues; a
  refresh_diagonal / refresh_capacity factorises, so the role counter counts it as an assembly).
- `time_check` groups items by `seconds_counter`; an item without one joins 'study' = study_seconds minus every counted
  group, so a role timer must not also sit inside another group's timer (mechanics 3D eigenfrequency clears its eigen
  items' seconds counter: the PCG timer already holds them).

## Damped iteration (`damped_iteration`)
- The one loop driver of nonlinear DiffusionSystem physics (potential electrolyte Newton, magnetic B-H Picard /
  Anderson); its knobs are physics settings, never a solver role. A caller whose run returns False (unconverged, the
  damped state kept) appends `last_warning` to its `runtime_warnings`; its `work_counters` merge the loop's only once
  the loop ran.
- Anderson mixes states linearly: the state must stay meaningful under linear combinations (magnetics mixes ln nu, not
  nu), and a `scale` array must not change in place during one run (the weighted history rows are kept by identity).
- A Picard on a coefficient (magnetostatics B-H nu) keeps the state in the physics, not in a material array (that holds
  the curve's parameters): the assembly reads the state array, `propose` assembles, solves and returns nu(|B|).

## DiffusionSystem (`fv_diffusion`, used by heat, potential and magnetic)
- `DiffusionPhysicsBase` forwards settings and the convergence record to the system (`_forwarded`): plain attributes on
  a physics never reach it, so a new setting is a forwarded property there. The model path injects the build's
  'linear' solver (`linear_solver=`: its precision wins, its counters are the summary 'solvers' role record; neither
  the system nor DiffusionPhysicsBase has work_counters).
- Post-processing (`flux_at_centres`, `dissipation_density`, `flux_jump_indicator`) reads the last assembly's cache
  and raises while `operator_dirty` (`heat_flux_magnitude` re-assembles first; potential reads right after a solve).
  A physics whose shown flux is not the system's (potential driven terminals, magnetic H with remanence, the AC
  phasor) overrides `_indicator_flux`.
- Two assembly paths: the first assembly and every wrap / c_xy / cut-cell case through `build_laplacian`, re-assemblies
  on the kept plain pattern through `fv_diffusion_fused` (selftest diffusion_fused_reassembly). A new operator term
  goes into both paths or makes `_fused_pattern` refuse it; `_keep_assembly` stores what both keep.
- One assembly at a time: every full assembly starts with `_release_assembly` (operators, assembly vectors, the
  PCGSolver binding), keeping only what the anchor shortcut compares (assembled c, C/dt, anchor code). A new kept
  assembly array joins both `_release_assembly` and `_keep_assembly`; nothing may read the previous assembly inside
  assemble except those. The `LaplacianPattern` assembly maps (slot_positions, entry_rows) are released after the
  first elimination where `_fused_pattern` holds and rebuilt transiently on access (full_operator); systems
  re-assembling through build_laplacian keep them (no rebuild sync per step; estimate: `conformal=`, cross_term, wrap).
- One device CSR per assembly (A_solve): A_eff is never kept (`assembled` marks a kept assembly). Its readers take the
  kept face terms: `_off_diagonal_product` (A_eff u_fix on free rows: refresh_diagonal / refresh_held_values),
  `fv_diffusion_3d.crossing_outflow` (fixed_region_outflow) and the transient `full_operator()` (thermal noise,
  parity checks; a dense slot array + slot_positions pass, so never per step). A new off-diagonal operator term goes
  into `fv_diffusion_3d._off_diagonal_slots`, `off_diagonal_product` and `crossing_outflow` too, else those three
  silently drop it (selftest diffusion_fused_reassembly checks them against A_solve / rhs_offset). Assemble deletes
  A_eff and its window temporaries before `_factorize_solver` (the multigrid setup / factor plan is the build peak).
  A direct or banded linear solver (`ignores_preconditioner`) gets no Jacobi diagonal (`_keeps_inverse_diagonal`).
- Precision (why the system dtype, not float32 storage, follows the linear solver): A_solve, the CG vectors and the
  conduction diagonal must share the solve precision, since rounding a Laplacian's diagonal and off-diagonals apart
  breaks its exact zero row sums: a spurious sink of ~eps a_ii per row, amplified by kappa in the low modes (the
  'single' failures: float32 CG floor ~1e-6, iron |B| errors of percents in magnetostatics). Rounding coefficients
  consistently (g to float32, then the diagonal as their float64 sum) perturbs only the physics by ~eps32 (the
  material arrays are float32 already), with no kappa factor: the route to float32 operator storage beside a float64
  CG, not yet built. pcg 'mixed' is no memory saving here (the float64 A_solve stays for the residual, plus its
  float32 copy) and its CSR path is Jacobi only, so the gpu Automatic stays double multigrid (its hierarchy is float32).
- c_xy cross term (2D Cartesian only): never drop its diagonal part (row sums would act as a spurious exchange with
  u = 0 at window / region corners). Definite on graded cells only for c_xy^2 < c_xx c_yy x 4 r / (1 + r)^2 per axis
  (r = neighbour width ratio), stricter than the continuum bound. Whether c_xy is nonzero is decided (one sync) only
  when `cross_term_dirty` is set (heat / potential `materials_changed` of the xy array): a script writing the xy array
  in place must call `materials_changed` too.
- CG solves for u - u_fix, so b holds only [c u] drives: never put held values [u] into b (against eps0-scaled rows the
  tolerance then stops CG after a few iterations). Script constructors keep max_iter = 5000; the model's 0 = auto
  (`pcg.resolve_max_iterations`, Jacobi needs ~10 iterations per cell of the longest window axis; 'multigrid' needs
  nothing from the system).
- Anchors: `assemble(check_anchors=False)` only for a caller that changed coefficient values but not which c are > 0,
  nor held cells or faces (the magnetic B-H iterations after a loop's first); k(T) / sigma(T) call
  `mark_coefficients_changed` instead (one pattern test, the full check only when a cell moved). On the gpu with a
  PCGSolver the check is read at the next `solve()`, so a floating region raises from that solve, not from `assemble`.
  A new input of the labelling needs its bit in the `_anchor_code` (else the cached signature skips the relabel).
- In-place refreshes instead of a full assembly (each returns False when it cannot, and the caller assembles):
  `refresh_diagonal` / `refresh_capacity` (losses or C/dt), `update_face_value` (a face value of t; relies on outer
  faces carrying no c_xy flux), `refresh_held_values` (set_fixed alone on cells already held). Combine
  update_face_value before the set_fixed calls (it refuses once the operator is dirty).
- Held cells are ideal (c -> inf): a held-free face takes the free half cell through `held=` of
  `fv_diffusion_3d.face_conductances` / `face_lower_shares`, which assembly, flux and dissipation all pass: a new
  consumer of face conductances passes it too. A k = 0 / sigma_dc = 0 held cell still anchors its neighbours.
- Cut cells (2D, solver_defaults.conformal_boundaries): held edges by `core.cut_cells.locate_held_edges` right after
  `set_fixed` of a region with a drawn outline (heat add_fixed_temperature, potential add_fixed_potential `inside=`);
  insulated walls located by the next assembly (`_set_insulated_wall_locator`) and again whenever its conducting cells
  change (k(T) / sigma(T): walls follow drawn material outlines only, a material switching as a whole). Their volume
  fractions scale `cell_area`: point sources added before the first assembly keep the geometric area;
  `dissipation_density` divides by `geometric_cell_area`, `flux_at_centres` by the full face areas.
- Cut-cell perimeter terms (same option; definition_helpers `outline_pieces_of`): flux features call
  `add_boundary_flux(weights, value, outline=True)` with weights = owner sums of piece length (x the piece's ring
  factor) / geometric volume, kept as totals when volume fractions change. Convection: `add_surface_loss` (totals
  [c], kept out of the rescale; `set_diagonal_loss` on the same cells would drop them: no caller combines the two).
  Accuracy: perimeter terms and apertures are first order in the cell rings next to a curved insulated wall (dipole
  tails a few % of the rise two cells out), exact totals beyond ~5 cells; held cut-cell edges are second order. A
  second-order Neumann wall needs unknowns in the slivers (a true cut-cell FV); Johansen-Colella face-centroid
  interpolation would make the operator nonsymmetric (PCG).
- Geometry = the window metric (`core.metric`): an axis face (r = 0, theta = 0 / pi) has zero conductance and rhs, so
  any BC there is void and anchors come from face conductance > 0, never from the BC kind. (r, z) totals
  (`fixed_region_outflow`, `face_inward_flux`) are whole-ring [c u m]; densities keep their units. Area factors stay
  separable 1D (`FaceGeometry`), never full-window arrays.
- Azimuthal order (`azimuthal_order=m`, axisymmetric (r, z) only): `azimuthal_diagonal` enters the operator diagonal
  BEFORE the conduction-diagonal copy, so the refreshes keep it; it is the axis rule (u_m -> 0 at r = 0, no BC) and
  anchors every c_xx > 0 cell. u = r_c (m = 1) is exact on any r widths, r_c^2 (m = 2) on uniform dr; the axis cell's
  flux_at_centres averages in the zero axis face (half the value).
- Phi wrap (`wrap_sign`): `wrap_conductance` is a separate value beside the in-window `face_conductances`, so every
  consumer handles it (flux, dissipation, fixed_region_outflow: an antiperiodic pair's rows sum to 2 g; the anchor
  check's `_join_wrapped_regions`: an antiperiodic loop anchors its region, a periodic one keeps the constant mode).
- 3D (`domain.dimensions`): `system.metric` stays None (`metric_3d`); `dissipation_density` / `flux_jump_indicator`
  are 2D only (potential runs 3D only on tetrahedra). Face drives (a cell vector w whose physical flux is -c (grad u -
  w): magnetostatics_3d remanence and coils) are the `face_drive_*_3d` helpers of `fv_diffusion_3d`.
