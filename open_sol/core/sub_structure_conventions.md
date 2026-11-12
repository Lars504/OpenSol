# open_sol/core — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Gotchas of core modules only; general array /
coordinate rules stay in the root file.

## Backends (`backend`; solver_plan S6)
- `core.backend` alone imports cupy / cupyx at module level (try / except, no driver call): `cp`, `cupyx_sparse`,
  `cupyx_ndimage` are the real modules or stand-ins. Stand-ins pass numpy's scalar types through (`cp.float32` is
  `np.float32`), return a stand-in kernel from RawKernel / RawModule / ElementwiseKernel / ReductionKernel (module-level
  kernel definitions import), never match isinstance and raise `BackendUnavailable` on any other call.
  `cupy_installed()` = cupy imported; `gpu_available()` adds one cached CUDA device query (worker thread only);
  `default_backend_name()` = 'gpu' with cupy, else 'cpu' (script objects built without a backend).
- The run backend lives on the Domain (`domain.backend` = `Backend(name, xp)`, `domain.xp`; Domain / Domain3D take it
  from their MaterialGrid); `use_backend(name)` switches an array-free domain (the builder calls it right after
  `make_domain` with `schema.solver_backend(spec)`). Build arrays come from `domain.xp` / `ctx.xp` /
  `array_module_of(input)`, host copies from `to_host`, device syncs from `domain.backend.synchronize()`. A cupy kernel
  used on a cpu path needs a numpy twin with the same arguments next to it (fv_diffusion `_source_rhs_host`, heat
  `_rise_to_temperature_host`, potential `_add_scaled_host`; explicit engines: "Kernel twins"), changed in the same
  edit. Never pass `where=None` to `xp.copyto`: cupy reads it as no mask, numpy as False (nothing written); branch on
  a None mask instead (MaterialGrid.assign / assign_window).
- `gpu_memory` without cupy: pool counters 0 (`_NoPool`; the ledger records 0 B per stage), `release_all` = gc;
  `cpu_device_info()` / `cpu_memory_status()` (host RAM as device memory, no budget) are what the worker emits without
  a usable GPU. `MemoryLedger(backend='cpu')` counts tracemalloc current / peak bytes (`_TracedHost`: pre-imports
  `_CPU_BUILD_MODULES` so their module objects are not booked, starts tracing unless already on, resets the peak; the
  builder calls `stop()` in a finally), compared with gpu_steady / gpu_build_peak like the pool; no run-memory record
  on the cpu (tracing would slow the run). `MemoryLedger.track` also times each stage (`stage_seconds`; `add_time`
  books the untracked validation / geometry phases; `ledger_time_text` = the "Build time by stage" line, logged also
  after a failed build; summary.json build.stage_seconds). `owned_bytes(..., backend='cpu')` counts numpy buffers (a
  view counts its base once; every reachable array incl. small host tables); the default 'gpu' ignores numpy arrays.
  `Simulation.memory_report` passes the Domain's backend and prints `cpu_memory_status()` on the cpu.
- Every physics type declares 'cpu' (`PhysicsType.backends`): host solves through scipy, fields on `domain.xp`,
  explicit-engine kernels and coupling-transform kernels are `core.kernels` twins (other transforms allocate on their
  input's / the Domain's xp); `FieldProbe` keeps its buffer on the first record's module. `FieldProbe.record` of a
  float32 / float64 cupy field is one `core.probe_store` launch (`trace[slot] = field[element offset]`, any strides);
  numpy, float16 (em3d half_storage), complex or out-of-shape cells take the indexed `record_value`. A coupling runs on the cpu
  once both partner types declare it. gpu only: the half_storage steppers and the cusolver_qr back end (solver
  backend_misfit).
- Memory estimates: `diffusion_system_estimate(direct=True)` (a host SuperLU sparse_direct linear solver, the cpu
  Automatic) adds the host factor (`host_real_lu_bytes`: 2D n log2 n, 3D n^(4/3) fill, LU_3D_ENTRIES_PER_UNKNOWN_4_3
  uncalibrated) and its CSC input copy as host items; on the cpu `check_memory` counts them with the device items.
  `direct=<host direct name>` (MUMPS / PARDISO; `definition_helpers.linear_solver_direct` returns the back end name)
  books instead `host_direct.factor_host_bytes` (the SuperLU heuristic x `superlu_bytes_scale`: LDL^T / LU fill,
  workspace and mixed ratios, all uncalibrated: refit from summary.json host_direct_lu_entries / host_direct_peak_bytes)
  + `solver_host_bytes` (function-level import of solvers.host_direct, a listed exception of the core rule). A
  cuDSS 'linear' back end is not in it (core may not import solvers.cudss_factor): the caller combines
  `fem.diffusion.cudss_diffusion_estimate(input_copy=False)` (fem sub file). Its build extra is the larger of the
  pattern build + anchor check and the multigrid setup with assemble's locals alive at `factorize`
  (_FACTORIZE_ALIVE_*): the pattern-build temporaries are freed before it, so the two never add. flux temporaries
  are the transient phase 'flux post-processing'.

## GPU memory (`gpu_memory`, `host_memory`)
- `gpu_memory.set_budget()` limits cupy's pool to min(90 % of the device, free + pool reserved - 512 MiB): overruns
  raise OutOfMemoryError instead of spilling. Recomputed before every job (worker: after `release_session`; `api.run`:
  at start); when it raises the job fails (Issue at ('model',)), never runs without a pool limit (Windows sysmem
  fallback is 10-100x slower). Pool counters are process-wide (`BuildSummary.session_gpu_bytes` is one build's share).
  `memory_status()` is a driver query: <= ~1 Hz.
- `Simulation.memory_report()` / `owned_bytes` count a shared array for its first owner (materials, physics,
  couplings); memory_report also lists host bytes of CPU factorisations (`host_owners`: FDFD `host_factor_bytes` = L +
  U of the last scipy_splu), which the model path reports only in the study log (`solve_summary`). Outside the cupy
  pool: the cusolver_qr workspace (guarded by `solvers.sparse_direct.check_solver_size`) and every cuDSS factor
  (guarded by cuDSS's own plan estimate in `solvers.cudss_factor`; estimated as MemoryEstimate gpu_external by
  `cudss_factor.factor_device_bytes`, measured by the process-wide live / high-water counters of cudss_factor). A
  MUMPS / PARDISO factor (`solvers.host_direct`) holds host memory outside numpy (its `host_bytes`): guarded by the
  library's own estimate after analysis against HOST_FRACTION of `host_memory.available_bytes()` (no guard when that is
  None), estimated as host items; a check measuring numpy host arrays does not see it.
- `release_all`'s full `gc.collect` walks every tracked object (modules, registry: ~25 ms). A sweep
  (`study._run_sweep`) calls `freeze_collect_baseline()` after its first value's collect and `thaw_collect_baseline()`
  at its end, so the per-value collects walk only sweep-made objects; skipped when the gc is frozen already
  (selftest_cache freezes for the whole selftest), never unfreezing a freeze of its own caller.

## Accepted host syncs (the root rule: none per step in hot loops)
- The build's Courant check; one `Device().synchronize()` per study chunk and one at the end of each study run
  (`study_seconds`); heat / potential CG, one per `pcg.check_interval(preconditioner, backend)` iterations (gpu 16,
  multigrid 2; the gpu anchor-code comparison of a re-assembly is read right before it); one per
  `DiffusionSystem.refresh_diagonal` / `refresh_capacity` (pinned / anchor cells unchanged) and `refresh_held_values`
  (held count); one per radiation term
  and steady heat radiation re-linearisation (its convergence test); the FDFD solve
  (scipy_splu: one rhs / solution transfer each); one HarmonicModeMonitor transfer per solve; one grid-row transfer per
  FDFD port mode re-solve (per reassembly) and per FDTD mode-port recalibration after the first step (material drift
  check, re-solve above 1e-3); the E_z ohmic-box refresh per `materials_changed` of sigma_e; the builder's material
  range checks (model sub file, Schema); the laminar-flow velocity check once per lattice check interval (>= 100
  steps); per Newton iteration of a device tangent (fem.elasticity3d.TangentAssembler3D(device=True)) the displacement
  up / force down and the operator norm of `CudssFactorization.refactorize_device_values` (its scatter sums are
  cupyx.scatter_add, never cupy.bincount, which syncs on max(x) per call); per assembly of a solid_mechanics_3d system
  with the multigrid preconditioner (`physics.mechanics.multigrid3d.BrickMultigrid.build`) the power-iteration dots of
  every level (2 x POWER_ITERATIONS + norms) and the 81 coarsest probe reads (never inside a V-cycle).

## Kernel twins (`kernels`; solver_plan S6, the explicit engines on the cpu backend)
- `twin(name, gpu_kernel, cpu_fn, numba=None)` (Elementwise / Reduction / any cupy callable) and `raw_twin(...)`
  (RawKernel, called `kernel(grid, block, args)`) return a registered `KernelTwin` (`REGISTRY`, unique names
  '<package>.<kernel>', a duplicate raises), kept under the old kernel name so call sites never change: the cupy kernel
  runs when the first array argument (raw: first array in args) is cupy, the host twin when it is numpy. One code path
  per engine: arrays from the Domain's `xp`, kernels are twins. `axis_first_reduce(kernel, *args, combine=None)`: a
  ReductionKernel (or twin; kwargs reach the cupy kernel only) to one value, on cupy arrays of >= AXIS_FIRST_MIN_SIZE
  values first over args[0]'s shortest axis, then combine (default sum; study's `_peak` recurses for 3D partials);
  numpy / small / 1D: the plain call.
- Host twin contract: the same positional arguments as the cupy call (Elementwise: inputs then outputs; Raw: the args
  tuple, grid / block ignored), outputs written IN PLACE (`out[...] =`, `+=`: outputs are often box / line views,
  never rebind them), the same dtypes (float64 where the cuda body uses double; numpy keeps float32 for float32 arrays
  with numpy / Python scalars), the same branch rules (periodic wraps, own faces) and broadcasting (1D spacings per row
  / column). Reductions return a 0-d numpy array (callers take float()). A wrong twin shows as an analytic-case or
  parity failure of the CPU-vs-GPU self-tests (physics/em selftests_backend). Twins may allocate per call (cpu
  reference path); numpy scalars of the GPU path work unchanged (`int(...)` them before slicing).
- numba (optional, never required): `numba=factory(numba_module) -> compiled callable` with the numpy twin's
  signature, built lazily at the first host call when `numba_available()` (importable, not switched off by
  `set_numba(False)` / OPEN_SOL_CPU_KERNELS=numpy). A factory failure or a numba error / TypeError at a call (typing,
  dispatch: before the twin writes) warns once and falls back to numpy for good; any other exception is re-raised
  (in-place outputs may be partly updated). numba twins only for per-step / per-iteration kernels (2D / 3D Yee updates,
  em3d CPML / ohmic loss, elastic_waves velocity / stress, laminar_flow lattice step, solid_mechanics_3d CG step:
  stiffness / mass products, block preconditioner, fused CG pair; fem.element_tangent_3d per Newton iteration).
  Host-only numba loops without a cupy kernel are `host_twin(name, numpy_form, numba_factory)` (pcg short-vector dot /
  axpy, fixed_pattern row sum / block expansion / restriction, host_direct scaled_values): never a hand-rolled
  try / except around njit. It registers at its first call and returns the same twin later, so modules that must not
  import core.kernels at module level (pcg, host_direct, fixed_pattern) call it inside the function. The numpy form
  takes the numba loop's arguments; a size threshold calls `twin.cpu` below it, `add(numba=True)`-style switches check
  `twin.host_kind == 'numba'`.
  `@njit(parallel=True)` loops over i with prange reading arrays by [i, j(, k)]: every array a numba twin
  indexes must have that dimensionality even when unread (placeholders (1, 1, 1)), numba types every branch. The prange
  index is uint64: a conditional mixing it with a signed int (`j if c else i`, `0 if c else i + 1`) unifies to float64
  and fails as an index (the twin then falls back to numpy), so every numba twin rebinds it at the loop top: `for
  prange_i in numba.prange(n): i = np.int64(prange_i)`. Variant flags are closure constants of the factory; a Python
  wrapper may unpack / reshape the args (absent arrays -> same-type dummies). Each numba twin gets a CPU case
  `<...>_numba` (numba vs numpy on random inputs, `selftest_helpers.numba_twin_result`). `host_kinds()` lists which
  host twin each name runs.
- Precision variants (the 2D FDTD 'double' stepper, mechanics kernels3d / multigrid3d): `real_raw_twin(name, source, entry, cpu, numba=None,
  dtype_argument=0)` wraps a `RealRawKernel` (CUDA source with `real_t` / `REAL_IS_DOUBLE`, compiled per dtype of
  args[dtype_argument]; intrinsics such as __fmul_rn need an `#if REAL_IS_DOUBLE` branch); ElementwiseKernels take the
  field dtype as a type placeholder (T / F) and cast float32 material inputs explicitly. Their scalars must be numpy
  scalars of the field dtype (cupy casts Elementwise scalars to T, so a float32 dt silently rounds the step; a RawKernel
  reads the wrong bytes). Numpy twins lift float32 material arrays to the output dtype first (`astype(out.dtype,
  copy=False)`: under numpy 1.x value-based casting float32 array x float64 scalar stays float32).
- A new explicit engine on the cpu: wrap every RawKernel / ElementwiseKernel / ReductionKernel with twin / raw_twin,
  take every array module from the Domain / an input (`array_module_of`), pass `xp=` into the objects that allocate
  (fields, CPML, sources, monitors), declare `PhysicsType.backends = ('gpu', 'cpu')`, refuse what stays gpu-only
  (half_storage: explicit_kernels backend_misfit), add a `selftests_backend.backend_parity_case` wrapping its analytic
  cases.

## Half storage (`half_storage`; solver_plan S5)
- float16 arrays (em_fdtd_3d / elastic_waves fields, laminar_flow populations) are touched only by own kernels:
  RawKernels through `storage_preamble` (FLOAD / FSTORE), ElementwiseKernels with a T placeholder through
  `STORED_ACCESS` (`stored_load` / `stored_set` overloaded on float and unsigned short) called with `half_bits(array)`
  (the uint16 view). cupy's own float16 kernels (astype / copyto / arithmetic / fills INTO float16) construct `__half`
  from float / int, which fails at ptxas ("Unresolved extern function '_ZN6__halfC1Ef'") on some CUDA / cupy pairs;
  bit-level ops on the uint16 view are fine (zero fill, `*= bool mask`, sign flip by xor 0x8000). Reading float16 into
  float32 by cupy (astype, `(float)` in a kernel) still works there (convert those readers too if one fails).

## Graded meshes (`mesh`)
- Graded grids are rectilinear: a refinement fills whole columns / rows (3D: also z layers, same march).
  `_build_1d_mesh` marches on the target profile graded to slope (r - 1) / r (`_graded_profile`), each cell at the
  smaller target of its two ends (`_march_segment`; else the ratio clamp lifts a refinement's first cell: selftest
  mesh_tensor_options anisotropic_y_fine), so refinements keep their target and the ramps (growth 2 - 1/r per cell,
  also in `_graded_axis_cell_bound`, which bounds the same profile piecewise so overlapping refinements count once) lie
  outside them. Rectangle edges are snap nodes (`snap_positions`: exact cell edges, so masks see no staircase; dropped
  within 0.25 dx_target of the window edge or of a kept snap); the march runs per segment between them. A segment
  shorter than its target caps it (one cell; a slot sets dt; the bound adds the same cap). Each segment's remainder is
  rounded to whole cells and its cells rescaled to its length (never a short last cell: it would set dx_min, hence dt;
  selftest mesh_graded_end_cells), the correction tapered to ~0 toward snap nodes: adjacent ratios there can exceed
  max_ratio by ~3/n^2 for an n-cell segment, and an object spanning no whole number of target cells gets dx_min up to
  ~0.7 / n (relative) below target, so a self-test pinning dx_min sizes its box in whole target cells (e.g.
  heat_3d_graded_z_robin).
- S7.1 options (model side: model sub file, Meshing): `add_refinement_band(axis, lo, hi, dx, snap_edges)` refines one
  axis (the model meshes every object refinement through it, per axis; script regions / circles give the same
  profile); `MeshConfig(axis_max_ratios=)` per-axis ratio (`ratio(axis)`: march, slope and that axis' PML ramp;
  `max_ratio` the default); outer grading (`outer_dx`, `outer_ratio`): outside `outer_core(axis)` (span of that
  axis' refinement intervals; none without refinements) the target is outer_dx and slope / clamp use the outer ratio
  (`_graded_profile` takes per-sample ratios, `climb`), so cells grow from the refinement edge straight to outer_dx
  (dx_base matters only inside the core; PML cells still grade toward dx_base, i.e. shrink beside outer cells);
  `set_axis_edges` (via `complete_axis_edges`) replaces an axis' march (no snaps, bands or refinements along it);
  `segment_widths` = uniform / geometric / tanh pieces. `add_snap_edges` adds snap nodes without a band (no effect on
  `outer_core`); `window_edges(axis)` = float64 physical edges with snaps and ends exact (the float32 widths drift
  ~1e-7 relative): the builtin tetrahedron mesher's axes.
- Padding (`cell_spacings` / `build`, 3D `z_cell_spacings`): PML cells grow outward from the window edge cell by the
  axis max_ratio up to dx_base (adjacent ratios never exceed it); the CPML sigma grading still assumes dx_base cells,
  so a long ramp (finely meshed edge) absorbs somewhat less. `ghost_faces` (an EM mirror's ghost, duck-typed
  `boundary_config.ghost_faces`, passed by Domain.from_mesh and model.meshing) copy the edge cell's width exactly (a
  mirror image needs equal widths across the plane).

## Grid transfers (`grid_transfer`; S7.2 per-physics meshes)
- `GridTransfer(kind, source_edges, source_windows, target_edges, target_windows, measures, backend)` (or
  `.between(kind, source_domain, target_domain, axisymmetric)`): one CSR per axis (float32 values, int32 indices;
  scipy on 'cpu', cupyx on 'gpu'; float64 originals in `host_matrices`), applied separably. Edges are the Domains'
  full-grid edges in one frame (window lower corner = 0); only source WINDOW cells are read.
- 'conservative': weights = measure of each piece of the merged window edges / measure of its target cell, target
  window ends snapped onto the source's (float32 width sums differ by ulps; uniform meshes may overshoot the window by
  < 1 cell); measures from `axis_measures(coordinates, axisymmetric, dimensions)` (length; r dr of (r, z) / (r, phi);
  r^2 dr and sin(theta) dtheta of (r, theta)): the window integral is exact in float64 (`host_matrices`; the float32
  matrices ~1e-7 relative); target padding rows are empty (0). 'interpolate': linear between source window centres at
  every target centre (padding included), clamped to the outer centres (edge extension): exact for linear fields; no
  angle-wrap seam handling.
- Real arrays only (complex raises; float64 rounded to float32). Allocates the products per apply (fine at the
  coupling cadence; not timed: a time needs a device sync). Estimate: `memory_estimate.grid_transfer_estimate`.
  Domains carry `grid_kind` ('tensor'; fem 'triangles') and `mesh_name`; `model.mesh_transfer.make_transfer`
  dispatches (a triangle mesh on either side: `TriangleTransfer`, fem sub file).
- `MaterialGrid(n)` (ny None) is the 1D per-element grid of a triangle mesh: code taking `materials.nx / ny` as a cell
  grid checks the domain's `grid_kind`.

## Mirror planes (`mirror`, numpy only; symmetry 2.5)
- `mirror_faces(spec, physics=None)` = {face: sign}: EM bc_<face> 'symmetric' +1 / 'antisymmetric' -1 and the outer
  faces of the enabled `MIRROR_FEATURES` (membrane symmetry +1 / antisymmetry -1; `feature_mirror_faces`;
  `physics_mirror_faces` = both of one physics, also used by `parity_tag`). With a physics name, faces it does not set
  itself get +1 (non-EM fields unfold even: their face on the plane must be insulated,
  `definition_helpers.mirror_face_issues` warns otherwise), except the `FACE_FEATURE_SIGNS` types, signed by their own
  face feature on the plane: magnetostatics / magnetic_fields_ac (symmetry +1 = A even, magnetic_insulation or unset
  -1), magnetostatics_3d (the opposite: insulation or unset +1 = phi_m even, symmetry -1), membrane (symmetry or unset
  +1, antisymmetry / clamped / simply_supported -1); the last enabled one wins; selections expanded by
  `model.geometry.expand_entities` (core's one lazy model import). The model's mirror planes are the EM ones and the
  MIRROR_FEATURES faces: a magnetic-only model has none, a membrane half model has its symmetry face.
- Image rule (`image_sign`): polar (E-like) tangential s / normal -s, axial (H-like) the opposite, 'scalar' s, 'even'
  +1. `FieldInfo.parity` names it per renderer field ('' = even when unsigned, else unknown: `unfold_field` raises);
  `FieldInfo.staggered` = axes of hi-face samples (plane value 0 when odd, else the nearest sample: exact only for odd
  staggered components). Non-EM parities (s = +1, plane on the window edge): potential, laminar_flow, solid_mechanics
  (2D / 3D: shear ij = axial of the third axis), drift_diffusion, elastic_waves (v polar staggered, sxy axial_z on
  corners, normal stresses even; its 'rigid' face pins the window face where another physics pads, so the slip wall is
  on the EM plane; any other face there warns); magnetic types A / J polar_z, B / H axial_x / _y with their own face
  sign. `FieldInfo.plane_signs` ('x_lo:1 ...', `plane_signs_text`) replaces the face signs for one field
  (`field_plane_signs`, read by unfold_field): the nlfd channels (nlfd sub file).
- `unfold_field(field_data, faces)` (host, GUI-thread safe): x / y faces only (a z cut shows no z image); marks its
  result `FieldData.unfolded` and returns such a field unchanged (parity-sector snapshots are stored unfolded: never
  mirror twice; results_io saves the flag).
- `touch_factor(faces, touched)` = 2 per touched mirror face: Session 'integral' (selection reaching the window edge
  layer), 2D flux lines (`placement.line_mirror_factor`), 3D flux planes / boxes (`monitors3d.mirror_factor`), held
  currents / charges (`PotentialPhysics.mirror_factor`: a held face on or meeting the plane, held cells reaching its
  edge layer; faces from `set_mirror_faces` at create). The integrand is assumed mirror-even. heat reports no feature
  totals.
- `plane_wave_mirror_issue(face, sign, propagation, polarization)`: a TFSF box may open onto a mirror plane only when
  the incident wave is even about it (E_p tangential: symmetric; E_p normal: antisymmetric; never normal to the
  propagation); None = it may (2D `placement.tfsf_open_faces`, em3d `add_plane_wave`, both validations).
- Parity sectors (2.6): `sector_specs(spec, faces=None)` = 2^k spec copies over the mirror faces (each sign
  combination on every physics setting the face; MIRROR_FEATURES swapped by `_swap_mirror_features`, split per face
  when a feature selects several sector faces or other entities; `SECTOR_FEATURE_PARAMS` set on every copy, e.g.
  membrane port sector_drive; label in the title). A source without definite parity runs every sector:
  `combine_sector_fields` = mean of the unfolded fields (linear signed components only: unsigned / phase fields
  raise); `combine_sector_totals` = mean / 2^k of quadratic totals over regions / surfaces symmetric about every plane
  (not a line off the plane); `combine_sector_far_fields` = per angle the mean of the complex amplitudes (amplitude_re /
  _im, power = |amplitude|^2, 3D summed over a trailing x, y, z axis; power_per_cw rescaled).
  `combine_sector_result(results, n_faces)` = the rule stored per feature result: far_field the far-field mean;
  flux_spectrum with mirror_factor 2^k in every sector: power / power_per_cw as totals (entry combined_sectors); 'xy'
  with extras sector_rule 'linear' (`combine_sector_linear`: mean of each <name>_re / _im pair, _abs / _phase
  recomputed); 'xy' with sector_modes (`combine_sector_modes`: guide modes renumbered to the unfolded guide, signed mean
  of mode_<k>_re / _im, frequency_series / averaged_series alike; port powers = |combined amplitude|^2 x the sector's
  power per |amplitude|^2, port_incident_power = mean / sector_drive_scale, then `combine_sector_port_totals` sums the
  ports of one physics into incident_power / power_fraction_<n>: exact also for k > 1, unlike mean / 2^k); None
  otherwise (a flux line off a plane, tables, other probes). Study loop: `model.study_sectors` (model sub file).
  `parity_tag(spec, names)` = label text of the named physics' own mirror faces in sector_specs order (eigen snapshot
  labels); `own_mirror_faces(physics_spec)` the faces one physics sets itself. Self-tests em_fdfd_parity_sectors
  (scripted and study setting), em_fdfd_parity_sectors_eigen / _sweep, em_fdtd_parity_sectors_time (last two: per-angle
  far fields), membrane_parity_sectors_driven / _port / _eigen.

## Cut geometry (`cut_cells`, host numpy; solver_plan S7.6)
- Everything works on a membership test inside(x, y) -> bool of float64 host arrays (model.boundary_locator builds it
  from the drawn shapes): `crossing_fractions` bisects BISECTION_STEPS times (needs differing end memberships),
  `free_fractions` (outside share of segments) searches EDGE_SAMPLES sub-intervals with one crossing each (thinner
  features vanish), `free_areas` integrates AREA_ROWS midpoint rows along the axis the caller picks (pick by the face
  fractions, as physics/em/conformal_pec does; exact for straight edges away from kinks). Callers locate only links
  between cells whose centre tests differ: a cut no centre sees stays staircased. The 2D names wrap
  `segment_crossings` / `segment_free_fractions` / `rectangle_free_areas` (coordinate tuples of any dimension;
  em_fdtd_3d's conformal PEC passes 3D points); `union_test` takes any number of coordinates.
- `outline_pieces(region, x_edges, y_edges, inside)` (perimeter fluxes): every candidate cell (8-neighbours of a centre
  change) is tested on its faces inset by EDGE_INSET (1e-9 of a width), so an outline lying ON a cell face is a face
  piece (inside-length difference of the two inset faces: aligned rectangles exact, the window edge never an outline)
  and a cut through a cell a cell piece: -sum of outward face normals x inside lengths (the chord; exact for straight
  edges), its length refined by OUTLINE_REFINEMENT_LEVELS rounds of perpendicular-bisector points when it has two
  boundary transitions (several transitions keep the chord). Owner: the cell when its centre is in region, else the
  region 4-neighbour across the most inside face (the sliver-merge rule of the insulated walls), a diagonal one last;
  offset = owner centre -> piece along the normal.
- Polar / spherical windows: every function takes the logical coordinates (x = r - radius_min, y an angle); the
  caller's inside(x, y) maps them (boundary_locator with a DomainMap). Fractions are shares of the logical link / face /
  cell (exact along r and arcs; areas O(dr / r) off the metric weighting inside a cut cell). `outline_pieces(...,
  radius_min=)` turns lengths, normals and offsets into the local frame (dr, r d_angle) at each piece's radius, so arcs
  about the origin are exact. Links / faces across an angle_wrap seam are never cut (staircased).
- `near_centre_changes(region)` = the candidate cut cells (shared with outline_pieces); `rectangle_inside_fractions` =
  inside area share with the rows picked from the side fractions (2D FDTD conformal pole weights,
  physics/em/conformal_poles).
- `dual_cell_cuts(x_centres, y_centres, solid, inside)` (em_modes conformal PEC): free fractions of centre links whose
  ends differ and free areas of the mixed squares between four centres, floored at DUAL_CUT_FLOOR (0.02), 1 elsewhere;
  physics/em/cross_section_modes neutralises the entries crossing a wall plane (em sub file).
- Honouring types of `solver_defaults.conformal_boundaries` (`schema.conformal_boundaries_physics`: the type's
  `conformal_dimensions`, default (2,), and in 2D its `conformal_coordinates`, default cartesian; heat /
  electric_currents / electrostatics also polar / spherical; em_fdtd_3d (3,)): em_fdtd_3d (pec_domain Dey-Mittra H
  faces, em3d sub file), em_fdtd_ez / _hz (also cut-cell pole shares of dispersive materials, em sub file "Conformal
  poles"), em_fdfd_ez / _hz and em_modes (pec_domain; FDFD weights keep the operator symmetric, its
  `HarmonicSystem.pec_cells` = the fixed cells only), laminar_flow (Bouzidi, fluid sub file), heat / electric_currents
  / electrostatics (DiffusionSystem below). The locator (`model.boundary_locator.GeometryLocator`, host numpy:
  `inside(entities)`, `material_index(*coordinates)`, `material_test(index)`; 3D `geometry.contains_points_3d`) applies the build-mask
  rules; model sub file "Conformal boundaries".

## Coordinate metric (`metric`, numpy only; symmetry 2.16)
- `Domain(coordinates=Coordinates(...))` (default `CARTESIAN`) is 'cartesian' or 'polar' (x = r - radius_min [m], y =
  angle - angle_min [rad]). Widths, edges, centres, masks and `pos_to_cell` stay logical on a polar Domain;
  `Domain.dx` / `grid.dt` mix units there and mean nothing (2D). The space stays the physics' own flag: polar +
  axisymmetric = spherical (r, theta), theta from the rotation axis, 0..pi. The y_lo / y_hi faces of a polar window (at
  most 2 pi) are ordinary outer faces (insulated = mirror) unless `Coordinates.angle_wrap` (`ANGLE_WRAPS`: 'periodic' /
  'antiperiodic', planar (r, phi) only, `wrap_sign` +1 / -1; spherical raises in `window_metric`) joins them: u beyond
  y_hi = wrap_sign x u at y_lo, the real angular Bloch phases 0 / pi (DiffusionSystem is real). EM physics (polar FDFD,
  em_fdtd_3d sectors) read angle_wrap only as "faces joined": their phase is bloch_ky = the azimuthal order m [1/rad],
  F(phi + span) = F(phi) e^{-j m span} (antiperiodic C_N sector: m = N / 2).
- `Coordinates.line` (1D models): the window is one cell across (Cartesian 1 m: totals per m^2; polar one 2 pi cell:
  per m of cylinder; polar + axisymmetric one 0..pi cell: whole spheres) with the ordinary exact metric, so
  DiffusionSystem needs nothing (line order, tridiagonal: solvers.banded). Consumers read `line` only for units /
  labels (`label()`, potential `flux_unit` per m^2 on a slab).
- `window_metric(dx_window, dy_window, coordinates, axisymmetric)` -> `WindowMetric` of `Separable` products (along_x[i]
  x along_y[j], host float64): x / y face areas, physical widths (y = r_c d_angle on polar), volumes; exact for (x, y),
  (r, z), (r, phi) and (r, theta) (faces within `AXIS_ANGLE_SNAP` of 0 / pi get sin 0: float32 widths). Curvature
  enters only through areas and volumes (the arithmetic r_f / dr form): uniform sources are exact (parabolas),
  source-free radial profiles second order. Kernels gain 1D arrays, never per-cell metrics. `azimuthal_order=m` ((r, z)
  only, raises elsewhere): m != 0 weighs rings by `ring_angle(m)` = pi (the cos^2 integral) instead of 2 pi; code
  weighting (r, z) rings itself takes `ring_angle`, never 2 pi.
- Physics opt in by `PhysicsSolver.coordinate_grids` (default ('cartesian',); DiffusionPhysicsBase both;
  MagneticPhysics Cartesian: B / coils read Cartesian widths); `Simulation.add_physics` refuses a physics whose list
  misses the domain's grid. Keep it in step with `PhysicsType.coordinate_grids`; deliberate exceptions: FDFDPhysics and
  EMPhysics3D are polar in scripts only (em sub file FDFD "Polar", em3d sub file "Cylindrical sectors").
  `to_cartesian` / `from_cartesian` map logical positions to the drawing plane and back (planar phi wrapped about the
  sector centre); `Domain.from_widths` builds the padding-free polar Domain of the model path. Object-perimeter fluxes
  on 2D polar / spherical windows weight each exposed face by `WindowMetric.x_face_volume_ratios` /
  `y_face_volume_ratios` with the normal estimate on the physical widths (`definition_helpers.perimeter_flux_weights`).
  Not polar yet: thermal noise (heat refuses), drift_diffusion (Cartesian metric only), perimeter fluxes in 3D sectors
  and 1D models (validation refuses).
- 3D (symmetry 2.21): `Domain3D(coordinates=)` / `from_widths(..., coordinates=, backend=)` with polar Coordinates = a
  cylindrical (r, phi, z) sector (angle_wrap allowed, line refused); `window_metric_3d` -> `WindowMetric3D` of
  `Separable3` products (face_area[a] with n_a + 1 entries along a, length[a], volume). The 3D DiffusionSystem reads it
  (heat scripts and model path); em_fdtd_3d (scripts) reads `yee_radii` (r_c, r_f, link = mean of adjacent centres);
  every other 3D physics stays ('cartesian',), refused there by `Simulation.add_physics`. A polar Domain3D, unlike 2D,
  sets a meaningful `grid.dt` (`Grid3D.use_polar_time_step`: innermost arc r_c[0] dphi_min as the phi width; r at grid
  x edge 0 = radius_min minus the x_lo padding; a first cell centre at r <= 0 raises ValueError).
- Components are in the grid frame: `renderer.polar_field_infos(field_infos, axisymmetric)` relabels names ending _x /
  _y / _xx / _yy whose labels end ' x' / ' y' / ' xx' / ' yy' as radial / angular; a polar-ready renderer with such
  components sets it as its instance field_infos (potential does; heat has none).

## 3D contracts (B1: `grid3d`, `domain3d`, MaterialGrid; for 3D physics such as B2 FDTD / B3 heat)
- `Domain3D`: `grid` = `Grid3D` (float32 `dx_arr / dy_arr / dz_arr` incl. PML, `nx, ny, nz`, `shape`, `n_cells`, `dx /
  dy / dz` minima, `dt` = courant x 3D CFL limit), `materials` = `MaterialGrid(nx, ny, nz)`, `npml_sides` over
  `FACES_3D`, `ox / oy / oz`, `size_x / y / z` (requested), `uniform_dx` (None whenever any axis is graded) /
  `non_uniform`. Per-axis accessors return 3-tuples: `physical_window()`, `window_shape()`, `cell_widths / cell_edges /
  cell_centers` (float64 read-only), `host_centres(clamp)` (float32 1D build-mask centres, PML clamped),
  `coordinate_grids` (zero-stride X, Y, Z); `physical_extent()` is 6 values, `plane_extent(axis='z')` the 4 of a cut
  normal to axis; `pos_to_cell(x, y, z)`, `pos_to_cell_z`, `validate_physical_pos(x, y, z)`. No rect_mask /
  circle_mask (scripting is 2D).
- Flat index `(i * ny + j) * nz + k` (C-order): kernels get k-stride 1, j-stride nz, i-stride ny nz.
- z cells (`meshing._z_cells`): uniform dz = dx on uniform grids; on graded grids `MeshConfig(size_z=)` marches z like
  x / y (`add_refinement_band('z', ...)` = each refinement's object z_extent + margin, unset point / line z = the
  whole height; prism z0 / z1 = z snap nodes; `z_cell_spacings` pads the z PML). z bands never touch the xy profiles
  and vice versa (a band refines whole z layers). z PML counts come from `boundary_config.npml_sides['z_lo' /
  'z_hi']` (missing = 0, so the 2D EM BoundaryConfig gives no z PML: a 3D boundary config must add them and keep
  `differences()`). Consumers read per-axis widths (em3d `engine.widths`, fv_diffusion_3d, ElasticSystem3D), never
  `grid.dz`.
- Materials: `MaterialGrid.assign_window(name, window, mask, value)` writes a scalar or window-broadcastable array under
  a window-broadcastable bool mask (prefer it over full-grid `assign` in 3D). The builder paints an int16
  `ctx.material_index` (1 + region index) per z-slab run, one 2D raster per distinct cross-section
  (`geometry.cross_section_runs`), then writes each region's properties under its window mask; x, y dependent values
  are (wx, wy, 1) arrays, z-dependent ones are written one z plane at a time (range check accumulated on the device).
  `BoxValues`: a z-dependent `eval_field` value on the mask's bounding box (undefined outside). Accumulating consumers
  add it on the box only (`add_masked_to` / `copy_masked_to`; `DiffusionSystem.add_masked` / `copy_masked`: heat
  sources and initial T, potential sources, solid_mechanics_3d loads); consumers handing window arrays to a system call
  take `DiffusionSystem.window_values` = `window_array(window)` (zeros outside the box). `MaterialRegion.mask` is None
  in 3D (a full-grid mask only for regions with state-dependent properties, for their DynamicMaterialUpdate).
- BuildContext in 3D (valid until `release_caches`): `ctx.dimensions`, `object_box_mask(name)` (cached ((sx, sy, sz),
  cropped bool), read-only; `object_window_mask` raises), `domain_mask(entities)` (full-grid bool),
  `region_window_mask(region)` (window + fresh cropped mask: pole / dispersive boxes), `region_mask(region)` (full-grid
  bool), `height(name)` (z of a point / line, mid-height when unset), `point / line / rect` (xy). `eval_field` with x, y
  returns an (nx, ny, 1) layer, with z a BoxValues. `perimeter_mask / perimeter_weights` are 2D only.
- Geometry: `rasterize_box(name, resolved, x, y, z centres, xp)` / `box_window`, `rasterize_window(..., z=)` (cross
  section), `z_extent`, `point_z`; prism z intervals half-open with float32 compares, spheres closed.
- Renderers of 3D physics: `device_field(component)` returns the (wx, wy, wz) window array (a view where possible); an
  optional `device_plane(component, axis, index_window)` ('x' -> (wy, wz), 'y' -> (wx, wz), 'z' -> (wx, wy)) avoids
  materialising derived 3D fields; `Session.field / display_field(..., plane=(axis, position))` use it, else slice
  `device_field`. `Session.evaluate` is volume-weighted in 3D ('integral' = true volume integral) with
  `rasterize_box` selections.

## Harmonic probe record (`harmonic_probe`, numpy + core.backend; plan P0)
- FEM frequency_response solvers (solid, membrane, pressure acoustics) record complex phasors per Hz through
  `HarmonicProbeRecord(components, unit, label).record(frequency_hz, values)` (cupy 0-d values kept unsynced until
  `recorded()`) and `harmonic_probe_result(record)` ('xy' vs Frequency [Hz], series `<c>_re/_im/_abs/_phase`, phase in
  rad). `two_wave_fit` and `pole_fit` rely on the `_re` / `_im` names. The FDFD HarmonicProbe stays separate. Check in
  selftests_harmonic_probe, case registered in model/selftests_analytic.

## Resonance finder (`resonance`, host numpy; the shared row is `model.definition_resonance`)
- `ResonanceMonitor(probes, band_min, band_max, band, start_time, sample_delay_steps, point_names)` wraps probes of one
  component with `get_data() -> (times, values)` and `field_component` (same steps; traces cut to the shortest);
  several probes = one joint fit (`harmonic_inversion` of a 2D signal: per-channel Hankel matrices scaled to unit peak,
  stacked for the shared poles; amplitudes / phases per channel, the row's amplitude / phase from the channel where it
  is largest, the amplitude floor per channel). The analysis runs only in `result(waveforms, unit, dt)` (cached per
  (sample count, start, dt)): dt is passed at result time since a physics may fix its step only at finalize
  (elastic_waves). Reference time = recorded time - sample_delay_steps x dt (EM H: 0.5; probes stamping their own times,
  AcousticProbe velocities, declare 0). `pulse_end_time(waveforms)` = the latest `duration` of any waveform (gaussian /
  ricker 2 t0, acoustic HeatingPulse delay + 4.5 width).
- Its decimation is NOT the DFT `sample_interval`: the trace is mixed to the band centre, Kaiser-filtered (100 dB; pass
  |f| <= W, stop |f| >= rate - W) and decimated on the host; the FIR gain is divided out exactly per pole (polyval(taps,
  zeta)), and the first n_taps samples after the start are spent on the filter (low-Q modes decay there). Poles are
  kept only when two pencil sizes (N/3, N/2) agree within 1e-2 |s| / omega. The result must never raise for a short
  run (the study end computes every table result, `_feature_results`): failures go to the 'note' extra with empty
  rows. Band 'wavelength' adds the 'wavelength' column (vacuum C0 / f); 'frequency' has none.
- Memory: `analysis_host_bytes(n_steps, n_channels)` (host temporaries of one analysis); the growable probe traces are
  counted by the physics estimator through `model.definition_resonance.point_trace_items` (a FeatureType estimator does
  not see the study duration). The shared row's apply hands each point to the physics' `add_probe` as a one-object
  selection (`dataclasses.replace` of the ResolvedSelection), so add_probe keeps its single-object rule.

## Run-time estimate (`time_estimate` pure Python, `time_profile`; time_estimation_plan)
- Ops carry sizes only (bytes, rows / nnz, unknowns, flops, calls); `price` maps each to one profile curve:
  Elementwise -> 'fused' ('fused_numba' on the cpu when measured) at max(launches x the curve at bytes / launches, the
  curve at all bytes): the curve repeats one call on arrays of its size (cache-hot below the cpu L2 / L3 cliffs), while
  an op's launches stream distinct arrays, so an op's bytes are the distinct bytes all its launches move (a loop over
  the same small arrays is a count of single-launch ops, not launches). Spmv 'spmv', Reduction 'dot' (+ sync on
  read_back), Transfer 'transfer' and Sync only on the gpu, HostLU / HostSolve 'host_{lu,solve}_{2,3}d[_complex]' (no
  complex curve: COMPLEX_LU_FACTOR x the real one), DenseFlops 'gemm' / 'host_gemm', HostStream 'host_stream',
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
  point); no solver role, its knobs are physics settings. Array-module agnostic, imports numpy only.
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
  the (m + 1) x m Gram products per iteration. The state must stay meaningful under linear combinations (magnetics
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
- `assemble` caches face conductances, outer face terms, owned c_xx / c_yy copies (`assembled_c_xx / _yy`) and, with a
  2D cross term, `cross` (`CrossQuads`); `flux_at_centres` (= `axis_flux_at_centres` + `cross_flux_at_centres`) /
  `dissipation_density` read only that cache and raise while `operator_dirty` (`heat_flux_magnitude` re-assembles
  first; potential calls them right after a solve, E = -grad V from the axis part alone, J / D from both).
  `flux_jump_indicator(flux)` (S7.7 A2, `DiffusionPhysicsBase.error_indicator`) reads the same cache: a physics whose
  shown flux is not the system's (potential driven terminals, magnetic H with remanence, the AC phasor) passes it via
  `_indicator_flux`.
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
  values but not which c are > 0, nor held cells or faces (the magnetic B-H iterations after a loop's first). The check
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
