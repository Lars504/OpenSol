# open_sol/physics/fluid — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Laminar flow gotchas.

## FluidPhysics (`solver.py`, kernels in `lattice.py`)
- Type `laminar_flow` (default instance 'fluid') shares `rho` (one Param object) with heat.
- D2Q9 LBM on the physical window only, uniform square cells (prepare raises otherwise; every fluid feature has
  `requires_uniform_mesh`, so graded / auto meshes fall back to uniform). TRT with Lambda = 3/16 (halfway bounce-back
  exact for Poiseuille, steady results independent of tau), He-Luo incompressible equilibrium (u = j + F/2), Guo
  forcing, populations stored shifted g = f - w (float32 precision of small u). One fused pull kernel per step, two
  population buffers swapped.
- Backends ('gpu', 'cpu'): arrays on `domain.xp`; `to_host` on the cpu returns the live array: copy before keeping it
  (as `_refresh_wall_links`). Every kernel a `core.kernels` twin; a change to `_STEP_SOURCE` changes `_step_host` and
  `_step_numba` in the same edit (`_face_velocity_gradient_host` mirrors the kernel's if / else chain by `np.select`).
  `_step_numba` (single only) uses float32 constants / tables: a float64 literal or int table entry promotes to
  float64 in numba. It compiles with `cache=True` (uncached, the ~9 s compile lands in whichever of fluid_step_numba /
  the cpu_gpu parities runs first, so their wall times swap with the selftest order; a "Cannot cache" NumbaWarning = a
  closure value stopped pickling). The helpers `neighbour_velocity` / `face_velocity_gradient` are the module globals
  `_numba_neighbour_velocity` / `_numba_face_velocity_gradient` (`_numba_step_helpers`). As closure cells, their
  per-process dispatcher uuids missed the kernel's cache in every process. half_storage stays gpu-only.
- Step arguments (`_step_arguments`, `lattice.STORE_*`): `_run_steps(n, final_mode)` runs all but the last step at
  STORE_BOUNDARY (lattice macros written on the window boundary cells only, for the pressure rule's previous
  velocity; interior values stale), the last at STORE_LATTICE (march: the check reads them) or STORE_PUBLISHED
  (transient: lattice_to_physical fused in, zeros in solids; `_extend_into_ring` after). Host code reading
  `_lattice_x / _y / _deviation` does so only between calls. On the gpu, one fluid viscosity reads 1-element rates
  (`relaxation_stride` 0, `_update_relaxation`), and the 1-element forces `_uniform_force_x / _y` (`force_stride` 0,
  `_lattice_force_stride` per force refresh) when the physics never got a force (`_force_present`: add_body_force
  with a non-zero value, inject_source; zeros) or, stationary only, the lattice force is one value on every fluid
  cell (one min / max sync per refresh; transient keeps stride 1: no sync per step). The kernel reads forces on
  fluid cells only. The host twins take stride 0 too (broadcast / np.full) but the solver passes stride 1 there (one
  numba signature). A new writer of body_force_* sets `_force_present` and `_force_dirty`.
- Step graphs (gpu, `STEP_GRAPHS`): STORE_BOUNDARY runs replay CUDA graphs of GRAPH_STEPS (32) launches, the rest
  of the run graphs of 16, 8, 4, 2 (binary split, all even: buffers end where they started) and at most one single
  launch, so a check interval's leftover and a transient call of >= 3 lattice steps replay too (`_fixed_block`, as
  em3d: eager first, then capture; a failed capture warns and launches singly). The key holds the graph size, the
  argument ids (kept alive) and scalar values: populations, macros, rates, forces, face tables are rewritten in
  place, never reassigned while a graph lives; a new reassignable kernel argument joins `_step_arguments` (so the
  key). `_MAX_STEP_GRAPHS` (16) holds 5 sizes per argument set: below 5 the keys would clear before any capture. The
  run-time recipe counts one launch per march step.
- Checks (`_velocity_change`, steady march and transient health): one `lattice.velocity_check` pass (max |du_x|,
  max |du_y|, max u^2 as uint32 float bits by atomicMax: NaN wins; saved = u) and one 12-byte read, not a dozen cupy
  launches; peak = float32 sqrt of the max u^2 (the old float32 max |u|).
- Stepper precision (`FluidPhysics(precision=)`): half_storage stores g x POPULATION_SCALE (1024) as float16 via
  POP_LOAD / POP_STORE, everything else float32. Transient only (raises, validate errors). Host code writing
  populations goes through `_store_populations`; masking multiplies the uint16 bits (`stored = half_bits(array);
  stored *= keep`), since cupy's float16 kernels do not compile on every CUDA / cupy pair.
- Face codes (lattice.FACE_*, FACES order): wall (moving bounce-back), slip (specular mirror: the mirrored population of
  the upstream cell along the face, n - c_t; cell n itself for the normal link or where n - c_t is outside / solid),
  velocity (moving bounce-back, rho0 = 1), pressure (anti-bounce-back with the previous velocity; face value =
  deviation 3 p / (rho U^2); plus the TRT second-order term from the previous step's velocity gradient, read from the
  neighbours' post-collision populations since velocity arrays are rewritten in place: without it a sheared face
  injects transverse momentum (fluid_poiseuille_pressure ~0.15); the gradient must be at the link midpoint r - c_q/2,
  half a cell along the face for diagonals (at the cell: ~0.013 left)), periodic (both faces of an axis, else
  `_update_face_values` raises). Unreferenced faces are fixed walls. Corner diagonals take the y face's rule when it is
  wall / slip, else the x face's.
- Units: U = dx / dt, p = (rho_lat - 1) / 3 rho_SI U^2 (gauge), lattice force = f dt^2 / (rho_SI dx). dt_lbm =
  (relaxation_time - 1/2) dx^2 / (3 nu_max); other fluids tau = 1/2 + 3 nu dt / dx^2 (raises below 1/2 + 1e-3, warns
  below 1/2 + 0.02). A changed lattice dt (viscosity change when steady, or a runner step != dt_lbm) re-equilibrates
  at the same SI u, p (`_rescale_populations`).
- Fluid cells = viscosity > 0 (default 0 = solid); rho must be > 0 there. Cells turning fluid on a materials change
  start at rest.
- Interpolated walls (conformal_boundaries): `_refresh_lattice` rebuilds wall links when the fluid mask changes
  (`bouzidi.wall_link_fractions`; a material is solid where most of its window cells are); `step_kernel(precision,
  bouzidi=True)` applies linear Bouzidi on those links, outer faces keep their rules. Links across a periodic axis are
  located as the kernel streams them (`periodic_spans=`; a halfway seam column broke x-invariance, u_y ~1e-2). The wrap
  maps a point rounding onto hi back to lo (lo + mod(-tiny, period) == hi lies outside a shape drawn up to the window
  edge, and the bisection's first midpoint of a seam diagonal sits on the seam). Not mass-conserving; the steady
  result depends weakly on tau (halfway + TRT 3/16 does not).
- stationary: dt_limit inf, `step` marches only when dirty; convergence every `_check_interval` steps (max(100, 3.5 L,
  5 % of the slowest diffusive decay L^2 / (pi^2 nu_lat)); the decay L doubles on an axis with exactly one slip /
  pressure face: a quarter-wave mode). Auto max_steps = 30 decays (`steady_march_steps`, also the cap of the time
  estimate: never the raw max_steps, 0 = auto); the estimate's march (`definition._slowest_shear_rate`) ignores mode
  symmetry and acoustic transients. Converged when the geometric extrapolation r / (1 - r) <= tolerance * peak, or the
  change stopped decreasing below 1e-5 of the peak (float32 floor; without the stall test a tolerance below ~1e-6 would
  always hit max_steps), on two checks in a row; else `n_unconverged` / `last_warning`.
- A march from rest with one pressure face is slow (decay length 2 L: ~280k steps for 100 cells, ~1 ms / step on cpu):
  models with known flow start by initial_velocity; otherwise relaxation_time ~2 cuts steps ~3x (halfway walls only:
  the Bouzidi steady state moves with tau).
- The float32 steady state is biased: per-step rounding (~1e-7 of u) acts as a source the slowest decay integrates, up
  to ~eps32 x decay steps (falling_film u_surface at 10 / 20 / 40 cells: ~1.4e-5 / 4e-5 / 2.3e-4, growing ~(H/dx)^2,
  unchanged by a longer march); reference tolerances below that cannot pass.
- transient: dt_limit = dt_lbm, ceil(dt / dt_lbm) steps per call, health check (NaN, Mach warn 0.2, raise 0.7) per check
  interval. A shorter runner step (a wave physics) lowers tau: validation errors for EM FDTD (`mesh.dt`) and for
  elastic_waves when `acoustics.definition.estimated_time_step` (imported inside `_wave_step_issues`) gives 3 nu dt /
  dx^2 < MIN_TAU_OFFSET. `n_solves` counts marches / transient calls (coupling revisions).
- SI outputs (u_x, u_y, u_abs, pressure) full-grid, written on the window after each march / step, edge-extended into
  a PML ring; vorticity on request (renderer).
- initial_velocity (`set_initial_flow`): kept until `finalize` (dt_lbm known); `_start_from_initial_flow` writes the
  equilibrium populations, sets `_lattice_dt`, publishes t = 0 fields. Must stay after `_refresh_lattice` (it zeroes
  cells that were not fluid before).
- Values of t (model sub file "Feature values of t"): faces by `set_face_value` (same kind, `_faces_dirty` only, no
  lattice rebuild; a parabolic inlet re-reads its rows, one host sync per change), forces by adding the change to
  body_force_*_ext. Stationary flow re-marches after each change. The published velocity lags the probe time by ~half a
  step (`selftests_time.py`).

## Convection (`convection.py`)
- `AdvectiveHeatSource` is only the transform; the coupling fluid_heat_convection (`coupling/definitions_fluid.py`) is
  a 'sum' CouplingTerm (source u_x) with `also_reads` (fluid u_y, heat T), so it re-transfers when fluid OR heat solved,
  and puts both in one stationary Picard loop (relaxed as a loop-back). Reads heat's solve-precision `system.unknown2d`,
  `bcs` (Dirichlet faces = inflow temperature), `T0`, rho / cp; advective form, central faces, nothing through solid
  faces.
- The Picard loop converges only while transverse conduction damps every mode (held walls: gain ~ Pe / (2 pi)). With
  only the inflow face holding T, the y-uniform axial mode has gain ~ 2 Pe L / (pi H) (~20 in
  parallel_plate_uniform_flux_nusselt): no relaxation helps. Such models march a time_dependent study instead (steady
  flow, transient heat, u dt / dx < ~1, a few flow-through times).

## Self-tests (`selftests.py`)
- `FLUID_CASES` holds fluid_poiseuille_force, fluid_poiseuille_pressure, fluid_heat_convection_plug (starts at u = U by
  initial_velocity: from rest ~280k steps) and fluid_poiseuille_startup_half_storage; `SELFTEST_CASES` registers only
  the half_storage start-up, the others run inside the parity cases (`_COVERED_BY_EXAMPLES`). Template
  heated_microchannel uses library water.
- `selftests_backend` (heavy): fluid_cpu_gpu (incl. fluid_bouzidi_channel, measured = max u_x so the parity is on a flow
  value), fluid_heat_convection_cpu_gpu, fluid_step_numba (`.cpu` vs `kernel(None, None, args)` shares the numba compile
  with fluid_cpu_gpu; a twin that fell back to numpy gives error inf; without numba it passes vacuously; at
  STORE_PUBLISHED into a ring-padded full grid, the first face set at stride 0). The gpu store modes, strides and
  step graphs are covered only by the fluid_cpu_gpu parities.

## Triangles / cuDSS: decision (agent_help batch, 2026-10-23; nothing implemented)
- LBM stays on the uniform tensor lattice (`grids = ('tensor',)`); no unstructured LBM (loses exact streaming, adds
  numerical viscosity; Bouzidi already gives curved walls). No linear solve, so cuDSS has no role.
- The real candidate is a separate steady FEM path on triangles (the march is slow and float32-biased): Taylor-Hood P2
  / P1, Picard or Newton, float64; saddle point (Stokes symmetric indefinite LDL^T, Oseen / Newton LU), one factor per
  iteration on one pattern -> cuDSS plan reuse, double. Needs vector P2 + P1 assembly, Dirichlet walls, do-nothing
  outlet, slip by normal constraint, pressure gauge, per-element outputs, fluid_heat_convection on triangles. Cost L;
  gate: a steady model whose march time or float32 bias the LBM cannot meet. Transient flow stays LBM.
