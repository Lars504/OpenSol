# open_sol/physics/fluid — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Laminar flow gotchas. The lattice scheme, face rules,
store modes, strides and twin rules: `lattice.py` comments; units, march, step graphs and checks: `solver.py` constants
and docstrings; settings: their help texts.

## FluidPhysics (`solver.py`, kernels in `lattice.py`)
- Interpolated walls (Bouzidi, `bouzidi.py`, conformal_boundaries): not mass-conserving, and the steady result depends
  weakly on tau (halfway + TRT 3/16 does not); outer faces keep their rules.
- A march from rest with one pressure face is slow (decay length 2 L: ~280k steps for 100 cells, ~1 ms / step on cpu):
  models with known flow start by initial_velocity; otherwise relaxation_time ~2 cuts steps ~3x (halfway walls only:
  the Bouzidi steady state moves with tau).
- The float32 steady state is biased: per-step rounding (~1e-7 of u) acts as a source the slowest decay integrates, up
  to ~eps32 x decay steps (falling_film u_surface at 10 / 20 / 40 cells: ~1.4e-5 / 4e-5 / 2.3e-4, growing ~(H/dx)^2,
  unchanged by a longer march); reference tolerances below that cannot pass.
- Transient: the published velocity lags the probe time by ~half a lattice step (`selftests_time.py` tolerances).

## Convection (`convection.py`)
- When the stationary Picard loop of fluid_heat_convection converges (held walls vs only the inflow face holding T,
  whose axial-mode gain is ~20 in parallel_plate_uniform_flux_nusselt) and the time_dependent alternative: the
  coupling description (`coupling/definitions_fluid.py`).

## Self-tests
- The gpu store modes, strides and step graphs are covered only by the fluid_cpu_gpu parities (`selftests_backend.py`).

## Triangles / cuDSS: decision (nothing implemented)
- LBM stays on the uniform tensor lattice (`grids = ('tensor',)`): no unstructured LBM (loses exact streaming, adds
  numerical viscosity; Bouzidi already gives curved walls), no linear solve, so no cuDSS role. A steady flow on
  triangles would be a separate float64 FEM path (Taylor-Hood P2 / P1, Picard or Newton, one LU per iteration on one
  pattern: cuDSS plan reuse), worth it only for a steady model whose march time or float32 bias the LBM cannot meet;
  transient flow stays LBM.
