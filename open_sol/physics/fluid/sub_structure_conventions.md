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
- Symmetry feature = the slip face (kind 'slip' in `_FACE_FEATURE_KINDS`) as a model mirror plane; slip is the exact
  even mirror of the lattice (halfway bounce-back where the mirrored upstream cell is solid). Corner diagonals beside a
  slip face take the other face's rule (two slip faces: the double mirror), so a face-rule change touches the corner
  rule. A parabolic inlet run ending on a slip face takes the half parabola of its image (`_parabolic_shape`). The
  pressure faces' TRT shear term reads neighbours beyond a slip face as their mirror image (`neighbour_velocity`, all
  three twins): its one-sided fallback at the plane biased the whole face column ~0.14 / N (u_max of the half
  poiseuille_channel_pressure 7e-3 at 20 cells). Not mirror-exact: Bouzidi links of a solid touching the plane, a wall
  moving normal to the plane. Validation: another
  face feature or a periodic pair on the plane errors, another physics' plane where the flow face is not slip warns,
  a body force along a plane normal warns (its image is the opposite force). Session integrals of u normal to a plane
  are odd: their unfolded value is 0, not the doubled half. Covered by examples/fluid/poiseuille_channel_pressure.

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
