# open_sol/physics/semiconductor — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Drift-diffusion gotchas. Module roles: their
docstrings (`structure_index.md`); the Newton / continuation / chord / pattern rules: `drift_diffusion.py` comments and
docstrings; back ends, fallbacks and estimates: `_linear_solve` / `_factor_solve` and `definition.py`.

## Numerics
- Geometry enters only through `face_length` (ring face area when axisymmetric) and `cell_area` (ring volume), so a new
  term built from them needs no axisymmetric branch; triangles carry 2 pi in their P1 weights.
- Chord steps count in `newton_iterations` (and `chord_steps`): factorisations < iterations, so a self-test or time
  check counting factorisations per Newton iteration must allow for them.
- Newton stall in strong inversion (gated stacks, no current): a near-null Jacobian mode (surface electrons tied to the
  body only by SRH generation) stalls updates near psi_s = 0.8 V with tau 1e-6 s; the MOS examples use tau 1e-10 s. A
  solver-side fix (equilibrium shortcut or quasi-Fermi variables) is open.
- Analytic models of uniform bars: semiconductor regions a few Debye lengths apart couple electrostatically, and an
  insulator cell between held contact cells bends the potential at the contact edge; use two physics on one region or
  keep insulators off the contact columns.
- A contact current is a difference of majority-carrier SG terms, so it has a float64 round-off floor ~1e-16 of q D N /
  h, independent of the Newton tolerance (~1e-12 A/m at N_D 1e24, 2 nm cells; shockley_diode_iv: 3 % at ~2e-11 A/m).
- Mirror planes: `set_mirror_faces(mirror_faces(spec))` at create; `contact_currents` / `gate_charge` (so table
  results and scalars) multiply by `mirror_factor` of the region's held sites (tensor: the window edge cell layer;
  triangles: a held node on the window line), the core stays drawn-half. The `symmetry` feature has no apply: every
  outer face is already insulating, the exact even mirror of the FV cells and of the box method. Covered by
  examples/semiconductor/analytic_photoconductor_gain.

## Memory and precision
- The Newton state, Jacobian and SG / Bernoulli terms stay float64 on the host (carriers span ~1e-150 to 1e26 scaled,
  contact currents are differences of SG terms); the device holds only the float32 output fields (read by couplings
  and state-dependent materials) and a cuDSS factor. The kept coupled factor lives across solves (plan reuse); a cold
  start frees it before the equilibrium Poisson LU, so the estimate's Poisson phase before the coupled one holds.
  Parking the output fields would save ~70 B per full-grid cell beside a factor of a few kB per window cell: kept.

## Self-tests
- Tensor-mesh physics is checked by the examples (examples/semiconductor, examples/multiphysics); cases cover only what
  no example does. Self-tests needing SuperLU pin a scipy_splu node (Automatic is cuDSS on the gpu). Solve cost grows
  with the y extent (flat index i * wy + j: band ~3 wy): keep self-test windows short in y.
- Not covered: gates on triangles, and the EM couplings on triangles (sized per element by `materials.shape`) in a model
  run.
