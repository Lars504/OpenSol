# open_sol/physics/semiconductor — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Drift-diffusion gotchas.

## Layout
- `drift_diffusion.py` (numpy float64, host; cupy only in the cusolver_qr branch): `DriftDiffusionCore` = the
  geometry-free Newton / continuation / residual / Jacobian / contact-current core on sites (`cell_area`) and faces
  (`face_lo` / `face_hi`); `DriftDiffusionSystem` = tensor window cells (flat index i * wy + j; interior x faces then y
  faces). `drift_diffusion_fem.py`: `FEMDriftDiffusionSystem` on triangles. A geometry's set_inputs writes the arrays
  `_finish_inputs` lists, then calls it.
- `solver.py`: `DriftDiffusionPhysics` (float32 full-grid `fields` on `domain.xp`, gpu and cpu, edge-extended into the
  PML ring). `solver_fem.py`: `FEMDriftDiffusionPhysics` (`self.window = slice(None)` so the inherited feature API
  indexes per-element arrays).
- `couplings.py`: transforms of the two EM couplings (buffers on the run backend's xp, the plasma kernel a
  `core.kernels` twin); their CouplingTypes live in `coupling/definitions_semiconductor.py`. `definition.py`: type
  'drift_diffusion' (default instance 'semiconductor'; shares `eps_dc` / `eps_dc_xx` / `eps_dc_yy` and their arrays
  with electrostatics).

## Numerics
- Axisymmetric (shared `definition_helpers` setting): `face_length` = ring face area and `cell_area` = ring volume from
  the (r, z) `core.metric.WindowMetric`; every term follows from those two arrays, so residual rows need no branch.
  Contact currents / gate charges are whole-ring A / C (`current_unit` / `charge_unit`). The x_lo outer face has zero
  area.
- Unknowns interleaved per site (psi / V_T, n / C_s, p / C_s; C_s = max(|doping|, n_i) of the semiconductor sites).
  `charge_fraction` = semiconductor share of a control volume (tensor 0 / 1, interface triangle nodes in between):
  Poisson charge, recombination and their Jacobian carry it; carrier unknowns exist where it is > 0. cusolver_qr gets
  the interleaved system as a (wx, 3 wy) grid (band estimate up to 3x low). Rows are L1-equilibrated before every
  factorisation; every back end factorises the real Jacobian (SuperLU and cuDSS `real=True`).
- SG face terms: electron current out of the lower cell = -q T_n, hole current q T_p (contact currents and J fields use
  exactly these); d = 0 when a side is no semiconductor.
- Pinned unknowns (identity rows): contact cells all three (neutral equilibrium + V / V_T), gate cells u, insulator
  cells n and p. Gates are ideal conductors (held-face Poisson conductance). A contact cell without n_i > 0 is no
  contact cell; a gate on a semiconductor cell raises at solve. Gate potentials are referred to the intrinsic level:
  textbook phi_ms = work_function_difference + V_bulk (stated in the help).
- Newton: potential updates log-damped, carrier updates multiplicative when negative, floored at 1e-150 C_s (an
  underflowed n made dn / n overflow and NaN reach SuperLU as 'exactly singular'); a non-finite residual, Jacobian or
  update is a non-converged Newton (continuation halves its step), never factorised. Continuation from the last
  converged state (rescaled when C_s / V_T change): voltages (contacts then gates) and generation ramp, step <=
  max_bias_step, halved down to 1/64; a failed warm continuation restarts cold, a failed cold one raises RuntimeError.
  Secant predictor (`_predicted_state`) from a ramp's second converged step, half the iteration limit; a failed
  prediction is counted (`rejected_predictions`) and the rest of that ramp goes unpredicted.
- Study warm_start: `warm_start_fields` = the core's `converged_state()` (host physical psi / n / p / generation per
  site + the voltages); `seed_warm_start` keeps it as `_converged` when sites, contacts and gates match
  (`seed_converged_state`, else a cold start), so the next sweep value ramps from the previous value's voltages. It
  seeds no state slot (fields come from the solve); `warm_start_extrapolate` stays off (own secant predictor).
- Newton stall in strong inversion (gated stacks, no current): a near-null Jacobian mode (surface electrons tied to the
  body only by SRH generation) stalls updates near psi_s = 0.8 V with tau 1e-6 s; the MOS examples use tau 1e-10 s. A
  solver-side fix (equilibrium shortcut or quasi-Fermi variables) is open.
- Analytic models of uniform bars: semiconductor regions a few Debye lengths apart couple electrostatically, and an
  insulator cell between held contact cells bends the potential at the contact edge; use two physics on one region or
  keep insulators off the contact columns.
- Recombination: SRH + radiative B + Auger. Auger coefficients (~1e-43 m^6/s, float32 subnormals) are no material
  property: the auger_recombination feature writes float64 constants per window cell / element into the system
  (`set_auger`, later features win), never the MaterialGrid.
- `max_iterations` setting = solver kwarg `max_newton_iterations`; `temperature` defaults to 300 K (library data), by
  design not heat's T0 293.15 K; selftests pass it explicitly.
- RerunSolver (sweep reuse_build): `reset_records` only marks dirty (the converged state stays the continuation
  start); ohmic_contact / gate `voltage` follow live parameters (`set_contact_voltage` / `set_gate_voltage` write the
  core's voltage lists; the table results read them, never the frozen region's build value). Doping, generation and
  materials are not followed (a spec with them live is rebuilt per value).
- `prepare` constructs a new system, dropping doping / contacts / gates while `self.contacts` keeps their indices:
  only `__init__` calls it; a re-prepare would need the features re-applied.
- Every input change (features, `materials_changed` of a schema array, `inject_source('generation')`) marks dirty;
  `step` re-solves only then and counts `n_solves` (coupling skip rule).
- Scalars `current_<contact>` / `charge_<gate>` after the enabled features (reads physics_spec); {} before the first
  solve.
- Contact currents = continuity reaction of the held sites (faces to non-members + held volume (R - G)): 0 volume term
  on tensors, on triangles the lumped generation of free elements on outline nodes, so carrier balances are exact on
  both. Gate charges = D flux out of the gate sites.
- A contact current is a difference of majority-carrier SG terms, so it has a float64 round-off floor ~1e-16 of q D N /
  h, independent of the Newton tolerance (~1e-12 A/m at N_D 1e24, 2 nm cells; shockley_diode_iv: 3 % at ~2e-11 A/m).

## Triangle meshes (`drift_diffusion_fem.py`, `solver_fem.py`)
- Box method on the P1 nodes of the WINDOW elements (PML ring not solved; ring nodes get the clamped window field):
  face coefficients = sum of `fem.assembly.p1_edge_weights`, so Poisson is exactly P1 FEM with lumped charge; control
  volumes = `p1_lumped_weights`. Weights carry 2 pi when axisymmetric.
- Node inputs = semiconductor-volume means of element values (a junction on element edges has mixed doping at its
  nodes). Contacts hold every node of their semiconductor elements, gates every node of theirs (potential only; no
  held-face conductance), later features win shared nodes; generation from free semiconductor elements only.
- Monotonicity needs non-negative face coefficients (Delaunay); a negative one is counted (`negative_faces`) and logged
  once per new count via `runtime_warnings`. Not clipped.
- Outputs: nodal fields linear across an insulator element touching the semiconductor (a one-element carrier tail in
  display / integrals); per element E = -grad psi, J = least-squares of the element's SG edge projections
  (`_tangent_solve`).
- Linear solver: SuperLU or cuDSS (cusolver_qr refused: no grid index order); size rule on 3 x nodes (SuperLU only).
  Time estimate: none; memory constants uncalibrated.
- Couplings size buffers by `ctx.domain.materials.shape`, so they work per element. Untested in a model run.

## Self-tests
- Tensor-mesh physics is checked by the examples (examples/semiconductor, examples/multiphysics); cases cover only what
  no example does. `selftests.py` holds no case: shared fixtures (pn junction specs, `_check_pn_junction`) and the
  template 'silicon_pn_photodiode'. semiconductor_cpu_gpu pins a scipy_splu node. Solve cost is set by the y extent
  (band ~3 wy): keep self-test windows short in y.
- `selftests_fem.py` (cpu, triangles; tolerances first estimates). Gates on triangles have no self-test.

## cuDSS and MUMPS / PARDISO 'linear' back ends
- Role `CUDSS_LINEAR_ROLE` (shared with em_fdfd_bor / _conical), tensor and triangles (size rules return None on
  cuDSS). `_linear_solve`: equilibrated COO -> host CSR (explicit zeros kept), `factorize_host(real=True, previous=)`:
  real LU (nonsymmetric Jacobian). `_kept_factors` keeps the last factor per unknown count (Poisson n, coupled 3 n) on
  `drift_diffusion.KEPT_FACTOR_BACK_ENDS` (CUDSS_BACK_ENDS + host_direct.HOST_DIRECT_BACK_ENDS: cuDSS and the opt-in
  MUMPS / PARDISO back ends; any other back end falls through to the cusolver_qr grid path, so a new kept-factor back
  end joins that tuple): the pattern is fixed while the pinned set is, so later iterations refactorise on the kept plan
  / analysis; a changed pinned set re-plans (pattern compare). The Poisson factor is released after the cold start
  (`release_kept_factors`). CudssAccuracyError / HostDirectAccuracyError are re-raised as their own ValueError,
  Cudss- / HostDirectSingularError as the singular-system one; a MUMPS / PARDISO refusal never falls back.
- Host direct size rules: `solver.linear_size_issue` gives MUMPS / PARDISO the scipy_splu rule (unknowns_per_cell 3;
  else size_issue would size them as a cusolver_qr grid), `solver_fem.fem_linear_size_issue` the SPLU limits named by
  the library (uncalibrated). Memory: `definition._host_factor_items` (the SuperLU heuristic x
  `host_direct.superlu_bytes_scale`, real LU, + `solver_host_bytes`); time: the sparse_direct recipe prices them as
  SuperLU items. Self-test semiconductor_host_direct (`selftests_cudss.py`, cpu; skipped without python-mumps /
  pydiso): the forward junction on pardiso / mumps / pardiso_mixed vs a scipy_splu node.
- Automatic: 'auto' resolves to cuDSS on the gpu with nvmath. A refusal (inside `factorize_host` or a deferred
  cudss_mixed check in `solve_host` -> `linear_solver.fall_back`) switches it to SuperLU for good and releases every
  kept cuDSS factor: `_factor_solve` reads `linear_solver.back_end` per call, never a cached choice. Self-tests needing
  SuperLU pin a scipy_splu node; semiconductor_cudss also runs Automatic expecting auto_fallbacks 0.
- cudss_mixed: float32 factor refined to 1e-12 or replaced by the double factor (counters single_residual, double
  fallbacks).
- Estimates: memory `_cudss_estimate` (factors made in the study: the coupled factor's full E copy
  (`operator_copy_bytes(symmetric=False)`), rhs buffer, row scale and device-route index maps (`index_map_bytes`, no
  transpose map: an LU) are run items; the cold start's Poisson LU is released before the coupled one is planned, so
  its whole life is one phase before them (`core.memory_estimate.phase_before_items`); transient phases
  'refactorisation' (refresh or probe peak, `refresh_peak_bytes(n_unknowns=, mixed=)`) and CUDSS_SOLVE_PHASE
  (`solve_peak_bytes`); only the coupled factor external (`factor_device_bytes`, value_bytes 8, dimensions 2;
  photodiode measured the coupled one alone outside the pool); CSR entries per cell derived from the stencil (29:
  Poisson row 5 + n + p, each continuity row 5 + 5 V + the other carrier)); time on the
  cuDSS 2D curves at value_bytes 8. time_check: each plan item is measured by the role's cudss_plans x 1 /
  `_PLANNED_SYSTEMS`, never by newton_iterations.
