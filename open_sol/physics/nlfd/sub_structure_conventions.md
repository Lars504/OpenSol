# open_sol/physics/nlfd — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first; FDFD gotchas: `open_sol/physics/em/sub_structure_conventions.md`).
NLFD-chi: cascaded 2D frequency-domain chi(n) mixing (`nonlinear_optics_plan.md`).
Type `em_fdfd_nonlinear` (family 'em', default instance 'nl', studies nonlinear_frequency_domain and optimization):
its channel features wrap the em_fdfd_ez / _hz FeatureTypes, so a change there changes it too.

## Layout and imports
- `tensors.py`, `mixing.py`: numpy + core.constants only (validation, GUI thread). `sheets.py`: numpy at module level
  (site builders / evaluation take `xp`, cupyx only inside `_scatter_add`). No nlfd module imports cupy; the rest work
  on the run backend.
- Backends ('gpu', 'cpu'; no cpu refusal left): every array on `domain.xp`; sources' math functions take `xp`
  explicitly, iteration / adjoint helpers the module of their inputs (`array_module_of`), host copies by `to_host`;
  masks checked as bool arrays of `self.xp` (`_check_mask`); a private linear solver gets
  `backend=domain.backend.name`. Self-test `nlfd_cpu_gpu` (heavy).
- definition_sheets.py is imported by definition.py and imports `_model_channels` from it only inside its line
  validate (cycle); its smoothed apply imports model.boundary_locator inside the function. selftests_sheets.py imports
  selftests.py private strip helpers (registered after the other cases); selftests_backend.py imports selftests /
  selftests_sheets / selftests_adjoint builders and checks (registered last).
- physics/nlfd imports physics.em, never the reverse; `__init__` exports ChannelSpec and NonlinearHarmonicPhysics,
  never the definition. definition.py imports physics.em definition modules and registers selftests_feedback /
  selftests_adjoint cases after the selftests.py ones.
- Linear solver: the 'linear' role (DIRECT_LINEAR_ROLE); the one runtime is shared by every channel FDFDPhysics (each
  keeps its own factor; a back_end string builds a private one), so its record holds the summed factorisations;
  `work_counters` = channels, sweeps, solves.
- Per-channel boundary configs (symmetry 2.2 / 2.5): `NonlinearHarmonicPhysics.boundary_config` is the shared one (k =
  0, the inputs' mirror parity; a Bloch phase raises: k is per channel); each channel gets
  `solver.channel_boundary_config` with `Channel.bloch_k` (inputs: harmonic_channel bloch_kx / _ky; generated: the
  coefficient combination of the inputs') and, for an even channel (`Channel.parity_exponent` = 0), antisymmetric faces
  turned symmetric (P of E^n has parity s^n). Terms landing by frequency only must match k (mod 2 pi / L) and parity
  (`mixing.inexact_term_issues`: solver ValueError at `_build_mixing`, validation `_channel_phase_issues`). The physics
  settings omit the FDFD bloch_kx / bloch_ky.
- Bloch wrap factors in C / C*: `ChannelFieldStore.wrap_factors[index]`; C of a channel's field takes its own factor,
  C* of a target density the target's (`centres_to_faces(wrap_factor=)`, `density_face_currents(target=)`,
  `DensityPhasor(target=)`), the adjoint transposes the source's / target's. Sheet sites are not phased (refused with
  k != 0 on any channel: `_check_sheet_bloch`). The Kerr operator M is periodic, so its face_average needs no factor.
- Unfolding: `NonlinearRenderer.field_info` adds `FieldInfo.plane_signs` of the selected channel where they differ
  from the physics faces, so even channels unfold with their own parity; poling is 'even'.

## Model layer (definition.py, model/study_nonlinear.py)
- Channels come from the enabled harmonic_channel features at create (feature name = channel name; no apply);
  `_model_channels` / `_channel_specs` are the one CPU resolution for validation, memory estimate and create.
- Channel features (`_CHANNEL_FEATURE_NAMES`) = the em_fdfd_ez / _hz FeatureType of the channel's polarisation:
  params = 'channel' + the union of both forms (field_probe component default None = the out-of-plane field); FDFD
  apply / validate / result get a feature-spec view with only their own params and (validate) a physics view with only
  that channel's channel features, so TFSF / mode-source checks stay per channel. Single channel only (no 'all' copy).
  pec_domain and conducting_sheet act on every channel (sigma_s per channel omega; own validate, since the FDFD one
  reads the 'wavelength' setting this type lacks); effective_index, resonance_table, phase_change_medium are not
  offered. adjoint_objective is a channel feature; design_region is the em_fdfd FeatureType itself on
  `add_design_region` (writes eps_r, eps_xx, eps_yy). Optimization: pump_depleted with a chi3 medium needs
  iteration_method newton (`adjoint._check_differentiable`, which also refuses eps_xy).
- Material vocabulary = the em_modes rows and schema 'modes', so all three permittivity arrays exist;
  `permittivity_arrays` / `inject_source` take every registered one.
- `state()` always has ez_re / _im and hz_re / _im (zeros for an absent polarisation) and `NONLINEAR_FIELD_INFOS` is
  the union of both FDFD renderers, so state_fields / output_fields hold for every instance.
- Validation (`_mixing_issues`) uses the same mixing table as the solver. Undepleted guard
  (`_undepleted_validity_issues`): generated field ~ omega |P_NL / eps0| L / (2 c) per term (point sources /
  field-dependent values not estimated: their terms skipped), warning above 0.3 of the weakest source field. Overlapping
  media of one order: warning (tensors add). pump_depleted: warnings for cusolver_qr (refactorises every sweep / GMRES
  product) and mode lines crossing a chi3 medium; free_carriers undepleted: ignored (warning). visible_if is ANY, so
  relaxation / anderson_depth stay visible for newton (help says 'picard only'). The nonlinear settings are own Params
  appended after the FDFD settings.
- Reused sweep builds (model sub file "Reused sweep builds"): NonlinearHarmonicPhysics is a RerunSolver
  (`reset_records`: dirty + `ChannelFieldStore.clear`, a cold start from zero fields with versions bumped, never
  reset to 0, or version-keyed caches such as the power scalars could match a stale entry; channel monitors keep one
  value per omega, so a rerun overwrites them; channel factors stay). Cold, not warm: each value gives the result of
  its own build (no Kerr hysteresis branch picked by the sweep order), and on the SHG example sweep the warm start
  took 226 instead of 204 channel solves. Only `_FOLLOWING_SOURCE_PARAMS` follow (line_source / gaussian_beam /
  plane_wave_tfsf amplitude, point_source current / magnetic_current / dipole_moment, mode_source power, and phase;
  re-set by the built source's `set_amplitude(*values of its amplitude_params)`: the flags are the union, the apply
  follows the names its phasor declares); any other swept value (geometry, media, wavelengths) refuses the reuse. A new
  following param needs an in-place setter on its FDFD source object, never a rebuilt source (the channel's source
  list and the nonlinear phasors hold the object).
- Per-process chi3 (`thg_scale`, `fwm_scale`) rides on `MediumTensor.process_scales`, so validation and the guard see
  it; Miller's rule needs chi1 at each channel frequency and applies only at build (`_miller_table`): validation lists
  unscaled factors, the run log scaled ones.

## Self-tests and templates (selftests.py)
- Strip cases (`_strip_spec`, `_channel_strip_spec`): y-periodic strip, one medium in every cell, so the run is the 1D
  3-point problem and references are exact lattice sums (`_lattice_sh`, `_cell_wavenumber`). Changing the strip (graded
  mesh, material boundaries, a second medium) breaks them. Depleted SHG phase-matches by birefringence
  (`_birefringent_eps`) with lattice CMT references (1e-2 = CMT accuracy). SPM: delta 2e-3 sets half its tolerance, do
  not raise it. QPM, THG, OPA gain are checked only by the examples/em analytic_nlfd_* models.
- `nlfd_newton_tangent` uses `_TangentStore` (channels / versions / key only): FreeCarriers must keep needing nothing
  more from the store (`refresh_from_store` aside). Depleted cases run once per iteration_method.
- Templates: `ppln_slab_shg` dbeta / C_cmt are computed by `_ppln_design` for the default slab, wavelength and mesh
  only; the LN index is an inline one-pole Lorentz fit, not a library material. `silicon_thg_waveguide`: the nk_fit
  range must hold the TH (pump 1.36-1.44 um). `shg_saturation_tanh`: `_depleted_design` valid for that strip only.
  `shg_metasurface_optimization` coefficients illustrative.
- Study: `solve_nonlinear(before_channel, after_sweep)` (checkpoint per channel solve and per Newton GMRES product),
  snapshots in `cascade.order` ('(not converged)' appended), the first input channel re-selected; summary
  `nonlinear_solve`, `feedback`. Solution feedback (model sub file): couplings from the nonlinear physics into
  background physics run feedback rounds (pump_depleted warm-starts from the store).
- selftests_feedback.py (`nlfd_tpa_heat_feedback`): its slab has no linear loss and one ez channel (changing either
  breaks the Q reference); the sigma3 sheet's q is ~1e3 x the bulk peak, so items are normalised per region.
- selftests_backend.py (`nlfd_cpu_gpu`): reuses the spec builders and checks of selftests / selftests_sheets /
  selftests_adjoint, so changing those changes it too; probe scale floored at 1e-12 of the largest probe.
- selftests_adjoint.py: `nlfd_depleted_adjoint_gradient` uses a fictitious d33 10 nm/V (10 um/V makes picard
  diverge); in `nlfd_sheet_carrier_adjoint_gradient` n0 comes from background_index (a block-eps-dependent n0 would
  enter the finite difference but is held in the adjoint). Only the *_gradient runs carry the eps_sensitivity derived
  value (`_gradient_derived`: it triggers the adjoint).

## Channels and the mixing table
- A channel = (frequency, polarisation): 'ez' carries E_z, 'hz' E_x, E_y. A 'both' spec becomes `<name>_ez` and
  `<name>_hz` (two factorisations: memory doubles).
- Generated frequencies = integer combinations of the inputs (`coefficients`): a process lands exactly by its
  coefficient vector; separately given wavelengths match within 1e-12 (`exact` False), near misses < 1e-3 listed. No
  automatic closure of the frequency set.
- Rank: inputs 0, a combination 1 + max rank of its channels; `solve_order` = rank, then omega. Undepleted keeps terms
  whose sources all rank below the target (no SPM / XPM / back-action); pump_depleted applies every term.
- One term per multiset of signed sources, factor 2^(1-n) x orderings: valid because `MediumTensor.from_tensor`
  symmetrises source indices. Kleinman symmetry is NOT enforced (`tensors.kleinman_asymmetry`; non-Kleinman values
  make the power identity inexact).
- Entries below 1e-14 max |T| dropped (`COMPONENT_RTOL`); a term with no entry for the channel polarisations is dropped
  with a note.
- Crystal classes: full Voigt forms (3m ties d15 = d31; 6mm / mm2 / 32 keep d15, d24, d14 free). Cut presets put the
  c axis along z / x / y by cyclic permutations.

## Sources and collocation
- ez: rhs += -omega^2 mu0 P_z at the centres. hz: P at the centres from E collocated by C =
  `harmonic.face_mean_at_centres`, distributed by `sources.centres_to_faces` (C*, the adjoint of C in the area-weighted
  products), J_F = -j omega C* P, exposed as `face_currents`. Any new nonlinear term must collocate / distribute through
  C / C* (or its own exact adjoint pair): only then is sum_k W_k = 0 for real Kleinman tensors
  (nlfd_collocation_transpose). C* is not `yee_grid.face_average` at non-periodic ends. The C* / C*^T weights live on
  the backend in `ChannelFieldStore.weights(axis)` (`sources.collocation_weights`); run-time callers pass them
  (`weights=`), so a face-current or adjoint evaluation uploads nothing.
- Sources never read a channel's FDFDPhysics, only `ChannelFieldStore` (flat complex128, captured after every channel
  solve, version per channel): a channel's own solve cannot change what its sources returned for it, and pump-depleted
  mixing writes the store only. Reading `physics.face_fields()` inside a source would recurse for hz self-terms.
- `NonlinearPolarizationPhasor` caches P and face currents by the sources' store versions; undepleted: an uncaptured
  source raises, pump-depleted (`skip_unsolved`) gives 0. Full-grid buffers per target channel (and per diagnostic
  phasor after `scalars()`): run items of the memory estimate.
- Media `cell_factor`: chi(2) poling sign int8, chi(3) float32 n0^2 (`background_index`, else
  `background_index_squared(omega_ref)` evaluated once at apply: later eps changes do not update it). The chi(3)
  MediumTensor is built with n0 = 1.
- Kerr (pump-depleted, picard only): `KerrOperatorTerm` puts the diagonal `kerr_diagonal` into
  `set_extra_permittivity` on the media's union box before the channel solve and adds back -(that change) on the stored
  field as a source, so the fixed point is the rhs form (power-exact); the operator part only stabilises. Sources in a
  Kerr region calibrate with the lagged Kerr eps, ports never see it; q_abs includes its Im part.
- Free carriers (`FreeCarriers`, chi3 media with a `CarrierModel`): N = lifetime max(G, 0) per medium box, cached by
  every Kerr channel's store version; every channel's phasor adds the carrier term (rhs form), with picard each channel
  also gets a KerrOperatorTerm carrying it lagged; such a phasor lists all carrier channels in `source_indices`. Keep N
  float64 first (`N * n0 * complex`): float32 times a Python complex is complex64 in cupy.
- Newton tangents (`NewtonLinearization`): a new nonlinear term must add its exact tangent here, else Newton converges
  only linearly (or not).

## Sheets (sheets.py, definition_sheets.py; plan 3.3 / 3.4)
- A `SheetMedium` lives in `solver.sheets`, never `solver.media` (poling_map, Miller, carriers, Kerr boxes read media).
  Its terms enter the one mixing table through a representative tensor computed from the same local tensor in
  validation and build; `MixingTerm.components` of a sheet term only filter / describe, values come from `site_tensor`.
- Sites sample the raw store fields (`grid_fields`), never the C-collocated centres, and deposit the exact adjoint (ez
  centre density, hz face densities added to the face currents without C*). `power()` = bulk centre work + sheet site
  work; the Kerr operator term and carriers read `bulk_terms` only; absorption, nonlinear_absorption and the adjoint
  read both.
- `sheets._scatter_add` on cupy is one float64 `cupyx.scatter_add` into the complex target's real view (interleaved
  re / im slots): targets are flat contiguous complex128, and the deposit needs no full-grid temporaries (the 'sheet
  evaluation' phase holds site-sized arrays only).
- `sheet_site_absorption` books site work in the cell owning the tap, so sum A q = -site work exactly.
  `sheet_cotangents`: no C^T (the sites sample the store layout).
- Surface sites = faces between a selected and an unselected cell; E_n = the C mean at the selvedge cell. 'smoothed'
  normal from the drawn shape (bisection between the face's cell centres + a circle of radius 1/4 around it; staircase
  normal where a bracket fails); length = face x |n_across|. Never from cell fill fractions (mesh-independent ~2 deg
  error stalled nlfd_surface_disk_shg's convergence). Line sites = conducting_sheet sites (hz `sheet_line` refuses the
  outer grid wall whenever an hz channel exists).
- sigma3 sheets are two media (THG / TFG and '<name>:kerr') separated by process scales 0; 'current' response divides
  by j omega of the TARGET, so no power identity holds. Sheet SPM / XPM: rhs form only (no KerrOperatorTerm, no
  channel q_abs share).
- Self-test gotchas: `nlfd_surface_sheet` relies on E_n = 0 in the strip and the float32 grid width (`_YEE_CELL`) for
  1e-9, face E_y from `_lattice_face_value`; `nlfd_sheet_nonlinearity` ez_pump is exact only because h2 and h3 are both
  rank 1 with one source sheet each; `nlfd_surface_normal_jump` judges the order only when |r - 1| >= `_ORDER_FLOOR`
  (symmetric probes cancel O(h) terms); `nlfd_surface_disk_shg` (heavy, gpu, eps 1 disk): first-order error, so 5 nm
  must not be worse than 10 nm.

## Solver
- `NonlinearHarmonicPhysics` owns one FDFDPhysics per channel on the shared Domain / MaterialGrid; features apply to
  `channel(name)`. Phasors attach at `finalize` (or first solve): `_build_mixing` raises when nothing couples or nothing
  acts undepleted.
- `inject_source` writes the shared permittivity array once and calls `materials_changed` on every channel. `state()`
  absent polarisation reads a shared zeros array: never write into state arrays. `take_absorbed_power` (absorption.py)
  = sum over channels (q_abs - `extra_permittivity_loss`) + `nonlinear_absorption_density` (+ sheet site absorption):
  method independent. A new lossy nonlinear term adds its density there.
- Cascade adjoint (adjoint.py, `eps_sensitivity()` cached per channel n_solves and objectives; renderer
  eps_sensitivity / design_density are physics-wide): reverse solve order, per channel one `transposed_channel_solve`,
  P cotangents via `polarization_cotangents` (Wirtinger) and `collocation_transpose` onto the sources
  (`_channel_backward`, one writer for both modes). pump_depleted (`depleted_cotangents`): (I - T) nu = T(c_obj) by
  `iteration.real_gmres`, RuntimeError above ADJOINT_FAIL_RESIDUAL; needs factors of the plain operator, so picard Kerr
  / carrier operator terms are refused (and eps_xy). A new term needs its transpose here.
- `scalars()`: power_exchange_imbalance is an implementation check, not convergence.
- Memory estimate (`_memory_estimate`): every channel's `fdfd_memory_estimate` part (labelled per channel: each keeps
  its own system and factor, so their run items and cuDSS external items add, their transient phases stay apart) +
  the nlfd part. Run items: store, cached densities / face currents, cell areas / q_abs total, the cascade adjoint
  EpsSensitivity (once per physics; channel adjoint_objective features have no estimate) and the pump-depleted forward
  buffers (Anderson / Kerr terms, Newton basis / tangents: freed after the solve but alive across channel
  (re)assemblies, so never a phase). Phases: 'nonlinear densities' (collocated hz fields, box products), 'absorbed
  power', 'cascade adjoint' (temporaries + the picard GMRES basis; after the forward solve). Sheet features: the kept
  `_sheet_density` is a run item, one evaluation's site / scatter temporaries their 'sheet evaluation' phase.
- pump_depleted picard (`iteration.solve_pump_depleted`): block Gauss-Seidel sweeps in solve order, residual = update
  norm (not operator residual), `AndersonMixer` with real gamma (G conjugates fields: only real-linear); the loop stops
  right after a sweep, so store and channel solutions agree. Continuation wraps linear sources in `ScaledSource`
  (attribute pass-through, ports keep working), restored in `finally`. Not converged: CascadeRecord converged False;
  non-finite: RuntimeError.
- Store-vector algebra (Anderson Gram and mix, `real_gmres` CGS2, `block_norms`) runs on the float64 views of the
  contiguous complex128 vectors (Re <u, v> = view dot, real coefficients = real GEMVs, no conj copies or (depth, N)
  temporaries) with one device -> host transfer per Gram, GMRES column or set of block norms: what `apply` returns
  and what reaches those helpers must be complex128 (made contiguous there).
- newton (`iteration.solve_newton_krylov`): G_J = block Jacobi; J v by swapping a channel's sources for its tangent
  `DensityPhasor` for one solve (restored in `finally`; never reassembled). `real_gmres` in Re <u, v>; `block_weights`
  give a zero channel the largest weight, never 1 (fields ~1e8 V/m: weight 1 lets a generated channel dominate the
  Armijo norm and every step backtracks the pump to 0). After the loop the store captures G_J(x). Memory
  (GMRES_RESTART + 1 + NEWTON_BUFFERS) store vectors (definition imports the constants). One block_norms transfer per
  evaluation gives both the residual and the scaled ||F|| (the accepted trial's norms are the next step's), and
  `real_gmres(rhs_norm=)` takes that ||F||, so a Newton step syncs once per GMRES column and once per trial.

## cuDSS (agent_help batch; verdict, what is wired)
- Wiring: channels factorise through the shared 'linear' runtime, so a cuDSS back end reaches them without nlfd
  changes. `adjoint.transposed_channel_solve` tries `solve_adjoint`; on NotImplementedError it solves D A^-1 D^-1 b
  (D = `channel_symmetrizer`) and checks the residual against SYMMETRIZED_TRANSPOSE_RESIDUAL (1e-8) -> RuntimeError
  naming scipy_splu. A Bloch phase re-raises NotImplementedError. A back end without a transposed solve must raise
  NotImplementedError (any other exception skips the route). CPU self-test `nlfd_symmetrized_transpose`.
- Validation: a back end other than scipy_splu / cusolver_qr / the host direct ones with picard + Kerr gives an info
  (refactorises every sweep; newton reuses one factor per channel).
- MUMPS / PARDISO (DIRECT_LINEAR_ROLE lists `HOST_DIRECT_BACK_ENDS`; opt-in, either backend): the channels are
  FDFDPhysics, so they factor the LDL^T of D A on the host (physics/em sub file "MUMPS / PARDISO"), recycled by GMRES
  like SuperLU (picard Kerr / carrier sweeps keep the factor, the time items' SuperLU recycling holds) and solved
  transposed natively (`transposed_channel_solve` never needs the symmetrizer route for them); the symmetrizer-route
  error text names them beside scipy_splu. Memory: fdfd_memory_estimate's host factor items per channel.
- Verdict: candidate, strongest for pump_depleted newton and the depleted adjoint; LDL^T of D A (Bloch / eps_xy: LU),
  'mixed' once the 2D Yee single_residual is below the 1e-3 gate, else double. Picard Kerr / carriers refactor per
  sweep unless plan reuse is used.
- Why 'auto' is scipy_splu on the gpu: DIRECT_LINEAR_ROLE has no `auto_back_ends` (by design). Right for the strip examples
  (~1e4 unknowns per channel: gstrf ~2 ms, a SuperLU solve ~0.2 ms, below cuDSS's per-solve host overhead and its
  ~15 ms plan per channel build); a device factor pays from ~1e5-1e6 cells, where SuperLU solves cost 10-100 ms and a
  Newton run takes 50-200 of them.
- Opt-in: DIRECT_LINEAR_ROLE (em_fdfd_nonlinear only) lists the FDFD back ends (cudss / cudss_mixed included) but no
  `auto_back_ends`. The time recipe prices a back end without GMRES recycling (cuDSS, cusolver_qr) as one
  refactorisation + solve per picard Kerr re-solve ('Kerr re-solves (reassembly, refactorisation)'), no recycled GMRES
  items. The em_fdfd adjoint has the same transposed-solve gap; moving the symmetrizer route into
  FDFDPhysics.solve_adjoint would make `transposed_channel_solve` a plain call.
- Unmeasured on the opt-in: factor_seconds vs SuperLU, device_bytes, single_residual / refinement steps, solves per
  nonlinear solve, transposed-solve residual, wall time vs scipy_splu on a >= 1e5-cell grid; the cuDSS path has no
  nlfd self-test.

## Triangles: decision / plan (agent_help batch; nothing implemented)
- Gate: channels would inherit the em_fdfd_ez / _hz triangle path, which first needs a driven dof-vector solve,
  `_solve(record_monitors=False)`, load-vector sources, per-element extra permittivity, `solve_adjoint` (D = I) and
  per-element permittivity gradients. ~2-3k lines in nlfd after that.
- Yee-bound pieces to replace: ChannelFieldStore layout, C / C*, media boxes, KerrOperatorTerm, sheets, FreeCarriers,
  absorption, adjoint face weights / wall rows, renderer, undepleted guard, estimates.
- Mixing on triangles: P at quadrature points from E = Phi u, b = Phi^T W P (exact transposes: the work identity holds
  by construction); rule degree (n + 1) p. hz: prefer an E_t Nedelec formulation. Kerr / carriers newton-only first.
  Sheets on interface edges (exact normals). No periodic triangle meshes, so the strip references do not carry over.
  FEM pencils are complex symmetric (LDL^T without a symmetrizer).
