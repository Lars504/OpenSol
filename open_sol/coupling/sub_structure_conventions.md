# open_sol/coupling — conventions & gotchas

Sub file of the root `structure_conventions.md`. Runner and coupling gotchas.

## Runner and couplings
- Cadence: dt_sync = min `dt_limit()`; a tick steps solvers with `tick % every == 0` over every * dt_sync, then applies
  the couplings on the runner-owned global tick (a coupling added mid-run stays in phase): ticks every-1, 2 every-1,
  ... (just before a target with that every steps: window-averaged Joule terms), or with `after_source_step` ticks 0,
  every, ... (right after a source with that every steps, else its targets lag one more interval): DynamicMaterialUpdate
  with source fields, heat_em_thermo_optic / field_transfer when the source's every equals the coupling's; scripts pass
  it (`standard.thermo_optic` defaults True, right for every None).
- The clock (`runner.t`, `tick`) advances BEFORE the couplings apply, so a coupling sees the start time of the next
  steps; time-dependent values read it there (`material_update.py`: a DynamicMaterialUpdate of t = every the smallest
  stride, no after_source_step, rewritten on every cadenced tick; TimeDependentUpdate = a feature value of t, `target`
  its physics, every its stride, setter(value_at(runner.t)) when the clock moved).
- `prepare` raises when dt_limit < every * dt_sync (EM `every > 1` raises; transient heat needs every * dt_sync <=
  time_step) or a coupling names an unattached physics; resolves a CouplingTerm / CouplingGroup every=None (script
  default) to the largest stride of its source / target (the model's null every); sets
  `getattr(solver, 'engine', solver).enable_q_accum` when a coupling reads q_accum / q_accum_time (not jz_total: live
  with any Drude pole). It returns stride-cost `prepare_warnings` (only `Simulation.run` warns them; the model path
  validates strides): every * dt_sync < 1e-3 dt_limit, a `solves_every_step` solver (steady heat) at every < 100, a
  q_accum / q_accum_time reader whose every differs from its target's (each transfer resets the window). The runner
  never calls `solver.prepare` / `finalize` (`Simulation.finalize` does: builder, first `run`).
- `flush()` (end of every time-dependent run, also after cancel): `flush` of every coupling without after_source_step,
  one `step(t, inf)` per quasi-static physics, then `flush` of the after_source_step ones (their quasi-static targets
  are left dirty, not re-solved).
- Coupling protocol: `apply`, `apply_now`, `flush`, `source_fields`, `every` (int, or None for prepare) (+ optional
  `after_source_step`, default False); a relaxed stationary loop-back coupling also needs `transfer_value` / `write`
  on it or its `terms`. `CouplingTerm.apply_now` = `transform(field, source, target)` (None injects nothing), its
  `flush` uses `flush_transform` (default transform); `CouplingGroup(terms, every)` applies its terms in order (their
  every unused); DynamicMaterialUpdate (flush = apply_now) writes under its mask, then `materials_changed(frozenset)` on
  every solver.
- Change check: `apply` skips a transfer while `coupling_term.state_revisions` of its source (`state_revision`, else
  `n_solves`, else `engine.n_step`; a solver with neither always transfers) is unchanged since the last `apply_now` /
  `flush` (those stay unconditional). Sound only because every state change marks the solver dirty: a new solver must
  re-solve and count after a slot injection or a `materials_changed` of an array it reads, else readers keep stale
  data. A transform reading more than `source_field` declares those `(physics, field)` pairs as
  `CouplingTerm(also_reads=...)` (they join source_fields, loop-back detection and the check); reading the target's own
  state also sets `CouplingType.reads_target` (the time estimate's fixed point re-solves it).
- Backends: CouplingTerm / SlotSum / field_transfer buffers and every transform run on the run backend (a kernel is a
  `core.kernels` twin; a persistent buffer is allocated at create on `ctx.xp` or lazily on its input's module, never
  with `cp`); no backend rule of its own: validation refuses the cpu through the partners' `PhysicsType.backends`. A
  partner method a transform calls (acoustics `elastic_strain` / `bound_charge`, the EM loss protocol) is that
  physics' port. `selftests_backend.py` (coupling_cpu_gpu, coupling_fdtd_cpu_gpu, pockels_cpu_gpu;
  `coupling_backend_case`: parity 1e-3): a case whose check merges sub-checks lists each parity value in details
  'parity_measured', else only 'measured' is compared.
- Writer rule (model.validate `_check_slot_writers`, from `CouplingType.writes`): a (target, slot) takes one
  overwriting coupling ('' = field_transfer) or any number of E13 contributors of one kind and one effective every; a
  state-dependent permittivity property on a written eps slot is an EM validation error. EM `ez_source` / `hz_source`
  keep the array reference (exactly cupy float32 (nx, ny), checked; FDTD pole / PEC masks exactly cupy bool:
  `_check_cell_array`) and add it every step; `eps_r_update` overwrites all of eps_r (PML incl.) and rebuilds at once;
  `eps_xx/yy_update` copy and leave the E face coefficients dirty; heat `volumetric_power` copies into `Q`.
- 3D chains (em_fdtd_3d + 3D heat) run the same couplings on (nx, ny, nz) arrays (buffers take the MaterialGrid
  shape); thermo-optic writes only the target's `permittivity_arrays` (em_fdtd_3d: eps_xx / eps_yy only when
  allocated). Self-test em3d_heat_chain: physics/em3d/selftests_materials.py. The (1 + alpha dT) expansion factor is
  the in-plane (and 3D / BOR phase-index) image; on a cross-section plane (`definitions._CROSS_SECTION_EM_TYPES`:
  em_modes) it goes on eps_zz only (`thermo_optic(with_expansion=('eps_r',))`); a new cross-section EM type joins
  that set.
- EM loss protocol (every absorbing EM solver): `state()['q_accum']` (None allowed), settable `enable_q_accum` on
  `getattr(solver, 'engine', solver)`, `take_absorbed_power()` = FDTD mean W/m^3 since the last call, then reset (None
  for an empty window); FDFD q_abs of the last solve (never reset; None before the first). em_heat_joule accepts only
  family-'em' types with an 'absorbed_power' output (not em_modes: q_accum None, no take_absorbed_power), lists
  'q_accum' only to enable accumulation, and its every must equal the heat every; `flush` sends the last partial window
  merged with the previous full one (`merge_previous=True`, also `last_q_accum`), so the final Q does not depend on
  duration mod every * dt. A cw window W is exact only over whole half periods (else off by |sin(omega W)| / (omega
  W); `_cw_ripple_issues` warns above 1 %). Transient heat steps [kW, (k+1)W] with the power of [(k-1)W, kW]: first
  interval sourceless, last window never stepped. From em_fdfd_nonlinear it is a solution-feedback coupling of the
  nonlinear_frequency_domain study (model sub file) carrying linear + two-photon + free-carrier absorption.
- heat_em_thermo_optic (every family-'em' type): one 'index' contributor per index slot of the target (ez
  `eps_r_update` from n0_zz / dndT_zz; hz `eps_xx/yy_update` from _xx / _yy; em_modes all three), dn = (n0 + dndT dT)
  sqrt((1 + alpha_xx dT)(1 + alpha_yy dT)) - n0: the alpha factor goes on eps (never on n: that doubles it), the
  transformation-optics image of the in-plane stretch (H_z only matches the phase index), exact only for alpha uniform
  over the mode (alpha-0 cladding under-counts). Library alpha is free-crystal (a clamped film: override). An enabled
  solid_em_strain_optic with deformation on the same EM physics (`solid_supplies_deformation`) drops the alpha factor
  (`with_expansion=False`). Isotropic dndT expands to _xx / _yy / _zz; no off-diagonal dndT / alpha; applied right after
  each heat step (EM sees T of the previous window's power). Dispersive cells: n0 = sqrt(background eps_r / eps_inf),
  pole chi untouched (shift too small by ~n(omega) / n0; validation warning only).
- electrostatics_em_pockels (`definitions_pockels.py`, transform `physics/potential/pockels.py`): 'index' contributors
  like thermo-optic, dn_ii = -1/2 n0_ii^3 r_ii E_k (E_k = E_x / E_y by `field_component`; r_xx / _yy / _zz arrays on the
  electrostatics schema, default 0; n0 = the E13 reference), plus r_i1 E_x + r_i2 E_y when the crystal arrays
  pockels_r11 .. _r62 are allocated (clamped r^S: piezo strain adds its own dn via acousto-optic). The xy row goes into
  em_fdfd_hz eps_xy_update; anti-plane yz / xz rows, and xy on in-plane types without that slot, are dropped
  (`definitions_acoustics.crystal_off_diagonal_issues` warns per crystal, also for acousto-optic). E is 0 inside
  terminals and edge-extended into the PML ring, like V.
- Extension modules: `definitions.register` registers its own types, then imports every `coupling/definitions_*.py`
  (pkgutil, name order) and calls `register(registry, physics)` with `definitions.RegisteredPhysics` (names, EM /
  absorbing / optical type tuples); each returns early without its partners. A new coupling topic = a new
  `definitions_<topic>.py`, no edit in definitions.py. They import definitions' E13 helpers, so definitions never
  imports them at module level; order must not matter (a check of another module's coupling type belongs in the
  template or selftest, not in register).
- Off-diagonal eps_xy (em_fdfd_hz): `index_terms.index_contributor(..., off_diagonal_transform=)` appends a 'sum' term
  into eps_xy_update after the index terms (one group, so the diagonal SlotSums record n0 first; transforms call
  `index_reference.in_plane_reference_indices`), declared by `writes_index_and_off_diagonal_slots`. Delta eps_xy =
  -n0_xx^2 n0_yy^2 Delta(1/n^2)_xy (first order), added to the material eps_xy. Transforms:
  `acoustics.couplings.acousto_optic_shear_shift` (always data) and `potential.pockels.pockels_shear_shift` (None
  without the tensor arrays: nothing injected, no eps_xy allocated).
- semiconductor (`definitions_semiconductor.py`, transforms `physics/semiconductor/couplings.py`):
  semiconductor_em_plasma_dispersion = isotropic dn = -(a N + b P^c) (one shared buffer per drift_diffusion
  `n_solves`, each term copies it); em_semiconductor_generation overwrites 'generation', G = eta take_absorbed_power()
  / (hbar omega): an FDTD source needs photon_wavelength and no second absorbed-power reader (each take resets the
  window: validation).
- acoustics (`definitions_acoustics.py`, transforms physics/acoustics/couplings.py; cadence notes in the
  physics/acoustics sub file): acoustics_em_acousto_optic ('index' + the hz eps_xy term) and em_acoustics_photoacoustic
  ('sum' into heating_power, q_abs copied; covered only by examples/acoustics/photoacoustic_beer_lambert). Self-tests
  selftests_acoustics.py (acousto_optic_uniform_strain: also the normal-strain xy row p61 / p62 into hz eps_xy;
  acousto_optic_shear_strain: plane S wave in a periodic box, e_xx = e_yy = 0 exactly, g recomputed from sxy).
- piezo (`definitions_piezo.py`): electrostatics_acoustics_piezo = a CouplingGroup of two overwriting terms E_x / E_y ->
  electric_field_x / _y (acoustics applies the stress at injection), after_source_step when the strides match.
  back_action 'bound_charge' (default) prepends an overwriting acoustics -> electrostatics 'charge_density' term
  (bound_charge_transform; source field 'sxx' unused) and `_CoupledResolve` (no slot, `_source_changed` False,
  `target` for the runner check) calling `solve_coupled_source(acoustics.time)`: charge -> re-solve -> E in one apply,
  no lag; create also calls `enable_bound_charge` (stiffened dt) before runner.prepare. writes() declares only the E
  slots, so validation checks the charge_density writer itself (error for another writer; every == the acoustics'
  every); a 'sum' charge contributor would leave the group (SlotSum routing) and break the order. 'none' = one-way.
  Self-tests / templates in selftests_piezo.py; tfln_acousto_optic_modulator registers only when em_fdfd_hz is an
  absorbing optical type (it also uses acousto-optic and Pockels; untuned). Case notes: piezo_thickness_slab_launch
  (one-way; plates on the y faces: uniform E_y, slab faces act as force sheets; exact trace at the vx face, half a cell
  above the probe point), piezo_bound_charge_strip (two-way, vs the 1D FV closed form of the recorded strain),
  piezo_stiffened_bar_resonance (heavy; stiffened dt; one-way run for c^E), electrostatics_waveform_superposition (the
  elastic_waves fluid is only the clock).
- fluid (`definitions_fluid.py`, transform `physics/fluid/convection.py`): fluid_heat_convection = 'sum' into heat
  volumetric_power with `also_reads` heat T (advective source depends on T). Selftest / heated_microchannel template in
  physics/fluid.
- magnetic (`definitions_magnetic.py`): magnetic_heat_induction, magnetic_solid_lorentz_force; notes, transforms,
  selftests in physics/magnetic (sub file).
- phase change (`definitions_phase_change.py`): heat_em_phase_change is a side-effect coupling: its transform hands
  (T, heat.time) to `physics/em/phase_change.update_phase_change` and returns None, but it declares '' writes on every
  eps slot (one term per slot) so the writer rule and the runner refuse other eps writers there. Errors outside
  time_dependent / for steady heat. em_heat_joule warns about transient heat only for FDTD sources (FDFD + transient
  heat is the GST switching set-up). Self-tests / template: physics/em selftests_phase_change.
- membrane (`definitions_membrane.py`): solid_membrane_tension = a group of three overwriting terms stress_ij x h ->
  coupled_tension_ij (h = param thickness, else the solid thickness); planar plane-stress solid on the membrane's
  triangle mesh; needs StudySpec.solved_physics ['membrane'] (solid = background; validation info otherwise). The 2D
  solid has no periodic setting: a periodic-cell prestress = rollers on all faces on a mirror_x / mirror_y mesh (equals
  the periodic state at zero mean strain), valid only for a mirror-symmetric cell. Self-tests selftests_membrane.py
  (membrane_film_tension_uniform, membrane_film_tension_cell_bloch_band: heavy, M-E1(a) / (b)).

## Registered coupling types (catalogue)
- Always: `em_heat_joule`, `heat_em_thermo_optic`, `field_transfer`. Conditional (module, registers when ..., slot):
  `currents_heat_joule` (electric_currents); `em_semiconductor_generation`, `semiconductor_em_plasma_dispersion`
  ('index'; drift_diffusion); `fluid_heat_convection` (laminar_flow + heat); `heat_solid_thermal_strain` (targets
  solid_mechanics and _3d), `solid_em_strain_optic` (source solid_mechanics; 'index'), `em_solid_optical_force` (2D;
  'sum' into body_force_x / _y) (all physics/mechanics); `magnetic_heat_induction` ('sum' into volumetric_power),
  `magnetic_solid_lorentz_force` ('sum' body_force, beside em_solid_optical_force) (definitions_magnetic: a magnetic
  type + heat / solid_mechanics); `electrostatics_em_pockels` (electrostatics + an EM type; 'index', adds with
  thermo-optic); `acoustics_em_acousto_optic` ('index'), `em_acoustics_photoacoustic` ('sum' heating_power)
  (elastic_waves + an absorbing EM type with steps_in_time None and 'time_dependent' support, i.e. em_fdfd_*);
  `electrostatics_acoustics_piezo` (elastic_waves + electrostatics; overwrites electric_field_x / _y, two-way also
  charge_density, undeclared); `heat_em_phase_change` (heat + an EM type with phase_change_medium; overwrites the eps
  slots); `solid_membrane_tension` (solid_mechanics + membrane; overwrites coupled_tension_xx / _yy / _xy).
- 'sum' contributors to heat volumetric_power: em_heat_joule, currents_heat_joule, fluid_heat_convection,
  magnetic_heat_induction.
- A coupling type that reads q_accum is listed in `physics/em/definition_common._Q_ACCUM_COUPLINGS` (memory estimate,
  Bloch / em3d refusals). A q_accum reader sharing a source slot with an after-source contributor raises at build.

## Couplings across meshes (S7.2, mesh plan P2)
- `CouplingType.mesh_transfer` (None = one mesh; validation error across meshes): 'field' = the transform runs on the
  target mesh and sees the source as a `TransferredPhysics` (coupling_term.py): `state()[name]` interpolated onto the
  target cells on first access per source state revision (one view per physics shared by every term of the coupling,
  builder `shared_views`: an index coupling into 2-3 slots interpolates once; any field it reads; arrays of ndim >=
  `source_ndim`: 2, or 1 from a
  triangle mesh, whose per-element / P1 / P2 arrays `mesh_transfer.TriangleTransfer` tells apart by length; a triangle
  target gets per-element values), `domain` = the target's, other attributes the solver's own (T0, time). 'value' = it
  runs on the source mesh (create gets the source mesh's context: buffers are source-shaped), the target seen without
  state, the result moved by `value_kind` ('conservative': totals kept; 'interpolate'). `partner_materials`: the
  transform reads the partner's material arrays through the view's domain (thermo-optic: heat dndT / alpha on the EM
  mesh; em_solid_optical_force: youngs_modulus), so the build paints that type's arrays on the transform mesh too. A new
  transform reading more of its partner than state / scalars keeps to this or sets mesh_transfer None.
- Classes: 'field' heat_em_thermo_optic, field_transfer, heat_em_phase_change, semiconductor_em_plasma_dispersion;
  'value' conservative em_heat_joule, currents_heat_joule, magnetic_heat_induction, em_semiconductor_generation,
  em_acoustics_photoacoustic, em_solid_optical_force, magnetic_solid_lorentz_force; 'value' interpolate
  heat_solid_thermal_strain (reads heat.temperature_rise() on the heat window); None electrostatics_em_pockels,
  acoustics_em_acousto_optic, electrostatics_acoustics_piezo, fluid_heat_convection, solid_em_strain_optic (reads the
  solid's fields object and n0 on its own grid), solid_membrane_tension.
- `CouplingTerm.set_mesh_transfers(source_transfer= | value_transfer=)` (builder, every term); `transfer_value`
  returns target-shaped arrays (a 'value' result in the persistent `_moved` buffer), so buffers, SlotSum, relaxation
  and `write` are unchanged; the change check stays on the real solvers. `DynamicMaterialUpdate(notify=)` limits
  materials_changed to its MaterialGrid's mesh (None: all). Transfers: `core.grid_transfer`, cached per (source mesh,
  target mesh, kind) in the build. Self-tests `selftests_mesh_transfer.py` (their checks compute totals themselves;
  two_mesh_currents_heat's 'two' run, heat on an identical second mesh, must reproduce the shared run bit for bit).

## Additive source slots (E13, `slot_sum.py`)
- Contributor API (no builder or validate change needed): (1) the transform returns a full-grid (nx, ny), 3D (nx, ny,
  nz) or per-element (n_elements,) array (float32 or castable; None = keep the last value) of 'sum' = the slot's own
  quantity or 'index' = dn of the slot's permittivity component; (2) `create` returns `CouplingTerm(...,
  contribution='sum' | 'index')` or a `CouplingGroup` of such terms (one per slot, all contributors) with
  `after_source_step(ctx, spec, every)`; an index coupling returns `index_contributor(ctx, spec, every, source_field,
  transform_for_slot)`; (3) the CouplingType sets `writes=` (`writes_index_slots`, `writes_heat_source`, or its own
  `{slot: kind}` matching the terms); `index_slots(physics_type)` / `target_type(vctx, spec)` give the slots. Index
  helpers live in `index_terms.py` (<- coupling_term, physics.em.index_reference only: physics/mechanics imports it),
  also `physics_and_type(vctx, name)` ((None, None) when unknown), the one endpoint lookup of validate callbacks.
- `MultiphysicsRunner.add_coupling` routes each contributor (a group's terms with the group's every /
  after_source_step) into the one `SlotSum` of its (target, slot), appended to `runner.couplings` at the first
  contributor's position (contributors never sit there themselves). SlotSum + overwriting coupling on one slot, or
  another kind / every, raise (validation reports both first). SlotSum: shared `every`, `after_source_step` = any
  contributor's (the whole slot then transfers on phase 1), `source_fields` = union, `name` 'a + b', `transfer_value` /
  `write` (loop-back relaxation on the combined value), no `terms`. Mixed phases with a contributor reading
  `slot_sum.LOSS_FIELDS` (em_heat_joule beside magnetic_heat_induction at equal stride) raise in `SlotSum.add` (phase 1
  would shift the loss window one tick); remedy: the after-source source's every != the coupling's. model.validate
  does not see this yet (error at build).
- Each transfer refreshes the contributors (`apply`: only those whose source changed; `flush`: flush_transform) into
  their persistent `buffer`, then injects 'sum' = sum of buffers (a lone buffer directly: ez_source then holds the
  contributor's buffer) or 'index' = (n0 + sum dn)^2 (`index_reference.index_shift_to_permittivity`). A contributor
  that never produced data is left out; no data at all injects nothing.
- Reference index (`physics/em/index_reference.py`: eps_r / eps_xx / eps_yy -> n0_zz / _xx / _yy, INDEX_SLOTS =
  `<array>_update`): an 'index' SlotSum records n0 right before its first transfer (it is that array's only writer, so
  this is the unperturbed eps); heat also records all components at prepare / finalize. Transforms read n0 via
  `reference_index(materials, slot)` (raises when unrecorded), never compute their own.
- Gauss-Seidel (stationary): a SlotSum applies after the last of its sources steps (two heat sources both fresh);
  Picard relaxation acts on the combined value.
- Electro-thermal loop (sigma_dc(T)): keep currents every 1 (re-solves only when dirty) so q_joule before heat step
  k+1 uses sigma_dc(T_k); currents every = heat every gives T_{k+1} = F(T_{k-1}).
- field_transfer keeps one persistent full-grid buffer per coupling (the target holds it); x, y are clamped centres.
- Self-tests: `coupling/selftests.py` (em_heat_thermo_optic_strip and currents_heat_thermo_optic_strip run only inside
  coupling_cpu_gpu; examples/multiphysics analytic_lossy_slab_heating / analytic_thermo_optic_phase_shifter check the
  same chains; conduction checks are exact 3-point FV identities, 1e-4); fdtd_heat_joule_energy holds only while the
  pulse has died before the never-stepped last windows. `selftests_pockels.py` reuses the strip helpers (plates on the
  y faces give an exactly uniform E_y; template ln_phase_modulator: terminals + pec_domain). `selftests_slot_sum.py`
  (heat_sources_superposition_strip). `selftests_index_sum.py` runs only inside pockels_cpu_gpu
  (pockels_phase_index_sum; the Pockels phase itself is examples/multiphysics/analytic_pockels_phase_vpi).
