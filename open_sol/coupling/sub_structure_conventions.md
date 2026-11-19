# open_sol/coupling — conventions & gotchas

Sub file of the root `structure_conventions.md`: runner, coupling protocol, E13 source slots, transfers across meshes.
Per-topic notes (transform physics, cadence, self-tests) live in the CouplingType `description` in
`coupling/definitions*.py`, the transform module's header and the partner physics' sub file.

## Runner and couplings
- Cadence: dt_sync = min `dt_limit()`; a tick steps the solvers with `tick % every == 0` over every * dt_sync, moves
  the clock (`runner.t`, `tick`: couplings see the start time of the next steps), then applies the couplings on the
  runner-owned global tick (one added mid-run stays in phase): ticks every-1, 2 every-1, ... (just before a target
  with that every steps: the window-averaged Joule terms), or with `after_source_step` ticks 0, every, ... (right after
  a source with that every steps, else its targets lag one more interval). Creates take it from
  `index_terms.after_source_step(ctx, spec, every)`, never a stride test of their own; scripts pass it
  (`standard.thermo_optic` defaults True, right for every None).
- Runner hand-off: with one solver at stride 1 and an empty coupling schedule (time updates and dynamic materials are
  couplings) a solver with `advance_ticks(n_ticks, t, dt_sync, stop_time) -> (ticks run, t)` takes the whole
  `advance` chunk. It must run exactly the tick loop (step(t, dt_sync) while fewer than n_ticks ran and t < stop_time,
  t += dt_sync per tick), so outputs and stop times land on the same ticks. Only elastic_waves has it (physics/acoustics
  sub file).
- `prepare`'s stride-cost `prepare_warnings` are warned only by `Simulation.run` (the model path validates strides).
  The runner never calls `solver.prepare` / `finalize` (`Simulation.finalize` does: builder, first `run`).
- `flush()` (end of every time-dependent run, also after a cancel) flushes the after_source_step couplings after the
  final `step(t, inf)` of the quasi-static physics: their quasi-static targets are left dirty, not re-solved.
- Coupling protocol: the attributes `add_coupling` checks (+ optional `after_source_step`, default False); a relaxed
  stationary loop-back coupling also needs `transfer_value` / `write` on it or its `terms` (model/study.py).
- Change check (`coupling_term.state_revisions`): `apply` skips a transfer while the revisions of the physics it reads
  are unchanged since the last transfer (`apply_now` / `flush` stay unconditional). Sound only because every state
  change marks the solver dirty: a new solver must re-solve and count after a slot injection or a `materials_changed`
  of an array it reads, else readers keep stale data. A transform reading more than `source_field` declares those
  `(physics, field)` pairs as `CouplingTerm(also_reads=...)` (they join source_fields, loop-back detection and the
  check); reading the target's own state also sets `CouplingType.reads_target`.
- Backends: no rule of their own (validation refuses the cpu through the partners' `PhysicsType.backends`); buffers
  and transforms follow the root "Arrays, coordinates" rule. A partner method a transform calls (acoustics
  `elastic_strain` / `bound_charge`, the EM loss protocol) is that physics' port. Parity cases: `selftests_backend.py`.
- Slots: writer rule in the root file ("Source slot writers"); a state-dependent permittivity property on a written
  eps slot is an EM validation error (`definition_common`). Whether a slot keeps the array by reference (EM
  `ez_source` / `hz_source`, elastic_waves `heating_power`: then the coupling's buffer stays live until the next
  transfer) or copies it, and when coefficients rebuild: the target solver's `inject_source` docstring.
- EM loss protocol: the comment at the top of `definitions.py`. Each `take_absorbed_power()` resets an FDTD window, so
  a reader's every equals its target's (prepare warning) and a time-stepped source has one reader
  (`CouplingType.reads_absorbed_power`: validation error; the flag also counts as a q_accum reference in
  `definition_common._q_accum_referenced`).
- Extension modules: each `coupling/definitions_<topic>.py` (discovered by `definitions.register`) returns early
  without its partners; order must not matter, so a check of another module's coupling type belongs in a template or
  selftest, not in register.
- Two-way piezo (`definitions_piezo.py`): one CouplingGroup charge term -> `_CoupledResolve` -> field terms, in that
  order (no lag); a 'sum' charge contributor would leave the group (SlotSum routing) and break the order. `writes()`
  declares only the E slots (charge_density is on the source physics), so `_two_way_issues` checks that writer itself.

## Registered coupling types
- Each `definitions*.py` `register` registers its types only when their partner types are registered (its docstring
  says when); `api_reference.md` lists them with endpoints, slots and cross-mesh class (`mesh_transfer=`).

## Couplings across meshes
- `CouplingType.mesh_transfer` / `value_kind` / `partner_materials` (comments in model/schema.py): 'field' transforms
  run on the target mesh and see the source as a `TransferredPhysics` (one view per physics shared by every term of a
  coupling: builder `shared_views`); 'value' transforms run on the source mesh (create gets that mesh's ctx: buffers
  are source-shaped) and `transfer_value` moves the result into the persistent target-shaped `_moved`. Buffers,
  SlotSum, relaxation and `write` are unchanged; the change check stays on the real solvers;
  `DynamicMaterialUpdate(notify=)` limits materials_changed to its mesh.

## Additive source slots (E13, `slot_sum.py`)
- Contributor API (no builder or validate change): (1) the transform returns the slot's full-grid (nx, ny), 3D (nx,
  ny, nz) or per-element (n_elements,) array (shape from `ctx.domain.materials.shape`; float32 or castable; None =
  keep the last value): 'sum' = the slot's own quantity, 'index' = dn of the slot's permittivity component; (2)
  `create` returns `CouplingTerm(..., contribution='sum' | 'index')` or a `CouplingGroup` of them (one per slot) with
  `after_source_step(ctx, spec, every)`; an index coupling returns `index_terms.index_contributor`; (3) the
  CouplingType sets `writes=` (`writes_index_slots`, `writes_index_and_off_diagonal_slots`, `writes_heat_source`, or
  its own `{slot: kind}` matching the terms). Validate callbacks look endpoints up with `index_terms.physics_and_type`.
- Mixed phases on one slot with a contributor reading `slot_sum.LOSS_FIELDS` (em_heat_joule beside
  magnetic_heat_induction at equal stride) raise in `SlotSum.add` at build; model.validate does not see it yet.
  Remedy: give the after-source contributor's source an every other than the coupling's.
- Index slots = `index_reference.INDEX_SLOTS` (`<array>_update` of eps_r / eps_xx / eps_yy): a family-'em' type takes
  every index coupling (thermo-optic, Pockels, plasma dispersion, strain-optic, acousto-optic) through them.
- Reference index n0 (`physics/em/index_reference.py`): transforms read it through `reference_index` /
  `in_plane_reference_indices` (raise when unrecorded), never compute their own: an 'index' SlotSum records it right
  before its first transfer (the array's only writer, so still unperturbed), heat records every component at prepare /
  finalize.
