# open_sol/physics/nlfd — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first; FDFD gotchas: `open_sol/physics/em/sub_structure_conventions.md`).
Type `em_fdfd_nonlinear` (family 'em', default instance 'nl', studies nonlinear_frequency_domain and optimization).
The physics (channels, mixing table, C / C* collocation, sheets, iteration, adjoint, absorption) is in the module
headers and help texts; this file keeps the cross-file rules.

## Coupling to em_fdfd
- Channel features wrap the em_fdfd_ez / _hz FeatureTypes (definition.py header), so a change there reaches every
  channel. `_merged_help` prints the clauses both forms share once and each differing run as 'E_z channels: ...
  H_z channels: ...': keep the em_fdfd_ez / _hz texts identical sentence for sentence where they can be.
- One 'linear' runtime serves every channel FDFDPhysics (each keeps its own factor), so its record sums the channels'
  factorisations. Adjoint solves are `FDFDPhysics.solve_adjoint` (cuDSS LU and MUMPS / PARDISO transposes: em sub
  file "FDFD").

## Channels: Bloch k and mirror parity
- Each channel solves on `solver.channel_boundary_config` (its Bloch k; an even channel turns antisymmetric faces
  symmetric); the shared `boundary_config` has k = 0 and the inputs' parity, and the settings omit the FDFD bloch_kx /
  _ky. A term landing by frequency only must match k and parity: `mixing.inexact_term_issues`, raised by the solver
  (`_build_mixing`) and reported by validation (`_channel_phase_issues`) alike. The renderer's FieldInfo carries the
  selected channel's `plane_signs` where they differ from the shared faces, so even channels unfold with their own
  parity.
- Bloch wrap: C of a channel's field takes its own wrap factor, C* of a target density the target's (sources.py
  header); sheet sites take none, so sheets refuse k != 0 (`_check_sheet_bloch`).

## Adding a nonlinear term
A new term (bulk, sheet or carrier-like) must:
- collocate / distribute through C / C* (sources.py) or its own exact adjoint pair, never `yee_grid.face_average` (it
  differs at non-periodic ends): only then sum_k W_k = 0 for real Kleinman tensors (nlfd_collocation_transpose,
  scalar power_exchange_imbalance);
- read `ChannelFieldStore`, never a channel's FDFDPhysics (sources.py header);
- add its exact tangent to `NewtonLinearization` (else Newton converges linearly, or not);
- add its transpose to adjoint.py (`_channel_backward`; pump_depleted `depleted_cotangents`);
- when lossy, add its density to absorption.py (`absorbed_power_density`: method independent);
- add its buffers to `_memory_estimate` (run items vs phases: its comments);
- weigh by the cell areas through `solver.cell_area` (`sources.CellAreas`: separable widths, `integral` / `divided`),
  never a full-grid area array (8 B per cell).

The time recipe's channel assemblies are `definition_fdfd.fdfd_assembly_ops` (the mean channel's gpu transfer): a
change of the FDFD assembly's ops reaches nlfd through it.

Sheets: a `SheetMedium` lives in `solver.sheets`, never `solver.media` (poling_map, Miller, carriers and Kerr boxes
iterate media). Its table entry is the host representative tensor (validation and build compute it alike); the values
come from `site_tensor` (`MixingTerm.components` of a sheet term only filter / describe). Kerr operator terms and
carriers read a phasor's `bulk_terms` only; absorption and the adjoint read the sheet terms too.

## Krylov precision
- The Newton steps run `iteration.real_gmres(single_basis=True)` (complex64 basis, forcing raised to
  GMRES_SINGLE_MIN_TOLERANCE); the pump-depleted adjoint keeps complex128 (it solves to the iteration tolerance). A new
  real_gmres caller below that floor passes single_basis=False (ValueError otherwise). Float32 rows meet only float32
  operands (a float64 operand makes matmul copy the basis to float64); the estimator (`definition._memory_estimate`
  Newton items, the adjoint's excess over them) and the time recipe (`krylov`) count the complex64 rows. Checked by
  (d) of nlfd_newton_tangent and the newton model cases.
- Picard's Anderson history (dX, dF) is complex64 (`iteration.ANDERSON_HISTORY_DTYPE`, error argument there); x, f,
  the store and the residual stay complex128, so no tolerance floor follows. The history meets only float32 operands
  (gamma cast first) and its Gram is summed in float64 from chunk copies (`ANDERSON_GRAM_CHUNK`, own rcond): a change
  of the mixer's buffers changes the 'Anderson mixing' item and recipe of `definition` (checked by (e) of
  nlfd_newton_tangent).

## Self-tests
- selftests_sheets / selftests_backend import private spec builders and checks of selftests / selftests_sheets /
  selftests_adjoint: changing one changes those cases too. What each reference relies on (strip layout, tolerances,
  fictitious coefficients) is in the comment above the case.

## cuDSS / host direct (verdict)
- DIRECT_LINEAR_ROLE has no `auto_back_ends` by design: 'auto' is scipy_splu on the gpu. Right for the strip examples
  (~1e4 unknowns per channel: gstrf ~2 ms, a SuperLU solve ~0.2 ms, below cuDSS's per-solve host overhead and its
  ~15 ms plan per channel build); a device factor pays from ~1e5-1e6 cells, where SuperLU solves cost 10-100 ms and a
  Newton run takes 50-200 of them. cuDSS (cudss / cudss_mixed: only below the 1e-3 single_residual gate) is an opt-in
  candidate, strongest for pump_depleted newton and the depleted adjoint; picard Kerr / carrier sweeps refactorise it
  (no GMRES recycling; validation info, time recipe). MUMPS / PARDISO behave as SuperLU (em sub file "MUMPS /
  PARDISO").
- Unmeasured: factor_seconds vs SuperLU, device_bytes, single_residual, solves per nonlinear solve, wall time vs
  scipy_splu on a >= 1e5-cell grid; no nlfd cuDSS self-test.

## Triangles (decision; nothing implemented)
- Gate: channels would inherit the em_fdfd_ez / _hz triangle path, which first needs a driven dof-vector solve,
  `_solve(record_monitors=False)`, load-vector sources, per-element extra permittivity, `solve_adjoint` and
  per-element permittivity gradients; ~2-3k lines in nlfd after that.
- Yee-bound pieces to replace: ChannelFieldStore layout, C / C*, media boxes, KerrOperatorTerm, sheets, FreeCarriers,
  absorption, adjoint face weights / wall rows, renderer, undepleted guard, estimates.
- Mixing on triangles: P at quadrature points from E = Phi u, b = Phi^T W P (exact transposes: the work identity holds
  by construction); rule degree (n + 1) p. hz: prefer an E_t Nedelec formulation. Kerr / carriers newton-only first.
  Sheets on interface edges (exact normals). No periodic triangle meshes, so the strip references do not carry over.
