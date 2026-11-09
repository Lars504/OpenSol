# open_sol/physics/magnetic — conventions & gotchas

Sub file of the root `structure_conventions.md`. DiffusionSystem (assembly, PCG, held cells) is in the core sub file.

## Layout and registration
- Module contents: `structure_index.md`. Non-obvious registrations: `selftests_ac_fem.py` by definition_ac;
  `selftests_fem.py` (+ iron_c_core_saturation) by definition_magnetostatics; `selftests_3d.py` / `selftests_fem3d.py`
  by definition3d; `selftests_coupled.py` (+ induction_billet, coupling-name literals) by
  coupling/definitions_magnetic, not by definition.py. `selftests_ac.py` has no cases of its own (its 1D models run in
  selftests_backend / selftests_coupled). Type names are literals (`MAGNETOSTATICS_TYPE_NAME`, `MAGNETIC_AC_TYPE_NAME`,
  `MAGNETOSTATICS_3D_TYPE_NAME`), repeated in the definitions. The curl assembly lives in `fem/magnetostatics.py`.
- `definition.py` registers magnetostatics, magnetic_fields_ac, then magnetostatics_3d. The AC type reuses the
  magnetostatics mu_dc Param objects and FeatureTypes magnetic_insulation / symmetry / external_field /
  fixed_potential (`validate_magnetic_common` serves both; `_validate_external_field` reads the physics' own type
  settings) and shares `sigma_dc` / _xx / _yy and their arrays with electric_currents. Both 2D types: default instance
  'magnetic', family ''. magnetostatics_3d: own mu_dc Params (mu_dc_xx / _yy / _zz, remanence_z), shares the 2D
  saturation_polarization / bh_curve and nonlinear settings objects (isotropic only); no current_source, no couplings.
- `__init__.py` resolves names lazily (module `__getattr__`); the definition modules import only materials /
  renderer / selftest template names / circuit and fem estimate constants at module level (solvers inside `create`,
  cases inside each `_selftest_cases`, BCSide inside apply callbacks), no cupy.

## MagneticPhysics
- Backends gpu and cpu: every array op through `self.xp` (set in prepare, MagneticACPhysics too), fields with `xp=`;
  the AC host build reads by `to_host`, copying masks / held cells with `np.array` (to_host returns the live numpy
  array on the cpu). couplings.py is backend-neutral, so both couplings run on the cpu. `selftests_backend.py`:
  magnet_cpu_gpu (holds the only tabulated-curve tensor case), magnet_ac_cpu_gpu (registered by definition_ac).
- Unknown: planar A_z [T m]; axisymmetric psi = r A_phi [T m^2] on the PLANAR DiffusionSystem (`axisymmetric=False`)
  with c = nu / r_c. Never the ring-weighted operator (its 2 pi r face weights cancel 1 / r; psi = B r^2 / 2 is then
  no solution). `MagneticPhysics.axisymmetric` is the physics' flag (Session ring weights, totals x 2 pi); so the
  inherited `add_boundary_flux` / `perimeter_flux_weights` get planar weights (no feature uses them; a perimeter
  surface-current feature would need the 1 / r of the psi form).
- Coefficient swap: c_xx = nu_yy / r, c_yy = nu_xx / r (B_y = -dA/dx sees mu_yy); nu = 1 / (mu0 mu_dc), mu_dc > 0
  checked at every assembly (one sync). Axisymmetric arrays: x = r, y = z.
- `flux_at_centres` is -c grad u: planar (H_y, -H_x), axisymmetric (-H_z, H_r). H = physical face flux (operator +
  remanence face terms) averaged to centres, B = H / nu + B_r (exact at magnet edges); B, H = 0 in held cells.
- Faces: every face starts Dirichlet A = 0 (insulation); `symmetry` is Neumann 0. On a model mirror plane the fields
  unfold with the face's own sign (`core.mirror.FACE_FEATURE_SIGNS`: symmetry A even, insulation A odd; A / J polar_z,
  B / H axial); magnetic faces never make a mirror plane themselves. Axisymmetric x_lo is the axis: psi = 0,
  `set_face_bc('x_lo', ...)` raises, validation (`axisymmetric_issues`, 'error') rejects features selecting it.
- Axis rule: x_lo = `BCSide('dirichlet', value=0.0, conductance_factor=2.0)` (half cell's series resistance puts c at
  r_c / 2), set in `__init__`; the core face terms carry it into assembly, flux_at_centres, field_energy, the remanence
  face drive and face_inward_flux, and ac_operator's `boundary_terms` reads it too. Never add an axis term elsewhere
  (extra_diag stays unused). The remanence axis face drive is halved (`_remanence_face_drives`: int_0^r_c r B dr = B
  r_c^2 / 2), so uniform B_rz keeps psi = B r^2 / 2 discrete-exact.
- Graded r (open): every psi-form half cell uses resistance (dr / 2) r_c / nu, the exact ring integral is (dr / 2)(r_c
  +- dr / 4) / nu; the +-dr / 4 cancel only for equal neighbour dr, so uniform B_z, the remanence face drive and the
  external-ring x_hi value are exact on UNIFORM r only (face flux error ~(dr_L - dr_R) / (8 r), worst near the axis;
  ac_operator likewise). flux_concentrator_needle (+ _ac) grades r; every axisymmetric self-test uses uniform r. Fix:
  per-face half-cell resistances (conductance hook or a 1 / r WindowMetric in core.fv_diffusion), mirrored in
  ac_operator and face_drive_terms, plus graded-r self-test variants.
- External field (`set_external_field`): the outermost window cells are held at the external potential at their faces
  toward the free cells (planar A = B_x y - B_y x at (x_face, y_c) / (x_c, y_face); corners touch no free cell;
  axisymmetric psi = B r_c^2 / 2 on the z rows, (B / 2) r_c (2 r_face - r_c) on x_hi, B_x must be 0). Centre values
  would give B (1 + dx / L). Coordinates from the window's lo edge.
- Remanence: optional arrays remanence_x / _y (allocated only when a material sets them). Drive v with physical flux -c
  (grad u - v): planar (-B_ry, B_rx), axisymmetric (r B_rz, 0) (radial remanence raises). `face_drive_terms` (held
  halves 0, outer faces by their face-term conductance, Neumann none); the rhs = minus its net outflow, added after
  `build_rhs` to the free rows of `rhs` and to `rhs_full`; recomputed after each assembly / remanence change.
- Coils (stranded): J = turns current / area; linkage = turns / area * sum u * cell area (x 2 pi axisymmetric);
  series coils +I / -I: per-coil inductances add to 2 W / I^2. `field_energy` = (1/2) u^T K u over faces touching a
  free cell (held-free faces count the free half), the discrete identity W = I Lambda / 2; w_mag (centre B) integrates
  to it only approximately at interfaces.
- `step` re-solves when dirty (operator: mu_dc, faces, held cells, J_s, add_bh_curve; rhs: sources, coils,
  inject_source, remanence).
- Saturation (`saturation.py`): J_s array, Frohlich-Kennelly B = mu0 H + J_s a H / (1 + a H), a = mu0 (mu_dc - 1) /
  J_s (mu_dc = initial permeability), secant nu(|B|), per axis with the same centre |B|. Fixed point (`self.picard`,
  core.damped_iteration) on ln nu (float64 (2, k) `_log_reluctivity_state` on the k saturating cells only, flat
  window indices `_nonlinear_flat`, J_s and the curve masks compacted alike; `_coefficients` scatters it into the linear
  1 / (mu0 mu_dc); the 3D tensor ln mu_r state likewise (k,): Anderson cannot make nu negative and never runs over
  linear cells),
  reset by prepare, add_bh_curve and `_refresh_saturation` when the saturating set changes or empties (warm-starts
  later loops; all three tensor / tetrahedron solvers follow this rule). Only a loop's first assembly reads / checks
  mu_dc and runs the anchor check (`_assemble(full=False)`). Fields / operator are those of the last iterate's solve,
  not the final proposal. nonlinear_method 'auto' = 'anderson' here ('newton' on triangles); MagneticPhysics refuses
  'newton'. Validation per mesh kind is definition_magnetostatics `_mesh_kind_issues`, sharing `_method_misfit` /
  `_anisotropic_saturation` with the GUI mesh form's `_mesh_method` text: a new mesh-kind rule goes into both.
  Undamped Picard diverges where the current imposes H past the knee; Anderson converges (example
  `examples/magnetic/bh_curve_table_core.json`). No Newton tangent: the exact Jacobian of the centre-|B| secant has a
  13-point stencil and the 9-point c_xy approximation is not positive definite under strong saturation. Saturation +
  remanence (or bh_curve + J_s / remanence) in one material: validation error (cells raise); magnetic_fields_ac
  ignores J_s (warning). Energy / inductance are secant values.
- Tabulated curve `bh_curve` (schema kind 'table', H [A/m] / B [T] increasing, no array): `_create` calls
  `add_bh_curve(region.mask, rows)` per `ctx.material_regions` entry (2D masks live until release_caches; scripts call
  it directly, `Simulation.add_material` refuses tables). `saturation.bh_curve` prepends the origin (any other H or B =
  0 rejected); slope mu0 beyond the last row; isotropic, mu_dc ignored (warning). Table proposals are H-based (centre
  |H| = nu |B|), J_s ones B-based: with s = d ln B / d ln H and w the cell's share of its circuit reluctance, the Picard
  factor is -(1/s - 1) w B-based but (1 - s)(1 - w) H-based. A B-based table loop diverged on the mu0 extension; deep
  J_s saturation under a current drive has the same limit (Anderson covers s down to ~0.12).

## magnetostatics_3d (MagneticPhysics3D, magnetostatics_3d_plan)
- Separate 3D-only type (tensor, gpu + cpu). Unknown u = mu0 phi_m [T m] on the 7-point DiffusionSystem with c = mu_dc
  per axis (NO xx / yy swap: that belongs to the 2D curl form); the operator face flux c (w - grad u) is B. Currents
  only through `coil` (no current_source / source slot: a free J has no T without a solve).
- Face kinds swapped against 2D: every face starts insulated (Neumann B.n = 0), `symmetry` is Dirichlet u = 0;
  FACE_FEATURE_SIGNS opposite (insulation +1 = phi_m even), so B / H declare `polar_*` parities, J `axial_*`.
- Anchor: with no Dirichlet face and no held cell, `_update_corner_anchor` (every full assembly) holds cell (0, 0, 0)
  at u = 0 (it reads B = 0). `add_ideal_iron` / `set_external_field` release it BEFORE holding cells; a later
  Dirichlet face releases it at the next assembly. A new holding method must release it first, else the next assembly
  may clear a user-held corner cell.
- `ideal_iron` (mu -> inf, H = 0) holds u = mu0 x potential [A]. External ring: all six outer layers at u = -B . r at
  each layer's face toward the free cells (x, then y, then z layers: shared edges take the last value); exact for a
  uniform field on graded cells.
- One drive w = B_r / mu_dc + mu0 I T per coil through `core.fv_diffusion_3d.face_drive_terms_3d` (recomputed after
  every assembly and coil change); rhs `face_drive_divergence_3d` added after build_rhs to free rows and rhs_full. B =
  flux_at_centres + `face_terms_at_centres_3d`; H = (B - B_r) / (mu0 mu_dc). Never compute B from T - grad u
  (decision 7: cancellation in a coil's iron core).
- Coils (`coils3d.coil_drive_per_ampere`, host): per slice normal to the axis, hole = binary_fill_holes(winding) minus
  winding (winding on the slice edge or without a hole raises); s = d_out / (d_in + d_out); T per ampere = turns / h *
  s; J = curl T from centre gradients, once per add_coil; MagneticFields3D allocates J_x / _y / _z with the first coil.
  Current > 0 = counter-clockwise seen from +axis.
- Energy = `face_flow_energy_3d` / (2 mu0) and linkage = sum over axis-normal faces of B_n A x
  `face_drive_integrals_3d` of T per ampere (both from `_total_face_flows`): W = Lambda I / 2 holds to the CG residual
  only when every held cell is at 0 (corner anchor); a ring, ideal iron at a potential or a symmetry face add a
  boundary term. Keep both on the same face sets if either changes. build_rhs gets a zero window; `add_boundary_flux`
  raises.
- Saturation: `_create` passes `ctx.region_mask(region)` (3D regions carry no mask) to `add_bh_curve`; the 2D tensor
  loop ('newton' refused) on ONE isotropic float64 state ln mu_r of the saturating cells (`_log_permeability_state`),
  c = exp(state) on all three axes there. Both curves propose H-based (|H| = |B| / (mu0 mu_r) from face fluxes; 2D
  runs J_s B-based). J_s cells need equal mu_dc_xx / _yy / _zz. mu_eff (allocated by the first refresh that finds a
  saturating cell) = mean assembled c; else the renderer shows mean mu_dc.
- Self-tests (`selftests_3d.py`): the mu_r 1000 sphere uses first-order Richardson (2 fine - coarse, 1 %): a
  staircased high-mu body has excess polarizability first order in h (+6 % / +3 % core field at a / 10 / a / 20),
  never within 2 % at affordable resolutions. The Ampere checks telescope only on uniform cells across the winding.
  Paper-matched run: `experiments/Ben_flux/needle_3d/`.

## magnetostatics_3d on tetrahedra (MagneticFEMPhysics3D, solver_fem3d.py)
- Create dispatches on `ctx.domain.grid_kind` (`discretization` ignored on tensor meshes). FEMDiffusionPhysicsBase3D:
  u = mu0 phi_m by Lagrange P1 / P2, C = diag(mu_dc) per element; Automatic 'linear' on tetrahedra = sparse_direct (pcg
  does not declare tetrahedra): 'auto' = double cuDSS Cholesky on the gpu with nvmath-python (refreshed on its plan per
  saturation iteration), else host SuperLU.
- Drive: K u = int grad v . (mu0 mu T + B_r) dV (`solve(extra_load=)`), B = mu (mu0 T - grad u) + B_r; never B from T
  - grad phi_m alone. Insulation natural (incl. the drive), symmetry u = 0. The drive load depends on mu (coils):
  rebuilt per solve.
- Coil T: Nedelec-1 dofs = line integrals of turns / h s e_axis per edge (`coils3d.coil_edge_drive_per_ampere`), s
  from `_slice_ramp` on a lattice over the winding's bounding box, each edge clipped to the winding's axial extent.
  Circulation around any mesh-edge loop = turns x enclosed current whatever the alignment (nodal P1 T would smear the
  end planes); J constant per element; several coils add. The winding must be one piece along its axis.
- Constraints: ideal iron = `add_held` at mu0 Omega (excluded from energy / linkage); external field = `add_fixed_dofs`
  u = -B . r on every window-face dof (no held ring), re-set at each full assembly (wins over faces / held); corner
  anchor = window node nearest the lower corner, an `add_fixed_dofs` handle refreshed per full assembly (empty when
  something else anchors). No release ordering needed (unlike the tensor anchor).
- Post-processing at RULE_DEGREE 2 (exact for affine elements, P0 mu), in POST_CHUNK blocks; with exact quadrature
  in load, stiffness and integrals W = Lambda I / 2 holds to round-off (anchor only). Display: phi_m nodal, the rest P0
  (sampled B jumps at element faces). Saturation: compact state ln mu_r over saturating elements, |H| = B_rms / (mu0
  mu_r), refactor per iteration; the total-T rule values are kept per element chunk (`_coil_field`, keyed by the coil
  currents, cleared per full assembly, at most `COIL_FIELD_CACHE_BYTES`). w_mag = B^2 / (2 mu0 mu) (the energy
  scalar's density, as 2D and the tensor 3D solver; not B . H / 2 inside magnets).
- Coil results carry 'energy' (both mesh kinds); the identity self-test reads it there since field derived values
  raise on tetrahedra. Not done: mirror-plane unfolding (unchecked), nodal B display, Lorentz force.
- cuDSS (unmeasured): operator real SPD. Linear models: one factor, few solves; host SuperLU fine up to ~1e5 P2 dofs,
  above cuDSS pays (3D fill). Saturation refactors per iteration (15-50): plan reuse (`CudssFactorization.refactorize`)
  is the win; cudss_mixed is risky at mu contrasts 1e3-1e5 (measure single_residual on an iron-core model first). The
  opt-in MUMPS / PARDISO back ends of the shared diffusion role refactorise on their kept analysis the same way
  (`previous=` through DiffusionDirectSolver); their mixed variants carry the same float32 risk (unmeasured).
- Memory: magnetostatics / magnetostatics_3d on every mesh kind add a cuDSS 'linear' factor through
  `fem.diffusion.cudss_diffusion_estimate` (tensor: on the kept A_solve; FEM: via `fem_diffusion_estimate[_3d](cudss=)`),
  `refactorizes` when a material saturates (transient 'refactorisation' per iteration / Newton step). 3D tensor field
  update and energy / linkage temporaries are transient phases. magnet3d_*_sphere coarse (149k cells, pcg multigrid):
  the estimated build peak is ~1.5x the measured one, from `solvers.multigrid.setup_extra_bytes` (products n x 7^2 x
  32 B, ~2x what that setup takes); the fine variant (593k, products capped) is ~10 % low. Both figures predate
  `setup_extra_bytes` taking the largest setup step instead of the sum of three (lower on both: re-measure).

## Triangle meshes (MagneticFEMPhysics, magnetic plan section 10)
- `solver_fem.TriangleMagneticBase` holds what MagneticFEMPhysics and MagneticACFEMPhysics share (faces, node probes,
  element masks, the current_density slot, ring, the Kelly loop `_kelly_indicator` over each class's
  `_edge_tangential_field`); `ElementFields(names, ...)` their per-element arrays. AC coil checks / R_dc of both AC
  meshes: `circuit.coil_drive` / `stranded_resistance`.
- Create dispatches on `ctx.domain.grid_kind` (magnetic_fields_ac too). Unknown A itself (planar A_z, axisymmetric
  A_phi; NOT psi: P1 cannot hold psi = B r^2 / 2 and int (nu / r) |grad psi|^2 is singular on axis elements); field /
  probe A mean the same on both mesh kinds. B-space form (`fem.magnetostatics`): B = Q a at quadrature points,
  axisymmetric (-dA/dz, dA/dr + A / r) with per-radian r weights; H = diag(nu_xx, nu_yy) B, NO xx / yy swap.
- One quadrature for W, R and K_t (cached per system in `blocks`): a rule change must keep all three on it, else
  Newton loses its quadratic rate and the line search its consistency (the tangent self-test catches it).
- Material law (`MagneticFEMSystem.constitutive`): linear per axis or a saturating curve per element (`curve_index`,
  -1 linear): `saturation_curve_terms` (J_s needs isotropic mu_dc: validation + run error), `tabulated_curve_terms`
  (B-indexed, w from `BHCurve.node_energies`). D = nu I + (nu_d - nu) e e^T: SPD for any increasing curve. Remanence
  (linear elements only, radial allowed axisymmetric) is a LOAD, not part of the law; energy scalar = int w dV (true
  int H dB when saturating, not the secant value).
- Constraints (`_constrain`, later wins): Dirichlet faces, symmetry natural, external field A_ext on every
  window-edge dof (no held ring: the whole window is free), held elements (axisymmetric A_phi = psi / r per dof, psi !=
  0 on the axis refused), axis x_lo A = 0. Planar anchor check `floating_region_count` on the reduced operator;
  axisymmetric skips it (A / r anchors).
- Solve: linear = one factor per material / constraint change via the 'linear' role runtime. Saturating = Newton on W
  (`_newton`) from the B = 0 limit solve, Armijo backtracking to DampingSettings.min_relaxation (no decrease: warning,
  stop); anderson / picard refused, nonlinear_relaxation unused. Unconverged loops append to runtime_warnings.
- State per element (D12); B / H / w_mag 0 in held elements and energy skips them (`active_elements`: an axisymmetric
  held psi / r interpolates to a spurious B), nodal display averages still include them. `fields` (`ElementFields`)
  carries the tensor names, so LorentzForceDensity reads it unchanged on one mesh. Display: A nodal, B / H nodal
  averages, rest P0; PML ring dofs clamped window values, ring elements carry no state. Derived values read the nodal
  display fields, so a material outline smears normal H / tangential B into the boundary elements (a mean H over an
  iron slab read ~2x): self-tests read such fields at interior points ('at'), never region means.
- Coils: linkage by `region_integrals`; probes A at a point's node; error_indicator Kelly tangential-H jumps.
  Estimates: time = DiffusionSystem recipe with (NEWTON_ITERATIONS_GUESS + 1) solves per saturating solve
  (uncalibrated; quadrature / line-search work not priced).
- Open: mirror-plane unfolding (not checked on triangles), a same-mesh Lorentz-force self-test, an axisymmetric
  remanence self-test (A_phi of an axial magnet shell has a 1 / r part, not in P2: needs a convergence tolerance), held
  regions in any triangle self-test.
- cuDSS (unmeasured): reduced K and K_t real SPD; Newton one factor + ONE solve per step with a fixed pattern per
  constraint set (`_to_csr` and `tangent[free][:, free]` keep explicit zeros). Without plan reuse cuDSS replans every
  step and loses to SuperLU; with it (`previous=`, keyed on the constraint revision: `operator_dirty` / `_constrain`
  resets it) it pays above ~2e5 dofs and in sweeps; Amdahl: host law / assembly / line search cap the gain at ~2x.
  Double only: mu contrasts 1e3-1e5 give cond ~ mu_r h^-2 past the float32 limit, so 'cudss', never 'cudss_mixed'.

## MagneticACPhysics (magnetic_fields_ac)
- A MagneticPhysics subclass: the real DiffusionSystem assembly still runs (mu_dc check, held cells, ring, anchor
  check; its linear solver is `_AssemblyOnlySolver`: no PCG workspace or device CSR product, and the anchor check runs
  at once on the gpu too, never deferred to a CG solve), then `_build_host_system` (ac_operator) assembles complex K
  with c = nu / r, nu = 1 / (mu0 (mu_dc - j mu_dc_loss)) (Im nu > 0 is loss under exp(+j omega t)), axis
  conductance_factor 2, + j omega sigma A / r; held rows / columns dropped (their values drive the free rhs). Complex
  face conductances are series means of complex c, so K(Re nu) + j K(Im nu) is NOT used (wrong at interfaces).
  `_operator_with` keeps the stencil without the sigma diagonal (`_kept_operator`, keyed on nu_xx / nu_yy / held /
  the BC values; booked as one more host CSR): a sigma-only rebuild (sigma_dc(heat_T)) copies it and re-adds only
  j omega sigma A / r (every diagonal entry is stored, so `setdiag` keeps the pattern).
- sigma_dc_xx must equal _yy (run error); stranded coil and held cells get sigma 0 for the eddy term.
- Solid conductors (circuit.py): J = sigma (v_k - j omega u) / r, v_k planar V/m, axisymmetric V / (2 pi). Current
  drive = one bordered row scaled by 1 / (j omega) (complex symmetric, SuperLU symmetric mode); voltage drive moves the
  term to the rhs. Both go through the 'linear' role runtime (`host_solver=`, sparse_direct host only; cusolver_qr is
  a validation error). `circuit.RecycledFactor` (shared with triangles) keys the factor on `_operator_revision`, runs
  GMRES with the old factor first after an operator change, recycles only a `recyclable` factor and calls a replaced
  factor's `release()`. ConductorRow.row_offset is 0 on tensor meshes (held cells have sigma_eddy 0).
- Post-processing is host float64, copied to float32 fields one window array at a time (the run item); energy /
  magnetic_loss are exact face sums (`face_power`); stranded V = R_dc I + j omega linkage. `n_solves` counts host
  solves (the DiffusionSystem's stays 0); warm start seeds J_coupled only.

## magnetic_fields_ac on triangles (MagneticACFEMPhysics, solver_ac_fem.py)
- Create dispatches on `ctx.domain.grid_kind`. A PhysicsSolver of its own (not a MagneticFEMPhysics subclass) keeping
  a `MagneticFEMSystem` only for the dof map, curl blocks and constraints; that system's real 'linear' wrapper is never
  factorised. Unknowns and constraints as magnetostatics on triangles.
- Operator on ONE rule (curl blocks): complex symmetric K (unconjugated) + j omega `block_mass` (sigma_eddy 0 in held /
  stranded). Solid conductor: g = int sigma phi / r^p dV, G = int sigma / r^(2p) dV (NOT sum g when axisymmetric), I = v
  G - j omega g . a. Load, mass, rows and loss integral on one rule make the circuit row and the Poynting identity exact
  to round-off.
- A solid conductor touching the axis raises (int sigma / r dA diverges; the tensor scheme hides it behind r_c);
  validation cannot see it. Planar anchor check on Re of the reduced operator (asked even where sigma would anchor).
- w_mag, q_magnetic and the power integral skip held regions (spurious psi / r B). Display A_re / _im / _abs nodal, B
  nodal averages, rest P0; probes |A| at the node.
- Couplings: induction heat / Lorentz force read per-element `fields` on the same mesh; across meshes the transforms'
  `magnetic.fields` reads are not covered (mesh_transfer 'value' passes the source field only).
- Estimates uncalibrated (_FEM_POINT_VALUES, _FEM_CSR_COPIES, FEM_FILL_FACTOR); assembly / post-processing / GMRES not
  estimated. Self-test tolerances (`selftests_ac_fem.py`) are first estimates.
- cuDSS: complex symmetric, not Hermitian, on both mesh kinds: LDL^T (`RecycledFactor`: `factorize_host(matrix,
  symmetric=True, previous=)`, default row_scale) through `MAGNETIC_AC_LINEAR_ROLE` ('auto' = cudss on triangles). A
  cuDSS factor is not recyclable, so an operator change refactorises on its plan. complex64 'mixed' is doubtful at mu
  contrasts 1e3-1e5 plus circuit rows, so the role offers the host direct back ends in double only
  (HOST_DIRECT_DOUBLE_BACK_ENDS): a MUMPS / PARDISO LDL^T is recyclable like SuperLU (an operator change: GMRES on the
  old factor), keeps the SuperLU size rule (`circuit.triangle_size_issue(back_end=)`, tensor `size_issue` of the host
  back end: texts name the library by `host_direct.factor_label`; validation and the solvers alike), and the
  cusolver_qr refusal text lists the role's back ends. Solve summaries name the factor's library (tensor
  `solver_ac.solve_summary`; triangles print the back end).
- Memory (`definition_ac._cudss_estimate`): the factor is made at the first solve, so its pool copies
  (`operator_copy_bytes`, complex128), rhs buffer and row scale are run items, the factor external
  (`factor_device_bytes`, value_bytes 16, dimensions 2); an operator following a state adds the device route's index
  maps (run item) and its refresh peak (transient 'refactorisation'); circuit border entries are left out. A MUMPS /
  PARDISO factor: `definition_ac._host_factor_items` (the SuperLU heuristic x `host_direct.superlu_bytes_scale`, LDL^T,
  + `solver_host_bytes`; uncalibrated).

## Couplings (`couplings.py`; types in coupling/definitions_magnetic.py)
- magnetic_heat_induction (AC -> heat): 'sum' of q_eddy + q_magnetic (one buffer) into volumetric_power,
  after_source_step. Stranded-coil resistive loss is not transferred (no field holds it).
- magnetic_solid_lorentz_force (either type -> solid_mechanics): group of two 'sum' terms (body_force_x / _y, source
  field B_abs) sharing one LorentzForceDensity refreshed per `n_solves`; planar f = (-J B_y, J B_x), DC J = J_coupled +
  J_applied, AC Re(J_total B*) / 2. Axisymmetric would be f = (J B_z, -J B_r) ((r, z, phi) is left-handed), but
  solid_mechanics is plane strain: validation errors and the transform raises. Adds with em_solid_optical_force.

## Self-tests (`selftests.py`)
- 1D layout 1 x 0.1 mm, 10 um cells, symmetry y faces, band on cell edges; the expectation is the exact DISCRETE
  solution on the run's cells (`_planar_line_solution`, `_axisymmetric_line_solution` with the doubled axis
  conductance), float64 CG 1e-12, tolerance 1e-6. magnet_permeable_slab has B_x normal to the slab faces (the plan's
  B_y orientation would not be uniform). magnet_saturating_sheet: 'picard', relaxation 0.44 = 2 / (2 + k_max), on a
  sparse_direct node (`_model(direct=True)`). No tensor flux-driven saturation case; Anderson deep saturation is the
  example `bh_curve_table_core.json`. The tabulated sheet (no origin row, H past the last row; exact B = forward
  np.interp at the centre H) runs only in magnet_cpu_gpu (`_tabulated_sheet_specs`, exact pcg node).
- Templates flux_concentrator_needle / _ac (same `needle_model`); paper-matched versions (Carey 2026 Eq. 2) are in
  `experiments/Ben_flux/`. `selftests_ac.py` reuses the DC 1D layout; its skin slab / solid strip run only inside
  magnet_ac_cpu_gpu (the skin slab also in magnet_induction_heating_strip); circuit-row / Poynting identities enter
  the error scaled by 1e-6 / 1e-10.
- `selftests_fem.py` (cpu, host SuperLU): magnet_fem_tangent_check (central differences of `weak_form` W / R against
  K_t, + Newton quadratic tail) is the regression test of the Newton stage; expectations are the CONTINUOUS closed
  forms (P2 holds piecewise-quadratic A exactly), unlike the FV discrete identities of selftests.py;
  _remanent_slab is the only remanence case on triangles. Float64 scalars / derived 1e-9, float32 probes 1e-6.
- `selftests_coupled.py`: magnet_induction_heating_strip, magnet_lorentz_force_sheet; both repeat on the cpu
  (variant 'cpu').
