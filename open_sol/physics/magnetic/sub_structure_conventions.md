# open_sol/physics/magnetic — conventions & gotchas

Sub file of the root `structure_conventions.md`. Module contents: `structure_index.md`; the solver classes' docstrings
hold the discretisations (unknowns, face rules, drives, coils, energy / linkage). DiffusionSystem (assembly, held
cells, anchors) and the damped iteration are in the core sub file, FEM assembly in the fem sub file.

## Layout and registration
- `definition.py` registers magnetostatics, magnetic_fields_ac (reusing the magnetostatics mu_dc Params, face /
  held-region FeatureTypes and `validate_magnetic_common`; `sigma_dc` Params and arrays shared with electric_currents),
  then magnetostatics_3d (own per-axis mu_dc Params, the 2D saturation / bh_curve / nonlinear settings objects).
  `selftests_coupled.py` is registered by coupling/definitions_magnetic; `selftests_ac.py` has no cases of its own.
- Every solver of both 2D types and the 3D type dispatches on `ctx.domain.grid_kind` in its definition's create:
  tensor (MagneticPhysics / MagneticACPhysics / MagneticPhysics3D), triangles (MagneticFEMPhysics /
  MagneticACFEMPhysics), tetrahedra (MagneticFEMPhysics3D).

## Shared rules (every magnetic solver)
- Snapshots: `solution_state` copies what the renderer reads (tensor `_solution_fields`; FEM element fields, `values`,
  nodal display arrays; 3D FEM the assembled mu while saturating) and, with materials, the rendered mu_dc (/ sigma);
  the saturation / Newton state stays live. Arrays a solve replaces are restored by replacement, never in place.
- Saturation loops (tensor 2D / 3D, tetrahedra): the compact float64 log state over the saturating cells / elements
  warm-starts the next loop; prepare, add_bh_curve and a changed or empty saturating set reset it. Loop iterations
  compute only |B| of the saturating items, the fields (and energy / linkage) once after the loop: the shown fields
  are the last iterate's solve, not the final proposal. Anderson mixes ln nu / ln mu (never nu: it cannot go
  negative). Saturation + remanence in one material: validation error (`refuse_curve_conflicts` at run time);
  magnetic_fields_ac ignores J_s (warning).
- nonlinear_method 'auto' = 'newton' on triangles, 'anderson' elsewhere (tensor, tetrahedra); each solver refuses the
  other kind's methods.
  Validation (`_mesh_kind_issues`) and the GUI mesh text (`_mesh_method`) share `_method_misfit` /
  `_anisotropic_saturation`: a new mesh-kind rule goes into both.
- Memory: magnetostatics / magnetostatics_3d add a cuDSS 'linear' factor through
  `fem.diffusion.cudss_diffusion_estimate` (refactorises when a material saturates); a saturating tensor model adds
  `kept_preconditioner_bytes`. The time recipe still prices one setup / factorisation per saturation iteration (an over-estimate since the kept preconditioner).
  MagneticPhysics keeps no window nu / mu copy: while its saturation loop runs only the mu_dc window views (float32
  material arrays) and the compact assembled nu of the saturating cells (`_assembled_reluctivity`, exactly what the
  operator holds: the loop state moves on after the last solve, so B / |B| never read exp(state)); every reader of
  nu goes through `_reluctivities`. MagneticPhysics3D likewise keeps views. Compact reads of a view take
  `unravel_index` (a ravel of a window view copies the window).

## MagneticPhysics (2D tensor)
- Axisymmetric unknown psi = r A_phi on the PLANAR DiffusionSystem (`axisymmetric=False`, c = nu / r_c): never the
  ring-weighted operator (its 2 pi r face weights cancel the 1 / r; psi = B r^2 / 2 is then no solution). So the
  inherited `add_boundary_flux` / `perimeter_flux_weights` get planar weights: a perimeter surface-current feature
  would need the 1 / r of the psi form.
- Axis rule: x_lo = `BCSide('dirichlet', 0, conductance_factor=2.0)` set in `__init__`; the core face terms carry it
  into assembly, flux, energy, the remanence face drive and face_inward_flux, and ac_operator's `boundary_terms` reads
  it. Never add an axis term elsewhere (extra_diag stays unused). The axis remanence drive is halved, so uniform B_z
  keeps psi = B r^2 / 2 discrete-exact.
- Open (graded r): every psi-form half cell uses resistance (dr / 2) r_c / nu, the exact ring integral (dr / 2)(r_c
  +- dr / 4) / nu; the +-dr / 4 cancel only for equal neighbour dr, so uniform B_z, the remanence drive and the x_hi
  ring value are exact on UNIFORM r only (face flux error ~(dr_L - dr_R) / (8 r), worst near the axis; ac_operator
  alike). flux_concentrator_needle (+ _ac) grades r; every axisymmetric self-test is uniform. Fix: per-face half-cell
  resistances (a conductance hook or a 1 / r WindowMetric in core.fv_diffusion), mirrored in ac_operator and
  face_drive_terms, plus graded-r self-test variants.
- Mirror planes: the face's own sign (`core.mirror.FACE_FEATURE_SIGNS`: symmetry A even, insulation A odd; A / J
  polar_z, B / H axial). `symmetry` IS a model mirror plane (`MIRROR_FEATURES`, all three types; 3D -1),
  magnetic_insulation never is; the type has no Neumann face that is not a mirror, so a face next to ideal iron and
  both y faces of a 1D slice double the totals too (x 4 in the 1D self-tests / examples, `selftests._Y_MIRROR_FACTOR`).
  Every solver is a `mirror_totals.MirrorTotals`: `_create` calls `set_mirror_faces` before any feature applies (it
  raises once a coil exists, and on the axisymmetric x_lo / x_hi); energy and losses x `window_mirror_factor`, a coil
  reaching a plane gets `mirror_factor` at add_coil (stranded in series, solid in parallel; a solid conductor on an
  odd plane raises). A new total of these solvers multiplies by the same factors. Validation:
  `definition_magnetostatics.mirror_plane_issues` (also 3D); parity sectors refuse (one sign only).
- RerunSolver (sweep reuse_build): MagneticPhysics follows current_source densities (`add_source` of the difference)
  and coil currents (`set_coil_current`; the live current is `_coil_currents`, CoilRegion.current stays the build
  value); magnetic_fields_ac follows `frequency` (`set_frequency`), not its coil values. The triangle and 3D
  magnetostatics solvers have no hook yet.
- Kept preconditioner (`kept_preconditioner.py`, 2D and 3D tensor loops): the pcg's own `_multigrid` stays None (the
  wrapper owns the hierarchy), and the wrapper is no PCGSolver, so the system's anchor check is not deferred to the
  CG's first sync. A new `_assemble` caller inside a loop must `offer_state` first, else nothing is kept. Keeping the
  hierarchy across a reused sweep's loops saved 1 of 9 setups and cost 20 % more CG iterations (bh example), so each
  loop's first assembly builds fresh.
- Single precision: the float32 CG stalls near a 1e-6 relative residual and iron cells (c 1 / mu_r of air's) keep |B|
  errors of percents (bh example: H_cores 8 % off at K = 100), so no stop floor can be met (a rounding floor of the
  tolerance did nothing): `Float32StallWatch` ends the loop. Validation warns on 'single' with a mu_dc contrast >= 100
  (`single_precision_contrast_issues`, 2D and 3D); the role keeps 'single' (dropping it would clash with
  solver_defaults.precision).
- Why no tensor Newton: the exact Jacobian of the centre-|B| secant has a 13-point stencil and the 9-point c_xy
  approximation is not positive definite under strong saturation. Undamped Picard diverges where the current drives H
  past the knee; Anderson converges (`examples/magnetic/bh_curve_table_core.json`).
- B-H tables propose H-based (centre |H| = nu |B|), J_s curves B-based: with s = d ln B / d ln H and w the cell's
  share of its circuit reluctance, the Picard factor is -(1/s - 1) w B-based but (1 - s)(1 - w) H-based. A B-based
  table loop diverged on the mu0 extension; deep J_s saturation under a current drive has the same limit (Anderson
  covers s down to ~0.12). Tables reach `add_bh_curve` per `ctx.material_regions` entry; `Simulation.add_material`
  refuses them.
- Energy / inductance are secant values; `field_energy` (1/2 u^T K u) is the discrete identity W = I Lambda / 2, w_mag
  (centre B) integrates to it only approximately at interfaces.

## magnetostatics_3d (MagneticPhysics3D, tensor)
- Against 2D: no xx / yy swap (c = mu_dc per axis); face kinds swapped (insulation = Neumann default, symmetry =
  Dirichlet u = 0), so FACE_FEATURE_SIGNS are opposite (insulation +1 = phi_m even): B / H declare `polar_*`
  parities, J `axial_*`. Currents only through `coil` (a free J has no T without a solve).
- Corner anchor: a new holding method must call `_release_corner_anchor` BEFORE holding cells, else the next assembly
  may clear a user-held corner cell.
- W = Lambda I / 2 holds to the CG residual only when every held cell is at 0 (corner anchor); a ring, ideal iron at a
  potential or a symmetry face add a boundary term. `field_energy` and `coil_results` read the same
  `_total_face_flows`: keep them on the same face sets.
- Saturation: one isotropic ln mu_r state; both curves propose H-based (2D runs J_s B-based); J_s cells need equal
  mu_dc_xx / _yy / _zz.
- `selftests_3d.py`: the mu_r 1000 sphere uses first-order Richardson (a staircased high-mu body has excess
  polarizability first order in h: +6 % / +3 % core field at a / 10 / a / 20). The Ampere checks telescope only on
  uniform cells across the winding.
- Memory: magnet3d_*_sphere coarse (149k cells, pcg multigrid) estimated build peak ~1.5x measured (from
  `solvers.multigrid.setup_extra_bytes`), the fine one (593k) ~10 % low; both predate setup_extra_bytes taking the
  largest setup step: re-measure.

## magnetostatics_3d on tetrahedra (MagneticFEMPhysics3D)
- Automatic 'linear' = sparse_direct (pcg does not declare tetrahedra): double cuDSS Cholesky on the gpu with
  nvmath-python, else host SuperLU. Constraints by `add_fixed_dofs` handles re-set per full assembly: no release
  ordering (unlike the tensor anchor).
- Coil results carry 'energy' (also tensor): the identity self-test reads it there, since field derived values raise
  on tetrahedra. Not done: a tetrahedra mirror-plane self-test, nodal B display, Lorentz force.
- cuDSS (unmeasured): saturation refactors per iteration (15-50), so plan reuse (`CudssFactorization.refactorize`) is
  the win; cudss_mixed and the mixed host back ends are risky at mu contrasts 1e3-1e5 (measure single_residual on an
  iron-core model first).

## Triangle meshes (MagneticFEMPhysics, MagneticACFEMPhysics)
- Unknown A itself (planar A_z, axisymmetric A_phi), NOT psi: P1 cannot hold psi = B r^2 / 2 and int (nu / r) |grad
  psi|^2 is singular on axis elements. B-space form (`fem.magnetostatics`), H = diag(nu_xx, nu_yy) B: NO xx / yy swap.
- One quadrature for W, R and K_t (cached per system in `blocks`): a rule change keeps all three on it, else Newton
  loses its quadratic rate and the line search its consistency (magnet_fem_tangent_check catches it).
- Derived values read the nodal display fields, so a material outline smears normal H / tangential B into the
  boundary elements (a mean H over an iron slab read ~2x): self-tests read such fields at interior points ('at'),
  never region means. Held elements carry B / H / w_mag 0 (`active_elements`: an axisymmetric held psi / r
  interpolates to a spurious B), but the nodal display averages include them.
- cuDSS: Newton = one factor + ONE solve per step on a fixed pattern per constraint set (`_to_csr` and
  `tangent[free][:, free]` keep explicit zeros); plan reuse (`previous=`, keyed on the constraint revision) pays above
  ~2e5 dofs and in sweeps (Amdahl: host law / assembly / line search cap the gain at ~2x). Double only: cond ~ mu_r
  h^-2 is past float32 at mu contrasts 1e3-1e5.
- The Session refuses max / min over mirror images on triangles: triangle self-tests read scalars with operation
  'expression' (`selftests._scalar_value`) and fields 'at' points.
- Open: a same-mesh Lorentz-force self-test, an axisymmetric remanence self-test
  (A_phi of an axial magnet shell has a 1 / r part, not in P2: needs a convergence tolerance), held regions in any
  triangle self-test.

## magnetic_fields_ac
- Tensor MagneticACPhysics runs the real DiffusionSystem assembly only for the mu_dc check, held cells, ring and the
  anchor check (`_AssemblyOnlySolver`), then assembles complex K in `ac_operator`. Complex face conductances are
  series means of complex c, so K(Re nu) + j K(Im nu) is NOT used (wrong at interfaces).
- Triangles: MagneticACFEMPhysics is no MagneticFEMPhysics subclass; it keeps a `MagneticFEMSystem` for the dof map,
  curl blocks and constraints only (its real 'linear' wrapper is never factorised). Load, mass, circuit rows and the
  loss integral on one rule make the circuit row and the Poynting identity exact to round-off. A solid conductor
  touching the axis raises (int sigma / r dA diverges); validation cannot see it.
- Factor (`circuit.RecycledFactor`, both meshes): complex symmetric, not Hermitian: LDL^T through
  `MAGNETIC_AC_LINEAR_ROLE` ('auto' = cudss on triangles). A cuDSS factor is not recyclable (an operator change
  refactorises on its plan); a SuperLU / MUMPS / PARDISO one is (GMRES on the old factor). complex64 'mixed' is
  doubtful at mu contrasts 1e3-1e5 plus circuit rows, so the host back ends are double only and keep the SuperLU size
  rule (`circuit.triangle_size_issue(back_end=)`, tensor `size_issue`; texts name the library by
  `host_direct.factor_label`); cusolver_qr is a validation error.
- Memory (`definition_ac._cudss_estimate`, `_host_factor_items`): uncalibrated; circuit border entries, assembly,
  post-processing and GMRES are left out.
- Couplings across meshes: induction heat / Lorentz force read per-element `fields` on the same mesh; the transforms'
  `magnetic.fields` reads across meshes are not covered (mesh_transfer 'value' passes the source field only).

## Self-tests
- Tensor (`selftests.py`, `selftests_ac.py`, `selftests_backend.py`): expectations are the exact DISCRETE solutions on
  the run's cells (`_planar_line_solution`, `_axisymmetric_line_solution` with the doubled axis conductance), float64
  CG 1e-12, tolerance 1e-6; circuit-row / Poynting identities enter the error scaled by 1e-6 / 1e-10. Triangles
  (`selftests_fem.py`) use the CONTINUOUS closed forms (P2 holds piecewise-quadratic A exactly). No tensor
  flux-driven saturation case; Anderson deep saturation is the example `bh_curve_table_core.json`.
- Paper-matched versions of the needle templates (Carey 2026 Eq. 2): `experiments/Ben_flux/` (3D: `needle_3d/`).
