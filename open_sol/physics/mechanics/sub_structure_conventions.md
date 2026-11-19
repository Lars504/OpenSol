# open_sol/physics/mechanics — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Solid mechanics (C3). Designs (repository root):
`solid_mechanics_3d_plan.md`, `piezo_3d_plan.md`.

## Layout (2D and 3D)
- Tensor meshes: `solver_common.SolidMechanicsBase` (subclass contract in its docstring) under `solver.py` (2D) /
  `solver3d.py`; element meshes: `FEMSolidCommon` under `solver_fem.py` / `solver_fem3d.py`. A tensor solver writes
  outputs only through `_show_outputs` (or `_extend_into_pml` after its own extra writes): a bare `_write_outputs`
  leaves the PML ring stale.
- Definitions: `definition_common.py` (pieces both types share), `definition.py` (2D type, every solid coupling,
  registers definition3d), `definition3d.py`. A solid that `StudySpec.solved_physics` leaves out is a stationary
  background: validate / estimators rebind their context by `definition_common.background_context`, create reads
  `solid_study`; a new study check goes through them, never `spec.study.type` of an unbound context.
- Type catalogue: `solid_mechanics` (default 'solid'; shares alpha / alpha_xx / alpha_yy and their arrays with heat,
  adds alpha_zz; rho read only by the modal and frequency_response studies; youngs_modulus / poisson_ratio /
  photoelastic_p11 / _p12 are elastic_waves' Param objects, `material_params.py`; optional c_ij, piezoelectric,
  axial_wavenumber = 2.5D guided modes, frequency_response study, periodic_x / _y + bloch_kx / _ky Bloch cells on both
  mesh kinds: section "Bloch cells, 2D"). `solid_mechanics_3d` (3D only, separate type like em_fdtd_3d; default
  'solid'; same material Params incl. rho; tensor meshes: lowest modes only; tetrahedra: target_frequency / search
  targets / band, 21 c_IJ, piezoelectric (charges in C, static on 'linear' sparse_direct, short / open modes on
  'eigen'), frequency_response on 'linear').
- Exact self-test cases (1e-8 .. 1e-9) need float32-exact material constants (`selftest_helpers.float32_value`); an
  isotropic E / nu pair is not exact even then (its float32 storage shifts frequencies ~7e-9), so exact roller boxes
  use orthotropic c_ij.

## Backends (gpu and cpu; solver_plan S8 E)
- Both types run on 'gpu' and 'cpu'; modal, guided and piezo solves are host on both.
- 3D: every kernels3d / multigrid3d RawKernel is a twin. The numpy twins allocate per call, so a cpu CG iteration is
  far slower (the estimator books their temporaries as cpu build extra); the per-iteration kernels have numba twins. A
  numba twin that needs another kernel calls its registered twin (`_TWINS[name]`), never its factory, so each kernel
  is compiled / cache-loaded once per process.
- Parity cases (`selftests_backend.solid_parity_case`) grade the cpu runs by the wrapped case's own check and the named
  fields against the gpu runs relative to the max of the gpu fields of the same kind (u / strain / stress + von_mises):
  name a NONZERO field of each kind (a zero u_x under rollers would be graded by its own round-off); a field only the
  parity reads needs a closed-form item in the case's check.

## SolidMechanicsPhysics / ElasticSystem
- Static, small strain; every coefficient goes through `stiffness.ElementStiffness` (E / nu or c_ij), never lambda /
  mu directly. Rules are written for plane strain unless they say otherwise; out_of_plane plane_stress /
  generalized_plane_strain: `solver.py` header, `stiffness.operator_coefficients`, `generalized_plane_strain.py`.
  Setting thickness (planar) only scales reactions (N): the system stays per m of depth.
- Axisymmetric: x = r from the x_lo window edge (padding ignored), y = z. A new load goes through `add_stress_load` /
  `add_volume_load` / `face_node_lengths` / `segment_node_lengths` (ring weights), never `element_area` products.
- Q1 accuracy (not in the help texts): stiff in pure bending, so mesh thin films with >= 4 cells over the thickness;
  plane-strain bending stiffness is high by ~c / n^2 (n cells over the depth, square cells), c = (1 - nu) / 2 + nu^2 /
  (1 - 2 nu): 3.5 % at nu = 0.3, n = 4 (examples/mechanics/simply_supported_beam_gravity.json matches it to ~1e-3). A
  fixed_domain cell holds all four corners, suppressing its free strain: the far field gets a rigid kink (translation
  + rotation ~ strain x dx^2 / depth), so far-field displacements carry an offset (stresses fine).
- Rigid-mode check (`ElasticSystem._check_rigid_modes`): hinge (corner) contacts count as joined, so a mechanism
  through a corner contact shows as an unconverged CG, not as a refusal; validation only checks that both directions
  are held somewhere.
- Linear solver: `definition.SOLID_LINEAR_ROLE` (its comment holds the choices and their measurements: block-Jacobi
  pcg, 'mixed' on the gpu, sparse_direct as the stationary / time_dependent default); the 2D binding (CUDA-graph check
  blocks, float32 copies of 'mixed', the factor deferred to the next `solve` so modal and piezo solves never factorise
  it): `ElasticSystem._bind_linear_solver`.
- Reactions are K u - f of the last solve (`load_full` of the last `_build_rhs`): every load change sets
  `_loads_dirty` (it also drops the frequency response's kept `_harmonic_drive`), so loads never change between a solve
  and a read. n_solves doubles as the couplings' revision counter.

## Anisotropic stiffness (`stiffness.py`, c_ij entries; piezoelectric plan stage 5)
- Voigt conventions, the Lame fallback per component and the formulations: `stiffness.py` header; which entry
  allocates which arrays: `materials.OPTIONAL_STIFFNESS_SCHEMA`. c11 / c12 / c22 / c66 / c16 / c26 are elastic_waves'
  arrays (its kind / unit / bounds; an allocated c16 / c26 switches its monoclinic kernel on in a shared model).
- solid = c11 > 0, so a c_ij-only cell is solid and must set every entry its formulation reads (a crystal entry expands
  to all: no youngs_modulus needed).

## Eigenfrequency study, 2D (`modal.py`)
- Pencil, ranking, rigid modes, several shifts, band search and merge: `modal.py` header. Every solver (tensor 2D,
  triangles, tetrahedra; not the matrix-free solver3d) builds its pencil in `_modal_pencil(lowest_target)` ->
  (ModalPencil, finish), shared by `solve_eigenfrequencies` and `modal.MultiShiftModes`: a new pencil variant goes
  there once (finish = piezo expansion, on tetrahedra squeeze films). Estimators read `modal_kept_count` and
  `definition_helpers.eigen_search_counts`.
- Modal (and frequency-response) operators are host float64 rebuilt from the ElementStiffness
  (`_build_harmonic_operators`), never the solve-precision device CSR: a single-precision 'linear' node must not
  perturb the pencil.
- `select_resonance(k)` shows the mode without a static solve: n_solves stays 0, so support reactions raise (results
  left out with a warning).
- ModalPencil vouches its terms symmetric (`symmetric_known`, no check; see `_pencil`): a new term that breaks symmetry
  must drop the vouch. Hermitian (Bloch) counts on cuDSS / MUMPS / PARDISO: `arpack_shift_invert.count_below`.

## Piezoelectric plane strain (`piezo.py`, setting piezoelectric; piezoelectric plan stage 5)
- Unknowns, signs, electrodes, the column map G and its scaling: `piezo.py` header; every G^T X G goes through
  `fem.assembly.congruence`.
- The 2D static solve (`_solve_system`) is a private host SuperLU of G^T A G (counted as piezo_*, outside any role):
  the 'linear' pcg is still assembled, unused. The harmonic system is kept while mass and loss are the same objects
  (`PiezoSystem.harmonic_system`): callers cache them, a new object is a new reduction.

## Frequency response (`frequency_response.py`, `bvd.py`; study frequency_response, model/study_frequency_response.py)
- In Hz (the EM frequency_domain study stays wavelength-based); convention, losses and the factor routes:
  `frequency_response.py` header and `HarmonicSystem`. 2D tensor solids use a private SuperLU outside the roles
  (counters harmonic_* / piezo_harmonic_*; the 'linear' pcg assembled, unused); triangles and tetrahedra their
  'linear' sparse_direct runtime (3D operator class 'spd' is nominal there).
- Shown fields are the real parts with dT = 0 and no sigma0 (`_write_outputs(thermal=False)` on every mesh kind;
  solid_frequency_response_bar's 'heated' run pins it on the tensor mesh). On the tensor mesh support reactions and
  electrode tables raise and probes record nothing (FEM meshes: section "Prestress, damped modes, complex records").
- Reduced sweeps (fem sub file "Reduced sweeps") and rows-only direct solves: after `HarmonicSystem.set_output_rows`
  `solve(fields=False)` is exact ONLY at the registered rows, so a solver's records read only rows it registered
  (supports, probes, `PiezoSystem.electrode_rows`). Fields only at `wants_fields` frequencies; a skipped last one is
  shown by `publish_frequency_response`.

## Guided (2.5D) modes (`guided_modes.py`, setting axial_wavenumber)
- Formulation (u_z = -i w, isotropic cells only): `guided_modes.py` header. The optomechanics study creates the solid
  `guided_by_study` for backward scattering (`definition_optomechanics.backward_scattering`, read by create, validate
  and the estimator) and calls `set_axial_wavenumber(2 Re beta)` after the optical solve; `modal_overlap_data` then
  carries 3 dofs per node.

## Bloch cells, 2D (`bloch2d.py`, `definition_bloch2d.py`; settings periodic_x / _y, bloch_kx / _ky)
- Same phase convention and reduction as the tetrahedra cells (`bloch3d`, section "Tetrahedron meshes"): the 2D
  identification is a `bloch3d.BlochCell3D` with z never periodic (`vector_masters(c)` for every interleaved layout: 2
  plane strain, 3 guided (u_x, u_y, w) and 3 piezo (u_x, u_y, phi), all sharing the node phase), prolongation
  `bloch3d.bloch_prolongation(master, phase, free)`. Tensor: `tensor_bloch_cell` pairs the window corner nodes by index
  (x_hi column -> x_lo, y_hi row -> y_lo, corners chained by `fem.dofs.bloch_master_map`; L = the window widths sum, so
  graded cells need no matching), built once in prepare. Triangles: `system_fem.ElasticFEMSystem.periodic` makes every
  assembly rebuild `bloch` on the SOLID map (`fem.dofs.bloch_masters`: a void-touching side raises when the solid does
  not reach both sides alike; L = the node extent, a y_lo / y_hi PML ring strip keeps the x extent).
- Rigid check: regions join through partner pairs (`bloch2d.joined_regions`); a wrapping region cannot rotate, keeps
  nothing at a non-trivial phase and its unheld translations at phase 1 (counted with rigid_motions_allowed = target >
  0, else refused); a region touching no partner keeps the ordinary rules. The tensor system gained `bloch` /
  `rigid_motions_allowed` / `n_rigid_motions` for this only (still no free bodies otherwise); guided at k_z = 0 adds the
  u_z translation (`_guided_bloch_rigid_motions`) instead of check_antiplane_supports.
- Eigenfrequency study; a plain periodic cell (every bloch_k 0) on triangles also stationary / time_dependent:
  `ElasticFEMSystem.solve` factors K_r = P^T K P of the real prolongation (`_periodic_p`, per assembly; the ordinary
  support rules apply, `static_periodic_cell` turns the Bloch support rules off), reactions stay K u - f at held dofs.
  Validation `bloch_issues_2d` refuses the rest (tensor statics: the PCG has no P; frequency_response: HarmonicSystem's
  output rows assume no column map); stress_stiffening / geometric_nonlinearity, generalized plane strain and
  axisymmetric are refused (no periodic prestate / superposition / axis rule). Supports
  and electrodes on a periodic face are validation errors; a face of the other axis may reach the periodic corners
  (consistent: a master is held with any dof it stands for). Floating electrodes touching a periodic side raise in
  `_assemble_piezo` (their G tie would fight the phase); `PiezoSystem(require_ground=)` and `_piezo_issues` drop the
  ground rule at a non-trivial phase.
- Triangles + elastic PML on the other axis (a surface-wave cell): the damped pencil reduced by a complex P is neither
  complex symmetric nor Hermitian, so `ModalPencil._pencil` asks a general ShiftPencil (symmetric = vouched) and
  `eigs_shifted` honours the caller pencil's flag (LU). Tensor complex modes: `_show_complex_mode` (real parts, u_abs).
- ARPACK can return only some copies of an exactly degenerate band (the Q1 square cell's 4-fold c_T 2 pi / a band at
  Gamma came back 3-fold with n = 4; guided at k_z = 0 its 8-fold in-plane + antiplane band 4-fold even with n = 10):
  cases ask a few more modes than they grade, or grade at a k off the symmetry point.
- Self-tests `selftests_bloch2d.py` (also the template solid_bloch_band_diagram: definition.py imports it, so its
  sibling selftests imports stay inside functions; listed in import_contracts TEMPLATE_SOURCES); example
  examples/mechanics/rayleigh_wave_bloch_dispersion.json. Not built: periodic frequency responses (the harmonic
  admittance of an infinite IDT would need a prolongation through HarmonicSystem's output rows), tensor periodic
  statics, Bloch with piezoelectric on triangles (no 2D piezo there). The error_indicator treats periodic sides as free
  edges (one-sided traction jumps).

## Eigenfrequency study, 3D tensor meshes (`ElasticSystem3D.lowest_modes`; tetrahedra: section "Tetrahedron meshes")
- Decision: matrix-free shift-invert at sigma = 0 (no CSR in 3D; a 3D elastic SuperLU fills host memory at ~30-40^3
  cells). K - sigma M is indefinite above omega_1^2, which CG cannot solve, so the tensor mesh finds the LOWEST n
  modes only (target_frequency / search 'targets' / 'band' are validation errors,
  `eigen_search_issues(tensor_mesh_3d=True)`). Interior targets would need MINRES / a direct factor; free-free a
  negative shift + rigid filter.
- `modal.solve_lowest_modes_matrix_free` -> `ArpackShiftInvertSolver.eigsh_inverse` (OPinv = one PCG solve, M = the
  kernels3d mass product) on host vectors of the free dofs. Lanczos tolerance = max(node tolerance,
  `LANCZOS_TOLERANCE_PER_CG` x CG tolerance): with 0 the inexact inner solves floor the Ritz residual and ARPACK
  restarts forever (capped `MAX_LANCZOS_RESTARTS`). Validation errors on a single-precision 'linear' node here.
- An eigen node of type block_eigen (opt-in, `eigen_role(block_eigen=True)`; tensor grids only, so tetrahedra misfit
  it) takes `ElasticSystem3D._lowest_modes_block` -> `modal.solve_lowest_modes_block` (info 'method' 'block_eigen'):
  full-dof rows masked by `free`, ONE launch for all rows of a brick product / mass product / block Jacobi
  (`BrickKernels.product_rows` / `mass_product_rows` / `precondition_rows`: the kernels' `_rows` entries, one thread
  per (row, node), the one-vector body shared as a __device__ function; numpy twin `kernels3d.rows_host` = the
  one-vector twin per row); float32 rows on float32 copies (the mixed linear solver's `_single_operator` +
  widths_single when kept, else made per solve); the 'multigrid' preconditioner reuses `self.multigrid` even after its
  CG guard fell back (LOBPCG only needs it SPD), else builds a private BrickMultigrid on the float32 copies (block
  Jacobi + warning when the build raises), and cycles the block through `multigrid3d.BlockCycle` (the V-cycle of every
  row per launch of the mg3 `_rows` entries; own (rows, n) work rows per level = `block_cycle_bytes`, the hierarchy's
  operators / tables / coarsest inverse shared; float64 rows cast in and out through its float32 rows; sized by the
  first call's rows, regrown if more come). A change of a one-vector kernel body or of `BrickMultigrid._cycle` /
  `_smooth` / transfers must keep the `_rows` entry and `BlockCycle` in step. Exactly n_resonances pairs are asked
  (no EXTRA_EIGENVALUES: the guard vectors are the margin). No inner CG, rhs / CG unknown untouched. Still needs a
  float64 'linear' solver (the system dtype). Estimators: `_block_eigen_values(context)` switches the 3D memory items
  (block_eigen.block_bytes + copies + private hierarchy + block_cycle_bytes, transient 'eigen solve') and the time
  recipe (block_eigen.time_estimate; setup_ops flags multigrid). LOBPCG cannot target interior modes either.
  Self-tests: section "Self-tests" (solid_3d_eigenfrequency_block_eigen, _block_eigen_levels).
- Inner solves count in `n_solves`: `reaction` raises after an eigen solve / shown mode until the next static
  re-solve (`_shows_mode`), and `rhs` / the CG unknown are scratch (`_rhs_built` cleared).
- Exact 3D roller-box modes need pairwise coprime cell counts per axis (9 x 5 x 4: equal phases on two axes give
  degenerate pairs); odd N_x has no node at the mode-1 sine peak (expected shape divides by max_i sin(pi i / N_x)).

## Optomechanical g0 (`optomechanics.py`, design: `optomechanics_plan.md`)
- Host numpy only (no cupy import; em helpers inside functions). The optomechanics study solves the optics, then
  `solve_eigenfrequencies`, then `OptomechanicalCoupling.evaluate(optical, solid)` per feature (solver's
  `optomechanical_couplings`, filled by the apply); `modal_overlap_data()` hands widths / edges, solid cells, p11 / p12
  / p44 and the host `modal_mass`.
- Face rule: interior faces between two window cells, Q_d = mean of the face's two corner nodes, Delta eps = lo - hi;
  `face_mean` / `face_jump` are the one face helper (optical_force too). em_modes faces: ghost-filled cell eps, |E_tan|^2
  as neighbour Yee means, D_n = eps0 eps_face E_n, per-component bend factors (`ModePointMaterials` centre /
  face_factor, z_centre / z_face_factor). The photoelastic overlap face-averages the cell Delta eps exactly as the
  operator does, so it is the exact first-order perturbation of a TE slab.
- em_fdfd_hz (`in_plane_hz_overlap`): face fields from `FDFDPhysics.face_fields()` (curl at the assembly omega,
  rescaled by omega_t / omega_res; evaluate selects the coupling's resonance and re-selects the shown one); a missing lo
  face of grid row 0 reads 0 (the operator's rule, not `lower_face_partner`'s duplicate). Denominator mu0 = 1 / (eps0
  C0^2), never core MU0 (its 1e-10 mismatch shows in the exact cases).
- om_integrand (`requires_study='optomechanics'`, parity even): stored by evaluate, shown for `selected_resonance` of
  the first evaluated coupling. Kept out of `state_fields` (`state()` has no such array): a new study-only renderer
  field must stay out too. Tensor meshes only: `SolidFEMRenderer` drops it from its `field_infos`, since default
  snapshots store every listed field (`_stored_field_keys`) and a listed field the renderer cannot read fails the study.
- Normalisation: |Q| = 1 at the reference node (peak or nearest reference_x / _y); g0 independent of it, g_om and m_eff
  not. m_eff = length x 2^(EM mirror faces) x Q^T M Q: EM mirror faces need solid rollers (validation warning); the
  em_modes mirror_factor cancels in the in-plane ratio. brillouin_gain (last column): em_modes with
  mechanical_quality_factor only (else NaN).
- Backward scattering (scattering 'backward', em_modes only): guided modes at k_z = 2 Re beta (evaluate refuses
  another k_z); `cross_section_overlap(backward=True)`: Stokes = (E_x, E_y, -E_z), so every |E_z|^2 term flips sign;
  strain of (u_x, u_y, -i w) with d/dz = +i k_z (`guided_axial_strains`). The sign of the shear terms rests on the
  optomechanics.py header derivation: om_crystal_photoelastic_contraction pins the assembly, not the convention. g_om
  / g0 / dn_eff_dx hold Re kappa; |Q| and m_eff include w.
- Strain only on solid cells (c11 > 0): a void cladding with p != 0 adds nothing. Moving boundaries on the periodic
  seam are not summed.
- photoelastic_p44 (2D solid_mechanics only, `SHEAR_PHOTOELASTIC_SCHEMA`; elastic_waves / solid_mechanics_3d lack
  it): default NaN, resolved per cell to (p11 - p12) / 2 by `shear_photoelastic_coefficient` — never compute with the
  raw array elsewhere. Delta eps_xy at cell centres (ez: no term). Optical operators hold diagonal eps, so this is a
  continuum form without a discrete reference: exactness cases pass p44 = 0.
- Crystal photoelasticity: every cell via `inverse_permittivity_changes` + `permittivity_changes`; cells of a material
  with a crystal entry with photoelastic data take its rotated p (`definition_optomechanics.crystal_photoelastic_tensor`;
  None when the material sets any photoelastic_* itself: user values win). The apply stores (host full-grid mask, p)
  in `crystal_regions` from ctx.material_regions, so before release_caches. Every overlap callback takes (xx, yy, zz,
  xy, xz, yz). Guided (backward) modes refuse crystal c_ij.
- Subpixel smoothing: the overlap keeps the staircase (face jumps / photoelastic eps unsmoothed; ez subtracts the
  'cells' deltas, `_unsmoothed_cell_eps`), D_n / fields / energy from the smoothed solve: not the exact first-order
  change of the smoothed operator (validation info).
- Validation infos (`_material_issues`, CPU on the material preview): zero p, crystal data on void, no permittivity
  jump at a solid cell, in-plane plane-stress / neither length rules.
- Self-tests `selftests_optomechanics.py`, `selftests_optomechanics_crystal.py`, `selftests_brillouin.py` (registration
  and templates in `definition_optomechanics`). Gotcha: a sin(pi j / N) roller mode is even about the centre, its S_yy
  odd: zero overlap with TE0 (om_forward_brillouin_gain uses sin(2 pi j / N)).

## FEM / axisymmetric overlaps (`optomechanics_fem.py`, plans O1 / O2)
- Host numpy; every integral carries 2 pi r. Moving boundary: n out of side 0, Delta = side 0 - side 1, isotropic eps
  at interfaces (eps_phiphi); photoelastic isotropic p only (a crystal tensor is not phi-invariant). 3D solids:
  phi-average on `ring_points` (n_phi > highest order + 1). Pass `geometry_order = layout.nedelec.geometry_order` to
  `interface_sides` so sample points match the field map.
- `evaluate` routes em_fdfd_bor optics or modal_overlap_data `kind` != 'tensor' to `axisymmetric_coupling_table`.
  `bor_optical_samples` needs the FEMBORPhysics `assembly` (a tensor-mesh BOR raises); interfaces touching PEC / the
  PML ring dropped. Solids kind 'triangles' (axisymmetric) or 'tetrahedra', sampled by point location in their own
  mesh. q_n from side 0 if solid, else side 1, else 0 (never the zero-padded field of a void element). m >= 1: x
  `ring_factor(m)`, photoelastic 0. m_eff = Q^T M Q [kg] (no length / mirror factors), no om_integrand. Params
  `prescribed_normal_displacement`, `axis_x` / `axis_y` (drawing y = 3D z).
- A 3D solid with em_fdfd_bor optics is wired at runtime / validation only: em_fdfd_bor is 2D, so no model holds both
  yet.
- Self-tests: om_fem_* (CPU), `selftests_optomechanics_bor.py` (heavy, cpu backend).

## Squeeze film (`squeeze_film.py` = the tet runtime, plan S6w; the film itself is `fem/squeeze_film.py`)
- Physics-neutral film in fem/squeeze_film.py, shared with membrane; this module keeps `film_faces`,
  `sealed_rim_axes`, `SqueezeFilmDamping`. Gas Params = `model.definition_helpers.SQUEEZE_FILM_GAS_PARAMS` (shared
  objects).
- Feature squeeze_film_damping (definition3d, tetrahedra; eigenfrequency / optomechanics only, warning elsewhere: not
  coupled into frequency_response): films evaluated in `solve_eigenfrequencies` after the pencil solve (`_modal_pencil`
  finish, on the merged list after a search); footprint from `_solid` of the last assembly, rebuilt when it changes;
  m_eff = Re(psi^H M psi) of the 1 m peak shape. resonance_table appends `GAS_TABLE_COLUMNS` (totals over films).
  Refused: Bloch cells, PML faces, closed incompressible film. definition3d imports squeeze_film (and fem.surface_mesh)
  at module level for its choice constants.

## SolidMechanics3DPhysics / ElasticSystem3D (solid_mechanics_3d)
- Isotropic 3D law sigma = lambda tr(eps) I + 2 mu eps - beta dT; forces, tractions and reactions in N.
- Bricks = window cells, nodes C-order, dof = 3 node + component, node = (i ny_nodes + j) nz_nodes + k; widths as 1D
  device arrays (graded axes need no kernel change); 3 x nodes < 2^31 (int32 indexing, prepare raises).
- No CSR: matrix-free product of `kernels3d.BrickKernels` (lazy float / double RawKernels incl. fused load / outputs:
  no float64 window temporaries). The PCGSolver is factorized once in `prepare` with masked product + block
  preconditioner callbacks (they read the latest assembly). `reaction` uses the CG unknown `_solution` as scratch and
  keeps K u there while (n_solves, assemblies) are unchanged (every PCG solve, Lanczos inner ones too, bumps
  n_solves); anything else that writes `_solution` or `unknown` between solves must reset `_reaction_key`.
- GPU product and mass kernels work in the per-axis sum / difference (Walsh) basis of each element's corner values
  (kernels3d preamble: the 1D integrals are orthogonal there, ~240 flops per (node, element) visit against ~520 for the
  pairwise blocks; one deterministic gather launch, no extra memory). `brick_block` stays for the diagonal blocks.
  Every CG step runs as `kernels3d.FusedBrickIteration` (pcg fused_iteration: elastic_update_precondition +
  elastic_direction_product, grid-stride with block partial sums that the next launch sums itself) and the gpu check
  blocks as CUDA graphs (pcg capture_blocks): `assemble` calls `release_block_graphs()` after replacing lame_lambda / mu
  / the float32 copies; anything else that replaces an array the fused operator reads must do the same.
- pcg preconditioner 'multigrid' (opt-in, speed plan A2; design `.temp_scratchpad/solidmech3d_multigrid_design.md`):
  `multigrid3d.BrickMultigrid` is the precondition callback (no fused iteration then; the check blocks of 2 are still
  captured, so a V-cycle must stay sync- and allocation-free: everything it touches is allocated in its __init__ in
  `prepare`, and `build` after each assembly, before `release_block_graphs`). Levels: `level_element_shapes` (shared
  with the estimators), elastic_product / diagonal_blocks / reduce_invert_blocks on a BrickKernels per coarse level;
  the cycle is float32 at every precision: level 0 is the float32 operator copy (`_single_operator`, also built for a
  double node with multigrid; a double CG casts the residual in and `_cycle_out` back; float64 cycles were 19x slower
  on a GPU with slow float64). Coarse
  levels are rediscretised, not Galerkin (volume-mean Lame values, injected holds; lost holds become pseudo-inverse
  null space), except the coarsest dense operator: P^T A P of the level above, probed by the same 27 colours (nested
  Q1 basis: 27-node stencil); rediscretised it over-stiffened a thin plate in void (solid_3d_multigrid_void fell back).
  Coarse free dofs: only SUPPORT holds are injected (`build(held=)`, the system's `held`, never `~free`); a void pin is
  not a hold: a coarse dof is free when a free fine dof lies in its interpolation stencil (`_reached_dofs`: P^T of the
  free mask > 0), so void coarse nodes beside a solid carry its surface values. Injecting `~free` kept only the
  mid-plane layer of a plate whose faces sit on odd nodes (surface values prolonged to half: no rigid modes, the coarse
  correction could not bend; the cycle ran as its smoother alone and tripped the guard). The probes run in the cycle
  dtype (float32): counters `multigrid_coarsest_kept` / `multigrid_coarsest_condition` show what they had to resolve
  (trouble expected towards 1e7: soft eigenvalues lost in the probe rounding; then probe in float64). Intermediate
  rediscretised levels still over-stiffen a thin plate that half-fills its coarse cells (bending ~4x; no self-test
  with >= 3 levels in void yet). Transfers are separable 1D gathers in the node coordinates (graded widths: weights != 1/2); restriction
  = the exact transpose (gather form), so pre / post Chebyshev + R = P^T keep the cycle symmetric for plain PCG.
  `_guarded_solve` (static and every Lanczos inner solve) caps max_iterations at `multigrid_budget()` (the block-Jacobi
  estimate CG_ITERATIONS_PER_AXIS_CELL x longest axis over CYCLE_COST_IN_JACOBI_ITERATIONS, within
  [FALLBACK_MIN_ITERATIONS, FALLBACK_ITERATIONS]): a solve unconverged within it switches the system to block Jacobi
  for good (warning, `multigrid_fallbacks`) and re-solves from there (its failed attempt is taken out of
  n_unconverged). A mixed solve that stalled (`PCGSolver.last_stalled`: round-off floor of the system, below the
  budget) is no fallback: block Jacobi stops at the same residual after its full iteration count (void plate: 3e-11 on
  both, tolerance 1e-12); it stays reported unconverged. Self-tests `solid_3d_multigrid` and `solid_3d_multigrid_void` (thin plate in void) fail on a
  fallback, so the guard cannot hide a weak cycle. Counters via `PCGSolver.preconditioner_counters`; estimator items 'multigrid levels' / build
  transient / coarsest host build, time recipe `_multigrid_ops` (cycle as preconditioner_ops, build as setup_ops).
- pcg 'mixed' (`CG_ELASTIC_3D_LINEAR_ROLE`; 2D: section "SolidMechanicsPhysics / ElasticSystem"): float64 system + `_single_operator`
  float32 copy per assemble (blocks inverted in float64, then cast). Automatic on the gpu backend (speed plan A1: the
  float64 product runs at the FP64 rate of consumer cards; 25 s -> 1.5 s at 2e5 nodes); cpu Automatic stays double;
  solver_defaults.precision replaces it. Kernels pick their real type from args[0]: every array argument of one call
  must share the dtype. Eigenfrequency accepts mixed. Estimator: mixed workspace (+ the fused second direction) +
  float32 copy item; time recipe `iteration_ops` = the fused pair.
- Node blocks are not diagonal: reduce them (fixed rows / columns zeroed, 1 on the diagonal) BEFORE inversion;
  inverting the full block puts free residual into held components at a roller node (silently wrong solve).
- Rigid-mode check = rank of the 6 x 6 Gram matrix of rigid-mode rows at each region's held dofs (26-connected;
  RIGID_RANK_TOLERANCE); `rigid_motion_text` names the slide / axis (shared with tetrahedra). Same hinge gap as 2D.
- 2D rules that do not apply: face filters use FACES_3D; body_force / temperature_change values with z are
  `BoxValues` added on the selection box only (`add_masked_to`: never a window / full-grid float temporary); no
  strain-optic / optical-force coupling (em_fdtd_3d's uint16 material index cannot hold three continuous permittivity
  components; plan section 7). CG iterations scale with the longest edge: keep 3D self-test strips short.
- Memory estimate: nodal vectors + PCGSolver `workspace_bytes` + masks, inverse blocks, element arrays; supports book
  their held dof mask as a feature item.

## Triangle meshes (`solver_fem.py`, `system_fem.py`, fem/elasticity.py; mesh plan stage T5)
- Shared with tetrahedra: `solver_common.FEMSolidCommon` (records / Newton reporting, element masks and values, dT and
  rho reads, `_loads_dirty`, `_ensure_outputs` over the class's `_output_groups`) on `HostInputCache` (alpha, dT,
  sigma0, beta, rho host copies kept read-only until materials_changed / inject_source / an assembly / a temperature
  feature: a new mutation of those inputs calls `_drop_host_inputs`). Tetrahedra also keep the sigma0 force
  -int B^T sigma0 dV there (`_initial_stress_force`: the finalize imbalance check and the static load share it).
- Outputs are lazy as on tetrahedra (`_OUTPUT_GROUPS` primary: u, u_abs, Im u; derived: strain, stress, energies);
  materials_changed and inject_source compute pending groups first.
- solid_mechanics dispatches on the mesh kind: `_create_solid` builds SolidMechanicsFEMPhysics on a TriangleDomain.
  `discretization` quadratic (P2, default) / linear, isoparametric on curved edges. Runs stationary, time_dependent,
  eigenfrequency, frequency_response, optomechanics (axisymmetric with em_fdfd_bor only; `set_axial_wavenumber`
  raises), all out-of-plane conditions and axisymmetric, c_ij via ElementStiffness (ring elements void unless an
  elastic PML), Bloch cells (section "Bloch cells, 2D"). Refused (`_triangle_issues`): piezoelectric (so no electrode /
  admittance), guided modes, the planar
  optomechanics study, solid_em_strain_optic from it, a 'linear' node other than sparse_direct (back_end choices
  DIFFUSION_DIRECT_BACK_ENDS keep cusolver_qr out; cg_solver_issues is tensor only: it reads nx).
- Why the gaps stay open: piezo needs phi on every window element + G on a DofMap (~a piezo_fem.py, would unlock
  admittance); guided modes need a 3-component guided stiffness; the planar optomechanics overlap is written on Yee
  faces (needs em_modes on triangles); solid_em_strain_optic has no mesh_transfer. Optical force into a triangle solid
  works through em_solid_optical_force's mesh_transfer.
- Elastic PML ring (`pml2d.py`, pml_<face> / pml_cells / pml_reflection; ring from `pml3d.PMLPadding`, outer side
  clamped): eigenfrequency only; static / prestress solves see the ring unstretched and clamped; pencil from
  `system.stiffness(stretched=True)` in a damped ModalPencil with the ring filter. No PML on the axis. Ring strains /
  energies are not physical.
- Azimuthal order m >= 1 (triangles, eigenfrequency, isotropic only): `components` = 3 (U_phi shown in u_z, rings
  weighted pi); the m = 1 axis tie u_phi = -u_r is `system.prolongation` (pass it to the
  ModalPencil); rigid motions 2 at m = 1, none at m >= 2; static, frequency response, c_ij refused;
  stress_stiffening at m >= 1 takes the initial stress only.
- Energy fields (`fields.ENERGY_DENSITY_FIELDS`) are display / probe fields, not in `state()`; tensor meshes render
  0 (energy probes there: validation error).
- Dofs: vector Lagrange 2 d + component on solid window elements (void-only nodes have none). Holds live on the
  full-mesh dof ids, so they survive a changed solid set; `face_edges` takes the window edge geometrically (also with
  a PML ring). Segment loads act on mesh edges (`ctx.boundary_edges`, any direction); probes on `ctx.point_node`.
- Solve: K_ff factorised lazily by the first static solve after an assembly (eigen / frequency response never
  factorise K alone) through `sparse_direct.DiffusionDirectSolver` over the 'linear' SparseDirectSolver runtime (the one
  place a cuDSS back end plugs in for static K; scripts get a private runtime, fem_* counters). Rigid check = rank of
  (1, 0), (0, 1), (-y, x) at held dofs per region (axisymmetric: a held u_z). Reactions = K u - f at held dofs.
  Eigen: consistent mass on the 'eigen' runtime (superlu / cudss / cudss_mixed; the SuperLU size rule is skipped for
  cuDSS); select_resonance makes reactions raise.
- Generalized plane strain: the tensor superposition on the K factor (`_solve_generalized`, `_unit_axial` kept until
  the next assembly); `_term_integral` uses the same quadrature as the loads so the conditions hold to round-off;
  reference = mesh window centre, moments about the drawing origin (generalized_plane_strain helpers reused via a
  SimpleNamespace with `reference`). Modal / harmonic outputs use eps_zz = 0.
- Run-time recipe `definition._fem_time_estimate`: constants uncalibrated, calibrate from the solid_fem_* time split.
  Session field sampling is no physics work (lands in the study rest).
- Frequency response (elastic only): the kept `_harmonic_system()` (K (+ K_sigma), eta K, consistent mass on the
  'linear' runtime: host SuperLU or cuDSS DevicePencil, one plan per sweep), keyed by the identity of
  (system.operator, `_harmonic_operators`, `_prestress`): any replaced = rebuilt, old factor handed on as previous.
  Reactions and the error indicator raise while it is shown (`_shows_harmonic`).
- Axisymmetric: stiffness / stress loads on the degree-6 rule (N_a N_b / r is rational); on the axis hoop strain =
  d u_r / d r. fem.assembly.assemble_load's default degree is 2 order + 3 with r (7 for P2: above the largest rule),
  so assemble_body_load passes its own degree.
- Fields: state() and slots one float32 per element (centroid, 0 off the solid and in the ring; P0). `fem_field`
  display: u exact nodal, strain / stress averaged at each dof over the solid elements around it (smeared across a
  material interface: read interface stresses from one side).
- `error_indicator()`: Kelly traction jumps, free / traction edges one-sided, fully held components skipped, no
  element residual; ValueError while a mode is shown.
- Self-tests `selftests_fem.py` (tolerances first estimates). solid_fem_kirsch sets h_min: the default dx / 100 floored
  the hole sizes (1 % low stresses).
- cuDSS verdict per solve: static K gains little (2D SuperLU fast to ~1e6 dofs; Cholesky through the
  DiffusionDirectSolver back end when it exists). Eigen (indefinite above omega_1: LDL^T) runs on cuDSS double / mixed;
  eigsh_shifted equilibrates by (|diag K| + |sigma| |diag M|)^-1/2, never |diag(K - sigma M)| (can cancel at sigma >
  0). Frequency response is the best candidate (complex symmetric, one pattern, 100-600 frequencies: plan reuse),
  double (near a resonance a float32 factor needs many refinement steps).

## Tetrahedron meshes (solid_mechanics_3d; `solver_fem3d.py`, `system_fem3d.py`, fem/elasticity3d.py)
- Dispatches on the mesh kind (every feature declares ('tensor', 'tetrahedra')): `_create_solid_3d` builds
  SolidMechanicsFEMPhysics3D. Settings `discretization` (P2 default / P1, affine), `target_frequency` (a nonzero value
  on a tensor mesh is a validation error), piezoelectric, periodic_* / bloch_k*, pml_*, electrode_edge_size.
  frequency_response on tetrahedra only.
- 21 c_IJ (`stiffness3d.py`, `STIFFNESS_TENSOR_ARRAYS_3D`, Voigt 4 yz, 5 xz, 6 xy; 2D names share their Params,
  `STIFFNESS_TENSOR_PROPERTIES_3D`): any entry allocates all 21; the nine orthotropic ones default NaN (= Lame per
  component), the other twelve 0. solid = c11 > 0; `check_positive_definite_3d` only when some solid matrix differs from
  its Lame matrix. Thermal stress is the full Voigt beta = c alpha (shear rows of a rotated crystal enter). Bricks refuse
  c_ij (`_tensor_issues`). A 3D crystal entry needs `'plane': 'full'` (materials sub file).
- Dofs: full-mesh scalar DofMap3D, vector dof 3 d + component; holds (n_dofs, 3); active = dofs of solid window
  elements, free = active and not held. Supports on window faces and (tetrahedra only) object interface faces
  (`ctx.boundary_faces`): fixed / displacement / boundary_load allow_interior, roller outer-face only. Tractions only
  on faces beside a solid element (a void-face load would leak into shared nodes).
- Estimates: displacement dofs = 3 x scalar dofs x `_solid_fraction` (mesh estimate's `material_elements` of materials
  setting youngs_modulus or c11), piezo potential on every scalar dof; full-mesh arrays keep the full scalar count.
  Every unknown / entry helper takes the spec.
- Outputs are lazy (`_OUTPUT_GROUPS`: primary, mechanical, electric, energy): `_write_outputs` only marks groups
  pending; `lagrange_field`, probes and `state()` compute them. Whatever changes unknown / potential / stiffness /
  inputs without a new `_write_outputs` (`inject_source`, `materials_changed`, `_assemble`) calls `_ensure_outputs()`
  first, and a new reader of
  `_nodal` or per-element outputs goes through it (why: the eigen study selects every mode for snapshots; ~47 s at 2.2e5
  P2 piezo elements with eager passes).
- Snapshot solutions (`FEMSolidCommon.solution_state`, triangles too; hooks `_shown_solution` / `_show_solution_state`
  / `_computed_outputs`): without materials the state is lazy (unknown, potential, complex solution, GPS terms, the
  `_write_outputs` settings + input fields; restore re-marks the groups pending and raises when `_stiffness` is
  another assembly's), with materials (couplings may re-assemble) the computed outputs + `_nodal` + E / nu. A new
  per-solve value an output group reads joins `_shown_solution`. Tensor solids (`SolidMechanicsBase`) copy every output
  / input field (eager writes). All raise while `_unpublished_frequency` is set.
- The point groups (mechanical, energy, electric's D) read one shared `_point_values(points[, imaginary])` = (values,
  strains) over every solid element (READ-ONLY; copy before subtracting alpha dT), one `element_strains_3d` per
  (solution part, point set); `_write_outputs` and `_assemble` (new `_solid`) clear it and the last pending point group
  drops it (`_release_point_values`), so it never outlives one shown solution. A new output at element points takes
  its strains from there (rows of a solid subset by searchsorted), never a fresh evaluation.
- Piezoelectric (`piezo_fem3d.py` + `piezo.PiezoSystem`): unknowns [u (3 n interleaved) | phi (n)] (block layout);
  K_uphi on solid elements, K_phiphi on every window element (void dielectric); K_uu = the system's CSR
  (`system.assemble` still builds holds / active set / rigid check, its K factor is never made). Materials
  `OPTIONAL_PIEZO_SCHEMA_3D` (18 piezo_e<i><J>, six eps_dc_*; shared 2D names keep their Params). Static
  `_solve_piezo`: G^T A G factorised through the 'linear' runtime's factorize_host(real, symmetric) once per assembly
  (previous factor handed over); reactions = u rows of A x - f; Q in C. electrode / electrode_domain tetrahedra only.
  Electric outputs always allocated per element on tetrahedra (0 without piezo); E / D smeared at a permittivity jump
  and next to an equipotential electrode_domain; stress -= e^T E. Tensor renderer returns zeros for them, they stay out
  of state_fields, electric probes on the tensor mesh are refused. A cuDSS 'linear' back end skips the SuperLU size rule
  and host factor item.
- Piezo modes: PiezoSystem + `coupled_mass_3d` to `PiezoSystem.modal_pencil` on 'eigen' (2D pencil rules: voltage = short,
  floating = open; their values ignored). `select_resonance` splits u / `_potential` and stores per-mode electrode V / Q.
  k_eff^2 needs two runs.
- Resistive electrode load (connection 'resistive_load', 3D only; load_resistance): tied like floating (no charge
  term: `_charge_terms` reads floating ones only), adds j s^2 / (omega R) to the master diagonal (`PiezoSystem.loads`).
  Eigen: the 'loaded' ModalPencil freezes 1 / omega at the target (target_frequency > 0 required), eigs_shifted, then
  Newton per mode on x^T T(omega) x = 0 (transpose, not conjugate); info load_residual (large = mode far from target).
  Modes complex (`_write_complex_outputs`, no electrode values), DAMPED_TABLE_COLUMNS. Frequency response:
  `HarmonicSystem(loads=)` adds (j / omega) B via pattern_sum (master diagonal is in K's pattern). Static solves raise;
  Bloch cells refuse it. Estimators still price the real eigen factor (they see settings, not features): ~4x low.
- Frequency response (tetrahedra only: the matrix-free CG cannot factor a complex pencil): `_harmonic_operators` =
  (consistent mass, loss) once per assembly / loss / rho change, then an elastic `HarmonicSystem` (`_harmonic`) or the
  PiezoSystem's, both on `system.runtime`. `_write_complex_outputs` sets `_shows_complex` (callers set
  `_shows_harmonic`): reactions / electrode tables raise, probes record nothing. Admittance =
  `replace(ADMITTANCE_FEATURE, grids=(tetrahedra,))`. Estimates uncalibrated.
- Bloch / Floquet cells (`bloch3d.py`; periodic_* / bloch_k*): `system.bloch` = BlochCell3D of
  `fem.dofs3d.bloch_masters` (shared with em3d; phases within 1e-15 of real made real, so k L = pi stays a real pencil);
  `vector_masters` / `coupled_masters` lift it (u and phi share the phase), `bloch_prolongation` gives P (a master is
  fixed with any dof it stands for). solve_modes(prolongation=) -> complex Hermitian pencil, complex modes shown via
  `_write_complex_outputs`. Rigid check joins regions across periodic partners and returns early for a non-trivial
  phase; with `rigid_motions_allowed` it counts instead of raising and drops |omega^2| <= RIGID_EIGENVALUE_FRACTION x
  omega_t^2. A piezo cell needs a ground electrode only for a trivial phase (`require_ground=`); electrodes touching a
  periodic face raise. Eigenfrequency only (`_bloch_issues`; `_support_issues` skips its support errors for Bloch
  cells). `_domain_requirements` declares periodic axes so gmsh writes periodic surfaces. Not built: static periodic
  solves.
- mode_shapes feature (`mode_shapes.py`, result kind 'mode_shapes' in `schema.RESULT_KIND_DOCS`; tetrahedra only,
  eigenfrequency / optomechanics, a validation error elsewhere; `gui_run_studies` = MODAL_STUDIES: a GUI Compute adds
  one to a solid without it, model sub file "GUI run features"): reads `eigen_solve` like resonance_table (same row
  order) and the public `solid_elements` (read-only host copy of `_solid`) & window elements. Surface =
  `interface_faces` of that set, oriented outward by the owner element's centroid; P2 faces split 1 -> 4 at their edge
  dofs (`face_local_dofs` order a, b, c, ab, ac, bc), nodes = the used face dofs ascending (P1: unsplit). The ModeShapes
  runtime caches the surface per (dof map, solid set); displacement = rows of the first 3 n_dofs (piezo phi dropped),
  float32, solver normalisation. bloch_k / periodic from `system.bloch` (0 off periodic axes), period = node extent
  (the L of the phase). No memory estimate: r x n_surface x 24 B per value, a fraction of the eigenvectors the eigen
  items count; a sweep keeps one per value (host, results.npz). Setting `fields` (default 'stress'): stress (+ the
  piezo potential) / stress_strain also keep the complex nodal stress / strain (`mode_surface_tensors`: the same
  `_point_outputs` + `nodal_average_3d` as `_write_mechanical_outputs` with thermal / prestress off, i.e. what a
  selected mode's GUI field shows, but for Re and Im of a complex mode each: select_resonance shows only Re; elements
  limited to the solid ones touching a surface dof, exact there), 48 B per node and mode each; density (nodal mean of
  rho) always. The surface fields are the page's basis (viz mode_player derives von Mises, principal stresses, energy
  densities): a new derived quantity belongs on the page, a new linear field (e.g. E / D) here. Self-test
  solid_fem3d_bloch_mode_shapes (template cell at k a = pi / 2: phase exactly -j, the opposite sign fails by O(1); fields
  stress_strain) pins the convention, the frequencies, the closed outward surface (divergence-theorem volume) and
  Hooke's law between the stored stress and strain (one isotropic material: exact up to float32).
- Elastic PML (`pml3d.py`): `_domain_requirements` returns a `pml3d.PMLPadding` (meshing grows the shell, materials by
  the clamped-centroid rule); shell carries stiffness (`_elastic_elements`), outer faces held at 0, s = 1 - j beta (d /
  w)^2 frequency independent (pencil stays linear in omega^2). a damped ModalPencil (`shell_dofs`) keeps Re
  sqrt(lambda) > 0, drops modes with > half |u|^2 on shell-only dofs. Eigenfrequency only, no piezo, no periodic
  axis, target > 0, no supports / loads / electrodes on a PML face (its window side is interior); `_support_issues`
  skips its rules. Not built: PML with piezo (G / scaling are real) and in frequency response.
- electrode_edge_size: `PhysicsType.mesh_edge_sizes` names electrode entities -> `TetrahedronSizing.edge_sizes` (gmsh
  only; info when unused).
- Mirror check (`_mirror_issues`): warns per Roller whose selected materials' c_IJ / piezo_e* / eps_dc_* flip sign under
  that reflection (`crystal_tensors.mirror_violation`): a half model would differ from the full one.
- Solve: K_ff lazily via DiffusionDirectSolver on the 'linear' runtime (grid_defaults tetrahedra -> sparse_direct; pcg
  does not declare 'tetrahedra'). Eigen on 'eigen' eigsh_shifted (superlu / cudss / cudss_mixed refined to 1e-12; the
  SuperLU size limit is skipped for cuDSS); the tensor mesh ignores the factorization (info).
- Free bodies: create passes `free_body` = `definition_common.free_body_allowed(study)` (eigenfrequency at target > 0, or
  frequency_response), prepare sets `system.rigid_motions_allowed`: rigid motions counted in `n_rigid_motions` (6 per
  unsupported region); static solves raise while > 0 (`_refuse_free_body`), the frequency response does not, an eigen
  solve at target 0 raises. Mode solvers take `rigid_motions` on top of n + EXTRA_EIGENVALUES (a low target finds the
  rigid cluster first) and drop the floor; the floor assumes eigen tolerance 0: a loose tolerance (~1e-6) can keep
  rigid modes at ~1e-3 f_t. Validation (`_free_body_modes`) turns support errors into one info. Estimators add
  `_estimated_rigid_motions` (partial supports not counted). The 2D triangle solid does the same (3 planar / 1
  axisymmetric per region); tensor meshes keep refusing free bodies.
- Fields: state() / slots one float32 per element (P0); display by `lagrange_field` (fem.renderer3d); stresses smeared
  across interfaces. heat_solid_thermal_strain copies a window-less heat's per-element rise (same-mesh).
- Estimates: vector unknowns 3 x (nodes + edges P2); SPLU_3D_WARN / LIMIT on vector unknowns (skipped for cuDSS
  eigen). Time: recipes at the vector unknowns x `_TET_LU_TIME_SCALE` / `_TET_SOLVE_TIME_SCALE` (P2 calibrated on
  solid_fem3d_roller_box_modes, P1 a guess). Do not price at stored entries / 7 "equivalent" unknowns: the LU curve
  grows ~n^2.15, so the 13x denser vector P2 rows came out ~60x too slow. Periodic axes multiply both host scales by
  `_periodic_separator_scales`; the cuDSS 3D recipe ignores it.
- Self-tests (tolerances first estimates): `selftests_fem3d.py`, `selftests_piezo_fem3d.py`,
  `selftests_frequency_response_3d.py`, `selftests_bloch3d.py` (template solid_bloch_band_diagram_3d: definition3d
  imports the module, so its sibling selftests imports stay inside functions), `selftests_pml3d.py`,
  `selftests_mesh_symmetry3d.py`, `selftests_free_modes3d.py` (nu = 0 bar: lambda = 0 makes the longitudinal mode exact
  for a free body), `selftests_eigen_search.py` (2D cases registered by definition.py), `selftests_prestress_modes.py`
  (2D cases registered by definition.py), `selftests_swept_plate.py` (heavy). Exact piezo values need float32-rounded
  constants.

## Prestress, damped modes, complex records (new_modules_plan K2, K3, K4; triangles and tetrahedra only)
- initial_stress_* (`materials.PRESTRESS_ARRAYS` 2D xx, yy, xy, zz; `_3D` Voigt; each allocated when set; one Param per
  name): `_initial_stress()` per solid element, `_total_load()` adds the initial-stress force (and the thermal load),
  static outputs add sigma0 (`_output_prestress`; modes / responses show no sigma0). 2D column order maps to fem rows by
  `_stress_rows()` (axisymmetric rr, zz, hoop, rz); plane stress refuses zz, generalized plane strain any. Tensor meshes
  refuse both (`prestress_issues(fem=False)`).
- stress_stiffening (both types; refused with piezo): hook `solve_static_state(with_loads)` keeps `_prestress` until
  the next `_assemble` / `materials_changed` / `inject_source`; `_pencil_stiffness(with_loads)` =
  `pattern_sum([K, K_sigma])`. with_loads True (eigen, optomechanics) = static solve of every load; False (frequency
  response: loads are the drive) = homogeneous solve of the initial-stress and thermal loads
  (`_total_load(dead_loads=False)`, stress sigma0 - beta dT + C eps(u): `_fixed_stress()`); none of them = K_sigma
  0. Bloch / PML / free body take sigma0 alone. The frequency response load is the dead loads only
  (`_external_load`). Studies call it via `study._solve_static_states` after their background stage. The hook ends
  with `system.release_factor()`: a kept static factor took the device memory the shift-invert factor then needs
  (CudssMemoryError, then a ~4.6 min SuperLU retry). Exception (tetrahedra, linear stress stiffening with loads,
  `_pencil_handover`: 'linear' back end = eigen factorization, cuDSS / MUMPS / PARDISO, no piezo, no loss): K_ff is
  factored as LDL^T (`DiffusionDirectSolver.positive_definite` False, the pencil's kind) and handed over
  (`DirectElasticSystem.hand_over_factor`, `_pencil_factor`) as the real ModalPencil's `factor`, so the first shift
  refactorises it on the kept analysis / plan (same free pattern: K + K_sigma on K's kept pattern) instead of a second
  one; the pencil releases it, `_drop_harmonic` releases an unused one. One factor at a time either way, so the memory
  phases ('static state', then the eigen ones) are unchanged.
- Eigen loss: mechanical_loss_factor -> K + K_sigma + j eta K_el (K_sigma lossless) in a damped ModalPencil;
  info['damped'] adds DAMPED_TABLE_COLUMNS; a damped 2D mode is shown like a response. Piezo modes ignore the loss.
- Complex records: frequency_probe (`definition_common.frequency_probe_feature`) -> HarmonicProbeRecord filled by
  `_record_harmonic`; supports `register_support` and record complex A x - f (2D x thickness); while a response is shown
  the support result is `frequency_response.reaction_table`. u_*_im fields (`requires_study` frequency_response, not
  in state()); tensor renderers raise for them (stored ones refused).

## Geometric nonlinearity, energy densities (tetrahedra, triangles; plan K6, K5)
- Settings `GEOMETRIC_NONLINEARITY_PARAMS` (shared 2D / 3D): every static solve is a total-Lagrangian Newton
  (`_nonlinear_state` via core.damped_iteration, proposer `tangent_update.newton_proposer`, tangent by
  `system.solve_tangent_reduced` on the fixed pattern of `TangentAssembler[3D]`, built lazily per assembly; the
  pencils' K_T too). Load steps ramp dead loads and prescribed displacements; S0 = sigma0 - beta dT is not ramped.
  Triangles: the K factor is refreshed after a tangent took it; plane stress = SVK with S_zz = 0; refused with m >= 1,
  PML, generalized plane strain, piezo; prescribed_displacement_domain is triangles only in 2D.
- tangent_update 'when_slow' (`TangentPolicy`, the default since round 5 of the 3D speed plan): reuse the held factor (`solve_last_tangent`) while the residual falls
  below SLOW_RESIDUAL_RATIO x the previous one. The held factor lives only inside one `_nonlinear_state` (`policy.reset()`
  there and before a warm start's cold ramp): any other factorisation on the 'linear' runtime replaces it, so never
  reuse it across solves. Time recipe keeps the full-Newton count. The factor / solve layer of both FEM systems is
  `system_fem.DirectElasticSystem` (K_ff on demand, `_tangent_held` guards `solve_last_tangent`, hooks `_check_static`
  / `_reaction_mask`; 3D drops its device solution vector on `release_factor`). Geometric nonlinearity (not piezo):
  `ElasticFEMSystem3D.assemble(defer_operator=True)` builds K at the first read of `operator` (pencils, frequency
  response, a linear solve), so a static Newton solve never assembles it; "assembled?" checks go through
  `_operator_missing()`, never `operator is None` (that read would assemble K).
- Device tangents (3D: gpu backend AND a cuDSS 'linear' back end): `TangentAssembler3D(device=True)`; held
  `policy.data` is device data, so code using the assembler's data goes through its methods (`full_product`,
  `solve_tangent_device`), never `assembler.full(data) @ host_vector`. Memory `_device_tangent_items` (one chunk of
  the assembler's byte-capped device size: fem sub file "Device tangents"). The K pattern of the tetrahedra
  (`assemble_elastic_stiffness_3d`) and the tangent's are one kept block pattern per mesh (fem sub file "Block
  patterns"): the host estimate counts it once, under the stiffness.
- 2D tangents stay host: planar `TangentAssembler` groups run `fem.tangent_kernel2d` (numba prange per element, the
  2D twin of tangent_kernel3d) once numba is in sys.modules or the map has >= NUMBA_MIN_ELEMENTS elements (a gpu run
  never pays the numba start for a small map); axisymmetric keeps the batched numpy formulas. Self-test
  solid_fem_tangent_kernel_numba (selftests_nonlinear.py, cpu) compares the twins.
- Newton start (`nonlinear_start.py`): sweep seed (`seed_warm_start`; 2D on full-mesh dof rows), else the last
  converged state of the same kind of this assembly (`forget_last` in `_assemble`), else u = 0. A failed warm start
  (incl. NaN / singular raise) reruns the cold ramp from 0 ('warm start failed: cold ramp'). `warm_start_fields` is {}
  without a converged nonlinear state. Time recipe keeps the cold count.
- Reactions = `system.residual` (internal force), never K u - f. Outputs after a nonlinear solve: Green-Lagrange
  strain and second Piola-Kirchhoff stress (not Cauchy). `prestressed` is true with either setting; `_pencil_stiffness`
  returns K_T of the nonlinear state (frequency response: about the state of sigma0 - beta dT, no loads).
- prescribed_displacement_domain on every node of a 3D selection locks cross-section rotation (not the shell no-slip
  variant); a deflection hold on a whole end face of a thin structure adds a spurious quartic energy (Duffing alpha + 2
  .. 5 %): the Duffing self-tests hold mid-plane / mid-line nodes through the API and take k from a linear-load solve
  (fitting k too trades ~1 % against alpha).
- strain / kinetic energy densities: static 1/2 eps_el : C : eps_el and 0; modes / responses time averages; tensor
  meshes show 0 and refuse probes.
- Initial-stress balance: finalize build warning (validation has no mesh) when sigma0 is not self-equilibrated and no
  static solve relaxes it (Bloch, PML, free body).

## Thermoelastic damping (`thermoelastic.py`, feature thermoelastic_damping; definition_common)
- Post-processing of the last `eigen_solve` (both modal studies), evaluated on the first result read and kept per
  solve object (identity): solid_mechanics tensor (own Q1 2 x 2 Gauss scalar assembly, not the elastic stencils) and
  triangles, solid_mechanics_3d tetrahedra (a tensor 3D mesh raises: matrix-free modes). Formula, sign convention and
  discretisation in the module header; the help text is the user contract (keep both in step).
- Needs `modal_mass` of the same solve (stored energy = omega^2 Re(u^H M u), never K: complex PML / loss pencils) and
  the solver's beta (`_thermal_coefficients` / `_thermal_stress` of the last assembly, plane stress reduced): a new
  solid variant listing the modal studies keeps both, or the feature's validation must refuse it. Refused: piezo,
  guided, azimuthal_order >= 1, Bloch cells; PML ring / shell elements are left out of the thermal problem.
- One host SuperLU per mode through a private `SparseDirectSolver('scipy_splu', 'cpu')` (no solver role: a feature
  evaluation like the squeeze films; not in work_counters); host memory booked by the feature's memory_estimate.
- q_other: a damped pencil's Q (PML + loss factor on element meshes) or, on the tensor mesh (undamped pencil), the
  perturbative eta from `_harmonic_operators[1]`; tetrahedra add the squeeze films' gas Q (never both for one loss).
- Materials: k / k_xx / k_yy / k_zz / cp are solid Params mirroring heat's (kind, unit, bounds, flags: the registry
  check), arrays `OPTIONAL_TED_SCHEMA` (feature-gated entries: allocated only with an enabled thermoelastic_damping
  feature and a material setting one of them, so plain solids of library silicon / silica carry none; a heat physics
  allocates them anyway). Code reading them outside the feature must not assume they exist.
- Accuracy (selftests_thermoelastic.py vs Lifshitz-Roukes): planar tensor 16 cells over h 0.4 % (8 cells: 1 % off the
  Debye peak), tensor axisymmetric slower (+5 / +1.4 / +0.45 % at 4 / 8 / 16 cells over h), P2 triangles / tetrahedra
  ~0.3 % at 4 / 2 elements over h; slender-beam theory itself is (h / L)^2 off (Timoshenko shear energy, ~0.5 % at L /
  h = 20).

## Reused sweep builds (SweepSpec.reuse_build; model sub file)
- Flagged values (`definition_common.following`): body_force f_*, boundary_load / segment_load t_*, displacement /
  prescribed_displacement_domain u_*. Applies add the build value, then `follow_load` (adds the change; tractions
  merged into one entry) or `follow_held_values` (`set_held_values` on the hold's own mask, unchanged components None;
  no re-assembly). Only the FEM solids are RerunSolvers (`reset_records`: records cleared, prestress dropped, loads
  dirty; the last Newton state stays the start); tensor solids register no follower, so such a model is rebuilt per
  value. A follower's held value wins over a later overlapping hold of another feature.

## Couplings and optics hooks
- heat_solid_thermal_strain (definition.py; targets both types, `writes` {'temperature_rise': ''}):
  `heat.temperature_rise()` into one full-grid buffer (PML ring 0); heat T0 = stress-free temperature. A heat without
  `window` (FEM) gives one value per element: copied whole.
- solid_em_strain_optic (`strain_optic.py`, E13 'index' contributor per index slot of any family-'em' target): dn =
  (n0 + dn_pe) sqrt(1 + w (strain_xx + strain_yy)) - n0 of the ELASTIC strain (strain - alpha dT, e_zz = -alpha_zz dT:
  the measured dndT already holds the free expansion). No p44 / strain_xy term. With deformation on,
  `coupling.definitions.solid_supplies_deformation` makes heat_em_thermo_optic on the same EM physics drop its (1 +
  alpha dT): never both images.
- em_solid_optical_force (`optical_force.py`, em_fdfd_ez only): face form of -(eps0/4)|E_z|^2 grad eps_r split between
  solid cells, 'sum' into body_force_x / _y (CouplingGroup with magnetic_solid_lorentz_force, buffers refreshed per EM
  n_solves). hz and absorption / pole forces not modelled. Mean |E_z|^2 of two cells (not the conserved Re(E_i
  conj(E_i+1))): O((n k0 dx)^2 / 2) error per interface.
- `material_params.py`: Params used verbatim by elastic_waves (edit the help there). `strain_optic_index_shift` is
  public (acousto-optic reuses it); `photoelastic_tensor_index_shift` / `PHOTOELASTIC_TENSOR_ROWS` and
  `photoelastic_shear_permittivity_shift` / `PHOTOELASTIC_SHEAR_ROW` serve only acoustics: the solid coupling stays
  isotropic and writes no eps_xy.

## Memory estimates (definition.py, definition3d.py; model sub file "Memory estimate parts")
- Phases (transient items, the largest one counts): 'static solve' (tensor solve temporaries), 'stiffness
  re-assembly', 'static factorisation' (FEM: the values of a refactorisation on the kept plan, booked with
  `refreshed`), 'static state'
  (everything of the 'linear' factor in a modal / response study: `solve_static_state` ends with `release_factor`;
  tetrahedra book it for stress stiffening too, definition3d `_static_state_solved`, memory and time). Both come from
  `linear_factor_items`: `DirectElasticSystem._solve_free` (K_ff by `fem.fixed_pattern.free_restriction`, kept per
  pattern, so `ModalPencil._reduced` of K + K_sigma on that pattern takes values only) plans a cuDSS K_ff factor from the HOST CSR
  (`factorize_host_operator(symmetric_known=True)`: equilibration and lower take on the host; MUMPS / PARDISO take the
  same call, so K_ff skips the host asymmetry check there too), so the pool holds
  `operator_copy_bytes` (lower-triangle CSR, diagonal; mixed: + the float32 copy) + rhs buffer / row scale + a
  solve's vectors, and no device CSR of K_ff, index maps or symmetry check (they cost ~60 B per K_ff entry and set
  the pool peak of a prestressed modal study). `refreshed` (2D host Newton tangents on cuDSS: every later
  `refactorize` on the planned pattern is the device value refresh) adds the index maps and the refresh values; a
  re-assembled K_ff on an unchanged free set refreshes the same way and is NOT booked (no setting tells it),
  'eigen assembly' (tensor mass / guided temporaries before the host copy), the ModalPencil phases (released after
  each solve / search) in the order `arpack_shift_invert` allocates: 'eigen pencil set-up' (`EIGEN_SETUP_PHASE`:
  stored terms; the upload transients only of a pencil that is not vouched symmetric: a vouched one takes its lower
  triangle on the host), 'eigen factorisation' (`EIGEN_FACTOR_PHASE`, eigs_shifted only: the factor is planned and
  probed before `device_mass()` and the Arnoldi basis exist), 'eigen solve'
  (`EIGEN_PHASE`: the stored terms, the factor's index maps (a lower-stored pencil: the lower triangle's alone,
  `index_map_bytes(lower_only=True)`) and equilibrated values, the combined values of the
  factorisation (eigsh: M is uploaded before it), the mass product = the host M on its own pattern (`mass_entries`:
  entries / components for a vector Lagrange mass; the callers pass it), vectors
  and, on the eigs path, the device Krylov-Schur basis); the eigen factor is phase-bound to the last two. Pencil
  items are counted array by array at the stored layout ((entries + dofs) / 2 values per term when lower-stored,
  value size of the pencil), not through `sparse_direct.device_pencil_bytes` (sized for a complex full-pattern
  refresh: 2.5x the arrays of a real lower-stored pencil),
  'Newton tangent assembly' (device chunk temporaries), 'frequency response' (tensor device temporaries). Kept: the
  static factor's copies / vectors / solution while the assembly holds, the frequency-response HarmonicSystem
  (`harmonic_pencil_items`), device tangent caches, generalized-plane-strain unit solves.
- Shared helpers in definition.py (definition3d imports them inside its estimator): `linear_factor_items` (FEM
  'linear' factor, kept or 'static state'), `eigen_basis_counts` (n_wanted, basis: a PML doubles the asked modes),
  `eigen_device_items` (-> transient items by phase, host items, factor tuples, {factor: phases}, basis_on_device:
  the caller's `_eigen_host_items` then books only the eigenvectors on the host; complex_eigs = the eigs path: PML,
  loss, loaded electrodes; a real or Hermitian Bloch pencil is eigsh with a host Lanczos basis),
  `harmonic_pencil_items`. Every cuDSS factor is an external item through `definition_helpers.cudss_external`
  (dimensions 2 on triangles / tensor 2D, 3 on tetrahedra; on a built element mesh `factor_fill` gives the counted
  `FactorFill` of `mesh.dissection` instead: node pairs at the element order x components^2 x the factored share); in an eigenfrequency study the static factor is
  phase-bound to 'static state' (`LINEAR_STATIC_STATE_PHASE`, released before the pencil's) and the eigen factor to
  its phases; a prestressed frequency response still books one_at_a_time. A factor refreshed from a DevicePencil gets no CSR copies on top
  (its maps / transients hold the planned values: membrane_guide_pml and pressure_acoustics_rigid_disk measured);
  a factor of a host CSR (below DEVICE_PENCIL_MIN_ENTRIES, the static K_ff) does (`operator_copy_bytes`). S5c: a
  cudss_mixed eigen pencil below MIXED_EIGEN_MIN_UNKNOWNS factors in double on the host route (no DevicePencil):
  solid_fem3d_pml_bar_radiation (gpu, 6e3 unknowns) measured 5 MB pool where the DevicePencil items gave 84 MB.
- Host factors of a MUMPS / PARDISO back end / factorization: `definition3d.host_factor_items(name, superlu_label,
  superlu_bytes, n_unknowns, entries, symmetric=, complex_values=)` (definition imports it at module level, as
  `eigen_host_direct`) scales a SuperLU heuristic item by `host_direct.superlu_bytes_scale`, relabels it and adds the
  factor object's own copies (`host_direct.solver_host_bytes`, as `fem_system_estimate`; uncalibrated; any other
  name keeps the one SuperLU item; `fill=` the counted `FactorFill` of `factor_fill`: tetrahedra book the counted
  triangle instead, as cuDSS: the 3D n^(4/3) heuristic booked 33 GiB for the 2-layer film sin_square_membrane_modes_3d):
  static K / Newton tangent / piezo coupled operator and the frequency-response
  pencil (LDL^T), the eigen factor (`eigen_host_direct`: S5c, a small *_mixed pencil factors in double; a Bloch
  phase: LU; the 2D caller doubles every item of a complex pencil, copies included). Size rules keep the SPLU / SPLU_3D limits
  (uncalibrated) with the library named; time recipes price them as SuperLU (`backend` 'cpu').
- Unverified against counters (no gpu run in selftest_out): the static 'linear' cuDSS items, the solid
  frequency-response HarmonicSystem, solid DevicePencils, device Newton tangents.

## Self-tests (`selftests.py`)
- Shared case helpers (reuse, never a local copy): `selftests.py` (`_lame`, `_elastic`, `_rectangle`, `_reaction`,
  `_window_fields`, the bilayer strip), `selftests_3d.py` (`_box`, 3D `_reaction`), `selftests_fem3d.py` (`_solid_spec`,
  `_probe`, `_probes`, `_point`), `selftests_eigenfrequency.py` (`_symbols`, `_table_frequencies`, `_add_rank_items`:
  rank-order table frequencies and snapshot counts), `selftests_formulations.py` (GPS layers: `_gps_materials`,
  `_layer_constants`), `selftests_nonlinear.py` (string warm start / tangent update twins). Template modules
  (`selftests.py`, `selftests_3d.py`, `selftests_bloch3d.py`, `selftests_optomechanics.py`,
  `selftests_thermoelastic.py`) import no non-template sibling at module level (get_registry loads them).
- Nonlinear: `selftests_nonlinear3d.py` (definition3d loader; solid_fem3d_newton_gpu needs cuDSS), `selftests_nonlinear.py`
  (2D; its helpers serve the 3D twins), sweep_nonlinear_string_warm_start (model/selftests_sweep).
- Exact-in-Q1 cases (1e-6, float64 CG to 1e-12): solid_bilayer_tension_graded (variants 'mixed' / 'mixed_cpu': pcg
  mixed on both backends), solid_simple_shear (the only gamma != 0 case: pins c66). Heat coupling, body force, traction and reactions are covered by examples. solid_em_strain_optic_plate
  and em_solid_optical_force_slab (`selftests_optical_force.py`) are registered only inside solid_optics_cpu_gpu.
- `selftests_formulations.py` (reuses selftests.py helpers): plane stress, generalized plane strain (+ bending),
  axisymmetric cases.
- `selftests_3d.py` (exact trilinear, 1e-6): solid_3d_column_z_expressions expects the discrete solution with the
  solver's LUMPED cell loads f_c V / 8 (not the consistent load for non-constant f). Variant 'mixed' on
  solid_3d_uniaxial_block_graded, solid_3d_column_z_expressions, solid_3d_eigenfrequency_roller_box;
  solid_3d_eigenfrequency_block_eigen (`selftests_eigenfrequency.py`) is the same box by a block_eigen eigen node
  (variant 'double' on the eigen role; its cpu run: the wrapper solid_mechanics_3d_block_eigen_cpu of
  `selftests_backend.py`, a case of its own so the cpu block callbacks time apart); that box is below
  multigrid3d.COARSEST_DOFS (one level: the dense coarsest inverse only), so solid_3d_eigenfrequency_block_eigen_levels
  runs the block on half-size cells (two levels: the full block V-cycle) against the Lanczos run of the same model.
  Examples
  thermal_inclusion_sphere_3d (mean-stress identities exact: centre strains x V are the element integrals),
  rotating_disk_3d.
