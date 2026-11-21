# open_sol/physics/mechanics — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Solid mechanics (C3).

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

## Backends (gpu and cpu)
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
  pcg, 'mixed' on the gpu, sparse_direct as the stationary / time_dependent default); the 2D binding (inverse node
  blocks, workspace, CUDA-graph check blocks, float32 copies of 'mixed', or the vouched-symmetric direct factor) is made
  by the first `solve` / `solve_homogeneous` after an assembly (`ElasticSystem._bind_if_stale`), so modal,
  frequency-response and piezo runs hold none of it (the estimate's `pcg_bound`), and a re-assembly releases the old
  A_solve and binding first (`_release_binding`): a new reader of the binding goes through `_bind_if_stale`.
- Reactions are K u - f of the last solve (`load_full` of the last `_build_rhs`): every load change sets
  `_loads_dirty` (it also drops the frequency response's kept `_harmonic_drive`), so loads never change between a solve
  and a read. n_solves doubles as the couplings' revision counter.
- 2D ElasticSystem keeps no full K on the device, only A_solve and the float64 K rows of the held dofs
  (`_held_row_block`, the reactions' rows; `reaction` refuses a mask over free dofs): a new reader of K rows or K u
  extends that block or rebuilds from the stencil, never re-keeps the full operator (one value array per entry).

## Mirror planes (feature symmetry, both types)
- `symmetry` (`definition_common.SYMMETRY_FEATURE`) = the Roller apply / result whose faces are model mirror planes
  (s = +1). It is in `SUPPORT_FEATURES` and `ROLLER_FEATURES`: code testing `feature.type == 'roller'` for the normal
  hold tests ROLLER_FEATURES instead.
- Validation (`definition_common.symmetry_issues` + `pml_crossing_issues`, a PML on a face of another axis crosses the
  plane): errors for piezoelectric, materials failing `mirror_violation`, uniform shear initial stress with a normal
  index; 2D generalized plane strain bending terms across the plane, 3D optomechanical_coupling and a squeeze film gap
  side on the plane. Physics level: `own_mirror_face_conflicts`, `model_mirror_issues` (warning per model mirror plane
  no Roller / Symmetry / Fixed holds).
- Results the solid reports itself stay those of the drawn part (support / harmonic reaction tables, squeeze-film c /
  k); the optomechanics m_eff is the unfolded body's (a solid symmetry face forces the optics to mirror it; em_fdfd_bor
  optics: `OptomechanicalCoupling.mirror_factor` from `mirror_faces(ctx.spec)`). Parity sectors raise (no odd solid
  feature). A vented squeeze-film rim on a plane raises in `_vent_edges`.
- Axis: `AXIS_UNFOLD_TYPES['solid_mechanics']` +1 is right for every renderer field at azimuthal_order 0.

## Anisotropic stiffness (`stiffness.py`, c_ij entries)
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

## Piezoelectric plane strain (`piezo.py`, setting piezoelectric)
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
- Real stays real: HarmonicSystem keeps K real without a loss term, so a lossless pencil without a resistive load
  factors real (half the factor, the complex rhs split by solve_host / solve): a new per-frequency term adds out of
  place (never `+=` a complex term into the real values), and the estimators book complex terms / factor only with
  `LOSS_ARRAYS` set or a resistive_load (definition3d `_lossy_response`). Its pencil terms are vouched symmetric
  (`symmetric_known`): a new unsymmetric term must drop the vouch.
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
- One machinery with the tetrahedra cells: a 2D cell is a `bloch3d.BlochCell3D` with z never periodic (convention and
  rigid-motion rules: bloch2d / bloch3d headers). Every interleaved layout (plane strain 2, guided 3, piezo 3) takes
  `vector_masters(components)` and P = `fem.dofs.bloch_prolongation`. Tensor cells are built once in prepare
  (`tensor_bloch_cell`); triangles rebuild `bloch` on the solid map at every assembly (`ElasticFEMSystem.periodic`).
- Which studies / settings run: periodic_x help text and `bloch_issues_2d`. Why the gaps stay: tensor statics (the PCG
  has no P), frequency_response (HarmonicSystem's output rows assume no column map; the admittance of an infinite IDT
  would need P through them), prestress / generalized plane strain / axisymmetric (no periodic prestate /
  superposition / axis rule).
- Triangles + an elastic PML on the other axis (surface-wave cell): a complex P reduction of the damped (complex
  symmetric) pencil is neither complex symmetric nor Hermitian, so `ModalPencil._pencil` asks a general ShiftPencil and
  `eigs_shifted` factors it by LU.
- ARPACK can return only some copies of an exactly degenerate band (the Q1 square cell's 4-fold c_T 2 pi / a band at
  Gamma came back 3-fold with n = 4; guided at k_z = 0 its 8-fold band 4-fold even with n = 10): cases ask a few more
  modes than they grade, or grade at a k off the symmetry point.

## Eigenfrequency study, 3D tensor meshes (`ElasticSystem3D.lowest_modes`; tetrahedra: section "Tetrahedron meshes")
- LOWEST n modes only (`eigen_search_issues(tensor_mesh_3d=True)`): there is no CSR in 3D (a 3D elastic SuperLU fills
  host memory at ~30-40^3 cells), and K - sigma M is indefinite above omega_1^2, which neither the inner CG of the
  shift-invert Lanczos at sigma = 0 nor LOBPCG (block_eigen) handles. Interior targets would need MINRES / a direct
  factor, free-free a negative shift + rigid filter.
- block_eigen (opt-in, tensor grids only): `ElasticSystem3D._lowest_modes_block`. Each block callback is ONE launch
  for all rows: the kernels' `_rows` entries share the one-vector __device__ body, while `multigrid3d.BlockCycle`
  repeats `BrickMultigrid._cycle` / `_smooth` / transfers on rows (change them together). Estimators switch by
  `definition3d._block_eigen_values(context)`.
- Exact roller-box modes need pairwise coprime cell counts per axis (9 x 5 x 4: equal phases on two axes give
  degenerate pairs); odd N_x has no node at the mode-1 sine peak (expected shape divides by max_i sin(pi i / N_x)).

## Optomechanical g0 (`optomechanics.py`, `optomechanics_fem.py`)
- Conventions, face rule, overlaps (ez / hz / cross-section / backward / crystal), normalisation, m_eff, Brillouin gain
  and the subpixel staircase: the two module headers; the solid hands its data only through `modal_overlap_data()`.
- om_integrand (`requires_study='optomechanics'`) is not in `state_fields` (`state()` has no such array): a new
  study-only renderer field must stay out too. Tensor meshes only: `SolidFEMRenderer` drops it from its `field_infos`,
  since default snapshots store every listed field (`_stored_field_keys`) and a listed field the renderer cannot read
  fails the study.
- Case gotcha: a sin(pi j / N) roller mode is even about the centre, its S_yy odd: zero overlap with TE0
  (om_forward_brillouin_gain uses sin(2 pi j / N)).

## Squeeze film (`squeeze_film.py` = the tet runtime; the film itself is `fem/squeeze_film.py`, fem sub file)
- Films are evaluated in the `_modal_pencil` finish (on the merged list after a search), so every modal path of the
  tetrahedra keeps that finish; they are not coupled into frequency_response (validation warning).

## SolidMechanics3DPhysics / ElasticSystem3D (solid_mechanics_3d)
- Matrix-free (no CSR): `kernels3d.BrickKernels` products, every CG step a `FusedBrickIteration` and the gpu check
  blocks CUDA graphs; captured arrays and the reaction scratch: `ElasticSystem3D` comments (`_fused_operator`,
  `_reaction_key`). CG iterations scale with the longest edge: keep 3D self-test strips short.
- ElasticSystem3D keeps no u_fix / K u_fix vectors: `held_values` exists only once a hold writes a value other than a
  scalar 0 (`_held_values_for`; None = every held value 0), K u_fix is one product per `build_rhs` on the CG unknown
  (`_solution`, which also resets `_reaction_key`), and `solve` rewrites the fixed dofs of unknown
  (`_write_fixed_values`). `applied_body_force` / `applied_load` likewise wait for the first body force
  (`body_force_target`) / traction (the load kernel's has_applied_* flags skip them). A new
  writer of held values goes through `_held_values_for`; the estimate books both by feature (`_PRESCRIBING_FEATURES`,
  body_force). The reduced inverse node blocks are float32 at every solve precision (`kernels3d.PRECONDITIONER_DTYPE`,
  its error argument at the constant): formed / inverted in the solve dtype in a transient, every kernel reading them
  takes `const float*` (a new one too; block_precondition3 takes its real type from the residual, `_DTYPE_ARGUMENTS`),
  and the mixed / block-eigen float32 operator sets share the array.
- Outputs are computed on read (`SolidMechanics3DPhysics.output`; state() is a Mapping, probes and the renderer read
  through it; MechanicsFields3D holds the inputs only): `_write_outputs` only records the shown solution
  (`_ShownSolution`: the live unknown, the assembly's lambda / mu, and once a temperature input was written its free
  thermal strains alpha_i dT, so later in-place changes of dT / alphas never reach the shown outputs). Code that
  changes the unknown without a new show calls `_freeze_unknown()` first (assembly, warm start) or `_drop_shown()`
  (a solve, a selected mode); a new temperature input writer sets `_thermal_inputs`. One computed output is kept (the
  latest); a reader of several keeps its own references. The elastic_output kernel and its numpy twin change together.
- pcg preconditioner 'multigrid' (opt-in): design and known weakness in the `multigrid3d.py` header, the fallback guard
  in `_guarded_solve`. Self-tests `solid_3d_multigrid` and `solid_3d_multigrid_void` fail on a fallback, so the guard
  cannot hide a weak cycle.
- No strain-optic / optical-force coupling in 3D: em_fdtd_3d's uint16 material index cannot hold three continuous
  permittivity components.

## Triangle meshes (`solver_fem.py`, `system_fem.py`, fem/elasticity.py)
- solid_mechanics dispatches on the mesh kind (`_create_solid` -> SolidMechanicsFEMPhysics); shared with tetrahedra:
  `solver_common.FEMSolidCommon` / `HostInputCache` (a new mutation of alpha / dT / sigma0 / beta / rho calls
  `_drop_host_inputs`). What runs and what is refused: the type help and `_triangle_issues`.
- Why the gaps stay open: piezo needs phi on every window element + G on a DofMap (~a piezo_fem.py, would unlock
  admittance); guided modes need a 3-component guided stiffness; the planar optomechanics overlap is written on Yee
  faces (needs em_modes on triangles); solid_em_strain_optic has no mesh_transfer (the optical force reaches a
  triangle solid through em_solid_optical_force's mesh_transfer).
- cuDSS verdict: static K gains little (2D SuperLU is fast to ~1e6 dofs); eigen (indefinite above omega_1: LDL^T) runs
  on cuDSS double / mixed; frequency response is the best candidate (complex symmetric, one pattern, 100-600
  frequencies: plan reuse), double (near a resonance a float32 factor needs many refinement steps).
- solid_fem_kirsch sets h_min: the default dx / 100 floored the hole sizes (1 % low stresses).

## Tetrahedron meshes (solid_mechanics_3d; `solver_fem3d.py`, `system_fem3d.py`, fem/elasticity3d.py)
- `_create_solid_3d` builds SolidMechanicsFEMPhysics3D on 'tetrahedra' (every feature declares both grids); what runs
  only there: the type description and validation. Conventions per topic: the headers of `solver_fem3d`, `stiffness3d`,
  `piezo_fem3d` / `piezo`, `bloch3d`, `pml3d`, `mode_shapes`, `frequency_response`, `modal`.
- Outputs are lazy (`_OUTPUT_GROUPS`, both FEM solids): `_write_outputs` only marks the groups pending. Code that
  changes unknown / potential / stiffness / inputs without a new `_write_outputs` calls `_ensure_outputs()` first, and
  a new reader of `_nodal` or the per-element outputs goes through it.
- The point groups share `_point_values` (read-only: copy before subtracting alpha dT; dropped by
  `_release_point_values`): a new output at element points takes its strains from there (rows of a solid subset by
  searchsorted), never a fresh `element_strains_3d`. A new per-solve value an output group reads joins
  `_shown_solution` (snapshot solutions: `FEMSolidCommon.solution_state`).
- Piezoelectric: K_uu is the system's CSR (`system.assemble` still builds holds / active set / rigid check; its K
  factor is never made). Electric outputs are allocated per element on tetrahedra even without piezo (0); the tensor
  solid keeps them out of state_fields (its renderer returns zeros).
- Not built: PML with piezo (G and its scaling are real) or in frequency response; static periodic solves.
- mode_shapes (a GUI Compute adds one in the modal studies: model sub file "GUI run features") has no memory estimate:
  r x n_surface x 24 B per value, a fraction of the eigenvectors the eigen items count (a sweep keeps one per value in
  results.npz). A new derived quantity belongs on the page (viz mode_player), a new linear field (e.g. E / D) in
  `mode_surface_tensors`. solid_fem3d_bloch_mode_shapes pins the phase convention (the opposite sign fails by O(1)).
- Free bodies (`definition_common.free_body_allowed`, also triangles): `rigid_motions_allowed` counts the rigid motions
  (`n_rigid_motions`) instead of refusing; static solves raise while any is free, the frequency response does not;
  mode solvers ask them on top of n + EXTRA_EIGENVALUES and drop them below RIGID_EIGENVALUE_FRACTION. Tensor meshes
  refuse free bodies.

## Prestress, damped modes, complex records (triangles and tetrahedra only)
- initial_stress_* (`materials.PRESTRESS_ARRAYS` / `_3D`, each allocated when set): static outputs add sigma0
  (`_output_prestress`), modes and responses show none; 2D columns map to fem rows by `_stress_rows()`.
- stress_stiffening / geometric_nonlinearity: the studies call the PrestressSolver hook `solve_static_state(with_loads)`
  after their background stage (`study._solve_static_states`); the kept `_prestress` enters every pencil through
  `_pencil_stiffness` (a new pencil path reads it there) until the next `_assemble` / `materials_changed` /
  `inject_source`. The hook ends by releasing the static factor or (tetrahedra, `_pencil_handover`) handing it to the
  modal pencil: one factor at a time, which the memory phases ('static state', then the eigen ones) assume.
- Complex records: frequency_probe -> `HarmonicProbeRecord` (`_record_harmonic`); while a response is shown a support's
  result is `frequency_response.reaction_table`. u_*_im fields are `requires_study` frequency_response, not in state().

## Geometric nonlinearity (tetrahedra, triangles)
- The 'when_slow' held tangent factor (`TangentPolicy`) lives only inside one `_nonlinear_state` (reset there and
  before a cold ramp): any other factorisation on the 'linear' runtime replaces it, so never reuse it across solves.
- Factor / solve layer of both FEM systems: `system_fem.DirectElasticSystem`. 3D geometric nonlinearity (not piezo)
  defers K (`assemble(defer_operator=True)`: built at the first read of `operator`): "assembled?" checks go through
  `_operator_missing()`, never `operator is None` (that read assembles K).
- Device tangents (3D: gpu backend and a cuDSS 'linear' back end): held `policy.data` is device data, so code using the
  assembler's data goes through its methods (`full_product`, `solve_tangent_device`), never
  `assembler.full(data) @ host_vector`. Memory and the shared K / tangent block pattern (counted once, under the
  stiffness): fem sub file "Device tangents", "Block patterns". `newton_proposer` drops the full tangent data before
  each factorisation (only the free-free copy reaches the factor) and modified Newton holds only the host held-dof
  rows (`tangent_update.HeldRows`), dropped as soon as an iteration refactorises: the estimate books the full values
  in the 'Newton tangent assembly' phase. The reused factor's lift is rows^T correction, exact only for a symmetric
  tangent: a non-symmetric term (follower loads) must keep the held columns instead.
- Case gotcha: prescribed_displacement_domain on every node of a 3D selection locks cross-section rotation; a deflection
  hold on a whole end face of a thin structure adds a spurious quartic energy (Duffing alpha + 2 .. 5 %): the Duffing
  self-tests hold mid-plane / mid-line nodes through the API and take k from a linear-load solve (fitting k too trades
  ~1 % against alpha).

## Thermoelastic damping (`thermoelastic.py`, feature thermoelastic_damping; definition_common)
- Formula, sign convention and discretisation: module header; the help text is the user contract (keep both in step).
- It reads `modal_mass` of the same solve (stored energy omega^2 Re(u^H M u), never K: complex PML / loss pencils) and
  the solver's beta of the last assembly (`_thermal_coefficients` / `_thermal_stress`): a new solid variant listing
  the modal studies keeps both, or the feature's validation refuses it.
- One private host `SparseDirectSolver('scipy_splu', 'cpu')` per mode, like the squeeze films: no solver role, not in
  work_counters, host memory in the feature's memory_estimate.
- k / k_xx / k_yy / k_zz / cp are allocated only with an enabled feature (or a heat physics): code reading them
  outside the feature must not assume they exist.
- Accuracy vs Lifshitz-Roukes (selftests_thermoelastic.py): planar tensor 0.4 % at 16 cells over h (8: 1 % off the
  Debye peak), tensor axisymmetric +5 / +1.4 / +0.45 % at 4 / 8 / 16 cells, P2 triangles / tetrahedra ~0.3 % at 4 / 2
  elements over h; slender-beam theory itself is (h / L)^2 off (~0.5 % at L / h = 20).

## Reused sweep builds (SweepSpec.reuse_build; model sub file)
- Followed values: `definition_common.FOLLOWING_LOAD_NAMES` through `following`; their applies add the build value and
  wire `follow_load` / `follow_held_values`. Only the FEM solids are RerunSolvers (`reset_records`), so a tensor solid
  is rebuilt per value. A follower's changed held value wins over a later overlapping hold of another feature.

## Couplings and optics hooks
- heat_solid_thermal_strain, solid_em_strain_optic and em_solid_optical_force: formulas, writers and limits in their
  CouplingType descriptions (definition.py).
- `material_params.py`: Params used verbatim by elastic_waves (edit the help there). `strain_optic_index_shift` is
  public (acousto-optic reuses it); `photoelastic_tensor_index_shift` / `photoelastic_shear_permittivity_shift` and
  their rows serve only acoustics: the solid coupling stays isotropic and writes no eps_xy.

## Memory estimates (definition.py, definition3d.py; model sub file "Memory estimate parts")
- Every FEM solid estimator books its factors through the shared helpers, never items of its own: definition.py
  `linear_factor_items` (the 'linear' K_ff / tangent factor, kept or phase 'static state'), `eigen_basis_counts`,
  `eigen_device_items` (the ModalPencil phases in the order arpack_shift_invert allocates), `harmonic_pencil_items`,
  `factor_fill`; definition3d `host_factor_items` / `eigen_host_direct` (MUMPS / PARDISO). What each item holds and
  why: their comments. definition imports definition3d at module level, so definition3d imports these inside its
  estimators. Term dtypes follow the code: ShiftPencil keeps a real M float64 in a complex pencil (`mass_bytes`: 16
  only with a PML stretch or a Bloch phase), its device M is `device_matrix_bytes`.
- Unverified against counters (no gpu run in selftest_out): the static 'linear' cuDSS items, the solid
  frequency-response HarmonicSystem, solid DevicePencils, device Newton tangents.

## Self-tests (`selftests.py`)
- Shared case helpers live in selftests.py, selftests_3d.py, selftests_fem3d.py, selftests_eigenfrequency.py,
  selftests_formulations.py and selftests_nonlinear.py (the 3D nonlinear twins reuse its helpers): reuse, never a local
  copy. Case notes are in the case descriptions.
- Template modules (mechanics entries of `TEMPLATE_SOURCES`, model/import_contracts.py) import no non-template sibling
  at module level: get_registry loads them, and contract selftests_lazy does not follow their imports.
