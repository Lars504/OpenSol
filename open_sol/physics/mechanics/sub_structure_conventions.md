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
  mesh kinds: section "Bloch cells, 2D"). Both FEM types: the shear_modulus / bulk_modulus route, viscoelastic moduli
  and the setting formulation (sections "Viscoelastic moduli ..." and "Mixed u-p formulation").
  `solid_mechanics_3d` (3D only, separate type like em_fdtd_3d; default 'solid'; same material Params incl. rho; tensor
  meshes: lowest modes only; tetrahedra: target_frequency / search targets / band, 21 c_IJ, piezoelectric (charges in
  C, static on 'linear' sparse_direct, short / open modes on 'eigen'), frequency_response on 'linear').
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
  pcg, 'mixed' on the gpu, sparse_direct as the stationary / time_dependent default); the 2D binding (CUDA-graph check
  blocks, float32 copies of 'mixed', the factor deferred to the next `solve` so modal and piezo solves never factorise
  it): `ElasticSystem._bind_linear_solver`.
- Reactions are K u - f of the last solve (`load_full` of the last `_build_rhs`): every load change sets
  `_loads_dirty` (it also drops the frequency response's kept `_harmonic_load`), so loads never change between a solve
  and a read. n_solves doubles as the couplings' revision counter.

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
  k); the optomechanics m_eff and the brillouin_scattering psd are the unfolded body's (a solid symmetry face forces
  the optics to mirror it; em_fdfd_bor optics: `OptomechanicalCoupling.mirror_factor` from `mirror_faces(ctx.spec)`;
  BrillouinDrive's mirror_factor likewise). Parity sectors raise (no odd solid feature). A vented squeeze-film rim on
  a plane raises in `_vent_edges`.
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
  (counters harmonic_* / piezo_harmonic_*; the 'linear' pcg assembled, unused): elastic ones on one HarmonicSystem of
  runtime None kept while the harmonic operators stay (`_kept_harmonic`, direct and reduced sweeps, with or without
  physics drives: section "Spontaneous Brillouin"), piezoelectric ones on the PiezoSystem's; triangles and tetrahedra
  their 'linear' sparse_direct runtime (3D operator class 'spd' is nominal there).
- Shown fields are the real parts with dT = 0 and no sigma0 (`_write_outputs(thermal=False)` on every mesh kind;
  solid_frequency_response_bar's 'heated' run pins it on the tensor mesh). On the tensor mesh support reactions and
  electrode tables raise and probes record nothing (FEM meshes: section "Prestress, damped modes, complex records").
- Reduced sweeps (fem sub file "Reduced sweeps") and rows-only direct solves: after `HarmonicSystem.set_output_rows`
  `solve(fields=False)` is exact ONLY at the registered rows, so a solver's records read only rows it registered
  (supports, probes, `PiezoSystem.electrode_rows`). Fields only at `wants_fields` frequencies; a skipped last one is
  shown by `publish_frequency_response`. Readouts of a whole-field weight (physics drives) are HarmonicSystem
  functionals instead (`set_functionals`: `last_functionals` = W^H x on every solve route, reduced models included;
  `solve_columns` for more right-hand sides on the last direct factor), never rows of x.
- azimuthal_order m >= 1 (triangles) is driven by prescribed displacements or a physics drive only (cos(m phi) loads
  are refused); a support's F_x is sum R_r - sum R_phi at m = 1 (the net force along phi = 0), both components 0 at
  m >= 2; its W = sum R u_held takes every component. Triangles: record rows and the residual are solid-map rows, `held` /
  `held_values` full-mesh ids (index them by `dofs.global_ids[rows]`); tetrahedra: both full-mesh.
- A reaction of a fully held stiff domain in a soft / viscous medium carries its own K rows' round-off (a rigid
  translation cancels in K X only to ~1e-16 E per entry): a polystyrene bead in water at 1 Hz has Re F ~1e-4 of the
  drag (E / (omega mu) ~ 6e11), above its physical inertia; grade only the dominant part (bead_wall_brenner example).

## Viscoelastic moduli and the N-term harmonic system (`viscoelastic.py`, `definition_viscoelastic.py`; FEM meshes)
- Law, routes, member masks and groups: `viscoelastic.py` header. Displacement formulation: groups only over the
  omega-dependent members, each adding its excess G*(w) - G0 (1 + j eta) on unit-modulus D_g / V_g (the relaxed part
  stays inside K + j L); mixed: every member is a group (the pressure zones). At most MAX_GROUPS (validation V20 for
  plain numbers, the solver for expressions): a spatially varying modulus is an error, not N groups.
- Group terms are `frequency_response.FrequencyTerm`s added at fixed slots of the summed pattern per frequency (pencil
  `extra=`, host route `values[positions]`), never pencil terms of their own nor re-assembled per frequency; reduced
  sweeps refuse any term (`_refuse_reduced`; validation V7 / V7b). A `SchurScaleTerm` only enters the equilibration
  diagonal: A, its products, residual rows and every solution handed out (solve, `_rows_solve`, last_solution) stay
  unscaled.
- `terms=()` / `scale_terms=()` stay bit-identical to the plain pencil (solid_harmonic_terms compares the bits): every
  new branch of HarmonicSystem is guarded by `self._terms` / `self._scale_terms` (functionals: `self._functionals`).
- Every study but frequency_response reads the relaxed G0, K0 alone (V16 warns for user-set values). Shown harmonic
  stresses on FEM meshes (responses and damped modes) are Re(C*(omega) eps) with the loss part (`_harmonic_moduli`);
  energies stay the storage part; tensor meshes unchanged.
- viscosity is laminar_flow's property too: a viscosity-only cell joins the solid only with formulation mixed (void in
  the displacement formulation: V15 warning, not an error, since library water carries it). The viscoelastic arrays
  are not STIFFNESS_ARRAYS (a T-dependent laminar_flow viscosity must not re-assemble the solid in the displacement
  formulation): their change drops the kept harmonic operators (and the element moduli); with formulation mixed it also
  sets `operator_dirty` (membership follows the viscosity), so the next solve re-assembles.

## Mixed u-p formulation (setting formulation; `solver_common.mixed_harmonic_operators`, fem `mixed_elasticity*`)
- x = [u; p], u in the system's numbering (2D the solid map, 3D the full map); the system's static K (relaxed moduli:
  the full lambda of E / nu and K0 members, deviatoric where K0 is unset, liquids 0) carries the dof map and the rigid
  check only and is never factorised. Reactions are the u rows of A x - f,
  so they include -B^T p; a record reads u rows only (`_split_harmonic`).
- Pressure zones: one P1 space per parameter group, duplicated where groups meet (the pressure jumps there); at m >= 1
  the axis pressure dofs are held at 0 (a cos(m phi) pressure is regular there only when 0).
- Gauge pin: the constant pressure of an incompressible region enclosed by held displacements is a null vector, so one
  dof is pinned (`_pressure_gauges`), which drops the region's volume equation 1^T B u = 0. Regions are connected over
  the UNION of the incompressible groups (fem `pressure_regions(space, groups)` joins their duplicated dofs at shared
  nodes): at an interface of two incompressible groups the B^T 1 terms cancel, so e.g. an oil drop in water in a held
  container is one enclosed region though neither group alone is (per-group detection left it singular). Held
  displacements that change that volume make the system unsolvable: `HarmonicOperators.check_held_volumes` refuses
  them per solve (from `_harmonic_fixed`: a new solve path takes its held vector from there). A region of one group
  with every displacement held is refused at assembly. Both rely on the exact row sums B^T 1 (fem sub file "Mixed u-p").
- Held regions (*_domain supports) sharing a compressible solid's material share its P1 pressure and drag it toward 0
  (div u = 0 there; rigid_inclusion_compressible_frequency_response: 2.5e-3 vs 1.2e-5 at K = G): own material.
- Stored zeros vs zero-coefficient terms: an incompressible group's zero pressure block stays in the pattern (PARDISO
  symmetric types need every diagonal stored): as stored zeros of the operator without a column map, but through the
  m = 1 axis-tie column map a congruence drops exact zeros, so there it is a FrequencyTerm of coefficient 0, which makes
  reduced sweeps refuse that system (V7b).
- The pressure display is nodal and averages the groups meeting at a node (a jump smears over one element, like
  stresses); stresses at element points take the per-zone values (`pressure_at`), so derived integrals of stresses see
  the jump.
- Saddle-point equilibration: symmetric scaling only (R A R with d = |diag| + on the pressure rows the SchurScaleTerm
  proxy sum_j B_pj^2 / d_j of the u diagonals, per frequency: every scaled B row has norm <= 1 whatever h or the
  viscous / inertial balance; a fixed weight diag M_p / |G*| is ~(h / delta)^2 too large where w^2 rho h^2 dominates
  |G*| in d_u). It reaches every back end, host SuperLU included (`scale_superlu`, the private SuperLU too; solvers sub
  file "sparse_direct"): unscaled, SuperLU's diagonal pivot test compares u rows against p rows and loses the pressure
  field while reactions survive (max |p| of solid_stokes_second_problem's cpu variant, the sphere pressure of
  solid_oscillating_sphere_drag). Never a row-only scale or a sign flip of the pressure rows, which would break the
  complex symmetry cuDSS / PARDISO LDL^T and the lower-stored DevicePencil rely on.

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

## Spontaneous Brillouin (feature brillouin_scattering; `brillouin.py`, `brillouin_drive.py`, `definition_brillouin.py`)
- Math, conventions, references and fit: `brillouin.py` header; constructor values, flow, sampling, eps per point,
  interfaces and references of the runtime: `brillouin_drive.py` header; FEM loads: fem `elastic_drives.py` (fem sub
  file). The drive is the transpose of the optomechanics overlap (non-conjugated E_c E_i, the same Nye / Johnson
  forms): change both together (Q1 cell / face drives are the transposes of `element_strains` /
  `face_displacements`: brillouin_tensor_transpose). Moduli laws are `viscoelastic.ModulusLaw.at` (the one formula),
  called once per omega array (`brillouin.modulus_values`; the scalar path of `at` / `excess` stays the Python-number
  one the harmonic terms use).
- Flow: the apply adds a BrillouinDrive (constructor = the frozen contract) in every study, inert outside
  frequency_response (validation warns); that study binds it through `solver_hooks.PhysicsDriveSolver` after the
  background stage and the static states, before `reset_frequency_response` (model sub file "Physics drives"), never
  at apply. Bind returns ONE read-only complex W (n_positions x n_parts, n_unknowns; mixed: pressure rows 0): row 0 is
  the harmonic load and every row a HarmonicSystem functional, so the spectrum is `last_functionals` /
  `solve_columns` (section "Frequency response"). It reads the optics through `solver_hooks.DriveOpticsSolver` (em
  `fdfd_2d` / `fdfd_bor`) as the background left them and never re-solves them (a changed optical system raises in
  `solve_moved_beam` / `drive_fields`). Each solid binds on its own: the m = 0 + m = 2 pair repeats the optical scan
  solves.
- The drive must be the only load (the FDT reads -Im(w^H u) of the drive alone): validation (`_other_load_issues`:
  load features, holds != 0, couplings into body_force slots), bind and every driven solve (`refuse_drive_loads`)
  refuse the same; a temperature change stays a static prestate. Reduced sweeps take one column
  (`refuse_reduced_drive_columns`, validation `_sweep_issues`).
- Both solid solvers call the module functions of `solver.py` (`checked_physics_drive`, `refuse_drive_loads`,
  `bind_drive_columns`, `refuse_reduced_drive_columns`, `record_drive_columns`); inside the drive the mesh kinds differ
  only in its geometry object (`_ElementGeometry` / `_CellGeometry`).
- optical_physics / collection_physics are kind 'str', not 'ref': a 'ref' physics Param must share the feature's mesh
  (`validate._check_physics_refs`) and the optics run on their own tensor mesh, so `definition_brillouin` checks
  name, type, beams, wavelength and grid itself (`_optical_issues`, `_collection_issues`, `_same_optical_grid`).
- Tensor (Q1) solids: planar plane strain, mechanical_loss_factor only, the optics on the SAME mesh (`_tensor_issues`;
  bind compares the centres): the drive reads the optical cells as its own (no interpolation, no window hull rule).
  Their drives use the solid's kept HarmonicSystem (runtime None, private SuperLU; section "Frequency response") for
  the functionals and column solves: one factor alive (`factor = None` before each solve), dropped after each
  frequency's readouts and in reset / publish (`_drop_harmonic_factor`), since that host factor lives outside numpy
  and no ledger sees it.
- Mirror planes count x 2 (mirror_factor): each must be held by a solid Roller / Symmetry (the even mirror of u), no
  scan offset along its normal, none with em_fdfd_bor, none with reference 'homogeneous' (the FFT needs the unfolded
  drive) (`_mirror_issues`, errors; the planes it reports skip the generic `model_mirror_issues` warning, which counts
  Fixed as a holder: `brillouin_issues(reported_planes=)`).
- Window rule: a sampler covers the hull of its sample coordinates plus one spacing (`brillouin.bilinear_weights`
  raises beyond), so validation keeps the drive box inside the first / last window cell centre on every face but the
  mirror planes and the BOR axis x_lo (`_window_issues`): change the clamp and the check together.
- Sampling bias: a face mean along the beam (psd low by cos^4(k h / 2)) biases the values both references read, plain
  bilinear interpolation (~(q h)^2 / 12) the local reference too, so a reference comparison misses such a bias: pin a
  new sampling path against an analytic case (selftests_brillouin_spectrum water cases).
- Moving boundary (triangles): never side samples (they reach across the staircase into the other material), the
  continuous E_par and D_n at the Gauss point itself (`brillouin_drive.py` header). Q1: the two-sided face mean (open:
  ~1.5 % low in the moving-boundary part at 10 nm glass).
- Homogeneous reference: the zero padding per axis follows the phonon attenuation (`brillouin.reference_padding`; a
  fixed padding 2 would leave a ~12-15 % image share in the 2D water cases); MAX_REFERENCE_CELLS raises at run and in
  validation (`homogeneous_padding` in both).
- Axisymmetric: two order-1 optical fields drive the solid orders 0 and 2 only, so the full co-polarised spectrum is
  two solid physics on one triangle mesh, each with a brillouin_scattering on the same optics (the one azimuthal-order
  split: `definition_helpers.ORDER_SPLIT_READERS`, root "Physics definitions"); their psd add in a derived curve (no
  combined result). Odd parity is the even solve rotated by pi / (2 m); the m = 0 torsion is dropped (its share warned
  above TORSION_WARNING). Ring weights ring_angle(m) against the BOR standing-wave power (em sub file).
- Elastic PML: drive points and reference cells on window elements only (`_drive_inputs`); the ring's absorbed power
  is part of -Im(w^H u).
- Memory: n_positions x n_unknowns x 16 B of W (a 100-position map of a 5e5-dof solid: 0.8 GB host); a streaming
  option (re-solving the optics per frequency) is open. A bind of more than one photoelastic column also keeps the
  strain operator of every drive point until it returns (fem `DrivePoints.kept`, size in its comment), which the bind
  estimate (`_bind_bytes`) counts with the held point arrays. `result()` is built once per bind and records and kept
  (references and peak fits), handed out as a deep copy.
- Not built: 3D (em_fdfd_3d has no driven solve or 3D focused source; a water focal volume is ~5e7 Yee unknowns and a
  P2 phonon side ~1e7 dofs, beyond a direct factor; brillouin.py already works on six Voigt components, so a
  `fem/elastic_drives3d.py` twin plus a solid_mechanics_3d registration would be additive, an analytic Debye
  illumination without an optical solve first); a co-moving collection beam (scans move the illumination only); an
  m = 0 torsion solid (cross-polarised shear peaks of solids); reduced sweeps of several columns (block moment
  matching); anisotropic interface eps (sides isotropic); spatially varying materials (staircase eps fallback).

## Squeeze film (`squeeze_film.py` = the tet runtime; the film itself is `fem/squeeze_film.py`, fem sub file)
- Films are evaluated in the `_modal_pencil` finish (on the merged list after a search), so every modal path of the
  tetrahedra keeps that finish; they are not coupled into frequency_response (validation warning).

## SolidMechanics3DPhysics / ElasticSystem3D (solid_mechanics_3d)
- Matrix-free (no CSR): `kernels3d.BrickKernels` products, every CG step a `FusedBrickIteration` and the gpu check
  blocks CUDA graphs; captured arrays and the reaction scratch: `ElasticSystem3D` comments (`_fused_operator`,
  `_reaction_key`). CG iterations scale with the longest edge: keep 3D self-test strips short.
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
- An x / y body force is one centroid value per element (`ctx.eval_field`), so it caps P2 convergence at O(h^2)
  (sphere_breathing_viscoelastic_frequency_response); no load is normal to a curved boundary (tractions in model axes).
- Elastic PML ring (`pml2d.py` header): eigenfrequency and frequency_response, one stretch at 2 pi target_frequency in
  both (a sweep's reference: validation infos / warns on the reflection R^(f_min / target) below it). Every pencil
  term of the ring is stretched through `ElasticFEMSystem.stretch_arguments` (K via `_pencil_stiffness`, K_sigma,
  loss, mass, the viscoelastic groups, mixed D_g / B / M_p,g): a new harmonic term on the solid map takes them too, or
  the ring reflects. FrequencyTerm X and the pencil's mass are then complex (DevicePencil term bytes (16, 16)). The
  axis rule continues through the y rings (`_axis_ids`, PML only: the window-only `face_edges` would leave the ring's
  axis free). Physics drives act on window elements only (`_drive_inputs`); modal reduced sweeps refused (complex
  pencil), adaptive fine. Tetrahedra: still eigen only.

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
  stiffness): fem sub file "Device tangents", "Block patterns".
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
  estimators.
- Viscoelastic group terms and mixed u-p blocks: `definition_viscoelastic` (`viscoelastic_counts`,
  `viscoelastic_memory_items`, `viscoelastic_time_items`), shared by both estimators. The factorised pencil of mixed is
  the saddle point: every frequency-response item takes `ViscoelasticCounts.unknowns` / `entries`, the factor fill is
  the u-block fill times `fill_scale` ((dofs per node)^2: `scaled_fill` on a counted FactorFill, the SuperLU heuristic
  alike), and 3D `_tetrahedra_unknowns` / `_entries` include the pressure, so the size rule and the time recipe see it.
  Member shares (`_solid_fraction`, the pressure nodes) count the moduli route and, mixed, the liquids.
- An elastic PML in the 2D frequency response makes those matrices complex: `viscoelastic_memory_items(stretched=)`
  and the frequency-response branch of `_fem_memory_estimate` (complex loss / mass, the kept stretched K of the
  displacement formulation, pencil term bytes) switch on it; a new stretched allocation follows the same flag.
- brillouin_scattering items: `definition_brillouin.brillouin_memory_items` / `brillouin_time_items`, called by both
  2D estimator branches (triangles, tensor), mirror bind / record / result and the HarmonicSystem column path (one W,
  bind transients, local-reference bins, cuDSS device weights only on cuDSS: the caller passes `device`, False on the
  tensor solid's private SuperLU); a new allocation there updates them. The MaterialGrid arrays the bind copies come
  from `context.mesh_material_arrays()` (meshing.material_array_names, optional arrays included), the triangle drive
  elements from `context.selection_cells`, a tensor solid's optical cells from its own MeshEstimate (a triangle
  solid's context has no estimate of the optics' mesh: base-dx approximation). Their `*_GUESS` constants are
  uncalibrated.
- Unverified against counters (no gpu run in selftest_out): the static 'linear' cuDSS items, the solid
  frequency-response HarmonicSystem (with its viscoelastic group terms and mixed u-p blocks), solid DevicePencils,
  device Newton tangents, the brillouin_scattering items.

## Self-tests (`selftests.py`)
- Shared case helpers live in selftests.py, selftests_3d.py, selftests_fem3d.py, selftests_eigenfrequency.py,
  selftests_formulations.py and selftests_nonlinear.py (the 3D nonlinear twins reuse its helpers): reuse, never a local
  copy. Case notes are in the case descriptions.
- A frequency_response run with several frequencies keeps only the last one live: derived field values read that one,
  earlier frequencies are checked through frequency_probe series and support reaction tables (selftests_viscoelastic).
- Template modules (mechanics entries of `TEMPLATE_SOURCES`, model/import_contracts.py) import no non-template sibling
  at module level: get_registry loads them, and contract selftests_lazy does not follow their imports.
