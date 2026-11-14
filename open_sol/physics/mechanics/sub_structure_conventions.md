# open_sol/physics/mechanics — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Solid mechanics (C3). Designs (repository root):
`solid_mechanics_3d_plan.md`, `piezo_3d_plan.md`.

## Layout (2D and 3D)
- `solver_common.py`: SolidMechanicsBase (PhysicsSolver contract, dirty logic of `step`, reactions, warm start,
  `_element_lame`), SupportRegion, ElasticSystemBase (precision / tolerance, one `solvers.pcg.PCGSolver`). Subclasses
  SolidMechanicsPhysics (`solver.py`, 2D) / SolidMechanics3DPhysics (`solver3d.py`) supply `_new_fields`,
  `_pml_copy_pairs`, `_assemble`, `_build_rhs`, `_write_outputs` (no PML copy: the base step does it after), renderer,
  support / load / probe methods; `_solve_system` (default PCG) is overridden only by the 2D piezoelectric solid.
- Fields carry `shape` (re-allocated on a grid-shape change); `fields.py` / `fields3d.py` hold the output / input /
  slot names (`kernels3d.OUTPUT_ARRAYS_3D` = 3D output order).

## Backends (gpu and cpu; solver_plan S8 E)
- Both types ('gpu', 'cpu'). Every array comes from the system's `xp` / `sparse` / `ndimage` (backend =
  `domain.backend.name`); host copies via `core.backend.to_host`, never `cp.asnumpy` / `.get()`. `prepare` raises on a
  domain of another backend; a build-injected PCGSolver must match (ValueError).
- 2D: `_block_precondition` is a twin ('mechanics.elastic_block_jacobi'); `stiffness_stencil` / `mass_stencil`
  default xp = module of their first array, `ElasticPattern` defaults to numpy (the device assembly passes xp). Modal,
  guided and piezo solves are host.
- 3D: every kernels3d RawKernel is a twin ('mechanics3d.<kernel>', core.kernels.real_raw_twin: GPU side compiled per real type of args[0]; multigrid3d alike). numpy
  twins = 2-point Gauss per axis (equal to the CUDA Walsh-basis forms up to rounding), temporaries per call: a cpu CG
  iteration is far slower (estimator books the twin temporaries as cpu build extra). numba twins (`_NUMBA_TWINS`,
  cache=True) of the per-iteration kernels: elastic_product and elastic_mass_product (closed-form per-node gathers),
  block_precondition3 and the fused CG pair; CPU case solid_3d_product_numba compares each with its numpy twin. A numba
  twin that needs another kernel calls its registered twin (`_TWINS[name]`), never its factory, so each kernel is
  compiled / cache-loaded once per process.
- Parity cases (`selftests_backend.py`, `solid_parity_case`: solid_mechanics_cpu, solid_mechanics_3d_cpu,
  solid_mechanics_3d_block_eigen_cpu, solid_optics_cpu_gpu) wrap exact cases: the case's own check grades the cpu runs; named fields must match the gpu
  runs within FIELD_TOLERANCE relative to the max of the gpu fields of the same kind (u / strain / stress + von_mises),
  so name a NONZERO field of each kind (a zero u_x under rollers would be graded by its own round-off); a field only
  the parity reads needs a closed-form item in the case's check; every cpu run must record the solid 'linear' role on
  'cpu'. solid_3d_rigid_mode_check runs on both backends itself.
- Definitions: `definition_common.py` (build's linear solver, MATERIAL_PROPERTIES = one Param per name for both
  types, support / material / state helpers, the apply / result callbacks both types share: body force, temperature
  change, support and electrode tables), `definition.py` (2D type + every solid coupling, registers definition3d),
  `definition3d.py` (solver imported in create, SupportRegion / probe_result inside callbacks). A solid
  that StudySpec.solved_physics leaves out is a stationary background: validate / estimators rebind their context by
  `definition_common.background_context`, create reads `solid_study`; a new study check goes through them, never
  spec.study.type. No CG settings on the physics; estimator dtype names map to PCGSolver names by
  `materials.SOLVER_PRECISIONS`.
- Type catalogue: `solid_mechanics` (default 'solid'; shares alpha / alpha_xx / alpha_yy and their arrays with heat,
  adds alpha_zz; rho read only by eigenfrequency / frequency_response; youngs_modulus / poisson_ratio /
  photoelastic_p11 / _p12 are elastic_waves' Param objects, `material_params.py`; optional c_ij, piezoelectric,
  axial_wavenumber = 2.5D guided modes (u_z nonzero only there), frequency_response study). `solid_mechanics_3d` (3D
  only, separate type like em_fdtd_3d; default 'solid'; same material Params incl. rho; tensor meshes: lowest modes
  only; tetrahedra: target_frequency / search targets / band, 21 c_IJ, piezoelectric (18 piezo_e*, six eps_dc*,
  charges in C, static on 'linear' sparse_direct, short / open modes on 'eigen'), frequency_response on 'linear').

## SolidMechanicsPhysics / ElasticSystem
- Static, small strain; cells isotropic (E / nu -> `materials.lame_coefficients`) or c_ij: every coefficient goes
  through `stiffness.ElementStiffness`, never lambda / mu directly. sigma = C (eps - alpha dT), dT =
  fields.temperature_rise (slot) + temperature_rise_ext (features); thermal stress per kelvin `_thermal_coefficients`
  = C : alpha. Rules are written for plane strain unless they say otherwise.
- out_of_plane (planar only): plane_strain eps_zz = 0; plane_stress = static condensation of sigma_zz = 0
  (`stiffness.operator_coefficients(plane_stress=True)`, c33 = 1 in void cells; eps_zz output per cell, stress_zz 0);
  generalized_plane_strain eps_zz = e0 + e1 x + e2 y (the industry-standard form, setting bending picks the free terms)
  by superposition (`generalized_plane_strain.py`, `_solve_generalized`): the e = 0 solve + one
  `ElasticSystem.solve_homogeneous` per free term (f_k = -int B^T (c13, c23, c36) h_k dA, h about the window centre,
  kept in `_unit_axial_solve` until the next `_assemble`); the 3 x 3 Schur matrix S_mk = int c33 h_k h_m - u_k . f_m is
  exact for bilinear u. Moments about the drawing origin (M_x = int y sigma_zz, M_y = -int x sigma_zz). `load_full`
  gains sum e_k f_k so reactions stay K u - f (triangles: section "Triangle meshes").
- Only the planar plane-strain solid has piezoelectric, guided modes and the planar optomechanics study (constructor
  ValueError + validation; an axisymmetric triangle solid runs optomechanics with em_fdfd_bor optics, section "FEM /
  axisymmetric overlaps"); eigenfrequency refuses generalized plane strain (no mass on e). Setting thickness (planar)
  only scales `reaction` (N, `force_unit`): the system stays per m of depth.
- Axisymmetric (setting geometry, `ElasticSystem.axisymmetric` set before prepare): x = r from the x_lo window edge
  (padding ignored), y = z; `gauss_stiffness_stencil` (2 x 2 Gauss, weight 2 pi r, hoop N / r) and
  `axisymmetric_mass_stencil` (exact). Loads go through `add_stress_load` / `add_volume_load` / `face_node_lengths`
  (ring weights): a new load must use them, never `element_area` products. prepare holds u_r = 0 on x_lo (axis rule);
  rigid-mode check and validation need a held u_y only. Forces / reactions whole ring in N; the _zz outputs are hoop
  components (renderer relabels).
- Discretisation: bilinear Q1 elements = window cells, nodal (u_x, u_y) on cell corners, dof = 2 node + component, node
  = i ny_nodes + j; exact closed-form element integrals; dense per-node stencil compacted to CSR by
  `ElasticPattern.stencil_positions`. Q1 locks toward nu = 0.5 (validation warns above 0.45) and is stiff in pure
  bending: mesh thin films with >= 4 cells over the thickness. Plane-strain bending stiffness is high by ~c / n^2 (n
  cells over the depth, square cells), c = (1 - nu) / 2 + nu^2 / (1 - 2 nu): 3.5 % at nu = 0.3, n = 4
  (examples/mechanics/simply_supported_beam_gravity.json matches it to ~1e-3).
- A fixed_domain pin cell holds all four corners at 0, suppressing that cell's free strain: the far field gets a rigid
  kink (translation + rotation ~ strain x dx^2 / depth), so far-field displacements carry an offset (stresses fine).
  No single-node / single-component point support.
- Void cells (effective c11 <= 0: no youngs_modulus and no c11): nodes touching only void are pinned at 0 (empty
  rows), outputs 0, body forces dropped. Outputs (u = mean of 4 corners, strain / stress from centre gradients,
  strain_xy = half the engineering shear) are float32 full-grid, edge-extended into the PML.
- Supports hold dofs (`ElasticSystem.hold`; later holds win the value, mask = union), eliminated as in
  DiffusionSystem. Supports act on outer-face nodes (corners included) or on corner nodes of selected cells
  (fixed_domain); no object-perimeter supports or loads.
- segment_load (2D only; not in solid_mechanics_3d: a partial face there needs a rectangle on the face): an
  axis-parallel line on its nearest node line, nodal forces t x `segment_node_lengths` (exact 1D shape-function
  integrals over the clipped segment, axisymmetric ring-weighted), so the total is exactly t L with ends inside cells.
- Rigid-mode check (`_check_rigid_modes`, each assembly whose solid / held sets changed): 8-connected solid regions
  (hinge contacts count as joined, so a mechanism through a corner contact shows as an unconverged CG instead); each
  needs a held u_x, a held u_y, and held u_x at > 1 row j or held u_y at > 1 column i (exact rank condition;
  axisymmetric: a held u_y only). Validation checks only that both directions are held somewhere.
- Block-Jacobi PCG (stop reference 'rhs'): `assemble` factorizes it with the A_solve CSR and the 2x2 node-block
  inverse (diagonal fallback when det <= 0). max_iterations 0 = twice `solvers.pcg.resolve_max_iterations` (set by
  `definition_common.solid_linear_solver`; elastic conditioning ~2-4x worse than diffusion). The role is
  `definition.SOLID_LINEAR_ROLE` (local, not definition_helpers'). gpu check blocks are CUDA graphs (capture_blocks;
  every `assemble` factorizes again, which starts new graphs: anything replacing A_solve / the inverse blocks without
  `_bind_linear_solver` must call `release_block_graphs`). 'mixed' (Automatic on the gpu outside the study defaults):
  float32 copies of A_solve and the inverse blocks per assembly, products through `system._CsrApply` (pcg's CSR
  RawKernel: a cupyx SpMV cannot be captured), the float64 A_solve refreshes the outer residual.
- sparse_direct (Automatic in stationary / time_dependent on both backends, `study_defaults`; back_end 'auto' = cudss
  on the gpu with nvmath, else host SuperLU; its size rule and the fit guard count 2 unknowns per node,
  `direct_components=2`): `ElasticSystemBase` wraps the SparseDirectSolver in `DiffusionDirectSolver` (3D refuses
  it, `accepts_direct`); `_bind_linear_solver` only marks `_factor_stale`, and the next `solve` / `solve_homogeneous`
  factorises A_solve (cuDSS: `factorize_device_csr`, the kept factor refreshed on its plan), so piezoelectric and modal
  solves never factorise it. Over the GPU budget (host factor: the available RAM or 2e6 unknowns) the fit guard falls
  back to pcg (gpu: mixed; estimator: no PCG workspace,
  `fem.diffusion.cudss_diffusion_estimate`; time recipe at `_Q1_FILL_SCALE`). A pcg node with no precision still takes
  'mixed' (`schema.with_node_precision`).
- Reactions: `ElasticSystem.reaction(dof_mask)` = sum of K u - f (float64) over the held dofs (N/m); needs
  `load_full` of the last build_rhs, so loads must not change between solve and read (they change only through
  features / inject_source, which mark dirty).
- `step` re-solves when operator dirty (`materials.STIFFNESS_ARRAYS` changed, supports), loads dirty (alpha, slots,
  features) or first call; n_solves = coupling revision counter. A re-assembly allocates the new operator before the
  old is freed: the estimator books it as a run item when a stiffness references a state field
  (`_stiffness_follows_state`).

## Anisotropic stiffness (`stiffness.py`, c_ij entries; piezoelectric plan stage 5)
- Industry-standard plane strain: the full law in Voigt model axes (1 xx, 2 yy, 3 zz, 6 xy, engineering shear) with
  out-of-plane strains held (anti-plane u_z clamped, no c_4J / c_5J). Arrays c11, c12, c13, c22, c23, c33, c16, c26,
  c36, c66 (`materials.OPTIONAL_STIFFNESS_SCHEMA`): any entry allocates the eight besides c16 / c26; those only by
  themselves (an allocated pair switches elastic_waves' monoclinic kernel on in a shared model). c11 / c12 / c22 /
  c66 / c16 / c26 are elastic_waves' arrays (its kind / unit / bounds); NaN or unallocated = unset. 2D type;
  solid_mechanics_3d reads the 21 c_IJ on tetrahedra only (bricks refuse them, `_tensor_issues`).
- `element_stiffness` = per component the finite entry, else the Lame value (c11 = c22 = c33 = lambda + 2 mu, c12 =
  c13 = c23 = lambda, c66 = mu, c16 = c26 = c36 = 0); solid = c11 > 0, so a c_ij-only cell is solid and must set every
  entry its formulation reads (a crystal entry expands to all: no youngs_modulus needed). Monoclinic c16 / c26 enter
  `stiffness_stencil` and the shear thermal load (`add_stress_load(stress_xy=)`, planar only);
  `check_positive_definite` raises per formulation. Axisymmetric: orthotropic only (c16 / c26 / c36 raise). Guided
  modes refuse any c_ij (the real form u_z -> -i u_z needs isotropy). stress_zz = T_3 (- e_k3 E_k with piezo).
- Self-tests `selftests_anisotropy.py`: solid_anisotropic_operator (CPU), solid_anisotropic_uniform_strain (exact Q1);
  orthotropic roller-box modes: solid_eigenfrequency_roller_box.

## Eigenfrequency study, 2D (`modal.py`)
- `solve_eigenfrequencies(n)` re-assembles if dirty and uses the host float64 K and consistent Q1 mass
  (`_build_harmonic_operators`, cached `_harmonic_operators`; a solid cell with rho <= 0 raises), never the
  solve-precision device CSR (a single-precision 'linear' node must not perturb the pencil), with the free-dof mask
  of the last assembly, through the 'eigen' runtime's `eigsh_shifted` (SuperLU, or cudss / cudss_mixed real LDL^T,
  also for guided and indefinite piezo pencils; a float32 factor missing 1e-12 in 6 refinement steps raises; scripts
  get a private ArpackShiftInvertSolver reported in `work_counters`) for n + 4 eigenvalues, ranked by |omega -
  omega_t| (target_frequency, 0 = lowest). Held dofs are 0 (Displacement values, loads, thermal strain, slots
  ignored; no stress stiffening on the tensor mesh: section "Prestress"). The rigid-mode check keeps K positive
  definite (no free-free plane-strain modal analysis on the tensor mesh); a nonzero target is factored at omega_t^2 (1
  + 1e-4), so a target on a resonance is not singular.
- `select_resonance(k)` writes the peak-normalised mode (peak node |u| = 1 m, largest component positive) into
  `system.unknown`, `_write_outputs(thermal=False)`; n_solves stays 0, so support reactions raise (results left out
  with a warning). rho is in both material schemas (3D: one more float32 full-grid array in every study).
- Shared definition pieces (`definition_common`): `resonance_table_feature(order_text)` (callbacks read
  `physics.eigen_solve`), `modal_study_issues`, `modal_resonance_count`.
- `ModalPencil` (2D and 3D): free-dof reduction (or P^H K P) once; kind 'symmetric' / 'damped' (loss, PML) /
  'loaded' (resistive electrode); one factor kept across shifts (`previous=`, cuDSS plan reuse); K, M, load matrix as
  host CSR, plus the eigen runtime's `ShiftPencil` (`_pencil()`) that count_below / eigsh_shifted / eigs_shifted all
  take, so every shift and count refreshes one factor on one pattern (one cuDSS plan per pencil). Eigen factorization
  'auto' resolves on the gpu with nvmath to cudss (triangles and tetrahedra: ARPACK's 37-63 applications favour the
  double factor over mixed + GMRES refinement), else superlu; on a cuDSS
  refusal the runtime falls back to SuperLU for good (that shift is redone); Hermitian (Bloch) pencils are counted by
  a counting SuperLU so band searches stay verified (solvers sub file arpack_shift_invert). From
  `DEVICE_PENCIL_MIN_ENTRIES` stored entries on cuDSS it holds a DevicePencil (pool memory ~2-3 value arrays + refresh
  transients; E5 200 nm: ~1 GiB pool high-water; estimators: section "Memory estimates"). ModalPencil vouches its terms are
  symmetric (`symmetric_known`, no check; a complex Bloch prolongation keeps the check): a new term that breaks
  symmetry must drop the vouch. `solve_modes` = `ModalPencil(...).solve_at` + `release()`: a caller holding a pencil must release it (cuDSS factor lives outside the cupy pool).
- Multi-shift: `solve_modes_at_shifts(pencil, omega_targets, n_per_shift)` (union of 'nearest' solves),
  `solve_band_modes(pencil, f_lo, f_hi, n_per_slice, max_modes)` (inertia-counted for 'symmetric', 50 % overlap +
  warning otherwise); both release and return frequency order with info target_frequency NaN, 'targets' / 'band',
  'expected_count', 'shifts'. Merge clusters within MERGE_TOLERANCE (loaded 1e-5) keep independent vectors only,
  M-orthonormalised for 'symmetric' (a degenerate pair stays two modes). Band rigid floor = RIGID_EIGENVALUE_FRACTION
  x upper edge; a pencil needing a shift > 0 starts at BAND_FIRST_SHIFT_FRACTION x omega_hi.
- Study search 'targets' / 'band' (`modal.MultiShiftModes` mixin of solver.py, solver_fem.py, solver_fem3d.py; not
  the matrix-free solver3d): every solver builds its pencil in `_modal_pencil(lowest_target)` -> (ModalPencil,
  finish), also used by solve_eigenfrequencies, so a new pencil variant goes there once; finish = piezo expansion and,
  on tetrahedra, squeeze films (on the merged list). Rebuilt per study call. Piezoelectric bands are refused (solver
  and `eigen_search_issues`); estimators read `modal_kept_count` and `definition_helpers.eigen_search_counts`.
- MUMPS / PARDISO eigen factorizations (opt-in, `eigen_role(mixed_factorization=True)` offers mumps / pardiso and
  their _mixed variants): host LDL^T of K - sigma M through the same ModalPencil (`previous=`: analysis reused across
  shifts), host route (`on_device` stays cuDSS-only). Band counts from the library inertia (perturbed pivots, Hermitian
  pencils: a counting SuperLU, arpack_shift_invert); a mixed factor's counts are inexact (the float32 / complex64
  warning of `solve_band_modes`). `_failure_text` words a HostDirectAccuracyError (probe missed) by its library
  (`host_direct.factor_label`), a singular one as for SuperLU; `_check_size` keeps the SPLU limits and names the
  library. Self-test solid_eigenfrequency_band_host_direct (`selftests_eigen_search.py`, cpu; skipped without
  python-mumps / pydiso): the rectangle band on mumps / pardiso.
- Exact roller-box cases need float32-exact material constants: an isotropic E / nu storage alone shifts the
  frequencies ~7e-9 (0.73 of the 1e-8 tolerance in the 3D and guided roller boxes); solid_eigenfrequency_roller_box
  uses orthotropic c_ij only (`selftests_eigenfrequency.py`).

## Piezoelectric plane strain (`piezo.py`, setting piezoelectric; piezoelectric plan stage 5)
- Unknowns (u_x, u_y, phi) per node, dof = 3 node + component, host float64: `coupled_operator` = symmetric
  indefinite [[K_uu, K_uphi], [K_phiu, -K_phiphi]] (e on solid cells only, eps0 eps_dc on every window cell: void is
  dielectric), `coupled_mass` = u mass only. Signs: phi rows are int grad N . D = -q, so an electrode's charge is
  -sum of its phi rows of A x; a reaction is the u rows of A x - f (`reaction` overrides the PCG one: K_uu u misses the
  coupling).
- Electrodes (`piezo.Electrode`; electrode on outer faces, electrode_domain on cell corners): 'voltage' holds phi
  (modal: 0 = short), 'floating' ties nodes to a master through the column map G (x = G x_r) carrying `charge`
  (modal: 0 = open). A node in two electrodes raises; at least one voltage electrode required. G also scales phi by s =
  sqrt(max K_uu diag / max K_phiphi diag) (SuperLU threshold pivoting needs balanced blocks). Every G^T X G goes
  through `fem.assembly.congruence` (voltage-only G is diagonal: a row / column scaling, not two sparse products).
- `PiezoSystem` / `_electrode_map` / `Electrode.dofs` work on explicit dof indices (2D `interleaved_layout`, 3D
  `piezo_fem3d.block_layout`); `potential_active` fixes phi dofs no element carries (3D PML shell). `runtime` = a
  'linear' SparseDirectSolver (3D) or None (2D: private SuperLU counted as piezo_*).
- Static: `_solve_system` = one host SuperLU of G^T A G on the free dofs (factor kept until the next `_assemble`), not
  the 'linear' pcg (still assembled, unused); counted as piezo_factorizations / _solves / _factor_host_bytes outside
  any role, and `system.direct_solves` (part of n_solves).
- Modal: (G^T A G, G^T M G) to `modal.solve_modes` with `displacement_dofs` (peak normalisation on u only; vectors
  come back as G x_r with tied phi filled); massless phi makes M semi-definite (ARPACK mode 3 accepts it); the
  factor of A - sigma M is indefinite but non-singular. `select_resonance` splits (u_x, u_y, phi). modal_overlap_data
  raises (no optomechanics), no guided modes.
- Outputs: electric_potential / electric_field_x / _y (every window cell, 0 without piezo), stress_* -= e^T E,
  stress_zz with piezo_e13 / _e23 (read nowhere else). Validation warns when a piezo material has neither
  youngs_modulus nor c11.
- Materials: piezo_e* (elastic_waves' names + solid-only piezo_e13 / _e23; one entry allocates all eleven) and
  eps_dc_* (electrostatics' names, three) are `OPTIONAL_PIEZO_SCHEMA`; prepare adds missing ones (estimator books
  them); `materials_changed` of them re-assembles.
- Harmonic: `PiezoSystem.solve_harmonic` = a `HarmonicSystem` of (A, L, M) with the same G and free set, kept while
  mass and loss are the same objects (callers cache them: a new object = a new reduction).
- Exact cases at 1e-8 .. 1e-9 need float32-exact constants (`selftest_helpers.float32_value`): the MaterialGrid stores
  float32 (e = 1.36 reads 1.05e-8 off). Self-tests `selftests_piezo.py` (20 x 2 bar, exact Q1): solid_piezo_bar_static,
  solid_piezo_bar_modes.

## Frequency response (`frequency_response.py`, `bvd.py`; study frequency_response, model/study_frequency_response.py)
- In Hz (the EM frequency_domain study stays wavelength-based): `solve_frequency_response(f)` (after
  `reset_frequency_response()`) = one complex factor of (K + j L - omega^2 M) x = f (exp(+j omega t)) on the free dofs
  of the last assembly. `HarmonicSystem` reduces K, L, M once and sums per frequency by `fem.assembly.pattern_sum`
  (value-independent pattern, so a cuDSS plan holds); on a cuDSS runtime the terms stay on the device (DevicePencil),
  on MUMPS / PARDISO they are a HostPencil of its own term arrays (`runtime.pencil`; never modify them in place),
  and each frequency calls `factorize_pencil(row_scale=(|diag K| + omega^2 |diag M|)^-1/2, previous=)`; else
  `factorize_host` (SuperLU); without a runtime a private SuperLU. A host direct refresh defers its probe to the
  solve (double): a HostDirectAccuracyError can come from `solve_host`, as a cuDSS one does. 2D tensor solids: private SuperLU outside the roles
  (counters harmonic_* / piezo_harmonic_*; the 'linear' pcg assembled, unused); triangles and tetrahedra: their
  'linear' sparse_direct runtime (3D operator class 'spd' is nominal there). Loads, held values, electrode voltages and
  floating charges are phase-0 amplitudes (dead loads only: the thermal and initial-stress loads are static, never a
  drive; tensor: `_total_load(thermal=False)`, kept as host copies in `_harmonic_drive` until `_loads_dirty` is set or
  an assembly, so a new load mutation must set `_loads_dirty`; FEM: `_external_load()`). Planar and axisymmetric; generalized plane
  strain and guided modes raise.
- Losses (`OPTIONAL_LOSS_SCHEMA`, read only here): mechanical_loss_factor (c -> c (1 + j eta), L = eta K of solid
  cells), piezo dielectric_loss_tangent (eps -> eps (1 - j tan delta)): loss operator `coupled_operator(eta c, 0, -tan
  delta eps)`. Host K / L / M are rebuilt in float64 from the ElementStiffness, never copied from the device CSR; kept
  in `_harmonic_operators` until the next assembly or a loss / rho change.
- Outputs (`_show_harmonic`; tetrahedra `_write_complex_outputs`): real parts of the harmonic amplitude alone (stress,
  elastic strain and energy densities with dT = 0 and no sigma0, as a mode: `_write_outputs(thermal=False)` on every
  mesh kind), u_abs = |u| of the complex centre displacement; on the tensor mesh support
  reactions and electrode tables raise and probes record nothing (FEM meshes: section "Prestress, damped modes,
  complex records"). Each solve appends a `FrequencyPoint`. solid_frequency_response_bar's 'heated' run (the elastic
  bar with a uniform temperature_change) pins this on the tensor mesh: its u_x and stresses equal the unheated run's.
- Reduced sweeps (sweep_method adaptive / modal; fem sub file "Reduced sweeps"): `HarmonicSystem.plan_reduced(omegas,
  method, fixed, load, extra_rhs)`; `solve` raises when fixed / load / extra_rhs differ from the planned ones.
  `set_output_rows(rows)`: `solve(fields=False)` returns vectors exact ONLY at registered rows, so a solver's records
  must read only rows it registered (3D: element nodes of each frequency probe + held support rows; 2D triangles:
  probe + support rows; electrodes: `PiezoSystem.electrode_rows`). Modal = `solve_harmonic_resonances` (private host
  shift-invert; the 'eigen' role is modal studies only), refused with a column map / resistive load. 2D tensor solids
  plan a private `_reduced_harmonic`; triangles, 3D and piezo plan the HarmonicSystem they keep.
  `reset_frequency_response` clears every plan. Fields only where `wants_fields` (direct sweeps too, R6), the rest via
  `publish_frequency_response`. Elastic direct solves of a skipped frequency return x and A x at the registered rows
  only; the free solution stays where the factor solved it (cuDSS: device) until `last_solution()` at publish. A
  piezoelectric direct solve stays full (charges read every phi row), kept in `_unpublished_solution`.
- admittance feature (`definition_frequency_response.py`): Y = j omega Q / V of the named (or only driven) voltage
  electrode x thickness; extras `bvd.bvd_fit` (NaN + fit_warning below 6 points or without a dip above the peak; one
  resonance per fit band), kappa_e / omega_C0_Z (`bvd.external_coupling`, load_impedance) need C0 / L1 in F / H:
  depth factor = the 2D thickness (None = per m: NaN + kappa_e_warning) or 1.0 for solid_mechanics_3d
  (`getattr(solver, 'thickness', 1.0)`).
- Self-tests `selftests_frequency_response.py` (solid_bvd_fit_round_trip, solid_bvd_kappa_e,
  solid_frequency_response_bar), `selftests_reduced_sweep.py`. Example
  examples/mechanics/yz_ln_thickness_resonator_admittance.json.

## Guided (2.5D) modes (`guided_modes.py`, setting axial_wavenumber)
- Eigenfrequency study only (info elsewhere, error in optomechanics). The optomechanics study instead creates the
  solid `guided_by_study` when an enabled optomechanical_coupling has scattering 'backward'
  (`definition_optomechanics.backward_scattering`, read by create, validate, estimator) and calls
  `set_axial_wavenumber(2 Re beta)` after the optical solve; until then `axial_wavenumber` is None and `_guided_pencil`
  raises.
- u ~ exp(i (k_z z - omega t)), unknowns (u_x, u_y, w), u_z = -i w: real symmetric only for isotropic cells (normal-
  shear coupling needs the complex form). Physical u = (u_x cos, u_y cos, w sin); strains / stresses are phi = 0 values
  (xz / yz shear not output).
- Host numpy assembly (`guided_operators`, 3-component `ElasticPattern(xp=np)`), no device / plane-strain assembly.
  u_z is held by Fixed and Fixed (domain) only; Roller / Displacement leave it free. Rigid modes: none for k_z > 0
  (supports optional); at k_z = 0 the plane-strain check + `check_antiplane_supports`.
- `select_resonance` splits into `system.unknown` and `_axial_nodes` (read only with thermal False);
  `modal_overlap_data` carries `axial_wavenumber` (None = plane strain), 3 dofs per node then.
- Self-tests `selftests_guided_modes.py`; solid_guided_modes_roller_box pins the sign of the k terms by the mode-1
  shape.

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
  full-dof rows masked by `free`, one brick product / mass product / preconditioner launch per row; float32 rows on
  float32 copies (the mixed linear solver's `_single_operator` + widths_single when kept, else made per solve); the
  'multigrid' preconditioner reuses `self.multigrid` even after its CG guard fell back (LOBPCG only needs it SPD),
  else builds a private BrickMultigrid on the float32 copies (block Jacobi + warning when the build raises); float64
  rows through it cast in (multigrid copyto) and out (`cycle_out`). No inner CG, rhs / CG unknown untouched. Still
  needs a float64 'linear' solver (the system dtype). Estimators: `_block_eigen_values(context)` switches the 3D
  memory items (block_eigen.block_bytes + copies + private hierarchy, transient 'eigen solve') and the time recipe
  (block_eigen.time_estimate; setup_ops flags multigrid). LOBPCG cannot target interior modes either. Self-tests:
  section "Self-tests" (solid_3d_eigenfrequency_block_eigen).
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
  Q1 basis: 27-node stencil); rediscretised it over-stiffened a thin plate in void (solid_3d_multigrid_void fell back). Transfers are separable 1D gathers in the node coordinates (graded widths: weights != 1/2); restriction
  = the exact transpose (gather form), so pre / post Chebyshev + R = P^T keep the cycle symmetric for plain PCG.
  `_guarded_solve` (static and every Lanczos inner solve) caps max_iterations at `multigrid_budget()` (the block-Jacobi
  estimate CG_ITERATIONS_PER_AXIS_CELL x longest axis over CYCLE_COST_IN_JACOBI_ITERATIONS, within
  [FALLBACK_MIN_ITERATIONS, FALLBACK_ITERATIONS]): a solve unconverged within it switches the system to block Jacobi
  for good (warning, `multigrid_fallbacks`) and re-solves from there (its failed attempt is taken out of
  n_unconverged). Self-tests `solid_3d_multigrid` and `solid_3d_multigrid_void` (thin plate in void) fail on a
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
  elastic PML). Refused (`_triangle_issues`): piezoelectric (so no electrode / admittance), guided modes, the planar
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
  (`selftests.py`, `selftests_3d.py`, `selftests_bloch3d.py`, `selftests_optomechanics.py`) import no non-template
  sibling at module level (get_registry loads them).
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
  `selftests_backend.py`, a case of its own so the cpu block callbacks time apart). Examples
  thermal_inclusion_sphere_3d (mean-stress identities exact: centre strains x V are the element integrals),
  rotating_disk_3d.
