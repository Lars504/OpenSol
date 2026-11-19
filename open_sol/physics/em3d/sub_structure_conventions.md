# open_sol/physics/em3d — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first, incl. "3D models", and the core sub file's "3D
contracts"). em_fdtd_3d (B2) and the eigensolver em_fdfd_3d; 2D EM rules: `open_sol/physics/em/sub_structure_conventions.md`.
`em_fdfd_3d` (3D only, family 'em', default instance 'em') registers right after em_fdtd_3d and shares its material
Params, Yee grid, faces and mirrors; eigenfrequency study only (host SuperLU 'eigen' role, own 3D size rule; CPML,
subpixel smoothing), pec_domain and resonance_table; no conformal PEC.

## Engine (engine3d, kernels3d, boundaries3d)
- Layout, step order, edges, PEC, mirrors, Bloch and sector mechanics: `engine3d` header; coefficient, CPML (fused
  into `update_h_3d` / `update_e_3d`, no launch of its own), radius, wrap and precision rules: `kernels3d` header and
  preamble; CPML layers: `boundaries3d` (grading only through `axis_cpml_profile`, an axis view of the 2D
  `cpml_axis_profiles`). Shared 3D EM members: physics3d (`EM3DSlots`, `StaticEM3DContract`, `complex_cell_materials`,
  `check_pole`), `renderer3d.YeeRenderer3D`, `mode_ports3d.axis_permittivity` (E component -> MaterialGrid array),
  `boundaries3d.along`: reuse, never a local copy.
- Step graphs (gpu, `STEP_GRAPHS`; capture rules: root "Performance"): `advance` runs H (Yee + CPML per set +
  conformal faces) and E (Yee + CPML per set + the probe launch) through `_fixed_block`. Fields, psi, material_id,
  face_id and the 1D arrays are rewritten in place only; a new reassignable kernel argument joins the block's key. A
  change of what the blocks hold updates `definition._step_launches` (gpu: weight 1, no probe launches).
- Probes (gpu): FieldProbes are recorded at the end of the E block; a new eager E-side writer after it (source,
  pole, mirror row) joins `_probes_need_resample`.
- Materials: eps_r (= eps_zz), sigma_e, mu_r always; eps_xx / eps_yy / sigma_m only when a material sets them
  (`optional_material_schema`): never assume them (`engine.material_arrays`, `physics.permittivity_arrays`,
  `axis_permittivity`). The uint16 index (`material_index3d` header: re-quantisation of > 65534 values, error in
  `engine.quantisation_errors`, ~10 full-grid passes per continuous array per material change) is rebuilt only at
  finalize and after a material change (dirty flag in `step`).
- E coefficients are computed per step in the kernels (PEC = the last id, eps -1; subpixel faces via
  `_face_args`): a new consumer (TFSF, mode source, pole cb, dipole weight) calls `face_coefficients` /
  `engine.face_cb_into` (host: `kernels3d.pec_face_values`), never stores per-cell arrays or its own face mean. An
  H-side consumer reads table_mu and table_sigma_m alike (`edge_coefficients`).
- Bloch k != 0: only monitors read the imaginary set (j ph F_im, `em.monitors._field_sets`); probes, renderer,
  state() and couplings read the real set; q_accum and the TFSF plane wave raise; mode ports solve the Bloch mode of
  the port plane (`mode_ports3d.port_boundary`; the source drives the real set). A centre mean of cell 0 on a phased
  axis needs its wrapped lo face x e^{+j k L} (as DFTPlane3D, harmonic3d), else ~10 % off. Gaps: a z-normal DFT layer
  at cell 0 of a phased z axis, oblique 3D TFSF, a Bloch mode-source model run. Self-tests `selftests_bloch3d`.
- Mirrors: results unfold (root "Arrays, coordinates"): mode ports map them onto cross-section mirrors, flux planes /
  boxes x `monitors3d.mirror_factor`, far field by images (`NearToFar3D(mirrors=)`; mirrors on both faces of an axis
  or a periodic axis raise), a TFSF box face on a mirror only when the incident wave is even about it
  (`core.mirror.plane_wave_mirror_issue`). Half vs full runs agree to float32 round-off only: compare mid-pulse
  snapshots, not fields after the pulse left (`selftests_mirror3d`).
- Poles: the region box rule of `_pole_region` (one cell lo-side, a periodic axis from cell 0 spans the axis) is
  shared by `deposit_pole_work` and `definition._pole_box_cells` (the estimate): change them together. Losses: sigma_m loss is not counted.
- Backends (core sub file "Kernel twins"): every array from `engine.xp`; numba twins only for update_h / update_e /
  ohmic_loss (CPU case em3d_kernels_numba). The numpy twins cache coefficients by input identity
  (`kernels3d._host_cached`): a new writer of material_id / tables / face_id must also replace the tables, as
  `rebuild_materials` does.
- half_storage (`kernels3d` header, `engine.set_field_scale`, `physics3d.HALF_STORAGE_PEAK`): a new field reader
  converts with `engine.field_scale(component)`, a new writer in physical units refuses (`_require_single`); psi,
  poles, q_accum and DFT stay float32 / complex.

## Cylindrical sectors (symmetry 2.21; engine3d, kernels3d, selftests_cylindrical3d)
- Polar `Domain3D` (scripts only; mechanics, faces and the r_hi CPML approximation: `engine3d` header,
  `_check_cylindrical`). A new curl kernel takes the radial factor arrays (`engine._radial_h` / `_radial_e`; ones on
  Cartesian grids), never a Cartesian-only shortcut; a feature placed by Cartesian widths calls `_require_cartesian`
  (grep it for the refused list). Self-test em3d_cylindrical_sector_cavity (heavy).

## Subpixel smoothing (igor 1.5; subpixel_smoothing3d, geometry in model/subpixel_geometry)
- Refreshed by every `rebuild_materials` from the current tables; E consumers take it through `_face_args` (Engine
  bullet on E coefficients); mode ports through `mode_ports3d._smoothing_deltas`. Not smoothed: TFSF background, the
  stability check, sigma_e / mu / pole membership. Self-test: em3d part of `subpixel_smoothing_cpu_gpu`.

## Conformal PEC (solver_plan S7.6; conformal_pec3d, kernels3d conformal_h_3d)
- Method and stability floor: `conformal_pec3d` header; wiring: `physics3d.add_conformal_pec_region` (staircase
  cells at once) and `_apply_conformal_pec` (once at finalize). The floor's eigenvalue check ignores mixed staircase /
  conformal contacts and dielectrics. At the default courant 0.99 ~70 % of cut faces are floored (TM101 9.5e-3 vs
  staircase 5.2e-2): full accuracy needs a smaller courant (`_conformal_pec_note`). numpy twin, no numba; memory:
  `boundary_locator.conformal_memory_estimate`. Self-test em3d_conformal_pec_cavity (heavy).

## Frequency-domain eigensolver em_fdfd_3d (operators3d, fdfd3d, renderer_fdfd3d, definition_fdfd3d; igor3d_plan)
- Same grid / faces / walls / mirrors as em_fdtd_3d: a ring-down and the FDFD eigenvalues agree through asin(dt omega
  / 2) / (pi dt) (em3d_fdfd_vs_fdtd). `operators3d` (host scipy, header) builds the curls from
  `physics.em.operators.difference_matrix_1d` and the live E unknowns: two-PEC-cell faces pinned, the normal E of the
  last row of a non-periodic axis dropped (kept, it adds spurious 2D modes), symmetric mirror = normal E / tangential H
  pinned + ghost dropped, antisymmetric = ghost tangential E folded in with -1, the plane's normal E kept with half its
  dual volume. Any 3D host curl-curl (conformal check, tests) goes through it, never a private assembly.
- `FDFDPhysics3D`: materials read from the MaterialGrid (no uint16 index), face eps under `kernels3d.pec_face_values`;
  A = -K + k_t^2 diag(eps_face), solved by `solve_resonances(spectral_map='squared_cayley')`: K's gradient null space
  would rank ahead of every k > sqrt(2) k_t with plain shift-invert. Fields stay compressed until `select_resonance`
  (complex64 full-grid E incl. ghost images, H at the complex omega). SuperLU on
  `operators3d.nested_dissection_ordering` (measured ~equal to MMD so far). `ARPACK_TOLERANCE_FLOOR` (1e-10, cpml
  models only: closed ones keep machine precision for Q = inf) and `ARNOLDI_BASIS_PER_WANTED` (5 k unless
  subspace_size) belong together (the floor alone was worse); memory / time estimates use the same basis.
- Estimates: `CURL_CURL_LU_SCALE` (`SolverWork.factor_scale`: ~7x the 7-point curve, n^0.3 above 2.5e4) and with cpml
  `applications_per_basis` 0.22 x cells along the box, both fitted on 18^3-26^3 at n = 3 x cells (the time recipe and
  the size rule keep that n). Memory (A5): n = `live_unknowns_estimate(mesh, config)` (live_unknowns_3d's layer rules
  without PEC objects, a product per component: a change of those rules changes both; 3 x cells read 11 % high on the
  Mie example), and the factor = the counted box nested dissection of the whole grid (`_box_fill` ->
  `sparse_direct.box_dissection_entries(shape, n / cells)`, cached): `host_counted_lu_bytes` for SuperLU (it factors in
  that grid ND order) and MUMPS / PARDISO's scaled item, `FactorFill` for cuDSS (CUDSS_FILL_3D read 48 / 76 % low) from
  the same boxes with `yee_faces=True`: cuDSS orders unknowns itself, and the normal E of a plane does not couple
  across it, so its separators are the two tangential components (the cell-wise count read a 240 x 33 x 29 grid 54 %
  high; the host items keep the cell-wise count, SuperLU eliminates whole cells). No
  driven em_fdfd_3d factor exists (eigenfrequency only). Size rule
  `sparse_direct.vector_3d_size_issue`: uncalibrated. Factorization cudss skips both
  (`definition_helpers.eigen_factorization`, `eigen_solver.factorization`) and the ND ordering. `FDFDSystem3D.symmetrizer`
  D makes D A complex symmetric (None with a Bloch phase) -> cuDSS LDL^T; the eigen role accepts cudss_mixed, and
  Automatic is cudss_mixed on the gpu with nvmath (`auto_factorizations` tensor / tetrahedra; SuperLU factored the Mie
  examples in 35-48 s), SuperLU on the cpu or after a cuDSS refusal (tetrahedra: bounded, see "Size rule" below).
  Factorization mumps / pardiso (+ _mixed; solvers.host_direct, opt-in, either backend): the same LDL^T of D A on the
  host, no ND ordering (`elimination_order` None for every `LDLT_FACTORIZATIONS` entry: the libraries order
  themselves), but SuperLU's `vector_3d_size_issue` still applies (uncalibrated for them) plus the library's own
  memory check; a mixed one takes the polished path of `solve_resonances` (host solves with MIXED_EIGS_REFINEMENTS)
  and the time estimate's MIXED_APPLICATIONS_PER_BASIS (measured on cudss_mixed only); with cpml a double LDL^T
  (cudss / mumps / pardiso) is polished too (ARPACK 1e-6 instead of ARPACK_TOLERANCE_FLOOR, no refinement steps) and
  takes the same estimate branch. Host factor items:
  `memory_estimates.host_factor_items` on the SuperLU heuristic (tensor and tetrahedra, counted fill included).
- Subpixel: every assembly adds `fdfd3d.smoothing_deltas` (background eps only; none at PEC) to the face eps; FDFD
  reads MaterialGrid values, FDTD its index tables: they agree unless the FDTD re-quantises. The subpixel memory
  estimate still counts the FDTD face index (small overestimate).
- CPML: complex stretch frozen at omega_t (still reflectionless, leaky poles keep their Q); PML filter =
  PML_RESONANCE_FRACTION on the energy weights.
- Region filter: resonance_table selection + min_region_fraction -> `set_resonance_region` (tensor: host bool cells,
  each owning its +x / +y / +z E faces; a mirror-plane face belongs to the ghost, so a region on a mirror reads slightly
  low; tetrahedra: per element, via `FEMPhysics3D.resonance_region`). 'all' raises at apply. Not in a self-test.
- Snapshot solutions: em_fdfd_3d (tensor and tetrahedra) has no SolutionStateSolver: eigenfrequency only, its
  snapshots are select_resonance selections; a driven solve (per wavelength / iteration) must add the hook.
- Not yet: conformal PEC, driven solves. Renderer: raw `<c>_re / _im / _abs` at their samples, centre moduli via
  `yee_grid.lower_face_partner` (Bloch: wrapped face x e^{+j k L}); e_abs primary.
- Self-tests `selftests_fdfd3d` (CPU operator cases; model runs on gpu with the cudss_mixed eigen node of
  `_eigen_solvers`, only em3d_fdfd_eigen_pec_box also on the cpu: keep heavy cases and examples off the cpu); templates
  pec_box_eigenmodes_3d, nanobeam_mirror_cell_bands_3d (Automatic). Open cavity = example
  examples/em/fdfd_mie_sphere_resonance_3d (gpu only): smoothed 1.0e-2 blue, staircase 2.8e-3 red: the subpixel
  average is diagonal with in-plane normals per z layer, biased low on oblique faces, so it need not beat the staircase
  on curved 3D surfaces.
- Band diagrams: a parameter sweep over bloch_k<axis> (fresh factor per k). A uniform mesh has ceil(size / dx) cells:
  make the period whole cells or the Bloch period is the meshed one. Bands above the light line are leaky.
- Cost: direct factor time ~ n^2; host SuperLU > 10 min at ~1e5 unknowns, gpu cudss_mixed ~1 s at 2.2e5. Study
  convergence by perturbing downward (thinner PML, coarser cells), never by larger grids.
- Mirror sectors of a sphere: m_z (E azimuthal about z) is antisymmetric about x = 0, y = 0, symmetric about z = 0; p_z
  symmetric about x / y, antisymmetric about z. Put the sphere centre on the window corner.

## Tetrahedral Nedelec eigen path (fem3d; igor3d_plan section 7)
- `FEMPhysics3D(mesh, BoundaryConfig3D, materials, background, element_order 1 / 2, wavelength, eigen_solver)` on a
  `fem.mesh3d.TetrahedronMesh` (scripts): `solve_eigenfrequencies` / `select_resonance` as em_fdfd_3d; `invalidate()`
  re-assembles at the next solve. `fields_located` keeps its read-only E / H per located array pair (identity of the
  read-only arrays of `TetrahedronDomain.locate`, up to FIELD_CACHE_POINTS) until the next `select_resonance`, so the
  renderer's per-component reads of a plane evaluate the Nedelec basis once; writable located arrays are not cached.
- Model path: em_fdfd_3d on 'tetrahedra' creates `fem_solver3d.FEMResonancePhysics3D` (materials per element from the
  MaterialGrid at every assembly; PEC / pole masks from `_apply_pec` / `write_dispersive_regions_3d`, which branch on
  `ctx.domain.grid_kind`; 'discretization' = Nedelec degree, quadratic default) and `renderer_fem3d.FEMRenderer3D`
  (NaN outside the mesh). Periodic faces need mesher 'builtin' without swept x / y sides (Nedelec pairs need the same
  node order both sides; gmsh periodic surfaces do not keep it; swept z sides do). Size rule on `nedelec_unknowns`;
  estimates `_tetrahedra_memory_estimate` (Bloch: cuDSS copy = whole LU matrix; as the tensor estimate, the own eigen
  factor's pool copies are EIGEN_PHASE transient items and the factor an external item via `_cudss_eigen_items`,
  the em sub file's "Memory estimators" phases) and the Yee recipe at Yee-equivalent unknowns (uncalibrated) for the
  run time. Factor size (SuperLU and cuDSS): on a built mesh `nedelec_factor_entries` = `MeshDissection.nedelec_pairs`
  (dof sites: edges, + faces at degree 2) x (dofs per site)^2, a FactorFill / `host_counted_lu_bytes`; size-based counts
  only fall back to the n^(4/3) law at the Yee-equivalent unknowns (~5x high at degree 2: the 3.5 x unknowns inside
  n^(4/3)). The host SuperLU here orders by MMD, not the counted dissection: compare factor_host_bytes in the log. Field derived values via the Session's `_evaluate_tetrahedra`: mode volumes need a point
  object when the selection touches a mirror face.
- Formulation (fem3d header): unconjugated Galerkin, K and M complex symmetric with PML and loss
  (`solve_resonances(symmetric=True)`); B = -M sparse (`_PencilMass`). Equilibration (|diag K| + k_t^2 |diag M|)^-1/2,
  never |diag A|. K, M, W share one pattern (element_blocks_csr, window + PML parts), so A = k_t^2 M - K is a
  values-only sum (`_shifted_operator`; Bloch P^H A P falls back to pattern_combination); window W blocks are the M
  blocks' real view when eps is real and >= 0 there.
- PML = the mesh's node extent beyond `mesh.window` per face, must match the cpml faces exactly (`pml_shell_3d`
  raises); sigma_max at width / cpml_npml (cpml_npml only sets grading strength). PML element rules are 2 degrees higher.
- Walls fix tangential dofs on the bounding-box side (`face_tags`); cpml faces: the shell's outer side. A mirror plane
  is the window face itself (no ghost).
- Periodic / Bloch (`bloch_identification`): `periodic_dof_partners` REQUIRES pairs that keep node order (builtin Kuhn
  does; gmsh would need renumbering or dof signs). Axes chain (corner dofs get both phases), period = node extent (no
  shell on a periodic axis). A_free = P^H A P; `symmetric` False with a complex phase -> cuDSS LU, no Rayleigh polish.
  A PEC element or wall on either side pins the master.
- Size rule: host SuperLU (and MUMPS / PARDISO) under `vector_3d_size_issue(n x YEE_EQUIVALENT_UNKNOWNS)`
  (uncalibrated). A4: SuperLU (the cpu Automatic) also warns on a built mesh above `NEDELEC_SUPERLU_SLOW_BYTES` of
  counted fill (`host_counted_lu_bytes`; tens of seconds: 44k unknowns took 47.5 s), and the gpu 'auto' runtime gets
  `fallback_refusal` (`_nedelec_fallback_refusal`, counted on the TetrahedronMesh only when a cuDSS refusal would fall
  back): above that fill the refusal raises with the remedy (gpu cudss_mixed, mumps / pardiso, coarser mesh, mirrors)
  instead of a silent minutes-long SuperLU. Uncalibrated (the threshold is a guess between 0.2 and 0.6 GiB).
- Self-tests `selftests_fem3d` (tolerances first estimates); Mie sphere example examples/em/fem_mie_sphere_resonance_3d
  (gpu).
- Open: cudss_mixed refinements per caller (belongs in physics/em/eigenfrequency.py), periodic gmsh meshes, driven
  solves, interface faces.

## em_fdtd_3d on tetrahedra / cuDSS: decision and plan (not built)
- em_fdtd_3d stays tensor-only (Yee); no tetrahedral time-domain solver, no cuDSS in the stepper.
- DGTD would be a second solver (not a mesh-kind branch), L-XL (~6-10 kLOC): every source / monitor redone on faces,
  local time stepping, DG CFS-PML, ADE poles; ~1 KB per tet (p = 3) vs Yee 26 B per cell. Gate: a reference case Yee +
  subpixel + conformal PEC on a graded mesh cannot reach within device memory (curved metal / plasmonic gaps << cell);
  then start at p = 2-3 without LTS.
- FETD (Newmark) rejected: a step is a bandwidth-bound solve (~10-30 ms at 2e5 unknowns), 1e4 steps take minutes vs
  ~1 s Yee, large dt is dispersive. Tet-conforming frequency-domain answers belong in the em_fdfd_3d FEM path (next: a
  driven solve, LDL^T on cuDSS).
- Cheaper Yee options by value / cost: (1) 3D cut-cell pole weights (as 2D `physics/em/conformal_poles`), S-M; (2)
  full-tensor Kottke averaging, M; (3) subgridding, L (late-time instability risk).
- cuDSS for mode ports deferred: `solve_port_modes` uses the default ArpackShiftInvertSolver (the type has no 'eigen'
  role); complex nonsymmetric LU, planes <= ~2e5 unknowns where 2D SuperLU takes seconds (gain expected small,
  unmeasured). Wait for cuDSS plan reuse across monitor wavelengths and a run where n_wavelengths x factor seconds
  dominate. Then: `eigen_role(('time_dependent', 'frequency_domain'))` (shows on every em_fdtd_3d model: gate it),
  `ctx.solvers(...)['eigen']` into ModeSource3D / ModeMonitor3D, skip the host `size_issue_for`, device items as
  em_eme_3d definition3d; measure `info['factor_seconds']` per wavelength at 1e5 / 2e5.

## Sources, monitors, results
- Dipole (`sources3d`): moment I l [A m] on the E face of its axis nearest the point, J = I l / V (V = dual length x
  two cell widths), weight = live face cb x I l / V on the device (`calibrate` after each rebuild); injected at t + dt/2.
- Point probe = core FieldProbe with iz (H dt/2 older). resonance_monitor = the shared row over
  `em_resonance_probe` (hx / hy / hz delayed 0.5 step); em3d_resonance_pec_box: pec walls at the centres of the
  outermost cells, L = (n - 1) dx.
- Flux (`PlaneFlux3D`, `FluxBox3D`): planes on cell edges, E averaged over the two layers across the plane, trapezoid
  weights along face-sampled edges; spectra phased at the E time, cos(omega dt/2) on readout (2D convention); a plane
  needs a cell on both sides (edge 1..n-1). A periodic tangential axis spanning the grid wraps the face weights (else
  half a cell of area is lost). power_per_cw in W.
- DFT layer (`DFTPlane3D`): E averaged to the centres of one window z layer, 'dft_fields' with an extra 'z'.
- Harmonic FDTD (`harmonic3d`): window DFT of E at their faces (24 B per window cell and wavelength), centre-averaged
  on read; each component also accumulates the lo face row of window cell 0 along its own axis when the window starts
  inside the grid (padding / mirror ghost: a symmetric mirror's pinned normal E halves that row, as in the renderer and
  DFTPlane3D), else the periodic wrap or its own face; absorbed_power_harmonic = 1/2 (sigma_e - omega eps0 Im chi) |E|^2 (isotropic, 0 on PEC).
- absorbed_power (renderer) = current q_accum window, else the last, else zeros (after `deposit_pole_work`).
- Renderer: `device_plane(component, axis, index)` touches only one plane (harmonic phasors cut along their own axis
  average two face planes); `device_field` allocates window-sized temporaries.

## Definition, self-test
- `definition.py`: bc_<face> over FACES_3D (cpml / pec / pmc / periodic / symmetric / antisymmetric), bloch_k* (memory
  estimate doubles fields, CPML, poles), courant_factor (3D CFL); materials = 'modes' rows of `em_material_params`
  (the builder skips unallocated optional arrays); slots eps_xx_update / eps_yy_update raise at inject when their array
  is unallocated (heat_em_thermo_optic writes only `permittivity_arrays`). Feature validates mirror every build-time
  placement refusal on the MeshEstimate (`definition_sources3d.window_axes` / `nearest_window_edge` /
  `plane_edge_message` / `box_edge_issues`): a new refusal in physics3d gets its validate twin. pec_domain on 'all' is a
  validation error.
- Memory: fields 24 B (12 half_storage) + index 2 B per cell, CPML slabs (cpml faces only), pole boxes, q_accum 8
  B/cell when tracking; build extra 12 B/cell when every array is uniform (`_materials_vary` false), else 40 B/cell: the
  ledger warns when a new index path changes these. `_time_estimate` counts launches (`_FIELD_LAUNCH_WEIGHT`, 2 Yee
  launches per field set, CPML fused in: none of its own, `_FEATURE_LAUNCHES`) and on cpu without numba
  `_NUMPY_FIELD_PASSES` / `_NUMPY_CPML_PASSES` (the numpy twins' layer passes): a new per-step kernel or numpy twin
  pass updates these counts.
- Self-test helpers in `selftests.py`: `DEFAULT_COURANT`, `cubic_time_step`, `dipole_selftest_spec()`,
  `dipole_flux_errors`: a new 3D case reuses them. Durations sit just above the validation bound 2 t0 + diagonal x
  n_max / C0 (warning only); templates keep longer ones.
- Self-test files: `selftests_sources3d`, `selftests_materials.py` (incl. em3d_material_requantisation),
  `selftests_backend.py` (heavy, `backend_parity_case`: cases whose measured is an error near 0 or a count compare a
  physical value instead, `_case(name, parity_value)`). em3d_dipole_radiated_power_graded raises unless the build shows
  the grading; em3d_far_field_dipole's half_storage variant raises unless its stepper node records half_storage.

## Plane wave, mode ports, far field (tfsf3d, mode_ports3d, near_to_far3d, definition_sources3d)
- Source protocol: an object in `EMPhysics3D.sources` with `inject(fields, t)` (t + dt/2, after the E corrections),
  `calibrate(engine)` (after every rebuild), optional `inject_magnetic(fields, t)` (after H update + CPML, before the
  pole drive) and `warnings` (finalize copies them to `finalize_warnings`, later recalibrations to
  `runtime_warnings`); a `temporal` joins the shared S(omega). Sources / ports read materials through the engine
  tables (re-quantised values), never the MaterialGrid. Monitors' `calibrate` runs after the sources' (mode monitors
  solve there).
- TFSF (`PlaneWaveSource3D`): axis-aligned, any direction / transverse polarisation, generic patch kernel (module
  header). Open sides: a periodic axis, with both transverse axes periodic the downstream face on the window edge, and
  a transverse face on a mirror of the wave's parity. Incident = 1D aux line (em/sources `_tfsf_aux_h / _e_graded`,
  negated coefficients) with Mur ends matched to the numerical phase velocity at f0 (plain S' = S echoes ~3e-3);
  background read at calibrate from the cell before the upstream face (PEC raises), scalar db / cb (each E patch's cb
  with its own component's eps), so box faces must sit in one uniform lossless medium (calibrate warns). Graded meshes:
  the aux line takes the box cells' widths plus uniform margins (`_aux_line_terms(..., index)`).
- Mode ports (`ModeSource3D`, `ModeMonitor3D`): port frame (u, v, d) cyclic = (x, y, z), `_plane` transposes for d = 1.
  One cell layer's materials from the tables (PEC = table_eps_z < 0; PEC cells keep MaterialGrid values for the face
  means), `_PortBoundary`, solved at the numerical frequency (beta_d = (2/D) arcsin(K D/2)). Source corrections: one
  `mode_port_pair_3d` launch per kind (both components share a layer), args cached per field set and dropped at every
  calibrate, so the coefficient / incident planes stay contiguous (u, v) arrays and fields are rewritten in place; amplitude sqrt(P / (cos(omega dt/2) cos(beta_d D/2))) so a
  flux plane reads P at f0. Modes solved until the first step only; within one `_refresh_materials` round ports on
  layers with equal materials (digest of the CrossSectionMaterials arrays), the same normal axis, omega (12 digits) and
  mode count share one solve (`physics.port_mode_solves`): the solve objects are shared, never mutate them. Monitor = `PlaneFlux3D` + per-frequency
  `mode_overlap`, a+- = (X / c +- Y) / (2 N) unconjugated; fraction_* needs exactly one ModeSource3D.
- Near-to-far (`NearToFar3D`): a `FluxBox3D`'s pairs give J / M on their own sample grids; result 'far_field' as a cut
  plane ('power' a.u. per sr without the 1/2 and the flux cos(omega dt/2)), so 2D consumers keep working. Flux boxes and
  far fields of the same bounds / frequencies share one FluxBox3D (`_shared_flux_box`); a NearToFar3D is not in
  `physics.monitors`, so code walking monitors meets the box once and never the far field.
- Self-tests `selftests_sources3d` (Fresnel / Drude film vs the discrete Yee forms, TFSF leakage < 1e-6 on graded and
  closed boxes, mode-source transmission, far-field dipole).
- Metal-guide ports: keep the pulse spectrum off the mode cutoff (zero group velocity rings ~t^-1/2, truncating the
  monitors' DFT: ripple in forward power; pec_waveguide_te10_ports_3d at bandwidth 0.4: T / R off by 0.08).
- Open: ports touching pec / pmc faces are approximate (engine updates the last E face row, the solver drops it); port
  PML uses one grading cell for both axes; no mode re-solve after material drift; no phase_* in the 3D mode monitor; far
  field assumes vacuum outside the box.
