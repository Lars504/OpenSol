# open_sol/physics/em3d — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first, incl. "3D models", and the core sub file's "3D
contracts"). em_fdtd_3d (B2) and the eigensolver em_fdfd_3d (tensor Yee and tetrahedral Nedelec); 2D EM rules:
`open_sol/physics/em/sub_structure_conventions.md`.

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
- Parked material arrays (model path, gpu): when nothing rewrites them during the run (`definition.
  _parked_material_arrays`: no coupling into the physics, every em material property constant or x / y / z of the
  parameters, no other physics on the mesh registering the name) the first index build parks them on the host
  (`MaterialGrid.park`) and only then allocates the fields (`Fields3D(allocate=False)`): the build peak never holds
  both (~12-24 B per cell less). Readers take `MaterialGrid.device_array(name, index)` (renderer eps_r window,
  harmonic energy per x-slab part (`harmonic3d.ENERGY_CHUNK_CELLS`, it runs at the run peak), state()['eps_r']), mode
  ports upload their layer plane, a rebuild reads parked arrays chunk by chunk (`engine3d._UploadedChunks`:
  material_index3d reads inputs only as `array[chunk]`, keep it so; at most two chunks alive, the estimate's term).
  A new reader of a MaterialGrid array of this engine goes through those, or the rule must exclude its case;
  `PhysicsType.parked_material_arrays` makes meshing leave the arrays to this type's estimate.
- E coefficients are computed per step in the kernels (PEC = the last id, eps -1; subpixel faces via
  `_face_args`): a new consumer (TFSF, mode source, dipole weight; the pole corrections in their kernel) calls
  `face_coefficients` / `engine.face_cb_into` (host: `kernels3d.pec_face_values`), never stores per-cell arrays or its
  own face mean. An
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
  shared by `deposit_pole_work` and `definition._pole_box_cells` (the estimate): change them together. A box stores
  only the pole states (+ face work while accumulating) and a uint8 membership box one row longer per axis: the
  face weights (`kernels3d.pole_face_weights`) and cb are computed in `pole_drive_3d` / `pole_correct_3d` (one launch
  per pole and field set; checked against the stored path by em3d_pole_kernels, the only Lorentz case).
- Losses: q_accum and the last window live on the loss box `engine.q_box` (ohmic box union pole windows, grown never
  shrunk by `_fit_q_box`, which every rebuild, enable and add_pole call); the last window is a host (pinned) copy
  filled asynchronously per window reset, read only through `host_last_q_accum` (the event wait), as the 2D engines.
  Readers go through `q_selection` / `q_grid` (full-grid copies: state()['q_accum'], take_absorbed_power), never
  index q_accum by grid slices; a new writer stays inside q_box. sigma_m loss is not
  counted.
- Backends (core sub file "Kernel twins"): every array from `engine.xp`; numba twins only for update_h / update_e /
  ohmic_loss (CPU case em3d_kernels_numba). The numpy twins cache coefficients by input identity
  (`kernels3d._host_cached`): a new writer of material_id / tables / face_id must also replace the tables, as
  `rebuild_materials` does.
- half_storage (`kernels3d` header, `engine.set_field_scale`, `physics3d.HALF_STORAGE_PEAK`): a new field reader
  converts with `engine.field_scale(component)`, a new writer in physical units refuses (`_require_single`); psi,
  poles, q_accum and DFT stay float32 / complex.

## Cylindrical sectors (engine3d, kernels3d, selftests_cylindrical3d)
- Polar `Domain3D` (scripts only; mechanics, faces and the r_hi CPML approximation: `engine3d` header,
  `_check_cylindrical`). A new curl kernel takes the radial factor arrays (`engine._radial_h` / `_radial_e`; ones on
  Cartesian grids), never a Cartesian-only shortcut; a feature placed by Cartesian widths calls `_require_cartesian`
  (grep it for the refused list). Self-test em3d_cylindrical_sector_cavity (heavy).

## Subpixel smoothing (subpixel_smoothing3d, geometry in model/subpixel_geometry)
- Refreshed by every `rebuild_materials` from the current tables; E consumers take it through `_face_args` (Engine
  bullet on E coefficients); mode ports through `mode_ports3d._smoothing_deltas`. Only the uint16 face index stays on
  the device: the geometry is a host copy (`SubpixelSmoothing3D.geometry`, numpy) uploaded one face kind per refresh
  (estimate: `model.subpixel_geometry._memory_estimate_3d`). Not smoothed: TFSF background, the
  stability check, sigma_e / mu / pole membership. Self-test: em3d part of `subpixel_smoothing_cpu_gpu`.

## Conformal PEC (conformal_pec3d, kernels3d conformal_h_3d)
- Method and stability floor: `conformal_pec3d` header; wiring: `physics3d.add_conformal_pec_region` (staircase
  cells at once) and `_apply_conformal_pec` (once at finalize). The floor's eigenvalue check ignores mixed staircase /
  conformal contacts and dielectrics. At the default courant 0.99 ~70 % of cut faces are floored (TM101 9.5e-3 vs
  staircase 5.2e-2): full accuracy needs a smaller courant (`_conformal_pec_note`). numpy twin, no numba; memory:
  `boundary_locator.conformal_memory_estimate`. Self-test em3d_conformal_pec_cavity (heavy).

## Frequency-domain eigensolver em_fdfd_3d (operators3d, fdfd3d, fem3d, definition_fdfd3d)
- em_fdfd_3d reuses em_fdtd_3d's grid, faces, mirrors, PEC face rule and material vocabulary (fdfd3d /
  definition_fdfd3d import them from kernels3d / definition): a change of an FDTD rule changes the eigensolver too;
  ring-down and FDFD eigenvalues agree through asin(dt omega / 2) / (pi dt) (em3d_fdfd_vs_fdtd). Any 3D host
  curl-curl (conformal check, tests) goes through `operators3d`, never a private assembly.
- Feature applies shared with em_fdtd_3d (`_apply_pec`, `write_dispersive_regions_3d`) branch on
  `ctx.domain.grid_kind`: a shared apply handles 'tensor' and 'tetrahedra' (em_fdfd_3d on tetrahedra =
  `fem_solver3d.FEMResonancePhysics3D` around `fem3d.FEMPhysics3D`).
- Self-tests `selftests_fdfd3d` / `selftests_fem3d`: model runs on the gpu with cudss_mixed; keep heavy cases and
  examples off the cpu (host SuperLU takes minutes there).
- Device memory of a solve is solve_resonances' (em sub file "FDFD", eigenfrequency): the estimators book it through
  `resonance_search_items` with the storage the solve picks (`definition_fdfd3d._polished_search`, the memory items
  and the time recipe alike: an LDL^T without Bloch phase that is mixed, or double with a cpml face, is polished:
  on the tensor grid the filter's D v block; tetrahedra: lower triangles of a symmetric search, no D v; the basis
  stays complex128 either way, `_single_basis`: squared Cayley, em sub file),
  so a change of what fdfd3d / fem3d pass to it (symmetrizer, symmetric, a sparse mass) changes both estimators.
  Nothing else of em_fdfd_3d lives on the device: the tensor resonance fields (`FDFDPhysics3D.fields`, complex64) are
  host numpy on either backend, and `renderer_fdfd3d` does its math on the host per read window / plane and uploads
  the float32 result (tetrahedra likewise, `renderer_fem3d`). A new reader of them works on the host or uploads per
  read, never a kept device copy. Its MaterialGrid arrays are parked on the host at create (tensor grids, gpu model
  path: `definition_fdfd3d._parked_material_arrays`, the shared rule minus arrays whose state field a material
  expression or field_transfer reads): the assembly reads them through `to_host`, the renderer's eps_r window and
  `StaticEM3DContract.state()` (`em.materials.material_state`) through `device_array`; a new reader does too.
- Cost: direct factor time ~ n^2; host SuperLU > 10 min at ~1e5 unknowns, gpu cudss_mixed ~1 s at 2.2e5. Study
  convergence by perturbing downward (thinner PML, coarser cells), never by larger grids.
- Band diagrams: a parameter sweep over bloch_k<axis> (fresh factor per k). The Bloch period is the meshed one and a
  uniform mesh has ceil(size / dx) cells: make the period whole cells. Bands above the light line are leaky.
- Mirror sectors of a sphere (centre on the window corner): m_z (E azimuthal about z) is antisymmetric about x = 0,
  y = 0, symmetric about z = 0; p_z symmetric about x / y, antisymmetric about z.
- Open: driven solves (then a SolutionStateSolver hook, root "Snapshot solutions"), conformal PEC, periodic gmsh
  meshes (Nedelec pairs need equal node order), no self-test of a resonance_table region; uncalibrated: the size rule
  `vector_3d_size_issue` (also for MUMPS / PARDISO), `NEDELEC_SUPERLU_SLOW_BYTES` and the tetrahedral run time.

## em_fdtd_3d on tetrahedra / cuDSS: decision and plan (not built)
- em_fdtd_3d stays tensor-only (Yee); no tetrahedral time-domain solver, no cuDSS in the stepper.
- DGTD would be a second solver (not a mesh-kind branch), L-XL (~6-10 kLOC; ~1 KB per tet at p = 3 vs Yee 26 B per
  cell). Gate: a reference case that Yee + subpixel + conformal PEC on a graded mesh cannot fit in device memory; then
  p = 2-3 without local time stepping. FETD rejected (a solve per step: minutes vs ~1 s Yee). Tet-conforming
  frequency-domain answers belong in the em_fdfd_3d FEM path.
- Cheaper Yee options by value / cost: 3D cut-cell pole weights (as 2D `physics/em/conformal_poles`), S-M; full-tensor
  Kottke averaging, M; subgridding, L (late-time instability risk).
- Mode ports have no 'eigen' role (`solve_port_modes`: a private ArpackShiftInvertSolver; planes <= ~2e5 unknowns, 2D
  SuperLU takes seconds). Revisit when cuDSS plans are reused across monitor wavelengths and n_wavelengths x factor
  seconds dominate a run: then an `eigen_role(('time_dependent', 'frequency_domain'))` gated so it does not show on
  every em_fdtd_3d model, passed into ModeSource3D / ModeMonitor3D, device items as em_eme_3d definition3d.

## Sources, monitors, definition
- Source protocol: an object in `EMPhysics3D.sources` with `inject(fields, t)`, `calibrate(engine)` (after every
  material rebuild), optional `inject_magnetic(fields, t)` (step order: engine3d header) and `warnings` (finalize
  copies them to `finalize_warnings`, later recalibrations to `runtime_warnings`); a `temporal` joins the shared
  S(omega). Sources / ports read materials through the engine tables (re-quantised values), never the MaterialGrid.
  Monitors' `calibrate` runs after the sources' (mode monitors solve there).
- Feature validate mirrors every build-time placement refusal of physics3d on the MeshEstimate
  (`definition_sources3d.window_axes` / `nearest_window_edge` / `plane_edge_message` / `box_edge_issues`): a new
  refusal in physics3d gets its validate twin.
- A new per-step kernel, feature kernel or numpy-twin pass updates the run-time counts in `definition`
  (`_step_launches`, `_FEATURE_LAUNCHES`, `_NUMPY_FIELD_PASSES` / `_NUMPY_CPML_PASSES`).
- Monitor spectra (flux planes / boxes, far fields, mode monitors: every PlaneFlux3D) live in a
  `monitors3d.SpectrumStore` of the stepper's spectrum_precision (explicit_kernels opt-in setting, offered by
  `definition.EM3D_STEPPER_ROLE` only; EMPhysics3D passes it to every PlaneFlux3D / FluxBox3D / ModeMonitor3D). With
  mixed2 the device arrays hold only the current complex64 block, the totals are on the host: a reader goes through
  `host_pairs` / `store.host()`, never `to_host(pair['e_dft'])`; a writer launches into `store.arrays` with
  `store.kernels` and calls `store.sampled()` once per sample; a new spectral monitor exposes `spectrum_stores` (work
  counter `spectrum_folds`), books its arrays through `definition_sources3d.spectra_memory`
  (`memory_estimate_context`) and joins `definition._SPECTRUM_STORES` (fold time item).
- The mixed2 fold (async copy into pinned staging + zero fill on the current stream, event waited at the next fold or
  read) runs eagerly from `update`: spectral monitors stay outside the step graphs (it is a host-side decision per
  sample). The block length SPECTRUM_BLOCK_SAMPLES (64) follows from the error target at its constant; changing it
  changes the spectrum_precision help numbers (`solvers/definition.py`) and the em3d_spectra_mixed2_* tolerances.
- The complex64 box pair kernel `em.monitors._dft_accum_pair` (DFTPlane3D) averages in the field dtype F: float16
  arithmetic under half_storage (core sub file "Half storage" ptxas hazard); the mixed2 block kernels convert to float
  first. Open: no self-test of mixed2 on half_storage, on a Bloch k != 0 set, on the cpu backend or with a mode monitor
  (they share the PlaneFlux3D path the dipole / column cases test).
- em_fdtd_3d self-tests reuse the helpers of `selftests.py` (courant, time step, dipole spec, flux errors). Durations
  sit just above the validation bound 2 t0 + diagonal x n_max / C0 (warning only); templates keep longer ones.
- Metal-guide mode ports: keep the pulse spectrum off the mode cutoff (zero group velocity rings ~t^-1/2, truncating
  the monitors' DFT: ripple in forward power; pec_waveguide_te10_ports_3d at bandwidth 0.4: T / R off by 0.08).
- Open: ports touching pec / pmc faces are approximate (mode_ports3d header); port PML uses one grading cell for both
  axes; no mode re-solve after material drift; no phase_* in the 3D mode monitor; far field assumes vacuum outside
  the box.
