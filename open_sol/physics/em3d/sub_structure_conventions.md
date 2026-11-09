# open_sol/physics/em3d — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first, incl. "3D models", and the core sub file's "3D
contracts"). em_fdtd_3d (B2) and the eigensolver em_fdfd_3d; 2D EM rules: `open_sol/physics/em/sub_structure_conventions.md`.
`em_fdfd_3d` (3D only, family 'em', default instance 'em') registers right after em_fdtd_3d and shares its material
Params, Yee grid, faces and mirrors; eigenfrequency study only (host SuperLU 'eigen' role, own 3D size rule; CPML,
subpixel smoothing), pec_domain and resonance_table; no conformal PEC.

## Engine (engine3d, kernels3d, boundaries3d)
- CPML slab launches are cached per (slab, field set, ids of the material tables / subpixel face arrays, curl factor)
  (`FDTDEngine3D._cpml_launch`): tables are replaced only by rebuild_materials and the scale only by set_field_scale,
  which change the key; field / psi / coefficient arrays are rewritten in place, never reassigned.
- Contract members shared by the 3D EM physics live in physics3d (`EM3DSlots`: permittivity_arrays / inject_source;
  `StaticEM3DContract` for em_fdfd_3d; `complex_cell_materials`, `check_pole`); the Yee renderers subclass
  `renderer3d.YeeRenderer3D`; `mode_ports3d.AXIS_PERMITTIVITY` is the one E component -> permittivity array map;
  `boundaries3d.along` the one axis index helper.
- Layout = the A5 cross-section layout extended along z: E_x / E_y / E_z on the +x / +y / +z face of cell (i, j, k),
  H_x on the (+y, +z) edge, H_y (+x, +z), H_z (+x, +y). An E face's eps (own component) / sigma_e = width-weighted
  mean of its two cells (`yee_grid.face_shares`; subpixel overrides interface faces); an H edge's mu_r / sigma_m = its
  lower cell's (`edge_coefficients`, semi-implicit 1 / (1 + sigma_m dt / (2 mu)); any H-side consumer, e.g. TFSF /
  mode-port db, reads table_mu and table_sigma_m alike). Per axis 'e' rows = centres, 'h' rows = hi faces, so
  `physics.em.boundaries.cpml_axis_profiles` is reused via `boundaries3d.axis_cpml_profile` (an axis view, never a copy).
- Materials: MaterialGrid eps_r (= eps_zz) / sigma_e / mu_r always; eps_xx / eps_yy only when a material sets eps_xx /
  _yy / _zz / dndT_xx / _yy / _zz (else E_x / E_y read eps_r), sigma_m only when set (`optional_material_schema`):
  never assume the arrays exist (`engine.material_arrays`, `physics.permittivity_arrays`). `rebuild_materials` builds a
  uint16 `material_id` + float32 tables per id (table_eps_x / _y / _z, table_sigma, table_mu, table_sigma_m) via
  `material_index3d.build_material_index` in x-slab chunks (several host syncs; only at finalize and after a material
  change: dirty flag checked in `step`).
- Index fallback (thermo-optic, GRIN): > 65534 distinct values (or joint count) re-quantises instead of raising:
  occupied bins of the smallest fitting width (geometric bisection to 10 %, previous width as hint), representative =
  member mean, so a one-value bin (air 1.0) stays exact; error <= bin width (`engine.quantisation_errors`, note logged
  once per array). Joint combinations ranked stage by stage (no int64 overflow, no per-cell storage); ValueError only
  when halving every > 256-level array's budget still does not fit. Cost: ~10 full-grid passes per continuous array
  per material change.
- PEC = the last id (table eps -1): a face between two PEC cells is pinned (ca = cb = 0), PEC / other takes the other
  cell. ca / cb are computed per step in the kernels (`face_coefficients`); any new coefficient consumer (TFSF, mode
  source) calls the same device helper or `engine.face_cb_into`, never stores per-cell arrays. Host form:
  `kernels3d.pec_face_values` (numpy twin and em_fdfd_3d face eps call it).
- Edges: non-periodic axes get ih[n-1] = 0 and read cell 0's missing lo H as 0, so the grid edge is PMC half a cell
  beyond the outermost centres: bc 'pmc' is a no-op, 'pec' gives the outermost window layer the PEC id. A dipole in the
  wall layer of a pec face parallel to it is refused (validate and add_dipole); normal to it, it may sit next to it.
  Periodic axes wrap in the kernels and the pole boxes.
- Bloch k != 0 (symmetry 2.8): `BoundaryConfig3D(bloch_k=)` (bloch_kx / _ky / _kz on periodic axes), F(x + L) = F(x)
  e^{-j k L}, L = summed grid widths. The engine then holds `fields.imag`, `cpml_imag` and per pole `imag_drude` /
  `imag_lorentz`; both sets step with the same kernels whose wrap reads mix in the partner set (`kernels3d._WRAP_LOAD`;
  real-only passes partner = itself, c 1, s 0: bit-identical). Walls / mirror rows act on both sets; sources drive the
  real set only; PlaneFlux3D, DFTPlane3D and the harmonic accumulator add j ph F_im (`em.monitors._field_sets`), the
  harmonic shutoff energy sums both; probes, renderer, state and couplings read the real set. A centre mean of cell 0
  on a phased axis needs its wrapped lo face x e^{+j k L} (DFTPlane3D accumulates cell 0's own and wrapped faces apart,
  combined in `spectrum()`; else ~10 % off in that row; harmonic3d wraps its whole-axis spectrum the same way; missing
  still: a k = 0 layer on a phased z axis). q_accum and the TFSF plane wave raise with k != 0 (validation errors first); mode ports solve the
  Bloch mode of the port plane's periodic transverse axes (`mode_ports3d.port_boundary`; the mode source drives the
  real set, so DFT monitors read the +k mode). Self-tests in `selftests_bloch3d`: em3d_bloch_empty_lattice,
  em3d_bloch_array_monitors (pins the e^{-j k L} sign; its flux plane opposite the dipole needs an odd x cell count),
  em3d_bloch_port_modes; template photonic_crystal_slab_bands_3d. Open: oblique 3D TFSF, a Bloch mode-source model
  run, harmonic3d with k != 0.
- Mirrors (symmetry 2.5, E parity as 2D): one ghost cell (`MIRROR_GHOST_CELLS`; core.mesh copies the edge width on x /
  y, `MeshConfig.z_cell_spacings(ghost_faces=)` on graded z), so the plane is the window edge. The ghost / inner /
  plane-face rows and the pec wall layers come only from `boundaries3d.mirror_planes` / `pec_wall_layers` (engine3d and
  operators3d both read them). `engine3d._mirror_rows`:
  symmetric pins the plane row's tangential H and normal E (after inject_magnetic / after the E sources),
  antisymmetric writes the ghost's tangential E = -inner (`cp.negative(..., out=)`). Mode ports map mirrors onto the
  cross-section mirrors (powers of the full structure); flux planes / boxes reaching a mirror double
  (`monitors3d.mirror_factor`); the far field unfolds by images (`NearToFar3D(mirrors=)`, signs from
  `core.mirror.image_sign`, box faces on a plane dropped; mirrors on both faces of an axis or any periodic axis:
  ValueError + validation error). A TFSF box face reaching a transverse mirror about which the incident wave is even
  (`core.mirror.plane_wave_mirror_issue`: symmetric for a plane normal to the third axis, antisymmetric for one normal
  to the polarisation; else ValueError + validation error) opens through the ghost layer. Far-field results carry
  complex `far_field_amplitude` (amplitude_re / _im) for parity-sector combination. Self-tests `selftests_mirror3d`
  (em3d_mirror_dipole_pair, em3d_tfsf_mirror): half and full runs agree only to float32 round-off (FMA rounding), so a
  field compared after the pulse has left can miss 1e-4 by orders: compare mid-pulse snapshots.
- CPML: psi slabs only inside the layers, kappa folded into the 1D curl arrays, psi correction with live db / cb after
  the Yee kernel (as 2D). Sigma grading cell size per axis = the larger outermost width of that axis.
- Poles (`add_pole`): one `_PoleRegion` per material region (keyed by its cropped mask, kept alive by the physics) on
  the region box extended one cell lo-side; per component a face weight (membership share) and a live cb box (refreshed
  every rebuild). The last face follows `face_shares`: on a periodic axis a region starting at cell 0 gets a box
  spanning the axis (membership[n] = membership[0]); non-periodic: membership[n] = membership[n-1];
  `deposit_pole_work` mirrors both, the estimator sizes the box from `MaterialRegionInfo.box_bounds`. Stability:
  `stability_issue` (finest widths, region pole terms, min of the three eps tables) at finalize.
- Losses (EM loss protocol; `enable_q_accum`: q_accum full grid + last window 8 B/cell at first enable): ohmic
  `ohmic_loss_3d` over the box of sigma_e > 0 cells after the sources, each face's sigma E^2 V split to its two cells by
  half-cell conductance (exact except next to PEC). Poles: time-centred face work in per-region `work` boxes, deposited
  into the dispersive cells only by `deposit_pole_work` (take, reset, state(), renderer). sigma_m loss is not counted.
- Kernel flat indices are `long long` (`idx_t`): `long` is 32 bit on Windows.
- Backends (core sub file "Kernel twins"): every em3d kernel is a `core.kernels` twin (field kernels per precision
  'em3d.<kernel>.<precision>'; 2D DFT / TFSF aux / port / pole correction twins shared), every array from `engine.xp`
  (incl. CPML slabs `cpml_slabs(..., xp)`, monitor spectra `xp=`, port planes, harmonic accumulators); host reads via
  `to_host`; material_index3d / subpixel_smoothing3d follow their inputs (`array_module_of`). Host twins compute in
  float32 (cpu refuses half_storage). Numba twins for update_h / update_e / cpml_h / cpml_e / ohmic_loss (CPU case
  em3d_kernels_numba): the no-smoothing face id placeholder is (1, 1, 1) (numba types fid[i, j, k] even with use_face
  0); they compile with `cache=True` (an uncached first cpu run compiles inside its study: 17x step-time miss; a
  "Cannot cache" NumbaWarning = a closure value stopped pickling). The numpy twins cache per-step coefficients in
  `kernels3d._host_cached`, keyed by input array identity (weak refs; ~32 B/cell host, not estimated): valid only
  because material_id / tables / face_id change solely in `rebuild_materials`, which writes new table arrays, so a new
  writer must also replace the tables. A new kernel gets its numpy twin in the same edit; a new allocation takes
  `engine.xp`.
- Stepper precision (`FDTDEngine3D(precision=)`): field kernels compiled per precision (`kernels3d.field_kernels`,
  FLOAD / FSTORE). half_storage stores the six components as float16 E / e_scale, H / h_scale (h_scale = e_scale /
  eta0); e_scale (power of two, `set_field_scale`, fixed before the first step) puts the sources' `peak_field()` at
  HALF_STORAGE_PEAK = 16; curl factors e_to_h = eta0, h_to_e = 1 / eta0 are kernel scalars (1 in single). Every
  reader converts (`field_scale(component)`; flux `host_pairs`, `DFTPlane3D.intensity`, FieldProbe `value_scale` via
  `EMPhysics3D._set_field_scale`, renderer, ohmic scale, dipole weight, `state()` Mapping of float32 copies). Writers in
  physical units (TFSF, mode ports, poles, harmonic3d) refuse half_storage (`_require_single`, add_pole; definition
  feature precisions / validate): a new field reader or writer must scale or refuse. psi, poles, q_accum, DFT stay
  float32 / complex. float16 writes never use cupy's float16 kernels (unresolved `__half` on some CUDA / cupy pairs):
  dipole add via `half_bits(target)` + `STORED_ACCESS`, mirror rows on the uint16 bits (core.half_storage).

## Cylindrical sectors (symmetry 2.21; engine3d, kernels3d, selftests_cylindrical3d)
- A polar `Domain3D` (x = r - radius_min, y = phi [rad]) is an (r, phi, z) sector; scripts only (EMPhysics3D lists
  'polar', the PhysicsType stays Cartesian). Components keep _x / _y names (= r / phi). The integral-form curls are
  per-i 1D factors of the update kernels from `core.metric.yee_radii` (kernels3d header); Cartesian passes ones
  (exact), so a new curl kernel takes the same arrays (`engine._radial_h` / `_radial_e`, CPML `_cpml_weights[(axis,
  kind)]`), never a Cartesian-only shortcut.
- dt: `Grid3D.use_polar_time_step` uses the innermost arc r_c[0] dphi_min as phi width (`stability_issue` the same).
  Annular only: radius_min > 0 (`_check_cylindrical`).
- Faces: x_lo pec / pmc (no padding at r_lo), x_hi pec / pmc / cpml; pmc = tangential H 0 on the grid face, pec = the
  outermost cell layer. The r_hi CPML stretches d/dr only (no complex-r stretch of 1 / r: approximate at small k r; a
  ring-down keeps ~1.2e-2 of its first peak after 2000 steps: a clean open r_hi wants the complex radius stretch of
  fdtd_bor). phi faces pec / pmc / mirrors, or periodic when angle_wrap joins them; wrap phase bloch_ky = azimuthal
  order m [1/rad], F(phi + span) = F(phi) e^{-j m span} (the domain's antiperiodic sign is not read). Phase 0 or pi
  (within 1e-5 rad) steps the real set alone with c = +-1 (`_real_phases`; sectors only, Cartesian Bloch keeps the
  imaginary set at any k != 0).
- Supported: dipoles (`engine.face_dual_volume`), probes, DFT layers, PEC, poles, losses, harmonic FDTD. Refused
  (`EMPhysics3D._require_cartesian`): flux planes / boxes, TFSF, mode ports, far field, conformal PEC, subpixel.
- Self-test em3d_cylindrical_sector_cavity (heavy; semi-discrete Bessel roots, nu = (2 / dphi) sin(m dphi / 2)).

## Subpixel smoothing (igor 1.5; subpixel_smoothing3d, geometry in model/subpixel_geometry)
- `set_subpixel_smoothing(InterfaceGeometry3D)` -> `engine.subpixel_smoothing`; `refresh(engine)` at the end of every
  `rebuild_materials` (after PEC ids, before pole cb) averages the CURRENT table eps of the representative cells (0 =
  no override next to PEC), re-quantises the triples into a uint16 `face_id` (full grid, 2 B/cell) + `table` (0 entry =
  keep that component's mean); its deviation joins `quantisation_errors`.
- `face_coefficients` takes FACE_PARAMS (fid, t_face, use_face) and replaces only eps when neither cell is PEC; every
  caller (update_e_3d, cpml_e_3d, box_face_cb_3d -> pole cb, dipole weights, mode-port cb planes) passes
  `engine._face_args()`. A new E coefficient consumer goes through these helpers, never its own face mean.
- Mode ports: `port_materials` adds `_smoothing_deltas` as `CrossSectionMaterials.eps_point_deltas` so port modes see
  the stepped eps.
- Not smoothed: TFSF background, the stability check (raw tables), sigma_e / mu / pole membership. Self-test: em3d part
  of `subpixel_smoothing_cpu_gpu`. em_fdfd_3d applies the same average on the host (`interface_eps`).

## Conformal PEC (solver_plan S7.6; conformal_pec3d, kernels3d conformal_h_3d)
- conformal_boundaries reaches em_fdtd_3d in 3D models only (`conformal_dimensions` (3,)): `_apply_pec` calls
  `add_conformal_pec_region(window, mask, locator.inside([object]))` per object = `add_pec` (staircase cells: sources,
  ports, poles, calibrations, stability read them) + the outline test. `finalize` (after the first rebuild, once) builds
  `ConformalPEC3D` on the regions' box + STABILITY_BAND_CELLS + 3 and hands it to `engine.set_conformal_faces`.
- Dey-Mittra on the H faces (conformal_pec3d header). A cut face (a conformal and a free corner, no other PEC corner,
  all corners in the window: no wraps, ghosts, PML) weights its edges (A / A_eff) l / L. Edges with both ends free stay
  full (a cut no centre sees stays staircased; thin sheets between centres vanish); edges between two metal centres stay
  pinned even when partly free (unlike 2D H_z).
- Floor A_eff = max(A_free, min(A, stable)): a Gershgorin heuristic in 3D (overestimates; floored faces sharing an E
  edge sum: a sphere cavity reached 1.01 x the leap-frog limit at row_scale 1). So the build measures the largest
  eigenvalue of the vacuum C0^2 curl_h curl_e (operators3d, `weighted_curl_e`) on live E edges within
  STABILITY_BAND_CELLS = 4 of the corrected faces (host `eigs`, dense below 400 edges) and lowers row_scale until <=
  0.99^2 of the limit (6 rounds, then 0: A_eff = A). The check ignores mixed staircase / conformal contacts and
  dielectrics. `finalize_warnings` only when > half the cut faces are floored or row_scale hit 0 (default courant 0.99
  usually: ~70 % floored, TM101 9.5e-3 low vs staircase 5.2e-2: full accuracy needs a smaller courant). Weights reach ~4.
- Kernel `conformal_h_3d` (per precision; numpy twin, no numba) right after `update_h_3d` of each field set (Bloch:
  both), before the H CPML; 28 device B per face; estimate `boundary_locator.conformal_memory_estimate`.
- Self-test em3d_conformal_pec_cavity (`selftests_conformal3d`, heavy).

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
  `applications_per_basis` 0.22 x cells along the box, both fitted on 18^3-26^3: expect drift far beyond. Size rule
  `sparse_direct.vector_3d_size_issue`, `host_lu_bytes(n, dimensions=3)`: uncalibrated. Factorization cudss skips both
  (`definition_helpers.eigen_factorization`, `eigen_solver.factorization`) and the ND ordering. `FDFDSystem3D.symmetrizer`
  D makes D A complex symmetric (None with a Bloch phase) -> cuDSS LDL^T; the eigen role accepts cudss_mixed.
  Factorization mumps / pardiso (+ _mixed; solvers.host_direct, opt-in, either backend): the same LDL^T of D A on the
  host, no ND ordering (`elimination_order` None for every `LDLT_FACTORIZATIONS` entry: the libraries order
  themselves), but SuperLU's `vector_3d_size_issue` still applies (uncalibrated for them) plus the library's own
  memory check; a mixed one takes the polished path of `solve_resonances` (host solves with MIXED_EIGS_REFINEMENTS)
  and the time estimate's MIXED_APPLICATIONS_PER_BASIS (measured on cudss_mixed only). Host factor items:
  `memory_estimates.host_factor_items` on the SuperLU heuristic (tensor and tetrahedra, counted fill included).
- Subpixel: every assembly adds `fdfd3d.smoothing_deltas` (background eps only; none at PEC) to the face eps; FDFD
  reads MaterialGrid values, FDTD its index tables: they agree unless the FDTD re-quantises. The subpixel memory
  estimate still counts the FDTD face index (small overestimate).
- CPML: complex stretch frozen at omega_t (still reflectionless, leaky poles keep their Q); PML filter =
  PML_RESONANCE_FRACTION on the energy weights.
- Region filter: resonance_table selection + min_region_fraction -> `set_resonance_region` (tensor: host bool cells,
  each owning its +x / +y / +z E faces; a mirror-plane face belongs to the ghost, so a region on a mirror reads slightly
  low; tetrahedra: per element, via `FEMPhysics3D.resonance_region`). 'all' raises at apply. Not in a self-test.
- Not yet: conformal PEC, driven solves. Renderer: raw `<c>_re / _im / _abs` at their samples, centre moduli via
  `yee_grid.lower_face_partner` (Bloch: wrapped face x e^{+j k L}); e_abs primary.
- Self-tests `selftests_fdfd3d` (CPU operator cases; model runs on gpu with the cudss_mixed eigen node of
  `_eigen_solvers`, only em3d_fdfd_eigen_pec_box also on the cpu: keep heavy cases and examples off the cpu); templates
  pec_box_eigenmodes_3d, nanobeam_mirror_cell_bands_3d (Automatic). Open cavity = example
  examples/em3d/fdfd_mie_sphere_resonance_3d (gpu only): smoothed 1.0e-2 blue, staircase 2.8e-3 red: the subpixel
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
  re-assembles at the next solve.
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
  never |diag A|.
- PML = the mesh's node extent beyond `mesh.window` per face, must match the cpml faces exactly (`pml_shell_3d`
  raises); sigma_max at width / cpml_npml (cpml_npml only sets grading strength). PML element rules are 2 degrees higher.
- Walls fix tangential dofs on the bounding-box side (`face_tags`); cpml faces: the shell's outer side. A mirror plane
  is the window face itself (no ghost).
- Periodic / Bloch (`bloch_identification`): `periodic_dof_partners` REQUIRES pairs that keep node order (builtin Kuhn
  does; gmsh would need renumbering or dof signs). Axes chain (corner dofs get both phases), period = node extent (no
  shell on a periodic axis). A_free = P^H A P; `symmetric` False with a complex phase -> cuDSS LU, no Rayleigh polish.
  A PEC element or wall on either side pins the master.
- Size rule: host SuperLU (and MUMPS / PARDISO) under `vector_3d_size_issue(n x YEE_EQUIVALENT_UNKNOWNS)`
  (uncalibrated).
- Self-tests `selftests_fem3d` (tolerances first estimates); Mie sphere example examples/em3d/fem_mie_sphere_resonance_3d
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
  ledger warns when a new index path changes these. `_time_estimate` counts launches (`_FIELD_LAUNCH_WEIGHT`, 2 per
  CPML face, `_FEATURE_LAUNCHES`) and on cpu without numba `_NUMPY_FIELD_PASSES` / `_NUMPY_CPML_PASSES`: a new
  per-step kernel or numpy twin pass updates these counts.
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
  means), `_PortBoundary`, solved at the numerical frequency (beta_d = (2/D) arcsin(K D/2)). Source corrections via
  em/mode_ports `_port_correction` with live cb / db planes; amplitude sqrt(P / (cos(omega dt/2) cos(beta_d D/2))) so a
  flux plane reads P at f0. Modes solved until the first step only. Monitor = `PlaneFlux3D` + per-frequency
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
