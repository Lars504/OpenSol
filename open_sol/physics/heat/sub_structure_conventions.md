# open_sol/physics/heat — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Heat gotchas; DiffusionSystem (assembly, PCG, held
cells, metric, cut cells) is in `open_sol/core/sub_structure_conventions.md`.

## HeatPhysics
- HeatPhysics = DiffusionSystem (`heat.system`, `system.linear_solver`: script default Jacobi-PCG; model Automatic
  (CG_DIFFUSION_LINEAR_ROLE): multigrid-PCG on the gpu, host SuperLU on the cpu, banded in 1D) + T0 / capacity / Q /
  probes; settings and record from `DiffusionPhysicsBase` (shared with potential). Route CG through
  `DiffusionSystem.solve()`. BCSide validates per kind: dirichlet needs value, robin needs h and `external` (no T_inf
  default: never a silent 0 K ambient), neumann value defaults to 0; arguments of another kind raise. T is
  edge-extended into the PML after each solve. k_xy: 2D Cartesian planar only (uniform or graded).
- THERMAL_SCHEMA defaults k = rho = cp = 0: cells without k do not conduct; cells with an empty operator row (no k,
  loss or transient rho cp) are pinned and drop their Q in both time modes (a default capacity let them integrate Q
  adiabatically and run away). Assembly raises for a conducting region without an anchor (Dirichlet, h > 0 Robin with
  k_normal > 0, fixed cell, volumetric loss, transient rho cp > 0 in any of its cells); in a build at finalize.
- After writing k_* / rho / cp directly call `materials_changed(names)` or `rebuild_operator()`. A second `prepare`
  re-allocates the fixed-temperature and volumetric-loss buffers: never after features are applied. rho / cp
  (`_CAPACITY_MATERIALS`) never dirty the operator: a transient step replaces only C/dt
  (`DiffusionSystem.refresh_capacity`, also for a new step length; full assembly when it refuses); steady heat ignores
  them.
- Thermo-optic n0 (zz from eps_r, xx / yy from eps_xx / eps_yy; `physics/em/index_reference.record_reference_index`)
  are recorded in `prepare` and `finalize` and again by the 'index' SlotSum right before its first transfer, so the
  order of physics creation does not matter. heat/coupling.py transforms return dn (E13), evaluated as dndT dT s + n0 (s
  - 1) with s - 1 = q / (1 + s) (never (n0 + dndT dT) s - n0: float32 cancellation at small rises); the expansion factor
  s = sqrt(s_x s_y) on every component assumes alpha_xx = alpha_yy. T0 is both the initial and the reference
  temperature.
- Transient `step(t, every * dt_sync)` stores T at the end of the implicit-Euler interval: `heat.time`, its
  `time_label` and probe times lead runner.t / Snapshot.time by up to (every-1) dt_sync (the last may pass t_end;
  documented, not deferred).
- Backends: fields on `domain.xp`; the rise / temperature conversions pick the cupy kernel or its numpy twin in
  `prepare`; the thermo-optic kernel is a twin ('heat.thermo_optic_index_shift'); thermal_noise.py reads every array by
  `to_host` (the particle operator is assembled on the host on both backends). Self-test diffusion_cpu_gpu
  (`selftests_backend.py`, also covers electric_currents): heated_slab on the cpu Automatic SuperLU vs Q L^2 / (8 k), the
  point source (cpu vs gpu and its exact energy balance: the only point-source / heat_flux / Robin balance case) and the
  terminal divider cpu vs gpu.
- Radiation (feature radiation): tensor meshes per term the radiating window cells (flat) and their area; the
  loss A eps sigma (T^4 - T_amb^4) is Newton-linearised at the cell-centre T and written as a cell loss on the rise
  (`DiffusionSystem.add_cell_loss` adds the change to extra_diag / extra_rhs, so `refresh_diagonal` replaces the
  diagonal without a re-assembly). Steady `step` re-linearises and re-solves until the radiating T changes by <=
  RADIATION_TOLERANCE (relative; at most RADIATION_MAX_ITERATIONS, else a `runtime_warnings` entry); transient steps
  linearise at the step start. Outer faces take the metric face area of their boundary cells, perimeters the staircase
  `perimeter_flux_weights` (no cut-cell outline). It adds to any face condition (not in the 'later wins' set).
  Triangles / tetrahedra (`FEMHeatMixin._add_radiation_term`): one extra Robin edge / surface condition per term whose
  h / external are arrays per mesh edge / face (`EdgeCondition`; fem.diffusion `robin_assembled` / `robin_anchoring`:
  an array is assembled on all its entities, zeros included, and anchors where h > 0), linearised at the edge
  midpoint / face centroid T (hooks `_entity_rise` etc.); the same steady loop, but each re-linearisation changes h,
  so it re-assembles (only the Robin term: the fem diffusion systems keep unchanged terms and slot maps, fem sub file
  "Re-assembly") and refactorises. Self-tests heat_radiation_slab (tensor), heat_radiation_slab_triangles (P2).
- `Q` is the coupled slot (overwritten per transfer), `Q_ext` the applied sources; W/m^3 times the cell area (3D:
  volume). Model vocabulary `cp`; `cp_` only in library `Material.thermal` (`Material.model_properties` renames it).

## Triangle meshes (FEMHeatPhysics, `solver_fem.py`, mesh plan T4; fem sub file "Scalar diffusion")
- FEMHeatPhysics on a TriangleDomain (setting `discretization`: P2 default, P1): same field names, slot, feature API
  and T0 handling (unknown = T - T0); state() arrays per element, renderer T / T_rise nodal, the rest P0. Apply
  callbacks branch on `ctx.domain.grid_kind`: temperature / convection on objects = Dirichlet / Robin on their outline
  edges, heat_flux on objects = an edge source, point sources and probes at the point's node.
- Refused: azimuthal_order != 0 and the thermal_noise study (validation), the noise features (grids);
  conformal_boundaries is a no-op. A step re-assembles (and refactors) only when k, a BC kind / Robin h, held cells, rho
  / cp or the step length changed (no capacity-only refresh); a Dirichlet value or Robin external of t only refreshes
  the lift / loads (fem sub file), so fixed-dt transients factor once.
- Linear solver: Automatic = sparse_direct through `DiffusionDirectSolver`; back_end 'auto' = a double cuDSS Cholesky
  on the gpu with nvmath-python (`auto_back_ends` triangles / tetrahedra), host SuperLU on the cpu, without nvmath or
  after a cuDSS refusal. 'auto' never picks 'cudss_mixed' (magnetostatics shares the role; its mu contrasts rule out a
  float32 factor). cuDSS factorises the reduced operator (real symmetric, SPD when anchored, consistent capacity mass) by
  real Cholesky with plan reuse (`factorize_host(..., real=True, positive_definite=True, previous=)`), shared with the
  FV path. It pays for long fixed-dt transients above ~1e5 dofs and coupled k(T) re-factors; a small 2D steady solve
  pays the cuDSS plan (and nvmath's first-use start) where SuperLU takes ms: back_end 'scipy_splu' keeps it on the host.
  Opt-in MUMPS / PARDISO back ends (DIFFUSION_DIRECT_BACK_ENDS) factor it on the host as well (PARDISO Cholesky, MUMPS
  LDL^T; analysis reused while the pattern holds), keep the SuperLU size rule and are booked by the fem estimate's
  direct_factor_item.
  'cudss_mixed' is unmeasured (kappa ~ (L / h)^2 x the k contrast). A pcg node works (multigrid on P2 untested: its
  aggregation counts negative off-diagonals only, P2 stiffness has positive ones); transient pcg steps stop against the
  step's initial residual. Time estimate = the 2D DiffusionSystem recipe at `fem.diffusion.fem_time_system`
  (uncalibrated).
- Memory (`_memory_estimate`, every mesh kind): `fem_diffusion_estimate[_3d]` (fem sub file) with the 'linear' cuDSS
  back end (`fem.diffusion.cudss_back_end`), `refactorizes` = k (transient: rho, cp) follows a state or t
  (`_operator_follows_state`), and the active share `_active_fraction` (k > 0, transient rho and cp; all with an
  out-of-plane convection loss); tensor: the DiffusionSystem + `cudss_diffusion_estimate(input_copy=False)`, the
  thermal_noise line_block chunk a transient phase. heat_fem_cpu_gpu (gpu, cuDSS, 31 % of the window not conducting):
  triangles carry no material_elements yet, so operator and factor are sized on all dofs (~1.8x the measured reduced
  system): the estimate is high until the TriangleMeshEstimate gets them.
- Gap (not built): azimuthal_order (an m^2 / r mass term, axis dofs held at m != 0); the thermal_noise study on
  triangles (K = reduced FEM operator, C = rho cp consistent mass, A = k_zz mass, b by the element weighting; needs
  em_modes mode-pair weightings on triangles and per-element `material_regions`; the shifted solves would fit cuDSS LDL^T
  with plan reuse better than line_block).
- Self-tests `selftests_fem.py` (probe results only): heat_fem_disc_heater, heat_fem_transient_slab (slowest-mode decay
  vs ln(1 + mu dt) / dt, Automatic and an exact pcg node), heat_fem_cpu_gpu, heat_fem_host_direct (the disc heater on
  the installed MUMPS / PARDISO back ends vs a scipy_splu node, cpu; skipped without python-mumps / pydiso). Not
  covered: values of t on triangles.

## Tetrahedron meshes (FEMHeatPhysics3D, `solver_fem3d.py`; fem sub file "Scalar diffusion on tetrahedra")
- `solver_fem.FEMHeatMixin` holds everything FEMHeatPhysics and FEMHeatPhysics3D share (steps, state, sources,
  T0, flux on read); the classes add their outline / interface API, renderer and `_coefficient_tensor`.
- FEMHeatPhysics3D on a TetrahedronDomain (setting `discretization` P2 / P1, affine tets): FEMHeatPhysics names /
  slot / state, k = diag(k_xx, k_yy, k_zz) (k_xy refused), renderer `fem.renderer3d.LagrangeSampleRenderer`. Apply
  callbacks branch on `grid_kind == 'tetrahedra'`: temperature / convection / heat_flux on objects act on their
  interface faces (`ctx.boundary_faces`; heat_flux = a sheet source heating both sides), point_heat_source = the exact
  point load at `ctx.point_position(name)`, temperature_probe evaluates T at the point (no point nodes).
  `_three_d_issues` allows object boundary features only on tetrahedra and refuses the thermal_noise study there.
  Finalize warns on slivers (POOR_RADIUS_RATIO).
- Linear solver: Automatic = sparse_direct ('auto' = cuDSS double on the gpu with nvmath-python, else host SuperLU
  with `diffusion_size_issue(n_nodes + n_edges, 3)`); a pcg node works too (any PCGSolver-protocol solver). One
  factorisation per assembly (steady: per material change; transient: per step length / capacity change), so fixed-dt
  transients and coupled re-solves reuse a factor; 'cudss' / 'cudss_mixed' unmeasured on sliver-rich gmsh meshes.
  Estimates `fem_diffusion_estimate_3d` (as 2D, active share from the TetrahedronMeshEstimate's material_elements),
  time = the 3D recipe with `fem.diffusion.lagrange_entries_per_dof` (`fem_time_system`), both uncalibrated. Field derived values raise on tetrahedra (Session): self-tests read probes.
- Self-tests `selftests_fem3d.py` (builtin Kuhn meshes, probes only): heat_fem3d_slab (P2 / P1 nodally exact; P2 with a
  convective face), heat_fem3d_interface, heat_fem3d_transient_slab, heat_fem3d_cpu_gpu.

## Thermal noise (`thermal_noise.py`, features thermorefractive_noise / particle_noise, study thermal_noise)
- S_X(f) = 4 k_B T0^2 integral dq/2pi |g~(q - q0)|^2 Re b^H (j omega C + K + k_zz q^2 A)^-1 b, b = w A: one-sided,
  checked against the lumped C, G identity and the sum rule (variance k_B T0^2 sum |w|^2 A / (rho cp)). K = the steady
  `operator_full` on the host restricted to the solved cells: held cells and Dirichlet faces are u = 0 reservoirs,
  cells with neither a conductance diagonal nor rho cp drop out. The heat 'shifted' solver role (thermal_noise only,
  `HeatPhysics(shifted_solver=)`): Automatic 'sparse_direct' = one host SuperLU per (frequency, distinct |q|): cost
  grows with n_frequencies x axial samples, never per weighting cell. Its setting_choices offer back_end auto / scipy_splu
  / cusolver_qr and the host direct back ends (HOST_DIRECT_BACK_ENDS). `spectral_densities` calls
  `factorize_host(symmetric=SHIFTED_OPERATOR_SYMMETRIC, previous=)`: K is symmetric by construction (planar FV
  conductance form, energy-based k_xy term; the particle operator too), so the shifted operator is complex symmetric
  and a MUMPS / PARDISO back end factors LDL^T (its asymmetry check falls back to LU). Only a host direct factor is
  kept (`_shift_factor`) and passed as previous= to the next shift (analysis reused while the K + diag pattern holds;
  a pattern change re-analyses); `shifted_spectral_densities` releases it at the end (`release_shift_factor`). SuperLU
  ignores both options and its factor is still dropped after each shift (never two alive). A new subclass with an
  unsymmetric K sets SHIFTED_OPERATOR_SYMMETRIC False.
- A 'line_block' node as the shifted solver: every (frequency, axial sample) shift of a channel goes to the one
  LineBlockSolver of the role, shared by every noise operator (re-bound by `set_operator(owner=)`; contribution parts as
  rhs rows); rejected shifts go through `spectral_densities` on a private SparseDirectSolver. Device state is released
  after `solve_thermal_noise`; rejected_shifts = fallbacks in the summary 'solvers' record. Mixed precision rejects very
  ill-conditioned shifts (low f on an insulated region: the static null mode).
- Heat in the thermal_noise study: `require_anchor=False`, never stepped (not a background physics); T0 is the
  (uniform) fluctuation temperature (a heated background is refused: couplings reading heat are errors).
- Mode pair: w = omega eps0 / 4 sum_i 2 n_i dndT_i E_inc,i E_partner,i with UNconjugated 1 W centre fields, the partner
  travelling against the collected direction (Lorentz reciprocity). Backward copy of a solved (forward, exp(-j beta z))
  mode: E_t kept, E_z negated, so backscatter into itself weights |E_t|^2 - |E_z|^2, forward self-scatter |E|^2. An em
  mirror face lies on the window edge, so heat's insulated face there is the even mirror; the spectrum is x 2 per mirror
  face. The outermost window cell of every em pec / pmc face is a ghost outside the wall plane: `wall_ghost_cells`
  zeroes w there, but it stays in the heat solve with that face's heat condition (cutting it would silently insulate
  e.g. a substrate sink).
- Null modes: a connected solved region without anchor (Dirichlet / h > 0 face, link to a held cell, volumetric loss;
  particles: always) has u = const in the null space of K at q = 0 (or with k_zz = 0): its share s |sum w A|^2 / sum C
  never relaxes and is reported as `static_variance`; `band_fraction` is relative to variance - static_variance (q x
  axial length <= 1e-9 counts as q = 0).
- Illumination weighting: E_inc = the last solve of an em_fdfd_ez / _hz background physics at cell centres (hz:
  `face_mean_at_centres` of `face_fields()`), partner as for a mode pair, q0 = beta_coll. Run errors for unequal
  boundary configs or wavelengths. Both EM physics share one BoundaryConfig, so under a mirror w stays even (the
  insulated heat mirror is right); an illumination of the other parity needs a full domain. w in W^0.5 / (K m^3) per
  FDFD source amplitude, S in W/Hz.
- Every fluctuating cell with w != 0 needs rho cp > 0 (else infinite variance): feature validation and `check_weight` at
  run time.
- Contributions: apply reads `ctx.material_regions` masks (2D, released by `release_caches` after feature apply); a
  selection over several materials splits w by material ('other' = weighted cells in no material): one back
  substitution per part on the shared factor. Contribution variances sum to the total and psd_cross integrates to 0
  (heat_noise_sum_rule).
- Mode selection: mode_pair dicts carry (selection, number, direction); `resolve_mode_channels` makes one channel per
  collected mode (te / tm = `fundamental_mode`: first mode in rank order with te_fraction > 0.5 / <= 0.5; te_and_tm =
  both; incident 'collected' = each channel's own mode), each with its own weighting, q0 and factorisations.
  `thermal_noise_result` prefixes psd_<tag> / suffixes extras _<tag> only with several channels, so single-mode results
  keep plain names (the selftests read those).
- Particle noise (`HarmonicParticleOperator`): dilute particles on the selected cells reuse the solve with C = 1, K =
  -div D grad from D on the selected cells only (no flux out of the selection; heat materials, held cells and boundary
  features play no part), a = D, strength n0 = c N_A instead of k_B T0^2, weighting with dndT_i -> dn/dc M / N_A
  (`index_increment`). Both operator kinds share `_HarmonicDiffusionOperator`; `operator_for(particle)` caches each.
  Extras (`_particle_info`): bound signal max |w|^2 and position, sensing volume (integral |wg|^2)^2 / integral |wg|^4,
  psd_single = psd / (n0 V_s). Hosted by heat only for the noise machinery.

## Definition and self-tests
- Heat: material Params default to None (only set properties written); every study but time_dependent forces steady
  mode; no `domain_requirements` (no PML, no Courant). A later boundary feature on the same face wins (warning).
- RerunSolver (sweep reuse_build; FV and FEM): `reset_records` clears the probe series. Followed values
  (`follows_parameters`): temperature (outer faces by `set_face_value`; objects: tensor `add_fixed_temperature` again
  on the kept mask, triangles / tetrahedra the edge / surface condition handle), heat_source Q (a parameter-only number:
  the change added to Q_ext) and point_heat_source P (the time setter). h / T_ext / heat_flux are not followed (a live
  value there rebuilds per sweep value).
- Values of t (time_dependent: temperature on outer faces only (error with objects: held cells cannot follow it),
  heat_flux, heat_source Q, point_heat_source P): faces through `set_face_value` (in-place rhs in 2D FV; FEM 2D / 3D refresh lift and loads, same factor),
  sources / perimeter fluxes add the change to Q_ext (float32 accumulation). Taken at the step start t_n although the
  implicit step solves for t_n + dt (first-order lag, the convention of every t-value). Self-test
  heat_time_dependent_values (`selftests_time.py`).
- EM mirror planes: only the default insulated face is the even image of the mirror-even EM losses, so a temperature /
  heat_flux / convection feature on an em mirror face warns (`definition_helpers.mirror_face_issues`; error in a
  thermal_noise study, whose x 2 per mirror assumes the even mirror). Unfolding treats heat fields as even (s = +1).
  An odd heat mirror needs no new code: a temperature face is exactly the antisymmetric image of T - T_face.
- Setting `geometry` 'axisymmetric' = `HeatPhysics(axisymmetric=True)`: x = r, x_lo the axis; densities keep their
  units; `add_point_heat_source` divides by the ring volume (P = whole-ring W). Object-perimeter heat_flux / convection
  take `definition_helpers.perimeter_flux_weights` (x-faces by r_face / r_c; polar / spherical: both face kinds by
  metric area over volume), never the planar `ctx.perimeter_weights`. Validation `axisymmetric_issues` (shared with
  potential): warning for features selecting x_lo and for couplings to a planar partner, error for a k_xy material.
  Field derived values of the whole model are weighted by 2 pi r_c (`TensorAccess._ring_weights`, keyed on the
  solver's `axisymmetric`).
- Conformal boundaries (core sub file "Cut-cell perimeter terms"): object-perimeter heat_flux / convection go through
  `outline_pieces_of`; convection = `HeatPhysics.add_surface_loss` with h L / (1 + h delta / k_n), k_n the owner's
  normal k read at feature apply (a later k(T) does not update it; k_n = 0 gives the bare h L). Polar / spherical
  models honour it too (cut cells on the logical cells).
- 3D (`dimensions` (2, 3)): HeatPhysics on a Domain3D (7-point system with k_zz; THERMAL_SCHEMA always holds k_zz,
  isotropic k writes it, 2D ignores it). Point sources and probes take the point's height (Q_ext += P / cell volume).
  Validation `_three_d_issues`: errors for boundary features on objects on tensor meshes (perimeter masks are 2D only:
  use outer faces incl. z_lo / z_hi or fixed_temperature_domain), out_of_plane_convection, axisymmetric and k_xy
  materials. heat_source with an x, y expression still goes through `eval_field` (one full-grid float32 build
  temporary). Template 'heated_chip_3d'; selftest heat_3d_field_access (product of sines = exact 7-point eigenvector:
  per-axis k, z faces); heat_3d_graded_z_robin = the only 3D Robin self-test (graded cases keep Q = 0: source-free 1D
  conduction is exact on any widths, a heated slab is not). Uniform-mesh 3D behaviour: examples cube_cooling_3d,
  heated_sphere_3d, square_pin_fin_3d.
- Thermal noise self-tests (`selftests_thermal_noise.py`): heat_noise_cosine_lorentzian (cell-centred cosine = exact
  eigenvector of the insulated FV operator), heat_noise_sum_rule, heat_noise_pec_box_mode (TE01 of an em_modes pec box,
  exact discrete closed form incl. 1 W normalisation and static share; the ghost ring has no heat material; mirrored
  half run pins the mirror plane and factor; *_cpu copies, gpu vs cpu psd), heat_noise_line_block (a second identical
  heat physics on a line_block node: series agree, 0 rejected shifts). Not covered: illumination weighting and
  particle_noise. Template waveguide_thermal_noise (half domain).
- Self-test heat_axisymmetric_inner_flux: heat_flux on a tube's inner perimeter (pins the perimeter x-face radius).
- Setting `azimuthal_order` (`definition_helpers.AZIMUTHAL_ORDER_PARAM`, shared with potential; core sub file
  "Azimuthal order"): T - T0 is the cos(m phi) amplitude (T0 drops out at m != 0), every ring total and derived
  integral carries pi, a point_heat_source is the order-m part of a point source at phi = 0; `_has_anchor` is skipped at
  m != 0. Validation `azimuthal_order_issues`: axisymmetric on a Cartesian domain, one m per model, a coupling warning.
  Self-test heat_azimuthal_order_2 (+ variant cpu).
- Polar / spherical heat (DomainSpec.coordinates; `mapped_domain_issues` refuses k_xy, and heat_flux / convection on
  objects in 3D sectors / 1D models; example heat/polar_annulus_log_profile.json; scripts: `HeatPhysics` on a polar
  `Domain`, axisymmetric=True for (r, theta)). `selftests_coordinates.py` uses padding-free Domains from
  `NonUniformGrid2D` widths through the script API, reading the float64 `system.unknown2d`: metric_coordinate_geometry,
  heat_polar_metric / heat_spherical_metric (uniformly heated parabolas need uniform dr; source-free chains on graded
  dr), heat_polar_wrap (periodic full ring and antiperiodic C_4 sector; Q = q0 / r_c^2 keeps graded dr exact). Thermal
  noise and object-perimeter fluxes refuse polar.
- 3D cylindrical sectors (core sub file "Coordinate metric", model sub file "Cylindrical (r, phi, z) sectors"):
  `HeatPhysics` on a polar `Domain3D.from_widths(...)` solves (r, phi, z) with k_xx / k_yy / k_zz = radial / angular /
  axial, phi wrap included; models: DomainSpec polar + size_z (3D rules: boundary features on outer faces only; the
  estimator adds the wrap's 9-slot build array). `selftests_cylindrical_3d.py` (cpu): metric_cylindrical_geometry_3d,
  heat_cylindrical_sector_3d, model_cylindrical_sector_paths, heat_cylindrical_sector_3d_model; it reuses the widths /
  axis / Dirichlet / radial-chain / refusal helpers of `selftests_coordinates.py` (same CONDUCTIVITY), as
  `selftests_fem3d.py` the probe and decay-rate helpers of `selftests_fem.py`.
- Unbounded quasi-static heat (symmetry 2.15): outer grading (`MeshSpec.outer_dx` / `outer_ratio`) puts a held far face
  ~1e3 x the structure away on a few dozen cells (heat_unbounded_disc_heater, `selftests_unbounded.py`). Arithmetic-
  metric FV needs bounded neighbour ratios, so there is no rational infinite-element layer.
- 1D models (`dimensions` (1, 2, 3)): HeatPhysics unchanged on the one-cell-across Domain, Automatic banded; validation
  refuses out_of_plane_convection and the noise features. Self-tests line_slab / _cylinder / _sphere in
  model/selftests_line.py.
- Self-test coverage: heat cases are discrete FV identities (Robin strips on a graded dx != dy mesh, point source +
  heat_flux energy balance in diffusion_cpu_gpu); uniform-mesh slabs, the lumped implicit-Euler transient and the
  axisymmetric heated cylinder only by examples (heat/heated_slab_parabola, composite_wall,
  lumped_plate_heating_dt_sweep, mechanics/axisymmetric_heated_cylinder_thermal_stress, heat/cylinder_cooling_bessel).
  k_xy is pinned by heat_cross_conduction_box_graded (constant-valued faces cannot impose a sloped boundary for a
  uniform-gradient test), heat_cross_conduction_graded_convergence and heat_cross_flux_dissipation (a prescribed linear
  T written into `system.unknown2d` after the solve).
