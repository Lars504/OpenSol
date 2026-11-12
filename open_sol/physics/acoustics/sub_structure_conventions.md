# open_sol/physics/acoustics — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Elastic / acoustic wave gotchas.

## AcousticPhysics (`elastic_waves`, default instance 'acoustics')
- Velocity-stress staggered grid (Virieux, plane strain) on the full grid, EM Yee index layout (idx = i * ny + j): vx /
  vy on the hi x / y face (half steps), sxx / syy at the centre, sxy on the hi-hi corner; tension positive, pressure =
  -(sxx + syy) / 2. Two fused RawKernels per step (velocity, stress) with the CPML psi updates; sources scatter after
  each (forces at t_n, volume / heat-pulse thermoelastic at t_n + dt/2); the coupled heating slot's -gruneisen q is
  fused into the stress kernel (args gruneisen, heating, thermal_scale = dt x envelope at t_n + dt/2 / stress_scale; 0 =
  none: dummies, never read; the numpy / numba twins likewise); probes record stresses at t_n+1, velocities stamped
  t_n+1/2, all probes of a physics in one 'probe_record' field kernel per step (`sources.ProbeRecorder`: a (rows,
  probes) stored-unit buffer, host times; a probe added after steps starts at the next row).
- Per-step launch arguments are built once per (dt, configuration) (`_launch_arguments`, dropped by `_configure`, the
  storage scales and `materials_changed`): every array in them must stay persistent (rewritten in place, never
  reassigned); a new argument that changes per step goes outside the cache (as the thermal triple).
- Non-periodic outer faces are slip walls (v_n = 0, shear 0) by construction (missing lo neighbour reads 0, hi face /
  corner pinned by zero coefficients). 'absorbing' = that wall behind a CPML in the outermost `pml_cells` cells
  (`cpml.py`, kappa 1, alpha 0; reflects ~exp(-0.45 (c dt / L) ln(1/R)^2): 7 % at R = 0.01, 80 cells, courant 0.9).
  NO domain_requirements (never conflicts with an EM boundary config): alone, its layers lie inside the physical window
  (validation info; points in them warn). 'rigid' on a face another physics padded pins the window face (`_pins`; the
  padding behind steps but stays decoupled); 'periodic' on a padded face is a validation error.
- Mirror planes: a 'rigid' face on an EM mirror face is the even mirror (renderer declares parities, Results unfold);
  another face type on a model mirror face warns (`_mirror_plane_issues`).
- Coefficients (one RawKernel, no temporaries, safe on per-step material updates): isotropic moduli from rho and the
  speeds when sound_speed > 0, else from youngs_modulus > 0 / poisson_ratio (library solids carry only E / nu), else 0;
  c11 / c12 / c22 / c66 numbers override per entry (default NaN = unset, tested `x != x`). Faces take width-weighted
  mean density and damping, corners the harmonic mean of 4 shear moduli. rho = 0 with no stiffness = vacuum (free
  surface). Validation (`_material_moduli`) mirrors the precedence and errors on a non-positive-definite stiffness.
- Monoclinic c16 / c26 (`MONOCLINIC_ARRAYS`: either set allocates both, default 0, no isotropic fallback): the
  'stress_strain_monoclinic' kernel writes this step's CPML-stretched rates to `_strain_rates`, then
  `monoclinic_stress` adds the 4-corner / 4-cell mean cross terms. The two means are adjoint (plain means, no area
  weights on graded cells): keep them mirrored, Q = inf of elastic_monoclinic_plane_waves depends on it. Monoclinic
  always allocates the strain accumulator (`elastic_strain` copies it). Planar single precision only (ValueError;
  validation with axisymmetric / half_storage, `_PLANAR_ONLY_ENTRIES`); validation requires the 3 x 3 (c11, c12, c16;
  c22, c26; c66) positive definite. Not checked: CPML stability in strongly anisotropic media (Becache 2003; the strips
  are normal incidence only).
- `rho` is shared with heat / fluid; youngs_modulus / poisson_ratio / photoelastic_p11 / _p12 are the very Param
  objects of `physics/mechanics/material_params.py` (elastic_waves registers first: keep them shared, never copies).
  `gruneisen` is read every step by the thermoelastic sources.
- `elastic_strain()`: persistent pair = [[c11, c12], [c12, c22]]^-1 of the centre stresses; a fluid (det <= 1e-6 c11
  c22) gives half the volumetric strain each; thermoelastic stress reads as minus the thermal strain; e_zz never formed.
  With piezo arrays it copies the strain accumulator (total kinematic strain: heating reads as expansion).
- dt: `stable_dt` = courant / (c_bound sqrt(1/dx_min^2 + 1/dy_min^2)), c_bound^2 = max over cells of the stability
  stiffness S x the largest 1 / rho_face (vacuum faces halve rho). S (`kernels.stability_stiffness`, host twin
  `definition.stability_stiffness`) = lambda_max of the acoustic tensor at the corner wavenumbers, at least max(c11,
  c22): the exact von Neumann limit of a uniform medium (max(c11, c22) alone diverges for silicon [100] above courant
  ~0.92). c16 / c26 and the two-way piezo stiffening enter as an absolute-value bound (dt shorter than exact, ~8 % at
  10 % of c11). CPML reference speed stays sqrt(max(c11, c22) / rho). Derived at construction (deferred without
  stiffness) and at `finalize` (one host sync). `step(t, span)` honours a shorter runner span by reconfiguring
  coefficients and CPML once (and back): each switch is a first-order phase error, so phase-sensitive / resonance runs
  avoid mixed spans; runtime material rewrites are not stability-checked. MeshEstimate.dt / check.json dt_estimate
  exclude it (validate logs `definition.estimated_time_step`, also read by laminar_flow's validation).
- Renderer `device_field` pressure / elastic_strain_* return window views of persistent buffers rewritten by the next
  call (copy to keep one).
- Axisymmetric (r, z) (shared `geometry` setting; space 'axisymmetric' sets bc_x_lo 'rigid'): needs x_lo 'rigid' and
  unpadded (ValueError; validation), since the slip wall IS the axis rule. `*_axisymmetric_kernel` variants (`#define
  AXISYMMETRIC`) add hoop stress `fields.stt` (None in planar) and radius rows `axisymmetric.kernel_rows`; c_tt = c11,
  c_rt = c_zt = c12 (c_ij crystals warn). Exact discrete adjoint of the ring-volume energy: keep any change to one
  update mirrored in the other, or energy conservation is lost. x-face densities use ring-volume shares
  (`density_shares`, so the dt bound reads them too); dt x `radial_stability_factor` (Gershgorin bound, mirrored in
  `definition._cell_weights`). Point forces / volume rates are per unit ring length; pressure = -(srr + szz + stt) / 3.
  The r CPML stretches the radial derivatives only. stt exists only in axisymmetric physics: the renderer drops it from
  its per-instance `field_infos` in planar, AcousticProbe raises, validation errors on a planar stt probe /
  resonance_monitor / stored field (`_hoop_output_issues`); output_fields keep it (class attribute). Not in state().
- Stepper precision (role 'stepper' = `HALF_STORAGE_STEPPER_ROLE`, `AcousticPhysics(precision=)`):
  `kernels.field_kernels(precision)` compiles every kernel touching the stepped fields per precision; `HALF_FIELDS`
  uses plain read-modify-write scatters instead of atomicAdd, so one scatter's indices must stay distinct. half_storage
  stores float16 stress / stress_scale and velocity / velocity_scale, powers of two fixed by `_set_storage_scales` at
  the first finalize (largest source `peak_stress` at HALF_STORAGE_PEAK = 16 stored units). Every reader / writer
  converts (sources, thermoelastic kernels, pressure / strain kernels, `physical_array`, `state()` float32 copies
  (a lazy Mapping in every precision: pressure runs its kernel only when read),
  probes x `value_scale`); psi slabs, coefficients, pressure / strain buffers stay float32. Refused (validation
  `_half_storage_issues` + ValueError): piezo arrays and any coupling into the physics (amplitude unknown when the scale
  is fixed); resonance_monitor by its feature precisions. A new field reader or writer must scale or refuse. Writers into
  float16 fields never use cupy's float16 kernels: `half_bits(field)` + `STORED_ACCESS` (core.half_storage).
- Backends ('gpu', 'cpu'): every `kernels.py` kernel is a `core.kernels` twin ('acoustics.<name>'; single-precision
  field kernels registered by `field_kernels('single')` at the first AcousticPhysics, half_storage ones stay plain
  RawKernels); arrays on `domain.xp`. Host twins write in place through flat views: fields must stay C-contiguous
  (np.add.at for atomicAdd ones); a CUDA source change changes its `_*_host` twin in the same edit. half_storage on the
  cpu: ValueError. numba twins of the six single velocity / stress updates unpack args through `_velocity_arguments` /
  `_stress_arguments`, so a new kernel argument goes there once; they compile with `cache=True` (keyed per closure
  variant: after an edit of kernels.py the first cpu process compiles again, profile twice). Piezo kernels and the
  volume scatter have twins but no parity case.
- heat_pulse_source takes `xp.where(mask, power_density, 0)` on the selection box: the expression may be non-finite
  outside (0 * NaN would fill the run with NaN).
- initial_velocity: written into vx / vy at apply (single precision only), at the staggered face centres
  (`definition._face_velocity`), so the scheme starts from v^(-1/2) = v0, s^0 = 0; pinned faces drop it at step 1.
- `heating_power` slot: kept by reference, times the optional gaussian `heating_envelope`; no `engine` attribute, so
  couplings reading it always transfer (no n_step revision skip).
- resonance_monitor: shared row with band 'frequency' (Hz, no wavelength column); dt = `solver.dt` read at result time
  (fixed only at finalize); no sample delay (AcousticProbe stamps velocities t - dt/2 itself); pulse end from
  `source_waveforms()` (`HeatingPulse.duration` = delay + 4.5 width). Pressure sees no S mode of a solid. Probe traces
  are counted by the estimator in `definition._time_step_estimate` steps.

## Piezoelectric (piezoelectric_plan.md stages 1-3; coupling in coupling/definitions_piezo.py)
- `piezo_e11 .. _e26` [C/m^2] (reduced Voigt, J = 1 xx / 2 yy / 6 xy, `OPTIONAL_ACOUSTIC_SCHEMA`): any one allocates
  all six (the solver adds missing ones for scripts), `AcousticPhysics.piezoelectric` True; planar only.
- Slots `electric_field_x / _y` apply at the injection (`_apply_field_component`), never at the next step: they are the
  electrostatics' live E arrays, which its own runner step may overwrite first (spec order). The first transfer only
  copies E into E_last (initial state = equilibrium of that field); later ones apply -e dE (`piezo_component_stress`,
  `piezo_shear_stress_kernel` on corners with c66c > 0). Recorded stresses (probes inside `advance`) lack that step's
  dE, velocities do not.
- Strain accumulator (`_strain`, with piezo or c16 / c26): `STRAIN_ACCUMULATOR` field kernels add dt x the stress
  update's strain rate (CPML-stretched; corner shear only where c66c > 0, the adjoint of the shear stress rule).
  Kinematic: volume / thermoelastic sources do not enter it.
- Two-way (`enable_bound_charge(eps_xx, eps_yy, eps_xy)`, called by the coupling create after finalize, before
  runner.prepare): allocates P, rho_b and re-derives dt with the stiffened c^D (eps arrays kept by reference; runtime eps
  rewrites not re-checked). `bound_charge(window, own_faces)` (duck-typed electrostatics `window` / `bcs`): rho_b = -div
  P in face form (edge face: cell P on Dirichlet / Robin, 0 on insulating). Stress and charge are an adjoint pair for
  uniform eps: keep them mirrored (Q = inf of piezo_stiffened_bar_resonance depends on it). Discrete stiffening of mode
  k is e^2 / eps cos^2(k h / 2).
- Not built: pyroelectric / thermal-strain charge, axisymmetric piezo.
- Validation `_crystal_issues` (first elastic_waves physics): error when a crystal entry does not expand, info of the
  reduced constants, warning for an anti-plane coupling above 1e-6 (u_z clamped). Memory: accumulator (+ two-way P /
  rho_b, from the spec's couplings by `two_way_piezo`) steady, E_last + shear drive run items; dt estimate stiffened for
  two-way.

## Optics couplings (`couplings.py`, types in coupling/definitions_acoustics.py)
- acoustics_em_acousto_optic: E13 'index' contributor per index slot of an em_fdfd_* target, transform
  `strain_optic.strain_optic_index_shift` (e_zz 0, deformation 0); with the photoelastic tensor arrays
  (`PHOTOELASTIC_TENSOR_ARRAYS` p16 .. p66, any one allocates all ten, NaN = isotropic fallback) the slot's row p_i1
  e_xx + p_i2 e_yy + p_i6 g, g = `elastic_shear_strain()` (4-corner mean of the accumulator shear, else of sxy / c66c;
  persistent buffers, run items). em_fdfd_hz also gets the xy row as a 'sum' Delta eps_xy into eps_xy_update
  (`acousto_optic_shear_shift`; the estimator counts the shear buffers then). The FDFD joins the time_dependent study
  (dt_limit inf, re-solves only when dirty); the coupling inherits its every k (after_source_step False), so it
  transfers on ticks k-1, 2k-1, ... and the FDFD re-solves on the strain of its own tick (stroboscopic). An em_fdtd_*
  in the same run is a validation error.
- em_acoustics_photoacoustic: 'sum' contributor copying FDFD q_abs into heating_power; transfers only after FDFD
  re-solves (n_solves). Use the gaussian heating_envelope (instantaneous: warning). Tick 0's acoustic step runs before
  the first transfer (no heating in [0, dt]).

## Self-tests (`selftests.py`), template
- 1D strips periodic in y (4 cells) with absorbing x faces; elastic_wave_speeds' run p_wave_y (orthotropic, c22 !=
  c11) is the only check of c22, the d/dy stencils and the y CPML. acoustic_plate_resonances: slip walls on grid edges
  separate P and S modes exactly on the staggered grid. Resonance-case durations sit near the validation limit of 10
  band-floor periods after the pulse end 2 t0 (warning only): shortening one, recompute (duration - 8 / f_ricker) x
  frequency_min. elastic_axisymmetric_cylinder re-derives the discrete eigenproblem in numpy in the check (independent
  of `axisymmetric.py`). elastic_waves_half_storage raises unless summary 'solvers' records half_storage.
  elastic_monoclinic_plane_waves: the absorbing-strip cross-polarised run is the only check of the c16 / c26 signs (the
  ring spectrum sees their squares). Template 'photoacoustic_disc'.
- `ELASTIC_CASES` holds every case; `SELFTEST_CASES` (registered) leaves out `_COVERED_BY_EXAMPLES`
  (acoustic_normal_reflection, acoustic_thermoelastic_slab, acoustic_plate_resonances,
  elastic_axisymmetric_hoop_stress: examples/acoustics check them); the parity cases of selftests_backend look cases
  up in ELASTIC_CASES.
- Example `examples/acoustics/axisymmetric_heated_core_hoop_stress`: on the uniform (r, z) grid the static Lame
  solution is discrete-exact with r^2 -> r_c^2 - dx^2 / 4 in the cell stresses.
- coupling/selftests_acoustics.py: acousto_optic_uniform_strain (only check of p61 / p62), acousto_optic_shear_strain
  (shear columns and eps_xy). The photoacoustic coupling is checked by examples/acoustics/photoacoustic_beer_lambert.

## pressure_acoustics (frequency domain; `definition_harmonic.py`, `solver_harmonic_fem.py` / `_fem3d.py`)
- Registered by `definition.register` after elastic_waves (import inside it): rho / sound_speed are the elastic_waves
  Param objects (`_ELASTIC_PROPERTIES`), never copies. Triangles (planar / axisymmetric m = 0) and tetrahedra only,
  frequency_response study only (`step` does nothing). Template 'rigid_disk_scattering' lives in selftests_harmonic
  (TEMPLATE_SOURCES pair).
- `AcousticHarmonicBase` (2D file) holds everything dimension-free; subclasses supply mesh-kind hooks. Features only
  record; the fem.harmonic system is built at finalize / after a rho or sound_speed change (`_build_system`): a = 1 /
  rho, mass = 1 / (rho c^2), active = window (or every element with a PML) minus rigid_obstacle elements; a fluid
  element without rho / c > 0 is a run error.
- Per-frequency terms never re-assemble: `set_frequency_load` (full-numbering extra load) and `set_held_offsets`.
  Motion boundaries and point sources are (j omega)^power x a build-time load; the background adds its terms per
  frequency.
- Reduced sweeps (fem sub file "Reduced sweeps"): `plan_frequency_response` sets frequency load / held offsets to None,
  passes the (j omega)^p loads as PowerCoefficients and the background as `extra_loads` vector_of + `held_offsets_of`
  (`_background_at` / `_background_nodal_at`, each cached for its last omega, so held offsets and `_show_solution` never
  assemble the load); with a background the model has no moments (snapshots only). Modal is refused by the eigen
  pencil when an impedance boundary damps (use adaptive). In a planned sweep `solve_frequency_response` never sets the
  per-frequency terms; probe / surface records always go through `system.linear_output`; fields only where
  `wants_fields`, else `publish_frequency_response`. Direct sweeps take the same policy (the direct solve still writes
  `system.values`, which `_show_solution` reads).
- Outer-face conditions act on the window edge (`window_face_edges` / `window_faces`), never on mesh face tags: a wall
  continues along a PML side as sound hard, sources stay out of the layer; last feature per face counts,
  object-outline conditions accumulate (held values: later wins). A PML face takes no condition.
- Scattered field: the source lives only on contrast elements (above CONTRAST_RTOL = 1e-6 relative: float32
  MaterialGrid values equal to the background count as background), rigid-obstacle window elements and non-PML outer
  faces; nothing at the PML interface. Total p = p_s + p_b on window fluid dofs, p_s in the PML, 0 in obstacles; probes
  / surface results add p_b exactly at their points.
- 'linear' role `PRESSURE_LINEAR_ROLE`: back_end auto / scipy_splu / cudss / cudss_mixed (cusolver_qr refused:
  fem.harmonic's factorize_host would run SuperLU instead). On cuDSS the reduced pencil stays on the device, one plan
  per sweep. Memory: the kept DevicePencil / solution / publish maps are run items (no factor CSR copies on top:
  rigid_disk gpu measured within 4 %), the factor an external item (`definition_helpers.cudss_external`; fill rule
  2D -12 % on rigid_disk, small ducts ~-55 %: the per-factor fixed bytes of cudss_factor dominate there). The opt-in
  MUMPS / PARDISO back ends (DIFFUSION_DIRECT_BACK_ENDS) take fem.harmonic's host route (LDL^T of the complex
  symmetric A(omega), analysis reused per frequency, SuperLU size rules): memory = the SuperLU heuristic x
  `host_direct.superlu_bytes_scale` + `solver_host_bytes` (uncalibrated); time: the SuperLU items.
- PML strength from the fastest sound speed of the layer's fluid at `pml_frequency` (default lowest study frequency,
  `study_frequency_band` = a copy of the study's spacing rule: physics must not import model.study*).
- surface_pressure: int p dA / int p weight dA as dot products with unit / weight loads; side filters by the normal out
  of the selected object; axisymmetric whole rings, planar 2D per metre. The exact p_b part uses
  `SurfaceMonitor.quadrature` (built once): triangles read fem.assembly's edge-rule cache (`_edge_quadrature(self.full,
  entities, order + 2 + axisymmetric)`), tetrahedra `triangle_face_rule(order + 2)`: a change of either default degree /
  face rule in fem.assembly / assembly3d changes it here too.
- Probes keep their element dofs and basis weights (`_point_weights`, at add): a probe read is one dot product, so a
  change of the field's basis / geometry order must change `_point_weights` too.
- Shown fields: `_show_solution` keeps host `_scattered` and `_background_nodal`; `_publish` writes only per-element
  state (on cupy after a cuDSS solve from `system.device_values()`). The host total is formed on first read: readers
  call `_host_total()`, never `_total` directly.
- `assembly_seconds` (work counter) = wall time of every `_build_system`, carried as seconds_counter by the time
  recipe: code calling `_build_system` must not wrap it in another timed group.
- Self-tests selftests_harmonic (CPU, each with a SelfTestVariant 'gpu' at the same tolerance, but
  pressure_acoustics_host_direct: the PML duct at two frequencies on mumps_mixed / pardiso vs a scipy_splu node, cpu
  only, skipped without python-mumps / pydiso); pressure_acoustics_rigid_disk uses t / a = 0.001 (the square-rim thickness correction ~(t / a) ln(a / t) is +3 % at the template's 0.01).
- Open: losses, eigenfrequency study, structural couplings, m > 0, Bloch / periodic faces, interior zero-thickness
  rigid sheets (need doubled dofs).

## Open
- Axisymmetric: torsional (v_phi) waves, stretched-r CPML for the hoop terms, transversely isotropic hoop stiffness.
- half_storage: coupled slots (needs a scale from the coupling or a runtime rescale), piezo, c16 / c26 (the
  accumulator adds stored-unit rates), measured speed-up.
- Not built: outer traction-free faces (use vacuum cells); CPML alpha / kappa; acousto-optic moving-boundary term,
  anti-plane (yz / xz) index change, xy on EM types without eps_xy, out-of-plane strain of thermoelastic stress; library
  fluid sound speeds (water / air have no stiffness data: set sound_speed).

## elastic_waves on triangles / cuDSS: decision (not built)
- Tensor velocity-stress only (`grids = ('tensor',)`); no linear solve in the stepper, so cuDSS has no role.
  Triangles would pay only for curved traction-free surfaces / interfaces (staircase Rayleigh / whispering-gallery
  modes); eigenfrequencies / harmonic response of such shapes already run conforming in solid_mechanics.
- Cheapest route if a gate case appears: explicit P1 FEM on `fem/elasticity.py` (row-sum lumped mass, one SpMV per
  step, dt by power iteration). P2 needs mass-lumped P2 + bubble (row-sum P2 lumping gives zero / negative vertex
  masses). Also missing then: absorbing layer on unstructured meshes, point-located sources / probes, per-element
  slots, axisymmetric / monoclinic variants. DG only for strong-contrast fluid-solid interfaces.
