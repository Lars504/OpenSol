# open_sol/physics/em - conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). EM gotchas; the EM loss protocol and thermo-optic
slots are in `open_sol/coupling/sub_structure_conventions.md`. Current state, no changelog.

## Registered EM types (catalogue; details in the sections below)
- `em_fdtd_ez`, `em_fdtd_hz`, `em_fdfd_ez`, `em_fdfd_hz` (default instance 'em'; two EM physics need distinct names
  and equal bc_* / pml_* settings; FDFD registered after FDTD so the FDTD material Params stay first per name; only
  em_fdfd_hz has eps_xy and its 'sum' slot eps_xy_update, section FDFD).
- `em_modes` (xy is the waveguide cross-section, so no in-plane EM type shares its model, except em_fdfd_ez / _hz in
  a thermal_noise study (same cross-section at k_z = 0 as illumination); default instance 'modes'; registered after the four in-plane types).
- `em_fdfd_bor` ((r, z) half plane, one azimuthal order m or an order list (eigen orders; driven: the 3D sum of
  off-axis point sources); family 'em', default 'em', registered after em_modes, refuses every other EM type in the
  model; 'modes' vocabulary read as eps_rr = eps_xx, eps_phiphi = eps_r, eps_zz = eps_yy) and its twin `em_fdtd_bor`
  (same operator, vocabulary and refusal; registered after it; real materials only; azimuthal_orders list = driven
  order sum, one stepper per order and source family).
- `em_fdfd_conical` (symmetry 2.9: z-invariant structure at out-of-plane k_z, all six components on the em_modes
  layout and walls; family 'em', default 'em', registered last; 'modes' vocabulary; stationary, frequency_domain,
  eigenfrequency (resonances at k_z = guided modes of beta = k_z); point / line currents of any direction, probes,
  W/m flux lines).
- Root rule "allocate where needed": poles comply in E_z, H_z and FDFD (masks cropped to their box); not yet: H_z
  `eps_x` / `eps_y`, per-cell `da` / `db`, TFSF outline masks, FDFD CSR matrices. `fdtd_hz_2d_kernels.py`
  re-exports the shared launch / Lorentz / H-coefficient / `drude_e_correction` / `q_accum_je_half` kernels, never
  copies them.

## Imports (inside physics/em; the cross-package rules are in the root file)
- Host only / never cupy: `placement`, `slab_modes` (scipy lazily, numpy fallbacks), `cross_section_modes`
  (`size_issue_for` / `line_block_bytes` GUI-thread safe), `eigenfrequency`, `body_of_revolution`, `fem_modes`.
  yee_grid, operators and dispersion take the array module as `xp`.
- Inside functions only: scipy in `sheet_conductivity`, `cross_section_modes` (and solvers.line_block),
  `body_of_revolution`, `fdtd_bor` (also the package's `_pole_step_term` in refresh_coefficients); memory_estimates
  imports `bor_line_block` / `fem_modes` in a function; selftests_optimization imports `design_density` in its check.
- Never the reverse: definition_conical -> solvers/fdfd_conical_fem, definition_modes -> solvers/modes_fem,
  definition_bor -> solvers/fdfd_bor_fem, solvers/fdfd_bor -> `bor_ports`, fdfd_2d -> `adjoint` (duck-types the
  physics); nothing `fdfd_ports` imports imports it back. `conical` uses the public `cross_section_layout` /
  `assemble_cross_section` / `fill_images` / `to_centres` of `cross_section_modes` (one writer of the wall and point
  rules). `solvers/__init__` leaves fdfd_2d and modes_2d out (they import most siblings); `physics/em/__init__`
  imports them last.
- Run-time recipes: definition_fdfd / definition_fdtd have their own (fdtd_time_step_estimate 2D only; on the cpu only
  memory items named in `_numba_stepped` are priced as numba twins: a renamed fields / coefficients item falls back to
  the numpy price); definition_common holds the host-solved types' helpers (physics_counter, role_counter,
  scaled_ops, kept_line_block_factors, resonance_eigenpairs, eigen_backend_kwargs, fem_resonance_time_estimate (the
  BOR / conical triangles estimate); validation: mesh_kind, BOTH_GRIDS,
  EIGENFREQUENCY_STUDY / OPTOMECHANICS_STUDY, raw_settings, eigen_only_feature_issues, validate_driven_point,
  resonance_table_study_issues,
  scipy_issues, host_matrix_back_end_issues, missing_source_issues, fem_size_issues, other_em_physics_labels,
  auto_mesh_issues, resolved_solved_wavelengths (definition_bor: azimuthal_order_list_issues, axis_wall_settings,
  cylindrical_material_params); apply callbacks:
  apply_field_probe / _flux_monitor / _far_field, apply_resonance_table and apply_component_point_source of BOR,
  conical and fdtd_bor: never a local copy); definition_fdtd_bor shares
  memory_estimates `_fdtd_bor_members`; memory_estimates `host_factorization` is the factorization rule of the
  host-matrix types (FDFD on triangles, conical, BOR) and `_fem_resonance_memory_estimate` the BOR / conical triangles
  estimate. em_modes, em_fdfd_bor and em_fdfd_conical solve on host SuperLU on either
  backend: recipes pass `backend='cpu'` to solver_work (cuDSS 'linear' items ignore it); line_block 'shifted' factors
  stay on the run backend.
- Self-test helpers: shared helpers live in selftests, selftests_hz, selftests_mode_ports (`slab_waveguide_spec`),
  selftests_fdfd (`cpu_specs`), selftests_cross_section, selftests_bor, selftests_fdtd_bor, selftests_modes_fem.
  selftests_operators / _modes / _cross_section / _resonance / _effective_index / _conducting_sheet / _polar /
  _material_fit import physics modules inside their checks.

## FDTD and shared EM
- 'ez': Fields2D_Ez, FDTDEngine2D_Ez, EMRenderer, eps_r. 'hz': Fields2D_Hz, FDTDEngine2D_Hz, EMRendererHz, eps_xx /
  eps_yy (never eps_r; filled from eps_r when it exists). `placement.PERMITTIVITY_ARRAYS` (=
  `permittivity_arrays`) names them, `PERMITTIVITY_SLOTS` their `<array>_update` slots (read everywhere, error
  messages too); direct MaterialGrid writes need `em.materials_changed({...})`.
- Both FDTD engines subclass `fdtd_2d_base._YeeEngine2DBase` (also `_check_cell_array`, `record_dispersive_region`,
  `add_pec_cells` shared with FDFDPhysics; the base holds inject_source (slot `<pol>_source` kept as
  `_out_of_plane_source`), update_e, add_nonlinear / add_gain): a new engine sets `_E_COEFFICIENT_MATERIALS`,
  `_rebuild_e_coefficients`, `_update_e_set`, `_accumulate_e_new_half` and the hooks `_permittivity_written` /
  `_nonlinear_added` / `_sigma_e_changed` (finalize and sigma_e writes: loss boxes), never copies. Bad cell masks
  raise TypeError (`xp=` default cupy: cpu-ready callers such as ModePhysics / EMEPhysics pass `domain.xp`).
- FDTD on both backends: `EMPhysics.xp = domain.xp`; the engine raises unless fields, BoundaryManager / CPMLSided
  share the MaterialGrid's xp; sources and monitors take `xp=` (None = default backend). Every FDTD-path kernel is a
  `core.kernels` twin (core sub file "Kernel twins"): a new kernel gets its numpy twin in the same edit, host reads
  via `to_host`. Not on the cpu yet (no refusal, they fail at the first transfer): Pockels, acousto-optic,
  semiconductor couplings. Selftests `em_fdtd_cpu_gpu` / `em_fdtd_cpu_gpu_media` (heavy).
- Stepper precision ('single' | 'double', STEPPER_ROLE; solver_plan S5): `EMPhysics(precision=)` builds fields /
  BoundaryManager / CPMLSided / TFSF with `dtype=`; the engine takes `dtype` from its fields (BoundaryManager must
  match) and keeps every field-side array in it (coefficients, spacings, CPML, pole buffers, jz_total, TFSF aux).
  Kernel scalars are numpy scalars of the field dtype (`engine._real(...)`, `_dt_real`, `_inv_dt_real`): never
  DTYPE(...) on a field-side kernel. Stay float32 under double: MaterialGrid arrays, coupling slots, q_accum, PEC keep
  masks, source profiles / calibrations, FieldProbe traces, box DFT spectra (complex64; line spectra complex128),
  subpixel coefficients (scattered into float64 ca / cb), the Courant check; renderers return float32, state()
  float64. Refused under double: nonlinear_medium / gain_medium (FeatureType.precisions ('single',); add_nonlinear /
  add_gain raise). Selftest em_fdtd_double_cavity_phase (heavy).
- MaterialGrid allocates no array by itself: EM arrays come from `register_em_arrays` (FDTD engines, FDFDPhysics) and
  the EM material_schema, so an eps_r write before any E_z solver exists raises. Code outside EM checks
  `property_names()` first.
- EM solvers take cell geometry and vocabulary from `placement` (`probe_cell`, `point_source_cell`, `flux_line_cells`,
  `tfsf_box_cells`, `port_cells`, `sheet_line`, ...) and `boundaries.solver_boundary_config`: a new EM solver calls
  them, never copies them (messages and checking order stay those of EMPhysics).
- Yee spacing: kernels, CPML kappa and A1 operators use `yee_grid.grid_cell_widths(domain.grid)` via
  `yee_inverse_spacings`, NOT `Domain.cell_widths()`. `*_e` = 1/w_i (centre), `*_h` = 1/mean(w_i, w_(i+1) mod n) (hi
  face; last = periodic wrap). Non-periodic: the last face row is never updated and cell 0 reads its missing lo face
  as 0, so the grid edge is PMC (ez) / PEC (hz) half a cell beyond the outermost centres (only cpml sides see it).
- Source times = the centre of the update they follow (E: t+dt/2; H: t), never the new field time: dt/2 late makes
  DFT / S(omega) lead the FDFD phasor by e^{+j omega dt/2}. Sources take the time as given (no offset).
- ez step: update_h (+ CPML, `apply_h_walls`), magnetic injection at t, `apply_h_walls`, update_e (poles from E^n +
  E^n loss half, Yee, CPML / BC post, corrections), electric sources at t+dt/2, `ez_source`, `apply_ez_walls`, PEC
  keep, E^{n+1} loss half (`_accumulate_e_new_half`), monitors at t+dt.
- hz: H_z at the centre, E_x on its hi-y face, E_y on its hi-x face; the last face row updates only on periodic axes.
  Step: `refresh_coefficients`, update_h, `inject_magnetic` at t, `hz_source`, `apply_hz_walls`, update_e,
  `inject_electric` at t+dt/2, `apply_exy_walls`, PEC keep, E^{n+1} loss half, monitors: E at t, H_z at t - dt/2.
- PEC: every E-coefficient rebuild folds the keep mask into ca = cb = 0 (`_fold_pec_keep`); the per-step PEC keep
  re-zeroes only `_pec_box` after injection and wall re-pin. A new coefficient path must end with `_fold_pec_keep`.
- Conformal PEC (solver_defaults.conformal_boundaries, S7.6; `conformal_pec.py`, host): `pec_cells` is the staircase
  mask at once (ports, poles, calibrations, FDFD read it); the engine gets the conformal data once in `finalize`.
  E_z: H faces into a conformal cell scale their curl term (`set_conformal_faces`, wall distance floored); H_z
  (Dey-Mittra): face keeps only where the face lies wholly in metal, cut cells weight their curl
  (`set_conformal_cells`, area floored). Both floors keep each Gershgorin row <= (0.99 x 2 / dt)^2 so dt stays stable
  (selftest conformal_pec_operator); at courant 0.99 the floors bite (disk-cavity errors ~ 1 / R), courant_factor 0.7
  relaxes them. Kernels (`solvers/fdtd_conformal_kernels`) run right after each Yee H kernel, no wraps. Not
  conformal: BOR, the Courant check, calibrations.
- Conformal poles (same option, FDTD only; `conformal_poles.py`): `definition_common.write_dispersive_materials` reads
  `BuildContext.conformal_locator_of` and passes ConformalPoleWeights to every pole's `add_dispersive(weights=)`
  (E_z: cell inside-area share; H_z: half-cell shares combined with face_average shares). `dispersive_regions` keeps
  the cell mask; the Courant check reads `_pole_stability_regions`. Exact first order for E tangential only (hz E
  normal keeps the arithmetic share). Not conformal: FDFD / em_modes poles. Selftests em_conformal_pole_weights,
  em_conformal_drude_film (heavy).
- FDTD step graphs (gpu, `fdtd_2d_base.STEP_GRAPHS`, as em3d `_fixed_block`): `update_h` runs Yee + conformal + H CPML
  + the post's H rows of every field set as one CUDA graph (`_h_block`), `_update_e_set` Yee + E CPML + E rows per set
  (key ('e', set index)); the first step of a key runs eagerly, the second captures, later ones replay. Drude /
  Lorentz poles join the E graph (pole updates, E step, pole corrections; key ('e', set, n_drude, n_lorentz): poles are
  only appended) unless loss accumulation, a gain / nonlinear medium or the eager row copies sit between them. The wall
  re-pins after the sources, the other pole paths, gain / nonlinear media, PEC keep and monitors stay eager, so
  the engines call the CPML (`_cpml_post`) and `apply_*_walls` apart instead of `apply_*_post`. Outer-face rows are
  one fused launch per call (`boundaries` wall_centre_rows / wall_plane_rows: corners = s_x s_y x the inner corner
  cell, what the sequential row copies give), face types read once at BoundaryManager construction; with < 3 cells
  along a pinned axis (`fused_walls` False) the old cupy row copies run eagerly outside the graphs (a column-row copy
  overlaps its source row in cupy's bounds check, which copies the input: an allocation inside a capture). Fields, psi, ca / cb / da / db, the 1D spacings
  and the conformal arrays are rewritten in place only (a new `set_conformal_*` tuple changes the H key); a new
  replaceable input of a block joins its key. A failed capture warns and steps eagerly for good. The time recipe
  books 1 + n_sets graph launches on the gpu instead of the weighted Yee / CPML launches.
- FDTD probes (gpu): the FieldProbes among the monitors are taken over at their first step (`_take_probe_launch`:
  `core.monitors.FieldProbeRows`, float32 / float64 cupy fields of the engine) and sampled by one launch at the end of
  the real set's E graph (`_e_graph`: the launch id joins the key); `_finish_step` stamps the row and skips their
  `record`. The record precedes the eager tail, so `_probes_need_resample` re-samples the row (one eager launch) when
  a probe reads an E component (`_E_COMPONENTS`) and the tail may have written one: wall re-pin, PEC keep, non-fused
  rows, poles with the loss halves, media, an E-side source array (ez_source) or an E-side injection (only an exact
  PointSource (ez) / ElectricDipoleSource (hz) at DTYPE(amplitude temporal(t_half)) == 0 counts as none, the value
  its injection skips at). A new eager E-side writer after the E graph joins that check. H probes never re-sample.
  The time recipe books no probe launches on the gpu.
- Both engines re-pin walls after injection (new engines must too): a source on a pinned row / face does nothing (line
  / point / mode sources warn). The re-pin is skipped (`_walls_need_repin`, exact: the rows are idempotent) only when
  every source is exactly a PointSource / MagneticLineCurrentSource / ElectricDipoleSource whose cells are >= 2 cells
  inside every pinned axis (`BoundaryManager.clear_of_walls`) and no source array, pole, gain or nonlinear medium
  writes after the fused rows; a new source type or post-wall writer re-pins until it is listed there. An E_z symmetric face next to the copied row is frozen, not re-zeroed, so an
  injection there accumulates. `engine.time`, `q_accum_time` are host float64.
- H_z face values (eps, sigma_e, pole weights, PEC) are defined only by `yee_grid.face_average` (width-weighted dual
  mean; uniform = plain mean bit for bit), `face_pec` (either cell; not em_modes), `face_partner`; FDFD / slab modes /
  sheet calibrations must use the same weighting: port modes are exact only while they agree. The fused
  `e_face_coefficients` kernel inlines the mean (`face_shares`, no FMA): a weighting change must update it too
  (em_operators_consistency part E). Faces -> centres: `yee_grid.lower_face` / `lower_face_partner`.
- Sources are current-calibrated: point = filament current I [A]; line / gaussian beam = sheet current radiating
  `amplitude` V/m each side in the local medium (a span end on a cpml window edge continues through the PML, so a
  window-wide line is no slit): K = 2 amplitude cos(kappa/2) / eta(omega) of the line cells (validation warns when
  media along / past the line differ, and for off-centre or truncated beams), kappa = `yee_grid.sheet_launch_factor`;
  FDTD uses |eta / cos| at the centre frequency, FDFD the exact complex value. The TFSF aux source carries the same
  cos(kappa0/2) (raises at <= pi cells per wavelength; Mur ends need the carrier-matched S, plain S echoes ~3e-3).
  hz: magnetic line current K_m [V], dipole M [A], sheet currents on the hi face (a line in the last cell of a
  non-periodic axis raises). `add_point_source` ez only; magnetic line current / dipole hz only.
- Calibration is lazy (dirty flag on source add, `materials_changed`, slots); `EMPhysics.step` = refresh, recalibrate,
  advance; before `finalize` calibrated sources inject 0; `engine.advance` skips calibration. Recalibration keeps the
  sheet amplitude and K_m fixed, so power drifts with n(T); I and M give n-independent power. cb must stay live
  (freezing it freezes only sqrt(eps / mu_r)). Help texts state this (`definition_common`: POINT_SOURCE_SIGN,
  SHEET_POWER_DRIFT, MAGNETIC_CURRENT_DRIFT, beam_params, sheet_amplitude_param).
- Coefficient rebuilds: E_z at once; H_z `eps_xx/yy_update` sets a dirty flag for `refresh_coefficients()` (cb is
  also the Drude correction factor); `materials_changed` rebuilds at once ({mu_r, sigma_m} -> da / db).
- `EMPhysics.finalize` raises unless eps*mu_r >= (C0 dt)^2 (1/dx_i^2 + 1/dy_j^2) + mu_r P everywhere incl. PML, P =
  sum of (wp dt)^2/4 (Drude) or delta_eps (w0 dt)^2/(4 - (w0 dt)^2) (Lorentz): leap-frog poles are NOT stable
  whenever Yee is (eps_inf = 1 aluminium needs dx <= ~5 nm; selftest em_drude_courant_bound). Scripts skipping
  `finalize` and runtime slot rewrites are unchecked.
- J.E accounting only while `engine.enable_q_accum` (`q_accum` None before the first enable; `jz_total` exists only
  with a Drude pole). `absorbed_power` = current window, else `last_q_accum`, else zeros; without em_heat_joule the
  window is the whole run. The model enables it for track_absorption or any reference (`_q_accum_referenced`).
- Pole stepping (em3d too): J += damping J + drive E^n (velocity form, damping = c1 - 1: never a float32 c1 ~ 1), P
  from the pre-update E^n; then E^{n+1} = E_pred - cb J^{n+1/2} (Lorentz - (cb/dt) dP): cb carries the sigma_e
  factor 1 / (1 + k), never 1 / eps (over-couples a conducting cell vs the FDFD chi). Never update J from the
  predictor E^{n+1} (unstable). Lorentz buffers P and dP: never a1 P^n + a2 P^{n-1} in float32.
- Pole loss must be time-centred, as two halves (E^n at the pole update, E^{n+1} at the end of advance after
  corrections, injection, re-pin and PEC keep, so they telescope on source cells): ez `q_accum_je_half` (twice),
  `q_accum_lorentz_half`; hz `pole_power_*_half`, split per face by `q_accum_hz` (exact only with width-weighted face
  values).
- `add_drude(set_background=True)` writes eps = eps_inf under the mask; `add_dispersive` keeps it. mu_r / sigma_m
  reach da / db only via `materials_changed` / `finalize`. E_z ohmic loss lives on the box of sigma_e > 0 cells, H_z
  `q_accum_hz` on the loss box (sigma_e > 0 cells + the pole power face boxes grown one cell along the normal,
  `_loss_box_launch`, refreshed after finalize / a sigma_e change / a new pole or nonlinear region): a direct sigma_e
  write without `materials_changed` loses heat outside the old box. H_z poles share `_pole_regions`
  per distinct mask (edit once per region); H_z pole faces cache box views (`_attach_face_views`) of ex / ey, cb_x /
  cb_y and the power totals: rewrite those arrays in place only, never reassign.
- BoundaryConfig PML cells must equal `Domain.npml_sides`; value equality, unhashable. `_validate` runs at
  construction and again in `solver_boundary_config`. BoundaryManager / CPMLSided are per polarisation (the engine
  refuses the other one: psi signs differ, it would amplify).
- CPML layout only in `boundaries.cpml_side_rows` / `cpml_axis_profiles` (ValueError below n_layers*npml + 2 cells). E rows at centres
  (lo 1..npml, hi n-npml-1..n-2), H rows at hi faces (lo 0..npml-1, hi n-npml-1..n-2); E rows 0 / n-1 and H row n-1
  never stretched, so the layout mirrors exactly (E j <-> n-1-j, H j <-> n-2-j): mirror half models need that (em3d
  `cpml_slabs` takes the same rows). sigma_max is the vacuum optimum (users tune `pml_sigma_factor` in claddings).
  The psi term takes the engine's live Yee coefficient (ez db / cb, hz db / cb_x / cb_y), never dt / eps: else a
  lossy / magnetic cell crossing the PML reflects. psi runs fused after the Yee kernel (valid only because a Yee
  kernel never writes its own curl inputs).
- Bloch (A6 FDTD; EME refuses k != 0 via `operators.check_zero_bloch`; `cross_section_modes` handles it for em_modes,
  conical and em_fdtd_3d ports, other callers check themselves): F(x + L) = F(x) e^{-j k L}, L = sum of grid widths.
  `has_bloch_phase` builds a second, imaginary `_FieldSet` (`fields.imag`, own psi via `BoundaryManager.partner()`;
  else None); only the wraps mix the sets (a real engine passes partner = itself, bit-identical). Pole buffers are
  lists by `_FieldSet.index`. Sources drive the real set only, except `BlochLineSource` /
  `BlochTangentialSheetCurrentSource` (-sin(k.r) into the imaginary set at the electric source time); `update`
  monitors add j F_im, so +omega is the +k response; probes, renderers, state, couplings read the real set.
  enable_q_accum raises. Known gaps: hz face-to-centre collocation of cell 0 lacks the Bloch phase; mode ports solve k
  = 0 (validation warns). New wrap kernels must take partner + wrap arguments; C5 media refuse k != 0.
- Walls: E_z pec = 0 at the outermost centre, pmc = copied from the neighbour; H_z pec = tangential E 0 on the inner
  face, pmc = H_z 0 at the outermost centre: ez and hz walls are half a cell apart. Mirrors (symmetry 2.5; symmetric
  = E even = PMC, antisymmetric = PEC): `npml_sides` pads `MIRROR_GHOST_CELLS` (1) ghost cell (`ghost_faces`: edge
  cell's width and material) so the plane is the inner face of the outermost cell, as for walls (PHY-3). H_z:
  symmetric hz[0] = -hz[1], antisymmetric hz[0] = hz[1] + tangential E 0; E_z: ez[0] = +/- ez[1] (hi faces alike,
  rows -1 / -2); `apply_h_walls` zeroes the pmc faces 0 / n-2. A mirror
  face's padding is no CPML: code sizing CPML reads the bc type, never npml_sides alone (`_cpml_psi_bytes`).
- Unfolded results (symmetry 2.5, `core.mirror`): flux multiplies by `placement.line_mirror_factor` (extra
  'mirror_factor'; monitor objects stay raw; flux adjoint objectives carry it), Session 'integral' likewise; far
  fields add the image contour (`NearToFarGeometry.images`, `core.mirror.image_sign`; mirrors on both faces of an
  axis raise). Not unfolded: 2D mode-port powers, source 'power', probes, DFT monitors. Every signed renderer
  FieldInfo declares `parity` and `staggered`: a new signed field must too, else unfolding raises.
- TFSF: FDTD and FDFD on uniform or graded grids (`_PlaneWaveBox(cell_widths=)`; FDFD Bloch ky needs uniform y); +x;
  FDTD vacuum incident field, FDFD the background's (box edges in one uniform medium: `tfsf_outline_warnings`,
  `validate_tfsf`); one scattered-field cell inside the window (FDFD: or a full periodic y axis), except a y mirror
  face of the wave's parity (ez symmetric, hz antisymmetric; `placement.tfsf_open_faces`; other mirrors raise), where
  the box opens (selftest em_tfsf_mirror). Amplitude = incident E_z (hz: E_y). NF2FF: closed contour through the
  centres of columns / rows x0, x1 = cell(hi)+1, y0, y1 (x0, y0 >= 1), hi-face field averaged onto centres (any
  width: graded meshes fine). `monitors.near_to_far_pattern` is the one far-field formula (2D prefactor included);
  FDTD re-times H spectra by e^{+j omega dt/2} first. far_field 'power': FDTD |pattern|^2 / eta0 'a.u.' (no 1/2),
  FDFD / (2 eta0) 'W/(m rad)'; amplitude_re / _im give power = |amplitude|^2
  (`core.mirror.combine_sector_far_fields`).
- Every scripting line feature takes `(axis, position, span)`, axis 'x' = line along x at y = position; only the
  internal `FluxMonitor / HarmonicFluxMonitor.axis` is the normal. `flux()` = Yee discrete Re(E x H*) (no 1/2)
  through the line cell's hi face, > 0 toward +x/+y. DFT phases use the actual sample time (hz H_z: t - dt/2). hz
  DFT / renderer E collocate to centres, hz `state()` omits staggered ex / ey; probes stay raw (H dt/2 older); ez
  renderer hx / hy stay raw face samples.
- Every EM renderer exposes e_abs, e_intensity and h_abs at centres (`renderer._MODULUS_FIELD_INFOS`): keep them on any
  new EM renderer. Renderers of centre-field dicts (ModeRenderer, BORRenderer, ConicalRenderer) subclass
  `renderer.CentreFieldRenderer`; the FDTD pair's dispatch is `_FDTDRendererBase._fdtd_field` (`material_names`,
  `_polarization_field`). FDFD hz also exposes ex_re / ex_im / ey_re / ey_im (`_IN_PLANE_E_PART_INFOS`); EME omits them.
- Monitors take `polarization` ('ez' default); dft_fields / far_field carry 'quantity' (plots, GUI read it); hz
  dft_fields with freq_index adds 'hz_intensity' (not in loaded Results). flux_spectrum 'unit': FDTD 'a.u.'
  (dt-weighted), FDFD 'W/m'. When all sources share one waveform (coupling drives not counted), results add
  cw-equivalents (`results._shared_waveform_spectrum`: 'power_per_cw', 'power_per_cw_*', 'intensity_per_cw_factor').
- DFT spectra are dt-weighted (flux ~1e-40): products in float64 on the host (float32 flushes to 0), relative
  thresholds only. Line monitors share `monitors._LineSpectra` and accumulate in complex128 (float32 loses
  near-cancelling fluxes); DFT boxes complex64. DFTMonitor / `_LineSpectra` build their kernel argument views once per
  field object (`_launch_list`): field and spectrum arrays are rewritten in place, never reassigned. DFTMonitor `downsample`: point samples (`dft_sample_slice`, no block
  average), `x_slice / y_slice` stay the whole box. Result maps use `viz.visualization.dft_map_edges`, never
  `viz.common.edges_from_centers`.
- DFT decimation: `update` monitors sample every m = `sample_interval` steps, weighted m dt; `EMPhysics.finalize`
  (only at n_step 0) sets m = `monitors.dft_sample_interval` (largest m with 1 / (m dt) >= max monitor frequency +
  max source `temporal.spectral_limit()`, times NONLINEAR_HARMONIC_ORDER 3 with a nonlinear medium: its SHG / THG
  would fold into the monitor bands; cw -> 1). Coupling drives are not in the bound; monitors added after
  finalize keep m = 1. Any S(omega) compared with a monitor must use its grid (`waveform_spectrum(..., m)`).
- Convention (E7): exp(+j omega t); DFT accumulates exp(-i omega t) dt; lossy Im < 0; PML s = kappa + sigma/(alpha +
  j omega eps0). `susceptibility(omega)` is the continuous chi (FDTD differs at O((omega dt)^2)).
  `dispersion.complex_permittivity` takes one component; H_z callers face_average first.
- nk_fit (A7, `material_fit`, host only): eps = (n - j k)^2, bounded least squares with a closed-form Jacobian
  (`fit_drude_lorentz.jacobian`: a pole formula or residual scaling change must update it); eps_inf >= 1, poles >= 0,
  omega_0 <= 5 x band max (Courant), Drude only when Re eps < 0; errors scored after float32 rounding.
  `DISPERSION_POLE_PARAMS['nk_fit']` has no pole class: every path sees the expanded drude / lorentz dicts.
  Validation (`nk_fit_issues`) warns when `model_band` leaves the fit range. Selftest em_nk_fit.
- A1 operators: flat index i*ny + j; `dx_e / dy_e` hi-face -> centre, `dx_h / dy_h` centre -> hi face; non-periodic
  `*_h` row n-1 zero, `*_e` row 0 = +w_0 (as the kernels; complex symmetric up to row scaling). omega None = real
  kappa stretch as FDTD (selftest em_operators_consistency). pec / pmc / mirror faces are plain edges (caller sets
  outer rows); Bloch phases the wrap entries (`bloch_wrap_phases`; then not complex symmetric, nothing relies on it).
  `yee_operators(xp=cupy)` assembles with scipy on the host and uploads each CSR once (the cupyx coo / kron chain was
  ~50 ms of launches per call); without scipy it assembles on the device (same canonical CSR, same values).
  FDFDPhysics on a Cartesian grid takes them from `KeptYeeOperators` (kron patterns and entry maps kept per grid, a
  new omega rebuilds only the four host 1D differences and gathers them; same pattern and values, the ops of every
  omega share the kept index arrays: never modify ops in place); polar grids call `yee_operators` per omega.
  `KeptYeeOperators.helmholtz` (FDFDPhysics._helmholtz) builds the Helmholtz sum on a kept term plan (`_HelmholtzPlan`:
  per output entry its product terms, gathered and summed level by level, no SpGEMM / geam): one fixed canonical
  pattern for every omega and weight set (zero weights stay explicit zeros, unlike `helmholtz_operator`, which drops
  them), so a wavelength scan refreshes cuDSS on one plan; the operators share its indices / indptr: never
  eliminate_zeros / sort them in place. eps_xy couplings are added after it; ops it did not make fall back.
  `helmholtz_operator`, `pin_cells` = D A D + (I - D), `pin_image_cells` (pinned image folded into its neighbour's
  diagonal). Never multiply by a sparse diagonal (`diags(...) @ A`): a cuSPARSE SpGEMM whose workspace grows with the
  rows (5e6 rows asked 6.5 GB); scale stored entries (`_column_scaled`, `_kept_entries`, `_entry_rows`) keeping the
  old product order so values stay identical.
- Slab modes (`slab_modes`, A3): `SlabLine` = one frequency, m cells + m+1 faces (face k = lo face of cell k); ez q =
  eps mu_r, hz eps_t / 1/eps_l (`face_average`); PEC cells inactive. Branch: Re k_t > 0 if Re k_t^2 > 0, else Im
  k_t < 0. Never force Im k_t <= 0: a guided mode touching a PML gets a tiny Im k_t^2 of either sign and would flip
  to backward. `dt` / `normal_spacing` give the FDTD's discrete omega / k. PML-localised modes dropped
  (`drop_pml_modes`; eme passes False), the rest by descending Re k_t^2 then Im k_t^2; fewer than n_modes may return.
  Normalised to +-1 W/m of discrete Yee flux (incl. cos(omega dt/2)), largest |u| real > 0 at `peak_cell` (lo-most
  within 1e-6, so mirror peaks tie deterministically; compare asymmetric lines only after `align_phase`); every mode
  of a solve is normalised in one pass (`_normalised_modes`: one row per mode, line arrays once; the profiles are row
  views of one block), `_normalised_mode` is its 1-mode call (track_slab_mode).
  `mode_overlap` is unconjugated (0 between modes of one line); degenerate complex pairs are bilinear-orthogonalised
  (`_orthogonalise_degenerate`; selftest em_slab_degenerate_orthogonal). Complex lines > 600 cells shift-invert at
  (omega_t/C0)^2 max Re q (can miss modes far above). Plasmonic hz: selftest em_slab_spp_discrete.
- Mode ports (`mode_ports`): `PortLine` maps a line to a SlabLine like an industry-standard port boundary: a span end at
  a CPML window edge continues through the PML; at a grid-edge wall or mirror the pinned outermost cell is dropped (end
  condition from the wall type); a full periodic span is periodic; other ends are artificial metal (warned when the mode
  exceeds 1e-2 of its peak there). eps_t is the line cell's value (guide assumed invariant along the normal). PortLine
  serves FDTD and FDFD ports and never reads `physics.engine`.
- ModeSource = one-sided TFSF line (interface face f = c-1 positive / c negative): magnetic correction uses the
  E-type incident at t, electric the H-type at t+dt/2; sign = sign_n x direction (ez x -1, y +1; hz x +1, y -1);
  incident(t) = Re(P) s(t) + Im(P) `temporal.quadrature(t)` (gaussian / cw only). After the first step the modes are
  re-solved when the line's materials drift above `_RESOLVE_DRIFT` (1e-3), tracked by overlap
  (`tracked_mode_number`; `mode_number` stays the request). `EMPhysics.runtime_warnings` (append-only) gets a
  monitor's first re-solve, new mode warnings, and a failed re-solve (keeps the old modes, stops following). Before
  the solve nothing is injected and results raise. `finalize_warnings` holds port warnings incl.
  `PortLine.placement_warnings` (a face on a stretched CPML H row is ignored by the correction; warned, not refused).
  Gaussian env cos is no Hilbert partner of env sin, so ModeSource warns above bandwidth 0.6. Exact at f0 only:
  fraction_* report off-f0 leakage as reflection / transmission loss (help texts say so).
- ModeMonitor: DFT of u and t at the E time, H-type spectrum re-timed by e^{+j omega dt/2}; the mode is solved at the
  middle frequency and followed outward by `slab_modes.track_slab_mode` (full solve + `track_mode` below overlap
  0.9; finalize warns). power_* = 2 |a|^2; fraction_* / phase_* only with exactly one ModeSource, from its sampled
  S(omega) on the monitor's grid (`core.sources.sampled_waveform_spectrum`); NaN where |S|^2 < 1e-6 |S(f0)|^2 (the
  source's carrier, never the band max) or, cw, |f - f0| n_step dt >= 1.
- resonance_monitor (A6; shared row in `model.definition_resonance`, numerics `core.resonance`):
  `definition_resonance.em_resonance_probe` is the EM ResonanceProbe for em_fdtd_ez / _hz and em_fdtd_3d (H
  components 0.5 step delayed); apply adds a plain FieldProbe. Amplitude / phase refer to start_time (H: record
  time - dt/2).

## em_fdtd_ez / _hz on triangles / cuDSS: decision (agent_help batch, 2026-10-23; nothing implemented)
- Decision: tensor Yee only, as em_fdtd_3d (em3d sub file "em_fdtd_3d on tetrahedra / cuDSS"); `grids =
  ('tensor',)` refuses a 'triangles' mesh. The 2D steppers hide no linear solve worth cuDSS (ports / TFSF are 1D
  tridiagonal lines). 2D DGTD only if a reference case misses its target with the cheaper local fixes (conformal
  PEC / pole weights, subpixel smoothing, graded meshes); 2D FETD rejected (a solve per step): conforming
  frequency-domain answers belong in em_fdfd on triangles.

## Nonlinear and gain media (C5; solvers/fdtd_nonlinear, solvers/fdtd_gain, definition_nonlinear)
- FDTD only, domain features `nonlinear_medium` (chi2 d_eff + poling, Kerr n2, TPA beta, TPA free carriers) and
  `gain_medium` (four-level two-electron, Chang-Taflove) of em_fdtd_ez / _hz, not material properties (they allocate
  only on their selection's bounding box). FD types: unknown-feature error; validation errors for a frequency_domain
  study and for Bloch k != 0 (`add_nonlinear` / `add_gain` raise too: E^2 of the real set is not |E|^2).
- `self.nonlinear` is rebuilt per add (one box per cluster of touching regions, `_region_clusters`, across a periodic
  wrap; later regions win; carrier densities reset); `self.gain_media` one GainRegion per feature (overlaps add).
  `_update_e_set` order: nonlinear `store_previous`, poles, gain `update_polarization`, Yee + BC post, Drude / Lorentz
  / gain corrections, nonlinear `correct` (must stay the last correction: it subtracts the linear update x_lin), gain
  `update_populations`; sources, walls, PEC after. Cached box views of `_nonlinear_components()` (hz eps absolute,
  `eps_scale` 1 / EPS0) assume those arrays are only rewritten in place.
- chi2 = 2 d_eff x per-cell poling sign, NOT scaled by n0; one scalar per component, no tensor / cross terms (crystal
  tensors: physics/nlfd). n0^2 is the live permittivity array (dispersive: eps_inf); chi3 / TPA / FCA / FCD scale
  with n0^2 / n0 in the kernel. Carrier density and populations are float64 (dt / tau and kappa dt^2 vanish in
  float32). hz: per in-plane component on its own faces (E_i^2, not |E|^2): exact for a field along one axis only; an
  interface face of weight w carries w x the response but the bulk carriers / populations (`_PER_MATERIAL_POWERS`;
  fdtd_gain divides by its weight).
- The TPA conductivity must be time-centred, g = c_tpa ((E^n + E^{n+1}) / 2)^2 inside the implicit solve: a lagged
  (E^n)^2 gives a Kerr phase error (-6 % at beta / (2 k0 n2) = 2) that loss-only checks miss. A dispersion-free slab
  phase-matches the third harmonic, which with TPA cascades back as extra phase loss (analytic_nonlinear_tpa_kerr_slab).
- Loss: TPA + FCA work (time-centred): ez into q_accum, hz into the face pole power totals (`_pole_cell` /
  `_pole_power_boxes`: a pole face next to a nonlinear-only cell splits its pole power with it). Gain work is not
  counted; gain / nonlinear media are outside the build Courant check, sheet calibrations and port mode solves.
- Self-tests selftests_nonlinear (ez and hz; n0 = 2 pins the n0 factors): Kerr, TPA, carriers, small-signal gain,
  chi2 SHG vs the lattice Green's sum. Template `nonlinear_silicon_slab`; threshold example
  analytic_gain_threshold_cavity.json: a tight threshold reference must use the leap-frog line (W = (2 / dt)
  sin(omega dt / 2), gamma sin(omega dt) / dt, sigma cos(omega dt / 2)); the continuous line is 7 % off at omega dt
  0.22. Not done: lasing above threshold (resonance fit drops growing poles), hz cross-component Kerr / TPA terms.

## Phase-change medium (C5; phase_change, definition_phase_change, coupling/definitions_phase_change)
- Feature `phase_change_medium` of all four in-plane types (linear at optical time scales: no superposition breaker);
  `PhaseChangeMedium` in `solver.phase_change_media` (EMPhysics and FDFDPhysics). It OWNS eps / sigma_e of its cells:
  each update rewrites Re eps into every permittivity array and -Im eps omega0 eps0 into sigma_e, then one
  `materials_changed`.
- Driven only by coupling `heat_em_phase_change`, whose transform calls `update_phase_change(em, T, heat.time)` and
  returns None (a slot array cannot carry the heat time). It declares '' writes of every permittivity slot, so the
  writer rule refuses thermo-optic / Pockels / field_transfer / state-dependent eps on that physics. A repeated time
  (end-of-run flush) is a no-op.
- `Kinetics` (xp-agnostic): theta += k(T_end) dt (JMAK: exact for isothermal steps of any split), k capped at 1 /
  tau_min above T_fast (plain Arrhenius would recrystallise every melt-quench), T >= T_melt sets theta = 0. GST-225
  numbers live once in `phase_change.GST_225`. Switching needs em_fdfd_* + transient heat; FDTD validation warns for
  runs far shorter than tau_min. Self-tests phase_change_kinetics, em_hz_phase_change_anneal; example
  analytic_phase_change_jmak_anneal.json; template `phase_change_gst_switch`.

## Harmonic FDTD (em_fdtd_* in a frequency_domain study; harmonic_fdtd.py)
- `model.study._run_harmonic_pulse` calls `EMPhysics.attach_harmonic(frequencies)` after finalize, before the first
  step (else ValueError): a `HarmonicAccumulator` (DFTMonitor, complex64) appended to `monitors` with its own
  `dft_sample_interval`. Every source must share one gaussian / ricker waveform (`_shared_pulse`,
  `definition_harmonic_fdtd`). Only the FDTD physics steps (couplings reading frozen physics are refused); auto_shutoff
  after 2 t0 or max_duration.
- `normalize` divides once by `results._shared_waveform_spectrum` (S over steps 1..n_step), so phasors follow the FDFD
  convention (phase 0 = DFT / S); a sample grid change or a step before attach breaks that. Feature monitors keep
  unnormalised spectra. Renderer fields `requires_study='frequency_domain'`; `harmonic_primary` is the default stored
  field; `absorbed_power_harmonic` uses the continuous chi (hz per component, unlike the FDFD face mean). The sheet
  calibration is exact at f0 only (away from it: the sheet's |eta / cos| mismatch).
- Self-tests em_harmonic_fdtd_vs_fdfd (pins both source signs), em_hz_harmonic_fdtd_slab (only the last wavelength
  stays live: read stored snapshots); template cylinder_harmonic_fdtd. Result `harmonic_pulse` -> summary.json (api).

## FDFD (em_fdfd_ez / em_fdfd_hz)
- `FDFDPhysics` shares EMPhysics' attributes (domain, sources, monitors, dispersive_regions, pec_cells,
  permittivity_arrays, ...), so PortLine, couplings and the Visualizer treat both alike. `step` solves only when
  dirty, `solve_harmonic` always; `inject_source` takes only eps slots (a per-step source has no harmonic meaning); hz
  creates missing eps_xx / eps_yy from eps_r.
- time_dependent study: allowed beside a time-stepped physics (dt_limit inf; steps every `every` ticks, re-solves only
  when dirty); `_time_dependent_issues` errors beside any em_fdtd_*.
- Backends: every array on `self.xp` (cupy + cupyx CSR or numpy + scipy CSR), masks included; cpu = scipy_splu only;
  eigenfrequency gets host copies. Sources keep host arrays until the first solve (`fdfd_sources._on_backend`). Must
  stay backend-neutral (shared with FDTD): `materials.register_em_arrays`, PortLine `_host_rows`, PhaseChangeMedium,
  `_phasor_part(xp)`, adjoint.py, DesignRegion. Automatic without scipy is the only cpu refusal
  (`direct_solver_issues`). Self-tests em_fdfd_cpu_gpu, em_fdfd_adjoint_cpu_gpu.
- Conventions (harmonic.py header): exp(+j omega t), u at cell centres on the A1 flat index, assembly / solve /
  `solution` complex128. Source sign: every FDTD source ADDS +c J s(t), so J_F = -J_std, M_F = -M_std; the FDFD keeps
  the FDTD amplitude meanings: phase 0 = FDTD DFT / S(omega), FDTD sin carrier = phase -90, `phase` [deg] 0 = cos. A
  line current gives E_z = +(omega mu0 I / 4) H0^(2)(k r), H_z = +(omega eps0 K_m / 4) H0^(2)(k r)
  (em_fdfd_line_current_hankel).
- Assembly: ez face weight 1/mu of the face's lower cell; hz 1/eps_face from `face_average` of the complex eps, 0 on
  `face_pec` faces; poles add their continuous chi. PEC: ez rows pinned; the rhs is zeroed on PEC cells after every
  source in both polarisations (an hz PEC cell keeps the uncoupled k0^2 mu row, so H_z = 0).
- Conformal PEC (S7.6): `ConformalFDFD` built once on the host, reset by any new PEC region; pec_cells = the staircase
  (ports, sources, TFSF / eps_xy checks read it). ez: face weights into the metal x 1 / max(t, CONDITIONING_FLOOR
  0.02). hz (Dey-Mittra): `face_fractions` / `cell_area_fractions` (>= 0.02), area-scaled operator (complex symmetric,
  so reciprocal), while `weight_hi_*_faces` stay physical for face_fields; face_current_rhs, q_abs, adjoint dA and
  eigen mass take f / a. hz `system.pec_cells` = fixed cells only (cut cells with a metal centre stay unknowns): rhs
  zeroing, wall-fill re-zero, eigen pinning and adjoint fold read `system.pec_cells`, never `physics.pec_cells`. Not
  conformal: eps_xy couplings, slab-mode ports, sheet calibrations. Self-tests em_fdfd_conformal_disk_eigen,
  em_fdfd_conformal_reciprocity.
- hz off-diagonal eps_xy (harmonic.py header; the only EM type with it): material `eps_xy` + private float32
  `_coupled_eps_xy` of slot `eps_xy_update` (OFF_DIAGONAL_SLOT), real, set to 0 in PEC cells and the outer two cell
  rows of every wall / mirror face (`pin_image_cells` folds a pinned cell into ONE neighbour: a 9-point stencil there
  would be wrong). Faces hold the effective 1/eta_ii = eps_ii - eps_xy^2 / eps_jj; cell_eps_xx / _yy stay raw for
  q_abs. `off_diagonal_face_couplings` drops zero-weight / never-updated faces (`active_coupling_faces`), keeping the
  operator complex symmetric. `HarmonicSystem.off_diagonal` None when all zero (one `cp.any` sync). Not seen by:
  slab-mode ports, sheet calibrations, TFSF background (`_off_diagonal_line_warnings` at finalize only),
  em_fdtd_hz / em_modes / BOR (couplings drop the xy row with a crystal warning), the NLFD cascade adjoint
  (`_check_differentiable` refuses). Renderer `eps_xy` parity 'even'. Self-tests em_fdfd_hz_rotated_slab,
  em_fdfd_hz_anisotropic_reciprocity.
- Dirty flags: `_system_dirty` (materials_changed of a read array, slots, PEC, poles, new omega) reassembles and
  refactorises; `_solution_dirty` (that or a new source) re-solves. scipy_splu at an unchanged omega (Picard step)
  solves by host GMRES preconditioned with the old factor, refactorising when the residual misses `RECYCLE_RTOL`
  within `RECYCLE_MAX_ITERATIONS`; a new omega always refactorises. A1 operators cached per omega (`_yee_ops`).
- Linear solver: 'linear' role (`SparseDirectSolver`, `FDFDPhysics(linear_solver=)`), 'auto' resolved at finalize;
  scipy_splu without scipy raises naming 'pip install scipy'. Eigenfrequency uses the 'eigen' role (also
  optomechanics). Size guards: an overrun never fails cleanly (the cusolver_qr workspace bypasses the pool budget and
  can freeze Windows / crash the driver), so sizes are refused before solving: `size_issue` is the one rule for
  validation and solver (scipy_splu warns > 1e6, refuses > 2e6 nx * ny; cusolver_qr `qr_workspace_bytes` = 64 B nx ny
  band, band from `qr_band_order`, which `factorize` also applies; warns > 512 MiB, refuses > 2 GiB);
  `check_solver_size` (finalize, every factorisation) also refuses a QR workspace over half the free device memory
  (after freeing the cupy pool above 512 MiB). A solve cannot be cancelled (Progress before each solve, cancel acts
  between). SuperLU: MMD_AT_PLUS_A, SymmetricMode, diag_pivot_thresh 0.1 (1 swaps rows off the ordering);
  `factorize_host(ordering=)` runs NATURAL. The study logs `solve_summary()` (not on the ABC) after the first solve.
  The HarmonicSystem keeps A on the device on both back ends (TFSF / mode rhs need it).
- cuDSS 'linear' (cudss / cudss_mixed, `FDFD_DIRECT_BACK_ENDS`, GPU only): tensor meshes factorise D A
  (`_symmetrizer` = `stretched_cell_areas`, None with a Bloch phase; a failed symmetry check, e.g. eps_xy, gives a
  general LU). Not recyclable: a system change keeps the factor as `_cudss_previous` (for its plan; its device memory
  stays during the next assembly) and refactorises it in place (`previous=`). An adjoint solve on a general LU
  factorises the transpose (`_transposed_solve`: one extra factor outside the pool, not estimated). Triangles:
  `factorize_host(symmetric=True, row_scale=system.equilibration, previous=)` (SuperLU ignores the options). Selftest
  em_fdfd_cudss. B1a: Automatic is cuDSS on the gpu with nvmath-python when the role lists `auto_back_ends`
  (validation / estimators follow the resolved back end); an 'auto' runtime that cuDSS refuses falls back to SuperLU
  (solvers sub file B1b) and fdfd_2d / fdfd_2d_fem / fdfd_bor / fdfd_conical free `_cudss_previous`. A deferred
  cuDSS check (a refreshed factor judged by its first solve) raises CudssAccuracyError from the solve: fdfd_2d
  `_solve_linear` (driven and transposed) and host_fdfd `_checked_solve_host` (BOR / conical driven solves, BOR order
  sums) catch it, `host_fdfd.fell_back_to_superlu` (this miss switched the 'auto' runtime, or an earlier one did: a
  shared solver) and refactorise on SuperLU. Those paths refresh with `defer_check=True` (fdfd_2d driven + transposed
  factorize, host_fdfd `_factorize_operator`); BOR's C^T W C - k0^2 M also passes `symmetric_by_construction` ->
  canonicalised in place + `symmetric_known` (no asymmetry check), conical (D A) not. fdfd_2d_fem and every other
  `factorize_host(previous=)` caller has no catch: its refreshes stay probed (defer_check False). Self-tests
  comparing with "SuperLU" (em_fdfd_cudss, selftests_fdfd line_block, em_bor_cudss_driven, em_fdfd_conical_cudss)
  must give the reference an explicit scipy_splu node, never Automatic.
- MUMPS / PARDISO (solvers.host_direct; 'linear' back_end and 'eigen' factorization mumps / pardiso (+ _mixed), opt-in,
  either backend, optional python-mumps / pydiso; every EM type, solvers sub file "host_direct"): code tests the
  tuples / predicates, never the names (`HOST_DIRECT_BACK_ENDS`, `sparse_direct.SYMMETRIC_OPTION_BACK_ENDS` /
  `MIXED_BACK_ENDS`, `arpack_shift_invert.LDLT_FACTORIZATIONS` / `MIXED_FACTORIZATIONS`, `is_host_direct_factor`,
  `factor_label`; help texts take `definition_common.HOST_DIRECT_NAMES`). They get the cuDSS factor options (fdfd_2d:
  symmetric + `_symmetrizer` D, so the LDL^T of D A, LU when the asymmetry check fails; host_fdfd
  `_linear_symmetric_options`: BOR / conical row scale / symmetrizer; fdfd_2d_fem: symmetric + equilibration), but
  are recyclable and transpose natively like SuperLU: no `_transposed_solve`, GMRES recycling at an unchanged omega,
  no `_cudss_previous` on tensor meshes (fdfd_2d_fem keeps one as previous: analysis reused on the same pattern). Size
  rules: SuperLU's everywhere (`size_issue` / `fem_*_size_issue` / `size_issue_for_unknowns` take the host names;
  `fem_modes.host_solver_label` names the library), plus the library's own memory check after the analysis.
  Eigen: `_eigen_symmetrizer` returns D for every LDLT_FACTORIZATIONS entry; `solve_resonances` polishes every own
  mixed LDL^T (factor_precision 'mixed' and symmetric: cuDSS on the device, host `solve_host(refinement_steps=
  MIXED_EIGS_REFINEMENTS)`), fem_modes likewise; every polished solve is refined (eigs on unrefined solves was tried
  and removed: the Rayleigh move squared misses the 1 / gap of a near eigenvector, eigenfrequency header); and a double own LDL^T of a complex A when the caller passed a
  tolerance_floor > 0 (open models): ARPACK to MIXED_ARPACK_TOLERANCE + the quotients, no refinement steps (`refine`
  is mixed only); a closed lossless model keeps its floor (Q = inf). E2: 2D FDFD passes `rayleigh_symmetrizer` =
  `_adjoint_symmetrizer()` (D measured, None with a Bloch phase / eps_xy), so a complex A (PML, loss) is polished
  the same way on any factor (SuperLU, the line-block factor, a double LDL^T); a real A keeps machine precision. An
  own host factor logs `_host_direct_factor_text`; elimination
  orders (em3d) are skipped. Texts: solve_summary / direct_text / `HOST_DIRECT_KINDS` (em_modes factor_kind prefixes)
  name the description and the library's factor bytes. Estimates: `memory_estimates.host_factor_items` (the SuperLU
  heuristic x `superlu_bytes_scale`, LDL^T where D A / the pencil is symmetric, + `solver_host_bytes`; an eigen mixed
  request below MIXED_EIGEN_MIN_UNKNOWNS booked double) at every place a SuperLU factor item was booked (2D FDFD,
  triangles, BOR, conical, em_modes, em3d, eme); the hz eps_xy extra fill stays at SuperLU size; time recipes are the
  role's (sparse_direct / arpack host items). Self-test em_fdfd_host_direct (selftests_fdfd, cpu, heavy, skipped
  without both libraries).
- line_block 'linear' node (opt-in, Automatic stays sparse_direct): one host operator copy (`set_operator(owner=)`,
  block axis = the periodic one; two periodic axes: ValueError / validation error) and one kept factor, no GMRES
  recycling. An adjoint solve runs on the kept factor as x = D solve(g / D) when D A is complex symmetric
  (`_adjoint_symmetrizer`: D = `_symmetrizer`, asymmetry measured once per system against cudss_factor
  SYMMETRY_TOLERANCE; None with a Bloch phase or eps_xy). A rejected solve marks the system `_REJECTED`
  (line_block_fallbacks) and, like a non-symmetric adjoint solve, goes to `_reference_solver` (SparseDirectSolver
  'auto'). Wavelength batches: the frequency_domain study calls
  `plan_harmonic_frequencies(omegas)` before its loop and `(())` after (`_solve_wavelengths`; not the coupled scan /
  optimization); up to `MAX_BATCH_WAVELENGTHS` omegas factorised together (`factorize_operators`, capped by
  MEMORY_FRACTION). Each wavelength's own `_assemble` must give the identical operator, guarded by `_material_serial`
  (bumped by every `_mark_system_dirty` but `omega_only`), so any change of the assembled system must go through
  `_mark_system_dirty`. Eigenfrequency wraps `_system_line_factor` as `LineBlockFactorization`; a
  RejectedLineBlockSolve redoes it on SuperLU. Self-tests em_fdfd_line_block, _line_block_wavelengths,
  em_fdfd_eigenfrequency_line_block.
- Bloch k != 0 (symmetry 2.2): only the wrap entries change. `lower_wrap_factors` = e^{+j k L} per axis: every
  face-to-centre mean across the wrap takes it (`face_mean_at_centres(wrap_factor=)`; q_abs needs none; NF2FF boxes
  never wrap), as do a dipole on the last face and a mode source's upstream cell. Sheets carry e^{-j k.r}
  (`bloch_line_phase`); TFSF with Bloch ky = the oblique discrete wave. Port lines spanning a whole Bloch period are
  refused (`_check_bloch_line`, `bloch_period_line_issues`). nlfd takes each channel's factor in its C / C*.
- Polar (r, phi) grids (symmetry 2.16 / 2.18; scripts only, `coordinate_grids` lists 'polar'; PhysicsTypes stay
  Cartesian until the model path exists): the metric lives only in the A1 ops (`polar_radius_min=`, complex r~ in
  the x_hi CPML); `FDFDPhysics.metric` gives point-source areas, arc flux weights and eigen PML shares. Faces
  (`_check_polar_boundaries`): x_lo pec / pmc or the origin ('pmc' placeholder, wall row dropped); x_hi cpml / pec /
  pmc (PML on the last r width, no mesh_config); phi faces pec / pmc / mirrors, periodic / bloch only when angle_wrap
  joins them (bloch_ky = azimuthal order m [1/rad]). `_require_cartesian` refuses line sources / beams, TFSF, dipoles,
  mode ports, NF2FF, objectives, design regions, sheets, conformal PEC, subpixel smoothing, eps_xy. Closed forms are
  semi-discrete: order nu = (2 / dphi) sin(m dphi / 2). Self-tests selftests_polar.
- Faces: cpml, periodic, pec / pmc walls and mirrors with the FDTD rows (`_WALL_SIGNS`: outermost cell = sign x inner
  neighbour; ez pec / hz pmc 0, ez pmc / symmetric and hz pec / antisymmetric +1, ez antisymmetric / hz symmetric
  -1): outer cells pinned via `pin_image_cells` (rhs zeroed, filled after the solve, PEC re-zeroed); a +1 row also
  zeroes the inner face weight, so an hz face current there drops (as the FDTD re-pin). `_PINNED_CELL_BCS` /
  `_PINNED_FACE_BCS` must match the FDTD's.
- `face_fields()` (cached per solve): hi faces; hz = discrete curl PLUS source `face_currents` (dipole, sheet), so E is
  exact on a source face. TFSF and mode phasors only add to the rhs, so the field jumps across their interface: no
  measuring face (monitor line cell hi face, probe hx / hy, NF2FF samples) may sit on a mode source's interface face
  (hi face of line cell - 1 for 'positive', of the line cell for 'negative') or on / across a TFSF outline; a far-field
  box around a TFSF box needs >= 2 cells on the lo sides and its x1 / y1 edges past it (FDTD: 1 cell). Validation
  warns (`_total_field_regions`, `_mixed_region_texts`); hz q_abs beside such a face is wrong in lossy cells, unchecked.
- TFSF (`PlaneWavePhasor`): b += (1 - Q) A (Q u_inc) - Q A ((1 - Q) u_inc) through `system.operator`; exact whenever
  u_inc solves the background rows (medium of outline cell (x0 - 1, y0), read per solve), so it is exact only with
  that one medium around the box. Bloch ky needs uniform y cells (else ValueError / validation error); uniform columns
  use the discrete-dispersion wave (raises at <= pi cells per medium wavelength), graded ones a march on the operator's
  x line (kept off uniform lines: round-off growth in lossy media). Phase referred to the centre of column x0 (FDTD
  TFSF phases differ); hz amplitude = incident E_y. `full_period_y`: no y edges. Selftest em_fdfd_tfsf_graded_leakage.
- Mode ports (`fdfd_ports`): unlike the drift-gated FDTD ports, modes are re-solved at finalize and after every
  reassembly (`PortLine.solve_modes` returns the last SlabModes when omega, counts, spacing, host rows and pole objects
  are unchanged). Source = the TFSF formula on the line cells and their upstream neighbours (periodic normal axis: half
  period >= 4 cells or ValueError); `add_to_rhs` refuses a mode solved at another omega. The source re-selects
  `mode_number` per solve; the monitor tracks by overlap + `align_phase`. power_* = |a|^2 W/m (FDTD 2 |a|^2). Only
  finalize's port warnings are reported.
- `create` (FDFD and em_modes) builds at `first_solved_wavelength`, so ports select modes there; auto mesh at
  `shortest_solved_wavelength` with the static index; the resolution warning checks `all_solved_wavelengths` with |n|
  of the complex eps(omega) plus >= 2 cells per skin depth, so metals the static index (Drude eps_inf) misses are
  flagged.
- Monitors record one value per omega (repeat overwrites) on the device; results ascending wavelengths; a monitor
  added after a solve records it at once. HarmonicProbe: ez hx / hy raw on the +y / +x face, hz ex / ey collocated.
  Flux = 1/2 Re(E x H*) [W/m], discretely conserved; modes normalised to that discrete flux.
- q_abs: ez 1/2 omega eps0 (-Im eps) |E_z|^2; hz per component the face-mean |E|^2 (`face_mean_at_centres`) x -Im
  eps_ii; magnetic loss not counted. Renderer primary `<field>_re` (the stationary fixed point compares it).
- Eigenfrequency (A6, `eigenfrequency.py`, host only): `solve_resonances` is dimension-free ('shift_invert' 2D,
  'squared_cayley' for curl-curl with a gradient null space; sparse mass / energy weights via `_PencilMass`, a pinned
  row absent, not zero; em_eigen_sparse_mass; a host B v from HOST_MASS_THREAD_MIN_ENTRIES runs by row blocks on a
  process-wide thread pool, `_mass_pool`, relying on scipy's CSR kernel releasing the GIL). 2D shift-invert factor: host SuperLU (`_eigen_size_warnings`), a
  line_block node's kept factor, or cudss / cudss_mixed / the host direct ones (`_eigen_symmetrizer`, LDL^T of D A).
  Own factor at sigma =
  k_t^2 (1 + SHIFT_RELATIVE_OFFSET): a target on a lossless resonance gave a singular factor cuDSS refused. Drops Re
  omega <= 1e-6 omega_t (hz Neumann constant) and PML resonances (> 0.5 energy outside `physical_window`); too few
  left: ARPACK with 2x the eigenpairs, `MAX_WIDENINGS` times (estimates size the first pass only; cost ~
  eigenpairs^2, so ask only for resonances that exist near the target). Own cuDSS factor, or a host factor on a gpu
  runtime where `device_arnoldi_fits(host_matvec=True)`: device Arnoldi (`eigs_device`, the first pass reserving
  WIDENING_BASIS_RESERVE x its basis rows), widenings resume the Krylov-Schur decomposition in place (`basis_kept` in
  the info). Any other host factor: the host twin (`eigs_host`, `_HostKrylovSchur`, numpy rows, same reserve and resume;
  host ARPACK only below DEVICE_ARNOLDI_MIN_RATIO), so cpu runs follow the gpu algorithm. The candidate filter (`_ResonanceFilter`, built once per solve: on the device on a gpu runtime or own
  cuDSS factor, else host) takes per column v^H B v, v^H A v, the D-scaled Rayleigh terms and the energy / window /
  region sums from ONE sparse product per operator and block of FILTER_BLOCK_COLUMNS vectors; never per-column host
  csr_matvec loops (tetrahedral Mie: 2.4 s of 8 s). `squared_cayley_roots` takes those products, not the vectors.
  Estimators book it via `memory_estimates.resonance_search_items` and `eigenfrequency.resonance_filter_time_item`
  (2D FDFD, BOR, conical, em_fdfd_3d). Dispersion and PML stretch stay frozen at omega_t
  (first-order error; help texts say so). Fields are 1 at the first argmax |u|: mirror-equal peaks leave the sign to
  round-off. `select_resonance(k)` sets `selected_resonance` (hz `face_fields` skips source currents); `_solution_dirty`
  stays True so a later step() re-solves the driven problem. Region (P2): `set_resonance_region`, column
  `region_energy_fraction` last; on tensor grids the region is `ctx.domain_mask`, so an object touching the window
  edge counts the PML cells beyond it (clamped centres; example dbr_cavity_q stops its gaps one cell short). Self-tests em_fdfd_eigenfrequency_pec_cavity, em_resonance_region_fraction,
  em_fdfd_resonance_region; template disk_resonator_eigenmodes.
- Adjoint (C7, `adjoint.py`, derivation in its header): objectives subclass the flux / mode monitors; J = sum weight
  * value (flux unfolded by `mirror_factor`, mode power raw). `eps_sensitivity()` is lazy, cached per (n_solves,
  objectives), zeros while a resonance is shown. `solve_adjoint` uses `self.system`: never call it after
  `solve_eigenfrequencies` reassembled. The adjoint source is folded through the transpose of the wall fill; hz
  differentiates V incl. the `pin_image_cells` fold, plus (as nlfd) the objectives' face fields (`face_cotangents` =
  the objective face parts) and the rhs D_e V J of the sources' summed `face_currents`. Held fixed: source
  calibrations (the J values), TFSF / port incident fields, port modes; sigma_e, poles, mu_r not differentiated. `mode_amplitude_weights` is the one map behind
  mode_amplitudes and the mode objective. eps_xy: `EpsSensitivity.off_diagonal_density` ('eps_xy_sensitivity', hz
  only, parity 'even' although its true image is odd). physics/nlfd/adjoint.py chains `objective_cotangents` ->
  `adjoint_source` -> `solve_adjoint` -> `permittivity_gradients` -> `sensitivity_density`, so a signature change must
  reach it. Self-tests em_fdfd_adjoint_gradient, em_fdfd_hz_eps_xy_adjoint_gradient.
- Design regions (`design_density.py`): `DesignMap` (host only) = conic filter -> tanh projection -> eps;
  `density_gradient` is its exact transpose (em_fdfd_design_chain_rule). `DesignRegion` writes eps into every
  permittivity array of its physics (hz: xx and yy, so dJ/d eps = the sum) via `MaterialGrid.assign_window`; the caller
  then calls `materials_changed`. Regions of one physics may not share cells; the study refuses regions of different
  physics writing one array on a common cell, and chains the gradient of every physics whose `cell_gradients` names a
  region's array (same-polarisation physics share eps_r). Study: model/study_optimization.py (model sub file);
  `set_projection_beta` takes effect at the next `set_density`. Template waveguide_bend_optimization.
- Conducting sheets (`sheet_conductivity.py`): enter the OPERATOR only (ez diagonal on the line cells, hz hi-face
  weights, after face averaging, before PEC / even-wall zeroing); `cell_eps` / `eps_hi_*_faces` stay bulk (source /
  TFSF calibrations, eigen mass). Port modes and MaterialGrid maps never see a sheet (`_mark_system_dirty(ports=
  False)`); sigma(omega) cached (graphene Kubo: scipy quad). Self-tests em_graphene_kubo_limits, em_fdfd_conducting_sheet.
- `set_extra_permittivity(values, box)`: physics-private complex eps before face averaging, never in the MaterialGrid;
  sources, q_abs, adjoint see it, port modes do not. `_solve(record_monitors=False)` + `record_monitors()` later.
  Such a solve defers q_abs and the float32 state arrays (`_refresh_pending_state`): `state()`, `take_absorbed_power`,
  `record_monitors` and a reassembly (it needs the old system's face fields) write them first; the wall rows,
  solution and solved_omega are current at once. New readers of `_q_abs` / `_field_re` / `_field_im` refresh first.
- Phasor sources with `set_amplitude(*values of amplitude_params)` re-set in place (a reused sweep build: nlfd
  `_FOLLOWING_SOURCE_PARAMS`): point / magnetic / dipole currents, sheet currents, TFSF plane wave, mode source
  (rescales its solved incident phasors; power <= 0 raises at the run, not at validation).
- NLFD-chi (physics/nlfd): one FDFDPhysics per harmonic channel; nonlinear terms are sources whose `add_to_rhs` /
  `face_currents` read other channels' solutions (nothing assumes the rhs constant beyond `_solution_dirty`). Its
  channel features wrap the FDFD FeatureTypes: renaming an FDFD feature param, or a validate resolving FDFD params
  from the raw feature params, reaches it too.
- `yee_grid.cell_row` is the shared outer-row view, so adjoint.py needs no fdfd_2d import. `solve_summary` keys its
  factor text on `linear.back_end`, never on host_factor_bytes being None.

## em_fdfd_ez / _hz on triangles (fem_fdfd, solvers/fdfd_2d_fem)
- The four triangle EM physics (FEMFDFDPhysics, FEMModePhysics, FEMBORPhysics, FEMConicalPhysics) subclass
  `solvers/fem_em_base.TriangleEMPhysics` (checks, per-element `element_fields`, state / field_dofs / fem_field
  skeleton and nodal cache, slots, finalize size rule, renderer, take_absorbed_power, PEC-keyed layout cache,
  `_element_materials`, poles / PEC masks): a new one sets the class attributes and the hooks `_build_layout`,
  `_has_shown_field`, `_shown_nodal_fields`, `_last_unknowns` (`_system_changed` when it caches an assembly), never
  copies. FDFDPhysics and FEMFDFDPhysics share `solvers/harmonic_shell.HarmonicEMShell` (omega, state, slots incl. hz
  eps_xy_update, materials_changed, loss protocol, step / solve_harmonic, eps_xy). Point sources / probes are located
  once per layout (`fem_fdfd._basis_at`, `FEMFDFDLayout.point_bases`).
- One type, two solvers: create builds `FEMFDFDPhysics` when `ctx.domain.grid_kind` is 'triangles' (walls from
  `fdfd_domain_requirements(settings)`: no BoundaryConfig), else FDFDPhysics; setting `discretization` (Lagrange 2 /
  1; tensor meshes ignore it; nlfd omits it). point_source / field_probe / pec_domain / resonance_table `grids` hold
  'triangles'; point_source / field_probe validate per mesh kind (`_per_mesh_kind`); `_triangle_issues` refuses the
  optimization / optomechanics / thermal_noise studies and periodic faces, notes a cusolver_qr node (host SuperLU
  used) and applies `fem_fdfd_size_issue`; `direct_solver_issues` skips its grid size rule on non-tensor estimates.
- Formulation, signs, walls, PML: `fem_fdfd` header. ez essential on pec / antisymmetric / cpml faces and the PEC
  closure, hz essential only on pmc / symmetric (PEC outlines natural: no staircase). hz stiffness eps_t / det(eps_t)
  with eps_xy keeps the operator complex symmetric. The operator is -(Yee operator), so loads keep the Yee source
  amplitudes.
- `_solve`: `factorize_host(symmetric=True, row_scale=system.equilibration, previous=)` (host SuperLU, or cuDSS LDL^T
  refactorising `_cudss_previous`, which also keeps a MUMPS / PARDISO factor for its analysis); no GMRES recycling (a
  Picard step refactorises). Eigenfrequency
  `solve_fem_fdfd_resonances` (A = K - k_t^2 M, mass -M, symmetric, equilibration). State = centroid values, q_abs =
  element means. Renderer: FEMFieldRenderer with `_TRIANGLE_FIELDS`; `nodal_fields` averages over non-PEC elements.
  Subpixel smoothing / conformal boundaries accepted and ignored. `error_indicator` = Kelly jumps, as em_modes.
- Sizes (uncalibrated): FEM_FDFD_ENTRIES_PER_ROW, FEM_FDFD_LU_FILL, FEM_FDFD_ASSEMBLY_BYTES_PER_ENTRY / _CALLS;
  `memory_estimates._fdfd_fem_memory_estimate`. Self-tests selftests_fdfd_fem (em_fdfd_fem_pec_box, _point_sources,
  _disk_model, _driven_model, _cudss).

## Cross-section modes (cross_section_modes, A5 numerics)
- Host only (numpy / scipy; never cupy arrays; a line-block factor keeps its own device state). xy = cross-section,
  fields ~ exp(j omega t - j beta z), Im n_eff < 0 lossy. `solve_cross_section_modes` = `prepare_cross_section_modes`
  (CrossSectionProblem, operator A - sigma B) + `solve_prepared_modes(problem, ..., factorization=None)` (SuperLU or
  any solve_host factor). `line_block_layout`: lines = rows j (block_axis 0) or columns i (1), the periodic axis inside
  the blocks; two periodic axes refused (A couples h_x(i, j) to h_y(i - 1 .. i, j .. j + 1)). Yee cell projected,
  material at E_z (cell centre): H_x / E_y hi-y face, H_y / E_x hi-x face, H_z corner; eps_zz and mu cell values,
  eps_xx / eps_yy arithmetic `face_average` (normal component first order at interfaces; subpixel there a follow-up).
  Unknowns h = eta0 [H_x; H_y], A h = beta^2 B h. PEC = eps^-1 0 on cells and on faces between two PEC cells (B
  singular: infinite-beta rows, never selected). Never `face_pec` here: E_x / E_y sit on faces normal to them, and its
  either-cell rule zeroes the normal E at a conductor (a parallel plate of N gap cells gives n_eff^2 ~ eps (N+1)/(N-1)).
  TE slab = slab_modes 'ez' to round-off; TM slab vs 'hz' only to discretisation order (dual point set).
- Walls, all types on one plane: the inner face of the outermost grid cell (a mirror's padded ghost outside the window,
  a wall's outermost window cell); ghost material = inner neighbour's (`_ghost_filled`); centre quantities = s x
  neighbour (s -1 pec / antisymmetric, +1 pmc / symmetric) via a column map folded into Dx^h / Dy^h; plane-face
  unknowns free (s -1) or dropped (s +1); faces beyond dropped. Half a cell off the FDTD / FDFD ez pec and hz pmc walls
  (a cavity differs by one cell). cpml / natural ends drop the last face. Power, te_fraction and areas weight only
  points inside the walls (plane faces half) and count a mirrored mode whole (`mirror_factor` 2 per mirror face).
- Conformal PEC (`CrossSectionMaterials.pec_cuts` = `core.cut_cells.dual_cell_cuts`; module header): Faraday split into
  `dx_h_centres` / `dy_h_centres` (on E_z) and `dx_h` / `dy_h` (A^-1 Dx^h diag(t_y) ..., A the dual free area) so Qx
  Pz_y = Qy Pz_x and the div B = 0 elimination / h_z recovery (`_System.dx_h`) stay exact; eps^-1 masks, B, D^e and
  power areas stay staircase. `_wall_neutral_cuts`: links / dual cells crossing a wall plane or the periodic seam are 1,
  ghost links copy the inner neighbour (column maps commute with diag(t)). Fractions >= DUAL_CUT_FLOOR (conditioning,
  no dt). Self-test conformal_pec_modes (model/selftests_conformal_cuts).
- `mode_point_materials` / `ModePhysics.point_materials()`: ghost-filled, face-averaged, bend-mapped eps of the E points
  (`_point_materials`, the eps half of `_assemble`: one writer); the em_eme_3d adjoint transposes it (physics/eme sub
  file). Bend factors are per component: a consumer weighting E_z uses the z factors.
- Mirror exactness needs mu uniform across the plane: H_y / H_z take mu of their lower cell (not mirror symmetric).
- Solve: sigma = (k0 n_guess)^2 (1 + 1e-6) (keeps a TEM mode off the shift), n_guess default max Re sqrt(eps mu) of
  the straight structure; ARPACK on (A - sigma B)^-1 B (`eigen_solver=`: em_modes passes its 'eigen' role runtime), k =
  `mode_eigen_count`: min(2 n + 10, unknowns - 2), but n + `ABOVE_SPECTRUM_EXTRA_EIGENPAIRS` (4) when
  `CrossSectionProblem.default_shift_above_spectrum` (operator eps^-1 / mu real and positive, no bend, default guess)
  and PML modes are dropped; n + `KEPT_PML_EXTRA_EIGENPAIRS` (6) when they are kept (EME, S3). Without PML every
  beta^2 is real below sigma, so the k nearest are the k highest-ranked for any k >= n (exact); on CPML the nearest
  non-PML modes are those of highest Re beta^2 up to strongly leaky ones, and a solve whose dropped PML modes leave
  fewer than n candidates is redone at 2 n + 10 (same factor, applications summed); a solve keeping them is redone
  there when the n-th ranked mode lies near the converged disk edge (`_kept_pml_near_cut`: sigma - Re beta^2 >
  KEPT_PML_CUT_FRACTION 0.5 x the largest |beta^2 - sigma| returned; a PML mode beyond the disk outranks it only with
  |Im beta^2| > 0.87 x the radius). 2 n + 10 made ARPACK converge the PML / cladding cluster it then dropped
  (em_modes_cpu_gpu: 1782 -> 411 applications). A new way to put an eigenvalue above the default shift (gain, negative
  eps, a mapped metric) must clear that flag. Estimators take the wide `mode_eigen_count(n, unknowns)` (upper figure;
  they cannot see the materials). An own factor of an operator without imaginary part (walls, lossless, no PML / Bloch
  phase) is float64 (`factorize_host(real=True)`) and host ARPACK then runs real (E1: `eigs(real=True)`, dnaupd, real
  v0: start modes turned real by `_real_aligned`; the device Krylov-Schur stays complex, two real solves per vector).
  Estimators test `memory_estimates.real_cross_section_operator(settings, context)` (no cpml, no Bloch phase, no
  sigma_e / sigma_m / poles): the em_modes memory books the float64 SuperLU factor and host basis, the time recipe
  value_bytes 8 (unless cuDSS, line_block or the device Krylov-Schur of a gpu run). Ranked by descending Re
  beta^2 (ties by Im), as slab_modes (Re n_eff would let evanescent
  modes outrank weakly guided ones); PML modes (> half |h|^2 dA on stretched points) dropped unless
  `drop_pml_modes=False` (physics/eme), so fewer than asked may return. `size_issue_for` =
  `sparse_direct.size_issue(..., unknowns_per_cell=2)`. Non-convergence keeps converged pairs and warns in
  `info['warnings']`. Tolerance machine precision, `PML_ARPACK_TOLERANCE_FLOOR` (1e-12) when dropping PML modes on CPML,
  then also a `PML_ARNOLDI_BASIS_PER_WANTED` (5) x n_wanted basis (`mode_default_basis`; the em_modes memory and time
  estimates take it via `cpml_cross_section(settings)`; scipy's 2 k + 1 hit the 300-restart cap in em_modes_cpu_gpu,
  2479 applications). `solve_prepared_modes(tolerance_floor=)` raises it: ModePhysics' own solves pass
  `MODE_ARPACK_TOLERANCE_FLOOR` (1e-10, E3), its group-index solves (the +- 1e-3 difference amplifies the error 500x),
  detached solves, ports, EME and the CPU cases keep machine precision. EME (drop_pml_modes False) keeps scipy's basis.
  `warm_guard=True` adds `arpack_shift_invert.guarded_start_vector`'s random share to a start from earlier modes (E5:
  ModePhysics starts each solve from the last solve's modes, the em3d port source re-solve from its last modes).
- Degenerate pairs (fibre HE11) come back as arbitrary non-orthogonal combinations; separate them with mirrors.
  `mode_overlap` = |c_ab c_ba| / |c_aa c_bb|; `mode_group_indices` re-solves at omega (1 +- 1e-3) with the modes as
  ARPACK v0, tracks by overlap (NaN + warning below 0.9): pass the solve's n_eff_guess.
- Bend `(R, x_ref, full-grid x edges, model)`, s = 1 + (x - x_ref)/R at each Yee point's own x (`_bend_factors`, one
  writer for operator and `ModePointMaterials`); ValueError where s <= 0. 'exact' (default): eps and mu x diag(s, s,
  1/s) (diagonal mu in the operator); fields are the mapped components (E_z, H_z = s x physical E_phi, H_phi); power /
  `mode_overlap` keep the straight sums, only volume integrals carry the metric (via the mapped eps). 'conformal': eps x
  s^2 (second order in w / R). Inside x CPML s is taken at the complex stretched x~ = x + integral (s_pml - 1) dx
  (`_stretched_x_edges`, each layer from its inner edge) to keep the layer matched; leaky bend modes want pml_kappa_max
  1 (default 15 reflects grazing radiation). `mode_point_materials` keeps the real factors. Self-tests (CPU):
  em_cross_section_bend (selftests_bend: bent slab vs slab_modes lines and the exact transform u = x_ref + R ln(r / R));
  em_shg_overlap_bend (that transform as a straight cross-section; mu_yy = s^2 given at the hi-u face cell since H_y
  takes its lower cell's mu); em_modes_bend_loss (model run, leaky E_y mode vs a 1D finite-volume solve in the check).
- Self-tests `CROSS_SECTION_CPU_SELFTESTS` (selftests_cross_section.py): slabs vs slab_modes, walled guides vs exact
  discrete spectrum, mirror halves vs full domain, quarter fibre vs Bessel roots at the area-matched staircase radius.

## Conical / out-of-plane k_z (em_fdfd_conical; conical, solvers/fdfd_conical, renderer_conical, definition_conical; 2.9)
- Host only; module header = the formulation. E unknowns (E_x, E_y, E_z) on the cross_section_modes layout (same
  walls, mirrors, Bloch phases, CPML, point eps / lower-cell mu), fields ~ exp(-j k_z z): a change to that layout /
  point rules changes both. `assemble_conical` (complex k_z allowed); `solve_conical_driven` (J_F, FDTD sign; at k_z = 0
  the E_z block is the em_fdfd_ez operator but with the cross-section wall planes, half a cell off the FDFD ez pec / hz
  pmc walls); `solve_conical_resonances` (C_H mu^-1 C_E = k0^2 eps at fixed k_z via
  `eigenfrequency.solve_resonances(spectral_map='squared_cayley')` as BOR; static and > 0.5 PML-share resonances
  dropped, ranked by |omega - omega_guess|, cpml stretched at omega_guess only). ARPACK floor 1e-10
  (`_ARPACK_TOLERANCE_FLOOR`), shift offset eigenfrequency.SHIFT_RELATIVE_OFFSET (1e-4); cross_section_modes keeps 1e-6
  and machine precision (its tests need round-off). Why: with the target on a resonance (k_z = beta of a mode) 1e-6
  amplified rounding past the tolerance and one uninterruptible ARPACK call ran for minutes. Squared Cayley replaced
  (A - sigma B)^-1 A / sigma, which ranked resonances with k0^2 < ~sigma / 2 behind every one above. Refused: pec_cuts,
  bends.
- cuDSS / symmetrizer (module header "Symmetry"): the Yee conical operator is NOT complex symmetric as assembled, but D
  A is for D = (A_x, A_y, -A_z) (layout face / centre lengths x CPML stretch, E_z block negated; `_symmetrizer`). Kept
  by PML, loss, complex k_z, walls / mirrors, PEC, subpixel eps; a Bloch phase breaks it (`ConicalSystem.symmetrizer`
  None: cudss an LU, cudss_mixed a complex64 LU refined to 1e-12 per solve). The eigen solve passes symmetrizer=D and
  equilibration (|D| (|diag curl-curl| + k0^2 |eps|))^-1/2 (diag A nearly cancels): cudss LDL^T of D A, cudss_mixed
  (`eigen_role(mixed_factorization=True)`) loose solves + float64 Rayleigh polish. Eigen 'auto' = cudss on the gpu with
  nvmath (host SuperLU fill grows ~n^3 here). Driven 'linear': `CUDSS_LINEAR_ROLE` (definition_helpers);
  `ConicalPhysics._solve_linear` (HostFDFDBase._factorize_operator) factors D A (`factorize_host(symmetric=True,
  symmetrizer=D, row_scale=conical_row_scale, previous=)`), no
  GMRES recycling: a system change moves the factor to `_cudss_previous` for refactorisation on the kept plan; released
  before an eigen-solve. A cuDSS factor skips the SuperLU size rule (assemble_conical(check_size=False), finalize,
  `_solver_issues`); memory estimate counts the device operator copy and the factor as an external item. Selftest
  em_fdfd_conical_cudss (selftests_conical). Verdict: 3 unknowns per cell, SuperLU fill limits near ~1e6 unknowns, so
  cuDSS pays for large cross-sections and long k_z sweeps; single_residual gate 1e-3 (above it use cudss).
- Sources (`point_current_terms`): integrated cell current I; 'z' on the centre, 'x' / 'y' half on each of the cell's
  two faces by dual area (`placement.dipole_faces`; the wrapped lo face of cell 0 on a periodic axis carries
  e^{-j k L}). Line source = sum of its cells (exp(-j k s) along a Bloch axis, s from the grid lo corner; cpml edges
  continue through the PML). `source_power` = 1/2 Re E . J_F* dA (J_F is minus the textbook current).
- Flux (`line_flux`): line-cell E with H on their hi face; centre pairs by centre lengths, face pairs by face lengths
  with the two end faces half (periodic: both ends add to the wrap face; a wall / mirror end face keeps its layout
  share). Exact discrete Poynting only for full-period / wall-to-wall line pairs (source power minus absorption between
  them); shorter lines and closed boxes are second order. Mirror factor `placement.line_mirror_factor`.
- ConicalPhysics: FDFDPhysics attribute set, polarization 'conical' (sheet placement helpers refuse it), material
  schema 'modes' (eps_r = eps_zz on E_z), `grid_cell_widths` as em_modes; a material / omega / k_z change builds a new
  ConicalSystem; the driven SuperLU factor ('cusolver_qr' refused) is kept per system and recycled by GMRES after a
  material change at the same omega, k_z and active set. Subpixel smoothing as em_modes; no conformal PEC. Not yet: a
  conical TFSF plane wave (a phased line source across a Bloch period stands in), far field, mode ports, line-block.
- Self-tests (selftests_conical): `CONICAL_CPU_SELFTESTS` em_conical_empty_lattice, em_conical_current_sheets;
  `CONICAL_MODEL_SELFTESTS` em_fdfd_conical_sheet (flux identity vs source_power), em_fdfd_conical_lattice (exact Yee
  dispersion). Walled guide = example em/analytic_conical_walled_guide. selftests_conical_fem holds
  em_conical_symmetrizer (D A symmetric to 1e-12, Bloch -> None) and em_conical_yee_cudss (gpu, LDL^T and polish).

## em_fdfd_conical on triangles (fem_conical, solvers/fdfd_conical_fem; agent_help batch)
- One type, two solvers: `definition_conical._create` builds `FEMConicalPhysics` when `ctx.domain.grid_kind` is
  'triangles' (walls from `fdfd_domain_requirements(settings)`), else ConicalPhysics. Setting `discretization`
  (quadratic default, linear). Sources / probes / flux monitors stay tensor; `_triangle_issues` refuses every study but
  eigenfrequency and periodic / Bloch faces (no periodic mesh), and applies the FEM size rule (fem_modes
  `size_issue_for_unknowns`, `FEM_CONICAL_SIZE_REMEDY`; none for cudss). Sizes are em_modes' (same spaces);
  FEM_CONICAL_ASSEMBLY_BYTES_PER_ENTRY / _CALLS uncalibrated.
- Formulation (`fem_conical` module header): the planar fem_bor pencil (r -> 1, m -> k_z) on fem_modes' parts (e = E_t
  Nedelec, w = -j E_z Lagrange); complex symmetric (D = I) with loss and PML, exact discrete statics. It is exactly the
  em_modes Lee-Sun-Cendes pencil at fixed k_z (pinned by em_conical_fem_modes_equivalence): a change of fem_modes'
  `_coefficients` / walls changes both types.
- Reuse: walls / PEC / mirrors / PML ring = `fem_mode_layout` (via `fem_conical_layout`); fields = fem_modes' Faraday at
  beta = k_z through `FEMConicalField` (duck-types the FEMMode attributes), H at the target omega. Solve:
  `solve_fem_conical_resonances` = solve_resonances (squared Cayley, symmetric=True, equilibration (|diag K| + k_t^2
  |diag M|)^-1/2): cudss LDL^T / cudss_mixed + polish as fem_bor.
- FEMConicalPhysics: ConicalPhysics' eigen attribute set (conformal_locator None so apply_pec calls add_pec_region,
  set_subpixel_smoothing ignored), no driven pieces; state / materials per element, `field_dofs` P0; `error_indicator`
  = fem_modes.mode_error_indicator of the shown resonance. Layout rebuilt only when the PEC elements change.
- Missing on triangles (refused): driven solves, line block, subpixel / conformal (moot), complex k_z. Driven deferred:
  conical grating incidence needs periodic faces; once periodic meshes (as em3d's builtin Kuhn tets) or em_fdfd_ez /
  _hz triangle sources exist, share their FEM source / probe / flux pieces instead of a conical copy.
- Self-tests (selftests_conical_fem; tolerances first estimates): em_conical_fem_pec_box (TE^z / TM^z closed forms, no
  spurious), em_conical_fem_modes_equivalence (1e-8), em_conical_fem_cudss (gpu, 1e-8: unpolished
  double cudss on the device Krylov-Schur ~5e-9), em_fdfd_conical_fem_pipe (model run).

## em_modes model layer (solvers/modes_2d, definition_modes)
- `ModePhysics` has the FDFDPhysics attribute set (polarization 'modes' = the `EM_MATERIAL_SCHEMAS` key with eps_r,
  eps_xx, eps_yy; `placement` helpers refuse it: no sources). Every `solve_modes()` / `solve_harmonic(omega)` re-solves
  from the MaterialGrid, so `materials_changed` only bumps `_material_serial`, which stales a kept line-block batch (so
  do inject_source, add_pec_region, add_dispersive, set_subpixel_smoothing: a new way to change the solved system must
  bump it too). After each solve it selects `mode_index` (= setting mode_number, clamped, warning when fewer modes) and
  calls `record(physics)` on its monitors. `select_mode(k)` fills the float32 `mode_fields` in place (`[...] =`):
  `ModeRenderer` and `state()` read them, so keys stay live arrays. Solve and group-index warnings go to
  `runtime_warnings` (the size warning once, via finalize_warnings).
- Roles: 'eigen' (arpack_shift_invert) and 'shifted' (Automatic = the eigen role's host SuperLU). A line_block shifted
  node (`ModePhysics(shifted_solver=)`): every operator comes from a kept `factorize_operators` batch
  (`_batched_problem`, `_problem_sequence`: this solve's omegas incl. group-index ones via
  `mode_group_indices(solve_at=)`, plus later wavelengths `plan_harmonic_frequencies` planned at the current serial), up to
  fdfd_line_block.MAX_BATCH_WAVELENGTHS, kept until solved, stale or re-bound. A RejectedLineBlockSolve redoes that
  problem on SuperLU (warning, counter line_block_fallbacks). `solve_detached` (shg_overlap, bend_overlap) uses the
  eigen solver's own factor. Two periodic axes: ValueError at finalize, validation error. Estimator:
  `cross_section_modes.line_block_bytes`, no SuperLU item. Selftest em_modes_line_block.
- Eigen factorization (`eigen_role(mixed_factorization=True)`): superlu, cudss and the host direct doubles (both
  meshes), the MIXED_FACTORIZATIONS (triangles
  only: `_solver_issues` refuses it on tensor meshes because A_xx mixes eps^-1 on both sides (ezz^-1 and eyy^-1 terms):
  no diagonal D makes D A symmetric, so no LDL^T / polish, and 2D factors are cheap anyway). A symmetric Yee form (the
  mixed [E_t; E_z] pencil): not done. With a cudss node (`ModePhysics._on_device`) every solve prepares with
  `check_size=False`, finalize / validation skip `size_issue_for`, the memory estimate counts the device CSR copy;
  mode_group_indices always gets `solve_at=ModePhysics._solve_at`.
- Bloch k (symmetry 2.2): settings bloch_kx / bloch_ky (`definition_bloch.bloch_params`); `ModePhysics.wrap_factors` =
  `cross_section_modes.bloch_wrap_factors` enter `centre_fields(wrap_factors=)`. Code reading the centre fields without
  the factors refuses k != 0: shg_overlap (its 2 omega modes would need 2 k), thermal_noise `mode_pair_weighting`,
  optomechanics `_cross_section_mode` (validation errors). Self-test em_modes_bloch_plate.
- solver_defaults.conformal_boundaries: pec_domain (definition_common.apply_pec, duck-typed on `conformal_locator`) ->
  `add_conformal_pec_region(mask, inside)`; `_dual_cuts()` builds the cuts once (host, full grid incl. PML) and every
  `_cross_section_materials` passes them, so detached / group-index / thermal-noise solves see the same operator.
- Backends ('gpu', 'cpu'): the solve is host code on both; mode_fields, masks (`_check_cell_array(xp=)`), inject_source
  and subpixel smoothing follow `domain.xp`. Self-test em_modes_cpu_gpu (selftests_modes_backend).
- Scalar outputs (E5) `n_eff_1 .. n_eff_<n_modes>` = Re n_eff; `scalars()` lists found modes only, so a missing mode's
  derived value is NaN + warning. thermo_optic_phase_shifter reads `modes_n_eff_1`, the strip template the mode_table
  (both paths stay covered).
- `shg_overlap` (nonlinear_overlap.py, host numpy): monitor whose `record` runs a second eigen-solve at 2 omega
  (`ModePhysics.solve_detached`: nothing selected or recorded), pump / harmonic paired by rank (no tracking): kappa =
  sum A conj(e_2w) . (d : e_w e_w) over the selected cells (`point_materials()` areas, `centre_fields`), efficiency
  (omega eps0 / 2)^2 |kappa|^2. d is a model-axis Voigt d_il (physics.em never imports nlfd's crystal tensors). Bend
  (`bend_overlap_factors`): per arc length at x_ref, exact dV = s dA dz with E_phi = E_z / s, conformal s^2 dA; d in
  the local frame (ring_crystal 'local') or 'fixed' in the chip (bend only: `ring_harmonics` from RING_SAMPLES,
  `ring_row` appends RING_SERIES per `monitor.series`; signed R, phi = z / R, order m at delta_beta R + m = 0; self-test
  em_shg_overlap_ring). Mirror faces refused (constructor + validation): the integrand has no single parity.
  `bend_overlap` (BendOverlapMonitor): a straight `solve_detached(..., straight=True)`, each bent mode matched to the
  straight mode of largest `mode_overlap` (junction loss in dB; bend required; mirrors fine);
  self-test em_modes_bend_overlap. Self-test em_modes_shg_overlap (periodic-grid TEM modes: closed form to round-off).
- No absorbed power: no `take_absorbed_power`, `state()['q_accum']` None, no 'absorbed_power' output, which keeps
  em_modes out of em_heat_joule's sources (coupling/definitions). Thermo-optic writes all three arrays.
- `definition_modes.register` (called last from definition.register) registers the type, templates,
  `CROSS_SECTION_CPU_SELFTESTS` + `EM_MODES_SELFTEST_CASES`. Validation adds: another family-'em' type in the model
  (error), scipy missing (error; finalize raises too), mode_number > n_modes, `size_issue_for`, bend (periodic x error,
  `modes_2d.bend_factor_message`, warning without cpml on the outer side); bend reference defaults to size_x / 2 in both
  validation and ModePhysics; setting bend_model ('exact', 'conformal'). mode_table / mode_dispersion: selection_kind
  None.
- Vacuum pec box = example analytic_modes_pec_rectangular_waveguide.json; its spec / check stay in selftests_em_modes
  for em_modes_line_block. Templates strip_waveguide_modes, thermo_optic_phase_shifter (10 nm in the gold heater so
  the skin-depth check passes).
- Carriers (C2 stage 1, `selftests_carriers`, no solver code): N, P [m^-3] enter only as eps_r / sigma_e expressions
  (`plasma_dispersion_expressions`, Soref-Bennett 1.55 um; `plasma_dispersion` is its numpy twin, keep both in step;
  alpha stays at its 1.55 um value at other wavelengths). Carriers from electrostatics V are state-dependent materials
  (DMUs reading `electrostatics_V`) applied by the mode_analysis / frequency_domain background solve, never
  `_apply_background_couplings`. Depletion charge prescribed (closed-form W(V_R)); self-consistent needs Newton: Picard
  on exp(V / V_T) diverges. Example analytic_modes_soref_bennett_guide.json; self-test carriers_depletion_junction.

## em_modes on triangles (fem_modes, solvers/modes_fem; mesh plan 2.6, stage T3)
- One type, two solvers: `definition_modes._create` builds `FEMModePhysics` when `ctx.domain.grid_kind` is 'triangles'
  (walls from `fdfd_domain_requirements(settings)`: TriangleDomain has no BoundaryConfig), else ModePhysics. Setting
  `discretization` (quadratic default, linear). shg_overlap / bend_overlap stay tensor; `_triangle_issues` refuses
  periodic faces, bends and the thermal_noise / optomechanics studies, and applies the FEM size rule.
- Formulation: `fem_modes` module header (Lee-Sun-Cendes, A x = -beta^2 B x, x = [e_t; e_z], E_t = e_t / beta, E_z =
  -j e_z; ranking, PML drop, 1 W, peak phase as cross_section_modes). Stored FEMMode coefficients are physical
  ([E_t; E_z]); raw vectors (ARPACK v0) divide the z part by -j beta.
- PML ring: `pml_ring` raises when mesh.pml_faces widths disagree with the cpml faces; the stretch is the tensor CPML
  grading as a continuous function of depth (`boundaries._grading_values`, pml_dx = width / cpml_npml = max element
  size); ring outer side pec. pml_kappa_max 1 for leaky modes (kappa > 1 compresses radiating waves).
- Walls: essential = boundary edges of the non-PEC element set except outer edges tagged pmc / symmetric, plus Lagrange
  nodes of PEC elements; mirror factor 2 per mirror face.
- FEMMode duck-types the cross_section_modes.mode_overlap attributes (ex, ey, hx, hy, x_face_area, y_face_area at window
  quadrature samples): mode_overlap, ModeTableMonitor / ModeDispersionMonitor serve it unchanged; a change of those
  attributes must keep both solvers working. group_index on triangles: `fem_modes.fem_group_indices`, the
  Hellmann-Feynman form of the complex symmetric pencil A + beta^2 B (x its own left vector; A', B' central differences
  of two assemblies, no eigen-solve, no tracking); the Yee em_modes keeps the tracked `mode_group_indices`.
  ModePhysics and FEMModePhysics share `modes_2d._ModeStudyMixin` (study hooks, post-solve bookkeeping, labels,
  scalars, table / dispersion monitors): a solver gives `_solve_once(omega) -> (modes, info, warnings)` and
  select_mode. Both are RerunSolvers (sweep reuse_build): `reset_records` sets omega back to the wavelength setting's
  and calls `clear()` of every monitor that has one (ModeDispersionMonitor, ShgOverlapMonitor keep one row per solved
  omega; a new accumulating monitor adds `clear`); no em_modes value follows (its `wavelength` Param is not flagged).
- State: materials, state fields and eps slots are one value per element; `field_dofs(name)` = the P0 DofMap (contract
  of model/mesh_transfer). `fem_field`: materials P0, mode fields P1 / P2 nodal (`fem_modes.nodal_fields`: mean over
  the node's non-PEC elements, 0 at nodes of PEC elements only), cached per selected mode.
- Subpixel smoothing / conformal boundaries accepted and ignored (mesh follows outlines); `conformal_locator` None.
- Sizes: `estimated_unknowns`, FEM_LU_FILL_FACTOR (4, uncalibrated: compare summary 'solvers' factor_host_bytes) in the
  size rule, `_modes_fem_memory_estimate`, `_triangle_time_estimate` (FEM_MODE_ASSEMBLY_BYTES_PER_ENTRY / _CALLS
  uncalibrated). `size_issue_for_unknowns(n, factorization)` is None for cudss / cudss_mixed (memory estimate counts
  `cudss_operator_bytes`); work_counters add the last solve's unknowns / elements (the rib self-test reads them).
- `error_indicator()` (plan A3): `fem_modes.mode_error_indicator` on the solved layout / materials
  (`_solved_materials`): jumps of H . t, H_z and normal D across interior non-PEC window edges; ring, PEC and outer edges none, no element
  residual. The model self-test `adaptation_triangle_parts` imports this file's slab helpers.
- Shift offset: `prepare_fem_modes` takes cross_section_modes' `_SHIFT_RELATIVE_OFFSET` (1e-6) and recovers beta^2 with
  that same shift; otherwise the default guess sits exactly on a TEM mode (em_modes_fem_tem_on_guess). A shift that
  close to a mode makes its theta ~1e6 x the far wanted ones: ARPACK at tolerance 0 (eps) may then not converge them
  (rounding noise, 300 restarts, a warning); that self-test runs ARPACK at 1e-8.
- Factor: `solve_fem_modes` calls `factorize_host(operator, symmetric=True, row_scale=problem.row_scale)`: the pencil is
  complex symmetric (D = I), so cudss factors an LDL^T (LU if the symmetry check fails); row_scale replaces
  |diag A|^-1/2, which nearly cancels. SuperLU ignores both. A change that breaks the symmetry (conjugated tests, full
  anisotropic tensors with a non-diagonal stretch) loses the LDL^T and the cudss_mixed polish. cudss_mixed (LDL^T
  only): ARPACK on solves with MIXED_EIGS_REFINEMENTS (1) to MIXED_ARPACK_TOLERANCE (1e-6), then
  `_rayleigh_beta_squared` per vector (left = right eigenvector; info rayleigh_change, single_residual); mode vectors
  stay ~1e-6. A mixed LU refines every solve to 1e-12 instead. Selftests em_modes_fem_cudss (gpu: compares the leaky
  mode only; the other listed modes sit far from the shift and change with factor rounding) and
  em_modes_fem_cudss_model. cuDSS gain in 2D is device memory / host relief at the largest P2 meshes, not flops.
- Missing on triangles vs Yee (refused by validation): periodic / Bloch faces, bend_radius (would need nu_t / nu_zz from
  mu tensors, s continued into an x PML ring, the overlaps' metric), shg_overlap / bend_overlap, line_block shifted
  solver, thermal_noise / optomechanics (Yee overlaps), subpixel / conformal (moot).
- Self-tests (selftests_modes_fem): em_modes_fem_slab, em_modes_fem_tem_on_guess, em_modes_fem_leaky (CPU; leaky
  reference = the exact leaky root, not a tensor run), em_modes_fem_pipe / _rib (heavy) / _heated_strip (model runs).

## Body of revolution (em_fdfd_bor; body_of_revolution, bor_ports, solvers/fdfd_bor, renderer_bor, definition_bor; symmetry 2.3)
- BORPhysics and ConicalPhysics share `solvers/host_fdfd.HostFDFDBase` (solver setup and cusolver_qr refusal, omega,
  state / slots / finalize / loss protocol / labels / counters, the kept driven factor: `_operator_replaced`,
  `_factorize_operator`, `_solve_kept_factor`, cuDSS plan keep / release, the solve_summary factor texts, the float32
  display writer, poles / PEC, `_add_source`, `add_field_probe`: `fdfd_monitors.CentrePhasorProbe` of the class'
  field_components); their display fields are `harmonic.shown_phasor_fields`.
  `bor_boundary_config`, `bor_flux_line`, `bor_far_field_box` (solvers/fdfd_bor) serve BOR FDFD and FDTD alike.
  Wavelength batches keep their later members' (materials, assembly) for their own solve (`_batch_assemblies`).
- Host only (numpy / scipy; module header = the formulation). x = r from the x_lo grid edge (the axis: never a PML
  side; `bc_x_lo` has the single choice 'axis', mapped to a pmc face for the PML padding only), y = z, fields ~
  exp(-j m phi). Unknowns E_r (+r edges), u = -j E_phi (centres), E_z (+z edges): every curl entry is real, so the
  operator is complex symmetric and a lossless resonance is real up to one phase (E_phi ~ sin(m phi)); recovery
  E_phi = j u. Ring areas / volumes per radian are exact (rbar L = integral r dr), so discrete gradients are exact
  statics (k = 0) and the scheme is second order (em_bor_pec_cylinder).
- Resonances: `solve_bor_resonances` = `eigenfrequency.solve_resonances(spectral_map='squared_cayley')` (statics and
  massless rows map to 0; root nearer the Rayleigh quotient, `eigenfrequency.squared_cayley_roots`; round-off statics
  below 1e-6 k_t^2 dropped; statics need Re omega > 1e-6 omega_t). Why: plain shift-invert lets the static cluster take a rounding-dependent number of ARPACK
  slots (non-reproducible resonances above sqrt(2) k_t); the plain Cayley map ranks every k > k_t ahead of
  k < k_t / sqrt(2) (lost TM010). Host operator A = system.operator = K - k_t^2 M, mass -M; an own factor is of K - s M
  at s = k_t^2 (1 + `eigenfrequency.SHIFT_RELATIVE_OFFSET`) (a lossless target on a resonance made A singular), a
  passed line-block factor of system.operator fits as is. `order_eigen_solve` adds the info keys merge_order_solves
  reads (the triangle path shares it). `solve_conical_resonances` and fem_conical use the same map.
- cuDSS (agent_help batch): C^T diag(V_a / mu) C - k0^2 diag(V_E eps) with real C (the u choice) is complex symmetric
  itself, PML and loss included (C^T W C is a transpose product, no conjugate): D = I, no symmetrizer. Eigen:
  `solve_bor_resonances` passes symmetric=True and equilibration (|diag K| + k_t^2 |M|)^-1/2, so cudss factors an
  LDL^T; cudss_mixed (`eigen_role(mixed_factorization=True)`) runs loose solves + the float64 Rayleigh quotient (info
  rayleigh_change). No SuperLU size rule for an own cuDSS factor
  (solve_bor_resonances, BORPhysics.finalize, `_solver_issues`). Driven: `CUDSS_LINEAR_ROLE` offers cudss /
  cudss_mixed; `BORPhysics._factorize_host` factors an LDL^T (row scale `body_of_revolution.bor_row_scale`, also the
  eigen-solve's), not recyclable:
  `_retire_factorization` keeps a replaced factor as `_cudss_previous` and the next factorisation (wavelength,
  material, azimuthal_order, each order-sum term) refactorises it in place; the plan is reused within an axis class
  (m = 0 / |m| >= 1: an order sum 0..3 = 2 plans). Released before an eigen-solve; the time recipe drops GMRES
  recycling. Selftest em_bor_cudss_driven (selftests_bor_orders). A line_block 'shifted' node factorises the
  shift-invert operators instead (validation warning: cudss unused unless an order is rejected). 2D fill is mild
  (SuperLU copes to ~1e6 unknowns): the gain (unmeasured) is host relief at the largest many-order grids.
- Axis: E_r, H_z, H_phi sit on it, E_phi / E_z / H_r half a cell off. The axis E_r dual face and H_phi dual edge have
  zero measure, so neither is ever an unknown: m = 0 keeps the H_z disc face, |m| >= 1 drops all three (`axis_class`
  0 / 1). At |m| = 1 the massless E_r row is the regularity slaving E_r(0) = -u_0 / m = j E_phi(r_0) / m and
  H_phi(0) = -j m H_r(r_0): `yee_fields` fills them (`_regular_axis`), the FDTD twin writes them into x / h after each
  update (they feed no update), and an 'r' source half on the axis edge moves to u of cell 0 as -b / m
  (point_current_rhs, BORTimeStepper.add_current); at m = 0 and |m| >= 2 that half is lost. Never give the axis E_r a
  ring volume (r0^2 / 2 made cell 0's Gauss row non-divergence-free and distorted the near-axis field at every
  resolution; example analytic_bor_transverse_dipole_m1 pins it). Recovered axis values carry no flux weight
  (line_flux / box_flux) but do enter the far-field samples.
- Walls all on the outer grid edge (unlike the 2D FDFD, whose pec / pmc planes are half a cell apart): pec = the centre
  value beyond reads 0, normal E on the edge free with half a dual volume; pmc and the natural end beyond a cpml layer
  drop the edge E and edge H rows. pec / pmc on z_lo = the antisymmetric / symmetric mirror; no periodic faces, no
  mirror choices (BORPhysics raises for a scripted symmetric / antisymmetric face: its padded ghost cell would move the
  plane off the window edge).
- PML: the cpml_axis_profiles centre-row stretch integrated into complex coordinates; the complex r enters every metric
  factor (cylindrical PML; em_bor_radiation). Never a separate kappa grading.
- Cell geometry from `Domain.cell_widths()` (float64), not `grid_cell_widths`: em_fdtd_bor steps the same operator on
  the same widths, Session's ring weights use the same edges. Materials: the 'modes' schema in drawing-plane names,
  diagonal in cylindrical components: eps_xx = eps_rr (E_r), eps_r = eps_phiphi (E_phi; the eps_zz property writes
  it), eps_yy = the axial eps_zz (E_z); `BORMaterials(eps, mu, pec, eps_rr, eps_zz)`, a None component = eps. E_r /
  E_z / mu points take ring-volume / length weighted means of their non-PEC cells (`_point_mean`: a conductor's surface
  E_r / H see the dielectric beside it); PEC points = every touching cell PEC. q_abs sums (-Im eps_c) |E_c|^2 per
  component.
- Orders: `BORAssembly` (one omega and material state) keeps C = C0 + m C1 and the active sets / curl columns per
  `axis_class`; `system(m)` is one product C(m)^T W C(m) (`_curl_gram`; 3 x fewer host products than K0 + m K1 +
  m^2 K2 for the usual one order per assembly), so an m sweep reads materials and assembles once. `assemble_bor` =
  BORAssembly(...).system(m); `order_family(class)` = (K0 - k0^2 M, (K1, K2), unknown_index), the line-block shift
  family (built per call, equal to system(m) up to rounding). Setting `azimuthal_orders`: the eigenfrequency study runs `solve_bor_resonances` per order on the one
  assembly and `merge_order_solves` concatenates order by order; `BORResonance.azimuthal_order` (text
  'm = <m>, lambda = ...'); resonance numbers are global, so `select_resonance` rebuilds the order's system from
  `_eigen_assembly` when it changes; `bor_resonance_table_result` appends column azimuthal_order last (column positions
  stay). In stationary / frequency_domain it is passed as `driven_orders`. Scripts set `BORPhysics.azimuthal_order`
  (keeps the assembly, drops the factor, ports re-solve).
- Driven order sum (`driven_orders`, m >= 0 distinct; body_of_revolution header): every point source is a 3D element at
  phi = 0 of moment I d: `BORPointCurrent.term_rhs` = the ring current I (r, z) or I d / (2 pi r) (phi) at every order,
  except the axis-cell 'r' at |m| = 1, whose moved half -b / m flips with m (term_rhs(order=-m)). Per order one factor
  of A(m) (SuperLU or the line-block batch per axis class; nothing kept, no recycling) and
  `order_pair_fields(minus_rhs=b(-m))` (x_-m = P A(m)^-1 P b(-m): one solve when P b(-m) = +-b(m), two for mixed
  r / z + phi sources; -m fields = +m ones with E_phi, H_r, H_z negated). Monitors take `record_order(physics, system,
  terms)` per order, then `record(physics)` (a new monitor type needs both; mode ports are refused at solve and by
  validation): flux = sum of the term fluxes, `power_orders` per |m|; `order_sum_far_field` adds amplitudes with
  exp(-j m phi) at phi = 0 (cos(angle) >= 0) or pi; probes / shown fields = the phi = 0 sum (`yee()`; field integrals
  of it are no 3D integrals), q_abs the phi mean (so em_heat_joule stays right). The -m partner relies on rotational
  invariance and on building b(-m) itself (point currents: term_rhs(order=-m); ports: refused).
- Line block (bor_line_block, role 'shifted', every study; Automatic stays SuperLU): lines = z rows or r columns,
  whichever is cheaper. `order_factorizations`: one set_operator per axis class, one kept LineBlockFactor per chunk of
  orders, released before the next. `solvers.line_block.LineBlockFactorization` raises `RejectedLineBlockSolve` above the accept tolerance: the eigen path redoes that order with SuperLU, the
  driven path marks the system rejected (SuperLU until it changes); both warn. The host SuperLU size rule applies only
  to those fallbacks (finalize, `_solve`, validation and `solve_bor_resonances` with a passed factor skip it;
  `_factorize_host` raises). No GMRES recycling on line-block factors. The role solver is shared between driven and eigen use (`owner=self`; `_line_factor.valid` turns False once
  re-bound). Wavelength batches (single-order driven; an order sum keeps per-order batches):
  `plan_harmonic_frequencies` as FDFDPhysics; an omega not in the kept batch assembles the next up to
  fdfd_line_block.MAX_BATCH_WAVELENGTHS planned omegas (stopping at a differing active set) into one
  `wavelength_factorizations` batch keyed by (`_material_serial`, order): every `_mark_assembly_dirty` but `omega_only`
  bumps the serial, so a change of the assembled system must go through it; `_drop_line_factor` keeps the batch, the
  eigen / order-sum paths release it. The estimate counts one wavelength. Selftest em_bor_line_block_wavelengths.
- BORPhysics: FDFDPhysics attribute set, polarization 'bor' (sheet placement helpers refuse it; flux_line_cells /
  box_cells / port_cells serve it), `axisymmetric` True (2 pi r ring weights of the travelling wave; a standing wave
  integrates to half). Roles: 'linear' = sparse_direct (only `factorize_host` / `solve_recycled_host`: host SuperLU
  whatever the back_end; 'cusolver_qr' is a validation error and refused by the constructor), 'eigen' =
  arpack_shift_invert (superlu / cudss / cudss_mixed; `solve_bor_resonances(..., eigen_solver=)`); scripts get private
  ones reported in `work_counters`. A material / omega change re-reads the MaterialGrid into a new assembly. The driven
  SuperLU factor is kept while omega, m and materials stay; after a material change at the same omega, m and active
  set (thermo-optic Picard) GMRES on the old factor runs first (sparse_direct RECYCLE_*). `centre_phasors` (host) feeds
  the probes, `yee()` (cached per driven solve) the flux / far-field monitors. Point currents (`point_current_rhs`)
  use the FDTD sign as the 2D FDFD. Estimator `memory_estimates.bor_memory_estimate` (host factor
  2 x host_lu_bytes(3 n); cudss: operator copies (eigen: EIGEN_PHASE) and an external factor item; line_block by
  the path the study takes: eigen and driven order sums bind the order family, `bor_line_block.line_block_bytes`
  run items (K0 / K1 / K2 caches, min(orders, MAX_CHUNK_SHIFTS) factors); single-order driven solves hold whole
  operators, `driven_line_block_bytes` (one operator and factor, or the batch of min(study wavelengths,
  MAX_BATCH_WAVELENGTHS) operators with a factor each; no K1 / K2 caches)); its time recipe leaves a cudss eigen
  node "not estimated". The tensor cuDSS factor is counted, not grown by n log2 n: `_bor_factor_entries` runs the
  fem.dissection count on the operator's cliques (one per curl point: its a_phi / a_r / a_z members; m = 0: the u
  and E_r / E_z blocks apart) -> `cudss_factor.FactorFill` (values + kept matrix + per-row part; cached per grid,
  seconds above ~1e6 cells). The growth law held no per-row part (em_bor_cpu_gpu ports, 19,200 unknowns: 812,107
  entries x 16 B = 13.0 MB of 25.6 MB permanent, the rest the kept matrix and ~540 B / row; it booked 1.5x the
  entries and still read 27 % low). The other tensor cuDSS estimators (conical, modes, 2D FDFD) still use it.
- Flux (module header): `box_flux` (closed cell box) equals `source_power` = pi / (omega mu0) Im(x^H b) to solver
  precision when its faces lie in lossless PML-free cells. r-face weights: inner cell centre r for E_phi H_z, edge
  rbar for E_z H_phi. `line_flux` (flux_monitor): share-weighted span ends (axis edge weight 0); exact for a line from
  the axis across the whole field.
- Far field: `far_field_samples` (a box bottom on a pec / pmc z_lo wall at grid edge 0 adds mirror images) +
  `far_field_power` (Jacobi-Anger, observer at phi = 0). Bessel calls dominate: keep new far-field paths on
  `_RadiationKernel` (J_|n| once per sample geometry). The result angle runs from +r toward +z (theta =
  arccos(sin angle)); cos < 0 is the phi = pi side.
- Ports (bor_ports header: exact discrete modes of a 3-row strip, quadratic pencil in psi = exp(-j beta h), one-sided
  TFSF source). Lines horizontal only (validation error otherwise), from the axis or annular (`start` > 0: strip cells
  below are PEC); an end on the last window cell continues through the r_hi cpml or ends on its wall, an end inside is
  pec. A metal end is exact when the model's cells beyond it are PEC (metal radius = start - 1/2 cells); an inner end
  without model PEC counts in edge_field and warns. A flux_monitor line across a conductor surface should start on the
  axis (the surface E_r's dual is half in the PEC cell, which an annular line's span end drops). n_group: mass term
  only (no material dispersion, PML frozen). Ports on identical lines (span, row width, line materials, mode count)
  share one PortModes per (omega, order, assembly_count) (`BORPhysics.port_mode_solves`): PortModes / PortMode are
  read-only, each port tracks and phase-aligns its own copy (`aligned_mode`), never mutate a shared one. A new solve
  of a port line starts ARPACK from its last PortModes at the same order ([X; psi X] summed + the guarded random share,
  E5; `solve_port_modes(previous=)`, only when the slice keys are equal), else from the seeded random vector.
- `BORAssembly` builds C0 / C1 as one COO per matrix (`_block_csc`, the canonical CSC scipy's bmat of the kron blocks
  gives: same pattern, same values); a new curl block goes in there as (offsets, sign) + `_identity_kron` /
  `_kron_identity` entries, never as a scipy kron / bmat (their overhead was ~70 % of the assembly on small grids).
- Backends ('gpu', 'cpu'): every solve, factor and port is host; only BORPhysics.fields, pec_cells / pole masks and the
  MaterialGrid live on `domain.xp` (material reads by `to_host`, masks checked with `xp=`). A line_block shifted node
  runs on the model's backend. Keep new paths xp-generic (em_bor_cpu_gpu).
- Not yet (plan 2.3): the 2.16 metric layer, ports on vertical lines (the metric breaks the psi^s ansatz), a measured
  line-block vs SuperLU crossover, line-block port mode solves.
- `BORAssembly(..., omega=None)` is em_fdtd_bor's real (unstretched) metric: `system()` / `order_family()` refuse it;
  `curl_parts(m)`, `e_point_means` / `a_point_means` and `yee_centres` serve both types: a change to the point or axis
  rules changes both.
- Self-tests (selftests_bor; CPU unless noted): em_bor_pec_cylinder (m = 1 re-solved with the target on its discrete
  resonance), em_bor_uniaxial_cavity, em_bor_radiation, em_bor_order_sweep, em_bor_coax_ports, em_bor_cudss (gpu);
  template microdisk_bor_eigenmodes; examples analytic_bor_dipole_radiation.json, analytic_bor_pipe_te11_section.json.
  selftests_bor_orders: em_bor_line_block_family, em_bor_line_block_resonances, em_bor_order_list_model,
  em_bor_offaxis_dipole (mirrored -m vs direct A(-m)), em_bor_offaxis_dipole_model. Multi-quantity CPU checks return
  normalised errors (error / tolerance). selftests_bor_backend: em_bor_cpu_gpu (heavy). SuperLU-only analytic model
  runs use the cpu backend; line_block runs keep the default (the cupy line block is its own implementation). Backend
  comparisons scale a component by a common peak (er / ez by the |E| peak): a symmetry-zero component is roundoff and
  differs between numpy and cupy.

## em_fdfd_bor on triangles (fem_bor, solvers/fdfd_bor_fem; agent_help batch)
- One type, two solvers: `definition_bor._create` builds `FEMBORPhysics` when `ctx.domain.grid_kind` is 'triangles'
  (walls from `_domain_requirements(settings)`: the axis x_lo as pmc, no ring there), else BORPhysics. Setting
  `discretization` (quadratic default, linear; ignored on tensor meshes). PhysicsType / resonance_table / pec_domain
  `grids` hold 'triangles'; every source / monitor / port stays tensor; `_triangle_issues` refuses every study that does
  not eigen-solve it (`definition_common.eigen_solved`; `fem_bor.electric_field_at` serves the optomechanics overlap)
  and applies the FEM size rule (fem_modes `size_issue_for_unknowns` with `FEM_BOR_SIZE_REMEDY`; none for cudss).
  Unknown / fill / cuDSS byte estimates are em_modes' (`fem_modes.estimated_unknowns` etc.), uncalibrated.
- Formulation: `fem_bor` module header. e = (E_r, E_z) Nedelec, w = r u = -j r E_phi Lagrange of the same degree;
  K(m) = K0 + m K1 + m^2 K2 (`FEMBORAssembly` once per omega / material state, `system(m)` restricts to the order's
  free set) and discrete gradients (grad p, -m p) are exact statics. Never change w to u: the kernel would leave the
  discrete space (spurious modes for m != 0).
- Axis: w fixed on the axis at every order, E_z for |m| >= 1 (`free_by_class`). The 1 / r integrands are log-singular
  on axis elements unless G_r -> 0; interior Dunavant points make that a penalty: linear elements lock w / rot e in the
  axis layer (m = 0, |m| = 1 converge slowly), quadratic do not. Every assembly uses `FEM_BOR_DEGREE` (6). Axis edges
  are found geometrically (outer edges with both nodes on x = window x0), not by edge_tags; a mesh below r = 0 is
  refused. Display samples axis nodes `AXIS_NODE_SHRINK` toward the centroid (divisions by r), so axis values are
  approximate.
- PML: fem_modes `pml_ring` (r_hi and z rings, pec outer side), stretched radius `stretched_radius` in every metric
  factor (complex symmetric). The Yee path ends a cpml in a pmc edge instead: compare far from the PML only.
- Solve: `solve_fem_bor_resonances` = eigenfrequency.solve_resonances (squared Cayley, host K - k_t^2 M, mass -M,
  unstretched energy matrix, symmetric=True, equilibration as the Yee path) + `order_eigen_solve`; cudss / cudss_mixed
  as the Yee path (LDL^T, D = I).
- FEMBORPhysics: BORPhysics' eigen attribute set (conformal_locator None, so apply_pec calls add_pec_region), no driven
  pieces. e_intensity / q_abs = ring-weighted element means (`element_means`); `fem_field`: materials and
  absorbed_power P0, other fields nodal (`nodal_fields`, means over a node's non-PEC elements). Layout rebuilt only when
  the PEC elements change; every solve re-reads the MaterialGrid.
- Missing on triangles (refused): driven solves (flux / far-field / port need FEM equivalents), line block (no line
  structure), subpixel / conformal (moot). Wanted next: driven solves on a 'linear' cuDSS back end for wavelength
  sweeps.
- Self-tests (selftests_bor_fem, registered by definition_bor; not yet run, tolerances first estimates; CPU unless
  noted): em_bor_fem_pec_cylinder (no spurious), em_bor_fem_uniaxial_cavity (eps_rr / eps_phiphi / eps_zz mapping),
  em_bor_fem_mie_sphere (curved interface + PML vs Mie poles), em_bor_fem_cudss (gpu), em_bor_fem_sphere_model. The Mie
  pole helpers copy physics/em3d/selftests_fdfd3d's (physics/em may not import em3d): flagged duplication.

## Body-of-revolution FDTD (em_fdtd_bor; fdtd_bor, solvers/fdtd_bor, renderer_fdtd_bor, definition_fdtd_bor; 2.3)
- Triangles / cuDSS decision: Yee only, grids ('tensor',): the leap-frog stepper has no linear solve; a DGTD / FETD on
  triangles would be a second solver (conforming answers belong in em_fdfd_bor on triangles).
- fdtd_bor module header = the scheme. One real field set on the em_fdfd_bor full layouts (inactive / PEC points held
  0, never removed, so Yee views are plain reshapes; the |m| = 1 axis E_r / H_phi are overwritten with the slaved
  values after each update, `_axis_slaving`): x = (E_r, u, E_z), h = (H_r, H_z, H_phi) with S = -1 folded into the a_r
  rows, so h is the physical H of the standing wave (E_r, E_z, H_phi ~ cos(m phi); u = E_phi, H_r, H_z ~ sin(m phi)).
  Operators are host products of `stretched_assembly(...).curl_parts(m)` (= `BORAssembly(..., None)` on kappa-widened
  r_hi cpml widths): the same C and point rules as em_fdfd_bor, hence the exact leap-frog map (DFT phasors solve
  em_fdfd_bor at omega_tilde; resonances at (2 / dt) asin(omega dt / 2)). DFT timing: x at n dt, h and the sources at
  (n + 1/2) dt; `source_spectrum` sums S on the injection times. Changing a sample time breaks the exact map
  (selftests em_fdtd_bor_*).
- Phasor convention (`phasor_yee`): j on E_phi, H_r, H_z; a 'phi' current q_u = r I s(t) is the em_fdfd_bor phi
  current at phase +90 deg: powers and moduli agree, phases of phi sources differ.
- dt = courant x 2 / omega_max of the vacuum operator with the build's PEC (`stable_time_step`: dense eigvalsh up to
  400 active points, else eigsh 'LA' with a 1e-5 margin; an order sum seeds each order's eigsh with the previous
  order's eigenvector, `warm_start`), not the 2D CFL rule; `refresh_coefficients` refuses a point
  eps / mu below 1 and, with poles, eps - sum w P below `stability_ratio`^2 (P = physics/em `_pole_step_term`). PEC and
  poles only before finalize; material changes re-read the arrays and refresh decay / gain, injection, pole correction
  and field-injection weights.
- z CPML: 1 / kappa on the C0 d_z rows plus one psi driver (float32 CSR of the stretched rows) per side;
  `cpml_axis_profiles`: edge k = H row k - 1, centres = E rows.
- r_hi CPML is cylindrical: the em_fdfd_bor radius stretch at alpha = 0 (pml_alpha_max acts on z only). kappa enters
  the assembly widths (shared by every stepper of an order sum, so flux / source helpers see rho = r inside the window
  only); the layer rows leave the main product and go through `_RadialPlan` / `_BatchRadial`, added into the curl
  buffer before the fused update. It then differs from em_fdfd_bor only by the recursive-convolution discretisation
  (em_fdtd_bor_cylindrical_pml).
- Poles (`_PolePoints`, velocity form as the 2D engines) on active E points with e_point_means share w > 0; update
  from x^n between the H and E advances, correction after the E advance. `leapfrog_susceptibility` = the stepped chi:
  with it DFT phasors solve em_fdfd_bor at omega_tilde exactly (em_fdtd_bor_dispersive_cavity);
  BORFDTDPhysics.stepped_materials(omega) builds those materials (sigma terms x cos(omega dt / 2) too), as the ports do.
- Stepping: every cupy kernel (`_UPDATE` / `_UPDATE_PML`, `_RADIAL_POLES`, `_DRUDE_STEP` / `_LORENTZ_STEP` /
  `_POLE_CORRECT`, `_FIELD_INJECT`, `_POINT_ADD`, `_AXIS_COPY`, the DFT kernels, `_CSR_PRODUCT`) has a numpy twin in
  the same class; runs on 'gpu' and 'cpu': keep every new path xp-generic.
- Lockstep batch (fdtd_bor header): a BORTimeStepper holds host plans and coefficients only; its x / h are rows of a
  `_LockstepBatch` (a single order: its own batch of one; BOROrderSum: one batch of all members, `own_batch=False`),
  which steps every member in one launch per kernel (block-diagonal CSR products by `_CSR_PRODUCT`, not cuSPARSE).
  So the launch count per step does not grow with the member count. A new per-step term goes into the batch (flat
  indices = member position x layout size + own index), never into a per-member loop; host coefficient changes reach
  the device through `_LockstepBatch.upload` / `injections_changed`; spectra live in stacked `_SpectraGroup`s whose
  member views are BORSpectra.
- Physics: BORFDTDPhysics builds the stepper in finalize (sources / monitors attach then); `engine` serves the shared
  resonance_monitor (H components delayed 0.5 step) and BORHarmonic. No absorbed_power (no em_heat_joule), no TFSF
  plane wave / mirrors. Dispersive regions are recorded by `add_dispersive` until finalize. Flux / far field reuse
  body_of_revolution.line_flux / far_field_samples on the DFT phasors with the real-metric assembly as `system`;
  power_per_cw / W/sr only with one shared source waveform.
- Mode ports (fdtd_bor_ports header; single order, refused with azimuthal_orders): bor_ports.solve_port_modes at
  omega_tilde of the carrier / each monitor frequency, line materials from stepped_materials; the source adds two
  `add_field_injection` sets (H interface rows at n dt, upstream E_r / u rows at (n + 1/2) dt) from `port_operators`
  and the PortModes `strip` curl; E_z on the interface edge counts as total in both directions, so only d_z entries
  cross. The monitor decomposes with bor_ports.slice_amplitudes. Modes are solved once (no material-drift re-solve);
  off the carrier the launched power deviates (em_fdtd_bor_pipe_ports normalises by the incident forward power).
- Engine interface: `engine` is a BORTimeStepper or (azimuthal_orders) a `BOROrderSum`; physics, monitors, BORHarmonic
  and the renderer use only dt / n_step / time / orders / order_sum, step, add_current, add_spectra (BORSpectra /
  OrderSumSpectra, both with `scale`), `spectra_terms` ([(signed order, yee dict)]: single order phasor_yee, order sum
  physical travelling terms), `harmonic_yee`, `probe_value` (`FieldProbe.record_value`), yee_views, field_energy,
  refresh_coefficients. A new consumer goes through these, never through x / h / fields of one stepper.
- Driven order sum (fdtd_bor header): (order, family) steppers on one shared real-metric assembly, families from the
  source components at finalize (`order_sum_families`; a later source of a new kind raises), lockstep dt = courant x
  min of `order_time_step_limit`; member currents `order_sum_current` (2 x the em_fdfd_bor term at m > 0, -2 x on the
  phi family); phi = 0 field by `phi_zero_weights`; flux per |m| = line_flux of the combined signed-order terms (A and B
  families add inside one signed order before the flux: their cross terms cancel only between +m and -m); far field
  `order_sum_far_field`. Results as em_fdfd_bor's order sum. Memory: `_fdtd_bor_members` counts the steppers.
- Self-tests (selftests_fdtd_bor): em_fdtd_bor_cavity_ringdown (vs leap-frog-corrected solve_bor_resonances),
  em_fdtd_bor_radiated_power, em_fdtd_bor_model_ringdown (E_z and E_phi tables: in the PEC cylinder TE modes have no
  E_z and TM_mn0 no E_phi, so one component misses lines). selftests_fdtd_bor_orders (heavy): em_fdtd_bor_offaxis_dipole,
  em_fdtd_bor_offaxis_dipole_model. Gotcha: an order cut errs on the phi = 0 pattern by amplitude terms (coherent sum),
  not power shares (at k rho_s = 0.39, |m| <= 1 is 7 % off U_max while m >= 2 holds ~1 % of the power); a phi current
  needs one order more (its order m pairs J_{m-1}, J_{m+1}). selftests_fdtd_bor_media: em_fdtd_bor_dispersive_cavity,
  em_fdtd_bor_dispersive_model, em_fdtd_bor_cylindrical_pml (vs same-grid 5e-3 and far-PML em_fdfd_bor 3e-2: a PML
  reflection r moves a radial line's source power by up to 2 |r| at any radius, so a far-PML grid is no tighter
  reference), em_fdtd_bor_pipe_ports (heavy, launch at the carrier).

## Subpixel smoothing (igor 1.5; subpixel_smoothing.py, geometry in model/subpixel_geometry.py)
- Switch: `solver_defaults.subpixel_smoothing` (model-wide), honoured by the types with
  `PhysicsType.subpixel_smoothing` (em_fdtd_ez / _hz, em_fdfd_ez / _hz, em_modes; em_fdtd_3d: em3d sub file) outside
  the optimization study (`schema.subpixel_smoothing_physics`, the one rule), on both backends (backend-neutral geometry
  build, `core.kernels` twins for the four averaging kernels; `subpixel_smoothing_cpu_gpu`). The builder computes one
  `core.interface_geometry.InterfaceGeometry` after the static materials and calls `set_subpixel_smoothing(geometry)`
  right after create; the solvers own it (no ctx reference: Session.close frees it).
- The geometry stores n_d^2 of the hz layout (faces_y: n_x^2 of E_x, faces_x: n_y^2 of E_y). em_modes ('modes') has
  E_x on hi-x and E_y on hi-y faces: it uses the same sets with 1 - n_d^2 (`_COMPLEMENT_NORMAL`) and the 'cells' set
  for E_z (arithmetic). Its deltas (`point_deltas`) enter `CrossSectionMaterials.eps_point_deltas`, which
  `cross_section_modes._point_materials` adds before the bend map, so every solve, `mode_group_indices`,
  `point_materials` (thermal noise, optomechanics, overlaps) and an em3d port (its own deltas) see the same eps.
- Locations: 'cells' (E_z), 'faces_y' (E_x on hi-y faces), 'faces_x' (E_y on hi-x faces; dual volume = two half cells;
  the last face of a non-periodic axis is never a location). Only volumes two materials share; each holds fill, n_d^2
  and two representative cells (nearest of a 3 x 3 / 3 x 4 neighbourhood with build material a / b). Averaged eps =
  1 / (n^2 <1/eps> + (1 - n^2) / <eps>) of the CURRENT eps of those two cells (E_z: <eps>): never write the averaged
  value into the MaterialGrid (couplings and n0 read the raw cells; the averaging re-runs on every rebuild).
- Geometry (model/subpixel_geometry.py; two rasters per build, 64 x 4 and 4 x 64 samples per cell): rasters clamped like the build centres, moments on the unclamped positions, so a
  PML-extended interface keeps its normal; a vanishing moment gives n^2 = 1/2. A sliver that covers no cell centre in the neighbourhood has no representative cell: that
  location stays unsmoothed.
- Consumers (all at once, or the port modes / FDTD / FDFD disagree): the FDTD engines call
  `subpixel_smoothing.correct_fdtd(engine)` inside `_rebuild_e_coefficients` after the fused kernel and BEFORE the PEC
  fold (ca / cb through the shared `e_coefficients` formula; hz also eps_x / eps_y, so the C5 media see it);
  FDFDPhysics `_assemble` adds averaged - standard (`add_deltas`) to the cell (ez) / face (hz) eps after the complex
  face means; `PortLine._host_rows` appends one `line_delta` row per permittivity row (material_drift sees them). Not
  averaged (documented limits): sigma_e, mu, pole membership, the FDTD sheet / point calibrations, the E_z nonlinear
  n0^2 (raw eps_r), the Courant check (raw eps), the q_accum split.
- A new kernel or consumer of face / cell eps on these types applies the same averaging (or reads eps_x / eps_y, the
  FDFD system, em_modes `point_materials`), and `subpixel_smoothing_cpu_gpu` (model/selftests_subpixel) gets a check
  for it. em_modes self-tests: the consistency case (complement normal, E_z = the E_z FDTD cells) and example
  em/analytic_modes_subpixel_slab_thickness (the smoothed n_eff still moves with the interface shift at O(dx^2): the
  interface sample weighs its volume's eps by E^2 at the sample).

## Definitions and self-tests
- Symmetry self-tests (`selftests_symmetry.py`, mostly 1e-5): em_fdfd_mirror_unfold (mirror_factor 2 / 1),
  em_fdfd_reciprocity, em_fdfd_parity_sectors (vs the two `core.mirror` sector runs and the study setting
  parity_sectors), em_fdfd_parity_sectors_eigen (symmetric sector = odd m), em_fdfd_parity_sectors_sweep,
  em_fdtd_parity_sectors_time (1e-4: float32 half / full runs differ by fused multiply-add round-off, which the
  post-pulse residual field magnifies), em_tfsf_mirror. The FDTD variants reuse `_mirror_model` and swap the type /
  params (`_fdtd_sector_spec`).
- Backend of the analytic FDFD / BOR self-tests: every FDFD path factorises on the host, so the analytic model cases
  run on the cpu (`selftests_fdfd.cpu_specs` in their case `specs`, never in the shared spec builders, which
  em_fdfd_cpu_gpu / em_bor_cpu_gpu / em_fdfd_adjoint_cpu_gpu reuse per backend); line_block runs keep the default
  backend. The gpu runs of the FDFD types are those parity cases (`_groups`; physics named 'em'); a new feature needs
  no gpu model of its own (root file "Selftest structure"). Not covered: conformal PEC in a
  driven run (the reciprocity pair names its physics from_a / from_b).
- Memory estimators (`memory_estimates.py`, E8): FDTD / FDFD / em_modes physics, the FDTD dft / flux / mode / far-field
  monitors and every pec_domain, each counted from the engine code it names (SuperLU host factor by
  `core.memory_estimate.host_lu_bytes`; the cusolver_qr workspace is outside the pool and stays with `size_issue`).
  Change an engine allocation -> change its estimator. Run-time device bytes (harmonic FDTD accumulators, FieldProbe
  trace growth) are `run_items`, never steady (the build ledger would flag them); field_probe / resonance_monitor traces
  are counted by the FDTD physics estimator (a feature estimator has no study duration). An eigen-solved FDFD physics
  (`definition_common.eigen_solved`: eigenfrequency study, or the optical physics an optomechanical_coupling names)
  replaces the driven SuperLU host items by the shift-invert ones. Run phases (transient_items, the phase constants at
  the top of memory_estimates; em3d imports them): the kept system / solution / a kept driven cuDSS factor's pool
  copies, rhs buffer and row scale are run items (`kept_cudss_items`; its solve vectors CUDSS_SOLVE_PHASE, never
  kept); ASSEMBLY_PHASE the FDFD assembly temporaries (`_FDFD_ASSEMBLY_BYTES`: the term plan's build sort, or a later
  omega's term buffer; the plan itself is a run item, `_FDFD_PLAN_BYTES`); a single solve (no `later_factorisations`) assembles before its solution and factor
  exist, so that phase is booked net of them (`core.memory_estimate.phase_before_items`); SOLVE_PHASE a solve's
  material temporaries; EIGEN_PHASE everything solve_resonances / a mode solve allocates and drops
  (`cudss_own_factor_bytes`: its own cuDSS factor's copies and one solve's vectors, device
  Krylov basis, `resonance_search_items`' filter: min(FILTER_BLOCK_COLUMNS, n_wanted) wide blocks); REFACTOR_PHASE
  `refresh_peak_bytes(n_unknowns=, mixed=)` (+ the kept `index_map_bytes` as a run item) only where the engine refactorises a kept factor
  (`previous=`: fdfd_2d, fdfd_2d_fem, host_fdfd
  BOR / conical driven) AND `later_factorisations(context)` (or a BOR order sum) says the study factorises again
  (em_modes / FEM modes / eigen searches plan a fresh factor per solve: no refresh); ADJOINT_PHASE per
  adjoint_objective feature (one shared transposed solve: the labelled per-feature phases make the largest count
  once). Every cuDSS factor is an external item from `cudss_factor_items` (factor_device_bytes on the full pattern,
  dimensions 2 here, 3 in em3d; an LU (Bloch, em_modes): symmetric=False, x cudss_factor.CUDSS_LU_OVER_LDLT; hz
  eps_xy stays an LDL^T of D A (it passes the runtime asymmetry check); an eigen factor's precision through
  `eigen_mixed`: cudss_mixed below MIXED_EIGEN_MIN_UNKNOWNS factors in double); an FDFD LU with an adjoint objective
  adds the transpose's factor. Entries per row are counted from the operator's stencil, never measured (tensor BOR
  `_bor_entries_per_row`: Cᵀ W C of the Yee curl, 19/3 for m = 0, 9 for m >= 1; 25 overbooked 3-4x): every
  per-entry item (copies, index maps, refresh, fill) scales with it. Still 25 in definition_bor's time estimate
  (`_BOR_ENTRIES_PER_ROW`, its assembly bytes per entry calibrated on it).
- `definition_common.write_dispersive_materials` (used by `create_em_physics` and the FDFD create) skips dispersive
  regions with an empty mask and writes each permittivity array's background (owner property, else the first Drude
  eps_inf, else 1) unless state-dependent; `create_em_physics` also sets `enable_q_accum` for track_absorption or when
  the model reads q_accum; neither finalizes. Gotcha: the PhysicsType needs `uses_dispersion=True` (default False),
  else the builder drops every pole silently (no eps_inf, no poles). `DISPERSION_POLE_PARAMS` is the pole vocabulary.
  The material Params of all five types come from `em_material_params(polarization, helps, tips)` ('ez' / 'hz' /
  'modes'); a type supplies only its help texts (em_modes overrides all four and the eps_zz tip, whose shared text says
  "the only one this physics uses").
- GUI texts (`Param.tip`, `choice_labels`, type `summary`): a tip every EM type shares lives in definition_common
  (`_MATERIAL_TIPS`, tip constants such as EZ_CURRENT_TIP, LINE_AMPLITUDE_TIPS, MODE_POWER_TIP,
  PROBE_COMPONENT_CHOICE_LABELS); type-specific tips and summaries stay in definition_fdtd / definition_fdfd.
- Texts and Params that differ only by solver ('fdtd' | 'fdfd') are built once in definition_common:
  `sheet_amplitude_param(polarization, solver, beam)`, `mode_end_text(polarization, solver)` (PortLine serves both, so
  the wall ends are the same); `quasi_static_domain_requirements` is the FDFD and em_modes one (no Courant factor).
- Split: definition_common = vocabulary, texts, build / result callbacks, physics validation (`make_validate_physics`,
  imported by em3d too); definition_feature_validation = the 2D feature placement checks; it imports
  definition_common, never the reverse.
- C7 validation (optimization study only): an adjoint_objective needs an enabled design_region on a physics of its
  type (else a constant J term: ez regions write eps_r, hz eps_xx / eps_yy); design regions sharing material-preview
  pixels are errors (same physics, same array or ez + hz), source / port / objective lines crossing a region warnings
  (definition_design, `geometry.domain_object_rasterizer`); an eps_xy reaching an em_fdfd_hz physics is an info
  (definition_fdfd `_off_diagonal_issues`: held fixed by the design and the gradient).
- FDFD feature validation reuses the FDTD callbacks with `time_domain=False` (`make_validate_mode_monitor`,
  `validate_far_field`, `monitor_line_interface_issues`); far field via `make_far_field_result_callback(params,
  harmonic_far_field_result)`; physics validation is `make_validate_physics(slot_arrays, time_domain=False,
  solved_wavelengths=all_solved_wavelengths)` plus `unused_wavelength_issues` and its own linear-solver checks.
  `FACE_MEAN`, `hz_point_source_params` and `validate_hz_point_source` are shared by the H_z types.
- FDTD and FDFD register the same `apply_*` callbacks: `source_time_kwargs` passes `phase=` when the feature has a
  Param 'phase' (FDFD) and the FDTD waveform when it has 'waveform' or no 'phase' (an FDTD port with a launch phase,
  em_fdtd_bor mode_source, gets both); monitors get `wavelengths=` only when they have 'wavelength_start'. So never
  give an FDFD source a 'waveform' Param or an FDFD monitor spectral params. Feature-only results go through
  `object_result(fn, with_solver)`.
- Unit-slip guards (lengths are SI m): `make_validate_physics` warns for a source (FDTD) / shortest solved (FDFD)
  wavelength below 1e-9 m or above 100 domain diagonals, and (time_dependent) for a duration below the longest pulse's
  2 t0 plus diagonal * n_max / C0 with a spectral monitor; the mesh-too-large error names the unit.
  `definition_helpers.study_duration` is the one duration lookup (also em_heat_joule).
- A horizontal line gives line_source / beam / mode_source / mode_monitor axis 'x' (along it) and flux_monitor axis
  'y'. TFSF / NF2FF margin checks and the mode-line checks (`_mode_line_messages`, `_pinned_positions(settings, axis,
  n, polarization)`) use the MeshEstimate edges with `core.domain.containing_cell` to mirror the runtime; every PortLine
  refusal must be a validation error first. em_heat_joule finds pulsed sources through a Param 'waveform' other than
  'cw'.
- Bloch model layer (`definition_bloch`, FDTD, FDFD and em_modes types; em_modes refuses k != 0 for shg_overlap and
  the thermal_noise / optomechanics studies, `_bloch_issues`): settings bloch_kx / bloch_ky [rad/m] (visible with
  periodic faces; `bloch_params`, `fdfd_bloch_params`); `fdtd_domain_requirements` / `fdfd_domain_requirements` turn
  periodic faces with k != 0 into bloch faces; FDFD validation `fdfd_bloch_issues` plus the period-spanning port lines
  (`is_fdfd_port_feature`); template fdfd_band_diagram; self-tests em_fdfd_oblique_fresnel /
  em_fdfd_hz_brewster_oblique, em_fdfd_oblique_tfsf, em_fdfd_bloch_empty_lattice (selftests_bloch). FDTD:
  `with_bloch_validation` adds k on a non-periodic axis (warning), absorbed power with k != 0 (error), real-set-only
  sources (info), k = 0 port modes (warning). `register` adds the 'bloch_band_diagram' template and `selftests_bloch`
  (`oblique_fresnel_case` stays for the backend case em_fdtd_cpu_gpu_media).
- Self-tests: the FDTD plane-wave, Fresnel (+ graded) and Drude cylinder cases of both polarisations come from
  `selftests_hz.plane_wave_case / fresnel_case / mie_drude_cylinder_case(physics_type, polarization)` (the
  'fresnel_interface' template is `fresnel_device_spec`); the Mie / Drude vacuum references run on a smaller domain
  with every shape shifted by whole cells (the TFSF aux line is box-relative; keep the shift a whole number of cells);
  em_wall_reflection_phase (FDTD) and em_fdfd_pec_reflection_phase pin every wall / mirror / pec_domain plane
  (`selftests.wall_reflection_spec`); em_operators_consistency needs a GPU (part B steps both engines against the A1
  operators: never a Python copy of the kernels); selftests_mode_ports holds the 'directional_coupler' template
  (`directional_coupler_spec(physics_type, source_params, spectral, study)`; em_hz_directional_coupler_fdtd_vs_fdfd),
  selftests_fdfd the 'fd_joule_heating' template; graded `*_graded` Fresnel variants (`selftests_hz.graded_interface_mesh`,
  inv_dx != inv_dy) raise when the mesh fell back to uniform (`require_graded`).
- Case records in tolerance units (worst of {item: error / tolerance}, tolerance 1) come from
  `model.selftest_helpers.normalized_result` / `tolerance_units_result` (re-exported by selftests for its siblings),
  never a local copy; selftests imports selftests_hz, so selftests_hz never imports selftests.

## Effective index 2.5D (effective_index, definition_effective_index; B4)
- `effective_index(indices, thicknesses, wavelength, slab_polarization, mode_number, cell_size)` (host, lru-cached:
  validation calls it per check) builds its own SlabLine along the stack normal: whole cells per layer (interfaces on
  faces), Dirichlet walls in claddings deepened (x4 while the mode is not found, else to 14 decay lengths, cap 50
  wavelengths). 'te' = slab 'ez', 'tm' = slab 'hz' with face weight 1 / (width-weighted mean eps): E parallel to the
  layers is continuous on an interface face, so the harmonic mean would be the wrong (first-order) choice. Top-view
  pairing: 'te' <-> em_*_hz, 'tm' <-> em_*_ez.
- Feature `effective_index` (all four in-plane types, appended after pec_domain): `apply` writes n_eff^2 into every
  `solver.permittivity_arrays` entry under the selection + `materials_changed`, after the static materials and the
  dispersive backgrounds, before finalize (so ports, Courant and the heat n0 see it); poles / sigma_e stay, a
  state-dependent permittivity material overwrites it (dynamic updates run after features). The auto mesh does not see
  n_eff (validation warning). Selftests em_effective_index_slab (CPU) and em_effective_index_rib (template
  effective_index_rib_waveguide, em_fdfd_hz).
