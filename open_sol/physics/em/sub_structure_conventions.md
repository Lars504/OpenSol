# open_sol/physics/em - conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). EM gotchas; the EM loss protocol and thermo-optic
slots are in `open_sol/coupling/sub_structure_conventions.md`. Current state, no changelog.

## Registered EM types (catalogue; details in the sections below)
Registration order: `definition.register` (FDTD before FDFD: the FDTD material Params stay first per name); each
type's vocabulary and refusals are in its help texts and validation.
- `em_fdtd_ez` / `_hz`, `em_fdfd_ez` / `_hz` (in-plane; only em_fdfd_hz has eps_xy with its 'sum' slot eps_xy_update,
  and gyration_z: section FDFD).
- `em_modes` (xy = waveguide cross-section; no other EM type in its model, except em_fdfd_ez / _hz in a thermal_noise
  study), `em_fdfd_bor` / `em_fdtd_bor` ((r, z) half plane; no other EM type), `em_fdfd_conical` (z-invariant
  structure at out-of-plane k_z): the 'modes' material vocabulary.
- Gyration (`schema.GYRATION_PROPERTY_NAMES`): a type reads a component only by declaring its material Param;
  validation refuses a non-zero one in a model holding an EM type that does not (`definition_common._material_issues`).
- Root rule "allocate where needed": poles comply in E_z, H_z and FDFD (masks cropped to their box); not yet: H_z
  `eps_x` / `eps_y`, per-cell `da` / `db`, TFSF outline masks, FDFD CSR matrices.

## Imports (inside physics/em; the cross-package rules are in the root file)
- `physics/em/__init__` imports solvers.fdfd_2d / modes_2d last (they import most siblings; `solvers/__init__` leaves
  them out). Host-only modules say so in their header.
- Definitions share `definition_common` (validation blocks, run-time estimates of the host-solved types, apply
  callbacks; (r, z) helpers in definition_bor) and `memory_estimates` (`host_factorization`,
  `_fem_resonance_memory_estimate`, `_fdtd_bor_members`): reuse, never a local copy. em_modes, em_fdfd_bor and
  em_fdfd_conical solve on the host on either backend: their recipes pass `backend='cpu'` to solver_work (line_block
  'shifted' factors stay on the run backend).
- Self-test helpers live mostly in `selftests` (also selftests_fdfd `cpu_specs`, selftests_mode_ports
  `slab_waveguide_spec`, selftests_bor, selftests_eigenfrequency): grep the sibling imports before writing one.

## FDTD and shared EM
- Permittivity array / slot names per polarisation only from `placement.PERMITTIVITY_ARRAYS` / `PERMITTIVITY_SLOTS`
  (hz reads eps_xx / eps_yy, never eps_r), never literals. A direct MaterialGrid write is followed by
  `em.materials_changed({...})`, else coefficients, source calibrations and the sigma_e loss boxes stay stale (a
  sigma_e write then loses heat outside the old box).
- MaterialGrid allocates no array by itself: EM arrays come from `materials.register_em_arrays` (FDTD engines,
  FDFDPhysics) and the EM material_schema, so an eps_r write before any E_z solver exists raises; code outside EM
  checks `property_names()` first.
- Both FDTD engines subclass `fdtd_2d_base._YeeEngine2DBase` (the hooks a new engine fills: its class docstring); the
  cell-mask helpers there (`record_dispersive_region`, `add_pec_cells`; array checks: `core.backend.check_backend_array`)
  also serve FDFD, em_modes, BOR and EME. EM solvers take cell geometry and per-polarisation vocabulary from `placement` and
  `boundaries.solver_boundary_config`: reuse, never copy (messages and checking order stay those of EMPhysics).
- Backends: every FDTD-path kernel is a `core.kernels` twin (core sub file "Kernel twins"): a new kernel gets its numpy
  twin in the same edit, host reads via `to_host`; sources and monitors take `xp=` (None = the default backend: EM
  solvers pass `domain.xp`). Selftests `em_fdtd_cpu_gpu` / `em_fdtd_cpu_gpu_media` (heavy).
- Stepper precision ('single' | 'double', STEPPER_ROLE; solver_plan S5): `EMPhysics(precision=)` builds fields /
  BoundaryManager / CPMLSided / TFSF with `dtype=`; every field-side array (coefficients, spacings, CPML, pole buffers,
  jz_total, TFSF aux) is in the fields' dtype and kernel scalars are numpy scalars of it (`engine._real(...)`,
  `_dt_real`, `_inv_dt_real`), never DTYPE(...). float32 also under double: MaterialGrid arrays, coupling slots,
  q_accum, PEC keep masks, source profiles / calibrations, FieldProbe traces, box DFT spectra (complex64; line spectra
  complex128), subpixel coefficients (scattered into float64 ca / cb), the Courant check; renderers return float32,
  state() float64. Selftest em_fdtd_double_cavity_phase (heavy).
- Yee edge: on a non-periodic axis the last face row is never updated and cell 0 reads its missing lo face as 0, so
  the grid edge is PMC (ez) / PEC (hz) half a cell beyond the outermost centres (only cpml sides see it).
- Times: a source injects at the centre time of the update it follows (E: t+dt/2; H: t), never the new field time
  (dt/2 late makes DFT / S(omega) lead the FDFD phasor by e^{+j omega dt/2}); when monitors run, H-type fields are dt/2
  behind E (`monitors.py` re-times them). Step order: each engine's `advance` docstring.
- PEC: ca = cb = 0 is folded in at the end of every E-coefficient rebuild (`_fold_pec_keep`: a new coefficient path
  ends with it); the per-step keep re-zeroes only `_pec_box` after injection and wall re-pin.
- Conformal PEC (solver_defaults.conformal_boundaries, S7.6; method: `conformal_pec.py` header): `pec_cells` stays the
  staircase mask (ports, poles, calibrations, FDFD read it); the engine gets the conformal data once in `finalize`.
  The stability floors bite at courant 0.99 (disk-cavity errors ~ 1 / R), courant_factor 0.7 relaxes them (selftest
  conformal_pec_operator). Not conformal: BOR, the Courant check, calibrations.
- Conformal poles (same option, FDTD only; `conformal_poles.py`): `definition_common.write_dispersive_materials` reads
  `BuildContext.conformal_locator_of` and passes ConformalPoleWeights to every pole's `add_dispersive(weights=)`;
  `dispersive_regions` keeps the cell mask, the Courant check reads `_pole_stability_regions`. Not conformal: FDFD /
  em_modes poles. Selftests em_conformal_pole_weights, em_conformal_drude_film (heavy).
- FDTD step graphs (gpu; capture rules: root file; what each graph holds: `fdtd_2d_base` header, fused rows:
  `boundaries`): fields, psi, ca / cb / da / db, the 1D spacings, the conformal arrays and the H_z pole face totals are
  rewritten in place only, never reassigned (graphs and the H_z pole face views hold them); a new replaceable input of
  a block joins its key (a new `set_conformal_*` tuple changes the H key; poles are only appended, their counts are in
  the E key). Poles join the E graph unless loss accumulation, a gain / nonlinear medium or the eager row copies
  (`fused_walls` False: < 3 cells along a pinned axis) sit between them. A change of what is captured updates
  `definition_fdtd.fdtd_time_step_estimate` (gpu: 1 + n_sets launches, no probe launches).
- FDTD probes (gpu): FieldProbes are recorded at the end of the real set's E graph (`core.monitors.FieldProbeRows`) and
  re-sampled after the eager tail per `_probes_need_resample`: a new eager E-side writer after the E graph joins it.
- Wall re-pin: both engines re-pin the outer-face rows after injection (a new engine must too), so a source on a
  pinned row / face does nothing (line / point / mode sources warn). `_walls_need_repin` skips it only for the exact
  point-source types of `_point_source_cells` clear of the walls and no post-wall writer: a new source type or post-wall
  writer re-pins until listed there. An E_z symmetric face next to the copied row is frozen, not re-zeroed, so an
  injection there accumulates.
- H_z face values and faces <-> centres only through `yee_grid` (its face-material comment is the definition): port
  modes are exact only while FDTD, FDFD, slab modes and sheet calibrations agree; the fused `e_face_coefficients`
  kernel inlines the mean, so a weighting change updates it too (em_operators_consistency part E).
- Sources are current-calibrated (`sources.py` calibrations; what users are told: the `definition_common` help texts):
  FDTD sheets use |eta / cos(kappa/2)| at the centre frequency, FDFD the exact complex value. Calibration is lazy
  (dirty flag, applied in `EMPhysics.step` / `finalize`): before `finalize` calibrated sources inject 0 and
  `engine.advance` alone never recalibrates. Injection weights carry cb: recompute them when it changes, never freeze.
- Courant: `EMPhysics.finalize` checks Yee + leap-frog pole stability per cell incl. PML (poles are NOT stable whenever
  Yee is: eps_inf = 1 aluminium needs dx <= ~5 nm; selftest em_drude_courant_bound); scripts skipping `finalize` and
  runtime slot rewrites are unchecked.
- J.E accounting runs only while `engine.enable_q_accum` (the model enables it for track_absorption or any reference:
  `definition_common._q_accum_referenced`); `absorbed_power` = the current window, else `last_q_accum`, else zeros;
  without em_heat_joule the window is the whole run.
- Poles (em3d too): the velocity-form ADE, the cb factor of the E correction and the two time-centred loss halves are
  derived in the `fdtd_ez_2d_kernels` / `fdtd_hz_2d_kernels` comments; a new pole path follows them (never J from the
  predictor E^{n+1}, never a1 P^n + a2 P^{n-1} in float32).
- Boundaries (`boundaries.py`; config checks, CPML row layout, psi coefficient rule and wall / mirror rows are stated
  there): CPML rows only from `cpml_side_rows` / `cpml_axis_profiles` (em3d `cpml_slabs` too), never re-derived. ez
  and hz walls are half a cell apart. A mirror face's `npml_sides` padding is one ghost cell, not CPML: code sizing
  CPML reads the bc type, never npml_sides alone (`_cpml_psi_bytes`).
- Bloch k != 0 (FDTD mechanism: `fdtd_2d_base` engine init and `_FieldSet`): EME refuses it
  (`operators.check_zero_bloch`), `cross_section_modes` handles it for em_modes, conical and em_fdtd_3d ports; any
  other caller checks k itself. Probes, renderers, state() and couplings read the real set only; enable_q_accum and
  C5 media raise. New wrap kernels take the partner fields + wrap sines of `_FieldSet`. Known gaps: FDTD hz
  face-to-centre collocation of cell 0 lacks the Bloch phase; mode ports solve k = 0 (validation warns).
- Unfolded results (symmetry 2.5, `core.mirror`): flux multiplies by `placement.line_mirror_factor` (extra
  'mirror_factor'; monitor objects stay raw; flux adjoint objectives carry it), Session 'integral' likewise; far
  fields add the image contour (`NearToFarGeometry.images`). Not unfolded: 2D mode-port powers, source 'power',
  probes, DFT monitors.
- TFSF (FDTD and FDFD, uniform or graded; `placement.tfsf_box_cells`): +x only; FDTD injects the vacuum incident
  field, FDFD the background's (`materials.tfsf_outline_warnings`); amplitude = incident E_z (hz: E_y).
  `monitors.near_to_far_pattern` is the one far-field formula (FDTD re-times H spectra by e^{+j omega dt/2} first);
  far_field 'power' is FDTD |pattern|^2 / eta0 'a.u.' (no 1/2) but FDFD / (2 eta0) 'W/(m rad)': compare the two
  only with that factor.
- Every scripting line feature takes `(axis, position, span)`, axis 'x' = line along x at y = position; only the
  internal `FluxMonitor / HarmonicFluxMonitor.axis` is the normal. `flux()` = Yee discrete Re(E x H*) (no 1/2)
  through the line cell's hi face, > 0 toward +x/+y. DFT phases use the actual sample time (hz H_z: t - dt/2). hz
  DFT / renderer E collocate to centres, hz `state()` omits staggered ex / ey; probes stay raw (H dt/2 older); ez
  renderer hx / hy stay raw face samples.
- Every physics/em renderer exposes e_abs, e_intensity and h_abs at centres (`renderer._MODULUS_FIELD_INFOS`).
  Renderers of centre-field dicts subclass `renderer.CentreFieldRenderer`; the FDTD pair dispatches in
  `_FDTDRendererBase._fdtd_field`.
- Snapshot solutions (root rule; contents per solver: its `solution_state` docstring): harmonic FDTD phasors, modes
  and resonances are selected, never copied; restores never call materials_changed or reassemble; FDFD keeps the
  adjoint gradient only when computed for that solve, else `eps_sensitivity` raises on the restored snapshot.
- Monitor results: dft_fields / far_field carry 'quantity' (plots, GUI read it); flux_spectrum 'unit' FDTD 'a.u.'
  (dt-weighted), FDFD 'W/m'; hz dft_fields with freq_index adds 'hz_intensity' (absent from loaded Results). One
  shared source waveform (coupling drives not counted) adds cw-equivalents (`results._shared_waveform_spectrum`).
- DFT spectra are dt-weighted (flux ~1e-40): form products in float64 on the host (float32 flushes to 0) and use
  relative thresholds only. Result maps take `viz.visualization.dft_map_edges`, never `viz.common.edges_from_centers`
  (downsampled maps).
- DFT decimation (`monitors.dft_sample_interval`): `EMPhysics.finalize` sets m only at n_step 0; coupling drives are
  not in the bound; monitors added after finalize keep m = 1. Any S(omega) compared with a monitor must use its grid
  (`waveform_spectrum(..., m)`).
- Convention (E7): exp(+j omega t); DFT accumulates exp(-i omega t) dt; lossy Im < 0; PML s = kappa + sigma/(alpha +
  j omega eps0). `susceptibility(omega)` is the continuous chi (FDTD differs at O((omega dt)^2)).
  `dispersion.complex_permittivity` takes one component; H_z callers face_average first.
- nk_fit (`material_fit`, host only): `fit_drude_lorentz.jacobian` is closed-form, so a change of a pole formula
  (dispersion) or of the residual scaling updates it. `DISPERSION_POLE_PARAMS['nk_fit']` has no pole class: every
  solver path sees the expanded drude / lorentz dicts.
- A1 operators (`operators.py`; layout, stretch, edges and Bloch wraps in its header; selftest
  em_operators_consistency): `KeptYeeOperators` / its Helmholtz plan share their index arrays with every op they
  issue (all omegas): never modify ops in place, eliminate_zeros or sort them (the plan keeps zero weights explicit so
  a wavelength scan refreshes cuDSS on one pattern). eps_xy couplings are added after the Helmholtz sum. On the device
  never multiply by a sparse diagonal (`diags(...) @ A`: a cuSPARSE SpGEMM whose workspace grows with the rows, 5e6
  rows asked 6.5 GB); scale stored entries (`_column_scaled`, `_kept_entries`, `_entry_rows`).
- Slab modes (`slab_modes`, A3; branch, normalisation, overlap and degenerate pairs: its header and docstrings): never
  force Im k_t <= 0 (header); compare asymmetric lines only after `align_phase`; complex lines above
  `DENSE_COMPLEX_MAX_CELLS` shift-invert near max Re q and can miss modes far above it.
- Mode ports (`mode_ports`; span ends: `PortLine`): PortLine serves FDTD and FDFD ports and never reads
  `physics.engine`. ModeSource is a one-sided TFSF line exact at f0 only: fraction_* report off-f0 leakage as
  reflection / transmission loss (help texts say so); it warns above bandwidth 0.6 (env cos is no Hilbert partner of
  env sin). After the first step ports re-solve on material drift (`ModePortSetup`, warnings into the append-only
  `runtime_warnings`). ModeMonitor fraction_* / phase_* need exactly one ModeSource (its sampled S(omega) on the
  monitor's grid).

## em_fdtd_ez / _hz on triangles / cuDSS: decision (nothing implemented)
- Decision: tensor Yee only, as em_fdtd_3d (em3d sub file "em_fdtd_3d on tetrahedra / cuDSS"); `grids =
  ('tensor',)` refuses a 'triangles' mesh. The 2D steppers hide no linear solve worth cuDSS (ports / TFSF are 1D
  tridiagonal lines). 2D DGTD only if a reference case misses its target with the cheaper local fixes (conformal
  PEC / pole weights, subpixel smoothing, graded meshes); 2D FETD rejected (a solve per step): conforming
  frequency-domain answers belong in em_fdfd on triangles.

## Nonlinear and gain media (C5; solvers/fdtd_nonlinear, solvers/fdtd_gain, definition_nonlinear)
- Domain features `nonlinear_medium` / `gain_medium` of em_fdtd_ez / _hz only, not material properties (allocated on
  their selection's bounding box); frequency_domain studies, Bloch k != 0 and the double stepper are refused. Kernel
  physics: the module headers and `_correction_kernel`; step order: `_update_e_set` (nonlinear `correct` stays the
  last correction: it subtracts the linear update). Cached box views of `_nonlinear_components()` assume those
  arrays are rewritten in place only.
- chi2 = 2 d_eff x per-cell poling sign, NOT scaled by n0; one scalar per component, no tensor / cross terms (crystal
  tensors: physics/nlfd). hz acts per in-plane component on its own faces (E_i^2, not |E|^2): exact for a field along
  one axis only.
- The TPA conductivity must be time-centred (`_correction_kernel`): a lagged (E^n)^2 gives a Kerr phase error that
  loss-only checks miss. A dispersion-free slab phase-matches the third harmonic, which with TPA cascades back as
  extra phase loss (analytic_nonlinear_tpa_kerr_slab).
- Loss: TPA + FCA work goes into q_accum (ez) / the face pole power totals (hz; a pole face next to a nonlinear-only
  cell splits its pole power with it). Gain work is not counted; both media are outside the build Courant check,
  sheet calibrations and port mode solves.
- A tight gain-threshold reference must use the leap-frog line (W = (2 / dt) sin(omega dt / 2), gamma sin(omega dt) /
  dt, sigma cos(omega dt / 2)): the continuous line is 7 % off at omega dt 0.22 (analytic_gain_threshold_cavity).
  Not done: lasing above threshold (the resonance fit drops growing poles), hz cross-component Kerr / TPA terms.

## Phase-change medium (C5; phase_change, definition_phase_change, coupling/definitions_phase_change)
- Feature `phase_change_medium` of all four in-plane types (`PhaseChangeMedium` in `solver.phase_change_media`)
  OWNS eps / sigma_e of its cells: each update rewrites every permittivity array and sigma_e, then one
  `materials_changed`.
- Driven only by coupling `heat_em_phase_change` (its transform calls `update_phase_change(em, T, heat.time)` and
  returns None: a slot array cannot carry the heat time). It declares '' writes of every permittivity slot, so the
  writer rule refuses thermo-optic / Pockels / field_transfer / state-dependent eps on that physics. A repeated time
  (end-of-run flush) is a no-op.
- GST-225 numbers live once in `phase_change.GST_225` (library materials read it too). Kinetics: `Kinetics`
  docstring (the rate cap above T_fast keeps a melt-quench amorphous).

## Harmonic FDTD (em_fdtd_* in a frequency_domain study; harmonic_fdtd.py)
- Flow: `model.study._run_frequency_domain` docstring. `attach_harmonic` before the first step (after it: ValueError) appends a
  `HarmonicAccumulator` (DFTMonitor, complex64, own `dft_sample_interval`) to `monitors`; all sources share one
  gaussian / ricker waveform (`_shared_pulse`, validation `definition_harmonic_fdtd`).
- `normalize` divides once by `results._shared_waveform_spectrum` (S over steps 1..n_step), so phasors follow the FDFD
  convention (phase 0 = DFT / S); changing the accumulator's sample grid breaks that. Feature monitors keep
  unnormalised spectra. Renderer fields carry `requires_study` (`schema.FREQUENCY_DOMAIN_STUDY`, also the BOR FDTD's);
  `harmonic_primary` is the default stored field; `absorbed_power_harmonic` takes the continuous chi at the centres
  (hz per component, unlike the FDFD face mean). A calibrated sheet source is exact at f0 only (elsewhere its
  |eta / cos| mismatch).
- Self-tests em_harmonic_fdtd_vs_fdfd (pins both source signs), em_hz_harmonic_fdtd_slab (only the last wavelength
  stays live: read stored snapshots); template cylinder_harmonic_fdtd. Result `harmonic_pulse` -> summary.json (api).

## FDFD (em_fdfd_ez / em_fdfd_hz)
- `FDFDPhysics` (HarmonicEMShell) shares EMPhysics' attributes (domain, sources, monitors, dispersive_regions,
  pec_cells, permittivity_arrays, ...), so PortLine, couplings and the Visualizer treat both alike.
- Backends: every array on `self.xp` (cupy + cupyx CSR or numpy + scipy CSR), masks included; eigenfrequency gets host
  copies; sources keep host arrays until the first solve (`fdfd_sources._on_backend`). Shared with FDTD, so they must
  stay backend-neutral: `materials.register_em_arrays`, PortLine `_host_rows`, PhaseChangeMedium, `_phasor_part(xp)`,
  adjoint.py, DesignRegion. Self-tests em_fdfd_cpu_gpu, em_fdfd_adjoint_cpu_gpu.
- Conventions: harmonic.py header (exp(+j omega t); every FDTD source ADDS +c J s(t), so J_F = -J_std and the FDFD
  keeps the FDTD amplitude meanings: phase 0 = FDTD DFT / S(omega)); u at cell centres on the A1 flat index, assembly /
  solve / `solution` complex128; FDTD sin carrier = phase -90, `phase` [deg] 0 = cos. A line current gives E_z =
  +(omega mu0 I / 4) H0^(2)(k r), H_z = +(omega eps0 K_m / 4) H0^(2)(k r) (em_fdfd_line_current_hankel).
- Conformal PEC (S7.6): `ConformalFDFD` built once on the host, reset by any new PEC region; pec_cells = the staircase
  (ports, sources, TFSF / eps_xy checks read it). ez: face weights into the metal x 1 / max(t, CONDITIONING_FLOOR).
  hz (Dey-Mittra): `face_fractions` / `cell_area_fractions`, area-scaled operator (complex symmetric, so reciprocal),
  while `weight_hi_*_faces` stay physical for face_fields; face_current_rhs, q_abs, adjoint dA and eigen mass take
  f / a. hz `system.pec_cells` = fixed cells only (cut cells with a metal centre stay unknowns): rhs zeroing, wall-fill
  re-zero, eigen pinning and adjoint fold read `system.pec_cells`, never `physics.pec_cells`. Not conformal: eps_xy
  couplings, slab-mode ports, sheet calibrations. Self-tests em_fdfd_conformal_disk_eigen,
  em_fdfd_conformal_reciprocity.
- hz off-diagonal eps_xy (harmonic.py header; the only EM type with it): material `eps_xy` + private float32
  `_coupled_eps_xy` of slot `eps_xy_update` (OFF_DIAGONAL_SLOT), real, zeroed in PEC cells and the outer two cell rows
  of every wall / mirror face (`pin_image_cells` folds a pinned cell into ONE neighbour: a 9-point stencil there would
  be wrong). Faces hold the effective 1/eta_ii = eps_ii - eps_xy^2 / eps_jj; cell_eps_xx / _yy stay raw for q_abs.
  `off_diagonal_face_couplings` drops zero-weight / never-updated faces (`active_coupling_faces`), keeping the operator
  complex symmetric. Not seen by: slab-mode ports, sheet calibrations, TFSF background (`_off_diagonal_line_warnings`
  at finalize only), em_fdtd_hz / em_modes / BOR (couplings drop the xy row with a crystal warning), the NLFD cascade
  adjoint (`_check_differentiable` refuses). Renderer `eps_xy` parity 'even'. Self-tests em_fdfd_hz_rotated_slab,
  em_fdfd_hz_anisotropic_reciprocity.
- hz gyrotropic gyration_z (harmonic.py header; material only, no slot, no renderer field): `assembly_gyration` (same
  zeroing as eps_xy), `HarmonicSystem.cell_gyration`; the two face couplings take their own eta_xy / eta_yx
  (`off_diagonal_face_couplings(eta_yx=)`), so A(-g) = D^-1 A(g)^T D: non-reciprocal. Code assuming D A symmetric must
  check `cell_gyration` (`_symmetrizer` returns None with it: cuDSS / MUMPS / PARDISO factor a general LU, eigen
  pencils too; the adjoint is the true transposed solve). Lossless g keeps the discrete Poynting balance exact, q_abs
  unchanged (diagonal loss). The X maps take plain cell means across an interface (no Kottke-like weighting), so a
  guide's non-reciprocal phase shift converges at first order (~6 % at 10 nm cells, Si / Ce:YIG; omega itself second
  order): checks Richardson-extrapolate. Not seen by what ignores eps_xy (above); refusals: `_gyration_issues`.
  Self-tests em_fdfd_hz_gyrotropic_reciprocity (H(B; A, g) = H(A; B, -g), flux ring balance; flux lines measure on
  their cells' hi faces, so a closed ring needs lines at cell centres spanning the cells between the other pair's
  faces), em_fdfd_hz_gyrotropic_nrps (Bloch eigen vs `selftests_gyrotropy.exact_slab_beta`),
  em_fdfd_hz_gyrotropic_adjoint_gradient.
- Dirty flags: `_system_dirty` (materials_changed of a read array, slots, PEC, poles, new omega) reassembles and
  refactorises; `_solution_dirty` (that or a new source) re-solves. A recyclable factor at an unchanged omega (Picard
  step) preconditions host GMRES instead (`_solve_linear`); A1 operators cached per omega (`_yee_ops`).
- Linear solver: 'linear' role (`SparseDirectSolver`, `FDFDPhysics(linear_solver=)`), 'auto' resolved at finalize;
  eigenfrequency uses the 'eigen' role (also optomechanics). Size guards: a cusolver_qr workspace overrun bypasses the
  pool budget and can freeze Windows / crash the driver instead of failing, so sizes are refused before solving:
  `size_issue` is the one rule for validation and solver, `check_solver_size` (finalize, every factorisation) also
  refuses a QR workspace over half the free device memory. A solve cannot be cancelled (Progress before each solve,
  cancel acts between). The study logs `solve_summary()` (not on the ABC) after the first solve. The HarmonicSystem
  keeps A on the device on both back ends (TFSF / mode rhs need it).
- cuDSS 'linear' (cudss / cudss_mixed, GPU only): tensor meshes factorise D A (`_symmetrizer` =
  `stretched_cell_areas`; a failed symmetry check, e.g. eps_xy, gives a general LU). Not recyclable: a system change
  keeps the factor as `_cudss_previous` (for its plan; its device memory stays during the next assembly) and
  refactorises it in place (`previous=`). An adjoint solve on a general LU factorises the transpose
  (`_transposed_solve`: one extra factor outside the pool, not estimated). Triangles: `factorize_host(symmetric=True,
  row_scale=system.equilibration, previous=)`. Automatic = cuDSS on the gpu with nvmath (B1a, solvers sub file); an
  'auto' runtime that cuDSS refuses falls back to SuperLU and fdfd_2d / fdfd_2d_fem / fdfd_bor / fdfd_conical free
  `_cudss_previous`. Deferred checks: fdfd_2d driven + transposed factorize and host_fdfd `_factorize_operator` refresh
  with `defer_check=True`, so CudssAccuracyError comes from the solve: fdfd_2d `_solve_linear` and host_fdfd
  `_checked_solve_host` (BOR / conical driven solves, BOR order sums) catch it, ask `host_fdfd.fell_back_to_superlu`
  and refactorise on SuperLU. A new refresh path passes defer_check only with that catch; fdfd_2d_fem and every other
  `factorize_host(previous=)` caller has none, so its refreshes stay probed. BOR's C^T W C - k0^2 M also passes
  `symmetric_by_construction` (canonicalised in place + `symmetric_known`, no asymmetry check), conical (D A) not.
  Self-tests comparing with "SuperLU" (em_fdfd_cudss, selftests_fdfd line_block, em_bor_cudss_driven,
  em_fdfd_conical_cudss) must give the reference an explicit scipy_splu node, never Automatic.
- MUMPS / PARDISO (solvers sub file "host_direct"; 'linear' back_end and 'eigen' factorization mumps / pardiso
  (+ _mixed), opt-in, either backend, every EM type): code tests the tuples / predicates, never the names
  (`HOST_DIRECT_BACK_ENDS`, `sparse_direct.SYMMETRIC_OPTION_BACK_ENDS` / `MIXED_BACK_ENDS`,
  `arpack_shift_invert.LDLT_FACTORIZATIONS` / `MIXED_FACTORIZATIONS`, `is_host_direct_factor`, `factor_label`; help
  texts `definition_common.HOST_DIRECT_NAMES`). They get the cuDSS factor options (fdfd_2d: symmetric + `_symmetrizer`
  D; host_fdfd `_linear_symmetric_options`; fdfd_2d_fem: symmetric + equilibration) but are recyclable and transpose
  natively like SuperLU: no `_transposed_solve`, GMRES recycling at an unchanged omega, no `_cudss_previous` on tensor
  meshes (fdfd_2d_fem keeps one as previous: analysis reused on the same pattern). Size rules: SuperLU's everywhere
  (`size_issue` / `fem_*_size_issue` / `size_issue_for_unknowns` take the host names), plus the library's own memory
  check after the analysis. Texts: solve_summary / direct_text / `HOST_DIRECT_KINDS` (em_modes factor_kind prefixes)
  name the description and the library's factor bytes; an own eigen host factor logs `_host_direct_factor_text`.
  Estimates: `memory_estimates.host_factor_items` at every place a SuperLU factor item was booked (2D FDFD, triangles,
  BOR, conical, em_modes, em3d, eme); the hz eps_xy extra fill stays at SuperLU size. Self-test em_fdfd_host_direct.
- line_block 'linear' node (opt-in; solvers sub file line_block): one host operator copy (`set_operator(owner=)`, block
  axis = the periodic one) and one kept factor, no GMRES recycling. An adjoint solve runs on the kept factor as
  x = D solve(g / D) when D A is complex symmetric (`_adjoint_symmetrizer`, measured once per system); a rejected solve
  marks the system `_REJECTED` (line_block_fallbacks) and, like a non-symmetric adjoint solve, goes to the private
  `_reference_solver` (role-less 'auto' = host SuperLU). Wavelength batches (`plan_harmonic_frequencies`, `model.solver_hooks.HarmonicFrequencyPlanner`):
  each wavelength's own `_assemble` must give the identical operator, guarded by `_material_serial` (bumped by every
  `_mark_system_dirty` but `omega_only`), so any change of the assembled system must go through `_mark_system_dirty`.
  Self-tests em_fdfd_line_block, _line_block_wavelengths, em_fdfd_eigenfrequency_line_block.
- Bloch k != 0 (symmetry 2.2): only the wrap entries change. `lower_wrap_factors` = e^{+j k L} per axis: every
  face-to-centre mean across the wrap takes it (`face_mean_at_centres(wrap_factor=)`; q_abs needs none; NF2FF boxes
  never wrap), as do a dipole on the last face and a mode source's upstream cell. Sheets carry e^{-j k.r}
  (`bloch_line_phase`); TFSF with Bloch ky = the oblique discrete wave. Port lines spanning a whole Bloch period are
  refused (`_check_bloch_line`, `bloch_period_line_issues`). nlfd takes each channel's factor in its C / C*.
- Polar (r, phi) grids (symmetry 2.16 / 2.18; scripts only, `coordinate_grids` lists 'polar'; PhysicsTypes stay
  Cartesian until the model path exists): the metric lives only in the A1 ops (`polar_radius_min=`, complex r~ in
  the x_hi CPML); `FDFDPhysics.metric` gives point-source areas, arc flux weights and eigen PML shares. Faces:
  `_check_polar_boundaries`; bloch_ky = azimuthal order m [1/rad]. Every Cartesian-only feature calls
  `_require_cartesian` (grep it; eps_xy / gyration_z raise in `_assemble`): a new one that assumes Cartesian cells
  must call it too. Closed forms are semi-discrete: order nu = (2 / dphi) sin(m dphi / 2). Self-tests selftests_polar.
- Faces: cpml, periodic, pec / pmc walls and mirrors with the FDTD rows (`boundaries.WALL_SIGNS`): outer cells pinned
  via `pin_image_cells` (rhs zeroed, filled after the solve, PEC re-zeroed); a +1 row also zeroes the inner face
  weight (`PINNED_FACE_TYPES`), so an hz face current there drops (as the FDTD re-pin). Placement checks of the model
  path and of PortLine derive their pinned rows from the same tables.
- `face_fields()` (cached per solve): hi faces; hz = discrete curl PLUS source `face_currents` (dipole, sheet), so E is
  exact on a source face. TFSF and mode phasors only add to the rhs, so the field jumps across their interface: no
  measuring face (monitor line cell hi face, probe hx / hy, NF2FF samples) may sit on a mode source's interface face
  (hi face of line cell - 1 for 'positive', of the line cell for 'negative') or on / across a TFSF outline; a far-field
  box around a TFSF box needs >= 2 cells on the lo sides and its x1 / y1 edges past it (FDTD: 1 cell). Validation
  warns (`_total_field_regions`, `_mixed_region_texts`); hz q_abs beside such a face is wrong in lossy cells, unchecked.
- TFSF (`PlaneWavePhasor`): b += (1 - Q) A (Q u_inc) - Q A ((1 - Q) u_inc) through `system.operator`; exact whenever
  u_inc solves the background rows (medium of outline cell (x0 - 1, y0), read per solve), so only with that one
  medium around the box. Bloch ky needs uniform y cells; uniform columns use the discrete-dispersion wave, graded ones
  a march on the operator's x line (kept off uniform lines: round-off growth in lossy media). Phase referred to the
  centre of column x0 (FDTD TFSF phases differ); hz amplitude = incident E_y. `full_period_y`: no y edges. Selftest
  em_fdfd_tfsf_graded_leakage.
- Mode ports (`fdfd_ports`): unlike the drift-gated FDTD ports, modes are re-solved at finalize and after every
  reassembly (`PortLine.solve_modes` returns the last SlabModes when omega, counts, spacing, host rows and pole objects
  are unchanged). Source = the TFSF formula on the line cells and their upstream neighbours; `add_to_rhs` refuses a
  mode solved at another omega. The source re-selects `mode_number` per solve; the monitor tracks by overlap +
  `align_phase`. power_* = |a|^2 W/m (FDTD 2 |a|^2). Only finalize's port warnings are reported.
- `create` (FDFD and em_modes) builds at `first_solved_wavelength`, so ports select modes there; auto mesh at
  `shortest_solved_wavelength` with the static index; the resolution warning checks `all_solved_wavelengths` with |n|
  of the complex eps(omega) plus >= 2 cells per skin depth, so metals the static index (Drude eps_inf) misses are
  flagged.
- Monitors record one value per omega (repeat overwrites) on the device; results ascending wavelengths; a monitor
  added after a solve records it at once. HarmonicProbe: ez hx / hy raw on the +y / +x face, hz ex / ey collocated.
  Flux = 1/2 Re(E x H*) [W/m], discretely conserved; modes normalised to that discrete flux.
- q_abs: ez 1/2 omega eps0 (-Im eps) |E_z|^2; hz per component the face-mean |E|^2 x -Im eps_ii; magnetic loss not
  counted. Renderer primary `<field>_re` (the stationary fixed point compares it).
- Eigenfrequency (A6, `eigenfrequency.py`, host only): `solve_resonances` is dimension-free and shared (2D FDFD, BOR,
  conical, em_fdfd_3d, FEM); spectral maps, the own-factor shift, PML / region filters, widenings, device / host
  Krylov-Schur and the mixed / E2 Rayleigh polish are in its module header. 2D factor: host SuperLU
  (`_eigen_size_warnings`), a line_block node's kept factor (RejectedLineBlockSolve -> SuperLU), or the eigen role's
  cuDSS / host direct LDL^T of D A (`_eigen_symmetrizer`); `rayleigh_symmetrizer` = `_adjoint_symmetrizer()`. Cost ~
  eigenpairs^2, so ask only for resonances that exist near the target; estimators book
  `memory_estimates.resonance_search_items` and `eigenfrequency.resonance_filter_time_item` (first pass only). Per-
  column measures go through `_ResonanceFilter` (one sparse product per operator and block), never per-column host
  csr_matvec loops (tetrahedral Mie: 2.4 s of 8 s). Fields are 1 at the first argmax |u|: mirror-equal peaks leave the
  sign to round-off. `select_resonance(k)` sets `selected_resonance` (hz `face_fields` skips source currents);
  `_solution_dirty` stays True so a later step() re-solves the driven problem. Region (P2): on tensor grids
  `ctx.domain_mask`, so an object touching the window edge counts the PML cells beyond it (clamped centres; example
  dbr_cavity_q stops its gaps one cell short). Self-tests em_fdfd_eigenfrequency_pec_cavity,
  em_resonance_region_fraction, em_fdfd_resonance_region, em_eigen_sparse_mass; template disk_resonator_eigenmodes.
- Adjoint (C7, `adjoint.py`, derivation in its header): objectives subclass the flux / mode monitors; J = sum weight
  * value (flux unfolded by `mirror_factor`, mode power raw). `eps_sensitivity()` is lazy, cached per (n_solves,
  objectives), zeros while a resonance is shown; `solve_adjoint` refuses a system reassembled since the last driven
  solve or a shown resonance. The adjoint source is folded through the transpose of the wall fill; hz differentiates V
  incl. the `pin_image_cells` fold, plus (as nlfd) the objectives' face fields (`face_cotangents` = the objective face
  parts) and the rhs D_e V J of the sources' summed `face_currents`. Held fixed: source calibrations (the J values),
  TFSF / port incident fields, port modes; sigma_e, poles, mu_r not differentiated. `mode_amplitude_weights` is the
  one map behind mode_amplitudes and the mode objective. eps_xy: `EpsSensitivity.off_diagonal_density`
  ('eps_xy_sensitivity', hz only, parity 'even' although its true image is odd); gyration_z: one cotangent per map
  (h_xy onto the hi-y faces, h_yx onto the hi-x faces), chain rule on q = eps_xy^2 + g^2 with g held fixed (no dJ /
  dg). physics/nlfd/adjoint.py chains `objective_cotangents` -> `adjoint_source` -> `solve_adjoint` ->
  `permittivity_gradients` -> `sensitivity_density`, so a signature change must reach it. Self-tests
  em_fdfd_adjoint_gradient, em_fdfd_hz_eps_xy_adjoint_gradient.
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
  False)`); sigma(omega) cached (graphene Kubo: scipy quad). Self-tests em_graphene_kubo_limits,
  em_fdfd_conducting_sheet.
- `set_extra_permittivity(values, box)` (docstring): `_solve(record_monitors=False)` defers q_abs and the float32 state
  arrays (`_refresh_pending_state`); new readers of `_q_abs` / `_field_re` / `_field_im` refresh first.
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
  `solvers/fem_em_base.TriangleEMPhysics` (contract in its class comment); FDFDPhysics and FEMFDFDPhysics share
  `solvers/harmonic_shell.HarmonicEMShell`. A new triangle EM physics, or a member both 2D FDFD physics need, goes
  there, never into a copy.
- One type, two solvers: create builds FEMFDFDPhysics on a 'triangles' mesh (walls from
  `fdfd_domain_requirements(settings)`: a TriangleDomain carries no BoundaryConfig). A feature running there lists
  'triangles' in its `grids` and validates per mesh kind (`definition_fdfd._per_mesh_kind`); what triangles refuse is
  `_triangle_issues`.
- Formulation, walls, PML, signs: `fem_fdfd` module header. The operator is -(Yee operator) of the same continuum
  problem, so loads keep the Yee source amplitudes. Driven solves refactorise on every system change (no GMRES
  recycling as on the Yee grid; cuDSS / MUMPS / PARDISO keep their analysis in `_cudss_previous`).
- Self-tests: selftests_fdfd_fem.

## Cross-section modes (cross_section_modes)
- Host only. Formulation, Yee point layout, walls / mirrors, Bloch phases, bends and conformal PEC: module header. The
  wall plane is half a cell off the FDTD / FDFD ez pec and hz pmc walls (a cavity between them differs by one cell).
- One writer of the E-point eps: `_point_materials` (ghost fill, face means, subpixel deltas, bend factors), used by
  the assembly and by `mode_point_materials` / `ModePhysics.point_materials()`; the em_eme_3d adjoint transposes it
  (physics/eme sub file). Bend factors are per component: a consumer weighting E_z uses the z factors.
- Solve rules (shift, eigenpair counts, PML drop and ranking, tolerance floors, Arnoldi basis, the real float64 path):
  the constant comments and `solve_prepared_modes`. The estimators cannot see the materials and follow them through
  `mode_eigen_count` (wide), `mode_default_basis(..., cpml_cross_section(settings))` and
  `memory_estimates.real_cross_section_operator`: a change to the solve's count, basis or real test changes those too.
- Degenerate pairs (fibre HE11) come back as arbitrary non-orthogonal combinations; separate them with mirrors.
  `mode_group_indices` re-solves with the modes as ARPACK v0 and tracks by overlap: pass the solve's n_eff_guess.
- Self-tests: `CROSS_SECTION_CPU_SELFTESTS` (selftests_cross_section: a TE slab matches slab_modes 'ez' to round-off,
  TM matches 'hz' only to discretisation order, dual point set), selftests_bend, model/selftests_conformal_cuts.

## Conical / out-of-plane k_z (em_fdfd_conical; conical, solvers/fdfd_conical, renderer_conical, definition_conical)
- Host only. Formulation, symmetrizer D, source and flux conventions (J_F is minus the textbook current): `conical`
  module header and docstrings; factor reuse (GMRES recycling, cuDSS refactorisation): the ConicalPhysics class comment.
- Built on the cross_section_modes layout and `assemble_cross_section` parts (walls, mirrors, Bloch phases, CPML, point
  eps, lower-cell mu): a change there changes both types. At k_z = 0 the E_z block is the em_fdfd_ez operator but with
  the cross-section wall planes.
- A cuDSS factor skips the host SuperLU size rule (`_refresh_system` / `_solve` of ConicalPhysics, definition
  `_solver_issues`, `solve_conical_resonances`): a change of that rule touches every caller of `conical.size_issue_for`.
- Verdict: 3 unknowns per cell and host SuperLU fill limit it near ~1e6 unknowns, so cuDSS pays for large
  cross-sections and long k_z sweeps; cudss_mixed only while its single_residual stays below 1e-3 (above it use cudss).
- Self-tests: selftests_conical (CPU numerics, model runs), selftests_conical_fem (also the Yee symmetrizer and cuDSS
  checks); walled guide = example em/analytic_conical_walled_guide.

## em_fdfd_conical on triangles (fem_conical, solvers/fdfd_conical_fem)
- Formulation and the reused fem_modes spaces, walls, PML ring and Faraday recovery: `fem_conical` module header. It is
  exactly the em_modes Lee-Sun-Cendes pencil at fixed k_z (pinned by em_conical_fem_modes_equivalence): a change of
  fem_modes' `_coefficients`, walls or layout changes both types.
- Eigenfrequency only (`definition_conical._triangle_issues`). Driven triangle solves (conical grating incidence also
  needs periodic meshes) should share fem_fdfd's point loads and probes (`point_load`, `_basis_at`), not copy them.
- Self-tests: selftests_conical_fem.

## em_modes model layer (solvers/modes_2d, definition_modes)
- ModePhysics has the FDFDPhysics attribute set (class comment; polarization 'modes', no sources). Every solve re-reads
  the MaterialGrid; `_material_serial` only stales a kept line-block batch, so a new way to change the solved system
  bumps it (`__init__` comment). `select_mode` writes `mode_fields` in place: ModeRenderer and `state()` hold those
  arrays. Line-block 'shifted' batches and the detached solves (shg_overlap, bend_overlap): the class comment.
- A cudss eigen node skips the host SuperLU size rule in `_solve_at` / `solve_detached` (check_size), `finalize` and
  `_solver_issues`: change them together. Not done: a symmetric Yee form (the mixed [E_t; E_z] pencil), which would
  allow LDL^T and the _mixed factorizations on tensor meshes (`_solver_issues` says why they are refused).
- Bloch k != 0: every reader of `centre_fields` passes `ModePhysics.wrap_factors` or refuses k != 0 (shg_overlap,
  heat thermal_noise `mode_pair_weighting`, optomechanics `_cross_section_mode`); a new consumer does the same.

## em_modes on triangles (fem_modes, solvers/modes_fem)
- `definition_modes._create` builds FEMModePhysics on a 'triangles' mesh (walls from `fdfd_domain_requirements`); what
  triangles refuse: `_triangle_issues`, `_gyration_issues` and the tensor-only features' `grids`. Formulation, walls,
  PML ring, factor and Rayleigh polish: `fem_modes` module header. FEMMode stores physical [E_t; E_z]; `_eigenvector`
  gives the raw ARPACK vector.
- ModePhysics and FEMModePhysics share `modes_2d._ModeStudyMixin`, and FEMMode duck-types the `mode_overlap`
  attributes, so mode_overlap and the table / dispersion monitors serve both: a change there keeps both solvers
  working. n_group: `fem_group_indices` (Hellmann-Feynman, no tracking) here, the tracked `mode_group_indices` on Yee.
- The pencil is complex symmetric (unconjugated tests, diagonal stretch), as are fem_conical's and fem_bor's:
  conjugated tests or full anisotropic tensors with a non-diagonal stretch lose the LDL^T and the _mixed polish.
  cuDSS in 2D buys device memory / host relief at the largest P2 meshes, not flops.
- The size and time figures (`estimated_unknowns`, FEM_LU_FILL_FACTOR, FEM_MODE_ASSEMBLY_*) are uncalibrated; the
  triangle em_fdfd_bor / em_fdfd_conical estimates reuse them.
- bend_radius on triangles (refused) would need nu_t / nu_zz from the mu tensors, s continued into an x PML ring and
  the overlaps' metric.
- Self-tests: selftests_modes_fem.

## Body of revolution (em_fdfd_bor; body_of_revolution, bor_ports, solvers/fdfd_bor, renderer_bor, definition_bor)
- Formulation (unknowns, axis rules, walls, PML, orders, materials, sources, flux, far field, driven order sum):
  `body_of_revolution` module header; solve levels, factor reuse, wavelength batches, line block and cuDSS: the
  BORPhysics class comment; ports: `bor_ports` header. BORPhysics and ConicalPhysics share
  `solvers/host_fdfd.HostFDFDBase` (a member both need goes there); `bor_boundary_config`, `bor_flux_line` and
  `bor_far_field_box` serve BOR FDFD and FDTD alike.
- em_fdtd_bor steps the same operator: `BORAssembly(..., omega=None)` is its real metric, and `curl_parts`,
  `e_point_means` / `a_point_means` and `yee_centres` serve both types, so a change to the point or axis rules changes
  both (example analytic_bor_transverse_dipole_m1 pins the axis rule).
- `_material_serial` keys the line-block wavelength batch: every `_mark_assembly_dirty` but `omega_only` bumps it, so a
  change of the assembled system goes through it.
- The host SuperLU size rule binds only an own SuperLU / MUMPS / PARDISO factor: a cuDSS factor or a line_block node
  skips it in `_solve`, `HostFDFDBase.finalize`, `_solver_issues` and `solve_bor_resonances` (passed factor); a rejected
  line-block solve checks it in `_factorize_host` before the SuperLU fallback. Change them together.
- Driven order sum: a new source type builds its own b(-m) (`term_rhs(system, order=)`, never a mirrored b(m)), a new
  monitor type needs `record_order` and `record`; mode ports are refused.
- 2D fill is mild (SuperLU copes to ~1e6 unknowns): cuDSS (unmeasured) buys host relief at the largest many-order
  grids. Not yet: the 2.16 metric layer, a measured line-block vs SuperLU crossover, line-block port mode solves.
- Self-tests: selftests_bor, selftests_bor_orders, selftests_bor_backend; examples em/analytic_bor_*. SuperLU-only
  analytic model runs use the cpu backend, line_block runs the default (the cupy line block is its own
  implementation). Backend comparisons scale a component by a common peak (er / ez by the |E| peak): a symmetry-zero
  component is round-off and differs between numpy and cupy.

## em_fdfd_bor on triangles (fem_bor, solvers/fdfd_bor_fem)
- Eigenfrequency only: `definition_bor._create` builds FEMBORPhysics on a 'triangles' mesh (walls from
  `_domain_requirements(settings)`: the axis x_lo as pmc, no ring there); `_triangle_issues` refuses every study that
  does not eigen-solve it, the feature `grids` every source / monitor / port. Formulation, axis, walls, PML: `fem_bor`
  module header.
- Compare with the Yee path far from the PML only (its cpml ends in a pmc edge, the ring in a pec side). Linear
  elements lock w / rot e in the axis layer (m = 0, |m| = 1 converge slowly): keep quadratic. Axis display values are
  approximate (`AXIS_NODE_SHRINK`).
- Wanted next: driven solves (FEM flux / far-field / port equivalents) on a 'linear' cuDSS back end for wavelength
  sweeps.
- Self-tests: selftests_bor_fem (not yet run; tolerances first estimates).

## Body-of-revolution FDTD (em_fdtd_bor; fdtd_bor, fdtd_bor_ports, solvers/fdtd_bor, definition_fdtd_bor)
- Yee only (grids ('tensor',)): the leap-frog has no linear solve; conforming answers belong in em_fdfd_bor on triangles.
- Scheme (layouts, phasor convention, time step, CPML, poles, driven order sum, lockstep batch): `fdtd_bor` module
  header; ports: `fdtd_bor_ports` header. It steps the em_fdfd_bor curl (`stretched_assembly(...).curl_parts(m)`), so
  DFT phasors solve em_fdfd_bor at omega_tilde exactly: a change of a sample or injection time, or of the point / axis
  rules of body_of_revolution, changes both types (selftests em_fdtd_bor_*, example analytic_bor_transverse_dipole_m1).
- r_hi CPML kappa enters the assembly widths, shared by every stepper of an order sum: flux / source helpers see
  rho = r inside the window only.
- Its kernels are `core.kernels` twins named 'em.bor_fdtd_*' (defined in fdtd_bor with their numpy twins).
- A new per-step term goes into `_LockstepBatch` (flat index = member position x layout size + own index), never into
  a per-member loop; host coefficient changes reach the device through `upload` / `injections_changed`.
- Physics, monitors, BORHarmonic and the renderer use only the engine interface of the `BOROrderSum` docstring
  (BORTimeStepper has the same), never x / h of one stepper.
- Order-sum references: the phi = 0 pattern adds the order amplitudes coherently, so an order cut errs by amplitude
  terms, not power shares, and a phi current needs one order more (`selftests_fdtd_bor_orders._MODEL_ORDERS`).

## Subpixel smoothing (subpixel_smoothing.py, geometry in model/subpixel_geometry.py)
- Switch, honouring types and the no-MaterialGrid rule: root "Physics definitions"; formula, locations and kinds:
  `subpixel_smoothing` / `core.interface_geometry` / `model.subpixel_geometry` headers. The solvers own their
  SubpixelSmoothing (no ctx reference: Session.close frees it).
- Consumers change together, or port modes / FDTD / FDFD disagree: FDTD `correct_fdtd` in `_rebuild_e_coefficients`
  (before the PEC fold; hz also eps_x / eps_y, which the nonlinear / gain media read), FDFD `_assemble` (`add_deltas`
  after the complex face means), `PortLine._host_rows` (`line_delta` rows), em_modes `point_deltas` (section
  "Cross-section modes"). A new kernel or consumer of face / cell eps on these types applies the same averaging (or
  reads eps_x / eps_y, the FDFD system, em_modes `point_materials`) and gets a check in `subpixel_smoothing_cpu_gpu`
  (model/selftests_subpixel). Not averaged beyond what the help text names: the FDTD sheet / point calibrations, the
  E_z nonlinear n0^2, the Courant check, the q_accum split.

## Definitions and self-tests
- Analytic FDFD / BOR model cases run on the cpu (`selftests_fdfd.cpu_specs` docstring); the `*_cpu_gpu` parity
  cases (`selftests_fdfd_backend._groups`, physics named 'em') are the FDFD types' gpu runs, so a new feature needs no
  gpu model of its own. Not covered on the gpu: conformal PEC in a driven run.
- Memory estimators (`memory_estimates.py`; em3d and eme reuse its helpers and phase constants): per-entry items
  (copies, index maps, refresh, fill) scale with entries per row counted from the operator's stencil
  (`_bor_entries_per_row`), never measured. Phases: a kept driven cuDSS factor's pool copies are run items
  (`kept_cudss_items`), its solve vectors CUDSS_SOLVE_PHASE; REFACTOR_PHASE only where the engine refactorises a kept
  factor (`previous=`) AND `later_factorisations` says the study factorises again (mode / eigen searches plan a fresh
  factor per solve); a single solve's ASSEMBLY_PHASE is booked net of the solution and factor it precedes
  (`phase_before_items`); ADJOINT_PHASE per adjoint_objective (labelled per feature: the largest counts once); every
  cuDSS factor is an external item from `cudss_factor_items` (an FDFD LU with an adjoint objective adds the
  transpose's).
- Material Params of the 2D EM types come from `definition_common.em_material_params` (a type supplies only help
  texts); GUI tips every EM type shares live in definition_common (`_MATERIAL_TIPS` and the `*_TIP` constants),
  type-specific tips and summaries in definition_fdtd / definition_fdfd; texts and Params that differ only by solver
  ('fdtd' | 'fdfd') are built once there (e.g. `sheet_amplitude_param`).
- FDFD features reuse the FDTD validate / result callbacks with `time_domain=False`. FDTD and FDFD register the same
  `apply_*` callbacks, which tell them apart by Param names (`source_time_kwargs`, `_monitor_wavelength_kwargs`): never
  give an FDFD source a 'waveform' Param or an FDFD monitor spectral params; em_heat_joule likewise finds pulsed
  sources by a 'waveform' other than 'cw'.
- A horizontal line gives line_source / beam / mode_source / mode_monitor axis 'x' (along it) and flux_monitor axis
  'y' (its normal). Placement checks look cells up on the MeshEstimate as the runtime does
  (`definition_feature_validation`); every PortLine refusal must be a validation error first.
- em_operators_consistency (gpu) steps both engines against the A1 operators: never a Python copy of the kernels.

## Effective index 2.5D (effective_index, definition_effective_index)
- Feature `effective_index` writes n_eff^2 in its apply, before finalize, so ports, the Courant check and the heat n0
  see it; a state-dependent permittivity overwrites it (dynamic updates run after features). Method, polarisation
  pairing and the 'tm' face weight: `effective_index` header and `_stack_line`.
