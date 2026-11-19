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
  cell-mask helpers there (`_check_cell_array`, `record_dispersive_region`, `add_pec_cells`) also serve FDFD, em_modes,
  BOR and EME. EM solvers take cell geometry and per-polarisation vocabulary from `placement` and
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
- Flow in the root file (frequency_domain): `attach_harmonic` before the first step (after it: ValueError) appends a
  `HarmonicAccumulator` (DFTMonitor, complex64, own `dft_sample_interval`) to `monitors`; all sources share one
  gaussian / ricker waveform (`_shared_pulse`, validation `definition_harmonic_fdtd`).
- `normalize` divides once by `results._shared_waveform_spectrum` (S over steps 1..n_step), so phasors follow the FDFD
  convention (phase 0 = DFT / S); changing the accumulator's sample grid breaks that. Feature monitors keep
  unnormalised spectra. Renderer fields carry `requires_study` (renderer `_HARMONIC_STUDY`, also the BOR FDTD's);
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
  `_reference_solver` (role-less 'auto' = host SuperLU). Wavelength batches (`plan_harmonic_frequencies`, root file):
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
- Leaky bend modes want pml_kappa_max 1 (the default 15 reflects grazing radiation).
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
- Non-reciprocal phase shift (`cross_section_modes.nonreciprocal_shift`, `ModePhysics._nonreciprocal_shifts` after
  every solve -> `delta_betas`, scalar outputs delta_beta_<k>, mode_table / mode_dispersion column delta_beta): first
  order in the material gyration_x / gyration_y (they never enter the operator; FEMModePhysics gives 0, validation
  refuses them on triangles, with a bend and with a mirror face normal to the other transverse axis). The normal E_x /
  E_y reach the centres as (face eps x E) / cell eps (a plain E mean is first order at interfaces); with
  solver_defaults.subpixel_smoothing the shift converges at second order, without it at first. gyration is never
  smoothed (cell-centre staircase): keep magneto-optic interfaces on cell edges. Frame map to em_fdfd_hz: (x, y, z)_modes
  = (z, y, -x)_plane, so gyration_x = -gyration_z of the in-plane guide (self-test em_modes_nrps_first_order vs the
  exact slab; example analytic_modes_nrps_ce_yig_slab).
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
