# open_sol/physics/eme — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first; slab modes / PortLine / FDFD / cross-section mode
gotchas: `open_sol/physics/em/sub_structure_conventions.md`). Eigenmode expansion `em_eme` (2D) and `em_eme_3d`
(`eme_plan.md`).
Types: `em_eme` (family 'em', default instance 'eme', 2D planar, settings polarization / propagation_axis, 'modes'
vocabulary; stationary, frequency_domain, optimization (own adjoint_objective on port power fractions, J unit 1; the
em design_region on eps_r (ez) / eps_xx, eps_yy (hz)); refuses every other EM type; absorbed_power from the last solve,
so an em_heat_joule source) and `em_eme_3d` (3D only, propagation z, same vocabulary / refusals, stationary /
frequency_domain (adjoint_objective gives eps_sensitivity; no 3D design_region / optimization), z faces fixed pec
ports, 'eigen' solver role).

## Layout and imports
- Host numpy only: `sections.py`, `mode_sets.py`, `interfaces.py`, `cascade.py`, `adjoint.py` (CPU self-tests use them
  directly; mode_sets imports physics.em inside functions). Anything a CPU self-test needs (unit_cell, repeated_link,
  Bloch modes, bent_line, Gauss means) lives there, never in solver.py. Run backend (`domain.xp`, gpu and cpu on both
  types): `solver.py` (EMEBase, EMEPhysics 2D, monitors, results), `solver3d.py`, `renderer.py`, `renderer3d.py`, the
  optimization study; masks checked on the run backend (`_check_cell_array(xp=)`, `_check_box_mask`).
- definition.py imports solver and renderer; definition3d.py imports definition.py's shared feature pieces and
  em3d.boundaries3d.BoundaryConfig3D (the only em3d import); definition.register imports definition3d / selftests3d
  inside the function; `__init__` imports none of them. physics.eme imports physics.em, never the reverse.
  selftests3d and selftests_adjoint import selftests helpers.

## Sections and mode sets
- One slice per cell layer normal to the propagation axis (2D: x or y; 3D: z); the faces along that axis are
  semi-infinite ports at the window planes (their bc / PML not read; periodic and mirrors refused; 3D: z faces pec, no
  z padding). Transverse periodic faces: k = 0 only (2D `check_zero_bloch`, 3D `EMEPhysics3D.__init__`). A section =
  consecutive slices whose host material rows (read arrays, pole masks, PEC, bend labels) equal the FIRST slice's
  (`section_tolerance` 0) or stay within tolerance x each row's max |value| of it (so a smooth taper does not merge by
  small steps); solved on that first slice. `Section.key` = blake2b of its rows.
- Positions along the axis (bend / periodic / length-sweep ranges, Section.start) are measured from the window's lo
  edge (`solver.slice_positions`). A feature range selects sections (or slices, bends) whose CENTRE lies in it.
- 2D: each section line is a `PortLine` with `solve_slab_modes(..., drop_pml_modes=False)`. 3D: the layer's
  CrossSectionMaterials and `solve_cross_section_modes(..., drop_pml_modes=False)` on the kernel (float32) xy widths
  with the 2D view of the x / y faces (`solver3d.cross_section_boundary`); section lengths from float64
  `Domain.cell_widths()`. PML modes stand for radiation and are needed for completeness.
- Ranking: the solver's, then PML modes (> half their weight on stretched points) last (`ModeSet.pml`), so port mode k
  = the k-th mode mode ports / em_modes would find.
- Cache: entries carry over between solves at an unchanged omega (a thermo-optic Picard loop re-solves only changed
  slices); capped at `MODE_CACHE_BYTES` per solve, carried-over entries count too (else the cache grows by up to the cap
  per solve); uncached sections are re-solved for the field reconstruction (`mode_resolves`). At a new omega 3D starts
  ARPACK from the last entry's start shell (a CrossSectionMode copy with only summed H_x, H_y: never use it as a mode);
  2D ignores it. Mode ranks are per wavelength (crossings may swap port mode numbers).
- Normalisation (mode_sets header): unconjugated <e, h> = 1 with stretched widths / dual areas (3D e = [E_x; -E_y], h =
  [H_y; H_x]); never the conjugated power overlap. 3D degenerate modes are recombined bilinear-orthogonal. Power uses
  `ModeSet.power` (3D: whole mirrored mode).

## Bends (eme_bend, `add_bend`)
- The drawing is the straightened bend: the range length is the arc length at reference radius R. Each bend adds a
  label row to the slice rows (sections split at the range ends); `add_bend` clears the mode cache (the label does not
  encode R).
- 2D (`mode_sets.bent_line`): the exact cylindrical operator for ez and hz, k = beta at r = R; the along-axis field
  divides the line flux by r_f/R (`section_fields(face_scale=)`; entry = (port, line, face_scale, modes)).
  absorbed_power is the physical density: its integral over the straightened drawing misses r / R.
- 3D: the cross_section_modes bend tuple (R, reference_x, full-grid x edges, 'exact'); bends about y only. Its
  y-invariant TE block equals the 2D ez bent_line exactly (self-test).

## Interfaces and cascade
- Interface formula: interfaces.py header; dimension-agnostic. Complete basis (2D n_modes 0) = exact (S reciprocal,
  power conserved); truncation shows as the section table's reciprocity_defect (warning above 1e-3). 3D bases are
  always truncated.
- Discrete propagation (`EMEPhysics(discrete_spacing=D)`, self-tests only): with a complete basis it solves the
  em_fdfd_ez and em_fdfd_hz lattices of the sectioned guide (eme_vs_fdfd_step_ez / _hz). A change of the FDFD hz face
  averaging (harmonic means, subpixel on) breaks the hz identity.
- Cascade: Redheffer star products only (never transfer matrices); CascadeBuilder holds 8 N^2 complex per section at
  the peak. Its forward pass keeps only s21 / s22 (`_extend_left`), the backward pass only s11 / s12 (`_extend_right`),
  and the total comes from the end partials: half a star per interface and pass (time estimate 1 star per section). `section_amplitudes`: forward at a section's lo end, backward at its hi end, so every factor is exp(-j k d)
  with d >= 0 (stable with evanescent modes).
- Port S-matrix reference planes = the window faces; port modes = first / last section's modes.

## Periodic block (eme_periodic, `add_periodic`)
- Sections with centres in the range = the unit cell; `cascade.unit_cell` = junction (last -> first) then the cell.
  With n_periods > 1 the interface after the cell becomes `repeated_link` = cell^(n - 1) * exit interface, so cascade,
  partials and field need no special case: the drawn cell shows the first period, later sections the field after the
  last period, the hi port plane is (n - 1) periods beyond the window face. A section must follow the cell
  (ValueError). `_link` lets `assemble_total` rebuild the link (length sweep).
- Bloch modes (`cascade.bloch_modes`): lam = exp(-j K period); forward = |lam| < 1 or unit circle with net forward
  power; n_bloch unfolded to the zone nearest mode 1's length-weighted mean Re k.

## EMEBase, EMEPhysics, EMEPhysics3D and the model layer
- EMEBase: sections -> cached mode sets -> interfaces -> cascade -> scalars -> `_reconstruct` -> monitors; a subclass
  supplies `_find_sections`, `_solve_section(section, omega, previous_entry)` (entry tuple ending with the ModeSet) and
  `_reconstruct`. step solves when dirty (read arrays, PEC, poles, excitation, omega, bends, periodic), solve_harmonic
  always. inject_source takes all three permittivity slots.
- Absorbed power (EM loss protocol): `q_abs` float32 full grid = `take_absorbed_power()` of the last solve (never
  reset, None before), `state()['q_accum']` None. q = 1/2 omega eps0 L <|E|^2> (sigma_m and PML stretch excluded, as
  FDFD), <> = 3-point Gauss-Legendre cell mean along the axis, so its integral is the exact absorbed power (1 - R - T).
  3D evaluates Gauss rows only on sections whose first layer has loss; PEC cells lossless.
- 2D field at cell centres per section; wall cells beside the line range take sign x neighbour (|sign| for intensities
  and the Gauss mean). The renderer offers the union of the ez / hz FDFD fields (other polarisation reads 0).
- 3D field: complex64 E at the centres of every window z layer + float32 q_abs (28 B per cell; backward E_z opposite
  sign), in blocks of FIELD_BLOCK_SLICES; renderer3d derives per cut plane (`device_plane`).
- 3D PEC: cropped box masks (`add_pec(window, mask)`, 'all' refused), one slice row per box, the cross_section_modes
  PEC rule.
- 3D eigen solves use the 'eigen' role; `host_factor_bytes` = largest SuperLU factor of the last solve. 'cudss' via
  `eigen_solver.factorize_host` (one plan + general LU per distinct section: the transverse-H operator has no known
  symmetriser); 'cudss_mixed' not offered (ARPACK at machine precision would need full refinement, not repaid). With
  cudss the SuperLU size rule is skipped and the memory estimate drops the SuperLU item. Not done: plan reuse across
  sections (same pattern).
- Mesh kinds: tensor Domain3D only. Tetrahedra do not fit (no planar cross-section per z layer); the conforming route
  (one shared xy triangle mesh, prism mesh kind, FEM versions of reconstruction / absorption / adjoint / bends / PEC)
  is not done.
- Length sweep (`add_length_sweep`): `_solve` keeps every interface and section k (`sweep_data`); the monitor
  re-cascades with `assemble_total` through `port_fractions(smatrix)`. Attaching marks the physics dirty.
- Excitation: `set_excitation` / eme_ports (one per physics); default mode 1 at 1 W/m (3D: 1 W) from lo. Scalars
  transmission_<k> (k <= n_port_modes of the first enabled eme_ports), transmission, reflection.
- Memory estimates: every slice taken as a section (upper figure); host items dominate. Device transients are
  MemoryEstimate phases: 'field reconstruction' (one upload block at a time), 3D 'section mode solve' (one cross-section
  solve at a time: the factor is local to `solve_prepared_modes`, so cuDSS copies, solve vectors, B and the device
  Krylov basis are freed per section), 2D 'adjoint sensitivity' (`sensitivity_density` float64 temporaries). A cudss
  eigen node's factor is one external item (`cudss_factor.factor_device_bytes`, general LU, dimensions 2). The adjoint
  EpsSensitivity is one per physics (EMEBase._sensitivity): counted by the physics estimators when any
  adjoint_objective is enabled; adjoint_objective has no estimate of its own. If plan reuse across sections lands
  (factor kept between sections), the factor stays external but the copies become run items.

## em_eme (2D) on triangles / cuDSS: decision (agent_help batch, 2026-10-23; nothing implemented)
- No triangles: a 2D section is a 1D line solve, so a triangle mesh has no line to solve on (`grids = ('tensor',)`);
  the analogue would be higher-order 1D lines, not needed while Yee lines match em_fdfd exactly.
- No cuDSS: lines need no sparse factor; the rest is dense N x N. Only complete bases with many sections might gain
  from a dense GPU path: unmeasured, not planned.

## Cascade adjoint (adjoint.py, 2D ez and hz, 3D; optimization study 2D only)
- Formula: adjoint.py header (no differentiation through the mode solves). Exact only for a complete basis (2D n_modes
  0; never in a 3D model). A change of the interface matching or the mode normalisation breaks it: rerun
  eme_cascade_adjoint_gradient / eme_3d_cascade_adjoint_gradient.
- EMEBase holds add_objective / eps_sensitivity (cache per (n_solves, objectives)); subclasses `_cascade_sensitivity`.
  3D (`cross_section_gradient`): a change of the cross_section_modes point-eps rule (`_point_materials`: ghost fill,
  face_average, bend factors, eps_point_deltas) or of the dual areas must be mirrored there. Every window z layer is
  evaluated (no 3D design regions).
- hz: eps enters rho and the face conductances, not the index term (`solver._HZ_LINE_ARRAYS`; faces back to cells by
  `face_gradient_to_cells` with PortLine's face index). A change of the PortLine hz face rule or of the end conductances
  (`longitudinal_flux` end factors) must be mirrored there. Design regions write eps_xx and eps_yy.
- `cascade` (partials) is kept only while an EMEObjective monitor exists; eps_sensitivity reuses the last mode cache and
  (2D) evaluates only the design-region slices. Merged sections: every slice is integrated over its own range.
- eme_periodic n_periods > 1: a cell slice's gradient sums the per-period products (`adjoint.amplitude_pairs` over
  `PeriodicAdjointFields`, `cascade.period_plane_amplitudes`, `cascade.cell_cascade`). A change of the periodic link
  (repeated_link / unit_cell order) must be mirrored there and in `interfaces.periodic_cascade_mode_sets`.
- Refused: the discrete variant; design regions on the first / last window slice (ports fixed: build raises);
  section_tolerance > 0 in the optimization study (validation error). Bends: the r / R factors (`_centre_scale`,
  face_scale for hz).
- J = sum of power fractions (unit 1, `EMEBase.objective_unit`, passed to `DesignRegion(objective_unit=)`);
  eps_sensitivity in 1/m^2.

## Self-tests (selftests.py, selftests3d.py)
- 2D CPU cases on hand-built SlabLines (`_slab_line`: uniform cells, mu 1); model runs eme_lossy_film_absorption,
  eme_vs_fdfd_step_ez / _hz, eme_radiating_step_pml_modes (em_fdfd_* on cpu vs the CPU discrete EME of FDFD / PortLine
  line replicas, `_step_line`; the em_eme default-backend item is skipped without a usable GPU). A change of the PortLine
  end rules or the CPML profile layout must be mirrored in `_step_line`.
- Covered only by examples: the ez periodic-block model (analytic_eme_bragg_periodic_block.json) and the ez bend model
  (analytic_eme_bend_whispering_gallery.json).
- CPU cases reuse one mode solve for identical inputs; `selftests3d._pec_reference_sets` is memoised across eme_cpu_gpu
  and eme_3d_cudss_pec_model: never mutate the returned ModeSets.
- 3D CPU (`_cross_section_set`): eme_3d_uniform_identity, eme_3d_field_reduces_to_slab (TE only: the bent E_z is not
  pinned); eme_3d_lossy_film_model uses eps_yy = 0.9 eps_xx to make the x-polarised mode non-degenerate;
  eme_3d_cudss_pec_model skips without nvmath (specs() returns no run). eme_complete_basis_identities also checks the
  length-sweep re-cascade.
- selftests_adjoint.py: eme_3d_cascade_adjoint_gradient gets complete bases from a dense `eigs` override of
  ArpackShiftInvertSolver (shift away from every eigenvalue; a small closed pec box is below cutoff, so wall variants
  are pec x / pmc y; float32-exact steps 2^-9); eme_taper_optimization / _hz pass on closure of 1 - J0 (J <= 1 caps a
  gain ratio at 1 / J0). Heavy flags are set only on the selftests_adjoint cases.
- selftests_backend.py: eme_cpu_gpu (registered inside definition.register).
- Templates linear_taper_eme, mmi_1x2_eme, thermo_optic_phase_shifter_eme_3d.
