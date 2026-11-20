# open_sol/physics/eme — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first; slab modes / PortLine / FDFD / cross-section mode
gotchas: `open_sol/physics/em/sub_structure_conventions.md`). Eigenmode expansion `em_eme` (2D) and `em_eme_3d`;
vocabulary, studies and refusals are in their help texts and validation, the method in the module headers.

## Layout
- Host numpy only: `sections.py`, `mode_sets.py`, `interfaces.py`, `cascade.py`, `adjoint.py`; the CPU self-tests use
  them directly, so anything a CPU self-test needs (unit_cell, repeated_link, Bloch modes, bent_line, Gauss means)
  lives there, never in solver.py / solver3d.py (run backend `domain.xp`).

## Sections, mode cache, normalisation
- A slice joins a section while its rows stay within tolerance of the section's FIRST slice, not of its neighbour,
  so a smooth taper never merges by small steps (`sections.find_sections`).
- Mode cache: entries carried over at an unchanged omega count toward `MODE_CACHE_BYTES` like new ones, else the
  cache would grow by up to the cap per solve. At a new omega 3D starts ARPACK from the entry's start shell (summed
  H_x, H_y only): never use the shell as a mode. Mode ranks are per wavelength (crossings may swap port mode numbers).
- Overlaps are the unconjugated bilinear <e, h> (mode_sets header), never the conjugated power overlap; powers come
  from `ModeSet.power` (3D: the whole mirrored mode).

## Interfaces and cascade
- Cascade by Redheffer star products only, never transfer matrices (they blow up with evanescent modes). The
  functions on the length-sweep re-cascade path (`EMEBase.assemble_total`: star, append_phase, total_smatrix,
  smatrix_power, unit_cell) must keep broadcasting leading batch axes (the stretch factors are stacked there).
- The memory and time estimators assume the CascadeBuilder footprint (8 N^2 complex per section, half a star per
  interface and pass): a change of what the builder keeps changes `definition._memory_estimate`,
  `definition3d._memory_estimate` and `shared_time_estimate`.
- Discrete propagation (`EMEPhysics(discrete_spacing=)`, self-tests only): a complete basis reproduces the em_fdfd_ez
  / _hz lattices (interfaces header); a change of the FDFD hz face averaging (arithmetic face_average, subpixel off)
  breaks eme_vs_fdfd_step_hz.

## Model layer
- Snapshot solutions (`EMEBase.solution_state` / `restore_solution`): a new renderer input that changes per solve
  joins the state. eps_sensitivity of a restored earlier solve comes only from the density cached at that solve (the
  optimization study's evaluations cache it; frequency_domain only when the field is stored), never from the live
  cascade.
- Absorbed power excludes sigma_m and the PML stretch (as FDFD); PEC cells are lossless (2D `_loss`, 3D
  `_layer_loss`).
- Memory estimates: adjoint_objective has no estimate of its own: the physics estimators book the one
  EpsSensitivity per physics (`EMEBase._sensitivity`) when any adjoint_objective is enabled.
- em_eme_3d keeps the reconstructed E (`EMEPhysics3D.field_arrays`, complex64) as host numpy on either backend: only
  the renderer (host math on the read window / plane, one float32 upload) and solution_state read it. q_abs stays on
  the run backend (state() feeds the loss couplings). Its EpsSensitivity cell_gradients are host float64 too (only
  the float32 density is on the device: no 3D design regions), unlike 2D em_eme's run-backend ones that
  study_optimization gathers. A new device reader of either uploads per read, never a kept device copy; 2D em_eme
  keeps its float32 field arrays on the device (they are state() keys).

## Meshes and cuDSS: decision (nothing implemented)
- em_eme: tensor only (`grids = ('tensor',)`): a 2D section is a 1D line solve, so a triangle mesh has no line to solve
  on (the analogue would be higher-order 1D lines, not needed while Yee lines match em_fdfd exactly). No cuDSS: lines
  need no sparse factor, the rest is dense N x N; a dense GPU path might pay only for complete bases with many
  sections (unmeasured, not planned).
- em_eme_3d: tensor Domain3D only. Tetrahedra do not fit (no planar cross-section per z layer); the conforming route
  (one shared xy triangle mesh, prism mesh kind, FEM versions of reconstruction / absorption / adjoint / bends / PEC)
  is not done. Its cudss eigen node factorises a general LU per distinct section; sections of one pattern reuse the
  plan inside a run_study (cudss_factor "Parked plans").

## Cascade adjoint (adjoint.py; optimization study 2D only)
- Formula and slice integrals: adjoint.py header (exact only for a complete basis; keep every slice factor of modulus
  <= 1). A change of the interface matching or the mode normalisation breaks it: rerun eme_cascade_adjoint_gradient /
  eme_3d_cascade_adjoint_gradient.
- Mirrors of other code: 3D `cross_section_gradient` transposes the cross_section_modes point-eps rule
  (`_point_materials`: ghost fill, face_average, bend factors, eps_point_deltas) and the dual areas; 2D hz
  (`solver._HZ_LINE_ARRAYS`, `face_gradient_to_cells` with PortLine's face index) follows the PortLine hz face rule
  and the `longitudinal_flux` end factors; eme_periodic n_periods > 1 (`PeriodicAdjointFields`,
  `cascade.period_plane_amplitudes`, `cascade.cell_cascade`) follows the repeated_link / unit_cell order, as does
  `interfaces.periodic_cascade_mode_sets`. A change of any of them is mirrored there.

## Self-tests
- A change of the PortLine end rules or the CPML profile layout is mirrored in `selftests._step_line` (the CPU line
  replicas of eme_vs_fdfd_step_* / eme_radiating_step_pml_modes).
- Covered only by examples: the ez periodic-block model (analytic_eme_bragg_periodic_block.json) and the ez bend model
  (analytic_eme_bend_whispering_gallery.json): run them after a change there.
- CPU cases share memoised mode solves (`selftests3d._pec_reference_sets` across eme_cpu_gpu and
  eme_3d_cudss_pec_model): never mutate the returned ModeSets.
- eme_3d_cascade_adjoint_gradient gets complete bases from a dense `eigs` override of ArpackShiftInvertSolver (a small
  closed pec box is below cutoff, so the wall variants are pec x / pmc y; float32-exact steps 2^-9).
