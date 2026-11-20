# open_sol/physics/acoustics — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Cross-file rules only: staggering, slip walls, CPML,
coefficients, time step, axisymmetric and half_storage rules, sources and probes are in the solver.py / kernels.py /
sources.py docstrings and kernel comments and the elastic_waves help texts (definition.py); pressure_acoustics in the
solver_harmonic_fem.py header and the fem sub file "Complex scalar harmonic systems" / "Reduced sweeps".

## elastic_waves (`AcousticPhysics`)
- Step graphs (gpu): cached launches (`_launch_arguments`, the ProbeRecorder's, a source's `table_launches`) replay as
  CUDA graphs keyed by the tuples' ids, so every array they hold is rewritten in place, never reassigned (material
  rebuilds, `restore_solution` likewise); a per-step scalar (the heating triple) stays outside the cache and splits
  the run. `advance` and `_advance_block` (the multi-step `advance_ticks` runner hand-off, coupling sub file "Runner
  hand-off") build the same launch list in the same order: a new launch goes into both, a new eager one also stops
  `_quiet_steps`. A source's table values are its eager `inject_*` values: change both together.
- Mirrors that change in the same edit: each CUDA kernel and its `_*_host` numpy twin (the numba twins read their args
  through `_velocity_arguments` / `_stress_arguments`); the kernel's coefficient precedence and
  `definition._material_moduli`; `kernels.stability_stiffness` and `definition.stability_stiffness` /
  `piezo_stiffening`; the axisymmetric dt factor and `definition._cell_weights`.
- Coefficient storage (`AcousticPhysics._set_storage`, mirrored by the memory estimate through `material_sets`): va_x /
  va_y exist only while a cell has acoustic_damping (None: the '_lossless' velocity kernels imply va from vb != 0), and
  c22 IS the c11 array while no cell sets c11 / c22. `_choose_storage` decides from the values at every
  `_update_time_step` (build time), `materials_changed` only upgrades (no sync). A new reader of va / c22 checks the
  layout (`va_x is None`, `c22 is c11`; the coefficients kernel writes neither then), and a new cached launch holding
  c22 is dropped with `_launch_cache` / `_pulse_tables` when the layout changes. The rigid keep mask is bool.
- Adjoint pairs (energy conservation; their checks fail otherwise): the monoclinic corner / cell means
  (elastic_monoclinic_plane_waves Q = inf), the axisymmetric velocity / stress updates (ring-volume energy), the piezo
  stress / bound charge (coupling/selftests_piezo piezo_stiffened_bar_resonance Q = inf).
- No `domain_requirements`: absorbing layers lie inside the window, so elastic_waves never conflicts with an EM boundary
  config. Its dt exists only after the build: check.json dt_estimate / MeshEstimate.dt leave it out; validation logs
  `definition.estimated_time_step` (also read by fluid's validation).
- Mirror planes: s = +1 on every plane (no odd face: v_t = 0, s_nn = 0 would need samples on the lo face). A 'rigid'
  face on a model mirror face is the even mirror; bc 'symmetric' is the same slip wall that also makes the face a
  model mirror plane (`own_mirror_faces` reads the value; `_solver_faces` hands the solver 'rigid', so `_pins` and
  couplings never see 'symmetric'); `mirror_feature_issues` does not treat it as an EM partner. Renderer parities: v
  polar staggered, sxy axial_z staggered, normal / hoop stresses, pressure, strains 'scalar' (not 'even', so other
  physics' parity sectors can combine them). The (r, z) axis unfold is exact: v_r / s_rz are 0 on the axis.
  elastic_wave_speeds run p_wave_mirror is the only check of bc 'symmetric'.

## pressure_acoustics mirror planes (`symmetry` sound hard, `antisymmetry` sound soft)
- `_create` passes `mirror_faces(spec, name)` to `set_mirror_faces` before the features apply. The solver drops object
  entities lying on a plane from boundaries and surface monitors (a cut object's window-edge outline is inside the
  unfolded object: `TriangleMesh` edge tags count window-boundary edges as outline; line entities on a plane too, so a
  sheet source there is not modelled) and `_mirror_factors` makes surface results those of the unfolded surface: x 2
  measure per reached plane, x (1 + s) on the p integrand (riding on the loads and the p_b quadrature rows,
  `integrand_factors`); entities on the plane count once. Point sources are not rescaled (the parity-sector
  convention: a source plus its images). Probe and surface results carry sector_rule 'linear'. Only check of
  `_mirror_factors`: pressure_acoustics_duct_mirror.

## Couplings (types in coupling/definitions_acoustics.py / definitions_piezo.py, transforms in couplings.py)
- Cadence: acoustics_em_acousto_optic inherits the FDFD's every k (after_source_step False), so it transfers on ticks
  k-1, 2k-1, ... and the FDFD re-solves on the strain of its own tick (stroboscopic). em_acoustics_photoacoustic
  transfers only after an FDFD re-solve (n_solves); tick 0's acoustic step runs before the first transfer (no heating
  in [0, dt]). The piezo electric_field slots apply at the transfer (they are the electrostatics' live E arrays), so
  stresses recorded by probes inside that tick's `advance` lack its dE; velocities do not.

## Not built / open
- elastic_waves: torsional (v_phi) waves, stretched-r CPML for the hoop terms, transversely isotropic hoop stiffness;
  half_storage with coupled slots (needs a scale from the coupling or a runtime rescale), piezo, c16 / c26 (the
  accumulator adds stored-unit rates), measured speed-up; outer traction-free faces (use vacuum cells); CPML alpha /
  kappa (CPML stability in strongly anisotropic media unchecked, Becache 2003); pyroelectric / thermal-strain charge,
  axisymmetric piezo; acousto-optic moving-boundary term, anti-plane (yz / xz) index change, xy on EM types without
  eps_xy, out-of-plane strain of thermoelastic stress; library fluid sound speeds (water / air: set sound_speed). Piezo
  kernels and the volume scatter have cpu twins but no cpu-vs-gpu parity case.
- pressure_acoustics: losses, eigenfrequency study, structural couplings, m > 0, Bloch / periodic faces, interior
  zero-thickness rigid sheets (need doubled dofs).
- elastic_waves on triangles (decision: not built). The stepper has no linear solve, so cuDSS has no role; triangles
  would pay only for curved traction-free surfaces / interfaces (staircase Rayleigh / whispering-gallery modes), whose
  eigenfrequencies / harmonic response already run conforming in solid_mechanics. Cheapest route if a gate case
  appears: explicit P1 FEM on `fem/elasticity.py` (row-sum lumped mass, one SpMV per step, dt by power iteration); P2
  needs mass-lumped P2 + bubble (row-sum P2 lumping gives zero / negative vertex masses); also missing then: absorbing
  layer on unstructured meshes, point-located sources / probes, per-element slots, axisymmetric / monoclinic variants.
  DG only for strong-contrast fluid-solid interfaces.
