# open_sol/physics/potential — conventions & gotchas

Sub file of the root `structure_conventions.md` (read that first). Potential gotchas; DiffusionSystem (assembly, PCG,
held cells, metric, cut cells) is in `open_sol/core/sub_structure_conventions.md`.

## PotentialPhysics
- Types `electric_currents` (default instance 'currents') and `electrostatics`: both family '', one factory in
  definition.py. Heat -> currents feedback is a state-dependent `sigma_dc` expression of `heat_T` (a
  DynamicMaterialUpdate), not a coupling.
- PotentialPhysics (`kind` 'currents' / 'electrostatics' -> `KINDS[kind]`, a KindSpec of label, coefficient scale /
  arrays, source slot / field, flux names, units, anchor text) = DiffusionSystem (as heat, reference 0; settings, record,
  `set_face_bc`, `add_boundary_flux`, probes from `DiffusionPhysicsBase`) + PotentialFields2D (V, E_x, E_y, flux_x /
  flux_y, source, source_ext, q_joule currents only, rho_ion electrostatics only; c_ion_k on demand). V, E, flux and
  q_joule are edge-extended into the PML ring like T (a field-dependent 'all' material or eps field_transfer then sees
  no step at the CPML).
- Backends: fields and ionic / driven-terminal work arrays on `physics.xp`; the driven superposition uses `_add_scaled`
  or its numpy twin; pockels.py kernels are `core.kernels` twins. Covered by heat's diffusion_cpu_gpu (terminal
  divider: it reuses `selftests._terminal_divider_specs` / `_check_terminal_divider` on a currents-only run, so keep
  their default call currents-only) and the gpu primary + cpu variant of potential_azimuthal_transverse_field and
  electrostatics_electrolyte_stern_nonlinear; other tiny tensor cases run on the cpu backend (launch latency dominates).
- Twin cases (`selftests.merged_kind_check`, `TWIN_TEXT`): uncoupled currents + electrostatics in one model on the same
  cells, each kind's check kept, the case reports the worse error / tolerance; derived value names are per kind
  (`F_core_<kind>`) and `q_joule` is qualified (`currents_q_joule`).
- Coefficient arrays `sigma_dc_xx / _yy / _xy` (S/m, default 0 = insulator) and `eps_dc_xx / _yy / _xy` (default 1,
  x EPS0 at assembly), plus `sigma_dc_zz` / `eps_dc_zz` (read only on tetrahedra; isotropic sigma_dc / eps_dc write
  them; eps_dc_zz is also solid_mechanics_3d's optional array, same Param). The material vocabulary always carries the
  `_dc` infix, never the optical sigma_e / eps_r. The electrostatics schema also registers the Pockels arrays `r_xx /
  r_yy / r_zz` (`POCKELS_SCHEMA`; read only by electrostatics_em_pockels via `pockels.py`, never by the operator) and,
  as `optional_material_schema` (any one set allocates all eight, default 0; a crystal entry sets them), the in-plane
  crystal tensor `POCKELS_TENSOR_ARRAYS` pockels_r11 .. _r62 (row xx / yy / zz / xy vs E_x / E_y,
  `pockels.POCKELS_TENSOR_ROWS`; row xy = `pockels_shear_shift`, the em_fdfd_hz eps_xy). The electrostatics charge field
  is `rho_free` (`rho` is heat's density, which would force qualified names).
- `step` re-solves only when dirty (operator: `materials_changed` of its coefficient arrays, `set_face_bc`,
  `add_fixed_potential`; rhs: `inject_source`, `add_source`, `add_boundary_flux`) or at the first call; probes sample
  after a re-solve and at every `step(t, inf)` (a stationary iteration still records; flush gives the end-of-run
  sample), never on an unchanged finite-dt tick (every = 1 would store one copy per EM step).
- RerunSolver (sweep reuse_build; FV and FEM 2D / 3D): `reset_records` clears the probe series; no feature value
  follows yet (a live voltage / terminal / source value rebuilds per sweep value).
- E = axis flux / c at the centres (`axis_flux_at_centres`; 0 in non-conducting and held cells); J / D add
  `cross_flux_at_centres`; q_joule = `DiffusionSystem.dissipation_density` (each interior face's g (du)^2 split by
  half-cell resistance, `fv_diffusion_3d.face_lower_shares`, so a metal contact gets ~0; outer Dirichlet / Robin faces whole to
  their cell): its window integral == sum over held features of V x flux + sum of V x source x area + sum of V x inward
  boundary flux x length (exact). Shared help texts are formatted per kind (`_XY_HELP_TEXTS`, `_SETTINGS[kind]`,
  `_COMMON_DESCRIPTION`): electrostatics has no q_joule and every cell conducts.
- `held_flux(HeldRegion)` = `face_inward_flux` of the held outer faces (only those whose current BCSide is still
  dirichlet at `region.voltage`, or its Stern-layer Robin BC: `_face_held_by`; a later feature on the face wins) +
  `fixed_region_outflow` of the held cells (A_full V - b_full of the un-eliminated system in float64 from mask-edge
  faces plus exact row sums: never a masked SpMV residual, whose interior rows leave ~eps g_terminal V of rounding
  each), positive out of the held region (sources inside a terminal subtracted), times `mirror_factor(region)` (x 2 per
  mirror plane a held face lies on or meets, or held cells reaching their window edge layer; `_region_flux` / driven
  basis fluxes stay drawn-half values, held_flux scales once; self-test currents_mirror_held_flux). A cell held by two
  features (`set_fixed`: the later value wins) still counts in both regions; `definition_helpers.held_object_conflicts`
  warns on a possible overlap (shared object or 'all', or intersecting bounding boxes: hedged).
- Scalar outputs: `current_<feature>` (currents) / `charge_<feature>` (electrostatics) per enabled voltage / terminal =
  `held_flux` of the regions registered by `add_held_region`; {} before the first solve; a name equal to an output field
  (terminal 'source' -> current_source) is not declared. Pinned by potential_divider_plates_graded and the examples'
  derived values.
- Setting `geometry` 'axisymmetric' = `PotentialPhysics(axisymmetric=True)`: densities keep their units, held fluxes /
  scalar outputs are whole rings (`PotentialPhysics.flux_unit` A / C instead of `KindSpec.flux_unit` A/m / C/m: table
  result units follow it), object-perimeter current_density / surface_charge go through `perimeter_flux_weights`.
  Validation `axisymmetric_issues` with prefix sigma_dc / eps_dc. Checked only by examples axisymmetric_coax_capacitor
  / axisymmetric_ring_electrode_resistor.
- Setting `azimuthal_order` (shared with heat; core sub file "Azimuthal order"): V, E / J / D (in-plane r / z
  amplitudes, no phi component), sources and held values are cos(m phi) amplitudes; held fluxes / scalars are cos(m phi)
  moments (pi rings); q_joule includes the azimuthal term. No anchor needed (validation check skipped); a nonlinear
  electrolyte is refused (validation and `add_electrolyte`). Self-test potential_azimuthal_transverse_field (twin,
  `selftests_axisymmetric.py`; variant cpu).
- Polar / spherical domains (both types list 'polar'): `mapped_domain_issues` refuses sigma_dc_xy / eps_dc_xy and
  object current_density / surface_charge except in 1D models; 2D perimeter weights by metric, cut cells with
  conformal_boundaries (selftest conformal_currents_spherical). Components _x / _y and coefficients _xx / _yy are radial
  / angular (names kept; `_PotentialRenderer` relabels by `core.renderer.polar_field_infos`, so field_infos is an
  instance attribute on polar grids). Examples polar_sector_angular_resistor / spherical_two_layer_capacitor;
  self-tests `selftests_coordinates.py` (model path, Results.field(native=True)): currents_polar_sector,
  currents_polar_wrap (the sin(phi_j) eigenvector is odd about the wrap ray, so V, J_y and P_joule pin the wrap).
- 1D models (`dimensions` (1, 2)): same solver on the one-cell-across Domain, Automatic banded; `flux_unit` per m^2 on a
  slab, per m on a cylinder, whole on a sphere. Self-tests: capacitor runs of model/selftests_line.py.
- Electrolyte (electrostatics only, `KindSpec.with_ions`): feature `electrolyte` (parallel float_lists concentrations
  [mol/L] / valences, temperature, bulk_potential, model 'linear' | 'nonlinear'; uniform over the selection) ->
  `add_electrolyte(...)` keeps an `IonicMedium` and the union mask `_ion_mask`; math in `electrolyte.py`
  (`boltzmann_terms` exponent clipped at +-80). Every `_assemble(potential)` writes the WHOLE ionic term as the tangent
  at `potential` by `set_diagonal_loss(_ion_mask, k0, V0 + rho(V0) / k0)` (linear: exactly the k-weighted V_bulk), then
  clears held cells (`set_diagonal_loss(fixed_user, 0, ...)`: terminals are ion-free conductors, else held_flux would
  count their ionic charge). Never `add_diagonal_loss` (it would pile up per Newton iteration). rho_ion = `_ionic_terms`
  at the solved V (exact Boltzmann charge, 0 on held cells), not the tangent. It anchors at runtime (extra_diag > 0) and
  in validation (`_has_anchor`: sum z^2 c > 0). Scalars ionic_strength_ / debye_length_<feature> (bulk linear Debye
  length at the selection's mean static eps_dc at apply). Validation: one valence per concentration, none 0, neutral
  bulk (error), info with lambda_D x sqrt(eps_dc), warning for nonlinear + float32 with nonlinear_tolerance < 1e-3,
  warning beyond `MAX_SPECIES_FIELDS` species.
- Concentration fields `c_ion_1 .. c_ion_4` (mol/L): species k = the k-th list entry of every IonicMedium (overlaps
  add), from the float32 V window (linear: c (1 - z psi / V_T), so their charge sum is rho_ion in both models; m != 0:
  values at phi = 0), 0 on held cells; computed on demand by `ion_concentration` (renderer only: output_fields, NOT
  state_fields, no full-grid array, no PML copy). The renderer lists them up to `ion_species_count` at its creation
  (Session renderers come after the build; a script renderer made before add_electrolyte lacks them): storing c_ion_3
  of a 2-species model fails at snapshot setup. Their temporaries stay below the ionic-diagonal run item (no estimator
  item).
- Stern layer (voltage params stern_thickness / stern_permittivity, outer faces only: error with objects) =
  `held_face_bc`: Robin h = C_S = eps0 eps_S / d_S, external = voltage (no cells, no ions).
  `HeldRegion.stern_capacitance` makes `_face_held_by` own a face while it is still that Robin BC; driven faces pass it
  to `set_driven_face_voltage(..., stern_capacitance)` (the basis BCs zero its external). Terminals / voltage object
  perimeters have none (would need per-face conductance factors on held edges).
- Nonlinear (any medium 'nonlinear'): `step` runs `_solve_nonlinear` = `core.damped_iteration.DampedIteration`
  (`self.newton`, electrostatics settings nonlinear_tolerance / _max_iterations / _relaxation, step cap
  `NEWTON_MAX_STEP` = 4) on a float64 copy of V, scale = min V_T / max |z|; propose = `_assemble(V,
  diagonal_only=True)` (ionic tangent, then `refresh_diagonal`; full assembly only when it returns False) +
  `seed_unknown(V)` + build_rhs + solve. Unconverged: damped V kept, warning to `runtime_warnings`. `work_counters` =
  nonlinear_runs / _iterations / _unconverged and diagonal_refreshes only when nonlinear, else {}. Memory: the
  electrolyte FeatureType estimator counts the masks (kept) and the tangent / Newton temporaries (transient phase
  'Newton iteration' / 'ionic diagonal'; taken as not overlapping the physics' own refactorisation phase).
- Memory of the physics (`_make_memory_estimate`): a cuDSS 'linear' back end adds
  `fem.diffusion.cudss_diffusion_estimate` (tensor: on the kept A_solve; element meshes: `fem_diffusion_estimate[_3d]
  (cudss=)`), `refactorizes` = `_operator_changes` (a nonlinear electrolyte, a coefficient of the kind following a
  state; also the time recipe's operator_changes); element meshes size the system on `_active_fraction`
  (electric_currents: sigma_dc materials, all with an electrolyte; electrostatics: all). Flux post-processing is a
  transient phase (diffusion_system_estimate), the driven terminal bases and dissipation shares stay run items.
- Self-tests `selftests_electrolyte.py`: electrostatics_electrolyte_slit, _charged_wall (electrolyte the only anchor),
  _pore_axisymmetric, _gouy_chapman, _boltzmann_2_1, _stern_linear / _stern_nonlinear (nonlinear: gpu primary + variant
  cpu); slit, both Boltzmann walls and the Stern cases also pin c_ion_k against V and their charge sum against rho_ion.
- Driven terminals (electrostatics only): voltage / terminal params waveform (none | ricker | gaussian | cw), frequency,
  bandwidth -> `TerminalWaveform` (frozen, equal entries share a basis) passed to `add_fixed_potential(...,
  waveform)` / `set_driven_face_voltage`. Every hold is kept in order (`_cell_holds`, `_driven_faces` with the BCSide
  object: a later BC on the face drops the drive). With any waveform `step` runs `_step_driven`: `_solve_bases` after an
  operator change (one solve per waveform with its holds at their voltage and everything else 0, then the
  DC basis last, which stays the live assembly; BC objects are swapped on `self.bcs` and restored, never via
  set_face_bc; the first basis assembles, the rest only re-eliminate: `update_face_value` per swapped face +
  `DiffusionSystem.refresh_held_values`, a full assemble when either refuses, e.g. in 3D), the DC basis alone after a source change, then `_superpose(t + dt)` (t when dt is inf). `held_flux` /
  scalars superpose per-basis region fluxes (`_region_flux` judges face ownership on the saved BCs). Coupling revision
  = `state_revision` = n_solves + superpositions. Not with an electrolyte; stationary gives s(0) ~ 0 (warning). Memory:
  bases as run items. Self-test electrostatics_waveform_superposition (coupling/selftests_piezo.py).
- `solve_coupled_source(time)` = `step(time, inf)`: the two-way piezo re-solve inside a tick after it injected the bound
  charge (driven: DC basis re-solve, superposed at `time`). Probes replace a sample at the same time when time > 0, so a
  trace has one sample per step; stationary iterations at t = 0 still append.
- Self-tests (`selftests.py`): the 'thermo_optic_heater' template and the twin potential_divider_plates_graded (cpu,
  float64 CG to 1e-12, tolerance 1e-6, closed forms on the actual graded cells: edges rebuilt from the float64 centres,
  rectangles on float32 centres like the build; raises when the mesh fell back to uniform). Uniform-mesh cases are
  examples (series_parallel_resistors, layered_parallel_plate_capacitor, graded_resistor_current_drive,
  uniform_injection_square_plate, space_charge_surface_charge_slab).

## Triangle meshes (FEMPotentialPhysics, `solver_fem.py`, mesh plan T4; fem sub file "Scalar diffusion")
- FEMPotentialPhysics on a TriangleDomain (setting `discretization`): KINDS vocabulary, slots, state names and scalar
  outputs as PotentialPhysics, state per element (q_joule = element dissipation / area, so its integral = sum of V x
  held current exactly); V nodal in the renderer.
- Held regions are `FEMHeldRegion`: a voltage on objects = Dirichlet on their outline edges, a terminal = every dof of
  its elements; `held_flux` = the face flux while the face still holds the region's BC (`_face_held_by`;
  `FEMDiffusionSystem.face_flux`: the Dirichlet reaction of the dofs that face won, so a corner of two held faces counts
  once, or the Stern Robin flux) + the reaction of its outline / element dofs, x 2 pi axisymmetric, x mirror_factor.
- Electrolytes (model 'linear' only): mask_window is a host bool per element; every `rebuild_operator` rewrites the
  WHOLE loss term (`FEMDiffusionSystem.set_loss`) as k (V - V_bulk) over the media, 0 on held (terminal) elements; the
  inside of a voltage object is not held (its outline is) and carries ions when selected. rho_ion / c_ion_k from the
  element mean V, so int rho_ion = minus the sum of held charges exactly; c_ion_k via `fem_field` up to
  `ion_species_count`. Debye-length eps_dc mean: area-weighted. Nonlinear would need a Newton loop on a
  per-quadrature-point Boltzmann mass (`DampedIteration` with `assemble_mass` of the tangent).
- Refused: nonlinear electrolytes, voltage waveforms, azimuthal_order != 0 (validation). Gap: driven terminals would fit
  cheaply (one factor, each basis a lift), not built. Time estimate = 2D recipe at `fem.diffusion.fem_time_system`
  (uncalibrated). cuDSS: heat sub file; potential re-factors only on material / BC-kind / held changes, so a coupled
  sigma(T) loop is the plan-reuse case.
- Self-tests `selftests_fem.py`: electrostatics_fem_coax (C' vs 1e-5, observed order >= 2.5: non-nested meshes give a
  noisy estimate of the ~4 P2 energy order), currents_heat_fem_joule (heat on the same triangles and on a tensor mesh
  through the conservative transfer), electrostatics_fem_electrolyte_slit (tolerances first estimates).

## Tetrahedron meshes (FEMPotentialPhysics3D, `solver_fem3d.py`, piezo_3d_plan stage 6; fem sub file "Scalar diffusion on tetrahedra")
- Both types have `dimensions` (1, 2, 3), but 3D runs only on 'tetrahedra': `_three_d_issues` refuses a 3D tensor mesh
  (the create raises, estimators return nothing), axisymmetric / azimuthal_order, model mirror faces (held fluxes would
  need the 3D touch factor), voltage waveforms, `<prefix>_xy` materials; the Stern layer in the voltage validate;
  electrolytes by their FeatureType grids.
- `solver_fem.FEMPotentialMixin` holds what FEMPotentialPhysics and FEMPotentialPhysics3D share (kind, fields per
  `_axes`, steps, sources, held reactions `_held_reaction` over the class's `_held_dofs`, publish of V / E / flux /
  q_joule); REMEDY_TEXT / RENDERERS come from solver.py.
- FEMPotentialPhysics3D on FEMDiffusionPhysicsBase3D: diagonal tensor (`DIAGONAL_ARRAYS`), state per element plus E_z,
  the z flux and the zz array (rho_ion zeros for the vocabulary); renderer `LagrangeSampleRenderer` with the 2D infos (no
  c_ion_k) + `renderer.THREE_D_FIELD_INFOS`, listed as output fields only (not state_fields, so not in expressions).
- Held regions `FEMHeldRegion3D`: a voltage on objects = Dirichlet on their interface faces (an object touching the
  window also holds its faces on the window planes: keep conductors there, e.g. a coax outer conductor as 'window minus
  cylinder'); `held_flux` units A / C (`KindSpec.ring_flux_unit`). Flux features on objects are sheet sources
  (`add_surface_flux`), probes evaluate V at the point.
- Estimates: `fem_diffusion_estimate_3d` with the 2D field count + 2, time = 3D recipe (`fem_time_system`,
  uncalibrated). Not checked: couplings of these 3D physics (currents_heat_joule to heat on tetrahedra, Pockels /
  piezo), the gpu backend.
- Self-tests `selftests_fem3d.py`: electrostatics_fem3d_plates (exact C and Gauss, 1e-9), electrostatics_fem3d_coax
  (gmsh, skipped without it; curvature_factor 0.15 on both circles so facet ratios match; C within 2e-3),
  currents_fem3d_resistor (int q_joule = V I within 1e-6 for float32 storage).
