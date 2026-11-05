# open_sol/physics/membrane — conventions & gotchas

Sub file of the root `structure_conventions.md`. The `membrane` type (new_modules_plan S4, S7): tensioned film or
Kirchhoff plate, D (1 + j eta) del^4 w - div(T (1 + j eta) grad w) - rho_s omega^2 w = p, exp(+j omega t). Coupling
(S9): coupling/definitions_membrane.py (solid_membrane_tension -> coupled_tension_*; display tension_xy = total xy).

## Layout
- Module contents: `structure_index.md`. Solvers are host numpy, the state arrays on the Domain's xp (gpu and cpu).
  The package imports no other physics package. The selftests_* modules import constants / helpers of selftests.py
  (selftests_plate_ports.py also selftests_plate_pml.py).

## Model
- Grids: triangles in 2D, the tensor grid only for 1D models (validate refuses a 2D tensor mesh); create dispatches on
  `ctx.domain.grid_kind`. Studies: eigenfrequency, frequency_response only (`step` does nothing).
- Materials: `tension` (writes tension_xx / _yy), `tension_xx` / `_yy`, `tension_xy` (optional array),
  `areal_density`, `loss_factor`, `bending_stiffness`, `plate_poisson_ratio` (optional arrays). The source slots
  `coupled_tension_xx / _yy / _xy` ADD to the material tension: a coupling writing the full prestress leaves the
  material tension unset.
- Plate switch = PRESENCE of the bending_stiffness array (any material sets it, 0 included), read once in `prepare`
  (`MembraneBase.bending`): 1D Hermite cubic, 2D interior penalty (quadratic only; ports on a face with D > 0 are plate
  ports, mode_projection refused). Validation mirrors it by the material's property key (`_materials_setting`), the
  estimates by `material_sets` (literal 0 = unset there: estimates only).
- Hole = rho_s <= 0, every tension component 0 (material + coupled) and D = 0: inactive elements, free outline. The
  test reads the slots, so a coupled tension can switch elements on: the active set follows `_read_coefficients`.
- Element order: own `discretization` Param (also for 1D Lagrange line elements; `definition_helpers.element_order`).
- 1D: per cell = per line element, dofs ascend (fem.line_fem); `axial_wavenumber` k_y adds T_yy (1 + j eta) k_y^2
  (T_xy ignored, validation warns), with D also 2 D k_y^2 and D k_y^4; forces per unit length along y; clamped /
  line_force / simply_supported on objects is refused.
- Hermite slope dofs are solved scaled (unknown h w': `LineHarmonicSystem.dof_scale` = 1 / h on slopes; K, C, M by
  congruence; `solve` / `ScaledEigenPencil.spread` return physical values): the SI w / w' rows are ~1 / h^2 apart. The
  line_fem slope basis is h x O(1), so a scale of h (not 1 / h) widens the gap to 1 / h^4 (plate_ports' strip eig
  failed on exactly that). `select_resonance` renormalises on the displacement dofs only (`_displacement_dofs`), since
  `fem.harmonic_resonances` peak-normalises over every entry.

## Solve
- System built lazily by `_ready_system(omega_reference)`, rebuilt when the PML stretch key changes (reference
  frequency, ring wave speed) or a feature adds a condition / load / port (`_discard_system`); coefficient changes
  only re-set coefficients (`materials_changed` of COEFFICIENT_ARRAYS, `inject_source`, incl. plate D and port media).
  PML reference: `pml_frequency`, else the eigen target, else the first frequency solved since
  `reset_frequency_response` (fixed over the sweep); the stretch is kept between calls (`_pml_of`), so a sweep does not
  rebuild `pml_ring`.
- `_publish` forms centroid phasors on the device on a cupy domain (`_device_phasor_maps`) from `self._device_values`
  when set (R6: the last direct cuDSS solve's `system.device_values()`), else one upload of `self.values`: code setting
  `self.values` any other way sets `_device_values` None.
- Eigen: `fem.harmonic_resonances.solve_system_resonances` on the 'eigen' role (real symmetric without loss / PML, else
  complex symmetric shift-invert); target 0 needs a clamp (K singular). Fields peak-normalised to 1 m real. Search
  'targets' / 'band': `MembraneBase._multi_shift_solve` builds the system once and runs `fem.resonance_search` on
  `_shift_solver(system)` (a Bloch cell overrides it with the reduced pencil): a new pencil variant overrides
  `_shift_solver`, not the search. Inertia counts only for a real pencil, through an `InertiaCounter` (kept factor,
  refactorised in place) released by `_multi_shift_solve`: a new caller of `_shift_solver` that counts releases it.
- Driven: `system.solve(omega)` = one complex factor per frequency via the 'linear' role's `factorize_host(symmetric=
  True, previous=)` (HarmonicPencil fixes the pattern: a cuDSS plan survives the sweep; fem sub file).
  `definition.MEMBRANE_LINEAR_ROLE` (frequency_response only; cusolver_qr refused; cuDSS only on triangles). A system
  rebuild releases the old factor first (`_release_factor`: cuDSS lies outside the cupy pool).
- sweep_method adaptive / modal: `plan_frequency_response` (after `reset_frequency_response`, which drops any old
  model) sets the PML reference and calls `system.plan_reduced` (fem sub file "Reduced sweeps"); fields only where
  `wants_fields` (study option 'fields'), probes / mode projections always through `system.linear_output`
  (`_probe_value` must stay linear in its values), `self.values` / `_publish` only with fields
  (`publish_frequency_response` at the end). Direct sweeps take the same policy (`plan_frequency_response` 'direct'
  only stores it). Any system rebuild / coefficient re-apply mid-sweep makes `system.solve` raise; ports are refused
  (validation and plan). Modal uses the 'eigen' role, so it is declared for frequency_response too.
- Validation applies the host SuperLU size rule (SPLU_* on n_nodes (+ n_edges for P2), holes included) to the factor
  the study uses, none for cuDSS; estimates count a cuDSS factor instead of the host factor as an external item
  (`definition_helpers.cudss_external`) plus its pool items: frequency_response run items (kept DevicePencil, solve
  vectors, kept solution; no CSR copies on top: the pencil's maps / transients hold the planned values, measured on
  membrane_guide_pml), eigenfrequency transient items of phase 'eigen solve' (fem.harmonic_resonances releases the
  shift factor; a real pencil's DevicePencil, a complex one's host-CSR copies, the device Krylov-Schur basis). Time:
  cuDSS priced at `CUDSS_FILL_SCALE[order]` (x BENDING_ENTRY_FACTOR for a plate; uncalibrated), host SuperLU at
  FEM_FILL_FACTOR (plate: factor x BENDING_SEPARATOR_WIDTH^3, solves x ^2); the memory estimate's L + U does not carry
  the plate x 4 yet.
- Build timer: `_ready_system` assembles itself (`ensure_assembled`) and adds its wall time to `assembly_seconds`,
  the seconds_counter of the recipe's build items (`definition._build_items`). Code calling `_ready_system` must not
  wrap it in another timed group; later work on the system (Bloch / per-frequency reduction, port DtN) stays in the
  study's untimed rest.
- Faces (triangles): a clamp beats a line force on the same face (one condition per face); forces on one face add.
  Pressure is dropped on PML ring elements; line forces / clamps on a PML face are refused / warned.
- Line objects: `clamped_line` / `simply_supported_line` (selection kind 'line', triangles only) reuse the face
  applies with `ctx.boundary_edges` (interior edges, both sides active) as Dirichlet edge conditions; with bending
  clamped_line also adds slope edges (Nitsche one-sided per side, no interior penalty there), simply_supported_line
  keeps the interior penalty. A line end on a periodic side warns unless the line spans the period.
- Mirror planes: `symmetry` (even) / `antisymmetry` (odd = the simply_supported hold + `mark_odd_mirror_faces`, which
  only tells squeeze films to vent). symmetry holds nothing on a film and only the slope with bending (2D Nitsche slope
  edges without the w Dirichlet, 1D the Hermite w' dof); a hold on the same face wins (warns), PML faces refused.
  Outer faces are MODEL mirror planes (`core.mirror.MIRROR_FEATURES`; outlines are not): a half model unfolds,
  integrals touching the face x 2, port powers x 2 per mirror wall. `FACE_FEATURE_SIGNS`: tension_xy undeclared (it
  flips under every mirror whatever the sign of w: never unfolded / combined). The squeeze film table is NOT doubled.
- Parity sectors: `core.mirror.sector_specs` swaps symmetry <-> antisymmetry on the sector faces (multi-face features
  split per face) and sets every port's `sector_drive`. Validation (`_parity_sector_issues`) refuses a mirror feature
  selecting a sector face with other entities and a clamped / simply_supported sector face. Combined results: probe
  `sector_rule` 'linear', port / mode_projection `sector_modes`.
- PML padding: `MembranePadding` as DomainRequirements; the ring faces must equal the pml faces
  (`fem.pml.pml_ring(faces=)` raises otherwise, e.g. another physics ringing more).

## Plate (bending_fem.py)
- C0 interior penalty on P2 Lagrange: interior edges of the active set get -{M_nn}[dn] twice + sigma [dn][dn];
  clamped faces / outlines are slope-held one-sided per active side (Nitsche, `slope_edges`), simply supported / free
  edges carry nothing. sigma = PLATE_PENALTY (16) x D / side height (> 6 needed on straight P2). Curved-element
  Hessians include the map's second derivatives. Eigenvalue error O(h^2): resolve the clamped boundary layer sqrt(D /
  T) (0.38 um for 80 nm SiN at 1 GPa).
- A tiny D is not a film near clamped walls: the layer moves the outer solution's wall inward by sqrt(D / T) even
  when unresolved (D = 1e-12 in the port tunnelling model lowers T11 by 17 %).
- plate_poisson_ratio enters M only; 1D ignores it.
- PML (plan F1): every assembler takes `stretch`; penalty x J (n . S^-2 n)^2 (`penalty_scale`: the plan's J / s_n^2
  would rotate the penalty's phase against the rest of the form); edge points from `EdgeSides.points`, the seam takes
  each side's own. Outer sides of PML faces are slope-held too. The strength uses the propagating plate wave
  (`MembraneCoefficients.wave_speed`); the evanescent partner only gains a phase, so a plate ring needs one element
  layer per cell (pml_layers >= pml_cells; `_bending_issues` warns): 8 layers across 24 cells reflected 0.79, 24
  layers 4e-3; a weaker stretch made it worse.
- Port edges carry no plate edge terms (the plate port DtN supplies V_n and M_nn), so no penalty there.

## Bloch cells (solver_fem.py)
- periodic_x / _y -> DomainRequirements.periodic_axes; eigenfrequency only. `fem.dofs.bloch_masters` on the ACTIVE
  DofMap, reduced P^H K P, P^H M P; pencil 'general' when P is complex (complex Hermitian: never the complex symmetric
  factor). With bending, `assemble_seam_bending` adds the seam edge pairs before the reduction (else the hi-side slope
  is a hinge).

## Ports and mode projections (ports.py)
- port: one outer face (no ring, not periodic, one per face), frequency_response only (D > 0 at the face: plate port;
  bending on with D = 0 there stays a film port). Span = active window edges on the face (one contiguous piece),
  profiles sin(n pi s / W). Free walls (a guide end on a window face with symmetry and no hold; `_free_walls`, decided
  in `_new_system` for ports AND projections since symmetry may apply after them) get sin / cos((n - 1/2) pi s / W);
  both walls free raises; an object outline wall stays clamped. Medium = mean of the port's elements: density / loss
  uniform to PORT_UNIFORM_RTOL, tension spread up to param `tension_tolerance` (raise it for a solid-coupled tension);
  above 1e-6 but accepted: RuntimeWarning + result extras (the study has no run-log channel for solver warnings). DtN
  values depend on omega: K carries the pattern as explicit zeros and `frequency_terms` adds the values per frequency
  (same reduced pattern). incident_power = sum over all ports, port_incident_power its own.
- Mirror walls (`_mirror_walls`, ports AND projections): `ports.unfolded_modes` numbers half mode k on the unfolded
  guide (+1: 2k - 1, -1: 2k; a mirror at s = 0 signs them, since the unfolded profile starts at the image wall).
  incident_mode(s) are in the unfolded numbering: a mode of the other parity raises at build unless `sector_drive`,
  which skips it and drives the matching one at 2 A. power_factor = 2 per mirror wall; amplitudes stay half-guide.
  Plain half-run results keep the half numbering.
- mode_projection (triangles): sample points located once at apply (outside the mesh: build error; keep few points
  on strongly graded meshes). Result 'xy' = the SHOWN solution; every driven solve also records a_n(p) (extras
  `frequencies` + `frequency_series`, frequency_response only). result_axis 'frequency' swaps the result for 'xy' vs
  frequency (raises before a driven run; averaged mask fixed at apply). Output / input ratio = two projections + a
  derived curve. Refused with bending (sin profiles are not plate modes).

## Plate ports (plate_ports.py, plan F2)
- Dispatch in `_new_system`: bending on and any port element with D != 0 -> `PlatePort`; `_set_port_medium` re-checks
  at every coefficient apply (D switching 0 <-> > 0 raises) and adds D (1 + j eta) and nu to the uniformity checks.
- Strip: Hermite cubic on the port edges' end nodes; wall kind from the window face's hold, else the outline edges at
  the corner node (Dirichlet + slope = clamped, Dirichlet only = supported, slope only = symmetry, none = free). A
  clamped wall holds w' by the 2D clamped edges' Nitsche terms (`_wall_nitsche`), so an unresolved boundary layer
  behaves alike on both sides. A wall on a PML face raises.
- Modes per frequency (cached for the last omega): dense eig of the 2 n companion pencil on traces scaled by ell (mean
  strip width; unscaled the blocks sat ~h^4 below w's and eig returned huge spurious roots as mode 1). The DtN keeps
  ALL 2 n modes (truncating breaks the exact power balance: R is complex per mode); mode_count only limits reported /
  incident modes, so it must cover the propagating ones for a power balance. Trace map = L2 projection onto the free
  strip space (real, frequency-independent, built in `assemble`).
- No unfolded numbering (modes numbered on the drawn guide), sector_modes None (sector_drive with a mirror wall
  raises), power_factor as film ports. Strip eig estimate is a lower bound (QZ does not reach the gemm rate). Reduced
  sweeps refuse plate ports.

## Squeeze film (squeeze_film.py, plan F4; the film is fem/squeeze_film.py, fem sub file)
- Feature squeeze_film_damping (monitor, footprint selection, triangles, eigenfrequency only; no gap_side: c, k are
  quadratic in w). Evaluated after every non-Bloch eigen solve, also after a search (merged list); a Bloch cell raises.
  The 1 m-peak mode in full numbering goes to a SqueezeFilm on the membrane's own mesh / order (`active` = footprint &
  window & not hole, per evaluation), m_eff on the pencil mass; rebuilt only when the active set or vents change.
- open_edges seals (`_film_seals`): window edges of clamped / simply_supported / symmetry faces minus antisymmetry
  faces, symmetry outlines, and Dirichlet edge holds with membrane on both sides; so a clamped outline with a hole
  outside vents, a clamped_line inside seals. A footprint reaching a PML ring raises at the first evaluation.
- resonance_table appends `GAS_TABLE_COLUMNS`. Not coupled into frequency_response (needs a monolithic film block).

## Fields and results
- state(): w_re / w_im / w_abs at the element centroid (1D: cell centre) of the SHOWN solution. Display w_* nodal (1D
  cell centres), tension_xx / _yy = material + coupled.
- field_probe = `HarmonicProbeRecord` component 'w', cleared by `reset_frequency_response`; eigen studies record
  nothing.

## Self-tests (`selftests.py`, model runs on the cpu; tolerances first estimates)
- membrane_line_loss: closed forms on the meshed length (a 1D window's float32 cell widths shorten L by 2.5e-8, x 2 Q
  near f_1 = 5e-5 of the peak). membrane_guide_pml variant gpu covers the cuDSS device pencil and device publish.
  membrane_port_tunnelling also holds the 't11_half' mirror run and the result_axis 'frequency' twin.
  `_plate_strip_frequency` brackets the clamped root n in q2 in (n, n + 1) pi / W.
- membrane_parity_sectors_driven compares rasters pixel-wise outside 4 h of the drive point and its image: the two
  pixel grids differ and a raw peak sits on the point force's log spike (it would measure pixel placement).
- membrane_squeeze_film_drum: a centre point puts a node at the mode peak; the half model clamps the rim through a
  'frame' difference object (a clipped circle's outline would include the x_hi chord) and seals the x_hi symmetry face.
- membrane_plate_port_film_limit uses D = 1e-16 (1e-12 shifts the walls, Plate above).
- Not built: error indicator, scalar outputs, mode_projection on plate modes, eigenfrequency with ports, line forces
  on line objects, a port with coupled (non-uniform) tension, a cuDSS self-test of the driven solve.
