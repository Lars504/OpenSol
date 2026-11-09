# open_sol/fem — conventions & gotchas

FEM on triangles (solver_plan S7.5, mesh_implementation_plan.md section 2) and tetrahedra (section "Tetrahedra",
igor3d_plan section 7). Triangle users: em_modes / em_fdfd_bor (physics/em), heat / electric_currents / electrostatics
(`diffusion.py`), solid_mechanics (`elasticity.py`), magnetostatics / magnetic_fields_ac (`magnetostatics.py`),
drift_diffusion (box method over assembly's `p1_*` weights); tetrahedra users: em3d FEMPhysics3D, heat /
electric_currents / electrostatics / magnetostatics_3d (`diffusion3d.py`), solid_mechanics_3d (`elasticity3d.py`); the
physics side lives in each physics sub file ("Triangle meshes"). A physics on triangles exposes `solver.field_dofs(name)`
(P0 for every state field and slot): the contract the transfers build on. Error indicators (`indicators.py`) exist for
all but drift_diffusion.

## Layout and imports
- Module contents: `structure_index.md`. Beyond it:
  - `selftests.py` (FEM_SELFTEST_CASES) is registered by model.study.register; new fem case lists append there.
  - Private helpers used across modules (changing them touches the users): elasticity / magnetostatics use assembly's
    `_element_groups` / `_to_csr`, harmonic `_coefficient_values`, elasticity3d assembly3d's `_element_list` /
    `_rule_degree` / `_volume_weights` and elasticity's `_fixed_field`, indicators / surface_mesh elements'
    `_inverse_2x2`.
  - Shared helpers: `assembly.nodal_average` (nodal recovery: strain / stress, B / H); `diffusion.py`'s
    `window_face_edges`, `ring_dof_locations`, `floating_region_count` serve every scalar triangle system
    (FEMDiffusionSystem, MagneticFEMSystem).
  - Box-method weights (drift_diffusion): `p1_edge_weights` = (m, 3) -K_(k+1)(k+2) of local edge k (scalar / diagonal /
    full coefficient, x centroid r axisymmetric): summed over an edge's elements = the edge-averaged FE / SG
    coefficient; `p1_lumped_weights` = area / 3, or area (r_k + 3 r_c) / 12.
  - `elasticity.assemble_stress_load(weight=)`: a point weight of degree <= 1 (generalized plane strain 1, x - x_ref,
    y - y_ref) stays exact at its rule degree 2 order - 1.
  - `magnetostatics.curl_quadrature` caches CurlBlocks (Q, weights, points, basis) per element chunk; magnetic_fields_ac's
    `block_mass` / `block_load` use the same rule (complex values allowed).
  - `surface_mesh.normal_displacement` = exact P1 / P2 trace of w = -u . n_out; `lagrange_field_function` affine only.
  - `squeeze_film.py`: numpy only at module level (fem.harmonic inside functions) so model.definition_helpers imports
    its constants cheaply.
- FixedPatternCsr / TangentAssembler[3D]: `full(data)` and `reduced(data)` share the cached index arrays (only data is
  new), so the pattern is value-independent; never sort / eliminate_zeros / sum_duplicates those CSRs in place
  (`full(copy_pattern=True)` / `tangent()` give own copies, e.g. for pencils). The reduced CSR is canonical: hand it to
  `DiffusionDirectSolver.factorize_host_operator(symmetric_known=True)`. The per-group bincount spans the group's slot
  range: element orderings that scatter a chunk over the whole matrix make each call O(nnz) per chunk.
  TangentAssembler3D caches rule gradients / weights only below `TANGENT_GRADIENT_CACHE_BYTES` (P2 ~6.7 kB per element;
  the int32 scatter map is always kept); `internal_force_and_tangent[_3d]` stays the one-off COO assembly (the
  self-test's reference).
- Device tangents (`TangentAssembler3D(device=True)`): the Green-Lagrange helpers take xp from their first array
  (`elasticity3d._array_module`): keep new code in them xp-generic. S0 is uploaded once per `initial_stress` object (a
  new S0 value needs a new StressField). `force` / `force_and_tangent` return the force on the host, the data on the
  device; `full` / `reduced` give cupyx CSRs, `reduced_values` the values in the HOST reduced pattern order (what
  `DiffusionDirectSolver.factorize_device_values` takes with `pattern.reduced_indptr / reduced_indices`),
  `full_product` a host vector, `tangent()` a host CSR. Device sums are atomic scatter_add (round-off differs from the
  host bincount). The 2D TangentAssembler has no device twin.
- `assembly._to_csr` only sums duplicates and keeps explicit zeros, so every assembly over one dof map (same free /
  fixed split) has one CSR pattern whatever the coefficients: the precondition of cuDSS plan reuse for Newton tangents
  and coefficient updates. Never `eliminate_zeros` an operator a factor will be refreshed with; sum terms with
  `assembly.pattern_sum` (root "Solver contract"; the 2D / 3D diffusion systems do). `assembly.CsrAccumulator`
  adds its 2e7-entry batches with `+` (bounded peak memory; its zeros are structural, so the pattern is stable).
- Default rule degrees are clamped to the largest Dunavant rule (6): an axisymmetric P2 load (degree 7) is not exact.
- fem <- numpy at module level (scipy inside functions); never model / physics. core only in `domain.py` / `domain3d.py`
  (core.constants / material_grid / metric), `diffusion.py` / `diffusion3d.py` (core.backend / constants / fv_diffusion
  BCSide / memory_estimate / monitors / solver_base; solvers.sparse_direct / multigrid inside functions), `renderer.py` /
  `renderer3d.py` (core.renderer), `transfer.py` (core.backend inside `FEMTransfer.apply`), `fixed_pattern.py` /
  `elasticity3d.py` (core.backend inside functions); `fem/__init__` imports none of these (import by module); core
  never imports fem. physics imports fem, never the reverse. Model side: `model/triangle_meshing.py` (the only reader
  of geometry shape internals for the mesher; sizes from `meshing.triangle_sizing`); `meshing.make_domain` of a
  'triangles' mesh returns `TriangleDomain(triangulate(...))`. selftests*.py follow the root selftest import rule.

## Mesh data
- Coordinates in m, origin at the physical-domain corner; the PML layer ring lies outside the window (< 0 / > size).
  `TriangleMesh` frozen (eq=False); triangles int32 counter-clockwise (build_triangle_mesh fixes orientation).
- Local edge k = nodes (k + 1, k + 2) mod 3, opposite node k; `edges` lower node first; `edge_signs` +1 when the local
  edge runs lower -> higher (Nedelec orientation); `edge_elements[:, 1]` -1 on the outer boundary. `connectivity` raises
  on an edge of > 2 elements.
- Curved edges: `edge_curve` >= 0 indexes `curves` (circle / ellipse `Curve`, t = angle of (a cos t, b sin t) in the
  turned frame); `midside_points` = P2 midside nodes, projected onto the exact curve on curved edges, chord midpoints
  elsewhere. Composite boundaries are pieces of their primitives' curves. Arcs (rounded_rectangle corners): `Curve(...,
  t0, span)`; build_pslg merges arcs on one whole curve, so `PSLG.curves` / `TriangleMesh.curves` stay whole curves.
  `mirrored_curve` maps t -> -t.
- Tags: `region` = index into `region_names` (non-consumed domain objects, spec order) of the LAST one holding the
  element centroid, -1 none; ring elements use the centroid clamped just inside the window (materials extruded). For
  any object use `triangle_meshing.element_masks`, never `region` (objects overlap). `edge_tags`: outer faces =
  boundary edges on the sides of the nodes' bounding box (the ring's outer side when ringed), object outlines = edges
  where centroid membership changes, line entities = their constraint edges. `point_tags`: point entity -> node (exact
  coordinates; points outside the window dropped).
- `pml_faces` uint8 bits `FACE_BITS` (x_lo 1, x_hi 2, y_lo 4, y_hi 8); corner blocks carry two bits and sit in the x
  strips. Ring = n_layers structured layers extruded from the window-edge nodes, quads split into right triangles (no
  quality guarantee: MeshQuality judges interior elements only).
- `digest` = sha256 of points + triangles; the built-in mesher is deterministic (no jitter, stable sorts): same model,
  same machine, same digest.

## Mesher (built-in)
- PSLG (`build_pslg`): window edges tagged x_lo / x_hi / y_lo / y_hi; equal curves merged; every crossing inserted
  (line-line exact, line-curve in the curve's unit-circle frame, circle-circle closed form, else chords + 2D Newton;
  tangencies give one point); ends within MERGE_TOLERANCE (1e-9 x longest window side) snap; collinear overlaps split
  and dedup; pieces outside the window dropped (objects reaching into the PML are clipped). Each piece is sampled by the
  size field (curves also by curvature_factor x radius of curvature, floored at h_min) in a canonical direction, so
  overlapping pieces give identical vertices; every count / ceil on a piece takes a round-off guard (copies with
  ulp-different ends, e.g. a mirror image, must not round to different grids, or Qhull later drops coincident split
  points). Input angles < 60 deg are flagged (`small_angles`, `acute_vertices`).
- Refinement (`mesh_pslg`) runs in normalised coordinates (window corner 0, longest side 1); nodes are only appended,
  never moved, until smoothing. Delaunay by Qhull (a dropped point raises), except rounds adding <= `_LOCAL_UPDATE_SHARE`
  (25%) nodes: `_local_delaunay` re-triangulates only the cavity (circumcircle enlarged by 1e-7 holds a new node); the
  patch must tile the cavity (`_tiles_cavity`: unique directed edges, same area, same boundary except along window
  sides), else the round rebuilds. A local update returns kept rows first and `_RefineState` carries their circumcircles
  and last round's bad / size_bad flags, so a round evaluates only new rows: any new per-triangle quantity must be
  carried the same way or computed for all rows.
- `_delaunay` adds a far frame triangle (`_QHULL_FRAME`, `_FRAME_RADIUS`) and drops its triangles (plain Qhull was
  20-50x slower on inputs with thousands of collinear hull nodes, e.g. a film top); the rest must cover the convex hull
  (area, 1e-9), else plain Qhull. Co-circular ties may differ from plain Qhull (digest changes). Cavity patches skip the
  hull check (`cover_hull=False`). Qhull options, translation or point order flip call times between ~0.05 s and seconds
  and flip ties; QJ is faster but drops triangles. Tried and dropped: scipy incremental `add_points` (2.4x slower),
  seed lattices / Poisson-disk seeds (slower overall), thinning spacing 0.35 R (more nodes) or 0.7 R (no faster).
- Edges keyed `low x 2^31 + high` (stable across rounds): the size field is evaluated only at midpoints of edges new
  since the last quality round (`_edge_sizes`). A segment is encroached when it is no Delaunay edge or an adjacent apex
  sees it at > 90 deg (strict; the apex test only on local edges along present segments; a segment shorter than h_min
  is never split for an apex). Encroached segments split first (midpoint; with exactly one acute end at a power-of-two
  distance from it: concentric shells, else acute crossings ping-pong down to coincident points; curved split points
  projected). Bad triangles: smallest angle < MIN_ANGLE (28 deg) or an edge > h(midpoint); quality-only refinement is
  skipped when the shortest edge < h_min or spans two pieces of an acute apex at equal distance (Shewchuk). Circumcentres
  encroaching a segment split it instead (quality-only: segments > h_min only); accepted ones are thinned greedily
  (largest circumradius first, spacing 0.5 R; `_spaced` vectorised, same set as plain greedy). MAX_ROUNDS (400) without
  convergence raises RuntimeError.
- Smoothing keeps connectivity, moves only free nodes, and keeps a move only when the smallest angle of the node's
  elements grows (compared as sine: 2 area / two longer edges), so the global minimum angle never drops. PSLG vertices
  get their exact input coordinates back; nodes within the tolerance of a window side snap onto it (the ring matches
  window-edge nodes by exact coordinates).
- SizeField: min(h_max, region sizes inside, h_k + (growth - 1) d over sampled sources: outlines, lines, points,
  curvature sizes per value), floored at h_min; `meshing.triangle_size_field` is the estimate's raster twin (smallest
  radius of curvature per ellipse).
- RawMesh keeps the PSLG vertices first in PSLG order (vertex_tags index nodes directly).
- Periodic sides (`TriangleSizing.periodic_axes` -> `build_pslg(periodic_axes=)`): `periodic_pslg` copies each side's
  vertices onto the opposite side (lo wins), `mesh_pslg` splits paired segments together (`_split`), so side nodes match
  exactly; gmsh: side lines `Transfinite Curve {...} = 2` (not `Periodic Curve`). A mirror axis matches through the
  reflection (only the other axis paired in the half mesh). Refused: a PML ring on a periodic axis, a curve along a
  periodic side. No periodic size field. `build_triangle_mesh` checks the partners of every declared axis.

## gmsh (optional)
- Never imported, linked or vendored: `find_gmsh` (OPEN_SOL_GMSH, else PATH); `mesh_with_gmsh` writes the same PSLG as
  a .geo in normalised coordinates (gmsh tolerances are absolute), one Line per segment (tag = index + 1), the window
  loop as surface, the rest embedded, the size field as a background view (.pos; exact, instead of Distance / Threshold
  fields), runs `gmsh -2 -format msh41` in a temp folder; line entity tag = PSLG segment. Nodes on curved lines are
  projected onto their curve; linear elements are requested (P2 midside nodes are open_sol's).
- gmsh reads user option files: every .geo (2D and gmsh3d) pins `PINNED_OUTPUT_OPTIONS` (scaling 1, ASCII, no
  parametric coordinates). gmsh can exit 0 after an error: failures quote its Error lines (`gmsh_errors`) and the node
  extent written (normalised), so a scaled or empty mesh shows. Numbers into .geo / .pos only via `gmsh_io.geo_number`
  (root numpy 2 repr gotcha; gmsh reads 'np.float64(...)' as an unknown option).
- gmsh3d size fields: `SizedFace` / `face_size_texts` = per refined plane face a Max of two MathEvals (constant size,
  unclamped ramp size + slope (distance - band); both out of the Min list); size points = radius-0 Ball fields with a
  linear ramp; the numpy twins in `model.meshing.tetrahedron_element_density` use the same formulas (keep them equal).
- gmsh's .geo lexer reads <= 1024 characters per quoted string: a longer MathEval silently truncates the .geo (later
  fields and the Background Field gone, h_max mesh, exit 0). Every MathEval goes through `gmsh3d._math_eval_rows`
  (ValueError above `GEO_STRING_MAX`), and gmsh3d fails on any 'Error' line. Shorten with Max / Min combining fields,
  never a MathEval reading another field (`F<n>`: gmsh -3 ran for minutes on a 15k-element cube).

## Elements and geometry
- Reference triangle (0, 0), (1, 0), (0, 1); L0 = 1 - xi - eta, L1 = xi, L2 = eta. P2 local nodes: corners, then 3 + k
  = midside of local edge k (full-mesh P2 dofs = nodes, then n_points + edge: the display numbering).
- Nedelec first kind, orientation lower -> higher node: degree 1 = sign x W_ab of local edge k; degree 2 = per edge
  (L_lo W, L_hi W), W = W_(lo, hi) (no signs needed between neighbours), then L_0 W_12, L_1 W_20. `nedelec_basis`
  applies the edge signs itself: dof maps carry none. Vectors = J^-T (covariant Piola), curls = reference curl / det J.
- Geometry order (D6): 1 = every element affine (the domain is the node polygon), 2 = elements with a curved edge
  mapped by P2 through `midside_points`. A DofMap carries it (`lagrange_dofs` default P1 -> 1, P0 / P2 -> 2;
  `nedelec_dofs`: the degree); P2 on order 1 is subparametric. Isoparametric P2 reproduces linear fields exactly on
  curved elements, quadratic ones only on affine elements.
- Quadrature degree = 2 x order (stiffness 2 x order - 2, +1 axisymmetric, loads +2), curved elements >= `CURVED_DEGREE`
  (6). Edges: Gauss on [0, 1] from the lower to the higher node, +2 degrees on geometry order 2.

## Dofs and assembly
- `DofMap.elements` = subset (bool mask or indices) or all; dofs numbered by sorted global id, so a full-mesh P1 map is
  the node numbering. `boundary_edges()` = edges with exactly one subset element; `edge_dofs(edges)` = dofs to fix on
  them (Lagrange nodes + P2 edge node; Nedelec tangential dofs).
- Coefficients: numbers, per-element arrays over the WHOLE mesh ((n_elements,), (n_elements, 2) diagonal, (n_elements,
  2, 2)), or callables f(points (m, q, 2), mesh elements (m,)) -> (m, q) / (m, q, 2, 2) (pass `degree`). Edge
  coefficients: number, per mesh edge, or f(points (k, q, 2), edges (k,)). A constant tensor goes per element
  (`np.broadcast_to(K, (n_elements, 2, 2))`): a bare (2, 2) array reads as per-element values.
- Assembly over chunks of `CHUNK_ELEMENTS`, straight and curved groups with their own rules. Element blocks are batched
  matmuls (`_tensor_blocks`, elasticity `_material_blocks` / `_basis_product_blocks`): tensors only broadcast, never
  materialised to (m, q, 2, 2); a shared rule broadcast, never copied per element. They differ from einsum at ~1e-16.
  Basis values only: `elements.lagrange_values`.
- Edge terms: the geometry part of the rule is cached per (edge set, degree) in `DofMap.edge_quadrature` (up to
  EDGE_QUADRATURE_ENTRIES), so repeated loads pay only the coefficient. Cached arrays are read-only: an edge coefficient
  callable must not write into its points / edges. 3D face terms likewise keep (face-local dofs, areas) per face set in
  `DofMap3D.face_quadrature` (FACE_QUADRATURE_ENTRIES, read-only). `eliminate_dirichlet` lifts fixed values into the rhs; eigenproblems
  use `restrict`.
- `assemble_gradient_coupling(nedelec, lagrange)` needs both maps on the same elements and geometry order.
- Periodic / Bloch (2D twins of mesh3d / dofs3d, Lagrange only): `mesh.periodic_node_partners`, `dofs.periodic_dof_partners`,
  `dofs.bloch_masters(dofs, {axis: k})` -> (master, phase e^{-jkL} per crossed period; corners chain both axes),
  `dofs.bloch_prolongation` -> (columns, P), x = P x_r (float64 when every phase is real); callers form P^H A P.

## Domain, sampling, transfers
- TriangleDomain: `grid_kind` 'triangles', MaterialGrid (n_elements,) (D12), `coordinate_grids()` = float32 centroid
  arrays (ring centroids clamped as `element_masks`), so `BuildContext.eval_field` works unchanged; `dx` = shortest
  edge. No cell_widths / pos_to_cell.
- Session contract (model sub file "Per-mesh Session"): a triangle physics' renderer gives `fem_field(component)` ->
  (Lagrange DofMap, real coefficients, host or device); the Session rasterises (`TriangleDomain.located`), triangulates
  (`display_triangulation`; node values by `node_values` = coefficients scattered by global id, never by locating the
  nodes) and reduces derived values over `quadrature_samples(mesh, 4, window elements)`.
- `TriangleDomain.located(points, order)` / `quadrature_located(degree, elements)` remember each point set (blake2b; up
  to LOCATE_CACHE_POINTS) as a `LocatedPoints` (location + Lagrange basis per order, gradients when read);
  `evaluate(location=LocatedPoints)` reuses them. Everything is READ-ONLY (copy first); `locate` returns read-only
  elements. `raster_sampler` / `locator` are small module caches for scripts and self-tests; Session code never keeps
  basis arrays at module level (root "Selftest reuse").
- PointLocator: uniform bucket grid (~1 element per bucket) over element boxes (curved boxes include the midside node);
  affine reference coordinates, Newton on the P2 map for curved elements (a miss = inf); deepest containing element
  wins, ties to the lowest index (deterministic on shared edges). Locate with the dof map's geometry order. On strongly
  graded meshes one bucket can hold ~40 % of the elements (seconds for ~10^4 points in the refined zone): locate few
  points there or read values from the dofs.
- Transfers (`FEMTransfer.apply(array, out=None)`, flattened C order, tensor side = full (nx, ny) grid incl. padding):
  interpolate clamps into the source window (padding / ring get edge values). Conservative counts window elements and
  cells only: tensor -> P0 samples each element with a subdivided degree-2 rule (<= `MAX_SUBDIVISIONS`) and, when the
  target covers every window element, rescales each cell column to the cell's exact measure (exact window total);
  triangles -> tensor bins degree-4 (6 axisymmetric) samples (exact total); triangles -> triangles only to quadrature
  accuracy. `axisymmetric=True` weights by r. A conservative target on triangles must be P0. Window-side points are
  interpolated at the transfer's inset position (MERGE_TOLERANCE x side inside). The build uses them through
  `model.mesh_transfer.TriangleTransfer` (float32 in / out, source space by array length).

## Scalar diffusion (`diffusion.py`, stage T4)
- FEMDiffusionSystem: unknown u (heat: T - T0) in the full-mesh numbering (`values`), host float64. `assemble` builds
  the Lagrange map of the ACTIVE elements (window elements with conduction, loss or capacity; geometry order = element
  order), adds loss / capacity / Robin masses, fixes Dirichlet edge dofs and held regions (later conditions win shared
  dofs by a sequence counter; `set_face_value`, the value of time t, keeps the face's sequence so corner ownership
  does not move between the build and the first step; 3D alike), checks every free connected region for an anchor
  (topology: dofs of loss / capacity elements or Robin edges, else reduced row sum > 1e-10 x diagonal; as 3D) and
  factorises the reduced CSR through a PCGSolver-protocol solver (None / SparseDirectSolver -> DiffusionDirectSolver).
  Dofs of no active element keep their value; PML ring dofs get the field at their clamped window position after every
  solve.
- Factor reuse: a condition change of the same kind (Robin: same h; `_same_operator`) keeps the factor; Neumann values
  are read every solve, a Dirichlet value / Robin external re-runs `_refresh_constant_terms` on the next solve (lift by
  the cached free x fixed CSR); only kind / h changes, held regions, losses and materials re-assemble. `set_loss`
  replaces every add_loss (potential's electrolytes). Stop reference: transient 'initial_residual', steady 'rhs'.
- `EdgeCondition` h / external may be arrays per mesh edge (3D: per mesh face; heat radiation): code reading them
  goes through `robin_assembled(condition)` / `robin_anchoring(entities, condition)` (and indicators `_on_edges`),
  never `condition.h > 0.0`; `_same_operator` compares arrays by value.
- Outer faces act on the WINDOW EDGE (`face_edges`), never on the mesh's outer face tags (ring's outer side). Object
  outlines = `BuildContext.boundary_edges`; a flux there is a line source feeding both sides (unlike tensor perimeter
  cells inside the object).
- Axisymmetric: every integral per radian (weights r); point sources = whole-ring totals / 2 pi; physics multiply held
  fluxes by 2 pi. Reactions (A u - b over fixed dofs) balance the sources exactly. Per outer face use `face_flux(face)`
  (counts only dofs its condition won, so a shared corner counts once); `inward_flux(face_edges, condition)` counts
  every dof of the edges.
- State (D12): per element the centroid value (`centroid_values`) and element means of gradients / dissipation
  (`element_integrals`, r-weighted axisymmetric); the nodal unknown goes to the renderer. FEMDiffusionPhysicsBase holds
  the shared physics API (face BCSides, outline conditions with handles for values of t, held / initial values,
  losses, node probes, field_dofs, set_conformal_boundaries a no-op).
- fem_diffusion_estimate: entries per dof 7 (P1) / 12 (P2), FEM_FILL_FACTOR 2 on the 5-point SuperLU heuristic,
  uncalibrated. Both FEM estimates go through `fem_system_estimate`, which follows `_factorize`: the run-backend CSR of
  the reduced operator is kept only by pcg (+ CG vectors, multigrid hierarchy; setup = build extra); SuperLU (`.get()`)
  and cuDSS (`factorize_device_csr`) take it as a build temporary. Operator and solver are sized on the active dofs
  (`active_fraction`: the physics' `painted_fraction` of its conducting materials from the mesh estimate's
  `material_elements`: triangle and tetrahedron estimates both carry it (meshing `_material_elements_2d` /
  `_material_elements`, the window elements only); 1 where it is empty). `refactorizes` (a coefficient following a state, a Newton / B-H loop):
  transient phase 'refactorisation' (a new operator copy, + the multigrid setup / cuDSS refresh peak).
- `direct` of both FEM estimates is `definition_helpers.linear_solver_direct` (the resolved non-cuDSS back end name,
  truthy; True = SuperLU): `direct_factor_item` books a MUMPS / PARDISO factor (`host_direct_name`: a name of
  `host_direct.HOST_DIRECT_BACK_ENDS`, never a literal list) at the SuperLU heuristic x
  `host_direct.superlu_bytes_scale` (real Cholesky / LDL^T; float32 values when mixed; uncalibrated) and
  `fem_system_estimate(host_direct=)` adds its own copies (`host_direct.solver_host_bytes`); SuperLU items unchanged.
- cuDSS diffusion factor (`cudss_diffusion_estimate`, shared by the FEM estimates and the tensor DiffusionSystem
  callers, `input_copy=False` there: A_solve is kept): the device route keeps the index maps
  (`cudss_factor.index_map_bytes`; one refresh's peak: `refresh_peak_bytes`); the lower-triangle CSR copies share the maps'
  indices, so only their values count beside them), the lower values, diagonal, rhs buffer and row scale; build extra
  = the first plan's peak (the transpose map's int64 keys with the input values and their equilibrated copy; the kept
  arrays planned after it subtracted: heat_fem_cpu_gpu measured ~39 B per entry); transient 'linear solve' =
  `cudss_factor.solve_peak_bytes` (mixed: the GMRES basis); the factor itself external (`factor_device_bytes`,
  dimensions 2 / 3). Entries per row of every Lagrange estimate: `lagrange_entries_per_dof(mesh, order)`, counted
  from the mesh estimate's nodes / edges / faces / elements (each dof + 2 x the dof pairs sharing an element: P2
  triangles ~11.5, measured 11.2 on heat_fem_cpu_gpu), never a constant.
  `cudss_back_end(solvers)` reads the resolved 'linear' back end. `fem_time_system` = (dofs, CSR entries per row) handed to `diffusion_time_estimate(entries_per_row=)`:
  real unknowns x entries / stencil entries, never inflated "equivalent cells" (LU curves grow ~n^1.5 / ~n^2).

## Factor entry count (`dissection.py`)
- `element_dissection_pairs(connectivity, node_points)`: the entries of an elimination order built on the mesh, per
  region the smaller of two (bottom up over the cut tree; per level by bincounts / reduceat, no Python loop over
  regions). Cut: the plane through the mean centroid, normal to the axis of largest spread; upper half = the elements
  wholly above it (their lowest coordinate), never by centroid: the tetrahedra of one swept prism column share a
  footprint but not a centroid, so a centroid split tears the column (and zigzags through a structured row) and
  the separator thickens (1.56x cuDSS's entries on sin_square_membrane_modes_3d, with or without sweeps). Separator
  = unmarked nodes both halves share (for P2 one surface of nodes; upper-side nodes next to a lower-side node are
  two node layers: 2.0x), border = earlier separator nodes the region's elements touch, S (S + 1) / 2 + S B. Sweep:
  the region's nodes in coordinate order along that axis, border last, envelope rows (first neighbour to the node;
  border node: first neighbour to the end), both directions; it wins on a strip without borders, little on a slice
  between two cuts (the border at the sweep's start is carried to its end). Node pairs of one factor triangle: a vector operator stores (dofs per
  node)^2 per pair. Both orders are ones a solver could use, so the count bounds a good ordering from above and
  nothing in it is fitted; a count below cudss_lu_entries / eigen_lu_entries (summary.json) would mean cuDSS orders
  worse than this, a lead on its ordering options.
- The "elements" may be any clique cover of an operator's pattern with a point per unknown (repeated members
  allowed): `physics/em/memory_estimates._bor_factor_entries` passes the curl points of the tensor BOR operator.
- `MeshDissection(mesh)` (triangles and tetrahedra; order 2 = nodes + edges through `element_edges`; `nedelec_pairs`
  on tetrahedra: edge midpoints, + face centroids at degree 2, as the dof sites) counts per element order on first
  use; equal per mesh digest. `meshing.built_mesh_counts` carries it ('dissection'), `with_built_counts` attaches it as
  `estimate.dissection` (None on unbuilt meshes; the GUI's built counts carry it too, so its first count runs on the GUI
  thread), consumers: `physics/mechanics/definition.factor_fill` -> `cudss_factor.FactorFill`,
  `physics/em3d/definition_fdfd3d.nedelec_factor_entries` (FactorFill and host SuperLU bytes).

## Error indicators (`indicators.py`, mesh plan A3)
- eta^2 per mesh element (float64), energy-norm weighted like the tensor A2 hooks (1 / c, 0 where c is 0); hooks
  `error_indicator()` of FEMHeatPhysics / FEMPotentialPhysics (`diffusion_indicator`), FEMModePhysics
  (`fem_modes.mode_error_indicator`), SolidMechanicsFEMPhysics and MagneticFEMPhysics (their own jumps).
- `edge_sides(mesh, edges, inside, geometry_order)`: side 0 = an element of the set, side 1 the neighbour if in the set
  (else -1), Gauss points of `edge_geometry`, unit normals out of side 0. Evaluate a side by
  `sides.reference_points(side, rows)` (with that element's edge sign: both sides see the same physical points) and
  `lagrange_at` / `nedelec_at`. `add_edge_terms`: h_E x share (1/2 two-sided, 1 one-sided) x per-side compliance.
  physics/membrane/bending_fem uses `edge_sides` too: keep its side / normal conventions.
- `diffusion_indicator`: free elements = active minus held; element residual h_K^2 / c_K int R^2 (P2 Hessian from the
  affine corner map, + (C grad u)_r / r axisymmetric; point sources not counted); edge jump minus the edge condition's
  Neumann / Robin value (later sequence wins); Dirichlet edges and one-sided edges next to held elements carry none,
  insulated outlines of non-conducting regions are one-sided.

## Tetrahedra (igor3d_plan section 7; host numpy, scipy inside functions)
- The 2D modules stay 2D: no dimension switches inside them; dimension-free helpers are imported (assembly3d re-exports
  `eliminate_dirichlet` / `restrict` / `DirichletSystem`; diffusion3d uses diffusion's EdgeCondition as FaceCondition,
  `floating_region_count`, `element_means`).
- Lagrange on tets (`lagrange_dofs_3d(mesh, 0 | 1 | 2)`, P0 = the field_dofs map of per-element state): coefficients
  number / (n,) / (n, 3) diagonal / (n, 3, 3) / callable -> (m, q[, 3[, 3]]); face coefficients number / (n_faces,) /
  f(points (k, q, 3), faces). `elements` = None / indices / bool mask; matrices keep the full numbering (callers
  restrict). Face integrals use only the face's own dofs in `DofMap3D.face_local_dofs` order (nodes ascending, then
  edges ab, ac, bc), so outer and interface faces work alike. `rule_values_3d` returns sorted elements: the order the
  gradient load reads. Default degrees: stiffness 2 (order - 1), mass 2 order, loads order + 2 (varying callables:
  pass degree).
- Faces: `interface_faces(mesh, member)` = faces with exactly one member element; `window_faces(mesh, face)` = interface
  faces of window elements on that window plane. Object interfaces: `TetrahedronMesh.object_faces`, made by
  `model.tetrahedral_meshing.tetrahedralize` from `element_membership` (contains_points_3d at `inset_centroids()` +
  entity majority) = the same function as the builder's element masks, so tags and masks never disagree;
  `BuildContext.boundary_faces(entities)` reads them. `face_tags` stays outer-only: boundary faces on the node bounding
  box (the PML shell's outer side), not the window faces.
- Periodic sides: `mesh3d.periodic_node_partners` (ValueError when sides do not match) and `dofs3d.periodic_dof_partners`
  (Lagrange P1 / P2, Nedelec 1 / 2; Nedelec raises for pairs not keeping node order, which would need dof signs).
  `dofs3d.bloch_masters` chains the axes (phase e^{-j k L}, L = node extent; within 1e-15 of real made real); em3d
  `fem3d.bloch_identification` and the solid's `bloch3d` build on it, prolongation stays the caller's. The Kuhn split
  satisfies both; gmsh meshes only with `periodic_axes` (Lagrange only: node order is the mesher's).
- Swept prism layers (`prism_layers.py`; model sub file "Swept prism layers"): node k n + i = (point i, z_k), element 3 p
  + j; `split_prisms` cuts each quad face from its smallest-key vertex, so neighbours agree with any strict order (keys
  must differ within a prism). `periodic_order_key` gives hi-side nodes their partner's key: Lagrange partners hold on
  x / y / z; Nedelec x / y need the same node order on both sides, which the 2D mesher does not give (em_fdfd_3d refuses
  periodic x / y on swept meshes). Split tets can have obtuse dihedrals, so `n_poor` may count some.
- `structured_box_mesh` takes any increasing edges (Kuhn split): each tetrahedron is an orthoscheme, every dihedral <=
  90 deg at any aspect (no slivers; only the radius ratio drops), periodic pairing kept under any grading.
- `TetrahedronQuality.n_poor` (solver sliver warnings, GUI) = radius ratio < POOR_RADIUS_RATIO AND an obtuse dihedral
  (`largest_dihedral_cosines` < OBTUSE_DIHEDRAL_COS): the ratio alone flags graded orthoschemes; `smallest` stays the
  plain minimum ratio.
- PointLocator3D: k-d tree passes over the previous misses (FIRST_CANDIDATES 8, LOCATE_CANDIDATES 24,
  FALLBACK_CANDIDATES 128), then brute force over elements whose box holds the point (one numpy pass per point: many
  misses on a huge mesh are slow). A point on a face between equidistant centroids may land in either.
- `TetrahedronDomain.locate` remembers each point set (blake2b; up to LOCATE_CACHE_POINTS) and returns READ-ONLY arrays;
  callers copy first, renderers keep no location cache of their own.
- `renderer3d.LagrangeSampleRenderer(physics, name, field_infos, time_label)`: the physics provides `domain` and
  `lagrange_field(component)` -> (DofMap3D, coefficients); device_field = centroid values. `evaluate_lagrange_3d` gives
  NaN at element -1.
- Orientation: element nodes stored ascending, so local edges / faces agree in both elements and Nedelec needs no
  signs. Never reorder `tets` rows after building. det J may be negative: integrals use |det J|, curls divide by the
  signed det J.
- Nedelec degree 2 = 2 per edge + 2 per face (L_a W_bc, L_b W_ca of the ascending face), no interior dofs. Every entity
  keeps its dof (no element subsets): callers pin the dofs of excluded elements. `assemble_nedelec_3d` coefficients:
  only diagonal tensors.
- Quadrature: centroid (1), 4-point (2), collapsed Gauss-Jacobi products (3-8: n^3 points, not symmetric).
- `region` = the last solid (gmsh3d `region_of`) holding the element centroid clamped into the window; gmsh meshes keep
  `entities` and a volume whose tags agree on >= ENTITY_MAJORITY (0.9) takes that tag throughout (faceted curved
  interfaces misread centroids). `entity_majority` is one bincount over the cached `_entity_groups`, never a loop over
  entities (it runs per object of every selection: objects x volumes x elements). `inset_centroids()` is cached and
  read-only (a `replace` copy recomputes).
- `fem.assembly.congruence(T, A)` = T^T A T (a diagonal T scales entries in A's pattern, explicit zeros kept): use it
  for electrode / dof maps instead of `T.T @ A @ T` (two SpGEMMs).
- gmsh3d result caches: in process the last GMSH_CACHE_SIZE (4) clipped results per (executable, .geo text, box)
  (copies handed out, failures never cached; tags and object faces rebuilt from current geometry): anything gmsh's
  output depends on must be in the .geo text (the key); a gmsh binary replaced at the same path needs a new process. On
  disk (`OPEN_SOL_MESH_CACHE`, `mesh_solids_with_gmsh(disk_cache=)`, '' off): the RAW `read_msh_tetrahedra` arrays
  (before `_clipped_mesh`, so clipping changes are never stale), `.npz` keyed by sha256 of (`_MESH_CACHE_FORMAT`, gmsh
  path, size, mtime, .geo text): bump the format when cached arrays change meaning. Temp file + `os.replace`; a bad file
  is re-meshed; LRU by mtime above MESH_CACHE_MAX_BYTES. Self-tests clearing the memory cache count `gmsh_run_count()`.
- Composite sizes (`SizedRegion`, model: `tetrahedral_meshing.sized_regions`): a refined composite is a field tree,
  never sizes on its inputs (a hole would be refined over its whole volume): union Min, intersection Max, difference
  Max(base, inverted subtracted); inverted solid = Min over one MathEval per face of size + slope max(depth, 0);
  inverted composites by De Morgan. Faces of a subtracted solid at / beyond the base's box side (`_kept_faces`) are left
  out of its depth (else a hole through a film is refined by its own top / bottom). Inner nodes stay out of the
  background Min list. A cylinder leaf is a ramped MathEval of its exact outside distance, never the Cylinder field (its
  step left a refined ring's outer surface at h_max). `SizedRegion.sizes` is the numpy twin; the estimate rasterises the
  mask with the same ramp instead.
- Surface sizes (`geo_text(sized_surfaces=)`, RefinementSpec.where 'surface'): root = Max(region tree with
  `ramped=True` leaves, inverted tree) = size + slope |distance to the surface|. Ramped leaves are MathEvals of the exact
  outside distance (`_outside_distance_text`), never Box / Ball / Cylinder fields. CSG min / max distances (inside a
  union the depth is a lower bound: inner faces never refined). Twin `SizedRegion.surface_distance` / `surface_sizes`
  (the estimate uses it). Numbers at 12 digits (`_short_number`): a cylinder is ~820 of GEO_STRING_MAX characters.
- gmsh's Cylinder field is VIn where |(X - centre) . axis| < |axis|^2: the axis vector is the HALF length
  (`_size_field` writes ZAxis = height / 2 at mid height).
- gmsh3d writes OpenCASCADE .geo normalised to the longest box side, fragments everything at once (conforming
  interfaces, the window box its own fragment), saves every fragment and keeps the volumes whose tet centroids lie
  mostly inside the box (`_inside_fragments`). Never select inside gmsh (BooleanIntersection per solid and `Volume In
  BoundingBox` both left outside parts in the mesh). A kept mesh beyond the box or not filling it exactly raises; any
  Error line fails the run. Sizes: Ball / Box fields ramp over Thickness (h_max - size) / (growth - 1); the Cylinder
  field has no Thickness (jumps to h_max at its surface); + MeshSizeFromCurvature (2 pi / curvature_factor), Netgen;
  needs gmsh >= 4.8. `periodic_axes`: `_periodic_rows` pairs hi / lo surfaces of the same translated box (`Periodic
  Surface ... Translate`); a periodic axis must carry no PML shell (ValueError). `Solid.edge_size` (electrode edges,
  RefinementSpec.where 'edges'): one MathEval per box edge / cylinder rim (only arithmetic and Sqrt; max(a, 0) as (a +
  Sqrt(a a)) / 2), clipped by MeshSizeMax: no Distance field (its sampling option was renamed between gmsh versions).
  `Solid.edge_distance` is the estimate's numpy twin.

## Scalar diffusion on tetrahedra (`diffusion3d.py`)
- FEMDiffusionSystem3D mirrors FEMDiffusionSystem (same owner protocol), differences: full-mesh numbering throughout
  (free = dofs of active elements minus fixed), tensor (n_elements, 3, 3), outer faces on `window_faces`,
  `add_surface_condition(faces, condition)` (a face with an active element on either side; a Neumann value on a
  two-sided face is a sheet source feeding both sides), `add_point_source(element, reference, power)` (exact delta;
  dropped in inactive elements), `add_fixed_dofs` / `set_fixed_dofs` (a set takes a new sequence and so wins shared
  dofs; empty ids = none), `solve(source, extra_load=)`, PML shell dofs at clamped positions (`shell_dof_locations`,
  1e-9 x side inside on every axis). Reactions / face_flux as 2D; factor reuse as 2D (`_note_change` /
  `_refresh_constant_terms`: a Dirichlet value or Robin external of the same h only refreshes lift and loads).
- Anchors by topology (`floating_region_count(reduced, anchored_rows)`): loss / capacity / Robin mass row sums are ~h^2
  a / k of the stiffness diagonal (~1e-12 for a weak loss on 0.3 um cells), below the ANCHOR_RTOL row-sum test, which
  refused such systems as singular. The 2D system passes the same topology anchors.
- `diffusion.FEMDiffusionPhysicsCommon` is the dimension-free physics wrapper (records, face BCs on `_faces`, held /
  initial / loss terms, field helpers) under FEMDiffusionPhysicsBase (triangles) and FEMDiffusionPhysicsBase3D.
- FEMDiffusionPhysicsBase3D: BCSides on FACES_3D, surface conditions, held / initial / loss on element masks,
  `add_point_source_at` / `add_probe_at(point)` (located by `domain.locate`), `field_dofs` = P0, `nodal_field` /
  `element_field` for lagrange_field.
- `fem_diffusion_estimate_3d`: `lagrange_entries_per_dof` (P1 V + 2E, P2 V + 7E + 12F + 6T per V + E), FEM_FILL_FACTOR_3D (2 / 3) on
  `host_real_lu_bytes(n, dimensions=3)`, uncalibrated (MUMPS / PARDISO scaled from it by the 2D `direct_factor_item`);
  the rest is the 2D `fem_system_estimate` (cuDSS dimensions 3, `active_fraction` from the TetrahedronMeshEstimate's
  material_elements).

## Vector elasticity on tetrahedra (`elasticity3d.py`)
- Interleaved vector dofs 3 d + component on a full-mesh scalar DofMap3D; Voigt rows (xx, yy, zz, yz, xz, xy) with
  engineering shear; material = (n_elements, 6, 6) per element. Matrices keep the full vector numbering. The stiffness
  COO is summed into CSR every 2e7 entries (`assembly.CsrAccumulator`, P2: 900 entries per element), int32 indices below 2^31.
  `stretch=` (elastic PML: callable -> complex (m, q, 3)): gradients / s per axis, weights x s_x s_y s_z, rule + 2;
  unconjugated, so complex symmetric.
- Piezo pieces (the block operator is physics/mechanics/piezo_fem3d.py): `assemble_piezo_coupling_3d` = CSR (3 n, n)
  of int B^T e^T grad N (K_uphi; K_phiu = its transpose); the phi block is `assemble_lagrange_stiffness_3d` with the
  full eps0 eps; `element_gradients_3d` returns sorted elements.

## Complex scalar harmonic systems (`harmonic.py`, `harmonic3d.py`, `pml.py`, `harmonic_resonances.py`; plan S1, S2, S8)
- -div(a grad u) + (b + j omega c - omega^2 m) u = f; K, C, M on one fixed pattern (`HarmonicPencil`), one factor per
  frequency via `factorize_host(..., symmetric=, previous=)` (cuDSS plan reuse). On a cuDSS back end `_factorize` hands
  the reduced K, C, M to the device once per assembly (`_device_terms`: DevicePencil) and each frequency calls
  `factorize_pencil` (no host A(omega)); frequency_terms (ports) go in through `extra=` at slots mapped once; a block
  outside the pattern, or unsymmetric with a symmetric pencil, takes `factorize_host`. `HarmonicPencil.of` takes a
  shared pattern as is, else the pattern_sum union; index maps from one pass over the stored entries (scipy fancy
  indexing is slow at 1e7 entries). Active elements = `set_active(mask)`, not a > 0. Robin: n . a grad u = value -
  (admittance + j omega damping) u.
- MUMPS / PARDISO back ends take the host route (`device_pencil` is None off cuDSS): `factorize_host(previous=)`
  refactorises on the kept analysis while the pattern holds. Factor errors become ValueErrors at `_factorize`:
  CudssAccuracyError / HostDirectAccuracyError (a RuntimeError, caught first) -> `_accuracy_error` naming the library
  (`host_direct.factor_label` of the runtime's back end), any other RuntimeError (zero pivot) -> the singular text;
  HostDirectMemoryError (a ValueError) passes as is. Host factors probe at factorize, so `_solve_free` never expects
  their accuracy error (caught all the same; no fallback: only an 'auto' cuDSS runtime falls back).
- PML stretch = the pml3d map with beta fixed at build from a reference frequency (`pml_ring` / `pml_shell`), so K and
  M do not depend on omega; PML faces held at 0 on their outer side by default (`pml_outer`). `stretch_derivatives`
  (C1 profile) serves fourth-order forms (membrane plate). Only ring / shell elements (a node beyond a layered face) are
  integrated with the stretched callable at the raised rule (+ PML_EXTRA_DEGREE); the rest keep the plain coefficient
  at the default rule (3D `SplitCoefficient`, 2D `assemble_stiffness` / `assemble_mass(elements=)` parts summed by
  pattern_sum).
- Face conditions act on the mesh's outer edge / face tags, so a wall continues along the other faces' PML layers.
  Edge, face and point terms are not stretched: keep Robin / Neumann boundaries and point loads out of the PML.
- `eigen_pencil` refuses damping (quadratic in omega). `solve_system_resonances(system, target_frequency [Hz], ...)` ->
  HarmonicResonanceSolve. Several shifts (`resonance_search.py`): a caller's `solve_at(f, n)` (+ optional
  `count_below(omega^2)`), merged by Re omega, duplicates of two shifts dropped by Hermitian-inner-product independence
  (degenerate pairs kept); info target_frequency NaN + targets / band / expected_count / shifts. Band trust radius = the
  farthest kept resonance of a shift: complex pencils with large Im omega may hide resonances outside it (warning).
- Per-frequency terms never re-assemble or change the reduced pattern. Code outside fem uses only the public API of
  `HarmonicSystemBase`: `frequency_terms(omega)` -> (CSR added to A(omega), its pattern already in K as explicit zeros,
  or None; load or None) (membrane DtN), `set_frequency_load(full)` / `set_held_offsets(full)` (survive re-assembly;
  pressure_acoustics' background), `solution` / `factor` / `fixed_value`, `ensure_assembled()`. `load_vector`,
  `operator` and `reactions` hold pencil terms only. Subclass hooks (`_assemble_terms`, ...) for subclasses only; one
  outside fem: `physics/membrane/solver_line.LineHarmonicSystem`.

## Squeeze film (`squeeze_film.py`; runtimes: physics/mechanics/squeeze_film.py tets, physics/membrane/squeeze_film.py)
- `SqueezeFilm` on a ScalarHarmonicSystem (a = H^3 / (12 mu_eff), c = H / p_a; Veijola mu_eff); responses per unit
  velocity, F = int p conj(w) dA, c_psi = -Re F, k = omega Im F. `active` restricts the film; vents default to its
  `active_boundary_edges` and must be boundary edges of it; the rest seal. A per-element gap is checked on active
  elements only. Ambient None = incompressible: one factor for every mode; a closed incompressible film raises.
- w as coefficients of the film's OWN full Lagrange numbering goes through the film mass matrix (exact on
  isoparametric elements). Anything else (number, P1 trace under a P2 film, callable) goes through the volume source on
  the ACTIVE dof map (never the full map: a boundary dof's support outside the film would count). `gas_damping_totals`
  sums films (c, k, gamma add); `film_table` is the shared 'table' result.

## Reduced sweeps (`reduced_harmonic.py`, experiment plan A1 / A2)
- Contract in the module header: terms / loads = fixed matrices / vectors x `PowerCoefficient(factor, power)` (or any
  analytic coefficient; a `HarmonicLoad(vector_of=)` or non-analytic coefficient turns moments off), a caller
  `factorize(omega)` -> factor with `solve_host`, never released by the model (one factor alive: refactorise in place).
- `HarmonicSystemBase.plan_reduced(omegas, method)` from `reduced_terms()` (in the system's own scaled unknowns);
  `solve()` raises when the operator, `_load` or `_fixed_value` objects changed after planning. `frequency_terms` are
  refused. Modal: raw eigenvectors of `eigen_pencil()` (refuses damping) + A(omega_c)^-1 b_q per load.
- Per-frequency loads / held values: `plan_reduced(extra_loads=[HarmonicLoad], held_offsets_of=f(omega))`;
  `set_frequency_load` / `set_held_offsets` must be None while planned (would count twice). The sum of vector_of loads
  is cached per omega with S b and V^H b: a vector_of must not change while the model lives (a changed background needs
  a new plan_reduced).
- A4 (`reduced_outputs.py`): `solve(omega, fields=False)` keeps only y (`solution` None, `reactions` refused);
  `linear_output(key, apply)` caches apply of each basis column per key (apply linear in full-numbering values);
  `publish_fields()` writes V y. Values pass `_physical_values(full)` (line Hermite slopes) in every write and in
  linear_output's columns: never scale values after solve. Solvers that skipped fields publish in
  `publish_frequency_response`. Solids (`ReducedRows`): records are exact only at the precomputed rows.
- The residual estimate is sketched (sign embedding, exact up to EXACT_SKETCH_LIMIT), not a Gram quadratic form (which
  cancels near a resonance). Error ~ Q x residual near a peak: 1e-10 tolerances for 1e-6 errors at Q 100. A tolerance
  at or below DEFLATION_TOLERANCE (1e-10) is unreachable (training stops with 'added no new direction'). Training
  costs O(m k^3) per iteration: fine to k ~ 100.

## Line elements (`line_fem.py`, plan S3)
- Dofs ascend along the line (P2: node 2k, midpoint 2e+1; Hermite: w 2k, w' 2k+1, physical slope), so operators are
  banded. Edges = `Domain.cell_edges()[0]` of a 1D model. Second derivatives only on Hermite. Hermite w / w' rows differ
  by ~1/h^2 in SI: `band_solve` / `line_eigenpairs` equilibrate symmetrically, other direct solves should too. A node
  point goes to the upper element. `band_solve` refactorises per call: sweeps use the role's `factorize_host` (sparse_direct fits 'line' through the
  tensor rule of `roles.solver_misfit`).

## Prestress (new_modules_plan K1; `elasticity.py` / `elasticity3d.py`)
- `StressField` (dimension-free) = prescribed + C eps(displacement) - thermal, evaluated by each assembler at its own
  rule points (no stored samples); a varying callable wants `degree=`. Voigt rows: planar (xx, yy, xy), axisymmetric
  (rr, zz, tt, rz), 3D (xx, yy, zz, yz, xz, xy), Pa. Planar K_sigma never sees sigma_zz.
- K + K_sigma have different patterns: `pattern_sum` where a factor is refreshed. `initial_stress_force[_3d]` has the
  sign of the thermal load. Green-Lagrange internal force / tangent share one rule (4 (order - 1), axisymmetric 6);
  nonlinear reactions come from the internal force, not K u - f.

## Self-tests
- Most fem cases are CPU `check({})` cases with first-estimate tolerances; error = number of failed sub-checks.
  Files: selftests_harmonic, selftests_mesh_symmetry, selftests_mesh_sizes3d / selftests_refinement_places (gmsh parts
  skipped without gmsh; `fem_tetrahedron_composite_sizes` / `fem_gmsh3d_disk_cache` clear gmsh3d's memory cache),
  selftests_periodic_mesh, selftests_builtin_tetrahedra, selftests_swept_prisms, selftests_rounded_rectangle,
  selftests_prestress, selftests_reduced_harmonic, selftests_elasticity3d, selftests_tetrahedra,
  selftests_diffusion3d, selftests_diffusion, selftests_mesher, selftests_elements, selftests_display.
- `fem_array_triangle_mesh` (model/selftests_geometry_array): an array of circles meshes to the same digest as the
  circles written out; keep it when build_pslg's merging or ordering changes.
- `fem_diffusion3d_identities`: the loss step uses a, c / dt ~1e12 (a weaker mass term leaves the uniform mode
  ill-conditioned, ~1e-4); shell dofs at 1e-8 since `shell_dof_locations` clamps inside the window.
- Disc meshes come from `selftests_mesher.disc_mesh(h, ring)`, a module cache shared by every fem case: never write into
  a mesh's arrays in a check; `disc_region_mesh(h)` gives the ringed 0.12 um mesh; reuse existing sizes (a new h_max
  costs a new mesh).
- `fem_mesher_conformity`: the MeshSpec path mesh must have the digest of the raw `mesh_pslg` mesh of the explicit
  TriangleSizing (a schema default change that splits them fails `same_mesh_twice`: then pass the MeshSpec fields).
- The P2 quadratic patch is exact on affine elements only (isoparametric P2 checks linear fields).
