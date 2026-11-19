# open_sol/fem — conventions & gotchas

FEM on triangles (solver_plan S7.5, mesh_implementation_plan.md section 2) and tetrahedra (section "Tetrahedra",
igor3d_plan section 7). Which physics run on element meshes and their per-element state / `field_dofs` / `fem_field`
contract: root "Meshes per physics"; the physics side lives in each physics sub file ("Triangle meshes"). Error
indicators: section "Error indicators" (users: grep `def error_indicator`).

## Layout and imports
- Module contents: `structure_index.md`. Import rules: contracts fem_below_model, fem_init_no_core,
  fem_squeeze_film_light (`model/import_contracts.py`). `selftests.py` (FEM_SELFTEST_CASES, registered by
  model.study.register) collects every fem case list: a new one appends there.
- Private helpers (`_name`) are imported across fem modules and by a few physics modules (grep `import .*\b_` under
  fem / physics): changing one touches those users. Shared triangle helpers to reuse, never re-derive:
  `assembly.nodal_average` (nodal recovery), `diffusion.window_face_edges` / `ring_dof_locations` /
  `floating_region_count`, `mesher.window_inset` / `clamp_into_window` (window sides and the ring's edge extension).

## Mesh data
- Field meanings: `TriangleMesh` / `PSLG` field comments (fem/mesh.py). Object membership of elements comes from
  `triangle_meshing.element_masks`, never `region` (objects overlap; region is the LAST object holding the centroid).
- The built-in mesher is deterministic (no jitter, stable sorts): same model, same machine, same `digest`. No
  stored-digest self-test catches a change, so a speed change to the mesher must keep every digest (compare them).

## Mesher (built-in)
- Algorithm: `mesher.py` docstrings (build_pslg, mesh_pslg, _RefineState, _delaunay, smooth). Traps:
  - every count / ceil on a PSLG piece takes a round-off guard (`_dense_count`, `_sample_counts`): overlapping copies
    with ulp-different ends (a mirror image) must sample identical vertices, or Qhull later drops coincident points;
  - a local Delaunay update returns kept rows first and `_RefineState` carries their circumcircles and last round's
    bad / size_bad flags, so a round evaluates only new rows: a new per-triangle quantity is carried the same way or
    computed for all rows;
  - Qhull options, translation or point order flip call times between ~0.05 s and seconds and flip co-circular ties
    (digests); QJ is faster but drops triangles. Tried and dropped: scipy incremental `add_points` (2.4x slower), seed
    lattices / Poisson-disk seeds (slower overall), circumcentre thinning spacing 0.35 R (more nodes) or 0.7 R (no
    faster).
- Periodic sides (`TriangleSizing.periodic_axes`): side nodes match exactly (`periodic_pslg` + paired splits); gmsh
  pins them by `Transfinite Curve = 2` (`write_geo`). A mirror axis pairs through the reflection (only the other axis
  is paired in the half mesh). No periodic size field.

## gmsh (optional)
- An external process (never imported, linked or vendored): rules in the `gmsh_io` / `gmsh3d` headers (pinned output
  options against user option files, exit 0 after an error, MathEval length). Numbers into .geo / .pos only via
  `gmsh_io.geo_number`. 3D size fields and their estimate twins: model sub file "Tetrahedron meshes".

## Elements and geometry
- Local numbering and bases: `elements.py` header and `nedelec_basis` (it applies the edge signs itself: dof maps carry
  none). Geometry order (D6): 1 = affine (the domain is the node polygon), 2 = P2 map on elements with a curved edge
  (`lagrange_dofs` default: P1 -> 1, P0 / P2 -> 2; `nedelec_dofs`: the degree). Isoparametric P2 reproduces linear
  fields exactly on curved elements, quadratic ones only on affine elements (P2 on order 1 is subparametric).
- Rule degrees: `assembly._degree` (2 x order, clamped to the largest Dunavant rule 6: an axisymmetric P2 load is not
  exact), curved elements >= `CURVED_DEGREE`, edges +2 on geometry order 2.

## Dofs and assembly
- Coefficient forms: `assembly` / `assembly3d` headers. A constant tensor goes per element
  (`np.broadcast_to(K, (n_elements, 2, 2))`): a bare (2, 2) array reads as per-element values.
- Element blocks are batched matmuls (`_tensor_blocks`, elasticity `_material_blocks` / `_basis_product_blocks`):
  coefficient tensors and shared rules broadcast, never materialised per (element, point); basis values alone via
  `elements.lagrange_values`. Element-constant coefficients on straight elements take the affine reference form.
  Mesh-constant integrals are cached read-only per dof map (`assembly._cached`, `DofMap.edge_quadrature`,
  `DofMap3D.face_quadrature`), so repeated terms pay only the coefficient.
- Axisymmetric weights differ by module: fem.assembly (scalar stiffness / mass / loads) integrates x r, fem.elasticity
  x 2 pi r (`RING`, whole rings). Code mixing them (thermoelastic damping: the scalar thermal operators against
  `assemble_strain_source`) scales the scalar ones by RING, else a ratio of the two is off by 2 pi.
- Stable patterns: `_to_csr`, `blocks_csr`, `element_blocks_csr` and FixedPatternCsr keep explicit zeros, so every
  assembly over one dof map (same free / fixed split) has one CSR pattern whatever the coefficients: the precondition
  of cuDSS plan reuse. Never `eliminate_zeros` such an operator; sum terms per root "Solver contract"
  (`pattern_sum`). `CsrAccumulator` adds its batches with `+` (exact-zero sums drop): not for a pattern refreshed in
  place.
- FixedPatternCsr (`fixed_pattern.py` header and docstrings: numba row sum / block scatter, cold thresholds, kept
  patterns and free restrictions): kept objects share their index arrays (`kept_pattern`, `with_groups`,
  `free_restriction`, `full()` / `reduced()` results): never sort / eliminate_zeros / sum_duplicates such a CSR in
  place or write into its index arrays (`full(copy_pattern=True)` / `TangentAssembler3D.tangent()` give own copies,
  e.g. for pencils). The reduced CSR is canonical: hand it to
  `DiffusionDirectSolver.factorize_host_operator(symmetric_known=True)`. Element orderings that scatter a chunk over
  the whole matrix make each bincount O(nnz) (root bincount rule). The numba paths are host twins compared by
  fem_fixed_pattern_row_sum_numba / fem_elastic_assembly_numba; in a cold process the first numba user pays the import
  + cache load (~1 s: on a 3D P2 profile it sat inside the pattern build).
- Block patterns (`FixedPatternCsr(block=3)`, interleaved vector dofs 3 d + component): which 3D terms share the kept
  base per mesh (stiffness, TangentAssembler3D, geometric stiffness, compact scalar mass): elasticity3d header;
  `check=False` rule: `FixedPatternCsr.add_elements`.
- Device tangents (`TangentAssembler3D(device=True)`, class docstring): the Green-Lagrange helpers take xp from their
  first array (`elasticity3d._array_module`): keep new code in them xp-generic. Device sums are atomic scatter_add
  (round-off differs from the host bincount). The memory estimate books the byte-capped device chunk (mechanics
  definition3d calls `device_tangent_chunk_elements` / `tangent_chunk_element_bytes`). The 2D TangentAssembler has
  no device twin.
- Element kernels (`tangent_kernel3d`: CUDA / numba / batched numpy-cupy; `tangent_kernel2d`: numba / numpy, planar
  groups only; module headers): a formula change edits the CUDA source, the numba twin and the batched elasticity
  helpers together (fem3d_tangent_assembler[_gpu], fem3d_tangent_kernel_numba, solid_fem_tangent_kernel_numba compare
  them).
- Bloch (2D twins of mesh3d / dofs3d, Lagrange only; `dofs` header): `bloch_master_map(partners_of)` needs no mesh
  object (mechanics bloch2d pairs tensor-grid Q1 nodes with it): reuse it for a new grid kind.

## Domain, sampling, transfers
- TriangleDomain (`domain.py`): per-element MaterialGrid (D12), `coordinate_grids()` = centroids clamped into the
  window (`clamp_into_window`, as `element_masks`), so `BuildContext.eval_field` works unchanged.
- Session reads: `model/field_access_triangles.TriangleAccess` (`fem_field` contract, rasters, display triangulation,
  quadrature-sampled derived values); node values by `sampling.node_values` (global ids, never located nodes).
- Located points (`TriangleDomain.located` / `quadrature_located`, `LocatedPoints.sampler`): a READ-ONLY points array
  seen before is matched by identity and a sampler is kept per dof map by identity, so callers keep raster / window
  point sets read-only and reuse them (TriangleAccess keeps them per raster shape), and a renderer keeps its DofMap (a
  new map per read rebuilds the sampler). Session code never keeps basis arrays at module level (root "Selftest
  reuse"); `raster_sampler` / `locator` module caches are for scripts and self-tests.
- PointLocator: locate with the dof map's geometry order. On strongly graded meshes one bucket can hold ~40 % of the
  elements (seconds for ~10^4 points in the refined zone): locate few points there or read values from the dofs.
- Transfers (`transfer.py` header and docstrings): the build uses them through `model.mesh_transfer.TriangleTransfer`
  (float32 in / out, source space by array length). Window-side points are evaluated at `clamp_into_window`.

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
  FEMDiffusionSystemCommon (its anchor check: `floating_region_count`), `element_means`).
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
- `TetrahedronDomain.locate` remembers each point set (blake2b; up to LOCATE_CACHE_POINTS; a read-only array seen
  before by identity) and returns READ-ONLY arrays; callers copy first, renderers keep no location cache of their own
  and pass the points they get on unchanged. `sample_plane(axis, index)` keeps the flat read-only lattice plane (up to
  one lattice), so the Session's cuts / volumes locate without hashing.
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
  disk (`mesh_cache_folder`: `mesh_solids_with_gmsh(disk_cache=)` first ('' off), then off inside
  `default_disk_cache_off` (api.selftest / selftest_examples enter it BEFORE the DependencyTrace: a disk hit would hide
  the mesher code from the selftest cache tracer, and entering it inside the trace would make every case depend on
  gmsh3d), then `OPEN_SOL_MESH_CACHE` ('' off), unset = `default_mesh_cache_folder()` (the user's cache folder: on by
  default for api.run / CLI / GUI); the profilers set `OPEN_SOL_MESH_CACHE` '' (tools_profilers `__init__` and
  profiler_harness), so they time gmsh, never a disk read): the RAW `read_msh_tetrahedra` arrays
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
  field has no Thickness (jumps to h_max at its surface); + MeshSizeFromCurvature (2 pi / curvature_factor), Optimize 1, OptimizeNetgen 0 (Netgen
  doubled the gmsh time of a 10k-tet mesh for a 1 % quality quantile of 0.20 instead of 0.17; a mesh-dependent
  tolerance or sliver warning that needs it back changes every gmsh tet mesh);
  needs gmsh >= 4.8. `periodic_axes`: `_periodic_rows` pairs hi / lo surfaces of the same translated box (`Periodic
  Surface ... Translate`); a periodic axis must carry no PML shell (ValueError). `Solid.edge_size` (electrode edges,
  RefinementSpec.where 'edges'): one MathEval per box edge / cylinder rim (only arithmetic and Sqrt; max(a, 0) as (a +
  Sqrt(a a)) / 2), clipped by MeshSizeMax: no Distance field (its sampling option was renamed between gmsh versions).
  `Solid.edge_distance` is the estimate's numpy twin.

## Scalar diffusion on tetrahedra (`diffusion3d.py`)
- FEMDiffusionSystem3D mirrors FEMDiffusionSystem (same owner protocol; the shared part, from the solver binding,
  sequences, factor-reuse flags, held regions and losses to the anchor check, factorisation and reduced solve, is
  `diffusion.FEMDiffusionSystemCommon`, `_full_ids` mapping system dofs into `values`), differences: full-mesh
  numbering throughout (free = dofs of active elements minus fixed), tensor (n_elements, 3, 3), outer faces on `window_faces`,
  `add_surface_condition(faces, condition)` (a face with an active element on either side; a Neumann value on a
  two-sided face is a sheet source feeding both sides), `add_point_source(element, reference, power)` (exact delta;
  dropped in inactive elements), `add_fixed_dofs` / `set_fixed_dofs` (a set takes a new sequence and so wins shared
  dofs; empty ids = none), `solve(source, extra_load=)`, PML shell dofs at clamped positions (`shell_dof_locations`,
  1e-9 x side inside on every axis). Reactions / face_flux as 2D; factor reuse as 2D (`_note_change` /
  `_refresh_constant_terms`: a Dirichlet value or Robin external of the same h only refreshes lift and loads).
- Anchors by topology (`floating_region_count(reduced, anchored_rows)`): loss / capacity / Robin mass row sums are ~h^2
  a / k of the stiffness diagonal (~1e-12 for a weak loss on 0.3 um cells), below the ANCHOR_RTOL row-sum test, which
  refused such systems as singular. The 2D system passes the same topology anchors.
- Re-assembly (2D and 3D, `fem.diffusion`): `KeptTerms` returns the stiffness / loss / capacity / Robin matrices again
  while their inputs repeat (DofMap or active mask, coefficient arrays by value; a radiation or sigma(T) loop
  re-assembles only what moved), `SlotSum` sums terms by kept slots in the summed pattern (= pattern_sum: every entry
  kept), `ReducedSplit` gathers the reduced / coupling values by kept slot maps and reuses the anchor regions while the
  reduced zero entries repeat; the 2D DofMap is kept while the active set repeats (its integral caches stay warm). A
  term whose pattern depends on its values (CsrAccumulator batches) only rebuilds the slots. Memory: one fem estimate
  host item for the kept stiffness and maps (`fem_system_estimate(kept_terms=True)`; a physics with a system class of
  its own that borrows the estimate passes False: magnetostatics on triangles, `MagneticFEMSystem`).
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
- Point evaluations (`element_strains_3d` (linear / green_lagrange; `green_lagrange_strains_3d` is it),
  `element_gradients_3d`): per chunk of POINT_CHUNK_ELEMENTS one GEMM of the element values with the constant
  reference tables, then grad = J^-T grad_ref by components from `element_inverse_jacobians_3d` (J^-1 of every mesh
  element, cached on `DofMap3D.element_geometry` per array module, host copy read-only; a degenerate element anywhere
  in the mesh raises on first use). element_strains_3d runs on the displacement's module (cupy: J^-1 and element dofs
  uploaded once per map); the physics outputs stay on the host.
- `elements3d.lagrange_basis_3d` memoises point sets up to LAGRANGE_MEMO_POINTS (latest LAGRANGE_MEMO_ENTRIES) and
  returns those arrays READ-ONLY: never write into a basis / reference gradient table; the lookup stays inside the
  function so a memo hit still traces it (root "Selftest reuse"). A caller that needs no gradients passes
  `gradients=False` (None in their place; `evaluate_lagrange_3d` of located samples: the P2 (p, 10, 3) table cost most
  of its time).
- Piezo pieces (the block operator is physics/mechanics/piezo_fem3d.py): `assemble_piezo_coupling_3d` = CSR (3 n, n)
  of int B^T e^T grad N (K_uphi; K_phiu = its transpose); the phi block is `assemble_lagrange_stiffness_3d` with the
  full eps0 eps; `element_gradients_3d` returns sorted elements.

## Complex scalar harmonic systems (`harmonic.py`, `harmonic3d.py`, `pml.py`, `harmonic_resonances.py`; plan S1, S2, S8)
- -div(a grad u) + (b + j omega c - omega^2 m) u = f; K, C, M on one fixed pattern (`HarmonicPencil`), one factor per
  frequency via `factorize_host(..., symmetric=, previous=)` (cuDSS plan reuse). On a cuDSS back end `_factorize` hands
  the reduced K, C, M to the device once per assembly (`_device_terms`: DevicePencil) and each frequency calls
  `factorize_pencil` (no host A(omega)); frequency_terms (ports) go in through `extra=` at slots mapped once; a block
  outside the pattern, or unsymmetric with a symmetric pencil, takes `factorize_host`. `HarmonicPencil.of` takes a
  shared pattern as is, else the largest term's pattern when it holds every other term's (volume K beside an edge /
  face damping term: `_covering_term`, checked entry by entry; terms must be canonical CSR), else the pattern_sum
  union; index maps from one pass over the stored entries (scipy fancy
  indexing is slow at 1e7 entries). Active elements = `set_active(mask)`, not a > 0. Robin: n . a grad u = value -
  (admittance + j omega damping) u.
- MUMPS / PARDISO back ends: `_device_terms` asks `linear_solver.pencil(...)`, a HostPencil (full pattern, host
  values), so `_factorize` takes the same factorize_pencil branch: the analysis kept, no host A(omega), canonical copy
  or asymmetry check per frequency; `_extra_on_device` tests `pencil.symmetric` (not lower_stored) before adding a
  block. Real K, M without damping give a real pencil, hence a real LDL^T solving the complex rhs as two real solves
  (the host route's A(omega) was always complex). Factor errors become ValueErrors at `_factorize`:
  CudssAccuracyError / HostDirectAccuracyError (a RuntimeError, caught first) -> `_accuracy_error` naming the library
  (`host_direct.factor_label` of the runtime's back end), any other RuntimeError (zero pivot) -> the singular text;
  HostDirectMemoryError (a ValueError) passes as is. A HostPencil refresh defers its probe to the first solve (double
  factors), so `_solve_free` meets a HostDirectAccuracyError there too (no fallback: only an 'auto' cuDSS runtime
  falls back).
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
- Shared tetrahedron-mesh check helpers live in selftests_mesh_sizes3d (`_field_tree` / `_field_value`: parse and
  evaluate a .geo field tree; `_refusal_text`, `_ring_spec`, `_ring_point`, `_tet_stats`); selftests_harmonic's
  `_checker` (SubChecks + add_bound) also serves reduced_harmonic, resonance_search and mechanics squeeze_film.
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
