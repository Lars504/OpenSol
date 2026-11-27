# open_sol/fem — conventions & gotchas

FEM on triangles and tetrahedra (section "Tetrahedra"). Which physics run on element meshes and their per-element
state / `field_dofs` / `fem_field` contract: root "Meshes per physics"; the physics side lives in each physics sub
file ("Triangle meshes"). Error indicators: section "Error indicators" (users: grep `def error_indicator`).

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
- Elastic drives (`elastic_drives.py`: point-stress and edge-traction drives at given quadrature points, the Brillouin
  loads) take B, rows and ring weights from elasticity's own helpers (`_form_operator_at`, the stiffness groups' form;
  `_weights`, `_azimuthal_terms`), so they pair with K / K_m by construction: a change of the elasticity rows, weights
  or element chunks moves both (fem_elastic_drive_* pin transpose, loads and the order-m phi integral). Their
  `DrivePoints` hold the dof map they were built on and refuse another: rebuild them after a re-assembly gives a new
  map (a kept strain operator, `keep_operator`, dies with them). Edge drives take points, weights and normals from
  `indicators.edge_sides` and the basis from `assembly._edge_quadrature` unchecked: both must keep one Gauss rule and
  `edge_geometry` per degree (fem_elastic_drive_constant_stress compares the edge drive with assemble_edge_traction).
  A stress drive on a mixed u-p solid takes `projection=` (`PressureProjection`: the solid's own B, M_p,g and free
  pressure dofs, never a re-assembly on other rules): its mean normal stress goes in as B^T M_p^-1 int N_p s dV, since
  the non-P1 part of tr eps carries no bulk stiffness and a plain point drive excites it (elastic_drives header). Its
  normal rows must stay the rows `mixed_elasticity` sums into tr eps (fem_elastic_drive_pressure_projection).
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

## Scalar diffusion (`diffusion.py`; 3D: section "Scalar diffusion on tetrahedra")
- Contract (numbering, active elements, condition sequences, anchors, factor reuse, ring extension): the
  FEMDiffusionSystem[Common] docstrings and `floating_region_count`. Only a condition kind / Robin h change, held
  regions, losses and materials re-assemble; Dirichlet / Robin-external values refresh the lift on the next solve and
  Neumann values are read every solve, so a fixed-dt transient factors once. Terms whose inputs repeat (KeptTerms,
  SlotSum, ReducedSplit) are reused, so a radiation or sigma(T) loop re-assembles only what moved, on one pattern.
- Outer faces act on the WINDOW EDGE (`face_edges`), not the mesh's outer face tags (the harmonic systems do the
  opposite). Object outlines = `BuildContext.boundary_edges`; a flux there is a line source feeding both sides (unlike
  tensor perimeter cells inside the object).
- Axisymmetric integrals, reactions and fluxes are per radian: physics multiply held fluxes by 2 pi, point sources are
  whole-ring totals. `face_flux(face)` counts only the dofs its condition won (a shared corner once);
  `inward_flux(edges, condition)` counts every dof of the edges.
- Estimates (`fem_diffusion_estimate[_3d]`, `fem_system_estimate`, `direct_factor_item`, `cudss_diffusion_estimate`:
  docstrings; all uncalibrated): entries per row only from `lagrange_entries_per_dof` (mesh-estimate counts; P2
  triangles ~11.5, measured 11.2 on heat_fem_cpu_gpu), never a constant; time estimates take `fem_time_system` (real
  unknowns x entries per row), never inflated "equivalent cells" (LU costs grow ~n^1.5 / ~n^2). The cuDSS first plan
  peak was measured at ~39 B per entry (heat_fem_cpu_gpu).

## Factor entry count (`dissection.py`)
- Method and why: module header (cut at the elements wholly above the plane, never by centroid: a centroid split tears
  swept prism columns, 1.56x cuDSS's entries on sin_square_membrane_modes_3d). The count bounds a good ordering from
  above and fits nothing: a count below cudss_lu_entries / eigen_lu_entries (summary.json) means cuDSS orders worse,
  a lead on its ordering options.
- "Elements" may be any clique cover of an operator's pattern with a point per unknown (physics/em/memory_estimates
  `_bor_factor_entries`: curl points of the tensor BOR operator).
- `MeshDissection` (counted on first use, equal per digest) reaches estimates as `estimate.dissection` through
  `meshing.built_mesh_counts` / `with_built_counts` (None on unbuilt meshes; the GUI's built counts carry it, so its
  first count runs on the GUI thread); consumers: grep `dissection` under physics.

## Error indicators (`indicators.py`)
- eta^2 per element (float64), energy-norm weighted like the tensor A2 hooks (1 / c, 0 where c is 0); hooks: grep
  `def error_indicator`. `edge_sides` / `add_edge_terms` / `lagrange_at` / `nedelec_at` are shared by em (fem_fdfd,
  fem_modes), magnetic and membrane bending_fem: a change to the side / normal / edge-sign conventions touches them.

## Tetrahedra (host numpy, scipy inside functions)
- The 2D modules stay 2D (no dimension switches); dimension-free pieces are imported by the 3D ones (assembly3d
  re-exports the Dirichlet helpers, diffusion3d reuses diffusion's EdgeCondition and Common classes).
- Window extension: `mesh3d.clamp_into_window_3d` (inset_centroids, region tags, shell dofs; 2D twin
  `mesher.clamp_into_window`), never a hand-written clip.
- Faces: `TetrahedronMesh.object_faces` come from `model.tetrahedral_meshing.tetrahedralize` via `element_membership`
  (the builder's mask function at `inset_centroids()`), so tags and masks never disagree; `BuildContext.boundary_faces`
  reads them. `face_tags` are outer-only (the node bounding box: the PML shell's outer side); window planes are
  `window_faces`.
- Element nodes ascend (Nedelec needs no signs): never reorder `tets` rows or their nodes after building.
- Periodic sides: `mesh3d.periodic_node_partners` -> `dofs3d.periodic_dof_partners` / `bloch_masters` (em3d
  `fem3d.bloch_identification`, the solid's `bloch3d`; prolongation stays the caller's). The Kuhn split pairs under any
  grading; gmsh only with `periodic_axes` (Lagrange only: node order is the mesher's); swept prisms pair Lagrange on
  x / y / z, Nedelec x / y not (the 2D mesher gives no matching node order: em_fdfd_3d refuses periodic x / y on swept
  meshes). Split prisms can be obtuse, so `n_poor` may count some.
- `region` / object membership: the last solid holding the inset centroid; gmsh volumes whose tags agree on
  >= ENTITY_MAJORITY take that tag throughout. `entity_majority` is one bincount over the cached `_entity_groups`,
  never a loop over entities (it runs per object of every selection).
- Location (`TetrahedronDomain.locate`, `PointLocator3D`): READ-ONLY arrays remembered per point set; callers copy
  first, renderers keep no location cache and pass the points on unchanged (`sample_plane` arrays match by identity).
  Points missed by the k-d tree passes fall to a per-point brute force (slow on a huge mesh); a point on a face may land
  in either element.
- `fem.assembly.congruence(T, A)` = T^T A T (a diagonal T scales A's pattern, explicit zeros kept): use it for
  electrode / dof maps instead of `T.T @ A @ T` (two SpGEMMs).
- gmsh3d (header: fragmenting and clipping, size fields, composite / surface sizes, Cylinder field, caches): its
  output may depend only on the .geo text (the cache key, in memory and on disk); bump `_MESH_CACHE_FORMAT` when cached
  arrays change meaning; a gmsh binary replaced at the same path needs a new process. Size fields keep numpy twins
  (`SizedRegion.sizes` / `surface_sizes`, `Solid.edge_distance`) the estimate uses: change both. Self-tests clearing
  the memory cache (`fem_tetrahedron_composite_sizes`, `fem_gmsh3d_disk_cache`) count `gmsh_run_count()`.

## Scalar diffusion on tetrahedra (`diffusion3d.py`)
- FEMDiffusionSystem3D / FEMDiffusionPhysicsBase3D mirror the 2D classes through `diffusion.FEMDiffusionSystemCommon`
  / `FEMDiffusionPhysicsCommon`: a change to conditions, sequences, anchors, factor reuse or held / loss terms goes
  there, for both. 3D-only: full-mesh numbering throughout, outer faces on `window_faces`, a Neumann value on a
  two-sided interface face is a sheet source feeding both sides, `add_fixed_dofs` / `set_fixed_dofs` (a set takes a
  new sequence and wins shared dofs), exact point deltas (dropped in inactive elements).

## Vector elasticity on tetrahedra (`elasticity3d.py`)
- Numbering, Voigt rows, patterns, PML stretch: module header. Point evaluations (`element_strains_3d`,
  `element_gradients_3d`) run on the displacement's module (cupy: J^-1 and element dofs uploaded once per map); physics
  outputs stay on the host.
- `elements3d.lagrange_basis_3d` returns memoised READ-ONLY arrays: never write into a basis / gradient table; the
  memo lookup stays inside the function (root "Selftest reuse"). Callers that need no gradients pass
  `gradients=False` (the P2 (p, 10, 3) table cost most of a located-sample evaluation).

## Complex scalar harmonic systems (`harmonic.py`, `harmonic3d.py`, `pml.py`, resonances)
- Contract (operator, pencil and pattern reuse, cuDSS / MUMPS / PARDISO pencils, conditions, PML, resonances): the
  module headers. Unlike the diffusion systems, face conditions act on the mesh's outer edge / face tags (a wall
  continues along the other faces' PML layers) and the active elements are `set_active(mask)`, not a > 0. Edge, face
  and point terms are not stretched: keep Robin / Neumann boundaries and point loads out of the PML.
- Per-frequency terms never re-assemble or change the reduced pattern. Code outside fem uses only the public API of
  `HarmonicSystemBase` (header: `frequency_terms`, `set_frequency_load` / `set_held_offsets`, `solution` / `factor` /
  `fixed_value`, `ensure_assembled()`); `load_vector`, `operator` and `reactions` hold pencil terms only. Subclass
  hooks (`_assemble_terms`, `_physical_values`, ...) are for subclasses (grep `class .*HarmonicSystem` under physics).

## Mixed u-p (`mixed_elasticity.py`, `mixed_elasticity3d.py`; the mechanics mixed formulation)
- B and M_p take the elasticity rules and weights (`_operator_at` / `_azimuthal_terms`: x 2 pi r RING, x pi r at m >= 1)
  and tr eps from the stiffness's own strain operator rows, never fem.assembly's x r: B must match K's rules. The
  integrands are polynomial on isoparametric P2 elements at the stiffness's rule degrees (CURVED_DEGREE,
  AXISYMMETRIC_DEGREE; 3D affine), so the row sums B^T 1 are exact to round-off: the mechanics gauge detection and
  volume check (tolerance 1e-9) rely on it; a lower rule there breaks them. A PML stretch (`stretch=`, 2D) keeps it:
  tr eps J is an exact divergence in the stretched coordinates and the planar rule gains + 2 as the stiffness's (the
  axisymmetric corner integrand reaches degree 6).
- Pressure maps per group take the displacement map's geometry order; a group's dofs are offsets[g] + its own
  numbering (nodes shared by groups carry one dof per group): node values go through `nodal_pressure[_3d]` (groups
  averaged) or `pressure_at[_3d]`, never a global-id lookup; regions across groups through `pressure_regions(space,
  groups)` (joins the duplicates by `dof_nodes()`).
- Group assemblers (`assemble_elastic_stiffness_groups[_3d]`, `assemble_pressure_terms[_3d]`): one strain operator per
  element chunk for every group and set; 3D CsrAccumulators per (group, set), never `kept_pattern` (group patterns
  would evict K's kept block pattern). `embedded` / `saddle_coupling` keep explicit zeros (index shifts and
  pattern_sum, no bmat).

## Squeeze film (`squeeze_film.py`; runtimes: physics/mechanics/squeeze_film.py tets, physics/membrane/squeeze_film.py)
- Equation, units, vents and the two routes of w: module header. A shape that is not the film's own full coefficients
  goes through the volume source on the ACTIVE dof map (never the full map: a boundary dof's support outside the film
  would count).

## Reduced sweeps (`reduced_harmonic.py`, `reduced_outputs.py`)
- Contract: module headers and `HarmonicSystemBase.plan_reduced`. Per-frequency loads / held values go in as
  `plan_reduced(extra_loads=, held_offsets_of=)` with set_frequency_load / set_held_offsets None (refused otherwise);
  `solve()` raises when the operator, load or held values changed after planning; a vector_of must not change while
  the model lives (a changed background needs a new plan).
- `_physical_values(full)` (line Hermite slopes) applies in every write and in `linear_output`'s columns: never
  scale values after solve.

## Line elements (`line_fem.py`)
- Edges = `Domain.cell_edges()[0]` of a 1D model. Hermite w / w' rows differ by ~1 / h^2 in SI: `band_solve` /
  `line_eigenpairs` equilibrate symmetrically, other direct solves should too. `band_solve` refactorises per call:
  sweeps use the role's `factorize_host`.

## Prestress (`elasticity.py` / `elasticity3d.py`)
- `StressField` Voigt rows: planar (xx, yy, xy), axisymmetric (rr, zz, tt, rz), 3D (xx, yy, zz, yz, xz, xy), Pa; a
  varying callable wants `degree=`. Planar K_sigma never sees sigma_zz.
- 2D K + K_sigma have different patterns: `pattern_sum` where a factor is refreshed (3D shares the kept block
  pattern). `initial_stress_force[_3d]` has the sign of the thermal load; nonlinear reactions come from the internal
  force, not K u - f.

## Self-tests
- Most fem cases are CPU `check({})` cases with first-estimate tolerances; error = number of failed sub-checks; gmsh
  parts skip without gmsh. Shared helpers: selftests_mesh_sizes3d (.geo field trees, refusal text, ring specs; grep its
  importers) and selftests_harmonic `_checker` (SubChecks + add_bound; also reduced_harmonic, resonance_search and
  mechanics squeeze_film).
- Disc meshes come from `selftests_mesher.disc_mesh(h, ring)` / `disc_region_mesh(h)`, a module cache shared by every
  fem case: never write into their arrays; reuse existing sizes (a new h_max costs a new mesh).
- `fem_array_triangle_mesh` (model/selftests_geometry_array: an array of circles meshes to the digest of the circles
  written out) and `fem_mesher_conformity` (the MeshSpec path mesh has the digest of the raw `mesh_pslg` mesh of the
  explicit TriangleSizing: a schema default change that splits them fails `same_mesh_twice`; then pass the MeshSpec
  fields) guard build_pslg / meshing changes.
