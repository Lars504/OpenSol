# open_sol — structure & conventions

Gotchas and cross-file code flow only; signatures and help texts live in the code. Current state, no changelog.
Folder-specific gotchas live in `sub_structure_conventions.md` next to the code (index in section 3): read those of the
folders you work in. Only gotchas every agent needs stay here. A sub file keeps cross-file rules, the why and traps at
call sites, never what a module header, docstring, help text or comment already says.

Outside the package: `examples/` (model files and scripts, indexed by `examples/README.md`); `experiments/` (the
user's private research models in the same format, one subfolder each, loaded by path; not in the Examples menu, not
checked by selftest examples_validate); `tools_profilers/` (developer benchmarks of solvers, selftest runs and VRAM:
imports open_sol, never imported by it, not in the pip package; `tools_profilers/time_profilers/README.md`; its
solver_profiler is not `python -m open_sol profile`, the machine profile of the run-time estimate). User-facing:
`README.md`, `INSTALL.md` (conda, optional cupy / nvmath / MUMPS / PARDISO), `nvmath_guide.md` (cuDSS notes);
`pyproject.toml` makes `pip install -e .` work, else run from the repository root.

## 1. Structure

What each module holds: `structure_index.md` (generated from the module docstrings, one line per module; read the
folders you work in by its Contents line ranges). Folders:

```
open_sol/
├── __init__.py      # public scripting API; imports without cupy (core.backend stand-ins; nothing allocates at import)
├── api.py           # headless (no Qt) entry points: describe, template / load / save, check, scene_figure, run,
│                    # selftest, Results, RunError
├── __main__.py      # CLI over api: describe | template | check | run | show | selftest | mesh-study | profile;
│                    # exit 0 ok / 1 model, run, selftest or doc-check failure / 2 usage
├── api_reference.md # describe() text for agents ('## Contents' of line ranges: read it in offset / limit slices);
│                    # api_templates.md the template JSON; both written only by `describe --update-reference`
├── simulation.py    # Simulation: Domain + named physics + one MultiphysicsRunner (scripting API)
├── core/            # constants (import, never redefine), grids and graded meshes, Domain / Domain3D, MaterialGrid,
│                    # PhysicsSolver ABC, backend (cpu / gpu) and kernel twins, GPU / host memory, memory and run-time
│                    # estimates, DiffusionSystem, metric, mirror planes, cut cells, grid transfers, resonance finder
├── materials/       # library Materials (one file each), REGISTRY, crystal tensors
├── physics/<pkg>/   # one package per physics family (solvers, renderers, definition*.py, selftests*.py): em (2D
│                    # FDTD / FDFD, modes, BOR, conical, spherical), em3d, eme, nlfd, heat, potential (electric_currents,
│                    # electrostatics), mechanics, membrane, semiconductor, acoustics, fluid, magnetic
├── solvers/         # solver roles, SolverTypes (definition.py) and the role runtimes (pcg + multigrid, line_block,
│                    # sparse_direct, banded, arpack_shift_invert, block_eigen, cudss_factor, host_direct,
│                    # explicit_kernels)
├── coupling/        # CouplingTerm / CouplingGroup, DynamicMaterialUpdate, MultiphysicsRunner, SlotSum, the coupling
│                    # type definitions per partner family (definitions_*.py) and their self-tests
├── model/           # spec, schema (registry), expressions, geometry, validate, meshing, builder (-> Session),
│                    # session, studies (study*.py), analytic comparison, describe, templates, memory / time
│                    # estimates, selftest machinery (cache, examples, import_contracts, structure_index)
├── fem/             # triangle / tetrahedron / line meshes, mesher, gmsh I/O, elements, dofs, assembly, sampling,
│                    # FEM systems (diffusion, elasticity, magnetostatics, complex harmonic), PML stretches
├── io/              # model JSON and exports (model_io), run directories (results_io)
├── viz/             # Qt-free matplotlib figures, HTML field / mode players
├── mesh_study/      # the CLI mesh-study (runs, records, gate report)
└── gui/             # PySide6 + pyqtgraph application
```

Import rules live as data in `open_sol/model/import_contracts.py` (`CONTRACTS`, each with its rule text; the built-in
selftest import_contracts checks every import of the package against them and flags stale exceptions): a new layer or
exception is an edit there, not here. The ones that shape everyday code:
- Optional packages: cupy / cupyx at module level only in `core.backend` (others: `from <..>core.backend import cp` /
  `cupyx_sparse as cpsp` / `cupyx_ndimage as cpndi`; a real `import cupy` only inside a GPU-only function), so
  open_sol imports without cupy. scipy only inside functions (use it on the host where cupyx falls short: no `eigs`,
  no shift-invert, `spsolve` refactorises per call). PySide6 / pyqtgraph only in gui; nvmath only inside functions;
  python-mumps / pydiso only inside functions of `solvers.host_direct`.
- Layers: core imports nothing outside core but its listed exceptions; fem never imports model, physics or coupling;
  solvers runtimes import only solvers and core. `physics/<pkg>` never imports the model runtime (meshing, builder,
  study, session, validate, ...): shared study rules go to core or `model.definition_helpers`; `open_sol.materials`
  only inside functions (materials imports physics.em). physics.em is upstream (other physics packages import it,
  never the reverse; per-polarisation permittivity array / slot names: `physics.em.placement`); acoustics imports
  mechanics, never the reverse. Chains inside physics/em: em sub file "Imports".
- `selftests*.py` import at module level only core.constants, model.spec / schema / selftest_helpers and sibling
  selftests modules (physics modules and scipy inside the checks), and reuse `model.selftest_helpers` (multi-check
  case records, relative errors, Results reads, skips), never a local copy. Nothing else imports a selftests module at
  module level or in `register` (contract selftests_lazy): a definition imports its cases inside its selftest loader,
  so `get_registry()` (GUI, check, describe) loads only the selftests modules that hold templates (`TEMPLATE_SOURCES`).
- `get_registry()` imports model.study, solvers.definition, every `physics/<pkg>/definition.py`, then
  `coupling.definitions` (which discovers every `coupling/definitions_*.py`), calling each `register`: a new physics
  package or coupling topic needs no central edit; calling `get_registry()` inside a `register` raises.
- model.study imports model.builder at module level: nothing builder / session imports may import model.study. `api`
  is imported only by the package root, `__main__` and mesh_study (a few others inside functions); gui by nothing.
- Cycles are broken by an import inside a function (e.g. coupling.definitions' `definitions_*`, model.meshing's
  triangle / tetrahedral meshers, em3d `definition_fdfd3d`): keep such imports there.

Code flow. Scripting: `Simulation(size_x, size_y, dx)` / `from_mesh(cfg)` = Domain + one EMPhysics 'em';
`Simulation.multiphysics(domains)` + `add_physics(name, solver, every[, domain])` + `add_coupling` otherwise (unknown
attributes fall through to a single physics); `run` finalizes new physics, prepares the runner, advances in chunks and
flushes (its docstring: chunked scripts pass `flush=False` but for the last chunk). Model path: ModelSpec ->
`validate` (CPU, Issues at GUI node paths) -> `build` (worker thread: validate, parameters, geometry, one Domain per
used mesh, static materials, per physics `create` + `add_physics` + feature `apply`, dynamic updates `apply_now`,
`sim.finalize([name])` per solver (its `finalize_warnings` logged as warnings, optional `finalize_notes` as info
lines), couplings, `runner.prepare`, `release_caches`) -> Session. `run_study` builds once per sweep value (once for
the whole sweep with `SweepSpec.reuse_build`: model sub file "Reused sweep builds"; once per refinement level or
evaluation: `study_adaptation`, `study_estimation`) and runs the StudyType's `run` (built-in studies in
`model/study.py`, the others in `model/study_*.py`; each `_run_*` docstring gives its order). Studies reach solvers
only through the hooks of `model/solver_hooks.py` (section "Solver contract"). Harmonic, mode and resonance studies
first bring the background physics (those without the study's hook) to a stationary fixed point and apply background
-> study-physics couplings once (per point instead when a coupling reads the harmonic physics:
`StudyType.coupled_points`); they store one labelled Snapshot per point and leave one solution live. Headless:
`api.check` (CPU -> check.json), `api.run` (run_study on the calling thread -> Results + run directory),
`api.selftest` / `api.selftest_examples` (-> selftest reports); the CLI wraps them; file formats in
`describe('results')`.

## 2. General gotchas

- Shared names come from their constants, never string literals in code: study names `model.schema.*_STUDY`, mesh
  kinds `schema.MESH_KINDS` / `core.constants.FREE_MESH_KINDS`, faces `FACES` / `FACES_3D`, mirror signs
  `core.mirror.MIRROR_SIGNS` (physics / feature type names stay literals).

### Arrays, coordinates
- Per-cell state: float32 (`DTYPE`) arrays of the run backend (`domain.xp`: cupy on 'gpu', numpy on 'cpu'; code
  shared by both backends takes xp from the Domain or `core.backend.array_module_of`, never `cp` directly; core sub
  file "Backends"), `(nx, ny)`, `[i, j]` i along x, SI, full grid incl. the PML ring (window from `(ox, oy)`);
  renderer / Session fields and FieldData are window-shaped. Array dataclasses: `frozen=True, eq=False`. Exceptions:
  a 'half_storage' stepper stores float16 in scaled units (code reading or writing those arrays converts through the
  engine's scale or refuses half_storage, never with cupy's own float16 kernels: core sub file "Half storage"); a
  'double' stepper (em_fdtd_ez / _hz) steps float64 fields (code adding to them takes the engine's `dtype`: em sub
  file "Stepper precision"). Renderers and results always return float32 physical values.
- numpy 2 reprs scalars as 'np.float64(0.25)': text for other programs (gmsh .geo, scripts, CSV headers) takes
  `float(value)` / `.tolist()` first, never `{value!r}` of a numpy scalar.
- Coordinates only from `Domain.cell_widths / cell_edges / cell_centers` (float64, cached read-only: copy before
  writing); Yee stencils and H_z source weights use `physics.em.yee_grid.grid_cell_widths(grid)` (the kernels'
  widths). On a polar `Domain.coordinates` (model: `DomainSpec.coordinates` 'polar' / 'spherical') the logical x is
  r - radius_min and y an angle in radians: areas / volumes come from `core.metric` (core sub file), never from
  products of widths, and a physics runs there only when its `coordinate_grids` list the grid. Model geometry stays in
  the drawing plane: code mapping between it and the cells goes through `geometry.DomainMap` (model sub file "Polar /
  spherical domains"), never through the logical widths. Vector / tensor components keep their _x / _y names in the
  grid frame (a renderer relabels them with `core.renderer.polar_field_infos`).
- `pos_to_cell*` = containing cell (an edge coordinate goes to the upper cell), clamped. Spans / boxes cover
  `[cell(lo), cell(hi)+1)`: an edge exactly on a cell edge takes one extra cell. Rectangles / polygons test centres
  half-open, circles / ellipses closed.
- `coordinate_grids()` X / Y are zero-stride views of the 1D centres (cupy does not make them read-only): consumers
  stay elementwise and never write into them. They are clamped by default (PML cells take the nearest window centre:
  masks, x / y expressions); `rect_mask / circle_mask` use unclamped centres.
- `Domain.size_x/y` are requested; `physical_extent()` is the meshed window (use it for display). `Domain.dx` =
  dx_min on graded grids. `boundary_config=None` pads `npml` (15) on every side.
- `npml_sides` is padding, not only PML: an EM mirror face (symmetric / antisymmetric) pads one ghost cell
  (`BoundaryConfig.ghost_faces`), so the mirror plane is the window edge for every physics of the model; code sizing
  CPML layers reads the bc type, not the count. Model mirror planes also come from features (the `symmetry` feature
  of every non-EM type, membrane / pressure_acoustics also `antisymmetry`: `core.mirror.MIRROR_FEATURES`; validated
  through `definition_helpers.mirror_feature_issues`): code listing them calls `mirror_faces(spec[, physics])`, never
  reads bc settings or feature types; displays unfold over `unfold_faces(spec, physics)`. Results count the unfolded
  structure: a new signed renderer field declares `FieldInfo.parity`, a new total over a selection / line uses
  `touch_factor`, a new far-field result keeps its complex amplitude, a new result kind that parity sectors combine
  gets its `combine_sector_result` rule (core sub file "Mirror planes").
- `geometry.resolve_geometry` returns more names than `spec.geometry`: array copies ('<array>[i,j]',
  `geometry.is_array_copy`) are resolved shapes of their own, so meshers and outlines see every copy; code listing
  objects from the resolved dict skips them (model sub file "Arrays").
- Courant: `courant` / `courant_factor` = fraction of the CFL limit of the finest cells (2D; 3D models: the 3D limit),
  in (0, 1]; dt only from `core.grid.courant_time_step` (3D: `grid3d.courant_time_step_3d`); uniform cell counts and
  cell edges only from `core.grid.uniform_cell_count` / `edges_from_widths`.

### Meshes per physics (model sub file "Per-physics meshes", "Per-mesh Session"; gui sub file "Meshes")
- `ModelSpec.meshes` (named MeshSpecs, first = default) and `PhysicsSpec.mesh` ('' = meshes[0]): model code finds a
  physics' mesh by `spec.mesh_of(physics)` and passes mesh names to meshing; `spec.mesh` / `ModelSpec(mesh=)` are
  meshes[0] conveniences for scripts and self-tests only. Inside a validate callback `vctx.mesh` is that physics'
  estimate (None on a mesh kind the type does not run on). Types, features and solvers run on the mesh kinds their
  `grids` hold; code reading tensor fields of an estimate (nx, x_edges, dx_cells) or of a Domain (`physical_window`,
  `cell_centers`) checks the kind first (an element domain has none).
- Build: one Domain + BuildContext per used mesh; creates / applies get `ctx.for_physics(name)`, so `ctx.domain` is
  the physics' own mesh, never assume it is the Session's (`sim.domain` = the first mesh; `sim.domain_of(name)`).
  Cross-mesh couplings: coupling sub file "Couplings across meshes". Session reads of a physics (fields, snapshots,
  feature-result callbacks) act on its own mesh (`Session` docstring): a result callback never assumes the first mesh.
  Per-kind behaviour lives in `session.access` (`model/field_access*.py`) and `ctx.masks` (`model/build_masks.py`): a
  new mesh kind or window map is a new subclass of each, never a `grid_kind` branch in Session or BuildContext.
- Element meshes (`core.constants.FREE_MESH_KINDS`: 'triangles', 'tetrahedra' (3D Cartesian only); code testing "not
  tensor" tests membership there, never == 'triangles'; model sub file "Triangle meshes" / "Tetrahedron meshes", fem
  sub file): a `TriangleDomain` / `TetrahedronDomain` has a per-element MaterialGrid and no tensor accessors (3D code
  branches on `domain.grid_kind` before physical_window / cell_centers / object_box_mask); boundary selections are
  faces (`ctx.boundary_faces`), points are located (`ctx.point_position`), never nodes. A physics there keeps every
  state() array and source slot per element ((n_elements,): centroid values / element means), `field_dofs(name)`
  their P0 DofMap, so state-dependent materials, same-mesh couplings and contributor buffers see per-element arrays;
  its create branches on `ctx.domain.grid_kind` (the type keeps its name) and its renderer feeds the Session through
  `fem_field(component)`.
- MeshSpec.sweep_axis 'z' (swept prism layers, model sub file) is neither the Kuhn split nor gmsh 3D: code branching
  on `sizing.mesher` checks `sizing.sweep_axis` first.

### 1D models (model sub file "1D models")
- `DomainSpec.size_y` None = 1D (`ModelSpec.dimensions` 1; a type runs there only with 1 in
  `PhysicsType.dimensions`): internally a padding-free 2D `Domain` one cell across on `Coordinates(line=True)`, so
  every 2D path runs unchanged. Code that must know branches on `spec.dimensions` / `DomainMap.line` /
  `Coordinates.line`, never on ny == 1. Outer faces x_lo / x_hi only (`spec.outer_faces`). Automatic solvers resolve
  per grid kind (`schema.solver_grid(spec)` 'line': solvers sub file "banded").

### 3D models (core sub file "3D contracts")
- `DomainSpec.size_z` None = 2D (every 2D path unchanged), a value = 3D: `make_domain` returns `core.domain3d.Domain3D`
  (`dimensions` 3; `Domain.dimensions` 2), arrays `(nx, ny, nz)` C-order (z contiguous), every per-axis accessor
  returns 3-tuples. Code shared by 2D and 3D unpacks `[:2]` or branches on `domain.dimensions`, never on shapes.
- 'polar' + size_z = a cylindrical (r, phi, z) sector with logical (r, phi) cells: 3D code taking xy positions from
  `host_centres` / `rasterize_box` branches on `domain_map.mapped` (model sub file "Cylindrical (r, phi, z) sectors").
- z cells are graded like x / y: 3D code reads per-axis widths (`cell_widths()`, `grid.dz_arr`), never a scalar dz.
  Object shapes at a height only through `geometry` (prisms may be extruded along x / y or tapered: model sub file "3D
  prisms").
- `FACES` stays the 2D tuple (PML sides, BCs, templates); `FACES_3D` / `spec.OUTER_FACES_3D` add z_lo / z_hi, which
  are reserved names in every model but valid entities only in 3D (`spec.outer_faces(spec)`).
- `PhysicsType.dimensions` (default (2,)) decides where a type may run (validate); a type whose vocabulary changes in
  3D is a separate 3D-only type (the `*_3d` types), else one type with (2, 3). Results stay 2D cuts
  (`Session.field(..., plane=(axis, position))`; model sub file "3D field access").
- 3D builds never allocate full-grid float temporaries: materials are painted per z-slab run, full-grid bool / int16
  build masks only through BuildContext 3D helpers, `eval_field` of x, y gives an (nx, ny, 1) layer and of z a
  `BoxValues` on the mask's bounding box. z is a field coordinate only in 3D (`expressions.THREE_D_COORDINATE_NAMES`;
  reserved everywhere, a validation error in 2D).

### Solver contract
- Contract: `core.solver_base.PhysicsSolver` plus `renderer()` (not on the ABC, but called); `state()` keys are
  public, unknown slots raise. Quasi-static solvers (steady heat, potential, FDFD) have dt_limit inf, solve in
  `step(t, inf)` and record probes with `replace_same_time=t > 0.0` (a re-solve at the same time replaces the sample).
- Studies reach solvers only through the runtime-checkable Protocols of `model/solver_hooks.py` (`isinstance`, never
  `hasattr`). The builder raises right after create when a solver satisfies none of the `StudyType.solver_protocols`
  of a study its type lists, so a new solver variant of a type needs the hooks of every study the type lists (unless
  validate refuses that study for the variant), and a new study hook goes into its Protocol, never a bare method name.
- Snapshot solutions (GUI runs only: `RunControl.keep_solutions`; model sub file): a solver whose state a study
  changes between snapshots (time stepping, per-wavelength / per-frequency / per-iteration solves) implements
  `solver_hooks.SolutionStateSolver`, else its intermediate snapshots show only the stored fields; a new renderer
  input changed during a study joins its `solution_state`.
- Work counters: a new quasi-static solver adds `work_counters()` (not on the ABC; summary.json `solvers`, model sub
  file "Timing") with cumulative, machine-independent counts (iterations, factorisations) besides any seconds, on its
  role runtime when the solve is a solver role, else on the physics (never both).
- Nonlinear loop state that warm-starts later loops (saturation, k(T), ...) is cleared when the nonlinear set of
  cells / elements changes or empties. Fixed-point maps on conjugated fields are only real-linear: their mixing /
  least-squares / Krylov coefficients are real.
- scipy `csr + csr` drops entries that sum to exactly zero, so the pattern depends on the values: an operator whose
  factor / plan is refreshed in place is summed with `fem.assembly.pattern_sum` (or one COO list), never chained `+`;
  a shifted pencil K - sigma M likewise (`solvers.arpack_shift_invert.pattern_combination`: at sigma = 0 `K - 0 * M`
  would lose M's own entries).

### Physics definitions
- Catalogue (types, couplings, studies, default instances): `api_reference.md`, physics/<pkg> sub files, coupling sub
  file "Registered coupling types", model sub file "Study catalogue". Polarisations are named by the out-of-plane
  field (`ez`, `hz`), never TE / TM. Same-named material properties of several types (rho, k, cp, ...) are one
  property: `register_physics` raises unless kind, unit, bounds and flags agree.
- A new EM type sets `family='em'` (coupling/definitions builds heat_em_thermo_optic / em_heat_joule from the
  registered family-'em' types), keeps `state()['q_accum']` (None allowed) and the EM loss protocol when it absorbs
  (`coupling/definitions.py` header), takes thermo-optic writes through `index_reference.INDEX_SLOTS`, and builds its
  `source_slots` with `definition_common.source_slot_docs`.
- A feature that reads another physics' solved state reads it through a study hook after that physics is solved
  (optomechanics: the study hands `runner.physics[...]` to the coupling's evaluate; frequency_response:
  `solver_hooks.PhysicsDriveSolver`), never in its apply: build order is not solve order. Such a read is no coupling
  (coupling sub file "Not a coupling").
- A type solving on several mesh kinds (one solver class per kind) keeps the rules every kind follows as module
  functions of its tensor `solver.py` that the element solvers call, never a copy (potential / heat sub files).
- Source slot writers: every CouplingType declares `writes(vctx, spec)` (schema comment); validate allows one
  overwriting writer per (target, slot) or several contributors of one kind and one every, summed by one SlotSum per
  slot. A new coupling into an existing slot is a contributor (API: coupling sub file "Additive source slots"). A
  coupling create (also in a physics definition) takes `after_source_step` from `index_terms.after_source_step(ctx,
  spec, every)`, never a stride test of its own.
- Axisymmetric (r, z) is the per-physics setting `geometry`, always through the `definition_helpers` 'geometry'
  helpers (`geometry_param`, `is_axisymmetric`, `AXISYMMETRIC_SPACE_SETTINGS`, `axisymmetric_issues`), so moving it
  to the domain touches one place; the BOR types carry it with the single choice 'axisymmetric'. The solver exposes
  `axisymmetric` (Session ring-weights the derived values reading its fields). With `azimuthal_order` m != 0 rings
  carry pi: code weighting (r, z) rings takes `core.metric.ring_angle` / the solver's `ring_angle`, never a literal
  2 pi. Orders may differ only for a `definition_helpers.ORDER_SPLIT_READERS` pair; a ring-weighted derived value
  whose angle is then ambiguous is judged by `ring_angle_conflict` (Session raises, validate refuses it). A symmetry
  option that changes what the drawn plane is becomes a space (`schema.SPACES`, GUI only), never a free-standing
  setting.
- Help texts are API contract (describe -> api_reference.md): units, sign / axis conventions, typical values and
  gotchas, ASCII only (describe raises); describe prints the Param's unit, default, range and choices, so the help
  need not restate them. Unit '1' = dimensionless, '' non-numeric; units with spaces ('W/(m K)'), never '*'; a unit
  that depends on the space goes in `Param.space_units`. A gotcha a user can hit belongs in a help text or validation
  message, not only here.
- GUI texts (`Param.tip`, `choice_labels`, every `summary`, `PhysicsType.mesh_method`; empty falls back to help /
  description; describe ignores them) are for a first-time user: no abbreviations, formulas or snake_case keys, refer
  to settings by their label. Labels appear in both.
- Shared names (industry-standard if any): magnitudes `<name>_abs` (u_abs, e_abs, heat_flux_abs); resonance tables
  resonance_number / frequency / frequency_imag [Hz, > 0 decays] / q_factor (+ region_energy_fraction last); mode
  results n_eff (real part) / n_eff_imag; 1-based mode choices `mode_number`; a body force feature `body_force`
  (params f_x / f_y, slots and coupled fields body_force_x / _y); probe choice `component`. Field names are scoped by
  physics: u is the velocity in laminar_flow, the displacement in solid_mechanics.
- Time t (simulated time, 0 outside a time_dependent study): state_dependent material properties and
  `Param.time_dependent` feature params (parameters and t only) may use it, re-evaluated before every step at its
  start time (implicit steps see a one-step lag). A feature param gets the flag only together with an incremental
  setter in its apply (`BuildContext.follow_time`; model sub file "Feature values of t"); likewise
  `Param.follows_parameters` needs a `BuildContext.follow_parameters` setter (an apply that skips it on some grid
  kind makes the sweep warn and build every value: `sweep_reuse.unfollowed_values`); followed values are
  range-checked per sweep value only in the setter (material values are, by their Param). `state_dependent` on a
  material property promises a full re-read of its array on `materials_changed` (model sub file "Reused sweep
  builds").
- Selftest structure:
  - Cases span each module as light as possible: every physics type, feature, solver role and mesh kind it supports.
  - Each case runs once, on its natural backend; each example once, on its own (`solver_defaults.backend`: an
    example that needs the GPU sets 'gpu').
  - Examples are included only if they have a sufficiently accurate analytical solution (a derived value with a
    reference or an analytic curve); the others are not selftests and are not run or listed.
  - Each physics type and solver runtime has at least one cpu run and one gpu run; a `*_cpu_gpu` parity case counts
    as both.
  - A backend variant is only for a module with no other case on that backend. Precision variants are a separate
    question and stay.
- Self-tests: `selftests*.py` modules hold SelfTestCases and template factories; a definition's `register` registers
  types, templates, then its selftest loader (`registry.register_selftests`); a new template's (definition, selftests
  module) pair joins `TEMPLATE_SOURCES`. `check` gets each run's live Results (closed afterwards) and may use only
  `status`, `summary()`, `field(name[, plane=, unfold=True, native=True])`, `snapshot_field(index, name[, plane=])`,
  `volume(name)`, `result(physics, feature, **options)`, `sector_result(physics, feature[, combined=True])` and
  `sweep_field(index, name)` (helpers: `model.selftest_helpers`). Backends: `SelfTestCase.backends` (model sub file);
  another backend or solver precision with its own tolerance is a `SelfTestCase.variants` entry (solvers sub file),
  not a copied case. Set `SelfTestCase.heavy` on a case that took >= 5 s (record wall_time). Case and template notes
  live in the sub files.
- Self-test modules never rebind a module-level constant later in the file (later cases silently read the new value).
- Self-test runs store only the final state (`StudyType.final_state_settings`): a case whose check reads
  intermediate snapshots sets output_interval / n_snapshots itself, an example names the snapshot in an analytic
  curve (open_sol sub file "api.run").
- Selftest reuse (`model.selftest_cache`): with an out_dir, an unnamed case whose last run passed is not rerun while
  the package code units it depended on (traced per function) and the environment are unchanged; CLI `--reset` (api
  `reset=True`) reruns all. A dependency the tracer cannot see (a non-.py data file read at run time, code run in a
  subprocess) needs `--reset`, or its reader adding the file to the traced set. Built-ins are never reused.
- Selftest reuse and shared code: editing a `selftest_cache.UNTRACKED_PATHS` file (shared plumbing) reruns nothing:
  run `selftest --reset` (e.g. overnight) after such edits; a new shared plumbing file joins that list. Tracked code
  every case runs still reruns everything when its module-level code changes; untracked code must not call into a
  model-specific module for every build (`materials._is_crystal_entry` checks the dict itself instead of running
  crystal_tensors code), or every case depends on that module. An in-process cache of expensive results (meshes,
  factors) kept across api runs hides code from the tracer: scope it to one `run_study`, as
  `meshing.element_meshes_kept`, `cudss_factor.cudss_orderings_kept` and `host_direct.host_analyses_kept`. A disk
  cache across processes is off inside the self-tests (`gmsh3d.default_disk_cache_off`).
- Timing on a loaded shared machine: threaded OpenBLAS can slow host SuperLU factorisations 50-300x (seen 0.1 s ->
  30 s), so time self-tests and examples with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
- Solver roles: a physics' numerical jobs are `PhysicsType.solver_roles`; the build creates their runtimes in a stage
  of its own and the physics create takes them from `ctx.solvers(name)`, never building one itself (scripts still
  may). Solver knobs are never physics settings (a new type declares a role, a shared one from definition_helpers
  when it fits). Validate, estimators and every other reader take a role's type and values from the one resolution
  `schema.resolved_solvers` (`ValidationContext.solvers`, `MemoryContext.solvers`), never a physics setting or a
  node's raw settings. A model / template / self-test that needs other values (a CG tolerance) adds a solver node
  (`PhysicsSpec.solvers`; self-tests: `selftest_helpers.with_exact_cg`). Automatic per backend / grid / study, study
  defaults that do not fit, opt-in back ends, precision: solvers sub file "Model path".
- Subpixel smoothing and conformal boundaries are model-wide (`solver_defaults.subpixel_smoothing` /
  `conformal_boundaries`), never physics settings: a type honours them through its `PhysicsType` flags (schema
  comments: the builder hands the solver the geometry / locator right after create). The averaged eps never enters
  the MaterialGrid: every face / cell eps consumer of an honouring type reads the averaged values (em sub file
  "Subpixel smoothing"). Conformal: cell masks stay the cell-centre staircase (sources, ports, couplings, results
  read them); only the operator links at the wall change, each correction limited so the time step / conditioning is
  unchanged (the smallest cut never sets dt); honouring types and the locator: core sub file "Cut geometry".
- Estimates mirror the solver: a setting where 0 = auto is resolved through the solver's own resolver (e.g.
  `fluid.solver.steady_march_steps`), never used raw. Every memory budget check judges
  `meshing.fitted_memory_estimate` through `meshing.backend_memory` at `schema.memory_check_level(spec)` (model sub
  file "Memory estimate").
  A new physics / allocating feature MUST add its memory estimator (`PhysicsType.memory_estimate` /
  `FeatureType.memory_estimate`), counted from its own allocations (not the MaterialGrid arrays): the big arrays the
  solver holds, summed where they are held together, sizes derived from the code (stencil entry counts, dtypes),
  never fitted to measured bytes, so it sits just below the measured bytes. 'over' (books what the solver does not
  hold) and 'unaccounted' (an array no item lists, or memory the code could free) are leads on the solver
  (`builder.estimate_verdict`). A new allocation in an engine updates its estimator (else the build ledger warns);
  anything allocated lazily after `finalize` is a run item, or a transient item of its phase when it is freed again
  (phases never overlap, the estimate takes the largest: split a solve into phases in the order the code allocates;
  build temporaries are the part's build_extra). Factors outside the pools: cuDSS an external item from
  `cudss_factor.factor_device_bytes` (`definition_helpers.cudss_external(phases=)` when released after its solve),
  MUMPS / PARDISO a host item from `host_direct.factor_host_bytes` (or SuperLU x `superlu_bytes_scale`) plus
  `solver_host_bytes` (uncalibrated). Item kinds: `core.memory_estimate.MemoryEstimate` docstring.
  A new physics / solver type adds its run-time recipe (`PhysicsType.time_estimate` / `SolverType.time_estimate`:
  machine-independent op counts, never seconds; iteration counts derived a priori, never fitted to run counters: a
  count deviation is a lead on the solver; a kernel bound by arithmetic also counts its `Elementwise.flops`, since
  float64 runs at 1/64 rate on consumer GPUs: core sub file "Run-time estimate"; model sub file "Run-time estimate
  rules").

### GPU memory and session lifetime
- GPU functions only on the GPU-owning thread (the worker, or the `api.run` caller); `gpu_memory.format_bytes` is the
  only one the GUI thread calls. cupy's pool runs under a budget: overruns raise OutOfMemoryError. Pool releases go
  through `gpu_memory`, never a bare `free_all_blocks`, and never in step loops: `release_all` between jobs / builds /
  sweep values, `release_cached_blocks` (or `pool_room_bytes(release_cached=True)`) mid-run before sizing to the free
  device memory (core sub file "GPU memory").
- Memory outside the cupy pool (a cuDSS factor, the cusolver_qr workspace) or outside numpy (a MUMPS / PARDISO
  `host_direct.HostDirectFactorization`) is invisible to the budget and the ledgers: code holding a factor (a
  ModalPencil, a harmonic system's factor) calls its `release()` before dropping or replacing it, or passes it as
  `previous=` to the next factorisation of the same pattern. Host direct threads, the PARDISO / OpenBLAS clash and
  its memory guard: host_direct.py header (rules across files: solvers sub file "host_direct").
- A traceback keeps a failed build's frames and locals alive: inside `except` record only text, call
  `release_session` after the block; an exception kept or re-raised later goes through
  `gpu_memory.drop_exception_frames` first.
- Session close / release, OOM handling and the build ledger: model sub file ("Build memory", "Session lifetime",
  "GPU OOM").

### Performance
- No per-step host syncs or allocations in hot loops: preallocate, update in place, fused kernels. The accepted syncs
  are listed in the core sub file "Accepted host syncs" (a new one goes there).
- Host numpy over elements x points: never a broadcast `np.matmul` of 2x2 / 3x3 matrices per point or
  `np.linalg.det / inv` on 3x3 stacks (numpy loops per batch, no BLAS: write components out, e.g.
  `fem.mesh3d.span_adjugate`, or one GEMM over shared rows), never `np.add.at` (bincount: `fem.assembly.scatter_local`,
  `fem.fixed_pattern.scatter_add` (complex-safe) / `scatter_rows` (trailing axes)); a bincount costs its whole index
  window, not its entries: element chunks whose indices span the target (P2 numbering: nodes first, edge dofs last)
  scatter directly instead (`FixedPatternCsr.add_elements`, numba); mesh-constant integrals are cached per dof map,
  derived fields computed on read (fem sub file "Dofs and assembly").
- RawKernel scalars are numpy scalars of the declared C type (`DTYPE(...)`, `np.int32(...)` cached like
  `self._nx_i32`, `np.float64(...)`; `real_t` = the field dtype of a RealRawKernel); Python floats / ints silently
  corrupt them. Bool arrays: cupy bool, `const bool*`. A kernel a cpu-capable physics launches is a `core.kernels`
  twin, its numpy (and numba) twin written in the same edit (core sub file "Kernel twins").
- Non-uniform spacing and CPML kappa are folded into the 1D `inv_dx/inv_dy` arrays: never branch kernels on
  uniformity.
- A `cp.ReductionKernel` reducing a whole array to one scalar runs on one thread block (RTX 3050: ~13 GB/s float32 /
  25 GB/s float64, vs ~200 GB/s for `cupy.cublas.dot(out=)` or a multi-block RawKernel; 1e6 float64: 636 vs 84 us):
  a full reduction in a hot loop uses a multi-block one (`solvers.pcg.device_vector_dot`), or reduces the shortest
  axis first (`core.kernels.axis_first_reduce`, for fused ReductionKernels of >= 2D arrays).
- cupy refuses cuSPARSE calls during stream capture (NotImplementedError), so a CUDA graph cannot contain cupyx SpMV:
  use the CSR RawKernels of `solvers.pcg` (`CSR_ROW_SOURCE`, `DeviceCsrProduct.apply(vector, out)`; solvers sub file
  "pcg"). Below ~1e6 unknowns a cupyx SpMV costs more host time (~60 us: descriptors, workspace, output allocation)
  than device time.
- Launch-bound steppers replay their fixed-argument launches as CUDA graphs (gpu only; one eager run per key, then
  capture; a failed capture warns and stays eager): grep `_fixed_block`, `_run_fixed` (multi-step blocks behind the
  runner hand-off `advance_ticks`: coupling sub file) and pcg `capture_blocks` (a multigrid V-cycle launches its
  kernels into the CG block's capture, never a graph of its own inside it). A graph holds array pointers and scalar
  values of capture time: every array a block reads is rewritten in place, or its id is in the graph key and the
  entry keeps it alive; every scalar is fixed for the engine's life or in the key. Inside a block: no host sync, no
  allocation (also not cupy's copy of an input that overlaps the output, e.g. `a[:, 0] = a[:, 1]`), no kernel with
  per-step scalars (sources, poles, walls, monitors stay outside, unless a device counter replaces the scalar: probe
  rows `core.monitors.FieldProbeRows`, pulse tables), no timed profiler part. A new kernel argument of a captured
  launch follows these rules.
- A per-frequency (or per-iteration) refactorisation of a fixed-pattern pencil goes through
  `SparseDirectSolver.pencil` + `factorize_pencil` (a cuDSS DevicePencil; a HostPencil on MUMPS / PARDISO), never a
  host A(omega) into `factorize_host`: the host canonicalising, asymmetry check and lower-triangle take cost more than
  the cuDSS factorisation itself (measured ~250 of ~370 ms per frequency), and factorize_host with `previous=` still
  pays them although it refreshes the values on the device (solvers sub file "sparse_direct").
- Point location dominates reads on element meshes (renderers, checks, derived values): locate through
  `TetrahedronDomain.locate` / `TriangleDomain.located` (remembers point sets, a kept read-only points array by
  identity, and on triangles the sampling operator per dof map; read-only results), never a fresh locator or points
  array per component or plane. A selftest whose time split shows a large 'other' is paying for its check's reads
  (model sub file "CLI time split").
- Allocate where needed (mandatory for new engines): pole arrays on their bounding box, optional accumulators once
  enabled, build temporaries only in BuildContext caches (`release_caches()`); status of the EM engines: physics/em
  sub file "Registered EM types".

### Generated output and git
- After any change to the registry, a help text, a template or library data regenerate `api_reference.md` and
  `api_templates.md` (`python -m open_sol describe --update-reference` from the package's parent) or selftest
  `api_reference_current` fails. The same command rewrites `structure_index.md` (in a checkout): needed after adding,
  moving or removing a module or changing the first sentence of a module docstring (selftest
  `structure_index_current`). Every new module starts with a docstring whose first sentence says what it holds.
- Every folder a tool writes results into (its default `--out` too) is named `out/` or ends in `_out/`: the tracked
  `.gitignore` ignores both, so a new tool needs no new rule. Example outputs (`examples/**/*_out/`) are not
  committed either.
- The tracked `.gitignore` is public and generic; private folder names (experiments, plans, articles, side projects)
  go in `.git/info/exclude`, which every worktree of this clone shares and which is never pushed.

## 3. Sub files (`sub_structure_conventions.md`)

- `open_sol/`: api.run (GPU thread, private copy, failures), Results lifetime, parameters of drawn results, one
  `_figure_jobs`, describe entries of new api / CLI arguments.
- `open_sol/mesh_study/`: parts, records and labels, variants.
- `open_sol/fem/`: layout and imports, mesh data, built-in mesher, gmsh, elements and geometry, dofs and assembly
  (FixedPatternCsr, block patterns, device tangents, Bloch), domain / sampling / transfers, scalar diffusion (2D and
  tetrahedra), factor entry count, error indicators, tetrahedra, vector elasticity on tetrahedra, complex scalar
  harmonic systems, mixed u-p, squeeze film, reduced sweeps, line elements, prestress, self-tests.
- `open_sol/core/`: backends, GPU memory, accepted host syncs, kernel twins (numba traps), half storage, graded meshes,
  grid transfers, mirror planes, cut geometry, coordinate metric, 3D contracts, harmonic probe record, resonance
  finder, run-time estimate, damped iteration, DiffusionSystem (`fv_diffusion`; heat, potential, magnetic).
- `open_sol/solvers/`: layout and imports, the model path (spec -> resolution -> build -> summary), pcg, multigrid,
  line_block, sparse_direct (entry points, cuDSS opt-in and 'auto', defer_check, ordering reuse), banded,
  arpack_shift_invert, block_eigen, tried and dropped, host_direct, explicit_kernels.
- `open_sol/coupling/`: runner and couplings (cadence, hand-off, flush, protocol, change check, slots, EM loss
  protocol, not a coupling: study-time drives, extension modules), registered coupling types, couplings across meshes,
  additive source slots (`slot_sum`).
- `open_sol/physics/em/`: type catalogue, imports, FDTD and shared EM (engines, sources, boundaries, Bloch, TFSF,
  monitors, slab modes, mode ports), nonlinear / gain and phase-change media, harmonic FDTD, FDFD (ports,
  eigenfrequency, adjoint, design regions, conducting sheets, focused beams), cross-section modes, conical k_z,
  em_modes, body of revolution (FDFD, FDTD, focused beams), spherical multipoles, the triangle paths and decisions,
  subpixel smoothing, definitions and self-tests, effective index.
- `open_sol/physics/eme/`: layout, sections / mode cache / normalisation, interfaces and cascade, model layer, meshes
  and cuDSS (decision), cascade adjoint, self-tests.
- `open_sol/physics/nlfd/`: coupling to em_fdfd, channels (Bloch k, mirror parity), adding a nonlinear term,
  self-tests, cuDSS / host direct verdict, triangles (decision).
- `open_sol/physics/em3d/`: engine (step graphs, materials, Bloch, mirrors, poles, backends, half_storage),
  cylindrical sectors, subpixel smoothing, conformal PEC, em_fdfd_3d (tensor and tetrahedra), em_fdtd_3d on
  tetrahedra (decision), sources / monitors / definition.
- `open_sol/physics/heat/`: solvers (shared names, values of t, direct back ends, outer grading), thermal noise (and
  its Brillouin counterpart), self-tests (+ the DiffusionSystem notes of core).
- `open_sol/physics/potential/`: solvers (kinds, `_dc` vocabulary, probes), element meshes, self-tests and open items
  (+ the DiffusionSystem notes of core).
- `open_sol/physics/mechanics/`: layout, backends, SolidMechanicsPhysics / ElasticSystem, anisotropic stiffness,
  eigenfrequency (2D, 3D), piezoelectric plane strain, frequency response, guided modes, 2D Bloch cells,
  optomechanical g0, spontaneous Brillouin (brillouin_scattering), squeeze film, ElasticSystem3D, triangle and
  tetrahedron meshes, prestress / damped modes / complex records, geometric nonlinearity (held factor, device
  tangents), thermoelastic damping, viscoelastic moduli and the N-term harmonic system, mixed u-p formulation, reused
  sweep builds, couplings and optics hooks, memory estimates, self-tests.
- `open_sol/physics/membrane/`: layout, model (plate switch, system lifetime), self-tests.
- `open_sol/physics/acoustics/`: elastic_waves (step graphs, kernel / numpy mirrors, adjoint pairs), couplings, not
  built / open (incl. pressure_acoustics).
- `open_sol/physics/fluid/`: FluidPhysics (Bouzidi walls, steady march, float32 bias), convection, self-tests,
  triangles / cuDSS (decision).
- `open_sol/physics/magnetic/`: layout and registration, shared rules (snapshots, saturation loops, memory),
  MagneticPhysics (2D tensor), magnetostatics_3d (tensor, tetrahedra), triangle meshes, magnetic_fields_ac,
  self-tests.
- `open_sol/physics/semiconductor/`: numerics (faces, chord steps, Newton stall, contact currents), self-tests.
- `open_sol/materials/`: descriptions, library optical data, merged per-physics data, crystal entries.
- `open_sol/model/`: model layer (expressions, names, analytic / derived curves and values, curve errors, selftest
  examples / cache / subsets / backends / timing, schema, 3D prisms, polar / spherical / cylindrical domains, 1D
  models, arrays, per-physics meshes / build / Session, triangle / tetrahedron / swept prism meshing, memory estimate,
  build memory, session lifetime, GPU OOM, conformal, subpixel, materials, BuildContext masks, Session reads, ring
  angle of derived values, validation, worker, progress stages, run_study, reused sweep builds, studies, physics
  drives, snapshot solutions, solution feedback, parity sectors, mesh refinement levels, parameter estimation,
  run-time estimate rules, solve flow); registry / templates / describe; import contracts.
- `open_sol/io/`: atomic writes, writer / reader pairs, non-finite floats.
- `open_sol/gui/`: thread rule, bridge state, View in browser, post-processing edits, result plot views and plot
  settings, canvas (display grid, views, sweep sliders, layout, Export plot, interaction, fields), Examples menu,
  windows, documents and node paths, physics menu, memory budget, meshes and mesh refinement / estimation forms,
  element meshes, solver fits, snapshots, GUI texts, colours, polar domains, 3D views, mesh form, solve flow window.
- `open_sol/viz/`: matplotlib figures (visualization (open: BOR annotations), result_plots, curve_figure,
  scene_figure), HTML pages (mode_player, field_viewer: self-contained, plot settings, layers, clocks, start-up,
  overlay, key bindings).
- `experiments/`: paper experiments (`<paper><year>/`, run_all), link to `build_experiments_guide.md` (layout, time
  estimates, cuDSS, model gotchas).
- `tools_profilers/time_profilers/`: layout and imports (nothing here is checked by open_sol), solver_profiler
  (registry choices, result / partial markers, timers, pinned ignored settings, matrix-free problems), case_profilers
  (script layout, wrap before the registry, unresolved parts, timed syncs, markers, profile points, reuse, solution
  check), example_profilers (choice rule, built on case_profilers, engine parts, backend).
- `tools_profilers/memory_profilers/`: vram_profile (private names it patches, pool used vs reserved, memGetInfo
  caveats).
