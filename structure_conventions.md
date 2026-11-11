# open_sol — structure & conventions

Gotchas and cross-file code flow only; signatures and help texts live in the code. Current state, no changelog.
Examples live outside the package (`examples/README.md`); `experiments/` holds the user's research models in the same
format (one subfolder each, loaded by path; not in the Examples menu, not checked by selftest examples_validate).
`gpu_tet_meshing/` is a side project outside the package (plan.md + prototypes; open_sol never imports it).
`dev_tools/` holds developer tooling outside the package (imports open_sol, never imported by it, not shipped):
`dev_tools/time_profilers/solver_profiler` benchmarks every cpu / gpu solver choice (enumerated from the registry) on
problems assembled by open_sol's own operator code and on whole model files; it is not `python -m open_sol profile` (the run-time estimate's profile).
`dev_tools/time_profilers/case_profilers` times the parts of whole selftest runs per solver engine (host hot spots,
syncs); `dev_tools/memory_profilers/vram_profile` measures pool / cuDSS device bytes against the memory estimate.
User-facing quickstarts: `README.md`; the conda install guide (environment, optional cupy, nvmath, MUMPS, PARDISO):
`INSTALL.md`; `pyproject.toml` (setuptools, version from `open_sol.__version__`, numpy / scipy / matplotlib required,
cupy / Qt / nvmath-python / python-mumps / pydiso as extras; ships api_reference.md and api_templates.md as package
data) makes `pip install -e .` work, else run from the repository root.
Folder-specific gotchas live in `sub_structure_conventions.md` next to the code (index in section 3): read those of the
folders you work in. Only gotchas every agent needs stay here.

## 1. Structure

What each module holds: `structure_index.md` (generated from the module docstrings, one line per module; read the
folders you work in by its Contents line ranges). Folders:

```
open_sol/
├── __init__.py      # public API; importing open_sol loads cupy when it imports, else runs on stand-ins (core.backend;
│                    # nothing allocates at import)
├── api.py           # headless (no Qt): describe, template / load / save, check, scene_figure, run, selftest (+ the
│                    # built-in cases), Results, RunError, ProgressPrinter (also the CLI's)
├── __main__.py      # CLI describe | template | check | run | show | selftest | mesh-study | profile; check / run / show
│                    # take --set NAME=EXPR; exit 0 ok / 1 model, run, selftest or doc-check failure / 2 usage
├── api_reference.md # describe() text for agents ('## Contents' of line ranges: read it in offset / limit slices);
│                    # api_templates.md the template JSON; both written only by `describe --update-reference`
├── simulation.py    # Simulation: Domain + named physics + one MultiphysicsRunner (scripting API)
├── core/            # constants (DTYPE, C0, FACES, physical constants: import, never redefine), grids, graded meshes,
│                    # Domain / Domain3D, MaterialGrid, PhysicsSolver ABC, backend (cpu / gpu), kernel twins, GPU / host
│                    # memory, memory and run-time estimates, DiffusionSystem (fv_diffusion: heat, potential, magnetic),
│                    # metric, mirror planes, cut cells, grid transfers, resonance finder (core sub file)
├── materials/       # library Materials (one file each), REGISTRY, effective_material_data, crystal_tensors
├── physics/<pkg>/   # one package per physics family: solvers, renderers, definition*.py (registered types, features,
│                    # templates), selftests*.py and a sub file: em (2D FDTD / FDFD, em_modes, BOR, conical, nonlinear /
│                    # gain / phase-change media, adjoint and design regions), em3d (3D FDTD / FDFD / tetrahedral FEM),
│                    # eme, nlfd, heat (+ thermal noise), potential, mechanics (2D / 3D, modal, piezo, optomechanics),
│                    # membrane (film / plate, triangles + 1D, ports, Bloch), semiconductor, acoustics (elastic_waves;
│                    # frequency-domain pressure_acoustics), fluid, magnetic (static, AC, 3D)
├── solvers/         # solver roles, SolverTypes (definition.py) and the role runtimes: pcg (+ multigrid), line_block,
│                    # sparse_direct, banded, arpack_shift_invert, cudss_factor, host_direct (MUMPS / PARDISO),
│                    # explicit_kernels (solvers sub file)
├── coupling/        # CouplingTerm / CouplingGroup, DynamicMaterialUpdate, MultiphysicsRunner, E13 SlotSum, the
│                    # coupling type definitions per partner family (definitions*.py) and their self-tests
├── model/           # spec, schema (registry), expressions, geometry, validate, meshing, builder (-> Session), session,
│                    # studies (study*.py), describe, templates, memory / time estimates, selftest machinery (cache,
│                    # examples, import_contracts, structure_index)
├── fem/             # triangle and tetrahedron meshes, mesher, gmsh I/O, elements, dofs, assembly, sampling, FEM
│                    # diffusion / elasticity / magnetostatics systems, complex scalar harmonic systems, PML
│                    # stretches, neutral resonance solve, 1D line elements (fem sub file)
├── io/              # model JSON and exports (model_io), run directories (results_io)
├── viz/             # Qt-free matplotlib figures: Visualizer, result_plots, curve_figure, scene_figure
├── mesh_study/      # solver_plan S7.8 study (CLI mesh-study)
└── gui/             # PySide6 + pyqtgraph application (gui sub file)
```

Import rules live as data in `open_sol/model/import_contracts.py` (`CONTRACTS`: importers, allowed / refused targets,
module level or anywhere, named exceptions, each with its rule text); the built-in selftest import_contracts checks every
import of the package against them and flags stale rules. A new layer or exception is an edit there, not here. The
ones that shape everyday code:
- `core` imports nothing outside core but its listed exceptions (solvers.pcg in fv_diffusion; the rest inside
  functions). cupy / cupyx at module level only in `core.backend` (others: `from <..>core.backend import cp` /
  `cupyx_sparse as cpsp` / `cupyx_ndimage as cpndi`; a real `import cupy` only inside a GPU-only function), so open_sol
  imports without cupy. scipy (installed: use it on the host where cupyx falls short: no `eigs`, no shift-invert,
  `spsolve` refactorises per call) only inside functions. PySide6 / pyqtgraph only in gui. The optional direct solver
  wrappers python-mumps (`mumps`) / pydiso only inside functions of `solvers.host_direct`.
- `physics/<pkg>` <- core, solvers, fem and model.schema / spec / geometry / expressions / definition_helpers /
  definition_resonance (a definition registering a coupling type also coupling.coupling_term / index_terms); never the
  model runtime (meshing, builder, study, session, validate): shared study rules go to core or
  model.definition_helpers. `open_sol.materials` only inside functions (materials imports physics.em). physics.em is
  upstream (em3d, eme, nlfd, heat, ... import it, never the reverse); acoustics imports mechanics, never the reverse.
  Chains inside physics/em: em sub file "Imports".
- `selftests*.py`: at module level only core.constants, model.spec / schema / selftest_helpers and sibling selftests
  modules (reuse `model.selftest_helpers`, never a local copy: SubChecks for pass / fail sub-checks, worst_result /
  scaled_worst_result / normalized_result / tolerance_units_result / failed_count_result, relative errors,
  solver_role, probe_phasor, skip_result; repeat type names as literals); physics modules and
  scipy inside their checks. Nothing else imports a selftests module at module level or in `register` (contract
  selftests_lazy): a definition imports its cases inside its selftest loader, so `get_registry()` (GUI, check,
  describe) loads only the selftests modules that hold templates (`TEMPLATE_SOURCES` in import_contracts).
- `get_registry()` imports `model.study`, `solvers.definition` (the SolverTypes, before any PhysicsType.solver_roles is
  checked), every `physics/<pkg>/definition.py` (alphabetical), then `coupling.definitions`, calling each
  `register(registry)` (coupling.definitions then discovers every `coupling/definitions_*.py` and calls its
  `register(registry, physics)`: a new coupling topic needs no central edit); calling it inside a `register` raises.
  model.study imports model.builder at module level:
  nothing builder / session imports may import model.study.
- `api` is imported only by the package root, `__main__` and mesh_study (gui only inside a worker-thread function);
  gui is imported by nothing.
- Cycles broken by an import inside a function (Python raises at import time otherwise): coupling/definitions imports
  its `definitions_*` modules (by importlib) inside `register`; model.meshing imports triangle_meshing / tetrahedral_meshing inside
  functions; physics/em3d `definition_fdfd3d` is imported inside its package's register.

Code flow: `Simulation(size_x, size_y, dx)` / `from_mesh(cfg)` = Domain + one EMPhysics 'em' (`polarization` 'ez' /
'hz'); `Simulation.multiphysics(domain or [domains])` + `add_physics(name, solver, every[, domain])` + `add_coupling`
otherwise; unknown
attributes fall through to a single physics. `run` = raise when nothing steps in time, `Simulation.finalize()` of
the physics not finalized yet (`warnings.warn` of their `finalize_warnings`; `_finalized_names` keeps a later run
from re-recording heat n0 from heated eps), `runner.prepare()` (`warnings.warn` of its `prepare_warnings`, every
`run`), chunks of `advance(n, t_end)`, `flush()` (skipped with `flush=False`: chunked scripts flush only the last
chunk, else every chunk adds a transfer). Model path: ModelSpec -> `validate` (CPU, Issues at GUI node paths)
-> `build` (worker thread: validate, parameters, geometry, domain, static materials, per physics `create` +
`add_physics` + feature `apply`, dynamic updates `apply_now`, `sim.finalize([name])` per solver (then its
`finalize_warnings`, if any, are logged as warnings and its optional `finalize_notes` as info lines, e.g. assembly
timings), couplings, `runner.prepare`, `release_caches`) -> Session;
`run_study` builds once per sweep value (SweepSpec.reuse_build: once, followed feature values re-set per value, model
sub file "Reused sweep builds"; StudySpec.adaptation: once per mesh refinement level,
`study_adaptation.run_adaptation`; StudySpec.estimation: once per evaluation, `study_estimation.run_estimation`) and runs the StudyType's `run`: time_dependent (`runner.advance` chunks),
stationary (`solve_stationary` over every physics), frequency_domain (harmonic physics = solvers with
`solve_harmonic`, background = the rest: `solve_stationary` on the background, every background -> harmonic coupling
`apply_now` once, then the pulse of the time-stepped harmonic physics (solvers with `attach_harmonic`: em_fdtd_*,
`_run_harmonic_pulse` steps only them, no coupling, until the window energy decayed; then S(omega) normalisation),
then (after the optional hook `plan_harmonic_frequencies(omegas)`: FDFD / em_fdfd_bor / em_modes line-block batches) per wavelength
in solve order `select_harmonic_frequency(index)` / `solve_harmonic(omega)` of each harmonic physics + one Snapshot labelled 'lambda = <nm> nm'; the last wavelength stays live; with a coupling / material reading
a harmonic physics instead per wavelength `omega` set + `solve_stationary` over every physics, continued from the
previous wavelength: `StudyType.coupled_points`), mode_analysis (the same background step, then
`solve_modes()` of every solver that has it and one Snapshot per mode number via `select_mode(k)`, labelled 'mode k:
n_eff = ...'; `select_mode(mode_index)` stays live), eigenfrequency (the same background step, then
`solve_eigenfrequencies(n)` and one Snapshot per resonance via `select_resonance(k)`, labelled 'resonance k: lambda =
... nm, Q = ...'; resonance 1 stays live), optimization (scipy L-BFGS-B over the design_region densities: per evaluation
write eps, `solve_harmonic` + `eps_sensitivity()` of every FDFD / em_eme physics, chain rule; optional projection-beta
continuation stages; one Snapshot per accepted iteration labelled 'iteration k: J = ...', design_density stored by
default; the returned design stays live). Headless: `api.check` (CPU -> check.json), `api.run`
(run_study on the calling thread -> Results + run directory), `api.selftest` (-> selftest_report_<subset>.json),
`api.selftest_examples` (api.run of every analytic example, `tolerance` of derived values / analytic and derived
curves -> selftest_report_analytic.json); the CLI wraps them (`selftest --examples`); file formats in `describe('results')`.

## 2. General gotchas

### Arrays, coordinates
- Per-cell state: float32 (`DTYPE`) arrays of the run backend (`domain.xp`: cupy on 'gpu', numpy on 'cpu'; code
  shared by both backends takes xp from the Domain or `core.backend.array_module_of`, never `cp` directly; core sub
  file "Backends"), `(nx, ny)`, `[i, j]` i along x, SI, full grid incl. the PML ring (window
  from `(ox, oy)`); renderer / Session fields and FieldData are window-shaped. Array dataclasses: `frozen=True,
  eq=False`. Exceptions: a 'half_storage' stepper (em_fdtd_3d fields, elastic_waves fields, laminar_flow populations)
  stores float16 in scaled units: code reading or writing those arrays converts through the engine's scale or refuses
  half_storage, never with cupy's own float16 kernels (core sub file "Half storage"); a 'double' stepper (em_fdtd_ez /
  _hz) steps float64 fields: code adding to them takes the engine's `dtype` (physics/em sub file "Stepper
  precision"). Renderers and results always return float32 physical values.
- numpy 2 reprs scalars as 'np.float64(0.25)': text for other programs (gmsh .geo, scripts, CSV headers) takes
  `float(value)` / `.tolist()` first, never `{value!r}` of a numpy scalar.
- Coordinates only from `Domain.cell_widths/cell_edges/cell_centers` (float64, cached read-only: copy before
  writing); Yee stencils and H_z source weights use `yee_grid.grid_cell_widths(grid)` instead. Face names:
  `core.constants.FACES`. They are logical: on a polar `Domain.coordinates` (model: `DomainSpec.coordinates` 'polar'
  / 'spherical') x is r - radius_min and y an angle in radians, and areas / volumes come from `core.metric` (core sub
  file), never from products of widths; a physics runs there only when its `coordinate_grids` list the grid
  (`Simulation.add_physics`; the model: `PhysicsType.coordinate_grids`). Model geometry stays in the drawing plane:
  code mapping between it and the cells goes through `geometry.DomainMap` (model sub file "Polar / spherical
  domains"), never through the logical widths. Vector / tensor components keep their _x / _y names in the grid frame
  (radial / angular there; a renderer relabels them with `core.renderer.polar_field_infos`).
- `pos_to_cell*` = containing cell (1e-9-cell snap, an edge coordinate goes to the upper cell), clamped;
  `validate_physical_pos` and the model point check inclusive within 1e-9 max(size_x, size_y). Spans / boxes cover `[cell(lo), cell(hi)+1)`: an edge exactly on a cell edge
  takes one extra cell. Rectangles / polygons test centres half-open, circles / ellipses closed.
- `coordinate_grids(clamp_to_physical=True)` gives PML cells the nearest window centre (masks, x/y expressions);
  `rect_mask/circle_mask` use unclamped centres. X / Y are zero-stride `broadcast_to` views of the 1D centres (cupy
  does not make them read-only): consumers stay elementwise, never write into them (in-place ops only on results).
- `Domain.size_x/y` are requested; `physical_extent()` is the meshed window (use it for display). `Domain.dx` =
  dx_min on graded grids. `boundary_config=None` pads `npml` (15) on every side (PML cell grading: core sub file
  "Graded meshes").
- `npml_sides` is padding, not only PML: an EM mirror face (symmetric / antisymmetric) pads one ghost cell of the
  window edge cell's width (`BoundaryConfig.ghost_faces`), so the mirror plane is the window edge for every physics
  of the model (heat / potential faces, drawn geometry). Code sizing CPML layers reads the bc type, not the count.
  em_fdtd_3d mirrors likewise (em3d sub file). Membrane symmetry / antisymmetry faces are model mirror planes too
  (`core.mirror.MIRROR_FEATURES`): code listing mirror faces calls `mirror_faces(spec)`, never reads bc settings.
  Results count the unfolded structure (core sub file "Mirror planes"):
  integrals / fluxes / held currents reaching a mirror face x 2 per face, derived max / min over the images too, 2D / 3D far fields with the image currents
  (results also carry the complex amplitude_re / _im), unfolded fields; a new signed renderer field declares
  `FieldInfo.parity`, a new total over a selection / line uses `touch_factor`, a new far-field result its complex
  amplitude (parity sectors combine it per angle), a new result that parity sectors should combine a rule's extras
  (`combine_sector_result`: sector_rule 'linear', sector_modes).
- `geometry.resolve_geometry` returns more names than `spec.geometry`: array copies ('<array>[i,j]',
  `geometry.is_array_copy`) are resolved shapes of their own, so meshers and outlines see every copy; code listing
  objects from the resolved dict skips them (model sub file "Arrays").
- Courant: `courant` / `courant_factor` always = fraction of the 2D CFL limit of the finest cells, in (0, 1], default
  0.99; dt only from `core.grid.courant_time_step` (Grid2D, NonUniformGrid2D, `meshing.estimate_mesh`); uniform
  cell counts and cell edges likewise only from `core.grid.uniform_cell_count` / `edges_from_widths`.

### Meshes per physics (S7.2 / S7.5; model sub file "Per-physics meshes", "Triangle meshes"; GUI: gui sub file "Meshes")
- `ModelSpec.meshes` (named MeshSpecs, first = default) and `PhysicsSpec.mesh` ('' = meshes[0]): model code finds a
  physics' mesh by `spec.mesh_of(physics)` and passes mesh names to meshing; `spec.mesh` / `ModelSpec(mesh=)` are
  meshes[0] conveniences for scripts and self-tests only. Inside a physics / feature / coupling validate callback
  `vctx.mesh` is that physics' estimate (None on a mesh kind the type does not run on). A MeshSpec kind 'triangles'
  runs only types / features / solvers whose `grids` hold it; code reading tensor fields of an estimate (nx, x_edges,
  dx_cells) or of a Domain (`physical_window`) checks the kind first (a TriangleDomain has neither).
- Build (P2): one Domain + BuildContext per used mesh; creates / applies get `ctx.for_physics(name)` (that mesh),
  so `ctx.domain` is the physics' own mesh, never assume it is the Session's (`sim.domain` = the first mesh;
  `sim.domain_of(name)`). Cross-mesh couplings: `CouplingType.mesh_transfer` + `partner_materials` (coupling sub file
  "Couplings across meshes"); a transform must not reach into its partner beyond the view's state / domain / scalar
  attributes. Session (P3): fields, snapshots and feature-result callbacks of a physics act on its own mesh (a result
  callback never assumes the first mesh), derived values on the mesh of the first physics their expression names
  (model sub file "Per-mesh Session"). Everything a Session read does per grid kind lives in `session.access`
  (`model/field_access*.py`: one FieldAccess per kind; tensor windows split further by DomainMap), and everything a
  BuildContext mask / coordinate / boundary lookup does per kind in `ctx.masks` (`model/build_masks.py`): a new mesh
  kind or window map is a new subclass of each, never a `grid_kind` branch in Session or BuildContext.
- MeshSpec kind 'tetrahedra' (3D Cartesian only; model sub file "Tetrahedron meshes", fem sub file "Tetrahedra"):
  `schema.FREE_MESH_KINDS` = the element kinds (code testing "not tensor" must not test == 'triangles'); a
  `fem.domain3d.TetrahedronDomain` has per-element MaterialGrid and NO tensor accessors: 3D code branches on
  `domain.grid_kind` before physical_window / cell_centers / object_box_mask; MeshSpec.sweep_axis 'z' (swept prism
  layers, model sub file) is neither the Kuhn split nor gmsh 3D: code branching on `sizing.mesher` checks
  `sizing.sweep_axis` first; boundary selections are faces
  (`BuildContext.boundary_faces`), points are located (`ctx.point_position`), never nodes. em_fdfd_3d, heat,
  electric_currents, electrostatics (3D on tetrahedra only), magnetostatics_3d, solid_mechanics_3d and pressure_acoustics run on it.
- A physics on a triangle mesh (em_modes, em_fdfd_ez / _hz (point sources, probes, resonances), em_fdfd_bor
  (eigenfrequency, optomechanics optics) and em_fdfd_conical (eigenfrequency only), heat, electric_currents, electrostatics, solid_mechanics, magnetostatics,
  magnetic_fields_ac, drift_diffusion, membrane, pressure_acoustics) keeps every state() array and source slot per element ((n_elements,), D12:
  values at the element centroid / element means), `field_dofs(name)` their P0 DofMap, so state-dependent materials,
  same-mesh couplings and contributor buffers see per-element arrays; its create branches on `ctx.domain.grid_kind`
  (a type keeps its name, D4) and its renderer feeds the Session through `fem_field(component)`.

### 1D models (symmetry 2.20; model sub file "1D models")
- `DomainSpec.size_y` None = 1D (`ModelSpec.dimensions` 1; types need 1 in `PhysicsType.dimensions`: heat,
  electric_currents, electrostatics, membrane (its own line elements, fem.line_fem)): internally a padding-free 2D
  `Domain` one cell across on
  `Coordinates(line=True)`, so every 2D path runs unchanged. Code that must know branches on `spec.dimensions` /
  `DomainMap.line` / `Coordinates.line`, never on ny == 1. Outer faces x_lo / x_hi only (`spec.outer_faces`).
  Automatic solvers resolve per grid kind (`schema.solver_grid(spec)` 'line': solvers sub file "banded").

### 3D models (spec v2, B1; contracts in the core sub file)
- `DomainSpec.size_z` None = 2D (every 2D path unchanged), a value = 3D: `make_domain` returns `core.domain3d.Domain3D`
  (`dimensions` 3; `Domain.dimensions` 2), arrays `(nx, ny, nz)` C-order (z contiguous), every per-axis accessor
  returns 3-tuples. Code shared by 2D and 3D unpacks `[:2]` or branches on `domain.dimensions`, never on shapes.
- 'polar' + size_z = a cylindrical (r, phi, z) sector with logical (r, phi) cells: 3D code taking xy positions from
  `host_centres` / `rasterize_box` branches on `domain_map.mapped` (model sub file "Cylindrical (r, phi, z) sectors").
- z cells are graded like x / y on nonuniform / auto meshes: 3D code reads per-axis widths (`cell_widths()`,
  `grid.dz_arr`), never a scalar dz. Object shapes at a height only through `geometry` (prisms may be extruded along
  x / y or tapered: model sub file "3D prisms").
- `FACES` stays the 2D tuple (PML sides, BCs, templates); `FACES_3D` / `spec.OUTER_FACES_3D` add z_lo / z_hi, which
  are reserved names in every model but valid entities only in 3D (`spec.outer_faces(spec)`).
- `PhysicsType.dimensions` (default (2,)) decides where a type may run (validate); a type whose vocabulary changes in
  3D is a separate 3D-only type (em_fdtd_3d, em_fdfd_3d, em_eme_3d, solid_mechanics_3d, magnetostatics_3d), else one
  type with (2, 3) (heat; electric_currents / electrostatics, whose 3D runs only on tetrahedra: their validate refuses
  a 3D tensor mesh). In 3D, courant = fraction of the 3D CFL limit (`grid3d.courant_time_step_3d`); results stay
  2D cuts (`Session.field(..., plane=(axis, position))`; model sub file "3D field access").
- 3D builds never allocate full-grid float temporaries: materials are painted per z-slab run, full-grid bool / int16
  build masks only through BuildContext 3D helpers, `eval_field` of x, y gives an (nx, ny, 1) layer and of z a
  `BoxValues` on the mask's bounding box (core sub file "3D contracts"). z is a field coordinate only in 3D
  (`expressions.THREE_D_COORDINATE_NAMES`; reserved everywhere, a validation error in 2D).

### Solver contract
- Contract: `prepare / step / dt_limit / state / inject_source`, hooks `finalize / materials_changed`,
  `material_schema`, plus `renderer()` (not on the ABC, but called). `state()` keys are public; unknown slots raise.
  Quasi-static solvers (steady heat, potential, FDFD) have dt_limit inf and solve in `step(t, inf)` (FDFD and
  potential only when dirty); the study hooks `solve_harmonic(omega)` (FDFD, em_modes) and `solve_modes()` /
  `select_mode(k)` / `n_eff_text(k)` / `mode_index` (em_modes) are not on the ABC and are called only by the
  frequency_domain / mode_analysis study (em_modes `step` does nothing: the stationary study cannot drive it);
  likewise `solve_eigenfrequencies(n)` / `select_resonance(k)` (FDFD incl. em_fdfd_3d, solid_mechanics,
  solid_mechanics_3d (tensor meshes: lowest modes only), membrane; eigenfrequency and optomechanics (2D, 3D
  tetrahedra) studies;
  the returned solve is duck-typed: `summary()`, `info['warnings']`, resonances with `text()`; eigenfrequency search
  targets / band: optional `MultiShiftEigenSolver`, listed in `PhysicsType.eigen_searches`) and
  `attach_harmonic(frequencies)` / `select_harmonic_frequency(index)` (EMPhysics, frequency_domain only).
- Study hooks are the runtime-checkable Protocols of `model/solver_hooks.py` (studies test `isinstance(solver,
  ModeSolver)`, never `hasattr`; `StudySpec.solved_physics` restricts the solved ones to a list, the rest run as
  background physics: model sub file); `StudyType.solver_protocols` lists those a solver of a physics whose type lists the
  study satisfies (one of them), and the builder raises right after create when it satisfies none, so a solver
  variant missing a hook fails at build instead of running silently as background. A new study hook: add it to its
  Protocol (or a new Protocol there); a new solver variant of a type: it needs the hooks of every study the type
  lists, unless validate refuses that study for the variant.
- Scalar outputs (E5): `PhysicsSolver.scalars()` -> {name: float} (default {}) with the names declared by
  `PhysicsType.scalar_outputs(settings_values, physics_spec)` (CPU; physics_spec for names following features, e.g.
  drift_diffusion `current_<contact>`) (+ required `scalar_outputs_help`); derived values read them as
  `<physics>_<name>` only (`schema.scalar_output_names`; no plain alias), a declared name the solver did not report
  gives NaN + warning. Validate rejects a scalar named like a field of its type and a parameter named like one.
- Quasi-static solvers record probes with `replace_same_time=t > 0.0` (a re-solve at the same time replaces the sample).
- Nonlinear loop state that warm-starts later loops (saturation, k(T), ...) is cleared when the nonlinear set of
  cells / elements changes or empties. Fixed-point maps on conjugated fields are only real-linear: their mixing /
  least-squares / Krylov coefficients are real.
- scipy `csr + csr` drops entries that sum to exactly zero, so the pattern depends on the values: an operator whose
  factor / plan is refreshed in place is summed with `fem.assembly.pattern_sum` (or one COO list), never chained `+`;
  a shifted pencil K - sigma M likewise (`solvers.arpack_shift_invert.pattern_combination`: at sigma = 0 `K - 0 * M`
  would lose M's own entries).

### Physics definitions
- Polarisations are named by the out-of-plane field (`ez`, `hz`), never TE / TM. The type / coupling / study
  catalogue (default instances, registration order, which material Params types share, when a coupling registers) is
  in `api_reference.md` and the sub files: physics/<pkg> "type" bullets, coupling sub file "Registered coupling
  types", model sub file "Study catalogue". A material property shared by several types is ONE Param object (the
  first registered type's per name; e.g. rho: heat, laminar_flow, elastic_waves, solid_mechanics), so editing it
  changes every type. `PhysicsType.family` ('em', 'heat', '' none): coupling/definitions builds its CouplingTypes
  from the registered family-'em' types (heat_em_thermo_optic: all; em_heat_joule: those with an 'absorbed_power'
  output field, so not em_modes), so a new EM type needs family='em', `state()['q_accum']` (None allowed), the EM loss
  protocol when it absorbs, and thermo-optic source slots among eps_r_update / eps_xx_update / eps_yy_update (coupling
  sub file). `PhysicsType.source_slots` is a dict slot -> (unit, meaning) (describe prints both; consumers use it as
  the name sequence); EM types build theirs with `definition_common.source_slot_docs`.
- Source slot writers (E13): every CouplingType declares `writes(vctx, spec) -> {slot: '' | 'sum' | 'index'}`;
  model.validate allows one overwriting ('') writer per (target, slot) or several contributors of one kind and one
  every; the runner sums contributors through one SlotSum per slot ('index': eps = (n0 + sum dn)^2, n0 from
  physics/em/index_reference). A new coupling into an existing slot is a contributor (API: coupling sub file).
- `PhysicsType.group` (schema `PHYSICS_GROUPS`) and `spaces_2d` / `space_settings` (schema `SPACES`: planar,
  cross_section, axisymmetric = what the drawn plane is, plus '3d' from `dimensions`) are GUI only (describe ignores
  them, validate keeps its own rules; Add physics: gui sub file). A new type sets both; a new symmetry option that
  changes what the plane is adds a space instead of a free-standing setting.
- Axisymmetric (r, z) is the per-physics setting `geometry` (heat, electric_currents, electrostatics, magnetostatics,
  magnetic_fields_ac, drift_diffusion, elastic_waves (its space also sets bc_x_lo 'rigid', the axis rule),
  solid_mechanics (the solver holds u_r = 0 on the axis itself; whole-ring forces in N)),
  always through `definition_helpers`: `geometry_param` (shared help / tip around the type's sentences),
  `is_axisymmetric`, `AXISYMMETRIC_SPACE_SETTINGS` and `axisymmetric_issues` (x_lo axis features, <prefix>_xy
  materials, one warning naming the coupled partners that are planar = their type has no `geometry` setting or it is
  'planar'), so moving the setting to the domain (symmetry plan 2.12 / 2.16) touches one place. The solver exposes
  `axisymmetric` (Session ring-weights the derived values reading its fields; field-free expressions: those on its
  mesh). heat / electric_currents / electrostatics add `azimuthal_order` m (symmetry 2.14, `definition_helpers.AZIMUTHAL_ORDER_PARAM`; solid_mechanics too: triangles,
  eigenfrequency / optomechanics studies only): fields are cos(m phi) amplitudes
  and m != 0 rings carry pi, so code weighting (r, z) rings takes `core.metric.ring_angle` / the solver's `ring_angle`,
  never a literal 2 pi. em_fdfd_bor and em_fdtd_bor have the same
  `geometry` setting with the single choice 'axisymmetric' (so the partner rule counts them as axisymmetric) and
  always set `axisymmetric`.
- Help texts are API contract (describe -> api_reference.md): units, sign / axis conventions, typical values and
  gotchas, ASCII only (describe raises). Unit '1' = dimensionless (GUI hides it, also for integer counts / numbers),
  '' non-numeric; units with spaces ('W/(m K)'), never '*'. A unit that depends on the space (planar W/m, axisymmetric
  / 3D W) goes in `Param.space_units` (`schema.param_in_space`). A gotcha a user can hit belongs in a help text or
  validation message, not only here.
- Shared names (industry-standard if any): magnitudes `<name>_abs` (u_abs, e_abs, heat_flux_abs); resonance tables
  resonance_number / frequency / frequency_imag [Hz, > 0 decays] / q_factor (FDFD / BOR eigenfrequency and every
  resonance_monitor; + region_energy_fraction last with a region selection); mode results n_eff (real part) / n_eff_imag; 1-based mode choices `mode_number`; a body force
  feature `body_force` (params f_x / f_y, slots and coupled fields body_force_x / _y); probe choice `component`.
  Field names are scoped by physics: u is the velocity in laminar_flow, the displacement in solid_mechanics.
- Time t (simulated time, 0 outside a time_dependent study): state_dependent material properties and feature params
  flagged `Param.time_dependent` (parameters and t only) may use it; both are re-evaluated before every step at its
  start time (the runner clock moves before the couplings apply; implicit steps see a one-step lag). A feature param
  only gets the flag together with an incremental setter in its apply (`BuildContext.follow_time`; model sub file
  "Feature values of t"). Likewise `Param.follows_parameters` (SweepSpec.reuse_build) needs a
  `BuildContext.follow_parameters` setter on every grid kind whose solver is a RerunSolver, else a reused build
  silently keeps the build value; values are not range-checked per sweep value (model sub file "Reused sweep builds").
- GUI texts are separate: `Param.tip` (tooltip; '\n\n' between paragraphs, '- ' list lines), `Param.choice_labels`
  (drop-down texts; a `replace(param, choices=...)` that narrows the choices must narrow choice_labels too, else
  Param.__post_init__ raises) and `summary` on PhysicsType / FeatureType / CouplingType / StudyType / TemplateEntry /
  GeometryKind / Material (tree / menu tooltips, settings panel), `PhysicsType.mesh_method` (mesh form: how a type
  solves on each mesh kind + what that method cannot handle, mirroring validation; a type whose method changes with
  the mesh kind sets it); empty falls back to help / description. They are
  for a first-time user: no abbreviations, formulas or snake_case keys, refer to settings by their label; describe
  ignores them. Labels appear in both (they are in api_reference.md).
- After any change to the registry, a help text, a template or library data regenerate `api_reference.md` and
  `api_templates.md` (`python -m open_sol describe --update-reference` from the package's parent, writes both) or
  selftest `api_reference_current` fails
  (the documentation rules of describe: model sub file). The same command rewrites `structure_index.md`: after adding,
  moving or removing a module or changing the first sentence of a module docstring, else selftest
  `structure_index_current` fails. Every new module starts with a docstring whose first sentence says what it holds.
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
- Self-tests (E12): `physics/<name>/selftests*.py` and `coupling/selftests.py` hold SelfTestCases and template
  factories; `definition.register` registers types, templates, then its selftest loader
  (`registry.register_selftests(_selftest_cases)`: a module-level function that imports the selftests modules inside
  and returns their cases; `registry.selftests()` runs the loaders once, on first use), `coupling.definitions.register`
  its coupling types, then its loader. A new template still lives in a selftests module: add the (definition,
  selftests module) pair to `TEMPLATE_SOURCES`. api built-ins templates_validate, examples_validate (every examples/**/*.json of a
  repository checkout), api_reference_current, import_contracts and structure_index_current run first (a named run:
  only the built-ins it names); the
  registered cases (and the examples of an examples run) then run never-timed first, then by the cache's last
  wall time, ties in registration (example: path) order. A
  case with `specs()` {} runs `check({})` directly. `check` gets each run's live Results (closed afterwards), may use
  only `status`, `summary()`, `field(name[, plane=, unfold=True, native=True])`, `snapshot_field(index, name[, plane=])`,
  `volume(name)`, `result(physics, feature, **options)`, `sector_result(physics, feature[, combined=True])`,
  `sweep_field(index, name)`, and returns {'measured', 'expected', 'error', 'tolerance', 'details'} (passed when error
  <= tolerance). Backends of a case: `SelfTestCase.backends` (model sub file). A variant (another backend / solver
  precision with its own tolerance, within the rules above) is a `SelfTestCase.variants` entry (`SelfTestVariant`,
  solvers sub file), not a copied case. Case and template notes live in the sub files.
- `SelfTestCase.heavy`: set True on a case that took >= 5 s in a selftest run (record wall_time). CLI `selftest`
  runs the non-heavy cases, `--heavy` the heavy ones, `--all` everything (subsets and report files: model sub file).
- Selftest reuse (`model.selftest_cache`): with an out_dir, an unnamed case whose last run passed (an example: passed or
  no_tolerance; 'failed' / 'error' always rerun) is not rerun while the package code units it depended on (sys.settrace call events at
  function level; model sub file) and the environment are unchanged; CLI `--reset` (api `reset=True`) reruns all. A
  dependency the tracer cannot see (a non-.py data file read at run time, code run in a subprocess) needs `--reset`,
  or its reader adding the file to the traced set. Built-ins are never reused.
- Selftest reuse and shared code: `selftest_cache.UNTRACKED_PATHS` (the runner, model building / validation / specs /
  estimates, the study runner `model/study.py` + `study_adaptation.py` (the per-type study_*.py files with their own
  numerics stay tracked), `materials/__init__`, io, gui) is no dependency of any case and the names its code uses are not
  followed, so editing those files reruns nothing: run `selftest --reset` (e.g. overnight) after such edits. A new
  shared plumbing file belongs on that list. Tracked code every case runs (a solver, mesher or fem module used by
  all) still reruns everything when its module-level code changes; untracked code must not call into a
  model-specific module for every build (`materials._is_crystal_entry` checks the dict itself instead of running
  crystal_tensors code), or every case depends on that module. An in-process cache of expensive results (meshes,
  factors) kept across api runs hides code from the tracer (a case reusing another case's result never calls the
  code that made it): scope such caches to one `run_study`, as `meshing.element_meshes_kept` and
  `cudss_factor.cudss_orderings_kept` (cuDSS re-plans of a known pattern, also the remembered rejected mixed factors:
  solvers sub file "Ordering reuse").
- Self-test modules never rebind a module-level constant later in the file (later cases silently read the new value).
- Work counters: a new quasi-static solver adds `work_counters()` (not on the ABC; summary.json `solvers`, model sub
  file "Timing") with cumulative, machine-independent counts (iterations, factorisations) besides any seconds, on its
  role runtime when the solve is a solver role, else on the physics (never both).
- Solver roles (solver_plan Part I, stage A): a physics' numerical jobs are `PhysicsType.solver_roles`; the build
  creates their runtimes in a stage of its own and the physics create takes them from `ctx.solvers(name)`, never
  building one itself (scripts still may). Solver knobs are never physics settings (a new type declares a role, a
  shared one from definition_helpers when it fits); validate and estimators read the resolution
  (`ValidationContext.solvers`, `MemoryContext.solvers`), never a physics setting. Automatic = the role's default
  type at its defaults (a sparse_direct back_end 'auto' may resolve to cuDSS per role and grid on the gpu:
  `SolverRole.auto_back_ends`, solvers sub file; the optional host MUMPS / PARDISO back ends of `solvers.host_direct`
  are opt-in only, never auto: a role lists them in its `setting_choices`, every such tuple derived from
  `host_direct.HOST_DIRECT_*`, never spelled out); a model / template / self-test that needs other values (e.g. a CG tolerance) adds a solver
  node and assigns it (`PhysicsSpec.solvers`; self-tests: `selftest_helpers.with_exact_cg`). The model-wide
  `solver_defaults.precision` (S9) fills the precision of every solver that offers it and sets none: resolve solver
  values only through `schema.resolved_solvers` / `with_default_precision`, never `fill_defaults` of the node
  settings alone. Rules: solvers sub file.
- Subpixel smoothing (igor 1.5) and conformal boundaries (solver_plan S7.6) are model-wide
  (`solver_defaults.subpixel_smoothing` / `conformal_boundaries`), never physics settings: a type honours them by
  `PhysicsType.subpixel_smoothing` + `set_subpixel_smoothing(geometry)` / `PhysicsType.conformal_boundaries` +
  `set_conformal_boundaries(locator)` right after create (before its features apply); where they apply is
  `schema.subpixel_smoothing_physics` / `schema.conformal_boundaries_physics`. The averaged eps never enters the
  MaterialGrid: every face / cell eps consumer of an honouring type reads the averaged values (physics/em sub file
  "Subpixel smoothing"). Conformal: cell masks stay the cell-centre staircase (sources, ports, couplings, results read
  them); only the operator links at the wall change, each correction limited so the time step / conditioning is
  unchanged (the smallest cut never sets dt); honouring types and the locator: core sub file "Cut geometry".
- Estimates mirror the solver: a setting where 0 = auto is resolved through the solver's own resolver (e.g.
  `fluid.solver.steady_march_steps`), never used raw. Every memory budget check goes through `meshing.backend_memory`
  and takes its issue level from `schema.memory_check_level(spec)` (`solver_defaults.memory_check` 'warning' runs an
  over-budget model anyway; a refusal is final only on a built element mesh (its counts and `estimate.dissection`, the
  counted factor fill), so build / api.check mesh first when the estimate of an unbuilt triangle / tetrahedron mesh is
  over; model sub file "Memory estimate parts").
  A new physics / allocating feature MUST add its memory estimator (`PhysicsType.memory_estimate` /
  `FeatureType.memory_estimate`), counted from its own allocations (not the MaterialGrid arrays): the big arrays the
  solver holds, summed where they are held together, sizes derived from the code (stencil entry counts, dtypes),
  never fitted to measured bytes, so it sits just below the measured bytes (small arrays left out). An estimate
  above the measured bytes ('over') books what the solver does not hold; far below ('unaccounted') an array no item
  lists: a missing item or memory the code could free; both are leads on the solver (`builder.estimate_verdict`); a new allocation in
  an engine updates its estimator (else the build ledger warns), and anything allocated lazily after `finalize` is a
  run item, or a transient item of its phase when it is freed again (arrays of one solve / adjoint / refactorisation:
  phases never overlap, the estimate takes the largest; split a solve into phases in the order the code allocates, so
  no phase sums arrays that are never live together; build temporaries are the part's build_extra); a cuDSS factor
  (outside the cupy pool) is an external item from `cudss_factor.factor_device_bytes`, bound to the phases it lives in
  when it is released after its solve (`cudss_external(phases=)`); a MUMPS / PARDISO factor (host memory outside
  numpy) is a host item from `host_direct.factor_host_bytes` (or a SuperLU item x `superlu_bytes_scale`) plus
  `solver_host_bytes`, its constants uncalibrated. A new physics / solver type adds its run-time recipe (`PhysicsType.time_estimate` /
  `SolverType.time_estimate`: machine-independent op counts, never seconds; iteration counts derived a priori, never
  fitted to run counters: a count deviation is a lead on the solver; a kernel bound by arithmetic, not bytes, also
  counts its `Elementwise.flops`, since float64 runs at 1/64 rate on consumer GPUs: core sub file "Run-time
  estimate"). Model sub file "Memory estimate parts",
  "Run-time estimate rules".

### GPU memory and session lifetime
- GPU functions only on the GPU-owning thread (the worker, or the `api.run` caller); `gpu_memory.format_bytes` is the
  only one the GUI thread calls. cupy's pool runs under a budget (core sub file "GPU memory"): overruns raise
  OutOfMemoryError. `free_all_blocks` only between jobs / builds / sweep values, never in step loops.
- A cuDSS factor (and the cusolver_qr workspace) lives outside the cupy pool, so the budget never sees it: code holding
  one (a ModalPencil, a harmonic system's factor) releases it (`release()`) before dropping or replacing it. Likewise
  a MUMPS / PARDISO factor (`host_direct.HostDirectFactorization`) holds host memory outside numpy: release it or pass
  it as `previous=` (refactorised on the same pattern) before the next one; its own guard (the library's estimate
  after analysis vs HOST_FRACTION of the available RAM) raises. Their threads are the library defaults
  (OMP_NUM_THREADS / MKL_NUM_THREADS; no open_sol setting): oversubscription with numba / BLAS threads is the user's
  to set, except PARDISO beside an OpenBLAS numpy / scipy (MKL solves alternating with threaded BLAS calls run 4 - 25 x
  slower): a HostDirectFactorization holds the OpenBLAS pools at one thread during its own calls, and code looping
  around a PARDISO factor with BLAS work of its own (a Krylov / ARPACK loop) wraps the loop in
  `host_direct.quiet_openblas(name)` (solvers sub file "host_direct").
- A traceback keeps a failed build's frames and locals alive: inside `except` record only text, call
  `release_session` after the block; an exception kept or re-raised later goes through
  `gpu_memory.drop_exception_frames` first.
- Session close / release, OOM handling and the build ledger: model sub file ("Build memory", "Session lifetime").

### Performance
- No per-step host syncs or allocations in hot loops: preallocate, update in place, fused kernels.
- Host numpy over elements x points: never a broadcast `np.matmul` of 2x2 / 3x3 matrices per point or
  `np.linalg.det / inv` on 3x3 stacks (numpy loops per batch, no BLAS: write components out, e.g.
  `fem.mesh3d.span_adjugate`, or one GEMM over shared rows), never `np.add.at` (bincount: `fem.assembly.scatter_local`,
  `fem.fixed_pattern.scatter_add` (complex-safe) / `scatter_rows` (trailing axes)); mesh-constant integrals are cached
  per dof map, derived fields computed on read (fem sub file "Dofs and assembly").
- `gpu_memory.release_all` / cuFFT plan cache: core sub file "GPU memory" (an unconditional clear is a hidden full
  `gc.collect()`).
  `MaterialGrid.assign` uses `cp.copyto(where=)`; `FieldProbe` transfers once in `get_data()`. The accepted syncs are
  listed in the core sub file "Accepted host syncs" (a new one goes there).
- RawKernel scalars are numpy scalars of the declared C type (`DTYPE(...)`, `np.int32(...)` cached like
  `self._nx_i32`, `np.float64(...)`; `real_t` = the field dtype of a RealRawKernel); Python floats / ints silently
  corrupt them. Bool arrays: cupy bool, `const
  bool*`. A kernel a cpu-capable physics launches is a `core.kernels` twin (numpy twin of the same call, written in
  the same edit; core sub file "Kernel twins").
- Non-uniform spacing and CPML kappa are folded into the 1D `inv_dx/inv_dy` arrays: never branch kernels on
  uniformity.
- A `cp.ReductionKernel` reducing a whole array to one scalar runs on one thread block (RTX 3050: ~13 GB/s float32 /
  25 GB/s float64, vs ~200 GB/s for `cupy.cublas.dot(out=)` or a multi-block RawKernel; 1e6 float64: 636 vs 84 us):
  a full reduction in a hot loop uses a multi-block one (`solvers.pcg.device_vector_dot`), or reduces the shortest
  axis first (`core.kernels.axis_first_reduce`, for fused ReductionKernels of >= 2D arrays). cupy refuses cuSPARSE calls during stream capture
  (NotImplementedError), so a CUDA graph cannot contain cupyx SpMV: use the CSR RawKernels of `solvers.pcg`
  (`CSR_ROW_SOURCE`, `_DeviceCsrProduct`; the multigrid V-cycle is captured that way). Below ~1e6 unknowns a cupyx
  SpMV costs more host time (~60 us: descriptors, workspace, output allocation) than device time.
- A per-frequency (or per-iteration) cuDSS refactorisation of a fixed-pattern pencil goes through
  `sparse_direct.DevicePencil` / `factorize_pencil` (or `refactorize_device_values`), never a host A(omega) into
  `factorize_host`: the host canonicalising, asymmetry check and lower-triangle take cost more than the cuDSS
  factorisation itself (E7: ~250 of ~370 ms per frequency; solvers sub file "sparse_direct"). Since H1,
  factorize_host with previous= on the planned pattern refreshes on the device (values uploaded only), but it still
  pays the host assembly / canonical check that a DevicePencil saves.
- Point location dominates reads on element meshes (renderers, checks, derived values): locate through
  `TetrahedronDomain.locate` / `TriangleDomain.located` (remembers point sets and, on triangles, the Lagrange basis;
  read-only results), never a fresh locator per component or plane. A selftest whose time split shows a large 'other' is paying for its check's reads (model sub file).
- Allocate where needed (mandatory for new engines): pole arrays on their bounding box, optional accumulators once
  enabled, build temporaries only in BuildContext caches (`release_caches()`); status of the EM engines: physics/em
  sub file "Registered EM types".

### Generated output and git
- Every folder a tool writes results into (its default `--out` too) is named `out/` or ends in `_out/`: the tracked
  `.gitignore` ignores both, so a new tool needs no new rule. Committed example outputs (`examples/**/*_out/`) are
  re-included there; their `.npz` / `.png` stay ignored.
- The tracked `.gitignore` is public and generic; private folder names (experiments, plans, articles, side projects)
  go in `.git/info/exclude`, which every worktree of this clone shares and which is never pushed.

## 3. Sub files (`sub_structure_conventions.md`)

- `open_sol/`: api.run, Results lifetime (api.py, `__main__`), describe entries of new api / CLI arguments.
- `open_sol/mesh_study/`: import rule, parts, records and labels, variants, example inputs.
- `open_sol/fem/`: layout and imports, triangle mesh data (edge order / signs, curved edges, tags, PML layer ring),
  the built-in mesher (PSLG, refinement, smoothing, size field), gmsh, elements / geometry order, dofs and assembly
  (coefficient forms), TriangleDomain, point location, transfers, tetrahedra (ascending-node orientation, Nedelec 2
  face dofs, gmsh -3, entity majority, TetrahedronDomain, swept prism layers, sliver count, gmsh size fields and mesh
  caches), fixed-pattern Newton tangents (host / device), complex scalar harmonic systems / PML stretches / neutral
  resonance solve / several-shift search, squeeze film, reduced sweeps, line elements, prestress assemblers, self-tests.
- `open_sol/core/`: backends (`core.backend`, cpu / gpu), GPU memory (budget, memory_report), accepted host syncs,
  half storage, graded meshes (`core.mesh`, PML grading), grid transfers (`core.grid_transfer`), mirror planes
  (`core.mirror`), cut geometry (`core.cut_cells`, conformal honouring types), the coordinate metric (`core.metric`),
  3D contracts (Domain3D, BuildContext 3D), the resonance finder (`core.resonance`), run-time estimate ops, the
  damped-iteration API (`core.damped_iteration`), DiffusionSystem (`fv_diffusion`: assembly cache, c_xy term, PCG /
  direct, held cells, cut-cell held edges, axisymmetric and 3D operators; shared by heat and potential).
- `open_sol/solvers/`: layout and imports, the model path (spec, backends, resolved_solvers, validate, feature
  precisions, build stage, summary 'solvers'), pcg (stop reference, mixed, users), multigrid, line_block (layout,
  shift family, chunks, rejected shifts), sparse_direct (cuDSS plan reuse, host / device value refreshes), banded (grid
  kinds, 1D Automatic), arpack_shift_invert (several shifts, inertia counts), host_direct (MUMPS / PARDISO library
  notes), explicit_kernels.
- `open_sol/coupling/`: runner cadence, prepare / flush, coupling protocol, coupling type catalogue, q_accum readers,
  couplings across meshes (views, transfer classes), E13 contributor API / SlotSum, EM loss
  protocol, thermo-optic, Pockels, semiconductor / acoustics / piezo / fluid / magnetic / phase-change coupling
  modules, field_transfer, coupling self-tests.
- `open_sol/physics/em/`: EM type catalogue, imports inside physics/em, FDTD engines, sources, PML, walls, mirrors,
  Bloch, TFSF / NF2FF, monitors, slab modes, mode ports, nonlinear / gain / phase-change media, harmonic FDTD, FDFD
  (Bloch, ports, eigenfrequency, adjoint, design regions, conducting sheets), cross-section modes, em_modes,
  body-of-revolution FDFD, subpixel smoothing, EM definitions and self-tests, effective index.
- `open_sol/physics/eme/`: sections, bilinear mode sets (2D slab / 3D cross-section, PML modes last), interface formula,
  discrete variant (exact ez and hz), cascade / partial cascades, periodic block, bends, absorbed power, mode tracking,
  EMEBase / EMEPhysics / EMEPhysics3D, cascade adjoint, EME self-tests.
- `open_sol/physics/nlfd/`: harmonic channels, ranks and the mixing table, C / C* collocation, channel features,
  NonlinearHarmonicPhysics, nonlinear sheets (site sampling / adjoint deposit).
- `open_sol/physics/em3d/`: 3D Yee layout, material index, PEC / walls, Bloch, mirrors, CPML slabs, poles, losses,
  stepper precision, subpixel smoothing, flux planes, harmonic FDTD, source protocol (inject_magnetic hook), TFSF
  plane wave, A5 mode ports, 3D near-to-far, em_fdfd_3d, the tetrahedral FEMPhysics3D (PML shell, walls, equilibration)
  and em_fdfd_3d on 'tetrahedra' meshes.
- `open_sol/physics/heat/`: HeatPhysics, thermal noise (thermorefractive / particle), heat definition and self-tests
  (+ the DiffusionSystem notes of core).
- `open_sol/physics/potential/`: PotentialPhysics, axisymmetric / polar, electrolyte (Poisson-Boltzmann), driven
  terminals, self-tests, triangle and tetrahedron meshes (+ the DiffusionSystem notes of core).
- `open_sol/physics/mechanics/`: 2D / 3D split (SolidMechanicsBase), type catalogue, SolidMechanicsPhysics /
  ElasticSystem (Q1 layout, supports, rigid-mode check, loads), anisotropic c_ij stiffness, eigenfrequency study (2D,
  3D), piezoelectric plane strain (coupled host operator, electrodes, short / open modes), frequency response (losses,
  admittance, BVD fit, reduced / modal sweeps), optomechanical g0, ElasticSystem3D
  (matrix-free bricks, reduced blocks, Gram rank check, 2D rules that do not apply), tetrahedra (modal pencils and
  target / band searches, PML, Bloch, squeeze film), geometric nonlinearity (warm start, tangent updates, GPU tangents),
  thermal strain coupling, optics hooks, self-tests.
- `open_sol/physics/membrane/`: layout, materials / holes / coupled tension and the plate switch, lazy system and PML
  reference, eigen and driven solves (reduced sweeps, target / band searches), Hermite slope scaling, plate (interior
  penalty, plate PML, plate ports), Bloch cells, ports / mode projections, parity sectors, squeeze film, faces, fields,
  self-tests.
- `open_sol/physics/acoustics/`: AcousticPhysics (staggering, slip walls, CPML, coefficients, monoclinic c16 / c26,
  time step, axisymmetric, stepper precision), piezoelectric, optics couplings, self-tests, pressure_acoustics
  (frequency domain).
- `open_sol/physics/fluid/`: FluidPhysics (lattice units, faces, half_storage, Bouzidi walls, steady march),
  convection coupling, fluid self-tests.
- `open_sol/physics/magnetic/`: psi formulation on the planar operator, axis rule, coefficient swap, external-field
  ring, remanence face sources, saturation / B-H tables, triangle meshes (A_phi in B-space, one quadrature for W / R /
  K_t, Newton with the energy line search), the AC host solve and circuit rows, couplings, magnetic self-tests,
  magnetostatics_3d (scalar potential, swapped face kinds, corner anchor, ring, 3D face drives, coil T and the
  energy / linkage identity, the isotropic H-based saturation loop).
- `open_sol/physics/semiconductor/`: DriftDiffusionSystem (scaling, SG faces, pinned rows, Newton / continuation),
  contacts / gates, recombination, self-tests.
- `open_sol/materials/`: library data conventions, crystal entries.
- `open_sol/model/`: expressions, analytic comparison / derived curves, schema (table properties), spec (3D, prisms,
  composites), polar / spherical domains (DomainMap), meshing (builtin tetrahedra, swept prism layers, mesh refinement
  levels), memory estimate (parts, budgets), builder (conformal, subpixel, build memory / ledger, build masks / field
  access), materials, session (3D field access, requires_study fields, lifetime, OOM), validation, symmetry check,
  worker, progress stages, studies (catalogue, solution feedback, parity sectors, reused sweep builds, parameter
  estimation), resonance_monitor row, run-time estimate rules (build estimate of element meshes), selftest cache /
  subsets / reports / backends, templates, describe.
- `open_sol/io/`: model / results files, curve exports, result kinds.
- `open_sol/gui/`: GUI (thread rule, result plot views, progress, documents and node paths, canvas, 3D views, forms).
- `open_sol/viz/`: Visualizer, common, result_plots, curve_figure, scene_figure.
- `experiments/`: paper experiments (`<paper><year>/`, run_all), link to `build_experiments_guide.md` (layout, time
  estimates, cuDSS, model gotchas).
- `dev_tools/time_profilers/`: solver_profiler (parent / child layout, imports, solver enumeration rule, result
  line), case_profilers (script layout, wrap before the registry, unresolved parts); usage in its README.md.
- `dev_tools/memory_profilers/`: vram_profile (usage, what it measures, the private names it patches).
